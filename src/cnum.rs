//! Arbitrary-precision complex number primitives: precision conversion, parsing,
//! formatting, and elementary operations (`pow`, `log_b`) used across algorithms.

use std::{
    ffi::CStr,
    fmt,
    sync::atomic::{AtomicBool, Ordering},
};

use gmp_mpfr_sys::mpfr;
use rug::{ops::Pow, Complex, Float, Integer};

static QUIET: AtomicBool = AtomicBool::new(false);

pub fn set_quiet(quiet: bool) {
    QUIET.store(quiet, Ordering::Relaxed);
}

/// Verbosity gate. Diagnostics print to stderr by default; set `SILENT=1` (or
/// `t`/`true`/`y`/`yes`) to suppress all per-iteration output and produce only
/// the final result on stdout. Designed for the CLI tool: under-the-hood
/// algorithm probing during long runs is the common case, and the silent
/// switch is for the rare case where stderr noise interferes with scripting.
pub fn verbose() -> bool {
    !QUIET.load(Ordering::Relaxed) && !silent()
}

fn silent() -> bool {
    match std::env::var("SILENT") {
        Ok(v) => is_truthy(&v),
        Err(_) => false,
    }
}

fn is_truthy(s: &str) -> bool {
    matches!(
        s.trim().to_ascii_lowercase().as_str(),
        "1" | "t" | "true" | "y" | "yes" | "on"
    )
}

/// Convert decimal-digit precision to MPC precision bits, including a guard band
/// for iterative algorithms; the guard is not a proved error bound.
///
/// The guard is `max(64, digits/10)`.
pub fn digits_to_bits(digits: u64) -> u64 {
    checked_digits_to_bits(digits).expect("invalid decimal precision")
}

pub fn checked_digits_to_bits(digits: u64) -> Result<u64, String> {
    init_mpfr();
    if digits == 0 {
        return Err("precision must be positive".into());
    }
    // 3.32193 is an upper bound on log2(10), not a floating-point estimate.
    let main = (u128::from(digits) * 332_193).div_ceil(100_000);
    let guard = u128::from(digits / 10).max(64);
    let bits = u64::try_from(main + guard)
        .map_err(|_| "requested precision exceeds MPFR's native bit count")?;
    check_precision(bits)?;
    Ok(bits)
}

/// Retain decimal significand information before exact domain/parity decisions.
/// Numeric syntax is still validated separately by the parser.
pub fn checked_input_precision(digits: u64, inputs: &[&str]) -> Result<u64, String> {
    let mut prec = checked_digits_to_bits(digits)?;
    for input in inputs {
        let significand = input.split(['e', 'E', '@']).next().unwrap_or(input);
        let significant = significand.trim_matches(|c: char| !matches!(c, '1'..='9'));
        let input_digits = u64::try_from(significant.bytes().filter(u8::is_ascii_digit).count())
            .map_err(|_| "input precision exceeds the supported digit count")?;
        if input_digits > digits {
            prec = prec.max(checked_digits_to_bits(input_digits)?);
        }
    }
    Ok(prec)
}

pub fn init_mpfr() {
    // MPFR's exponent range is thread-local; widening preserves existing values.
    unsafe {
        let min = mpfr::get_emin_min();
        let max = mpfr::get_emax_max();
        if mpfr::get_emin() != min {
            assert_eq!(mpfr::set_emin(min), 0, "MPFR rejected its minimum exponent");
        }
        if mpfr::get_emax() != max {
            assert_eq!(mpfr::set_emax(max), 0, "MPFR rejected its maximum exponent");
        }
    }
}

pub fn exponent_range() -> (mpfr::exp_t, mpfr::exp_t) {
    init_mpfr();
    unsafe { (mpfr::get_emin(), mpfr::get_emax()) }
}

pub fn check_precision(prec: u64) -> Result<(), String> {
    let max = rug::float::prec_max_64();
    if prec < u64::from(rug::float::prec_min()) || prec > max {
        return Err(format!(
            "working precision {prec} bits is outside MPFR's native precision range (maximum {max})"
        ));
    }
    Ok(())
}

pub fn checked_usize(value: &Float) -> Option<usize> {
    if !value.is_finite() || !value.is_integer() || *value < 0 || *value > usize::MAX {
        return None;
    }
    value.to_integer()?.to_usize()
}

pub fn check_complex_storage(count: u128, prec: u64) -> Result<(), String> {
    check_storage(count, prec, 2, std::mem::size_of::<Complex>())
}

pub fn check_float_storage(count: u128, prec: u64) -> Result<(), String> {
    check_storage(count, prec, 1, std::mem::size_of::<Float>())
}

fn check_storage(count: u128, prec: u64, components: u128, header: usize) -> Result<(), String> {
    check_precision(prec)?;
    let limb_bits = std::mem::size_of::<gmp_mpfr_sys::gmp::limb_t>() * 8;
    let float_bytes = u128::from(prec).div_ceil(limb_bits as u128) * (limb_bits / 8) as u128;
    let bytes = count
        .checked_mul(header as u128 + components * float_bytes)
        .ok_or("working storage exceeds addressable memory")?;
    if bytes > usize::MAX as u128 {
        return Err(format!(
            "{count} working values at {prec} bits require at least {bytes} bytes, exceeding addressable memory"
        ));
    }
    Ok(())
}

pub fn require_precision(prec: u64, digits: u64) -> Result<(), String> {
    let required = checked_digits_to_bits(digits)?;
    if prec < required {
        return Err(format!(
            "requested {digits} digits require at least {required} working bits; received {prec}"
        ));
    }
    check_precision(prec)
}

/// Parse a real decimal string into an arbitrary-precision `Float`.
pub fn parse_float(s: &str, prec: u64) -> Result<Float, String> {
    init_mpfr();
    check_precision(prec)?;
    let parsed = Float::parse(s).map_err(|e| format!("invalid number {:?}: {}", s, e))?;
    let value = Float::with_val_64(prec, parsed);
    if !value.is_finite() {
        return Err(format!(
            "number {:?} must be finite and within MPFR's exponent range",
            s
        ));
    }
    if value.is_zero() {
        let significand = s.split(['e', 'E', '@']).next().unwrap_or(s);
        if significand.bytes().any(|c| matches!(c, b'1'..=b'9')) {
            return Err(format!("number {:?} underflows MPFR's exponent range", s));
        }
    }
    Ok(value)
}

pub fn is_finite(z: &Complex) -> bool {
    z.real().is_finite() && z.imag().is_finite()
}

pub fn abs(z: &Complex, prec: u64) -> Float {
    init_mpfr();
    Float::with_val_64(prec, z.abs_ref())
}

pub fn decimal(s: &str, prec: u64) -> Float {
    init_mpfr();
    Float::with_val_64(prec, Float::parse(s).expect("invalid decimal constant"))
}

pub fn env_float(name: &str, default: &str, prec: u64) -> Result<Float, String> {
    match std::env::var(name) {
        Ok(value) => parse_float(&value, prec).map_err(|e| format!("{name}: {e}")),
        Err(std::env::VarError::NotPresent) => Ok(decimal(default, prec)),
        Err(e) => Err(format!("{name}: {e}")),
    }
}

pub fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    match std::env::var(name) {
        Ok(value) => value.parse().map_err(|e| format!("{name}: {e}")),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(e) => Err(format!("{name}: {e}")),
    }
}

pub fn epsilon(digits: u64, prec: u64) -> Float {
    init_mpfr();
    let exponent = -Integer::from(digits);
    let value = Float::with_val_64(prec, 10).pow(&exponent);
    assert!(
        !value.is_zero(),
        "decimal tolerance underflows MPFR's exponent range"
    );
    value
}

pub fn working_epsilon(prec: u64) -> Float {
    init_mpfr();
    let shift =
        usize::try_from(prec.saturating_sub(32)).expect("precision exceeds addressable bits");
    let value = Float::with_val_64(prec, 1) >> shift;
    assert!(
        !value.is_zero(),
        "binary tolerance underflows MPFR's exponent range"
    );
    value
}

pub fn eta_lower(prec: u64) -> Float {
    init_mpfr();
    (-Float::with_val_64(prec, 1).exp()).exp()
}

pub fn eta_upper(prec: u64) -> Float {
    init_mpfr();
    Float::with_val_64(prec, 1).exp().recip().exp()
}

pub fn checked_exp(z: &Complex, prec: u64) -> Result<Complex, String> {
    let (min, max) = exponent_range();
    if !is_finite(z) {
        return Err("exponential argument is non-finite".into());
    }
    // Avoid MPC's huge-angle reduction when the real part already proves range failure.
    if *z.real() > max {
        return Err("exponential overflow: result exceeds MPFR's exponent range".into());
    }
    if *z.real() < min {
        return Err("exponential underflow: a finite exponential cannot be zero".into());
    }
    let value = Complex::with_val_64(prec, z.exp_ref());
    if !is_finite(&value) {
        return Err("exponential overflow: result exceeds MPFR's exponent range".into());
    }
    if is_zero(&value) {
        return Err("exponential underflow: a finite exponential cannot be zero".into());
    }
    // For a finite binary y, cos(y) cannot vanish, and sin(y)=0 only at y=0.
    if value.real().is_zero() || (!z.imag().is_zero() && value.imag().is_zero()) {
        return Err(
            "exponential underflow: a nonzero component exceeds MPFR's exponent range".into(),
        );
    }
    Ok(value)
}

/// Parse a complex number from two decimal strings (real, imaginary).
pub fn parse_complex(re: &str, im: &str, prec: u64) -> Result<Complex, String> {
    let r = parse_float(re, prec)?;
    let i = parse_float(im, prec)?;
    Ok(Complex::with_val_64(prec, (r, i)))
}

/// Format a complex number to (real_str, imag_str) using `digits` significant
/// decimal digits each.
pub fn format_complex(z: &Complex, digits: usize) -> (String, String) {
    (
        format_float(z.real(), digits),
        format_float(z.imag(), digits),
    )
}

/// Format a `Float` to a decimal string with `digits` significant digits.
/// Special-cases NaN / inf / zero so output is parseable and stable.
pub fn format_float(f: &Float, digits: usize) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    decimal_string(f, digits.max(1), false, false)
}

pub fn format_float_roundtrip(f: &Float) -> String {
    decimal_string(f, 0, false, false)
}

fn decimal_string(f: &Float, digits: usize, scientific: bool, upper: bool) -> String {
    init_mpfr();
    if f.is_nan() {
        return if f.is_sign_negative() { "-NaN" } else { "NaN" }.into();
    }
    if f.is_infinite() {
        return if f.is_sign_negative() {
            "-inf".into()
        } else {
            "inf".into()
        };
    }
    if f.is_zero() {
        return if f.is_sign_negative() {
            "-0".into()
        } else {
            "0".into()
        };
    }
    let mut exponent = 0;
    // Rug 1.30's formatter reserves space proportional to the exponent itself.
    // MPFR allocates only the significand here; its exponent is separate.
    let mut text = unsafe {
        let raw = mpfr::get_str(
            std::ptr::null_mut(),
            &mut exponent,
            10,
            digits,
            f.as_raw(),
            mpfr::rnd_t::RNDN,
        );
        assert!(
            !raw.is_null(),
            "MPFR could not allocate a decimal significand"
        );
        let text = CStr::from_ptr(raw)
            .to_str()
            .expect("non-ASCII MPFR decimal")
            .to_owned();
        mpfr::free_str(raw);
        text
    };
    let sign = usize::from(text.starts_with('-'));
    let count = text.len() - sign;
    let point = if scientific || exponent <= 0 || u128::try_from(exponent).unwrap() > count as u128
    {
        exponent -= 1;
        1
    } else {
        let point = usize::try_from(exponent).expect("decimal point exceeds significand");
        exponent = 0;
        point
    };
    if point < count {
        text.insert(sign + point, '.');
    }
    if scientific || exponent != 0 {
        use std::fmt::Write;
        write!(&mut text, "{}{exponent}", if upper { 'E' } else { 'e' }).unwrap();
    }
    text
}

pub struct DisplayFloat<'a>(pub &'a Float);

impl DisplayFloat<'_> {
    fn format(&self, f: &mut fmt::Formatter<'_>, scientific: bool, upper: bool) -> fmt::Result {
        let text = decimal_string(self.0, f.precision().unwrap_or(0), scientific, upper);
        let (positive, magnitude) = match text.strip_prefix('-') {
            Some(magnitude) => (false, magnitude),
            None => (true, text.as_str()),
        };
        f.pad_integral(positive, "", magnitude)
    }
}

impl fmt::Display for DisplayFloat<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.format(f, false, false)
    }
}

impl fmt::LowerExp for DisplayFloat<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.format(f, true, false)
    }
}

impl fmt::UpperExp for DisplayFloat<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.format(f, true, true)
    }
}

pub struct DisplayComplex<'a>(pub &'a Complex);

impl fmt::Display for DisplayComplex<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let digits = f.precision().unwrap_or(0);
        write!(
            f,
            "({} {})",
            decimal_string(self.0.real(), digits, false, false),
            decimal_string(self.0.imag(), digits, false, false)
        )
    }
}

/// Complex exponentiation `b^e = exp(e * ln(b))`.
///
/// Uses the principal branch via MPC's `ln`/`exp`, with intermediate rounding.
pub fn pow_complex(b: &Complex, e: &Complex, prec: u64) -> Complex {
    init_mpfr();
    let ln_b = Complex::with_val_64(prec, b.ln_ref());
    let prod = Complex::with_val_64(prec, &ln_b * e);
    Complex::with_val_64(prec, prod.exp_ref())
}

/// Complex logarithm in arbitrary base: `log_b(z) = ln(z) / ln(b)`. Principal branch.
pub fn log_b_complex(z: &Complex, b: &Complex, prec: u64) -> Complex {
    init_mpfr();
    let ln_z = Complex::with_val_64(prec, z.ln_ref());
    let ln_b = Complex::with_val_64(prec, b.ln_ref());
    Complex::with_val_64(prec, &ln_z / &ln_b)
}

/// Returns true iff `b` is exactly the real number 1.
pub fn is_one(b: &Complex) -> bool {
    if !b.imag().is_zero() {
        return false;
    }
    let one = Float::with_val_64(b.real().prec_64(), 1);
    *b.real() == one
}

/// Returns true iff `b` is exactly 0 (both real and imaginary parts).
pub fn is_zero(b: &Complex) -> bool {
    b.real().is_zero() && b.imag().is_zero()
}

/// If `h` is a real integer (zero imaginary part, no fractional part) that fits
/// in `i64`, return it. Otherwise return `None`.
pub fn as_integer(h: &Complex) -> Option<i64> {
    if !h.imag().is_zero() {
        return None;
    }
    let re = h.real();
    if !re.is_finite() {
        return None;
    }
    if !re.is_integer() || *re < i64::MIN || *re > i64::MAX {
        return None;
    }
    let i: Integer = re.to_integer()?;
    i.to_i64()
}

/// Constant `1` as a `Complex` at the given precision.
pub fn one(prec: u64) -> Complex {
    init_mpfr();
    Complex::with_val_64(prec, (1, 0))
}

/// Constant `0` as a `Complex` at the given precision.
pub fn zero(prec: u64) -> Complex {
    init_mpfr();
    Complex::with_val_64(prec, (0, 0))
}
