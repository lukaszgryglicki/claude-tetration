//! Arbitrary-precision complex number primitives: precision conversion, parsing,
//! formatting, and elementary operations (`pow`, `log_b`) used across algorithms.

use std::sync::atomic::{AtomicBool, Ordering};

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
pub fn digits_to_bits(digits: u64) -> u32 {
    checked_digits_to_bits(digits).expect("invalid decimal precision")
}

pub fn checked_digits_to_bits(digits: u64) -> Result<u32, String> {
    const MAX_DIGITS: u64 = 1_000_000_000;
    if digits == 0 || digits > MAX_DIGITS {
        return Err(format!(
            "precision must be positive and at most the maximum {MAX_DIGITS} digits"
        ));
    }
    // 3.32193 is an upper bound on log2(10), not a floating-point estimate.
    let main = (digits * 332_193).div_ceil(100_000);
    let guard: u64 = std::cmp::max(64, digits / 10);
    let bits = u32::try_from(main + guard)
        .map_err(|_| "requested precision exceeds the supported bit count")?;
    check_precision_range(bits)?;
    Ok(bits)
}

/// Retain decimal significand information before exact domain/parity decisions.
/// Numeric syntax is still validated separately by the parser.
pub fn checked_input_precision(digits: u64, inputs: &[&str]) -> Result<u32, String> {
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

fn check_precision_range(prec: u32) -> Result<(), String> {
    let span = i64::from(prec);
    if -span < i64::from(rug::float::exp_min()) || span > i64::from(rug::float::exp_max()) {
        return Err(format!(
            "working precision {prec} bits exceeds MPFR's exponent range for nonzero tolerances"
        ));
    }
    Ok(())
}

pub fn require_precision(prec: u32, digits: u64) -> Result<(), String> {
    let required = checked_digits_to_bits(digits)?;
    if prec < required {
        return Err(format!(
            "requested {digits} digits require at least {required} working bits; received {prec}"
        ));
    }
    check_precision_range(prec)
}

/// Parse a real decimal string into an arbitrary-precision `Float`.
pub fn parse_float(s: &str, prec: u32) -> Result<Float, String> {
    let parsed = Float::parse(s).map_err(|e| format!("invalid number {:?}: {}", s, e))?;
    let value = Float::with_val(prec, parsed);
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

pub fn abs(z: &Complex, prec: u32) -> Float {
    Float::with_val(prec, z.abs_ref())
}

pub fn decimal(s: &str, prec: u32) -> Float {
    Float::with_val(prec, Float::parse(s).expect("invalid decimal constant"))
}

pub fn env_float(name: &str, default: &str, prec: u32) -> Result<Float, String> {
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

pub fn epsilon(digits: u64, prec: u32) -> Float {
    Float::with_val(prec, 10).pow(digits).recip()
}

pub fn working_epsilon(prec: u32) -> Float {
    Float::with_val(prec, 1) >> prec.saturating_sub(32)
}

pub fn eta_lower(prec: u32) -> Float {
    (-Float::with_val(prec, 1).exp()).exp()
}

pub fn eta_upper(prec: u32) -> Float {
    Float::with_val(prec, 1).exp().recip().exp()
}

pub fn checked_exp(z: &Complex, prec: u32) -> Result<Complex, String> {
    if !is_finite(z) {
        return Err("exponential argument is non-finite".into());
    }
    let value = Complex::with_val(prec, z.exp_ref());
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
pub fn parse_complex(re: &str, im: &str, prec: u32) -> Result<Complex, String> {
    let r = parse_float(re, prec)?;
    let i = parse_float(im, prec)?;
    Ok(Complex::with_val(prec, (r, i)))
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
    f.to_string_radix(10, Some(digits.max(1)))
}

/// Complex exponentiation `b^e = exp(e * ln(b))`.
///
/// Uses the principal branch via MPC's `ln`/`exp`, with intermediate rounding.
pub fn pow_complex(b: &Complex, e: &Complex, prec: u32) -> Complex {
    let ln_b = Complex::with_val(prec, b.ln_ref());
    let prod = Complex::with_val(prec, &ln_b * e);
    Complex::with_val(prec, prod.exp_ref())
}

/// Complex logarithm in arbitrary base: `log_b(z) = ln(z) / ln(b)`. Principal branch.
pub fn log_b_complex(z: &Complex, b: &Complex, prec: u32) -> Complex {
    let ln_z = Complex::with_val(prec, z.ln_ref());
    let ln_b = Complex::with_val(prec, b.ln_ref());
    Complex::with_val(prec, &ln_z / &ln_b)
}

/// Returns true iff `b` is exactly the real number 1.
pub fn is_one(b: &Complex) -> bool {
    if !b.imag().is_zero() {
        return false;
    }
    let one = Float::with_val(b.real().prec(), 1);
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
    if !re.is_integer() {
        return None;
    }
    let i: Integer = re.to_integer()?;
    i.to_i64()
}

/// Constant `1` as a `Complex` at the given precision.
pub fn one(prec: u32) -> Complex {
    Complex::with_val(prec, (1, 0))
}

/// Constant `0` as a `Complex` at the given precision.
pub fn zero(prec: u32) -> Complex {
    Complex::with_val(prec, (0, 0))
}
