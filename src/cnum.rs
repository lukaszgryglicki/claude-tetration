//! Arbitrary-precision complex number primitives: precision conversion, parsing,
//! formatting, and elementary operations (`pow`, `log_b`) used across algorithms.

use rug::{Complex, Float, Integer};

/// Convert decimal-digit precision to MPC precision bits, including a guard band
/// that absorbs cumulative rounding loss in iterative algorithms.
///
/// The guard is `max(64, digits/10)` — generous enough for the long Schröder /
/// Cauchy series used in later phases.
pub fn digits_to_bits(digits: u64) -> u32 {
    // log2(10) ≈ 3.321928094887362; multiply and round up.
    let main: u64 = ((digits as f64) * std::f64::consts::LOG2_10).ceil() as u64;
    let guard: u64 = std::cmp::max(64, digits / 10);
    let total = main.saturating_add(guard);
    if total > (u32::MAX as u64) {
        // MPC's prec_t is at least 32-bit. Anything past 4 billion bits is
        // beyond what MPFR can represent in any case.
        panic!("requested precision {} digits exceeds MPC's representable range", digits);
    }
    total as u32
}

/// Parse a real decimal string into an arbitrary-precision `Float`.
pub fn parse_float(s: &str, prec: u32) -> Result<Float, String> {
    let parsed = Float::parse(s).map_err(|e| format!("invalid number {:?}: {}", s, e))?;
    Ok(Float::with_val(prec, parsed))
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
    (format_float(z.real(), digits), format_float(z.imag(), digits))
}

/// Format a `Float` to a decimal string with `digits` significant digits.
/// Special-cases NaN / inf / zero so output is parseable and stable.
pub fn format_float(f: &Float, digits: usize) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f.is_sign_negative() { "-inf".into() } else { "inf".into() };
    }
    if f.is_zero() {
        return if f.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    f.to_string_radix(10, Some(digits.max(1)))
}

/// Complex exponentiation `b^e = exp(e * ln(b))`.
///
/// Uses the principal branch via MPC's `ln`/`exp`. For real-positive b this
/// agrees with the real `pow` to the last bit.
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
