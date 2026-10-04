//! Direct iteration for integer heights.
//!
//! For non-negative `n`: `F_b(0) = 1`, `F_b(n+1) = b^F_b(n)`.
//! For `n = -1`: `F_b(-1) = 0` by the principal backward-iteration convention.
//! For `n ≤ -2`: undefined for general `b` (would require `log_b(0)` and beyond);
//! degenerate bases have the separate conventions below.

use rug::Complex;

use crate::cnum;

/// Resource limit for direct iteration; individual steps can also exceed
/// MPFR's exponent range.
pub const MAX_INTEGER_HEIGHT: i64 = 100_000;

/// Compute `F_b(n)` by direct iteration. Nondegenerate bases refuse `n ≤ -2`
/// and positive heights above `MAX_INTEGER_HEIGHT`.
pub fn tetrate_integer(b: &Complex, n: i64, prec: u32) -> Result<Complex, String> {
    if !cnum::is_finite(b) {
        return Err("tetration base must be finite".into());
    }
    if cnum::is_one(b) {
        return Ok(cnum::one(prec));
    }
    if cnum::is_zero(b) {
        return if n < 0 {
            Err("base zero is defined only for non-negative integer heights".into())
        } else if n % 2 == 0 {
            Ok(cnum::one(prec))
        } else {
            Ok(cnum::zero(prec))
        };
    }
    if n == 0 {
        return Ok(cnum::one(prec));
    }
    if n == -1 {
        return Ok(cnum::zero(prec));
    }
    if n < -1 {
        return Err(format!(
            "integer height {} is undefined for tetration (would require log_b(0) and beyond)",
            n
        ));
    }
    if n > MAX_INTEGER_HEIGHT {
        return Err(format!(
            "integer height {} exceeds MAX_INTEGER_HEIGHT={}",
            n, MAX_INTEGER_HEIGHT
        ));
    }
    if b.imag().is_zero() && *b.real() == -1 {
        return Ok(Complex::with_val(prec, b));
    }
    let ln_b = Complex::with_val(prec, b.ln_ref());
    let mut acc = Complex::with_val(prec, b);
    for level in 2..=n {
        let argument = Complex::with_val(prec, &ln_b * &acc);
        acc = cnum::checked_exp(&argument, prec)
            .map_err(|e| format!("integer tower at height {}: {}", level, e))?;
    }
    Ok(acc)
}
