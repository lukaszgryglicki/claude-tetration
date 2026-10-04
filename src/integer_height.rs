//! Direct iteration for integer heights.
//!
//! For non-negative `n`: `F_b(0) = 1`, `F_b(n+1) = b^F_b(n)`.
//! For `n = -1`: `F_b(-1) = 0` by the principal backward-iteration convention.
//! For `n ≤ -2`: undefined for general `b` (would require `log_b(0)` and beyond);
//! degenerate bases have the separate conventions below.

use rug::{Complex, Float, Integer};

use crate::cnum;

/// Compute `F_b(n)` by direct iteration. Nondegenerate bases refuse `n ≤ -2`.
pub fn tetrate_integer(b: &Complex, n: i64, prec: u64) -> Result<Complex, String> {
    cnum::check_precision(prec)?;
    tetrate_integer_height(b, &Float::with_val_64(prec.max(64), n), prec)
}

pub(crate) fn tetrate_integer_height(b: &Complex, n: &Float, prec: u64) -> Result<Complex, String> {
    cnum::init_mpfr();
    cnum::check_precision(prec)?;
    if !cnum::is_finite(b) || !n.is_finite() || !n.is_integer() {
        return Err("integer tetration requires a finite base and integer height".into());
    }
    if cnum::is_one(b) {
        return Ok(cnum::one(prec));
    }
    if cnum::is_zero(b) {
        return if *n < 0 {
            Err("base zero is defined only for non-negative integer heights".into())
        } else if (n.clone() >> 1u32).is_integer() {
            Ok(cnum::one(prec))
        } else {
            Ok(cnum::zero(prec))
        };
    }
    if n.is_zero() {
        return Ok(cnum::one(prec));
    }
    if *n == -1 {
        return Ok(cnum::zero(prec));
    }
    if *n < -1 {
        return Err(format!(
            "integer height {} is undefined for tetration (would require log_b(0) and beyond)",
            cnum::DisplayFloat(n)
        ));
    }
    if b.imag().is_zero() && *b.real() == -1 {
        return Ok(Complex::with_val_64(prec, b));
    }
    let ln_b = Complex::with_val_64(prec, b.ln_ref());
    let mut acc = Complex::with_val_64(prec, b);
    let mut level = Integer::from(2);
    while *n >= level {
        if cnum::verbose() && (level == 2 || level.is_divisible_u(1024)) {
            eprintln!(
                "integer tower: height {level} of {:.8}",
                cnum::DisplayFloat(n)
            );
        }
        let argument = Complex::with_val_64(prec, &ln_b * &acc);
        acc = cnum::checked_exp(&argument, prec)
            .map_err(|e| format!("integer tower at height {}: {}", level, e))?;
        level += 1;
    }
    Ok(acc)
}
