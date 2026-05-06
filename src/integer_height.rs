//! Direct iteration for integer heights.
//!
//! For non-negative `n`: `F_b(0) = 1`, `F_b(n+1) = b^F_b(n)`.
//! For `n = -1`: `F_b(-1) = 0` (the unique value that gives `F_b(0) = b^0 = 1`).
//! For `n ≤ -2`: undefined for general `b` (would require `log_b(0)` and beyond);
//! we error out — see plan.

use rug::Complex;

use crate::cnum;

/// Maximum integer height we'll iterate to before refusing. The tower grows so
/// fast that the result is astronomical past ~6, but precision-bounded so we
/// allow up to a few thousand levels for users who want to see growth at high
/// precision (each level costs one `b^x` which is one ln + one exp).
pub const MAX_INTEGER_HEIGHT: i64 = 100_000;

/// Compute `F_b(n)` for integer `n` by direct iteration. Errors for `n ≤ -2`
/// (undefined in general) and for `|n|` larger than `MAX_INTEGER_HEIGHT`.
pub fn tetrate_integer(b: &Complex, n: i64, prec: u32) -> Result<Complex, String> {
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
    let mut acc = cnum::one(prec);
    for _ in 0..n {
        acc = cnum::pow_complex(b, &acc, prec);
    }
    Ok(acc)
}
