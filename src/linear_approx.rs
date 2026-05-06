//! Linear (C⁰) approximation to tetration.
//!
//! Conventional construction: on the strip `Re(h) ∈ [0,1)`, define
//! `F_b(h) = 1 + h(b - 1)` (linear interpolation between `F_b(0)=1` and
//! `F_b(1)=b`). For `Re(h) ∉ [0,1)`, lift/lower by integer steps using the
//! functional equation `F(h+1) = b^F(h)`.
//!
//! This is C⁰-continuous at integers but NOT analytic. It serves as a fallback
//! when no proper algorithm has been implemented for the requested base/height,
//! and as an initial guess for iterative algorithms in later phases.

use rug::{Complex, Float};

use crate::cnum;

/// Linear approximation to `F_b(h)`. Returns `Err` if `Re(h)` is so large that
/// the integer offset doesn't fit in `i64` (the tower would have astronomically
/// many levels anyway).
pub fn tetrate_linear(b: &Complex, h: &Complex, prec: u32) -> Result<Complex, String> {
    // n = floor(Re(h)); f = h - n so Re(f) ∈ [0, 1).
    let mut re_floor: Float = h.real().clone();
    re_floor.floor_mut();
    let n_int = re_floor
        .to_integer()
        .ok_or_else(|| "height real part is non-finite".to_string())?;
    let n: i64 = n_int
        .to_i64()
        .ok_or_else(|| "height integer-offset doesn't fit in i64".to_string())?;
    if n.unsigned_abs() > crate::integer_height::MAX_INTEGER_HEIGHT as u64 {
        return Err(format!(
            "linear-approx integer offset |{}| exceeds MAX_INTEGER_HEIGHT",
            n
        ));
    }

    let n_complex = Complex::with_val(prec, (n, 0));
    let f = Complex::with_val(prec, h - &n_complex);

    // Base case on the unit strip: F(f) = 1 + f * (b - 1).
    let one = cnum::one(prec);
    let bm1 = Complex::with_val(prec, b - &one);
    let f_bm1 = Complex::with_val(prec, &f * &bm1);
    let mut acc = Complex::with_val(prec, &one + &f_bm1);

    if n > 0 {
        // Lift: apply b^· n times.
        for _ in 0..n {
            acc = cnum::pow_complex(b, &acc, prec);
        }
    } else if n < 0 {
        // Lower: apply log_b · |n| times.
        // Each step has a branch ambiguity for non-real b; we use the principal branch.
        for _ in 0..(-n) {
            acc = cnum::log_b_complex(&acc, b, prec);
        }
    }

    Ok(acc)
}
