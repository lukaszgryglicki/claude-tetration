//! Base classification: decide which tetration algorithm should handle a base.
//!
//! For the iteration `f(z) = b^z`, the attracting fixed point is
//!     `L = -W₀(-ln b) / ln b`
//! and the multiplier is `λ = f'(L) = L · ln b`. The "Shell-Thron region" is the
//! set of bases where `|λ| ≤ 1`; inside this region, regular tetration via
//! Schröder's equation is the natural construction. Outside, we need
//! Kouznetsov / Paulsen-Cowgill etc.
//!
//! Classification reports `|λ|` so the dispatcher can apply a guard band around
//! `|λ| = 1` (Schröder convergence stalls there).

use rug::{Complex, Float};

use crate::{cnum, lambertw};

/// Classification of a base for tetration purposes.
#[derive(Debug, Clone)]
pub enum Region {
    /// `b == 1` — `b^^h = 1` for all `h`.
    BaseOne,
    /// `b == 0` — alternates `1, 0, 1, 0, ...` for non-negative integer height.
    BaseZero,
    /// `|λ| < 0.95`. Regular tetration via Schröder's equation works comfortably.
    ShellThronInterior(FixedPointData),
    /// `0.95 ≤ |λ| ≤ 1.05`. Schröder unreliable; defer to Paulsen-Cowgill (or
    /// linear approx fallback) — convergence near the parabolic boundary stalls.
    ShellThronBoundary(FixedPointData),
    /// `|λ| > 1.05` and base is real positive (`> 1`, with the special case
    /// `b > e^(1/e)`). Kouznetsov's Cauchy-contour method is the strongest fit.
    OutsideShellThronRealPositive(FixedPointData),
    /// `|λ| > 1.05` and base is general complex (real negative, imaginary, or
    /// truly complex). Paulsen-Cowgill is the most general method.
    OutsideShellThronGeneral(FixedPointData),
}

/// Data attached to every non-special classification: the attracting fixed
/// point `L` (where `b^L = L`), the multiplier `λ = L · ln b`, and `|λ|` as f64
/// for fast region tests.
#[derive(Debug, Clone)]
pub struct FixedPointData {
    pub fixed_point: Complex,
    pub lambda: Complex,
    pub lambda_abs: f64,
}

impl Region {
    pub fn name(&self) -> &'static str {
        match self {
            Region::BaseOne => "base-one",
            Region::BaseZero => "base-zero",
            Region::ShellThronInterior(_) => "shell-thron-interior",
            Region::ShellThronBoundary(_) => "shell-thron-boundary",
            Region::OutsideShellThronRealPositive(_) => "outside-shell-thron-real-positive",
            Region::OutsideShellThronGeneral(_) => "outside-shell-thron-general",
        }
    }
}

/// Inner / outer band thresholds for `|λ|`. The narrow band around `|λ| = 1`
/// avoids Schröder when convergence rate is too slow.
pub const SHELL_THRON_INTERIOR_THRESHOLD: f64 = 0.95;
pub const SHELL_THRON_OUTER_THRESHOLD: f64 = 1.05;

/// Classify base `b`. Computes the fixed-point data via Lambert W and labels
/// the region used by `dispatch::tetrate`.
pub fn classify(b: &Complex, prec: u32) -> Result<Region, String> {
    if cnum::is_one(b) {
        return Ok(Region::BaseOne);
    }
    if cnum::is_zero(b) {
        return Ok(Region::BaseZero);
    }

    // L = -W₀(-ln b) / ln b
    let ln_b = Complex::with_val(prec, b.ln_ref());
    let neg_ln_b = Complex::with_val(prec, -&ln_b);
    let w = lambertw::w0(&neg_ln_b, prec)?;
    let neg_w = Complex::with_val(prec, -&w);
    let l = Complex::with_val(prec, &neg_w / &ln_b);
    let lambda = Complex::with_val(prec, &l * &ln_b);
    let lambda_abs_f = Float::with_val(prec, lambda.abs_ref());
    let lambda_abs = lambda_abs_f.to_f64();

    let data = FixedPointData {
        fixed_point: l,
        lambda,
        lambda_abs,
    };

    let is_real_positive = b.imag().is_zero() && b.real().is_sign_positive() && !b.real().is_zero();

    if lambda_abs < SHELL_THRON_INTERIOR_THRESHOLD {
        Ok(Region::ShellThronInterior(data))
    } else if lambda_abs <= SHELL_THRON_OUTER_THRESHOLD {
        Ok(Region::ShellThronBoundary(data))
    } else if is_real_positive {
        Ok(Region::OutsideShellThronRealPositive(data))
    } else {
        Ok(Region::OutsideShellThronGeneral(data))
    }
}
