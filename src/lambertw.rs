//! Complex Lambert W function at arbitrary precision.
//!
//! Provides the principal branch `W₀` and the `W₋₁` branch. Used by
//! `regions.rs` to compute the attracting fixed point `L = -W₀(-ln b) / ln b`
//! of the map `z ↦ b^z` (and by Schröder/Kouznetsov in later phases).
//!
//! Algorithm:
//!   1. Compute a robust f64 initial guess (asymptotic / small-arg /
//!      branch-point series depending on `|z|` and `|z + 1/e|`).
//!   2. Refine to ~ f64 accuracy with a few Newton iterations in f64
//!      complex arithmetic (cheap).
//!   3. Refine to full MPC precision with Halley iteration. Halley has
//!      cubic convergence, so even 1 M-digit targets need only ~12 iterates
//!      after the f64 seed.
//!
//! Convergence criterion: stop when `|Δ| < 2^{-(prec - 8)}` measured by
//! `Float::get_exp()` on `|Δ|`.
//!
//! Branch policy:
//!   * `W₀` is single-valued on `ℂ \ (−∞, −1/e]` and matches MPFR/MPC log
//!     conventions there.
//!   * `W₋₁` is the branch with `W₋₁(z) → −∞` as `z → 0⁻` along the real
//!     axis, with the standard cut on `(−∞, 0)`.

use rug::{Complex, Float};

use crate::cnum;

const HALLEY_MAX_ITERS: u32 = 200;
const F64_NEWTON_ITERS: u32 = 30;

/// Principal-branch Lambert W (`W₀`) at arbitrary-precision complex `z`.
pub fn w0(z: &Complex, prec: u32) -> Result<Complex, String> {
    if z.real().is_zero() && z.imag().is_zero() {
        return Ok(cnum::zero(prec));
    }
    let zf = (z.real().to_f64(), z.imag().to_f64());
    let w_seed = initial_guess_w0(zf);
    let w_seed = refine_f64(zf, w_seed, F64_NEWTON_ITERS);
    let mut w = Complex::with_val(prec, (w_seed.0, w_seed.1));
    halley_refine(&mut w, z, prec)?;
    Ok(w)
}

/// `W₋₁` branch at arbitrary-precision complex `z`.
pub fn wm1(z: &Complex, prec: u32) -> Result<Complex, String> {
    if z.real().is_zero() && z.imag().is_zero() {
        return Err("W_{-1}(0) is undefined (it is -infinity)".into());
    }
    let zf = (z.real().to_f64(), z.imag().to_f64());
    let w_seed = initial_guess_wm1(zf);
    let w_seed = refine_f64(zf, w_seed, F64_NEWTON_ITERS);
    let mut w = Complex::with_val(prec, (w_seed.0, w_seed.1));
    halley_refine(&mut w, z, prec)?;
    Ok(w)
}

/// Halley refinement of `w` toward the root of `w · exp(w) − z = 0`.
///
/// Halley's update: `w_{n+1} = w_n − f / (f' − f f'' / (2 f'))`,
/// with `f = w e^w − z`, `f' = (w+1) e^w`, `f'' = (w+2) e^w`.
/// Substituting:
///     `w_{n+1} = w_n − f / ( (w+1) e^w − (w+2) f / (2 (w+1)) )`.
fn halley_refine(w: &mut Complex, z: &Complex, prec: u32) -> Result<(), String> {
    let one = cnum::one(prec);
    let two = Complex::with_val(prec, (2, 0));
    // Convergence target: |Δ| < 2^{−(prec − 8)}.
    // Float::get_exp returns `e` such that the value is in [2^(e−1), 2^e),
    // so log2(|Δ|) ≤ get_exp(|Δ|).
    let target_exp: i32 = -((prec as i32) - 8);
    let mut prev_dexp: Option<i32> = None;
    let mut stall_count: u32 = 0;

    for iter in 0..HALLEY_MAX_ITERS {
        let exp_w = Complex::with_val(prec, w.exp_ref());
        let we = Complex::with_val(prec, &*w * &exp_w);
        let f = Complex::with_val(prec, &we - z);
        let wp1 = Complex::with_val(prec, &*w + &one);
        let wp2 = Complex::with_val(prec, &*w + &two);
        let two_wp1 = Complex::with_val(prec, &wp1 * &two);

        // correction = (w+2) · f / (2 (w+1))
        let f_wp2 = Complex::with_val(prec, &f * &wp2);
        let correction = Complex::with_val(prec, &f_wp2 / &two_wp1);

        // denom = (w+1) e^w − correction
        let denom_a = Complex::with_val(prec, &wp1 * &exp_w);
        let denom = Complex::with_val(prec, &denom_a - &correction);
        if denom.real().is_zero() && denom.imag().is_zero() {
            return Err(format!("Halley denominator zero at iter {}", iter));
        }
        let delta = Complex::with_val(prec, &f / &denom);
        let new_w = Complex::with_val(prec, &*w - &delta);

        // Convergence: examine |Δ| via abs_ref → Float.
        let delta_abs = Float::with_val(prec, delta.abs_ref());
        let dexp = delta_abs.get_exp();

        *w = new_w;

        match dexp {
            None => return Ok(()), // delta is 0 / NaN (treat as converged or done)
            Some(e) => {
                if e <= target_exp {
                    return Ok(());
                }
                // Divergence guard: cubic Halley should drop |Δ|'s exponent
                // every step. Tolerate the occasional stall (the f64 seed can
                // briefly worsen at full precision) but bail after a streak.
                if let Some(prev) = prev_dexp {
                    if e >= prev {
                        stall_count += 1;
                        if stall_count >= 5 {
                            return Err(format!(
                                "Halley stalled at iter {} (delta exp {} ≥ prev {}, {} consecutive)",
                                iter, e, prev, stall_count
                            ));
                        }
                    } else {
                        stall_count = 0;
                    }
                }
                prev_dexp = Some(e);
            }
        }
    }
    Err(format!("Halley did not converge in {} iterations", HALLEY_MAX_ITERS))
}

// ============ f64 helpers ============

fn c_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn c_div(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}

fn c_exp(z: (f64, f64)) -> (f64, f64) {
    let m = z.0.exp();
    (m * z.1.cos(), m * z.1.sin())
}

fn c_log(z: (f64, f64)) -> (f64, f64) {
    let r = (z.0 * z.0 + z.1 * z.1).sqrt();
    (r.ln(), z.1.atan2(z.0))
}

fn c_sqrt(z: (f64, f64)) -> (f64, f64) {
    let r = (z.0 * z.0 + z.1 * z.1).sqrt();
    let mag = r.sqrt();
    let arg = z.1.atan2(z.0) * 0.5;
    (mag * arg.cos(), mag * arg.sin())
}

fn c_abs(z: (f64, f64)) -> f64 {
    (z.0 * z.0 + z.1 * z.1).sqrt()
}

/// f64 Newton iteration on `w · exp(w) = z`. Used to refine the heuristic
/// initial guess to ~ f64 accuracy before promoting to MPC.
fn refine_f64(z: (f64, f64), w0: (f64, f64), iters: u32) -> (f64, f64) {
    let (mut wr, mut wi) = w0;
    for _ in 0..iters {
        let ew = c_exp((wr, wi));
        let we = c_mul((wr, wi), ew);
        let f = (we.0 - z.0, we.1 - z.1);
        if c_abs(f) < 1e-15 * (c_abs((wr, wi)) + 1.0) {
            break;
        }
        // Newton: Δ = f / ((1+w) e^w) = (w − z e^{−w}) / (1+w)
        let wp1 = (1.0 + wr, wi);
        let denom = c_mul(wp1, ew);
        let d = c_div(f, denom);
        let next = (wr - d.0, wi - d.1);
        if !next.0.is_finite() || !next.1.is_finite() {
            // Stay with previous guess; full-precision Halley will refine.
            break;
        }
        wr = next.0;
        wi = next.1;
    }
    (wr, wi)
}

const ONE_OVER_E: f64 = 0.36787944117144233; // 1/e

fn initial_guess_w0(z: (f64, f64)) -> (f64, f64) {
    // Branch-point series: W₀(z) ≈ −1 + p − p²/3 + 11 p³/72 − 43 p⁴/540 + ...
    // with p = sqrt(2(ez + 1)). Widened threshold to 0.5 so real z slightly
    // below −1/e (e.g. z = −ln 2 ≈ −0.693, |z+1/e| ≈ 0.325) gets a complex
    // seed instead of a real log(1+z) that traps Halley on the real axis.
    let ze1 = (z.0 + ONE_OVER_E, z.1);
    if c_abs(ze1) < 0.5 {
        let arg = (2.0 * std::f64::consts::E * ze1.0,
                   2.0 * std::f64::consts::E * ze1.1);
        let p = c_sqrt(arg);
        let p2 = c_mul(p, p);
        let p3 = c_mul(p2, p);
        let p4 = c_mul(p3, p);
        let mut w = (-1.0 + p.0, p.1);
        w.0 -= p2.0 / 3.0; w.1 -= p2.1 / 3.0;
        w.0 += 11.0 * p3.0 / 72.0; w.1 += 11.0 * p3.1 / 72.0;
        w.0 -= 43.0 * p4.0 / 540.0; w.1 -= 43.0 * p4.1 / 540.0;
        return w;
    }
    if c_abs(z) < 0.3 {
        // Small argument: W₀(z) ≈ z − z² + 3 z³/2 − 8 z⁴/3
        let z2 = c_mul(z, z);
        let z3 = c_mul(z2, z);
        let z4 = c_mul(z3, z);
        let mut w = z;
        w.0 -= z2.0; w.1 -= z2.1;
        w.0 += 1.5 * z3.0; w.1 += 1.5 * z3.1;
        w.0 -= 8.0 / 3.0 * z4.0; w.1 -= 8.0 / 3.0 * z4.1;
        return w;
    }
    // For large |z|, log(z) − log(log(z)) is accurate.
    if c_abs(z) > 5.0 {
        let l1 = c_log(z);
        let l2 = c_log(l1);
        return (l1.0 - l2.0, l1.1 - l2.1);
    }
    // Negative real z below the branch point that fell outside the branch-
    // point window: log(z) is genuinely complex (atan2 picks +π), so the
    // asymptotic seed is complex and Halley can leave the real axis.
    if z.0 < -ONE_OVER_E && z.1 == 0.0 {
        let l1 = c_log(z);
        let l2 = c_log(l1);
        return (l1.0 - l2.0, l1.1 - l2.1);
    }
    // Moderate |z|: log(1+z) is a smooth, robust seed except near z = −1.
    let zp1 = (1.0 + z.0, z.1);
    if c_abs(zp1) > 0.3 {
        return c_log(zp1);
    }
    // z near −1: fall back to asymptotic.
    let l1 = c_log(z);
    let l2 = c_log(l1);
    (l1.0 - l2.0, l1.1 - l2.1)
}

fn initial_guess_wm1(z: (f64, f64)) -> (f64, f64) {
    // Branch-point series with negative p:
    let ze1 = (z.0 + ONE_OVER_E, z.1);
    if c_abs(ze1) < 0.3 {
        let arg = (2.0 * std::f64::consts::E * ze1.0,
                   2.0 * std::f64::consts::E * ze1.1);
        let p = c_sqrt(arg);
        let p2 = c_mul(p, p);
        let p3 = c_mul(p2, p);
        // W₋₁ ≈ −1 − p − p²/3 − 11 p³/72 − ...
        let mut w = (-1.0 - p.0, -p.1);
        w.0 -= p2.0 / 3.0; w.1 -= p2.1 / 3.0;
        w.0 -= 11.0 * p3.0 / 72.0; w.1 -= 11.0 * p3.1 / 72.0;
        return w;
    }
    // Otherwise: asymptotic with branch shift.
    // For z near 0⁻ (real, slightly negative): W₋₁(z) ~ log(−z) − log(−log(−z)).
    // For complex z, choose log branch via −2π i shift relative to W₀ when needed.
    let l1 = c_log(z);
    // Shift by −2π i to get the W₋₁ branch from the W₀ formula.
    let l1 = (l1.0, l1.1 - 2.0 * std::f64::consts::PI);
    let l2 = c_log(l1);
    (l1.0 - l2.0, l1.1 - l2.1)
}
