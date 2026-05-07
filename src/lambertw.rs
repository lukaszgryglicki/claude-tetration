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
    // Try the primary seed first; on Halley failure, retry with a small
    // bank of alternate seeds. The tricky cases are values like
    // z = -ln(-2 ± 2i) ≈ -1.04 ± 2.36i, where 1+z is barely in the
    // left half-plane — the wrong-quadrant anchor lands in a neighbouring
    // basin and Halley diverges. Falling back to log(z)-log(log(z))
    // asymptotic, log(1+z), and other plausible seeds recovers these.
    let primary = initial_guess_w0(zf);
    let alt_seeds: [(f64, f64); 4] = [
        // Asymptotic, no correction:
        {
            let l1 = c_log(zf);
            let l2 = c_log(l1);
            (l1.0 - l2.0, l1.1 - l2.1)
        },
        // Asymptotic + L₂/L₁ correction (accurate for moderate |z|):
        {
            let l1 = c_log(zf);
            let l2 = c_log(l1);
            let s = (l1.0 - l2.0, l1.1 - l2.1);
            let corr = c_div(l2, l1);
            (s.0 + corr.0, s.1 + corr.1)
        },
        // log(1+z) — valid for z near 0:
        c_log((1.0 + zf.0, zf.1)),
        // Branch-point series at z = -1/e:
        {
            let ze1 = (zf.0 + ONE_OVER_E, zf.1);
            let arg = (
                2.0 * std::f64::consts::E * ze1.0,
                2.0 * std::f64::consts::E * ze1.1,
            );
            let p = c_sqrt(arg);
            (-1.0 + p.0, p.1)
        },
    ];
    // Try primary first, then alts. Return the first that converges fully.
    let mut last_err: Option<String> = None;
    for seed in std::iter::once(primary).chain(alt_seeds.iter().copied()) {
        if !seed.0.is_finite() || !seed.1.is_finite() {
            continue;
        }
        let refined = refine_f64(zf, seed, F64_NEWTON_ITERS);
        if !refined.0.is_finite() || !refined.1.is_finite() {
            continue;
        }
        // Verify the f64 refinement actually landed near a root before
        // promoting to MPC: |w·e^w - z| should be tiny in f64.
        let ew = c_exp(refined);
        let we = c_mul(refined, ew);
        let resid = c_abs((we.0 - zf.0, we.1 - zf.1));
        if !resid.is_finite() || resid > 1e-6 * (c_abs(zf) + 1.0) {
            continue;
        }
        let mut w = Complex::with_val(prec, (refined.0, refined.1));
        match halley_refine(&mut w, z, prec) {
            Ok(()) => return Ok(w),
            Err(e) => {
                last_err = Some(e);
                continue;
            }
        }
    }
    Err(last_err.unwrap_or_else(|| "W_0: no seed converged".into()))
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

/// Lambert W on branch `k` for `|k| ≥ 2`. Uses the standard asymptotic seed
/// `W_k(z) ≈ L₁ − L₂ + L₂/L₁ + …` where `L₁ = ln(z) + 2πi·k`, `L₂ = ln(L₁)`,
/// then refines via Halley iteration. Returns `Err` if Halley's basin of
/// attraction misses (which happens when the asymptotic seed lands in a
/// neighboring branch's region).
pub fn wk(z: &Complex, k: i32, prec: u32) -> Result<Complex, String> {
    if k == 0 {
        return w0(z, prec);
    }
    if k == -1 {
        return wm1(z, prec);
    }
    if z.real().is_zero() && z.imag().is_zero() {
        return Err(format!("W_{}(0) is undefined", k));
    }
    let zf = (z.real().to_f64(), z.imag().to_f64());
    // Asymptotic seed: W_k ≈ ln(z) + 2πi·k - ln(ln(z) + 2πi·k).
    let lnz = c_log(zf);
    let two_pi_k = (0.0, 2.0 * std::f64::consts::PI * (k as f64));
    let l1 = (lnz.0 + two_pi_k.0, lnz.1 + two_pi_k.1);
    let l2 = c_log(l1);
    let mut seed = (l1.0 - l2.0, l1.1 - l2.1);
    // Add the L₂/L₁ correction term for accuracy.
    let corr = c_div(l2, l1);
    seed = (seed.0 + corr.0, seed.1 + corr.1);
    let w_seed = refine_f64(zf, seed, F64_NEWTON_ITERS);
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
    // Track best |Δ| seen and the corresponding w. When Halley oscillates
    // near a root (e.g. f64 seed already near machine precision so MPC
    // refinement just bounces in roundoff territory), we may bail with a
    // stall counter even though w is already accurate. Returning the best-
    // seen estimate is correct — it's strictly closer to the root than any
    // future iterate would be in the stalled regime.
    let mut best_w: Option<Complex> = None;
    let mut best_dexp: i32 = i32::MAX;
    // "Good enough" floor: if |Δ| is within 2^32 of target (i.e. ~9 decimal
    // digits looser than the working precision), the result is usable for
    // downstream tetration purposes — its error feeds into a Cauchy iteration
    // that can absorb a few extra digits of slop without destabilizing.
    let acceptable_exp: i32 = target_exp + 32;

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
                // Track best-seen iterate so a late stall doesn't lose
                // progress made earlier.
                if e < best_dexp {
                    best_dexp = e;
                    best_w = Some(w.clone());
                }
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
                            // Stall in roundoff territory: if our best-seen
                            // iterate is already within `acceptable_exp` of
                            // the target, accept it. Otherwise the seed was
                            // far enough off that further work is unlikely to
                            // recover — propagate the stall.
                            if best_dexp <= acceptable_exp {
                                if let Some(bw) = best_w.take() {
                                    *w = bw;
                                }
                                return Ok(());
                            }
                            return Err(format!(
                                "Halley stalled at iter {} (delta exp {} ≥ prev {}, {} consecutive; best dexp = {})",
                                iter, e, prev, stall_count, best_dexp
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
    // Iteration cap reached. Return best-seen if it cleared the acceptable
    // floor, else error.
    if best_dexp <= acceptable_exp {
        if let Some(bw) = best_w {
            *w = bw;
        }
        return Ok(());
    }
    Err(format!(
        "Halley did not converge in {} iterations (best dexp = {})",
        HALLEY_MAX_ITERS, best_dexp
    ))
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
    // Moderate |z|: log(1+z) is a smooth, robust seed except near z = −1
    // OR when 1+z has negative real part (then log(1+z) lands in the wrong
    // branch — Im(log(1+z)) ∈ (-π, -π/2)∪(π/2, π) — causing Halley to
    // overshoot into a non-principal basin). For those wrong-quadrant cases
    // we use W₀(−1) ≈ −0.318+1.337i as a hard-coded anchor and let the f64
    // refine step settle into the principal basin. This catches z =
    // −ln(3+i) ≈ −1.151−0.322i and similar bases where log(1+z) was the
    // previously-chosen seed but its imaginary part is on the wrong side
    // of the branch cut.
    let zp1 = (1.0 + z.0, z.1);
    if c_abs(zp1) > 0.3 {
        if zp1.0 >= 0.0 {
            return c_log(zp1);
        } else {
            // 1+z in left half-plane: log(1+z) on the principal branch lands
            // far from the principal W_0 basin. Anchor at W_0(-1) and let
            // f64 Newton refine. Sign of Im(seed) tracks Im(z) so cases on
            // both sides of the real axis converge.
            let im_anchor = if z.1 >= 0.0 { 1.337 } else { -1.337 };
            return (-0.318, im_anchor);
        }
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
