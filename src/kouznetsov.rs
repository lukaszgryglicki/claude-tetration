//! Kouznetsov-Trappmann Cauchy iteration for tetration.
//!
//! Implements the natural complex tetration `F` satisfying:
//!   F(0) = 1
//!   F(z+1) = b^F(z)
//!   F(z̄) = F̄(z)                          (real on the real axis)
//!   F(z) → L̄ as Im(z) → +∞                (upper fixed point)
//!   F(z) → L  as Im(z) → -∞                (lower fixed point)
//!
//! Used for real bases `b > e^(1/e)`, where σ̃ Taylor at the complex fixed
//! point has too small a radius to reach 1−L directly (Schröder fails) and the
//! repelling-case σ̃-shift hits a singularity at `L+w=0` (also fails).
//!
//! The algorithm samples F at N points along the line Re(z) = 0.5 and
//! iteratively refines via Cauchy's integral on the rectangle
//! Re ∈ [-0.5, 1.5], Im ∈ [-T, T]:
//!   - top edge (Im = T):    F = L̄
//!   - bottom edge (Im = -T): F = L
//!   - right edge (Re = 1.5): F(1.5+it) = b^F(0.5+it)
//!   - left edge (Re = -0.5): F(-0.5+it) = log_b(F(0.5+it))
//!
//! Picard alone on this Cauchy operator has spectral radius near 1 (slow / no
//! contraction), so we wrap it in **Anderson acceleration** (depth 5). At each
//! step Anderson uses recent (x, T(x)) pairs to extrapolate the fixed point via
//! a small complex least-squares system. This recovers super-linear convergence
//! on operators where Picard merely oscillates.
//!
//! At convergence, F at any complex height `h` is found by integer-shifting `h`
//! into Re ∈ [0, 1] and applying the Cauchy formula one more time, then
//! iterating `b^·` (`shift > 0`) or `log_b` (`shift < 0`) back.

use rug::{float::Constant, Complex, Float, Integer, Rational};
use std::sync::{Arc, Mutex, OnceLock};

use crate::{
    cnum,
    fft::{cross_correlate_with_kernel, precompute_kernel_fft, KernelFft},
    lambertw,
    regions::FixedPointData,
};
use cnum::{log_magnitude, DisplayComplex, DisplayFloat};

/// Compute `F_b(h)` via Newton-Kantorovich Cauchy iteration on the
/// Kouznetsov-style rectangle. Works for general complex bases `b` outside
/// Shell-Thron: the natural F is fixed by the asymptotic conditions
/// `F → L_upper` (Im → +∞) and `F → L_lower` (Im → −∞), where the two
/// fixed points of `b^z = z` sit in opposite half-planes.
///
/// For real bases `b > e^(1/e)`, the two fixed points are complex conjugates
/// (Schwarz reflection), and the natural F satisfies `F(z̄) = F̄(z)`; the
/// iteration symmetrizes each iterate to keep us on that manifold. For
/// general complex bases the conjugate symmetry is broken, so we drop the
/// symmetrize step and pick `L_upper / L_lower` from the `W₀` and `W₋₁`
/// branches independently.
/// Per-base precomputed state for Kouznetsov Cauchy iteration. Captures
/// every piece of the reconstruction that depends only on `b` (not on the
/// height `h`): the fixed-point pair, the rectangle nodes/weights, the
/// converged samples on Re(z)=0.5, and the normalization shift δ such that
/// F(δ)=1.
///
/// Hoisting this out of `tetrate_kouznetsov` lets a grid evaluator amortize
/// the expensive setup (typically dominating per-call cost) across many
/// heights for the same base.
#[derive(Clone)]
pub struct KouznetsovState {
    pub samples: Vec<Complex>,
    pub nodes: Vec<Float>,
    pub weights: Vec<Float>,
    pub t_max: Float,
    pub l_upper: Complex,
    pub l_lower: Complex,
    pub ln_b: Complex,
    pub shift: Complex,
    pub prec: u64,
    pub digits: u64,
    pub normalized: bool,
    /// Achieved boundary residual, not a forward-error or branch certificate.
    /// Internal warm states may miss the full 10^-(digits+3) target;
    /// eval_kouznetsov rejects such candidates as final answers.
    pub residual: Float,
    /// Whether the left-edge integrand log was built with the two-sided
    /// anchored unwrap (cut-base ε-walk states) or the pointwise principal
    /// log (all other bases; the historically correct operator).
    pub two_sided: bool,
}

pub fn tetrate_kouznetsov(
    b: &Complex,
    h: &Complex,
    fp: &FixedPointData,
    prec: u64,
    digits: u64,
) -> Result<Complex, String> {
    let state = setup_kouznetsov(b, fp, prec, digits)?;
    eval_kouznetsov(&state, b, h)
}

/// Compute the per-base Kouznetsov state. This is the expensive step: it
/// runs the Newton-Kantorovich Cauchy iteration to convergence and then
/// finds the normalization shift δ. Most heights reuse this state;
/// ill-conditioned shifts may require a more accurate reconstruction.
pub fn setup_kouznetsov(
    b: &Complex,
    fp: &FixedPointData,
    prec: u64,
    digits: u64,
) -> Result<KouznetsovState, String> {
    crate::mt::init_pool()?;
    cnum::require_precision(prec, digits)?;
    if !cnum::is_finite(b)
        || cnum::is_zero(b)
        || cnum::is_one(b)
        || digits == 0
        || !cnum::is_finite(&fp.fixed_point)
        || !cnum::is_finite(&fp.lambda)
    {
        return Err("Kouznetsov setup requires a finite nondegenerate base, fixed point, and positive precision".into());
    }
    // Decide regime: real positive base allows Schwarz symmetry; everything
    // else (real negative, imaginary, general complex) does not.
    let use_schwarz = is_real_positive(b);
    if !use_schwarz {
        // Bases on the boundary or with degenerate fixed points are still
        // routed away by `dispatch.rs`; we just need to ensure ln(b) is
        // well-defined here.
        if cnum::is_zero(b) {
            return Err("Kouznetsov: b = 0 unsupported".into());
        }
    }

    // Fixed-point pair. For real bases > e^(1/e) we use `fp.fixed_point` =
    // `-W₀(-ln b)/ln b` and its complex conjugate (since the two fixed points
    // are conjugate in the real case, regardless of which branch was sampled).
    // For complex bases the two fixed points come from distinct W branches
    // (W₀ and W₋₁), so we recompute both explicitly.
    let ln_b = cnum::ln_complex(b, prec);
    let neg_ln_b = Complex::with_val_64(prec, -&ln_b);

    let raw = fp.fixed_point.clone();
    let (l_lower, l_upper) = if use_schwarz {
        let raw_conj = Complex::with_val_64(prec, raw.conj_ref());
        let raw_imag_neg = raw.imag().is_sign_negative();
        if raw_imag_neg {
            (raw, raw_conj)
        } else {
            (raw_conj, raw)
        }
    } else {
        // For complex bases the natural fixed-point pair is the analytic
        // continuation from the real-base case: L_+ = -W₀(-ln b)/ln b is
        // continuous, but its partner is NOT generally `-W₋₁(-ln b)/ln b`
        // — that's a different W-branch sheet whose value can jump
        // discontinuously as b crosses real (e.g. b=2+0.001i lands W₋₁ near
        // 3.5+10.9i instead of the natural near-conjugate 0.825−1.566i).
        //
        // Instead, find the partner by Newton-iterating `b^z = z` from
        // `conj(L_+)`. For real b this seed is already a fixed point (zero
        // iterations); for slightly-complex b it converges to the natural
        // near-conjugate fixed point in 5–20 iterations.
        let w0_val = lambertw::w0(&neg_ln_b, prec)?;
        let neg_w0 = Complex::with_val_64(prec, -&w0_val);
        let l_plus = Complex::with_val_64(prec, &neg_w0 / &ln_b);
        let seed = Complex::with_val_64(prec, l_plus.conj_ref());
        let mut l_minus = newton_fixed_point(&ln_b, &seed, prec).map_err(|e| {
            format!(
                "Kouznetsov: could not find partner fixed point near conj(L_+): {}",
                e
            )
        })?;
        let im_plus = l_plus.imag().clone();
        let im_minus_init = l_minus.imag().clone();
        // If Newton-from-conjugate landed in the same half-plane as L_+, the
        // rectangle Cauchy formula's boundary conditions become ill-posed.
        // Search W_k branches (k = 1, -2, 2, -3, 3, …) for a fixed point in
        // the opposite half-plane. This is a heuristic Paulsen-Cowgill-style
        // branch selection; for negative real bases and far-complex bases the
        // resulting tetration may not be the canonical Kneser-Kouznetsov F,
        // but it satisfies F(0)=1 and F(z+1)=b^F(z), which is verified by
        // the iteration's residual.
        //
        // (Cut bases 0 < b < e^{-e} never reach this search: their pair
        // (W₀, W₊₁) — same-half-plane, opposite arg λ — is injected
        // directly into `setup_kouznetsov_core` by the ε-walker.)
        let mut _wk_used = false;
        if im_plus.is_sign_negative() == im_minus_init.is_sign_negative()
            && im_plus.clone().abs() > cnum::decimal("1e-12", prec)
            && im_minus_init.clone().abs() > cnum::decimal("1e-12", prec)
        {
            // Search W_k for an opposite-half-plane fixed point. k=-1 is the
            // canonical Kneser partner for real bases (since conj(W_0) ≈ W_-1
            // there); for complex bases where Newton-from-conjugate fails, W_-1
            // is still typically the correct analytic continuation. We collect
            // ALL valid candidates and rank by:
            //   1. |Im(L_k)| > 0.05 (avoid degenerate near-real-axis strips —
            //      W_+1 for b=(-0.8,0.4i) gives Im=-0.01, useless),
            //   2. smallest |k| (closest to canonical Kneser pair).
            const W_K_SEARCH: &[i32] = &[-1, 1, -2, 2, -3, 3, -4, 4, -5, 5];
            let min_im_strip = cnum::decimal("0.05", prec);
            let debug_wk = cnum::verbose();
            let mut candidates: Vec<(i32, Complex, Float)> = Vec::new();
            for &k in W_K_SEARCH {
                let wk_val = match lambertw::wk(&neg_ln_b, k, prec) {
                    Ok(v) => v,
                    Err(e) => {
                        if debug_wk {
                            eprintln!("kouz wk search: W_{} failed: {}", k, e);
                        }
                        continue;
                    }
                };
                let neg_wk = Complex::with_val_64(prec, -&wk_val);
                let l_k = Complex::with_val_64(prec, &neg_wk / &ln_b);
                let im_k = l_k.imag().clone();
                let re_k = l_k.real().clone();
                // Verify L_k is genuinely a fixed point of b^z = z (Halley
                // can converge to a nearby branch for poor seeds).
                let bz = match cnum::checked_exp(&Complex::with_val_64(prec, &l_k * &ln_b), prec) {
                    Ok(value) => value,
                    Err(error) => {
                        if debug_wk {
                            eprintln!("kouz wk search: W_{k} fixed-point check failed: {error}");
                        }
                        continue;
                    }
                };
                let resid =
                    Float::with_val_64(prec, Complex::with_val_64(prec, &bz - &l_k).abs_ref());
                let opposite = im_k.is_sign_negative() != im_plus.is_sign_negative()
                    && im_k.clone().abs() > min_im_strip;
                if debug_wk {
                    eprintln!(
                        "kouz wk search: k={:>+}  L={:.4}+{:.4}i  resid={:.2e}  opposite={}  |im|>{}={}",
                        k, DisplayFloat(&re_k), DisplayFloat(&im_k), DisplayFloat(&resid),
                        opposite, DisplayFloat(&min_im_strip), im_k.clone().abs() > min_im_strip
                    );
                }
                if opposite
                    && resid.is_finite()
                    && resid
                        < cnum::epsilon(digits.saturating_add(3), prec)
                            * cnum::abs(&l_k, prec).max(&Float::with_val_64(prec, 1))
                {
                    candidates.push((k, l_k, im_k));
                }
            }
            // Pick smallest |k| (canonical preference) — already partially ordered
            // by W_K_SEARCH but we re-sort to be safe and explicit.
            candidates.sort_by(|a, b| {
                a.0.unsigned_abs()
                    .cmp(&b.0.unsigned_abs())
                    .then(a.0.cmp(&b.0))
            });
            if let Some((kchosen, l_k, _imk)) = candidates.into_iter().next() {
                if debug_wk {
                    eprintln!("kouz wk search: chose k={}", kchosen);
                }
                l_minus = l_k;
                _wk_used = true;
            } else {
                return Err(format!(
                    "Kouznetsov: fixed-point pair (L_+ = {:.4}+{:.4}i, L_- = {:.4}+{:.4}i) lies in the same half-plane; W_k search across k∈±[1..5] found no opposite-half-plane partner with |Im|>{}. The implemented contour is unavailable; this does not establish nonexistence.",
                    DisplayFloat(l_plus.real()),
                    DisplayFloat(&im_plus),
                    DisplayFloat(l_minus.real()),
                    DisplayFloat(&im_minus_init),
                    DisplayFloat(&min_im_strip),
                ));
            }
        }
        let im_minus = l_minus.imag();
        if im_plus >= *im_minus {
            (l_minus, l_plus)
        } else {
            (l_plus, l_minus)
        }
    };
    let mut state = setup_kouznetsov_core(
        b,
        l_upper,
        l_lower,
        prec,
        digits,
        use_schwarz,
        None,
        false,
        !use_schwarz,
        false,
        1,
    )?;
    if !use_schwarz {
        // Relaxed internal states are warm candidates only. Final answers
        // require the full residual target, which is still not a certificate
        // of forward accuracy or canonicality.
        let band_gate = cnum::epsilon(digits.saturating_add(3), prec);
        if !(state.residual.is_finite() && state.residual <= band_gate) {
            // Retry with the anchored two-sided left-edge unwrap before giving
            // up: the pointwise principal log can mis-branch the left-edge
            // integrand for complex-base geometries and produce a *phantom*
            // O(1) residual (observed b=-0.8+0.4i: 1.58 principal vs 9.5e-4
            // two-sided at identical samples-quality). If the retry passes the
            // gate it satisfies the residual contract; if both stall, refuse.
            if cnum::verbose() {
                eprintln!(
                    "kouz gate: principal-log residual {:.3e} > gate {:.1e}; retrying with two-sided unwrap",
                    DisplayFloat(&state.residual), DisplayFloat(&band_gate)
                );
            }
            let retry = setup_kouznetsov_core(
                b,
                state.l_upper.clone(),
                state.l_lower.clone(),
                prec,
                digits,
                use_schwarz,
                None,
                false,
                true,
                true,
                1,
            );
            match retry {
                Ok(r) if r.residual.is_finite() && r.residual <= band_gate => state = r,
                Ok(r) => {
                    return Err(format!(
                        "Kouznetsov complex-base solve stalled: boundary residual {:.3e} \
                         (principal log) / {:.3e} (two-sided unwrap retry) both exceed \
                         the requested boundary target {:.1e}; \
                         samples at this level are unreliable (parabolic-band stall)",
                        DisplayFloat(&state.residual),
                        DisplayFloat(&r.residual),
                        DisplayFloat(&band_gate),
                    ));
                }
                Err(e) => {
                    return Err(format!(
                        "Kouznetsov complex-base solve stalled: boundary residual {:.3e} \
                         exceeds the requested boundary target {:.1e} \
                         and the two-sided unwrap retry failed ({}); \
                         samples at this level are unreliable (parabolic-band stall)",
                        DisplayFloat(&state.residual),
                        DisplayFloat(&band_gate),
                        e,
                    ));
                }
            }
        }
        state.shift = find_normalization_shift(
            &state.samples,
            &state.nodes,
            &state.weights,
            &state.t_max,
            &state.l_upper,
            &state.l_lower,
            &state.ln_b,
            prec,
            digits,
            use_schwarz,
            state.two_sided,
        )?;
        state.normalized = true;
        if cnum::verbose() {
            eprintln!(
                "kouz normalization shift δ = {:.6e} + {:.6e}i (such that F(δ)=1)",
                DisplayFloat(state.shift.real()),
                DisplayFloat(state.shift.imag()),
            );
        }
    }
    Ok(state)
}

type WarmGuess<'a> = dyn Fn(&Float) -> Result<Complex, String> + Sync + 'a;

/// Core Kouznetsov solver: given the asymptotic fixed-point pair
/// (`l_upper` for t → +∞, `l_lower` for t → −∞ on the sample line
/// Re z = 1/2), build the grid, run the LM Newton-Kantorovich iteration
/// (with multi-start retries) and normalize. Split out of
/// [`setup_kouznetsov`] so callers that must *override* the pair-selection
/// heuristics (e.g. the cut-base path, whose pair is (W₀, W₊₁) — both
/// fixed points in the closed upper half-plane, which the generic
/// opposite-half-plane search would reject) can inject the pair directly.
///
/// `skip_norm` stores δ = 0 for continuation warm sources or candidates awaiting
/// the final boundary gate. Such states must be normalized before evaluation.
///
/// `node_boost` multiplies the automatic node count (power-of-2 preserving).
/// The cut-base walker passes 2 when the previous curve has a deep pinch
/// (a zero of F within ~0.1 of the sample line): the left-edge integrand
/// `ln F` is then near-singular and the trapezoidal floor at the standard
/// density sits right at the acceptance gate (observed 1.0e-8 at b=0.06,
/// ε≈0.102, |F|min=4.2e-2); doubling the density squares that floor away.
#[allow(clippy::too_many_arguments)]
fn setup_kouznetsov_core(
    b: &Complex,
    l_upper: Complex,
    l_lower: Complex,
    prec: u64,
    digits: u64,
    use_schwarz: bool,
    warm_guess: Option<&WarmGuess<'_>>,
    warm_only: bool,
    skip_norm: bool,
    two_sided: bool,
    node_boost: usize,
) -> Result<KouznetsovState, String> {
    if !cnum::is_finite(b)
        || cnum::is_zero(b)
        || cnum::is_one(b)
        || digits == 0
        || !cnum::is_finite(&l_upper)
        || !cnum::is_finite(&l_lower)
    {
        return Err("Kouznetsov core requires finite nondegenerate inputs".into());
    }
    let ln_b = cnum::ln_complex(b, prec);
    // λ = (ln b)·L drives each side's decay rate. F → L_upper as t → +∞ like
    // λ_up^{it} (rate |arg λ_up|) and F → L_lower as t → −∞ (rate |arg λ_low|).
    // For Schwarz-conjugate pairs the two rates coincide; for asymmetric pairs
    // (cut bases) the smaller one is binding.
    let lambda_upper = Complex::with_val_64(prec, &ln_b * &l_upper);
    let lambda_lower = Complex::with_val_64(prec, &ln_b * &l_lower);

    // Both fixed-point tails must meet the precision budget, on either log sheet.
    let arg_lambda = slowest_decay_rate(&lambda_upper, &lambda_lower, prec);
    let t_max = contour_height(digits, &arg_lambda, prec)?;

    // Trapezoidal node count: scales with both `digits` and `t_max`, with the
    // analyticity-strip width `|arg(λ)|` driving the per-node convergence rate.
    let n_nodes = pick_node_count(digits, &t_max, prec)?
        .checked_mul(node_boost.max(1))
        .ok_or("Kouznetsov node count overflow")?;

    let fft_len = crate::fft::kernel_fft_len(n_nodes, prec)?;
    if cnum::verbose() {
        eprintln!("kouz resources: {n_nodes} nodes, FFT length {fft_len}, {prec} bits; allocating geometry");
    }

    let nodes = build_uniform_nodes(&t_max, n_nodes, prec);
    let weights = build_trapezoidal_weights(&t_max, n_nodes, prec);

    if cnum::verbose() {
        eprintln!(
            "kouz setup: |arg(λ)|={:.4}  t_max={:.2}  n_nodes={}  L_upper={:.4}+{:.4}i  L_lower={:.4}+{:.4}i",
            DisplayFloat(&arg_lambda),
            DisplayFloat(&t_max),
            n_nodes,
            DisplayFloat(l_upper.real()),
            DisplayFloat(l_upper.imag()),
            DisplayFloat(l_lower.real()),
            DisplayFloat(l_lower.imag()),
        );
    }

    let debug_phase = cnum::verbose();
    let phase_start = std::time::Instant::now();

    // Helper: build + symmetrize an initial guess, with an optional target_mid
    // override (for multi-start retries when the default cap fails).
    let make_initial = |target_override: Option<&Float>| -> Vec<Complex> {
        let mut init = initial_guess_with_target(
            &nodes,
            b,
            &l_upper,
            &l_lower,
            &arg_lambda,
            prec,
            target_override,
        );
        init[0] = l_lower.clone();
        init[n_nodes - 1] = l_upper.clone();
        if use_schwarz {
            symmetrize_schwarz(&mut init, prec);
        }
        init
    };

    // Warm start: if the caller supplied a pointwise guess F(0.5+it), try the
    // LM solve from it before falling back to the cold tanh/sech bridge. Used
    // by the cut-base path, where the cold guess sits outside the Newton
    // basin but a Schröder solve at a nearby ST-interior base is available.
    let warm_result: Option<Result<(Vec<Complex>, Float), String>> = warm_guess.map(|wg| {
        let evaluate = |t: &Float| {
            wg(t).map_err(|e| format!("warm guess eval failed at t={}: {e}", DisplayFloat(t)))
        };
        let winit_res: Result<Vec<Complex>, String> = if crate::mt::mt_enabled() {
            use rayon::prelude::*;
            nodes.par_iter().map(evaluate).collect()
        } else {
            nodes.iter().map(evaluate).collect()
        };
        let mut winit = winit_res?;
        winit[0] = l_lower.clone();
        winit[n_nodes - 1] = l_upper.clone();
        if use_schwarz {
            symmetrize_schwarz(&mut winit, prec);
        }
        if debug_phase {
            eprintln!(
                "kouz phase: warm-start guess built ({} nodes, {:.2}s)",
                n_nodes,
                phase_start.elapsed().as_secs_f64()
            );
        }
        iterate_newton(
            winit,
            &nodes,
            &weights,
            &t_max,
            &l_upper,
            &l_lower,
            &ln_b,
            prec,
            digits,
            use_schwarz,
            two_sided,
        )
    });

    let lm_result = match warm_result {
        Some(Ok(s)) => Ok(s),
        warm_outcome => {
            if let Some(Err(ref e)) = warm_outcome {
                if warm_only {
                    return Err(format!("Kouznetsov warm-only solve failed: {}", e));
                }
                if debug_phase {
                    eprintln!(
                        "kouz: warm start failed ({}); falling back to cold guess",
                        e
                    );
                }
            }
            let initial = make_initial(None);
            if debug_phase {
                eprintln!(
                    "kouz phase: initial_guess done ({:.2}s)",
                    phase_start.elapsed().as_secs_f64()
                );
            }
            // Default solver: Levenberg-Marquardt Newton-Kantorovich. Newton's
            // Jacobian-based step finds the right descent direction even when the
            // Cauchy operator T has spectral radius > 1 (which it does for typical
            // real bases b > e^(1/e)), where Picard / Anderson without history
            // diverge. The two non-default solvers stay around as debugging aids:
            //   * `TET_KOUZ_ANDERSON=1`: Anderson-accelerated Picard (works when T
            //     is a contraction, fails for higher b).
            //   * `TET_KOUZ_PICARD=1`: damped Picard, useful for spectrum diagnosis.
            if std::env::var_os("TET_KOUZ_ANDERSON").is_some() {
                iterate_anderson(
                    initial,
                    &nodes,
                    &weights,
                    &t_max,
                    &l_upper,
                    &l_lower,
                    &ln_b,
                    prec,
                    digits,
                    use_schwarz,
                    two_sided,
                )
            } else if std::env::var_os("TET_KOUZ_PICARD").is_some() {
                iterate_picard(
                    initial,
                    &nodes,
                    &weights,
                    &t_max,
                    &l_upper,
                    &l_lower,
                    &ln_b,
                    prec,
                    digits,
                    use_schwarz,
                    two_sided,
                )
            } else {
                iterate_newton(
                    initial,
                    &nodes,
                    &weights,
                    &t_max,
                    &l_upper,
                    &l_lower,
                    &ln_b,
                    prec,
                    digits,
                    use_schwarz,
                    two_sided,
                )
            }
        }
    };

    // Multi-start: if the default initial guess fails to converge (LM "no
    // descent step" — typical for bases like b≈5 where the converged F̃[mid]≈0.73
    // is well below the default cap of 1.5), retry with alternative target_mid
    // values. Two retries cover the b∈[η,e²] gap.
    let (samples, achieved_residual) = match lm_result {
        Ok(s) => s,
        Err(ref e)
            if e.contains("no descent step")
                || e.contains("no convergence")
                || e.contains("stagnation") =>
        {
            if debug_phase {
                eprintln!("kouz: LM failed with default guess ({}); retrying with alternative initial guesses", e);
            }
            let retry_targets = ["0.75", "1.1", "0.5", "1.25", "0.9", "1.4"];
            let mut last_err = e.clone();
            let mut found = None;
            for target in retry_targets {
                let target = cnum::decimal(target, prec);
                if debug_phase {
                    eprintln!("kouz: retry with target_mid={}", DisplayFloat(&target));
                }
                let retry_init = make_initial(Some(&target));
                match iterate_newton(
                    retry_init,
                    &nodes,
                    &weights,
                    &t_max,
                    &l_upper,
                    &l_lower,
                    &ln_b,
                    prec,
                    digits,
                    use_schwarz,
                    two_sided,
                ) {
                    Ok(s) => {
                        found = Some(s);
                        break;
                    }
                    Err(e2) => {
                        last_err = e2;
                    }
                }
            }
            found.ok_or(last_err)?
        }
        Err(e) => return Err(e),
    };
    if debug_phase {
        eprintln!(
            "kouz phase: iterate_newton done ({:.2}s elapsed since symmetrize)",
            phase_start.elapsed().as_secs_f64()
        );
    }
    let phase_start = std::time::Instant::now();

    // The functional equation `F(z+1) = b^F(z)` plus the boundary conditions
    // `F → L_upper / L_lower` are invariant under any horizontal shift c ∈ ℝ:
    // if F is a solution, so is F(·+c). The discretized iteration therefore
    // has a one-parameter family of fixed points; whichever one we converge to
    // depends on the initial guess. The *natural* Kouznetsov F is pinned by
    // the additional condition F(0) = 1.
    //
    // After Anderson converges to *some* F in the family, find the shift δ
    // such that F(δ) = 1, then for user height h evaluate F(h + δ). That maps
    // our (arbitrary-phase) F onto the natural F̃ via F̃(h) = F(h + δ).
    let shift = if skip_norm {
        Complex::new_64(prec)
    } else {
        find_normalization_shift(
            &samples,
            &nodes,
            &weights,
            &t_max,
            &l_upper,
            &l_lower,
            &ln_b,
            prec,
            digits,
            use_schwarz,
            two_sided,
        )?
    };
    if debug_phase && !skip_norm {
        eprintln!(
            "kouz phase: find_normalization_shift done ({:.2}s)",
            phase_start.elapsed().as_secs_f64()
        );
        eprintln!(
            "kouz normalization shift δ = {:.6e} + {:.6e}i (such that F(δ)=1)",
            DisplayFloat(shift.real()),
            DisplayFloat(shift.imag()),
        );
    }
    Ok(KouznetsovState {
        samples,
        nodes,
        weights,
        t_max,
        l_upper,
        l_lower,
        ln_b,
        shift,
        prec,
        digits,
        normalized: !skip_norm,
        residual: achieved_residual,
        two_sided,
    })
}

/// Evaluate from cached samples, refining internal accuracy when height
/// conditioning consumes the requested-digit margin.
///
/// The functional-equation residual is a consistency check, not an independent
/// error bound: both sides use the same reconstruction and recurrence.
pub fn eval_kouznetsov(
    state: &KouznetsovState,
    b: &Complex,
    h: &Complex,
) -> Result<Complex, String> {
    eval_kouznetsov_at_digits(state, b, h, state.digits)
}

/// Reuse a validated state with a fixed, possibly lower, output-digit goal.
pub fn eval_kouznetsov_at_digits(
    state: &KouznetsovState,
    b: &Complex,
    h: &Complex,
    requested_digits: u64,
) -> Result<Complex, String> {
    cnum::checked_digits_to_bits(requested_digits)?;
    let mut refined = None;
    loop {
        let current = refined.as_ref().unwrap_or(state);
        let (value, log_amplification) = eval_kouznetsov_once(current, b, h)?;
        let Some(digits) =
            height_precision_digits(requested_digits, current.digits, &log_amplification)?
        else {
            return Ok(value);
        };
        if cnum::verbose() {
            eprintln!(
                "kouz height conditioning: refining from {} to {digits} internal digits for {} output digits",
                current.digits, requested_digits
            );
        }
        refined = Some(refine_kouznetsov_precision(current, b, digits)?);
    }
}

fn height_precision_digits(
    requested: u64,
    current: u64,
    log_amplification: &Float,
) -> Result<Option<u64>, String> {
    let prec = log_amplification.prec_64();
    if log_amplification.is_infinite() && log_amplification.is_sign_positive() {
        let next = current
            .checked_mul(2)
            .ok_or("unresolved numerical zero at the supported precision limit")?
            .max(requested);
        cnum::checked_digits_to_bits(next)?;
        return Ok(Some(next));
    }
    let lost = (log_amplification.clone() / Float::with_val_64(prec, 10).ln()).ceil();
    if !lost.is_finite() || lost < 0 || lost > u64::MAX {
        return Err("height conditioning requires more than MPFR's native precision".into());
    }
    let lost = lost
        .to_integer()
        .and_then(|value| value.to_u64())
        .ok_or("height conditioning exceeds the supported precision count")?;
    let required = requested
        .checked_add(lost)
        .ok_or("height conditioning exceeds the supported precision count")?;
    if required <= current.saturating_add(3) {
        return Ok(None);
    }
    cnum::checked_digits_to_bits(required)?;
    Ok(Some(required))
}

fn refine_kouznetsov_precision(
    state: &KouznetsovState,
    b: &Complex,
    digits: u64,
) -> Result<KouznetsovState, String> {
    let prec = cnum::checked_digits_to_bits(digits)?.max(state.prec);
    let ln_b = cnum::ln_complex(b, prec);
    let l_upper = newton_fixed_point(&ln_b, &Complex::with_val_64(prec, &state.l_upper), prec)?;
    let l_lower = newton_fixed_point(&ln_b, &Complex::with_val_64(prec, &state.l_lower), prec)?;
    let use_schwarz = is_real_positive(b)
        && state.l_upper == Complex::with_val_64(state.prec, state.l_lower.conj_ref());
    let old_nodes = pick_node_count(state.digits, &state.t_max, state.prec)?;
    let node_boost = state.samples.len().div_ceil(old_nodes).max(1);
    let mut refined = setup_kouznetsov_core(
        b,
        l_upper,
        l_lower,
        prec,
        digits,
        use_schwarz,
        None,
        false,
        true,
        state.two_sided,
        node_boost,
    )?;
    let target = cnum::epsilon(digits.saturating_add(3), prec);
    if !refined.residual.is_finite() || refined.residual > target {
        return Err(format!(
            "Kouznetsov precision refinement boundary residual {} exceeds target {}",
            DisplayFloat(&refined.residual),
            DisplayFloat(&target)
        ));
    }
    refined.shift = find_normalization_shift(
        &refined.samples,
        &refined.nodes,
        &refined.weights,
        &refined.t_max,
        &refined.l_upper,
        &refined.l_lower,
        &refined.ln_b,
        prec,
        digits,
        use_schwarz,
        refined.two_sided,
    )?;
    refined.normalized = true;
    Ok(refined)
}

fn eval_kouznetsov_once(
    state: &KouznetsovState,
    b: &Complex,
    h: &Complex,
) -> Result<(Complex, Float), String> {
    cnum::init_mpfr();
    let prec = state.prec;
    if !cnum::is_finite(b) || !cnum::is_finite(h) || !state.normalized {
        return Err("Kouznetsov evaluation requires finite inputs and a normalized state".into());
    }
    if state.ln_b != cnum::ln_complex(b, prec) {
        return Err("Kouznetsov base does not match the cached state".into());
    }
    let required_residual = cnum::epsilon(state.digits.saturating_add(3), prec);
    if !state.residual.is_finite() || state.residual > required_residual {
        return Err(format!(
            "Kouznetsov boundary residual {} exceeds requested target {}; candidate state is not an answer",
            DisplayFloat(&state.residual), DisplayFloat(&required_residual)
        ));
    }
    if state.samples.len() < 3
        || state.samples.len() != state.nodes.len()
        || state.samples.len() != state.weights.len()
        || !state.samples.iter().all(cnum::is_finite)
        || !state
            .nodes
            .iter()
            .chain(&state.weights)
            .all(Float::is_finite)
        || !state.t_max.is_finite()
        || state.t_max <= 0
        || !cnum::is_finite(&state.ln_b)
        || cnum::is_zero(&state.ln_b)
    {
        return Err("Kouznetsov state has invalid samples or contour geometry".into());
    }
    if h.imag().is_zero() && h.real().is_integer() && *h.real() <= -2 {
        return Err(format!(
            "integer height {} is undefined for tetration (would require log_b(0) and beyond)",
            DisplayFloat(h.real())
        ));
    }
    if h.imag().is_zero() && *h.real() == -1 {
        return Ok((cnum::zero(prec), Float::new_64(prec)));
    }
    let h_shifted = Complex::with_val_64(prec, h + &state.shift);
    let (f_h, log_amplification) = eval_at_height_with_conditioning(
        &h_shifted,
        &state.samples,
        &state.nodes,
        &state.weights,
        &state.t_max,
        &state.l_upper,
        &state.l_lower,
        &state.ln_b,
        prec,
        state.two_sided,
    )?;

    // A finite value can have an overflowing successor; prefer its predecessor.
    let forward = cnum::is_zero(&f_h);
    let h_adjacent = Complex::with_val_64(prec, &h_shifted + if forward { 1 } else { -1 });
    let f_adjacent = eval_at_height(
        &h_adjacent,
        &state.samples,
        &state.nodes,
        &state.weights,
        &state.t_max,
        &state.l_upper,
        &state.l_lower,
        &state.ln_b,
        prec,
        state.two_sided,
    )?;
    let (before, after) = if forward {
        (&f_h, &f_adjacent)
    } else {
        (&f_adjacent, &f_h)
    };
    let exp_arg = Complex::with_val_64(prec, before * &state.ln_b);
    let b_pow_f = cnum::checked_exp(&exp_arg, prec)?;
    let diff = Complex::with_val_64(prec, after - &b_pow_f);
    let scale = cnum::abs(after, prec).max(&Float::with_val_64(prec, 1));
    let rel = cnum::abs(&diff, prec) / scale;
    let tolerance = cnum::epsilon(state.digits, prec);
    if !cnum::is_finite(&f_h) || !rel.is_finite() || rel > tolerance {
        return Err(format!(
            "Kouznetsov functional-equation residual {} exceeds requested tolerance {}",
            DisplayFloat(&rel),
            DisplayFloat(&tolerance)
        ));
    }
    if cnum::verbose() {
        eprintln!(
            "kouz eval: functional-equation relative residual {} ({} step)",
            DisplayFloat(&rel),
            if forward { "forward" } else { "backward" }
        );
    }
    Ok((f_h, log_amplification))
}

/// Normalize the same integer-extended reconstruction used for returned values.
#[allow(clippy::too_many_arguments)]
fn find_normalization_shift(
    samples: &[Complex],
    nodes: &[Float],
    weights: &[Float],
    t_max: &Float,
    l_upper: &Complex,
    l_lower: &Complex,
    ln_b: &Complex,
    prec: u64,
    digits: u64,
    real_axis: bool,
    two_sided: bool,
) -> Result<Complex, String> {
    let one = Complex::with_val_64(prec, (1u32, 0));
    let two = Float::with_val_64(prec, 2u32);
    // Cube-root rule for central FD: ε ≈ δ_f^(1/3) where δ_f is the working
    // precision. With δ_f ≈ 10^(-digits), ε ≈ 10^(-digits/3 - 3).
    let eps_f = cnum::epsilon(digits.saturating_add(8) / 3, prec);
    let eps = Complex::with_val_64(prec, (eps_f.clone(), 0));
    let two_eps = Complex::with_val_64(prec, (Float::with_val_64(prec, &eps_f * &two), 0));

    let debug_norm = cnum::verbose();
    let evaluate = |h: &Complex| {
        eval_at_height(
            h, samples, nodes, weights, t_max, l_upper, l_lower, ln_b, prec, two_sided,
        )
    };

    // For real-positive bases the natural Kneser F satisfies Schwarz reflection
    // F(z̄)=F̄(z), so F̃ is real on the real axis and the normalization shift δ
    // (the unique solution of F̃(δ)=1) MUST be real — otherwise F(real h)=F̃(real
    // h+δ) acquires a spurious imaginary part and is no longer the canonical
    // real-on-real tetration. The Newton solve runs in complex arithmetic and a
    // complex root chosen by the grid search (conjugate pairs share |c|) would
    // silently break this. `finalize` projects the converged δ onto the real
    // axis when `real_axis` is set; the grid search below is also restricted to
    // real seeds in that case so the projected root genuinely satisfies F̃(δ)=1.
    let target = cnum::epsilon(digits.saturating_add(3), prec);
    let finalize = |c: Complex| -> Result<Complex, String> {
        let result = if real_axis {
            Complex::with_val_64(
                prec,
                (Float::with_val_64(prec, c.real()), Float::new_64(prec)),
            )
        } else {
            c
        };
        let value = evaluate(&result)?;
        let residual = cnum::abs(&Complex::with_val_64(prec, value - &one), prec);
        if !cnum::is_finite(&result) || !residual.is_finite() || residual > target {
            return Err(format!(
                "normalization of returned shift failed: residual {}",
                DisplayFloat(&residual)
            ));
        }
        Ok(result)
    };

    let try_newton = |seed: &Complex| -> Result<(Complex, Float), String> {
        let mut c = seed.clone();
        let mut best_resid = Float::with_val_64(prec, rug::float::Special::Infinity);
        let mut checkpoint = c.clone();
        let mut iter = Integer::new();
        loop {
            let f_c = evaluate(&c)?;
            let resid = Complex::with_val_64(prec, &f_c - &one);
            let resid_abs = cnum::abs(&resid, prec);
            if !resid_abs.is_finite() {
                return Err("non-finite".into());
            }
            if resid_abs < best_resid {
                best_resid = resid_abs.clone();
            }
            if resid_abs < target {
                return Ok((finalize(c)?, resid_abs));
            }
            let c_plus = Complex::with_val_64(prec, &c + &eps);
            let c_minus = Complex::with_val_64(prec, &c - &eps);
            let f_plus = evaluate(&c_plus)?;
            let f_minus = evaluate(&c_minus)?;
            let diff = Complex::with_val_64(prec, &f_plus - &f_minus);
            let derivative = Complex::with_val_64(prec, &diff / &two_eps);
            if !cnum::is_finite(&derivative) || cnum::is_zero(&derivative) {
                return Err("normalization derivative is zero or non-finite".into());
            }
            let step = Complex::with_val_64(prec, &resid / &derivative);
            c = Complex::with_val_64(prec, &c - &step);
            if c == checkpoint {
                break;
            }
            if iter.is_power_of_two() {
                checkpoint = c.clone();
            }
            if debug_norm {
                eprintln!(
                    "kouz normalization iter {iter}: residual {:.3e}",
                    DisplayFloat(&resid_abs)
                );
            }
            iter += 1;
        }
        Err(format!(
            "normalization did not converge: best residual {}",
            DisplayFloat(&best_resid)
        ))
    };

    // Fast path: Newton from c=0. Real positive bases land here; F̃(0) is
    // already very close to 1 and Newton converges in 4-6 iters.
    if let Ok((c0, _r0)) = try_newton(&cnum::zero(prec)) {
        if debug_norm {
            eprintln!(
                "kouz norm: Newton from c=0 converged to ({:.6},{:.6}i)",
                DisplayFloat(c0.real()),
                DisplayFloat(c0.imag()),
            );
        }
        return Ok(c0);
    }

    // Slower path: grid search + Newton from EACH seed. Collect every
    // converged root, then pick the one with smallest |c_root| (with a
    // deterministic tiebreak on Im, then Re). Picking by |c_root| (the
    // converged value) rather than |c_seed| makes the choice precision-
    // independent: Newton's basin maps each seed to a structural root of
    // F̃(c)=1, and those roots don't move with N — only WHICH seeds map to
    // WHICH roots may shift slightly, but the set of reachable roots stays
    // the same. So we collect them all and pick the smallest.
    let re_steps = [0i32, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6];
    // Real-positive bases: only seed on the real axis so the converged root is
    // real (δ must be real for Schwarz canonicality — see `finalize`).
    let im_steps: &[i32] = if real_axis {
        &[0]
    } else {
        &[0, 1, -1, 2, -2, 4, -4]
    };

    #[derive(Clone)]
    struct Root {
        c: Complex,
        re: Float,
        im: Float,
        abs: Float,
    }
    let mut roots: Vec<Root> = Vec::new();
    for &cr in &re_steps {
        for &ci in im_steps {
            let seed = Complex::with_val_64(
                prec,
                (
                    Float::with_val_64(prec, cr) / 4,
                    Float::with_val_64(prec, ci) / 4,
                ),
            );
            match try_newton(&seed) {
                Ok((c_root, _r)) => {
                    let re = c_root.real().clone();
                    let im = c_root.imag().clone();
                    let abs = cnum::abs(&c_root, prec);
                    roots.push(Root {
                        c: c_root,
                        re,
                        im,
                        abs,
                    });
                }
                Err(why) => {
                    if debug_norm {
                        eprintln!("kouz norm: seed {} failed: {}", DisplayComplex(&seed), why);
                    }
                }
            }
        }
    }

    if !roots.is_empty() {
        roots.sort_by(|a, b| {
            a.abs
                .total_cmp(&b.abs)
                .then(a.im.total_cmp(&b.im))
                .then(a.re.total_cmp(&b.re))
        });
        let chosen = roots.into_iter().next().unwrap();
        if debug_norm {
            eprintln!(
                "kouz norm: chose smallest-|c| Newton root ({:.6},{:.6}i) |c|={:.3e}",
                DisplayFloat(&chosen.re),
                DisplayFloat(&chosen.im),
                DisplayFloat(&chosen.abs),
            );
        }
        return Ok(chosen.c);
    }

    if debug_norm {
        // Diagnostic landscape: F̃ on a coarse c-grid, to distinguish
        // "search box too small" from "spurious operator fixed point".
        eprintln!("kouz norm: FAILURE landscape F̃(c):");
        for ci in [-2i32, -1, 0, 1, 2] {
            for cr in [-6i32, -3, 0, 3, 6] {
                let c = Complex::with_val_64(
                    prec,
                    (
                        Float::with_val_64(prec, cr) / 4,
                        Float::with_val_64(prec, ci),
                    ),
                );
                match evaluate(&c) {
                    Ok(value) => eprintln!(
                        "  F({}) = Ok({})",
                        DisplayComplex(&c),
                        DisplayComplex(&value)
                    ),
                    Err(error) => eprintln!("  F({}) = Err({error:?})", DisplayComplex(&c)),
                }
            }
        }
    }
    Err(format!(
        "Kouznetsov normalization: no grid seed produced Newton-converged \
         root of F̃(c)=1 with target {:.3e}; no normalized value can be returned \
         by this solve. This is not proof that a mathematical solution is absent.",
        DisplayFloat(&target)
    ))
}

/// True iff `b` is a real positive number (Im(b)=0, Re(b)>0). Real positive
/// bases admit Schwarz reflection symmetry `F(z̄) = F̄(z)`; everything else
/// (real negative, imaginary, general complex) does not.
fn is_real_positive(b: &Complex) -> bool {
    b.imag().is_zero() && b.real().is_sign_positive() && !b.real().is_zero()
}

/// Find a fixed point of `b^z = z` by Newton iteration starting from `seed`.
///
/// Used to locate `L_lower` for non-real bases by starting from
/// `conj(L_upper)`. For real bases, conj(L_upper) is exact (zero iterations).
/// For slightly-complex bases the conjugate is close to a true fixed point
/// and Newton converges quickly; this is the natural analytic continuation
/// of the real case and avoids the W₋₁ branch-cut discontinuity that
/// `-W₋₁(-ln b)/ln b` exhibits for slightly-off-real bases.
///
/// f(z)  = b^z - z
/// f'(z) = b^z · ln_b - 1
pub(crate) fn newton_fixed_point(
    ln_b: &Complex,
    seed: &Complex,
    prec: u64,
) -> Result<Complex, String> {
    let one = Complex::with_val_64(prec, (1, 0));
    // Target |f| < 2^-(prec - 16); leave a small guard so the loop terminates.
    let target_tol = Float::with_val_64(prec, 1)
        >> usize::try_from(prec.saturating_sub(16)).expect("precision exceeds addressable bits");
    if !cnum::is_finite(ln_b) || !cnum::is_finite(seed) {
        return Err("Newton fixed-point requires finite inputs".into());
    }
    let mut z = seed.clone();
    let mut checkpoint = z.clone();
    let mut iter = Integer::new();
    loop {
        let arg = Complex::with_val_64(prec, &z * ln_b);
        let bz = cnum::checked_exp(&arg, prec)?;
        let f = Complex::with_val_64(prec, &bz - &z);
        let fabs = Float::with_val_64(prec, f.abs_ref());
        if fabs.is_finite() && fabs < target_tol {
            return Ok(z);
        }
        let fp = {
            let t = Complex::with_val_64(prec, &bz * ln_b);
            Complex::with_val_64(prec, &t - &one)
        };
        if !cnum::is_finite(&fp) || cnum::is_zero(&fp) {
            return Err("Newton fixed-point: derivative is zero or non-finite".into());
        }
        let delta = Complex::with_val_64(prec, &f / &fp);
        z -= &delta;
        if z == checkpoint {
            return Err("Newton fixed-point stagnation: repeated iterate".into());
        }
        if iter.is_power_of_two() {
            checkpoint = z.clone();
        }
        iter += 1;
    }
}

fn arg_abs(z: &Complex, prec: u64) -> Float {
    Float::with_val_64(prec, z.arg_ref()).abs()
}

fn slowest_decay_rate(upper: &Complex, lower: &Complex, prec: u64) -> Float {
    arg_abs(upper, prec).min(&arg_abs(lower, prec))
}

fn contour_height(digits: u64, arg_lambda: &Float, prec: u64) -> Result<Float, String> {
    if !arg_lambda.is_finite() || *arg_lambda <= 0 {
        return Err(format!(
            "Kouznetsov: |arg(lambda)|={} gives a degenerate contour",
            DisplayFloat(arg_lambda)
        ));
    }
    let height = (Float::with_val_64(prec, digits.saturating_add(8))
        * Float::with_val_64(prec, 10).ln()
        / arg_lambda)
        .max(&Float::with_val_64(prec, 8));
    if !height.is_finite() {
        return Err("Kouznetsov contour size exceeded the exponent range".into());
    }
    Ok(height)
}

/// Trapezoidal-rule node count for the Kouznetsov rectangle.
///
/// Two error sources contribute:
/// * Boundary mismatch at `t=±T` (the integrand isn't zero there): plain
///   trapezoidal gives `O(h²)`, killed by Euler-Maclaurin (see
///   `compute_em_correction_z0`). Negligible after EM.
/// * Bulk error on the analytic integrand: spectrally, `~ exp(-2π·σ/h)`
///   where `σ` is the analyticity-strip half-width of the integrand (viewed
///   as a function of real `T_var`). Empirically `σ_eff ≈ 0.30` for the
///   left-edge log integrand `log_b(F(0.5+iT_var))` — bounded by F's branch
///   structure off the midline. So bulk error `~ exp(−1.88·N/T)` for typical
///   real bases > η, and reaching `10^{−(digits+5)}` needs
///   `N ≥ (digits+5)·ln(10)·T / (π·σ_eff)`.
///
/// We round up and apply a 1.2× margin. We do NOT divide by `|arg(λ)|` like
/// the previous formula did — the analyticity-strip geometry doesn't depend
/// on it, and the t_max selection already absorbs `arg(λ)`'s effect on
/// contour height.
fn pick_node_count(digits: u64, t_max: &Float, prec: u64) -> Result<usize, String> {
    let required = (Float::with_val_64(prec, digits.saturating_add(5))
        * Float::with_val_64(prec, 10).ln()
        * t_max
        * 4u32
        / Float::with_val_64(prec, Constant::Pi))
    .ceil();
    let n_bulk = cnum::checked_usize(&required)
        .ok_or("Kouznetsov required node count exceeds addressable memory")?;
    let clamped = n_bulk.max(80);
    // FFT-friendly snap. The cross-correlation pads to next_power_of_two(2N−1).
    // For N in [2^(k−1)+1, 2^k], that padded length is constant at 2^(k+1).
    // Snapping N up to 2^k (the right edge of its bucket) gives maximal
    // trapezoidal accuracy at zero extra FFT cost. `next_power_of_two(2^k)`
    // returns 2^k itself, so this is a no-op when N is already a power of two.
    clamped
        .checked_next_power_of_two()
        .ok_or("Kouznetsov FFT node count overflow".into())
}

fn build_uniform_nodes(t_max: &Float, n: usize, prec: u64) -> Vec<Float> {
    let two_t = Float::with_val_64(prec, t_max * 2u32);
    let delta = Float::with_val_64(prec, &two_t / (n - 1));
    (0..n)
        .map(|k| {
            Float::with_val_64(
                prec,
                -(t_max.clone()) + Float::with_val_64(prec, &delta * k),
            )
        })
        .collect()
}

/// Enforce Schwarz reflection symmetry on a sample vector: F at node `t_k` and
/// at node `t_{n-1-k}` (i.e. `−t_k` since the grid is symmetric about 0) must
/// be conjugates. We average each (k, n-1-k) pair toward the symmetric subspace.
///
/// The natural Kouznetsov F satisfies `F(z̄) = F̄(z)` on the strip — both the
/// boundary conditions and the right/left-edge functional equations preserve
/// this. But finite-precision LM steps can break it; symmetrizing each
/// iterate keeps the iteration on the symmetric manifold (halves the effective
/// problem dimension and prevents asymmetric drift modes from growing).
fn symmetrize_schwarz(samples: &mut [Complex], prec: u64) {
    let n = samples.len();
    let half = cnum::decimal("0.5", prec);
    for k in 0..n / 2 {
        let m = n - 1 - k;
        let re_avg = Float::with_val_64(
            prec,
            (Float::with_val_64(prec, samples[k].real() + samples[m].real())) * &half,
        );
        let im_avg = Float::with_val_64(
            prec,
            (Float::with_val_64(prec, samples[k].imag() - samples[m].imag())) * &half,
        );
        let neg_im = Float::with_val_64(prec, -&im_avg);
        samples[k] = Complex::with_val_64(prec, (re_avg.clone(), im_avg));
        samples[m] = Complex::with_val_64(prec, (re_avg, neg_im));
    }
    // Middle node (if n is odd) must have purely real F.
    if n % 2 == 1 {
        let mid = n / 2;
        let re = Float::with_val_64(prec, samples[mid].real());
        samples[mid] = Complex::with_val_64(prec, (re, Float::new_64(prec)));
    }
}

fn build_trapezoidal_weights(t_max: &Float, n: usize, prec: u64) -> Vec<Float> {
    let two_t = Float::with_val_64(prec, t_max * 2u32);
    let delta = Float::with_val_64(prec, &two_t / (n - 1));
    let half_delta = Float::with_val_64(prec, &delta / 2u32);
    let mut w = vec![delta.clone(); n];
    w[0] = half_delta.clone();
    w[n - 1] = half_delta;
    w
}

// =====================================================================
// Euler-Maclaurin boundary correction for trapezoidal-rule Cauchy integrals.
//
// The integrands `g_R(t) = b^F(0.5+it)/(1.5+it−z₀)` and
// `g_L(t) = log_b(F(0.5+it))/(−0.5+it−z₀)` are NOT zero at the contour
// truncation `t=±T`: their values are L_{up/down}/(c±iT−z₀), only `1/T`
// small. So plain trapezoidal has O(h²) error and quickly hits a floor
// far above requested precision (e.g. ~6e−6 for digits=15 with N=2635).
//
// Euler-Maclaurin says
//   ∫_{-T}^{T} g(t) dt = T_n − Σ_{k=1}^∞ B_{2k}/(2k)! · h^{2k}
//                                  · [g^{(2k−1)}(T) − g^{(2k−1)}(−T)].
// With `F(±T) = L_{up/down}` (boundary pinning) and `F^{(j)}(±T)` below the
// precision floor (asymptote reached up to `exp(−|arg(λ)|·T)`), each
// derivative is closed-form
//   g^{(j)}(T)  ≈ L_+ · (−i)^j · j! / (c+iT)^{j+1}.
// The correction simplifies to
//   corr = −i · Σ_k |B_{2k}|/(2k) · h^{2k}
//                · [L_+ /(c+iT)^{2k} − L_− /(c−iT)^{2k}].
// Near contour ends the denominator is O(1), not O(T). Spacing and the
// evaluation strip's clearance determine the useful asymptotic order.
// =====================================================================

fn em_coefficients(n_terms: usize) -> Result<Arc<Vec<Rational>>, String> {
    #[derive(Default)]
    struct Cache {
        row: Vec<Rational>,
        coefficients: Arc<Vec<Rational>>,
    }
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    let size = n_terms
        .checked_mul(2)
        .and_then(|v| v.checked_add(1))
        .ok_or("Euler-Maclaurin coefficient count exceeds addressable memory")?;
    std::alloc::Layout::array::<Rational>(size)
        .map_err(|_| "Euler-Maclaurin coefficients exceed addressable memory")?;
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "Euler-Maclaurin coefficient cache is poisoned")?;
    while cache.coefficients.len() < n_terms {
        let m = cache.row.len();
        cache.row.push(Rational::from((1, m + 1)));
        // Akiyama-Tanigawa, with exact rationals throughout.
        for j in (1..=m).rev() {
            cache.row[j - 1] = Rational::from(&cache.row[j - 1] - &cache.row[j]) * j;
        }
        if m != 0 && m % 2 == 0 {
            let coefficient = cache.row[0].clone().abs() / m;
            Arc::make_mut(&mut cache.coefficients).push(coefficient);
        }
    }
    Ok(Arc::clone(&cache.coefficients))
}

/// Choose a spacing-aware EM order before its asymptotic remainder estimate grows.
/// The estimate covers horizontal clearance >= 1/2 throughout the evaluation strip.
fn em_n_terms(
    prec: u64,
    spacing: &Float,
    l_upper: &Complex,
    l_lower: &Complex,
) -> Result<usize, String> {
    if std::env::var_os("TET_KOUZ_NO_EM").is_some() {
        return Ok(0);
    }
    if std::env::var_os("TET_KOUZ_EM_K").is_some() {
        let n = cnum::env_usize("TET_KOUZ_EM_K", 0)?;
        cnum::check_complex_storage(n as u128, prec)
            .map_err(|e| format!("Euler-Maclaurin order (TET_KOUZ_EM_K): {e}"))?;
        return Ok(n);
    }
    if !spacing.is_finite()
        || *spacing <= 0
        || !cnum::is_finite(l_upper)
        || !cnum::is_finite(l_lower)
    {
        return Err("Euler-Maclaurin order requires positive spacing and finite boundaries".into());
    }
    let log_step = (spacing.clone().ln() - Float::with_val_64(prec, Constant::Pi).ln()) * 2;
    let log_scale = log_magnitude(l_upper, prec)
        .max(&log_magnitude(l_lower, prec))
        .max(&Float::new_64(prec));
    // Constant-kernel estimate: 8*max(1, |L±|)*(2k)!*(h/pi)^(2k).
    let mut log_bound = log_scale + Float::with_val_64(prec, 16).ln() + &log_step;
    let log_target = cnum::working_epsilon(prec).ln();
    let mut n = 1usize;
    while log_bound > log_target {
        let two_n = Integer::from(n) * 2;
        let log_ratio = Float::with_val_64(prec, Integer::from(&two_n + 1)).ln()
            + Float::with_val_64(prec, Integer::from(&two_n + 2)).ln()
            + &log_step;
        if log_ratio >= 0 {
            if cnum::verbose() {
                eprintln!(
                    "kouz EM: working-precision target is below this grid's optimal remainder estimate; \
                     using K={n}, log estimate {} versus log target {}",
                    DisplayFloat(&log_bound), DisplayFloat(&log_target)
                );
            }
            break;
        }
        log_bound += log_ratio;
        n = n
            .checked_add(1)
            .ok_or("Euler-Maclaurin order exceeds addressable memory")?;
    }
    cnum::check_complex_storage(n as u128, prec)
        .map_err(|e| format!("Euler-Maclaurin order: {e}"))?;
    Ok(n)
}

/// Pre-bake `[ -i · |B_{2k}|/(2k) · h^{2k} ]` for k=1..=K. The `-i` factor
/// (from `(−i)^{2k−1} = (−1)^k · i` combined with the alternating sign of
/// `B_{2k}`) simplifies to a uniform `-i` across all k.
fn build_em_h_powers(h: &Float, n_terms: usize, prec: u64) -> Result<Vec<Complex>, String> {
    if n_terms == 0 {
        return Ok(Vec::new());
    }
    cnum::check_complex_storage(n_terms as u128, prec)?;
    let coefficients = em_coefficients(n_terms)?;
    let neg_i = Complex::with_val_64(prec, (Float::new_64(prec), Float::with_val_64(prec, -1i32)));
    let h_sq = Float::with_val_64(prec, h * h);
    let mut h_pow = h_sq.clone();
    let mut out = Vec::with_capacity(n_terms);
    for (k, coefficient) in coefficients.iter().take(n_terms).enumerate() {
        let coef_real = Float::with_val_64(prec, coefficient);
        let scaled = Float::with_val_64(prec, &coef_real * &h_pow);
        if !scaled.is_finite() || scaled.is_zero() {
            return Err("Euler-Maclaurin coefficient exceeded MPFR's exponent range".into());
        }
        out.push(Complex::with_val_64(prec, &neg_i * &scaled));
        if k + 1 < n_terms {
            h_pow = Float::with_val_64(prec, &h_pow * &h_sq);
        }
    }
    Ok(out)
}

/// Closed-form Euler-Maclaurin correction for both edges at evaluation point
/// `z₀`. Caller subtracts the returned `(corr_R, corr_L)` from the trapezoidal
/// `r_int` / `l_int` to obtain integration error `O(h^{2K+2})` (vs `O(h²)` for
/// plain trapezoidal). `em_h_powers` is the pre-baked coefficient sequence
/// from `build_em_h_powers`.
fn compute_em_correction_z0(
    z0: &Complex,
    t_max: &Float,
    l_upper: &Complex,
    l_lower: &Complex,
    em_h_powers: &[Complex],
    prec: u64,
) -> (Complex, Complex) {
    if em_h_powers.is_empty() {
        return (cnum::zero(prec), cnum::zero(prec));
    }
    let cp1 = Complex::with_val_64(prec, (cnum::decimal("1.5", prec), 0));
    let cm1 = Complex::with_val_64(prec, (cnum::decimal("-0.5", prec), 0));
    let it_max = Complex::with_val_64(prec, (Float::new_64(prec), t_max.clone()));
    let neg_it_max = Complex::with_val_64(prec, -&it_max);

    // c±iT − z₀ for right edge (c=1.5) and left edge (c=−0.5).
    let c_r = Complex::with_val_64(prec, &cp1 - z0);
    let c_l = Complex::with_val_64(prec, &cm1 - z0);
    let cr_pos = Complex::with_val_64(prec, &c_r + &it_max);
    let cr_neg = Complex::with_val_64(prec, &c_r + &neg_it_max);
    let cl_pos = Complex::with_val_64(prec, &c_l + &it_max);
    let cl_neg = Complex::with_val_64(prec, &c_l + &neg_it_max);

    // (c±iT−z₀)² — incremental power update inside the loop multiplies by
    // these to advance from (c±iT−z₀)^{2k} to (c±iT−z₀)^{2(k+1)}.
    let cr_pos_sq = Complex::with_val_64(prec, &cr_pos * &cr_pos);
    let cr_neg_sq = Complex::with_val_64(prec, &cr_neg * &cr_neg);
    let cl_pos_sq = Complex::with_val_64(prec, &cl_pos * &cl_pos);
    let cl_neg_sq = Complex::with_val_64(prec, &cl_neg * &cl_neg);

    let one_c = Complex::with_val_64(prec, (Float::with_val_64(prec, 1u32), 0));
    let mut pr_pos = one_c.clone();
    let mut pr_neg = one_c.clone();
    let mut pl_pos = one_c.clone();
    let mut pl_neg = one_c;

    let mut corr_r = cnum::zero(prec);
    let mut corr_l = cnum::zero(prec);
    for coef in em_h_powers {
        pr_pos = Complex::with_val_64(prec, &pr_pos * &cr_pos_sq);
        pr_neg = Complex::with_val_64(prec, &pr_neg * &cr_neg_sq);
        pl_pos = Complex::with_val_64(prec, &pl_pos * &cl_pos_sq);
        pl_neg = Complex::with_val_64(prec, &pl_neg * &cl_neg_sq);

        let r_pos = Complex::with_val_64(prec, l_upper / &pr_pos);
        let r_neg = Complex::with_val_64(prec, l_lower / &pr_neg);
        let r_diff = Complex::with_val_64(prec, &r_pos - &r_neg);
        let r_term = Complex::with_val_64(prec, &r_diff * coef);
        corr_r = Complex::with_val_64(prec, &corr_r + &r_term);

        let l_pos = Complex::with_val_64(prec, l_upper / &pl_pos);
        let l_neg = Complex::with_val_64(prec, l_lower / &pl_neg);
        let l_diff = Complex::with_val_64(prec, &l_pos - &l_neg);
        let l_term = Complex::with_val_64(prec, &l_diff * coef);
        corr_l = Complex::with_val_64(prec, &corr_l + &l_term);
    }
    (corr_r, corr_l)
}

fn initial_guess_with_target(
    nodes: &[Float],
    b: &Complex,
    l_upper: &Complex,
    l_lower: &Complex,
    arg_lambda: &Float,
    prec: u64,
    target_mid_override: Option<&Float>,
) -> Vec<Complex> {
    // Smooth shape combining three pieces:
    //   * tanh-blend asymptote pinning F → L_upper as t→+∞ and F → L_lower as
    //     t→-∞. Slope `arg(λ)` matches the true exponential decay rate.
    //   * sech-bump correction at t=0 pushing F(0.5+0i) to a real central
    //     value `target_mid`. We pick `target_mid = min(√b, 1 + (b−1)·0.4)`
    //     so:
    //       - For bases near the boundary (b≈e) where F̃(0.5)≈√b, we use √b.
    //       - For larger bases (b≥4) where F̃(0.5) ≪ √b (because F̃ is highly
    //         convex), the second term dominates and keeps the initial guess
    //         in a plausibly-flat region of T. A guess like √10≈3.16 lands
    //         the iteration in a regime where b^F[mid] = 10^3.16 ≈ 1450
    //         dominates the Cauchy integrand and causes immediate divergence.
    //     Without this throttle, real bases b ≥ 4 fail to converge: T applied
    //     to the initial guess overshoots by orders of magnitude due to b^F's
    //     exponential amplification at right-edge samples.
    let half = cnum::decimal("0.5", prec);
    let one = Float::with_val_64(prec, 1u32);
    let rate = Float::with_val_64(prec, arg_lambda);
    let sqrt_b = Complex::with_val_64(prec, b.sqrt_ref());
    // Cap on |target_mid|. Empirically, the converged F̃[mid] = F̃(0+0i) for
    // the natural Kneser solution sits in [0.6, 1.4] across the profiled real-
    // positive bases (b=2 → 1.25, b=5 → 0.73, b=10 → 1.10, b=50 → 0.69,
    // b=100 → 1.40, b=200 → 1.30, b=500 → 0.70, b=1000 → 1.03). The published
    // F_b(0.5) values are different (F_b(0.5) = F̃(0.5+δ) with shift δ chosen
    // so F̃(δ)=1) and grow ~ln(b), but it's F̃[mid] that controls the initial
    // basin selection.
    //
    // Initial-guess sensitivity is real: cap=1.5 lets b≤200 converge but b≥500
    // descends to a wrong-basin attractor near F̃[mid]≈0; cap=1.0 lets b≥50
    // converge but flips b=10 into a different wrong basin (F̃[mid]≈0.09).
    // The Kneser basin's "attractor radius" along this axis shrinks as b grows.
    //
    // Smoothly decreasing cap from 1.5 (small b) to 0.7 (huge b) tracks the
    // boundary: cap = clamp(1.5 − 0.1·max(0, ln|b|−2), 0.7, 1.5). Anchor at
    // b=e^2 (no shrinkage), shrinks 0.1 per unit increase in ln|b| above 2.
    // We cap by magnitude (not real part) so complex bases — whose √b is
    // generically off the real axis — get a non-zero target.
    let sqrt_b_abs = cnum::abs(&sqrt_b, prec);
    let b_abs = cnum::abs(b, prec);
    let cap = (cnum::decimal("1.5", prec) - (b_abs.ln() - 2u32).max(&Float::new_64(prec)) / 10u32)
        .max(&cnum::decimal("0.7", prec))
        .min(&cnum::decimal("1.5", prec));
    let target_mid = if let Some(override_val) = target_mid_override {
        // Explicit override for retry attempts (multi-start strategy).
        let dir = Complex::with_val_64(prec, &sqrt_b / &sqrt_b_abs);
        Complex::with_val_64(prec, &dir * Float::with_val_64(prec, override_val))
    } else if sqrt_b_abs <= cap {
        sqrt_b.clone()
    } else {
        let scale = Float::with_val_64(prec, cap / sqrt_b_abs);
        Complex::with_val_64(prec, &sqrt_b * &scale)
    };
    let mid = {
        let two = Float::with_val_64(prec, 2u32);
        let sum = Complex::with_val_64(prec, l_upper + l_lower);
        Complex::with_val_64(prec, &sum / &two)
    };
    let bump_ampl = Complex::with_val_64(prec, &target_mid - &mid);

    nodes
        .iter()
        .map(|t| {
            let scaled_t = Float::with_val_64(prec, &rate * t);
            let tanh_t = Float::with_val_64(prec, scaled_t.tanh_ref());
            let w_upper =
                Float::with_val_64(prec, (Float::with_val_64(prec, &one + &tanh_t)) * &half);
            let w_lower = Float::with_val_64(prec, &one - &w_upper);
            let scaled_upper = Complex::with_val_64(prec, l_upper * &w_upper);
            let scaled_lower = Complex::with_val_64(prec, l_lower * &w_lower);
            let asymp = Complex::with_val_64(prec, &scaled_upper + &scaled_lower);
            let cosh_t = Float::with_val_64(prec, scaled_t.cosh_ref());
            let sech_t = Float::with_val_64(prec, &one / &cosh_t);
            let bump = Complex::with_val_64(prec, &bump_ampl * &sech_t);
            Complex::with_val_64(prec, &asymp + &bump)
        })
        .collect()
}

/// Branch-corrected `ln(samples[j])` along the mid-line sample curve.
///
/// The left rectangle edge represents `F(−0.5+it) = log_b F(0.5+it)`. The
/// true left-edge values are continuous in `t` and approach `L_up` at the
/// top, which forces the branch anchor `ln F → ln_b·L_up` (the fixed-point
/// identity `log_b L_up = L_up` in log form) and *continuous* branch
/// selection down the curve; the same argument at the bottom shows the true
/// curve's unwrapped log lands on `ln_b·L_low` automatically. A pointwise
/// principal log breaks continuity whenever the sample curve crosses
/// `(−∞, 0]` — which happens for cut-band bases, where `L_low` sits at
/// `Re < 0` and the curve spirals into it — silently kinking the discrete
/// operator by `2πi/ln_b` at the crossing nodes. LM then stalls on an
/// O(1e-2) residual plateau it cannot descend from (observed on the
/// cut-base ε-walk near ε ≈ 1.35). Unwrapping restores a continuous
/// operator; for curves that never cross the cut (all previously working
/// bases) it reduces to the principal log exactly.
///
/// **Two-sided anchoring**: the top half of the curve (`j ≥ n/2`) is
/// unwrapped walking *down* from the `ln_b·L_up` anchor; the bottom half
/// walking *up* from the `ln_b·L_low` anchor. Both asymptote tails are
/// therefore always branch-consistent with their own limits. For the
/// *true* solution continuity forces the two walks to agree at the joint
/// (net winding m = 0), so the result equals the single-anchor unwrap.
/// Transient LM iterates — notably the cold tanh-bridge guess — can sit
/// in a wrong homotopy class rel the origin (odd number of cut
/// crossings); then the two halves disagree by `2πi·m` at the *one fixed
/// interior joint node*, a localized integrand error from which LM
/// descends fine. The historical alternative — reverting the whole curve
/// to the pointwise principal log when m ≠ 0 — made the discrete operator
/// *discontinuous* in the samples (trial steps toggling the m
/// classification switch the operator wholesale), which near the
/// Shell–Thron crossing of the cut-base ε-walk (ε ≈ 1.55 at x = 0.04)
/// produced Jacobian/residual inconsistency and "no descent at iter 0"
/// from arbitrarily good warm starts. Two-sided anchoring keeps the
/// operator continuous in every homotopy class; the joint index is fixed
/// (n/2) so no data-dependent placement can reintroduce a discontinuity.
///
/// `two_sided=false` uses the pointwise principal log. Ordinary complex-base
/// solves can also retry with two-sided anchoring after a rejected residual;
/// neither branch choice alone validates a candidate.
fn unwrapped_ln_samples(
    samples: &[Complex],
    l_upper: &Complex,
    l_lower: &Complex,
    ln_b: &Complex,
    prec: u64,
    two_sided: bool,
) -> Vec<Complex> {
    let n = samples.len();
    if !two_sided {
        return samples.iter().map(|s| cnum::ln_complex(s, prec)).collect();
    }
    let two_pi = Float::with_val_64(prec, Constant::Pi) * 2u32;
    let debug = cnum::verbose() && std::env::var_os("TET_KOUZ_UNWRAP_DEBUG").is_some();
    // Joint between the two anchored walks: fixed mid index (samples are
    // ordered bottom→top; nodes near n/2 sit mid-curve where |F| is
    // largest for cut bases, far from both the origin and the asymptotes).
    let joint = n / 2;
    let mut any_nonzero = false;
    // Top half [joint..n): walk down from the ln_b·L_up anchor.
    let top_anchor = Complex::with_val_64(prec, ln_b * l_upper);
    let mut ref_im = top_anchor.imag().clone();
    let mut top: Vec<Complex> = Vec::with_capacity(n - joint);
    for sample in samples[joint..n].iter().rev() {
        let pl = cnum::ln_complex(sample, prec);
        let pl_im = pl.imag();
        let k = ((ref_im - pl_im) / &two_pi).round();
        let adjusted = if k.is_zero() {
            pl
        } else {
            any_nonzero = true;
            let delta = Float::with_val_64(prec, &two_pi * &Float::with_val_64(prec, k));
            let delta_c = Complex::with_val_64(prec, (Float::new_64(prec), delta));
            Complex::with_val_64(prec, &pl + &delta_c)
        };
        ref_im = adjusted.imag().clone();
        top.push(adjusted);
    }
    top.reverse();
    // Bottom half [0..joint): walk up from the ln_b·L_low anchor.
    let bot_anchor = Complex::with_val_64(prec, ln_b * l_lower);
    let mut ref_im = bot_anchor.imag().clone();
    let mut out: Vec<Complex> = Vec::with_capacity(n);
    for sample in samples.iter().take(joint) {
        let pl = cnum::ln_complex(sample, prec);
        let pl_im = pl.imag();
        let k = ((ref_im - pl_im) / &two_pi).round();
        let adjusted = if k.is_zero() {
            pl
        } else {
            any_nonzero = true;
            let delta = Float::with_val_64(prec, &two_pi * &Float::with_val_64(prec, k));
            let delta_c = Complex::with_val_64(prec, (Float::new_64(prec), delta));
            Complex::with_val_64(prec, &pl + &delta_c)
        };
        ref_im = adjusted.imag().clone();
        out.push(adjusted);
    }
    out.extend(top);
    if debug && any_nonzero {
        // Joint mismatch (2π multiples) indicates a wrong-homotopy-class
        // iterate; the error stays localized at the joint node (see doc).
        let lo_im = out[joint.saturating_sub(1)].imag();
        let hi_im = out[joint.min(n - 1)].imag();
        let m = (Float::with_val_64(prec, hi_im - lo_im) / &two_pi).round();
        let mut n_dev = 0usize;
        for (u, s) in out.iter().zip(samples.iter()) {
            let pl = cnum::ln_complex(s, prec);
            let d = cnum::abs(&Complex::with_val_64(prec, u - &pl), prec);
            if d > 1 {
                n_dev += 1;
            }
        }
        eprintln!(
            "kouz unwrap: joint winding m = {}, {} nodes branch-corrected (of {})",
            DisplayFloat(&m),
            n_dev,
            n
        );
    }
    out
}

fn validate_cauchy_data(
    samples: &[Complex],
    nodes: &[Float],
    weights: &[Float],
    t_max: &Float,
    ln_b: &Complex,
) -> Result<(), String> {
    if samples.len() < 2
        || samples.len() != nodes.len()
        || samples.len() != weights.len()
        || !t_max.is_finite()
        || *t_max <= 0
        || !cnum::is_finite(ln_b)
        || cnum::is_zero(ln_b)
        || samples
            .iter()
            .any(|s| !cnum::is_finite(s) || cnum::is_zero(s))
        || nodes.iter().any(|t| !t.is_finite())
        || weights.iter().any(|w| !w.is_finite() || *w <= 0)
    {
        return Err("Cauchy operator requires finite nonzero samples/log(base) and matching finite geometry".into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cauchy_eval(
    z0: &Complex,
    samples: &[Complex],
    nodes: &[Float],
    weights: &[Float],
    t_max: &Float,
    l_upper: &Complex,
    l_lower: &Complex,
    ln_b: &Complex,
    prec: u64,
    two_sided: bool,
) -> Result<Complex, String> {
    validate_cauchy_data(samples, nodes, weights, t_max, ln_b)?;
    let cp1 = Complex::with_val_64(prec, (cnum::decimal("1.5", prec), 0));
    let cm1 = Complex::with_val_64(prec, (cnum::decimal("-0.5", prec), 0));
    let ln_unwrapped = unwrapped_ln_samples(samples, l_upper, l_lower, ln_b, prec, two_sided);

    let mut r_int = cnum::zero(prec);
    let mut l_int = cnum::zero(prec);
    for k in 0..nodes.len() {
        let it = Complex::with_val_64(prec, (Float::new_64(prec), nodes[k].clone()));

        // F(c+1+it_k) = b^samples[k] = exp(ln_b · F)
        let exp_arg = Complex::with_val_64(prec, ln_b * &samples[k]);
        let b_f = cnum::checked_exp(&exp_arg, prec)?;
        let cp1_plus_it = Complex::with_val_64(prec, &cp1 + &it);
        let denom_r = Complex::with_val_64(prec, &cp1_plus_it - z0);
        let term_r = Complex::with_val_64(prec, &b_f / &denom_r);
        r_int += Complex::with_val_64(prec, &term_r * &weights[k]);

        // F(c-1+it_k) = log_b(samples[k]) = ln(F) / ln_b, with the branch of
        // ln(F) unwrapped along the sample curve (anchored at ln_b·L_up).
        let log_b_s = Complex::with_val_64(prec, &ln_unwrapped[k] / ln_b);
        let cm1_plus_it = Complex::with_val_64(prec, &cm1 + &it);
        let denom_l = Complex::with_val_64(prec, &cm1_plus_it - z0);
        let term_l = Complex::with_val_64(prec, &log_b_s / &denom_l);
        l_int += Complex::with_val_64(prec, &term_l * &weights[k]);
    }

    // Euler-Maclaurin boundary correction. The trapezoidal sums above have
    // O(h²) error driven by `g'(±T)`, which is closed-form (since F is
    // pinned to L_{up/down} at the contour endpoints). Subtracting the
    // first K EM terms drops the error to O(h^{2K+2}).
    //
    // `cauchy_eval` is called with arbitrary z₀, so we cannot reuse the
    // per-row precomputation that `apply_t_fft` does — but the cost (one
    // power-of-2 sequence times K terms) is O(K) MPC ops, negligible.
    let h = if nodes.len() >= 2 {
        Float::with_val_64(prec, &nodes[1] - &nodes[0])
    } else {
        Float::with_val_64(prec, 1u32)
    };
    let n_terms = em_n_terms(prec, &h, l_upper, l_lower)?;
    let em_h_powers = build_em_h_powers(&h, n_terms, prec)?;
    let (corr_r, corr_l) =
        compute_em_correction_z0(z0, t_max, l_upper, l_lower, &em_h_powers, prec);
    let r_int = Complex::with_val_64(prec, &r_int - &corr_r);
    let l_int = Complex::with_val_64(prec, &l_int - &corr_l);

    // Top edge contributes L̄ · ln((c-1+iT-z0)/(c+1+iT-z0)).
    let it_max = Complex::with_val_64(prec, (Float::new_64(prec), t_max.clone()));
    let neg_it_max = Complex::with_val_64(prec, -&it_max);
    let cm1_plus_itmax = Complex::with_val_64(prec, &cm1 + &it_max);
    let cp1_plus_itmax = Complex::with_val_64(prec, &cp1 + &it_max);
    let top_num = Complex::with_val_64(prec, &cm1_plus_itmax - z0);
    let top_den = Complex::with_val_64(prec, &cp1_plus_itmax - z0);
    let top_ratio = Complex::with_val_64(prec, &top_num / &top_den);
    let ln_top = cnum::ln_complex(&top_ratio, prec);

    // Bottom edge contributes L · ln((c+1-iT-z0)/(c-1-iT-z0)).
    let cp1_minus_itmax = Complex::with_val_64(prec, &cp1 + &neg_it_max);
    let cm1_minus_itmax = Complex::with_val_64(prec, &cm1 + &neg_it_max);
    let bot_num = Complex::with_val_64(prec, &cp1_minus_itmax - z0);
    let bot_den = Complex::with_val_64(prec, &cm1_minus_itmax - z0);
    let bot_ratio = Complex::with_val_64(prec, &bot_num / &bot_den);
    let ln_bot = cnum::ln_complex(&bot_ratio, prec);

    let pi_f = Float::with_val_64(prec, rug::float::Constant::Pi);
    let two_pi_f = Float::with_val_64(prec, &pi_f * 2u32);
    let two_pi_i = Complex::with_val_64(prec, (Float::new_64(prec), two_pi_f.clone()));

    let diff = Complex::with_val_64(prec, &r_int - &l_int);
    let part1 = Complex::with_val_64(prec, &diff / &two_pi_f);

    let up_term = Complex::with_val_64(prec, l_upper * &ln_top);
    let dn_term = Complex::with_val_64(prec, l_lower * &ln_bot);
    let upper_lower_sum = Complex::with_val_64(prec, &up_term + &dn_term);
    let part2 = Complex::with_val_64(prec, &upper_lower_sum / &two_pi_i);

    let result = Complex::with_val_64(prec, &part1 + &part2);
    if !cnum::is_finite(&result) {
        return Err("Cauchy reconstruction produced a non-finite value".into());
    }
    Ok(result)
}

/// Apply the Cauchy operator `T` to all sample points.
#[allow(clippy::too_many_arguments)]
fn apply_t(
    samples: &[Complex],
    nodes: &[Float],
    weights: &[Float],
    t_max: &Float,
    l_upper: &Complex,
    l_lower: &Complex,
    ln_b: &Complex,
    prec: u64,
    two_sided: bool,
) -> Result<Vec<Complex>, String> {
    let mut out = Vec::with_capacity(nodes.len());
    for k in 0..nodes.len() {
        let z0 = Complex::with_val_64(prec, (cnum::decimal("0.5", prec), nodes[k].clone()));
        out.push(cauchy_eval(
            &z0, samples, nodes, weights, t_max, l_upper, l_lower, ln_b, prec, two_sided,
        )?);
    }
    Ok(out)
}

/// Anderson-accelerated Picard iteration: each step extrapolates from a
/// sliding history of recent residuals to find the fixed point of T faster
/// than plain Picard. With pinned boundary samples (so Cauchy doesn't see its
/// own pole), T is contractive in interior modes and Anderson converges
/// super-linearly.
///
/// Algorithm (Type-II Anderson with depth m):
///   1. Compute r_k = T(x_k) − x_k.
///   2. Maintain Δx and Δr histories of length up to m (last m steps).
///   3. Solve small least-squares: γ = argmin ‖r_k − Δr · γ‖.
///   4. Update: x_{k+1} = x_k + β·r_k − (Δx + β·Δr)·γ
///      with β = mixing parameter (smaller for stability when T is borderline).
///
/// We work on the *interior* samples only (boundary samples stay pinned).
#[allow(clippy::too_many_arguments)]
fn iterate_anderson(
    initial: Vec<Complex>,
    nodes: &[Float],
    weights: &[Float],
    t_max: &Float,
    l_upper: &Complex,
    l_lower: &Complex,
    ln_b: &Complex,
    prec: u64,
    digits: u64,
    use_schwarz: bool,
    two_sided: bool,
) -> Result<(Vec<Complex>, Float), String> {
    let n = initial.len();
    let n_int = n - 2; // number of interior samples that actually iterate
    let target = cnum::epsilon(digits.saturating_add(3), prec);
    let debug = cnum::verbose();

    let depth = cnum::env_usize("TET_KOUZ_ANDERSON_DEPTH", 8)?;
    let beta = cnum::env_float("TET_KOUZ_ANDERSON_BETA", "1", prec)?;
    if beta <= 0 || beta > 1 || depth == 0 {
        return Err("Anderson requires 0 < beta <= 1 and positive depth".into());
    }

    let mut x = initial;
    let mut best_x = x.clone();
    let mut best_residual = Float::with_val_64(prec, rug::float::Special::Infinity);

    // History of Δx[k] and Δr[k] (interior samples flattened as 1D vector).
    let mut hist_dx: Vec<Vec<Complex>> = Vec::new();
    let mut hist_dr: Vec<Vec<Complex>> = Vec::new();
    let mut prev_x_int: Option<Vec<Complex>> = None;
    let mut prev_r_int: Option<Vec<Complex>> = None;
    let mut checkpoint = (
        x.clone(),
        hist_dx.clone(),
        hist_dr.clone(),
        prev_x_int.clone(),
        prev_r_int.clone(),
    );
    let mut iter = Integer::new();

    loop {
        let f = apply_t(
            &x, nodes, weights, t_max, l_upper, l_lower, ln_b, prec, two_sided,
        )?;
        let mut r_int: Vec<Complex> = Vec::with_capacity(n_int);
        let mut x_int: Vec<Complex> = Vec::with_capacity(n_int);
        let mut r_norm = Float::new_64(prec);
        for i in 1..n - 1 {
            let d = Complex::with_val_64(prec, &f[i] - &x[i]);
            let m = cnum::abs(&d, prec);
            if !m.is_finite() {
                return Err(format!("Anderson residual is non-finite at sample {}", i));
            }
            if m > r_norm {
                r_norm = m;
            }
            r_int.push(d);
            x_int.push(x[i].clone());
        }

        if r_norm < best_residual {
            best_residual = r_norm.clone();
            best_x = x.clone();
        }

        if debug && (iter < 10 || iter.is_divisible_u(25)) {
            let mid_idx = nodes.len() / 2;
            let xm_re = x[mid_idx].real();
            let xm_im = x[mid_idx].imag();
            eprintln!(
                "kouz Anderson iter {:>4}: ‖r‖∞ = {:.3e}  depth={}  F(0.5)≈{:.4}+{:.4}i  best={:.3e}  (target {:.3e})",
                iter, DisplayFloat(&r_norm), hist_dx.len(), DisplayFloat(xm_re),
                DisplayFloat(xm_im), DisplayFloat(&best_residual), DisplayFloat(&target)
            );
        }

        if r_norm < target {
            return Ok((x, r_norm));
        }
        if !r_norm.is_finite() {
            // Anderson can blow up after hitting the floor; return best iterate.
            if best_residual.is_finite() {
                if debug {
                    eprintln!(
                        "kouz Anderson: residual non-finite at iter {}; returning best ({:.3e})",
                        iter,
                        DisplayFloat(&best_residual)
                    );
                }
                return Ok((best_x, best_residual));
            }
            return Err(format!(
                "Kouznetsov Anderson: residual non-finite at iter {}",
                iter
            ));
        }
        // Adaptive mixing: when residual is large the operator is far from
        // its fixed point and a full step (β=1) easily overshoots into log/exp
        // overflow territory. Damp aggressively until r is back below O(0.1),
        // then unleash the user-specified β. Also clear Anderson history while
        // we damp, so unstable past steps aren't extrapolated through.
        let effective_beta = if r_norm > 10 {
            cnum::decimal("0.02", prec)
        } else if r_norm > 1 {
            cnum::decimal("0.1", prec)
        } else if r_norm > cnum::decimal("0.1", prec) {
            cnum::decimal("0.5", prec)
        } else {
            beta.clone()
        };
        let beta_f = effective_beta.clone();
        if effective_beta < cnum::decimal("0.5", prec) {
            // In the damping phase, treat each step as a fresh start: skip
            // Anderson extrapolation entirely (history would amplify the
            // unstable initial transients).
            hist_dx.clear();
            hist_dr.clear();
            prev_x_int = None;
            prev_r_int = None;
        }

        // Update history.
        if let (Some(prev_x), Some(prev_r)) = (&prev_x_int, &prev_r_int) {
            let mut dx: Vec<Complex> = Vec::with_capacity(n_int);
            let mut dr: Vec<Complex> = Vec::with_capacity(n_int);
            for i in 0..n_int {
                dx.push(Complex::with_val_64(prec, &x_int[i] - &prev_x[i]));
                dr.push(Complex::with_val_64(prec, &r_int[i] - &prev_r[i]));
            }
            hist_dx.push(dx);
            hist_dr.push(dr);
            if hist_dx.len() > depth {
                hist_dx.remove(0);
                hist_dr.remove(0);
            }
        }
        prev_x_int = Some(x_int.clone());
        prev_r_int = Some(r_int.clone());

        // Solve small LS: γ = argmin ‖Δr · γ − r_int‖² (least-squares).
        // Build normal equations (m×m): A_ij = <Δr_j, Δr_i>, b_i = <Δr_i, r_int>.
        let m = hist_dr.len();
        cnum::check_complex_storage((m as u128 + 1) * m as u128, prec)?;
        let mut gamma: Vec<Complex> = vec![cnum::zero(prec); m];
        if m > 0 {
            let mut a: Vec<Vec<Complex>> = vec![vec![cnum::zero(prec); m]; m];
            let mut bvec: Vec<Complex> = vec![cnum::zero(prec); m];
            for j in 0..m {
                for k in 0..m {
                    let mut s = cnum::zero(prec);
                    for (dr_j, dr_k) in hist_dr[j].iter().zip(&hist_dr[k]) {
                        let conj_jk = Complex::with_val_64(prec, dr_j.conj_ref());
                        let prod = Complex::with_val_64(prec, &conj_jk * dr_k);
                        s = Complex::with_val_64(prec, &s + &prod);
                    }
                    a[j][k] = s;
                }
                let mut s = cnum::zero(prec);
                for i in 0..n_int {
                    let conj_j = Complex::with_val_64(prec, hist_dr[j][i].conj_ref());
                    let prod = Complex::with_val_64(prec, &conj_j * &r_int[i]);
                    s = Complex::with_val_64(prec, &s + &prod);
                }
                bvec[j] = s;
            }
            // Tikhonov regularization to handle near-singular A.
            let reg_f = cnum::epsilon(12, prec);
            for (j, row) in a.iter_mut().enumerate() {
                row[j] = Complex::with_val_64(prec, &row[j] + &reg_f);
            }
            gamma = match solve_complex_lin(&a, &bvec, prec) {
                Ok(v) => v,
                Err(e) => {
                    if debug {
                        eprintln!("kouz Anderson: least-squares step failed ({e}); using damped Picard step");
                    }
                    vec![cnum::zero(prec); m]
                }
            };
        }

        // x_{k+1} = x_k + β·r_k − (Δx + β·Δr)·γ
        //
        // Per-sample step capping. T(F) involves `b^F` on the right edge of
        // the contour: a small change in `F[k]` at one node induces a change
        // of order `b · b^F · ΔF` in `T(F)`, which is huge for `b > e^(1/e)`
        // and large `|F|`. Even with small β the unbounded direction `r[k]`
        // can take a single sample into a regime where the next `T(F)` is
        // exponentially worse, and the iteration cascades to overflow.
        //
        // Cap the per-sample magnitude of the step to `step_cap`, scaled by
        // `1 + |x[i]|` so steps stay proportional to the local F magnitude.
        // Start strict (`base_cap = 0.3`) when r is large and relax as we
        // approach a fixed point.
        let base_cap = if r_norm > 10 {
            cnum::decimal("0.1", prec)
        } else if r_norm > 1 {
            cnum::decimal("0.3", prec)
        } else {
            Float::with_val_64(prec, rug::float::Special::Infinity)
        };

        let mut new_x_int: Vec<Complex> = Vec::with_capacity(n_int);
        for i in 0..n_int {
            let beta_r = Complex::with_val_64(prec, &r_int[i] * &beta_f);
            let mut step_i = beta_r;
            for j in 0..m {
                let beta_dr = Complex::with_val_64(prec, &hist_dr[j][i] * &beta_f);
                let term = Complex::with_val_64(prec, &hist_dx[j][i] + &beta_dr);
                let prod = Complex::with_val_64(prec, &term * &gamma[j]);
                step_i = Complex::with_val_64(prec, &step_i - &prod);
            }
            if base_cap.is_finite() {
                let step_mag = cnum::abs(&step_i, prec);
                let x_mag = cnum::abs(&x_int[i], prec);
                let cap_i = Float::with_val_64(prec, &base_cap * (x_mag + 1u32));
                if step_mag > cap_i && step_mag.is_finite() {
                    let scale_f = Float::with_val_64(prec, cap_i / step_mag);
                    step_i = Complex::with_val_64(prec, &step_i * &scale_f);
                }
            }
            let update = Complex::with_val_64(prec, &x_int[i] + &step_i);
            new_x_int.push(update);
        }

        // Reassemble full x: pinned boundary, Anderson-updated interior.
        let mut x_new = Vec::with_capacity(n);
        x_new.push(x[0].clone());
        for c in new_x_int {
            x_new.push(c);
        }
        x_new.push(x[n - 1].clone());
        if use_schwarz {
            symmetrize_schwarz(&mut x_new, prec);
        }
        x = x_new;
        if x == checkpoint.0
            && hist_dx == checkpoint.1
            && hist_dr == checkpoint.2
            && prev_x_int == checkpoint.3
            && prev_r_int == checkpoint.4
        {
            return validate_best_residual(
                &best_residual,
                digits,
                use_schwarz,
                "Anderson repeated state",
            )
            .map(|_| (best_x, best_residual));
        }
        if iter.is_power_of_two() {
            checkpoint = (
                x.clone(),
                hist_dx.clone(),
                hist_dr.clone(),
                prev_x_int.clone(),
                prev_r_int.clone(),
            );
        }
        iter += 1;
    }
}

/// Damped Picard iteration: x ← (1-α)·x + α·T(x). Used as a debugging baseline
/// to verify convergence of the Cauchy operator and explore stable α values.
/// Activated by `TET_KOUZ_PICARD=1`. The Newton/LM path is the production
/// algorithm; this exists for experimental comparison.
#[allow(clippy::too_many_arguments)]
fn iterate_picard(
    initial: Vec<Complex>,
    nodes: &[Float],
    weights: &[Float],
    t_max: &Float,
    l_upper: &Complex,
    l_lower: &Complex,
    ln_b: &Complex,
    prec: u64,
    digits: u64,
    use_schwarz: bool,
    two_sided: bool,
) -> Result<(Vec<Complex>, Float), String> {
    let n = initial.len();
    let target = cnum::epsilon(digits.saturating_add(3), prec);
    let debug = cnum::verbose();

    // Mixing parameter. α=1 is pure Picard. Smaller values (0.05–0.3) under-
    // relax to handle T's spectral radius near 1 in some directions.
    let alpha = cnum::env_float("TET_KOUZ_ALPHA", "0.3", prec)?;
    if alpha <= 0 || alpha > 1 {
        return Err("Picard requires 0 < alpha <= 1".into());
    }
    let alpha_f = alpha.clone();
    let one_minus_alpha = Float::with_val_64(prec, 1) - &alpha;

    let mut x = initial;
    let mut checkpoint = x.clone();
    let mut iter = Integer::new();

    loop {
        let f = apply_t(
            &x, nodes, weights, t_max, l_upper, l_lower, ln_b, prec, two_sided,
        )?;
        let mut r_norm = Float::new_64(prec);
        // Skip boundary samples: their values are pinned (see tetrate_kouznetsov),
        // and Cauchy at those z₀'s is degenerate anyway.
        for i in 1..n - 1 {
            let d = Complex::with_val_64(prec, &f[i] - &x[i]);
            let m = cnum::abs(&d, prec);
            if !m.is_finite() {
                return Err(format!("Picard residual is non-finite at sample {}", i));
            }
            if m > r_norm {
                r_norm = m;
            }
        }
        if debug && (iter < 20 || iter.is_divisible_u(50)) {
            let mid_idx = nodes.len() / 2;
            let xm_re = x[mid_idx].real();
            let xm_im = x[mid_idx].imag();
            // Find the index where residual is max (interior only).
            let mut max_idx = 1usize;
            let mut max_val = Float::new_64(prec);
            for i in 1..n - 1 {
                let d = Complex::with_val_64(prec, &f[i] - &x[i]);
                let m = cnum::abs(&d, prec);
                if m > max_val {
                    max_val = m;
                    max_idx = i;
                }
            }
            let max_t = &nodes[max_idx];
            eprintln!(
                "kouz Picard iter {:>4}: ‖r‖∞ = {:.3e}  α={:.2}  F(0.5)≈{:.4}+{:.4}i  argmax_t={:.3} (idx {})",
                iter, DisplayFloat(&r_norm), DisplayFloat(&alpha), DisplayFloat(xm_re),
                DisplayFloat(xm_im), DisplayFloat(max_t), max_idx
            );
        }
        if r_norm < target {
            return Ok((x, r_norm));
        }
        if !r_norm.is_finite() {
            return Err(format!(
                "Kouznetsov Picard: residual non-finite at iter {} (α={})",
                iter,
                DisplayFloat(&alpha)
            ));
        }
        // Mix interior samples; keep boundary pinned to L_lower / L_upper.
        let mut x_new = Vec::with_capacity(n);
        x_new.push(x[0].clone());
        for i in 1..n - 1 {
            let lhs = Complex::with_val_64(prec, &x[i] * &one_minus_alpha);
            let rhs = Complex::with_val_64(prec, &f[i] * &alpha_f);
            x_new.push(Complex::with_val_64(prec, &lhs + &rhs));
        }
        x_new.push(x[n - 1].clone());
        if use_schwarz {
            symmetrize_schwarz(&mut x_new, prec);
        }
        x = x_new;
        if x == checkpoint {
            return Err(format!(
                "Kouznetsov Picard repeated an iterate at residual {}",
                DisplayFloat(&r_norm)
            ));
        }
        if iter.is_power_of_two() {
            checkpoint = x.clone();
        }
        iter += 1;
    }
}

fn dump_residual(
    path: &std::path::Path,
    nodes: &[Float],
    samples: &[Complex],
    residuals: &[Complex],
    evaluated: &[Complex],
    prec: u64,
) -> Result<(), String> {
    use std::io::Write;
    if nodes.len() != samples.len()
        || nodes.len() != residuals.len()
        || nodes.len() != evaluated.len()
    {
        return Err("residual dump arrays have inconsistent lengths".into());
    }
    let write = || -> std::io::Result<()> {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        let mut out = std::io::BufWriter::new(file);
        writeln!(out, "t\tre_F\tim_F\tabs_r\tre_T\tim_T")?;
        for i in 0..nodes.len() {
            writeln!(
                out,
                "{}\t{}\t{}\t{}\t{}\t{}",
                cnum::format_float_roundtrip(&nodes[i]),
                cnum::format_float_roundtrip(samples[i].real()),
                cnum::format_float_roundtrip(samples[i].imag()),
                cnum::format_float_roundtrip(&cnum::abs(&residuals[i], prec)),
                cnum::format_float_roundtrip(evaluated[i].real()),
                cnum::format_float_roundtrip(evaluated[i].imag())
            )?;
        }
        out.flush()?;
        out.get_ref().sync_all()
    };
    write().map_err(|e| format!("residual dump {}: {e}", path.display()))?;
    if cnum::verbose() {
        eprintln!("kouz residual dump: {}", path.display());
    }
    Ok(())
}

/// Newton-Kantorovich iteration on the Cauchy operator.
///
/// We're solving for samples `F` such that `T(F) = F`, where `T` is the Cauchy
/// formula on the rectangle (right edge `b^F`, left edge `log_b F`, top/bottom
/// constants `L_upper / L_lower`). At each Newton step we form the residual
/// `r = T(F) − F`, build the Jacobian `J = I − DT` analytically, and solve
/// `J · δ = r`, then update `F ← F + δ`.
///
/// Why Newton instead of Picard / Anderson: the spectral radius of `T` is
/// slightly above 1 in some directions on this problem, so plain Picard
/// diverges and Anderson stalls (verified empirically — residual hits a floor
/// near 10⁻² and then drifts). Newton uses the Jacobian to take optimal-size
/// steps along each eigendirection regardless of the spectrum.
///
/// Each entry of `DT` is closed-form:
///
///   DT[k][j] = (w_j / 2π) · [ b^F[j]·ln(b) / (1+i(t_j−t_k))
///                            − 1/(F[j]·ln(b)) / (−1+i(t_j−t_k)) ]
///
/// **Linear solve via Newton-Krylov GMRES.** We never form J explicitly.
/// The matvec `(I − DT + μ·I)·v` is computed in O(N²) using two precomputed
/// helper arrays (per-sample factors `b^F·ln(b)` and `1/(F·ln(b))`, and
/// per-offset denominator inverses `1/(±1+i(t_j−t_k))` that depend only on
/// the uniform grid). GMRES with restarted Arnoldi solves the linear system
/// to a relative tolerance of `1e-3` in K Krylov steps; total cost
/// `O(K·N²)` beats dense LU's `O(N³)/3` for `N > ~300` and is the only
/// tractable option above N ≈ 5000 (where LU's `N³` cost runs into days
/// even at moderate precision).
///
/// Solve `(J + μ I) δ = r`, where `J = I - DT`, with residual-scaled
/// GMRES tolerances and capped corrections. Increase the diagonal shift on
/// rejection and decrease it on acceptance. This is regularized Newton,
/// not least-squares Levenberg-Marquardt; logs retain the historical LM label.
/// Only a decreasing finite infinity-norm residual is accepted.
#[allow(clippy::too_many_arguments)]
fn iterate_newton(
    initial: Vec<Complex>,
    nodes: &[Float],
    weights: &[Float],
    t_max: &Float,
    l_upper: &Complex,
    l_lower: &Complex,
    ln_b: &Complex,
    prec: u64,
    digits: u64,
    use_schwarz: bool,
    two_sided: bool,
) -> Result<(Vec<Complex>, Float), String> {
    let n = initial.len();
    let target = cnum::epsilon(digits.saturating_add(3), prec);
    let debug = cnum::verbose();
    static DUMP_RUN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dump = std::env::var_os("TET_KOUZ_RESID_DUMP").map(|prefix| {
        (
            prefix,
            DUMP_RUN.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        )
    });
    if dump.as_ref().is_some_and(|(prefix, _)| prefix.is_empty()) {
        return Err("TET_KOUZ_RESID_DUMP requires a nonempty output path prefix".into());
    }

    let mut x = initial;
    let mut best_x = x.clone();
    let mut best_residual = Float::with_val_64(prec, rug::float::Special::Infinity);
    // Large μ approaches a damped Picard direction, not a gradient direction;
    // increasing it does not guarantee descent.
    let mut mu = Float::with_val_64(prec, 1);

    // FFT-domain kernels. The right/left edge denominator inverses depend
    // only on the uniform grid spacing — building them once per call collapses
    // every later matvec to two length-`m` FFTs (m = next_pow2(3N − 2)). The
    // same struct also carries per-row Euler-Maclaurin corrections, so the
    // residual evaluator just adds a precomputed shift instead of
    // recomputing closed-form boundary terms each iteration.
    let kernel_start = std::time::Instant::now();
    let kernels = build_cauchy_kernels(nodes, t_max, l_upper, l_lower, prec)?;
    if debug {
        eprintln!(
            "kouz phase: build_cauchy_kernels done ({:.2}s)",
            kernel_start.elapsed().as_secs_f64()
        );
    }

    let mid_idx = nodes.len() / 2;
    if debug {
        eprintln!(
            "kouz LM start: n={} mid_idx={} mid_t={} F[mid]={}+{}i",
            nodes.len(),
            mid_idx,
            DisplayFloat(&nodes[mid_idx]),
            DisplayFloat(x[mid_idx].real()),
            DisplayFloat(x[mid_idx].imag()),
        );
    }

    let mut iter = Integer::new();
    loop {
        let iter_start = std::time::Instant::now();
        let matvec_start = std::time::Instant::now();
        let f = apply_t_fft(
            &x, nodes, weights, t_max, l_upper, l_lower, ln_b, &kernels, prec, two_sided,
        )?;
        let matvec_secs = matvec_start.elapsed().as_secs_f64();
        let mut r = Vec::with_capacity(n);
        let mut r_norm = Float::new_64(prec);
        for i in 0..n {
            let d = Complex::with_val_64(prec, &f[i] - &x[i]);
            // Boundary samples are pinned (corner Cauchy singularity); zero
            // out their residual so the linear solve does not try to update
            // them. We track r_norm only over interior samples for the same
            // reason — boundary "residuals" are pure discretization noise.
            if i == 0 || i == n - 1 {
                r.push(cnum::zero(prec));
                continue;
            }
            let m = cnum::abs(&d, prec);
            if !m.is_finite() {
                return Err(format!("Newton residual is non-finite at sample {}", i));
            }
            if m > r_norm {
                r_norm = m;
            }
            r.push(d);
        }

        if debug {
            // Print F at the node closest to t=0 (the real-axis sample); for
            // b=e the natural Kouznetsov F_e(0.5) ≈ 1.6463. Knowing whether x
            // is converging toward this value tells us if we're in the right
            // basin of attraction.
            let xm_re = x[mid_idx].real();
            let xm_im = x[mid_idx].imag();
            eprintln!(
                "kouz LM iter {:>3}: ‖r‖∞ = {:.3e}  μ={:.2e}  F(0.5)≈{:.4}+{:.4}i  (target {:.3e})  matvec={:.2}s",
                iter, DisplayFloat(&r_norm), DisplayFloat(&mu), DisplayFloat(xm_re),
                DisplayFloat(xm_im), DisplayFloat(&target), matvec_secs
            );
        }
        if let Some((prefix, run)) = &dump {
            let mut path = prefix.clone();
            path.push(format!("-{}-{run}-iter{iter}.tsv", std::process::id()));
            dump_residual(std::path::Path::new(&path), nodes, &x, &r, &f, prec)?;
        }

        if r_norm.is_nan() || !r_norm.is_finite() {
            if debug {
                eprintln!(
                    "kouz LM: residual non-finite at iter {}; best so far {:.3e}",
                    iter,
                    DisplayFloat(&best_residual)
                );
            }
            return validate_best_residual(
                &best_residual,
                digits,
                use_schwarz,
                "residual became non-finite",
            )
            .map(|_| (best_x, best_residual));
        }
        if r_norm < best_residual {
            best_residual = r_norm.clone();
            best_x = x.clone();
        }
        if r_norm < target {
            return Ok((x, r_norm));
        }
        // Precompute J = I − DT helpers ONCE per Newton iteration. They
        // depend on `x` but not on the LM damping μ, so we hoist them out
        // of the inner LM line search.
        let (b_f_ln, inv_f_ln) = precompute_dt_factors(&x, ln_b, prec)?;

        // Continue damping until descent or no representable update remains.
        let mut accepted = false;
        let mut lm_try = Integer::new();
        loop {
            if !mu.is_finite() {
                break;
            }
            if debug {
                eprintln!("kouz LM trial {lm_try}: mu={:.3e}", DisplayFloat(&mu));
            }
            lm_try += 1;
            // Boundary-pinned matvec: rows 0 and n-1 act as identity (so
            // δ[0] = r[0] = 0 and δ[n-1] = r[n-1] = 0 stay zero throughout
            // the Krylov build); interior rows compute (1+μ)·v − DT·v.
            let mu_local = mu.clone();
            let matvec = |v: &[Complex]| -> Vec<Complex> {
                let dt_v = apply_dt_v_fft(&b_f_ln, &inv_f_ln, &kernels, weights, v, prec);
                let scale = Float::with_val_64(prec, &mu_local + 1);
                let mut out = Vec::with_capacity(n);
                for k in 0..n {
                    let scaled = Complex::with_val_64(prec, &v[k] * &scale);
                    out.push(Complex::with_val_64(prec, &scaled - &dt_v[k]));
                }
                out[0] = v[0].clone();
                out[n - 1] = v[n - 1].clone();
                out
            };
            // Residual-scaled forcing, with a requested-precision floor.
            // GMRES measures relative L2 residual of the shifted system;
            // the outer infinity norm sets this heuristic tolerance.
            let abs_floor = cnum::epsilon(digits.saturating_add(5), prec);
            let abs_target = (Float::with_val_64(prec, &r_norm * &r_norm) / 10u32).max(&abs_floor);
            let inner_tol = if r_norm > 0 {
                (abs_target / &r_norm).min(&cnum::decimal("1e-3", prec))
            } else {
                cnum::decimal("1e-3", prec)
            };
            let restart = 80.min(n);
            let delta = match gmres_complex(matvec, &r, &inner_tol, restart, prec) {
                Ok(d) => d,
                Err(e) => {
                    if debug {
                        eprintln!(
                            "kouz LM iter {}: GMRES failed (tol={:.2e}): {}",
                            iter,
                            DisplayFloat(&inner_tol),
                            e
                        );
                    }
                    if !e.contains("stagnation") && !e.contains("zero pivot") {
                        return Err(e);
                    }
                    mu *= 4;
                    continue;
                }
            };

            // Newton step cap: when the step is much larger than the current
            // residual, the linearization is being trusted further than it
            // should be. Cap ‖δ‖∞ at `4·max(‖r‖∞, 0.05)` so that as r→0 the
            // method takes shorter, more reliable steps. Without this, big
            // steps near a small residual can land in a region where J is
            // near-singular and the next iteration cannot recover.
            let max_step = r_norm.clone().max(&cnum::decimal("0.05", prec)) * 4u32;
            let mut delta_inf = Float::new_64(prec);
            for d in &delta {
                let m = cnum::abs(d, prec);
                if !m.is_finite() {
                    return Err("Newton correction is non-finite".into());
                }
                if m > delta_inf {
                    delta_inf = m;
                }
            }
            let scale = if delta_inf > max_step {
                max_step / delta_inf
            } else {
                Float::with_val_64(prec, 1)
            };
            let scale_f = Float::with_val_64(prec, scale);

            let mut x_trial = Vec::with_capacity(n);
            for i in 0..n {
                let scaled = Complex::with_val_64(prec, &delta[i] * &scale_f);
                x_trial.push(Complex::with_val_64(prec, &x[i] + &scaled));
            }
            // Re-pin boundaries (defensive: scale may not have been applied
            // exactly to those rows if the solve had pivoting noise) and
            // re-impose Schwarz reflection to keep iterates on the natural
            // F's symmetry manifold (real-base regime only).
            x_trial[0] = l_lower.clone();
            x_trial[n - 1] = l_upper.clone();
            if use_schwarz {
                symmetrize_schwarz(&mut x_trial, prec);
            }
            if x_trial == x {
                break;
            }
            // Cheap basin guard: for real-positive bases > e^(1/e), the natural
            // Kneser F is real and POSITIVE on Re(z)=0. If the trial step
            // pushes F[mid] negative, we have crossed into a wrong basin
            // (e.g. settling at F[mid]≈−0.7 with a local-but-not-global
            // residual minimum). Reject the step BEFORE the expensive
            // apply_t_fft / residual computation — this saves ~1 matvec per
            // bad LM trial, which matters because large bases trigger many
            // such trials per outer iteration.
            let mid_idx = n / 2;
            let f_mid_re_trial = x_trial[mid_idx].real();
            if use_schwarz && *f_mid_re_trial < 0 {
                mu *= 4;
                continue;
            }
            let f_trial = match apply_t_fft(
                &x_trial, nodes, weights, t_max, l_upper, l_lower, ln_b, &kernels, prec, two_sided,
            ) {
                Ok(value) => value,
                Err(error) => {
                    if debug {
                        eprintln!("kouz LM rejected invalid trial: {error}");
                    }
                    mu *= 4;
                    continue;
                }
            };
            let mut r_trial_norm = Float::new_64(prec);
            let mut bad = false;
            for i in 1..n - 1 {
                let dd = Complex::with_val_64(prec, &f_trial[i] - &x_trial[i]);
                let m = cnum::abs(&dd, prec);
                if !m.is_finite() {
                    bad = true;
                    break;
                }
                if m > r_trial_norm {
                    r_trial_norm = m;
                }
            }

            if !bad && r_trial_norm < r_norm {
                if debug {
                    eprintln!(
                        "kouz LM accepted: residual {:.6e}, decrease {:.3e}",
                        DisplayFloat(&r_trial_norm),
                        DisplayFloat(&Float::with_val_64(prec, &r_norm - &r_trial_norm))
                    );
                }
                x = x_trial;
                // Successful step: shrink μ aggressively when we beat the
                // residual by a healthy margin (push toward Newton); shrink
                // mildly for marginal improvement.
                if r_trial_norm < Float::with_val_64(prec, &r_norm / 2) {
                    mu /= 4;
                } else {
                    mu *= cnum::decimal("0.7", prec);
                }
                mu = mu.max(&cnum::working_epsilon(prec));
                accepted = true;
                break;
            }
            // Reject: try a larger diagonal shift.
            mu *= 4;
        }
        if !accepted {
            // Failure to find descent is not an error bound. Any retained
            // candidate still needs the full final gate before evaluation.
            if debug {
                eprintln!(
                    "kouz LM: no descent step at iter {} (μ={:.2e}); best residual {:.3e}",
                    iter,
                    DisplayFloat(&mu),
                    DisplayFloat(&best_residual)
                );
            }
            return validate_best_residual(&best_residual, digits, use_schwarz, "no descent step")
                .map(|_| (best_x, best_residual));
        }
        if debug {
            eprintln!(
                "kouz LM iter {:>3}: total wall {:.2}s",
                iter,
                iter_start.elapsed().as_secs_f64()
            );
        }
        iter += 1;
    }
}

/// Retain a finite, bounded internal candidate after a stalled solve.
/// This filter is not an accuracy or canonicality certificate. Final answers
/// must independently satisfy the full requested boundary-residual target.
fn validate_best_residual(
    best_residual: &Float,
    digits: u64,
    use_schwarz: bool,
    reason: &str,
) -> Result<(), String> {
    let prec = best_residual.prec_64();
    let threshold = if use_schwarz {
        cnum::epsilon(digits / 3, prec)
            .max(&cnum::decimal("1e-6", prec))
            .min(&cnum::decimal("1e-3", prec))
    } else {
        Float::with_val_64(prec, 5)
    };
    if best_residual.is_finite() && *best_residual <= threshold {
        // Internal acceptance must not be mistaken for a usable final answer.
        let full_target = cnum::epsilon(digits.saturating_add(3), prec);
        if *best_residual > full_target && cnum::verbose() {
            eprintln!(
                "kouz: retaining an unconverged internal candidate ({reason}), residual \
                 {:.2e}; it cannot be returned as a {digits}-digit answer",
                DisplayFloat(best_residual)
            );
        }
        Ok(())
    } else {
        Err(format!(
            "Kouznetsov Newton did not converge ({reason}); best residual {:.3e} \
             exceeds acceptance threshold {:.3e} for {digits} digits",
            DisplayFloat(best_residual),
            DisplayFloat(&threshold)
        ))
    }
}

/// Build `J = I − DT` for the Cauchy operator. `DT[k][j]` is the partial
/// derivative of `T(F)[k]` with respect to `F[j]`, derived analytically from
/// the right-edge term `b^F[j]/(1.5+it_j−z_k)` and left-edge term
/// `log_b(F[j])/(−0.5+it_j−z_k)` (the top/bottom edges contribute constants
/// independent of `F`, so they don't enter the Jacobian).
///
/// **Now unused in production:** the GMRES path computes `J·v` matrix-free via
/// `apply_dt_v`. Kept as a reference implementation for testing and for the
/// `TET_KOUZ_DENSE=1` debug fallback (not yet wired up, but the logic is
/// straightforward should we need to compare against dense LU).
#[allow(dead_code)]
fn compute_jacobian_minus_dt(
    samples: &[Complex],
    nodes: &[Float],
    weights: &[Float],
    ln_b: &Complex,
    prec: u64,
) -> Vec<Vec<Complex>> {
    let n = nodes.len();
    let pi_f = Float::with_val_64(prec, rug::float::Constant::Pi);
    let two_pi = Float::with_val_64(prec, &pi_f * 2u32);

    // Precompute b^F[j]·ln(b) and 1/(F[j]·ln(b)), needed in every column.
    let mut b_f_ln: Vec<Complex> = Vec::with_capacity(n);
    let mut inv_f_ln: Vec<Complex> = Vec::with_capacity(n);
    for sample in &samples[..n] {
        let exp_arg = Complex::with_val_64(prec, ln_b * sample);
        let bf = Complex::with_val_64(prec, exp_arg.exp_ref());
        b_f_ln.push(Complex::with_val_64(prec, &bf * ln_b));
        let f_ln = Complex::with_val_64(prec, sample * ln_b);
        let one_c = Complex::with_val_64(prec, (Float::with_val_64(prec, 1u32), 0));
        inv_f_ln.push(Complex::with_val_64(prec, &one_c / &f_ln));
    }

    let one_re = Float::with_val_64(prec, 1u32);
    let neg_one_re = Float::with_val_64(prec, -1i32);
    let mut jac = vec![vec![cnum::zero(prec); n]; n];

    for k in 0..n {
        for j in 0..n {
            let dt_im = Float::with_val_64(prec, &nodes[j] - &nodes[k]);
            // 1+i(t_j-t_k)
            let denom_r = Complex::with_val_64(prec, (one_re.clone(), dt_im.clone()));
            // -1+i(t_j-t_k)
            let denom_l = Complex::with_val_64(prec, (neg_one_re.clone(), dt_im));

            let term_r = Complex::with_val_64(prec, &b_f_ln[j] / &denom_r);
            let term_l = Complex::with_val_64(prec, &inv_f_ln[j] / &denom_l);
            let bracket = Complex::with_val_64(prec, &term_r - &term_l);
            let scaled = Complex::with_val_64(prec, &bracket * &weights[j]);
            let dt_kj = Complex::with_val_64(prec, &scaled / &two_pi);

            // J = I − DT
            if k == j {
                jac[k][j] = Complex::with_val_64(prec, Complex::with_val_64(prec, (1, 0)) - &dt_kj);
            } else {
                jac[k][j] = Complex::with_val_64(prec, -&dt_kj);
            }
        }
    }
    jac
}

/// Solve a small dense complex linear system `A x = b` via Gaussian elimination
/// with partial pivoting. `A` is square. Returns Err on singular pivot.
fn solve_complex_lin(
    a_in: &[Vec<Complex>],
    b_in: &[Complex],
    prec: u64,
) -> Result<Vec<Complex>, String> {
    let n = a_in.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut a: Vec<Vec<Complex>> = a_in.to_vec();
    let mut b: Vec<Complex> = b_in.to_vec();

    for k in 0..n {
        // Partial pivot: find row with largest |a[i][k]| for i ≥ k.
        let mut pivot = k;
        let mut max_abs = Float::with_val_64(prec, a[k][k].abs_ref());
        for (i, row) in a.iter().enumerate().skip(k + 1) {
            let abs_i = Float::with_val_64(prec, row[k].abs_ref());
            if abs_i > max_abs {
                max_abs = abs_i;
                pivot = i;
            }
        }
        if !max_abs.is_finite() || max_abs.is_zero() {
            return Err(format!("singular at column {}", k));
        }
        if pivot != k {
            a.swap(k, pivot);
            b.swap(k, pivot);
        }

        // Eliminate below.
        for i in (k + 1)..n {
            let factor = Complex::with_val_64(prec, &a[i][k] / &a[k][k]);
            let (above, below) = a.split_at_mut(i);
            for (entry, pivot_entry) in below[0][k..].iter_mut().zip(&above[k][k..]) {
                let prod = Complex::with_val_64(prec, &factor * pivot_entry);
                *entry = Complex::with_val_64(prec, &*entry - &prod);
            }
            let prod = Complex::with_val_64(prec, &factor * &b[k]);
            b[i] = Complex::with_val_64(prec, &b[i] - &prod);
        }
    }

    // Back-substitute.
    let mut x = vec![cnum::zero(prec); n];
    for i in (0..n).rev() {
        let mut sum = b[i].clone();
        for j in (i + 1)..n {
            let prod = Complex::with_val_64(prec, &a[i][j] * &x[j]);
            sum = Complex::with_val_64(prec, &sum - &prod);
        }
        x[i] = Complex::with_val_64(prec, &sum / &a[i][i]);
    }
    Ok(x)
}

#[allow(clippy::too_many_arguments)]
fn eval_at_height(
    h: &Complex,
    samples: &[Complex],
    nodes: &[Float],
    weights: &[Float],
    t_max: &Float,
    l_upper: &Complex,
    l_lower: &Complex,
    ln_b: &Complex,
    prec: u64,
    two_sided: bool,
) -> Result<Complex, String> {
    eval_at_height_with_conditioning(
        h, samples, nodes, weights, t_max, l_upper, l_lower, ln_b, prec, two_sided,
    )
    .map(|(value, _)| value)
}

#[allow(clippy::too_many_arguments)]
fn eval_at_height_with_conditioning(
    h: &Complex,
    samples: &[Complex],
    nodes: &[Float],
    weights: &[Float],
    t_max: &Float,
    l_upper: &Complex,
    l_lower: &Complex,
    ln_b: &Complex,
    prec: u64,
    two_sided: bool,
) -> Result<(Complex, Float), String> {
    if !cnum::is_finite(h) || !t_max.is_finite() || *t_max <= 0 {
        return Err("Cauchy evaluation requires finite height and positive contour size".into());
    }
    let shift = h.real().clone().floor();
    let h_strip = Complex::with_val_64(prec, h - &shift);

    if h_strip.imag().clone().abs() >= *t_max {
        return Err("height lies on or outside the Cauchy contour; a fixed-point limit is not a finite-height value".into());
    }
    let f_strip = cauchy_eval(
        &h_strip, samples, nodes, weights, t_max, l_upper, l_lower, ln_b, prec, two_sided,
    )?;

    let mut f = f_strip;
    let forward = shift > 0;
    let shifts = shift.abs();
    let mut step = Integer::new();
    let mut log_amplification = Float::new_64(prec);
    let log_ln_b = log_magnitude(ln_b, prec);
    while shifts > step {
        if cnum::verbose() && (step == 0 || step.is_divisible_u(1024)) {
            eprintln!(
                "kouz height shift: step {step} of {:.8}",
                DisplayFloat(&shifts)
            );
        }
        if forward {
            let amplification = log_magnitude(&f, prec).max(&Float::new_64(prec)) + &log_ln_b;
            log_amplification += amplification.max(&Float::new_64(prec));
            let exp_arg = Complex::with_val_64(prec, ln_b * &f);
            f = cnum::checked_exp(&exp_arg, prec)?;
        } else {
            if !cnum::is_finite(&f) || cnum::is_zero(&f) {
                return Err("Cauchy logarithmic shift reached a zero or non-finite value".into());
            }
            let amplification = -log_magnitude(&f, prec).min(&Float::new_64(prec)) - &log_ln_b;
            log_amplification += amplification.max(&Float::new_64(prec));
            let ln_f = cnum::ln_complex(&f, prec);
            f = Complex::with_val_64(prec, &ln_f / ln_b);
        }
        step += 1;
    }
    if !cnum::is_finite(&f) {
        return Err("Cauchy evaluation produced a non-finite value".into());
    }
    log_amplification -= log_magnitude(&f, prec).min(&Float::new_64(prec));
    Ok((f, log_amplification))
}

// =====================================================================
// Newton-Krylov GMRES path
//
// Replaces the dense O(N³) LU factorization in iterate_newton with a
// matrix-free Krylov solve. The Jacobian J = I − DT is never materialised;
// we only need J·v for arbitrary v, which can be computed in O(N²) using
// the analytic form of DT plus a few precomputed factors.
//
// Cost comparison at N nodes, K Krylov dimension:
//   * Dense LU per LM try: O(N²) build + O(N³)/3 elimination.
//   * GMRES per LM try:    K matvecs each O(N²)  +  O(K²) for Givens.
// For N > a few hundred and K in [50, 200], GMRES is cheaper. The win
// grows linearly in N (since LU's N³ vs Krylov's K·N²), making
// 50-digit-precision regimes (N ≈ 15000) tractable when LU would not be.
// =====================================================================

/// Precompute the per-sample factors that appear in every column of DT:
///   b_f_ln[j]  = b^F[j] · ln(b)         (right-edge derivative)
///   inv_f_ln[j] = 1 / (F[j] · ln(b))    (left-edge derivative)
/// Depends only on `samples` (not on the input vector v of a matvec or on
/// the LM damping μ), so it can be hoisted outside both the GMRES loop and
/// the LM line search.
fn precompute_dt_factors(
    samples: &[Complex],
    ln_b: &Complex,
    prec: u64,
) -> Result<(Vec<Complex>, Vec<Complex>), String> {
    let one_c = Complex::with_val_64(prec, (Float::with_val_64(prec, 1u32), 0));
    let factors = |s: &Complex| -> Result<(Complex, Complex), String> {
        let exp_arg = Complex::with_val_64(prec, ln_b * s);
        let bf = cnum::checked_exp(&exp_arg, prec)?;
        let b_f_ln = Complex::with_val_64(prec, &bf * ln_b);
        let f_ln = Complex::with_val_64(prec, s * ln_b);
        let inv_f_ln = Complex::with_val_64(prec, &one_c / &f_ln);
        if !cnum::is_finite(&b_f_ln) || !cnum::is_finite(&inv_f_ln) {
            return Err("Cauchy derivative factors are non-finite".into());
        }
        Ok((b_f_ln, inv_f_ln))
    };
    let pairs: Result<Vec<_>, String> = if crate::mt::mt_enabled() {
        use rayon::prelude::*;
        samples.par_iter().with_min_len(8).map(factors).collect()
    } else {
        samples.iter().map(factors).collect()
    };
    Ok(pairs?.into_iter().unzip())
}

/// Precompute the per-(j−k) denominator inverses that appear in DT:
///   inv_denom_r[j−k+N−1] = 1 / (1 + i(t_j − t_k))
///   inv_denom_l[j−k+N−1] = 1 / (−1 + i(t_j − t_k))
///
/// On a uniform grid t_k = −T + k·δ, the differences `t_j − t_k = (j−k)·δ`
/// take only `2N−1` distinct values, so we compute and cache them once per
/// Cauchy iteration. Inside the matvec inner loop this turns 2 complex
/// divisions per (k, j) pair into 2 cache lookups + 2 complex multiplies,
/// roughly halving total cost.
#[allow(dead_code)]
fn precompute_denom_inverses(nodes: &[Float], prec: u64) -> (Vec<Complex>, Vec<Complex>) {
    let n = nodes.len();
    let mut inv_r: Vec<Complex> = Vec::with_capacity(2 * n - 1);
    let mut inv_l: Vec<Complex> = Vec::with_capacity(2 * n - 1);
    let one_re = Float::with_val_64(prec, 1u32);
    let neg_one_re = Float::with_val_64(prec, -1i32);
    let one_c = Complex::with_val_64(prec, (Float::with_val_64(prec, 1u32), 0));
    let delta = if n >= 2 {
        Float::with_val_64(prec, &nodes[1] - &nodes[0])
    } else {
        Float::with_val_64(prec, 1u32)
    };
    let max_offset = Integer::from(n - 1);
    for d_idx in 0..(2 * n - 1) {
        let d_signed = Integer::from(d_idx) - &max_offset;
        let dt = Float::with_val_64(prec, &delta * &d_signed);
        let dr = Complex::with_val_64(prec, (one_re.clone(), dt.clone()));
        let dl = Complex::with_val_64(prec, (neg_one_re.clone(), dt));
        inv_r.push(Complex::with_val_64(prec, &one_c / &dr));
        inv_l.push(Complex::with_val_64(prec, &one_c / &dl));
    }
    (inv_r, inv_l)
}

/// Bundle of FFT-domain kernels and per-row Euler-Maclaurin corrections for
/// the Cauchy operator on the uniform grid.
///
/// The FFT kernels are pure functions of the grid spacing — they don't depend
/// on the sample values F or on any Newton iterate — so we precompute their
/// FFTs once per `tetrate_kouznetsov` call and reuse them in every
/// `apply_t_fft` (residual evaluation) and `apply_dt_v_fft` (Jacobian-vector
/// product).
///
/// The EM corrections also depend only on the grid (and on the fixed-point
/// asymptote at `t=±T`, which is pinned), so they too are computed once per
/// call and used as a fixed additive shift in `apply_t_fft`. Because they
/// don't depend on F, they have zero derivative with respect to F and so do
/// not appear in the Jacobian (`apply_dt_v_fft` is unaffected).
pub(crate) struct CauchyKernels {
    pub right: KernelFft, // 1 / (1 + i·δ·d) for d in [-(N-1), N-1]
    pub left: KernelFft,  // 1 / (-1 + i·δ·d) for d in [-(N-1), N-1]
    /// Per-row right-edge EM correction divided by 2π so it's directly
    /// subtractable from `r_part[k]` in `apply_t_fft`. One entry per node
    /// `z₀ = 0.5 + i·t_k`.
    pub em_right_over_2pi: Vec<Complex>,
    /// Same for left edge.
    pub em_left_over_2pi: Vec<Complex>,
}

/// Build FFT-prepared kernels and per-row EM corrections.
/// FFT kernels: right edge `1 / (1 + i(t_j − t_k))`, left edge
/// `1 / (−1 + i(t_j − t_k))`. EM corrections close the trapezoidal floor by
/// subtracting the closed-form `O(h²)` boundary term.
pub(crate) fn build_cauchy_kernels(
    nodes: &[Float],
    t_max: &Float,
    l_upper: &Complex,
    l_lower: &Complex,
    prec: u64,
) -> Result<CauchyKernels, String> {
    let n = nodes.len();
    if n < 2 || !t_max.is_finite() || *t_max <= 0 || nodes.iter().any(|t| !t.is_finite()) {
        return Err("Cauchy kernels require finite nonempty geometry".into());
    }
    let one_re = Float::with_val_64(prec, 1u32);
    let neg_one_re = Float::with_val_64(prec, -1i32);
    let one_c = Complex::with_val_64(prec, (one_re.clone(), Float::new_64(prec)));
    let delta = if n >= 2 {
        Float::with_val_64(prec, &nodes[1] - &nodes[0])
    } else {
        Float::with_val_64(prec, 1u32)
    };
    let n_terms = em_n_terms(prec, &delta, l_upper, l_lower)?;
    let max_offset = Integer::from(n - 1);
    let mut h_r: Vec<Complex> = Vec::with_capacity(2 * n - 1);
    let mut h_l: Vec<Complex> = Vec::with_capacity(2 * n - 1);
    for d_idx in 0..(2 * n - 1) {
        let d_signed = Integer::from(d_idx) - &max_offset;
        let dt = Float::with_val_64(prec, &delta * &d_signed);
        let dr = Complex::with_val_64(prec, (one_re.clone(), dt.clone()));
        let dl = Complex::with_val_64(prec, (neg_one_re.clone(), dt));
        h_r.push(Complex::with_val_64(prec, &one_c / &dr));
        h_l.push(Complex::with_val_64(prec, &one_c / &dl));
    }
    let right = precompute_kernel_fft(&h_r, n, prec);
    let left = precompute_kernel_fft(&h_l, n, prec);

    // EM corrections per row.
    let em_h_powers = build_em_h_powers(&delta, n_terms, prec)?;

    let pi_f = Float::with_val_64(prec, Constant::Pi);
    let two_pi = Float::with_val_64(prec, &pi_f * 2u32);
    let inv_two_pi = Float::with_val_64(prec, Float::with_val_64(prec, 1u32) / &two_pi);

    let mut em_right_over_2pi = Vec::with_capacity(n);
    let mut em_left_over_2pi = Vec::with_capacity(n);
    for node in nodes {
        let z0 = Complex::with_val_64(prec, (cnum::decimal("0.5", prec), node.clone()));
        let (corr_r, corr_l) =
            compute_em_correction_z0(&z0, t_max, l_upper, l_lower, &em_h_powers, prec);
        em_right_over_2pi.push(Complex::with_val_64(prec, &corr_r * &inv_two_pi));
        em_left_over_2pi.push(Complex::with_val_64(prec, &corr_l * &inv_two_pi));
    }

    if cnum::verbose() {
        eprintln!(
            "kouz EM: K={} (h/T={:.3e})",
            n_terms,
            DisplayFloat(&Float::with_val_64(prec, &delta / t_max).abs())
        );
    }

    Ok(CauchyKernels {
        right,
        left,
        em_right_over_2pi,
        em_left_over_2pi,
    })
}

/// FFT-based Jacobian-vector product `(DT)·v`. Same math as the dense
/// `apply_dt_v` but with the inner O(N²) double loop replaced by two
/// length-`m` FFT-driven cross-correlations (`m = next_pow2(3N − 2)`).
pub(crate) fn apply_dt_v_fft(
    b_f_ln: &[Complex],
    inv_f_ln: &[Complex],
    kernels: &CauchyKernels,
    weights: &[Float],
    v: &[Complex],
    prec: u64,
) -> Vec<Complex> {
    let n = b_f_ln.len();
    let pi_f = Float::with_val_64(prec, Constant::Pi);
    let two_pi = Float::with_val_64(prec, &pi_f * 2u32);
    let one_re = Float::with_val_64(prec, 1u32);
    let inv_two_pi = Float::with_val_64(prec, &one_re / &two_pi);

    // Pre-scale: a_r[j] = b_f_ln[j]·v[j]·w_j/(2π), a_l[j] similar.
    let (a_r, a_l): (Vec<Complex>, Vec<Complex>) = if crate::mt::mt_enabled() {
        use rayon::prelude::*;
        (0..n)
            .into_par_iter()
            .with_min_len(8)
            .map(|j| {
                let w_over_2pi = Float::with_val_64(prec, &weights[j] * &inv_two_pi);
                let bv = Complex::with_val_64(prec, &b_f_ln[j] * &v[j]);
                let ar = Complex::with_val_64(prec, &bv * &w_over_2pi);
                let iv = Complex::with_val_64(prec, &inv_f_ln[j] * &v[j]);
                let al = Complex::with_val_64(prec, &iv * &w_over_2pi);
                (ar, al)
            })
            .unzip()
    } else {
        let mut a_r: Vec<Complex> = Vec::with_capacity(n);
        let mut a_l: Vec<Complex> = Vec::with_capacity(n);
        for j in 0..n {
            let w_over_2pi = Float::with_val_64(prec, &weights[j] * &inv_two_pi);
            let bv = Complex::with_val_64(prec, &b_f_ln[j] * &v[j]);
            a_r.push(Complex::with_val_64(prec, &bv * &w_over_2pi));
            let iv = Complex::with_val_64(prec, &inv_f_ln[j] * &v[j]);
            a_l.push(Complex::with_val_64(prec, &iv * &w_over_2pi));
        }
        (a_r, a_l)
    };

    let r_part = cross_correlate_with_kernel(&a_r, &kernels.right, prec);
    let l_part = cross_correlate_with_kernel(&a_l, &kernels.left, prec);

    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        out.push(Complex::with_val_64(prec, &r_part[k] - &l_part[k]));
    }
    out
}

/// FFT-based Cauchy operator `T(F)`. Same math as `apply_t` (which calls
/// `cauchy_eval` once per row) but with the `O(N²)` interior contributions
/// computed by two FFT cross-correlations and only the boundary corrections
/// (`l_upper · ln_top + l_lower · ln_bot`) computed per-row in `O(N)` total.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_t_fft(
    samples: &[Complex],
    nodes: &[Float],
    weights: &[Float],
    t_max: &Float,
    l_upper: &Complex,
    l_lower: &Complex,
    ln_b: &Complex,
    kernels: &CauchyKernels,
    prec: u64,
    two_sided: bool,
) -> Result<Vec<Complex>, String> {
    validate_cauchy_data(samples, nodes, weights, t_max, ln_b)?;
    let n = samples.len();
    let pi_f = Float::with_val_64(prec, Constant::Pi);
    let two_pi_f = Float::with_val_64(prec, &pi_f * 2u32);
    let two_pi_i = Complex::with_val_64(prec, (Float::new_64(prec), two_pi_f.clone()));
    let one_re = Float::with_val_64(prec, 1u32);
    let inv_two_pi = Float::with_val_64(prec, &one_re / &two_pi_f);

    // Build right- and left-edge values, pre-scaled by w_j / (2π) so the
    // cross-correlation result is the in-strip part of T(F).
    let ln_unwrapped = unwrapped_ln_samples(samples, l_upper, l_lower, ln_b, prec, two_sided);
    let edge_values = |j: usize| -> Result<(Complex, Complex), String> {
        let w_over_2pi = Float::with_val_64(prec, &weights[j] * &inv_two_pi);
        let exp_arg = Complex::with_val_64(prec, ln_b * &samples[j]);
        let bf = cnum::checked_exp(&exp_arg, prec)?;
        let ar = Complex::with_val_64(prec, &bf * &w_over_2pi);
        let log_b_s = Complex::with_val_64(prec, &ln_unwrapped[j] / ln_b);
        let al = Complex::with_val_64(prec, &log_b_s * &w_over_2pi);
        if !cnum::is_finite(&ar) || !cnum::is_finite(&al) {
            return Err("Cauchy edge integrand is non-finite".into());
        }
        Ok((ar, al))
    };
    let edges: Result<Vec<_>, String> = if crate::mt::mt_enabled() {
        use rayon::prelude::*;
        (0..n)
            .into_par_iter()
            .with_min_len(8)
            .map(edge_values)
            .collect()
    } else {
        (0..n).map(edge_values).collect()
    };
    let (a_r, a_l): (Vec<_>, Vec<_>) = edges?.into_iter().unzip();

    let r_part = cross_correlate_with_kernel(&a_r, &kernels.right, prec);
    let l_part = cross_correlate_with_kernel(&a_l, &kernels.left, prec);

    // Boundary corrections (per row, O(N) total).
    let cp1 = Complex::with_val_64(prec, (cnum::decimal("1.5", prec), 0));
    let cm1 = Complex::with_val_64(prec, (cnum::decimal("-0.5", prec), 0));
    let it_max = Complex::with_val_64(prec, (Float::new_64(prec), t_max.clone()));
    let neg_it_max = Complex::with_val_64(prec, -&it_max);
    let cm1_plus_itmax = Complex::with_val_64(prec, &cm1 + &it_max);
    let cp1_plus_itmax = Complex::with_val_64(prec, &cp1 + &it_max);
    let cp1_minus_itmax = Complex::with_val_64(prec, &cp1 + &neg_it_max);
    let cm1_minus_itmax = Complex::with_val_64(prec, &cm1 + &neg_it_max);

    // Per-row boundary correction: each row k is an independent computation
    // (2 complex ln + a handful of mults) reading shared precomputed values,
    // so the MT branch is a pure element-wise map — bit-identical.
    let row = |k: usize| -> Complex {
        let z0 = Complex::with_val_64(prec, (cnum::decimal("0.5", prec), nodes[k].clone()));

        // Euler-Maclaurin boundary correction: subtract the closed-form O(h²)
        // (and higher-order, up to K terms) error of the trapezoidal sums.
        // `em_*_over_2pi` is precomputed in `build_cauchy_kernels` and already
        // carries the 1/(2π) factor that `r_part`, `l_part` carry.
        let r_corrected = Complex::with_val_64(prec, &r_part[k] - &kernels.em_right_over_2pi[k]);
        let l_corrected = Complex::with_val_64(prec, &l_part[k] - &kernels.em_left_over_2pi[k]);
        let part1 = Complex::with_val_64(prec, &r_corrected - &l_corrected);

        let top_num = Complex::with_val_64(prec, &cm1_plus_itmax - &z0);
        let top_den = Complex::with_val_64(prec, &cp1_plus_itmax - &z0);
        let top_ratio = Complex::with_val_64(prec, &top_num / &top_den);
        let ln_top = cnum::ln_complex(&top_ratio, prec);

        let bot_num = Complex::with_val_64(prec, &cp1_minus_itmax - &z0);
        let bot_den = Complex::with_val_64(prec, &cm1_minus_itmax - &z0);
        let bot_ratio = Complex::with_val_64(prec, &bot_num / &bot_den);
        let ln_bot = cnum::ln_complex(&bot_ratio, prec);

        let up_term = Complex::with_val_64(prec, l_upper * &ln_top);
        let dn_term = Complex::with_val_64(prec, l_lower * &ln_bot);
        let upper_lower_sum = Complex::with_val_64(prec, &up_term + &dn_term);
        let part2 = Complex::with_val_64(prec, &upper_lower_sum / &two_pi_i);

        Complex::with_val_64(prec, &part1 + &part2)
    };

    let out: Vec<_> = if crate::mt::mt_enabled() {
        use rayon::prelude::*;
        (0..n).into_par_iter().with_min_len(8).map(row).collect()
    } else {
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            out.push(row(k));
        }
        out
    };
    if !out.iter().all(cnum::is_finite) {
        return Err("Cauchy FFT operator produced a non-finite result".into());
    }
    Ok(out)
}

/// Apply (DT)·v in O(N²) without ever forming DT. Uses precomputed per-sample
/// factors (`b_f_ln`, `inv_f_ln`) and per-offset denominator inverses
/// (`inv_denom_r`, `inv_denom_l`). Kept as the dense reference implementation;
/// production iterate_newton uses `apply_dt_v_fft` instead.
#[allow(dead_code)]
fn apply_dt_v(
    b_f_ln: &[Complex],
    inv_f_ln: &[Complex],
    inv_denom_r: &[Complex],
    inv_denom_l: &[Complex],
    weights: &[Float],
    v: &[Complex],
    prec: u64,
) -> Vec<Complex> {
    let n = b_f_ln.len();
    let pi_f = Float::with_val_64(prec, rug::float::Constant::Pi);
    let two_pi = Float::with_val_64(prec, &pi_f * 2u32);
    let one_re = Float::with_val_64(prec, 1u32);
    let inv_two_pi = Float::with_val_64(prec, &one_re / &two_pi);

    // Pre-scale: `bfl_v[j]` = b_f_ln[j]·v[j]·w_j/(2π), `ifl_v[j]` similar.
    // Pulling the scalar out of the inner k-loop saves N² multiplications.
    let mut bfl_v: Vec<Complex> = Vec::with_capacity(n);
    let mut ifl_v: Vec<Complex> = Vec::with_capacity(n);
    for j in 0..n {
        let w_over_2pi = Float::with_val_64(prec, &weights[j] * &inv_two_pi);
        let bv = Complex::with_val_64(prec, &b_f_ln[j] * &v[j]);
        bfl_v.push(Complex::with_val_64(prec, &bv * &w_over_2pi));
        let iv = Complex::with_val_64(prec, &inv_f_ln[j] * &v[j]);
        ifl_v.push(Complex::with_val_64(prec, &iv * &w_over_2pi));
    }

    let mut out = vec![cnum::zero(prec); n];
    for (k, value) in out.iter_mut().enumerate() {
        let mut acc = cnum::zero(prec);
        for j in 0..n {
            // d = j − k, indexed at d + (N − 1).
            let idx = j + (n - 1 - k);
            let term_r = Complex::with_val_64(prec, &bfl_v[j] * &inv_denom_r[idx]);
            let term_l = Complex::with_val_64(prec, &ifl_v[j] * &inv_denom_l[idx]);
            let dt_kj_v = Complex::with_val_64(prec, &term_r - &term_l);
            acc = Complex::with_val_64(prec, &acc + &dt_kj_v);
        }
        *value = acc;
    }
    out
}

/// L2 norm of a complex vector.
fn vector_norm_complex(v: &[Complex], prec: u64) -> Float {
    let mut norm = Float::new_64(prec);
    for c in v {
        let abs = Float::with_val_64(prec, c.abs_ref());
        norm = Float::with_val_64(prec, norm.hypot_ref(&abs));
    }
    norm
}

/// Hermitian inner product `<u, v> = Σ ū·v` (conjugate on first argument so
/// `<u, u>` is real and equals `‖u‖²`).
fn inner_product_complex(u: &[Complex], v: &[Complex], prec: u64) -> Complex {
    let mut s = cnum::zero(prec);
    for i in 0..u.len() {
        let conj_u = Complex::with_val_64(prec, u[i].conj_ref());
        let prod = Complex::with_val_64(prec, &conj_u * &v[i]);
        s = Complex::with_val_64(prec, &s + &prod);
    }
    s
}

/// Restarted GMRES with arbitrary-precision complex arithmetic.
///
/// Solves `A x = rhs` where `A` is given implicitly via a matvec closure.
/// Returns `Ok(x)` if the relative residual `‖rhs − A x‖ / ‖rhs‖` falls below
/// `tol_rel`. Restart on progress; enlarge a stalled Krylov window up to `n`.
///
/// Uses modified Gram-Schmidt Arnoldi and complex Givens rotations of the form
///   G = [[c, s], [−s̄, c]]   with `c ∈ ℝ≥0`, `s ∈ ℂ`, `c² + |s|² = 1`
/// where `c = |a|/τ`, `s = (a/|a|)·b̄/τ`, `τ = √(|a|²+|b|²)`. After each
/// rotation the (k+1, k) entry of the Hessenberg matrix is zeroed and the
/// reduced upper triangular system is solved by back-substitution at restart.
fn gmres_complex<F>(
    matvec: F,
    rhs: &[Complex],
    tol_rel: &Float,
    restart: usize,
    prec: u64,
) -> Result<Vec<Complex>, String>
where
    F: Fn(&[Complex]) -> Vec<Complex>,
{
    let n = rhs.len();
    let zero_c = cnum::zero(prec);
    let one_re = Float::with_val_64(prec, 1u32);
    let zero_re = Float::new_64(prec);

    let rhs_norm = vector_norm_complex(rhs, prec);
    if !rhs_norm.is_finite() || !tol_rel.is_finite() || *tol_rel <= 0 || *tol_rel >= 1 {
        return Err("GMRES requires finite data and 0 < relative tolerance < 1".into());
    }
    if rhs_norm.is_zero() {
        return Ok(vec![zero_c; n]);
    }
    let target_abs = Float::with_val_64(prec, &rhs_norm * tol_rel);
    if !target_abs.is_finite() || target_abs.is_zero() {
        return Err("GMRES target is outside the exponent range".into());
    }

    let mut x = vec![zero_c.clone(); n];
    let mut restart = restart.max(1).min(n);
    let mut outer = Integer::new();

    loop {
        cnum::check_complex_storage((restart as u128 + 1) * (n as u128 + restart as u128), prec)?;
        let ax = matvec(&x);
        if ax.len() != n || !ax.iter().all(cnum::is_finite) {
            return Err("GMRES matvec returned invalid values or dimensions".into());
        }
        let r: Vec<Complex> = rhs
            .iter()
            .zip(ax.iter())
            .map(|(bi, axi)| Complex::with_val_64(prec, bi - axi))
            .collect();
        let beta = vector_norm_complex(&r, prec);
        if !beta.is_finite() {
            return Err("GMRES residual norm is non-finite".into());
        }
        if beta < target_abs {
            return Ok(x);
        }
        let inv_beta = Float::with_val_64(prec, &one_re / &beta);
        let v0: Vec<Complex> = r
            .iter()
            .map(|c| {
                if inv_beta.is_finite() && !inv_beta.is_zero() {
                    Complex::with_val_64(prec, c * &inv_beta)
                } else {
                    Complex::with_val_64(prec, c / &beta)
                }
            })
            .collect();
        let mut basis: Vec<Vec<Complex>> = Vec::with_capacity(restart + 1);
        basis.push(v0);

        let mut h_mat: Vec<Vec<Complex>> = vec![vec![zero_c.clone(); restart]; restart + 1];
        let mut cs: Vec<Float> = vec![Float::with_val_64(prec, 1u32); restart];
        let mut sn: Vec<Complex> = vec![zero_c.clone(); restart];
        let mut g: Vec<Complex> = vec![zero_c.clone(); restart + 1];
        g[0] = Complex::with_val_64(prec, (beta.clone(), zero_re.clone()));

        let mut k_done = 0usize;

        for k in 0..restart {
            let mut w = matvec(&basis[k]);
            if w.len() != n || !w.iter().all(cnum::is_finite) {
                return Err("GMRES Arnoldi matvec returned invalid data".into());
            }
            // Modified Gram-Schmidt: orthogonalize w against basis[0..=k].
            for j in 0..=k {
                let h_jk = inner_product_complex(&basis[j], &w, prec);
                for i in 0..n {
                    let term = Complex::with_val_64(prec, &h_jk * &basis[j][i]);
                    w[i] = Complex::with_val_64(prec, &w[i] - &term);
                }
                h_mat[j][k] = h_jk;
            }
            let h_kp1_k_re = vector_norm_complex(&w, prec);
            if !h_kp1_k_re.is_finite() {
                return Err("GMRES Arnoldi norm is non-finite".into());
            }
            h_mat[k + 1][k] = Complex::with_val_64(prec, (h_kp1_k_re.clone(), zero_re.clone()));

            // Apply previously-stored Givens rotations to column k of H.
            for j in 0..k {
                let h1 = h_mat[j][k].clone();
                let h2 = h_mat[j + 1][k].clone();
                let term1 = Complex::with_val_64(prec, &h1 * &cs[j]);
                let term2 = Complex::with_val_64(prec, &sn[j] * &h2);
                h_mat[j][k] = Complex::with_val_64(prec, &term1 + &term2);
                let conj_sn = Complex::with_val_64(prec, sn[j].conj_ref());
                let term3 = Complex::with_val_64(prec, &conj_sn * &h1);
                let term4 = Complex::with_val_64(prec, &h2 * &cs[j]);
                h_mat[j + 1][k] = Complex::with_val_64(prec, &term4 - &term3);
            }

            // Construct new Givens rotation that zeros h_mat[k+1][k].
            let a = h_mat[k][k].clone();
            let bv = h_mat[k + 1][k].clone();
            let abs_a = Float::with_val_64(prec, a.abs_ref());
            let abs_b = Float::with_val_64(prec, bv.abs_ref());
            if !abs_a.is_finite() || !abs_b.is_finite() {
                return Err("GMRES Givens rotation is non-finite".into());
            }

            if abs_b.is_zero() {
                cs[k] = Float::with_val_64(prec, 1u32);
                sn[k] = zero_c.clone();
            } else if abs_a.is_zero() {
                cs[k] = Float::new_64(prec);
                let inv_abs_b = Float::with_val_64(prec, &one_re / &abs_b);
                let conj_b = Complex::with_val_64(prec, bv.conj_ref());
                sn[k] = if inv_abs_b.is_finite() && !inv_abs_b.is_zero() {
                    Complex::with_val_64(prec, &conj_b * &inv_abs_b)
                } else {
                    Complex::with_val_64(prec, &conj_b / &abs_b)
                };
            } else {
                let asq = Float::with_val_64(prec, &abs_a * &abs_a);
                let bsq = Float::with_val_64(prec, &abs_b * &abs_b);
                let sum_sq = Float::with_val_64(prec, &asq + &bsq);
                let norm = if sum_sq.is_finite() && !sum_sq.is_zero() {
                    Float::with_val_64(prec, sum_sq.sqrt_ref())
                } else {
                    Float::with_val_64(prec, abs_a.hypot_ref(&abs_b))
                };
                cs[k] = Float::with_val_64(prec, &abs_a / &norm);
                let inv_abs_a = Float::with_val_64(prec, &one_re / &abs_a);
                let alpha = if inv_abs_a.is_finite() && !inv_abs_a.is_zero() {
                    Complex::with_val_64(prec, &a * &inv_abs_a)
                } else {
                    Complex::with_val_64(prec, &a / &abs_a)
                };
                let conj_b = Complex::with_val_64(prec, bv.conj_ref());
                let alpha_conj_b = Complex::with_val_64(prec, &alpha * &conj_b);
                let inv_norm = Float::with_val_64(prec, &one_re / &norm);
                sn[k] = if inv_norm.is_finite() && !inv_norm.is_zero() {
                    Complex::with_val_64(prec, &alpha_conj_b * &inv_norm)
                } else {
                    Complex::with_val_64(prec, &alpha_conj_b / &norm)
                };
            }

            // Apply the new rotation to column k.
            let term_a = Complex::with_val_64(prec, &a * &cs[k]);
            let term_b = Complex::with_val_64(prec, &sn[k] * &bv);
            h_mat[k][k] = Complex::with_val_64(prec, &term_a + &term_b);
            h_mat[k + 1][k] = zero_c.clone();

            // Apply the new rotation to g (right-hand side after rotations).
            let g_k = g[k].clone();
            let new_g_k = Complex::with_val_64(prec, &g_k * &cs[k]);
            let conj_sn_k = Complex::with_val_64(prec, sn[k].conj_ref());
            let neg_term = Complex::with_val_64(prec, &conj_sn_k * &g_k);
            let new_g_kp1 = Complex::with_val_64(prec, -&neg_term);
            g[k] = new_g_k;
            g[k + 1] = new_g_kp1;

            k_done = k + 1;

            let resid_est = cnum::abs(&g[k + 1], prec);
            if !resid_est.is_finite() {
                return Err("GMRES residual estimate is non-finite".into());
            }
            if cnum::verbose() && (k % 8 == 0 || k_done == restart || resid_est < target_abs) {
                eprintln!("kouz GMRES cycle {outer}: Krylov {k_done}/{restart}, residual estimate {:.4e}, target {:.3e}",
                    DisplayFloat(&resid_est), DisplayFloat(&target_abs));
            }
            if resid_est < target_abs {
                break;
            }
            if h_kp1_k_re.is_zero() {
                // Lucky breakdown: Krylov subspace is invariant; current
                // y solves the system exactly within this subspace.
                break;
            }

            let inv_h = Float::with_val_64(prec, &one_re / &h_kp1_k_re);
            let v_next: Vec<Complex> = w
                .iter()
                .map(|c| {
                    if inv_h.is_finite() && !inv_h.is_zero() {
                        Complex::with_val_64(prec, c * &inv_h)
                    } else {
                        Complex::with_val_64(prec, c / &h_kp1_k_re)
                    }
                })
                .collect();
            basis.push(v_next);
        }

        // Solve the (k_done × k_done) upper-triangular system H y = g.
        let mut y = vec![zero_c.clone(); k_done];
        for i in (0..k_done).rev() {
            let mut sum = g[i].clone();
            for j in (i + 1)..k_done {
                let prod = Complex::with_val_64(prec, &h_mat[i][j] * &y[j]);
                sum = Complex::with_val_64(prec, &sum - &prod);
            }
            // Defensive: pivot can become tiny if A is rank-deficient on the
            // Krylov subspace; in that case the back-sub blows up. Bail on
            // the restart and let the outer loop retry from the new x.
            let pivot_abs = cnum::abs(&h_mat[i][i], prec);
            if pivot_abs.is_zero() || !pivot_abs.is_finite() {
                return Err(format!(
                    "GMRES: zero pivot at row {} in Hessenberg solve",
                    i
                ));
            }
            y[i] = Complex::with_val_64(prec, &sum / &h_mat[i][i]);
        }

        // x ← x + V_k · y.
        for j in 0..k_done {
            for i in 0..n {
                let term = Complex::with_val_64(prec, &basis[j][i] * &y[j]);
                x[i] = Complex::with_val_64(prec, &x[i] + &term);
            }
        }

        let ax = matvec(&x);
        if ax.len() != n || !ax.iter().all(cnum::is_finite) {
            return Err("GMRES final matvec returned invalid data".into());
        }
        let residual: Vec<_> = rhs
            .iter()
            .zip(&ax)
            .map(|(b, a)| Complex::with_val_64(prec, b - a))
            .collect();
        let actual_norm = vector_norm_complex(&residual, prec);
        if !actual_norm.is_finite() {
            return Err("GMRES actual residual is non-finite".into());
        }
        if actual_norm <= target_abs {
            return Ok(x);
        }
        if cnum::verbose() {
            eprintln!(
                "kouz GMRES cycle {outer}: actual residual {:.6e}",
                DisplayFloat(&actual_norm)
            );
        }
        if actual_norm >= beta {
            if restart == n {
                return Err(format!(
                    "GMRES stagnation at full Krylov dimension {n}, residual {}",
                    DisplayFloat(&actual_norm)
                ));
            }
            restart = restart.saturating_mul(2).min(n);
            if cnum::verbose() {
                eprintln!("kouz GMRES: increasing stalled Krylov window to {restart}");
            }
        }
        outer += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Continuation-based solver for near-parabolic bases (Class A)
// ─────────────────────────────────────────────────────────────────────────────

/// Resample a converged Kouznetsov solution onto a new uniform grid.
///
/// Old grid: `old_n` nodes uniformly spaced over `[−old_t_max, +old_t_max]`.
/// New grid: `new_nodes` (may be wider/denser than the old one).
///
/// Cauchy resampling is only a warm start. Beyond the old contour, fixed-point
/// seeds are allowed here, never as final finite-height answers.
fn resample_to_grid(
    old: &KouznetsovState,
    new_nodes: &[Float],
    l_upper: &Complex,
    l_lower: &Complex,
    prec: u64,
) -> Result<Vec<Complex>, String> {
    if old.samples.len() < 3
        || old.samples.len() != old.nodes.len()
        || old.samples.len() != old.weights.len()
    {
        return Err("Cauchy resampling requires a valid source grid".into());
    }
    let eval = |t: &Float| {
        let value = if t.clone().abs() >= old.t_max {
            if t.is_sign_negative() {
                l_lower.clone()
            } else {
                l_upper.clone()
            }
        } else {
            let height = Complex::with_val_64(prec, (cnum::decimal("0.5", prec), t));
            cauchy_eval(
                &height,
                &old.samples,
                &old.nodes,
                &old.weights,
                &old.t_max,
                &old.l_upper,
                &old.l_lower,
                &old.ln_b,
                prec,
                old.two_sided,
            )?
        };
        if !t.is_finite() || !cnum::is_finite(&value) {
            Err("Cauchy resampling produced a non-finite seed".into())
        } else {
            Ok(value)
        }
    };
    if crate::mt::mt_enabled() {
        use rayon::prelude::*;
        new_nodes.par_iter().map(eval).collect()
    } else {
        new_nodes.iter().map(eval).collect()
    }
}

/// Continuation-based Kouznetsov solver for near-parabolic real bases.
///
/// Step from a larger real base toward the exact target, using Cauchy
/// resampling for warm starts. This is not a convergence guarantee; resampling
/// is quadratic in the grid sizes and can dominate high-precision runs.
/// Only real `b > e^(1/e)` is supported by this continuation path.
pub fn setup_kouznetsov_continuation(
    b_target: &Complex,
    fp_target: &FixedPointData,
    prec: u64,
    digits: u64,
) -> Result<KouznetsovState, String> {
    crate::mt::init_pool()?;
    cnum::require_precision(prec, digits)?;
    if !cnum::is_finite(b_target) || !b_target.imag().is_zero() || digits == 0 {
        return Err(
            "continuation solver requires a finite real base and positive precision".into(),
        );
    }
    let b_re_target = b_target.real().clone();
    let eta = cnum::eta_upper(prec);
    if b_re_target <= eta {
        return Err(format!(
            "continuation: b={:.5} ≤ η={:.5}",
            DisplayFloat(&b_re_target),
            DisplayFloat(&eta)
        ));
    }

    let arg_lambda_tgt = arg_abs(&fp_target.lambda, prec);
    let t_max_tgt = contour_height(digits, &arg_lambda_tgt, prec)?;
    let n_bulk_tgt = pick_node_count(digits, &t_max_tgt, prec)?;
    crate::fft::kernel_fft_len(n_bulk_tgt, prec)?;

    // Start farther from the parabolic boundary for the cold solve.
    // b=1.75 has |arg(λ)|≈0.9, giving n_nodes≈4096 at 20 digits.
    let b_start = (b_re_target.clone() + cnum::decimal("0.35", prec))
        .max(&cnum::decimal("1.75", prec))
        .min(&cnum::decimal("2.5", prec));

    // Limit base increments to 0.025; this does not guarantee convergence.
    let mut n_steps = (Float::with_val_64(prec, &b_start - &b_re_target).abs()
        / cnum::decimal("0.025", prec))
    .ceil();
    n_steps += 1;
    n_steps = n_steps.max(&Float::with_val_64(prec, 2));
    if !n_steps.is_finite() {
        return Err("continuation step count exceeds MPFR's exponent range".into());
    }

    if cnum::verbose() {
        eprintln!(
            "kouz continuation: b_target={:.5}  b_start={:.5}  n_steps={}",
            DisplayFloat(&b_re_target),
            DisplayFloat(&b_start),
            DisplayFloat(&n_steps)
        );
    }

    let mut prev_state: Option<KouznetsovState> = None;

    let mut step = Integer::new();
    while n_steps >= step {
        let frac = Float::with_val_64(prec, &step) / &n_steps;
        let b_step = if n_steps == step {
            b_re_target.clone()
        } else {
            b_start.clone() + Float::with_val_64(prec, &b_re_target - &b_start) * frac
        };
        let b_cplx = if n_steps == step {
            Complex::with_val_64(prec, b_target)
        } else {
            Complex::with_val_64(prec, (&b_step, 0))
        };

        // Compute fixed-point pair for this b value.
        let ln_b = cnum::ln_complex(&b_cplx, prec);
        let neg_ln_b = Complex::with_val_64(prec, -&ln_b);
        let w0_val = lambertw::w0(&neg_ln_b, prec)
            .map_err(|e| format!("continuation step {}: W₀ failed: {}", step, e))?;
        let l_raw = Complex::with_val_64(prec, -w0_val / &ln_b);
        // Ensure l_upper has Im > 0 (the convention for Schwarz-symmetric bases).
        let l_upper_step = if l_raw.imag().is_sign_negative() {
            Complex::with_val_64(prec, l_raw.conj_ref())
        } else {
            l_raw
        };
        let l_lower_step = Complex::with_val_64(prec, l_upper_step.conj_ref());
        let lambda_upper = Complex::with_val_64(prec, &ln_b * &l_upper_step);
        let arg_lambda = arg_abs(&lambda_upper, prec);

        let t_max_fp = contour_height(digits, &arg_lambda, prec)?;
        let n_nodes = pick_node_count(digits, &t_max_fp, prec)?;

        crate::fft::kernel_fft_len(n_nodes, prec)?;

        let nodes = build_uniform_nodes(&t_max_fp, n_nodes, prec);
        let weights = build_trapezoidal_weights(&t_max_fp, n_nodes, prec);

        if cnum::verbose() {
            eprintln!(
                "kouz cont step {}/{}: b={:.5}  |arg(λ)|={:.4}  t_max={:.1}  n={}  warm={}",
                step,
                DisplayFloat(&n_steps),
                DisplayFloat(&b_step),
                DisplayFloat(&arg_lambda),
                DisplayFloat(&t_max_fp),
                n_nodes,
                prev_state.is_some()
            );
        }

        let initial = if let Some(ref prev) = prev_state {
            // Resample previous solution onto the new (potentially wider/denser) grid.
            let mut init = resample_to_grid(prev, &nodes, &l_upper_step, &l_lower_step, prec)?;
            // Always re-pin the boundary samples (they may have been approximated
            // as L± during resampling, but the exact new L values differ slightly).
            init[0] = l_lower_step.clone();
            init[n_nodes - 1] = l_upper_step.clone();
            // Real base: symmetrize so the solver stays on the Schwarz manifold.
            symmetrize_schwarz(&mut init, prec);
            init
        } else {
            // Cold start for the first step.
            let fp_step = FixedPointData {
                fixed_point: l_upper_step.clone(),
                lambda: lambda_upper.clone(),
                lambda_abs: Float::with_val_64(prec, lambda_upper.abs_ref()),
            };
            let cold = setup_kouznetsov(&b_cplx, &fp_step, prec, digits)?;
            // Resample cold solution onto `nodes` in case step-0 geometry differs.
            let mut init = resample_to_grid(&cold, &nodes, &l_upper_step, &l_lower_step, prec)?;
            init[0] = l_lower_step.clone();
            init[n_nodes - 1] = l_upper_step.clone();
            symmetrize_schwarz(&mut init, prec);
            init
        };

        // Run LM with the warm (or cold-resampled) initial.
        let (samples, step_residual) = iterate_newton(
            initial,
            &nodes,
            &weights,
            &t_max_fp,
            &l_upper_step,
            &l_lower_step,
            &ln_b,
            prec,
            digits,
            true,  // use_schwarz = true for real b
            false, // principal log: continuation stays on historical operator
        )?;

        let shift = find_normalization_shift(
            &samples,
            &nodes,
            &weights,
            &t_max_fp,
            &l_upper_step,
            &l_lower_step,
            &ln_b,
            prec,
            digits,
            true, // continuation solver only supports real b → real shift
            false,
        )?;

        prev_state = Some(KouznetsovState {
            samples,
            nodes,
            weights,
            t_max: t_max_fp,
            l_upper: l_upper_step,
            l_lower: l_lower_step,
            ln_b,
            shift,
            prec,
            digits,
            normalized: true,
            residual: step_residual,
            two_sided: false,
        });
        step += 1;
    }

    prev_state.ok_or_else(|| "continuation: no state produced".into())
}

/// Serialize a mid-walk continuation state with full precision and atomic rename.
/// Requested checkpoint failures are errors, not silent loss of resumable work.
fn save_cut_ckpt(
    path: &str,
    b_re: &Float,
    digits: u64,
    eps_cur: &Float,
    arg_up: &Float,
    arg_low: &Float,
    state: &KouznetsovState,
) -> Result<(), String> {
    use std::io::Write;
    let f = cnum::format_float_roundtrip;
    let tmp = format!("{}.tmp-{}", path, std::process::id());
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .map_err(|e| format!("create checkpoint temporary file {}: {}", tmp, e))?;
    let result = (|| -> std::io::Result<()> {
        let mut out = std::io::BufWriter::new(file);
        writeln!(out, "TETCKPT2\n{}\n{digits}\n{}", f(b_re), state.prec)?;
        writeln!(
            out,
            "{} {} {} {}",
            f(eps_cur),
            f(arg_up),
            f(arg_low),
            f(&state.residual)
        )?;
        writeln!(out, "{}", f(&state.t_max))?;
        for value in [&state.l_upper, &state.l_lower, &state.ln_b] {
            writeln!(out, "{} {}", f(value.real()), f(value.imag()))?;
        }
        writeln!(out, "{}", state.samples.len())?;
        for i in 0..state.samples.len() {
            writeln!(
                out,
                "{}\t{}\t{}\t{}",
                f(&state.nodes[i]),
                f(&state.weights[i]),
                f(state.samples[i].real()),
                f(state.samples[i].imag())
            )?;
        }
        out.flush()?;
        out.get_ref().sync_all()?;
        std::fs::rename(&tmp, path)
    })();
    if let Err(e) = result {
        if let Err(cleanup) = std::fs::remove_file(&tmp) {
            return Err(format!(
                "checkpoint save failed ({e}); cleanup of {tmp} also failed ({cleanup})"
            ));
        }
        return Err(format!("checkpoint save to {path} failed: {e}"));
    }
    if cnum::verbose() {
        eprintln!(
            "kouz cut-base walk: checkpoint saved at epsilon={} ({} nodes) -> {}",
            DisplayFloat(eps_cur),
            state.samples.len(),
            path
        );
    }
    Ok(())
}

/// Load an exactly matching full-precision warm state; only a missing file is None.
fn load_cut_ckpt(
    path: &str,
    b_re: &Float,
    digits: u64,
    prec: u64,
) -> Result<Option<(Float, Float, Float, KouznetsovState)>, String> {
    let txt = match std::fs::read_to_string(path) {
        Ok(txt) => txt,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("cannot read checkpoint {path}: {e}")),
    };
    let decoded = (|| -> Option<(Float, Float, Float, KouznetsovState)> {
    let mut lines = txt.lines();
    if lines.next()? != "TETCKPT2" {
        return None;
    }
    let pf = |s: &str| -> Option<Float> {
        cnum::parse_float(s.trim(), prec).ok()
    };
    let b_stored = pf(lines.next()?)?;
    if b_stored != *b_re {
        return None;
    }
    let d_stored: u64 = lines.next()?.trim().parse().ok()?;
    if d_stored != digits {
        return None;
    }
    let p_stored: u64 = lines.next()?.trim().parse().ok()?;
    if p_stored != prec {
        return None;
    }
    let scalars: Vec<Float> = lines
        .next()?
        .split_whitespace()
        .map(pf).collect::<Option<Vec<_>>>()?;
    if scalars.len() != 4 {
        return None;
    }
    let eps_cur = scalars[0].clone();
    let arg_up = scalars[1].clone();
    let arg_low = scalars[2].clone();
    let residual = scalars[3].clone();
    if eps_cur <= 0 || residual < 0 {
        return None;
    }
    let t_max = pf(lines.next()?)?;
    if t_max <= 0 {
        return None;
    }
    let mut pc = || -> Option<Complex> {
        let ln = lines.next()?;
        let mut it = ln.split_whitespace();
        let re = pf(it.next()?)?;
        let im = pf(it.next()?)?;
        if it.next().is_some() {
            return None;
        }
        Some(Complex::with_val_64(prec, (re, im)))
    };
    let l_upper = pc()?;
    let l_lower = pc()?;
    let ln_b = pc()?;
    let n: usize = lines.next()?.trim().parse().ok()?;
    if n < 4 || lines.clone().count() != n {
        return None;
    }
    crate::fft::kernel_fft_len(n, prec).ok()?;
    cnum::check_complex_storage(n as u128 * 2, prec).ok()?;
    let mut nodes = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let ln = lines.next()?;
        let mut it = ln.split('\t');
        nodes.push(pf(it.next()?)?);
        weights.push(pf(it.next()?)?);
        let re = pf(it.next()?)?;
        let im = pf(it.next()?)?;
        if it.next().is_some() {
            return None;
        }
        samples.push(Complex::with_val_64(prec, (re, im)));
    }
    let base = Complex::with_val_64(prec, (b_re, &eps_cur));
    if lines.next().is_some() || nodes != build_uniform_nodes(&t_max, n, prec)
        || weights != build_trapezoidal_weights(&t_max, n, prec)
        || samples.first()? != &l_lower || samples.last()? != &l_upper
        || ln_b != cnum::ln_complex(&base, prec)
    {
        return None;
    }
    let state = KouznetsovState {
        samples,
        nodes,
        weights,
        t_max,
        l_upper,
        l_lower,
        ln_b,
        shift: Complex::with_val_64(prec, (Float::new_64(prec), Float::new_64(prec))),
        prec,
        digits,
        normalized: false,
        residual,
        two_sided: true,
    };
    Some((eps_cur, arg_up, arg_low, state))
    })().ok_or_else(|| format!(
        "checkpoint {path} is malformed, non-finite, incompatible (TETCKPT2 required), or does not exactly match base/precision/geometry"
    ))?;
    Ok(Some(decoded))
}

fn cut_schedule(
    start: &Float,
    ratio: &Float,
    prec: u64,
) -> Result<std::collections::VecDeque<Float>, String> {
    if !start.is_finite() || !ratio.is_finite() || *start <= 0 || *ratio <= 0 || *ratio >= 1 {
        return Err("cut-base schedule requires a positive anchor and 0 < ratio < 1".into());
    }
    let mut queue = std::collections::VecDeque::new();
    let endpoint = cnum::decimal("1e-3", prec);
    if *start > endpoint {
        let count = ((endpoint.clone().ln() - start.clone().ln()) / ratio.clone().ln()).ceil();
        let count =
            cnum::checked_usize(&count).ok_or("cut-base schedule exceeds addressable memory")?;
        std::alloc::Layout::array::<Float>(count)
            .map_err(|_| "cut-base schedule exceeds addressable memory")?;
        cnum::check_float_storage(count as u128, prec)?;
    }
    let mut epsilon = Float::with_val_64(prec, start * ratio);
    while epsilon > endpoint {
        queue.push_back(epsilon.clone());
        let next = Float::with_val_64(prec, &epsilon * ratio);
        if next >= epsilon {
            return Err("cut-base schedule cannot decrease epsilon at this precision".into());
        }
        epsilon = next;
    }
    queue.push_back(Float::new_64(prec));
    Ok(queue)
}

/// Experimental upper-half-plane continuation for real `0 < b < e^{-e}`.
///
/// The intended target is `lim_{ε→0+} F(b+iε, h)`, where that limit exists.
/// Start at `b+2i`, germ-track the fixed-point pair, and Cauchy-resample warm
/// states while decreasing ε. Failed steps are bisected. Historical walks
/// made partial progress, but did not establish a normalized endpoint or
/// uniqueness of this branch; see FAILURE_CASES.md J.
///
/// Distinct fixed points and decaying local modes do not prove global
/// solvability of the contour equation. A final ε=0 state must pass the full
/// requested residual/normalization gates and is still not an error certificate.
pub fn setup_kouznetsov_cut_base(
    b_re: &Float,
    prec: u64,
    digits: u64,
) -> Result<KouznetsovState, String> {
    crate::mt::init_pool()?;
    cnum::require_precision(prec, digits)?;
    if !b_re.is_finite() || *b_re <= 0 || *b_re >= cnum::eta_lower(prec) || digits == 0 {
        return Err(format!(
            "cut-base solve: Re b = {:.6} is not in the cut band (0, e^-e)",
            DisplayFloat(b_re)
        ));
    }

    let verbose = cnum::verbose();
    let base_at = |eps: &Float| -> Complex { Complex::with_val_64(prec, (b_re, eps)) };
    let arg_of = |z: &Complex| -> Float { Float::with_val_64(prec, z.arg_ref()) };
    let dist = |a: &Complex, c: &Complex| -> Float {
        let d = Complex::with_val_64(prec, a - c);
        cnum::abs(&d, prec)
    };
    // Walk-continuous argument: principal arg shifted by 2πk to land within
    // π of the previous step's value (germ consistency across the ±π seam —
    // at ε = 0, λ_up is real negative and the one-sided limit is arg = +π).
    let two_pi = Float::with_val_64(prec, Constant::Pi) * 2u32;
    let arg_cont = |z: &Complex, prev: &Float| -> Float {
        let a = arg_of(z);
        let turns = (Float::with_val_64(prec, prev - &a) / &two_pi).round();
        a + turns * &two_pi
    };

    // ---- Anchor: cold solve at b + iε_anchor (outside Shell–Thron) ----
    let eps_anchor = cnum::env_float("TET_KOUZ_CUT_ANCHOR", "2", prec)?;
    let sched_ratio = cnum::env_float("TET_KOUZ_CUT_RATIO", "0.72", prec)?;
    let anchor_schedule = cut_schedule(&eps_anchor, &sched_ratio, prec)?;
    // ---- Checkpoint/resume (TET_KOUZ_CUT_CKPT=path) ----
    // Walks run for hours and used to lose everything to an external
    // timeout/power cut (observed twice: walk13 lost ~7 h at ε≈1.006, an
    // earlier session lost a record run to a host power failure). With the
    // env set, every accepted step persists (ε, pair, args, full samples) via
    // atomic tmp+rename; a restart with the same env warm-resumes from the
    // saved frontier exactly like a just-solved step, skipping the anchor.
    let ckpt_path = match std::env::var("TET_KOUZ_CUT_CKPT") {
        Ok(path) if !path.is_empty() => Some(path),
        Ok(_) => return Err("TET_KOUZ_CUT_CKPT requires a nonempty path".into()),
        Err(std::env::VarError::NotPresent) => None,
        Err(e) => return Err(format!("TET_KOUZ_CUT_CKPT: {e}")),
    };
    let resumed = match ckpt_path.as_deref() {
        Some(path) => load_cut_ckpt(path, b_re, digits, prec)?,
        None => None,
    };
    let resuming = resumed.is_some();
    let mut state;
    let mut eps_cur;
    let mut l_up;
    let mut l_low;
    let mut arg_up;
    let mut arg_low;
    if let Some((r_eps, r_up, r_low, r_state)) = resumed {
        if verbose {
            eprintln!(
                "kouz cut-base walk: RESUMED from checkpoint at ε={:.6e} ({} nodes, residual {:.3e})",
                DisplayFloat(&r_eps),
                r_state.samples.len(),
                DisplayFloat(&r_state.residual),
            );
        }
        l_up = r_state.l_upper.clone();
        l_low = r_state.l_lower.clone();
        arg_up = r_up;
        arg_low = r_low;
        eps_cur = r_eps;
        state = r_state;
    } else {
        let b_anchor = base_at(&eps_anchor);
        let ln_b_anchor = cnum::ln_complex(&b_anchor, prec);
        let neg_ln_b_anchor = Complex::with_val_64(prec, -&ln_b_anchor);
        let w0_val = lambertw::w0(&neg_ln_b_anchor, prec)?;
        let neg_w0 = Complex::with_val_64(prec, -&w0_val);
        l_up = Complex::with_val_64(prec, &neg_w0 / &ln_b_anchor);
        let w1_val = lambertw::wk(&neg_ln_b_anchor, 1, prec)?;
        let neg_w1 = Complex::with_val_64(prec, -&w1_val);
        l_low = Complex::with_val_64(prec, &neg_w1 / &ln_b_anchor);

        arg_up = arg_of(&Complex::with_val_64(prec, &ln_b_anchor * &l_up));
        arg_low = arg_of(&Complex::with_val_64(prec, &ln_b_anchor * &l_low));
        if !(arg_up > 0 && arg_low < 0) {
            return Err(format!(
                "cut-base walk: anchor pair at b+{}i is not decay-compatible (arg λ_up={:.4}, arg λ_low={:.4})",
                DisplayFloat(&eps_anchor), DisplayFloat(&arg_up), DisplayFloat(&arg_low)
            ));
        }
        if verbose {
            eprintln!(
                "kouz cut-base walk: anchor ε={}  L_up={:.4}+{:.4}i (argλ={:+.4})  L_low(W₊₁)={:.4}+{:.4}i (argλ={:+.4})",
                DisplayFloat(&eps_anchor),
                DisplayFloat(l_up.real()),
                DisplayFloat(l_up.imag()),
                DisplayFloat(&arg_up),
                DisplayFloat(l_low.real()),
                DisplayFloat(l_low.imag()),
                DisplayFloat(&arg_low),
            );
        }
        state = setup_kouznetsov_core(
            &b_anchor,
            l_up.clone(),
            l_low.clone(),
            prec,
            digits,
            false,
            None,
            false,
            true,
            true, // two-sided anchored unwrap: load-bearing for the cut walk
            1,
        )
        .map_err(|e| {
            format!(
                "cut-base walk: anchor solve at b+{}i failed: {}",
                DisplayFloat(&eps_anchor),
                e
            )
        })?;
        eps_cur = eps_anchor;
    }

    // ---- Geometric descent schedule, then the final hop to ε = 0 ----
    let mut queue = if resuming {
        cut_schedule(&eps_cur, &sched_ratio, prec)?
    } else {
        anchor_schedule
    };

    let mut solves = Integer::new();
    let mut rescue_pattern: Option<Vec<i32>> = None;
    let mut steps_since_rescue = Integer::from(7);
    while let Some(eps_next) = queue.pop_front() {
        solves += 1;

        let b_next = base_at(&eps_next);
        let ln_b_next = cnum::ln_complex(&b_next, prec);

        // Germ-track the pair by Newton from the previous values; W-branch
        // labels are meaningless mid-walk (L_low crosses Im = 0 near ε≈0.7).
        let track = |seed: &Complex| -> Result<Complex, String> {
            newton_fixed_point(&ln_b_next, seed, prec)
        };
        let step_fail =
            |msg: String,
             queue: &mut std::collections::VecDeque<Float>,
             eps_cur: &Float,
             eps_next: &Float|
             -> Result<(), String> {
                // Bisect: geometric mean for interior steps, arithmetic halving
                // for the final hop to 0. Give up when the interval collapses.
                let mid = if *eps_next > 0 {
                    let product = Float::with_val_64(prec, eps_cur * eps_next);
                    if product.is_finite() && !product.is_zero() {
                        product.sqrt()
                    } else {
                        Float::with_val_64(prec, eps_cur.sqrt_ref())
                            * Float::with_val_64(prec, eps_next.sqrt_ref())
                    }
                } else {
                    Float::with_val_64(prec, eps_cur / 2)
                };
                if !mid.is_finite() || mid >= *eps_cur || mid <= *eps_next {
                    return Err(format!(
                        "cut-base walk: stuck at ε={:.6e} → {:.6e} ({}); bisection floor reached",
                        DisplayFloat(eps_cur),
                        DisplayFloat(eps_next),
                        msg
                    ));
                }
                if cnum::verbose() {
                    eprintln!(
                    "kouz cut-base walk: step ε={:.6e} → {:.6e} failed ({}); bisecting at ε={:.6e}",
                    DisplayFloat(eps_cur), DisplayFloat(eps_next), msg, DisplayFloat(&mid)
                );
                }
                queue.push_front(eps_next.clone());
                queue.push_front(mid);
                Ok(())
            };

        let (l_up_next, l_low_next) = match (track(&l_up), track(&l_low)) {
            (Ok(u), Ok(l)) => (u, l),
            (u, l) => {
                let msg = format!(
                    "fixed-point tracking failed: up={:?} low={:?}",
                    u.err(),
                    l.err()
                );
                step_fail(msg, &mut queue, &eps_cur, &eps_next)?;
                continue;
            }
        };

        // Sanity guards: bounded per-step movement, non-collapsed pair,
        // walk-continuous decay compatibility.
        let move_up = dist(&l_up_next, &l_up);
        let move_low = dist(&l_low_next, &l_low);
        let sep = dist(&l_up_next, &l_low_next);
        let arg_up_next = arg_cont(
            &Complex::with_val_64(prec, &ln_b_next * &l_up_next),
            &arg_up,
        );
        let arg_low_next = arg_cont(
            &Complex::with_val_64(prec, &ln_b_next * &l_low_next),
            &arg_low,
        );
        if move_up > 1 || move_low > 1 || sep < cnum::decimal("0.05", prec) {
            let msg = format!(
                "pair guard tripped (move_up={:.3}, move_low={:.3}, sep={:.3})",
                DisplayFloat(&move_up),
                DisplayFloat(&move_low),
                DisplayFloat(&sep)
            );
            step_fail(msg, &mut queue, &eps_cur, &eps_next)?;
            continue;
        }
        if !(arg_up_next > 0 && arg_low_next < 0) {
            let msg = format!(
                "decay compatibility lost (arg λ_up={:.4}, arg λ_low={:.4})",
                DisplayFloat(&arg_up_next),
                DisplayFloat(&arg_low_next)
            );
            step_fail(msg, &mut queue, &eps_cur, &eps_next)?;
            continue;
        }

        if verbose {
            eprintln!(
                "kouz cut-base walk: ε={:.6e} → {:.6e}  L_up={:.4}+{:.4}i (argλ={:+.4})  L_low={:.4}+{:.4}i (argλ={:+.4})  [solve {}]",
                DisplayFloat(&eps_cur),
                DisplayFloat(&eps_next),
                DisplayFloat(l_up_next.real()),
                DisplayFloat(l_up_next.imag()),
                DisplayFloat(&arg_up_next),
                DisplayFloat(l_low_next.real()),
                DisplayFloat(l_low_next.imag()),
                DisplayFloat(&arg_low_next),
                solves,
            );
        }

        // Warm start: Cauchy-resample the previous solution onto the new
        // grid. Beyond the previous strip height, clamp t — out there F is
        // within grid accuracy of the fixed points anyway.
        //
        // ---- Homotopy-wall handling ----
        // Between the Shell–Thron crossings (ε ≈ 1.55 down to ≈ 0.083 at
        // x = 0.04) a zero of F drifts along/near the sample line
        // Re z = 1/2, so the true curve t ↦ F(1/2+it) changes winding class
        // around 0 repeatedly as ε descends. LM cannot tunnel between
        // classes: a warm start in the wrong class stalls with "no descent"
        // at iter 0 regardless of how small Δε is. Rescue: multiply the
        // warm profile by the smooth phase corrector e^{s·2πi·ramp(t)} —
        // ≡ 1 at both tails (endpoint values and pins unchanged) but
        // winding once around 0 across the pinch node (min |F|, where the
        // zero crosses and values are cheapest to rotate) — i.e. jump the
        // class by hand, then let LM polish.
        //
        // Jumps are gated to *tight* steps (bisection already ground the
        // interval to <2%) and their results to clean quadratic convergence.
        // Both gates are load-bearing: the discrete system admits spurious
        // 1-periodic-dressed near-solutions ("ghosts"), and a class-jumped
        // warm start on a coarse step can stagnation-converge onto one
        // (observed: +1 jump on the coarse 2.0→1.44 step accepted residual
        // 7e-14 via the relaxed stagnation gate with F(0.5) ≈ −0.25−0.53i
        // instead of the true ≈ 0.47+0.28i). True continuation steps always
        // converge quadratically to the full target, so requiring that of
        // jumped results costs nothing and rejects every ghost. Class drift
        // direction is locally persistent, so remember the last winning
        // sign and try it first while in a wall band.
        let t_clamp = cnum::decimal("0.92", prec) * &state.t_max;
        let prev = state;
        // Pinch nodes: up to 3 well-separated interior local minima of |F|
        // on the previous curve, deepest first. Near the ε→0 endgame the
        // curve dips toward 0 at SEVERAL t simultaneously (observed at
        // b=0.06, ε≈0.196: |F| minima 4.6e-2 at t=-29.4 and ~1e-1 at
        // t=-32.3), and the winding class can change at any of them —
        // a corrector at only the deepest pinch cannot reach the true
        // class when the drift happened at the other zero (walk died
        // exactly this way with all single-pinch attempts stalling O(1)).
        let pinches: Vec<(Float, Float)> = {
            let n = prev.samples.len();
            let abs: Vec<Float> = prev.samples.iter().map(|s| cnum::abs(s, prec)).collect();
            if !abs.iter().all(Float::is_finite) {
                return Err("cut-base warm samples have non-finite magnitude".into());
            }
            let ts = &prev.nodes;
            let mut mins: Vec<(Float, Float)> = (1..n.saturating_sub(1))
                .filter(|&k| abs[k] <= abs[k - 1] && abs[k] <= abs[k + 1])
                .map(|k| (ts[k].clone(), abs[k].clone()))
                .collect();
            mins.sort_by(|a, b| a.1.total_cmp(&b.1));
            let mut kept: Vec<(Float, Float)> = Vec::new();
            for m in mins {
                if kept.len() >= 3 {
                    break;
                }
                if kept.iter().all(|k| {
                    Float::with_val_64(prec, &k.0 - &m.0).abs() >= cnum::decimal("1.5", prec)
                }) {
                    kept.push(m);
                }
            }
            if kept.is_empty() {
                kept.push((
                    Float::new_64(prec),
                    Float::with_val_64(prec, rug::float::Special::Infinity),
                ));
            }
            kept
        };
        // Deep pinch ⇒ the left-edge integrand ln F is near-singular on the
        // line and the trapezoidal floor at standard density sits at the
        // acceptance gate (observed: clean quadratic convergence flooring at
        // 1.0e-8, b=0.06, ε≈0.102, |F|min=4.2e-2 — a pure resolution kill,
        // not a class problem). Double the node density there; healthy band
        // pinches sit at |F| ≈ 0.2–0.5 and never trigger this. Second tier:
        // below |F|min<0.05 even the doubled grid skates at the gate
        // (observed: rejections at 1.04–1.07e-8 with n=8192, b=0.06,
        // ε≈0.089–0.092), so quadruple instead (16384, still ≤ N_MAX).
        let node_boost = if pinches[0].1 < cnum::decimal("0.05", prec) {
            4
        } else if pinches[0].1 < cnum::decimal("0.12", prec) {
            2
        } else {
            1
        };
        let make_warm = |combo: &[i32]| {
            let samples = prev.samples.clone();
            let nodes = prev.nodes.clone();
            let weights = prev.weights.clone();
            let t_max = prev.t_max.clone();
            let lu = prev.l_upper.clone();
            let ll = prev.l_lower.clone();
            let lnb = prev.ln_b.clone();
            let t_clamp = t_clamp.clone();
            let two_pi = two_pi.clone();
            let correctors: Vec<(Float, i32)> = pinches
                .iter()
                .zip(combo.iter())
                .filter(|(_, &s)| s != 0)
                .map(|(p, &s)| (p.0.clone(), s))
                .collect();
            move |t: &Float| -> Result<Complex, String> {
                let tc = t.clone().max(&(-t_clamp.clone())).min(&t_clamp);
                let z0 = Complex::with_val_64(prec, (cnum::decimal("0.5", prec), tc));
                let v = cauchy_eval(
                    &z0, &samples, &nodes, &weights, &t_max, &lu, &ll, &lnb, prec, true,
                )?;
                if !cnum::is_finite(&v) {
                    return Err("cut-base Cauchy warm start is non-finite".into());
                }
                if correctors.is_empty() {
                    return Ok(v);
                }
                // ramp per pinch: →1 deep below it, →0 above it; the product
                // of correctors is ≡1 at both tails and inserts one winding
                // loop (of the requested sign) across each active pinch.
                let mut theta = Float::new_64(prec);
                for (t_p, sgn) in &correctors {
                    let scaled = Float::with_val_64(prec, t - t_p) / cnum::decimal("0.75", prec);
                    let ramp = (Float::with_val_64(prec, 1) - scaled.tanh()) / 2u32;
                    theta += ramp * &two_pi * *sgn;
                }
                let phase_arg = Complex::with_val_64(
                    prec,
                    (Float::new_64(prec), Float::with_val_64(prec, theta)),
                );
                let phase = cnum::checked_exp(&phase_arg, prec)?;
                Ok(Complex::with_val_64(prec, &v * &phase))
            }
        };
        let step_tight = (eps_next > 0
            && Float::with_val_64(prec, &eps_cur - &eps_next)
                < cnum::decimal("0.02", prec) * &eps_cur)
            || (eps_next.is_zero() && eps_cur < cnum::decimal("1e-2", prec));
        let np = pinches.len();
        let combo_key = |c: &[i32]| -> Vec<i32> { c.to_vec() };
        let attempts: Vec<Vec<i32>> = if step_tight {
            let mut list: Vec<Vec<i32>> = Vec::new();
            let push_unique = |c: Vec<i32>, list: &mut Vec<Vec<i32>>| {
                if !list.iter().any(|e| e == &c) {
                    list.push(c);
                }
            };
            // Class drift direction is locally persistent: try the last
            // winning corrector pattern first (rank-aligned to the current
            // depth-sorted pinch list), then plain, then the rest.
            if let Some(p) = &rescue_pattern {
                let mut c = vec![0; np];
                for (i, s) in p.iter().enumerate() {
                    if i < np {
                        c[i] = *s;
                    }
                }
                if c.iter().any(|&s| s != 0) {
                    push_unique(c, &mut list);
                }
            }
            push_unique(vec![0; np], &mut list);
            for i in 0..np {
                for s in [1, -1] {
                    let mut c = vec![0; np];
                    c[i] = s;
                    push_unique(c, &mut list);
                }
            }
            if np >= 2 {
                for s0 in [1, -1] {
                    for s1 in [1, -1] {
                        let mut c = vec![0; np];
                        c[0] = s0;
                        c[1] = s1;
                        push_unique(c, &mut list);
                    }
                }
            }
            list
        } else {
            vec![vec![0; np]]
        };
        // Ghost filter: wrong-family "ghosts" have only ever been captured
        // from a *coarse-step* jump (walk evidence: +1 jump on the coarse
        // 2.0→1.44 step stagnation-accepted at 7e-14 with F(0.5) ≈
        // −0.25−0.53i instead of the true ≈ 0.47+0.28i); every tight-step
        // jump observed has landed on the true continuation. Tight-only
        // jumping is therefore the primary filter. This residual gate is
        // the backstop: true tight-step continuations converge quadratically
        // to ~1e-(digits+3), stalling short only when the winding zero sits
        // nearly on the sample line. Observed true-continuation floors RISE
        // as the walk descends toward the zero's closest approach —
        // 1.4e-21 (ε≈1.03) → 5.7e-18 (ε≈0.99) → 3e-14 (ε≈1.04, paced) →
        // 9.6e-13 (ε≈0.90) / 1.7e-12 (b=0.06, ε≈0.92) with clearly
        // decelerating growth (projected peak ~1e-11 … 1e-10) — while
        // wrong-family results only ever appear as stagnation acceptances
        // at 1.9e-7 and above (the one true ghost, from a since-forbidden
        // 28% coarse jump, sat at 7e-14 — coarse jumps no longer exist).
        // 10^(−0.4·digits) = 1e-8 for the standard 20-digit run: decades
        // above the projected floor peak, 18× below the nearest observed
        // wrong-family stall. Every acceptance above 10^-(digits+1) prints
        // an honesty warning with the achieved residual.
        let clean_target = if eps_next.is_zero() {
            cnum::epsilon(digits.saturating_add(3), prec)
        } else {
            (-(Float::with_val_64(prec, digits) * 2u32 / 5u32) * Float::with_val_64(prec, 10).ln())
                .exp()
        };
        let mut solved: Option<(KouznetsovState, bool)> = None;
        let mut last_err = String::new();
        for combo in &attempts {
            let is_jump = combo.iter().any(|&s| s != 0);
            if verbose && is_jump {
                let desc: Vec<String> = pinches
                    .iter()
                    .zip(combo.iter())
                    .filter(|(_, &s)| s != 0)
                    .map(|(p, &s)| {
                        format!(
                            "{:+}@t={:.3}(|F|min={:.3e})",
                            s,
                            DisplayFloat(&p.0),
                            DisplayFloat(&p.1)
                        )
                    })
                    .collect();
                eprintln!(
                    "kouz cut-base walk: homotopy-jump attempt at ε={:.6e}, correctors [{}]",
                    DisplayFloat(&eps_next),
                    desc.join(", ")
                );
            }
            let warm = make_warm(combo);
            // Reactive resolution escalation: when a solve is rejected as a
            // near-miss, double resolution while the same construction remains eligible.
            let mut boost_try = node_boost;
            let outcome = loop {
                let r = setup_kouznetsov_core(
                    &b_next,
                    l_up_next.clone(),
                    l_low_next.clone(),
                    prec,
                    digits,
                    false,
                    Some(&warm),
                    true,
                    !eps_next.is_zero(),
                    true,
                    boost_try,
                );
                match r {
                    Ok(s)
                        if (s.residual.is_nan() || s.residual > clean_target)
                            && s.residual.is_finite()
                            && s.residual <= Float::with_val_64(prec, &clean_target * 1000u32) =>
                    {
                        let next_boost = boost_try
                            .checked_mul(2)
                            .ok_or("cut-base node refinement exceeds addressable memory")?;
                        if verbose {
                            eprintln!(
                                "kouz cut-base walk: near-miss at ε={:.6e} (residual {:.3e}, gate {:.1e}, boost {}×); escalating node tier to {}×",
                                DisplayFloat(&eps_next), DisplayFloat(&s.residual),
                                DisplayFloat(&clean_target), boost_try, next_boost
                            );
                        }
                        boost_try = next_boost;
                        continue;
                    }
                    other => break other,
                }
            };
            match outcome {
                Ok(s) => {
                    if !s.residual.is_finite() || s.residual > clean_target {
                        if verbose {
                            eprintln!(
                                "kouz cut-base walk: rejecting result at ε={:.6e} (jump={}): residual {:.3e} not cleanly converged (need ≤ {:.1e}; stagnation-accepted results can be wrong-family ghosts)",
                                DisplayFloat(&eps_next), is_jump, DisplayFloat(&s.residual), DisplayFloat(&clean_target)
                            );
                        }
                        last_err = format!(
                            "result rejected: residual {:.3e} above clean target {:.1e} (no descent surrogate)",
                            DisplayFloat(&s.residual), DisplayFloat(&clean_target)
                        );
                        continue;
                    }
                    if verbose && is_jump {
                        eprintln!(
                            "kouz cut-base walk: homotopy jump converged at ε={:.6e} (combo {:?})",
                            DisplayFloat(&eps_next),
                            combo
                        );
                    }
                    if verbose && s.residual > cnum::epsilon(digits.saturating_add(3), prec) {
                        eprintln!(
                            "kouz: retaining internal cut-base candidate at epsilon={:.6e}, residual {:.3e}; it does not meet the final 1e-{} target",
                            DisplayFloat(&eps_next), DisplayFloat(&s.residual), digits + 3
                        );
                    }
                    if is_jump {
                        rescue_pattern = Some(combo_key(combo));
                    }
                    solved = Some((s, is_jump));
                    break;
                }
                Err(e) => {
                    let wall_like = e.contains("no descent") || e.contains("stagnation");
                    last_err = e;
                    if !wall_like {
                        break;
                    }
                }
            }
        }
        match solved {
            Some((s, won_jump)) => {
                state = s;
                eps_cur = eps_next;
                l_up = l_up_next;
                l_low = l_low_next;
                arg_up = arg_up_next;
                arg_low = arg_low_next;
                // Wall-band pacing: while recently rescued, prepend a fine
                // (1.5%, immediately jump-eligible) target so each band step
                // costs ~1 solve instead of a full coarse-target fail →
                // bisect → fail → rescue cascade. Five consecutive plain
                // wins mean the winding zero has moved off the sample line;
                // fall back to the geometric schedule already in the queue.
                if won_jump {
                    steps_since_rescue = Integer::new();
                } else {
                    steps_since_rescue += 1;
                }
                if steps_since_rescue <= 5 && eps_cur > cnum::decimal("1e-3", prec) {
                    let fine = Float::with_val_64(prec, &eps_cur * cnum::decimal("0.985", prec));
                    if queue.front().is_some_and(|front| {
                        *front < Float::with_val_64(prec, &fine * cnum::decimal("0.9995", prec))
                    }) {
                        queue.push_front(fine);
                    }
                } else if steps_since_rescue == 6 {
                    rescue_pattern = None;
                }
                if eps_cur > 0 {
                    if let Some(p) = ckpt_path.as_deref() {
                        save_cut_ckpt(p, b_re, digits, &eps_cur, &arg_up, &arg_low, &state)?;
                    }
                }
            }
            None => {
                state = prev;
                step_fail(
                    format!("LM solve failed: {}", last_err),
                    &mut queue,
                    &eps_cur,
                    &eps_next,
                )?;
                continue;
            }
        }
    }

    if !eps_cur.is_zero() {
        return Err(format!(
            "cut-base walk: schedule ended at ε={:.6e} without reaching 0",
            DisplayFloat(&eps_cur)
        ));
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_guess_preserves_tiny_complex_base_components() {
        for digits in [50, 70, 1000] {
            let prec = cnum::digits_to_bits(digits);
            let reference_prec = prec + 128;
            let (min, _) = cnum::exponent_range();
            let tiny = Float::with_val_64(prec, 1) >> usize::try_from((1 - min) / 2).unwrap();
            let base = Complex::with_val_64(prec, (2, &tiny));
            let samples = initial_guess_with_target(
                &[Float::new_64(prec)],
                &base,
                &Complex::with_val_64(prec, (1, 1)),
                &Complex::with_val_64(prec, (1, -1)),
                &Float::with_val_64(prec, 1),
                prec,
                None,
            );
            let root = Float::with_val_64(reference_prec, 2).sqrt();
            let expected_imaginary = Float::with_val_64(
                prec,
                Float::with_val_64(reference_prec, &tiny)
                    / Float::with_val_64(reference_prec, &root * 2u32),
            );
            assert!(
                samples[0].real() == &Float::with_val_64(prec, root)
                    && samples[0].imag() == &expected_imaginary
                    && !samples[0].imag().is_zero(),
                "initial guess lost the tiny component at {digits} digits"
            );
        }
    }

    #[test]
    fn cauchy_quadrature_resolves_constants_at_high_precision_near_boundaries() {
        let mut failures = Vec::new();
        for digits in [110, 130] {
            let prec = cnum::digits_to_bits(digits);
            let base = Complex::with_val_64(prec, 2);
            let crate::regions::Region::OutsideShellThronRealPositive(fp) =
                crate::regions::classify(&base, prec).unwrap()
            else {
                panic!("base-2 geometry");
            };
            let t_max = contour_height(digits, &arg_abs(&fp.lambda, prec), prec).unwrap();
            let n = pick_node_count(digits, &t_max, prec).unwrap();
            let nodes = build_uniform_nodes(&t_max, n, prec);
            let weights = build_trapezoidal_weights(&t_max, n, prec);
            let constant = Complex::with_val_64(prec, 2);
            let samples = vec![constant.clone(); n];
            let ln_base = Complex::with_val_64(prec, Float::with_val_64(prec, 2).ln() / 2);
            let target = cnum::epsilon(digits + 3, prec);
            for (real, imaginary) in [
                ("0.5", Float::new_64(prec)),
                ("0.5", nodes[n - 2].clone()),
                ("0", nodes[n - 2].clone()),
            ] {
                let height = Complex::with_val_64(prec, (cnum::decimal(real, prec), &imaginary));
                let actual = cauchy_eval(
                    &height, &samples, &nodes, &weights, &t_max, &constant, &constant, &ln_base,
                    prec, false,
                )
                .unwrap();
                let error = cnum::abs(&Complex::with_val_64(prec, actual - &constant), prec);
                eprintln!(
                    "constant_cauchy_probe digits={digits} re={real} im={:.8e} nodes={n} error={:.8e}",
                    DisplayFloat(&imaginary), DisplayFloat(&error)
                );
                if error > target {
                    failures.push(format!(
                        "{digits} digits, re={real}, error {} exceeds {}",
                        DisplayFloat(&error),
                        DisplayFloat(&target)
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    fn asymmetric_contours_budget_for_both_fixed_point_tails() {
        for digits in [10, 70] {
            let prec = cnum::digits_to_bits(digits);
            let base = Complex::with_val_64(prec, -2);
            let negative_log =
                Complex::with_val_64(prec, -Complex::with_val_64(prec, base.ln_ref()));
            let upper = Complex::with_val_64(prec, -lambertw::w0(&negative_log, prec).unwrap());
            let lower = Complex::with_val_64(prec, -lambertw::wk(&negative_log, 1, prec).unwrap());
            let rate = slowest_decay_rate(&upper, &lower, prec);
            assert_eq!(rate, arg_abs(&lower, prec));
            assert!(rate < arg_abs(&upper, prec));
            let height = contour_height(digits, &rate, prec).unwrap();
            let budget = cnum::epsilon(digits + 8, prec)
                * (Float::with_val_64(prec, 1) + cnum::working_epsilon(prec));
            for boundary in [&upper, &lower] {
                let tail = (-Float::with_val_64(prec, &height * arg_abs(boundary, prec))).exp();
                assert!(tail <= budget);
            }
            let old_height = contour_height(digits, &arg_abs(&upper, prec), prec).unwrap();
            let old_lower_tail = (-Float::with_val_64(prec, old_height * &rate)).exp();
            assert!(old_lower_tail > cnum::epsilon(digits + 3, prec));
        }
    }

    #[test]
    fn node_and_em_orders_have_no_practical_precision_ceiling() {
        for (digits, expected_nodes) in [(200, 524_288), (1500, 16_777_216)] {
            let prec = cnum::digits_to_bits(digits);
            let base = Complex::with_val_64(prec, 2);
            let crate::regions::Region::OutsideShellThronRealPositive(fp) =
                crate::regions::classify(&base, prec).unwrap()
            else {
                panic!("base-2 region");
            };
            let height = contour_height(digits, &arg_abs(&fp.lambda, prec), prec).unwrap();
            let nodes = pick_node_count(digits, &height, prec).unwrap();
            assert_eq!(nodes, expected_nodes);
            assert!(crate::fft::kernel_fft_len(nodes, prec).unwrap() >= nodes);
            let spacing = Float::with_val_64(prec, &height * 2) / (nodes - 1);
            assert!(em_n_terms(prec, &spacing, &fp.fixed_point, &fp.fixed_point).unwrap() > 20);
        }
        let prec = cnum::digits_to_bits(70);
        let height = contour_height(70, &cnum::decimal("1e-30", prec), prec).unwrap();
        assert!(pick_node_count(70, &height, prec)
            .unwrap_err()
            .contains("addressable memory"));
        assert!(pick_node_count(70, &cnum::decimal("1e1000000000000000000", prec), prec).is_err());
    }

    #[test]
    fn extended_bernoulli_coefficients_match_exact_values_and_zeta() {
        use rug::{integer::IntegerExt64, ops::Pow};
        let coefficients = em_coefficients(40).unwrap();
        for (k, numerator, denominator) in [
            (1, "1", "12"),
            (6, "691", "32760"),
            (20, "261082718496449122051", "541200"),
            (21, "1520097643918070802691", "75852"),
            (25, "495057205241079648212477525", "3300"),
        ] {
            let expected = Rational::from((
                Integer::from_str_radix(numerator, 10).unwrap(),
                Integer::from_str_radix(denominator, 10).unwrap(),
            ));
            assert_eq!(coefficients[k - 1], expected);
        }
        assert!(coefficients[39].numer().significant_bits_64() > 128);
        let prec = cnum::digits_to_bits(120);
        let two_pi = Float::with_val_64(prec, Constant::Pi) * 2u32;
        for (k, coefficient) in coefficients.iter().take(40).enumerate() {
            let n = 2 * (k + 1);
            let factorial = Integer::from(Integer::factorial(n as u32));
            let reference =
                Float::with_val_64(prec, factorial) * 2 * Float::with_val_64(prec, n).zeta()
                    / two_pi.clone().pow(n as u32)
                    / n;
            let actual = Float::with_val_64(prec, coefficient);
            let error = Float::with_val_64(prec, &actual - &reference).abs();
            assert!(
                error <= cnum::epsilon(100, prec) * actual.abs(),
                "Bernoulli coefficient {n}"
            );
        }
        assert!(em_coefficients(usize::MAX)
            .unwrap_err()
            .contains("addressable memory"));
    }

    #[test]
    fn cut_schedule_can_exceed_two_thousand_steps() {
        let prec = cnum::digits_to_bits(70);
        let start = Float::with_val_64(prec, 2);
        let queue = cut_schedule(&start, &cnum::decimal("0.999", prec), prec).unwrap();
        assert!(queue.len() > 2000);
        let mut previous = start;
        for next in queue {
            assert!(next < previous);
            previous = next;
        }
        assert!(previous.is_zero());
    }

    fn checkpoint_fixture(digits: u64) -> (Float, Float, KouznetsovState) {
        let prec = cnum::digits_to_bits(digits);
        let base = cnum::decimal("0.04", prec);
        let eps = cnum::decimal(
            "0.12345678901234567890123456789012345678901234567890123456789",
            prec,
        );
        let b = Complex::with_val_64(prec, (&base, &eps));
        let t_max = Float::with_val_64(prec, 10);
        let upper = Complex::with_val_64(prec, (1, 2));
        let lower = Complex::with_val_64(prec, (1, -2));
        let samples = vec![
            lower.clone(),
            Complex::with_val_64(
                prec,
                (cnum::decimal("1e-1000", prec), cnum::epsilon(digits, prec)),
            ),
            Complex::with_val_64(prec, (cnum::decimal("1e1000", prec), eps.clone())),
            upper.clone(),
        ];
        (
            base,
            eps,
            KouznetsovState {
                samples,
                nodes: build_uniform_nodes(&t_max, 4, prec),
                weights: build_trapezoidal_weights(&t_max, 4, prec),
                t_max,
                l_upper: upper,
                l_lower: lower,
                ln_b: Complex::with_val_64(prec, b.ln_ref()),
                shift: cnum::zero(prec),
                prec,
                digits,
                normalized: false,
                residual: cnum::epsilon(digits + 3, prec),
                two_sided: true,
            },
        )
    }

    fn checkpoint_path() -> String {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let serial = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir()
            .join(format!(
                "tetration-checkpoint-{}-{serial}",
                std::process::id()
            ))
            .to_str()
            .unwrap()
            .to_owned()
    }

    #[test]
    fn checkpoints_support_more_than_32768_nodes_and_native_exponents() {
        let digits = 70;
        let (base, eps, mut state) = checkpoint_fixture(digits);
        let n = 32_769;
        state.nodes = build_uniform_nodes(&state.t_max, n, state.prec);
        state.weights = build_trapezoidal_weights(&state.t_max, n, state.prec);
        state.samples = vec![cnum::one(state.prec); n];
        state.samples[0] = state.l_lower.clone();
        state.samples[n - 1] = state.l_upper.clone();
        state.samples[1] = cnum::parse_complex(
            "1e1000000000000000000",
            "1e-1000000000000000000",
            state.prec,
        )
        .unwrap();
        let path = checkpoint_path();
        let arg = Float::with_val_64(state.prec, Constant::Pi);
        save_cut_ckpt(&path, &base, digits, &eps, &arg, &-arg.clone(), &state).unwrap();
        let (_, _, _, loaded) = load_cut_ckpt(&path, &base, digits, state.prec)
            .unwrap()
            .unwrap();
        assert!(loaded.samples == state.samples);
        assert!(loaded.nodes == state.nodes);
        assert!(loaded.weights == state.weights);
        let original = std::fs::read_to_string(&path).unwrap();
        let malformed = original.replacen(&format!("\n{n}\n"), &format!("\n{}\n", usize::MAX), 1);
        std::fs::write(&path, malformed).unwrap();
        assert!(load_cut_ckpt(&path, &base, digits, state.prec).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn checkpoint_round_trip_preserves_full_precision() {
        for digits in [50, 70, 1000] {
            let (base, eps, state) = checkpoint_fixture(digits);
            let path = checkpoint_path();
            let arg_up = Float::with_val_64(state.prec, Constant::Pi);
            let arg_low = -arg_up.clone();
            assert!(load_cut_ckpt(&path, &base, digits, state.prec)
                .unwrap()
                .is_none());
            save_cut_ckpt(&path, &base, digits, &eps, &arg_up, &arg_low, &state).unwrap();
            let (loaded_eps, loaded_up, loaded_low, loaded) =
                load_cut_ckpt(&path, &base, digits, state.prec)
                    .unwrap()
                    .unwrap();
            assert_eq!(loaded_eps, eps);
            assert_eq!(loaded_up, arg_up);
            assert_eq!(loaded_low, arg_low);
            assert_eq!(loaded.samples, state.samples);
            assert_eq!(loaded.nodes, state.nodes);
            assert_eq!(loaded.weights, state.weights);
            assert_eq!(loaded.residual, state.residual);
            assert_eq!(loaded.ln_b, state.ln_b);
            assert!(!loaded.normalized);
            assert!(load_cut_ckpt(&path, &base, digits + 1, state.prec).is_err());
            assert!(load_cut_ckpt(&path, &base, digits, state.prec + 1).is_err());
            let other_base = base.clone() + cnum::epsilon(digits, state.prec);
            assert!(load_cut_ckpt(&path, &other_base, digits, state.prec).is_err());
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn checkpoint_rejects_corruption_and_reports_io_failures() {
        let (base, eps, state) = checkpoint_fixture(70);
        let path = checkpoint_path();
        let arg_up = Float::with_val_64(state.prec, 2);
        let arg_low = Float::with_val_64(state.prec, -2);
        save_cut_ckpt(&path, &base, 70, &eps, &arg_up, &arg_low, &state).unwrap();
        let original = std::fs::read_to_string(&path).unwrap();
        for (line, replacement) in [
            (0, "TETCKPT1"),
            (1, "NaN"),
            (4, "0.1 bad 2 -2 1e-73"),
            (4, "0.1 2 -2 NaN"),
            (4, "0.1 2 -2 1e-73 extra"),
            (4, "-1 2 -2 0"),
            (4, "0.1 2 -2 -1"),
            (5, "0"),
            (6, "1 2 extra"),
            (8, "0 0"),
            (9, "32769"),
            (10, "NaN\t1\t1\t-2"),
            (10, "-9\t1\t1\t-2"),
            (11, "0\t0\tNaN\t0"),
            (11, "0\t0\t1\t0\textra"),
        ] {
            let mut lines: Vec<_> = original.lines().map(str::to_owned).collect();
            lines[line] = replacement.into();
            std::fs::write(&path, lines.join("\n")).unwrap();
            assert!(
                load_cut_ckpt(&path, &base, 70, state.prec).is_err(),
                "line {line}: {replacement}"
            );
        }
        for text in [
            format!("{original}unexpected\n"),
            original.lines().take(12).collect::<Vec<_>>().join("\n"),
        ] {
            std::fs::write(&path, text).unwrap();
            assert!(load_cut_ckpt(&path, &base, 70, state.prec).is_err());
        }
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert!(load_cut_ckpt(&path, &base, 70, state.prec).is_err());
        assert!(save_cut_ckpt(&path, &base, 70, &eps, &arg_up, &arg_low, &state).is_err());
        assert!(!std::path::Path::new(&format!("{}.tmp-{}", path, std::process::id())).exists());
        std::fs::remove_dir(path).unwrap();
    }

    #[test]
    fn residual_dumps_preserve_precision_and_report_collisions() {
        let (_, _, state) = checkpoint_fixture(1000);
        let path = std::path::PathBuf::from(checkpoint_path());
        let residuals = vec![Complex::with_val_64(state.prec, cnum::epsilon(1000, state.prec)); 4];
        dump_residual(
            &path,
            &state.nodes,
            &state.samples,
            &residuals,
            &state.samples,
            state.prec,
        )
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        for (i, line) in text.lines().skip(1).enumerate() {
            let values: Vec<_> = line
                .split('\t')
                .map(|s| cnum::parse_float(s, state.prec).unwrap())
                .collect();
            assert_eq!(values.len(), 6);
            assert_eq!(values[0], state.nodes[i]);
            assert_eq!(&values[1], state.samples[i].real());
            assert_eq!(&values[2], state.samples[i].imag());
            assert_eq!(values[3], cnum::epsilon(1000, state.prec));
            assert_eq!(values[4], values[1]);
            assert_eq!(values[5], values[2]);
        }
        assert_eq!(text.lines().count(), 5);
        assert!(dump_residual(
            &path,
            &state.nodes,
            &state.samples,
            &residuals,
            &state.samples,
            state.prec
        )
        .is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn cauchy_operators_reject_nonfinite_and_exponent_range_errors() {
        for digits in [50, 70, 1000] {
            let prec = cnum::digits_to_bits(digits);
            let t_max = Float::with_val_64(prec, 1);
            let nodes = build_uniform_nodes(&t_max, 4, prec);
            let weights = build_trapezoidal_weights(&t_max, 4, prec);
            let one = cnum::one(prec);
            let height = Complex::with_val_64(prec, cnum::decimal("0.5", prec));
            let kernels = build_cauchy_kernels(&nodes, &t_max, &one, &one, prec).unwrap();
            for value in ["0", "NaN", "inf", "1e1000", "-1e1000"] {
                let bad = Complex::with_val_64(
                    prec,
                    Float::with_val_64(prec, Float::parse(value).unwrap()),
                );
                let mut samples = vec![one.clone(); 4];
                samples[1] = bad;
                assert!(cauchy_eval(
                    &height, &samples, &nodes, &weights, &t_max, &one, &one, &one, prec, false
                )
                .is_err());
                assert!(apply_t_fft(
                    &samples, &nodes, &weights, &t_max, &one, &one, &one, &kernels, prec, false
                )
                .is_err());
                assert!(precompute_dt_factors(&samples, &one, prec).is_err());
            }
        }
    }

    #[test]
    fn cauchy_resampling_preserves_sub_machine_grid_differences() {
        let (_, _, mut state) = checkpoint_fixture(70);
        state.samples = vec![cnum::one(state.prec); 4];
        state.l_lower = cnum::one(state.prec);
        state.l_upper = cnum::one(state.prec);
        state.ln_b = cnum::one(state.prec);
        let near_zero = cnum::epsilon(60, state.prec);
        let heights = vec![
            Float::new_64(state.prec),
            near_zero.clone(),
            state.t_max.clone() * 2u32,
            state.t_max.clone() * -2i32,
        ];
        let next_upper = Complex::with_val_64(state.prec, (1, &near_zero));
        let next_lower = Complex::with_val_64(state.prec, (1, -&near_zero));
        let actual =
            resample_to_grid(&state, &heights, &next_upper, &next_lower, state.prec).unwrap();
        assert_ne!(actual[0], actual[1]);
        for (i, height) in heights.iter().take(2).enumerate() {
            let z = Complex::with_val_64(state.prec, (cnum::decimal("0.5", state.prec), height));
            assert_eq!(
                actual[i],
                cauchy_eval(
                    &z,
                    &state.samples,
                    &state.nodes,
                    &state.weights,
                    &state.t_max,
                    &state.l_upper,
                    &state.l_lower,
                    &state.ln_b,
                    state.prec,
                    state.two_sided
                )
                .unwrap()
            );
        }
        assert_eq!(actual[2], next_upper);
        assert_eq!(actual[3], next_lower);
    }

    #[test]
    fn cut_schedule_validates_ratio_and_decreases_without_rounding_to_f64() {
        let prec = cnum::digits_to_bits(70);
        let start = Float::with_val_64(prec, 2);
        for ratio in [
            "0",
            "-0.5",
            "1",
            "1.1",
            "0.999999999999999999999999999999999999999999999999999999999999",
        ] {
            assert!(cut_schedule(&start, &cnum::decimal(ratio, prec), prec).is_err());
        }
        let ratio = cnum::decimal(
            "0.720000000000000000000000000000000000000000000000000000000001",
            prec,
        );
        let queue = cut_schedule(&start, &ratio, prec).unwrap();
        let mut previous = start;
        for epsilon in queue {
            assert!(epsilon.is_finite() && epsilon < previous && epsilon >= 0);
            if !epsilon.is_zero() {
                assert_eq!(epsilon, Float::with_val_64(prec, &previous * &ratio));
            }
            previous = epsilon;
        }
        assert!(previous.is_zero());
        assert!(cut_schedule(&Float::new_64(prec), &ratio, prec).is_err());
    }

    #[test]
    fn cached_kouznetsov_rejects_a_different_base() {
        let (base, eps, mut state) = checkpoint_fixture(50);
        state.normalized = true;
        let wrong_base =
            Complex::with_val_64(state.prec, (base + cnum::epsilon(45, state.prec), eps));
        let error = eval_kouznetsov(&state, &wrong_base, &cnum::zero(state.prec)).unwrap_err();
        assert!(error.contains("base does not match"), "{error}");
    }

    #[test]
    fn normalization_refuses_a_spurious_root_outside_the_raw_contour() {
        let prec = cnum::digits_to_bits(50);
        let t_max = Float::with_val_64(prec, 10);
        let nodes = build_uniform_nodes(&t_max, 64, prec);
        let weights = build_trapezoidal_weights(&t_max, 64, prec);
        let fixed = Complex::with_val_64(prec, 2);
        let samples = vec![fixed.clone(); 64];
        let base = Complex::with_val_64(prec, 2);
        let ln_b = Complex::with_val_64(prec, base.ln_ref());
        // The inconsistent synthetic samples previously produced a raw root
        // near -4.978, although the returned function's anchor error was >2.
        let error = find_normalization_shift(
            &samples, &nodes, &weights, &t_max, &fixed, &fixed, &ln_b, prec, 50, true, false,
        )
        .unwrap_err();
        assert!(
            error.contains("no normalized value can be returned"),
            "{error}"
        );
    }

    #[test]
    fn solvers_reject_nan_instead_of_reporting_zero_residual() {
        let prec = cnum::digits_to_bits(50);
        let t_max = Float::with_val_64(prec, 1);
        let nodes = build_uniform_nodes(&t_max, 4, prec);
        let weights = build_trapezoidal_weights(&t_max, 4, prec);
        let upper = Complex::with_val_64(prec, (1, 1));
        let lower = Complex::with_val_64(prec, (1, -1));
        let mut samples = vec![cnum::one(prec); 4];
        samples[1] = Complex::with_val_64(prec, Float::with_val_64(prec, rug::float::Special::Nan));
        for solve in [iterate_newton, iterate_anderson, iterate_picard] {
            assert!(solve(
                samples.clone(),
                &nodes,
                &weights,
                &t_max,
                &upper,
                &lower,
                &cnum::one(prec),
                prec,
                50,
                false,
                false
            )
            .is_err());
        }
        assert!(find_normalization_shift(
            &samples,
            &nodes,
            &weights,
            &t_max,
            &upper,
            &lower,
            &cnum::one(prec),
            prec,
            50,
            false,
            false
        )
        .is_err());
    }

    #[test]
    fn gmres_preserves_tiny_nonzero_rhs_at_high_precision() {
        for digits in [50, 70, 1000] {
            let prec = cnum::digits_to_bits(digits);
            for value in [
                "1e-1000",
                "1e-200000000",
                "1e200000000",
                "1e-1000000000000000000",
                "1e1000000000000000000",
            ] {
                let rhs = vec![
                    Complex::with_val_64(prec, cnum::decimal(value, prec)),
                    cnum::zero(prec),
                ];
                let target = cnum::epsilon(digits, prec);
                let actual = gmres_complex(
                    |v| {
                        v.iter()
                            .map(|x| Complex::with_val_64(prec, x * 2u32))
                            .collect()
                    },
                    &rhs,
                    &target,
                    2,
                    prec,
                )
                .unwrap();
                assert!(!cnum::is_zero(&actual[0]));
                let residual = Complex::with_val_64(prec, actual[0].clone() * 2u32 - &rhs[0]);
                assert!(cnum::abs(&residual, prec) <= target * cnum::abs(&rhs[0], prec));
            }
        }
    }

    #[test]
    fn gmres_rejects_invalid_data_and_checks_actual_residual() {
        let prec = cnum::digits_to_bits(50);
        let rhs = vec![cnum::one(prec)];
        let tol = cnum::epsilon(50, prec);
        assert!(gmres_complex(|_| Vec::new(), &rhs, &tol, 1, prec).is_err());
        let nan = Complex::with_val_64(prec, Float::with_val_64(prec, rug::float::Special::Nan));
        assert!(gmres_complex(|v| v.to_vec(), &[nan], &tol, 1, prec).is_err());
        let calls = std::cell::Cell::new(0);
        let inconsistent = |v: &[Complex]| {
            let count = calls.get();
            calls.set(count + 1);
            v.iter()
                .map(|x| Complex::with_val_64(prec, x * if count < 2 { 1u32 } else { 2u32 }))
                .collect()
        };
        assert!(gmres_complex(inconsistent, &rhs, &tol, 1, prec).is_err());
    }

    #[test]
    fn gmres_continues_past_eight_restarts_and_expands_a_stalled_window() {
        let prec = cnum::digits_to_bits(100);
        let rhs = vec![cnum::one(prec); 2];
        let tolerance = cnum::epsilon(80, prec);
        let calls = std::cell::Cell::new(0usize);
        let answer = gmres_complex(
            |v| {
                calls.set(calls.get() + 1);
                vec![v[0].clone(), Complex::with_val_64(prec, &v[1] * 2)]
            },
            &rhs,
            &tolerance,
            1,
            prec,
        )
        .unwrap();
        assert!(
            calls.get() > 30,
            "must exercise more than eight restart cycles"
        );
        let residual = vec![
            Complex::with_val_64(prec, &answer[0] - 1),
            Complex::with_val_64(prec, Complex::with_val_64(prec, &answer[1] * 2) - 1),
        ];
        assert!(
            vector_norm_complex(&residual, prec) <= &tolerance * vector_norm_complex(&rhs, prec)
        );

        let rhs = vec![cnum::one(prec), cnum::zero(prec)];
        let answer = gmres_complex(
            |v| vec![v[1].clone(), Complex::with_val_64(prec, -&v[0])],
            &rhs,
            &tolerance,
            1,
            prec,
        )
        .unwrap();
        assert!(cnum::abs(&answer[0], prec) < tolerance);
        assert!(cnum::abs(&Complex::with_val_64(prec, &answer[1] - 1), prec) < tolerance);
    }

    #[test]
    fn height_conditioning_refines_accuracy_not_just_storage_precision() {
        for digits in [40, 70, 1000] {
            let prec = cnum::digits_to_bits(digits);
            let log_gain = cnum::decimal("18.75", prec) * Float::with_val_64(prec, 10).ln();
            assert_eq!(
                height_precision_digits(digits, digits, &log_gain).unwrap(),
                Some(digits + 19)
            );
            assert_eq!(
                height_precision_digits(digits, digits + 19, &log_gain).unwrap(),
                None
            );
            let tolerance = cnum::epsilon(digits, prec);
            let large_log_gain = Float::with_val_64(prec, 100) * Float::with_val_64(prec, 10).ln();
            let next = cnum::conditioned_precision(&large_log_gain, &tolerance, prec)
                .unwrap()
                .unwrap();
            assert!(next > prec);
            assert_eq!(
                cnum::conditioned_precision(&large_log_gain, &tolerance, next).unwrap(),
                None
            );
            let small_gain = cnum::decimal("2.5", prec) * Float::with_val_64(prec, 10).ln();
            assert_eq!(
                height_precision_digits(digits, digits, &small_gain).unwrap(),
                None
            );
            assert!(
                height_precision_digits(digits, digits, &cnum::decimal("1e1000", prec)).is_err()
            );
            assert!(
                cnum::conditioned_precision(&cnum::decimal("1e1000", prec), &tolerance, prec)
                    .is_err()
            );
        }
    }

    #[test]
    fn height_conditioning_logs_do_not_overflow_with_the_complex_magnitude() {
        let prec = cnum::digits_to_bits(70);
        let reference_prec = cnum::digits_to_bits(100);
        let (_, max) = cnum::exponent_range();
        let mut component = Float::with_val_64(prec, 1) << usize::try_from(max - 1).unwrap();
        component *= cnum::decimal("1.5", prec);
        let value = Complex::with_val_64(prec, (&component, &component));
        assert!(cnum::is_finite(&value));
        assert!(!cnum::abs(&value, prec).is_finite());
        let actual = log_magnitude(&value, prec);
        let expected = Float::with_val_64(reference_prec, &component).ln()
            + Float::with_val_64(reference_prec, 2).ln() / 2;
        let error = Float::with_val_64(reference_prec, actual - &expected).abs() / expected;
        assert!(error < cnum::epsilon(70, reference_prec));
    }

    #[test]
    fn cauchy_height_conditioning_tracks_exponential_and_logarithmic_sensitivity() {
        let prec = cnum::digits_to_bits(70);
        let t_max = Float::with_val_64(prec, 10);
        let nodes = build_uniform_nodes(&t_max, 256, prec);
        let weights = build_trapezoidal_weights(&t_max, 256, prec);
        let one = cnum::one(prec);
        let samples = vec![one.clone(); nodes.len()];
        let height = cnum::parse_complex("0.25", "0", prec).unwrap();
        let value = cauchy_eval(
            &height, &samples, &nodes, &weights, &t_max, &one, &one, &one, prec, false,
        )
        .unwrap();
        for shift in [-2i32, -1, 0, 1, 2] {
            let mut expected = value.clone();
            let mut expected_log_gain = Float::new_64(prec);
            for _ in 0..shift.unsigned_abs() {
                let magnitude = cnum::abs(&expected, prec);
                let gain = if shift > 0 {
                    magnitude.max(&Float::with_val_64(prec, 1)).ln()
                } else {
                    -magnitude.min(&Float::with_val_64(prec, 1)).ln()
                };
                expected_log_gain += gain;
                expected = if shift > 0 {
                    cnum::checked_exp(&expected, prec).unwrap()
                } else {
                    Complex::with_val_64(prec, expected.ln_ref())
                };
            }
            expected_log_gain -= cnum::abs(&expected, prec)
                .min(&Float::with_val_64(prec, 1))
                .ln();
            let (actual, log_gain) = eval_at_height_with_conditioning(
                &(height.clone() + shift),
                &samples,
                &nodes,
                &weights,
                &t_max,
                &one,
                &one,
                &one,
                prec,
                false,
            )
            .unwrap();
            assert_eq!(actual, expected);
            assert!(
                Float::with_val_64(prec, log_gain - expected_log_gain).abs()
                    < cnum::epsilon(70, prec)
            );
        }
    }

    #[test]
    fn cauchy_height_shift_performs_exactly_the_requested_steps() {
        for digits in [50, 70, 1000] {
            let prec = cnum::digits_to_bits(digits);
            let t_max = Float::with_val_64(prec, 10);
            let nodes = build_uniform_nodes(&t_max, 256, prec);
            let weights = build_trapezoidal_weights(&t_max, 256, prec);
            let one = cnum::one(prec);
            let samples = vec![one.clone(); nodes.len()];
            let height = Complex::with_val_64(prec, cnum::decimal("0.25", prec));
            let value = cauchy_eval(
                &height, &samples, &nodes, &weights, &t_max, &one, &one, &one, prec, false,
            )
            .unwrap();
            for shift in [-2i32, -1, 0, 1, 2] {
                let mut expected = value.clone();
                for _ in 0..shift.unsigned_abs() {
                    expected = if shift > 0 {
                        cnum::checked_exp(&expected, prec).unwrap()
                    } else {
                        Complex::with_val_64(prec, expected.ln_ref())
                    };
                }
                let shifted_height = height.clone() + shift;
                let actual = eval_at_height(
                    &shifted_height,
                    &samples,
                    &nodes,
                    &weights,
                    &t_max,
                    &one,
                    &one,
                    &one,
                    prec,
                    false,
                )
                .unwrap();
                assert!(actual == expected, "{digits} digits, shift {shift}");
            }
        }
    }

    #[test]
    fn cauchy_geometry_never_saturates_height_or_returns_an_asymptote() {
        let prec = cnum::digits_to_bits(50);
        let t_max = Float::with_val_64(prec, 1);
        let nodes = build_uniform_nodes(&t_max, 4, prec);
        let weights = build_trapezoidal_weights(&t_max, 4, prec);
        let samples = vec![cnum::one(prec); 4];
        for (re, im) in [
            ("1e1000", "0"),
            ("-9223372036854775808", "1"),
            ("10001", "0"),
            ("0.5", "1"),
            ("0.5", "-1"),
            ("0.5", "1e1000"),
            ("0.5", "-1e1000"),
        ] {
            let height = cnum::parse_complex(re, im, prec).unwrap();
            assert!(eval_at_height(
                &height,
                &samples,
                &nodes,
                &weights,
                &t_max,
                &cnum::one(prec),
                &cnum::one(prec),
                &cnum::one(prec),
                prec,
                false
            )
            .is_err());
        }
    }

    /// Two-sided anchored unwrap must reconstruct a smooth log-curve whose
    /// imaginary part sweeps far outside (−π, π], exactly where the
    /// principal log wraps.
    #[test]
    fn unwrap_reconstructs_smooth_log_curve() {
        for digits in [50, 70, 1000] {
            let prec = cnum::digits_to_bits(digits);
            let n = 64usize;
            let ln_b = cnum::one(prec);
            let l_up = cnum::parse_complex("0.3", "4", prec).unwrap();
            let l_low = cnum::parse_complex("0.2", "-4", prec).unwrap();
            let g: Vec<Complex> = (0..n)
                .map(|i| {
                    let s = Float::with_val_64(prec, i) / (n - 1);
                    let re = cnum::decimal("0.2", prec) + s.clone() / 10u32;
                    let im = s * 8u32 - 4u32;
                    Complex::with_val_64(prec, (re, im))
                })
                .collect();
            let samples: Vec<_> = g
                .iter()
                .map(|gi| cnum::checked_exp(gi, prec).unwrap())
                .collect();
            let out = unwrapped_ln_samples(&samples, &l_up, &l_low, &ln_b, prec, true);
            for (o, gi) in out.iter().zip(g.iter()) {
                let d = cnum::abs(&Complex::with_val_64(prec, o - gi), prec);
                assert!(d < cnum::epsilon(digits, prec), "unwrap deviated by {d}");
            }
        }
    }

    /// A curve that gains one extra winding mid-curve (top tail lands on the
    /// anchor branch + 2πi) must be reconstructed per-anchor on each half —
    /// the 2π mismatch stays localized at the joint instead of corrupting a
    /// whole half with the principal branch.
    #[test]
    fn unwrap_localizes_joint_winding_mismatch() {
        for digits in [50, 70, 1000] {
            let prec = cnum::digits_to_bits(digits);
            let n = 64usize;
            let two_pi = Float::with_val_64(prec, Constant::Pi) * 2u32;
            let ln_b = cnum::one(prec);
            let l_up = cnum::parse_complex("0.3", "4", prec).unwrap();
            let l_low = cnum::parse_complex("0.2", "-4", prec).unwrap();
            let g: Vec<Complex> = (0..n)
                .map(|i| {
                    let s = Float::with_val_64(prec, i) / (n - 1);
                    let ramp = ((s.clone() - cnum::decimal("0.5", prec)) * 8u32).tanh() / 2u32
                        + cnum::decimal("0.5", prec);
                    let re = cnum::decimal("0.2", prec) + s.clone() / 10u32;
                    let im = s * 8u32 - 4u32 + &two_pi * ramp;
                    Complex::with_val_64(prec, (re, im))
                })
                .collect();
            let samples: Vec<_> = g
                .iter()
                .map(|gi| cnum::checked_exp(gi, prec).unwrap())
                .collect();
            let out = unwrapped_ln_samples(&samples, &l_up, &l_low, &ln_b, prec, true);
            for (i, (o, gi)) in out.iter().zip(g.iter()).enumerate() {
                let shift = if i < n / 2 {
                    Float::new_64(prec)
                } else {
                    -two_pi.clone()
                };
                let expected =
                    Complex::with_val_64(prec, gi + Complex::with_val_64(prec, (0, shift)));
                let d = cnum::abs(&Complex::with_val_64(prec, o - &expected), prec);
                assert!(
                    d < cnum::epsilon(digits, prec),
                    "node {i}: anchored unwrap error {d}"
                );
            }
        }
    }
}
