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

use rug::{float::Constant, Complex, Float};

use crate::{
    cnum,
    fft::{cross_correlate_with_kernel, precompute_kernel_fft, KernelFft},
    lambertw,
    regions::FixedPointData,
};

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
pub struct KouznetsovState {
    pub samples: Vec<Complex>,
    pub nodes: Vec<Float>,
    pub weights: Vec<Float>,
    pub t_max: Float,
    pub l_upper: Complex,
    pub l_lower: Complex,
    pub ln_b: Complex,
    pub shift: Complex,
    pub prec: u32,
}

pub fn tetrate_kouznetsov(
    b: &Complex,
    h: &Complex,
    fp: &FixedPointData,
    prec: u32,
    digits: u64,
) -> Result<Complex, String> {
    let state = setup_kouznetsov(b, fp, prec, digits)?;
    eval_kouznetsov(&state, b, h)
}

/// Compute the per-base Kouznetsov state. This is the expensive step: it
/// runs the Newton-Kantorovich Cauchy iteration to convergence and then
/// finds the normalization shift δ. After this returns, evaluating F at
/// any height is cheap (`eval_kouznetsov`).
pub fn setup_kouznetsov(
    b: &Complex,
    fp: &FixedPointData,
    prec: u32,
    digits: u64,
) -> Result<KouznetsovState, String> {
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
    let ln_b = Complex::with_val(prec, b.ln_ref());
    let neg_ln_b = Complex::with_val(prec, -&ln_b);

    let raw = fp.fixed_point.clone();
    let (l_lower, l_upper) = if use_schwarz {
        let raw_conj = Complex::with_val(prec, raw.conj_ref());
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
        let neg_w0 = Complex::with_val(prec, -&w0_val);
        let l_plus = Complex::with_val(prec, &neg_w0 / &ln_b);
        let seed = Complex::with_val(prec, l_plus.conj_ref());
        let mut l_minus = newton_fixed_point(&ln_b, &seed, prec).map_err(|e| {
            format!(
                "Kouznetsov: could not find partner fixed point near conj(L_+): {}",
                e
            )
        })?;
        let im_plus = Float::with_val(prec, l_plus.imag()).to_f64();
        let im_minus_init = Float::with_val(prec, l_minus.imag()).to_f64();
        // If Newton-from-conjugate landed in the same half-plane as L_+, the
        // rectangle Cauchy formula's boundary conditions become ill-posed.
        // Search W_k branches (k = 1, -2, 2, -3, 3, …) for a fixed point in
        // the opposite half-plane. This is a heuristic Paulsen-Cowgill-style
        // branch selection; for negative real bases and far-complex bases the
        // resulting tetration may not be the canonical Kneser-Kouznetsov F,
        // but it satisfies F(0)=1 and F(z+1)=b^F(z), which is verified by
        // the iteration's residual.
        let mut _wk_used = false;
        if im_plus.signum() == im_minus_init.signum()
            && im_plus.abs() > 1e-12
            && im_minus_init.abs() > 1e-12
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
            const MIN_IM_STRIP: f64 = 0.05;
            let debug_wk = cnum::verbose();
            let mut candidates: Vec<(i32, Complex, f64)> = Vec::new();
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
                let neg_wk = Complex::with_val(prec, -&wk_val);
                let l_k = Complex::with_val(prec, &neg_wk / &ln_b);
                let im_k = Float::with_val(prec, l_k.imag()).to_f64();
                let re_k = Float::with_val(prec, l_k.real()).to_f64();
                // Verify L_k is genuinely a fixed point of b^z = z (Halley
                // can converge to a nearby branch for poor seeds).
                let bz = Complex::with_val(
                    prec,
                    Complex::with_val(prec, &l_k * &ln_b).exp_ref(),
                );
                let resid = Float::with_val(
                    prec,
                    Complex::with_val(prec, &bz - &l_k).abs_ref(),
                )
                .to_f64();
                let opposite = im_k.signum() != im_plus.signum()
                    && im_k.abs() > MIN_IM_STRIP;
                if debug_wk {
                    eprintln!(
                        "kouz wk search: k={:>+}  L={:.4}+{:.4}i  resid={:.2e}  opposite={}  |im|>{}={}",
                        k, re_k, im_k, resid, opposite, MIN_IM_STRIP, im_k.abs() > MIN_IM_STRIP
                    );
                }
                if opposite && resid < 1e-6 {
                    candidates.push((k, l_k, im_k));
                }
            }
            // Pick smallest |k| (canonical preference) — already partially ordered
            // by W_K_SEARCH but we re-sort to be safe and explicit.
            candidates.sort_by(|a, b| a.0.unsigned_abs().cmp(&b.0.unsigned_abs())
                .then(a.0.cmp(&b.0)));
            if let Some((kchosen, l_k, _imk)) = candidates.into_iter().next() {
                if debug_wk {
                    eprintln!("kouz wk search: chose k={}", kchosen);
                }
                l_minus = l_k;
                _wk_used = true;
            } else {
                return Err(format!(
                    "Kouznetsov: fixed-point pair (L_+ = {:.4}+{:.4}i, L_- = {:.4}+{:.4}i) lies in the same half-plane; W_k search across k∈±[1..5] found no opposite-half-plane partner with |Im|>{}. Paulsen-Cowgill conformal mapping is required.",
                    Float::with_val(prec, l_plus.real()).to_f64(),
                    im_plus,
                    Float::with_val(prec, l_minus.real()).to_f64(),
                    im_minus_init,
                    MIN_IM_STRIP,
                ));
            }
        }
        let im_minus = Float::with_val(prec, l_minus.imag()).to_f64();
        if im_plus >= im_minus {
            (l_minus, l_plus)
        } else {
            (l_plus, l_minus)
        }
    };
    // λ_upper = (ln b)·L_upper drives the contour decay rate.
    let lambda_upper = Complex::with_val(prec, &ln_b * &l_upper);

    // Decay rate of F → fixed point: ~ exp(-T·|arg(λ̄)|). Pick T so the tail
    // beyond ±T is below 10^{-(digits+8)}.
    let arg_lambda = arg_abs_f64(&lambda_upper, prec);
    if arg_lambda <= 1e-3 {
        return Err(format!(
            "Kouznetsov: |arg(λ)| = {} too small (degenerate contour)",
            arg_lambda
        ));
    }
    let t_max_f64 = ((digits as f64 + 8.0) * std::f64::consts::LN_10 / arg_lambda).max(8.0);
    let t_max = Float::with_val(prec, t_max_f64);

    // Trapezoidal node count: scales with both `digits` and `t_max`, with the
    // analyticity-strip width `|arg(λ)|` driving the per-node convergence rate.
    let n_nodes = pick_node_count(digits, t_max_f64, arg_lambda);

    // Parabolic-boundary guard: near |λ|=1 with arg(λ)≈0 the strip height
    // t_max = O(1/arg(λ)) blows up, requiring n_nodes ≫ 32 K and ~20s/LM-iter.
    // We cannot converge in any reasonable time — bail with a clean ERR.
    // (Kouznetsov 2009 §6 / Écalle theory needed for this regime.)
    const N_MAX_PRACTICAL: usize = 32_768;
    if n_nodes > N_MAX_PRACTICAL {
        return Err(format!(
            "Kouznetsov: |arg(λ)| = {:.4} too small (parabolic boundary), \
             requires n_nodes={} > {} — not supported; needs Abel/Écalle theory",
            arg_lambda, n_nodes, N_MAX_PRACTICAL
        ));
    }

    let nodes = build_uniform_nodes(&t_max, n_nodes, prec);
    let weights = build_trapezoidal_weights(&t_max, n_nodes, prec);

    if cnum::verbose() {
        eprintln!(
            "kouz setup: |arg(λ)|={:.4}  t_max={:.2}  n_nodes={}  L_upper={:.4}+{:.4}i  L_lower={:.4}+{:.4}i",
            arg_lambda,
            t_max_f64,
            n_nodes,
            Float::with_val(prec, l_upper.real()).to_f64(),
            Float::with_val(prec, l_upper.imag()).to_f64(),
            Float::with_val(prec, l_lower.real()).to_f64(),
            Float::with_val(prec, l_lower.imag()).to_f64(),
        );
    }

    let debug_phase = cnum::verbose();
    let phase_start = std::time::Instant::now();

    // Helper: build + symmetrize an initial guess, with an optional target_mid
    // override (for multi-start retries when the default cap fails).
    let make_initial = |target_override: Option<f64>| -> Vec<Complex> {
        let mut init = initial_guess_with_target(
            &nodes, b, &l_upper, &l_lower, arg_lambda, prec, target_override,
        );
        init[0] = l_lower.clone();
        init[n_nodes - 1] = l_upper.clone();
        if use_schwarz {
            symmetrize_schwarz(&mut init, prec);
        }
        init
    };

    let initial = make_initial(None);
    if debug_phase {
        eprintln!("kouz phase: initial_guess done ({:.2}s)", phase_start.elapsed().as_secs_f64());
    }
    let phase_start = std::time::Instant::now();
    // Default solver: Levenberg-Marquardt Newton-Kantorovich. Newton's
    // Jacobian-based step finds the right descent direction even when the
    // Cauchy operator T has spectral radius > 1 (which it does for typical
    // real bases b > e^(1/e)), where Picard / Anderson without history
    // diverge. The two non-default solvers stay around as debugging aids:
    //   * `TET_KOUZ_ANDERSON=1`: Anderson-accelerated Picard (works when T
    //     is a contraction, fails for higher b).
    //   * `TET_KOUZ_PICARD=1`: damped Picard, useful for spectrum diagnosis.
    let lm_result = if std::env::var_os("TET_KOUZ_ANDERSON").is_some() {
        iterate_anderson(
            initial, &nodes, &weights, &t_max, &l_upper, &l_lower, &ln_b, prec, digits,
            use_schwarz,
        )
    } else if std::env::var_os("TET_KOUZ_PICARD").is_some() {
        iterate_picard(
            initial, &nodes, &weights, &t_max, &l_upper, &l_lower, &ln_b, prec, digits,
            use_schwarz,
        )
    } else {
        iterate_newton(
            initial, &nodes, &weights, &t_max, &l_upper, &l_lower, &ln_b, prec, digits,
            use_schwarz,
        )
    };

    // Multi-start: if the default initial guess fails to converge (LM "no
    // descent step" — typical for bases like b≈5 where the converged F̃[mid]≈0.73
    // is well below the default cap of 1.5), retry with alternative target_mid
    // values. Two retries cover the b∈[η,e²] gap.
    let samples = match lm_result {
        Ok(s) => s,
        Err(ref e) if e.contains("no descent step") || e.contains("no convergence") => {
            if debug_phase {
                eprintln!("kouz: LM failed with default guess ({}); retrying with alternative initial guesses", e);
            }
            let retry_targets: &[f64] = &[0.75, 1.1, 0.5, 1.25];
            let mut last_err = e.clone();
            let mut found = None;
            for &target in retry_targets {
                if debug_phase {
                    eprintln!("kouz: retry with target_mid={}", target);
                }
                let retry_init = make_initial(Some(target));
                match iterate_newton(
                    retry_init, &nodes, &weights, &t_max, &l_upper, &l_lower, &ln_b, prec, digits,
                    use_schwarz,
                ) {
                    Ok(s) => { found = Some(s); break; }
                    Err(e2) => { last_err = e2; }
                }
            }
            found.ok_or(last_err)?
        }
        Err(e) => return Err(e),
    };
    if debug_phase {
        eprintln!("kouz phase: iterate_newton done ({:.2}s elapsed since symmetrize)", phase_start.elapsed().as_secs_f64());
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
    let shift = find_normalization_shift(
        &samples, &nodes, &weights, &t_max, &l_upper, &l_lower, &ln_b, prec, digits,
    )?;
    if debug_phase {
        eprintln!("kouz phase: find_normalization_shift done ({:.2}s)", phase_start.elapsed().as_secs_f64());
        eprintln!(
            "kouz normalization shift δ = {:.6e} + {:.6e}i (such that F(δ)=1)",
            Float::with_val(prec, shift.real()).to_f64(),
            Float::with_val(prec, shift.imag()).to_f64(),
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
    })
}

/// Evaluate `F_b(h)` from a precomputed `KouznetsovState`. Cheap (one Cauchy
/// integral plus integer-shifting via b^· / log_b iterations); reuses the
/// expensive samples and normalization shift from `setup_kouznetsov`.
pub fn eval_kouznetsov(state: &KouznetsovState, b: &Complex, h: &Complex) -> Result<Complex, String> {
    let h_shifted = Complex::with_val(state.prec, h + &state.shift);
    eval_at_height(
        b,
        &h_shifted,
        &state.samples,
        &state.nodes,
        &state.weights,
        &state.t_max,
        &state.l_upper,
        &state.l_lower,
        &state.ln_b,
        state.prec,
    )
}

/// Find the unique real shift δ such that `F(δ) = 1`, where `F` is the
/// converged Cauchy reconstruction. Newton's method on the scalar equation
/// `g(c) = cauchy_eval(c) − 1 = 0`, starting from c=0. The derivative
/// `g'(c) = F'(c)` is computed by central finite difference (Cauchy
/// reconstruction is locally smooth in c, so finite difference is stable).
///
/// Convergence is typically 4–6 iterations (quadratic) from c=0 because the
/// initial guess shape (tanh+sech bump) places F(0) within ~0.1 of 1, so c is
/// small and Newton is in its quadratic-convergence regime immediately.
///
/// `eps_f` (FD step size) and `target` scale with `digits`: at high precision,
/// a fixed `eps=1e-8` would limit derivative accuracy to ~16 digits, so we use
/// the cube-root rule `eps ≈ 10^(-digits/3)` (optimal for central differences,
/// balancing truncation `O(ε²·f''')` against roundoff `O(δ_f/ε)`).
#[allow(clippy::too_many_arguments)]
fn find_normalization_shift(
    samples: &[Complex],
    nodes: &[Float],
    weights: &[Float],
    t_max: &Float,
    l_upper: &Complex,
    l_lower: &Complex,
    ln_b: &Complex,
    prec: u32,
    digits: u64,
) -> Result<Complex, String> {
    let one = Complex::with_val(prec, (1u32, 0));
    let two = Float::with_val(prec, 2u32);
    // Cube-root rule for central FD: ε ≈ δ_f^(1/3) where δ_f is the working
    // precision. With δ_f ≈ 10^(-digits), ε ≈ 10^(-digits/3 - 3).
    let eps_pow = -((digits as i32 + 8) / 3).min(150);
    let eps_f64 = 10f64.powi(eps_pow);
    let eps_f = Float::with_val(prec, eps_f64);
    let eps = Complex::with_val(prec, (eps_f.clone(), 0));
    let two_eps = Complex::with_val(prec, (Float::with_val(prec, &eps_f * &two), 0));

    let debug_norm = cnum::verbose();

    // Newton-Kantorovich step. Returns the best (c, |F(c)-1|) seen during
    // iteration. Two return cases:
    //   * Ok with residual ≤ target — Newton fully converged (canonical case).
    //   * Ok with residual > target — Newton stalled near a root but didn't
    //     reach full precision (non-canonical W_k case where F̃ has its own
    //     discretization floor below which Newton cannot drive the residual).
    //   * Err if Newton diverged (NaN) or hit a degenerate derivative far
    //     from any root.
    // We run Newton from each grid seed and keep all converged-or-stalled
    // results, then pick the smallest-|c| one. Tracking BEST-during-iteration
    // (rather than just final) makes the choice precision-independent: at
    // higher digits the same root is found, just with a tighter residual.
    let target_pow = -((digits as i32 * 2 / 3).min(200));
    let target = 10f64.powi(target_pow);
    // Acceptance threshold for stalled Newton: a seed is "good enough" if
    // Newton drove the residual below this floor, even if it didn't reach
    // target. 1e-4 is well below typical F̃ values (which range over the
    // entire complex plane near boundary L_+/L_-) and below the basin radius
    // around any root, so anything reaching this floor is genuinely near a
    // root of F̃-1.
    let stall_floor = 1e-4f64;
    let try_newton = |seed: &Complex| -> Result<(Complex, f64), String> {
        let mut c = seed.clone();
        let mut best_c = c.clone();
        let mut best_resid = f64::INFINITY;
        for _ in 0..40usize {
            let f_c = cauchy_eval(
                &c, samples, nodes, weights, t_max, l_upper, l_lower, ln_b, prec,
            );
            let resid = Complex::with_val(prec, &f_c - &one);
            let resid_abs = Float::with_val(prec, resid.abs_ref()).to_f64();
            if !resid_abs.is_finite() {
                if best_resid < stall_floor {
                    return Ok((best_c, best_resid));
                }
                return Err("non-finite".into());
            }
            if resid_abs < best_resid {
                best_resid = resid_abs;
                best_c = c.clone();
            }
            if resid_abs < target {
                return Ok((c, resid_abs));
            }
            let c_plus = Complex::with_val(prec, &c + &eps);
            let c_minus = Complex::with_val(prec, &c - &eps);
            let f_plus = cauchy_eval(
                &c_plus, samples, nodes, weights, t_max, l_upper, l_lower, ln_b, prec,
            );
            let f_minus = cauchy_eval(
                &c_minus, samples, nodes, weights, t_max, l_upper, l_lower, ln_b, prec,
            );
            let diff = Complex::with_val(prec, &f_plus - &f_minus);
            let derivative = Complex::with_val(prec, &diff / &two_eps);
            let der_abs = Float::with_val(prec, derivative.abs_ref()).to_f64();
            if der_abs < 1e-50 {
                if best_resid < stall_floor {
                    return Ok((best_c, best_resid));
                }
                return Err("F'(c)≈0".into());
            }
            let step = Complex::with_val(prec, &resid / &derivative);
            c = Complex::with_val(prec, &c - &step);
        }
        // Out of iterations. Accept if best-during-iteration is below stall
        // floor (Newton was in a root's basin); else reject.
        if best_resid < stall_floor {
            Ok((best_c, best_resid))
        } else {
            Err(format!("did_not_converge_resid={:.3e}", best_resid))
        }
    };

    // Fast path: Newton from c=0. Real positive bases land here; F̃(0) is
    // already very close to 1 and Newton converges in 4-6 iters.
    if let Ok((c0, _r0)) = try_newton(&cnum::zero(prec)) {
        if debug_norm {
            eprintln!(
                "kouz norm: Newton from c=0 converged to ({:.6},{:.6}i)",
                Float::with_val(prec, c0.real()).to_f64(),
                Float::with_val(prec, c0.imag()).to_f64(),
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
    let re_steps = [
        0.0f64, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0, 1.25, -1.25, 1.5, -1.5,
    ];
    let im_steps = [0.0f64, 0.25, -0.25, 0.5, -0.5, 1.0, -1.0];

    #[derive(Clone)]
    struct Root {
        c: Complex,
        re: f64,
        im: f64,
        abs: f64,
    }
    let mut roots: Vec<Root> = Vec::new();
    let mut best_resid_seed: Option<(Complex, f64)> = None;
    for &cr in &re_steps {
        for &ci in &im_steps {
            let seed = Complex::with_val(
                prec,
                (Float::with_val(prec, cr), Float::with_val(prec, ci)),
            );
            match try_newton(&seed) {
                Ok((c_root, _r)) => {
                    let re = Float::with_val(prec, c_root.real()).to_f64();
                    let im = Float::with_val(prec, c_root.imag()).to_f64();
                    let abs = (re * re + im * im).sqrt();
                    roots.push(Root { c: c_root, re, im, abs });
                }
                Err(why) => {
                    if let Some(last) = why.strip_prefix("did_not_converge_resid=") {
                        if let Ok(rval) = last.parse::<f64>() {
                            if let Some((_, br)) = &best_resid_seed {
                                if rval < *br {
                                    best_resid_seed = Some((seed, rval));
                                }
                            } else {
                                best_resid_seed = Some((seed, rval));
                            }
                        }
                    }
                }
            }
        }
    }

    if !roots.is_empty() {
        // Cluster near-identical converged roots (within 1e-6) so seeds that
        // hit the same root don't bias toward arbitrary winners.
        // Then pick the cluster representative with smallest |c|, tiebreak
        // Im ascending, Re ascending.
        roots.sort_by(|a, b| {
            a.abs
                .partial_cmp(&b.abs)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.im.partial_cmp(&b.im).unwrap_or(std::cmp::Ordering::Equal))
                .then(a.re.partial_cmp(&b.re).unwrap_or(std::cmp::Ordering::Equal))
        });
        let chosen = roots.into_iter().next().unwrap();
        if debug_norm {
            eprintln!(
                "kouz norm: chose smallest-|c| Newton root ({:.6},{:.6}i) |c|={:.3e}",
                chosen.re, chosen.im, chosen.abs,
            );
        }
        return Ok(chosen.c);
    }

    // No seed converged Newton. If a partial-progress seed has low residual,
    // return it as best-effort. Otherwise error out — the (L_+, L_-) pair
    // doesn't admit a normalizable Kouznetsov reconstruction.
    if let Some((c_partial, rval)) = best_resid_seed {
        if rval < 0.1 {
            if debug_norm {
                eprintln!(
                    "kouz norm: no seed Newton-converged; returning best-partial \
                     c=({:.4},{:.4}i) |F-1|={:.3e}",
                    Float::with_val(prec, c_partial.real()).to_f64(),
                    Float::with_val(prec, c_partial.imag()).to_f64(),
                    rval
                );
            }
            return Ok(c_partial);
        }
    }
    Err(format!(
        "Kouznetsov normalization: no grid seed produced Newton-converged \
         root of F̃(c)=1 with target {:.3e}; the (L_+, L_-) pair gives a Cauchy \
         strip whose values miss F(c)=1 in [-1.5,1.5]×[-1.0,1.0]. This usually \
         means the fixed-point pair is not the canonical Kneser pair (e.g. via \
         W_k search for non-real bases) and a conformal mapping \
         (Paulsen-Cowgill) is required.",
        target
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
fn newton_fixed_point(
    ln_b: &Complex,
    seed: &Complex,
    prec: u32,
) -> Result<Complex, String> {
    let one = Complex::with_val(prec, (1, 0));
    // Target |f| < 2^-(prec - 16); leave a small guard so the loop terminates.
    let target_log2: i32 = -((prec as i32).saturating_sub(16));
    let target_tol = Float::with_val(prec, Float::with_val(prec, 1.0) >> -target_log2);
    let mut z = seed.clone();
    for _ in 0..200 {
        let arg = Complex::with_val(prec, &z * ln_b);
        let bz = Complex::with_val(prec, arg.exp_ref());
        let f = Complex::with_val(prec, &bz - &z);
        let fabs = Float::with_val(prec, f.abs_ref());
        if fabs < target_tol {
            return Ok(z);
        }
        let fp = {
            let t = Complex::with_val(prec, &bz * ln_b);
            Complex::with_val(prec, &t - &one)
        };
        if Float::with_val(prec, fp.abs_ref()).to_f64() < 1e-30 {
            return Err("Newton fixed-point: derivative vanished".into());
        }
        let delta = Complex::with_val(prec, &f / &fp);
        z -= &delta;
    }
    Err("Newton fixed-point: did not converge in 200 iterations".into())
}

fn arg_abs_f64(z: &Complex, prec: u32) -> f64 {
    let im = Float::with_val(prec, z.imag()).to_f64();
    let re = Float::with_val(prec, z.real()).to_f64();
    im.atan2(re).abs()
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
fn pick_node_count(digits: u64, t_max: f64, _arg_lambda: f64) -> usize {
    const SIGMA_EFF: f64 = 0.30;
    const MARGIN: f64 = 1.2;
    let n_bulk = (MARGIN
        * (digits as f64 + 5.0)
        * std::f64::consts::LN_10
        * t_max
        / (std::f64::consts::PI * SIGMA_EFF))
        .ceil() as usize;
    // Hard floor: 80 nodes for reasonable resolution. Hard cap: 60k — beyond
    // that the FFT-domain matvec gets prohibitive even at moderate precision.
    let clamped = n_bulk.clamp(80, 60_000);
    if clamped >= 60_000 {
        return clamped;
    }
    // FFT-friendly snap. The cross-correlation pads to next_power_of_two(2N−1).
    // For N in [2^(k−1)+1, 2^k], that padded length is constant at 2^(k+1).
    // Snapping N up to 2^k (the right edge of its bucket) gives maximal
    // trapezoidal accuracy at zero extra FFT cost. `next_power_of_two(2^k)`
    // returns 2^k itself, so this is a no-op when N is already a power of two.
    clamped.next_power_of_two()
}

/// Estimated trapezoidal-truncation floor for residual: `exp(-|arg(λ)|·N/T)`.
/// Used by debug printing and by precision-ceiling logic.
#[allow(dead_code)]
fn trapezoidal_floor(arg_lambda: f64, n_nodes: usize, t_max: f64) -> f64 {
    let arg_safe = arg_lambda.max(0.05);
    (-(arg_safe * n_nodes as f64 / t_max)).exp()
}

fn build_uniform_nodes(t_max: &Float, n: usize, prec: u32) -> Vec<Float> {
    let two_t = Float::with_val(prec, t_max * 2u32);
    let delta = Float::with_val(prec, &two_t / ((n - 1) as u32));
    (0..n)
        .map(|k| Float::with_val(prec, -(t_max.clone()) + Float::with_val(prec, &delta * k as u32)))
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
fn symmetrize_schwarz(samples: &mut [Complex], prec: u32) {
    let n = samples.len();
    let half = Float::with_val(prec, 0.5f64);
    for k in 0..n / 2 {
        let m = n - 1 - k;
        let re_avg = Float::with_val(
            prec,
            (Float::with_val(prec, samples[k].real() + samples[m].real())) * &half,
        );
        let im_avg = Float::with_val(
            prec,
            (Float::with_val(prec, samples[k].imag() - samples[m].imag())) * &half,
        );
        let neg_im = Float::with_val(prec, -&im_avg);
        samples[k] = Complex::with_val(prec, (re_avg.clone(), im_avg));
        samples[m] = Complex::with_val(prec, (re_avg, neg_im));
    }
    // Middle node (if n is odd) must have purely real F.
    if n % 2 == 1 {
        let mid = n / 2;
        let re = Float::with_val(prec, samples[mid].real());
        samples[mid] = Complex::with_val(prec, (re, Float::new(prec)));
    }
}

fn build_trapezoidal_weights(t_max: &Float, n: usize, prec: u32) -> Vec<Float> {
    let two_t = Float::with_val(prec, t_max * 2u32);
    let delta = Float::with_val(prec, &two_t / ((n - 1) as u32));
    let half_delta = Float::with_val(prec, &delta / 2u32);
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
// Each EM term gains roughly `(h/T)²` over the previous, so K ≈ 7–10 lifts
// the floor from O(h²) ≈ 10⁻⁶ to well below 10⁻⁵⁰ at our typical h/T.
//
// EM is asymptotic, not convergent — Bernoulli numbers grow factorially —
// but the optimal truncation `K_opt ≈ T·π/h` is in the thousands for our
// regime, so K=10 is far inside the safe range.
// =====================================================================

/// `|B_{2k}| / (2k)` as an exact rational, expressed as `(numerator_str,
/// denominator_str)` so we can fit Bernoulli numerators that overflow `u64`
/// (k ≥ 18 do). Strings are parsed into `rug::Integer` at use time. Covers
/// k=1..=20, which is enough margin to hit `10^-(digits+3)` floor up to
/// digits ≈ 100 with `h ~ 10^-2`.
fn em_coef_rational(k: usize) -> Option<(&'static str, &'static str)> {
    match k {
        1 => Some(("1", "12")),
        2 => Some(("1", "120")),
        3 => Some(("1", "252")),
        4 => Some(("1", "240")),
        5 => Some(("1", "132")),
        6 => Some(("691", "32760")),
        7 => Some(("1", "12")),
        8 => Some(("3617", "8160")),
        9 => Some(("43867", "14364")),
        10 => Some(("174611", "6600")),
        11 => Some(("854513", "3036")),
        12 => Some(("236364091", "65520")),
        13 => Some(("8553103", "156")),
        14 => Some(("23749461029", "24360")),
        15 => Some(("8615841276005", "429660")),
        16 => Some(("7709321041217", "16320")),
        17 => Some(("2577687858367", "204")),
        18 => Some(("26315271553053477373", "69090840")),
        19 => Some(("2929993913841559", "228")),
        20 => Some(("261082718496449122051", "541200")),
        _ => None,
    }
}

/// Pick how many EM terms to compute. We default to K=12 (the largest size
/// our `em_coef_rational` table holds) because:
///
/// * The closed-form derivative formula evaluates `1/(c±iT−z₀)^{2k}` —
///   when `z₀` sits near a boundary node `T_k ≈ ±T`, that denominator can
///   shrink to `O(1)` instead of `O(T)`, so EM terms decay much slower
///   than the naive `(h/T)^{2k}` argument suggests.
/// * The cost is O(K · N) per call (one closed-form evaluation per row),
///   negligible against the O(N²) FFT matvec.
/// * Empirically on `b=2, digits=30, N=8264, T=80`: K=6 stalls Newton at
///   ~6e-28 while K=12 reaches 5e-36, eight orders of magnitude better,
///   for ~zero extra CPU.
///
/// `TET_KOUZ_NO_EM=1` returns 0 (skip EM entirely) for A/B testing;
/// `TET_KOUZ_EM_K=<n>` overrides for diagnostics.
///
/// Auto-pick: each additional EM term reduces the boundary-floor residual by
/// ~2.5 decades (empirical at b=2, h ≈ 10^{-2}). K=12 reaches ~10^{-48} for
/// digits=50; we need digits+3 of headroom, so scale K with digits beyond 30.
fn em_n_terms(_h: f64, _t_max: f64, digits: u64) -> usize {
    if std::env::var_os("TET_KOUZ_NO_EM").is_some() {
        return 0;
    }
    if let Ok(s) = std::env::var("TET_KOUZ_EM_K") {
        if let Ok(k) = s.parse::<usize>() {
            return k.min(20);
        }
    }
    let extra = (digits.saturating_sub(30) / 5) as usize;
    (12 + extra).min(20)
}

/// Pre-bake `[ -i · |B_{2k}|/(2k) · h^{2k} ]` for k=1..=K. The `-i` factor
/// (from `(−i)^{2k−1} = (−1)^k · i` combined with the alternating sign of
/// `B_{2k}`) simplifies to a uniform `-i` across all k.
fn build_em_h_powers(h: &Float, n_terms: usize, prec: u32) -> Vec<Complex> {
    if n_terms == 0 {
        return Vec::new();
    }
    let neg_i = Complex::with_val(
        prec,
        (Float::new(prec), Float::with_val(prec, -1i32)),
    );
    let h_sq = Float::with_val(prec, h * h);
    let mut h_pow = h_sq.clone();
    let mut out = Vec::with_capacity(n_terms);
    for k in 1..=n_terms {
        let (num_str, den_str) =
            em_coef_rational(k).expect("EM coefficient table exhausted (k > 20)");
        let num_int = rug::Integer::from_str_radix(num_str, 10)
            .expect("valid Bernoulli numerator literal");
        let den_int = rug::Integer::from_str_radix(den_str, 10)
            .expect("valid Bernoulli denominator literal");
        let coef_real = Float::with_val(
            prec,
            Float::with_val(prec, &num_int) / Float::with_val(prec, &den_int),
        );
        let scaled = Float::with_val(prec, &coef_real * &h_pow);
        out.push(Complex::with_val(prec, &neg_i * &scaled));
        if k < n_terms {
            h_pow = Float::with_val(prec, &h_pow * &h_sq);
        }
    }
    out
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
    prec: u32,
) -> (Complex, Complex) {
    if em_h_powers.is_empty() {
        return (cnum::zero(prec), cnum::zero(prec));
    }
    let cp1 = Complex::with_val(prec, (Float::with_val(prec, 1.5f64), 0));
    let cm1 = Complex::with_val(prec, (Float::with_val(prec, -0.5f64), 0));
    let it_max = Complex::with_val(prec, (Float::new(prec), t_max.clone()));
    let neg_it_max = Complex::with_val(prec, -&it_max);

    // c±iT − z₀ for right edge (c=1.5) and left edge (c=−0.5).
    let c_r = Complex::with_val(prec, &cp1 - z0);
    let c_l = Complex::with_val(prec, &cm1 - z0);
    let cr_pos = Complex::with_val(prec, &c_r + &it_max);
    let cr_neg = Complex::with_val(prec, &c_r + &neg_it_max);
    let cl_pos = Complex::with_val(prec, &c_l + &it_max);
    let cl_neg = Complex::with_val(prec, &c_l + &neg_it_max);

    // (c±iT−z₀)² — incremental power update inside the loop multiplies by
    // these to advance from (c±iT−z₀)^{2k} to (c±iT−z₀)^{2(k+1)}.
    let cr_pos_sq = Complex::with_val(prec, &cr_pos * &cr_pos);
    let cr_neg_sq = Complex::with_val(prec, &cr_neg * &cr_neg);
    let cl_pos_sq = Complex::with_val(prec, &cl_pos * &cl_pos);
    let cl_neg_sq = Complex::with_val(prec, &cl_neg * &cl_neg);

    let one_c = Complex::with_val(prec, (Float::with_val(prec, 1u32), 0));
    let mut pr_pos = one_c.clone();
    let mut pr_neg = one_c.clone();
    let mut pl_pos = one_c.clone();
    let mut pl_neg = one_c;

    let mut corr_r = cnum::zero(prec);
    let mut corr_l = cnum::zero(prec);
    for coef in em_h_powers {
        pr_pos = Complex::with_val(prec, &pr_pos * &cr_pos_sq);
        pr_neg = Complex::with_val(prec, &pr_neg * &cr_neg_sq);
        pl_pos = Complex::with_val(prec, &pl_pos * &cl_pos_sq);
        pl_neg = Complex::with_val(prec, &pl_neg * &cl_neg_sq);

        let r_pos = Complex::with_val(prec, l_upper / &pr_pos);
        let r_neg = Complex::with_val(prec, l_lower / &pr_neg);
        let r_diff = Complex::with_val(prec, &r_pos - &r_neg);
        let r_term = Complex::with_val(prec, &r_diff * coef);
        corr_r = Complex::with_val(prec, &corr_r + &r_term);

        let l_pos = Complex::with_val(prec, l_upper / &pl_pos);
        let l_neg = Complex::with_val(prec, l_lower / &pl_neg);
        let l_diff = Complex::with_val(prec, &l_pos - &l_neg);
        let l_term = Complex::with_val(prec, &l_diff * coef);
        corr_l = Complex::with_val(prec, &corr_l + &l_term);
    }
    (corr_r, corr_l)
}

fn initial_guess(
    nodes: &[Float],
    b: &Complex,
    l_upper: &Complex,
    l_lower: &Complex,
    arg_lambda: f64,
    prec: u32,
) -> Vec<Complex> {
    initial_guess_with_target(nodes, b, l_upper, l_lower, arg_lambda, prec, None)
}

fn initial_guess_with_target(
    nodes: &[Float],
    b: &Complex,
    l_upper: &Complex,
    l_lower: &Complex,
    arg_lambda: f64,
    prec: u32,
    target_mid_override: Option<f64>,
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
    let half = Float::with_val(prec, 0.5f64);
    let one = Float::with_val(prec, 1u32);
    let rate = Float::with_val(prec, arg_lambda);
    let sqrt_b = {
        let half_c = Complex::with_val(prec, (half.clone(), 0));
        cnum::pow_complex(b, &half_c, prec)
    };
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
    let sqrt_b_abs = Float::with_val(prec, sqrt_b.abs_ref()).to_f64();
    let b_abs = Float::with_val(prec, b.abs_ref()).to_f64();
    let cap: f64 = (1.5 - 0.1 * (b_abs.ln() - 2.0).max(0.0)).clamp(0.7, 1.5);
    let target_mid = if let Some(override_val) = target_mid_override {
        // Explicit override for retry attempts (multi-start strategy).
        let dir = Complex::with_val(prec, &sqrt_b / Float::with_val(prec, sqrt_b_abs.max(1e-15)));
        Complex::with_val(prec, &dir * Float::with_val(prec, override_val))
    } else if sqrt_b_abs <= cap {
        sqrt_b.clone()
    } else {
        let scale = Float::with_val(prec, cap / sqrt_b_abs);
        Complex::with_val(prec, &sqrt_b * &scale)
    };
    let mid = {
        let two = Float::with_val(prec, 2u32);
        let sum = Complex::with_val(prec, l_upper + l_lower);
        Complex::with_val(prec, &sum / &two)
    };
    let bump_ampl = Complex::with_val(prec, &target_mid - &mid);

    nodes
        .iter()
        .map(|t| {
            let scaled_t = Float::with_val(prec, &rate * t);
            let tanh_t = Float::with_val(prec, scaled_t.tanh_ref());
            let w_upper = Float::with_val(prec, (Float::with_val(prec, &one + &tanh_t)) * &half);
            let w_lower = Float::with_val(prec, &one - &w_upper);
            let scaled_upper = Complex::with_val(prec, l_upper * &w_upper);
            let scaled_lower = Complex::with_val(prec, l_lower * &w_lower);
            let asymp = Complex::with_val(prec, &scaled_upper + &scaled_lower);
            let cosh_t = Float::with_val(prec, scaled_t.cosh_ref());
            let sech_t = Float::with_val(prec, &one / &cosh_t);
            let bump = Complex::with_val(prec, &bump_ampl * &sech_t);
            Complex::with_val(prec, &asymp + &bump)
        })
        .collect()
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
    prec: u32,
) -> Complex {
    let cp1 = Complex::with_val(prec, (Float::with_val(prec, 1.5f64), 0));
    let cm1 = Complex::with_val(prec, (Float::with_val(prec, -0.5f64), 0));

    let mut r_int = cnum::zero(prec);
    let mut l_int = cnum::zero(prec);
    for k in 0..nodes.len() {
        let it = Complex::with_val(prec, (Float::new(prec), nodes[k].clone()));

        // F(c+1+it_k) = b^samples[k] = exp(ln_b · F)
        let exp_arg = Complex::with_val(prec, ln_b * &samples[k]);
        let b_f = Complex::with_val(prec, exp_arg.exp_ref());
        let cp1_plus_it = Complex::with_val(prec, &cp1 + &it);
        let denom_r = Complex::with_val(prec, &cp1_plus_it - z0);
        let term_r = Complex::with_val(prec, &b_f / &denom_r);
        r_int += Complex::with_val(prec, &term_r * &weights[k]);

        // F(c-1+it_k) = log_b(samples[k]) = ln(F) / ln_b
        let ln_s = Complex::with_val(prec, samples[k].ln_ref());
        let log_b_s = Complex::with_val(prec, &ln_s / ln_b);
        let cm1_plus_it = Complex::with_val(prec, &cm1 + &it);
        let denom_l = Complex::with_val(prec, &cm1_plus_it - z0);
        let term_l = Complex::with_val(prec, &log_b_s / &denom_l);
        l_int += Complex::with_val(prec, &term_l * &weights[k]);
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
        Float::with_val(prec, &nodes[1] - &nodes[0])
    } else {
        Float::with_val(prec, 1u32)
    };
    let h_f64 = h.to_f64();
    let t_max_f64 = t_max.to_f64();
    // Use a generous fixed K=10 here (matches the dispatcher's typical pick;
    // overestimating costs only K extra MPC ops).
    let n_terms = if std::env::var_os("TET_KOUZ_NO_EM").is_some() {
        0
    } else {
        em_n_terms(h_f64, t_max_f64, 50).max(1)
    };
    let em_h_powers = build_em_h_powers(&h, n_terms, prec);
    let (corr_r, corr_l) =
        compute_em_correction_z0(z0, t_max, l_upper, l_lower, &em_h_powers, prec);
    let r_int = Complex::with_val(prec, &r_int - &corr_r);
    let l_int = Complex::with_val(prec, &l_int - &corr_l);

    // Top edge contributes L̄ · ln((c-1+iT-z0)/(c+1+iT-z0)).
    let it_max = Complex::with_val(prec, (Float::new(prec), t_max.clone()));
    let neg_it_max = Complex::with_val(prec, -&it_max);
    let cm1_plus_itmax = Complex::with_val(prec, &cm1 + &it_max);
    let cp1_plus_itmax = Complex::with_val(prec, &cp1 + &it_max);
    let top_num = Complex::with_val(prec, &cm1_plus_itmax - z0);
    let top_den = Complex::with_val(prec, &cp1_plus_itmax - z0);
    let top_ratio = Complex::with_val(prec, &top_num / &top_den);
    let ln_top = Complex::with_val(prec, top_ratio.ln_ref());

    // Bottom edge contributes L · ln((c+1-iT-z0)/(c-1-iT-z0)).
    let cp1_minus_itmax = Complex::with_val(prec, &cp1 + &neg_it_max);
    let cm1_minus_itmax = Complex::with_val(prec, &cm1 + &neg_it_max);
    let bot_num = Complex::with_val(prec, &cp1_minus_itmax - z0);
    let bot_den = Complex::with_val(prec, &cm1_minus_itmax - z0);
    let bot_ratio = Complex::with_val(prec, &bot_num / &bot_den);
    let ln_bot = Complex::with_val(prec, bot_ratio.ln_ref());

    let pi_f = Float::with_val(prec, rug::float::Constant::Pi);
    let two_pi_f = Float::with_val(prec, &pi_f * 2u32);
    let two_pi_i = Complex::with_val(prec, (Float::new(prec), two_pi_f.clone()));

    let diff = Complex::with_val(prec, &r_int - &l_int);
    let part1 = Complex::with_val(prec, &diff / &two_pi_f);

    let up_term = Complex::with_val(prec, l_upper * &ln_top);
    let dn_term = Complex::with_val(prec, l_lower * &ln_bot);
    let upper_lower_sum = Complex::with_val(prec, &up_term + &dn_term);
    let part2 = Complex::with_val(prec, &upper_lower_sum / &two_pi_i);

    Complex::with_val(prec, &part1 + &part2)
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
    prec: u32,
) -> Vec<Complex> {
    let mut out = Vec::with_capacity(nodes.len());
    for k in 0..nodes.len() {
        let z0 = Complex::with_val(prec, (Float::with_val(prec, 0.5f64), nodes[k].clone()));
        out.push(cauchy_eval(
            &z0, samples, nodes, weights, t_max, l_upper, l_lower, ln_b, prec,
        ));
    }
    out
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
    prec: u32,
    digits: u64,
    use_schwarz: bool,
) -> Result<Vec<Complex>, String> {
    let n = initial.len();
    let n_int = n - 2; // number of interior samples that actually iterate
    let max_iters = 400usize;
    let target = 10f64.powf(-(digits as f64) - 3.0);
    let debug = cnum::verbose();

    let depth: usize = std::env::var("TET_KOUZ_ANDERSON_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let beta: f64 = std::env::var("TET_KOUZ_ANDERSON_BETA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);

    let mut x = initial;
    let mut prev_residual = f64::INFINITY;
    let mut stagnation = 0u32;
    let mut iters_since_best = 0u32;
    let mut best_x = x.clone();
    let mut best_residual = f64::INFINITY;

    // History of Δx[k] and Δr[k] (interior samples flattened as 1D vector).
    let mut hist_dx: Vec<Vec<Complex>> = Vec::new();
    let mut hist_dr: Vec<Vec<Complex>> = Vec::new();
    let mut prev_x_int: Option<Vec<Complex>> = None;
    let mut prev_r_int: Option<Vec<Complex>> = None;

    for iter in 0..max_iters {
        let f = apply_t(&x, nodes, weights, t_max, l_upper, l_lower, ln_b, prec);
        let mut r_int: Vec<Complex> = Vec::with_capacity(n_int);
        let mut x_int: Vec<Complex> = Vec::with_capacity(n_int);
        let mut r_norm = 0f64;
        for i in 1..n - 1 {
            let d = Complex::with_val(prec, &f[i] - &x[i]);
            let m = Float::with_val(prec, d.abs_ref()).to_f64();
            if m > r_norm {
                r_norm = m;
            }
            r_int.push(d);
            x_int.push(x[i].clone());
        }

        if r_norm < best_residual * (1.0 - 1e-9) {
            best_residual = r_norm;
            best_x = x.clone();
            iters_since_best = 0;
        } else {
            iters_since_best += 1;
        }

        if debug && (iter < 10 || iter % 25 == 0) {
            let mid_idx = nodes.len() / 2;
            let xm_re = Float::with_val(prec, x[mid_idx].real()).to_f64();
            let xm_im = Float::with_val(prec, x[mid_idx].imag()).to_f64();
            eprintln!(
                "kouz Anderson iter {:>4}: ‖r‖∞ = {:.3e}  depth={}  F(0.5)≈{:.4}+{:.4}i  best={:.3e}  (target {:.3e})",
                iter, r_norm, hist_dx.len(), xm_re, xm_im, best_residual, target
            );
        }

        if r_norm < target {
            return Ok(x);
        }
        if !r_norm.is_finite() {
            // Anderson can blow up after hitting the floor; return best iterate.
            if best_residual < f64::INFINITY {
                if debug {
                    eprintln!(
                        "kouz Anderson: residual non-finite at iter {}; returning best ({:.3e})",
                        iter, best_residual
                    );
                }
                return Ok(best_x);
            }
            return Err(format!(
                "Kouznetsov Anderson: residual non-finite at iter {}",
                iter
            ));
        }
        // Anderson typically descends fast then oscillates near the floor.
        // Two stop signals:
        //   * `stagnation` — consecutive monotonic-no-progress iterations
        //     (only triggers when Anderson genuinely flatlines).
        //   * `iters_since_best` — iterations since `best_residual` improved
        //     (catches oscillating-but-not-improving cases, where Anderson
        //     swings above and below the best without ever beating it).
        if r_norm > prev_residual * 0.999 {
            stagnation += 1;
            if stagnation > 6 {
                if debug {
                    eprintln!(
                        "kouz Anderson: monotonic stagnation at iter {}, returning best ({:.3e})",
                        iter, best_residual
                    );
                }
                return Ok(best_x);
            }
        } else {
            stagnation = 0;
        }
        if iters_since_best > 30 {
            if debug {
                eprintln!(
                    "kouz Anderson: best-stagnation at iter {} (no improvement for 30 iters), returning best ({:.3e})",
                    iter, best_residual
                );
            }
            return Ok(best_x);
        }
        prev_residual = r_norm;

        // Adaptive mixing: when residual is large the operator is far from
        // its fixed point and a full step (β=1) easily overshoots into log/exp
        // overflow territory. Damp aggressively until r is back below O(0.1),
        // then unleash the user-specified β. Also clear Anderson history while
        // we damp, so unstable past steps aren't extrapolated through.
        let effective_beta = if r_norm > 10.0 {
            0.02
        } else if r_norm > 1.0 {
            0.1
        } else if r_norm > 0.1 {
            0.5
        } else {
            beta
        };
        let beta_f = Float::with_val(prec, effective_beta);
        if effective_beta < 0.5 {
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
                dx.push(Complex::with_val(prec, &x_int[i] - &prev_x[i]));
                dr.push(Complex::with_val(prec, &r_int[i] - &prev_r[i]));
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
        let mut gamma: Vec<Complex> = vec![cnum::zero(prec); m];
        if m > 0 {
            let mut a: Vec<Vec<Complex>> = vec![vec![cnum::zero(prec); m]; m];
            let mut bvec: Vec<Complex> = vec![cnum::zero(prec); m];
            for j in 0..m {
                for k in 0..m {
                    let mut s = cnum::zero(prec);
                    for i in 0..n_int {
                        let conj_jk =
                            Complex::with_val(prec, hist_dr[j][i].conj_ref());
                        let prod = Complex::with_val(prec, &conj_jk * &hist_dr[k][i]);
                        s = Complex::with_val(prec, &s + &prod);
                    }
                    a[j][k] = s;
                }
                let mut s = cnum::zero(prec);
                for i in 0..n_int {
                    let conj_j =
                        Complex::with_val(prec, hist_dr[j][i].conj_ref());
                    let prod = Complex::with_val(prec, &conj_j * &r_int[i]);
                    s = Complex::with_val(prec, &s + &prod);
                }
                bvec[j] = s;
            }
            // Tikhonov regularization to handle near-singular A.
            let reg_f = Float::with_val(prec, 1e-12f64);
            for j in 0..m {
                a[j][j] = Complex::with_val(prec, &a[j][j] + &reg_f);
            }
            gamma = solve_complex_lin(&a, &bvec, prec)
                .unwrap_or_else(|_| vec![cnum::zero(prec); m]);
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
        let base_cap = if r_norm > 10.0 {
            0.1
        } else if r_norm > 1.0 {
            0.3
        } else {
            f64::INFINITY
        };

        let mut new_x_int: Vec<Complex> = Vec::with_capacity(n_int);
        for i in 0..n_int {
            let beta_r = Complex::with_val(prec, &r_int[i] * &beta_f);
            let mut step_i = beta_r;
            for j in 0..m {
                let beta_dr = Complex::with_val(prec, &hist_dr[j][i] * &beta_f);
                let term = Complex::with_val(prec, &hist_dx[j][i] + &beta_dr);
                let prod = Complex::with_val(prec, &term * &gamma[j]);
                step_i = Complex::with_val(prec, &step_i - &prod);
            }
            if base_cap.is_finite() {
                let step_mag = Float::with_val(prec, step_i.abs_ref()).to_f64();
                let x_mag = Float::with_val(prec, x_int[i].abs_ref()).to_f64();
                let cap_i = base_cap * (1.0 + x_mag);
                if step_mag > cap_i && step_mag.is_finite() {
                    let scale_f = Float::with_val(prec, cap_i / step_mag);
                    step_i = Complex::with_val(prec, &step_i * &scale_f);
                }
            }
            let update = Complex::with_val(prec, &x_int[i] + &step_i);
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
    }
    Err(format!(
        "Kouznetsov Anderson: no convergence in {} iters (residual {:.3e})",
        max_iters, prev_residual
    ))
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
    prec: u32,
    digits: u64,
    use_schwarz: bool,
) -> Result<Vec<Complex>, String> {
    let n = initial.len();
    let max_iters = 2000usize;
    let target = 10f64.powf(-(digits as f64) - 3.0);
    let debug = cnum::verbose();

    // Mixing parameter. α=1 is pure Picard. Smaller values (0.05–0.3) under-
    // relax to handle T's spectral radius near 1 in some directions.
    let alpha: f64 = std::env::var("TET_KOUZ_ALPHA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.3);
    let alpha_f = Float::with_val(prec, alpha);
    let one_minus_alpha = Float::with_val(prec, 1.0f64 - alpha);

    let mut x = initial;
    let mut prev_residual = f64::INFINITY;
    let mut stagnation = 0u32;

    for iter in 0..max_iters {
        let f = apply_t(&x, nodes, weights, t_max, l_upper, l_lower, ln_b, prec);
        let mut r_norm = 0f64;
        // Skip boundary samples: their values are pinned (see tetrate_kouznetsov),
        // and Cauchy at those z₀'s is degenerate anyway.
        for i in 1..n - 1 {
            let d = Complex::with_val(prec, &f[i] - &x[i]);
            let m = Float::with_val(prec, d.abs_ref()).to_f64();
            if m > r_norm {
                r_norm = m;
            }
        }
        if debug && (iter < 20 || iter % 50 == 0) {
            let mid_idx = nodes.len() / 2;
            let xm_re = Float::with_val(prec, x[mid_idx].real()).to_f64();
            let xm_im = Float::with_val(prec, x[mid_idx].imag()).to_f64();
            // Find the index where residual is max (interior only).
            let mut max_idx = 1usize;
            let mut max_val = 0f64;
            for i in 1..n - 1 {
                let d = Complex::with_val(prec, &f[i] - &x[i]);
                let m = Float::with_val(prec, d.abs_ref()).to_f64();
                if m > max_val {
                    max_val = m;
                    max_idx = i;
                }
            }
            let max_t = Float::with_val(prec, &nodes[max_idx]).to_f64();
            eprintln!(
                "kouz Picard iter {:>4}: ‖r‖∞ = {:.3e}  α={:.2}  F(0.5)≈{:.4}+{:.4}i  argmax_t={:.3} (idx {})",
                iter, r_norm, alpha, xm_re, xm_im, max_t, max_idx
            );
        }
        if r_norm < target {
            return Ok(x);
        }
        if !r_norm.is_finite() {
            return Err(format!(
                "Kouznetsov Picard: residual non-finite at iter {} (α={})",
                iter, alpha
            ));
        }
        if r_norm > prev_residual * 0.999 {
            stagnation += 1;
            if stagnation > 30 {
                return Err(format!(
                    "Kouznetsov Picard: stagnated at residual {:.3e} after {} iters (α={})",
                    r_norm, iter, alpha
                ));
            }
        } else {
            stagnation = 0;
        }
        prev_residual = r_norm;

        // Mix interior samples; keep boundary pinned to L_lower / L_upper.
        let mut x_new = Vec::with_capacity(n);
        x_new.push(x[0].clone());
        for i in 1..n - 1 {
            let lhs = Complex::with_val(prec, &x[i] * &one_minus_alpha);
            let rhs = Complex::with_val(prec, &f[i] * &alpha_f);
            x_new.push(Complex::with_val(prec, &lhs + &rhs));
        }
        x_new.push(x[n - 1].clone());
        if use_schwarz {
            symmetrize_schwarz(&mut x_new, prec);
        }
        x = x_new;
    }
    Err(format!(
        "Kouznetsov Picard: no convergence in {} iters (residual {:.3e})",
        max_iters, prev_residual
    ))
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
/// Newton-Kantorovich theory says solving the inner linear system to
/// `min(0.1, ‖r‖)` relative tolerance preserves quadratic convergence; we
/// pick `1e-3` as a uniform safe choice.
///
/// Damped step: full Newton step is taken first; if it *increases* residual,
/// halve until it doesn't (up to 6 halvings). On line-search exhaustion we
/// abort so that the dispatcher can surface a clean error.
#[allow(clippy::too_many_arguments)]
fn iterate_newton(
    initial: Vec<Complex>,
    nodes: &[Float],
    weights: &[Float],
    t_max: &Float,
    l_upper: &Complex,
    l_lower: &Complex,
    ln_b: &Complex,
    prec: u32,
    digits: u64,
    use_schwarz: bool,
) -> Result<Vec<Complex>, String> {
    let n = initial.len();
    // 80-iter cap: small/medium bases converge quadratically in ~10-15 iters
    // and never approach the cap. Larger bases (b≥50) sit in linear-descent
    // for ~30-50 iters before Newton kicks in and finishes in 2-3 quadratic
    // iters. The slow-progress check (every 15 iters from iter 10) catches
    // non-canonical W_k cases earlier with the best-so-far residual, so the
    // cap only matters for the slow-but-eventually-converging large-base
    // regime.
    let max_iters = 80usize;
    let target = 10f64.powf(-(digits as f64) - 3.0);
    let debug = cnum::verbose();

    let mut x = initial;
    let mut prev_residual = f64::INFINITY;
    let mut stagnation = 0u32;
    let mut best_x = x.clone();
    let mut best_residual = f64::INFINITY;
    // Snapshot of best_residual every 30 iters; if we haven't dropped by 10x
    // since the snapshot, abandon (slow-progress stagnation, distinct from
    // the per-iter stagnation check below). For real positive bases the
    // iteration converges quadratically once close, so 10x in 30 iters is a
    // very loose bound. For W_k-search bases (negative real) the iteration
    // typically hits a discretization floor and never converges to target.
    let mut slow_progress_anchor: f64 = f64::INFINITY;
    let mut slow_progress_anchor_iter: usize = 0;
    // Levenberg-Marquardt damping. μ adapts: shrink on success (gain a faster,
    // more Newton-like step next time), grow on rejection (more like gradient
    // descent). Starts at a moderate value so the first step doesn't blow up.
    let mut mu = 1.0f64;

    // FFT-domain kernels. The right/left edge denominator inverses depend
    // only on the uniform grid spacing — building them once per call collapses
    // every later matvec to two length-`m` FFTs (m = next_pow2(3N − 2)). The
    // same struct also carries per-row Euler-Maclaurin corrections, so the
    // residual evaluator just adds a precomputed shift instead of
    // recomputing closed-form boundary terms each iteration.
    let kernel_start = std::time::Instant::now();
    let kernels = build_cauchy_kernels(nodes, t_max, l_upper, l_lower, prec, digits);
    if debug {
        eprintln!("kouz phase: build_cauchy_kernels done ({:.2}s)", kernel_start.elapsed().as_secs_f64());
    }

    let mid_idx = nodes.len() / 2;
    if debug {
        eprintln!(
            "kouz LM start: n={} mid_idx={} mid_t={} F[mid]={}+{}i",
            nodes.len(),
            mid_idx,
            Float::with_val(prec, &nodes[mid_idx]).to_f64(),
            Float::with_val(prec, x[mid_idx].real()).to_f64(),
            Float::with_val(prec, x[mid_idx].imag()).to_f64(),
        );
    }

    for iter in 0..max_iters {
        let iter_start = std::time::Instant::now();
        let matvec_start = std::time::Instant::now();
        let f = apply_t_fft(&x, nodes, weights, t_max, l_upper, l_lower, ln_b, &kernels, prec);
        let matvec_secs = matvec_start.elapsed().as_secs_f64();
        let mut r = Vec::with_capacity(n);
        let mut r_norm = 0f64;
        for i in 0..n {
            let d = Complex::with_val(prec, &f[i] - &x[i]);
            // Boundary samples are pinned (corner Cauchy singularity); zero
            // out their residual so the linear solve does not try to update
            // them. We track r_norm only over interior samples for the same
            // reason — boundary "residuals" are pure discretization noise.
            if i == 0 || i == n - 1 {
                r.push(cnum::zero(prec));
                continue;
            }
            let m = Float::with_val(prec, d.abs_ref()).to_f64();
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
            let xm_re = Float::with_val(prec, x[mid_idx].real()).to_f64();
            let xm_im = Float::with_val(prec, x[mid_idx].imag()).to_f64();
            eprintln!(
                "kouz LM iter {:>3}: ‖r‖∞ = {:.3e}  μ={:.2e}  F(0.5)≈{:.4}+{:.4}i  (target {:.3e})  matvec={:.2}s",
                iter, r_norm, mu, xm_re, xm_im, target, matvec_secs
            );
        }

        if r_norm.is_nan() || !r_norm.is_finite() {
            if debug {
                eprintln!(
                    "kouz LM: residual non-finite at iter {}; best so far {:.3e}",
                    iter, best_residual
                );
            }
            return validate_best_residual(best_residual, digits, use_schwarz, "residual became non-finite")
                .map(|_| best_x);
        }
        if r_norm < best_residual {
            best_residual = r_norm;
            best_x = x.clone();
        }
        if r_norm < target {
            return Ok(x);
        }
        // Slow-progress stagnation: take a snapshot of best_residual every 15
        // iters (after iter 10 to skip the initial transient). If 15 iters
        // later best_residual hasn't dropped by 10x, abandon. This catches
        // cases where each iter makes ~1% progress but we'll never reach
        // `target` (e.g., W_k-search negative real bases hitting the
        // discretization floor at ~1e-8 regardless of node count).
        if iter == 10 {
            slow_progress_anchor = best_residual;
            slow_progress_anchor_iter = 10;
        } else if iter > 10 && iter == slow_progress_anchor_iter + 15 {
            if best_residual > 0.1 * slow_progress_anchor {
                if debug {
                    eprintln!(
                        "kouz LM: slow-progress stagnation at iter {} (anchor {:.3e} -> best {:.3e}, <10x in 15 iters)",
                        iter, slow_progress_anchor, best_residual
                    );
                }
                return validate_best_residual(best_residual, digits, use_schwarz, "slow-progress stagnation")
                    .map(|_| best_x);
            }
            slow_progress_anchor = best_residual;
            slow_progress_anchor_iter = iter;
        }
        // Stagnation: residual not dropping ≥ 0.1% for 8 iters. We've likely
        // converged to the discrete fixed point exactly but its residual is
        // bounded below by the trapezoidal/contour discretization error
        // (~10^-3 to 10^-5 depending on b and N). Return the best iterate.
        if r_norm > prev_residual * 0.999 {
            stagnation += 1;
            if stagnation > 8 {
                if debug {
                    eprintln!(
                        "kouz LM: stagnated at iter {} (residual {:.3e}, likely discretization floor); best {:.3e}",
                        iter, r_norm, best_residual
                    );
                }
                return validate_best_residual(best_residual, digits, use_schwarz, "per-iter stagnation")
                    .map(|_| best_x);
            }
        } else {
            stagnation = 0;
        }
        prev_residual = r_norm;

        // Precompute J = I − DT helpers ONCE per Newton iteration. They
        // depend on `x` but not on the LM damping μ, so we hoist them out
        // of the inner LM line search.
        let (b_f_ln, inv_f_ln) = precompute_dt_factors(&x, ln_b, prec);

        // Levenberg-Marquardt: solve (J + μ·I)·δ = r. Inner loop adaptively
        // grows μ until the trial step actually decreases residual; then
        // shrinks μ for the next outer iteration. 25 trials is enough to
        // grow μ from 1e-15 (deep-Newton end) all the way to ~1e0 (full
        // gradient-descent regime) — which lets large-base cases like
        // b=200 escape transient saddles where the "natural" Newton step
        // overshoots into a worse basin.
        let mut accepted = false;
        for _lm_try in 0..25 {
            // Boundary-pinned matvec: rows 0 and n-1 act as identity (so
            // δ[0] = r[0] = 0 and δ[n-1] = r[n-1] = 0 stay zero throughout
            // the Krylov build); interior rows compute (1+μ)·v − DT·v.
            let mu_local = mu;
            let matvec = |v: &[Complex]| -> Vec<Complex> {
                let dt_v = apply_dt_v_fft(&b_f_ln, &inv_f_ln, &kernels, weights, v, prec);
                let scale = Float::with_val(prec, 1.0 + mu_local);
                let mut out = Vec::with_capacity(n);
                for k in 0..n {
                    let scaled = Complex::with_val(prec, &v[k] * &scale);
                    out.push(Complex::with_val(prec, &scaled - &dt_v[k]));
                }
                out[0] = v[0].clone();
                out[n - 1] = v[n - 1].clone();
                out
            };
            // Krylov dimension and restart count: for typical Cauchy-Newton
            // matrices (well-conditioned with LM damping), 80 inner steps
            // are plenty; 8 restarts buys robustness against difficult
            // bases without ballooning memory (V matrix uses 80·N complex
            // words, which is ≪ the dense N×N Jacobian we're avoiding).
            //
            // Eisenstat-Walker type 2 forcing: η_rel = 0.1·r_norm gives
            // ‖J·δ + r‖ ≤ 0.1·r_norm·‖r‖ = 0.1·r_norm² in absolute terms,
            // which delivers quadratic Newton-Krylov convergence (each
            // outer iteration roughly squares the residual reduction).
            //
            // `gmres_complex` takes RELATIVE tolerance (vs ‖rhs‖ = ‖r‖):
            //   target_abs = ‖rhs‖ · tol_rel
            // To request quadratic forcing we set tol_rel = 0.1·r_norm.
            // For very small r_norm this would push target_abs below MPC's
            // working-precision noise floor, so we cap absolute target
            // against `10^{-(digits+5)}` and back-convert to relative form.
            // Cap relative tol at 1e-3 to skip pointless tight solves when
            // r_norm is still O(1).
            let abs_floor = 10f64.powf(-(digits as f64) - 5.0);
            let abs_target = (0.1f64 * r_norm * r_norm).max(abs_floor);
            let inner_tol = if r_norm > 0.0 {
                (abs_target / r_norm).min(1e-3)
            } else {
                1e-3
            };
            let restart = 80.min(n);
            let delta = match gmres_complex(matvec, &r, inner_tol, 8, restart, prec) {
                Ok(d) => d,
                Err(e) => {
                    if debug {
                        eprintln!(
                            "kouz LM iter {}: GMRES failed (tol={:.2e}): {}",
                            iter, inner_tol, e
                        );
                    }
                    mu *= 4.0;
                    continue;
                }
            };

            // Newton step cap: when the step is much larger than the current
            // residual, the linearization is being trusted further than it
            // should be. Cap ‖δ‖∞ at `4·max(‖r‖∞, 0.05)` so that as r→0 the
            // method takes shorter, more reliable steps. Without this, big
            // steps near a small residual can land in a region where J is
            // near-singular and the next iteration cannot recover.
            let max_step = 4.0 * r_norm.max(0.05);
            let mut delta_inf = 0f64;
            for d in &delta {
                let m = Float::with_val(prec, d.abs_ref()).to_f64();
                if m > delta_inf {
                    delta_inf = m;
                }
            }
            let scale = if delta_inf > max_step {
                max_step / delta_inf
            } else {
                1.0
            };
            let scale_f = Float::with_val(prec, scale);

            let mut x_trial = Vec::with_capacity(n);
            for i in 0..n {
                let scaled = Complex::with_val(prec, &delta[i] * &scale_f);
                x_trial.push(Complex::with_val(prec, &x[i] + &scaled));
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
            // Cheap basin guard: for real-positive bases > e^(1/e), the natural
            // Kneser F is real and POSITIVE on Re(z)=0. If the trial step
            // pushes F[mid] negative, we have crossed into a wrong basin
            // (e.g. settling at F[mid]≈−0.7 with a local-but-not-global
            // residual minimum). Reject the step BEFORE the expensive
            // apply_t_fft / residual computation — this saves ~1 matvec per
            // bad LM trial, which matters because large bases trigger many
            // such trials per outer iteration.
            let mid_idx = n / 2;
            let f_mid_re_trial = Float::with_val(prec, x_trial[mid_idx].real()).to_f64();
            if use_schwarz && f_mid_re_trial < 0.0 {
                mu *= 4.0;
                if mu > 1e8 {
                    break;
                }
                continue;
            }
            let f_trial = apply_t_fft(
                &x_trial, nodes, weights, t_max, l_upper, l_lower, ln_b, &kernels, prec,
            );
            let mut r_trial_norm = 0f64;
            let mut bad = false;
            for i in 1..n - 1 {
                let dd = Complex::with_val(prec, &f_trial[i] - &x_trial[i]);
                let m = Float::with_val(prec, dd.abs_ref()).to_f64();
                if !m.is_finite() {
                    bad = true;
                    break;
                }
                if m > r_trial_norm {
                    r_trial_norm = m;
                }
            }

            if !bad && r_trial_norm < r_norm {
                x = x_trial;
                // Successful step: shrink μ aggressively when we beat the
                // residual by a healthy margin (push toward Newton); shrink
                // mildly for marginal improvement.
                if r_trial_norm < r_norm * 0.5 {
                    mu *= 0.25;
                } else {
                    mu *= 0.7;
                }
                mu = mu.max(1e-15);
                accepted = true;
                break;
            }
            // Reject: grow μ to take a smaller, more gradient-like step.
            mu *= 4.0;
            if mu > 1e8 {
                break;
            }
        }
        if !accepted {
            // LM line-search exhausted (μ has grown without finding a step
            // that decreases the residual). This typically means we are at
            // the discretization floor — the discrete fixed point's residual
            // is bounded below by the trapezoidal/contour truncation error,
            // and Newton has nowhere left to go. Return the best iterate so
            // the dispatcher can deliver an answer at the discretization-
            // limited precision rather than failing the whole call.
            if debug {
                eprintln!(
                    "kouz LM: no descent step at iter {} (μ={:.2e}); best residual {:.3e}",
                    iter, mu, best_residual
                );
            }
            return validate_best_residual(best_residual, digits, use_schwarz, "no descent step")
                .map(|_| best_x);
        }
        if debug {
            eprintln!(
                "kouz LM iter {:>3}: total wall {:.2}s",
                iter,
                iter_start.elapsed().as_secs_f64()
            );
        }
    }

    validate_best_residual(
        best_residual,
        digits,
        use_schwarz,
        &format!("did not converge in {} iterations", max_iters),
    )
    .map(|_| best_x)
}

/// Decide whether `best_residual` from a non-converged LM run is close enough
/// to the discretization floor that returning it as a best-effort answer is
/// honest, or whether it represents a catastrophic failure (wrong basin /
/// numerical garbage) and we should refuse to return a wrong answer.
///
/// Two regimes:
///   * `use_schwarz = true` (real-positive bases > e^(1/e)): the iteration
///     should converge cleanly because Schwarz reflection halves the unknowns
///     and pins the symmetry. Failure to drop residual below ~10^(-digits/3)
///     means LM ended in a wrong basin (e.g. b=200 settling at F[mid]≈−0.75
///     with residual 0.5, which then produces a numerical-garbage F̃(0.5)).
///     Threshold is `10^(-digits/3)` clamped to `[1e-6, 1e-3]`.
///   * `use_schwarz = false` (complex / negative-real / unit-circle bases):
///     the iteration is harder — Newton may stall at a discretization floor
///     well above zero while the Cauchy reconstruction at user heights is
///     still smooth and satisfies `F(z+1) = b^F(z)` to many digits. Be more
///     permissive: accept best-effort up to residual ~5, refusing only the
///     truly catastrophic cases (non-finite, residual ≥ 5).
fn validate_best_residual(
    best_residual: f64,
    digits: u64,
    use_schwarz: bool,
    reason: &str,
) -> Result<(), String> {
    let threshold = if use_schwarz {
        10f64.powf(-(digits as f64) / 3.0).clamp(1e-6, 1e-3)
    } else {
        5.0
    };
    if best_residual.is_finite() && best_residual <= threshold {
        Ok(())
    } else {
        Err(format!(
            "Kouznetsov Newton did not converge ({reason}); best residual {best_residual:.3e} \
             exceeds acceptance threshold {threshold:.3e} for {digits} digits"
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
    prec: u32,
) -> Vec<Vec<Complex>> {
    let n = nodes.len();
    let pi_f = Float::with_val(prec, rug::float::Constant::Pi);
    let two_pi = Float::with_val(prec, &pi_f * 2u32);

    // Precompute b^F[j]·ln(b) and 1/(F[j]·ln(b)), needed in every column.
    let mut b_f_ln: Vec<Complex> = Vec::with_capacity(n);
    let mut inv_f_ln: Vec<Complex> = Vec::with_capacity(n);
    for j in 0..n {
        let exp_arg = Complex::with_val(prec, ln_b * &samples[j]);
        let bf = Complex::with_val(prec, exp_arg.exp_ref());
        b_f_ln.push(Complex::with_val(prec, &bf * ln_b));
        let f_ln = Complex::with_val(prec, &samples[j] * ln_b);
        let one_c = Complex::with_val(prec, (Float::with_val(prec, 1u32), 0));
        inv_f_ln.push(Complex::with_val(prec, &one_c / &f_ln));
    }

    let one_re = Float::with_val(prec, 1u32);
    let neg_one_re = Float::with_val(prec, -1i32);
    let mut jac = vec![vec![cnum::zero(prec); n]; n];

    for k in 0..n {
        for j in 0..n {
            let dt_im = Float::with_val(prec, &nodes[j] - &nodes[k]);
            // 1+i(t_j-t_k)
            let denom_r = Complex::with_val(prec, (one_re.clone(), dt_im.clone()));
            // -1+i(t_j-t_k)
            let denom_l = Complex::with_val(prec, (neg_one_re.clone(), dt_im));

            let term_r = Complex::with_val(prec, &b_f_ln[j] / &denom_r);
            let term_l = Complex::with_val(prec, &inv_f_ln[j] / &denom_l);
            let bracket = Complex::with_val(prec, &term_r - &term_l);
            let scaled = Complex::with_val(prec, &bracket * &weights[j]);
            let dt_kj = Complex::with_val(prec, &scaled / &two_pi);

            // J = I − DT
            if k == j {
                jac[k][j] = Complex::with_val(prec, Complex::with_val(prec, (1, 0)) - &dt_kj);
            } else {
                jac[k][j] = Complex::with_val(prec, -&dt_kj);
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
    prec: u32,
) -> Result<Vec<Complex>, String> {
    let n = a_in.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut a: Vec<Vec<Complex>> = a_in.iter().map(|row| row.clone()).collect();
    let mut b: Vec<Complex> = b_in.to_vec();

    for k in 0..n {
        // Partial pivot: find row with largest |a[i][k]| for i ≥ k.
        let mut pivot = k;
        let mut max_abs = Float::with_val(prec, a[k][k].abs_ref());
        for i in (k + 1)..n {
            let abs_i = Float::with_val(prec, a[i][k].abs_ref());
            if abs_i > max_abs {
                max_abs = abs_i;
                pivot = i;
            }
        }
        if max_abs.to_f64() < 1e-100 {
            return Err(format!("singular at column {}", k));
        }
        if pivot != k {
            a.swap(k, pivot);
            b.swap(k, pivot);
        }

        // Eliminate below.
        for i in (k + 1)..n {
            let factor = Complex::with_val(prec, &a[i][k] / &a[k][k]);
            for j in k..n {
                let prod = Complex::with_val(prec, &factor * &a[k][j]);
                a[i][j] = Complex::with_val(prec, &a[i][j] - &prod);
            }
            let prod = Complex::with_val(prec, &factor * &b[k]);
            b[i] = Complex::with_val(prec, &b[i] - &prod);
        }
    }

    // Back-substitute.
    let mut x = vec![cnum::zero(prec); n];
    for i in (0..n).rev() {
        let mut sum = b[i].clone();
        for j in (i + 1)..n {
            let prod = Complex::with_val(prec, &a[i][j] * &x[j]);
            sum = Complex::with_val(prec, &sum - &prod);
        }
        x[i] = Complex::with_val(prec, &sum / &a[i][i]);
    }
    Ok(x)
}

#[allow(clippy::too_many_arguments)]
fn eval_at_height(
    b: &Complex,
    h: &Complex,
    samples: &[Complex],
    nodes: &[Float],
    weights: &[Float],
    t_max: &Float,
    l_upper: &Complex,
    l_lower: &Complex,
    ln_b: &Complex,
    prec: u32,
) -> Result<Complex, String> {
    // Integer-shift h so 0 ≤ Re(h_strip) ≤ 1.
    let re_h = h.real().to_f64();
    let shift = re_h.floor() as i64;
    let h_strip = Complex::with_val(prec, h - shift);

    // Asymptotic regime: when |Im(h_strip)| exceeds the Cauchy contour height
    // t_max, h is outside the rectangle and the Cauchy formula is invalid (it
    // would integrate to ~0 by holomorphy, or NaN at the boundary). The
    // natural Kneser-Kouznetsov F has well-defined limits:
    //   F(z) → L_upper as Im(z) → +∞
    //   F(z) → L_lower as Im(z) → −∞
    // and the residual decays like exp(−|Im(z)|·|arg(λ)|), so for Im outside
    // ±t_max the asymptote is accurate to the requested precision (since
    // t_max was chosen as (digits+8)·ln(10)/|arg(λ)|).
    //
    // After picking the asymptote we still apply the b^· (or log_b) recursion
    // for the `shift` integer Re-translation. Since L_upper / L_lower are
    // fixed points of b^·, b^L = L, so the recursion is a no-op in exact
    // arithmetic — but we run it anyway for symmetry with the in-strip path.
    let im_h_strip = Float::with_val(prec, h_strip.imag()).to_f64();
    let t_max_f64 = t_max.to_f64();
    if im_h_strip > t_max_f64 {
        // Above the contour: F(z) → L_upper. Since b^L_upper = L_upper
        // (fixed-point identity), iterating b^· `shift` times is a no-op in
        // exact arithmetic and only injects rounding noise — return L_upper
        // directly. Same logic for the lower asymptote.
        return Ok(l_upper.clone());
    }
    if im_h_strip < -t_max_f64 {
        return Ok(l_lower.clone());
    }
    let f_strip = cauchy_eval(
        &h_strip, samples, nodes, weights, t_max, l_upper, l_lower, ln_b, prec,
    );

    let mut f = f_strip;
    if shift > 0 {
        for _ in 0..shift {
            let exp_arg = Complex::with_val(prec, ln_b * &f);
            f = Complex::with_val(prec, exp_arg.exp_ref());
        }
    } else if shift < 0 {
        for _ in 0..(-shift) {
            let ln_f = Complex::with_val(prec, f.ln_ref());
            f = Complex::with_val(prec, &ln_f / ln_b);
        }
    }
    let _ = b; // referenced to keep the signature stable; recursion uses ln_b.
    Ok(f)
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
    prec: u32,
) -> (Vec<Complex>, Vec<Complex>) {
    let n = samples.len();
    let mut b_f_ln = Vec::with_capacity(n);
    let mut inv_f_ln = Vec::with_capacity(n);
    let one_c = Complex::with_val(prec, (Float::with_val(prec, 1u32), 0));
    for j in 0..n {
        let exp_arg = Complex::with_val(prec, ln_b * &samples[j]);
        let bf = Complex::with_val(prec, exp_arg.exp_ref());
        b_f_ln.push(Complex::with_val(prec, &bf * ln_b));
        let f_ln = Complex::with_val(prec, &samples[j] * ln_b);
        inv_f_ln.push(Complex::with_val(prec, &one_c / &f_ln));
    }
    (b_f_ln, inv_f_ln)
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
fn precompute_denom_inverses(
    nodes: &[Float],
    prec: u32,
) -> (Vec<Complex>, Vec<Complex>) {
    let n = nodes.len();
    let mut inv_r: Vec<Complex> = Vec::with_capacity(2 * n - 1);
    let mut inv_l: Vec<Complex> = Vec::with_capacity(2 * n - 1);
    let one_re = Float::with_val(prec, 1u32);
    let neg_one_re = Float::with_val(prec, -1i32);
    let one_c = Complex::with_val(prec, (Float::with_val(prec, 1u32), 0));
    let delta = if n >= 2 {
        Float::with_val(prec, &nodes[1] - &nodes[0])
    } else {
        Float::with_val(prec, 1u32)
    };
    let max_offset = (n as i64) - 1;
    for d_idx in 0..(2 * n - 1) {
        let d_signed = (d_idx as i64) - max_offset;
        let dt = Float::with_val(prec, &delta * d_signed);
        let dr = Complex::with_val(prec, (one_re.clone(), dt.clone()));
        let dl = Complex::with_val(prec, (neg_one_re.clone(), dt));
        inv_r.push(Complex::with_val(prec, &one_c / &dr));
        inv_l.push(Complex::with_val(prec, &one_c / &dl));
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
    prec: u32,
    digits: u64,
) -> CauchyKernels {
    let n = nodes.len();
    let one_re = Float::with_val(prec, 1u32);
    let neg_one_re = Float::with_val(prec, -1i32);
    let one_c = Complex::with_val(prec, (one_re.clone(), Float::new(prec)));
    let delta = if n >= 2 {
        Float::with_val(prec, &nodes[1] - &nodes[0])
    } else {
        Float::with_val(prec, 1u32)
    };
    let max_offset = (n as i64) - 1;
    let mut h_r: Vec<Complex> = Vec::with_capacity(2 * n - 1);
    let mut h_l: Vec<Complex> = Vec::with_capacity(2 * n - 1);
    for d_idx in 0..(2 * n - 1) {
        let d_signed = (d_idx as i64) - max_offset;
        let dt = Float::with_val(prec, &delta * d_signed);
        let dr = Complex::with_val(prec, (one_re.clone(), dt.clone()));
        let dl = Complex::with_val(prec, (neg_one_re.clone(), dt));
        h_r.push(Complex::with_val(prec, &one_c / &dr));
        h_l.push(Complex::with_val(prec, &one_c / &dl));
    }
    let right = precompute_kernel_fft(&h_r, n, prec);
    let left = precompute_kernel_fft(&h_l, n, prec);

    // EM corrections per row.
    let h_f64 = delta.to_f64();
    let t_max_f64 = t_max.to_f64();
    let n_terms = em_n_terms(h_f64, t_max_f64, digits);
    let em_h_powers = build_em_h_powers(&delta, n_terms, prec);

    let pi_f = Float::with_val(prec, Constant::Pi);
    let two_pi = Float::with_val(prec, &pi_f * 2u32);
    let inv_two_pi = Float::with_val(prec, Float::with_val(prec, 1u32) / &two_pi);

    let mut em_right_over_2pi = Vec::with_capacity(n);
    let mut em_left_over_2pi = Vec::with_capacity(n);
    for k in 0..n {
        let z0 = Complex::with_val(prec, (Float::with_val(prec, 0.5f64), nodes[k].clone()));
        let (corr_r, corr_l) = compute_em_correction_z0(
            &z0,
            t_max,
            l_upper,
            l_lower,
            &em_h_powers,
            prec,
        );
        em_right_over_2pi.push(Complex::with_val(prec, &corr_r * &inv_two_pi));
        em_left_over_2pi.push(Complex::with_val(prec, &corr_l * &inv_two_pi));
    }

    if cnum::verbose() {
        eprintln!(
            "kouz EM: K={} (h/T={:.3e})",
            n_terms,
            (h_f64 / t_max_f64.max(1e-300)).abs()
        );
    }

    CauchyKernels {
        right,
        left,
        em_right_over_2pi,
        em_left_over_2pi,
    }
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
    prec: u32,
) -> Vec<Complex> {
    let n = b_f_ln.len();
    let pi_f = Float::with_val(prec, Constant::Pi);
    let two_pi = Float::with_val(prec, &pi_f * 2u32);
    let one_re = Float::with_val(prec, 1u32);
    let inv_two_pi = Float::with_val(prec, &one_re / &two_pi);

    // Pre-scale: a_r[j] = b_f_ln[j]·v[j]·w_j/(2π), a_l[j] similar.
    let mut a_r: Vec<Complex> = Vec::with_capacity(n);
    let mut a_l: Vec<Complex> = Vec::with_capacity(n);
    for j in 0..n {
        let w_over_2pi = Float::with_val(prec, &weights[j] * &inv_two_pi);
        let bv = Complex::with_val(prec, &b_f_ln[j] * &v[j]);
        a_r.push(Complex::with_val(prec, &bv * &w_over_2pi));
        let iv = Complex::with_val(prec, &inv_f_ln[j] * &v[j]);
        a_l.push(Complex::with_val(prec, &iv * &w_over_2pi));
    }

    let r_part = cross_correlate_with_kernel(&a_r, &kernels.right, prec);
    let l_part = cross_correlate_with_kernel(&a_l, &kernels.left, prec);

    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        out.push(Complex::with_val(prec, &r_part[k] - &l_part[k]));
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
    prec: u32,
) -> Vec<Complex> {
    let n = samples.len();
    let pi_f = Float::with_val(prec, Constant::Pi);
    let two_pi_f = Float::with_val(prec, &pi_f * 2u32);
    let two_pi_i = Complex::with_val(prec, (Float::new(prec), two_pi_f.clone()));
    let one_re = Float::with_val(prec, 1u32);
    let inv_two_pi = Float::with_val(prec, &one_re / &two_pi_f);

    // Build right- and left-edge values, pre-scaled by w_j / (2π) so the
    // cross-correlation result is the in-strip part of T(F).
    let mut a_r: Vec<Complex> = Vec::with_capacity(n);
    let mut a_l: Vec<Complex> = Vec::with_capacity(n);
    for j in 0..n {
        let w_over_2pi = Float::with_val(prec, &weights[j] * &inv_two_pi);
        let exp_arg = Complex::with_val(prec, ln_b * &samples[j]);
        let bf = Complex::with_val(prec, exp_arg.exp_ref());
        a_r.push(Complex::with_val(prec, &bf * &w_over_2pi));
        let ln_s = Complex::with_val(prec, samples[j].ln_ref());
        let log_b_s = Complex::with_val(prec, &ln_s / ln_b);
        a_l.push(Complex::with_val(prec, &log_b_s * &w_over_2pi));
    }

    let r_part = cross_correlate_with_kernel(&a_r, &kernels.right, prec);
    let l_part = cross_correlate_with_kernel(&a_l, &kernels.left, prec);

    // Boundary corrections (per row, O(N) total).
    let cp1 = Complex::with_val(prec, (Float::with_val(prec, 1.5f64), 0));
    let cm1 = Complex::with_val(prec, (Float::with_val(prec, -0.5f64), 0));
    let it_max = Complex::with_val(prec, (Float::new(prec), t_max.clone()));
    let neg_it_max = Complex::with_val(prec, -&it_max);
    let cm1_plus_itmax = Complex::with_val(prec, &cm1 + &it_max);
    let cp1_plus_itmax = Complex::with_val(prec, &cp1 + &it_max);
    let cp1_minus_itmax = Complex::with_val(prec, &cp1 + &neg_it_max);
    let cm1_minus_itmax = Complex::with_val(prec, &cm1 + &neg_it_max);

    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let z0 = Complex::with_val(prec, (Float::with_val(prec, 0.5f64), nodes[k].clone()));

        // Euler-Maclaurin boundary correction: subtract the closed-form O(h²)
        // (and higher-order, up to K terms) error of the trapezoidal sums.
        // `em_*_over_2pi` is precomputed in `build_cauchy_kernels` and already
        // carries the 1/(2π) factor that `r_part`, `l_part` carry.
        let r_corrected = Complex::with_val(prec, &r_part[k] - &kernels.em_right_over_2pi[k]);
        let l_corrected = Complex::with_val(prec, &l_part[k] - &kernels.em_left_over_2pi[k]);
        let part1 = Complex::with_val(prec, &r_corrected - &l_corrected);

        let top_num = Complex::with_val(prec, &cm1_plus_itmax - &z0);
        let top_den = Complex::with_val(prec, &cp1_plus_itmax - &z0);
        let top_ratio = Complex::with_val(prec, &top_num / &top_den);
        let ln_top = Complex::with_val(prec, top_ratio.ln_ref());

        let bot_num = Complex::with_val(prec, &cp1_minus_itmax - &z0);
        let bot_den = Complex::with_val(prec, &cm1_minus_itmax - &z0);
        let bot_ratio = Complex::with_val(prec, &bot_num / &bot_den);
        let ln_bot = Complex::with_val(prec, bot_ratio.ln_ref());

        let up_term = Complex::with_val(prec, l_upper * &ln_top);
        let dn_term = Complex::with_val(prec, l_lower * &ln_bot);
        let upper_lower_sum = Complex::with_val(prec, &up_term + &dn_term);
        let part2 = Complex::with_val(prec, &upper_lower_sum / &two_pi_i);

        out.push(Complex::with_val(prec, &part1 + &part2));
    }
    out
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
    prec: u32,
) -> Vec<Complex> {
    let n = b_f_ln.len();
    let pi_f = Float::with_val(prec, rug::float::Constant::Pi);
    let two_pi = Float::with_val(prec, &pi_f * 2u32);
    let one_re = Float::with_val(prec, 1u32);
    let inv_two_pi = Float::with_val(prec, &one_re / &two_pi);

    // Pre-scale: `bfl_v[j]` = b_f_ln[j]·v[j]·w_j/(2π), `ifl_v[j]` similar.
    // Pulling the scalar out of the inner k-loop saves N² multiplications.
    let mut bfl_v: Vec<Complex> = Vec::with_capacity(n);
    let mut ifl_v: Vec<Complex> = Vec::with_capacity(n);
    for j in 0..n {
        let w_over_2pi = Float::with_val(prec, &weights[j] * &inv_two_pi);
        let bv = Complex::with_val(prec, &b_f_ln[j] * &v[j]);
        bfl_v.push(Complex::with_val(prec, &bv * &w_over_2pi));
        let iv = Complex::with_val(prec, &inv_f_ln[j] * &v[j]);
        ifl_v.push(Complex::with_val(prec, &iv * &w_over_2pi));
    }

    let max_offset = (n as i64) - 1;
    let mut out = vec![cnum::zero(prec); n];
    for k in 0..n {
        let mut acc = cnum::zero(prec);
        for j in 0..n {
            // d = j − k, indexed at d + (N − 1).
            let idx = ((j as i64) - (k as i64) + max_offset) as usize;
            let term_r = Complex::with_val(prec, &bfl_v[j] * &inv_denom_r[idx]);
            let term_l = Complex::with_val(prec, &ifl_v[j] * &inv_denom_l[idx]);
            let dt_kj_v = Complex::with_val(prec, &term_r - &term_l);
            acc = Complex::with_val(prec, &acc + &dt_kj_v);
        }
        out[k] = acc;
    }
    out
}

/// L2 norm of a complex vector.
fn vector_norm_complex(v: &[Complex], prec: u32) -> Float {
    let mut sum = Float::new(prec);
    for c in v {
        let abs = Float::with_val(prec, c.abs_ref());
        let sq = Float::with_val(prec, &abs * &abs);
        sum = Float::with_val(prec, &sum + &sq);
    }
    Float::with_val(prec, sum.sqrt_ref())
}

/// Hermitian inner product `<u, v> = Σ ū·v` (conjugate on first argument so
/// `<u, u>` is real and equals `‖u‖²`).
fn inner_product_complex(u: &[Complex], v: &[Complex], prec: u32) -> Complex {
    let mut s = cnum::zero(prec);
    for i in 0..u.len() {
        let conj_u = Complex::with_val(prec, u[i].conj_ref());
        let prod = Complex::with_val(prec, &conj_u * &v[i]);
        s = Complex::with_val(prec, &s + &prod);
    }
    s
}

/// Restarted GMRES with arbitrary-precision complex arithmetic.
///
/// Solves `A x = rhs` where `A` is given implicitly via a matvec closure.
/// Returns `Ok(x)` if the relative residual `‖rhs − A x‖ / ‖rhs‖` falls below
/// `tol_rel`, or `Err` if `max_outer` restarts exhaust without convergence.
///
/// Uses modified Gram-Schmidt Arnoldi and complex Givens rotations of the form
///   G = [[c, s], [−s̄, c]]   with `c ∈ ℝ≥0`, `s ∈ ℂ`, `c² + |s|² = 1`
/// where `c = |a|/τ`, `s = (a/|a|)·b̄/τ`, `τ = √(|a|²+|b|²)`. After each
/// rotation the (k+1, k) entry of the Hessenberg matrix is zeroed and the
/// reduced upper triangular system is solved by back-substitution at restart.
fn gmres_complex<F>(
    matvec: F,
    rhs: &[Complex],
    tol_rel: f64,
    max_outer: usize,
    restart: usize,
    prec: u32,
) -> Result<Vec<Complex>, String>
where
    F: Fn(&[Complex]) -> Vec<Complex>,
{
    let n = rhs.len();
    let zero_c = cnum::zero(prec);
    let one_re = Float::with_val(prec, 1u32);
    let zero_re = Float::new(prec);

    let rhs_norm = vector_norm_complex(rhs, prec);
    let rhs_norm_f64 = rhs_norm.to_f64();
    if rhs_norm_f64 == 0.0 {
        return Ok(vec![zero_c; n]);
    }
    let target_abs = rhs_norm_f64 * tol_rel;

    let mut x = vec![zero_c.clone(); n];
    let restart = restart.max(1);

    for _outer in 0..max_outer {
        let ax = matvec(&x);
        let r: Vec<Complex> = rhs
            .iter()
            .zip(ax.iter())
            .map(|(bi, axi)| Complex::with_val(prec, bi - axi))
            .collect();
        let beta = vector_norm_complex(&r, prec);
        let beta_f64 = beta.to_f64();
        if beta_f64 < target_abs {
            return Ok(x);
        }
        let inv_beta = Float::with_val(prec, &one_re / &beta);
        let v0: Vec<Complex> = r
            .iter()
            .map(|c| Complex::with_val(prec, c * &inv_beta))
            .collect();
        let mut basis: Vec<Vec<Complex>> = Vec::with_capacity(restart + 1);
        basis.push(v0);

        let mut h_mat: Vec<Vec<Complex>> = vec![vec![zero_c.clone(); restart]; restart + 1];
        let mut cs: Vec<Float> = vec![Float::with_val(prec, 1u32); restart];
        let mut sn: Vec<Complex> = vec![zero_c.clone(); restart];
        let mut g: Vec<Complex> = vec![zero_c.clone(); restart + 1];
        g[0] = Complex::with_val(prec, (beta.clone(), zero_re.clone()));

        let mut k_done = 0usize;
        let mut converged_inner = false;

        for k in 0..restart {
            let mut w = matvec(&basis[k]);
            // Modified Gram-Schmidt: orthogonalize w against basis[0..=k].
            for j in 0..=k {
                let h_jk = inner_product_complex(&basis[j], &w, prec);
                for i in 0..n {
                    let term = Complex::with_val(prec, &h_jk * &basis[j][i]);
                    w[i] = Complex::with_val(prec, &w[i] - &term);
                }
                h_mat[j][k] = h_jk;
            }
            let h_kp1_k_re = vector_norm_complex(&w, prec);
            let h_kp1_k_re_f64 = h_kp1_k_re.to_f64();
            h_mat[k + 1][k] =
                Complex::with_val(prec, (h_kp1_k_re.clone(), zero_re.clone()));

            // Apply previously-stored Givens rotations to column k of H.
            for j in 0..k {
                let h1 = h_mat[j][k].clone();
                let h2 = h_mat[j + 1][k].clone();
                let term1 = Complex::with_val(prec, &h1 * &cs[j]);
                let term2 = Complex::with_val(prec, &sn[j] * &h2);
                h_mat[j][k] = Complex::with_val(prec, &term1 + &term2);
                let conj_sn = Complex::with_val(prec, sn[j].conj_ref());
                let term3 = Complex::with_val(prec, &conj_sn * &h1);
                let term4 = Complex::with_val(prec, &h2 * &cs[j]);
                h_mat[j + 1][k] = Complex::with_val(prec, &term4 - &term3);
            }

            // Construct new Givens rotation that zeros h_mat[k+1][k].
            let a = h_mat[k][k].clone();
            let bv = h_mat[k + 1][k].clone();
            let abs_a = Float::with_val(prec, a.abs_ref());
            let abs_b = Float::with_val(prec, bv.abs_ref());
            let abs_a_f64 = abs_a.to_f64();
            let abs_b_f64 = abs_b.to_f64();

            if abs_b_f64 == 0.0 {
                cs[k] = Float::with_val(prec, 1u32);
                sn[k] = zero_c.clone();
            } else if abs_a_f64 == 0.0 {
                cs[k] = Float::new(prec);
                let inv_abs_b = Float::with_val(prec, &one_re / &abs_b);
                let conj_b = Complex::with_val(prec, bv.conj_ref());
                sn[k] = Complex::with_val(prec, &conj_b * &inv_abs_b);
            } else {
                let asq = Float::with_val(prec, &abs_a * &abs_a);
                let bsq = Float::with_val(prec, &abs_b * &abs_b);
                let sum_sq = Float::with_val(prec, &asq + &bsq);
                let norm = Float::with_val(prec, sum_sq.sqrt_ref());
                cs[k] = Float::with_val(prec, &abs_a / &norm);
                let inv_abs_a = Float::with_val(prec, &one_re / &abs_a);
                let alpha = Complex::with_val(prec, &a * &inv_abs_a);
                let conj_b = Complex::with_val(prec, bv.conj_ref());
                let alpha_conj_b = Complex::with_val(prec, &alpha * &conj_b);
                let inv_norm = Float::with_val(prec, &one_re / &norm);
                sn[k] = Complex::with_val(prec, &alpha_conj_b * &inv_norm);
            }

            // Apply the new rotation to column k.
            let term_a = Complex::with_val(prec, &a * &cs[k]);
            let term_b = Complex::with_val(prec, &sn[k] * &bv);
            h_mat[k][k] = Complex::with_val(prec, &term_a + &term_b);
            h_mat[k + 1][k] = zero_c.clone();

            // Apply the new rotation to g (right-hand side after rotations).
            let g_k = g[k].clone();
            let new_g_k = Complex::with_val(prec, &g_k * &cs[k]);
            let conj_sn_k = Complex::with_val(prec, sn[k].conj_ref());
            let neg_term = Complex::with_val(prec, &conj_sn_k * &g_k);
            let new_g_kp1 = Complex::with_val(prec, -&neg_term);
            g[k] = new_g_k;
            g[k + 1] = new_g_kp1;

            k_done = k + 1;

            let resid_est_f64 = Float::with_val(prec, g[k + 1].abs_ref()).to_f64();
            if resid_est_f64 < target_abs {
                converged_inner = true;
                break;
            }
            if h_kp1_k_re_f64 == 0.0 {
                // Lucky breakdown: Krylov subspace is invariant; current
                // y solves the system exactly within this subspace.
                converged_inner = true;
                break;
            }

            let inv_h = Float::with_val(prec, &one_re / &h_kp1_k_re);
            let v_next: Vec<Complex> = w
                .iter()
                .map(|c| Complex::with_val(prec, c * &inv_h))
                .collect();
            basis.push(v_next);
        }

        // Solve the (k_done × k_done) upper-triangular system H y = g.
        let mut y = vec![zero_c.clone(); k_done];
        for i in (0..k_done).rev() {
            let mut sum = g[i].clone();
            for j in (i + 1)..k_done {
                let prod = Complex::with_val(prec, &h_mat[i][j] * &y[j]);
                sum = Complex::with_val(prec, &sum - &prod);
            }
            // Defensive: pivot can become tiny if A is rank-deficient on the
            // Krylov subspace; in that case the back-sub blows up. Bail on
            // the restart and let the outer loop retry from the new x.
            let pivot_abs = Float::with_val(prec, h_mat[i][i].abs_ref()).to_f64();
            if pivot_abs == 0.0 || !pivot_abs.is_finite() {
                return Err(format!("GMRES: zero pivot at row {} in Hessenberg solve", i));
            }
            y[i] = Complex::with_val(prec, &sum / &h_mat[i][i]);
        }

        // x ← x + V_k · y.
        for j in 0..k_done {
            for i in 0..n {
                let term = Complex::with_val(prec, &basis[j][i] * &y[j]);
                x[i] = Complex::with_val(prec, &x[i] + &term);
            }
        }

        if converged_inner {
            return Ok(x);
        }
    }

    Err(format!(
        "GMRES: failed to converge in {} restarts of size {}",
        max_outer, restart
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Continuation-based solver for near-parabolic bases (Class A)
// ─────────────────────────────────────────────────────────────────────────────

/// Resample a converged Kouznetsov solution onto a new uniform grid.
///
/// Old grid: `old_n` nodes uniformly spaced over `[−old_t_max, +old_t_max]`.
/// New grid: `new_nodes` (may be wider/denser than the old one).
///
/// For new nodes that fall within the old range: linear interpolation between
/// adjacent old samples. For new nodes beyond `±old_t_max`: pin to the
/// asymptotic fixed-point value (`l_upper` / `l_lower`).
fn resample_to_grid(
    old_samples: &[Complex],
    old_t_max_f64: f64,
    new_nodes: &[Float],
    l_upper: &Complex,
    l_lower: &Complex,
    prec: u32,
) -> Vec<Complex> {
    let old_n = old_samples.len();
    if old_n < 2 {
        return new_nodes.iter().map(|_| Complex::with_val(prec, (0u32, 0u32))).collect();
    }
    let old_step = 2.0 * old_t_max_f64 / (old_n - 1) as f64;

    new_nodes
        .iter()
        .map(|t_new| {
            let t = Float::with_val(64, t_new).to_f64();
            if t <= -old_t_max_f64 {
                l_lower.clone()
            } else if t >= old_t_max_f64 {
                l_upper.clone()
            } else {
                // Linear interpolation between neighbouring old samples.
                let frac_idx = (t + old_t_max_f64) / old_step;
                let i0 = frac_idx.floor() as usize;
                let i1 = (i0 + 1).min(old_n - 1);
                let alpha = frac_idx - i0 as f64;
                let re0 = Float::with_val(64, old_samples[i0].real()).to_f64();
                let re1 = Float::with_val(64, old_samples[i1].real()).to_f64();
                let im0 = Float::with_val(64, old_samples[i0].imag()).to_f64();
                let im1 = Float::with_val(64, old_samples[i1].imag()).to_f64();
                Complex::with_val(prec, (
                    Float::with_val(prec, re0 + alpha * (re1 - re0)),
                    Float::with_val(prec, im0 + alpha * (im1 - im0)),
                ))
            }
        })
        .collect()
}

/// Continuation-based Kouznetsov solver for near-parabolic real bases.
///
/// The direct `setup_kouznetsov` fails for `b` near `η = e^(1/e)` because
/// `|arg(λ)| → 0` forces `n_nodes ≫ 32768` and each LM matvec takes ~20s.
/// With a warm start from a nearby-`b` solution the LM converges in 3–5
/// Newton steps (instead of 40+), making even `n_nodes = 65536` tractable.
///
/// **Strategy**: step `b` from `b_start ≈ 1.75` (where `n_nodes ≤ 4096`)
/// toward `b_target` in increments of `≤ 0.025`. At each step, resample the
/// previous samples onto the new (wider/denser) grid as the initial guess.
/// The gradual doubling of `n_nodes` distributes the work across `O(log N)`
/// cheap initial steps.
///
/// Only supports real `b > η`; complex near-parabolic bases are rare and out
/// of scope.
pub fn setup_kouznetsov_continuation(
    b_target: &Complex,
    fp_target: &FixedPointData,
    prec: u32,
    digits: u64,
) -> Result<KouznetsovState, String> {
    let b_re_target = Float::with_val(64, b_target.real()).to_f64();
    let b_im_target = Float::with_val(64, b_target.imag()).to_f64();
    if b_im_target.abs() > 1e-10 {
        return Err("continuation solver only supports real b".into());
    }
    let eta = std::f64::consts::E.powf(1.0 / std::f64::consts::E);
    if b_re_target <= eta {
        return Err(format!("continuation: b={:.5} ≤ η={:.5}", b_re_target, eta));
    }

    // Upfront cap check: compute n_nodes for b_target before doing any work.
    // If target itself exceeds the continuation cap, fail immediately.
    const N_CONTINUATION_CAP: usize = 131_072;
    {
        let arg_lambda_tgt = arg_abs_f64(&fp_target.lambda, 64);
        if arg_lambda_tgt > 1e-6 {
            let t_max_tgt = ((digits as f64 + 8.0) * std::f64::consts::LN_10 / arg_lambda_tgt).max(8.0);
            // Use uncapped n_bulk (not pick_node_count which caps at 60K) to detect when
            // the true required nodes exceed the continuation budget.
            const SIGMA_EFF: f64 = 0.30;
            const MARGIN: f64 = 1.2;
            let n_bulk_tgt = (MARGIN
                * (digits as f64 + 5.0)
                * std::f64::consts::LN_10
                * t_max_tgt
                / (std::f64::consts::PI * SIGMA_EFF))
                .ceil() as usize;
            if n_bulk_tgt > N_CONTINUATION_CAP {
                return Err(format!(
                    "continuation: parabolic boundary — b={:.5} requires ~{} nodes > cap={} \
                     (|arg(λ)|={:.4}); needs Abel/Écalle theory",
                    b_re_target, n_bulk_tgt, N_CONTINUATION_CAP, arg_lambda_tgt
                ));
            }
        }
    }

    // Starting point: far enough from η that the direct solver handles it
    // without triggering the parabolic-boundary cap.
    // b=1.75 has |arg(λ)|≈0.9, giving n_nodes≈4096 at 20 digits.
    let b_start = (b_re_target + 0.35_f64).max(1.75_f64).min(2.5_f64);

    // Step count: ≤ 0.025 per step keeps the warm-start error small enough
    // for quadratic LM convergence in ≤ 5 iterations.
    let n_steps = ((b_start - b_re_target) / 0.025).ceil() as usize + 1;
    let n_steps = n_steps.max(2);

    if cnum::verbose() {
        eprintln!(
            "kouz continuation: b_target={:.5}  b_start={:.5}  n_steps={}",
            b_re_target, b_start, n_steps
        );
    }

    let mut prev_state: Option<KouznetsovState> = None;

    for step in 0..=n_steps {
        let frac = step as f64 / n_steps as f64;
        let b_step = b_start * (1.0 - frac) + b_re_target * frac;

        let b_cplx = Complex::with_val(prec, (Float::with_val(prec, b_step), Float::new(prec)));

        // Compute fixed-point pair for this b value.
        let ln_b = Complex::with_val(prec, b_cplx.ln_ref());
        let neg_ln_b = Complex::with_val(prec, -&ln_b);
        let w0_val = lambertw::w0(&neg_ln_b, prec)
            .map_err(|e| format!("continuation step {}: W₀ failed: {}", step, e))?;
        let l_raw = Complex::with_val(prec, -w0_val / &ln_b);
        // Ensure l_upper has Im > 0 (the convention for Schwarz-symmetric bases).
        let l_upper_step = if Float::with_val(64, l_raw.imag()).to_f64() < 0.0 {
            Complex::with_val(prec, l_raw.conj_ref())
        } else {
            l_raw
        };
        let l_lower_step = Complex::with_val(prec, l_upper_step.conj_ref());
        let lambda_upper = Complex::with_val(prec, &ln_b * &l_upper_step);
        let arg_lambda = arg_abs_f64(&lambda_upper, prec);

        if arg_lambda <= 1e-3 {
            return Err(format!(
                "continuation step {}: |arg(λ)|={:.4} too small at b={:.5}",
                step, arg_lambda, b_step
            ));
        }

        let t_max_f64 = ((digits as f64 + 8.0) * std::f64::consts::LN_10 / arg_lambda).max(8.0);
        let t_max_fp = Float::with_val(prec, t_max_f64);
        let n_nodes = pick_node_count(digits, t_max_f64, arg_lambda);

        if n_nodes > N_CONTINUATION_CAP {
            return Err(format!(
                "continuation step {}: n_nodes={} > {} cap (b={:.5}, |arg(λ)|={:.4})",
                step, n_nodes, N_CONTINUATION_CAP, b_step, arg_lambda
            ));
        }

        let nodes = build_uniform_nodes(&t_max_fp, n_nodes, prec);
        let weights = build_trapezoidal_weights(&t_max_fp, n_nodes, prec);

        if cnum::verbose() {
            eprintln!(
                "kouz cont step {}/{}: b={:.5}  |arg(λ)|={:.4}  t_max={:.1}  n={}  warm={}",
                step, n_steps, b_step, arg_lambda, t_max_f64, n_nodes,
                prev_state.is_some()
            );
        }

        let mut initial = if let Some(ref prev) = prev_state {
            // Resample previous solution onto the new (potentially wider/denser) grid.
            let old_t_max = Float::with_val(64, &prev.t_max).to_f64();
            let mut init = resample_to_grid(
                &prev.samples,
                old_t_max,
                &nodes,
                &l_upper_step,
                &l_lower_step,
                prec,
            );
            // Always re-pin the boundary samples (they may have been approximated
            // as L± during resampling, but the exact new L values differ slightly).
            init[0] = l_lower_step.clone();
            init[n_nodes - 1] = l_upper_step.clone();
            // Real base: symmetrize so the solver stays on the Schwarz manifold.
            symmetrize_schwarz(&mut init, prec);
            init
        } else {
            // Cold start for the first (safe) step. Use the normal capped solver.
            let fp_step = FixedPointData {
                fixed_point: l_upper_step.clone(),
                lambda: lambda_upper.clone(),
                lambda_abs: Float::with_val(64, lambda_upper.abs_ref()).to_f64(),
            };
            let cold = setup_kouznetsov(&b_cplx, &fp_step, prec, digits)?;
            // Resample cold solution onto `nodes` in case step-0 geometry differs.
            let old_t_max = Float::with_val(64, &cold.t_max).to_f64();
            let mut init = resample_to_grid(
                &cold.samples,
                old_t_max,
                &nodes,
                &l_upper_step,
                &l_lower_step,
                prec,
            );
            init[0] = l_lower_step.clone();
            init[n_nodes - 1] = l_upper_step.clone();
            symmetrize_schwarz(&mut init, prec);
            init
        };

        // Run LM with the warm (or cold-resampled) initial.
        let samples = iterate_newton(
            initial,
            &nodes,
            &weights,
            &t_max_fp,
            &l_upper_step,
            &l_lower_step,
            &ln_b,
            prec,
            digits,
            true, // use_schwarz = true for real b
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
        });

        // If this step is already at the target (within float rounding), stop.
        if (b_step - b_re_target).abs() < 1e-10 {
            break;
        }
    }

    // Ignore fp_target (already computed inline above); suppress unused-variable warning.
    let _ = fp_target;

    prev_state.ok_or_else(|| "continuation: no state produced".into())
}
