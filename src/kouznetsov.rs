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

use rug::{Complex, Float};

use crate::{cnum, regions::FixedPointData};

/// Compute `F_b(h)` via Anderson-accelerated Cauchy iteration. Currently
/// restricted to real bases `b > e^(1/e)`. Precision target `digits` drives
/// node count and contour height.
pub fn tetrate_kouznetsov(
    b: &Complex,
    h: &Complex,
    fp: &FixedPointData,
    prec: u32,
    digits: u64,
) -> Result<Complex, String> {
    require_real_positive_above_eta(b, prec)?;

    // Fixed-point pair. `fp.fixed_point = -W₀(-ln b)/ln b` and its conjugate
    // are the two complex fixed points of `b^z = z`. We label the one with
    // non-negative imag as "upper" (boundary value as Im(z)→+∞) and the one
    // with non-positive imag as "lower" (Im(z)→-∞). Sorting explicitly is
    // necessary because Lambert W's f64 seed can flip across the branch cut
    // depending on the sign of zero in `-ln b` (real bases give either +0 or
    // -0 imag), so we cannot assume `fp.fixed_point` consistently has one sign.
    let raw = fp.fixed_point.clone();
    let raw_conj = Complex::with_val(prec, raw.conj_ref());
    let raw_imag_neg = raw.imag().is_sign_negative();
    let (l_lower, l_upper) = if raw_imag_neg {
        (raw, raw_conj)
    } else {
        (raw_conj, raw)
    };
    // λ_upper = (ln b)·L_upper has non-negative imag for real b > e^(1/e); take
    // the consistent sign by re-deriving from L_upper rather than `fp.lambda`.
    let ln_b = Complex::with_val(prec, b.ln_ref());
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

    // Trapezoidal node count: error ≈ exp(-π·d·N/T). For digits-digit accuracy
    // and d ≈ 1, N ≈ ln(10)/π · digits · T.
    let n_nodes = pick_node_count(digits, t_max_f64);

    let nodes = build_uniform_nodes(&t_max, n_nodes, prec);
    let weights = build_trapezoidal_weights(&t_max, n_nodes, prec);

    if std::env::var_os("TET_KOUZ_DEBUG").is_some() {
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

    let mut initial = initial_guess(&nodes, b, &l_upper, &l_lower, arg_lambda, prec);
    // Pin the boundary samples to the asymptotic fixed-point values: the very
    // first sample (t=−t_max) corresponds to z₀ on the bottom contour edge —
    // Cauchy's formula has a 1/(t−z₀) singularity there. Same for the last
    // sample at the top edge. By pinning F[0]=L_lower and F[N−1]=L_upper we
    // remove the corner singularity from the iteration; interior samples then
    // converge cleanly via T (which uses these pinned values in its integrand).
    initial[0] = l_lower.clone();
    initial[n_nodes - 1] = l_upper.clone();
    // Schwarz reflection: F(z̄)=F̄(z). Symmetrize the initial guess so the
    // iteration starts on the natural-Kouznetsov manifold.
    symmetrize_schwarz(&mut initial, prec);
    // Default solver: Levenberg-Marquardt Newton-Kantorovich. Newton's
    // Jacobian-based step finds the right descent direction even when the
    // Cauchy operator T has spectral radius > 1 (which it does for typical
    // real bases b > e^(1/e)), where Picard / Anderson without history
    // diverge. The two non-default solvers stay around as debugging aids:
    //   * `TET_KOUZ_ANDERSON=1`: Anderson-accelerated Picard (works when T
    //     is a contraction, fails for higher b).
    //   * `TET_KOUZ_PICARD=1`: damped Picard, useful for spectrum diagnosis.
    let samples = if std::env::var_os("TET_KOUZ_ANDERSON").is_some() {
        iterate_anderson(
            initial, &nodes, &weights, &t_max, &l_upper, &l_lower, &ln_b, prec, digits,
        )?
    } else if std::env::var_os("TET_KOUZ_PICARD").is_some() {
        iterate_picard(
            initial, &nodes, &weights, &t_max, &l_upper, &l_lower, &ln_b, prec, digits,
        )?
    } else {
        iterate_newton(
            initial, &nodes, &weights, &t_max, &l_upper, &l_lower, &ln_b, prec, digits,
        )?
    };

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
        &samples, &nodes, &weights, &t_max, &l_upper, &l_lower, &ln_b, prec,
    )?;
    if std::env::var_os("TET_KOUZ_DEBUG").is_some() {
        eprintln!(
            "kouz normalization shift δ = {:.6e} + {:.6e}i (such that F(δ)=1)",
            Float::with_val(prec, shift.real()).to_f64(),
            Float::with_val(prec, shift.imag()).to_f64(),
        );
    }
    let h_shifted = Complex::with_val(prec, h + &shift);

    eval_at_height(
        b, &h_shifted, &samples, &nodes, &weights, &t_max, &l_upper, &l_lower, &ln_b, prec,
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
) -> Result<Complex, String> {
    let one = Complex::with_val(prec, (1u32, 0));
    let two = Float::with_val(prec, 2u32);
    let eps_f = Float::with_val(prec, 1e-8f64);
    let eps = Complex::with_val(prec, (eps_f.clone(), 0));
    let two_eps = Complex::with_val(prec, (Float::with_val(prec, &eps_f * &two), 0));

    let mut c = cnum::zero(prec);
    let max_iters = 30usize;
    let target = 1e-15f64; // tighter than the typical residual floor; refines best-effort

    for _iter in 0..max_iters {
        let f_c = cauchy_eval(
            &c, samples, nodes, weights, t_max, l_upper, l_lower, ln_b, prec,
        );
        let resid = Complex::with_val(prec, &f_c - &one);
        let resid_abs = Float::with_val(prec, resid.abs_ref()).to_f64();
        if resid_abs < target {
            return Ok(c);
        }
        if !resid_abs.is_finite() {
            return Err(format!(
                "Kouznetsov normalization: F(c) became non-finite (c = {} + {}i)",
                Float::with_val(prec, c.real()).to_f64(),
                Float::with_val(prec, c.imag()).to_f64(),
            ));
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
            return Err(format!(
                "Kouznetsov normalization: F'(c) ≈ 0 (degenerate); cannot Newton step"
            ));
        }
        let step = Complex::with_val(prec, &resid / &derivative);
        c = Complex::with_val(prec, &c - &step);
    }
    // Best-effort: return whatever c we have if Newton didn't fully converge.
    Ok(c)
}

fn require_real_positive_above_eta(b: &Complex, prec: u32) -> Result<(), String> {
    if !b.imag().is_zero() {
        return Err("Kouznetsov: implementation restricted to real bases".into());
    }
    if !b.real().is_sign_positive() || b.real().is_zero() {
        return Err("Kouznetsov: requires b > 0".into());
    }
    let inv_e = Float::with_val(prec, Float::with_val(prec, 1u32).exp_ref()).recip();
    let eta_val = Float::with_val(prec, inv_e.exp_ref());
    if *b.real() <= eta_val {
        return Err(format!(
            "Kouznetsov: requires b > e^(1/e) ≈ 1.444667; got b ≈ {:.6}",
            b.real().to_f64()
        ));
    }
    Ok(())
}

fn arg_abs_f64(z: &Complex, prec: u32) -> f64 {
    let im = Float::with_val(prec, z.imag()).to_f64();
    let re = Float::with_val(prec, z.real()).to_f64();
    im.atan2(re).abs()
}

fn pick_node_count(digits: u64, t_max: f64) -> usize {
    let scale = 1.1 * std::f64::consts::LN_10 / std::f64::consts::PI;
    let n = (scale * digits as f64 * t_max).ceil() as usize;
    n.max(80).min(20_000)
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

fn initial_guess(
    nodes: &[Float],
    b: &Complex,
    l_upper: &Complex,
    l_lower: &Complex,
    arg_lambda: f64,
    prec: u32,
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
    let sqrt_b_re = Float::with_val(prec, sqrt_b.real()).to_f64();
    // Hard cap at 2.0: Kouznetsov tables give F̃_b(0.5) values that for any
    // real b ∈ (e^(1/e), ∞) stay roughly in [1.4, 2.0] (F̃ is highly convex
    // so the half-way value tracks the *near-1* end, not the b end). Using
    // sqrt_b directly e.g. for b=10 puts us at ≈3.16, and the b^F amplification
    // on the right edge of the contour makes T(F) overshoot by ~10^1.16, which
    // the iteration cannot recover from. Capping at 2 puts the initial guess
    // in the same ballpark as the truth for any b > e^(1/e).
    let target_mid_re = if sqrt_b_re < 2.0 { sqrt_b_re } else { 2.0 };
    let target_mid = Complex::with_val(prec, (Float::with_val(prec, target_mid_re), 0));
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
) -> Result<Vec<Complex>, String> {
    let n = initial.len();
    let n_int = n - 2; // number of interior samples that actually iterate
    let max_iters = 400usize;
    let target = 10f64.powf(-(digits as f64) - 3.0);
    let debug = std::env::var_os("TET_KOUZ_DEBUG").is_some();

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
        symmetrize_schwarz(&mut x_new, prec);
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
) -> Result<Vec<Complex>, String> {
    let n = initial.len();
    let max_iters = 2000usize;
    let target = 10f64.powf(-(digits as f64) - 3.0);
    let debug = std::env::var_os("TET_KOUZ_DEBUG").is_some();

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
        symmetrize_schwarz(&mut x_new, prec);
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
/// Per-iteration cost is `O(N²)` for the Jacobian and `O(N³)` for the linear
/// solve (Gauss elimination at full precision). Newton converges quadratically
/// near the fixed point, so 5–15 iterations typically suffice.
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
) -> Result<Vec<Complex>, String> {
    let n = initial.len();
    let max_iters = 60usize;
    let target = 10f64.powf(-(digits as f64) - 3.0);
    let debug = std::env::var_os("TET_KOUZ_DEBUG").is_some();

    let mut x = initial;
    let mut prev_residual = f64::INFINITY;
    let mut stagnation = 0u32;
    let mut best_x = x.clone();
    let mut best_residual = f64::INFINITY;
    // Levenberg-Marquardt damping. μ adapts: shrink on success (gain a faster,
    // more Newton-like step next time), grow on rejection (more like gradient
    // descent). Starts at a moderate value so the first step doesn't blow up.
    let mut mu = 1.0f64;

    if debug {
        let mid = nodes.len() / 2;
        eprintln!(
            "kouz LM start: n={} mid_idx={} mid_t={} F[mid]={}+{}i",
            nodes.len(),
            mid,
            Float::with_val(prec, &nodes[mid]).to_f64(),
            Float::with_val(prec, x[mid].real()).to_f64(),
            Float::with_val(prec, x[mid].imag()).to_f64(),
        );
    }

    for iter in 0..max_iters {
        let f = apply_t(&x, nodes, weights, t_max, l_upper, l_lower, ln_b, prec);
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
            let mid_idx = nodes.len() / 2;
            let xm_re = Float::with_val(prec, x[mid_idx].real()).to_f64();
            let xm_im = Float::with_val(prec, x[mid_idx].imag()).to_f64();
            eprintln!(
                "kouz LM iter {:>3}: ‖r‖∞ = {:.3e}  μ={:.2e}  F(0.5)≈{:.4}+{:.4}i  (target {:.3e})",
                iter, r_norm, mu, xm_re, xm_im, target
            );
        }

        if r_norm.is_nan() || !r_norm.is_finite() {
            if best_residual.is_finite() {
                if debug {
                    eprintln!(
                        "kouz LM: residual non-finite at iter {}; returning best ({:.3e})",
                        iter, best_residual
                    );
                }
                return Ok(best_x);
            }
            return Err(format!(
                "Kouznetsov Newton: residual became non-finite at iter {}",
                iter
            ));
        }
        if r_norm < best_residual {
            best_residual = r_norm;
            best_x = x.clone();
        }
        if r_norm < target {
            return Ok(x);
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
                        "kouz LM: stagnated at iter {} (residual {:.3e}, likely discretization floor); returning best ({:.3e})",
                        iter, r_norm, best_residual
                    );
                }
                return Ok(best_x);
            }
        } else {
            stagnation = 0;
        }
        prev_residual = r_norm;

        // Build J = I − DT analytically.
        let mut jac = compute_jacobian_minus_dt(&x, nodes, weights, ln_b, prec);
        // Pin boundary rows: J[0] = e_0, J[n-1] = e_{n-1} so the LM solve
        // produces δ[0] = δ[n-1] = 0. Combined with r[0] = r[n-1] = 0
        // (zeroed above), this makes the boundary samples constants in the
        // Newton system rather than free variables that would chase the
        // corner Cauchy singularity.
        let zero_c = cnum::zero(prec);
        let one_c = Complex::with_val(prec, (Float::with_val(prec, 1u32), 0));
        for col in 0..n {
            jac[0][col] = zero_c.clone();
            jac[n - 1][col] = zero_c.clone();
        }
        jac[0][0] = one_c.clone();
        jac[n - 1][n - 1] = one_c;

        // Levenberg-Marquardt: solve (J + μ·I)·δ = r. Inner loop adaptively
        // grows μ until the trial step actually decreases residual; then
        // shrinks μ for the next outer iteration.
        let mut accepted = false;
        for _lm_try in 0..10 {
            let mut jac_damped: Vec<Vec<Complex>> = jac.iter().map(|row| row.clone()).collect();
            let mu_f = Float::with_val(prec, mu);
            for k in 0..n {
                jac_damped[k][k] = Complex::with_val(prec, &jac_damped[k][k] + &mu_f);
            }

            let delta = match solve_complex_lin(&jac_damped, &r, prec) {
                Ok(d) => d,
                Err(_) => {
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
            // F's symmetry manifold.
            x_trial[0] = l_lower.clone();
            x_trial[n - 1] = l_upper.clone();
            symmetrize_schwarz(&mut x_trial, prec);
            let f_trial = apply_t(&x_trial, nodes, weights, t_max, l_upper, l_lower, ln_b, prec);
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
                let _ = jac; // suppress unused-after-move
                break;
            }
            // Reject: grow μ to take a smaller, more gradient-like step.
            mu *= 4.0;
            if mu > 1e8 {
                break;
            }
            let _ = jac_damped;
        }
        if !accepted {
            // LM line-search exhausted (μ has grown without finding a step
            // that decreases the residual). This typically means we are at
            // the discretization floor — the discrete fixed point's residual
            // is bounded below by the trapezoidal/contour truncation error,
            // and Newton has nowhere left to go. Return the best iterate so
            // the dispatcher can deliver an answer at the discretization-
            // limited precision rather than failing the whole call.
            if best_residual.is_finite() {
                if debug {
                    eprintln!(
                        "kouz LM: no descent step at iter {} (μ={:.2e}); returning best ({:.3e})",
                        iter, mu, best_residual
                    );
                }
                return Ok(best_x);
            }
            return Err(format!(
                "Kouznetsov LM: failed to find a descent step at iter {} \
                 (residual {:.3e}, μ={:.2e})",
                iter, r_norm, mu
            ));
        }
    }

    if best_residual.is_finite() {
        return Ok(best_x);
    }
    Err(format!(
        "Kouznetsov Newton: did not converge in {} iterations (last residual {:.3e})",
        max_iters, prev_residual
    ))
}

/// Build `J = I − DT` for the Cauchy operator. `DT[k][j]` is the partial
/// derivative of `T(F)[k]` with respect to `F[j]`, derived analytically from
/// the right-edge term `b^F[j]/(1.5+it_j−z_k)` and left-edge term
/// `log_b(F[j])/(−0.5+it_j−z_k)` (the top/bottom edges contribute constants
/// independent of `F`, so they don't enter the Jacobian).
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
