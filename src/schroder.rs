//! Schröder regular tetration for Shell-Thron interior bases.
//!
//! For `f(z) = b^z` with attracting fixed point `L = -W₀(-ln b) / ln b` and
//! multiplier `λ = L · ln b` (`|λ| < 1`), the Schröder function `σ` solving
//! `σ(L) = 0`, `σ'(L) = 1`, `σ(f(z)) = λ σ(z)` linearises the dynamics. From it,
//! `F_b(z) = L + σ̃⁻¹(σ̃(1 − L) · λ^z)` where `σ̃(w) = σ(L + w)`. This satisfies
//! `F_b(0) = 1` exactly and `F_b(z+1) = b^{F_b(z)}` analytically.
//!
//! Coefficients of `σ̃` are computed from the functional equation
//! `σ̃(λ h(w)) = λ σ̃(w)` with `h(w) = (b^{L+w} − L)/λ`. Writing
//! `h(w) = w·q(w)` factors out the leading order; the resulting recursion is
//! `c_N (λ^N − λ) = − Σ c_n λ^n [w^{N−n}] q(w)^n` (sum from `n = 1` to `N − 1`),
//! with `q[j] = (ln b)^j / (j+1)!`. The cancellation `λ = L·ln b` in
//! `h_k = a_k/λ = (ln b)^{k−1}/k!` keeps the recursion clean even at very high
//! precision. Series reversion produces `σ̃⁻¹`; evaluation uses Horner.
//!
//! When `|t| = |σ̃(1−L)·λ^z|` falls outside the safe radius for σ̃⁻¹, we shift
//! `h` by an integer `k`: compute `F(z+k)` (smaller `|t|`) then iterate `b^·`
//! or `log_b` back. Sign of `k` flips with the sign of `log|λ|`.
//!
//! Two complementary shift mechanisms make this work for both attracting
//! (|λ|<1) and repelling (|λ|>1) fixed points:
//!   * σ̃-shift on the input `w₀ = 1−L`: when the σ̃ Taylor at 0 doesn't reach
//!     `w₀`, iterate the dynamics to a `w` that is in the Taylor disk and
//!     compensate via `σ̃(φ(w)) = λ·σ̃(w)`. For attracting we forward-iterate
//!     `φ`; for repelling we backward-iterate `φ⁻¹(w) = log_b(L+w) − L`.
//!   * h-shift on the output: shift `h` by integer `k` to bring `t = σ̃(w₀)·λ^h`
//!     inside the σ̃⁻¹ Taylor disk, then iterate `b^·` (`k<0`) or `log_b`
//!     (`k>0`) back.

use rug::{ops::Pow, Complex, Float, Integer};

use crate::{cnum, regions::FixedPointData};
use cnum::{DisplayComplex, DisplayFloat};

/// Cached per-base Schröder state. Built once via `setup_schroder` and reused
/// for many heights via `eval_schroder`. Amortises the O(N²) σ̃ Taylor build
/// across all heights — for grid sweeps over many `(b, h)` cells with the same
/// base, this is a 10-100× speedup at digits ≥ 20.
#[derive(Clone)]
pub struct SchroderState {
    /// Attracting (|λ|<1) or repelling (|λ|>1) fixed point of `b^z = z`.
    pub l: Complex,
    /// `ln(λ)` precomputed for `λ^h = exp(h · ln λ)` per cell.
    pub ln_lambda: Complex,
    /// `ln(b)` precomputed for the integer-shift `b^·` / `log_b` chain.
    pub ln_b: Complex,
    /// `|λ|` drives shift direction.
    pub lam_abs: Float,
    /// σ̃⁻¹ Taylor coefficients (series reversion of σ̃). `sigma_inv[i]` is the
    /// `i`-th coefficient; `sigma_inv[0] = 0`.
    pub sigma_inv: Vec<Complex>,
    /// `σ̃(1 − L)` — entry point for the formula `F(z) = L + σ̃⁻¹(s1·λ^z)`.
    pub s1: Complex,
    /// `safe_radius = 0.5 · max(|1−L|, 0.5)`. Heuristic outer bound for σ̃⁻¹ convergence.
    pub safe_radius: Float,
    /// Actual inner radius at which σ̃ was evaluated during setup (after φ-shifts).
    /// This guides shifts; it is not a convergence proof for the inverse series.
    pub sigma_inner_radius: Float,
    /// MPC bit precision the state was built at — must match the precision
    /// used for per-cell evaluation.
    pub prec: u64,
}

/// Compute `F_b(h)` via Schröder expansion at the upper fixed point.
///
/// Works for both `|λ| < 1` (Shell-Thron interior, attracting fixed point) and
/// `|λ| > 1` (outside Shell-Thron, repelling fixed point). The recursion
/// `c_N (λ^N − λ) = …` is well-defined whenever λ is not a root of unity, so
/// the only failure modes are the parabolic boundary `|λ| = 1` (resonance) and
/// arguments where the series doesn't converge.
///
/// The shift mechanism direction depends on `|λ|`:
///   * `|λ| < 1`: shift right, `F(z) = ln_b(F(z+k))` after `k > 0` steps.
///   * `|λ| > 1`: shift left, `F(z) = b^^k(F(z−k))` after `k > 0` steps.
///
/// Note: for `|λ| > 1` and real bases, this construction provides a valid
/// holomorphic `F` satisfying `F(z+1) = b^{F(z)}` but is not necessarily real
/// on the real axis (the Kneser/Kouznetsov "natural" tetration adds a
/// Riemann-map correction on top — future work).
pub fn tetrate_schroder(
    b: &Complex,
    h: &Complex,
    fp_data: &FixedPointData,
    prec: u64,
) -> Result<Complex, String> {
    // setup_schroder now includes the anchor check (F(0)=1), so any degenerate
    // state is rejected before we reach eval.
    let state = setup_schroder(b, fp_data, prec)?;
    eval_schroder(&state, h)
}

/// Post-validate `F(h)` by checking the functional equation
/// `F(h+1) = b^F(h)` numerically. The σ̃ Taylor series can converge inside
/// the heuristic `safe_radius` yet still evaluate to a wrong value when the
/// actual radius of convergence of `σ⁻¹` is smaller (this happens for real
/// bases just below η, where `|λ| → 1`). Failure rejects an inconsistent
/// reconstruction; passing is not an independent accuracy certificate.
fn validate_functional_equation(
    state: &SchroderState,
    h: &Complex,
    f_h: &Complex,
    prec: u64,
) -> Result<(), String> {
    let one = cnum::one(prec);
    let h_plus_one = Complex::with_val_64(prec, h + &one);
    let f_h_plus_one = eval_schroder_raw(state, &h_plus_one)?;

    // b^F(h) = exp(F(h) · ln b)
    let exponent = Complex::with_val_64(prec, f_h * &state.ln_b);
    let b_pow_f_h = cnum::checked_exp(&exponent, prec)?;

    let diff = Complex::with_val_64(prec, &f_h_plus_one - &b_pow_f_h);
    let diff_abs = cnum::abs(&diff, prec);
    let f_abs = cnum::abs(&f_h_plus_one, prec).max(&Float::with_val_64(prec, 1));
    let rel = Float::with_val_64(prec, &diff_abs / &f_abs);

    let tol = cnum::working_epsilon(prec);
    if !rel.is_finite() || rel > tol {
        if cnum::verbose() {
            eprintln!(
                "schröder validation FAILED: |F(h+1) − b^F(h)| = {:.3e}, |F(h+1)| ≈ {:.3e}, rel = {:.3e}",
                DisplayFloat(&diff_abs), DisplayFloat(&f_abs), DisplayFloat(&rel)
            );
        }
        return Err(format!(
            "Schröder series evaluation failed validation: \
             |F(h+1) − b^F(h)| / max(|F(h+1)|, 1) = {:.3e} exceeds {:.0e} \
             — Taylor series likely outside true radius of convergence (|λ|={:.4})",
            DisplayFloat(&rel),
            DisplayFloat(&tol),
            DisplayFloat(&state.lam_abs)
        ));
    }
    if cnum::verbose() {
        eprintln!(
            "schröder validation OK: |F(h+1) − b^F(h)| / max(|F(h+1)|, 1) = {:.3e}",
            DisplayFloat(&rel)
        );
    }
    Ok(())
}

/// Build the Schröder state for base `b`. All the heavy lifting (σ̃ Taylor
/// coefficient recursion, series reversion, σ̃-shift to evaluate at `1 − L`)
/// lives here. Per-cell evaluation via `eval_schroder` is then O(N) plus the
/// integer-shift chain.
pub fn setup_schroder(
    b: &Complex,
    fp_data: &FixedPointData,
    prec: u64,
) -> Result<SchroderState, String> {
    crate::mt::init_pool()?;
    cnum::check_precision(prec)?;
    if !cnum::is_finite(b)
        || !cnum::is_finite(&fp_data.fixed_point)
        || !cnum::is_finite(&fp_data.lambda)
    {
        return Err("Schröder setup requires finite inputs".into());
    }
    let l = &fp_data.fixed_point;
    let lambda = &fp_data.lambda;
    let lam_abs = fp_data.lambda_abs.clone();
    if !(lam_abs > 0 && lam_abs.is_finite()) {
        return Err(format!(
            "Schröder: invalid |λ| = {}",
            DisplayFloat(&lam_abs)
        ));
    }
    // The build recursion divides by `λ^N − λ`, which vanishes only at
    // λ ∈ root-of-unity. For |λ|=1 exactly (true parabolic), we'd hit a
    // zero denominator at N=1 (since λ^1 − λ = 0 trivially). Block only
    // a very tight band around |λ|=1; let σ̃-shift + extended N handle
    // the boundary band (|λ|=0.95–0.99).
    if Float::with_val_64(prec, &lam_abs - 1).abs() < cnum::decimal("0.005", prec) {
        return Err(format!(
            "Schröder unreliable on parabolic boundary (|λ| = {})",
            DisplayFloat(&lam_abs)
        ));
    }

    let one = cnum::one(prec);
    let w0 = Complex::with_val_64(prec, &one - l);
    let w0_abs = cnum::abs(&w0, prec);
    if !(w0_abs.is_finite() && w0_abs > 0) {
        return Err(format!("Schröder: bad |1−L| = {}", DisplayFloat(&w0_abs)));
    }

    // Cheap pre-check at modest precision: if the σ̃ Taylor terms at w₀ are
    // clearly diverging AND the σ̃-shift mechanism can't rescue (orbit hits a
    // singularity, or we're in the repelling case where φ⁻¹ at w₀=1−L lands on
    // L+w=0), bail before paying for an O(N²) build at user precision. The
    // shift mechanism is given a chance through `eval_sigma_with_shift`; this
    // probe only filters obviously-hopeless cases.
    radius_probe(b, l, lambda, prec, &w0, &w0_abs)?;

    let n_terms = pick_n_terms(&lam_abs, prec)?;
    let (sigma, sigma_inv) = build_series(b, lambda, prec, n_terms)?;

    // σ̃(1 − L) is needed for the F(z) = L + σ̃⁻¹(σ̃(1−L)·λ^z) formula. If the
    // Taylor at 0 doesn't reach `w₀ = 1 − L` directly (radius too small), use
    // the functional equation σ̃(φ(w)) = λ·σ̃(w) with φ(w) = b^(L+w) − L.
    //   * |λ|<1 (attracting): forward-iterate φ; it contracts toward 0.
    //   * |λ|>1 (repelling): backward-iterate φ⁻¹(w) = log_b(L+w) − L; 0 is
    //     attracting for φ⁻¹ (multiplier 1/λ, |1/λ|<1), so a starting point in
    //     its basin contracts to 0.
    //
    // A principal backward orbit can hit 1 → 0 → log_b(0). Refuse that
    // singular shift chain; a root on another logarithm branch is not a
    // justified replacement. The dispatcher may try its other constructions.
    let (s1, sigma_inner_radius) = eval_sigma_with_shift(&sigma, b, l, lambda, prec, &w0)?;

    let ln_lambda = Complex::with_val_64(prec, lambda.ln_ref());
    let ln_b = Complex::with_val_64(prec, b.ln_ref());
    let safe_radius = w0_abs.clone().max(&cnum::decimal("0.5", prec)) / 2;

    if cnum::verbose() {
        let s1_abs = cnum::abs(&s1, prec);
        eprintln!(
            "schröder setup: |λ|={:.6} |1−L|={:.6} |s1|={:.6} safe={:.6} inner_r={:.6} N={}",
            DisplayFloat(&lam_abs),
            DisplayFloat(&w0_abs),
            DisplayFloat(&s1_abs),
            DisplayFloat(&safe_radius),
            DisplayFloat(&sigma_inner_radius),
            n_terms
        );
    }

    let base_state = SchroderState {
        l: l.clone(),
        ln_lambda,
        ln_b,
        lam_abs,
        sigma_inv,
        s1,
        safe_radius,
        sigma_inner_radius,
        prec,
    };

    // Anchor check: F(0)=1 must hold. If σ̃(1−L) is degenerate (e.g. near
    // the parabolic boundary where σ̃⁻¹ diverges at |s1|), eval_schroder
    // silently returns F≡L for all h. The state now carries sigma_inner_radius
    // so eval_schroder knows to keep |t| well within the convergence disk.
    let zero = cnum::zero(prec);
    let one_c = cnum::one(prec);
    let f_zero = eval_schroder_raw(&base_state, &zero)?;
    let anchor_diff = Complex::with_val_64(prec, &f_zero - &one_c);
    let anchor_err = cnum::abs(&anchor_diff, prec);
    if !anchor_err.is_finite() || anchor_err > cnum::working_epsilon(prec) {
        return Err(format!(
            "Schröder anchor check failed: F(0) = {} (expected 1.0); \
             σ̃-shift likely produced degenerate fixed-point solution F≡L",
            DisplayComplex(&f_zero)
        ));
    }

    Ok(base_state)
}

/// Evaluate `F_b(h)` from a cached `SchroderState`. The expensive σ̃ build is
/// already amortised; this call costs one `λ^h` exponential, one O(N) Horner
/// pass, and (rarely) a few `b^·`/`log_b` integer-shift iterations.
///
/// Uses `sigma_inner_radius` as the target for the h-shift, which was
/// determined during setup as the radius at which σ̃ actually converged.
/// This ensures σ̃⁻¹ is evaluated well inside its convergence disk.
pub fn eval_schroder(state: &SchroderState, h: &Complex) -> Result<Complex, String> {
    cnum::init_mpfr();
    if !cnum::is_finite(h) {
        return Err("Schröder height must be finite".into());
    }
    // Roundoff in F(-1) must not turn log(0) into a finite surrogate.
    if h.imag().is_zero() && h.real().is_integer() && *h.real() <= -2 {
        return Err(format!(
            "integer height {} is undefined for tetration (would require log_b(0) and beyond)",
            DisplayFloat(h.real())
        ));
    }
    let value = eval_schroder_raw(state, h)?;
    validate_functional_equation(state, h, &value, state.prec)?;
    Ok(value)
}

fn eval_schroder_raw(state: &SchroderState, h: &Complex) -> Result<Complex, String> {
    let prec = state.prec;
    let lam_h = lambda_pow(h, &state.ln_lambda, prec)?;
    let t = Complex::with_val_64(prec, &state.s1 * &lam_h);
    let t_abs = cnum::abs(&t, prec);

    if !t_abs.is_finite() || t_abs <= 0 {
        return Err(format!("Schröder: bad |t| = {}", DisplayFloat(&t_abs)));
    }

    // The σ̃⁻¹ convergence radius can be smaller than sigma_inner_radius (which
    // tracks σ̃ in the w-domain, not σ̃⁻¹ in the t-domain). The strategy:
    //   1. Try direct eval_series_checked at |t|. If succeeds, return.
    //   2. Otherwise, find the smallest k such that eval_series_checked succeeds
    //      at |t · λ^k|. Each iteration halves the target.
    //
    // This handles bases near η where σ̃⁻¹ has a small effective radius even
    // when sigma_inner_radius reports otherwise.
    if let Ok(inv_t) = eval_series_checked(&state.sigma_inv, &t, prec) {
        return Ok(Complex::with_val_64(prec, &state.l + &inv_t));
    }

    let log_lam_abs = state.lam_abs.clone().ln();
    let initial_target = state
        .sigma_inner_radius
        .clone()
        .min(&state.safe_radius)
        .min(&t_abs);

    let mut effective_target = initial_target / 2;
    let (k, mut f) = loop {
        let ratio = Float::with_val_64(prec, &effective_target / &t_abs);
        if !ratio.is_finite() || ratio <= 0 {
            return Err("Schröder height-shift target exceeded MPFR's exponent range".into());
        }
        let k_raw = ratio.ln() / &log_lam_abs;
        let mut k = if log_lam_abs < 0 {
            k_raw.ceil()
        } else {
            k_raw.floor()
        };
        if !k.is_finite() {
            return Err("Schröder required height shift exceeded MPFR's exponent range".into());
        }
        if k == 0 {
            // Need at least one shift to move strictly inside effective_target
            k = Float::with_val_64(prec, if log_lam_abs < 0 { 1 } else { -1 });
        }

        let h_shifted = Complex::with_val_64(prec, h + &k);
        let lam_h_shifted = lambda_pow(&h_shifted, &state.ln_lambda, prec)?;
        let t_shifted = Complex::with_val_64(prec, &state.s1 * &lam_h_shifted);
        if !cnum::is_finite(&t_shifted) || cnum::is_zero(&t_shifted) {
            return Err("Schröder height shift lost its nonzero inverse-series coordinate".into());
        }
        match eval_series_checked(&state.sigma_inv, &t_shifted, prec) {
            Ok(inv_t_shifted) => {
                let f0 = Complex::with_val_64(prec, &state.l + &inv_t_shifted);
                if cnum::verbose() {
                    let t_sh = cnum::abs(&t_shifted, prec);
                    eprintln!(
                        "schröder eval: k={} |t_shifted|={:.6e} (target={:.3e})",
                        DisplayFloat(&k),
                        DisplayFloat(&t_sh),
                        DisplayFloat(&effective_target)
                    );
                }
                break (k, f0);
            }
            Err(_) => {
                effective_target /= 2;
            }
        }
    };

    // The unwinding chains below must never pass through an exact 0 / ∞ /
    // NaN. For very large |ln b| (e.g. b = 10⁶) the `b^·` chain can underflow
    // to an exact ±0 in one step (Re(f·ln b) below the exponent range), after
    // which the orbit degenerates to the …, 0, 1, b, … alternation. Such a
    // chain SELF-VALIDATES the functional equation (both F(h) and F(h+1) come
    // from the same corrupted alternation), so the guard must live here, not
    // in the post-check. Similarly, the log chain dies if it hits 0 exactly.
    let check_chain = |f: &Complex, step: &Integer| -> Result<(), String> {
        let fa = cnum::abs(f, prec);
        if !cnum::is_finite(f) || fa.is_zero() {
            return Err(format!(
                "Schröder: integer-shift chain degenerated at step {} \
                 (|F| = {}); result would be underflow/overflow garbage",
                step,
                DisplayFloat(&fa)
            ));
        }
        Ok(())
    };
    let forward_log = k > 0;
    let shifts = k.abs();
    let mut step = Integer::new();
    while shifts > step {
        if cnum::verbose() && (step == 0 || step.is_divisible_u(1024)) {
            eprintln!(
                "schröder unwind: step {step} of {:.8}",
                DisplayFloat(&shifts)
            );
        }
        if forward_log {
            check_chain(&f, &step)?;
            let ln_f = Complex::with_val_64(prec, f.ln_ref());
            f = Complex::with_val_64(prec, &ln_f / &state.ln_b);
        } else {
            let exponent = Complex::with_val_64(prec, &f * &state.ln_b);
            f = cnum::checked_exp(&exponent, prec)?;
            check_chain(&f, &step)?;
        }
        step += 1;
    }
    check_chain(&f, &step)?;
    Ok(f)
}

/// `λ^h = exp(h · ln λ)`. The principal branch of `ln λ` is fine inside the
/// Shell-Thron interior because `λ` is never zero there (`λ = 0` only when
/// `ln b = 0`, i.e., `b = 1`, which is filtered out earlier).
fn lambda_pow(h: &Complex, ln_lambda: &Complex, prec: u64) -> Result<Complex, String> {
    let exponent = Complex::with_val_64(prec, h * ln_lambda);
    cnum::checked_exp(&exponent, prec)
}

/// Cheap pre-check: build ~80 σ̃ coefficients and do a
/// trial run of the σ̃-shift mechanism. If neither direct evaluation nor the
/// shift can produce a finite σ̃(w₀), bail before paying for the full O(N²)
/// build at user precision. This catches the b=e/b=2/b=10/b=−2 cases where
/// the φ⁻¹ orbit immediately hits L+w=0 and the log branch cut.
fn radius_probe(
    b: &Complex,
    l: &Complex,
    lambda: &Complex,
    user_prec: u64,
    w0: &Complex,
    w_abs: &Float,
) -> Result<(), String> {
    if !(w_abs.is_finite() && *w_abs > 0) {
        return Err(format!("radius probe: bad |w| = {}", DisplayFloat(w_abs)));
    }
    let probe_prec = user_prec;
    let probe_n = 80usize;
    let b_p = Complex::with_val_64(probe_prec, b);
    let l_p = Complex::with_val_64(probe_prec, l);
    let lambda_p = Complex::with_val_64(probe_prec, lambda);
    let w0_p = Complex::with_val_64(probe_prec, w0);
    let (c_probe, _) = build_series(&b_p, &lambda_p, probe_prec, probe_n)?;

    // Try the same evaluation strategy that the user-precision path will use.
    // If this works at probe precision, we expect it to work at user precision.
    if eval_sigma_with_shift(&c_probe, &b_p, &l_p, &lambda_p, probe_prec, &w0_p).is_ok() {
        return Ok(());
    }

    // Fallback: even if the shift mechanism fails at the probe level, the
    // user-precision build might just barely succeed (extra digits → tighter
    // convergence checks). Allow it through if the σ̃ coefficients aren't
    // pathologically blown up. This keeps us forgiving for borderline cases.
    let mut last_terms: Vec<Float> = Vec::with_capacity(probe_n);
    for (n, coefficient) in c_probe.iter().enumerate().skip(1) {
        let cn_abs = cnum::abs(coefficient, probe_prec);
        if !cn_abs.is_finite() {
            return Err(format!("Schröder σ̃ probe: coefficient {} is non-finite", n));
        }
        last_terms.push(cn_abs * w_abs.clone().pow(&Integer::from(n)));
    }
    let m = last_terms.len();
    if m >= 60 {
        let recent = Float::with_val_64(probe_prec, Float::sum(last_terms[m - 20..].iter())) / 20;
        let earlier =
            Float::with_val_64(probe_prec, Float::sum(last_terms[m - 40..m - 20].iter())) / 20;
        if recent > Float::with_val_64(probe_prec, &earlier / 2) {
            return Err(format!(
                "Schröder probe: σ̃ Taylor radius < |1−L| = {:.3} and σ̃-shift cannot rescue \
                 (recent term mean {:.3e} ≥ earlier {:.3e})",
                DisplayFloat(w_abs),
                DisplayFloat(&recent),
                DisplayFloat(&earlier)
            ));
        }
    }
    Ok(())
}

fn pick_n_terms(lambda_abs: &Float, prec: u64) -> Result<usize, String> {
    // Truncation error is dominated by ρ^N with ρ ≤ |t|/R_{σ̃⁻¹}. The shift
    // mechanism caps |t| ≲ 0.5·|1 − L|, making ρ ≲ 0.5 in adverse cases.
    // Hitting d decimal digits then needs N ≳ d/log10(1/ρ) ≈ 3.5·d. Scale
    // with decimal digits, not bits.
    let digits = u128::from(prec) * 30_103 / 100_000;
    let near_boundary = Float::with_val_64(prec, 1) - lambda_abs;
    let bonus = if near_boundary < cnum::decimal("0.2", prec) {
        250
    } else if near_boundary < cnum::decimal("0.4", prec) {
        80
    } else {
        0
    };
    usize::try_from((digits * 4 + 80 + bonus).max(150))
        .map_err(|_| "Schröder series length exceeds addressable memory".into())
}

fn build_series(
    b: &Complex,
    lambda: &Complex,
    prec: u64,
    m: usize,
) -> Result<(Vec<Complex>, Vec<Complex>), String> {
    let size = (m as u128 + 1)
        .checked_mul(m as u128 + 1)
        .ok_or("Schröder coefficient matrix exceeds addressable memory")?;
    cnum::check_complex_storage(size, prec)?;
    if cnum::verbose() {
        eprintln!("schröder build: {m} terms at {prec} bits; quadratic storage, cubic work");
    }
    let ln_b = Complex::with_val_64(prec, b.ln_ref());

    // q[j] = (ln b)^j / (j+1)! for j = 0..m-1; q has length m.
    let mut q: Vec<Complex> = Vec::with_capacity(m);
    let mut ln_b_pow = cnum::one(prec); // (ln b)^j
    let mut fact = rug::Integer::from(1u32); // (j+1)!
    for j in 0..m {
        fact *= j + 1;
        let q_j = Complex::with_val_64(prec, &ln_b_pow / &fact);
        q.push(q_j);
        ln_b_pow = Complex::with_val_64(prec, &ln_b_pow * &ln_b);
    }

    // q_pow[n] = q^n truncated to degree (m - n) for n = 1..=m-1.
    // q_pow[0] is unused.
    let mut q_pow: Vec<Vec<Complex>> = Vec::with_capacity(m);
    q_pow.push(Vec::new());
    if m >= 2 {
        q_pow.push(q.clone());
    }
    for n in 2..=m - 1 {
        if cnum::verbose() && (n.is_power_of_two() || n == m - 1) {
            eprintln!("schröder powers: {n}/{}", m - 1);
        }
        let need = m - n; // truncate to degree `need`
        let prev = &q_pow[n - 1];
        let mut new_pow = vec![cnum::zero(prec); need + 1];
        let i_max = prev.len().min(need + 1);
        for i in 0..i_max {
            let j_max = q.len().min(need + 1 - i);
            for j in 0..j_max {
                let prod = Complex::with_val_64(prec, &prev[i] * &q[j]);
                new_pow[i + j] += prod;
            }
        }
        q_pow.push(new_pow);
    }

    // λ^k cache for k = 0..=m.
    let mut lam_pow: Vec<Complex> = Vec::with_capacity(m + 1);
    lam_pow.push(cnum::one(prec));
    for _ in 1..=m {
        let next = Complex::with_val_64(prec, lam_pow.last().unwrap() * lambda);
        lam_pow.push(next);
    }

    // c[1..=m]: σ̃ coefficients with c[1] = 1.
    let mut c: Vec<Complex> = vec![cnum::zero(prec); m + 1];
    c[1] = cnum::one(prec);

    for big_n in 2..=m {
        if cnum::verbose() && (big_n.is_power_of_two() || big_n == m) {
            eprintln!("schröder coefficients: {big_n}/{m}");
        }
        let mut sum = cnum::zero(prec);
        for n in 1..big_n {
            let idx = big_n - n;
            if idx >= q_pow[n].len() {
                continue;
            }
            let term1 = Complex::with_val_64(prec, &c[n] * &lam_pow[n]);
            let term2 = Complex::with_val_64(prec, &term1 * &q_pow[n][idx]);
            sum += term2;
        }
        let denom = Complex::with_val_64(prec, &lam_pow[big_n] - lambda);
        if denom.real().is_zero() && denom.imag().is_zero() {
            return Err(format!(
                "Schröder resonance: λ^{} − λ = 0 (root-of-unity multiplier)",
                big_n
            ));
        }
        let neg_sum = Complex::with_val_64(prec, -&sum);
        c[big_n] = Complex::with_val_64(prec, &neg_sum / &denom);
    }

    drop(q_pow);

    let d = reverse_series(&c, m, prec);
    Ok((c, d))
}

/// Series reversion: given `σ̃(w) = w + Σ_{n≥2} c_n w^n` (so `c[1] = 1`),
/// compute `d_n` such that `σ̃⁻¹(t) = t + Σ_{n≥2} d_n t^n`. Solves
/// `σ̃(σ̃⁻¹(t)) = t` order by order using `pw[k][N] = [t^N] (σ̃⁻¹)^k`.
fn reverse_series(c: &[Complex], m: usize, prec: u64) -> Vec<Complex> {
    let mut d: Vec<Complex> = vec![cnum::zero(prec); m + 1];
    d[1] = cnum::one(prec);

    let mut pw: Vec<Vec<Complex>> = Vec::with_capacity(m + 2);
    pw.push(Vec::new());
    let mut pw1 = vec![cnum::zero(prec); m + 1];
    pw1[1] = cnum::one(prec);
    pw.push(pw1);
    for _ in 2..=m {
        pw.push(vec![cnum::zero(prec); m + 1]);
    }

    for big_n in 2..=m {
        if cnum::verbose() && (big_n.is_power_of_two() || big_n == m) {
            eprintln!("schröder reversion: {big_n}/{m}");
        }
        for k in 2..=big_n {
            let mut s = cnum::zero(prec);
            // pw[k][big_n] = Σ_{j=1..=big_n-k+1} d[j] · pw[k-1][big_n - j]
            for j in 1..=(big_n - k + 1) {
                let term = Complex::with_val_64(prec, &d[j] * &pw[k - 1][big_n - j]);
                s += term;
            }
            pw[k][big_n] = s;
        }
        let mut rhs = cnum::zero(prec);
        for k in 2..=big_n {
            let term = Complex::with_val_64(prec, &c[k] * &pw[k][big_n]);
            rhs -= term;
        }
        d[big_n] = rhs;
        pw[1][big_n] = d[big_n].clone();
    }
    d
}

/// Evaluate `σ̃(w₀)` from σ̃ Taylor coefficients, optionally using the
/// functional-equation shift `σ̃(φ(w)) = λ·σ̃(w)` (`φ(w) = b^(L+w) − L`) when
/// the direct Taylor at 0 doesn't converge at w₀.
///
///   * |λ|<1 (attracting): forward-iterate φ. φ contracts toward 0 with rate
///     |λ|, so eventually `w_curr` enters the Taylor disk. Compensate by
///     `σ̃(w₀) = σ̃(φⁿ(w₀)) / λⁿ`.
///   * |λ|>1 (repelling): backward-iterate `φ⁻¹(w) = log_b(L+w) − L`. 0 is an
///     attracting fixed point of φ⁻¹ (multiplier 1/λ, |1/λ|<1), so points in
///     its basin contract to 0. Compensate by `σ̃(w₀) = σ̃(φ⁻ⁿ(w₀)) · λⁿ`.
///     Uses principal log; this is fine as long as the orbit `L + φ⁻ᵏ(w)`
///     stays away from the origin (the log branch point).
///
/// Returns `(σ̃(w0), inner_radius)` where `inner_radius` is the `|w_curr|` at
/// which the Taylor series was actually evaluated (after φ-shifts). This lets
/// callers know the proven convergent radius for future `σ̃⁻¹` evaluations.
fn eval_sigma_with_shift(
    sigma: &[Complex],
    b: &Complex,
    l: &Complex,
    lambda: &Complex,
    prec: u64,
    w0: &Complex,
) -> Result<(Complex, Float), String> {
    // Fast path: direct Taylor at 0 reaches w₀.
    let w0_abs = cnum::abs(w0, prec);
    let direct = eval_series_checked(sigma, w0, prec);
    if let Ok(v) = direct {
        return Ok((v, w0_abs));
    }

    let lam_abs = cnum::abs(lambda, prec);
    let attracting = lam_abs < 1;

    let ln_b = Complex::with_val_64(prec, b.ln_ref());
    let mut w_curr = w0.clone();
    let mut n_shifts = Integer::new();
    let mut orbit_checkpoint = w0.clone();
    let sigma_at_curr = loop {
        match eval_series_checked(sigma, &w_curr, prec) {
            Ok(s) => break s,
            Err(e) => {
                let l_plus_w = Complex::with_val_64(prec, l + &w_curr);
                let lpw_abs = cnum::abs(&l_plus_w, prec);
                if !lpw_abs.is_finite() || lpw_abs.is_zero() {
                    return Err(format!(
                        "σ̃-shift: L+w_curr = 0 or non-finite (|·|={}); cannot continue \
                         (orbit hit a singularity of φ or φ⁻¹)",
                        DisplayFloat(&lpw_abs)
                    ));
                }
                if attracting {
                    // φ(w) = b^(L+w) − L = exp(ln_b·(L+w)) − L
                    let exp_arg = Complex::with_val_64(prec, &l_plus_w * &ln_b);
                    let bw = cnum::checked_exp(&exp_arg, prec)?;
                    w_curr = Complex::with_val_64(prec, &bw - l);
                } else {
                    // φ⁻¹(w) = log_b(L+w) − L = ln(L+w)/ln_b − L (principal log)
                    let ln_lpw = Complex::with_val_64(prec, l_plus_w.ln_ref());
                    let logb_lpw = Complex::with_val_64(prec, &ln_lpw / &ln_b);
                    w_curr = Complex::with_val_64(prec, &logb_lpw - l);
                }
                n_shifts += 1;
                if w_curr == orbit_checkpoint {
                    return Err(format!(
                        "Schröder sigma-shift repeated an orbit value without convergence: {e}"
                    ));
                }
                if n_shifts.is_power_of_two() {
                    orbit_checkpoint = w_curr.clone();
                    if cnum::verbose() {
                        eprintln!(
                            "schröder sigma-shift: {n_shifts} steps, |w|={:.6e}",
                            DisplayFloat(&cnum::abs(&w_curr, prec))
                        );
                    }
                }
            }
        }
    };

    let inner_radius = cnum::abs(&w_curr, prec);

    if n_shifts == 0 {
        return Ok((sigma_at_curr, inner_radius));
    }

    if cnum::verbose() {
        eprintln!(
            "schröder: σ̃-shift converged after {} {} steps (|λ|={:.6})",
            n_shifts,
            if attracting {
                "forward φ"
            } else {
                "backward φ⁻¹"
            },
            DisplayFloat(&lam_abs)
        );
    }

    let mut lam_pow = cnum::one(prec);
    let mut unwound = Integer::new();
    while unwound < n_shifts {
        lam_pow = Complex::with_val_64(prec, &lam_pow * lambda);
        unwound += 1;
    }
    let value = if attracting {
        Complex::with_val_64(prec, &sigma_at_curr / &lam_pow)
    } else {
        Complex::with_val_64(prec, &sigma_at_curr * &lam_pow)
    };
    if !cnum::is_finite(&value) || cnum::is_zero(&value) {
        return Err("Schröder shift lost the nonzero normalization coordinate".into());
    }
    Ok((value, inner_radius))
}

/// Check the computed tail and a roundoff estimate against working precision.
/// This is a convergence diagnostic, not a rigorous infinite-tail bound.
fn eval_series_checked(coeffs: &[Complex], w: &Complex, prec: u64) -> Result<Complex, String> {
    if coeffs.len() < 2 || !cnum::is_finite(w) {
        return Err("Schröder series requires coefficients and a finite argument".into());
    }
    let high = coeffs.len() - 1;
    let mut acc = cnum::zero(prec);
    let mut w_pow = cnum::one(prec);
    let mut term_sum = Float::new_64(prec);
    // Track the final ~5% of terms.
    let tail_start = high
        .saturating_sub(high / 20)
        .max(high.saturating_sub(50))
        .max(1);
    let mut tail_sum = Float::new_64(prec);
    for (i, coefficient) in coeffs.iter().enumerate().skip(1) {
        w_pow = Complex::with_val_64(prec, &w_pow * w);
        if !cnum::is_finite(&w_pow) || (cnum::is_zero(&w_pow) && !cnum::is_zero(w)) {
            return Err(format!(
                "Schröder series power {} exceeded the exponent range",
                i
            ));
        }
        let term = Complex::with_val_64(prec, coefficient * &w_pow);
        let term_abs = cnum::abs(&term, prec);
        if !term_abs.is_finite() {
            return Err(format!("Schröder series term {} overflowed", i));
        }
        term_sum += &term_abs;
        if i >= tail_start {
            tail_sum += term_abs;
        }
        acc += term;
    }
    let tolerance = cnum::working_epsilon(prec) * cnum::abs(&acc, prec);
    let roundoff = (term_sum * high * high)
        >> usize::try_from(prec).expect("precision exceeds addressable bits");
    if !cnum::is_finite(&acc) || tail_sum > tolerance || roundoff > tolerance {
        return Err(format!(
            "Schröder series not accurate enough at |w|={}: tail {}, roundoff estimate {}, tolerance {}",
            DisplayFloat(&cnum::abs(w, prec)), DisplayFloat(&tail_sum),
            DisplayFloat(&roundoff), DisplayFloat(&tolerance)
        ));
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_driven_series_order_is_not_clamped_to_1500() {
        let prec = cnum::digits_to_bits(1000);
        let lambda = cnum::decimal("0.5", prec);
        assert!(pick_n_terms(&lambda, prec).unwrap() > 1500);
        let base = Complex::with_val_64(prec, 2);
        let multiplier = Complex::with_val_64(prec, lambda);
        assert!(build_series(&base, &multiplier, prec, usize::MAX)
            .unwrap_err()
            .contains("addressable memory"));
    }

    #[test]
    fn series_tail_has_no_machine_precision_floor() {
        for digits in [50, 70, 1000] {
            let prec = cnum::digits_to_bits(digits);
            for scale in ["1", "1e-1000"] {
                let mut coefficients = vec![cnum::zero(prec); 41];
                coefficients[1] = Complex::with_val_64(prec, cnum::decimal(scale, prec));
                coefficients[40] = coefficients[1].clone() * cnum::epsilon(20, prec);
                assert!(eval_series_checked(&coefficients, &cnum::one(prec), prec).is_err());
                coefficients[40] = coefficients[1].clone() * cnum::epsilon(digits + 40, prec);
                assert!(eval_series_checked(&coefficients, &cnum::one(prec), prec).is_ok());
            }
        }
    }

    #[test]
    fn nonfinite_series_and_unrepresentable_exponent_are_errors() {
        let prec = cnum::digits_to_bits(50);
        let coefficients = [
            cnum::zero(prec),
            cnum::one(prec),
            Complex::with_val_64(prec, Float::with_val_64(prec, rug::float::Special::Nan)),
        ];
        assert!(eval_series_checked(&coefficients, &cnum::one(prec), prec).is_err());
        assert!(lambda_pow(
            &Complex::with_val_64(prec, cnum::decimal("1e1000", prec)),
            &cnum::one(prec),
            prec
        )
        .is_err());
    }
}
