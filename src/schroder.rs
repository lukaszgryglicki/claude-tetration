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

use rug::{Complex, Float};

use crate::{cnum, regions::FixedPointData};

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
    /// `|λ|` as f64 — drives shift direction and safe-radius decisions.
    pub lam_abs: f64,
    /// σ̃⁻¹ Taylor coefficients (series reversion of σ̃). `sigma_inv[i]` is the
    /// `i`-th coefficient; `sigma_inv[0] = 0`.
    pub sigma_inv: Vec<Complex>,
    /// `σ̃(1 − L)` — entry point for the formula `F(z) = L + σ̃⁻¹(s1·λ^z)`.
    pub s1: Complex,
    /// `safe_radius = 0.5 · max(|1−L|, 0.5)`. Heuristic outer bound for σ̃⁻¹ convergence.
    pub safe_radius: f64,
    /// Actual inner radius at which σ̃ was evaluated during setup (after φ-shifts).
    /// σ̃⁻¹ is guaranteed convergent at `|t| ≤ sigma_inner_radius` (since σ̃ and σ̃⁻¹
    /// share approximately the same convergence radius near the identity).
    pub sigma_inner_radius: f64,
    /// MPC bit precision the state was built at — must match the precision
    /// used for per-cell evaluation.
    pub prec: u32,
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
    prec: u32,
) -> Result<Complex, String> {
    // setup_schroder now includes the anchor check (F(0)=1), so any degenerate
    // state is rejected before we reach eval.
    let state = setup_schroder(b, fp_data, prec)?;
    let f_h = eval_schroder(&state, h)?;
    validate_functional_equation(&state, h, &f_h, prec)?;
    Ok(f_h)
}

/// Post-validate `F(h)` by checking the functional equation
/// `F(h+1) = b^F(h)` numerically. The σ̃ Taylor series can converge inside
/// the heuristic `safe_radius` yet still evaluate to a wrong value when the
/// actual radius of convergence of `σ⁻¹` is smaller (this happens for real
/// bases just below η, where `|λ| → 1`). The functional equation is the
/// gold-standard correctness check: if it fails, we know `F(h)` is wrong
/// even though the algorithm reported success.
fn validate_functional_equation(
    state: &SchroderState,
    h: &Complex,
    f_h: &Complex,
    prec: u32,
) -> Result<(), String> {
    let one = cnum::one(prec);
    let h_plus_one = Complex::with_val(prec, h + &one);
    let f_h_plus_one = eval_schroder(state, &h_plus_one)?;

    // b^F(h) = exp(F(h) · ln b)
    let exponent = Complex::with_val(prec, f_h * &state.ln_b);
    let b_pow_f_h = Complex::with_val(prec, exponent.exp_ref());

    let diff = Complex::with_val(prec, &f_h_plus_one - &b_pow_f_h);
    let diff_abs = Float::with_val(prec, diff.abs_ref()).to_f64();
    let f_abs = Float::with_val(prec, f_h_plus_one.abs_ref()).to_f64().max(1.0);
    let rel = diff_abs / f_abs;

    // Tolerance: 1e-6 absolute / relative is generous enough that a working
    // 20-digit Schröder eval easily passes, but catches the corruption band
    // where the wrong value differs by O(1) or more.
    let tol = 1e-6;
    if !rel.is_finite() || rel > tol {
        if cnum::verbose() {
            eprintln!(
                "schröder validation FAILED: |F(h+1) − b^F(h)| = {:.3e}, |F(h+1)| ≈ {:.3e}, rel = {:.3e}",
                diff_abs, f_abs, rel
            );
        }
        return Err(format!(
            "Schröder series evaluation failed validation: \
             |F(h+1) − b^F(h)| / max(|F(h+1)|, 1) = {:.3e} exceeds {:.0e} \
             — Taylor series likely outside true radius of convergence (|λ|={:.4})",
            rel, tol, state.lam_abs
        ));
    }
    if cnum::verbose() {
        eprintln!(
            "schröder validation OK: |F(h+1) − b^F(h)| / max(|F(h+1)|, 1) = {:.3e}",
            rel
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
    prec: u32,
) -> Result<SchroderState, String> {
    let l = &fp_data.fixed_point;
    let lambda = &fp_data.lambda;
    let lam_abs = fp_data.lambda_abs;
    if !(lam_abs > 0.0 && lam_abs.is_finite()) {
        return Err(format!("Schröder: invalid |λ| = {}", lam_abs));
    }
    // The build recursion divides by `λ^N − λ`, which vanishes only at
    // λ ∈ root-of-unity. For |λ|=1 exactly (true parabolic), we'd hit a
    // zero denominator at N=1 (since λ^1 − λ = 0 trivially). Block only
    // a very tight band around |λ|=1; let σ̃-shift + extended N handle
    // the boundary band (|λ|=0.95–0.99).
    if (lam_abs - 1.0).abs() < 0.005 {
        return Err(format!(
            "Schröder unreliable on parabolic boundary (|λ| = {})",
            lam_abs
        ));
    }

    let one = cnum::one(prec);
    let w0 = Complex::with_val(prec, &one - l);
    let w0_abs = Float::with_val(prec, w0.abs_ref()).to_f64();
    if !(w0_abs.is_finite() && w0_abs > 0.0) {
        return Err(format!("Schröder: bad |1−L| = {}", w0_abs));
    }

    // Cheap pre-check at modest precision: if the σ̃ Taylor terms at w₀ are
    // clearly diverging AND the σ̃-shift mechanism can't rescue (orbit hits a
    // singularity, or we're in the repelling case where φ⁻¹ at w₀=1−L lands on
    // L+w=0), bail before paying for an O(N²) build at user precision. The
    // shift mechanism is given a chance through `eval_sigma_with_shift`; this
    // probe only filters obviously-hopeless cases.
    radius_probe(b, l, lambda, prec, &w0, w0_abs)?;

    let n_terms = pick_n_terms(lam_abs, prec);
    let (sigma, sigma_inv) = build_series(b, lambda, prec, n_terms)?;

    // σ̃(1 − L) is needed for the F(z) = L + σ̃⁻¹(σ̃(1−L)·λ^z) formula. If the
    // Taylor at 0 doesn't reach `w₀ = 1 − L` directly (radius too small), use
    // the functional equation σ̃(φ(w)) = λ·σ̃(w) with φ(w) = b^(L+w) − L.
    //   * |λ|<1 (attracting): forward-iterate φ; it contracts toward 0.
    //   * |λ|>1 (repelling): backward-iterate φ⁻¹(w) = log_b(L+w) − L; 0 is
    //     attracting for φ⁻¹ (multiplier 1/λ, |1/λ|<1), so a starting point in
    //     its basin contracts to 0.
    let (s1, sigma_inner_radius) = eval_sigma_with_shift(&sigma, b, l, lambda, prec, &w0)?;

    let ln_lambda = Complex::with_val(prec, lambda.ln_ref());
    let ln_b = Complex::with_val(prec, b.ln_ref());
    let safe_radius = 0.5 * w0_abs.max(0.5);

    if cnum::verbose() {
        let s1_abs = Float::with_val(prec, s1.abs_ref()).to_f64();
        eprintln!(
            "schröder setup: |λ|={:.6} |1−L|={:.6} |s1|={:.6} safe={:.6} inner_r={:.6} N={}",
            lam_abs, w0_abs, s1_abs, safe_radius, sigma_inner_radius, n_terms
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
    let f_zero = eval_schroder(&base_state, &zero)?;
    let anchor_diff = Complex::with_val(prec, &f_zero - &one_c);
    let anchor_err = Float::with_val(prec, anchor_diff.abs_ref()).to_f64();
    if !anchor_err.is_finite() || anchor_err > 1e-6 {
        return Err(format!(
            "Schröder anchor check failed: F(0) = {} (expected 1.0); \
             σ̃-shift likely produced degenerate fixed-point solution F≡L",
            f_zero
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
    let prec = state.prec;
    let lam_h = lambda_pow(h, &state.ln_lambda, prec);
    let t = Complex::with_val(prec, &state.s1 * &lam_h);
    let t_abs = Float::with_val(prec, t.abs_ref()).to_f64();

    if !t_abs.is_finite() || t_abs <= 0.0 {
        return Err(format!("Schröder: bad |t| = {}", t_abs));
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
        return Ok(Complex::with_val(prec, &state.l + &inv_t));
    }

    let log_lam_abs = state.lam_abs.ln(); // negative (|λ|<1) or positive (>1)
    let initial_target = state.sigma_inner_radius.min(state.safe_radius).min(t_abs);

    let mut effective_target = initial_target * 0.5;
    let mut found: Option<(i64, Complex)> = None;
    for _attempt in 0..30 {
        let ratio = (effective_target / t_abs).ln();
        let k_raw = ratio / log_lam_abs;
        let mut k: i64 = if log_lam_abs < 0.0 {
            k_raw.ceil() as i64
        } else {
            k_raw.floor() as i64
        };
        if k == 0 {
            // Need at least one shift to move strictly inside effective_target
            k = if log_lam_abs < 0.0 { 1 } else { -1 };
        }
        if k.unsigned_abs() > 5000 {
            return Err(format!(
                "Schröder: requested shift |k|={} too large; argument too far from fixed point",
                k.unsigned_abs()
            ));
        }

        let h_shifted = Complex::with_val(prec, h + k);
        let lam_h_shifted = lambda_pow(&h_shifted, &state.ln_lambda, prec);
        let t_shifted = Complex::with_val(prec, &state.s1 * &lam_h_shifted);
        match eval_series_checked(&state.sigma_inv, &t_shifted, prec) {
            Ok(inv_t_shifted) => {
                let f0 = Complex::with_val(prec, &state.l + &inv_t_shifted);
                found = Some((k, f0));
                if cnum::verbose() {
                    let t_sh = Float::with_val(prec, t_shifted.abs_ref()).to_f64();
                    eprintln!(
                        "schröder eval: k={} |t_shifted|={:.6e} (target={:.3e})",
                        k, t_sh, effective_target
                    );
                }
                break;
            }
            Err(_) => {
                effective_target *= 0.5;
            }
        }
    }
    let (k, mut f) = found.ok_or_else(|| {
        format!(
            "Schröder: σ̃⁻¹ never converged after k-shift attempts (|t|={:.3e}, |λ|={:.4})",
            t_abs, state.lam_abs
        )
    })?;

    if k > 0 {
        // F(h) = log_b applied k times to F(h+k).
        for _ in 0..k {
            let ln_f = Complex::with_val(prec, f.ln_ref());
            f = Complex::with_val(prec, &ln_f / &state.ln_b);
        }
    } else {
        // k < 0: F(h) = b^· applied |k| times to F(h+k) = F(h−|k|).
        for _ in 0..(-k) {
            let exponent = Complex::with_val(prec, &f * &state.ln_b);
            f = Complex::with_val(prec, exponent.exp_ref());
        }
    }
    Ok(f)
}

/// `λ^h = exp(h · ln λ)`. The principal branch of `ln λ` is fine inside the
/// Shell-Thron interior because `λ` is never zero there (`λ = 0` only when
/// `ln b = 0`, i.e., `b = 1`, which is filtered out earlier).
fn lambda_pow(h: &Complex, ln_lambda: &Complex, prec: u32) -> Complex {
    let exponent = Complex::with_val(prec, h * ln_lambda);
    Complex::with_val(prec, exponent.exp_ref())
}

/// Cheap pre-check at modest precision: build ~80 σ̃ coefficients and do a
/// trial run of the σ̃-shift mechanism. If neither direct evaluation nor the
/// shift can produce a finite σ̃(w₀), bail before paying for the full O(N²)
/// build at user precision. This catches the b=e/b=2/b=10/b=−2 cases where
/// the φ⁻¹ orbit immediately hits L+w=0 and the log branch cut.
fn radius_probe(
    b: &Complex,
    l: &Complex,
    lambda: &Complex,
    user_prec: u32,
    w0: &Complex,
    w_abs: f64,
) -> Result<(), String> {
    if !(w_abs.is_finite() && w_abs > 0.0) {
        return Err(format!("radius probe: bad |w| = {}", w_abs));
    }
    let probe_prec = 256u32.min(user_prec);
    let probe_n = 80usize;
    let b_p = Complex::with_val(probe_prec, b);
    let l_p = Complex::with_val(probe_prec, l);
    let lambda_p = Complex::with_val(probe_prec, lambda);
    let w0_p = Complex::with_val(probe_prec, w0);
    let (c_probe, _) = build_series(&b_p, &lambda_p, probe_prec, probe_n)?;

    // Try the same evaluation strategy that the user-precision path will use.
    // If this works at probe precision, we expect it to work at user precision.
    if eval_sigma_with_shift(&c_probe, &b_p, &l_p, &lambda_p, probe_prec, &w0_p).is_ok()
    {
        return Ok(());
    }

    // Fallback: even if the shift mechanism fails at the probe level, the
    // user-precision build might just barely succeed (extra digits → tighter
    // convergence checks). Allow it through if the σ̃ coefficients aren't
    // pathologically blown up. This keeps us forgiving for borderline cases.
    let mut last_terms: Vec<f64> = Vec::with_capacity(probe_n);
    for n in 1..c_probe.len() {
        let cn_abs = Float::with_val(probe_prec, c_probe[n].abs_ref()).to_f64();
        if !cn_abs.is_finite() {
            return Err(format!(
                "Schröder σ̃ probe: coefficient {} is non-finite",
                n
            ));
        }
        last_terms.push(cn_abs * w_abs.powi(n as i32));
    }
    let m = last_terms.len();
    if m >= 60 {
        let recent: f64 = last_terms[m - 20..].iter().sum::<f64>() / 20.0;
        let earlier: f64 = last_terms[m - 40..m - 20].iter().sum::<f64>() / 20.0;
        if recent > earlier * 0.5 {
            return Err(format!(
                "Schröder probe: σ̃ Taylor radius < |1−L| = {:.3} and σ̃-shift cannot rescue \
                 (recent term mean {:.3e} ≥ earlier {:.3e})",
                w_abs, recent, earlier
            ));
        }
    }
    Ok(())
}

fn pick_n_terms(lambda_abs: f64, prec: u32) -> usize {
    // Truncation error is dominated by ρ^N with ρ ≤ |t|/R_{σ̃⁻¹}. The shift
    // mechanism caps |t| ≲ 0.5·|1 − L|, making ρ ≲ 0.5 in adverse cases.
    // Hitting d decimal digits then needs N ≳ d/log10(1/ρ) ≈ 3.5·d. Scale
    // with decimal digits, not bits.
    //
    // The σ̃-shift mechanism (in eval_sigma_with_shift and eval_schroder)
    // brings |w_curr| and |t| arbitrarily close to 0, so even when the series
    // radius R_σ is small (near-boundary |λ|), we can use a moderate N. The
    // O(N³) build cost dominates, so keep N capped at 1500 — adequate when
    // σ̃-shift is doing its job. Cases that genuinely need N>1500 won't be
    // helped by larger N (the build cost would be hours).
    let digits = (prec as f64 * std::f64::consts::LOG10_2) as usize;
    let near_boundary = 1.0 - lambda_abs;
    let bonus = if near_boundary < 0.2 { 250 } else if near_boundary < 0.4 { 80 } else { 0 };
    let base = digits.saturating_mul(4) + 80 + bonus;
    base.clamp(150, 1500)
}

fn build_series(
    b: &Complex,
    lambda: &Complex,
    prec: u32,
    m: usize,
) -> Result<(Vec<Complex>, Vec<Complex>), String> {
    let ln_b = Complex::with_val(prec, b.ln_ref());

    // q[j] = (ln b)^j / (j+1)! for j = 0..m-1; q has length m.
    let mut q: Vec<Complex> = Vec::with_capacity(m);
    let mut ln_b_pow = cnum::one(prec); // (ln b)^j
    let mut fact = rug::Integer::from(1u32); // (j+1)!
    for j in 0..m {
        fact *= (j as u32) + 1;
        let q_j = Complex::with_val(prec, &ln_b_pow / &fact);
        q.push(q_j);
        ln_b_pow = Complex::with_val(prec, &ln_b_pow * &ln_b);
    }

    // q_pow[n] = q^n truncated to degree (m - n) for n = 1..=m-1.
    // q_pow[0] is unused.
    let mut q_pow: Vec<Vec<Complex>> = Vec::with_capacity(m);
    q_pow.push(Vec::new());
    if m >= 2 {
        q_pow.push(q.clone());
    }
    for n in 2..=m - 1 {
        let need = m - n; // truncate to degree `need`
        let prev = &q_pow[n - 1];
        let mut new_pow = vec![cnum::zero(prec); need + 1];
        let i_max = prev.len().min(need + 1);
        for i in 0..i_max {
            let j_max = q.len().min(need + 1 - i);
            for j in 0..j_max {
                let prod = Complex::with_val(prec, &prev[i] * &q[j]);
                new_pow[i + j] += prod;
            }
        }
        q_pow.push(new_pow);
    }

    // λ^k cache for k = 0..=m.
    let mut lam_pow: Vec<Complex> = Vec::with_capacity(m + 1);
    lam_pow.push(cnum::one(prec));
    for _ in 1..=m {
        let next = Complex::with_val(prec, lam_pow.last().unwrap() * lambda);
        lam_pow.push(next);
    }

    // c[1..=m]: σ̃ coefficients with c[1] = 1.
    let mut c: Vec<Complex> = vec![cnum::zero(prec); m + 1];
    c[1] = cnum::one(prec);

    for big_n in 2..=m {
        let mut sum = cnum::zero(prec);
        for n in 1..big_n {
            let idx = big_n - n;
            if idx >= q_pow[n].len() {
                continue;
            }
            let term1 = Complex::with_val(prec, &c[n] * &lam_pow[n]);
            let term2 = Complex::with_val(prec, &term1 * &q_pow[n][idx]);
            sum += term2;
        }
        let denom = Complex::with_val(prec, &lam_pow[big_n] - lambda);
        if denom.real().is_zero() && denom.imag().is_zero() {
            return Err(format!(
                "Schröder resonance: λ^{} − λ = 0 (root-of-unity multiplier)",
                big_n
            ));
        }
        let neg_sum = Complex::with_val(prec, -&sum);
        c[big_n] = Complex::with_val(prec, &neg_sum / &denom);
    }

    drop(q_pow);

    let d = reverse_series(&c, m, prec);
    Ok((c, d))
}

/// Series reversion: given `σ̃(w) = w + Σ_{n≥2} c_n w^n` (so `c[1] = 1`),
/// compute `d_n` such that `σ̃⁻¹(t) = t + Σ_{n≥2} d_n t^n`. Solves
/// `σ̃(σ̃⁻¹(t)) = t` order by order using `pw[k][N] = [t^N] (σ̃⁻¹)^k`.
fn reverse_series(c: &[Complex], m: usize, prec: u32) -> Vec<Complex> {
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
        for k in 2..=big_n {
            let mut s = cnum::zero(prec);
            // pw[k][big_n] = Σ_{j=1..=big_n-k+1} d[j] · pw[k-1][big_n - j]
            for j in 1..=(big_n - k + 1) {
                let term = Complex::with_val(prec, &d[j] * &pw[k - 1][big_n - j]);
                s += term;
            }
            pw[k][big_n] = s;
        }
        let mut rhs = cnum::zero(prec);
        for k in 2..=big_n {
            let term = Complex::with_val(prec, &c[k] * &pw[k][big_n]);
            rhs -= term;
        }
        d[big_n] = rhs;
        pw[1][big_n] = d[big_n].clone();
    }
    d
}

/// Evaluate `Σ_{i=1}^{coeffs.len()-1} coeffs[i] · w^i` via Horner. The constant
/// term (`coeffs[0]`) is treated as zero (both σ̃ and σ̃⁻¹ vanish at 0).
fn eval_series(coeffs: &[Complex], w: &Complex, prec: u32) -> Complex {
    if coeffs.len() < 2 {
        return cnum::zero(prec);
    }
    let high = coeffs.len() - 1;
    let mut acc = coeffs[high].clone();
    for i in (1..high).rev() {
        acc = Complex::with_val(prec, &acc * w);
        acc = Complex::with_val(prec, &acc + &coeffs[i]);
    }
    Complex::with_val(prec, &acc * w)
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
/// Returns `(σ̃(w0), inner_radius)` where `inner_radius` is the `|w_curr|` at
/// which the Taylor series was actually evaluated (after φ-shifts). This lets
/// callers know the proven convergent radius for future `σ̃⁻¹` evaluations.
fn eval_sigma_with_shift(
    sigma: &[Complex],
    b: &Complex,
    l: &Complex,
    lambda: &Complex,
    prec: u32,
    w0: &Complex,
) -> Result<(Complex, f64), String> {
    // Fast path: direct Taylor at 0 reaches w₀.
    let w0_abs = Float::with_val(prec, w0.abs_ref()).to_f64();
    let direct = eval_series_checked(sigma, w0, prec);
    if let Ok(v) = direct {
        return Ok((v, w0_abs));
    }

    let lam_abs = Float::with_val(prec, lambda.abs_ref()).to_f64();
    let attracting = lam_abs < 1.0;

    let ln_b = Complex::with_val(prec, b.ln_ref());
    let mut w_curr = w0.clone();
    let mut n_shifts: u32 = 0;
    let max_shifts: u32 = 500;
    let sigma_at_curr = loop {
        match eval_series_checked(sigma, &w_curr, prec) {
            Ok(s) => break s,
            Err(e) => {
                if n_shifts >= max_shifts {
                    return Err(format!(
                        "σ̃-shift exhausted {} iterations without entering Taylor disk \
                         (|λ|={:.3}, attracting={}): {}",
                        max_shifts, lam_abs, attracting, e
                    ));
                }
                let l_plus_w = Complex::with_val(prec, l + &w_curr);
                let lpw_abs = Float::with_val(prec, l_plus_w.abs_ref()).to_f64();
                if !lpw_abs.is_finite() || lpw_abs == 0.0 {
                    return Err(format!(
                        "σ̃-shift: L+w_curr = 0 or non-finite (|·|={}); cannot continue \
                         (orbit hit a singularity of φ or φ⁻¹)",
                        lpw_abs
                    ));
                }
                if attracting {
                    // φ(w) = b^(L+w) − L = exp(ln_b·(L+w)) − L
                    let exp_arg = Complex::with_val(prec, &l_plus_w * &ln_b);
                    let bw = Complex::with_val(prec, exp_arg.exp_ref());
                    w_curr = Complex::with_val(prec, &bw - l);
                } else {
                    // φ⁻¹(w) = log_b(L+w) − L = ln(L+w)/ln_b − L (principal log)
                    let ln_lpw = Complex::with_val(prec, l_plus_w.ln_ref());
                    let logb_lpw = Complex::with_val(prec, &ln_lpw / &ln_b);
                    w_curr = Complex::with_val(prec, &logb_lpw - l);
                }
                n_shifts += 1;
            }
        }
    };

    let inner_radius = Float::with_val(prec, w_curr.abs_ref()).to_f64();

    if n_shifts == 0 {
        return Ok((sigma_at_curr, inner_radius));
    }

    if cnum::verbose() {
        eprintln!(
            "schröder: σ̃-shift converged after {} {} steps (|λ|={:.6})",
            n_shifts,
            if attracting { "forward φ" } else { "backward φ⁻¹" },
            lam_abs
        );
    }

    let mut lam_pow = cnum::one(prec);
    for _ in 0..n_shifts {
        lam_pow = Complex::with_val(prec, &lam_pow * lambda);
    }
    let value = if attracting {
        Complex::with_val(prec, &sigma_at_curr / &lam_pow)
    } else {
        Complex::with_val(prec, &sigma_at_curr * &lam_pow)
    };
    Ok((value, inner_radius))
}

/// Evaluate the series term by term and check that the tail decays. The
/// principled convergence check is "average of last few terms much smaller
/// than max term seen": for a truly-convergent series (`|w| < R`), the tail
/// drops like `(|w|/R)^n` and this ratio is microscopic. For a divergent or
/// barely-convergent series, the tail dominates.
fn eval_series_checked(coeffs: &[Complex], w: &Complex, prec: u32) -> Result<Complex, String> {
    if coeffs.len() < 2 {
        return Ok(cnum::zero(prec));
    }
    let high = coeffs.len() - 1;
    let mut acc = cnum::zero(prec);
    let mut w_pow = cnum::one(prec);
    let mut max_term: f64 = 0.0;
    // Track average of the final ~5% of terms.
    let tail_start = high.saturating_sub(high / 20).max(high.saturating_sub(50)).max(1);
    let mut tail_sum: f64 = 0.0;
    let mut tail_count: u32 = 0;
    for i in 1..=high {
        w_pow = Complex::with_val(prec, &w_pow * w);
        let term = Complex::with_val(prec, &coeffs[i] * &w_pow);
        let term_abs = Float::with_val(prec, term.abs_ref()).to_f64();
        if !term_abs.is_finite() {
            return Err(format!("Schröder series term {} overflowed", i));
        }
        if term_abs > max_term {
            max_term = term_abs;
        }
        if i >= tail_start {
            tail_sum += term_abs;
            tail_count += 1;
        }
        acc += term;
    }
    // The tail mean must be small relative to the required precision.
    // A fixed 1% threshold (0.01) passes for slowly-converging series near η,
    // where the series IS technically convergent but the N-term truncation gives
    // only a few digits of accuracy (e.g. b=1.4375 gives F(0) error 3.7e-5).
    // Using a precision-calibrated threshold forces more φ-shifts until the
    // evaluation point is deep inside the convergence disk.
    if max_term > 1e-30 && tail_count > 0 {
        let tail_mean = tail_sum / (tail_count as f64);
        let digits = (prec as f64 * std::f64::consts::LOG10_2) as i32;
        // Require tail < 10^(-(digits+4)): the 4-digit margin leaves room for
        // the series-to-series composition (σ̃ then σ̃⁻¹) and integer shifts.
        let tol = 10f64.powi(-(digits + 4));
        if tail_mean > max_term * tol.max(1e-15) {
            return Err(format!(
                "Schröder σ̃ series not accurate enough at |w|={:.3e}: \
                 tail mean {:.3e} vs max {:.3e} (tol={:.1e})",
                Float::with_val(prec, w.abs_ref()).to_f64(),
                tail_mean, max_term, tol
            ));
        }
    }
    Ok(acc)
}
