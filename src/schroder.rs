//! Schröder regular tetration for Shell-Thron interior bases.
//!
//! For `f(z) = b^z` with attracting fixed point `L = -W₀(-ln b) / ln b` and
//! multiplier `λ = L · ln b` (`|λ| < 1`), the Schröder function `σ` solving
//! `σ(L) = 0`, `σ'(L) = 1`, `σ(f(z)) = λ σ(z)` linearises the dynamics. From it,
//! `F_b(z) = L + σ̃⁻¹(σ̃(1 − L) · λ^z)` where `σ̃(w) = σ(L + w)`. This satisfies
//! `F_b(0) = 1` exactly and `F_b(z+1) = b^{F_b(z)}` analytically.
//!
//! Strictly attracting fixed points use the inverse Poincare germ directly:
//! `psi(lambda*t) = L*(exp(Log(b)*psi(t))-1)`, `psi'(0)=1`. Its differentiated
//! coefficient recurrence needs quadratic work and linear coefficient storage.
//! A conservative analytic disk controls the Taylor tail; normalization uses
//! the actual forward orbit from 1 and local inversion of this germ.
//!
//! The retained non-attracting construction computes `σ̃` from the equation
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

use rug::{float::Round, ops::Pow, Complex, Float, Integer};

use crate::{cnum, regions::FixedPointData};
use cnum::{DisplayComplex, DisplayFloat};

/// Cached per-base Schröder state. Built once via `setup_schroder` and reused
/// for many heights via `eval_schroder`. Strict attraction uses an O(N²)
/// inverse-series build; other states retain the classical construction.
#[derive(Clone)]
pub struct SchroderState {
    /// Original base retained for higher-precision reconstruction.
    pub base: Complex,
    /// Attracting (|λ|<1) or repelling (|λ|>1) fixed point of `b^z = z`.
    pub l: Complex,
    /// `ln(λ)` precomputed for `λ^h = exp(h · ln λ)` per cell.
    pub ln_lambda: Complex,
    /// `ln(b)` precomputed for the integer-shift `b^·` / `log_b` chain.
    pub ln_b: Complex,
    /// `|λ|` drives shift direction.
    pub lam_abs: Float,
    /// σ̃⁻¹ Taylor coefficients. `sigma_inv[i]` is the
    /// `i`-th coefficient; `sigma_inv[0] = 0`.
    pub sigma_inv: Vec<Complex>,
    /// `σ̃(1 − L)` — entry point for the formula `F(z) = L + σ̃⁻¹(s1·λ^z)`.
    pub s1: Complex,
    /// Outer evaluation target; inside the analytic disk for attracting states.
    pub safe_radius: Float,
    /// Local inverse target, or the classical σ̃ input radius after φ-shifts.
    /// The latter alone is not a convergence proof for the inverse series.
    pub sigma_inner_radius: Float,
    /// Conservative analytic disk for the attracting inverse germ. This
    /// controls truncation, not all floating-point or continuation errors.
    pub inverse_radius: Option<Float>,
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

pub(crate) fn tetrate_schroder_at_digits(
    b: &Complex,
    h: &Complex,
    fp_data: &FixedPointData,
    prec: u64,
    digits: u64,
) -> Result<Complex, String> {
    let state = setup_schroder(b, fp_data, prec)?;
    eval_schroder_at_digits(&state, h, digits)
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
    tolerance: &Float,
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

    let tol = tolerance;
    if !rel.is_finite() || rel > *tol {
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
            DisplayFloat(tol),
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

/// Build and normalize the inverse germ for base `b`. Strict attraction uses
/// direct Poincare coefficients; other states retain σ̃ and series reversion.
/// Per-cell evaluation is O(N) plus the integer-shift chain.
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
    if lam_abs < 1 {
        return setup_poincare(b, fp_data, prec);
    }
    // The attracting-disk argument does not validate the retained
    // near-neutral non-attracting construction.
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

    let ln_lambda = cnum::ln_complex(lambda, prec);
    let ln_b = cnum::ln_complex(b, prec);
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
        base: b.clone(),
        l: l.clone(),
        ln_lambda,
        ln_b,
        lam_abs,
        sigma_inv,
        s1,
        safe_radius,
        sigma_inner_radius,
        inverse_radius: None,
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

fn poincare_coefficients(
    ln_b: &Complex,
    lambda: &Complex,
    prec: u64,
    order: usize,
) -> Result<Vec<Complex>, String> {
    if order < 1 {
        return Err("Poincare series requires a positive order".into());
    }
    cnum::check_complex_storage((order as u128 + 1) * 2, prec)?;
    let mut powers = Vec::with_capacity(order + 1);
    powers.push(cnum::one(prec));
    for n in 1..=order {
        let power = Complex::with_val_64(prec, &powers[n - 1] * lambda);
        if !cnum::is_finite(&power) || cnum::is_zero(&power) {
            return Err("Poincare multiplier power exceeded MPFR's exponent range".into());
        }
        powers.push(power);
    }
    let mut inverse = vec![cnum::zero(prec); order + 1];
    inverse[1] = cnum::one(prec);
    for n in 2..=order {
        let mut sum = cnum::zero(prec);
        for k in 1..n {
            let product = Complex::with_val_64(prec, &inverse[k] * &inverse[n - k]);
            sum += Complex::with_val_64(prec, product * &powers[n - k]) * Integer::from(k);
        }
        // Differentiate psi(lambda*t) = L*expm1(Log(b)*psi(t)).
        let denominator = Complex::with_val_64(prec, &powers[n] - lambda) * Integer::from(n);
        if cnum::is_zero(&denominator) {
            return Err(format!("Poincare resonance at order {n}"));
        }
        inverse[n] = Complex::with_val_64(prec, sum * ln_b) / denominator;
        if !cnum::is_finite(&inverse[n]) {
            return Err(format!("Poincare coefficient {n} exceeded MPFR's range"));
        }
        if cnum::verbose() && (n.is_power_of_two() || n == order) {
            eprintln!("Poincare coefficients: {n}/{order}");
        }
    }
    Ok(inverse)
}

fn eval_poincare_series(
    coefficients: &[Complex],
    coordinate: &Complex,
    radius: &Float,
    prec: u64,
) -> Result<(Complex, Complex), String> {
    let magnitude = cnum::abs(coordinate, prec);
    if !magnitude.is_finite() || !radius.is_finite() || magnitude >= *radius || *radius <= 0 {
        return Err("Poincare coordinate is outside its analytic inverse disk".into());
    }
    if coefficients.len() < 2 {
        return Err("Poincare inverse requires a linear coefficient".into());
    }
    let mut value = coefficients
        .last()
        .ok_or("Poincare series has no coefficients")?
        .clone();
    let mut derivative = cnum::zero(prec);
    let rounding_unit = Float::with_val_64(prec, 16)
        >> usize::try_from(prec).map_err(|_| "Poincare precision is not addressable")?;
    let mut roundoff = cnum::abs(&value, prec) * &rounding_unit;
    for coefficient in coefficients[..coefficients.len() - 1].iter().rev() {
        derivative = Complex::with_val_64(prec, derivative * coordinate) + &value;
        let product = Complex::with_val_64(prec, value * coordinate);
        roundoff = roundoff * &magnitude
            + (cnum::abs(&product, prec) + cnum::abs(coefficient, prec)) * &rounding_unit;
        value = product + coefficient;
    }
    if !cnum::is_finite(&value) || !cnum::is_finite(&derivative) || !roundoff.is_finite() {
        return Err("Poincare evaluation exceeded MPFR's range".into());
    }
    if magnitude.is_zero() {
        return Ok((value, derivative));
    }
    let goal = cnum::working_epsilon(prec).ln() + cnum::log_magnitude(&value, prec);
    let ratio = Float::with_val_64(prec, &magnitude / radius);
    let log_radius = radius.clone().ln();
    let log_ratio = cnum::log_magnitude(coordinate, prec) - &log_radius;
    let log_tail = log_radius
        + Float::with_val_64(prec, 2).ln()
        + log_ratio * Integer::from(coefficients.len())
        - (Float::with_val_64(prec, 1) - ratio).ln();
    if log_tail > goal || roundoff.ln() > goal {
        return Err("Poincare inverse tail or roundoff estimate misses working accuracy".into());
    }
    Ok((value, derivative))
}

fn setup_poincare(b: &Complex, fp: &FixedPointData, prec: u64) -> Result<SchroderState, String> {
    let gap = Float::with_val_64(prec, 1) - &fp.lambda_abs;
    let gain = -gap.ln() * 4 + Float::with_val_64(prec, prec).ln() * 2;
    let work_prec =
        cnum::conditioned_precision(&gain, &cnum::working_epsilon(prec), prec)?.unwrap_or(prec);
    let fp_prec = work_prec
        .checked_add(32)
        .ok_or("Poincare fixed-point precision overflow")?;
    cnum::check_precision(fp_prec)?;
    let ln_b = cnum::ln_complex(b, fp_prec);
    let seed = Complex::with_val_64(fp_prec, &fp.fixed_point);
    let l = crate::kouznetsov::newton_fixed_point(&ln_b, &seed, fp_prec)?;
    let l = Complex::with_val_64(work_prec, l);
    let ln_b = cnum::ln_complex(b, work_prec);
    let lambda = Complex::with_val_64(work_prec, &l * &ln_b);
    let lam_abs = Float::with_val_round_64(work_prec, lambda.abs_ref(), Round::Up).0;
    if !(lam_abs > 0 && lam_abs < 1) {
        return Err(
            "Poincare construction requires a resolved strictly attracting fixed point".into(),
        );
    }
    let log_size = Float::with_val_round_64(work_prec, ln_b.abs_ref(), Round::Up).0;
    let gap = Float::with_val_round_64(
        work_prec,
        &Float::with_val_64(work_prec, 1) - &lam_abs,
        Round::Down,
    )
    .0;
    let denominator = log_size * 16;
    let radius = Float::with_val_round_64(work_prec, &gap / &denominator, Round::Down).0;
    if !radius.is_finite() || radius <= 0 {
        return Err("Poincare inverse disk is not representable".into());
    }
    if cnum::verbose() {
        eprintln!(
            "Poincare setup: {work_prec} internal bits, inverse radius {}",
            DisplayFloat(&radius)
        );
    }
    let target = Float::with_val_64(work_prec, &radius / 64);
    let mut point = cnum::one(work_prec);
    let mut power = cnum::one(work_prec);
    let mut shifts = Integer::new();
    let mut checkpoint = point.clone();
    let displacement = loop {
        let displacement = Complex::with_val_64(work_prec, &point - &l);
        if cnum::abs(&displacement, work_prec) <= target {
            if cnum::is_zero(&displacement) {
                return Err("Poincare normalization lost its nonzero displacement".into());
            }
            break displacement;
        }
        point = cnum::checked_exp(&Complex::with_val_64(work_prec, &point * &ln_b), work_prec)?;
        power = Complex::with_val_64(work_prec, power * &lambda);
        shifts += 1;
        if !cnum::is_finite(&power) || cnum::is_zero(&power) {
            return Err("Poincare normalization multiplier exceeded MPFR's range".into());
        }
        if point == checkpoint {
            return Err(
                "Poincare normalization orbit repeated before reaching its local disk".into(),
            );
        }
        if shifts.is_power_of_two() {
            checkpoint = point.clone();
            if cnum::verbose() {
                eprintln!("Poincare normalization orbit: {shifts} steps");
            }
        }
    };
    // For |t| <= 2*|displacement|, q=2*|displacement|/radius <= 1/32
    // bounds the relative tail by 8*q^N. Reserve a factor16 for inversion.
    let log_ratio = cnum::log_magnitude(&displacement, work_prec)
        + Float::with_val_64(work_prec, 2).ln()
        - radius.clone().ln();
    let order = ((cnum::working_epsilon(work_prec).ln() - Float::with_val_64(work_prec, 128).ln())
        / log_ratio)
        .ceil()
        .max(&Float::with_val_64(work_prec, 1))
        .to_integer()
        .and_then(|value| value.to_usize())
        .ok_or("Poincare series order exceeds addressable memory")?;
    let inverse = poincare_coefficients(&ln_b, &lambda, work_prec, order)?;
    let mut coordinate = displacement.clone();
    let local_goal = cnum::working_epsilon(work_prec) * cnum::abs(&displacement, work_prec);
    loop {
        let (value, derivative) = eval_poincare_series(&inverse, &coordinate, &radius, work_prec)?;
        let residual = Complex::with_val_64(work_prec, value - &displacement);
        let error = cnum::abs(&residual, work_prec);
        if error <= local_goal {
            break;
        }
        if cnum::is_zero(&derivative) {
            return Err("Poincare normalization inverse has a zero derivative".into());
        }
        let correction = Complex::with_val_64(work_prec, residual / derivative);
        let mut damping = Float::with_val_64(work_prec, 1);
        loop {
            let step = Complex::with_val_64(work_prec, &correction * &damping);
            let candidate = Complex::with_val_64(work_prec, &coordinate - step);
            if candidate == coordinate {
                return Err("Poincare normalization stalled at working precision".into());
            }
            if let Ok((value, _)) = eval_poincare_series(&inverse, &candidate, &radius, work_prec) {
                if cnum::abs(
                    &Complex::with_val_64(work_prec, value - &displacement),
                    work_prec,
                ) < error
                {
                    coordinate = candidate;
                    break;
                }
            }
            damping /= 2;
        }
    }
    let mut state = SchroderState {
        base: b.clone(),
        l,
        ln_lambda: cnum::ln_complex(&lambda, work_prec),
        ln_b,
        lam_abs,
        sigma_inv: inverse,
        s1: Complex::with_val_64(work_prec, coordinate / power),
        safe_radius: Float::with_val_64(work_prec, &radius / 4),
        sigma_inner_radius: Float::with_val_64(work_prec, &radius / 4),
        inverse_radius: Some(radius),
        prec: work_prec,
    };
    let anchor = eval_schroder_raw(&state, &cnum::zero(work_prec))?;
    let anchor_error = cnum::abs(&Complex::with_val_64(work_prec, anchor - 1), work_prec);
    if !anchor_error.is_finite() || anchor_error > cnum::working_epsilon(prec) {
        return Err(format!(
            "Poincare normalization failed: |F(0)-1|={}",
            DisplayFloat(&anchor_error)
        ));
    }
    state.l = Complex::with_val_64(prec, state.l);
    state.ln_lambda = Complex::with_val_64(prec, state.ln_lambda);
    state.ln_b = cnum::ln_complex(b, prec);
    state.lam_abs = Float::with_val_64(prec, state.lam_abs);
    state.s1 = Complex::with_val_64(prec, state.s1);
    state.sigma_inv = state
        .sigma_inv
        .into_iter()
        .map(|coefficient| Complex::with_val_64(prec, coefficient))
        .collect();
    state.safe_radius = Float::with_val_round_64(prec, state.safe_radius, Round::Down).0;
    state.sigma_inner_radius =
        Float::with_val_round_64(prec, state.sigma_inner_radius, Round::Down).0;
    state.inverse_radius = state
        .inverse_radius
        .map(|radius| Float::with_val_round_64(prec, radius, Round::Down).0);
    state.prec = prec;
    Ok(state)
}

/// Evaluate from a cached state, refining when conditioning consumes its
/// original working-precision tolerance.
///
/// Uses `sigma_inner_radius` as the target for the h-shift, which was
/// determined during setup as the radius at which σ̃ actually converged.
/// This ensures σ̃⁻¹ is evaluated well inside its convergence disk.
pub fn eval_schroder(state: &SchroderState, h: &Complex) -> Result<Complex, String> {
    eval_schroder_with_tolerance(state, h, cnum::working_epsilon(state.prec))
}

/// Evaluate at a fixed decimal-digit goal, independent of refined working precision.
pub fn eval_schroder_at_digits(
    state: &SchroderState,
    h: &Complex,
    digits: u64,
) -> Result<Complex, String> {
    cnum::require_precision(state.prec, digits)?;
    eval_schroder_with_tolerance(state, h, cnum::epsilon(digits, state.prec))
}

fn eval_schroder_with_tolerance(
    state: &SchroderState,
    h: &Complex,
    tolerance: Float,
) -> Result<Complex, String> {
    cnum::init_mpfr();
    if !cnum::is_finite(h) {
        return Err("Schröder height must be finite".into());
    }
    if !cnum::is_finite(&state.base)
        || cnum::is_zero(&state.base)
        || cnum::is_one(&state.base)
        || state.ln_b != cnum::ln_complex(&state.base, state.prec)
    {
        return Err("Schröder base does not match a finite nondegenerate cached state".into());
    }
    // Roundoff in F(-1) must not turn log(0) into a finite surrogate.
    if h.imag().is_zero() && h.real().is_integer() && *h.real() <= -2 {
        return Err(format!(
            "integer height {} is undefined for tetration (would require log_b(0) and beyond)",
            DisplayFloat(h.real())
        ));
    }
    if h.imag().is_zero() && *h.real() == -1 {
        return Ok(cnum::zero(state.prec));
    }
    let validation_tolerance = Float::with_val_64(state.prec, &tolerance / 1000);
    let mut refined = None;
    loop {
        let current = refined.as_ref().unwrap_or(state);
        let (value, log_amplification) = eval_schroder_with_conditioning(current, h)?;
        if let Some(prec) =
            cnum::conditioned_precision(&log_amplification, &validation_tolerance, current.prec)?
        {
            if cnum::verbose() {
                eprintln!(
                    "schroder height conditioning: refining from {} to {prec} working bits",
                    current.prec
                );
            }
            let fp_prec = prec
                .checked_add(32)
                .ok_or("Schröder fixed-point guard precision overflow")?;
            cnum::check_precision(fp_prec)?;
            let ln_base = cnum::ln_complex(&current.base, fp_prec);
            let seed = Complex::with_val_64(fp_prec, &current.l);
            let l = crate::kouznetsov::newton_fixed_point(&ln_base, &seed, fp_prec)?;
            let l = Complex::with_val_64(prec, l);
            let ln_base = cnum::ln_complex(&current.base, prec);
            let lambda = Complex::with_val_64(prec, &l * &ln_base);
            let fp = FixedPointData {
                fixed_point: l,
                lambda_abs: cnum::abs(&lambda, prec),
                lambda,
            };
            refined = Some(setup_schroder(&current.base, &fp, prec)?);
            continue;
        }
        validate_functional_equation(current, h, &value, &validation_tolerance, current.prec)?;
        return Ok(value);
    }
}

fn eval_schroder_raw(state: &SchroderState, h: &Complex) -> Result<Complex, String> {
    eval_schroder_with_conditioning(state, h).map(|(value, _)| value)
}

fn eval_inverse(state: &SchroderState, coordinate: &Complex) -> Result<Complex, String> {
    match &state.inverse_radius {
        Some(radius) => eval_poincare_series(&state.sigma_inv, coordinate, radius, state.prec)
            .map(|(value, _)| value),
        None => eval_series_checked(&state.sigma_inv, coordinate, state.prec),
    }
}

fn inverse_sum_conditioning(
    state: &SchroderState,
    height: &Complex,
    coordinate: &Complex,
    inverse: &Complex,
    value: &Complex,
) -> Result<Float, String> {
    let prec = state.prec;
    let mut power = cnum::one(prec);
    let mut sensitivity = Float::new_64(prec);
    for (index, coefficient) in state.sigma_inv.iter().enumerate().skip(1) {
        power = Complex::with_val_64(prec, &power * coordinate);
        let term = Complex::with_val_64(prec, coefficient * &power);
        sensitivity += cnum::abs(&term, prec) * Integer::from(index);
    }
    if !sensitivity.is_finite() {
        return Err("Schröder inverse-series sensitivity is non-finite".into());
    }
    let phase_gain = (cnum::log_magnitude(height, prec)
        + cnum::log_magnitude(&state.ln_lambda, prec))
    .max(&Float::new_64(prec));
    let input_scale = sensitivity.ln() + phase_gain;
    let scale = cnum::log_magnitude(&state.l, prec)
        .max(&cnum::log_magnitude(inverse, prec))
        .max(&input_scale)
        .max(&Float::new_64(prec));
    Ok(
        (scale - cnum::log_magnitude(value, prec).max(&Float::new_64(prec)))
            .max(&Float::new_64(prec)),
    )
}

fn eval_schroder_with_conditioning(
    state: &SchroderState,
    h: &Complex,
) -> Result<(Complex, Float), String> {
    let prec = state.prec;
    let exponent = Complex::with_val_64(prec, h * &state.ln_lambda);
    let log_t_abs = cnum::log_magnitude(&state.s1, prec) + exponent.real();
    let log_lam_abs = state.lam_abs.clone().ln();
    if !log_t_abs.is_finite() || !log_lam_abs.is_finite() || log_lam_abs.is_zero() {
        return Err(
            "Schröder coordinate or multiplier has an invalid logarithmic magnitude".into(),
        );
    }

    // The σ̃⁻¹ convergence radius can be smaller than sigma_inner_radius (which
    // tracks σ̃ in the w-domain, not σ̃⁻¹ in the t-domain). The strategy:
    //   1. Try direct eval_series_checked at |t|. If succeeds, return.
    //   2. Otherwise, find the smallest k such that eval_series_checked succeeds
    //      at |t · λ^k|. Each iteration halves the target.
    //
    // This handles bases near η where σ̃⁻¹ has a small effective radius even
    // when sigma_inner_radius reports otherwise.
    let mut initial_target = state.sigma_inner_radius.clone().min(&state.safe_radius);
    if state
        .inverse_radius
        .as_ref()
        .is_none_or(|radius| log_t_abs < radius.clone().ln())
    {
        match inverse_coordinate(state, h) {
            Ok(t) => {
                initial_target = initial_target.min(&cnum::abs(&t, prec));
                if let Ok(inv_t) = eval_inverse(state, &t) {
                    let value = Complex::with_val_64(prec, &state.l + &inv_t);
                    let log_amplification = inverse_sum_conditioning(state, h, &t, &inv_t, &value)?
                        - cnum::log_magnitude(&value, prec).min(&Float::new_64(prec));
                    return Ok((value, log_amplification));
                }
            }
            Err(error) => {
                if cnum::verbose() {
                    eprintln!("schroder coordinate requires a height shift: {error}");
                }
            }
        }
    }

    let mut effective_target: Float = initial_target / 2;
    let (k, mut f, mut log_amplification) = loop {
        if !effective_target.is_finite() || effective_target <= 0 {
            return Err("Schröder height-shift target exceeded MPFR's exponent range".into());
        }
        let k_raw = (effective_target.clone().ln() - &log_t_abs) / &log_lam_abs;
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
        let t_shifted = inverse_coordinate(state, &h_shifted)?;
        match eval_inverse(state, &t_shifted) {
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
                let log_amplification =
                    inverse_sum_conditioning(state, &h_shifted, &t_shifted, &inv_t_shifted, &f0)?;
                break (k, f0, log_amplification);
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
    let log_ln_b = cnum::log_magnitude(&state.ln_b, prec);
    while shifts > step {
        if cnum::verbose() && (step == 0 || step.is_divisible_u(1024)) {
            eprintln!(
                "schröder unwind: step {step} of {:.8}",
                DisplayFloat(&shifts)
            );
        }
        if forward_log {
            check_chain(&f, &step)?;
            let input_scale = -cnum::log_magnitude(&f, prec).min(&Float::new_64(prec));
            let ln_f = cnum::ln_complex(&f, prec);
            f = Complex::with_val_64(prec, &ln_f / &state.ln_b);
            let amplification =
                input_scale - &log_ln_b - cnum::log_magnitude(&f, prec).max(&Float::new_64(prec));
            log_amplification += amplification.max(&Float::new_64(prec));
        } else {
            let amplification = cnum::log_magnitude(&f, prec).max(&Float::new_64(prec)) + &log_ln_b;
            log_amplification += amplification.max(&Float::new_64(prec));
            let exponent = Complex::with_val_64(prec, &f * &state.ln_b);
            f = cnum::checked_exp(&exponent, prec)?;
            check_chain(&f, &step)?;
        }
        step += 1;
    }
    check_chain(&f, &step)?;
    log_amplification -= cnum::log_magnitude(&f, prec).min(&Float::new_64(prec));
    Ok((f, log_amplification))
}

fn inverse_coordinate(state: &SchroderState, h: &Complex) -> Result<Complex, String> {
    let prec = state.prec;
    if let Ok(power) = lambda_pow(h, &state.ln_lambda, prec) {
        let coordinate = Complex::with_val_64(prec, &state.s1 * power);
        if cnum::is_finite(&coordinate) && !cnum::is_zero(&coordinate) {
            return Ok(coordinate);
        }
    }
    // Combine real scales without taking arg(s1), which would perturb real axes.
    let scale = state
        .s1
        .real()
        .clone()
        .abs()
        .max(&state.s1.imag().clone().abs());
    if !scale.is_finite() || scale <= 0 {
        return Err("Schröder normalization coordinate is zero or non-finite".into());
    }
    let unit = Complex::with_val_64(prec, &state.s1 / &scale);
    let mut exponent = Complex::with_val_64(prec, h * &state.ln_lambda);
    exponent += scale.ln();
    let coordinate = Complex::with_val_64(prec, unit * cnum::checked_exp(&exponent, prec)?);
    if !cnum::is_finite(&coordinate) || cnum::is_zero(&coordinate) {
        return Err("Schröder inverse coordinate exceeded MPFR's exponent range".into());
    }
    Ok(coordinate)
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
    let ln_b = cnum::ln_complex(b, prec);

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

    let ln_b = cnum::ln_complex(b, prec);
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
                    let ln_lpw = cnum::ln_complex(&l_plus_w, prec);
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
    fn inverse_poincare_coefficients_match_classical_reversion() {
        let prec = cnum::digits_to_bits(70);
        for (re, im) in [
            ("1.25", "0"),
            ("0.5", "0"),
            ("1.3", "0.1"),
            ("1.444666", "0"),
            ("0.0665", "0"),
        ] {
            let base = cnum::parse_complex(re, im, prec).unwrap();
            let ln_base = Complex::with_val_64(prec, base.ln_ref());
            let w = crate::lambertw::w0(&Complex::with_val_64(prec, -&ln_base), prec).unwrap();
            let multiplier = Complex::with_val_64(prec, -w);
            let (_, classical) = build_series(&base, &multiplier, prec, 32).unwrap();
            let inverse = poincare_coefficients(&ln_base, &multiplier, prec, 32).unwrap();
            for (index, (actual, expected)) in inverse.iter().zip(&classical).enumerate() {
                let error = cnum::abs(&Complex::with_val_64(prec, actual - expected), prec);
                let scale = cnum::abs(expected, prec).max(&Float::with_val_64(prec, 1));
                assert!(
                    error < cnum::epsilon(60, prec) * scale,
                    "{re}+{im}i coefficient {index}: {error}"
                );
            }
        }
    }

    #[test]
    fn poincare_series_requires_analytic_disk_and_small_tail() {
        let prec = cnum::digits_to_bits(70);
        let radius = Float::with_val_64(prec, 1);
        let mut coefficients = vec![cnum::zero(prec); 513];
        coefficients[1] = cnum::one(prec);
        let coordinate = Complex::with_val_64(prec, cnum::decimal("0.25", prec));
        let (value, derivative) =
            eval_poincare_series(&coefficients, &coordinate, &radius, prec).unwrap();
        assert_eq!(value, coordinate);
        assert_eq!(derivative, cnum::one(prec));
        assert!(eval_poincare_series(&coefficients[..2], &coordinate, &radius, prec).is_err());
        assert!(eval_poincare_series(&coefficients, &cnum::one(prec), &radius, prec).is_err());
        assert!(eval_poincare_series(&coefficients[..1], &coordinate, &radius, prec).is_err());
        for invalid in [rug::float::Special::Nan, rug::float::Special::Infinity] {
            assert!(eval_poincare_series(
                &coefficients,
                &coordinate,
                &Float::with_val_64(prec, invalid),
                prec
            )
            .is_err());
        }
        assert!(poincare_coefficients(
            &cnum::one(prec),
            &Complex::with_val_64(prec, cnum::decimal("0.5", prec)),
            prec,
            usize::MAX
        )
        .unwrap_err()
        .contains("addressable"));
    }

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
