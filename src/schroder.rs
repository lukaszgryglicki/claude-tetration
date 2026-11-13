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

use rug::{
    float::{Constant, Round},
    ops::Pow,
    Complex, Float, Integer,
};

use crate::{cnum, regions::FixedPointData, schroder_jumps::OrbitJumps};
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
    /// With a long series (`LongSeries`): `inverse_radius` is the radius of
    /// the largest disk on which the inverse germ is analytic and one-to-one,
    /// and the series is long enough for `safe_radius`.
    pub univalent: bool,
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
        univalent: false,
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

/// `psi(t)` and `psi'(t)` on the disk of radius `radius`. With `univalent`,
/// `psi` is one-to-one on that disk and the tail follows from
/// `|c_n| <= n radius^(1-n)` (de Branges); otherwise `radius` is the
/// conservative disk of `setup_poincare`.
pub(crate) fn eval_poincare_series(
    coefficients: &[Complex],
    coordinate: &Complex,
    radius: &Float,
    univalent: bool,
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
    let terms = Integer::from(coefficients.len());
    let margin = (Float::with_val_64(prec, 1) - ratio).ln();
    let log_tail = if univalent {
        log_radius + Float::with_val_64(prec, &terms).ln() + log_ratio * terms - margin * 2u32
    } else {
        log_radius + Float::with_val_64(prec, 2).ln() + log_ratio * terms - margin
    };
    if log_tail > goal || roundoff.ln() > goal {
        return Err("Poincare inverse tail or roundoff estimate misses working accuracy".into());
    }
    Ok((value, derivative))
}

/// A normalization orbit of at least this many steps makes the long inverse
/// series worth its coefficients.
const LONG_SERIES_STEPS: u32 = 256;
/// Next to `lambda = 1` Taylor jumps are cheaper than a series longer than
/// this, next to `lambda = -1` than one eight times as long.
const LONG_SERIES_TERMS: u32 = 4096;

/// The plan of a long inverse series for a nearly neutral multiplier.
///
/// `psi(t / lambda) = log_b(L + psi(t)) - L` continues the inverse germ to
/// the disk whose boundary holds the coordinate of the height -2: `L + psi`
/// vanishes only at the coordinate of the height -1. On that disk `psi` is
/// one-to-one, the Schroder function of the basin being its inverse, so its
/// coefficients obey `|c_n| <= n R^(1-n)` (de Branges). That bounds the tail
/// of the series on `|t| <= R e^(-x)`, where about `nats / x` terms replace
/// all but `x / |ln lambda|` logarithms of a height reduction.
struct LongSeries {
    x: Float,
    /// Guard of the longer coefficient recursion, in nats.
    gain: Float,
}

impl LongSeries {
    fn plan(b: &Complex, fp: &FixedPointData, prec: u64) -> Result<Option<LongSeries>, String> {
        let gap = Float::with_val_64(prec, 1) - &fp.lambda_abs;
        let rate = -Float::with_val_64(prec, fp.lambda_abs.ln_ref());
        if !gap.is_finite() || gap <= 0 || !rate.is_finite() || rate <= 0 {
            return Ok(None);
        }
        let ln_b = cnum::ln_complex(b, prec);
        let short = Float::with_val_64(prec, &gap / (cnum::abs(&ln_b, prec) * 16u32));
        let start = cnum::abs(&Complex::with_val_64(prec, &fp.fixed_point - 1u32), prec);
        let steps = (start * 64u32 / short).ln() / &rate;
        if !steps.is_finite() || steps < LONG_SERIES_STEPS {
            return Ok(None);
        }
        // Taylor jumps cross the whole orbit next to `lambda = 1`; next to
        // `lambda = -1` a tail of single steps remains, and a turning
        // multiplier leaves them only the end of the orbit.
        let l = Complex::with_val_64(prec, &fp.fixed_point);
        let lambda = Complex::with_val_64(prec, &l * &ln_b);
        let near = |distance: Complex| cnum::abs(&distance, prec) * 32u32 <= 1u32;
        let limit = if near(Complex::with_val_64(prec, &lambda - 1u32)) {
            Some(LONG_SERIES_TERMS)
        } else if near(Complex::with_val_64(prec, &lambda + 1u32)) {
            Some(LONG_SERIES_TERMS * 8)
        } else {
            None
        };
        let mut jumps = false;
        for logarithm in [false, true] {
            match OrbitJumps::new(&ln_b, &l, logarithm, prec)? {
                Some(OrbitJumps::Real(_)) => return Ok(None),
                Some(OrbitJumps::Complex(_)) => jumps = true,
                None => {}
            }
        }
        // The cost of a call, in products: `K^2 / 2` for `K = nats / x`
        // coefficients, the series at about a hundred samples of each of its
        // three logarithm chains, and `x / rate` logarithms of each sample.
        // The weights affect speed only.
        let nats = Float::with_val_64(prec, prec) * Float::with_val_64(prec, 2).ln()
            - Float::with_val_64(prec, gap.ln_ref()) * 4u32;
        let orbit = Float::with_val_64(prec, 16384) / &rate;
        let series = Float::with_val_64(prec, 2048) * &nats;
        let setup = nats.clone().square();
        let x = cnum::cost_balance(&orbit, &series, &setup);
        if !x.is_finite() || x <= 0 {
            return Ok(None);
        }
        let terms = Float::with_val_64(prec, &nats / &x) * 2u32;
        if jumps && limit.is_some_and(|limit| terms > limit) {
            return Ok(None);
        }
        let gain = terms.ln() * 4u32
            - Float::with_val_64(prec, x.ln_ref()).min(&Float::new_64(prec)) * 3u32;
        Ok(Some(LongSeries { x, gain }))
    }

    /// The number of terms whose tail stays below the working resolution on
    /// `|t| <= R e^(-x)`.
    fn order(&self, work: u64) -> Result<usize, String> {
        let x = Float::with_val_64(work, &self.x);
        let ratio = Float::with_val_64(work, -&x).exp();
        // `(N + 1) q^N <= eps (1 - q)^2 / 256` puts the de Branges tail below
        // `eps / 64` of the smallest value of `psi` on `|t| = q R`.
        let base = Float::with_val_64(work, 256).ln()
            - cnum::working_epsilon(work).ln()
            - (Float::with_val_64(work, 1) - ratio).ln() * 2u32;
        let mut order = Float::with_val_64(work, &base / &x).ceil();
        loop {
            if !order.is_finite() {
                return Err("its length is not finite".into());
            }
            let next = ((order.clone() + 1u32).ln() + &base) / &x;
            let next = next.ceil();
            if next <= order {
                break;
            }
            order = next;
        }
        order
            .to_integer()
            .and_then(|order| order.to_usize())
            .ok_or_else(|| "its length is not addressable".into())
    }

    /// The coefficients, the radius of the disk of `psi` and the radius on
    /// which the series is summed.
    #[allow(clippy::too_many_arguments)]
    fn build(
        &self,
        ln_b: &Complex,
        l: &Complex,
        lambda: &Complex,
        lam_abs: &Float,
        s1: &Complex,
        short: &Float,
        prec: u64,
        work: u64,
    ) -> Result<(Vec<Complex>, Float, Float), String> {
        let size = Float::with_val_round_64(work, s1.abs_ref(), Round::Down).0;
        let square = Float::with_val_round_64(work, lam_abs.square_ref(), Round::Up).0;
        let disk = Float::with_val_round_64(work, &size / &square, Round::Down).0;
        let ratio = Float::with_val_64(work, -&self.x).exp();
        let reach = Float::with_val_round_64(work, &disk * &ratio, Round::Down).0;
        if !reach.is_finite() || reach <= Float::with_val_64(work, short * 4u32) {
            return Err("it reaches no further than the short one".into());
        }
        let order = self.order(work)?;
        if cnum::verbose() {
            eprintln!(
                "Poincare setup: long series of {order} terms on {} of its disk of radius {}",
                DisplayFloat(&ratio),
                DisplayFloat(&disk)
            );
        }
        let inverse = poincare_long_coefficients(ln_b, lambda, work, order)?;
        // The coefficients are as good as the functional equation of their
        // series where it is summed.
        let tolerance = cnum::working_epsilon(prec) >> 10u32;
        let pi = Float::with_val_64(work, Constant::Pi);
        let unit = Complex::with_val_64(work, s1 / &size);
        for part in 0..8u32 {
            let angle = Float::with_val_64(work, &pi * part) / 4u32;
            let phase = Complex::with_val_64(work, (angle.cos_ref(), angle.sin_ref()));
            let outer = Complex::with_val_64(work, &unit * phase) * &reach;
            let inner = Complex::with_val_64(work, &outer * lambda);
            let (value, _) = eval_poincare_series(&inverse, &outer, &disk, true, work)?;
            let (image, _) = eval_poincare_series(&inverse, &inner, &disk, true, work)?;
            let exponent = Complex::with_val_64(work, value * ln_b);
            let expected =
                Complex::with_val_64(work, cnum::checked_exp(&exponent, work)? - 1u32) * l;
            let miss = cnum::abs(&Complex::with_val_64(work, &image - expected), work);
            if !miss.is_finite()
                || miss > Float::with_val_64(work, &tolerance * cnum::abs(&image, work))
            {
                return Err("its functional equation fails where it is summed".into());
            }
        }
        Ok((inverse, disk, reach))
    }
}

/// `poincare_coefficients` with one product per term of the convolution.
fn poincare_long_coefficients(
    ln_b: &Complex,
    lambda: &Complex,
    prec: u64,
    order: usize,
) -> Result<Vec<Complex>, String> {
    cnum::check_complex_storage((order as u128 + 1) * 3, prec)?;
    let mut inverse = vec![cnum::zero(prec), cnum::one(prec)];
    // weighted[k] = k c_k, scaled[k] = c_k lambda^k.
    let mut weighted = inverse.clone();
    let mut scaled = vec![cnum::zero(prec), lambda.clone()];
    let mut power = lambda.clone();
    for n in 2..=order {
        power *= lambda;
        let mut sum = cnum::zero(prec);
        for k in 1..n {
            sum += &weighted[k] * &scaled[n - k];
        }
        let denominator = Complex::with_val_64(prec, &power - lambda) * Integer::from(n);
        if cnum::is_zero(&denominator) || cnum::is_zero(&power) {
            return Err(format!("Poincare resonance at order {n}"));
        }
        let coefficient = Complex::with_val_64(prec, sum * ln_b) / denominator;
        if !cnum::is_finite(&coefficient) {
            return Err(format!("Poincare coefficient {n} exceeded MPFR's range"));
        }
        if cnum::verbose() && (n.is_power_of_two() || n == order) {
            eprintln!("Poincare coefficients: {n}/{order}");
        }
        weighted.push(Complex::with_val_64(prec, &coefficient * Integer::from(n)));
        scaled.push(Complex::with_val_64(prec, &coefficient * &power));
        inverse.push(coefficient);
    }
    Ok(inverse)
}

fn setup_poincare(b: &Complex, fp: &FixedPointData, prec: u64) -> Result<SchroderState, String> {
    if let Some(plan) = LongSeries::plan(b, fp, prec)? {
        match setup_poincare_with(b, fp, prec, Some(&plan)) {
            Ok(state) => return Ok(state),
            Err(reason) => {
                if cnum::verbose() {
                    eprintln!("Poincare setup: long series unavailable ({reason})");
                }
            }
        }
    }
    setup_poincare_with(b, fp, prec, None)
}

fn setup_poincare_with(
    b: &Complex,
    fp: &FixedPointData,
    prec: u64,
    plan: Option<&LongSeries>,
) -> Result<SchroderState, String> {
    let gap = Float::with_val_64(prec, 1) - &fp.lambda_abs;
    let mut gain = -gap.ln() * 4 + Float::with_val_64(prec, prec).ln() * 2;
    if let Some(plan) = plan {
        gain += &plan.gain;
    }
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
    let mut jumps = OrbitJumps::new(&ln_b, &l, false, work_prec)?;
    let mut jump_count = Integer::new();
    let mut point = cnum::one(work_prec);
    let mut power = cnum::one(work_prec);
    let mut shifts = Integer::new();
    let mut orbit_updates = Integer::new();
    let mut checkpoint = point.clone();
    let displacement = loop {
        let displacement = Complex::with_val_64(work_prec, &point - &l);
        if cnum::abs(&displacement, work_prec) <= target {
            if cnum::is_zero(&displacement) {
                return Err("Poincare normalization lost its nonzero displacement".into());
            }
            break displacement;
        }
        let jump = match &mut jumps {
            Some(cache) => cache.advance(&point, None)?,
            None => None,
        };
        if let Some(jump) = jump {
            point = jump.value;
            if lambda.imag().is_zero() {
                power *= Float::with_val_64(work_prec, lambda.real()).pow(&jump.steps);
            } else {
                power *= lambda.clone().pow(&jump.steps);
            }
            shifts += jump.steps;
            jump_count += 1;
            if cnum::verbose() && jump_count.is_power_of_two() {
                eprintln!("Poincare normalization: {jump_count} Taylor jumps, {shifts} steps");
            }
        } else {
            point = cnum::checked_exp(&Complex::with_val_64(work_prec, &point * &ln_b), work_prec)?;
            power = Complex::with_val_64(work_prec, power * &lambda);
            shifts += 1;
        }
        if !cnum::is_finite(&power) || cnum::is_zero(&power) {
            return Err("Poincare normalization multiplier exceeded MPFR's range".into());
        }
        orbit_updates += 1;
        if point == checkpoint {
            return Err(
                "Poincare normalization orbit repeated before reaching its local disk".into(),
            );
        }
        if orbit_updates.is_power_of_two() {
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
        let (value, derivative) =
            eval_poincare_series(&inverse, &coordinate, &radius, false, work_prec)?;
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
            if let Ok((value, _)) =
                eval_poincare_series(&inverse, &candidate, &radius, false, work_prec)
            {
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
    let s1 = Complex::with_val_64(work_prec, coordinate / power);
    let (inverse, radius, target) = match plan {
        Some(plan) => plan.build(&ln_b, &l, &lambda, &lam_abs, &s1, &radius, prec, work_prec)?,
        None => {
            let target = Float::with_val_64(work_prec, &radius / 4);
            (inverse, radius, target)
        }
    };
    let mut state = SchroderState {
        base: b.clone(),
        l,
        ln_lambda: cnum::ln_complex(&lambda, work_prec),
        ln_b,
        lam_abs,
        sigma_inv: inverse,
        s1,
        safe_radius: target.clone(),
        sigma_inner_radius: target,
        inverse_radius: Some(radius),
        univalent: plan.is_some(),
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
    if cnum::is_zero(h) {
        return Ok(cnum::one(state.prec));
    }
    if cnum::is_one(h) {
        return Ok(Complex::with_val_64(state.prec, &state.base));
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
        Some(radius) => eval_poincare_series(
            &state.sigma_inv,
            coordinate,
            radius,
            state.univalent,
            state.prec,
        )
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

fn component_conditioning(value: &Complex, prec: u64) -> Float {
    let magnitude = cnum::log_magnitude(value, prec);
    let smallest = [value.real(), value.imag()]
        .into_iter()
        .filter(|component| !component.is_zero())
        .map(|component| component.clone().abs().ln())
        .fold(magnitude.clone(), |scale, component| scale.min(&component));
    (magnitude - smallest).max(&Float::new_64(prec))
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
    // A long series is summed up to its target, a short one inside its disk.
    let direct = if state.univalent {
        log_t_abs <= initial_target.clone().ln()
    } else {
        state
            .inverse_radius
            .as_ref()
            .is_none_or(|radius| log_t_abs < radius.clone().ln())
    };
    if direct {
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

    let mut effective_target: Float = if state.univalent {
        initial_target
    } else {
        initial_target / 2
    };
    let (k, f, log_amplification) = loop {
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
        let t_shifted = if has_real_negative_multiplier(state) {
            real_negative_coordinate(state, h, &k)?
        } else {
            inverse_coordinate(state, &h_shifted)?
        };
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
    let shifts = k.clone().abs();
    let log_ln_b = cnum::log_magnitude(&state.ln_b, prec);
    // A real base with a real fixed point needs no tracing. Its inverse germ
    // has real coefficients, so the image of the disk of that germ is
    // symmetric about the real axis, and `b^z` is one-to-one on that image:
    // it holds no two conjugate points `2 pi / |ln b|` apart and, being
    // connected, lies in `|Im z| < pi / |ln b|`. Every logarithm that the
    // tracing would fix is therefore the principal one.
    let real_base = state.ln_b.imag().is_zero() && state.l.imag().is_zero();
    // `F(h + k + offset)` and `|dF/ds|` there, at the top of the chain.
    let top = |offset: &Float| -> Result<Sample, String> {
        let mut height = Complex::with_val_64(prec, h + offset);
        if height.imag().is_zero() {
            // A sample within the rounding of that sum of an integer height
            // is taken there: a real base has real values at those heights,
            // and its Taylor jumps need them exactly real.
            let whole = height.real().clone().round();
            let scale = whole.clone().abs().max(&Float::with_val_64(prec, 1));
            let shift =
                usize::try_from(prec.saturating_sub(8)).map_err(|_| "precision overflow")?;
            if Float::with_val_64(prec, &whole - height.real()).abs() <= scale >> shift {
                height = Complex::with_val_64(prec, whole);
            }
        }
        let coordinate = if has_real_negative_multiplier(state) {
            real_negative_coordinate(state, &height, &k)?
        } else {
            inverse_coordinate(state, &Complex::with_val_64(prec, &height + &k))?
        };
        match &state.inverse_radius {
            Some(radius) => {
                let (displacement, slope) = eval_poincare_series(
                    &state.sigma_inv,
                    &coordinate,
                    radius,
                    state.univalent,
                    prec,
                )?;
                let rate = Complex::with_val_64(prec, slope * &coordinate) * &state.ln_lambda;
                Ok((
                    Complex::with_val_64(prec, &state.l + displacement),
                    Some(cnum::abs(&rate, prec)),
                ))
            }
            None => {
                let displacement = eval_series_checked(&state.sigma_inv, &coordinate, prec)?;
                Ok((Complex::with_val_64(prec, &state.l + displacement), None))
            }
        }
    };
    let traced = forward_log && !real_base;
    let start = (f, log_amplification);
    let start_speed = if traced {
        top(&Float::new_64(prec))?.1
    } else {
        None
    };
    let mut offsets = Trail::offsets(state);
    let mut anchor = if traced { Anchor::new(h, prec) } else { None };
    let mut own = None;
    if let Some(anchor) = &anchor {
        if !offsets.contains(&anchor.offset) {
            offsets.push(anchor.offset.clone());
            offsets.sort_by(|a, b| a.partial_cmp(b).expect("finite trail offsets"));
            own = Some(anchor.offset.clone());
        }
    }
    // The integer tower checks every branch of a real height above -1, the
    // final value included (see `Anchor`): the distances of neighbouring
    // samples are then judged only once it has refuted a chain. Other heights
    // depend on that judgement.
    let mut judged = anchor.as_ref().is_none_or(|anchor| anchor.ceiling < 0);
    loop {
        let (mut f, mut log_amplification) = start.clone();
        let mut speed = start_speed.clone();
        let mut step = Integer::new();
        // A long series starts its chains far from the fixed point.
        let mut jumps = if state.univalent {
            None
        } else {
            OrbitJumps::new(&state.ln_b, &state.l, forward_log, prec)?
        };
        let mut jump_count = Integer::new();
        let mut updates = Integer::new();
        let mut jump_error_log = Float::new_64(prec);
        let mut recurrence = cnum::Recurrence::<()>::new();
        let mut trail: Option<Trail> = None;
        if traced && jumps.is_some() {
            // The samples follow every Taylor jump from the top.
            let built = Trail::new(&offsets, &top)?;
            if let Some(anchor) = &mut anchor {
                let left = shifts
                    .to_integer()
                    .ok_or("Schröder height shift is not finite")?;
                if anchor.holds(&built, &left, state)? == Some(false) {
                    return Err(Anchor::SERIES.into());
                }
            }
            trail = Some(built);
        }
        let mut missing: Vec<Float> = Vec::new();
        // An error made at one step grows by the product of the later
        // factors: `net` is the logarithm of the product of all factors so
        // far, `lowest` its smallest value on the way, `expanding` the sum of
        // the factors above 1 alone.
        let mut net = Float::new_64(prec);
        let mut lowest = Float::new_64(prec);
        let mut expanding = Float::new_64(prec);
        // The chain is repeated once samples are missing: go on collecting
        // them for as many steps again, no further.
        let mut flagged: Option<Integer> = None;
        while shifts > step {
            if !missing.is_empty() {
                let since = flagged.get_or_insert_with(|| step.clone());
                if step > Integer::from(&*since * 2u32) + 64u32 {
                    break;
                }
            }
            if cnum::verbose() && (step == 0 || step.is_divisible_u(1024)) {
                eprintln!(
                    "schröder unwind: step {step} of {:.8}",
                    DisplayFloat(&shifts)
                );
            }
            let mut jump = match &mut jumps {
                Some(cache) => {
                    let remaining = Float::with_val_round_64(prec, &shifts - &step, Round::Down).0;
                    cache.advance(&f, Some(&remaining))?
                }
                None => None,
            };
            if trail.is_some() {
                // The samples have to make the jump as well. Where one of
                // them cannot, shorter jumps are tried, and after the
                // shortest the chain goes on in single steps.
                let mut refused = false;
                while let Some(candidate) = &jump {
                    let mut followed = trail.clone();
                    let made = followed.as_mut().is_some_and(|samples| {
                        samples.descend(&candidate.steps, &mut jumps, state).is_ok()
                    });
                    if made {
                        trail = followed;
                        break;
                    }
                    refused = true;
                    let shorter = Float::with_val_64(prec, Integer::from(&candidate.steps >> 1u32));
                    jump = match &mut jumps {
                        Some(cache) => cache.advance(&f, Some(&shorter))?,
                        None => None,
                    };
                }
                if refused && jump.is_none() {
                    jumps = None;
                }
            }
            if let Some(jump) = jump {
                check_chain(&f, &step)?;
                let input_scale = cnum::log_magnitude(&f, prec).max(&Float::new_64(prec));
                f = jump.value;
                check_chain(&f, &step)?;
                let output_scale = cnum::log_magnitude(&f, prec).max(&Float::new_64(prec));
                let amplification = jump.derivative.clone().ln() + input_scale - &output_scale;
                net += &amplification;
                lowest = lowest.min(&net);
                expanding += amplification.max(&Float::new_64(prec));
                let error_log = jump.error.ln() - cnum::working_epsilon(prec).ln() - output_scale;
                jump_error_log = jump_error_log.max(&error_log);
                if let Some(speed) = &mut speed {
                    *speed *= &jump.derivative;
                }
                step += jump.steps;
                jump_count += 1;
                recurrence.reset();
                if cnum::verbose() && jump_count.is_power_of_two() {
                    eprintln!("schroder unwind: {jump_count} Taylor jumps, {step} steps");
                }
            } else if forward_log {
                check_chain(&f, &step)?;
                let input_scale = -cnum::log_magnitude(&f, prec).min(&Float::new_64(prec));
                let ln_f = cnum::ln_complex(&f, prec);
                let mut next = Complex::with_val_64(prec, &ln_f / &state.ln_b);
                // Where |lambda^(h+2)| < 1 the Schroder coordinate of the height
                // lies inside the first singularity of the inverse: the
                // tetration is single-valued there, and continuity along the
                // real shifts of the height fixes the branch. Outside, the
                // logarithms are the principal ones.
                let beyond = Complex::with_val_64(prec, h + &shifts) - &step + 1u32;
                let inside = Complex::with_val_64(prec, beyond * &state.ln_lambda)
                    .real()
                    .is_sign_negative();
                let continuous = traced && inside;
                if continuous {
                    let left = (shifts.clone() - &step)
                        .to_integer()
                        .ok_or("Schröder height shift is not finite")?;
                    if trail.is_none() {
                        let mut built = Trail::new(&offsets, &top)?;
                        built.descend(&step, &mut jumps, state)?;
                        if let Some(anchor) = &mut anchor {
                            if anchor.holds(&built, &left, state)? == Some(false) {
                                return Err(Anchor::SERIES.into());
                            }
                        }
                        trail = Some(built);
                    }
                    if let Some(samples) = &mut trail {
                        samples.step(&f, &speed, &mut next, &state.ln_b, judged, &mut missing)?;
                        if let Some(anchor) = &mut anchor {
                            let left = Integer::from(&left - 1u32);
                            if anchor.holds(samples, &left, state)? == Some(false) {
                                anchor.refute(&step, samples, &mut missing)?;
                                judged = true;
                                break;
                            }
                        }
                    }
                    if let Some(speed) = &mut speed {
                        *speed /= cnum::abs(&f, prec) * cnum::abs(&state.ln_b, prec);
                    }
                } else {
                    trail = None;
                }
                let amplification = input_scale
                    - &log_ln_b
                    - cnum::log_magnitude(&next, prec).max(&Float::new_64(prec));
                net += &amplification;
                lowest = lowest.min(&net);
                expanding += amplification.max(&Float::new_64(prec));
                step += 1;
                let remaining = Float::with_val_64(prec, &shifts - &step)
                    .to_integer()
                    .ok_or("Schröder height shift is not finite")?;
                if !continuous {
                    if let Some((period, ())) = recurrence.step(&next, || ()) {
                        // The logarithms repeat exactly: whole periods change nothing.
                        step += Integer::from(&remaining / &period) * &period;
                    }
                    if remaining > 0 && state.ln_b.imag().is_zero() && next.real() == f.real() {
                        if let Some((settled, growth)) =
                            settle_imaginary(&next, state.ln_b.real(), &remaining, prec)?
                        {
                            // Once the real part is stationary and the imaginary
                            // part is below its resolution, each logarithm divides
                            // the latter by `Re(f) ln b`.
                            net += &growth;
                            expanding += growth;
                            f = settled;
                            step += &remaining;
                            break;
                        }
                    }
                }
                f = next;
            } else {
                let amplification =
                    cnum::log_magnitude(&f, prec).max(&Float::new_64(prec)) + &log_ln_b;
                net += &amplification;
                lowest = lowest.min(&net);
                expanding += amplification.max(&Float::new_64(prec));
                let exponent = Complex::with_val_64(prec, &f * &state.ln_b);
                f = cnum::checked_exp(&exponent, prec)?;
                check_chain(&f, &step)?;
                step += 1;
            }
            updates += 1;
        }
        if missing.is_empty() {
            // One level further down, the samples from the anchor up are
            // continued from the final value itself: the tower value there
            // confirms its branch. Where that cannot be checked, the samples
            // have to be judged.
            let verdict = match (&mut anchor, &trail) {
                (Some(anchor), Some(samples)) if anchor.ceiling >= 0 => {
                    let verdict = anchor.confirms(samples, &f, state)?;
                    if verdict == Some(false) {
                        anchor.refute(&step, samples, &mut missing)?;
                    }
                    verdict
                }
                _ => None,
            };
            if verdict != Some(true) && !judged {
                judged = true;
                if missing.is_empty() {
                    continue;
                }
            }
            judged |= verdict == Some(false);
        }
        if !missing.is_empty() {
            // Some neighbours were too far apart to fix a branch: add samples
            // between them and repeat the chain.
            offsets.extend(missing);
            offsets.sort_by(|a, b| a.partial_cmp(b).expect("finite trail offsets"));
            offsets.dedup();
            if cnum::verbose() {
                eprintln!(
                    "schröder unwind: branches by continuity need {} samples per unit of height",
                    offsets.len()
                );
            }
            continue;
        }
        check_chain(&f, &step)?;
        // The largest growth of an error made at any step, with one rounding
        // per step; never more than the sum of the expanding steps.
        let rounded = Float::with_val_64(prec, Integer::from(&updates + 1u32)).ln();
        log_amplification += expanding.min(&(net - lowest + rounded));
        log_amplification -= cnum::log_magnitude(&f, prec).min(&Float::new_64(prec));
        if jump_count > 0 {
            log_amplification += jump_error_log + Float::with_val_64(prec, updates + 1).ln();
            if matches!(&jumps, Some(OrbitJumps::Complex(_))) {
                // Complex error disks must protect each returned component, not just |F|.
                log_amplification += component_conditioning(&f, prec);
            }
        }
        if let Some(own) = &own {
            offsets.retain(|offset| offset != own);
        }
        Trail::remember(state, &offsets);
        return Ok((f, log_amplification));
    }
}

thread_local! {
    /// The sample offsets that the last logarithm chain of this thread needed,
    /// with the fixed point of its base: other heights of that base start
    /// from them instead of finding them again. They only set how densely
    /// the curve is sampled, never a value.
    static TRAIL_OFFSETS: std::cell::RefCell<Option<(Complex, Vec<Float>)>> =
        const { std::cell::RefCell::new(None) };
}

/// The integer tower is kept this far up for the check of `Anchor`.
const ANCHOR_TOWER: usize = 1 << 16;

/// For a real height: the sample of a `Trail` at the next integer above the
/// chain point. The tetration there is a value of the integer tower, which
/// the traced branches have to reproduce at every level.
struct Anchor {
    offset: Float,
    /// The height of that sample at the end of the chain.
    ceiling: Integer,
    /// `b^^m` for `m = -1, 0, 1, ...`.
    tower: Vec<Complex>,
    /// The step of the last disagreement.
    refuted: Option<Integer>,
}

impl Anchor {
    const SERIES: &str = "Schröder: the series of this base contradicts its integer tower";

    fn new(h: &Complex, prec: u64) -> Option<Anchor> {
        if !h.imag().is_zero() {
            return None;
        }
        let mut ceiling = h.real().clone().ceil();
        if ceiling == *h.real() {
            ceiling += 1u32;
        }
        let offset = Float::with_val_64(prec, &ceiling - h.real());
        if !offset.is_finite() || offset <= 0 || offset > 1 {
            return None;
        }
        Some(Anchor {
            offset,
            ceiling: ceiling.to_integer()?,
            tower: Vec::new(),
            refuted: None,
        })
    }

    fn index(&self, trail: &Trail) -> Option<usize> {
        trail
            .offsets
            .iter()
            .position(|offset| *offset == self.offset)
    }

    /// Whether the samples `left` levels above the end of the chain hold the
    /// value of the integer tower; `None` above the part of the tower kept.
    fn holds(
        &mut self,
        trail: &Trail,
        left: &Integer,
        state: &SchroderState,
    ) -> Result<Option<bool>, String> {
        let Some(index) = self.index(trail) else {
            return Ok(None);
        };
        let level = Integer::from(&self.ceiling + left);
        self.matches(&trail.values[index], &level, state)
    }

    /// As `holds`, one level below the end of the chain, whose final value
    /// `point` is the sample at offset 1 there: a wrong branch of `point`
    /// would move the sample at the anchor with it.
    fn confirms(
        &mut self,
        trail: &Trail,
        point: &Complex,
        state: &SchroderState,
    ) -> Result<Option<bool>, String> {
        let Some(index) = self.index(trail) else {
            return Ok(None);
        };
        let value = trail.below(point, index, &state.ln_b)?;
        let level = Integer::from(&self.ceiling - 1u32);
        self.matches(&value, &level, state)
    }

    /// A chain that the tower has refuted is repeated with twice the samples,
    /// unless the chain before it failed no earlier.
    fn refute(
        &mut self,
        step: &Integer,
        trail: &Trail,
        missing: &mut Vec<Float>,
    ) -> Result<(), String> {
        if self.refuted.as_ref().is_some_and(|last| step <= last) {
            return Err(
                "Schröder: the logarithm branches of this height contradict the integer tower"
                    .into(),
            );
        }
        if cnum::verbose() {
            eprintln!("schröder unwind: step {step} contradicts the integer tower");
        }
        self.refuted = Some(step.clone());
        trail.halve(missing)
    }

    /// Whether `value` is `b^^level`.
    fn matches(
        &mut self,
        value: &Complex,
        level: &Integer,
        state: &SchroderState,
    ) -> Result<Option<bool>, String> {
        let prec = state.prec;
        // The tower starts at `b^^(-1)`.
        let Some(position) = Integer::from(level + 1u32).to_usize() else {
            return Ok(None);
        };
        if position >= ANCHOR_TOWER {
            return Ok(None);
        }
        if self.tower.is_empty() {
            self.tower.push(cnum::zero(prec));
        }
        while self.tower.len() <= position {
            let exponent =
                Complex::with_val_64(prec, &self.tower[self.tower.len() - 1] * &state.ln_b);
            self.tower.push(cnum::checked_exp(&exponent, prec)?);
        }
        let apart = cnum::abs(
            &Complex::with_val_64(prec, value - &self.tower[position]),
            prec,
        );
        // Neighbouring branches lie `2 pi / |ln b|` apart.
        let spacing = Float::with_val_64(prec, Constant::Pi) / cnum::abs(&state.ln_b, prec);
        Ok(Some(apart * 2u32 <= spacing))
    }
}

/// A value of the tetration and, where known, `|dF/ds|` along real shifts `s`.
type Sample = (Complex, Option<Float>);

/// Samples of the curve `s -> F(h + s)`, `s` real, over one unit of height
/// above the current point of a logarithm chain: offsets `0 < u <= 1`, the
/// values there and, where known, the speeds `|dF/ds|`.
///
/// The tetration is continuous along that curve, so the logarithm of every
/// sample lies on the branch next to that of its neighbour, and the sample at
/// `u = 1` of the level below is the chain point of this level, which takes
/// no logarithm. This fixes the branches without assuming that they are the
/// principal ones: next to the Shell-Thron boundary the curve winds around 0.
#[derive(Clone)]
struct Trail {
    offsets: Vec<Float>,
    values: Vec<Complex>,
    speeds: Vec<Option<Float>>,
}

impl Trail {
    /// Eight samples per unit of height, or those remembered for this base.
    fn offsets(state: &SchroderState) -> Vec<Float> {
        let prec = state.prec;
        let remembered = TRAIL_OFFSETS.with(|kept| {
            kept.borrow().as_ref().and_then(|(fixed, offsets)| {
                let apart = cnum::abs(&Complex::with_val_64(prec, fixed - &state.l), prec);
                (apart <= cnum::abs(&state.l, prec) >> 20u32).then(|| offsets.clone())
            })
        });
        match remembered {
            Some(offsets) => offsets
                .iter()
                .map(|offset| Float::with_val_64(prec, offset))
                .collect(),
            None => (1..=8u32)
                .map(|part| Float::with_val_64(prec, part) >> 3u32)
                .collect(),
        }
    }

    fn remember(state: &SchroderState, offsets: &[Float]) {
        if offsets.len() > 8 {
            TRAIL_OFFSETS.with(|kept| {
                *kept.borrow_mut() = Some((state.l.clone(), offsets.to_vec()));
            });
        }
    }

    fn new(
        offsets: &[Float],
        top: &dyn Fn(&Float) -> Result<Sample, String>,
    ) -> Result<Trail, String> {
        let mut values = Vec::with_capacity(offsets.len());
        let mut speeds = Vec::with_capacity(offsets.len());
        for offset in offsets {
            let (value, speed) = top(offset)?;
            values.push(value);
            speeds.push(speed);
        }
        Ok(Trail {
            offsets: offsets.to_vec(),
            values,
            speeds,
        })
    }

    /// Take every sample `steps` logarithms down without its neighbours: by
    /// Taylor jumps, or by principal logarithms where the sample is closer to
    /// the fixed point than half the distance of the latter from the cut.
    fn descend(
        &mut self,
        steps: &Integer,
        jumps: &mut Option<OrbitJumps>,
        state: &SchroderState,
    ) -> Result<(), String> {
        let prec = state.prec;
        let reach = if state.l.real().is_sign_negative() {
            Float::with_val_64(prec, state.l.imag().abs_ref())
        } else {
            cnum::abs(&state.l, prec)
        };
        let ln_b_size = cnum::abs(&state.ln_b, prec);
        for (value, speed) in self.values.iter_mut().zip(self.speeds.iter_mut()) {
            let mut left = steps.clone();
            while left > 0 {
                let jump = match jumps {
                    Some(cache) => cache.advance(value, Some(&Float::with_val_64(prec, &left)))?,
                    None => None,
                };
                if let Some(jump) = jump {
                    if let Some(speed) = speed {
                        *speed *= &jump.derivative;
                    }
                    *value = jump.value;
                    left -= jump.steps;
                    continue;
                }
                let distance = cnum::abs(&Complex::with_val_64(prec, &*value - &state.l), prec);
                if distance * 2u32 > reach {
                    return Err(
                        "Schröder: a logarithm branch cannot be fixed by continuity after Taylor jumps"
                            .into(),
                    );
                }
                if let Some(speed) = speed {
                    *speed /= cnum::abs(value, prec) * &ln_b_size;
                }
                *value = cnum::ln_complex(value, prec) / &state.ln_b;
                left -= 1;
            }
        }
        Ok(())
    }

    /// The sample at `index` one level further down, continued from the
    /// chain point `point` of this level.
    fn below(&self, point: &Complex, index: usize, ln_b: &Complex) -> Result<Complex, String> {
        let prec = point.prec_64().0;
        let mut value = point.clone();
        for position in (index..self.values.len() - 1).rev() {
            if cnum::is_zero(&self.values[position]) {
                return Err("Schröder: logarithm of zero while tracing a branch".into());
            }
            let mut candidate = cnum::ln_complex(&self.values[position], prec) / ln_b;
            continue_branch(&mut candidate, &value, ln_b, prec)?;
            value = candidate;
        }
        Ok(value)
    }

    /// Offsets halfway between all neighbours.
    fn halve(&self, missing: &mut Vec<Float>) -> Result<(), String> {
        let prec = self.offsets[0].prec_64();
        let shift = usize::try_from(prec / 2).map_err(|_| "precision overflow")?;
        let floor = Float::with_val_64(prec, 1) >> shift;
        let mut below = Float::new_64(prec);
        let mut added = false;
        for offset in &self.offsets {
            let gap = Float::with_val_64(prec, offset - &below);
            if gap >= floor {
                missing.push(Float::with_val_64(prec, &below + gap / 2u32));
                added = true;
            }
            below = offset.clone();
        }
        if !added {
            return Err(
                "Schröder: the logarithm branches of this height cannot be fixed by continuity"
                    .into(),
            );
        }
        Ok(())
    }

    /// One logarithm of the chain point `point` (speed `speed`), whose
    /// principal value divided by `ln b` is `next`: moves `next` and the
    /// samples to their continuous branches. With `judged`, offsets between
    /// neighbours that are too far apart to decide are added to `missing`.
    fn step(
        &mut self,
        point: &Complex,
        speed: &Option<Float>,
        next: &mut Complex,
        ln_b: &Complex,
        judged: bool,
        missing: &mut Vec<Float>,
    ) -> Result<(), String> {
        let prec = next.prec_64().0;
        let ln_b_size = cnum::abs(ln_b, prec);
        let rate = |value: &Complex, speed: &Option<Float>| {
            speed
                .as_ref()
                .map(|speed| Float::with_val_64(prec, speed / cnum::abs(value, prec)))
        };
        let last = self.values.len() - 1;
        let zero = Float::new_64(prec);
        // The pair of neighbours (`near` below `far`) is close enough when the
        // argument turns by at most an eighth of a turn between them and the
        // relative change of the value over their distance stays below 1/2 at
        // both ends. Otherwise samples are added: next to an end where the
        // value changes fast (a zero of the tetration is near) at distances
        // doubling from about 1/(2 rate), and in the middle.
        let mut judge = |turn: Float,
                         near: (&Float, Option<Float>),
                         far: (&Float, Option<Float>)|
         -> Result<(), String> {
            let gap = Float::with_val_64(prec, far.0 - near.0);
            let mut ladder = |from: &Float, rate: &Option<Float>, forward: bool| {
                let Some(rate) = rate else { return false };
                if !rate.is_finite() || Float::with_val_64(prec, rate * &gap) * 2u32 <= 1u32 {
                    return false;
                }
                // Powers of two keep the samples of all steps on one grid.
                let exponent = Float::with_val_64(prec, rate * 2u32).recip().log2().floor();
                let mut reach = Float::with_val_64(prec, 2).pow(exponent);
                while reach.is_normal() && Float::with_val_64(prec, &reach * 2u32) <= gap {
                    let offset = if forward {
                        Float::with_val_64(prec, from + &reach)
                    } else {
                        Float::with_val_64(prec, from - &reach)
                    };
                    missing.push(offset);
                    reach *= 2u32;
                }
                true
            };
            if !judged {
                return Ok(());
            }
            let fast = ladder(near.0, &near.1, true) | ladder(far.0, &far.1, false);
            if fast || turn * 8u32 > 1u32 {
                let shift = usize::try_from(prec / 2).map_err(|_| "precision overflow")?;
                if gap < (Float::with_val_64(prec, 1) >> shift) {
                    return Err(
                        "Schröder: the logarithm branches of this height cannot be fixed by continuity"
                            .into(),
                    );
                }
                missing.push(Float::with_val_64(prec, near.0 + gap / 2u32));
            }
            Ok(())
        };
        let mut after = self.values.clone();
        after[last] = point.clone();
        for index in (0..last).rev() {
            if cnum::is_zero(&self.values[index]) {
                return Err("Schröder: logarithm of zero while tracing a branch".into());
            }
            let mut candidate = cnum::ln_complex(&self.values[index], prec) / ln_b;
            let turn = continue_branch(&mut candidate, &after[index + 1], ln_b, prec)?;
            judge(
                turn,
                (
                    &self.offsets[index],
                    rate(&self.values[index], &self.speeds[index]),
                ),
                (
                    &self.offsets[index + 1],
                    rate(&self.values[index + 1], &self.speeds[index + 1]),
                ),
            )?;
            after[index] = candidate;
        }
        let turn = continue_branch(next, &after[0], ln_b, prec)?;
        judge(
            turn,
            (&zero, rate(point, speed)),
            (&self.offsets[0], rate(&self.values[0], &self.speeds[0])),
        )?;
        for index in 0..last {
            if let Some(speed) = &mut self.speeds[index] {
                *speed /= cnum::abs(&self.values[index], prec) * &ln_b_size;
            }
        }
        self.speeds[last] = speed.clone();
        self.values = after;
        Ok(())
    }
}

/// Move `candidate`, a logarithm divided by `ln b`, to the branch nearest to
/// `reference`. Returns the turn of the argument between the two points whose
/// logarithms these are, as a fraction of a full turn.
fn continue_branch(
    candidate: &mut Complex,
    reference: &Complex,
    ln_b: &Complex,
    prec: u64,
) -> Result<Float, String> {
    let turn = Complex::with_val_64(prec, (0, Float::with_val_64(prec, Constant::Pi) * 2u32));
    let ratio = Complex::with_val_64(prec, reference - &*candidate) * ln_b / &turn;
    if !cnum::is_finite(&ratio) {
        return Err("Schröder: logarithm branch selection is not finite".into());
    }
    let branch = ratio.real().clone().round();
    let miss = Float::with_val_64(prec, ratio.real() - &branch).abs();
    if !branch.is_zero() {
        *candidate += Complex::with_val_64(prec, &turn * &branch) / ln_b;
    }
    Ok(miss)
}

/// `steps` further logarithms to a real base of a point whose real part `x`
/// no longer changes and whose imaginary part `y` has `(y/x)^2` below the
/// working resolution: every step then divides `y` by `x ln b` exactly, to
/// that resolution. Returns the point and the logarithm of the factor by
/// which the rounding of the exponent grows; `None` when the point is not of
/// that kind.
fn settle_imaginary(
    point: &Complex,
    ln_b: &Float,
    steps: &Integer,
    prec: u64,
) -> Result<Option<(Complex, Float)>, String> {
    let (x, y) = (point.real(), point.imag());
    if y.is_zero() || !x.is_finite() || *x <= 0 {
        return Ok(None);
    }
    let bits = usize::try_from(prec / 2 + 2).map_err(|_| "precision overflow")?;
    if y.clone().abs() >= (x.clone() >> bits) {
        return Ok(None);
    }
    let rate = Float::with_val_64(prec, x * ln_b);
    if !rate.is_finite() || rate <= 1 {
        return Ok(None);
    }
    let exponent = Float::with_val_64(prec, rate.ln() * steps);
    let settled = Float::with_val_64(prec, y * (-exponent.clone()).exp());
    if settled.is_zero() || !settled.is_finite() {
        return Err(
            "Schröder: the imaginary part at this height is below MPFR's exponent range".into(),
        );
    }
    let growth = exponent.ln().max(&Float::new_64(prec));
    Ok(Some((Complex::with_val_64(prec, (x, settled)), growth)))
}

fn inverse_coordinate(state: &SchroderState, h: &Complex) -> Result<Complex, String> {
    let prec = state.prec;
    if has_real_negative_multiplier(state) {
        return real_negative_coordinate(state, h, &Float::new_64(prec));
    }
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

fn has_real_negative_multiplier(state: &SchroderState) -> bool {
    state.l.imag().is_zero()
        && state.ln_b.imag().is_zero()
        && *state.l.real() > 0
        && *state.ln_b.real() < 0
}

fn real_negative_coordinate(
    state: &SchroderState,
    h: &Complex,
    shift: &Float,
) -> Result<Complex, String> {
    let prec = state.prec;
    if !cnum::is_finite(h) || !shift.is_finite() || !shift.is_integer() {
        return Err(
            "Negative-multiplier coordinate requires a finite height and integer shift".into(),
        );
    }
    let scale = state
        .s1
        .real()
        .clone()
        .abs()
        .max(&state.s1.imag().clone().abs());
    if !scale.is_finite() || scale <= 0 {
        return Err("Schroder normalization coordinate is zero or non-finite".into());
    }
    let unit = Complex::with_val_64(prec, &state.s1 / &scale);
    let mut pi = Float::with_val_64(prec, rug::float::Constant::Pi);
    let mut phase = Complex::with_val_64(prec, (h.real().cos_pi_ref(), h.real().sin_pi_ref()));
    if *state.ln_lambda.imag() < 0 {
        phase = phase.conj();
        pi = -pi;
    }
    if Float::with_val_64(prec, shift % 2_u32) != 0 {
        phase = -phase;
    }
    // Keep the integer phase exact, and do not lose a tiny h in h + shift.
    let mut exponent = Complex::with_val_64(prec, h * state.ln_lambda.real());
    *exponent.mut_real() += Float::with_val_64(prec, shift * state.ln_lambda.real());
    *exponent.mut_real() -= Float::with_val_64(prec, h.imag() * pi);
    *exponent.mut_real() += scale.ln();
    let coordinate = Complex::with_val_64(prec, unit * phase * cnum::checked_exp(&exponent, prec)?);
    if !cnum::is_finite(&coordinate) || cnum::is_zero(&coordinate) {
        return Err("Negative-multiplier inverse coordinate exceeded MPFR's exponent range".into());
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

    fn negative_coordinate_state(prec: u64) -> SchroderState {
        let mut lambda = Complex::with_val_64(prec, -1);
        lambda += cnum::decimal("1e-45", prec);
        let l = Complex::with_val_64(prec, lambda.exp_ref());
        let ln_b = Complex::with_val_64(prec, &lambda / &l);
        SchroderState {
            base: cnum::checked_exp(&ln_b, prec).unwrap(),
            l,
            ln_lambda: cnum::ln_complex(&lambda, prec),
            ln_b,
            lam_abs: cnum::abs(&lambda, prec),
            sigma_inv: Vec::new(),
            s1: cnum::one(prec),
            safe_radius: Float::with_val_64(prec, 1),
            sigma_inner_radius: Float::with_val_64(prec, 1),
            inverse_radius: None,
            univalent: false,
            prec,
        }
    }

    #[test]
    fn anchor_compares_samples_with_the_integer_tower() {
        let prec = cnum::digits_to_bits(20);
        let base = cnum::parse_complex("0.5", "0", prec).unwrap();
        let fp = match crate::regions::classify(&base, prec).unwrap() {
            crate::regions::Region::ShellThronInterior(fp) => fp,
            other => panic!("expected strict attraction, got {other:?}"),
        };
        let state = setup_schroder(&base, &fp, prec).unwrap();
        let height = |re: &str, im: &str| cnum::parse_complex(re, im, prec).unwrap();
        assert!(Anchor::new(&height("0.25", "0.1"), prec).is_none());
        let whole = Anchor::new(&height("2", "0"), prec).unwrap();
        assert_eq!(
            (whole.offset, whole.ceiling),
            (Float::with_val_64(prec, 1), Integer::from(3))
        );
        let mut anchor = Anchor::new(&height("0.25", "0"), prec).unwrap();
        assert_eq!(anchor.offset, 0.75);
        assert_eq!(anchor.ceiling, 1);
        // Two levels above the end of the chain the anchor sample is b^^3.
        let half = Float::with_val_64(prec, 0.5);
        let third = half.clone().pow(half.clone().pow(&half));
        let mut trail = Trail {
            offsets: vec![anchor.offset.clone(), Float::with_val_64(prec, 1)],
            values: vec![Complex::with_val_64(prec, third), cnum::one(prec)],
            speeds: vec![None, None],
        };
        let two = Integer::from(2);
        assert_eq!(anchor.holds(&trail, &two, &state).unwrap(), Some(true));
        assert_eq!(
            anchor
                .holds(&trail, &Integer::from(ANCHOR_TOWER), &state)
                .unwrap(),
            None
        );
        let turn = Complex::with_val_64(prec, (0, Float::with_val_64(prec, Constant::Pi) * 2u32));
        trail.values[0] += turn / &state.ln_b;
        assert_eq!(anchor.holds(&trail, &two, &state).unwrap(), Some(false));
        // A refuted chain is repeated with twice the samples, but only while
        // the contradictions come later and later.
        let mut missing = Vec::new();
        anchor
            .refute(&Integer::from(5), &trail, &mut missing)
            .unwrap();
        assert_eq!(
            missing,
            [
                Float::with_val_64(prec, 0.375),
                Float::with_val_64(prec, 0.875)
            ]
        );
        anchor
            .refute(&Integer::from(6), &trail, &mut missing)
            .unwrap();
        assert!(anchor
            .refute(&Integer::from(6), &trail, &mut missing)
            .is_err());
    }

    #[test]
    fn negative_coordinates_preserve_native_height_components_and_integer_parity() {
        let prec = cnum::digits_to_bits(110);
        let state = negative_coordinate_state(prec);
        let pi = Float::with_val_64(prec, rug::float::Constant::Pi);
        let large: Integer = Integer::from(1) << 140;
        for shift in [large.clone(), Integer::from(&large + 1), -large] {
            let mut amplitude = state.lam_abs.clone().pow(&shift);
            if shift.is_odd() {
                amplitude = -amplitude;
            }
            let shift = Float::with_val_64(prec, shift);
            for text in ["1e-80", "1e-1000", "1e-1000000000000000000"] {
                let epsilon = cnum::decimal(text, prec);
                for (height, expected_imaginary) in [
                    (
                        Complex::with_val_64(prec, (0, &epsilon)),
                        Float::with_val_64(prec, &amplitude * state.ln_lambda.real()) * &epsilon,
                    ),
                    (
                        Complex::with_val_64(prec, (&epsilon, 0)),
                        Float::with_val_64(prec, &amplitude * &pi) * &epsilon,
                    ),
                ] {
                    let actual = real_negative_coordinate(&state, &height, &shift).unwrap();
                    for (actual, expected) in [
                        (actual.real(), &amplitude),
                        (actual.imag(), &expected_imaginary),
                    ] {
                        let error = Float::with_val_64(prec, actual - expected).abs();
                        assert!(
                            error < cnum::epsilon(70, prec) * expected.clone().abs(),
                            "{text}, shift={shift}: {error}",
                        );
                    }
                }
            }
            let half = cnum::parse_complex("0.5", "0", prec).unwrap();
            let actual = real_negative_coordinate(&state, &half, &shift).unwrap();
            assert!(actual.real().is_zero());
            let expected = amplitude * state.lam_abs.clone().sqrt();
            assert!(
                Float::with_val_64(prec, actual.imag() - &expected).abs()
                    < cnum::epsilon(70, prec) * expected.abs()
            );
        }
    }

    #[test]
    fn negative_coordinates_preserve_sheet_and_validate_inputs() {
        let prec = cnum::digits_to_bits(70);
        let mut state = negative_coordinate_state(prec);
        let height = cnum::parse_complex("0.5", "0.25", prec).unwrap();
        let upper = inverse_coordinate(&state, &height).unwrap();
        *state.ln_lambda.mut_imag() *= -1;
        let lower = inverse_coordinate(&state, &height.conj()).unwrap();
        assert_eq!(lower, upper.conj());
        for shift in [
            cnum::decimal("0.5", prec),
            Float::with_val_64(prec, rug::float::Special::Infinity),
            Float::with_val_64(prec, rug::float::Special::Nan),
        ] {
            assert!(real_negative_coordinate(&state, &cnum::one(prec), &shift).is_err());
        }
        let nan = Complex::with_val_64(prec, rug::float::Special::Nan);
        assert!(real_negative_coordinate(&state, &nan, &Float::new_64(prec)).is_err());
        state.s1 = cnum::zero(prec);
        assert!(inverse_coordinate(&state, &cnum::one(prec)).is_err());
    }

    #[test]
    fn cached_exact_anchors_keep_working_precision() {
        let digits = 50;
        let prec = cnum::digits_to_bits(digits);
        let base = cnum::parse_complex("1.25", "0.125", 32).unwrap();
        let crate::regions::Region::ShellThronInterior(fp) =
            crate::regions::classify(&base, prec).unwrap()
        else {
            panic!("expected a strict attractor");
        };
        let state = setup_schroder(&base, &fp, prec).unwrap();
        for (height, expected) in [
            (-1, cnum::zero(prec)),
            (0, cnum::one(prec)),
            (1, Complex::with_val_64(prec, &base)),
        ] {
            let height = Complex::with_val_64(prec, height);
            let actual = eval_schroder_at_digits(&state, &height, digits).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(actual.real().prec_64(), prec);
            assert_eq!(actual.imag().prec_64(), prec);
        }
    }

    #[test]
    fn complex_jump_conditioning_protects_both_components() {
        let prec = cnum::digits_to_bits(100);
        for (real, imaginary) in [
            ("1", "1e-40"),
            ("-1", "1e-40"),
            ("1e-40", "1"),
            ("1e-40", "-1"),
            ("1e1000", "1e960"),
            ("1e-960", "1e-1000"),
        ] {
            let value = cnum::parse_complex(real, imaginary, prec).unwrap();
            let gain = component_conditioning(&value, prec);
            let next = cnum::conditioned_precision(&gain, &cnum::working_epsilon(prec), prec)
                .unwrap()
                .unwrap();
            assert_eq!(next, prec + 165, "{real}+{imaginary}i");
        }
        for (real, imaginary) in [("1", "0"), ("0", "1"), ("-1", "0"), ("0", "-1")] {
            let value = cnum::parse_complex(real, imaginary, prec).unwrap();
            assert!(component_conditioning(&value, prec).is_zero());
        }
    }

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
            eval_poincare_series(&coefficients, &coordinate, &radius, false, prec).unwrap();
        assert_eq!(value, coordinate);
        assert_eq!(derivative, cnum::one(prec));
        assert!(
            eval_poincare_series(&coefficients[..2], &coordinate, &radius, false, prec).is_err()
        );
        assert!(
            eval_poincare_series(&coefficients, &cnum::one(prec), &radius, false, prec).is_err()
        );
        assert!(
            eval_poincare_series(&coefficients[..1], &coordinate, &radius, false, prec).is_err()
        );
        for invalid in [rug::float::Special::Nan, rug::float::Special::Infinity] {
            assert!(eval_poincare_series(
                &coefficients,
                &coordinate,
                &Float::with_val_64(prec, invalid),
                false,
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
