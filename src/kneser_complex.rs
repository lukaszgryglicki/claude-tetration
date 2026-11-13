//! Tetration for complex bases outside the Shell-Thron region by a two-sided
//! theta mapping.
//!
//! With `a = Log b` and `Im b >= 0`, the primary fixed points of `b^z` are
//! `L+ = -W_0(-a)/a` and `L- = -W_1(-a)/a`; both depend analytically on `b` in
//! the upper half-plane, `arg(a L+)` lies in `(0, pi)`, `arg(a L-)` in
//! `(-pi, 0)`, and `L-` always repels. Where `L+` repels as well, the
//! tetration tends to `L+` as `Im z -> +inf` and to `L-` as `Im z -> -inf`:
//!
//! `F(z) = S+(z + theta+(z))` for `Im z >= y0`, `F(z) = S-(z + theta-(z))` for
//! `Im z <= -y0`, with the regular superfunctions `S+-` and 1-periodic
//! `theta+-` holding only the modes `exp(+-2 pi i k z)`, `k >= 0`. Between the
//! two lines `F` is a complex Taylor polynomial. This is the function the
//! Cauchy solver in `kouznetsov.rs` approximates, and for a real base above
//! `e^(1/e)` it is the Kneser tetration of `kneser.rs`.
//!
//! The fixed-point map, the Anderson mixing and the precision ladder follow
//! `kneser.rs`. Nothing here is Schwarz symmetric: the Cauchy contour is the
//! whole circle, the normalizing shift is complex, and every logarithm branch
//! is chosen by continuity (along the sample lines, along the contour, and
//! down a vertical segment for heights left of the Taylor disk). The
//! discretization is enlarged when the consistency gates show that the
//! function is less regular than for a real base. The gates are consistency
//! checks, not a certified forward-error bound.

use std::collections::VecDeque;

use rug::{float::Constant, ops::Pow, Complex, Float, Integer};

use crate::kneser::{
    exp_complex, map_points, nats, solve_linear, theta_value, to_count, KneserGeometry,
    ANDERSON_DEPTH, FIRST_LEVEL_DIGITS, GUARD_DIGITS,
};
use crate::{cnum, fft, kneser, lambertw};
use cnum::DisplayFloat;

/// Largest change of argument, as a fraction of a turn, between a value and
/// the neighbour that fixes the branch of its logarithm.
const BRANCH_TOLERANCE: (u32, u32) = (2, 5);
/// Precision of the path planning in the `ln b` plane and of fixed-point
/// seeds: both only start iterations that converge at the working precision.
const SCOUT_BITS: u64 = 128;

/// `P(z)` and `P'(z)` for complex coefficients.
fn poly_value(coefficients: &[Complex], z: &Complex, prec: u64) -> (Complex, Complex) {
    let mut terms = coefficients.iter().rev();
    let mut value = match terms.next() {
        Some(leading) => Complex::with_val_64(prec, leading),
        None => return (cnum::zero(prec), cnum::zero(prec)),
    };
    let mut derivative = cnum::zero(prec);
    for coefficient in terms {
        derivative *= z;
        derivative += &value;
        value *= z;
        value += coefficient;
    }
    (value, derivative)
}

fn two_pi_i(prec: u64) -> Complex {
    Complex::with_val_64(prec, (0, Float::with_val_64(prec, Constant::Pi) * 2u32))
}

/// Move `candidate`, one value of a logarithm divided by `ln b`, onto the
/// branch nearest to `reference`. Returns the branch number.
fn nearest_branch(
    candidate: &mut Complex,
    reference: &Complex,
    ln_b: &Complex,
    prec: u64,
) -> Result<Integer, String> {
    let turn = two_pi_i(prec);
    let ratio = Complex::with_val_64(prec, reference - &*candidate) * ln_b / &turn;
    if !cnum::is_finite(&ratio) {
        return Err("logarithm branch selection is not finite".into());
    }
    // Only the argument decides the branch; the modulus may change by any
    // factor next to a zero.
    let branch = ratio.real().clone().round();
    let miss = Float::with_val_64(prec, ratio.real() - &branch).abs();
    if miss * BRANCH_TOLERANCE.1 > BRANCH_TOLERANCE.0 {
        return Err("ambiguous logarithm branch (values are not continuous)".into());
    }
    let branch = branch
        .to_integer()
        .ok_or("logarithm branch number is not finite")?;
    if branch != 0 {
        *candidate += Complex::with_val_64(prec, &turn * &branch) / ln_b;
    }
    Ok(branch)
}

/// The base-dependent analytic data at one fixed point and one precision.
#[derive(Clone)]
struct Side {
    prec: u64,
    ln_b: Complex,
    l: Complex,
    kappa: Complex,
    /// The branch `k` of `(Log w + 2 pi i k) / ln b` that fixes `L`.
    branch: Integer,
    psi: Vec<Complex>,
    psi_radius: Float,
}

impl Side {
    fn new(ln_b: &Complex, seed: &Complex, upper: bool, prec: u64) -> Result<Side, String> {
        let name = if upper { "upper" } else { "lower" };
        let ln_b = Complex::with_val_64(prec, ln_b);
        let seed = Complex::with_val_64(prec, seed);
        let l = crate::kouznetsov::newton_fixed_point(&ln_b, &seed, prec)?;
        let mut lambda = Complex::with_val_64(prec, &l * &ln_b);
        if upper && lambda.imag().is_zero() {
            // A real multiplier is a limit from the upper half base-plane,
            // where its argument is positive.
            lambda.mut_imag().abs_mut();
        }
        let kappa = cnum::ln_complex(&lambda, prec);
        if !cnum::is_finite(&kappa) || *kappa.real() <= 0 {
            return Err(format!("the {name} fixed point is not repelling"));
        }
        if (upper && *kappa.imag() <= 0) || (!upper && *kappa.imag() >= 0) {
            return Err(format!("the {name} multiplier rotates the wrong way"));
        }
        let turns = Float::with_val_64(prec, lambda.imag() - cnum::ln_complex(&l, prec).imag())
            / (Float::with_val_64(prec, Constant::Pi) * 2u32);
        let branch = turns
            .round()
            .to_integer()
            .ok_or("the fixed-point logarithm branch is not finite")?;
        // Stay inside the disk where no low-order resonance inflates the series.
        let scale = cnum::abs(&ln_b, prec) * 2u32;
        let mut radius = cnum::decimal("0.25", prec);
        let mut power = lambda.clone();
        for order in 1..=8u32 {
            let gap = cnum::abs(&Complex::with_val_64(prec, &power - 1u32), prec);
            radius = radius.min(&Float::with_val_64(prec, gap.root(order) / &scale));
            power *= &lambda;
        }
        loop {
            if !radius.is_finite() || radius <= 0 {
                return Err(format!(
                    "degenerate Poincare radius at the {name} fixed point"
                ));
            }
            let (psi, peak) = Self::psi_series(&ln_b, &lambda, &radius, prec)?;
            if peak <= Float::with_val_64(prec, &radius * 2u32) {
                return Ok(Side {
                    prec,
                    ln_b,
                    l,
                    kappa,
                    branch,
                    psi,
                    psi_radius: radius,
                });
            }
            radius /= 2u32;
        }
    }

    /// The coefficients of `psi` (`psi(lambda t) = b^(L + psi(t)) - L`) up to
    /// working accuracy on `|t| <= radius`, and the largest term there.
    fn psi_series(
        ln_b: &Complex,
        lambda: &Complex,
        radius: &Float,
        prec: u64,
    ) -> Result<(Vec<Complex>, Float), String> {
        let shift = usize::try_from(prec).map_err(|_| "precision is not addressable")?;
        let tolerance = Float::with_val_64(prec, radius) >> (shift + 8);
        let mut psi = vec![cnum::zero(prec), cnum::one(prec)];
        let mut powers = vec![cnum::one(prec), lambda.clone()];
        let mut radius_power = radius.clone();
        let mut peak = radius.clone();
        let mut small = 0u32;
        loop {
            let n = psi.len();
            cnum::check_complex_storage(n as u128 * 2 + 2, prec)?;
            powers.push(Complex::with_val_64(prec, &powers[n - 1] * lambda));
            let mut sum = cnum::zero(prec);
            for k in 1..n {
                let product = Complex::with_val_64(prec, &psi[k] * &psi[n - k]);
                sum += Complex::with_val_64(prec, product * &powers[n - k]) * Integer::from(k);
            }
            let denominator = Complex::with_val_64(prec, &powers[n] - lambda) * Integer::from(n);
            if cnum::is_zero(&denominator) {
                return Err(format!("Poincare resonance at order {n}"));
            }
            let coefficient = Complex::with_val_64(prec, sum * ln_b) / denominator;
            if !cnum::is_finite(&coefficient) {
                return Err(format!("Poincare coefficient {n} is not finite"));
            }
            radius_power *= radius;
            let term = cnum::abs(&coefficient, prec) * &radius_power;
            if !term.is_finite() {
                return Err(format!("Poincare term {n} is not finite"));
            }
            small = if term < tolerance { small + 1 } else { 0 };
            peak = peak.max(&term);
            psi.push(coefficient);
            if small >= 8 && n >= 8 {
                return Ok((psi, peak));
            }
        }
    }

    /// `psi(t)` and `psi'(t)` with the arithmetic rounded to `prec` bits.
    fn psi_value(&self, t: &Complex, prec: u64) -> (Complex, Complex) {
        let mut terms = self.psi[1..].iter().rev();
        let leading = terms.next().expect("linear Poincare coefficient");
        let mut quotient = Complex::with_val_64(prec, leading);
        let mut slope = cnum::zero(prec);
        for coefficient in terms {
            slope *= t;
            slope += &quotient;
            quotient *= t;
            quotient += coefficient;
        }
        let derivative = Complex::with_val_64(prec, &slope * t) + &quotient;
        (quotient * t, derivative)
    }

    /// `S(w)` and `S'(w)` with the arithmetic rounded to `prec` bits.
    fn superfunction(&self, w: &Complex, prec: u64) -> Result<(Complex, Complex), String> {
        let exponent = Complex::with_val_64(prec, &self.kappa * w);
        if !cnum::is_finite(&exponent) {
            return Err("superfunction argument is not finite".into());
        }
        let excess = Float::with_val_64(prec, exponent.real() - self.psi_radius.clone().ln())
            / self.kappa.real();
        let steps = if excess > 0 {
            excess
                .ceil()
                .to_integer()
                .ok_or("superfunction step count is not finite")?
        } else {
            Integer::new()
        };
        let turns = Complex::with_val_64(prec, &self.kappa * &steps);
        let reduced = Complex::with_val_64(prec, &exponent - &turns);
        if *reduced.real() < -(nats(prec) + 64u32) {
            return Ok((Complex::with_val_64(prec, &self.l), cnum::zero(prec)));
        }
        let t = exp_complex(&reduced, prec)?;
        let (psi, slope) = self.psi_value(&t, prec);
        let mut value = Complex::with_val_64(prec, &self.l + psi);
        let mut derivative = Complex::with_val_64(prec, &self.kappa * &t) * slope;
        let mut done = Integer::new();
        while done < steps {
            value = exp_complex(&Complex::with_val_64(prec, &value * &self.ln_b), prec)?;
            derivative *= &value;
            derivative *= &self.ln_b;
            done += 1;
        }
        if !cnum::is_finite(&value) || !cnum::is_finite(&derivative) {
            return Err("superfunction value exceeds MPFR's range".into());
        }
        Ok((value, derivative))
    }

    /// `steps` further inverse steps of a point at `L + displacement` inside
    /// the Poincare disk, in closed form: the new point and its derivative
    /// with respect to the old one.
    fn contract(
        &self,
        displacement: &Complex,
        steps: &Integer,
    ) -> Result<(Complex, Complex), String> {
        let prec = self.prec;
        let exponent = -Complex::with_val_64(prec, &self.kappa * steps);
        if !cnum::is_finite(&exponent) || *exponent.real() < -(nats(prec) + 64u32) {
            return Ok((self.l.clone(), cnum::zero(prec)));
        }
        let t = self.invert_psi(displacement)?;
        let scale = exp_complex(&exponent, prec)?;
        let (_, before) = self.psi_value(&t, prec);
        let (psi, after) = self.psi_value(&Complex::with_val_64(prec, &t * &scale), prec);
        if cnum::is_zero(&before) {
            return Err("Poincare inversion has a zero derivative".into());
        }
        Ok((
            Complex::with_val_64(prec, &self.l + psi),
            scale * after / before,
        ))
    }

    /// The `t` with `psi(t) = displacement`, for a displacement well inside
    /// the Poincare disk: Newton at low precision, then one step per doubling.
    fn invert_psi(&self, displacement: &Complex) -> Result<Complex, String> {
        let prec = self.prec;
        if cnum::is_zero(displacement) {
            return Err("Abel coordinate is singular at the fixed point".into());
        }
        let mut work = prec.min(64);
        let mut t = Complex::with_val_64(work, displacement);
        let mut best: Option<(Float, Complex)> = None;
        loop {
            let (value, slope) = self.psi_value(&t, work);
            let residual = Complex::with_val_64(work, value - displacement);
            let error = cnum::abs(&residual, work);
            if let Some((previous, argument)) = &best {
                if error >= *previous {
                    t = argument.clone();
                    break;
                }
            }
            if error.is_zero() {
                break;
            }
            if cnum::is_zero(&slope) {
                return Err("Poincare inversion has a zero derivative".into());
            }
            let next = Complex::with_val_64(work, &t - residual / slope);
            best = Some((error, std::mem::replace(&mut t, next)));
        }
        while work < prec {
            work = work.saturating_mul(2).min(prec);
            let (value, slope) = self.psi_value(&t, work);
            if cnum::is_zero(&slope) {
                return Err("Poincare inversion has a zero derivative".into());
            }
            let residual = Complex::with_val_64(work, value - displacement);
            t = Complex::with_val_64(work, &t - residual / slope);
        }
        let (value, _) = self.psi_value(&t, prec);
        let error = cnum::abs(&Complex::with_val_64(prec, value - displacement), prec);
        let shift = usize::try_from(prec.saturating_sub(24)).map_err(|_| "precision overflow")?;
        if error > cnum::abs(displacement, prec) >> shift {
            return Err("Poincare inversion missed working accuracy".into());
        }
        Ok(t)
    }
}

/// Seeds for the two primary fixed points: the conjugate pair of a real base
/// above `e^(1/e)`, otherwise the `W_0` and `W_1` branches at `-ln b`.
fn fixed_point_seeds(ln_b: &Complex, prec: u64) -> Result<(Complex, Complex), String> {
    let argument = Complex::with_val_64(prec, -ln_b);
    let principal = lambertw::w0(&argument, prec)?;
    let first = cnum::checked_exp(&Complex::with_val_64(prec, -&principal), prec)?;
    if ln_b.imag().is_zero() && *ln_b.real() > Float::with_val_64(prec, -1).exp() {
        let upper = if first.imag().is_sign_negative() {
            first.conj()
        } else {
            first
        };
        let lower = Complex::with_val_64(prec, upper.conj_ref());
        return Ok((upper, lower));
    }
    let second = match lambertw::wk(&argument, 1, prec) {
        Ok(second) => second,
        Err(_) => second_branch(&argument, prec)?,
    };
    Ok((first, Complex::with_val_64(prec, -second) / ln_b))
}

/// `W_1(z)` to seed accuracy from `w = Log z + 2 pi i - Log w`, a contraction
/// because `|W_1| > 1`.
fn second_branch(z: &Complex, prec: u64) -> Result<Complex, String> {
    let work = prec.min(SCOUT_BITS);
    let target = cnum::ln_complex(z, work) + two_pi_i(work);
    let mut w = target.clone();
    let mut least: Option<Float> = None;
    let mut stalled = 0u32;
    loop {
        if cnum::is_zero(&w) || !cnum::is_finite(&w) {
            return Err("the W_1 fixed-point seed did not converge".into());
        }
        let next = Complex::with_val_64(work, &target - cnum::ln_complex(&w, work));
        let change = cnum::abs(&Complex::with_val_64(work, &next - &w), work);
        w = next;
        if change.is_zero() {
            return Ok(w);
        }
        if least.as_ref().is_none_or(|best| change < *best) {
            least = Some(change);
            stalled = 0;
        } else {
            stalled += 1;
            if stalled > 8 {
                let settled = least.is_some_and(|best| best <= cnum::abs(&w, work) >> 60usize);
                return if settled {
                    Ok(w)
                } else {
                    Err("the W_1 fixed-point seed did not converge".into())
                };
            }
        }
    }
}

/// Where a level takes `ln b` from.
#[derive(Clone)]
enum LogBase {
    /// The principal logarithm of this base.
    Of(Complex),
    /// A point of a continuation path in the `ln b` plane.
    Is(Complex),
}

impl LogBase {
    fn at(&self, prec: u64) -> Result<Complex, String> {
        let ln_b = match self {
            LogBase::Of(base) => {
                let mut base = Complex::with_val_64(prec, base);
                if cnum::is_zero(&base) || !cnum::is_finite(&base) {
                    return Err("the base must be finite and nonzero".into());
                }
                if base.imag().is_zero() {
                    // Log of a negative real base is ln|b| + i pi.
                    base.mut_imag().abs_mut();
                }
                cnum::ln_complex(&base, prec)
            }
            LogBase::Is(value) => Complex::with_val_64(prec, value),
        };
        if ln_b.imag().is_sign_negative() && !ln_b.imag().is_zero() {
            return Err("the construction needs Im b >= 0 (reflect the base first)".into());
        }
        Ok(ln_b)
    }
}

enum Arc {
    Upper(Complex),
    Lower(Complex),
    Right(Complex),
    Left(Complex),
}

struct Level {
    upper: Side,
    lower: Side,
    digits: u64,
    centre: Float,
    height: Float,
    /// Normalization point relative to the centre and its value: F(0)=1 or F(-1)=0.
    anchor: (Float, u32),
    modes: usize,
    order: usize,
    /// Samples on `Im z = y0` and `Im z = -y0`: offset from the centre, and `z`.
    line_up: Vec<(Complex, Complex)>,
    line_dn: Vec<(Complex, Complex)>,
    theta_scale: Vec<Complex>,
    circle: Vec<Arc>,
    twist: Vec<Complex>,
    weights: Option<Vec<Float>>,
}

struct Step {
    taylor: Vec<Complex>,
    theta_up: Vec<Complex>,
    theta_dn: Vec<Complex>,
    negative: Float,
    shift: Float,
}

/// `F(z)`, `F'(z)` and `|S'|` from one side's theta series, for `|Im z| >= y0`.
fn theta_side_value(
    side: &Side,
    theta: &[Complex],
    lower: bool,
    z: &Complex,
    prec: u64,
) -> Result<(Complex, Complex, Float), String> {
    let (shift, slope) = if lower {
        let (shift, slope) = theta_value(theta, &Complex::with_val_64(prec, -z), prec)?;
        (shift, -slope)
    } else {
        theta_value(theta, z, prec)?
    };
    let (value, rate) = side.superfunction(&Complex::with_val_64(prec, z + shift), prec)?;
    let sensitivity = cnum::abs(&rate, prec);
    Ok((value, rate * (slope + 1u32), sensitivity))
}

impl Level {
    fn new(
        base: &LogBase,
        geometry: &KneserGeometry,
        digits: u64,
        escalation: u32,
        seeds: Option<&(Complex, Complex)>,
    ) -> Result<Level, String> {
        let prec =
            cnum::checked_digits_to_bits(digits.checked_add(8).ok_or("precision count overflow")?)?;
        let ln_b = base.at(prec)?;
        let (upper_seed, lower_seed) = match seeds {
            Some((upper, lower)) => (upper.clone(), lower.clone()),
            None => fixed_point_seeds(&ln_b, prec)?,
        };
        let upper = Side::new(&ln_b, &upper_seed, true, prec)?;
        let lower = Side::new(&ln_b, &lower_seed, false, prec)?;
        let centre = cnum::parse_float(&geometry.centre, prec)?;
        let radius = cnum::parse_float(&geometry.radius, prec)?;
        let height = cnum::parse_float(&geometry.height, prec)?;
        let one = Float::with_val_64(prec, 1);
        let half = Float::with_val_64(prec, 0.5);
        let reach = Float::with_val_64(prec, &centre + 2u32);
        if !(height > 0 && height < radius && radius < reach && centre > -2 && centre <= 0) {
            return Err("unsupported contour geometry".into());
        }
        // Largest distance from the centre at which the Taylor polynomial is used.
        let cosine = (one.clone() - Float::with_val_64(prec, &height / &radius).square()).sqrt();
        let arc = (radius.clone().square() + 1u32
            - Float::with_val_64(prec, &radius * 2u32) * cosine)
            .sqrt();
        let segment = (height.clone().square() + (one.clone() >> 2u32)).sqrt();
        let used = arc.max(&segment);
        if used >= reach || used > radius {
            return Err("contour geometry leaves the Taylor disk".into());
        }
        let mut goal =
            Float::with_val_64(prec, digits + GUARD_DIGITS) * Float::with_val_64(prec, 10).ln();
        for _ in 0..escalation {
            goal = goal * 3u32 / 2u32;
        }
        let two_pi = Float::with_val_64(prec, Constant::Pi) * 2u32;
        let modes = to_count(
            Float::with_val_64(prec, &goal / &two_pi) / &height + 2u32,
            "theta mode count",
        )?;
        let sample_count = (2 * modes + 2).next_power_of_two().max(16);
        let order = to_count(
            Float::with_val_64(prec, &goal / Float::with_val_64(prec, &reach / &used).ln()) + 4u32,
            "Taylor order",
        )?;
        let circle_count = to_count(
            Float::with_val_64(
                prec,
                &goal / Float::with_val_64(prec, &reach / &radius).ln(),
            ),
            "Cauchy node count",
        )?
        .max(order + 2)
        .next_power_of_two()
        .max(32);
        cnum::check_complex_storage(
            (sample_count as u128 * 2 + circle_count as u128 + order as u128) * 4,
            prec,
        )?;

        let line = |sign: i32| -> Vec<(Complex, Complex)> {
            let y = Float::with_val_64(prec, &height * sign);
            (0..sample_count)
                .map(|j| {
                    let x = Float::with_val_64(prec, j) / Float::with_val_64(prec, sample_count)
                        - &half;
                    let z =
                        Complex::with_val_64(prec, (Float::with_val_64(prec, &centre + &x), &y));
                    (Complex::with_val_64(prec, (x, &y)), z)
                })
                .collect()
        };
        let offset = Float::with_val_64(prec, &centre - &half);
        let theta_scale = (0..=modes)
            .map(|k| {
                let growth =
                    Float::with_val_64(prec, &two_pi * &height) * Float::with_val_64(prec, k);
                let angle =
                    -Float::with_val_64(prec, &two_pi * &offset) * Float::with_val_64(prec, k);
                let scale = growth.exp() / Float::with_val_64(prec, sample_count);
                Complex::with_val_64(
                    prec,
                    (
                        Float::with_val_64(prec, angle.cos_ref()) * &scale,
                        Float::with_val_64(prec, angle.sin_ref()) * &scale,
                    ),
                )
            })
            .collect();
        let pi = Float::with_val_64(prec, Constant::Pi);
        let floor = Float::with_val_64(prec, -&height);
        let circle = (0..circle_count)
            .map(|j| {
                let angle = Float::with_val_64(prec, &pi * Float::with_val_64(prec, 2 * j + 1))
                    / Float::with_val_64(prec, circle_count);
                let x = Float::with_val_64(prec, angle.cos_ref()) * &radius;
                let y = Float::with_val_64(prec, angle.sin_ref()) * &radius;
                if y >= height {
                    Arc::Upper(Complex::with_val_64(prec, (x + &centre, y)))
                } else if y <= floor {
                    Arc::Lower(Complex::with_val_64(prec, (x + &centre, y)))
                } else if x > 0 {
                    Arc::Right(Complex::with_val_64(prec, (x - &one, y)))
                } else {
                    Arc::Left(Complex::with_val_64(prec, (x + &one, y)))
                }
            })
            .collect();
        let mut radius_power = Float::with_val_64(prec, circle_count);
        let twist = (0..=order)
            .map(|n| {
                let angle = -Float::with_val_64(prec, &pi * Float::with_val_64(prec, n))
                    / Float::with_val_64(prec, circle_count);
                let factor = Complex::with_val_64(
                    prec,
                    (
                        Float::with_val_64(prec, angle.cos_ref()),
                        Float::with_val_64(prec, angle.sin_ref()),
                    ),
                ) / &radius_power;
                radius_power *= &radius;
                factor
            })
            .collect();
        let weights = if radius == 1 {
            None
        } else {
            let square = radius.clone().square();
            let mut weight = one.clone();
            Some(
                (0..=order)
                    .map(|_| {
                        let current = weight.clone();
                        weight *= &square;
                        current
                    })
                    .collect(),
            )
        };
        let anchor = kneser::anchor_of(&centre, prec);
        Ok(Level {
            line_up: line(1),
            line_dn: line(-1),
            upper,
            lower,
            digits,
            centre,
            height,
            anchor,
            modes,
            order,
            theta_scale,
            circle,
            twist,
            weights,
        })
    }

    fn prec(&self) -> u64 {
        self.upper.prec
    }

    /// One side's theta coefficients from the polynomial on its line, and the
    /// size of the discarded modes. All samples take the same number of
    /// logarithms; each branch is fixed by its neighbour, and the last sample
    /// by the first one a level earlier, since `F(z + 1 - n) = F(z - (n - 1))`.
    fn theta(
        &self,
        side: &Side,
        line: &[(Complex, Complex)],
        taylor: &[Complex],
        lower: bool,
    ) -> Result<(Vec<Complex>, Float), String> {
        let prec = self.prec();
        let mut points = map_points(line, |(argument, _)| {
            Ok(poly_value(taylor, argument, prec).0)
        })?;
        let target = Float::with_val_64(prec, &side.psi_radius / 4u32);
        let mut levels = Integer::new();
        // The closest approach since the last checkpoint and in the stretch
        // before it: orbits spiral, so single distances are not monotonic.
        let mut closest: Option<Float> = None;
        let mut checkpoint: Option<Float> = None;
        loop {
            let worst = points.iter().fold(Float::new_64(prec), |worst, w| {
                worst.max(&cnum::abs(&Complex::with_val_64(prec, w - &side.l), prec))
            });
            if !worst.is_finite() {
                return Err("a logarithmic orbit is not finite".into());
            }
            if worst <= target {
                break;
            }
            let nearest = match closest.take() {
                Some(seen) => seen.min(&worst),
                None => worst,
            };
            if levels >= 16 && levels.is_power_of_two() {
                if checkpoint.as_ref().is_some_and(|last| nearest >= *last) {
                    return Err("logarithmic orbits do not approach the fixed point".into());
                }
                checkpoint = Some(nearest);
            } else {
                closest = Some(nearest);
            }
            let wrap = points[0].clone();
            let mut next = map_points(&points, |w| {
                if cnum::is_zero(w) {
                    return Err("logarithm of zero on a theta line".to_string());
                }
                Ok(cnum::ln_complex(w, prec) / &side.ln_b)
            })?;
            for j in (0..next.len()).rev() {
                let (head, tail) = next.split_at_mut(j + 1);
                let reference = tail.first().unwrap_or(&wrap);
                nearest_branch(&mut head[j], reference, &side.ln_b, prec)?;
            }
            points = next;
            levels += 1;
        }
        let logs = map_points(&points, |w| {
            let t = side.invert_psi(&Complex::with_val_64(prec, w - &side.l))?;
            Ok(cnum::ln_complex(&t, prec))
        })?;
        let turn = two_pi_i(prec);
        let mut values: Vec<Complex> = Vec::with_capacity(logs.len());
        let mut previous: Option<Complex> = None;
        for (mut log, (_, z)) in logs.into_iter().zip(line) {
            if let Some(last) = &previous {
                let turns =
                    (Float::with_val_64(prec, last.imag() - log.imag()) / turn.imag()).round();
                if !turns.is_zero() {
                    log += Complex::with_val_64(prec, &turn * &turns);
                }
            }
            values.push(Complex::with_val_64(prec, &log / &side.kappa) + &levels - z);
            previous = Some(log);
        }
        let closing = Complex::with_val_64(prec, &values[values.len() - 1] - &values[0])
            * &side.kappa
            / &turn;
        if !closing.real().is_finite() || closing.real().clone().abs() > 0.5 {
            return Err("theta is not periodic on its line".into());
        }
        if lower {
            for value in values.iter_mut() {
                value.conj_mut();
            }
        }
        fft::fft(&mut values, prec, false);
        let count = values.len();
        let negative = values[count / 2..]
            .iter()
            .fold(Float::new_64(prec), |worst, v| {
                worst.max(&cnum::abs(v, prec))
            })
            / Float::with_val_64(prec, count);
        let theta = values[..=self.modes]
            .iter()
            .zip(&self.theta_scale)
            .map(|(value, scale)| {
                let coefficient = Complex::with_val_64(prec, value * scale);
                if lower {
                    coefficient.conj()
                } else {
                    coefficient
                }
            })
            .collect();
        Ok((theta, negative))
    }
}

/// Replace `P(z)` by `P(z + shift)`; the series is summed to working accuracy.
fn shift_taylor(taylor: &mut [Complex], shift: &Complex, prec: u64) -> Result<(), String> {
    if cnum::is_zero(shift) || taylor.len() < 2 {
        return Ok(());
    }
    let bits = usize::try_from(prec).map_err(|_| "precision is not addressable")?;
    let scale = taylor.iter().fold(Float::new_64(prec), |scale, c| {
        scale.max(&cnum::abs(c, prec))
    });
    let tolerance = scale >> (bits + 8);
    let mut term = taylor.to_vec();
    let mut small = 0u32;
    for k in 1..taylor.len() {
        let mut largest = Float::new_64(prec);
        for n in 0..taylor.len() - k {
            let mut next = Complex::with_val_64(prec, &term[n + 1] * shift);
            next *= Integer::from(n + 1);
            next /= Integer::from(k);
            taylor[n] += &next;
            largest = largest.max(&cnum::abs(&next, prec));
            term[n] = next;
        }
        if !largest.is_finite() {
            return Err("Taylor shift exceeded MPFR's range".into());
        }
        small = if largest <= tolerance { small + 1 } else { 0 };
        if small >= 2 {
            break;
        }
    }
    Ok(())
}

/// `F(z) -> F(z + s)` maps `theta(z)` to `theta(z + s) + s`.
fn shift_theta(
    theta: &mut [Complex],
    shift: &Complex,
    lower: bool,
    prec: u64,
) -> Result<(), String> {
    let mut exponent = Complex::with_val_64(prec, two_pi_i(prec) * shift);
    if lower {
        exponent = -exponent;
    }
    let step = exp_complex(&exponent, prec)?;
    let mut phase = cnum::one(prec);
    for coefficient in theta.iter_mut().skip(1) {
        phase *= &step;
        *coefficient *= &phase;
    }
    theta[0] += shift;
    Ok(())
}

impl Level {
    /// One application of the fixed-point map.
    fn apply(&self, taylor: &[Complex]) -> Result<Step, String> {
        let prec = self.prec();
        let (mut theta_up, negative_up) = self.theta(&self.upper, &self.line_up, taylor, false)?;
        let (mut theta_dn, negative_dn) = self.theta(&self.lower, &self.line_dn, taylor, true)?;
        let ln_b = &self.upper.ln_b;
        let mut contour = map_points(&self.circle, |arc| match arc {
            Arc::Upper(z) => Ok(theta_side_value(&self.upper, &theta_up, false, z, prec)?.0),
            Arc::Lower(z) => Ok(theta_side_value(&self.lower, &theta_dn, true, z, prec)?.0),
            Arc::Right(argument) => {
                let (value, _) = poly_value(taylor, argument, prec);
                exp_complex(&Complex::with_val_64(prec, value * ln_b), prec)
            }
            Arc::Left(argument) => {
                let (value, _) = poly_value(taylor, argument, prec);
                if cnum::is_zero(&value) {
                    return Err("logarithm of zero on the Cauchy contour".to_string());
                }
                Ok(cnum::ln_complex(&value, prec) / ln_b)
            }
        })?;
        if !contour.iter().all(cnum::is_finite) {
            return Err("non-finite value on the Cauchy contour".into());
        }
        // The left arc continues the upper part of the contour.
        for j in 1..contour.len() {
            if matches!(self.circle[j], Arc::Left(_)) {
                let (head, tail) = contour.split_at_mut(j);
                nearest_branch(&mut tail[0], &head[j - 1], ln_b, prec)?;
            }
        }
        fft::fft(&mut contour, prec, false);
        let mut taylor: Vec<Complex> = contour[..=self.order]
            .iter()
            .zip(&self.twist)
            .map(|(value, twist)| Complex::with_val_64(prec, value * twist))
            .collect();

        let (origin, anchor_value) = &self.anchor;
        let mut root = Complex::with_val_64(prec, (origin, 0));
        let mut previous: Option<Float> = None;
        loop {
            let (value, slope) = poly_value(&taylor, &root, prec);
            let defect = value - *anchor_value;
            let size = cnum::abs(&defect, prec);
            if size.is_zero() || previous.as_ref().is_some_and(|last| size >= *last) {
                break;
            }
            if !cnum::is_finite(&slope) || cnum::is_zero(&slope) {
                return Err("normalization has a degenerate slope".into());
            }
            root -= defect / slope;
            let moved = cnum::abs(&Complex::with_val_64(prec, &root - origin), prec);
            if !moved.is_finite() || moved > 0.25 {
                return Err("normalization root left the Taylor disk".into());
            }
            previous = Some(size);
        }
        let shift = root - origin;
        shift_taylor(&mut taylor, &shift, prec)?;
        if !cnum::is_zero(&shift) {
            shift_theta(&mut theta_up, &shift, false, prec)?;
            shift_theta(&mut theta_dn, &shift, true, prec)?;
        }
        Ok(Step {
            taylor,
            theta_up,
            theta_dn,
            negative: negative_up.max(&negative_dn),
            shift: cnum::abs(&shift, prec),
        })
    }

    fn inner(&self, a: &[Complex], b: &[Complex]) -> Float {
        let prec = self.prec();
        let mut sum = Float::new_64(prec);
        for (n, (x, y)) in a.iter().zip(b).enumerate() {
            let mut term = Float::with_val_64(prec, x.real() * y.real());
            term += Float::with_val_64(prec, x.imag() * y.imag());
            if let Some(weights) = &self.weights {
                term *= &weights[n];
            }
            sum += term;
        }
        sum
    }

    /// The largest consistency defect of a converged step: both junctions,
    /// the functional equation across the strip, the normalization and the
    /// discarded modes.
    fn defect(&self, step: &Step) -> Result<Float, String> {
        let prec = self.prec();
        let mut worst = step.negative.clone();
        let record = |name: &str, defect: Float, worst: &mut Float| {
            if cnum::verbose() {
                eprintln!("kneser-c: gate {name}: {:.3e}", DisplayFloat(&defect));
            }
            *worst = worst.clone().max(&defect);
        };
        let one = Float::with_val_64(prec, 1);
        for quarter in -2i32..=2 {
            let x = Float::with_val_64(prec, quarter) / 4u32;
            for lower in [false, true] {
                let y = if lower {
                    Float::with_val_64(prec, -&self.height)
                } else {
                    self.height.clone()
                };
                let argument = Complex::with_val_64(prec, (&x, &y));
                let z =
                    Complex::with_val_64(prec, (Float::with_val_64(prec, &x + &self.centre), &y));
                let (side, theta) = if lower {
                    (&self.lower, &step.theta_dn)
                } else {
                    (&self.upper, &step.theta_up)
                };
                let (outer, _, _) = theta_side_value(side, theta, lower, &z, prec)?;
                let (inner, _) = poly_value(&step.taylor, &argument, prec);
                let scale = cnum::abs(&outer, prec).max(&one);
                let defect = cnum::abs(&Complex::with_val_64(prec, outer - inner), prec) / scale;
                record(
                    if lower {
                        "lower junction"
                    } else {
                        "upper junction"
                    },
                    defect,
                    &mut worst,
                );
            }
        }
        let half = Float::with_val_64(prec, 0.5);
        for part in -1i32..=1 {
            let y = Float::with_val_64(prec, &self.height * part) / 2u32;
            let left = Complex::with_val_64(prec, (Float::with_val_64(prec, -&half), &y));
            let right = Complex::with_val_64(prec, (&half, &y));
            let (before, _) = poly_value(&step.taylor, &left, prec);
            let (after, _) = poly_value(&step.taylor, &right, prec);
            let image = exp_complex(&Complex::with_val_64(prec, before * &self.upper.ln_b), prec)?;
            let scale = cnum::abs(&after, prec).max(&one);
            let defect = cnum::abs(&Complex::with_val_64(prec, after - image), prec) / scale;
            record("functional equation", defect, &mut worst);
        }
        let (origin, anchor_value) = &self.anchor;
        let (value, _) = poly_value(&step.taylor, &Complex::with_val_64(prec, (origin, 0)), prec);
        record(
            "normalization",
            cnum::abs(&(value - *anchor_value), prec),
            &mut worst,
        );
        if !worst.is_finite() {
            return Err("a consistency gate is not finite".into());
        }
        Ok(worst)
    }

    /// Anderson-accelerated iteration to the level's fixed point.
    fn solve(&self, start: Vec<Complex>) -> Result<(Step, Float), String> {
        let prec = self.prec();
        let tolerance = cnum::epsilon(self.digits + 3, prec);
        let mut x = start;
        let mut history: VecDeque<(Vec<Complex>, Vec<Complex>)> = VecDeque::new();
        let mut best: Option<(Float, Vec<Complex>)> = None;
        let mut since_best = 0usize;
        let mut iteration = 0u64;
        let mut accelerated = false;
        loop {
            iteration += 1;
            let step = match self.apply(&x) {
                Ok(step) => step,
                Err(error) => match &best {
                    Some((_, taylor)) if accelerated => {
                        if cnum::verbose() {
                            eprintln!(
                                "kneser-c: accelerated iterate rejected ({error}); plain step"
                            );
                        }
                        x = taylor.clone();
                        history.clear();
                        accelerated = false;
                        continue;
                    }
                    _ => return Err(error),
                },
            };
            let residual: Vec<Complex> = step
                .taylor
                .iter()
                .zip(&x)
                .map(|(new, old)| Complex::with_val_64(prec, new - old))
                .collect();
            let norm = self.inner(&residual, &residual).sqrt();
            if !norm.is_finite() {
                return Err("fixed-point residual is not finite".into());
            }
            if cnum::verbose() {
                eprintln!(
                    "kneser-c: {} digits, iteration {iteration}: residual {:.3e}, negative modes {:.3e}, shift {:.3e}",
                    self.digits,
                    DisplayFloat(&norm),
                    DisplayFloat(&step.negative),
                    DisplayFloat(&step.shift)
                );
            }
            if norm <= tolerance {
                return Ok((step, norm));
            }
            let improved = best
                .as_ref()
                .is_none_or(|(least, _)| norm < Float::with_val_64(prec, least / 2u32));
            if improved {
                best = Some((norm.clone(), step.taylor.clone()));
                since_best = 0;
            } else {
                since_best += 1;
                let (least, taylor) = best.as_ref().expect("best iterate");
                if since_best > 4 * ANDERSON_DEPTH + 20 {
                    return Err(format!(
                        "iteration stalled at residual {:.3e} (target {:.3e})",
                        DisplayFloat(least),
                        DisplayFloat(&tolerance)
                    ));
                }
                if accelerated && norm > Float::with_val_64(prec, least * 16u32) {
                    x = taylor.clone();
                    history.clear();
                    accelerated = false;
                    continue;
                }
            }
            history.push_back((step.taylor.clone(), residual));
            if history.len() > ANDERSON_DEPTH + 1 {
                history.pop_front();
            }
            x = step.taylor;
            accelerated = false;
            let depth = history.len() - 1;
            if depth == 0 {
                continue;
            }
            let (image, residual) = history.back().expect("current iterate");
            let differences: Vec<Vec<Complex>> = history
                .iter()
                .take(depth)
                .map(|(_, old)| {
                    residual
                        .iter()
                        .zip(old)
                        .map(|(new, old)| Complex::with_val_64(prec, new - old))
                        .collect()
                })
                .collect();
            let matrix = differences
                .iter()
                .map(|p| differences.iter().map(|q| self.inner(p, q)).collect())
                .collect();
            let right = differences
                .iter()
                .map(|p| self.inner(p, residual))
                .collect();
            if let Some(gamma) = solve_linear(matrix, right) {
                for ((old, _), weight) in history.iter().take(depth).zip(&gamma) {
                    for ((target, new), old) in x.iter_mut().zip(image).zip(old) {
                        *target -= Complex::with_val_64(prec, new - old) * weight;
                    }
                }
                accelerated = true;
            }
        }
    }
}

fn initial_taylor(
    ln_b: &Complex,
    centre: &Float,
    order: usize,
    prec: u64,
) -> Result<Vec<Complex>, String> {
    // F(z) = (e^((z+1)k) - 1)/(e^k - 1), k = Log ln b: F(-1) = 0, F(0) = 1, F'(0) = ln(b) F'(-1).
    let rate = cnum::ln_complex(ln_b, prec);
    if !cnum::is_finite(&rate) {
        return Err("the base has no finite logarithm".into());
    }
    let position = Float::with_val_64(prec, centre + 1u32);
    let mut taylor = Vec::with_capacity(order + 1);
    if cnum::abs(&rate, prec) < cnum::epsilon(6, prec) {
        taylor.push(Complex::with_val_64(prec, (&position, 0)));
        taylor.push(cnum::one(prec));
        taylor.resize(order + 1, cnum::zero(prec));
        return Ok(taylor);
    }
    let denominator = exp_complex(&rate, prec)? - 1u32;
    let growth = exp_complex(&Complex::with_val_64(prec, &rate * &position), prec)?;
    taylor.push(Complex::with_val_64(prec, &growth - 1u32) / &denominator);
    let mut term = growth / denominator;
    for n in 1..=order {
        term *= &rate;
        term /= Integer::from(n);
        taylor.push(term.clone());
    }
    Ok(taylor)
}

/// Converged tetration data for one complex base.
#[derive(Clone)]
pub struct ComplexKneserState {
    upper: Side,
    lower: Side,
    base: LogBase,
    geometry: KneserGeometry,
    escalation: u32,
    centre: Float,
    height: Float,
    theta_up: Vec<Complex>,
    theta_dn: Vec<Complex>,
    taylor: Vec<Complex>,
    /// Internal accuracy goal in decimal digits; the gates passed at this level.
    pub digits: u64,
    pub prec: u64,
    /// Largest accepted consistency defect, not a forward-error bound.
    pub residual: Float,
}

/// A warm start: Taylor data, the first level to solve, fixed-point seeds
/// (only meaningful for the same base) and the size escalation reached.
struct Start {
    taylor: Vec<Complex>,
    digits: u64,
    seeds: Option<(Complex, Complex)>,
    escalation: u32,
}

fn solve(
    base: &LogBase,
    geometry: &KneserGeometry,
    target: u64,
    start: Option<Start>,
) -> Result<ComplexKneserState, String> {
    let (mut digits, mut taylor, mut seeds, mut escalation) = match start {
        Some(start) => (
            start.digits.min(target),
            Some(start.taylor),
            start.seeds,
            start.escalation,
        ),
        None => (FIRST_LEVEL_DIGITS.min(target), None, None, 0),
    };
    let mut coarse: Option<Float> = None;
    loop {
        let level = Level::new(base, geometry, digits, escalation, seeds.as_ref())?;
        let prec = level.prec();
        if cnum::verbose() {
            eprintln!(
                "kneser-c: level {digits} digits ({prec} bits): {} theta modes, {} samples, Taylor order {}, {} contour nodes, {}+{} Poincare terms",
                level.modes,
                level.line_up.len(),
                level.order,
                level.circle.len(),
                level.upper.psi.len() - 1,
                level.lower.psi.len() - 1
            );
        }
        let mut initial: Vec<Complex> = match &taylor {
            Some(previous) => previous
                .iter()
                .take(level.order + 1)
                .map(|c| Complex::with_val_64(prec, c))
                .collect(),
            None => initial_taylor(&level.upper.ln_b, &level.centre, level.order, prec)?,
        };
        initial.resize(level.order + 1, cnum::zero(prec));
        let (step, norm) = level.solve(initial)?;
        let defect = level.defect(&step)?.max(&norm);
        let limit = cnum::epsilon(digits, prec);
        let fixed_points = (level.upper.l.clone(), level.lower.l.clone());
        if defect > limit {
            // The function is less regular than the sizes assume: enlarge them
            // by 3/2 for as long as that reduces the defect at least fourfold.
            let hopeless = coarse
                .as_ref()
                .is_some_and(|last| defect > Float::with_val_64(prec, last / 4u32));
            if hopeless {
                return Err(format!(
                    "consistency defect {:.3e} exceeds the internal target {:.3e}",
                    DisplayFloat(&defect),
                    DisplayFloat(&limit)
                ));
            }
            if cnum::verbose() {
                eprintln!(
                    "kneser-c: consistency defect {:.3e} above {:.3e}; enlarging the discretization",
                    DisplayFloat(&defect),
                    DisplayFloat(&limit)
                );
            }
            coarse = Some(defect);
            escalation += 1;
            taylor = Some(step.taylor);
            seeds = Some(fixed_points);
            continue;
        }
        coarse = None;
        if digits < target {
            taylor = Some(step.taylor);
            seeds = Some(fixed_points);
            digits = digits.saturating_mul(2).min(target);
            continue;
        }
        return Ok(ComplexKneserState {
            base: base.clone(),
            geometry: geometry.clone(),
            escalation,
            centre: level.centre.clone(),
            height: level.height.clone(),
            theta_up: step.theta_up,
            theta_dn: step.theta_dn,
            taylor: step.taylor,
            digits,
            prec,
            residual: defect,
            upper: level.upper,
            lower: level.lower,
        });
    }
}

fn geometries(ln_b: &Complex) -> Vec<KneserGeometry> {
    let shifted = KneserGeometry {
        centre: "-1".into(),
        radius: "0.7".into(),
        height: "0.3".into(),
    };
    let small = KneserGeometry {
        centre: "-0.5".into(),
        radius: "0.75".into(),
        height: "0.35".into(),
    };
    if *ln_b.real() > kneser::LARGE_BASE_LOG {
        vec![shifted, KneserGeometry::default()]
    } else {
        vec![KneserGeometry::default(), small, shifted]
    }
}

/// Polylines in the `ln b` plane from a real base to the target, in the order
/// to try. The Shell-Thron region lies in `-e <= Re <= 0.87`, `Im <= 2.11` of
/// the strip `0 <= Im <= pi`: a path goes straight up from a base above
/// `e^(1/e)` where `Re ln b >= 1`, straight from the cut base `e^-3.2` where
/// `Re ln b <= -3.2` (not every contour can start there), and through the
/// corridor above the region.
fn continuation_paths(ln_b: &Complex) -> Vec<Vec<Complex>> {
    let prec = SCOUT_BITS;
    let re = Float::with_val_64(prec, ln_b.real());
    let im = Float::with_val_64(prec, ln_b.imag());
    let point = |x: &Float, y: &Float| Complex::with_val_64(prec, (x, y));
    let zero = Float::new_64(prec);
    if re >= 1 {
        return vec![vec![point(&re, &zero), point(&re, &im)]];
    }
    let mut paths = Vec::new();
    let cut = cnum::decimal("-3.2", prec);
    if re <= cut {
        paths.push(vec![point(&cut, &zero), point(&re, &im)]);
    }
    let start = Float::with_val_64(prec, 1.5);
    let corridor = im.clone().max(&cnum::decimal("2.6", prec));
    let mut path = vec![
        point(&start, &zero),
        point(&start, &corridor),
        point(&re, &corridor),
    ];
    if corridor != im {
        path.push(point(&re, &im));
    }
    paths.push(path);
    paths
}

/// The first precision level at a real point of the `ln b` plane.
fn start_at_real(
    ln_b: &Complex,
    geometry: &KneserGeometry,
    first: u64,
) -> Result<ComplexKneserState, String> {
    let here = LogBase::Is(ln_b.clone());
    let direct = match solve(&here, geometry, first, None) {
        Ok(state) => return Ok(state),
        Err(error) => error,
    };
    if *ln_b.real() < 0 {
        return Err(format!("no start at the cut base e^-3.2: {direct}"));
    }
    // The real solver has its own continuation for large bases.
    let prec = cnum::checked_digits_to_bits(first + 8)?;
    let real = Complex::with_val_64(prec, (Float::with_val_64(prec, ln_b.real()).exp(), 0));
    let digits = first.saturating_sub(GUARD_DIGITS).max(1);
    let seed = kneser::setup_kneser_with_geometry(&real, prec, digits, geometry)?;
    let taylor = seed
        .taylor()
        .iter()
        .map(|c| Complex::with_val_64(prec, (c, 0)))
        .collect();
    let start = Start {
        taylor,
        digits: first,
        seeds: None,
        escalation: 0,
    };
    solve(&here, geometry, first, Some(start))
}

/// Reach the base along one of the `continuation_paths`.
fn continue_in_base(
    base: &LogBase,
    geometry: &KneserGeometry,
    target: u64,
) -> Result<ComplexKneserState, String> {
    let mut failures = Vec::new();
    for path in continuation_paths(&base.at(SCOUT_BITS)?) {
        match continue_along(base, geometry, target, &path) {
            Ok(state) => return Ok(state),
            Err(error) => failures.push(error),
        }
    }
    Err(failures.join("; "))
}

/// Walk a path in the `ln b` plane, each stage warm-started from its
/// predecessor at the first precision level.
fn continue_along(
    base: &LogBase,
    geometry: &KneserGeometry,
    target: u64,
    path: &[Complex],
) -> Result<ComplexKneserState, String> {
    let prec = SCOUT_BITS;
    let first = FIRST_LEVEL_DIGITS.min(target);
    let mut state = start_at_real(&path[0], geometry, first)?;
    for segment in path.windows(2) {
        let span = Complex::with_val_64(prec, &segment[1] - &segment[0]);
        let mut done = Float::new_64(prec);
        let mut stride = Float::with_val_64(prec, 1);
        // The solution one step back on this segment, for a secant predictor.
        let mut behind: Option<(Float, Vec<Complex>)> = None;
        while done < 1 {
            let next = Float::with_val_64(prec, &done + &stride).min(&Float::with_val_64(prec, 1));
            let here = Complex::with_val_64(prec, &span * &next) + &segment[0];
            let taylor = match &behind {
                Some((at, old)) => {
                    let ratio = Float::with_val_64(prec, &next - &done)
                        / Float::with_val_64(prec, &done - at);
                    state
                        .taylor
                        .iter()
                        .zip(old)
                        .map(|(new, old)| {
                            let work = new.prec_64().0;
                            Complex::with_val_64(work, new - old) * &ratio + new
                        })
                        .collect()
                }
                None => state.taylor.clone(),
            };
            let start = Start {
                taylor,
                digits: first,
                seeds: None,
                escalation: state.escalation,
            };
            match solve(&LogBase::Is(here.clone()), geometry, first, Some(start)) {
                Ok(solved) => {
                    if cnum::verbose() {
                        eprintln!(
                            "kneser-c: base continuation reached ln b = {:.6} + {:.6}i",
                            DisplayFloat(here.real()),
                            DisplayFloat(here.imag())
                        );
                    }
                    let passed = std::mem::replace(&mut state, solved);
                    behind = Some((done, passed.taylor));
                    done = next;
                    stride *= 2u32;
                }
                Err(error) => {
                    stride /= 2u32;
                    // A step below the resolution of the first level changes nothing.
                    let reach = Float::with_val_64(prec, &stride * cnum::abs(&span, prec));
                    if reach < cnum::epsilon(first, prec) {
                        return Err(format!(
                            "stalled before ln b = {:.6} + {:.6}i: {error}",
                            DisplayFloat(here.real()),
                            DisplayFloat(here.imag())
                        ));
                    }
                }
            }
        }
    }
    let start = Start {
        taylor: state.taylor,
        digits: first,
        seeds: None,
        escalation: state.escalation,
    };
    solve(base, geometry, target, Some(start))
}

fn describe(geometry: &KneserGeometry) -> String {
    format!(
        "contour {}/{}/{}",
        geometry.centre, geometry.radius, geometry.height
    )
}

/// Solve for the tetration of a base with `Im b >= 0` whose primary fixed
/// points both repel, at `digits` output digits.
pub fn setup_complex_kneser(
    b: &Complex,
    prec: u64,
    digits: u64,
) -> Result<ComplexKneserState, String> {
    crate::mt::init_pool()?;
    cnum::require_precision(prec, digits)?;
    let target = digits
        .checked_add(GUARD_DIGITS)
        .ok_or("precision count overflow")?;
    let base = LogBase::Of(b.clone());
    let candidates = geometries(&base.at(SCOUT_BITS)?);
    // Fail at once when the base is outside the scope of the construction.
    Level::new(
        &base,
        &candidates[0],
        FIRST_LEVEL_DIGITS.min(target),
        0,
        None,
    )?;
    let mut failures = Vec::new();
    for geometry in &candidates {
        match solve(&base, geometry, target, None) {
            Ok(state) => return Ok(state),
            Err(error) => failures.push(format!("{}: {error}", describe(geometry))),
        }
        if cnum::verbose() {
            eprintln!("kneser-c: {}", failures[failures.len() - 1]);
        }
    }
    for geometry in &candidates {
        match continue_in_base(&base, geometry, target) {
            Ok(state) => return Ok(state),
            Err(error) => failures.push(format!(
                "{} continued in the base: {error}",
                describe(geometry)
            )),
        }
        if cnum::verbose() {
            eprintln!("kneser-c: {}", failures[failures.len() - 1]);
        }
    }
    Err(format!(
        "two-sided theta mapping failed ({})",
        failures.join("; ")
    ))
}

/// As `setup_complex_kneser`, on one explicit contour; different contours must agree.
pub fn setup_complex_kneser_with_geometry(
    b: &Complex,
    prec: u64,
    digits: u64,
    geometry: &KneserGeometry,
) -> Result<ComplexKneserState, String> {
    crate::mt::init_pool()?;
    cnum::require_precision(prec, digits)?;
    let target = digits
        .checked_add(GUARD_DIGITS)
        .ok_or("precision count overflow")?;
    let base = LogBase::Of(b.clone());
    let direct = match solve(&base, geometry, target, None) {
        Ok(state) => return Ok(state),
        Err(error) => error,
    };
    continue_in_base(&base, geometry, target)
        .map_err(|error| format!("direct start: {direct}; base continuation: {error}"))
}

impl ComplexKneserState {
    /// The two fixed points approached as `Im z -> +inf` and `Im z -> -inf`.
    pub fn fixed_points(&self) -> (&Complex, &Complex) {
        (&self.upper.l, &self.lower.l)
    }

    /// The same base and contour at a higher internal accuracy.
    pub(crate) fn refine(&self, target: u64) -> Result<ComplexKneserState, String> {
        let start = Start {
            taylor: self.taylor.clone(),
            digits: self
                .digits
                .saturating_mul(2)
                .min(target)
                .max(self.digits.min(target)),
            seeds: Some((self.upper.l.clone(), self.lower.l.clone())),
            escalation: self.escalation,
        };
        solve(&self.base, &self.geometry, target, Some(start))
    }

    /// Branch numbers of the logarithms that take `F(u)` to `F(u - n)` for a
    /// point `u` of the central cell, by continuity along the vertical
    /// segment from the nearer theta line, where no logarithm is needed. The
    /// cuts of `F` therefore run along the real axis to the left of `-2`. The
    /// list ends early once the whole segment lies in the Poincare disk of
    /// the fixed point on its own branch: the rest is a contraction.
    fn scout(&self, argument: &Complex, steps: &Integer) -> Result<Vec<Integer>, String> {
        let prec = self.prec;
        let lower = argument.imag().is_sign_negative() && !argument.imag().is_zero();
        let (side, theta) = if lower {
            (&self.lower, &self.theta_dn)
        } else {
            (&self.upper, &self.theta_up)
        };
        let ln_b = Complex::with_val_64(prec, &self.upper.ln_b);
        let line = if lower {
            Float::with_val_64(prec, -&self.height)
        } else {
            Float::with_val_64(prec, &self.height)
        };
        let x = Float::with_val_64(prec, argument.real());
        let drop = Float::with_val_64(prec, argument.imag() - &line);
        let top = Complex::with_val_64(prec, (Float::with_val_64(prec, &x + &self.centre), &line));
        let value_at =
            |y: &Float| poly_value(&self.taylor, &Complex::with_val_64(prec, (&x, y)), prec).0;
        let log_of = |value: &Complex| -> Result<Complex, String> {
            if cnum::is_zero(value) {
                return Err("logarithm of zero while tracing a branch".into());
            }
            Ok(cnum::ln_complex(value, prec) / &ln_b)
        };
        // Ordinates from the theta line to the height, refined where two
        // neighbours are too far apart to fix a branch.
        let mut ordinates: Vec<Float> = (0..=8u32)
            .map(|s| Float::with_val_64(prec, &drop * s) / 8u32 + &line)
            .collect();
        // levels[m][i]: the value at ordinate i after m logarithms.
        let mut levels: Vec<Vec<Complex>> = vec![ordinates.iter().map(&value_at).collect()];
        let mut branches = Vec::new();
        let mut done = Integer::new();
        while done < *steps {
            done += 1;
            let z = Complex::with_val_64(prec, &top - &done);
            let (reference, _, _) = theta_side_value(side, theta, lower, &z, prec)?;
            let mut next = vec![reference];
            let mut branch = Integer::new();
            let mut index = 1;
            while index < ordinates.len() {
                let mut candidate = log_of(&levels[levels.len() - 1][index])?;
                match nearest_branch(&mut candidate, &next[index - 1], &ln_b, prec) {
                    Ok(found) => {
                        branch = found;
                        next.push(candidate);
                        index += 1;
                    }
                    Err(error) => {
                        // Add the midpoint of the two ordinates, with its history.
                        let middle =
                            Float::with_val_64(prec, &ordinates[index - 1] + &ordinates[index])
                                / 2u32;
                        if middle == ordinates[index - 1] || middle == ordinates[index] {
                            return Err(format!(
                                "could not trace the logarithm branch of the height: {error}"
                            ));
                        }
                        let mut value = value_at(&middle);
                        for (depth, level) in levels.iter_mut().enumerate() {
                            if depth > 0 {
                                value = log_of(&value)?;
                                nearest_branch(&mut value, &level[index - 1], &ln_b, prec)?;
                            }
                            level.insert(index, value.clone());
                        }
                        ordinates.insert(index, middle);
                    }
                }
            }
            let settled = branch == side.branch
                && next.iter().all(|w| {
                    cnum::abs(&Complex::with_val_64(prec, w - &side.l), prec) * 4u32
                        <= side.psi_radius
                });
            branches.push(branch);
            levels.push(next);
            if settled {
                break;
            }
        }
        Ok(branches)
    }

    /// `F(h)`, `F'(h)` and the logarithm of the factor by which the state's
    /// absolute accuracy is amplified in the relative error of `F(h)`.
    pub(crate) fn evaluate(&self, h: &Complex) -> Result<(Complex, Complex, Float), String> {
        let prec = self.prec;
        if !cnum::is_finite(h) {
            return Err("the height must be finite".into());
        }
        // The height is reduced exactly: one given to more digits than the
        // state carries must not round onto an integer.
        let exact = prec.max(h.real().prec_64()).max(h.imag().prec_64());
        let z = Complex::with_val_64(exact, h);
        let floor = Float::with_val_64(prec, -&self.height);
        let ln_b = &self.upper.ln_b;
        let (value, derivative, sensitivity) = if *z.imag() >= self.height {
            theta_side_value(&self.upper, &self.theta_up, false, &z, prec)?
        } else if *z.imag() <= floor {
            theta_side_value(&self.lower, &self.theta_dn, true, &z, prec)?
        } else {
            let offset = Float::with_val_64(exact, z.real() - &self.centre).round();
            let shift = offset
                .to_integer()
                .ok_or("the height shift is not finite")?;
            let argument = Complex::with_val_64(
                exact,
                (
                    Float::with_val_64(exact, z.real() - &self.centre) - &offset,
                    z.imag(),
                ),
            );
            let (mut value, mut derivative) = poly_value(&self.taylor, &argument, prec);
            let mut sensitivity = Float::with_val_64(prec, 1);
            let steps = shift.clone().abs();
            let mut done = Integer::new();
            if shift > 0 {
                let mut recurrence = cnum::Recurrence::<(Complex, Float)>::new();
                while done < steps {
                    let exponent = Complex::with_val_64(prec, &value * ln_b);
                    value = match cnum::exp_or_vanishing(&exponent, prec)
                        .map_err(|e| format!("height {}: {e}", DisplayFloat(z.real())))?
                    {
                        Some(value) => value,
                        // Below MPFR's range: the next level is b^0 = 1 at every precision.
                        None if Integer::from(&done + 1u32) < steps => cnum::zero(prec),
                        None => {
                            return Err(format!(
                                "height {}: {}",
                                DisplayFloat(z.real()),
                                cnum::VANISHING
                            ))
                        }
                    };
                    let factor = Complex::with_val_64(prec, &value * ln_b);
                    sensitivity *= cnum::abs(&factor, prec);
                    derivative *= factor;
                    done += 1;
                    let kept = || (derivative.clone(), sensitivity.clone());
                    if let Some((period, (slope, weight))) = recurrence.step(&value, kept) {
                        // The values repeat exactly: whole periods only scale the derivative.
                        let periods = Integer::from(&steps - &done) / &period;
                        if periods > 0 && !cnum::is_zero(&slope) && !weight.is_zero() {
                            let multiplier = Complex::with_val_64(prec, &derivative / &slope);
                            derivative *= multiplier.pow(&periods);
                            sensitivity *=
                                Float::with_val_64(prec, &sensitivity / &weight).pow(&periods);
                        }
                        done += periods * period;
                    }
                }
            } else if shift < 0 {
                let branches = self.scout(&argument, &steps)?;
                let turn = two_pi_i(prec);
                for branch in &branches {
                    if cnum::is_zero(&value) {
                        return Err(format!(
                            "integer height {} is undefined for tetration (would require log_b(0) and beyond)",
                            DisplayFloat(&Float::with_val_64(prec, z.real()).round())
                        ));
                    }
                    let factor = Complex::with_val_64(prec, &value * ln_b);
                    let mut next = cnum::ln_complex(&value, prec);
                    if *branch != 0 {
                        next += Complex::with_val_64(prec, &turn * branch);
                    }
                    value = next / ln_b;
                    sensitivity /= cnum::abs(&factor, prec);
                    derivative /= factor;
                    done += 1;
                }
                if done < steps {
                    // The scout stopped inside the Poincare disk of one fixed point.
                    let lower = argument.imag().is_sign_negative() && !argument.imag().is_zero();
                    let side = if lower { &self.lower } else { &self.upper };
                    let displacement = Complex::with_val_64(prec, &value - &side.l);
                    if cnum::abs(&displacement, prec) * 2u32 > side.psi_radius {
                        return Err("the height reduction lost the fixed-point branch".into());
                    }
                    let remaining = Integer::from(&steps - &done);
                    let (settled, factor) = side.contract(&displacement, &remaining)?;
                    value = settled;
                    sensitivity *= cnum::abs(&factor, prec);
                    derivative *= factor;
                }
            }
            (value, derivative, sensitivity)
        };
        if !cnum::is_finite(&value) || !cnum::is_finite(&derivative) || !sensitivity.is_finite() {
            return Err("the evaluation exceeded MPFR's range".into());
        }
        let zero = Float::new_64(prec);
        let log_amplification = if sensitivity.is_zero() {
            zero
        } else {
            (sensitivity.ln() - cnum::log_magnitude(&value, prec)).max(&zero)
        };
        Ok((value, derivative, log_amplification))
    }
}

/// Evaluate at `digits` output digits, refining the state when the height
/// consumes its accuracy margin.
pub fn eval_complex_kneser(
    state: &ComplexKneserState,
    h: &Complex,
    prec: u64,
    digits: u64,
) -> Result<Complex, String> {
    cnum::require_precision(prec, digits)?;
    if h.imag().is_zero() && h.real().is_integer() {
        if *h.real() <= -2 {
            return Err(format!(
                "integer height {} is undefined for tetration (would require log_b(0) and beyond)",
                DisplayFloat(h.real())
            ));
        }
        if *h.real() == -1 {
            return Ok(cnum::zero(prec));
        }
        if h.real().is_zero() {
            return Ok(cnum::one(prec));
        }
    }
    let mut refined: Option<ComplexKneserState> = None;
    loop {
        let current = refined.as_ref().unwrap_or(state);
        let (value, _, log_amplification) = current.evaluate(h)?;
        let decade = Float::with_val_64(log_amplification.prec_64(), 10).ln();
        let lost = (log_amplification / decade).ceil();
        let lost = lost
            .to_integer()
            .and_then(|lost| lost.to_u64())
            .ok_or("height conditioning exceeds the supported precision count")?;
        let needed = digits
            .checked_add(lost)
            .and_then(|sum| sum.checked_add(GUARD_DIGITS / 2))
            .ok_or("height conditioning exceeds the supported precision count")?;
        if needed <= current.digits {
            // Inputs parsed at `prec` stay adequate while their guard bits cover the loss.
            let conditioned = cnum::checked_digits_to_bits(digits + lost)?;
            let output = if lost <= 12 {
                prec
            } else {
                prec.max(conditioned)
            };
            return Ok(Complex::with_val_64(output, value));
        }
        if cnum::verbose() {
            eprintln!(
                "kneser-c height conditioning: refining from {} to {} internal digits",
                current.digits,
                needed + GUARD_DIGITS / 2
            );
        }
        refined = Some(current.refine(needed + GUARD_DIGITS / 2)?);
    }
}

pub fn tetrate_complex_kneser(
    b: &Complex,
    h: &Complex,
    prec: u64,
    digits: u64,
) -> Result<Complex, String> {
    let state = setup_complex_kneser(b, prec, digits)?;
    eval_complex_kneser(&state, h, prec, digits)
}
