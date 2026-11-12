//! Kneser tetration for real bases above `e^(1/e)` by theta-mapping iteration.
//!
//! With `a = ln b`, the upper fixed point `L` of `b^z`, `lambda = a*L` and
//! `kappa = Log(lambda)`, the entire Poincare function `psi`
//! (`psi(lambda*t) = L*(exp(a*psi(t)) - 1)`, `psi'(0) = 1`) gives the regular
//! superfunction `S(w) = L + psi(exp(kappa*w))`. The real-analytic tetration is
//! `F(z) = S(z + theta(z))` for `Im z > 0`, `theta` 1-periodic with only
//! non-negative Fourier modes, and `F(conj z) = conj F(z)`.
//!
//! `F` is stored as a real Taylor polynomial `P` at `c` for `|Im z| < y0` and as
//! `theta` for `|Im z| >= y0`. One step of the fixed-point map: sample
//! `theta = alpha(P(z)) - z` on `Im z = y0` (`alpha = S^-1`), drop negative
//! modes, rebuild `P` by Cauchy's formula on `|z - c| = R` (using `S(z+theta)`
//! above `y0` and the functional equation of the old `P` on the short arcs),
//! then shift so that `F(0) = 1`. Anderson mixing and a precision ladder
//! accelerate it. The acceptance gates are consistency checks, not a
//! certified forward-error bound.

use std::collections::VecDeque;

use rug::{float::Constant, Complex, Float, Integer};

use crate::{cnum, fft, lambertw};
use cnum::DisplayFloat;

pub(crate) const GUARD_DIGITS: u64 = 6;
pub(crate) const FIRST_LEVEL_DIGITS: u64 = 12;
pub(crate) const ANDERSON_DEPTH: usize = 10;
pub(crate) const LARGE_BASE_LOG: u32 = 16;

pub(crate) fn exp_complex(z: &Complex, prec: u64) -> Result<Complex, String> {
    let moderate = |x: &Float| {
        x.is_zero()
            || (x.is_finite()
                && x.get_exp().is_some_and(|e| {
                    e <= 20 && i64::from(e) >= -i64::try_from(prec).unwrap_or(i64::MAX)
                }))
    };
    if moderate(z.real()) && moderate(z.imag()) {
        return Ok(Complex::with_val_64(prec, z.exp_ref()));
    }
    cnum::checked_exp(z, prec)
}

fn poly_value(coefficients: &[Float], z: &Complex, prec: u64) -> Complex {
    let mut terms = coefficients.iter().rev();
    let mut value = match terms.next() {
        Some(leading) => Complex::with_val_64(prec, (leading, 0)),
        None => return cnum::zero(prec),
    };
    for coefficient in terms {
        value *= z;
        value += coefficient;
    }
    value
}

fn poly_value_derivative(coefficients: &[Float], z: &Complex, prec: u64) -> (Complex, Complex) {
    let mut terms = coefficients.iter().rev();
    let mut value = match terms.next() {
        Some(leading) => Complex::with_val_64(prec, (leading, 0)),
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

fn poly_real(coefficients: &[Float], x: &Float, prec: u64) -> (Float, Float) {
    let mut terms = coefficients.iter().rev();
    let mut value = match terms.next() {
        Some(leading) => Float::with_val_64(prec, leading),
        None => return (Float::new_64(prec), Float::new_64(prec)),
    };
    let mut derivative = Float::new_64(prec);
    for coefficient in terms {
        derivative *= x;
        derivative += &value;
        value *= x;
        value += coefficient;
    }
    (value, derivative)
}

pub(crate) fn nats(prec: u64) -> Float {
    Float::with_val_64(64, prec) * Float::with_val_64(64, Constant::Log2)
}

/// The base-dependent analytic data at one working precision.
#[derive(Clone)]
struct Model {
    prec: u64,
    ln_b: Float,
    l: Complex,
    kappa: Complex,
    psi: Vec<Complex>,
    psi_radius: Float,
}

impl Model {
    fn new(base: &Float, seed: Option<&Complex>, prec: u64) -> Result<Model, String> {
        cnum::check_precision(prec)?;
        let ln_b = Float::with_val_64(prec, base.ln_ref());
        let ln_b_complex = Complex::with_val_64(prec, (&ln_b, 0));
        let seed = match seed {
            Some(seed) => Complex::with_val_64(prec, seed),
            None => {
                let argument = Complex::with_val_64(prec, -&ln_b_complex);
                let w = lambertw::w0(&argument, prec)?;
                let l = cnum::checked_exp(&Complex::with_val_64(prec, -&w), prec)?;
                if l.imag().is_sign_negative() {
                    l.conj()
                } else {
                    l
                }
            }
        };
        let l = crate::kouznetsov::newton_fixed_point(&ln_b_complex, &seed, prec)?;
        if *l.imag() <= 0 {
            return Err("Kneser: no fixed point in the open upper half-plane".into());
        }
        let lambda = Complex::with_val_64(prec, &l * &ln_b);
        let kappa = cnum::ln_complex(&lambda, prec);
        if *kappa.real() <= 0 || *kappa.imag() <= 0 {
            return Err("Kneser: the upper fixed point is not repelling".into());
        }
        let gap = cnum::abs(&Complex::with_val_64(prec, &lambda - 1u32), prec);
        let psi_radius =
            (gap / Float::with_val_64(prec, &ln_b * 2u32)).min(&cnum::decimal("0.25", prec));
        if !psi_radius.is_finite() || psi_radius <= 0 {
            return Err("Kneser: degenerate Poincare radius".into());
        }
        let psi = Self::psi_series(&ln_b, &lambda, &psi_radius, prec)?;
        Ok(Model {
            prec,
            ln_b,
            l,
            kappa,
            psi,
            psi_radius,
        })
    }

    fn psi_series(
        ln_b: &Float,
        lambda: &Complex,
        radius: &Float,
        prec: u64,
    ) -> Result<Vec<Complex>, String> {
        let shift = usize::try_from(prec).map_err(|_| "Kneser precision is not addressable")?;
        let tolerance = Float::with_val_64(prec, radius) >> (shift + 8);
        let mut psi = vec![cnum::zero(prec), cnum::one(prec)];
        let mut powers = vec![cnum::one(prec), lambda.clone()];
        let mut radius_power = radius.clone();
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
                return Err(format!("Kneser: Poincare resonance at order {n}"));
            }
            let coefficient = Complex::with_val_64(prec, sum * ln_b) / denominator;
            if !cnum::is_finite(&coefficient) {
                return Err(format!("Kneser: Poincare coefficient {n} is not finite"));
            }
            radius_power *= radius;
            let term = cnum::abs(&coefficient, prec) * &radius_power;
            if term < tolerance {
                small += 1;
            } else {
                small = 0;
            }
            psi.push(coefficient);
            if small >= 3 && n >= 8 {
                return Ok(psi);
            }
        }
    }

    fn psi_value(&self, t: &Complex) -> (Complex, Complex) {
        self.psi_value_at(t, self.prec)
    }

    /// `psi(t)` and `psi'(t)` with the arithmetic rounded to `prec` bits.
    fn psi_value_at(&self, t: &Complex, prec: u64) -> (Complex, Complex) {
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
}

impl Model {
    /// `S(w)` and `S'(w)`.
    fn superfunction(&self, w: &Complex) -> Result<(Complex, Complex), String> {
        let prec = self.prec;
        let exponent = Complex::with_val_64(prec, &self.kappa * w);
        if !cnum::is_finite(&exponent) {
            return Err("Kneser: superfunction argument is not finite".into());
        }
        let excess = Float::with_val_64(prec, exponent.real() - self.psi_radius.clone().ln())
            / self.kappa.real();
        let steps = if excess > 0 {
            excess
                .ceil()
                .to_integer()
                .ok_or("Kneser: superfunction step count is not finite")?
        } else {
            Integer::new()
        };
        let turns = Complex::with_val_64(prec, &self.kappa * &steps);
        let reduced = Complex::with_val_64(prec, &exponent - &turns);
        if *reduced.real() < -(nats(prec) + 64u32) {
            return Ok((self.l.clone(), cnum::zero(prec)));
        }
        let t = exp_complex(&reduced, prec)?;
        let (psi, slope) = self.psi_value(&t);
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
            return Err("Kneser: superfunction value exceeds MPFR's range".into());
        }
        Ok((value, derivative))
    }

    /// The Abel coordinate `alpha(w)` with `S(alpha(w)) = w`, for `Im w > 0`.
    fn abel(&self, w: &Complex) -> Result<Complex, String> {
        let prec = self.prec;
        if !cnum::is_finite(w) || *w.imag() <= 0 {
            return Err(
                "Kneser: Abel coordinate needs a point in the open upper half-plane".into(),
            );
        }
        let target = Float::with_val_64(prec, &self.psi_radius / 4u32);
        let mut point = w.clone();
        let mut steps = Integer::new();
        let mut checkpoint = point.clone();
        let displacement = loop {
            let displacement = Complex::with_val_64(prec, &point - &self.l);
            if cnum::abs(&displacement, prec) <= target {
                break displacement;
            }
            point = cnum::ln_complex(&point, prec) / &self.ln_b;
            steps += 1;
            if !cnum::is_finite(&point) || point == checkpoint {
                return Err("Kneser: logarithmic orbit did not approach the fixed point".into());
            }
            if steps.is_power_of_two() {
                checkpoint = point.clone();
            }
        };
        let t = self.invert_psi(&displacement)?;
        Ok(cnum::ln_complex(&t, prec) / &self.kappa + steps)
    }

    /// The `t` with `psi(t) = displacement`, for a displacement well inside
    /// the Poincare disk.
    fn invert_psi(&self, displacement: &Complex) -> Result<Complex, String> {
        let prec = self.prec;
        if cnum::is_zero(displacement) {
            return Err("Kneser: Abel coordinate is singular at the fixed point".into());
        }
        // Newton for psi(t) = displacement: converge at low precision, then
        // one step per doubling of the precision.
        let mut work = prec.min(64);
        let mut t = Complex::with_val_64(work, displacement);
        let mut best: Option<(Float, Complex)> = None;
        loop {
            let (value, slope) = self.psi_value_at(&t, work);
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
                return Err("Kneser: Poincare inversion has a zero derivative".into());
            }
            let next = Complex::with_val_64(work, &t - residual / slope);
            best = Some((error, std::mem::replace(&mut t, next)));
        }
        while work < prec {
            work = work.saturating_mul(2).min(prec);
            let (value, slope) = self.psi_value_at(&t, work);
            if cnum::is_zero(&slope) {
                return Err("Kneser: Poincare inversion has a zero derivative".into());
            }
            let residual = Complex::with_val_64(work, value - displacement);
            t = Complex::with_val_64(work, &t - residual / slope);
        }
        let (value, _) = self.psi_value(&t);
        let error = cnum::abs(&Complex::with_val_64(prec, value - displacement), prec);
        let shift = usize::try_from(prec.saturating_sub(24)).map_err(|_| "precision overflow")?;
        if error > cnum::abs(displacement, prec) >> shift {
            return Err("Kneser: Poincare inversion missed working accuracy".into());
        }
        Ok(t)
    }

    /// `steps` further logarithms of a point at `L + displacement` inside the
    /// Poincare disk, in closed form: the new point and its derivative with
    /// respect to the old one.
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
        let (_, before) = self.psi_value(&t);
        let (psi, after) = self.psi_value(&Complex::with_val_64(prec, &t * &scale));
        if cnum::is_zero(&before) {
            return Err("Kneser: Poincare inversion has a zero derivative".into());
        }
        Ok((
            Complex::with_val_64(prec, &self.l + psi),
            scale * after / before,
        ))
    }

    fn period(&self) -> Complex {
        let two_pi = Float::with_val_64(self.prec, Constant::Pi) * 2u32;
        Complex::with_val_64(self.prec, (0, two_pi)) / &self.kappa
    }
}

/// `theta(z)` and `theta'(z)` from the Fourier coefficients.
pub(crate) fn theta_value(
    theta: &[Complex],
    z: &Complex,
    prec: u64,
) -> Result<(Complex, Complex), String> {
    let two_pi = Float::with_val_64(prec, Constant::Pi) * 2u32;
    let exponent = Complex::with_val_64(
        prec,
        (
            -Float::with_val_64(prec, z.imag() * &two_pi),
            Float::with_val_64(prec, z.real() * &two_pi),
        ),
    );
    if !cnum::is_finite(&exponent) {
        return Err("Kneser: theta argument is not finite".into());
    }
    let q = if *exponent.real() < -(nats(prec) + 64u32) {
        cnum::zero(prec)
    } else {
        exp_complex(&exponent, prec)?
    };
    let mut terms = theta.iter().rev();
    let mut value = terms.next().ok_or("Kneser: empty theta series")?.clone();
    let mut slope = cnum::zero(prec);
    for coefficient in terms {
        slope *= &q;
        slope += &value;
        value *= &q;
        value += coefficient;
    }
    let derivative = Complex::with_val_64(prec, (0, two_pi)) * q * slope;
    Ok((value, derivative))
}

/// Contour geometry: Taylor centre `c`, Cauchy radius `R`, junction height `y0`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KneserGeometry {
    pub centre: String,
    pub radius: String,
    pub height: String,
}

impl Default for KneserGeometry {
    fn default() -> Self {
        KneserGeometry {
            centre: "-0.5".into(),
            radius: "1".into(),
            height: "0.5".into(),
        }
    }
}

impl KneserGeometry {
    /// Large bases need a contour further left, where the values stay moderate.
    pub fn for_base(base: &Float) -> KneserGeometry {
        if *base > Float::with_val_64(64, LARGE_BASE_LOG).exp() {
            KneserGeometry {
                centre: "-1".into(),
                radius: "0.7".into(),
                height: "0.3".into(),
            }
        } else {
            KneserGeometry::default()
        }
    }
}

enum Arc {
    Upper(Complex),
    Right(Complex),
    Left(Complex),
}

struct Level {
    model: Model,
    digits: u64,
    centre: Float,
    height: Float,
    /// Normalization point relative to the centre and its value: F(0)=1 or F(-1)=0.
    anchor: (Float, u32),
    modes: usize,
    order: usize,
    samples: Vec<(Complex, Complex)>,
    theta_scale: Vec<Complex>,
    circle: Vec<Arc>,
    twist: Vec<Complex>,
    weights: Option<Vec<Float>>,
}

struct Step {
    taylor: Vec<Float>,
    theta: Vec<Complex>,
    negative: Float,
    shift: Float,
}

pub(crate) fn anchor_of(centre: &Float, prec: u64) -> (Float, u32) {
    if *centre < -0.5 {
        (-Float::with_val_64(prec, centre + 1u32), 0)
    } else {
        (Float::with_val_64(prec, -centre), 1)
    }
}

pub(crate) fn to_count(value: Float, what: &str) -> Result<usize, String> {
    cnum::checked_usize(&value.ceil()).ok_or_else(|| format!("Kneser: {what} is not addressable"))
}

impl Level {
    fn new(
        base: &Float,
        geometry: &KneserGeometry,
        digits: u64,
        seed: Option<&Complex>,
    ) -> Result<Level, String> {
        let prec = cnum::checked_digits_to_bits(
            digits
                .checked_add(8)
                .ok_or("Kneser: precision count overflow")?,
        )?;
        let model = Model::new(base, seed, prec)?;
        let centre = cnum::parse_float(&geometry.centre, prec)?;
        let radius = cnum::parse_float(&geometry.radius, prec)?;
        let height = cnum::parse_float(&geometry.height, prec)?;
        let one = Float::with_val_64(prec, 1);
        let half = Float::with_val_64(prec, 0.5);
        let reach = Float::with_val_64(prec, &centre + 2u32);
        if !(height > 0 && height < radius && radius < reach && centre > -2 && centre <= 0) {
            return Err("Kneser: unsupported contour geometry".into());
        }
        // Largest distance from the centre at which the Taylor polynomial is used.
        let cosine = (one.clone() - Float::with_val_64(prec, &height / &radius).square()).sqrt();
        let arc = (radius.clone().square() + 1u32
            - Float::with_val_64(prec, &radius * 2u32) * cosine)
            .sqrt();
        let segment = (height.clone().square() + 0.25f64).sqrt();
        let used = arc.max(&segment);
        if used >= reach || used > radius {
            return Err("Kneser: contour geometry leaves the Taylor disk".into());
        }
        let goal =
            Float::with_val_64(prec, digits + GUARD_DIGITS) * Float::with_val_64(prec, 10).ln();
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
            (sample_count as u128 + circle_count as u128 + order as u128) * 4,
            prec,
        )?;

        let samples = (0..sample_count)
            .map(|j| {
                let x =
                    Float::with_val_64(prec, j) / Float::with_val_64(prec, sample_count) - &half;
                let z =
                    Complex::with_val_64(prec, (Float::with_val_64(prec, &centre + &x), &height));
                (Complex::with_val_64(prec, (x, &height)), z)
            })
            .collect();
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
        let circle = (0..circle_count / 2)
            .map(|j| {
                let angle = Float::with_val_64(prec, &pi * Float::with_val_64(prec, 2 * j + 1))
                    / Float::with_val_64(prec, circle_count);
                let x = Float::with_val_64(prec, angle.cos_ref()) * &radius;
                let y = Float::with_val_64(prec, angle.sin_ref()) * &radius;
                if y >= height {
                    Arc::Upper(Complex::with_val_64(prec, (x + &centre, y)))
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
        let anchor = anchor_of(&centre, prec);
        Ok(Level {
            model,
            digits,
            centre,
            height,
            anchor,
            modes,
            order,
            samples,
            theta_scale,
            circle,
            twist,
            weights,
        })
    }
}

pub(crate) fn map_points<T: Sync, U: Send>(
    points: &[T],
    evaluate: impl Fn(&T) -> Result<U, String> + Sync + Send,
) -> Result<Vec<U>, String> {
    if crate::mt::mt_enabled() {
        use rayon::prelude::*;
        points.par_iter().map(evaluate).collect()
    } else {
        points.iter().map(evaluate).collect()
    }
}

/// Replace `P(z)` by `P(z + shift)`; the series is summed to working accuracy.
fn shift_taylor(taylor: &mut [Float], shift: &Float, prec: u64) -> Result<(), String> {
    if shift.is_zero() || taylor.len() < 2 {
        return Ok(());
    }
    let bits = usize::try_from(prec).map_err(|_| "Kneser precision is not addressable")?;
    let scale = taylor
        .iter()
        .fold(Float::new_64(prec), |scale, c| scale.max(&c.clone().abs()));
    let tolerance = scale >> (bits + 8);
    let mut term = taylor.to_vec();
    let mut small = 0u32;
    for k in 1..taylor.len() {
        let mut largest = Float::new_64(prec);
        for n in 0..taylor.len() - k {
            let mut next = Float::with_val_64(prec, &term[n + 1] * shift);
            next *= Integer::from(n + 1);
            next /= Integer::from(k);
            taylor[n] += &next;
            largest = largest.max(&next.clone().abs());
            term[n] = next;
        }
        if !largest.is_finite() {
            return Err("Kneser: Taylor shift exceeded MPFR's range".into());
        }
        small = if largest <= tolerance { small + 1 } else { 0 };
        if small >= 2 {
            break;
        }
    }
    Ok(())
}

impl Level {
    fn prec(&self) -> u64 {
        self.model.prec
    }

    /// One application of the fixed-point map.
    fn apply(&self, taylor: &[Float]) -> Result<Step, String> {
        let prec = self.prec();
        let model = &self.model;
        let mut values = map_points(&self.samples, |(argument, z)| {
            let alpha = model.abel(&poly_value(taylor, argument, prec))?;
            Ok(Complex::with_val_64(prec, alpha - z))
        })?;
        let period = model.period();
        for j in 1..values.len() {
            let gap = Complex::with_val_64(prec, &values[j - 1] - &values[j]) / &period;
            let turns = gap.real().clone().round();
            if !turns.is_zero() {
                values[j] += Complex::with_val_64(prec, &period * &turns);
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
        let mut theta: Vec<Complex> = values[..=self.modes]
            .iter()
            .zip(&self.theta_scale)
            .map(|(value, scale)| Complex::with_val_64(prec, value * scale))
            .collect();

        let upper = map_points(&self.circle, |arc| match arc {
            Arc::Upper(z) => {
                let (shift, _) = theta_value(&theta, z, prec)?;
                let (value, _) = model.superfunction(&Complex::with_val_64(prec, z + shift))?;
                Ok(value)
            }
            Arc::Right(argument) => {
                let value = poly_value(taylor, argument, prec);
                exp_complex(&Complex::with_val_64(prec, value * &model.ln_b), prec)
            }
            Arc::Left(argument) => {
                let value = poly_value(taylor, argument, prec);
                if cnum::is_zero(&value) {
                    return Err("Kneser: logarithm of zero on the Cauchy contour".into());
                }
                Ok(cnum::ln_complex(&value, prec) / &model.ln_b)
            }
        })?;
        if !upper.iter().all(cnum::is_finite) {
            return Err("Kneser: non-finite value on the Cauchy contour".into());
        }
        let mut contour: Vec<Complex> = upper.clone();
        contour.extend(upper.iter().rev().map(|value| value.clone().conj()));
        fft::fft(&mut contour, prec, false);
        let mut taylor: Vec<Float> = contour[..=self.order]
            .iter()
            .zip(&self.twist)
            .map(|(value, twist)| Complex::with_val_64(prec, value * twist).into_real_imag().0)
            .collect();

        let (origin, anchor_value) = &self.anchor;
        let mut root = origin.clone();
        let mut previous: Option<Float> = None;
        loop {
            let (value, slope) = poly_real(&taylor, &root, prec);
            let defect = value - *anchor_value;
            let size = defect.clone().abs();
            if size.is_zero() || previous.as_ref().is_some_and(|last| size >= *last) {
                break;
            }
            if !slope.is_finite() || slope <= 0 {
                return Err("Kneser: normalization lost monotonicity".into());
            }
            root -= defect / slope;
            if !root.is_finite() || Float::with_val_64(prec, &root - origin).abs() > 0.25 {
                return Err("Kneser: normalization root left the Taylor disk".into());
            }
            previous = Some(size);
        }
        let shift = root - origin;
        shift_taylor(&mut taylor, &shift, prec)?;
        // F(z) -> F(z + s) maps theta(z) to theta(z + s) + s.
        if !shift.is_zero() {
            let two_pi = Float::with_val_64(prec, Constant::Pi) * 2u32;
            let angle = Float::with_val_64(prec, &two_pi * &shift);
            let step = Complex::with_val_64(
                prec,
                (
                    Float::with_val_64(prec, angle.cos_ref()),
                    Float::with_val_64(prec, angle.sin_ref()),
                ),
            );
            let mut phase = cnum::one(prec);
            for coefficient in theta.iter_mut().skip(1) {
                phase *= &step;
                *coefficient *= &phase;
            }
            theta[0] += &shift;
        }
        Ok(Step {
            taylor,
            theta,
            negative,
            shift: shift.abs(),
        })
    }

    fn inner(&self, a: &[Float], b: &[Float]) -> Float {
        let prec = self.prec();
        let mut sum = Float::new_64(prec);
        match &self.weights {
            Some(weights) => {
                for ((x, y), w) in a.iter().zip(b).zip(weights) {
                    sum += Float::with_val_64(prec, x * y) * w;
                }
            }
            None => {
                for (x, y) in a.iter().zip(b) {
                    sum += Float::with_val_64(prec, x * y);
                }
            }
        }
        sum
    }
}

pub(crate) fn solve_linear(mut a: Vec<Vec<Float>>, mut b: Vec<Float>) -> Option<Vec<Float>> {
    let n = b.len();
    for column in 0..n {
        let pivot = (column..n).max_by(|&p, &q| {
            a[p][column]
                .clone()
                .abs()
                .partial_cmp(&a[q][column].clone().abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if a[pivot][column].is_zero() || !a[pivot][column].is_finite() {
            return None;
        }
        a.swap(column, pivot);
        b.swap(column, pivot);
        for row in column + 1..n {
            let factor = a[row][column].clone() / &a[column][column];
            if factor.is_zero() {
                continue;
            }
            let (upper, lower) = a.split_at_mut(row);
            for (target, source) in lower[0][column..].iter_mut().zip(&upper[column][column..]) {
                *target -= factor.clone() * source;
            }
            let delta = factor * &b[column];
            b[row] -= delta;
        }
    }
    for row in (0..n).rev() {
        for k in row + 1..n {
            let delta = a[row][k].clone() * &b[k];
            b[row] -= delta;
        }
        b[row] /= &a[row][row];
        if !b[row].is_finite() {
            return None;
        }
    }
    Some(b)
}

impl Level {
    /// Anderson-accelerated iteration to the level's fixed point.
    fn solve(&self, start: Vec<Float>) -> Result<(Step, Float), String> {
        let prec = self.prec();
        let tolerance = cnum::epsilon(self.digits + 3, prec);
        let mut x = start;
        let mut history: VecDeque<(Vec<Float>, Vec<Float>)> = VecDeque::new();
        let mut best: Option<(Float, Vec<Float>)> = None;
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
                            eprintln!("kneser: accelerated iterate rejected ({error}); plain step");
                        }
                        x = taylor.clone();
                        history.clear();
                        accelerated = false;
                        continue;
                    }
                    _ => return Err(error),
                },
            };
            let residual: Vec<Float> = step
                .taylor
                .iter()
                .zip(&x)
                .map(|(new, old)| Float::with_val_64(prec, new - old))
                .collect();
            let norm = self.inner(&residual, &residual).sqrt();
            if !norm.is_finite() {
                return Err("Kneser: fixed-point residual is not finite".into());
            }
            if cnum::verbose() {
                eprintln!(
                    "kneser: {} digits, iteration {iteration}: residual {:.3e}, negative modes {:.3e}, shift {:.3e}",
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
                        "Kneser iteration stalled at residual {:.3e} (target {:.3e})",
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
            let differences: Vec<Vec<Float>> = history
                .iter()
                .take(depth)
                .map(|(_, old)| {
                    residual
                        .iter()
                        .zip(old)
                        .map(|(new, old)| Float::with_val_64(prec, new - old))
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
                        *target -= Float::with_val_64(prec, new - old) * weight;
                    }
                }
                accelerated = true;
            }
        }
    }
}

fn initial_taylor(ln_b: &Float, centre: &Float, order: usize, prec: u64) -> Vec<Float> {
    // F(z) = expm1((z+1)k)/expm1(k), k = ln ln b: F(-1) = 0, F(0) = 1, F'(0) = ln(b) F'(-1).
    let rate = Float::with_val_64(prec, ln_b.ln_ref());
    let position = Float::with_val_64(prec, centre + 1u32);
    let mut taylor = Vec::with_capacity(order + 1);
    if rate.clone().abs() < 1e-6 {
        taylor.push(position);
        taylor.push(Float::with_val_64(prec, 1));
        taylor.resize(order + 1, Float::new_64(prec));
        return taylor;
    }
    let denominator = rate.clone().exp_m1();
    let argument = Float::with_val_64(prec, &position * &rate);
    taylor.push(argument.clone().exp_m1() / &denominator);
    let mut term = argument.exp() / denominator;
    for n in 1..=order {
        term *= &rate;
        term /= Integer::from(n);
        taylor.push(term.clone());
    }
    taylor
}

/// Converged tetration data for one real base above `e^(1/e)`.
#[derive(Clone)]
pub struct KneserState {
    model: Model,
    base: Float,
    geometry: KneserGeometry,
    centre: Float,
    height: Float,
    theta: Vec<Complex>,
    taylor: Vec<Float>,
    /// Internal accuracy goal in decimal digits; the gates passed at this level.
    pub digits: u64,
    pub prec: u64,
    /// Largest accepted consistency defect, not a forward-error bound.
    pub residual: Float,
}

/// A warm start: Taylor data, the first level to solve, and a fixed-point seed
/// (only meaningful for the same base).
struct Start {
    taylor: Vec<Float>,
    digits: u64,
    seed: Option<Complex>,
}

impl Start {
    fn refine(state: &KneserState, target: u64) -> Start {
        Start {
            taylor: state.taylor.clone(),
            digits: state
                .digits
                .saturating_mul(2)
                .min(target)
                .max(state.digits.min(target)),
            seed: Some(state.model.l.clone()),
        }
    }
}

fn solve(
    base: &Float,
    geometry: &KneserGeometry,
    target: u64,
    start: Option<Start>,
) -> Result<KneserState, String> {
    let (mut digits, mut taylor, mut seed) = match start {
        Some(start) => (start.digits.min(target), Some(start.taylor), start.seed),
        None => (FIRST_LEVEL_DIGITS.min(target), None, None),
    };
    loop {
        let level = Level::new(base, geometry, digits, seed.as_ref())?;
        let prec = level.prec();
        if cnum::verbose() {
            eprintln!(
                "kneser: level {digits} digits ({prec} bits): {} theta modes, {} samples, Taylor order {}, {} contour nodes, {} Poincare terms",
                level.modes,
                level.samples.len(),
                level.order,
                level.circle.len() * 2,
                level.model.psi.len() - 1
            );
        }
        let mut start: Vec<Float> = match &taylor {
            Some(previous) => previous
                .iter()
                .take(level.order + 1)
                .map(|c| Float::with_val_64(prec, c))
                .collect(),
            None => initial_taylor(&level.model.ln_b, &level.centre, level.order, prec),
        };
        start.resize(level.order + 1, Float::new_64(prec));
        let (step, norm) = level.solve(start)?;
        if digits < target {
            taylor = Some(step.taylor);
            seed = Some(level.model.l.clone());
            digits = digits.saturating_mul(2).min(target);
            continue;
        }
        let mut state = KneserState {
            base: Float::with_val_64(prec, base),
            geometry: geometry.clone(),
            centre: level.centre.clone(),
            height: level.height.clone(),
            theta: step.theta,
            taylor: step.taylor,
            digits,
            prec,
            residual: norm.max(&step.negative),
            model: level.model,
        };
        state.check_gates()?;
        return Ok(state);
    }
}

impl KneserState {
    fn check_gates(&mut self) -> Result<(), String> {
        let prec = self.prec;
        let mut worst = self.residual.clone();
        let record = |name: &str, defect: Float, worst: &mut Float| {
            if cnum::verbose() {
                eprintln!("kneser: gate {name}: {:.3e}", DisplayFloat(&defect));
            }
            *worst = worst.clone().max(&defect);
        };
        for quarter in -2i32..=2 {
            let x = Float::with_val_64(prec, quarter) / 4u32;
            let argument = Complex::with_val_64(prec, (&x, &self.height));
            let z = Complex::with_val_64(prec, (x + &self.centre, &self.height));
            let (shift, _) = theta_value(&self.theta, &z, prec)?;
            let (upper, _) = self
                .model
                .superfunction(&Complex::with_val_64(prec, &z + shift))?;
            let lower = poly_value(&self.taylor, &argument, prec);
            let scale = cnum::abs(&upper, prec).max(&Float::with_val_64(prec, 1));
            let defect = cnum::abs(&Complex::with_val_64(prec, upper - lower), prec) / scale;
            record("junction", defect, &mut worst);
        }
        let half = Float::with_val_64(prec, 0.5);
        let (left, _) = poly_real(&self.taylor, &Float::with_val_64(prec, -&half), prec);
        let (right, _) = poly_real(&self.taylor, &half, prec);
        let image = Float::with_val_64(prec, &left * &self.model.ln_b).exp();
        let scale = right.clone().abs().max(&Float::with_val_64(prec, 1));
        record(
            "functional equation",
            Float::with_val_64(prec, &right - &image).abs() / scale,
            &mut worst,
        );
        let (anchor, anchor_value) = anchor_of(&self.centre, prec);
        let (origin, _) = poly_real(&self.taylor, &anchor, prec);
        record("normalization", (origin - anchor_value).abs(), &mut worst);
        let limit = cnum::epsilon(self.digits, prec);
        if !worst.is_finite() || worst > limit {
            return Err(format!(
                "Kneser consistency defect {:.3e} exceeds the internal target {:.3e}",
                DisplayFloat(&worst),
                DisplayFloat(&limit)
            ));
        }
        self.residual = worst;
        Ok(())
    }
}

impl KneserState {
    pub fn base(&self) -> &Float {
        &self.base
    }

    /// `F(h)`, `F'(h)` and the logarithm of the factor by which the state's
    /// absolute accuracy is amplified in the relative error of `F(h)`.
    pub(crate) fn evaluate(&self, h: &Complex) -> Result<(Complex, Complex, Float), String> {
        let prec = self.prec;
        if !cnum::is_finite(h) {
            return Err("Kneser height must be finite".into());
        }
        let lower = h.imag().is_sign_negative() && !h.imag().is_zero();
        // The height is reduced exactly: one given to more digits than the
        // state carries must not round onto an integer.
        let exact = prec.max(h.real().prec_64()).max(h.imag().prec_64());
        let z = if lower {
            Complex::with_val_64(exact, h.conj_ref())
        } else {
            Complex::with_val_64(exact, h)
        };
        let (mut value, mut derivative, sensitivity) = if *z.imag() >= self.height {
            let (shift, slope) = theta_value(&self.theta, &z, prec)?;
            let (value, rate) = self
                .model
                .superfunction(&Complex::with_val_64(prec, &z + shift))?;
            let sensitivity = cnum::abs(&rate, prec);
            (value, rate * (slope + 1u32), sensitivity)
        } else {
            let offset = Float::with_val_64(exact, z.real() - &self.centre).round();
            let shift = offset
                .to_integer()
                .ok_or("Kneser: height shift is not finite")?;
            let argument = Complex::with_val_64(
                exact,
                (
                    Float::with_val_64(exact, z.real() - &self.centre) - &offset,
                    z.imag(),
                ),
            );
            let (mut value, mut derivative) = poly_value_derivative(&self.taylor, &argument, prec);
            let mut sensitivity = Float::with_val_64(prec, 1);
            let mut done = Integer::new();
            let steps = shift.clone().abs();
            while done < steps {
                if shift > 0 {
                    value = cnum::checked_exp(
                        &Complex::with_val_64(prec, &value * &self.model.ln_b),
                        prec,
                    )
                    .map_err(|e| format!("Kneser height {}: {e}", DisplayFloat(z.real())))?;
                    let factor = Complex::with_val_64(prec, &value * &self.model.ln_b);
                    sensitivity *= cnum::abs(&factor, prec);
                    derivative *= factor;
                } else {
                    if cnum::is_zero(&value) {
                        return Err(format!(
                            "integer height {} is undefined for tetration (would require log_b(0) and beyond)",
                            DisplayFloat(&Float::with_val_64(prec, z.real()).round())
                        ));
                    }
                    if value.imag().is_zero() {
                        // The principal logarithm of a negative real has argument +pi.
                        value.mut_imag().abs_mut();
                    }
                    let factor = Complex::with_val_64(prec, &value * &self.model.ln_b);
                    let next = cnum::ln_complex(&value, prec) / &self.model.ln_b;
                    sensitivity /= cnum::abs(&factor, prec);
                    derivative /= factor;
                    if next == value {
                        break;
                    }
                    value = next;
                }
                done += 1;
                if shift < 0 && done < steps {
                    // Near the fixed point the remaining logarithms have a closed form.
                    let displacement = Complex::with_val_64(prec, &value - &self.model.l);
                    if cnum::abs(&displacement, prec) * 4u32 <= self.model.psi_radius {
                        let remaining = Integer::from(&steps - &done);
                        let (settled, factor) = self.model.contract(&displacement, &remaining)?;
                        value = settled;
                        sensitivity *= cnum::abs(&factor, prec);
                        derivative *= factor;
                        break;
                    }
                }
            }
            (value, derivative, sensitivity)
        };
        if !cnum::is_finite(&value) || !cnum::is_finite(&derivative) || !sensitivity.is_finite() {
            return Err("Kneser evaluation exceeded MPFR's range".into());
        }
        if lower {
            value = value.conj();
            derivative = derivative.conj();
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

impl KneserState {
    /// The same base and contour at a higher internal accuracy.
    pub(crate) fn refine(&self, target: u64) -> Result<KneserState, String> {
        solve(
            &self.base,
            &self.geometry,
            target,
            Some(Start::refine(self, target)),
        )
    }

    pub(crate) fn fixed_point(&self) -> &Complex {
        &self.model.l
    }

    pub(crate) fn taylor(&self) -> &[Float] {
        &self.taylor
    }
}

fn real_base(b: &Complex, prec: u64) -> Result<Float, String> {
    if !cnum::is_finite(b) || !b.imag().is_zero() || *b.real() <= cnum::eta_upper(prec) {
        return Err("Kneser construction requires a real base above e^(1/e)".into());
    }
    Ok(Float::with_val_64(prec, b.real()))
}

/// Solve for the tetration of a real base `b > e^(1/e)` at `digits` output digits.
pub fn setup_kneser(b: &Complex, prec: u64, digits: u64) -> Result<KneserState, String> {
    setup_kneser_with_geometry(b, prec, digits, &KneserGeometry::for_base(b.real()))
}

/// As `setup_kneser`, on an explicit contour; different contours must agree.
pub fn setup_kneser_with_geometry(
    b: &Complex,
    prec: u64,
    digits: u64,
    geometry: &KneserGeometry,
) -> Result<KneserState, String> {
    crate::mt::init_pool()?;
    cnum::require_precision(prec, digits)?;
    let base = real_base(b, prec)?;
    let target = digits
        .checked_add(GUARD_DIGITS)
        .ok_or("Kneser: precision count overflow")?;
    let direct = match solve(&base, geometry, target, None) {
        Ok(state) => return Ok(state),
        Err(error) => error,
    };
    if cnum::verbose() {
        eprintln!("kneser: direct start failed ({direct}); continuing in the base");
    }
    continue_in_base(&base, geometry, target)
        .map_err(|error| format!("Kneser: direct start: {direct}; base continuation: {error}"))
}

/// Reach a large base through a chain of smaller ones, each warm-started
/// from its predecessor at the first precision level.
fn continue_in_base(
    base: &Float,
    geometry: &KneserGeometry,
    target: u64,
) -> Result<KneserState, String> {
    let prec = base.prec_64();
    let first = FIRST_LEVEL_DIGITS.min(target);
    let goal = Float::with_val_64(prec, base.ln_ref());
    let mut rate = goal.clone();
    while rate > LARGE_BASE_LOG {
        rate /= 2u32;
    }
    let mut state = solve(&rate.clone().exp(), geometry, first, None)?;
    let mut factor = Float::with_val_64(prec, 2);
    while rate < goal {
        let next = Float::with_val_64(prec, &rate * &factor).min(&goal);
        let start = Start {
            taylor: state.taylor.clone(),
            digits: first,
            seed: None,
        };
        match solve(&next.clone().exp(), geometry, first, Some(start)) {
            Ok(solved) => {
                if cnum::verbose() {
                    eprintln!(
                        "kneser: base continuation reached ln b = {:.6}",
                        DisplayFloat(&next)
                    );
                }
                state = solved;
                rate = next;
            }
            Err(error) => {
                factor = factor.sqrt();
                // A step below the resolution of the first level changes nothing.
                if Float::with_val_64(prec, &factor - 1u32) < cnum::epsilon(first, prec) {
                    return Err(format!(
                        "stalled at ln b = {:.6}: {error}",
                        DisplayFloat(&rate)
                    ));
                }
            }
        }
    }
    let start = Start {
        taylor: state.taylor,
        digits: first,
        seed: None,
    };
    solve(base, geometry, target, Some(start))
}

/// Evaluate at `digits` output digits, refining the state when the height
/// consumes its accuracy margin.
pub fn eval_kneser(
    state: &KneserState,
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
    let mut refined: Option<KneserState> = None;
    loop {
        let current = refined.as_ref().unwrap_or(state);
        let (value, _, log_amplification) = current.evaluate(h)?;
        let lost = (log_amplification / Float::with_val_64(64, 10).ln()).ceil();
        let lost = lost
            .to_integer()
            .and_then(|lost| lost.to_u64())
            .ok_or("Kneser: height conditioning exceeds the supported precision count")?;
        let needed = digits
            .checked_add(lost)
            .and_then(|sum| sum.checked_add(GUARD_DIGITS / 2))
            .ok_or("Kneser: height conditioning exceeds the supported precision count")?;
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
                "kneser height conditioning: refining from {} to {} internal digits",
                current.digits,
                needed + GUARD_DIGITS / 2
            );
        }
        refined = Some(current.refine(needed + GUARD_DIGITS / 2)?);
    }
}

pub fn tetrate_kneser(b: &Complex, h: &Complex, prec: u64, digits: u64) -> Result<Complex, String> {
    let state = setup_kneser(b, prec, digits)?;
    eval_kneser(&state, h, prec, digits)
}
