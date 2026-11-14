//! The base `e^(1/e)` itself: tetration at a parabolic fixed point.
//!
//! For `b = e^(1/e)` the two real fixed points of `b^z` have met at `e`, and
//! the multiplier there is 1: no Schroder function exists. With
//! `z = e (1 + u)` the map `b^z` is `u -> e^u - 1 = u + u^2/2 + ...`, and the
//! orbit of 1 approaches the fixed point along the attracting direction
//! `u < 0`, like `-2/n`. The Fatou coordinate `alpha` of that petal (the Abel
//! function, `alpha(e^u - 1) = alpha(u) + 1`) takes the place of the
//! logarithm of the Schroder function:
//!
//! `F(h) = e (1 + psi(h + alpha(u0)))`, `psi = alpha^-1`, `u0 = 1/e - 1`.
//!
//! It is the limit of the regular iteration of the bases below `e^(1/e)` and
//! of the Kneser tetration of the bases above it.
//!
//! The coordinate is `alpha(u) = lim (A(u_n) - n)` along the orbit `u_n` of
//! `u`, for `A(u) = -2/u + ln(-u)/3 + c_1 u + ... + c_N u^N`, the first terms
//! of the formal solution of the Abel equation. That series diverges (its
//! coefficients grow like `(k - 1)! / (4 pi)^k`), but the limit exists for
//! every `N`, and what a finite `n` leaves of it is the sum of the defects
//! `A(e^u - 1) - A(u) - 1` over the rest of the orbit. The defect is an
//! analytic function of `u` that begins with `u^(N + 2)`: it is measured on
//! the circle `|u| = r` on which the series is used, which bounds it inside,
//! and along an orbit it falls like `|u|^(N + 2)`.

use std::sync::{Arc, Mutex};

use rug::{float::Constant, integer::IntegerExt64, Assign, Complex, Float, Integer};

use crate::{cnum, constants};
use cnum::DisplayFloat;

/// `e^(1/e)`, correctly rounded to `prec` bits.
pub(crate) fn eta(prec: u64) -> Result<Float, String> {
    constants::named("eta")
        .ok_or("the constant eta is not defined")?
        .value(prec)
}

/// Whether a base is `e^(1/e)` itself, for a result of `digits` digits: a
/// real number that is `e^(1/e)` correctly rounded to the precision that it
/// has, or to the bits that it carries where it was given more room later.
/// Those must be 32 bits more than the digits, and 64 at least. Such a base
/// is the number `e^(1/e)`, not its rounding; the two have the same
/// requested digits up to heights of about `10^9` (the value is
/// differentiable in the base there, with a slope of 1.7 times a large
/// height). A decimal number is read with 64 bits more than its digits and
/// is never that close; the constant and the expressions for it are.
pub(crate) fn is_eta(b: &Complex, digits: u64) -> bool {
    let x = b.real();
    if !b.imag().is_zero() || !x.is_normal() || x.is_sign_negative() || cnum::exponent(x) != Some(1)
    {
        return false;
    }
    let Some((mantissa, _)) = x.to_integer_exp() else {
        return false;
    };
    let Some(low) = mantissa.find_one_64(0) else {
        return false;
    };
    // 3.32193 is an upper bound on log2(10), as in `cnum::digits_to_bits`.
    let least = u64::try_from((u128::from(digits) * 332_193).div_ceil(100_000) + 32)
        .unwrap_or(u64::MAX)
        .max(64);
    let carried = mantissa.significant_bits_64() - low;
    [x.prec_64(), carried]
        .into_iter()
        .any(|bits| bits >= least && matches!(eta(bits), Ok(value) if value == *x))
}

/// `e^u - 1` without the cancellation of a small `u`. `None` where the
/// imaginary part of `u` is so large that `prec` bits do not hold its
/// place on the circle (the sine of such a number takes its whole length),
/// unless the real part makes `e^u` vanish next to 1 whatever that place.
fn exp_m1(u: &Complex, prec: u64) -> Option<Complex> {
    if u.imag().is_zero() {
        let real = Float::with_val_64(prec, u.real().exp_m1_ref());
        return Some(Complex::with_val_64(prec, (real, 0)));
    }
    // `e^Re u < 2^-(prec + 1)`, from `ln 2 > 1/2`.
    if *u.real() < -Float::with_val_64(prec, prec) - 1u32 {
        return Some(Complex::with_val_64(prec, (-1, 0)));
    }
    let unresolved =
        cnum::exponent(u.imag()).is_some_and(|exponent| i128::from(exponent) > i128::from(prec));
    if unresolved {
        return None;
    }
    let (sine, cosine) = Float::with_val_64(prec, u.imag()).sin_cos(Float::new_64(prec));
    let half = Float::with_val_64(prec, u.imag() / 2u32).sin();
    let grown = Float::with_val_64(prec, u.real().exp_ref());
    let real = Float::with_val_64(prec, u.real().exp_m1_ref()) * cosine - half.square() * 2u32;
    Some(Complex::with_val_64(prec, (real, grown * sine)))
}

/// The principal `ln(1 + u)` without the cancellation of a small `u`, with
/// the argument `pi` on the cut; `None` at `u = -1`.
fn ln_1p(u: &Complex, prec: u64) -> Option<Complex> {
    let (x, y) = (u.real(), u.imag());
    let shifted = Float::with_val_64(prec, x + 1u32);
    if y.is_zero() {
        if shifted.is_zero() {
            return None;
        }
        return Some(if shifted.is_sign_positive() {
            Complex::with_val_64(prec, (Float::with_val_64(prec, x.ln_1p_ref()), 0))
        } else {
            let pi = Float::with_val_64(prec, Constant::Pi);
            Complex::with_val_64(prec, (shifted.abs().ln(), pi))
        });
    }
    let small = |part: &Float| Float::with_val_64(prec, part * 2u32).abs() <= 1u32;
    let real = if small(x) && small(y) {
        // `|1 + u|^2 - 1`.
        let excess = Float::with_val_64(prec, x * 2u32)
            + Float::with_val_64(prec, x.square_ref())
            + Float::with_val_64(prec, y.square_ref());
        excess.ln_1p() / 2u32
    } else {
        let norm = Float::with_val_64(prec, shifted.square_ref())
            + Float::with_val_64(prec, y.square_ref());
        norm.ln() / 2u32
    };
    let imaginary = Float::with_val_64(prec, y.atan2_ref(&shifted));
    Some(Complex::with_val_64(prec, (real, imaginary)))
}

/// The coefficients `c_k` of the formal Fatou coordinate
/// `-2/u + ln(-u)/3 + sum c_k u^k` of `u -> e^u - 1`, one after another.
///
/// With `A(u) = sum c_k u^k` the Abel equation is `A(e^u - 1) - A(u) = R(u)`,
/// `R(u) = 1 + 2/(e^u - 1) - 2/u - ln((e^u - 1)/u)/3`, whose coefficient of
/// `u^n` is `2 beta_(n+1) - beta_n/(3n)` for `beta_j = B_j/j!`, the
/// coefficients of `u/(e^u - 1)`. The coefficient of `u^n` in `(e^u - 1)^k`
/// is `t(k, n)`, with `(n + 1) t(k, n + 1) = k (t(k - 1, n) + t(k, n))`: one
/// column `n` is kept. The coefficient of `u^n` of the equation then gives
/// `c_(n-1)`, because `t(n, n) = 1` and `t(n - 1, n) = (n - 1)/2`.
struct Recursion {
    work: u64,
    /// `1/j!`.
    factorials: Vec<Float>,
    /// `beta_j = B_j/j!`.
    bernoulli: Vec<Float>,
    /// `t(k, n)` for `k = 0..=n`.
    column: Vec<Float>,
    /// `c_1, c_2, ...`.
    coefficients: Vec<Float>,
}

impl Recursion {
    fn new(work: u64) -> Recursion {
        let one = Float::with_val_64(work, 1);
        Recursion {
            work,
            factorials: vec![one.clone(), one.clone()],
            bernoulli: vec![one.clone()],
            column: vec![Float::new_64(work), one],
            coefficients: Vec::new(),
        }
    }

    /// `beta_j` for `j <= index`, each from its tangent number `T_k` (the
    /// integers of `tan x = sum T_k x^(2k-1)/(2k-1)!`, by the recurrence of
    /// Brent and Harvey, whose steps multiply by small numbers):
    /// `B_2k = (-1)^(k-1) 2k T_k / (4^k (4^k - 1))`. The list is computed
    /// anew, twice as long, when it runs out.
    fn bernoulli(&mut self, index: usize) {
        if self.bernoulli.len() > index {
            return;
        }
        let count = (index / 2 + 1).max(64).max(self.bernoulli.len());
        let mut tangent = vec![Integer::new(); count + 1];
        tangent[1] = Integer::from(1);
        for k in 2..=count {
            tangent[k] = Integer::from(&tangent[k - 1] * (k as u64 - 1));
        }
        for k in 2..=count {
            for j in k..=count {
                let below = Integer::from(&tangent[j - 1] * ((j - k) as u64));
                tangent[j] *= (j - k + 2) as u64;
                tangent[j] += below;
            }
        }
        while self.factorials.len() < 2 * count {
            let j = self.factorials.len() as u64;
            let last = self.factorials.last().expect("1/0! is there");
            self.factorials
                .push(Float::with_val_64(self.work, last / j));
        }
        let one = Float::with_val_64(self.work, 1);
        self.bernoulli = vec![one.clone(), -one / 2u32];
        for (k, number) in tangent.iter().enumerate().skip(1) {
            let power = Integer::from(1) << (2 * k);
            let denominator = Integer::from(&power - 1u32) * power;
            let value =
                Float::with_val_64(self.work, number) * &self.factorials[2 * k - 1] / denominator;
            self.bernoulli.push(if k % 2 == 0 { -value } else { value });
            self.bernoulli.push(Float::new_64(self.work));
        }
    }

    /// The coefficients up to `c_count`.
    fn extend(&mut self, count: usize) {
        while self.coefficients.len() < count {
            // The column of `n`.
            let n = self.column.len();
            self.column.push(Float::new_64(self.work));
            for k in (1..=n).rev() {
                let sum = Float::with_val_64(self.work, &self.column[k - 1] + &self.column[k]);
                self.column[k] = sum * (k as u64) / (n as u64);
            }
            self.bernoulli(n + 1);
            let right = Float::with_val_64(self.work, &self.bernoulli[n + 1] * 2u32)
                - Float::with_val_64(self.work, &self.bernoulli[n] / (3 * n as u64));
            // One rounded product a term: the guard bits cover the sum, and
            // an exact dot product takes more than twice as long.
            let mut known = Float::new_64(self.work);
            let mut product = Float::new_64(self.work);
            for (coefficient, entry) in self.coefficients[..n - 2].iter().zip(&self.column[1..]) {
                product.assign(coefficient * entry);
                known += &product;
            }
            let next = (right - known) / &self.column[n - 1];
            self.coefficients.push(next);
        }
    }
}

/// The Fatou coordinate of the attracting petal of `u -> e^u - 1`, for
/// values that are correct to `prec` bits.
pub(crate) struct Fatou {
    prec: u64,
    /// The precision of the coefficients and of every evaluation.
    work: u64,
    /// `c_1, ..., c_N`.
    coefficients: Vec<Float>,
    /// The series is used for `|u| <= radius`.
    radius: Float,
    /// `2/radius + 2`: where the coordinate `-2/u` of such a point lies.
    reach: Float,
    /// `alpha(1/e - 1)`: the coordinate of the height 0.
    anchor: Complex,
    e: Float,
}

/// The states built last, by the precision that they serve.
static STATES: Mutex<Vec<Arc<Fatou>>> = Mutex::new(Vec::new());

impl Fatou {
    /// The state for values correct to `prec` bits.
    pub(crate) fn at(prec: u64) -> Result<Arc<Fatou>, String> {
        let kept = |states: &Vec<Arc<Fatou>>| states.iter().find(|s| s.prec == prec).cloned();
        if let Some(state) = STATES.lock().ok().and_then(|states| kept(&states)) {
            return Ok(state);
        }
        let state = Arc::new(Fatou::build(prec)?);
        if let Ok(mut states) = STATES.lock() {
            if states.len() == 4 {
                states.remove(0);
            }
            states.push(state.clone());
        }
        Ok(state)
    }

    fn build(prec: u64) -> Result<Fatou, String> {
        cnum::init_mpfr();
        let work = prec.checked_add(64).ok_or("precision count overflow")?;
        cnum::check_precision(work)?;
        let shift = usize::try_from(prec).map_err(|_| "precision overflow")?;
        let goal = Float::with_val_64(work, 1) >> shift;
        // The smallest term of the series at `|u| = r` is about
        // `exp(-4 pi / r)`. A radius below the largest that reaches the
        // goal needs fewer terms and a longer orbit: the terms cost their
        // square, so the radius shrinks as the precision grows.
        let four_pi = Float::with_val_64(work, Constant::Pi) * 4u32;
        let bits = prec.checked_add(32).ok_or("precision count overflow")?;
        let stretch = Float::with_val_64(work, prec) / 4096u32 + Float::with_val_64(work, 3) / 2u32;
        let mut radius = Float::with_val_64(work, &four_pi / bits)
            / Float::with_val_64(work, Constant::Log2)
            / stretch;
        let mut recursion = Recursion::new(work);
        let mut state = Fatou {
            prec,
            work,
            coefficients: Vec::new(),
            radius: radius.clone(),
            reach: Float::new_64(work),
            anchor: cnum::zero(work),
            e: Float::with_val_64(work, 1).exp(),
        };
        for _attempt in 0..64 {
            let reach = Float::with_val_64(work, 2u32 / &radius) + 2u32;
            // What is left of a coordinate is at most `2 reach` defects.
            let budget = Float::with_val_64(work, &goal / &reach) / 2u32;
            let last = Float::with_val_64(work, &four_pi / &radius)
                .ceil()
                .to_integer()
                .and_then(|terms| terms.to_usize())
                .ok_or("the Fatou coordinate needs too many terms")?
                + 8;
            cnum::check_float_storage(3 * last as u128, work)?;
            // The first term left out, `c_k u^k`, leaves the defect
            // `(k/2) c_k u^(k+1)`: two of them in a row below the budget.
            let mut power = Float::with_val_64(work, radius.square_ref());
            let mut small = 0u32;
            let mut terms = None;
            for k in 1..=last {
                recursion.extend(k);
                let size = Float::with_val_64(work, recursion.coefficients[k - 1].abs_ref());
                let estimate = size * &power * (k as u64);
                // The circle is measured with a factor between its points.
                small = if Float::with_val_64(work, &estimate * 8u32) <= budget {
                    small + 1
                } else {
                    0
                };
                if small == 2 {
                    terms = Some((k - 2).max(1));
                    break;
                }
                power *= &radius;
            }
            state.radius = radius.clone();
            state.reach = reach;
            while let Some(count) = terms {
                recursion.extend(count);
                state.coefficients = recursion.coefficients[..count].to_vec();
                let defect = state.defect()?;
                if cnum::verbose() {
                    eprintln!(
                        "parabolic: {count} terms on |u| <= {:.6}, defect {:.3e} (budget {:.3e})",
                        DisplayFloat(&state.radius),
                        DisplayFloat(&defect),
                        DisplayFloat(&budget)
                    );
                }
                if defect <= budget {
                    state.anchor = state.anchor()?;
                    return Ok(state);
                }
                terms = (count + 4 <= last).then_some(count + 4);
            }
            radius = radius * 7u32 / 8u32;
        }
        Err("the Fatou coordinate of the base e^(1/e) did not reach the working accuracy".into())
    }

    /// `P(u) = sum c_k u^(k-1)` and its derivative.
    fn polynomial(&self, u: &Complex) -> (Complex, Complex) {
        let work = self.work;
        let mut value = cnum::zero(work);
        let mut derivative = cnum::zero(work);
        for coefficient in self.coefficients.iter().rev() {
            derivative = Complex::with_val_64(work, derivative * u) + &value;
            value = Complex::with_val_64(work, value * u) + coefficient;
        }
        (value, derivative)
    }

    /// `A(u)` and `A'(u)`.
    fn coordinate(&self, u: &Complex) -> (Complex, Complex) {
        let work = self.work;
        let (value, derivative) = self.polynomial(u);
        let inverse = Complex::with_val_64(work, u.recip_ref());
        let log = cnum::ln_complex(&Complex::with_val_64(work, -u), work);
        let coordinate = Complex::with_val_64(work, &value * u) + log / 3u32
            - Complex::with_val_64(work, &inverse * 2u32);
        let slope = Complex::with_val_64(work, derivative * u)
            + value
            + Complex::with_val_64(work, &inverse / 3u32)
            + Complex::with_val_64(work, inverse.square_ref()) * 2u32;
        (coordinate, slope)
    }

    /// The largest defect `|A(e^u - 1) - A(u) - 1|` at eight points of the
    /// circle `|u| = radius`: the defect is analytic inside, and next to
    /// its first term `u^(N+2)` its modulus hardly varies on the circle.
    fn defect(&self) -> Result<Float, String> {
        let work = self.work;
        let mut largest = Float::new_64(work);
        // The coefficients are real: the lower half mirrors the upper one.
        for part in 0u32..5 {
            let angle = Float::with_val_64(work, Constant::Pi) * part / 4u32;
            let (sine, cosine) = angle.sin_cos(Float::new_64(work));
            let u = Complex::with_val_64(work, (cosine, sine)) * &self.radius;
            let image = exp_m1(&u, work).ok_or("the Fatou coordinate has no defect there")?;
            // `q = (e^u - 1)/u - 1`: `2/u - 2/(e^u - 1) = 2 q/(e^u - 1)`.
            let quotient = Complex::with_val_64(work, &image - &u) / &u;
            let log = ln_1p(&quotient, work).ok_or("the Fatou coordinate has no defect at 0")?;
            let (above, _) = self.polynomial(&image);
            let (below, _) = self.polynomial(&u);
            let defect = Complex::with_val_64(work, above * &image)
                - Complex::with_val_64(work, below * &u)
                + log / 3u32
                + Complex::with_val_64(work, quotient * 2u32) / &image
                - 1u32;
            largest = largest.max(&cnum::abs(&defect, work));
        }
        if !largest.is_finite() {
            return Err("the defect of the Fatou coordinate is not finite".into());
        }
        // The points between the eight.
        Ok(largest * 4u32)
    }

    /// Whether the series gives the coordinate of `u`: the orbit of `u`
    /// stays in `|u| <= radius`. With `zeta = -2/u`, which a step moves by
    /// about 1 to the right, that is `Re zeta >= 0` and `|zeta| >= 2/radius`,
    /// or `|Im zeta| >= 2/radius + 2`, which an orbit keeps to within
    /// `pi/3` of it.
    fn covers(&self, u: &Complex) -> bool {
        let work = self.work;
        let size = cnum::abs(u, work);
        if size.is_zero() || !size.is_finite() {
            return false;
        }
        if !u.real().is_sign_positive() && size <= self.radius {
            return true;
        }
        let side = Float::with_val_64(work, u.imag().abs_ref()) * 2u32 / size.square();
        side >= self.reach
    }

    /// The orbit of `u` until the series covers it: the point, the steps,
    /// and whether the orbit passed 0 of the tower (`u = -1`) within the
    /// working precision on the principal branch.
    fn orbit(&self, start: &Complex) -> Result<(Complex, Integer, Option<Integer>), String> {
        let work = self.work;
        let mut u = Complex::with_val_64(work, start);
        let mut steps = Integer::new();
        let mut checkpoint = u.clone();
        let pi = Float::with_val_64(work, Constant::Pi);
        let mut principal = true;
        let mut collapsed = None;
        loop {
            if cnum::is_zero(&u) {
                return Err("super-logarithm is singular at the fixed point of the base".into());
            }
            if self.covers(&u) {
                return Ok((u, steps, collapsed));
            }
            if u.real().is_sign_positive() {
                // On the repelling side of the fixed point: `-2/u` moves by
                // one a step, and has to pass the fixed point first.
                let size = cnum::abs(&u, work);
                let away = Float::with_val_64(work, u.real() * 2u32) / size.square();
                if u.imag().is_zero() {
                    return Err(
                        "the orbit of z leaves the basin of the fixed point e: z lies beyond it"
                            .into(),
                    );
                }
                if away > Float::with_val_64(work, 1u32) << 40u32 {
                    return Err(format!(
                        "the orbit of z stays next to the fixed point e for about {:.3e} steps",
                        DisplayFloat(&away)
                    ));
                }
            }
            principal &= *u.imag() <= pi && *u.imag() > -pi.clone();
            u = exp_m1(&u, work).ok_or(
                "the orbit of z cannot be followed at this precision: z lies next to the \
                 boundary of the basin of the fixed point e",
            )?;
            steps += 1;
            if !cnum::is_finite(&u) {
                return Err("the orbit of z leaves the basin of the fixed point e".into());
            }
            if u.imag().is_zero() && *u.real() == -1 && collapsed.is_none() {
                if !principal {
                    return Err(
                        "the orbit of z vanishes off the principal branch of the tetration".into(),
                    );
                }
                collapsed = Some(steps.clone());
            }
            if u == checkpoint {
                return Err("the orbit of z does not approach the fixed point e".into());
            }
            if steps.is_power_of_two() {
                checkpoint = u.clone();
                if cnum::verbose() && steps > 1024 {
                    eprintln!("parabolic: forward orbit, {steps} steps");
                }
            }
        }
    }

    /// `alpha(u)` for a point of the basin, and the size of what was summed.
    fn alpha(&self, u: &Complex) -> Result<(Complex, Float, Option<Integer>), String> {
        let work = self.work;
        let (point, steps, collapsed) = self.orbit(u)?;
        let (coordinate, _) = self.coordinate(&point);
        let scale = cnum::abs(&coordinate, work) + Float::with_val_64(work, &steps);
        Ok((
            coordinate - Float::with_val_64(work, &steps),
            scale,
            collapsed,
        ))
    }

    fn anchor(&self) -> Result<Complex, String> {
        let work = self.work;
        // `1/e - 1`: the tower value 1.
        let start = Float::with_val_64(work, -1).exp_m1();
        let (coordinate, _, _) = self.alpha(&Complex::with_val_64(work, (start, 0)))?;
        Ok(coordinate)
    }

    /// `psi(target)`: the point of the petal with the coordinate `target`,
    /// which must lie where `covers` holds.
    fn inverse(&self, target: &Complex) -> Result<Complex, String> {
        let work = self.work;
        let mut u = Complex::with_val_64(work, target.recip_ref()) * -2i32;
        let mut best: Option<(Float, Complex)> = None;
        loop {
            if cnum::is_zero(&u) || !cnum::is_finite(&u) {
                return Err("the inverse Fatou coordinate left its petal".into());
            }
            let (value, slope) = self.coordinate(&u);
            let residual = Complex::with_val_64(work, value - target);
            let error = cnum::abs(&residual, work);
            if let Some((previous, argument)) = &best {
                if error >= *previous {
                    u = argument.clone();
                    break;
                }
            }
            if error.is_zero() {
                return Ok(u);
            }
            let next = Complex::with_val_64(work, &u - residual / slope);
            best = Some((error, std::mem::replace(&mut u, next)));
        }
        let (error, _) = best.ok_or("the inverse Fatou coordinate did not start")?;
        let shift = usize::try_from(work.saturating_sub(24)).map_err(|_| "precision overflow")?;
        if error > cnum::abs(target, work) >> shift {
            return Err("the inverse Fatou coordinate did not converge".into());
        }
        Ok(u)
    }

    /// `F(h)` at the working precision, and the logarithm of the factor by
    /// which the chain of logarithms below the petal can have grown an
    /// error, as `schroder::eval_schroder_with_conditioning` counts it.
    fn tetrate(&self, h: &Complex) -> Result<(Complex, Float), String> {
        let work = self.work;
        let y = Complex::with_val_64(work, h + &self.anchor);
        // The least shift to the right after which the series covers the
        // point: `Re >= 1` and `|y + k| >= reach`, or `|Im y| >= reach + 2`.
        let side = Float::with_val_64(work, y.imag().abs_ref());
        let mut shift = Integer::new();
        if side < Float::with_val_64(work, &self.reach + 2u32) {
            let across = (Float::with_val_64(work, self.reach.square_ref()) - side.square())
                .max(&Float::with_val_64(work, 1))
                .sqrt();
            let needed = Float::with_val_64(work, across - y.real()).ceil();
            if needed > 0 {
                shift = needed
                    .to_integer()
                    .ok_or("the height reduction is not finite")?;
            }
        }
        let (mut u, top) = loop {
            let top = Complex::with_val_64(work, &y + Float::with_val_64(work, &shift));
            let u = self.inverse(&top)?;
            if self.covers(&u) {
                break (u, top);
            }
            shift += 1;
        };
        // Digits of the height that the shift has cancelled.
        let before = cnum::abs(&y, work) + Float::with_val_64(work, &shift);
        let mut log_amplification = (before / cnum::abs(&top, work))
            .ln()
            .max(&Float::new_64(work));
        let zero = Float::new_64(work);
        let mut net = Float::new_64(work);
        let mut lowest = Float::new_64(work);
        let mut expanding = Float::new_64(work);
        // `ln |F|` of the point above the one that is being computed.
        let mut above: Option<Float> = None;
        let mut count = |above: &Float, here: &Float| {
            // The factor of a relative error (of a value below 1: of an
            // absolute one) through `log_b`, with `ln b = 1/e`.
            let factor =
                Float::with_val_64(work, 1u32 - above.clone().min(&zero)) - here.clone().max(&zero);
            net += &factor;
            lowest = lowest.clone().min(&net);
            expanding += factor.max(&zero);
        };
        let mut left = shift.clone();
        while left > 0 {
            let next = ln_1p(&u, work).ok_or_else(|| {
                "the height lies at a singularity of the tetration (logarithm of zero)".to_string()
            })?;
            // `ln |e (1 + u)| = 1 + Re ln(1 + u)`.
            let here = Float::with_val_64(work, next.real() + 1u32);
            if let Some(above) = &above {
                count(above, &here);
            }
            above = Some(here);
            u = next;
            left -= 1;
            if !cnum::is_finite(&u) {
                return Err("the height reduction left MPFR's range".into());
            }
            if cnum::verbose() && left.is_divisible_u(1 << 20) && left > 0 {
                eprintln!("parabolic: {left} logarithms left");
            }
        }
        let value = Complex::with_val_64(work, &u + 1u32) * &self.e;
        if let Some(above) = &above {
            let here = match ln_1p(&u, work) {
                Some(log) => Float::with_val_64(work, log.real() + 1u32),
                None => Float::with_val_64(work, rug::float::Special::NegInfinity),
            };
            count(above, &here);
        }
        let rounded = Float::with_val_64(work, Integer::from(&shift + 1u32)).ln();
        log_amplification += expanding.min(&(net - lowest + rounded));
        if !log_amplification.is_finite() {
            return Err("the conditioning of the height reduction is not finite".into());
        }
        Ok((value, log_amplification))
    }

    /// The height of `z` and the size of what was subtracted for it; the
    /// flag marks an orbit that passed 0 within the working precision, with
    /// the integer height that it then has.
    fn height(&self, z: &Complex, given: u64) -> Result<(Complex, Float, Option<Integer>), String> {
        let work = self.work;
        let u = Complex::with_val_64(work, z / &self.e) - 1u32;
        // A value that the bits it was given with cannot tell from `e`.
        let shift = usize::try_from(given.saturating_sub(8)).map_err(|_| "precision overflow")?;
        if cnum::abs(&u, work) <= Float::with_val_64(work, 1) >> shift {
            return Err("super-logarithm is singular at the fixed point of the base".into());
        }
        let (coordinate, scale, collapsed) = self.alpha(&u)?;
        let size = scale + cnum::abs(&self.anchor, work) + 4u32;
        Ok((coordinate - &self.anchor, size, collapsed))
    }
}

/// `F(h)` for the base `e^(1/e)`, correct to `digits` digits with inputs of
/// `prec` bits. The value carries more bits where the height had to be read
/// with more.
pub(crate) fn tetrate(h: &Complex, prec: u64, digits: u64) -> Result<Complex, String> {
    cnum::init_mpfr();
    cnum::require_precision(prec, digits)?;
    if !cnum::is_finite(h) {
        return Err("tetration height must be finite".into());
    }
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
        if *h.real() == 1 {
            return Ok(Complex::with_val_64(prec, (eta(prec)?, 0)));
        }
    }
    let tolerance = cnum::epsilon(digits, prec) / 1000u32;
    let mut bits = prec;
    loop {
        let state = Fatou::at(bits)?;
        let (value, log_amplification) = state.tetrate(&Complex::with_val_64(bits, h))?;
        match cnum::conditioned_precision(&log_amplification, &tolerance, bits)? {
            Some(next) => {
                if cnum::verbose() {
                    eprintln!(
                        "parabolic height conditioning: refining from {bits} to {next} working bits"
                    );
                }
                bits = next;
            }
            None => {
                let mut value = Complex::with_val_64(bits, value);
                // A real height above -2 has a real value.
                if value.imag().is_zero() {
                    value.mut_imag().abs_mut();
                }
                return Ok(value);
            }
        }
    }
}

/// The height `h` with `F(h) = z` for the base `e^(1/e)` at `work` bits:
/// the Fatou coordinate of `z` less that of 1. With it the logarithm of the
/// factor by which a relative error of `z`, or of the arithmetic, grows in
/// the relative error of the height, and whether the height is a singular
/// integer at this precision (the orbit of `z` passes 0). `given` is the
/// number of bits that `z` was read with.
pub(crate) fn height(z: &Complex, work: u64, given: u64) -> Result<(Complex, Float, bool), String> {
    let state = Fatou::at(work)?;
    let (height, scale, collapsed) = state.height(z, given)?;
    if let Some(steps) = collapsed {
        // `z` is `F(-1 - steps)` to every bit.
        let whole = Float::with_val_64(work, -(steps + 1u32));
        return Ok((
            Complex::with_val_64(work, (whole, 0)),
            Float::new_64(work),
            true,
        ));
    }
    if cnum::is_zero(&height) {
        return Ok((cnum::zero(work), Float::new_64(work), false));
    }
    let size = cnum::abs(&height, work);
    let mut loss = Float::with_val_64(work, &scale / &size).ln();
    // The response to a relative nudge of `z`.
    let shift = usize::try_from(work / 2).map_err(|_| "precision overflow")?;
    let nudge = Float::with_val_64(work, 1) >> shift;
    if !cnum::is_zero(z) {
        let moved = Complex::with_val_64(work, z * (nudge.clone() + 1u32));
        if let Ok((other, _, None)) = state.height(&moved, given) {
            let change = cnum::abs(&Complex::with_val_64(work, other - &height), work);
            let response = change / nudge / &size;
            if response.is_finite() && !response.is_zero() {
                loss = loss.max(&response.ln());
            }
        }
    }
    Ok((height, loss.max(&Float::new_64(work)), false))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &Complex, b: &Complex, digits: u64, prec: u64) -> bool {
        let gap = cnum::abs(&Complex::with_val_64(prec, a - b), prec);
        gap <= cnum::abs(b, prec).max(&Float::with_val_64(prec, 1)) * cnum::epsilon(digits, prec)
    }

    #[test]
    fn the_base_is_recognized_by_its_value() {
        for digits in [1, 20, 77, 1000] {
            let prec = cnum::digits_to_bits(digits);
            let base = Complex::with_val_64(prec, (eta(prec).unwrap(), 0));
            assert!(is_eta(&base, digits), "{digits} digits");
            // The same number with more bits is the same base.
            assert!(is_eta(&Complex::with_val_64(prec + 300, &base), digits));
            // The expression of the command line.
            let written = crate::expression::complex("e^(1/e)", prec).unwrap();
            assert!(is_eta(&written, digits), "e^(1/e) at {digits} digits");
            // Too few bits of it for the digits that are asked for.
            assert!(!is_eta(&base, digits + 40));
            // A base that leaves it in the 40th bit is another base.
            let offset = Float::with_val_64(prec, 1) >> 40u32;
            for factor in [offset.clone() + 1u32, 1u32 - offset] {
                let other = Complex::with_val_64(prec, &base * factor);
                assert!(!is_eta(&other, digits));
            }
        }
        let prec = cnum::digits_to_bits(20);
        for other in [
            "1.4446678610097661337",
            "1.4375",
            "1.5",
            "2",
            "-1.4446678610097661337",
        ] {
            let base = cnum::parse_complex(other, "0", prec).unwrap();
            assert!(!is_eta(&base, 20), "{other}");
        }
        let complex = Complex::with_val_64(prec, (eta(prec).unwrap(), 1));
        assert!(!is_eta(&complex, 20));
    }

    #[test]
    fn the_first_coefficients_are_the_known_fractions() {
        let work = 200;
        let mut recursion = Recursion::new(work);
        recursion.extend(6);
        // -1/36, 1/540, 1/7776, -71/435456, 8759/163296000, 31/20995200.
        let expected: [(i64, u64); 6] = [
            (-1, 36),
            (1, 540),
            (1, 7776),
            (-71, 435_456),
            (8759, 163_296_000),
            (31, 20_995_200),
        ];
        for (found, (numerator, denominator)) in recursion.coefficients.iter().zip(expected) {
            let fraction = Float::with_val_64(work, numerator) / denominator;
            let gap = Float::with_val_64(work, found - &fraction).abs();
            assert!(
                gap < Float::with_val_64(work, 1) >> 180u32,
                "{found} {fraction}"
            );
        }
    }

    #[test]
    fn the_coordinate_satisfies_the_abel_equation_and_the_tower() {
        for digits in [15, 40, 120] {
            let prec = cnum::digits_to_bits(digits);
            let state = Fatou::at(prec).unwrap();
            // Inside the circle of the series the defect is below the goal.
            let goal = Float::with_val_64(prec, 1) >> usize::try_from(prec).unwrap();
            assert!(state.defect().unwrap() < goal);
            // F(2) = eta^eta and F(3) = eta^(eta^eta), not special cases.
            let base = eta(prec).unwrap();
            let mut tower = Complex::with_val_64(prec, (&base, 0));
            for height in [2, 3] {
                tower = cnum::pow_complex(&Complex::with_val_64(prec, (&base, 0)), &tower, prec);
                let (found, _) = state
                    .tetrate(&Complex::with_val_64(prec, (height, 0)))
                    .unwrap();
                assert!(close(&found, &tower, digits + 5, prec), "F({height})");
            }
        }
    }

    #[test]
    fn logarithms_and_exponentials_keep_small_arguments() {
        let prec = 200;
        let tiny = Float::with_val_64(prec, 1) >> 150u32;
        let u = Complex::with_val_64(prec, (&tiny, &tiny));
        // e^u - 1 = u + u^2/2 + ..., ln(1 + u) = u - u^2/2 + ...
        let up = exp_m1(&u, prec).unwrap();
        let down = ln_1p(&up, prec).unwrap();
        assert!(close(&down, &u, 55, prec));
        assert!(ln_1p(&Complex::with_val_64(prec, (-1, 0)), prec).is_none());
        // The argument on the cut is pi.
        let cut = ln_1p(&Complex::with_val_64(prec, (-3, 0)), prec).unwrap();
        assert!(*cut.imag() > 3 && *cut.real() > 0);
        // Next to -1 the modulus is not lost.
        let near = Complex::with_val_64(prec, (-1, &tiny));
        let log = ln_1p(&near, prec).unwrap();
        let expected = tiny.clone().ln();
        assert!(Float::with_val_64(prec, log.real() - &expected).abs() < tiny);
    }
}
