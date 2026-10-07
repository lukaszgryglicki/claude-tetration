use rug::{
    float::Round,
    ops::{AddAssignRound, DivAssignRound, MulAssignRound, SubAssignRound},
    Complex, Float, Integer,
};

use crate::{cnum, schroder_jumps::Jump};

const PAIRED_TAIL_BOUND: u32 = 3;

#[derive(Clone)]
struct Coefficient {
    value: Complex,
    error: Float,
    norm: Float,
}

impl Coefficient {
    fn new(value: Complex, error: Float) -> Self {
        let prec = error.prec_64();
        let mut norm = value.real().clone().abs();
        norm.add_assign_round(value.imag().clone().abs(), Round::Up);
        Self {
            value,
            error: Float::with_val_round_64(prec, error, Round::Up).0,
            norm,
        }
    }

    fn exact(value: Complex, prec: u64) -> Self {
        Self::new(value, Float::new_64(prec))
    }

    fn zero(prec: u64) -> Self {
        Self::exact(cnum::zero(prec), prec)
    }

    fn is_zero(&self) -> bool {
        self.norm.is_zero() && self.error.is_zero()
    }

    fn magnitude(&self) -> Float {
        Float::with_val_round_64(self.error.prec_64(), &self.norm + &self.error, Round::Up).0
    }

    fn add_assign(&mut self, other: &Self, unit: &Float) {
        let prec = unit.prec_64();
        let mut rounding = Float::with_val_round_64(prec, &self.norm + &other.norm, Round::Up).0;
        rounding.mul_assign_round(unit, Round::Up);
        self.error.add_assign_round(&other.error, Round::Up);
        self.error.add_assign_round(rounding, Round::Up);
        self.value += &other.value;
        self.norm = self.value.real().clone().abs();
        self.norm
            .add_assign_round(self.value.imag().clone().abs(), Round::Up);
    }

    fn multiply(&self, other: &Self, unit: &Float) -> Self {
        let prec = unit.prec_64();
        let mut error = Float::with_val_round_64(prec, &self.norm * &other.error, Round::Up).0;
        error.add_assign_round(
            Float::with_val_round_64(prec, &other.norm * &self.error, Round::Up).0,
            Round::Up,
        );
        error.add_assign_round(
            Float::with_val_round_64(prec, &self.error * &other.error, Round::Up).0,
            Round::Up,
        );
        let mut rounding = Float::with_val_round_64(prec, &self.norm * &other.norm, Round::Up).0;
        rounding.mul_assign_round(unit, Round::Up);
        error.add_assign_round(rounding, Round::Up);
        Self::new(
            Complex::with_val_64(prec, &self.value * &other.value),
            error,
        )
    }

    fn scale(&self, factor: &Float, divide: bool, unit: &Float) -> Self {
        let prec = unit.prec_64();
        let mut error = Float::with_val_round_64(prec, &self.norm * unit, Round::Up).0;
        error.add_assign_round(&self.error, Round::Up);
        let value = if divide {
            error.div_assign_round(factor, Round::Up);
            Complex::with_val_64(prec, &self.value / factor)
        } else {
            error.mul_assign_round(factor, Round::Up);
            Complex::with_val_64(prec, &self.value * factor)
        };
        Self::new(value, error)
    }

    fn negative(mut self) -> Self {
        self.value = -self.value;
        self
    }
}

fn product(a: &[Coefficient], b: &[Coefficient], order: usize, unit: &Float) -> Vec<Coefficient> {
    let mut result = vec![Coefficient::zero(unit.prec_64()); order + 1];
    for (i, x) in a.iter().enumerate().take(order + 1) {
        if x.is_zero() {
            continue;
        }
        for (j, y) in b.iter().enumerate().take(order + 1 - i) {
            if !y.is_zero() {
                result[i + j].add_assign(&x.multiply(y, unit), unit);
            }
        }
    }
    result
}

fn compose(a: &[Coefficient], b: &[Coefficient], unit: &Float) -> Vec<Coefficient> {
    let prec = unit.prec_64();
    let order = a.len() - 1;
    let block = (order + 1).isqrt() + 1;
    let mut powers = Vec::with_capacity(block + 1);
    let mut one = vec![Coefficient::zero(prec); order + 1];
    one[0] = Coefficient::exact(cnum::one(prec), prec);
    powers.push(one);
    powers.push(b.to_vec());
    for _ in 2..=block {
        powers.push(product(powers.last().unwrap(), b, order, unit));
    }
    let mut result = vec![Coefficient::zero(prec); order + 1];
    for group in (0..=order / block).rev() {
        if group < order / block {
            result = product(&result, &powers[block], order, unit);
        }
        for j in 0..block.min(order + 1 - group * block) {
            let weight = &a[group * block + j];
            if !weight.is_zero() {
                for (output, power) in result.iter_mut().zip(&powers[j]) {
                    if !power.is_zero() {
                        output.add_assign(&weight.multiply(power, unit), unit);
                    }
                }
            }
        }
    }
    result
}

pub(crate) struct ComplexOrbitJumps {
    logarithm: bool,
    prec: u64,
    order: usize,
    minimum_steps: Integer,
    lambda: Complex,
    gap: Float,
    logarithm_base: Complex,
    fixed_point: Complex,
    defect: Float,
    unit: Float,
    pair_bounds: Option<[Float; 3]>,
    jets: Vec<Vec<Coefficient>>,
}

impl ComplexOrbitJumps {
    pub fn new(
        ln_b: &Complex,
        fixed_point: &Complex,
        logarithm: bool,
        prec: u64,
    ) -> Result<Option<Self>, String> {
        Self::with_pairs(ln_b, fixed_point, logarithm, prec, false)
    }

    pub fn new_paired(
        ln_b: &Complex,
        fixed_point: &Complex,
        logarithm: bool,
        prec: u64,
    ) -> Result<Option<Self>, String> {
        Self::with_pairs(ln_b, fixed_point, logarithm, prec, true)
    }

    fn with_pairs(
        ln_b: &Complex,
        fixed_point: &Complex,
        logarithm: bool,
        prec: u64,
        paired: bool,
    ) -> Result<Option<Self>, String> {
        cnum::init_mpfr();
        cnum::check_precision(prec)?;
        if !cnum::is_finite(ln_b) || !cnum::is_finite(fixed_point) {
            return Err("Complex Schroder jumps require finite parameters".into());
        }
        if (!paired && ln_b.imag().is_zero() && fixed_point.imag().is_zero())
            || *fixed_point.real() <= 0
        {
            return Ok(None);
        }
        let unit = Float::with_val_64(prec, 16)
            >> usize::try_from(prec).map_err(|_| "Complex jump precision is not addressable")?;
        if unit.is_zero() {
            return Err("Complex jump roundoff unit exceeds MPFR's range".into());
        }
        let a = Coefficient::exact(ln_b.clone(), prec);
        let fixed = Coefficient::exact(fixed_point.clone(), prec);
        let q = a.multiply(&fixed, &unit);
        let lambda = q.value.clone();
        let lower = Float::with_val_round_64(prec, lambda.abs_ref(), Round::Down).0;
        let upper = Float::with_val_round_64(prec, lambda.abs_ref(), Round::Up).0;
        if lower <= 0
            || upper >= 1
            || (paired && Float::with_val_round_64(prec, &upper + &q.error, Round::Up).0 >= 1)
        {
            return Ok(None);
        }
        let pair_bounds = if paired {
            let coefficient = Coefficient::exact(lambda.clone(), prec);
            // Pairing exposes the quadratic cancellation at lambda = -1.
            let mut quadratic = coefficient.clone();
            quadratic.add_assign(&Coefficient::exact(cnum::one(prec), prec), &unit);
            if quadratic.magnitude() > Float::with_val_64(prec, 1) / 32 {
                return Ok(None);
            }
            let squared = coefficient.multiply(&coefficient, &unit);
            let mut cubic = squared.scale(
                &Float::with_val_64(prec, if logarithm { 2 } else { 1 }),
                false,
                &unit,
            );
            cubic.add_assign(
                &coefficient.scale(&Float::with_val_64(prec, 3), false, &unit),
                &unit,
            );
            cubic.add_assign(
                &Coefficient::exact(
                    Complex::with_val_64(prec, if logarithm { 2 } else { 1 }),
                    prec,
                ),
                &unit,
            );
            if logarithm {
                let inverse = Coefficient::new(
                    Complex::with_val_64(prec, lambda.recip_ref()),
                    Float::with_val_round_64(prec, &unit * 2, Round::Up).0,
                );
                let inverse2 = inverse.multiply(&inverse, &unit);
                let inverse4 = inverse2.multiply(&inverse2, &unit);
                quadratic = quadratic.multiply(&inverse4, &unit);
                cubic = cubic.multiply(&inverse4.multiply(&inverse2, &unit), &unit);
            } else {
                quadratic = quadratic.multiply(&squared, &unit);
                cubic = cubic.multiply(&squared, &unit);
            }
            quadratic = quadratic.scale(&Float::with_val_64(prec, 2), true, &unit);
            cubic = cubic.scale(&Float::with_val_64(prec, 6), true, &unit);
            let lower2 = Float::with_val_round_64(prec, lower.square_ref(), Round::Down).0;
            let mut growth = Float::with_val_round_64(prec, lower2.recip_ref(), Round::Up).0;
            growth.sub_assign_round(1, Round::Up);
            Some([growth, quadratic.magnitude(), cubic.magnitude()])
        } else {
            None
        };
        let order = usize::try_from(prec.div_ceil(if paired { 2 } else { 4 }) + 16)
            .map_err(|_| "Complex jump order exceeds addressable memory")?;
        let minimum_steps = Integer::from(
            order
                .checked_next_power_of_two()
                .ok_or("Complex jump order exceeds addressable memory")?,
        );
        let gap = Float::with_val_round_64(prec, Float::with_val_64(prec, 1) - &lower, Round::Up).0;
        if Float::with_val_round_64(prec, &gap * &minimum_steps, Round::Up).0
            > Float::with_val_64(prec, 1) / 8
        {
            return Ok(None);
        }
        if Float::with_val_round_64(prec, &lower - &q.error, Round::Down).0
            <= Float::with_val_64(prec, 7) / 8
        {
            return Err("Complex jump fixed-point product is unresolved".into());
        }
        let defect = if logarithm {
            let value = cnum::ln_complex(fixed_point, prec);
            let mut error = Float::with_val_round_64(prec, value.abs_ref(), Round::Up)
                .0
                .max(&Float::with_val_64(prec, 1));
            error.mul_assign_round(&unit, Round::Up);
            let mut difference = Coefficient::new(value, error);
            difference.add_assign(&q.clone().negative(), &unit);
            let mut defect = difference.magnitude();
            defect.add_assign_round(
                Float::with_val_round_64(prec, &q.error * 8, Round::Up).0,
                Round::Up,
            );
            defect
        } else {
            let exponent = Float::with_val_round_64(prec, lambda.real() + &q.error, Round::Up).0;
            let bound = Float::with_val_round_64(prec, exponent.exp_ref(), Round::Up).0;
            let mut error = Float::with_val_round_64(prec, &q.error + &unit, Round::Up).0;
            error.mul_assign_round(bound, Round::Up);
            let exponential = Coefficient::new(cnum::checked_exp(&lambda, prec)?, error);
            let mut difference = a.multiply(&exponential, &unit);
            difference.add_assign(&Coefficient::exact(-lambda.clone(), prec), &unit);
            let mut defect = difference.magnitude();
            defect.mul_assign_round(3, Round::Up);
            defect.add_assign_round(&q.error, Round::Up);
            defect
        };
        if !defect.is_finite() {
            return Err("Complex jump fixed-point defect exceeded MPFR's range".into());
        }
        Ok(Some(Self {
            logarithm,
            prec,
            order,
            minimum_steps,
            lambda,
            gap,
            logarithm_base: ln_b.clone(),
            fixed_point: fixed_point.clone(),
            defect,
            unit,
            pair_bounds,
            jets: Vec::new(),
        }))
    }

    fn radius(&self, level: usize) -> Result<Float, String> {
        let Some([linear, quadratic, cubic]) = &self.pair_bounds else {
            return Ok(Float::with_val_64(self.prec, 1)
                >> level
                    .checked_add(2)
                    .ok_or("Complex jump radius is not addressable")?);
        };
        let pairs = Integer::from(1) << level;
        let half = Float::with_val_64(self.prec, 1) / 2;
        if Float::with_val_round_64(self.prec, linear * &pairs, Round::Up).0 >= half {
            return Err("Paired jump linear growth exceeds its analytic disk bound".into());
        }
        // m * growth <= 1/2 keeps every paired iterate inside 2R, with 2R <= 1/4.
        let mut radius = Float::with_val_64(self.prec, 1) >> level.div_ceil(2).max(3);
        loop {
            if radius.is_zero() {
                return Err("Paired jump radius exceeds MPFR's range".into());
            }
            let diameter = Float::with_val_64(self.prec, &radius * 2);
            let mut growth = Float::with_val_64(self.prec, PAIRED_TAIL_BOUND);
            growth.mul_assign_round(&diameter, Round::Up);
            growth.add_assign_round(cubic, Round::Up);
            growth.mul_assign_round(&diameter, Round::Up);
            growth.add_assign_round(quadratic, Round::Up);
            growth.mul_assign_round(&diameter, Round::Up);
            growth.add_assign_round(linear, Round::Up);
            growth.mul_assign_round(&pairs, Round::Up);
            if growth <= half {
                return Ok(radius);
            }
            radius >>= 1;
        }
    }

    fn next_jet(&mut self) -> Result<(), String> {
        let prec = self.prec;
        let block = (self.order + 1).isqrt() + 1;
        cnum::check_float_storage(
            (self.order as u128 + 1) * (self.jets.len() as u128 + 4 * (block as u128 + 8)) * 4,
            prec,
        )?;
        let coefficients = if let Some(previous) = self.jets.last() {
            if self.pair_bounds.is_some() {
                let old_radius = self.radius(self.jets.len() - 1)?;
                let new_radius = self.radius(self.jets.len())?;
                let ratio = Float::with_val_64(prec, &new_radius / &old_radius);
                let inverse = Float::with_val_64(prec, &old_radius / &new_radius);
                let mut power = Float::with_val_64(prec, 1);
                let inner: Vec<_> = previous
                    .iter()
                    .map(|value| {
                        let result = value.scale(&power, false, &self.unit);
                        power *= &ratio;
                        result
                    })
                    .collect();
                compose(previous, &inner, &self.unit)
                    .iter()
                    .map(|value| value.scale(&inverse, false, &self.unit))
                    .collect::<Vec<_>>()
            } else {
                let inner: Vec<_> = previous
                    .iter()
                    .enumerate()
                    .map(|(n, value)| {
                        value.scale(&(Float::with_val_64(prec, 1) >> n), false, &self.unit)
                    })
                    .collect();
                let two = Float::with_val_64(prec, 2);
                compose(previous, &inner, &self.unit)
                    .iter()
                    .map(|value| value.scale(&two, false, &self.unit))
                    .collect::<Vec<_>>()
            }
        } else {
            let mut result = vec![Coefficient::zero(prec); self.order + 1];
            let inverse = Coefficient::new(
                Complex::with_val_64(prec, self.lambda.recip_ref()),
                Float::with_val_round_64(prec, &self.unit * 2, Round::Up).0,
            );
            result[1] = if self.logarithm {
                inverse.clone()
            } else {
                Coefficient::exact(self.lambda.clone(), prec)
            };
            let radius = self.radius(0)?;
            for n in 2..=self.order {
                let mut value = result[n - 1].scale(&radius, false, &self.unit);
                if self.logarithm {
                    value = value.multiply(&inverse, &self.unit);
                    value = value.scale(&Float::with_val_64(prec, n - 1), false, &self.unit);
                }
                result[n] = value.scale(&Float::with_val_64(prec, n), true, &self.unit);
            }
            if self.pair_bounds.is_some() {
                compose(&result, &result, &self.unit)
            } else {
                result
            }
        };
        if coefficients.iter().any(|value| {
            !cnum::is_finite(&value.value) || !value.error.is_finite() || value.error < 0
        }) {
            return Err("Complex jump coefficients exceeded MPFR's range".into());
        }
        if cnum::verbose() {
            eprintln!(
                "complex {}{} jump: 2^{} steps, order {}, {} bits",
                if self.pair_bounds.is_some() {
                    "paired "
                } else {
                    ""
                },
                if self.logarithm { "log" } else { "exp" },
                self.jets.len() + usize::from(self.pair_bounds.is_some()),
                self.order,
                prec
            );
        }
        self.jets.push(coefficients);
        Ok(())
    }

    pub fn advance(
        &mut self,
        point: &Complex,
        maximum: Option<&Float>,
    ) -> Result<Option<Jump>, String> {
        let prec = self.prec;
        if !cnum::is_finite(point)
            || maximum.is_some_and(|limit| !limit.is_finite() || *limit < 0 || !limit.is_integer())
        {
            return Err("Complex jump requires a finite point and integer step bound".into());
        }
        if self.pair_bounds.is_some()
            && self.logarithm_base.imag().is_zero()
            && self.fixed_point.imag().is_zero()
            && !point.imag().is_zero()
        {
            // Retain scalar component-wise arithmetic below a norm-only jump's resolution.
            let mut resolution =
                Float::with_val_round_64(prec, point.real().abs_ref(), Round::Up).0;
            resolution.mul_assign_round(&self.unit, Round::Up);
            if point.imag().clone().abs() < resolution {
                return Ok(None);
            }
        }
        let mut difference = Coefficient::exact(point.clone(), prec);
        difference.add_assign(
            &Coefficient::exact(-self.fixed_point.clone(), prec),
            &self.unit,
        );
        let y = difference.multiply(
            &Coefficient::exact(self.logarithm_base.clone(), prec),
            &self.unit,
        );
        let magnitude = Float::with_val_round_64(prec, y.value.abs_ref(), Round::Up).0;
        let paired = self.pair_bounds.is_some();
        let mut steps = Integer::from(if paired { 2 } else { 1 });
        let mut level = 0usize;
        loop {
            let doubled = Integer::from(&steps * 2);
            if maximum.is_some_and(|limit| *limit < doubled)
                || (!paired
                    && Float::with_val_round_64(prec, &magnitude * &doubled, Round::Up).0
                        > Float::with_val_64(prec, 1) / 64)
                || Float::with_val_round_64(prec, &self.gap * &doubled, Round::Up).0
                    > Float::with_val_64(prec, 1) / 8
            {
                break;
            }
            let next_level = level
                .checked_add(1)
                .ok_or("Complex jump table is not addressable")?;
            if paired && magnitude > self.radius(next_level)? / 4 {
                break;
            }
            steps = doubled;
            level = next_level;
        }
        if steps < self.minimum_steps {
            return Ok(None);
        }
        let mut radius = self.radius(level)?;
        let perturbation = loop {
            if radius.is_zero() {
                return Err("Complex jump radius exceeds MPFR's range".into());
            }
            let mut error = Float::with_val_round_64(prec, &self.defect * &steps, Round::Up).0;
            error.add_assign_round(&y.error, Round::Up);
            error.mul_assign_round(if paired { 64 } else { 8 }, Round::Up);
            if error < Float::with_val_64(prec, &radius / if paired { 16 } else { 2 }) {
                break error;
            }
            steps >>= 1;
            if steps < self.minimum_steps {
                return Ok(None);
            }
            level -= 1;
            if paired {
                radius = self.radius(level)?;
            } else {
                radius *= 2;
            }
        };
        while self.jets.len() <= level {
            self.next_jet()?;
        }
        let mut input = Complex::with_val_64(prec, &y.value / &radius);
        if self.logarithm {
            input = -input;
        }
        let ratio = Float::with_val_round_64(prec, input.abs_ref(), Round::Up).0;
        if ratio > Float::with_val_64(prec, 1) / if paired { 4 } else { 16 } {
            return Err("Complex jump is outside its analytic evaluation disk".into());
        }
        let input = Coefficient::exact(input, prec);
        let mut value = Coefficient::zero(prec);
        let mut derivative = Coefficient::zero(prec);
        for coefficient in self.jets[level].iter().rev() {
            derivative = derivative.multiply(&input, &self.unit);
            derivative.add_assign(&value, &self.unit);
            value = value.multiply(&input, &self.unit);
            value.add_assign(coefficient, &self.unit);
        }
        let mut power = Float::with_val_64(prec, 1);
        for _ in 0..self.order {
            power.mul_assign_round(&ratio, Round::Up);
        }
        let denominator =
            Float::with_val_round_64(prec, &Float::with_val_64(prec, 1) - &ratio, Round::Down).0;
        let mut tail = Float::with_val_round_64(prec, &power * &ratio, Round::Up).0;
        tail.mul_assign_round(2, Round::Up);
        tail.div_assign_round(&denominator, Round::Up);
        value.error.add_assign_round(tail, Round::Up);
        value = value.scale(&radius, false, &self.unit);
        value.error.add_assign_round(&perturbation, Round::Up);
        let mut derivative_tail = power;
        derivative_tail.mul_assign_round(Integer::from(self.order + 1) * 2, Round::Up);
        derivative_tail.div_assign_round(
            Float::with_val_round_64(prec, denominator.square_ref(), Round::Down).0,
            Round::Up,
        );
        derivative
            .error
            .add_assign_round(derivative_tail, Round::Up);
        let mut derivative_perturbation =
            Float::with_val_round_64(prec, &perturbation / &radius, Round::Up).0;
        derivative_perturbation.mul_assign_round(if paired { 32 } else { 4 }, Round::Up);
        derivative
            .error
            .add_assign_round(derivative_perturbation, Round::Up);
        let mut derivative_bound =
            Float::with_val_round_64(prec, derivative.value.abs_ref(), Round::Up).0;
        derivative_bound.add_assign_round(derivative.error, Round::Up);

        if self.logarithm {
            value = value.negative();
        }
        let a_lower = Float::with_val_round_64(prec, self.logarithm_base.abs_ref(), Round::Down).0;
        let mut rounding = Float::with_val_round_64(prec, &value.norm * &self.unit, Round::Up).0;
        rounding.add_assign_round(value.error, Round::Up);
        rounding.div_assign_round(a_lower, Round::Up);
        value = Coefficient::new(
            Complex::with_val_64(prec, value.value / &self.logarithm_base),
            rounding,
        );
        value.add_assign(
            &Coefficient::exact(self.fixed_point.clone(), prec),
            &self.unit,
        );
        if !cnum::is_finite(&value.value)
            || !value.error.is_finite()
            || !derivative_bound.is_finite()
        {
            return Err("Complex jump evaluation exceeded MPFR's range".into());
        }
        Ok(Some(Jump {
            value: value.value,
            error: value.error,
            derivative: derivative_bound,
            steps,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters(prec: u64, real: &str, imaginary: &str) -> (Complex, Complex) {
        let lambda = cnum::parse_complex(real, imaginary, prec).unwrap();
        let fixed = Complex::with_val_64(prec, lambda.exp_ref());
        let a = Complex::with_val_64(prec, lambda / &fixed);
        (a, fixed)
    }

    #[test]
    fn paired_majorants_bound_tails_and_intermediate_maps() {
        let prec = 256;
        let x: Float = Float::with_val_64(prec, 1) / 4;
        let mu: Float = Float::with_val_64(prec, 31) / 32;
        let half: Float = Float::with_val_64(prec, 1) / 2;
        let mu2 = Float::with_val_64(prec, mu.square_ref());
        let mu4 = Float::with_val_64(prec, mu2.square_ref());
        let mu6 = Float::with_val_64(prec, &mu2 * &mu4);
        let first_exp = Float::with_val_round_64(prec, x.exp_m1_ref(), Round::Up).0;
        let mut exp_tail = Float::with_val_round_64(prec, first_exp.exp_m1_ref(), Round::Up).0;
        exp_tail.sub_assign_round(&x, Round::Up);
        exp_tail.sub_assign_round(Float::with_val_64(prec, 1) / 16, Round::Up);
        let mut exp_cubic = Float::with_val_64(prec, 5);
        exp_cubic.div_assign_round(384, Round::Down);
        exp_tail.sub_assign_round(exp_cubic, Round::Up);
        exp_tail.mul_assign_round(256, Round::Up);
        let inverse_majorant = |value: &Float| {
            let mut ratio = value.clone();
            ratio.div_assign_round(&mu, Round::Up);
            let mut complement = Float::with_val_64(prec, 1);
            complement.sub_assign_round(ratio, Round::Down);
            assert!(complement > 0);
            -Float::with_val_round_64(prec, complement.ln_ref(), Round::Down).0
        };
        let first_log = inverse_majorant(&x);
        let mut log_tail = inverse_majorant(&first_log);
        let mut linear = x.clone();
        linear.div_assign_round(&mu2, Round::Down);
        let mut quadratic: Float = mu.clone() + 1;
        quadratic.div_assign_round(Float::with_val_64(prec, &mu4 * 32), Round::Down);
        let mut cubic = Float::with_val_64(prec, &mu2 * 2);
        cubic.add_assign_round(Float::with_val_64(prec, &mu * 3), Round::Down);
        cubic.add_assign_round(2, Round::Down);
        cubic.div_assign_round(Float::with_val_64(prec, &mu6 * 384), Round::Down);
        log_tail.sub_assign_round(linear, Round::Up);
        log_tail.sub_assign_round(quadratic, Round::Up);
        log_tail.sub_assign_round(cubic, Round::Up);
        log_tail.mul_assign_round(256, Round::Up);
        let mut inverse_derivative = Float::with_val_64(prec, 1);
        inverse_derivative.div_assign_round(mu - &half, Round::Up);
        let forward_derivative = Float::with_val_round_64(prec, half.exp_ref(), Round::Up).0;
        assert!(exp_tail > 0 && exp_tail < PAIRED_TAIL_BOUND);
        assert!(log_tail > 0 && log_tail < PAIRED_TAIL_BOUND);
        assert!(first_exp < half && first_log < half);
        assert!(inverse_derivative < 3 && forward_derivative < 3);
    }

    #[test]
    fn paired_jumps_enclose_direct_values_and_derivatives() {
        let prec = cnum::digits_to_bits(70);
        let high = cnum::digits_to_bits(180);
        for (lr, li) in [
            ("-0.99999999", "0"),
            ("-0.99999999", "0.000000005"),
            ("-0.99999999", "-0.000000005"),
            ("-0.9998", "0.0199"),
        ] {
            let (a, fixed) = parameters(prec, lr, li);
            for backward in [false, true] {
                let mut cache = ComplexOrbitJumps::new_paired(&a, &fixed, backward, prec)
                    .unwrap()
                    .unwrap();
                for (re, im) in [("-0.001", "0"), ("0.001", "0.001"), ("0.003", "-0.001")] {
                    let y = cnum::parse_complex(re, im, prec).unwrap();
                    let point = Complex::with_val_64(prec, y / &a) + &fixed;
                    let jump = cache
                        .advance(&point, Some(&Float::with_val_64(prec, 1500)))
                        .unwrap()
                        .unwrap();
                    assert_eq!(jump.steps, 1024);
                    let mut expected = Complex::with_val_64(high, &point);
                    let a = Complex::with_val_64(high, &a);
                    let mut derivative = cnum::one(high);
                    for _ in 0..1024 {
                        if backward {
                            derivative /= Complex::with_val_64(high, &expected * &a);
                            expected = Complex::with_val_64(high, expected.ln_ref()) / &a;
                        } else {
                            let exponent = Complex::with_val_64(high, &expected * &a);
                            expected = Complex::with_val_64(high, exponent.exp_ref());
                            derivative *= Complex::with_val_64(high, &expected * &a);
                        }
                    }
                    let error =
                        cnum::abs(&Complex::with_val_64(high, &jump.value - expected), high);
                    assert!(
                        error <= jump.error,
                        "{lr}+{li}i, backward={backward}: {error} > {}",
                        jump.error
                    );
                    assert!(cnum::abs(&derivative, high) <= jump.derivative);
                    assert!(jump.error < cnum::epsilon(70, prec));
                }
            }
        }
    }

    #[test]
    fn paired_jumps_account_for_rounded_fixed_points() {
        let prec = cnum::digits_to_bits(70);
        let high = cnum::digits_to_bits(180);
        let (a, mut fixed) = parameters(prec, "-0.99999999", "0.000000005");
        fixed -= cnum::parse_complex("1e-12", "1e-13", prec).unwrap();
        let point = Complex::with_val_64(prec, &fixed - cnum::decimal("0.001", prec));
        for backward in [false, true] {
            let mut cache = ComplexOrbitJumps::new_paired(&a, &fixed, backward, prec)
                .unwrap()
                .unwrap();
            let jump = cache
                .advance(&point, Some(&Float::with_val_64(prec, 1024)))
                .unwrap()
                .unwrap();
            assert_eq!(jump.steps, 1024);
            let mut expected = Complex::with_val_64(high, &point);
            let a = Complex::with_val_64(high, &a);
            for _ in 0..1024 {
                expected = if backward {
                    Complex::with_val_64(high, expected.ln_ref()) / &a
                } else {
                    let exponent = Complex::with_val_64(high, &expected * &a);
                    Complex::with_val_64(high, exponent.exp_ref())
                };
            }
            let error = cnum::abs(&Complex::with_val_64(high, &jump.value - expected), high);
            assert!(error <= jump.error, "{error} > {}", jump.error);
            assert!(error > cnum::epsilon(40, high));
        }
    }

    #[test]
    fn paired_jump_applicability_and_principal_log_guards() {
        let prec = cnum::digits_to_bits(70);
        let (a, fixed) = parameters(prec, "-0.99999999", "0");
        assert!(ComplexOrbitJumps::new(&a, &fixed, true, prec)
            .unwrap()
            .is_none());
        for logarithm in [false, true] {
            let selected =
                crate::schroder_jumps::OrbitJumps::new(&a, &fixed, logarithm, prec).unwrap();
            assert!(matches!(
                selected,
                Some(crate::schroder_jumps::OrbitJumps::Complex(cache))
                    if cache.pair_bounds.is_some()
            ));
        }
        assert!(
            ComplexOrbitJumps::new_paired(&a, &(-fixed.clone()), true, prec)
                .unwrap()
                .is_none()
        );
        for (re, im) in [
            ("0.99999999", "0"),
            ("0", "0.99999999"),
            ("-0.9", "0.01"),
            ("-1", "0"),
            ("-1", "0.001"),
        ] {
            let (other_a, other_fixed) = parameters(prec, re, im);
            assert!(
                ComplexOrbitJumps::new_paired(&other_a, &other_fixed, true, prec)
                    .unwrap()
                    .is_none()
            );
        }
        let unresolved = Complex::with_val_64(prec, -1)
            + (Float::with_val_64(prec, 1) >> usize::try_from(prec).unwrap());
        let unresolved_fixed = Complex::with_val_64(prec, unresolved.exp_ref());
        let unresolved_a = Complex::with_val_64(prec, unresolved / &unresolved_fixed);
        assert!(
            ComplexOrbitJumps::new_paired(&unresolved_a, &unresolved_fixed, true, prec,)
                .unwrap()
                .is_none()
        );
        let nan = Complex::with_val_64(prec, rug::float::Special::Nan);
        assert!(ComplexOrbitJumps::new_paired(&nan, &fixed, true, prec).is_err());
        assert!(ComplexOrbitJumps::new_paired(&a, &fixed, true, u64::MAX).is_err());
        let mut cache = ComplexOrbitJumps::new_paired(&a, &fixed, true, prec)
            .unwrap()
            .unwrap();
        assert!(cache.advance(&nan, None).is_err());
        for maximum in [
            Float::with_val_64(prec, -1),
            cnum::decimal("0.5", prec),
            Float::with_val_64(prec, rug::float::Special::Infinity),
        ] {
            assert!(cache.advance(&fixed, Some(&maximum)).is_err());
        }
        for maximum in [0, 1, 3] {
            assert!(cache
                .advance(&fixed, Some(&Float::with_val_64(prec, maximum)))
                .unwrap()
                .is_none());
        }
        assert!(cache.advance(&cnum::one(prec), None).unwrap().is_none());
    }

    #[test]
    fn paired_real_jumps_preserve_tiny_imaginary_scalar_paths() {
        let prec = cnum::digits_to_bits(70);
        let (a, fixed) = parameters(prec, "-0.99999999", "0");
        let bound = Float::with_val_64(prec, 1024);
        for backward in [false, true] {
            let mut cache = ComplexOrbitJumps::new_paired(&a, &fixed, backward, prec)
                .unwrap()
                .unwrap();
            let mut point = Complex::with_val_64(prec, &fixed + cnum::decimal("0.001", prec));
            assert!(cache.advance(&point, Some(&bound)).unwrap().is_some());
            for imaginary in ["1e-1000", "-1e-1000", "1e-1000000000000000000"] {
                *point.mut_imag() = cnum::decimal(imaginary, prec);
                assert!(cache.advance(&point, Some(&bound)).unwrap().is_none());
            }
        }
    }

    #[test]
    fn paired_jump_counts_exceed_machine_integers() {
        let prec = cnum::digits_to_bits(110);
        let mut lambda = Complex::with_val_64(prec, -1);
        lambda += cnum::decimal("1e-45", prec);
        *lambda.mut_imag() = cnum::decimal("5e-46", prec);
        let fixed = Complex::with_val_64(prec, lambda.exp_ref());
        let a = Complex::with_val_64(prec, lambda / &fixed);
        let y = cnum::parse_complex("1e-24", "2e-25", prec).unwrap();
        let point = Complex::with_val_64(prec, y / &a) + &fixed;
        let mut cache = ComplexOrbitJumps::new_paired(&a, &fixed, true, prec)
            .unwrap()
            .unwrap();
        let jump = cache.advance(&point, None).unwrap().unwrap();
        assert!(jump.steps > u128::MAX);
        assert!(jump.steps.is_even());
        assert!(jump.error < cnum::epsilon(70, prec));
        let half = Float::with_val_64(prec, &jump.steps) / 2;
        let first = cache.advance(&point, Some(&half)).unwrap().unwrap();
        let second = cache.advance(&first.value, Some(&half)).unwrap().unwrap();
        assert_eq!(first.steps + second.steps, jump.steps);
        let error = cnum::abs(&Complex::with_val_64(prec, jump.value - second.value), prec);
        assert!(error <= jump.error + first.error * second.derivative + second.error);
    }

    #[test]
    fn complex_jumps_enclose_direct_values_and_derivatives() {
        let prec = cnum::digits_to_bits(70);
        let high = cnum::digits_to_bits(180);
        for (lr, li) in [
            ("0.99999999", "0.000000005"),
            ("-0.99999999", "0.000000005"),
            ("0", "0.99999999"),
            ("0.000000005", "-0.99999999"),
        ] {
            let (a, fixed) = parameters(prec, lr, li);
            for backward in [false, true] {
                let mut cache = ComplexOrbitJumps::new(&a, &fixed, backward, prec)
                    .unwrap()
                    .unwrap();
                for (re, im) in [
                    ("-0.00001", "0"),
                    ("0.00001", "0.00001"),
                    ("0.00001", "-0.00001"),
                ] {
                    let y = cnum::parse_complex(re, im, prec).unwrap();
                    let point = Complex::with_val_64(prec, y / &a) + &fixed;
                    let jump = cache
                        .advance(&point, Some(&Float::with_val_64(prec, 1500)))
                        .unwrap()
                        .unwrap();
                    assert_eq!(jump.steps, 1024);
                    let mut expected = Complex::with_val_64(high, &point);
                    let a = Complex::with_val_64(high, &a);
                    let mut derivative = cnum::one(high);
                    for _ in 0..1024 {
                        if backward {
                            derivative /= Complex::with_val_64(high, &expected * &a);
                            expected = Complex::with_val_64(high, expected.ln_ref()) / &a;
                        } else {
                            let exponent = Complex::with_val_64(high, &expected * &a);
                            expected = Complex::with_val_64(high, exponent.exp_ref());
                            derivative *= Complex::with_val_64(high, &expected * &a);
                        }
                    }
                    let error =
                        cnum::abs(&Complex::with_val_64(high, &jump.value - expected), high);
                    assert!(
                        error <= jump.error,
                        "{lr}+{li}i, backward={backward}: {error} > {}",
                        jump.error
                    );
                    assert!(cnum::abs(&derivative, high) <= jump.derivative);
                    assert!(jump.error < cnum::epsilon(70, prec));
                }
            }
        }
    }

    #[test]
    fn complex_jumps_account_for_rounded_fixed_points() {
        let prec = cnum::digits_to_bits(70);
        let high = cnum::digits_to_bits(180);
        let (a, mut fixed) = parameters(prec, "0.99999999", "0.000000005");
        fixed -= cnum::parse_complex("1e-8", "1e-9", prec).unwrap();
        let point = Complex::with_val_64(prec, &fixed - cnum::decimal("1e-5", prec));
        for backward in [false, true] {
            let mut cache = ComplexOrbitJumps::new(&a, &fixed, backward, prec)
                .unwrap()
                .unwrap();
            let jump = cache
                .advance(&point, Some(&Float::with_val_64(prec, 1024)))
                .unwrap()
                .unwrap();
            let mut expected = Complex::with_val_64(high, &point);
            let a = Complex::with_val_64(high, &a);
            for _ in 0..1024 {
                expected = if backward {
                    Complex::with_val_64(high, expected.ln_ref()) / &a
                } else {
                    let exponent = Complex::with_val_64(high, &expected * &a);
                    Complex::with_val_64(high, exponent.exp_ref())
                };
            }
            let error = cnum::abs(&Complex::with_val_64(high, &jump.value - expected), high);
            assert!(error <= jump.error, "{error} > {}", jump.error);
            assert!(error > cnum::epsilon(40, high));
        }
    }

    #[test]
    fn complex_jump_applicability_and_principal_log_guards() {
        let prec = cnum::digits_to_bits(70);
        let (a, fixed) = parameters(prec, "0.99999999", "0.000000005");
        assert!(ComplexOrbitJumps::new(&a, &(-fixed.clone()), true, prec)
            .unwrap()
            .is_none());
        assert!(
            ComplexOrbitJumps::new(&cnum::one(prec), &cnum::one(prec), true, prec)
                .unwrap()
                .is_none()
        );
        for (re, im) in [("0.9", "0.01"), ("1", "0.001")] {
            let (other_a, other_fixed) = parameters(prec, re, im);
            assert!(ComplexOrbitJumps::new(&other_a, &other_fixed, true, prec)
                .unwrap()
                .is_none());
        }
        let nan = Complex::with_val_64(prec, rug::float::Special::Nan);
        assert!(ComplexOrbitJumps::new(&nan, &fixed, true, prec).is_err());
        assert!(ComplexOrbitJumps::new(&a, &fixed, true, u64::MAX).is_err());
        let mut cache = ComplexOrbitJumps::new(&a, &fixed, true, prec)
            .unwrap()
            .unwrap();
        assert!(cache.advance(&nan, None).is_err());
        for maximum in [
            Float::with_val_64(prec, -1),
            cnum::decimal("0.5", prec),
            Float::with_val_64(prec, rug::float::Special::Infinity),
        ] {
            assert!(cache.advance(&fixed, Some(&maximum)).is_err());
        }
        for maximum in [0, 3] {
            assert!(cache
                .advance(&fixed, Some(&Float::with_val_64(prec, maximum)))
                .unwrap()
                .is_none());
        }
        assert!(cache.advance(&cnum::one(prec), None).unwrap().is_none());
    }

    #[test]
    fn complex_jump_counts_exceed_machine_integers() {
        let prec = cnum::digits_to_bits(160);
        let mut lambda = cnum::one(prec);
        lambda -= cnum::decimal("1e-50", prec);
        *lambda.mut_imag() = cnum::decimal("5e-51", prec);
        let fixed = Complex::with_val_64(prec, lambda.exp_ref());
        let a = Complex::with_val_64(prec, lambda / &fixed);
        let y = cnum::parse_complex("-1e-52", "1e-53", prec).unwrap();
        let point = Complex::with_val_64(prec, y / &a) + &fixed;
        let mut cache = ComplexOrbitJumps::new(&a, &fixed, true, prec)
            .unwrap()
            .unwrap();
        let jump = cache.advance(&point, None).unwrap().unwrap();
        assert!(jump.steps > u128::MAX);
        assert!(jump.error < cnum::epsilon(70, prec));
        let half = Float::with_val_64(prec, &jump.steps) / 2;
        let first = cache.advance(&point, Some(&half)).unwrap().unwrap();
        let second = cache.advance(&first.value, Some(&half)).unwrap().unwrap();
        assert_eq!(first.steps + second.steps, jump.steps);
        let error = cnum::abs(&Complex::with_val_64(prec, jump.value - second.value), prec);
        assert!(error <= jump.error + first.error * second.derivative + second.error);
    }
}
