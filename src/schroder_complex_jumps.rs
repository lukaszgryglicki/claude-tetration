use rug::{
    float::Round,
    ops::{AddAssignRound, DivAssignRound, MulAssignRound},
    Complex, Float, Integer,
};

use crate::{cnum, schroder_jumps::Jump};

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
    jets: Vec<Vec<Coefficient>>,
}

impl ComplexOrbitJumps {
    pub fn new(
        ln_b: &Complex,
        fixed_point: &Complex,
        logarithm: bool,
        prec: u64,
    ) -> Result<Option<Self>, String> {
        cnum::init_mpfr();
        cnum::check_precision(prec)?;
        if !cnum::is_finite(ln_b) || !cnum::is_finite(fixed_point) {
            return Err("Complex Schroder jumps require finite parameters".into());
        }
        if (ln_b.imag().is_zero() && fixed_point.imag().is_zero()) || *fixed_point.real() <= 0 {
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
        if lower <= 0 || upper >= 1 {
            return Ok(None);
        }
        let order = usize::try_from(prec.div_ceil(4) + 16)
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
            jets: Vec::new(),
        }))
    }

    fn next_jet(&mut self) -> Result<(), String> {
        let prec = self.prec;
        let block = (self.order + 1).isqrt() + 1;
        cnum::check_float_storage(
            (self.order as u128 + 1) * (self.jets.len() as u128 + 4 * (block as u128 + 8)) * 4,
            prec,
        )?;
        let coefficients = if let Some(previous) = self.jets.last() {
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
            let radius = Float::with_val_64(prec, 1) / 4;
            for n in 2..=self.order {
                let mut value = result[n - 1].scale(&radius, false, &self.unit);
                if self.logarithm {
                    value = value.multiply(&inverse, &self.unit);
                    value = value.scale(&Float::with_val_64(prec, n - 1), false, &self.unit);
                }
                result[n] = value.scale(&Float::with_val_64(prec, n), true, &self.unit);
            }
            result
        };
        if coefficients.iter().any(|value| {
            !cnum::is_finite(&value.value) || !value.error.is_finite() || value.error < 0
        }) {
            return Err("Complex jump coefficients exceeded MPFR's range".into());
        }
        if cnum::verbose() {
            eprintln!(
                "complex {} jump: 2^{} steps, order {}, {} bits",
                if self.logarithm { "log" } else { "exp" },
                self.jets.len(),
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
        let mut steps = Integer::from(1);
        let mut level = 0usize;
        loop {
            let doubled = Integer::from(&steps * 2);
            if maximum.is_some_and(|limit| *limit < doubled)
                || Float::with_val_round_64(prec, &magnitude * &doubled, Round::Up).0
                    > Float::with_val_64(prec, 1) / 64
                || Float::with_val_round_64(prec, &self.gap * &doubled, Round::Up).0
                    > Float::with_val_64(prec, 1) / 8
            {
                break;
            }
            steps = doubled;
            level = level
                .checked_add(1)
                .ok_or("Complex jump table is not addressable")?;
        }
        if steps < self.minimum_steps {
            return Ok(None);
        }
        let mut radius = Float::with_val_64(prec, 1)
            >> level
                .checked_add(2)
                .ok_or("Complex jump radius is not addressable")?;
        let perturbation = loop {
            if radius.is_zero() {
                return Err("Complex jump radius exceeds MPFR's range".into());
            }
            let mut error = Float::with_val_round_64(prec, &self.defect * &steps, Round::Up).0;
            error.add_assign_round(&y.error, Round::Up);
            error.mul_assign_round(8, Round::Up);
            if error < Float::with_val_64(prec, &radius / 2) {
                break error;
            }
            steps >>= 1;
            if steps < self.minimum_steps {
                return Ok(None);
            }
            level -= 1;
            radius *= 2;
        };
        while self.jets.len() <= level {
            self.next_jet()?;
        }
        let mut input = Complex::with_val_64(prec, &y.value / &radius);
        if self.logarithm {
            input = -input;
        }
        let ratio = Float::with_val_round_64(prec, input.abs_ref(), Round::Up).0;
        if ratio > Float::with_val_64(prec, 1) / 16 {
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
            Float::with_val_round_64(prec, Float::with_val_64(prec, 1) - &ratio, Round::Down).0;
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
        derivative_perturbation.mul_assign_round(4, Round::Up);
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
