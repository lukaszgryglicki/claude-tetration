use rug::{
    float::Round,
    ops::{AddAssignRound, DivAssignRound, MulAssignRound},
    Complex, Float, Integer,
};

use crate::{cnum, schroder_complex_jumps::ComplexOrbitJumps};

struct TaylorJet {
    lower: Vec<Float>,
    upper: Vec<Float>,
}

pub(crate) enum OrbitJumps {
    Real(RealOrbitJumps),
    Complex(ComplexOrbitJumps),
}

impl OrbitJumps {
    pub fn new(
        ln_b: &Complex,
        fixed_point: &Complex,
        logarithm: bool,
        prec: u64,
    ) -> Result<Option<Self>, String> {
        if ln_b.imag().is_zero() && fixed_point.imag().is_zero() {
            Ok(RealOrbitJumps::new(ln_b, fixed_point, logarithm, prec)?.map(Self::Real))
        } else {
            Ok(ComplexOrbitJumps::new(ln_b, fixed_point, logarithm, prec)?.map(Self::Complex))
        }
    }

    pub fn advance(
        &mut self,
        point: &Complex,
        maximum: Option<&Float>,
    ) -> Result<Option<Jump>, String> {
        match self {
            Self::Real(real) => real.advance(point, maximum),
            Self::Complex(complex) => complex.advance(point, maximum),
        }
    }
}

pub(crate) struct RealOrbitJumps {
    logarithm: bool,
    prec: u64,
    order: usize,
    minimum_steps: Integer,
    lambda: Float,
    gap: Float,
    logarithm_base: Float,
    fixed_point: Float,
    defect: Float,
    unit: Float,
    jets: Vec<TaylorJet>,
}

pub(crate) struct Jump {
    pub value: Complex,
    pub derivative: Float,
    pub error: Float,
    pub steps: Integer,
}

fn distance_upper(a: &Float, b: &Float, prec: u64) -> Float {
    Float::with_val_round_64(prec, a - b, Round::AwayZero)
        .0
        .abs()
}

fn product(a: &[Float], b: &[Float], order: usize, prec: u64, round: Round) -> Vec<Float> {
    let mut result = vec![Float::new_64(prec); order + 1];
    for (i, x) in a.iter().enumerate().take(order + 1) {
        if x.is_zero() {
            continue;
        }
        for (j, y) in b.iter().enumerate().take(order + 1 - i) {
            if !y.is_zero() {
                let term = Float::with_val_round_64(prec, x * y, round).0;
                result[i + j].add_assign_round(&term, round);
            }
        }
    }
    result
}

fn compose(a: &[Float], b: &[Float], prec: u64, round: Round) -> Vec<Float> {
    let order = a.len() - 1;
    let block = (order + 1).isqrt() + 1;
    let mut powers = Vec::with_capacity(block + 1);
    let mut one = vec![Float::new_64(prec); order + 1];
    one[0] = Float::with_val_64(prec, 1);
    powers.push(one);
    powers.push(b.to_vec());
    for _ in 2..=block {
        powers.push(product(powers.last().unwrap(), b, order, prec, round));
    }
    let mut result = vec![Float::new_64(prec); order + 1];
    for group in (0..=order / block).rev() {
        if group < order / block {
            result = product(&result, &powers[block], order, prec, round);
        }
        for j in 0..block.min(order + 1 - group * block) {
            let weight = &a[group * block + j];
            if weight.is_zero() {
                continue;
            }
            for (output, power) in result.iter_mut().zip(&powers[j]) {
                if !power.is_zero() {
                    let term = Float::with_val_round_64(prec, weight * power, round).0;
                    output.add_assign_round(&term, round);
                }
            }
        }
    }
    result
}

impl RealOrbitJumps {
    pub fn new(
        ln_b: &Complex,
        fixed_point: &Complex,
        logarithm: bool,
        prec: u64,
    ) -> Result<Option<Self>, String> {
        cnum::init_mpfr();
        cnum::check_precision(prec)?;
        if !cnum::is_finite(ln_b) || !cnum::is_finite(fixed_point) {
            return Err("Schroder orbit jumps require finite base and fixed point".into());
        }
        if !ln_b.imag().is_zero()
            || !fixed_point.imag().is_zero()
            || *ln_b.real() <= 0
            || *fixed_point.real() <= 0
        {
            return Ok(None);
        }
        let lambda = Float::with_val_64(prec, ln_b.real() * fixed_point.real());
        if lambda <= 0 || lambda >= 1 {
            return Ok(None);
        }
        let order = usize::try_from(prec.div_ceil(4) + 16)
            .map_err(|_| "Schroder jump order exceeds addressable memory")?;
        let minimum_steps = Integer::from(
            order
                .checked_next_power_of_two()
                .ok_or("Schroder jump order exceeds addressable memory")?,
        );
        let one = Float::with_val_64(prec, 1);
        let gap = Float::with_val_round_64(prec, &one - &lambda, Round::Up).0;
        if Float::with_val_round_64(prec, &gap * &minimum_steps, Round::Up).0
            > Float::with_val_64(prec, 1) / 8
        {
            return Ok(None);
        }
        let block = (order + 1).isqrt() + 1;
        cnum::check_float_storage((order as u128 + 1) * (block as u128 + 8) * 4, prec)?;
        let unit = Float::with_val_64(prec, 1)
            >> usize::try_from(prec).map_err(|_| "Schroder jump precision is not addressable")?;
        if unit.is_zero() {
            return Err("Schroder jump roundoff unit exceeds MPFR's exponent range".into());
        }
        let q_lower =
            Float::with_val_round_64(prec, ln_b.real() * fixed_point.real(), Round::Down).0;
        let q_upper = Float::with_val_round_64(prec, ln_b.real() * fixed_point.real(), Round::Up).0;
        let product_error =
            distance_upper(&q_lower, &lambda, prec).max(&distance_upper(&q_upper, &lambda, prec));
        let defect = if logarithm {
            let lower = Float::with_val_round_64(prec, fixed_point.real().ln_ref(), Round::Down).0;
            let upper = Float::with_val_round_64(prec, fixed_point.real().ln_ref(), Round::Up).0;
            let mut defect =
                distance_upper(&lower, &q_upper, prec).max(&distance_upper(&upper, &q_lower, prec));
            defect.add_assign_round(product_error * 8, Round::Up);
            defect
        } else {
            let mut lower = Float::with_val_round_64(prec, q_lower.exp_ref(), Round::Down).0;
            let mut upper = Float::with_val_round_64(prec, q_upper.exp_ref(), Round::Up).0;
            lower.mul_assign_round(ln_b.real(), Round::Down);
            upper.mul_assign_round(ln_b.real(), Round::Up);
            let mut defect =
                distance_upper(&lower, &lambda, prec).max(&distance_upper(&upper, &lambda, prec));
            defect.mul_assign_round(3, Round::Up);
            defect.add_assign_round(product_error, Round::Up);
            defect
        };
        if !defect.is_finite() || q_lower <= Float::with_val_64(prec, 7) / 8 {
            return Err("Schroder orbit-jump fixed-point bounds are invalid".into());
        }
        Ok(Some(Self {
            logarithm,
            prec,
            order,
            minimum_steps,
            lambda,
            gap,
            logarithm_base: ln_b.real().clone(),
            fixed_point: fixed_point.real().clone(),
            defect,
            unit,
            jets: Vec::new(),
        }))
    }

    fn next_jet(&mut self) -> Result<(), String> {
        let prec = self.prec;
        let block = (self.order + 1).isqrt() + 1;
        cnum::check_float_storage(
            (self.order as u128 + 1)
                * (2 * (self.jets.len() as u128 + 1) + 4 * (block as u128 + 8)),
            prec,
        )?;
        let build = |round| {
            if let Some(previous) = self.jets.last() {
                let coefficients = if round == Round::Down {
                    &previous.lower
                } else {
                    &previous.upper
                };
                // For R_m=1/(4m), P_2m(u)=2*P_m(P_m(u/2)).
                let inner: Vec<_> = coefficients
                    .iter()
                    .enumerate()
                    .map(|(n, value)| Float::with_val_round_64(prec, value >> n, round).0)
                    .collect();
                let mut result = compose(coefficients, &inner, prec, round);
                for value in &mut result {
                    value.mul_assign_round(2, round);
                }
                result
            } else {
                let radius = Float::with_val_64(prec, 1) / 4;
                let mut result = vec![Float::new_64(prec); self.order + 1];
                result[1] = if self.logarithm {
                    Float::with_val_round_64(prec, self.lambda.recip_ref(), round).0
                } else {
                    self.lambda.clone()
                };
                for n in 2..=self.order {
                    let mut value =
                        Float::with_val_round_64(prec, &result[n - 1] * &radius, round).0;
                    if self.logarithm {
                        value.mul_assign_round(Integer::from(n - 1), round);
                        value.div_assign_round(&self.lambda, round);
                    }
                    value.div_assign_round(Integer::from(n), round);
                    result[n] = value;
                }
                result
            }
        };
        let lower = build(Round::Down);
        let upper = build(Round::Up);
        if lower
            .iter()
            .zip(&upper)
            .any(|(lo, hi)| *lo < 0 || lo > hi || !hi.is_finite())
        {
            return Err("Schroder orbit-jump coefficients exceeded MPFR's range".into());
        }
        if cnum::verbose() {
            eprintln!(
                "schroder {} jump: 2^{} steps, order {}, {} bits",
                if self.logarithm { "log" } else { "exp" },
                self.jets.len(),
                self.order,
                prec
            );
        }
        self.jets.push(TaylorJet { lower, upper });
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
            return Err(
                "Schroder orbit jump requires a finite point and integer step bound".into(),
            );
        }
        let y = (Complex::with_val_64(prec, point - &self.fixed_point)) * &self.logarithm_base;
        let magnitude = Float::with_val_round_64(prec, y.abs_ref(), Round::Up).0;
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
                .ok_or("Schroder jump table exceeds addressable memory")?;
        }
        if steps < self.minimum_steps {
            return Ok(None);
        }
        let mut input_error = Float::with_val_round_64(prec, point.abs_ref(), Round::Up).0;
        input_error.add_assign_round(&self.fixed_point, Round::Up);
        input_error.mul_assign_round(&self.logarithm_base, Round::Up);
        input_error.mul_assign_round(&self.unit, Round::Up);
        input_error.mul_assign_round(8, Round::Up);
        let mut radius = Float::with_val_64(prec, 1)
            >> level
                .checked_add(2)
                .ok_or("Schroder jump radius is not addressable")?;
        let perturbation = loop {
            if radius.is_zero() {
                return Err("Schroder jump radius exceeds MPFR's exponent range".into());
            }
            let mut error = Float::with_val_round_64(prec, &self.defect * &steps, Round::Up).0;
            error.add_assign_round(&input_error, Round::Up);
            error.mul_assign_round(8, Round::Up);
            // The analytic m-fold maps stay inside2R; reserve the disk3R for perturbations.
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
        let jet = &self.jets[level];
        let mut input = Complex::with_val_64(prec, y / &radius);
        if self.logarithm {
            input = -input;
        }
        let ratio = Float::with_val_round_64(prec, input.abs_ref(), Round::Up).0;
        if ratio > Float::with_val_64(prec, 1) / 16 {
            return Err("Schroder orbit jump is outside its analytic evaluation disk".into());
        }
        let mut value = cnum::zero(prec);
        let mut derivative = cnum::zero(prec);
        let mut coefficient_error = Float::new_64(prec);
        let mut derivative_error = Float::new_64(prec);
        let mut norm = Float::new_64(prec);
        for (lo, hi) in jet.lower.iter().zip(&jet.upper).rev() {
            derivative = Complex::with_val_64(prec, derivative * &input) + &value;
            value = Complex::with_val_64(prec, value * &input) + lo;
            derivative_error.mul_assign_round(&ratio, Round::Up);
            derivative_error.add_assign_round(&coefficient_error, Round::Up);
            coefficient_error.mul_assign_round(&ratio, Round::Up);
            coefficient_error.add_assign_round(
                Float::with_val_round_64(prec, hi - lo, Round::Up).0,
                Round::Up,
            );
            norm.mul_assign_round(&ratio, Round::Up);
            norm.add_assign_round(hi, Round::Up);
        }
        let mut power = Float::with_val_64(prec, 1);
        for _ in 0..self.order {
            power.mul_assign_round(&ratio, Round::Up);
        }
        let one = Float::with_val_64(prec, 1);
        let denominator = Float::with_val_round_64(prec, &one - &ratio, Round::Down).0;
        let mut tail = Float::with_val_round_64(prec, &power * &ratio, Round::Up).0;
        tail.mul_assign_round(2, Round::Up);
        tail.div_assign_round(&denominator, Round::Up);
        let n = Integer::from(self.order + 1);
        let mut gamma = Float::with_val_round_64(prec, &self.unit * &n, Round::Up).0;
        gamma.mul_assign_round(32, Round::Up);
        if gamma >= 1 {
            return Err("Schroder orbit-jump precision does not resolve Horner roundoff".into());
        }
        let roundoff_denominator = Float::with_val_round_64(prec, &one - &gamma, Round::Down).0;
        gamma.div_assign_round(&roundoff_denominator, Round::Up);
        let mut error = coefficient_error;
        error.add_assign_round(&tail, Round::Up);
        error.add_assign_round(
            Float::with_val_round_64(prec, &gamma * &norm, Round::Up).0,
            Round::Up,
        );
        error.mul_assign_round(&radius, Round::Up);
        error.add_assign_round(&perturbation, Round::Up);

        let mut derivative_tail = power;
        derivative_tail.mul_assign_round(&n, Round::Up);
        derivative_tail.mul_assign_round(2, Round::Up);
        let denominator_squared =
            Float::with_val_round_64(prec, denominator.square_ref(), Round::Down).0;
        derivative_tail.div_assign_round(&denominator_squared, Round::Up);
        derivative_error.add_assign_round(&derivative_tail, Round::Up);
        let n_squared = n.square();
        let mut derivative_roundoff =
            Float::with_val_round_64(prec, &self.unit * &n_squared, Round::Up).0;
        derivative_roundoff.mul_assign_round(256, Round::Up);
        derivative_roundoff.div_assign_round(&roundoff_denominator, Round::Up);
        derivative_error.add_assign_round(&derivative_roundoff, Round::Up);
        let mut derivative_perturbation =
            Float::with_val_round_64(prec, &perturbation / &radius, Round::Up).0;
        derivative_perturbation.mul_assign_round(4, Round::Up);
        derivative_error.add_assign_round(&derivative_perturbation, Round::Up);
        let mut derivative = Float::with_val_round_64(prec, derivative.abs_ref(), Round::Up).0;
        derivative.add_assign_round(&derivative_error, Round::Up);

        value *= &radius;
        if self.logarithm {
            value = -value;
        }
        value /= &self.logarithm_base;
        let mut output_error = Float::with_val_round_64(prec, value.abs_ref(), Round::Up).0;
        output_error.add_assign_round(&self.fixed_point, Round::Up);
        output_error.mul_assign_round(&self.unit, Round::Up);
        output_error.mul_assign_round(8, Round::Up);
        value += &self.fixed_point;
        if point.imag().is_zero() && value.imag().is_zero() {
            *value.mut_imag() = Float::new_64(prec);
        }
        error.div_assign_round(&self.logarithm_base, Round::Up);
        error.add_assign_round(&output_error, Round::Up);
        if !cnum::is_finite(&value) || !error.is_finite() || !derivative.is_finite() {
            return Err("Schroder orbit-jump evaluation exceeded MPFR's range".into());
        }
        Ok(Some(Jump {
            value,
            derivative,
            error,
            steps,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters(prec: u64, gap: &str) -> (Complex, Complex) {
        let lambda: Float = Float::with_val_64(prec, 1) - cnum::decimal(gap, prec);
        let fixed = Float::with_val_64(prec, lambda.exp_ref());
        let logarithm = Float::with_val_64(prec, &lambda / &fixed);
        (
            Complex::with_val_64(prec, logarithm),
            Complex::with_val_64(prec, fixed),
        )
    }

    #[test]
    fn jumps_enclose_direct_orbits_and_derivatives() {
        let prec = cnum::digits_to_bits(70);
        let high = cnum::digits_to_bits(180);
        let (logarithm, fixed) = parameters(prec, "1e-8");
        for backward in [false, true] {
            let mut jumps = OrbitJumps::new(&logarithm, &fixed, backward, prec)
                .unwrap()
                .unwrap();
            for (re, im) in [
                ("-0.00001", "0"),
                ("0.00001", "0.00001"),
                ("0.00001", "-0.00001"),
            ] {
                let y = cnum::parse_complex(re, im, prec).unwrap();
                let point = Complex::with_val_64(prec, y / &logarithm) + &fixed;
                let maximum = Float::with_val_64(prec, 1500);
                let jump = jumps.advance(&point, Some(&maximum)).unwrap().unwrap();
                assert_eq!(jump.steps, 1024);
                let mut expected = Complex::with_val_64(high, &point);
                let a = Complex::with_val_64(high, &logarithm);
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
                let error = cnum::abs(&Complex::with_val_64(high, &jump.value - expected), high);
                assert!(
                    error <= jump.error,
                    "{backward} {re}+{im}i: {error} > {}",
                    jump.error
                );
                assert!(cnum::abs(&derivative, high) <= jump.derivative);
                assert!(jump.error < cnum::epsilon(70, prec));
                if point.imag().is_zero() {
                    assert!(jump.value.imag().is_zero());
                    assert!(!jump.value.imag().is_sign_negative());
                }
                assert!(jumps
                    .advance(&point, Some(&Float::with_val_64(prec, 3)))
                    .unwrap()
                    .is_none());
            }
        }
    }

    #[test]
    fn jumps_account_for_a_rounded_fixed_point() {
        let prec = cnum::digits_to_bits(70);
        let high = cnum::digits_to_bits(180);
        let (logarithm, mut fixed) = parameters(prec, "1e-8");
        fixed -= cnum::decimal("1e-8", prec);
        let point = Complex::with_val_64(prec, &fixed - cnum::decimal("1e-5", prec));
        for backward in [false, true] {
            let mut jumps = OrbitJumps::new(&logarithm, &fixed, backward, prec)
                .unwrap()
                .unwrap();
            let jump = jumps
                .advance(&point, Some(&Float::with_val_64(prec, 1024)))
                .unwrap()
                .unwrap();
            let mut expected = Complex::with_val_64(high, &point);
            let a = Complex::with_val_64(high, &logarithm);
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
    fn jump_applicability_preserves_other_regimes_and_rejects_invalid_inputs() {
        let prec = cnum::digits_to_bits(70);
        let (a, fixed) = parameters(prec, "1e-8");
        let complex_a = a.clone() + cnum::parse_complex("0", "1e-100", prec).unwrap();
        assert!(matches!(
            OrbitJumps::new(&complex_a, &fixed, true, prec).unwrap(),
            Some(OrbitJumps::Complex(_))
        ));
        for logarithm in [-a.clone(), cnum::one(prec)] {
            assert!(OrbitJumps::new(&logarithm, &fixed, true, prec)
                .unwrap()
                .is_none());
        }
        let (ordinary_a, ordinary_fixed) = parameters(prec, "0.1");
        assert!(OrbitJumps::new(&ordinary_a, &ordinary_fixed, true, prec)
            .unwrap()
            .is_none());
        assert!(OrbitJumps::new(&a, &fixed, true, u64::MAX).is_err());
        let nan = Complex::with_val_64(prec, rug::float::Special::Nan);
        assert!(OrbitJumps::new(&nan, &fixed, true, prec).is_err());
        let mut jumps = OrbitJumps::new(&a, &fixed, true, prec).unwrap().unwrap();
        assert!(jumps.advance(&nan, None).is_err());
        for maximum in [
            Float::with_val_64(prec, -1),
            cnum::decimal("0.5", prec),
            Float::with_val_64(prec, rug::float::Special::Infinity),
        ] {
            assert!(jumps.advance(&fixed, Some(&maximum)).is_err());
        }
        assert!(jumps
            .advance(&fixed, Some(&Float::new_64(prec)))
            .unwrap()
            .is_none());
        assert!(jumps.advance(&cnum::one(prec), None).unwrap().is_none());
    }

    #[test]
    fn jump_counts_exceed_machine_integers() {
        let prec = cnum::digits_to_bits(100);
        let (a, fixed) = parameters(prec, "1e-30");
        let point = &fixed + cnum::parse_complex("-1e-32", "1e-33", prec).unwrap();
        let point = Complex::with_val_64(prec, point);
        let mut jumps = OrbitJumps::new(&a, &fixed, true, prec).unwrap().unwrap();
        let jump = jumps.advance(&point, None).unwrap().unwrap();
        assert!(jump.steps > u64::MAX);
        assert!(cnum::is_finite(&jump.value));
        assert!(jump.error < cnum::epsilon(70, prec));
        let half = Float::with_val_64(prec, &jump.steps) / 2;
        let first = jumps.advance(&point, Some(&half)).unwrap().unwrap();
        let second = jumps.advance(&first.value, Some(&half)).unwrap().unwrap();
        assert_eq!(first.steps + second.steps, jump.steps);
        let difference = cnum::abs(&Complex::with_val_64(prec, jump.value - second.value), prec);
        assert!(difference <= jump.error + first.error * second.derivative + second.error);
    }
}
