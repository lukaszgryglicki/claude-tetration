//! Named real constants for the textual inputs (`pi`, `e`, `eta`, ...) and
//! for the constant expressions of `expression.rs`.
//!
//! A name stands for the constant at whatever precision the input is read
//! at: a constant of MPFR is rounded correctly, every other one is computed
//! with guard bits from correctly rounded functions, from a series whose
//! tail is bounded, or from a Newton iteration whose residual is checked.

use rug::{float::Constant, Complex, Float, Integer};

use crate::{cnum, lambertw};

/// One named constant.
pub struct Named {
    pub name: &'static str,
    /// Other names of the constant.
    pub aliases: &'static [&'static str],
    pub meaning: &'static str,
    value: fn(u64) -> Result<Float, String>,
}

/// Every named constant, in the order of the documentation.
pub const TABLE: &[Named] = &[
    Named {
        name: "pi",
        aliases: &["π"],
        meaning: "circle constant",
        value: pi,
    },
    Named {
        name: "e",
        aliases: &[],
        meaning: "Euler's number",
        value: |prec| closed(prec, |work| Ok(one(work).exp())),
    },
    Named {
        name: "tau",
        aliases: &["τ"],
        meaning: "2 pi",
        value: |prec| Ok(pi(prec)? * 2u32),
    },
    Named {
        name: "halfpi",
        aliases: &[],
        meaning: "pi/2",
        value: |prec| Ok(pi(prec)? / 2u32),
    },
    Named {
        name: "deg",
        aliases: &["degree"],
        meaning: "one degree in radians, pi/180",
        value: |prec| closed(prec, |work| Ok(pi(work)? / 180u32)),
    },
    Named {
        name: "sqrtpi",
        aliases: &["√π"],
        meaning: "square root of pi",
        value: |prec| closed(prec, |work| Ok(pi(work)?.sqrt())),
    },
    Named {
        name: "eta",
        aliases: &["η", "towermax"],
        meaning: "e^(1/e): the largest base whose infinite tower converges",
        value: |prec| closed(prec, |work| Ok((-one(work)).exp().exp())),
    },
    Named {
        name: "towermin",
        aliases: &[],
        meaning: "e^(-e): the smallest base whose infinite tower converges",
        value: |prec| closed(prec, |work| Ok((-one(work).exp()).exp())),
    },
    Named {
        name: "einv",
        aliases: &[],
        meaning: "1/e",
        value: |prec| closed(prec, |work| Ok((-one(work)).exp())),
    },
    Named {
        name: "gelfond",
        aliases: &[],
        meaning: "Gelfond's constant e^pi",
        value: |prec| closed(prec, |work| Ok(pi(work)?.exp())),
    },
    Named {
        name: "gelfondschneider",
        aliases: &[],
        meaning: "Gelfond-Schneider constant 2^sqrt(2)",
        value: |prec| closed(prec, |work| Ok((root(work, 2) * ln2(work)).exp())),
    },
    Named {
        name: "ramanujan",
        aliases: &[],
        meaning: "Ramanujan's constant e^(pi sqrt 163)",
        value: |prec| closed(prec, |work| Ok((pi(work)? * root(work, 163)).exp())),
    },
    Named {
        name: "phi",
        aliases: &["φ", "golden"],
        meaning: "golden ratio (1 + sqrt 5)/2",
        value: |prec| closed(prec, |work| Ok((root(work, 5) + 1u32) / 2u32)),
    },
    Named {
        name: "silver",
        aliases: &[],
        meaning: "silver ratio 1 + sqrt 2",
        value: |prec| closed(prec, |work| Ok(root(work, 2) + 1u32)),
    },
    Named {
        name: "plastic",
        aliases: &[],
        meaning: "plastic number: x^3 = x + 1",
        value: |prec| {
            closed(prec, |work| {
                let part = root(work, 69) / 18u32;
                let half = Float::with_val_64(work, 0.5);
                let upper = Float::with_val_64(work, &half + &part).cbrt();
                Ok(upper + Float::with_val_64(work, &half - &part).cbrt())
            })
        },
    },
    Named {
        name: "supergolden",
        aliases: &[],
        meaning: "supergolden ratio: x^3 = x^2 + 1",
        value: |prec| {
            closed(prec, |work| {
                let part = root(work, 93) * 3u32;
                let upper = ((Float::with_val_64(work, &part + 29u32)) / 2u32).cbrt();
                let lower = ((Float::with_val_64(work, 29u32 - &part)) / 2u32).cbrt();
                Ok((upper + lower + 1u32) / 3u32)
            })
        },
    },
    Named {
        name: "tribonacci",
        aliases: &[],
        meaning: "tribonacci constant: x^3 = x^2 + x + 1",
        value: |prec| {
            closed(prec, |work| {
                let part = root(work, 33) * 3u32;
                let upper = Float::with_val_64(work, &part + 19u32).cbrt();
                let lower = Float::with_val_64(work, 19u32 - &part).cbrt();
                Ok((upper + lower + 1u32) / 3u32)
            })
        },
    },
    Named {
        name: "sqrt2",
        aliases: &["√2"],
        meaning: "square root of 2",
        value: |prec| Ok(root(prec, 2)),
    },
    Named {
        name: "sqrt3",
        aliases: &["√3"],
        meaning: "square root of 3",
        value: |prec| Ok(root(prec, 3)),
    },
    Named {
        name: "sqrt5",
        aliases: &["√5"],
        meaning: "square root of 5",
        value: |prec| Ok(root(prec, 5)),
    },
    Named {
        name: "ln2",
        aliases: &[],
        meaning: "natural logarithm of 2",
        value: |prec| Ok(ln2(prec)),
    },
    Named {
        name: "ln10",
        aliases: &[],
        meaning: "natural logarithm of 10",
        value: |prec| Ok(Float::with_val_64(prec, 10).ln()),
    },
    Named {
        name: "gamma",
        aliases: &["γ", "eulergamma"],
        meaning: "Euler-Mascheroni constant",
        value: |prec| Ok(euler(prec)),
    },
    Named {
        name: "catalan",
        aliases: &[],
        meaning: "Catalan's constant",
        value: |prec| Ok(Float::with_val_64(prec, Constant::Catalan)),
    },
    Named {
        name: "apery",
        aliases: &["zeta3"],
        meaning: "Apery's constant zeta(3)",
        value: |prec| Ok(Float::with_val_64(prec, Float::zeta_u(3))),
    },
    Named {
        name: "zeta2",
        aliases: &["basel"],
        meaning: "zeta(2) = pi^2/6",
        value: |prec| closed(prec, |work| Ok(pi(work)?.square() / 6u32)),
    },
    Named {
        name: "omega",
        aliases: &["Ω"],
        meaning: "omega constant W(1): omega e^omega = 1",
        value: |prec| {
            closed(prec, |work| {
                let w = lambertw::w0(&Complex::with_val_64(work, 1), work)?;
                Ok(Float::with_val_64(work, w.real()))
            })
        },
    },
    Named {
        name: "dottie",
        aliases: &[],
        meaning: "Dottie number: cos x = x",
        value: dottie,
    },
    Named {
        name: "lemniscate",
        aliases: &["ϖ"],
        meaning: "lemniscate constant Gamma(1/4)^2/(2 sqrt(2 pi))",
        value: |prec| {
            closed(prec, |work| {
                let quarter = Float::with_val_64(work, 0.25).gamma().square();
                Ok(quarter / ((pi(work)? * 2u32).sqrt() * 2u32))
            })
        },
    },
    Named {
        name: "gauss",
        aliases: &[],
        meaning: "Gauss's constant 1/agm(1, sqrt 2)",
        value: |prec| closed(prec, |work| Ok(one(work).agm(&root(work, 2)).recip())),
    },
    Named {
        name: "parabolic",
        aliases: &[],
        meaning: "universal parabolic constant sqrt 2 + ln(1 + sqrt 2)",
        value: |prec| {
            closed(prec, |work| {
                let sqrt2 = root(work, 2);
                Ok(Float::with_val_64(work, &sqrt2 + 1u32).ln() + sqrt2)
            })
        },
    },
    Named {
        name: "levy",
        aliases: &[],
        meaning: "Levy's constant e^(pi^2/(12 ln 2))",
        value: |prec| {
            closed(prec, |work| {
                Ok((pi(work)?.square() / (ln2(work) * 12u32)).exp())
            })
        },
    },
    Named {
        name: "khinchin",
        aliases: &[],
        meaning: "Khinchin's constant",
        value: khinchin,
    },
    Named {
        name: "gompertz",
        aliases: &[],
        meaning: "Euler-Gompertz constant",
        value: gompertz,
    },
    Named {
        name: "soldner",
        aliases: &[],
        meaning: "Ramanujan-Soldner constant: zero of the logarithmic integral",
        value: soldner,
    },
    Named {
        name: "laplace",
        aliases: &[],
        meaning: "Laplace limit",
        value: laplace,
    },
    Named {
        name: "erdosborwein",
        aliases: &[],
        meaning: "Erdos-Borwein constant: sum of 1/(2^n - 1)",
        value: erdos_borwein,
    },
    Named {
        name: "cahen",
        aliases: &[],
        meaning: "Cahen's constant",
        value: cahen,
    },
    Named {
        name: "recfib",
        aliases: &[],
        meaning: "reciprocal Fibonacci constant: sum of 1/F_n",
        value: reciprocal_fibonacci,
    },
    Named {
        name: "niven",
        aliases: &[],
        meaning: "Niven's constant",
        value: niven,
    },
    Named {
        name: "mertens",
        aliases: &[],
        meaning: "Meissel-Mertens constant",
        value: mertens,
    },
    Named {
        name: "twinprime",
        aliases: &[],
        meaning: "twin prime constant",
        value: twin_prime,
    },
];

impl Named {
    /// The constant rounded to `prec` bits.
    pub fn value(&self, prec: u64) -> Result<Float, String> {
        cnum::init_mpfr();
        cnum::check_precision(prec)?;
        let value = (self.value)(prec).map_err(|reason| format!("{}: {reason}", self.name))?;
        if !value.is_finite() {
            return Err(format!("{}: the constant is not finite", self.name));
        }
        Ok(Float::with_val_64(prec, value))
    }
}

/// The constant with the name or alias `text`. The case of ASCII letters
/// does not matter; a Greek alias is taken as written.
pub fn named(text: &str) -> Option<&'static Named> {
    let lower = text.to_ascii_lowercase();
    TABLE
        .iter()
        .find(|named| named.name == lower || named.aliases.contains(&lower.as_str()))
        .or_else(|| TABLE.iter().find(|named| named.aliases.contains(&text)))
}

/// The list of the constants for the usage text.
pub fn help() -> String {
    let mut text = String::new();
    for named in TABLE {
        let mut names = vec![named.name];
        names.extend(named.aliases);
        text.push_str(&format!("    {:<28} {}\n", names.join(", "), named.meaning));
    }
    text
}

/// The precision of a computation that ends with `prec` bits.
fn guarded(prec: u64, guard: u64) -> Result<u64, String> {
    let work = prec.checked_add(guard).ok_or("precision count overflow")?;
    cnum::check_precision(work)?;
    Ok(work)
}

/// A closed form of correctly rounded functions, computed with guard bits.
fn closed(prec: u64, form: impl Fn(u64) -> Result<Float, String>) -> Result<Float, String> {
    form(guarded(prec, 64)?)
}

fn one(prec: u64) -> Float {
    Float::with_val_64(prec, 1)
}

fn pi(prec: u64) -> Result<Float, String> {
    Ok(Float::with_val_64(prec, Constant::Pi))
}

fn ln2(prec: u64) -> Float {
    Float::with_val_64(prec, Constant::Log2)
}

fn euler(prec: u64) -> Float {
    Float::with_val_64(prec, Constant::Euler)
}

fn root(prec: u64, of: u32) -> Float {
    Float::with_val_64(prec, of).sqrt()
}

/// `2^-bits` at the precision `prec`.
fn tiny(prec: u64, bits: u64) -> Result<Float, String> {
    let shift = usize::try_from(bits).map_err(|_| "precision is not addressable")?;
    Ok(one(prec) >> shift)
}

/// The number of bits of a term count, as guard bits for its sum.
fn count_bits(count: u64) -> u64 {
    u64::from(u64::BITS - count.leading_zeros())
}

/// Newton's iteration `x -> x - step(x)` from `start` at `work` bits until
/// the step is below the last bits; `residual` is then checked there.
fn newton(
    work: u64,
    start: Float,
    step: impl Fn(&Float) -> Result<Float, String>,
    residual: impl Fn(&Float) -> Result<Float, String>,
) -> Result<Float, String> {
    let small = tiny(work, work - 8)?;
    let mut x = start;
    for _ in 0..256 {
        let change = step(&x)?;
        if !change.is_finite() {
            return Err("Newton's iteration left the real numbers".into());
        }
        let settled = change.clone().abs() <= Float::with_val_64(work, &small * x.clone().abs());
        x -= change;
        if settled {
            let miss = residual(&x)?.abs();
            if miss > tiny(work, work - 16)? {
                return Err("Newton's iteration missed its root".into());
            }
            return Ok(x);
        }
    }
    Err("Newton's iteration did not settle".into())
}

/// `cos x = x`.
fn dottie(prec: u64) -> Result<Float, String> {
    let work = guarded(prec, 64)?;
    newton(
        work,
        Float::with_val_64(work, 3) / 4u32,
        |x| {
            let (sin, cos) = x.clone().sin_cos(Float::new_64(work));
            Ok(Float::with_val_64(work, x - cos) / (sin + 1u32))
        },
        |x| Ok(Float::with_val_64(work, x - x.clone().cos())),
    )
}

/// The zero of `ln x + s - ln(1 + s)`, `s = sqrt(1 + x^2)`.
fn laplace(prec: u64) -> Result<Float, String> {
    let work = guarded(prec, 64)?;
    let value = |x: &Float| {
        let s = (x.clone().square() + 1u32).sqrt();
        let defect = x.clone().ln() + &s - Float::with_val_64(work, &s + 1u32).ln();
        (defect, s)
    };
    newton(
        work,
        Float::with_val_64(work, 2) / 3u32,
        |x| {
            let (defect, s) = value(x);
            let slope = x.clone().recip() + Float::with_val_64(work, x / (s + 1u32));
            Ok(defect / slope)
        },
        |x| Ok(value(x).0),
    )
}

/// `Ei(x) = gamma + ln x + sum x^k/(k k!)` for `0 < x < 1`: the terms fall
/// faster than `1/k!`, so the first one below the last bit ends the sum.
fn exponential_integral(x: &Float, work: u64) -> Result<Float, String> {
    let small = tiny(work, work + 4)?;
    let mut sum = euler(work) + x.clone().ln();
    let mut power = one(work);
    let mut factorial = Integer::from(1);
    let mut k = 0u32;
    loop {
        k += 1;
        power *= x;
        factorial *= k;
        let scale = Float::with_val_64(work, &factorial) * k;
        let term = Float::with_val_64(work, &power / scale);
        sum += &term;
        if term < small {
            return Ok(sum);
        }
    }
}

/// `e^x` at the zero `x` of `Ei`.
fn soldner(prec: u64) -> Result<Float, String> {
    let work = guarded(prec, 64)?;
    let x = newton(
        work,
        Float::with_val_64(work, 3) / 8u32,
        |x| Ok(exponential_integral(x, work)? * x / x.clone().exp()),
        |x| exponential_integral(x, work),
    )?;
    Ok(x.exp())
}

/// `e E_1(1)` with `E_1(1) = -gamma + sum (-1)^(k+1)/(k k!)`: an alternating
/// series with falling terms, ended by the first term below the last bit.
fn gompertz(prec: u64) -> Result<Float, String> {
    let work = guarded(prec, 64)?;
    let small = tiny(work, work + 4)?;
    let mut sum = -euler(work);
    let mut factorial = Integer::from(1);
    let mut k = 0u32;
    loop {
        k += 1;
        factorial *= k;
        let term = (Float::with_val_64(work, &factorial) * k).recip();
        if k % 2 == 1 {
            sum += &term;
        } else {
            sum -= &term;
        }
        if term < small {
            return Ok(sum * one(work).exp());
        }
    }
}

/// `sum 1/(2^n - 1)`: the tail after `n = N` is below `2^(1-N)`.
fn erdos_borwein(prec: u64) -> Result<Float, String> {
    let terms = guarded(prec, 72)?;
    let work = guarded(prec, 64 + count_bits(terms))?;
    let mut sum = Float::new_64(work);
    let mut power = Integer::from(1);
    for _ in 0..terms {
        power <<= 1u32;
        sum += Float::with_val_64(work, &Integer::from(&power - 1u32)).recip();
    }
    Ok(sum)
}

/// `sum (-1)^i/(s_i - 1)` over Sylvester's sequence `s_0 = 2`,
/// `s_(i+1) = s_i^2 - s_i + 1`: alternating with terms that fall, so the
/// first term below the last bit bounds the rest.
fn cahen(prec: u64) -> Result<Float, String> {
    let work = guarded(prec, 64)?;
    let small = tiny(work, work + 4)?;
    let mut sum = Float::new_64(work);
    let mut s = Integer::from(2);
    let mut positive = true;
    loop {
        let term = Float::with_val_64(work, &Integer::from(&s - 1u32)).recip();
        if positive {
            sum += &term;
        } else {
            sum -= &term;
        }
        if term < small {
            return Ok(sum);
        }
        positive = !positive;
        s = Integer::from(&s * &s) - &s + 1u32;
    }
}

/// `sum 1/F_n`: `F_(n+1) >= (3/2) F_n` from `n = 2` on, so the tail after
/// a term is at most twice that term.
fn reciprocal_fibonacci(prec: u64) -> Result<Float, String> {
    // F_n exceeds 2^(n/2 - 1): this many terms reach the last bit.
    let terms = guarded(prec, 80)?
        .checked_mul(2)
        .ok_or("precision count overflow")?;
    let work = guarded(prec, 64 + count_bits(terms))?;
    let small = tiny(work, work + 8)?;
    let mut sum = Float::new_64(work);
    let (mut current, mut next) = (Integer::from(1), Integer::from(1));
    loop {
        let term = Float::with_val_64(work, &current).recip();
        sum += &term;
        if term < small {
            return Ok(sum);
        }
        let following = Integer::from(&current + &next);
        current = std::mem::replace(&mut next, following);
    }
}

/// `zeta(n) - 1` with `bits` correct bits after its leading one: the
/// difference is about `2^-n`.
fn zeta_excess(n: u32, bits: u64) -> Result<Float, String> {
    let prec = guarded(bits, u64::from(n) + 16)?;
    Ok(Float::with_val_64(prec, Float::zeta_u(n)) - 1u32)
}

/// The number of terms after which a series with terms below `2^-n` has
/// reached the last bit of `work` bits.
fn dyadic_terms(work: u64) -> Result<u32, String> {
    u32::try_from(work + 8).map_err(|_| "precision is too large for this constant".into())
}

/// `ln K = (1/ln 2) sum_{n>=1} (zeta(2n) - 1)/n (1 - 1/2 + ... + 1/(2n-1))`
/// (Bailey, Borwein, Crandall): the terms are below `2 4^-n`.
fn khinchin(prec: u64) -> Result<Float, String> {
    let terms = dyadic_terms(guarded(prec, 64)?)? / 2 + 2;
    let work = guarded(prec, 64 + count_bits(u64::from(terms)))?;
    let mut sum = Float::new_64(work);
    let mut alternating = Float::new_64(work);
    for n in 1..=terms {
        if n > 1 {
            alternating -= Float::with_val_64(work, 2 * n - 2).recip();
        }
        alternating += Float::with_val_64(work, 2 * n - 1).recip();
        let excess = zeta_excess(2 * n, work)?;
        sum += Float::with_val_64(work, excess * &alternating) / n;
    }
    Ok((sum / ln2(work)).exp())
}

/// `1 + sum_{k>=2} (1 - 1/zeta(k))`: the terms are below `2^(1-k)`.
fn niven(prec: u64) -> Result<Float, String> {
    let terms = dyadic_terms(guarded(prec, 64)?)?;
    let work = guarded(prec, 64 + count_bits(u64::from(terms)))?;
    let mut sum = one(work);
    for k in 2..=terms + 1 {
        let excess = zeta_excess(k, work)?;
        sum += Float::with_val_64(work, &excess / Float::with_val_64(work, &excess + 1u32));
    }
    Ok(sum)
}

/// The Moebius function.
fn moebius(mut n: u32) -> i32 {
    let mut sign = 1;
    let mut prime = 2;
    while prime * prime <= n {
        if n.is_multiple_of(prime) {
            n /= prime;
            if n.is_multiple_of(prime) {
                return 0;
            }
            sign = -sign;
        }
        prime += 1;
    }
    if n > 1 {
        -sign
    } else {
        sign
    }
}

/// `gamma + sum_{n>=2} mu(n)/n ln zeta(n)`: the terms are below `2^(1-n)`.
fn mertens(prec: u64) -> Result<Float, String> {
    let terms = dyadic_terms(guarded(prec, 64)?)?;
    let work = guarded(prec, 64 + count_bits(u64::from(terms)))?;
    let mut sum = euler(work);
    for n in 2..=terms + 1 {
        let sign = moebius(n);
        if sign == 0 {
            continue;
        }
        let term = Float::with_val_64(work, zeta_excess(n, work)?.ln_1p()) / n;
        if sign > 0 {
            sum += term;
        } else {
            sum -= term;
        }
    }
    Ok(sum)
}

/// `prod_{n>=2} (zeta(n) (1 - 2^-n))^(-I_n)` with
/// `I_n = (1/n) sum_{d|n} mu(d) 2^(n/d)` (Flajolet, Vardi). The factor is
/// `1 + 3^-n + 5^-n + ...` and `I_n < 2^n/n`, so the logarithms of the
/// terms are below `2 (2/3)^n`.
fn twin_prime(prec: u64) -> Result<Float, String> {
    // (2/3)^n falls below 2^-(work + 8) after (work + 8)/log2(3/2) terms.
    let reach = guarded(prec, 72)?
        .checked_mul(12)
        .ok_or("precision count overflow")?
        / 7;
    let terms = u32::try_from(reach + 2).map_err(|_| "precision is too large for this constant")?;
    let work = guarded(prec, 64 + count_bits(u64::from(terms)))?;
    let mut sum = Float::new_64(work);
    for n in 2..=terms {
        let mut weight = Integer::new();
        for divisor in (1..=n).filter(|divisor| n.is_multiple_of(*divisor)) {
            let sign = moebius(divisor);
            if sign != 0 {
                let power = Integer::from(1) << (n / divisor);
                weight += power * sign;
            }
        }
        let weight = weight.div_exact(&Integer::from(n));
        // zeta(n) (1 - 2^-n) - 1 is about 3^-n, the difference of two
        // numbers of size 2^-n: a relative loss of (3/2)^n < 2^n.
        let zeta = zeta_excess(n, guarded(work, u64::from(n))?)?;
        let shift = usize::try_from(n).map_err(|_| "precision is not addressable")?;
        let prec = zeta.prec_64();
        let factor = Float::with_val_64(
            prec,
            &zeta - (Float::with_val_64(prec, &zeta + 1u32) >> shift),
        );
        sum += Float::with_val_64(work, factor.ln_1p()) * weight;
    }
    Ok((-sum).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::ops::Pow;

    fn power(x: &Float, y: u32) -> Float {
        x.clone().pow(y)
    }

    /// 70 significant digits of every constant from mpmath, an independent
    /// implementation.
    const REFERENCES: &[(&str, &str)] = &[
        (
            "pi",
            "3.141592653589793238462643383279502884197169399375105820974944592307816",
        ),
        (
            "e",
            "2.718281828459045235360287471352662497757247093699959574966967627724077",
        ),
        (
            "tau",
            "6.283185307179586476925286766559005768394338798750211641949889184615633",
        ),
        (
            "phi",
            "1.618033988749894848204586834365638117720309179805762862135448622705260",
        ),
        (
            "gamma",
            "0.5772156649015328606065120900824024310421593359399235988057672348848677",
        ),
        (
            "catalan",
            "0.9159655941772190150546035149323841107741493742816721342664981196217630",
        ),
        (
            "apery",
            "1.202056903159594285399738161511449990764986292340498881792271555341838",
        ),
        (
            "zeta2",
            "1.644934066848226436472415166646025189218949901206798437735558229370007",
        ),
        (
            "ln2",
            "0.6931471805599453094172321214581765680755001343602552541206800094933936",
        ),
        (
            "ln10",
            "2.302585092994045684017991454684364207601101488628772976033327900967573",
        ),
        (
            "sqrt2",
            "1.414213562373095048801688724209698078569671875376948073176679737990732",
        ),
        (
            "sqrt3",
            "1.732050807568877293527446341505872366942805253810380628055806979451933",
        ),
        (
            "sqrt5",
            "2.236067977499789696409173668731276235440618359611525724270897245410521",
        ),
        (
            "sqrtpi",
            "1.772453850905516027298167483341145182797549456122387128213807789852911",
        ),
        (
            "halfpi",
            "1.570796326794896619231321691639751442098584699687552910487472296153908",
        ),
        (
            "eta",
            "1.444667861009766133658339108596430223058595453242253165820522664303855",
        ),
        (
            "towermin",
            "0.06598803584531253707679018759684642493857704825279643640247354156673633",
        ),
        (
            "einv",
            "0.3678794411714423215955237701614608674458111310317678345078368016974615",
        ),
        (
            "omega",
            "0.5671432904097838729999686622103555497538157871865125081351310792230458",
        ),
        (
            "dottie",
            "0.7390851332151606416553120876738734040134117589007574649656806357732847",
        ),
        (
            "gelfond",
            "23.14069263277926900572908636794854738026610624260021199344504640952434",
        ),
        (
            "gelfondschneider",
            "2.665144142690225188650297249873139848274211313714659492835979593364920",
        ),
        (
            "ramanujan",
            "262537412640768743.9999999999992500725971981856888793538563373369908627",
        ),
        (
            "silver",
            "2.414213562373095048801688724209698078569671875376948073176679737990732",
        ),
        (
            "plastic",
            "1.324717957244746025960908854478097340734404056901733364534015050302828",
        ),
        (
            "supergolden",
            "1.465571231876768026656731225219939108025577568472285701643183111249263",
        ),
        (
            "tribonacci",
            "1.839286755214161132551852564653286600424178746097592246778758639404203",
        ),
        (
            "lemniscate",
            "2.622057554292119810464839589891119413682754951431623162816821703800791",
        ),
        (
            "gauss",
            "0.8346268416740731862814297327990468089939930134903470024498273701036820",
        ),
        (
            "levy",
            "3.275822918721811159787681882453843863608475525982374149405198924190723",
        ),
        (
            "khinchin",
            "2.685452001065306445309714835481795693820382293994462953051152345557219",
        ),
        (
            "erdosborwein",
            "1.606695152415291763783301523190924580480579671505756435778079553691418",
        ),
        (
            "cahen",
            "0.6434105462883380261822543077575647632865878602682395059870309203074928",
        ),
        (
            "recfib",
            "3.359885666243177553172011302918927179688905133731968486495553815325130",
        ),
        (
            "gompertz",
            "0.5963473623231940743410784993692793760741778601525487815734849104823272",
        ),
        (
            "soldner",
            "1.451369234883381050283968485892027449493032283648015863093004557662426",
        ),
        (
            "laplace",
            "0.6627434193491815809747420971092529070562335491150224175203925349909719",
        ),
        (
            "parabolic",
            "2.295587149392638074034298049189490387597832203638583483929975346644110",
        ),
        (
            "deg",
            "0.01745329251994329576923690768488612713442871888541725456097191440171009",
        ),
        (
            "niven",
            "1.705211140105367764288551453434508160762027651653469099994284906547313",
        ),
        (
            "twinprime",
            "0.6601618158468695739278121100145557784326233602847334133194484233354056",
        ),
        (
            "mertens",
            "0.2614972128476427837554268386086958590515666482611992061920642139249245",
        ),
    ];

    /// Correct decimal digits of `value` against `reference`.
    fn digits(value: &Float, reference: &Float) -> Float {
        let prec = reference.prec_64();
        let miss = Float::with_val_64(prec, value - reference).abs();
        if miss.is_zero() {
            return Float::with_val_64(prec, rug::float::Special::Infinity);
        }
        -(miss / reference.clone().abs()).log10()
    }

    #[test]
    fn every_constant_equals_its_independent_reference() {
        assert_eq!(TABLE.len(), REFERENCES.len());
        let prec = cnum::digits_to_bits(64);
        let wide = cnum::digits_to_bits(80);
        for named in TABLE {
            let (_, reference) = REFERENCES
                .iter()
                .find(|(name, _)| *name == named.name)
                .unwrap_or_else(|| panic!("no reference for {}", named.name));
            let reference = cnum::parse_float(reference, wide).unwrap();
            let value = named.value(prec).unwrap();
            assert_eq!(value.prec_64(), prec, "{}", named.name);
            assert!(digits(&value, &reference) >= 63, "{} = {value}", named.name);
        }
    }

    #[test]
    fn the_digits_do_not_depend_on_the_precision_asked_for() {
        // The value at 300 digits, cut to 100 and to 15 digits, is the value
        // computed at 100 and at 15: guard bits, series lengths and Newton
        // iterations follow the request.
        let wide = cnum::digits_to_bits(300);
        for named in TABLE {
            let exact = named.value(wide).unwrap();
            for narrow in [100, 15] {
                let prec = cnum::digits_to_bits(narrow);
                let value = named.value(prec).unwrap();
                let lost = Float::with_val_64(wide, &value - &exact).abs();
                let ulp = Float::with_val_64(wide, exact.clone().abs()) >> (prec as usize - 1);
                assert!(lost <= ulp, "{} at {narrow} digits", named.name);
            }
        }
    }

    #[test]
    fn closed_forms_satisfy_their_equations() {
        let prec = cnum::digits_to_bits(200);
        let value = |name: &str| named(name).unwrap().value(prec).unwrap();
        let near = |a: Float, b: Float| digits(&a, &b) >= 198;
        let (plastic, supergolden, tribonacci) =
            (value("plastic"), value("supergolden"), value("tribonacci"));
        assert!(near(power(&plastic, 3), plastic.clone() + 1u32));
        assert!(near(power(&supergolden, 3), power(&supergolden, 2) + 1u32));
        assert!(near(
            power(&tribonacci, 3),
            power(&tribonacci, 2) + &tribonacci + 1u32
        ));
        assert!(near(value("lemniscate"), value("pi") * value("gauss")));
        let omega = value("omega");
        assert!(near(omega.clone() * omega.exp(), one(prec)));
        let eta = value("eta");
        assert!(near(eta.pow(value("e")), value("e")));
        assert!(near(value("towermin").ln(), -value("e")));
    }

    #[test]
    fn names_and_aliases() {
        let prec = cnum::digits_to_bits(30);
        let get = |text: &str| named(text).unwrap().value(prec).unwrap();
        assert_eq!(get("pi"), get("π"));
        assert_eq!(get("PI"), get("pi"));
        assert_eq!(get("E"), get("e"));
        assert_eq!(get("towermax"), get("eta"));
        assert_eq!(get("Ω"), get("omega"));
        for text in ["", "p", "pie", "ω", "inf", "nan", "x", "i"] {
            assert!(named(text).is_none(), "{text:?}");
        }
        // No name is listed twice, and none reads as a number.
        let mut seen = std::collections::HashSet::new();
        for constant in TABLE {
            for text in std::iter::once(&constant.name).chain(constant.aliases) {
                assert!(seen.insert(*text), "{text} is listed twice");
                assert!(Float::parse(*text).is_err(), "{text} is a number");
                assert!(std::ptr::eq(named(text).unwrap(), constant), "{text}");
            }
        }
    }
}
