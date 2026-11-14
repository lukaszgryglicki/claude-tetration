//! Constant expressions for the textual inputs: `pi*e^e - pow(2, 0.5)`,
//! `ate(2+i, 1000-500i)`, `2^^2`.
//!
//! An input is a decimal number, read as before, or a complex expression of
//! numbers, the named constants of `constants.rs`, `i`, `+ - * / ^ ^^`,
//! brackets and functions, tetration and its inverses among them. The
//! grammar is the one of the plotters `tet-line` and
//! `rust-opengl-complex-drawer` without their variable: an input stands for
//! one number, and that number is computed to the precision the input is
//! read at.
//!
//! The expression is evaluated twice, node by node: once with every
//! operation rounded down, and once, with twenty more digits, with every
//! operation rounded up. An inexact operation therefore leaves different
//! numbers in the two evaluations, and a part that they share is exact. The
//! two values have to agree beyond the requested precision in each part;
//! where digits cancel, their number is read from the disagreement and
//! added to the working precision. A part that is zero in both evaluations
//! is accepted where it is exact by construction; a zero made of inexact
//! operands is a total cancellation, like the two signs of `sin(pi)`, and
//! after three of those in a row the input is refused. An expression of
//! real operations on real numbers stays exactly real.

use std::cmp::Ordering;
use std::sync::Mutex;

use rug::{
    float::Round,
    ops::{
        AddAssignRound, AssignRound, DivAssignRound, MulAssignRound, NegAssign, PowAssignRound,
        SubAssignRound,
    },
    Complex, Float,
};

use crate::constants::{self, Named};
use crate::{cnum, dispatch, inverse, lambertw};

type Round2 = (Round, Round);
type Ordering2 = (Ordering, Ordering);

/// One of the two evaluations: the bits and the rounding direction of every
/// operation, and the decimal digits asked of the tetrations in it.
#[derive(Clone, Copy)]
struct Level {
    prec: u64,
    digits: u64,
    round: Round,
}

impl Level {
    fn new(digits: u64, round: Round) -> Result<Level, String> {
        Ok(Level {
            prec: cnum::checked_digits_to_bits(digits)?,
            digits,
            round,
        })
    }

    fn both(self) -> Round2 {
        (self.round, self.round)
    }

    fn float<T>(self, value: T) -> Float
    where
        Float: AssignRound<T, Round = Round, Ordering = Ordering>,
    {
        Float::with_val_round_64(self.prec, value, self.round).0
    }

    fn real(self, x: Float) -> Complex {
        Complex::with_val_64(self.prec, (x, 0))
    }

    fn one(self) -> Complex {
        Complex::with_val_64(self.prec, 1)
    }
}

/// The value of an operation: real by construction, whole in both parts by
/// construction, or complex.
enum Out {
    Real(Float),
    Whole(Complex),
    Complex(Complex),
}

impl Out {
    fn complex(self, level: Level) -> Complex {
        match self {
            Out::Real(x) => level.real(x),
            Out::Whole(z) | Out::Complex(z) => z,
        }
    }

    /// The parts whose zero needs no exact operands: the imaginary part of
    /// a real value, and both parts of a whole one.
    fn free(&self) -> [bool; 2] {
        match self {
            Out::Real(_) => [false, true],
            Out::Whole(_) => [true, true],
            Out::Complex(_) => [false, false],
        }
    }
}

type Apply = fn(&[Complex], Level) -> Result<Out, String>;
type RealOp = fn(&mut Float, Round) -> Ordering;
type ComplexOp = fn(&mut Complex, Round2) -> Ordering2;

/// One function of the expressions.
pub struct Function {
    pub names: &'static [&'static str],
    /// The numbers of arguments it takes.
    pub arguments: &'static [usize],
    pub meaning: &'static str,
    apply: Apply,
}

macro_rules! function {
    ($names:expr, $arguments:expr, $meaning:expr, $apply:expr $(,)?) => {
        Function {
            names: &$names,
            arguments: &$arguments,
            meaning: $meaning,
            apply: $apply,
        }
    };
}

fn is_real(z: &Complex) -> bool {
    z.imag().is_zero()
}

fn everywhere(_: &Float) -> bool {
    true
}

fn positive(x: &Float) -> bool {
    *x > 0
}

fn not_negative(x: &Float) -> bool {
    *x >= 0
}

fn within_one(x: &Float) -> bool {
    *x >= -1 && *x <= 1
}

fn from_one(x: &Float) -> bool {
    *x >= 1
}

/// A function that is real on the real numbers of `domain`, and its complex
/// continuation elsewhere.
fn analytic(
    z: &[Complex],
    level: Level,
    domain: fn(&Float) -> bool,
    on_reals: RealOp,
    elsewhere: ComplexOp,
) -> Result<Out, String> {
    if is_real(&z[0]) && domain(z[0].real()) {
        return of_real(z, level, on_reals);
    }
    let mut value = z[0].clone();
    elsewhere(&mut value, level.both());
    Ok(Out::Complex(value))
}

/// MPC reads the side of a branch cut from the sign of a zero part, and the
/// zeros here are positive. A number on a cut takes the side that continues
/// the function counter-clockwise around the branch point, which is the
/// side of the positive zero except on two cuts: the one of asin, acos and
/// atanh above 1 ...
fn lower_side(z: &mut Complex) {
    if z.imag().is_zero() && *z.real() > 1 {
        z.mut_imag().neg_assign();
    }
}

/// ... and the one of asinh and atan below -i.
fn left_side(z: &mut Complex) {
    if z.real().is_zero() && *z.imag() < -1 {
        z.mut_real().neg_assign();
    }
}

fn reals(z: &[Complex]) -> Result<Vec<&Float>, String> {
    if z.iter().all(is_real) {
        Ok(z.iter().map(Complex::real).collect())
    } else {
        Err("takes real arguments only".into())
    }
}

fn of_real(z: &[Complex], level: Level, operation: RealOp) -> Result<Out, String> {
    let mut value = reals(z)?[0].clone();
    operation(&mut value, level.round);
    Ok(Out::Real(value))
}

fn of_reals(
    z: &[Complex],
    level: Level,
    operation: fn(&mut Float, &Float, Round) -> Ordering,
) -> Result<Out, String> {
    let x = reals(z)?;
    let mut value = x[0].clone();
    operation(&mut value, x[1], level.round);
    Ok(Out::Real(value))
}

fn real_second(z: &[Complex]) -> Result<(), String> {
    if is_real(&z[1]) {
        Ok(())
    } else {
        Err("takes a real second argument".into())
    }
}

fn add(a: &Complex, b: &Complex, level: Level) -> Complex {
    let mut value = a.clone();
    value.add_assign_round(b, level.both());
    value
}

fn subtract(a: &Complex, b: &Complex, level: Level) -> Complex {
    let mut value = a.clone();
    value.sub_assign_round(b, level.both());
    value
}

fn multiply(a: &Complex, b: &Complex, level: Level) -> Complex {
    let mut value = a.clone();
    value.mul_assign_round(b, level.both());
    value
}

fn divide(a: &Complex, b: &Complex, level: Level) -> Result<Complex, String> {
    if cnum::is_zero(b) {
        return Err("division by zero".into());
    }
    let mut value = a.clone();
    value.div_assign_round(b, level.both());
    Ok(value)
}

fn power(base: &Complex, exponent: &Complex, level: Level) -> Out {
    if is_real(base) && is_real(exponent) {
        let (x, y) = (base.real(), exponent.real());
        // Real where the real power is: no negative base, or a whole exponent.
        if not_negative(x) || y.is_integer() {
            let mut value = x.clone();
            value.pow_assign_round(y, level.round);
            return Out::Real(value);
        }
    }
    let mut value = base.clone();
    value.pow_assign_round(exponent, level.both());
    Out::Complex(value)
}

fn logarithm(z: &Complex, level: Level) -> Out {
    if is_real(z) && positive(z.real()) {
        let mut value = z.real().clone();
        value.ln_round(level.round);
        return Out::Real(value);
    }
    let mut value = z.clone();
    value.ln_round(level.both());
    Out::Complex(value)
}

fn modulus(z: &Complex, level: Level) -> Float {
    level.float(z.abs_ref())
}

fn argument(z: &Complex, level: Level) -> Result<Float, String> {
    if cnum::is_zero(z) {
        return Err("the argument of zero is not defined".into());
    }
    Ok(level.float(z.arg_ref()))
}

/// `cos t + i sin t`.
fn turn(angle: &Float, level: Level) -> Complex {
    let (mut re, mut im) = (angle.clone(), angle.clone());
    re.cos_round(level.round);
    im.sin_round(level.round);
    Complex::with_val_64(level.prec, (re, im))
}

fn cube_root(z: &[Complex], level: Level) -> Result<Out, String> {
    if is_real(&z[0]) && not_negative(z[0].real()) {
        return of_real(z, level, Float::cbrt_round);
    }
    let mut third = Float::with_val_64(level.prec, 1);
    third.div_assign_round(3u32, level.round);
    Ok(power(&z[0], &level.real(third), level))
}

fn root(z: &[Complex], level: Level) -> Result<Out, String> {
    let inverse = tidy(divide(&level.one(), &z[1], level)?);
    Ok(power(&z[0], &inverse, level))
}

fn whole_power(z: &[Complex], level: Level, signed: bool) -> Result<Out, String> {
    let whole = is_real(&z[1]) && z[1].real().is_integer();
    if !whole {
        return Err("takes a whole exponent".into());
    }
    if !signed && z[1].real().is_sign_negative() {
        return Err("takes an exponent that is not negative".into());
    }
    Ok(power(&z[0], &z[1], level))
}

fn log(z: &[Complex], level: Level) -> Result<Out, String> {
    let value = logarithm(&z[0], level);
    let Some(base) = z.get(1) else {
        return Ok(value);
    };
    let base = logarithm(base, level).complex(level);
    Ok(Out::Complex(divide(&value.complex(level), &base, level)?))
}

fn sign(z: &[Complex], level: Level) -> Result<Out, String> {
    let mut value = z[0].clone();
    if !cnum::is_zero(&value) {
        value.div_assign_round(&modulus(&z[0], level), level.both());
    }
    Ok(Out::Complex(value))
}

fn by_parts(z: &[Complex], level: Level, operation: fn(Float) -> Float) -> Complex {
    let (re, im) = z[0].clone().into_real_imag();
    Complex::with_val_64(level.prec, (operation(re), operation(im)))
}

fn lambert(z: &[Complex], level: Level) -> Result<Out, String> {
    let w = lambertw::w0(&z[0], level.prec)?;
    // Real from -1/e on.
    let edge = -Float::with_val_64(level.prec, -1).exp();
    Ok(if is_real(&z[0]) && *z[0].real() >= edge {
        Out::Real(w.real().clone())
    } else {
        Out::Complex(w)
    })
}

/// Every function, in the order of the documentation.
pub const FUNCTIONS: &[Function] = &[
    function!(["tet"], [2], "tet(b, h) = b^^h, tetration", |z, level| {
        dispatch::tetrate(&z[0], &z[1], level.prec, level.digits).map(Out::Complex)
    }),
    function!(
        ["slog", "ate"],
        [2],
        "slog(b, z): the height h with b^^h = z",
        |z, level| inverse::slog(&z[0], &z[1], level.prec, level.digits).map(Out::Complex)
    ),
    function!(
        ["ssrt", "sroot"],
        [2],
        "ssrt(z, h): the base b with b^^h = z",
        |z, level| inverse::sroot(&z[0], &z[1], level.prec, level.digits).map(Out::Complex)
    ),
    function!(["sqrt"], [1], "square root", |z, level| {
        analytic(
            z,
            level,
            not_negative,
            Float::sqrt_round,
            Complex::sqrt_round,
        )
    }),
    function!(["cbrt"], [1], "cube root, principal value", cube_root),
    function!(["root"], [2], "root(z, n) = z^(1/n), principal value", root),
    function!(["sqr", "square"], [1], "square", |z, level| {
        Ok(Out::Complex(multiply(&z[0], &z[0], level)))
    }),
    function!(["pow", "powc"], [2], "pow(z, w) = z^w", |z, level| {
        Ok(power(&z[0], &z[1], level))
    }),
    function!(["powf"], [2], "powf(z, x) = z^x, real x", |z, level| {
        real_second(z)?;
        Ok(power(&z[0], &z[1], level))
    }),
    function!(["powi"], [2], "powi(z, n) = z^n, whole n", |z, level| {
        whole_power(z, level, true)
    }),
    function!(
        ["powu"],
        [2],
        "powu(z, n) = z^n, whole n that is not negative",
        |z, level| whole_power(z, level, false)
    ),
    function!(["exp"], [1], "exponential", |z, level| {
        analytic(z, level, everywhere, Float::exp_round, Complex::exp_round)
    }),
    function!(["exp2"], [1], "exp2(z) = 2^z", |z, level| {
        analytic(z, level, everywhere, Float::exp2_round, Complex::exp2_round)
    }),
    function!(["expf"], [2], "expf(z, x) = x^z, real x", |z, level| {
        real_second(z)?;
        Ok(power(&z[1], &z[0], level))
    }),
    function!(["ln"], [1], "natural logarithm", |z, level| {
        Ok(logarithm(&z[0], level))
    }),
    function!(
        ["log"],
        [1, 2],
        "log(z) = ln z; log(z, b) = ln z / ln b",
        log
    ),
    function!(["log2"], [1], "logarithm to the base 2", |z, level| {
        analytic(z, level, positive, Float::log2_round, |z, round| {
            z.ln_round(round);
            let two = Float::with_val_64(z.prec_64().0, 2);
            let mut scale = two;
            scale.ln_round(round.0);
            z.div_assign_round(&scale, round)
        })
    }),
    function!(["log10"], [1], "logarithm to the base 10", |z, level| {
        analytic(z, level, positive, Float::log10_round, Complex::log10_round)
    }),
    function!(["sin"], [1], "sine", |z, level| {
        analytic(z, level, everywhere, Float::sin_round, Complex::sin_round)
    }),
    function!(["cos"], [1], "cosine", |z, level| {
        analytic(z, level, everywhere, Float::cos_round, Complex::cos_round)
    }),
    function!(["tan"], [1], "tangent", |z, level| {
        analytic(z, level, everywhere, Float::tan_round, Complex::tan_round)
    }),
    function!(["cot"], [1], "cotangent", |z, level| {
        analytic(z, level, everywhere, Float::cot_round, |z, round| {
            z.tan_round(round);
            z.recip_round(round)
        })
    }),
    function!(["sec"], [1], "secant", |z, level| {
        analytic(z, level, everywhere, Float::sec_round, |z, round| {
            z.cos_round(round);
            z.recip_round(round)
        })
    }),
    function!(["csc"], [1], "cosecant", |z, level| {
        analytic(z, level, everywhere, Float::csc_round, |z, round| {
            z.sin_round(round);
            z.recip_round(round)
        })
    }),
    function!(["asin", "arcsin"], [1], "inverse sine", |z, level| {
        analytic(z, level, within_one, Float::asin_round, |z, round| {
            lower_side(z);
            z.asin_round(round)
        })
    }),
    function!(["acos", "arccos"], [1], "inverse cosine", |z, level| {
        analytic(z, level, within_one, Float::acos_round, |z, round| {
            lower_side(z);
            z.acos_round(round)
        })
    }),
    function!(["atan", "arctan"], [1], "inverse tangent", |z, level| {
        analytic(z, level, everywhere, Float::atan_round, |z, round| {
            left_side(z);
            z.atan_round(round)
        })
    }),
    function!(
        ["atan2"],
        [2],
        "atan2(y, x): the argument of x + iy, real x and y",
        |z, level| of_reals(z, level, Float::atan2_round)
    ),
    function!(["sinh"], [1], "hyperbolic sine", |z, level| {
        analytic(z, level, everywhere, Float::sinh_round, Complex::sinh_round)
    }),
    function!(["cosh"], [1], "hyperbolic cosine", |z, level| {
        analytic(z, level, everywhere, Float::cosh_round, Complex::cosh_round)
    }),
    function!(["tanh"], [1], "hyperbolic tangent", |z, level| {
        analytic(z, level, everywhere, Float::tanh_round, Complex::tanh_round)
    }),
    function!(["coth"], [1], "hyperbolic cotangent", |z, level| {
        analytic(z, level, everywhere, Float::coth_round, |z, round| {
            z.tanh_round(round);
            z.recip_round(round)
        })
    }),
    function!(["sech"], [1], "hyperbolic secant", |z, level| {
        analytic(z, level, everywhere, Float::sech_round, |z, round| {
            z.cosh_round(round);
            z.recip_round(round)
        })
    }),
    function!(["csch"], [1], "hyperbolic cosecant", |z, level| {
        analytic(z, level, everywhere, Float::csch_round, |z, round| {
            z.sinh_round(round);
            z.recip_round(round)
        })
    }),
    function!(
        ["asinh", "arcsinh"],
        [1],
        "inverse hyperbolic sine",
        |z, level| {
            analytic(z, level, everywhere, Float::asinh_round, |z, round| {
                left_side(z);
                z.asinh_round(round)
            })
        }
    ),
    function!(
        ["acosh", "arccosh"],
        [1],
        "inverse hyperbolic cosine",
        |z, level| { analytic(z, level, from_one, Float::acosh_round, Complex::acosh_round,) }
    ),
    function!(
        ["atanh", "arctanh"],
        [1],
        "inverse hyperbolic tangent",
        |z, level| {
            analytic(z, level, within_one, Float::atanh_round, |z, round| {
                lower_side(z);
                z.atanh_round(round)
            })
        }
    ),
    function!(
        ["abs", "mag", "mod", "norm", "polar_r", "radius"],
        [1],
        "absolute value",
        |z, level| Ok(Out::Real(modulus(&z[0], level)))
    ),
    function!(
        ["abs2", "mag2", "norm_sqr", "normsq"],
        [1],
        "square of the absolute value",
        |z, level| Ok(Out::Real(level.float(z[0].norm_ref())))
    ),
    function!(
        ["l1_norm", "l1", "manhattan", "taxicab"],
        [1],
        "|re z| + |im z|",
        |z, level| {
            let mut sum = z[0].real().clone().abs();
            sum.add_assign_round(&z[0].imag().clone().abs(), level.round);
            Ok(Out::Real(sum))
        }
    ),
    function!(
        ["arg", "phase", "polar_theta", "theta"],
        [1],
        "argument, in (-pi, pi]",
        |z, level| argument(&z[0], level).map(Out::Real)
    ),
    function!(["to_polar"], [1], "abs(z) + i arg(z)", |z, level| {
        let parts = (modulus(&z[0], level), argument(&z[0], level)?);
        Ok(Out::Complex(Complex::with_val_64(level.prec, parts)))
    }),
    function!(["re", "real"], [1], "real part", |z, _| {
        Ok(Out::Real(z[0].real().clone()))
    }),
    function!(["im", "imag"], [1], "imaginary part", |z, _| {
        Ok(Out::Real(z[0].imag().clone()))
    }),
    function!(["conj", "conjugate"], [1], "complex conjugate", |z, _| {
        Ok(Out::Complex(z[0].clone().conj()))
    }),
    function!(
        ["inv", "recip", "inverse", "finv"],
        [1],
        "1/z",
        |z, level| divide(&level.one(), &z[0], level).map(Out::Complex)
    ),
    function!(
        ["sgn", "sign", "signum"],
        [1],
        "z/abs(z), and 0 for 0",
        sign
    ),
    function!(
        ["cis"],
        [1],
        "cis(x) = cos x + i sin x, real x",
        |z, level| { Ok(Out::Complex(turn(reals(z)?[0], level))) }
    ),
    function!(["scale"], [2], "scale(z, x) = z x, real x", |z, level| {
        real_second(z)?;
        Ok(Out::Complex(multiply(&z[0], &z[1], level)))
    }),
    function!(
        ["unscale"],
        [2],
        "unscale(z, x) = z/x, real x",
        |z, level| {
            real_second(z)?;
            divide(&z[0], &z[1], level).map(Out::Complex)
        }
    ),
    function!(["fdiv"], [2], "fdiv(z, w) = z/w", |z, level| {
        divide(&z[0], &z[1], level).map(Out::Complex)
    }),
    function!(
        ["complex", "rect", "new"],
        [2],
        "complex(x, y) = x + iy, real x and y",
        |z, level| {
            let x = reals(z)?;
            let parts = (x[0].clone(), x[1].clone());
            Ok(Out::Complex(Complex::with_val_64(level.prec, parts)))
        }
    ),
    function!(
        ["polar", "from_polar"],
        [2],
        "polar(r, t) = r (cos t + i sin t), real r and t",
        |z, level| {
            let x = reals(z)?;
            let mut value = turn(x[1], level);
            value.mul_assign_round(x[0], level.both());
            Ok(Out::Complex(value))
        }
    ),
    function!(
        ["floor"],
        [1],
        "largest whole number not above, of each part",
        |z, level| Ok(Out::Whole(by_parts(z, level, Float::floor)))
    ),
    function!(
        ["ceil"],
        [1],
        "smallest whole number not below, of each part",
        |z, level| Ok(Out::Whole(by_parts(z, level, Float::ceil)))
    ),
    function!(
        ["round"],
        [1],
        "nearest whole number, halves away from zero, of each part",
        |z, level| Ok(Out::Whole(by_parts(z, level, Float::round)))
    ),
    function!(
        ["trunc"],
        [1],
        "whole number towards zero, of each part",
        |z, level| Ok(Out::Whole(by_parts(z, level, Float::trunc)))
    ),
    function!(
        ["frac", "fract"],
        [1],
        "what trunc leaves, of each part",
        |z, level| Ok(Out::Complex(by_parts(z, level, Float::fract)))
    ),
    function!(
        ["minabs"],
        [2],
        "the argument of the smaller absolute value",
        |z, _| {
            let first = z[0].cmp_abs(&z[1]) != Some(Ordering::Greater);
            Ok(Out::Complex(z[usize::from(!first)].clone()))
        }
    ),
    function!(
        ["maxabs"],
        [2],
        "the argument of the larger absolute value",
        |z, _| {
            let first = z[0].cmp_abs(&z[1]) != Some(Ordering::Less);
            Ok(Out::Complex(z[usize::from(!first)].clone()))
        }
    ),
    function!(["avg"], [2], "avg(z, w) = (z + w)/2", |z, level| {
        let mut mean = add(&z[0], &z[1], level);
        mean /= 2u32;
        Ok(Out::Complex(mean))
    }),
    function!(["min"], [2], "the smaller of two real numbers", |z, _| {
        let x = reals(z)?;
        Ok(Out::Real(x[0].clone().min(x[1])))
    }),
    function!(["max"], [2], "the larger of two real numbers", |z, _| {
        let x = reals(z)?;
        Ok(Out::Real(x[0].clone().max(x[1])))
    }),
    function!(
        ["hypot"],
        [2],
        "hypot(x, y) = sqrt(x^2 + y^2), real x and y",
        |z, level| of_reals(z, level, Float::hypot_round)
    ),
    function!(
        ["agm"],
        [2],
        "arithmetic-geometric mean of two real numbers",
        |z, level| of_reals(z, level, Float::agm_round)
    ),
    function!(
        ["gamma", "tgamma"],
        [1],
        "gamma function of a real number (without brackets, gamma is Euler's constant)",
        |z, level| of_real(z, level, Float::gamma_round)
    ),
    function!(
        ["lgamma"],
        [1],
        "logarithm of |gamma(x)|, real x",
        |z, level| of_real(z, level, |x, round| x.ln_abs_gamma_round(round).1)
    ),
    function!(
        ["fact", "factorial"],
        [1],
        "fact(x) = gamma(x + 1), real x",
        |z, level| {
            of_real(z, level, |x, round| {
                x.add_assign_round(1u32, round);
                x.gamma_round(round)
            })
        }
    ),
    function!(
        ["digamma"],
        [1],
        "logarithmic derivative of the gamma function, real x",
        |z, level| of_real(z, level, Float::digamma_round)
    ),
    function!(
        ["zeta"],
        [1],
        "Riemann zeta function of a real number",
        |z, level| of_real(z, level, Float::zeta_round)
    ),
    function!(
        ["erf"],
        [1],
        "error function of a real number",
        |z, level| { of_real(z, level, Float::erf_round) }
    ),
    function!(
        ["erfc"],
        [1],
        "complementary error function of a real number",
        |z, level| of_real(z, level, Float::erfc_round)
    ),
    function!(
        ["ei"],
        [1],
        "exponential integral Ei of a real number",
        |z, level| of_real(z, level, Float::eint_round)
    ),
    function!(["li2"], [1], "dilogarithm of a real number", |z, level| {
        of_real(z, level, Float::li2_round)
    }),
    function!(
        ["w", "lambertw"],
        [1],
        "Lambert W, principal branch",
        lambert
    ),
];

/// The list of the functions for the usage text.
pub fn help() -> String {
    let mut text = String::new();
    for function in FUNCTIONS {
        text.push_str(&format!(
            "    {:<36} {}\n",
            function.names.join(", "),
            function.meaning
        ));
    }
    text
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Number(String),
    Name(String),
    Symbol(char),
}

/// The operator `^^` as one symbol.
const TETRATION: char = '⇑';

fn tokens(text: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = text.chars().collect();
    let digit = |at: usize| chars.get(at).is_some_and(char::is_ascii_digit);
    let mut found = Vec::new();
    let mut at = 0;
    while at < chars.len() {
        let c = chars[at];
        if c.is_whitespace() {
            at += 1;
        } else if c.is_ascii_digit() || (c == '.' && digit(at + 1)) {
            let start = at;
            while digit(at) {
                at += 1;
            }
            if chars.get(at) == Some(&'.') {
                at += 1;
                while digit(at) {
                    at += 1;
                }
            }
            // An exponent needs its digits: `2e` is 2 times the constant e.
            if matches!(chars.get(at), Some('e' | 'E')) {
                let sign = usize::from(matches!(chars.get(at + 1), Some('+' | '-')));
                if digit(at + 1 + sign) {
                    at += 1 + sign;
                    while digit(at) {
                        at += 1;
                    }
                }
            }
            found.push(Token::Number(chars[start..at].iter().collect()));
        } else if c.is_alphabetic() || c == '_' || c == '√' {
            let start = at;
            at += 1;
            while chars
                .get(at)
                .is_some_and(|c| c.is_alphanumeric() || *c == '_')
            {
                at += 1;
            }
            found.push(Token::Name(chars[start..at].iter().collect()));
        } else {
            let pair = chars.get(at + 1) == Some(&c);
            let symbol = match c {
                '+' | '-' | '/' | '(' | ')' | ',' => c,
                '−' => '-',
                '×' | '·' => '*',
                '÷' => '/',
                '^' if pair => TETRATION,
                '^' => '^',
                '*' if pair => '^',
                '*' => '*',
                other => return Err(format!("unexpected character {other:?}")),
            };
            at += 1 + usize::from(pair && matches!(c, '^' | '*'));
            found.push(Token::Symbol(symbol));
        }
    }
    Ok(found)
}

enum Node {
    Number(String),
    Constant(&'static Named),
    Imaginary,
    Negate(Box<Node>),
    Binary(char, Box<Node>, Box<Node>),
    Call(&'static Function, Vec<Node>),
}

struct Parser {
    tokens: Vec<Token>,
    at: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.at)
    }

    fn take(&mut self, symbol: char) -> bool {
        let found = self.peek() == Some(&Token::Symbol(symbol));
        self.at += usize::from(found);
        found
    }

    /// The operators from the strength `least` on, as in the plotters:
    /// `+ -` (1), then `* /` and the product without a sign of `2pi` and
    /// `3(1+i)` (2), then a leading sign (3), then `^` (4), then `^^` (5).
    /// The last two group from the right: `-2^2 = -4`, `2^-2 = 1/4`,
    /// `2^^3^2 = (2^^3)^2`.
    fn expression(&mut self, least: u8) -> Result<Node, String> {
        let mut left = if self.take('-') {
            Node::Negate(Box::new(self.expression(3)?))
        } else if self.take('+') {
            self.expression(3)?
        } else {
            self.atom()?
        };
        loop {
            let after_number = matches!(self.tokens[self.at - 1], Token::Number(_));
            let (operator, strength, written) = match self.peek() {
                Some(Token::Symbol(symbol @ ('+' | '-'))) => (*symbol, 1, true),
                Some(Token::Symbol(symbol @ ('*' | '/'))) => (*symbol, 2, true),
                Some(Token::Symbol('^')) => ('^', 4, true),
                Some(Token::Symbol(TETRATION)) => (TETRATION, 5, true),
                Some(Token::Name(_) | Token::Symbol('(')) => ('*', 2, false),
                // Two numbers in a row are a slip, not a product.
                Some(Token::Number(_)) if !after_number => ('*', 2, false),
                _ => return Ok(left),
            };
            if strength < least {
                return Ok(left);
            }
            self.at += usize::from(written);
            let right = self.expression(strength + u8::from(strength < 4))?;
            left = Node::Binary(operator, Box::new(left), Box::new(right));
        }
    }

    fn atom(&mut self) -> Result<Node, String> {
        let token = self.peek().cloned().ok_or("the expression ends early")?;
        self.at += 1;
        match token {
            Token::Number(text) => Ok(Node::Number(text)),
            Token::Symbol('(') => {
                let inner = self.expression(0)?;
                if !self.take(')') {
                    return Err("a closing bracket is missing".into());
                }
                Ok(inner)
            }
            Token::Name(name) => self.named(&name),
            Token::Symbol(TETRATION) => Err("unexpected ^^".into()),
            Token::Symbol(symbol) => Err(format!("unexpected {symbol:?}")),
        }
    }

    /// A function with its arguments, the imaginary unit or a constant.
    fn named(&mut self, name: &str) -> Result<Node, String> {
        let lower = name.to_ascii_lowercase();
        let function = FUNCTIONS
            .iter()
            .find(|function| function.names.contains(&lower.as_str()));
        let called = self.peek() == Some(&Token::Symbol('('));
        if let (Some(function), true) = (function, called) {
            self.at += 1;
            let mut arguments = vec![self.expression(0)?];
            while self.take(',') {
                arguments.push(self.expression(0)?);
            }
            if !self.take(')') {
                return Err(format!("a closing bracket is missing after {name}("));
            }
            if !function.arguments.contains(&arguments.len()) {
                return Err(format!(
                    "{name} does not take {} arguments",
                    arguments.len()
                ));
            }
            return Ok(Node::Call(function, arguments));
        }
        if lower == "i" || lower == "j" {
            return Ok(Node::Imaginary);
        }
        if let Some(named) = constants::named(name) {
            return Ok(Node::Constant(named));
        }
        Err(match (function, called) {
            (Some(_), _) => format!("{name} needs its arguments in brackets"),
            (None, true) => format!("unknown function {name}"),
            (None, false) => format!(
                "unknown name {name} (an input has no variables; --help lists the constants)"
            ),
        })
    }
}

fn parse(text: &str) -> Result<Node, String> {
    let mut parser = Parser {
        tokens: tokens(text)?,
        at: 0,
    };
    let node = parser.expression(0)?;
    match parser.peek() {
        None => Ok(node),
        Some(Token::Symbol(symbol)) => Err(format!("unexpected {symbol:?}")),
        Some(Token::Number(text) | Token::Name(text)) => {
            Err(format!("an operator is missing before {text}"))
        }
    }
}

/// How an evaluation ends without a value.
enum Failure {
    /// The two evaluations have no digit in common at this precision; the
    /// reason is there when only one of them could be made.
    Cancelled(Option<String>),
    Error(String),
}

/// Both evaluations of a node: rounded down, and rounded up with more digits.
type Pair = [Complex; 2];

/// Both evaluations of the real and of the imaginary part.
fn parts(value: &Pair) -> [[&Float; 2]; 2] {
    [
        [value[0].real(), value[1].real()],
        [value[0].imag(), value[1].imag()],
    ]
}

/// An inexact operation is rounded down in one evaluation and up in the
/// other: a part that they share is exact.
fn exact(part: [&Float; 2]) -> bool {
    part[0] == part[1]
}

fn zero(part: [&Float; 2]) -> bool {
    part[0].is_zero() && part[1].is_zero()
}

/// A product that is known exactly: a zero factor, or two exact factors.
fn known(a: [&Float; 2], b: [&Float; 2]) -> bool {
    zero(a) || zero(b) || (exact(a) && exact(b))
}

/// An operand that is exact or lies on an axis.
fn plain(value: &Pair) -> bool {
    let [re, im] = parts(value);
    zero(re) || zero(im) || (exact(re) && exact(im))
}

/// The zeros of a value made positive: a zero part lies on the axis, not on
/// a side of it.
fn tidy(mut z: Complex) -> Complex {
    let (re, im) = z.as_mut_real_imag();
    for part in [re, im] {
        if part.is_zero() {
            part.abs_mut();
        }
    }
    z
}

/// One operation in both evaluations, and the parts whose zero is exact
/// by construction. A value that only one of them has is a cancellation.
fn both(
    levels: &[Level; 2],
    operation: impl Fn(usize, Level) -> Result<Out, String>,
) -> Result<(Pair, [bool; 2]), Failure> {
    let run = |at: usize| {
        let level = levels[at];
        let out = operation(at, level)?;
        let free = out.free();
        let value = Complex::with_val_round_64(level.prec, out.complex(level), level.both()).0;
        if value.real().is_nan() || value.imag().is_nan() {
            return Err("the value is not defined".to_string());
        }
        if !cnum::is_finite(&value) {
            return Err("the value is infinite or exceeds MPFR's exponent range".into());
        }
        Ok((tidy(value), free))
    };
    match (run(0), run(1)) {
        (Ok((low, free)), Ok((high, also))) => {
            Ok(([low, high], [free[0] && also[0], free[1] && also[1]]))
        }
        (Err(reason), Err(_)) => Err(Failure::Error(reason)),
        (Err(reason), Ok(_)) | (Ok(_), Err(reason)) => Err(Failure::Cancelled(Some(reason))),
    }
}

/// A zero part is kept where `allowed` says that it is exact by
/// construction; made of inexact operands it is a cancellation.
fn settled_zeros(value: Pair, allowed: [bool; 2]) -> Result<Pair, Failure> {
    let [re, im] = parts(&value);
    if (zero(re) && !allowed[0]) || (zero(im) && !allowed[1]) {
        return Err(Failure::Cancelled(None));
    }
    Ok(value)
}

fn number(text: &str, level: Level) -> Result<Float, String> {
    // The checks of the decimal reader, then the number in the direction of
    // this evaluation.
    cnum::parse_float(text, level.prec)?;
    let parsed = Float::parse(text).map_err(|error| format!("invalid number {text:?}: {error}"))?;
    Ok(level.float(parsed))
}

fn binary(operator: char, a: &Complex, b: &Complex, level: Level) -> Result<Out, String> {
    Ok(match operator {
        '+' => Out::Complex(add(a, b, level)),
        '-' => Out::Complex(subtract(a, b, level)),
        '*' => Out::Complex(multiply(a, b, level)),
        '/' => Out::Complex(divide(a, b, level)?),
        '^' => power(a, b, level),
        _ => Out::Complex(
            dispatch::tetrate(a, b, level.prec, level.digits)
                .map_err(|reason| format!("^^: {reason}"))?,
        ),
    })
}

impl Node {
    /// Both evaluations of the node.
    fn value(&self, levels: &[Level; 2]) -> Result<Pair, Failure> {
        match self {
            Node::Number(text) => {
                both(levels, |_, level| number(text, level).map(Out::Real)).map(|(value, _)| value)
            }
            Node::Constant(named) => both(levels, |_, level| {
                // Correct to 64 bits more, then rounded in the direction of
                // this evaluation.
                let guarded = level
                    .prec
                    .checked_add(64)
                    .ok_or("precision count overflow")?;
                Ok(Out::Real(level.float(named.value(guarded)?)))
            })
            .map(|(value, _)| value),
            Node::Imaginary => Ok(levels.map(|level| Complex::with_val_64(level.prec, (0, 1)))),
            Node::Negate(inner) => Ok(inner.value(levels)?.map(|value| tidy(-value))),
            Node::Binary(operator, left, right) => {
                let (a, b) = (left.value(levels)?, right.value(levels)?);
                let (value, _) =
                    both(levels, |at, level| binary(*operator, &a[at], &b[at], level))?;
                let ([p, q], [r, s]) = (parts(&a), parts(&b));
                let allowed = match operator {
                    '+' | '-' => [exact(p) && exact(r), exact(q) && exact(s)],
                    '*' | '/' => [known(p, r) && known(q, s), known(p, s) && known(q, r)],
                    _ => [plain(&a) && plain(&b); 2],
                };
                settled_zeros(value, allowed)
            }
            Node::Call(function, arguments) => {
                let arguments = arguments
                    .iter()
                    .map(|argument| argument.value(levels))
                    .collect::<Result<Vec<Pair>, Failure>>()?;
                let (value, free) = both(levels, |at, level| {
                    let operands: Vec<Complex> =
                        arguments.iter().map(|pair| pair[at].clone()).collect();
                    (function.apply)(&operands, level)
                })
                .map_err(|failure| {
                    let named = |reason| format!("{}: {reason}", function.names[0]);
                    match failure {
                        Failure::Cancelled(reason) => Failure::Cancelled(reason.map(named)),
                        Failure::Error(reason) => Failure::Error(named(reason)),
                    }
                })?;
                let plain = arguments.iter().all(plain);
                settled_zeros(value, [plain || free[0], plain || free[1]])
            }
        }
    }

    /// The value for `prec` bits, with the bits of the second evaluation.
    /// The two evaluations must agree beyond `prec` bits in each part.
    /// Where they do not, the bits on which they differ were lost to
    /// cancellation, and the next pair works with that many more. A part
    /// that cancels to nothing three times in a row, the digits more than
    /// doubling each time, is an error.
    fn settled(&self, prec: u64) -> Result<Complex, String> {
        // 30103/100000 is below log10(2); with the 2 the count is an upper one.
        let decimal = |bits: u64| bits.saturating_mul(30_103) / 100_000 + 2;
        let wanted = prec.checked_add(16).ok_or("precision count overflow")?;
        let ceiling = decimal(wanted).saturating_mul(64);
        let mut digits = decimal(wanted);
        let mut cancelled = 0;
        let mut why = None;
        while digits <= ceiling && cancelled < 3 {
            let high = digits.checked_add(20).ok_or("precision count overflow")?;
            let levels = [
                Level::new(digits, Round::Down)?,
                Level::new(high, Round::Up)?,
            ];
            let all = levels[0].prec;
            let missing = match self.value(&levels) {
                Ok([low, high]) => {
                    let missing = lost(low.real(), high.real(), wanted, all).max(lost(
                        low.imag(),
                        high.imag(),
                        wanted,
                        all,
                    ));
                    if missing == 0 {
                        return Ok(high);
                    }
                    why = None;
                    missing
                }
                Err(Failure::Cancelled(reason)) => {
                    why = reason;
                    all
                }
                Err(Failure::Error(reason)) => return Err(reason),
            };
            cancelled = if missing == all { cancelled + 1 } else { 0 };
            digits = digits.saturating_add(decimal(missing)).saturating_add(20);
        }
        Err(match why {
            Some(reason) => {
                format!("{reason} in one of the two evaluations at every precision tried")
            }
            None => "the value cancels below every precision tried (if it is zero, write 0)".into(),
        })
    }
}

/// The bits that a part lost: none where the two evaluations agree on
/// `wanted` bits, all of them where they share no digit.
fn lost(low: &Float, high: &Float, wanted: u64, all: u64) -> u64 {
    if low == high {
        return 0;
    }
    if low.is_zero() || high.is_zero() || low.is_sign_negative() != high.is_sign_negative() {
        return all;
    }
    let gap = Float::with_val_64(all, low - high);
    let agreed = match (high.get_exp(), gap.get_exp()) {
        (Some(size), Some(miss)) => u64::try_from(i64::from(size) - i64::from(miss)).unwrap_or(0),
        _ => 0,
    };
    if agreed >= wanted {
        0
    } else {
        all.saturating_sub(agreed).max(1)
    }
}

/// The expressions settled last. The entry points read their inputs again
/// when an algorithm raises the precision; a tetration inside an input is
/// then not computed twice for the same bits.
static SETTLED: Mutex<Vec<(String, u64, Complex)>> = Mutex::new(Vec::new());

/// The settled value of an expression for `prec` bits, not yet rounded to
/// them.
fn evaluated(text: &str, prec: u64) -> Result<Complex, String> {
    let settled = |cache: &Vec<(String, u64, Complex)>| {
        let found = cache
            .iter()
            .find(|(of, bits, _)| of == text && *bits == prec);
        found.map(|(.., value)| value.clone())
    };
    if let Some(value) = SETTLED.lock().ok().and_then(|cache| settled(&cache)) {
        return Ok(value);
    }
    let value = parse(text)
        .and_then(|node| node.settled(prec))
        .map_err(|reason| format!("invalid number {text:?}: {reason}"))?;
    if let Ok(mut cache) = SETTLED.lock() {
        if cache.len() == 8 {
            cache.remove(0);
        }
        cache.push((text.to_string(), prec, value.clone()));
    }
    Ok(value)
}

/// A named constant with an optional sign.
fn signed_name(text: &str) -> Option<(bool, &'static Named)> {
    let trimmed = text.trim();
    let (negative, name) = match trimmed.strip_prefix('-') {
        Some(rest) => (true, rest.trim()),
        None => (false, trimmed.strip_prefix('+').unwrap_or(trimmed).trim()),
    };
    constants::named(name).map(|named| (negative, named))
}

/// A decimal number written in one piece. The decimal reader skips spaces
/// and underscores, so that `2 1` would be 21: such a text is left to the
/// grammar, where two numbers in a row are an error.
fn decimal(text: &str) -> bool {
    let gap = |c: char| c.is_whitespace() || c == '_';
    !text.trim().contains(gap) && Float::parse(text).is_ok()
}

/// The complex number that `text` stands for, to `prec` bits: a decimal
/// number, a named constant or a constant expression.
pub fn complex(text: &str, prec: u64) -> Result<Complex, String> {
    if decimal(text) {
        let value = cnum::parse_float(text, prec)?;
        return Ok(tidy(Complex::with_val_64(prec, (value, 0))));
    }
    if let Some((negative, named)) = signed_name(text) {
        let value = named.value(prec)?;
        let value = if negative { -value } else { value };
        return Ok(Complex::with_val_64(prec, (value, 0)));
    }
    Ok(Complex::with_val_64(prec, evaluated(text, prec)?))
}

/// The precision at which the inputs of a run are read: `prec`, or more for
/// an expression with a number of more digits written in it, and for one so
/// large that `prec` bits would make a whole number of a value that is
/// another number. Whether a base is one, a height is whole, and whether it
/// is even, decides exact cases: `re(1.0…01)` with fifty zeros is not the
/// base one at ten digits, and `10^100 + 1` must stay odd.
pub fn input_precision(inputs: &[&str], prec: u64) -> Result<u64, String> {
    let written = |text: &&&str| !decimal(text) && signed_name(text).is_none();
    let mut prec = prec;
    for text in inputs.iter().filter(written) {
        // A text that has no tokens is refused by the evaluation below.
        for token in tokens(text).unwrap_or_default() {
            if let Token::Number(number) = token {
                prec = prec.max(cnum::checked_input_precision(1, &[&number])?);
            }
        }
    }
    let mut needed = prec;
    for text in inputs.iter().filter(written) {
        let value = evaluated(text, prec)?;
        for part in [value.real(), value.imag()] {
            let rounded = Float::with_val_64(prec, part);
            let above = part.get_exp().and_then(|above| u64::try_from(above).ok());
            // Fewer than 64 bits are left below the unit, and none of them is set.
            match above {
                Some(above)
                    if rounded.is_integer()
                        && rounded != *part
                        && above.saturating_add(64) > prec =>
                {
                    let raised = prec.checked_add(above).ok_or("precision count overflow")?;
                    needed = needed.max(raised);
                }
                _ => {}
            }
        }
    }
    cnum::check_precision(needed)?;
    Ok(needed)
}

/// The real number that `text` stands for, to `prec` bits. Where a number
/// is given by its real and its imaginary part, each of the two is real: a
/// complex value is refused, not added to the other part.
pub fn real(text: &str, prec: u64) -> Result<Float, String> {
    let (re, im) = complex(text, prec)?.into_real_imag();
    if im.is_zero() {
        return Ok(re);
    }
    Err(format!(
        "invalid number {text:?}: it is not real (imaginary part {}); each of four numbers is \
         a real or an imaginary part: take re(...) of it, or give two complex numbers instead \
         of four",
        cnum::format_float(&im, 10)
    ))
}

/// The number `first + i second` of two real inputs.
pub fn pair(first: &str, second: &str, prec: u64) -> Result<Complex, String> {
    let (re, im) = (real(first, prec)?, real(second, prec)?);
    Ok(Complex::with_val_64(prec, (re, im)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digits(value: &Float, reference: &Float) -> Float {
        let prec = reference.prec_64();
        let miss = Float::with_val_64(prec, value - reference).abs();
        if miss.is_zero() {
            return Float::with_val_64(prec, rug::float::Special::Infinity);
        }
        -(miss / reference.clone().abs()).log10()
    }

    /// The value to `least` digits; an exact zero where the reference is 0.
    fn check(text: &str, reference: (&str, &str), requested: u64, least: u32) {
        let value = complex(text, cnum::digits_to_bits(requested))
            .unwrap_or_else(|error| panic!("{text}: {error}"));
        let wide = cnum::digits_to_bits(requested + 20);
        for (part, reference) in [(value.real(), reference.0), (value.imag(), reference.1)] {
            let reference = cnum::parse_float(reference, wide).unwrap();
            if reference.is_zero() {
                assert!(part.is_zero(), "{text} = {value}");
            } else {
                assert!(digits(part, &reference) >= least, "{text} = {value}");
            }
        }
    }

    #[test]
    fn expressions_equal_independent_references() {
        // mpmath, 60 digits: (expression, real part, imaginary part).
        for (text, re, im) in [
            ("-2^2 + 2^-2 + 2^3^2 + 2**3", "516.25", "0"),
            ("(pi + 1e-40) - pi", "1e-40", "0"),
            (
                "pi*e^e-pow(2, 0.5)",
                "46.1943053660313545855098891412422604801188997804070227383532",
                "0",
            ),
            (
                "e^(1/e)",
                "1.44466786100976613365833910859643022305859545324225316582052",
                "0",
            ),
            (
                "e^-e",
                "0.0659880358453125370767901875968464249385770482527964364024735",
                "0",
            ),
            (
                "sin(1)^2 + cos(1)^2 - 1/3",
                "0.666666666666666666666666666666666666666666666666666666666667",
                "0",
            ),
            (
                "gamma(1/3) * gamma - ln(10)/zeta(3) + W(1)",
                "0.197931063095274072469638942456607501685704785095999710629804",
                "0",
            ),
            (
                "atan2(1, -1) + log(1024, 2) + fact(5) + root(27, 3) + cbrt(8)",
                "137.356194490192344928846982537459627163147877049531329365731",
                "0",
            ),
            (
                "exp(pi*sqrt(163)) - 262537412640768744",
                "-0.00000000000074992740280181431112064614366266300913729246258962158441131",
                "0",
            ),
            (
                "(2+3i)*(1-500i)/sqrt(-4) + i^i + 2pi",
                "-492.008935116469651614527757613606015461571783359618018749975",
                "-751.0",
            ),
            (
                "(-8)^(1/3) + ln(-1) + exp(2i) + W(-1+i)",
                "0.856589823330667665131663314333581446577664429460939363637327",
                "6.82445397468964923865770400494623916562190682231859202850373",
            ),
            (
                "asin(2+i) + acosh(0.5+i) + sin(1+i) + abs(3-4i) + arg(-1+i) + conj(1+2i) + re(3+4i)*im(3+4i)",
                "23.6442251265362567044126413456627811015825119345218610010232",
                "1.32567292309060470711483638591753888767098248083126984512946",
            ),
            (
                "log2(8i) + log10(-100) + tan(1+i) + cot(1+i) + sec(i) + csc(1) + atan(2i+1) + sinh(1+i) + cosh(i) + tanh(1-i) + asinh(2i+1) + atanh(0.5+i)",
                "12.6322152025270293076055455637065610266529337038144329369152",
                "7.186545767651779602407746694692522854540679638212261146252",
            ),
            (
                "cbrt(-8) + root(-27, 3) + root(2i, 1+i) + exp2(1+i) + expf(1+i, 3) + powf(1+i, 0.5) + powi(1+i, -3) + powu(2i, 5)",
                "9.05953848193621252384550208311050072518608914917927680361172",
                "41.8027357300161122184954571587902201913731926247152309192177",
            ),
            (
                "sqr(1+2i) + inv(1+i) + sgn(3+4i) + cis(2) + scale(1+i, 3) + unscale(1+i, 4) + fdiv(1+i, 2-i) + complex(1, 2) + polar(2, 1)",
                "3.21445777518913704780430498538519101769862007016029956458504",
                "12.7422393964414747087010245091723428419473810930446323997245",
            ),
            (
                "to_polar(1+i) + avg(1+i, 3) + minabs(1+i, 3) + maxabs(1+i, -3) + abs2(1+2i) + l1(3-4i) + log(8i, 2)",
                "16.4142135623730950488016887242096980785696718753769480731767",
                "4.55157823431104521442950231867820912613522308370567424733674",
            ),
            (
                "floor(2.7-1.5i) + ceil(2.2-1.5i) + round(2.5-1.5i) + trunc(2.7-1.7i) + frac(2.75-1.25i)",
                "10.75",
                "-6.25",
            ),
            (
                "sech(1+i) + csch(1+i) + coth(1+i) + min(2, 3) + max(2, 3) + hypot(3, 4) + agm(1, 2)",
                "13.1270732061264450533705730884632286711689105075399556018128",
                "-1.43022342074587615065043374243549329319192519310448311507164",
            ),
            (
                "lgamma(0.5) + digamma(1) + erf(1) + erfc(2) + ei(1) + li2(0.5) + tgamma(4.5) + factorial(0.5)",
                "15.8378414707950855658068380698976354867356153618471681287632",
                "0",
            ),
        ] {
            check(text, (re, im), 50, 49);
        }
    }

    #[test]
    fn a_value_on_a_branch_cut_is_the_usual_one() {
        // mpmath, 60 digits; a 0 is a part that has to be exactly zero.
        for (text, re, im) in [
            (
                "asin(2)",
                "1.57079632679489661923132169163975144209858469968755291048747",
                "-1.31695789692481670862504634730796844402698197146751647976847",
            ),
            (
                "asin(-2)",
                "-1.57079632679489661923132169163975144209858469968755291048747",
                "1.31695789692481670862504634730796844402698197146751647976847",
            ),
            (
                "acos(2)",
                "0",
                "1.31695789692481670862504634730796844402698197146751647976847",
            ),
            (
                "acos(-2)",
                "3.14159265358979323846264338327950288419716939937510582097494",
                "-1.31695789692481670862504634730796844402698197146751647976847",
            ),
            (
                "atanh(2)",
                "0.549306144334054845697622618461262852323745278911374725867347",
                "-1.57079632679489661923132169163975144209858469968755291048747",
            ),
            (
                "atanh(-2)",
                "-0.549306144334054845697622618461262852323745278911374725867347",
                "1.57079632679489661923132169163975144209858469968755291048747",
            ),
            (
                "acosh(0.5)",
                "0",
                "1.04719755119659774615421446109316762806572313312503527365831",
            ),
            (
                "acosh(-2)",
                "1.31695789692481670862504634730796844402698197146751647976847",
                "3.14159265358979323846264338327950288419716939937510582097494",
            ),
            (
                "asinh(2i)",
                "1.31695789692481670862504634730796844402698197146751647976847",
                "1.57079632679489661923132169163975144209858469968755291048747",
            ),
            (
                "asinh(-2i)",
                "-1.31695789692481670862504634730796844402698197146751647976847",
                "-1.57079632679489661923132169163975144209858469968755291048747",
            ),
            (
                "atan(2i)",
                "1.57079632679489661923132169163975144209858469968755291048747",
                "0.549306144334054845697622618461262852323745278911374725867347",
            ),
            (
                "atan(-2i)",
                "-1.57079632679489661923132169163975144209858469968755291048747",
                "-0.549306144334054845697622618461262852323745278911374725867347",
            ),
            ("sqrt(-4)", "0", "2.0"),
            (
                "ln(-1)",
                "0",
                "3.14159265358979323846264338327950288419716939937510582097494",
            ),
            (
                "(-8)^(1/3)",
                "1.0",
                "1.73205080756887729352744634150587236694280525381038062805581",
            ),
            (
                "W(-1)",
                "-0.318131505204764135312654251587664517203517613871399866922379",
                "1.33723570143068940890116214319371061253950213846051241887631",
            ),
            (
                "log10(-100)",
                "2.0",
                "1.36437635384184134748578362543135577021012748372392539990085",
            ),
            (
                "log2(-8)",
                "3.0",
                "4.53236014182719380962768294571666681017186146772379558418602",
            ),
            (
                "cbrt(-8)",
                "1.0",
                "1.73205080756887729352744634150587236694280525381038062805581",
            ),
            (
                "sqrt(-pi)",
                "0",
                "1.77245385090551602729816748334114518279754945612238712821381",
            ),
            (
                "sin(i*pi)",
                "0",
                "11.5487393572577483779773343153884096844951890663947894552322",
            ),
            ("cosh(i*pi/3)", "0.5", "0"),
            (
                "(-pi)^0.5",
                "0",
                "1.77245385090551602729816748334114518279754945612238712821381",
            ),
            (
                "arg(-2)",
                "3.14159265358979323846264338327950288419716939937510582097494",
                "0",
            ),
            ("(-2)^3", "-8.0", "0"),
            ("(-2)^-2", "0.25", "0"),
        ] {
            check(text, (re, im), 50, 49);
        }
    }

    #[test]
    fn the_grammar_is_the_one_of_the_plotters() {
        let prec = cnum::digits_to_bits(30);
        let value = |text: &str| complex(text, prec).unwrap_or_else(|error| panic!("{error}"));
        for (text, whole) in [
            // A sign binds less than a power, a power less than a tetration;
            // both group from the right.
            ("-2^2", -4),
            ("2^3^2", 512),
            ("2^^3^2", 256),
            ("3^2^^2", 81),
            ("2^^2^^2", 65536),
            ("-2^^2", -4),
            ("-2*3", -6),
            ("2--3", 5),
            ("6/3/2", 1),
            ("2*-3", -6),
            // The product without a sign.
            ("2(3+4)", 14),
            ("(1+2)(3+4)", 21),
            ("(2) 3", 6),
            ("2sqrt(4)", 4),
            ("i i", -1),
            ("J*j", -1),
            ("2e3", 2000),
            ("2e+3", 2000),
            ("1.5e1", 15),
            ("√2 ** 2", 2),
            ("6 ÷ 3 × 2 − 1", 3),
            ("2·3", 6),
            ("gamma(5)", 24),
        ] {
            assert_eq!(value(text), whole, "{text}");
        }
        assert_eq!(value("2^-2"), 0.25);
        assert_eq!(value(".5"), 0.5);
        for (text, same) in [
            ("2pi", "tau"),
            ("2 pi", "tau"),
            ("pi 2", "tau"),
            ("PI", "pi"),
            ("2e", "e+e"),
            ("e(2)", "e+e"),
            ("2e+pi", "e+e+pi"),
            ("gamma", "eulergamma"),
            ("500i", "complex(0, 500)"),
        ] {
            assert_eq!(value(text), value(same), "{text}");
        }
    }

    #[test]
    fn tetration_and_its_inverses_inside_expressions() {
        let prec = cnum::digits_to_bits(20);
        let value = |text: &str| complex(text, prec).unwrap();
        assert_eq!(value("2^^2"), 4);
        assert_eq!(value("2^^3 + tet(2, 2)"), 20);
        // 1.5^^2 = 1.5^1.5, a fractional height and the inverses.
        let near = |a: &Complex, b: &Complex| {
            cnum::abs(&Complex::with_val_64(prec, a - b), prec) < cnum::epsilon(19, prec)
        };
        assert!(near(&value("1.5^^2"), &value("1.5^1.5")));
        let half = value("2^^0.5");
        let direct = dispatch::tetrate(&value("2"), &value("0.5"), prec, 20).unwrap();
        assert!(near(&half, &direct), "{half}");
        assert!(near(&value("slog(2, 2^^0.5)"), &value("0.5")));
        assert!(near(&value("ate(2, 16)"), &value("3")));
        assert!(near(&value("ssrt(16, 3)"), &value("2")));
        assert!(near(&value("sroot(2^^0.5, 0.5)"), &value("2")));
        // The two parts of a number may be any real expressions.
        let base = pair("2^^2", "abs(sqrt(-1))", prec).unwrap();
        assert_eq!(base, Complex::with_val_64(prec, (4, 1)));
        let base = pair("pi", "-e", prec).unwrap();
        assert_eq!(*base.real(), *value("pi").real());
        assert_eq!(*base.imag(), *value("-e").real());
        let error = complex("0^^0.5", prec).unwrap_err();
        assert!(
            error.starts_with("invalid number") && error.contains("^^"),
            "{error}"
        );
    }

    #[test]
    fn decimals_names_and_signs() {
        let prec = cnum::digits_to_bits(30);
        // A decimal number is read exactly as before.
        for text in ["0.1", "-12345.6789", "1e-5", "2.5E3", "+7", " 12 ", "1@3"] {
            let value = complex(text, prec).unwrap();
            assert_eq!(*value.real(), cnum::parse_float(text, prec).unwrap());
            assert!(value.imag().is_zero());
        }
        // Not with a gap in it: `2 1` is two numbers, not 21.
        for (text, reason) in [
            ("2 1", "an operator is missing before 1"),
            ("0.5 0", "an operator is missing before 0"),
            ("1_2", "unknown name _2"),
            ("1 . 5", "unexpected character '.'"),
            ("1e 2 3", "an operator is missing before 3"),
        ] {
            let error = complex(text, prec).unwrap_err();
            assert!(error.contains(reason), "{text}: {error}");
            assert!(input_precision(&[text], prec).is_err(), "{text}");
        }
        assert_eq!(complex("- 3", prec).unwrap(), complex("-3", prec).unwrap());
        // A name alone is the constant rounded to nearest, and so is the
        // value of an expression.
        let pi = constants::named("pi").unwrap().value(prec).unwrap();
        for text in ["pi", " PI ", "+π", "pi + 0", "4*atan(1)", "2*halfpi"] {
            assert_eq!(*complex(text, prec).unwrap().real(), pi, "{text}");
        }
        assert_eq!(*complex(" -PI ", prec).unwrap().real(), -pi);
    }

    #[test]
    fn cancellation_is_measured() {
        // Fifty digits that cancel, then three hundred, are followed.
        check("(pi + 1e-50) - pi", ("1e-50", "0"), 40, 39);
        check("(pi + 1e-300) - pi", ("1e-300", "0"), 40, 39);
        // A term below the precision of both evaluations is not lost when
        // the larger one cancels later.
        check("((pi + 1e-300) - pi) * 1e300 + 1", ("2", "0"), 40, 39);
        let prec = cnum::digits_to_bits(40);
        let value = complex("(1 + 2^-159 + 2^-230) - 1", prec).unwrap();
        let exact = complex("2^-159 + 2^-230", prec).unwrap();
        assert_eq!(value, exact);
        assert_eq!(exact.real().get_exp(), Some(-158));
    }

    #[test]
    fn a_large_value_keeps_its_whole_part() {
        let prec = cnum::digits_to_bits(15);
        // 10^100 + 1 has 333 bits: the run reads its inputs with that many more.
        let raised = input_precision(&["0", "10^100 + 1", "pi", "2^200"], prec).unwrap();
        assert_eq!(raised, prec + 333);
        let odd = complex("10^100 + 1", raised).unwrap();
        assert!(odd.real().is_integer());
        assert!(odd.real().to_integer().unwrap().is_odd());
        // A value that is not whole is not made whole either.
        let raised = input_precision(&["e^300"], prec).unwrap();
        assert_eq!(raised, prec + 433);
        assert!(!complex("e^300", raised).unwrap().real().is_integer());
        // Nothing changes for values of ordinary size, near a whole number or not.
        for text in [
            "2^100",
            "10^30 + 1/2",
            "3 + 1e-200",
            "ate(2, 16)",
            "-pi",
            "0.1",
            "1.5e-300 * 2e+250 + 0.000123456789012345i",
        ] {
            assert_eq!(input_precision(&[text], prec).unwrap(), prec, "{text}");
        }
    }

    #[test]
    fn a_number_written_in_an_expression_keeps_its_digits() {
        let prec = cnum::digits_to_bits(15);
        let near_one = format!("1.{}1", "0".repeat(40));
        let digits = cnum::digits_to_bits(42);
        // Next to the name `re`, after an exponent, as an imaginary part.
        for text in [
            format!("re({near_one})"),
            format!("1e-3 + {near_one}i"),
            format!("2 * {near_one}e-7 + pi"),
            format!("{near_one}^^2"),
        ] {
            let raised = input_precision(&["pi", &text, "2"], prec).unwrap();
            assert_eq!(raised, digits, "{text}");
        }
        let text = format!("re({near_one})");
        let base = complex(&text, input_precision(&[&text], prec).unwrap()).unwrap();
        assert!(*base.real() > 1 && base.imag().is_zero());
        assert_eq!(base, complex(&near_one, digits).unwrap());
        // A decimal input alone is sized by the caller, as before.
        assert_eq!(input_precision(&[&near_one], prec).unwrap(), prec);
    }

    #[test]
    fn a_zero_is_exact_or_refused() {
        let prec = cnum::digits_to_bits(40);
        // Exact by construction: exact operands, a zero factor, an axis.
        for (text, zero_re, zero_im) in [
            ("1/4 + 1/4 - 1/2", true, true),
            ("sin(0) + ln(1)", true, true),
            ("floor(sqrt(2)/2)", true, true),
            ("(1+i)^2", true, false),
            ("(1+i)*(1-i)", false, true),
            ("sqrt(2)*i", true, false),
            ("pi*i/2", true, false),
            ("i*pi*i", false, true),
            ("abs(pi+i*e)", false, true),
            ("im(pi)", true, true),
            ("sqrt(2)*i + sqrt(3)*i", true, false),
            ("round(0.3 + 0.4i)", true, true),
            ("floor(0.3 + 1.4i)", true, false),
        ] {
            let value = complex(text, prec).unwrap_or_else(|error| panic!("{text}: {error}"));
            assert_eq!(value.real().is_zero(), zero_re, "{text} = {value}");
            assert_eq!(value.imag().is_zero(), zero_im, "{text} = {value}");
        }
        // A zero that no precision shows is not guessed, in either part.
        for text in [
            "sin(pi)",
            "sqrt(2)^2 - 2",
            "exp(ln(3)) - 3",
            "exp(i*pi)",
            "exp(i*pi/2)*2",
            "pi - pi",
            "(pi + pi*i)^2",
            "(pi + 1e-2000) - pi",
            "1/(pi - pi)",
        ] {
            let error = complex(text, prec).unwrap_err();
            assert!(error.contains("cancels"), "{text}: {error}");
        }
    }

    #[test]
    fn a_real_or_an_imaginary_part_is_real() {
        let prec = cnum::digits_to_bits(30);
        for (text, same) in [
            ("-12.5", "-12.5"),
            ("-pi", "-pi"),
            ("sqrt(2)^3", "2*sqrt(2)"),
            ("abs(3+4i)", "5"),
            ("re(2+i)", "2"),
            ("real(pi*i + e)", "e"),
            ("im(2+3i)", "3"),
            ("(1+i)*(1-i)", "2"),
            ("re(exp(i*pi))", "-1"),
            ("ssrt(16, 3)", "2"),
        ] {
            let value = real(text, prec).unwrap_or_else(|error| panic!("{text}: {error}"));
            assert_eq!(value, *complex(same, prec).unwrap().real(), "{text}");
        }
        for (text, part) in [
            ("i", "1.000000000"),
            ("2-3i", "-3.000000000"),
            ("sqrt(-4)", "2.000000000"),
            ("ln(-1)", "3.141592654"),
            ("1+1e-30i", "1.000000000e-30"),
        ] {
            let error = real(text, prec).unwrap_err();
            let start = format!("invalid number {text:?}: it is not real (imaginary part {part});");
            assert!(error.starts_with(&start), "{error}");
            assert!(error.contains("re(...)") && error.contains("two complex numbers"));
            for (first, second) in [(text, "0"), ("0", text)] {
                assert_eq!(pair(first, second, prec).unwrap_err(), error);
            }
        }
        // A part that cannot be told from zero is neither real nor complex.
        for text in ["exp(i*pi)", "pi*i - i*pi + 2"] {
            assert!(real(text, prec).unwrap_err().contains("cancels"), "{text}");
        }
        // A number written out is the number of its two parts, and -0 is 0.
        for (text, re, im) in [
            ("0.1+0.2i", "0.1", "0.2"),
            ("-0.1-0.2i", "-0.1", "-0.2"),
            ("1e-5 - 2.5e3i", "1e-5", "-2.5e3"),
            ("-0.3i", "0", "-0.3"),
            ("i", "0", "1"),
            ("7", "7", "0"),
            ("-0", "0", "0"),
            ("0", "-0", "-0"),
        ] {
            let (value, parts) = (complex(text, prec).unwrap(), pair(re, im, prec).unwrap());
            assert_eq!(value, parts, "{text}");
            for part in [value.real(), value.imag(), parts.real(), parts.imag()] {
                assert!(part.is_sign_positive() || !part.is_zero(), "{text}");
            }
        }
    }

    #[test]
    fn errors_say_why() {
        let prec = cnum::digits_to_bits(20);
        for (text, reason) in [
            ("bad", "unknown name bad"),
            ("x + 1", "unknown name x"),
            ("z", "unknown name z"),
            ("2 +", "ends early"),
            ("(1 + 2", "closing bracket"),
            ("2 3 + 1", "operator is missing before 3"),
            ("foo(1)", "unknown function foo"),
            ("ln + 1", "ln needs its arguments in brackets"),
            ("sqrt(1, 2)", "does not take 2 arguments"),
            ("gamma(i)", "real arguments only"),
            ("powi(2, 0.5)", "whole exponent"),
            ("powu(2, -1)", "not negative"),
            ("scale(2, i)", "real second argument"),
            ("ln(0)", "infinite"),
            ("1/(2-2)", "division by zero"),
            ("root(2, 0)", "division by zero"),
            ("2 $ 3", "unexpected character"),
            ("exp(1e30)", "infinite"),
            ("arg(0)", "not defined"),
            ("gamma(0)", "infinite"),
            ("gamma(-1)", "not defined"),
            ("1e999999999999999999999 + 1", "must be finite"),
        ] {
            let error = complex(text, prec).unwrap_err();
            assert!(error.starts_with("invalid number"), "{text}: {error}");
            assert!(error.contains(reason), "{text}: {error}");
        }
        // Non-finite spellings stay errors of the decimal reader.
        for text in ["nan", "inf", "-inf", "Infinity"] {
            assert!(complex(text, prec).is_err(), "{text}");
        }
    }

    #[test]
    fn every_function_is_listed_once_and_evaluates() {
        let prec = cnum::digits_to_bits(25);
        let mut seen = std::collections::HashSet::new();
        for function in FUNCTIONS {
            for name in function.names {
                assert!(seen.insert(*name), "{name} is listed twice");
                assert!(
                    constants::named(name).is_none() || *name == "gamma",
                    "{name}"
                );
                let arguments = match *name {
                    "powi" | "powu" => "0.5, 3".to_string(),
                    "sroot" | "ssrt" => "1.25, 0.5".to_string(),
                    _ => vec!["0.5"; function.arguments[0]].join(", "),
                };
                let text = format!("{name}({arguments})");
                let value = complex(&text, prec).unwrap_or_else(|error| panic!("{text}: {error}"));
                assert!(cnum::is_finite(&value), "{text}");
            }
        }
        assert_eq!(help().lines().count(), FUNCTIONS.len());
    }
}
