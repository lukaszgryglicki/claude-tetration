//! Arbitrary-precision complex tetration.
//!
//! Computes `F_b(h)` where `F_b(0) = 1` and `F_b(z+1) = b^F_b(z)` for complex `b`
//! and complex `h`, at user-specified decimal precision.
//!
//! See `dispatch::tetrate` for the algorithm-selection entry point. The CLI in
//! `main.rs` is a thin wrapper around `tetrate_str`.
//!
//! # Numerical contract
//!
//! Binding for every change to this crate (README § 7.1):
//!
//! * Accuracy comes before speed: a faster method must return the same digits.
//! * A result is a value the algorithm has converged to at the requested
//!   precision. Interpolations, extrapolations, truncated asymptotics, values
//!   of another solution family and other approximations are never returned;
//!   where nothing better is available the answer is an `Err`.
//! * Approximations may only start or steer an iteration that then converges
//!   by itself; its last steps and the final check run at the full working
//!   precision.
//! * Every number that enters a result is an MPFR/MPC value with at least the
//!   requested digits plus guard digits, and more where digits are lost: the
//!   loss is measured and the computation repeated at a higher precision.
//!   Real quantities are `rug::Float`, complex ones `rug::Complex`, at the
//!   working precision, also where they only steer the computation. An `f64`
//!   or a fixed-size float needs a reason (a starting value, or a proof that
//!   no digit depends on it) and a line in the table of README § 7.1.
//! * No fixed cap on iterations, series lengths or grid sizes may limit the
//!   accuracy silently.
//! * Support for new bases, heights or values is complete only when
//!   tetration, the super-logarithm (`--slog`) and the super-root (`--sroot`)
//!   all work correctly on them, each with its own regression test.

pub mod cnum;
pub mod constants;
pub mod dispatch;
pub mod expression;
pub mod fft;
pub mod integer_height;
pub mod inverse;
pub mod kneser;
pub mod kneser_complex;
pub mod kouznetsov;
pub mod lambertw;
pub mod mt;
pub mod regions;
pub mod schroder;
mod schroder_complex_jumps;
mod schroder_jumps;

/// One number of a run as it is written: a complex input, or its real and
/// its imaginary part.
#[derive(Clone, Copy)]
enum Input<'a> {
    Number(&'a str),
    Parts(&'a str, &'a str),
}

impl<'a> Input<'a> {
    fn texts(self) -> Vec<&'a str> {
        match self {
            Input::Number(text) => vec![text],
            Input::Parts(re, im) => vec![re, im],
        }
    }

    fn value(self, prec: u64) -> Result<rug::Complex, String> {
        match self {
            Input::Number(text) => expression::complex(text, prec),
            Input::Parts(re, im) => cnum::parse_complex(re, im, prec),
        }
    }
}

type Solve = fn(&rug::Complex, &rug::Complex, u64, u64) -> Result<rug::Complex, String>;

/// Top-level string-in / string-out API. Parses precision and complex inputs as
/// decimal strings, dispatches to the appropriate algorithm, and formats the
/// result back to decimal. Returns `(real_part, imaginary_part)`. Each of the
/// four numbers must be real.
pub fn tetrate_str(
    prec_str: &str,
    base_re: &str,
    base_im: &str,
    height_re: &str,
    height_im: &str,
) -> Result<(String, String), String> {
    solve_str(
        prec_str,
        [
            Input::Parts(base_re, base_im),
            Input::Parts(height_re, height_im),
        ],
        dispatch::tetrate,
        "tetration",
    )
}

/// `tetrate_str` with the base and the height as one complex number each.
pub fn tetrate_complex_str(
    prec_str: &str,
    base: &str,
    height: &str,
) -> Result<(String, String), String> {
    solve_str(
        prec_str,
        [Input::Number(base), Input::Number(height)],
        dispatch::tetrate,
        "tetration",
    )
}

/// String API for the super-logarithm: the height `h` with `F_b(h) = z`.
pub fn slog_str(
    prec_str: &str,
    base_re: &str,
    base_im: &str,
    value_re: &str,
    value_im: &str,
) -> Result<(String, String), String> {
    solve_str(
        prec_str,
        [
            Input::Parts(base_re, base_im),
            Input::Parts(value_re, value_im),
        ],
        inverse::slog,
        "inverse tetration",
    )
}

/// `slog_str` with the base and the value as one complex number each.
pub fn slog_complex_str(
    prec_str: &str,
    base: &str,
    value: &str,
) -> Result<(String, String), String> {
    solve_str(
        prec_str,
        [Input::Number(base), Input::Number(value)],
        inverse::slog,
        "inverse tetration",
    )
}

/// String API for the super-root: the base `b` with `F_b(h) = z`.
pub fn sroot_str(
    prec_str: &str,
    value_re: &str,
    value_im: &str,
    height_re: &str,
    height_im: &str,
) -> Result<(String, String), String> {
    solve_str(
        prec_str,
        [
            Input::Parts(value_re, value_im),
            Input::Parts(height_re, height_im),
        ],
        inverse::sroot,
        "inverse tetration",
    )
}

/// `sroot_str` with the value and the height as one complex number each.
pub fn sroot_complex_str(
    prec_str: &str,
    value: &str,
    height: &str,
) -> Result<(String, String), String> {
    solve_str(
        prec_str,
        [Input::Number(value), Input::Number(height)],
        inverse::sroot,
        "inverse tetration",
    )
}

fn solve_str(
    prec_str: &str,
    inputs: [Input; 2],
    solve: Solve,
    name: &str,
) -> Result<(String, String), String> {
    let digits: u64 = prec_str
        .parse()
        .map_err(|_| format!("invalid precision: {:?}", prec_str))?;
    let texts = [inputs[0].texts(), inputs[1].texts()].concat();
    let prec = cnum::checked_input_precision(digits, &texts)?;
    let mut prec = expression::input_precision(&texts, prec)?;
    let result = loop {
        let first = inputs[0].value(prec)?;
        let second = inputs[1].value(prec)?;
        let value = solve(&first, &second, prec, digits)?;
        let next_prec = value.real().prec_64().max(value.imag().prec_64());
        if next_prec > prec
            && (inputs[0].value(next_prec)? != first || inputs[1].value(next_prec)? != second)
        {
            if cnum::verbose() {
                eprintln!(
                    "tet: reparsing decimal inputs at {next_prec} bits after precision refinement"
                );
            }
            prec = next_prec;
            continue;
        }
        break value;
    };
    if !cnum::is_finite(&result) {
        return Err(format!("{name} returned a non-finite result"));
    }
    let output_digits =
        usize::try_from(digits).map_err(|_| "output precision exceeds addressable memory")?;
    Ok(cnum::format_complex(&result, output_digits))
}
