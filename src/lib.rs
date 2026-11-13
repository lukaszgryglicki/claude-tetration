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

pub mod cnum;
pub mod dispatch;
pub mod fft;
pub mod integer_height;
pub mod inverse;
pub mod kneser;
pub mod kneser_complex;
pub mod kouznetsov;
pub mod lambertw;
pub mod linear_approx;
pub mod mt;
pub mod regions;
pub mod schroder;
mod schroder_complex_jumps;
mod schroder_jumps;

/// Top-level string-in / string-out API. Parses precision and complex inputs as
/// decimal strings, dispatches to the appropriate algorithm, and formats the
/// result back to decimal. Returns `(real_part, imaginary_part)`.
pub fn tetrate_str(
    prec_str: &str,
    base_re: &str,
    base_im: &str,
    height_re: &str,
    height_im: &str,
) -> Result<(String, String), String> {
    let digits: u64 = prec_str
        .parse()
        .map_err(|_| format!("invalid precision: {:?}", prec_str))?;
    let mut prec =
        cnum::checked_input_precision(digits, &[base_re, base_im, height_re, height_im])?;
    let result = loop {
        let b = cnum::parse_complex(base_re, base_im, prec)?;
        let h = cnum::parse_complex(height_re, height_im, prec)?;
        let value = dispatch::tetrate(&b, &h, prec, digits)?;
        let next_prec = value.real().prec_64().max(value.imag().prec_64());
        if next_prec > prec
            && (cnum::parse_complex(base_re, base_im, next_prec)? != b
                || cnum::parse_complex(height_re, height_im, next_prec)? != h)
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
        return Err("tetration returned a non-finite result".into());
    }
    let output_digits =
        usize::try_from(digits).map_err(|_| "output precision exceeds addressable memory")?;
    Ok(cnum::format_complex(&result, output_digits))
}

/// String API for the super-logarithm: the height `h` with `F_b(h) = z`.
pub fn slog_str(
    prec_str: &str,
    base_re: &str,
    base_im: &str,
    value_re: &str,
    value_im: &str,
) -> Result<(String, String), String> {
    inverse_str(
        prec_str,
        [base_re, base_im, value_re, value_im],
        inverse::slog,
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
    inverse_str(
        prec_str,
        [value_re, value_im, height_re, height_im],
        inverse::sroot,
    )
}

fn inverse_str(
    prec_str: &str,
    inputs: [&str; 4],
    solve: fn(&rug::Complex, &rug::Complex, u64, u64) -> Result<rug::Complex, String>,
) -> Result<(String, String), String> {
    let digits: u64 = prec_str
        .parse()
        .map_err(|_| format!("invalid precision: {:?}", prec_str))?;
    let mut prec = cnum::checked_input_precision(digits, &inputs)?;
    let result = loop {
        let first = cnum::parse_complex(inputs[0], inputs[1], prec)?;
        let second = cnum::parse_complex(inputs[2], inputs[3], prec)?;
        let value = solve(&first, &second, prec, digits)?;
        let next_prec = value.real().prec_64().max(value.imag().prec_64());
        if next_prec > prec
            && (cnum::parse_complex(inputs[0], inputs[1], next_prec)? != first
                || cnum::parse_complex(inputs[2], inputs[3], next_prec)? != second)
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
        return Err("inverse tetration returned a non-finite result".into());
    }
    let output_digits =
        usize::try_from(digits).map_err(|_| "output precision exceeds addressable memory")?;
    Ok(cnum::format_complex(&result, output_digits))
}
