//! Arbitrary-precision complex tetration.
//!
//! Computes `F_b(h)` where `F_b(0) = 1` and `F_b(z+1) = b^F_b(z)` for complex `b`
//! and complex `h`, at user-specified decimal precision.
//!
//! See `dispatch::tetrate` for the algorithm-selection entry point. The CLI in
//! `main.rs` is a thin wrapper around `tetrate_str`.

pub mod cnum;
pub mod dispatch;
pub mod fft;
pub mod integer_height;
pub mod kouznetsov;
pub mod lambertw;
pub mod linear_approx;
pub mod mt;
pub mod regions;
pub mod schroder;

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
