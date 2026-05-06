//! Arbitrary-precision complex tetration.
//!
//! Computes `F_b(h)` where `F_b(0) = 1` and `F_b(z+1) = b^F_b(z)` for complex `b`
//! and complex `h`, at user-specified decimal precision.
//!
//! See `dispatch::tetrate` for the algorithm-selection entry point. The CLI in
//! `main.rs` is a thin wrapper around `tetrate_str`.

pub mod cnum;
pub mod dispatch;
pub mod integer_height;
pub mod kouznetsov;
pub mod lambertw;
pub mod linear_approx;
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
    if digits == 0 {
        return Err("precision must be a positive integer".into());
    }
    let prec = cnum::digits_to_bits(digits);
    let b = cnum::parse_complex(base_re, base_im, prec)?;
    let h = cnum::parse_complex(height_re, height_im, prec)?;
    let result = dispatch::tetrate(&b, &h, prec, digits)?;
    Ok(cnum::format_complex(&result, digits as usize))
}
