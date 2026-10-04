//! Retired linear surrogate API. Use the analytic dispatcher instead.

use rug::Complex;

/// Kept as an explicit error for callers of the former approximation API.
#[deprecated(note = "linear interpolation is not analytic tetration; use dispatch::tetrate")]
pub fn tetrate_linear(_b: &Complex, _h: &Complex, _prec: u64) -> Result<Complex, String> {
    Err("linear tetration approximation is disabled: it does not converge to an analytic tetration value".into())
}
