//! Phase-4 verification: Schröder regular tetration for Shell-Thron interior
//! bases.
//!
//! The defining property is the Abel-tetration functional equation
//!     F_b(z + 1) = b^{F_b(z)}.
//! With Schröder construction we also have F_b(0) = 1, F_b(1) = b, F_b(2)=b^b
//! by analytical identity. Numerical residuals reflect series truncation only.

use rug::{Complex, Float};

use tetration::{cnum, dispatch};

fn parse(re: &str, im: &str, prec: u32) -> Complex {
    cnum::parse_complex(re, im, prec).unwrap()
}

fn abs(z: &Complex, prec: u32) -> Float {
    Float::with_val(prec, z.abs_ref())
}

/// Returns -log10(|a - b|) — i.e., approximate matching digits.
fn matching_digits(a: &Complex, b: &Complex, prec: u32) -> f64 {
    let diff = Complex::with_val(prec, a - b);
    let da = abs(&diff, prec);
    if da.is_zero() {
        return f64::INFINITY;
    }
    -da.to_f64().log10()
}

fn check_functional_eq(b_re: &str, b_im: &str, z_re: &str, z_im: &str, digits: u64, expected_match: f64) {
    let prec = cnum::digits_to_bits(digits);
    let b = parse(b_re, b_im, prec);
    let z = parse(z_re, z_im, prec);
    let one = parse("1", "0", prec);
    let z_plus_1 = Complex::with_val(prec, &z + &one);

    let fz = dispatch::tetrate(&b, &z, prec, digits).unwrap();
    let fz1 = dispatch::tetrate(&b, &z_plus_1, prec, digits).unwrap();
    let b_to_fz = cnum::pow_complex(&b, &fz, prec);

    let m = matching_digits(&fz1, &b_to_fz, prec);
    assert!(
        m >= expected_match,
        "b={}+{}i z={}+{}i: F(z+1)={}+{}i  b^F(z)={}+{}i  matched {} digits",
        b_re, b_im, z_re, z_im,
        fz1.real(), fz1.imag(),
        b_to_fz.real(), b_to_fz.imag(),
        m
    );
}

#[test]
fn t300_sqrt2_functional_eq_at_half() {
    check_functional_eq("1.4142135623730950488", "0", "0.5", "0", 50, 35.0);
}

#[test]
fn t301_sqrt2_functional_eq_at_one_and_a_half() {
    check_functional_eq("1.4142135623730950488", "0", "1.5", "0", 50, 35.0);
}

#[test]
fn t302_sqrt2_functional_eq_negative_half() {
    // F(-0.5) requires the series to handle |λ^{-0.5}|>1; if the shift logic
    // engages, F(-0.5) = log_b(F(0.5)) gives the same answer.
    check_functional_eq("1.4142135623730950488", "0", "-0.5", "0", 50, 35.0);
}

#[test]
fn t310_b_one_point_two_functional_eq() {
    check_functional_eq("1.2", "0", "0.5", "0", 50, 40.0);
}

#[test]
fn t311_b_one_point_two_complex_height() {
    check_functional_eq("1.2", "0", "0.5", "0.3", 50, 35.0);
}

#[test]
fn t320_b_zero_point_five_functional_eq() {
    // b = 0.5, λ = -0.444 (inside Shell-Thron, real-negative multiplier).
    check_functional_eq("0.5", "0", "0.7", "0", 50, 35.0);
}

#[test]
fn t330_complex_base_functional_eq() {
    // b = 1.3 + 0.1i — interior of Shell-Thron with truly complex multiplier.
    check_functional_eq("1.3", "0.1", "0.4", "0.2", 50, 30.0);
}

#[test]
fn t340_high_precision_functional_eq() {
    // At 100 digits, expect at least ~70 digits of agreement.
    check_functional_eq("1.4142135623730950488", "0", "0.5", "0", 100, 70.0);
}

#[test]
fn t350_continuity_at_integer() {
    // F(1 + ε) should be very close to F(1) = b for small real ε.
    let digits = 50;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("1.2", "0", prec);
    let h = parse("1.0000000001", "0", prec);
    let f = dispatch::tetrate(&b, &h, prec, digits).unwrap();
    let diff = Complex::with_val(prec, &f - &b);
    let da = abs(&diff, prec).to_f64();
    assert!(da < 1e-9, "F(1.0000000001) - b ≈ {}, expected < 1e-9", da);
}

#[test]
fn t351_continuity_at_two() {
    // F(2 - ε) should be close to F(2) = b^b.
    let digits = 50;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("1.2", "0", prec);
    let h_int = parse("2", "0", prec);
    let h_near = parse("1.99999999999", "0", prec);
    let f_int = dispatch::tetrate(&b, &h_int, prec, digits).unwrap();
    let f_near = dispatch::tetrate(&b, &h_near, prec, digits).unwrap();
    let diff = Complex::with_val(prec, &f_int - &f_near);
    let da = abs(&diff, prec).to_f64();
    assert!(da < 1e-9, "F(2) − F(2−1e-11) ≈ {}, expected < 1e-9", da);
}

#[test]
fn t360_precision_scaling() {
    // Same input at p and 2p digits; first p−5 digits must agree.
    let digits_lo = 40;
    let digits_hi = 80;
    let prec_lo = cnum::digits_to_bits(digits_lo);
    let prec_hi = cnum::digits_to_bits(digits_hi);
    let b_lo = parse("1.4142135623730950488", "0", prec_lo);
    let b_hi = parse("1.4142135623730950488", "0", prec_hi);
    let h_lo = parse("0.5", "0", prec_lo);
    let h_hi = parse("0.5", "0", prec_hi);
    let f_lo = dispatch::tetrate(&b_lo, &h_lo, prec_lo, digits_lo).unwrap();
    let f_hi = dispatch::tetrate(&b_hi, &h_hi, prec_hi, digits_hi).unwrap();
    // Promote f_lo to high precision for comparison.
    let f_lo_hi = Complex::with_val(prec_hi, &f_lo);
    let diff = Complex::with_val(prec_hi, &f_hi - &f_lo_hi);
    let da = abs(&diff, prec_hi).to_f64();
    assert!(da < 1e-30, "precision-scaling residual {} too large", da);
}
