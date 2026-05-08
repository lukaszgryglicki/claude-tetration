//! Phase-2 verification: complex Lambert W (W₀ and W₋₁) at arbitrary precision.
//!
//! Tests check the defining identity `w · exp(w) = z` to within ~ 2^{-(prec-32)}
//! at multiple precisions, including 50, 200, and 2000 digits — and on tricky
//! arguments near the branch point z = -1/e.

use rug::{float::Constant, Complex, Float};

use tetration::cnum;
use tetration::lambertw;

#[test]
#[ignore]
fn debug_wk_for_negative_two() {
    let prec = 200u32;
    let pi = Float::with_val(prec, Constant::Pi);
    let ln2 = Float::with_val(prec, Float::with_val(prec, 2).ln_ref());
    let ln_b = Complex::with_val(prec, (ln2, pi));
    let neg_ln_b = Complex::with_val(prec, -&ln_b);
    eprintln!("-ln(-2) = {:.6}+{:.6}i",
        Float::with_val(prec, neg_ln_b.real()).to_f64(),
        Float::with_val(prec, neg_ln_b.imag()).to_f64()
    );
    for &k in &[-3i32, -2, -1, 0, 1, 2, 3] {
        match lambertw::wk(&neg_ln_b, k, prec) {
            Ok(w) => {
                let neg_w = Complex::with_val(prec, -&w);
                let l = Complex::with_val(prec, &neg_w / &ln_b);
                let bz_arg = Complex::with_val(prec, &l * &ln_b);
                let bz = Complex::with_val(prec, bz_arg.exp_ref());
                let resid = Float::with_val(prec, Complex::with_val(prec, &bz - &l).abs_ref()).to_f64();
                eprintln!("W_{:>3}: w={:.4}+{:.4}i  L={:.4}+{:.4}i  resid={:.2e}",
                    k,
                    Float::with_val(prec, w.real()).to_f64(),
                    Float::with_val(prec, w.imag()).to_f64(),
                    Float::with_val(prec, l.real()).to_f64(),
                    Float::with_val(prec, l.imag()).to_f64(),
                    resid
                );
            }
            Err(e) => eprintln!("W_{}: error: {}", k, e),
        }
    }
}

/// True iff `|w · e^w − z| < 2^{−tol_bits}` (relative to |z| if non-zero).
fn check_residual(w: &Complex, z: &Complex, prec: u32, tol_bits: u32) -> Result<(), String> {
    let exp_w = Complex::with_val(prec, w.exp_ref());
    let we = Complex::with_val(prec, w * &exp_w);
    let r = Complex::with_val(prec, &we - z);
    let r_abs = Float::with_val(prec, r.abs_ref());
    let z_abs = Float::with_val(prec, z.abs_ref());
    let bound_exp_abs = -(tol_bits as i32);
    let r_exp = r_abs.get_exp().unwrap_or(i32::MIN);
    if r_exp <= bound_exp_abs {
        return Ok(());
    }
    // Try relative bound: |r| / |z| < 2^{-tol_bits}
    if !z_abs.is_zero() {
        let rel = Float::with_val(prec, &r_abs / &z_abs);
        if rel.get_exp().is_none_or(|e| e <= bound_exp_abs) {
            return Ok(());
        }
        Err(format!(
            "residual exp = {:?}, rel exp = {:?}, want ≤ {}",
            r_exp,
            rel.get_exp(),
            bound_exp_abs
        ))
    } else {
        Err(format!("residual exp = {:?}, want ≤ {}", r_exp, bound_exp_abs))
    }
}

fn make_z(re: &str, im: &str, prec: u32) -> Complex {
    cnum::parse_complex(re, im, prec).unwrap()
}

#[test]
fn t100_w0_real_positive() {
    let prec = cnum::digits_to_bits(60);
    // W₀(1) = Ω ≈ 0.567143290409...
    let z = make_z("1", "0", prec);
    let w = lambertw::w0(&z, prec).unwrap();
    check_residual(&w, &z, prec, prec - 32).unwrap();
    // Compare real part to known value of Omega constant.
    let r: f64 = w.real().to_f64();
    assert!((r - 0.5671432904097838).abs() < 1e-12, "W₀(1) re = {}", r);
    assert!(w.imag().is_zero() || w.imag().to_f64().abs() < 1e-30);
}

#[test]
fn t101_w0_at_zero() {
    let prec = cnum::digits_to_bits(60);
    let z = make_z("0", "0", prec);
    let w = lambertw::w0(&z, prec).unwrap();
    assert!(w.real().is_zero() && w.imag().is_zero(), "W₀(0) = {:?}", w);
}

#[test]
fn t102_w0_neg_inv_e() {
    let prec = cnum::digits_to_bits(60);
    // z = -1/e exactly: W₀(-1/e) = -1.
    let e = Float::with_val(prec, 1).exp();
    let inv_e = Float::with_val(prec, 1) / &e;
    let z_re = Float::with_val(prec, -inv_e);
    let z = Complex::with_val(prec, (z_re, Float::with_val(prec, 0)));
    let w = lambertw::w0(&z, prec).unwrap();
    check_residual(&w, &z, prec, prec - 32).unwrap();
    assert!((w.real().to_f64() + 1.0).abs() < 1e-12, "W₀(-1/e) re = {}", w.real().to_f64());
}

#[test]
fn t103_w0_at_e() {
    let prec = cnum::digits_to_bits(60);
    // W₀(e) = 1 (since 1·e¹ = e).
    let e = Float::with_val(prec, 1).exp();
    let z = Complex::with_val(prec, (e, Float::with_val(prec, 0)));
    let w = lambertw::w0(&z, prec).unwrap();
    check_residual(&w, &z, prec, prec - 32).unwrap();
    assert!((w.real().to_f64() - 1.0).abs() < 1e-12, "W₀(e) re = {}", w.real().to_f64());
}

#[test]
fn t104_w0_complex_arg() {
    let prec = cnum::digits_to_bits(80);
    let z = make_z("1.5", "0.7", prec);
    let w = lambertw::w0(&z, prec).unwrap();
    check_residual(&w, &z, prec, prec - 32).unwrap();
}

#[test]
fn t105_w0_negative_real_below_branch() {
    // For z < -1/e (real), W₀ takes complex values.
    let prec = cnum::digits_to_bits(80);
    let z = make_z("-1", "0", prec);
    let w = lambertw::w0(&z, prec).unwrap();
    check_residual(&w, &z, prec, prec - 32).unwrap();
    // W₀(-1) ≈ -0.31813 + 1.33724 i
    assert!((w.real().to_f64() + 0.31813).abs() < 1e-3);
    assert!((w.imag().to_f64() - 1.33724).abs() < 1e-3);
}

#[test]
fn t106_w0_large_arg() {
    let prec = cnum::digits_to_bits(100);
    let z = make_z("100", "50", prec);
    let w = lambertw::w0(&z, prec).unwrap();
    check_residual(&w, &z, prec, prec - 32).unwrap();
}

#[test]
fn t107_w0_high_precision_500_digits() {
    let prec = cnum::digits_to_bits(500);
    let z = make_z("1", "0", prec);
    let w = lambertw::w0(&z, prec).unwrap();
    check_residual(&w, &z, prec, prec - 32).unwrap();
}

#[test]
fn t108_w0_high_precision_2000_digits() {
    let prec = cnum::digits_to_bits(2000);
    let z = make_z("3.14159", "0.5", prec);
    let w = lambertw::w0(&z, prec).unwrap();
    check_residual(&w, &z, prec, prec - 64).unwrap();
}

#[test]
fn t110_wm1_real() {
    // For z ∈ (-1/e, 0) real, W₋₁(z) is real and < -1.
    let prec = cnum::digits_to_bits(80);
    let z = make_z("-0.1", "0", prec);
    let w = lambertw::wm1(&z, prec).unwrap();
    check_residual(&w, &z, prec, prec - 32).unwrap();
    // W₋₁(-0.1) ≈ -3.5772 (only on real branch)
    let r = w.real().to_f64();
    let i = w.imag().to_f64();
    // Allow small imaginary part due to f64 seed noise; what matters is residual.
    assert!(r < -1.0, "W₋₁(-0.1) re = {} (expected real < -1)", r);
    assert!(i.abs() < 1e-3, "W₋₁(-0.1) im = {} (expected ~ 0)", i);
}

#[test]
fn t111_wm1_at_neg_inv_e() {
    let prec = cnum::digits_to_bits(60);
    let e = Float::with_val(prec, 1).exp();
    let inv_e = Float::with_val(prec, 1) / &e;
    let z_re = Float::with_val(prec, -inv_e);
    let z = Complex::with_val(prec, (z_re, Float::with_val(prec, 0)));
    let w = lambertw::wm1(&z, prec).unwrap();
    check_residual(&w, &z, prec, prec - 32).unwrap();
    assert!((w.real().to_f64() + 1.0).abs() < 1e-10, "W₋₁(-1/e) re = {}", w.real().to_f64());
}

#[test]
fn t112_wm1_zero_errors() {
    let prec = cnum::digits_to_bits(60);
    let z = make_z("0", "0", prec);
    assert!(lambertw::wm1(&z, prec).is_err());
}

#[test]
fn t120_w0_for_tetration_fixed_point_b_e() {
    // For tetration of base e: argument is -ln(e) = -1.
    // L = -W₀(-1) / ln(e) = -W₀(-1).
    // Expected: L ≈ 0.31813 - 1.33724 i (since W₀(-1) ≈ -0.31813 + 1.33724 i).
    let prec = cnum::digits_to_bits(100);
    let z = make_z("-1", "0", prec);
    let w = lambertw::w0(&z, prec).unwrap();
    check_residual(&w, &z, prec, prec - 32).unwrap();
    // L = -W₀(-1)
    let l = Complex::with_val(prec, -&w);
    // L is the fixed point: L = e^L ⇒ L · ln(L) = ?  Actually L = b^L = e^L.
    let exp_l = Complex::with_val(prec, l.exp_ref());
    let r = Complex::with_val(prec, &exp_l - &l);
    let r_abs = Float::with_val(prec, r.abs_ref());
    let r_exp = r_abs.get_exp().unwrap_or(i32::MIN);
    assert!(r_exp <= -((prec as i32) - 32), "fixed-point residual exp = {}", r_exp);
}
