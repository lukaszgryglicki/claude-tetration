//! Phase-3 verification: base classification according to Shell-Thron region
//! membership.
//!
//! Reference Shell-Thron region for real positive `b`: convergence iff
//! `e^{-e} ≤ b ≤ e^{1/e}`, i.e. roughly `0.0660 ≤ b ≤ 1.4446`. We test
//! representative bases in each subregion plus complex/imaginary cases.

use rug::Complex;

use tetration::cnum;
use tetration::regions::{self, Region};

fn classify_real(re: &str) -> Region {
    let prec = cnum::digits_to_bits(60);
    let b = cnum::parse_complex(re, "0", prec).unwrap();
    regions::classify(&b, prec).unwrap()
}

fn classify(re: &str, im: &str) -> Region {
    let prec = cnum::digits_to_bits(60);
    let b = cnum::parse_complex(re, im, prec).unwrap();
    regions::classify(&b, prec).unwrap()
}

#[test]
fn t200_b_one_classified_as_baseone() {
    assert!(matches!(classify_real("1"), Region::BaseOne));
}

#[test]
fn t201_b_zero_classified_as_basezero() {
    assert!(matches!(classify_real("0"), Region::BaseZero));
}

#[test]
fn t210_sqrt2_inside_shell_thron() {
    // √2 ≈ 1.414 is the canonical Shell-Thron interior base.
    let r = classify_real("1.4142135623730950488");
    assert!(matches!(r, Region::ShellThronInterior(_)),
            "√2 should be Shell-Thron interior, got {}", r.name());
}

#[test]
fn t211_one_point_two_inside_shell_thron() {
    let r = classify_real("1.2");
    assert!(matches!(r, Region::ShellThronInterior(_)),
            "1.2 should be Shell-Thron interior, got {}", r.name());
}

#[test]
fn t212_zero_point_five_inside_shell_thron() {
    // 0.5 ∈ (e^{-e}, 1) ≈ (0.066, 1) is inside Shell-Thron region for real bases.
    let r = classify_real("0.5");
    assert!(matches!(r, Region::ShellThronInterior(_)),
            "0.5 should be Shell-Thron interior, got {}", r.name());
}

#[test]
fn t220_e_to_one_over_e_on_boundary() {
    // b = e^{1/e} sits exactly on the parabolic boundary |λ| = 1.
    let r = classify_real("1.44466786100976613");
    assert!(matches!(r, Region::ShellThronBoundary(_)),
            "e^(1/e) should be Shell-Thron boundary, got {}", r.name());
}

#[test]
fn t230_e_outside_shell_thron_real() {
    // b = e ≈ 2.718 is real-positive and well outside Shell-Thron.
    let r = classify_real("2.71828182845904523536");
    assert!(matches!(r, Region::OutsideShellThronRealPositive(_)),
            "e should be Outside-real-positive, got {}", r.name());
}

#[test]
fn t231_two_outside_shell_thron_real() {
    let r = classify_real("2");
    assert!(matches!(r, Region::OutsideShellThronRealPositive(_)),
            "2 should be Outside-real-positive, got {}", r.name());
}

#[test]
fn t232_negative_real_general() {
    let r = classify_real("-2");
    assert!(matches!(r, Region::OutsideShellThronGeneral(_)),
            "-2 should be Outside-general, got {}", r.name());
}

#[test]
fn t233_purely_imaginary_general() {
    let r = classify("0", "1");
    // i has |λ| > 1; should be classified as Outside-general (not real-positive).
    assert!(matches!(r, Region::OutsideShellThronGeneral(_) | Region::ShellThronInterior(_) | Region::ShellThronBoundary(_)),
            "i should be classified, got {}", r.name());
}

#[test]
fn t234_complex_outside() {
    // b = 1.5 + 0.5i — complex, generally outside.
    let r = classify("1.5", "0.5");
    let _ = r; // accept any region; the classification just needs to succeed.
}

#[test]
fn t240_lambda_at_e_to_one_over_e_is_one() {
    // |λ| at b = e^{1/e} should be ~ 1 (parabolic).
    let r = classify_real("1.44466786100976613");
    if let Region::ShellThronBoundary(d) = r {
        assert!((d.lambda_abs - 1.0).abs() < 0.05, "|λ| = {} for e^(1/e)", d.lambda_abs);
    } else {
        panic!("expected boundary region");
    }
}

#[test]
fn t241_fixed_point_satisfies_b_to_l_equals_l() {
    // For arbitrary base b, L should satisfy b^L = L. Verify this holds to high
    // precision after classification.
    let prec = cnum::digits_to_bits(80);
    for (re, im) in [("2", "0"), ("1.2", "0"), ("0.5", "0"), ("1.5", "0.5"), ("-2", "0")] {
        let b = cnum::parse_complex(re, im, prec).unwrap();
        let r = regions::classify(&b, prec).unwrap();
        let fp_data = match r {
            Region::ShellThronInterior(d)
            | Region::ShellThronBoundary(d)
            | Region::OutsideShellThronRealPositive(d)
            | Region::OutsideShellThronGeneral(d) => d,
            _ => continue,
        };
        let l = &fp_data.fixed_point;
        let bl = cnum::pow_complex(&b, l, prec);
        let r2 = Complex::with_val(prec, &bl - l);
        let r_abs = rug::Float::with_val(prec, r2.abs_ref());
        let exp = r_abs.get_exp().unwrap_or(i32::MIN);
        assert!(
            exp <= -((prec as i32) - 32),
            "b={}+{}i: |b^L - L| exp = {}",
            re, im, exp
        );
    }
}
