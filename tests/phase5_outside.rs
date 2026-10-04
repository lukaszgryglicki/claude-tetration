//! Phase-5 verification: tetration for bases outside the Shell-Thron region
//! (real positive `> e^(1/e)` and general complex).
//!
//! Covers existing regular/Kouznetsov branches, real-boundary fallback,
//! large-base range handling and specific honest refusals. Functional-equation
//! agreement is a consistency check, not independent accuracy certification.

use rug::{Complex, Float};

use tetration::{cnum, dispatch, kouznetsov, regions};

fn parse(re: &str, im: &str, prec: u64) -> Complex {
    cnum::parse_complex(re, im, prec).unwrap()
}

fn matching_digits(a: &Complex, b: &Complex, prec: u64) -> Float {
    assert!(cnum::is_finite(a) && cnum::is_finite(b));
    let diff = Complex::with_val_64(prec, a - b);
    let da = Float::with_val_64(prec, diff.abs_ref());
    if da.is_zero() {
        return Float::with_val_64(prec, rug::float::Special::Infinity);
    }
    -da.log10()
}

fn check_functional_eq(
    b_re: &str,
    b_im: &str,
    z_re: &str,
    z_im: &str,
    digits: u64,
    expected_match: u64,
) {
    let prec = cnum::digits_to_bits(digits);
    let b = parse(b_re, b_im, prec);
    let z = parse(z_re, z_im, prec);
    let one = parse("1", "0", prec);
    let z_plus_1 = Complex::with_val_64(prec, &z + &one);

    let fz = dispatch::tetrate(&b, &z, prec, digits).unwrap();
    let fz1 = dispatch::tetrate(&b, &z_plus_1, prec, digits).unwrap();
    let b_to_fz = cnum::pow_complex(&b, &fz, prec);

    let m = matching_digits(&fz1, &b_to_fz, prec);
    assert!(
        m >= expected_match,
        "b={}+{}i z={}+{}i: matched {} digits (expected ≥ {})",
        b_re,
        b_im,
        z_re,
        z_im,
        m,
        expected_match,
    );
}

#[allow(dead_code)]
fn check_unsupported(b_re: &str, b_im: &str, z_re: &str, z_im: &str, digits: u64) {
    let prec = cnum::digits_to_bits(digits);
    let b = parse(b_re, b_im, prec);
    let z = parse(z_re, z_im, prec);
    let result = dispatch::tetrate(&b, &z, prec, digits);
    let err = result.expect_err(&format!(
        "expected dispatch to error out for b={}+{}i z={}+{}i (no algorithm available); \
         instead got Ok",
        b_re, b_im, z_re, z_im
    ));
    assert!(
        err.contains("unsupported case"),
        "expected an 'unsupported case' error, got: {}",
        err
    );
}

// ---------- Schröder-applicable cases (close to Shell-Thron boundary) ----

#[test]
fn t410_complex_base_just_outside() {
    // b = 1.5 + 0.5i — typically near boundary, Schröder converges.
    check_functional_eq("1.5", "0.5", "0.5", "0.0", 50, 25);
}

#[test]
fn t411_imaginary_base_modest() {
    // b = 0.3i: complex but not extreme.
    check_functional_eq("0", "0.3", "0.5", "0", 40, 20);
}

// ---------- Real bases > e^(1/e) (Newton-Kantorovich Kouznetsov) ----
// These bases used to error out (Schröder doesn't reach 1−L from L). They now
// route to Newton-Kantorovich Cauchy iteration. The current discretization
// floor at modest digit counts is around 1e−4 on the boundary residual; the
// functional equation residual via the converged Cauchy interpolant is
// typically a few digits better than that, so we ask for ≥ 3 matched digits.

#[test]
fn t420_real_e_via_kouznetsov() {
    check_functional_eq("2.71828182845904523536", "0", "0.5", "0", 10, 3);
}

#[test]
fn t421_real_two_via_kouznetsov() {
    check_functional_eq("2", "0", "0.5", "0", 10, 3);
}

#[test]
fn t422_real_ten_via_kouznetsov() {
    check_functional_eq("10", "0", "0.3", "0", 10, 3);
}

// ---------- Slightly-complex bases (Newton-from-conjugate fixed-point) ----
// For b just off the real positive axis, the Kouznetsov rectangle still works
// once Schwarz symmetry is dropped and the partner fixed point is found by
// Newton iteration starting from `conj(L_+)`. This avoids the W₋₁ branch-cut
// jump that breaks the naive `-W₋₁(-ln b)/ln b` partner choice.

#[test]
fn t425_slightly_complex_base_kouznetsov() {
    // b = 2 + 0.001i: barely off the real axis, should converge cleanly.
    check_functional_eq("2", "0.001", "0.5", "0", 10, 3);
}

#[test]
fn t426_moderately_complex_base_kouznetsov() {
    // b = 2 + 0.1i: 10% imaginary part. Still on the natural-pair side.
    check_functional_eq("2", "0.1", "0.5", "0", 10, 3);
}

#[test]
fn t427_slightly_complex_base_complex_height() {
    // Complex base + complex height. Cross-validates that eval_at_height on
    // the converged interpolant satisfies F(h+1) = b^F(h) when h itself
    // is off the real axis.
    check_functional_eq("2", "0.1", "0.5", "0.5", 10, 3);
}

#[test]
fn t428_boundary_band_real_via_kouznetsov() {
    // b=1.5 sits in the parabolic boundary band (|λ|≈1.033) where Schröder is
    // unreliable but Newton-Kantorovich Kouznetsov still converges because
    // |arg(λ)| is far from 0. Routes through ShellThronBoundary → Kouznetsov.
    let digits = 10;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("1.5", "0", prec);
    let h = parse("0.5", "0", prec);
    let regions::Region::ShellThronBoundary(fp) = regions::classify(&b, prec).unwrap() else {
        panic!("expected boundary band");
    };
    let state = kouznetsov::setup_kouznetsov(&b, &fp, prec, digits)
        .expect("existing direct solver must converge at b=1.5");
    let direct = kouznetsov::eval_kouznetsov(&state, &b, &h).unwrap();
    let dispatched = dispatch::tetrate(&b, &h, prec, digits)
        .expect("a failed continuation must not skip a converging direct method");
    assert!(matching_digits(&dispatched, &direct, prec) >= digits);
    let next =
        kouznetsov::eval_kouznetsov(&state, &b, &Complex::with_val_64(prec, &h + 1)).unwrap();
    assert!(matching_digits(&next, &cnum::pow_complex(&b, &dispatched, prec), prec) >= digits);
}

// ---------- Cases that have no working algorithm (must error out) ----

#[test]
fn t423_negative_real_via_wk_search() {
    let digits = 10;
    let prec = cnum::digits_to_bits(digits);
    let error = dispatch::tetrate(
        &parse("-2", "0", prec),
        &parse("0.4", "0.1", prec),
        prec,
        digits,
    )
    .unwrap_err();
    assert!(
        error.contains("unsupported case") && error.contains("boundary residual"),
        "{error}"
    );
}

#[test]
fn t424_imaginary_supported_via_sigma_shift() {
    // b = i has |λ| ≈ 0.89, so it's actually inside Shell-Thron — but |1−L|
    // is outside the σ̃ Taylor disk at L. The σ̃-shift mechanism rescues
    // this case via the functional equation. Verify F(z+1) ≈ b^F(z).
    check_functional_eq("0", "1", "0.5", "0", 30, 20);
}

// ---------- Large real bases (regression for the smooth target_mid cap) ----
// Bases b ∈ {50, 100, 200, 500, 1000} stress the `target_mid` initial-guess
// formula in `kouznetsov.rs::initial_guess_with_target` (≈ lines 986-1011):
// the cap `clamp(1.5 − 0.1·max(0, ln|b|−2), 0.7, 1.5)` keeps the Levenberg-
// Marquardt iterate inside the correct basin of attraction for the Kneser
// fixed-point. Without the cap, `target_mid = √b` lands far from the true
// F̃[mid] for large bases and the LM leaks into the wrong basin
// (F̃[mid] → 0). Verified results match the table in FAILURE_CASES.md §E.
//
// Probed at 10 digits (~7s per case) so all three tests stay default-runnable.

#[test]
fn t440_large_base_b100_functional_eq() {
    // F_100(0.5) ≈ 4.213104547 per FAILURE_CASES §E
    check_functional_eq("100", "0", "0.5", "0", 10, 3);
}

#[test]
fn t441_large_base_b1000_functional_eq() {
    // F_1000(0.5) ≈ 6.391336395 per FAILURE_CASES §E. Largest base in the
    // verified table; defends the cap-clamp at the high end (cap=0.7).
    check_functional_eq("1000", "0", "0.5", "0", 10, 3);
}

#[test]
fn t442_large_base_b50_value_check() {
    // F_50(0.5) ≈ 3.6480… per FAILURE_CASES §E. Spot-check that the result
    // is in the documented range (within 1e-3 of expected).
    let digits = 10u64;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("50", "0", prec);
    let h = parse("0.5", "0", prec);
    let f = dispatch::tetrate(&b, &h, prec, digits).unwrap();
    let f_re = f.real();
    assert!(
        Float::with_val_64(prec, f_re - cnum::decimal("3.6480", prec)).abs()
            < cnum::epsilon(3, prec),
        "F_50(0.5) = {} but expected ≈ 3.6480 (FAILURE_CASES §E)",
        f_re
    );
}

// ---------- Integer heights still work exactly ----
// Even when no continuous algorithm is available, integer heights bypass the
// region-based dispatch and use direct iteration.

#[test]
fn t430_integer_endpoint_exact() {
    let digits = 30;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("2.71828182845904523536", "0", prec);
    let h_int = parse("1", "0", prec);
    let f = dispatch::tetrate(&b, &h_int, prec, digits).unwrap();
    let diff = Complex::with_val_64(prec, &f - &b);
    let da = Float::with_val_64(prec, diff.abs_ref());
    assert!(da < cnum::epsilon(25, prec), "F_e(1) − e differs by {}", da);
}
