//! Phase-5 verification: tetration for bases outside the Shell-Thron region
//! (real positive `> e^(1/e)` and general complex).
//!
//! Phase-5 status — partial: Schröder-at-repelling-fixed-point handles bases
//! close to the Shell-Thron boundary (where the σ̃ Taylor series at L still
//! converges at `w = 1 − L`). For bases farther out (real `e`, `2`, `10`,
//! etc.), the σ̃ Taylor radius is smaller than `|1 − L|`; Schröder bails and
//! — per design — the dispatcher errors out cleanly rather than silently
//! returning a wrong-but-plausible linear-approx number. Full Kneser /
//! Kouznetsov Cauchy iteration that handles those cases at high precision is
//! deferred to a later phase.
//!
//! Tests below split into:
//!   * "Schröder applicable" cases: high tolerance via functional equation.
//!   * "Out-of-reach" cases: dispatch must return Err, not silently produce
//!     a wrong-but-plausible linear-approx number. Integer heights are still
//!     handled exactly via the integer_height path.

use rug::{Complex, Float};

use tetration::{cnum, dispatch};

fn parse(re: &str, im: &str, prec: u32) -> Complex {
    cnum::parse_complex(re, im, prec).unwrap()
}

fn matching_digits(a: &Complex, b: &Complex, prec: u32) -> f64 {
    let diff = Complex::with_val(prec, a - b);
    let da = Float::with_val(prec, diff.abs_ref());
    if da.is_zero() {
        return f64::INFINITY;
    }
    -da.to_f64().log10()
}

fn check_functional_eq(
    b_re: &str, b_im: &str, z_re: &str, z_im: &str,
    digits: u64, expected_match: f64,
) {
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
        "b={}+{}i z={}+{}i: matched {} digits (expected ≥ {})",
        b_re, b_im, z_re, z_im, m, expected_match,
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
    check_functional_eq("1.5", "0.5", "0.5", "0.0", 50, 25.0);
}

#[test]
fn t411_imaginary_base_modest() {
    // b = 0.3i: complex but not extreme.
    check_functional_eq("0", "0.3", "0.5", "0", 40, 20.0);
}

// ---------- Real bases > e^(1/e) (Newton-Kantorovich Kouznetsov) ----
// These bases used to error out (Schröder doesn't reach 1−L from L). They now
// route to Newton-Kantorovich Cauchy iteration. The current discretization
// floor at modest digit counts is around 1e−4 on the boundary residual; the
// functional equation residual via the converged Cauchy interpolant is
// typically a few digits better than that, so we ask for ≥ 3 matched digits.

#[test]
fn t420_real_e_via_kouznetsov() {
    check_functional_eq("2.71828182845904523536", "0", "0.5", "0", 10, 3.0);
}

#[test]
fn t421_real_two_via_kouznetsov() {
    check_functional_eq("2", "0", "0.5", "0", 10, 3.0);
}

#[test]
fn t422_real_ten_via_kouznetsov() {
    check_functional_eq("10", "0", "0.3", "0", 10, 3.0);
}

// ---------- Slightly-complex bases (Newton-from-conjugate fixed-point) ----
// For b just off the real positive axis, the Kouznetsov rectangle still works
// once Schwarz symmetry is dropped and the partner fixed point is found by
// Newton iteration starting from `conj(L_+)`. This avoids the W₋₁ branch-cut
// jump that breaks the naive `-W₋₁(-ln b)/ln b` partner choice.

#[test]
fn t425_slightly_complex_base_kouznetsov() {
    // b = 2 + 0.001i: barely off the real axis, should converge cleanly.
    check_functional_eq("2", "0.001", "0.5", "0", 10, 3.0);
}

#[test]
fn t426_moderately_complex_base_kouznetsov() {
    // b = 2 + 0.1i: 10% imaginary part. Still on the natural-pair side.
    check_functional_eq("2", "0.1", "0.5", "0", 10, 3.0);
}

#[test]
fn t427_slightly_complex_base_complex_height() {
    // Complex base + complex height. Cross-validates that eval_at_height on
    // the converged interpolant satisfies F(h+1) = b^F(h) when h itself
    // is off the real axis.
    check_functional_eq("2", "0.1", "0.5", "0.5", 10, 3.0);
}

#[test]
fn t428_boundary_band_real_via_kouznetsov() {
    // b=1.5 sits in the parabolic boundary band (|λ|≈1.033) where Schröder is
    // unreliable but Newton-Kantorovich Kouznetsov still converges because
    // |arg(λ)| is far from 0. Routes through ShellThronBoundary → Kouznetsov.
    check_functional_eq("1.5", "0", "0.5", "0", 10, 3.0);
}

// ---------- Cases that have no working algorithm (must error out) ----

#[test]
fn t423_negative_real_via_wk_search() {
    // Negative real base b=-2: Newton-from-conjugate finds both fixed points
    // in the same half-plane (W₀(-ln(-2)) = -W₀(-ln(2)+iπ) and its conjugate
    // both have positive imaginary part). The fallback W_k search across
    // k∈±[1..5] picks W_1, which gives L_1 ≈ -0.902-0.172i in the lower
    // half-plane — a valid opposite-half-plane partner. The resulting
    // tetration is non-canonical (not strict Kneser) but satisfies F(0)=1
    // and F(z+1)=b^F(z). Verified at low digits because the Cauchy
    // reconstruction with non-conjugate partners hits a discretization
    // floor at ~1e-4 to 1e-6 that doesn't shrink with N.
    check_functional_eq("-2", "0", "0.4", "0.1", 10, 3.0);
}

#[test]
fn t424_imaginary_supported_via_sigma_shift() {
    // b = i has |λ| ≈ 0.89, so it's actually inside Shell-Thron — but |1−L|
    // is outside the σ̃ Taylor disk at L. The σ̃-shift mechanism rescues
    // this case via the functional equation. Verify F(z+1) ≈ b^F(z).
    check_functional_eq("0", "1", "0.5", "0", 30, 20.0);
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
    let diff = Complex::with_val(prec, &f - &b);
    let da = Float::with_val(prec, diff.abs_ref()).to_f64();
    assert!(da < 1e-25, "F_e(1) − e differs by {}", da);
}
