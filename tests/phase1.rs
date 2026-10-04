//! Phase-1 verification: integer heights, special bases, basic CLI argument
//! handling.
//!
//! Tests run via the public `tetrate_str` API so they exercise the same path
//! the CLI uses.

use rug::{float::Constant, Complex, Float};
use tetration::{cnum, tetrate_str};

/// Convenience: run `tetrate_str` with `(prec, br, bi, hr, hi)`, panicking on
/// error and returning `(re, im)`.
fn tet(prec: &str, br: &str, bi: &str, hr: &str, hi: &str) -> (String, String) {
    tetrate_str(prec, br, bi, hr, hi).expect("tetrate_str should succeed")
}

fn assert_number(actual: &str, expected: &str, digits: u64) {
    let prec = cnum::digits_to_bits(digits);
    let actual = cnum::parse_float(actual, prec).unwrap();
    let expected = cnum::parse_float(expected, prec).unwrap();
    let error = Float::with_val(prec, &actual - &expected).abs();
    assert!(
        error < cnum::epsilon(digits, prec),
        "{actual} differs from {expected} by {error}"
    );
}

// --------------------------------------------------------------------------
// b^^0 = 1 for various bases
// --------------------------------------------------------------------------

#[test]
fn t000_height_zero_is_one() {
    for (br, bi) in [
        ("2", "0"),
        ("3.14159", "0"),
        ("1.5", "0.5"),
        ("0.5", "0"),
        ("-1", "0"),
        ("0.5", "-0.7"),
    ] {
        let (re, im) = tet("30", br, bi, "0", "0");
        assert_number(&re, "1", 20);
        assert_eq!(
            im.trim_start_matches('-'),
            "0",
            "F_b(0) im={} for b={}+{}i",
            im,
            br,
            bi
        );
    }
}

// --------------------------------------------------------------------------
// b^^1 = b
// --------------------------------------------------------------------------

#[test]
fn t010_height_one_is_b() {
    for (br, bi) in [
        ("2", "0"),
        ("3.14159265358979323846", "0"),
        ("1.5", "0.5"),
        ("-2", "0.7"),
    ] {
        let (re, im) = tet("30", br, bi, "1", "0");
        assert_number(&re, br, 20);
        assert_number(&im, bi, 20);
    }
}

// --------------------------------------------------------------------------
// b^^2 = b^b ; verify by comparing to the direct computation 2^^2=4, 3^^2=27, etc.
// --------------------------------------------------------------------------

#[test]
fn t020_two_tetrated_two_is_four() {
    let (re, im) = tet("50", "2", "0", "2", "0");
    assert_number(&re, "4", 40);
    assert_eq!(im.trim_start_matches('-'), "0", "im = {}", im);
}

#[test]
fn t021_three_tetrated_two_is_twentyseven() {
    let (re, im) = tet("50", "3", "0", "2", "0");
    assert_number(&re, "27", 30);
    assert_eq!(im.trim_start_matches('-'), "0", "im = {}", im);
}

#[test]
fn t022_two_tetrated_three_is_sixteen() {
    // 2^^3 = 2^(2^2) = 2^4 = 16
    let (re, im) = tet("50", "2", "0", "3", "0");
    assert_number(&re, "16", 30);
    assert_eq!(im.trim_start_matches('-'), "0", "im = {}", im);
}

#[test]
fn t023_two_tetrated_four_is_65536() {
    // 2^^4 = 2^(2^^3) = 2^16 = 65536
    let (re, im) = tet("50", "2", "0", "4", "0");
    assert_number(&re, "65536", 20);
    assert_eq!(im.trim_start_matches('-'), "0", "im = {}", im);
}

// --------------------------------------------------------------------------
// b = 1: always 1
// --------------------------------------------------------------------------

#[test]
fn t030_base_one() {
    for h in ["0", "1", "5", "-1", "0.5", "100"] {
        let (re, im) = tet("30", "1", "0", h, "0");
        assert_number(&re, "1", 20);
        assert_eq!(im.trim_start_matches('-'), "0", "1^^{} im = {}", h, im);
    }
}

// --------------------------------------------------------------------------
// b = 0: alternating (0^^0=1, 0^^1=0, 0^^2=1, 0^^3=0, ...)
// --------------------------------------------------------------------------

#[test]
fn t040_base_zero_alternation() {
    for (h, expected) in [("0", "1"), ("1", "0"), ("2", "1"), ("3", "0"), ("4", "1")] {
        let (re, _im) = tet("30", "0", "0", h, "0");
        assert_number(&re, expected, 20);
    }
}

#[test]
fn t041_base_zero_non_integer_errors() {
    let result = tetrate_str("30", "0", "0", "0.5", "0");
    assert!(result.is_err(), "0^^0.5 should error, got {:?}", result);
}

#[test]
fn t042_base_zero_negative_integer_errors() {
    let result = tetrate_str("30", "0", "0", "-1", "0");
    assert!(result.is_err(), "0^^(-1) should error, got {:?}", result);
}

// --------------------------------------------------------------------------
// Negative integer heights: F(-1) = 0, F(-2) errors
// --------------------------------------------------------------------------

#[test]
fn t050_neg_one_height() {
    let (re, _im) = tet("30", "2", "0", "-1", "0");
    assert_number(&re, "0", 20);
}

#[test]
fn t051_neg_two_height_errors() {
    let result = tetrate_str("30", "2", "0", "-2", "0");
    assert!(result.is_err(), "2^^(-2) should error, got {:?}", result);
}

// --------------------------------------------------------------------------
// Complex base, integer height: works via direct iteration
// --------------------------------------------------------------------------

#[test]
fn t060_complex_base_integer_height() {
    let digits = 50;
    let prec = cnum::digits_to_bits(digits);
    let (re, im) = tet("50", "1", "1", "2", "0");
    let actual = cnum::parse_complex(&re, &im, prec).unwrap();
    let half_ln2 = Float::with_val(prec, 2).ln() / 2;
    let quarter_pi = Float::with_val(prec, Constant::Pi) / 4;
    let radius = Float::with_val(prec, &half_ln2 - &quarter_pi).exp();
    let angle = Float::with_val(prec, &half_ln2 + &quarter_pi);
    let expected = Complex::with_val(
        prec,
        (
            Float::with_val(prec, angle.cos_ref()) * &radius,
            Float::with_val(prec, angle.sin_ref()) * &radius,
        ),
    );
    assert!(
        cnum::abs(&Complex::with_val(prec, actual - expected), prec) < cnum::epsilon(digits, prec)
    );
}

// --------------------------------------------------------------------------
// Schröder continuity: F(n + ε) → F(n) as ε → 0 (Shell-Thron interior bases
// where the regular-iteration algorithm runs).
// --------------------------------------------------------------------------

#[test]
fn t070_schroder_continuity_near_zero() {
    // F_{√2}(1e-10) is within ~1e-9 of F_{√2}(0)=1 by continuity.
    let (re, _im) = tet("30", "1.4142135623730950488", "0", "1e-10", "0");
    assert_number(&re, "1", 8);
}

#[test]
fn t071_schroder_continuity_near_one() {
    // F_{√2}(1 − 1e-10) → √2 by continuity. F_{√2}(1) = √2 ≈ 1.4142135623…
    let (re, _im) = tet("30", "1.4142135623730950488", "0", "0.9999999999", "0");
    assert_number(&re, "1.4142135623730950488", 8);
}

// --------------------------------------------------------------------------
// High-precision sanity: F_b(0) should be exactly 1 with as many digits as asked
// --------------------------------------------------------------------------

#[test]
fn t080_high_precision_zero_height() {
    let (re, im) = tet("1000", "2.7182818284", "0.0", "0", "0");
    // re should be "1.000…0e0" (1000 chars after the leading 1).
    assert!(
        re.starts_with("1.0"),
        "high-precision 1 starts with {}",
        &re[..30.min(re.len())]
    );
    assert_eq!(
        re.split(['e', 'E'])
            .next()
            .unwrap()
            .chars()
            .filter(|c| c.is_ascii_digit())
            .count(),
        1000
    );
    assert_number(&re, "1", 1000);
    assert_eq!(im.trim_start_matches('-'), "0");
}

// --------------------------------------------------------------------------
// CLI argument errors
// --------------------------------------------------------------------------

#[test]
fn t090_bad_precision_errors() {
    assert!(tetrate_str("0", "2", "0", "1", "0").is_err());
    assert!(tetrate_str("abc", "2", "0", "1", "0").is_err());
    assert!(tetrate_str("-5", "2", "0", "1", "0").is_err());
}

#[test]
fn t091_bad_number_errors() {
    assert!(tetrate_str("30", "not_a_number", "0", "1", "0").is_err());
    assert!(tetrate_str("30", "2", "xyz", "1", "0").is_err());
}
