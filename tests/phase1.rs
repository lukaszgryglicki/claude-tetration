//! Phase-1 verification: integer heights, special bases, basic CLI argument
//! handling.
//!
//! Tests run via the public `tetrate_str` API so they exercise the same path
//! the CLI uses.

use tetration::tetrate_str;

/// Convenience: run `tetrate_str` with `(prec, br, bi, hr, hi)`, panicking on
/// error and returning `(re, im)`.
fn tet(prec: &str, br: &str, bi: &str, hr: &str, hi: &str) -> (String, String) {
    tetrate_str(prec, br, bi, hr, hi).expect("tetrate_str should succeed")
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
        let r: f64 = re.parse().unwrap();
        assert!((r - 1.0).abs() < 1e-20, "F_b(0) re={} for b={}+{}i", re, br, bi);
        assert_eq!(im.trim_start_matches('-'), "0", "F_b(0) im={} for b={}+{}i", im, br, bi);
    }
}

// --------------------------------------------------------------------------
// b^^1 = b
// --------------------------------------------------------------------------

#[test]
fn t010_height_one_is_b() {
    for (br, bi) in [("2", "0"), ("3.14159265358979323846", "0"), ("1.5", "0.5"), ("-2", "0.7")] {
        let (re, im) = tet("30", br, bi, "1", "0");
        // Compare to b directly by re-running with height 0 and base = b — i.e. just check
        // that re/im strings round-trip via reformat at lower precision.
        let (re_b, im_b) = tet("30", br, bi, "0", "0"); // gives 1, not b. Use the original strings instead.
        let _ = (re_b, im_b);
        // Direct compare: parse re/im as f64 and check approximate equality with the input.
        let r_got: f64 = re.parse().unwrap();
        let i_got: f64 = im.parse().unwrap();
        let r_exp: f64 = br.parse().unwrap();
        let i_exp: f64 = bi.parse().unwrap();
        assert!((r_got - r_exp).abs() < 1e-20, "re={} expected {} for b={}+{}i", re, br, br, bi);
        assert!((i_got - i_exp).abs() < 1e-20, "im={} expected {} for b={}+{}i", im, bi, br, bi);
    }
}

// --------------------------------------------------------------------------
// b^^2 = b^b ; verify by comparing to the direct computation 2^^2=4, 3^^2=27, etc.
// --------------------------------------------------------------------------

#[test]
fn t020_two_tetrated_two_is_four() {
    let (re, im) = tet("50", "2", "0", "2", "0");
    let r: f64 = re.parse().unwrap();
    assert!((r - 4.0).abs() < 1e-40, "2^^2 = {}", re);
    assert_eq!(im.trim_start_matches('-'), "0", "im = {}", im);
}

#[test]
fn t021_three_tetrated_two_is_twentyseven() {
    let (re, im) = tet("50", "3", "0", "2", "0");
    let r: f64 = re.parse().unwrap();
    assert!((r - 27.0).abs() < 1e-30, "3^^2 = {}", re);
    assert_eq!(im.trim_start_matches('-'), "0", "im = {}", im);
}

#[test]
fn t022_two_tetrated_three_is_sixteen() {
    // 2^^3 = 2^(2^2) = 2^4 = 16
    let (re, im) = tet("50", "2", "0", "3", "0");
    let r: f64 = re.parse().unwrap();
    assert!((r - 16.0).abs() < 1e-30, "2^^3 = {}", re);
    assert_eq!(im.trim_start_matches('-'), "0", "im = {}", im);
}

#[test]
fn t023_two_tetrated_four_is_65536() {
    // 2^^4 = 2^(2^^3) = 2^16 = 65536
    let (re, im) = tet("50", "2", "0", "4", "0");
    let r: f64 = re.parse().unwrap();
    assert!((r - 65536.0).abs() < 1e-20, "2^^4 = {}", re);
    assert_eq!(im.trim_start_matches('-'), "0", "im = {}", im);
}

// --------------------------------------------------------------------------
// b = 1: always 1
// --------------------------------------------------------------------------

#[test]
fn t030_base_one() {
    for h in ["0", "1", "5", "-1", "0.5", "100"] {
        let (re, im) = tet("30", "1", "0", h, "0");
        let r: f64 = re.parse().unwrap();
        assert!((r - 1.0).abs() < 1e-20, "1^^{} re = {}", h, re);
        assert_eq!(im.trim_start_matches('-'), "0", "1^^{} im = {}", h, im);
    }
}

// --------------------------------------------------------------------------
// b = 0: alternating (0^^0=1, 0^^1=0, 0^^2=1, 0^^3=0, ...)
// --------------------------------------------------------------------------

#[test]
fn t040_base_zero_alternation() {
    for (h, expected) in [("0", 1.0), ("1", 0.0), ("2", 1.0), ("3", 0.0), ("4", 1.0)] {
        let (re, _im) = tet("30", "0", "0", h, "0");
        let r: f64 = re.parse().unwrap_or(0.0);
        assert!((r - expected).abs() < 1e-20, "0^^{} = {}, expected {}", h, re, expected);
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
    let r: f64 = re.parse().unwrap_or(99.0);
    assert!(r.abs() < 1e-20, "2^^(-1) re = {}", re);
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
    // (1+i)^^2 = (1+i)^(1+i) = exp((1+i)*ln(1+i)) where ln(1+i) = ln(√2)+iπ/4
    // = exp((1+i)*(0.5 ln 2 + i π/4)) = exp(0.5 ln 2 - π/4 + i*(0.5 ln 2 + π/4))
    // Real part ≈ exp(0.5*0.693 - π/4) cos(0.5*0.693 + π/4) = exp(-0.4388) cos(1.132)
    //          ≈ 0.6450 * 0.4271 ≈ 0.2755
    let (re, im) = tet("30", "1", "1", "2", "0");
    let r: f64 = re.parse().unwrap();
    let i: f64 = im.parse().unwrap();
    assert!((r - 0.2745784).abs() < 1e-3, "Re((1+i)^^2) = {}", re);
    assert!((i - 0.5837571).abs() < 1e-3, "Im((1+i)^^2) = {}", im);
}

// --------------------------------------------------------------------------
// Schröder continuity: F(n + ε) → F(n) as ε → 0 (Shell-Thron interior bases
// where the actual algorithm runs — base 2 is now unsupported by design).
// --------------------------------------------------------------------------

#[test]
fn t070_schroder_continuity_near_zero() {
    // F_{√2}(1e-10) is within ~1e-9 of F_{√2}(0)=1 by continuity.
    let (re, _im) = tet("30", "1.4142135623730950488", "0", "1e-10", "0");
    let r: f64 = re.parse().unwrap();
    assert!((r - 1.0).abs() < 1e-8, "F_{{√2}}(1e-10) ≈ {}", re);
}

#[test]
fn t071_schroder_continuity_near_one() {
    // F_{√2}(1 − 1e-10) → √2 by continuity. F_{√2}(1) = √2 ≈ 1.4142135623…
    let (re, _im) = tet("30", "1.4142135623730950488", "0", "0.9999999999", "0");
    let r: f64 = re.parse().unwrap();
    assert!(
        (r - std::f64::consts::SQRT_2).abs() < 1e-8,
        "F_{{√2}}(1−1e-10) ≈ {}",
        re
    );
}

// --------------------------------------------------------------------------
// High-precision sanity: F_b(0) should be exactly 1 with as many digits as asked
// --------------------------------------------------------------------------

#[test]
fn t080_high_precision_zero_height() {
    let (re, im) = tet("1000", "2.7182818284", "0.0", "0", "0");
    // re should be "1.000…0e0" (1000 chars after the leading 1).
    assert!(re.starts_with("1.0"), "high-precision 1 starts with {}", &re[..30.min(re.len())]);
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
