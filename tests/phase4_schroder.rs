//! Phase-4 verification: Schröder regular tetration for Shell-Thron interior
//! bases.
//!
//! The defining property is the Abel-tetration functional equation
//!     F_b(z + 1) = b^{F_b(z)}.
//! With Schröder construction we also have F_b(0) = 1, F_b(1) = b, F_b(2)=b^b
//! by analytical identity. These numerical checks are not accuracy certificates.

use rug::{Complex, Float};

use tetration::{cnum, dispatch, regions, schroder};

fn parse(re: &str, im: &str, prec: u64) -> Complex {
    cnum::parse_complex(re, im, prec).unwrap()
}

fn abs(z: &Complex, prec: u64) -> Float {
    Float::with_val_64(prec, z.abs_ref())
}

/// Returns -log10(|a - b|) — i.e., approximate matching digits.
fn matching_digits(a: &Complex, b: &Complex, prec: u64) -> Float {
    assert!(cnum::is_finite(a) && cnum::is_finite(b));
    let diff = Complex::with_val_64(prec, a - b);
    let da = abs(&diff, prec);
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
        "b={}+{}i z={}+{}i: F(z+1)={}+{}i  b^F(z)={}+{}i  matched {} digits",
        b_re,
        b_im,
        z_re,
        z_im,
        fz1.real(),
        fz1.imag(),
        b_to_fz.real(),
        b_to_fz.imag(),
        m
    );
}

#[test]
fn t300_sqrt2_functional_eq_at_half() {
    check_functional_eq("1.4142135623730950488", "0", "0.5", "0", 50, 35);
}

#[test]
fn t301_sqrt2_functional_eq_at_one_and_a_half() {
    check_functional_eq("1.4142135623730950488", "0", "1.5", "0", 50, 35);
}

#[test]
fn t302_sqrt2_functional_eq_negative_half() {
    // F(-0.5) requires the series to handle |λ^{-0.5}|>1; if the shift logic
    // engages, F(-0.5) = log_b(F(0.5)) gives the same answer.
    check_functional_eq("1.4142135623730950488", "0", "-0.5", "0", 50, 35);
}

#[test]
fn t310_b_one_point_two_functional_eq() {
    check_functional_eq("1.2", "0", "0.5", "0", 50, 40);
}

#[test]
fn t311_b_one_point_two_complex_height() {
    check_functional_eq("1.2", "0", "0.5", "0.3", 50, 35);
}

#[test]
fn t320_b_zero_point_five_functional_eq() {
    // b = 0.5, λ = -0.444 (inside Shell-Thron, real-negative multiplier).
    check_functional_eq("0.5", "0", "0.7", "0", 50, 35);
}

#[test]
fn t330_complex_base_functional_eq() {
    // b = 1.3 + 0.1i — interior of Shell-Thron with truly complex multiplier.
    check_functional_eq("1.3", "0.1", "0.4", "0.2", 50, 30);
}

#[test]
fn t340_high_precision_functional_eq() {
    // At 100 digits, expect at least ~70 digits of agreement.
    check_functional_eq("1.4142135623730950488", "0", "0.5", "0", 100, 70);
}

#[test]
fn t350_continuity_at_integer() {
    // F(1 + ε) should be very close to F(1) = b for small real ε.
    let digits = 50;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("1.2", "0", prec);
    let h = parse("1.0000000001", "0", prec);
    let f = dispatch::tetrate(&b, &h, prec, digits).unwrap();
    let diff = Complex::with_val_64(prec, &f - &b);
    let da = abs(&diff, prec);
    assert!(
        da < cnum::epsilon(9, prec),
        "F(1.0000000001) - b ≈ {}, expected < 1e-9",
        da
    );
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
    let diff = Complex::with_val_64(prec, &f_int - &f_near);
    let da = abs(&diff, prec);
    assert!(
        da < cnum::epsilon(9, prec),
        "F(2) − F(2−1e-11) ≈ {}, expected < 1e-9",
        da
    );
}

#[test]
fn t370_setup_eval_matches_direct() {
    // Cached setup + eval must produce bit-identical results to the direct
    // `tetrate_schroder` wrapper across many heights for the same base. This
    // is what grid_runner relies on for its 100× speedup at digits ≥ 20.
    let digits = 30u64;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("1.2", "-0.4", prec);
    let region = regions::classify(&b, prec).expect("classify");
    let fp = match &region {
        regions::Region::ShellThronInterior(d) => d,
        other => panic!("expected ShellThronInterior, got {}", other.name()),
    };
    let state = schroder::setup_schroder(&b, fp, prec).expect("setup");
    let heights = ["0.5", "1.5", "-0.5", "0.7", "-1.3"];
    for h_str in &heights {
        for h_im_str in &["0", "0.3", "-0.7"] {
            let h = parse(h_str, h_im_str, prec);
            let direct = schroder::tetrate_schroder(&b, &h, fp, prec).expect("direct");
            let cached = schroder::eval_schroder(&state, &h).expect("cached");
            assert_eq!(direct, cached, "cache mismatch at h=({h_str},{h_im_str})");
        }
    }
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
    let f_lo_hi = Complex::with_val_64(prec_hi, &f_lo);
    let diff = Complex::with_val_64(prec_hi, &f_hi - &f_lo_hi);
    let da = abs(&diff, prec_hi);
    assert!(
        da < cnum::epsilon(digits_lo - 5, prec_hi),
        "precision-scaling residual {} too large",
        da
    );
}

#[test]
fn t380_far_left_heights_end_in_closed_form() {
    // One principal logarithm per unit of height until the orbit is
    // stationary: the vanishing imaginary part of a real base then follows in
    // closed form, and exact cycles are skipped by whole periods. The
    // expected digits are those of the plain loop.
    assert_eq!(
        tetration::tetrate_str("15", "1.3", "0", "-1000000.5", "0").unwrap(),
        (
            "7.85706535116432".to_string(),
            "2.31452068032643e-314164".to_string()
        )
    );
    let cycle = ("-1.14012370876".to_string(), "7.06579007931e-1".to_string());
    for height in ["-1000.5", "-1000000000000.5"] {
        assert_eq!(
            tetration::tetrate_str("12", "0", "1", height, "0").unwrap(),
            cycle,
            "h={height}"
        );
    }
    assert_eq!(
        tetration::tetrate_str("15", "0.3", "0.3", "-1000000.5", "0").unwrap(),
        (
            "-1.21065168093612".to_string(),
            "3.22367768947075e-1".to_string()
        )
    );
    // An imaginary part below MPFR's range is an error, not a zero.
    let error = tetration::tetrate_str("15", "1.3", "0", "-1000000000000000000000000000000.5", "0")
        .unwrap_err();
    assert!(error.contains("exponent range"), "{error}");
}

#[test]
fn t390_logarithm_branches_follow_the_tetration_continuously() {
    // |lambda| = 0.992 with a turning multiplier: the curve s -> F(h + s)
    // winds around 0, so the logarithms of the height reduction are not the
    // principal ones. Those gave a discontinuous function of the real height
    // here (1.5688 - 1.5661i between 0.56 and 0.60).
    let (re, im) = ("1.764630858490303", "0.1690076064324368");
    let prec = cnum::digits_to_bits(15);
    let values: Vec<Complex> = ["0.52", "0.54", "0.56", "0.58", "0.60", "0.62"]
        .iter()
        .map(|height| {
            let (value_re, value_im) = tetration::tetrate_str("15", re, im, height, "0").unwrap();
            parse(&value_re, &value_im, prec)
        })
        .collect();
    assert_eq!(
        tetration::tetrate_str("15", re, im, "0.60", "0").unwrap(),
        (
            "1.53538151660357".to_string(),
            "7.81658334622440e-2".to_string()
        )
    );
    // Third differences of an analytic function at this spacing are tiny;
    // across a jump they are of the order of the jump.
    for window in values.windows(4) {
        let third = Complex::with_val_64(prec, &window[3] - &window[0])
            - Complex::with_val_64(prec, &window[2] - &window[1]) * 3u32;
        assert!(abs(&third, prec) < 1e-3, "third difference {third}");
    }
    // Outside the half-plane |lambda^(h+2)| < 1, where the regular tetration
    // is single-valued, the height reduction keeps its principal logarithms.
    assert_eq!(
        tetration::tetrate_str("15", "0.2", "0.4", "2.5", "-1").unwrap(),
        (
            "2.01498533063010".to_string(),
            "-9.63361309384064e-1".to_string()
        )
    );
}

#[test]
fn t391_long_inverse_series_for_nearly_neutral_multipliers() {
    // lambda = -0.9986. The short series of the attracting disk and its chain
    // of 7500 logarithms give these 50 digits as well; the long series is
    // summed on most of the disk whose boundary holds the coordinate of the
    // height -2, about a hundred levels above the height.
    let prec = cnum::digits_to_bits(50);
    let state = |re: &str| {
        let base = parse(re, "0", prec);
        let fp = match regions::classify(&base, prec).unwrap() {
            regions::Region::ShellThronInterior(fp) | regions::Region::ShellThronBoundary(fp) => fp,
            other => panic!("expected an attracting base, got {other:?}"),
        };
        schroder::setup_schroder(&base, &fp, prec).unwrap()
    };
    let near = state("0.0665");
    assert!(near.univalent);
    assert_eq!(near.prec, prec);
    assert!(near.inverse_radius.as_ref().unwrap() > &near.safe_radius);
    assert!(!state("1.4142135623730951").univalent);
    assert_eq!(
        tetration::tetrate_str("50", "0.0665", "0", "0.5", "0").unwrap(),
        (
            "3.6762381657683836609365798186519960013603167802599e-1".to_string(),
            "3.3901837390606446623420497690676123803623304409316e-2".to_string()
        )
    );
    // Heights outside the half-plane of the series keep their values.
    assert_eq!(
        tetration::tetrate_str("15", "0.0665", "0", "-1000000.5", "0").unwrap(),
        (
            "3.66828727476977e-1".to_string(),
            "-4.81788392990710e-2".to_string()
        )
    );
    // |lambda| = 0.998 at argument 1: the digits of the chain of 6400 traced
    // levels that the short series needs (three and a half minutes).
    assert_eq!(
        tetration::tetrate_str("15", "1.76873553494", "0.164701833467", "0.5", "0").unwrap(),
        (
            "1.47919432026926".to_string(),
            "1.69457889363597e-2".to_string()
        )
    );
    // |lambda| = 0.999: the same digits at two precisions, and the
    // super-logarithm, which follows the orbit of exponentials and inverts
    // the series on a smaller disk, returns the height.
    let (re, im) = ("1.76941482678", "0.163978190153");
    let value = tetration::tetrate_str("25", re, im, "0.5", "0").unwrap();
    assert_eq!(
        value,
        (
            "1.479974722806787003598171".to_string(),
            "1.347300134691683414106475e-2".to_string()
        )
    );
    assert_eq!(
        tetration::tetrate_str("15", re, im, "0.5", "0").unwrap(),
        (
            "1.47997472280679".to_string(),
            "1.34730013469168e-2".to_string()
        )
    );
    let height = tetration::slog_str("15", re, im, &value.0, &value.1).unwrap();
    let small = cnum::digits_to_bits(15);
    let miss = Complex::with_val_64(small, parse(&height.0, &height.1, small) - 0.5f32);
    assert!(abs(&miss, small) < 1e-13, "slog returned {height:?}");
}
