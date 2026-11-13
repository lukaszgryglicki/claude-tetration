//! Two-sided theta-mapping solver for complex, negative and cut bases
//! (`src/kneser_complex.rs`).
//!
//! Evidence, not a certificate: agreement with the Kneser solver on the real
//! axis, with values the Cauchy solver produced for the same bases, between
//! different contours, the functional equation at complex heights and the
//! reflection convention.

use rug::{Complex, Float};

use tetration::{cnum, dispatch, kneser, kneser_complex};

fn parse(re: &str, im: &str, prec: u64) -> Complex {
    cnum::parse_complex(re, im, prec).unwrap()
}

fn agreement(a: &Complex, b: &Complex, prec: u64) -> Float {
    let difference = cnum::abs(&Complex::with_val_64(prec, a - b), prec);
    if difference.is_zero() {
        return Float::with_val_64(prec, rug::float::Special::Infinity);
    }
    -(difference / cnum::abs(b, prec)).log10()
}

fn contour(centre: &str, radius: &str, height: &str) -> kneser::KneserGeometry {
    kneser::KneserGeometry {
        centre: centre.into(),
        radius: radius.into(),
        height: height.into(),
    }
}

#[test]
fn t1800_real_bases_reproduce_the_kneser_solver() {
    let digits = 25u64;
    let prec = cnum::digits_to_bits(digits);
    for base in ["2", "3", "1.6"] {
        let b = parse(base, "0", prec);
        let real = kneser::setup_kneser(&b, prec, digits).unwrap();
        let complex = kneser_complex::setup_complex_kneser(&b, prec, digits).unwrap();
        for (re, im) in [
            ("0.5", "0"),
            ("-1.5", "0"),
            ("0.3", "0.2"),
            ("2.5", "-1"),
            ("-0.7", "3"),
        ] {
            let h = parse(re, im, prec);
            let expected = kneser::eval_kneser(&real, &h, prec, digits).unwrap();
            let actual = kneser_complex::eval_complex_kneser(&complex, &h, prec, digits).unwrap();
            assert!(
                agreement(&actual, &expected, prec) >= digits,
                "b={base} h={re}+{im}i: {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn t1801_values_of_the_cauchy_solver_are_reproduced() {
    // F_b(1/2) printed by the Kouznetsov path before this solver existed (26-52 s each).
    for (re, im, expected_re, expected_im) in [
        ("2", "0.1", "1.45957161575665", "3.05769359306522e-2"),
        ("2", "0.3", "1.46578919281063", "9.10145518721148e-2"),
        ("3", "0.5", "1.71490079555383", "1.01468785448296e-1"),
        ("3", "4", "2.00359606445356", "5.83823967111922e-1"),
        ("10", "5", "2.54550059082652", "3.14690482023885e-1"),
    ] {
        assert_eq!(
            tetration::tetrate_str("15", re, im, "0.5", "0").unwrap(),
            (expected_re.to_string(), expected_im.to_string()),
            "b={re}+{im}i"
        );
    }
}

/// Thirty digits on which the contours -0.5/0.75/0.35, -1/0.7/0.3 and
/// -0.6/0.8/0.3 agree, at the heights 0.5, -1.5+0.2i and 1.3-2i.
type Pair = (&'static str, &'static str);

const REFERENCES: [(&str, &str, [Pair; 3]); 5] = [
    (
        "-2",
        "0",
        [
            (
                "1.23499204711001806741684243512",
                "1.72765186259320471027645429067",
            ),
            (
                "-6.98592420699095608266144945758e-2",
                "3.16528491614682003033844754417e-1",
            ),
            (
                "-2.44740421788856262722228299301e-1",
                "1.19763546305152743317621694269e-1",
            ),
        ],
    ),
    (
        "3",
        "4",
        [
            (
                "2.00359606445356404731913347078",
                "5.83823967111921778551511987461e-1",
            ),
            (
                "-3.22286490325166977758513422805e-1",
                "3.47471488668148941890930712897e-1",
            ),
            (
                "-1.01381684628366619893019181747",
                "-6.47387903671622879534199815275e-1",
            ),
        ],
    ),
    (
        "-1",
        "2",
        [
            (
                "1.45336611150905558030528568694",
                "1.16828211170582927858684009606",
            ),
            (
                "-1.30601949765508112827626909652e-1",
                "3.96768167462366093301290887687e-1",
            ),
            (
                "-4.29899180897713048783774411425e-1",
                "4.76098259126775510695175796640e-1",
            ),
        ],
    ),
    (
        "0.04",
        "0",
        [
            (
                "6.63831355231474524653734313751e-2",
                "4.47944861924265149963222115626e-1",
            ),
            (
                "2.83617316995158413369345599239e-1",
                "2.01718860667329459500722642265e-1",
            ),
            (
                "3.75041763278386912087202224630e-2",
                "-6.73967540042881833401260428688e-2",
            ),
        ],
    ),
    (
        "0.01",
        "0",
        [
            (
                "-1.25023808503236713358729653966e-1",
                "4.13987966241729684229874554858e-1",
            ),
            (
                "2.28766335661523434111243134953e-1",
                "1.52037997691021943734672504061e-1",
            ),
            (
                "-1.68444674358235556495355215901e-3",
                "-1.22102618948678147893393676165e-1",
            ),
        ],
    ),
];

const HEIGHTS: [Pair; 3] = [("0.5", "0"), ("-1.5", "0.2"), ("1.3", "-2")];

#[test]
fn t1802_negative_complex_and_cut_bases_through_the_dispatcher() {
    // -2 and the cut bases had no value before; all go through `dispatch`.
    let digits = 22u64;
    let prec = cnum::digits_to_bits(digits);
    let wide = cnum::digits_to_bits(40);
    for (re, im, values) in REFERENCES {
        let b = parse(re, im, prec);
        for ((h_re, h_im), (v_re, v_im)) in HEIGHTS.iter().zip(values) {
            let actual = dispatch::tetrate(&b, &parse(h_re, h_im, prec), prec, digits).unwrap();
            let expected = parse(v_re, v_im, wide);
            assert!(
                agreement(&actual, &expected, wide) >= digits,
                "b={re}+{im}i h={h_re}+{h_im}i: {actual}"
            );
        }
    }
}

#[test]
fn t1803_an_independent_contour_and_base_continuation_agree() {
    // The default contour cannot start at the cut base 0.04 directly; it is
    // reached through bases above the Shell-Thron region and must end at the
    // same values as the smaller contours behind the references.
    let digits = 20u64;
    let prec = cnum::digits_to_bits(digits);
    let wide = cnum::digits_to_bits(40);
    for (index, geometry) in [
        (3usize, contour("-0.5", "1", "0.5")),
        (0, contour("-0.4", "0.9", "0.4")),
    ] {
        let (re, im, values) = REFERENCES[index];
        let b = parse(re, im, prec);
        let state = kneser_complex::setup_complex_kneser_with_geometry(&b, prec, digits, &geometry)
            .unwrap();
        for ((h_re, h_im), (v_re, v_im)) in HEIGHTS.iter().zip(values) {
            let h = parse(h_re, h_im, prec);
            let actual = kneser_complex::eval_complex_kneser(&state, &h, prec, digits).unwrap();
            let expected = parse(v_re, v_im, wide);
            assert!(
                agreement(&actual, &expected, wide) >= digits,
                "b={re}+{im}i h={h_re}+{h_im}i: {actual}"
            );
        }
    }
}

#[test]
fn t1804_functional_equation_reflection_and_fixed_points() {
    let digits = 20u64;
    let prec = cnum::digits_to_bits(digits);
    for (re, im) in [("-1", "2"), ("0.04", "0"), ("0.5", "2")] {
        let b = parse(re, im, prec);
        let ln_b = cnum::ln_complex(&b, prec);
        let state = kneser_complex::setup_complex_kneser(&b, prec, digits).unwrap();
        let eval =
            |h: &Complex| kneser_complex::eval_complex_kneser(&state, h, prec, digits).unwrap();
        // Across both theta lines, inside the strip and through the logarithms.
        for (h_re, h_im) in [
            ("0.2", "0.45"),
            ("0.2", "0.55"),
            ("-0.3", "-0.52"),
            ("-2.4", "0.1"),
            ("0.7", "-3"),
        ] {
            let h = parse(h_re, h_im, prec);
            let next = eval(&Complex::with_val_64(prec, &h + 1u32));
            let image =
                cnum::checked_exp(&Complex::with_val_64(prec, eval(&h) * &ln_b), prec).unwrap();
            assert!(
                agreement(&next, &image, prec) >= digits - 2,
                "b={re}+{im}i h={h_re}+{h_im}i"
            );
        }
        // The function tends to the two fixed points far up and far down.
        let (upper, lower) = state.fixed_points();
        assert!(agreement(&eval(&parse("0.3", "200", prec)), upper, prec) >= digits);
        assert!(agreement(&eval(&parse("0.3", "-200", prec)), lower, prec) >= digits);
        // Far to the left the height reduction ends at the fixed point of its side.
        assert!(agreement(&eval(&parse("-1e30", "0.1", prec)), upper, prec) >= digits);
        assert!(agreement(&eval(&parse("-1e30", "-0.1", prec)), lower, prec) >= digits);
    }
    // Im b < 0 is the reflected function.
    let above = dispatch::tetrate(
        &parse("-1", "2", prec),
        &parse("0.4", "-0.3", prec),
        prec,
        digits,
    )
    .unwrap();
    let below = dispatch::tetrate(
        &parse("-1", "-2", prec),
        &parse("0.4", "0.3", prec),
        prec,
        digits,
    )
    .unwrap();
    assert!(agreement(&below, &above.conj(), prec) >= digits);
}

#[test]
fn t1805_continuity_across_the_real_base_axis_and_precision_ladder() {
    // A cut base is the limit from the upper half-plane, and a real base above
    // e^(1/e) is approached by complex ones.
    let digits = 20u64;
    let prec = cnum::digits_to_bits(digits);
    let h = parse("0.5", "0", prec);
    for (base, slope_digits) in [("0.04", 8i32), ("2", 8)] {
        let on = dispatch::tetrate(&parse(base, "0", prec), &h, prec, digits).unwrap();
        let near = dispatch::tetrate(&parse(base, "1e-10", prec), &h, prec, digits).unwrap();
        assert!(
            agreement(&near, &on, prec) >= slope_digits,
            "b={base}: {near} vs {on}"
        );
    }
    // The same digits at a higher precision level.
    let (re, im) = tetration::tetrate_str("50", "-2", "0", "0.5", "0").unwrap();
    assert!(re.starts_with("1.234992047110018067416842435"), "{re}");
    assert!(im.starts_with("1.727651862593204710276454290"), "{im}");
}

#[test]
fn t1807_a_very_small_base_is_reached_along_the_cut() {
    // No direct start converges at 1e-5; the base is reached from e^-3.2
    // along the cut. The much longer path through complex bases on another
    // contour ended at the same digits.
    assert_eq!(
        tetration::tetrate_str("15", "1e-5", "0", "0.5", "0").unwrap(),
        (
            "-4.55644111497146e-1".to_string(),
            "-1.16859876798655e-1".to_string()
        )
    );
}

#[test]
fn t1806_heights_next_to_the_singularities_and_scope() {
    // Not the integer -2: reduced exactly, with the logarithm branch traced.
    let nines = "99999999999999999999999999999999999999999999999999";
    let (re, im) = tetration::tetrate_str("15", "2", "0.3", &format!("-1.{nines}"), "0").unwrap();
    assert_eq!(
        (re.as_str(), im.as_str()),
        ("-156.186050317294", "32.8917524203511")
    );
    // The cut runs along the real axis to the left of -2 only.
    let digits = 15u64;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("2", "0.3", prec);
    let eval =
        |re: &str, im: &str| dispatch::tetrate(&b, &parse(re, im, prec), prec, digits).unwrap();
    assert!(agreement(&eval("-1.9999", "1e-12"), &eval("-1.9999", "-1e-12"), prec) >= 7);
    let jump = Complex::with_val_64(prec, eval("-2.0001", "1e-12") - eval("-2.0001", "-1e-12"));
    let period = Complex::with_val_64(
        prec,
        (0, Float::with_val_64(prec, rug::float::Constant::Pi) * 2u32),
    ) / cnum::ln_complex(&b, prec);
    assert!(agreement(&jump, &period, prec) >= 6, "{jump} vs {period}");
    // Integer heights at and below -2 stay undefined; bases inside the
    // Shell-Thron region are outside the scope of this solver.
    assert!(tetration::tetrate_str("15", "2", "0.3", "-2", "0").is_err());
    let inside = parse("2", "0.6", prec);
    let error = kneser_complex::setup_complex_kneser(&inside, prec, digits)
        .err()
        .unwrap();
    assert!(error.contains("not repelling"), "{error}");
}

#[test]
fn t1808_a_level_below_the_exponent_range_restarts_the_tower() {
    // A level of this tower lies below MPFR's range, so the following ones
    // are 1, b, b^b, b^b^b at every precision; with b^b = 10^-200 the tower
    // then repeats with period 3 far beyond the printed digits.
    assert_eq!(
        tetration::tetrate_str("15", "-100", "0", "100.5", "0").unwrap(),
        tetration::tetrate_str("15", "-100", "0", "3", "0").unwrap()
    );
    assert_eq!(
        tetration::tetrate_str("15", "-100", "0", "100.5", "0").unwrap(),
        (
            "1.00000000000000".to_string(),
            "3.14159265358979e-200".to_string()
        )
    );
}

#[test]
fn t1809_a_nearly_neutral_multiplier_starts_directly() {
    // |lambda| = 1.005 at the upper fixed point: the logarithmic orbits spiral
    // in slowly, which must not be taken for a failure to approach it.
    let digits = 15u64;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("1.7734612812", "0.159600507187", prec);
    let ln_b = cnum::ln_complex(&b, prec);
    let state = kneser_complex::setup_complex_kneser(&b, prec, digits).unwrap();
    let eval = |re: &str, im: &str| {
        kneser_complex::eval_complex_kneser(&state, &parse(re, im, prec), prec, digits).unwrap()
    };
    let value = eval("0.5", "0");
    assert_eq!(
        cnum::format_complex(&value, digits as usize),
        (
            "1.38765359983500".to_string(),
            "5.52522025324407e-2".to_string()
        )
    );
    let image =
        cnum::checked_exp(&Complex::with_val_64(prec, eval("-0.5", "0") * &ln_b), prec).unwrap();
    assert!(agreement(&value, &image, prec) >= digits - 1);
}

#[test]
fn t1810_a_long_poincare_series_reaches_the_lines() {
    // |lambda| = 1.0005 at the upper fixed point (argument 1): the digits of
    // the short series, whose orbits took thousands of logarithms per sample
    // (100 s). The long series holds the lines and takes none.
    assert_eq!(
        tetration::tetrate_str(
            "15",
            "1.77043116020741634070338878631",
            "0.162889518463665419508723173616",
            "0.5",
            "0"
        )
        .unwrap(),
        (
            "1.38672194162660".to_string(),
            "5.64853994314319e-2".to_string()
        )
    );
    // |lambda| = 1 + 10^-8, out of reach of the short series: the same
    // digits on another contour, and the functional equation.
    let digits = 15u64;
    let prec = cnum::digits_to_bits(digits);
    let b = parse(
        "1.770092737094460584327615948947547484885",
        "0.1632528288422982831409167375865243798265",
        prec,
    );
    let ln_b = cnum::ln_complex(&b, prec);
    let state = kneser_complex::setup_complex_kneser(&b, prec, digits).unwrap();
    let second = kneser_complex::setup_complex_kneser_with_geometry(
        &b,
        prec,
        digits,
        &contour("-0.5", "0.75", "0.35"),
    )
    .unwrap();
    let eval = |state: &kneser_complex::ComplexKneserState, re: &str, im: &str| {
        kneser_complex::eval_complex_kneser(state, &parse(re, im, prec), prec, digits).unwrap()
    };
    for ((re, im), expected) in [
        (("0.5", "0"), ("1.38661782875409", "5.66219905969793e-2")),
        (("0.5", "0.25"), ("1.34819262932619", "2.39545550581152e-1")),
        (
            ("-1.5", "0"),
            ("-9.71562534278606e-1", "9.51318587182804e-2"),
        ),
        (("-0.7", "2"), ("9.24791415358686e-1", "1.12800371163643")),
    ] {
        let value = eval(&state, re, im);
        let (real, imaginary) = cnum::format_complex(&value, digits as usize);
        assert_eq!(
            (real.as_str(), imaginary.as_str()),
            expected,
            "h={re}+{im}i"
        );
        assert!(
            agreement(&value, &eval(&second, re, im), prec) >= digits - 1,
            "h={re}+{im}i"
        );
    }
    let image = cnum::checked_exp(
        &Complex::with_val_64(prec, eval(&state, "-0.5", "0") * &ln_b),
        prec,
    )
    .unwrap();
    assert!(agreement(&eval(&state, "0.5", "0"), &image, prec) >= digits - 1);
}

#[test]
fn t1811_cut_bases_next_to_the_flip_take_higher_lines() {
    // lambda = -1.0056 at the real fixed point of 0.064. On the default
    // contour the upper line leaves the domain of the Abel function there,
    // and no number of theta modes helps; the solver moves to a contour
    // with higher lines. The explicit contours -0.5/1.2/0.9 and
    // -0.5/1.35/1.1 gave these digits, and 30 requested digits agree.
    let digits = 15u64;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("0.064", "0", prec);
    let ln_b = cnum::ln_complex(&b, prec);
    let state = kneser_complex::setup_complex_kneser(&b, prec, digits).unwrap();
    let eval = |re: &str, im: &str| {
        kneser_complex::eval_complex_kneser(&state, &parse(re, im, prec), prec, digits).unwrap()
    };
    for ((re, im), expected) in [
        (("0.5", "0"), ("1.44775599906460e-1", "4.43320899614581e-1")),
        (
            ("0.5", "0.25"),
            ("3.16405891384551e-1", "2.10445927511775e-1"),
        ),
        (
            ("-1.5", "0"),
            ("2.28011900720536e-1", "3.72774621673599e-1"),
        ),
    ] {
        let (real, imaginary) = cnum::format_complex(&eval(re, im), digits as usize);
        assert_eq!(
            (real.as_str(), imaginary.as_str()),
            expected,
            "h={re}+{im}i"
        );
    }
    let image =
        cnum::checked_exp(&Complex::with_val_64(prec, eval("-0.5", "0") * &ln_b), prec).unwrap();
    assert!(agreement(&eval("0.5", "0"), &image, prec) >= digits - 1);
}
