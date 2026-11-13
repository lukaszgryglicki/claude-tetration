//! Kneser theta-mapping solver for real bases above `e^(1/e)`.
//!
//! References: the 100-digit `fatou.gp` value of `2^^0.5` and the external
//! 100-digit value of `e^^0.5` (see `tests/phase9_comprehensive.rs`); other
//! checks compare independent contours, the functional equation and
//! higher-precision recomputation.

use std::process::Command;

use rug::{Complex, Float};

use tetration::{cnum, dispatch, kneser};

const TWO_HALF: &str = "1.458781816036421700683971661038587135296606605330907141888207751656464160881083209019144256057703804";
const E_HALF: &str = "1.646354233751194580971924031592114518205311648969041530394879047518804627043365960655603009256205342";
const E: &str = "2.718281828459045235360287471352662497757247093699959574966967627724076630353547594571382178525166427427466391932003059921817413596629043572900334295260595630738";

fn parse(re: &str, im: &str, prec: u64) -> Complex {
    cnum::parse_complex(re, im, prec).unwrap()
}

fn agreement(a: &Complex, b: &Complex, prec: u64) -> Float {
    let difference = cnum::abs(&Complex::with_val_64(prec, a - b), prec);
    if difference.is_zero() {
        return Float::with_val_64(prec, rug::float::Special::Infinity);
    }
    let scale = cnum::abs(b, prec).max(&Float::with_val_64(prec, 1e-300));
    -(difference / scale).log10()
}

fn contour(centre: &str, radius: &str, height: &str) -> kneser::KneserGeometry {
    kneser::KneserGeometry {
        centre: centre.into(),
        radius: radius.into(),
        height: height.into(),
    }
}

#[test]
fn t1600_base_two_half_matches_the_independent_reference() {
    for digits in [15u64, 50, 100] {
        let (re, im) = tetration::tetrate_str(&digits.to_string(), "2", "0", "0.5", "0").unwrap();
        let prec = cnum::digits_to_bits(digits);
        let expected =
            cnum::format_float(&cnum::parse_float(TWO_HALF, prec).unwrap(), digits as usize);
        assert_eq!(re, expected, "digits={digits}");
        assert_eq!(im, "0", "digits={digits}");
    }
}

#[test]
fn t1601_base_e_half_matches_the_external_reference() {
    let digits = 80u64;
    let prec = cnum::digits_to_bits(digits);
    let value =
        dispatch::tetrate(&parse(E, "0", prec), &parse("0.5", "0", prec), prec, digits).unwrap();
    assert!(value.imag().is_zero());
    assert!(agreement(&value, &parse(E_HALF, "0", prec), prec) >= digits);
}

#[test]
fn t1602_independent_contours_agree() {
    let digits = 30u64;
    let prec = cnum::digits_to_bits(digits);
    let base = parse("3", "0", prec);
    let states: Vec<_> = [
        contour("-0.5", "1", "0.5"),
        contour("-0.7", "0.8", "0.35"),
        contour("-1", "0.7", "0.3"),
        contour("0", "1", "0.4"),
    ]
    .iter()
    .map(|geometry| kneser::setup_kneser_with_geometry(&base, prec, digits, geometry).unwrap())
    .collect();
    for (re, im) in [
        ("0.5", "0"),
        ("-0.5", "0.25"),
        ("0.3", "1.5"),
        ("2.25", "0"),
        ("-1.75", "0"),
        ("1.6", "-0.45"),
        ("-3.3", "-2"),
    ] {
        let height = parse(re, im, prec);
        let first = kneser::eval_kneser(&states[0], &height, prec, digits).unwrap();
        for state in &states[1..] {
            let other = kneser::eval_kneser(state, &height, prec, digits).unwrap();
            assert!(
                agreement(&other, &first, prec) >= digits,
                "h={re}+{im}i: {} digits",
                cnum::DisplayFloat(&agreement(&other, &first, prec))
            );
        }
    }
}

#[test]
fn t1603_functional_equation_reality_and_conjugation() {
    let digits = 25u64;
    let prec = cnum::digits_to_bits(digits);
    let base = parse("10", "0", prec);
    let state = kneser::setup_kneser(&base, prec, digits).unwrap();
    let ln_b = cnum::ln_complex(&base, prec);
    for (re, im) in [
        ("0.25", "0"),
        ("-0.75", "0"),
        ("0.4", "0.2"),
        ("-0.5", "0.49"),
        ("-0.5", "0.51"),
        ("0.1", "3"),
        ("-4.2", "0.7"),
        ("1.2", "-0.3"),
    ] {
        let height = parse(re, im, prec);
        let value = kneser::eval_kneser(&state, &height, prec, digits).unwrap();
        let next = kneser::eval_kneser(
            &state,
            &Complex::with_val_64(prec, &height + 1u32),
            prec,
            digits,
        )
        .unwrap();
        let image = cnum::checked_exp(&Complex::with_val_64(prec, &value * &ln_b), prec).unwrap();
        assert!(
            agreement(&next, &image, prec) >= digits - 2,
            "h={re}+{im}i: {} digits",
            cnum::DisplayFloat(&agreement(&next, &image, prec))
        );
        if im == "0" {
            assert!(value.imag().is_zero(), "h={re}");
        }
        let mirrored = kneser::eval_kneser(
            &state,
            &Complex::with_val_64(prec, height.conj_ref()),
            prec,
            digits,
        )
        .unwrap();
        assert_eq!(mirrored, value.conj(), "h={re}+{im}i");
    }
}

#[test]
fn t1604_large_and_near_parabolic_bases() {
    let digits = 15u64;
    let prec = cnum::digits_to_bits(digits);
    let half = parse("0.5", "0", prec);
    for (base, contours, expected) in [
        (
            "1e30",
            [contour("-1", "0.7", "0.3"), contour("-0.7", "0.8", "0.35")],
            "645.547844724944",
        ),
        (
            "1.46",
            [contour("-0.5", "1", "0.5"), contour("-0.5", "1", "0.3")],
            "1.26383460066838",
        ),
    ] {
        let b = parse(base, "0", prec);
        for geometry in &contours {
            let state = kneser::setup_kneser_with_geometry(&b, prec, digits, geometry).unwrap();
            let value = kneser::eval_kneser(&state, &half, prec, digits).unwrap();
            assert_eq!(
                cnum::format_complex(&value, digits as usize).0,
                expected,
                "b={base}"
            );
            let before =
                kneser::eval_kneser(&state, &parse("-0.5", "0", prec), prec, digits).unwrap();
            let image = cnum::checked_exp(
                &Complex::with_val_64(prec, before * cnum::ln_complex(&b, prec)),
                prec,
            )
            .unwrap();
            assert!(agreement(&value, &image, prec) >= digits - 1, "b={base}");
        }
    }
    let automatic = dispatch::tetrate(&parse("1e30", "0", prec), &half, prec, digits).unwrap();
    assert_eq!(
        cnum::format_complex(&automatic, digits as usize).0,
        "645.547844724944"
    );
}

#[test]
fn t1605_special_and_singular_heights() {
    let digits = 25u64;
    let prec = cnum::digits_to_bits(digits);
    let base = parse("2", "0", prec);
    let state = kneser::setup_kneser(&base, prec, digits).unwrap();
    let eval = |re: &str, im: &str| kneser::eval_kneser(&state, &parse(re, im, prec), prec, digits);
    assert!(eval("-2", "0").unwrap_err().contains("undefined"));
    assert!(eval("-7", "0").unwrap_err().contains("undefined"));
    assert!(cnum::is_zero(&eval("-1", "0").unwrap()));
    assert!(cnum::is_one(&eval("0", "0").unwrap()));
    assert!(agreement(&eval("1", "0").unwrap(), &base, prec) >= digits);
    // F(-2.5) = log_2 F(-1.5) with F(-1.5) < 0: the principal branch has Im = pi/ln 2.
    let below = eval("-2.5", "0").unwrap();
    let pi = Float::with_val_64(prec, rug::float::Constant::Pi);
    let expected = pi / Float::with_val_64(prec, 2).ln();
    let defect = Float::with_val_64(prec, below.imag() - &expected).abs();
    assert!(defect < cnum::epsilon(digits, prec));
    let mirrored = eval("-2.5", "-1e-30").unwrap();
    assert!(*mirrored.imag() < 0);
}

#[test]
fn t1606_height_conditioning_keeps_requested_digits() {
    let digits = 20u64;
    let prec = cnum::digits_to_bits(digits);
    let high_digits = 70u64;
    let high = cnum::digits_to_bits(high_digits);
    let two = parse("2", "0", high);
    let ln_two = cnum::ln_complex(&two, high);
    let mut reference =
        dispatch::tetrate(&two, &parse("0.5", "0", high), high, high_digits).unwrap();
    for _ in 0..4 {
        reference =
            cnum::checked_exp(&Complex::with_val_64(high, reference * &ln_two), high).unwrap();
    }
    let value = dispatch::tetrate(
        &parse("2", "0", prec),
        &parse("4.5", "0", prec),
        prec,
        digits,
    )
    .unwrap();
    assert!(agreement(&value, &reference, high) >= digits);

    let near = "-0.99999999999999999";
    let small = dispatch::tetrate(
        &parse("2", "0", prec),
        &parse(near, "0", prec),
        prec,
        digits,
    )
    .unwrap();
    let exact = dispatch::tetrate(&two, &parse(near, "0", high), high, high_digits).unwrap();
    assert!(agreement(&small, &exact, high) >= digits);
}

#[test]
fn t1608_far_left_and_near_integer_heights() {
    // Far to the left the logarithm chain is finished in closed form at the
    // fixed point instead of running one step per unit of height.
    let fixed = (
        "8.24678546142074e-1".to_string(),
        "1.56743212384965".to_string(),
    );
    for (re, im) in [("-1000000000000.5", "0"), ("-1e30", "0.25")] {
        assert_eq!(
            tetration::tetrate_str("15", "2", "0", re, im).unwrap(),
            fixed,
            "h={re}+{im}i"
        );
    }
    // Heights given to more digits than the solver state carries are reduced
    // exactly: these are not the integers -2 and -1.
    let nines = "99999999999999999999999999999999999999999999999999";
    assert_eq!(
        tetration::tetrate_str("15", "2", "0", &format!("-1.{nines}"), "0").unwrap(),
        ("-165.736790909823".to_string(), "0".to_string())
    );
    assert_eq!(
        tetration::tetrate_str("15", "2", "0", &format!("-0.{nines}"), "0").unwrap(),
        ("1.28308240957212e-50".to_string(), "0".to_string())
    );
}

#[test]
fn t1609_bases_next_to_the_parabolic_base() {
    // Next to e^(1/e) the Poincare series is summed on a disk reaching most
    // of the way to the other fixed point, which keeps the logarithmic orbits
    // short. Independent contours agree and satisfy the recurrence.
    let digits = 20u64;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("1.45", "0", prec);
    let ln_b = cnum::ln_complex(&b, prec);
    for geometry in [contour("-0.5", "1", "0.5"), contour("-0.5", "1", "0.3")] {
        let state = kneser::setup_kneser_with_geometry(&b, prec, digits, &geometry).unwrap();
        let eval =
            |re: &str, im: &str| kneser::eval_kneser(&state, &parse(re, im, prec), prec, digits);
        let value = eval("0.5", "0").unwrap();
        assert_eq!(
            cnum::format_complex(&value, digits as usize).0,
            "1.2594864327497848579"
        );
        let before = eval("-0.5", "0.25").unwrap();
        let image = cnum::checked_exp(&Complex::with_val_64(prec, before * &ln_b), prec).unwrap();
        assert!(agreement(&eval("0.5", "0.25").unwrap(), &image, prec) >= digits - 1);
    }
    assert_eq!(
        tetration::tetrate_str("15", "1.445", "0", "0.5", "0").unwrap(),
        ("1.25729872483980".to_string(), "0".to_string())
    );
}

#[test]
fn t1607_cli_is_real_and_serial_mt_identical() {
    let run = |threads: Option<&str>, arguments: &[&str]| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_tet"));
        command
            .args(arguments)
            .env("SILENT", "1")
            .env_remove("TET_MT");
        if let Some(threads) = threads {
            command.env("TET_MT", threads);
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap()
    };
    let real = run(None, &["30", "2", "0", "0.5", "0"]);
    assert_eq!(real, "1.45878181603642170068397166104\n0\n");
    for arguments in [
        ["40", "3", "0", "0.5", "0.25"],
        ["40", "3", "0", "-1.5", "2"],
        ["25", "1e30", "0", "0.5", "0"],
        ["15", "1.46", "0", "0.5", "0.25"],
    ] {
        assert_eq!(
            run(None, &arguments),
            run(Some("4"), &arguments),
            "{arguments:?}"
        );
    }
}
