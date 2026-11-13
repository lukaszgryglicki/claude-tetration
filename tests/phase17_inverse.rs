//! Inverses of tetration: super-logarithm (`--slog`) and super-root (`--sroot`).
//!
//! Round trips use forward values computed at a much higher precision; the
//! super square root is checked against `exp(W(ln z))`.

use std::process::Command;

use rug::{Complex, Float};

use tetration::{cnum, dispatch, inverse, lambertw};

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

const HIGH_DIGITS: u64 = 70;

fn forward(base: (&str, &str), height: (&str, &str)) -> Complex {
    let prec = cnum::digits_to_bits(HIGH_DIGITS);
    dispatch::tetrate(
        &parse(base.0, base.1, prec),
        &parse(height.0, height.1, prec),
        prec,
        HIGH_DIGITS,
    )
    .unwrap()
}

#[test]
fn t1700_slog_exact_values() {
    assert_eq!(
        tetration::slog_str("30", "2", "0", "16", "0").unwrap(),
        (
            "3.00000000000000000000000000000".to_string(),
            "0".to_string()
        )
    );
    assert_eq!(
        tetration::slog_str("30", "2", "0", "65536", "0").unwrap().0,
        "4.00000000000000000000000000000"
    );
    for (re, im) in [
        ("2", "0"),
        ("1.3", "0"),
        ("0.5", "0.5"),
        ("10", "0"),
        ("0.2", "-0.4"),
    ] {
        assert_eq!(tetration::slog_str("20", re, im, "1", "0").unwrap().0, "0");
        assert_eq!(
            tetration::slog_str("20", re, im, "0", "0").unwrap().0,
            "-1.0000000000000000000"
        );
        assert_eq!(
            tetration::slog_str("20", re, im, re, im).unwrap().0,
            "1.0000000000000000000"
        );
    }
}

#[test]
fn t1701_slog_round_trips() {
    let digits = 25u64;
    let prec = cnum::digits_to_bits(digits);
    let high = cnum::digits_to_bits(HIGH_DIGITS);
    for (base, heights) in [
        (
            ("2", "0"),
            vec![
                ("0.5", "0"),
                ("-0.5", "0"),
                ("2.75", "0"),
                ("-1.5", "0"),
                ("0.3", "0.4"),
                ("1.2", "-0.7"),
            ],
        ),
        (
            ("10", "0"),
            vec![("0.25", "0"), ("1.5", "0"), ("-1.9", "0"), ("-0.4", "0.1")],
        ),
        (
            ("1.3", "0"),
            vec![("0.5", "0"), ("-0.5", "0"), ("3.25", "0"), ("0.3", "0.4")],
        ),
        (
            (
                "1.41421356237309504880168872420969807856967187537694807317667973799",
                "0",
            ),
            vec![("0.5", "0"), ("7.5", "0")],
        ),
        (
            ("0.5", "0"),
            vec![("0.5", "0"), ("-0.25", "0"), ("0.2", "0.3")],
        ),
        (("1.3", "0.1"), vec![("0.5", "0"), ("0.3", "0.4")]),
        (("0", "1"), vec![("0.5", "0"), ("0.25", "-0.5")]),
        (("0.3", "-0.2"), vec![("0.4", "0.1")]),
    ] {
        for height in heights {
            let value = forward(base, height);
            let found = inverse::slog(&parse(base.0, base.1, high), &value, high, digits).unwrap();
            let expected = parse(height.0, height.1, prec);
            assert!(
                agreement(&found, &expected, high) >= digits,
                "b={base:?} h={height:?}: {} digits",
                cnum::DisplayFloat(&agreement(&found, &expected, high))
            );
        }
    }
}

#[test]
fn t1702_slog_conventions_and_errors() {
    let digits = 20u64;
    let prec = cnum::digits_to_bits(digits);
    // sqrt(2): real below the fixed point 2, complex with Im = pi/ln(ln 2) between 2 and 4.
    let root_two = parse(
        "1.4142135623730950488016887242096980785696718753769",
        "0",
        prec,
    );
    let below = inverse::slog(&root_two, &parse("1.5", "0", prec), prec, digits).unwrap();
    assert!(below.imag().is_zero() && *below.real() > 1 && *below.real() < 2);
    let between = inverse::slog(&root_two, &parse("3", "0", prec), prec, digits).unwrap();
    let pi = Float::with_val_64(prec, rug::float::Constant::Pi);
    let expected = pi / Float::with_val_64(prec, 2).ln().ln();
    assert!(
        Float::with_val_64(prec, between.imag() - &expected).abs()
            < cnum::epsilon(digits - 2, prec)
    );
    let outside = inverse::slog(&root_two, &parse("5", "0", prec), prec, digits).unwrap_err();
    assert!(outside.contains("basin"), "{outside}");
    for base in ["0", "1"] {
        let error = inverse::slog(
            &parse(base, "0", prec),
            &parse("2", "0", prec),
            prec,
            digits,
        )
        .unwrap_err();
        assert!(error.contains("undefined"), "{error}");
    }
    // Reflection in the base.
    let upper = inverse::slog(
        &parse("0.4", "0.3", prec),
        &parse("0.6", "0.2", prec),
        prec,
        digits,
    )
    .unwrap();
    let lower = inverse::slog(
        &parse("0.4", "-0.3", prec),
        &parse("0.6", "-0.2", prec),
        prec,
        digits,
    )
    .unwrap();
    assert_eq!(lower, upper.conj());
    // Real base above eta: real targets give real heights, monotonically.
    let two = parse("2", "0", prec);
    let mut previous = Float::with_val_64(prec, -2.5);
    for target in [
        "-1000", "-30", "-3", "1e-30", "0.001", "0.999", "3", "1e10", "1e1000",
    ] {
        let height = inverse::slog(&two, &parse(target, "0", prec), prec, digits).unwrap();
        assert!(height.imag().is_zero(), "z={target}");
        assert!(*height.real() > previous, "z={target}");
        previous = height.real().clone();
    }
    // Far below the range of F on (-2, -1) the height rounds to the singular -2.
    assert_eq!(
        tetration::slog_str("20", "2", "0", "-1000", "0").unwrap().0,
        "-2.0000000000000000000"
    );
}

#[test]
fn t1703_sroot_values() {
    assert_eq!(
        tetration::sroot_str("30", "16", "0", "3", "0").unwrap(),
        (
            "2.00000000000000000000000000000".to_string(),
            "0".to_string()
        )
    );
    assert_eq!(
        tetration::sroot_str("30", "65536", "0", "4", "0")
            .unwrap()
            .0,
        "2.00000000000000000000000000000"
    );
    assert_eq!(
        tetration::sroot_str("20", "7", "0.5", "1", "0").unwrap().0,
        "7.0000000000000000000"
    );
    assert!(tetration::sroot_str("20", "7", "0", "0", "0").is_err());
    assert!(tetration::sroot_str("20", "7", "0", "-1", "0").is_err());

    // Super square root: b^b = z has b = exp(W(ln z)).
    let digits = 40u64;
    let prec = cnum::digits_to_bits(digits);
    for target in ["2", "10", "1.5", "1000000"] {
        let z = parse(target, "0", prec);
        let w = lambertw::w0(&cnum::ln_complex(&z, prec), prec).unwrap();
        let expected = cnum::checked_exp(&w, prec).unwrap();
        let found = inverse::sroot(&z, &parse("2", "0", prec), prec, digits).unwrap();
        assert!(found.imag().is_zero());
        assert!(agreement(&found, &expected, prec) >= digits, "z={target}");
    }
    // Complex target at an integer height.
    let z = parse("1.2", "0.3", prec);
    let found = inverse::sroot(&z, &parse("2", "0", prec), prec, digits).unwrap();
    let image = cnum::pow_complex(&found, &found, prec);
    assert!(agreement(&image, &z, prec) >= digits - 1);
}

#[test]
fn t1704_sroot_round_trips() {
    let digits = 20u64;
    let high = cnum::digits_to_bits(HIGH_DIGITS);
    for (base, height) in [
        ("2", "2.5"),
        ("1.3", "0.5"),
        ("1.2", "3.75"),
        ("3", "0.5"),
        ("1.6", "1.25"),
    ] {
        let value = forward((base, "0"), (height, "0"));
        let found = inverse::sroot(&value, &parse(height, "0", high), high, digits).unwrap();
        let expected = parse(base, "0", high);
        assert!(
            agreement(&found, &expected, high) >= digits,
            "b={base} h={height}: {} digits",
            cnum::DisplayFloat(&agreement(&found, &expected, high))
        );
    }
    // A complex solution reached by continuation inside the regular region.
    let value = forward(("1.25", "0.05"), ("0.5", "0.1"));
    let found = inverse::sroot(&value, &parse("0.5", "0.1", high), high, digits).unwrap();
    assert!(agreement(&found, &parse("1.25", "0.05", high), high) >= digits);
}

#[test]
fn t1706_real_axis_branch_does_not_depend_on_the_digit_count() {
    // sigma(z)/sigma(1) is a negative real: Im Log is exactly pi, whatever
    // the rounding of pi * steps does at each precision.
    for digits in ["13", "14", "15", "16", "21", "22"] {
        let (re, im) = tetration::slog_str(digits, "0.5", "0", "0.2", "0").unwrap();
        assert!(re.starts_with("9.0704074230"), "{digits} digits: {re}");
        assert!(im.starts_with("-3.6012143413"), "{digits} digits: {im}");
    }
    for digits in ["15", "16"] {
        let (re, im) = tetration::slog_str(digits, "0.3", "0", "-3", "0").unwrap();
        assert!(re.starts_with("9.1915119353"), "{digits} digits: {re}");
        assert!(im.starts_with("-5.6290203086"), "{digits} digits: {im}");
    }
}

#[test]
fn t1707_ill_conditioned_inverses_keep_their_digits() {
    // F_b(1e-40) = 1 + 4.5e-41: the base is fixed by the ratio of two tiny numbers.
    let value = "1.000000000000000000000000000000000000000045";
    let height = "0.0000000000000000000000000000000000000001";
    assert_eq!(
        tetration::sroot_str("15", value, "0", height, "0")
            .unwrap()
            .0,
        "1.25365288039857"
    );
    assert_eq!(
        tetration::sroot_str("30", value, "0", height, "0")
            .unwrap()
            .0,
        "1.25365288039857303498422460940"
    );
    // Just above the fold of b^b at b = 1/e: a 1e-200 change of the target
    // must not move the printed digits.
    let fold = "0.69220062755534635386542199718278976149067802929754473593891489996517155902908537";
    let padded = format!("{fold}{}1", "0".repeat(120));
    let root = tetration::sroot_str("40", fold, "0", "2", "0").unwrap();
    assert_eq!(root.0, "3.678794411714423215955237701614608674459e-1");
    assert_eq!(
        tetration::sroot_str("40", &padded, "0", "2", "0").unwrap(),
        root
    );
    // A target within 2e-46 of the fixed point of base 1/2.
    let near = "0.641185744504985984486200482114823666562820957";
    let padded = format!("{near}{}1", "0".repeat(58));
    let height = tetration::slog_str("40", "0.5", "0", near, "0").unwrap();
    assert!(
        height.0.starts_with("8.9611698384629244756755925649"),
        "{}",
        height.0
    );
    assert_eq!(
        tetration::slog_str("40", "0.5", "0", &padded, "0").unwrap(),
        height
    );
}

#[test]
fn t1708_real_targets_far_from_the_origin() {
    for (target, expected) in [
        ("-3", "-1.89844554507553"),
        ("-130", "-2.00000000000000"),
        ("-165", "-2.00000000000000"),
        ("-1e30", "-2.00000000000000"),
        ("1e100", "4.65794468978138"),
        ("1e1000000000000", "5.30990630346441"),
        ("1e1000000000000000", "5.36086261591029"),
    ] {
        assert_eq!(
            tetration::slog_str("15", "2", "0", target, "0").unwrap(),
            (expected.to_string(), "0".to_string()),
            "target {target}"
        );
    }
    assert_eq!(
        tetration::slog_str("15", "10", "0", "-45", "0").unwrap().0,
        "-2.00000000000000"
    );
}

#[test]
fn t1709_slog_of_cut_bases_long_paths_and_detours() {
    let wide_digits = 30u64;
    let wide = cnum::digits_to_bits(wide_digits);
    for (base, target, expected, round_trip) in [
        // A real fixed point of the cut base lies on the segment 1 -> z; the
        // limit of bases in the upper half-plane passes below it.
        (
            ("0.04", "0"),
            "0.1",
            ("-1.01343950923799", "1.34822863318788e-1"),
            12,
        ),
        (
            ("0.04", "0"),
            "-0.5",
            ("-9.10477245538674e-1", "-3.83301303043979e-1"),
            12,
        ),
        // The straight path meets an image of the cut of the tetration.
        (
            ("0", "3"),
            "-3",
            ("9.06956866310467e-1", "-1.69518476458942"),
            12,
        ),
        // A long path is traversed geometrically.
        (
            ("3", "4"),
            "1e100",
            ("2.47017034675853", "-5.13254077845973e-1"),
            9,
        ),
    ] {
        let found = tetration::slog_str("15", base.0, base.1, target, "0").unwrap();
        assert_eq!(
            (found.0.as_str(), found.1.as_str()),
            expected,
            "b={base:?} z={target}"
        );
        let image = dispatch::tetrate(
            &parse(base.0, base.1, wide),
            &parse(&found.0, &found.1, wide),
            wide,
            wide_digits,
        )
        .unwrap();
        assert!(
            agreement(&image, &parse(target, "0", wide), wide) >= round_trip,
            "b={base:?} z={target}: {image}"
        );
    }
}

#[test]
fn t1710_slog_next_to_the_singular_heights_of_regular_bases() {
    // The orbit of the target passes 0 within the working precision: the
    // height rounds to a singular height, with the convention of its
    // neighbours (-30 is still verified by forward evaluation).
    let limit = (
        "7.50119449840629e-1".to_string(),
        "-9.68029912394000e-1".to_string(),
    );
    for target in ["-30", "-60", "-100"] {
        assert_eq!(
            tetration::slog_str("15", "0.5", "0", target, "0").unwrap(),
            limit,
            "z={target}"
        );
    }
    assert_eq!(
        tetration::slog_str("15", "1.3", "0", "-1e30", "0")
            .unwrap()
            .0,
        "-2.00000000000000"
    );
    // Off the principal branch there is no such height.
    let error = tetration::slog_str("15", "0.5", "0.5", "-100", "0").unwrap_err();
    assert!(error.contains("principal branch"), "{error}");
}

#[test]
fn t1711_sroot_at_negative_heights_and_complex_targets() {
    // As b grows, F_b(h) falls from 1 to 0 on -1 < h < 0 and rises from
    // -inf to 0 on -2 < h < -1.
    for (value, height, expected) in [
        ("0.5", "-0.5", "2.68778856702334"),
        ("-1", "-1.5", "1.75033148344121"),
    ] {
        assert_eq!(
            tetration::sroot_str("15", value, "0", height, "0").unwrap(),
            (expected.to_string(), "0".to_string()),
            "z={value} h={height}"
        );
    }
    let digits = 20u64;
    let high = cnum::digits_to_bits(HIGH_DIGITS);
    for (base, height) in [("2", "-0.5"), ("3", "-1.5")] {
        let value = forward((base, "0"), (height, "0"));
        let found = inverse::sroot(&value, &parse(height, "0", high), high, digits).unwrap();
        assert!(
            agreement(&found, &parse(base, "0", high), high) >= digits,
            "b={base} h={height}: {found}"
        );
    }
    // A complex target: the path is followed at reduced accuracy and its end
    // polished at full precision.
    let found = tetration::sroot_str("15", "1", "1", "0.5", "0").unwrap();
    assert_eq!(
        (found.0.as_str(), found.1.as_str()),
        ("2.98515676693044e-1", "1.76387940838634")
    );
    let wide = cnum::digits_to_bits(30);
    let image = dispatch::tetrate(
        &parse(&found.0, &found.1, wide),
        &parse("0.5", "0", wide),
        wide,
        30,
    )
    .unwrap();
    assert!(
        agreement(&image, &parse("1", "1", wide), wide) >= 13,
        "{image}"
    );
}

#[test]
fn t1705_cli_flags() {
    let run = |arguments: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_tet"))
            .args(arguments)
            .env("SILENT", "1")
            .output()
            .unwrap()
    };
    let text = |arguments: &[&str]| {
        let output = run(arguments);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap()
    };
    assert_eq!(
        text(&["20", "2", "0", "3", "0"]),
        "16.000000000000000000\n0\n"
    );
    for flag in ["--slog", "--ate"] {
        assert_eq!(
            text(&[flag, "20", "2", "0", "16", "0"]),
            "3.0000000000000000000\n0\n"
        );
        assert_eq!(
            text(&["-q", "20", "2", "0", "16", "0", flag]),
            "3.0000000000000000000\n0\n"
        );
    }
    for flag in ["--sroot", "--ssrt"] {
        assert_eq!(
            text(&[flag, "20", "16", "0", "3", "0"]),
            "2.0000000000000000000\n0\n"
        );
    }
    assert_eq!(
        run(&["--slog", "--sroot", "20", "2", "0", "16", "0"])
            .status
            .code(),
        Some(2)
    );
    assert_eq!(
        run(&["--slog", "20", "2", "0", "16"]).status.code(),
        Some(2)
    );
    let failure = run(&["--slog", "20", "1", "0", "2", "0"]);
    assert_eq!(failure.status.code(), Some(1));
    assert!(failure.stdout.is_empty());
    assert!(String::from_utf8_lossy(&failure.stderr).contains("undefined"));
}
