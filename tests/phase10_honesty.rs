use std::process::Command;

use rug::{float::Special, Complex, Float};
use tetration::{
    cnum, dispatch, integer_height, kouznetsov, lambertw, regions, schroder, tetrate_str,
};

fn parse(re: &str, im: &str, prec: u64) -> Complex {
    cnum::parse_complex(re, im, prec).unwrap()
}

fn assert_close(actual: &Complex, expected: &Complex, digits: u64, prec: u64) {
    assert!(actual.real().is_finite() && actual.imag().is_finite());
    assert!(expected.real().is_finite() && expected.imag().is_finite());
    let error = Float::with_val_64(
        prec,
        Complex::with_val_64(prec, actual - expected).abs_ref(),
    );
    let scale = Float::with_val_64(prec, expected.abs_ref()).max(&Float::with_val_64(prec, 1));
    let tolerance = Float::with_val_64(prec, Float::parse(format!("1e-{digits}")).unwrap()) * scale;
    assert!(
        error < tolerance,
        "error {error} exceeds {tolerance}: {actual} vs {expected}"
    );
}

#[test]
fn nonfinite_strings_are_not_complex_inputs() {
    for bad in ["NaN", "nan", "inf", "-inf", "Infinity"] {
        assert!(tetrate_str("50", bad, "0", "0", "0").is_err(), "base {bad}");
        assert!(
            tetrate_str("50", "1", "0", bad, "0").is_err(),
            "height {bad}"
        );
        assert!(
            tetrate_str("50", "1", bad, "0", "0").is_err(),
            "imaginary base {bad}"
        );
        assert!(
            tetrate_str("50", "1", "0", "0", bad).is_err(),
            "imaginary height {bad}"
        );
    }
}

#[test]
fn direct_apis_reject_nonfinite_values() {
    let prec = cnum::digits_to_bits(50);
    let zero = cnum::zero(prec);
    let one = cnum::one(prec);
    for special in [Special::Nan, Special::Infinity, Special::NegInfinity] {
        let bad = Complex::with_val_64(prec, (Float::with_val_64(prec, special), 0));
        assert!(dispatch::tetrate(&bad, &zero, prec, 50).is_err());
        assert!(dispatch::tetrate(&one, &bad, prec, 50).is_err());
        assert!(integer_height::tetrate_integer(&bad, 0, prec).is_err());
        assert!(lambertw::w0(&bad, prec).is_err());
        assert!(lambertw::wm1(&bad, prec).is_err());
        assert!(lambertw::wk(&bad, 2, prec).is_err());
        assert!(regions::classify(&bad, prec).is_err());
    }
}

#[test]
fn integer_towers_reject_overflow_and_underflow() {
    for digits in [1, 10, 50, 70, 1000] {
        let prec = cnum::digits_to_bits(digits);
        let two = parse("2", "0", prec);
        for n in [6, 7] {
            assert!(
                integer_height::tetrate_integer(&two, n, prec).is_err(),
                "{digits} digits, n={n}"
            );
        }
        let tiny_power = parse("-1e1000", "0", prec);
        assert!(integer_height::tetrate_integer(&tiny_power, 2, prec).is_err());
    }
}

#[test]
fn attracting_bases_never_get_the_two_fixed_point_function() {
    // Where the fixed point attracts, regular iteration defines b^^h: when it
    // cannot deliver a height, the Cauchy construction (which converges to
    // the two-fixed-point function there) must not answer in its place.
    let error =
        tetrate_str("15", "1.3", "0", "-1000000000000000000000000000000.5", "0").unwrap_err();
    assert!(error.contains("regular iteration failed"), "{error}");
    assert!(error.contains("not substituted"), "{error}");
}

#[test]
fn finite_numbers_outside_f64_range_are_preserved() {
    let prec = cnum::digits_to_bits(50);
    for re in ["1e1000", "1e-1000", "-1e1000"] {
        let b = parse(re, "0", prec);
        let got = integer_height::tetrate_integer(&b, 1, prec).unwrap();
        assert_eq!(got, b, "F_b(1) must preserve the base exactly");
    }
}

#[test]
fn mpfr_limits_reject_unrepresentable_precision_before_allocation() {
    let error = cnum::checked_digits_to_bits(u64::MAX).unwrap_err();
    assert!(error.contains("MPFR"), "{error}");
    let error = cnum::require_precision(rug::float::prec_max_64() + 1, 1).unwrap_err();
    assert!(error.contains("native precision range"), "{error}");
}

#[test]
fn mpfr_limits_reject_single_component_exponential_underflow() {
    let prec = cnum::digits_to_bits(70);
    let real =
        Float::with_val_64(prec, cnum::exponent_range().0) * Float::with_val_64(prec, 2).ln();
    let real_argument = Complex::with_val_64(prec, (real.clone(), 0));
    let real_value = cnum::checked_exp(&real_argument, prec).unwrap();
    assert!(real_value.real().is_finite() && !real_value.real().is_zero());
    assert!(real_value.imag().is_zero());
    for imaginary in ["0.1", "-0.1"] {
        let argument = Complex::with_val_64(prec, (real.clone(), cnum::decimal(imaginary, prec)));
        let error = cnum::checked_exp(&argument, prec).unwrap_err();
        assert!(error.contains("underflow"), "{error}");
    }
}

#[test]
fn integer_path_agrees_with_independent_root_at_all_precisions() {
    for digits in [1, 2, 10, 50, 70, 1000] {
        let prec = cnum::digits_to_bits(digits);
        let b = parse("0.5", "0", prec);
        let got = integer_height::tetrate_integer(&b, 2, prec).unwrap();
        let expected = Complex::with_val_64(prec, (Float::with_val_64(prec, 2).sqrt().recip(), 0));
        assert_close(&got, &expected, digits, prec);
    }
}

#[test]
fn negative_one_integer_towers_do_not_amplify_roundoff() {
    for digits in [50, 70, 1000, 1, 10] {
        let prec = cnum::digits_to_bits(digits);
        let base = parse("-1", "0", prec);
        for n in [100, 1, 2, 100_000, 100_001, i64::MAX] {
            let value = integer_height::tetrate_integer(&base, n, prec).unwrap();
            assert_eq!(
                value, base,
                "(-1)^^{n} must stay exactly -1 at {digits} digits"
            );
        }
        assert_eq!(
            integer_height::tetrate_integer(&base, 0, prec).unwrap(),
            cnum::one(prec)
        );
        assert_eq!(
            integer_height::tetrate_integer(&base, -1, prec).unwrap(),
            cnum::zero(prec)
        );
        assert!(integer_height::tetrate_integer(&base, -2, prec)
            .unwrap_err()
            .contains("undefined"));
    }
}

#[test]
fn string_inputs_preserve_near_one_base_identity() {
    for digits in [50u64, 70, 1000, 1, 10] {
        let zeros = "0".repeat((digits + 100) as usize);
        let base = format!("1.{zeros}1");
        for input in [
            base.clone(),
            format!("{base}e0"),
            format!("{base}E+0"),
            format!("1{zeros}1@-{}", zeros.len() + 1),
        ] {
            let result = tetrate_str(&digits.to_string(), &input, "0", "-1", "0").unwrap();
            assert_eq!(result, ("0".into(), "0".into()), "{digits} digits");
        }
    }
}

#[test]
fn string_inputs_preserve_large_integer_height_parity() {
    for digits in [50u64, 70, 1000, 1, 10] {
        let zeros = "0".repeat((digits + 100) as usize);
        let height = format!("1{zeros}1");
        for suffix in ["", "e0", "E+0", "@0"] {
            let input = format!("{height}{suffix}");
            let result = tetrate_str(&digits.to_string(), "0", "0", &input, "0").unwrap();
            assert_eq!(result, ("0".into(), "0".into()), "{digits} digits");
        }
    }
}

#[test]
fn string_inputs_do_not_round_noninteger_heights_into_the_domain() {
    for digits in [50u64, 70, 1000, 1, 10] {
        let zeros = "0".repeat((digits + 100) as usize);
        let height = format!("2.{zeros}1");
        for suffix in ["", "e0", "E+0", "@0"] {
            let input = format!("{height}{suffix}");
            let error = tetrate_str(&digits.to_string(), "0", "0", &input, "0").unwrap_err();
            assert_eq!(
                error,
                "tetration of 0 is only defined for non-negative integer heights"
            );
        }
    }
}

#[test]
fn string_inputs_size_precision_without_counting_padding_or_exponents() {
    let requested = cnum::digits_to_bits(50);
    for input in [
        "0",
        "-0.000e1000",
        "+0001.00000",
        "1e1000",
        "1E-1000",
        "1@+1000",
        "100000000000000000000000000000000000000000000000000000000000",
        "0.000000000000000000000000000000000000000000000000000000000001",
    ] {
        assert_eq!(
            cnum::checked_input_precision(50, &[input]).unwrap(),
            requested
        );
    }
    assert_eq!(cnum::checked_input_precision(50, &[]).unwrap(), requested);
    let significant = format!("1{}2", "0".repeat(99));
    let padded = format!("+000.{significant}00000E-1000");
    for index in 0..4 {
        let mut inputs = ["0"; 4];
        inputs[index] = &padded;
        assert_eq!(
            cnum::checked_input_precision(50, &inputs).unwrap(),
            cnum::digits_to_bits(101)
        );
        assert_eq!(
            cnum::checked_input_precision(1000, &inputs).unwrap(),
            cnum::digits_to_bits(1000)
        );
    }
    let error = tetrate_str("0", "bad", "0", "0", "0").unwrap_err();
    assert!(error.starts_with("precision must be positive"), "{error}");
    let error = tetrate_str("50", "bad", "0", "0", "0").unwrap_err();
    assert!(error.starts_with("invalid number"), "{error}");
}

#[test]
fn regular_tetration_matches_independent_orbit_references() {
    // Independent 260-dps mpmath forward-orbit/log-unwinding limits at two
    // depths agree beyond 85 digits; these are numerical references, not proofs.
    let cases = [
        ("1.2", "0", "0.4", "0.2",
         "1.123041284712457633959705037767083499411825618006021958828151581998267777889473716559655656944533088",
         "0.0415780231567696201806234813913195653743527870708176758832892869774307737090146786913271017798960804"),
        ("0.5", "0", "0.5", "0",
         "0.6297862283961248719630625995270858162215811299326939762118616269805828949253589192160869360320315197",
         "0.2178618631250840283749249956377183445680226015543687268045008416490672957765003972384445665729990976"),
        ("1.3", "0.1", "0.5", "0",
         "1.191980303617951340789425900782955071500230393724753768954209891574713250302556878339282890706609406",
         "0.04995444900678871179128776367335578855200561261753461869084981582608475523527791164540169651419883399"),
        ("1.414213562373095048801688724209698078569671875376948073176679737990732478462107038850387534327641572735013846230912297024924836056",
         "0", "0.5", "0",
         "1.243621627668521804295098983609402931688198356615523786426384266739854041905083929661122492605564522",
         "0"),
        ("0", "1", "0.5", "0",
         "1.166700913570474569302791134642004482190693021035846011735709811377731613258125406699631416904801441",
         "0.7345635369867213296594900876244841035951645261032671807793422871560665728962293309214211653770510143"),
        ("0.06532815548685941170602228572087796068919127777026847203844880615106896673044874502304736002883775306821851270780321205611544847999",
         "0.05", "0.5", "0",
         "0.1689649492331378970351211219479528781425282760799041330892265221087495696674286292130752597917072094",
         "0.2696467237440392147737159824632838291186366645664904251455452962553422438579608115287666385045057096"),
    ];
    for digits in [1, 10, 50, 70] {
        let prec = cnum::digits_to_bits(digits);
        for (br, bi, hr, hi, fr, fi) in cases {
            let base = parse(br, bi, prec);
            let height = parse(hr, hi, prec);
            let expected = parse(fr, fi, prec);
            let actual = dispatch::tetrate(&base, &height, prec, digits).unwrap();
            assert_close(&actual, &expected, digits, prec);
            if digits == 50 && base.imag() > &0 {
                let lower = dispatch::tetrate(&base.conj(), &height.conj(), prec, digits).unwrap();
                assert_close(&lower, &expected.conj(), digits, prec);
            }
        }
    }
}

#[test]
fn lambert_near_branch_point_retains_requested_digits() {
    for digits in [50, 70, 1000] {
        let prec = cnum::digits_to_bits(digits);
        let reference_prec = prec * 2 + 64;
        for re in [
            "-0.3678794411714423215955237701614608674458111310317678345078368016974614957448",
            "-0.3678794411714423215955237701614608674458111310317678345078368016974614957450",
        ] {
            let z = parse(re, "0", prec);
            let exact_same_input = Complex::with_val_64(reference_prec, &z);
            for branch in [0, -1] {
                let actual = lambertw::wk(&z, branch, prec).unwrap();
                let reference = lambertw::wk(&exact_same_input, branch, reference_prec).unwrap();
                assert_close(
                    &Complex::with_val_64(reference_prec, actual),
                    &reference,
                    digits,
                    reference_prec,
                );
            }
        }
    }
}

#[test]
fn lambert_lower_cut_branch_matches_independent_reference() {
    // mpmath 1.4.1, 85 decimal digits; this checks branch identity, not only w*exp(w)=z.
    let prec = cnum::digits_to_bits(70);
    let cases = [
        (
            "-0.2",
            "-1e-10",
            "-3.72232048486357370717711524712061662866199694473706785525726979723485561165",
            "-7.3872302100526325804172739749758857080423583138401074101874340810400984943",
        ),
        (
            "-0.4",
            "-0.1",
            "-2.93982462896246683790604090331543112378968233064576777231186214144402668454",
            "-7.22244219042365932676409940101421020592833533585341887368989878902407776869",
        ),
    ];
    for (re, im, wr, wi) in cases {
        let actual = lambertw::wm1(&parse(re, im, prec), prec).unwrap();
        assert_close(&actual, &parse(wr, wi, prec), 65, prec);
    }
}

#[test]
fn lambert_extreme_finite_arguments_do_not_use_machine_seeds() {
    let prec = cnum::digits_to_bits(70);
    let small = lambertw::wm1(&parse("-1e-1000", "0", prec), prec).unwrap();
    assert_close(
        &small,
        &parse(
            "-2310.33023874784054589842688636390557370980554563492849160234910889443238797",
            "0",
            prec,
        ),
        65,
        prec,
    );
    let large = lambertw::wk(&parse("1e1000", "0", prec), 1, prec).unwrap();
    assert_close(
        &large,
        &parse(
            "2294.84666794021966347478055808813704067888452305363158968018893160367149476",
            "6.28044855227557381288270879149266268332385978523396121696449531059621688352",
            prec,
        ),
        65,
        prec,
    );
}

#[test]
fn quiet_flags_preserve_numeric_output_and_fatal_errors() {
    for flag in ["--quiet", "--silent", "-q"] {
        let out = Command::new(env!("CARGO_BIN_EXE_tet"))
            .args([flag, "50", "0.5", "0", "2", "0"])
            .env_remove("SILENT")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{flag}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(out.stderr.is_empty());
        assert_eq!(String::from_utf8_lossy(&out.stdout).lines().count(), 2);
        let bad = Command::new(env!("CARGO_BIN_EXE_tet"))
            .args([flag, "50", "NaN", "0", "0", "0"])
            .output()
            .unwrap();
        assert_eq!(bad.status.code(), Some(1));
        assert!(bad.stdout.is_empty());
        assert!(String::from_utf8_lossy(&bad.stderr).contains("error:"));
    }
}

#[test]
fn thousand_digit_tolerances_are_finite_and_nonzero() {
    for digits in [1, 10, 50, 70, 1000] {
        let prec = cnum::digits_to_bits(digits);
        let epsilon = cnum::epsilon(digits, prec);
        assert!(epsilon.is_finite() && epsilon > 0 && epsilon < 1);
        let expected = cnum::decimal(&format!("1e-{digits}"), prec);
        assert!(
            (epsilon.clone() - expected.clone()).abs() <= cnum::working_epsilon(prec) * expected
        );
        assert!(cnum::working_epsilon(prec) > 0);
        assert!(cnum::working_epsilon(prec) < epsilon);
    }
}

#[test]
fn region_boundary_distinguishes_beyond_machine_precision() {
    let prec = cnum::digits_to_bits(70);
    let threshold = cnum::decimal(regions::SHELL_THRON_INTERIOR_THRESHOLD, prec);
    let delta = cnum::epsilon(60, prec);
    for sign in [-1i32, 1] {
        let lambda = threshold.clone() + delta.clone() * sign;
        let b = Complex::with_val_64(prec, (lambda.clone() * (-lambda).exp()).exp());
        let region = regions::classify(&b, prec).unwrap();
        assert_eq!(
            matches!(region, regions::Region::ShellThronInterior(_)),
            sign < 0
        );
        assert_eq!(
            matches!(region, regions::Region::ShellThronBoundary(_)),
            sign > 0
        );
    }
}

#[test]
fn degenerate_base_parity_supports_arbitrary_integer_heights() {
    let prec = cnum::digits_to_bits(70);
    for (height, expected) in [
        (
            "1000000000000000000000000000000000000000000000000000000000000",
            1,
        ),
        (
            "1000000000000000000000000000000000000000000000000000000000001",
            0,
        ),
    ] {
        let h = parse(height, "0", prec);
        assert_eq!(
            dispatch::tetrate(&cnum::zero(prec), &h, prec, 70).unwrap(),
            Complex::with_val_64(prec, expected)
        );
        assert_eq!(
            dispatch::tetrate(&cnum::one(prec), &h, prec, 70).unwrap(),
            cnum::one(prec)
        );
    }
}

#[test]
fn kouznetsov_normalization_checks_the_returned_function() {
    let digits = 10;
    let prec = cnum::digits_to_bits(digits);
    let base = parse("2", "0", prec);
    let regions::Region::OutsideShellThronRealPositive(fp) =
        regions::classify(&base, prec).unwrap()
    else {
        panic!("unexpected base-2 region");
    };
    let state = kouznetsov::setup_kouznetsov(&base, &fp, prec, digits).unwrap();
    for (height, expected) in [("0", "1"), ("1", "2"), ("-1", "0")] {
        let actual = kouznetsov::eval_kouznetsov(&state, &base, &parse(height, "0", prec)).unwrap();
        assert_close(&actual, &parse(expected, "0", prec), digits, prec);
        if height == "-1" {
            assert!(cnum::is_zero(&actual));
        }
    }
}

#[test]
fn cached_evaluators_reject_negative_integer_singularities() {
    for digits in [10, 50, 70] {
        let prec = cnum::digits_to_bits(digits);
        for real in ["1.2", "0.5"] {
            let base = parse(real, "0", prec);
            let regions::Region::ShellThronInterior(fp) = regions::classify(&base, prec).unwrap()
            else {
                panic!("expected attracting base");
            };
            let state = schroder::setup_schroder(&base, &fp, prec).unwrap();
            for height in ["-2", "-3"] {
                let result = schroder::eval_schroder(&state, &parse(height, "0", prec));
                assert!(
                    result.is_err(),
                    "cached base {real}, height {height}: {result:?}"
                );
            }
        }
    }
    let digits = 10;
    let prec = cnum::digits_to_bits(digits);
    let base = parse("2", "0", prec);
    let regions::Region::OutsideShellThronRealPositive(fp) =
        regions::classify(&base, prec).unwrap()
    else {
        panic!("unexpected base-2 region");
    };
    let state = kouznetsov::setup_kouznetsov(&base, &fp, prec, digits).unwrap();
    for height in ["-2", "-3"] {
        let result = kouznetsov::eval_kouznetsov(&state, &base, &parse(height, "0", prec));
        assert!(
            result.is_err(),
            "cached base 2, height {height}: {result:?}"
        );
    }
}

#[test]
fn inconsistent_precision_requests_are_errors() {
    let prec = cnum::digits_to_bits(50);
    let one = cnum::one(prec);
    let height = parse("0.5", "0", prec);
    for digits in [0, 1000, u64::MAX] {
        assert!(dispatch::tetrate(&one, &height, prec, digits).is_err());
    }
    for digits in [0, u64::MAX] {
        assert!(cnum::checked_digits_to_bits(digits).is_err());
    }
}

/// A source file without its comments, without the contents of its string
/// and character literals and without its test module.
fn production_code(source: &str) -> Vec<char> {
    let source: Vec<char> = source
        .split("#[cfg(test)]\nmod tests")
        .next()
        .unwrap_or_default()
        .chars()
        .collect();
    let mut code = Vec::with_capacity(source.len());
    let mut at = 0;
    while at < source.len() {
        let ahead = |offset: usize| source.get(at + offset).copied();
        match source[at] {
            '/' if ahead(1) == Some('/') => {
                while at < source.len() && source[at] != '\n' {
                    at += 1;
                }
            }
            '"' => {
                at += 1;
                while at < source.len() && source[at] != '"' {
                    let escape = source[at] == '\\';
                    at += usize::from(escape);
                    if source.get(at) == Some(&'\n') {
                        code.push('\n');
                    }
                    at += 1;
                }
                at += 1;
                code.push(' ');
            }
            '\'' if ahead(1) == Some('\\') => {
                at += 3;
                while at < source.len() && source[at] != '\'' {
                    at += 1;
                }
                at += 1;
                code.push(' ');
            }
            '\'' if ahead(2) == Some('\'') => {
                at += 3;
                code.push(' ');
            }
            other => {
                code.push(other);
                at += 1;
            }
        }
    }
    code
}

/// Machine-size numbers in `code`, with their lines: names of the machine
/// float types, floating-point literals, MPFR values or precisions given by
/// an integer literal, and constants that name a number of bits or digits.
fn machine_numbers(code: &[char]) -> Vec<(usize, String)> {
    let word = |c: char| c.is_alphanumeric() || c == '_';
    let digits = |mut at: usize| {
        while code
            .get(at)
            .is_some_and(|c| c.is_ascii_digit() || *c == '_')
        {
            at += 1;
        }
        at
    };
    let text = |from: usize, to: usize| code[from..to].iter().collect::<String>();
    let mut found = Vec::new();
    let mut line = 1;
    let mut at = 0;
    while at < code.len() {
        let start = at;
        let first = code[at];
        if first.is_ascii_digit() {
            let radix = first == '0' && matches!(code.get(at + 1), Some('x' | 'b' | 'o'));
            at = digits(at);
            let mut float = false;
            if !radix {
                let after = code.get(at + 1).copied().unwrap_or(' ');
                if code.get(at) == Some(&'.')
                    && after != '.'
                    && !after.is_alphabetic()
                    && after != '_'
                {
                    float = true;
                    at = digits(at + 1);
                }
                let sign = usize::from(matches!(code.get(at + 1), Some('+' | '-')));
                if matches!(code.get(at), Some('e' | 'E'))
                    && code.get(at + 1 + sign).is_some_and(|c| c.is_ascii_digit())
                {
                    float = true;
                    at = digits(at + 1 + sign);
                }
            }
            let number = at;
            while code.get(at).is_some_and(|c| word(*c)) {
                at += 1;
            }
            let field = start >= 2 && code[start - 1] == '.' && code[start - 2] != '.';
            if (float || code.get(number) == Some(&'f')) && !field {
                found.push((line, text(start, at)));
            }
        } else if first.is_alphabetic() || first == '_' {
            while code.get(at).is_some_and(|c| word(*c)) {
                at += 1;
            }
            let name = text(start, at);
            let before = text(start.saturating_sub(9), start);
            let sized = name.starts_with("with_val")
                || name == "digits_to_bits"
                || (name == "min" && before.ends_with("prec."))
                || (name.starts_with("new")
                    && (before.ends_with("Float::") || before.ends_with("Complex::")));
            if name.contains("f64") || name.contains("f32") {
                found.push((line, name));
            } else if name == "const" {
                let mut end = at + 1;
                while code.get(end).is_some_and(|c| word(*c)) {
                    end += 1;
                }
                let constant = text(at + 1, end);
                if constant.ends_with("_BITS") || constant.ends_with("_DIGITS") {
                    found.push((line, format!("const {constant}")));
                }
            } else if sized && code.get(at) == Some(&'(') {
                let mut argument = at + 1;
                while code.get(argument).is_some_and(|c| c.is_whitespace()) {
                    argument += 1;
                }
                let end = digits(argument);
                if end > argument && !code.get(end).is_some_and(|c| word(*c) || *c == '.') {
                    let owner = if name == "min" { "prec." } else { "" };
                    found.push((line, format!("{owner}{name}({}", text(argument, end))));
                }
            }
        } else {
            line += usize::from(first == '\n');
            at += 1;
        }
    }
    found
}

#[test]
fn machine_numbers_stay_out_of_the_arithmetic() {
    // README § 7.1 lists every machine float and every number of a fixed
    // size in `src/` and `examples/` with the reason why no digit depends on
    // it. A new one fails here until it is listed there and permitted below.
    let permitted = |file: &str, finding: &str| {
        matches!(finding, "0.25" | "0.5" | "1.5")
            || matches!(
                (file, finding),
                ("src/kouznetsov.rs", "as_secs_f64")
                    | ("src/kneser.rs", "with_val_64(64")
                    | ("src/kneser.rs" | "src/kneser_complex.rs", "prec.min(64")
                    | (
                        "src/kneser.rs",
                        "const GUARD_DIGITS" | "const FIRST_LEVEL_DIGITS"
                    )
                    | ("src/kneser_complex.rs", "const SCOUT_BITS")
                    | (
                        "src/inverse.rs",
                        "const LOOSE_BITS" | "const CHECK_DIGITS" | "const WALK_DIGITS"
                    )
                    | (
                        "examples/edge_runner.rs" | "examples/grid_runner.rs",
                        "as_secs_f64" | "f64" | "100.0"
                    )
                    | ("examples/test_w0.rs", "digits_to_bits(8")
            )
    };
    let sample = "let a = x.to_f64() * 1e-6 + 2.5f32; // 0.1 as f64\n\
                  let b = Float::with_val(53, \"0.1\") + pair.0 + (1..3).len() as u32;\n\
                  let c = prec.min(64) + 0x1e5 + 7.max(2) + 0.5 + '1' as u32;\n\
                  const START_BITS: u64 = 128; const ORDER: usize = 8;\n\
                  let d = Float::with_val_64(cnum::digits_to_bits(15), prec.min(limit));";
    let found: Vec<String> = machine_numbers(&production_code(sample))
        .into_iter()
        .filter(|(_, finding)| !permitted("src/sample.rs", finding))
        .map(|(line, finding)| format!("{line}: {finding}"))
        .collect();
    assert_eq!(
        found,
        [
            "1: to_f64",
            "1: 1e-6",
            "1: 2.5f32",
            "2: with_val(53",
            "3: prec.min(64",
            "4: const START_BITS",
            "5: digits_to_bits(15"
        ]
    );

    let mut unlisted = Vec::new();
    for directory in ["src", "examples"] {
        let root = format!("{}/{directory}", env!("CARGO_MANIFEST_DIR"));
        let mut paths: Vec<_> = std::fs::read_dir(root)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
            .collect();
        paths.sort();
        assert!(paths.len() > 2, "{paths:?}");
        for path in paths {
            let file = format!(
                "{directory}/{}",
                path.file_name().unwrap().to_string_lossy()
            );
            let code = production_code(&std::fs::read_to_string(&path).unwrap());
            for (line, finding) in machine_numbers(&code) {
                if !permitted(&file, &finding) {
                    unlisted.push(format!("{file}:{line}: {finding}"));
                }
            }
        }
    }
    assert!(
        unlisted.is_empty(),
        "machine-size numbers that README § 7.1 does not list:\n{}",
        unlisted.join("\n")
    );
}

#[test]
fn invalid_numerical_environment_overrides_are_errors() {
    for (name, value, anderson) in [
        ("TET_MT", "not-a-count", false),
        ("TET_MT", "18446744073709551615", false),
        ("TET_KOUZ_EM_K", "not-a-count", false),
        ("TET_KOUZ_EM_K", "18446744073709551615", false),
        ("TET_KOUZ_ANDERSON_DEPTH", "not-a-count", true),
        ("TET_KOUZ_RESID_DUMP", "", false),
    ] {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_tet"));
        // TET_CAUCHY selects the Cauchy solver, which reads these overrides.
        cmd.args(["--quiet", "10", "2", "0.1", "0.5", "0"])
            .env("TET_CAUCHY", "1")
            .env_remove("TET_KOUZ_NO_EM")
            .env_remove("TET_KOUZ_ANDERSON")
            .env_remove("TET_KOUZ_PICARD")
            .env(name, value);
        if anderson {
            cmd.env("TET_KOUZ_ANDERSON", "1");
        }
        let out = cmd.output().unwrap();
        assert_eq!(out.status.code(), Some(1), "{name}={value}: {:?}", out);
        assert!(out.stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&out.stderr).contains(name),
            "{name}: {:?}",
            out
        );
    }
}

#[test]
fn library_mt_respects_thread_counts_and_reports_conflicts() {
    if let Ok(mode) = std::env::var("TET_TEST_MT_CHILD") {
        if mode == "preinitialized" {
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build_global()
                .unwrap();
            assert!(tetration::mt::init_pool()
                .unwrap_err()
                .contains("Rayon pool"));
        } else if mode == "serial" {
            assert!(!tetration::mt::mt_enabled());
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build_global()
                .unwrap();
        } else {
            assert!(tetration::mt::mt_enabled());
            assert_eq!(rayon::current_num_threads(), mode.parse::<usize>().unwrap());
        }
        return;
    }
    for mode in ["2", "4", "preinitialized", "serial"] {
        let mut cmd = Command::new(std::env::current_exe().unwrap());
        cmd.args([
            "--exact",
            "library_mt_respects_thread_counts_and_reports_conflicts",
            "--nocapture",
        ])
        .env("TET_TEST_MT_CHILD", mode)
        .env("RAYON_NUM_THREADS", "1");
        if mode == "serial" {
            cmd.env_remove("TET_MT");
        } else {
            cmd.env("TET_MT", if mode == "preinitialized" { "4" } else { mode });
        }
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "{mode}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn serial_and_mt_cli_results_are_bit_identical() {
    for args in [
        ["50", "1.2", "0", "0.4", "0.2"],
        ["50", "1.3", "0.1", "0.5", "0"],
        ["10", "2", "0", "0.5", "0"],
    ] {
        let serial = Command::new(env!("CARGO_BIN_EXE_tet"))
            .arg("--quiet")
            .args(args)
            .env_remove("TET_MT")
            .output()
            .unwrap();
        let parallel = Command::new(env!("CARGO_BIN_EXE_tet"))
            .arg("--quiet")
            .args(args)
            .env("TET_MT", "4")
            .env("RAYON_NUM_THREADS", "1")
            .output()
            .unwrap();
        assert!(
            serial.status.success(),
            "serial: {}",
            String::from_utf8_lossy(&serial.stderr)
        );
        assert!(
            parallel.status.success(),
            "MT: {}",
            String::from_utf8_lossy(&parallel.stderr)
        );
        assert!(serial.stderr.is_empty() && parallel.stderr.is_empty());
        assert_eq!(serial.stdout, parallel.stdout, "{args:?}");
    }
}
