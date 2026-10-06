use rug::{Complex, Float};
use std::process::Command;
use tetration::{cnum, dispatch, lambertw, regions, schroder};

#[test]
fn native_scale_attractors_preserve_fractional_height_components() {
    let reference_prec = cnum::digits_to_bits(200);
    for exponent in [
        "1000",
        "10000000000000000",
        "100000000000000000",
        "1000000000000000000",
    ] {
        let imaginary = format!("1e-{exponent}");
        let base_reference = cnum::parse_complex("1", &imaginary, reference_prec).unwrap();
        let epsilon = cnum::decimal(&imaginary, reference_prec);
        let small = Float::with_val_64(reference_prec, &epsilon / 2).sqrt();
        let large = (Float::with_val_64(reference_prec, &epsilon * 2))
            .sqrt()
            .recip();
        // The omitted relative corrections are O(sqrt(epsilon)), below200 digits.
        let negative_half = Complex::with_val_64(reference_prec, (1, -&small));
        let negative_three_halves = Complex::with_val_64(reference_prec, (-&large, &large));
        let logarithm = Complex::with_val_64(reference_prec, base_reference.ln_ref());
        let mut negative_fourteen_half = negative_three_halves.clone();
        for _ in 0..13 {
            negative_fourteen_half =
                Complex::with_val_64(reference_prec, negative_fourteen_half.ln_ref()) / &logarithm;
        }
        for digits in [1, 5, 10, 20, 50, 70, 100] {
            let prec = cnum::digits_to_bits(digits);
            let base = cnum::parse_complex("1", &imaginary, prec).unwrap();
            let fp = match regions::classify(&base, prec).unwrap() {
                regions::Region::ShellThronInterior(fp) => fp,
                other => panic!("expected strict attraction, got {other:?}"),
            };
            let state = schroder::setup_schroder(&base, &fp, prec).unwrap();
            assert_eq!(
                state.sigma_inv.len(),
                2,
                "the analytic tail permits a linear germ"
            );
            for (height, expected) in [
                ("0.5", &base_reference),
                ("-0.5", &negative_half),
                ("-1.5", &negative_three_halves),
                ("-14.5", &negative_fourteen_half),
            ] {
                eprintln!("native cached: exponent={exponent}, height={height}, digits={digits}");
                let height_value = cnum::parse_complex(height, "0", prec).unwrap();
                let actual =
                    schroder::eval_schroder_at_digits(&state, &height_value, digits).unwrap();
                let expected = cnum::format_complex(expected, usize::try_from(digits).unwrap());
                assert_eq!(
                    cnum::format_complex(&actual, usize::try_from(digits).unwrap()),
                    expected,
                    "imaginary base {imaginary}, height {height}, digits {digits}"
                );
                if exponent == "1000000000000000000" {
                    for mt in [false, true] {
                        eprintln!("native CLI: height={height}, digits={digits}, MT={mt}");
                        let mut command = Command::new(env!("CARGO_BIN_EXE_tet"));
                        command.args([
                            "--quiet",
                            &digits.to_string(),
                            "1",
                            &imaginary,
                            height,
                            "0",
                        ]);
                        if mt {
                            command.env("TET_MT", "2").env("RAYON_NUM_THREADS", "2");
                        } else {
                            command.env_remove("TET_MT").env_remove("RAYON_NUM_THREADS");
                        }
                        let output = command.output().unwrap();
                        assert!(output.status.success(), "{output:?}");
                        assert!(output.stderr.is_empty(), "{output:?}");
                        assert_eq!(
                            output.stdout,
                            format!("{}\n{}\n", expected.0, expected.1).as_bytes(),
                            "height {height}, digits {digits}, MT={mt}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn regular_half_height_matches_independent_thousand_digit_reference() {
    // Independent2100-digit orbit/log runs at depths1941/1989 agree beyond1012 digits.
    let reference = cnum::decimal(
        "1.163595361965008849025943273358436312883942418749454480271873486455251493247827083781757429515488421248577610201192969547676535572801536796859839261917436210163140015665534388050464123305878782938625520248433676991624663877877940421132766526305383401561638952168002197943383598116805087826977506052221977786352192135603414369411675277349287133013211239765061863092667659017148048996217899480925801924436757910779313920239314781761991500860127100592809093489347158728982793049267979743705580317854155043056442760339422691332559497655449446017832569652847517805838078435357988346361590012432718993393121663522605280313506064437318633625781438284097946599501485258500986026662925744003419771004820435101522162686265862746307543233438203039660357678770791473296611768268024798758159411036127683028574352629692406076118923292175727813375963349993467059562988401712174937994958365422988498680424980997842602791261105880331989222791550153345463469269144728806459861918679664159494814282281044780553445508244823345224026494946124193340732367981835",
        cnum::digits_to_bits(1100),
    );
    let prec = cnum::digits_to_bits(1000);
    let base = cnum::parse_complex("1.25", "0", prec).unwrap();
    let height = cnum::parse_complex("0.5", "0", prec).unwrap();
    let actual = dispatch::tetrate(&base, &height, prec, 1000).unwrap();
    assert!(actual.imag().is_zero());
    assert!(
        Float::with_val_64(reference.prec_64(), actual.real() - &reference).abs()
            < cnum::epsilon(1000, reference.prec_64())
    );
    assert_eq!(
        cnum::format_complex(&actual, 1000),
        cnum::format_complex(&Complex::with_val_64(reference.prec_64(), reference), 1000)
    );
}

#[test]
fn native_complex_heights_match_independent_asymptotics() {
    // Independent220/280-digit mpmath references agree beyond203 digits.
    let references = [
        (
            "-0.5",
            "1",
            "6.735193423870095168342421791418567174533352918029330883183169462235440358013840946990316146600747094278022812e-500000000000000002",
        ),
        (
            "-1.5",
            "6.735193423870095168342421791418567174533352918029330883183169462235440358013840946990316146600747094278022812e499999999999999998",
            "-6.718644541276922913424435127857335788607611534345563557296649492999487538180005653622040476611136721647969765e499999999999999999",
        ),
        (
            "-14.5",
            "-1.570796326794896619913509868560672103611260628357011485503311864272373584970219121478936874117301887137061079e1000000000000000000",
            "-2.302585092994045726298555573825142338090376409504330858858875130330548031653679556470894958192405911431370419e1000000000000000018",
        ),
    ];
    for (height, re, im) in references {
        let reference = cnum::parse_complex(re, im, cnum::digits_to_bits(140)).unwrap();
        for digits in [5, 10, 20, 50, 70, 100] {
            let (re, im) = cnum::format_complex(&reference, digits);
            let expected = format!("{re}\n{im}\n");
            for mt in [false, true] {
                let mut command = Command::new(env!("CARGO_BIN_EXE_tet"));
                command.args([
                    "--quiet",
                    &digits.to_string(),
                    "1",
                    "1e-1000000000000000000",
                    height,
                    "0.25",
                ]);
                if mt {
                    command.env("TET_MT", "2").env("RAYON_NUM_THREADS", "2");
                } else {
                    command.env_remove("TET_MT").env_remove("RAYON_NUM_THREADS");
                }
                let output = command.output().unwrap();
                assert!(output.status.success(), "{output:?}");
                assert!(output.stderr.is_empty(), "{output:?}");
                assert_eq!(
                    output.stdout,
                    expected.as_bytes(),
                    "height {height}, MT={mt}"
                );
            }
        }
    }
}

#[test]
fn tiny_lambert_arguments_preserve_secondary_components() {
    // W(-epsilon^2/2 - i*epsilon) has Re=epsilon^2/2+O(epsilon^4)
    // and Im=-epsilon+O(epsilon^3); these corrections are below100 digits.
    let reference_prec = cnum::digits_to_bits(180);
    for (exponent, squared_exponent) in
        [("1000", "2001"), ("10000000000000000", "20000000000000001")]
    {
        let re = format!("5e-{squared_exponent}");
        let im = format!("-1e-{exponent}");
        let reference = cnum::parse_complex(&re, &im, reference_prec).unwrap();
        for digits in [70, 100] {
            let prec = cnum::digits_to_bits(digits);
            let input = cnum::parse_complex(&format!("-{re}"), &im, prec).unwrap();
            let actual = lambertw::w0(&input, prec).unwrap();
            let output_digits = usize::try_from(digits).unwrap();
            assert_eq!(
                cnum::format_complex(&actual, output_digits),
                cnum::format_complex(&reference, output_digits)
            );
        }
    }
}

// Independent forward-orbit/log references at240 and300 decimal digits;
// 110/135-digit depth targets agree beyond100 digits. No inverse series.
const FRINGE_REFERENCES: [(&str, &str, &str, &str); 3] = [
    (
        "1.444666",
        "0",
        "1.2655647868566379508647220704469430366450876563198296563847079228153825765785440759952554168687071467814418",
        "0.10765928766204063851271866249545787770432656190422190341755839518262611309047275683212420039020974788608277125",
    ),
    (
        "0.0665",
        "0",
        "0.36814245071016899704353091478747754266301666113364017915099453485972432810582461575812943769352116919743788701",
        "0.019833170610419060142423589092883344470146947086999541910945007317920336951432843199957122743751232012391676652",
    ),
    (
        "1.980042301122793883205552038769930319569163330071908679961709828263266126528188560211196925825416867",
        "1.190116848996558709064465375821260685166072869458403452714324612411072765362447609742552177662114766",
        "0.96061735585832470150830420278243515975908707263792262023629560099806007362657491774671028434264766261243006362",
        "0.72329660852020489108199728593869069005228692240128111309146903935257779583263831045135508639811610800052656086",
    ),
];

#[test]
fn strictly_attracting_fringe_matches_independent_references() {
    let reference_prec = cnum::digits_to_bits(130);
    for (br, bi, re, im) in FRINGE_REFERENCES {
        let reference = cnum::parse_complex(re, im, reference_prec).unwrap();
        for digits in [1, 5, 10, 20, 50, 70, 100] {
            let prec = cnum::checked_input_precision(digits, &[br, bi, "0.5", "0.25"]).unwrap();
            let base = cnum::parse_complex(br, bi, prec).unwrap();
            let height = cnum::parse_complex("0.5", "0.25", prec).unwrap();
            let actual = dispatch::tetrate(&base, &height, prec, digits).unwrap();
            for (got, expected) in [
                (actual.real(), reference.real()),
                (actual.imag(), reference.imag()),
            ] {
                let error = Float::with_val_64(reference_prec, got - expected).abs();
                assert!(
                    error < cnum::epsilon(digits, reference_prec) * expected.clone().abs(),
                    "base {br}+{bi}i, digits {digits}, error {error}"
                );
            }
            let output_digits = usize::try_from(digits).unwrap();
            assert_eq!(
                cnum::format_complex(&actual, output_digits),
                cnum::format_complex(&reference, output_digits),
                "base {br}+{bi}i, digits {digits}"
            );
        }
    }
}

#[test]
fn fringe_cli_preserves_reference_digits_in_serial_and_mt() {
    let reference_prec = cnum::digits_to_bits(130);
    for (br, bi, re, im) in FRINGE_REFERENCES {
        let reference = cnum::parse_complex(re, im, reference_prec).unwrap();
        let mut cases = vec![(bi.to_owned(), "0.25", reference.clone())];
        if bi != "0" {
            cases.push((
                format!("-{bi}"),
                "-0.25",
                Complex::with_val_64(reference_prec, reference.conj_ref()),
            ));
        }
        for (bi, hi, reference) in cases {
            for digits in [50, 100] {
                let (re, im) = cnum::format_complex(&reference, digits);
                let expected = format!("{re}\n{im}\n");
                for mt in [false, true] {
                    let mut command = Command::new(env!("CARGO_BIN_EXE_tet"));
                    command.args(["--quiet", &digits.to_string(), br, &bi, "0.5", hi]);
                    if mt {
                        command.env("TET_MT", "2").env("RAYON_NUM_THREADS", "2");
                    } else {
                        command.env_remove("TET_MT").env_remove("RAYON_NUM_THREADS");
                    }
                    let output = command.output().unwrap();
                    assert!(output.status.success(), "{output:?}");
                    assert!(output.stderr.is_empty(), "{output:?}");
                    assert_eq!(
                        output.stdout,
                        expected.as_bytes(),
                        "base {br}+{bi}i, MT={mt}"
                    );
                }
            }
        }
    }
}

#[test]
fn attracting_boundary_cache_preserves_family_and_conditioning() {
    let digits = 50;
    let prec = cnum::digits_to_bits(digits);
    let base = cnum::parse_complex("1.444666", "0", prec).unwrap();
    let fp = match regions::classify(&base, prec).unwrap() {
        regions::Region::ShellThronBoundary(fp) => fp,
        other => panic!("expected boundary band, got {other:?}"),
    };
    let state = schroder::setup_schroder(&base, &fp, prec).unwrap();
    assert_eq!(
        state.prec, prec,
        "construction must not ratchet caller precision"
    );
    assert!(state
        .inverse_radius
        .as_ref()
        .is_some_and(|radius| *radius > 0));
    for (hr, hi) in [
        ("0", "0"),
        ("0.5", "0.25"),
        ("1.5", "0.25"),
        ("-0.5", "0.25"),
    ] {
        let height = cnum::parse_complex(hr, hi, prec).unwrap();
        let cached = schroder::eval_schroder_at_digits(&state, &height, digits).unwrap();
        let direct = dispatch::tetrate(&base, &height, prec, digits).unwrap();
        let error = cnum::abs(&Complex::with_val_64(prec, &cached - direct), prec);
        assert!(error < cnum::epsilon(digits, prec));
        assert!(
            cached.real().prec_64() < cnum::digits_to_bits(150),
            "near-fixed-point log unwinds must not invent exponential conditioning"
        );
    }
}
