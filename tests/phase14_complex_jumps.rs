use rug::{Complex, Float};
use std::process::Command;
use tetration::{cnum, regions, schroder};

const BASE_REAL: &str = "1.444667861009766";
const BASE_IMAGINARY: &str = "1e-15";
const REAL_HEIGHT_REAL: &str = "1.257153075054172563551531590671643219611306152940475975160674149303621644152552513426719255558612721960811964590038721622162805952870842256304665248216864144169";
const REAL_HEIGHT_IMAGINARY: &str = "0.0000000000000004385818001708944786021563997090834047942908634766767671342065706452930191623532733829073340427747202846815745914501092325558031265519850884546489777140623790769";
const COMPLEX_HEIGHT_REAL: &str = "1.265565575490916314181666715214839323707360120115670900985307456974093332205203628764314127291655885765653778106975156639175513028654104160166572764621168251308";
const COMPLEX_HEIGHT_IMAGINARY: &str = "0.1076597647881093880349737940426673509444821724877566295177289745737270853135619950206052877631745255778512885586858705041898335427947673069758478659480470334713";
const DEEP_BASE_REAL: &str = "1.444667861009766133658339108596430223058595453242253165820522";
const DEEP_HEIGHT_REAL: &str = "1.25715307505417262217164656477154560315780099474532991847539345914909192294372925231161722940242687400541700044417612067407607376072928734085377306047001868843";
const DEEP_HEIGHT_IMAGINARY: &str = "4.385818001708944293258142591228540779279593979879898033357964447086951528321589141395860469399080080071804515905592603976955001760268806261416608530034798394009e-41";

#[test]
fn complex_near_neutral_values_match_independent_references() {
    // Separate360-digit triangular orbit/log limits at two depths agree
    // beyond106 relative digits, including the tiny imaginary component.
    let reference_prec = cnum::digits_to_bits(180);
    for digits in [20, 50, 100] {
        let prec =
            cnum::checked_input_precision(digits, &[BASE_REAL, BASE_IMAGINARY, "0.5", "0.25"])
                .unwrap();
        let base = cnum::parse_complex(BASE_REAL, BASE_IMAGINARY, prec).unwrap();
        let regions::Region::ShellThronBoundary(fp) = regions::classify(&base, prec).unwrap()
        else {
            panic!("expected a complex attracting boundary");
        };
        assert!(fp.lambda_abs < 1);
        assert!(!fp.lambda.imag().is_zero());
        let state = schroder::setup_schroder(&base, &fp, prec).unwrap();
        for (height_imaginary, re, im) in [
            ("0", REAL_HEIGHT_REAL, REAL_HEIGHT_IMAGINARY),
            ("0.25", COMPLEX_HEIGHT_REAL, COMPLEX_HEIGHT_IMAGINARY),
        ] {
            let height = cnum::parse_complex("0.5", height_imaginary, prec).unwrap();
            let actual = schroder::eval_schroder_at_digits(&state, &height, digits).unwrap();
            let expected = cnum::parse_complex(re, im, reference_prec).unwrap();
            for (actual, expected) in [
                (actual.real(), expected.real()),
                (actual.imag(), expected.imag()),
            ] {
                let error = Float::with_val_64(reference_prec, actual - expected).abs();
                assert!(error < cnum::epsilon(digits, reference_prec) * expected.clone().abs());
            }
            assert_eq!(
                cnum::format_complex(&actual, digits as usize),
                cnum::format_complex(&expected, digits as usize),
                "height0.5+{height_imaginary}i, digits={digits}",
            );
        }
    }
}

#[test]
fn complex_near_neutral_cli_preserves_serial_mt_and_reflection() {
    let reference = cnum::parse_complex(
        COMPLEX_HEIGHT_REAL,
        COMPLEX_HEIGHT_IMAGINARY,
        cnum::digits_to_bits(180),
    )
    .unwrap();
    for (base_imaginary, height_imaginary, reflected) in
        [(BASE_IMAGINARY, "0.25", false), ("-1e-15", "-0.25", true)]
    {
        let expected = if reflected {
            reference.clone().conj()
        } else {
            reference.clone()
        };
        let (real, imaginary) = cnum::format_complex(&expected, 20);
        let expected = format!("{real}\n{imaginary}\n");
        for mt in [false, true] {
            let mut command = Command::new(env!("CARGO_BIN_EXE_tet"));
            command.args([
                "--quiet",
                "20",
                BASE_REAL,
                base_imaginary,
                "0.5",
                height_imaginary,
            ]);
            if mt {
                command.env("TET_MT", "2").env("RAYON_NUM_THREADS", "2");
            } else {
                command.env_remove("TET_MT").env_remove("RAYON_NUM_THREADS");
            }
            let output = command.output().unwrap();
            assert!(output.status.success(), "{output:?}");
            assert!(output.stderr.is_empty(), "{output:?}");
            assert_eq!(output.stdout, expected.as_bytes());
        }
    }
}

#[test]
fn deep_complex_jump_preserves_tiny_component_digits() {
    // Independent360-digit orbit/log limits agree beyond81 relative imaginary digits.
    let reference = cnum::parse_complex(
        DEEP_HEIGHT_REAL,
        DEEP_HEIGHT_IMAGINARY,
        cnum::digits_to_bits(180),
    )
    .unwrap();
    let (real, imaginary) = cnum::format_complex(&reference, 50);
    let output = Command::new(env!("CARGO_BIN_EXE_tet"))
        .args(["--quiet", "50", DEEP_BASE_REAL, "1e-40", "0.5", "0"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    assert!(output.stderr.is_empty(), "{output:?}");
    assert_eq!(output.stdout, format!("{real}\n{imaginary}\n").as_bytes());
}

#[test]
fn complex_near_neutral_cache_preserves_normalization_and_singularities() {
    let digits = 20;
    let prec = cnum::digits_to_bits(digits);
    let base = cnum::parse_complex(BASE_REAL, BASE_IMAGINARY, prec).unwrap();
    let regions::Region::ShellThronBoundary(fp) = regions::classify(&base, prec).unwrap() else {
        panic!("expected a complex attracting boundary");
    };
    let state = schroder::setup_schroder(&base, &fp, prec).unwrap();
    let anchor = schroder::eval_schroder_at_digits(&state, &cnum::zero(prec), digits).unwrap();
    assert_eq!(anchor, cnum::one(prec));
    assert_eq!(
        schroder::eval_schroder_at_digits(&state, &cnum::one(prec), digits).unwrap(),
        base
    );
    let minus_one = Complex::with_val_64(prec, -1);
    assert!(cnum::is_zero(
        &schroder::eval_schroder_at_digits(&state, &minus_one, digits).unwrap()
    ));
    for height in [-2, -3] {
        let height = Complex::with_val_64(prec, height);
        assert!(schroder::eval_schroder_at_digits(&state, &height, digits)
            .unwrap_err()
            .contains("undefined"));
    }
}
