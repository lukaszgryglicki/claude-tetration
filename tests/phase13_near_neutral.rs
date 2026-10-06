use rug::{Complex, Float};
use std::process::Command;
use tetration::{cnum, regions, schroder};

const BASE: &str = "1.444667861009766";
const REAL_REFERENCE: &str = "1.257153075054172563551531590671458882657588099868146519075509383039283963152328860924332384527919880191605183624752032947410449324312605190275858248898866039404";
const COMPLEX_REAL: &str = "1.265565575490916570561830338520122837269307970470374668564550990010460704859405491777906793563220396039962438951148060113813834509946615006858223802982859800487";
const COMPLEX_IMAGINARY: &str = "0.1076597647881089642685375153955445345121618005253379046013651039482480553698723729047845472448279721023146944150941075338389416976251214180360000612603503266514";
const CLOSER_BASE: &str = "1.444667861009766133658339108596430223058595453242253165820522";

#[test]
fn closer_attractor_matches_independent_hundred_digit_reference() {
    // Independent360-digit orbit/log limits agree beyond120 digits after
    // approximately1.75e32/2.11e32 iterations, not native integer counters.
    let expected = cnum::parse_complex(
        "1.265565575490916627201748381492953442447791839923112046058359712141383885271338394658161975328134000770162012607430943509840198704291918731129772388779097974209",
        "0.1076597647881089985358843656767515142739493452366518198164725709877255319442017631751207628506773110283979510611216686029762562643419405700221773755301929144846",
        cnum::digits_to_bits(180),
    )
    .unwrap();
    let actual = tetration::tetrate_str("100", CLOSER_BASE, "0", "0.5", "0.25").unwrap();
    assert_eq!(actual, cnum::format_complex(&expected, 100));
}

#[test]
fn near_neutral_values_match_independent_orbit_references() {
    // Independent360-digit orbit/log references, with no Poincare coefficients,
    // agree beyond120 digits at two depths exceeding12 billion iterations.
    let reference_prec = cnum::digits_to_bits(180);
    for digits in [20, 50, 100] {
        let prec = cnum::digits_to_bits(digits);
        let base = cnum::parse_complex(BASE, "0", prec).unwrap();
        let regions::Region::ShellThronBoundary(fp) = regions::classify(&base, prec).unwrap()
        else {
            panic!("expected near-neutral attracting boundary");
        };
        assert!(fp.lambda_abs < 1);
        let state = schroder::setup_schroder(&base, &fp, prec).unwrap();
        for (imaginary, real_reference, imaginary_reference) in [
            ("0", REAL_REFERENCE, "0"),
            ("0.25", COMPLEX_REAL, COMPLEX_IMAGINARY),
        ] {
            let height = cnum::parse_complex("0.5", imaginary, prec).unwrap();
            let actual = schroder::eval_schroder_at_digits(&state, &height, digits).unwrap();
            let expected =
                cnum::parse_complex(real_reference, imaginary_reference, reference_prec).unwrap();
            let error = cnum::abs(
                &Complex::with_val_64(reference_prec, &actual - &expected),
                reference_prec,
            );
            assert!(error < cnum::epsilon(digits, reference_prec));
            assert_eq!(
                cnum::format_complex(&actual, digits as usize),
                cnum::format_complex(&expected, digits as usize),
                "height0.5+{imaginary}i, digits={digits}"
            );
            if imaginary == "0" {
                assert!(actual.imag().is_zero());
            }
        }
    }
}

#[test]
fn near_neutral_cli_preserves_precision_and_serial_mt_identity() {
    let reference =
        cnum::parse_complex(COMPLEX_REAL, COMPLEX_IMAGINARY, cnum::digits_to_bits(180)).unwrap();
    let (real, imaginary) = cnum::format_complex(&reference, 50);
    let expected = format!("{real}\n{imaginary}\n");
    for mt in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_tet"));
        command.args(["--quiet", "50", BASE, "0", "0.5", "0.25"]);
        if mt {
            command.env("TET_MT", "2").env("RAYON_NUM_THREADS", "2");
        } else {
            command.env_remove("TET_MT").env_remove("RAYON_NUM_THREADS");
        }
        let result = command.output().unwrap();
        assert!(result.status.success(), "{result:?}");
        assert!(result.stderr.is_empty(), "{result:?}");
        assert_eq!(result.stdout, expected.as_bytes());
    }
}

#[test]
fn near_neutral_cache_preserves_conjugation_and_exact_singularities() {
    let digits = 50;
    let prec = cnum::digits_to_bits(digits);
    let base = cnum::parse_complex(BASE, "0", prec).unwrap();
    let regions::Region::ShellThronBoundary(fp) = regions::classify(&base, prec).unwrap() else {
        panic!("expected attracting boundary");
    };
    let state = schroder::setup_schroder(&base, &fp, prec).unwrap();
    let positive = cnum::parse_complex("0.5", "0.25", prec).unwrap();
    let negative = cnum::parse_complex("0.5", "-0.25", prec).unwrap();
    let upper = schroder::eval_schroder_at_digits(&state, &positive, digits).unwrap();
    let lower = schroder::eval_schroder_at_digits(&state, &negative, digits).unwrap();
    assert_eq!(upper, lower.conj());
    let anchor = schroder::eval_schroder_at_digits(&state, &cnum::zero(prec), digits).unwrap();
    assert!(cnum::abs(&Complex::with_val_64(prec, anchor - 1), prec) < cnum::epsilon(digits, prec));
    let minus_one = Complex::with_val_64(prec, -1);
    assert!(cnum::is_zero(
        &schroder::eval_schroder_at_digits(&state, &minus_one, digits).unwrap()
    ));
    let singular = Complex::with_val_64(prec, Float::with_val_64(prec, -2));
    assert!(schroder::eval_schroder_at_digits(&state, &singular, digits)
        .unwrap_err()
        .contains("undefined"));
}
