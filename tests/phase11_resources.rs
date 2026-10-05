use rug::{float::Constant, Complex, Float, Integer};
use tetration::cnum::{self, DisplayComplex, DisplayFloat};
use tetration::{kouznetsov, regions};

#[test]
fn integer_towers_retain_significant_digits_at_large_native_exponents() {
    use rug::ops::Pow;

    let reference_prec = cnum::digits_to_bits(200);
    let exponent = Integer::from(15).pow(15);
    let reference = Float::with_val_64(reference_prec, 15).pow(&exponent);
    assert!(reference.is_finite());
    assert_eq!(
        cnum::format_float(&reference, 100),
        "3.646844018452076009536426678607249890237808805138402713433662167024194558211242552864229187268787815e515003176870815367"
    );
    for digits in [40, 60, 100] {
        let (real, imaginary) =
            tetration::tetrate_str(&digits.to_string(), "15", "0", "3", "0").unwrap();
        assert_eq!(
            real,
            cnum::format_float(&reference, usize::try_from(digits).unwrap()),
            "{digits} digits"
        );
        assert_eq!(imaginary, "0");
    }

    let digits = 70;
    let prec = cnum::digits_to_bits(digits);
    let base = cnum::parse_complex("1e-1000000", "0", prec).unwrap();
    let value = tetration::dispatch::tetrate(&base, &cnum::one(prec), prec, digits).unwrap();
    assert_eq!(value, base);
    assert_eq!(value.real().prec_64(), prec);
}

#[test]
fn schroder_conditioning_preserves_small_values_and_near_singular_heights() {
    let mut near_minus_one = cnum::parse_complex("-1", "0", 512).unwrap();
    near_minus_one += Float::with_val_64(512, 1) >> 128u32;
    let mut near_minus_two = cnum::parse_complex("-2", "0", 512).unwrap();
    near_minus_two += Float::with_val_64(512, 1) >> 128u32;
    let mut runs = Vec::new();
    for digits in [40, 60, 80] {
        let prec = cnum::digits_to_bits(digits);
        let base = cnum::parse_complex("1.25", "0", prec).unwrap();
        let regions::Region::ShellThronInterior(fp) = regions::classify(&base, prec).unwrap()
        else {
            panic!("base1.25 must use the regular construction");
        };
        let state = tetration::schroder::setup_schroder(&base, &fp, prec).unwrap();
        let mut values = Vec::new();
        for reference_height in [&near_minus_one, &near_minus_two] {
            let height = Complex::with_val_64(prec, reference_height);
            assert_eq!(&height, reference_height);
            let value =
                tetration::schroder::eval_schroder_at_digits(&state, &height, digits).unwrap();
            assert!(cnum::is_finite(&value) && !cnum::is_zero(&value));
            assert!(value.imag().is_zero());
            values.push(value);
        }
        assert!(cnum::is_zero(
            &tetration::schroder::eval_schroder_at_digits(
                &state,
                &cnum::parse_complex("-1", "0", prec).unwrap(),
                digits
            )
            .unwrap()
        ));
        runs.push((digits, values));
    }
    for (digits, values) in &runs[..2] {
        for (actual, reference) in values.iter().zip(&runs[2].1) {
            let digits = usize::try_from(*digits).unwrap();
            assert_eq!(
                cnum::format_complex(actual, digits),
                cnum::format_complex(reference, digits)
            );
        }
    }
}

#[test]
fn large_imaginary_heights_preserve_phase_precision() {
    use rug::ops::Pow;

    let exact_height = Integer::from(10).pow(40u32);
    let mut values = Vec::new();
    for digits in [40, 60, 80] {
        let prec = cnum::digits_to_bits(digits);
        let parsed_height = cnum::parse_float("1e40", prec).unwrap();
        assert_eq!(parsed_height, exact_height);
        let (real, imaginary) =
            tetration::tetrate_str(&digits.to_string(), "1.25", "0", "0", "1e40").unwrap();
        values.push((
            digits,
            cnum::parse_complex(&real, &imaginary, cnum::digits_to_bits(120)).unwrap(),
        ));
    }
    for (digits, value) in &values[..2] {
        let digits = usize::try_from(*digits).unwrap();
        assert_eq!(
            cnum::format_complex(value, digits),
            cnum::format_complex(&values[2].1, digits)
        );
    }
}

#[test]
fn string_precision_refinement_keeps_the_original_digit_goal() {
    let height = format!("-0.{}", "9".repeat(40));
    let mut values = Vec::new();
    for digits in [40, 60, 80] {
        let (real, imaginary) =
            tetration::tetrate_str(&digits.to_string(), "1.25", "0", &height, "0").unwrap();
        assert_eq!(imaginary, "0");
        let parsed = cnum::parse_float(&real, cnum::digits_to_bits(120)).unwrap();
        assert!(!parsed.is_zero());
        values.push((digits, real, parsed));
    }
    for (digits, real, _) in &values[..2] {
        assert_eq!(
            real,
            &cnum::format_float(&values[2].2, usize::try_from(*digits).unwrap())
        );
    }
}

#[test]
fn kouznetsov_large_height_retains_requested_significant_digits() {
    let digits = 40;
    let prec = cnum::digits_to_bits(digits);
    let reference_prec = cnum::digits_to_bits(100);
    let base = cnum::parse_complex("2", "0", prec).unwrap();
    let regions::Region::OutsideShellThronRealPositive(fp) =
        regions::classify(&base, prec).unwrap()
    else {
        panic!("base2 must use the real-positive Kouznetsov construction");
    };
    let state = kouznetsov::setup_kouznetsov(&base, &fp, prec, digits).unwrap();
    // Cross-precision regression fixture, not an independent canonicality certificate.
    let seed = cnum::decimal(
        "1.33973255870712201977668814097251442779349041198532936171062",
        reference_prec,
    );
    let ln_two = Float::with_val_64(reference_prec, 2).ln();
    for (height, shifts) in [("0.375", 0), ("5.375", 5)] {
        let mut reference = seed.clone();
        for _ in 0..shifts {
            reference = (reference * &ln_two).exp();
        }
        let actual = kouznetsov::eval_kouznetsov(
            &state,
            &base,
            &cnum::parse_complex(height, "0", prec).unwrap(),
        )
        .unwrap();
        let (re, im) = cnum::format_complex(&actual, usize::try_from(digits).unwrap());
        println!("precision_probe\t{digits}\t{height}\t{re}\t{im}");
        assert_eq!(
            re,
            cnum::format_float(&reference, usize::try_from(digits).unwrap()),
            "height {height}"
        );
        let imaginary_error =
            Float::with_val_64(reference_prec, actual.imag().abs_ref()) / &reference;
        assert!(imaginary_error < cnum::epsilon(digits, reference_prec));
        if shifts > 0 {
            assert!(actual.real().prec_64() > prec);
        } else {
            let height = cnum::parse_complex(height, "0", prec).unwrap();
            let lower_goal =
                kouznetsov::eval_kouznetsov_at_digits(&state, &base, &height, 20).unwrap();
            assert_eq!(actual, lower_goal);
            assert!(kouznetsov::eval_kouznetsov_at_digits(&state, &base, &height, 0).is_err());
        }
    }
}

#[test]
fn constants_are_computed_at_requested_precision_not_from_fixed_tables() {
    for digits in [70, 1000, 10_000] {
        let prec = cnum::digits_to_bits(digits);
        let reference_prec = cnum::digits_to_bits(digits + 30);
        let pi_reference = (Float::with_val_64(reference_prec, 1) / 5u32).atan() * 16u32
            - (Float::with_val_64(reference_prec, 1) / 239u32).atan() * 4u32;
        let mut e_reference = Float::with_val_64(reference_prec, 1);
        let mut term = e_reference.clone();
        let mut n = Integer::new();
        let tail_target = cnum::epsilon(digits + 20, reference_prec);
        loop {
            n += 1;
            term /= &n;
            e_reference += &term;
            if term < tail_target {
                break;
            }
        }
        let lower_reference = (-e_reference.clone()).exp();
        let upper_reference = e_reference.clone().recip().exp();
        for (name, actual, reference) in [
            ("pi", Float::with_val_64(prec, Constant::Pi), pi_reference),
            ("e", Float::with_val_64(prec, 1).exp(), e_reference),
            (
                "ln(2)",
                Float::with_val_64(prec, 2).ln(),
                (Float::with_val_64(reference_prec, 1) / 3u32).atanh() * 2u32,
            ),
            (
                "ln(10)",
                Float::with_val_64(prec, 10).ln(),
                (Float::with_val_64(reference_prec, 9) / 11u32).atanh() * 2u32,
            ),
            ("e^-e", cnum::eta_lower(prec), lower_reference),
            ("e^(1/e)", cnum::eta_upper(prec), upper_reference),
        ] {
            let error = Float::with_val_64(reference_prec, &actual - &reference).abs();
            assert!(
                error < cnum::epsilon(digits, reference_prec),
                "{name} at {digits} digits"
            );
            let output_digits = usize::try_from(digits).unwrap();
            assert_eq!(
                cnum::format_float(&actual, output_digits),
                cnum::format_float(&reference, output_digits),
                "{name} at {digits} digits"
            );
        }
    }
}

#[test]
fn native_precision_metadata_has_no_billion_digit_or_u32_ceiling() {
    let native_max = rug::float::prec_max_64();
    if native_max > u64::from(u32::MAX) {
        let bits = cnum::checked_digits_to_bits(1_500_000_000).unwrap();
        assert!(bits > u64::from(u32::MAX));
        cnum::require_precision(bits, 1_500_000_000).unwrap();
    }
    assert!(cnum::checked_digits_to_bits(0).is_err());
    assert!(cnum::checked_digits_to_bits(u64::MAX).is_err());
    cnum::check_precision(native_max).unwrap();
    assert!(cnum::check_precision(native_max + 1).is_err());

    let (mut low, mut high) = (1, u64::MAX);
    while high - low > 1 {
        let mid = low + (high - low) / 2;
        if cnum::checked_digits_to_bits(mid).is_ok() {
            low = mid;
        } else {
            high = mid;
        }
    }
    assert!(cnum::checked_digits_to_bits(low).unwrap() <= native_max);
    assert!(cnum::checked_digits_to_bits(low + 1).is_err());
}

#[test]
fn safe_decimal_formatting_matches_rug_at_ordinary_exponents() {
    let prec = cnum::digits_to_bits(100);
    for text in [
        "0",
        "-0",
        "1",
        "-1",
        "0.125",
        "-0.125",
        "12345.6789",
        "-12345.6789",
        "0.0000123456789",
        "1e1000",
        "-1e-1000",
        "3.1415926535897932384626433832795028841971693993751",
        "NaN",
        "-NaN",
        "inf",
        "-inf",
    ] {
        let value = Float::with_val_64(prec, Float::parse(text).unwrap());
        for digits in [0usize, 1, 2, 3, 50, 70, 1000] {
            let expected = if value.is_nan() {
                "NaN".into()
            } else {
                value.to_string_radix(10, Some(digits.max(1)))
            };
            assert_eq!(
                cnum::format_float(&value, digits),
                expected,
                "{text}, {digits}"
            );
            assert_eq!(
                format!("{:.digits$}", DisplayFloat(&value)),
                format!("{value:.digits$}")
            );
            assert_eq!(
                format!("{:.digits$e}", DisplayFloat(&value)),
                format!("{value:.digits$e}")
            );
            assert_eq!(
                format!("{:.digits$E}", DisplayFloat(&value)),
                format!("{value:.digits$E}")
            );
        }
        assert_eq!(cnum::format_float_roundtrip(&value), value.to_string());
        assert_eq!(format!("{}", DisplayFloat(&value)), value.to_string());
        assert_eq!(
            format!("{:+024.8e}", DisplayFloat(&value)),
            format!("{value:+024.8e}")
        );
        let complex = Complex::with_val_64(prec, (&value, &value));
        assert_eq!(
            format!("{:.5}", DisplayComplex(&complex)),
            format!("{complex:.5}")
        );
        assert_eq!(format!("{}", DisplayComplex(&complex)), complex.to_string());
    }
}

#[test]
fn huge_decimal_exponents_need_only_significand_storage() {
    if i128::from(cnum::exponent_range().1) < 4_000_000_000_000_000_000 {
        return;
    }
    let prec = cnum::digits_to_bits(100);
    for exponent in [
        "10000000000",
        "-10000000000",
        "1000000000000000000",
        "-1000000000000000000",
    ] {
        for sign in ["", "-"] {
            let text = format!("{sign}1e{exponent}");
            let value = cnum::parse_float(&text, prec).unwrap();
            assert!(!value.is_zero() && value.is_finite());
            let formatted = cnum::format_float(&value, 70);
            assert_eq!(formatted, format!("{sign}1.{}e{exponent}", "0".repeat(69)));
            let roundtrip = cnum::format_float_roundtrip(&value);
            assert!(
                roundtrip.len() < 150,
                "exponent should not expand to integer zeros"
            );
            assert!(cnum::parse_float(&roundtrip, prec).unwrap() == value);
            assert_eq!(format!("{:.70e}", DisplayFloat(&value)), formatted);
            assert!(
                format!(
                    "{}",
                    DisplayComplex(&Complex::with_val_64(prec, (&value, &value)))
                )
                .len()
                    < 305
            );
            assert!(cnum::as_integer(&Complex::with_val_64(prec, (&value, 0))).is_none());
        }
    }
}

#[test]
fn native_binary_exponent_extremes_roundtrip_without_large_allocations() {
    let prec = cnum::digits_to_bits(100);
    let (min, max) = cnum::exponent_range();
    let smallest = Float::with_val_64(prec, 1) >> usize::try_from(1 - min).unwrap();
    let largest_power = Float::with_val_64(prec, 1) << usize::try_from(max - 1).unwrap();
    for value in [smallest, largest_power] {
        assert!(value.is_finite() && !value.is_zero());
        let text = cnum::format_float_roundtrip(&value);
        assert!(text.len() < 150);
        assert!(cnum::parse_float(&text, prec).unwrap() == value);
    }
}

#[test]
fn out_of_range_exponentials_fail_before_huge_angle_reduction() {
    let prec = cnum::digits_to_bits(70);
    for (real, reason) in [("1e1000", "overflow"), ("-1e1000", "underflow")] {
        let argument = cnum::parse_complex(real, "1e1000000", prec).unwrap();
        let error = cnum::checked_exp(&argument, prec).unwrap_err();
        assert!(error.contains(reason), "{error}");
    }
}

#[test]
fn every_calling_and_rayon_worker_thread_uses_the_native_exponent_range() {
    let expected = cnum::exponent_range();
    std::thread::spawn(move || {
        tetration::mt::init_pool().unwrap();
        let actual = unsafe {
            (
                gmp_mpfr_sys::mpfr::get_emin(),
                gmp_mpfr_sys::mpfr::get_emax(),
            )
        };
        assert_eq!(actual, expected);
    })
    .join()
    .unwrap();
    tetration::mt::init_pool().unwrap();
    if tetration::mt::mt_enabled() {
        for actual in rayon::broadcast(|_| unsafe {
            (
                gmp_mpfr_sys::mpfr::get_emin(),
                gmp_mpfr_sys::mpfr::get_emax(),
            )
        }) {
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn fft_and_formatting_entry_points_initialize_fresh_threads() {
    let expected_range = cnum::exponent_range();
    let prec = cnum::digits_to_bits(70);
    let exponent = if i128::from(expected_range.1) > 4_000_000_000_000_000_000 {
        "1000000000000000000"
    } else {
        "1000000"
    };
    for sign in ["", "-"] {
        for operation in 0..4 {
            let value = cnum::parse_complex(&format!("1e{sign}{exponent}"), "0", prec).unwrap();
            let expected = value.clone();
            std::thread::spawn(move || {
                let result = match operation {
                    0 => tetration::fft::convolve(&[value], &[Complex::with_val_64(prec, 1)], prec)
                        [0]
                    .clone(),
                    1 => tetration::fft::precompute_kernel_fft(&[value], 1, prec).coeffs[0].clone(),
                    2 => {
                        let kernel = tetration::fft::KernelFft {
                            coeffs: vec![Complex::with_val_64(prec, 1); 2],
                            n: 1,
                            m: 2,
                        };
                        tetration::fft::cross_correlate_with_kernel(&[value], &kernel, prec)[0]
                            .clone()
                    }
                    _ => {
                        assert!(format!("{}", DisplayFloat(value.real())).len() < 150);
                        value
                    }
                };
                assert!(
                    result == expected,
                    "entry point {operation} changed the value"
                );
                let actual_range = unsafe {
                    (
                        gmp_mpfr_sys::mpfr::get_emin(),
                        gmp_mpfr_sys::mpfr::get_emax(),
                    )
                };
                assert_eq!(actual_range, expected_range);
            })
            .join()
            .unwrap();
        }
    }
}

#[test]
fn significant_digits_do_not_limit_the_magnitude() {
    let digits = 100_000;
    let (re, im) = tetration::tetrate_str(&digits.to_string(), "10", "0", "3", "0").unwrap();
    assert_eq!(re, format!("1.{}e10000000000", "0".repeat(digits - 1)));
    assert_eq!(im, "0");
}

#[test]
fn exact_degenerate_towers_accept_heights_beyond_native_integers() {
    for height in [
        "100001",
        "9223372036854775808",
        "1e1000",
        "1e1000000000000000000",
    ] {
        let (re, im) = tetration::tetrate_str("70", "-1", "0", height, "0").unwrap();
        assert_eq!(re, format!("-1.{}", "0".repeat(69)));
        assert_eq!(im, "0");
        let (re, im) = tetration::tetrate_str("70", "1", "0", height, "0").unwrap();
        assert_eq!(re, format!("1.{}", "0".repeat(69)));
        assert_eq!(im, "0");
    }
    for height in ["9223372036854775808", "1e1000", "1e1000000000000000000"] {
        let (re, im) = tetration::tetrate_str("70", "0", "0", height, "0").unwrap();
        assert_eq!(re, format!("1.{}", "0".repeat(69)));
        assert_eq!(im, "0");
    }
}

#[test]
fn scaled_exponential_matches_mpc_rounding_at_ordinary_exponents() {
    for digits in [50, 70, 1000] {
        let prec = cnum::digits_to_bits(digits);
        for real in ["-10", "-1", "-0.1", "0", "1e-200", "0.3", "1", "3", "10"] {
            for imaginary in ["-10", "-1", "-0.1", "-0", "0", "1e-200", "0.3", "1", "10"] {
                let argument = cnum::parse_complex(real, imaginary, prec).unwrap();
                let expected = Complex::with_val_64(prec, argument.exp_ref());
                let actual = cnum::checked_exp(&argument, prec).unwrap();
                assert!(
                    actual == expected,
                    "exp({real}+i*{imaginary}) at {digits} digits: got {}, expected {}",
                    DisplayComplex(&actual),
                    DisplayComplex(&expected)
                );
                assert_eq!(
                    actual.imag().is_sign_negative(),
                    expected.imag().is_sign_negative(),
                    "imaginary sign for exp({real}+i*{imaginary})"
                );
            }
        }
    }
}

#[test]
fn scaled_exponential_preserves_native_minimum_input_components() {
    for digits in [70, 1000] {
        let prec = cnum::digits_to_bits(digits);
        let reference_prec = prec + 128;
        let (min, _) = cnum::exponent_range();
        let tiny = Float::with_val_64(prec, 1) >> usize::try_from(1 - min).unwrap();
        let e = Float::with_val_64(reference_prec, 1).exp();
        for sign in [-1, 1] {
            let signed_tiny = Float::with_val_64(prec, &tiny * sign);
            let argument = Complex::with_val_64(prec, (1, &signed_tiny));
            let actual = cnum::checked_exp(&argument, prec).unwrap();
            let expected_imaginary =
                Float::with_val_64(prec, Float::with_val_64(reference_prec, &e * &signed_tiny));
            assert!(
                actual.real() == &Float::with_val_64(prec, &e)
                    && actual.imag() == &expected_imaginary,
                "minimum imaginary input at {digits} digits: {}",
                DisplayComplex(&actual)
            );

            let angle = cnum::decimal("0.1", prec);
            let argument = Complex::with_val_64(prec, (&signed_tiny, &angle));
            let actual = cnum::checked_exp(&argument, prec).unwrap();
            assert!(
                actual.real() == &Float::with_val_64(prec, angle.cos_ref())
                    && actual.imag() == &Float::with_val_64(prec, angle.sin_ref()),
                "minimum real input at {digits} digits: {}",
                DisplayComplex(&actual)
            );
        }
        let underflow = Complex::with_val_64(prec, (-1, &tiny));
        let error = cnum::checked_exp(&underflow, prec).unwrap_err();
        assert!(error.contains("underflow"), "{error}");
    }
}

#[test]
fn scaled_exponential_supports_finite_components_at_native_range_edges() {
    for digits in [70, 1000] {
        let prec = cnum::digits_to_bits(digits);
        let reference_prec = prec + 192;
        let (min, max) = cnum::exponent_range();
        let ln_two = Float::with_val_64(prec, 2).ln();
        for (name, scale, offset, angle) in [
            (
                "lower",
                min,
                Float::with_val_64(prec, &ln_two * 5),
                cnum::decimal("0.1", prec),
            ),
            (
                "upper",
                max,
                cnum::decimal("1.1", prec).ln(),
                Float::with_val_64(prec, Constant::Pi) / 4u32,
            ),
        ] {
            let real = Float::with_val_64(prec, Float::with_val_64(prec, scale) * &ln_two) + offset;
            if name == "upper" {
                assert!(Float::with_val_64(prec, real.exp_ref()).is_infinite());
            }
            let argument = Complex::with_val_64(prec, (&real, &angle));
            let actual = cnum::checked_exp(&argument, prec).unwrap();
            let reference_real = Float::with_val_64(reference_prec, &real)
                - Float::with_val_64(reference_prec, scale)
                    * Float::with_val_64(reference_prec, 2).ln();
            let amplitude = reference_real.exp();
            for (component, trigonometric) in [
                (
                    actual.real(),
                    Float::with_val_64(reference_prec, angle.cos_ref()),
                ),
                (
                    actual.imag(),
                    Float::with_val_64(reference_prec, angle.sin_ref()),
                ),
            ] {
                let mut expected = Float::with_val_64(reference_prec, &amplitude * trigonometric);
                let shift = usize::try_from(scale.unsigned_abs()).unwrap();
                if scale < 0 {
                    expected >>= shift;
                } else {
                    expected <<= shift;
                }
                let expected = Float::with_val_64(prec, expected);
                assert!(
                    component == &expected,
                    "{name} boundary at {digits} digits: got {}, expected {}",
                    DisplayFloat(component),
                    DisplayFloat(&expected)
                );
            }
        }
    }
}
