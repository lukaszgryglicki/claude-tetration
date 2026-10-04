use rug::{Complex, Float};
use tetration::cnum::{self, DisplayComplex, DisplayFloat};

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
