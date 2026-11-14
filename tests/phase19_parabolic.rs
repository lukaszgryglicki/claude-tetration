//! Phase-19 verification: the base `e^(1/e)` itself.
//!
//! The fixed point `e` of that base is parabolic (multiplier 1): the
//! tetration is the inverse of the Fatou coordinate of its attracting petal
//! (`src/parabolic.rs`, README section 6.11). The references of `t1901` and
//! `t1905` come from an independent implementation of the same definition in
//! mpmath at 60 and 70 digits; `t1902` compares with the bases next to
//! `e^(1/e)`, which other solvers of this crate compute.

use rug::{Complex, Float};

use tetration::{cnum, dispatch, inverse};

fn abs(z: &Complex, prec: u64) -> Float {
    Float::with_val_64(prec, z.abs_ref())
}

fn eta(prec: u64) -> Complex {
    cnum::parse_complex("eta", "0", prec).unwrap()
}

fn parse(text: &str, prec: u64) -> Complex {
    tetration::expression::complex(text, prec).unwrap()
}

/// `|a - b| <= 10^-digits max(|b|, 1)`.
fn agree(a: &Complex, b: &Complex, digits: u64, prec: u64) -> bool {
    let gap = abs(&Complex::with_val_64(prec, a - b), prec);
    gap <= abs(b, prec).max(&Float::with_val_64(prec, 1)) * cnum::epsilon(digits, prec)
}

#[test]
fn t1901_values_equal_the_independent_references() {
    // Real heights on both sides of -1, complex heights, a height on the cut
    // left of -2 (the limit from above), heights left of it, a high tower.
    for (height, expected) in [
        ("0.5", ("1.257153075054172622171646564771545603158", "0")),
        ("2.5", ("1.793547876057851987699343513492580948981", "0")),
        ("-1.5", ("-1.290340716109089597650282455523536647709", "0")),
        (
            "0.5+i",
            (
                "1.379563412960519507089378438708816544294",
                "3.938793109454438618269292818088839768700e-1",
            ),
        ),
        (
            "-0.5",
            ("6.220779817189264160616748623482828035432e-1", "0"),
        ),
        ("-1.9", ("-5.058680565129762065116950813167046384905", "0")),
        (
            "3+20i",
            (
                "2.638270468256375915081429887505302879208",
                "2.395288069370469381381089941109999943952e-1",
            ),
        ),
        (
            "0.25-0.75i",
            (
                "1.231239819237669884267995882122203729171",
                "-3.583814580896733426071638534335774470478e-1",
            ),
        ),
        (
            "-2.5",
            (
                "6.929071755215875477116089126805375084328e-1",
                "8.539734222673567065463550869546574495035",
            ),
        ),
        (
            "-2.5+0.3i",
            (
                "9.174923566244586110064472828638441630114e-1",
                "6.177892973073860646122057164525408407076",
            ),
        ),
        (
            "-7.25",
            (
                "3.987271019404725013551018421716641674239",
                "3.129519481787094019689224544089510371069e-1",
            ),
        ),
        ("1000", ("2.712872863905652059821730070118230061753", "0")),
        ("1025", ("2.713004178372011782406559889584193921172", "0")),
    ] {
        let found = tetration::tetrate_complex_str("40", "eta", height).unwrap();
        assert_eq!((found.0.as_str(), found.1.as_str()), expected, "h={height}");
    }
    // The base as it may be written, and as four real numbers.
    let half = "1.257153075054172622171646564771545603158";
    for base in ["e^(1/e)", "exp(1/e)", "exp(exp(-1))", "η", "towermax"] {
        let found = tetration::tetrate_complex_str("40", base, "0.5").unwrap();
        assert_eq!((found.0.as_str(), found.1.as_str()), (half, "0"), "{base}");
    }
    let found = tetration::tetrate_str("40", "eta", "0", "0.5", "0").unwrap();
    assert_eq!((found.0.as_str(), found.1.as_str()), (half, "0"));
}

#[test]
fn t1902_the_bases_on_both_sides_approach_it() {
    // Regular iteration below e^(1/e) and the Kneser tetration above it
    // meet at the parabolic tetration, with one slope in the base: at the
    // height 0.5 the value moves by 0.4386 times the change of the base.
    let digits = 20;
    let prec = cnum::digits_to_bits(digits);
    let height = parse("0.5", prec);
    let at = dispatch::tetrate(&eta(prec), &height, prec, digits).unwrap();
    let slope = |offset: &str| {
        let base = parse(&format!("eta{offset}"), prec);
        let value = dispatch::tetrate(&base, &height, prec, digits).unwrap();
        let step = Complex::with_val_64(prec, &base - eta(prec));
        Complex::with_val_64(prec, value - &at) / step
    };
    let (below, above) = (slope("-1e-6"), slope("+1e-3"));
    for (side, found) in [("below", &below), ("above", &above)] {
        assert!(found.imag().is_zero(), "{side}: {found}");
        assert!(
            *found.real() > 0.4380 && *found.real() < 0.4392,
            "{side}: slope {found}"
        );
    }
    // A complex height and a height between -2 and -1, from below.
    for (text, digits_agreeing) in [("0.5+i", 8), ("-1.5", 8), ("3+20i", 7)] {
        let height = parse(text, prec);
        let at = dispatch::tetrate(&eta(prec), &height, prec, digits).unwrap();
        let near = dispatch::tetrate(&parse("eta-1e-10", prec), &height, prec, digits).unwrap();
        assert!(agree(&near, &at, digits_agreeing, prec), "h={text}");
    }
}

#[test]
fn t1903_functional_equation_tower_and_precisions() {
    let digits = 30;
    let prec = cnum::digits_to_bits(digits);
    let base = eta(prec);
    for text in [
        "0.3",
        "-1.7",
        "0.5+2i",
        "-3.5-0.25i",
        "40.25",
        "-0.5+30i",
        "1e6+0.5",
    ] {
        let height = parse(text, prec);
        let next = Complex::with_val_64(prec, &height + 1u32);
        let lower = dispatch::tetrate(&base, &height, prec, digits).unwrap();
        let upper = dispatch::tetrate(&base, &next, prec, digits).unwrap();
        let image = cnum::pow_complex(&base, &lower, prec);
        assert!(agree(&upper, &image, digits - 2, prec), "h={text}");
    }
    // The tower is iterated up to 1024 levels and read from the Fatou
    // coordinate above: the two meet.
    let tower = dispatch::tetrate(&base, &parse("1024", prec), prec, digits).unwrap();
    let above = dispatch::tetrate(&base, &parse("1025", prec), prec, digits).unwrap();
    assert!(agree(
        &above,
        &cnum::pow_complex(&base, &tower, prec),
        digits - 2,
        prec
    ));
    // Each precision has its own radius and number of terms of the series:
    // a longer result begins with the shorter one.
    let long = tetration::tetrate_complex_str("300", "eta", "0.5+0.25i").unwrap();
    for digits in ["12", "100"] {
        let short = tetration::tetrate_complex_str(digits, "eta", "0.5+0.25i").unwrap();
        let prec = cnum::digits_to_bits(300);
        let found = cnum::parse_complex(&short.0, &short.1, prec).unwrap();
        let reference = cnum::parse_complex(&long.0, &long.1, prec).unwrap();
        let agreeing: u64 = digits.parse().unwrap();
        assert!(agree(&found, &reference, agreeing - 1, prec), "{digits}");
    }
}

#[test]
fn t1904_heights_far_to_the_right_and_next_to_the_singularity() {
    let digits = 25;
    let prec = cnum::digits_to_bits(digits);
    let base = eta(prec);
    let e = Float::with_val_64(prec, 1).exp();
    // F(h) = e - 2e/h + ... far to the right, at any size of the height.
    for (text, size) in [("1e30", "1e30"), ("1e30+0.5i", "1e30"), ("3e50i", "3e50")] {
        let value = dispatch::tetrate(&base, &parse(text, prec), prec, digits).unwrap();
        let gap = abs(&Complex::with_val_64(prec, &value - &e), prec);
        let expected = Float::with_val_64(prec, &e * 2u32) / cnum::decimal(size, prec);
        let ratio = Float::with_val_64(prec, gap / expected);
        assert!(ratio > 0.999999 && ratio < 1.000001, "h={text}: {ratio}");
    }
    // Next to -2 the value is e ln(h + 2) + c: nine more powers of ten in
    // the distance move it by e ln(10^-9).
    let near = tetration::tetrate_complex_str("25", "eta", "-2+1e-20").unwrap();
    let nearer = tetration::tetrate_complex_str("25", "eta", "-2+1e-29").unwrap();
    assert_eq!((near.1.as_str(), nearer.1.as_str()), ("0", "0"));
    let long = cnum::digits_to_bits(40);
    let moved = cnum::decimal(&nearer.0, long) - cnum::decimal(&near.0, long);
    let expected = Float::with_val_64(long, 1).exp() * cnum::decimal("1e-9", long).ln();
    assert!(
        Float::with_val_64(long, moved - &expected).abs() < 1e-16,
        "{} {}",
        near.0,
        nearer.0
    );
    // On the cut left of -2 the value is the limit from above, and the
    // values below the cut are the conjugates.
    let cut = dispatch::tetrate(&base, &parse("-2.5", prec), prec, digits).unwrap();
    let above = dispatch::tetrate(&base, &parse("-2.5+1e-22i", prec), prec, digits).unwrap();
    let below = dispatch::tetrate(&base, &parse("-2.5-1e-22i", prec), prec, digits).unwrap();
    assert!(agree(&above, &cut, 18, prec));
    assert!(agree(&below, &cut.clone().conj(), 18, prec));
    assert!(*cut.imag() > 8);
    // The integers at and below -2 stay undefined.
    for height in ["-2", "-5"] {
        assert!(tetration::tetrate_complex_str("20", "eta", height).is_err());
    }
}

#[test]
fn t1905_super_logarithm_references_and_round_trips() {
    for (value, expected) in [
        ("1.5", ("1.17847006973612261450329583503", "0")),
        ("0.3", ("-7.94206993357713877779870358587e-1", "0")),
        (
            "1.2+0.4i",
            (
                "1.52101816178460678595646466149e-1",
                "7.91913187846825708312377820016e-1",
            ),
        ),
        ("-3", ("-1.76912696701255138075340160022", "0")),
    ] {
        let found = tetration::slog_complex_str("30", "eta", value).unwrap();
        assert_eq!((found.0.as_str(), found.1.as_str()), expected, "z={value}");
    }
    let digits = 25;
    let prec = cnum::digits_to_bits(digits);
    let base = eta(prec);
    // Heights of every kind come back from their values.
    for text in ["0.5", "-1.5", "0.5+i", "17.25", "-0.9+0.01i", "2-3i"] {
        let height = parse(text, prec);
        let value = dispatch::tetrate(&base, &height, prec, digits).unwrap();
        let back = inverse::slog(&base, &value, prec, digits).unwrap();
        assert!(agree(&back, &height, digits - 3, prec), "h={text}: {back}");
    }
    // A value next to the fixed point has a large height, 2e/(e - z).
    let high = tetration::slog_complex_str("20", "eta", "e-1e-30").unwrap();
    assert!(high.0.starts_with("5.43656365"), "{}", high.0);
    // The special values, and a value below every finite tower.
    for (value, expected) in [("1", "0"), ("0", "-1"), ("eta", "1"), ("-1e6", "-2")] {
        let found = tetration::slog_complex_str("20", "eta", value).unwrap();
        let prec = cnum::digits_to_bits(20);
        let found = cnum::parse_complex(&found.0, &found.1, prec).unwrap();
        let expected = cnum::parse_complex(expected, "0", prec).unwrap();
        assert_eq!(found, expected, "z={value}");
    }
    // The fixed point itself has no height, and real values beyond it are
    // not values of this tetration.
    for (value, reason) in [("e", "singular"), ("3", "beyond"), ("e+1e-30", "beyond")] {
        let error = tetration::slog_complex_str("20", "eta", value).unwrap_err();
        assert!(error.contains(reason), "z={value}: {error}");
    }
}

#[test]
fn t1906_super_root_returns_the_base() {
    // The values of t1901 at 20 digits: the base with that value at that
    // height is e^(1/e), to every requested digit.
    let expected = "1.4446678610097661337";
    for (value, height) in [
        ("1.2571530750541726222", "0.5"),
        ("1.7935478760578519877", "2.5"),
        ("-1.2903407161090895977", "-1.5"),
        ("0.62207798171892641606", "-0.5"),
        ("1.3795634129605195071+0.39387931094544386183i", "0.5+i"),
    ] {
        let found = tetration::sroot_complex_str("20", value, height).unwrap();
        assert_eq!(
            (found.0.as_str(), found.1.as_str()),
            (expected, "0"),
            "z={value} h={height}"
        );
    }
    // The root that comes back is the base itself for the next run.
    let prec = cnum::digits_to_bits(20);
    let value = dispatch::tetrate(&eta(prec), &parse("0.5", prec), prec, 20).unwrap();
    let root = inverse::sroot(&value, &parse("0.5", prec), prec, 20).unwrap();
    assert_eq!(root, eta(prec));
    // A target that is not the value of that base is left to the search.
    let found = tetration::sroot_complex_str("15", "1.25", "0.5").unwrap();
    assert_eq!(
        (found.0.as_str(), found.1.as_str()),
        ("1.42846985922260", "0")
    );
}

#[test]
fn t1908_super_roots_just_below_the_base() {
    // The values of the bases 1e-4 below e^(1/e) at five heights and 1e-10
    // below it at one (regular iteration, 30 digits). The search takes
    // e^(1/e) for the upper end of its bracket and asks no base above it:
    // the theta mapping takes ever longer there, and next to the last
    // root it did not come back.
    for (digits, value, height, expected) in [
        (
            "15",
            "1.25710921503065608490900106555",
            "0.5",
            "1.44456786100977",
        ),
        (
            "15",
            "6.22100201838948117079477367644e-1",
            "-0.5",
            "1.44456786100977",
        ),
        (
            "15",
            "-1.29048644889336635777994202837",
            "-1.5",
            "1.44456786100977",
        ),
        (
            "15",
            "1.79324268618399092662471635719",
            "2.5",
            "1.44456786100977",
        ),
        (
            "15",
            "1.86958096773149994856761931551",
            "3",
            "1.44456786100977",
        ),
        (
            "15",
            "1.25715307501031444215271375230",
            "0.5",
            "1.44466786090977",
        ),
    ] {
        let found = tetration::sroot_complex_str(digits, value, height).unwrap();
        assert_eq!(
            (found.0.as_str(), found.1.as_str()),
            (expected, "0"),
            "z={value} h={height}"
        );
    }
}

#[test]
fn t1907_the_other_neutral_bases_are_refused_at_once() {
    // A multiplier on the unit circle other than 1: the base e^(-e) (-1) and
    // the complex bases exp(lambda e^(-lambda)). Regular iteration and the
    // two-sided theta mapping ran without end there; the refusal says what
    // the multiplier is. A base below the real axis is reflected first.
    for (base, kind) in [
        ("towermin", "root of unity of order 2"),
        ("e^(-e)", "root of unity of order 2"),
        ("exp(i*exp(-i))", "root of unity of order 4"),
        ("exp(-i*exp(i))", "root of unity of order 4"),
        (
            "exp(exp(2*pi*i/3)*exp(-exp(2*pi*i/3)))",
            "root of unity of order 3",
        ),
        ("exp(exp(i)*exp(-exp(i)))", "not a root of unity"),
    ] {
        for height in ["0.5", "0.25+2i"] {
            let error = tetration::tetrate_complex_str("20", base, height).unwrap_err();
            assert!(
                error.contains("Shell-Thron boundary") && error.contains(kind),
                "{base} at {height}: {error}"
            );
        }
        let error = tetration::slog_complex_str("20", base, "0.5").unwrap_err();
        assert!(error.contains(kind), "slog, {base}: {error}");
    }
    // Their towers are computed.
    let found = tetration::tetrate_complex_str("20", "towermin", "3").unwrap();
    assert_eq!(
        (found.0.as_str(), found.1.as_str()),
        ("1.0311386287497114777e-1", "0")
    );
    let found = tetration::slog_complex_str("20", "towermin", "towermin").unwrap();
    assert_eq!(
        (found.0.as_str(), found.1.as_str()),
        ("1.0000000000000000000", "0")
    );
    // The bases next to them are what they were: e^(-e) + 10^-4, whose
    // value at 0.5 is already next to the fixed point 1/e.
    let found = tetration::tetrate_complex_str("20", "towermin+1e-4", "0.5").unwrap();
    assert_eq!(
        (found.0.as_str(), found.1.as_str()),
        ("3.6782846812463796796e-1", "1.5030963164342977103e-2")
    );
}
