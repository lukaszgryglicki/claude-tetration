//! Phase-9 comprehensive coverage: region edges, reality, Schwarz reflection,
//! precision scaling, and ABSOLUTE reference values across every algorithm path.
//!
//! ## Why this file does not lean on functional-equation self-checks
//!
//! Earlier phases validate the Kouznetsov path with `F(z+1) = b^F(z)`. That
//! check is **nearly tautological** for the Cauchy interpolant: `eval_at_height`
//! reduces any height to the strip `Re ∈ [0,1)` and then *computes* `F(z+1)` as
//! `b^(cauchy_eval(z_strip))` — literally one extra `b^·` step applied to the
//! same interpolant value. So `F(z+1) = b^F(z)` holds to full precision by
//! construction and validates only the `b^·`/`log_b` recursion plumbing, not
//! the tetration *values*. (Measured: b=−2 and b=0.5+1.5i both pass the FE
//! check to full precision precisely because it is trivial.)
//!
//! The genuine correctness signals this file uses instead:
//!   * **Absolute reference values** — published Kneser constants (e^^0.5) and
//!     high-precision regression anchors cross-validated by reality + scaling.
//!   * **Precision scaling** — the same input at `p` and `2p` digits must agree
//!     to ~`p` digits (numerical convergence evidence, not a proof).
//!   * **Reality** — `Im(F_b(h)) = 0` for real `b ∈ (1, ∞)` and real `h`
//!     (canonical Kneser is real on the real axis). NB: this does NOT hold for
//!     `b ∈ (0,1)` where the multiplier `λ < 0` makes `λ^h` genuinely complex.
//!   * **Schwarz reflection** — `F_b(h̄) = conj(F_b(h))` for real `b > 1`.
//!
//! Slow Kouznetsov-heavy cases (base e, large bases, precision scaling, unit
//! circle) are `#[ignore]`d; run them with
//! `cargo test --release --test phase9_comprehensive -- --ignored`.

use rug::{Complex, Float};

use tetration::{cnum, dispatch, kouznetsov, regions};

fn parse(re: &str, im: &str, prec: u64) -> Complex {
    cnum::parse_complex(re, im, prec).unwrap()
}

fn abs(z: &Complex, prec: u64) -> Float {
    assert!(cnum::is_finite(z));
    Float::with_val_64(prec, z.abs_ref())
}

fn im_abs(z: &Complex, prec: u64) -> Float {
    assert!(cnum::is_finite(z));
    Float::with_val_64(prec, z.imag().abs_ref())
}

/// ≈ number of matching significant digits between `a` and `b`.
fn matching_digits(a: &Complex, b: &Complex, prec: u64) -> Float {
    assert!(cnum::is_finite(a) && cnum::is_finite(b));
    let diff = Complex::with_val_64(prec, a - b);
    let da = abs(&diff, prec);
    if da.is_zero() {
        return Float::with_val_64(prec, rug::float::Special::Infinity);
    }
    let scale = abs(b, prec).max(&Float::with_val_64(prec, 1));
    -(da / scale).log10()
}

fn region_of(b: &Complex, prec: u64) -> regions::Region {
    regions::classify(b, prec).unwrap()
}

// ────────────────────────────────────────────────────────────────────────────
// 1. Region classification: thresholds and edges from all sides
// ────────────────────────────────────────────────────────────────────────────

/// The dispatcher routes on `|λ|`: `<0.95` interior, `≤1.05` boundary, `>1.05`
/// outside. Verify every classified base's region label is consistent with its
/// reported `lambda_abs`, across a fine sweep of real bases that crosses both
/// thresholds — and that all three regions actually occur.
#[test]
fn t900_region_thresholds_consistent_real_sweep() {
    let prec = cnum::digits_to_bits(40);
    let mut seen_interior = false;
    let mut seen_boundary = false;
    let mut seen_outside = false;
    // b from 1.30 to 1.80 in steps of 0.01 spans |λ| ≈ 0.5 .. 1.3, crossing both
    // 0.95 and 1.05 thresholds (η = e^(1/e) ≈ 1.4447 sits in the middle).
    let mut bi = 130;
    while bi <= 180 {
        let b_str = format!("1.{:02}", bi - 100);
        let b = parse(&b_str, "0", prec);
        let r = region_of(&b, prec);
        match &r {
            regions::Region::ShellThronInterior(d) => {
                seen_interior = true;
                assert!(
                    d.lambda_abs < cnum::decimal(regions::SHELL_THRON_INTERIOR_THRESHOLD, prec),
                    "b={}: labelled interior but |λ|={} ≥ 0.95",
                    b_str,
                    d.lambda_abs
                );
            }
            regions::Region::ShellThronBoundary(d) => {
                seen_boundary = true;
                assert!(
                    d.lambda_abs >= cnum::decimal(regions::SHELL_THRON_INTERIOR_THRESHOLD, prec)
                        && d.lambda_abs
                            <= cnum::decimal(regions::SHELL_THRON_OUTER_THRESHOLD, prec),
                    "b={}: labelled boundary but |λ|={} outside [0.95,1.05]",
                    b_str,
                    d.lambda_abs
                );
            }
            regions::Region::OutsideShellThronRealPositive(d) => {
                seen_outside = true;
                assert!(
                    d.lambda_abs > cnum::decimal(regions::SHELL_THRON_OUTER_THRESHOLD, prec),
                    "b={}: labelled outside but |λ|={} ≤ 1.05",
                    b_str,
                    d.lambda_abs
                );
            }
            other => panic!("b={}: unexpected region {}", b_str, other.name()),
        }
        bi += 1;
    }
    assert!(
        seen_interior && seen_boundary && seen_outside,
        "sweep should cross all three regions (interior={}, boundary={}, outside={})",
        seen_interior,
        seen_boundary,
        seen_outside
    );
}

/// Just inside / just outside the interior threshold (|λ|=0.95) from both sides,
/// located by bisection so the test is robust to the exact η value.
#[test]
fn t901_interior_boundary_threshold_both_sides() {
    let prec = cnum::digits_to_bits(70);
    let lam = |b_re: &Float| -> Float {
        let b = Complex::with_val_64(prec, b_re);
        match region_of(&b, prec) {
            regions::Region::ShellThronInterior(d)
            | regions::Region::ShellThronBoundary(d)
            | regions::Region::OutsideShellThronRealPositive(d)
            | regions::Region::OutsideShellThronGeneral(d) => d.lambda_abs,
            other => panic!("unexpected region {}", other.name()),
        }
    };
    // |λ| is increasing in b on (1, η). Bisect for |λ| = 0.95.
    let (mut lo, mut hi) = (cnum::decimal("1.30", prec), cnum::decimal("1.4446", prec));
    let threshold = cnum::decimal("0.95", prec);
    for _ in 0..240 {
        let mid = Float::with_val_64(prec, &lo + &hi) / 2;
        if lam(&mid) < threshold {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let b_at = Float::with_val_64(prec, lo + hi) / 2;
    // A hair below the crossing must be interior; a hair above must be boundary.
    let delta = cnum::epsilon(60, prec);
    let b_below = Complex::with_val_64(prec, Float::with_val_64(prec, &b_at - &delta));
    let b_above = Complex::with_val_64(prec, Float::with_val_64(prec, &b_at + &delta));
    assert!(
        matches!(
            region_of(&b_below, prec),
            regions::Region::ShellThronInterior(_)
        ),
        "b={} (just below |λ|=0.95) should be interior",
        b_below
    );
    assert!(
        matches!(
            region_of(&b_above, prec),
            regions::Region::ShellThronBoundary(_)
        ),
        "b={} (just above |λ|=0.95) should be boundary",
        b_above
    );
}

// ────────────────────────────────────────────────────────────────────────────
// 2. Reality on the real axis (Schröder interior, b ∈ (1, η))
// ────────────────────────────────────────────────────────────────────────────

/// For real `b ∈ (1, η)` (positive multiplier λ) and real height, the canonical
/// tetration is real-valued. Assert `Im(F)≈0` NUMERICALLY (earlier phases used
/// brittle string checks that miss values like `1.2e-34`).
#[test]
fn t910_reality_schroder_interior_real_bases() {
    let digits = 40u64;
    let prec = cnum::digits_to_bits(digits);
    let tol = cnum::epsilon(digits - 6, prec);
    let bases = ["1.1", "1.2", "1.3", "1.4", "1.4142135623730950488"];
    let heights = ["0.5", "0.3", "0.7", "1.5", "2.5", "-0.5", "-1.5"];
    for b_str in &bases {
        let b = parse(b_str, "0", prec);
        for h_str in &heights {
            let h = parse(h_str, "0", prec);
            let f = dispatch::tetrate(&b, &h, prec, digits).unwrap();
            assert!(
                im_abs(&f, prec) < tol,
                "Im(F_{}({})) = {:.3e} should be ≈0 (real base in (1,η), real height)",
                b_str,
                h_str,
                im_abs(&f, prec)
            );
        }
    }
}

/// Counterpoint: for `b ∈ (0,1)` the multiplier `λ < 0`, so `λ^h` and hence
/// `F_b(h)` are GENUINELY complex for non-integer real `h`. Document/lock this
/// so a future "force reality" change cannot silently corrupt these bases.
#[test]
fn t911_b_below_one_gives_complex_result() {
    let digits = 40u64;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("0.5", "0", prec);
    let f = dispatch::tetrate(&b, &parse("0.7", "0", prec), prec, digits).unwrap();
    assert!(
        im_abs(&f, prec) > cnum::epsilon(2, prec),
        "F_0.5(0.7) should be genuinely complex (λ<0), got Im={:.3e}",
        im_abs(&f, prec)
    );
    // Integer heights remain real, though: 0.5^^2 = 0.5^0.5 = 1/√2.
    let f2 = dispatch::tetrate(&b, &parse("2", "0", prec), prec, digits).unwrap();
    assert!(
        im_abs(&f2, prec) < cnum::epsilon(digits, prec),
        "0.5^^2 should be real"
    );
    let expected = Float::with_val_64(prec, 2).sqrt().recip();
    assert!(Float::with_val_64(prec, f2.real() - expected).abs() < cnum::epsilon(digits, prec));
}

// ────────────────────────────────────────────────────────────────────────────
// 3. Absolute reference values (regression anchors), Schröder paths
// ────────────────────────────────────────────────────────────────────────────

/// √2 ^^ 0.5. High-precision anchor (60 digits), cross-checked by precision
/// scaling in t931. √2 is the canonical Shell-Thron interior base.
#[test]
fn t920_reference_sqrt2_half() {
    let digits = 60u64;
    let prec = cnum::digits_to_bits(digits);
    let b = Complex::with_val_64(prec, Float::with_val_64(prec, 2).sqrt());
    let h = parse("0.5", "0", prec);
    let f = dispatch::tetrate(&b, &h, prec, digits).unwrap();
    let reference = parse(
        "1.243621627668521804295098983609402931688198356615523786426384266739854041905083929661122492605564522",
        "0",
        prec,
    );
    let m = matching_digits(&f, &reference, prec);
    assert!(m >= 55, "√2^^0.5 matched only {} digits vs reference", m);
    assert!(
        im_abs(&f, prec) < cnum::epsilon(50, prec),
        "√2^^0.5 must be real"
    );
}

/// b = 1.2 ^^ 0.5 — interior real base, second anchor.
#[test]
fn t921_reference_1p2_half() {
    let digits = 60u64;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("1.2", "0", prec);
    let f = dispatch::tetrate(&b, &parse("0.5", "0", prec), prec, digits).unwrap();
    let reference = parse(
        "1.13626248672712808418530091864742602558935546503741898785324",
        "0",
        prec,
    );
    let m = matching_digits(&f, &reference, prec);
    assert!(m >= 55, "1.2^^0.5 matched only {} digits", m);
    assert!(im_abs(&f, prec) < cnum::epsilon(50, prec));
}

/// Imaginary base b = i (|λ|≈0.89, inside ST but reached via σ̃-shift). Complex
/// reference value anchor + Schwarz reflection check `F_{-i}(h̄)=conj(F_i(h))`.
#[test]
fn t922_reference_imaginary_base_and_schwarz() {
    let digits = 40u64;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("0", "1", prec);
    let h = parse("0.5", "0", prec);
    let f = dispatch::tetrate(&b, &h, prec, digits).unwrap();
    let reference = parse(
        "1.166700913570474569302791134642004482191",
        "0.7345635369867213296594900876244841035952",
        prec,
    );
    let m = matching_digits(&f, &reference, prec);
    assert!(m >= 35, "i^^0.5 matched only {} digits", m);

    // Schwarz across the real axis of the BASE: F_{b̄}(h̄) = conj(F_b(h)).
    // For b=i, h real: F_{-i}(0.5) should equal conj(F_i(0.5)).
    let b_conj = parse("0", "-1", prec);
    let f_conj = dispatch::tetrate(&b_conj, &h, prec, digits).unwrap();
    let target = Complex::with_val_64(prec, f.conj_ref());
    let ms = matching_digits(&f_conj, &target, prec);
    assert!(
        ms >= 35,
        "Schwarz F_{{-i}}(0.5)=conj(F_i(0.5)) matched {} digits",
        ms
    );
}

// ────────────────────────────────────────────────────────────────────────────
// 4. Precision scaling (the genuine accuracy test)
// ────────────────────────────────────────────────────────────────────────────

/// Schröder interior: the same input at 128 and 256 digits must agree to
/// ~120 digits. Stresses the σ̃ Taylor build at high precision (the user
/// explicitly asked for 128-digit-class precision tests).
#[test]
fn t930_precision_scaling_schroder_128_256() {
    let (lo, hi, want) = (128u64, 256u64, 118);
    let prec_lo = cnum::digits_to_bits(lo);
    let prec_hi = cnum::digits_to_bits(hi);
    let b_lo = parse("1.3", "0", prec_lo);
    let b_hi = parse("1.3", "0", prec_hi);
    let h_lo = parse("0.5", "0", prec_lo);
    let h_hi = parse("0.5", "0", prec_hi);
    let f_lo = dispatch::tetrate(&b_lo, &h_lo, prec_lo, lo).unwrap();
    let f_hi = dispatch::tetrate(&b_hi, &h_hi, prec_hi, hi).unwrap();
    let f_lo_hi = Complex::with_val_64(prec_hi, &f_lo);
    let m = matching_digits(&f_lo_hi, &f_hi, prec_hi);
    assert!(
        m >= want,
        "Schröder {}→{} digit scaling: only {} digits agree",
        lo,
        hi,
        m
    );
}

/// Complex interior base precision scaling, 80 → 160 digits.
#[test]
fn t931_precision_scaling_complex_base() {
    let (lo, hi) = (80u64, 160u64);
    let prec_lo = cnum::digits_to_bits(lo);
    let prec_hi = cnum::digits_to_bits(hi);
    let b_lo = parse("1.3", "0.2", prec_lo);
    let b_hi = parse("1.3", "0.2", prec_hi);
    let h_lo = parse("0.4", "0.3", prec_lo);
    let h_hi = parse("0.4", "0.3", prec_hi);
    let f_lo = dispatch::tetrate(&b_lo, &h_lo, prec_lo, lo).unwrap();
    let f_hi = dispatch::tetrate(&b_hi, &h_hi, prec_hi, hi).unwrap();
    let f_lo_hi = Complex::with_val_64(prec_hi, &f_lo);
    let m = matching_digits(&f_lo_hi, &f_hi, prec_hi);
    assert!(
        m >= 72,
        "complex-base {}→{} scaling: only {} digits agree",
        lo,
        hi,
        m
    );
}

// ────────────────────────────────────────────────────────────────────────────
// 5. Integer-height edges and undefined-domain contracts
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn t940_integer_height_edges() {
    let digits = 50u64;
    let prec = cnum::digits_to_bits(digits);
    // 3^^3 = 3^(3^3) = 3^27 = 7625597484987.
    let b = parse("3", "0", prec);
    let f = dispatch::tetrate(&b, &parse("3", "0", prec), prec, digits).unwrap();
    assert!(
        matching_digits(&f, &parse("7625597484987", "0", prec), prec) >= 40,
        "3^^3 = 7625597484987"
    );
    // F(0)=1, F(1)=b exactly for an arbitrary complex base.
    let bc = parse("1.7", "0.9", prec);
    let f0 = dispatch::tetrate(&bc, &parse("0", "0", prec), prec, digits).unwrap();
    assert!(
        abs(
            &Complex::with_val_64(prec, &f0 - parse("1", "0", prec)),
            prec
        ) < cnum::epsilon(45, prec)
    );
    let f1 = dispatch::tetrate(&bc, &parse("1", "0", prec), prec, digits).unwrap();
    assert!(abs(&Complex::with_val_64(prec, &f1 - &bc), prec) < cnum::epsilon(45, prec));
    // F(-1) = 0.
    let fm1 = dispatch::tetrate(&bc, &parse("-1", "0", prec), prec, digits).unwrap();
    assert!(abs(&fm1, prec) < cnum::epsilon(45, prec), "F(-1) must be 0");
}

#[test]
fn t941_undefined_domains_error() {
    let digits = 30u64;
    let prec = cnum::digits_to_bits(digits);
    // h ≤ -2 integer is undefined (log_b(0) chain).
    for hr in ["-2", "-3", "-5"] {
        let b = parse("2", "0", prec);
        assert!(
            dispatch::tetrate(&b, &parse(hr, "0", prec), prec, digits).is_err(),
            "F_2({}) should error (undefined)",
            hr
        );
    }
    // b=0 non-integer / negative height is undefined; non-negative integer ok.
    let zero = parse("0", "0", prec);
    assert!(dispatch::tetrate(&zero, &parse("0.5", "0", prec), prec, digits).is_err());
    assert!(dispatch::tetrate(&zero, &parse("-1", "0", prec), prec, digits).is_err());
    assert_eq!(
        dispatch::tetrate(&zero, &parse("2", "0", prec), prec, digits).unwrap(),
        cnum::one(prec)
    );
}

// ────────────────────────────────────────────────────────────────────────────
// 6. Kouznetsov real base (b=2): one setup, many checks — reality, reference,
//    Schwarz, asymptote, integer-shift recursion. Default-runnable (~1 min).
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn t950_kouznetsov_b2_reality_reference_schwarz() {
    let digits = 12u64;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("2", "0", prec);
    let fp = match region_of(&b, prec) {
        regions::Region::OutsideShellThronRealPositive(d) => d,
        other => panic!("b=2 should be outside-real-positive, got {}", other.name()),
    };
    let state = kouznetsov::setup_kouznetsov(&b, &fp, prec, digits).expect("b=2 setup");

    // (a) Reality for real heights, across the strip and through the integer
    //     shift recursion (Re>1 and Re<0).
    let real_tol = cnum::epsilon(9, prec);
    for h_str in &["0.5", "0.3", "1.5", "2.5", "-0.5", "-1.5"] {
        let h = parse(h_str, "0", prec);
        let f = kouznetsov::eval_kouznetsov(&state, &b, &h).unwrap();
        assert!(
            im_abs(&f, prec) < real_tol,
            "Im(F_2({})) = {:.3e} should be ≈0",
            h_str,
            im_abs(&f, prec)
        );
    }

    // (b) Independent fatou.gp numerical reference: gp-tetration values.json,
    //     sexp|2|0.5|500. This is not an interval certificate.
    let f05 = kouznetsov::eval_kouznetsov(&state, &b, &parse("0.5", "0", prec)).unwrap();
    let reference = parse("1.458781816036421700683971661038587135296606605330907141888207751656464160881083209019144256057703804", "0", prec);
    let m = matching_digits(&f05, &reference, prec);
    assert!(
        m >= 9,
        "2^^0.5 matched only {} digits vs reference (got {})",
        m,
        f05.real()
    );

    // (c) Schwarz reflection in the HEIGHT: F_2(h̄) = conj(F_2(h)).
    let hp = parse("0.4", "0.3", prec);
    let hm = parse("0.4", "-0.3", prec);
    let fp_v = kouznetsov::eval_kouznetsov(&state, &b, &hp).unwrap();
    let fm_v = kouznetsov::eval_kouznetsov(&state, &b, &hm).unwrap();
    let target = Complex::with_val_64(prec, fp_v.conj_ref());
    let ms = matching_digits(&fm_v, &target, prec);
    assert!(
        ms >= 9,
        "Schwarz F_2(h̄)=conj(F_2(h)) matched only {} digits",
        ms
    );

    // (d) A limiting fixed point is not a checked finite-height value.
    for sign in [-1, 1] {
        let h = Complex::with_val_64(prec, (0, (state.t_max.clone() + 1) * sign));
        let error = kouznetsov::eval_kouznetsov(&state, &b, &h).unwrap_err();
        assert!(error.contains("outside the Cauchy contour"), "{error}");
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 7. Slow Kouznetsov cases (#[ignore]) — external reference, precision scaling,
//    complex heights, large bases, unit circle.
//    Run: cargo test --release --test phase9_comprehensive -- --ignored
// ────────────────────────────────────────────────────────────────────────────

/// Independent fatou.gp numerical reference, gp-tetration values.json
/// sexp|e|0.5|500; not an interval certificate.
#[test]
#[ignore]
fn t960_reference_e_half_external() {
    let digits = 20u64;
    let prec = cnum::digits_to_bits(digits);
    let b = Complex::with_val_64(prec, (Float::with_val_64(prec, 1).exp(), 0));
    let f = dispatch::tetrate(&b, &parse("0.5", "0", prec), prec, digits).unwrap();
    let reference = parse("1.646354233751194580971924031592114518205311648969041530394879047518804627043365960655603009256205342", "0", prec);
    let m = matching_digits(&f, &reference, prec);
    assert!(
        m >= digits,
        "e^^0.5 matched only {} digits vs independent reference",
        m
    );
    assert!(
        im_abs(&f, prec) < cnum::epsilon(digits, prec),
        "e^^0.5 must be real, Im={:.3e}",
        im_abs(&f, prec)
    );
}

/// Kouznetsov precision scaling: b=2 at 12 vs 20 digits must agree to ≥11
/// digits. This checks numerical consistency, not a grid-independent limit.
#[test]
#[ignore]
fn t961_kouznetsov_precision_scaling_b2() {
    let lo = 12u64;
    let hi = 20u64;
    let prec_lo = cnum::digits_to_bits(lo);
    let prec_hi = cnum::digits_to_bits(hi);
    let b_lo = parse("2", "0", prec_lo);
    let b_hi = parse("2", "0", prec_hi);
    let h_lo = parse("0.5", "0", prec_lo);
    let h_hi = parse("0.5", "0", prec_hi);
    let f_lo = dispatch::tetrate(&b_lo, &h_lo, prec_lo, lo).unwrap();
    let f_hi = dispatch::tetrate(&b_hi, &h_hi, prec_hi, hi).unwrap();
    let f_lo_hi = Complex::with_val_64(prec_hi, &f_lo);
    let m = matching_digits(&f_lo_hi, &f_hi, prec_hi);
    assert!(
        m >= 11,
        "Kouznetsov b=2 {}→{} scaling: only {} digits agree",
        lo,
        hi,
        m
    );
}

/// Complex heights with larger |Im| for a Kouznetsov real base (coverage gap:
/// earlier tests cap at |Im(h)|=0.5). One setup, several heights.
#[test]
#[ignore]
fn t962_kouznetsov_b2_complex_heights() {
    let digits = 12u64;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("2", "0", prec);
    let fp = match region_of(&b, prec) {
        regions::Region::OutsideShellThronRealPositive(d) => d,
        other => panic!("unexpected {}", other.name()),
    };
    let state = kouznetsov::setup_kouznetsov(&b, &fp, prec, digits).unwrap();
    // Schwarz must hold for each: F(h̄)=conj(F(h)).
    for (hr, hi) in &[
        ("0.5", "1.5"),
        ("0.5", "2.5"),
        ("1.5", "0.8"),
        ("-0.5", "1.2"),
    ] {
        let h = parse(hr, hi, prec);
        let h_conj = parse(hr, &format!("-{}", hi), prec);
        let f = kouznetsov::eval_kouznetsov(&state, &b, &h).unwrap();
        let fc = kouznetsov::eval_kouznetsov(&state, &b, &h_conj).unwrap();
        assert!(cnum::is_finite(&f), "F_2({}+{}i) must be finite", hr, hi);
        let target = Complex::with_val_64(prec, f.conj_ref());
        let m = matching_digits(&fc, &target, prec);
        assert!(m >= 9, "Schwarz failed for h={}+{}i: {} digits", hr, hi, m);
    }
}

/// Large real base regression (cap-clamp basin selection): reality + plausible
/// magnitude at b=100.
#[test]
#[ignore]
fn t963_large_base_b100_reality() {
    let digits = 12u64;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("100", "0", prec);
    let f = dispatch::tetrate(&b, &parse("0.5", "0", prec), prec, digits).unwrap();
    assert!(
        im_abs(&f, prec) < cnum::epsilon(8, prec),
        "F_100(0.5) should be real, Im={:.3e}",
        im_abs(&f, prec)
    );
    assert!(
        (f.real() - cnum::decimal("4.2131", prec)).abs() < cnum::epsilon(2, prec),
        "F_100(0.5) ≈ 4.213 (FAILURE_CASES §E), got {}",
        f.real()
    );
}

/// Additional unit-circle directions missed by the rectangular grid.
/// This is an availability check, not an independent accuracy certificate.
#[test]
#[ignore]
fn t964_unit_circle_bases() {
    let digits = 12u64;
    let prec = cnum::digits_to_bits(digits);
    for numerator in 1..=3 {
        let angle: Float = Float::with_val_64(prec, rug::float::Constant::Pi) * numerator / 5;
        let b = Complex::with_val_64(prec, (angle.clone().cos(), angle.sin()));
        let h = parse("0.5", "0", prec);
        let f =
            dispatch::tetrate(&b, &h, prec, digits).unwrap_or_else(|e| panic!("b={b} failed: {e}"));
        assert!(
            cnum::is_finite(&f),
            "F for unit-circle base {b} must be finite"
        );
    }
}

#[test]
fn t965_kouznetsov_unaddressable_geometry_is_explicit() {
    let digits = 70;
    let prec = cnum::digits_to_bits(digits);
    let base = Complex::with_val_64(prec, cnum::eta_upper(prec) + cnum::epsilon(60, prec));
    let fp = match region_of(&base, prec) {
        regions::Region::OutsideShellThronRealPositive(fp)
        | regions::Region::ShellThronBoundary(fp) => fp,
        _ => panic!("unexpected near-parabolic region"),
    };
    for setup in [
        kouznetsov::setup_kouznetsov,
        kouznetsov::setup_kouznetsov_continuation,
    ] {
        let error = setup(&base, &fp, prec, digits)
            .err()
            .expect("this geometry exceeds native addressability, not a fixed node budget");
        assert!(error.contains("addressable memory"), "{error}");
        assert!(
            !error.contains("needs Abel"),
            "a resource cap is not a theorem requirement: {error}"
        );
    }
}
