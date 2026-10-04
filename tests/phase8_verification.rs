//! Phase-8 verification battery.
//!
//! Cross-cuts Phase 4–7 with: (a) golden integer-height values computed two
//! ways and cross-checked at high precision; (b) functional-equation residual
//! sampling at random (deterministic) heights in each region; (c) precision
//! scaling — same input at p and 2p digits, first p−10 digits agree.

use rug::{Complex, Float};

use tetration::{cnum, dispatch, kouznetsov, regions};

fn parse(re: &str, im: &str, prec: u64) -> Complex {
    cnum::parse_complex(re, im, prec).unwrap()
}

fn abs(z: &Complex, prec: u64) -> Float {
    assert!(cnum::is_finite(z));
    Float::with_val_64(prec, z.abs_ref())
}

fn matching_digits(a: &Complex, b: &Complex, prec: u64) -> Float {
    assert!(cnum::is_finite(a) && cnum::is_finite(b));
    let diff = Complex::with_val_64(prec, a - b);
    let da = abs(&diff, prec);
    if da.is_zero() {
        return Float::with_val_64(prec, rug::float::Special::Infinity);
    }
    -da.log10()
}

// ---------- Golden integer-height cross-checks ----------

/// `F_b(n)` via dispatch must equal the unrolled tower `b^(b^(...))` to within
/// the requested precision (modulo guard bits). Done at moderate precision so
/// the test is fast — accuracy of integer iteration scales perfectly with
/// `prec`, so 100-digit confidence is ample.
fn unrolled_tower(b: &Complex, n: i64, prec: u64) -> Complex {
    if n == 0 {
        return cnum::one(prec);
    }
    let mut acc = cnum::one(prec);
    for _ in 0..n {
        acc = cnum::pow_complex(b, &acc, prec);
    }
    acc
}

#[test]
fn t800_golden_two_to_four() {
    // 2^^4 = 2^(2^(2^2)) = 2^16 = 65536.
    let digits = 50;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("2", "0", prec);
    let h = parse("4", "0", prec);
    let f = dispatch::tetrate(&b, &h, prec, digits).unwrap();
    let expected = parse("65536", "0", prec);
    let m = matching_digits(&f, &expected, prec);
    assert!(m >= 45, "2^^4 differs: matched only {} digits", m);
}

#[test]
fn t801_golden_e_to_three() {
    // F_e(3) = e^(e^e). Cross-check vs unrolled tower at 100 digits.
    let digits = 100;
    let prec = cnum::digits_to_bits(digits);
    let b = parse(
        "2.71828182845904523536028747135266249775724709369995",
        "0",
        prec,
    );
    let h = parse("3", "0", prec);
    let f = dispatch::tetrate(&b, &h, prec, digits).unwrap();
    let expected = unrolled_tower(&b, 3, prec);
    let m = matching_digits(&f, &expected, prec);
    assert!(m >= 90, "e^^3 differs: matched only {} digits", m);
}

#[test]
fn t802_golden_complex_base_integer_height() {
    // (1+i)^^4 — complex base, integer height. Cross-check unrolled tower.
    let digits = 50;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("1", "1", prec);
    let h = parse("4", "0", prec);
    let f = dispatch::tetrate(&b, &h, prec, digits).unwrap();
    let expected = unrolled_tower(&b, 4, prec);
    let m = matching_digits(&f, &expected, prec);
    assert!(m >= 45, "(1+i)^^4 differs: matched only {} digits", m);
}

#[test]
fn t803_golden_negative_integer_height() {
    // F_b(-1) = log_b(1) = 0 by definition.
    let digits = 30;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("2.71828182845904523536", "0", prec);
    let h = parse("-1", "0", prec);
    let f = dispatch::tetrate(&b, &h, prec, digits).unwrap();
    assert!(
        abs(&f, prec) < cnum::epsilon(25, prec),
        "F_e(-1) ≈ {}, expected 0",
        abs(&f, prec)
    );
}

// ---------- Random functional-equation sampling ----------

/// Verify F(z+1) = b^F(z) at deterministic-but-spread-out points inside
/// Shell-Thron, where Schröder is fully applicable.
#[test]
fn t810_random_functional_eq_shell_thron() {
    let digits = 40;
    let prec = cnum::digits_to_bits(digits);
    let one = parse("1", "0", prec);
    // Bases scattered through Shell-Thron interior.
    let bases = [
        ("1.4142135623730950488", "0"),
        ("1.2", "0"),
        ("0.5", "0"),
        ("1.3", "0.1"),
        ("0.7", "0.2"),
        ("1.1", "-0.05"),
    ];
    let heights = [
        ("0.5", "0"),
        ("1.5", "0"),
        ("-0.3", "0"),
        ("0.5", "0.3"),
        ("0.1", "-0.2"),
    ];
    for (br, bi) in &bases {
        for (zr, zi) in &heights {
            let b = parse(br, bi, prec);
            let z = parse(zr, zi, prec);
            let z1 = Complex::with_val_64(prec, &z + &one);
            let fz = dispatch::tetrate(&b, &z, prec, digits).unwrap();
            let fz1 = dispatch::tetrate(&b, &z1, prec, digits).unwrap();
            let lhs = cnum::pow_complex(&b, &fz, prec);
            let m = matching_digits(&fz1, &lhs, prec);
            assert!(
                m >= digits - 5,
                "b={}+{}i z={}+{}i: matched only {} digits",
                br,
                bi,
                zr,
                zi,
                m
            );
        }
    }
}

// ---------- Continuity at integer heights ----------

#[test]
fn t820_continuity_multiple_directions() {
    // F(1 + ε·d) for tiny ε in several directions d ∈ {1, i, 1+i, …} all
    // approach F(1) = b. Verifies the dispatcher routes through the same
    // formula on all sides of an integer.
    let digits = 30;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("1.2", "0", prec); // Shell-Thron interior.
    let f_int = dispatch::tetrate(&b, &parse("1", "0", prec), prec, digits).unwrap();
    let directions = [
        ("1e-9", "0"),
        ("-1e-9", "0"),
        ("0", "1e-9"),
        ("0", "-1e-9"),
        ("7.07e-10", "7.07e-10"),
    ];
    for (er, ei) in &directions {
        let h = Complex::with_val_64(prec, &parse("1", "0", prec) + parse(er, ei, prec));
        let f = dispatch::tetrate(&b, &h, prec, digits).unwrap();
        let diff = Complex::with_val_64(prec, &f - &f_int);
        let da = abs(&diff, prec);
        assert!(
            da < cnum::epsilon(7, prec),
            "direction ε=({},{}): F(1+ε)−b = {}, expected ≪1",
            er,
            ei,
            da
        );
    }
}

// ---------- Precision scaling ----------

#[test]
fn t830_precision_scaling_p_2p_complex() {
    // Same input at p=40 and p=80 digits — first ~30 digits must agree.
    let digits_lo = 40;
    let digits_hi = 80;
    let prec_lo = cnum::digits_to_bits(digits_lo);
    let prec_hi = cnum::digits_to_bits(digits_hi);
    let b_lo = parse("1.3", "0.1", prec_lo);
    let b_hi = parse("1.3", "0.1", prec_hi);
    let h_lo = parse("0.4", "0.2", prec_lo);
    let h_hi = parse("0.4", "0.2", prec_hi);
    let f_lo = dispatch::tetrate(&b_lo, &h_lo, prec_lo, digits_lo).unwrap();
    let f_hi = dispatch::tetrate(&b_hi, &h_hi, prec_hi, digits_hi).unwrap();
    let f_lo_hi = Complex::with_val_64(prec_hi, &f_lo);
    let m = matching_digits(&f_lo_hi, &f_hi, prec_hi);
    assert!(
        m >= digits_lo - 5,
        "precision-scaling 40→80: only {} digits agree",
        m
    );
}

#[test]
fn t831_precision_scaling_p_2p_4p_real_interior() {
    let pairs = [(30, 60), (60, 120)];
    for (lo, hi) in pairs {
        let prec_lo = cnum::digits_to_bits(lo);
        let prec_hi = cnum::digits_to_bits(hi);
        let b_lo = parse("0.5", "0", prec_lo);
        let b_hi = parse("0.5", "0", prec_hi);
        let h_lo = parse("0.7", "0", prec_lo);
        let h_hi = parse("0.7", "0", prec_hi);
        let f_lo = dispatch::tetrate(&b_lo, &h_lo, prec_lo, lo).unwrap();
        let f_hi = dispatch::tetrate(&b_hi, &h_hi, prec_hi, hi).unwrap();
        let f_lo_hi = Complex::with_val_64(prec_hi, &f_lo);
        let m = matching_digits(&f_lo_hi, &f_hi, prec_hi);
        assert!(
            m >= lo - 5,
            "precision-scaling {}→{}: only {} digits agree",
            lo,
            hi,
            m
        );
    }
}

// ---------- Cross-validation: dispatch agrees with itself ----------

#[test]
fn t840_dispatch_idempotent_on_integer() {
    // F_b(n) by integer-iteration path == F_b(n + 0i). Tested at 50 digits.
    let digits = 50;
    let prec = cnum::digits_to_bits(digits);
    let bases = [("1.5", "0.5"), ("2", "0"), ("0.3", "0.7")];
    for (br, bi) in &bases {
        let b = parse(br, bi, prec);
        for n in 0..5 {
            let h_int = Complex::with_val_64(prec, (n, 0));
            let h_complex = parse(&format!("{n}.0"), "0.0", prec);
            let f1 = dispatch::tetrate(&b, &h_int, prec, digits).unwrap();
            let f2 = dispatch::tetrate(&b, &h_complex, prec, digits).unwrap();
            let m = matching_digits(&f1, &f2, prec);
            assert!(m >= 45, "b={}+{}i n={}: matched {} digits", br, bi, n, m);
        }
    }
}

#[test]
fn t850_kouznetsov_refuses_unchecked_real_base_asymptotes() {
    let digits = 15;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("2", "0", prec);
    let regions::Region::OutsideShellThronRealPositive(fp) = regions::classify(&b, prec).unwrap()
    else {
        panic!("unexpected base-2 region");
    };
    let state = kouznetsov::setup_kouznetsov(&b, &fp, prec, digits).unwrap();
    for (re, sign) in [(0, 1), (0, -1), (50, 1)] {
        let h = Complex::with_val_64(prec, (re, (state.t_max.clone() + 1) * sign));
        let error = kouznetsov::eval_kouznetsov(&state, &b, &h).unwrap_err();
        assert!(error.contains("outside the Cauchy contour"), "{error}");
    }
}

#[test]
fn t852_unit_circle_bases_functional_eq() {
    // The coarse rectangular grid misses these unit-circle directions.
    let digits = 25;
    let prec = cnum::digits_to_bits(digits);
    let one = parse("1", "0", prec);
    for (numerator, denominator) in [(1, 6), (1, 4), (1, 3), (1, 2), (3, 4)] {
        let angle: Float =
            Float::with_val_64(prec, rug::float::Constant::Pi) * numerator / denominator;
        let b = if denominator == 2 {
            parse("0", "1", prec)
        } else {
            Complex::with_val_64(prec, (angle.clone().cos(), angle.sin()))
        };
        for (zr, zi) in &[("0.4", "0"), ("0.5", "0.3"), ("-0.2", "0.1")] {
            let z = parse(zr, zi, prec);
            let z1 = Complex::with_val_64(prec, &z + &one);
            if numerator == 3 {
                let error = dispatch::tetrate(&b, &z, prec, digits).unwrap_err();
                assert!(
                    error.contains("unsupported case") && error.contains("residual"),
                    "{error}"
                );
                break;
            }
            let fz = dispatch::tetrate(&b, &z, prec, digits).unwrap();
            let fz1 = dispatch::tetrate(&b, &z1, prec, digits).unwrap();
            let lhs = cnum::pow_complex(&b, &fz, prec);
            let m = matching_digits(&fz1, &lhs, prec);
            assert!(
                m >= digits - 5,
                "b={} z={}+{}i: F(z+1) vs b^F(z) matched only {} digits",
                b,
                zr,
                zi,
                m
            );
        }
    }
}

#[test]
fn t851_kouznetsov_refuses_unchecked_complex_base_asymptotes() {
    let digits = 12;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("2", "0.1", prec);
    let regions::Region::OutsideShellThronGeneral(fp) = regions::classify(&b, prec).unwrap() else {
        panic!("unexpected near-real complex-base region");
    };
    let state = kouznetsov::setup_kouznetsov(&b, &fp, prec, digits).unwrap();
    let beyond: Float = state.t_max.clone() + state.shift.imag().clone().abs() + 1;
    for multiple in [1, -1, 2, -2] {
        let h = Complex::with_val_64(prec, (0, beyond.clone() * multiple));
        let error = kouznetsov::eval_kouznetsov(&state, &b, &h).unwrap_err();
        assert!(error.contains("outside the Cauchy contour"), "{error}");
    }
}

#[test]
fn t860_schwarz_reflection_conjugate_base() {
    // F_b(h) = conj(F_{b̄}(h̄)) for canonical Kneser tetration. Verify that
    // computing via conjugate base gives matching results for Im(b)<0 cases
    // that previously hung (wrong-basin normalization shift). Tested at 20
    // digits so each Kouznetsov setup completes in <600s.
    let digits = 20u64;
    let prec = cnum::digits_to_bits(digits);
    // Each entry: (b_re, b_im_pos, h_re, h_im) — dispatcher computes both
    // b=b_re+b_im_pos·i and b=b_re-b_im_pos·i and checks conj symmetry.
    //
    // The old -0.8+0.4i "reference" was a same-family discretization artifact
    // (FAILURE_CASES.md A.2). This case must refuse, not pass on symmetry alone.
    let cases = [
        ("1.2", "0.4", "0.5", "0"),   // Shell-Thron interior, quick
        ("1.2", "0.4", "0.5", "0.3"), // complex height
        ("-0.8", "0.4", "0.5", "0"),  // outside ST, near-real
    ];
    for (br, bi_pos, hr, hi) in &cases {
        let b_pos = parse(br, bi_pos, prec); // Im(b) > 0
        let b_neg = parse(br, &format!("-{}", bi_pos), prec); // Im(b) < 0 → Schwarz path
        let h = parse(hr, hi, prec);
        let h_conj = parse(hr, &format!("-{}", hi), prec);

        if *br == "-0.8" {
            for (base, height) in [(&b_pos, &h), (&b_neg, &h_conj)] {
                let error = dispatch::tetrate(base, height, prec, digits).unwrap_err();
                assert!(error.contains("unsupported case")
                    && (error.contains("residual")
                        || error.contains("Kouznetsov normalization: no grid seed produced Newton-converged root")),
                    "{error}");
            }
            continue;
        }
        let f_pos = dispatch::tetrate(&b_pos, &h, prec, digits).unwrap();
        let f_neg = dispatch::tetrate(&b_neg, &h_conj, prec, digits)
            .unwrap_or_else(|e| panic!("b={}-{}i h={}-{}i failed: {}", br, bi_pos, hr, hi, e));

        // f_neg should equal conj(f_pos)
        let f_pos_conj = Complex::with_val_64(prec, f_pos.conj_ref());
        let m = matching_digits(&f_neg, &f_pos_conj, prec);
        assert!(
            m >= digits - 5,
            "Schwarz symmetry failed for b={}±{}i h={}+{}i: F_b̄(h̄)={} but conj(F_b(h))={}, {} digits",
            br, bi_pos, hr, hi, f_neg, f_pos_conj, m
        );
    }
}

#[test]
fn t870_parabolic_boundary_refuses_unvalidated_extrapolation() {
    for digits in [20, 50, 70] {
        let prec = cnum::digits_to_bits(digits);
        let b = parse("1.4448", "0", prec);
        let h = parse("0.5", "0", prec);
        let error = dispatch::tetrate(&b, &h, prec, digits).unwrap_err();
        assert!(
            error.contains("unsupported case") && error.contains("unchecked polynomial"),
            "{error}"
        );
    }
}

#[test]
fn t871_near_parabolic_refusal_is_consistent_across_heights() {
    let digits = 25u64;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("1.444667861009766", "0", prec);
    for height in ["0.5", "1.5"] {
        let error = dispatch::tetrate(&b, &parse(height, "0", prec), prec, digits).unwrap_err();
        assert!(
            error.contains("unsupported case") && error.contains("unchecked polynomial"),
            "{error}"
        );
    }
}

#[test]
fn t872_parabolic_complex_heights_do_not_use_a_surrogate() {
    let digits = 25u64;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("1.4447", "0", prec);
    for height in ["0.5", "1.5"] {
        let error = dispatch::tetrate(&b, &parse(height, "0.5", prec), prec, digits).unwrap_err();
        assert!(
            error.contains("unsupported case") && error.contains("unchecked polynomial"),
            "{error}"
        );
    }
}

#[test]
#[ignore] // Slow: exercises both continuation and the direct fallback.
fn t880_parabolic_boundary_rejects_legacy_extrapolation() {
    // The committed implementation also stalled at the first continuation
    // step, then returned unchecked Richardson output. Its old "full
    // precision" reference was neither reproduced nor independently verified.
    let digits = 20u64;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("1.46", "0", prec);
    let h = parse("0.5", "0", prec);
    let error = dispatch::tetrate(&b, &h, prec, digits).unwrap_err();
    assert!(
        error.contains("unsupported case")
            && error.contains("continuation: Kouznetsov Newton did not converge")
            && error.contains("direct Kouznetsov: Kouznetsov Newton did not converge")
            && error.contains("unchecked polynomial extrapolation is not a tetration result"),
        "{error}"
    );
}

/// t890: deep-parabolic-band complex base must NOT return garbage with Ok.
///
/// Regression for the 2026-08-23 stalled-solve acceptance bug
/// (FAILURE_CASES § A.1): at b = 0.0653281554868594 + 0.025i
/// (|λ| = 0.995, deep in the Shell–Thron parabolic band on the
/// oscillating side) Schröder refuses, the Kouznetsov LM stalls at an
/// O(1) boundary residual, and the old relaxed acceptance (residual ≤ 5)
/// returned stalled samples as a final answer: RC=0 with values that
/// diverged to 10^6913 under upward iteration while the true orbit is
/// bounded (integer-height F(48) = 0.1353 − 0.0070i).
///
/// A bounded-looking answer is not enough to replace this refusal contract.
#[test]
fn t890_deep_band_complex_base_no_garbage() {
    let digits = 10u64;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("0.0653281554868594", "0.025", prec);
    let h = parse("48.013", "0", prec);
    let error = dispatch::tetrate(&b, &h, prec, digits).unwrap_err();
    assert!(
        error.contains("unsupported case")
            && (error.contains("residual")
                || error.contains(
                    "Kouznetsov normalization: no grid seed produced Newton-converged root"
                )),
        "{error}"
    );
}
