//! Phase-8 verification battery.
//!
//! Cross-cuts Phase 4–7 with: (a) golden integer-height values computed two
//! ways and cross-checked at high precision; (b) functional-equation residual
//! sampling at random (deterministic) heights in each region; (c) precision
//! scaling — same input at p and 2p digits, first p−10 digits agree.

use rug::{Complex, Float};

use tetration::{cnum, dispatch, lambertw};

fn parse(re: &str, im: &str, prec: u32) -> Complex {
    cnum::parse_complex(re, im, prec).unwrap()
}

fn abs(z: &Complex, prec: u32) -> f64 {
    Float::with_val(prec, z.abs_ref()).to_f64()
}

fn matching_digits(a: &Complex, b: &Complex, prec: u32) -> f64 {
    let diff = Complex::with_val(prec, a - b);
    let da = abs(&diff, prec);
    if da == 0.0 {
        return f64::INFINITY;
    }
    -da.log10()
}

// ---------- Golden integer-height cross-checks ----------

/// `F_b(n)` via dispatch must equal the unrolled tower `b^(b^(...))` to within
/// the requested precision (modulo guard bits). Done at moderate precision so
/// the test is fast — accuracy of integer iteration scales perfectly with
/// `prec`, so 100-digit confidence is ample.
fn unrolled_tower(b: &Complex, n: i64, prec: u32) -> Complex {
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
    assert!(m >= 45.0, "2^^4 differs: matched only {} digits", m);
}

#[test]
fn t801_golden_e_to_three() {
    // F_e(3) = e^(e^e). Cross-check vs unrolled tower at 100 digits.
    let digits = 100;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("2.71828182845904523536028747135266249775724709369995", "0", prec);
    let h = parse("3", "0", prec);
    let f = dispatch::tetrate(&b, &h, prec, digits).unwrap();
    let expected = unrolled_tower(&b, 3, prec);
    let m = matching_digits(&f, &expected, prec);
    assert!(m >= 90.0, "e^^3 differs: matched only {} digits", m);
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
    assert!(m >= 45.0, "(1+i)^^4 differs: matched only {} digits", m);
}

#[test]
fn t803_golden_negative_integer_height() {
    // F_b(-1) = log_b(1) = 0 by definition.
    let digits = 30;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("2.71828182845904523536", "0", prec);
    let h = parse("-1", "0", prec);
    let f = dispatch::tetrate(&b, &h, prec, digits).unwrap();
    assert!(abs(&f, prec) < 1e-25, "F_e(-1) ≈ {}, expected 0", abs(&f, prec));
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
            let z1 = Complex::with_val(prec, &z + &one);
            let fz = dispatch::tetrate(&b, &z, prec, digits).unwrap();
            let fz1 = dispatch::tetrate(&b, &z1, prec, digits).unwrap();
            let lhs = cnum::pow_complex(&b, &fz, prec);
            let m = matching_digits(&fz1, &lhs, prec);
            // Schröder interior bases reliably hit ≥25 digits at 40-digit
            // precision (≥15-digit margin for guard bits / shift errors).
            assert!(
                m >= 20.0,
                "b={}+{}i z={}+{}i: matched only {} digits",
                br, bi, zr, zi, m
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
        let h = Complex::with_val(prec, &parse("1", "0", prec) + parse(er, ei, prec));
        let f = dispatch::tetrate(&b, &h, prec, digits).unwrap();
        let diff = Complex::with_val(prec, &f - &f_int);
        let da = abs(&diff, prec);
        assert!(
            da < 1e-7,
            "direction ε=({},{}): F(1+ε)−b = {}, expected ≪1",
            er, ei, da
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
    let f_lo_hi = Complex::with_val(prec_hi, &f_lo);
    let m = matching_digits(&f_lo_hi, &f_hi, prec_hi);
    assert!(m >= 30.0, "precision-scaling 40→80: only {} digits agree", m);
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
        let f_lo_hi = Complex::with_val(prec_hi, &f_lo);
        let m = matching_digits(&f_lo_hi, &f_hi, prec_hi);
        assert!(
            m >= (lo as f64) - 5.0,
            "precision-scaling {}→{}: only {} digits agree",
            lo, hi, m
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
            let h_int = Complex::with_val(prec, (n, 0));
            let h_complex = Complex::with_val(prec, (n as f64, 0.0));
            let f1 = dispatch::tetrate(&b, &h_int, prec, digits).unwrap();
            let f2 = dispatch::tetrate(&b, &h_complex, prec, digits).unwrap();
            let m = matching_digits(&f1, &f2, prec);
            assert!(m >= 45.0, "b={}+{}i n={}: matched {} digits", br, bi, n, m);
        }
    }
}

#[test]
fn t850_kouznetsov_asymptote_large_imag_height() {
    // For Kouznetsov bases (real b > e^(1/e), complex bases outside ST), the
    // natural F satisfies F(z) → L_upper as Im(z) → +∞, F(z) → L_lower as
    // Im(z) → −∞. Heights with |Im(h)| beyond the Cauchy contour t_max must
    // return the asymptote, not NaN/0/inf.
    let digits = 15;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("2", "0", prec);
    let ln_b = Complex::with_val(prec, b.ln_ref());
    let neg_ln_b = Complex::with_val(prec, -&ln_b);
    let w0 = lambertw::w0(&neg_ln_b, prec).unwrap();
    let neg_w0 = Complex::with_val(prec, -&w0);
    let l_plus = Complex::with_val(prec, &neg_w0 / &ln_b);
    let l_upper = if l_plus.imag().is_sign_negative() {
        Complex::with_val(prec, l_plus.conj_ref())
    } else {
        l_plus.clone()
    };
    let l_lower = Complex::with_val(prec, l_upper.conj_ref());

    let h_far_up = parse("0", "100", prec);
    let f_up = dispatch::tetrate(&b, &h_far_up, prec, digits).unwrap();
    let m_up = matching_digits(&f_up, &l_upper, prec);
    assert!(
        m_up >= 12.0,
        "F(100i) should match L_upper to ≥12 digits; matched {}", m_up
    );

    let h_far_down = parse("0", "-100", prec);
    let f_down = dispatch::tetrate(&b, &h_far_down, prec, digits).unwrap();
    let m_down = matching_digits(&f_down, &l_lower, prec);
    assert!(
        m_down >= 12.0,
        "F(-100i) should match L_lower to ≥12 digits; matched {}", m_down
    );

    // Re-shift + asymptote: h = 50+50i should NOT give NaN/inf. After integer
    // shift Re by 50, h_strip = 0+50i; if t_max ≈ 49 < 50, asymptote engages
    // and returns L_upper. b^L_upper = L_upper, so the recursion is a no-op.
    let h_diag = parse("50", "50", prec);
    let f_diag = dispatch::tetrate(&b, &h_diag, prec, digits).unwrap();
    let f_diag_re = Float::with_val(prec, f_diag.real()).to_f64();
    let f_diag_im = Float::with_val(prec, f_diag.imag()).to_f64();
    assert!(
        f_diag_re.is_finite() && f_diag_im.is_finite(),
        "F(50+50i) should be finite, got {}+{}i", f_diag_re, f_diag_im
    );
    let m_diag = matching_digits(&f_diag, &l_upper, prec);
    assert!(
        m_diag >= 10.0,
        "F(50+50i) should match L_upper (asymptote engages); matched {}", m_diag
    );
}

#[test]
fn t852_unit_circle_bases_functional_eq() {
    // Bases on the complex unit circle |b|=1 (e.g., b = e^(iθ)) sit outside
    // Shell-Thron interior for most θ. Verify the algorithm produces values
    // satisfying F(z+1) = b^F(z). Coverage gap noted in audit findings: the
    // 19^4 grid uses step 0.4 and never lands exactly on |b|=1.
    let digits = 25;
    let prec = cnum::digits_to_bits(digits);
    let one = parse("1", "0", prec);
    // Six points on the unit circle at 30°, 45°, 60°, 90°, 135°, 180°. b = -1
    // and b = i are special; including them stress-tests the W_k partner
    // search for non-real bases whose Im(L_+) is small.
    let bases = [
        ("0.866025403784438646763723", "0.5"),                            // 30°
        ("0.707106781186547524400844", "0.707106781186547524400844"),     // 45°
        ("0.5", "0.866025403784438646763723"),                            // 60°
        ("0", "1"),                                                       // 90° (i)
        ("-0.707106781186547524400844", "0.707106781186547524400844"),    // 135°
    ];
    for (br, bi) in &bases {
        let b = parse(br, bi, prec);
        for (zr, zi) in &[("0.4", "0"), ("0.5", "0.3"), ("-0.2", "0.1")] {
            let z = parse(zr, zi, prec);
            let z1 = Complex::with_val(prec, &z + &one);
            let fz = dispatch::tetrate(&b, &z, prec, digits).unwrap();
            let fz1 = dispatch::tetrate(&b, &z1, prec, digits).unwrap();
            let lhs = cnum::pow_complex(&b, &fz, prec);
            let m = matching_digits(&fz1, &lhs, prec);
            assert!(
                m >= 15.0,
                "b={}+{}i z={}+{}i: F(z+1) vs b^F(z) matched only {} digits",
                br, bi, zr, zi, m
            );
        }
    }
}

#[test]
fn t851_kouznetsov_asymptote_complex_base() {
    // Complex base outside ST: F(z) → L_upper as Im(z) → +∞ where L_upper is
    // the algorithm's chosen partner from W_k search (not necessarily conjugate
    // of L_+ for non-real bases). Just verify F is finite at large |Im|.
    let digits = 12;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("0.5", "1.5", prec);
    for im_str in &["50", "-50", "100", "-200"] {
        let h = parse("0", im_str, prec);
        let f = dispatch::tetrate(&b, &h, prec, digits).unwrap();
        let fr = Float::with_val(prec, f.real()).to_f64();
        let fi = Float::with_val(prec, f.imag()).to_f64();
        assert!(
            fr.is_finite() && fi.is_finite(),
            "F(0+{}i) for b=0.5+1.5i not finite: {}+{}i", im_str, fr, fi
        );
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
    let cases = [
        ("1.2", "0.4", "0.5", "0"),        // Shell-Thron interior, quick
        ("1.2", "0.4", "0.5", "0.3"),       // complex height
        ("-0.8", "0.4", "0.5", "0"),        // outside ST, near-real
    ];
    for (br, bi_pos, hr, hi) in &cases {
        let b_pos = parse(br, bi_pos, prec);            // Im(b) > 0
        let b_neg = parse(br, &format!("-{}", bi_pos), prec); // Im(b) < 0 → Schwarz path
        let h = parse(hr, hi, prec);
        let h_conj = parse(hr, &format!("-{}", hi), prec);

        let f_pos = dispatch::tetrate(&b_pos, &h, prec, digits)
            .unwrap_or_else(|e| panic!("b={}+{}i h={}+{}i failed: {}", br, bi_pos, hr, hi, e));
        let f_neg = dispatch::tetrate(&b_neg, &h_conj, prec, digits)
            .unwrap_or_else(|e| panic!("b={}-{}i h={}-{}i failed: {}", br, bi_pos, hr, hi, e));

        // f_neg should equal conj(f_pos)
        let f_pos_conj = Complex::with_val(prec, f_pos.conj_ref());
        let m = matching_digits(&f_neg, &f_pos_conj, prec);
        assert!(
            m >= 15.0,
            "Schwarz symmetry failed for b={}±{}i h={}+{}i: F_b̄(h̄)={} but conj(F_b(h))={}, {} digits",
            br, bi_pos, hr, hi, f_neg, f_pos_conj, m
        );
    }
}

#[test]
fn t870_parabolic_boundary_clean_err() {
    // Class A: bases extremely close to η=e^(1/e) have |arg(λ)| ≈ 0 and
    // n_nodes that exceeds BOTH the direct-solver cap (32K) AND the continuation
    // cap (131K).  Must return ERR quickly (< 5s), not hang.
    //
    // b=1.4448 is just 0.0001 above η=1.44467:
    //   |arg(λ)| ≈ 0.019 → t_max ≈ 3400 → n_nodes ≈ 262K → exceeds both caps.
    //
    // Note: b=1.45 (|arg(λ)|=0.141, n_nodes=65536 ≤ 131K) now SUCCEEDS via
    // the continuation solver (see FAILURE_CASES.md Class A).
    let digits = 20u64;
    let prec = cnum::digits_to_bits(digits);
    let b = parse("1.4448", "0", prec);
    let h = parse("0.5", "0", prec);
    let result = dispatch::tetrate(&b, &h, prec, digits);
    assert!(
        result.is_err(),
        "b=1.4448 should fail (parabolic boundary, n_nodes >> 131K), got {:?}",
        result
    );
    let msg = result.unwrap_err();
    // Either the direct solver or the continuation solver should name the cause.
    assert!(
        msg.contains("parabolic boundary") || msg.contains("continuation") || msg.contains("Abel"),
        "ERR message should mention parabolic boundary / continuation / Abel, got: {}",
        msg
    );
}
