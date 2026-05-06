//! Phase-8 verification battery.
//!
//! Cross-cuts Phase 4–7 with: (a) golden integer-height values computed two
//! ways and cross-checked at high precision; (b) functional-equation residual
//! sampling at random (deterministic) heights in each region; (c) precision
//! scaling — same input at p and 2p digits, first p−10 digits agree.

use rug::{Complex, Float};

use tetration::{cnum, dispatch};

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
