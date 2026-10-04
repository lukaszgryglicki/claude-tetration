//! Complex Lambert W, using arbitrary-precision seeds and Halley refinement.
//! Root convergence and branch identity are checked separately.

use rug::{float::Constant, Complex, Float, Integer};

use crate::cnum;

pub fn w0(z: &Complex, prec: u64) -> Result<Complex, String> {
    let work_prec = working_precision(z, prec)?;
    w0_at_precision(z, work_prec, prec).map(|w| Complex::with_val_64(prec, w))
}

fn working_precision(z: &Complex, prec: u64) -> Result<u64, String> {
    cnum::init_mpfr();
    cnum::check_precision(prec)?;
    if prec == 0 || !cnum::is_finite(z) {
        return Err("Lambert W requires finite input and positive working precision".into());
    }
    let distance = Complex::with_val_64(prec, z + Float::with_val_64(prec, -1).exp());
    if cnum::abs(&distance, prec) < cnum::decimal("0.1", prec) {
        // Near -1/e a rounded zero residual can conceal a square-root-sized
        // root error. Refine with extra bits before rounding the returned W.
        let guarded = prec
            .checked_mul(2)
            .and_then(|bits| bits.checked_add(64))
            .ok_or_else(|| {
                "Lambert W branch-point guard precision exceeds the supported bit range".to_string()
            })?;
        cnum::check_precision(guarded)?;
        Ok(guarded)
    } else {
        Ok(prec)
    }
}

fn w0_at_precision(z: &Complex, prec: u64, target_prec: u64) -> Result<Complex, String> {
    if !cnum::is_finite(z) {
        return Err("W_0: argument must be finite".into());
    }
    if cnum::is_zero(z) {
        return Ok(cnum::zero(prec));
    }
    let log_z = Complex::with_val_64(prec, z.ln_ref());
    let log_log_z = Complex::with_val_64(prec, log_z.ln_ref());
    let asymptotic = Complex::with_val_64(prec, &log_z - &log_log_z);
    let correction = Complex::with_val_64(prec, &log_log_z / &log_z);
    let zp1 = Complex::with_val_64(prec, z + 1);
    let seeds = [
        initial_guess_w0(z, prec),
        asymptotic.clone(),
        Complex::with_val_64(prec, &asymptotic + &correction),
        Complex::with_val_64(prec, zp1.ln_ref()),
        branch_point_seed(z, false, prec),
    ];
    let mut last_error = "W_0: no finite seed".to_string();
    for mut w in seeds {
        if !cnum::is_finite(&w) {
            continue;
        }
        match halley_refine(&mut w, z, prec, target_prec)
            .and_then(|_| verify_branch(&w, z, 0, prec, target_prec))
        {
            Ok(()) => return Ok(w),
            Err(e) => last_error = e,
        }
    }
    Err(last_error)
}

pub fn wm1(z: &Complex, prec: u64) -> Result<Complex, String> {
    nonprincipal(z, -1, prec)
}

pub fn wk(z: &Complex, k: i32, prec: u64) -> Result<Complex, String> {
    if k == 0 {
        w0(z, prec)
    } else {
        nonprincipal(z, k, prec)
    }
}

fn nonprincipal(z: &Complex, k: i32, prec: u64) -> Result<Complex, String> {
    let work_prec = working_precision(z, prec)?;
    nonprincipal_at_precision(z, k, work_prec, prec).map(|w| Complex::with_val_64(prec, w))
}

fn nonprincipal_at_precision(
    z: &Complex,
    k: i32,
    prec: u64,
    target_prec: u64,
) -> Result<Complex, String> {
    if !cnum::is_finite(z) {
        return Err(format!("W_{k}: argument must be finite"));
    }
    if cnum::is_zero(z) {
        return Err(format!("W_{k}(0) is undefined"));
    }
    let mut l1 = Complex::with_val_64(prec, z.ln_ref());
    let turn = Float::with_val_64(prec, Constant::Pi) * 2 * k;
    l1 += Complex::with_val_64(prec, (0, turn));
    let l2 = Complex::with_val_64(prec, l1.ln_ref());
    let mut seed = Complex::with_val_64(prec, &l1 - &l2);
    seed += Complex::with_val_64(prec, &l2 / &l1);
    let distance = Complex::with_val_64(prec, z + Float::with_val_64(prec, -1).exp());
    if k == -1 && *z.imag() >= 0 && cnum::abs(&distance, prec) < cnum::decimal("0.3", prec) {
        let mut w = branch_point_seed(z, true, prec);
        if halley_refine(&mut w, z, prec, target_prec)
            .and_then(|_| verify_branch(&w, z, k, prec, target_prec))
            .is_ok()
        {
            return Ok(w);
        }
    }
    halley_refine(&mut seed, z, prec, target_prec)?;
    verify_branch(&seed, z, k, prec, target_prec)?;
    Ok(seed)
}

fn halley_refine(w: &mut Complex, z: &Complex, prec: u64, target_prec: u64) -> Result<(), String> {
    let target = Float::with_val_64(prec, 1)
        >> usize::try_from(target_prec.saturating_sub(8))
            .expect("precision exceeds addressable bits");
    let acceptable = Float::with_val_64(prec, cnum::working_epsilon(target_prec));
    let mut best_delta = Float::with_val_64(prec, rug::float::Special::Infinity);
    let mut best_w = w.clone();
    let mut checkpoint = w.clone();
    let mut iter = Integer::new();
    loop {
        let exp_w = Complex::with_val_64(prec, w.exp_ref());
        let f = Complex::with_val_64(prec, Complex::with_val_64(prec, &*w * &exp_w) - z);
        if !cnum::is_finite(&f) {
            return Err(format!("Halley non-finite residual at iteration {iter}"));
        }
        if cnum::is_zero(&f) {
            return Ok(());
        }
        let wp1 = Complex::with_val_64(prec, &*w + 1);
        let wp2 = Complex::with_val_64(prec, &*w + 2);
        let correction = Complex::with_val_64(
            prec,
            Complex::with_val_64(prec, &wp2 * &f) / Complex::with_val_64(prec, &wp1 * 2),
        );
        let denominator =
            Complex::with_val_64(prec, Complex::with_val_64(prec, &wp1 * &exp_w) - correction);
        if cnum::is_zero(&denominator) || !cnum::is_finite(&denominator) {
            return Err(format!("Halley singular denominator at iteration {iter}"));
        }
        let delta = Complex::with_val_64(prec, &f / &denominator);
        let next = Complex::with_val_64(prec, &*w - &delta);
        if !cnum::is_finite(&delta) || !cnum::is_finite(&next) {
            return Err(format!("Halley non-finite update at iteration {iter}"));
        }
        let scale = cnum::abs(&next, prec).max(&Float::with_val_64(prec, 1));
        let relative_delta = cnum::abs(&delta, prec) / scale;
        *w = next;
        if relative_delta < best_delta {
            best_delta = relative_delta.clone();
            best_w = w.clone();
        }
        if relative_delta <= target {
            return Ok(());
        }
        if *w == checkpoint {
            break;
        }
        if iter.is_power_of_two() {
            checkpoint = w.clone();
        }
        iter += 1;
    }
    if best_delta <= acceptable {
        *w = best_w;
        Ok(())
    } else {
        Err(format!(
            "Halley did not converge: repeated iterate with relative update {}",
            cnum::DisplayFloat(&best_delta)
        ))
    }
}

fn verify_branch(
    w: &Complex,
    z: &Complex,
    k: i32,
    prec: u64,
    target_prec: u64,
) -> Result<(), String> {
    if !cnum::is_finite(w) {
        return Err(format!("W_{k}: non-finite root"));
    }
    let tolerance = Float::with_val_64(prec, cnum::working_epsilon(target_prec));
    let residual = Complex::with_val_64(
        prec,
        Complex::with_val_64(prec, w * Complex::with_val_64(prec, w.exp_ref())) - z,
    );
    if !cnum::is_finite(&residual)
        || cnum::abs(&residual, prec) > Float::with_val_64(prec, &tolerance * cnum::abs(z, prec))
    {
        return Err(format!("W_{k}: root residual exceeds working precision"));
    }
    let branch_point = -Float::with_val_64(prec, -1).exp();
    if z.imag().is_zero() && *z.real() < 0 && *z.real() >= branch_point && (k == 0 || k == -1) {
        let on_real_branch = w.imag().clone().abs() <= tolerance
            && if k == 0 {
                *w.real() >= -1
            } else {
                *w.real() <= -1
            };
        return if on_real_branch {
            Ok(())
        } else {
            Err(format!("W_{k}: wrong real branch"))
        };
    }
    // Taking principal logs labels the root away from the real two-root interval.
    let mut branch_residual =
        Complex::with_val_64(prec, w + Complex::with_val_64(prec, w.ln_ref()));
    branch_residual -= Complex::with_val_64(prec, z.ln_ref());
    branch_residual -=
        Complex::with_val_64(prec, (0, Float::with_val_64(prec, Constant::Pi) * 2 * k));
    let scale = cnum::abs(w, prec).max(&Float::with_val_64(prec, 1));
    if cnum::abs(&branch_residual, prec) > tolerance * scale * 8 {
        Err(format!(
            "W_{k}: converged to a different logarithmic branch"
        ))
    } else {
        Ok(())
    }
}

fn branch_point_seed(z: &Complex, negative: bool, prec: u64) -> Complex {
    let mut p = Complex::with_val_64(
        prec,
        Complex::with_val_64(prec, z * Float::with_val_64(prec, 1).exp()) + 1,
    );
    p *= 2;
    p.sqrt_mut();
    if negative {
        p = -p;
    }
    let p2 = Complex::with_val_64(prec, &p * &p);
    let p3 = Complex::with_val_64(prec, &p2 * &p);
    let p4 = Complex::with_val_64(prec, &p3 * &p);
    let mut w = Complex::with_val_64(prec, &p - 1);
    w -= Complex::with_val_64(prec, &p2 / 3);
    w += Complex::with_val_64(prec, p3 * 11 / 72);
    w -= Complex::with_val_64(prec, p4 * 43 / 540);
    w
}

fn initial_guess_w0(z: &Complex, prec: u64) -> Complex {
    let distance = Complex::with_val_64(prec, z + Float::with_val_64(prec, -1).exp());
    if cnum::abs(&distance, prec) < cnum::decimal("0.5", prec) {
        return branch_point_seed(z, false, prec);
    }
    let magnitude = cnum::abs(z, prec);
    if magnitude < cnum::decimal("0.3", prec) {
        let z2 = Complex::with_val_64(prec, z * z);
        let z3 = Complex::with_val_64(prec, &z2 * z);
        let z4 = Complex::with_val_64(prec, &z3 * z);
        let mut w = Complex::with_val_64(prec, z - z2);
        w += Complex::with_val_64(prec, z3 * 3 / 2);
        w -= Complex::with_val_64(prec, z4 * 8 / 3);
        return w;
    }
    let zp1 = Complex::with_val_64(prec, z + 1);
    if magnitude <= 5 && cnum::abs(&zp1, prec) > cnum::decimal("0.3", prec) {
        if *zp1.real() >= 0 {
            return Complex::with_val_64(prec, zp1.ln_ref());
        }
        let im = if *z.imag() >= 0 { "1.337" } else { "-1.337" };
        return Complex::with_val_64(
            prec,
            (cnum::decimal("-0.318", prec), cnum::decimal(im, prec)),
        );
    }
    let l1 = Complex::with_val_64(prec, z.ln_ref());
    Complex::with_val_64(prec, &l1 - Complex::with_val_64(prec, l1.ln_ref()))
}
