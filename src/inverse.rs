//! Inverses of tetration.
//!
//! * Super-logarithm `slog_b(z)`: the height `h` with `F_b(h) = z`.
//!   - Strictly attracting bases: `h = Log(sigma(z)/sigma(1)) / Log(lambda)` with
//!     the Schroder coordinate `sigma`; the principal `Log` makes `slog_b(1) = 0`.
//!   - Real bases above `e^(1/e)`: the real inverse on the real axis, continued
//!     to complex `z` without crossing the rays `{x +- i Im(L) : x <= Re(L)}`.
//!   - Other bases: Newton continuation of `F_b(0) = 1` along the segment `1 -> z`.
//! * Super-root: the base `b` with `F_b(h) = z`. Real `z > 1`, `h > 0` use a
//!   bracket in `b > 1`; other inputs continue that solution to `(z, h)`.
//!
//! Every result is checked by forward evaluation; a failed continuation is an
//! error, never a guess.

use rug::{float::Constant, Complex, Float, Integer};

use crate::{
    cnum, dispatch, kneser, kneser_complex, regions, schroder, schroder_jumps::OrbitJumps,
};
use cnum::DisplayFloat;

/// Relative residual, in bits, at which an intermediate Newton solve of a
/// continuation is accepted.
const LOOSE_BITS: usize = 24;

/// `f(x, t)` and its derivative in `x`, for a path parameter `t`.
type Forward<'a> = dyn FnMut(&Complex, &Float) -> Result<(Complex, Complex), String> + 'a;

/// Newton iteration for `f(x, t) = goal` from `x`. Intermediate goals stop at a
/// loose relative residual; a polished goal runs to the residual floor.
/// Leaving the trust region or stalling above the loose level is a failure.
fn newton(
    f: &mut Forward,
    start: &Complex,
    t: &Float,
    goal: &Complex,
    polish: bool,
    trust: &dyn Fn(&Complex, &Complex) -> bool,
    prec: u64,
) -> Result<(Complex, Float), String> {
    let accept = cnum::abs(goal, prec).max(&Float::with_val_64(prec, 1)) >> LOOSE_BITS;
    let mut x = start.clone();
    let (mut value, mut slope) = f(&x, t)?;
    let mut error = cnum::abs(&Complex::with_val_64(prec, &value - goal), prec);
    loop {
        if error.is_zero() || (!polish && error <= accept) {
            return Ok((x, error));
        }
        let stalled = |reason: &str| {
            if error <= accept {
                Ok((x.clone(), error.clone()))
            } else {
                Err(reason.to_string())
            }
        };
        if cnum::is_zero(&slope) || !cnum::is_finite(&slope) {
            return stalled("inverse iteration met a singular derivative");
        }
        let step = Complex::with_val_64(prec, &value - goal) / &slope;
        if !trust(&x, &step) {
            return stalled("inverse iteration left its trust region");
        }
        let candidate = Complex::with_val_64(prec, &x - &step);
        let (next_value, next_slope) = match f(&candidate, t) {
            Ok(pair) => pair,
            Err(reason) => return stalled(&reason),
        };
        let next_error = cnum::abs(&Complex::with_val_64(prec, &next_value - goal), prec);
        // Less than one part in 1024: the iteration has stopped converging.
        let slow = Float::with_val_64(prec, &error - Float::with_val_64(prec, &error >> 10u32));
        if next_error >= slow {
            if next_error < error && next_error <= accept {
                return Ok((candidate, next_error));
            }
            return stalled("inverse iteration did not reduce the residual");
        }
        x = candidate;
        value = next_value;
        slope = next_slope;
        error = next_error;
    }
}

/// Follow a root of `f(x, t) = goal(t)` from `t = 0` (root `x`) to `t = end`.
/// Steps are halved until Newton converges; the endpoint is polished.
fn follow(
    f: &mut Forward,
    goal: &dyn Fn(&Float) -> Complex,
    mut x: Complex,
    end: u32,
    trust: &dyn Fn(&Complex, &Complex) -> bool,
    prec: u64,
) -> Result<(Complex, Float), String> {
    let end = Float::with_val_64(prec, end);
    let mut t = Float::new_64(prec);
    let mut step = Float::with_val_64(prec, 1);
    loop {
        let reach = Float::with_val_64(prec, &t + &step).min(&end);
        let last = reach >= end;
        match newton(f, &x, &reach, &goal(&reach), last, trust, prec) {
            Ok((next, residual)) => {
                if last {
                    return Ok((next, residual));
                }
                x = next;
                t = reach;
                step *= 2u32;
            }
            Err(reason) => {
                step /= 2u32;
                if cnum::verbose() {
                    eprintln!(
                        "inverse: continuation step rejected at t = {:.6} ({reason})",
                        DisplayFloat(&t)
                    );
                }
                // A step of the path parameter below the loose Newton tolerance
                // moves the goal by less than that tolerance: the path is blocked.
                if step < Float::with_val_64(prec, 1) >> LOOSE_BITS {
                    return Err(format!(
                        "continuation stalled at path parameter {:.6} ({reason}); \
                         the target may lie on a branch cut or at a singularity",
                        DisplayFloat(&t)
                    ));
                }
            }
        }
    }
}

/// Position `t` on the polyline `points[0] -> points[1] -> ...` (unit time per leg).
fn polyline(points: &[Complex], t: &Float, prec: u64) -> Complex {
    let last = points.len() - 1;
    let leg = t
        .clone()
        .floor()
        .to_integer()
        .and_then(|leg| leg.to_usize())
        .unwrap_or(last)
        .min(last.saturating_sub(1));
    if last == 0 {
        return points[0].clone();
    }
    let fraction = Float::with_val_64(prec, t - Float::with_val_64(prec, leg));
    let span = Complex::with_val_64(prec, &points[leg + 1] - &points[leg]);
    Complex::with_val_64(prec, &points[leg] + span * fraction)
}

fn decimal_loss(log_factor: &Float) -> Result<u64, String> {
    if !log_factor.is_finite() {
        return Err("inverse conditioning is not finite".into());
    }
    let lost = (log_factor.clone() / Float::with_val_64(64, 10).ln())
        .ceil()
        .max(&Float::new_64(64));
    lost.to_integer()
        .and_then(|lost| lost.to_u64())
        .ok_or_else(|| "inverse conditioning exceeds the supported precision count".into())
}

/// Precision of a returned value: above `prec` only when the inputs parsed at
/// `prec` can no longer carry the requested digits.
fn output_precision(prec: u64, digits: u64, lost: u64) -> Result<u64, String> {
    if lost <= 12 {
        return Ok(prec);
    }
    let needed = digits
        .checked_add(lost)
        .ok_or("inverse conditioning exceeds the supported precision count")?;
    Ok(prec.max(cnum::checked_digits_to_bits(needed)?))
}

fn principal(mut w: Complex, prec: u64) -> Complex {
    let pi = Float::with_val_64(prec, Constant::Pi);
    let two_pi = Float::with_val_64(prec, &pi * 2u32);
    let turns = Float::with_val_64(prec, w.imag() / &two_pi).round();
    *w.mut_imag() -= turns * &two_pi;
    if *w.imag() <= -pi.clone() {
        *w.mut_imag() += &two_pi;
    } else if *w.imag() > pi {
        *w.mut_imag() -= &two_pi;
    }
    w
}

/// The Schroder coordinate of `z` as `(t, steps)`: `sigma(z) = t / lambda^steps`.
fn schroder_coordinate(
    state: &schroder::SchroderState,
    z: &Complex,
) -> Result<(Complex, Integer), String> {
    let prec = state.prec;
    let radius = state
        .inverse_radius
        .as_ref()
        .ok_or("super-logarithm needs a strictly attracting regular state")?;
    let reach = Float::with_val_64(prec, radius / 2u32);
    let mut jumps = OrbitJumps::new(&state.ln_b, &state.l, false, prec)?;
    let mut point = Complex::with_val_64(prec, z);
    let mut steps = Integer::new();
    let mut updates = Integer::new();
    let mut checkpoint = point.clone();
    let coordinate = loop {
        let displacement = Complex::with_val_64(prec, &point - &state.l);
        if cnum::is_zero(&displacement) {
            return Err("super-logarithm is singular at the fixed point of the base".into());
        }
        if cnum::abs(&displacement, prec) < reach {
            if let Some(t) = invert_poincare(state, radius, &displacement) {
                break t;
            }
        }
        let jump = match &mut jumps {
            Some(cache) => cache.advance(&point, None)?,
            None => None,
        };
        if let Some(jump) = jump {
            point = jump.value;
            steps += jump.steps;
        } else {
            let exponent = Complex::with_val_64(prec, &point * &state.ln_b);
            point = cnum::checked_exp(&exponent, prec).map_err(|e| {
                format!("the orbit of z leaves the basin of the attracting fixed point ({e})")
            })?;
            steps += 1;
        }
        updates += 1;
        if !cnum::is_finite(&point) || point == checkpoint {
            return Err("the orbit of z does not approach the attracting fixed point".into());
        }
        if updates.is_power_of_two() {
            checkpoint = point.clone();
            if cnum::verbose() {
                eprintln!("slog: forward orbit, {steps} steps");
            }
        }
    };
    Ok((coordinate, steps))
}

/// `Log(sigma(z)/sigma(1)) / Log(lambda)` for a strictly attracting state, and
/// the logarithm of the factor by which a relative error of `z` (or of the
/// arithmetic) grows in the relative error of the result.
fn regular_slog(state: &schroder::SchroderState, z: &Complex) -> Result<(Complex, Float), String> {
    let prec = state.prec;
    let log_anchor = cnum::ln_complex(&state.s1, prec);
    let raw = |z: &Complex| -> Result<(Complex, Complex, Integer, Float), String> {
        let (coordinate, steps) = schroder_coordinate(state, z)?;
        let log_coordinate = cnum::ln_complex(&coordinate, prec);
        let turns = Complex::with_val_64(prec, &state.ln_lambda * &steps);
        let scale = cnum::abs(&log_coordinate, prec)
            + cnum::abs(&log_anchor, prec)
            + cnum::abs(&turns, prec)
            + 4u32;
        let w = Complex::with_val_64(prec, &log_coordinate - &log_anchor) - turns;
        Ok((w, coordinate, steps, scale))
    };
    let (unreduced, coordinate, steps, scale) = raw(z)?;
    let mut w = principal(unreduced.clone(), prec);
    let real_multiplier = state.l.imag().is_zero() && state.ln_b.imag().is_zero();
    if real_multiplier && coordinate.imag().is_zero() && state.s1.imag().is_zero() {
        // On the real axis the imaginary part is exactly 0 or pi; rounding
        // must not choose between pi and -pi.
        let mut odd = coordinate.real().is_sign_negative() != state.s1.real().is_sign_negative();
        if !state.ln_lambda.imag().is_zero() && steps.is_odd() {
            odd = !odd;
        }
        *w.mut_imag() = if odd {
            Float::with_val_64(prec, Constant::Pi)
        } else {
            Float::new_64(prec)
        };
    }
    if cnum::is_zero(&w) {
        return Ok((cnum::zero(prec), Float::new_64(prec)));
    }
    let size = cnum::abs(&w, prec);
    let mut loss = Float::with_val_64(prec, &scale / &size).ln();
    // The response to a relative nudge of z covers the cancellation in the
    // displacement from the fixed point and the expansion along the orbit.
    let shift = usize::try_from(prec / 2).map_err(|_| "precision overflow")?;
    let nudge = Float::with_val_64(prec, 1) >> shift;
    if !cnum::is_zero(z) {
        let moved = Complex::with_val_64(prec, z * (nudge.clone() + 1u32));
        let (other, _, _, _) = raw(&moved)?;
        let change = principal(Complex::with_val_64(prec, other - &unreduced), prec);
        let response = cnum::abs(&change, prec) / nudge / &size;
        if response.is_finite() && !response.is_zero() {
            loss = loss.max(&response.ln());
        }
    }
    Ok((w / &state.ln_lambda, loss.max(&Float::new_64(prec))))
}

/// Solve `psi(t) = displacement` inside the Poincare disk.
fn invert_poincare(
    state: &schroder::SchroderState,
    radius: &Float,
    displacement: &Complex,
) -> Option<Complex> {
    let prec = state.prec;
    let mut t = displacement.clone();
    let mut best: Option<(Float, Complex)> = None;
    loop {
        let (value, slope) =
            schroder::eval_poincare_series(&state.sigma_inv, &t, radius, prec).ok()?;
        let residual = Complex::with_val_64(prec, value - displacement);
        let error = cnum::abs(&residual, prec);
        if let Some((previous, argument)) = &best {
            if error >= *previous {
                t = argument.clone();
                break;
            }
        }
        if error.is_zero() {
            return Some(t);
        }
        if cnum::is_zero(&slope) {
            return None;
        }
        let next = Complex::with_val_64(prec, &t - residual / slope);
        best = Some((error, std::mem::replace(&mut t, next)));
    }
    let (error, _) = best?;
    let shift = usize::try_from(prec.saturating_sub(24)).ok()?;
    (error <= cnum::abs(displacement, prec) >> shift).then_some(t)
}

fn slog_regular(b: &Complex, z: &Complex, prec: u64, digits: u64) -> Result<Complex, String> {
    let checked = digits
        .checked_add(CHECK_DIGITS)
        .ok_or("super-logarithm precision overflow")?;
    let mut work = prec.max(cnum::checked_digits_to_bits(checked)?);
    loop {
        let base = Complex::with_val_64(work, b);
        let fp = match regions::classify(&base, work)? {
            regions::Region::ShellThronInterior(d) | regions::Region::ShellThronBoundary(d) => d,
            _ => return Err("super-logarithm: base is not strictly attracting".into()),
        };
        let state = schroder::setup_schroder(&base, &fp, work)?;
        let (h, loss) = regular_slog(&state, &Complex::with_val_64(work, z))?;
        let lost = decimal_loss(&loss)?;
        let conditioned = checked
            .checked_add(lost)
            .ok_or("super-logarithm conditioning overflow")?;
        let needed = cnum::checked_digits_to_bits(conditioned)?;
        if needed > work {
            if cnum::verbose() {
                eprintln!("slog: cancellation; refining from {work} to {needed} working bits");
            }
            work = needed;
            continue;
        }
        check_inverse(
            &mut |h| schroder::eval_schroder_at_digits(&state, h, conditioned),
            &h,
            &Complex::with_val_64(work, z),
            digits,
            lost,
            work,
        )?;
        return Ok(Complex::with_val_64(
            output_precision(prec, digits, lost)?,
            h,
        ));
    }
}

const CHECK_DIGITS: u64 = 10;

/// The forward image of `x` must reproduce `target` to within the change
/// caused by a few units in the last requested digit of `x`.
///
/// `forward` must be accurate to `digits + lost + CHECK_DIGITS` digits, where
/// `lost` is the number of digits the inverse loses to conditioning.
fn check_inverse(
    forward: &mut dyn FnMut(&Complex) -> Result<Complex, String>,
    x: &Complex,
    target: &Complex,
    digits: u64,
    lost: u64,
    prec: u64,
) -> Result<(), String> {
    let image = forward(x)?;
    let error = cnum::abs(&Complex::with_val_64(prec, &image - target), prec);
    let scale = cnum::abs(target, prec).max(&cnum::abs(&image, prec));
    let floor = digits
        .checked_add(lost)
        .and_then(|sum| sum.checked_add(CHECK_DIGITS - 2))
        .ok_or("inverse conditioning exceeds the supported precision count")?;
    let allowed = scale * cnum::epsilon(floor, prec)
        + nudge_response(forward, x, &image, digits, prec)? * 16u32;
    if !error.is_finite() || error > allowed {
        return Err(format!(
            "inverse verification failed: forward image differs by {:.3e} (allowed {:.3e})",
            DisplayFloat(&error),
            DisplayFloat(&allowed)
        ));
    }
    Ok(())
}

fn step_within(limit: f64) -> impl Fn(&Complex, &Complex) -> bool {
    move |_, step| {
        let size = Float::with_val(64, step.abs_ref());
        size.is_finite() && size <= limit
    }
}

/// A real target of a real base above `e^(1/e)`, reduced by `log_b` / `b^x`
/// steps to `value` in `(0, 1]`: `F(reduced) = value`, `height = reduced + shift`.
struct RealHeight {
    height: Float,
    reduced: Float,
    value: Float,
    /// The height rounds onto a singular integer `<= -2`: the target is so
    /// negative that the distance to it is below the resolution.
    collapsed: bool,
}

/// The real height `h` with `F(h) = x`, solved on the reduced equation to the
/// accuracy of the state.
fn kneser_real_height(state: &kneser::KneserState, x: &Float) -> Result<RealHeight, String> {
    let prec = state.prec;
    let ln_b = Float::with_val_64(prec, state.base().ln_ref());
    let mut value = Float::with_val_64(prec, x);
    let mut shift = Integer::new();
    while value > 1 {
        value = value.ln() / &ln_b;
        shift += 1;
    }
    while value <= 0 {
        value = (value * &ln_b).exp();
        shift -= 1;
    }
    let finish = |reduced: Float, value: Float| {
        let height = Float::with_val_64(prec, &reduced + &shift);
        let collapsed = height.is_integer() && height <= -2;
        RealHeight {
            height,
            reduced,
            value,
            collapsed,
        }
    };
    if value.is_zero() {
        return Ok(finish(Float::with_val_64(prec, -1), value));
    }
    let bits = usize::try_from(prec / 2).map_err(|_| "precision overflow")?;
    if value < Float::with_val_64(prec, 1) >> bits {
        // F(-1 + d) = F'(-1) d (1 + O(d)).
        let (_, slope, _) = state.evaluate(&Complex::with_val_64(prec, (-1, 0)))?;
        let reduced = Float::with_val_64(prec, &value / slope.real()) - 1u32;
        return Ok(finish(reduced, value));
    }
    let (mut low, mut high) = (Float::with_val_64(prec, -1), Float::new_64(prec));
    let mut guess = Float::with_val_64(prec, &value - 1u32);
    let mut best: Option<(Float, Float)> = None;
    loop {
        let (image, slope, _) = state.evaluate(&Complex::with_val_64(prec, (&guess, 0)))?;
        let defect = Float::with_val_64(prec, image.real() - &value);
        let size = defect.clone().abs();
        if best.as_ref().is_none_or(|(least, _)| size < *least) {
            best = Some((size.clone(), guess.clone()));
        }
        if size.is_zero() {
            break;
        }
        if defect > 0 {
            high = guess.clone();
        } else {
            low = guess.clone();
        }
        let mut next = Float::with_val_64(prec, &guess - defect / slope.real());
        if !(next > low && next < high) {
            next = Float::with_val_64(prec, &low + &high) / 2u32;
        }
        if next == guess || !(next > low && next < high) {
            break;
        }
        guess = next;
    }
    let (_, reduced) = best.expect("at least one iterate");
    Ok(finish(reduced, value))
}

fn slog_kneser(b: &Complex, z: &Complex, prec: u64, digits: u64) -> Result<Complex, String> {
    let checked = digits
        .checked_add(CHECK_DIGITS)
        .ok_or("super-logarithm precision overflow")?;
    let mut state =
        kneser::setup_kneser(b, prec.max(cnum::checked_digits_to_bits(checked)?), checked)?;
    loop {
        let work = state.prec;
        let target = Complex::with_val_64(work, z);
        let refine = |state: &kneser::KneserState, lost: u64| -> Result<Option<u64>, String> {
            let needed = checked
                .checked_add(lost)
                .ok_or("super-logarithm conditioning overflow")?;
            Ok((needed + 3 > state.digits).then_some(needed + 6))
        };
        if target.imag().is_zero() {
            // A real target has a real height: solve and verify the reduced equation.
            let found = kneser_real_height(&state, target.real())?;
            if found.collapsed {
                // Correctly rounded; the forward value at the singular integer does not exist.
                return Ok(Complex::with_val_64(prec, (found.height, 0)));
            }
            let point = Complex::with_val_64(work, (&found.reduced, 0));
            let (image, slope, _) = state.evaluate(&point)?;
            let response = Float::with_val_64(work, slope.real().abs_ref())
                * Float::with_val_64(work, found.height.abs_ref());
            if response.is_zero() || !response.is_finite() {
                return Err("super-logarithm: degenerate real height".into());
            }
            let loss = (-response.clone().ln()).max(&Float::new_64(work));
            let lost = decimal_loss(&loss)?;
            if let Some(target_digits) = refine(&state, lost)? {
                if cnum::verbose() {
                    eprintln!("slog: conditioning; refining to {target_digits} internal digits");
                }
                state = state.refine(target_digits)?;
                continue;
            }
            let error = Float::with_val_64(work, image.real() - &found.value).abs();
            let floor = checked
                .checked_add(lost)
                .ok_or("super-logarithm conditioning overflow")?
                - 2;
            let allowed = response * cnum::epsilon(digits, work) * 16u32
                + Float::with_val_64(work, found.value.abs_ref()) * cnum::epsilon(floor, work);
            if !error.is_finite() || error > allowed {
                return Err(format!(
                    "inverse verification failed: reduced image differs by {:.3e} (allowed {:.3e})",
                    DisplayFloat(&error),
                    DisplayFloat(&allowed)
                ));
            }
            return Ok(Complex::with_val_64(
                output_precision(prec, digits, lost)?,
                (found.height, 0),
            ));
        }
        // Continue from the real axis, to the right of the fixed points when the
        // target lies beyond their height, so that the cuts are not crossed.
        // Inside the strip any real anchor serves; -1 keeps it off the singular heights.
        let fixed = state.fixed_point();
        let beyond = Float::with_val_64(work, target.imag().abs_ref()) >= *fixed.imag();
        let anchor = if beyond && target.real() <= fixed.real() {
            Float::with_val_64(work, fixed.real() + 1u32)
        } else {
            Float::with_val_64(work, target.real()).max(&Float::with_val_64(work, -1))
        };
        let found = kneser_real_height(&state, &anchor)?;
        let start = Complex::with_val_64(work, (found.height, 0));
        let points = [
            Complex::with_val_64(work, (&anchor, 0)),
            Complex::with_val_64(work, (&anchor, target.imag())),
            target.clone(),
        ];
        let (h, _) = follow(
            &mut |h, _| state.evaluate(h).map(|(value, slope, _)| (value, slope)),
            &|t| polyline(&points, t, work),
            start,
            2,
            &step_within(0.5),
            work,
        )?;
        let (value, slope, amplification) = state.evaluate(&h)?;
        // Relative error of h: the image error divided by |F'(h) h|.
        let loss = amplification + cnum::log_magnitude(&value, work)
            - cnum::log_magnitude(&slope, work)
            - cnum::log_magnitude(&h, work);
        let lost = decimal_loss(&loss.max(&Float::new_64(work)))?;
        if let Some(target_digits) = refine(&state, lost)? {
            if cnum::verbose() {
                eprintln!("slog: conditioning; refining to {target_digits} internal digits");
            }
            state = state.refine(target_digits)?;
            continue;
        }
        let conditioned = checked
            .checked_add(lost)
            .ok_or("super-logarithm conditioning overflow")?;
        check_inverse(
            &mut |h| kneser::eval_kneser(&state, h, work, conditioned),
            &h,
            &target,
            digits,
            lost,
            work,
        )?;
        return Ok(Complex::with_val_64(
            output_precision(prec, digits, lost)?,
            h,
        ));
    }
}

/// The change of the forward image when `x` moves by one unit of its last
/// requested digit, the larger of the two directions: next to a fold of the
/// forward map one direction can lead to the twin root and show no change.
fn nudge_response(
    forward: &mut dyn FnMut(&Complex) -> Result<Complex, String>,
    x: &Complex,
    image: &Complex,
    digits: u64,
    prec: u64,
) -> Result<Float, String> {
    let mut response = Float::new_64(prec);
    if cnum::is_zero(x) {
        return Ok(response);
    }
    let unit = cnum::epsilon(digits, prec);
    for factor in [
        Float::with_val_64(prec, 1u32 + &unit),
        Float::with_val_64(prec, 1u32 - &unit),
    ] {
        let moved = forward(&Complex::with_val_64(prec, x * factor))?;
        response = response.max(&cnum::abs(&Complex::with_val_64(prec, moved - image), prec));
    }
    Ok(response)
}

/// Digits lost to conditioning, from the response of the forward map to a
/// nudge of the answer in its last requested digit. `None` when the response
/// is below the resolution of the forward values.
fn nudge_loss(
    forward: &mut dyn FnMut(&Complex) -> Result<Complex, String>,
    x: &Complex,
    target: &Complex,
    digits: u64,
    prec: u64,
) -> Result<Option<u64>, String> {
    let image = forward(x)?;
    let change = nudge_response(forward, x, &image, digits, prec)?;
    if change.is_zero() {
        return Ok(None);
    }
    let scale = cnum::abs(&image, prec).max(&cnum::abs(target, prec));
    let loss = (cnum::epsilon(digits, prec) * scale / change).ln();
    decimal_loss(&loss.max(&Float::new_64(prec))).map(Some)
}

/// Complex bases whose two primary fixed points repel: continue `F_b(0) = 1`
/// along the segment `1 -> z` on the cached theta-mapping state. `None` when
/// the base is outside the scope of that solver.
fn slog_complex(
    b: &Complex,
    z: &Complex,
    prec: u64,
    digits: u64,
) -> Result<Option<Complex>, String> {
    let checked = digits
        .checked_add(CHECK_DIGITS)
        .ok_or("super-logarithm precision overflow")?;
    let start = prec.max(cnum::checked_digits_to_bits(checked)?);
    let mut state = match kneser_complex::setup_complex_kneser(b, start, checked) {
        Ok(state) => state,
        Err(reason) => {
            if cnum::verbose() {
                eprintln!("slog: theta-mapping state unavailable ({reason})");
            }
            return Ok(None);
        }
    };
    loop {
        let work = state.prec;
        let target = Complex::with_val_64(work, z);
        let points = [cnum::one(work), target.clone()];
        let (h, _) = follow(
            &mut |h, _| state.evaluate(h).map(|(value, slope, _)| (value, slope)),
            &|t| polyline(&points, t, work),
            cnum::zero(work),
            1,
            &step_within(0.5),
            work,
        )?;
        let (value, slope, amplification) = state.evaluate(&h)?;
        let loss = amplification + cnum::log_magnitude(&value, work)
            - cnum::log_magnitude(&slope, work)
            - cnum::log_magnitude(&h, work);
        let lost = decimal_loss(&loss.max(&Float::new_64(work)))?;
        let conditioned = checked
            .checked_add(lost)
            .ok_or("super-logarithm conditioning overflow")?;
        if conditioned + 3 > state.digits {
            if cnum::verbose() {
                eprintln!(
                    "slog: conditioning; refining to {} internal digits",
                    conditioned + 6
                );
            }
            state = state.refine(conditioned + 6)?;
            continue;
        }
        check_inverse(
            &mut |h| kneser_complex::eval_complex_kneser(&state, h, work, conditioned),
            &h,
            &target,
            digits,
            lost,
            work,
        )?;
        return Ok(Some(Complex::with_val_64(
            output_precision(prec, digits, lost)?,
            h,
        )));
    }
}

/// Other bases: continue `F_b(0) = 1` along the segment `1 -> z`.
fn slog_generic(b: &Complex, z: &Complex, prec: u64, digits: u64) -> Result<Complex, String> {
    let mut extra = 0u64;
    loop {
        let checked = digits
            .checked_add(CHECK_DIGITS)
            .and_then(|sum| sum.checked_add(extra))
            .ok_or("super-logarithm precision overflow")?;
        let work = prec.max(cnum::checked_digits_to_bits(checked)?);
        let base = Complex::with_val_64(work, b);
        let target = Complex::with_val_64(work, z);
        let delta = cnum::epsilon(checked / 2, work);
        let mut forward = |h: &Complex| -> Result<Complex, String> {
            dispatch::tetrate(&base, h, work, checked)
                .map(|value| Complex::with_val_64(work, value))
        };
        let points = [cnum::one(work), target.clone()];
        let (h, _) = follow(
            &mut |h, _| {
                let value = forward(h)?;
                let ahead = forward(&Complex::with_val_64(work, h + &delta))?;
                let behind = forward(&Complex::with_val_64(work, h - &delta))?;
                let slope = Complex::with_val_64(work, ahead - behind) / &delta / 2u32;
                Ok((value, slope))
            },
            &|t| polyline(&points, t, work),
            cnum::zero(work),
            1,
            &step_within(0.5),
            work,
        )?;
        let lost = match nudge_loss(&mut forward, &h, &target, digits, work)? {
            Some(lost) => lost,
            None => extra.saturating_add(digits),
        };
        if lost > extra {
            if cnum::verbose() {
                eprintln!("slog: conditioning; {lost} more digits");
            }
            extra = lost;
            continue;
        }
        check_inverse(&mut forward, &h, &target, digits, extra, work)?;
        return Ok(Complex::with_val_64(
            output_precision(prec, digits, lost)?,
            h,
        ));
    }
}

/// Super-logarithm: the height `h` with `F_b(h) = z` (see the module notes for
/// the branch conventions).
pub fn slog(b: &Complex, z: &Complex, prec: u64, digits: u64) -> Result<Complex, String> {
    crate::mt::init_pool()?;
    cnum::require_precision(prec, digits)?;
    if !cnum::is_finite(b) || !cnum::is_finite(z) {
        return Err("super-logarithm base and argument must be finite".into());
    }
    if cnum::is_zero(b) || cnum::is_one(b) {
        return Err("super-logarithm is undefined for the bases 0 and 1".into());
    }
    if !b.imag().is_zero() && b.imag().is_sign_negative() {
        let mirrored = slog(
            &Complex::with_val_64(prec, b.conj_ref()),
            &Complex::with_val_64(prec, z.conj_ref()),
            prec,
            digits,
        )?;
        return Ok(mirrored.conj());
    }
    if cnum::is_one(z) {
        return Ok(cnum::zero(prec));
    }
    if cnum::is_zero(z) {
        return Ok(Complex::with_val_64(prec, (-1, 0)));
    }
    if z == b {
        return Ok(cnum::one(prec));
    }
    let attracting = matches!(
        regions::classify(b, prec)?,
        regions::Region::ShellThronInterior(d) | regions::Region::ShellThronBoundary(d)
            if d.lambda_abs < 1
    );
    if attracting {
        slog_regular(b, z, prec, digits)
    } else if b.imag().is_zero() && *b.real() > cnum::eta_upper(prec) {
        slog_kneser(b, z, prec, digits)
    } else if let Some(h) = slog_complex(b, z, prec, digits)? {
        Ok(h)
    } else {
        slog_generic(b, z, prec, digits)
    }
}

fn real_complex(x: &Float, prec: u64) -> Complex {
    Complex::with_val_64(prec, (x, 0))
}

/// `ln F_b(h) - ln z` for real `b > 1`, `h > 0`, `z > 1`, with `b = exp(rate)`.
fn real_defect(rate: &Float, h: &Float, z: &Float, digits: u64) -> Result<Float, String> {
    let prec = cnum::checked_digits_to_bits(digits)?.max(rate.prec_64());
    let base = Float::with_val_64(prec, rate.exp_ref());
    let value = match dispatch::tetrate(
        &real_complex(&base, prec),
        &real_complex(h, prec),
        prec,
        digits,
    ) {
        Ok(value) => value,
        // A tower beyond MPFR's range is above every finite target.
        Err(error) if error.contains("overflow") => {
            return Ok(Float::with_val_64(prec, rug::float::Special::Infinity));
        }
        Err(error) => return Err(error),
    };
    if !value.imag().is_zero() || *value.real() <= 0 {
        return Err("super-root: the forward value left the positive real axis".into());
    }
    Ok(Float::with_val_64(prec, value.real().ln_ref()) - Float::with_val_64(prec, z.ln_ref()))
}

/// The real base `b > 1` with `F_b(h) = z` for real `z > 1`, `h > 0`: forward
/// values to `coarse` digits while bracketing, to `checked` digits at the end,
/// at `work` bits (enough to hold `z` and `h` exactly).
fn sroot_real(z: &Float, h: &Float, work: u64, checked: u64, coarse: u64) -> Result<Float, String> {
    let z = Float::with_val_64(work, z);
    let h = Float::with_val_64(work, h);
    let report = |stage: &str, rate: &Float| {
        if cnum::verbose() {
            eprintln!("sroot: {stage}, ln b = {:.12}", DisplayFloat(rate));
        }
    };
    // F_z(h) <= z for h <= 1 and >= z for h >= 1: ln z is one end of the bracket.
    // For h >= 2, F_b(h) >= b^b, so b = e max(ln z, 1) already lies above the root.
    let mut low = Float::with_val_64(work, z.ln_ref());
    if h >= 2 {
        low = low.max(&Float::with_val_64(work, 1)).ln() + 1u32;
    }
    let mut high = low.clone();
    let mut low_defect = real_defect(&low, &h, &z, coarse)?;
    let mut high_defect = low_defect.clone();
    if low_defect.is_zero() {
        return Ok(low.exp());
    }
    if low_defect < 0 {
        loop {
            high *= 2u32;
            report("bracketing upward", &high);
            high_defect = real_defect(&high, &h, &z, coarse)?;
            if high_defect >= 0 {
                break;
            }
            low = high.clone();
            low_defect = high_defect.clone();
        }
    } else {
        loop {
            low /= 2u32;
            report("bracketing downward", &low);
            low_defect = real_defect(&low, &h, &z, coarse)?;
            if low_defect <= 0 {
                break;
            }
            high = low.clone();
            high_defect = low_defect.clone();
        }
    }
    // Illinois regula falsi on the bracket.
    let narrow = cnum::epsilon(coarse.saturating_sub(8).max(4), work);
    let mut side = 0i8;
    let mut rate = high.clone();
    while Float::with_val_64(work, &high - &low) > Float::with_val_64(work, &high * &narrow) {
        let width = Float::with_val_64(work, &high - &low);
        rate = if high_defect.is_finite() {
            Float::with_val_64(work, &low * &high_defect - &high * &low_defect)
                / Float::with_val_64(work, &high_defect - &low_defect)
        } else {
            Float::with_val_64(work, &low + &high) / 2u32
        };
        if !(rate > low && rate < high) {
            rate = Float::with_val_64(work, &low + &high) / 2u32;
        }
        report("bracketed", &rate);
        let defect = real_defect(&rate, &h, &z, coarse)?;
        if defect.is_zero() {
            low = rate.clone();
            high = rate.clone();
            break;
        }
        if defect > 0 {
            high = rate.clone();
            high_defect = defect;
            if side == 1 {
                low_defect /= 2u32;
            }
            side = 1;
        } else {
            low = rate.clone();
            low_defect = defect;
            if side == -1 {
                high_defect /= 2u32;
            }
            side = -1;
        }
        if Float::with_val_64(work, &high - &low) >= width {
            break;
        }
    }
    if !high_defect.is_finite() {
        return Err("super-root: no finite forward value above the root".into());
    }
    // Secant refinement at full precision.
    let mut previous = if rate == low { high } else { low };
    let mut previous_defect = real_defect(&previous, &h, &z, checked)?;
    let mut defect = real_defect(&rate, &h, &z, checked)?;
    let floor = cnum::epsilon(checked, work);
    loop {
        if defect.is_zero() || previous_defect == defect {
            break;
        }
        let step = Float::with_val_64(work, &rate - &previous) * &defect
            / Float::with_val_64(work, &defect - &previous_defect);
        let next = Float::with_val_64(work, &rate - &step);
        if !(next.is_finite() && next > 0) {
            return Err("super-root: secant iteration left the domain b > 1".into());
        }
        report("refining", &next);
        let small = step.clone().abs() <= Float::with_val_64(work, &next * &floor);
        let next_defect = real_defect(&next, &h, &z, checked)?;
        if next_defect.clone().abs() >= defect.clone().abs() {
            break;
        }
        previous = std::mem::replace(&mut rate, next);
        previous_defect = std::mem::replace(&mut defect, next_defect);
        if small {
            break;
        }
    }
    Ok(rate.exp())
}

/// Super-root: the base `b` with `F_b(h) = z`.
pub fn sroot(z: &Complex, h: &Complex, prec: u64, digits: u64) -> Result<Complex, String> {
    crate::mt::init_pool()?;
    cnum::require_precision(prec, digits)?;
    if !cnum::is_finite(z) || !cnum::is_finite(h) {
        return Err("super-root argument and height must be finite".into());
    }
    if h.imag().is_zero() && h.real().is_integer() {
        if cnum::is_one(h) {
            return Ok(Complex::with_val_64(prec, z));
        }
        if *h.real() < 1 {
            return Err(
                "super-root: b^^h does not determine the base at integer heights below 1".into(),
            );
        }
    }
    if cnum::is_one(z) {
        return Ok(cnum::one(prec));
    }
    let real_case = z.imag().is_zero() && h.imag().is_zero() && *z.real() > 1 && *h.real() > 0;
    // A target close to 1 must be resolved against its distance from 1.
    let mut extra = 0u64;
    if real_case {
        let nearness = Float::with_val_64(prec, z.real().ln_ref());
        if nearness < 1 {
            extra = decimal_loss(&(-nearness.ln()))?;
        }
    }
    loop {
        let checked = digits
            .checked_add(CHECK_DIGITS)
            .and_then(|sum| sum.checked_add(extra))
            .ok_or("super-root precision overflow")?;
        let work = prec.max(cnum::checked_digits_to_bits(checked)?);
        let coarse = extra.saturating_add(24).min(checked);
        let target = Complex::with_val_64(work, z);
        let height = Complex::with_val_64(work, h);
        let forward = |b: &Complex, height: &Complex| -> Result<Complex, String> {
            dispatch::tetrate(b, height, work, checked)
                .map(|value| Complex::with_val_64(work, value))
        };
        let root = if real_case {
            real_complex(
                &sroot_real(target.real(), height.real(), work, checked, coarse)?,
                work,
            )
        } else {
            // Continue a real solution: first move z, then h.
            let size = cnum::abs(&target, work);
            let anchor_z = if size > 1.0001 {
                size
            } else {
                Float::with_val_64(work, 2)
            };
            let anchor_h = if *height.real() >= 0.25 {
                Float::with_val_64(work, height.real())
            } else {
                Float::with_val_64(work, 1)
            };
            let start = if anchor_h == 1 {
                anchor_z.clone()
            } else {
                sroot_real(&anchor_z, &anchor_h, work, checked, coarse)?
            };
            let from = real_complex(&anchor_z, work);
            let level = real_complex(&anchor_h, work);
            let delta = cnum::epsilon(checked / 2, work);
            let walk = |targets: &[Complex], heights: &[Complex]| {
                follow(
                    &mut |b, t| {
                        let height = polyline(heights, t, work);
                        let value = forward(b, &height)?;
                        let nudge = Complex::with_val_64(work, b * &delta);
                        let ahead = forward(&Complex::with_val_64(work, b + &nudge), &height)?;
                        let behind = forward(&Complex::with_val_64(work, b - &nudge), &height)?;
                        Ok((
                            value,
                            Complex::with_val_64(work, ahead - behind) / nudge / 2u32,
                        ))
                    },
                    &|t| polyline(targets, t, work),
                    real_complex(&start, work),
                    u32::try_from(targets.len() - 1).expect("short path"),
                    &|b, step| {
                        Float::with_val(64, step.abs_ref()) * 4u32
                            <= Float::with_val(64, b.abs_ref())
                    },
                    work,
                )
            };
            let direct = walk(
                &[from.clone(), target.clone(), target.clone()],
                &[level.clone(), level.clone(), height.clone()],
            );
            match direct {
                Ok((root, _)) => root,
                Err(reason) => {
                    // A singular value on the straight path (a fold of b -> F_b(h)
                    // on the real axis): pass above it.
                    let span = Complex::with_val_64(work, &target - &from);
                    let lift = Complex::with_val_64(work, (0, cnum::abs(&span, work) / 2u32));
                    let above = Complex::with_val_64(work, &from + span / 2u32) + lift;
                    if cnum::verbose() {
                        eprintln!("sroot: straight path failed ({reason}); passing above it");
                    }
                    let (root, _) = walk(
                        &[from.clone(), above, target.clone(), target.clone()],
                        &[level.clone(), level.clone(), level.clone(), height.clone()],
                    )
                    .map_err(|detour| format!("{reason}; path above the real axis: {detour}"))?;
                    root
                }
            }
        };
        let lost = match nudge_loss(&mut |b| forward(b, &height), &root, &target, digits, work)? {
            Some(lost) => lost,
            None => extra.saturating_add(digits),
        };
        if lost > extra {
            if cnum::verbose() {
                eprintln!("sroot: conditioning; {lost} more digits");
            }
            extra = lost;
            continue;
        }
        check_inverse(
            &mut |b| forward(b, &height),
            &root,
            &target,
            digits,
            extra,
            work,
        )?;
        return Ok(Complex::with_val_64(
            output_precision(prec, digits, lost)?,
            root,
        ));
    }
}
