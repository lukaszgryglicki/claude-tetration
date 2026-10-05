//! Algorithm dispatch: classify `(b, h)` and route to the right implementation.
//!
//! The dispatcher does these jobs, in order:
//!   1. Trivial bases (`b = 0`, `b = 1`) and direct integer-height iteration.
//!   2. Compute the base region (`regions::classify`).
//!   3. Route to the matching algorithm.
//!
//! Currently implemented algorithms:
//!   * Integer-height direct iteration (any base, integer height).
//!   * Schröder regular tetration at the principal fixed point — works for
//!     Shell-Thron interior bases (`|λ| < 0.95`) and for some bases just
//!     outside the boundary where σ̃ Taylor still reaches `1−L`.
//!   * Kouznetsov Cauchy iteration, with real-base and experimental cut-base
//!     continuation when direct construction fails.
//!
//! No unchecked polynomial surrogate is returned when these methods fail.

use rug::{Complex, Float};

use crate::{cnum, integer_height, kouznetsov, regions, schroder};

fn debug_enabled() -> bool {
    cnum::verbose()
}

fn dprint(s: &str) {
    if debug_enabled() {
        eprintln!("tet: {}", s);
    }
}

/// Regular iteration at a complex fixed point is not the real-base Kneser
/// construction, even at a complex height. These are necessary branch checks,
/// not a general canonicality certificate.
fn schroder_result_is_canonical(b: &Complex, h: &Complex, v: &Complex, prec: u64) -> bool {
    if !cnum::is_finite(v) {
        return false;
    }
    if !b.imag().is_zero() || *b.real() <= 1 {
        return true;
    }
    if *b.real() > cnum::eta_upper(prec) {
        return false;
    }
    if !h.imag().is_zero() || *h.real() <= -2 {
        return true;
    }
    let im_abs = Float::with_val_64(prec, v.imag().abs_ref());
    let scale = cnum::abs(v, prec).max(&Float::with_val_64(prec, 1));
    im_abs < cnum::working_epsilon(prec) * scale
}

/// Real-positive base strictly below e^{-e} (the "cut" segment).
fn is_cut_base(b: &Complex) -> bool {
    b.imag().is_zero() && *b.real() > 0 && *b.real() < cnum::eta_lower(b.real().prec_64())
}

/// Experimental upper-half-base-plane continuation for `0 < b < e^{-e}`.
/// A reached endpoint must pass the requested residual/normalization gates;
/// those checks alone establish neither existence of the limit nor uniqueness.
/// Do not force the selected complex branch to be real on real heights.
fn tetrate_cut_base(
    b: &Complex,
    h: &Complex,
    prec: u64,
    digits: u64,
    schroder_err: &str,
) -> Result<Complex, String> {
    dprint(&format!(
        "Schröder failed for cut base ({}); ε-continuation Kouznetsov",
        schroder_err
    ));
    dprint("cut base: attempting experimental epsilon-continuation from b+2i; this may take hours");
    (|| -> Result<Complex, String> {
        let b_re = Float::with_val_64(prec, b.real());
        let st = kouznetsov::setup_kouznetsov_cut_base(&b_re, prec, digits)?;
        let b_exact = Complex::with_val_64(prec, (b_re, Float::new_64(prec)));
        kouznetsov::eval_kouznetsov(&st, &b_exact, h)
    })()
    .map_err(|ke| {
        unsupported_msg(
            "real base on the cut 0 < b < e^{-e}",
            &format!(
                "Schröder: {}; \
                 ε-continuation Kouznetsov: {}",
                schroder_err, ke
            ),
        )
    })
}

/// Compute `F_b(h)` at the given precision (in MPC bits). `digits` is the
/// requested decimal precision used for accuracy gates and resource checks.
pub fn tetrate(b: &Complex, h: &Complex, prec: u64, digits: u64) -> Result<Complex, String> {
    crate::mt::init_pool()?;
    cnum::require_precision(prec, digits)?;
    if !cnum::is_finite(b) || !cnum::is_finite(h) {
        return Err("tetration base and height must be finite".into());
    }
    if digits == 0 {
        return Err("precision must be positive".into());
    }
    let result = tetrate_impl(b, h, prec, digits)?;
    if !cnum::is_finite(&result) {
        return Err("tetration produced a non-finite result".into());
    }
    Ok(result)
}

fn tetrate_impl(b: &Complex, h: &Complex, prec: u64, digits: u64) -> Result<Complex, String> {
    // ---- Special-case bases (don't need fixed-point computation) ----
    if cnum::is_one(b) {
        dprint("special case b=1 → 1");
        return Ok(cnum::one(prec));
    }
    if cnum::is_zero(b) {
        return tetrate_base_zero(h, prec, digits);
    }

    // ---- Integer heights: direct iteration regardless of region ----
    if h.imag().is_zero() && h.real().is_integer() {
        dprint(&format!(
            "integer height n={}",
            cnum::DisplayFloat(h.real())
        ));
        return integer_height::tetrate_integer_height(b, h.real(), prec, digits);
    }

    // ---- Schwarz reflection for Im(b) < 0 ----
    // Select the conjugate family: F_b(h) = conj(F_{b̄}(h̄)).
    // The Kouznetsov initial-guess shape (target_mid = √b sits in the upper
    // half-plane) and the Newton normalization-shift basin are tuned for the
    // Im(b) ≥ 0 orientation; flipping b into the lower half-plane mirrors the
    // strip and drives the shift Newton to spurious large-|c| roots far from
    // the origin (e.g. b=-3.6-0.4i landed at δ=1.5−9.5i instead of the small
    // δ=-0.5+0.25i that conjugacy demands). Reduce to the canonical
    // orientation by conjugating both inputs and the result.
    if !b.imag().is_zero() && b.imag().is_sign_negative() {
        dprint("Schwarz reflection: Im(b)<0, dispatching as conj(F_{b̄}(h̄))");
        let b_conj = Complex::with_val_64(prec, b.conj_ref());
        let h_conj = Complex::with_val_64(prec, h.conj_ref());
        let result = tetrate_impl(&b_conj, &h_conj, prec, digits)?;
        return Ok(Complex::with_val_64(prec, result.conj_ref()));
    }

    // ---- Region classification (drives algorithm choice) ----
    let region = regions::classify(b, prec)?;
    dprint(&format!("region = {}", region.name()));
    if debug_enabled() {
        if let Some(la) = lambda_abs_of(&region) {
            eprintln!("tet: |λ| ≈ {}", cnum::DisplayFloat(la));
        }
    }

    // ---- Routing ----
    // No silent linear-approximation fallback (per design): if the chosen
    // algorithm fails, propagate the Err so the caller sees an honest failure
    // rather than a wrong-but-plausible C^0 number.
    let regular_base = !b.imag().is_zero() || *b.real() <= cnum::eta_upper(prec);
    if !regular_base {
        dprint("real base above eta: regular fixed-point family is not Kneser; using Kouznetsov");
    }
    match &region {
        regions::Region::BaseOne | regions::Region::BaseZero => {
            // Already handled above.
            unreachable!()
        }
        regions::Region::ShellThronInterior(d) => {
            // Schröder is the primary method for |λ| < 0.95. For bases very
            // close to η (typically b ≳ 1.437 for real bases), the σ̃⁻¹ series
            // can diverge at |s1| even though |s1| < safe_radius (the heuristic
            // safe_radius estimate is too large near the parabolic boundary). In
            // those cases the anchor check in setup_schroder catches the failure.
            // Try the existing Kouznetsov construction if regular iteration fails.
            match schroder::tetrate_schroder_at_digits(b, h, d, prec, digits) {
                Ok(v) => return Ok(v),
                Err(e) => dprint(&format!(
                    "Schröder failed in Shell-Thron interior ({}); falling back to Kouznetsov",
                    e
                )),
            }
            kouznetsov::tetrate_kouznetsov(b, h, d, prec, digits).map_err(|kouz_err| {
                format!(
                    "Schröder failed and Kouznetsov fallback also failed: {}",
                    kouz_err
                )
            })
        }
        regions::Region::ShellThronBoundary(d) => {
            // Near |λ|=1, regular iteration can converge very slowly
            // (not every boundary point is parabolic).
            //
            // For real positive bases (b > η, |λ| just above 1) on this band,
            // |arg(λ)| is small so direct Kouznetsov has trouble converging on
            // a feasible-sized grid (see kouznetsov.rs for the parabolic-cap
            // logic). The continuation solver — which warm-starts from a base
            // farther from the boundary and walks back — is much more reliable
            // there, so we try it FIRST and fall through to direct
            // Kouznetsov if continuation fails. Neither path may substitute
            // an unchecked polynomial extrapolation.
            //
            // For complex bases on the boundary, |arg(λ)| can already be large,
            // so direct Kouznetsov often works; we keep the original order.
            if regular_base {
                match schroder::tetrate_schroder_at_digits(b, h, d, prec, digits) {
                    Ok(v) if schroder_result_is_canonical(b, h, &v, prec) => {
                        dprint("Schröder succeeded at boundary band");
                        return Ok(v);
                    }
                    Ok(_) => dprint(
                        "Schröder result fails the real-base branch requirements; trying Kouznetsov",
                    ),
                    Err(e) => dprint(&format!("Schröder failed at boundary band: {}", e)),
                }
            }
            let is_real_base = b.imag().is_zero() && !b.real().is_sign_negative();
            if is_real_base {
                // Preserve the complex cut-branch convention after regular
                // iteration fails, rather than forcing a real result.
                if is_cut_base(b) {
                    return tetrate_cut_base(b, h, prec, digits, "too slow on the parabolic band");
                }
                dprint("real-base boundary band: trying continuation solver");
                let cont_err = match try_continuation(b, h, d, prec, digits) {
                    Ok(v) => return Ok(v),
                    Err(error) => error,
                };
                dprint(&format!(
                    "continuation failed ({cont_err}); trying direct Kouznetsov"
                ));
                return kouznetsov::tetrate_kouznetsov(b, h, d, prec, digits).map_err(
                    |direct_err| {
                        unsupported_msg(
                            "Shell-Thron boundary band (|λ| ≈ 1)",
                            &format!(
                                "Schröder unavailable; continuation: {}; direct Kouznetsov: {}; \
                             unchecked polynomial extrapolation is not a tetration result",
                                cont_err, direct_err
                            ),
                        )
                    },
                );
            }
            dprint("complex-base boundary band: trying Newton-Kouznetsov");
            match kouznetsov::tetrate_kouznetsov(b, h, d, prec, digits) {
                Ok(v) => Ok(v),
                Err(why) => {
                    if why.contains("parabolic boundary")
                        || why.contains("degenerate contour")
                        || why.contains("addressable memory")
                    {
                        dprint(
                            "direct Kouznetsov hit a contour/precision limit; trying continuation",
                        );
                        match try_continuation(b, h, d, prec, digits) {
                            Ok(v) => return Ok(v),
                            Err(error) => dprint(&format!("continuation unavailable: {error}")),
                        }
                    }
                    Err(unsupported_msg(
                        "Shell-Thron boundary band (|λ| ≈ 1)",
                        &format!(
                            "Schröder regular tetration converges too slowly here; \
                             Newton-Kantorovich Kouznetsov failed: {}",
                            why
                        ),
                    ))
                }
            }
        }
        regions::Region::OutsideShellThronRealPositive(d) => {
            if regular_base {
                match schroder::tetrate_schroder_at_digits(b, h, d, prec, digits) {
                    Ok(v) if schroder_result_is_canonical(b, h, &v, prec) => {
                        dprint("Schröder succeeded at repelling fixed point");
                        return Ok(v);
                    }
                    Ok(_) => dprint(
                        "Schröder result fails the real-base branch requirements; trying Kouznetsov",
                    ),
                    Err(e) => {
                        if is_cut_base(b) {
                            return tetrate_cut_base(b, h, prec, digits, &e);
                        }
                        dprint(&format!(
                            "Schröder unavailable ({}); switching to Newton-Kouznetsov",
                            e
                        ));
                    }
                }
            }
            match kouznetsov::tetrate_kouznetsov(b, h, d, prec, digits) {
                Ok(v) => Ok(v),
                Err(why) => {
                    // For real positive bases on or near the parabolic boundary,
                    // try continuation on a relevant
                    // Kouznetsov failure (the classification can put a base in
                    // OutsideShellThronRealPositive but with |λ| only slightly
                    // > 1.05, where Kouznetsov still struggles).
                    let is_real_base = b.imag().is_zero();
                    let near_boundary = d.lambda_abs < cnum::decimal("1.10", prec);
                    let parabolic_signal = why.contains("parabolic boundary")
                        || why.contains("degenerate contour")
                        || why.contains("addressable memory")
                        || (is_real_base
                            && near_boundary
                            && (why.contains("did not converge")
                                || why.contains("residual")
                                || why.contains("n_nodes")));
                    if parabolic_signal {
                        dprint(
                            "Kouznetsov hit a contour/precision limit; trying continuation solver",
                        );
                        let cont_err = match try_continuation(b, h, d, prec, digits) {
                            Ok(v) => return Ok(v),
                            Err(error) => error,
                        };
                        Err(unsupported_msg(
                            "real base > e^(1/e)",
                            &format!(
                                "Schröder not applicable; Kouznetsov direct: {}; \
                                 continuation: {}",
                                why, cont_err
                            ),
                        ))
                    } else {
                        Err(unsupported_msg(
                            "real base > e^(1/e)",
                            &format!(
                                "Schröder regular tetration not applicable, and \
                                 Newton-Kantorovich Kouznetsov Cauchy iteration \
                                 failed: {}",
                                why
                            ),
                        ))
                    }
                }
            }
        }
        regions::Region::OutsideShellThronGeneral(d) => {
            // Try Schröder at the repelling fixed point first (cheap when it
            // works). For slightly-off-real bases, fall through to the
            // generalized Kouznetsov path: `is_real_positive(b)` switches off
            // Schwarz symmetry, and the partner fixed point is found by
            // Newton iteration starting from `conj(L_+)` (the analytic
            // continuation of the real-base conjugate pair). For truly
            // complex bases the two fixed points may fall in the same
            // half-plane, in which case Kouznetsov errors out cleanly —
            // this is a limitation of the implemented contour, not a
            // nonexistence theorem or proof that one particular method is needed.
            let schroder_err = match schroder::tetrate_schroder_at_digits(b, h, d, prec, digits) {
                Ok(v) => {
                    dprint("Schröder succeeded at repelling fixed point");
                    return Ok(v);
                }
                Err(error) => error,
            };
            dprint(&format!(
                "Schröder unavailable ({schroder_err}); trying Newton-Kouznetsov for complex base"
            ));
            match kouznetsov::tetrate_kouznetsov(b, h, d, prec, digits) {
                Ok(v) => Ok(v),
                Err(why) => Err(unsupported_msg(
                    "general complex base outside Shell-Thron",
                    &format!(
                        "Schröder: {}; \
                             Newton-Kantorovich Kouznetsov failed: {}",
                        schroder_err, why
                    ),
                )),
            }
        }
    }
}

/// Single source for the "this case isn't implemented" error message. Caller
/// decides what's printed; we just give a clean one-liner the CLI surfaces.
fn unsupported_msg(case: &str, why: &str) -> String {
    format!("unsupported case: {} — {}", case, why)
}

fn lambda_abs_of(region: &regions::Region) -> Option<&Float> {
    match region {
        regions::Region::ShellThronInterior(d)
        | regions::Region::ShellThronBoundary(d)
        | regions::Region::OutsideShellThronRealPositive(d)
        | regions::Region::OutsideShellThronGeneral(d) => Some(&d.lambda_abs),
        _ => None,
    }
}

/// Attempt the continuation-based Kouznetsov solver for near-parabolic real bases,
/// then evaluate at the requested height.
fn try_continuation(
    b: &Complex,
    h: &Complex,
    fp: &crate::regions::FixedPointData,
    prec: u64,
    digits: u64,
) -> Result<Complex, String> {
    let state = kouznetsov::setup_kouznetsov_continuation(b, fp, prec, digits)?;
    kouznetsov::eval_kouznetsov(&state, b, h)
}

/// Tetration with base 0 — convention:
///   * `0^^0 = 1`
///   * `0^^n` for positive integer `n`: `n` even → 1, `n` odd → 0
///     (because `0^0 = 1` and `0^k = 0` for `k > 0`).
///   * Negative integer / non-integer height: undefined.
fn tetrate_base_zero(h: &Complex, prec: u64, digits: u64) -> Result<Complex, String> {
    if !h.imag().is_zero() || !h.real().is_integer() {
        return Err("tetration of 0 is only defined for non-negative integer heights".into());
    }
    if *h.real() < 0 {
        return Err(format!(
            "tetration of 0 with negative integer height {} is undefined",
            cnum::DisplayFloat(h.real())
        ));
    }
    integer_height::tetrate_integer_height(&cnum::zero(prec), h.real(), prec, digits)
}
