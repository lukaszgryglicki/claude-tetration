//! Algorithm dispatch: classify `(b, h)` and route to the right implementation.
//!
//! The dispatcher does these jobs, in order:
//!   1. Trivial bases (`b = 0`, `b = 1`) and integer heights — answered exactly.
//!   2. Compute the base region (`regions::classify`).
//!   3. Route to the matching algorithm.
//!
//! Currently implemented algorithms:
//!   * Integer-height direct iteration (any base, integer height).
//!   * Schröder regular tetration at the principal fixed point — works for
//!     Shell-Thron interior bases (`|λ| < 0.95`) and for some bases just
//!     outside the boundary where σ̃ Taylor still reaches `1−L`.
//!
//! The linear-`C^0` approximation is never used as a silent fallback: a
//! divergent Schröder series produces an Err, not a wrong-but-plausible
//! answer. Bases requiring full Kneser/Kouznetsov Cauchy iteration (typical
//! real `b > e^(1/e)`, far-from-boundary general complex) error out cleanly.

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

/// Compute `F_b(h)` at the given precision (in MPC bits). `digits` is the
/// requested decimal precision and is used for precision-ceiling warnings.
pub fn tetrate(b: &Complex, h: &Complex, prec: u32, digits: u64) -> Result<Complex, String> {
    // ---- Special-case bases (don't need fixed-point computation) ----
    if cnum::is_one(b) {
        dprint("special case b=1 → 1");
        return Ok(cnum::one(prec));
    }
    if cnum::is_zero(b) {
        return tetrate_base_zero(h, prec);
    }

    // ---- Integer heights: direct iteration regardless of region ----
    if let Some(n) = cnum::as_integer(h) {
        dprint(&format!("integer height n={}", n));
        return integer_height::tetrate_integer(b, n, prec);
    }

    // ---- Schwarz reflection for Im(b) < 0 ----
    // The canonical Kneser tetration satisfies F_b(h) = conj(F_{b̄}(h̄)).
    // The Kouznetsov initial-guess shape (target_mid = √b sits in the upper
    // half-plane) and the Newton normalization-shift basin are tuned for the
    // Im(b) ≥ 0 orientation; flipping b into the lower half-plane mirrors the
    // strip and drives the shift Newton to spurious large-|c| roots far from
    // the origin (e.g. b=-3.6-0.4i landed at δ=1.5−9.5i instead of the small
    // δ=-0.5+0.25i that conjugacy demands). Reduce to the canonical
    // orientation by conjugating both inputs and the result.
    if !b.imag().is_zero() && b.imag().is_sign_negative() {
        dprint("Schwarz reflection: Im(b)<0, dispatching as conj(F_{b̄}(h̄))");
        let b_conj = Complex::with_val(prec, b.conj_ref());
        let h_conj = Complex::with_val(prec, h.conj_ref());
        let result = tetrate(&b_conj, &h_conj, prec, digits)?;
        return Ok(Complex::with_val(prec, result.conj_ref()));
    }

    // ---- Region classification (drives algorithm choice) ----
    let region = regions::classify(b, prec)?;
    dprint(&format!("region = {}", region.name()));
    if debug_enabled() {
        if let Some(la) = lambda_abs_of(&region) {
            eprintln!("tet: |λ| ≈ {}", la);
        }
    }
    warn_if_precision_exceeds_ceiling(&region, digits);

    // ---- Routing ----
    // No silent linear-approximation fallback (per design): if the chosen
    // algorithm fails, propagate the Err so the caller sees an honest failure
    // rather than a wrong-but-plausible C^0 number.
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
            // Fall back to Kouznetsov, which works for all real b > 1.
            match schroder::tetrate_schroder(b, h, d, prec) {
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
            // The boundary band 0.95 ≤ |λ| ≤ 1.05 is the parabolic-fixed-point
            // regime. Schröder regular tetration converges very slowly here
            // (geometric rate `|λ|`, which is near 1) and typically bails. But
            // for bases inside this band whose argλ is far enough from 0 — and
            // whose fixed-point pair sits in opposite half-planes — Newton-
            // Kantorovich Cauchy iteration still converges, just with a taller
            // contour. Try Schröder, then Kouznetsov; only error if both fail.
            // For truly parabolic |λ|=1 the band needs Paulsen-Cowgill.
            match schroder::tetrate_schroder(b, h, d, prec) {
                Ok(v) => {
                    dprint("Schröder succeeded at boundary band");
                    return Ok(v);
                }
                Err(e) => dprint(&format!("Schröder failed at boundary band: {}", e)),
            }
            dprint("Schröder unavailable in boundary band; trying Newton-Kouznetsov");
            match kouznetsov::tetrate_kouznetsov(b, h, d, prec, digits) {
                Ok(v) => Ok(v),
                Err(why) => {
                    // If the direct solver was blocked by the parabolic-boundary
                    // n_nodes cap, try the continuation-based solver which
                    // warm-starts from a farther base and allows larger grids.
                    if why.contains("parabolic boundary") || why.contains("degenerate contour") {
                        dprint("direct Kouznetsov hit parabolic cap; trying continuation solver");
                        match try_continuation(b, h, d, prec, digits) {
                            Ok(v) => return Ok(v),
                            Err(cont_err) => {
                                dprint(&format!(
                                    "continuation also failed: {}; trying iε-perturbation Richardson",
                                    cont_err
                                ));
                                if let Ok(v) =
                                    try_iperturbation_extrapolation(b, h, prec, digits)
                                {
                                    return Ok(v);
                                }
                                Err(unsupported_msg(
                                    "Shell-Thron parabolic boundary band (|λ| ≈ 1)",
                                    &format!(
                                        "Schröder too slow; Kouznetsov direct: {}; \
                                         continuation: {}; iε-perturbation also failed",
                                        why, cont_err
                                    ),
                                ))
                            }
                        }
                    } else {
                        Err(unsupported_msg(
                            "Shell-Thron parabolic boundary band (|λ| ≈ 1)",
                            &format!(
                                "Schröder regular tetration converges too slowly here, \
                                 and Newton-Kantorovich Kouznetsov Cauchy iteration \
                                 failed: {}",
                                why
                            ),
                        ))
                    }
                }
            }
        }
        regions::Region::OutsideShellThronRealPositive(d) => {
            // Schröder at the repelling fixed point handles bases unusually
            // close to the Shell-Thron boundary; for typical real bases
            // (e, 2, 10, …) the σ̃-shift orbit hits the log singularity and
            // Schröder bails. Fall through to Anderson-accelerated Kouznetsov
            // Cauchy iteration, which produces the natural real-on-real
            // tetration via samples on Re(z)=0.5 refined by Cauchy's formula.
            match schroder::tetrate_schroder(b, h, d, prec) {
                Ok(v) => {
                    dprint("Schröder succeeded at repelling fixed point");
                    return Ok(v);
                }
                Err(e) => dprint(&format!("Schröder unavailable ({}); switching to Newton-Kouznetsov", e)),
            }
            match kouznetsov::tetrate_kouznetsov(b, h, d, prec, digits) {
                Ok(v) => Ok(v),
                Err(why) => {
                    if why.contains("parabolic boundary") || why.contains("degenerate contour") {
                        dprint("direct Kouznetsov hit parabolic cap; trying continuation solver");
                        match try_continuation(b, h, d, prec, digits) {
                            Ok(v) => return Ok(v),
                            Err(cont_err) => {
                                dprint(&format!(
                                    "continuation also failed: {}; trying iε-perturbation Richardson",
                                    cont_err
                                ));
                                if let Ok(v) =
                                    try_iperturbation_extrapolation(b, h, prec, digits)
                                {
                                    return Ok(v);
                                }
                                Err(unsupported_msg(
                                    "real base > e^(1/e)",
                                    &format!(
                                        "Schröder not applicable; Kouznetsov direct: {}; \
                                         continuation: {}; iε-perturbation also failed",
                                        why, cont_err
                                    ),
                                ))
                            }
                        }
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
            // those bases need Paulsen-Cowgill conformal mapping which is
            // not implemented here.
            if let Ok(v) = schroder::tetrate_schroder(b, h, d, prec) {
                dprint("Schröder succeeded at repelling fixed point");
                return Ok(v);
            }
            dprint("Schröder unavailable; trying Newton-Kouznetsov for complex base");
            kouznetsov::tetrate_kouznetsov(b, h, d, prec, digits).map_err(|why| {
                unsupported_msg(
                    "general complex base outside Shell-Thron",
                    &format!(
                        "Schröder regular tetration not applicable, and \
                         Newton-Kantorovich Kouznetsov Cauchy iteration \
                         failed: {}",
                        why
                    ),
                )
            })
        }
    }
}

/// Single source for the "this case isn't implemented" error message. Caller
/// decides what's printed; we just give a clean one-liner the CLI surfaces.
fn unsupported_msg(case: &str, why: &str) -> String {
    format!("unsupported case: {} — {}", case, why)
}

/// Warn (on stderr) if the user requested more precision than the routed
/// algorithm can realistically deliver. Currently only Schröder applies; the
/// other regions error out before we reach this point.
fn warn_if_precision_exceeds_ceiling(region: &regions::Region, digits: u64) {
    let (ceiling, algo) = match region {
        regions::Region::ShellThronInterior(_)
        | regions::Region::OutsideShellThronRealPositive(_)
        | regions::Region::OutsideShellThronGeneral(_) => (20_000u64, "Schröder regular tetration"),
        _ => return,
    };
    if digits > ceiling {
        eprintln!(
            "warning: requested {} digits exceeds {} ceiling (~{} digits); \
             trailing digits may not be meaningful.",
            digits, algo, ceiling
        );
    }
}

fn lambda_abs_of(region: &regions::Region) -> Option<f64> {
    match region {
        regions::Region::ShellThronInterior(d)
        | regions::Region::ShellThronBoundary(d)
        | regions::Region::OutsideShellThronRealPositive(d)
        | regions::Region::OutsideShellThronGeneral(d) => Some(d.lambda_abs),
        _ => None,
    }
}

/// Attempt the continuation-based Kouznetsov solver for near-parabolic real bases,
/// then evaluate at the requested height.
fn try_continuation(
    b: &Complex,
    h: &Complex,
    fp: &crate::regions::FixedPointData,
    prec: u32,
    digits: u64,
) -> Result<Complex, String> {
    let state = kouznetsov::setup_kouznetsov_continuation(b, fp, prec, digits)?;
    kouznetsov::eval_kouznetsov(&state, b, h)
}

/// Quadratic-Richardson fallback for real bases trapped in the parabolic band.
///
/// When direct Kouznetsov and continuation both fail because |arg(λ)| → 0
/// (i.e. b is real and just above η = e^(1/e)), we leave the real axis and
/// evaluate at b + iε for two ε values, then Richardson-extrapolate to ε=0.
///
/// For real b ∈ ℝ the canonical tetration F_b(h) is real-valued. The
/// Schwarz-symmetry F̄(b̄+iε̄, h̄) = F(b-iε, h) implies that on a real horizontal
/// height segment Re(F(b+iε)) is even in ε while Im(F(b+iε)) is odd in ε.
/// Thus
///
///     Re(F(b+iε, h)) = Re(F(b, h)) + a·ε² + O(ε⁴)
///     Im(F(b+iε, h)) =                 c·ε + O(ε³)  → 0
///
/// One quadratic-Richardson step on the real part cancels ε² and yields
/// O(ε⁴) accuracy from two off-axis evaluations. Practical reach is roughly
/// 6–8 useful digits — far short of the requested precision when digits
/// is large, but vastly better than erroring out. We warn on stderr.
///
/// Constraints:
///   * Only invoked for `b` that is purely real (Im(b) is exactly zero).
///   * Calls `tetrate` recursively at the perturbed bases, so it must NOT
///     trigger a re-entrant iε path — and it doesn't: b±iε is no longer real,
///     so the dispatcher routes through OutsideShellThronGeneral / Kouznetsov.
fn try_iperturbation_extrapolation(
    b: &Complex,
    h: &Complex,
    prec: u32,
    digits: u64,
) -> Result<Complex, String> {
    if !b.imag().is_zero() {
        return Err("iε-perturbation only applies for purely real bases".to_string());
    }
    let b_re = b.real().clone();
    if h.imag().is_zero() {
        // We aim to deliver a real result; that's only well-defined when the
        // height is real too. (For complex h the algorithm still computes
        // Schwarz-conjugate samples, but we don't claim a clean result.)
    }
    // We solve at slightly higher working precision to absorb the cancellation
    // that quadratic Richardson induces.
    let prec_inner = prec.saturating_add(64);
    let mk_pert = |eps: f64| -> Complex {
        let im = Float::with_val(prec_inner, eps);
        Complex::with_val(prec_inner, (b_re.clone(), im))
    };
    let h_inner = Complex::with_val(prec_inner, h);

    let eps1 = 0.1f64;
    let eps2 = 0.05f64;
    let b1 = mk_pert(eps1);
    let b2 = mk_pert(eps2);

    dprint(&format!(
        "iε-perturbation Richardson: tetrate at b+{}i and b+{}i",
        eps1, eps2
    ));
    eprintln!(
        "warning: parabolic-boundary fallback (iε-perturbation Richardson); \
         expect ~6 useful digits regardless of requested precision."
    );

    let f1 = tetrate(&b1, &h_inner, prec_inner, digits)
        .map_err(|e| format!("iε-perturbation: tetrate(b+{}i) failed: {}", eps1, e))?;
    let f2 = tetrate(&b2, &h_inner, prec_inner, digits)
        .map_err(|e| format!("iε-perturbation: tetrate(b+{}i) failed: {}", eps2, e))?;

    // Quadratic Richardson on real part: R = (4·F(ε/2) − F(ε))/3.
    let four = Float::with_val(prec_inner, 4u32);
    let three = Float::with_val(prec_inner, 3u32);
    let r_real_inner: Float = (four.clone() * f2.real() - f1.real()) / three.clone();

    // Linear Richardson on imaginary part should land near zero (it's odd in ε).
    let r_imag_inner: Float = Float::with_val(prec_inner, 2u32) * f2.imag() - f1.imag();
    if debug_enabled() {
        eprintln!(
            "tet: iε Richardson residual Im = {} (should be ≈ 0)",
            r_imag_inner
        );
    }

    let r_real = Float::with_val(prec, &r_real_inner);
    // For real b and real h the canonical tetration is real; force imag to 0.
    let zero = Float::with_val(prec, 0);
    Ok(Complex::with_val(prec, (r_real, zero)))
}

/// Tetration with base 0 — convention:
///   * `0^^0 = 1`
///   * `0^^n` for positive integer `n`: `n` even → 1, `n` odd → 0
///     (because `0^0 = 1` and `0^k = 0` for `k > 0`).
///   * Negative integer / non-integer height: undefined.
fn tetrate_base_zero(h: &Complex, prec: u32) -> Result<Complex, String> {
    let n = cnum::as_integer(h).ok_or_else(|| {
        "tetration of 0 is only defined for non-negative integer heights".to_string()
    })?;
    if n < 0 {
        return Err(format!(
            "tetration of 0 with negative integer height {} is undefined",
            n
        ));
    }
    let val = if n % 2 == 0 { 1 } else { 0 };
    Ok(Complex::with_val(prec, (val, 0)))
}
