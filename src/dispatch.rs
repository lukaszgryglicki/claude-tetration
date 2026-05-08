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
            // (geometric rate `|λ|`, which is near 1) and typically bails.
            //
            // For real positive bases (b > η, |λ| just above 1) on this band,
            // |arg(λ)| is small so direct Kouznetsov has trouble converging on
            // a feasible-sized grid (see kouznetsov.rs for the parabolic-cap
            // logic). The continuation solver — which warm-starts from a base
            // farther from the boundary and walks back — is much more reliable
            // there, so we try it FIRST and only fall through to direct
            // Kouznetsov if continuation also fails. iε-perturbation Richardson
            // is the final fallback (real bases only, ~9 digits).
            //
            // For complex bases on the boundary, |arg(λ)| can already be large,
            // so direct Kouznetsov often works; we keep the original order.
            match schroder::tetrate_schroder(b, h, d, prec) {
                Ok(v) => {
                    dprint("Schröder succeeded at boundary band");
                    return Ok(v);
                }
                Err(e) => dprint(&format!("Schröder failed at boundary band: {}", e)),
            }
            let is_real_base =
                b.imag().is_zero() && !b.real().is_sign_negative();
            if is_real_base {
                // Direct Kouznetsov is essentially hopeless for real bases on
                // the parabolic boundary (|arg λ| → 0 forces t_max → ∞). We
                // skip it entirely and try, in order:
                //   1. Continuation solver (warm-starts from a non-parabolic
                //      base) — fast for moderate |λ| above 1.
                //   2. iε-perturbation Richardson — last resort, ~9 digits.
                dprint("real-base parabolic boundary: trying continuation solver");
                let cont_res = try_continuation(b, h, d, prec, digits);
                if let Ok(v) = cont_res {
                    return Ok(v);
                }
                let cont_err = cont_res.err().unwrap_or_default();
                dprint(&format!(
                    "continuation failed: {}; trying iε-perturbation Richardson",
                    cont_err
                ));
                if let Ok(v) = try_iperturbation_extrapolation(b, h, prec, digits) {
                    return Ok(v);
                }
                return Err(unsupported_msg(
                    "Shell-Thron parabolic boundary band (|λ| ≈ 1)",
                    &format!(
                        "Schröder too slow; continuation: {}; \
                         iε-perturbation Richardson also failed",
                        cont_err
                    ),
                ));
            }
            dprint("complex-base boundary band: trying Newton-Kouznetsov");
            match kouznetsov::tetrate_kouznetsov(b, h, d, prec, digits) {
                Ok(v) => Ok(v),
                Err(why) => {
                    if why.contains("parabolic boundary") || why.contains("degenerate contour") {
                        dprint("direct Kouznetsov hit parabolic cap; trying continuation");
                        if let Ok(v) = try_continuation(b, h, d, prec, digits) {
                            return Ok(v);
                        }
                    }
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
                    // For real positive bases on or near the parabolic boundary,
                    // try continuation + iε-perturbation Richardson on any
                    // Kouznetsov failure (the classification can put a base in
                    // OutsideShellThronRealPositive but with |λ| only slightly
                    // > 1.05, where Kouznetsov still struggles).
                    let is_real_base = b.imag().is_zero();
                    let near_boundary = d.lambda_abs < 1.10;
                    let parabolic_signal = why.contains("parabolic boundary")
                        || why.contains("degenerate contour")
                        || (is_real_base
                            && near_boundary
                            && (why.contains("did not converge")
                                || why.contains("residual")
                                || why.contains("n_nodes")));
                    if parabolic_signal {
                        dprint("Kouznetsov failed near parabolic boundary; trying continuation solver");
                        let cont_outcome = try_continuation(b, h, d, prec, digits);
                        if let Ok(v) = cont_outcome {
                            return Ok(v);
                        }
                        let cont_err = cont_outcome.err().unwrap_or_default();
                        if is_real_base {
                            dprint(&format!(
                                "continuation also failed: {}; trying iε-perturbation Richardson",
                                cont_err
                            ));
                            if let Ok(v) =
                                try_iperturbation_extrapolation(b, h, prec, digits)
                            {
                                return Ok(v);
                            }
                        }
                        Err(unsupported_msg(
                            "real base > e^(1/e)",
                            &format!(
                                "Schröder not applicable; Kouznetsov direct: {}; \
                                 continuation: {}; iε-perturbation also failed",
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

/// Four-point Richardson fallback for real bases trapped in the parabolic band.
///
/// When direct Kouznetsov and continuation both fail because |arg(λ)| → 0
/// (i.e. b is real and just above η = e^(1/e)), we leave the real axis and
/// evaluate at b + iε for several ε values, then Richardson-extrapolate to
/// ε = 0.
///
/// For real b ∈ ℝ the canonical Kneser tetration F_b(h) is real-valued. The
/// Schwarz-symmetry F̄(b̄+iε̄, h̄) = F(b-iε, h) implies that on a real horizontal
/// height segment Re(F(b+iε)) is even in ε while Im(F(b+iε)) is odd in ε.
/// Thus `Re(F(b+iε, h)) = Re(F(b, h)) + a₂·ε² + a₄·ε⁴ + a₆·ε⁶ + O(ε⁸)` and
/// `Im(F(b+iε, h)) = c₁·ε + c₃·ε³ + O(ε⁵) → 0`.
///
/// Romberg-style table on five evaluations at ε ∈ {0.1, 0.05, 0.025, 0.0125, 0.00625}:
/// `R₁(ε) = (4·F(ε/2) − F(ε))/3` cancels ε² → O(ε⁴);
/// `R₂(ε) = (16·R₁(ε/2) − R₁(ε))/15` cancels ε⁴ → O(ε⁶);
/// `R₃(ε) = (64·R₂(ε/2) − R₂(ε))/63` cancels ε⁶ → O(ε⁸);
/// `R₄(ε) = (256·R₃(ε/2) − R₃(ε))/255` cancels ε⁸ → O(ε¹⁰).
///
/// With five function evaluations this yields roughly 18–22 useful digits in
/// favourable regimes — theoretical residual O(0.00625¹⁰) ≈ 9e-23 — at the
/// cost of one additional perturbed Kouznetsov solve (~25% more wall time
/// vs R₃). The smallest ε used is 0.00625; empirically at b=1.4448 this
/// gives |λ|≈0.899 (still well-interior in Shell-Thron), and the perturbed
/// evaluation runs the Schröder interior path in ~8s at 20 digits.
///
/// Constraints:
///   * Only invoked for `b` purely real (Im(b) is exactly zero).
///   * Calls `tetrate` recursively at the perturbed bases. The dispatcher
///     routes b+iε as OutsideShellThronGeneral → Kouznetsov, so this does
///     not re-enter the iε fallback.
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
    let h_is_real = h.imag().is_zero();

    let prec_inner = prec.saturating_add(64);
    let mk_pert = |eps: f64| -> Complex {
        let im = Float::with_val(prec_inner, eps);
        Complex::with_val(prec_inner, (b_re.clone(), im))
    };
    let h_inner = Complex::with_val(prec_inner, h);
    let h_inner_conj = Complex::with_val(prec_inner, h.conj_ref());

    let eps0 = 0.1f64;
    let eps1 = 0.05f64;
    let eps2 = 0.025f64;
    let eps3 = 0.0125f64;
    let eps4 = 0.00625f64;

    dprint(&format!(
        "iε Richardson R₄: tetrate at b+{}i, b+{}i, b+{}i, b+{}i, b+{}i (h_is_real={})",
        eps0, eps1, eps2, eps3, eps4, h_is_real
    ));
    eprintln!(
        "warning: parabolic-boundary fallback (iε-perturbation Richardson R₄); \
         expect roughly 18–22 useful digits regardless of requested precision."
    );

    // Evaluate G(ε) at four ε. For real h, G(ε) := F(b+iε, h); the Schwarz
    // symmetry F̄(b̄+iε̄, h̄) = F(b−iε, h) gives Re(G) even and Im(G) odd in ε,
    // so the canonical real F(b, h) is the ε→0 limit of Re(G).
    //
    // For complex h, the parity breaks. We restore it by symmetrizing manually:
    //     G(ε) := (F(b+iε, h) + conj(F(b+iε, conj(h)))) / 2.
    // This averages the b+iε branch with the b−iε branch (the latter is
    // F(b−iε, h) = conj(F(b+iε, conj(h))) by Schwarz). The resulting G(ε) is
    // even in ε with leading correction O(ε²), exactly like the real-h case;
    // it is complex-valued (the desired F(b, h)).
    let eval = |eps: f64| -> Result<Complex, String> {
        let f_plus = tetrate(&mk_pert(eps), &h_inner, prec_inner, digits)
            .map_err(|e| format!("iε-perturbation: tetrate(b+{}i, h) failed: {}", eps, e))?;
        if h_is_real {
            return Ok(f_plus);
        }
        let f_plus_conjh = tetrate(&mk_pert(eps), &h_inner_conj, prec_inner, digits)
            .map_err(|e| format!("iε-perturbation: tetrate(b+{}i, conj(h)) failed: {}", eps, e))?;
        let f_plus_conjh_conj = Complex::with_val(prec_inner, f_plus_conjh.conj_ref());
        let two = Float::with_val(prec_inner, 2u32);
        let avg = (f_plus + f_plus_conjh_conj) / Complex::with_val(prec_inner, &two);
        Ok(avg)
    };

    let g0 = eval(eps0)?;
    let g1 = eval(eps1)?;
    let g2 = eval(eps2)?;
    let g3 = eval(eps3)?;
    let g4 = eval(eps4)?;

    // R₁ on G — four values at consecutive (ε, ε/2) pairs cancel ε² → O(ε⁴).
    let three = Complex::with_val(prec_inner, 3u32);
    let four = Complex::with_val(prec_inner, 4u32);
    let r1_a: Complex = (four.clone() * &g1 - &g0) / three.clone(); // pair (eps0, eps1)
    let r1_b: Complex = (four.clone() * &g2 - &g1) / three.clone(); // pair (eps1, eps2)
    let r1_c: Complex = (four.clone() * &g3 - &g2) / three.clone(); // pair (eps2, eps3)
    let r1_d: Complex = (four * &g4 - &g3) / three; // pair (eps3, eps4)

    // R₂: cancels ε⁴ → O(ε⁶).
    let fifteen = Complex::with_val(prec_inner, 15u32);
    let sixteen = Complex::with_val(prec_inner, 16u32);
    let r2_a: Complex = (sixteen.clone() * &r1_b - &r1_a) / fifteen.clone();
    let r2_b: Complex = (sixteen.clone() * &r1_c - &r1_b) / fifteen.clone();
    let r2_c: Complex = (sixteen * &r1_d - &r1_c) / fifteen;

    // R₃: cancels ε⁶ → O(ε⁸).
    let sixty_three = Complex::with_val(prec_inner, 63u32);
    let sixty_four = Complex::with_val(prec_inner, 64u32);
    let r3_a: Complex = (sixty_four.clone() * &r2_b - &r2_a) / sixty_three.clone();
    let r3_b: Complex = (sixty_four * &r2_c - &r2_b) / sixty_three;

    // R₄: cancels ε⁸ → O(ε¹⁰).
    let two_fifty_five = Complex::with_val(prec_inner, 255u32);
    let two_fifty_six = Complex::with_val(prec_inner, 256u32);
    let r4: Complex = (two_fifty_six * &r3_b - &r3_a) / two_fifty_five;

    if debug_enabled() {
        let r3_diff: Complex = Complex::with_val(prec_inner, &r3_a - &r3_b);
        eprintln!(
            "tet: iε Richardson R₃(a)−R₃(b) = {} (≈ leading O(ε⁸))",
            r3_diff
        );
    }

    if h_is_real {
        // Force imag to 0 for real-base, real-h Kneser tetration.
        let r_real = Float::with_val(prec, r4.real());
        let zero = Float::with_val(prec, 0);
        Ok(Complex::with_val(prec, (r_real, zero)))
    } else {
        Ok(Complex::with_val(prec, &r4))
    }
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
