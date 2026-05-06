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

use rug::Complex;

use crate::{cnum, integer_height, kouznetsov, regions, schroder};

fn debug_enabled() -> bool {
    std::env::var_os("TET_DEBUG").is_some_and(|v| !v.is_empty() && v != "0")
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
            schroder::tetrate_schroder(b, h, d, prec)
        }
        regions::Region::ShellThronBoundary(_) => Err(unsupported_msg(
            "Shell-Thron parabolic boundary band (|λ| ≈ 1)",
            "no convergent algorithm in this band; Paulsen-Cowgill / Kouznetsov \
             Cauchy iteration is required and not implemented in this build",
        )),
        regions::Region::OutsideShellThronRealPositive(d) => {
            // Schröder at the repelling fixed point handles bases unusually
            // close to the Shell-Thron boundary; for typical real bases
            // (e, 2, 10, …) the σ̃-shift orbit hits the log singularity and
            // Schröder bails. Fall through to Anderson-accelerated Kouznetsov
            // Cauchy iteration, which produces the natural real-on-real
            // tetration via samples on Re(z)=0.5 refined by Cauchy's formula.
            if let Ok(v) = schroder::tetrate_schroder(b, h, d, prec) {
                dprint("Schröder succeeded at repelling fixed point");
                return Ok(v);
            }
            dprint("Schröder unavailable; switching to Newton-Kouznetsov");
            kouznetsov::tetrate_kouznetsov(b, h, d, prec, digits).map_err(|why| {
                unsupported_msg(
                    "real base > e^(1/e)",
                    &format!(
                        "Schröder regular tetration not applicable, and \
                         Newton-Kantorovich Kouznetsov Cauchy iteration \
                         failed: {}",
                        why
                    ),
                )
            })
        }
        regions::Region::OutsideShellThronGeneral(d) => {
            // Try Schröder at the repelling fixed point first (cheap when it
            // works). The generalized Kouznetsov path exists in
            // `kouznetsov.rs` (no Schwarz symmetry, fixed points pulled from
            // both W₀ and W₋₁ branches) but is not wired in: Paulsen-Cowgill-
            // style branch selection / conformal mapping is required to place
            // the two fixed points on opposite contour edges for general
            // complex `b`, and that work is research-grade and pending.
            match schroder::tetrate_schroder(b, h, d, prec) {
                Ok(v) => Ok(v),
                Err(why) => Err(unsupported_msg(
                    "general complex base outside Shell-Thron",
                    &format!(
                        "Schröder regular tetration not applicable here ({}); \
                         Paulsen-Cowgill is not implemented in this build",
                        why
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
