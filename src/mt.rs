//! Opt-in multithreading ("MT mode").
//!
//! Everything here is gated on the `TET_MT` environment variable:
//!
//! * unset, empty, or `0` — MT mode **off** (the default). Every gated call
//!   site takes its original serial branch, byte-for-byte the same code path
//!   that shipped before MT mode existed. Nothing about the default build's
//!   numerical behavior changes.
//! * `1` — MT mode on, using all available logical cores.
//! * `n ≥ 2` — MT mode on with a global rayon pool of exactly `n` threads.
//!
//! # Bit-identical guarantee
//! The parallel branches are restricted to operations whose outputs are
//! independent per element (FFT butterflies over disjoint index pairs,
//! pointwise complex multiplies, per-row boundary corrections, per-node
//! transcendental maps). No floating-point **accumulation** is ever
//! reordered: GMRES inner products, norms, and Euler–Maclaurin sums stay
//! serial. MPFR/MPC arithmetic is correctly rounded and deterministic, so
//! with identical operands and operation order per element the MT-mode
//! outputs are bit-identical to the serial ones. This is verified by A/B
//! `diff` tests (see README §3, updates.md).

use std::sync::OnceLock;

/// Frozen at first use, including parse errors.
fn mt_setting() -> &'static Result<Option<usize>, String> {
    static SETTING: OnceLock<Result<Option<usize>, String>> = OnceLock::new();
    SETTING.get_or_init(|| {
        let raw = match std::env::var("TET_MT") {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => return Ok(None),
            Err(e) => return Err(format!("TET_MT: {e}")),
        };
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        let n = trimmed
            .parse::<usize>()
            .map_err(|e| format!("TET_MT: {e}"))?;
        if n > rayon::max_num_threads() {
            return Err(format!("TET_MT={n} exceeds Rayon's thread-count limit"));
        }
        Ok(match n {
            0 => None,
            1 => Some(
                std::thread::available_parallelism()
                    .map_err(|e| format!("cannot determine the available thread count: {e}"))?
                    .get(),
            ),
            n => Some(n),
        })
    })
}

/// True iff MT mode is enabled (`TET_MT` = 1 or ≥ 2).
/// Infallible low-level APIs panic on configuration errors; fallible entry
/// points call `init_pool` first and return those errors to their callers.
pub fn mt_enabled() -> bool {
    init_pool().expect("invalid tetration multithreading configuration");
    mt_setting()
        .as_ref()
        .expect("validated MT setting")
        .is_some()
}

/// Initialize once for CLI and library callers, before any parallel work.
/// An existing Rayon pool is an explicit conflict, not a silent thread-count
/// override. Serial mode never initializes a pool.
pub fn init_pool() -> Result<(), String> {
    static INITIALIZED: OnceLock<Result<(), String>> = OnceLock::new();
    INITIALIZED
        .get_or_init(|| {
            if let Some(n) = mt_setting().as_ref().map_err(Clone::clone)? {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(*n)
                    .build_global()
                    .map_err(|e| {
                        format!("cannot initialize the requested tetration Rayon pool: {e}")
                    })?;
                let actual = rayon::current_num_threads();
                if actual != *n {
                    return Err(format!(
                        "requested {n} MT threads, but Rayon initialized {actual}"
                    ));
                }
                if crate::cnum::verbose() {
                    eprintln!("tet: MT mode initialized with {actual} threads");
                }
            }
            Ok(())
        })
        .clone()
}
