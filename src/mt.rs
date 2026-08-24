//! Opt-in multithreading ("MT mode").
//!
//! Everything here is gated on the `TET_MT` environment variable:
//!
//! * unset, empty, or `0` — MT mode **off** (the default). Every gated call
//!   site takes its original serial branch, byte-for-byte the same code path
//!   that shipped before MT mode existed. Nothing about the default build's
//!   numerical behavior changes.
//! * `1` — MT mode on, using rayon's default global pool (all logical cores).
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

/// Parsed TET_MT setting: `None` = off, `Some(0)` = on with default pool,
/// `Some(n)` = on with n threads.
fn mt_setting() -> Option<usize> {
    static SETTING: OnceLock<Option<usize>> = OnceLock::new();
    *SETTING.get_or_init(|| {
        let raw = std::env::var("TET_MT").ok()?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        match trimmed.parse::<usize>() {
            Ok(0) => None,
            Ok(1) => Some(0),
            Ok(n) => Some(n),
            Err(_) => None,
        }
    })
}

/// True iff MT mode is enabled (`TET_MT` = 1 or ≥ 2).
pub fn mt_enabled() -> bool {
    mt_setting().is_some()
}

/// Install the requested global rayon pool size. Call once at startup,
/// before any rayon work. With `TET_MT` unset/0/1 this does nothing (the
/// pre-existing rayon call sites keep using the default pool exactly as
/// before). Errors (pool already built) are ignored — the default pool is a
/// safe fallback.
pub fn init_pool() {
    if let Some(n) = mt_setting() {
        if n >= 2 {
            let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
        }
    }
}
