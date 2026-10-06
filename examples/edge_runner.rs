//! Comprehensive edge-case runner for complex tetration.
//!
//! Covers every algorithmic boundary, axis, and extreme-magnitude case:
//!   1. Shell-Thron parabolic boundary (b near η = e^(1/e)) — exact & ε-perturbations
//!   2. Algorithm dispatch threshold (|λ| ≈ 0.95/1.05) — both sides
//!   3. Real axis: negative integers, negative non-integers, (0,1), (1,η), (η,∞)
//!   4. Imaginary axis: ±ki for integer and non-integer k
//!   5. Re = Im diagonal; Re = −Im anti-diagonal
//!   6. Pure negative integer bases: −1, −2, −3, −4, −5
//!   7. Pure imaginary integer bases: i, 2i, 3i, −i, −2i (Schwarz path)
//!   8. Near-zero bases and heights (|b| or |h| ≈ 1e-6)
//!   9. Large-magnitude: ±100, ±100i, ±100±100i for both b and h
//!  10. Integer heights: 0,1,2,3,−1 for every base
//!
//! For every non-integer-height result that succeeds, the functional equation
//! |F(h+1) − b^F(h)| / max(|F(h+1)|, 1) < 10^(-digits) is checked.
//! This is a consistency diagnostic, not an independent error certificate.
//!
//! Usage:
//!   cargo run --release --example edge_runner [digits]
//!   defaults: digits=20

use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;
use rug::{Complex, Float};
use tetration::{cnum, dispatch, kouznetsov, regions, schroder};

// ──────────────────────────────────────────────────────────────────────────────
// Test-case definitions
// ──────────────────────────────────────────────────────────────────────────────

/// A base to probe, with a human-readable category label.
#[derive(Clone)]
struct BaseCase {
    re: Float,
    im: Float,
    label: &'static str,
}

fn all_bases(prec: u64) -> Vec<BaseCase> {
    let eta = cnum::eta_upper(prec);
    let mut v: Vec<BaseCase> = Vec::new();
    for (offset, label) in [
        ("0", "ST/boundary: b=η rounded at working precision"),
        ("1e-5", "ST/boundary: b=η+1e-5 (ε outside)"),
        ("1e-3", "ST/boundary: b=η+1e-3 (ε outside)"),
        ("0.01", "ST/boundary: b=η+0.01 (outside)"),
        ("-1e-5", "ST/boundary: b=η-1e-5 (ε inside, Schröder)"),
        ("-1e-3", "ST/boundary: b=η-1e-3 (ε inside)"),
        ("-0.01", "ST/boundary: b=η-0.01 (inside)"),
    ] {
        v.push(BaseCase {
            re: eta.clone() + cnum::decimal(offset, prec),
            im: Float::new_64(prec),
            label,
        });
    }
    for (lambda, label) in [
        ("0.949999", "dispatch-threshold: lambda=0.949999 (interior)"),
        (
            "0.950001",
            "dispatch-threshold: lambda=0.950001 (boundary band)",
        ),
    ] {
        let lambda = cnum::decimal(lambda, prec);
        let re = (lambda.clone() * (-lambda).exp()).exp();
        v.push(BaseCase {
            re,
            im: Float::new_64(prec),
            label,
        });
    }
    for (re, im, label) in [
        ("0.1", "0.995", "ST/complex probe: 0.1+0.995i"),
        ("-0.5", "0.87", "ST/complex probe: -0.5+0.87i"),
        ("1.40", "0", "dispatch-threshold: b=1.40 (mid-range)"),
        ("1.43", "0", "dispatch-threshold: b=1.43 (near η)"),
        ("0", "0", "special: b=0"),
        ("1", "0", "special: b=1"),
        ("-1", "0", "real-neg-int: b=−1"),
        ("-2", "0", "real-neg-int: b=−2"),
        ("-3", "0", "real-neg-int: b=−3"),
        ("-4", "0", "real-neg-int: b=−4"),
        ("-5", "0", "real-neg-int: b=−5"),
        ("-0.5", "0", "real-neg: b=−0.5"),
        ("-1.5", "0", "real-neg: b=−1.5"),
        ("-2.5", "0", "real-neg: b=−2.5"),
        ("-3.5", "0", "real-neg: b=−3.5"),
        ("0.1", "0", "real-01: b=0.1"),
        ("0.5", "0", "real-01: b=0.5"),
        ("0.9", "0", "real-01: b=0.9"),
        ("0.99", "0", "real-01: b=0.99"),
        ("1.1", "0", "real-1eta: b=1.1 (Schröder)"),
        ("1.2", "0", "real-1eta: b=1.2 (Schröder)"),
        ("1.3", "0", "real-1eta: b=1.3 (Schröder)"),
        ("1.4", "0", "real-1eta: b=1.4 (Schröder)"),
        ("1.5", "0", "real-gt-eta: b=1.5"),
        ("1.6", "0", "real-gt-eta: b=1.6"),
        ("2", "0", "real-gt-eta: b=2"),
        ("3", "0", "real-gt-eta: b=3"),
        ("5", "0", "real-gt-eta: b=5"),
        ("10", "0", "real-gt-eta: b=10"),
        ("0", "1", "imag-int: b=i (|b|=1)"),
        ("0", "2", "imag-int: b=2i"),
        ("0", "3", "imag-int: b=3i"),
        ("0", "5", "imag-int: b=5i"),
        ("0", "0.5", "imag: b=0.5i (inside ST)"),
        ("0", "1.2", "imag: b=1.2i (near ST boundary)"),
        ("0", "1.5", "imag: b=1.5i"),
        ("0", "-1", "imag-neg-int: b=−i (Schwarz)"),
        ("0", "-2", "imag-neg-int: b=−2i (Schwarz)"),
        ("0", "-3", "imag-neg-int: b=−3i (Schwarz)"),
        ("0", "-0.5", "imag-neg: b=−0.5i (Schwarz)"),
        ("0", "-1.5", "imag-neg: b=−1.5i (Schwarz)"),
        ("0.5", "0.5", "diag-pos: b=0.5+0.5i"),
        ("1", "1", "diag-pos: b=1+i"),
        ("2", "2", "diag-pos: b=2+2i"),
        ("3", "3", "diag-pos: b=3+3i"),
        ("-1", "-1", "diag-neg: b=−1−i"),
        ("-2", "-2", "diag-neg: b=−2−2i (Schwarz)"),
        ("1", "-1", "anti-diag: b=1−i"),
        ("2", "-2", "anti-diag: b=2−2i (Schwarz)"),
        ("-1", "1", "anti-diag: b=−1+i"),
        ("-2", "2", "anti-diag: b=−2+2i"),
        ("1e-6", "0", "near-zero: b=1e-6+0i"),
        ("0", "1e-6", "near-zero: b=0+1e-6i"),
        ("1e-4", "1e-4", "near-zero: b=1e-4+1e-4i"),
        ("-1e-4", "0", "near-zero: b=−1e-4+0i"),
        ("100", "0", "large: b=100"),
        ("-100", "0", "large: b=−100"),
        ("0", "100", "large: b=100i"),
        ("0", "-100", "large: b=−100i (Schwarz)"),
        ("100", "100", "large: b=100+100i"),
        ("100", "-100", "large: b=100−100i (Schwarz)"),
        ("-100", "100", "large: b=−100+100i"),
        ("-100", "-100", "large: b=−100−100i (Schwarz)"),
        ("1.2", "3.5", "complex: b=1.2+3.5i"),
        ("-1.2", "1.2", "complex: b=−1.2+1.2i"),
        ("-3.6", "0.4", "complex: b=−3.6+0.4i"),
    ] {
        v.push(BaseCase {
            re: cnum::decimal(re, prec),
            im: cnum::decimal(im, prec),
            label,
        });
    }
    v.push(BaseCase {
        re: Float::with_val_64(prec, 1).exp(),
        im: Float::new_64(prec),
        label: "real-gt-eta: b=e",
    });
    v
}

/// Heights to probe for each base.
fn all_heights() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        // Integer heights
        ("0", "0", "h=0"),
        ("1", "0", "h=1"),
        ("2", "0", "h=2"),
        ("3", "0", "h=3"),
        ("-1", "0", "h=−1 (F=0)"),
        ("-2", "0", "h=−2 (undef except b=1)"),
        // Non-integer real
        ("0.5", "0", "h=0.5"),
        ("-0.5", "0", "h=−0.5"),
        ("0.3", "0", "h=0.3"),
        ("0.7", "0", "h=0.7"),
        ("1.5", "0", "h=1.5"),
        ("2.5", "0", "h=2.5"),
        // Complex heights
        ("0.5", "0.5", "h=0.5+0.5i"),
        ("0.5", "-0.5", "h=0.5−0.5i"),
        ("0.3", "0.7", "h=0.3+0.7i"),
        ("1", "1", "h=1+i"),
        ("2", "1", "h=2+i"),
        // Pure imaginary heights
        ("0", "0.5", "h=0.5i"),
        ("0", "1", "h=i"),
        ("0", "-0.5", "h=−0.5i"),
        // Large heights
        ("5", "0", "h=5"),
        ("10", "0", "h=10"),
        ("0", "5", "h=5i"),
        ("5", "5", "h=5+5i"),
        // Near-zero heights
        ("1e-6", "0", "h=1e-6"),
        ("0", "1e-6", "h=1e-6i"),
    ]
}

// ──────────────────────────────────────────────────────────────────────────────
// Cache infrastructure (mirrors grid_runner)
// ──────────────────────────────────────────────────────────────────────────────

enum BaseCache {
    SpecialBase,
    DispatchFallback,
    SchroderCached(schroder::SchroderState),
    KouznetsovCached(kouznetsov::KouznetsovState),
    SetupErrorFallback,
}

impl BaseCache {
    fn kind_label(&self) -> &'static str {
        match self {
            BaseCache::SpecialBase => "special",
            BaseCache::DispatchFallback => "dispatch",
            BaseCache::SchroderCached(_) => "schr",
            BaseCache::KouznetsovCached(_) => "kouz",
            BaseCache::SetupErrorFallback => "fallback",
        }
    }
}

fn build_cache(b: &Complex, prec: u64, digits: u64) -> BaseCache {
    if cnum::is_zero(b) || cnum::is_one(b) {
        return BaseCache::SpecialBase;
    }
    if b.imag().is_sign_negative() && !b.imag().is_zero() {
        return BaseCache::DispatchFallback;
    }
    let region = match regions::classify(b, prec) {
        Ok(r) => r,
        Err(_) => return BaseCache::DispatchFallback,
    };
    match &region {
        regions::Region::BaseZero | regions::Region::BaseOne => BaseCache::SpecialBase,
        regions::Region::ShellThronInterior(d) | regions::Region::ShellThronBoundary(d)
            if d.lambda_abs < 1 =>
        {
            match schroder::setup_schroder(b, d, prec) {
                Ok(s) => BaseCache::SchroderCached(s),
                Err(e) => {
                    if cnum::verbose() {
                        eprintln!("edge Schröder setup failed; deferring to dispatcher: {e}");
                    }
                    BaseCache::SetupErrorFallback
                }
            }
        }
        regions::Region::OutsideShellThronRealPositive(d)
            if b.imag().is_zero() && *b.real() > cnum::eta_upper(prec) =>
        {
            match kouznetsov::setup_kouznetsov(b, d, prec, digits) {
                Ok(s) => BaseCache::KouznetsovCached(s),
                Err(e) => {
                    if cnum::verbose() {
                        eprintln!("edge Kouznetsov setup failed; deferring to dispatcher: {e}");
                    }
                    BaseCache::SetupErrorFallback
                }
            }
        }
        _ => BaseCache::DispatchFallback,
    }
}

fn eval_cell(
    cache: &BaseCache,
    b: &Complex,
    h: &Complex,
    prec: u64,
    digits: u64,
) -> Result<Complex, String> {
    if h.imag().is_zero() && h.real().is_integer() {
        return dispatch::tetrate(b, h, prec, digits);
    }
    let cached = match cache {
        BaseCache::SpecialBase | BaseCache::DispatchFallback | BaseCache::SetupErrorFallback => {
            return dispatch::tetrate(b, h, prec, digits)
        }
        BaseCache::SchroderCached(state) => schroder::eval_schroder_at_digits(state, h, digits),
        BaseCache::KouznetsovCached(state) => {
            kouznetsov::eval_kouznetsov_at_digits(state, b, h, digits)
        }
    };
    match cached {
        Ok(value) => Ok(value),
        Err(error) => {
            if cnum::verbose() {
                eprintln!("edge cached evaluation failed; trying dispatcher: {error}");
            }
            dispatch::tetrate(b, h, prec, digits)
        }
    }
}

fn domain_undefined(b: &Complex, h: &Complex) -> Option<&'static str> {
    let integer = h.imag().is_zero() && h.real().is_integer();
    if cnum::is_one(b) {
        None
    } else if cnum::is_zero(b) && !(integer && *h.real() >= 0) {
        Some("b=0: only non-negative integer h defined")
    } else if !cnum::is_zero(b) && integer && *h.real() <= -2 {
        Some("integer height <= -2: log_b(0) is undefined")
    } else {
        None
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Functional-equation check
// ──────────────────────────────────────────────────────────────────────────────

/// Compute |F(h+1) − b^F(h)| / max(|F(h+1)|, 1).
/// An unavailable comparison is an explicit error, never a passing check.
fn functional_eq_residual(
    cache: &BaseCache,
    b: &Complex,
    h: &Complex,
    fh: &Complex,
    prec: u64,
    digits: u64,
) -> Result<Float, String> {
    if !cnum::is_finite(fh) {
        return Err("functional-equation input is non-finite".into());
    }
    let h1 = Complex::with_val_64(prec, h + 1u32);
    let fh1 = eval_cell(cache, b, &h1, prec, digits)?;
    let ln_b = cnum::ln_complex(b, prec);
    let b_to_fh = cnum::checked_exp(&Complex::with_val_64(prec, &ln_b * fh), prec)?;
    let diff = Complex::with_val_64(prec, &fh1 - &b_to_fh);
    let residual = cnum::abs(&diff, prec) / cnum::abs(&fh1, prec).max(&Float::with_val_64(prec, 1));
    if !residual.is_finite() {
        return Err("functional-equation residual is non-finite".into());
    }
    Ok(residual)
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct Stats {
    ok: usize,
    ok_feq_fail: usize, // ok result but f-eq check failed
    undef: usize,
    err: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().collect();
    let quiet = args
        .iter()
        .skip(1)
        .any(|s| matches!(s.as_str(), "--quiet" | "--silent" | "-q"));
    args.retain(|s| !matches!(s.as_str(), "--quiet" | "--silent" | "-q"));
    cnum::set_quiet(quiet);
    if args.len() > 2 {
        return Err("usage: edge_runner [--quiet] [digits]".into());
    }
    let digits = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("20")
        .parse::<u64>()
        .map_err(|e| format!("invalid digits: {e}"))?;
    let prec = cnum::checked_digits_to_bits(digits)?;
    let output_digits =
        usize::try_from(digits).map_err(|_| "output precision exceeds addressable memory")?;
    tetration::mt::init_pool()?;
    let bases = all_bases(prec);
    let heights = all_heights();

    let n_bases = bases.len();
    let n_heights = heights.len();
    let n_cells = n_bases * n_heights;

    if cnum::verbose() {
        eprintln!("edge_runner: digits={digits}, bases={n_bases}, heights={n_heights}, cells={n_cells}, MT={}",
            tetration::mt::mt_enabled());
    }

    let stdout_mutex: Arc<Mutex<()>> = Arc::new(Mutex::new(()));
    let total_stats: Arc<Mutex<Stats>> = Arc::new(Mutex::new(Stats::default()));
    let bases_done: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let global_start = Instant::now();

    // Print TSV header
    {
        let stdout = io::stdout();
        let mut g = stdout.lock();
        writeln!(g, "category\tb_re\tb_im\th_label\tstatus\tresult_re\tresult_im\tfeq_residual\telapsed_secs\terror")
            .map_err(|e| format!("write edge header: {e}"))?;
        g.flush().map_err(|e| format!("flush edge header: {e}"))?;
    }

    let process_base = |base_case: &BaseCase| -> Result<(), String> {
        let b = Complex::with_val_64(prec, (&base_case.re, &base_case.im));
        let t0 = Instant::now();
        let cache = build_cache(&b, prec, digits);

        let mut buf: Vec<u8> = Vec::with_capacity(n_heights * 120);
        let mut base_ok = 0usize;
        let mut base_ok_feq_fail = 0usize;
        let mut base_undef = 0usize;
        let mut base_err = 0usize;

        for &(h_re, h_im, h_label) in &heights {
            let h =
                Complex::with_val_64(prec, (cnum::decimal(h_re, prec), cnum::decimal(h_im, prec)));

            if let Some(reason) = domain_undefined(&b, &h) {
                use std::io::Write as _;
                writeln!(
                    &mut buf,
                    "{}\t{}\t{}\t{}\tundef\t\t\t\t0.000\t{}",
                    base_case.label,
                    cnum::DisplayFloat(&base_case.re),
                    cnum::DisplayFloat(&base_case.im),
                    h_label,
                    reason
                )
                .map_err(|e| format!("buffer edge row: {e}"))?;
                base_undef += 1;
                continue;
            }

            let ct0 = Instant::now();
            let result = eval_cell(&cache, &b, &h, prec, digits);
            let elapsed = ct0.elapsed().as_secs_f64();

            match result {
                Err(e) => {
                    use std::io::Write as _;
                    let one = e
                        .lines()
                        .next()
                        .unwrap_or("")
                        .chars()
                        .take(200)
                        .collect::<String>();
                    writeln!(
                        &mut buf,
                        "{}\t{}\t{}\t{}\terror\t\t\t\t{:.3}\t{}",
                        base_case.label,
                        cnum::DisplayFloat(&base_case.re),
                        cnum::DisplayFloat(&base_case.im),
                        h_label,
                        elapsed,
                        one
                    )
                    .map_err(|e| format!("buffer edge row: {e}"))?;
                    base_err += 1;
                }
                Ok(fh) => {
                    // Functional-equation check for non-integer heights.
                    let feq = if !(h.imag().is_zero() && h.real().is_integer()) {
                        match functional_eq_residual(&cache, &b, &h, &fh, prec, digits) {
                            Ok(residual) => Some(residual),
                            Err(error) => {
                                writeln!(&mut buf, "{}\t{}\t{}\t{}\terror\t\t\t\t{:.3}\tFE comparison unavailable: {}",
                                    base_case.label, cnum::DisplayFloat(&base_case.re), cnum::DisplayFloat(&base_case.im), h_label, elapsed,
                                    error.replace(['\n', '\t'], " "))
                                    .map_err(|e| format!("buffer edge row: {e}"))?;
                                base_err += 1;
                                continue;
                            }
                        }
                    } else {
                        None
                    };
                    let feq_str = match &feq {
                        Some(r) => format!("{:.2e}", cnum::DisplayFloat(r)),
                        None => "-".to_string(),
                    };
                    let feq_bad = feq.is_some_and(|r| r > cnum::epsilon(digits, prec));
                    let status = if feq_bad { "feq_fail" } else { "ok" };
                    let (re_s, im_s) = cnum::format_complex(&fh, output_digits);
                    use std::io::Write as _;
                    writeln!(
                        &mut buf,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t",
                        base_case.label,
                        cnum::DisplayFloat(&base_case.re),
                        cnum::DisplayFloat(&base_case.im),
                        h_label,
                        status,
                        re_s,
                        im_s,
                        feq_str,
                        elapsed
                    )
                    .map_err(|e| format!("buffer edge row: {e}"))?;
                    if feq_bad {
                        base_ok_feq_fail += 1;
                    } else {
                        base_ok += 1;
                    }
                }
            }
        }

        let base_elapsed = t0.elapsed().as_secs_f64();

        {
            let _guard = stdout_mutex.lock().unwrap();
            let stdout = io::stdout();
            let mut g = stdout.lock();
            g.write_all(&buf)
                .map_err(|e| format!("write edge rows: {e}"))?;
            g.flush().map_err(|e| format!("flush edge rows: {e}"))?;
        }
        {
            let mut s = total_stats.lock().unwrap();
            s.ok += base_ok;
            s.ok_feq_fail += base_ok_feq_fail;
            s.undef += base_undef;
            s.err += base_err;
        }
        let done = {
            let mut d = bases_done.lock().unwrap();
            *d += 1;
            *d
        };
        let wall = global_start.elapsed().as_secs_f64();
        if cnum::verbose() {
            eprintln!(
            "[{}/{}] {:<50}  cache={:<8}  ok={} feq_fail={} undef={} err={}  {:.1}s  wall={:.0}s",
            done, n_bases, base_case.label, cache.kind_label(),
            base_ok, base_ok_feq_fail, base_undef, base_err, base_elapsed, wall,
        );
        }
        Ok(())
    };
    if tetration::mt::mt_enabled() {
        bases.par_iter().try_for_each(process_base)?;
    } else {
        bases.iter().try_for_each(process_base)?;
    }

    let s = total_stats.lock().unwrap();
    let defined = n_cells - s.undef;
    let any_fail = s.err > 0 || s.ok_feq_fail > 0;
    if cnum::verbose() {
        eprintln!(
        "\nDONE: {} cells in {:.0}s\n  ok={} feq_fail={} undef={} err={}\n  {:.2}% ok of {} defined cells",
        n_cells, global_start.elapsed().as_secs_f64(),
        s.ok, s.ok_feq_fail, s.undef, s.err,
        if defined == 0 { 100.0 } else { 100.0 * (s.ok as f64) / (defined as f64) },
        defined,
    );
    }
    if any_fail {
        return Err(format!(
            "{} edge cells failed; {} functional-equation checks failed",
            s.err, s.ok_feq_fail
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixtures_use_working_precision_and_actual_dispatch_boundaries() {
        for digits in [50, 70, 1000] {
            let prec = cnum::digits_to_bits(digits);
            let bases = all_bases(prec);
            assert_eq!(bases[0].re, cnum::eta_upper(prec));
            let e = bases
                .iter()
                .find(|b| b.label == "real-gt-eta: b=e")
                .unwrap();
            assert_eq!(e.re, Float::with_val_64(prec, 1).exp());
            for (index, lambda, interior) in [(7, "0.949999", true), (8, "0.950001", false)] {
                let b = Complex::with_val_64(prec, (&bases[index].re, &bases[index].im));
                let region = regions::classify(&b, prec).unwrap();
                let actual = match region {
                    regions::Region::ShellThronInterior(fp) if interior => fp.lambda_abs,
                    regions::Region::ShellThronBoundary(fp) if !interior => fp.lambda_abs,
                    other => panic!("wrong threshold fixture: {}", other.name()),
                };
                assert!((actual - cnum::decimal(lambda, prec)).abs() < cnum::epsilon(digits, prec));
            }
            let decimal = bases
                .iter()
                .find(|b| b.label == "real-1eta: b=1.3 (Schröder)")
                .unwrap();
            assert_eq!(decimal.re, cnum::decimal("1.3", prec));
        }
    }

    #[test]
    fn functional_equation_diagnostics_keep_thousand_digit_residuals() {
        let digits = 1000;
        let prec = cnum::digits_to_bits(digits);
        let b = cnum::parse_complex("0.5", "0", prec).unwrap();
        let delta = cnum::epsilon(digits, prec);
        let perturbed = Complex::with_val_64(prec, Float::with_val_64(prec, 1) + &delta);
        let residual = functional_eq_residual(
            &BaseCache::DispatchFallback,
            &b,
            &cnum::zero(prec),
            &perturbed,
            prec,
            digits,
        )
        .unwrap();
        assert!(residual > delta.clone() * cnum::decimal("0.3", prec));
        assert!(residual < delta * cnum::decimal("0.4", prec));
        let nan = Complex::with_val_64(prec, Float::with_val_64(prec, rug::float::Special::Nan));
        assert!(functional_eq_residual(
            &BaseCache::SpecialBase,
            &b,
            &cnum::zero(prec),
            &nan,
            prec,
            digits
        )
        .is_err());
    }

    #[test]
    fn strictly_attracting_boundary_uses_the_regular_cache() {
        let digits = 50;
        let prec = cnum::digits_to_bits(digits);
        for (re, im) in [
            ("1.444666", "0"),
            ("0.0665", "0"),
            ("0.0653281554868594", "0.025"),
        ] {
            let base = cnum::parse_complex(re, im, prec).unwrap();
            let cache = build_cache(&base, prec, digits);
            assert!(
                matches!(&cache, BaseCache::SchroderCached(state) if state.inverse_radius.is_some())
            );
            for (hr, hi) in [
                ("0", "0"),
                ("3", "0"),
                ("-1", "0"),
                ("0.5", "0.25"),
                ("-0.5", "0.25"),
            ] {
                let height = cnum::parse_complex(hr, hi, prec).unwrap();
                let actual = eval_cell(&cache, &base, &height, prec, digits).unwrap();
                assert_eq!(
                    actual,
                    dispatch::tetrate(&base, &height, prec, digits).unwrap()
                );
                assert!(
                    functional_eq_residual(&cache, &base, &height, &actual, prec, digits).unwrap()
                        < cnum::epsilon(digits, prec)
                );
            }
        }
    }

    #[test]
    fn caches_preserve_dispatch_and_degenerate_domains() {
        let digits = 50;
        let prec = cnum::digits_to_bits(digits);
        for (re, im) in [("1.2", "-0.1"), ("1.5", "0"), ("-2", "0"), ("0.05", "0")] {
            let base = cnum::parse_complex(re, im, prec).unwrap();
            assert!(matches!(
                build_cache(&base, prec, digits),
                BaseCache::DispatchFallback
            ));
        }
        let base = cnum::parse_complex("1.2", "0", prec).unwrap();
        let height = cnum::parse_complex("0.4", "0.2", prec).unwrap();
        assert_eq!(
            eval_cell(
                &build_cache(&base, prec, digits),
                &base,
                &height,
                prec,
                digits
            )
            .unwrap(),
            dispatch::tetrate(&base, &height, prec, digits).unwrap()
        );
        let negative = cnum::parse_complex("-1e60", "0", prec).unwrap();
        assert!(domain_undefined(&cnum::one(prec), &negative).is_none());
        assert!(domain_undefined(&Complex::with_val_64(prec, 2), &negative).is_some());
        assert_eq!(
            eval_cell(
                &BaseCache::SpecialBase,
                &cnum::one(prec),
                &negative,
                prec,
                digits
            )
            .unwrap(),
            cnum::one(prec)
        );
    }
}
