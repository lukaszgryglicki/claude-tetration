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
//! |F(h+1) − b^F(h)| / max(|F(h+1)|, 1) < 1e-6 is checked automatically.
//!
//! Usage:
//!   cargo run --release --example edge_runner [digits]
//!   defaults: digits=20

use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;
use rug::{ops::Pow, Complex, Float};
use tetration::{cnum, dispatch, integer_height, kouznetsov, regions, schroder};

// ──────────────────────────────────────────────────────────────────────────────
// Test-case definitions
// ──────────────────────────────────────────────────────────────────────────────

/// A base to probe, with a human-readable category label.
#[derive(Clone)]
struct BaseCase {
    re: f64,
    im: f64,
    label: &'static str,
}

fn bc(re: f64, im: f64, label: &'static str) -> BaseCase {
    BaseCase { re, im, label }
}

fn all_bases() -> Vec<BaseCase> {
    let eta: f64 = std::f64::consts::E.powf(1.0 / std::f64::consts::E); // e^(1/e) ≈ 1.44467

    let mut v: Vec<BaseCase> = Vec::new();

    // 1. Shell-Thron parabolic boundary ─────────────────────────────────────
    v.push(bc(eta,               0.0, "ST/boundary: b=η exact"));
    v.push(bc(eta + 1e-5,        0.0, "ST/boundary: b=η+1e-5 (ε outside)"));
    v.push(bc(eta + 1e-3,        0.0, "ST/boundary: b=η+1e-3 (ε outside)"));
    v.push(bc(eta + 0.01,        0.0, "ST/boundary: b=η+0.01 (outside)"));
    v.push(bc(eta - 1e-5,        0.0, "ST/boundary: b=η-1e-5 (ε inside, Schröder)"));
    v.push(bc(eta - 1e-3,        0.0, "ST/boundary: b=η-1e-3 (ε inside)"));
    v.push(bc(eta - 0.01,        0.0, "ST/boundary: b=η-0.01 (inside)"));
    // Complex bases right on |λ|=1 circle:
    v.push(bc(0.1,  0.995, "ST/boundary: complex |λ|≈1 (0.1+0.995i)"));
    v.push(bc(-0.5, 0.87,  "ST/boundary: complex |λ|≈1 (-0.5+0.87i)"));

    // 2. Dispatch threshold |λ|=0.95 (Schröder/Kouznetsov switch) ───────────
    // b ≈ 1.37 has |λ| ≈ 0.94 (inside Schröder band), b ≈ 1.39 ≈ 0.96
    v.push(bc(1.37, 0.0, "dispatch-threshold: |λ|≈0.94 (Schröder side)"));
    v.push(bc(1.39, 0.0, "dispatch-threshold: |λ|≈0.96 (Kouznetsov side)"));
    v.push(bc(1.40, 0.0, "dispatch-threshold: b=1.40 (mid-range)"));
    v.push(bc(1.43, 0.0, "dispatch-threshold: b=1.43 (near η)"));

    // 3. Real axis ────────────────────────────────────────────────────────────
    // Negative integer bases
    v.push(bc(-1.0,  0.0, "real-neg-int: b=−1"));
    v.push(bc(-2.0,  0.0, "real-neg-int: b=−2"));
    v.push(bc(-3.0,  0.0, "real-neg-int: b=−3"));
    v.push(bc(-4.0,  0.0, "real-neg-int: b=−4"));
    v.push(bc(-5.0,  0.0, "real-neg-int: b=−5"));
    // Negative non-integer
    v.push(bc(-0.5,  0.0, "real-neg: b=−0.5"));
    v.push(bc(-1.5,  0.0, "real-neg: b=−1.5"));
    v.push(bc(-2.5,  0.0, "real-neg: b=−2.5"));
    v.push(bc(-3.5,  0.0, "real-neg: b=−3.5"));
    // (0, 1)
    v.push(bc(0.1,   0.0, "real-01: b=0.1"));
    v.push(bc(0.5,   0.0, "real-01: b=0.5"));
    v.push(bc(0.9,   0.0, "real-01: b=0.9"));
    v.push(bc(0.99,  0.0, "real-01: b=0.99"));
    // (1, η) — inside Shell-Thron
    v.push(bc(1.1,   0.0, "real-1eta: b=1.1 (Schröder)"));
    v.push(bc(1.2,   0.0, "real-1eta: b=1.2 (Schröder)"));
    v.push(bc(1.3,   0.0, "real-1eta: b=1.3 (Schröder)"));
    v.push(bc(1.4,   0.0, "real-1eta: b=1.4 (Schröder)"));
    // Above η
    v.push(bc(1.5,   0.0, "real-gt-eta: b=1.5"));
    v.push(bc(1.6,   0.0, "real-gt-eta: b=1.6"));
    v.push(bc(2.0,   0.0, "real-gt-eta: b=2"));
    v.push(bc(std::f64::consts::E, 0.0, "real-gt-eta: b=e"));
    v.push(bc(3.0,   0.0, "real-gt-eta: b=3"));
    v.push(bc(5.0,   0.0, "real-gt-eta: b=5"));
    v.push(bc(10.0,  0.0, "real-gt-eta: b=10"));

    // 4. Imaginary axis ───────────────────────────────────────────────────────
    // Positive imaginary integer
    v.push(bc(0.0,  1.0,  "imag-int: b=i (|b|=1)"));
    v.push(bc(0.0,  2.0,  "imag-int: b=2i"));
    v.push(bc(0.0,  3.0,  "imag-int: b=3i"));
    v.push(bc(0.0,  5.0,  "imag-int: b=5i"));
    // Positive imaginary non-integer
    v.push(bc(0.0,  0.5,  "imag: b=0.5i (inside ST)"));
    v.push(bc(0.0,  1.2,  "imag: b=1.2i (near ST boundary)"));
    v.push(bc(0.0,  1.5,  "imag: b=1.5i"));
    // Negative imaginary (Schwarz path)
    v.push(bc(0.0, -1.0,  "imag-neg-int: b=−i (Schwarz)"));
    v.push(bc(0.0, -2.0,  "imag-neg-int: b=−2i (Schwarz)"));
    v.push(bc(0.0, -3.0,  "imag-neg-int: b=−3i (Schwarz)"));
    v.push(bc(0.0, -0.5,  "imag-neg: b=−0.5i (Schwarz)"));
    v.push(bc(0.0, -1.5,  "imag-neg: b=−1.5i (Schwarz)"));

    // 5. Re = Im diagonal ─────────────────────────────────────────────────────
    v.push(bc( 0.5,  0.5, "diag-pos: b=0.5+0.5i"));
    v.push(bc( 1.0,  1.0, "diag-pos: b=1+i"));
    v.push(bc( 2.0,  2.0, "diag-pos: b=2+2i"));
    v.push(bc( 3.0,  3.0, "diag-pos: b=3+3i"));
    v.push(bc(-1.0, -1.0, "diag-neg: b=−1−i"));
    v.push(bc(-2.0, -2.0, "diag-neg: b=−2−2i (Schwarz)"));
    v.push(bc( 1.0, -1.0, "anti-diag: b=1−i"));
    v.push(bc( 2.0, -2.0, "anti-diag: b=2−2i (Schwarz)"));
    v.push(bc(-1.0,  1.0, "anti-diag: b=−1+i"));
    v.push(bc(-2.0,  2.0, "anti-diag: b=−2+2i"));

    // 6. Near-zero bases ──────────────────────────────────────────────────────
    v.push(bc(1e-6,  0.0,  "near-zero: b=1e-6+0i"));
    v.push(bc(0.0,   1e-6, "near-zero: b=0+1e-6i"));
    v.push(bc(1e-4,  1e-4, "near-zero: b=1e-4+1e-4i"));
    v.push(bc(-1e-4, 0.0,  "near-zero: b=−1e-4+0i"));

    // 7. Large-magnitude bases ────────────────────────────────────────────────
    v.push(bc( 100.0,    0.0,  "large: b=100"));
    v.push(bc(-100.0,    0.0,  "large: b=−100"));
    v.push(bc(   0.0,  100.0,  "large: b=100i"));
    v.push(bc(   0.0, -100.0,  "large: b=−100i (Schwarz)"));
    v.push(bc( 100.0,  100.0,  "large: b=100+100i"));
    v.push(bc( 100.0, -100.0,  "large: b=100−100i (Schwarz)"));
    v.push(bc(-100.0,  100.0,  "large: b=−100+100i"));
    v.push(bc(-100.0, -100.0,  "large: b=−100−100i (Schwarz)"));

    // 8. Mixed medium ─────────────────────────────────────────────────────────
    v.push(bc( 1.2,   3.5, "complex: b=1.2+3.5i"));
    v.push(bc(-1.2,   1.2, "complex: b=−1.2+1.2i"));
    v.push(bc(-3.6,   0.4, "complex: b=−3.6+0.4i"));

    v
}

/// Heights to probe for each base.
fn all_heights() -> Vec<(f64, f64, &'static str)> {
    vec![
        // Integer heights
        ( 0.0,  0.0, "h=0"),
        ( 1.0,  0.0, "h=1"),
        ( 2.0,  0.0, "h=2"),
        ( 3.0,  0.0, "h=3"),
        (-1.0,  0.0, "h=−1 (F=0)"),
        (-2.0,  0.0, "h=−2 (undef)"),
        // Non-integer real
        ( 0.5,  0.0, "h=0.5"),
        (-0.5,  0.0, "h=−0.5"),
        ( 0.3,  0.0, "h=0.3"),
        ( 0.7,  0.0, "h=0.7"),
        ( 1.5,  0.0, "h=1.5"),
        ( 2.5,  0.0, "h=2.5"),
        // Complex heights
        ( 0.5,  0.5, "h=0.5+0.5i"),
        ( 0.5, -0.5, "h=0.5−0.5i"),
        ( 0.3,  0.7, "h=0.3+0.7i"),
        ( 1.0,  1.0, "h=1+i"),
        ( 2.0,  1.0, "h=2+i"),
        // Pure imaginary heights
        ( 0.0,  0.5, "h=0.5i"),
        ( 0.0,  1.0, "h=i"),
        ( 0.0, -0.5, "h=−0.5i"),
        // Large heights
        ( 5.0,  0.0, "h=5"),
        (10.0,  0.0, "h=10"),
        ( 0.0,  5.0, "h=5i"),
        ( 5.0,  5.0, "h=5+5i"),
        // Near-zero heights
        (1e-6,  0.0, "h=1e-6"),
        ( 0.0,  1e-6,"h=1e-6i"),
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
    SetupError(String),
    SetupErrorFallback(String),
}

impl BaseCache {
    fn kind_label(&self) -> &'static str {
        match self {
            BaseCache::SpecialBase         => "special",
            BaseCache::DispatchFallback    => "dispatch",
            BaseCache::SchroderCached(_)   => "schr",
            BaseCache::KouznetsovCached(_) => "kouz",
            BaseCache::SetupError(_)       => "err",
            BaseCache::SetupErrorFallback(_)=> "fallback",
        }
    }
}

fn build_cache(b: &Complex, prec: u32, digits: u64) -> BaseCache {
    if cnum::is_zero(b) || cnum::is_one(b) {
        return BaseCache::SpecialBase;
    }
    let region = match regions::classify(b, prec) {
        Ok(r) => r,
        Err(_) => return BaseCache::DispatchFallback,
    };
    match &region {
        regions::Region::BaseZero | regions::Region::BaseOne => BaseCache::SpecialBase,
        regions::Region::ShellThronInterior(d) => {
            match schroder::setup_schroder(b, d, prec) {
                Ok(s) => BaseCache::SchroderCached(s),
                Err(e) => BaseCache::SetupError(format!("Schröder setup: {}", e)),
            }
        }
        regions::Region::ShellThronBoundary(d)
        | regions::Region::OutsideShellThronRealPositive(d)
        | regions::Region::OutsideShellThronGeneral(d) => {
            match kouznetsov::setup_kouznetsov(b, d, prec, digits) {
                Ok(s) => BaseCache::KouznetsovCached(s),
                Err(e) => BaseCache::SetupErrorFallback(format!("Kouznetsov setup: {}", e)),
            }
        }
    }
}

fn eval_cell(
    cache: &BaseCache,
    b: &Complex,
    h: &Complex,
    prec: u32,
    digits: u64,
) -> Result<Complex, String> {
    if let Some(n) = cnum::as_integer(h) {
        return integer_height::tetrate_integer(b, n, prec);
    }
    match cache {
        BaseCache::SpecialBase
        | BaseCache::DispatchFallback
        | BaseCache::SetupErrorFallback(_) => dispatch::tetrate(b, h, prec, digits),
        BaseCache::SchroderCached(state)   => schroder::eval_schroder(state, h),
        BaseCache::KouznetsovCached(state) => kouznetsov::eval_kouznetsov(state, b, h),
        BaseCache::SetupError(e)           => Err(e.clone()),
    }
}

fn domain_undefined(b: &Complex, h: &Complex) -> Option<&'static str> {
    if cnum::is_zero(b) {
        match cnum::as_integer(h) {
            Some(n) if n >= 0 => None,
            _ => Some("b=0: only non-negative integer h defined"),
        }
    } else if let Some(n) = cnum::as_integer(h) {
        if n <= -2 {
            Some("h≤−2 integer: log_b(0) chain undefined")
        } else {
            None
        }
    } else {
        None
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Functional-equation check
// ──────────────────────────────────────────────────────────────────────────────

/// Compute |F(h+1) − b^F(h)| / max(|F(h+1)|, 1).
/// Returns None if either eval fails (not a functional-eq failure).
fn functional_eq_residual(
    cache: &BaseCache,
    b: &Complex,
    h: &Complex,
    fh: &Complex,
    prec: u32,
    digits: u64,
) -> Option<f64> {
    let h1 = Complex::with_val(prec, h + 1u32);
    let fh1 = eval_cell(cache, b, &h1, prec, digits).ok()?;
    // b^F(h)
    let b_to_fh = Complex::with_val(prec, b.pow(fh));
    let diff = Complex::with_val(prec, &fh1 - &b_to_fh);
    let diff_abs = Float::with_val(prec, diff.abs_ref()).to_f64();
    let fh1_abs  = Float::with_val(prec, fh1.abs_ref()).to_f64();
    let denom = fh1_abs.max(1.0);
    Some(diff_abs / denom)
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
    let args: Vec<String> = std::env::args().collect();
    let digits: u64 = if args.len() >= 2 { args[1].parse().expect("bad digits") } else { 20 };
    let prec = cnum::digits_to_bits(digits);

    let bases   = all_bases();
    let heights = all_heights();

    let n_bases   = bases.len();
    let n_heights = heights.len();
    let n_cells   = n_bases * n_heights;

    eprintln!(
        "edge_runner: digits={}, bases={}, heights={}, cells={}, using all CPU cores via rayon",
        digits, n_bases, n_heights, n_cells
    );

    let stdout_mutex: Arc<Mutex<()>> = Arc::new(Mutex::new(()));
    let total_stats: Arc<Mutex<Stats>> = Arc::new(Mutex::new(Stats::default()));
    let bases_done: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let global_start = Instant::now();

    // Print TSV header
    {
        let stdout = io::stdout();
        let mut g = stdout.lock();
        writeln!(g, "category\tb_re\tb_im\th_label\tstatus\tresult_re\tresult_im\tfeq_residual\telapsed_secs\terror").ok();
        g.flush().ok();
    }

    bases.par_iter().for_each(|base_case| {
        let b = Complex::with_val(prec, (
            Float::with_val(prec, base_case.re),
            Float::with_val(prec, base_case.im),
        ));
        let t0 = Instant::now();
        let cache = build_cache(&b, prec, digits);

        let mut buf: Vec<u8> = Vec::with_capacity(n_heights * 120);
        let mut base_ok = 0usize;
        let mut base_ok_feq_fail = 0usize;
        let mut base_undef = 0usize;
        let mut base_err = 0usize;

        for &(h_re, h_im, h_label) in &heights {
            let h = Complex::with_val(prec, (
                Float::with_val(prec, h_re),
                Float::with_val(prec, h_im),
            ));

            if let Some(reason) = domain_undefined(&b, &h) {
                use std::io::Write as _;
                let _ = writeln!(&mut buf,
                    "{}\t{}\t{}\t{}\tundef\t\t\t\t0.000\t{}",
                    base_case.label, base_case.re, base_case.im, h_label, reason);
                base_undef += 1;
                continue;
            }

            let ct0 = Instant::now();
            let result = eval_cell(&cache, &b, &h, prec, digits);
            let elapsed = ct0.elapsed().as_secs_f64();

            match result {
                Err(e) => {
                    use std::io::Write as _;
                    let one = e.lines().next().unwrap_or("").chars().take(200).collect::<String>();
                    let _ = writeln!(&mut buf,
                        "{}\t{}\t{}\t{}\terror\t\t\t\t{:.3}\t{}",
                        base_case.label, base_case.re, base_case.im, h_label, elapsed, one);
                    base_err += 1;
                }
                Ok(fh) => {
                    // Functional-equation check for non-integer heights.
                    let feq = if cnum::as_integer(&h).is_none() {
                        functional_eq_residual(&cache, &b, &h, &fh, prec, digits)
                    } else {
                        None
                    };
                    let feq_str = match feq {
                        Some(r) => format!("{:.2e}", r),
                        None    => "—".to_string(),
                    };
                    let feq_bad = feq.map(|r| r > 1e-6).unwrap_or(false);
                    let status = if feq_bad { "ok_feq_fail" } else { "ok" };
                    let (re_s, im_s) = cnum::format_complex(&fh, digits as usize);
                    use std::io::Write as _;
                    let _ = writeln!(&mut buf,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t",
                        base_case.label, base_case.re, base_case.im, h_label,
                        status, re_s, im_s, feq_str, elapsed);
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
            g.write_all(&buf).ok();
            g.flush().ok();
        }
        {
            let mut s = total_stats.lock().unwrap();
            s.ok           += base_ok;
            s.ok_feq_fail  += base_ok_feq_fail;
            s.undef        += base_undef;
            s.err          += base_err;
        }
        let done = {
            let mut d = bases_done.lock().unwrap();
            *d += 1;
            *d
        };
        let wall = global_start.elapsed().as_secs_f64();
        eprintln!(
            "[{}/{}] {:<50}  cache={:<8}  ok={} feq_fail={} undef={} err={}  {:.1}s  wall={:.0}s",
            done, n_bases, base_case.label, cache.kind_label(),
            base_ok, base_ok_feq_fail, base_undef, base_err, base_elapsed, wall,
        );
    });

    let s = total_stats.lock().unwrap();
    let defined = n_cells - s.undef;
    let any_fail = s.err > 0 || s.ok_feq_fail > 0;
    eprintln!(
        "\nDONE: {} cells in {:.0}s\n  ok={} ok_feq_fail={} undef={} err={}\n  {:.2}% ok of {} defined cells",
        n_cells, global_start.elapsed().as_secs_f64(),
        s.ok, s.ok_feq_fail, s.undef, s.err,
        if defined == 0 { 100.0 } else { 100.0 * (s.ok as f64) / (defined as f64) },
        defined,
    );
    if any_fail {
        std::process::exit(1);
    }
}
