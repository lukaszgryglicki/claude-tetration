//! 19^4 grid runner: verify that every combination of (b_re, b_im, h_re, h_im)
//! on a regular grid returns a value at the requested precision.
//!
//! The hot loop amortises per-base setup (Kouznetsov samples + normalization
//! shift) across all heights for a given base. Without this caching, every
//! cell would pay the full Newton-Kantorovich Cauchy iteration cost; with it,
//! one base × N heights costs `setup + N · eval` where `eval` is one Cauchy
//! integral plus a few b^· iterations.
//!
//! Usage:
//!   grid_runner <digits> [step] [b_re_lo] [b_re_hi] [b_im_lo] [b_im_hi] \
//!               [h_re_lo] [h_re_hi] [h_im_lo] [h_im_hi]
//!
//!   defaults: step=0.4, all ranges = [-3.6, 3.6]
//!
//! Output (stdout): TSV with columns
//!   b_re  b_im  h_re  h_im  status  result_re  result_im  elapsed_secs  error
//!
//! status ∈ {ok, undef, error}.
//!   * `ok`      — algorithm produced a value at requested precision.
//!   * `undef`   — cell lies in a mathematically-undefined domain
//!     (b=0 with non-integer height; h ≤ −2 integer where
//!     F(h) would require log_b(0)). Not an algorithm failure.
//!   * `error`   — algorithm failed to produce a value despite the cell
//!     being well-defined; the error column carries the
//!     one-line reason.
//!
//! Progress (stderr): one line per base summarizing OK / ERROR counts and
//! per-base wall time.

use std::env;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;
use rug::{Complex, Float};
use tetration::{cnum, dispatch, integer_height, kouznetsov, regions, schroder};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: grid_runner <digits> [step] [b_re_lo b_re_hi b_im_lo b_im_hi h_re_lo h_re_hi h_im_lo h_im_hi]"
        );
        std::process::exit(2);
    }

    let digits: u64 = args[1].parse().expect("digits must be a positive integer");
    let step: f64 = if args.len() >= 3 { args[2].parse().expect("bad step") } else { 0.4 };
    let parse_or = |idx: usize, default: f64| -> f64 {
        if args.len() > idx { args[idx].parse().unwrap_or(default) } else { default }
    };
    let b_re_lo = parse_or(3, -3.6);
    let b_re_hi = parse_or(4, 3.6);
    let b_im_lo = parse_or(5, -3.6);
    let b_im_hi = parse_or(6, 3.6);
    let h_re_lo = parse_or(7, -3.6);
    let h_re_hi = parse_or(8, 3.6);
    let h_im_lo = parse_or(9, -3.6);
    let h_im_hi = parse_or(10, 3.6);

    let prec = cnum::digits_to_bits(digits);
    let b_re_axis = build_axis(b_re_lo, b_re_hi, step);
    let b_im_axis = build_axis(b_im_lo, b_im_hi, step);
    let h_re_axis = build_axis(h_re_lo, h_re_hi, step);
    let h_im_axis = build_axis(h_im_lo, h_im_hi, step);
    let n_dec = axis_decimals(step);

    let total_bases = b_re_axis.len() * b_im_axis.len();
    let total_heights = h_re_axis.len() * h_im_axis.len();
    let total_cells = total_bases * total_heights;

    eprintln!(
        "grid_runner: digits={}, step={}, bases={}×{}={}, heights={}×{}={}, total={} cells",
        digits,
        step,
        b_re_axis.len(),
        b_im_axis.len(),
        total_bases,
        h_re_axis.len(),
        h_im_axis.len(),
        total_heights,
        total_cells
    );

    // Per-base buffers are pushed through a Mutex-guarded shared stdout sink;
    // we hold the lock only briefly per base, not per cell.
    let out_mutex: Arc<Mutex<()>> = Arc::new(Mutex::new(()));
    {
        let stdout = io::stdout();
        let mut g = stdout.lock();
        writeln!(
            g,
            "b_re\tb_im\th_re\th_im\tstatus\tresult_re\tresult_im\telapsed_secs\terror"
        )
        .ok();
        g.flush().ok();
    }

    // Build the full base list so rayon can distribute across cores. Each
    // base is independent — its cache is per-base — so we parallelize the
    // OUTER loop and process each base's heights serially in the worker.
    let bases: Vec<(f64, f64)> = b_re_axis
        .iter()
        .flat_map(|&r| b_im_axis.iter().map(move |&i| (r, i)))
        .collect();

    let total_ok = Arc::new(Mutex::new(0usize));
    let total_undef = Arc::new(Mutex::new(0usize));
    let total_err = Arc::new(Mutex::new(0usize));
    let bases_done = Arc::new(Mutex::new(0usize));
    let grid_start = Instant::now();

    bases.par_iter().for_each(|&(b_re, b_im)| {
        let b = Complex::with_val(prec, (Float::with_val(prec, b_re), Float::with_val(prec, b_im)));
        let b_re_s = fmt_axis(b_re, n_dec);
        let b_im_s = fmt_axis(b_im, n_dec);
        let base_t0 = Instant::now();
        let cache = build_cache(&b, prec, digits);
        let mut base_ok: usize = 0;
        let mut base_undef: usize = 0;
        let mut base_err: usize = 0;
        // Buffer this base's rows; flush once at the end so output stays grouped.
        let mut buf: Vec<u8> = Vec::with_capacity(h_re_axis.len() * h_im_axis.len() * 80);
        for &h_re in &h_re_axis {
            let h_re_s = fmt_axis(h_re, n_dec);
            for &h_im in &h_im_axis {
                let h_im_s = fmt_axis(h_im, n_dec);
                let h = Complex::with_val(prec, (Float::with_val(prec, h_re), Float::with_val(prec, h_im)));
                // Domain pre-check — these are not algorithm failures, they
                // are mathematically-undefined cells. Tag them with status
                // `undef` so they don't pollute the algorithm-error count.
                if let Some(reason) = domain_undefined(&b, &h) {
                    use std::io::Write as _;
                    let _ = writeln!(
                        &mut buf,
                        "{}\t{}\t{}\t{}\tundef\t\t\t0.0000\t{}",
                        b_re_s, b_im_s, h_re_s, h_im_s, reason
                    );
                    base_undef += 1;
                    continue;
                }
                let cell_t0 = Instant::now();
                let result = eval_cell(&cache, &b, &h, prec, digits);
                let elapsed = cell_t0.elapsed().as_secs_f64();
                match result {
                    Ok(v) => {
                        let (re_str, im_str) = cnum::format_complex(&v, digits as usize);
                        use std::io::Write as _;
                        let _ = writeln!(
                            &mut buf,
                            "{}\t{}\t{}\t{}\tok\t{}\t{}\t{:.4}\t",
                            b_re_s, b_im_s, h_re_s, h_im_s, re_str, im_str, elapsed
                        );
                        base_ok += 1;
                    }
                    Err(why) => {
                        let one = first_line(&why);
                        use std::io::Write as _;
                        let _ = writeln!(
                            &mut buf,
                            "{}\t{}\t{}\t{}\terror\t\t\t{:.4}\t{}",
                            b_re_s, b_im_s, h_re_s, h_im_s, elapsed, one
                        );
                        base_err += 1;
                    }
                }
            }
        }

        // Atomically flush this base's rows and update counters.
        {
            let _guard = out_mutex.lock().unwrap();
            let stdout = io::stdout();
            let mut g = stdout.lock();
            g.write_all(&buf).ok();
            g.flush().ok();
        }
        {
            let mut t_ok = total_ok.lock().unwrap();
            *t_ok += base_ok;
        }
        {
            let mut t_undef = total_undef.lock().unwrap();
            *t_undef += base_undef;
        }
        {
            let mut t_err = total_err.lock().unwrap();
            *t_err += base_err;
        }
        let done = {
            let mut d = bases_done.lock().unwrap();
            *d += 1;
            *d
        };
        let base_elapsed = base_t0.elapsed().as_secs_f64();
        let total_elapsed = grid_start.elapsed().as_secs_f64();
        let (snap_ok, snap_undef, snap_err) = (
            *total_ok.lock().unwrap(),
            *total_undef.lock().unwrap(),
            *total_err.lock().unwrap(),
        );
        eprintln!(
            "[{}/{}] b=({:>+6},{:>+6}i)  cache={:<10} ok={} undef={} err={}  base_t={:.1}s  wall={:.0}s  ok_so_far={} undef_so_far={} err_so_far={}",
            done, total_bases, b_re_s, b_im_s, cache.kind_label(),
            base_ok, base_undef, base_err, base_elapsed, total_elapsed,
            snap_ok, snap_undef, snap_err,
        );
    });

    let final_ok = *total_ok.lock().unwrap();
    let final_undef = *total_undef.lock().unwrap();
    let final_err = *total_err.lock().unwrap();
    let defined = total_cells - final_undef;
    eprintln!(
        "DONE: {} cells in {:.0}s — ok={} undef={} err={} ({:.2}% ok of {} defined)",
        total_cells,
        grid_start.elapsed().as_secs_f64(),
        final_ok,
        final_undef,
        final_err,
        if defined == 0 { 100.0 } else { 100.0 * (final_ok as f64) / (defined as f64) },
        defined,
    );
    if final_err > 0 {
        std::process::exit(1);
    }
}

/// Pre-flight check for cells that are mathematically undefined regardless of
/// algorithm. These should not be reported as algorithm failures.
///
/// Two known undefined-domain cases:
///   1. b = 0 with non-non-negative-integer height — 0^^z requires the
///      alternation `0,1,0,1,…` for non-negative integer z; for fractional
///      or negative z it has no consistent definition.
///   2. h ≤ −2 integer — F(−n) = log_b(F(−n+1)). At n=1: F(−1) = log_b(F(0))
///      = log_b(1) = 0. At n=2: F(−2) = log_b(F(−1)) = log_b(0) = −∞.
///      Beyond −1, the iterated logarithm chains through log(0), undefined.
fn domain_undefined(b: &Complex, h: &Complex) -> Option<&'static str> {
    if cnum::is_zero(b) {
        // b=0 only defined for non-negative integer h.
        match cnum::as_integer(h) {
            Some(n) if n >= 0 => None,
            _ => Some("b=0: tetration only defined for non-negative integer heights"),
        }
    } else if let Some(n) = cnum::as_integer(h) {
        if n <= -2 {
            Some("integer height ≤ −2: requires log_b(0) chain, undefined")
        } else {
            None
        }
    } else {
        None
    }
}

fn build_axis(lo: f64, hi: f64, step: f64) -> Vec<f64> {
    let mut v = Vec::new();
    let n = ((hi - lo) / step).round() as i64 + 1;
    for i in 0..n {
        // Build via i*step + lo so each axis value is a single multiplication
        // away from clean. Then snap to step-multiple to wash away the FP
        // accumulation drift that produces e.g. 2.0000000000000004 from
        // -3.6 + 14*0.4.
        let raw = lo + (i as f64) * step;
        let snapped = (raw / step).round() * step;
        v.push(if snapped.abs() < 1e-12 { 0.0 } else { snapped });
    }
    v
}

/// Number of decimal places needed to faithfully display values on a `step`-
/// spaced axis. step=0.4 → 1 decimal; step=0.04 → 2; step=1 → 0.
fn axis_decimals(step: f64) -> usize {
    if step >= 1.0 {
        0
    } else {
        ((-step.log10()).ceil() as usize).min(10)
    }
}

/// Format an axis value as a clean decimal string (no `2.4000000000000004`
/// FP noise). Padding to `n_decimals` then trimming trailing zeros gives
/// "2.4" / "0" / "-3" cleanly.
fn fmt_axis(v: f64, n_decimals: usize) -> String {
    let s = format!("{:.*}", n_decimals + 2, v);
    // Trim trailing zeros, then a trailing decimal point if exposed.
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() || trimmed == "-" {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").chars().take(200).collect()
}

/// Per-base cached state. The expensive piece is the Kouznetsov state
/// (Newton-Kantorovich Cauchy iteration → samples + normalization shift);
/// other paths fall back to the existing per-call dispatcher.
enum BaseCache {
    /// b = 0 or b = 1 — handled by the dispatcher per cell (cheap).
    SpecialBase,
    /// Region classification failed (e.g. parabolic boundary at exact |λ|=1).
    /// Fall through to dispatcher per cell — it will produce a clean error.
    DispatchFallback,
    /// Schröder regular tetration with σ̃ Taylor coefficients cached. The
    /// O(N²) build_series happens once per base; per-cell evaluation is then
    /// one O(N) Horner pass. At digits ≥ 20 this is a 10-100× speedup over
    /// per-cell dispatch for ShellThronInterior bases.
    SchroderCached(schroder::SchroderState),
    /// Outside Shell-Thron and Schröder doesn't reach a probe height. Use
    /// the cached Kouznetsov state for all heights in this base.
    KouznetsovCached(kouznetsov::KouznetsovState),
    /// Per-base setup itself failed (e.g. fixed-point pair not openable for
    /// this complex base by W_k branch search). Every cell errors out with
    /// the same message. Kept as a separate variant so we can label it.
    #[allow(dead_code)]
    SetupError(String),
    /// Kouznetsov setup failed — fall through to dispatch per cell. Dispatch
    /// will try Schröder first which may cover a subset of heights.
    SetupErrorFallback(#[allow(dead_code)] String),
}

impl BaseCache {
    fn kind_label(&self) -> &'static str {
        match self {
            BaseCache::SpecialBase => "special",
            BaseCache::DispatchFallback => "dispatch",
            BaseCache::SchroderCached(_) => "schr",
            BaseCache::KouznetsovCached(_) => "kouz",
            BaseCache::SetupError(_) => "err",
            BaseCache::SetupErrorFallback(_) => "fallback",
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
            // Cache the σ̃ Taylor build so all heights in this base reuse it.
            // If setup fails (e.g. σ̃-shift orbit hits a singularity), every
            // cell would error with the same message; skip per-cell retry by
            // using SetupError directly.
            match schroder::setup_schroder(b, d, prec) {
                Ok(state) => BaseCache::SchroderCached(state),
                Err(e) => BaseCache::SetupError(format!("Schröder setup failed: {}", e)),
            }
        }
        regions::Region::ShellThronBoundary(d)
        | regions::Region::OutsideShellThronRealPositive(d)
        | regions::Region::OutsideShellThronGeneral(d) => {
            // ALWAYS try to set up Kouznetsov state for non-interior bases.
            // The earlier "probe Schröder at h=0.5 and use dispatch per cell"
            // strategy was unreliable: Schröder works for h near the fixed
            // point but fails for far h (where |s1·λ^h| exceeds the σ̃ Taylor
            // safe radius and the integer-shift mechanism saturates). Caching
            // Kouznetsov state up front means every height in the grid is
            // evaluated by the same cheap `eval_kouznetsov` call.
            //
            // If Kouznetsov setup fails for this base (degenerate fixed point
            // pair, parabolic |arg(λ)|≈0, etc.), fall through to dispatch
            // per cell as best-effort: dispatch will try Schröder first which
            // may still cover some heights cleanly.
            match kouznetsov::setup_kouznetsov(b, d, prec, digits) {
                Ok(state) => BaseCache::KouznetsovCached(state),
                Err(e) => BaseCache::SetupErrorFallback(format!("kouznetsov setup failed: {}", e)),
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
    // Integer heights short-circuit to direct iteration regardless of cache.
    if let Some(n) = cnum::as_integer(h) {
        return integer_height::tetrate_integer(b, n, prec);
    }
    match cache {
        BaseCache::SpecialBase
        | BaseCache::DispatchFallback
        | BaseCache::SetupErrorFallback(_) => dispatch::tetrate(b, h, prec, digits),
        BaseCache::SchroderCached(state) => schroder::eval_schroder(state, h),
        BaseCache::KouznetsovCached(state) => kouznetsov::eval_kouznetsov(state, b, h),
        BaseCache::SetupError(e) => Err(e.clone()),
    }
}
