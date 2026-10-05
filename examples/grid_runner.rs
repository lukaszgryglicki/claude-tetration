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
//!   * `ok`      — algorithm returned a finite value after its numerical checks,
//!     not an independent precision or uniqueness certificate.
//!   * `undef`   — cell lies in a mathematically-undefined domain
//!     (b=0 with non-integer height; h ≤ −2 integer where
//!     F(h) would require log_b(0)). Not an algorithm failure.
//!   * `error`   — this implementation failed; existence is not inferred.
//!
//! Progress (stderr): one line per base summarizing OK / ERROR counts and
//! per-base wall time.

use std::env;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;
use rug::{integer::IntegerExt64, Complex, Float, Integer};
use tetration::{cnum, dispatch, kouznetsov, regions, schroder};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args: Vec<String> = env::args().collect();
    let quiet = args
        .iter()
        .skip(1)
        .any(|s| matches!(s.as_str(), "--quiet" | "--silent" | "-q"));
    args.retain(|s| !matches!(s.as_str(), "--quiet" | "--silent" | "-q"));
    cnum::set_quiet(quiet);
    if !(2..=11).contains(&args.len()) {
        return Err("usage: grid_runner [--quiet] <digits> [step] [b_re_lo b_re_hi b_im_lo b_im_hi h_re_lo h_re_hi h_im_lo h_im_hi]".into());
    }
    let digits = args[1]
        .parse::<u64>()
        .map_err(|e| format!("invalid digits: {e}"))?;
    let prec = cnum::checked_input_precision(
        digits,
        &args[2..].iter().map(String::as_str).collect::<Vec<_>>(),
    )?;
    tetration::mt::init_pool()?;
    let arg =
        |idx: usize, default: &'static str| args.get(idx).map(String::as_str).unwrap_or(default);
    let step = arg(2, "0.4");
    let b_re_axis = build_axis(arg(3, "-3.6"), arg(4, "3.6"), step, prec)?;
    let b_im_axis = build_axis(arg(5, "-3.6"), arg(6, "3.6"), step, prec)?;
    let h_re_axis = build_axis(arg(7, "-3.6"), arg(8, "3.6"), step, prec)?;
    let h_im_axis = build_axis(arg(9, "-3.6"), arg(10, "3.6"), step, prec)?;
    let total_bases = b_re_axis
        .len()
        .checked_mul(b_im_axis.len())
        .ok_or("grid size overflow")?;
    let total_heights = h_re_axis
        .len()
        .checked_mul(h_im_axis.len())
        .ok_or("grid size overflow")?;
    let total_cells = total_bases
        .checked_mul(total_heights)
        .ok_or("grid size overflow")?;
    let output_digits =
        usize::try_from(digits).map_err(|_| "output precision exceeds addressable memory")?;
    std::alloc::Layout::array::<(usize, usize)>(total_bases)
        .map_err(|_| "base grid exceeds addressable memory")?;
    let integer_heights_only = h_im_axis.iter().all(|v| v.value.is_zero())
        && h_re_axis.iter().all(|v| v.value.is_integer());
    if cnum::verbose() {
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
    }

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
        .map_err(|e| format!("write grid header: {e}"))?;
        g.flush().map_err(|e| format!("flush grid header: {e}"))?;
    }

    // Build the full base list so rayon can distribute across cores. Each
    // base is independent — its cache is per-base — so we parallelize the
    // OUTER loop and process each base's heights serially in the worker.
    let bases: Vec<(usize, usize)> = (0..b_re_axis.len())
        .flat_map(|r| (0..b_im_axis.len()).map(move |i| (r, i)))
        .collect();

    let total_ok = Arc::new(Mutex::new(0usize));
    let total_undef = Arc::new(Mutex::new(0usize));
    let total_err = Arc::new(Mutex::new(0usize));
    let bases_done = Arc::new(Mutex::new(0usize));
    let grid_start = Instant::now();

    let process_base = |&(re_idx, im_idx): &(usize, usize)| -> Result<(), String> {
        let b_re = &b_re_axis[re_idx];
        let b_im = &b_im_axis[im_idx];
        let b = Complex::with_val_64(prec, (&b_re.value, &b_im.value));
        let b_re_s = &b_re.text;
        let b_im_s = &b_im.text;
        let base_t0 = Instant::now();
        let cache = if integer_heights_only {
            BaseCache::DispatchFallback
        } else {
            build_cache(&b, prec, digits)
        };
        let mut base_ok: usize = 0;
        let mut base_undef: usize = 0;
        let mut base_err: usize = 0;
        // Buffer this base's rows; flush once at the end so output stays grouped.
        let mut buf = Vec::new();
        for h_re in &h_re_axis {
            let h_re_s = &h_re.text;
            for h_im in &h_im_axis {
                let h_im_s = &h_im.text;
                let h = Complex::with_val_64(prec, (&h_re.value, &h_im.value));
                // Domain pre-check — these are not algorithm failures, they
                // are mathematically-undefined cells. Tag them with status
                // `undef` so they don't pollute the algorithm-error count.
                if let Some(reason) = domain_undefined(&b, &h) {
                    use std::io::Write as _;
                    writeln!(
                        &mut buf,
                        "{}\t{}\t{}\t{}\tundef\t\t\t0.0000\t{}",
                        b_re_s, b_im_s, h_re_s, h_im_s, reason
                    )
                    .map_err(|e| format!("buffer grid row: {e}"))?;
                    base_undef += 1;
                    continue;
                }
                let cell_t0 = Instant::now();
                let result = eval_cell(&cache, &b, &h, prec, digits).and_then(|value| {
                    let next_prec = value.real().prec_64().max(value.imag().prec_64());
                    if next_prec > prec
                        && (cnum::parse_complex(b_re_s, b_im_s, next_prec)? != b
                            || cnum::parse_complex(h_re_s, h_im_s, next_prec)? != h)
                    {
                        if cnum::verbose() {
                            eprintln!("grid: reparsing decimal coordinates at {next_prec} bits after precision refinement");
                        }
                        let (re, im) = tetration::tetrate_str(
                            &digits.to_string(),
                            b_re_s,
                            b_im_s,
                            h_re_s,
                            h_im_s,
                        )?;
                        return cnum::parse_complex(&re, &im, next_prec);
                    }
                    Ok(value)
                });
                let elapsed = cell_t0.elapsed().as_secs_f64();
                match result {
                    Ok(v) => {
                        let (re_str, im_str) = cnum::format_complex(&v, output_digits);
                        use std::io::Write as _;
                        writeln!(
                            &mut buf,
                            "{}\t{}\t{}\t{}\tok\t{}\t{}\t{:.4}\t",
                            b_re_s, b_im_s, h_re_s, h_im_s, re_str, im_str, elapsed
                        )
                        .map_err(|e| format!("buffer grid row: {e}"))?;
                        base_ok += 1;
                    }
                    Err(why) => {
                        let one = first_line(&why);
                        use std::io::Write as _;
                        writeln!(
                            &mut buf,
                            "{}\t{}\t{}\t{}\terror\t\t\t{:.4}\t{}",
                            b_re_s, b_im_s, h_re_s, h_im_s, elapsed, one
                        )
                        .map_err(|e| format!("buffer grid row: {e}"))?;
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
            g.write_all(&buf)
                .map_err(|e| format!("write grid rows: {e}"))?;
            g.flush().map_err(|e| format!("flush grid rows: {e}"))?;
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
        if cnum::verbose() {
            eprintln!(
            "[{}/{}] b=({:>+6},{:>+6}i)  cache={:<10} ok={} undef={} err={}  base_t={:.1}s  wall={:.0}s  ok_so_far={} undef_so_far={} err_so_far={}",
            done, total_bases, b_re_s, b_im_s, cache.kind_label(),
            base_ok, base_undef, base_err, base_elapsed, total_elapsed,
            snap_ok, snap_undef, snap_err,
        );
        }
        Ok(())
    };
    if tetration::mt::mt_enabled() {
        bases.par_iter().try_for_each(process_base)?;
    } else {
        bases.iter().try_for_each(process_base)?;
    }

    let final_ok = *total_ok.lock().unwrap();
    let final_undef = *total_undef.lock().unwrap();
    let final_err = *total_err.lock().unwrap();
    let defined = total_cells - final_undef;
    if cnum::verbose() {
        eprintln!(
            "DONE: {} cells in {:.0}s — ok={} undef={} err={} ({:.2}% ok of {} defined)",
            total_cells,
            grid_start.elapsed().as_secs_f64(),
            final_ok,
            final_undef,
            final_err,
            if defined == 0 {
                100.0
            } else {
                100.0 * (final_ok as f64) / (defined as f64)
            },
            defined,
        );
    }
    if final_err > 0 {
        return Err(format!("{final_err} grid cells could not be computed"));
    }
    Ok(())
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
    let integer = h.imag().is_zero() && h.real().is_integer();
    if cnum::is_one(b) {
        None
    } else if cnum::is_zero(b) && !(integer && *h.real() >= 0) {
        Some("b=0: tetration only defined for non-negative integer heights")
    } else if !cnum::is_zero(b) && integer && *h.real() <= -2 {
        Some("integer height <= -2: requires log_b(0), undefined")
    } else {
        None
    }
}

struct AxisValue {
    value: Float,
    text: String,
}

fn scaled_decimal(text: &str) -> Result<(Integer, i64), String> {
    let (mantissa, exponent) = text.split_once(['e', 'E']).unwrap_or((text, "0"));
    let exponent = exponent
        .parse::<i64>()
        .map_err(|e| format!("invalid axis exponent {text:?}: {e}"))?;
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let coefficient = format!("{whole}{fraction}");
    let unsigned = coefficient.strip_prefix(['+', '-']).unwrap_or(&coefficient);
    if unsigned.is_empty() || !unsigned.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("invalid decimal axis value {text:?}"));
    }
    let scale = i64::try_from(fraction.len())
        .map_err(|_| "axis decimal is too long")?
        .checked_sub(exponent)
        .ok_or("axis scale overflow")?;
    let coefficient = Integer::from_str_radix(&coefficient, 10)
        .map_err(|e| format!("invalid axis value {text:?}: {e}"))?;
    Ok((coefficient, scale))
}

fn format_scaled(coefficient: &Integer, scale: i64) -> String {
    if coefficient.is_zero() {
        return "0".into();
    }
    let digits = coefficient.clone().abs().to_string();
    let exponent = -i128::from(scale);
    let scientific = format!("{digits}e{exponent}");
    let mut text = if scale < 0 {
        scientific
    } else if scale == 0 {
        digits
    } else {
        let fixed_len = if digits.len() as u128 <= scale as u128 {
            scale as u128 + 2
        } else {
            digits.len() as u128 + 1
        };
        if (scientific.len() as u128) < fixed_len {
            scientific
        } else {
            let scale = usize::try_from(scale)
                .expect("selected fixed decimal label exceeds addressable memory");
            let fixed = if digits.len() <= scale {
                format!("0.{}{}", "0".repeat(scale - digits.len()), digits)
            } else {
                let split = digits.len() - scale;
                format!("{}.{}", &digits[..split], &digits[split..])
            };
            fixed.trim_end_matches('0').trim_end_matches('.').to_owned()
        }
    };
    if coefficient < &0 {
        text.insert(0, '-');
    }
    text
}

fn build_axis(lo: &str, hi: &str, step: &str, prec: u64) -> Result<Vec<AxisValue>, String> {
    cnum::check_precision(prec)?;
    let (lo, lo_scale) = scaled_decimal(lo)?;
    let (hi, hi_scale) = scaled_decimal(hi)?;
    let (step, step_scale) = scaled_decimal(step)?;
    if step <= 0 {
        return Err("grid axes require a positive step and lo <= hi".into());
    }
    let mut scale = step_scale;
    if !lo.is_zero() {
        scale = scale.max(lo_scale);
    }
    if !hi.is_zero() {
        scale = scale.max(hi_scale);
    }
    let align = |coefficient: Integer, old_scale: i64| -> Result<Integer, String> {
        if coefficient.is_zero() {
            return Ok(coefficient);
        }
        let places = u64::try_from(i128::from(scale) - i128::from(old_scale))
            .map_err(|_| "axis decimal alignment exceeds addressable memory")?;
        let bits = (u128::from(places) * 332_193).div_ceil(100_000);
        if bits.div_ceil(8) > isize::MAX as u128 {
            return Err("axis decimal alignment exceeds addressable memory".into());
        }
        Ok(coefficient * Integer::from(Integer::u64_pow_u64(10, places)))
    };
    let lo = align(lo, lo_scale)?;
    let hi = align(hi, hi_scale)?;
    let step = align(step, step_scale)?;
    if step <= 0 || hi < lo {
        return Err("grid axes require a positive step and lo <= hi".into());
    }
    let count = ((hi - &lo) / &step + 1u32)
        .to_usize()
        .ok_or("axis point count exceeds addressable memory")?;
    std::alloc::Layout::array::<AxisValue>(count)
        .map_err(|_| "axis array exceeds addressable memory")?;
    cnum::check_float_storage(count as u128, prec)?;
    let mut axis = Vec::with_capacity(count);
    for i in 0..count {
        let coefficient = lo.clone() + &step * i;
        let text = format_scaled(&coefficient, scale);
        let value = cnum::parse_float(&text, prec)?;
        if !value.is_finite() || (value.is_zero() && !coefficient.is_zero()) {
            return Err("grid coordinate exceeds MPFR's exponent range".into());
        }
        axis.push(AxisValue { value, text });
    }
    Ok(axis)
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
        regions::Region::ShellThronInterior(d) => match schroder::setup_schroder(b, d, prec) {
            Ok(state) => BaseCache::SchroderCached(state),
            Err(e) => BaseCache::SetupErrorFallback(format!("Schröder setup failed: {}", e)),
        },
        regions::Region::OutsideShellThronRealPositive(d)
            if b.imag().is_zero() && *b.real() > cnum::eta_upper(prec) =>
        {
            match kouznetsov::setup_kouznetsov(b, d, prec, digits) {
                Ok(state) => BaseCache::KouznetsovCached(state),
                Err(e) => BaseCache::SetupErrorFallback(format!("kouznetsov setup failed: {}", e)),
            }
        }
        // These routes have height-dependent fallbacks, continuation, or
        // reflection rules; do not substitute a different cached family.
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
        BaseCache::SpecialBase | BaseCache::DispatchFallback | BaseCache::SetupErrorFallback(_) => {
            return dispatch::tetrate(b, h, prec, digits)
        }
        BaseCache::SchroderCached(state) => schroder::eval_schroder_at_digits(state, h, digits),
        BaseCache::KouznetsovCached(state) => {
            kouznetsov::eval_kouznetsov_at_digits(state, b, h, digits)
        }
        BaseCache::SetupError(e) => Err(e.clone()),
    };
    match cached {
        Ok(value) => Ok(value),
        Err(error) => {
            if cnum::verbose() {
                eprintln!("grid cached evaluation failed; trying dispatcher: {error}");
            }
            dispatch::tetrate(b, h, prec, digits)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axes_preserve_requested_origin_zeros_and_bounds() {
        let prec = cnum::digits_to_bits(70);
        let fixed = build_axis(
            "0.050000000000000000000000000000000000000000000000000000000001",
            "0.050000000000000000000000000000000000000000000000000000000001",
            "0.4",
            prec,
        )
        .unwrap();
        assert_eq!(fixed.len(), 1);
        assert_eq!(
            fixed[0].text,
            "0.050000000000000000000000000000000000000000000000000000000001"
        );
        assert_eq!(fixed[0].value, cnum::decimal(&fixed[0].text, prec));
        let axis = build_axis("-0.3", "0.35", "0.1", prec).unwrap();
        assert_eq!(
            axis.iter().map(|v| v.text.as_str()).collect::<Vec<_>>(),
            ["-0.3", "-0.2", "-0.1", "0", "0.1", "0.2", "0.3"]
        );
        assert!(axis[3].value.is_zero());
        let tiny = build_axis("-1e-1000", "-1e-1000", "1", prec).unwrap();
        assert_eq!(tiny[0].value, cnum::decimal("-1e-1000", prec));
        assert_eq!(
            cnum::parse_float(&tiny[0].text, prec).unwrap(),
            tiny[0].value
        );
        assert_eq!(build_axis("+.5", "2.5e-0", "1e+0", prec).unwrap().len(), 3);
    }

    #[test]
    fn input_precision_preserves_axis_domains_and_parity() {
        for digits in [1, 10, 50, 70, 1000] {
            let zeros = "0".repeat((digits + 100) as usize);
            let base = format!("1.{zeros}1");
            let odd = format!("1{zeros}1");
            let fractional = format!("2.{zeros}1");
            let prec =
                cnum::checked_input_precision(digits, &[&base, &odd, &fractional, "1"]).unwrap();
            let base_axis = build_axis(&base, &base, "1", prec).unwrap();
            let odd_axis = build_axis(&odd, &odd, "1", prec).unwrap();
            let fractional_axis = build_axis(&fractional, &fractional, "1", prec).unwrap();
            assert_eq!(base_axis[0].text, base);
            assert_eq!(odd_axis[0].text, odd);
            assert_eq!(fractional_axis[0].text, fractional);
            let b = Complex::with_val_64(prec, (&base_axis[0].value, 0));
            let h = Complex::with_val_64(prec, (&odd_axis[0].value, 0));
            assert_eq!(
                eval_cell(
                    &BaseCache::DispatchFallback,
                    &b,
                    &Complex::with_val_64(prec, -1),
                    prec,
                    digits,
                )
                .unwrap(),
                cnum::zero(prec)
            );
            assert_eq!(
                eval_cell(&BaseCache::SpecialBase, &cnum::zero(prec), &h, prec, digits).unwrap(),
                cnum::zero(prec)
            );
            let h = Complex::with_val_64(prec, (&fractional_axis[0].value, 0));
            assert_eq!(
                domain_undefined(&cnum::zero(prec), &h),
                Some("b=0: tetration only defined for non-negative integer heights")
            );
        }
    }

    #[test]
    fn axes_reject_invalid_and_unbounded_requests() {
        let prec = cnum::digits_to_bits(50);
        for (lo, hi, step) in [
            ("0", "1", "0"),
            ("0", "1", "-0.1"),
            ("NaN", "1", "0.1"),
            ("0", "inf", "0.1"),
            ("1", "0", "0.1"),
            ("0", "1", "bad"),
            ("0", "1", "1e-1000"),
            ("0", "1.2.3", "1"),
            ("0", "1", "1e-9223372036854775808"),
        ] {
            assert!(build_axis(lo, hi, step, prec).is_err(), "{lo} {hi} {step}");
        }
        assert!(build_axis("0", "0", "1", u64::MAX).is_err());
    }

    #[test]
    fn axes_have_no_million_point_or_128_mib_budget() {
        let prec = cnum::digits_to_bits(50);
        let axis = build_axis("0", "1000001", "1", prec).unwrap();
        assert_eq!(axis.len(), 1_000_002);
        assert_eq!(axis.first().unwrap().text, "0");
        assert_eq!(axis.last().unwrap().text, "1000001");
        assert_eq!(axis.last().unwrap().value, 1_000_001);
    }

    #[test]
    fn axes_do_not_expand_common_exponents_into_integer_zeros() {
        let prec = cnum::digits_to_bits(70);
        for exponent in [
            10_000,
            1_000_001,
            10_000_000_000i64,
            -1_000_001,
            -10_000_000_000,
        ] {
            let lo = format!("1e{exponent}");
            let hi = format!("3e{exponent}");
            let axis = build_axis(&lo, &hi, &lo, prec).unwrap();
            assert_eq!(axis.len(), 3);
            for (i, coordinate) in axis.iter().enumerate() {
                assert!(coordinate.text.len() < 30);
                assert_eq!(
                    coordinate.value,
                    cnum::decimal(&format!("{}e{exponent}", i + 1), prec)
                );
            }
        }
        let axis = build_axis("0", "1e10000", "1e9995", prec).unwrap();
        assert_eq!(axis.len(), 100_001);
        assert_eq!(axis.last().unwrap().value, cnum::decimal("1e10000", prec));
    }

    #[test]
    fn cached_and_degenerate_routes_match_dispatch() {
        let digits = 50;
        let prec = cnum::digits_to_bits(digits);
        let h = cnum::parse_complex("0.4", "0.2", prec).unwrap();
        let b = cnum::parse_complex("1.2", "0", prec).unwrap();
        let cache = build_cache(&b, prec, digits);
        assert_eq!(
            eval_cell(&cache, &b, &h, prec, digits).unwrap(),
            dispatch::tetrate(&b, &h, prec, digits).unwrap()
        );
        for (re, im) in [("1.2", "-0.1"), ("1.5", "0"), ("-2", "0"), ("0.05", "0")] {
            let base = cnum::parse_complex(re, im, prec).unwrap();
            assert!(matches!(
                build_cache(&base, prec, digits),
                BaseCache::DispatchFallback
            ));
        }
        let negative = cnum::parse_complex("-1e1000", "0", prec).unwrap();
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
        let even = cnum::parse_complex("1e60", "0", prec).unwrap();
        assert!(domain_undefined(&cnum::zero(prec), &even).is_none());
        assert_eq!(
            eval_cell(
                &BaseCache::SpecialBase,
                &cnum::zero(prec),
                &even,
                prec,
                digits
            )
            .unwrap(),
            cnum::one(prec)
        );
    }
}
