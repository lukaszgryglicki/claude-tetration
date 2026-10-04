//! Arbitrary-precision complex FFT and convolution helpers.
//!
//! Used by the Kouznetsov Cauchy iteration to turn its O(N²) matrix-vector
//! products on the Cauchy operator (and its Jacobian) into O(N log N) FFT-based
//! convolutions. The matvecs in `apply_t` and `apply_dt_v` both have the
//! cross-correlation structure
//!
//! ```text
//! out[k] = Σ_{j=0..N-1} a[j] · h[j − k + (N−1)]
//! ```
//!
//! where the kernel `h` (denominator inverses on a uniform grid) depends only
//! on the offset `d = j − k`. That maps onto a single linear convolution of
//! length `3N − 2` after reversing `h`, which we compute via radix-2
//! Cooley-Tukey on the next power of two ≥ `3N − 2`.
//!
//! # Precision
//! All operations run at the same MPC precision as the rest of the algorithm.
//! Roundoff depends on transform length and input conditioning; arbitrary
//! working precision alone is not a returned-value error bound.
//!
//! # Twiddle factors
//! Twiddle roots `ω_M = exp(−2πi/M)` are computed once per call (only one is
//! needed; the rest are obtained by complex multiplication during the FFT).
//! For sizes that recur — Kouznetsov uses the same `M = next_pow2(3N − 2)`
//! across every Newton step at a given precision — callers can reuse the
//! transformed kernel via `precompute_kernel_fft` so we don't re-FFT a
//! fixed array on every matvec.

use rug::{float::Constant, Complex, Float};

/// In-place radix-2 Cooley-Tukey FFT for arbitrary-precision complex.
///
/// `a.len()` must be a power of two. `inverse = true` performs the inverse
/// transform with the customary `1/N` scaling; otherwise the forward transform
/// (un-normalized).
///
/// Dispatch: with `TET_MT` unset/0 (the default) this runs the original
/// serial implementation, untouched. With MT mode on it runs the parallel
/// variant, which is bit-identical (see `fft_mt`).
pub fn fft(a: &mut [Complex], prec: u64, inverse: bool) {
    if crate::mt::mt_enabled() {
        fft_mt(a, prec, inverse);
    } else {
        fft_serial(a, prec, inverse);
    }
}

/// Original serial FFT (the default path — code unchanged).
fn fft_serial(a: &mut [Complex], prec: u64, inverse: bool) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    assert!(
        n.is_power_of_two(),
        "fft length must be power of 2 (got {})",
        n
    );

    // Bit-reverse permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }

    // Cooley-Tukey butterflies.
    let pi = Float::with_val_64(prec, Constant::Pi);
    let two_pi = Float::with_val_64(prec, &pi * 2u32);
    let sign: i32 = if inverse { 1 } else { -1 };

    let mut size = 2usize;
    while size <= n {
        let half = size / 2;
        // omega_step = exp(sign · 2πi / size). Computing one root per stage
        // and stepping by complex multiplication inside the inner loop keeps
        // MPC trig calls out of the hot path.
        let theta_unsigned = Float::with_val_64(prec, &two_pi / size);
        let theta = Float::with_val_64(prec, &theta_unsigned * sign);
        let cos_t = Float::with_val_64(prec, theta.cos_ref());
        let sin_t = Float::with_val_64(prec, theta.sin_ref());
        let omega_step = Complex::with_val_64(prec, (cos_t, sin_t));

        let mut start = 0usize;
        while start < n {
            let mut omega = Complex::with_val_64(prec, (Float::with_val_64(prec, 1u32), 0));
            for k in 0..half {
                let t = Complex::with_val_64(prec, &omega * &a[start + k + half]);
                let u = a[start + k].clone();
                a[start + k] = Complex::with_val_64(prec, &u + &t);
                a[start + k + half] = Complex::with_val_64(prec, &u - &t);
                omega = Complex::with_val_64(prec, &omega * &omega_step);
            }
            start += size;
        }
        if size == n {
            break;
        }
        size <<= 1;
    }

    if inverse {
        let inv_n = Float::with_val_64(prec, 1u32) / Float::with_val_64(prec, n);
        for c in a.iter_mut() {
            *c = Complex::with_val_64(prec, &*c * &inv_n);
        }
    }
}

/// Parallel FFT for MT mode (`TET_MT` ≥ 1). Bit-identical to `fft_serial`:
///
/// * Each stage's twiddle factors are precomputed into a table by the **same
///   sequential recurrence** the serial code uses (`ω ← ω·ω_step` starting
///   from 1), so every butterfly sees the exact same correctly-rounded ω
///   value it would have seen serially. In the serial code every block within
///   a stage independently regenerates this identical sequence, so one table
///   per stage serves all blocks.
/// * Butterflies write disjoint pairs `(a[start+k], a[start+k+half])`:
///   blocks are disjoint `size`-sized chunks and, within a block, the k-th
///   butterfly touches only offsets `k` and `k+half`. Parallelizing over
///   blocks (`par_chunks_mut`) and over k (zip of the two half-slices)
///   reorders no floating-point accumulation — each output element is
///   produced by the same two operations on the same operands as in the
///   serial code, and MPC arithmetic is deterministic and correctly rounded.
fn fft_mt(a: &mut [Complex], prec: u64, inverse: bool) {
    use rayon::prelude::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    type TwiddleCache = Mutex<HashMap<(usize, u64, bool), Arc<Vec<Complex>>>>;

    // Twiddle-table cache. Keyed by (stage size, precision, direction);
    // tables are immutable once built and shared via Arc. FFT sizes and
    // precision repeat thousands of times within one solve (every matvec of
    // every Krylov step reuses the same padded length), so each table is
    // built exactly once per process — its sequential-recurrence cost
    // amortizes to zero and the butterflies get the full parallel speedup.
    static TWIDDLES: OnceLock<TwiddleCache> = OnceLock::new();

    let n = a.len();
    if n <= 1 {
        return;
    }
    assert!(
        n.is_power_of_two(),
        "fft length must be power of 2 (got {})",
        n
    );

    // Bit-reverse permutation (pure swaps; cheap, kept serial).
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }

    let pi = Float::with_val_64(prec, Constant::Pi);
    let two_pi = Float::with_val_64(prec, &pi * 2u32);
    let sign: i32 = if inverse { 1 } else { -1 };

    let mut size = 2usize;
    while size <= n {
        let half = size / 2;

        // Fetch (or build once) this stage's twiddle table. The values are
        // produced by the same sequential recurrence (ω ← ω·ω_step from 1)
        // the serial code runs per block, so they are bit-identical to the
        // serial code's ω sequence.
        let cache = TWIDDLES.get_or_init(|| Mutex::new(HashMap::new()));
        let twiddles: Arc<Vec<Complex>> = {
            let key = (size, prec, inverse);
            let hit = cache.lock().unwrap().get(&key).cloned();
            match hit {
                Some(t) => t,
                None => {
                    let theta_unsigned = Float::with_val_64(prec, &two_pi / size);
                    let theta = Float::with_val_64(prec, &theta_unsigned * sign);
                    let cos_t = Float::with_val_64(prec, theta.cos_ref());
                    let sin_t = Float::with_val_64(prec, theta.sin_ref());
                    let omega_step = Complex::with_val_64(prec, (cos_t, sin_t));
                    let mut table: Vec<Complex> = Vec::with_capacity(half);
                    let mut omega = Complex::with_val_64(prec, (Float::with_val_64(prec, 1u32), 0));
                    for _ in 0..half {
                        table.push(omega.clone());
                        omega = Complex::with_val_64(prec, &omega * &omega_step);
                    }
                    let arc = Arc::new(table);
                    cache.lock().unwrap().insert(key, Arc::clone(&arc));
                    arc
                }
            }
        };

        a.par_chunks_mut(size).for_each(|chunk| {
            let (lo, hi) = chunk.split_at_mut(half);
            lo.par_iter_mut()
                .zip(hi.par_iter_mut())
                .zip(twiddles.par_iter())
                .with_min_len(16)
                .for_each(|((u_ref, t_ref), w)| {
                    let t = Complex::with_val_64(prec, w * &*t_ref);
                    let u = u_ref.clone();
                    *u_ref = Complex::with_val_64(prec, &u + &t);
                    *t_ref = Complex::with_val_64(prec, &u - &t);
                });
        });
        if size == n {
            break;
        }
        size <<= 1;
    }

    if inverse {
        let inv_n = Float::with_val_64(prec, 1u32) / Float::with_val_64(prec, n);
        a.par_iter_mut().with_min_len(16).for_each(|c| {
            *c = Complex::with_val_64(prec, &*c * &inv_n);
        });
    }
}

/// Linear convolution `c[n] = Σ_m a[m] · b[n − m]`. Output length is
/// `a.len() + b.len() − 1`. Uses a single forward+forward+inverse FFT at
/// padded length `next_power_of_two(a.len() + b.len() − 1)`.
pub fn convolve(a: &[Complex], b: &[Complex], prec: u64) -> Vec<Complex> {
    crate::cnum::init_mpfr();
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let result_len = a
        .len()
        .checked_add(b.len() - 1)
        .expect("convolution length overflow");
    let m = checked_fft_len(result_len, prec).expect("convolution exceeds addressable memory");

    let mut a_pad: Vec<Complex> = Vec::with_capacity(m);
    let zero_c = Complex::with_val_64(prec, (Float::new_64(prec), Float::new_64(prec)));
    a_pad.extend(a.iter().cloned());
    while a_pad.len() < m {
        a_pad.push(zero_c.clone());
    }
    let mut b_pad: Vec<Complex> = Vec::with_capacity(m);
    b_pad.extend(b.iter().cloned());
    while b_pad.len() < m {
        b_pad.push(zero_c.clone());
    }

    fft(&mut a_pad, prec, false);
    fft(&mut b_pad, prec, false);

    if crate::mt::mt_enabled() {
        use rayon::prelude::*;
        a_pad
            .par_iter_mut()
            .zip(b_pad.par_iter())
            .with_min_len(16)
            .for_each(|(x, y)| {
                *x = Complex::with_val_64(prec, &*x * y);
            });
    } else {
        for i in 0..m {
            a_pad[i] = Complex::with_val_64(prec, &a_pad[i] * &b_pad[i]);
        }
    }

    fft(&mut a_pad, prec, true);

    a_pad.truncate(result_len);
    a_pad
}

/// Pre-computed FFT-domain kernel, ready for repeated convolution against
/// varying input arrays of fixed length `n`. The kernel is conceptually the
/// reversed array `h_rev[i] = h[2n−2−i]` (so a normal convolution gives the
/// shifted cross-correlation we want); we store its FFT at padded length
/// `m = next_pow2(3n − 2)`.
pub struct KernelFft {
    /// FFT(h_rev) at padded length `m`.
    pub coeffs: Vec<Complex>,
    pub n: usize,
    pub m: usize,
}

fn checked_fft_len(len: usize, prec: u64) -> Result<usize, String> {
    let m = len
        .checked_next_power_of_two()
        .ok_or("FFT padding exceeds addressable memory")?
        .max(2);
    std::alloc::Layout::array::<Complex>(m).map_err(|_| "FFT array exceeds addressable memory")?;
    crate::cnum::check_complex_storage(m as u128, prec)?;
    Ok(m)
}

pub(crate) fn kernel_fft_len(n: usize, prec: u64) -> Result<usize, String> {
    let len = n
        .checked_mul(3)
        .and_then(|v| v.checked_sub(2))
        .ok_or("Cauchy FFT length is zero or exceeds addressable memory")?;
    checked_fft_len(len, prec)
}

/// Pre-FFT a length-`(2n − 1)` kernel `h` for repeated use in
/// `cross_correlate_with_kernel`. `h[d]` corresponds to offset `d − (n − 1)`
/// in the math; that is, `h[n − 1]` is the d=0 entry.
pub fn precompute_kernel_fft(h: &[Complex], n: usize, prec: u64) -> KernelFft {
    crate::cnum::init_mpfr();
    let m = kernel_fft_len(n, prec).expect("Cauchy FFT exceeds addressable memory");
    assert_eq!(h.len(), 2 * n - 1, "kernel must have length 2n−1");

    let zero_c = Complex::with_val_64(prec, (Float::new_64(prec), Float::new_64(prec)));

    // h_rev[i] = h[2n−2−i] for i in 0..2n−1.
    let mut h_pad: Vec<Complex> = Vec::with_capacity(m);
    for i in 0..(2 * n - 1) {
        h_pad.push(h[(2 * n - 2) - i].clone());
    }
    while h_pad.len() < m {
        h_pad.push(zero_c.clone());
    }
    fft(&mut h_pad, prec, false);

    KernelFft {
        coeffs: h_pad,
        n,
        m,
    }
}

/// Compute `out[k] = Σ_{j=0..n−1} a[j] · h[j − k + (n − 1)]` for `k ∈ 0..n`,
/// using the pre-FFT'd kernel. `a.len()` must equal `kernel.n`.
///
/// Algorithm: cross-correlation = convolution of `a` with reversed `h`,
/// reading off the central `n` outputs. The kernel FFT is precomputed; this
/// call does one length-`m` forward FFT on `a`, a pointwise multiply, and one
/// inverse FFT.
pub fn cross_correlate_with_kernel(a: &[Complex], kernel: &KernelFft, prec: u64) -> Vec<Complex> {
    crate::cnum::init_mpfr();
    assert_eq!(a.len(), kernel.n, "input length must match kernel.n");
    let n = kernel.n;
    let m = kernel.m;
    let zero_c = Complex::with_val_64(prec, (Float::new_64(prec), Float::new_64(prec)));

    let mut a_pad: Vec<Complex> = Vec::with_capacity(m);
    a_pad.extend(a.iter().cloned());
    while a_pad.len() < m {
        a_pad.push(zero_c.clone());
    }
    fft(&mut a_pad, prec, false);

    if crate::mt::mt_enabled() {
        use rayon::prelude::*;
        a_pad
            .par_iter_mut()
            .zip(kernel.coeffs.par_iter())
            .with_min_len(16)
            .for_each(|(x, k)| {
                *x = Complex::with_val_64(prec, &*x * k);
            });
    } else {
        for (value, coefficient) in a_pad.iter_mut().zip(&kernel.coeffs) {
            *value = Complex::with_val_64(prec, &*value * coefficient);
        }
    }

    fft(&mut a_pad, prec, true);

    // The convolution result's index N−1 corresponds to lag p=0 (i.e. k=N−1),
    // and index 2N−2 corresponds to lag p=N−1 (k=0). So out[k] = conv[N−1+k].
    // We collect them with k ascending, which means stepping forward.
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        out.push(a_pad[(n - 1) + k].clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &Complex, b: &Complex, digits: u64) -> bool {
        let prec = a.prec_64().0;
        assert!(crate::cnum::is_finite(a) && crate::cnum::is_finite(b));
        let diff = Complex::with_val_64(prec, a - b);
        let abs = Float::with_val_64(prec, diff.abs_ref());
        abs < crate::cnum::epsilon(digits, prec)
    }

    #[test]
    fn fft_roundtrip_small() {
        let prec = 128u64;
        let n = 8;
        let mut x: Vec<Complex> = (0..n)
            .map(|i| Complex::with_val_64(prec, (i, i * 2)))
            .collect();
        let original: Vec<Complex> = x.to_vec();
        fft(&mut x, prec, false);
        fft(&mut x, prec, true);
        for i in 0..n {
            assert!(
                approx_eq(&x[i], &original[i], 30),
                "roundtrip mismatch at i={}",
                i
            );
        }
    }

    #[test]
    fn convolution_matches_direct() {
        let prec = 128u64;
        let a: Vec<Complex> = (0..4)
            .map(|i| Complex::with_val_64(prec, (i + 1, 0)))
            .collect();
        let b: Vec<Complex> = (0..3)
            .map(|i| Complex::with_val_64(prec, (2 * i + 1, 0)))
            .collect();

        let conv = convolve(&a, &b, prec);
        let m = a.len() + b.len() - 1;
        assert_eq!(conv.len(), m);

        // Direct: c[n] = Σ a[k] · b[n−k].
        for n in 0..m {
            let mut expected = Complex::with_val_64(prec, (0, 0));
            for k in 0..a.len() {
                if n >= k && n - k < b.len() {
                    let prod = Complex::with_val_64(prec, &a[k] * &b[n - k]);
                    expected = Complex::with_val_64(prec, &expected + &prod);
                }
            }
            assert!(
                approx_eq(&conv[n], &expected, 25),
                "convolution mismatch at n={}",
                n
            );
        }
    }

    #[test]
    fn cross_correlate_matches_direct() {
        let prec = 128u64;
        let n = 4usize;
        let a: Vec<Complex> = (0..n)
            .map(|i| Complex::with_val_64(prec, (i + 1, 0)))
            .collect();
        // h has length 2n−1 = 7
        let h: Vec<Complex> = (0..(2 * n - 1))
            .map(|i| Complex::with_val_64(prec, (i + 1, Float::with_val_64(prec, i) / 2)))
            .collect();

        let kernel = precompute_kernel_fft(&h, n, prec);
        let fft_out = cross_correlate_with_kernel(&a, &kernel, prec);
        assert_eq!(fft_out.len(), n);

        // Direct: out[k] = Σ_{j} a[j] · h[j − k + (n−1)]
        for (k, actual) in fft_out.iter().enumerate() {
            let mut expected = Complex::with_val_64(prec, (0, 0));
            for (j, value) in a.iter().enumerate() {
                let idx = j + (n - 1 - k);
                let prod = Complex::with_val_64(prec, value * &h[idx]);
                expected = Complex::with_val_64(prec, &expected + &prod);
            }
            assert!(
                approx_eq(actual, &expected, 25),
                "cross-correlation mismatch at k={} (expected {:?}, got {:?})",
                k,
                expected,
                actual
            );
        }
    }

    #[test]
    fn serial_and_mt_fft_agree_beyond_native_precision() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for digits in [50, 70, 1000] {
            let prec = crate::cnum::digits_to_bits(digits);
            let original: Vec<_> = (0..1024)
                .map(|i| {
                    Complex::with_val_64(
                        prec,
                        (
                            Float::with_val_64(prec, i + 1) / 7,
                            Float::with_val_64(prec, i) / 11,
                        ),
                    )
                })
                .collect();
            let mut serial = original.clone();
            let mut parallel = original.clone();
            for inverse in [false, true] {
                fft_serial(&mut serial, prec, inverse);
                pool.install(|| fft_mt(&mut parallel, prec, inverse));
                for (a, b) in serial.iter().zip(&parallel) {
                    assert_eq!(a, b, "{digits} digits, inverse={inverse}");
                    assert_eq!(a.real().is_sign_negative(), b.real().is_sign_negative());
                    assert_eq!(a.imag().is_sign_negative(), b.imag().is_sign_negative());
                }
            }
            for (actual, expected) in serial.iter().zip(&original) {
                assert!(approx_eq(actual, expected, digits));
            }
        }
    }
}
