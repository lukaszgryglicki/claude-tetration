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
//! FFT-induced roundoff over `M log₂ M` butterflies adds O(log M) digits of
//! noise, well inside the standard guard-bit budget for `digits` ≤ 10⁶.
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
pub fn fft(a: &mut [Complex], prec: u32, inverse: bool) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    assert!(n.is_power_of_two(), "fft length must be power of 2 (got {})", n);

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
    let pi = Float::with_val(prec, Constant::Pi);
    let two_pi = Float::with_val(prec, &pi * 2u32);
    let sign: i32 = if inverse { 1 } else { -1 };

    let mut size = 2usize;
    while size <= n {
        let half = size / 2;
        // omega_step = exp(sign · 2πi / size). Computing one root per stage
        // and stepping by complex multiplication inside the inner loop keeps
        // MPC trig calls out of the hot path.
        let theta_unsigned = Float::with_val(prec, &two_pi / (size as u32));
        let theta = Float::with_val(prec, &theta_unsigned * sign);
        let cos_t = Float::with_val(prec, theta.cos_ref());
        let sin_t = Float::with_val(prec, theta.sin_ref());
        let omega_step = Complex::with_val(prec, (cos_t, sin_t));

        let mut start = 0usize;
        while start < n {
            let mut omega = Complex::with_val(prec, (Float::with_val(prec, 1u32), 0));
            for k in 0..half {
                let t = Complex::with_val(prec, &omega * &a[start + k + half]);
                let u = a[start + k].clone();
                a[start + k] = Complex::with_val(prec, &u + &t);
                a[start + k + half] = Complex::with_val(prec, &u - &t);
                omega = Complex::with_val(prec, &omega * &omega_step);
            }
            start += size;
        }
        size <<= 1;
    }

    if inverse {
        let inv_n = Float::with_val(prec, 1u32) / Float::with_val(prec, n as u32);
        for c in a.iter_mut() {
            *c = Complex::with_val(prec, &*c * &inv_n);
        }
    }
}

/// Linear convolution `c[n] = Σ_m a[m] · b[n − m]`. Output length is
/// `a.len() + b.len() − 1`. Uses a single forward+forward+inverse FFT at
/// padded length `next_power_of_two(a.len() + b.len() − 1)`.
pub fn convolve(a: &[Complex], b: &[Complex], prec: u32) -> Vec<Complex> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let result_len = a.len() + b.len() - 1;
    let m = result_len.next_power_of_two().max(2);

    let mut a_pad: Vec<Complex> = Vec::with_capacity(m);
    let zero_c = Complex::with_val(prec, (Float::new(prec), Float::new(prec)));
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

    for i in 0..m {
        a_pad[i] = Complex::with_val(prec, &a_pad[i] * &b_pad[i]);
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

/// Pre-FFT a length-`(2n − 1)` kernel `h` for repeated use in
/// `cross_correlate_with_kernel`. `h[d]` corresponds to offset `d − (n − 1)`
/// in the math; that is, `h[n − 1]` is the d=0 entry.
pub fn precompute_kernel_fft(h: &[Complex], n: usize, prec: u32) -> KernelFft {
    assert_eq!(h.len(), 2 * n - 1, "kernel must have length 2n−1");
    let result_len = 3 * n - 2;
    let m = result_len.next_power_of_two().max(2);

    let zero_c = Complex::with_val(prec, (Float::new(prec), Float::new(prec)));

    // h_rev[i] = h[2n−2−i] for i in 0..2n−1.
    let mut h_pad: Vec<Complex> = Vec::with_capacity(m);
    for i in 0..(2 * n - 1) {
        h_pad.push(h[(2 * n - 2) - i].clone());
    }
    while h_pad.len() < m {
        h_pad.push(zero_c.clone());
    }
    fft(&mut h_pad, prec, false);

    KernelFft { coeffs: h_pad, n, m }
}

/// Compute `out[k] = Σ_{j=0..n−1} a[j] · h[j − k + (n − 1)]` for `k ∈ 0..n`,
/// using the pre-FFT'd kernel. `a.len()` must equal `kernel.n`.
///
/// Algorithm: cross-correlation = convolution of `a` with reversed `h`,
/// reading off the central `n` outputs. The kernel FFT is precomputed; this
/// call does one length-`m` forward FFT on `a`, a pointwise multiply, and one
/// inverse FFT.
pub fn cross_correlate_with_kernel(
    a: &[Complex],
    kernel: &KernelFft,
    prec: u32,
) -> Vec<Complex> {
    assert_eq!(a.len(), kernel.n, "input length must match kernel.n");
    let n = kernel.n;
    let m = kernel.m;
    let zero_c = Complex::with_val(prec, (Float::new(prec), Float::new(prec)));

    let mut a_pad: Vec<Complex> = Vec::with_capacity(m);
    a_pad.extend(a.iter().cloned());
    while a_pad.len() < m {
        a_pad.push(zero_c.clone());
    }
    fft(&mut a_pad, prec, false);

    for i in 0..m {
        a_pad[i] = Complex::with_val(prec, &a_pad[i] * &kernel.coeffs[i]);
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

    fn approx_eq(a: &Complex, b: &Complex, tol: f64) -> bool {
        let prec = a.prec().0;
        let diff = Complex::with_val(prec, a - b);
        let abs = Float::with_val(prec, diff.abs_ref()).to_f64();
        abs < tol
    }

    #[test]
    fn fft_roundtrip_small() {
        let prec = 128u32;
        let n = 8;
        let mut x: Vec<Complex> = (0..n)
            .map(|i| {
                Complex::with_val(prec, (Float::with_val(prec, i as f64), Float::with_val(prec, (i * 2) as f64)))
            })
            .collect();
        let original: Vec<Complex> = x.iter().cloned().collect();
        fft(&mut x, prec, false);
        fft(&mut x, prec, true);
        for i in 0..n {
            assert!(
                approx_eq(&x[i], &original[i], 1e-30),
                "roundtrip mismatch at i={}",
                i
            );
        }
    }

    #[test]
    fn convolution_matches_direct() {
        let prec = 128u32;
        let a: Vec<Complex> = (0..4)
            .map(|i| Complex::with_val(prec, (Float::with_val(prec, i as f64 + 1.0), 0)))
            .collect();
        let b: Vec<Complex> = (0..3)
            .map(|i| Complex::with_val(prec, (Float::with_val(prec, (2 * i + 1) as f64), 0)))
            .collect();

        let conv = convolve(&a, &b, prec);
        let m = a.len() + b.len() - 1;
        assert_eq!(conv.len(), m);

        // Direct: c[n] = Σ a[k] · b[n−k].
        for n in 0..m {
            let mut expected = Complex::with_val(prec, (0, 0));
            for k in 0..a.len() {
                if n >= k && n - k < b.len() {
                    let prod = Complex::with_val(prec, &a[k] * &b[n - k]);
                    expected = Complex::with_val(prec, &expected + &prod);
                }
            }
            assert!(
                approx_eq(&conv[n], &expected, 1e-25),
                "convolution mismatch at n={}",
                n
            );
        }
    }

    #[test]
    fn cross_correlate_matches_direct() {
        let prec = 128u32;
        let n = 4usize;
        let a: Vec<Complex> = (0..n)
            .map(|i| Complex::with_val(prec, (Float::with_val(prec, i as f64 + 1.0), 0)))
            .collect();
        // h has length 2n−1 = 7
        let h: Vec<Complex> = (0..(2 * n - 1))
            .map(|i| Complex::with_val(prec, (Float::with_val(prec, (i + 1) as f64), Float::with_val(prec, (i as f64) * 0.5))))
            .collect();

        let kernel = precompute_kernel_fft(&h, n, prec);
        let fft_out = cross_correlate_with_kernel(&a, &kernel, prec);

        // Direct: out[k] = Σ_{j} a[j] · h[j − k + (n−1)]
        for k in 0..n {
            let mut expected = Complex::with_val(prec, (0, 0));
            for j in 0..n {
                let idx = (j as i64) - (k as i64) + (n as i64) - 1;
                if idx >= 0 && (idx as usize) < h.len() {
                    let prod = Complex::with_val(prec, &a[j] * &h[idx as usize]);
                    expected = Complex::with_val(prec, &expected + &prod);
                }
            }
            assert!(
                approx_eq(&fft_out[k], &expected, 1e-25),
                "cross-correlation mismatch at k={} (expected {:?}, got {:?})",
                k,
                expected,
                fft_out[k]
            );
        }
    }
}
