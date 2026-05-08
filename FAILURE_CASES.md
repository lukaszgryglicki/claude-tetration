# Failure 4-Tuples for Tetration `tet b_re b_im h_re h_im`

Classes of currently-unsupported / failing inputs, organized by failure mode.
Each row is a 4-tuple `b_re b_im h_re h_im` — feed directly to `tet <prec>` for testing.

Legend:
- **ERR** — exits cleanly with non-zero status and a diagnostic on stderr; no result printed.
- **HANG** — does not complete within 90 s at precision 50; algorithm spins / fails to produce output.
- **WRONG** — historically produced garbage (e.g. magnitudes ~1e+3000); now generally captured as ERR after the two-tier residual gate, but listed because correctness is still unsolved.

---

## A. Shell-Thron parabolic boundary band  (|λ| ≈ 1)

Schröder regular tetration converges too slowly here (logarithmic rate);
Newton-Kantorovich Kouznetsov has no usable contour because the canonical
Kneser fixed-point pair degenerates onto the real axis. Needs Écalle/Abel
parabolic-iteration theory or a dedicated boundary algorithm.

| b_re | b_im | h_re | h_im | mode | note |
|---|---|---|---|---|---|
| 1.4446678610097661337 | 0 | 0.5 | 0 | ERR | exact η = e^(1/e) |
| 1.444667861009766 | 0 | 0.5 | 0 | ERR | t710 covers this |
| 1.45 | 0 | 0.5 | 0 | HANG | just-outside η, |λ| barely > 1 |
| 1.43 | 0 | 0.5 | 0 | OK | returns 1.2506... (good — Schröder works) |
| 1.44 | 0 | 0.5 | 0 | **WRONG** | returns 2.3082... but neighbors are ~1.25 — discontinuous, near-band Schröder unreliable |

---

## B. Negative real bases  (b ∈ ℝ, b < 0)

The dispatcher routes these to Newton-Kantorovich, but the canonical
fixed-point pair (L, L̄) is hard to choose: for |b|<1 in particular both
log-fixed-points are repelling; for b≈−1 the dynamics is degenerate.
Existing W_k branch search converges, but the Cauchy contour cannot be
oriented to enclose a Kneser-canonical pair. Needs branch-pair selection
logic specific to b<0.

| b_re | b_im | h_re | h_im | mode |
|---|---|---|---|---|
| -0.5 | 0 | 0.5 | 0 | ERR |
| -0.99 | 0 | 0.5 | 0 | ERR |
| -1 | 0 | 0.5 | 0 | ERR |
| -2 | 0 | 0.5 | 0 | ERR |
| -2 | 0 | 1.5 | 0 | ERR |
| -0.5 | 0 | 2.5 | 0 | ERR |

---

## C. Pure-imaginary bases  (b = i·y, y ≠ 0)

Boundary at `|b_im| ≈ 1.4` (where `|λ| crosses 0.95`):
- `|b_im| ≤ 1.3` → Schröder works (Shell-Thron interior).
- `|b_im| ≥ 1.4` → routes to Newton-Kouznetsov (boundary band) which
  converges to a wrong-basin attractor.

Diagnosis: from initial guess `F̃[mid] = √b ≈ 0.87 + 0.87i`, the LM
descent walks `F̃[mid]` toward `0+0i` (a degenerate fixed point of the
discretized Cauchy operator). Residual ‖r‖∞ stalls at ~5×10⁻⁸ around
iter 40 before Newton can reach quadratic convergence — this is a
genuine local minimum of the discretized residual, not the Kneser
solution. The non-Schwarz path lacks a basin guard analogous to the
real-positive `f_mid_re < 0` rejection.

Fix path forward: either (a) continuation from a known-good base
(Schröder seed at `|b_im|=1.3`, refined for the target via Kouznetsov
with the seed as initial guess), or (b) Paulsen-Cowgill conformal
mapping (the canonical algorithm for off-real bases), or (c) basin
guard rejecting `|F̃[mid]| < 0.5·|target_mid|` for non-Schwarz.

| b_re | b_im | h_re | h_im | mode |
|---|---|---|---|---|
| 0 | 0.5 | 0.5 | 0 | OK (returns ≈ 0.821 + 0.990i) |
| 0 | 1 | 0.5 | 0 | OK (returns 1.167 + 0.735i) |
| 0 | 1.1 | 0.5 | 0 | OK (1.171 + 0.693i) |
| 0 | 1.2 | 0.5 | 0 | OK (1.159 + 0.658i) |
| 0 | 1.3 | 0.5 | 0 | OK (1.130 + 0.629i) |
| 0 | 1.4 | 0.5 | 0 | HANG (boundary band, wrong-basin LM descent) |
| 0 | 1.5 | 0.5 | 0 | HANG |
| 0 | 5 | 0.5 | 0 | HANG |
| 0 | -3 | 0.5 | 0 | HANG |
| 0 | 2 | 1 | 0.5 | HANG |

---

## D. Complex bases far from real axis  — **RESOLVED** (Im(b)≥0; Im(b)<0 covered by Schwarz)

`b = a + bi` with large `|Im(b)|` relative to `|Re(b)|`. The Newton-
Kouznetsov path converges quadratically for Im(b)≥0 cases once past the
initial linear-descent phase (≥3-min at 20 digits). For Im(b)<0, the
**Schwarz reflection** `F_b(h) = conj(F_{b̄}(h̄))` reduces to Im(b)>0.

**Fix** (`dispatch.rs`): at entry, when Im(b)<0, dispatch via conjugate
base and conjugate height, then conjugate the result. This is an exact
mathematical identity for the canonical Kneser tetration, not an approximation.

| b_re | b_im | h_re | h_im | mode |
|---|---|---|---|---|
| 1.2 | 3.5 | 0.5 | 0 | OK (0.2024 + 0.5434i, ~3-4 min) |
| 0.5 | 2 | 0.5 | 0 | OK (0.2498 + 0.5270i) |
| -3.6 | -0.4 | 0.5 | 0 | **WAS HANG** → OK via Schwarz (~165s) |
| -1.2 | -1.2 | 0.5 | 0 | **WAS HANG** → OK via Schwarz (~548s) |
| -1 | 1 | 0.5 | 0 | ERR (Re(b)<0, Im(b)=0 → see Class B) |
| 2 | 5 | 0 | 0.5 | OK |

---

## E. Large positive real bases  (b ≫ e^(1/e))   — **RESOLVED**

Previously the LM/GMRES solver got stuck for large `|ln b|` because the
initial guess `target_mid = √b` sat far from the converged Kneser
F̃[mid], leaking into a wrong basin of attraction (F̃[mid] → 0).

**Fix** (kouznetsov.rs:986-1011): smooth base-dependent cap on the
target_mid magnitude, anchored at b=e² and shrinking by 0.1 per unit of
ln|b|, clamped to [0.7, 1.5]. Empirically tracks the true Kneser
F̃[mid] across the b∈[2, 1000] range so that the LM iteration starts
inside the correct basin. Also bumped LM `max_iters` 40→80 (linear-
descent phase grows with b before Newton kicks in).

| b_re | b_im | h_re | h_im | result (20 digits) | mode |
|---|---|---|---|---|---|
| 50 | 0 | 0.5 | 0 | 3.6480… | OK |
| 100 | 0 | 0.5 | 0 | 4.2131… | OK |
| 200 | 0 | 0.5 | 0 | 4.8185… | OK |
| 500 | 0 | 0.5 | 0 | 5.6842… | OK |
| 1000 | 0 | 0.5 | 0 | 6.3913… | OK |

---

## F. Real base b = 2 (the canonical case the user flagged)   — **RESOLVED**

`b = 2` sits in the `|λ| ≈ 1.23` regime. Every probed `h` (real, complex,
positive, negative, integer, non-integer) now converges. Same fix as
Class E: smooth target_mid cap + LM max_iters=80. Previously rejected
as HANG because the 90s probe timeout caught it mid-descent — actual
convergence completes in ~110-180s at 20 digits.

| b_re | b_im | h_re | h_im | result (20 digits) | mode |
|---|---|---|---|---|---|
| 2 | 0 | 0.5 | 0 | 1.4587818160364217112 | OK (~110s) |
| 2 | 0 | -0.5 | 0 | 0.5447641214595567443 | OK |
| 2 | 0 | 1.5 | 0 | 2.7487616545898225107 | OK |
| 2 | 0 | 2.5 | 0 | 6.7213994941488631299 | OK |
| 2 | 0 | 0.5 | 1 | 1.3232 + 0.8693i | OK |
| 2 | 0 | 1 | 1 | 1.6815 + 1.0501i | OK |
| 2 | 0 | 0.3 | 0.7 | 1.2266 + 0.6120i | OK |

---

## G. Negative integer heights (well-defined ill-cases)

Mathematically undefined past `h = -1`: `F(-1) = 0`, `F(-2) = log_b(0)`.
Currently errors cleanly — kept here so the implementing AI does **not**
attempt to silently extend.

| b_re | b_im | h_re | h_im | mode |
|---|---|---|---|---|
| 2 | 0 | -1 | 0 | OK (returns 0) |
| 2 | 0 | -2 | 0 | ERR (intentional) |
| 2 | 0 | -3 | 0 | ERR (intentional) |
| e | 0 | -2 | 0 | ERR (intentional) |

---

## H. Base = 0 with non-integer height

`F_0` is well-defined only for non-negative integer heights (then it is
the 0/1 alternation). Currently ERRs — keep that contract.

| b_re | b_im | h_re | h_im | mode |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | OK (returns 1) |
| 0 | 0 | 1 | 0 | OK (returns 0) |
| 0 | 0 | 2 | 0 | OK (returns 1) |
| 0 | 0 | 0.5 | 0 | ERR (intentional) |
| 0 | 0 | -1 | 0 | ERR (intentional) |
| 0 | 0 | 1 | 0.1 | ERR (intentional) |

---

---

## I. Silent-corruption cases — **NOW CAUGHT BY FUNCTIONAL-EQUATION POST-CHECK**

Schröder's `tetrate_schroder` was producing wrong numbers without warning
when the σ̃ Taylor series's heuristic `safe_radius` exceeded its true radius
of convergence (this happens for real bases just below η, where `|λ| → 1`).

Resolved: every Schröder result is now post-validated via
`|F(h+1) − b^F(h)| / max(|F(h+1)|, 1) < 1e-6`. Failure → clean error.

### I.1 Schröder near-η — now ERR (was WRONG)

| b_re | b_im | h_re | h_im | before | after |
|---|---|---|---|---|---|
| 1.435 | 0 | 0.5 | 0 | 1.2528955 ✓ | 1.2528955 ✓ |
| 1.438 | 0 | 0.5 | 0 | 1.2542204 ✓ | 1.2542204 ✓ |
| 1.439 | 0 | 0.5 | 0 | 1.2546613 ✓ | 1.2546613 ✓ |
| 1.440 | 0 | 0.5 | 0 | 2.3082521 ✗ | ERR (validation rel=4.7e-1) |
| 1.441 | 0 | 0.5 | 0 | -5.99e+11 ✗ | ERR (validation rel=1e0) |
| 1.443 | 0 | 0.5 | 0 | -9.12e+40 ✗ | ERR (validation rel=1e0) |
| 1.444 | 0 | 0.5 | 0 | ERR | ERR (boundary band, unchanged) |

### I.2 Other silent-corruption cases — now ERR

| b_re | b_im | h_re | h_im | before | after |
|---|---|---|---|---|---|
| 50 | 0 | 0.5 | 0 | inf | OK (large-base cap formula now converges Kouznetsov) |

### I.3 Schröder degenerate F≡L solution — now caught by anchor check

Schröder's σ̃-shift can collapse to F(z)=L (the trivial fixed-point
solution): `b^L=L` makes the functional-equation check pass trivially,
so a separate anchor check `F(0)=1` is required to detect this.
Resolved at `schroder.rs:tetrate_schroder` — every result is now
anchor-validated via `|F(0) − 1| < 1e-6` before functional-equation
post-validation.

---

## Working baseline (for reference / regression checks)

These currently succeed and should remain green.

| b_re | b_im | h_re | h_im | result |
|---|---|---|---|---|
| 1.4 | 0 | 0.5 | 0 | 1.2371826705352846999 |
| 1.4142135623730950488 | 0 | 0.5 | 0 | ≈ 1.2436 (√2 inside Shell-Thron) |
| 2.71828182845904523536 | 0 | 0.5 | 0 | ≈ 1.6463 (e via Newton-Kouznetsov) |
| 2.71828182845904523536 | 0 | 0.5 | 1 | 1.0969...+1.1821...i |
| 0 | 0.5 | 0.5 | 0 | 0.8208...+0.9904...i |
| 1 | 0 | 3.7 | 1.2 | 1 (b=1 special case) |
| 2 | 0 | 3 | 0 | 16 (integer height) |
| 2 | 0 | -1 | 0 | 0 |

---

## Priority ranking for the implementing AI

1. **I.1 (Schröder near-η silent corruption)** — produces wrong numbers
   without warning across `b ∈ [1.440, 1.443]` (and likely a wider band at
   higher precision). Either tighten dispatcher to `|λ| < 0.85` *or* add
   functional-equation post-check `|F(z+1) − b^F(z)| < ε` and error on
   failure. **This is the only silent-correctness bug — fix first.**
2. **F (b = 2 non-integer h)** — user explicitly cited this; canonical "outside
   Shell-Thron, real base just past η" failure. Fix here typically generalizes
   to E (large real bases).
3. **E (large real bases)** — same algorithm class as F but stresses initial
   guess / contour height scaling. Includes I.2 (`b=50` returning inf).
4. **A (Shell-Thron boundary band)** — mathematically the hardest; needs
   parabolic-iteration theory (Écalle, Abel function), not just better solver
   tuning. Possibly accept "unsupported" forever and document.
5. **C (pure imaginary bases)** — small magnitudes already work, so this is a
   matter of generalizing the working `b=0+0.5i` path.
6. **D (general complex far from real)** — needs the Riemann mapping /
   Schwarz-Christoffel conformal map originally proposed for the Paulsen path.
7. **B (negative real bases)** — needs careful W_k branch selection; lower
   priority because the function is multi-valued and convention-dependent.
