# Failure 4-Tuples for Tetration `tet b_re b_im h_re h_im`

Classes of currently-unsupported / failing inputs, organized by failure mode.
Each row is a 4-tuple `b_re b_im h_re h_im` — feed directly to `tet <prec>` for testing.

Legend:
- **ERR** — exits cleanly with non-zero status and a diagnostic on stderr; no result printed.
- **HANG** — does not complete within 90 s at precision 50; algorithm spins / fails to produce output.
- **WRONG** — historically produced garbage (e.g. magnitudes ~1e+3000); now generally captured as ERR after the two-tier residual gate, but listed because correctness is still unsolved.

---

## A. Shell-Thron parabolic boundary band  (|λ| ≈ 1)   — **PARTIAL: low-precision iε fallback**

Schröder regular tetration converges too slowly here (logarithmic rate);
Newton-Kantorovich Kouznetsov falls into a pathological scalability trap:
`|arg(λ)|` is tiny (e.g. 0.1411 rad for b=1.45), so the strip must extend to
`t_max ≈ π/|arg(λ)| ≈ 457`, requiring `n_nodes=65536`. Each LM matvec takes
~19s; convergence on this grid takes hours.

**Fallback** (`dispatch.rs:try_iperturbation_extrapolation`) — when both the
direct and continuation Kouznetsov solvers refuse for parabolic-cap reasons,
the dispatcher pivots: it computes the perturbed values
`F(b+ε_k·i, h)` for `ε_k ∈ {0.1, 0.05, 0.025, 0.0125, 0.00625}` (which take
the OutsideShellThronGeneral / Kouznetsov path successfully because |arg(λ)|
there is no longer ≈ 0; even ε=0.00625 gives |λ|≈0.899 — well-interior),
and combines them in a Romberg-style table:

```
R₁(ε) = (4·F(ε/2) − F(ε))/3        cancels ε²  → O(ε⁴)
R₂(ε) = (16·R₁(ε/2) − R₁(ε))/15    cancels ε⁴  → O(ε⁶)
R₃(ε) = (64·R₂(ε/2) − R₂(ε))/63    cancels ε⁶  → O(ε⁸)
R₄(ε) = (256·R₃(ε/2) − R₃(ε))/255  cancels ε⁸  → O(ε¹⁰)
```

For real h, Schwarz reflection makes Re(F(b+iε)) even in ε and Im(F)
odd in ε; the Richardson table works directly on the real part and the
imaginary residue collapses to zero. For complex h the parity breaks, so
the fallback symmetrises manually:
`G(ε) := (F(b+iε, h) + conj(F(b+iε, conj(h))))/2`
which restores ε-evenness at the cost of doubling the per-ε work
(10 tetrate calls instead of 5).

Theoretical R₄ residual at ε_min=0.00625 is O(ε¹⁰) ≈ 9e-23, but in practice
higher-order Taylor coefficients a₁₀, a₁₂… in `F(b+iε)` grow rapidly near
the parabolic boundary and limit empirical accuracy to **~15–17 digits**
regardless of requested precision. Functional-equation self-check
`F(h+1) − b^F(h)` gives:

  - b=η (worst case):   relative error 6.2e-17  → ~16 digits
  - b=1.4448:            relative error 1.3e-15  → ~15 digits

At b=1.4448 the empirical R₃ vs R₄ disagreement (~1e-12) reveals that
a₈ at this base is large (~10³), consistent with the parabolic-fixed-
point coefficient blow-up. Beyond ~17 digits at adversarial bases would
need a proper Abel/Écalle parabolic-iteration theory (or Kouznetsov's
2009 Abel-function construction).
The CLI prints a stderr warning. For research-grade precision in this band
a proper Abel/Écalle parabolic-iteration theory (or Kouznetsov's 2009
Abel-function construction, Math. Comp. §6) is still needed.

| b_re | b_im | h_re | h_im | mode | result |
|---|---|---|---|---|---|
| 1.4446678610097661337 | 0 | 0.5 | 0 | OK (~16 digits via iε R₄) | 1.25715307505417... + 0i |
| 1.444667861009766 | 0 | 0.5 | 0 | OK (~16 digits via iε R₄) | 1.25715... + 0i (t710) |
| 1.4447 | 0 | 0.5 | 0 | OK (~15 digits via iε R₄) | 1.25717... + 0i |
| 1.4448 | 0 | 0.5 | 0 | OK (~15 digits via iε R₄) | 1.25721102559202735... + 0i |
| 1.4447 | 0 | 0.5 | 0.5 | OK (~15 digits via iε R₄ complex-h) | 1.29015 + 0.21136i |
| 1.45 | 0 | 0.5 | 0 | OK | continuation solver succeeds at |λ|=1.003 |
| 1.46 | 0 | 0.5 | 0 | OK | continuation solver succeeds (1.2638346… at full prec, ~5 min) |
| 1.43 | 0 | 0.5 | 0 | OK | Schröder works at this distance |
| 1.44 | 0 | 0.5 | 0 | OK | Schröder converged (|λ|=0.873, post-validation rel=6.4e-29) |

---

## B. Negative real bases  (b ∈ ℝ, b < 0)   — **RESOLVED**

All negative real bases route to Newton-Kantorovich Kouznetsov and
converge. They just need longer timeouts than real-positive bases:
- b ∈ [-1.6, -0.4]: 82–137s at 20 digits
- b ∈ [-3.6, -2.0]: 38–540s at 20 digits
- b ≈ -0.5, -0.99, -1: similar to neighbours (confirmed entering LM with valid W_k pair)

The 19^4 old grid (step=0.4) showed all of {-0.4, -0.8, -1.2, -1.6, -2.0, -2.4, -2.8,
-3.2, -3.6} work with ok=360 and 0 errors. The FAILURE_CASES.md entries were based
on an older 90s timeout before the Newton-Kantorovich solver was tuned; they no longer
represent failures.

| b_re | b_im | h_re | h_im | mode | time |
|---|---|---|---|---|---|
| -0.4 | 0 | 0.5 | 0 | OK (grid) | 91s |
| -0.8 | 0 | 0.5 | 0 | OK (grid) | 88s |
| -1 | 0 | 0.5 | 0 | OK (confirmed entering LM) | ~100s est. |
| -2 | 0 | 0.5 | 0 | OK (grid) | 221s |
| -3.6 | 0 | 0.5 | 0 | OK (grid) | 38s |

---

## C. Pure-imaginary bases  (b = i·y, y ≠ 0)   — **RESOLVED**

All pure-imaginary bases are covered:
- `y > 0`, `|y| ≤ 1.3` → Schröder (Shell-Thron interior, fast).
- `y > 0`, `|y| ≥ 1.4` → Newton-Kouznetsov. Converges, just **slow**
  (3–6 min at 20 digits). Initial "HANG" diagnosis was a short-timeout
  artifact; the LM does converge to the correct Kneser solution.
  The old 19^4 grid confirmed: b=(0,2i) ok in 288s, b=(0,3.2i) in 316s,
  b=(0,3.6i) in 293s, all 360 heights, 0 errors.
- `y < 0` → **Schwarz reflection** to `b=0+|y|i` (same as y>0 path).

| b_re | b_im | h_re | h_im | mode | result |
|---|---|---|---|---|---|
| 0 | 0.5 | 0.5 | 0 | OK | 0.821 + 0.990i |
| 0 | 1.2 | 0.5 | 0 | OK | 1.159 + 0.658i (Schröder) |
| 0 | 1.4 | 0.5 | 0 | OK (480s) | 1.243 + 0.888i |
| 0 | 2 | 0.5 | 0 | OK (289s) | — |
| 0 | 3.6 | 0.5 | 0 | OK (294s) | — |
| 0 | 5 | 0.5 | 0 | OK (~400s est.) | — |
| 0 | -3 | 0.5 | 0 | OK via Schwarz (~316s) | conj(F_{0+3i}) |

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
| -1 | 1 | 0.5 | 0 | OK (-0.0804 + 0.3593i, ~7 min via direct Kouznetsov) |
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

## J. Real bases on the cut segment  0 < b < e^{-e}   — **PARTIAL: ε-continuation walker**

**Cases:** `tet <digits> 0.04 0 h_re h_im`, `tet <digits> 0.06 0 0.5 0`, …
(any real base strictly between 0 and η_low = e^{-e} ≈ 0.0659880358…).

**Mathematical status.** On this segment the real fixed point of `b^z` is
repelling with λ real < −1 (period-doubling regime): **no real-analytic
Kneser tetration exists**. The canonical value is *defined* here as the
boundary limit `lim_{ε→0⁺} F(b+iε, h)` — the analytic continuation from the
upper half b-plane, consistent with the Schwarz-reflection convention this
program uses for `Im(b) < 0`. It is genuinely **complex for non-integer real
heights**. Any fallback that forces a real result (e.g. the Schwarz-folded
iε-Richardson used on the parabolic band) is *invalid* here; the dispatcher
routes every cut base to the ε-continuation walker from both regions that
can contain one (`OutsideShellThronRealPositive` and `ShellThronBoundary`,
see `dispatch.rs:tetrate_cut_base`).

**Why direct solves fail.** At the real base the germ-tracked fixed-point
pair is (W₀, W₊₁) — *both* in the closed upper half-plane (the generic
opposite-half-plane W_k search rejects it), with genuinely asymmetric decay
rates. Cold Kouznetsov solves sit outside the Newton basin; Schröder's σ̃
series diverges (`σ̃ Taylor radius < |1−L|`).

**Construction** (`kouznetsov.rs:setup_kouznetsov_cut_base`): anchor a clean
Kouznetsov solve at `b + 2i`, then walk ε ↓ 0 along `b + iε` with
warm-started LM solves, tracking the (W₀, W₊₁) germ. Machinery grown over
ten walk campaigns at b = 0.04:

* **Two-sided anchored log-unwrap** (`unwrapped_ln_samples`, `two_sided =
  true` — cut walker ONLY; every other base class uses the pointwise
  principal log, see section below): keeps the left-edge integrand
  continuous when the sample curve crosses `(−∞, 0]`, which it always does
  near the cut (`L_low` has `Re < 0`).
* **Shell-Thron crossing wall** (ε ≈ 1.55 → ≈ 1.0 at b = 0.04): the walk
  crosses the ST boundary, where a *winding zero* of F rides along the
  sample line. Plain warm steps stall ("no descent"); the walker recovers
  with **homotopy jumps** — warm starts perturbed by ±1 winding at up to
  three pinch points (well-separated interior local minima of |F|; near
  the ε→0 endgame SEVERAL zeros straddle the line simultaneously, observed
  at b = 0.06, ε ≈ 0.196 with |F| minima 4.6e-2 at t = −29.4 and ~1e-1 at
  t = −32.3, and a single-pinch corrector cannot reach the true class) —
  accepted only from **tight steps** (< 2 % of ε): every
  tight jump ever observed (40+) landed on the true continuation, while the
  only wrong-family "ghost" ever produced came from a since-forbidden 28 %
  coarse jump.
* **Residual gate** (uniform over plain steps and jumps): accept only
  cleanly converged solves, `residual ≤ 10^(−0.4·digits)` (1e-8 at 20
  digits). True-continuation conditioning floors RISE as the winding zero
  nears the line (observed 1.4e-21 → 5.7e-18 → 3e-14 → ~1e-12 across
  campaigns, decelerating toward a ~1e-11…1e-10 peak), while wrong-family
  stalls only ever appeared at 1.9e-7 and above; anything accepted above
  `10^-(digits+1)` prints an honesty warning with the achieved residual.
  Stagnation-accepted garbage (residual ~1e-7 … 1) is rejected and
  triggers bisection.
* **Adaptive node boost**: when the previous curve's deepest pinch has
  |F|min < 0.12 the next solve doubles its node count. A zero within ~0.1
  of the line makes the left-edge integrand ln F near-singular; at the
  standard density the trapezoidal floor then lands at the gate scale
  (observed: clean convergence flooring at 1.022e-8, b=0.06, ε≈0.102 —
  killed the walk despite correct winding class). Doubling the density
  squares the floor away; healthy pinches (|F| ≈ 0.2–0.5) never trigger.
* **Wall-band pacing**: after any rescue, the next ≤ 5 targets are fine
  (1.5 %) steps — immediately jump-eligible, ~1 solve per band step instead
  of fail → bisect cascades.

**Verified working baseline** (20 digits): `b = −0.8+0.4i` (general complex,
regression witness for the two_sided split) →
`0.70282898263600754292 + 0.82145795139882997129i`.

**Honest limits / open items:**

* The walk is expensive: hours of warm solves for deep-in-the-band bases
  (b = 0.04). Shallow bases (0.05, 0.06) cross a thinner wall.
* At `b = 0.04 + 2i` with complex heights, an older reference value
  (`0.1772+0.4972i` for h = 0.04+2i-related grids) proved **grid-dependent
  and invalid**; only FE-validated values are quotable.
* If every attempt at a band step fails the gate, the walk fails honestly
  ("bisection floor reached") rather than continuing on a suspect state.

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
| 2 | 0 | 0.5 | 0 | 1.4587818160364217112 |
| 100000 | 0 | 0.5 | 0 | 12.387261344067895865 |
| 3000 | 0 | 0.5 | 0 | 7.6097169725553975773 |
| -2 | 0 | 0.5 | 0 | 0.0484014042…+0.3116188934…i |
| -0.8 | 0.4 | 0.5 | 0 | 0.70282898263600754292+0.82145795139882997129i (relaxed-gate best-effort; accuracy-warning expected) |

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
