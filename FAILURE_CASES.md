# Failure 4-Tuples for Tetration `tet b_re b_im h_re h_im`

Classes of currently-unsupported / failing inputs, organized by failure mode.
Each row is a 4-tuple `b_re b_im h_re h_im` — feed directly to `tet <prec>` for testing.

**October 2026 audit:** old August grids, timing tables and relaxed-gate
successes below are historical observations, not present-day accuracy
certificates. Unchecked Richardson, fixed-point substitution and low-accuracy
successful fallbacks are retired. Final Kouznetsov evaluation requires the
full `10^-(digits+3)` boundary target; even that is not a forward-error proof.
Current independent numerical witnesses and their limits are in README §5.1
and `tests/phase10_honesty.rs`.

Legend:
- **ERR** — exits cleanly with non-zero status and a diagnostic on stderr; no result printed.
- **HANG** — historical 90-second budget exceeded; not proof of a deadlock or mathematical impossibility.
- **WRONG** — historically produced garbage (e.g. magnitudes ~1e+3000); now generally captured as ERR after the two-tier residual gate, but listed because correctness is still unsolved.

---

## A. Shell-Thron boundary band (|λ| ≈ 1) — **PARTIAL; unchecked fallback removed**

Schröder regular tetration can converge geometrically too slowly near the boundary;
Newton-Kantorovich Kouznetsov falls into a pathological scalability trap:
`|arg(λ)|` is tiny (e.g. 0.1411 rad for b=1.45), so the strip must extend to
`t_max ≈ (digits+8)·ln(10)/|arg(λ)| ≈ 457` at 20 digits, requiring `n_nodes=65536`. Each LM matvec took
~19s; convergence on this grid takes hours.

The former five-level iε Richardson table is **removed**. Its levels could
disagree around `1e-12` while it returned arbitrarily many digits, and an FE
residual around `1e-16` was not a forward-error certificate. The current real
boundary route tries continuation and then the existing direct Kouznetsov
method. The direct method genuinely works at `b=1.5`; its formerly skipped
fallback is a routing regression covered by t428. If neither method works,
the result is an explicit error, not a polynomial substitute.

Rebuilding committed revision `26aec7f` reproduced the `b=1.46,h=0.5`
continuation failure at its first warm step (residual about `0.0964`).
That code then returned unchecked Richardson output
`1.2638346006665197568`, not the old t880 reference
`1.2638346032868236084`. The supposed full-precision continuation regression
was already invalid; t880 now requires explicit refusal from both existing
solver paths, rather than accepting that surrogate.

| b_re | b_im | h_re | h_im | mode | result |
|---|---|---|---|---|---|
| 1.444667861009766 | 0 | 0.5 | 0 | ERR | obsolete Richardson answer rejected (t710/t871) |
| 1.4448 | 0 | 0.5 | 0 | ERR | explicit refusal at 20/50/70 digits (t870) |
| 1.4447 | 0 | 0.5 | 0.5 | ERR | no complex-height surrogate (t872) |
| 1.5 | 0 | 0.5 | 0 | numerical success | existing direct method, 10-digit regression |
| 1.46 | 0 | 0.5 | 0 | ERR | continuation and direct solve stall at 20 digits (t880); old success claim retracted |
| 1.45 | 0 | 0.5 | 0 | historical only | old continuation claims not independent accuracy evidence |
| 1.43 / 1.44 | 0 | 0.5 | 0 | historical only | old regular-iteration observations, not current certificates |

### A.1 Complex bases deep in the band — silent garbage, now gated (RESOLVED as honest ERR)

Discovered 2026-08-23 during the § 5.4 chart campaign. For
`b = 0.0653281554868594 + 0.025i` (99% of `e^{−e}` lifted by `iε = 0.025`,
`|λ| = 0.9950`, `arg λ ≈ π − 0.07` — deep in the band on the *oscillating*
side), the chain went: Schröder refuses (parabolic) → Kouznetsov LM stalls
at `‖r‖∞ ≈ 1.5` after one step ("no descent") → the old relaxed non-Schwarz
acceptance (`residual ≤ 5`!) **accepted the stalled samples** → RC=0 with a
5-digit-looking answer. The garbage passed the FE post-check (the FE is
enforced by the evaluation recurrence itself, so a *wrong base segment*
still satisfies it) and was exposed only by physics: iterating the value
up in height blew up to `10^{6913}` by `h = 50` and `inf` at 51, while the
true orbit is bounded (integer-height cross-check `F(48) = 0.1353 − 0.0070i`;
independent 10- vs 15-digit runs disagreed *completely*, the smoking gun).

The original fix used `10^{−digits/3}` clamped to `[1e-6,1e-2]`. The October
audit replaced that final-answer gate with `10^{−(digits+3)}` and removed
Richardson fallback. Relaxed states may still be inspected internally, but
`eval_kouznetsov` rejects them as answers. Cross-precision disagreement is
decisive numerical evidence; comparing an integer orbit with a nearby
noninteger height is only a diagnostic, not an accuracy certificate.

| b_re | b_im | h_re | h_im | mode | result |
|---|---|---|---|---|---|
| 0.0653281554868594 | 0.025 | 48.013 | 0 | was: RC=0 garbage (−4.31+7.57i @10dig, −20.10+11.87i @15dig) | ERR; normalization/full requested boundary target required |
| 0.0653281554868594 | 0.05 | any swept | 0 | OK (Schröder, |λ|=0.978) | 1015-point sweep, zero errors |

### A.2 The t860 "canonical value" was a discretization artifact (pseudo-verification uncovered)

Follow-up discovery (2026-08-24) while tuning the § A.1 gate. The regression
test t860 asserted `F(0.5) = 0.70282898263600754292 + 0.82145795139882997129i`
for `b = −0.8 + 0.4i` (outside Shell-Thron, `|λ| ≈ 1.15`) — a value
"verified" against baseline ac19851 during an earlier campaign. The gate
rejected the solve behind it (LM stalls at `‖r‖∞ = 1.577`), which looked
like a gate false-positive… until cross-discretization probes:

| probe (all principal-log Kouznetsov, same binary family) | F(0.5) |
|---|---|
| 20 digits (n=2048) — the "canonical" | `0.7028 + 0.8215i` |
| 22 digits (n≈2048–4096) | `−0.0276 + 0.0767i` |
| 25 digits (n=4096) | `0.7152 + 0.7635i` |
| 12 digits, two-sided unwrap | `−0.1729 − 0.2208i` |
| 20 digits, two-sided unwrap | `−0.8 + 0.4i` (echo of b — downstream fallback junk) |

Every discretization gives a **different** value; the FE post-check passes
for all of them (recurrence-enforced, see § A.1). The "canonical" value was
blessed only because ac19851 used the *same* node count at the same digits —
the agreement was a discretization fingerprint, not verification
(pseudo-verification by shared ancestry). There is currently **no**
independently verified value of tetration at this base.

Two historical numerical observations:

1. **Phantom residual component.** The left-edge integrand `log_b F` built
   with the pointwise *principal* log mis-branches for this base and pins the
   reported residual at O(1) even where samples are smooth. The anchored
   two-sided unwrap drops the same solve's residual 1.577 → 9.5e-4. The
   remaining 9.5e-4 is scale-invariant in node count (9.496e-4 @ n=4096,
   9.503e-4 @ n=8192) and spatially broad (median 5.1e-4 across all 4094
   interior nodes, peak at t ≈ −3.1) — a genuine continuous-level
   obstruction candidate, not a proof excluding discretization error. The strip geometry chosen by the
   W_k search (`k = −1`, `L_upper = 0.30+0.30i`, `L_lower = 2.26−0.41i`)
   did not yield a converged solution at this tolerance. A zero/branch-point
   is one possible diagnosis, but a sampled `|F|≈0.44` and a persistent
   residual do not prove an in-strip zero or nonexistence of a solution.
2. **Solver response.** `setup_kouznetsov` now retries a gate-rejected solve
   with the two-sided unwrap before refusing (kills phantom-only stalls at
   the retry must also pass the gate). For b = −0.8+0.4i
   both attempts stall → honest ERR listing both residuals.

t860 now requires explicit refusal of the known unverified case and its
conjugate; independently supported cases must succeed. t852 likewise
distinguishes supported unit-circle directions from a known refusal.
Neither accepts an arbitrary error or a plausible symmetric value as proof
of correctness. Alternative contours and merged-fixed-point methods are
research directions, not established repairs for a proven in-strip zero.

---

## B. Negative real bases (b ∈ ℝ, b < 0) — **NOT generally verified**

October regression: at `b=-2, h=0.4+0.1i`, 10-digit direct attempts stall
around `1e-9`, above the `1e-13` boundary target, and must return an error.
The old three-digit FE agreement was not a ten-digit result.

Historical relaxed-gate timings, **not current success guarantees**:
- b ∈ [-1.6, -0.4]: 82–137s at 20 digits
- b ∈ [-3.6, -2.0]: 38–540s at 20 digits
- b ≈ -0.5, -0.99, -1: similar to neighbours (confirmed entering LM with valid W_k pair)

The old grid's zero-error counts and "entered LM" observations did not
independently establish returned accuracy or successful convergence.

| b_re | b_im | h_re | h_im | mode | time |
|---|---|---|---|---|---|
| -0.4 | 0 | 0.5 | 0 | OK (grid) | 91s |
| -0.8 | 0 | 0.5 | 0 | OK (grid) | 88s |
| -1 | 0 | 0.5 | 0 | OK (confirmed entering LM) | ~100s est. |
| -2 | 0 | 0.5 | 0 | OK (grid) | 221s |
| -3.6 | 0 | 0.5 | 0 | OK (grid) | 38s |

---

## C. Pure-imaginary bases (b = i·y, y ≠ 0) — **PARTIAL numerical coverage**

Base `i` now has independent 50/70-digit regular-iteration witnesses.
The following broader August observations were not revalidated as a class:
- `y > 0`, `|y| ≤ 1.3` → Schröder (Shell-Thron interior, fast).
- `y > 0`, `|y| ≥ 1.4` → Newton-Kouznetsov. Converges, just **slow**
  (3–6 min at 20 digits). Initial "HANG" diagnosis was a short-timeout
  artifact in those runs; convergence alone does not identify the canonical solution.
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

## D. Complex bases far from real axis — **historical observations, not class-wide coverage**

`b = a + bi` with large `|Im(b)|` relative to `|Re(b)|`. The Newton-
Kouznetsov path historically showed quadratic convergence in selected Im(b)≥0 cases once past the
initial linear-descent phase (≥3-min at 20 digits). For Im(b)<0, the
**Schwarz reflection** `F_b(h) = conj(F_{b̄}(h̄))` reduces to Im(b)>0.

**Fix** (`dispatch.rs`): at entry, when Im(b)<0, dispatch via conjugate
base and conjugate height, then conjugate the result. This is an exact
branch convention used by the dispatcher; it does not validate the upper-half-plane answer.

| b_re | b_im | h_re | h_im | mode |
|---|---|---|---|---|
| 1.2 | 3.5 | 0.5 | 0 | OK (0.2024 + 0.5434i, ~3-4 min) |
| 0.5 | 2 | 0.5 | 0 | OK (0.2498 + 0.5270i) |
| -3.6 | -0.4 | 0.5 | 0 | **WAS HANG** → OK via Schwarz (~165s) |
| -1.2 | -1.2 | 0.5 | 0 | **WAS HANG** → OK via Schwarz (~548s) |
| -1 | 1 | 0.5 | 0 | OK (-0.0804 + 0.3593i, ~7 min via direct Kouznetsov) |
| 2 | 5 | 0 | 0.5 | OK |

---

## E. Large positive real bases (b ≫ e^(1/e)) — **selected regressions**

October: finite `F_100(1.5)` and `F_1000(1.5)` were wrongly rejected because
their auxiliary successors overflowed. The FE check now uses a predecessor
where possible, preserving finite requested values. The 10-digit regressions
pass; the historical values below are not independent 20-digit certificates.

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

## F. Real base b = 2 — **independent 50-digit witness; explicit higher-precision limit**

At `h=0.5`, the October 50-digit result
`1.4587818160364217006839716610385871352966066053309` differs relatively by
`4.90e-51` from the independent fatou.gp reference. The old
`1.4587818160364217112` anchor below is inaccurate after about 16 digits.

At 70 digits direct setup requests 65536 nodes, above its 32768 budget.
The 1800-second CLI run timed out during continuation from the `b=2.35`
anchor, not after reaching `b=2`. No 70-digit base-2 accuracy claim follows.
Node-budget failure is not mathematical nonexistence or proof that Abel
theory is needed; t965 checks that distinction at 70/1000 digits.

Historical observations: `b=2` sits in the `|λ|≈1.23` regime. Same old seed fix as
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

For nondegenerate bases, the selected recurrence is undefined past `h=-1`:
`F(-1)=0`, `F(-2)=log_b(0)`. Base one is the constant-function exception.
Currently errors cleanly — kept here so the implementing AI does **not**
attempt to silently extend. Cached evaluators enforce the same domain:
otherwise roundoff in `F(-1)` can turn `log(0)` into a plausible finite value
(reproduced for cached `b=0.5,h=-2`; fixed in the October audit).

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

## I. Silent-corruption cases — **consistency checks, not accuracy proofs**

Schröder's `tetrate_schroder` was producing wrong numbers without warning
when the σ̃ Taylor series's heuristic `safe_radius` exceeded its true radius
of convergence (this happens for real bases just below η, where `|λ| → 1`).

Resolved: every Schröder result is now post-validated via
`|F(h+1) − b^F(h)| / max(|F(h+1)|,1)` against a working-precision MPFR
tolerance, not a fixed `1e-6` floor. Failure is an error; passing is not proof.

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
anchor-validated with a working-precision tolerance before FE validation.

---

## J. Real bases on the cut segment  0 < b < e^{-e}   — **PARTIAL: ε-continuation walker**

**Cases:** `tet <digits> 0.04 0 h_re h_im`, `tet <digits> 0.06 0 0.5 0`, …
(any real base strictly between 0 and η_low = e^{-e} ≈ 0.0659880358…).

**Mathematical status.** On this segment the real fixed point of `b^z` is
repelling with λ real < −1 (period-doubling regime). Pinch/zero diagnoses
below are historical numerical hypotheses, not certified zeros or a global
solvability argument. The intended experimental
branch is the boundary limit `lim_{ε→0⁺} F(b+iε, h)`, where it exists — continuation from the
upper half b-plane, consistent with the Schwarz-reflection convention this
program uses for `Im(b)<0`. The selected regular branches are generally
complex at noninteger real heights; forcing reality is not justified.
After regular iteration fails, both dispatch regions that
can contain one (`OutsideShellThronRealPositive` and `ShellThronBoundary`,
see `dispatch.rs:tetrate_cut_base`) route to the walker. This audit establishes
neither equivalence of all selected branches with that limit nor a walker endpoint.

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
  true` — also used by ordinary complex-base retries): tracks the left-edge integrand
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
  |F|min < 0.12 the next solve doubles its node count; below 0.05 it
  quadruples (n=16384, still ≤ N_MAX_PRACTICAL). A zero within ~0.1
  of the line makes the left-edge integrand ln F near-singular; at the
  standard density the trapezoidal floor then lands at the gate scale
  (observed: clean convergence flooring at 1.022e-8, b=0.06, ε≈0.102 —
  killed the walk despite correct winding class; then jump rejections
  skating at 1.04–1.07e-8 on the doubled grid at ε≈0.089–0.092).
  Boosting squares the floor away; healthy pinches (|F| ≈ 0.2–0.5)
  never trigger.
* **Reactive near-miss escalation** (added at the ε ≈ 0.068 wall,
  b = 0.06): the static |F|min tiers can miss — at ε ≈ 0.068 clean
  quadratic descents floored at 2.0–2.1e-8 (vs gate 1e-8) with |F|min
  just *above* the 4× threshold, so only the 2× tier fired and every
  combo was rejected as it skated the gate. Now a rejected solve whose
  residual is a *near-miss* (finite, ≤ 10³ × the clean gate — the
  signature of a resolution floor; ghost stalls sit at O(0.1–1)) is
  retried once at doubled node tier (up to 8× = 32768 nodes) before
  the walker moves on to bisection.
* **Wall-band pacing**: after any rescue, the next ≤ 5 targets are fine
  (1.5 %) steps — immediately jump-eligible, ~1 solve per band step instead
  of fail → bisect cascades.

**Retracted baseline:** the former `b=−0.8+0.4i` value
`0.70282898263600754292+0.82145795139882997129i` was a discretization
artifact, not a working baseline. See A.2.

**Honest limits / open items:**

* The walk is expensive: hours of warm solves for deep-in-the-band bases
  (b = 0.04). Shallow bases (0.05, 0.06) cross a thinner wall.
* At `b = 0.04 + 2i` with complex heights, an older reference value
  (`0.1772+0.4972i` for h = 0.04+2i-related grids) proved **grid-dependent
  and invalid**; FE agreement alone cannot validate a replacement value.
* If every attempt at a band step fails the gate, the walk fails honestly
  ("bisection floor reached") rather than continuing on a suspect state.

---

## Historical comparison list (not a present-day pass contract)

Use the independently sourced fixtures in phase10 and explicit refusal
contracts instead. This list preserves historical observations, not
certified digits or current success for every row.

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
| -2 | 0 | 0.5 | 0 | old low-accuracy output is not a verified reference |
| -0.8 | 0.4 | 0.5 | 0 | old value retracted; explicit refusal required |

---

## Research priorities after the correctness audit

Investigate validated error/conditioning bounds, high-precision real-base
methods, sectorial parabolic constructions, and controlled complex-base
contours/limits. Preserve independent references and explicit branch/domain
contracts. README §8.1 separates promising directions from proven coverage;
none of these new constructions was implemented by this audit.
