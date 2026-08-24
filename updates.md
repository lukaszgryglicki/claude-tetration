# Status Update — 2026-08-23

Goal recap: cover **all complex bases and all complex heights** for arbitrary-precision
tetration, or document honestly why a region cannot be covered. Standard validation
target: 20 digits (precision always a modest multiple of machine precision, never
100+ digits).

This update covers the work since resuming the effort (uncommitted layer on top of
`566c753`), current walk campaigns, and an honest assessment of what remains.

---

## 1. Fixed and verified (regressions + real bugs)

### 1.1 General-complex-base regression (t860) — FIXED, root-caused to 4 stacked changes

Witness: `tet 20 -0.8 0.4 0.5 0` (base −0.8+0.4i). It failed under the committed
cut-base research layer. Systematic diff against baseline `ac19851` found **four
independent causes**, all now fixed:

1. **Two-sided anchored ln-unwrap applied globally.** The cut-base walker needs a
   two-sided (bi-asymptotic) unwrap of `ln(F−L)` samples; applying it to ordinary
   bases broke the principal-branch fast path. Fix: `two_sided: bool` plumbed
   through the whole Kouznetsov chain (`unwrapped_ln_samples`, `cauchy_eval`,
   `apply_t`, `apply_t_fft`, `iterate_newton/anderson/picard`,
   `find_normalization_shift`, `eval_at_height`, `setup_kouznetsov_core`,
   `KouznetsovState`). Convention: `false` (principal) everywhere except the
   cut-base ε-walker, which passes `true`.
2. **W_k partner-search criterion changed.** The argλ-sign ("decay_ok") criterion
   picked wrong partner k=+2 instead of k=−1. Reverted to exact baseline
   (opposite Im-half-plane criterion). The cut walker injects its fixed-point pair
   directly and never uses this search, so the baseline logic is strictly correct.
3. **`t_max` sizing changed** (`.min(argλ_lower)` doubled grid to n=4096).
   Now gated on `two_sided` — baseline behaviour (λ_upper only) restored for
   ordinary bases.
4. **`validate_best_residual` non-Schwarz threshold clamped to 1e-2** — rejected
   baseline's best-effort acceptance and forced a failing iε fallback. Restored
   baseline threshold 5.0 (best-effort + honest accuracy warning). Walker safety
   does not rely on this (see §1.3).

Result: witness matches baseline **exactly to 20 digits**
(`0.70282898263600754292+0.82145795139882997129i`), and the full
`cargo test --release` suite is GREEN, including t860.

### 1.2 Dispatch bug: band-region cut bases returned WRONG answers with RC=0 — FIXED

`b=0.06` classifies into the ShellThronBoundary region (|λ|≈1 band), whose
real-base arm never checked `is_cut_base` — it went to Schwarz-folded iε
Richardson, which **forces real values**. On the cut segment the canonical value
is complex (lim ε→0⁺ from Im(b)>0, W₊₁ germ), so it silently returned a
real-valued answer with ~1e-2 error and exit code 0. This is worse than failing.

Fix: extracted shared `tetrate_cut_base()` in `src/dispatch.rs`; **both** region
arms (OutsideShellThronRealPositive and ShellThronBoundary) now route cut bases
to the ε-continuation walker. Verified `b=0.06` now routes correctly.

### 1.3 Uniform walker residual gate — safety hardening

The walker's residual gate previously applied only to jump steps; plain (sgn==0)
steps could accept stagnated O(1)-residual solves. Made the gate uniform over all
steps. Observed it correctly rejecting a residual-0.849 stagnation acceptance
that would previously have silently corrupted a walk. Any acceptance above
`10^-(digits+1)` prints an honesty warning with the achieved residual.

### 1.4 b=3000 precision restored

The threshold clamp (§1.1 item 4) had also degraded `b=3000, h=0.5` to 13–14
digits. Now exact to 20 digits: `7.6097169725553975773`.

### 1.5 Housekeeping

* Anchor-snap experiment removed from `unwrapped_ln_samples` (contradicted
  anchored semantics; broke 2 unit tests; never helped t860). Lib tests 5/5.
* Debug probes `examples/probe_kouz.rs`, `examples/probe_warm.rs` deleted.
* clippy warnings at baseline (13). Full test suite GREEN.

### 1.6 Complex-base stalled-solve acceptance — garbage with RC=0 — FIXED

Found 2026-08-23 by the chart campaign (details: FAILURE_CASES § A.1).
`iterate_newton`'s non-Schwarz acceptance gate was `residual ≤ 5` — meant
to let walker/continuation internals inspect near-miss solves, but it also
let the *direct* complex-base path return stalled O(1)-residual samples as
final answers. Witness: `tet 10 0.0653281554868594 0.025 48.013 0`
(|λ|=0.995, deep parabolic band) returned −4.31+7.57i at RC=0 (10- and
15-digit runs disagreed completely; true value 0.1353−0.0070i by integer
-height iteration). Fix: `setup_kouznetsov` re-gates the final residual at
`10^(−digits/3)` clamped [1e-6, 1e-2] for non-Schwarz bases; walker and
continuation paths (which self-gate) are untouched. Such bases now fall
through to iε-Richardson and, if that also fails (deep-band probes land
back in the band), ERR cleanly with the full chain.

### 1.7 t860 "canonical value" exposed as discretization artifact — tests rewritten

Follow-up to § 1.6 (2026-08-24, full anatomy: FAILURE_CASES § A.2). The
honesty gate "broke" t860/t852 — investigation showed the tests were
asserting garbage. The t860 witness `F(0.5) = 0.7028…+0.8215…i` at
`b=−0.8+0.4i` came from an LM solve stalled at residual 1.577; the value
was "verified" against baseline ac19851, which used the same node count at
the same digits — pseudo-verification by shared ancestry. Cross-
discretization probes give completely different values (20d: 0.70+0.82i,
22d: −0.03+0.08i, 25d: 0.72+0.76i, two-sided 12d: −0.17−0.22i). No
independently verified value exists at this base today.

Two code changes:
1. The stall at this base is partly a **phantom residual**: the pointwise
   principal log mis-branches the left-edge integrand; the anchored
   two-sided unwrap drops the reported residual 1.577 → 9.5e-4 on the
   same-quality samples. `setup_kouznetsov` now retries a gate-rejected
   solve with the two-sided unwrap before refusing (the retry must
   independently pass the gate — zero regression risk for passing bases).
2. The residual 9.5e-4 is genuine and node-count-invariant (n=4096 and
   n=8192 agree to 3 digits): the rectangle-Cauchy equation has no
   solution on this strip (|F| dips to 0.44 near the sample line —
   in-strip zero ⇒ left-edge log crosses a cut). Both attempts stall ⇒
   honest ERR.

t860/t852 rewritten to honesty semantics: Ok ⇒ Schwarz symmetry / FE must
hold to ≥15 digits; Err ⇒ must be the parabolic-band-stall/unsupported
error and the conjugate base must refuse symmetrically. Producing the old
"canonical" number is now itself a failure. README § 5.3/§ 7/§ 8 updated.

### Verified coverage battery (all exact to 20 digits unless noted)

| base | height | result | status |
|---|---|---|---|
| 2 | 0.5 | 1.4587818160364217112 | exact, matches baseline |
| 1e5 | 0.5 | 12.387261344067895865 | exact |
| 3000 | 0.5 | 7.6097169725553975773 | exact (was 13–14 digits) |
| −2 | 0.5 | 0.0484014042+0.3116188934i | pass |
| i | 0.5 | 1.1667009135+0.7345635369i | pass |
| −0.8+0.4i | 0.5 | 0.70282898263600754292+0.82145795139882997129i | exact (t860, was FAILING) |

---

## 2. Cut segment 0 < b < e^{−e} ≈ 0.0660 — the last uncovered region

This is the only remaining gap. The Schröder path is too slow on the parabolic
band; the direct approach is an ε-continuation walk: solve Kouznetsov at
b+iε₀ (anchor ε₀≈2), then walk ε↓0 with warm-started bi-asymptotic solves,
tracking the W₊₁/W₋₁ fixed-point pair.

### 2.1 Residual-gate tuning saga (this session)

True-continuation solves near the "winding zero" (a zero of F−L drifting toward
the sample line) hit **conditioning floors that rise as the walk descends**:

| campaign | gate | died/stalled at | floor observed |
|---|---|---|---|
| walk8 | 1e-15 | ε≈0.99 | 5.7e-18 |
| walk9 | 1e-15 | ε≈1.065 | 1.6–4.2e-14 (gate-rejected true continuations) |
| walk10 | 1e-12 | ε≈0.889 | 9.6e-13 accepted |
| cut006b (b=0.06) | 1e-12 | ε≈0.917 | 1.7e-12 gate-rejected a true continuation |
| **walk11 / cut006c** | **1e-8** | see §2.2 | up to 7.6e-10 / 6.2e-9 accepted |

Final gate: `10^(−0.4·digits)` = 1e-8 at 20 digits. Rationale: observed
true-continuation floors peaked at 6.2e-9; the nearest wrong-family stall ever
observed sits at 1.9e-7 (18× margin); the only true ghost ever seen (7e-14) came
from a since-forbidden 28% coarse jump — tight-only jumping (step <2% of ε) is
the primary ghost filter, and all 40+ tight jumps ever observed were true
continuations. Everything accepted above 1e-21 carries an honesty warning.

### 2.2 NEW RECORD and a NEW wall (b=0.06, run cut006c)

With the 1e-8 gate, the b=0.06 walk **passed the winding-zero band entirely**
(previous best: death at ε≈0.917) and descended to **ε≈0.196** — 4.7× deeper
than any previous campaign — before failing in a qualitatively new way:

* Failure mode: LM solver **O(1) residual stall** (2.3e0, "no descent step"),
  not a conditioning floor. Bisection to the floor ratio didn't help. RC=1.
* Root cause (from log forensics, superseding the earlier argλ→π guess):
  **two zeros of F straddle the sample line simultaneously** near the death
  point (|F| interior minima 4.6e-2 at t=−29.4 and ~1e-1 at t=−32.3). The
  homotopy-jump rescue applied a ±1 winding corrector only at the single
  deepest pinch; when the winding class drifts at the *other* zero (or at
  both), no reachable warm-start class matches the true continuation and
  every attempt stalls at O(1). The contour decay itself is healthy
  (t_max=45.97 from argλ_low≈−1.40; argλ_up→π only speeds upper decay).
* Fix implemented (this session): **multi-pinch rescue** — detect up to 3
  well-separated interior |F| minima, try ±1 correctors at each singly and
  all four sign pairs at the two deepest, remember the winning pattern
  across steps. Relaunched as cut006d: **passed the double-zero wall** at
  ε≈0.196 (winning corrector: pair −1@t=−29.4, +1@t=+46.0) and set a new
  record ε≈0.102 before dying to a THIRD, again distinct, wall:
* **Resolution wall at ε≈0.102** (cut006d): the solve converged cleanly but
  floored at residual 1.022e-8 — 2.2% above the gate. Pure conditioning:
  a zero of F sits at distance ~0.09 from the line (|F|min=4.2e-2), the
  left-edge integrand ln F is near-singular there, and the trapezoidal
  floor at n=4096 lands exactly at the gate scale. Not a class problem
  (all 12 corrector combos floored identically).
* Fix implemented: **adaptive node boost** — when the previous curve's
  deepest pinch has |F|min < 0.12, double the node count for the next
  solve (n 4096→8192; floor squares away, ~1e-16). Healthy pinches
  (0.2–0.5) never trigger it. Relaunched as walk13 (b=0.04) and cut006e
  (b=0.06) with 16h budgets.
* **2026-08-23 ~17:45 UTC — RECORD BROKEN (cut006e): ε ≈ 0.092.** The
  node-boost binary passed the ε=0.102 resolution wall (boost fired,
  n=8192 through the deep-pinch zone, residuals 3e-10–9e-10, honest
  warnings printed) and entered unexplored territory. Currently in
  turbulence approaching the second ST crossing (~0.08): steps toward
  0.088 rejected at O(0.1) residual (wrong-class signature), walker
  bisecting per design. All prior walls crossed in one run: winding band
  (gate), 0.196 double-zero (multi-pinch, pattern [0,-1,0] dominant),
  0.102 resolution (boost).

`walk11` (b=0.04, single-pinch binary) was stopped at ε≈1.24 and relaunched
as `walk12` under the multi-pinch binary; its previous frontier was 0.889.

* **2026-08-23 evening — FOURTH wall at ε ≈ 0.068 (cut006e) + fix.** Past
  the second ST crossing the walk hit a new pure-resolution wall: clean
  quadratic LM descents flooring at 1.999–2.056e-8 vs the 1e-8 gate,
  with |F|min sitting just *above* the 0.05 static 4×-tier threshold —
  so only the 2× tier fired, and 59 solves were rejected as near-misses
  over the run's lifetime. Fix: **reactive escalation** — a rejected
  solve whose residual is finite and ≤ 10³× the clean gate (resolution
  -floor signature; ghosts stall at O(0.1–1)) is retried once at doubled
  node tier, up to 8× = 32768 nodes. cut006e was killed and relaunched
  as **cut006f** under the new binary (16 h budget); walk13 (b=0.04)
  continues under the older static-tier binary for comparison.
* **2026-08-24 — cut006f outcome: timeout at ε ≈ 0.171 (16 h), no new
  record.** The from-scratch re-walk paid the whole wall band again in
  fine ~1 % steps and ran out of clock mid-solve (last accepted
  ε = 0.1740; the in-flight 0.1713 solve was at 6.9e-10 and converging
  when killed). Restart-from-zero economics are what killed both this
  and walk13 → checkpoint/resume implemented (§ 2.4). Relaunched as
  **cut007** (b=0.06, 20 digits, `TET_MT=6`, checkpointed, no external
  timeout); walk14 (b=0.04, 20 digits, `TET_MT=8`, checkpointed)
  running in parallel.

### 2.3 Honest assessment of the remaining gap

* **What is covered**: everything except real bases in (0, e^{−e}) at the ε→0
  endpoint. Complex bases arbitrarily close to the cut work (the walk itself IS
  a sequence of successful solves at b+iε down to ε≈0.2 — e.g. tet(0.06+0.2i)
  is solved cleanly with 20-digit residuals away from the band).
* **The wall is a rescue-reachability limit, not (yet) a structural one**:
  multiple zeros straddling the line multiply the winding classes; the
  implemented multi-pinch rescue enlarges the reachable set. If walks still
  die with all combos stalling, next candidates:
  1. ±2 windings at a single pinch (class change of 2 at one zero).
  2. Contour deformation away from the zeros (bend the sample line).
  3. **Accept partial coverage**: document that cut-base values are computable
     for Im(b)=ε down to the deepest ε reached and extrapolation below that is
     unverified.
* Richardson extrapolation of the walk values ε→0 is possible but would carry
  no internal residual proof — it would be reported with an honesty warning.

---

### 2.4 Walker checkpoint/resume (`TET_KOUZ_CUT_CKPT`)

Deep walks are multi-hour and were all-or-nothing: walk13 (b=0.04, 20
digits) died to an external timeout mid-solve at ε≈1.006 after ~7 h with
nothing salvageable. Now every accepted step atomically serializes the
full continuation state (base, digits, ε, branch args, t_max, fixed-point
pair, nodes/weights/samples at full decimal precision) to the file named
by `TET_KOUZ_CUT_CKPT`; on startup a checkpoint matching `b` (1e-12 rel)
and digits resumes from the saved frontier — anchor and all crossed walls
are never re-paid. Mismatch/corruption ⇒ silent cold start; save I/O
errors never kill a walk. Round-trip verified live (kill mid-walk →
resume from saved ε, not from the ε₀=2 anchor).

---

### 2.5 Dense multi-view 3D gallery + share-ready hero JPG (§ 5.4)

The first-draft charts (~1000 pts) drew the period-2 weave as 8-segment
polygons (one winding per Δx = 2 vs a 0.25 step). Re-swept all three bases
at 5× density — 3 × 5073 points, **zero solver errors** — and rewrote
`plot3d.py` as a true orthographic turntable renderer (arbitrary az/el,
painter-sorted depth shading, isotropic complex plane so spirals render as
true circles, `--xrange` region crops). 16 dense views now in
`docs/charts/` (oblique per-base + overlay, turntable az ∈ {12,55,75,90},
el=62 top, end-on weave/forest portraits — the "swirling circles" — region
close-ups, seam) plus `tet3d_hero.jpg` (3000×1875 raster, near-axial
vortex view) for sharing. `scripts/chartgallery.sh` reproduces everything;
`chartgen.sh` gained a numeric step-multiplier arg (0.2 = dense).

---

## 3. Files changed (uncommitted, on top of 566c753)

* `src/kouznetsov.rs` — two_sided plumbing; baseline restorations (W_k search,
  t_max, validate threshold); uniform residual gate + final 1e-8 tuning with
  full floor-history comment; multi-pinch homotopy rescue (up to 3 pinch
  points, sign combos, pattern memory); adaptive node boost near deep
  pinches (2-tier: 2× below |F|min<0.12, 4× below 0.05); anchor-snap removal;
  MT-mode parallel branches (element-wise maps in `precompute_dt_factors`,
  `apply_dt_v_fft` pre-scale, `apply_t_fft` edge build + boundary rows);
  reactive near-miss node-tier escalation in the walker attempt loop
  (§ 2.2 fourth wall); complex-base stalled-solve honesty gate in
  `setup_kouznetsov` (§ 1.6) with two-sided-unwrap retry (§ 1.7); walker
  checkpoint/resume `save_cut_ckpt`/`load_cut_ckpt` + resume-aware anchor
  block (§ 2.4).
* `src/mt.rs` — NEW: opt-in MT mode (`TET_MT` env; unset/0 = serial default,
  1 = all cores, n≥2 = fixed pool). Parsed once via OnceLock.
* `src/fft.rs` — `fft()` dispatches serial (default, original code moved
  verbatim to `fft_serial`) vs `fft_mt` (parallel butterflies over disjoint
  pairs + process-global twiddle cache built by the identical sequential
  recurrence → bit-identical outputs); pointwise products in `convolve` /
  `cross_correlate_with_kernel` gated the same way.
* `src/main.rs` — `mt::init_pool()` at startup (no-op unless TET_MT ≥ 2).
* `src/lib.rs` — `pub mod mt`.
* `scripts/plot3d.py` — NEW: dependency-free SVG 3D-projection plotter for
  chart sweeps (docs/charts/); breaks curves at non-finite / |f|>50 points.
* `scripts/chartgen.sh` — NEW: adaptive-grid sweep driver (x ∈ [−30, 120],
  ~1015 points, 14-way parallel) emitting the CSVs under docs/charts/data/.
* `docs/charts/` — NEW: 5 SVG charts + 4 sweep CSVs (README § 5.4 gallery).
* `src/dispatch.rs` — `tetrate_cut_base()` helper; both region arms route cut
  bases to the walker.
* `FAILURE_CASES.md` — section J (cut segment: math status, construction,
  honest limits, gate documentation); expanded working-baseline table.
* Deleted: `examples/probe_kouz.rs`, `examples/probe_warm.rs`.
* Test status: `cargo test --release` GREEN with t860/t852 rewritten to
  honesty semantics (§ 1.7) and t890 added; lib tests 5/5; clippy = 13 =
  baseline.
* MT A/B verification (2026-08-23): `TET_MT=16` vs serial on
  `tet 20 2 0 0.5 0` (Kouznetsov), `tet 20 0 1 0.5 0` (complex base),
  `tet 15 0.5 0 0.75 0` (Schröder) — stdout **bit-identical** in all three;
  lib tests 5/5 with and without the twiddle cache. Benchmark (2026-08-24,
  16-core box, moderate background load, `tet 30 2 0 0.5 0`): serial 732 s,
  `TET_MT=4` 362 s (2.0×), `TET_MT=16` 167 s (**4.4×**); 25-digit outputs
  `diff`-identical serial vs MT=16.

## 4. Next steps

0. OWNER DECISION (2026-08-23): Écalle/Abel parabolic implementation and
   Paulsen–Cowgill conformal machinery are DESCOPED (weeks/months-scale).
   Both documented in detail in README §8.1 (verdicts, estimates, expected
   problems, research directions). Remaining active work = cut segment only.

1. Let walk13 (b=0.04) and cut006e (b=0.06) run under the multi-pinch +
   node-boost binary. Walls crossed so far: winding-zero band (gate 1e-8),
   double-zero class wall at ε≈0.196 (multi-pinch), resolution wall at
   ε≈0.102 (node boost). If a new distinct wall appears, iterate; if walks
   reach ε=0, proceed to validation.
2. Update FAILURE_CASES.md section J with the final outcome; mark J RESOLVED
   or PARTIAL accordingly.
3. If a walk reaches ε=0: validate FE internally + mpmath cross-check, anchor
   robustness check (TET_KOUZ_CUT_ANCHOR=1.8), spot-check complex/negative
   heights, add results to the baseline table.
4. (2026-08-23 14:20 UTC re-verification, current build) All battery witnesses
   re-confirmed: 2/1e5/3000/√2/i/e exact to 20 digits, 3000@30 digits exact.
   Two observations on hard complex bases:
   - b=−2: honest warning, ~8 certified digits (residual 6.44e-9) — value's
     leading 8 digits match reference; acceptable, documented in README §5.2.
   - b=−0.8+0.4i (t860): final value still exact to all 20 reference digits,
     but the run now takes ~10 min (multi-start retries) and the warning
     quotes the residual of a FAILED attempt (1.58, "0 digits") rather than
     the accepted solve — over-pessimistic honesty reporting. Value correct;
     warning-attribution fix is a pending polish item (do not silence, just
     attribute the right residual).
5. Remove `/tmp/tet-base` worktree; archive walk logs; commit everything with
   the Co-authored-by trailer.
6. README.md added (2026-08-23): full academic-audience documentation with §0
   Tetration Forum credits (forum thread tid=1826 is the project's companion
   thread), verified-value examples only, coverage map with
   Verified/Pending/Known-bad sections. Keep in sync with walk outcomes.
