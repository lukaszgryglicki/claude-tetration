# Failure 4-Tuples for Tetration `tet b_re b_im h_re h_im`

Classes of currently-unsupported / failing inputs, organized by failure mode.
Each row is a 4-tuple `b_re b_im h_re h_im` — feed directly to `tet <prec>` for testing.

**October 2026 audit:** old August grids, timing tables and relaxed-gate
successes below are historical observations, not present-day accuracy
certificates. Unchecked Richardson, fixed-point substitution and low-accuracy
successful fallbacks are retired. Final Kouznetsov evaluation requires the
full `10^-(digits+3)` boundary target; even that is not a forward-error proof.
Current independent numerical witnesses and their limits are in README §5.1
and `tests/phase10_honesty.rs` / `tests/phase12_coverage.rs`.

In the October 5 bounded coverage pass, t852/t860 did not finish within external
300-second budgets, so their known-refusal contracts below remain unverified
on that frozen build. t428 completed its extended run; the previously cancelled
t423/t870/t872 were not restarted. A test timeout is not a solver refusal.

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

The October 5 inverse-Poincare construction removes the blanket exclusion of
strict attractors in this band. Independent 100-digit references now cover
`b=1.444666`, `b=0.0665`, a complex multiplier near `0.997i`, and the old
deep-band witness below. Genuinely neutral cases remain separate research;
arbitrarily close attraction does not imply practical completion time.

The October 6 Taylor-jump accelerator makes the positive-real-multiplier
near-neutral cases practical without selecting a parabolic surrogate.
t710/t871 now pass normally, and phase13 checks independent 100-digit
real/complex-height references, including a multiplier gap near `1.58e-30`.

The subsequent complex-coefficient engine extends guarded jumps to genuinely
complex strict attractors. Phase14 checks independent 100-digit real/complex
heights at `b=1.444667861009766+1e-15i`, and 50/70-digit half-heights at a
closer complex base with multiplier gap about `1.37e-20`. Component-aware
conditioning repairs an initially incorrect tiny imaginary component.
The paired-iterate extension also addresses strictly attracting multipliers
near `-1`, including real ones. Phase15 checks independent 20/50/100-digit
real/complex heights at `b=0.0659880358454` and
`b=0.0659880358454+1e-14i`, with multiplier gap about `2.44e-13`.
The deeper `b=0.0659880358453125370767901876+1e-40i`, gap about
`8.79e-30`, has independent reference data, but its preliminary 50-digit
CLI probe produced no output within an external 1800-second budget.
That is unverified production coverage, not a mathematical refusal.
These remain strictly attracting constructions:
genuinely neutral cases and the outside/cut constructions are not covered by
this acceleration. Moderate rotating/oscillating cases can cost
more than scalar iteration. Still closer complex probes with gap about
`1.87e-30` produced no output within external 600-second budgets; they are
unverified, not known mathematical failures.

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

**November 12:** real bases above `e^{1/e}` now use the theta-mapping solver
(README § 6.8) before any Cauchy solve. `b=1.46`, where both older solvers
stalled, takes about 3 seconds at 15 digits; three contours agree to 30 digits
(`F(0.5)=1.26383460066838162179258037381`, t880). `b=1.4448` takes about
8 minutes at 20 digits: `1.2572110255920272276` on two contours, and on the
line through the regular values `1.2571233116014142057` (`b=1.4446`) and
`1.2571496273469669777` (`b=1.44466`). The old Richardson output for `1.46`
(`1.2638346006665197568`) was correct to 12 digits only, the old t880
reference (`1.2638346032868236084`) to 9. Cost grows like
`1/(b−η)`; `b=1.4447` was not finished and `b=η` is unsupported.

**November 13:** next to `e^{1/e}` the theta-mapping solver sums the Poincaré
series on a disk that reaches most of the way to the conjugate fixed point
(README § 6.8). Same digits as before, at a cost like `(b−η)^(−1/2…−2/3)`:
`1.4448` 13 s at 15 digits (210 s) and 17 s at 20 (8 minutes); `1.4447`
`1.2571671704399650301` in 35 s; `1.44467` `1.2571540131755165475` in 2.6
minutes; `1.444668` `1.2571531360127560364` in 13 minutes.
With `1.2571530750541725636` at `b=1.444667861009766+1e-15i` (regular
iteration) these lie on one line of slope `0.438` in `b`: the two solvers
meet across `η`.

**November 13, complex bases on the ray `b = exp(t·e^{−t})`, `t = r·e^{i}`,
height `0.5`** (15 digits; bases given to 12 digits, values to 10):

| `r = \|λ\|` | b | before | now |
|---|---|---|---|
| 0.99 | `1.76325171225+0.17042904825i` | `1.461036997+0.04239393851i` (regular) | same |
| 0.992 | `1.76463085849+0.169007606432i` | regular, but discontinuous in the height: `1.5688−1.5661i` on `0.56…0.60` | `1.535381517+0.07816583346i` at `0.60`; continuous (§ I.5) |
| 0.995 | `1.76668937045+0.166862470066i` | `1.523673266−1.536641996i` (wrong branch on `0.46…0.76`) | `1.472187315+0.02903216044i` |
| 0.998 | `1.76873553494+0.164701833467i` | `1.386200173+0.05716750805i`: the two-fixed-point function (§ I.4) | `1.479194320+0.01694578894i` (regular; 0.3 s) |
| 0.999 | `1.76941482678+0.163978190153i` | the two-fixed-point function | `1.479974723+0.01347300135i` (regular; 2 s) |
| 0.9999 | `1.77002500256+0.163325448409i` | `1.386596989+0.05664930179i`: the two-fixed-point function (100 s) | `1.467257287+0.007326473264i` (regular; 45 s) |
| 0.99999 | `1.77008595817+0.163260098103i` | not tried | `1.464898331+0.004241706107i` (regular; 9 minutes) |
| 1.0001 | `1.77016044416+0.163180206721i` | timeout | more than ten minutes |
| 1.001 | `1.77076924149+0.162525774232i` | timeout | `1.386825937+0.05634871966i` (30 s) |
| 1.005 | `1.7734612812+0.159600507187i` | timeout (150 s) | `1.387653600+0.05525220253i` (7 s) |
| 1.01 | `1.77679451634+0.155905852021i` | `1.388677408+0.05387391870i` | same |

Inside, the rows from `0.99` to `0.998` were first computed with the short
inverse series and chains of up to 6400 traced levels (`0.998`: 3.5
minutes); the long series of README § 6.3 gives the same digits in a
fraction of a second and reaches `0.999` and `0.9999`, where the 15- and
25-digit results agree and `--slog` returns the height `0.5` to 25 digits,
and `0.99999`. Other arguments at 15 digits: `|λ| = 0.999` at argument 2
(`0.452995041531+1.95805392380i`) `1.22588546629+1.19464891798i` and at
argument 0.3 (`1.46911361352+0.00269860383680i`)
`1.22843150003+0.0670942796200i`, half a second each; `|λ| = 0.9999` at
argument 2.5 (`−0.247756897566+0.416984023274i`)
`1.02750968310+1.60411872078i`, two minutes.
Outside, the direct start of the two-sided solver was rejected by a test
that took the slow spiral of the logarithmic orbits for a failure to
approach the fixed point, and the continuation in the base then ran into the
region; the test now compares closest approaches (README § 6.10).

| b_re | b_im | h_re | h_im | mode | result |
|---|---|---|---|---|---|
| 1.444667861009766 | 0 | 0.5 | 0 | accelerated regular iteration | t710/t871 pass; independent reference checks through 100 digits in phase13 |
| 1.444667861009766 | 1e-15 | 0.5 | 0 | complex Taylor jumps | independent 100-digit components, including the tiny imaginary part |
| 1.444667861009766133658339108596430223058595453242253165820522 | 1e-40 | 0.5 | 0 | complex Taylor jumps | independent 50/70-digit components; full input and 50-digit regression in phase14 |
| 0.0659880358454 | 0 | 0.5 | 0 / 0.25 | paired Taylor jumps | independent components through 100 digits in phase15 |
| 0.0659880358454 | 1e-14 | 0.5 | 0 / 0.25 | paired Taylor jumps | independent components through 100 digits in phase15 |
| 0.0659880358453125370767901876 | 1e-40 | 0.5 | 0 | unverified production value | independent reference exists; preliminary 50-digit CLI exceeded 1800 seconds with no output |
| 1.4448 | 0 | 0.5 | 0 | theta mapping, long Poincaré series | `1.2572110255920272276`, two contours; 17 s since November 13 (t870) |
| 1.4447 | 0 | 0.5 | 0.5 | not finished | t872 (ignored) now requires a value satisfying the functional equation; expected to take hours |
| 1.5 | 0 | 0.5 | 0 | theta mapping | `1.28087727794027`, about 1 second; equals the mpmath prototype |
| 1.46 | 0 | 0.5 | 0 | theta mapping | `1.26383460066838162179258037381`, three contours (t880) |
| 1.45 | 0 | 0.5 | 0 | theta mapping | `1.2594864327497848579`, about 2 seconds, two contours (t1609) |
| 1.43 / 1.44 | 0 | 0.5 | 0 | historical only | old regular-iteration observations, not current certificates |

### A.1 Strictly attracting deep-band witness — now independently checked

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

The new inverse germ now computes the original `h=48.013` witness through
100 digits, matching independent 300-digit orbit/log references at two depths.
The value begins `0.13077855094231531 - 0.016878312175402064i`.
This is a new regular-iteration construction, not acceptance of the old
stalled Kouznetsov samples. t890 checks the actual reference digits.

| b_re | b_im | h_re | h_im | mode | result |
|---|---|---|---|---|---|
| 0.0653281554868594 | 0.025 | 48.013 | 0 | inverse-Poincare / regular iteration | independently checked at 10/50/70/100 digits (t890) |
| 0.0653281554868594 | 0.05 | any swept | 0 | historical Schröder sweep, |λ|=0.978 | 1015-point observation, not a current accuracy certificate |

### A.2 The t860 "canonical value" was a discretization artifact (pseudo-verification uncovered)

**November 12 — solved.** The two-sided theta mapping (README § 6.10) gives
`F(0.5) = 0.9259482458090558672818776052663824496889 +
1.377349898696509477995821668326992920743i` for `b = −0.8 + 0.4i`: the same
digits at 15, 20, 22, 25 and 40 requested digits and on four contours at 30
digits, in 1.4 s at 15 digits. The lower fixed point is `−1.1704 − 0.0100i`,
the one the Cauchy solver rejected as too close to the real axis. t860 now
asserts this value and its mirror image. The anatomy below is history.

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

## B. Negative real bases (b ∈ ℝ, b < 0) — **two-sided theta mapping (November 12)**

**November 12.** Negative bases are the edge `Im ln b = π` of the upper
half-plane and are solved like any complex base (README § 6.10), about two
seconds each at 15 digits. `b = −2`: three contours agree to 30 digits,
`F(0.5) = 1.23499204711001806741684243512 + 1.72765186259320471027645429067i`;
t423 (`h = 0.4 + 0.1i`, formerly a required refusal) now checks the
functional equation. The Cauchy solver did not finish `b = −2` in 25 minutes
and failed at `−0.5` after 8.

| b | F_b(0.5), 15 digits |
|---|---|
| −0.4 | 0.524672046754290 + 1.45009175176364i |
| −0.5 | 0.611237217314526 + 1.48633320332353i |
| −0.8 | 0.806054241447764 + 1.56464362991783i |
| −1 | 0.904526679067709 + 1.60295813897714i |
| −2 | 1.23499204711002 + 1.72765186259320i |
| −3.6 | 1.54416369343679 + 1.84097333234317i |
| −1000 | 5.90597919030667 + 3.27931958174185i |

Everything below is the history of the Cauchy solver on this class.

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

## C. Pure-imaginary bases (b = i·y, y ≠ 0) — **regular iteration inside Shell–Thron, theta mapping outside**

**November 12.** Outside the Shell–Thron region (`y ≳ 1.72`) the two-sided
theta mapping takes about two seconds at 15 digits: `F_{2i}(0.5) =
1.43175893986135 + 0.908794685856499i`, `F_{3i}(0.5) = 1.65790261147990 +
0.937237881153346i`, `F_{3.6i}(0.5) = 1.76346022431977 + 0.951658511027643i`,
`F_{5i}(0.5) = 1.95950649348755 + 0.979857985910872i`; `−3i` is the mirror
image of `3i`. Inside the region (`y = 1.4`: `1.08532783113677 +
0.606234807996957i`) the dispatcher returns the regular iteration, which is
a different function from the two-fixed-point one the Cauchy solver was
computing in August (`1.243 + 0.888i` below).

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

## D. Complex bases far from real axis — **two-sided theta mapping (November 12)**

**November 12.** A scan of 220 bases (real parts `−1000 … 10^30`, imaginary
parts `0 … 10^5`) solved all 182 outside the Shell–Thron region, on average in
1.8 s at 15 digits. The Cauchy solver agrees to all printed digits where it
still converges (`2+0.1i`, `2+0.3i`, `3+0.5i`, `3+4i`, `10+5i` at 15 digits,
`1.2+3.5i` at 10). The August values in the table below agree with neither
solver and are superseded:

| b | F_b(0.5), 15 digits |
|---|---|
| 1.2+3.5i | 1.80301447273349 + 0.758446089166562i |
| 0.5+2i | 1.46177225929122 + 0.776175151281838i |
| −3.6+0.4i | 1.56910293151866 + 1.78132354348639i |
| −1.2+1.2i | 1.27322999811991 + 1.30541999894135i |
| −1+i | 1.18206140425322 + 1.28363814455972i |
| −1+2i | 1.45336611150906 + 1.16828211170583i |
| 2+5i | 2.03474973368267 + 0.751640175323391i |

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

**November 12:** the theta-mapping solver reproduces these to 20 digits in
under a second each (`3.6480135102974247023`, `4.2131043841295…`,
`4.8184545998105586664`, `5.6841886762291272595`, `6.3913243528216…`) and
continues far beyond, on a contour shifted left for `ln b > 16` and, when the
direct start fails, by continuation in the base:

| b | F_b(0.5), 15 digits | time | contours agreeing |
|---|---|---|---|
| 1e6 | 16.3591261058616 | 0.5 s | 3 |
| 1e30 | 645.547844724944 | 1.5 s | 3 |
| 1e100 | 65637.4795183687 | 3 s | 3 |
| 1e1000 | 278773383556.290 | 9 s | 2 |
| 1e10000 | 1.34047882293562e23 | 34 s | 1 |
| 1e100000 | — | not finished in 300 s | — |

---

## F. Real base b = 2 — **independent 50-digit witness; high-precision resource costs**

At `h=0.5`, the October 50-digit result
`1.4587818160364217006839716610385871352966066053309` differs relatively by
`4.90e-51` from the independent fatou.gp reference. The old
`1.4587818160364217112` anchor below is inaccurate after about 16 digits.

The earlier 70-digit setup requested 65536 nodes, exceeding its former 32768
budget. Its 1800-second CLI run timed out during continuation from the
`b=2.35` anchor, not after reaching `b=2`; no 70-digit base-2 accuracy claim
follows. Fixed node budgets are now removed. Sizing regressions give 524288
nodes at 200 digits and 16777216 at 1500 digits, not completed-solve evidence.
Such resource/convergence failures are not nonexistence theorems; t965 now
checks genuinely unaddressable near-parabolic geometry without a fixed cap.

An exact-dyadic 40/60-digit comparison at `h=5.375` found only 28.87 matching
digits despite zero recurrence residuals; the `h=0.375` control rounded
correctly. Five exponentiations amplified the reconstruction error.
`limits-range-baseline-comparison.json` in the resource-audit archive retains
the failed evidence. Height-conditioned internal refinement passes the
40-digit regression, including the final-source rerun after the geometry
correction (`limits-post-geometry-conditioning-40.json`). Its real outputs
round to the retained higher-precision fixture; the relative imaginary
component at `h=5.375` is `2.35e-58`, below `1e-40`. This is cross-precision
regression evidence, not independent canonicality verification.

The old precision-only EM order also failed an exact constant-Cauchy identity:
at the 130-digit geometry, a near-endpoint error of `5.17e-104` exceeded the
`1e-133` target. Spacing/scale-aware asymptotic order selection reduces that
error to `7.00e-146` (`limits-geometry-library.log`). Fresh full-construction
validation now includes actual `2^^0.5` matching corroborated external data at
118 and 130 digits. Actual `2^^5.1` outputs from both constructions round
identically at 110 significant digits, with empirical relative agreement of
118.35 digits (`limits-post-geometry-high-comparison.json`). These numerical
cross-checks, unlike a quadrature test alone, directly check returned digits;
they are still not global forward-error or canonicality certificates.

**November 12:** base 2 is now solved by the theta-mapping iteration. `2^^0.5`
equals the `fatou.gp` reference at all 100 digits (7 seconds, single thread);
200 and 400 digits take 42 seconds and 7 minutes. The 60-digit fixture
`F_2(0.375)` is reproduced exactly, real heights have imaginary part exactly
zero, and the 20-digit values below are confirmed to about 16 digits (their
last digits were Cauchy-solver noise): `0.54476412145955673398`,
`2.7487616545898224907`, `6.7213994941488630369`,
`1.3232385095217583751+0.86933975675873261759i`,
`1.6815069453310134039+1.0501317101851206469i`,
`1.2266228648289182647+0.61198669331965251251i`.

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

### I.1 Historical heuristic-series failures near η

This table records the old guard's behavior, not current refusal contracts.
Strict attractors now use the analytic-disk inverse germ; the independently
checked `b=1.444666` witness is closer to the endpoint than these rows.

| b_re | b_im | h_re | h_im | before | after the old guard |
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

### I.4 Two-fixed-point values for attracting bases — now ERR

Where regular iteration failed for a base with an attracting fixed point,
the dispatcher went on to the Cauchy solver. That solver converges there,
but to the function with two fixed points, which inside the Shell–Thron
region is a different function (README § 1.2). The rows `0.998` and
`0.9999` of the table in § A show the effect: values that continue the
exterior function and differ from regular iteration by `0.09 − 0.04i`,
returned with exit code 0. Since November 13 a failed regular iteration is
final: `unsupported case: attracting fixed point — regular iteration failed
(…); the two-fixed-point tetration is a different function inside the
Shell-Thron region and is not substituted`.

### I.5 Wrong logarithm branch in the height reduction — fixed

Regular iteration computes `F(h) = log_b^k F(h+k)` and took the principal
logarithm at each of the `k` steps. `F(h+s)` is continuous in the real shift
`s`, so the right branch changes whenever `F(h+s)` crosses the negative real
axis; next to the Shell–Thron boundary, with a turning multiplier, it does.
At `b = 1.764630858490303+0.1690076064324368i` (`|λ| = 0.992`):

| h | before | now |
|---|---|---|
| 0.54 | `1.49462006762+0.0535217886000i` | same |
| 0.56 | `1.56884…−1.56608…i` | `1.50876266888+0.0616079638776i` |
| 0.60 | `1.58902944374−1.57832424633i` | `1.53538151660+0.0781658334622i` |
| 0.62 | `1.54771257676+0.0862704600435i` | same |
| −0.42 | `1.13975545306−1.55864814493i` | `0.729031668194−0.0414735321435i` |
| −1.42 | `0.851954368008−1.78295441118i` | `−0.550446473280−0.00745588081966i` |

Every branch satisfies the functional equation and `F(0)=1` is only tested
at one height, so no existing gate saw it. A scan of 37 bases found the
jumps only at `|λ| ≥ 0.992` on that ray (`r = 0.995`: wrong on `0.46…0.76`);
for `|λ| ≥ 0.998` the same error made the anchor `F(0)=1` fail, which was
reported as "Poincare normalization failed". The branches are now fixed by
continuity along the real shifts (README § 6.3) inside the half-plane
`|λ^{h+2}| < 1`, where the tetration is single-valued; outside it the
principal logarithms remain the convention. 200 values of 20 ordinary bases
are unchanged, and so are the 100- and 1000-digit references.

At real heights the traced branches are also compared with the integer
tower, level by level and once more below the result (README § 6.3). A
branch forced wrong at the last step of the chain for this base, or one
and five levels before it, ends in `the logarithm branches of this height
contradict the integer tower`; at `|λ| = 0.9999` the tower refutes the
first chain, sampled nine times per unit of height, and the denser repeat
passes.

### I.6 Bases just above `e^{−e}` (multiplier next to −1)

15 digits, height `0.5`; "before" is the binary of November 12.

| b | before | now |
|---|---|---|
| 0.0665 | `0.367623816576838+0.0339018373906064i`, 0.4 s | same digits, 0.2 s (long series, 1054 terms) |
| 0.0661 | 7.0 s | same digits, 1.8 s |
| 0.066 | `0.367873311806215+0.00520321138081477i`, 15 s | same digits, 5.5 s (5414 terms) |
| 0.06599 | `0.367878434246886+0.00210845265075503i`, 18 s | same digits, 16 s (10918 terms) |
| 0.065989 | `0.367878946861275+0.00147725133341689i`, 23 s | same digits, 25 s (14267 terms) |
| 0.0659881 | `0.367879408277819+0.000381065906287679i`, 33 s | same digits, 62 s (Taylor jumps, samples follow them) |
| 0.06598804 | `0.367879439041224+0.0000969740320825670i`, 45 s | same digits, 75 s |

With the branches traced but the short series kept, `0.066` took 142 s and
`0.06599` 270 s: after the Taylor jumps tens of thousands of single steps
remain, each with its samples. Closer to `e^{−e}` than `0.065989` the
estimated length of the long series exceeds 32768 terms and the jumps are
kept; a sample
at an integer height is then evaluated exactly there, because the paired
jumps of a real base refuse a value whose imaginary part is rounding noise
(`0.06598804` failed on that for a while).

---

## J. Real bases on the cut segment  0 < b < e^{-e}   — **SOLVED at the real base (November 12); ε-walker is a fallback**

**November 12.** The limit from the upper half base-plane is computed at the
real base itself by the two-sided theta mapping with the fixed points
`−W₀(−ln b)/ln b` (real, `λ < −1`) and `−W₁(−ln b)/ln b` — the pair the
walker tracks. No walk is needed: `tet 15 0.04 0 0.5 0` takes 3 seconds.

* `b = 0.04`: five contours agree to 25 digits at nine heights (including
  `−1.5`, `−2.5+0.1i`, `2.5`); three agree to 30 digits. The default contour
  cannot start at this base and reaches it by continuation through complex
  bases above the Shell–Thron region; it ends at the same digits.
  `F(0.5) = 0.0663831355231474524653734313751 +
  0.447944861924265149963222115626i`.
* `b = 0.01`: three contours to 30 digits, `F(0.5) =
  −0.125023808503236713358729653966 + 0.413987966241729684229874554858i`.
* Continuity from above: `b = 0.04 + 10^{−10}i` agrees with `b = 0.04` to
  the expected ten digits (t1805).
* Other bases at 15 digits: `0.06` (`λ = −1.007`, 21 s) `0.133551357296516 +
  0.444588502619596i`; `0.05` `0.102649474416971 + 0.447040991018417i`;
  `0.001` `−0.335658584585117 + 0.268881908378586i`.
* Very small bases are reached by continuation along the cut from `e^−3.2`:
  `10^−4` (9 s) `−0.438500971524013 + 0.0786239019766715i`, `10^−6` (13 s)
  `−0.405545233641488 − 0.295259582379654i`, `10^−10` (19 s)
  `0.170813938896644 − 0.648660777250294i`; the same digits come out of the
  much longer path through complex bases on another contour. `10^−30` takes
  49 s, `10^−100` two minutes, `10^−1000` two and a half.
* The Taylor series of `F` at `−1/2` converges only to radius about 1.2 at
  `b = 0.04` (a further zero of `F` puts a singularity nearer than `−2`), so
  the discretization is enlarged automatically and smaller contours are used.

The walker description below is kept for the fallback path and as history.

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
  quadruples (historically n=16384). A zero within ~0.1
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
  retried at successively doubled node tiers while the near-miss criteria
  hold. The former one-retry/8× budget is removed; addressability and
  numerical/branch checks remain.
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

Use the independently sourced fixtures in phases10/12 and explicit refusal
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
contracts. README §8.1 separates promising directions from proven coverage.
The October 5 strict-attractor inverse germ is implemented. Since November 12
the theta mappings cover real bases above `e^{1/e}`, complex and negative
bases outside the Shell–Thron region and the cut bases; neutral multipliers
and global certification remain research.
