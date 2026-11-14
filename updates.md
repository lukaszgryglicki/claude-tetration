# Status Update — 2026-11-14, evening (web: the cache can be listed, references on the page)

**`tet-webctl cache` lists what the cache holds.** One entry a line: the
function, the precision and the numbers of the function in the order base,
height, value, two as they were asked for and the third as it was computed
(`f=ate prec=20 base=2 height=4.0000000000000000000 value=65536`), from the
entry used longest ago to the one used last, then the calculations that
run. The server answers `GET /cache` only for a request from its own
machine that no proxy passed on; through the proxy the address is a 404
like any other that is not served.

**The cache may grow to 1024 MiB** (it was 64). That is a bound, not a
reservation: the memory is used as the results come.

**References on the page**: the thread about it on the Tetration Forum,
[Tetration online calculator](https://tetrationforum.org/showthread.php?tid=1830),
and the two viewers that plot with `tet`,
[tet-line](https://github.com/lukaszgryglicki/tet-line) and
[rust-opengl-complex-drawer](https://github.com/lukaszgryglicki/rust-opengl-complex-drawer).

**The settings file of the machine is the one that counts.**
`web/freebsd/tet-web.conf` in the repository is only what a first
installation writes; it now sets six hours, five minutes, two threads and
1024 MiB for each `tet`. `web/README.md` says how to copy a file to the
machine, where the server takes it over without a restart.

**Tests.** 38 unit tests and 18 tests of the server as a process; the new
ones list a cache with results, a refusal and a running calculation, and
ask for the listing as a proxy would.

# Status Update — 2026-11-14, afternoon (web: https://hyp4.duckdns.org, long calculations, a page without scripts)

**The public instance is <https://hyp4.duckdns.org/>, by HTTPS only.** Caddy
obtains and renews the certificate (Let's Encrypt) by itself; plain HTTP,
for the name and for the bare address, answers with nothing but a redirect.
The site is reachable over IPv4 and IPv6: the certificate authority
validated it over IPv6 from seven places.

**A calculation runs in the server, not in its request.** It ends with its
result, with a refusal of `tet` or at the time limit, whether or not
anybody still waits. The requests for it wait until it is two minutes old;
after that the reply has the status 202 and says that the calculation
continues, and the same request again finds it running, or its result: a
calculation is never started twice. On the public instance `2^^0.51` at
300 digits took 271 s: 202 after two minutes, 202 at once for each
repetition, then the 300 digits of `tet 300 2 0.51`.

**Limits.** The precision has no largest value on the server. As many
calculations run at a time as the machine has processors, four at least;
there is no queue: while all slots are busy a new calculation is turned
away at once (503, `all slots are busy calculating; try again later`). A
calculation may take three hours; one that is stopped there is kept as
such, so that the same request is answered at once, and is calculated
again once the limit has been raised. On Linux and FreeBSD a `tet` ends
with the server.

**Settings that change while the server runs.** The time limit, the time a
request waits, the number of slots, and the threads, memory and niceness of
one `tet` stand in a settings file (`/usr/local/etc/tet-web.conf`). The
server looks at it every two seconds: a change is in use at once, also for
the calculations that run, without a restart. `tet-webctl timeout 6h` sets
the time limit in one command.

**The page needs no JavaScript.** It is a plain form that the server
answers with the same page and the result in it, so it works in `links`
and `lynx`; a calculation that still runs makes the page ask again (a
refresh, and a link for the browsers that do not refresh). With JavaScript
the same page saves the reloads. Both were driven in Firefox, with scripts
and without, and in `links` 2.30.

**The log has the results.** Every request for a calculation is one line
with the client, the command and the result or the reason; a calculation
that no request waited for has a line of its own when it ends.

**Start, stop, status, log.** `web/ctl.sh` (installed as `tet-webctl`):
`start|up`, `stop|down`, `restart`, `status` (with the certificate and the
calculations that run), `log [N]`, `log -f`, `timeout [TIME]`, on the
machine itself or from a checkout over ssh. A stop ends the calculations
that run and takes two seconds. `web/README.md` has an administration part
with all of it, the certificate included.

**Tests.** 36 unit tests and 17 tests of the server as a process: the form
without a script, the page that asks again, any precision and a result of
200 000 digits, all slots busy, the settings file read again while a
calculation runs, the log lines, calculations that end with the server.

# Status Update — 2026-11-14 (web page and HTTP API; expressions of astronomic size)

**`web/`: a page and an API in front of the binary.** `tet-web` (Rust,
standard library only, a crate of its own in `web/server`) serves a page
with the precision, the function (`tet`, `ate`/`slog`, `ssrt`/`sroot`) and
two complex numbers or four real ones, and the same as an API:
`/?prec=20&func=ate&base=2&value=65536` answers in JSON,
`&format=text` with the two lines of `tet`. Each computation is one `tet`
process, so the numerical crate is neither linked nor changed and the
numbers are those of the command line. A `+` in a query is a plus sign.

**Cache, in memory only.** Finished results and the refusals of `tet` are
kept until the server stops; nothing is written anywhere. The same request
again is answered at once, a request whose computation is running waits for
it, and the cache is bounded in bytes (the entries used longest ago go
first).

**Limits.** A time limit per computation (the process is stopped; the
kernel limits its processor time as well), a number of computations at a
time with a bounded queue, a largest precision, optionally the address space
and the niceness of each `tet`. The statuses tell the cases apart: 422 is a
refusal of `tet` with its message, 400 a request that is not a computation,
503 a busy server, 504 the time limit.

**Installation.** `web/freebsd/install.sh` and `web/deploy.sh` put it on a
FreeBSD machine as an rc.d service behind Caddy: plain HTTP on the bare
address, or `https://name` only once a domain points to the machine (Caddy
obtains and renews the certificate). [web/README.md](web/README.md)
describes the API, the settings and the installation; README § 3.3 points
to it.

**Tests.** 18 unit tests and 9 tests of the server as a process, with a
shell script in the place of `tet` (`cd web/server && cargo test`).

**Expressions of astronomic size.** Found with the API: an expression whose
value has a binary exponent beyond ±2^31 stopped `tet` with a panic
(`e^(1e12)`, `2^(2^31)`, `2^-(2^32)`, `1e1000000000000 + 0`), where the
decimal number of that size is read (`tet 20 1e1000000000000 0 1 0`). The
reader of expressions took the exponent from rug, where it is an `i32`; it
now takes it from MPFR (`cnum::exponent`). Such a value is read like the
decimal number when there is no bit to read before its point
(`2^(2^32)`, `e^(-1e12)`: the printed digits are those of decimal arithmetic
outside MPFR), and refused with a reason when its whole part has more than
2^31 − 1 bits (`e^(1e12)`). Below that size nothing changes: the whole part
is read in full, however long that takes (`fact(1e6)` has 18 million bits).
Test `t716` and a unit test of `src/expression.rs`.

# Status Update — 2026-11-14 (two complex numbers or four real ones)

**Two arguments instead of four.** The two numbers of a run can each be one
complex argument: `tet 30 1.5+0.5i 1+0.7i` is `tet 30 1.5 0.5 1 0.7`,
`tet --slog 30 2 16` is `tet --slog 30 2 0 16 0`, and
`tet --sroot 30 16 3` is `tet --sroot 30 16 0 3 0`. The count of the
arguments tells the forms apart; any other count is a usage error. The
library has `tetrate_complex_str`, `slog_complex_str` and
`sroot_complex_str` next to the three functions of four strings. The
plotters `tet-line` and `rust-opengl-complex-drawer` take two complex
points in the same way, for the ends of a line and the corners of a
rectangle, and still call `tet` with four numbers.

**Of four numbers each must be real.** Until now an argument could be
complex itself and the number was `first + i·second`, so that
`tet 20 2+i 0 0.5 0` ran with the base `2 + i` and `tet 20 i -2i 0.5 0` with
the same one. Such an argument is now an error that names both remedies:
`re(z)` for the real part of anything, or the form with two numbers. The
example of the last entry reads
`tet 30 '2^^2 + i*ate(2+i, 1.5-0.5i)' 'pi/e + i*ssrt(16, 3)/4'` and prints
the digits it printed before. A part is real when its imaginary part is
zero by construction; `exp(i*pi)` stays refused in both forms and
`re(exp(i*pi))` is `−1`.

**Numbers written in an expression keep their digits.** The precision of a
run followed the digits of a decimal argument, and of an expression only
those before its first `e` or `E`. Every number written in an expression is
now counted: `tet 10 're(1.<110 zeros>1 + 2i)' 0 -1 0` prints `0`, the
value of `b^^(−1)` for a base that is not one, where it printed
`1.000000000` because the base had been rounded to exactly `1`. `-0` is
read as `0` in both forms.

**A number has no gap.** The decimal reader skips spaces and underscores:
`tet 20 '2 1' '0.5 0'`, a real and an imaginary part in one argument, would
have run as `21^^0.50`. A text with a gap is now left to the grammar, where
two numbers in a row are an error (`tet 20 '1 2' 0 1 0` printed `12`
before).

**Tests.** `t715` (the two forms agree in the three modes, flags anywhere,
long numbers, numbers with a gap, wrong counts), `t713` (real expressions;
a complex part is refused in each place and mode), two unit tests of
`src/expression.rs` (real parts; the digits of written numbers) and rows
for the gap in a third.

# Status Update — 2026-11-13, evening (two-sided solver next to the boundary; real bases without tracing; constants and expressions as arguments)

**A real base needs no branch tracing (proved, then removed).** For a real
base with a real attracting fixed point the inverse germ has real
coefficients and is one-to-one on its disk, `b^z` is one-to-one on the image
of that disk, and the image is connected and symmetric about the real axis;
it therefore lies in the strip `|Im z| < π/|ln b|`, where the principal
logarithm is the inverse step (README § 6.3). Regular iteration at real
bases no longer carries the samples of the tracing. 27 values (real and
complex heights, 15 to 100 digits) are the same digits with and without it;
the 100-digit `closer_attractor` case takes 421 s instead of 561 s. The
check against the integer tower is unchanged, and so is everything at
complex bases.

**Long inverse series in the two-sided solver.** A base just outside the
Shell–Thron region cost about `1/ln|λ|` logarithms per sample of a theta
line and as many exponentials per contour node. The coefficients of the
inverse series at the nearly neutral fixed point decay with a radius far
beyond the short one, so the series is now summed on most of its disk and
the lines are usually reached without any logarithm; a coordinate is
accepted only on a disk where the series is one-to-one
(`Σ|cₙ|ρⁿ < (7/8)π/|ln b|`), after a check of the functional equation
(README § 6.10). At 15 digits, argument 1 of the multiplier: `|λ| = 1.001`
3.5 s (51 s before), `1.0005` 3.9 s (100 s), `1.0002` 4.0 s (no result in 40
minutes), `1 + 10⁻⁵` 5 s; argument 2: `1 + 10⁻¹²` 5 s. The short and the
long series give the same 30 digits at `1.003`; at `1.0001` and `1 + 10⁻⁸`
three contours agree at real and complex heights. Slow cases remain where
the argument of `λ` is next to a low-order root of unity (at `1.0001`:
`0.2` 50 s, `π/2` 164 s, `2π/3` 13 minutes).

**Raised contours on the cut.** For `0 < b < e^{−e}` the disk of the series
is pinched by the 2-cycle, and the upper theta function is singular below
the height where its line leaves that disk. The solver now moves to taller
contours with the same centre when the first precision level shows this,
and keeps the contour for the later stages. Cut bases are solved up to
`0.0659`: `0.063` 32 s (91 s before), `0.064` 20 s, `0.065` a minute,
`0.0658` nine minutes, `0.0659` six (no result in 40 minutes before, from
`0.065` on); `0.064`, `0.065` and `0.0658` agree on three contours.
`0.06595` and `0.06598` gave no result in 45 minutes.

**The inverses on the new bases; rule 8.** README § 7.1 has a new rule:
support for new bases, heights or values is complete only when tetration,
the super-logarithm and the super-root all work there, each with its own
test. For the bases above: `--slog` at `|λ| = 1 + 10⁻⁸` and at the cut base
`0.064` (`t1712`), `--sroot` to the base with `|λ| = 1.0001` (`t1713`, two
minutes); forward tests `t1810`, `t1811`.

**Constants and expressions as arguments.** Each of the four numbers of the
command line is a decimal number, as before, or a constant expression:
42 named constants (`pi`, `e`, `eta = e^(1/e)`, `towermin = e^(−e)`, `phi`,
`gamma`, `omega`, … README § 3.2), `i`, `+ - * / ^ ^^`, brackets, products
without a sign, and 76 function rows, tetration and its inverses among
them: `tet 30 'pi*e^e-pow(2, 0.5)' 0 1 0`,
`tet 30 '2^^2' 'ate(2+i, 1.5-0.5i)' 'pi/e' 'ssrt(16, 3)/4'`. The grammar
and the names are those of the plotters `tet-line` and
`rust-opengl-complex-drawer`, which gained the same constants and take
constant expressions for their ranges.

* *Precision.* A name is the constant at the precision the run reads it
  (`tet 1000 pi 0 1 0`). An expression is evaluated twice, node by node,
  with every operation rounded down and with every operation rounded up
  and twenty more digits; the two must agree beyond the working precision
  of the run in each part, and bits lost to cancellation are added.
  `(pi + 1e-300) - pi` gives `1e-300`.
* *Zeros and cuts.* A zero part is kept where it is exact by construction
  (`pi*i`, `sqrt(-pi)`, `1/4 + 1/4 - 1/2`); a zero that inexact operands
  produce, or a part whose two evaluations share no digit (`sin(pi)`), is
  refused after three attempts with more than doubled digits. Zeros are
  positive, so a value on a cut is the usual one (`sqrt(-4) = 2i`,
  `asin(2) = π/2 − 1.317i`).
* *Whole parts.* An expression too large for the precision of the run to
  hold its whole part raises that precision, so `0^^(10^100 + 1)` is `0`
  and `0^^(10^100)` is `1`.
* *Tetration inside an argument* is computed by the library itself, twice
  and with about 26 and 46 digits more than the run; it has the domain of
  the mode it names (`ate(2+i, 1000-500i)` fails as
  `tet --slog 15 2 1 1000 -500` does).
* *Tests.* Every constant against a 70-digit mpmath reference; expressions,
  branch cuts and all functions against mpmath; the zero rule; the grammar;
  CLI tests `t712`–`t714` (digits of π at 200, `--help` and README list
  every name). 140 expressions give the same value in `tet` and in the
  `f64` parser of the drawer to 12 digits, and so do 31 more with
  `tet(b, h)` and values on branch cuts: the plotters now drop the sign of
  a zero as well (`sqrt(-4)` is `2i` in all three) and have `tet(b, h)`.

**Validation.** The default tier passed on the final tree, one target at a
time (library 70, phases 1–5, 7–12, 16–18, grid and edge runners,
documentation tests, 19 Python tests), and so did the slow suites 13–15
(17 tests; README § 10 has the times). `cargo fmt --check` and
`cargo clippy -D warnings` are clean. The guard test of README § 7.1 caught
one new fixed-size comparison while this was written; it is rewritten and
listed.

# Status Update — 2026-11-13, later (fixed-size numbers guarded by a test; slow suites verified)

No solver code changed in this update, and no value did.

**The review of fixed-size numbers is now a test.**
`machine_numbers_stay_out_of_the_arithmetic` (`tests/phase10_honesty.rs`)
scans `src/` and `examples/` for machine float types, floating-point
literals, MPFR numbers whose size is a literal number of bits or digits and
constants that name such a number. Whatever the table of README § 7.1 does
not list fails it. The table gained the rows the review had left out: the
order in which the two-sided solver tries its contours (from `ln b` at 128
bits), `GUARD_DIGITS` and `CHECK_DIGITS`, the statistics of the two grid
tools and the 8-digit diagnostic `examples/test_w0.rs`.

**The decimal constants of the Cauchy solver were read one by one** (about
fifty: damping factors, step caps, schedules, starting guesses, contour
geometry, thresholds; all MPFR numbers at the working precision). Each
starts an iteration, steers it or rejects; none accepts a result. The
loosest, the residual up to which a stalled iterate is kept as a warm start,
cannot reach an answer: a state is evaluated only below `10^−(digits+3)`,
the cut-base walk must end at `ε = 0` under the same gate, and a state
solved for fewer digits than the request plus the measured loss is solved
again.

**`linear_approx` is removed.** The module was an entry point that only
returned an error, and nothing used it.

**Slow suites 13–15: all 17 tests pass** (FreeBSD host, `TET_MT=2`, one test
per invocation; table in README § 10). Phase 13 took 22 s, 43 s, 66 s and
549 s (the 100-digit `closer_attractor` case, deferred in the October
ledger); phase 14 6 s, 124 s, 313 s and 785 s; phase 15 46 s, 40 s and 35 s
for the cache and the two native heights (12 and 15 minutes in October,
before the long inverse series), 228 s, 419 s and 1071 s for the real base
at 20, 50 and 100 digits and 276 s, 524 s and 1311 s for the complex one.
The default tier passed on the same tree.

# Status Update — 2026-11-13 (branch fix in regular iteration; wider domain; rules for every change)

**A wrong logarithm branch in regular iteration (fixed).** The height
reduction `F(h) = log_b^k F(h+k)` took the principal logarithm at every step.
That is the right branch only while `F(h+s)` (`s` real) has not gone around 0.
For a turning multiplier next to the Shell–Thron boundary it has: at
`b = 1.764630858490303+0.1690076064324368i` (`|λ| = 0.992`) the printed
`F(h)` jumped along the real axis (`1.4946+0.0535i` at `h=0.54`,
`1.5688−1.5661i` on `0.56…0.60`, `1.5477+0.0863i` at `0.62`; likewise near
`−0.43` and `−1.43`), and at `|λ| = 0.995` it was wrong on `0.46…0.76`. The
functional equation and `F(0)=1` hold on every branch, so no gate saw it.
The branches are now fixed by continuity along real shifts of the height
(`Trail` in `src/schroder.rs`, README § 6.3), inside the half-plane
`|λ^(h+2)| < 1` where the tetration is single-valued; outside it the
principal logarithms remain the convention, so every value there is
unchanged. Unchanged too: 200 values of 20 ordinary bases, the 100- and
1000-digit references, and all values of real bases between 1 and `e^(1/e)`
at real heights (provably principal; no samples are taken). Regular iteration
now also works where the wrong branch broke its anchor `F(0)=1`:
`|λ| = 0.998` gives `1.47919432026926+0.0169457889363597i`.

**The integer tower checks the branches of real heights.** One sample of
every traced chain sits at an integer height, where the tetration is
`b^^m`. It must reproduce that value at every level, and once more one
level below the result, which ties the branch of the result itself to an
exact value (`Anchor` in `src/schroder.rs`). A refuted chain is repeated
with more samples; a chain refuted twice at the same place is an error. For
real heights above `−1` the distances between samples are judged only after
such a refutation, which makes those chains two to seven times cheaper.
Complex heights and heights between `−2` and `−1` still rest on the judged
distances.

**A long inverse series for regular iteration** (README § 6.3). The series
`ψ` of `F(h) = L + ψ(s₁ λ^h)` was summed on a disk of radius
`(1−|λ|)/(16 |ln b|)`. It converges up to the coordinate of the height `−2`
and is one-to-one there, so de Branges' bound `|c_n| ≤ n R^(1−n)` gives its
tail anywhere inside. For nearly neutral multipliers it is now summed on
most of that disk, and a height needs tens or hundreds of logarithms where
it needed thousands: `|λ| = 0.998` (argument 1) 0.3 s instead of 3.5
minutes, `0.999` 2 s, `0.9999` 45 s (both new:
`1.47997472280679+0.0134730013469168i` and
`1.46725728715182+0.00732647326377828i`, the same at 25 digits, and
`--slog` returns the height), `0.99999` 9 minutes
(`1.46489833107812+0.00424170610714995i`). The cut side gains as well:
`0.0665` 0.2 s (0.4 s on November 12), `0.066` 5.5 s (15 s), `0.06599`
16 s (18 s). Nine 50-digit values equal the short series in every digit.
Taylor jumps stay
for real bases below `e^(1/e)` and for bases closer to `e^(−e)` than
`0.065989`; there the samples of the branch tracing follow the jumps, which
costs up to twice the time of November 12 (`0.0659881`: 62 s, was 33 s).
The conditioning estimate of a chain is now the largest product of the
later factors after any step instead of the sum of all expanding steps,
which asked for 717 bits at `|λ| = 0.9999`.

**No answers from the other family.** For a base with an attracting fixed
point a failed regular iteration used to fall through to the Cauchy solver,
which converges to the two-fixed-point function there (`1.3862+0.0572i` at
`|λ| = 0.998`, where regular iteration gives `1.4792+0.0169i`). It is now an
error that says so.

**Rules for every change** are written down in README § 7.1 and at the top
of `src/lib.rs`: accuracy before speed; a result is a converged value, never
an approximation, otherwise an error; every intermediate value carries at
least the requested precision; no machine numbers in the arithmetic. Every
`f64` and fixed-size MPFR number in `src/` was reviewed: the cost model of
the long Poincaré series, radii, tolerances, step tests and the branch scout
of the two-sided solver now run at the working precision; what remains
(starting values, sizes, exact constants, timing) is listed with its reason
in a table there.

**Real bases next to `e^(1/e)`** (README § 6.8). The Poincaré series is summed
on a disk that reaches most of the way to the conjugate fixed point, which
shortens the logarithmic orbits from `1/ln|λ|` to a fraction of that:
`1.4448` 13 s at 15 digits (210 s) and 17 s at 20 (8 minutes), `1.4447`
35 s, `1.44467` 2.6 minutes (both never finished before), `1.45` 1.8 s
(6 s). 47 values equal the digits of the previous path, and the slope of
`F_b(0.5)` in `b` is the same on both sides of `e^(1/e)`.

**Heights.** Far to the left regular iteration stops iterating once the
chain repeats exactly (whole periods are skipped) or, for a real base, once
the real part is stationary (the vanishing imaginary part follows in closed
form): `1.3^^(−1000000.5)` in 0.01 s instead of 13 s, same digits. Integer
towers skip exact periods and continue from `b^0 = 1` after a level below
MPFR's range: `0.04^^10^30`, `i^^10^50`, `(−10^30)^^3 = 1` are immediate;
their working precision no longer grows without end. The theta solvers
treat such levels the same way: `(−100)^^100.5 = 1 + 3.14159265358979e-200 i`.

**Complex bases next to the boundary from outside.** The two-sided solver
took the slow spiral of its logarithmic orbits for a failure and then tried
to reach the base through the region. It now compares closest approaches:
`|λ| = 1.001` at argument 1 takes 30 s (was a timeout), `0.5+2i` 5 s (12 s).

**Inverses.** `--slog`: real targets of cut bases (the path passes below the
real fixed point, as the limit of complex bases does), targets like `10^100`
(geometric traversal of the path), targets behind an image of the cut
(detours), and targets whose height rounds to a singular height of a regular
base (`slog_0.5(−100)`), which are returned as the limit of their neighbours.
`--sroot`: real roots for heights in `(−1, 0)` and `(−2, −1)`; complex
searches follow their path with 14-digit forward values and polish the end
at full precision (2 to 4 times faster, same digits); every solve above
`e^(1/e)` starts from the previous base.

**Still open.** Cut bases from `0.065` to `e^(−e)` and repelling multipliers
within `10^(−4)` of the unit circle (minutes to hours; the long series is
not yet built into the two-sided solver); attracting multipliers closer
than `10^(−5)` (ten to thirty times the cost per factor of ten);
`b − e^(1/e) < 10^(−7)`; the base `e^(1/e)` itself; super-roots whose path
meets the Shell–Thron boundary; a branch check for complex heights; the
branch convention of regular iteration outside `|λ^(h+2)| < 1` and on its
cut (principal logarithms, as before).

# Status Update — 2026-11-12 (theta-mapping solvers: real, complex, negative and cut bases; inverses)

**Real bases above `e^(1/e)`** are solved by a Kneser theta-mapping iteration
(`src/kneser.rs`, README § 6.8) instead of the Cauchy rectangle: `F(z) = S(z +
theta(z))` above the height `y0`, a real Taylor polynomial near the real axis,
Anderson-accelerated fixed-point steps and a precision ladder. All sizes are
linear in the digits. `2^^0.5`: 0.3 s at 15 digits (41 s before), equal to the
independent reference at all 100 digits in 7 s, 200 digits in 42 s, 400 in 7
minutes (75 s with `TET_MT=8`, bit-identical). Seven contours agree for base
2, four for base 3 at 30 digits, and nine base-2 heights equal the Cauchy
solver's 15 digits. Newly reachable: `b=1.6` (0.5 s; over 60 s before), `1.46`
(both old solvers stalled), `1.4448` (formerly refused; 8 minutes at 20
digits, two contours), `10^30` … `10^10000` (base continuation), and heights
of any imaginary part. Real heights return imaginary part exactly 0.

**Complex, negative and cut bases** whose two primary fixed points repel are
solved by the two-sided form of the same iteration (`src/kneser_complex.rs`,
README § 6.10): `theta` functions at `−W₀(−ln b)/ln b` above and
`−W₁(−ln b)/ln b` below, a complex Taylor polynomial between, logarithm
branches chosen by continuity, sizes and contours adapted when the gates
fail, continuation in the base when no direct start converges. Evidence: it
reproduces the Kneser solver on the real axis (25 digits) and the Cauchy
solver's 15 digits at `2+0.1i`, `2+0.3i`, `3+0.5i`, `3+4i`, `10+5i` (26–52 s
there, about a second here); three or more contours agree to 30 digits at
`−2`, `3+4i`, `−1+2i`, `−0.8+0.4i`, `0.04`, `0.01`; a 220-base scan (real
parts `−1000 … 10^30`, imaginary parts `0 … 10^5`) solved all 182 bases
outside the Shell–Thron region. Formerly failing or timing out: `−2`, `−0.5`,
`−0.8+0.4i` (the August artifact base: now the same digits at five
precisions), `0.5+2i`, `−1+2i`, `3i`. **The cut bases `0 < b < e^−e` are
computed at the real base itself**: `0.04` in 3 seconds, five contours to 25
digits, continuous with `0.04 + 10^−10 i`; the ε-walker is only a fallback.
Bases down to `10^−1000` work (continuation along the cut, two and a half
minutes; `10^−10` 19 s). An independent mpmath prototype of the iteration,
continued from a real base through the upper half base-plane, ends at the
same digits at `−2` and `−0.8+0.4i`: the directly started solutions are the
analytic continuation of the real Kneser tetration.

**Inverses** (`src/inverse.rs`, README § 6.9): `tet --slog` (alias `--ate`)
gives the height, `tet --sroot` (alias `--ssrt`) the base. Regular bases use
the closed form through the Schroder coordinate; real bases above `e^(1/e)`
solve a reduced real equation or continue from the real axis; complex,
negative and cut bases continue `F(0) = 1` on the cached two-sided state;
real super-roots use a bracket in `ln b`, complex ones a continuation that
passes above singular values. Every result is checked by forward evaluation,
and conditioning is measured: ill-conditioned answers are recomputed with
the lost digits added. The default invocation is unchanged.

**Review.** An independent review of the new code found five defects, all
fixed with regression tests: a digit-count-dependent branch of `--slog` for
real bases below 1; wrong trailing digits of ill-conditioned inverses (target
next to a fixed point, tiny heights, the fold of `b^b`); a hang at heights
like `−10^12`; refused real `--slog` targets such as `−130` or
`10^(10^15)`; and non-integer heights within the solver's rounding of `−1`,
`−2`, … taken for integers.

**Audit.** A 576-case probe of the previous paths (12 vs 24 digits,
functional equation, 30 s budget) was cut short by a reboot at 455 cases; all
315 that returned values were consistent. No wrong value was found there.

Tests: `tests/phase16_kneser.rs` (9), `tests/phase17_inverse.rs` (9) and
`tests/phase18_complex_kneser.rs` (8) are new. t880 asserts three
contour-independent 30-digit values at `b=1.46`; t423, t852 and t860, which
asserted refusals at `−2`, `e^(3πi/4)` and `−0.8+0.4i`, now check values;
t960, t961, t963, t964 are no longer ignored; t870/t872 are slow ignored
positive tests. `TET_CAUCHY=1` selects the Cauchy path for A/B comparison.

Still open: neutral multipliers, including the parabolic base, and the cost
towards them (`1/(b−η)` on the real axis, `1/ln|λ|` in general); the
two-fixed-point function inside the Shell–Thron region; complex super-roots at
non-integer heights are slow (each trial base is solved from scratch);
`10^100000`; forward-error enclosures; an external reference value for a
complex base.

---

# Status Update — 2026-10-06 (paired negative-multiplier extension)

Strictly attracting multipliers near `-1` now use guarded paired Taylor
iterations. Quadratic cancellation permits larger checked disks, while
coefficient/tail bounds, actual-map defects and logarithm guards preserve the
regular family. Original-step counts remain GMP integers, including beyond
`u128`; no public API/state field, dependency or solver limit was added.

Independent 20/50/100-digit library references cover real and genuinely complex
bases near the lower Shell-Thron boundary, at both real and complex heights.
Exact pi-scaled phases and separate integer shifts also repair native-scale
height components: actual 50-digit CLI regressions pass on both axes at
`epsilon=10^(-10^18)`, without artificial `sin(k*pi)` noise.

The frozen focused suite passes 184 Linux Rust tests and 68 native FreeBSD
tests; unchanged ST/MT outputs remain identical through 100000 digits.
All 40 scoped platform checks pass. All 80 targeted 4D cells complete through
140 digits with 128 matching lower-precision components; 16 directional
controls pass and both ordinary/reflected grids retain 50430 baseline cells.
README §5 records the scoped grids, platform evidence and archive.
A deeper 50-digit CLI probe and two initial 100-digit CLI platform probes
exceeded external budgets; independent reference data and raw-library passes
do not turn those timeouts into verified CLI values. Exactly neutral and
difficult outside/cut constructions remain research.

---

# Status Update — 2026-10-06 (complex near-neutral extension)

Guarded Taylor jumps now support genuinely complex strictly attracting
multipliers in both normalization and height unwinding. Complex coefficient
error disks, analytic tails, actual-map defects and principal-log guards
preserve the regular family and the old positive-real engine.
No public API/state field, dependency or solver limit was added.

Independent references verify 100-digit real/complex heights at
`b=1.444667861009766+1e-15i`, and 50/70-digit half-heights at the closer
61-digit real base plus `1e-40i`. The latter uncovered a five-last-place-unit
imaginary-component error: complex jumps now refine for each nonzero output
component, not only the whole norm. Exact cached anchors retain working
precision. Local primitive tests exercise logical counts beyond `u128`.

The reported FreeBSD static-link warning is handled with a FreeBSD-only
unused-driver-argument flag, retaining other linker diagnostics, caller flags
and real static-ELF checks.

The frozen source passes 167 focused Linux Rust tests, 97 native FreeBSD tests,
40 byte-identical ST/MT/platform checks and unchanged outputs through 100,000
digits. Old expensive/prohibited cases remain deferred rather than disguised
as passes. Rotating/oscillating controls agree with scalar iteration but may
be slower; still closer complex probes remain externally time-limited.
The completed grid ledger has135 targeted cells through140 digits,
216 matching lower-precision components,25215 old/new ordinary cells,
16810 conjugacy checks,8405 same-branch overlaps and9 scalar controls.
External batch/slice timeouts were recovered without rerunning successful
slices; no target cells are missing. These are scoped checks, not a global
accuracy or canonicality certificate.
README §5 records the precise reference, grid and archive scope.

---

# Status Update — 2026-10-06

Real near-neutral regular iteration now uses scaled dyadic Taylor jumps in
both forward normalization and height unwinding. Directed coefficient bounds,
analytic tails, fixed-point defects and conditioning protect the same
principal-log family; other regimes retain their individual-step path.
No public state/API field, dependency or artificial solver limit was added.

t710/t871 now pass without ignore attributes. Independent 360-digit orbit/log
references verify 100-digit real/complex-height outputs at multiplier gaps
about `2.24e-8` and `1.58e-30`, including logical counts beyond `u64`.
The first base's 20-digit half-height takes about 4.7 seconds; the closer
100-digit string-API regression takes about six minutes.

Focused validation: 158 Linux Rust passes, 88 FreeBSD passes, 30 ST/MT/platform
checks, 135 complete targeted grid cells and 216 component comparisons against
140 digits. Ordinary-grid/direct-iteration controls and unaffected outputs
through 100,000 digits remain identical. This is not universal coverage or a
full-suite certification. Exact scope, references and frozen artifacts are
linked in README §5.

---

# Status Update — 2026-10-05

Implemented a direct inverse-Poincare construction for every resolved strictly
attracting fixed point, including the former near-neutral exclusion band.
The quadratic recurrence avoids cubic sigma construction/reversion; a
conservative analytic inverse disk controls truncation. Order follows precision
and the actual normalization coordinate. Forward-orbit normalization, a fixed
requested-digit goal and corrected logarithm-unwind conditioning retain the
regular-iteration family and reject inaccurate reconstructions.

Independent references now check three new fringe bases and the original t890
deep-band witness through 100 digits, plus actual `b=1.25` noninteger real and
complex heights at 1000 digits. Native-scale complex bases and real/complex heights also match
independent asymptotic/log references through 100 digits, with exact ST/MT CLI
agreement. These are returned-value checks, not storage-type or FE-only claims.
Fixtures are in `tests/phase12_coverage.rs` and t890; t710/t871 are explicitly
ignored long-run success targets, not claimed passes.

The new native-scale fractional-height witness
`(1+i*10^(-10^18))^^(-1.5)` resolves 100,000 significant digits in both components,
with decimal exponent 499999999999999999. Its independent asymptotic reference
has a negligible analytic remainder and an exact integer check of the rounded
square-root mantissa. Six ST/MT/platform runs agree and take about 7 seconds
each. All 144 Linux/static-Linux/native-FreeBSD compatibility checks pass.

The completed attracting/fringe 4D grids contain 152,130 cells across
5/10/20/50/100/140 digits: 151,794 values, 336 undefined, 0 errors/unobserved.
All 252,990 compared components match 140-digit rounding. The separate coarse
outer survey remains partial (40 values, 40 undefined, 108 timed out, 217 not run),
not a full `[-5,5]^4` or global-canonicality certificate.

The final frozen Rust ledger is 204 passed, two externally timed out, four
deferred and nine ignored identities, not a 219/219 pass. t852/t860 remain
unverified at 300 seconds; t428 finishes its extended run. The original
t423/t870/t872 were not restarted. Native FreeBSD adds 148 focused passes,
and all 18 Python tests pass. No assertion failure was recorded.

Log-domain coordinate/height shifts remove avoidable intermediate exponent
failures without substituting a fixed point. Small-root Lambert convergence,
the equivalent fixed-point exponential identity, and an exact-squaring
MPFR `log1p`/`atan` path for unit-real logarithms remove demonstrated MPC
division/refinement stalls and allocations. The numerical libraries, native
limits, final residual gates and branch conventions are unchanged; no solver
iteration, order, time or memory cap was added.

README §5 records the completed numerical evidence and its scope. Frozen
sources/binaries, grids, independent references and bounded-validation logs are
in `~/tetration-coverage-2026-10-05-artifacts/`. Universal finite coverage,
neutral-sector matching, difficult outside/cut-base constructions and global
forward-error/uniqueness certificates remain unresolved. The older audit and
historical reports below are retained with their original scopes.

---

# Status Update — 2026-10-04

This is a correctness/precision audit, **not a new tetration construction or
a claim of universal canonical coverage**. Working precision, a small
functional-equation residual and agreement between related discretizations
are not accuracy certificates. The current contracts supersede the August
claims archived below.

## Correctness changes

- Fixed computational budgets are removed: node grids, series/EM orders,
  iteration counts, height shifts, continuation/refinement and chart runtimes
  are no longer stopped by small application ceilings. Native addressability,
  actual memory, exponent range and numerical/branch checks still apply.
  `TET_KOUZ_PATIENT` no longer changes stopping behavior.
- Numerical tolerances, residuals, branch/region tests, normalization,
  continuation endpoints and checkpoint data use MPFR/MPC precision.
  Unbounded counters and exact coefficients use GMP Integer/Rational.
  Bounded indexes, library precision/exponent metadata and display/timing
  statistics remain native. Rug's 64-bit precision APIs and the widest
  supported per-thread MPFR exponent range replace artificial precision caps.
  Unrepresentable tolerances/intermediates cannot become zero-valued success.
- π, e, logarithms and derived boundaries are computed at working precision.
  Exact-rational Bernoulli generation replaces the 20-entry EM table.
  Significand/exponent formatting bypasses Rug 1.30's exponent-sized allocation
  bug, including verbose diagnostics and streamed checkpoints.
- The checked exponential uses outward-rounded, binary-scaled MPFR bounds.
  This fixes MPC 1.4.1's infinite refinement on a single underflowed component,
  unnecessary precision proportional to tiny input exponents, and premature
  magnitude overflow despite representable components. Ordinary results match
  MPC bit-for-bit; native-edge witnesses use independently rescaled references.
  These primitive rounding bounds do not certify an entire tetration solve.
- Integer, Schroder and Kouznetsov evaluations now refine internal accuracy
  against a fixed output-digit goal. This repairs demonstrated precision loss
  in large towers, tiny near-singular outputs and large imaginary phases,
  including only 28.87 matching digits for a 40/60-digit `2^^5.375` baseline
  despite zero recurrence residuals. Refinement preserves the fixed-point
  pair/log sheet and does not discard imaginary components. Decimal inputs
  are reparsed when increased working precision changes their values.
- Contour sizing uses the slower of both fixed-point tails. EM order now follows
  spacing, scale and precision, stopping at optimal asymptotic truncation rather
  than a fixed budget. A 130-digit constant-Cauchy regression improves from
  `5.17e-104` error to `7.00e-146`, below its `1e-133` target. The former
  precision-only order could not supply the requested quadrature accuracy.
  Local conditioning/quadrature witnesses are not global error certificates;
  the corrected-source 40-digit large-height regression now passes. The final
  regression ledger has 203 passes and three user-cancelled, unverified cases
  (`t423`, `t870`, `t872`), not a fully passing 206-case suite. Actual `2^^0.5`
  outputs match external references at 118/130 digits, and `2^^5.1` agrees at
  its 110-digit goal across those construction precisions; full evidence is in
  `limits-post-geometry-high-comparison.json` in the resource-audit archive.
- String/CLI and grid working precision also preserves supplied significant
  decimal digits, without changing output precision. A long near-one base,
  noninteger height or huge odd height must not silently become a different
  exact special case. Redundant padding and exponent text are not counted.
- NaN/Inf inputs, exponential overflow/underflow (including loss of only one
  complex component), invalid precision and unsupported domains are errors.
  Cached evaluators now refuse the same exact negative-integer singularities
  as dispatch; roundoff in `F(-1)` must not manufacture finite `F(-2)`.
  Positive integer towers of exact base `-1` remain exactly `-1`; repeated
  log/exp roundoff previously grew to about `3e-21` at height 100 despite a
  50-digit request. Domain conventions remain; huge integral heights and parity
  no longer require exponent-sized integer conversion or a fixed height budget.
- Lambert W retains the existing seeds/Halley method, with corrected
  logarithm-branch identity and internal guard precision near `-1/e`.
  Requested stopping precision is separate from internal working precision.
- Unchecked fixed-ladder Richardson output is removed; the public linear
  surrogate is deprecated and returns an error. A limiting fixed point is
  not substituted for a finite off-contour height.
- Kouznetsov checks the actual normalized evaluator, cached base identity,
  finite operators, actual GMRES residual and the full
  `10^-(digits+3)` boundary target. Complex candidates pass that target before
  normalization, so an unusable state cannot spend millions of Newton steps
  before its inevitable refusal or prevent the existing two-sided retry.
  Backward FE validation avoids rejecting a
  finite requested value merely because its unnecessary successor overflows.
- Real-boundary dispatch tries the existing direct solver after failed
  continuation: `b=1.5` genuinely succeeds by this previously skipped route.
  Geometry/addressability and convergence failures are not mathematical
  nonexistence or proof that a particular future theory is required.
  Real bases above `e^(1/e)` skip the regular family that the existing
  canonicality guard would reject regardless of the work spent calculating it.
- Cauchy reconstruction replaces linear continuation resampling; target-base
  fixed points remain warm seeds outside the old contour, never final answers.
  Resampling can be very expensive at large grids.
- Checkpoints are atomic, full-precision `TETCKPT2`; incompatibility,
  corruption and I/O errors are explicit. Residual dumps honor their prefix.
  Library MT initialization honors the same explicit configuration as the CLI.
- Diagnostics remain on by default; `--quiet`, `--silent`, `-q` or
  `SILENT=1` suppress progress, not fatal errors. Runner coordinates are exact
  decimal grids. Plotting validates input/output, preserves failed samples,
  escapes SVG text and handles zero-span axes. Historical galleries were not
  regenerated or promoted to accuracy evidence.

## Numerical evidence and remaining limits

Six attracting regular-iteration cases have independent 260-digit mpmath
orbit/log-unwinding references, stabilized at two orbit depths. Actual CLI
rounding agrees at 50 and 70 digits. Durable regressions cover those references
at 1/10/50/70 digits, lower-half-plane cases, and 1000-digit primitives,
near-branch Lambert W, tolerances, kernels and checkpoint roundtrips.
Resource regressions exercise native exponent endpoints, >32768-node
checkpoints, >2000-step schedules, >1M-coordinate axes and >8 GMRES restarts.
Independent constant identities agree in returned digits at 70/1000/10000
digits; actual `10^^3` output retains 100000 significant digits with decimal
exponent 10000000000. The `e^^3` witness now computes e at working precision
instead of using a short decimal literal.

For `b=2,h=0.5`, the 50-digit result
`1.4587818160364217006839716610385871352966066053309` differs relatively by
`4.90e-51` from the independent fatou.gp reference. The old
`1.4587818160364217112` anchor was inaccurate after about 16 digits.
The earlier 70-digit direct setup requested 65536 nodes, above the former
32768 cap; its bounded CLI run timed out during continuation from `b=2.35`,
**before reaching `b=2`**. That log validates no 70-digit base-2 value.
The cap is now removed. Required grids of 524288 nodes at 200 digits and
16777216 at 1500 digits are verified by sizing tests without allocating
those grids; this is not evidence of completed high-precision solves.

An isolated rebuild of committed revision `26aec7f` also stalls at the first
`b=1.46` continuation step, then returns unchecked Richardson output that
disagrees with the old t880 reference after roughly eight digits. That
legacy "full precision" test claim is retracted, not treated as lost verified
coverage. The current dispatch explicitly refuses both failed solver paths.
The solver's historical "LM" label means diagonally regularized Newton,
not least-squares Levenberg-Marquardt; failure to find descent does not prove
a discretization floor. No new optimizer was implemented in this audit.

The old `b=-0.8+0.4i` value remains retracted. Known failed solves must refuse,
including when the stricter normalization check fails before the residual
gate. Finite all-height coverage is impossible under the nondegenerate
`F(-1)=0` convention: a finite exponential cannot produce that zero from
`F(-2)`. Existence/uniqueness of all intended complex-base and cut-limit
branches is not established.

See README §§5/7/8 and [the current failure atlas](FAILURE_CASES.md) for
evidence, limitations and the assessment of future work. Maintainer-local
logs, reference generator and snapshots are preserved in
`~/tetration-audit-2026-10-04-artifacts/`; they are not committed gallery data.
No new mathematical algorithm was implemented.

---

# Historical Status Update — 2026-08-23

**Archive, not current policy or verification.** The old 20-digit-only
validation policy is superseded. The t860 success in §1.1 was retracted in
§1.7; an FE check or a same-family baseline did not prove its printed digits.
The old clamped gates, Richardson successes and claims of a proven in-strip
zero are also superseded by the October audit.

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
