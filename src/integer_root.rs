//! The principal super-root at integer heights.
//!
//! With `w = Log b` the tower `b^^n` of principal powers is `E_n(w)`, where
//! `E_0 = 1` and `E_k(w) = exp(w E_{k-1}(w))`: entire functions of `w`. The
//! equation `b^^n = z` is `h(w) = w E_{n-1}(w) = log z`, and `h(w) = w + ...`
//! at 0. For `n = 2`, `h(w) = w e^w`: the function that the Lambert W inverts.
//!
//! The principal super-root is `exp(W(Log z))`, with the principal logarithm
//! and with `W` the inverse of `h` that is 0 at 0, continued along the
//! straight segment from 0 to `Log z` (in the plane of `z`: along `z^t`,
//! `0 <= t <= 1`). At the height 2 it is `exp(W_0(Log z))`; for a real
//! `z > 1` it is the real root; inside `|Log z| < 1` it tends to `z^(1/z)`,
//! the base whose infinite tower is `z`, as the height grows. A real `z`
//! below 1 whose segment meets a branch point (an even height, `z` below the
//! least value of the tower on `(0, 1)`) has the limit from above, as `W_0`:
//! the path stays on the real axis and passes above the branch point.
//!
//! The continuation is certified. At a point `(x, w)` of the path a disk
//! `|v - w| <= r` is found on which `|h'(v) - h'(w)| <= d < |h'(w)|`, from
//! enclosures of the recursion in disk arithmetic. On such a disk `h` is
//! one-to-one (the real part of `h'(v)/h'(w)` is positive on a convex set)
//! and `|h(u) - h(v)| >= (|h'(w)| - d) |u - v|`. For every `x'` within
//! `r (|h'(w)| - d)` of `h(w)` the chord iteration
//! `v -> v - (h(v) - x')/h'(w)` maps the disk into itself and contracts it,
//! so `h(v) = x'` has a root in the disk, the only one there, and it is the
//! continuation of `w` along the segment from `x` to `x'`: no step can
//! change the branch. Next to a branch point the disks shrink in proportion
//! to its distance and the working bits of the continuation grow with them;
//! a path that runs into a branch point itself ends in an error.
//!
//! The one branch point that a path does run into is a fold of the real
//! axis: a real `c` with `h'(c) = 0 < h''(c)`, where the real root ends. On a
//! disk about `c` on which `h''` stays within a quarter of `h''(c)`,
//! `h(v) - h(c) = (v - c)^2 phi(v)` with `phi` within a quarter of
//! `h''(c)/2`, so every `x` close enough to `h(c)` has exactly two roots in
//! the disk (Rouche), and along a path of such `x` the argument of
//! `v - c` is half of that of `x - h(c)` less that of `phi(v)`. A path that
//! comes in on the real axis with the real root right of `c` and passes
//! above `h(c)` therefore leaves with the root of positive imaginary part,
//! and one step takes it across.
//!
//! `e^w` is a base of the tetration only when `-pi < Im w <= pi`: the powers
//! of the tower are principal ones. Where the continuation ends outside of
//! that strip (at the height 4 for `|z|` below about `1e-115`, where the
//! root has wound around the origin of the plane of bases), the principal
//! branch has no base and the answer is an error.

use rug::{float::Constant, Complex, Float};

use crate::cnum;
use cnum::DisplayFloat;

/// Decimal digits of the continuation that selects the branch. Every step
/// of it is certified at the precision it runs at, with the rounding
/// counted, and the root it ends at is then computed at the working
/// precision: no digit of a result depends on this number.
const TRACK_DIGITS: u64 = 30;

/// The largest height whose super-root is the principal branch: a step of
/// the continuation costs as many exponentials as the tower has levels.
pub(crate) const LARGEST_HEIGHT: u32 = 512;

/// A disk of complex numbers with the bound that rounding alone accounts
/// for: the centre of a result may be off the value at the centres of the
/// arguments by `noise`, and no member is further than `rad` from it.
struct Ball {
    mid: Complex,
    rad: Float,
    noise: Float,
}

/// Disk arithmetic at a number of bits: each result contains every value of
/// the operation on members of its arguments.
struct Disks {
    bits: u64,
    /// Above the relative rounding error of a centre.
    ulp: Float,
    /// A factor above the rounding of the few real operations of a radius.
    wide: Float,
}

impl Disks {
    fn new(bits: u64) -> Disks {
        let size = usize::try_from(bits).expect("precision exceeds addressable bits");
        let one = Float::with_val_64(bits, 1);
        Disks {
            bits,
            ulp: one.clone() >> size.saturating_sub(4),
            wide: Float::with_val_64(bits, &one + (one.clone() >> (size / 2))),
        }
    }

    fn exact(&self, mid: Complex) -> Ball {
        Ball {
            mid,
            rad: Float::new_64(self.bits),
            noise: Float::new_64(self.bits),
        }
    }

    fn close(&self, mid: Complex, rad: Float, noise: Float) -> Option<Ball> {
        let rounding = cnum::abs(&mid, self.bits) * &self.ulp;
        let rad = (rad + &rounding) * &self.wide;
        let noise = (noise + rounding) * &self.wide;
        (cnum::is_finite(&mid) && rad.is_finite()).then_some(Ball { mid, rad, noise })
    }

    fn add(&self, a: &Ball, b: &Ball) -> Option<Ball> {
        self.close(
            Complex::with_val_64(self.bits, &a.mid + &b.mid),
            Float::with_val_64(self.bits, &a.rad + &b.rad),
            Float::with_val_64(self.bits, &a.noise + &b.noise),
        )
    }

    fn mul(&self, a: &Ball, b: &Ball) -> Option<Ball> {
        let size_a = cnum::abs(&a.mid, self.bits);
        let size_b = cnum::abs(&b.mid, self.bits);
        let spread = |of_a: &Float, of_b: &Float| {
            Float::with_val_64(self.bits, &size_a * of_b)
                + Float::with_val_64(self.bits, &size_b * of_a)
                + Float::with_val_64(self.bits, of_a * of_b)
        };
        self.close(
            Complex::with_val_64(self.bits, &a.mid * &b.mid),
            spread(&a.rad, &b.rad),
            spread(&a.noise, &b.noise),
        )
    }

    fn exp(&self, a: &Ball) -> Option<Ball> {
        let mid = cnum::checked_exp(&a.mid, self.bits).ok()?;
        let size = cnum::abs(&mid, self.bits);
        let rad = Float::with_val_64(self.bits, &size * a.rad.clone().exp_m1());
        let noise = size * a.noise.clone().exp_m1();
        self.close(mid, rad, noise)
    }

    /// Twice a disk: exact.
    fn twice(&self, a: &Ball) -> Ball {
        Ball {
            mid: Complex::with_val_64(self.bits, &a.mid << 1u32),
            rad: Float::with_val_64(self.bits, &a.rad << 1u32),
            noise: Float::with_val_64(self.bits, &a.noise << 1u32),
        }
    }
}

/// Enclosures of `h` and `h'` over the disk of the given radius about `w`,
/// and of `h''` when it is asked for. With `A = w E_{k-1}`:
/// `E_k' = E_k A'` and `E_k'' = E_k (A'^2 + A'')`.
fn enclosure(
    w: &Complex,
    radius: &Float,
    levels: u32,
    bits: u64,
    curved: bool,
) -> Option<(Ball, Ball, Option<Ball>)> {
    let disks = Disks::new(bits);
    let centre = Ball {
        mid: Complex::with_val_64(bits, w),
        rad: Float::with_val_64(bits, radius),
        noise: Float::new_64(bits),
    };
    let mut tower = disks.exact(cnum::one(bits));
    let mut slope = disks.exact(cnum::zero(bits));
    let mut bend = curved.then(|| disks.exact(cnum::zero(bits)));
    for _ in 1..levels {
        let exponent = disks.mul(&centre, &tower)?;
        let rate = disks.add(&tower, &disks.mul(&centre, &slope)?)?;
        let turn = match &bend {
            Some(bend) => Some(disks.add(&disks.twice(&slope), &disks.mul(&centre, bend)?)?),
            None => None,
        };
        tower = disks.exp(&exponent)?;
        if let Some(turn) = turn {
            let square = disks.mul(&rate, &rate)?;
            bend = Some(disks.mul(&tower, &disks.add(&square, &turn)?)?);
        }
        slope = disks.mul(&tower, &rate)?;
    }
    let second = match &bend {
        Some(bend) => Some(disks.add(&disks.twice(&slope), &disks.mul(&centre, bend)?)?),
        None => None,
    };
    Some((
        disks.mul(&centre, &tower)?,
        disks.add(&tower, &disks.mul(&centre, &slope)?)?,
        second,
    ))
}

/// `h(w) = w E_{n-1}(w)` and `h'(w)`.
fn values(w: &Complex, levels: u32, bits: u64) -> Option<(Complex, Complex)> {
    let mut tower = cnum::one(bits);
    let mut slope = cnum::zero(bits);
    for _ in 1..levels {
        let exponent = Complex::with_val_64(bits, w * &tower);
        let rate = Complex::with_val_64(bits, w * &slope) + &tower;
        tower = cnum::checked_exp(&exponent, bits).ok()?;
        slope = Complex::with_val_64(bits, &tower * &rate);
    }
    let value = Complex::with_val_64(bits, w * &tower);
    let rate = Complex::with_val_64(bits, w * &slope) + &tower;
    (cnum::is_finite(&value) && cnum::is_finite(&rate)).then_some((value, rate))
}

/// The principal logarithm; a zero has no sign, so the negative real axis
/// has the imaginary part `pi`.
fn logarithm(z: &Complex, bits: u64) -> Complex {
    if !z.imag().is_zero() {
        return cnum::ln_complex(&Complex::with_val_64(bits, z), bits);
    }
    let size = Float::with_val_64(bits, z.real().abs_ref()).ln();
    let turn = if z.real().is_sign_negative() {
        Float::with_val_64(bits, Constant::Pi)
    } else {
        Float::new_64(bits)
    };
    Complex::with_val_64(bits, (size, turn))
}

fn trace() -> bool { std::env::var("TET_TRACE").is_ok() } // TRACE
const BRANCH_POINT: &str =
    "super-root: the path of the principal branch to this value runs into a \
                            branch point: the value lies on a branch cut, or closer to one than \
                            the precision can tell";

/// Where the root for a point of the path is the only one: what tells that
/// it is the continuation.
enum Fence {
    /// The disk of the point before, on which `h` is one-to-one.
    Disk(Complex, Float),
    /// A disk about a fold of the real axis, for a value below the fold's:
    /// its only root with a positive imaginary part.
    Above(Complex, Float),
    /// Such a disk for a value above the fold's: its only real root right
    /// of the fold, which lies within the last number of the centre.
    Right(Complex, Float, Float),
}

/// A certified disk about a point `(x, w)` of the path.
struct Hold {
    /// `h(w)` and `h'(w)` as computed.
    value: Complex,
    rate: Complex,
    /// The root for `x` lies within this of `w`.
    offset: Float,
    /// How far a step may take `x`.
    room: Float,
    /// A disk of twice the radius is likely to serve as well.
    roomy: bool,
}

/// What a try to pass a fold of the real axis gives.
enum Crossing {
    /// The next point of the path and its root, the fence of the root,
    /// whether the point is the end of the path, and a radius to try there.
    Landing(Complex, Complex, Fence, bool, Float),
    /// Nothing from here: the fold, when one was found, is too far for a
    /// disk about it that reaches the path.
    Far(Option<Float>),
    /// The end of the path cannot be told from the fold with these bits.
    Blind,
}

/// A point `(x, w)` of the continuation, `h(w) = x`, and how it got there.
struct Path {
    levels: u32,
    /// Working bits; raised next to a branch point, up to `limit`.
    bits: u64,
    limit: u64,
    x: Complex,
    w: Complex,
    /// Radius of a disk about `w` on which `h` is one-to-one and which holds
    /// the root for `x`; zero before the first step.
    radius: Float,
    steps: u64,
}

impl Path {
    fn new(levels: u32, bits: u64, limit: u64) -> Path {
        Path {
            levels,
            bits,
            limit,
            x: cnum::zero(bits),
            w: cnum::zero(bits),
            radius: Float::new_64(bits),
            steps: 0,
        }
    }

    /// The largest disk tried about `w` (the radius is halved until one
    /// serves) on which `h'` stays within half of `h'(w)`, and what it tells
    /// about the root for `x`. `None` when the point cannot be certified
    /// with these bits: no such disk, a root that may lie outside of the
    /// fence, or no room for a step.
    fn hold(
        &self,
        x: &Complex,
        w: &Complex,
        radius: &mut Float,
        fence: Option<&Fence>,
    ) -> Option<Hold> {
        let bits = self.bits;
        let size = usize::try_from(bits).ok()?;
        let disks = Disks::new(bits);
        let scale = cnum::abs(w, bits).max(&Float::with_val_64(bits, 1));
        let smallest = Float::with_val_64(bits, &scale >> size);
        let (value, rate, grip, roomy) = loop {
            if let Some((value, rate, _)) = enclosure(w, radius, self.levels, bits, false) {
                let floor = cnum::abs(&rate.mid, bits) / &disks.wide - &rate.noise;
                let spread = Float::with_val_64(bits, &rate.rad + &rate.noise);
                if Float::with_val_64(bits, &spread << 1u32) <= floor {
                    let roomy = Float::with_val_64(bits, &spread << 4u32)
                        <= Float::with_val_64(bits, &floor * 3u32);
                    break (value, rate.mid, floor - spread, roomy);
                }
            }
            *radius >>= 1u32;
            if *radius < smallest {
                return None;
            }
        };
        // `grip |u - v| <= |h(u) - h(v)|` on the disk: the root for `x` is
        // within `defect / grip` of `w`.
        let reach = Float::with_val_64(bits, &*radius * &grip) / &disks.wide;
        let defect = cnum::abs(&Complex::with_val_64(bits, &value.mid - x), bits) * &disks.wide
            + &value.noise;
        let offset = Float::with_val_64(bits, &defect / &grip) * &disks.wide;
        let far = |centre: &Complex| {
            cnum::abs(&Complex::with_val_64(bits, w - centre), bits) * &disks.wide + &offset
        };
        let inside = match fence {
            None => true,
            Some(Fence::Disk(centre, room)) => far(centre) < *room,
            Some(Fence::Above(centre, room)) => far(centre) < *room && *w.imag() > offset,
            Some(Fence::Right(centre, room, place)) => {
                far(centre) < *room
                    && w.imag().is_zero()
                    && Float::with_val_64(bits, w.real() - centre.real()) - &offset > *place
            }
        };
        let room = Float::with_val_64(bits, &reach - (reach.clone() >> 3u32)) - defect;
        (inside && room > 0).then_some(Hold {
            value: value.mid,
            rate,
            offset,
            room,
            roomy,
        })
    }

    /// Continue the root along the straight segment from `x` to `to`. With
    /// `folds` the segment lies on the real axis: the root is real up to a
    /// fold, and the path passes above the fold.
    fn advance(&mut self, to: &Complex, folds: bool) -> Result<(), String> {
        // The last certified point with the radius of its disk.
        let mut anchor: Option<(Complex, Complex, Float)> = None;
        let mut fence: Option<Fence> = None;
        // The real point before this one with its `h'`: two of them tell
        // where `h'` vanishes.
        let mut trail: Option<(Float, Float)> = None;
        // The fold ahead, once it is known, and how far it was at the last
        // try to pass it: the next try is at half of that.
        let mut brink: Option<(Option<Float>, Float)> = None;
        let mut radius = self.first_radius();
        let mut arrived = false;
        loop {
            let bits = self.bits;
            let disks = Disks::new(bits);
            let x = Complex::with_val_64(bits, &self.x);
            let w = Complex::with_val_64(bits, &self.w);
            radius.set_prec_64(bits);
            // At the end `x` is the target rounded to these bits: the target
            // itself must have its root in the disk too.
            let held = self
                .hold(&x, &w, &mut radius, fence.as_ref())
                .filter(|hold| {
                    !arrived
                        || Float::with_val_64(bits, cnum::abs(to, bits) * &disks.ulp) <= hold.room
                });
            let Some(hold) = held else {
                if trace() { eprintln!("T FAIL at step {} bits {} x {:.12e} w {:.12e} {:.6e} r {:.3e}", self.steps, bits, x.real().to_f64(), w.real().to_f64(), w.imag().to_f64(), radius.to_f64()); } // TRACE
                self.more_bits()?;
                // Back to the last certified point, which stays certified.
                radius = match anchor.take() {
                    Some((x, w, room)) => {
                        self.x = x;
                        self.w = w;
                        room
                    }
                    None => self.first_radius(),
                };
                fence = None;
                trail = None;
                brink = None;
                arrived = false;
                continue;
            };
            self.radius = radius.clone();
            if trace() { eprintln!("T step {} bits {} x {:.12e} w {:.12e} {:.6e} r {:.3e} room {:.3e} off {:.3e} roomy {} arrived {}", self.steps, bits, x.real().to_f64(), w.real().to_f64(), w.imag().to_f64(), radius.to_f64(), hold.room.to_f64(), hold.offset.to_f64(), hold.roomy, arrived); } // TRACE
            if arrived {
                return Ok(());
            }
            let real = folds && w.imag().is_zero();
            if real {
                let rate = hold.rate.real().clone();
                // The fold as it is known, or where the secant of `h'`
                // through the two points vanishes while `h'` falls to zero.
                let aim = match (&brink, &trail) {
                    (Some((Some(centre), _)), _) => Some((centre.clone(), true)),
                    (_, Some((before, slope))) if rate > 0 && rate < *slope => {
                        let run = Float::with_val_64(bits, w.real() - before);
                        let fall = Float::with_val_64(bits, &rate - slope);
                        let ahead = Float::with_val_64(bits, &rate * &run) / fall;
                        Some((Float::with_val_64(bits, w.real() - ahead), false))
                    }
                    _ => None,
                };
                trail = Some((w.real().clone(), rate));
                if let Some((aim, known)) = aim {
                    let gap = Float::with_val_64(bits, w.real() - &aim);
                    let due = gap > 0
                        && brink.as_ref().is_none_or(|(_, last)| {
                            Float::with_val_64(bits, &gap << 1u32) <= *last
                        });
                    if due {
                        match self.fold(&x, &w, &aim, known, to, &hold) {
                            Crossing::Landing(goal, root, across, end, room) => {
                                if trace() { eprintln!("T   landing goal {:.12e} root {:.12e} {:.6e} end {} room {:.3e}", goal.real().to_f64(), root.real().to_f64(), root.imag().to_f64(), end, room.to_f64()); } // TRACE
                                anchor = Some((x, w, radius.clone()));
                                fence = Some(across);
                                trail = None;
                                brink = None;
                                self.x = goal;
                                self.w = root;
                                self.steps += 1;
                                arrived = end;
                                radius = room;
                                continue;
                            }
                            Crossing::Blind => {
                                if trace() { eprintln!("T   blind"); } // TRACE
                                // The same point with more bits.
                                self.more_bits()?;
                                trail = None;
                                brink = None;
                                continue;
                            }
                            Crossing::Far(centre) => {
                                if trace() { eprintln!("T   far {:?} gap {:.3e}", centre.as_ref().map(|c| c.to_f64()), gap.to_f64()); } // TRACE
                                let gap = match &centre {
                                    Some(centre) => Float::with_val_64(bits, w.real() - centre),
                                    None => gap,
                                };
                                brink = Some((centre, gap));
                            }
                        }
                    }
                }
            }
            let gap = Complex::with_val_64(bits, to - &x);
            // The end of the path is `to` rounded to these bits.
            let remaining = cnum::abs(&gap, bits) + cnum::abs(to, bits) * &disks.ulp;
            arrived = remaining <= hold.room;
            let goal = if arrived {
                Complex::with_val_64(bits, to)
            } else {
                x.clone() + gap * Float::with_val_64(bits, &hold.room / &remaining)
            };
            if !arrived && goal == x {
                if trace() { eprintln!("T   stuck"); } // TRACE
                // Next to a branch point the steps shrink with its
                // distance; this one is below the rounding of `x`.
                self.more_bits()?;
                trail = None;
                brink = None;
                continue;
            }
            let mut next = self.solve(&goal, &w, &radius, &hold.value, &hold.rate)?;
            if real {
                // A real value has a real root in a disk about a real point
                // that holds one root only.
                *next.mut_imag() = Float::new_64(bits);
            }
            fence = Some(Fence::Disk(w.clone(), radius.clone()));
            anchor = Some((x, w, radius.clone()));
            self.x = goal;
            self.w = next;
            self.steps += 1;
            if hold.roomy {
                radius <<= 1u32;
            }
        }
    }

    /// Twice the bits for the continuation, while that is allowed.
    fn more_bits(&mut self) -> Result<(), String> {
        if self.bits >= self.limit {
            return Err(BRANCH_POINT.into());
        }
        self.bits = self.bits.saturating_mul(2).min(self.limit);
        if cnum::verbose() {
            eprintln!(
                "sroot: next to a branch point at step {}; continuing with {} bits",
                self.steps, self.bits
            );
        }
        Ok(())
    }

    fn first_radius(&self) -> Float {
        if self.radius.is_zero() {
            Float::with_val_64(self.bits, 1)
        } else {
            self.radius.clone()
        }
    }

    /// The root of `h(v) = goal` in the disk about `w`, for a goal that the
    /// chord iteration with the slope `rate = h'(w)` reaches: Newton steps,
    /// and a chord step wherever a Newton step would leave the disk. The
    /// root is as accurate as the next disk needs it: its steps are
    /// followed to `2^-30` of the radius, and the next one is their square.
    fn solve(
        &self,
        goal: &Complex,
        w: &Complex,
        radius: &Float,
        value: &Complex,
        rate: &Complex,
    ) -> Result<Complex, String> {
        let bits = self.bits;
        let converged = Float::with_val_64(bits, radius >> 30u32);
        let mut v = w.clone() + Complex::with_val_64(bits, goal - value) / rate;
        let mut before: Option<Float> = None;
        let mut count = 0u64;
        loop {
            let (image, slope) = values(&v, self.levels, bits)
                .ok_or("super-root: the tower leaves the number range next to the root")?;
            let defect = Complex::with_val_64(bits, &image - goal);
            if cnum::is_zero(&defect) {
                return Ok(v);
            }
            let mut next = v.clone() - Complex::with_val_64(bits, &defect / &slope);
            let within = cnum::is_finite(&next)
                && cnum::abs(&Complex::with_val_64(bits, &next - w), bits) < *radius;
            if !within {
                next = v.clone() - Complex::with_val_64(bits, &defect / rate);
            }
            let step = cnum::abs(&Complex::with_val_64(bits, &next - &v), bits);
            v = next;
            // Small enough, or at the rounding of the tower: no longer shrinking.
            if step <= converged || before.as_ref().is_some_and(|last| step >= *last) {
                return Ok(v);
            }
            before = Some(step);
            count += 1;
            if count > bits.saturating_mul(4) {
                return Err("super-root: the iteration inside a disk did not settle".into());
            }
        }
    }

    /// Newton's iteration for `h(v) = goal` from a point next to the root.
    fn newton(&self, goal: &Complex, start: Complex) -> Option<Complex> {
        let bits = self.bits;
        let size = usize::try_from(bits).ok()?;
        let converged = Float::with_val_64(bits, 1) >> size.saturating_sub(12);
        let mut v = start;
        let mut before: Option<Float> = None;
        for _ in 0..bits {
            let (image, slope) = values(&v, self.levels, bits)?;
            let step = Complex::with_val_64(bits, &image - goal) / slope;
            let moved = cnum::abs(&step, bits);
            v -= step;
            if !cnum::is_finite(&v) {
                return None;
            }
            if moved <= Float::with_val_64(bits, cnum::abs(&v, bits) * &converged)
                || before.as_ref().is_some_and(|last| moved >= *last)
            {
                return Some(v);
            }
            before = Some(moved);
        }
        None
    }

    /// From the real point `(x, w)` to the other side of the fold of `h`
    /// next to `aim` (the fold itself when it is `known`), or to the end of
    /// the path when that lies before the fold.
    fn fold(
        &self,
        x: &Complex,
        w: &Complex,
        aim: &Float,
        known: bool,
        to: &Complex,
        hold: &Hold,
    ) -> Crossing {
        // `h'` has the sign of `h'(w)` on the disk of `w`, which holds the
        // root: positive there, the root lies right of the fold found here,
        // the only zero of `h'` on the real axis of its disk.
        if !(*hold.rate.real() > 0) {
            return Crossing::Far(None);
        }
        let centre = if known {
            Some(aim.clone())
        } else {
            self.critical(aim)
        };
        let Some(centre) = centre.filter(|centre| centre < w.real()) else {
            return Crossing::Far(None);
        };
        match self.cross(x, w, &centre, to, hold) {
            Some(crossing) => crossing,
            None => Crossing::Far(Some(centre)),
        }
    }

    /// The zero of `h'` next to `start` on the real axis, where `h''` is
    /// positive, by Newton's iteration.
    fn critical(&self, start: &Float) -> Option<Float> {
        let bits = self.bits;
        let size = usize::try_from(bits).ok()?;
        let point = Float::new_64(bits);
        let converged = Float::with_val_64(bits, 1) >> size.saturating_sub(12);
        let mut centre = start.clone();
        let mut before: Option<Float> = None;
        for _ in 0..24 {
            let at = Complex::with_val_64(bits, (&centre, 0));
            let (_, first, second) = enclosure(&at, &point, self.levels, bits, true)?;
            let curve = second?.mid;
            if !(*curve.real() > 0) {
                return None;
            }
            let step = Float::with_val_64(bits, first.mid.real() / curve.real());
            centre -= &step;
            if !centre.is_finite() {
                return None;
            }
            let moved = step.abs();
            if moved <= Float::with_val_64(bits, centre.clone().abs() * &converged)
                || before.as_ref().is_some_and(|last| moved >= *last)
            {
                return Some(centre);
            }
            before = Some(moved);
        }
        None
    }

    /// The enclosures of `h` and `h'` over the disk of the radius `reach`
    /// about the real `centre`, `h''` at the centre and a lower bound of the
    /// real part of `h''` on the disk, when `h''` stays within a quarter of
    /// its value at the centre there.
    fn curvature(&self, centre: &Float, reach: &Float) -> Option<(Ball, Ball, Float, Float)> {
        let bits = self.bits;
        let disks = Disks::new(bits);
        let at = Complex::with_val_64(bits, (centre, 0));
        let (value, first, second) = enclosure(&at, reach, self.levels, bits, true)?;
        let second = second?;
        let floor = Float::with_val_64(bits, second.mid.real() / &disks.wide) - &second.noise;
        let spread = Float::with_val_64(bits, &second.rad + &second.noise);
        (Float::with_val_64(bits, &spread << 2u32) <= floor).then(|| {
            let curve = second.mid.real().clone();
            (value, first, curve, floor - spread)
        })
    }

    /// The step across the fold next to `centre`, left of `w`; `None` when
    /// it cannot be certified from here.
    fn cross(
        &self,
        x: &Complex,
        w: &Complex,
        centre: &Float,
        to: &Complex,
        hold: &Hold,
    ) -> Option<Crossing> {
        let bits = self.bits;
        let disks = Disks::new(bits);
        let real = |value: &Float| Complex::with_val_64(bits, (value, 0));
        let gap = Float::with_val_64(bits, w.real() - centre);
        // The largest disk tried about it, of twice, four and eight times
        // the distance of `w`, on which `h''` stays within a quarter of its
        // value at the centre: `bound <= Re h''` there.
        let mut reach = Float::with_val_64(bits, &gap << 1u32);
        let mut widest = None;
        for _ in 0..3 {
            match self.curvature(centre, &reach) {
                Some(found) => widest = Some((found, reach.clone())),
                None => break,
            }
            reach <<= 1u32;
        }
        let ((value, first, curve, bound), reach) = widest?;
        // `h'` rises at least by `bound` a unit on the real axis of the
        // disk: its zero, the fold, lies within `place` of the centre, and
        // the value there within `band` of the value at the centre.
        let slant = cnum::abs(&first.mid, bits) * &disks.wide + &first.noise;
        let place = Float::with_val_64(bits, &slant / &bound) * &disks.wide;
        let band = Float::with_val_64(bits, &slant * &place) * &disks.wide + &value.noise;
        if Float::with_val_64(bits, &place << 3u32) >= reach {
            return None;
        }
        let room = Float::with_val_64(bits, &reach - &place);
        if trace() { eprintln!("T   fold centre {:.15e} gap {:.3e} reach {:.3e} place {:.3e} band {:.3e} slant {:.3e} bound {:.3e} off {:.3e}", centre.to_f64(), gap.to_f64(), reach.to_f64(), place.to_f64(), band.to_f64(), slant.to_f64(), bound.to_f64(), hold.offset.to_f64()); } // TRACE
        // The root of the path lies in the disk about the fold.
        if Float::with_val_64(bits, &gap + &hold.offset) + &place >= room {
            return None;
        }
        // Every value within `span` of the fold's has two roots within
        // `room` of the fold, and no more: real ones on the two sides of the
        // fold for a value above, a conjugate pair for a value below.
        let span = (Float::with_val_64(bits, &bound * room.clone().square()) / &disks.wide) >> 1u32;
        let level = value.mid.real().clone();
        let entry = Float::with_val_64(bits, x.real() - &level).abs() + &band;
        if entry >= span || Float::with_val_64(bits, &band << 3u32) >= span {
            return None;
        }
        let end = Float::with_val_64(bits, to.real());
        let ahead = Float::with_val_64(bits, &end - &level);
        if trace() { eprintln!("T   fold span {:.3e} entry {:.3e} ahead {:.3e}", span.to_f64(), entry.to_f64(), ahead.to_f64()); } // TRACE
        let fold = real(centre);
        if ahead > band {
            // The end of the path lies before the fold: the real root right
            // of it, where `h` rises.
            if Float::with_val_64(bits, &ahead + &band) >= span {
                return None;
            }
            let lift = (Float::with_val_64(bits, &ahead << 1u32) / &curve).sqrt();
            let start = real(&Float::with_val_64(bits, centre + &lift));
            let mut root = self.newton(&real(&end), start)?;
            let beyond = Float::with_val_64(bits, root.real() - centre);
            if !root.imag().is_zero() || beyond <= place {
                return None;
            }
            *root.mut_imag() = Float::new_64(bits);
            let across = Fence::Right(fold, room, place);
            return Some(Crossing::Landing(real(&end), root, across, true, beyond >> 1u32));
        }
        if Float::with_val_64(bits, -&ahead) > band {
            // Past the fold: the end of the path when it is near enough,
            // three quarters of the way out otherwise.
            let far = Float::with_val_64(bits, -&ahead) + &band;
            let near = Float::with_val_64(bits, &span - (span.clone() >> 3u32));
            let (goal, last) = if far < near {
                (end, true)
            } else {
                (
                    level.clone() - Float::with_val_64(bits, &span - (span.clone() >> 2u32)),
                    false,
                )
            };
            let depth = Float::with_val_64(bits, &level - &goal);
            let lift = (Float::with_val_64(bits, &depth << 1u32) / &curve).sqrt();
            let start = Complex::with_val_64(bits, (centre, &lift));
            let mut root = self.newton(&real(&goal), start)?;
            if root.imag().is_sign_negative() {
                root = root.conj();
            }
            if !(*root.imag() > 0) {
                return None;
            }
            let room_there = Float::with_val_64(bits, root.imag() >> 1u32);
            // Within `room` of the fold, which is within `place` of the centre.
            let across = Fence::Above(fold, room - place);
            return Some(Crossing::Landing(real(&goal), root, across, last, room_there));
        }
        Some(Crossing::Blind)
    }
}

/// Newton's iteration for `h(w) = log z` at `bits` bits from a root known
/// to fewer, doubling the bits on the way.
fn refine(z: &Complex, start: &Complex, levels: u32, bits: u64) -> Result<Complex, String> {
    let x = logarithm(z, bits);
    let size = usize::try_from(bits).map_err(|_| "precision overflow")?;
    let converged = Float::with_val_64(bits, 1) >> size.saturating_sub(10);
    let mut have = start.prec_64().0.min(bits);
    let mut w = start.clone();
    let mut before: Option<Float> = None;
    let mut count = 0u64;
    loop {
        have = have.saturating_mul(2).min(bits);
        let at = Complex::with_val_64(have, &w);
        let (image, slope) = values(&at, levels, have)
            .ok_or("super-root: the tower leaves the number range at the root")?;
        let step = Complex::with_val_64(have, &image - &x) / slope;
        let moved = cnum::abs(&step, have);
        w = at - step;
        if have < bits {
            continue;
        }
        if moved <= Float::with_val_64(bits, cnum::abs(&w, bits) * &converged) {
            return Ok(w);
        }
        if before.as_ref().is_some_and(|last| moved >= *last) {
            return Ok(w);
        }
        before = Some(moved);
        count += 1;
        if count > bits {
            return Err("super-root: the root did not settle at the working precision".into());
        }
    }
}

/// How a part of `e^w` stands next to zero: the distance of `|Im w|` from
/// the nearer of `pi/2` and `pi`, where the real or the imaginary part of
/// `e^w` vanishes, as `(distance, at pi)`.
fn edge(w: &Complex, bits: u64) -> (Float, bool) {
    let half = Float::with_val_64(bits, Constant::Pi) >> 1u32;
    let side = Float::with_val_64(bits, w.imag().abs_ref());
    let upright = Float::with_val_64(bits, &side - &half).abs();
    let flat = Float::with_val_64(bits, &side - (half << 1u32)).abs();
    if flat <= upright {
        (flat, true)
    } else {
        (upright, false)
    }
}

/// The principal super-root of `z` at the integer height `levels >= 2`, to
/// `digits` digits of each of its parts at `work` bits.
pub fn principal(z: &Complex, levels: u32, work: u64, digits: u64) -> Result<Complex, String> {
    cnum::require_precision(work, digits)?;
    if levels < 2 || !cnum::is_finite(z) || cnum::is_zero(z) {
        return Err(
            "super-root: the principal branch needs a height from 2 and a value that is \
                    not 0"
                .into(),
        );
    }
    let most = rug::float::prec_max_64();
    let track = work.min(cnum::checked_digits_to_bits(TRACK_DIGITS)?);
    let limit = work
        .saturating_mul(2)
        .saturating_add(track.saturating_mul(2))
        .min(most);
    let target = logarithm(z, work);
    let mut path = Path::new(levels, track, limit);
    // A positive value: the path lies on the real axis and passes above a
    // fold, which is the limit from above.
    let on_axis = z.imag().is_zero() && !z.real().is_sign_negative();
    path.advance(&target, on_axis)?;
    if cnum::verbose() {
        eprintln!(
            "sroot: principal branch at the height {levels}: {} certified steps, {} bits",
            path.steps, path.bits
        );
    }
    let mut root = path.w.clone();
    let tracked = root.clone();
    // The parts of `e^w` are those of `w` except next to `Im w = pi/2` and
    // `pi`, where one of them is small against the other: there the
    // distance of `Im w` from that line must have the digits. With 16
    // times the working bits it still may not: then that part is zero.
    let cap = work.saturating_mul(16).min(most);
    let mut bits = work;
    let mut vanished = None;
    loop {
        root = refine(z, &root, levels, bits)?;
        let moved = cnum::abs(
            &Complex::with_val_64(path.bits, &root - &tracked),
            path.bits,
        );
        if Float::with_val_64(path.bits, &moved << 1u32) >= path.radius {
            return Err("super-root: the root left the disk of the continuation".into());
        }
        let (distance, flat) = edge(&root, bits);
        let scale = cnum::abs(&root, bits).max(&Float::with_val_64(bits, 1));
        let lost = match (cnum::exponent(&scale), cnum::exponent(&distance)) {
            (Some(above), Some(below)) => u64::try_from((above - below).max(0)).ok(),
            _ => None,
        };
        // 32 bits of the guard may go.
        let needed = match lost {
            Some(lost) => work.saturating_add(lost.saturating_sub(32)),
            None => bits.saturating_mul(2),
        };
        if needed <= bits {
            break;
        }
        if bits >= cap {
            vanished = Some(flat);
            break;
        }
        bits = needed.min(cap);
        if cnum::verbose() {
            eprintln!("sroot: a part of the base is small; root with {bits} bits");
        }
    }
    let size = usize::try_from(bits).map_err(|_| "precision overflow")?;
    let scale = cnum::abs(&root, bits).max(&Float::with_val_64(bits, 1));
    let margin = Float::with_val_64(bits, &scale >> size.saturating_sub(8));
    let side = Float::with_val_64(bits, root.imag().abs_ref());
    if side > Float::with_val_64(bits, Constant::Pi) + margin {
        return Err(format!(
            "super-root: the principal branch has no base for this value at the height \
             {levels}: the root has left the plane of bases cut along the negative real axis \
             (the imaginary part of its logarithm is {:.6})",
            DisplayFloat(root.imag())
        ));
    }
    let mut base = cnum::checked_exp(&root, bits)?;
    match vanished {
        Some(true) => *base.mut_imag() = Float::new_64(bits),
        Some(false) => *base.mut_real() = Float::new_64(bits),
        None => {}
    }
    Ok(Complex::with_val_64(work, base))
}

/// The height as a number of levels, for the heights of the principal branch.
pub(crate) fn levels(height: &Complex) -> Option<u32> {
    if !height.imag().is_zero() || !height.real().is_integer() {
        return None;
    }
    let levels = height.real().to_integer()?.to_u32()?;
    (2..=LARGEST_HEIGHT).contains(&levels).then_some(levels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{integer_height, lambertw};

    fn point(re: &str, im: &str, bits: u64) -> Complex {
        cnum::parse_complex(re, im, bits).unwrap()
    }

    fn close(a: &Complex, b: &Complex, bits: u64, digits: u64) -> bool {
        let gap = cnum::abs(&Complex::with_val_64(bits, a - b), bits);
        gap <= cnum::abs(b, bits) * cnum::epsilon(digits, bits)
    }

    #[test]
    fn enclosures_hold_the_values_on_their_disks() {
        let bits = cnum::digits_to_bits(TRACK_DIGITS);
        for (levels, centre, radius) in [
            (2u32, ("-0.9", "0.1"), "0.05"),
            (3, ("0.4", "0.7"), "0.02"),
            (4, ("-0.7", "2.6"), "0.01"),
            (5, ("0.3", "-0.4"), "0.03"),
            (7, ("-1.2", "0.9"), "0.05"),
        ] {
            let w = point(centre.0, centre.1, bits);
            let r = cnum::decimal(radius, bits);
            let (value, rate, bend) = enclosure(&w, &r, levels, bits, true).unwrap();
            let bend = bend.unwrap();
            let (at, slope) = values(&w, levels, bits).unwrap();
            assert_eq!(value.mid, at);
            assert_eq!(rate.mid, slope);
            let none = Float::new_64(bits);
            let second = |v: &Complex| enclosure(v, &none, levels, bits, true).unwrap().2.unwrap();
            // Points of the disk, on its edge and inside.
            for k in 0..24u32 {
                let angle = Float::with_val_64(bits, Constant::Pi) * k / 12u32;
                let turn = Complex::with_val_64(bits, (angle.clone().cos(), angle.sin()));
                let reach = if k % 3 == 0 {
                    r.clone() >> 1u32
                } else {
                    r.clone()
                };
                let v = w.clone() + turn * reach;
                let (image, incline) = values(&v, levels, bits).unwrap();
                let curve = second(&v).mid;
                for (ball, member) in [(&value, &image), (&rate, &incline), (&bend, &curve)] {
                    let off = cnum::abs(&Complex::with_val_64(bits, member - &ball.mid), bits);
                    assert!(off <= ball.rad, "height {levels}, point {k}");
                    assert!(ball.noise <= ball.rad);
                }
            }
        }
    }

    #[test]
    fn the_second_derivative_is_the_slope_of_the_first() {
        let bits = cnum::digits_to_bits(60);
        let none = Float::new_64(bits);
        // A central difference over `2^-40` is off by about `2^-80`.
        let width = Float::with_val_64(bits, 1) >> 40u32;
        for (levels, re, im) in [
            (2u32, "-0.9", "0.1"),
            (3, "0.4", "0.7"),
            (4, "-1.3", "0"),
            (6, "0.25", "-0.3"),
        ] {
            let w = point(re, im, bits);
            let bend = enclosure(&w, &none, levels, bits, true)
                .unwrap()
                .2
                .unwrap()
                .mid;
            let (_, ahead) =
                values(&Complex::with_val_64(bits, &w + &width), levels, bits).unwrap();
            let (_, behind) =
                values(&Complex::with_val_64(bits, &w - &width), levels, bits).unwrap();
            let slope = Complex::with_val_64(bits, &ahead - &behind) / (width.clone() << 1u32);
            assert!(close(&bend, &slope, bits, 20), "height {levels}");
        }
    }

    #[test]
    fn the_recursion_is_the_tower_of_the_base() {
        let bits = cnum::digits_to_bits(40);
        for (levels, re, im) in [
            (2u32, "0.3", "0.4"),
            (3, "-0.2", "1.1"),
            (6, "0.25", "-0.3"),
        ] {
            let w = point(re, im, bits);
            let (value, _) = values(&w, levels, bits).unwrap();
            let base = cnum::checked_exp(&w, bits).unwrap();
            let tower = integer_height::tetrate_integer(&base, i64::from(levels), bits).unwrap();
            let image = cnum::checked_exp(&value, bits).unwrap();
            assert!(close(&image, &tower, bits, 36), "height {levels}");
        }
    }

    #[test]
    fn the_height_two_is_the_lambert_function() {
        let digits = 30u64;
        let bits = cnum::digits_to_bits(digits + 10);
        for (re, im) in [
            ("3", "2"),
            ("-1", "0"),
            ("-0.000000001", "0"),
            ("0.2", "-0.7"),
            ("-5", "-0.001"),
            ("1e-40", "1e-41"),
            ("-1e-300", "0"),
            // On the cut of the function: the values below `e^(-1/e)`.
            ("0.5", "0"),
            ("0.6922", "0"),
            ("0.6923", "0"),
            ("0.9", "0"),
            ("1e-30", "0"),
        ] {
            let z = point(re, im, bits);
            let found = principal(&z, 2, bits, digits).unwrap();
            let w = lambertw::w0(&logarithm(&z, bits), bits).unwrap();
            let expected = cnum::checked_exp(&w, bits).unwrap();
            assert!(close(&found, &expected, bits, digits), "z = {re} + {im}i");
        }
    }
}
