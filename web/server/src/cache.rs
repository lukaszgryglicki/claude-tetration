//! The results computed since the server started, in memory only: nothing
//! is written anywhere, and a restart begins with an empty cache.
//!
//! A computation is entered here when it is asked for first and stays
//! entered while it runs, however long that takes and whether or not the
//! requests for it still wait: the same computation is never started
//! twice. The cache is bounded in bytes, because anybody can ask for any
//! number of different things and a result can be megabytes long; beyond
//! the bound the results that were used longest ago are dropped.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Instant;

use crate::api::Outcome;

/// How a request came to its computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The outcome was in the cache.
    Hit,
    /// This request started the computation.
    Miss,
    /// An earlier request had started it.
    Wait,
}

impl Source {
    pub fn header(self) -> &'static str {
        match self {
            Source::Hit => "HIT",
            Source::Miss => "MISS",
            Source::Wait => "WAIT",
        }
    }
}

/// A computation that is under way, and its outcome once it is there.
pub struct Running {
    asked: Instant,
    outcome: Mutex<Option<Arc<Outcome>>>,
    done: Condvar,
}

impl Running {
    /// When the computation was asked for first.
    pub fn asked(&self) -> Instant {
        self.asked
    }

    fn outcome(&self) -> MutexGuard<'_, Option<Arc<Outcome>>> {
        self.outcome
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The outcome, if it is there by `deadline`; `None` when the
    /// computation still runs then.
    pub fn awaited(&self, deadline: Instant) -> Option<Arc<Outcome>> {
        let mut outcome = self.outcome();
        loop {
            if let Some(outcome) = outcome.as_ref() {
                return Some(outcome.clone());
            }
            let left = deadline
                .checked_duration_since(Instant::now())
                .filter(|left| !left.is_zero())?;
            outcome = self
                .done
                .wait_timeout(outcome, left)
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .0;
        }
    }
}

enum Entry {
    Running(Arc<Running>),
    Kept {
        outcome: Arc<Outcome>,
        bytes: usize,
        used: u64,
    },
}

#[derive(Default)]
struct State {
    entries: HashMap<String, Entry>,
    /// The kept entries by the tick of their last use.
    order: BTreeMap<u64, String>,
    tick: u64,
    bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub entries: usize,
    pub bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub waits: u64,
    pub dropped: u64,
}

pub struct Cache {
    limit: usize,
    state: Mutex<State>,
    counts: Mutex<Counts>,
}

/// What an entry costs: its texts and a round figure for the bookkeeping.
fn cost(key: &str, outcome: &Outcome) -> usize {
    let texts = match outcome {
        Outcome::Value { re, im, .. } => re.len() + im.len(),
        Outcome::Refused { message, .. } | Outcome::Failed { message } => message.len(),
        Outcome::Timeout { .. } | Outcome::Busy => 0,
    };
    2 * key.len() + texts + 256
}

/// What the cache has for a key.
pub enum Found {
    /// The outcome of a computation that ended.
    Kept(Arc<Outcome>),
    /// The computation that an earlier request started.
    Running(Arc<Running>),
    /// Nothing. The computation is now entered as under way, and the ticket
    /// is the duty to compute it.
    New(Ticket),
}

/// The duty to compute an entry. A ticket that is dropped without an
/// outcome, as by a panic, leaves a failure that is not kept, so that the
/// requests for the entry are not left waiting.
pub struct Ticket {
    cache: Arc<Cache>,
    key: String,
    running: Arc<Running>,
    finished: bool,
}

impl Ticket {
    pub fn running(&self) -> Arc<Running> {
        self.running.clone()
    }

    /// Hands the outcome to the requests that wait and keeps it if it
    /// lasts.
    pub fn finish(mut self, outcome: Outcome) {
        self.finished = true;
        self.cache
            .finish(&self.key, &self.running, Arc::new(outcome));
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        if !self.finished {
            let failed = Outcome::Failed {
                message: "the computation was interrupted".into(),
            };
            self.cache
                .finish(&self.key, &self.running, Arc::new(failed));
        }
    }
}

impl Cache {
    pub fn new(limit_bytes: usize) -> Cache {
        Cache {
            limit: limit_bytes,
            state: Mutex::new(State::default()),
            counts: Mutex::new(Counts::default()),
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        // A panic while the lock was held leaves consistent maps behind:
        // every change below is completed before the lock is released.
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn count(&self, change: impl FnOnce(&mut Counts)) {
        change(
            &mut self
                .counts
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
    }

    pub fn counts(&self) -> Counts {
        let state = self.state();
        let mut counts = *self
            .counts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        counts.entries = state.order.len();
        counts.bytes = state.bytes;
        counts
    }

    /// What there is for `key`: the kept outcome, the computation under
    /// way, or the ticket for a computation that is entered by this call.
    pub fn find(self: &Arc<Cache>, key: &str) -> Found {
        let found = {
            let mut guard = self.state();
            let state = &mut *guard;
            state.tick += 1;
            let tick = state.tick;
            match state.entries.get_mut(key) {
                Some(Entry::Kept { outcome, used, .. }) => {
                    state.order.remove(used);
                    *used = tick;
                    state.order.insert(tick, key.to_string());
                    Found::Kept(outcome.clone())
                }
                Some(Entry::Running(running)) => Found::Running(running.clone()),
                None => {
                    let running = Arc::new(Running {
                        asked: Instant::now(),
                        outcome: Mutex::new(None),
                        done: Condvar::new(),
                    });
                    state
                        .entries
                        .insert(key.to_string(), Entry::Running(running.clone()));
                    Found::New(Ticket {
                        cache: self.clone(),
                        key: key.to_string(),
                        running,
                        finished: false,
                    })
                }
            }
        };
        self.count(|counts| match found {
            Found::Kept(_) => counts.hits += 1,
            Found::Running(_) => counts.waits += 1,
            Found::New(_) => counts.misses += 1,
        });
        found
    }

    /// Drops the kept `outcome` of `key`, if that is still what is kept:
    /// it has stopped being what a computation would give now.
    pub fn forget(&self, key: &str, outcome: &Arc<Outcome>) {
        let mut guard = self.state();
        let state = &mut *guard;
        let (bytes, used) = match state.entries.get(key) {
            Some(Entry::Kept {
                outcome: kept,
                bytes,
                used,
            }) if Arc::ptr_eq(kept, outcome) => (*bytes, *used),
            _ => return,
        };
        state.order.remove(&used);
        state.bytes -= bytes;
        state.entries.remove(key);
    }

    fn finish(&self, key: &str, running: &Running, outcome: Arc<Outcome>) {
        let mut dropped = 0;
        {
            let mut state = self.state();
            state.entries.remove(key);
            let bytes = cost(key, &outcome);
            if outcome.lasting() && bytes <= self.limit {
                state.tick += 1;
                let used = state.tick;
                state.order.insert(used, key.to_string());
                state.bytes += bytes;
                state.entries.insert(
                    key.to_string(),
                    Entry::Kept {
                        outcome: outcome.clone(),
                        bytes,
                        used,
                    },
                );
                while state.bytes > self.limit {
                    let Some((_, oldest)) = state.order.pop_first() else {
                        break;
                    };
                    if let Some(Entry::Kept { bytes, .. }) = state.entries.remove(&oldest) {
                        state.bytes -= bytes;
                        dropped += 1;
                    }
                }
            }
        }
        if dropped > 0 {
            self.count(|counts| counts.dropped += dropped);
        }
        *running.outcome() = Some(outcome);
        running.done.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    const LONG: Duration = Duration::from_secs(30);

    fn value(text: &str) -> Outcome {
        Outcome::Value {
            re: text.to_string(),
            im: "0".to_string(),
            seconds: 0.5,
        }
    }

    /// Asks as a request does: a computation that nobody asked for before
    /// runs in a thread of its own, and the request waits `patience` for
    /// its outcome.
    fn ask(
        cache: &Arc<Cache>,
        key: &str,
        patience: Duration,
        compute: impl FnOnce() -> Outcome + Send + 'static,
    ) -> (Option<Arc<Outcome>>, Source) {
        let (running, source) = match cache.find(key) {
            Found::Kept(outcome) => return (Some(outcome), Source::Hit),
            Found::Running(running) => (running, Source::Wait),
            Found::New(ticket) => {
                let running = ticket.running();
                std::thread::spawn(move || {
                    let outcome = compute();
                    ticket.finish(outcome);
                });
                (running, Source::Miss)
            }
        };
        (running.awaited(Instant::now() + patience), source)
    }

    #[test]
    fn a_result_is_computed_once_and_then_kept() {
        let cache = Arc::new(Cache::new(1 << 20));
        let runs = Arc::new(AtomicUsize::new(0));
        let request = |key: &'static str| {
            let runs = runs.clone();
            ask(&cache, key, LONG, move || {
                runs.fetch_add(1, Ordering::SeqCst);
                value(key)
            })
        };
        assert_eq!(request("a"), (Some(Arc::new(value("a"))), Source::Miss));
        assert_eq!(request("a"), (Some(Arc::new(value("a"))), Source::Hit));
        assert_eq!(request("b").1, Source::Miss);
        assert_eq!(request("a").1, Source::Hit);
        assert_eq!(runs.load(Ordering::SeqCst), 2);
        let counts = cache.counts();
        assert_eq!(
            (counts.entries, counts.hits, counts.misses, counts.waits),
            (2, 2, 2, 0)
        );
        assert_eq!(counts.bytes, 2 * (2 + 2 + 256));
    }

    #[test]
    fn the_results_used_longest_ago_are_dropped_beyond_the_limit() {
        // Room for three entries of this size.
        let cache = Arc::new(Cache::new(3 * (2 + 2 + 256)));
        let source = |key: &'static str| ask(&cache, key, LONG, move || value(key)).1;
        for key in ["a", "b", "c"] {
            assert_eq!(source(key), Source::Miss);
        }
        assert_eq!(source("a"), Source::Hit);
        // `b` is now the one used longest ago.
        assert_eq!(source("d"), Source::Miss);
        assert_eq!(cache.counts().entries, 3);
        assert_eq!(cache.counts().dropped, 1);
        assert_eq!(source("a"), Source::Hit);
        assert_eq!(source("c"), Source::Hit);
        assert_eq!(source("d"), Source::Hit);
        assert_eq!(source("b"), Source::Miss);
        assert_eq!(cache.counts().bytes, 3 * (2 + 2 + 256));
        // A result larger than the whole cache is returned and not kept.
        let tiny = Arc::new(Cache::new(10));
        for _ in 0..2 {
            let found = ask(&tiny, "a", LONG, || value("a"));
            assert_eq!(found, (Some(Arc::new(value("a"))), Source::Miss));
        }
        assert_eq!(tiny.counts().entries, 0);
    }

    #[test]
    fn a_kept_outcome_can_be_forgotten() {
        let cache = Arc::new(Cache::new(1 << 20));
        let stopped = || Outcome::Timeout { limit: 1 };
        let (first, _) = ask(&cache, "long", LONG, stopped);
        let first = first.unwrap();
        // Another outcome than the one that is kept changes nothing.
        cache.forget("long", &Arc::new(stopped()));
        assert_eq!(ask(&cache, "long", LONG, stopped).1, Source::Hit);
        cache.forget("long", &first);
        assert_eq!(cache.counts().entries, 0);
        assert_eq!(cache.counts().bytes, 0);
        assert_eq!(ask(&cache, "long", LONG, || value("x")).1, Source::Miss);
        assert_eq!(ask(&cache, "long", LONG, stopped).1, Source::Hit);
    }

    #[test]
    fn refusals_and_time_limits_last_and_the_load_of_the_moment_does_not() {
        let cache = Arc::new(Cache::new(1 << 20));
        let refused = Outcome::Refused {
            message: "no".into(),
            seconds: 1.0,
        };
        let failed = Outcome::Failed {
            message: "x".into(),
        };
        for (key, outcome, kept) in [
            ("refused", refused, true),
            ("timeout", Outcome::Timeout { limit: 1 }, true),
            ("busy", Outcome::Busy, false),
            ("failed", failed, false),
        ] {
            for round in 0..2 {
                let expected = outcome.clone();
                let (found, source) = ask(&cache, key, LONG, move || expected);
                assert_eq!(found, Some(Arc::new(outcome.clone())));
                let expected = if kept && round == 1 {
                    Source::Hit
                } else {
                    Source::Miss
                };
                assert_eq!(source, expected, "{key} {round}");
            }
        }
        assert_eq!(cache.counts().entries, 2);
    }

    #[test]
    fn requests_for_a_running_computation_wait_for_it() {
        let cache = Arc::new(Cache::new(1 << 20));
        let runs = Arc::new(AtomicUsize::new(0));
        let mut threads = Vec::new();
        for _ in 0..8 {
            let (cache, runs) = (cache.clone(), runs.clone());
            threads.push(std::thread::spawn(move || {
                ask(&cache, "slow", LONG, move || {
                    runs.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(300));
                    value("slow")
                })
            }));
        }
        let results: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert_eq!(runs.load(Ordering::SeqCst), 1);
        assert!(results
            .iter()
            .all(|(outcome, _)| *outcome == Some(Arc::new(value("slow")))));
        let count = |wanted: Source| {
            results
                .iter()
                .filter(|(_, source)| *source == wanted)
                .count()
        };
        assert_eq!((count(Source::Miss), count(Source::Wait)), (1, 7));
        assert_eq!(ask(&cache, "slow", LONG, || value("again")).1, Source::Hit);
    }

    #[test]
    fn a_computation_goes_on_when_nobody_waits_for_it_any_longer() {
        let cache = Arc::new(Cache::new(1 << 20));
        let runs = Arc::new(AtomicUsize::new(0));
        let request = |patience: Duration| {
            let runs = runs.clone();
            ask(&cache, "long", patience, move || {
                runs.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(500));
                value("long")
            })
        };
        let short = Duration::from_millis(50);
        let started = Instant::now();
        // The request that starts it gives up, and so does the next: the
        // computation is under way all the same, once.
        assert_eq!(request(short), (None, Source::Miss));
        assert_eq!(request(short), (None, Source::Wait));
        assert!(started.elapsed() < Duration::from_millis(400));
        assert_eq!(cache.counts().entries, 0);
        // Its outcome is there for whoever waits long enough, and is kept.
        let waited = request(LONG);
        assert_eq!(waited, (Some(Arc::new(value("long"))), Source::Wait));
        assert!(started.elapsed() >= Duration::from_millis(500));
        let kept = request(short);
        assert_eq!(kept, (Some(Arc::new(value("long"))), Source::Hit));
        assert_eq!(runs.load(Ordering::SeqCst), 1);
        let counts = cache.counts();
        assert_eq!(
            (counts.entries, counts.hits, counts.misses, counts.waits),
            (1, 1, 1, 2)
        );
    }

    #[test]
    fn a_ticket_that_is_dropped_releases_the_requests_that_wait() {
        let cache = Arc::new(Cache::new(1 << 20));
        let (outcome, source) = ask(&cache, "bad", LONG, || panic!("computation failed"));
        assert_eq!(source, Source::Miss);
        assert!(
            matches!(outcome.as_deref(), Some(Outcome::Failed { message }) if message.contains("interrupted"))
        );
        // Nothing is left behind: the next request computes.
        assert_eq!(cache.counts().entries, 0);
        assert_eq!(ask(&cache, "bad", LONG, || value("x")).1, Source::Miss);
    }
}
