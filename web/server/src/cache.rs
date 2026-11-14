//! The results computed since the server started, in memory only: nothing
//! is written anywhere, and a restart begins with an empty cache.
//!
//! A request whose computation is already running waits for it instead of
//! starting a second one. The cache is bounded in bytes, because anybody
//! can ask for any number of different things; beyond the bound the results
//! that were used longest ago are dropped.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Instant;

use crate::api::Outcome;

/// How a request got its outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// It was in the cache.
    Hit,
    /// This request computed it.
    Miss,
    /// Another request was computing it; this one waited.
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
struct Running {
    outcome: Mutex<Option<Arc<Outcome>>>,
    done: Condvar,
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

/// Removes the mark of a running computation if its owner unwinds, so that
/// the requests waiting for it are not left waiting.
struct Owner<'a> {
    cache: &'a Cache,
    key: &'a str,
    running: Arc<Running>,
    finished: bool,
}

impl Drop for Owner<'_> {
    fn drop(&mut self) {
        if !self.finished {
            let failed = Outcome::Failed {
                message: "the computation was interrupted".into(),
            };
            self.cache.finish(self.key, &self.running, Arc::new(failed));
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

    /// The outcome for `key`: the kept one, the one of the computation that
    /// is running, or the one of `compute`, which then runs exactly once
    /// for all the requests that ask in the meantime. A request waits for a
    /// running computation until `deadline` and is `Busy` after it.
    pub fn outcome(
        &self,
        key: &str,
        deadline: Instant,
        compute: impl FnOnce() -> Outcome,
    ) -> (Arc<Outcome>, Source) {
        enum Found {
            Kept(Arc<Outcome>),
            Running(Arc<Running>),
            New(Arc<Running>),
        }
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
                        outcome: Mutex::new(None),
                        done: Condvar::new(),
                    });
                    state
                        .entries
                        .insert(key.to_string(), Entry::Running(running.clone()));
                    Found::New(running)
                }
            }
        };
        match found {
            Found::Kept(outcome) => {
                self.count(|counts| counts.hits += 1);
                (outcome, Source::Hit)
            }
            Found::Running(running) => {
                self.count(|counts| counts.waits += 1);
                (Self::awaited(&running, deadline), Source::Wait)
            }
            Found::New(running) => {
                self.count(|counts| counts.misses += 1);
                let mut owner = Owner {
                    cache: self,
                    key,
                    running,
                    finished: false,
                };
                let outcome = Arc::new(compute());
                self.finish(key, &owner.running, outcome.clone());
                owner.finished = true;
                (outcome, Source::Miss)
            }
        }
    }

    fn awaited(running: &Running, deadline: Instant) -> Arc<Outcome> {
        let mut outcome = running
            .outcome
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        loop {
            if let Some(outcome) = outcome.as_ref() {
                return outcome.clone();
            }
            let Some(left) = deadline
                .checked_duration_since(Instant::now())
                .filter(|left| !left.is_zero())
            else {
                return Arc::new(Outcome::Busy);
            };
            outcome = running
                .done
                .wait_timeout(outcome, left)
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .0;
        }
    }

    /// Hands the outcome to the waiting requests and keeps it if it lasts.
    fn finish(&self, key: &str, running: &Arc<Running>, outcome: Arc<Outcome>) {
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
        *running
            .outcome
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(outcome);
        running.done.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    fn value(text: &str) -> Outcome {
        Outcome::Value {
            re: text.to_string(),
            im: "0".to_string(),
            seconds: 0.5,
        }
    }

    fn soon() -> Instant {
        Instant::now() + Duration::from_secs(30)
    }

    #[test]
    fn a_result_is_computed_once_and_then_kept() {
        let cache = Cache::new(1 << 20);
        let runs = AtomicUsize::new(0);
        let ask = |key: &str| {
            cache.outcome(key, soon(), || {
                runs.fetch_add(1, Ordering::SeqCst);
                value(key)
            })
        };
        assert_eq!(ask("a"), (Arc::new(value("a")), Source::Miss));
        assert_eq!(ask("a"), (Arc::new(value("a")), Source::Hit));
        assert_eq!(ask("b").1, Source::Miss);
        assert_eq!(ask("a").1, Source::Hit);
        assert_eq!(runs.load(Ordering::SeqCst), 2);
        let counts = cache.counts();
        assert_eq!(
            (counts.entries, counts.hits, counts.misses, counts.waits),
            (2, 2, 2, 0)
        );
        assert_eq!(counts.bytes, 2 * (2 + 2 + 256));
    }

    #[test]
    fn refusals_last_and_the_load_of_the_moment_does_not() {
        let cache = Cache::new(1 << 20);
        let refused = Outcome::Refused {
            message: "no".into(),
            seconds: 1.0,
        };
        for (key, outcome, kept) in [
            ("refused", refused, true),
            ("timeout", Outcome::Timeout { limit: 1 }, false),
            ("busy", Outcome::Busy, false),
            (
                "failed",
                Outcome::Failed {
                    message: "x".into(),
                },
                false,
            ),
        ] {
            for round in 0..2 {
                let (found, source) = cache.outcome(key, soon(), || outcome.clone());
                assert_eq!(*found, outcome);
                let expected = if kept && round == 1 {
                    Source::Hit
                } else {
                    Source::Miss
                };
                assert_eq!(source, expected, "{key} {round}");
            }
        }
        assert_eq!(cache.counts().entries, 1);
    }

    #[test]
    fn the_results_used_longest_ago_are_dropped_beyond_the_limit() {
        // Room for three entries of this size.
        let cache = Cache::new(3 * (2 + 2 + 256));
        let ask = |key: &str| cache.outcome(key, soon(), || value(key)).1;
        for key in ["a", "b", "c"] {
            assert_eq!(ask(key), Source::Miss);
        }
        assert_eq!(ask("a"), Source::Hit);
        // `b` is now the one used longest ago.
        assert_eq!(ask("d"), Source::Miss);
        assert_eq!(cache.counts().entries, 3);
        assert_eq!(cache.counts().dropped, 1);
        assert_eq!(ask("a"), Source::Hit);
        assert_eq!(ask("c"), Source::Hit);
        assert_eq!(ask("d"), Source::Hit);
        assert_eq!(ask("b"), Source::Miss);
        assert_eq!(cache.counts().bytes, 3 * (2 + 2 + 256));
        // A result larger than the whole cache is returned and not kept.
        let tiny = Cache::new(10);
        assert_eq!(tiny.outcome("a", soon(), || value("a")).1, Source::Miss);
        assert_eq!(tiny.outcome("a", soon(), || value("a")).1, Source::Miss);
        assert_eq!(tiny.counts().entries, 0);
    }

    #[test]
    fn requests_for_a_running_computation_wait_for_it() {
        let cache = Arc::new(Cache::new(1 << 20));
        let runs = Arc::new(AtomicUsize::new(0));
        let mut threads = Vec::new();
        for _ in 0..8 {
            let (cache, runs) = (cache.clone(), runs.clone());
            threads.push(std::thread::spawn(move || {
                cache.outcome("slow", soon(), || {
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
            .all(|(outcome, _)| **outcome == value("slow")));
        let misses = results
            .iter()
            .filter(|(_, source)| *source == Source::Miss)
            .count();
        let waits = results
            .iter()
            .filter(|(_, source)| *source == Source::Wait)
            .count();
        assert_eq!((misses, waits), (1, 7));
        assert_eq!(
            cache.outcome("slow", soon(), || value("again")).1,
            Source::Hit
        );
    }

    #[test]
    fn waiting_ends_at_the_deadline_and_with_an_outcome_that_does_not_last() {
        let cache = Arc::new(Cache::new(1 << 20));
        let owner = {
            let cache = cache.clone();
            std::thread::spawn(move || {
                cache.outcome("slow", soon(), || {
                    std::thread::sleep(Duration::from_millis(600));
                    Outcome::Timeout { limit: 1 }
                })
            })
        };
        std::thread::sleep(Duration::from_millis(100));
        // Too impatient: busy, and the computation goes on.
        let impatient = cache.outcome("slow", Instant::now() + Duration::from_millis(50), || {
            value("x")
        });
        assert_eq!(impatient, (Arc::new(Outcome::Busy), Source::Wait));
        // Patient: the outcome of the owner, although it is not kept.
        let patient = cache.outcome("slow", soon(), || value("x"));
        assert_eq!(
            patient,
            (Arc::new(Outcome::Timeout { limit: 1 }), Source::Wait)
        );
        assert_eq!(owner.join().unwrap().1, Source::Miss);
        assert_eq!(cache.counts().entries, 0);
        // Nothing is left behind: the next request computes.
        assert_eq!(cache.outcome("slow", soon(), || value("x")).1, Source::Miss);
    }

    #[test]
    fn a_computation_that_panics_releases_the_requests_that_wait() {
        let cache = Arc::new(Cache::new(1 << 20));
        let owner = {
            let cache = cache.clone();
            std::thread::spawn(move || {
                cache.outcome("bad", soon(), || {
                    std::thread::sleep(Duration::from_millis(300));
                    panic!("computation failed");
                })
            })
        };
        std::thread::sleep(Duration::from_millis(100));
        let (outcome, source) = cache.outcome("bad", soon(), || value("x"));
        assert_eq!(source, Source::Wait);
        assert!(
            matches!(&*outcome, Outcome::Failed { message } if message.contains("interrupted"))
        );
        assert!(owner.join().is_err());
        assert_eq!(cache.outcome("bad", soon(), || value("x")).1, Source::Miss);
    }
}
