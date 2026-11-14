//! Running `tet`: a bounded number of computations at a time, each a child
//! process with a deadline.

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::api::{Job, Outcome};

/// How `tet` is run.
#[derive(Clone, Debug)]
pub struct Runner {
    /// The program: a path, or a name for the search path.
    pub tet: String,
    /// The time one computation may take.
    pub timeout: Duration,
    /// `TET_MT` for the child: its number of threads; `None` is serial.
    pub threads: Option<u64>,
    /// The address space one child may use, in bytes.
    pub memory: Option<u64>,
    /// The niceness of a child; `None` leaves it that of the server.
    pub nice: Option<i32>,
}

/// The most that is kept of what a child writes.
const OUTPUT_LIMIT: u64 = 1 << 20;

/// The slots for computations and the requests that wait for one.
pub struct Slots {
    jobs: usize,
    queue: usize,
    state: Mutex<(usize, usize)>,
    freed: Condvar,
}

/// A slot that is given back when it is dropped.
pub struct Slot<'a>(&'a Slots);

impl Slots {
    /// `jobs` computations at a time, and at most `queue` requests waiting.
    pub fn new(jobs: usize, queue: usize) -> Slots {
        Slots {
            jobs,
            queue,
            state: Mutex::new((0, 0)),
            freed: Condvar::new(),
        }
    }

    /// The numbers of running computations and of waiting requests.
    pub fn load(&self) -> (usize, usize) {
        *self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// A slot, as soon as one is free; `None` when too many requests wait
    /// already or none is free by `deadline`.
    pub fn take(&self, deadline: Instant) -> Option<Slot<'_>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.0 >= self.jobs {
            if state.1 >= self.queue {
                return None;
            }
            state.1 += 1;
            while state.0 >= self.jobs {
                let left = deadline
                    .checked_duration_since(Instant::now())
                    .filter(|left| !left.is_zero());
                let Some(left) = left else {
                    state.1 -= 1;
                    return None;
                };
                state = self
                    .freed
                    .wait_timeout(state, left)
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .0;
            }
            state.1 -= 1;
        }
        state.0 += 1;
        Some(Slot(self))
    }
}

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        let mut state = self
            .0
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.0 -= 1;
        drop(state);
        self.0.freed.notify_one();
    }
}

#[cfg(unix)]
mod limits {
    //! Limits that the kernel enforces on a child, whatever becomes of
    //! this server: processor time, so that a computation cannot outlive
    //! its deadline for long, address space, and a lower priority than the
    //! server that has to keep answering.

    use std::os::raw::c_int;
    use std::os::unix::process::CommandExt;
    use std::process::Command;

    #[repr(C)]
    struct Limit {
        soft: u64,
        hard: u64,
    }

    extern "C" {
        fn setrlimit(resource: c_int, limit: *const Limit) -> c_int;
        fn setpriority(which: c_int, who: c_int, priority: c_int) -> c_int;
    }

    const CPU: c_int = 0;
    /// `PRIO_PROCESS`, and 0 for the calling process.
    const PROCESS: (c_int, c_int) = (0, 0);
    #[cfg(target_os = "linux")]
    const ADDRESS_SPACE: Option<c_int> = Some(9);
    #[cfg(target_os = "freebsd")]
    const ADDRESS_SPACE: Option<c_int> = Some(10);
    #[cfg(not(any(target_os = "linux", target_os = "freebsd")))]
    const ADDRESS_SPACE: Option<c_int> = None;

    /// Whether the address space of a child can be limited here.
    pub fn memory_supported() -> bool {
        ADDRESS_SPACE.is_some()
    }

    pub fn apply(command: &mut Command, cpu_seconds: u64, memory: Option<u64>, nice: Option<i32>) {
        let set = move |resource: c_int, value: u64| {
            let limit = Limit {
                soft: value,
                hard: value,
            };
            // SAFETY: `setrlimit` reads the two words of the struct, whose
            // layout is that of `struct rlimit` on 64-bit Linux and FreeBSD.
            if unsafe { setrlimit(resource, &limit) } == 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error())
            }
        };
        // SAFETY: the closure runs in the child between fork and exec and
        // only makes the system calls `setpriority` and `setrlimit`, which
        // allocate nothing and take no lock.
        unsafe {
            command.pre_exec(move || {
                if let Some(nice) = nice {
                    // Only a higher number is allowed: a child that is nicer
                    // already stays as it is.
                    let _ = setpriority(PROCESS.0, PROCESS.1, nice);
                }
                set(CPU, cpu_seconds)?;
                match (ADDRESS_SPACE, memory) {
                    (Some(resource), Some(bytes)) => set(resource, bytes),
                    _ => Ok(()),
                }
            });
        }
    }
}

#[cfg(not(unix))]
mod limits {
    pub fn memory_supported() -> bool {
        false
    }

    pub fn apply(_: &mut std::process::Command, _: u64, _: Option<u64>, _: Option<i32>) {}
}

pub use limits::memory_supported;

/// Everything a pipe delivers, of which the first megabyte is kept.
fn collected(pipe: Option<impl Read + Send + 'static>) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut kept = Vec::new();
        if let Some(pipe) = pipe {
            let mut pipe = pipe.take(OUTPUT_LIMIT);
            let _ = pipe.read_to_end(&mut kept);
            // The rest, if any, is read and dropped so that the child never
            // blocks on a full pipe.
            let _ = std::io::copy(&mut pipe.into_inner(), &mut std::io::sink());
        }
        kept
    })
}

/// Waits for the child until `deadline`; `None` when it is still running.
fn exited(
    child: &mut Child,
    deadline: Instant,
) -> std::io::Result<Option<std::process::ExitStatus>> {
    let mut pause = Duration::from_micros(500);
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        let Some(left) = deadline
            .checked_duration_since(Instant::now())
            .filter(|left| !left.is_zero())
        else {
            return Ok(None);
        };
        std::thread::sleep(pause.min(left));
        pause = (pause * 2).min(Duration::from_millis(20));
    }
}

impl Runner {
    /// Runs `tet` for the job and reads what it printed.
    pub fn run(&self, job: &Job) -> Outcome {
        let failed = |message: String| Outcome::Failed { message };
        let mut command = Command::new(&self.tet);
        command
            .args(job.argv())
            .env("SILENT", "1")
            .env_remove("TET_MT")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let threads = self.threads.unwrap_or(1).max(1);
        if let Some(threads) = self.threads {
            command.env("TET_MT", threads.to_string());
        }
        // Processor time runs `threads` times faster than the clock at most.
        let cpu = (self.timeout.as_secs() + 5).saturating_mul(threads);
        limits::apply(&mut command, cpu, self.memory, self.nice);
        let started = Instant::now();
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => return failed(format!("tet could not be started: {error}")),
        };
        let (stdout, stderr) = (
            collected(child.stdout.take()),
            collected(child.stderr.take()),
        );
        let status = match exited(&mut child, started + self.timeout) {
            Ok(Some(status)) => status,
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return Outcome::Timeout {
                    limit: self.timeout.as_secs(),
                };
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return failed(format!("tet could not be waited for: {error}"));
            }
        };
        let seconds = started.elapsed().as_secs_f64();
        let text = |bytes: std::thread::Result<Vec<u8>>| {
            String::from_utf8_lossy(&bytes.unwrap_or_default()).into_owned()
        };
        let (stdout, stderr) = (text(stdout.join()), text(stderr.join()));
        match status.code() {
            Some(0) => {
                let mut lines = stdout.lines();
                match (lines.next(), lines.next(), lines.next()) {
                    (Some(re), Some(im), None) if !re.is_empty() && !im.is_empty() => {
                        Outcome::Value {
                            re: re.to_string(),
                            im: im.to_string(),
                            seconds,
                        }
                    }
                    _ => failed("tet did not print the two lines of a result".into()),
                }
            }
            Some(1) => {
                let message = stderr
                    .lines()
                    .rev()
                    .find_map(|line| line.strip_prefix("error: "))
                    .unwrap_or("tet failed without a message")
                    .to_string();
                Outcome::Refused { message, seconds }
            }
            Some(code) => failed(format!("tet ended with the unexpected exit code {code}")),
            None => failed(format!(
                "tet was stopped by the system ({status}): the computation needs more memory or \
                 processor time than this server allows"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn slots_are_bounded_and_waiting_is_bounded() {
        let slots = Arc::new(Slots::new(2, 1));
        let far = Instant::now() + Duration::from_secs(30);
        let first = slots.take(far).unwrap();
        let second = slots.take(far).unwrap();
        assert_eq!(slots.load(), (2, 0));
        // No slot is free: one request may wait, until its deadline.
        let started = Instant::now();
        assert!(slots.take(started + Duration::from_millis(100)).is_none());
        assert!(started.elapsed() >= Duration::from_millis(100));
        assert_eq!(slots.load(), (2, 0));
        let waiter = {
            let slots = slots.clone();
            std::thread::spawn(move || slots.take(far).is_some())
        };
        while slots.load().1 == 0 {
            std::thread::sleep(Duration::from_millis(5));
        }
        // The queue is full: the next request is turned away at once.
        let started = Instant::now();
        assert!(slots.take(far).is_none());
        assert!(started.elapsed() < Duration::from_secs(5));
        drop(first);
        assert!(waiter.join().unwrap());
        assert_eq!(slots.load(), (1, 0));
        drop(second);
        assert_eq!(slots.load(), (0, 0));
        assert!(slots.take(Instant::now()).is_some());
    }
}
