//! Running `tet`: a bounded number of computations at a time, each a child
//! process with a deadline, which ends with this server at the latest.

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::api::{Job, Outcome};
use crate::settings::Shared;

/// The most that is kept of what a child writes: about eight million
/// digits for each of the two numbers of a result.
const OUTPUT_LIMIT: usize = 16 << 20;

/// The computations that run. There is no queue: a computation gets a slot
/// at once, or none.
#[derive(Default)]
pub struct Slots {
    taken: AtomicUsize,
}

/// A slot that is given back when it is dropped.
pub struct Slot<'a>(&'a Slots);

impl Slots {
    /// How many computations run.
    pub fn taken(&self) -> usize {
        self.taken.load(Ordering::SeqCst)
    }

    /// A slot, if fewer than `jobs` are taken.
    pub fn take(&self, jobs: usize) -> Option<Slot<'_>> {
        let free = |taken: usize| (taken < jobs).then_some(taken + 1);
        self.taken
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, free)
            .ok()
            .map(|_| Slot(self))
    }
}

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.0.taken.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(unix)]
mod limits {
    //! What the kernel enforces on a child, whatever becomes of this
    //! server: its address space, a lower priority than the server that
    //! has to keep answering, and its end when the server ends.

    use std::io;
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

    /// `PRIO_PROCESS`, and 0 for the calling process.
    const PROCESS: (c_int, c_int) = (0, 0);
    /// `SIGKILL`.
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    const KILL: c_int = 9;

    /// Asks the kernel to end the calling process, a child that has not
    /// run `tet` yet, as soon as its parent ends: a computation may take
    /// hours, and nobody would stop it or read its result after the server.
    #[cfg(target_os = "freebsd")]
    fn end_with_parent() -> io::Result<()> {
        extern "C" {
            fn procctl(kind: c_int, id: i64, command: c_int, data: *mut c_int) -> c_int;
        }
        /// `P_PID` with the id 0 is the calling process.
        const BY_PID: c_int = 0;
        /// `PROC_PDEATHSIG_CTL`.
        const SIGNAL_AT_END_OF_PARENT: c_int = 11;
        let mut signal = KILL;
        // SAFETY: `procctl` reads the one `int` that `data` points to.
        match unsafe { procctl(BY_PID, 0, SIGNAL_AT_END_OF_PARENT, &mut signal) } {
            0 => Ok(()),
            _ => Err(io::Error::last_os_error()),
        }
    }

    #[cfg(target_os = "linux")]
    fn end_with_parent() -> io::Result<()> {
        extern "C" {
            fn prctl(option: c_int, ...) -> c_int;
        }
        /// `PR_SET_PDEATHSIG`. The signal is sent when the thread that
        /// started the child ends: here that thread waits for the child.
        const SIGNAL_AT_END_OF_PARENT: c_int = 1;
        // SAFETY: this option takes one more argument, an `unsigned long`.
        match unsafe { prctl(SIGNAL_AT_END_OF_PARENT, KILL as std::os::raw::c_ulong) } {
            0 => Ok(()),
            _ => Err(io::Error::last_os_error()),
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "freebsd")))]
    fn end_with_parent() -> io::Result<()> {
        Ok(())
    }
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

    pub fn apply(command: &mut Command, memory: Option<u64>, nice: Option<i32>) {
        extern "C" {
            fn getppid() -> c_int;
        }
        let server = std::process::id();
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
                Err(io::Error::last_os_error())
            }
        };
        // SAFETY: the closure runs in the child between fork and exec and
        // only makes system calls, which allocate nothing and take no lock.
        unsafe {
            command.pre_exec(move || {
                end_with_parent()?;
                // A server that ended before the kernel was asked is not
                // noticed any more: `tet` is not started then (`ESRCH`).
                if u32::try_from(getppid()) != Ok(server) {
                    return Err(io::Error::from_raw_os_error(3));
                }
                if let Some(nice) = nice {
                    // Only a higher number is allowed: a child that is nicer
                    // already stays as it is.
                    let _ = setpriority(PROCESS.0, PROCESS.1, nice);
                }
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

    pub fn apply(_: &mut std::process::Command, _: Option<u64>, _: Option<i32>) {}
}

pub use limits::memory_supported;

/// Everything a pipe delivers, of which the first sixteen megabytes are
/// kept, and whether there was more.
fn collected(pipe: Option<impl Read + Send + 'static>) -> std::thread::JoinHandle<(Vec<u8>, bool)> {
    std::thread::spawn(move || {
        let mut kept = Vec::new();
        let mut more = false;
        if let Some(pipe) = pipe {
            let mut pipe = pipe.take(OUTPUT_LIMIT as u64);
            let _ = pipe.read_to_end(&mut kept);
            // The rest, if any, is read and dropped so that the child never
            // blocks on a full pipe.
            let rest = std::io::copy(&mut pipe.into_inner(), &mut std::io::sink());
            more = rest.is_ok_and(|bytes| bytes > 0);
        }
        (kept, more)
    })
}

/// Waits for the child that was started at `started` while it has run for
/// less than `limit`, which is asked for again and again: it can change.
/// `None` when the child still runs at its limit, which is returned too.
fn exited(
    child: &mut Child,
    started: Instant,
    limit: impl Fn() -> Duration,
) -> std::io::Result<(Option<std::process::ExitStatus>, Duration)> {
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok((Some(status), limit()));
        }
        let (age, limit) = (started.elapsed(), limit());
        let Some(left) = limit.checked_sub(age).filter(|left| !left.is_zero()) else {
            return Ok((None, limit));
        };
        // The longer a child runs, the more rarely it is looked at: its end
        // is seen within a sixty-fourth of its time, a quarter of a second
        // at most.
        let pause = (age / 64).clamp(Duration::from_micros(500), Duration::from_millis(250));
        std::thread::sleep(pause.min(left));
    }
}

/// How `tet` is run.
pub struct Runner {
    /// The program: a path, or a name for the search path.
    pub tet: String,
}

impl Runner {
    /// Runs `tet` for the job and reads what it printed. The settings are
    /// those of the moment: the threads, the memory and the niceness when
    /// the child starts, the time limit for as long as it runs.
    pub fn run(&self, job: &Job, settings: &Shared) -> Outcome {
        let failed = |message: String| Outcome::Failed { message };
        let now = settings.now();
        let mut command = Command::new(&self.tet);
        command
            .args(job.argv())
            .env("SILENT", "1")
            .env_remove("TET_MT")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(threads) = now.threads {
            command.env("TET_MT", threads.to_string());
        }
        limits::apply(&mut command, now.memory, now.nice);
        let started = Instant::now();
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => return failed(format!("tet could not be started: {error}")),
        };
        let (stdout, stderr) = (
            collected(child.stdout.take()),
            collected(child.stderr.take()),
        );
        let status = match exited(&mut child, started, || settings.now().timeout) {
            Ok((Some(status), _)) => status,
            Ok((None, limit)) => {
                let _ = child.kill();
                let _ = child.wait();
                return Outcome::Timeout {
                    limit: limit.as_secs(),
                };
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return failed(format!("tet could not be waited for: {error}"));
            }
        };
        let seconds = started.elapsed().as_secs_f64();
        let (stdout, more) = stdout.join().unwrap_or_default();
        let (stderr, _) = stderr.join().unwrap_or_default();
        if more {
            return failed(format!(
                "the result is longer than the {} MiB that this server returns",
                OUTPUT_LIMIT >> 20
            ));
        }
        let (stdout, stderr) = (
            String::from_utf8_lossy(&stdout),
            String::from_utf8_lossy(&stderr),
        );
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
                "tet was stopped by the system ({status}): it needed more memory than this \
                 server allows, or it was ended by hand"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_are_taken_at_once_or_not_at_all() {
        let slots = Slots::default();
        let first = slots.take(2).unwrap();
        let second = slots.take(2).unwrap();
        assert_eq!(slots.taken(), 2);
        // None is free, and nothing waits for one.
        let started = Instant::now();
        assert!(slots.take(2).is_none());
        assert!(started.elapsed() < Duration::from_millis(100));
        assert_eq!(slots.taken(), 2);
        // The number of slots can change while some are taken.
        let third = slots.take(3).unwrap();
        assert!(slots.take(3).is_none() && slots.take(1).is_none());
        drop(first);
        assert!(slots.take(2).is_none());
        drop((second, third));
        assert_eq!(slots.taken(), 0);
        assert!(slots.take(1).is_some());
        assert!(slots.take(0).is_none());
    }
}
