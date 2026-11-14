//! The settings that can change while the server runs. They come from the
//! environment and, above it, from a file that the server reads again
//! whenever it changes: a new time limit needs no restart.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crate::api::span;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Settings {
    /// How long the requests for a computation wait, counted from the
    /// first of them.
    pub wait: Duration,
    /// How long a computation may take.
    pub timeout: Duration,
    /// Computations at a time.
    pub jobs: usize,
    /// `TET_MT` of a child, its number of threads; `None` is serial.
    pub threads: Option<u64>,
    /// The address space that a child may use, in bytes.
    pub memory: Option<u64>,
    /// The niceness of a child; `None` leaves it that of the server.
    pub nice: Option<i32>,
}

/// The names of these settings, in the environment and in the file.
pub const NAMES: [&str; 6] = [
    "TET_WEB_WAIT",
    "TET_WEB_TIMEOUT",
    "TET_WEB_JOBS",
    "TET_WEB_THREADS",
    "TET_WEB_MEMORY_MB",
    "TET_WEB_NICE",
];

/// How often the file is looked at.
const PERIOD: Duration = Duration::from_secs(2);

/// As many computations at a time as there are processors, four at least.
fn default_jobs() -> usize {
    std::thread::available_parallelism()
        .map_or(1, |count| count.get())
        .max(4)
}

/// A time in seconds: `90`, `90s`, `15m`, `3h`, `2d`.
fn seconds(name: &str, text: &str) -> Result<u64, String> {
    let digits = text.trim_end_matches(|c: char| !c.is_ascii_digit());
    let factor = match text[digits.len()..].trim() {
        "" | "s" => Some(1),
        "m" => Some(60),
        "h" => Some(3_600),
        "d" => Some(86_400),
        _ => None,
    };
    let whole = digits.bytes().all(|byte| byte.is_ascii_digit());
    factor
        .filter(|_| whole)
        .and_then(|factor| digits.parse::<u64>().ok()?.checked_mul(factor))
        .filter(|seconds| *seconds >= 1)
        .ok_or_else(|| {
            format!(
                "{name}={text:?} is not a time: a whole number of seconds, at least 1, or of \
                 minutes, hours or days as in 90m, 3h, 2d"
            )
        })
}

fn whole(name: &str, text: &str) -> Result<u64, String> {
    text.parse()
        .ok()
        .filter(|_| text.bytes().all(|byte| byte.is_ascii_digit()))
        .ok_or_else(|| format!("{name}={text:?} is not a whole number"))
}

impl Settings {
    /// The settings that `get` has values for; a name without a value, or
    /// with an empty one, has its default.
    pub fn from(get: impl Fn(&str) -> Option<String>) -> Result<Settings, String> {
        let text = |name: &str| {
            get(name)
                .map(|text| text.trim().to_string())
                .filter(|text| !text.is_empty())
        };
        let time = |name: &str, default: u64| match text(name) {
            Some(text) => seconds(name, &text),
            None => Ok(default),
        };
        let number = |name: &str| text(name).map(|text| whole(name, &text)).transpose();
        let jobs = match number("TET_WEB_JOBS")? {
            None => default_jobs(),
            Some(0) => return Err("TET_WEB_JOBS must be at least 1".into()),
            Some(jobs) => usize::try_from(jobs).map_err(|_| "TET_WEB_JOBS is too large")?,
        };
        let memory = number("TET_WEB_MEMORY_MB")?
            .filter(|megabytes| *megabytes > 0)
            .map(|megabytes| megabytes.checked_mul(1 << 20))
            .map(|bytes| bytes.ok_or("TET_WEB_MEMORY_MB is too large"))
            .transpose()?;
        if memory.is_some() && !crate::run::memory_supported() {
            return Err("TET_WEB_MEMORY_MB is not supported on this system".into());
        }
        let nice = number("TET_WEB_NICE")?
            .filter(|nice| *nice > 0)
            .map(|nice| i32::try_from(nice).ok().filter(|nice| *nice <= 20))
            .map(|nice| nice.ok_or("TET_WEB_NICE is at most 20"))
            .transpose()?;
        Ok(Settings {
            wait: Duration::from_secs(time("TET_WEB_WAIT", 120)?),
            timeout: Duration::from_secs(time("TET_WEB_TIMEOUT", 10_800)?),
            jobs,
            threads: number("TET_WEB_THREADS")?.filter(|threads| *threads > 0),
            memory,
            nice,
        })
    }

    /// The settings in a line, for the log.
    pub fn described(&self) -> String {
        let or = |value: Option<String>, none: &str| value.unwrap_or_else(|| none.to_string());
        format!(
            "a request waits {}, a calculation may take {}, {} at a time, threads of one tet: \
             {}, its memory: {}, its niceness: {}",
            span(self.wait.as_secs()),
            span(self.timeout.as_secs()),
            self.jobs,
            or(self.threads.map(|threads| threads.to_string()), "one"),
            or(
                self.memory.map(|bytes| format!("{} MiB", bytes >> 20)),
                "unlimited"
            ),
            or(self.nice.map(|nice| nice.to_string()), "that of the server"),
        )
    }
}

/// The settings of a file: lines `NAME=value`, where the value may stand in
/// double quotes; a `#` begins a comment.
pub fn parsed(text: &str) -> Result<Vec<(String, String)>, String> {
    let mut found: Vec<(String, String)> = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let wrong = |reason: String| format!("line {}: {reason}", index + 1);
        let (name, value) = line
            .split_once('=')
            .ok_or_else(|| wrong(format!("{line:?} is not NAME=value")))?;
        let (name, value) = (name.trim(), value.trim().trim_matches('"').trim());
        if !NAMES.contains(&name) {
            return Err(wrong(format!(
                "{name} is not a setting of this file, which has {}",
                NAMES.join(", ")
            )));
        }
        if found.iter().any(|(seen, _)| seen == name) {
            return Err(wrong(format!("{name} is set twice")));
        }
        found.push((name.to_string(), value.to_string()));
    }
    Ok(found)
}

/// The settings in use, and where they come from.
pub struct Shared {
    current: RwLock<Settings>,
    /// The values of the environment. The file overrides them name by name.
    environment: Vec<(String, String)>,
    file: Option<PathBuf>,
    /// The text of the file when the server started.
    loaded: Option<String>,
}

/// What the watcher knows of the file.
#[derive(Default)]
struct Seen {
    /// Its text when it was read last; `None` when it was not there.
    text: Option<String>,
    /// It was there, and at the last look it was not.
    gone: bool,
    /// Why it could not be read at the last look.
    unread: Option<String>,
}

impl Shared {
    /// The settings at the start of the server.
    pub fn load(
        environment: Vec<(String, String)>,
        file: Option<PathBuf>,
    ) -> Result<Shared, String> {
        let loaded = Self::text(file.as_deref())?;
        let settings = Self::settings(&environment, file.as_deref(), loaded.as_deref())?;
        Ok(Shared {
            current: RwLock::new(settings),
            environment,
            file,
            loaded,
        })
    }

    /// The text of the file; `None` when there is no file.
    fn text(file: Option<&Path>) -> Result<Option<String>, String> {
        let Some(path) = file else {
            return Ok(None);
        };
        match std::fs::read_to_string(path) {
            Ok(text) => Ok(Some(text)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(format!("{}: {error}", path.display())),
        }
    }

    /// The settings that the text of the file gives, over those of the
    /// environment. A file that is not there sets nothing.
    fn settings(
        environment: &[(String, String)],
        file: Option<&Path>,
        text: Option<&str>,
    ) -> Result<Settings, String> {
        let named = |reason: String| match file {
            Some(path) => format!("{}: {reason}", path.display()),
            None => reason,
        };
        let written = parsed(text.unwrap_or("")).map_err(named)?;
        let found = Settings::from(|name| {
            let value = written
                .iter()
                .chain(environment)
                .find(|(set, _)| set == name);
            value.map(|(_, value)| value.clone())
        });
        // The reason begins with the name of the setting: it says the file
        // too when the value stands there.
        found.map_err(|reason| {
            match written
                .iter()
                .any(|(name, _)| reason.starts_with(name.as_str()))
            {
                true => named(reason),
                false => reason,
            }
        })
    }

    pub fn now(&self) -> Settings {
        self.current
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Puts the settings of a text of the file in use. `None` when they
    /// are the ones in use already; an error leaves those as they are.
    fn put(&self, text: Option<&str>) -> Result<Option<Settings>, String> {
        let new = Self::settings(&self.environment, self.file.as_deref(), text)?;
        let mut current = self
            .current
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if *current == new {
            return Ok(None);
        }
        *current = new.clone();
        Ok(Some(new))
    }

    /// Looks at the file once and puts its settings in use if its text has
    /// changed. `Some` when there is something to say: the new settings, or
    /// why a file that changed has changed nothing.
    fn look(&self, seen: &mut Seen) -> Option<Result<Settings, String>> {
        let text = match Self::text(self.file.as_deref()) {
            Ok(text) => text,
            Err(reason) => {
                // Said once, not at every look.
                let news = seen.unread.as_ref() != Some(&reason);
                seen.unread = Some(reason.clone());
                return news.then_some(Err(reason));
            }
        };
        seen.unread = None;
        if text == seen.text {
            seen.gone = false;
            return None;
        }
        // A file that is being replaced is away for a moment: only one
        // that is still away at the next look sets nothing any more.
        if text.is_none() && !seen.gone {
            seen.gone = true;
            return None;
        }
        seen.gone = false;
        seen.text = text;
        self.put(seen.text.as_deref()).transpose()
    }

    /// Looks at the file again and again, in a thread of its own, and
    /// tells `report` the new settings, or why a changed file has changed
    /// nothing.
    pub fn watch(self: &Arc<Shared>, report: impl Fn(Result<&Settings, &str>) + Send + 'static) {
        if self.file.is_none() {
            return;
        }
        let shared = self.clone();
        let mut seen = Seen {
            text: self.loaded.clone(),
            ..Seen::default()
        };
        let watching = std::thread::Builder::new()
            .name("settings".into())
            .spawn(move || loop {
                match shared.look(&mut seen) {
                    Some(Ok(new)) => report(Ok(&new)),
                    Some(Err(reason)) => report(Err(&reason)),
                    None => {}
                }
                std::thread::sleep(PERIOD);
            });
        if let Err(error) = watching {
            eprintln!("tet-web: the settings file is not watched: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(list: &[(&str, &str)]) -> Vec<(String, String)> {
        list.iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    fn from(list: &[(&str, &str)]) -> Result<Settings, String> {
        let list = pairs(list);
        Settings::from(|name| {
            let found = list.iter().find(|(set, _)| set == name);
            found.map(|(_, value)| value.clone())
        })
    }

    #[test]
    fn the_defaults_are_two_minutes_three_hours_and_a_slot_for_each_processor() {
        let settings = from(&[]).unwrap();
        let processors = std::thread::available_parallelism().unwrap().get();
        assert_eq!(
            settings,
            Settings {
                wait: Duration::from_secs(120),
                timeout: Duration::from_secs(10_800),
                jobs: processors.max(4),
                threads: None,
                memory: None,
                nice: None,
            }
        );
        // An empty value is the default too.
        assert_eq!(
            from(&[("TET_WEB_TIMEOUT", ""), ("TET_WEB_JOBS", " ")]).unwrap(),
            settings
        );
    }

    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    #[test]
    fn values_are_numbers_and_times_have_units() {
        let settings = from(&[
            ("TET_WEB_WAIT", "90"),
            ("TET_WEB_TIMEOUT", "6h"),
            ("TET_WEB_JOBS", "2"),
            ("TET_WEB_THREADS", "3"),
            ("TET_WEB_MEMORY_MB", "512"),
            ("TET_WEB_NICE", "10"),
        ])
        .unwrap();
        assert_eq!(
            settings,
            Settings {
                wait: Duration::from_secs(90),
                timeout: Duration::from_secs(21_600),
                jobs: 2,
                threads: Some(3),
                memory: Some(512 << 20),
                nice: Some(10),
            }
        );
        assert_eq!(
            settings.described(),
            "a request waits 1 min 30 s, a calculation may take 6 hours, 2 at a time, threads \
             of one tet: 3, its memory: 512 MiB, its niceness: 10"
        );
        for (text, expected) in [
            ("1", 1),
            ("45s", 45),
            ("15m", 900),
            ("3h", 10_800),
            ("2d", 172_800),
            ("3 h", 10_800),
        ] {
            assert_eq!(seconds("T", text), Ok(expected), "{text}");
        }
        for text in [
            "0",
            "0h",
            "h",
            "3x",
            "-3",
            "1.5h",
            "3hh",
            "1e3",
            "99999999999999999999d",
        ] {
            let error = seconds("T", text).unwrap_err();
            assert!(
                error.starts_with(&format!("T={text:?} is not a time")),
                "{error}"
            );
        }
        for (list, reason) in [
            (
                &[("TET_WEB_JOBS", "0")][..],
                "TET_WEB_JOBS must be at least 1",
            ),
            (
                &[("TET_WEB_JOBS", "many")],
                "TET_WEB_JOBS=\"many\" is not a whole number",
            ),
            (&[("TET_WEB_NICE", "21")], "TET_WEB_NICE is at most 20"),
            (
                &[("TET_WEB_THREADS", "+2")],
                "TET_WEB_THREADS=\"+2\" is not a whole number",
            ),
            (&[("TET_WEB_WAIT", "0")], "TET_WEB_WAIT=\"0\" is not a time"),
        ] {
            let error = from(list).unwrap_err();
            assert!(error.starts_with(reason), "{error}");
        }
    }

    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    #[test]
    fn the_file_that_is_installed_sets_what_it_says() {
        let written = parsed(include_str!("../../freebsd/tet-web.conf")).unwrap();
        // It names every setting, so that each can be changed in place.
        let names: Vec<&str> = written.iter().map(|(name, _)| name.as_str()).collect();
        let mut expected = NAMES.to_vec();
        expected.swap(0, 1);
        assert_eq!(names, expected);
        let settings = Settings::from(|name| {
            let found = written.iter().find(|(set, _)| set == name);
            found.map(|(_, value)| value.clone())
        });
        assert_eq!(
            settings,
            Ok(Settings {
                wait: Duration::from_secs(120),
                timeout: Duration::from_secs(3 * 3600),
                jobs: default_jobs(),
                threads: None,
                memory: Some(512 << 20),
                nice: Some(10),
            })
        );
    }

    #[test]
    fn a_file_has_lines_of_names_and_values() {
        let text =
            "# The settings.\n\nTET_WEB_TIMEOUT=3h   # three hours\n  TET_WEB_WAIT = \"120\"\n\
                    TET_WEB_JOBS=\n#TET_WEB_NICE=5\n";
        assert_eq!(
            parsed(text).unwrap(),
            pairs(&[
                ("TET_WEB_TIMEOUT", "3h"),
                ("TET_WEB_WAIT", "120"),
                ("TET_WEB_JOBS", "")
            ])
        );
        assert_eq!(parsed("").unwrap(), []);
        for (text, reason) in [
            (
                "TET_WEB_TIMEOUT 3h",
                "line 1: \"TET_WEB_TIMEOUT 3h\" is not NAME=value",
            ),
            (
                "\nTET_WEB_TIMOUT=3h",
                "line 2: TET_WEB_TIMOUT is not a setting of this file",
            ),
            (
                "TET_WEB_LISTEN=0.0.0.0:80",
                "line 1: TET_WEB_LISTEN is not a setting of this file",
            ),
            (
                "TET_WEB_WAIT=1\nTET_WEB_WAIT=2",
                "line 2: TET_WEB_WAIT is set twice",
            ),
        ] {
            let error = parsed(text).unwrap_err();
            assert!(error.starts_with(reason), "{error}");
        }
    }

    #[test]
    fn the_file_is_above_the_environment_and_a_bad_file_changes_nothing() {
        let path = std::env::temp_dir().join(format!("tet-web-settings-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let environment = pairs(&[("TET_WEB_WAIT", "7"), ("TET_WEB_TIMEOUT", "50")]);
        let shared = Shared::load(environment, Some(path.clone())).unwrap();
        // No file: the environment and the defaults.
        let times = |shared: &Shared| {
            let now = shared.now();
            (now.wait.as_secs(), now.timeout.as_secs(), now.jobs)
        };
        let jobs = shared.now().jobs;
        assert_eq!(times(&shared), (7, 50, jobs));
        let mut seen = Seen::default();
        assert_eq!(shared.look(&mut seen), None);
        std::fs::write(&path, "TET_WEB_TIMEOUT=2h\nTET_WEB_JOBS=1\n").unwrap();
        let new = shared.look(&mut seen).unwrap().unwrap();
        assert_eq!(new, shared.now());
        assert_eq!(times(&shared), (7, 7200, 1));
        assert_eq!(shared.look(&mut seen), None);
        // Another text with the same settings is nothing to tell.
        std::fs::write(
            &path,
            "# Two hours.\nTET_WEB_TIMEOUT=120m\nTET_WEB_JOBS=1\n",
        )
        .unwrap();
        assert_eq!(shared.look(&mut seen), None);
        // What cannot be read as settings is said once and changes nothing.
        for (bad, reason) in [
            (
                "TET_WEB_TIMEOUT=soon\n",
                "TET_WEB_TIMEOUT=\"soon\" is not a time",
            ),
            (
                "TIMEOUT=5\n",
                "line 1: TIMEOUT is not a setting of this file",
            ),
        ] {
            std::fs::write(&path, bad).unwrap();
            let said = shared.look(&mut seen).unwrap().unwrap_err();
            let expected = format!("{}: {reason}", path.display());
            assert!(said.starts_with(&expected), "{said}");
            assert_eq!(shared.look(&mut seen), None);
            assert_eq!(times(&shared), (7, 7200, 1));
        }
        // A name that is taken out of the file is that of the environment
        // again.
        std::fs::write(&path, "TET_WEB_WAIT=9\n").unwrap();
        assert!(shared.look(&mut seen).unwrap().is_ok());
        assert_eq!(times(&shared), (9, 50, jobs));
        // A file that is away at one look is being replaced.
        std::fs::remove_file(&path).unwrap();
        assert_eq!(shared.look(&mut seen), None);
        assert_eq!(times(&shared), (9, 50, jobs));
        std::fs::write(&path, "TET_WEB_WAIT=11\n").unwrap();
        assert!(shared.look(&mut seen).unwrap().is_ok());
        assert_eq!(times(&shared), (11, 50, jobs));
        // One that stays away sets nothing any more.
        std::fs::remove_file(&path).unwrap();
        assert_eq!(shared.look(&mut seen), None);
        assert!(shared.look(&mut seen).unwrap().is_ok());
        assert_eq!(times(&shared), (7, 50, jobs));
        // A value of the environment that is wrong is not blamed on the file.
        let wrong = Shared::load(pairs(&[("TET_WEB_WAIT", "never")]), Some(path.clone()));
        let reason = wrong.err().unwrap();
        assert!(
            reason.starts_with("TET_WEB_WAIT=\"never\" is not a time"),
            "{reason}"
        );
    }
}
