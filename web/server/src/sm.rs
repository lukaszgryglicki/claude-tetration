//! The second program of the page: `symbolic-math`, a computer algebra
//! system. A request names its inputs, one a line, and a few options; the
//! server runs the program on them in a directory of its own and returns
//! what it printed and the files that it wrote there, pictures as a rule.
//!
//! The program runs in its sandbox (`SYMBOLIC_MATH_SANDBOX`): it reads and
//! writes plain names of that directory only. The directory is removed when
//! the run has ended; the files are kept with the output, in the cache.

use std::fmt::Write as _;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::api::{json_string, shell_quoted, Format, Outcome, Pending, Rejected, Written};
use crate::run::{exited, limits};
use crate::settings::Shared;

/// The longest text of inputs: it travels in the address of a request.
pub const INPUT_LIMIT: usize = 8000;
/// The most that is kept of what the program prints.
const OUTPUT_LIMIT: u64 = 4 << 20;
/// The files of one run that are kept, and their bytes together.
const FILES: usize = 32;
const FILES_LIMIT: u64 = 48 << 20;
/// The megabytes that the sandbox of the program lets it write.
const SANDBOX_MB: u64 = 48;
/// The largest file that the system lets the program write, its output
/// included.
const FILE_SIZE: u64 = 64 << 20;

/// What the program prints for a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The exact form and its numeric value.
    Both,
    Numeric,
    Exact,
}

/// The numbers that the program works with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    Complex,
    Real,
    Quaternion,
}

impl Mode {
    pub fn named(name: &str) -> Option<Mode> {
        match name {
            "both" => Some(Mode::Both),
            "numeric" => Some(Mode::Numeric),
            "exact" => Some(Mode::Exact),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Mode::Both => "both",
            Mode::Numeric => "numeric",
            Mode::Exact => "exact",
        }
    }
}

impl Domain {
    pub fn named(name: &str) -> Option<Domain> {
        match name {
            "complex" => Some(Domain::Complex),
            "real" => Some(Domain::Real),
            "quaternion" => Some(Domain::Quaternion),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Domain::Complex => "complex",
            Domain::Real => "real",
            Domain::Quaternion => "quaternion",
        }
    }
}

/// One run of the program: its inputs and its options.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    /// The inputs, one a line.
    pub text: String,
    /// Significant digits; without them the program takes 32, or as many
    /// as a number of the inputs has.
    pub digits: Option<u64>,
    pub mode: Mode,
    pub domain: Domain,
}

fn rejected(message: impl Into<String>) -> Rejected {
    Rejected {
        status: 400,
        message: message.into(),
    }
}

/// The inputs as they were typed: line ends of every kind are one, and
/// blanks at the ends of lines are nothing.
pub fn tidy(text: &str) -> String {
    let lines: Vec<&str> = text
        .split('\n')
        .map(|line| line.trim_end_matches('\r').trim_end())
        .collect();
    lines.join("\n").trim_matches('\n').to_string()
}

impl Program {
    /// The run that a query asks for, the format of the reply, and the
    /// file of the run that is asked for instead of it.
    pub fn parse(query: &[(String, String)]) -> Result<(Program, Format), Rejected> {
        let get = |name: &str| {
            query
                .iter()
                .find(|(found, _)| found == name)
                .map(|(_, value)| value.as_str())
        };
        for (name, _) in query {
            if !["expr", "digits", "mode", "domain", "format", "again"].contains(&name.as_str()) {
                return Err(rejected(format!(
                    "unknown parameter {name:?}: /sm takes expr, digits, mode (both, numeric, \
                     exact), domain (complex, real, quaternion) and format (json, text, html)"
                )));
            }
        }
        let format = match get("format") {
            None | Some("json") => Format::Json,
            Some("text") => Format::Text,
            Some(other) => {
                return Err(rejected(format!(
                    "format is json, text or html, not {other:?}"
                )))
            }
        };
        let text =
            tidy(get("expr").ok_or_else(|| {
                rejected("expr is missing: the inputs of symbolic-math, one a line")
            })?);
        if text.trim().is_empty() {
            return Err(rejected(
                "expr is empty: the inputs of symbolic-math, one a line",
            ));
        }
        if text.len() > INPUT_LIMIT {
            return Err(rejected(format!(
                "expr is longer than the {INPUT_LIMIT} bytes that this server takes"
            )));
        }
        if text
            .chars()
            .any(|c| c.is_control() && c != '\n' && c != '\t')
        {
            return Err(rejected("expr has a control character"));
        }
        let digits = match get("digits").map(str::trim) {
            None | Some("") => None,
            Some(text) => Some(
                text.parse::<u64>()
                    .ok()
                    .filter(|digits| (1..=1_000_000).contains(digits))
                    .filter(|_| text.bytes().all(|byte| byte.is_ascii_digit()))
                    .ok_or_else(|| {
                        rejected(format!(
                            "digits is a whole number from 1 to 1000000, not {text:?}"
                        ))
                    })?,
            ),
        };
        let mode = match get("mode") {
            None | Some("") => Mode::Both,
            Some(name) => Mode::named(name)
                .ok_or_else(|| rejected(format!("mode is both, numeric or exact, not {name:?}")))?,
        };
        let domain = match get("domain") {
            None | Some("") => Domain::Complex,
            Some(name) => Domain::named(name).ok_or_else(|| {
                rejected(format!(
                    "domain is complex, real or quaternion, not {name:?}"
                ))
            })?,
        };
        let program = Program {
            text,
            digits,
            mode,
            domain,
        };
        Ok((program, format))
    }

    /// The options of the program for this run.
    pub fn options(&self) -> Vec<String> {
        let mut options = Vec::new();
        if let Some(digits) = self.digits {
            options.push("-d".to_string());
            options.push(digits.to_string());
        }
        match self.mode {
            Mode::Both => {}
            Mode::Numeric => options.push("-n".to_string()),
            Mode::Exact => options.push("-e".to_string()),
        }
        match self.domain {
            Domain::Complex => {}
            Domain::Real => options.push("-r".to_string()),
            Domain::Quaternion => options.push("-q".to_string()),
        }
        options
    }

    /// The lines that are inputs: the program skips empty lines and the
    /// ones that begin with `#`.
    pub fn inputs(&self) -> Vec<&str> {
        self.text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .collect()
    }

    /// The command line to run by hand, quoted for a shell: every input is
    /// an argument.
    pub fn command(&self) -> String {
        let mut words = vec!["symbolic-math".to_string()];
        words.extend(self.options());
        words.extend(self.inputs().iter().map(|input| shell_quoted(input)));
        words.join(" ")
    }

    /// What identifies the run.
    pub fn key(&self) -> String {
        format!(
            "SM\n{}\n{}\n{}\n{}",
            self.digits
                .map_or(String::new(), |digits| digits.to_string()),
            self.mode.name(),
            self.domain.name(),
            self.text
        )
    }

    /// The run that `key` made a key from.
    pub fn of_key(key: &str) -> Option<Program> {
        let mut parts = key.strip_prefix("SM\n")?.splitn(4, '\n');
        let digits = match parts.next()? {
            "" => None,
            digits => Some(digits.parse().ok()?),
        };
        Some(Program {
            mode: Mode::named(parts.next()?)?,
            domain: Domain::named(parts.next()?)?,
            text: parts.next()?.to_string(),
            digits,
        })
    }

    /// A short name of the run for the addresses of its files: the same
    /// for the same run, whenever it is asked for.
    pub fn id(&self) -> String {
        id_of(&self.key())
    }
}

/// The name that the files of the run with this key have in addresses.
pub fn id_of(key: &str) -> String {
    // (two hashes with fixed keys: 128 bits that do not change between runs
    // of the server)
    let hash = |seed: u8| {
        #[allow(deprecated)]
        let mut hasher = std::hash::SipHasher::new_with_keys(0x7465_7477_6562, u64::from(seed));
        key.hash(&mut hasher);
        hasher.finish()
    };
    format!("{:016x}{:016x}", hash(1), hash(2))
}

/// A name that a file of a run may have: what the sandbox of the program
/// allows, so that it is safe in an address and in a header.
pub fn plain(name: &str) -> bool {
    (1..=96).contains(&name.len())
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-_.".contains(&byte))
}

/// The kind of a file by the end of its name, and whether a browser shows
/// it as a picture.
pub fn kind(name: &str) -> (&'static str, bool) {
    let ending = name.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    match ending.as_str() {
        "png" => ("image/png", true),
        "jxl" => ("image/jxl", false),
        "csv" | "txt" | "tsv" | "dat" => ("text/plain; charset=utf-8", false),
        _ => ("application/octet-stream", false),
    }
}

/// A number of bytes as people say it.
pub fn size(bytes: usize) -> String {
    match bytes {
        0..=1023 => format!("{bytes} bytes"),
        1024..=1_048_575 => format!("{:.1} KiB", bytes as f64 / 1024.0),
        _ => format!("{:.1} MiB", bytes as f64 / 1_048_576.0),
    }
}

/// How `symbolic-math` is run.
pub struct Runner {
    /// The program: a path, or a name for the search path.
    pub program: String,
    /// The search path of the program, on which it finds `tet`.
    pub path: String,
    /// Where the directories of the runs are made.
    pub work: PathBuf,
    /// What the program says about itself: its usage text and its lists
    /// of functions, constants and units.
    pub help: String,
    pub functions: String,
    pub constants: String,
    pub units: String,
}

static RUNS: AtomicU64 = AtomicU64::new(0);

#[cfg(unix)]
fn confine(command: &mut Command) {
    use std::os::raw::c_int;
    use std::os::unix::process::CommandExt;

    #[repr(C)]
    struct Limit {
        soft: u64,
        hard: u64,
    }
    extern "C" {
        fn setrlimit(resource: c_int, limit: *const Limit) -> c_int;
    }
    /// `RLIMIT_FSIZE` and `RLIMIT_CORE`, on Linux and on FreeBSD.
    const FILE: c_int = 1;
    const CORE: c_int = 4;
    // The program and the `tet` processes that it starts are one group, so
    // that the time limit ends all of them.
    command.process_group(0);
    // SAFETY: the closure runs in the child between fork and exec and only
    // makes system calls.
    unsafe {
        command.pre_exec(|| {
            for (resource, bytes) in [(FILE, FILE_SIZE), (CORE, 0)] {
                let limit = Limit {
                    soft: bytes,
                    hard: bytes,
                };
                if setrlimit(resource, &limit) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
}

#[cfg(not(unix))]
fn confine(_: &mut Command) {}

/// Ends the program with the processes that it started.
fn stop(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        extern "C" {
            fn kill(pid: std::os::raw::c_int, signal: std::os::raw::c_int) -> std::os::raw::c_int;
        }
        if let Ok(group) = std::os::raw::c_int::try_from(child.id()) {
            // SAFETY: a signal to the group of the child, which is its own.
            unsafe {
                kill(-group, 9);
            }
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// What `program` prints when it is asked for one of its texts.
fn told(program: &str, option: &str) -> Result<String, String> {
    let output = Command::new(program)
        .arg(option)
        .stdin(Stdio::null())
        .output()
        .map_err(|error| format!("{program} could not be run: {error}"))?;
    if !output.status.success() || output.stdout.is_empty() {
        return Err(format!("{program} {option} failed ({})", output.status));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

impl Runner {
    /// The runner of `program`, which finds `tet` in the directory of the
    /// `tet` of this server.
    pub fn new(program: String, tet: &str) -> Result<Runner, String> {
        let mut path = Vec::new();
        if let Some(directory) = Path::new(tet)
            .parent()
            .filter(|d| !d.as_os_str().is_empty())
        {
            path.push(directory.to_string_lossy().into_owned());
        }
        path.extend(["/usr/local/bin", "/usr/bin", "/bin"].map(str::to_string));
        Ok(Runner {
            help: told(&program, "--help")?,
            functions: told(&program, "--functions")?,
            constants: told(&program, "--constants")?,
            units: told(&program, "--units")?,
            path: path.join(":"),
            work: std::env::temp_dir().join(format!("tet-web-sm-{}", std::process::id())),
            program,
        })
    }

    /// A new directory for one run. The directory of all runs is there
    /// while one runs: the run that ends last takes it away, and may do so
    /// just when another one begins.
    fn directory(&self) -> std::io::Result<PathBuf> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.subsec_nanos());
        let directory = self.work.join(format!(
            "run-{}-{nanos}",
            RUNS.fetch_add(1, Ordering::SeqCst)
        ));
        let mut made = Ok(());
        for _ in 0..8 {
            std::fs::create_dir_all(&self.work)?;
            made = std::fs::create_dir(&directory);
            if made.is_ok() {
                break;
            }
        }
        made?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700))?;
        }
        Ok(directory)
    }

    /// Runs the program on the inputs and reads what it printed and wrote.
    pub fn run(&self, program: &Program, settings: &Shared) -> Outcome {
        let failed = |message: String| Outcome::Failed { message };
        let directory = match self.directory() {
            Ok(directory) => directory,
            Err(error) => return failed(format!("no directory for the run: {error}")),
        };
        let outcome = self.run_in(&directory, program, settings);
        let _ = std::fs::remove_dir_all(&directory);
        // (only when no other run has its directory there)
        let _ = std::fs::remove_dir(&self.work);
        outcome
    }

    fn run_in(&self, directory: &Path, program: &Program, settings: &Shared) -> Outcome {
        let failed = |message: String| Outcome::Failed { message };
        let now = settings.now();
        // What the program prints, errors among its results as they come.
        let printed = directory.join(".output");
        let (out, err) = match File::create(&printed).and_then(|out| Ok((out.try_clone()?, out))) {
            Ok(both) => both,
            Err(error) => return failed(format!("no file for the output: {error}")),
        };
        let mut command = Command::new(&self.program);
        command
            .args(program.options())
            .current_dir(directory)
            .env_clear()
            .env("PATH", &self.path)
            .env("HOME", directory)
            .env("TMPDIR", directory)
            .env("SYMBOLIC_MATH_SANDBOX", SANDBOX_MB.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::from(out))
            .stderr(Stdio::from(err));
        limits::apply(&mut command, now.memory, now.nice);
        confine(&mut command);
        let started = Instant::now();
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => return failed(format!("symbolic-math could not be started: {error}")),
        };
        // The inputs are shorter than a pipe holds: writing them never waits.
        if let Some(mut input) = child.stdin.take() {
            let _ = input.write_all(program.text.as_bytes());
            let _ = input.write_all(b"\n");
        }
        let status = match exited(&mut child, started, || settings.now().timeout) {
            Ok((Some(status), _)) => {
                // (what the program left running ends with it)
                stop(&mut child);
                status
            }
            Ok((None, limit)) => {
                stop(&mut child);
                return Outcome::Timeout {
                    limit: limit.as_secs(),
                };
            }
            Err(error) => {
                stop(&mut child);
                return failed(format!("symbolic-math could not be waited for: {error}"));
            }
        };
        let seconds = started.elapsed().as_secs_f64();
        let mut text = Vec::new();
        let read = File::open(&printed)
            .and_then(|file| file.take(OUTPUT_LIMIT + 1).read_to_end(&mut text));
        if let Err(error) = read {
            return failed(format!("the output could not be read: {error}"));
        }
        let more = text.len() as u64 > OUTPUT_LIMIT;
        text.truncate(OUTPUT_LIMIT as usize);
        let mut text = String::from_utf8_lossy(&text).into_owned();
        if more {
            let _ = write!(
                text,
                "\nnote: the output is longer than the {} MiB that this server returns\n",
                OUTPUT_LIMIT >> 20
            );
        }
        let files = match written(directory, &mut text) {
            Ok(files) => files,
            Err(error) => return failed(format!("the files could not be read: {error}")),
        };
        match status.code() {
            Some(code @ (0 | 1)) => Outcome::Output {
                text,
                files,
                clean: code == 0,
                seconds,
            },
            Some(2) => Outcome::Refused {
                message: text.trim().trim_start_matches("error: ").to_string(),
                seconds,
            },
            Some(code) => failed(format!(
                "symbolic-math ended with the unexpected exit code {code}"
            )),
            None => failed(format!(
                "symbolic-math was stopped by the system ({status}): it needed more memory than \
                 this server allows, wrote a larger file, or it was ended by hand"
            )),
        }
    }
}

/// The files that a run left in its directory, by their names; what is too
/// many or too much is named in the output instead.
fn written(directory: &Path, text: &mut String) -> std::io::Result<Vec<Written>> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if plain(&name) && entry.file_type()?.is_file() {
            names.push((name, entry.metadata()?.len()));
        }
    }
    names.sort();
    let (mut files, mut total) = (Vec::new(), 0u64);
    for (name, length) in names {
        if files.len() >= FILES || total + length > FILES_LIMIT {
            let _ = writeln!(
                text,
                "note: the file {name} is not kept: this server keeps {FILES} files and {} MiB of a run",
                FILES_LIMIT >> 20
            );
            continue;
        }
        total += length;
        files.push(Written {
            bytes: std::fs::read(directory.join(&name))?,
            name,
        });
    }
    Ok(files)
}

/// The request as the first fields of a JSON reply.
fn request_fields(program: &Program) -> Vec<String> {
    let mut fields = vec![format!("\"expr\":{}", json_string(&program.text))];
    if let Some(digits) = program.digits {
        fields.push(format!("\"digits\":{digits}"));
    }
    fields.push(format!("\"mode\":{}", json_string(program.mode.name())));
    fields.push(format!("\"domain\":{}", json_string(program.domain.name())));
    fields
}

/// The body of the reply about a run that has not ended.
pub fn rendered_pending(program: &Program, pending: &Pending, format: Format) -> String {
    if format == Format::Text {
        return format!("running: {}\n", pending.message());
    }
    let mut fields = request_fields(program);
    fields.push("\"running\":true".into());
    fields.push(format!("\"message\":{}", json_string(&pending.message())));
    fields.push("\"status\":202".into());
    fields.push(format!(
        "\"running_seconds\":{:.3}",
        pending.age.as_secs_f64()
    ));
    fields.push(format!("\"limit_seconds\":{}", pending.limit));
    fields.push("\"cached\":false".into());
    fields.push(format!("\"command\":{}", json_string(&program.command())));
    format!("{{{}}}\n", fields.join(","))
}

/// The body of the reply to a run.
pub fn rendered(program: &Program, outcome: &Outcome, cached: bool, format: Format) -> String {
    if format == Format::Text {
        return match outcome {
            Outcome::Output { text, .. } => text.clone(),
            other => format!("error: {}\n", other.error().unwrap_or_default()),
        };
    }
    let mut fields = request_fields(program);
    match outcome {
        Outcome::Output {
            text, files, clean, ..
        } => {
            let id = program.id();
            fields.push(format!("\"output\":{}", json_string(text)));
            let listed: Vec<String> = files
                .iter()
                .map(|file| {
                    format!(
                        "{{\"name\":{},\"bytes\":{},\"url\":{}}}",
                        json_string(&file.name),
                        file.bytes.len(),
                        json_string(&format!("/sm/file/{id}/{}", file.name))
                    )
                })
                .collect();
            fields.push(format!("\"files\":[{}]", listed.join(",")));
            fields.push(format!("\"ok\":{clean}"));
            if !clean {
                fields.push("\"status\":422".into());
            }
            fields.push(format!("\"id\":{}", json_string(&id)));
        }
        other => {
            fields.push(format!(
                "\"error\":{}",
                json_string(&other.error().unwrap_or_default())
            ));
            fields.push(format!("\"status\":{}", other.status()));
        }
    }
    if let Some(seconds) = outcome.seconds() {
        fields.push(format!("\"seconds\":{seconds:.3}"));
    }
    fields.push(format!("\"cached\":{cached}"));
    fields.push(format!("\"command\":{}", json_string(&program.command())));
    format!("{{{}}}\n", fields.join(","))
}

/// A run as one line for people.
pub fn listed(program: &Program, found: Result<&Outcome, std::time::Duration>) -> String {
    let mut words = vec!["f=symbolic-math".to_string()];
    words.extend(program.options());
    words.push(format!(
        "expr={}",
        shell_quoted(&program.text.replace('\n', "; "))
    ));
    match found {
        Ok(Outcome::Output { text, files, .. }) => {
            words.push(format!("lines={}", text.lines().count()));
            words.push(format!("files={}", files.len()));
        }
        Ok(other) => {
            let reason = other.error().unwrap_or_default();
            words.push(format!("error={}", shell_quoted(&reason)));
        }
        Err(age) => words.push(format!("running={}s", age.as_secs())),
    }
    words.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    #[test]
    fn a_query_is_a_run_of_the_program() {
        let (program, format) = Program::parse(&query(&[("expr", "2+2")])).unwrap();
        assert_eq!(format, Format::Json);
        assert_eq!(
            (program.digits, program.mode, program.domain),
            (None, Mode::Both, Domain::Complex)
        );
        assert_eq!(program.command(), "symbolic-math 2+2");
        let (program, format) = Program::parse(&query(&[
            ("expr", "a := 2\r\n\r\n# a remark\r\n  sin(a x)  \r\n"),
            ("digits", "50"),
            ("mode", "numeric"),
            ("domain", "real"),
            ("format", "text"),
        ]))
        .unwrap();
        assert_eq!(format, Format::Text);
        assert_eq!(program.text, "a := 2\n\n# a remark\n  sin(a x)");
        assert_eq!(program.inputs(), ["a := 2", "sin(a x)"]);
        assert_eq!(program.options(), ["-d", "50", "-n", "-r"]);
        assert_eq!(
            program.command(),
            "symbolic-math -d 50 -n -r 'a := 2' 'sin(a x)'"
        );
        assert_eq!(Program::of_key(&program.key()), Some(program.clone()));
        // The name of a run is the same whenever it is asked for.
        assert_eq!(program.id(), id_of(&program.key()));
        assert_eq!(program.id().len(), 32);
        let other = Program::parse(&query(&[("expr", "a := 2\nsin(a x)")]))
            .unwrap()
            .0;
        assert_ne!(program.id(), other.id());
    }

    #[test]
    fn what_is_no_run_is_refused() {
        for (pairs, message) in [
            (&[("digits", "5")][..], "expr is missing"),
            (&[("expr", "  \n ")], "expr is empty"),
            (
                &[("expr", "1"), ("digits", "0")],
                "digits is a whole number",
            ),
            (
                &[("expr", "1"), ("digits", "+5")],
                "digits is a whole number",
            ),
            (
                &[("expr", "1"), ("mode", "fast")],
                "mode is both, numeric or exact",
            ),
            (
                &[("expr", "1"), ("domain", "p-adic")],
                "domain is complex, real or quaternion",
            ),
            (
                &[("expr", "1"), ("format", "xml")],
                "format is json, text or html",
            ),
            (
                &[("expr", "1"), ("file", "x")],
                "unknown parameter \"file\"",
            ),
            (&[("expr", "1\u{0}2")], "control character"),
        ] {
            let refused = Program::parse(&query(pairs)).unwrap_err();
            assert_eq!(refused.status, 400);
            assert!(refused.message.contains(message), "{}", refused.message);
        }
        let long = "1+".repeat(INPUT_LIMIT / 2 + 1);
        assert!(Program::parse(&query(&[("expr", &long)])).is_err());
    }

    #[test]
    fn files_have_plain_names_and_kinds() {
        for name in ["plot.png", "walk3d.jxl", "data-1_2.csv", "A"] {
            assert!(plain(name), "{name}");
        }
        for name in ["", ".output", "..", "a/b", "a b", "x\"y", "a\nb", "é.png"] {
            assert!(!plain(name), "{name}");
        }
        assert_eq!(kind("plot.PNG"), ("image/png", true));
        assert_eq!(kind("walk.jxl"), ("image/jxl", false));
        assert_eq!(kind("m.csv"), ("text/plain; charset=utf-8", false));
        assert_eq!(kind("thing"), ("application/octet-stream", false));
        assert_eq!(size(100), "100 bytes");
        assert_eq!(size(2048), "2.0 KiB");
        assert_eq!(size(3 << 20), "3.0 MiB");
    }

    #[test]
    fn replies_name_the_output_and_the_files() {
        let program = Program::parse(&query(&[("expr", "save_png(\"p.png\", plot(x, x, 0, 1))")]))
            .unwrap()
            .0;
        let outcome = Outcome::Output {
            text: "saved p.png\n".into(),
            files: vec![Written {
                name: "p.png".into(),
                bytes: vec![1, 2, 3],
            }],
            clean: true,
            seconds: 0.25,
        };
        let json = rendered(&program, &outcome, false, Format::Json);
        assert!(json.contains("\"output\":\"saved p.png\\n\""), "{json}");
        assert!(json.contains(&format!(
            "\"files\":[{{\"name\":\"p.png\",\"bytes\":3,\"url\":\"/sm/file/{}/p.png\"}}]",
            program.id()
        )));
        assert!(json.contains("\"ok\":true") && json.contains("\"seconds\":0.250"));
        assert_eq!(
            rendered(&program, &outcome, true, Format::Text),
            "saved p.png\n"
        );
        let busy = rendered(&program, &Outcome::Busy, false, Format::Json);
        assert!(busy.contains("\"error\":\"all slots are busy") && busy.contains("\"status\":503"));
        assert_eq!(
            listed(&program, Ok(&outcome)),
            "f=symbolic-math expr='save_png(\"p.png\", plot(x, x, 0, 1))' lines=1 files=1"
        );
    }
}
