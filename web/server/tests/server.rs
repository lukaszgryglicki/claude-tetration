//! The server as a process: its page, its API, its cache and its limits,
//! with a shell script in the place of `tet`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Prints its mode, precision and arguments as the real part and its
/// `TET_MT` as the imaginary part; some first arguments are special.
const FAKE_TET: &str = r#"#!/bin/sh
if [ "$1" = --help ]; then echo 'Usage: tet (a script for the tests)'; exit 0; fi
echo "$*" >> "$0.calls"
[ "$SILENT" = 1 ] && [ "$1" = -q ] || { echo 'not called as expected' >&2; exit 3; }
shift
case "$1" in --slog | --ate | --sroot | --ssrt) mode=$1; shift ;; *) mode=tet ;; esac
prec=$1; shift
case "$1" in
    slow) echo $$ > "$0.pid"; exec sleep 30 ;;
    wait) sleep 1; printf '%s\n0\n' "$2" ;;
    refuse) echo 'a line of progress' >&2
        echo 'error: invalid number "refuse": unknown name refuse' >&2; exit 1 ;;
    usage) exit 2 ;;
    garbage) echo 'one line only' ;;
    crash) kill -KILL $$ ;;
    limits) printf '%s\n%s\n' "$(ulimit -t)" "$(ulimit -v)" ;;
    nice) printf '%s\n0\n' "$(ps -o nice= -p $$ | tr -d ' ')" ;;
    *) printf '%s|%s|%s\n%s\n' "$mode" "$prec" "$*" "${TET_MT-serial}" ;;
esac
"#;

static SERVERS: AtomicUsize = AtomicUsize::new(0);

struct Server {
    directory: PathBuf,
    child: Child,
    address: String,
}

struct Reply {
    status: u16,
    head: String,
    body: String,
}

impl Reply {
    fn header(&self, name: &str) -> Option<&str> {
        self.head
            .lines()
            .find_map(|line| line.strip_prefix(&format!("{name}: ")))
    }
}

fn fixture_directory() -> PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "tet-web-test-{}-{}",
        std::process::id(),
        SERVERS.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir(&directory).unwrap();
    let tet = directory.join("tet");
    std::fs::write(&tet, FAKE_TET).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tet, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    directory
}

fn command(directory: &std::path::Path, settings: &[(&str, &str)]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_tet-web"));
    for (name, _) in std::env::vars() {
        if name.starts_with("TET_") {
            command.env_remove(name);
        }
    }
    command
        .env("TET_WEB_LISTEN", "127.0.0.1:0")
        .env("TET_WEB_TET", directory.join("tet"))
        .envs(settings.iter().copied())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

impl Server {
    fn start(settings: &[(&str, &str)]) -> Server {
        let directory = fixture_directory();
        let mut child = command(&directory, settings).spawn().unwrap();
        let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
        let first = lines.next().expect("the server printed nothing").unwrap();
        let address = first
            .split("listening on ")
            .nth(1)
            .and_then(|rest| rest.split(": ").next())
            .unwrap_or_else(|| panic!("no address in {first:?}"))
            .to_string();
        // The log is read so that the server never blocks on it.
        std::thread::spawn(move || lines.for_each(drop));
        Server {
            directory,
            child,
            address,
        }
    }

    fn request(&self, method: &str, target: &str) -> Reply {
        let mut stream = TcpStream::connect(&self.address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(60)))
            .unwrap();
        write!(stream, "{method} {target} HTTP/1.1\r\nHost: test\r\n\r\n").unwrap();
        let mut text = String::new();
        stream.read_to_string(&mut text).unwrap();
        let (head, body) = text.split_once("\r\n\r\n").expect("no end of the head");
        let status = head
            .split(' ')
            .nth(1)
            .and_then(|code| code.parse().ok())
            .unwrap_or_else(|| panic!("no status in {head:?}"));
        Reply {
            status,
            head: head.to_string(),
            body: body.to_string(),
        }
    }

    fn get(&self, target: &str) -> Reply {
        self.request("GET", target)
    }

    /// The command lines with which `tet` was run.
    fn calls(&self) -> Vec<String> {
        std::fs::read_to_string(self.directory.join("tet.calls"))
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

#[test]
fn the_page_and_its_files_are_served() {
    let server = Server::start(&[("TET_WEB_REVISION", "abc1234")]);
    let page = server.get("/");
    assert_eq!(page.status, 200);
    assert_eq!(
        page.header("Content-Type"),
        Some("text/html; charset=utf-8")
    );
    assert!(page.body.starts_with("<!doctype html>") && page.body.contains("<title>tet"));
    let policy = page.header("Content-Security-Policy").unwrap();
    assert!(policy.contains("default-src 'none'") && policy.contains("script-src 'self'"));
    assert_eq!(page.header("X-Content-Type-Options"), Some("nosniff"));
    assert_eq!(page.header("Connection"), Some("close"));
    // A query without a parameter of the API is still the page.
    assert!(server
        .get("/?utm_source=x")
        .body
        .starts_with("<!doctype html>"));
    for (path, kind, text) in [
        (
            "/app.js",
            "text/javascript; charset=utf-8",
            "encodeURIComponent",
        ),
        ("/style.css", "text/css; charset=utf-8", "font"),
        ("/favicon.svg", "image/svg+xml", "<svg"),
        ("/favicon.ico", "image/svg+xml", "<svg"),
        (
            "/help",
            "text/plain; charset=utf-8",
            "Usage: tet (a script for the tests)",
        ),
        ("/healthz", "text/plain; charset=utf-8", "ok\n"),
        ("/robots.txt", "text/plain; charset=utf-8", "Disallow: /api"),
        (
            "/stats",
            "application/json; charset=utf-8",
            "\"revision\":\"abc1234\"",
        ),
    ] {
        let reply = server.get(path);
        assert_eq!(reply.status, 200, "{path}");
        assert_eq!(reply.header("Content-Type"), Some(kind), "{path}");
        assert!(reply.body.contains(text), "{path}: {}", reply.body);
        assert_eq!(
            reply.header("Content-Length"),
            Some(reply.body.len().to_string().as_str())
        );
    }
    let head = server.request("HEAD", "/");
    assert_eq!((head.status, head.body.as_str()), (200, ""));
    assert_eq!(
        head.header("Content-Length"),
        Some(page.body.len().to_string().as_str())
    );
    assert_eq!(server.get("/nothing").status, 404);
    assert_eq!(server.get("/index.html").status, 404);
    let post = server.request("POST", "/");
    assert_eq!(
        (post.status, post.header("Allow")),
        (405, Some("GET, HEAD"))
    );
    // A request head is at most 16 KiB, wherever its end arrives.
    let long = |length: usize| server.get(&format!("/{}", "a".repeat(length)));
    assert_eq!(long(16_000).status, 404);
    for length in [16_400, 17_000, 40_000, 400_000] {
        let refused = long(length);
        assert_eq!(refused.status, 431, "{length}");
        assert_eq!(refused.body, "the request head is too large\n");
    }
    assert!(server.calls().is_empty());
}

#[test]
fn the_api_runs_tet_and_keeps_what_it_computed() {
    let server = Server::start(&[]);
    let first = server.get("/?func=tet&base=2&height=0.5");
    assert_eq!(first.status, 200);
    assert_eq!(
        first.header("Content-Type"),
        Some("application/json; charset=utf-8")
    );
    assert_eq!(first.header("Cache-Control"), Some("no-store"));
    assert_eq!(first.header("Access-Control-Allow-Origin"), Some("*"));
    assert_eq!(first.header("X-Cache"), Some("MISS"));
    assert!(
        first.body.starts_with(
            "{\"func\":\"tet\",\"prec\":20,\"base\":\"2\",\"height\":\"0.5\",\"result\":\"value\",\
             \"re\":\"tet|20|2 0.5\",\"im\":\"serial\",\"seconds\":"
        ),
        "{}",
        first.body
    );
    assert!(first
        .body
        .ends_with(",\"cached\":false,\"command\":\"tet 20 2 0.5\"}\n"));

    // The same computation again, spelled in another order and at /api.
    let again = server.get("/api?height=0.5&prec=20&base=2&func=tet");
    assert_eq!((again.status, again.header("X-Cache")), (200, Some("HIT")));
    assert_eq!(
        again.body,
        first.body.replace("\"cached\":false", "\"cached\":true")
    );
    assert_eq!(server.calls(), ["-q 20 2 0.5"]);

    // A plus is a plus; %2B and %20 are a plus and a space; the text format
    // is what tet printed.
    let text = server.get("/?prec=5&func=ate&base=2&value=1.5+0.5i&format=text");
    assert_eq!(
        text.header("Content-Type"),
        Some("text/plain; charset=utf-8")
    );
    assert_eq!(
        (text.status, text.body.as_str()),
        (200, "--ate|5|2 1.5+0.5i\nserial\n")
    );
    let escaped = server.get("/?func=ssrt&value=1%20%2B%202&height=e%5E-e&format=text");
    assert_eq!(escaped.body, "--ssrt|20|1 + 2 e^-e\nserial\n");

    // ate and slog are one function: the second name is answered from the
    // result of the first.
    let slog = server.get("/?prec=5&func=slog&base=2&value=1.5+0.5i&format=text");
    assert_eq!(
        (slog.header("X-Cache"), slog.body.as_str()),
        (Some("HIT"), "--ate|5|2 1.5+0.5i\nserial\n")
    );

    // Four real numbers are another command line, and another entry.
    let four = server.get("/?func=tet&base_re=2&base_im=0&height_re=0.5&height_im=0");
    assert_eq!(four.header("X-Cache"), Some("MISS"));
    assert!(four.body.contains("\"re\":\"tet|20|2 0 0.5 0\""));
    assert!(four
        .body
        .contains("\"base_re\":\"2\",\"base_im\":\"0\",\"height_re\":\"0.5\""));
    assert_eq!(
        server.calls(),
        [
            "-q 20 2 0.5",
            "-q --ate 5 2 1.5+0.5i",
            "-q --ssrt 20 1 + 2 e^-e",
            "-q 20 2 0 0.5 0"
        ]
    );

    let stats = server.get("/stats").body;
    assert!(stats.contains("\"entries\":4") && stats.contains("\"hits\":2"));
    assert!(stats.contains("\"misses\":4") && stats.contains("\"running\":0"));
}

#[test]
fn refusals_are_kept_and_failures_are_not() {
    let server = Server::start(&[]);
    for (round, cache) in [(0, "MISS"), (1, "HIT")] {
        let refused = server.get("/?func=tet&base=refuse&height=1");
        assert_eq!(refused.status, 422, "{round}");
        assert_eq!(refused.header("X-Cache"), Some(cache));
        assert!(refused.body.contains(
            "\"error\":\"invalid number \\\"refuse\\\": unknown name refuse\",\"status\":422"
        ));
    }
    let text = server.get("/?func=tet&base=refuse&height=1&format=text");
    assert_eq!(
        (text.status, text.body.as_str()),
        (
            422,
            "error: invalid number \"refuse\": unknown name refuse\n"
        )
    );
    for (word, reason) in [
        ("usage", "unexpected exit code 2"),
        ("garbage", "did not print the two lines of a result"),
        ("crash", "stopped by the system"),
    ] {
        for _ in 0..2 {
            let failed = server.get(&format!("/?func=tet&base={word}&height=1"));
            assert_eq!(failed.status, 500, "{word}");
            assert_eq!(failed.header("X-Cache"), Some("MISS"), "{word}");
            assert!(failed.body.contains(reason), "{word}: {}", failed.body);
        }
    }
    assert_eq!(server.calls().len(), 1 + 3 * 2);

    // Requests that are no computation never reach tet.
    for (target, message) in [
        ("/?base=2&height=3", "func is missing"),
        ("/api", "func is missing"),
        ("/?func=tet&base=2", "func=tet takes base and height"),
        (
            "/?func=tet&base=2&height=3&heigth=4",
            "unknown parameter \\\"heigth\\\"",
        ),
        (
            "/?func=tet&base=2&height=3&base=4",
            "the parameter base is given twice",
        ),
        ("/?func=tet&base=%zz&height=3", "broken escape"),
        (
            "/?func=tet&base=2&height=3&prec=0",
            "prec is the number of digits",
        ),
        ("/?func=tet&base=2&height=3&prec=1001", "from 1 to 1000"),
        ("/?func=tet&base=--slog&height=3", "base is not a number"),
    ] {
        let reply = server.get(target);
        assert_eq!(reply.status, 400, "{target}");
        assert!(reply.body.starts_with("{\"error\":\""), "{target}");
        assert!(reply.body.contains(message), "{target}: {}", reply.body);
        assert!(reply.body.ends_with(",\"status\":400}\n"), "{target}");
    }
    let text = server.get("/?func=exp&format=text");
    assert_eq!(
        (text.status, text.body.as_str()),
        (
            400,
            "error: func is tet, ate, slog, sroot or ssrt, not \"exp\"\n"
        )
    );
    assert_eq!(server.calls().len(), 7);
}

#[test]
fn a_computation_ends_at_the_time_limit_and_is_not_kept() {
    let server = Server::start(&[("TET_WEB_TIMEOUT", "1")]);
    for round in 1..=2 {
        let started = Instant::now();
        let reply = server.get("/?func=tet&base=slow&height=1");
        assert!(started.elapsed() >= Duration::from_secs(1));
        assert!(started.elapsed() < Duration::from_secs(10));
        assert_eq!((reply.status, reply.header("X-Cache")), (504, Some("MISS")));
        assert!(reply
            .body
            .contains("\"error\":\"no result within 1 s, the limit of this server"));
        assert_eq!(server.calls().len(), round);
        // The process that was computing is gone.
        let pid = std::fs::read_to_string(server.directory.join("tet.pid")).unwrap();
        let alive = Command::new("kill")
            .args(["-0", pid.trim()])
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(!alive.success(), "process {pid} is still running");
    }
}

#[test]
fn one_slot_one_waiting_request_and_the_next_is_turned_away() {
    let server = Server::start(&[
        ("TET_WEB_JOBS", "1"),
        ("TET_WEB_QUEUE", "1"),
        ("TET_WEB_TIMEOUT", "5"),
    ]);
    let ask =
        |height: u32| server.get(&format!("/?func=tet&base=wait&height={height}&format=text"));
    std::thread::scope(|scope| {
        let running = scope.spawn(|| ask(1));
        std::thread::sleep(Duration::from_millis(300));
        let waiting = scope.spawn(|| ask(2));
        std::thread::sleep(Duration::from_millis(300));
        let started = Instant::now();
        let third = ask(3);
        assert!(started.elapsed() < Duration::from_millis(500));
        assert_eq!(
            (third.status, third.body.as_str()),
            (
                503,
                "error: the server is busy with other computations; try again\n"
            )
        );
        let (running, waiting) = (running.join().unwrap(), waiting.join().unwrap());
        assert_eq!((running.status, running.body.as_str()), (200, "1\n0\n"));
        assert_eq!((waiting.status, waiting.body.as_str()), (200, "2\n0\n"));
    });
    assert_eq!(server.calls().len(), 2);
    // Being turned away is not kept: the same request now runs.
    assert_eq!(ask(3).body, "3\n0\n");
}

#[test]
fn identical_requests_in_flight_are_one_computation() {
    let server = Server::start(&[("TET_WEB_JOBS", "4")]);
    let ask = || server.get("/?func=tet&base=wait&height=7&format=text");
    let replies: Vec<Reply> = std::thread::scope(|scope| {
        let asked: Vec<_> = (0..4).map(|_| scope.spawn(ask)).collect();
        asked
            .into_iter()
            .map(|reply| reply.join().unwrap())
            .collect()
    });
    assert!(replies
        .iter()
        .all(|reply| (reply.status, reply.body.as_str()) == (200, "7\n0\n")));
    let count = |cache: &str| {
        replies
            .iter()
            .filter(|reply| reply.header("X-Cache") == Some(cache))
            .count()
    };
    assert_eq!((count("MISS"), count("WAIT")), (1, 3));
    assert_eq!(server.calls(), ["-q 20 wait 7"]);
    assert_eq!(ask().header("X-Cache"), Some("HIT"));
}

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
#[test]
fn a_child_gets_its_threads_and_its_limits() {
    let server = Server::start(&[
        ("TET_WEB_THREADS", "3"),
        ("TET_WEB_TIMEOUT", "10"),
        ("TET_WEB_MEMORY_MB", "512"),
    ]);
    let threads = server.get("/?func=tet&base=2&height=3&format=text");
    assert_eq!(threads.body, "tet|20|2 3\n3\n");
    // Processor time for the time limit, five seconds more, three threads;
    // address space in kilobytes.
    let limits = server.get("/?func=tet&base=limits&height=1&format=text");
    assert_eq!(limits.body, "45\n524288\n");
}

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
#[test]
fn a_child_is_nicer_than_the_server_when_that_is_asked() {
    let own = Command::new("ps")
        .args(["-o", "nice=", "-p", &std::process::id().to_string()])
        .output()
        .unwrap();
    let own: i32 = String::from_utf8_lossy(&own.stdout).trim().parse().unwrap();
    let ask = |server: &Server| server.get("/?func=tet&base=nice&height=1&format=text");
    assert_eq!(ask(&Server::start(&[])).body, format!("{own}\n0\n"));
    // The niceness is the one asked for, unless the server is nicer already.
    let nicer = Server::start(&[("TET_WEB_NICE", "7")]);
    assert_eq!(ask(&nicer).body, format!("{}\n0\n", own.max(7)));
}

#[test]
fn settings_that_cannot_work_end_the_server_at_once() {
    let directory = fixture_directory();
    for (settings, reason) in [
        (
            &[("TET_WEB_TET", "/nonexistent/tet")][..],
            "could not be run",
        ),
        (&[("TET_WEB_JOBS", "0")], "TET_WEB_JOBS must be at least 1"),
        (&[("TET_WEB_NICE", "21")], "TET_WEB_NICE is at most 20"),
        (
            &[("TET_WEB_TIMEOUT", "soon")],
            "TET_WEB_TIMEOUT=\"soon\" is not a valid value",
        ),
        (&[("TET_WEB_LISTEN", "nowhere")], "cannot listen on nowhere"),
    ] {
        let output = command(&directory, settings).output().unwrap();
        assert_eq!(output.status.code(), Some(1), "{settings:?}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.starts_with("tet-web: ") && stderr.contains(reason),
            "{stderr}"
        );
    }
    let usage = command(&directory, &[]).arg("--help").output().unwrap();
    assert_eq!(usage.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&usage.stdout).starts_with("Usage: tet-web"));
    let _ = std::fs::remove_dir_all(&directory);
}
