//! The server as a process: its page, its API, its cache, its settings and
//! its log, with a shell script in the place of `tet`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
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
    longer) sleep 3; printf '%s\n0\n' "$2" ;;
    refuse) echo 'a line of progress' >&2
        echo 'error: invalid number "refuse": unknown name refuse' >&2; exit 1 ;;
    usage) exit 2 ;;
    garbage) echo 'one line only' ;;
    crash) kill -KILL $$ ;;
    big) head -c "$2" /dev/zero | tr '\0' 7; printf '\n0\n' ;;
    memory) printf '%s\n0\n' "$(ulimit -v)" ;;
    nice) printf '%s\n0\n' "$(ps -o nice= -p $$ | tr -d ' ')" ;;
    *) printf '%s|%s|%s\n%s\n' "$mode" "$prec" "$*" "${TET_MT-serial}" ;;
esac
"#;

static SERVERS: AtomicUsize = AtomicUsize::new(0);

struct Server {
    directory: PathBuf,
    child: Child,
    address: String,
    /// The lines that the server has written to its log, the first one
    /// included.
    log: Arc<Mutex<Vec<String>>>,
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
        Self::start_in(fixture_directory(), settings)
    }

    /// Starts the server with the fixtures of a directory, where a test
    /// may have put a settings file.
    fn start_in(directory: PathBuf, settings: &[(&str, &str)]) -> Server {
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
        let log = Arc::new(Mutex::new(vec![first]));
        let written = log.clone();
        std::thread::spawn(move || {
            for line in lines.map_while(Result::ok) {
                written.lock().unwrap().push(line);
            }
        });
        Server {
            directory,
            child,
            address,
            log,
        }
    }

    /// A request with more lines in its head, each ending in `\r\n`.
    fn ask(&self, method: &str, target: &str, headers: &str) -> Reply {
        let mut stream = TcpStream::connect(&self.address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(60)))
            .unwrap();
        write!(
            stream,
            "{method} {target} HTTP/1.1\r\nHost: test\r\n{headers}\r\n"
        )
        .unwrap();
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

    fn request(&self, method: &str, target: &str) -> Reply {
        self.ask(method, target, "")
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

    /// The lines of the log that have `part` in them.
    fn lines_with(&self, part: &str) -> Vec<String> {
        let log = self.log.lock().unwrap();
        log.iter()
            .filter(|line| line.contains(part))
            .cloned()
            .collect()
    }

    /// The line of the log with `part` in it; the server writes it after
    /// its reply, so that it may come a moment later.
    fn logged(&self, part: &str) -> String {
        let started = Instant::now();
        loop {
            if let Some(line) = self.lines_with(part).pop() {
                return line;
            }
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "no {part:?} in the log: {:#?}",
                self.log.lock().unwrap()
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Asks until the reply is as wanted, for twenty seconds at most.
    fn until(&self, target: &str, wanted: impl Fn(&Reply) -> bool) -> Reply {
        let started = Instant::now();
        loop {
            let reply = self.get(target);
            if wanted(&reply) {
                return reply;
            }
            assert!(
                started.elapsed() < Duration::from_secs(20),
                "{target}: {} {}",
                reply.status,
                reply.body
            );
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

/// Whether the process that the script `slow` became is still there.
fn slow_process_runs(server: &Server) -> bool {
    let pid = std::fs::read_to_string(server.directory.join("tet.pid")).unwrap();
    Command::new("kill")
        .args(["-0", pid.trim()])
        .stderr(Stdio::null())
        .status()
        .unwrap()
        .success()
}

const BUSY: &str = "all slots are busy calculating; try again later";

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
    // The page is a form that is sent to this server.
    assert!(policy.contains("form-action 'self'"));
    assert_eq!(page.header("Cache-Control"), Some("no-cache"));
    assert_eq!(page.header("X-Content-Type-Options"), Some("nosniff"));
    assert_eq!(page.header("Connection"), Some("close"));
    // As many calculations at a time as there are processors, four at least.
    let jobs = std::thread::available_parallelism().unwrap().get().max(4);
    for part in [
        "<form id=\"form\" method=\"get\" action=\"/\" novalidate autocomplete=\"off\">"
            .to_string(),
        "name=\"prec\" type=\"number\" min=\"1\" step=\"1\" value=\"20\"".to_string(),
        "<input type=\"radio\" name=\"func\" value=\"tet\" checked>".to_string(),
        "name=\"base\" value=\"2\"".to_string(),
        "name=\"height\" value=\"0.5\"".to_string(),
        "<section id=\"result\"></section>".to_string(),
        "not ready after 2 minutes continues\n    there, for up to 3 hours".to_string(),
        format!("At most {jobs} calculations run\n    at a time"),
        ", revision abc1234.".to_string(),
        // The documentation is a link away.
        "/blob/main/web/README.md\">this page and its API</a>".to_string(),
        "<a href=\"help\">the syntax of the arguments</a>".to_string(),
        // So are the thread about the page and the programs that plot with tet.
        "<a href=\"https://tetrationforum.org/showthread.php?tid=1830\">Tetration online calculator</a>"
            .to_string(),
        "<a href=\"https://github.com/lukaszgryglicki/tet-line\">tet-line</a>".to_string(),
        "<a href=\"https://github.com/lukaszgryglicki/rust-opengl-complex-drawer\">rust-opengl-complex-drawer</a>"
            .to_string(),
    ] {
        assert!(page.body.contains(&part), "{part}\n{}", page.body);
    }
    // A query without a parameter of the API is still the page.
    assert_eq!(server.get("/?utm_source=x").body, page.body);
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
        (
            "/status",
            "text/plain; charset=utf-8",
            "tet-web abc1234: up ",
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
    assert!(server.get("/stats").body.ends_with(&format!(
        ",\"limits\":{{\"wait_seconds\":120,\"timeout_seconds\":10800,\"jobs\":{jobs}}}}}\n"
    )));
    let status = server.get("/status").body;
    assert!(
        status.ends_with(&format!(
            "\ncalculations: 0 running, at most {jobs} at a time\n\
             cache: kept 0 (0 bytes), hits 0, misses 0, waits 0, dropped 0\n\
             limits: a request waits 2 minutes, a calculation may take 3 hours\n"
        )),
        "{status}"
    );
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
fn any_precision_goes_to_tet_and_a_long_result_comes_back_whole() {
    let server = Server::start(&[]);
    // The server has no largest precision of its own.
    let many = server.get("/?prec=123456789&func=tet&base=2&height=3&format=text");
    assert_eq!(
        (many.status, many.body.as_str()),
        (200, "tet|123456789|2 3\nserial\n")
    );
    // A result of two hundred thousand digits.
    let digits = "7".repeat(200_000);
    let long = server.get("/?func=tet&base=big&height=200000");
    assert_eq!(long.status, 200);
    assert!(long
        .body
        .contains(&format!("\"re\":\"{digits}\",\"im\":\"0\"")));
    let text = server.get("/?func=tet&base=big&height=200000&format=text");
    assert_eq!(text.header("X-Cache"), Some("HIT"));
    assert_eq!(text.body, format!("{digits}\n0\n"));
    // The page shows it as well.
    let page = server.get("/?format=html&func=tet&numbers=2&base=big&height=200000");
    assert!(page.body.contains(&format!("<dd id=\"re\">{digits}</dd>")));
    // The log has its beginning and its length.
    let line = server.logged("tet 20 big 200000 = ");
    assert!(
        line.ends_with(&format!(
            "s tet 20 big 200000 = {}...(200000 characters) 0",
            "7".repeat(100)
        )),
        "{line}"
    );
    assert!(line.len() < 400, "{line}");
    // What is longer than the server keeps is said to be so, and not kept.
    for _ in 0..2 {
        let more = server.get("/?func=tet&base=big&height=17000000&format=text");
        assert_eq!(more.header("X-Cache"), Some("MISS"));
        assert_eq!(
            (more.status, more.body.as_str()),
            (
                500,
                "error: the result is longer than the 16 MiB that this server returns\n"
            )
        );
    }
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
            "prec is the number of digits, a whole number from 1 on, not \\\"0\\\"",
        ),
        (
            "/?func=tet&base=2&height=3&prec=1e3",
            "prec is the number of digits",
        ),
        (
            "/?func=tet&base=2&height=3&format=xml",
            "format is json, text or html, not \\\"xml\\\"",
        ),
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
fn a_calculation_ends_at_the_time_limit_and_that_is_kept() {
    let server = Server::start(&[("TET_WEB_TIMEOUT", "1")]);
    let started = Instant::now();
    let reply = server.get("/?func=tet&base=slow&height=1");
    assert!(started.elapsed() >= Duration::from_secs(1));
    assert!(started.elapsed() < Duration::from_secs(10));
    assert_eq!((reply.status, reply.header("X-Cache")), (504, Some("MISS")));
    assert!(reply
        .body
        .contains("\"error\":\"no result within 1 s, the limit of this server"));
    // The process that was computing is gone.
    assert!(!slow_process_runs(&server));
    // It would be stopped again after as long a time: the same request is
    // answered at once, and nothing runs for it.
    let started = Instant::now();
    let again = server.get("/?func=tet&base=slow&height=1&format=text");
    assert!(started.elapsed() < Duration::from_millis(500));
    assert_eq!((again.status, again.header("X-Cache")), (504, Some("HIT")));
    assert!(again.body.starts_with("error: no result within 1 s, "));
    assert_eq!(server.calls().len(), 1);
}

#[test]
fn a_calculation_that_takes_longer_continues_and_is_found_again() {
    let server = Server::start(&[("TET_WEB_WAIT", "1")]);
    let target = "/?func=tet&base=longer&height=7";
    // The request waits one of the three seconds that the calculation takes.
    let started = Instant::now();
    let first = server.get(target);
    let waited = started.elapsed();
    assert!(waited >= Duration::from_secs(1) && waited < Duration::from_millis(2500));
    assert_eq!((first.status, first.header("X-Cache")), (202, Some("MISS")));
    assert_eq!(first.header("Cache-Control"), Some("no-store"));
    assert!(
        first.body.starts_with(
            "{\"func\":\"tet\",\"prec\":20,\"base\":\"longer\",\"height\":\"7\",\"running\":true,\
             \"message\":\"the calculation is taking longer than 1 s and continues on this server: \
             ask the same again later for the result; it may take up to 3 hours\",\"status\":202,\
             \"running_seconds\":"
        ),
        "{}",
        first.body
    );
    assert!(first
        .body
        .ends_with(",\"limit_seconds\":10800,\"cached\":false,\"command\":\"tet 20 longer 7\"}\n"));
    // The same request again is answered at once: the calculation is running.
    let asked = Instant::now();
    let second = server.get(&format!("{target}&format=text"));
    assert!(asked.elapsed() < Duration::from_millis(500));
    assert_eq!(
        (second.status, second.header("X-Cache")),
        (202, Some("WAIT"))
    );
    assert!(
        second
            .body
            .starts_with("running: this calculation is already running, for ")
            && second.body.ends_with(
                " now: allow it up to 3 hours to complete and ask the same again for the result\n"
            ),
        "{}",
        second.body
    );
    assert!(server.get("/stats").body.contains("\"running\":1,"));
    assert!(server
        .get("/status")
        .body
        .contains("\ncalculations: 1 running, at most "));
    // When it has ended, its result is there for every request.
    let result = server.until(target, |reply| reply.status != 202);
    assert_eq!(
        (result.status, result.header("X-Cache")),
        (200, Some("HIT"))
    );
    assert!(
        result
            .body
            .contains("\"result\":\"value\",\"re\":\"7\",\"im\":\"0\",\"seconds\":3."),
        "{}",
        result.body
    );
    assert!(result.body.contains(",\"cached\":true,"));
    assert_eq!(server.calls(), ["-q 20 longer 7"]);
    // No request waited for its end: the log has it, with the result.
    let done = server.logged(" - DONE ");
    assert!(
        done.contains(" - DONE tet 20 longer 7 -> 200 3.") && done.ends_with("s = 7 0"),
        "{done}"
    );
    assert_eq!(server.lines_with(" - DONE ").len(), 1);
    // The request that waited is in the log, and so is the one that got the
    // result. Those that only found the calculation running are not: a
    // page asks like that every few seconds.
    let found = server.logged(" HIT ");
    assert!(
        found.contains(&format!(" 127.0.0.1 GET {target} -> 200 0.0"))
            && found.contains("s HIT 3.")
            && found.ends_with("s tet 20 longer 7 = 7 0"),
        "{found}"
    );
    let asked = server.lines_with(" GET /?func=tet&base=longer");
    assert_eq!(asked.len(), 2, "{asked:#?}");
    assert!(
        asked[0].contains(&format!(" 127.0.0.1 GET {target} -> 202 1."))
            && asked[0].ends_with("s MISS tet 20 longer 7: still running after 1 s"),
        "{}",
        asked[0]
    );
}

#[test]
fn a_calculation_runs_to_its_end_without_its_client() {
    let server = Server::start(&[]);
    let target = "/?func=tet&base=wait&height=5&format=text";
    // The client hangs up without reading the reply.
    let mut stream = TcpStream::connect(&server.address).unwrap();
    write!(stream, "GET {target} HTTP/1.1\r\nHost: test\r\n\r\n").unwrap();
    drop(stream);
    std::thread::sleep(Duration::from_millis(1500));
    let found = server.get(target);
    assert_eq!((found.status, found.body.as_str()), (200, "5\n0\n"));
    assert_eq!(server.calls(), ["-q 20 wait 5"]);
}

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
#[test]
fn the_calculations_end_with_the_server() {
    let mut server = Server::start(&[("TET_WEB_WAIT", "1")]);
    assert_eq!(server.get("/?func=tet&base=slow&height=1").status, 202);
    assert!(slow_process_runs(&server));
    server.child.kill().unwrap();
    server.child.wait().unwrap();
    let ended = Instant::now();
    while slow_process_runs(&server) && ended.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(!slow_process_runs(&server), "tet outlived the server");
}

#[test]
fn when_all_slots_are_taken_a_new_calculation_is_turned_away_at_once() {
    let server = Server::start(&[("TET_WEB_JOBS", "2")]);
    let ask =
        |height: u32| server.get(&format!("/?func=tet&base=wait&height={height}&format=text"));
    std::thread::scope(|scope| {
        let (first, second) = (scope.spawn(|| ask(1)), scope.spawn(|| ask(2)));
        std::thread::sleep(Duration::from_millis(400));
        assert!(server
            .get("/status")
            .body
            .contains("\ncalculations: 2 running, at most 2 at a time\n"));
        // There is no queue: the third one is told so without any wait.
        let started = Instant::now();
        let third = ask(3);
        assert!(started.elapsed() < Duration::from_millis(400));
        assert_eq!((third.status, third.header("X-Cache")), (503, Some("MISS")));
        assert_eq!(third.body, format!("error: {BUSY}\n"));
        let json = server.get("/?func=tet&base=wait&height=3");
        assert_eq!(json.status, 503);
        assert!(
            json.body.contains(&format!(
                ",\"error\":\"{BUSY}\",\"status\":503,\"cached\":false,"
            )),
            "{}",
            json.body
        );
        // The page says the same.
        let page = server.get("/?format=html&func=tet&numbers=2&base=wait&height=3");
        assert_eq!(page.status, 200);
        assert!(page.body.contains(&format!(
            "<h2 id=\"result-title\">No result</h2>\n  <p id=\"error\" class=\"error\">{BUSY} \
             (HTTP 503)</p>"
        )));
        for running in [first, second] {
            assert_eq!(running.join().unwrap().status, 200);
        }
    });
    assert_eq!(server.calls().len(), 2);
    let line = server.logged(" -> 503 ");
    assert!(
        line.ends_with(&format!(" MISS tet 20 wait 3: {BUSY}")),
        "{line}"
    );
    // Being turned away is not kept: the same request now runs.
    assert_eq!(ask(3).body, "3\n0\n");
    assert_eq!(server.calls().len(), 3);
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

#[test]
fn the_cache_is_listed_for_this_machine_when_no_proxy_passed_the_request_on() {
    let server = Server::start(&[]);
    let empty = server.get("/cache");
    assert_eq!((empty.status, empty.body.as_str()), (200, ""));
    assert_eq!(
        empty.header("Content-Type"),
        Some("text/plain; charset=utf-8")
    );
    assert_eq!(empty.header("Cache-Control"), Some("no-store"));
    assert_eq!(server.get("/?func=slog&base=wait&value=65536").status, 200);
    assert_eq!(server.get("/?func=tet&base=refuse&height=1").status, 422);
    let parts = "/?prec=7&func=ssrt&value_re=1.5&value_im=-2&height_re=pi%20%2B%201&height_im=0";
    assert_eq!(server.get(parts).status, 200);
    // The first one again, under its other name: it is now the one used last.
    let again = server.get("/?func=ate&base=wait&value=65536");
    assert_eq!(again.header("X-Cache"), Some("HIT"));
    // A calculation runs on when its client has gone.
    let mut stream = TcpStream::connect(&server.address).unwrap();
    let slow = "/?func=tet&base=slow&height=0.5";
    write!(stream, "GET {slow} HTTP/1.1\r\nHost: test\r\n\r\n").unwrap();
    drop(stream);
    let listed = server.until("/cache", |reply| reply.body.contains(" running="));
    let lines: Vec<&str> = listed.body.lines().collect();
    assert_eq!(
        lines[..3],
        [
            "f=tet prec=20 base=refuse height=1 \
             error='invalid number \"refuse\": unknown name refuse'",
            "f=ssrt prec=7 base_re='--ssrt|7|1.5 -2 pi + 1 0' base_im=serial \
             height_re='pi + 1' height_im=0 value_re=1.5 value_im=-2",
            "f=ate prec=20 base=wait height=65536 value=65536",
        ],
        "{}",
        listed.body
    );
    let running = lines[3]
        .strip_prefix("f=tet prec=20 base=slow height=0.5 running=")
        .and_then(|rest| rest.strip_suffix('s'))
        .unwrap_or_else(|| panic!("{}", listed.body));
    assert!(running.parse::<u64>().unwrap() < 30, "{running}");
    assert_eq!(lines.len(), 4);
    assert!(listed.body.ends_with("s\n"));
    // A proxy names the client that it speaks for: what it passes on finds
    // nothing at this address, as at any other that is not served.
    let nothing = server.get("/nothing");
    for header in [
        "X-Forwarded-For: 203.0.113.7",
        "x-forwarded-proto: https",
        "X-Forwarded-Host: tet.example",
        "Forwarded: for=203.0.113.7",
        "Via: 1.1 Caddy",
        "X-Real-IP: 203.0.113.7",
    ] {
        let passed = server.ask("GET", "/cache", &format!("{header}\r\n"));
        assert_eq!(
            (passed.status, passed.body.as_str()),
            (404, nothing.body.as_str()),
            "{header}"
        );
    }
    assert_eq!(server.get("/cache/").status, 404);
    // Listing is no use of the cache and starts nothing.
    let status = server.get("/status").body;
    assert!(
        status.contains("\ncalculations: 1 running, at most ")
            && status.contains("\ncache: kept 3 (")
            && status.contains("), hits 1, misses 4, waits 0, dropped 0\n"),
        "{status}"
    );
    assert_eq!(server.calls().len(), 4);
}

#[test]
fn the_page_is_a_form_that_works_without_a_script() {
    let server = Server::start(&[]);
    // What a browser asks for when the form is sent as it starts.
    let target = "/?format=html&prec=20&func=tet&numbers=2&base=2&height=0.5";
    let page = server.get(target);
    assert_eq!((page.status, page.header("X-Cache")), (200, Some("MISS")));
    assert_eq!(
        page.header("Content-Type"),
        Some("text/html; charset=utf-8")
    );
    assert_eq!(page.header("Cache-Control"), Some("no-cache"));
    assert!(page
        .header("Content-Security-Policy")
        .is_some_and(|policy| policy.contains("form-action 'self'")));
    for part in [
        "<section id=\"result\">\n  <h2 id=\"result-title\">Result: the value</h2>\n",
        "<dt>real part</dt><dd id=\"re\">tet|20|2 0.5</dd>",
        "<dt>imaginary part</dt><dd id=\"im\">serial</dd>",
        "command: <code id=\"command\">tet 20 2 0.5</code><br>",
        "API: <a id=\"api\" href=\"/?prec=20&amp;func=tet&amp;base=2&amp;height=0.5\">",
        // The form is as it was sent.
        "<input type=\"radio\" name=\"func\" value=\"tet\" checked>",
        "name=\"base\" value=\"2\"",
        "name=\"height\" value=\"0.5\"",
    ] {
        assert!(page.body.contains(part), "{part}\n{}", page.body);
    }
    assert!(!page.body.contains("http-equiv") && !page.body.contains("data-running"));
    // The page and the API share what was computed.
    let api = server.get("/?func=tet&base=2&height=0.5&format=text");
    assert_eq!(
        (api.header("X-Cache"), api.body.as_str()),
        (Some("HIT"), "tet|20|2 0.5\nserial\n")
    );
    let again = server.get(target);
    assert_eq!(again.header("X-Cache"), Some("HIT"));
    assert!(again.body.contains(", returned from the cache</span>"));
    assert_eq!(server.calls(), ["-q 20 2 0.5"]);

    // A form writes a space as a plus sign, and a plus sign as an escape.
    let spaced = server
        .get("/?format=html&prec=5&func=ssrt&numbers=2&value=1+%2B+2&height=ate%282%2C+16%29");
    assert!(spaced.body.contains("name=\"value\" value=\"1 + 2\""));
    assert!(spaced.body.contains("name=\"height\" value=\"ate(2, 16)\""));
    assert!(spaced
        .body
        .contains("<dd id=\"re\">--ssrt|5|1 + 2 ate(2, 16)</dd>"));
    assert!(spaced.body.contains(
        "href=\"/?prec=5&amp;func=ssrt&amp;value=1%20%2B%202&amp;height=ate(2%2C%2016)\""
    ));
    assert_eq!(server.calls().len(), 2);

    // Without a script the fields are those of the choice before. The page
    // then has the fields of the new choice, and nothing is computed.
    let switched = server.get("/?format=html&prec=30&func=ate&numbers=2&base=3&height=0.25");
    for part in [
        "<input type=\"radio\" name=\"func\" value=\"ate\" checked>",
        "name=\"prec\" type=\"number\" min=\"1\" step=\"1\" value=\"30\"",
        "name=\"base\" value=\"3\"",
        "name=\"value\" value=\"65536\"",
        "<p id=\"note\" class=\"note\">The fields of the form are now those of the chosen",
    ] {
        assert!(switched.body.contains(part), "{part}\n{}", switched.body);
    }
    assert!(!switched.body.contains("name=\"height\""));
    let parts = server.get("/?format=html&func=tet&numbers=4&base=1.5%2B0.5i&height=3&show=1");
    for name in ["base_re", "base_im", "height_re", "height_im"] {
        assert!(parts.body.contains(&format!("name=\"{name}\"")), "{name}");
    }
    assert!(parts
        .body
        .contains("<input type=\"radio\" name=\"numbers\" value=\"4\" checked>"));
    let four = server
        .get("/?format=html&func=tet&numbers=4&base_re=1.5&base_im=0.5&height_re=1&height_im=0.7");
    assert!(four
        .body
        .contains("<dd id=\"re\">tet|20|1.5 0.5 1 0.7</dd>"));
    assert_eq!(server.calls().len(), 3);

    // What tet refuses and what is no request are said in the page.
    let refused = server.get("/?format=html&func=tet&numbers=2&base=refuse&height=1");
    assert_eq!(refused.status, 200);
    assert!(refused.body.contains(
        "<h2 id=\"result-title\">No result</h2>\n  <p id=\"error\" class=\"error\">invalid number \
         &quot;refuse&quot;: unknown name refuse (HTTP 422)</p>"
    ));
    assert_eq!(server.calls().len(), 4);
    for (target, message) in [
        (
            "/?format=html&prec=0&func=tet&numbers=2&base=2&height=3",
            "prec is the number of digits, a whole number from 1 on, not &quot;0&quot; (HTTP 400)",
        ),
        (
            "/?format=html&func=tet&numbers=2&base=--help&height=3",
            "base is not a number: &quot;--help&quot; (HTTP 400)",
        ),
        (
            "/?format=html&func=tet&base=%zz&height=3",
            "the query has a broken escape in &quot;base=%zz&quot; (HTTP 400)",
        ),
        (
            "/?format=html&func=tet&func=ate",
            "the parameter func is given twice (HTTP 400)",
        ),
    ] {
        let page = server.get(target);
        assert_eq!(page.status, 200, "{target}");
        assert!(page.body.starts_with("<!doctype html>"), "{target}");
        assert!(page.body.contains(message), "{target}: {}", page.body);
    }
    // What was typed stays text.
    let typed = "%3Cscript%3Ealert(1)%3C%2Fscript%3E%22%3E";
    let page = server.get(&format!(
        "/?format=html&func=tet&numbers=2&base={typed}&height=1"
    ));
    assert_eq!(page.body.matches("<script").count(), 1);
    assert!(page
        .body
        .contains("value=\"&lt;script&gt;alert(1)&lt;/script&gt;&quot;&gt;\""));
    assert_eq!(server.calls().len(), 5);
}

#[test]
fn the_page_of_a_calculation_that_still_runs_asks_again() {
    let server = Server::start(&[("TET_WEB_WAIT", "1")]);
    let target = "/?format=html&prec=20&func=tet&numbers=2&base=longer&height=7";
    let started = Instant::now();
    let first = server.get(target);
    assert!(started.elapsed() >= Duration::from_secs(1));
    assert_eq!((first.status, first.header("X-Cache")), (200, Some("MISS")));
    for part in [
        // Without scripts the browser asks again, or its user does; with
        // them the script of the page does.
        "<noscript><meta http-equiv=\"refresh\" content=\"5; url=/?format=html&amp;prec=20&amp;\
         func=tet&amp;numbers=2&amp;base=longer&amp;height=7&amp;again=1\"></noscript>\n<title>",
        "<section id=\"result\" data-running=\"5\">\n  <h2 id=\"result-title\">Still calculating</h2>",
        "<p id=\"note\" class=\"note\">The calculation is taking longer than 1 s and continues on \
         this server: ask the same again later for the result; it may take up to 3 hours. This \
         page asks again every 5 seconds; in a browser that does not, follow the link.</p>",
        "<p id=\"again\"><a href=\"/?format=html&amp;prec=20&amp;func=tet&amp;numbers=2&amp;\
         base=longer&amp;height=7&amp;again=1\">Ask again now</a></p>",
        "command: <code id=\"command\">tet 20 longer 7</code>",
    ] {
        assert!(first.body.contains(part), "{part}\n{}", first.body);
    }
    // The page asks again at the address that it was given for that, and
    // gets the next one.
    let asked = Instant::now();
    let second = server.get(&format!("{target}&again=1"));
    assert!(asked.elapsed() < Duration::from_millis(500));
    assert_eq!(second.header("X-Cache"), Some("WAIT"));
    assert!(second
        .body
        .contains("class=\"note\">This calculation is already running, for "));
    assert_eq!(
        second.body.matches("&amp;height=7&amp;again=2\"").count(),
        2
    );
    let result = server.until(target, |reply| !reply.body.contains("data-running"));
    assert!(
        result.body.contains("<dd id=\"re\">7</dd>"),
        "{}",
        result.body
    );
    assert!(!result.body.contains("http-equiv") && !result.body.contains("id=\"again\""));
    assert_eq!(server.calls(), ["-q 20 longer 7"]);
}

#[test]
fn the_log_tells_what_was_asked_and_what_was_returned() {
    let server = Server::start(&[]);
    let banner = server.logged(" tet-web listening on ");
    let jobs = std::thread::available_parallelism().unwrap().get().max(4);
    assert!(
        banner.contains(&format!(" tet-web listening on {}: tet=", server.address))
            && banner.ends_with(&format!(
                " revision=\"\"; a request waits 2 minutes, a calculation may take 3 hours, {jobs} \
                 at a time, threads of one tet: one, its memory: unlimited, its niceness: that of \
                 the server"
            )),
        "{banner}"
    );
    let target = "/?prec=20&func=ate&base=2&value=65536";
    server.get(target);
    let line = server.logged(" MISS ");
    let (time, rest) = line.split_once(' ').unwrap();
    // 2026-11-14T08:15:00Z
    assert_eq!((time.len(), &time[10..11], &time[19..]), (20, "T", "Z"));
    assert!(
        rest.starts_with(&format!("127.0.0.1 GET {target} -> 200 0."))
            && rest.contains("s MISS 0.")
            && rest.ends_with("s tet --ate 20 2 65536 = --ate|20|2 65536 serial"),
        "{line}"
    );
    // The proxy on the machine names the client that it speaks for.
    server.ask("GET", target, "X-Forwarded-For: 203.0.113.9, 10.0.0.1\r\n");
    let line = server.logged(" HIT ");
    assert!(
        line.contains(&format!(" 203.0.113.9 GET {target} -> 200 0."))
            && line.ends_with("s tet --ate 20 2 65536 = --ate|20|2 65536 serial"),
        "{line}"
    );
    // The page of a form is a request for a calculation like any other.
    server.get("/?format=html&func=ssrt&numbers=2&value=16&height=3");
    let line = server.logged(" tet --ssrt 20 16 3 = ");
    assert!(
        line.contains(" GET /?format=html&func=ssrt&numbers=2&value=16&height=3 -> 200 ")
            && line.ends_with(" = --ssrt|20|16 3 serial"),
        "{line}"
    );
    // What tet refused, and what never reached it.
    server.get("/?func=tet&base=refuse&height=1");
    let line = server.logged(" -> 422 ");
    assert!(
        line.ends_with("s tet 20 refuse 1: invalid number \"refuse\": unknown name refuse"),
        "{line}"
    );
    server.get("/?func=exp");
    let line = server.logged(" -> 400 ");
    assert!(
        line.ends_with("s rejected: func is tet, ate, slog, sroot or ssrt, not \"exp\""),
        "{line}"
    );
    // Another request has its line without more.
    server.get("/healthz");
    let line = server.logged(" GET /healthz ");
    let (_, took) = line.split_once(" GET /healthz -> 200 ").unwrap();
    let seconds = took
        .strip_suffix('s')
        .and_then(|took| took.parse::<f64>().ok());
    assert!(seconds.is_some(), "{line}");
}

#[test]
fn the_settings_file_is_read_again_when_it_changes() {
    let directory = fixture_directory();
    let file = directory.join("tet-web.conf");
    std::fs::write(&file, "# The limits.\nTET_WEB_TIMEOUT=1h\nTET_WEB_WAIT=1\n").unwrap();
    // The file is above the environment.
    let server = Server::start_in(
        directory,
        &[
            ("TET_WEB_CONFIG", file.to_str().unwrap()),
            ("TET_WEB_TIMEOUT", "50"),
            ("TET_WEB_JOBS", "3"),
        ],
    );
    let limits = |server: &Server| {
        let status = server.get("/status").body;
        let (_, limits) = status.rsplit_once("\nlimits: ").unwrap();
        limits.trim_end().to_string()
    };
    let hour = "a request waits 1 s, a calculation may take 1 hour";
    assert_eq!(limits(&server), hour);
    assert!(server
        .logged(" tet-web listening on ")
        .contains(&format!("; {hour}, 3 at a time, ")));
    let target = "/?func=tet&base=slow&height=1&format=text";
    assert_eq!(server.get(target).status, 202);
    assert!(slow_process_runs(&server));

    // A lower limit ends the calculation that runs beyond it, without a
    // restart of the server.
    std::fs::write(&file, "TET_WEB_TIMEOUT=2\nTET_WEB_WAIT=1\n").unwrap();
    let stopped = server.until(target, |reply| reply.status != 202);
    assert_eq!(
        (stopped.status, stopped.body.as_str()),
        (
            504,
            "error: no result within 2 s, the limit of this server; tet itself has no limit: \
             build it and run the command\n"
        )
    );
    assert!(!slow_process_runs(&server));
    let line = server.logged(" - SETTINGS ");
    assert!(
        line.ends_with(
            " - SETTINGS a request waits 1 s, a calculation may take 2 s, 3 at a time, threads \
             of one tet: one, its memory: unlimited, its niceness: that of the server"
        ),
        "{line}"
    );
    assert_eq!(
        limits(&server),
        "a request waits 1 s, a calculation may take 2 s"
    );
    assert!(server
        .get("/stats")
        .body
        .contains("\"limits\":{\"wait_seconds\":1,\"timeout_seconds\":2,\"jobs\":3}"));
    // What the limit stopped is kept.
    assert_eq!(server.get(target).header("X-Cache"), Some("HIT"));
    assert_eq!(server.calls().len(), 1);

    // A file that cannot be read as settings changes nothing, and the log
    // says why.
    std::fs::write(&file, "TET_WEB_TIMEOUT=soon\n").unwrap();
    let line = server.logged(" - SETTINGS not changed: ");
    assert!(
        line.contains(&format!(
            " - SETTINGS not changed: {}: TET_WEB_TIMEOUT=\"soon\" is not a time",
            file.display()
        )),
        "{line}"
    );
    assert_eq!(
        limits(&server),
        "a request waits 1 s, a calculation may take 2 s"
    );

    // With a higher limit the calculation may end: what was stopped below
    // it is asked for again.
    std::fs::write(
        &file,
        "TET_WEB_TIMEOUT=1h\nTET_WEB_WAIT=1\nTET_WEB_JOBS=5\n",
    )
    .unwrap();
    let started = Instant::now();
    while limits(&server) != hour {
        assert!(started.elapsed() < Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(server
        .get("/status")
        .body
        .contains("\ncalculations: 0 running, at most 5 at a time\n"));
    let again = server.get(target);
    assert_eq!((again.status, again.header("X-Cache")), (202, Some("MISS")));
    assert_eq!(server.calls().len(), 2);
    assert!(slow_process_runs(&server));
}

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
#[test]
fn a_child_gets_its_threads_and_its_memory_limit() {
    let server = Server::start(&[("TET_WEB_THREADS", "3"), ("TET_WEB_MEMORY_MB", "512")]);
    let threads = server.get("/?func=tet&base=2&height=3&format=text");
    assert_eq!(threads.body, "tet|20|2 3\n3\n");
    // The address space in kilobytes.
    let memory = server.get("/?func=tet&base=memory&height=1&format=text");
    assert_eq!(memory.body, "524288\n0\n");
    // Without the setting the limit is the one that the server has itself.
    let own = Command::new("sh")
        .args(["-c", "ulimit -v"])
        .output()
        .unwrap();
    let own = String::from_utf8_lossy(&own.stdout).trim().to_string();
    let free = Server::start(&[]);
    let memory = free.get("/?func=tet&base=memory&height=1&format=text");
    assert_eq!(memory.body, format!("{own}\n0\n"));
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
    let file = directory.join("bad.conf");
    std::fs::write(&file, "TET_WEB_TIMEOUT=3h\nTET_WEB_LISTEN=0.0.0.0:80\n").unwrap();
    let file = file.to_str().unwrap().to_string();
    let named = format!("{file}: line 2: TET_WEB_LISTEN is not a setting of this file");
    for (settings, reason) in [
        (
            &[("TET_WEB_TET", "/nonexistent/tet")][..],
            "could not be run",
        ),
        (&[("TET_WEB_JOBS", "0")], "TET_WEB_JOBS must be at least 1"),
        (&[("TET_WEB_WAIT", "0")], "TET_WEB_WAIT=\"0\" is not a time"),
        (&[("TET_WEB_NICE", "21")], "TET_WEB_NICE is at most 20"),
        (
            &[("TET_WEB_TIMEOUT", "soon")],
            "TET_WEB_TIMEOUT=\"soon\" is not a time: a whole number of seconds, at least 1, or \
             of minutes, hours or days as in 90m, 3h, 2d",
        ),
        (
            &[("TET_WEB_CACHE_MB", "much")],
            "TET_WEB_CACHE_MB=\"much\" is not a valid value",
        ),
        (&[("TET_WEB_CONFIG", file.as_str())], named.as_str()),
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
    let usage = String::from_utf8_lossy(&usage.stdout);
    assert!(usage.starts_with("Usage: tet-web") && usage.contains("TET_WEB_CONFIG"));
    let _ = std::fs::remove_dir_all(&directory);
}
