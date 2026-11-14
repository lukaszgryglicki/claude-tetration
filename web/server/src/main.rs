//! `tet-web`: a page and an HTTP API in front of the `tet` binary.
//!
//! `GET /` is the page. `GET /?func=tet&base=2&height=0.5` (or `/api?...`)
//! runs `tet 20 2 0.5` and returns what it printed. Every computation is a
//! child process with a deadline; finished results are kept in memory until
//! the server stops. The server speaks plain HTTP on a local address and is
//! meant to stand behind a reverse proxy that faces the network.

mod api;
mod cache;
mod http;
mod run;

use std::net::{TcpListener, TcpStream};
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use api::{Format, Job, Limits, Outcome};
use cache::{Cache, Source};
use http::{Request, Response, Unread};
use run::{Runner, Slots};

const INDEX: &str = include_str!("../../static/index.html");
const SCRIPT: &str = include_str!("../../static/app.js");
const STYLE: &str = include_str!("../../static/style.css");
const ICON: &str = include_str!("../../static/favicon.svg");

/// The policy of the page: its own script and style, requests to its own
/// origin, nothing else.
const POLICY: &str = "default-src 'none'; script-src 'self'; style-src 'self'; \
                      connect-src 'self'; img-src 'self'; base-uri 'none'; form-action 'none'; \
                      frame-ancestors 'none'";

/// The connections served at a time; each is one thread.
const CONNECTIONS: usize = 256;

const USAGE: &str = "\
Usage: tet-web

  Serves a web page and an HTTP API for the tet binary. There are no
  arguments; the settings are environment variables:

    TET_WEB_LISTEN      address to listen on                 127.0.0.1:8080
    TET_WEB_TET         the tet program                      tet
    TET_WEB_MAX_PREC    largest precision in digits          1000
    TET_WEB_TIMEOUT     seconds one computation may take     60
    TET_WEB_JOBS        computations at a time               2
    TET_WEB_QUEUE       requests that may wait for a slot    16
    TET_WEB_THREADS     threads of one tet (its TET_MT)      serial
    TET_WEB_MEMORY_MB   address space of one tet, in MiB     unlimited
    TET_WEB_NICE        niceness of one tet, up to 20        that of the server
    TET_WEB_CACHE_MB    memory for kept results, in MiB      64
    TET_WEB_REVISION    text shown as the version            (none)

  GET /                   the page
  GET /?func=...&...      the API; also at /api
  GET /help               the usage text of tet
  GET /stats              counters of the server
  GET /healthz            ok
";

struct Server {
    limits: Limits,
    runner: Runner,
    slots: Slots,
    cache: Cache,
    /// How long a request may wait for a slot or for a running computation
    /// before it is told that the server is busy.
    patience: Duration,
    help: String,
    revision: String,
    started: Instant,
    requests: AtomicU64,
    connections: AtomicUsize,
}

fn setting<T: std::str::FromStr>(name: &str, default: T) -> Result<T, String> {
    match std::env::var(name) {
        Ok(text) if !text.trim().is_empty() => text
            .trim()
            .parse()
            .map_err(|_| format!("{name}={text:?} is not a valid value")),
        Ok(_) | Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("{name}: {error}")),
    }
}

fn optional(name: &str) -> Result<Option<u64>, String> {
    Ok(Some(setting(name, 0u64)?).filter(|value| *value > 0))
}

fn positive(name: &str, default: u64) -> Result<u64, String> {
    match setting(name, default)? {
        0 => Err(format!("{name} must be at least 1")),
        value => Ok(value),
    }
}

impl Server {
    fn from_environment() -> Result<(Server, String), String> {
        let listen = setting("TET_WEB_LISTEN", "127.0.0.1:8080".to_string())?;
        let timeout = Duration::from_secs(positive("TET_WEB_TIMEOUT", 60)?);
        let megabytes = |name: &str| -> Result<Option<u64>, String> {
            optional(name)?
                .map(|value| {
                    value
                        .checked_mul(1 << 20)
                        .ok_or_else(|| format!("{name} is too large"))
                })
                .transpose()
        };
        let memory = megabytes("TET_WEB_MEMORY_MB")?;
        if memory.is_some() && !run::memory_supported() {
            return Err("TET_WEB_MEMORY_MB is not supported on this system".into());
        }
        let nice = optional("TET_WEB_NICE")?
            .map(|nice| {
                i32::try_from(nice)
                    .ok()
                    .filter(|nice| *nice <= 20)
                    .ok_or("TET_WEB_NICE is at most 20")
            })
            .transpose()?;
        let runner = Runner {
            tet: setting("TET_WEB_TET", "tet".to_string())?,
            timeout,
            threads: optional("TET_WEB_THREADS")?,
            memory,
            nice,
        };
        // The usage text of tet: it shows that the program runs, and it is
        // the syntax help of the page.
        let help = std::process::Command::new(&runner.tet)
            .arg("--help")
            .output()
            .map_err(|error| format!("{} could not be run: {error}", runner.tet))?;
        if !help.status.success() || help.stdout.is_empty() {
            return Err(format!("{} --help failed ({})", runner.tet, help.status));
        }
        let jobs = usize::try_from(positive("TET_WEB_JOBS", 2)?)
            .map_err(|_| "TET_WEB_JOBS is too large")?;
        let queue = usize::try_from(setting("TET_WEB_QUEUE", 16u64)?)
            .map_err(|_| "TET_WEB_QUEUE is too large")?;
        let cache = megabytes("TET_WEB_CACHE_MB")?.unwrap_or(64 << 20);
        let server = Server {
            limits: Limits {
                max_prec: positive("TET_WEB_MAX_PREC", 1000)?,
                max_argument: 4000,
            },
            runner,
            slots: Slots::new(jobs, queue),
            cache: Cache::new(usize::try_from(cache).map_err(|_| "TET_WEB_CACHE_MB is too large")?),
            patience: timeout,
            help: String::from_utf8_lossy(&help.stdout).into_owned(),
            revision: setting("TET_WEB_REVISION", String::new())?,
            started: Instant::now(),
            requests: AtomicU64::new(0),
            connections: AtomicUsize::new(0),
        };
        Ok((server, listen))
    }

    /// The reply to a query of the API, and what the log says about it.
    fn computed(&self, query: &str, format_hint: Format) -> (Response, String) {
        let parsed = http::parse_query(query)
            .map_err(|message| api::Rejected {
                status: 400,
                message,
            })
            .and_then(|query| Job::parse(&query, &self.limits));
        let (job, format) = match parsed {
            Ok(parsed) => parsed,
            Err(rejected) => {
                let body = api::rendered_rejection(&rejected, format_hint);
                let note = format!("rejected: {}", rejected.message);
                return (reply(rejected.status, format_hint, body), note);
            }
        };
        let deadline = Instant::now() + self.patience + self.runner.timeout;
        let (outcome, source) = self.cache.outcome(&job.key(), deadline, || {
            match self.slots.take(Instant::now() + self.patience) {
                Some(_slot) => self.runner.run(&job),
                None => Outcome::Busy,
            }
        });
        let body = api::rendered(&job, &outcome, source == Source::Hit, format);
        let response = reply(outcome.status(), format, body).with("X-Cache", source.header());
        let seconds = match &*outcome {
            Outcome::Value { seconds, .. } | Outcome::Refused { seconds, .. } => {
                format!(" {seconds:.3}s")
            }
            _ => String::new(),
        };
        (
            response,
            format!("{}{seconds} {}", source.header(), job.command()),
        )
    }

    fn stats(&self) -> String {
        let counts = self.cache.counts();
        let (running, waiting) = self.slots.load();
        format!(
            "{{\"revision\":{},\"uptime_seconds\":{},\"requests\":{},\"running\":{running},\
             \"waiting\":{waiting},\"cache\":{{\"entries\":{},\"bytes\":{},\"hits\":{},\"misses\":{},\
             \"waits\":{},\"dropped\":{}}},\"limits\":{{\"max_prec\":{},\"timeout_seconds\":{}}}}}\n",
            api::json_string(&self.revision),
            self.started.elapsed().as_secs(),
            self.requests.load(Ordering::Relaxed),
            counts.entries,
            counts.bytes,
            counts.hits,
            counts.misses,
            counts.waits,
            counts.dropped,
            self.limits.max_prec,
            self.runner.timeout.as_secs(),
        )
    }

    fn respond(&self, request: &Request) -> (Response, String) {
        if request.method != "GET" && request.method != "HEAD" {
            let response = Response::text(405, "only GET and HEAD are served here\n")
                .with("Allow", "GET, HEAD");
            return (response, String::new());
        }
        let query = request.query.as_deref().unwrap_or("");
        // The format of the reply to a request that cannot be parsed.
        let hint = if query.split('&').any(|piece| piece == "format=text") {
            Format::Text
        } else {
            Format::Json
        };
        let page = |body: &'static str, kind: &'static str| {
            Response::new(200, kind, body)
                .with("Cache-Control", "no-cache")
                .with("Content-Security-Policy", POLICY)
                .with("Referrer-Policy", "no-referrer")
        };
        // A query with a parameter of the API is a request to the API, also
        // when `func` is the one that is missing.
        let asks = query.split('&').any(|piece| {
            let name = piece.split('=').next().unwrap_or(piece);
            let number = name.trim_end_matches("_re").trim_end_matches("_im");
            ["func", "prec", "format", "base", "height", "value"].contains(&number)
        });
        let response = match request.path.as_str() {
            "/api" => return self.computed(query, hint),
            "/" if asks => return self.computed(query, hint),
            "/" => page(INDEX, "text/html; charset=utf-8"),
            "/app.js" => page(SCRIPT, "text/javascript; charset=utf-8"),
            "/style.css" => page(STYLE, "text/css; charset=utf-8"),
            "/favicon.svg" | "/favicon.ico" => page(ICON, "image/svg+xml"),
            "/help" => Response::text(200, self.help.clone()),
            "/healthz" => Response::text(200, "ok\n"),
            "/stats" => reply(200, Format::Json, self.stats()),
            "/robots.txt" => Response::text(200, "User-agent: *\nDisallow: /api\nDisallow: /*?\n"),
            _ => Response::text(404, "there is nothing here; the page is at /\n"),
        };
        (response, String::new())
    }

    fn serve(&self, mut stream: TcpStream, peer: String) {
        let started = Instant::now();
        let request = match http::read_request(&mut stream) {
            Ok(request) => request,
            Err(Unread::Closed) => return,
            Err(Unread::Status(status, reason)) => {
                let _ = http::write_response(
                    &mut stream,
                    &Response::text(status, format!("{reason}\n")),
                    true,
                );
                http::finish_unread(&mut stream);
                return;
            }
        };
        self.requests.fetch_add(1, Ordering::Relaxed);
        let (response, note) = self.respond(&request);
        let _ = http::write_response(&mut stream, &response, request.method != "HEAD");
        // The proxy on this machine names the client it speaks for.
        let client = request
            .header("x-forwarded-for")
            .and_then(|list| list.split(',').next())
            .map_or(peer, |first| first.trim().to_string());
        let target: String = match &request.query {
            Some(query) => format!("{}?{query}", request.path),
            None => request.path.clone(),
        };
        println!(
            "{} {client} {} {} -> {} {:.3}s {note}",
            timestamp(SystemTime::now()),
            request.method,
            target.chars().take(300).collect::<String>(),
            response.status,
            started.elapsed().as_secs_f64(),
        );
    }
}

/// A reply of the API: never stored by a browser or a proxy, so that the
/// only cache is the one of this server, and readable from any page.
fn reply(status: u16, format: Format, body: String) -> Response {
    let kind = match format {
        Format::Json => "application/json; charset=utf-8",
        Format::Text => "text/plain; charset=utf-8",
    };
    Response::new(status, kind, body)
        .with("Cache-Control", "no-store")
        .with("Access-Control-Allow-Origin", "*")
}

/// The time as `2026-11-14T08:15:00Z`.
fn timestamp(time: SystemTime) -> String {
    let seconds = time
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    let (days, rest) = (seconds / 86_400, seconds % 86_400);
    // The civil date of a day number, after Howard Hinnant.
    let shifted = days + 719_468;
    let (era, day_of_era) = (shifted / 146_097, shifted % 146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + u64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rest / 3_600,
        rest % 3_600 / 60,
        rest % 60
    )
}

fn main() -> ExitCode {
    if std::env::args().len() > 1 {
        print!("{USAGE}");
        return ExitCode::from(2);
    }
    let (server, listen) = match Server::from_environment() {
        Ok(found) => found,
        Err(reason) => {
            eprintln!("tet-web: {reason}");
            return ExitCode::FAILURE;
        }
    };
    let listener = match TcpListener::bind(&listen) {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("tet-web: cannot listen on {listen}: {error}");
            return ExitCode::FAILURE;
        }
    };
    let address = listener
        .local_addr()
        .map_or(listen, |address| address.to_string());
    println!(
        "{} tet-web listening on {address}: tet={} max_prec={} timeout={}s revision={:?}",
        timestamp(SystemTime::now()),
        server.runner.tet,
        server.limits.max_prec,
        server.runner.timeout.as_secs(),
        server.revision,
    );
    let server = Arc::new(server);
    for stream in listener.incoming() {
        let Ok(stream) = stream else {
            // Out of descriptors, for instance: let the load fall.
            std::thread::sleep(Duration::from_millis(50));
            continue;
        };
        let peer = stream
            .peer_addr()
            .map_or_else(|_| "?".to_string(), |address| address.ip().to_string());
        if server.connections.fetch_add(1, Ordering::SeqCst) >= CONNECTIONS {
            server.connections.fetch_sub(1, Ordering::SeqCst);
            let mut stream = stream;
            let busy = Response::text(503, "too many connections\n");
            let _ = http::write_response(&mut stream, &busy, true);
            continue;
        }
        let shared = server.clone();
        let spawned = std::thread::Builder::new()
            .name("connection".into())
            .stack_size(512 * 1024)
            .spawn(move || {
                // The count goes down however the thread ends.
                struct Open(Arc<Server>);
                impl Drop for Open {
                    fn drop(&mut self) {
                        self.0.connections.fetch_sub(1, Ordering::SeqCst);
                    }
                }
                let open = Open(shared);
                open.0.serve(stream, peer);
            });
        if spawned.is_err() {
            server.connections.fetch_sub(1, Ordering::SeqCst);
        }
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_are_utc_dates() {
        for (seconds, text) in [
            (0, "1970-01-01T00:00:00Z"),
            (951_782_399, "2000-02-28T23:59:59Z"),
            (951_782_400, "2000-02-29T00:00:00Z"),
            (1_794_643_200, "2026-11-14T08:00:00Z"),
            (4_102_444_799, "2099-12-31T23:59:59Z"),
        ] {
            assert_eq!(timestamp(UNIX_EPOCH + Duration::from_secs(seconds)), text);
        }
    }

    #[test]
    fn the_page_asks_only_its_own_origin() {
        // Nothing is loaded from elsewhere, and nothing is inline: the
        // policy sent with the page allows neither.
        for file in [INDEX, SCRIPT, STYLE] {
            for loaded in [
                "src=\"http",
                "src=\"//",
                "url(",
                "@import",
                "<style",
                "style=\"",
            ] {
                assert!(!file.contains(loaded), "{loaded}");
            }
        }
        assert!(INDEX.contains("<script src=\"app.js\" defer></script>"));
        assert!(INDEX.contains("<link rel=\"stylesheet\" href=\"style.css\">"));
        assert_eq!(INDEX.matches("<script").count(), 1);
        assert!(!INDEX.contains("onclick=") && !SCRIPT.contains("innerHTML"));
        assert!(ICON.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""));
    }
}
