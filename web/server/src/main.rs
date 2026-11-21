//! `tet-web`: a page and an HTTP API in front of the `tet` binary, and of
//! `symbolic-math` next to it (`/sm`, see `sm.rs`).
//!
//! `GET /` is the page, a form that works without a script.
//! `GET /?func=tet&base=2&height=0.5` (or `/api?...`) runs `tet 20 2 0.5`
//! and returns what it printed. Every computation is a child process with a
//! time limit. It runs to its end whether or not the request that started
//! it still waits: a request that has waited long enough is told to ask
//! again, and the same request finds the computation running, or its
//! result. Finished results are kept in memory until the server stops. The
//! server speaks plain HTTP on a local address and is meant to stand behind
//! a reverse proxy that faces the network.

mod api;
mod cache;
mod http;
mod page;
mod run;
mod settings;
mod sm;

use std::net::{IpAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use api::{Format, Job, Outcome, Pending};
use cache::{Cache, Found, Listed, Source, Ticket};
use http::{Request, Response, Unread};
use page::{Answer as Replied, Asked, Form, Shown, Site};
use run::{Runner, Slots};
use settings::Shared;
use sm::Program;

const SCRIPT: &str = include_str!("../../static/app.js");
const STYLE: &str = include_str!("../../static/style.css");
const ICON: &str = include_str!("../../static/favicon.svg");

/// The policy of the page: its own script and style, requests and its form
/// to its own origin, nothing else.
const POLICY: &str = "default-src 'none'; script-src 'self'; style-src 'self'; \
                      connect-src 'self'; img-src 'self'; frame-src 'self'; base-uri 'none'; \
                      form-action 'self'; frame-ancestors 'none'";

/// The policy of the page of all examples, which the page shows in a
/// frame: its style and nothing else, in a frame of this origin alone.
const FRAMED: &str = "default-src 'none'; style-src 'self'; img-src 'self'; base-uri 'none'; \
                      form-action 'none'; frame-ancestors 'self'";

/// The connections served at a time; each is one thread.
const CONNECTIONS: usize = 256;

const USAGE: &str = "\
Usage: tet-web

  Serves a web page and an HTTP API for the tet binary. There are no
  arguments; the settings are environment variables:

    TET_WEB_LISTEN      address to listen on                 127.0.0.1:8080
    TET_WEB_TET         the tet program                      tet
    TET_WEB_SM          the symbolic-math program            symbolic-math
    TET_WEB_CACHE_MB    most memory for kept results, in MiB 1024
    TET_WEB_REVISION    text shown as the version            (none)
    TET_WEB_CONFIG      a file with the settings below       (none)

  The settings below can stand in that file as well, as lines NAME=value.
  The file is read again whenever it changes, and what it sets is in use
  at once, without a restart:

    TET_WEB_TIMEOUT     time one calculation may take        3h
    TET_WEB_WAIT        time a request waits                 120
    TET_WEB_JOBS        calculations at a time               processors, at least 4
    TET_WEB_THREADS     threads of one tet (its TET_MT)      one
    TET_WEB_MEMORY_MB   address space of one tet, in MiB     unlimited
    TET_WEB_NICE        niceness of one tet, up to 20        that of the server

  A time is a number of seconds, or of minutes, hours or days: 90m, 3h, 2d.

  GET /                   the page, a form that needs no script
  GET /?func=...&...      the API; also at /api. With format=html, as the
                          form of the page sends it: the page with the result
  GET /help               the usage text of tet
  GET /sm?expr=...&...    the API of symbolic-math: its inputs, one a line,
                          with digits, mode and domain; with format=html the
                          page with the result
  GET /sm/help            the help of symbolic-math as a page; /sm/help.txt,
                          /sm/functions, /sm/constants, /sm/units as text
  GET /sm/examples        every line of examples of that help as a link
                          that runs it on the page
  GET /sm/file/ID/NAME    a file that a run wrote; /sm/output/ID its output
  GET /stats              counters of the server, as JSON
  GET /status             the same for people
  GET /cache              the kept results and the calculations that run,
                          one a line; only for a request from this machine
                          that no proxy passed on
  GET /healthz            ok

  A calculation that is not ready when its request has waited continues,
  for as long as it may take: the reply says so (status 202), and the same
  request again finds it running, or its result. When all slots are taken
  a new calculation is turned away at once (status 503).
";

/// What a computation is: a number of `tet`, or a run of `symbolic-math`.
#[derive(Clone, Debug)]
enum Work {
    Tet(Job),
    Sm(Program),
}

impl Work {
    fn key(&self) -> String {
        match self {
            Work::Tet(job) => job.key(),
            Work::Sm(program) => program.key(),
        }
    }

    fn command(&self) -> String {
        match self {
            Work::Tet(job) => job.command(),
            Work::Sm(program) => program.command(),
        }
    }
}

struct Server {
    runner: Runner,
    /// `symbolic-math`, where it is installed.
    sm: Option<sm::Runner>,
    /// The page of all the examples of its help, and their lines and
    /// headings.
    examples: String,
    listed: (usize, usize),
    slots: Slots,
    cache: Arc<Cache>,
    settings: Arc<Shared>,
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

/// What a request finds for its job: the outcome, or a computation that
/// has not ended.
type Answer = Result<Arc<Outcome>, Pending>;

impl Server {
    fn from_environment() -> Result<(Server, String), String> {
        let listen = setting("TET_WEB_LISTEN", "127.0.0.1:8080".to_string())?;
        let environment = settings::NAMES
            .iter()
            .filter_map(|name| Some((name.to_string(), std::env::var(name).ok()?)))
            .collect();
        let file = std::env::var_os("TET_WEB_CONFIG")
            .filter(|path| !path.is_empty())
            .map(PathBuf::from);
        let settings = Arc::new(Shared::load(environment, file)?);
        let runner = Runner {
            tet: setting("TET_WEB_TET", "tet".to_string())?,
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
        let cache = setting("TET_WEB_CACHE_MB", 1024u64)?
            .checked_mul(1 << 20)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or("TET_WEB_CACHE_MB is too large")?;
        // The second program is optional: without it the page says so.
        let named = std::env::var("TET_WEB_SM").is_ok_and(|name| !name.trim().is_empty());
        let sm = match sm::Runner::new(
            setting("TET_WEB_SM", "symbolic-math".to_string())?,
            &runner.tet,
        ) {
            Ok(sm) => Some(sm),
            Err(reason) if named => return Err(reason),
            Err(_) => None,
        };
        let groups = sm
            .as_ref()
            .map(|sm| sm::examples(&sm.help))
            .unwrap_or_default();
        let server = Server {
            runner,
            sm,
            examples: page::examples_page(&groups),
            listed: (
                groups.iter().map(|group| group.examples.len()).sum(),
                groups.len(),
            ),
            slots: Slots::default(),
            cache: Arc::new(Cache::new(cache)),
            settings,
            help: String::from_utf8_lossy(&help.stdout).into_owned(),
            revision: setting("TET_WEB_REVISION", String::new())?,
            started: Instant::now(),
            requests: AtomicU64::new(0),
            connections: AtomicUsize::new(0),
        };
        Ok((server, listen))
    }

    /// What there is for a job, and how this request came to it. A job that
    /// nobody asked for before is started; the request then waits for it,
    /// as do the others, until the computation is as old as a request
    /// waits. One that comes later is answered at once.
    fn asked(self: &Arc<Self>, job: &Work) -> (Answer, Source) {
        let now = self.settings.now();
        let key = job.key();
        let (running, source) = loop {
            match self.cache.find(&key) {
                Found::Kept(outcome) => {
                    // What was stopped at a lower limit than the one of
                    // now may end today.
                    let limit = now.timeout.as_secs();
                    if matches!(*outcome, Outcome::Timeout { limit: then } if then < limit) {
                        self.cache.forget(&key, &outcome);
                        continue;
                    }
                    return (Ok(outcome), Source::Hit);
                }
                Found::Running(running) => break (running, Source::Wait),
                Found::New(ticket) => {
                    let running = ticket.running();
                    self.start(job.clone(), ticket);
                    break (running, Source::Miss);
                }
            }
        };
        let deadline = running.asked() + now.wait;
        let waited = Instant::now() < deadline;
        match running.awaited(deadline) {
            Some(outcome) => (Ok(outcome), source),
            None => {
                let pending = Pending {
                    waited,
                    age: running.asked().elapsed(),
                    wait: now.wait.as_secs(),
                    limit: self.settings.now().timeout.as_secs(),
                };
                (Err(pending), source)
            }
        }
    }

    /// Starts the computation of a job in a thread of its own, where it
    /// runs to its end whatever becomes of the requests for it.
    fn start(self: &Arc<Self>, job: Work, ticket: Ticket) {
        let server = self.clone();
        let spawned = std::thread::Builder::new()
            .name("computation".into())
            .stack_size(512 * 1024)
            .spawn(move || server.compute(&job, ticket));
        if let Err(error) = spawned {
            // The ticket went with the thread: the entry has failed.
            eprintln!("tet-web: no thread for a computation: {error}");
        }
    }

    fn compute(&self, job: &Work, ticket: Ticket) {
        let asked = ticket.running().asked();
        let outcome = match (self.slots.take(self.settings.now().jobs), job, &self.sm) {
            (None, ..) => Outcome::Busy,
            (Some(_slot), Work::Tet(job), _) => self.runner.run(job, &self.settings),
            (Some(_slot), Work::Sm(program), Some(sm)) => sm.run(program, &self.settings),
            (Some(_slot), Work::Sm(_), None) => Outcome::Failed {
                message: "symbolic-math is not installed on this server".into(),
            },
        };
        // No request waited for the end of a long computation: the log has it.
        let age = asked.elapsed();
        if age > self.settings.now().wait {
            println!(
                "{} - DONE {} -> {} {:.3}s{}",
                timestamp(SystemTime::now()),
                api::brief(&job.command(), 300),
                outcome.status(),
                age.as_secs_f64(),
                outcome.logged(),
            );
        }
        ticket.finish(outcome);
    }

    /// The reply to a query of the API, and what the log says about it;
    /// nothing, for a request that is not worth a line.
    fn computed(self: &Arc<Self>, query: &str, format_hint: Format) -> (Response, Option<String>) {
        if query.split('&').any(|piece| piece == "format=html") {
            return self.shown(query);
        }
        let refused = |rejected: api::Rejected| {
            let body = api::rendered_rejection(&rejected, format_hint);
            let note = format!("rejected: {}", rejected.message);
            (reply(rejected.status, format_hint, body), Some(note))
        };
        let query = match http::parse_query(query) {
            Ok(query) => query,
            Err(message) => {
                return refused(api::Rejected {
                    status: 400,
                    message,
                })
            }
        };
        let (job, format) = match Job::parse(&query) {
            Ok(parsed) => parsed,
            Err(rejected) => return refused(rejected),
        };
        let work = Work::Tet(job.clone());
        let (answer, source) = self.asked(&work);
        let body = match &answer {
            Ok(outcome) => api::rendered(&job, outcome, source == Source::Hit, format),
            Err(pending) => api::rendered_pending(&job, pending, format),
        };
        let status = answer.as_ref().map_or(202, |outcome| outcome.status());
        let response = reply(status, format, body).with("X-Cache", source.header());
        (response, note(&work, source, &answer))
    }

    /// The reply to a query of the API of `symbolic-math`, and what the
    /// log says about it.
    fn ran(self: &Arc<Self>, query: &str, format_hint: Format) -> (Response, Option<String>) {
        if query.split('&').any(|piece| piece == "format=html") {
            return self.ran_shown(query);
        }
        let refused = |rejected: api::Rejected| {
            let body = api::rendered_rejection(&rejected, format_hint);
            let note = format!("rejected: {}", rejected.message);
            (reply(rejected.status, format_hint, body), Some(note))
        };
        if self.sm.is_none() {
            return refused(api::Rejected {
                status: 503,
                message: "symbolic-math is not installed on this server".into(),
            });
        }
        let query = match http::parse_query(query) {
            Ok(query) => query,
            Err(message) => {
                return refused(api::Rejected {
                    status: 400,
                    message,
                })
            }
        };
        let (program, format) = match Program::parse(&query) {
            Ok(parsed) => parsed,
            Err(rejected) => return refused(rejected),
        };
        let work = Work::Sm(program.clone());
        let (answer, source) = self.asked(&work);
        let body = match &answer {
            Ok(outcome) => sm::rendered(&program, outcome, source == Source::Hit, format),
            Err(pending) => sm::rendered_pending(&program, pending, format),
        };
        let status = answer.as_ref().map_or(202, |outcome| outcome.status());
        let response = reply(status, format, body).with("X-Cache", source.header());
        (response, note(&work, source, &answer))
    }

    /// The page for a query of `symbolic-math` with `format=html`: its
    /// form as a browser sends it, and the page with the result as the
    /// reply.
    fn ran_shown(self: &Arc<Self>, query: &str) -> (Response, Option<String>) {
        let tet = Form::from_query(&[]);
        let refused = |asked: &Asked, rejected: api::Rejected| {
            let replied = Replied::Rejected {
                message: &rejected.message,
                status: rejected.status,
            };
            let note = format!("rejected: {}", rejected.message);
            (
                self.page_both(&tet, &Shown::Nothing, asked, &replied),
                Some(note),
            )
        };
        let query = match http::parse_query(&query.replace('+', "%20")) {
            Ok(query) => query,
            Err(message) => {
                let rejected = api::Rejected {
                    status: 400,
                    message,
                };
                return refused(&Asked::from_query(&[]), rejected);
            }
        };
        let asked = Asked::from_query(&query);
        if self.sm.is_none() {
            let page = self.page_both(&tet, &Shown::Nothing, &asked, &Replied::Absent);
            return (page, Some(String::new()));
        }
        if !asked.complete {
            let page = self.page_both(&tet, &Shown::Nothing, &asked, &Replied::Nothing);
            return (page, Some(String::new()));
        }
        let program = match Program::parse(&asked.api_query()) {
            Ok((program, _)) => program,
            Err(rejected) => return refused(&asked, rejected),
        };
        let work = Work::Sm(program.clone());
        let (answer, source) = self.asked(&work);
        let replied = match &answer {
            Ok(outcome) => Replied::Outcome {
                program: &program,
                outcome,
                cached: source == Source::Hit,
            },
            Err(pending) => Replied::Running {
                program: &program,
                pending,
            },
        };
        let response = self
            .page_both(&tet, &Shown::Nothing, &asked, &replied)
            .with("X-Cache", source.header());
        (response, note(&work, source, &answer))
    }

    /// The kept outcome of the run of `symbolic-math` with the name `id`.
    fn kept(&self, id: &str) -> Option<Arc<Outcome>> {
        self.cache
            .listed()
            .into_iter()
            .find_map(|(key, entry)| match entry {
                Listed::Kept(outcome) if key.starts_with("SM\n") && sm::id_of(&key) == id => {
                    Some(outcome)
                }
                _ => None,
            })
    }

    /// A file that a run wrote (`/sm/file/ID/NAME`), or what the run
    /// printed (`/sm/output/ID`).
    fn written(&self, path: &str, query: &str) -> Response {
        let gone = || {
            Response::text(
                404,
                "this run is not kept any more: calculate it again on the page\n",
            )
        };
        let attached = |response: Response, name: &str| {
            response.with(
                "Content-Disposition",
                format!("attachment; filename=\"{name}\""),
            )
        };
        if let Some(id) = path.strip_prefix("/sm/output/") {
            return match self.kept(id).as_deref() {
                Some(Outcome::Output { text, .. }) => attached(
                    Response::text(200, text.clone()).with("Cache-Control", "no-store"),
                    &format!("symbolic-math-{}.txt", &id[..id.len().min(8)]),
                ),
                _ => gone(),
            };
        }
        let Some((id, name)) = path
            .strip_prefix("/sm/file/")
            .and_then(|rest| rest.split_once('/'))
            .filter(|(_, name)| sm::plain(name))
        else {
            return gone();
        };
        let Some(outcome) = self.kept(id) else {
            return gone();
        };
        let Outcome::Output { files, .. } = &*outcome else {
            return gone();
        };
        let Some(file) = files.iter().find(|file| file.name == name) else {
            return gone();
        };
        let (kind, _) = sm::kind(name);
        let response = Response::new(200, kind, file.bytes.clone())
            .with("Cache-Control", "no-store")
            .with("Content-Security-Policy", "default-src 'none'; sandbox");
        // Only pictures and texts are shown in the page of a browser.
        let download = query.split('&').any(|piece| piece == "download=1")
            || kind == "application/octet-stream";
        match download {
            true => attached(response, name),
            false => response,
        }
    }

    /// A text of `symbolic-math`: as a page, or as it is.
    fn told(&self, path: &str) -> Response {
        let Some(sm) = &self.sm else {
            return Response::text(503, "symbolic-math is not installed on this server\n");
        };
        match path {
            "/sm/help" => Response::new(
                200,
                "text/html; charset=utf-8",
                page::text_page("--help", &sm.help),
            )
            .with("Cache-Control", "no-cache")
            .with("Content-Security-Policy", POLICY)
            .with("Referrer-Policy", "no-referrer"),
            "/sm/examples" => Response::new(200, "text/html; charset=utf-8", self.examples.clone())
                .with("Cache-Control", "no-cache")
                .with("Content-Security-Policy", FRAMED)
                .with("Referrer-Policy", "no-referrer"),
            "/sm/help.txt" => Response::text(200, sm.help.clone()),
            "/sm/functions" => Response::text(200, sm.functions.clone()),
            "/sm/constants" => Response::text(200, sm.constants.clone()),
            _ => Response::text(200, sm.units.clone()),
        }
    }

    /// The page for a query with `format=html`: the form of the page as a
    /// browser sends it, and the page with the result as the reply.
    fn shown(self: &Arc<Self>, query: &str) -> (Response, Option<String>) {
        let refused = |form: &Form, rejected: api::Rejected| {
            let shown = Shown::Rejected {
                message: &rejected.message,
                status: rejected.status,
            };
            let note = format!("rejected: {}", rejected.message);
            (self.page(form, &shown), Some(note))
        };
        // A form sends a space as a plus sign, and a plus sign as an escape.
        let query = match http::parse_query(&query.replace('+', "%20")) {
            Ok(query) => query,
            Err(message) => {
                let rejected = api::Rejected {
                    status: 400,
                    message,
                };
                return refused(&Form::from_query(&[]), rejected);
            }
        };
        let form = Form::from_query(&query);
        if !form.complete {
            return (self.page(&form, &Shown::Fields), Some(String::new()));
        }
        let job = match Job::parse(&form.api_query()) {
            Ok((job, _)) => job,
            Err(rejected) => return refused(&form, rejected),
        };
        let work = Work::Tet(job.clone());
        let (answer, source) = self.asked(&work);
        let shown = match &answer {
            Ok(outcome) => Shown::Outcome {
                job: &job,
                outcome,
                cached: source == Source::Hit,
            },
            Err(pending) => Shown::Running { job: &job, pending },
        };
        let response = self.page(&form, &shown).with("X-Cache", source.header());
        (response, note(&work, source, &answer))
    }

    fn page(&self, form: &Form, shown: &Shown) -> Response {
        let replied = match self.sm {
            Some(_) => Replied::Nothing,
            None => Replied::Absent,
        };
        self.page_both(form, shown, &Asked::from_query(&[]), &replied)
    }

    fn page_both(&self, form: &Form, shown: &Shown, asked: &Asked, replied: &Replied) -> Response {
        let now = self.settings.now();
        let site = Site {
            wait: now.wait.as_secs(),
            limit: now.timeout.as_secs(),
            jobs: now.jobs,
            revision: &self.revision,
            examples: self.listed,
        };
        let body = page::rendered_both(&site, form, shown, asked, replied);
        Response::new(200, "text/html; charset=utf-8", body)
            .with("Cache-Control", "no-cache")
            .with("Content-Security-Policy", POLICY)
            .with("Referrer-Policy", "no-referrer")
    }

    fn stats(&self) -> String {
        let counts = self.cache.counts();
        let now = self.settings.now();
        format!(
            "{{\"revision\":{},\"uptime_seconds\":{},\"requests\":{},\"running\":{},\
             \"cache\":{{\"entries\":{},\"bytes\":{},\"hits\":{},\"misses\":{},\"waits\":{},\
             \"dropped\":{}}},\"limits\":{{\"wait_seconds\":{},\"timeout_seconds\":{},\
             \"jobs\":{}}}}}\n",
            api::json_string(&self.revision),
            self.started.elapsed().as_secs(),
            self.requests.load(Ordering::Relaxed),
            self.slots.taken(),
            counts.entries,
            counts.bytes,
            counts.hits,
            counts.misses,
            counts.waits,
            counts.dropped,
            now.wait.as_secs(),
            now.timeout.as_secs(),
            now.jobs,
        )
    }

    /// The counters of `stats` as a text for people.
    fn status(&self) -> String {
        let counts = self.cache.counts();
        let now = self.settings.now();
        let revision = match self.revision.as_str() {
            "" => String::new(),
            revision => format!(" {revision}"),
        };
        format!(
            "tet-web{revision}: up {}, requests {}\n\
             calculations: {} running, at most {} at a time\n\
             cache: kept {} ({} bytes), hits {}, misses {}, waits {}, dropped {}\n\
             limits: a request waits {}, a calculation may take {}\n",
            api::span(self.started.elapsed().as_secs()),
            self.requests.load(Ordering::Relaxed),
            self.slots.taken(),
            now.jobs,
            counts.entries,
            counts.bytes,
            counts.hits,
            counts.misses,
            counts.waits,
            counts.dropped,
            api::span(now.wait.as_secs()),
            api::span(now.timeout.as_secs()),
        )
    }

    /// Every entry of the cache as a line for people.
    fn cached(&self) -> String {
        let mut lines = String::new();
        for (key, entry) in self.cache.listed() {
            let found = match &entry {
                Listed::Kept(outcome) => Ok(&**outcome),
                Listed::Running(asked) => Err(asked.elapsed()),
            };
            let line = match (Job::of_key(&key), Program::of_key(&key)) {
                (Some(job), _) => api::listed(&job, found),
                (None, Some(program)) => sm::listed(&program, found),
                (None, None) => continue,
            };
            lines.push_str(&line);
            lines.push('\n');
        }
        lines
    }

    /// The reply to a request and what the log says about it. `direct`: the
    /// request came from this machine, and no proxy passed it on.
    fn respond(self: &Arc<Self>, request: &Request, direct: bool) -> (Response, Option<String>) {
        if request.method != "GET" && request.method != "HEAD" {
            let response = Response::text(405, "only GET and HEAD are served here\n")
                .with("Allow", "GET, HEAD");
            return (response, Some(String::new()));
        }
        let query = request.query.as_deref().unwrap_or("");
        // The format of the reply to a request that cannot be parsed.
        let hint = if query.split('&').any(|piece| piece == "format=text") {
            Format::Text
        } else {
            Format::Json
        };
        let file = |body: &'static str, kind: &'static str| {
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
        // The same for the other program, at its own address.
        let runs = query.split('&').any(|piece| {
            let name = piece.split('=').next().unwrap_or(piece);
            ["expr", "digits", "mode", "domain", "format"].contains(&name)
        });
        let response = match request.path.as_str() {
            "/api" => return self.computed(query, hint),
            "/" if asks => return self.computed(query, hint),
            "/sm" if runs => return self.ran(query, hint),
            "/sm" => self.page(&Form::from_query(&[]), &Shown::Nothing),
            "/sm/help" | "/sm/help.txt" | "/sm/functions" | "/sm/constants" | "/sm/units"
            | "/sm/examples" => self.told(&request.path),
            path if path.starts_with("/sm/file/") || path.starts_with("/sm/output/") => {
                self.written(path, query)
            }
            "/" => self.page(&Form::from_query(&[]), &Shown::Nothing),
            "/app.js" => file(SCRIPT, "text/javascript; charset=utf-8"),
            "/style.css" => file(STYLE, "text/css; charset=utf-8"),
            "/favicon.svg" | "/favicon.ico" => file(ICON, "image/svg+xml"),
            "/help" => Response::text(200, self.help.clone()),
            "/healthz" => Response::text(200, "ok\n"),
            "/stats" => reply(200, Format::Json, self.stats()),
            "/status" => Response::text(200, self.status()),
            // What everybody asked for is shown to this machine alone.
            "/cache" if direct => {
                Response::text(200, self.cached()).with("Cache-Control", "no-store")
            }
            "/robots.txt" => Response::text(
                200,
                "User-agent: *\nDisallow: /api\nDisallow: /sm/file/\nDisallow: /sm/output/\nDisallow: /*?\n",
            ),
            _ => Response::text(404, "there is nothing here; the page is at /\n"),
        };
        (response, Some(String::new()))
    }

    fn serve(self: &Arc<Self>, mut stream: TcpStream, peer: String) {
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
        // The proxy on this machine names the client that it speaks for,
        // and so shows what it passes on.
        let direct = peer
            .parse::<IpAddr>()
            .is_ok_and(|address| address.to_canonical().is_loopback())
            && !request.headers.iter().any(|(name, _)| {
                name.starts_with("x-forwarded-")
                    || ["forwarded", "via", "x-real-ip"].contains(&name.as_str())
            });
        let (response, note) = self.respond(&request, direct);
        let _ = http::write_response(&mut stream, &response, request.method != "HEAD");
        let Some(note) = note else {
            return;
        };
        // The proxy on this machine names the client it speaks for.
        let client = request
            .header("x-forwarded-for")
            .and_then(|list| list.split(',').next())
            .map(str::trim)
            .filter(|first| !first.is_empty() && first.bytes().all(|byte| byte.is_ascii_graphic()))
            .map_or(peer, str::to_string);
        let target: String = match &request.query {
            Some(query) => format!("{}?{query}", request.path),
            None => request.path.clone(),
        };
        let note = match note.as_str() {
            "" => note,
            said => format!(" {said}"),
        };
        println!(
            "{} {client} {} {} -> {} {:.3}s{note}",
            timestamp(SystemTime::now()),
            request.method,
            api::brief(&target, 300),
            response.status,
            started.elapsed().as_secs_f64(),
        );
    }
}

/// What the log says about the answer to a job: how the request came to
/// it, the command, and the result or the reason why there is none. A
/// request that only finds a computation still running, without having
/// waited for it, has no line: a page asks like that every few seconds.
fn note(job: &Work, source: Source, answer: &Answer) -> Option<String> {
    let command = api::brief(&job.command(), 300);
    match answer {
        Ok(outcome) => {
            let seconds = outcome
                .seconds()
                .map_or(String::new(), |seconds| format!(" {seconds:.3}s"));
            let found = outcome.logged();
            Some(format!("{}{seconds} {command}{found}", source.header()))
        }
        Err(pending) if pending.waited => Some(format!(
            "{} {command}: still running after {}",
            source.header(),
            api::span(pending.age.as_secs())
        )),
        Err(_) => None,
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
        "{} tet-web listening on {address}: tet={} symbolic-math={} revision={:?}; {}",
        timestamp(SystemTime::now()),
        server.runner.tet,
        server.sm.as_ref().map_or("none", |sm| sm.program.as_str()),
        server.revision,
        server.settings.now().described(),
    );
    // The settings file is read again whenever it changes.
    server.settings.watch(|read| {
        let time = timestamp(SystemTime::now());
        match read {
            Ok(new) => println!("{time} - SETTINGS {}", new.described()),
            Err(reason) => println!("{time} - SETTINGS not changed: {reason}"),
        }
    });
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
        for file in [page::TEMPLATE, SCRIPT, STYLE] {
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
        assert!(page::TEMPLATE.contains("<script src=\"app.js\" defer></script>"));
        assert!(page::TEMPLATE.contains("<link rel=\"stylesheet\" href=\"style.css\">"));
        assert_eq!(page::TEMPLATE.matches("<script").count(), 1);
        assert!(!page::TEMPLATE.contains("onclick=") && !SCRIPT.contains("innerHTML"));
        // The form goes to this server, which the policy has to allow.
        assert!(page::TEMPLATE.contains("<form id=\"form\" method=\"get\" action=\"/\" "));
        assert!(POLICY.contains("form-action 'self'"));
        assert!(ICON.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""));
    }
}
