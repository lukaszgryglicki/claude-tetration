//! The page, as the server renders it. It is a plain form, so that it works
//! without JavaScript: sending the form asks for this page again, with the
//! result in it. Where scripts run, `app.js` asks for that page itself and
//! puts its result into the page that is shown, which saves the reloads.

use std::fmt::Write as _;

use crate::api::{self, Func, Job, Outcome, Pending};

/// `index.html`, with a `{{name}}` wherever the server writes something.
pub const TEMPLATE: &str = include_str!("../../static/index.html");

/// How often a page whose calculation still runs asks again, in seconds.
const AGAIN: u64 = 5;

/// What the page says about the server.
pub struct Site<'a> {
    /// How long a request waits, and how long a calculation may take, in
    /// seconds.
    pub wait: u64,
    pub limit: u64,
    /// Calculations at a time.
    pub jobs: usize,
    pub revision: &'a str,
}

/// What the form shows: the choice of a function and of the kind of its
/// arguments, the precision and the numbers as they were typed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Form {
    pub func: Func,
    /// Four real numbers, not two complex ones.
    pub four: bool,
    pub prec: String,
    /// The fields of the numbers with their values, in the order of the
    /// command line.
    pub fields: Vec<(String, String)>,
    /// Whether the request had all of these fields, and so asks for a
    /// computation.
    pub complete: bool,
    /// How many times the page has asked again for a calculation that
    /// still runs.
    pub again: u64,
}

/// The numbers that the page starts with.
fn example(func: Func, name: &str) -> &'static str {
    if name.ends_with("_im") {
        return "0";
    }
    match (func, name.trim_end_matches("_re")) {
        (Func::Tet | Func::Slog, "base") => "2",
        (Func::Tet, _) => "0.5",
        (Func::Slog, _) => "65536",
        (Func::Sroot, "value") => "16",
        (Func::Sroot, _) => "3",
    }
}

impl Form {
    /// The form as a query fills it. A form sends all its fields, and
    /// without a script those are the fields of the choice that the page
    /// was made for: a name that the new choice does not have is ignored,
    /// and a field that did not arrive starts with its example.
    pub fn from_query(query: &[(String, String)]) -> Form {
        let get = |name: &str| {
            query
                .iter()
                .find(|(found, _)| found == name)
                .map(|(_, value)| value.as_str())
        };
        let func = get("func").and_then(Func::named).unwrap_or(Func::Tet);
        let two = func.numbers().map(str::to_string);
        let parts: Vec<String> = two
            .iter()
            .flat_map(|number| [format!("{number}_re"), format!("{number}_im")])
            .collect();
        let four = match get("numbers") {
            Some("4") => true,
            Some("2") => false,
            _ => parts.iter().any(|name| get(name).is_some()),
        };
        let names = if four { parts } else { two.to_vec() };
        let complete = get("show").is_none() && names.iter().all(|name| get(name).is_some());
        let fields = names
            .into_iter()
            .map(|name| {
                let value = get(&name)
                    .unwrap_or_else(|| example(func, &name))
                    .to_string();
                (name, value)
            })
            .collect();
        Form {
            func,
            four,
            prec: get("prec").unwrap_or("20").to_string(),
            fields,
            complete,
            again: get("again")
                .and_then(|count| count.parse().ok())
                .unwrap_or(0),
        }
    }

    /// The query of the API for what the form asks.
    pub fn api_query(&self) -> Vec<(String, String)> {
        let mut query = vec![
            ("prec".to_string(), self.prec.clone()),
            ("func".to_string(), self.func.title().to_string()),
        ];
        query.extend(self.fields.iter().cloned());
        query
    }

    /// The address of that query; `readable` leaves what people type in
    /// expressions as it is.
    fn api_address(&self, readable: bool) -> String {
        let pairs: Vec<String> = self
            .api_query()
            .iter()
            .map(|(name, value)| format!("{name}={}", in_query(value, readable)))
            .collect();
        format!("/?{}", pairs.join("&"))
    }

    /// The address of this page for what the form asks: what a browser
    /// asks for when the form is sent, with every value escaped in full.
    pub fn page_address(&self) -> String {
        let mut pairs = vec![
            "format=html".to_string(),
            format!("prec={}", in_query(&self.prec, false)),
            format!("func={}", self.func.title()),
            format!("numbers={}", if self.four { 4 } else { 2 }),
        ];
        let fields = self.fields.iter();
        pairs.extend(fields.map(|(name, value)| format!("{name}={}", in_query(value, false))));
        format!("/?{}", pairs.join("&"))
    }

    /// The address at which the page asks again for a calculation that
    /// still runs. It is another one each time: a browser like `links`
    /// does not ask for the address again that it shows.
    fn again_address(&self) -> String {
        format!(
            "{}&again={}",
            self.page_address(),
            self.again.saturating_add(1)
        )
    }
}

/// What the page shows below the form.
pub enum Shown<'a> {
    Nothing,
    /// The form was sent with the fields of another choice, as it is
    /// without a script: the page has the fields of this one now.
    Fields,
    /// The outcome of the computation that the form asked for.
    Outcome {
        job: &'a Job,
        outcome: &'a Outcome,
        cached: bool,
    },
    /// The computation has not ended.
    Running {
        job: &'a Job,
        pending: &'a Pending,
    },
    /// The form did not ask for a computation: the reason and the status
    /// that the API has for it.
    Rejected {
        message: &'a str,
        status: u16,
    },
}

/// A text as HTML, for an element or for an attribute in double quotes.
pub fn escaped(text: &str) -> String {
    let mut html = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => html.push_str("&amp;"),
            '<' => html.push_str("&lt;"),
            '>' => html.push_str("&gt;"),
            '"' => html.push_str("&quot;"),
            '\'' => html.push_str("&#39;"),
            c => html.push(c),
        }
    }
    html
}

/// A value for a query, as `encodeURIComponent` writes it; `readable`
/// leaves `+ / , ^` as well, which this server reads as they are.
fn in_query(value: &str, readable: bool) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        let plain = byte.is_ascii_alphanumeric()
            || b"-_.!~*'()".contains(&byte)
            || (readable && b"+/,^".contains(&byte));
        if plain {
            encoded.push(char::from(byte));
        } else {
            let _ = write!(encoded, "%{byte:02X}");
        }
    }
    encoded
}

/// The label of a field: `base b`, `height h, real part`.
fn label(name: &str) -> String {
    let (number, part) = match name.rsplit_once('_') {
        Some((number, "re")) => (number, ", real part"),
        Some((number, _)) => (number, ", imaginary part"),
        None => (name, ""),
    };
    let symbol = match number {
        "base" => "b",
        "height" => "h",
        _ => "z",
    };
    format!("{number} {symbol}{part}")
}

/// The seconds as the script of the page prints the number of the API.
fn seconds(seconds: f64) -> String {
    format!("{}", (seconds * 1000.0).round() / 1000.0)
}

/// The template with every `{{name}}` replaced by its value. What is put
/// in is not read again, so that no value can bring a name of its own.
fn filled(values: &[(&str, String)]) -> String {
    let mut page = String::with_capacity(TEMPLATE.len() + 2048);
    let mut rest = TEMPLATE;
    while let Some((before, after)) = rest.split_once("{{") {
        page.push_str(before);
        let (name, after) = after.split_once("}}").unwrap_or((after, ""));
        if let Some((_, value)) = values.iter().find(|(known, _)| *known == name) {
            page.push_str(value);
        }
        rest = after;
    }
    page.push_str(rest);
    page
}

/// What stands below the form, as the inside of its section, and whether
/// it is about a calculation that still runs.
fn result(form: &Form, shown: &Shown) -> (String, bool) {
    let mut html = String::new();
    let title = |html: &mut String, text: &str| {
        let _ = writeln!(html, "  <h2 id=\"result-title\">{}</h2>", escaped(text));
    };
    let failed = |html: &mut String, message: &str, status: u16| {
        title(html, "No result");
        let _ = writeln!(
            html,
            "  <p id=\"error\" class=\"error\">{} (HTTP {status})</p>",
            escaped(message)
        );
    };
    let command = |job: &Job| {
        format!(
            "command: <code id=\"command\">{}</code>",
            escaped(&job.command())
        )
    };
    // The small lines at the end: the time, the command, the address.
    let mut lines = Vec::new();
    let mut running = false;
    match shown {
        Shown::Nothing => return (html, false),
        Shown::Fields => {
            html.push_str(
                "  <p id=\"note\" class=\"note\">The fields of the form are now those of the \
                 chosen function and kind of arguments: fill them in and calculate.</p>\n",
            );
            return (html, false);
        }
        Shown::Outcome {
            job,
            outcome,
            cached,
        } => {
            match outcome {
                Outcome::Value { re, im, .. } => {
                    title(&mut html, &format!("Result: the {}", job.func.result()));
                    let _ = writeln!(
                        html,
                        "  <dl>\n    <dt>real part</dt><dd id=\"re\">{}</dd>\n    \
                         <dt>imaginary part</dt><dd id=\"im\">{}</dd>\n  </dl>",
                        escaped(re),
                        escaped(im)
                    );
                }
                other => failed(
                    &mut html,
                    &other.error().unwrap_or_default(),
                    other.status(),
                ),
            }
            let mut timing = Vec::new();
            if let Some(took) = outcome.seconds() {
                timing.push(format!("computed in {} s", seconds(took)));
            }
            if *cached {
                timing.push("returned from the cache".to_string());
            }
            if !timing.is_empty() {
                lines.push(format!(
                    "<span id=\"timing\">{}</span>",
                    escaped(&timing.join(", "))
                ));
            }
            lines.push(command(job));
        }
        Shown::Running { job, pending } => {
            running = true;
            title(&mut html, "Still calculating");
            let message = pending.message();
            let mut letters = message.chars();
            let first: String = letters
                .next()
                .into_iter()
                .flat_map(char::to_uppercase)
                .collect();
            let _ = writeln!(
                html,
                "  <p id=\"note\" class=\"note\">{}. This page asks again every {AGAIN} \
                 seconds; in a browser that does not, follow the link.</p>\n  \
                 <p id=\"again\"><a href=\"{}\">Ask again now</a></p>",
                escaped(&format!("{first}{}", letters.as_str())),
                escaped(&form.again_address())
            );
            lines.push(command(job));
        }
        Shown::Rejected { message, status } => failed(&mut html, message, *status),
    }
    lines.push(format!(
        "API: <a id=\"api\" href=\"{}\">{}</a>",
        escaped(&form.api_address(false)),
        escaped(&form.api_address(true))
    ));
    let _ = writeln!(
        html,
        "  <p class=\"hint\">\n    {}\n  </p>",
        lines.join("<br>\n    ")
    );
    (html, running)
}

/// The page with the form as given and with what is shown below it.
pub fn rendered(site: &Site, form: &Form, shown: &Shown) -> String {
    let checked = |on: bool| String::from(if on { " checked" } else { "" });
    let mut rows = String::new();
    for (name, value) in &form.fields {
        let _ = writeln!(
            rows,
            "    <div class=\"row\"><label for=\"{name}\">{}</label><input type=\"text\" \
             id=\"{name}\" name=\"{name}\" value=\"{}\" data-key=\"{}.{name}\" \
             spellcheck=\"false\" autocomplete=\"off\" autocapitalize=\"off\"></div>",
            label(name),
            escaped(value),
            form.func.title(),
        );
    }
    let (mut result, running) = result(form, shown);
    // A section with nothing in it is not shown.
    if !result.is_empty() {
        result.insert(0, '\n');
    }
    // Where scripts run, the script of the page asks again: it leaves the
    // page as it is, with what was typed since. Elsewhere the browser
    // does, and one that does not, as `links`, shows a link for it only
    // when the address is named.
    let (refresh, again) = match running {
        true => (
            format!(
                "<noscript><meta http-equiv=\"refresh\" content=\"{AGAIN}; url={}\"></noscript>\n",
                escaped(&form.again_address())
            ),
            format!(" data-running=\"{AGAIN}\""),
        ),
        false => (String::new(), String::new()),
    };
    let revision = match site.revision {
        "" => String::new(),
        revision => format!(", revision {}", escaped(revision)),
    };
    filled(&[
        ("refresh", refresh),
        ("prec", escaped(&form.prec)),
        ("func.tet", checked(form.func == Func::Tet)),
        ("func.ate", checked(form.func == Func::Slog)),
        ("func.ssrt", checked(form.func == Func::Sroot)),
        ("numbers.2", checked(!form.four)),
        ("numbers.4", checked(form.four)),
        ("arguments", rows),
        ("running", again),
        ("result", result),
        ("wait", api::span(site.wait)),
        ("limit", api::span(site.limit)),
        ("jobs", site.jobs.to_string()),
        ("revision", revision),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const SITE: Site = Site {
        wait: 120,
        limit: 10_800,
        jobs: 4,
        revision: "abc1234",
    };

    fn query(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    fn fields(form: &Form) -> Vec<(&str, &str)> {
        form.fields
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect()
    }

    #[test]
    fn every_name_in_the_template_has_a_value() {
        let page = rendered(&SITE, &Form::from_query(&[]), &Shown::Nothing);
        assert!(!page.contains("{{") && !page.contains("}}"));
        let mut names: Vec<&str> = TEMPLATE
            .split("{{")
            .skip(1)
            .map(|rest| rest.split_once("}}").expect("an open name").0)
            .collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            names,
            [
                "arguments",
                "func.ate",
                "func.ssrt",
                "func.tet",
                "jobs",
                "limit",
                "numbers.2",
                "numbers.4",
                "prec",
                "refresh",
                "result",
                "revision",
                "running",
                "wait",
            ]
        );
    }

    #[test]
    fn the_page_starts_as_a_form_that_needs_no_script() {
        let form = Form::from_query(&[]);
        assert!(!form.complete);
        let page = rendered(&SITE, &form, &Shown::Nothing);
        for part in [
            "<form id=\"form\" method=\"get\" action=\"/\" novalidate autocomplete=\"off\">",
            "<input type=\"hidden\" name=\"format\" value=\"html\">",
            "<input id=\"prec\" name=\"prec\" type=\"number\" min=\"1\" step=\"1\" value=\"20\"",
            "<input type=\"radio\" name=\"func\" value=\"tet\" checked>",
            "<input type=\"radio\" name=\"func\" value=\"ate\">",
            "<input type=\"radio\" name=\"numbers\" value=\"2\" checked>",
            "<input type=\"radio\" name=\"numbers\" value=\"4\">",
            "<label for=\"base\">base b</label><input type=\"text\" id=\"base\" name=\"base\" value=\"2\" data-key=\"tet.base\"",
            "<label for=\"height\">height h</label><input type=\"text\" id=\"height\" name=\"height\" value=\"0.5\"",
            "<button type=\"submit\" name=\"show\" value=\"1\">",
            "<button id=\"calculate\" type=\"submit\">Calculate</button>",
            "<section id=\"result\"></section>",
            "to as many digits as are asked for",
            "not ready after 2 minutes continues\n    there, for up to 3 hours",
            "At most 4 calculations run\n    at a time",
            "(AGPL-3.0), revision abc1234.",
        ] {
            assert!(page.contains(part), "{part}\n{page}");
        }
        // Nothing that a browser without scripts and styles would show
        // though it says nothing.
        for absent in ["http-equiv", "hidden>", "result-title", "max="] {
            assert!(!page.contains(absent), "{absent}\n{page}");
        }
    }

    #[test]
    fn a_query_fills_the_form_and_tells_whether_it_asks_for_a_computation() {
        let form = Form::from_query(&query(&[
            ("format", "html"),
            ("prec", "30"),
            ("func", "ate"),
            ("numbers", "2"),
            ("base", "e"),
            ("value", "10"),
        ]));
        assert_eq!(
            (form.func, form.four, form.complete),
            (Func::Slog, false, true)
        );
        assert_eq!(fields(&form), [("base", "e"), ("value", "10")]);
        assert_eq!(
            form.api_query(),
            query(&[
                ("prec", "30"),
                ("func", "ate"),
                ("base", "e"),
                ("value", "10")
            ])
        );
        // The other name of a function is the same button.
        let sroot = Form::from_query(&query(&[
            ("func", "sroot"),
            ("value", "16"),
            ("height", "3"),
        ]));
        assert_eq!(
            (sroot.func, sroot.complete, sroot.prec.as_str()),
            (Func::Sroot, true, "20")
        );
        // Without a script the fields are those of the choice before: the
        // ones that stay keep their values, the new ones get their examples.
        let switched = Form::from_query(&query(&[
            ("func", "ate"),
            ("numbers", "2"),
            ("base", "3"),
            ("height", "0.25"),
        ]));
        assert!(!switched.complete);
        assert_eq!(fields(&switched), [("base", "3"), ("value", "65536")]);
        let parts = Form::from_query(&query(&[
            ("func", "ssrt"),
            ("numbers", "4"),
            ("value", "16"),
            ("height", "3"),
        ]));
        assert!(parts.four && !parts.complete);
        assert_eq!(
            fields(&parts),
            [
                ("value_re", "16"),
                ("value_im", "0"),
                ("height_re", "3"),
                ("height_im", "0")
            ]
        );
        // The four names alone say which kind it is; `show` only shows.
        let four = [
            ("base_re", "1.5"),
            ("base_im", "0.5"),
            ("height_re", "1"),
            ("height_im", "0.7"),
        ];
        let named = Form::from_query(&query(&four));
        assert!(named.four && named.complete && named.func == Func::Tet);
        let mut shown = four.to_vec();
        shown.push(("show", "1"));
        assert!(!Form::from_query(&query(&shown)).complete);
        // An empty field is a field: tet says what is wrong with it.
        assert!(Form::from_query(&query(&[("base", ""), ("height", "")])).complete);
        // Anything else is the page as it starts.
        let unknown = Form::from_query(&query(&[("func", "exp"), ("numbers", "3")]));
        assert_eq!(unknown, Form::from_query(&[]));
    }

    fn job(form: &Form) -> Job {
        Job::parse(&form.api_query()).unwrap().0
    }

    #[test]
    fn a_result_is_shown_with_its_command_and_its_address() {
        let form = Form::from_query(&query(&[
            ("func", "ssrt"),
            ("prec", "5"),
            ("value", "10+i"),
            ("height", "pi/e + 1"),
        ]));
        let outcome = Outcome::Value {
            re: "2.1234".into(),
            im: "-5.0000e-2".into(),
            seconds: 1.5,
        };
        let shown = Shown::Outcome {
            job: &job(&form),
            outcome: &outcome,
            cached: true,
        };
        let page = rendered(&SITE, &form, &shown);
        for part in [
            "<input type=\"radio\" name=\"func\" value=\"ssrt\" checked>",
            "name=\"value\" value=\"10+i\" data-key=\"ssrt.value\"",
            "name=\"height\" value=\"pi/e + 1\"",
            "<section id=\"result\">\n  <h2 id=\"result-title\">Result: the base</h2>\n  <dl>\n    \
             <dt>real part</dt><dd id=\"re\">2.1234</dd>\n    \
             <dt>imaginary part</dt><dd id=\"im\">-5.0000e-2</dd>\n  </dl>\n  \
             <p class=\"hint\">\n    \
             <span id=\"timing\">computed in 1.5 s, returned from the cache</span><br>\n    \
             command: <code id=\"command\">tet --ssrt 5 10+i &#39;pi/e + 1&#39;</code><br>\n    \
             API: <a id=\"api\" \
             href=\"/?prec=5&amp;func=ssrt&amp;value=10%2Bi&amp;height=pi%2Fe%20%2B%201\">\
             /?prec=5&amp;func=ssrt&amp;value=10+i&amp;height=pi/e%20+%201</a>\n  </p>\n</section>",
        ] {
            assert!(page.contains(part), "{part}\n{page}");
        }
        for absent in ["http-equiv", "id=\"error\"", "id=\"note\"", "data-running"] {
            assert!(!page.contains(absent), "{absent}\n{page}");
        }
        assert_eq!(
            form.page_address(),
            "/?format=html&prec=5&func=ssrt&numbers=2&value=10%2Bi&height=pi%2Fe%20%2B%201"
        );
    }

    #[test]
    fn an_outcome_without_a_value_says_why() {
        let form = Form::from_query(&query(&[("base", "2"), ("height", "0.5"), ("prec", "900")]));
        let job = job(&form);
        let section = |outcome: Outcome, cached: bool| {
            let shown = Shown::Outcome {
                job: &job,
                outcome: &outcome,
                cached,
            };
            let page = rendered(&SITE, &form, &shown);
            let (_, rest) = page.split_once("<section id=\"result\">\n").unwrap();
            rest.split_once("</section>").unwrap().0.to_string()
        };
        let hint =
            "command: <code id=\"command\">tet 900 2 0.5</code><br>\n    API: <a id=\"api\" \
                    href=\"/?prec=900&amp;func=tet&amp;base=2&amp;height=0.5\">\
                    /?prec=900&amp;func=tet&amp;base=2&amp;height=0.5</a>\n  </p>\n";
        assert_eq!(
            section(Outcome::Busy, false),
            format!(
                "  <h2 id=\"result-title\">No result</h2>\n  <p id=\"error\" class=\"error\">all \
                 slots are busy calculating; try again later (HTTP 503)</p>\n  \
                 <p class=\"hint\">\n    {hint}"
            )
        );
        // What the time limit stopped is kept, and has no time of its own.
        assert_eq!(
            section(Outcome::Timeout { limit: 10_800 }, true),
            format!(
                "  <h2 id=\"result-title\">No result</h2>\n  <p id=\"error\" class=\"error\">no \
                 result within 3 hours, the limit of this server; tet itself has no limit: build \
                 it and run the command (HTTP 504)</p>\n  <p class=\"hint\">\n    \
                 <span id=\"timing\">returned from the cache</span><br>\n    {hint}"
            )
        );
    }

    #[test]
    fn what_was_typed_cannot_become_markup() {
        let typed = "\"><script>alert(1)</script>{{result}}&'";
        let form = Form::from_query(&query(&[("prec", typed), ("base", typed), ("height", "1")]));
        let refused = Outcome::Refused {
            message: format!("invalid number {typed:?}"),
            seconds: 0.004,
        };
        let rejected = Job::parse(&form.api_query()).unwrap_err();
        let page = rendered(
            &SITE,
            &form,
            &Shown::Rejected {
                message: &rejected.message,
                status: rejected.status,
            },
        );
        assert!(page.contains("<h2 id=\"result-title\">No result</h2>"));
        assert!(!page.contains("<dl>") && !page.contains("id=\"command\""));
        assert!(page.contains("(HTTP 400)</p>"));
        let fine = Form {
            prec: "20".into(),
            ..form
        };
        let shown = Shown::Outcome {
            job: &job(&fine),
            outcome: &refused,
            cached: false,
        };
        let page = [page, rendered(&SITE, &fine, &shown)].concat();
        assert!(page.contains("(HTTP 422)</p>"));
        assert!(page.contains("<span id=\"timing\">computed in 0.004 s</span>"));
        // The one script of each of the two pages is all there is, and a
        // name of the template that was typed stays what was typed.
        assert_eq!(page.matches("<script").count(), 2);
        assert!(!page.contains("alert(1)</script>"));
        assert!(page.contains(
            "value=\"&quot;&gt;&lt;script&gt;alert(1)&lt;/script&gt;{{result}}&amp;&#39;\""
        ));
        assert!(page.contains(
            "invalid number &quot;\\&quot;&gt;&lt;script&gt;alert(1)&lt;/script&gt;{{result}}&amp;&#39;&quot; \
             (HTTP 422)</p>"
        ));
        assert!(!page.contains("Result: the"));
    }

    #[test]
    fn a_calculation_that_runs_makes_the_page_ask_again() {
        let form = Form::from_query(&query(&[("prec", "300"), ("base", "2"), ("height", "0.5")]));
        let pending = Pending {
            waited: true,
            age: Duration::from_secs(120),
            wait: 120,
            limit: 10_800,
        };
        let shown = Shown::Running {
            job: &job(&form),
            pending: &pending,
        };
        let page = rendered(&SITE, &form, &shown);
        for part in [
            // A browser without scripts asks again by itself, or its user
            // does; with scripts the script of the page does.
            "<noscript><meta http-equiv=\"refresh\" content=\"5; \
             url=/?format=html&amp;prec=300&amp;func=tet&amp;numbers=2&amp;base=2&amp;height=0.5&amp;\
             again=1\"></noscript>\n<title>",
            "<section id=\"result\" data-running=\"5\">\n  \
             <h2 id=\"result-title\">Still calculating</h2>\n  \
             <p id=\"note\" class=\"note\">The calculation is taking longer than 2 minutes and \
             continues on this server: ask the same again later for the result; it may take up \
             to 3 hours. This page asks again every 5 seconds; in a browser that does not, \
             follow the link.</p>\n  \
             <p id=\"again\"><a \
             href=\"/?format=html&amp;prec=300&amp;func=tet&amp;numbers=2&amp;base=2&amp;height=0.5&amp;\
             again=1\">Ask again now</a></p>\n  <p class=\"hint\">\n    \
             command: <code id=\"command\">tet 300 2 0.5</code><br>\n    API: <a id=\"api\"",
        ] {
            assert!(page.contains(part), "{part}\n{page}");
        }
        assert!(!page.contains("<dl>") && !page.contains("id=\"timing\""));
        // Each time the page asks at another address, which a browser
        // cannot take for the page that it has.
        let asked = Form {
            again: 7,
            ..form.clone()
        };
        assert_eq!(asked, {
            let mut sent = query(&[("prec", "300"), ("base", "2"), ("height", "0.5")]);
            sent.push(("again".into(), "7".into()));
            Form::from_query(&sent)
        });
        let page = rendered(&SITE, &asked, &shown);
        assert_eq!(page.matches("&amp;height=0.5&amp;again=8\"").count(), 2);
        assert!(!page.contains("again=7") && !page.contains("again=1"));
        assert_eq!(asked.page_address(), form.page_address());
    }

    #[test]
    fn a_form_with_the_fields_of_another_choice_shows_those_of_its_own() {
        let form = Form::from_query(&query(&[
            ("format", "html"),
            ("func", "ate"),
            ("numbers", "4"),
            ("base", "3"),
            ("height", "0.25"),
        ]));
        assert!(!form.complete);
        let page = rendered(&SITE, &form, &Shown::Fields);
        for part in [
            "name=\"func\" value=\"ate\" checked>",
            "name=\"numbers\" value=\"4\" checked>",
            "<label for=\"base_re\">base b, real part</label>",
            "<label for=\"value_im\">value z, imaginary part</label>",
            "<section id=\"result\">\n  <p id=\"note\" class=\"note\">The fields of the form are \
             now those of the chosen function and kind of arguments: fill them in and \
             calculate.</p>\n</section>",
        ] {
            assert!(page.contains(part), "{part}\n{page}");
        }
        assert!(!page.contains("name=\"height\"") && !page.contains("http-equiv"));
    }

    #[test]
    fn values_are_written_for_a_query() {
        for (value, strict, readable) in [
            ("2", "2", "2"),
            ("1.5+0.5i", "1.5%2B0.5i", "1.5+0.5i"),
            (
                "pi/e + i*ssrt(16, 3)/4",
                "pi%2Fe%20%2B%20i*ssrt(16%2C%203)%2F4",
                "pi/e%20+%20i*ssrt(16,%203)/4",
            ),
            ("2^^0.5", "2%5E%5E0.5", "2^^0.5"),
            ("a&b=c#d%", "a%26b%3Dc%23d%25", "a%26b%3Dc%23d%25"),
            ("π", "%CF%80", "%CF%80"),
        ] {
            assert_eq!(in_query(value, false), strict);
            assert_eq!(in_query(value, true), readable);
        }
    }
}
