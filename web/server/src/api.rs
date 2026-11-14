//! From a query to a `tet` command line, and from its outcome to a reply.

use std::fmt::Write as _;
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Func {
    Tet,
    Slog,
    Sroot,
}

impl Func {
    pub fn named(name: &str) -> Option<Func> {
        match name {
            "tet" => Some(Func::Tet),
            "slog" | "ate" => Some(Func::Slog),
            "sroot" | "ssrt" => Some(Func::Sroot),
            _ => None,
        }
    }

    /// The name that the page has for the function.
    pub fn title(self) -> &'static str {
        match self {
            Func::Tet => "tet",
            Func::Slog => "ate",
            Func::Sroot => "ssrt",
        }
    }

    /// The two numbers that the function takes, in the order of the
    /// command line.
    pub fn numbers(self) -> [&'static str; 2] {
        match self {
            Func::Tet => ["base", "height"],
            Func::Slog => ["base", "value"],
            Func::Sroot => ["value", "height"],
        }
    }

    /// The number that it prints.
    pub fn result(self) -> &'static str {
        match self {
            Func::Tet => "value",
            Func::Slog => "height",
            Func::Sroot => "base",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Json,
    Text,
}

/// One computation: a function, a precision, and two complex numbers or the
/// four real parts of two.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Job {
    pub func: Func,
    /// The name the request used: `tet`, `ate`, `slog`, `sroot` or `ssrt`.
    pub spelled: String,
    pub prec: u64,
    /// The parameters with their values, in the order of the command line.
    pub arguments: Vec<(String, String)>,
}

/// A request that is not a computation: the status and the reason.
#[derive(Debug, PartialEq, Eq)]
pub struct Rejected {
    pub status: u16,
    pub message: String,
}

fn rejected(message: impl Into<String>) -> Rejected {
    Rejected {
        status: 400,
        message: message.into(),
    }
}

/// The words that `tet` takes for flags wherever they stand.
const FLAGS: [&str; 9] = [
    "-q", "--quiet", "--silent", "--slog", "--ate", "--sroot", "--ssrt", "--help", "-h",
];

impl Job {
    /// The job that a query asks for, and the format of the reply.
    pub fn parse(query: &[(String, String)]) -> Result<(Job, Format), Rejected> {
        let get = |name: &str| {
            query
                .iter()
                .find(|(found, _)| found == name)
                .map(|(_, value)| value.as_str())
        };
        let format = match get("format") {
            None | Some("json") => Format::Json,
            Some("text") => Format::Text,
            Some(other) => {
                return Err(rejected(format!(
                    "format is json, text or html, not {other:?}"
                )))
            }
        };
        let spelled = get("func")
            .ok_or_else(|| rejected("func is missing: tet, ate (slog) or ssrt (sroot)"))?;
        let func = Func::named(spelled).ok_or_else(|| {
            rejected(format!(
                "func is tet, ate, slog, sroot or ssrt, not {spelled:?}"
            ))
        })?;
        let prec = match get("prec") {
            None => 20,
            Some(text) => text
                .parse::<u64>()
                .ok()
                .filter(|prec| *prec >= 1)
                // Only digits: no sign, no space.
                .filter(|_| text.bytes().all(|byte| byte.is_ascii_digit()))
                .ok_or_else(|| {
                    rejected(format!(
                        "prec is the number of digits, a whole number from 1 on, not {text:?}"
                    ))
                })?,
        };
        let [first, second] = func.numbers();
        let two = [first.to_string(), second.to_string()];
        let four = [
            format!("{first}_re"),
            format!("{first}_im"),
            format!("{second}_re"),
            format!("{second}_im"),
        ];
        let takes = format!(
            "func={spelled} takes {first} and {second}, or {}, {}, {} and {}",
            four[0], four[1], four[2], four[3]
        );
        for (name, _) in query {
            let known = ["func", "prec", "format"].contains(&name.as_str())
                || two.contains(name)
                || four.contains(name);
            if !known {
                return Err(rejected(format!("unknown parameter {name:?}: {takes}")));
            }
        }
        let given = |names: &[String]| names.iter().filter(|name| get(name).is_some()).count();
        let names: &[String] = match (given(&two), given(&four)) {
            (2, 0) => &two,
            (0, 4) => &four,
            _ => return Err(rejected(takes)),
        };
        let mut arguments = Vec::new();
        for name in names {
            let value = get(name).unwrap_or_default();
            if value.chars().any(char::is_control) || FLAGS.contains(&value) {
                return Err(rejected(format!("{name} is not a number: {value:?}")));
            }
            arguments.push((name.clone(), value.to_string()));
        }
        let job = Job {
            func,
            spelled: spelled.to_string(),
            prec,
            arguments,
        };
        Ok((job, format))
    }

    /// The arguments of `tet` for this job.
    pub fn argv(&self) -> Vec<String> {
        let mut argv = vec!["-q".to_string()];
        if self.func != Func::Tet {
            argv.push(format!("--{}", self.spelled));
        }
        argv.push(self.prec.to_string());
        argv.extend(self.arguments.iter().map(|(_, value)| value.clone()));
        argv
    }

    /// The command line to run by hand, quoted for a shell.
    pub fn command(&self) -> String {
        let mut words = vec!["tet".to_string()];
        words.extend(self.argv().iter().skip(1).map(|word| shell_quoted(word)));
        words.join(" ")
    }

    /// What identifies the computation: `ate` and `slog` are one function,
    /// and so are `ssrt` and `sroot`. The arguments are kept as written.
    pub fn key(&self) -> String {
        let mut key = format!("{:?}\n{}", self.func, self.prec);
        for (_, value) in &self.arguments {
            key.push('\n');
            key.push_str(value);
        }
        key
    }

    /// The job that `key` made a key from, under the name that the page
    /// has for its function: the key does not tell `ate` from `slog`.
    pub fn of_key(key: &str) -> Option<Job> {
        let mut lines = key.split('\n');
        let name = lines.next()?;
        let func = [Func::Tet, Func::Slog, Func::Sroot]
            .into_iter()
            .find(|func| format!("{func:?}") == name)?;
        let prec = lines.next()?.parse().ok()?;
        let values: Vec<&str> = lines.collect();
        let [first, second] = func.numbers();
        let names = match values.len() {
            2 => vec![first.to_string(), second.to_string()],
            4 => vec![
                format!("{first}_re"),
                format!("{first}_im"),
                format!("{second}_re"),
                format!("{second}_im"),
            ],
            _ => return None,
        };
        let arguments = names
            .into_iter()
            .zip(values.iter().map(|value| value.to_string()))
            .collect();
        Some(Job {
            func,
            spelled: func.title().to_string(),
            prec,
            arguments,
        })
    }
}

/// A word as a POSIX shell reads it back.
pub fn shell_quoted(word: &str) -> String {
    let plain = |c: char| c.is_ascii_alphanumeric() || "._+-/,=:".contains(c);
    if !word.is_empty() && word.chars().all(plain) {
        word.to_string()
    } else {
        format!("'{}'", word.replace('\'', "'\\''"))
    }
}

/// A number of seconds as people say it: `45 s`, `2 minutes`, `3 hours`,
/// `2 min 30 s`, `4 h 57 min`.
pub fn span(seconds: u64) -> String {
    let (hours, minutes, rest) = (seconds / 3600, seconds % 3600 / 60, seconds % 60);
    let counted =
        |count: u64, unit: &str| format!("{count} {unit}{}", if count == 1 { "" } else { "s" });
    match (hours, minutes, rest) {
        (0, 0, _) => format!("{rest} s"),
        (0, _, 0) => counted(minutes, "minute"),
        (_, 0, 0) => counted(hours, "hour"),
        (0, _, _) => format!("{minutes} min {rest} s"),
        _ => format!("{hours} h {minutes} min"),
    }
}

/// How a computation ended.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// The two lines that `tet` printed.
    Value {
        re: String,
        im: String,
        seconds: f64,
    },
    /// `tet` refused the input or found no answer; its message.
    Refused { message: String, seconds: f64 },
    /// No answer within the time limit of the server, in seconds.
    Timeout { limit: u64 },
    /// No free slot: as many computations run as may run at a time.
    Busy,
    /// Something that is neither an answer nor a refusal of `tet`.
    Failed { message: String },
}

impl Outcome {
    pub fn status(&self) -> u16 {
        match self {
            Outcome::Value { .. } => 200,
            Outcome::Refused { .. } => 422,
            Outcome::Timeout { .. } => 504,
            Outcome::Busy => 503,
            Outcome::Failed { .. } => 500,
        }
    }

    /// A finished run of `tet` is the same whenever it is repeated, and a
    /// computation that the time limit stopped would be stopped again,
    /// after as long a time. The other outcomes depend on the load of the
    /// moment.
    pub fn lasting(&self) -> bool {
        matches!(
            self,
            Outcome::Value { .. } | Outcome::Refused { .. } | Outcome::Timeout { .. }
        )
    }

    pub fn error(&self) -> Option<String> {
        match self {
            Outcome::Value { .. } => None,
            Outcome::Refused { message, .. } | Outcome::Failed { message } => Some(message.clone()),
            Outcome::Timeout { limit } => Some(format!(
                "no result within {}, the limit of this server; tet itself has no limit: \
                 build it and run the command",
                span(*limit)
            )),
            Outcome::Busy => Some("all slots are busy calculating; try again later".into()),
        }
    }

    pub fn seconds(&self) -> Option<f64> {
        match self {
            Outcome::Value { seconds, .. } | Outcome::Refused { seconds, .. } => Some(*seconds),
            _ => None,
        }
    }

    /// What the log says about the outcome: the two numbers of a result,
    /// or the reason why there is none.
    pub fn logged(&self) -> String {
        match self {
            Outcome::Value { re, im, .. } => format!(" = {} {}", brief(re, 100), brief(im, 100)),
            other => format!(": {}", brief(&other.error().unwrap_or_default(), 300)),
        }
    }
}

/// A text for a line of the log: of a long one the beginning, and its
/// length.
pub fn brief(text: &str, most: usize) -> String {
    let length = text.chars().count();
    if length <= most {
        return text.to_string();
    }
    let beginning: String = text.chars().take(most).collect();
    format!("{beginning}...({length} characters)")
}

/// A string as a JSON string.
pub fn json_string(text: &str) -> String {
    let mut quoted = String::with_capacity(text.len() + 2);
    quoted.push('"');
    for c in text.chars() {
        match c {
            '"' => quoted.push_str("\\\""),
            '\\' => quoted.push_str("\\\\"),
            '\n' => quoted.push_str("\\n"),
            '\r' => quoted.push_str("\\r"),
            '\t' => quoted.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(quoted, "\\u{:04x}", u32::from(c));
            }
            c => quoted.push(c),
        }
    }
    quoted.push('"');
    quoted
}

/// What a request is told about a computation that has not ended.
#[derive(Clone, Debug, PartialEq)]
pub struct Pending {
    /// Whether the request waited for the computation. One that arrives
    /// when the others have stopped waiting is answered at once.
    pub waited: bool,
    /// How long ago the computation was asked for first.
    pub age: Duration,
    /// How long a request waits, and how long a computation may take, in
    /// seconds.
    pub wait: u64,
    pub limit: u64,
}

impl Pending {
    pub fn message(&self) -> String {
        let limit = span(self.limit);
        if self.waited {
            format!(
                "the calculation is taking longer than {} and continues on this server: ask \
                 the same again later for the result; it may take up to {limit}",
                span(self.wait)
            )
        } else {
            format!(
                "this calculation is already running, for {} now: allow it up to {limit} to \
                 complete and ask the same again for the result",
                span(self.age.as_secs())
            )
        }
    }
}

/// The request as the first fields of a JSON reply.
fn request_fields(job: &Job) -> Vec<String> {
    let mut fields = vec![
        format!("\"func\":{}", json_string(&job.spelled)),
        format!("\"prec\":{}", job.prec),
    ];
    for (name, value) in &job.arguments {
        fields.push(format!("{}:{}", json_string(name), json_string(value)));
    }
    fields
}

/// The body of the reply about a job whose computation has not ended.
pub fn rendered_pending(job: &Job, pending: &Pending, format: Format) -> String {
    if format == Format::Text {
        return format!("running: {}\n", pending.message());
    }
    let mut fields = request_fields(job);
    fields.push("\"running\":true".into());
    fields.push(format!("\"message\":{}", json_string(&pending.message())));
    fields.push("\"status\":202".into());
    fields.push(format!(
        "\"running_seconds\":{:.3}",
        pending.age.as_secs_f64()
    ));
    fields.push(format!("\"limit_seconds\":{}", pending.limit));
    fields.push("\"cached\":false".into());
    fields.push(format!("\"command\":{}", json_string(&job.command())));
    format!("{{{}}}\n", fields.join(","))
}

/// The body of the reply to a job.
pub fn rendered(job: &Job, outcome: &Outcome, cached: bool, format: Format) -> String {
    if format == Format::Text {
        return match outcome {
            Outcome::Value { re, im, .. } => format!("{re}\n{im}\n"),
            other => format!("error: {}\n", other.error().unwrap_or_default()),
        };
    }
    let mut fields = request_fields(job);
    match outcome {
        Outcome::Value { re, im, .. } => {
            fields.push(format!("\"result\":{}", json_string(job.func.result())));
            fields.push(format!("\"re\":{}", json_string(re)));
            fields.push(format!("\"im\":{}", json_string(im)));
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
    fields.push(format!("\"command\":{}", json_string(&job.command())));
    format!("{{{}}}\n", fields.join(","))
}

/// The two lines that `tet` prints as one complex number.
fn complex(re: &str, im: &str) -> String {
    match im {
        "0" => re.to_string(),
        signed if signed.starts_with(['-', '+']) => format!("{re}{signed}i"),
        _ => format!("{re}+{im}i"),
    }
}

/// A computation as one line for people: the function, the precision and
/// the numbers of the function in the order base, height, value. Two of
/// them are as they were asked for, and the third is the result, of a
/// computation that has one. Without a result the line ends with the
/// reason why there is none or, for a computation that has not ended
/// (`Err`), with the time that it has been running.
pub fn listed(job: &Job, found: Result<&Outcome, Duration>) -> String {
    let mut words = vec![
        format!("f={}", job.func.title()),
        format!("prec={}", job.prec),
    ];
    let parts = job.arguments.len() == 4;
    for number in ["base", "height", "value"] {
        if number != job.func.result() {
            for (name, value) in &job.arguments {
                let whole = name
                    .strip_suffix("_re")
                    .or_else(|| name.strip_suffix("_im"))
                    .unwrap_or(name);
                if whole == number {
                    words.push(format!("{name}={}", shell_quoted(value)));
                }
            }
        } else if let Ok(Outcome::Value { re, im, .. }) = found {
            if parts {
                words.push(format!("{number}_re={}", shell_quoted(re)));
                words.push(format!("{number}_im={}", shell_quoted(im)));
            } else {
                words.push(format!("{number}={}", shell_quoted(&complex(re, im))));
            }
        }
    }
    match found {
        Ok(Outcome::Value { .. }) => {}
        Ok(other) => {
            let reason = other.error().unwrap_or_default();
            words.push(format!("error={}", shell_quoted(&reason)));
        }
        Err(age) => words.push(format!("running={}s", age.as_secs())),
    }
    words.join(" ")
}

/// The body of the reply to a request that is not a job.
pub fn rendered_rejection(rejected: &Rejected, format: Format) -> String {
    match format {
        Format::Text => format!("error: {}\n", rejected.message),
        Format::Json => format!(
            "{{\"error\":{},\"status\":{}}}\n",
            json_string(&rejected.message),
            rejected.status
        ),
    }
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

    fn job(pairs: &[(&str, &str)]) -> Result<(Job, Format), Rejected> {
        Job::parse(&query(pairs))
    }

    #[test]
    fn the_three_functions_take_two_numbers_or_four_parts() {
        for (pairs, command) in [
            (
                &[("func", "tet"), ("base", "2"), ("height", "0.5")][..],
                "tet 20 2 0.5",
            ),
            (
                &[
                    ("prec", "50"),
                    ("func", "ate"),
                    ("base", "2"),
                    ("value", "65536"),
                ],
                "tet --ate 50 2 65536",
            ),
            (
                &[("func", "slog"), ("value", "1.5+0.5i"), ("base", "e^-e")],
                "tet --slog 20 'e^-e' 1.5+0.5i",
            ),
            (
                &[
                    ("func", "ssrt"),
                    ("value", "16"),
                    ("height", "3"),
                    ("prec", "1"),
                ],
                "tet --ssrt 1 16 3",
            ),
            (
                &[("func", "sroot"), ("height", "pi/e"), ("value", "-1")],
                "tet --sroot 20 -1 pi/e",
            ),
            (
                &[
                    ("func", "tet"),
                    ("height_im", "0.7"),
                    ("base_re", "1.5"),
                    ("height_re", "1"),
                    ("base_im", "0.5"),
                ],
                "tet 20 1.5 0.5 1 0.7",
            ),
            (
                &[
                    ("func", "ate"),
                    ("base_re", "2"),
                    ("base_im", "0"),
                    ("value_re", "re(2^^0.5)"),
                    ("value_im", "0"),
                    ("format", "json"),
                ],
                "tet --ate 20 2 0 're(2^^0.5)' 0",
            ),
            (
                &[
                    ("func", "ssrt"),
                    ("value_re", "16"),
                    ("value_im", "0"),
                    ("height_re", "3"),
                    ("height_im", "0"),
                ],
                "tet --ssrt 20 16 0 3 0",
            ),
        ] {
            let (job, format) = job(pairs).unwrap_or_else(|error| panic!("{pairs:?}: {error:?}"));
            assert_eq!(job.command(), command);
            assert_eq!(format, Format::Json);
            assert_eq!(job.argv()[0], "-q");
            let plain = command.replace('\'', "");
            assert_eq!(job.argv()[1..].join(" "), plain["tet ".len()..]);
        }
        let (_, format) = job(&[
            ("func", "tet"),
            ("base", "2"),
            ("height", "3"),
            ("format", "text"),
        ])
        .unwrap();
        assert_eq!(format, Format::Text);
    }

    #[test]
    fn the_names_of_one_function_share_a_key() {
        let key = |pairs: &[(&str, &str)]| job(pairs).unwrap().0.key();
        let ate = key(&[("func", "ate"), ("base", "2"), ("value", "16")]);
        assert_eq!(
            ate,
            key(&[
                ("value", "16"),
                ("func", "slog"),
                ("base", "2"),
                ("prec", "20")
            ])
        );
        assert_eq!(
            key(&[("func", "ssrt"), ("value", "16"), ("height", "3")]),
            key(&[("func", "sroot"), ("value", "16"), ("height", "3")])
        );
        // Another function, precision, form or spelling is another computation.
        for other in [
            &[("func", "tet"), ("base", "2"), ("height", "16")][..],
            &[
                ("func", "ate"),
                ("base", "2"),
                ("value", "16"),
                ("prec", "21"),
            ],
            &[("func", "ate"), ("base", "2"), ("value", "16.0")],
            &[("func", "ate"), ("base", "2"), ("value", " 16")],
            &[
                ("func", "ate"),
                ("base_re", "2"),
                ("base_im", "0"),
                ("value_re", "16"),
                ("value_im", "0"),
            ],
        ] {
            assert_ne!(key(other), ate, "{other:?}");
        }
    }

    #[test]
    fn requests_that_are_not_a_computation_say_why() {
        let takes = "func=tet takes base and height, or base_re, base_im, height_re and height_im";
        for (pairs, message) in [
            (&[("base", "2")][..], "func is missing: tet, ate (slog) or ssrt (sroot)".to_string()),
            (&[("func", "exp")], "func is tet, ate, slog, sroot or ssrt, not \"exp\"".to_string()),
            (&[("func", "TET")], "func is tet, ate, slog, sroot or ssrt, not \"TET\"".to_string()),
            (&[("func", "tet")], takes.to_string()),
            (&[("func", "tet"), ("base", "2")], takes.to_string()),
            (
                &[("func", "tet"), ("base", "2"), ("height_re", "1"), ("height_im", "0")],
                takes.to_string(),
            ),
            (
                &[("func", "tet"), ("base", "2"), ("height", "3"), ("base_re", "2")],
                takes.to_string(),
            ),
            (
                &[("func", "tet"), ("base_re", "2"), ("base_im", "0"), ("height_re", "3")],
                takes.to_string(),
            ),
            (
                &[("func", "tet"), ("base", "2"), ("value", "3")],
                format!("unknown parameter \"value\": {takes}"),
            ),
            (
                &[("func", "ate"), ("base", "2"), ("height", "3")],
                "unknown parameter \"height\": func=ate takes base and value, or base_re, base_im, \
                 value_re and value_im"
                    .to_string(),
            ),
            (
                &[("func", "ssrt"), ("base", "2"), ("height", "3")],
                "unknown parameter \"base\": func=ssrt takes value and height, or value_re, \
                 value_im, height_re and height_im"
                    .to_string(),
            ),
            (
                &[("func", "tet"), ("base", "2"), ("height", "3"), ("format", "xml")],
                "format is json, text or html, not \"xml\"".to_string(),
            ),
        ] {
            let error = job(pairs).unwrap_err();
            assert_eq!((error.status, error.message), (400, message), "{pairs:?}");
        }
        for prec in [
            "0",
            "-5",
            "+5",
            " 5",
            "5.0",
            "2e1",
            "",
            "x",
            "99999999999999999999",
        ] {
            let error = job(&[
                ("func", "tet"),
                ("base", "2"),
                ("height", "3"),
                ("prec", prec),
            ])
            .unwrap_err();
            assert!(
                error
                    .message
                    .starts_with("prec is the number of digits, a whole number from 1 on, not "),
                "{prec}: {}",
                error.message
            );
        }
        // A word that tet reads as a flag, or a control character, is no number.
        for word in [
            "-q", "--slog", "--ate", "--sroot", "--ssrt", "--help", "-h", "--quiet", "2\n3",
            "2\x00",
        ] {
            let error = job(&[("func", "tet"), ("base", word), ("height", "3")]).unwrap_err();
            assert_eq!(error.message, format!("base is not a number: {word:?}"));
        }
        // Any number of digits may be asked for, and a number may be long.
        let digits = "1".repeat(5000);
        let (many, _) = job(&[
            ("func", "tet"),
            ("base", &digits),
            ("height", "3"),
            ("prec", "1000000000"),
        ])
        .unwrap();
        assert_eq!(
            (many.prec, many.arguments[0].1.len()),
            (1_000_000_000, 5000)
        );
        // What only looks like one goes to tet.
        for word in ["-2", "--2", " -q", "-qq", "", "-pi"] {
            assert!(
                job(&[("func", "tet"), ("base", word), ("height", "3")]).is_ok(),
                "{word}"
            );
        }
    }

    #[test]
    fn words_are_quoted_for_a_shell() {
        for (word, quoted) in [
            ("2", "2"),
            ("-0.5", "-0.5"),
            ("1.5+0.5i", "1.5+0.5i"),
            ("pi/e", "pi/e"),
            ("2^^0.5", "'2^^0.5'"),
            ("ate(2, 16)", "'ate(2, 16)'"),
            ("2*pi", "'2*pi'"),
            ("", "''"),
            ("it's", "'it'\\''s'"),
            ("$(id)", "'$(id)'"),
            ("π", "'π'"),
        ] {
            assert_eq!(shell_quoted(word), quoted);
        }
    }

    #[test]
    fn replies_are_json_or_the_two_lines_of_tet() {
        let (job, _) = job(&[
            ("func", "ate"),
            ("base", "2"),
            ("value", "2^^3"),
            ("prec", "5"),
        ])
        .unwrap();
        let value = Outcome::Value {
            re: "3.0000".into(),
            im: "0".into(),
            seconds: 0.01234,
        };
        assert_eq!(
            rendered(&job, &value, false, Format::Json),
            "{\"func\":\"ate\",\"prec\":5,\"base\":\"2\",\"value\":\"2^^3\",\"result\":\"height\",\
             \"re\":\"3.0000\",\"im\":\"0\",\"seconds\":0.012,\"cached\":false,\
             \"command\":\"tet --ate 5 2 '2^^3'\"}\n"
        );
        assert_eq!(rendered(&job, &value, true, Format::Text), "3.0000\n0\n");
        let refused = Outcome::Refused {
            message: "invalid number \"x\": unknown name x".into(),
            seconds: 0.0,
        };
        assert_eq!(
            rendered(&job, &refused, true, Format::Json),
            "{\"func\":\"ate\",\"prec\":5,\"base\":\"2\",\"value\":\"2^^3\",\
             \"error\":\"invalid number \\\"x\\\": unknown name x\",\"status\":422,\"seconds\":0.000,\
             \"cached\":true,\"command\":\"tet --ate 5 2 '2^^3'\"}\n"
        );
        assert_eq!(
            rendered(&job, &refused, false, Format::Text),
            "error: invalid number \"x\": unknown name x\n"
        );
        for (outcome, status, lasting, word) in [
            (
                Outcome::Timeout { limit: 10_800 },
                504,
                true,
                "no result within 3 hours, the limit of this server",
            ),
            (
                Outcome::Busy,
                503,
                false,
                "all slots are busy calculating; try again later",
            ),
            (
                Outcome::Failed {
                    message: "tet could not be run".into(),
                },
                500,
                false,
                "could not be run",
            ),
            (refused.clone(), 422, true, "unknown name"),
            (value.clone(), 200, true, "3.0000"),
        ] {
            assert_eq!((outcome.status(), outcome.lasting()), (status, lasting));
            let body = rendered(&job, &outcome, false, Format::Json);
            assert!(body.contains(word) && body.ends_with("}\n"), "{body}");
            assert_eq!(body.contains("\"status\""), status != 200, "{body}");
        }
        let rejection = rejected("func is missing");
        assert_eq!(
            rendered_rejection(&rejection, Format::Json),
            "{\"error\":\"func is missing\",\"status\":400}\n"
        );
        assert_eq!(
            rendered_rejection(&rejection, Format::Text),
            "error: func is missing\n"
        );
    }

    #[test]
    fn a_computation_that_has_not_ended_is_reported_as_running() {
        let (job, _) = job(&[("func", "ssrt"), ("value", "10+i"), ("height", "0.3-i")]).unwrap();
        let first = Pending {
            waited: true,
            age: Duration::from_millis(120_004),
            wait: 120,
            limit: 10_800,
        };
        let message = "the calculation is taking longer than 2 minutes and continues on this \
                       server: ask the same again later for the result; it may take up to 3 hours";
        assert_eq!(
            rendered_pending(&job, &first, Format::Json),
            format!(
                "{{\"func\":\"ssrt\",\"prec\":20,\"value\":\"10+i\",\"height\":\"0.3-i\",\
                 \"running\":true,\"message\":\"{message}\",\"status\":202,\
                 \"running_seconds\":120.004,\"limit_seconds\":10800,\"cached\":false,\
                 \"command\":\"tet --ssrt 20 10+i 0.3-i\"}}\n"
            )
        );
        assert_eq!(
            rendered_pending(&job, &first, Format::Text),
            format!("running: {message}\n")
        );
        let later = Pending {
            waited: false,
            age: Duration::from_secs(345),
            ..first
        };
        assert_eq!(
            rendered_pending(&job, &later, Format::Text),
            "running: this calculation is already running, for 5 min 45 s now: allow it up to \
             3 hours to complete and ask the same again for the result\n"
        );
    }

    #[test]
    fn the_log_has_the_result_or_the_reason_and_cuts_what_is_long() {
        let value = Outcome::Value {
            re: "1.4587818160364217007".into(),
            im: "0".into(),
            seconds: 1.0,
        };
        assert_eq!(value.logged(), " = 1.4587818160364217007 0");
        let long = Outcome::Value {
            re: format!("1.{}", "4".repeat(300)),
            im: "-2.5e-7".into(),
            seconds: 1.0,
        };
        assert_eq!(
            long.logged(),
            format!(" = 1.{}...(302 characters) -2.5e-7", "4".repeat(98))
        );
        let refused = Outcome::Refused {
            message: "invalid number \"x\": unknown name x".into(),
            seconds: 0.0,
        };
        assert_eq!(refused.logged(), ": invalid number \"x\": unknown name x");
        assert_eq!(
            Outcome::Busy.logged(),
            ": all slots are busy calculating; try again later"
        );
        assert_eq!(brief("πππππ", 5), "πππππ");
        assert_eq!(brief("ππππππ", 5), "πππππ...(6 characters)");
    }

    #[test]
    fn an_entry_of_the_cache_is_one_line_with_its_numbers_and_its_result() {
        let value = |re: &str, im: &str| Outcome::Value {
            re: re.into(),
            im: im.into(),
            seconds: 1.0,
        };
        let refused = Outcome::Refused {
            message: "invalid number \"x\": unknown name x".into(),
            seconds: 0.0,
        };
        let running = Err(Duration::from_millis(150_900));
        let four = [
            ("func", "sroot"),
            ("value_re", "16"),
            ("value_im", "0"),
            ("height_re", "3"),
            ("height_im", "-1/2"),
        ];
        for (pairs, found, line) in [
            (
                &[("func", "ate"), ("base", "2"), ("value", "65536")][..],
                Ok(value("4.0000000000000000000", "0")),
                "f=ate prec=20 base=2 height=4.0000000000000000000 value=65536",
            ),
            (
                &[("func", "slog"), ("value", "65536"), ("base", "2")],
                running.clone(),
                "f=ate prec=20 base=2 value=65536 running=150s",
            ),
            (
                &[
                    ("prec", "5"),
                    ("func", "tet"),
                    ("base", "1.5+0.5i"),
                    ("height", "e^(1/e)"),
                ],
                Ok(value("1.1766", "6.0310e-1")),
                "f=tet prec=5 base=1.5+0.5i height='e^(1/e)' value=1.1766+6.0310e-1i",
            ),
            (
                &[("func", "tet"), ("base", "0.2+0.4i"), ("height", "2.5-i")],
                Ok(value("2.0149853306301", "-9.63361309384064e-1")),
                "f=tet prec=20 base=0.2+0.4i height=2.5-i value=2.0149853306301-9.63361309384064e-1i",
            ),
            (
                &[("func", "ssrt"), ("value", "16"), ("height", "3")],
                Ok(value("2.0000000000000000000", "0")),
                "f=ssrt prec=20 base=2.0000000000000000000 height=3 value=16",
            ),
            (
                &four,
                Ok(value("2.1", "-3e-2")),
                "f=ssrt prec=20 base_re=2.1 base_im=-3e-2 height_re=3 height_im=-1/2 \
                 value_re=16 value_im=0",
            ),
            (
                &four,
                running,
                "f=ssrt prec=20 height_re=3 height_im=-1/2 value_re=16 value_im=0 running=150s",
            ),
            (
                &[("func", "tet"), ("base", "x"), ("height", "pi + 1")],
                Ok(refused),
                "f=tet prec=20 base=x height='pi + 1' \
                 error='invalid number \"x\": unknown name x'",
            ),
            (
                &[("func", "tet"), ("base", "2"), ("height", "0.5")],
                Ok(Outcome::Timeout { limit: 10_800 }),
                "f=tet prec=20 base=2 height=0.5 error='no result within 3 hours, the limit of \
                 this server; tet itself has no limit: build it and run the command'",
            ),
        ] {
            let (asked, _) = job(pairs).unwrap();
            // The key of a job gives the job back, under the name of the page.
            let kept = Job::of_key(&asked.key()).unwrap();
            let expected = Job {
                spelled: asked.func.title().into(),
                ..asked.clone()
            };
            assert_eq!(kept, expected);
            assert_eq!(listed(&kept, found.as_ref().map_err(|age| *age)), line);
        }
        for key in [
            "",
            "Tet",
            "Tet\n20",
            "Tet\n20\n2",
            "Tet\n20\n2\n3\n4",
            "tet\n20\n2\n3",
        ] {
            assert_eq!(Job::of_key(key), None, "{key:?}");
        }
        assert_eq!(Job::of_key("Slog\nx\n2\n3"), None);
    }

    #[test]
    fn seconds_are_spelled_for_people() {
        for (seconds, text) in [
            (0, "0 s"),
            (45, "45 s"),
            (60, "1 minute"),
            (120, "2 minutes"),
            (150, "2 min 30 s"),
            (3_600, "1 hour"),
            (10_800, "3 hours"),
            (5_400, "1 h 30 min"),
            (17_844, "4 h 57 min"),
            (3_605, "1 h 0 min"),
        ] {
            assert_eq!(span(seconds), text);
        }
    }

    #[test]
    fn json_strings_escape_what_json_requires() {
        assert_eq!(
            json_string("a\"b\\c\nd\te\r\u{1}π"),
            "\"a\\\"b\\\\c\\nd\\te\\r\\u0001π\""
        );
    }
}
