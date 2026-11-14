//! From a query to a `tet` command line, and from its outcome to a reply.

use std::fmt::Write as _;

/// What the server allows a request to ask for.
#[derive(Clone, Debug)]
pub struct Limits {
    pub max_prec: u64,
    /// The longest argument, in bytes.
    pub max_argument: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Func {
    Tet,
    Slog,
    Sroot,
}

impl Func {
    fn named(name: &str) -> Option<Func> {
        match name {
            "tet" => Some(Func::Tet),
            "slog" | "ate" => Some(Func::Slog),
            "sroot" | "ssrt" => Some(Func::Sroot),
            _ => None,
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
    pub fn parse(query: &[(String, String)], limits: &Limits) -> Result<(Job, Format), Rejected> {
        let get = |name: &str| {
            query
                .iter()
                .find(|(found, _)| found == name)
                .map(|(_, value)| value.as_str())
        };
        let format = match get("format") {
            None | Some("json") => Format::Json,
            Some("text") => Format::Text,
            Some(other) => return Err(rejected(format!("format is json or text, not {other:?}"))),
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
                .filter(|prec| (1..=limits.max_prec).contains(prec))
                // Only digits: no sign, no space.
                .filter(|_| text.bytes().all(|byte| byte.is_ascii_digit()))
                .ok_or_else(|| {
                    rejected(format!(
                        "prec is the number of digits, a whole number from 1 to {}, not {text:?}",
                        limits.max_prec
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
            if value.len() > limits.max_argument {
                return Err(rejected(format!(
                    "{name} is longer than {} bytes",
                    limits.max_argument
                )));
            }
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
    /// No free slot within the time a request may wait.
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

    /// A finished run of `tet` is the same whenever it is repeated; the
    /// other outcomes depend on the load of the moment.
    pub fn lasting(&self) -> bool {
        matches!(self, Outcome::Value { .. } | Outcome::Refused { .. })
    }

    fn error(&self) -> Option<String> {
        match self {
            Outcome::Value { .. } => None,
            Outcome::Refused { message, .. } | Outcome::Failed { message } => Some(message.clone()),
            Outcome::Timeout { limit } => Some(format!(
                "no result within {limit} s, the limit of this server; tet itself has no limit: \
                 build it and run the command"
            )),
            Outcome::Busy => Some("the server is busy with other computations; try again".into()),
        }
    }

    fn seconds(&self) -> Option<f64> {
        match self {
            Outcome::Value { seconds, .. } | Outcome::Refused { seconds, .. } => Some(*seconds),
            _ => None,
        }
    }
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

/// The body of the reply to a job.
pub fn rendered(job: &Job, outcome: &Outcome, cached: bool, format: Format) -> String {
    if format == Format::Text {
        return match outcome {
            Outcome::Value { re, im, .. } => format!("{re}\n{im}\n"),
            other => format!("error: {}\n", other.error().unwrap_or_default()),
        };
    }
    let mut fields = vec![
        format!("\"func\":{}", json_string(&job.spelled)),
        format!("\"prec\":{}", job.prec),
    ];
    for (name, value) in &job.arguments {
        fields.push(format!("{}:{}", json_string(name), json_string(value)));
    }
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

    fn limits() -> Limits {
        Limits {
            max_prec: 1000,
            max_argument: 64,
        }
    }

    fn query(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    fn job(pairs: &[(&str, &str)]) -> Result<(Job, Format), Rejected> {
        Job::parse(&query(pairs), &limits())
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
        let long = "1".repeat(65);
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
                "format is json or text, not \"xml\"".to_string(),
            ),
            (
                &[("func", "tet"), ("base", &long), ("height", "3")],
                "base is longer than 64 bytes".to_string(),
            ),
        ] {
            let error = job(pairs).unwrap_err();
            assert_eq!((error.status, error.message), (400, message), "{pairs:?}");
        }
        for prec in [
            "0",
            "1001",
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
                    .starts_with("prec is the number of digits, a whole number from 1 to 1000"),
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
                Outcome::Timeout { limit: 60 },
                504,
                false,
                "no result within 60 s",
            ),
            (Outcome::Busy, 503, false, "busy"),
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
    fn json_strings_escape_what_json_requires() {
        assert_eq!(
            json_string("a\"b\\c\nd\te\r\u{1}π"),
            "\"a\\\"b\\\\c\\nd\\te\\r\\u0001π\""
        );
    }
}
