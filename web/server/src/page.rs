//! The page, as the server renders it. It is a plain form, so that it works
//! without JavaScript: sending the form asks for this page again, with the
//! result in it. Where scripts run, `app.js` asks for that page itself and
//! puts its result into the page that is shown, which saves the reloads.

use std::fmt::Write as _;

use crate::api::{self, Func, Job, Outcome, Pending};
use crate::sm::{self, Program};

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
    /// The examples of the help of `symbolic-math`: their lines and their
    /// headings.
    pub examples: (usize, usize),
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

/// The form of `symbolic-math`: the inputs and the options as they were
/// typed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Asked {
    pub expr: String,
    pub digits: String,
    /// `both`, `numeric` or `exact`.
    pub mode: String,
    /// `complex`, `real` or `quaternion`.
    pub domain: String,
    /// Whether the request had inputs, and so asks for a run.
    pub complete: bool,
    /// How many times the page has asked again for a run that has not
    /// ended.
    pub again: u64,
}

/// The inputs that the page starts with.
const FIRST: &str = "x^2 - 5x + 6 = 0\nintegrate(x^2*e^(-x), x, 0, inf)\nN(pi, 50)";

/// What the page offers to run at a click: a name and the inputs.
const EXAMPLES: [(&str, &str); 56] = [
    ("simplify", "10*x - (2+3)*x\nsin(x)^2 + cos(x)^2\n(x^2 - 1)/(x - 1)"),
    ("solve", "x^2 - 5x + 6 = 0\nsin(x) = 0\nx^5 + x = i"),
    ("a system", "x + y = 3, x - y = 1\nsolve([x^2 + y^2 = 25, x - y = 1], [x, y])"),
    ("inequalities", "x^2 - 5x + 6 < 0\nabs(x - 1) <= 2"),
    ("derivative", "diff(x^3*sin(x), x)\ndiff(e^(x*y), x, y)"),
    ("integral", "integrate(x^2*e^(-x), x, 0, inf)\nintegrate(1/(1 + x^2), x)"),
    ("limit and series", "limit(sin(x)/x, x, 0)\nseries(e^x, x, 0, 6)\nsum(1/k^2, k, 1, inf)"),
    ("differential equation", "dsolve(y'' + y = 0, y, x)"),
    (
        "transforms",
        "laplace(t*e^(-t), t, s)\nfourier(e^(-x^2), x, w)\nztrans(n, n, z)",
    ),
    (
        "special functions",
        "gamma(1/2)\nzeta(2)\nbesselj(0, 1.5)\nerf(1/2)",
    ),
    ("matrices", "A := [[2, 1], [1, 3]]\ndet(A)\ninv(A)\neig(A)"),
    ("quaternions", "domain(quaternion)\ni*j\nx^2 = -1"),
    ("1000 digits of pi", "N(pi, 1000)"),
    ("number theory", "factor(2^64 + 1)\nisprime(2^127 - 1)\ngcd(1071, 462)"),
    ("tetration", "2^^0.5\nslog(2, 65536)\n2^^x = 65536"),
    (
        "tetration as a chart",
        "plot(sqrt(2)^^x, x, -1.5, 4)\nplot([slog(sqrt(2), x), ssrt(x, 2)], x, 1.1, 1.9)",
    ),
    (
        "tetration in colours (PNG)",
        "save_png(\"tetration.png\", complexplot(sqrt(2)^^z, z, -4 - 2i, 4 + 2i), width = 160)",
    ),
    (
        "half iterate",
        "iterate(2x + 1, x, 1/2)\niterate(x^2, x, 1/2)\niterate(e^x - 1, x, 1/2, 1)",
    ),
    ("bases", "base_conv(pi, 16)\nbase_conv(1/7, 10, 3)\n0.(3) + 0.1(6)"),
    ("statistics", "mean([2, 4, 4, 4, 5, 5, 7, 9])\nstd([2, 4, 4, 4, 5, 5, 7, 9])"),
    ("units", "convert(100 km/h, m/s)"),
    (
        "systems of control",
        "s := tf(\"s\")\ng := 1/(s^2 + 3s + 2)\nstep(g)\nmargin(4/(s + 1)^3)\nhsvd(g)",
    ),
    (
        "filters",
        "butter(2, 0.3)\nfirpm(6, [0, 0.4, 0.6, 1], [1, 1, 0, 0])",
    ),
    (
        "the earth",
        "distance(52.2297, 21.0122, 40.7128, -74.006, \"wgs84\")",
    ),
    ("a chart as text", "plot(sin(x)/x, x, -20, 20)"),
    ("a chart as PNG", "save_png(\"plot.png\", plot(sin(x)/x, x, -20, 20), width = 900, height = 500)"),
    ("the walk of pi (PNG)", "save_png(\"walk.png\", walk(pi, 10, 20000), width = 900)"),
    ("a walk in space (JPEG XL)", "save_jxl(\"walk3d.jxl\", walk3d(e + pi*i, 12, 5000), width = 900)\nsave_png(\"walk3d.png\", walk3d(e + pi*i, 12, 5000), width = 900)"),
    ("primes (PNG)", "save_png(\"ulam.png\", ulamspiral(40000), width = 800)"),
    ("a surface (PNG)", "save_png(\"surface.png\", plot3d(sin(x)*cos(y), x, -3, 3, y, -3, 3), width = 900)"),
    ("a Bode chart (PNG)", "save_png(\"bode.png\", bodeplot(tf([1], [1, 0.4, 1])), width = 900, height = 700)"),
    (
        "a function as a 4D shape (moving PNG)",
        "save_png_fast(\"sin4d.png\", plot4d(sin(z)))",
    ),
    (
        "the same turned, as JPEG XL",
        "save_jxl_fast(\"sin4d.jxl\", plot4d(sin(z)), rotate(30, 20, 10))\nsave_png_fast(\"sin4d_turned.png\", plot4d(sin(z)), rotate(30, 20, 10))",
    ),
    (
        "every frame stretched (normalize)",
        "save_png_fast(\"gamma4d.png\", plot4d(gamma(z), z, 3), normalize, palette = heat)",
    ),
    (
        "a filled 4D body",
        "save_png_fast(\"body4d.png\", plot4d(z^2), fill = below, palette = magma)",
    ),
    (
        "a body between Re f(z) and Im f(z) (moving PNG)",
        "save_png_fast(\"between.png\", plot4d(sin(z), z, pi), fill = between)",
    ),
    (
        "the same body seen from outside (PNG)",
        "save_png(\"between_view.png\", view4d(sin(z), z, pi), fill = between, width = 900)",
    ),
    (
        "the 4D shape seen from outside (PNG)",
        "save_png(\"view.png\", view4d(sin(z)), width = 900)",
    ),
    (
        "mountains of the gamma function (PNG)",
        "save_png(\"gamma_view.png\", view4d(gamma(z), z, -4 - 3i, 4 + 3i), fill = below, palette = heat, perspective = 3, width = 600, voxels = 160)",
    ),
    (
        "the eye goes around a body (moving PNG)",
        "save_png_fast(\"around.png\", view4d((z^3 - 1)/(z^3 + 1), z, 2), fill = below, palette = magma, frames = 24)",
    ),
    (
        "several inputs and outputs",
        "H := [[1/(s + 1), 2/(s + 2)], [0, 1/(s + 3)]]\npole(H)\ntzero(H)\nstep(H)",
    ),
    (
        "a delay in a loop",
        "G := e^(-s)/(s + 1)\nstep(feedback(G, 1), [0, 1, 2, 3])\nmargin(4*e^(-s/4)/(s + 1))",
    ),
    (
        "a diagram of blocks",
        "r := sim([[\"u\", \"step\", 0], [\"e\", \"sum\", \"u\", \"-y\"], [\"v\", \"gain\", 2, \"e\"], [\"y\", \"tf\", [1], [1, 1], \"v\"]], 2, \"points\", 5)\ngetfield(r, \"y\")",
    ),
    ("the geoid", "egm96geoid(52, 21)\ngeoidheight(27.988, 86.925)"),
    (
        "solids of four dimensions",
        "volume4d(solid4d(\"tiger\", 2, 3, 1/2))\nlength(edges4d(solid4d(\"120-cell\")))\ndensity4d(solid4d(\"grand 600-cell\"), [0.001, 0.002, -0.001, 0.0005])",
    ),
    (
        "a tesseract turning in 4D (moving PNG)",
        "save_png(\"tesseract.png\", wire4d(solid4d(\"tesseract\"), \"xw\", \"yz\"), frames = 24, width = 400)",
    ),
    (
        "the 600-cell as a wire frame (PNG)",
        "save_png(\"cell600.png\", wire4d(solid4d(\"600-cell\")), width = 900)",
    ),
    (
        "a 4D solid as a shape (moving PNG)",
        "save_png_fast(\"solid.png\", plot4d(rotate4d(solid4d(\"tesseract\"), \"xw\", 30)))",
    ),
    (
        "a star polychoron seen from outside (PNG)",
        "save_png(\"star.png\", view4d(solid4d(\"great grand stellated 120-cell\")), width = 600, voxels = 128)",
    ),
    (
        "a walk of chance (PNG)",
        "save_png(\"chance.png\", randomwalk(20000, \"any\", 7), width = 900)",
    ),
    (
        "cell arrays and MAT-files",
        "c := {1, \"two\", [3, 4]}\ncellfun(length, {[1, 2], [3], []})\nmatwrite(\"v.mat\", \"x\", [[1, 2], [3, 4]], \"c\", c)\nmatread(\"v.mat\")",
    ),
    (
        "pictures as matrices",
        "bwlabel([[1, 0, 1], [1, 0, 0], [0, 0, 1]])\nimdilate([[0, 0, 0], [0, 1, 0], [0, 0, 0]], strel(\"disk\", 1))\ngraythresh([[0, 0, 255, 255], [0, 10, 250, 255]])",
    ),
    ("MATLAB's random numbers", "rng(0)\nrandn(1, 5)\nrng(0)\nrandperm(6)"),
    (
        "laws of whole numbers, lists of parameters",
        "rng(7)\npoissrnd([1, 2, 3])\npoissrnd(1000000)\npoisspdf(2, [1, 2])\nbinocdf(50000, 100000, 1/2)",
    ),
    (
        "a system sampled",
        "G := tf([1], [1, 2, 5])\nc2d(G, 1/10)\nc2d(G, 1/10, \"matched\")\nc2d(tf([1], [1, a]), 1/10, \"matched\")\nroots([1, a, b])",
    ),
    (
        "a network of windows",
        "net := dlnetwork([imageinputlayer([4, 4]), convolution2dlayer(2, 2), relulayer(), maxpooling2dlayer(2), flattenlayer(), fullyconnectedlayer(2), softmaxlayer()])\npredict(net, [[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]])",
    ),
];

impl Asked {
    /// The form as a query fills it; without inputs it has the ones that
    /// the page starts with, and asks for nothing.
    pub fn from_query(query: &[(String, String)]) -> Asked {
        let get = |name: &str| {
            query
                .iter()
                .find(|(found, _)| found == name)
                .map(|(_, value)| value.as_str())
        };
        let known = |value: Option<&str>, names: [&str; 3]| {
            value
                .filter(|value| names.contains(value))
                .unwrap_or(names[0])
                .to_string()
        };
        let expr = get("expr").map(sm::tidy);
        Asked {
            complete: expr.as_ref().is_some_and(|expr| !expr.trim().is_empty()),
            expr: expr.unwrap_or_else(|| FIRST.to_string()),
            digits: get("digits").unwrap_or("").trim().to_string(),
            mode: known(get("mode"), ["both", "numeric", "exact"]),
            domain: known(get("domain"), ["complex", "real", "quaternion"]),
            again: get("again")
                .and_then(|count| count.parse().ok())
                .unwrap_or(0),
        }
    }

    /// The query of the API for what the form asks.
    pub fn api_query(&self) -> Vec<(String, String)> {
        let mut query = vec![("expr".to_string(), self.expr.clone())];
        if !self.digits.is_empty() {
            query.push(("digits".to_string(), self.digits.clone()));
        }
        if self.mode != "both" {
            query.push(("mode".to_string(), self.mode.clone()));
        }
        if self.domain != "complex" {
            query.push(("domain".to_string(), self.domain.clone()));
        }
        query
    }

    fn api_address(&self, readable: bool) -> String {
        let pairs: Vec<String> = self
            .api_query()
            .iter()
            .map(|(name, value)| format!("{name}={}", in_query(value, readable)))
            .collect();
        format!("/sm?{}", pairs.join("&"))
    }

    /// The address of the page for what the form asks.
    pub fn page_address(&self) -> String {
        format!(
            "/sm?format=html&expr={}&digits={}&mode={}&domain={}",
            in_query(&self.expr, false),
            in_query(&self.digits, false),
            self.mode,
            self.domain
        )
    }

    fn again_address(&self) -> String {
        format!(
            "{}&again={}",
            self.page_address(),
            self.again.saturating_add(1)
        )
    }
}

/// What the page shows below the form of `symbolic-math`.
pub enum Answer<'a> {
    Nothing,
    /// The program is not on this server.
    Absent,
    /// The outcome of the run that the form asked for.
    Outcome {
        program: &'a Program,
        outcome: &'a Outcome,
        cached: bool,
    },
    /// The run has not ended.
    Running {
        program: &'a Program,
        pending: &'a Pending,
    },
    /// The form did not ask for a run: the reason and the status that the
    /// API has for it.
    Rejected {
        message: &'a str,
        status: u16,
    },
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

/// What stands below the form of `symbolic-math`, as the inside of its
/// section, and whether it is about a run that has not ended.
fn answer(asked: &Asked, answer: &Answer) -> (String, bool) {
    let mut html = String::new();
    let title = |html: &mut String, text: &str| {
        let _ = writeln!(html, "  <h2 id=\"sm-title\">{}</h2>", escaped(text));
    };
    let failed = |html: &mut String, message: &str, status: u16| {
        title(html, "No result");
        let _ = writeln!(
            html,
            "  <p id=\"sm-error\" class=\"error\">{} (HTTP {status})</p>",
            escaped(message)
        );
    };
    let command = |program: &Program| {
        format!(
            "command: <code id=\"sm-command\">{}</code>",
            escaped(&program.command())
        )
    };
    let mut lines = Vec::new();
    let mut running = false;
    match answer {
        Answer::Nothing => return (html, false),
        Answer::Absent => {
            failed(
                &mut html,
                "symbolic-math is not installed on this server",
                503,
            );
            return (html, false);
        }
        Answer::Outcome {
            program,
            outcome,
            cached,
        } => {
            match outcome {
                Outcome::Output {
                    text, files, clean, ..
                } => {
                    let id = program.id();
                    title(
                        &mut html,
                        if *clean {
                            "Result"
                        } else {
                            "Result, with errors"
                        },
                    );
                    let _ = writeln!(html, "  <pre id=\"sm-output\">{}</pre>", escaped(text));
                    if !files.is_empty() {
                        html.push_str("  <ul id=\"sm-files\" class=\"files\">\n");
                        for file in files {
                            let address = format!("/sm/file/{id}/{}", file.name);
                            let (kind, picture) = sm::kind(&file.name);
                            let what = match kind {
                                "image/png" => "PNG picture",
                                "image/jxl" => {
                                    "JPEG XL picture; few browsers show this format: download it"
                                }
                                _ => "file",
                            };
                            let _ = write!(
                                html,
                                "    <li><a href=\"{address}?download=1\">{}</a> ({what}, {})",
                                file.name,
                                sm::size(file.bytes.len())
                            );
                            if picture {
                                let _ = write!(
                                    html,
                                    "\n      <img class=\"picture\" src=\"{address}\" alt=\"{}\">",
                                    file.name
                                );
                            }
                            html.push_str("</li>\n");
                        }
                        html.push_str("  </ul>\n");
                    }
                    lines.push(format!(
                        "<a id=\"sm-download\" href=\"/sm/output/{id}\">Download the output</a> \
                         as a text file"
                    ));
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
                lines.insert(
                    0,
                    format!(
                        "<span id=\"sm-timing\">{}</span>",
                        escaped(&timing.join(", "))
                    ),
                );
            }
            lines.push(command(program));
        }
        Answer::Running { program, pending } => {
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
                "  <p id=\"sm-note\" class=\"note\">{}. This page asks again every {AGAIN} \
                 seconds; in a browser that does not, follow the link.</p>\n  \
                 <p id=\"sm-again\"><a href=\"{}\">Ask again now</a></p>",
                escaped(&format!("{first}{}", letters.as_str())),
                escaped(&asked.again_address())
            );
            lines.push(command(program));
        }
        Answer::Rejected { message, status } => failed(&mut html, message, *status),
    }
    lines.push(format!(
        "API: <a id=\"sm-api\" href=\"{}\">{}</a>",
        escaped(&asked.api_address(false)),
        escaped(&asked.api_address(true))
    ));
    let _ = writeln!(
        html,
        "  <p class=\"hint\">\n    {}\n  </p>",
        lines.join("<br>\n    ")
    );
    (html, running)
}

/// The examples of `symbolic-math` as links that run them.
fn examples() -> String {
    let links: Vec<String> = EXAMPLES
        .iter()
        .map(|(name, inputs)| {
            format!(
                "  <a class=\"example\" href=\"/sm?format=html&amp;expr={}\">{}</a>",
                in_query(inputs, false),
                escaped(name)
            )
        })
        .collect();
    format!("{}\n", links.join(",\n"))
}

/// The address of the page that runs `program`, at its result.
fn run_address(program: &Program) -> String {
    let mut address = format!("/sm?format=html&expr={}", in_query(&program.text, false));
    if let Some(digits) = program.digits {
        let _ = write!(address, "&digits={digits}");
    }
    if program.mode != sm::Mode::Both {
        let _ = write!(address, "&mode={}", program.mode.name());
    }
    if program.domain != sm::Domain::Complex {
        let _ = write!(address, "&domain={}", program.domain.name());
    }
    address.push_str("#sm-result");
    address
}

/// All the examples of the help of `symbolic-math` as a page of their own:
/// under every heading its lines, each a link that runs the line on the
/// page of the programs. The page of the programs shows it in a frame.
pub fn examples_page(groups: &[sm::Group]) -> String {
    let lines: usize = groups.iter().map(|group| group.examples.len()).sum();
    let mut html = format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>symbolic-math: the examples of its help</title>\n\
         <link rel=\"icon\" type=\"image/svg+xml\" href=\"/favicon.svg\">\n\
         <link rel=\"stylesheet\" href=\"/style.css\">\n</head>\n<body>\n<main class=\"examples\">\n\
         <p class=\"hint\">The {lines} lines of examples of \
         <a href=\"/sm/help\" target=\"_top\">the help of <code>symbolic-math</code></a> under \
         their {} headings: a heading opens, and a click on a line runs it on \
         <a href=\"/sm\" target=\"_top\">the page</a>.</p>\n",
        groups.len()
    );
    for group in groups {
        let _ = writeln!(
            html,
            "<details>\n<summary>{} <small>({})</small></summary>\n<ul>",
            escaped(&group.title),
            group.examples.len()
        );
        for example in &group.examples {
            // The line as the help has it: its options, then its inputs.
            let mut line = example.program.options().join(" ");
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(&example.program.text.replace('\n', "; "));
            let _ = write!(
                html,
                "<li><a class=\"example\" target=\"_top\" href=\"{}\">{}</a>",
                escaped(&run_address(&example.program)),
                escaped(&line)
            );
            if !example.shows.is_empty() {
                let _ = write!(html, " <small>{}</small>", escaped(&example.shows));
            }
            html.push_str("</li>\n");
        }
        html.push_str("</ul>\n</details>\n");
    }
    html.push_str("</main>\n</body>\n</html>\n");
    html
}

/// The place of all the examples of the help in the page: closed it is a
/// line, open a box that scrolls. Without examples there is none.
fn all_examples(site: &Site) -> String {
    let (lines, headings) = site.examples;
    if lines == 0 {
        return String::new();
    }
    format!(
        "<details id=\"sm-all\">\n<summary>All {lines} lines of examples of the help, under \
         {headings} headings (a click runs a line)</summary>\n\
         <iframe id=\"sm-all-frame\" src=\"/sm/examples\" loading=\"lazy\" \
         title=\"All the examples of the help of symbolic-math\"></iframe>\n</details>\n"
    )
}

/// A text of `symbolic-math` as a page: its help, or one of its lists.
pub fn text_page(title: &str, text: &str) -> String {
    format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>symbolic-math: {title}</title>\n\
         <link rel=\"icon\" type=\"image/svg+xml\" href=\"/favicon.svg\">\n\
         <link rel=\"stylesheet\" href=\"/style.css\">\n</head>\n<body>\n<main class=\"help\">\n\
         <h1>symbolic-math <small>{title}</small></h1>\n\
         <p><a href=\"/sm\">The page that runs it</a>; <a href=\"/sm/help\">the help</a> \
         (<a href=\"/sm/help.txt\">as text</a>); the lists of the \
         <a href=\"/sm/functions\">functions</a>, <a href=\"/sm/constants\">constants</a> and \
         <a href=\"/sm/units\">units</a>; \
         <a href=\"https://github.com/lukaszgryglicki/symbolic-math\">the source</a>.</p>\n\
         <pre>{}</pre>\n</main>\n</body>\n</html>\n",
        escaped(text)
    )
}

/// The page with the form as given and with what is shown below it; the
/// part of `symbolic-math` as it starts.
#[cfg(test)]
pub fn rendered(site: &Site, form: &Form, shown: &Shown) -> String {
    rendered_both(site, form, shown, &Asked::from_query(&[]), &Answer::Nothing)
}

/// The page with both forms as given and with what is shown below them.
pub fn rendered_both(
    site: &Site,
    form: &Form,
    shown: &Shown,
    asked: &Asked,
    replied: &Answer,
) -> String {
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
    let (mut answered, answering) = answer(asked, replied);
    if !answered.is_empty() {
        answered.insert(0, '\n');
    }
    let asks_again = |address: String| {
        format!(
            "<noscript><meta http-equiv=\"refresh\" content=\"{AGAIN}; url={}\"></noscript>\n",
            escaped(&address)
        )
    };
    let marked = |on: bool| match on {
        true => format!(" data-running=\"{AGAIN}\""),
        false => String::new(),
    };
    let refresh = match (running, answering) {
        (true, _) => asks_again(form.again_address()),
        (false, true) => asks_again(asked.again_address()),
        (false, false) => String::new(),
    };
    let again = marked(running);
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
        ("sm.expr", escaped(&asked.expr)),
        ("sm.digits", escaped(&asked.digits)),
        ("sm.mode.both", checked(asked.mode == "both")),
        ("sm.mode.numeric", checked(asked.mode == "numeric")),
        ("sm.mode.exact", checked(asked.mode == "exact")),
        ("sm.domain.complex", checked(asked.domain == "complex")),
        ("sm.domain.real", checked(asked.domain == "real")),
        (
            "sm.domain.quaternion",
            checked(asked.domain == "quaternion"),
        ),
        ("sm.examples", examples()),
        ("sm.all", all_examples(site)),
        ("sm.running", marked(answering)),
        ("sm.result", answered),
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
        examples: (0, 0),
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
    fn all_the_examples_of_the_help_are_links_that_run_them() {
        let run = |text: &str, digits, mode, domain| Program {
            text: text.to_string(),
            digits,
            mode,
            domain,
        };
        let (both, complex) = (sm::Mode::Both, sm::Domain::Complex);
        let groups = [
            sm::Group {
                title: "Sums & <more>".to_string(),
                examples: vec![
                    sm::Example {
                        program: run("1+1\nx^2 = \"4\"", None, both, complex),
                        shows: "two <inputs>".to_string(),
                    },
                    sm::Example {
                        program: run("pi", Some(50), sm::Mode::Numeric, sm::Domain::Real),
                        shows: String::new(),
                    },
                ],
            },
            sm::Group {
                title: "Quaternions".to_string(),
                examples: vec![sm::Example {
                    program: run("i*j", None, sm::Mode::Exact, sm::Domain::Quaternion),
                    shows: "k".to_string(),
                }],
            },
        ];
        let page = examples_page(&groups);
        for part in [
            "<main class=\"examples\">\n<p class=\"hint\">The 3 lines of examples of ",
            " under their 2 headings: ",
            "<details>\n<summary>Sums &amp; &lt;more&gt; <small>(2)</small></summary>\n<ul>\n",
            "<li><a class=\"example\" target=\"_top\" href=\"/sm?format=html&amp;expr=1%2B1%0Ax%5E2%20%3D%20%224%22#sm-result\">1+1; x^2 = &quot;4&quot;</a> <small>two &lt;inputs&gt;</small></li>\n",
            "<li><a class=\"example\" target=\"_top\" href=\"/sm?format=html&amp;expr=pi&amp;digits=50&amp;mode=numeric&amp;domain=real#sm-result\">-d 50 -n -r pi</a></li>\n</ul>\n</details>\n",
            "<summary>Quaternions <small>(1)</small></summary>",
            "href=\"/sm?format=html&amp;expr=i*j&amp;mode=exact&amp;domain=quaternion#sm-result\">-e -q i*j</a> <small>k</small></li>",
        ] {
            assert!(page.contains(part), "{part}\n{page}");
        }
        assert!(!page.contains("<script"));
        // The page of the programs has them in a frame, when there are any.
        let none = rendered(&SITE, &Form::from_query(&[]), &Shown::Nothing);
        assert!(!none.contains("sm-all"));
        let site = Site {
            examples: (3, 2),
            ..SITE
        };
        let some = rendered(&site, &Form::from_query(&[]), &Shown::Nothing);
        assert!(some.contains(
            "</p>\n<details id=\"sm-all\">\n<summary>All 3 lines of examples of the help, under 2 \
             headings (a click runs a line)</summary>\n<iframe id=\"sm-all-frame\" \
             src=\"/sm/examples\" loading=\"lazy\" title=\"All the examples of the help of \
             symbolic-math\"></iframe>\n</details>\n\n<section id=\"sm-result\">"
        ), "{some}");
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
                "sm.all",
                "sm.digits",
                "sm.domain.complex",
                "sm.domain.quaternion",
                "sm.domain.real",
                "sm.examples",
                "sm.expr",
                "sm.mode.both",
                "sm.mode.exact",
                "sm.mode.numeric",
                "sm.result",
                "sm.running",
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
