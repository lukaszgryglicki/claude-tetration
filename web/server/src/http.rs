//! The little of HTTP/1.1 that the page and the API need: one request per
//! connection, its head read with a deadline and a size limit, no body.

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

/// The largest request head that is read.
pub const HEAD_LIMIT: usize = 16 * 1024;
const HEAD_TIME: Duration = Duration::from_secs(15);
const WRITE_TIME: Duration = Duration::from_secs(15);

#[derive(Debug)]
pub struct Request {
    pub method: String,
    /// The path as it was sent, without the query.
    pub path: String,
    /// The text after the first `?`, still encoded.
    pub query: Option<String>,
    /// Header names in lower case.
    pub headers: Vec<(String, String)>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(found, _)| found == name)
            .map(|(_, value)| value.as_str())
    }
}

/// Why no request was read: nothing to answer, or a status to answer with.
#[derive(Debug, PartialEq)]
pub enum Unread {
    Closed,
    Status(u16, &'static str),
}

/// Reads the head of one request.
pub fn read_request(stream: &mut TcpStream) -> Result<Request, Unread> {
    let started = Instant::now();
    let mut head = Vec::with_capacity(1024);
    let mut chunk = [0u8; 2048];
    let end = loop {
        // The limit is that of the head itself, however the network cut it
        // into pieces.
        let end = find(&head, b"\r\n\r\n");
        if end.unwrap_or(head.len()) > HEAD_LIMIT {
            return Err(Unread::Status(431, "the request head is too large"));
        }
        if let Some(end) = end {
            break end;
        }
        let left = HEAD_TIME
            .checked_sub(started.elapsed())
            .filter(|left| !left.is_zero())
            .ok_or(Unread::Status(408, "the request took too long to arrive"))?;
        stream
            .set_read_timeout(Some(left))
            .map_err(|_| Unread::Closed)?;
        match stream.read(&mut chunk) {
            Ok(0) => return Err(Unread::Closed),
            Ok(count) => head.extend_from_slice(&chunk[..count]),
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                return Err(Unread::Status(408, "the request took too long to arrive"))
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => return Err(Unread::Closed),
        }
    };
    parse_head(&head[..end])
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// The request line and the headers, without the blank line that ends them.
pub fn parse_head(head: &[u8]) -> Result<Request, Unread> {
    let bad = |reason| Unread::Status(400, reason);
    let text = String::from_utf8_lossy(head);
    let mut lines = text.split("\r\n");
    let line = lines.next().unwrap_or("");
    let mut parts = line.split(' ');
    let (Some(method), Some(target), Some(version), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(bad("the request line is malformed"));
    };
    if !version.starts_with("HTTP/1.") {
        return Err(Unread::Status(505, "only HTTP/1.x is spoken here"));
    }
    if method.is_empty() || !method.bytes().all(|byte| byte.is_ascii_uppercase()) {
        return Err(bad("the request method is malformed"));
    }
    // A proxy may send the absolute form: keep what follows the authority.
    let target = match target
        .strip_prefix("http://")
        .or_else(|| target.strip_prefix("https://"))
    {
        Some(rest) => rest.find('/').map_or("/", |at| &rest[at..]),
        None => target,
    };
    // The target goes into the log as it is: nothing in it may be a line
    // or a command of its own there.
    let printable = target.bytes().all(|byte| byte.is_ascii_graphic());
    if !target.starts_with('/') || !printable {
        return Err(bad("the request target is malformed"));
    }
    let target = target.split('#').next().unwrap_or(target);
    let (path, query) = match target.split_once('?') {
        Some((path, query)) => (path, Some(query.to_string())),
        None => (target, None),
    };
    let mut headers = Vec::new();
    for line in lines {
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| bad("a header is malformed"))?;
        headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
    }
    Ok(Request {
        method: method.to_string(),
        path: path.to_string(),
        query,
        headers,
    })
}

/// `%XX` escapes decoded. A `+` stays a plus sign: the values are
/// mathematical expressions, and `base=2+i` typed into a browser means
/// `2+i`, not `2 i`. `None` for a broken escape or bytes that are not UTF-8.
pub fn percent_decode(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut at = 0;
    while at < bytes.len() {
        if bytes[at] == b'%' {
            // Two hexadecimal digits: `from_str_radix` alone would take a sign.
            let digits = text.get(at + 1..at + 3)?;
            if !digits.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return None;
            }
            decoded.push(u8::from_str_radix(digits, 16).ok()?);
            at += 3;
        } else {
            decoded.push(bytes[at]);
            at += 1;
        }
    }
    String::from_utf8(decoded).ok()
}

/// The parameters of a query string in the order written.
pub fn parse_query(raw: &str) -> Result<Vec<(String, String)>, String> {
    let mut found: Vec<(String, String)> = Vec::new();
    for piece in raw.split('&').filter(|piece| !piece.is_empty()) {
        let (name, value) = piece.split_once('=').unwrap_or((piece, ""));
        let decoded = |text: &str| {
            percent_decode(text)
                .ok_or_else(|| format!("the query has a broken escape in {piece:?}"))
        };
        let (name, value) = (decoded(name)?, decoded(value)?);
        if found.iter().any(|(seen, _)| *seen == name) {
            return Err(format!("the parameter {name} is given twice"));
        }
        found.push((name, value));
    }
    Ok(found)
}

pub struct Response {
    pub status: u16,
    pub content_type: &'static str,
    pub headers: Vec<(&'static str, String)>,
    pub body: Vec<u8>,
}

impl Response {
    pub fn new(status: u16, content_type: &'static str, body: impl Into<Vec<u8>>) -> Response {
        Response {
            status,
            content_type,
            headers: Vec::new(),
            body: body.into(),
        }
    }

    pub fn text(status: u16, body: impl Into<String>) -> Response {
        Response::new(status, "text/plain; charset=utf-8", body.into())
    }

    pub fn with(mut self, name: &'static str, value: impl Into<String>) -> Response {
        self.headers.push((name, value.into()));
        self
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        202 => "Accepted",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        422 => "Unprocessable Content",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        505 => "HTTP Version Not Supported",
        _ => "Status",
    }
}

/// Writes the response. A HEAD request gets the headers of the GET
/// response without its body.
pub fn write_response(stream: &mut TcpStream, response: &Response, body: bool) -> io::Result<()> {
    stream.set_write_timeout(Some(WRITE_TIME))?;
    let mut head = format!(
        "HTTP/1.1 {} {}\r\nServer: tet-web\r\nConnection: close\r\nContent-Type: {}\r\n\
         Content-Length: {}\r\nX-Content-Type-Options: nosniff\r\n",
        response.status,
        reason(response.status),
        response.content_type,
        response.body.len()
    );
    for (name, value) in &response.headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    if body {
        stream.write_all(&response.body)?;
    }
    stream.flush()
}

/// Ends a connection whose request was answered before it was read to its
/// end: the sending side is closed, and what the client still sends is read
/// and dropped for a moment. Closing with input left unread resets the
/// connection, and the client can lose the reply.
pub fn finish_unread(stream: &mut TcpStream) {
    const TIME: Duration = Duration::from_secs(2);
    const BYTES: usize = 1 << 20;
    let _ = stream.shutdown(std::net::Shutdown::Write);
    let _ = stream.set_read_timeout(Some(Duration::from_millis(250)));
    let started = Instant::now();
    let (mut dropped, mut chunk) = (0, [0u8; 4096]);
    while started.elapsed() < TIME && dropped < BYTES {
        match stream.read(&mut chunk) {
            Ok(count) if count > 0 => dropped += count,
            _ => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plus_is_a_plus_and_escapes_are_decoded() {
        for (raw, decoded) in [
            ("2+i", "2+i"),
            ("2%2Bi", "2+i"),
            ("pi%20%2B%201", "pi + 1"),
            ("e%5E-e", "e^-e"),
            ("%CF%80", "π"),
            ("", ""),
            ("100%25", "100%"),
        ] {
            assert_eq!(percent_decode(raw).as_deref(), Some(decoded), "{raw}");
        }
        for raw in ["%", "%2", "%zz", "%+1", "%FF", "%C3%28", "a%"] {
            assert_eq!(percent_decode(raw), None, "{raw}");
        }
    }

    #[test]
    fn queries_keep_their_order_and_refuse_duplicates() {
        let query = parse_query("prec=20&func=ate&base=2&value=1.5+0.5i&&flag").unwrap();
        let pairs: Vec<(&str, &str)> = query
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        assert_eq!(
            pairs,
            [
                ("prec", "20"),
                ("func", "ate"),
                ("base", "2"),
                ("value", "1.5+0.5i"),
                ("flag", "")
            ]
        );
        assert_eq!(parse_query("").unwrap(), []);
        assert_eq!(
            parse_query("a=1&a=2").unwrap_err(),
            "the parameter a is given twice"
        );
        assert!(parse_query("a=%G1").unwrap_err().contains("broken escape"));
        // An `=` inside a value belongs to the value.
        assert_eq!(parse_query("a=b=c").unwrap(), [("a".into(), "b=c".into())]);
    }

    #[test]
    fn request_heads_are_parsed_or_refused() {
        let request = parse_head(
            b"GET /?func=tet&base=2 HTTP/1.1\r\nHost: tet.sh\r\nX-Forwarded-For: 1.2.3.4, 5.6.7.8",
        )
        .unwrap();
        assert_eq!(request.method, "GET");
        assert_eq!(request.path, "/");
        assert_eq!(request.query.as_deref(), Some("func=tet&base=2"));
        assert_eq!(request.header("host"), Some("tet.sh"));
        assert_eq!(request.header("x-forwarded-for"), Some("1.2.3.4, 5.6.7.8"));
        assert_eq!(request.header("missing"), None);

        let request = parse_head(b"HEAD http://tet.sh/api?x=1#top HTTP/1.0").unwrap();
        assert_eq!(
            (request.method.as_str(), request.path.as_str()),
            ("HEAD", "/api")
        );
        assert_eq!(request.query.as_deref(), Some("x=1"));
        let request = parse_head(b"GET https://tet.sh HTTP/1.1").unwrap();
        assert_eq!((request.path.as_str(), request.query), ("/", None));

        for (head, status) in [
            (&b"GET /"[..], 400),
            (b"GET / HTTP/1.1 extra", 400),
            (b"GET / HTTP/2", 505),
            (b"get / HTTP/1.1", 400),
            (b"GET * HTTP/1.1", 400),
            (b"GET /\xc3\xa9 HTTP/1.1", 400),
            (b"GET /a\nb HTTP/1.1", 400),
            (b"GET /?x=\x1b[2J HTTP/1.1", 400),
            (b"GET /\tx HTTP/1.1", 400),
            (b"GET / HTTP/1.1\r\nno colon here", 400),
            (b"", 400),
        ] {
            match parse_head(head) {
                Err(Unread::Status(found, _)) => assert_eq!(found, status, "{head:?}"),
                other => panic!("{head:?}: {other:?}"),
            }
        }
    }
}
