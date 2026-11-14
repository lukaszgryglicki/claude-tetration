# The web page and HTTP API of `tet`

`tet-web` is a small server that puts a page and an HTTP API in front of the
`tet` binary. A request names a function, a precision and two numbers; the
server runs `tet` with them and returns what it printed. The numerical crate
is not linked and not changed: every computation is one `tet` process.

```
web/
  server/     the server: Rust, standard library only (crate tet-web)
  static/     the page: index.html, app.js, style.css (built into the binary)
  freebsd/    rc.d service, Caddy configurations, install.sh
  deploy.sh   copy the tree to a FreeBSD machine, build and install there
```

## Running it

```console
$ cargo build --release --bin tet            # the repository root: tet itself
$ cd web/server
$ TET_WEB_TET=../../target/release/tet cargo run --release
2026-11-14T08:15:00Z tet-web listening on 127.0.0.1:8080: tet=../../target/release/tet ...
```

`http://127.0.0.1:8080/` is the page: the precision (20 digits unless
changed), the function (`tet`, `ate` or `ssrt`), two complex numbers or four
real ones, and a button. The result is shown with the `tet` command that
computed it and the address of the same request to the API.

## The API

`GET /?func=...` (or `GET /api?func=...`) with these parameters:

| Parameter | Meaning |
|---|---|
| `func` | `tet`: the value `b^^h`; `ate` or `slog`: the height `h` with `b^^h = z`; `ssrt` or `sroot`: the base `b` with `b^^h = z` |
| `prec` | decimal digits of the result; 20 when it is left out |
| `base`, `height` | the two numbers of `tet`, each complex |
| `base`, `value` | the two numbers of `ate` / `slog` |
| `value`, `height` | the two numbers of `ssrt` / `sroot` |
| `base_re`, `base_im`, ... | the four-number form: each of the two numbers as a real and an imaginary part, each of them real |
| `format` | `json` (the default) or `text` |

Each number is a number or a constant expression of `tet` (README § 3.2):
`2`, `1.5+0.5i`, `pi/e`, `e^(1/e)`, `2^^0.5`, `ate(2, 16)`. In the query a
`+` is a plus sign, so that `base=2+i` can be typed by hand; a space is
`%20`, and `%2B` is a plus sign as well.

```console
$ curl 'http://127.0.0.1:8080/?prec=20&func=ate&base=2&value=65536'
{"func":"ate","prec":20,"base":"2","value":"65536","result":"height","re":"4.0000000000000000000","im":"0","seconds":0.842,"cached":false,"command":"tet --ate 20 2 65536"}
$ curl 'http://127.0.0.1:8080/?func=tet&base=1.5+0.5i&height=1+0.7i&format=text'
1.1766384980936338172
6.0310294707303037947e-1
$ curl 'http://127.0.0.1:8080/?func=ssrt&value_re=16&value_im=0&height_re=3&height_im=0&format=text'
2.0000000000000000000
0
```

`re` and `im` are the two lines of `tet`, as strings so that no digit is
lost; `result` says what they are (`value`, `height` or `base`); `seconds`
is the time `tet` took; `command` is the command line to run by hand. An
error has `error` and `status` instead of the result; in the text format it
is one line `error: ...`.

| Status | Meaning |
|---|---|
| 200 | the result |
| 400 | the request is not a computation: a missing or unknown parameter, a precision out of range |
| 422 | `tet` refused the input or found no answer; `error` is its message |
| 503 | all slots are busy and the queue is full, or no slot came free in time |
| 504 | no result within the time limit of the server |
| 500 | anything else, for instance a `tet` that the system stopped at its memory limit |

Replies of the API carry `X-Cache: HIT`, `MISS` or `WAIT` (see below),
`Cache-Control: no-store` and `Access-Control-Allow-Origin: *`. The other
paths: `/help` is the usage text of `tet`, `/stats` the counters and limits
of the server as JSON, `/healthz` answers `ok`.

## The cache

Finished computations are kept in memory and nowhere else: a restart of the
server begins with an empty cache. The same request again is answered at
once (`"cached":true`, `X-Cache: HIT`); a request whose computation is
running waits for it instead of starting a second one (`X-Cache: WAIT`).
Kept are results and the refusals of `tet`, which are the same whenever
they are repeated; a time limit or a busy server is not kept. `ate` and
`slog`, and `ssrt` and `sroot`, share their entries; arguments are compared
as they were written, so `2` and `2.0` are two entries. The cache holds at
most `TET_WEB_CACHE_MB`; beyond that the entries used longest ago are
dropped.

## Settings

Environment variables of `tet-web`; it takes no arguments.

| Variable | Default | Meaning |
|---|---|---|
| `TET_WEB_LISTEN` | `127.0.0.1:8080` | address to listen on |
| `TET_WEB_TET` | `tet` | the `tet` program: a path, or a name on `PATH` |
| `TET_WEB_MAX_PREC` | `1000` | largest precision in digits |
| `TET_WEB_TIMEOUT` | `60` | seconds one computation may take; it is stopped then |
| `TET_WEB_JOBS` | `2` | computations at a time |
| `TET_WEB_QUEUE` | `16` | requests that may wait for a slot |
| `TET_WEB_THREADS` | serial | threads of one `tet` (its `TET_MT`) |
| `TET_WEB_MEMORY_MB` | unlimited | address space of one `tet` (Linux, FreeBSD) |
| `TET_WEB_NICE` | that of the server | niceness of each `tet`, up to 20 |
| `TET_WEB_CACHE_MB` | `64` | memory for kept results |
| `TET_WEB_REVISION` | none | text that the page and `/stats` show as the version |

The server speaks plain HTTP/1.1, one request per connection, and is meant
to listen on a local address behind a reverse proxy that faces the network.
It writes one line per request to its standard output.

## Tests

```console
$ cd web/server
$ cargo test          # the units, and the server as a process with a script in the place of tet
$ cargo clippy --all-targets -- -D warnings
$ cargo fmt --check
```

## Installing on FreeBSD

`web/freebsd/install.sh` installs or updates everything on the machine it
runs on, as root, from a tree in which `tet` and `tet-web` are built;
`web/deploy.sh [user@host]` copies the working tree to a machine, builds
both there and runs it. Both can be repeated: a later run installs the new
binaries, restarts the server (its cache starts empty) and leaves what is
already in place as it is.

```console
$ web/deploy.sh                  # plain HTTP on the address of the machine
$ SITE=tet.sh web/deploy.sh      # https://tet.sh only, once the name points to the machine
```

What is installed:

* `/usr/local/bin/tet` and `/usr/local/bin/tet-web`;
* the service `tet_web` (`/usr/local/etc/rc.d/tet_web`): the server as the
  user `tetweb` on `127.0.0.1:8080`, kept running by daemon(8), its log in
  `/var/log/tet-web.log` (rotated by newsyslog);
* the packages `caddy` and `portacl-rc`: Caddy faces the network as the user
  `www` and passes the requests on;
* one marked rule in `/etc/pf.conf` when pf is enabled: TCP port 80, and 443
  for a domain.

`/etc/rc.conf` and `/etc/pf.conf` are kept as they were in
`/etc/rc.conf.before-tet-web` and `/etc/pf.conf.before-tet-web`.

With `SITE=ip` Caddy serves plain HTTP on every address
(`freebsd/Caddyfile.ip`). With a domain name it serves `https://name` only
(`freebsd/Caddyfile.domain`): Caddy obtains and renews the certificate by
itself, plain HTTP is answered with a redirect to `https://name`, and the
replies carry `Strict-Transport-Security`. The name has to point to the
machine before that, because the certificate authority checks it there.

The limits of the service are rc.conf variables, listed at the top of the rc
script; its defaults are 120 seconds, 2 computations at a time, 1024 MiB and
niceness 10 for each `tet`:

```console
# sysrc tet_web_timeout=300 && service tet_web restart
# service tet_web status
# tail -f /var/log/tet-web.log
```

To remove it: stop the services `tet_web`, `caddy` and `portacl`, delete
the two binaries, the rc script,
`/usr/local/etc/newsyslog.conf.d/tet-web.conf` and the Caddyfile, remove the
`tet_web_*`, `caddy_*` and `portacl_*` lines from `/etc/rc.conf` and the
lines marked `# tet-web` from `/etc/pf.conf` (`pfctl -f /etc/pf.conf`), then
`pw userdel tetweb` and `pkg delete caddy portacl-rc`.
