# The web page and HTTP API of `tet`

`tet-web` is a small server that puts a page and an HTTP API in front of the
`tet` binary. A request names a function, a precision and two numbers; the
server runs `tet` with them and returns what it printed. The numerical crate
is not linked and not changed: every computation is one `tet` process.

A public instance runs at <https://hyp4.duckdns.org/>, over IPv4 and IPv6.
It speaks HTTPS only: `http://hyp4.duckdns.org` and the bare address of the
machine answer with nothing but a redirect to `https://hyp4.duckdns.org`.

```
web/
  server/     the server: Rust, standard library only (crate tet-web)
  static/     the page: index.html (a template), app.js, style.css; all of
              them are built into the binary
  freebsd/    rc.d service, settings file, Caddy configurations, install.sh
  deploy.sh   copy the tree to a FreeBSD machine, build and install there
  ctl.sh      start, stop, status, log and time limit of the installed
              service (installed as tet-webctl)
```

The parts of this file: [Examples](#examples), [The page](#the-page),
[The API](#the-api), [Calculations that take longer](#calculations-that-take-longer),
[The cache](#the-cache), [Settings](#settings), [The log](#the-log),
[Running it](#running-it), [Installing on FreeBSD](#installing-on-freebsd),
[Administration](#administration).

## Examples

The reply is JSON, or the two lines of `tet` with `format=text`:

```console
$ curl 'https://hyp4.duckdns.org/?prec=20&func=ate&base=2&value=65536'
{"func":"ate","prec":20,"base":"2","value":"65536","result":"height","re":"4.0000000000000000000","im":"0","seconds":1.843,"cached":false,"command":"tet --ate 20 2 65536"}
$ curl 'https://hyp4.duckdns.org/?func=tet&base=2&height=0.5&format=text'
1.4587818160364217007
0
$ curl 'https://hyp4.duckdns.org/?prec=30&func=ssrt&value=16&height=3&format=text'
2.00000000000000000000000000000
0
$ curl 'https://hyp4.duckdns.org/?func=slog&base=e&value=10&format=text'
1.8425417879229475398
0
$ curl 'https://hyp4.duckdns.org/?func=sroot&value=10+i&height=2&format=text'
2.5090560476756793287
5.1911352949733962782e-2
```

Complex numbers, the four real parts of two numbers, expressions with
constants, functions and tetration inside:

```console
$ curl 'https://hyp4.duckdns.org/?func=tet&base=1.5+0.5i&height=1+0.7i&format=text'
1.1766384980936338172
6.0310294707303037947e-1
$ curl 'https://hyp4.duckdns.org/?prec=25&func=tet&base_re=1.5&base_im=0.5&height_re=1&height_im=0.7&format=text'
1.176638498093633817179947
6.031029470730303794686848e-1
$ curl 'https://hyp4.duckdns.org/?prec=30&func=tet&base=sqrt(2)&height=pi/e&format=text'
1.45708085024716451232466512408
0
$ curl 'https://hyp4.duckdns.org/?prec=30&func=tet&base=2^^0.5&height=ate(2,65536)/3&format=text'
1.56421525081821576076299857261
0
$ curl -G 'https://hyp4.duckdns.org/' --data-urlencode 'func=tet' --data-urlencode 'base=e^(1/e) + 0.1' --data-urlencode 'height=3/2' --data-urlencode 'format=text'
1.7593446795494862551
0
```

Any number of digits can be asked for; the server has no largest precision
of its own:

```console
$ curl 'https://hyp4.duckdns.org?prec=200&func=tet&base=1.444&height=9'
{"func":"tet","prec":200,"base":"1.444","height":"9","result":"value","re":"2.2791412188531754024460487965958041708527538273069445476337342180440070509188713438062245659382130472041611891989358361727705638706900687129222349315267379433861405147533581204785062424936783900850502","im":"0","seconds":0.004,"cached":false,"command":"tet 200 1.444 9"}
$ curl -s 'https://hyp4.duckdns.org/?prec=100000&func=tet&base=1.444&height=9&format=text' | wc -c
  100004
```

What `tet` refuses, and what is not a request for a computation:

```console
$ curl 'https://hyp4.duckdns.org/?func=tet&base=2&height=x'
{"func":"tet","prec":20,"base":"2","height":"x","error":"invalid number \"x\": unknown name x (an input has no variables; --help lists the constants)","status":422,"seconds":0.004,"cached":false,"command":"tet 20 2 x"}
$ curl 'https://hyp4.duckdns.org/?func=tet&base=2'
{"error":"func=tet takes base and height, or base_re, base_im, height_re and height_im","status":400}
$ curl 'https://hyp4.duckdns.org/?func=tet&base=2&height=3&prec=0'
{"error":"prec is the number of digits, a whole number from 1 on, not \"0\"","status":400}
```

A calculation that takes longer than the two minutes a request waits; the
last reply has 300 digits:

```console
$ curl 'https://hyp4.duckdns.org/?prec=300&func=tet&base=2&height=0.51&format=text'
running: the calculation is taking longer than 2 minutes and continues on this server: ask the same again later for the result; it may take up to 3 hours
$ curl 'https://hyp4.duckdns.org/?prec=300&func=tet&base=2&height=0.51&format=text'
running: this calculation is already running, for 2 min 25 s now: allow it up to 3 hours to complete and ask the same again for the result
$ curl 'https://hyp4.duckdns.org/?prec=300&func=tet&base=2&height=0.51&format=text'
1.46850203312048047550161815289101885815248225760477496374185862244737525454441844516356899578403946925...
0
```

Plain HTTP only shows the way:

```console
$ curl -s -o /dev/null -w '%{http_code} %{redirect_url}\n' 'http://hyp4.duckdns.org/?func=tet&base=2&height=3'
308 https://hyp4.duckdns.org/?func=tet&base=2&height=3
$ curl -sL 'http://hyp4.duckdns.org/?func=tet&base=2&height=3&format=text'
16.000000000000000000
0
```

## The page

`https://hyp4.duckdns.org/` is a form: the precision (20 digits unless
changed), the function (`tet`, `ate` or `ssrt`), two complex numbers or four
real ones, and the button `Calculate`. The result is shown below it with the
`tet` command that computed it and the address of the same request to the
API. Below the form are links to this documentation, to the thread about
the page on the Tetration Forum
([Tetration online calculator](https://tetrationforum.org/showthread.php?tid=1830))
and to the two viewers that plot with `tet`,
[tet-line](https://github.com/lukaszgryglicki/tet-line) and
[rust-opengl-complex-drawer](https://github.com/lukaszgryglicki/rust-opengl-complex-drawer).

The page needs no JavaScript. Its form is sent to the server as it is
(`GET /?format=html&prec=20&func=tet&numbers=2&base=2&height=0.5`), and the
server answers with the same page with the result in it. It works in text
browsers:

```console
$ links -dump 'https://hyp4.duckdns.org/?format=html&prec=20&func=ate&numbers=2&base=2&value=65536'
[...]
   Calculate

Result: the height

   real part
           4.0000000000000000000

   imaginary part
           0

   computed in 1.843 s, returned from the cache
   command: tet --ate 20 2 65536
   API: /?prec=20&func=ate&base=2&value=65536
[...]
```

Without JavaScript:

* The fields of the two numbers follow the choices of the function and of
  the kind of arguments when the form is sent. The button `Show the fields
  of this choice` does only that; `Calculate` after another choice shows the
  new fields first, with what was typed kept where the names stay.
* A calculation that is not ready after two minutes is shown as `Still
  calculating`. The page then asks again every five seconds by itself
  (`<meta http-equiv="refresh">`). `lynx` and `links` show a link for that
  at the top instead (`links -html-auto-refresh 1` follows it by itself),
  and the page has the link `Ask again now`.
* In the address of such a page a space is a `+` and a plus sign is `%2B`,
  as every browser writes a form.

With JavaScript the page is the same, and `app.js` saves the reloads: the
fields change at once, the result is taken out of the page that the server
answers with and put into the page that is shown, and a calculation that
takes longer is asked for again every five seconds. The address bar holds
the address of the form, so that a reload or a bookmark shows the same
result.

## The API

`GET /?func=...` (or `GET /api?func=...`) with these parameters:

| Parameter | Meaning |
|---|---|
| `func` | `tet`: the value `b^^h`; `ate` or `slog`: the height `h` with `b^^h = z`; `ssrt` or `sroot`: the base `b` with `b^^h = z` |
| `prec` | decimal digits of the result, a whole number from 1 on; 20 when it is left out |
| `base`, `height` | the two numbers of `tet`, each complex |
| `base`, `value` | the two numbers of `ate` / `slog` |
| `value`, `height` | the two numbers of `ssrt` / `sroot` |
| `base_re`, `base_im`, ... | the four-number form: each of the two numbers as a real and an imaginary part, each of them real |
| `format` | `json` (the default), `text`, or `html`: the page with the result, as its form asks for it |

Each number is a number or a constant expression of `tet` (README § 3.2):
`2`, `1.5+0.5i`, `pi/e`, `e^(1/e)`, `2^^0.5`, `ate(2, 16)`. In the query a
`+` is a plus sign, so that `base=2+i` can be typed by hand; a space is
`%20`, and `%2B` is a plus sign as well. (With `format=html` the query is
read as a browser writes a form: there a `+` is a space.)

`re` and `im` are the two lines of `tet`, as strings so that no digit is
lost; `result` says what they are (`value`, `height` or `base`); `seconds`
is the time `tet` took; `command` is the command line to run by hand. An
error has `error` and `status` instead of the result; in the text format it
is one line `error: ...`.

| Status | Meaning |
|---|---|
| 200 | the result |
| 202 | the calculation is not ready and continues: ask the same again later |
| 400 | the request is not a computation: a missing or unknown parameter, a precision that is not a whole number from 1 on |
| 422 | `tet` refused the input or found no answer; `error` is its message |
| 503 | all slots are busy calculating: try again later |
| 504 | no result within the time limit of a calculation |
| 500 | anything else, for instance a `tet` that the system stopped at its memory limit, or a result of more than 16 MiB |

A page (`format=html`) always has the status 200 and says the status of the
API in its text, as in `... (HTTP 422)`.

Replies of the API carry `X-Cache: HIT` (the outcome was kept), `MISS` (this
request started the calculation) or `WAIT` (an earlier one had),
`Cache-Control: no-store` and `Access-Control-Allow-Origin: *`. The other
paths: `/help` is the usage text of `tet`, `/stats` the counters and limits
of the server as JSON, `/status` the same as text, `/healthz` answers `ok`.
`/cache` lists what the cache holds, for the machine of the server alone
([The cache](#the-cache)).

## Calculations that take longer

A calculation runs in the server, not in the request that started it: it
ends with its result, with a refusal of `tet` or at the time limit
(`TET_WEB_TIMEOUT`, three hours), whether or not anybody still waits. The
requests for it wait until it is `TET_WEB_WAIT` old (two minutes), counted
from the first request. If it is not ready then, and for every request that
comes later, the reply has the status 202:

```
{"func":"tet","prec":300,"base":"2","height":"0.51","running":true,"message":"this calculation is already running, for 2 minutes now: allow it up to 3 hours to complete and ask the same again for the result","status":202,"running_seconds":120.923,"limit_seconds":10800,"cached":false,"command":"tet 300 2 0.51"}
```

In the text format it is the one line `running: ...`. The same request
again never starts the calculation a second time: it finds it running, or
finds its outcome in the cache.

At most `TET_WEB_JOBS` calculations run at a time: as many as the machine
has processors, and four at least. There is no queue. While all slots are
taken, a new calculation is turned away at once, and that is not kept: the
same request a little later is calculated.

```console
$ curl -si 'http://127.0.0.1:8080/?prec=130&func=tet&base=2&height=0.75&format=text' | grep -i -e '^HTTP' -e '^x-cache' -e '^error'
HTTP/1.1 503 Service Unavailable
X-Cache: MISS
error: all slots are busy calculating; try again later
$ curl 'http://127.0.0.1:8080/?prec=130&func=tet&base=2&height=0.75'
{"func":"tet","prec":130,"base":"2","height":"0.75","error":"all slots are busy calculating; try again later","status":503,"cached":false,"command":"tet 130 2 0.75"}
```

A calculation that the time limit stopped is kept as such, so that the same
request is answered at once instead of running for hours again; it is
calculated again once the limit has been raised. On Linux and FreeBSD the
calculations end with the server: stopping it ends them.

## The cache

Finished computations are kept in memory and nowhere else: a restart of the
server begins with an empty cache. The same request again is answered at
once (`"cached":true`, `X-Cache: HIT`). Kept are results, the refusals of
`tet`, which are the same whenever they are repeated, and the calculations
that the time limit stopped; a busy server or a failure is not kept. `ate`
and `slog`, and `ssrt` and `sroot`, share their entries, and so do the API
and the page; arguments are compared as they were written, so `2` and `2.0`
are two entries. The cache holds at most `TET_WEB_CACHE_MB` (1024 MiB
unless set): that much memory is not set aside, it is used as the results
come, and beyond it the entries used longest ago are dropped.

`GET /cache` lists the entries, one a line: the function, the precision and
the three numbers of the function in the order base, height, value, of
which two are as they were asked for and the third is the result. A
complex result of two complex numbers is one number (`1.17+6.03e-1i`); of
four real numbers the result has its two parts as well (`value_re`,
`value_im`). A word that a shell would not take as it is stands in quotes.
The kept entries come first, from the one used longest ago to the one used
last, then the calculations that run:

```console
$ curl -s http://127.0.0.1:8080/cache
f=ate prec=20 base=2 height=4.0000000000000000000 value=65536
f=tet prec=20 base=1.5+0.5i height=1+0.7i value=1.1766384980936338172+6.0310294707303037947e-1i
f=tet prec=20 base_re=1.5 base_im=0.5 height_re=1 height_im=0.7 value_re=1.1766384980936338172 value_im=6.0310294707303037947e-1
f=ssrt prec=20 base=2.0000000000000000000 height=3 value=16
f=tet prec=20 base=x height=0.5 error='invalid number "x": unknown name x (an input has no variables; --help lists the constants)'
f=tet prec=15 base=0.2+0.4i height=2.5-i value=2.01498533063010-9.63361309384064e-1i
f=tet prec=300 base=2 height=0.51 running=147s
```

That is what everybody asked for, so the server shows it only to a request
that comes from its own machine and that no proxy passed on: a request
with a header by which a proxy names its client (`X-Forwarded-For` and its
kin, `Forwarded`, `Via`, `X-Real-IP`) gets the 404 of an address where
nothing is. Caddy sets such headers on everything that it passes on; a
proxy that does not must not pass `/cache` on.

## Settings

`tet-web` takes no arguments. Five settings are environment variables, read
when it starts:

| Variable | Default | Meaning |
|---|---|---|
| `TET_WEB_LISTEN` | `127.0.0.1:8080` | address to listen on |
| `TET_WEB_TET` | `tet` | the `tet` program: a path, or a name on `PATH` |
| `TET_WEB_CACHE_MB` | `1024` | the most memory for kept results, in MiB |
| `TET_WEB_REVISION` | none | text that the page and `/stats` show as the version |
| `TET_WEB_CONFIG` | none | the settings file |

The other six can change while the server runs. They stand in the settings
file, one `NAME=value` a line (a `#` begins a comment), or in the
environment; the file is above the environment. The server reads the file
when it starts and looks at it every two seconds: what a changed file sets
is in use at once, without a restart, and the log says so. A file that
cannot be read as settings changes nothing, and the log says why.

| Setting | Default | Meaning |
|---|---|---|
| `TET_WEB_TIMEOUT` | `3h` | the longest that one calculation may take; it is stopped then. A calculation that runs when this changes gets the new limit |
| `TET_WEB_WAIT` | `120` | how long a request waits for its result |
| `TET_WEB_JOBS` | processors, at least 4 | calculations at a time |
| `TET_WEB_THREADS` | one | threads of one `tet` (its `TET_MT`) |
| `TET_WEB_MEMORY_MB` | unlimited | address space of one `tet` (Linux, FreeBSD) |
| `TET_WEB_NICE` | that of the server | niceness of each `tet`, up to 20 |

A time is a number of seconds, or of minutes, hours or days: `90`, `90m`,
`3h`, `2d`. [freebsd/tet-web.conf](freebsd/tet-web.conf) is the file that a
first installation puts on the machine: it sets six hours, five minutes,
two threads and 1024 MiB for each `tet`, and niceness 10. What the server
reads is the file on its machine, not the one in a checkout
([The other limits](#the-other-limits)).

The server speaks plain HTTP/1.1, one request per connection, and is meant
to listen on a local address behind a reverse proxy that faces the network
and speaks HTTPS.

## The log

The server writes one line for every request to its standard output: the
time (UTC), the client, the request, the status and the time the reply
took. A request for a calculation has more: whether the outcome was kept
(`HIT`), calculated for this request (`MISS`) or for an earlier one
(`WAIT`), the time `tet` took, the command, and the result after `=` or
the reason after `:`.

```
2026-11-14T14:53:49Z 4.1.4.150 GET /?func=ate&base=2&value=65536&format=text -> 200 1.721s MISS 1.721s tet --ate 20 2 65536 = 4.0000000000000000000 0
2026-11-14T14:53:51Z 4.1.4.150 GET /?format=html&prec=20&func=ssrt&numbers=2&value=16&height=3 -> 200 0.000s HIT 0.005s tet --ssrt 20 16 3 = 2.0000000000000000000 0
2026-11-14T14:54:26Z 2001:41d0:801:2000::1082 GET /?func=tet&base=2&height=3&format=text -> 200 0.004s MISS 0.004s tet 20 2 3 = 16.000000000000000000 0
2026-11-14T15:02:15Z 4.1.4.150 GET /?prec=300&func=tet&base=2&height=0.51&format=text -> 202 120.001s MISS tet 300 2 0.51: still running after 2 minutes
2026-11-14T15:04:45Z - DONE tet 300 2 0.51 -> 200 270.574s = 1.46850203312048047550161815289101885815248225760477496374185862244737525454441844516356899578403946...(301 characters) 0
2026-11-14T15:06:09Z 127.0.0.1 GET /?prec=130&func=tet&base=2&height=0.75&format=text -> 503 0.000s MISS tet 130 2 0.75: all slots are busy calculating; try again later
2026-11-14T15:08:15Z - SETTINGS a request waits 2 minutes, a calculation may take 6 hours, 4 at a time, threads of one tet: one, its memory: 512 MiB, its niceness: 10
```

A number of more than 100 characters is cut to its beginning and its
length. `DONE` is the end of a calculation that no request waited for.
`SETTINGS` is a change of the settings file. A request that only finds a
calculation still running, without having waited for it, has no line: a
page asks like that every five seconds.

## Running it

```console
$ cargo build --release --bin tet            # the repository root: tet itself
$ cd web/server
$ TET_WEB_TET=../../target/release/tet cargo run --release
2026-11-14T15:11:23Z tet-web listening on 127.0.0.1:8080: tet=../../target/release/tet revision=""; a request waits 2 minutes, a calculation may take 3 hours, 16 at a time, threads of one tet: one, its memory: unlimited, its niceness: that of the server
```

`http://127.0.0.1:8080/` is the page. The tests:

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
both there, runs it and checks the site from outside. Both can be repeated:
a later run installs the new binaries, restarts the server (its cache
starts empty, and the calculations that run are ended) and leaves what is
already in place as it is.

```console
$ SITE=hyp4.duckdns.org web/deploy.sh    # https://hyp4.duckdns.org only; the name points to the machine
$ web/deploy.sh                          # an update: the site that is installed stays
$ SITE=ip web/deploy.sh                  # plain HTTP on the address of the machine, without a name
```

What is installed:

* `/usr/local/bin/tet`, `/usr/local/bin/tet-web` and
  `/usr/local/bin/tet-webctl` (`web/ctl.sh`);
* the service `tet_web` (`/usr/local/etc/rc.d/tet_web`): the server as the
  user `tetweb` on `127.0.0.1:8080`, kept running by daemon(8), its log in
  `/var/log/tet-web.log` (rotated by newsyslog: ten files of 5 MB);
* `/usr/local/etc/tet-web.conf`, the settings file, only when there is
  none: what was set there stays;
* the packages `caddy` and `portacl-rc`: Caddy faces the network as the user
  `www` and passes the requests on;
* marked rules in `/etc/pf.conf` when pf is enabled: TCP port 80, and for a
  domain TCP and UDP port 443, for IPv4 and IPv6.

`/etc/rc.conf` and `/etc/pf.conf` are kept as they were in
`/etc/rc.conf.before-tet-web` and `/etc/pf.conf.before-tet-web`.

With a domain name Caddy serves `https://name` only
(`freebsd/Caddyfile.domain`): it obtains the certificate from Let's Encrypt
by itself and renews it by itself, plain HTTP is answered with a redirect
to `https://name` for the name and for the bare address, and the replies
carry `Strict-Transport-Security`. The name has to point to the machine
before that, because the certificate authority checks it there. The contact
address for the certificate authority is `ACME_EMAIL`, or the first line of
`web/acme-email.secret` (a file that is not in the repository); it is
optional, and without it the address that is installed stays.
With `SITE=ip` Caddy serves plain HTTP on every address
(`freebsd/Caddyfile.ip`).

## Administration

Everything below is done on the machine as root with `tet-webctl`, or from
a checkout of the repository on another machine with `web/ctl.sh`, which
runs the same over ssh (on `lgryglicki@ovh`, the machine of
`web/deploy.sh`, unless `user@host` is the last argument).

| Task | Command |
|---|---|
| start | `tet-webctl start` (or `up`) |
| stop | `tet-webctl stop` (or `down`) |
| restart | `tet-webctl restart` |
| what runs, counters, limits, certificate | `tet-webctl status` |
| the longest time of a calculation | `tet-webctl timeout` shows it, `tet-webctl timeout 6h` sets it |
| what was asked for and what was returned | `tet-webctl log`, `tet-webctl log 200`, `tet-webctl log -f` |
| what the cache holds | `tet-webctl cache` |
| install a new version | `web/deploy.sh` in the checkout |

### Start, stop, restart, status

`tet-webctl` acts on the server and on the proxy together.

```console
# tet-webctl start           # or up
# tet-webctl stop            # or down: stops both; the calculations that run are ended
# tet-webctl restart         # stop and start: the cache starts empty
# tet-webctl status
```

Both services start by themselves when the machine starts (`tet_web_enable`
and `caddy_enable` in `/etc/rc.conf`). The server is restarted by daemon(8)
if it ever ends by itself. A stop takes two seconds, a start three, a
restart four; `start` returns when the site answers:

```console
$ web/ctl.sh stop
Stopping tet_web.
Waiting for PIDS: 94587.
Stopping caddy... done
tet_web is not running.
caddy is not running.
$ web/ctl.sh start
Starting tet_web.
Starting caddy... done
Log: /var/log/caddy/caddy.log
tet_web is running as pid 95613.
caddy is running as pid 95706.
https://hyp4.duckdns.org/ answers
[...]
```

`status` ends with 0 when the server, the proxy and the site answer:

```console
$ web/ctl.sh status
tet_web is running as pid 91255.
caddy is running as pid 84124.
https://hyp4.duckdns.org/ answers
certificate of hyp4.duckdns.org: valid until Feb 12 13:55:00 2027 GMT; the proxy renews it by itself
tet-web 416b17b+: up 8 min 47 s, requests 195
calculations: 3 running, at most 4 at a time
cache: kept 13 (4502 bytes), hits 24, misses 18, waits 6, dropped 0
limits: a request waits 2 minutes, a calculation may take 3 hours
processes of tetweb:
  PID NI ELAPSED    TIME   RSS COMMAND
91256  0   08:47 0:00.33  5892 /usr/local/bin/tet-web
92337 10   02:09 1:57.19 10676 /usr/local/bin/tet -q 300 2 0.51
92339 10   00:50 0:41.52  8912 /usr/local/bin/tet -q 200 2 0.5
92340 10   00:26 0:19.60  5664 /usr/local/bin/tet -q --ssrt 20 10 0.5
```

The processes are the server and one `tet` for each calculation that runs,
with its time so far. One calculation can be ended alone with `kill` and
its PID; its request is told that `tet` was stopped, and nothing is kept.

The services alone, without `tet-webctl`:

```console
# service tet_web status | start | stop | restart
# service caddy status | start | stop | restart | reload | configtest
```

Caddy waits for the requests that it is passing on before it stops, and a
request may wait two minutes for a calculation: `service caddy stop` alone
can take that long. `tet-webctl stop` stops the server first, with which
those requests end.

### The longest time of a calculation

Six hours as installed (three without a settings file). `tet-webctl
timeout` shows it; with a time it sets it: a number of seconds, or of
minutes, hours or days (`90m`, `6h`, `2d`). Nothing is restarted and nothing
is lost: the server takes the new limit within a few seconds, also for the
calculations that run. Raise it to let one finish; lower it, and the ones
that are beyond it stop.

```console
$ web/ctl.sh timeout
TET_WEB_TIMEOUT=3h
limits: a request waits 2 minutes, a calculation may take 3 hours
$ web/ctl.sh timeout 6h
TET_WEB_TIMEOUT=6h
limits: a request waits 2 minutes, a calculation may take 6 hours
$ web/ctl.sh log 1
2026-11-14T15:08:15Z - SETTINGS a request waits 2 minutes, a calculation may take 6 hours, 4 at a time, threads of one tet: one, its memory: 512 MiB, its niceness: 10
```

The command writes the line `TET_WEB_TIMEOUT=6h` into the settings file
`/usr/local/etc/tet-web.conf`. The file can be edited by hand as well, and
it is where the other limits are changed in the same way, without a
restart: the time a request waits (`TET_WEB_WAIT`), the calculations at a
time (`TET_WEB_JOBS`), and the threads, the memory and the niceness of one
`tet` ([Settings](#settings)). After a change, `tet-webctl status` shows
the limits in use, and `tet-webctl log 1` the line `SETTINGS`, or
`SETTINGS not changed:` with the reason when the file has a mistake.

### The other limits

The server reads `/usr/local/etc/tet-web.conf` on its machine. The file
`web/freebsd/tet-web.conf` of the repository is only what a first
installation puts there: a change of it in a checkout changes nothing on
the machine, and `web/deploy.sh` leaves the file of the machine alone, so
that what was set there survives an update. To change a limit, edit the
file on the machine, or copy one there; nothing is restarted:

```console
# vi /usr/local/etc/tet-web.conf       # on the machine
$ scp web/freebsd/tet-web.conf lgryglicki@ovh:tet-web.conf.new          # or from a checkout
$ ssh lgryglicki@ovh 'sudo install -m 0644 -o root -g wheel tet-web.conf.new /usr/local/etc/tet-web.conf && rm tet-web.conf.new'
$ web/ctl.sh log 1
2026-11-14T16:24:30Z - SETTINGS a request waits 5 minutes, a calculation may take 6 hours, 4 at a time, threads of one tet: 2, its memory: 1024 MiB, its niceness: 10
```

### The cache

```console
# tet-webctl cache                     # every entry, one a line; then the calculations that run
# tet-webctl cache | wc -l             # how many
# tet-webctl cache | grep '^f=ate '    # the super-logarithms
# tet-webctl cache | cut -c 1-200      # long results cut short
```

```console
$ web/ctl.sh cache
f=ate prec=20 base=2 height=4.0000000000000000000 value=65536
f=ssrt prec=20 base=2.0000000000000000000 height=3 value=16
f=tet prec=300 base=2 height=0.51 running=147s
```

[The cache](#the-cache) explains the lines. An empty cache prints nothing
but `the cache is empty`. The cache is in the memory of the server only: a
restart empties it, and nothing else does. It may grow to 1024 MiB
(`tet_web_cache_mb` in `/etc/rc.conf`, read when the server starts);
`tet-webctl status` has its size in the line `cache: kept ...`.

### The log

```console
# tet-webctl log             # the last 20 lines about calculations and about the server itself
# tet-webctl log 200         # the last 200
# tet-webctl log -f          # everything, as it is written; Ctrl-C ends it
```

```console
$ web/ctl.sh log 3
2026-11-14T15:04:35Z 4.1.4.150 GET /?format=html&prec=20&func=ate&numbers=2&base=2&value=65536 -> 200 0.000s HIT 1.721s tet --ate 20 2 65536 = 4.0000000000000000000 0
2026-11-14T15:04:45Z - DONE tet 300 2 0.51 -> 200 270.574s = 1.46850203312048047550161815289101885815248225760477496374185862244737525454441844516356899578403946...(301 characters) 0
2026-11-14T15:04:59Z 4.1.4.150 GET /?prec=300&func=tet&base=2&height=0.51&format=text -> 200 0.000s HIT 270.574s tet 300 2 0.51 = 1.46850203312048047550161815289101885815248225760477496374185862244737525454441844516356899578403946...(301 characters) 0
```

[The log](#the-log) explains the lines. The file is `/var/log/tet-web.log`;
only root reads it, because it has the addresses of the clients. newsyslog
keeps ten older files of 5 MB each next to it, compressed
(`bzcat /var/log/tet-web.log.0.bz2`). Everything about one function, one
client or one status is a `grep` away:

```console
# grep ' tet --ate ' /var/log/tet-web.log        # every request for a super-logarithm
# grep ' -> 5' /var/log/tet-web.log              # what ended with 500, 503 or 504
# grep -c ' MISS ' /var/log/tet-web.log          # how many calculations were started
```

The log of the proxy, with what it does about the certificate, is
`/var/log/caddy/caddy.log`.

### Checks

```console
$ curl https://hyp4.duckdns.org/healthz
ok
$ curl https://hyp4.duckdns.org/status
tet-web 416b17b+: up 14 min 52 s, requests 246
calculations: 0 running, at most 4 at a time
cache: kept 26 (209120 bytes), hits 38, misses 28, waits 18, dropped 0
limits: a request waits 2 minutes, a calculation may take 3 hours
$ curl https://hyp4.duckdns.org/stats
{"revision":"416b17b+","uptime_seconds":892,"requests":247,"running":0,"cache":{"entries":26,"bytes":209120,"hits":38,"misses":28,"waits":18,"dropped":0},"limits":{"wait_seconds":120,"timeout_seconds":10800,"jobs":4}}
```

On the machine: who listens (the server on the local port 8080, the proxy
on 80 and 443 for IPv4 and IPv6), and what the packet filter lets in:

```console
$ sockstat -46 -l -P tcp -p 80,443,8080
USER   COMMAND      PID FD PROTO LOCAL ADDRESS         FOREIGN ADDRESS
tetweb tet-web    91256  3 tcp4  127.0.0.1:8080        *:*
www    caddy      84124  5 tcp46 *:443                 *:*
www    caddy      84124 10 tcp46 *:80                  *:*
# pfctl -sr | grep -E 'http|https'
pass in quick on vtnet0 proto tcp from any to (vtnet0) port = http flags S/SA keep state
pass in quick on vtnet0 proto tcp from any to (vtnet0) port = https flags S/SA keep state
pass in quick on vtnet0 proto udp from any to (vtnet0) port = https keep state
$ curl -s http://127.0.0.1:8080/status          # the server itself, without the proxy
$ curl -4 -s https://hyp4.duckdns.org/healthz   # by IPv4
$ curl -6 -s https://hyp4.duckdns.org/healthz   # by IPv6
```

### The certificate

There is nothing to do for it. Caddy obtained the certificate of
`hyp4.duckdns.org` from Let's Encrypt when the site was installed, and
renews it by itself about a month before it ends; a certificate is valid
for 90 days, and no cron job or other program is involved. All it needs is
what the site needs anyway: the name points to the machine, and the ports
80 and 443 are reachable.

`tet-webctl status` has the day until which the certificate is valid. More
about it:

```console
$ echo | openssl s_client -connect hyp4.duckdns.org:443 -servername hyp4.duckdns.org 2>/dev/null | openssl x509 -noout -issuer -subject -dates
issuer=C = US, O = Let's Encrypt, CN = YE1
subject=CN = hyp4.duckdns.org
notBefore=Nov 14 14:23:43 2026 GMT
notAfter=Feb 12 14:23:42 2027 GMT
# grep -E 'certificate obtained|renew|"level":"error"' /var/log/caddy/caddy.log | tail
```

The certificate and its key are in
`/var/db/caddy/data/caddy/certificates/acme-v02.api.letsencrypt.org-directory/hyp4.duckdns.org/`.
Should a new certificate ever be wanted before Caddy renews it: move that
directory away while the service is stopped. Caddy obtains a new one when
it starts. This was tried on 14 November 2026: the new certificate was
there seven seconds after the start. The one above is that one; the status
further up still shows the first one, of 13:55 the same day.

```console
# tet-webctl stop
# mv /var/db/caddy/data/caddy/certificates/acme-v02.api.letsencrypt.org-directory/hyp4.duckdns.org /var/tmp/
# tet-webctl start
# tet-webctl status          # "certificate of hyp4.duckdns.org: valid until" has the new day
# rm -r /var/tmp/hyp4.duckdns.org
```

If no certificate comes (`tet-webctl status`, `/var/log/caddy/caddy.log`),
stop again, move the directory back and start: the old certificate is in
use again. Let's Encrypt issues at most five certificates for one name in
a week, so this is nothing to repeat.

### The name

`hyp4.duckdns.org` is a name of DuckDNS with the two addresses of the
machine. They do not change by themselves. If the machine ever gets other
addresses, the name is set to them with the token of the DuckDNS account
(`web/token.secret` in the checkout, a file that is not in the repository;
`web/email.secret` next to it is the address of that account):

```console
$ curl "https://www.duckdns.org/update?domains=hyp4&token=$(cat web/token.secret)&ip=57.129.175.142&ipv6=2001:41d0:801:2000::1082&verbose=true"
OK
57.129.175.142
2001:41d0:801:2000::1082
NOCHANGE
$ host hyp4.duckdns.org
hyp4.duckdns.org has address 57.129.175.142
hyp4.duckdns.org has IPv6 address 2001:41d0:801:2000::1082
hyp4.duckdns.org mail is handled by 50 hyp4.duckdns.org.
```

For another name, `SITE=the.new.name web/deploy.sh` once it points to the
machine: Caddy obtains the certificate for it.

### Updating

`web/deploy.sh` in the checkout copies the tree, builds on the machine,
installs, restarts the server and checks the site: the page, the API, the
form, and that plain HTTP only redirects. Without `SITE` the site stays as
it is, and so do the settings in `/usr/local/etc/tet-web.conf`. The restart
ends the calculations that run and empties the cache.

### When something is wrong

| What is seen | What to do |
|---|---|
| the site does not answer | `tet-webctl status` says which of the two services is down; `tet-webctl start` |
| `all slots are busy calculating; try again later` | `tet-webctl status` lists the calculations that run and their times; wait, end one with `kill` and its PID, or raise `TET_WEB_JOBS` in the settings file |
| `no result within 3 hours, the limit of this server` | `tet-webctl timeout 12h`, then ask the same again: it is calculated again |
| `tet was stopped by the system (signal: ...): it needed more memory than this server allows, or it was ended by hand` | if nobody ended it: raise `TET_WEB_MEMORY_MB` in the settings file (1024 as installed) |
| a setting that was changed is not in use | `tet-webctl log 1`: `SETTINGS not changed:` and the reason |
| the browser warns about the certificate | `tet-webctl status`, then `/var/log/caddy/caddy.log`; the ports 80 and 443 have to be reachable from outside |
| a calculation should stop now | `kill` and its PID from `tet-webctl status` |

### Files

| File | What it is |
|---|---|
| `/usr/local/etc/tet-web.conf` | the settings that can change while the server runs |
| `/var/log/tet-web.log` | the log of the server |
| `/usr/local/etc/caddy/Caddyfile` | the site of the proxy: the name, HTTPS, the redirect |
| `/var/log/caddy/caddy.log` | the log of the proxy |
| `/var/db/caddy/` | the certificate and the account at the certificate authority |
| `/usr/local/etc/rc.d/tet_web` | the service; its start settings are `tet_web_*` in `/etc/rc.conf` |
| `/usr/local/bin/tet`, `tet-web`, `tet-webctl` | the programs |
| `/etc/pf.conf` | the packet filter; the lines marked `# tet-web` |

### Removing

`tet-webctl stop`, stop the service `portacl`, delete the three files in
`/usr/local/bin`, the rc script, `/usr/local/etc/tet-web.conf`,
`/usr/local/etc/newsyslog.conf.d/tet-web.conf` and the Caddyfile, remove the
`tet_web_*`, `caddy_*` and `portacl_*` lines from `/etc/rc.conf` and the
lines marked `# tet-web` from `/etc/pf.conf` (`pfctl -f /etc/pf.conf`), then
`pw userdel tetweb` and `pkg delete caddy portacl-rc`.
