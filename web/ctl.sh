#!/bin/sh
# Starts, stops and shows the web page and API of tet on the machine where
# web/freebsd/install.sh put them. That script installs this one there as
# tet-webctl:
#
#   tet-webctl start | up      start the server and the proxy in front of it
#   tet-webctl stop | down     stop both; the calculations that run are ended
#   tet-webctl restart         stop and start: the cache starts empty
#   tet-webctl status          what runs, what is being calculated, the
#                              counters, the limits, the certificate
#   tet-webctl log [N]         the last N lines of the log about calculations:
#                              what was asked for and what was returned (20)
#   tet-webctl log -f          the whole log as it is written, until Ctrl-C
#   tet-webctl timeout [TIME]  the longest that one calculation may take:
#                              shows it, or sets it to TIME (seconds, or
#                              90m, 3h, 2d) in the settings file, which the
#                              server reads again without a restart
#   tet-webctl cache           every entry of the cache, one a line, as
#                              f=ate prec=20 base=2 height=4.0000000000000000000 value=65536
#                              from the one used longest ago to the one used
#                              last, then the calculations that run
#
# It works as root and calls sudo for that. On a machine without the
# service, as in a checkout of the repository, it runs the installed script
# of another machine over ssh:
#
#   web/ctl.sh status [user@host]      the machine of web/deploy.sh by default
#
# status ends with 0 when the server, the proxy and the site answer.

set -eu

usage() {
    echo "usage: $0 start|up|stop|down|restart|status|log [N|-f]|timeout [TIME]|cache [user@host]" >&2
    exit 2
}
[ $# -ge 1 ] || usage
case "$1" in
    start | up) action=start ;;
    stop | down) action=stop ;;
    restart | status | log | timeout | cache) action=$1 ;;
    *) usage ;;
esac
shift
# What the action takes: a number of lines or -f, or a time.
argument=
case "$action:${1:-}" in
    log:-f | log:[0-9]* | timeout:[0-9]*)
        argument=$1
        shift
        ;;
esac
case "$action:$argument" in
    log:-f | *:) ;;
    log:*[!0-9]*) usage ;;
    timeout:*[!0-9smhd]* | timeout:*[smhd]?*) usage ;;
esac
[ $# -le 1 ] || usage
host=${1:-}

if [ -n "$host" ] || [ ! -x /usr/local/etc/rc.d/tet_web ]; then
    # A terminal there, so that Ctrl-C ends what follows the log.
    terminal=
    if [ "$argument" = -f ] && [ -t 0 ]; then terminal=-t; fi
    exec ssh $terminal "${host:-lgryglicki@ovh}" "sudo /usr/local/bin/tet-webctl $action $argument"
fi
if [ "$(id -u)" != 0 ]; then
    command -v sudo > /dev/null || { echo "$0: run it as root" >&2; exit 1; }
    exec sudo "$0" "$action" $argument
fi

setting() {
    found=$(sysrc -qn "$1" 2> /dev/null || true)
    echo "${found:-$2}"
}
listen=$(setting tet_web_listen 127.0.0.1:8080)
user=$(setting tet_web_runas tetweb)
log=$(setting tet_web_log /var/log/tet-web.log)
config=$(setting tet_web_config /usr/local/etc/tet-web.conf)

runs() { service "$1" status > /dev/null 2>&1; }
answers() { [ "$(fetch -q -T 5 -o - "$1/healthz" 2> /dev/null)" = ok ]; }

# The address at which the proxy serves the site: the name of the first
# site in its configuration, or the address of this machine for a port.
site() {
    label=$(awk '/^[^#[:space:]].*\{$/ { print $1; exit }' /usr/local/etc/caddy/Caddyfile 2> /dev/null || true)
    case "$label" in
        "") ;;
        :*)
            interface=$(route -n get default 2> /dev/null | awk '/interface:/ { print $2 }')
            address=$(ifconfig "${interface:-lo0}" 2> /dev/null | awk '$1 == "inet" { print $2; exit }')
            [ "$label" = :80 ] && label=
            echo "http://${address:-127.0.0.1}$label"
            ;;
        *) echo "https://$label" ;;
    esac
}

# Until when the certificate is valid that the proxy serves for a name.
certificate() {
    end=$(openssl s_client -connect 127.0.0.1:443 -servername "$1" < /dev/null 2> /dev/null |
        openssl x509 -noout -enddate 2> /dev/null | sed 's/^notAfter=//')
    if [ -n "$end" ]; then
        echo "certificate of $1: valid until $end; the proxy renews it by itself"
    else
        echo "certificate of $1: the proxy serves none"
        return 1
    fi
}

start() {
    runs tet_web || service tet_web start
    for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        answers "http://$listen" && break
        sleep 0.5
    done
    runs caddy || service caddy start
    # The proxy answers once it has its certificate, which it may have to
    # obtain first.
    url=$(site)
    for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
        [ -n "$url" ] || break
        answers "$url" && break
        sleep 1
    done
}

stop() {
    # The server first. The proxy waits for the requests that it is passing
    # on before it stops, and a request may wait minutes for a calculation:
    # those end with the server.
    if runs tet_web; then service tet_web stop || true; fi
    if runs caddy; then service caddy stop || true; fi
    # The calculations end with the server. Whatever else runs as its user
    # was left by a server that did not end them, and is ended here.
    for pid in $(pgrep -u "$user" || true); do
        kill "$pid" 2> /dev/null || true
    done
}

# The states of the two services; fails when one of them does not run.
services() {
    both=0
    for name in tet_web caddy; do
        service "$name" status 2>&1 || both=1
    done
    return "$both"
}

status() {
    fine=0
    services || fine=1
    url=$(site)
    if [ -n "$url" ] && answers "$url"; then
        echo "$url/ answers"
    else
        echo "${url:-the site of the proxy} does not answer"
        fine=1
    fi
    case "$url" in
        https://*) certificate "${url#https://}" || fine=1 ;;
    esac
    fetch -q -T 5 -o - "http://$listen/status" 2> /dev/null || {
        echo "tet-web does not answer on $listen"
        fine=1
    }
    echo "processes of $user:"
    ps -ww -U "$user" -o pid,nice,etime,time,rss,command || true
    return "$fine"
}

# The lines of the log that say what was asked for and what was returned,
# and those about the server itself: its starts and its settings.
calculations() {
    if [ "$argument" = -f ]; then
        exec tail -F "$log"
    fi
    grep -E ' (MISS|HIT|WAIT) | - DONE | - SETTINGS | rejected: | tet-web listening on ' "$log" |
        tail -n "${argument:-20}"
}

# Shows the longest that a calculation may take, after setting it.
limit() {
    if [ -n "$argument" ]; then
        if [ "${argument%[smhd]}" -lt 1 ]; then
            echo "$0: the time is at least 1 second" >&2
            exit 2
        fi
        if grep -q '^[[:space:]]*TET_WEB_TIMEOUT[[:space:]]*=' "$config" 2> /dev/null; then
            sed -i '' -e "s/^[[:space:]]*TET_WEB_TIMEOUT[[:space:]]*=.*/TET_WEB_TIMEOUT=$argument/" "$config"
        else
            echo "TET_WEB_TIMEOUT=$argument" >> "$config"
        fi
        # The server looks at the file every two seconds.
        sleep 3
    fi
    grep '^[[:space:]]*TET_WEB_TIMEOUT[[:space:]]*=' "$config" 2> /dev/null ||
        echo "$config does not set TET_WEB_TIMEOUT: 3 hours"
    fetch -q -T 5 -o - "http://$listen/status" 2> /dev/null | grep '^limits: ' || {
        echo "tet-web does not answer on $listen"
        return 1
    }
}

# The entries of the cache. The server lists them for a request from this
# machine only, and for none that the proxy passes on.
entries() {
    answers "http://$listen" || {
        echo "tet-web does not answer on $listen" >&2
        return 1
    }
    # grep passes every line on and tells whether there was one.
    fetch -q -T 60 -o - "http://$listen/cache" 2> /dev/null | grep '' ||
        echo "the cache is empty" >&2
}

case "$action" in
    start)
        start
        status
        ;;
    stop)
        stop
        # Stopped is what was asked for.
        services || true
        ;;
    restart)
        stop
        start
        status
        ;;
    status) status ;;
    log) calculations ;;
    timeout) limit ;;
    cache) entries ;;
esac
