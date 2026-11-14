#!/bin/sh
# Installs or updates the web page and HTTP API of tet on this FreeBSD
# machine. Run as root from a tree in which tet and tet-web are built:
#
#   sudo env SITE=ip sh web/freebsd/install.sh         plain HTTP on every address
#   sudo env SITE=tet.sh sh web/freebsd/install.sh     https://tet.sh only
#
# SITE         ip (default), or the domain name that points to this machine
# REVISION     text that the page and /stats show as the version
# ACME_EMAIL   contact address for the certificate authority (optional)
#
# What it changes, each step only when needed:
#   packages caddy and portacl-rc; the user tetweb;
#   /usr/local/bin/tet, /usr/local/bin/tet-web, /usr/local/etc/rc.d/tet_web,
#   /usr/local/etc/newsyslog.conf.d/tet-web.conf, /usr/local/etc/caddy/Caddyfile;
#   /etc/rc.conf: tet_web_*, caddy_*, portacl_*;
#   /etc/pf.conf: the lines marked "# tet-web" (ports 80, and 443 for a
#   domain).
# Both files are kept as they were in /etc/rc.conf.before-tet-web and
# /etc/pf.conf.before-tet-web.

set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
site=${SITE:-ip}
revision=${REVISION:-}
email=${ACME_EMAIL:-}

say() { printf '== %s\n' "$*"; }
fail() { printf 'install.sh: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = FreeBSD ] || fail "this script is for FreeBSD"
[ "$(id -u)" = 0 ] || fail "run it as root (sudo)"
tet="$root/target/release/tet"
web="$root/web/server/target/release/tet-web"
[ -x "$tet" ] || fail "$tet is missing: cargo build --release --locked --bin tet"
[ -x "$web" ] || fail "$web is missing: cargo build --release in web/server"
case "$site" in
    ip) ;;
    "" | .* | *. | *[!a-z0-9.-]*) fail "SITE is ip or a domain name in lower case, not '$site'" ;;
    *.*) ;;
    *) fail "SITE is ip or a domain name, not '$site'" ;;
esac
case "$revision$email" in *[!A-Za-z0-9.@_+-]*) fail "REVISION and ACME_EMAIL have no spaces or quotes" ;; esac

# The system files that are edited below, as they were before the first run.
[ -e /etc/rc.conf.before-tet-web ] || cp -p /etc/rc.conf /etc/rc.conf.before-tet-web

for package in caddy portacl-rc; do
    if ! pkg info -e "$package"; then
        say "installing the package $package"
        env ASSUME_ALWAYS_YES=yes pkg install -y "$package"
    fi
done

if ! pw usershow tetweb >/dev/null 2>&1; then
    say "creating the user tetweb"
    pw useradd tetweb -c "tet web page and API" -d /nonexistent -s /usr/sbin/nologin
fi

say "installing tet, tet-web and the service files"
install -m 0755 -o root -g wheel "$tet" /usr/local/bin/tet
install -m 0755 -o root -g wheel "$web" /usr/local/bin/tet-web
install -m 0555 -o root -g wheel "$here/tet_web" /usr/local/etc/rc.d/tet_web
install -d -m 0755 /usr/local/etc/newsyslog.conf.d
install -m 0644 -o root -g wheel "$here/newsyslog.conf" /usr/local/etc/newsyslog.conf.d/tet-web.conf
sysrc tet_web_enable=YES tet_web_revision="$revision" >/dev/null

say "starting tet-web"
service tet_web restart
ready=no
for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    if [ "$(fetch -q -o - http://127.0.0.1:8080/healthz 2>/dev/null)" = ok ]; then
        ready=yes
        break
    fi
    sleep 0.5
done
[ "$ready" = yes ] || fail "tet-web does not answer on 127.0.0.1:8080; see /var/log/tet-web.log"

# Caddy runs as www; mac_portacl lets that user bind the ports of the web.
say "letting the user www bind the ports 80 and 443"
sysrc portacl_users+=www >/dev/null
sysrc portacl_user_www_tcp="http https" portacl_user_www_udp="https" portacl_enable=YES >/dev/null
service portacl start

caddyfile=/usr/local/etc/caddy/Caddyfile
wanted=$(mktemp)
trap 'rm -f "$wanted" "${rules:-}"' EXIT
if [ "$site" = ip ]; then
    cat "$here/Caddyfile.ip" > "$wanted"
else
    {
        [ -z "$email" ] || printf '{\n\temail %s\n}\n\n' "$email"
        sed -e "s/@SITE@/$site/g" "$here/Caddyfile.domain"
    } > "$wanted"
fi
if [ -e "$caddyfile" ] && ! cmp -s "$caddyfile" "$caddyfile.sample" &&
    ! grep -q "tet-web behind Caddy" "$caddyfile"; then
    fail "$caddyfile is somebody else's configuration; add the site of $here by hand"
fi
if ! cmp -s "$wanted" "$caddyfile"; then
    say "writing $caddyfile for the site $site"
    install -m 0644 -o root -g wheel "$wanted" "$caddyfile"
fi
sysrc caddy_enable=YES caddy_user=www caddy_group=www >/dev/null
if service caddy status >/dev/null 2>&1; then
    say "reloading caddy"
    service caddy reload
else
    say "starting caddy"
    service caddy start
fi

# The packet filter: one rule for the web, marked, after the rules that are there.
interface=$(route -n get default 2>/dev/null | awk '/interface:/ { print $2 }')
if [ -r /etc/pf.conf ] && [ -n "$interface" ] && pfctl -s info 2>/dev/null | grep -q "Status: Enabled"; then
    rules=$(mktemp)
    grep -v "# tet-web" /etc/pf.conf > "$rules" || true
    if [ "$site" = ip ]; then
        echo "pass in quick on $interface proto tcp from any to ($interface) port 80 keep state # tet-web" >> "$rules"
    else
        echo "pass in quick on $interface proto tcp from any to ($interface) port { 80 443 } keep state # tet-web" >> "$rules"
        echo "pass in quick on $interface proto udp from any to ($interface) port 443 keep state # tet-web" >> "$rules"
    fi
    if ! cmp -s "$rules" /etc/pf.conf; then
        say "opening the web ports on $interface in /etc/pf.conf"
        pfctl -nf "$rules" || fail "the new packet filter rules do not parse; /etc/pf.conf is unchanged"
        [ -e /etc/pf.conf.before-tet-web ] || install -m 0600 -o root -g wheel /etc/pf.conf /etc/pf.conf.before-tet-web
        install -m 0600 -o root -g wheel "$rules" /etc/pf.conf
        pfctl -f /etc/pf.conf
    fi
else
    say "no enabled pf with /etc/pf.conf here: the packet filter is left alone"
fi

address=$(ifconfig "${interface:-lo0}" 2>/dev/null | awk '$1 == "inet" { print $2; exit }')
if [ "$site" = ip ]; then
    url="http://${address:-127.0.0.1}/"
else
    url="https://$site/"
fi
[ "$(fetch -q -o - http://127.0.0.1/healthz 2>/dev/null)" = ok ] || [ "$site" != ip ] ||
    fail "caddy does not pass http://127.0.0.1/healthz on to tet-web; see /var/log/caddy/caddy.log"
say "done: $url"
