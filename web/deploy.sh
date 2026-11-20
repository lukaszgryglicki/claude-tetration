#!/bin/sh
# Builds tet and tet-web on a FreeBSD machine and installs the web page and
# API there (web/freebsd/install.sh). Run it from a checkout of the
# repository; the working tree is copied as it is, committed or not.
#
#   SITE=hyp4.duckdns.org web/deploy.sh   to lgryglicki@ovh, at https://hyp4.duckdns.org only
#   web/deploy.sh                         an update: the site that is installed stays
#   SITE=ip web/deploy.sh                 plain HTTP on the address of the machine
#   web/deploy.sh user@host               another machine
#
# SITE                 the domain name that points to the machine, or ip;
#                      without it the installed site stays (ip on a new machine)
# ACME_EMAIL           contact address for the certificate authority; without
#                      it the first line of web/acme-email.secret, if that
#                      file is there (it is not in the repository, and is
#                      not copied); without both the installed one stays
# TET_WEB_REMOTE_DIR   where the tree is copied to, in the home there (tet-web/src)
# SYMBOLIC_MATH        a static symbolic-math binary for that machine, the
#                      second program of the page; without it the one that
#                      `make static` built in ../symbolic-math, if it is
#                      there; without both the installed one stays
#
# It needs ssh access to a user who may run sudo without a password there,
# and cargo, rsync and a C toolchain on that machine.

set -eu

cd "$(dirname "$0")/.."
host=${1:-lgryglicki@ovh}
site=${SITE:-}
remote=${TET_WEB_REMOTE_DIR:-tet-web/src}
email=${ACME_EMAIL:-}
if [ -z "$email" ] && [ -r web/acme-email.secret ]; then
    email=$(head -n 1 web/acme-email.secret | tr -d '[:space:]')
fi

# The commit, with a + when the tree has changes that are not committed.
revision=$(git rev-parse --short HEAD)
if [ -n "$(git status --porcelain)" ]; then
    revision="$revision+"
fi

echo "== copying the tree ($revision) to $host:$remote"
ssh "$host" "mkdir -p '$remote'"
{
    git ls-files -z
    git ls-files -z --others --exclude-standard
} | rsync -a --from0 --files-from=- ./ "$host:$remote/"

# The second program of the page is built here, not there: its static binary
# is copied next to the tree, where install.sh finds it.
sm=${SYMBOLIC_MATH:-}
if [ -z "$sm" ]; then
    for found in ../symbolic-math/target/*-freebsd/release/symbolic-math; do
        if [ -x "$found" ]; then sm=$found; fi
    done
fi
if [ -n "$sm" ]; then
    [ -x "$sm" ] || { echo "deploy.sh: $sm is no program" >&2; exit 1; }
    echo "== copying $sm to $host:$remote/symbolic-math.upload"
    rsync -a "$sm" "$host:$remote/symbolic-math.upload"
else
    ssh "$host" "rm -f '$remote/symbolic-math.upload'"
fi

echo "== building tet and tet-web there"
ssh "$host" "cd '$remote' && nice -n 10 cargo build --release --locked --bin tet &&
    cd web/server && nice -n 10 cargo build --release"

echo "== installing"
log=$(mktemp)
trap 'rm -f "$log"' EXIT
ssh "$host" "cd '$remote' &&
    sudo env SITE='$site' REVISION='$revision' ACME_EMAIL='$email' sh web/freebsd/install.sh" | tee "$log"
# The last line of the installation is the address of the site.
url=$(sed -n 's|^== done: \(.*\)/$|\1|p' "$log" | tail -n 1)
[ -n "$url" ] || { echo "deploy.sh: the installation did not finish" >&2; exit 1; }

echo "== checking $url from here"
# A certificate that Caddy is still obtaining takes a moment.
waited=0
until curl -fsS --max-time 10 -o /dev/null "$url/healthz" 2>/dev/null || [ "$waited" -ge 120 ]; do
    sleep 5
    waited=$((waited + 5))
done
failed=no
check() {
    found=$(curl -sS --max-time 60 "$url$1") || found="curl failed"
    if [ "$found" = "$2" ]; then
        echo "ok   $1"
    else
        echo "FAIL $1: $found (expected $2)"
        failed=yes
    fi
}
# Where a request ends up that is not followed: its status and the address
# that it is sent to.
sent() {
    curl -sS --max-time 20 -o /dev/null -w '%{http_code} %{redirect_url}' "$1" || echo "curl failed"
}
check "/healthz" "ok"
check "/?prec=30&func=tet&base=2&height=3&format=text" "16.0000000000000000000000000000
0"
check "/?func=ate&base=2&value=65536&format=text" "4.0000000000000000000
0"
check "/?func=ssrt&value=16&height=3&format=text" "2.0000000000000000000
0"
check "/?func=tet&base=1.5+0.5i&height=1+0.7i&format=text" "1.1766384980936338172
6.0310294707303037947e-1"
# The second program, where it is installed.
if [ "$(curl -sS --max-time 20 -o /dev/null -w '%{http_code}' "$url/sm/help.txt")" = 200 ]; then
    check "/sm?expr=10*x-(2%2B3)*x&format=text" "5x"
    check "/sm?expr=x^2-5x%2B6=0%0A2^^3&format=text" "x = 2
x = 3
16"
    check "/sm?expr=N(pi,40)&format=text" "3.141592653589793238462643383279502884197"
    check "/sm?expr=1/3%2B1/6&digits=40&mode=numeric&format=text" "0.5"
    form="/sm?format=html&expr=i%5E2&digits=&mode=both&domain=complex"
    if curl -sS --max-time 60 "$url$form" | grep -q '<pre id="sm-output">-1'; then
        echo "ok   $form is the page with the result"
    else
        echo "FAIL $form is not the page with the result"
        failed=yes
    fi
    # A picture: the run names it, and it is a PNG file at its address.
    picture=$(curl -sS --max-time 120 "$url/sm?expr=save_png(%22p.png%22,plot(sin(x),x,0,7),width=300)" |
        sed -n 's|.*"url":"\(/sm/file/[0-9a-f]*/p.png\)".*|\1|p')
    if [ -n "$picture" ] && curl -sS --max-time 60 "$url$picture" | head -c 4 | tail -c 3 | grep -q PNG; then
        echo "ok   $picture is a PNG picture"
    else
        echo "FAIL the picture of a run is not served ($picture)"
        failed=yes
    fi
else
    echo "note: symbolic-math is not installed there: /sm is not checked"
fi
page=$(curl -sS --max-time 60 "$url/" | head -1)
if [ "$page" = "<!doctype html>" ]; then
    echo "ok   / is the page"
else
    echo "FAIL / is not the page: $page"
    failed=yes
fi
# The page as a browser without scripts gets it when its form is sent.
form="/?format=html&prec=20&func=ssrt&numbers=2&value=16&height=3"
if curl -sS --max-time 60 "$url$form" | grep -q '<dd id="re">2.0000000000000000000</dd>'; then
    echo "ok   $form is the page with the result"
else
    echo "FAIL $form is not the page with the result"
    failed=yes
fi
case "$url" in
    https://*)
        # By plain HTTP there is nothing but the way to https, for the name
        # and for the bare address.
        name=${url#https://}
        address=$(ssh "$host" "route -n get default | awk '/interface:/ { print \$2 }' |
            xargs ifconfig | awk '\$1 == \"inet\" { print \$2; exit }'")
        query="/?func=tet&base=2&height=3"
        for plain in "$name" "$address"; do
            found=$(sent "http://$plain$query")
            case "$found" in
                "301 $url$query" | "308 $url$query")
                    echo "ok   http://$plain only sends to $url (${found%% *})"
                    ;;
                *)
                    echo "FAIL http://$plain$query: $found (expected 301 or 308 to $url$query)"
                    failed=yes
                    ;;
            esac
        done
        ;;
esac
[ "$failed" = no ] || exit 1
echo "== $url/ serves the page and the API (revision $revision)"
