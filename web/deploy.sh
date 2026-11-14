#!/bin/sh
# Builds tet and tet-web on a FreeBSD machine and installs the web page and
# API there (web/freebsd/install.sh). Run it from a checkout of the
# repository; the working tree is copied as it is, committed or not.
#
#   web/deploy.sh                  to lgryglicki@ovh, plain HTTP on its address
#   SITE=tet.sh web/deploy.sh      the same machine at https://tet.sh only
#   web/deploy.sh user@host        another machine
#
# SITE                 ip (default), or the domain name that points to the machine
# ACME_EMAIL           contact address for the certificate authority (optional)
# TET_WEB_REMOTE_DIR   where the tree is copied to, in the home there (tet-web/src)
#
# It needs ssh access to a user who may run sudo without a password there,
# and cargo, rsync and a C toolchain on that machine.

set -eu

cd "$(dirname "$0")/.."
host=${1:-lgryglicki@ovh}
site=${SITE:-ip}
remote=${TET_WEB_REMOTE_DIR:-tet-web/src}
email=${ACME_EMAIL:-}

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

echo "== building tet and tet-web there"
ssh "$host" "cd '$remote' && nice -n 10 cargo build --release --locked --bin tet &&
    cd web/server && nice -n 10 cargo build --release"

echo "== installing"
ssh "$host" "cd '$remote' &&
    sudo env SITE='$site' REVISION='$revision' ACME_EMAIL='$email' sh web/freebsd/install.sh" |
    tee /dev/stderr | grep '^== done: ' > /dev/null

if [ "$site" = ip ]; then
    address=$(ssh "$host" "route -n get default | awk '/interface:/ { print \$2 }' |
        xargs ifconfig | awk '\$1 == \"inet\" { print \$2; exit }'")
    url="http://$address"
else
    url="https://$site"
fi

echo "== checking $url from here"
# A certificate that Caddy is still obtaining takes a moment.
waited=0
until curl -fsS --max-time 10 -o /dev/null "$url/healthz" 2>/dev/null || [ "$waited" -ge 120 ]; do
    sleep 5
    waited=$((waited + 5))
done
check() {
    found=$(curl -sS --max-time 60 "$url$1") || found="curl failed"
    if [ "$found" = "$2" ]; then
        echo "ok   $1"
    else
        echo "FAIL $1: $found (expected $2)"
        failed=yes
    fi
}
failed=no
check "/healthz" "ok"
check "/?prec=30&func=tet&base=2&height=3&format=text" "16.0000000000000000000000000000
0"
check "/?func=ate&base=2&value=65536&format=text" "4.0000000000000000000
0"
check "/?func=ssrt&value=16&height=3&format=text" "2.0000000000000000000
0"
check "/?func=tet&base=1.5+0.5i&height=1+0.7i&format=text" "1.1766384980936338172
6.0310294707303037947e-1"
page=$(curl -sS --max-time 60 "$url/" | head -1)
if [ "$page" = "<!doctype html>" ]; then
    echo "ok   / is the page"
else
    echo "FAIL / is not the page: $page"
    failed=yes
fi
[ "$failed" = no ] || exit 1
echo "== $url/ serves the page and the API (revision $revision)"
