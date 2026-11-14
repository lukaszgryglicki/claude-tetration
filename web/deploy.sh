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
