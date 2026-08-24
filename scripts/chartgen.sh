#!/usr/bin/env bash
# Copyright 2026. Licensed under the Apache License, Version 2.0.
#
# chartgen.sh — sweep f(x) = b^^x over real x and emit a CSV of
#   x, Re f(x), Im f(x)
# for the 3D charts in docs/charts/ (rendered by plot3d.py).
#
# Usage:
#   scripts/chartgen.sh <b_re> <b_im> <out.csv> [coarse]
#
# The grid is adapted to the cut-segment phenomenology (b real in
# (0, e^{-e}), evaluated as b + iε):
#   [-30, -8)   step 0.1   — inter-pole spiral tail
#   [-8,  -3)   step 0.04  — pole forest (dense)
#   [-3,   8)   step 0.05  — seam through h = -1, 0, 1 and the transient
#   [8,  120]   step 0.25  — 2-cycle weave decay
# Integer x are dodged by +0.013: for cut bases every integer x ≤ -2 is a
# genuine pole (the h+1 recurrence hits log_b(0)), and the CLI honestly
# errors there. `coarse` multiplies all steps by 4 (convergence checks).
#
# Points run 14-way parallel via xargs; each point is an independent `tet`
# invocation at 10 digits (chart accuracy), 400 s timeout. Failed points
# are recorded as ERR (plot3d.py breaks the curve there).
set -u

TET="$(dirname "$0")/../target/release/tet"
B_RE=$1; B_IM=$2; OUT=$3; COARSE=${4:-}

TMP=$(mktemp)
python3 - "${COARSE:+4}" <<'PY' > "$TMP"
import sys
m = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else 1
xs = []
def rng(a, b, s):
    x = a
    while x < b:
        xs.append(x); x += s
rng(-30.0, -8.0, 0.1*m)
rng(-8.0, -3.0, 0.04*m)
rng(-3.0, 8.0, 0.05*m)
rng(8.0, 120.25, 0.25*m)
for x in xs:
    if abs(x - round(x)) < 1e-9:
        x += 0.013
    print(f"{x:.4f}")
PY

xargs -P 14 -I{} bash -c '
v=$(SILENT=1 nice -n 12 timeout 400 "'"$TET"'" 10 "'"$B_RE"'" "'"$B_IM"'" {} 0 2>/dev/null)
if [ $? -eq 0 ]; then
  echo "{},$(echo "$v" | sed -n 1p),$(echo "$v" | sed -n 2p)"
else
  echo "{},ERR,ERR"
fi' < "$TMP" | sort -t, -k1 -g > "$OUT"
rm -f "$TMP"

echo "done $OUT: $(grep -vc ERR "$OUT") ok / $(grep -c ERR "$OUT") err"
