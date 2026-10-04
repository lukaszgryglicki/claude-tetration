#!/usr/bin/env bash
# Copyright 2026. Licensed under the Apache License, Version 2.0.
#
# chartgen.sh — sweep f(x) = b^^x over real x and emit a CSV of
#   x, Re f(x), Im f(x)
# for the 3D charts in docs/charts/ (rendered by plot3d.py).
#
# Usage:
#   scripts/chartgen.sh <b_re> <b_im> <out.csv> [coarse|step_multiplier]
#
# The grid is adapted to the cut-segment phenomenology (b real in
# (0, e^{-e}), evaluated as b + iε):
#   [-30, -8)   step 0.1   — negative-height tail
#   [-8,  -3)   step 0.04  — singularity region (dense)
#   [-3,   8)   step 0.05  — seam through h = -1, 0, 1 and the transient
#   [8,  120]   step 0.25  — 2-cycle weave decay
# Integer x <= -2 are included explicitly so errors break plotted curves.
# `coarse` multiplies steps by 4. These qualitative 10-digit sweeps are not
# independent accuracy certificates.
#
# Points use at most four processes, or one process if inner TET_MT is on.
# Failed points are ERR rows, with reasons in <out.csv>.errors.log; the
# overall exit status is nonzero. SILENT=1 suppresses progress, not errors.
set -eu

TET="$(dirname "$0")/../target/release/tet"
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "usage: chartgen.sh <b_re> <b_im> <out.csv> [coarse|step_multiplier]" >&2
  exit 2
fi
MULT=${4:-1}
case "$MULT" in coarse) MULT=4 ;; esac
python3 - "$TET" "$1" "$2" "$3" "$MULT" <<'PY'
from concurrent.futures import ThreadPoolExecutor
import csv
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
import os
import subprocess
import sys

tet, base_re, base_im, output, raw_multiplier = sys.argv[1:]
try:
    multiplier = Decimal(raw_multiplier)
    base = (Decimal(base_re), Decimal(base_im))
except InvalidOperation as error:
    raise SystemExit(f"invalid decimal input: {error}")
if not all(x.is_finite() for x in base):
    raise SystemExit("base components must be finite")
if not multiplier.is_finite() or multiplier <= 0:
    raise SystemExit("step multiplier must be finite and positive")
if abs(multiplier.adjusted()) > 10000:
    raise SystemExit("step multiplier exceeds the sweep resource range")
multiplier = Fraction(multiplier)
pieces = [(-30, -8, "0.1"), (-8, -3, "0.04"), (-3, 8, "0.05"), (8, 120, "0.25")]
counts = []
for lo, hi, step in pieces:
    count = Fraction(hi - lo) / (Fraction(step) * multiplier)
    counts.append(-(-count.numerator // count.denominator))
if sum(counts) + 30 > 1_000_000:
    raise SystemExit("sweep exceeds the 1000000-point resource budget")
points = {Fraction(x) for x in range(-30, -1)}
points.add(Fraction(120))
for (lo, hi, step), count in zip(pieces, counts):
    points.update(Fraction(lo) + i * Fraction(step) * multiplier for i in range(count))


def decimal_text(value):
    with localcontext() as context:
        context.prec = len(str(abs(value.numerator))) + len(str(value.denominator)) + 8
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def evaluate(height):
    text = decimal_text(height)
    try:
        result = subprocess.run(
            [tet, "10", base_re, base_im, text, "0"],
            capture_output=True, text=True, timeout=400,
        )
    except subprocess.TimeoutExpired:
        return text, None, "tetration timed out after 400 seconds", ""
    except OSError as error:
        return text, None, str(error), ""
    lines = result.stdout.splitlines()
    valid = result.returncode == 0 and len(lines) == 2
    if valid:
        try:
            valid = all(Decimal(value).is_finite() for value in lines)
        except InvalidOperation:
            valid = False
    if not valid:
        reason = result.stderr.strip() or f"invalid numerical output or exit status {result.returncode}"
        return text, None, reason, ""
    return text, lines, None, result.stderr


workers = 1 if os.environ.get("TET_MT", "").strip() not in ("", "0") else min(4, os.cpu_count() or 1)
errors = 0
heights = sorted(points)
with ThreadPoolExecutor(max_workers=workers) as pool:
    with open(output, "w", newline="") as csv_file, open(output + ".errors.log", "w") as error_file:
        writer = csv.writer(csv_file)
        for start in range(0, len(heights), workers * 2):
            for height, values, error, diagnostics in pool.map(evaluate, heights[start:start + workers * 2]):
                if diagnostics:
                    sys.stderr.write(diagnostics)
                if error is None:
                    writer.writerow([height, *values])
                else:
                    errors += 1
                    writer.writerow([height, "ERR", "ERR"])
                    error_file.write(f"height={height}: {error}\n")
                    print(f"height={height}: {error}", file=sys.stderr)
if os.environ.get("SILENT", "").strip().lower() not in ("1", "t", "true", "y", "yes", "on"):
    print(f"done {output}: {len(points) - errors} ok / {errors} err")
sys.exit(int(errors != 0))
PY
