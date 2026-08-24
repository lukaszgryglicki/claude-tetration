#!/usr/bin/env bash
# Copyright 2026. Licensed under the Apache License, Version 2.0.
#
# chartgallery.sh — render the § 5.4 multi-view 3D gallery from the dense
# sweeps in docs/charts/data/*_dense.csv (produced by chartgen.sh with step
# multiplier 0.2, ~5070 points per base) via plot3d.py, plus the raster
# exports (PNG/JPG hero for sharing).
#
# Usage: scripts/chartgallery.sh
# Requires: python3 (plot3d.py is stdlib-only); rsvg-convert + ImageMagick
# for the raster hero (skipped with a note if missing).
set -eu
cd "$(dirname "$0")/.."
P="python3 scripts/plot3d.py"
D=docs/charts/data
O=docs/charts

C99=$D/b099eme_eps005_dense.csv
C100=$D/b100eme_eps005_dense.csv
C101=$D/b101eme_eps005_dense.csv
L99='b = 0.99·e^-e:#e8b04b'
L100='b = e^-e:#5fb0e8'
L101='b = 1.01·e^-e:#e86a6a'
SUB99='f(x) = b^^x, b = 0.99·e^-e + 0.05i, real heights x ∈ [-30, 120] — curve (x, Re F, Im F)'
SUBALL='f(x) = b^^x at b + 0.05i, b/e^-e ∈ {0.99, 1.00, 1.01} — real heights, complex values'

# ---- oblique hero views (full range), one per base + overlay
$P --az 35 --el 18 $O/tet3d_b099eme_dense.svg \
  'Tetration below the cut: b = 0.99·e^-e' "$SUB99" "$C99:$L99"
$P --az 35 --el 18 $O/tet3d_b100eme_dense.svg \
  'Tetration at the boundary: b = e^-e' \
  'f(x) = b^^x, b = e^-e + 0.05i, real heights x ∈ [-30, 120]' "$C100:$L100"
$P --az 35 --el 18 $O/tet3d_b101eme_dense.svg \
  'Tetration just above e^-e: b = 1.01·e^-e' \
  'f(x) = b^^x, b = 1.01·e^-e + 0.05i, real heights x ∈ [-30, 120]' "$C101:$L101"
$P --az 35 --el 18 $O/tet3d_triptych_dense.svg \
  'Three bases straddling e^-e' "$SUBALL" "$C99:$L99" "$C100:$L100" "$C101:$L101"

# ---- turntable series (hero base), four more azimuths
for AZ in 12 55 75 90; do
  $P --az "$AZ" --el 18 $O/tet3d_b099eme_az${AZ}.svg \
    "Turntable az=${AZ}°: b = 0.99·e^-e" "$SUB99" "$C99:$L99"
done
$P --az 35 --el 62 $O/tet3d_b099eme_top.svg \
  'High camera (el=62°): b = 0.99·e^-e' "$SUB99" "$C99:$L99"

# ---- down-the-x-axis portraits: the pure complex-plane swirl
$P --az 0 --el 0 --xrange 2:120 $O/tet3d_b099eme_endon_weave.svg \
  'The 2-cycle weave, end-on: b = 0.99·e^-e' \
  'x ∈ [2, 120] seen straight down the x axis — period-2 spiral converging to L' "$C99:$L99"
$P --az 0 --el 0 --xrange 2:120 $O/tet3d_triptych_endon_weave.svg \
  'The weave end-on, three bases' \
  'x ∈ [2, 120] down the x axis; the three spirals land on different fixed points' \
  "$C99:$L99" "$C100:$L100" "$C101:$L101"
$P --az 0 --el 0 --xrange -30:-3 $O/tet3d_b099eme_endon_forest.svg \
  'The pole forest, end-on: b = 0.99·e^-e' \
  'x ∈ [-30, -3] down the x axis — nested pole loops' "$C99:$L99"

# ---- region close-ups (oblique)
$P --az 30 --el 14 --xrange 2:40 $O/tet3d_b099eme_weave_closeup.svg \
  'Weave close-up: x ∈ [2, 40]' "$SUB99" "$C99:$L99"
$P --az 42 --el 24 --xrange -9:0 $O/tet3d_b099eme_forest_closeup.svg \
  'Pole-forest close-up: x ∈ [-9, 0]' "$SUB99" "$C99:$L99"
$P --az 25 --el 12 --xrange -3:12 --dot-ends $O/tet3d_b099eme_seam.svg \
  'The seam: x ∈ [-3, 12]' "$SUB99" "$C99:$L99"

# ---- raster hero for sharing (FB): near-axial vortex view, high res
if command -v rsvg-convert >/dev/null 2>&1; then
  $P --az 14 --el 10 --xrange -4.5:120 --size 3000x1875 $O/tet3d_hero.svg \
    'Complex tetration f(x) = b^^x near the base boundary e^-e' \
    'complex base b = 0.99·e^-e + 0.05i, real heights x ∈ [-4.5, 120] — curve (x, Re F, Im F) seen nearly down the height axis: the period-2 spiral drains into the fixed point' \
    "$C99:$L99"
  rsvg-convert -w 3000 $O/tet3d_hero.svg -o /tmp/tet3d_hero.png
  if command -v magick >/dev/null 2>&1; then
    magick /tmp/tet3d_hero.png -quality 94 $O/tet3d_hero.jpg
    rm -f /tmp/tet3d_hero.png
  else
    mv /tmp/tet3d_hero.png $O/tet3d_hero.png
  fi
else
  echo "note: rsvg-convert missing — raster hero skipped"
fi

echo "gallery done: $(ls -1 $O/*.svg | wc -l) SVGs$(ls $O/tet3d_hero.jpg 2>/dev/null | sed 's/.*/ + hero JPG/')"
