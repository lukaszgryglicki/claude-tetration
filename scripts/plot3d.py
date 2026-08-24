#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Dependency-free 3D curve plotter for tetration sweeps.

Renders (x, Re F, Im F) triples from CSV files as an SVG cabinet-projection
3D line, with floor/back-wall shadow projections and pole markers.

Usage:
  plot3d.py out.svg title subtitle spec [spec ...]

Each spec is  csv_path:label:color  where csv_path has lines `x,re,im`
(rows with ERR are treated as curve breaks). Pole markers are drawn at
integer x <= -2 (cut-base pole forest) when the x range covers them.

No external dependencies (matplotlib unavailable offline); pure stdlib.
"""
import math
import sys


def read_csv(path):
    segs, cur = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) != 3 or "ERR" in parts:
                if cur:
                    segs.append(cur)
                    cur = []
                continue
            try:
                x, re, im = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                if cur:
                    segs.append(cur)
                    cur = []
                continue
            if not (math.isfinite(x) and math.isfinite(re) and math.isfinite(im)):
                if cur:
                    segs.append(cur)
                    cur = []
                continue
            # Clip tower escapes (|F| beyond any plot scale) into curve breaks;
            # the raw CSV keeps the exact values.
            if math.hypot(re, im) > 50.0:
                if cur:
                    segs.append(cur)
                    cur = []
                continue
            cur.append((x, re, im))
    if cur:
        segs.append(cur)
    return segs


W, H = 1400, 900
MARGIN = 90
# Cabinet projection: Im axis (depth) foreshortened at 50%, 30 degrees.
DEPTH = 0.5
CA, SA = math.cos(math.radians(30)), math.sin(math.radians(30))


def make_proj(all_pts):
    xs = [p[0] for p in all_pts]
    res = [p[1] for p in all_pts]
    ims = [p[2] for p in all_pts]
    x0, x1 = min(xs), max(xs)
    r0, r1 = min(res), max(res)
    i0, i1 = min(ims), max(ims)
    # pad ranges
    def pad(a, b):
        d = (b - a) or 1.0
        return a - 0.05 * d, b + 0.05 * d
    x0, x1 = pad(x0, x1)
    r0, r1 = pad(r0, r1)
    i0, i1 = pad(i0, i1)
    # world scale: x -> horizontal span, re -> vertical span, im -> depth
    plot_w = W - 2 * MARGIN
    plot_h = H - 2 * MARGIN
    depth_px = DEPTH * plot_h * 0.9

    def proj(x, re, im):
        u = (x - x0) / (x1 - x0)              # 0..1 along x
        v = (re - r0) / (r1 - r0)             # 0..1 along Re (up)
        w = (im - i0) / (i1 - i0)             # 0..1 along Im (depth)
        sx = MARGIN + u * (plot_w - depth_px * CA) + w * depth_px * CA
        sy = H - MARGIN - v * (plot_h - depth_px * SA) - w * depth_px * SA
        return sx, sy

    return proj, (x0, x1, r0, r1, i0, i1)


def path_of(seg, proj, get):
    d = []
    for k, p in enumerate(seg):
        sx, sy = proj(*get(p))
        d.append(f"{'M' if k == 0 else 'L'}{sx:.1f},{sy:.1f}")
    return " ".join(d)


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    out, title, subtitle = sys.argv[1], sys.argv[2], sys.argv[3]
    curves = []
    for spec in sys.argv[4:]:
        path, label, color = spec.rsplit(":", 2)
        curves.append((read_csv(path), label, color))

    all_pts = [p for segs, _, _ in curves for s in segs for p in s]
    if not all_pts:
        sys.exit("no data")
    proj, (x0, x1, r0, r1, i0, i1) = make_proj(all_pts)

    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Georgia,serif">'
    )
    svg.append(f'<rect width="{W}" height="{H}" fill="#101418"/>')
    svg.append(
        f'<text x="{W/2}" y="42" fill="#e8e2d5" font-size="26" '
        f'text-anchor="middle">{title}</text>'
    )
    svg.append(
        f'<text x="{W/2}" y="68" fill="#9aa3ad" font-size="15" '
        f'text-anchor="middle">{subtitle}</text>'
    )

    # Axes: x (Re=0 plane if 0 in range else r0), Re, Im — drawn as 3D box edges.
    ax_re = 0.0 if r0 <= 0.0 <= r1 else r0
    ax_im = 0.0 if i0 <= 0.0 <= i1 else i0
    # x axis
    p1, p2 = proj(x0, ax_re, ax_im), proj(x1, ax_re, ax_im)
    svg.append(
        f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
        f'stroke="#5c6670" stroke-width="1.4"/>'
    )
    svg.append(
        f'<text x="{p2[0]+8:.1f}" y="{p2[1]+5:.1f}" fill="#8fa0b0" font-size="15">x</text>'
    )
    # Re axis at x0
    p3 = proj(x0, r1, ax_im)
    p0 = proj(x0, r0, ax_im)
    svg.append(
        f'<line x1="{p0[0]:.1f}" y1="{p0[1]:.1f}" x2="{p3[0]:.1f}" y2="{p3[1]:.1f}" '
        f'stroke="#5c6670" stroke-width="1.4"/>'
    )
    svg.append(
        f'<text x="{p3[0]-10:.1f}" y="{p3[1]-8:.1f}" fill="#8fa0b0" font-size="15">Re F</text>'
    )
    # Im axis at x0
    p4 = proj(x0, ax_re, i1)
    p5 = proj(x0, ax_re, i0)
    svg.append(
        f'<line x1="{p5[0]:.1f}" y1="{p5[1]:.1f}" x2="{p4[0]:.1f}" y2="{p4[1]:.1f}" '
        f'stroke="#5c6670" stroke-width="1.4" stroke-dasharray="5,3"/>'
    )
    svg.append(
        f'<text x="{p4[0]+6:.1f}" y="{p4[1]-6:.1f}" fill="#8fa0b0" font-size="15">Im F</text>'
    )

    # x ticks every nice interval
    span = x1 - x0
    step = 10 ** math.floor(math.log10(span / 6))
    for mult in (1, 2, 5, 10):
        if span / (step * mult) <= 9:
            step *= mult
            break
    t = math.ceil(x0 / step) * step
    while t <= x1:
        tp = proj(t, ax_re, ax_im)
        svg.append(
            f'<line x1="{tp[0]:.1f}" y1="{tp[1]-4:.1f}" x2="{tp[0]:.1f}" y2="{tp[1]+4:.1f}" '
            f'stroke="#5c6670" stroke-width="1.2"/>'
        )
        svg.append(
            f'<text x="{tp[0]:.1f}" y="{tp[1]+20:.1f}" fill="#7d8894" font-size="12" '
            f'text-anchor="middle">{t:g}</text>'
        )
        t += step
    # Re ticks
    rspan = r1 - r0
    rstep = 10 ** math.floor(math.log10(rspan / 4))
    for mult in (1, 2, 5, 10):
        if rspan / (rstep * mult) <= 6:
            rstep *= mult
            break
    t = math.ceil(r0 / rstep) * rstep
    while t <= r1:
        tp = proj(x0, t, ax_im)
        svg.append(
            f'<line x1="{tp[0]-4:.1f}" y1="{tp[1]:.1f}" x2="{tp[0]+4:.1f}" y2="{tp[1]:.1f}" '
            f'stroke="#5c6670" stroke-width="1.2"/>'
        )
        svg.append(
            f'<text x="{tp[0]-8:.1f}" y="{tp[1]+4:.1f}" fill="#7d8894" font-size="12" '
            f'text-anchor="end">{t:g}</text>'
        )
        t += rstep

    # Pole markers: integer x <= -2 within range (cut-base pole forest).
    if x0 < -2:
        k = -2
        drew_label = False
        while k >= x0:
            if k <= x1:
                pp0 = proj(k, r0, ax_im)
                pp1 = proj(k, r1, ax_im)
                svg.append(
                    f'<line x1="{pp0[0]:.1f}" y1="{pp0[1]:.1f}" x2="{pp1[0]:.1f}" '
                    f'y2="{pp1[1]:.1f}" stroke="#7a3b3b" stroke-width="0.8" '
                    f'stroke-dasharray="2,5" opacity="0.35"/>'
                )
                if not drew_label:
                    svg.append(
                        f'<text x="{pp1[0]:.1f}" y="{pp1[1]-6:.1f}" fill="#b06060" '
                        f'font-size="12" text-anchor="middle">poles at integer x ≤ −2</text>'
                    )
                    drew_label = True
            k -= 1

    # Shadows (floor: im -> i0 plane; back wall: re -> r0) for first curve only.
    segs0 = curves[0][0]
    for seg in segs0:
        if len(seg) < 2:
            continue
        svg.append(
            f'<path d="{path_of(seg, proj, lambda p: (p[0], p[1], i0))}" fill="none" '
            f'stroke="#2c3540" stroke-width="1.1"/>'
        )
        svg.append(
            f'<path d="{path_of(seg, proj, lambda p: (p[0], r0, p[2]))}" fill="none" '
            f'stroke="#232c36" stroke-width="1.1"/>'
        )

    # Curves.
    for segs, label, color in curves:
        for seg in segs:
            if len(seg) < 2:
                continue
            svg.append(
                f'<path d="{path_of(seg, proj, lambda p: p)}" fill="none" '
                f'stroke="{color}" stroke-width="1.8" stroke-linejoin="round"/>'
            )

    # Legend.
    ly = 100
    for _, label, color in curves:
        svg.append(
            f'<line x1="{W-330}" y1="{ly}" x2="{W-296}" y2="{ly}" stroke="{color}" '
            f'stroke-width="3"/>'
        )
        svg.append(
            f'<text x="{W-288}" y="{ly+5}" fill="#c9c2b4" font-size="14">{label}</text>'
        )
        ly += 24
    svg.append(
        f'<text x="{W-330}" y="{ly+6}" fill="#6d7884" font-size="12">shadows: floor = (x, Re F), '
        f'wall = (x, Im F)</text>'
    )

    svg.append("</svg>")
    with open(out, "w") as fh:
        fh.write("\n".join(svg))
    n = sum(len(s) for segs, _, _ in curves for s in segs)
    print(f"wrote {out}: {n} points, {len(curves)} curve(s)")


if __name__ == "__main__":
    main()
