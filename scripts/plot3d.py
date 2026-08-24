#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Dependency-free 3D curve plotter for tetration sweeps (v2).

Renders (x, Re F, Im F) triples from CSV files as an SVG orthographic
3D line with a real turntable camera, painter-sorted depth shading and
an *isotropic* complex plane (Re and Im share one scale and the screen
fit is uniform, so the period-2 spiral weave projects to true circles
instead of ellipses/polygons).

Usage:
  plot3d.py [options] out.svg title subtitle spec [spec ...]

Each spec is  csv_path:label:color  where csv_path has lines `x,re,im`
(ERR / non-finite / |f| > 50 rows break the curve). Options:

  --az DEG      turntable azimuth about the Re-axis (vertical). 0 looks
                straight down the x axis: the pure complex-plane portrait
                (swirls become circles). 90 is the classic side view
                (x horizontal, Re F up). Default 35.
  --el DEG      camera elevation. 0 = level, 90 = top-down (x vs Im F).
                Default 18.
  --size WxH    canvas in px. Default 2000x1250.
  --xrange A:B  crop to A <= x <= B before plotting (region portraits).
  --xscale S    world units of x per unit of |F| before projection
                (x is auto-compressed to ~2.2x the complex span if unset;
                irrelevant at --az 0).
  --no-shadows  skip floor/wall shadow projections.
  --dot-ends    mark first/last point of each curve.

World frame: X = x (scaled), Y = Im F (depth at az 90), Z = Re F (up).
No external dependencies; pure stdlib.
"""
import argparse
import math
import sys

BG = (0x10, 0x14, 0x18)
CLIP = 50.0


def read_csv(path, xr):
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
                x, re, im = (float(v) for v in parts)
            except ValueError:
                if cur:
                    segs.append(cur)
                    cur = []
                continue
            bad = not (math.isfinite(x) and math.isfinite(re) and math.isfinite(im))
            if not bad and xr and not (xr[0] <= x <= xr[1]):
                bad = True
            if bad or math.hypot(re, im) > CLIP:
                if cur:
                    segs.append(cur)
                    cur = []
                continue
            cur.append((x, re, im))
    if cur:
        segs.append(cur)
    return segs


def hexrgb(s):
    s = s.lstrip("#")
    return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))


def mix(c, t):
    """Mix color toward background: t=0 -> c, t=1 -> BG."""
    return "#%02x%02x%02x" % tuple(
        round(c[k] + (BG[k] - c[k]) * t) for k in range(3)
    )


class Camera:
    def __init__(self, az_deg, el_deg):
        a, e = math.radians(az_deg), math.radians(el_deg)
        ca, sa, ce, se = math.cos(a), math.sin(a), math.cos(e), math.sin(e)
        # Screen u (right), v (up), d (toward viewer) for world (X, Y, Z):
        #   az rotates about Z-up... we want az about the *vertical screen*
        #   axis with Z = Re F kept upright, so rotate in the X-Y plane.
        self.u = (sa, ca, 0.0)   # az 0: u = Y (Im F right)  az 90: u = X
        f = (ca, -sa, 0.0)       # horizontal forward
        # elevate: tilt forward down by el around u
        self.d = (f[0] * ce, f[1] * ce, se)          # toward viewer
        self.v = (-f[0] * se, -f[1] * se, ce)        # up
        if abs(az_deg) < 1e-9 and abs(el_deg) < 1e-9:
            self.u = (0.0, 1.0, 0.0)
            self.v = (0.0, 0.0, 1.0)
            self.d = (1.0, 0.0, 0.0)

    def proj(self, p):
        return (
            p[0] * self.u[0] + p[1] * self.u[1] + p[2] * self.u[2],
            p[0] * self.v[0] + p[1] * self.v[1] + p[2] * self.v[2],
            p[0] * self.d[0] + p[1] * self.d[1] + p[2] * self.d[2],
        )


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--az", type=float, default=35.0)
    ap.add_argument("--el", type=float, default=18.0)
    ap.add_argument("--size", default="2000x1250")
    ap.add_argument("--xrange", default=None)
    ap.add_argument("--xscale", type=float, default=None)
    ap.add_argument("--no-shadows", action="store_true")
    ap.add_argument("--dot-ends", action="store_true")
    ap.add_argument("out")
    ap.add_argument("title")
    ap.add_argument("subtitle")
    ap.add_argument("specs", nargs="+")
    args = ap.parse_args()

    W, H = (int(v) for v in args.size.lower().split("x"))
    margin = max(70, W // 22)
    xr = None
    if args.xrange:
        a, b = (float(v) for v in args.xrange.split(":"))
        xr = (min(a, b), max(a, b))

    curves = []
    for spec in args.specs:
        path, label, color = spec.rsplit(":", 2)
        segs = read_csv(path, xr)
        if segs:
            curves.append((segs, label, hexrgb(color)))
    pts = [p for segs, _, _ in curves for s in segs for p in s]
    if not pts:
        sys.exit("no data")

    # ---- world scaling: Re/Im isotropic, x compressed to a companion span
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    cspan = max(
        max(p[1] for p in pts) - min(p[1] for p in pts),
        max(p[2] for p in pts) - min(p[2] for p in pts),
        1e-9,
    )
    if args.xscale is not None:
        xs = args.xscale
    else:
        xs = (2.2 * cspan) / ((x1 - x0) or 1.0)

    def world(p):
        return (p[0] * xs, p[2], p[1])  # X = x, Y = Im, Z = Re

    cam = Camera(args.az, args.el)

    # ---- uniform screen fit (single scale for u and v: angles preserved)
    prj = [cam.proj(world(p)) for p in pts]
    u0, u1 = min(q[0] for q in prj), max(q[0] for q in prj)
    v0, v1 = min(q[1] for q in prj), max(q[1] for q in prj)
    d0, d1 = min(q[2] for q in prj), max(q[2] for q in prj)
    s = min(
        (W - 2 * margin) / ((u1 - u0) or 1e-9),
        (H - 2 * margin) / ((v1 - v0) or 1e-9),
    )
    ucx, vcx = (u0 + u1) / 2, (v0 + v1) / 2

    def scr(q):
        return (
            W / 2 + (q[0] - ucx) * s,
            H / 2 - (q[1] - vcx) * s,
        )

    def depth_t(q):
        if d1 - d0 < 1e-12:
            return 0.0
        return 1.0 - (q[2] - d0) / (d1 - d0)  # 0 near .. 1 far

    def spro(p):
        return cam.proj(world(p))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Georgia,serif">',
        f'<rect width="{W}" height="{H}" fill="#101418"/>',
        f'<text x="{W/2}" y="{margin*0.55:.0f}" fill="#e8e2d5" font-size="{W//60}" '
        f'text-anchor="middle">{args.title}</text>',
        f'<text x="{W/2}" y="{margin*0.55 + W//55:.0f}" fill="#9aa3ad" '
        f'font-size="{W//110}" text-anchor="middle">{args.subtitle}</text>',
    ]

    # ---- axes (world lines through the data box edges)
    r0 = min(p[1] for p in pts)
    r1 = max(p[1] for p in pts)
    i0 = min(p[2] for p in pts)
    i1 = max(p[2] for p in pts)
    ax_re = 0.0 if r0 <= 0.0 <= r1 else r0
    ax_im = 0.0 if i0 <= 0.0 <= i1 else i0

    def line3(pa, pb, dash=None, w=1.4, col="#5c6670", op=1.0):
        A, B = scr(spro(pa)), scr(spro(pb))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(
            f'<line x1="{A[0]:.1f}" y1="{A[1]:.1f}" x2="{B[0]:.1f}" y2="{B[1]:.1f}" '
            f'stroke="{col}" stroke-width="{w}" opacity="{op}"{d}/>'
        )
        return A, B

    down_axis = abs(args.az) < 3 and abs(args.el) < 3
    if not down_axis:
        _, xe = line3((x0, ax_re, ax_im), (x1, ax_re, ax_im))
        svg.append(
            f'<text x="{xe[0]+8:.1f}" y="{xe[1]+5:.1f}" fill="#8fa0b0" '
            f'font-size="{W//110}">x</text>'
        )
        # x ticks
        span = x1 - x0
        step = 10 ** math.floor(math.log10(span / 6))
        for m in (1, 2, 5, 10):
            if span / (step * m) <= 9:
                step *= m
                break
        t = math.ceil(x0 / step) * step
        while t <= x1:
            T = scr(spro((t, ax_re, ax_im)))
            svg.append(
                f'<line x1="{T[0]:.1f}" y1="{T[1]-4:.1f}" x2="{T[0]:.1f}" '
                f'y2="{T[1]+4:.1f}" stroke="#5c6670" stroke-width="1.2"/>'
            )
            svg.append(
                f'<text x="{T[0]:.1f}" y="{T[1]+20:.1f}" fill="#7d8894" '
                f'font-size="{W//140}" text-anchor="middle">{t:g}</text>'
            )
            t += step
    xa = x0 if not down_axis else (x0 + x1) / 2
    _, re_e = line3((xa, r0, ax_im), (xa, r1, ax_im))
    svg.append(
        f'<text x="{re_e[0]-10:.1f}" y="{re_e[1]-8:.1f}" fill="#8fa0b0" '
        f'font-size="{W//110}">Re F</text>'
    )
    _, im_e = line3((xa, ax_re, i0), (xa, ax_re, i1), dash="5,3")
    svg.append(
        f'<text x="{im_e[0]+6:.1f}" y="{im_e[1]-6:.1f}" fill="#8fa0b0" '
        f'font-size="{W//110}">Im F</text>'
    )
    if down_axis:
        # unit-scale grid rings centred on the fixed-point region for the
        # complex-plane portrait
        for rad in (0.5, 1.0, 1.5, 2.0):
            if rad > max(abs(r0), abs(r1), abs(i0), abs(i1)) * 1.2:
                break
            C = scr(spro((xa, 0.0, 0.0)))
            svg.append(
                f'<circle cx="{C[0]:.1f}" cy="{C[1]:.1f}" r="{rad*s:.1f}" '
                f'fill="none" stroke="#2a333d" stroke-width="1" '
                f'stroke-dasharray="3,6"/>'
            )
            svg.append(
                f'<text x="{C[0]+rad*s+4:.1f}" y="{C[1]:.1f}" fill="#4d5a66" '
                f'font-size="{W//150}">|F|={rad:g}</text>'
            )

    # ---- pole markers
    if x0 < -2 and not down_axis:
        k, first = -2, True
        while k >= x0:
            if k <= x1:
                A, B = line3(
                    (k, r0, ax_im), (k, r1, ax_im),
                    dash="2,5", w=0.8, col="#7a3b3b", op=0.35,
                )
                if first:
                    svg.append(
                        f'<text x="{B[0]:.1f}" y="{B[1]-6:.1f}" fill="#b06060" '
                        f'font-size="{W//140}" text-anchor="middle">poles at '
                        f'integer x &#8804; &#8722;2</text>'
                    )
                    first = False
            k -= 1

    # ---- shadows (first curve): floor Z->r0 plane, wall Y->i0
    if not args.no_shadows and not down_axis:
        for seg in curves[0][0]:
            if len(seg) < 2:
                continue
            for repl in (lambda p: (p[0], r0, p[2]), lambda p: (p[0], p[1], i0)):
                d = " ".join(
                    f"{'M' if k == 0 else 'L'}{scr(spro(repl(p)))[0]:.1f},"
                    f"{scr(spro(repl(p)))[1]:.1f}"
                    for k, p in enumerate(seg)
                )
                svg.append(
                    f'<path d="{d}" fill="none" stroke="#232c36" '
                    f'stroke-width="1.0"/>'
                )

    # ---- curves: chunked painter's algorithm with depth-shaded strokes
    CHUNK = 6
    chunks = []
    for ci, (segs, label, rgb) in enumerate(curves):
        for seg in segs:
            if len(seg) < 2:
                continue
            for a in range(0, len(seg) - 1, CHUNK):
                part = seg[a : a + CHUNK + 1]
                q = [spro(p) for p in part]
                dm = sum(v[2] for v in q) / len(q)
                chunks.append((dm, ci, q))
    chunks.sort(key=lambda c: c[0])  # far first
    for dm, ci, q in chunks:
        rgb = curves[ci][2]
        t = depth_t((0, 0, dm))
        col = mix(rgb, 0.15 + 0.55 * t)
        wd = 2.6 - 1.3 * t
        d = " ".join(
            f"{'M' if k == 0 else 'L'}{scr(v)[0]:.1f},{scr(v)[1]:.1f}"
            for k, v in enumerate(q)
        )
        svg.append(
            f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{wd:.2f}" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    if args.dot_ends:
        for segs, _, rgb in curves:
            flat = [p for s in segs for p in s]
            if not flat:
                continue
            for p, r in ((flat[0], 5), (flat[-1], 3.4)):
                P = scr(spro(p))
                svg.append(
                    f'<circle cx="{P[0]:.1f}" cy="{P[1]:.1f}" r="{r}" '
                    f'fill="{mix(rgb, 0.0)}" stroke="#e8e2d5" stroke-width="1"/>'
                )

    # ---- legend
    ly = margin + 14
    for _, label, rgb in curves:
        svg.append(
            f'<line x1="{W-330}" y1="{ly}" x2="{W-296}" y2="{ly}" '
            f'stroke="{mix(rgb, 0.0)}" stroke-width="3"/>'
        )
        svg.append(
            f'<text x="{W-288}" y="{ly+5}" fill="#c9c2b4" '
            f'font-size="{W//130}">{label}</text>'
        )
        ly += 26
    svg.append(
        f'<text x="{W-330}" y="{ly+4}" fill="#6d7884" font-size="{W//160}">'
        f'az={args.az:g}&#176; el={args.el:g}&#176; &#8226; near = bright, '
        f'far = faded</text>'
    )

    svg.append("</svg>")
    with open(args.out, "w") as fh:
        fh.write("\n".join(svg))
    n = sum(len(sg) for segs, _, _ in curves for sg in segs)
    print(f"wrote {args.out}: {n} points, {len(curves)} curve(s), az={args.az} el={args.el}")


if __name__ == "__main__":
    main()
