#!/usr/bin/env python3
"""Verify the reproduced proxy mesh against logged Metal fragments.

Interpolates each triangle's vertex positions at the pixel center (in Metal FB
coords) and compares with the logged fragment localPos values (STEP lines in
the post-winding-fix log). A fragment is "explained" when some covering cell
interpolates to the logged localPos. Fragments that do NOT match any full
triangle are frustum-clipped remnants (their localPos sits on the far face
z=1.0 near a shared far-face vertex); clipping is not modeled here.

Usage: python3 verify_fragments.py [data_dir] [log]
  data_dir : verts.txt/indices.txt/matrices.txt directory (default /tmp/bc/meshdump)
  log      : Metal stderr log to extract STEP localPos from (default
             /tmp/bc/update20/metal_proxy_cullfix.log)
"""
import re
import sys
import numpy as np

DATA = sys.argv[1] if len(sys.argv) > 1 else '/tmp/bc/meshdump'
LOG = sys.argv[2] if len(sys.argv) > 2 else '/tmp/bc/update20/metal_proxy_cullfix.log'

from winding import load_verts, load_indices, load_matrices, project, signed_area_fb

BSIZE = np.array([201.6, 201.6, 138.0])


def bary(pt, tri):
    v0 = tri[1] - tri[0]
    v1 = tri[2] - tri[0]
    v2 = pt - tri[0]
    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d20 = v2 @ v0
    d21 = v2 @ v1
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w])


def main():
    verts = load_verts(f"{DATA}/verts.txt")
    cells = load_indices(f"{DATA}/indices.txt")
    V, P = load_matrices(f"{DATA}/matrices.txt")
    clip, ndc, fb = project(verts, V, P)

    # Logged fragment localPos per pixel (STEP lines, first frame only): the
    # cap fragment (z~0.4486) and the far-face remnant (z=1.0).
    log = {}
    for line in open(LOG):
        m = re.search(r"STEP px=\((\d+), (\d+)\) .*?localPos=\(([-\d.eE+]+), ([-\d.eE+]+), ([-\d.eE+]+)\)", line)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        loc = (float(m.group(3)), float(m.group(4)), float(m.group(5)))
        if key not in log:
            log[key] = [loc]
        elif len(log[key]) < 4:
            log[key].append(loc)

    for (px, py), targets in sorted(log.items()):
        pt = np.array([px + 0.5, py + 0.5])
        print(f"\npx=({px},{py}) targets: {[(round(t[0], 4), round(t[1], 4), round(t[2], 4)) for t in targets]}")
        for (ci, npts, ids) in cells:
            tri = fb[ids]
            b = bary(pt, tri)
            if (b < -1e-9).any():
                continue
            loc = (verts[ids][:, 0] @ b, verts[ids][:, 1] @ b, verts[ids][:, 2] @ b)
            a = signed_area_fb(tri)
            print(f"  cell {ci:3d} A_FB={a:+14.1f} bary=({b[0]:.3f},{b[1]:.3f},{b[2]:.3f}) "
                  f"loc=({loc[0]:.6f},{loc[1]:.6f},{loc[2]:.6f})")


if __name__ == "__main__":
    main()
