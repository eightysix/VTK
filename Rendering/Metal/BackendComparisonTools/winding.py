#!/usr/bin/env python3
"""Analyze proxy mesh winding in Metal framebuffer space.

Loads the reproduced mesh (verts.txt, indices.txt) + camera matrices
(matrices.txt) dumped by dump_proxy_mesh.cxx, projects triangles to Metal FB
coords (y-down, top-left origin, 512x512 viewport), finds which triangles cover
given pixel centers, and reports each covering triangle's FB signed area
(A_FB, y-down; positive = counter-clockwise in the y-down framebuffer, which is
"back" under vtkMetalGPUVolumeRayCastMapper's MTLWindingClockwise + cull-Back),
the y-up window signed area (A_win = -A_FB, the OpenGL convention where CCW is
front), and which convention keeps/culls each triangle.

Usage:
  python3 winding.py [data_dir] [px,py ...]
    data_dir : directory holding verts.txt/indices.txt/matrices.txt
               (default /tmp/bc/meshdump)
    px,py... : pixel centers to test (default the update-21 gated set)
"""
import numpy as np
import re
import sys

DATA = sys.argv[1] if len(sys.argv) > 1 else '/tmp/bc/meshdump'


def load_verts(path):
    verts = []
    for line in open(path):
        m = re.match(r"\(([-\d.eE+]+), ([-\d.eE+]+), ([-\d.eE+]+)\)", line)
        if m:
            verts.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
    return np.array(verts)


def load_indices(path):
    cells = []  # (cellIdx, npts, [v0,v1,v2])
    for line in open(path):
        p = line.split()
        if len(p) >= 5:
            cells.append((int(p[0]), int(p[1]), [int(p[2]), int(p[3]), int(p[4])]))
    return cells


def load_matrices(path):
    lines = [l.strip() for l in open(path) if l.strip()]
    mat = {}
    cur = None
    for l in lines:
        if l in ("V", "P"):
            cur = l
            mat[cur] = []
        else:
            mat[cur].append([float(x) for x in l.split()])
    # dumped rows are VTK GetElement(r,c) = row-major; use directly as row-major
    return np.array(mat["V"]), np.array(mat["P"])


BSIZE = np.array([201.6, 201.6, 138.0])
H = 512.0


def fb_of_ndc(ndc):
    x = (ndc[0] * 0.5 + 0.5) * H
    y = (ndc[1] * 0.5 + 0.5) * H
    return np.array([x, y])


def project(verts, V, P):
    # modelPos = normalized * bsize (bounds start at 0)
    world = verts * BSIZE
    clip = []
    for w in world:
        w4 = np.array([w[0], w[1], w[2], 1.0])
        c = P @ (V @ w4)
        clip.append(c)
    clip = np.array(clip)
    ndc = clip[:, :3] / clip[:, 3:4]
    fb = np.array([fb_of_ndc(n) for n in ndc])
    return clip, ndc, fb


def signed_area_fb(tri):
    # tri: (3,2) FB coords (y down); positive = CCW in y-down window = GL back
    return 0.5 * (tri[0, 0] * (tri[1, 1] - tri[2, 1]) + tri[1, 0] * (tri[2, 1] - tri[0, 1]) +
                  tri[2, 0] * (tri[0, 1] - tri[1, 1]))


def covers(tri, pt, eps=1e-9):
    # barycentric point-in-triangle test
    v0 = tri[1] - tri[0]
    v1 = tri[2] - tri[0]
    v2 = pt - tri[0]
    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d20 = v2 @ v0
    d21 = v2 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return False
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u >= -eps and v >= -eps and w >= -eps


def main():
    verts = load_verts(f"{DATA}/verts.txt")
    cells = load_indices(f"{DATA}/indices.txt")
    V, P = load_matrices(f"{DATA}/matrices.txt")
    clip, ndc, fb = project(verts, V, P)
    w = clip[:, 3]
    print(f"verts={len(verts)} cells={len(cells)} w>0: {(w > 0).sum()}/{len(w)} "
          f"ndc.z range=({ndc[:, 2].min():.3f},{ndc[:, 2].max():.3f})")

    if len(sys.argv) > 2:
        pixels = [(int(sys.argv[i]), int(sys.argv[i + 1])) for i in range(2, len(sys.argv), 2)]
    else:
        pixels = [(490, 484), (495, 497), (256, 256), (466, 451), (150, 250), (482, 398), (480, 400)]
    for (px, py) in pixels:
        pt = np.array([px + 0.5, py + 0.5])
        hits = []
        for (ci, npts, ids) in cells:
            tri = fb[ids]
            if not covers(tri, pt):
                continue
            a = signed_area_fb(tri)
            d = (verts[ids[0]] + verts[ids[1]] + verts[ids[2]]) / 3.0
            hits.append((ci, a, d))
        print(f"\npx=({px},{py}) covering triangles: {len(hits)}")
        for (ci, a, d) in sorted(hits, key=lambda h: -abs(h[1]))[:20]:
            keep_clock = "keep(CW)" if a < 0 else "cull"
            keep_gl = "keep" if a < 0 else "cull"
            print(f"  cell {ci:3d} A_FB(y-down)={a:+.4f} [Metal Clockwise+Back: {keep_clock}] "
                  f"[GL CCW(y-up, -A): {keep_gl}] center_vol=({d[0]:.4f},{d[1]:.4f},{d[2]:.4f})")


if __name__ == "__main__":
    main()
