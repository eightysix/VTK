import re
import os
import sys
import numpy as np

# Full-frame back-out of the effective sample displacement for the
# CameraInsideTransformationNoTransform knife test, GL vs Metal.
#
# Data:
#   GL  - /tmp/bc/gl_attr_dump.raw : 6 frames x 3 passes x (512*512*4) f32
#         pass0 = (ip_textureCoords.xyz, float(flatVid))
#         pass1 = ip_debugClip.xyzw
#         pass2 = (ip_vertexPos.xyz, float(gl_PrimitiveID))
#         row 0 = gl_FragCoord y 0 (bottom-left).
#   GL  - /tmp/bc/u62_gl_vlog.log : GL_VERT (per-vertex clip+tex, 94 vids),
#         GL_CAPINDEX (126 triangles -> vids), GL_RAY (interpolated ref).
#   Metal - /tmp/bc/u63_metal.log : DEBUG STEP at 8203 px/frame (localPos =
#         interpolated texcoord, clip, primId), vertex_volume_main (95 vids).
#
# Both captures are frame 6 (last occurrence per key); GL camera position at
# frame 6 is identical in the two GL captures (verified: cam=(102.122314,
# 102.122314, 61.5619835)).

BC = os.environ.get("BC_DATA", "/tmp/bc")
W = H = 512


def f(x):
    return float(x)


# --------------------------------------------------------------------------
# Parsers (last-occurrence-per-key, frame 6)
# --------------------------------------------------------------------------
GL_VERT_PAT = re.compile(
    r"DEBUG GL_VERT (\d+) px=\(\d+, \d+\) clip=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) "
    r"pos=\([\d.e+-]+, [\d.e+-]+, [\d.e+-]+\) tex=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\)")
GL_CAPINDEX_PAT = re.compile(r"DEBUG GL_CAPINDEX (\d+) (\d+) (\d+) (\d+)")
MT_STEP_PAT = re.compile(
    r"DEBUG STEP px=\((\d+), (\d+)\).*flatVid=(\d+) primId=(\d+) cameraVol=\([^)]*\) "
    r"localPos=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\).*"
    r"clip=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\)")
MT_VERT_PAT = re.compile(
    r"vertex_volume_main vid=(\d+).*clip=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\).*"
    r"texcoord=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\)")


def load():
    glv = {}
    for line in open(BC + "/u62_gl_vlog.log"):
        mm = GL_VERT_PAT.search(line)
        if mm:
            glv[int(mm.group(1))] = mm
    idx = {}
    for line in open(BC + "/u62_gl_vlog.log"):
        mm = GL_CAPINDEX_PAT.search(line)
        if mm:
            idx[int(mm.group(1))] = (int(mm.group(2)), int(mm.group(3)), int(mm.group(4)))
    mtv = {}
    for line in open(BC + "/u63_metal.log"):
        mm = MT_VERT_PAT.search(line)
        if mm:
            mtv[int(mm.group(1))] = mm
    mt = {}
    for line in open(BC + "/u63_metal.log"):
        mm = MT_STEP_PAT.search(line)
        if mm:
            mt[(int(mm.group(1)), int(mm.group(2)))] = mm
    return glv, idx, mtv, mt


def load_attr(raw):
    """Return per-frame dicts: tex, clip, vpos, vid, primid arrays [H][W]."""
    d = np.fromfile(raw, dtype=np.float32)
    perpass = W * H * 4
    perframe = 3 * perpass
    nf = d.size // perframe
    frames = []
    for fr in range(nf):
        p0 = d[fr * perframe: fr * perframe + perpass].reshape(H, W, 4)
        p1 = d[fr * perframe + perpass: fr * perframe + 2 * perpass].reshape(H, W, 4)
        p2 = d[fr * perframe + 2 * perpass: (fr + 1) * perframe].reshape(H, W, 4)
        frames.append(
            dict(tex=p0[..., :3], vid=p0[..., 3], clip=p1, vpos=p2[..., :3],
                 primid=p2[..., 3]))
    return frames


def per_vertex(glv, vid):
    m = glv[vid]
    clip = np.array([f(m.group(i)) for i in (2, 3, 4, 5)], np.float64)
    tex = np.array([f(m.group(i)) for i in (6, 7, 8)], np.float64)
    return clip, tex


def backout_lam(tex, vtex, w):
    """Perspective-weight back-out from one pixel's 3-channel interpolated
    texcoord, given the 3 per-vertex texcoords (f64) and clip.w per vertex.
    Mirrors backout_u63.py: lam = (t0*w0, t1*w1, t2*w2) / sum, t0 = 1.
    Returns (lam, res)."""
    A = np.array([[tex[ax] - vtex[1][ax], tex[ax] - vtex[2][ax]] for ax in range(3)])
    b = np.array([vtex[0][ax] - tex[ax] for ax in range(3)])
    sol, res, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    t1, t2 = float(sol[0]), float(sol[1])
    t0 = 1.0
    lam = np.array([t0 * w[0], t1 * w[1], t2 * w[2]])
    lam = lam / lam.sum()
    return lam, res


def sample_ndc(mx, my):
    """GL/knife convention: GL pixel (mx,my), y up from bottom."""
    return ((mx + 0.5) / 256.0 - 1.0, (my + 0.5) / 256.0 - 1.0)


def main():
    glv, idx, mtv, mt = load()
    if len(sys.argv) > 1:
        raw = sys.argv[1]
    else:
        raw = BC + "/gl_attr_dump.raw"
    frames = load_attr(raw)
    print("attr frames:", len(frames))
    fr = frames[-1]

    # ---- GL full-frame back-out over the whole 512x512 grid ----
    base = np.empty((H, W, 2))
    for my in range(H):
        for mx in range(W):
            base[my, mx] = sample_ndc(mx, my)

    # Per-pixel covering triangle from primId; skip uncovled (primId<0).
    dgl = np.full((H, W, 2), np.nan)
    rgl = np.full((H, W), np.nan)
    covered = np.zeros((H, W), bool)
    tri_cnt = {}
    for my in range(H):
        for mx in range(W):
            pid = int(fr["primid"][my, mx])
            if pid < 0 or pid not in idx:
                continue
            tri = idx[pid]
            tri_cnt[pid] = tri_cnt.get(pid, 0) + 1
            w = []
            for v in tri:
                clip, _ = per_vertex(glv, v)
                w.append(f(clip[3]))
            vt = [per_vertex(glv, v)[1] for v in tri]
            tex = np.array([f(v) for v in fr["tex"][my, mx]], np.float64)
            lam, res = backout_lam(tex, vt, w)
            # effective sample NDC = sum lam_i * ndc_i
            ndc = []
            for v in tri:
                clip, _ = per_vertex(glv, v)
                ndc.append((f(clip[0]) / f(clip[3]), f(clip[1]) / f(clip[3])))
            sx = sum(lam[i] * ndc[i][0] for i in range(3))
            sy = sum(lam[i] * ndc[i][1] for i in range(3))
            dgl[my, mx] = (sx - base[my, mx][0], sy - base[my, mx][1])
            rgl[my, mx] = res.sum()
            covered[my, mx] = True

    nc = covered.sum()
    print("GL covered px:", nc, "unique tris:", len(tri_cnt))
    print("  tri 122 px:", tri_cnt.get(122, 0))
    gx = dgl[..., 0]; gy = dgl[..., 1]
    gmask = covered
    print("GL disp dx mean=%.4e  median=%.4e  std=%.4e" %
          (np.nanmean(gx[gmask]), np.nanmedian(gx[gmask]), np.nanstd(gx[gmask])))
    print("GL disp dy mean=%.4e  median=%.4e  std=%.4e" %
          (np.nanmean(gy[gmask]), np.nanmedian(gy[gmask]), np.nanstd(gy[gmask])))
    # per-triangle mean
    print("  per-tri mean dx (top 6 by px):")
    tp = sorted(tri_cnt.items(), key=lambda kv: -kv[1])[:6]
    for pid, cnt in tp:
        sub = np.logical_and(covered, fr["primid"] == float(pid))
        if sub.sum() == 0:
            continue
        print("    tri %3d (%4d px)  dx=%.3e  dy=%.3e" %
              (pid, cnt, np.nanmean(gx[sub]), np.nanmean(gy[sub])))

    # ---- Metal back-out at the logged STEP pixels ----
    dmt = {}
    rmt = {}
    for (mx, my), m in mt.items():
        tex = np.array([f(m.group(i)) for i in (5, 6, 7)], np.float64)
        pid = int(f(m.group(4)))
        if pid not in idx:
            continue
        tri = idx[pid]
        w = []
        for v in tri:
            clip, _ = per_vertex(glv, v)
            w.append(f(clip[3]))
        vt = [per_vertex(glv, v)[1] for v in tri]
        lam, res = backout_lam(tex, vt, w)
        ndc = []
        for v in tri:
            clip, _ = per_vertex(glv, v)
            ndc.append((f(clip[0]) / f(clip[3]), f(clip[1]) / f(clip[3])))
        bx, by = sample_ndc(mx, 511 - my)
        sx = sum(lam[i] * ndc[i][0] for i in range(3))
        sy = sum(lam[i] * ndc[i][1] for i in range(3))
        dmt[(mx, my)] = (sx - bx, sy - by)
        rmt[(mx, my)] = res.sum()
    print("Metal logged px:", len(mt), " back-out ok:", len(dmt))

    # ---- GL vs Metal at overlapping pixels ----
    print()
    print("%-9s | %-20s | %-20s | %-20s" %
          ("px", "disp GL (dx,dy)", "disp MT (dx,dy)", "delta (GL-MT)"))
    dgv = []; dmv = []; ddv = []
    for (mx, my), (dxm, dym) in dmt.items():
        dglv = dgl[511 - my, mx]  # Metal (mx,my) == GL (mx, 511-my) row
        if np.isnan(dglv[0]):
            continue
        dgv.append(dglv); dmv.append((dxm, dym))
        ddv.append((dglv[0] - dxm, dglv[1] - dym))
    n = len(dgv)
    if n:
        dgv = np.array(dgv); dmv = np.array(dmv); ddv = np.array(ddv)
        print("overlap px:", n)
        print("GL  disp mean (NDC): (%.4e, %.4e)" % (dgv[:, 0].mean(), dgv[:, 1].mean()))
        print("MT  disp mean (NDC): (%.4e, %.4e)" % (dmv[:, 0].mean(), dmv[:, 1].mean()))
        print("GL-MT delta mean (NDC): (%.4e, %.4e)" % (ddv[:, 0].mean(), ddv[:, 1].mean()))
        print("GL-MT delta mean (px):  (%.4f, %.4f)" % (ddv[:, 0].mean() * 256, ddv[:, 1].mean() * 256))
        print("GL-MT |delta| max (NDC): (%.4e, %.4e)" % (np.abs(ddv[:, 0]).max(), np.abs(ddv[:, 1]).max()))
        # ---- structure search: viewport position, pixel parity, delta fit ----
        keys = list(dmt.keys())
        def _valid(k):
            g = dgl[511 - k[1], k[0]]
            return not np.isnan(g[0])
        vmask = np.array([_valid(k) for k in keys])
        rows = np.array([k[1] for k in keys])[vmask]
        cols = np.array([k[0] for k in keys])[vmask]
        ddg = ddv
        # phase correlation: delta vs (x+y) parity / viewport quadrant
        par = (rows % 2).astype(float)
        for name, sel in (("even y", par == 0), ("odd  y", par == 1)):
            if sel.sum():
                print("  %s: n=%d  d(px)=(%+.4f, %+.4f)  std(px)=(%.4f, %.4f)" % (
                    name, sel.sum(), ddg[sel, 0].mean() * 256, ddg[sel, 1].mean() * 256,
                    ddg[sel, 0].std() * 256, ddg[sel, 1].std() * 256))
        q1 = (cols < 256) & (rows < 256); q2 = (cols >= 256) & (rows < 256)
        q3 = (cols < 256) & (rows >= 256); q4 = (cols >= 256) & (rows >= 256)
        for name, sel in (("Q1 tl", q1), ("Q2 tr", q2), ("Q3 bl", q3), ("Q4 br", q4)):
            if sel.sum():
                print("  %s: n=%d  d(px)=(%+.4f, %+.4f)  std(px)=(%.4f, %.4f)" % (
                    name, sel.sum(), ddg[sel, 0].mean() * 256, ddg[sel, 1].mean() * 256,
                    ddg[sel, 0].std() * 256, ddg[sel, 1].std() * 256))
        # linear fit of delta vs (x, y) in px
        X = np.column_stack([cols, rows, np.ones(len(rows))])
        for axn in (0, 1):
            b, r, rank, _ = np.linalg.lstsq(X, ddg[:, axn] * 256, rcond=None)
            print("  d%c(x,y) fit: x-coef=%+.4e  y-coef=%+.4e  const=%+.4e  resid_std=%.4fpx" %
                  ("x" if axn == 0 else "y", b[0], b[1], b[2], np.std(ddg[:, axn] * 256 - X @ b)))
        # quadratic fit (position structure check)
        Xq = np.column_stack([cols, rows, cols * cols, cols * rows, rows * rows,
                              np.ones(len(rows))])
        for axn in (0, 1):
            b, r, rank, _ = np.linalg.lstsq(Xq, ddg[:, axn] * 256, rcond=None)
            print("  d%c quad fit resid_std=%.4fpx (delta std=%.4fpx)" %
                  ("x" if axn == 0 else "y", np.std(ddg[:, axn] * 256 - Xq @ b),
                   ddg[:, axn].std() * 256))
        # proportional-to-disp check: resid must shrink a lot if delta ~ disp
        for axn in (0, 1):
            gv = dgv[:, axn] * 256
            b, r, rank, _ = np.linalg.lstsq(gv[:, None], ddg[:, axn] * 256, rcond=None)
            print("  d%c ~ k*disp: k=%+.4f  resid_std=%.4fpx" %
                  ("x" if axn == 0 else "y", b[0], np.std(ddg[:, axn] * 256 - gv * b[0])))
    # knife px: Metal coords; pair with GL y-flip (GL (mx,511-my))
    print()
    print("knife px (Metal coords, paired with GL y-flip):")
    for (mx, my) in [(397, 110), (349, 255), (482, 33), (469, 463)]:
        g = dgl[511 - my, mx]
        m = dmt.get((mx, my))
        line = "  GL(%3d,%3d) disp=(%+.3e, %+.3e)" % (mx, my, g[0], g[1])
        if m:
            line += "  MT disp=(%+.3e, %+.3e)  d=(%+.3e, %+.3e)" % (
                m[0], m[1], g[0] - m[0], g[1] - m[1])
        print(line)


if __name__ == "__main__":
    main()
