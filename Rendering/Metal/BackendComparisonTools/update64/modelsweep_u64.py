import os
import sys
import numpy as np

import field_u64 as F

# Model sweep: can GL's interpolated texcoord be reproduced by analytic
# perspective-correct interpolation evaluated at pixel_center + (ox, oy) for a
# single constant offset (ox, oy)?
#
# Data: GL raw attribute dump (frame 6) texcoord + primId at every pixel,
# per-vertex clip/tex from u62_gl_vlog.log, triangle 122 = (86,40,93).
#
# Result (update-64 findings sec 5b.1): best offset (0.020, -0.020) px matches
# only 431/16384 = 2.6% of pixels at 0 ulps; pixel center ~2.1%. No constant
# offset reproduces GL across the frame -> the residual is a driver-level
# attribute-interpolator difference, not an analytic persp-correct evaluation
# at any constant sample location.

BC = os.environ.get("BC_DATA", "/tmp/bc")


def persp_interp_f64(win, texs, ws, sx, sy):
    """Perspective-correct interpolate of texs at window-space samples (sx, sy).

    barycentric t = solve(win[0]-win[2], win[1]-win[2]; sample-win[2]),
    then val = sum_i (t_i/w_i) tex_i / sum_i (t_i/w_i), computed in f64,
    rounded to f32. Returns (N,3) f32.
    """
    A = np.array([[win[0][0] - win[2][0], win[1][0] - win[2][0]],
                  [win[0][1] - win[2][1], win[1][1] - win[2][1]]])
    Ai = np.linalg.inv(A)
    d = np.stack([sx - win[2][0], sy - win[2][1]], axis=1)
    t = d @ Ai.T
    t0 = t[:, 0]; t1 = t[:, 1]; t2 = 1 - t0 - t1
    denom = t0 / ws[0] + t1 / ws[1] + t2 / ws[2]
    num = (np.outer(t0 / ws[0], texs[0]) + np.outer(t1 / ws[1], texs[1])
           + np.outer(t2 / ws[2], texs[2]))
    return (num / denom[:, None]).astype(np.float32)


def main():
    glv, idx, mtv, mt = F.load()
    frames = F.load_attr(os.path.join(BC, "gl_attr_dump.raw"))
    fr = frames[-1]
    tri = idx[122]
    vdata = [F.per_vertex(glv, v) for v in tri]
    win = np.array([(c[0] / c[3] * 256 + 256, c[1] / c[3] * 256 + 256)
                    for c, t in vdata])
    ws = np.array([vdata[i][0][3] for i in range(3)])
    texs = np.array([vdata[i][1] for i in range(3)])

    # subsampled full-frame samples of triangle 122 (every 4th px)
    sub = []
    for my in range(0, 512, 4):
        for mx in range(0, 512, 4):
            if int(fr["primid"][my, mx]) == 122:
                sub.append((mx, my, fr["tex"][my, mx].astype(np.float32)))
    mx = np.array([s[0] for s in sub]); my = np.array([s[1] for s in sub])
    tex = np.stack([s[2] for s in sub])
    print("samples:", len(mx))

    step = 0.005
    best = (0, 0, 0)
    for ox in np.arange(-0.05, 0.051, step):
        for oy in np.arange(-0.05, 0.051, step):
            got = persp_interp_f64(win, texs, ws, mx + 0.5 + ox, my + 0.5 + oy)
            ok = int(np.sum(np.all(got == tex, axis=1)))
            if ok > best[0]:
                best = (ok, ox, oy)
    print("best: %d/%d (%.1f%%) at offset (%.3f, %.3f)px"
          % (best[0], len(mx), 100 * best[0] / len(mx), best[1], best[2]))

    got = persp_interp_f64(win, texs, ws, mx + 0.5, my + 0.5)
    n = np.sum(np.all(got == tex, axis=1))
    print("center: %d/%d (%.1f%%)" % (n, len(mx), 100 * n / len(mx)))


if __name__ == "__main__":
    main()
