import os
import numpy as np

import field_u64 as F

# Conditioning of the back-out least-squares problem on triangle 122 and the
# measured amplification of a 1-ulp f32 texcoord perturbation into the backed
# out effective sample displacement.
#
# Result (update-64 findings sec 5b): triangle 122's texcoords are
# near-degenerate (cond(A) ~ 4.2 but ~1700x sensitivity along the thin
# texcoord axis): +1 ulp on one texcoord channel moves the backed-out sample
# displacement by up to 0.0256 px. This is what turns the driver-level +/-1-3
# ulp interpolated-texcoord difference into the ~0.026-0.031 px per-pixel
# scatter of the full-frame GL-vs-Metal displacement delta.

BC = os.environ.get("BC_DATA", "/tmp/bc")


def main():
    glv, idx, mtv, mt = F.load()
    tri = idx[122]
    vtex = [F.per_vertex(glv, v)[1] for v in tri]
    print("tri 122 vids", tri)
    for v, t in zip(tri, vtex):
        print("  vid %d tex=%s" % (v, np.array2string(t, precision=9)))

    A = np.array([[vtex[0][ax] - vtex[1][ax], vtex[0][ax] - vtex[2][ax]]
                  for ax in range(3)])
    u, s, vt = np.linalg.svd(A)
    print("A singular values:", s, " cond:", s[0] / s[1])

    ndc = []
    w = []
    for v in tri:
        c, _ = F.per_vertex(glv, v)
        w.append(F.f(c[3]))
        ndc.append((F.f(c[0]) / F.f(c[3]), F.f(c[1]) / F.f(c[3])))
    ndc = np.array(ndc)
    print("vertex NDC:", ndc)
    print("vertex w:", w)

    def backout_disp(tex):
        b = np.array([vtex[0][ax] - tex[ax] for ax in range(3)])
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        t1, t2 = sol
        lam = np.array([1.0 * w[0], t1 * w[1], t2 * w[2]])
        lam = lam / lam.sum()
        return lam, sum(lam[i] * ndc[i][0] for i in range(3)), \
            sum(lam[i] * ndc[i][1] for i in range(3))

    tex0 = np.array([0.5, 0.5, 0.5])
    lam0, sx0, sy0 = backout_disp(tex0)
    for ax in range(3):
        tex = tex0.copy()
        tex[ax] += np.spacing(np.float32(tex[ax]))
        lam, sx, sy = backout_disp(tex)
        print("+1ulp on tex[%d]: disp delta (NDC) = (%+.2e, %+.2e)"
              " = (%+.4f px, %+.4f px)"
              % (ax, sx - sx0, sy - sy0, (sx - sx0) * 256, (sy - sy0) * 256))


if __name__ == "__main__":
    main()
