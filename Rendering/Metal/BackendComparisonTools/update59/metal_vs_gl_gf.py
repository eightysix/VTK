#!/usr/bin/env python3
"""Metal vs GL pre-store gf/alpha comparison at the 188 stored-diff px
(VolumeRayCastBackendComparisonFindingsUpdate59.md, sections 2-3).

Update 58 estimated gf agreement at mean 0.00004 u8 against a FRAME-1 GL dump.
With the frame-6-aligned dump the true delta is ~100x larger:

  gf |delta| at the 188 px: mean 0.31 u8, median 0.07 u8, max 8.25 u8
  111 channels > 0.5 u8, 50 > 1 u8, 9 > 2 u8
  alpha |delta|: mean 0.000141, max 0.001548

The signature (large color delta, negligible alpha delta) is a nearest-texel
selection flip at a grid-aligned ray: adjacent texels share opacity but differ
in color, and a sub-ulp lattice hair moves one backend's sample across the
texel boundary. It is not a composite-arithmetic difference (chains verified
bit-exact, updates 51/53) and not a sample-count break (that would move alpha
proportionally).

Inputs (see README.md in this directory for regeneration):
    BC_DATA/u60_gl_float.raw   frame-6-aligned GL gf dump (rows bottom-origin)
    BC_DATA/u59_metal_float.npy Metal frame-6 gf/alpha/lastIter (parse_metal_final.py)
    BC_DATA/u60_gl_clean.png   clean GL stored image
    BC_DATA/u59_metal.png      Metal stored image

Usage: BC_DATA=/path/to/data python3 metal_vs_gl_gf.py
"""
import os
import numpy as np
from PIL import Image

BC = os.environ.get("BC_DATA", "/tmp/bc")
N = 512


def main():
    raw = np.fromfile(os.path.join(BC, 'u60_gl_float.raw'), dtype=np.float32).reshape(N, N, 4)
    gl_gf = raw[::-1, :, :3].astype(np.float64)
    gl_a = raw[::-1, :, 3].astype(np.float64)

    m = np.load(os.path.join(BC, 'u59_metal_float.npy'))
    mg = m['g']
    ma = m['a']

    gl_img = np.array(Image.open(os.path.join(BC, 'u60_gl_clean.png'))).astype(np.float64)
    mt_img = np.array(Image.open(os.path.join(BC, 'u59_metal.png'))).astype(np.float64)

    dpx = np.all(gl_img == mt_img, axis=2) == False
    gidx = np.argwhere(dpx)
    print('GL-vs-Metal stored diff px:', dpx.sum(), '(expect 188)')

    d = gl_gf - mg
    dd = d[gidx[:, 0], gidx[:, 1]]
    print('gf |delta| at diff px [u8]: mean=%.6f std=%.6f median=%.6f p99=%.6f max=%.6f' % (
        np.abs(dd).mean() * 255, np.abs(dd).std() * 255, np.median(np.abs(dd)) * 255,
        np.percentile(np.abs(dd), 99) * 255, np.abs(dd).max() * 255))
    for thr in (0.5, 1.0, 2.0):
        print('  channels with |gf delta|>%.1f u8: %d' % (thr, (np.abs(dd) > thr / 255).sum()))

    ad = np.abs(gl_a - ma)[gidx[:, 0], gidx[:, 1]]
    print('alpha |delta| at diff px: mean=%.6f max=%.6f' % (ad.mean(), ad.max()))

    imgd = np.abs(gl_img - mt_img).max(axis=2)
    big = imgd > 1.0
    print('big stored-diff px (|d|>1):', big.sum(), '(expect 14)')
    if big.sum() > 0:
        big_a = ad[big[gidx[:, 0], gidx[:, 1]]]
        print('  gf |delta| at big px [u8]: mean=%.4f max=%.4f' % (
            np.abs(d)[big].mean() * 255, np.abs(d)[big].max() * 255))
        print('  alpha |delta| at big px: mean=%.6f max=%.6f' % (
            big_a.mean(), big_a.max()))

    print('14 big px detail (x,y) ch |stored| gf_delta_u8 alpha_delta:')
    for y, x in np.argwhere(big):
        row = []
        for c in range(3):
            sd = abs(gl_img[y, x, c] - mt_img[y, x, c])
            if sd > 1:
                row.append('c%d |img|=%d dgf=%+6.2f' % (c, sd, d[y, x, c] * 255))
        print('  (%3d,%3d) %-45s da=%.5f' % (x, y, ' '.join(row), abs(gl_a[y, x] - ma[y, x])))


if __name__ == '__main__':
    main()
