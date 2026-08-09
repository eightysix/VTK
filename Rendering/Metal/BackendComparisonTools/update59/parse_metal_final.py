#!/usr/bin/env python3
"""Parse the Metal full-field pre-store FINAL dump log into a numpy .npy,
keeping the LAST occurrence per pixel (last frame = the stored image frame)
(update 58 section 1, regenerating /tmp/bc/u59_metal_float.npy).

The log comes from a Metal run with VTK_METAL_FLOAT_DUMP=1 (turns the shader
FINAL log into a dump-all mode via the dumpAll gate). Each FINAL line carries
exactly the pre-store state:
    accCol = accumulatedColor (unblended gf), accOp = accumulatedOpacity
    (unclamped), lastIter = break iteration.

Row format:
    DEBUG FINAL px=(%d, %d) vp=(%f, %f) lastIter=%d accOp=%f accCol=(...) final=(...)

Output .npy is a (512,512) structured array with fields:
    a  (f64) alpha, g  (f64,(3,)) gf, f (f64,(3,)) final, it (i8/i64) lastIter

Usage: BC_DATA=/path/to/data python3 parse_metal_final.py
"""
import os
import re
import numpy as np

BC = os.environ.get("BC_DATA", "/tmp/bc")
LOG = os.path.join(BC, 'u59_metal_float.log')
OUT = os.path.join(BC, 'u59_metal_float.npy')
N = 512

PAT = re.compile(
    r'DEBUG FINAL px=\((\d+), (\d+)\) vp=\([\d.]+, [\d.]+\) lastIter=(\d+) '
    r'accOp=([\d.e+-]+) accCol=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) '
    r'final=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\)')


def main():
    a = np.zeros((N, N), np.float64)
    g = np.zeros((N, N, 3), np.float64)
    f = np.zeros((N, N, 3), np.float64)
    it = np.zeros((N, N), np.int64)
    n = 0
    with open(LOG) as log:
        for line in log:
            m = PAT.search(line)
            if not m:
                continue
            x, y = int(m.group(1)), int(m.group(2))
            a[y, x] = float(m.group(4))
            g[y, x, 0] = float(m.group(5))
            g[y, x, 1] = float(m.group(6))
            g[y, x, 2] = float(m.group(7))
            f[y, x, 0] = float(m.group(8))
            f[y, x, 1] = float(m.group(9))
            f[y, x, 2] = float(m.group(10))
            it[y, x] = int(m.group(3))
            n += 1
    cov = (g[:, :, 0] != 0) | (a != 0)
    dt = np.dtype([('a', np.float64), ('g', np.float64, (3,)), ('f', np.float64, (3,)),
                   ('it', np.int64)])
    out = np.zeros((N, N), dt)
    out['a'] = a
    out['g'] = g
    out['f'] = f
    out['it'] = it
    np.save(OUT, out)
    print('parsed FINAL rows: %d  coverage px: %d / %d  saved %s' % (n, cov.sum(), cov.size, OUT))


if __name__ == '__main__':
    main()
