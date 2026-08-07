#!/usr/bin/env python3
"""Replay Metal's logged per-sample scalars for px (422,92) from vol512.npy.

texel centers at (i+0.5)/512; Metal samples with trilinear (sVolume) at eval
(coordinate logged). raw is the volume value normalized /65535.

Usage: python3 replay_422_92.py [vol512.npy] [trace_422_92.txt]
"""
import re
import sys
import numpy as np

vol = np.load(sys.argv[1] if len(sys.argv) > 1 else 'vol512.npy').astype(np.float64) / 65535.0
trace = sys.argv[2] if len(sys.argv) > 2 else 'trace_422_92.txt'
N = 512
centers = (np.arange(N) + 0.5) / N  # texel center positions per axis

def trilinear(v, p):
    # p in [0,1]; value = interpolation over texel centers
    fp = p * N - 0.5  # texel index coords (center i at (i+0.5)/N -> fp i)
    i0 = np.floor(fp).astype(int)
    f = fp - i0
    i0 = np.clip(i0, 0, N - 2)
    i1 = i0 + 1
    fx, fy, fz = f[0], f[1], f[2]
    c000 = v[i0[0], i0[1], i0[2]]; c100 = v[i1[0], i0[1], i0[2]]
    c010 = v[i0[0], i1[1], i0[2]]; c110 = v[i1[0], i1[1], i0[2]]
    c001 = v[i0[0], i0[1], i1[2]]; c101 = v[i1[0], i0[1], i1[2]]
    c011 = v[i0[0], i1[1], i1[2]]; c111 = v[i1[0], i1[1], i1[2]]
    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz

lines = open(trace).read().strip().split('\n')
pat = re.compile(
    r'SAMPLE px=\(422, 92\) i=(\d+) t=([-\d.e+]+) tex=\(([-\d.e+]+), ([-\d.e+]+), ([-\d.e+]+)\) '
    r'eval=\(([-\d.e+]+), ([-\d.e+]+), ([-\d.e+]+)\) raw=([-\d.e+]+) norm=([-\d.e+]+) op=([-\d.e+]+) mip=([-\d.e+]+) '
    r'rgb=\(([-\d.e+]+), ([-\d.e+]+), ([-\d.e+]+)\)')

# dedupe by frame: samples repeat across frames. Take the first contiguous run
# with monotonic i.
seen = set()
frames = 0
prev_i = -2
best = []
for l in lines:
    m = pat.search(l)
    if not m:
        continue
    i = int(m.group(1))
    if i <= prev_i:
        frames += 1
        if frames > 1:
            break
        best = []
    prev_i = i
    best.append(m)
    if len(best) > 500:
        break

mx = 0.0
mxinfo = None
diffs = 0
print(f'parsed {len(best)} samples (frame 0)')
for m in best:
    i = int(m.group(1))
    t = float(m.group(2))
    tex = np.array([float(m.group(3)), float(m.group(4)), float(m.group(5))])
    ev = np.array([float(m.group(6)), float(m.group(7)), float(m.group(8))])
    raw = float(m.group(9))
    pred = trilinear(vol, ev)
    d = abs(pred - raw)
    if d > mx:
        mx = d
        mxinfo = (i, t, ev.copy(), raw, pred)
    if d > 2e-4:
        diffs += 1
print(f'max |pred-raw| = {mx:.2e} at sample {mxinfo}')
print(f'samples with |pred-raw| > 2e-4: {diffs} of {len(best)}')
# scatter of raw vs pred at the skull region
for m in best:
    i = int(m.group(1))
    if 125 <= i <= 170:
        ev = np.array([float(m.group(6)), float(m.group(7)), float(m.group(8))])
        raw = float(m.group(9))
        pred = trilinear(vol, ev)
        print(f'i={i} raw={raw:.6f} pred={pred:.6f} d={pred-raw:+.6f}')
