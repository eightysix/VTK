#!/usr/bin/env python3
"""Simulate Metal composite for px (422,92): default vs 4x (FineStep).

Questions answered:
  1. Does compositing the logged per-sample op/rgb reproduce the rendered pixel?
  2. Where does the front-to-back opacity saturate?
  3. Does a 4x-refined march (rebuilt TF with pre-integration factor /4)
     produce the same 8-bit output?  (explains the bit-identical FineStep)

Usage: python3 finestep_sim.py [vol512.npy] [trace_422_92.txt]
"""
import re
import sys
import numpy as np

vol = np.load(sys.argv[1] if len(sys.argv) > 1 else 'vol512.npy').astype(np.float64) / 65535.0
trace = sys.argv[2] if len(sys.argv) > 2 else 'trace_422_92.txt'
N = 512
centers = (np.arange(N) + 0.5) / N

SCALAR_MIN, SCALAR_MAX = 0.0, 4370.0
FACTOR_DEFAULT = 0.270059  # sampleDist/unitDist
FACTOR_FINE = 0.067515      # 0.25 * minWorldSpacing / unitDist

def trilinear(v, p):
    fp = p * N - 0.5
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

# --- TF evaluation (matches vtkOpenGLVolumeOpacityTable::InternalUpdate + ctf) ---
PTS_O = [(0, 0.00), (500, 0.02), (1000, 0.02), (1150, 0.85)]
PTS_C = [(0, (0.0, 0.0, 0.0)), (500, (1.0, 0.5, 0.3)), (1000, (1.0, 0.5, 0.3)), (1150, (1.0, 1.0, 0.9))]

def pw_eval(pts, x):
    x = float(x)
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            t = (x - x0) / (x1 - x0)
            if isinstance(y0, tuple):
                return tuple(a + (b - a) * t for a, b in zip(y0, y1))
            return y0 + (y1 - y0) * t
    return pts[-1][1]

# build opacity table width 1024 over [0,4370] like the GPU texture
W = 1024
def opacity_table(factor):
    xs = np.linspace(SCALAR_MIN, SCALAR_MAX, W)
    return np.array([1.0 - (1.0 - pw_eval(PTS_O, x)) ** factor for x in xs])

def tf_lookup(op_tab, scalar):
    t = np.clip(scalar, SCALAR_MIN, SCALAR_MAX)
    xf = (t - SCALAR_MIN) / (SCALAR_MAX - SCALAR_MIN) * (W - 1)
    i0 = int(np.floor(xf)); fr = xf - i0
    i0 = np.clip(i0, 0, W - 2)
    return op_tab[i0] * (1 - fr) + op_tab[i0 + 1] * fr

def color_lookup(scalar):
    return np.array(pw_eval(PTS_C, np.clip(scalar, SCALAR_MIN, SCALAR_MAX)))

def composite(samples):
    acc_c = np.zeros(3); acc_a = 0.0
    for rgb, op in samples:
        w = 1.0 - acc_a
        acc_c += w * rgb * op
        acc_a += w * op
        if acc_a >= 0.999999:
            break
    return acc_c, acc_a

def to8(c):
    return np.round(np.clip(c, 0, 1) * 255).astype(int)

# --- parse frame-0 trace samples for (422,92) ---
lines = open(trace).read().strip().split('\n')
pat = re.compile(
    r'SAMPLE px=\(422, 92\) i=(\d+) t=([-\d.e+]+) tex=\(([-\d.e+]+), ([-\d.e+]+), ([-\d.e+]+)\) '
    r'eval=\(([-\d.e+]+), ([-\d.e+]+), ([-\d.e+]+)\) raw=([-\d.e+]+) norm=([-\d.e+]+) op=([-\d.e+]+) mip=([-\d.e+]+) '
    r'rgb=\(([-\d.e+]+), ([-\d.e+]+), ([-\d.e+]+)\)')
samps = []
prev = -2
for l in lines:
    m = pat.search(l)
    if not m:
        continue
    i = int(m.group(1))
    if i <= prev:
        if samps:  # second frame start
            break
    prev = i
    ev = np.array([float(m.group(6)), float(m.group(7)), float(m.group(8))])
    raw = float(m.group(9))
    op = float(m.group(10))
    rgb = np.array([float(m.group(12)), float(m.group(13)), float(m.group(14))])
    samps.append((i, ev, raw, op, rgb))
print(f"parsed {len(samps)} samples")

# 1. composite from logged op/rgb
acc_c, acc_a = composite([(s[4], s[3]) for s in samps])
print(f"[1] composite(logged) color={acc_c} alpha={acc_a} 8bit={to8(acc_c)}")
print(f"    saturation: alpha>=0.999999 at sample # of {len(samps)}")

# find first index where acc_a >= 0.999999
a = 0.0
sat = None
for i, (_, _, _, op, rgb) in enumerate(samps):
    a += (1 - a) * op
    if a >= 0.999999:
        sat = i
        break
print(f"    alpha>=0.999999 at sample index {sat}")

# 2. simulate FineStep (4x): resample scalars at quarter-step positions
ev0 = samps[0][1]
ev1 = samps[1][1]
eval_step = ev1 - ev0
print(f"[2] eval_step={eval_step}")
op_tab_d = opacity_table(FACTOR_DEFAULT)
op_tab_f = opacity_table(FACTOR_FINE)

# default comb from trace: entry-relative t = (i+1)*step; fine comb t = (k+1)*step/4
def eval_at_t(t):
    # t entry-relative in same units as logged t; entry eval = ev0 - eval_step
    e_entry = ev0 - eval_step
    return e_entry + eval_step * t

t_last = samps[-1][1]
n_fine = int(np.ceil(t_last / (eval_step / 4)))
samps_fine = []
for k in range(n_fine):
    t = (k + 1) * (eval_step / 4)
    ev = eval_at_t(t)
    if ev.min() < 0 or ev.max() > 1:
        break
    scalar = trilinear(vol, ev) * 65535.0
    op = tf_lookup(op_tab_f, scalar)
    rgb = color_lookup(scalar)
    samps_fine.append((t, ev, op, rgb))
print(f"    fine comb: {len(samps_fine)} samples, last t={samps_fine[-1][0] if samps_fine else None}")

acc_c_f, acc_a_f = composite([(s[3], s[2]) for s in samps_fine])
print(f"[3] composite(FineStep 4x) color={acc_c_f} alpha={acc_a_f} 8bit={to8(acc_c_f)}")

# 3. also simulate default comb from vol512.npy for cross-check
samps_def = []
for (i, ev, raw, op, rgb) in samps:
    scalar = raw * 65535.0
    op2 = tf_lookup(op_tab_d, scalar)
    rgb2 = color_lookup(scalar)
    samps_def.append((ev, op2, rgb2))
acc_c_d, acc_a_d = composite([(s[2], s[1]) for s in samps_def])
print(f"[4] composite(default from npy+TF) color={acc_c_d} alpha={acc_a_d} 8bit={to8(acc_c_d)}")

print(f"[5] default 8bit == fine 8bit? {np.array_equal(to8(acc_c_d), to8(acc_c_f))}")
