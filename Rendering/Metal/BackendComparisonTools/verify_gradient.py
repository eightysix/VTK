#!/usr/bin/env python3
"""Verify the Metal per-sample gradient-opacity numbers against ground truth.

Reads /tmp/bc/vol512.npy (see make_vol512.py) and /tmp/bc/metal3.log (see
make_metal3_log.sh) and replays the shader's gradient computation:

  rawGrad_axis   = (sPX - sNX)/65535              # 16-bit unorm texture
  mag            = |rawGrad / spacing|             # model-space magnitude
  gradW          = mag / (0.5*range/(65535*avgSpacing))
  gf_input       = gradW * 0.25 * range            # data units
  gradOp         = gf(gf_input)                    # the user's gf table

Reproduces the report's key numbers: gradW_pred = 0.0354 vs log 0.0296 at
sample (256,256,168) (ratio 1.20x; all 33 logged samples within 35%), and
gf_input ~38.6 data units -> gf ~0.214 on both backends.

Usage: python3 verify_gradient.py [vol512.npy] [metal3.log]
"""
import re
import sys
import numpy as np

vol = np.load(sys.argv[1] if len(sys.argv) > 1 else 'vol512.npy')
log = open(sys.argv[2] if len(sys.argv) > 2 else 'metal3.log').read().splitlines()

spacing = np.array([0.39452054794520547, 0.39452054794520547, 0.2700587084148728])
RANGE = float(vol.max())
AVG = spacing.mean()
NORM = 65535.0
GRAD_MAX = 0.25 * RANGE  # Metal gf LUT spans [0, 0.25*range]
GRAD_NORM_FACTOR = 0.5 * RANGE / (NORM * AVG)

def gf(x):  # report gf: 0@0, 0.5@90, 0.7@100
    if x <= 0: return 0.0
    if x < 90: return 0.5 * x / 90
    if x < 100: return 0.5 + 0.2 * (x - 90) / 10
    return 0.7

def sample(p):
    i0 = np.floor(p).astype(int); t = p - i0
    c = [vol[i0[0],i0[1],i0[2]], vol[i0[0]+1,i0[1],i0[2]],
         vol[i0[0],i0[1]+1,i0[2]], vol[i0[0]+1,i0[1]+1,i0[2]],
         vol[i0[0],i0[1],i0[2]+1], vol[i0[0]+1,i0[1],i0[2]+1],
         vol[i0[0],i0[1]+1,i0[2]+1], vol[i0[0]+1,i0[1]+1,i0[2]+1]]
    w = [(1-t[0])*(1-t[1])*(1-t[2]), t[0]*(1-t[1])*(1-t[2]),
         (1-t[0])*t[1]*(1-t[2]), t[0]*t[1]*(1-t[2]),
         (1-t[0])*(1-t[1])*t[2], t[0]*(1-t[1])*t[2],
         (1-t[0])*t[1]*t[2], t[0]*t[1]*t[2]]
    return sum(w[k]*c[k] for k in range(8))

ps = re.compile(r'DEBUG SAMPLE px=\(\d+, \d+\) i=(\d+) .*eval=\(([\d.e-]+), ([\d.e-]+), ([\d.e-]+)\) raw=([\d.e-]+)')
pl = re.compile(r'DEBUG LIGHT px=\(\d+, \d+\) i=(\d+) .*gradW=([\d.e-]+) gradOp=([\d.e-]+)')
evald, rawd, gradw, gradop = {}, {}, {}, {}
for l in log:
    m = ps.search(l)
    if m:
        evald[int(m.group(1))] = np.array([float(m.group(2)), float(m.group(3)), float(m.group(4))])
        rawd[int(m.group(1))] = float(m.group(5))
    m = pl.search(l)
    if m:
        gradw[int(m.group(1))] = float(m.group(2))
        gradop[int(m.group(1))] = float(m.group(3))

STEP = 512 / 511  # gradStep in texel units
ratios = []
for i, e in evald.items():
    if i not in gradw:
        continue
    f = e * 512 - 0.5
    d = np.array([sample(f + np.eye(3)[k]*STEP) - sample(f - np.eye(3)[k]*STEP) for k in range(3)])
    mag = np.sqrt(((d / NORM / spacing) ** 2).sum())
    gradw_pred = mag / GRAD_NORM_FACTOR
    ratios.append(gradw_pred / gradw[i])
r = np.array(ratios)
print(f'gradW replay: {len(r)} samples, ratio mean {r.mean():.3f}, '
      f'median {np.median(r):.3f}, 100% within 35%: {100*((r>0.65)&(r<1.35)).mean():.0f}%')

i = 168
e = evald[i]
f = e * 512 - 0.5
d = np.array([sample(f + np.eye(3)[k]*STEP) - sample(f - np.eye(3)[k]*STEP) for k in range(3)])
mag = np.sqrt(((d / NORM / spacing) ** 2).sum())
gradw_pred = mag / GRAD_NORM_FACTOR
gf_input = gradw_pred * GRAD_MAX
print(f'sample (256,256,{i}): central diff d={np.round(d,1)} data units')
print(f'  gradW_pred={gradw_pred:.4f} vs log {gradw[i]:.4f} (ratio {gradw_pred/gradw[i]:.2f}x)')
print(f'  gf_input={gf_input:.2f} data units -> gf={gf(gf_input):.4f} (log gradOp={gradop[i]:.4f})')

relerr = []
for i, e in evald.items():
    raw = rawd[i] * NORM
    if raw < 500:
        continue
    relerr.append((sample(e * 512 - 0.5) - raw) / raw)
a = np.array(relerr)
print(f'raw-scalar probe (texel-center convention): n={len(a)}, median rel {np.median(a)*100:+.2f}%, '
      f'|err| p90 {np.percentile(np.abs(a),90)*100:.2f}%')
