#!/usr/bin/env python3
"""Replay Metal's per-sample gradient magnitude for the NoShade variant.

Reads /tmp/bc/vol512.npy and a Metal NoShade run's shader os_log (GRADOP
lines, which fire in the gradient-opacity block since shading is off) and
replays the shader's gradient computation against numpy ground truth:

  rawGrad_axis   = (sPX - sNX)/65535              # 16-bit unorm texture
  mag            = |rawGrad / spacing|             # model-space magnitude
  gradW          = mag / (0.5*range/(65535*avgSpacing))

Prints a gradW replay ratio over all logged samples and the gf input/value at
the canonical (256,256) samples.

Usage: python3 verify_gradient_noshade.py [vol512.npy] [metal_noshade.log]
"""
import re
import sys
import numpy as np

vol = np.load(sys.argv[1] if len(sys.argv) > 1 else 'vol512.npy')
log = open(sys.argv[2] if len(sys.argv) > 2 else 'metal_noshade.log').read().splitlines()

spacing = np.array([0.39452054794520547, 0.39452054794520547, 0.2700587084148728])
RANGE = float(vol.max())
AVG = spacing.mean()
NORM = 65535.0
GRAD_MAX = 0.25 * RANGE
GRAD_NORM_FACTOR = 0.5 * RANGE / (NORM * AVG)


def gf(x):
    if x <= 0:
        return 0.0
    if x < 90:
        return 0.5 * x / 90
    if x < 100:
        return 0.5 + 0.2 * (x - 90) / 10
    return 0.7


def sample(p):
    i0 = np.clip(np.floor(p).astype(int), 0, 510)
    t = p - np.floor(p)
    c = [vol[i0[0], i0[1], i0[2]], vol[i0[0] + 1, i0[1], i0[2]],
         vol[i0[0], i0[1] + 1, i0[2]], vol[i0[0] + 1, i0[1] + 1, i0[2]],
         vol[i0[0], i0[1], i0[2] + 1], vol[i0[0] + 1, i0[1], i0[2] + 1],
         vol[i0[0], i0[1] + 1, i0[2] + 1], vol[i0[0] + 1, i0[1] + 1, i0[2] + 1]]
    w = [(1 - t[0]) * (1 - t[1]) * (1 - t[2]), t[0] * (1 - t[1]) * (1 - t[2]),
         (1 - t[0]) * t[1] * (1 - t[2]), t[0] * t[1] * (1 - t[2]),
         (1 - t[0]) * (1 - t[1]) * t[2], t[0] * (1 - t[1]) * t[2],
         (1 - t[0]) * t[1] * t[2], t[0] * t[1] * t[2]]
    return sum(w[k] * c[k] for k in range(8))


ps = re.compile(r'DEBUG SAMPLE px=\((\d+), (\d+)\) i=(\d+) .*eval=\(([\d.e-]+), ([\d.e-]+), ([\d.e-]+)\) raw=([\d.e-]+)')
pg = re.compile(r'DEBUG GRADOP px=\((\d+), (\d+)\) i=(\d+) pos=\(([\d.e-]+), ([\d.e-]+), ([\d.e-]+)\) gradW=([\d.e-]+) gradOp=([\d.e-]+)')
evald, rawd, gradw, gradop = {}, {}, {}, {}
for l in log:
    m = ps.search(l)
    if m:
        key = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        evald[key] = np.array([float(m.group(4)), float(m.group(5)), float(m.group(6))])
        rawd[key] = float(m.group(7))
    m = pg.search(l)
    if m:
        key = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        gradw[key] = float(m.group(7))
        gradop[key] = float(m.group(8))

STEP = 512 / 511
ratios = []
for key, e in evald.items():
    if key not in gradw:
        continue
    f = e * 512 - 0.5
    d = np.array([sample(f + np.eye(3)[k] * STEP) - sample(f - np.eye(3)[k] * STEP) for k in range(3)])
    mag = np.sqrt(((d / NORM / spacing) ** 2).sum())
    gradw_pred = mag / GRAD_NORM_FACTOR
    ratios.append(gradw_pred / gradw[key])
r = np.array(ratios)
print(f'gradW replay: {len(r)} samples (gradient-opacity block)')
print(f'  ratio mean {r.mean():.3f}  median {np.median(r):.3f}  '
      f'p10 {np.percentile(r,10):.3f}  p90 {np.percentile(r,90):.3f}')
print(f'  within 5%: {100*((r>0.95)&(r<1.05)).mean():.0f}%  within 35%: {100*((r>0.65)&(r<1.35)).mean():.0f}%')

for px in [(256, 256), (80, 400), (150, 250), (422, 92), (372, 131)]:
    keys = [k for k in evald if k[0] == px[0] and k[1] == px[1]]
    if not keys:
        continue
    i = keys[0][2]
    e = evald[(px[0], px[1], i)]
    f = e * 512 - 0.5
    d = np.array([sample(f + np.eye(3)[k] * STEP) - sample(f - np.eye(3)[k] * STEP) for k in range(3)])
    mag = np.sqrt(((d / NORM / spacing) ** 2).sum())
    gradw_pred = mag / GRAD_NORM_FACTOR
    gw = gradw.get((px[0], px[1], i), float('nan'))
    gf_input = gradw_pred * GRAD_MAX
    print(f'px {px} first sample i={i}: gradW_pred={gradw_pred:.4f} vs log {gw:.4f} '
          f'(ratio {gradw_pred/gw:.2f}x)  gf_input={gf_input:.2f} -> gf={gf(gf_input):.3f} '
          f'(log gradOp={gradop.get((px[0],px[1],i), float("nan")):.3f})')
