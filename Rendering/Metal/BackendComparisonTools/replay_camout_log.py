#!/usr/bin/env python3
"""Replay front-to-back compositing from a per-sample Metal march log.

Reads a camera-outside Metal per-sample log (make_camout_log.sh) and re-runs
the front-to-back composite over the logged SAMPLE lines for one pixel, so the
effect of the composite opacity gate can be measured without the GPU:

  accC += (1 - accA) * rgb * op ;  accA += (1 - accA) * op
  break when accA >= 1 - 1/255 (OpenGL's g_opacityThreshold)

The 0.001h gate the Metal shader used to apply drops samples with op <= 0.001h
(which is why border rays accumulated nothing); this script reproduces that,
and the gate=0.0 case, for any pixel in the log.

Usage: python3 replay_camout_log.py <log> [px] [gate0,gate1,...]
Example: python3 replay_camout_log.py /tmp/bc/camout_ring.log 0,256 0.001,0.0
Output: per-gate accOp and final [R,G,B] (0..255) for the chosen pixel.
"""
import re
import sys

log = sys.argv[1] if len(sys.argv) > 1 else '/tmp/bc/camout_ring.log'
px = sys.argv[2] if len(sys.argv) > 2 else '0,256'
gates = [float(g) for g in (sys.argv[3].split(',') if len(sys.argv) > 3 else '0.001,0.0'.split(','))]

pat = re.compile(
    r'DEBUG SAMPLE px=\((\d+), (\d+)\) i=(\d+) .*?op=([0-9.e-]+) mip=([0-9.e-]+) '
    r'rgb=\(([0-9.e-]+), ([0-9.e-]+), ([0-9.e-]+)\)')
x, y = (int(v) for v in px.split(','))
ops, rgbs = [], []
for line in open(log):
    m = pat.search(line)
    if m and int(m.group(1)) == x and int(m.group(2)) == y:
        ops.append(float(m.group(4)))
        rgbs.append((float(m.group(6)), float(m.group(7)), float(m.group(8))))

def composite(op, rgb, gate):
    acc = [0.0, 0.0, 0.0]
    acc_a = 0.0
    for o, c in zip(op, rgb):
        if o <= gate:
            continue
        w = 1.0 - acc_a
        acc[0] += w * c[0] * o
        acc[1] += w * c[1] * o
        acc[2] += w * c[2] * o
        acc_a += w * o
        if acc_a >= 1.0 - 1.0 / 255.0:
            acc_a = 1.0
            break
    return acc_a, acc

print(f'{len(ops)} samples at px ({x},{y})')
for g in gates:
    a, c = composite(ops, rgbs, g)
    print(f'gate={g:7g}: accOp={a:.4f} rgb=[{int(255*c[0])},{int(255*c[1])},{int(255*c[2])}]')
