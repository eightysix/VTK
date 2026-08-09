#!/usr/bin/env python3
"""Knife-edge amplification at a near-grid-aligned ray: at (422,92) the y-step
is ~1e-5 texel/sample, so y stays within ~0.4 texel of a texel boundary for the
whole ray, and a ~1e-7 step difference changes which sample crosses the
boundary -> the ray tail samples a different texel column and the final color
swings by up to 14-17 LSB
(VolumeRayCastBackendComparisonFindingsUpdate48.md, section 4).

This is also the mechanism behind the 15 gated pixels in the +/-1 field, but
its requirement (ray ~parallel to a grid axis and grazing a boundary) is why it
cannot scale to the whole 63,690-px field.

Inputs (see README.md in this directory for regeneration):
    BC_DATA/u47_metal.log   Metal STEP row for (422,92)
    BC_DATA/gl372.log       GL_RAY (422,419) lines, one per frame (y-step varies
                            by ~1e-7 between frame 1 and frame 6)
    BC_DATA/vol512.npy

Expected output:
    metal final (0.9338203, 0.7517762, 0.6219854) -> [238 192 159], 170 samples
    frame1    step -> [238 190 157], 171 samples
    frame6    step -> [238 176 140]
    first divergence at sample index 0 (positions differ by ~2.4e-7, same texel)

Usage: BC_DATA=/path/to/data python3 knife_edge_422_92.py
"""
import os
import re
import numpy as np

BC = os.environ.get("BC_DATA", "/tmp/bc")
N = 512
W = 1024
FACTOR = 0.2700587213
PF_X = [0.0, 500.0, 1000.0, 1150.0]
PF_Y = [0.0, 0.02, 0.02, 0.85]
CTF_X = [0.0, 500.0, 1000.0, 1150.0]
CTF_R = [0.0, 1.0, 1.0, 1.0]
CTF_G = [0.0, 0.5, 0.5, 1.0]
CTF_B = [0.0, 0.3, 0.3, 0.9]


def f32(x):
    return np.float32(x)


def pf_eval(s):
    if s <= PF_X[0]:
        return PF_Y[0]
    for i in range(len(PF_X) - 1):
        x1, x2 = PF_X[i], PF_X[i + 1]
        if s < x2:
            t = (s - x1) / (x2 - x1)
            return (1 - t) * PF_Y[i] + t * PF_Y[i + 1]
    return PF_Y[-1]


def build_op_table():
    tab = np.zeros(W, np.float64)
    for i in range(W):
        tab[i] = pf_eval((i / (W - 1)) * 4370.0)
    for i in range(W):
        a = tab[i]
        if a > 0.0001:
            tab[i] = 1.0 - np.power(1.0 - a, FACTOR)
    return tab.astype(np.float32)


def ctf_eval(s):
    if s <= CTF_X[0]:
        return (CTF_R[0], CTF_G[0], CTF_B[0])
    for i in range(len(CTF_X) - 1):
        x1, x2 = CTF_X[i], CTF_X[i + 1]
        if s < x2:
            t = (s - x1) / (x2 - x1)
            return ((1 - t) * CTF_R[i] + t * CTF_R[i + 1],
                    (1 - t) * CTF_G[i] + t * CTF_G[i + 1],
                    (1 - t) * CTF_B[i] + t * CTF_B[i + 1])
    return (CTF_R[-1], CTF_G[-1], CTF_B[-1])


def build_color_table():
    tab = np.zeros((W, 3), np.float64)
    for i in range(W):
        tab[i] = ctf_eval((i / (W - 1)) * 4370.0)
    return tab.astype(np.float32)


OP_TABLE = build_op_table()
COL_TABLE = build_color_table()
SCALE = np.float32(65536.0 / np.float32(4370.0))


def tf_lookup(val):
    stored = f32(np.float64(val) / np.float64(65536.0))
    norm = f32(stored * SCALE)
    idx = min(W - 1, int(np.floor(min(max(float(norm), 0.0), 1.0) * W)))
    return norm, OP_TABLE[idx], COL_TABLE[idx]


def fma(a, b, c):
    return np.float32(np.float64(a) * np.float64(b) + np.float64(c))


def simulate(anchor, step, max_iter=600, collect=False):
    p = np.array(anchor, np.float32)
    accC = np.zeros(3, np.float32)
    accA = np.float32(0.0)
    hist = []
    for i in range(max_iter):
        tc = np.clip(p, 0.0, 1.0)
        tex = np.clip(np.floor(tc * N).astype(int), 0, N - 1)
        val = vol[tex[0], tex[1], tex[2]]
        _, op, rgb = tf_lookup(val)
        if op > 0.0:
            w = np.float32(1.0 - accA)
            accC = np.array([fma(w, f32(op * rgb[0]), accC[0]),
                             fma(w, f32(op * rgb[1]), accC[1]),
                             fma(w, f32(op * rgb[2]), accC[2])], np.float32)
            accA = fma(w, op, accA)
            if collect:
                hist.append((i, p.tolist(), [int(x) for x in tex],
                             float(op), float(accA)))
            if accA >= np.float32(1.0 - 1.0 / 255.0):
                accA = np.float32(1.0)
                break
        p = f32(p + step)
    return accC, accA, hist


def parse_step_line(l):
    m = re.search(r'DEBUG STEP px=\((\d+), (\d+)\)', l)
    if not m:
        return None

    def tri(name):
        t = re.search(name + r'=\(([^)]+)\)', l)
        return np.array([float(x) for x in t.group(1).split(',')], np.float32)
    return (int(m.group(1)), int(m.group(2))), {
        'localPos': tri('localPos'), 'evalStep': tri('evalStep')}


def main():
    global vol
    vol = np.load(os.path.join(BC, 'vol512.npy'))

    steps = {}
    for l in open(os.path.join(BC, 'u47_metal.log')):
        r = parse_step_line(l)
        if r:
            steps[r[0]] = r[1]
    s = steps[(422, 92)]

    gl_entries = []
    for l in open(os.path.join(BC, 'gl372.log')):
        m = re.search(r'GL_RAY px=\(422, 419\).*?origin=\(([^)]+)\) step=\(([^)]+)\)', l)
        if m:
            o = np.array([float(x) for x in m.group(1).split(',')], np.float32)
            st = np.array([float(x) for x in m.group(2).split(',')], np.float32)
            gl_entries.append((o, st))
    if not gl_entries:
        print('no GL_RAY (422,419) found in gl372.log')
        return

    print(f"metal localPos={s['localPos']} evalStep={s['evalStep']}")
    mc, mA, mh = simulate(f32(s['localPos'] + s['evalStep']), s['evalStep'], collect=True)
    print(f"metal final {mc} accA={mA} samples={len(mh)} -> "
          f"{np.round(np.clip(mc, 0, 1) * 255).astype(int)}")

    for k in (0, len(gl_entries) - 1):
        o, st = gl_entries[k]
        gc, gA, gh = simulate(o, st, collect=True)
        print(f"gl frame{k + 1} step={st} final {gc} samples={len(gh)} -> "
              f"{np.round(np.clip(gc, 0, 1) * 255).astype(int)}")

    o, st = gl_entries[0]
    gc, gA, gh = simulate(o, st, collect=True)
    for k in range(min(len(mh), len(gh))):
        mi, gi = mh[k], gh[k]
        if mi[2] != gi[2] or abs(mi[3] - gi[3]) > 0:
            print(f"first divergence at sample {k}: metal texel={mi[2]} "
                  f"op={mi[3]:.6f} pos={mi[1]} vs gl texel={gi[2]} op={gi[3]:.6f} "
                  f"pos={gi[1]}")
            break
    print(f"lattice diffs: anchor={f32(s['localPos'] + s['evalStep']) - o} "
          f"step={s['evalStep'] - st}")


if __name__ == '__main__':
    main()
