#!/usr/bin/env python3
"""Test the uniform-excess hypothesis for clean GL: is there a single delta d
(u8 units) such that round_half_even((model_accC + d/255)*255) reproduces clean
GL's stored bytes on ALL 68 gated pixels (while round-half-even(model_accC)
reproduces Metal)?

If yes: clean GL's final float is Metal's final float + a uniform d/255.
Prints the feasible d interval and, if empty, the worst offenders.
"""
import os
import re
import numpy as np
from PIL import Image

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
    return OP_TABLE[idx], COL_TABLE[idx]


def fma(a, b, c):
    return np.float32(np.float64(a) * np.float64(b) + np.float64(c))


def simulate(anchor, step, max_iter=600):
    p = np.array(anchor, np.float32)
    accC = np.zeros(3, np.float32)
    accA = np.float32(0.0)
    for _ in range(max_iter):
        tc = np.clip(p, 0.0, 1.0)
        tex = np.clip(np.floor(tc * N).astype(int), 0, N - 1)
        val = vol[tex[0], tex[1], tex[2]]
        op, rgb = tf_lookup(val)
        if op > 0.0:
            w = np.float32(1.0 - accA)
            accC = np.array([fma(w, f32(op * rgb[0]), accC[0]),
                             fma(w, f32(op * rgb[1]), accC[1]),
                             fma(w, f32(op * rgb[2]), accC[2])], np.float32)
            accA = fma(w, op, accA)
            if accA >= np.float32(1.0 - 1.0 / 255.0):
                accA = np.float32(1.0)
                break
        p = f32(p + step)
    return accC


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
    mt_img = np.array(Image.open(os.path.join(BC, 'u47_metal.png'))).astype(int)
    gl_img = np.array(Image.open(os.path.join(BC, 'u47_gl.png'))).astype(int)

    steps = {}
    for l in open(os.path.join(BC, 'u47_metal.log')):
        r = parse_step_line(l)
        if r:
            steps[r[0]] = r[1]
    print(f'gated pixels: {len(steps)}')

    lo = -1.0      # d must be >= lo (u8 units)
    hi = 1.0       # d must be <  hi
    n_flip = 0
    n_nodiff = 0
    ok_metal = 0
    constraints = []
    for px, s in steps.items():
        accC = simulate(f32(s['localPos'] + s['evalStep']), s['evalStep'])
        u8m = np.clip(np.round(np.clip(accC, 0, 1) * 255), 0, 255).astype(int)
        if np.array_equal(u8m, mt_img[px[1], px[0]]):
            ok_metal += 1
        for c in range(3):
            v = float(accC[c]) * 255.0
            m = mt_img[px[1], px[0], c]
            g = gl_img[px[1], px[0], c]
            if g == m + 1:                     # GL flipped up
                need = (m + 0.5) - v           # d >= need
                lo = max(lo, need)
                n_flip += 1
                constraints.append((px, c, 'flip', need))
            else:
                assert g == m, (px, m, g)
                must = (m + 0.5) - v           # d < must
                hi = min(hi, must)
                n_nodiff += 1
                constraints.append((px, c, 'keep', must))

    print(f'metal-match by model: {ok_metal}/{len(steps)}')
    print(f'flipped channels: {n_flip}, unchanged channels: {n_nodiff}')
    print(f'feasible uniform d (u8): [{lo:.6f}, {hi:.6f})')
    if lo < hi:
        d = 0.5 * (lo + hi)
        print(f'  -> single d={d:.6f} u8 (={d/255:.6e} float) reproduces clean GL on all channels')
    else:
        print(f'  -> NO single d works; worst offenders (need d >= X while another channel needs < X):')
        flips = sorted([x for x in constraints if x[3] == 'flip'],
                       key=lambda x: x[3], reverse=True)[:5]
        keeps = sorted([x for x in constraints if x[3] == 'keep'],
                       key=lambda x: x[3])[:5]
        for px, c, t, val in flips:
            print(f'    flip  {px} ch{c}: d>={val:.6f}')
        for px, c, t, val in keeps:
            print(f'    keep  {px} ch{c}: d<{val:.6f}')


if __name__ == '__main__':
    main()
