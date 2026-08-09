#!/usr/bin/env python3
"""Per-sample instrumentation at (93,201): find knife-edge TF-bin selections and
the marginal color/opacity contribution of each sample, to locate where clean
GL's ~+0.002 G/B accumulation could enter.
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

f32 = np.float32
f64 = np.float64


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

vol = np.load(os.path.join(BC, 'vol512.npy'))


def fma(a, b, c):
    return np.float32(np.float64(a) * np.float64(b) + np.float64(c))


def round_half_even(v):
    f = np.floor(v)
    d = v - f
    if d > 0.5:
        return f + 1
    if d < 0.5:
        return f
    return f if (int(f) % 2 == 0) else f + 1


def extract_step(px):
    target = "DEBUG STEP px=(%d, %d)" % px
    for l in open(os.path.join(BC, 'u47_metal.log')):
        if target in l:
            lp = re.search(r'localPos=\(([^)]+)\)', l)
            es = re.search(r'evalStep=\(([^)]+)\)', l)
            if lp and es:
                return (np.array([float(x) for x in lp.group(1).split(',')], np.float32),
                        np.array([float(x) for x in es.group(1).split(',')], np.float32))
    raise SystemExit("no STEP line for %s" % (px,))


def main():
    anchor, step = extract_step((93, 201))
    p = f32(anchor + step)
    accC = np.zeros(3, np.float32)
    accA = np.float32(0.0)
    last = None
    for i in range(600):
        tc = np.clip(p, 0.0, 1.0)
        tex = np.clip(np.floor(tc * N).astype(int), 0, N - 1)
        val = vol[tex[0], tex[1], tex[2]]
        stored = f32(f64(val) / f64(65536.0))
        norm = f32(stored * SCALE)
        coord = min(max(float(norm), 0.0), 1.0) * W
        idx = min(W - 1, int(np.floor(coord)))
        op = OP_TABLE[idx]
        rgb = COL_TABLE[idx]
        if op > 0.0:
            w = f32(1.0 - accA)
            dC = np.array([fma(w, f32(op * rgb[0]), f32(0.0)),
                           fma(w, f32(op * rgb[1]), f32(0.0)),
                           fma(w, f32(op * rgb[2]), f32(0.0))], np.float32)
            dA = fma(w, op, f32(0.0))
            if i < 8 or i > 100:
                edge = (coord - np.floor(coord))
                mark = " <== NEAR-BIN-EDGE" if min(edge, 1 - edge) < 0.02 else ""
                print(f"i={i:3d} tex={tex} val={int(val):5d} norm={norm:.8f} "
                      f"coord={coord:.6f} idx={idx:4d} op={op:.6f} rgb=({rgb[0]:.4f},{rgb[1]:.4f},{rgb[2]:.4f}) "
                      f"w={w:.6f} dC=({dC[0]:.6f},{dC[1]:.6f},{dC[2]:.6f}) dA={dA:.6f}{mark}")
            accC = np.array([fma(w, f32(op * rgb[0]), accC[0]),
                             fma(w, f32(op * rgb[1]), accC[1]),
                             fma(w, f32(op * rgb[2]), accC[2])], np.float32)
            accA = fma(w, op, accA)
            if accA >= np.float32(1.0 - 1.0 / 255.0):
                accA = np.float32(1.0)
                print(f"break at i={i} accC={accC} accA={accA}")
                break
        p = f32(p + step)
    u8 = np.array([round_half_even(np.clip(c, 0, 1) * 255) for c in accC])
    print("u8 =", u8, " accC =", accC)


if __name__ == '__main__':
    main()
