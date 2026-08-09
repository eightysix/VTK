#!/usr/bin/env python3
"""Update 50 - doubt #2: whole-loop composite reassociation bisect at the gated
pixels. Replays each gated pixel's lattice under candidate WHOLE-LOOP composite
forms and reports which (if any) reproduces clean GL instead of Metal.

Candidates:
    M  (Metal written)     w = f32(1-accA); accC = fma(w, src, accC); accA = fma(w, op, accA)
    G  (GL written form)   w = f32(1-op);    accC = fma(w, accC, src); accA = fma(w, accA, op)
    G2 (GL, mul+add)       w = f32(1-op);    accC = f32(w*accC) + src; accA = f32(w*accA) + op
    M2 (Metal, mul+add)    w = f32(1-accA);  accC = f32(w*src) + accC; accA = f32(w*op) + accA
    M3 (fma(1-accA, op*rgb)) fused color product into the fma (single fma per channel)
    M4 (muladd with w pre-scaled into rgb)

Usage: BC_DATA=/tmp/bc python3 whole_loop_variants.py
"""
import os
import re
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "update48"))

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


def simulate(anchor, step, variant, max_iter=600):
    p = np.array(anchor, np.float32)
    accC = np.zeros(3, np.float32)
    accA = np.float32(0.0)
    for _ in range(max_iter):
        tc = np.clip(p, 0.0, 1.0)
        tex = np.clip(np.floor(tc * N).astype(int), 0, N - 1)
        val = vol[tex[0], tex[1], tex[2]]
        _, op, rgb = tf_lookup(val)
        if op > 0.0:
            src = np.array([f32(op * rgb[0]), f32(op * rgb[1]), f32(op * rgb[2])], np.float32)
            if variant == 'M':  # Metal written form
                w = f32(1.0 - accA)
                accC = np.array([fma(w, src[0], accC[0]),
                                 fma(w, src[1], accC[1]),
                                 fma(w, src[2], accC[2])], np.float32)
                accA = fma(w, op, accA)
            elif variant == 'G':  # GL written form (weight from source alpha)
                w = f32(1.0 - op)
                accC = np.array([fma(w, accC[0], src[0]),
                                 fma(w, accC[1], src[1]),
                                 fma(w, accC[2], src[2])], np.float32)
                accA = fma(w, accA, op)
            elif variant == 'G2':  # GL written, mul+add (no fma)
                w = f32(1.0 - op)
                accC = np.array([f32(w * accC[0]) + src[0],
                                 f32(w * accC[1]) + src[1],
                                 f32(w * accC[2]) + src[2]], np.float32)
                accA = f32(w * accA) + op
            elif variant == 'M2':  # Metal written, mul+add
                w = f32(1.0 - accA)
                accC = np.array([f32(w * src[0]) + accC[0],
                                 f32(w * src[1]) + accC[1],
                                 f32(w * src[2]) + accC[2]], np.float32)
                accA = f32(w * op) + accA
            elif variant == 'M3':  # fma(1-accA, op, rgb) single-product per channel
                w = f32(1.0 - accA)
                accC = np.array([fma(f32(w * op), rgb[0], accC[0]),
                                 fma(f32(w * op), rgb[1], accC[1]),
                                 fma(f32(w * op), rgb[2], accC[2])], np.float32)
                accA = fma(w, op, accA)
            elif variant == 'M4':  # muladd, w pre-scaled into rgb
                w = f32(1.0 - accA)
                accC = np.array([fma(f32(w * rgb[0]), op, accC[0]),
                                 fma(f32(w * rgb[1]), op, accC[1]),
                                 fma(f32(w * rgb[2]), op, accC[2])], np.float32)
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

    variants = ['M', 'G', 'G2', 'M2', 'M3', 'M4']
    stats = {v: {'mt': 0, 'gl': 0, 'other': 0} for v in variants}
    for px, s in steps.items():
        anchor = f32(s['localPos'] + s['evalStep'])
        mref = mt_img[px[1], px[0]]
        gref = gl_img[px[1], px[0]]
        for v in variants:
            accC = simulate(anchor, s['evalStep'], v)
            u8 = np.clip(np.round(np.clip(accC, 0, 1) * 255), 0, 255).astype(int)
            if np.array_equal(u8, mref):
                stats[v]['mt'] += 1
            elif np.array_equal(u8, gref):
                stats[v]['gl'] += 1
            else:
                stats[v]['other'] += 1
        # flag pixels where GL differs from Metal
        if not np.array_equal(mref, gref):
            print(f'  GL-diff px {px} metal={mref.tolist()} gl={gref.tolist()}:', end='')
            for v in variants:
                accC = simulate(anchor, s['evalStep'], v)
                u8 = np.clip(np.round(np.clip(accC, 0, 1) * 255), 0, 255).astype(int)
                tag = 'M' if np.array_equal(u8, mref) else ('GL' if np.array_equal(u8, gref) else '?')
                print(f' {v}={tag}', end='')
            print()
    print()
    for v in variants:
        s_ = stats[v]
        print(f'{v}: metal-match {s_["mt"]}, GL-match {s_["gl"]}, other {s_["other"]}')


if __name__ == '__main__':
    main()
