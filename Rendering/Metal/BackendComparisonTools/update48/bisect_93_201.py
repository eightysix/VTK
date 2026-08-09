#!/usr/bin/env python3
"""Bisect clean GL's compile-level arithmetic divergence at pixel (93,201).

Metal's stored value is (247,170,130); clean GL is (247,171,131) (one channel
higher in G/B). The update-48 replay reproduces Metal 68/68 with:
    p0  = f32(localPos + evalStep)
    stored = f32(raw/65536); norm = f32(stored * 65536/4370)
    idx = floor(clamp(norm,0,1)*1024)
    w   = f32(1 - accA)
    accC = fma(w, f32(op*rgb), accC); accA = fma(w, op, accA)
    p   = f32(p + evalStep)
    store round-half-even *255

This script varies each arithmetic site and looks for the combination that
produces clean GL's (247,171,131). Usage:
    BC_DATA=/tmp/bc python3 bisect_93_201.py
"""
import os
import re
import itertools
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
SCALE_GL = np.float32(65536.0 / 4370.0)  # double-precision divide then float32


def fma(a, b, c):
    return np.float32(np.float64(a) * np.float64(b) + np.float64(c))


def muladd(a, b, c):
    return np.float32(np.float32(np.float32(a) * np.float32(b)) + np.float32(c))


def round_half_even(v):
    f = np.floor(v)
    d = v - f
    if d > 0.5:
        return f + 1
    if d < 0.5:
        return f
    return f if (int(f) % 2 == 0) else f + 1


def simulate(anchor, step, norm_div=65536.0, use_fma=True, w_double=False,
             break_strict=False, first_extra=False, max_iter=600):
    p = np.array(anchor, np.float32)
    accC = np.zeros(3, np.float32)
    accA = np.float32(0.0)
    for _ in range(max_iter):
        tc = np.clip(p, 0.0, 1.0)
        tex = np.clip(np.floor(tc * N).astype(int), 0, N - 1)
        val = vol[tex[0], tex[1], tex[2]]
        stored = f32(f64(val) / f64(norm_div))
        norm = f32(stored * SCALE_GL if SCALE_GL else stored * SCALE)
        idx = min(W - 1, int(np.floor(min(max(float(norm), 0.0), 1.0) * W)))
        op = OP_TABLE[idx]
        rgb = COL_TABLE[idx]
        if op > 0.0:
            w = np.float64(1.0 - float(accA)) if w_double else f32(1.0 - accA)
            if use_fma:
                accC = np.array([fma(w, f32(op * rgb[0]), accC[0]),
                                 fma(w, f32(op * rgb[1]), accC[1]),
                                 fma(w, f32(op * rgb[2]), accC[2])], np.float32)
            else:
                accC = np.array([muladd(w, f32(op * rgb[0]), accC[0]),
                                 muladd(w, f32(op * rgb[1]), accC[1]),
                                 muladd(w, f32(op * rgb[2]), accC[2])], np.float32)
            accA = fma(w, op, accA) if use_fma else muladd(w, op, accA)
            if accA >= np.float32(1.0 - 1.0 / 255.0):
                accA = np.float32(1.0)
                break
        p = f32(p + step)
    return accC


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
    global vol
    vol = np.load(os.path.join(BC, 'vol512.npy'))
    anchor, step = extract_step((93, 201))
    p0 = f32(anchor + step)
    mt = np.array(Image.open(os.path.join(BC, 'u47_metal.png')))[201, 93][:3]
    gl = np.array(Image.open(os.path.join(BC, 'u47_gl.png')))[201, 93][:3]
    print(f'(93,201) anchor={anchor} step={step} p0={p0}')
    print(f'metal stored={mt} gl stored={gl}')

    variants = {
        'metal-ref (fma, /65536)': dict(),
        'no-fma muladd': dict(use_fma=False),
        'norm /65535': dict(norm_div=65535.0),
        'weight in double': dict(w_double=True),
    }
    for name, kw in variants.items():
        accC = simulate(p0, step, **kw)
        u8 = np.array([round_half_even(np.clip(c, 0, 1) * 255) for c in accC])
        print(f'{name:32s} accC={accC} u8={u8}  ({"MATCH GL" if np.array_equal(u8, gl) else ""})')


if __name__ == '__main__':
    main()
