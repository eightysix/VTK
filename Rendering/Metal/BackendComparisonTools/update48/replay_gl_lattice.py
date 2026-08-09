#!/usr/bin/env python3
"""Replay GL's own debug-dumped lattice (gl372.log GL_RAY origin/step) on the
gated pixels that overlap the GL dump, and quantify the GL-vs-Metal lattice
parameter agreement (VolumeRayCastBackendComparisonFindingsUpdate48.md,
sections 2-3).

Findings it reproduces:
  - 14 of the 68 gated Metal pixels overlap the GL dump (y-flip: GL (x,511-y)).
  - Replaying GL's exact g_rayOrigin/g_dirStep reproduces clean GL (u47_gl.png)
    on exactly the pixels where clean-GL == clean-Metal (10/14), and fails on
    the 4 others; at (93,201) the GL-dump lattice recomposes to Metal
    (247,170,130) while clean GL is (247,171,131)  --  update 44's
    "debug-GL tracks Metal, not clean GL" at a real +/-1 pixel.
  - Lattice params agree to <1e-6: |tex-localPos| <= 5.4e-7,
    |g_rayOrigin-(localPos+evalStep)| <= 4.8e-7 (first frame) / 6.6e-7 (last),
    |g_dirStep-evalStep| <= 9.4e-8 (first) / 1.6e-7 (last).

Inputs (see README.md in this directory for regeneration):
    BC_DATA/gl372.log       debug-injected GL run, GL_RAY lines (15 px)
    BC_DATA/u47_metal.log   Metal STEP rows (localPos/evalStep)
    BC_DATA/u47_metal.png, BC_DATA/u47_gl.png
    BC_DATA/vol512.npy

Usage: BC_DATA=/path/to/data python3 replay_gl_lattice.py
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
    return norm, OP_TABLE[idx], COL_TABLE[idx]


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
        _, op, rgb = tf_lookup(val)
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


def load_gl_rays(keep):
    """keep='first' keeps the earliest GL_RAY line per pixel (frame 1),
    keep='last' the latest (final frame)."""
    gl_rays = {}
    for l in open(os.path.join(BC, 'gl372.log')):
        m = re.search(
            r'GL_RAY px=\((\d+), (\d+)\).*?origin=\(([^)]+)\) step=\(([^)]+)\).*?tex=\(([^)]+)\)',
            l)
        if not m:
            continue
        g = (int(m.group(1)), int(m.group(2)))
        origin = np.array([float(x) for x in m.group(3).split(',')], np.float32)
        step = np.array([float(x) for x in m.group(4).split(',')], np.float32)
        tex = np.array([float(x) for x in m.group(5).split(',')], np.float32)
        if keep == 'first' and g in gl_rays:
            continue
        gl_rays[g] = (origin, step, tex)
    return gl_rays


def to_u8(f):
    return np.clip(np.round(np.clip(f, 0, 1) * 255), 0, 255).astype(int)


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

    for keep in ('first', 'last'):
        gl_rays = load_gl_rays(keep)
        shared = sorted(mp for mp in steps if (mp[0], 511 - mp[1]) in gl_rays)
        ok = 0
        mx_tex = mx_anchor = mx_step = 0.0
        print(f'=== GL_RAY keep={keep} (frame dump): {len(shared)} shared pixels ===')
        for mpx in shared:
            origin, gstep, tex = gl_rays[(mpx[0], 511 - mpx[1])]
            s = steps[mpx]
            u8 = to_u8(simulate(origin, gstep))
            gl = gl_img[mpx[1], mpx[0]]
            mt = mt_img[mpx[1], mpx[0]]
            tag = 'GLOK' if np.array_equal(u8, gl) else 'GLx '
            if tag == 'GLOK':
                ok += 1
            mx_tex = max(mx_tex, float(np.max(np.abs(tex - s['localPos']))))
            mx_anchor = max(mx_anchor, float(
                np.max(np.abs(origin - f32(s['localPos'] + s['evalStep'])))))
            mx_step = max(mx_step, float(np.max(np.abs(gstep - s['evalStep']))))
            note = ''
            if np.array_equal(u8, mt) and not np.array_equal(u8, gl):
                note = f"  (replay==Metal {u8})"
            elif not np.array_equal(u8, gl) and np.array_equal(mt, gl):
                note = "  (clean-GL==clean-Metal here, replay matches neither)"
            print(f'  {str(mpx):>10} GL{str((mpx[0],511-mpx[1])):>12} '
                  f'{tag} replay={u8} gl={gl} mt={mt}{note}')
        print(f'  replay==clean-GL: {ok}/{len(shared)}')
        print(f'  max|tex-localPos|={mx_tex:.2e}  '
              f'max|origin-(localPos+evalStep)|={mx_anchor:.2e}  '
              f'max|g_dirStep-evalStep|={mx_step:.2e}')


if __name__ == '__main__':
    main()
