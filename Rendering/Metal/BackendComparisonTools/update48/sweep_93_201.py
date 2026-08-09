#!/usr/bin/env python3
"""Comprehensive variant sweep at (93,201) targeting clean GL's (247,171,131).

Varies: composite reassociation, weight precision, TF index width (1023/1024),
norm divisor, and trailing extra-sample count. Metal-ref must give (247,170,130);
we look for any variant that gives (247,171,131).
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


# composite modes; each returns updated (accC, accA)
def comp_fma(w, rgb, op, accC, accA):
    return np.array([fma(w, f32(op * rgb[0]), accC[0]),
                     fma(w, f32(op * rgb[1]), accC[1]),
                     fma(w, f32(op * rgb[2]), accC[2])], np.float32), fma(w, op, accA)


def comp_fma_wop(w, rgb, op, accC, accA):
    wop = f32(w * op)
    return np.array([fma(wop, rgb[0], accC[0]),
                     fma(wop, rgb[1], accC[1]),
                     fma(wop, rgb[2], accC[2])], np.float32), fma(w, op, accA)


def comp_fma_wsrc(w, rgb, op, accC, accA):
    ws = np.array([f32(w * rgb[0]), f32(w * rgb[1]), f32(w * rgb[2])], np.float32)
    return np.array([fma(ws[0], op, accC[0]),
                     fma(ws[1], op, accC[1]),
                     fma(ws[2], op, accC[2])], np.float32), fma(w, op, accA)


def comp_src_acc_aw(w, rgb, op, accC, accA):
    # (1-a)*src + acc  =  (src + acc) - a*src   ->  fma(-a, src, src+acc)
    tmp = np.array([f32(rgb[0] * op), f32(rgb[1] * op), f32(rgb[2] * op)], np.float32)
    return (np.array([fma(-w, tmp[0], f32(accC[0] + tmp[0])),
                      fma(-w, tmp[1], f32(accC[1] + tmp[1])),
                      fma(-w, tmp[2], f32(accC[2] + tmp[2]))], np.float32),
            fma(-w, op, f32(accA + op)))


def comp_muladd(w, rgb, op, accC, accA):
    return np.array([f32(f32(w * f32(op * rgb[0])) + accC[0]),
                     f32(f32(w * f32(op * rgb[1])) + accC[1]),
                     f32(f32(w * f32(op * rgb[2])) + accC[2])], np.float32), f32(f32(w * op) + accA)


COMPOSITES = {
    'fma': comp_fma,
    'fma(w*op,rgb)': comp_fma_wop,
    'fma(w*rgb,op)': comp_fma_wsrc,
    '(src+acc)-a*src': comp_src_acc_aw,
    'muladd': comp_muladd,
}


def simulate(anchor, step, comp, idx_width=W, norm_div=65536.0, extra=0, max_iter=700):
    p = np.array(anchor, np.float32)
    accC = np.zeros(3, np.float32)
    accA = np.float32(0.0)
    for _ in range(max_iter):
        tc = np.clip(p, 0.0, 1.0)
        tex = np.clip(np.floor(tc * N).astype(int), 0, N - 1)
        val = vol[tex[0], tex[1], tex[2]]
        stored = f32(f64(val) / f64(norm_div))
        norm = f32(stored * SCALE)
        coord = min(max(float(norm), 0.0), 1.0) * idx_width
        idx = min(idx_width - 1, int(np.floor(coord)))
        op = OP_TABLE[idx]
        rgb = COL_TABLE[idx]
        if op > 0.0:
            w = f32(1.0 - accA)
            accC, accA = comp(w, rgb, op, accC, accA)
            if accA >= np.float32(1.0 - 1.0 / 255.0):
                accA = np.float32(1.0)
                break
        p = f32(p + step)
    return accC, accA


def main():
    from PIL import Image
    anchor, step = extract_step((93, 201))
    p0 = f32(anchor + step)
    mt = np.array(Image.open(os.path.join(BC, 'u47_metal.png')))[201, 93][:3]
    gl = np.array(Image.open(os.path.join(BC, 'u47_gl.png')))[201, 93][:3]
    print(f'(93,201) metal={mt} gl={gl}')

    hits = []
    for cname, comp in COMPOSITES.items():
        for iw in (W, W - 1):
            for nd in (65536.0, 65535.0):
                accC, accA = simulate(p0, step, comp, iw, nd)
                u8 = np.array([round_half_even(np.clip(c, 0, 1) * 255) for c in accC])
                tag = f"comp={cname:18s} iw={iw:5d} nd={nd:8.0f} accC=({accC[0]:.6f},{accC[1]:.6f},{accC[2]:.6f}) accA={accA:.6f} u8={u8}"
                mark = " <== GL" if np.array_equal(u8, gl) else ""
                if np.array_equal(u8, gl):
                    hits.append(tag)
                print(tag + mark)
    print("\nGL-matching variants:", len(hits))
    for h in hits:
        print(" ", h)


if __name__ == '__main__':
    main()
