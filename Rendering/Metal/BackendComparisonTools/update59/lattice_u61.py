#!/usr/bin/env python3
"""Lattice comparison at the 14 knife-edge pixels: GL GL_RAY vs Metal STEP,
frame 6 (last occurrence per pixel)
(VolumeRayCastBackendComparisonFindingsUpdate59.md, section 4).

Findings it reproduces:
  - Step (Metal evalStep vs GL g_dirStep): essentially bit-identical,
    deltas <= ~6e-8 texels (x512) -- far below update 48's 1.6e-7 bound,
    which came from the debug-GL compile.
  - Anchor (GL g_rayOrigin vs Metal first sample localPos+evalStep): differs
    by 2e-5 .. 6.7e-5 texels (~1e-7 in [0,1] ~ 1 ulp of the float32
    interpolated texcoord). At grid-aligned rays this ~5e-5-texel offset
    crosses a nearest-texel boundary -> gf flip of 0.3-8 u8.
  - Interpolated clip: x/y/w match to <= ~6e-8; clip.z is NOT comparable
    between the dumps (GL_RAY ip_debugClip z vs Metal out.position z use
    different conventions).

Inputs (see README.md in this directory for regeneration):
    BC_DATA/u61_gl.log      GL_RAY lines (29 px, 6 frames)
    BC_DATA/u61_metal.log   Metal STEP lines (gated px, 6 frames)

The knife-edge pixels are paired by Metal screenPos (x,y) == GL (x, 511-y).

Usage: BC_DATA=/path/to/data python3 lattice_u61.py
"""
import os
import re
import numpy as np

BC = os.environ.get("BC_DATA", "/tmp/bc")

KNIFE = [(397, 110), (360, 229), (349, 255), (405, 171), (9, 18), (293, 298), (338, 432),
         (350, 5), (153, 32), (482, 33), (120, 167), (470, 269), (439, 281), (469, 463)]

GL_PAT = re.compile(
    r'DEBUG GL_RAY px=\((\d+), (\d+)\).*origin=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) '
    r'step=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\).*tex=\([\d.e+-]+, [\d.e+-]+, [\d.e+-]+\) '
    r'clip=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\)')
MT_PAT = re.compile(
    r'DEBUG STEP px=\((\d+), (\d+)\).*localPos=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) '
    r'clip=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\).*'
    r'evalStep=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\)')


def f(x):
    return float(x)


def parse_gl(path):
    gl = {}
    for line in open(path):
        mm = GL_PAT.search(line)
        if not mm:
            continue
        gl[(int(mm.group(1)), int(mm.group(2)))] = {
            'origin': [f(mm.group(3)), f(mm.group(4)), f(mm.group(5))],
            'step': [f(mm.group(6)), f(mm.group(7)), f(mm.group(8))],
            'clip': [f(mm.group(9)), f(mm.group(10)), f(mm.group(11)), f(mm.group(12))]}
    return gl


def parse_metal(path):
    mt = {}
    for line in open(path):
        mm = MT_PAT.search(line)
        if not mm:
            continue
        mt[(int(mm.group(1)), int(mm.group(2)))] = {
            'localPos': [f(mm.group(3)), f(mm.group(4)), f(mm.group(5))],
            'clip': [f(mm.group(6)), f(mm.group(7)), f(mm.group(8)), f(mm.group(9))],
            'evalStep': [f(mm.group(10)), f(mm.group(11)), f(mm.group(12))]}
    return mt


def main():
    gl = parse_gl(os.path.join(BC, 'u61_gl.log'))
    mt = parse_metal(os.path.join(BC, 'u61_metal.log'))
    print('GL_RAY parsed px:', len(gl), ' Metal STEP parsed px:', len(mt))

    print('%-9s %-34s %-22s %-22s' % (
        'px', 'clip delta x/y/w (GL-Metal)', 'anchor res (texels)', 'step diff (texels)'))
    for mx, my in KNIFE:
        key = (mx, 511 - my)
        if key not in gl or (mx, my) not in mt:
            print('(%3d,%3d) missing dump (gl=%d mt=%d)' % (mx, my, key in gl, (mx, my) in mt))
            continue
        g = gl[key]
        mm = mt[(mx, my)]
        cd = np.array(g['clip']) - np.array(mm['clip'])
        lp = np.array(mm['localPos'])
        es = np.array(mm['evalStep'])
        og = np.array(g['origin'])
        res = (og - (lp + es)) * 512.0
        sd = (np.array(g['step']) - es) * 512.0
        print('(%3d,%3d) %-34s %-22s %-22s' % (
            mx, my, '(%+.3e,%+.3e,%+.3e)' % tuple(cd[:3]),
            '(%+.3e,%+.3e,%+.3e)' % tuple(res), '(%+.3e,%+.3e,%+.3e)' % tuple(sd)))

    print()
    print('Reading: clip x/y/w <= ~6e-8; step <= ~6e-8 texels (bit-identical);')
    print('anchor res ~2e-5..6.7e-5 texels (~1 ulp) -> the knife-edge flip driver.')


if __name__ == '__main__':
    main()
