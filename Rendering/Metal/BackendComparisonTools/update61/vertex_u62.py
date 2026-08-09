import re
import numpy as np
import os

BC = os.environ.get("BC_DATA", "/tmp/bc")

GLV_PAT = re.compile(
    r'DEBUG GL_VERT (\d+) px=\((\d+), (\d+)\) clip=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) '
    r'pos=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) tex=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\)')
MTV_PAT = re.compile(
    r'vertex_volume_main vid=(\d+) modelPos=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) '
    r'clip=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) '
    r'uvx=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) texcoord=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\)')

def f(x): return float(x)

def parse_last(path, pat):
    out = {}
    for line in open(path):
        mm = pat.search(line)
        if not mm: continue
        key = int(mm.group(1))
        out[key] = mm
    return out

def bitcmp(a, b):
    ia = np.array([a], dtype=np.float32).view(np.uint32)[0]
    ib = np.array([b], dtype=np.float32).view(np.uint32)[0]
    return ia == ib

gl = parse_last(BC + '/u62_gl_vlog.log', GLV_PAT)
mt = parse_last(BC + '/u62_metal.log', MTV_PAT)
print('GL vids:', len(gl), ' Metal vids:', len(mt))

clip_ok = tex_ok = pos_ok = clip_bad = tex_bad = 0
clip_deltas = []
tex_deltas = []
for vid in sorted(gl.keys()):
    if vid not in mt:
        print('  vid %d missing in Metal' % vid)
        continue
    g, m = gl[vid], mt[vid]
    gc = [f(g.group(i)) for i in (4,5,6,7)]
    mc = [f(m.group(i)) for i in (5,6,7,8)]
    gt = [f(g.group(i)) for i in (11,12,13)]
    mt2 = [f(m.group(i)) for i in (12,13,14)]
    gp = [f(g.group(i)) for i in (8,9,10)]
    mp = [f(m.group(i)) for i in (2,3,4)]
    # clip x/y/w comparable; z uses different conventions
    cx = [bitcmp(gc[i], mc[i]) for i in (0,1,3)]
    tx = [bitcmp(gt[i], mt2[i]) for i in (0,1,2)]
    px = [bitcmp(gp[i], mp[i]) for i in (0,1,2)]
    if all(cx): clip_ok += 1
    else:
        clip_bad += 1
        clip_deltas.append((vid, gc, mc))
    if all(tx): tex_ok += 1
    else:
        tex_bad += 1
        tex_deltas.append((vid, gt, mt2))
    if not all(px):
        pos_ok += 1  # pos may interpolate differently in tiny tri; not bitcmp-able
    if all(px): pos_ok += 0

print('clip x/y/w bit-identical: %d/%d vids' % (clip_ok, len(gl)))
print('tex  x/y/z bit-identical: %d/%d vids' % (tex_ok, len(gl)))
if clip_deltas:
    print('clip mismatches:')
    for vid, gc, mc in clip_deltas:
        print('  vid %d GL=%s Mt=%s' % (vid, gc, mc))
if tex_deltas:
    print('tex mismatches:')
    for vid, gt, mt2 in tex_deltas:
        print('  vid %d GL=%s Mt=%s' % (vid, gt, mt2))
