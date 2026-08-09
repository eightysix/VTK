import re
import os
import numpy as np

BC = os.environ.get("BC_DATA", "/tmp/bc")
def f32(x): return float(np.float32(x))
def f(x): return float(x)

GL_RAY_PAT = re.compile(
    r'DEBUG GL_RAY px=\((\d+), (\d+)\).*tex=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\)')
MT_STEP_PAT = re.compile(
    r'DEBUG STEP px=\((\d+), (\d+)\).*localPos=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\)')
GL_VERT_PAT = re.compile(
    r'DEBUG GL_VERT (\d+) px=\(\d+, \d+\) clip=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) '
    r'pos=\([\d.e+-]+, [\d.e+-]+, [\d.e+-]+\) tex=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\)')
MT_VERT_PAT = re.compile(
    r'vertex_volume_main vid=(\d+).*clip=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\).*'
    r'texcoord=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\)')

KNIFE = [(397,110),(360,229),(349,255),(405,171),(9,18),(293,298),(338,432),
         (350,5),(153,32),(482,33),(120,167),(470,269),(439,281),(469,463)]

def parse_last(path, pat, key_idx):
    out = {}
    for line in open(path):
        mm = pat.search(line)
        if not mm: continue
        out[int(mm.group(key_idx))] = mm
    return out

glv = parse_last(BC+'/u62_gl_vlog.log', GL_VERT_PAT, 1)
mtv = parse_last(BC+'/u62_metal.log', MT_VERT_PAT, 1)
gl = {}; mt = {}
for line in open(BC+'/u62_gl_vlog.log'):
    mm = GL_RAY_PAT.search(line)
    if not mm: continue
    gl[(int(mm.group(1)), int(mm.group(2)))] = mm
for line in open(BC+'/u62_metal.log'):
    mm = MT_STEP_PAT.search(line)
    if not mm: continue
    mt[(int(mm.group(1)), int(mm.group(2)))] = mm

tri = (86, 40, 93)
vclip = []; vtex = []
for vid in tri:
    g, m = glv[vid], mtv[vid]
    gc = [f(g.group(i)) for i in (2,3,4,5)]
    gt = [f(g.group(i)) for i in (6,7,8)]
    mc = [f(m.group(i)) for i in (2,3,4,5)]
    mt_ = [f(m.group(i)) for i in (6,7,8)]
    assert all(np.array([gc[i] for i in (0,1,3)], dtype=np.float32) ==
               np.array([mc[i] for i in (0,1,3)], dtype=np.float32)), vid
    vclip.append(gc); vtex.append(gt)
w = [vclip[i][3] for i in range(3)]

def backout_lam(tex, vtex):
    A = []; b = []
    for ax in range(3):
        A.append([tex[ax] - vtex[1][ax], tex[ax] - vtex[2][ax]])
        b.append(vtex[0][ax] - tex[ax])
    sol, res, rank, _ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    t1 = float(sol[0]); t2 = float(sol[1]); t0 = 1.0
    lam = np.array([t0*w[0], t1*w[1], t2*w[2]])
    lam = lam/lam.sum()
    return lam

print('GL vs Metal per-pixel effective weights (backed out from each backend\'s own interpolated texcoord):')
print('%-9s | %-22s | %-22s | max|d lam|' % ('px','GL lam (l0,l1,l2)','Metal lam (l0,l1,l2)'))
for (mx,my) in KNIFE:
    gm = gl.get((mx, 511-my)); mm = mt.get((mx, my))
    if not gm or not mm: continue
    gt = [f(gm.group(i)) for i in (3,4,5)]
    ml = [f(mm.group(i)) for i in (3,4,5)]
    lgl = backout_lam(gt, vtex)
    lmt = backout_lam(ml, vtex)
    d = max(abs(lgl[i]-lmt[i]) for i in range(3))
    print('(%3d,%3d) | (%.9f, %.9f, %.9f) | (%.9f, %.9f, %.9f) | %.2e' % (
        mx,my,lgl[0],lgl[1],lgl[2],lmt[0],lmt[1],lmt[2],d))
