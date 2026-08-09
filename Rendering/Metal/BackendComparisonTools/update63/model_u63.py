import re, os
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
    vclip.append([f(g.group(i)) for i in (2,3,4,5)])
    vtex.append([f(g.group(i)) for i in (6,7,8)])
w = [vclip[i][3] for i in range(3)]
ndc32 = [(f32(f32(vclip[i][0])/f32(vclip[i][3])), f32(f32(vclip[i][1])/f32(vclip[i][3]))) for i in range(3)]

def bary_f32(ndc, samp):
    A = np.array([ndc[0][0], ndc[0][1]], dtype=np.float32)
    B = np.array([ndc[1][0], ndc[1][1]], dtype=np.float32)
    C = np.array([ndc[2][0], ndc[2][1]], dtype=np.float32)
    P = np.array([np.float32(samp[0]), np.float32(samp[1])])
    def cross(u, v): return np.float32(np.float32(u[0]*v[1]) - np.float32(u[1]*v[0]))
    den = cross(B-A, C-A)
    return (float(np.float32(cross(B-P, C-P)/den)), float(np.float32(cross(C-P, A-P)/den)),
            float(np.float32(cross(A-P, B-P)/den)))

def ulps(a, b):
    a = np.float32(a); b = np.float32(b)
    if a == b: return 0
    ia = a.view(np.uint32).item(); ib = b.view(np.uint32).item()
    if a < 0: ia ^= 0x80000000
    if b < 0: ib ^= 0x80000000
    return ia - ib

def interp(lam, wmodel):
    t = [lam[i]*wmodel[i] for i in range(3)]
    return [sum(t[i]*vtex[i][ax] for i in range(3))/sum(t) for ax in range(3)]

wm = {
  'affine':        [1.0, 1.0, 1.0],
  'persp_1/w':     [1.0/w[i] for i in range(3)],
  'persp_rcp':     [f32(1.0/w[i]) for i in range(3)],
  'inverse_w':     [w[i] for i in range(3)],
}
import itertools
print('Pixel-center f32-NDC weights, interpolation models vs logged texcoord (3/3 zero-ulp counts across 14 px):')
for name in wm:
    cg = cm = 0; tot = 0
    for (mx,my) in KNIFE:
        gm = gl.get((mx, 511-my)); mm = mt.get((mx, my))
        if not gm or not mm: continue
        base = ((mx+0.5)/256.0-1.0, (511-my+0.5)/256.0-1.0)
        lam = bary_f32(ndc32, base)
        pred = interp(lam, wm[name])
        gt = [f(gm.group(i)) for i in (3,4,5)]
        ml = [f(mm.group(i)) for i in (3,4,5)]
        tot += 1
        cg += sum(1 for ax in range(3) if ulps(pred[ax], gt[ax])==0) == 3
        cm += sum(1 for ax in range(3) if ulps(pred[ax], ml[ax])==0) == 3
    print('  %-12s GL 3/3: %d/14   Metal 3/3: %d/14' % (name, cg, cm))
