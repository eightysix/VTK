import re
import numpy as np
import ctypes
import itertools
import os

BC = os.environ.get("BC_DATA", "/tmp/bc")

libm = ctypes.CDLL('libSystem.B.dylib')
fmaf = libm.fmaf
fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
fmaf.restype = ctypes.c_float
def fma(a, b, c): return float(fmaf(ctypes.c_float(a), ctypes.c_float(b), ctypes.c_float(c)))
def f32(x): return float(np.float32(x))
def f(x): return float(x)

GL_RAY_PAT = re.compile(
    r'DEBUG GL_RAY px=\((\d+), (\d+)\).*tex=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) '
    r'clip=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) flatVid=(\d+) primId=(\d+)')
MT_STEP_PAT = re.compile(
    r'DEBUG STEP px=\((\d+), (\d+)\).*localPos=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) '
    r'clip=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\)')
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

glv = parse_last(BC + '/u62_gl_vlog.log', GL_VERT_PAT, 1)
mtv = parse_last(BC + '/u62_metal.log', MT_VERT_PAT, 1)
gl = {}; mt = {}
for line in open(BC + '/u62_gl_vlog.log'):
    mm = GL_RAY_PAT.search(line)
    if not mm: continue
    gl[(int(mm.group(1)), int(mm.group(2)))] = mm
for line in open(BC + '/u62_metal.log'):
    mm = MT_STEP_PAT.search(line)
    if not mm: continue
    mt[(int(mm.group(1)), int(mm.group(2)))] = mm

def ulps(a, b):
    a = np.float32(a); b = np.float32(b)
    if a == b: return 0
    ia = a.view(np.uint32).item(); ib = b.view(np.uint32).item()
    if a < 0: ia ^= 0x80000000
    if b < 0: ib ^= 0x80000000
    return ia - ib

tri = (86, 40, 93)
vclip = []; vtex = []
for vid in tri:
    g, m = glv[vid], mtv[vid]
    gc = [f(g.group(i)) for i in (2,3,4,5)]
    gt = [f(g.group(i)) for i in (6,7,8)]
    mc = [f(m.group(i)) for i in (2,3,4,5)]
    assert all(np.array([gc[i] for i in (0,1,3)], dtype=np.float32) ==
               np.array([mc[i] for i in (0,1,3)], dtype=np.float32)), vid
    vclip.append(gc); vtex.append(gt)
w = [vclip[i][3] for i in range(3)]
ndc32 = [(f32(f32(vclip[i][0])/f32(vclip[i][3])), f32(f32(vclip[i][1])/f32(vclip[i][3]))) for i in range(3)]

def bary_f32(ndc, samp):
    A = np.array([ndc[0][0], ndc[0][1]], dtype=np.float32)
    B = np.array([ndc[1][0], ndc[1][1]], dtype=np.float32)
    C = np.array([ndc[2][0], ndc[2][1]], dtype=np.float32)
    P = np.array([np.float32(samp[0]), np.float32(samp[1])])
    def cross(u, v): return np.float32(np.float32(u[0]*v[1]) - np.float32(u[1]*v[0]))
    den = cross(B-A, C-A)
    l0 = np.float32(cross(B-P, C-P)/den)
    l1 = np.float32(cross(C-P, A-P)/den)
    l2 = np.float32(cross(A-P, B-P)/den)
    return (float(l0), float(l1), float(l2))

# interpolation formula variants. attr = per-vertex scalar list; w = per-vertex w; lam = weights.
def v_seq(attr, lam):
    t = [f32(lam[i]/w[i]) for i in range(3)]
    num = f32(f32(f32(t[0]*attr[0]) + f32(t[1]*attr[1])) + f32(t[2]*attr[2]))
    den = f32(f32(t[0] + t[1]) + t[2])
    return f32(num/den)

def v_fma(attr, lam):
    t = [f32(lam[i]/w[i]) for i in range(3)]
    num = fma(t[2], attr[2], fma(t[1], attr[1], f32(t[0]*attr[0])))
    den = fma(t[2], 1.0, fma(t[1], 1.0, t[0]))
    return f32(num/den)

def v_fma2(attr, lam):
    # fma with reverse order
    t = [f32(lam[i]/w[i]) for i in range(3)]
    num = fma(t[0], attr[0], fma(t[1], attr[1], f32(t[2]*attr[2])))
    den = fma(t[0], 1.0, fma(t[1], 1.0, t[2]))
    return f32(num/den)

def v_rcp(attr, lam):
    t = [f32(lam[i]/w[i]) for i in range(3)]
    num = f32(f32(f32(t[0]*attr[0]) + f32(t[1]*attr[1])) + f32(t[2]*attr[2]))
    den = f32(f32(t[0] + t[1]) + t[2])
    return f32(num * f32(1.0/den))

def v_f64(attr, lam):
    t = [lam[i]/w[i] for i in range(3)]
    return sum(t[i]*attr[i] for i in range(3)) / sum(t)

def v_interp_pos(attr, lam):
    # interpolate the attribute in clip/w space then multiply by interpolated w
    t = [f32(lam[i]/w[i]) for i in range(3)]
    wnum = f32(f32(f32(t[0]*w[0]) + f32(t[1]*w[1])) + f32(t[2]*w[2]))  # = 1 always? no: sum t_i w_i = sum lam = 1
    den = f32(f32(t[0] + t[1]) + t[2])
    iw = f32(1.0/den)
    num = f32(f32(f32(t[0]*attr[0]) + f32(t[1]*attr[1])) + f32(t[2]*attr[2]))
    return f32(num * iw)

VARIANTS = [('seq', v_seq), ('fma', v_fma), ('fma2', v_fma2), ('rcp', v_rcp), ('f64', v_f64), ('interppos', v_interp_pos)]

# collect GL logged texcoord and Metal logged texcoord per pixel
data = []
for (mx,my) in KNIFE:
    gm = gl.get((mx, 511-my)); mm = mt.get((mx, my))
    if not gm or not mm: continue
    gt = [f(gm.group(i)) for i in (3,4,5)]
    mtp = [f(mm.group(i)) for i in (3,4,5)]
    samp = (f32((mx+0.5)/256.0-1.0), f32((511-my+0.5)/256.0-1.0))
    lam = bary_f32(ndc32, samp)
    data.append((mx,my,gt,mtp,lam))

for name, fn in VARIANTS:
    err_gl = []; err_mt = []
    for (mx,my,gt,mtp,lam) in data:
        for ax in range(3):
            attrs = [vtex[0][ax], vtex[1][ax], vtex[2][ax]]
            pred = fn(attrs, lam)
            err_gl.append(ulps(pred, gt[ax]))
            err_mt.append(ulps(pred, mtp[ax]))
    a_gl = np.array(err_gl); a_mt = np.array(err_mt)
    print('%-10s GL: sum|u|=%-5d  max=%+d  mean=%+.2f    Mt: sum|u|=%-5d  max=%+d  mean=%+.2f' % (
        name, abs(a_gl).sum(), a_gl.max(), a_gl.mean(), abs(a_mt).sum(), a_mt.max(), a_mt.mean()))
