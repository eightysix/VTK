import re
import numpy as np
import ctypes
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
    r'DEBUG GL_RAY px=\((\d+), (\d+)\).*origin=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) '
    r'step=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\).*'
    r'vpos=\([\d.e+-]+, [\d.e+-]+, [\d.e+-]+\) tex=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) '
    r'clip=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) flatVid=(\d+) primId=(\d+)')
MT_STEP_PAT = re.compile(
    r'DEBUG STEP px=\((\d+), (\d+)\).*screenPos=\(([\d.e+-]+), ([\d.e+-]+)\).*'
    r'localPos=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) '
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

def window_bary_f32(vclip, samp_ndc):
    # window coords from f32 divide, then area-ratio bary in f32
    P = np.array([f32(samp_ndc[0]), f32(samp_ndc[1])])
    A = np.array([f32(f32(vclip[0][0])/f32(vclip[0][3])), f32(f32(vclip[0][1])/f32(vclip[0][3]))])
    B = np.array([f32(f32(vclip[1][0])/f32(vclip[1][3])), f32(f32(vclip[1][1])/f32(vclip[1][3]))])
    C = np.array([f32(f32(vclip[2][0])/f32(vclip[2][3])), f32(f32(vclip[2][1])/f32(vclip[2][3]))])
    def cross(u, v): return f32(f32(u[0]*v[1]) - f32(u[1]*v[0]))
    den = cross(B-A, C-A)
    l0 = f32(cross(B-P, C-P)/den)
    l1 = f32(cross(C-P, A-P)/den)
    l2 = f32(cross(A-P, B-P)/den)
    return (l0, l1, l2)

def interp_seq_f32(attr, w, lam):
    # attr = (sum lam_i/w_i * attr_i) / (sum lam_i/w_i), sequential f32
    s = 0.0
    for i in range(3):
        t = f32(f32(lam[i]) / f32(w[i]))
        s = f32(s + f32(t * f32(attr[i])))
    d = 0.0
    for i in range(3):
        d = f32(d + f32(f32(lam[i]) / f32(w[i])))
    return f32(s / d)

def interp_fma_f32(attr, w, lam):
    # FMA accumulation: s = fma(t2, attr2, fma(t1, attr1, t0*attr0))
    t0 = f32(f32(lam[0])/f32(w[0])); t1 = f32(f32(lam[1])/f32(w[1])); t2 = f32(f32(lam[2])/f32(w[2]))
    num = fma(t2, f32(attr[2]), fma(t1, f32(attr[1]), f32(t0*f32(attr[0]))))
    den = fma(t2, 1.0, fma(t1, 1.0, t0))
    return f32(num / den)

def interp_f64(attr, w, lam):
    t = [lam[i]/w[i] for i in range(3)]
    return sum(t[i]*attr[i] for i in range(3)) / sum(t)

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

print('%-9s %-5s %-26s %-26s %-26s' % ('px','axis','seq_f32','fma_f32','f64'))
print('  values = ULP distance (prediction vs GL logged  /  vs Metal logged)')
for (mx,my) in KNIFE:
    gm = gl.get((mx, 511-my)); mm = mt.get((mx, my))
    if not gm or not mm: continue
    gt = [f(gm.group(i)) for i in (9,10,11)]
    mtp = [f(mm.group(i)) for i in (5,6,7)]
    # rasterizer sample in NDC (y up, GL gl_FragCoord convention)
    samp = ((mx+0.5)/256.0-1.0, (511-my+0.5)/256.0-1.0)
    lam = window_bary_f32(vclip, samp)
    for ax in range(3):
        a_gl = gt[ax]; a_mt = mtp[ax]
        attrs = [vtex[0][ax], vtex[1][ax], vtex[2][ax]]
        p_seq = interp_seq_f32(attrs, w, lam)
        p_fma = interp_fma_f32(attrs, w, lam)
        p_f64 = interp_f64(attrs, w, lam)
        print('(%3d,%3d) %-5s GL=%-5d/%-5d Mt=%-5d/%-5d seq: GL%+d Mt%+d | fma: GL%+d Mt%+d | f64: GL%+d Mt%+d' % (
            mx, my, 'xyz'[ax],
            0, 0, 0, 0,
            ulps(p_seq, a_gl), ulps(p_seq, a_mt),
            ulps(p_fma, a_gl), ulps(p_fma, a_mt),
            ulps(p_f64, a_gl), ulps(p_f64, a_mt)))
    print()
