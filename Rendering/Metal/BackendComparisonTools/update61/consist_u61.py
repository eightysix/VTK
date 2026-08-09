import re
import numpy as np
import os

BC = os.environ.get("BC_DATA", "/tmp/bc")

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

def interp_f64(attr, lam):
    t = [lam[i]/w[i] for i in range(3)]
    return sum(t[i]*attr[i] for i in range(3)) / sum(t)

def window_bary_f64(vclip, samp):
    A = np.array([vclip[0][0]/vclip[0][3], vclip[0][1]/vclip[0][3]])
    B = np.array([vclip[1][0]/vclip[1][3], vclip[1][1]/vclip[1][3]])
    C = np.array([vclip[2][0]/vclip[2][3], vclip[2][1]/vclip[2][3]])
    P = np.array(samp)
    def cross(u, v): return u[0]*v[1]-u[1]*v[0]
    den = cross(B-A, C-A)
    return (cross(B-P, C-P)/den, cross(C-P, A-P)/den, cross(A-P, B-P)/den)

print('%-9s | %-6s | %-13s %-13s %-13s | %-5s %-5s | %s' % (
    'px','axis','pred_f64','GL_log','Mt_log','GLU','MtU','dU'))
for (mx,my) in KNIFE:
    gm = gl.get((mx, 511-my)); mm = mt.get((mx, my))
    if not gm or not mm: continue
    gc = [f(gm.group(i)) for i in (6,7,8,9)]
    mc = [f(mm.group(i)) for i in (6,7,8,9)]
    samp = ((mx+0.5)/256.0-1.0, (511-my+0.5)/256.0-1.0)
    lam = window_bary_f64(vclip, samp)
    for ax in (0,1):
        pred = interp_f64([vclip[i][ax] for i in range(3)], lam)
        gu = ulps(pred, gc[ax]); mu = ulps(pred, mc[ax])
        print('(%3d,%3d) | %s    | %13.9g %13.9g %13.9g | %+d %+d | %+d' % (
            mx,my,'xy'[ax], pred, gc[ax], mc[ax], gu, mu, gu-mu))
    print()
