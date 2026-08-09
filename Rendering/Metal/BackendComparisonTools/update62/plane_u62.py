import re
import os
import numpy as np

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
wc = [vclip[i][3] for i in range(3)]

# ---- window (pixel) coordinates of the 3 vertices, bottom-up y (gl_FragCoord) ----
# f32 NDC already established; window = (ndc+1)*256 (framebuffer is 512 wide)
def window_vertices(f64=False):
    out = []
    for i in range(3):
        ndc = (f32(f32(vclip[i][0])/f32(vclip[i][3])),
               f32(f32(vclip[i][1])/f32(vclip[i][3])))
        if f64:
            wx = (ndc[0]+1.0)*256.0
            wy = (ndc[1]+1.0)*256.0
        else:
            wx = f32((ndc[0]+1.0)*256.0)
            wy = f32((ndc[1]+1.0)*256.0)
        out.append((wx, wy))
    return out

W = window_vertices(f64=True)   # values already f32-rounded ndc, product in f64
SAMP = {}
for (mx,my) in KNIFE:
    SAMP[(mx,my)] = ((mx+0.5), (511-my+0.5))

# ---- affine plane coefficients (c0 + c1*x + c2*y) via Cramer, f64 or f32 ----
def plane_cramer(pts, vals, f64):
    (x0,y0),(x1,y1),(x2,y2) = pts
    g0,g1,g2 = vals
    if f64:
        det = x0*(y1-y2) - y0*(x1-x2) + (x1*y2 - x2*y1)
        c1 = (g0*(y1-y2) - y0*(g1-g2) + (g1*y2 - g2*y1))/det
        c2 = (x0*(g1-g2) + g0*(x2-x1) + (x1*g2 - x2*g1))/det
        c0 = g0 - c1*x0 - c2*y0
        return (c0, c1, c2)
    else:
        det = f32(x0*f32(y1-y2) - f32(y0*f32(x1-x2)) + f32(f32(x1*y2) - f32(x2*y1)))
        c1 = f32(f32(f32(g0*f32(y1-y2)) - f32(y0*f32(g1-g2)) + f32(f32(g1*y2) - f32(g2*y1)))/det)
        c2 = f32(f32(f32(x0*f32(g1-g2)) + f32(g0*f32(x2-x1)) + f32(f32(x1*g2) - f32(x2*g1)))/det)
        c0 = f32(f32(g0 - f32(c1*x0)) - f32(c2*y0))
        return (c0, c1, c2)

def plane_diag(pts, vals, base, f64):
    # proper 2x2 solve from base vertex: c1*(x_i-x0)+c2*(y_i-y0)=g_i-g0 for i=1,2
    b = base
    (x0,y0),(x1,y1),(x2,y2) = pts
    g0,g1,g2 = vals
    if f64:
        D = (x1-x0)*(y2-y0) - (y1-y0)*(x2-x0)
        c1 = ((g1-g0)*(y2-y0) - (y1-y0)*(g2-g0))/D
        c2 = ((x1-x0)*(g2-g0) - (g1-g0)*(x2-x0))/D
        c0 = g0 - c1*x0 - c2*y0
    else:
        D = f32(f32(f32(x1-x0)*f32(y2-y0)) - f32(f32(y1-y0)*f32(x2-x0)))
        c1 = f32(f32(f32(f32(g1-g0)*f32(y2-y0)) - f32(f32(y1-y0)*f32(g2-g0)))/D)
        c2 = f32(f32(f32(f32(x1-x0)*f32(g2-g0)) - f32(f32(g1-g0)*f32(x2-x0)))/D)
        c0 = f32(f32(f32(g0 - f32(c1*x0)) - f32(c2*y0)))
    return (c0, c1, c2)

def eval_plane(coef, sx, sy, f64):
    c0,c1,c2 = coef
    return (c0 + c1*sx + c2*sy) if f64 else f32(f32(c0 + f32(c1*f32(sx))) + f32(c2*f32(sy)))

def affine_interp(pts, vals, sx, sy, mode, base=0):
    f64 = (mode in ('f64',))
    if mode in ('f64', 'f32'):
        coef = plane_cramer(pts, vals, f64)
    else:
        coef = plane_diag(pts, vals, base, f64)
    return eval_plane(coef, sx, sy, f64)

# perspective-correct: interp(a) = affine(a/w)/affine(1/w)
def persp_interp(pts, attrs, wc, sx, sy, mode, base=0):
    aw = [attrs[i]/wc[i] for i in range(3)]
    iw = [1.0/wc[i] for i in range(3)]
    if mode not in ('f64', 'f32'):
        aw = [f32(x) for x in aw]; iw = [f32(x) for x in iw]
    num = affine_interp(pts, aw, sx, sy, mode, base)
    den = affine_interp(pts, iw, sx, sy, mode, base)
    return num/den if 'f64' in mode else f32(num/den)

MODES = ['f64', 'f32', 'diag0_f64', 'diag0_f32', 'diag1_f32', 'diag2_f32']

# collect logged interpolated values per pixel
data = []
for (mx,my) in KNIFE:
    gm = gl.get((mx, 511-my)); mm = mt.get((mx, my))
    if not gm or not mm: continue
    gc = [f(gm.group(i)) for i in (6,7,8,9)]
    mc = [f(mm.group(i)) for i in (6,7,8,9)]
    gt = [f(gm.group(i)) for i in (3,4,5)]
    mtp = [f(mm.group(i)) for i in (3,4,5)]
    data.append((mx,my,gc,mc,gt,mtp))

print('window verts (f64): %s' % ([('%.9g'%p[0], '%.9g'%p[1]) for p in W]))
print()
print('A) clip.w = 1/affine(1/w) at sample vs logged clip.w (ulps, GL/Metal)')
for mode in MODES:
    eG = []; eM = []
    for (mx,my,gc,mc,gt,mtp) in data:
        sx, sy = SAMP[(mx,my)]
        w_at = affine_interp(W, [1.0/wc[i] for i in range(3)], sx, sy, mode)
        pred = f32(1.0/w_at)
        eG.append(ulps(pred, gc[3])); eM.append(ulps(pred, mc[3]))
    aG = np.array(eG); aM = np.array(eM)
    print('  %-10s GL sum|u|=%-4d max%+d mean%+.2f | Mt sum|u|=%-4d max%+d mean%+.2f' % (
        mode, abs(aG).sum(), aG.max(), aG.mean(), abs(aM).sum(), aM.max(), aM.mean()))

print()
print('B) clip.x/y = (sx/256-1)*w_at, (sy/256-1)*w_at vs logged clip.x/y (ulps, GL/Metal)')
for mode in MODES:
    eGx=[];eGy=[];eMx=[];eMy=[]
    for (mx,my,gc,mc,gt,mtp) in data:
        sx, sy = SAMP[(mx,my)]
        w_at = affine_interp(W, [1.0/wc[i] for i in range(3)], sx, sy, mode)
        wc_at = f32(1.0/w_at)
        pxc = f32((sx/256.0-1.0)*wc_at)
        pyc = f32((sy/256.0-1.0)*wc_at)
        eGx.append(ulps(pxc, gc[0])); eGy.append(ulps(pyc, gc[1]))
        eMx.append(ulps(pxc, mc[0])); eMy.append(ulps(pyc, mc[1]))
    aGx=np.array(eGx);aGy=np.array(eGy);aMx=np.array(eMx);aMy=np.array(eMy)
    print('  %-10s GL x sum|u|=%-4d y=%-4d | Mt x sum|u|=%-4d y=%-4d' % (
        mode, abs(aGx).sum(), abs(aGy).sum(), abs(aMx).sum(), abs(aMy).sum()))

print()
print('C) texcoord = persp(affine(a/w)/affine(1/w)) vs logged tex (ulps, GL/Metal)')
for mode in MODES:
    eG=[];eM=[]
    for (mx,my,gc,mc,gt,mtp) in data:
        sx, sy = SAMP[(mx,my)]
        for ax in range(3):
            pred = persp_interp(W, [vtex[i][ax] for i in range(3)], wc, sx, sy, mode)
            eG.append(ulps(pred, gt[ax])); eM.append(ulps(pred, mtp[ax]))
    aG=np.array(eG);aM=np.array(eM)
    print('  %-10s GL sum|u|=%-4d max%+d mean%+.2f | Mt sum|u|=%-4d max%+d mean%+.2f' % (
        mode, abs(aG).sum(), aG.max(), aG.mean(), abs(aM).sum(), aM.max(), aM.mean()))
