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

def interp_f64(attrs, lam):
    t = [lam[i]/w[i] for i in range(3)]
    return sum(t[i]*attrs[i] for i in range(3)) / sum(t)

data = []
for (mx,my) in KNIFE:
    gm = gl.get((mx, 511-my)); mm = mt.get((mx, my))
    if not gm or not mm: continue
    gt = [f(gm.group(i)) for i in (3,4,5)]
    mtp = [f(mm.group(i)) for i in (3,4,5)]
    base = ((mx+0.5)/256.0-1.0, (511-my+0.5)/256.0-1.0)
    data.append((mx,my,gt,mtp,base))

def predict(lam):
    return [interp_f64([vtex[i][ax] for i in range(3)], lam) for ax in range(3)]

def matches(base, dx, dy, target):
    lam = bary_f32(ndc32, (base[0]+dx, base[1]+dy))
    pred = predict(lam)
    return sum(1 for ax in range(3) if ulps(pred[ax], target[ax]) == 0)

# per-pixel scan for (dx,dy) making all 3 channels match GL exactly
solve_off = {}
print('Per-pixel search for sample offset (dx,dy) NDC s.t. all 3 tex channels == GL logged (0 ulps)')
print('%-9s | %-10s | %-10s | %-10s | %s' % ('px','best dx','best dy','#matching','per-channel ulps'))
for (mx,my,gt,mtp,base) in data:
    best = (0, None)
    for dx in np.arange(-2e-3, 2e-3, 2e-5):
        for dy in np.arange(-2e-3, 2e-3, 2e-5):
            n = matches(base, float(dx), float(dy), gt)
            if n > best[0]:
                best = (n, (float(dx), float(dy)))
            if n == 3: break
        if best[0] == 3: break
    n, (dx,dy) = best
    solve_off[(mx,my)] = (dx,dy)
    lam = bary_f32(ndc32, (base[0]+dx, base[1]+dy))
    pred = predict(lam)
    per = tuple(ulps(pred[ax], gt[ax]) for ax in range(3))
    print('(%3d,%3d) | %+1.2e | %+1.2e | %d/3        | %s' % (mx,my,dx,dy,n,per))

# ---- Result 4: the varying-path offset vs the position (clip) path are DECOUPLED ----
# clip interpolation uses the analytic pixel-center f32-NDC weights (offset ~ 0);
# the same offset that fixes texcoord breaks clip.x by thousands of ulps.
def interp_clip(lam):
    t = [lam[i]/w[i] for i in range(3)]
    return [sum(t[i]*vclip[i][ax] for i in range(3)) / sum(t) for ax in range(4)]

print()
print('clip.x ulps vs GL logged: at offset 0 vs at each pixel\'s tex-fixing offset')
for (mx,my,gt,mtp,base) in data:
    lam0 = bary_f32(ndc32, base)
    gm = gl.get((mx, 511-my)); gc = [f(gm.group(i)) for i in (6,7,8,9)]
    u0 = ulps(interp_clip(lam0)[0], gc[0])
    dx,dy = solve_off[(mx,my)]
    lam1 = bary_f32(ndc32, (base[0]+dx, base[1]+dy))
    u1 = ulps(interp_clip(lam1)[0], gc[0])
    print('(%3d,%3d) clip.x ulps @0=%+6d @tex-offset=%+6d' % (mx,my,u0,u1))
