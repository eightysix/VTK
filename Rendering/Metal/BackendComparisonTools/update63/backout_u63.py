import re
import os
import numpy as np

BC = os.environ.get("BC_DATA", "/tmp/bc")

def f32(x): return float(np.float32(x))
def f(x): return float(x)

GL_RAY_PAT = re.compile(
    r'DEBUG GL_RAY px=\((\d+), (\d+)\).*tex=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) '
    r'clip=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) flatVid=(\d+) primId=(\d+)')
GL_VERT_PAT = re.compile(
    r'DEBUG GL_VERT (\d+) px=\(\d+, \d+\) clip=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) '
    r'pos=\([\d.e+-]+, [\d.e+-]+, [\d.e+-]+\) tex=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\)')

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
gl = {}
for line in open(BC + '/u62_gl_vlog.log'):
    mm = GL_RAY_PAT.search(line)
    if not mm: continue
    gl[(int(mm.group(1)), int(mm.group(2)))] = mm

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
    g = glv[vid]
    vclip.append([f(g.group(i)) for i in (2,3,4,5)])
    vtex.append([f(g.group(i)) for i in (6,7,8)])
w = [vclip[i][3] for i in range(3)]
ndc32 = [(f32(f32(vclip[i][0])/f32(vclip[i][3])),
          f32(f32(vclip[i][1])/f32(vclip[i][3]))) for i in range(3)]

# --- back out effective perspective weights t_i (lam_i/w_i, up to scale) from the
# 3 logged texcoord channels (least squares), then lam_i = normalize(t_i * w_i) ---
def backout_lam(tex):
    # tex_ax = (sum_i t_i * vtex[i][ax]) / sum_i t_i ; t free up to scale. Let t0=1.
    # tex_ax = (vtex[0][ax] + t1*vtex[1][ax] + t2*vtex[2][ax]) / (1+t1+t2)
    # -> tex_ax*(1+t1+t2) = vtex[0][ax] + t1*vtex[1][ax] + t2*vtex[2][ax]
    # -> (tex_ax - vtex[1][ax])*t1 + (tex_ax - vtex[2][ax])*t2 = vtex[0][ax] - tex_ax
    A = []; b = []
    for ax in range(3):
        A.append([tex[ax] - vtex[1][ax], tex[ax] - vtex[2][ax]])
        b.append(vtex[0][ax] - tex[ax])
    sol, res, rank, _ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    t1 = float(sol[0]); t2 = float(sol[1]); t0 = 1.0
    lam = np.array([t0*w[0], t1*w[1], t2*w[2]])
    lam = lam/lam.sum()
    return lam, float(res[0]) if res.size else 0.0

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

print('Per-pixel backed-out effective weights (from GL texcoord) vs analytic pixel-center f32-NDC weights:')
print('%-9s | %-28s | %-28s | %-10s' % ('px','backed-out lam (l0,l1,l2)','an-f32N lam (l0,l1,l2)','rel d(lam) max'))
back = {}
for (mx,my) in KNIFE:
    gm = gl.get((mx, 511-my))
    if not gm: continue
    gt = [f(gm.group(i)) for i in (3,4,5)]
    base = ((mx+0.5)/256.0-1.0, (511-my+0.5)/256.0-1.0)
    lam, res = backout_lam(gt)
    ana = bary_f32(ndc32, base)
    rel = max(abs(lam[i]-ana[i])/max(abs(ana[i]),1e-12) for i in range(3))
    back[(mx,my)] = (lam, res)
    print('(%3d,%3d) | (%+.6f,%+.6f,%+.6f) | (%+.6f,%+.6f,%+.6f) | %+.2e' % (
        mx,my,lam[0],lam[1],lam[2],ana[0],ana[1],ana[2],rel))

print()
print('Residual of least-squares fit (should be ~0 if 3 channels are consistent with ONE weight set):')
for (mx,my) in KNIFE:
    if (mx,my) in back:
        lam, res = back[(mx,my)]
        print('(%3d,%3d) lstsq res = %.3e' % (mx,my,res))
