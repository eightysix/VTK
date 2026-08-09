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

def predict(lam):
    return [interp_f64([vtex[i][ax] for i in range(3)], lam) for ax in range(3)]

data = []
for (mx,my) in KNIFE:
    gm = gl.get((mx, 511-my))
    if not gm: continue
    gt = [f(gm.group(i)) for i in (3,4,5)]
    data.append((mx,my,gt,((mx+0.5)/256.0-1.0,(511-my+0.5)/256.0-1.0)))

# 1) fine per-pixel fit: find ALL (dx,dy) on a fine grid achieving 3/3, and the extent of that set
print('Fine per-pixel fit (step 1e-6) around the update-62 cluster:')
print('%-9s | %-20s | %-6s | %s' % ('px','3/3-offset span (dx range)','#3/3','phase(mx%2,my%2)'))
FINE = np.arange(-6e-4, 6e-4, 1e-6)
for (mx,my,gt,base) in data:
    xs=[]; ys=[]
    for dx in FINE:
        for dy in FINE:
            lam = bary_f32(ndc32, (base[0]+dx, base[1]+dy))
            pred = predict(lam)
            if all(ulps(pred[ax], gt[ax]) == 0 for ax in range(3)):
                xs.append(dx); ys.append(dy)
    if xs:
        print('(%3d,%3d) | dx[%+1.2e..%+1.2e] dy[%+1.2e..%+1.2e] | %4d   | (%d,%d)' % (
            mx,my,min(xs),max(xs),min(ys),max(ys),len(xs),mx%2,my%2))
    else:
        print('(%3d,%3d) | (no 3/3 on fine grid)                             | 0     | (%d,%d)' % (
            mx,my,mx%2,my%2))
