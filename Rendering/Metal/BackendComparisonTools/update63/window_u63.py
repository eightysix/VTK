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
ndc = [(f32(f32(vclip[i][0])/f32(vclip[i][3])),
        f32(f32(vclip[i][1])/f32(vclip[i][3]))) for i in range(3)]

# ---- window-position rounding variants: xw = (ndc+1)*256 for a 512-wide fb ----
# candidate sequences; the same f64 value then rounded to f32 is the 'exact' reference.
def window_variants(ndc):
    out = {}
    ref = [( (ndc[i][0]+1.0)*256.0, (ndc[i][1]+1.0)*256.0 ) for i in range(3)]
    out['exact_f64'] = ref
    out['seq_f32_muladd'] = [ (f32(f32(f32(ndc[i][0])*256.0)+256.0),
                               f32(f32(f32(ndc[i][1])*256.0)+256.0)) for i in range(3)]
    out['seq_f32_addmul'] = [ (f32(f32(f32(ndc[i][0])+1.0)*256.0),
                               f32(f32(f32(ndc[i][1])+1.0)*256.0)) for i in range(3)]
    out['f64prod_f32'] = [ (f32((ndc[i][0]+1.0)*256.0), f32((ndc[i][1]+1.0)*256.0)) for i in range(3)]
    out['f32half_512'] = [ (f32(f32(f32(ndc[i][0])*0.5+0.5)*512.0),
                            f32(f32(f32(ndc[i][1])*0.5+0.5)*512.0)) for i in range(3)]
    return out

WV = window_variants(ndc)

# sample pixel centers in window coords (y-up gl_FragCoord convention)
SAMP = { (mx,my): (mx+0.5, 511-my+0.5) for (mx,my) in KNIFE }

def bary_at(wnd, samp, f64=False):
    (x0,y0),(x1,y1),(x2,y2) = wnd
    (px,py) = samp
    if f64:
        den = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2)
        l0 = ((y1-y2)*(px-x2) + (x2-x1)*(py-y2))/den
        l1 = ((y2-y0)*(px-x2) + (x0-x2)*(py-y2))/den
        l2 = 1.0-l0-l1
        return (l0,l1,l2)
    def cross(u, v):
        return f32(f32(u[0]*v[1]) - f32(u[1]*v[0]))
    A=(x0,y0);B=(x1,y1);C=(x2,y2);P=(px,py)
    den = cross((B[0]-A[0],B[1]-A[1]),(C[0]-A[0],C[1]-A[1]))
    l0 = f32(cross((B[0]-P[0],B[1]-P[1]),(C[0]-P[0],C[1]-P[1]))/den)
    l1 = f32(cross((C[0]-P[0],C[1]-P[1]),(A[0]-P[0],A[1]-P[1]))/den)
    l2 = f32(cross((A[0]-P[0],A[1]-P[1]),(B[0]-P[0],B[1]-P[1]))/den)
    return (l0,l1,l2)

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
    data.append((mx,my,gt))

print('Per-vertex window coords (xw, yw) under each rounding variant:')
for name, wnd in WV.items():
    print('  %-14s %s' % (name, [('%.6g,%.6g'%p) for p in wnd]))

print()
print('Per-pixel implied sample-NDC offset (from rounded-window barycentric at pixel center)')
print('and texcoord ulps vs GL logged, for each rounding variant:')
hdr = 'px        | %s' % (' | '.join('%-12s %-6s %-6s' % (n,'dx','u') for n in WV))
print(hdr)
for (mx,my,gt) in data:
    cells = []
    for name, wnd in WV.items():
        lam = bary_at(wnd, SAMP[(mx,my)], f64=False)
        # implied sample position from the weights applied to exact-f64 windows
        ref = WV['exact_f64']
        px_imp = sum(lam[i]*ref[i][0] for i in range(3))
        py_imp = sum(lam[i]*ref[i][1] for i in range(3))
        dx_ndc = (px_imp - SAMP[(mx,my)][0])/256.0
        dy_ndc = (py_imp - SAMP[(mx,my)][1])/256.0
        pred = [interp_f64([vtex[i][ax] for i in range(3)], lam) for ax in range(3)]
        u = max(abs(ulps(pred[ax], gt[ax])) for ax in range(3))
        cells.append('%+1.2e/%+1.2e %+5d %+5d' % (dx_ndc, dy_ndc, u, u))
    print('(%3d,%3d) | %s' % (mx,my, ' | '.join(cells)))
