import re
import numpy as np
import os

BC = os.environ.get("BC_DATA", "/tmp/bc")

def f32(x): return np.float32(x)
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
CAP_PAT = re.compile(r'DEBUG GL_CAPINDEX (\d+) (\d+) (\d+) (\d+)')

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

cap = {}
for line in open(BC + '/u62_gl_vlog.log'):
    mm = CAP_PAT.search(line)
    if not mm: continue
    cap[int(mm.group(1))] = (int(mm.group(2)), int(mm.group(3)), int(mm.group(4)))

def ulps(a, b):
    a = f32(a); b = f32(b)
    if a == b: return 0
    ia = a.view(np.uint32).item(); ib = b.view(np.uint32).item()
    if a < 0: ia ^= 0x80000000
    if b < 0: ib ^= 0x80000000
    return ia - ib

def backout_weights(vclip, interp_clip):
    Cx = np.array([vclip[i][0] for i in range(3)])
    Cy = np.array([vclip[i][1] for i in range(3)])
    w  = np.array([vclip[i][3] for i in range(3)])
    S = 1.0 / interp_clip[3]
    M = np.array([[Cx[0], Cx[1], Cx[2]],
                  [Cy[0], Cy[1], Cy[2]],
                  [1.0,   1.0,   1.0]])
    rhs = np.array([interp_clip[0]*S, interp_clip[1]*S, S])
    t = np.linalg.solve(M, rhs)
    lam = t * w
    return lam / lam.sum()

def analytic_weights_f32(ndc, samp_ndc):
    A = f32(np.array(ndc[0])); B = f32(np.array(ndc[1])); C = f32(np.array(ndc[2])); P = f32(np.array(samp_ndc))
    def cross(u, v): return f32(f32(u[0]*v[1]) - f32(u[1]*v[0]))
    den = cross(f32(B-A), f32(C-A))
    l0 = f32(cross(f32(B-P), f32(C-P))/den)
    l1 = f32(cross(f32(C-P), f32(A-P))/den)
    l2 = f32(cross(f32(A-P), f32(B-P))/den)
    return (float(l0), float(l1), float(l2))

def analytic_weights_f64(ndc, samp_ndc):
    A = np.array(ndc[0], dtype=np.float64); B = np.array(ndc[1]); C = np.array(ndc[2]); P = np.array(samp_ndc)
    def cross(u, v): return u[0]*v[1] - u[1]*v[0]
    den = cross(B-A, C-A)
    return (cross(B-P, C-P)/den, cross(C-P, A-P)/den, cross(A-P, B-P)/den)

def predict_f32(clip, tex, weights):
    w  = np.array([clip[i][3] for i in range(3)], dtype=np.float32)
    lam = np.array(weights, dtype=np.float32)
    tw = lam / w
    denom = f32(tw.sum())
    pc = [float(f32(sum(np.float32(tw[i])*np.float32(clip[i][c]) for i in range(3))/denom)) for c in (0,1,3)]
    pt = [float(f32(sum(np.float32(tw[i])*np.float32(tex[i][c]) for i in range(3))/denom)) for c in (0,1,2)]
    return pc, pt

def predict_f64(clip, tex, weights):
    w  = np.array([clip[i][3] for i in range(3)], dtype=np.float64)
    lam = np.array(weights, dtype=np.float64)
    tw = lam / w
    denom = tw.sum()
    pc = [float(sum(tw[i]*clip[i][c] for i in range(3))/denom) for c in (0,1,3)]
    pt = [float(sum(tw[i]*tex[i][c] for i in range(3))/denom) for c in (0,1,2)]
    return pc, pt

tri = cap[122]
vclip = []; vtex = []
for vid in tri:
    g, m = glv[vid], mtv[vid]
    gc = [f(g.group(i)) for i in (2,3,4,5)]
    gt = [f(g.group(i)) for i in (6,7,8)]
    mc = [f(m.group(i)) for i in (2,3,4,5)]
    m2 = [f(m.group(i)) for i in (6,7,8)]
    assert all(np.array([gc[i] for i in (0,1,3)], dtype=np.float32) ==
               np.array([mc[i] for i in (0,1,3)], dtype=np.float32)), vid
    vclip.append(gc); vtex.append(gt)

# NDC via float32 divide
ndc = []
for v in vclip:
    ndc.append((float(f32(f32(v[0])/f32(v[3]))), float(f32(f32(v[1])/f32(v[3])))))

print('tri 122=%s' % (tri,))
for (mx,my) in KNIFE:
    gk = (mx, 511-my)
    mk = (mx, my)
    gm = gl.get(gk); mm = mt.get(mk)
    if not gm or not mm: continue
    gc = [f(gm.group(i)) for i in (12,13,14,15)]
    mc = [f(mm.group(i)) for i in (8,9,10,11)]
    gt = [f(gm.group(i)) for i in (9,10,11)]
    mtp = [f(mm.group(i)) for i in (5,6,7)]
    # rasterizer window sample: GL gl_FragCoord convention, y up
    sx = (mx+0.5)/256.0 - 1.0
    sy = (511-my+0.5)/256.0 - 1.0
    samp = (sx, sy)
    wg = backout_weights(vclip, gc)
    wm = backout_weights(vclip, mc)
    an32 = analytic_weights_f32(ndc, samp)
    an64 = analytic_weights_f64(ndc, samp)
    # predicted clip/tex from each weight set
    pg32, pt32 = predict_f32(vclip, vtex, an32)
    pg64, pt64 = predict_f64(vclip, vtex, an64)
    pgG, ptG = predict_f64(vclip, vtex, wg)
    # weight delta GL-Metal backed out, relative
    dgm = np.array(wg) - np.array(wm)
    rel = dgm / np.array(wg)
    line = '(%3d,%3d) %-12s %-8.6f %-8.6f %-8.6f' % (mx,my,'backGL',wg[0],wg[1],wg[2])
    print(line)
    print('          %-12s %-8.6f %-8.6f %-8.6f   rel dG-dM=(%+.1e,%+.1e,%+.1e)' % ('backMt',wm[0],wm[1],wm[2],rel[0],rel[1],rel[2]))
    print('          %-12s %-8.6f %-8.6f %-8.6f' % ('anF32',an32[0],an32[1],an32[2]))
    print('          %-12s %-8.6f %-8.6f %-8.6f' % ('anF64',an64[0],an64[1],an64[2]))
    # compare predicted clip to GL logged
    cGL = tuple(ulps(pgG[i], gc[i]) for i in (0,1,2))
    c32 = tuple(ulps(pg32[i], gc[i]) for i in (0,1,2))
    c64 = tuple(ulps(pg64[i], gc[i]) for i in (0,1,2))
    tGL = tuple(ulps(ptG[i], gt[i]) for i in range(3))
    t32 = tuple(ulps(pt32[i], gt[i]) for i in range(3))
    t64 = tuple(ulps(pt64[i], gt[i]) for i in range(3))
    print('          clipU backG=%s anF32=%s anF64=%s' % (cGL, c32, c64))
    print('          texU  backG=%s anF32=%s anF64=%s' % (tGL, t32, t64))
    print()
