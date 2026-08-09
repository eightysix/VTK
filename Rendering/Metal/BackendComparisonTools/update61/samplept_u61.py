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
# f32-rounded NDC (what the rasterizer's divide produces)
ndc32 = [(float(np.float32(np.float32(vclip[i][0])/np.float32(vclip[i][3]))),
          float(np.float32(np.float32(vclip[i][1])/np.float32(vclip[i][3])))) for i in range(3)]

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

print('Backing out effective barycentric weights from interpolated clip, then implied sample NDC.')
print('Compare implied sample NDC to pixel-center NDC.')
print('%-9s | %-6s | %-22s | %-22s | %-22s' % ('px','src','implied sample NDC','pixel-center NDC','delta (implied-center)'))
for (mx,my) in KNIFE:
    gm = gl.get((mx, 511-my)); mm = mt.get((mx, my))
    if not gm or not mm: continue
    gc = [f(gm.group(i)) for i in (6,7,8,9)]
    mc = [f(mm.group(i)) for i in (6,7,8,9)]
    cx = (mx+0.5)/256.0-1.0
    cy = (511-my+0.5)/256.0-1.0
    for name, interp in (('GL', gc), ('Mt', mc)):
        lam = backout_weights(vclip, interp)
        imp = (sum(lam[i]*ndc32[i][0] for i in range(3)),
               sum(lam[i]*ndc32[i][1] for i in range(3)))
        print('(%3d,%3d) | %-6s | (%13.6e, %13.6e) | (%13.6e, %13.6e) | (%+.3e, %+.3e)' % (
            mx,my,name,imp[0],imp[1],cx,cy,imp[0]-cx,imp[1]-cy))
    print()
