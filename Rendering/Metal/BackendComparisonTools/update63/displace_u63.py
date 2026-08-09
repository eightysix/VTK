import re, os
import numpy as np
BC = os.environ.get("BC_DATA", "/tmp/bc")
def f32(x): return float(np.float32(x))
def f(x): return float(x)

GL_RAY_PAT = re.compile(
    r'DEBUG GL_RAY px=\((\d+), (\d+)\).*tex=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\)')
MT_STEP_PAT = re.compile(
    r'DEBUG STEP px=\((\d+), (\d+)\).*localPos=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\)')
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
glv = parse_last(BC+'/u62_gl_vlog.log', GL_VERT_PAT, 1)
mtv = parse_last(BC+'/u62_metal.log', MT_VERT_PAT, 1)
gl = {}; mt = {}
for line in open(BC+'/u62_gl_vlog.log'):
    mm = GL_RAY_PAT.search(line)
    if not mm: continue
    gl[(int(mm.group(1)), int(mm.group(2)))] = mm
for line in open(BC+'/u62_metal.log'):
    mm = MT_STEP_PAT.search(line)
    if not mm: continue
    mt[(int(mm.group(1)), int(mm.group(2)))] = mm

tri = (86, 40, 93)
vclip = []; vtex = []
for vid in tri:
    g, m = glv[vid], mtv[vid]
    gc = [f(g.group(i)) for i in (2,3,4,5)]
    gt = [f(g.group(i)) for i in (6,7,8)]
    vclip.append(gc); vtex.append(gt)
w = [vclip[i][3] for i in range(3)]
ndc32 = [(f32(f32(vclip[i][0])/f32(vclip[i][3])), f32(f32(vclip[i][1])/f32(vclip[i][3]))) for i in range(3)]

def backout_lam(tex):
    A = []; b = []
    for ax in range(3):
        A.append([tex[ax] - vtex[1][ax], tex[ax] - vtex[2][ax]])
        b.append(vtex[0][ax] - tex[ax])
    sol, res, rank, _ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    t1 = float(sol[0]); t2 = float(sol[1]); t0 = 1.0
    lam = np.array([t0*w[0], t1*w[1], t2*w[2]])
    return lam/lam.sum()

def bary_f32(ndc, samp):
    A = np.array([ndc[0][0], ndc[0][1]], dtype=np.float32)
    B = np.array([ndc[1][0], ndc[1][1]], dtype=np.float32)
    C = np.array([ndc[2][0], ndc[2][1]], dtype=np.float32)
    P = np.array([np.float32(samp[0]), np.float32(samp[1])])
    def cross(u, v): return np.float32(np.float32(u[0]*v[1]) - np.float32(u[1]*v[0]))
    den = cross(B-A, C-A)
    return (float(np.float32(cross(B-P, C-P)/den)), float(np.float32(cross(C-P, A-P)/den)),
            float(np.float32(cross(A-P, B-P)/den)))

print('Effective sample displacement from pixel center (NDC), GL vs Metal, and delta:')
print('%-9s | %-22s | %-22s | %-16s | %-16s' % ('px','disp GL (dx,dy) NDC','disp Metal (dx,dy) NDC','disp delta (px)','disp delta (NDC)'))
dgls=[]; dmts=[]; dd=[]
for (mx,my) in KNIFE:
    gm = gl.get((mx, 511-my)); mm = mt.get((mx, my))
    if not gm or not mm: continue
    gt = [f(gm.group(i)) for i in (3,4,5)]
    ml = [f(mm.group(i)) for i in (3,4,5)]
    base = ((mx+0.5)/256.0-1.0, (511-my+0.5)/256.0-1.0)
    lgl = backout_lam(gt); lmt = backout_lam(ml)
    ana = bary_f32(ndc32, base)
    def disp(lam):
        px = sum(lam[i]*ndc32[i][0] for i in range(3)) - base[0]
        py = sum(lam[i]*ndc32[i][1] for i in range(3)) - base[1]
        return px, py
    dg = disp(lgl); dm = disp(lmt)
    dpx = ((dg[0]-dm[0])*256, (dg[1]-dm[1])*256)
    dgls.append(dg); dmts.append(dm); dd.append(dpx)
    print('(%3d,%3d) | (%+.4e, %+.4e)        | (%+.4e, %+.4e)        | (%+.4f, %+.4f) | (%+.4e, %+.4e)' % (
        mx,my,dg[0],dg[1],dm[0],dm[1],dpx[0],dpx[1],dg[0]-dm[0],dg[1]-dm[1]))

print()
import statistics as st
def rng(vals, i):
    xs = [v[i] for v in vals]
    return min(xs), max(xs)
print('GL  disp dx range', tuple('%.3e'%x for x in rng(dgls,0)), ' dy range', tuple('%.3e'%x for x in rng(dgls,1)))
print('MT  disp dx range', tuple('%.3e'%x for x in rng(dmts,0)), ' dy range', tuple('%.3e'%x for x in rng(dmts,1)))
print('GL-MT delta dx range (px)', tuple('%.3e'%x for x in rng(dd,0)), ' dy range (px)', tuple('%.3e'%x for x in rng(dd,1)))
print('GL disp mean', tuple('%.3e'%st.mean([v[i] for v in dgls]) for i in (0,1)))
print('MT disp mean', tuple('%.3e'%st.mean([v[i] for v in dmts]) for i in (0,1)))
print('GL-MT delta mean (px)', tuple('%.3e'%st.mean([v[i] for v in dd]) for i in (0,1)))
