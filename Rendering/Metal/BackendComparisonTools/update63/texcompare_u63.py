import re
import os
import numpy as np

BC = os.environ.get("BC_DATA", "/tmp/bc")
def f(x): return float(x)
def ulps(a, b):
    a = np.float32(a); b = np.float32(b)
    if a == b: return 0
    ia = a.view(np.uint32).item(); ib = b.view(np.uint32).item()
    if a < 0: ia ^= 0x80000000
    if b < 0: ib ^= 0x80000000
    return ia - ib

GL_RAY_PAT = re.compile(
    r'DEBUG GL_RAY px=\((\d+), (\d+)\).*tex=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\)')
MT_STEP_PAT = re.compile(
    r'DEBUG STEP px=\((\d+), (\d+)\).*localPos=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\)')

KNIFE = [(397,110),(360,229),(349,255),(405,171),(9,18),(293,298),(338,432),
         (350,5),(153,32),(482,33),(120,167),(470,269),(439,281),(469,463)]

gl = {}; mt = {}
for line in open(BC+'/u62_gl_vlog.log'):
    mm = GL_RAY_PAT.search(line)
    if not mm: continue
    gl[(int(mm.group(1)), int(mm.group(2)))] = mm
for line in open(BC+'/u62_metal.log'):
    mm = MT_STEP_PAT.search(line)
    if not mm: continue
    mt[(int(mm.group(1)), int(mm.group(2)))] = mm

print('GL interpolated texcoord (tex=) vs Metal interpolated texcoord (localPos=), 14 knife px:')
print('%-9s | %-30s | %-30s | %s' % ('px','GL tex','Metal localPos','ulp diff'))
for (mx,my) in KNIFE:
    gm = gl.get((mx, 511-my)); mm = mt.get((mx, my))
    if not gm or not mm:
        print('(%3d,%3d) MISSING gl=%s mt=%s' % (mx,my,bool(gm),bool(mm))); continue
    gt = [f(gm.group(i)) for i in (3,4,5)]
    mls = [f(mm.group(i)) for i in (3,4,5)]
    d = tuple(ulps(gt[ax], mls[ax]) for ax in range(3))
    print('(%3d,%3d) | (%.9e, %.9e, %.9e) | (%.9e, %.9e, %.9e) | %s' % (
        mx,my,gt[0],gt[1],gt[2],mls[0],mls[1],mls[2],d))
