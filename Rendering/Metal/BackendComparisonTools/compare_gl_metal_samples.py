#!/usr/bin/env python3
"""Compare per-sample GL and Metal ray-cast logs for one screen pixel.

Pairing rule: the Metal MARCH/SAMPLE logs use Metal screenPos (origin top-left);
the GL_SAMPLE log uses glReadPixels coords (origin bottom-left). For a 512x512
viewport the SAME physical pixel is
    Metal (x, y)  ==  GL (x, 511 - y)
so the GL log for the Metal pixel (422,92) lives under GL_SAMPLE px=(422,419).
Verify the GL ray for that flipped pixel matches the Metal MARCH ray before
trusting the comparison (both should share origin/step to ~1e-5).

Note: the earlier confusion comparing GL (422,92) against Metal (422,92) was
wrong -- those are different physical pixels and the sample positions diverge
systematically. The y-flipped pairing makes positions agree to <=1e-5 and raw
agree to <=1e-6 except at a handful of samples (i=30, 134, 167 in the NoJitter
test, which are frame-ordering / camera-animation artifacts, not backend
differences).

Usage: python3 compare_gl_metal_samples.py [gl_samples.log] [metal_samples.log] [Mx] [My]
  gl_samples.log  : stderr of the GL run (VTK_GL_RAY_DUMP=1 VTK_GL_SAMPLE_DUMP=1 VTK_GL_SAMPLE_DUMP_PX=Gx,Gy)
  metal_samples.log: stderr of the Metal run (MTL_LOG_LEVEL=... MTL_LOG_TO_STDERR=1)
  Mx,My           : Metal pixel, default 422 92. The GL pixel is read as (Mx, 511-My).

Reference test used to develop/validate this tool (NoJitter variant of the
camera-inside scene, 6 frames, deterministic per backend):

  # GL
  VTK_GL_RAY_DUMP=1 VTK_GL_SAMPLE_DUMP=1 VTK_GL_SAMPLE_DUMP_PX=422,419 \
    build_macos_metal/bin/vtkRenderingVolumeCxxTests \
      TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
      --vtk-factory-prefer RenderingBackend=OpenGL \
      -D build_macos_metal/ExternalData/Testing -T build_macos_metal/Testing/Temporary \
      -V /tmp/bc/dummy_baseline.png 2> gl_samples.log

  # Metal
  MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=16777216 MTL_LOG_TO_STDERR=1 \
    build_macos_metal/bin/vtkRenderingVolumeCxxTests \
      TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
      --vtk-factory-prefer RenderingBackend=Metal \
      -D build_macos_metal/ExternalData/Testing -T build_macos_metal/Testing/Temporary \
      -V /tmp/bc/dummy_baseline.png 2> metal_samples.log

(VTK_GL_RAY_DUMP is required: the sample dump block sits under the RAY dump
gate. Both runs need a wrong baseline so the test renders and dumps.)
"""
import re
import sys


def parse_gl(path, px):
    out = {}
    pat = re.compile(
        r'GL_SAMPLE px=\((\d+), (\d+)\) i=(\d+) raw=([0-9.e+-]+) '
        r'pos=\(([0-9.e+-]+), ([0-9.e+-]+), ([0-9.e+-]+)\)')
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m and (int(m.group(1)), int(m.group(2))) == px:
                out.setdefault(int(m.group(3)), []).append(
                    (float(m.group(4)), float(m.group(5)),
                     float(m.group(6)), float(m.group(7))))
    return out


def parse_metal(path, px):
    out = {}
    pat = re.compile(
        r'SAMPLE px=\((\d+), (\d+)\) i=(\d+) t=([0-9.e+-]+) '
        r'tex=\(([0-9.e+-]+), ([0-9.e+-]+), ([0-9.e+-]+)\) '
        r'eval=\(([0-9.e+-]+), ([0-9.e+-]+), ([0-9.e+-]+)\) raw=([0-9.e+-]+)')
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m and (int(m.group(1)), int(m.group(2))) == px:
                out.setdefault(int(m.group(3)), []).append(
                    (float(m.group(4)), float(m.group(5)), float(m.group(6)),
                     float(m.group(7)), float(m.group(8)), float(m.group(9)),
                     float(m.group(10)), float(m.group(11))))
    return out


def main():
    gl_path = sys.argv[1] if len(sys.argv) > 1 else 'gl_samples.log'
    me_path = sys.argv[2] if len(sys.argv) > 2 else 'metal_samples.log'
    mx = int(sys.argv[3]) if len(sys.argv) > 3 else 422
    my = int(sys.argv[4]) if len(sys.argv) > 4 else 92
    gy = 511 - my  # glReadPixels y (bottom origin) for the same physical pixel

    gl = parse_gl(gl_path, (mx, gy))
    me = parse_metal(me_path, (mx, my))

    print(f'GL   pixel (Mx, 511-My) = ({mx}, {gy})')
    print(f'GL   per-i counts: {sorted(set(len(v) for v in gl.values()))}, '
          f'i range {min(gl)}..{max(gl)}')
    print(f'Metal per-i counts: {sorted(set(len(v) for v in me.values()))}, '
          f'i range {min(me)}..{max(me)}')

    print(f'\n{"i":>4} | {"GL pos":^26} | {"M eval":^26} | {"dist":>8} | '
          f'{"GLraw":>9} {"Mraw":>9} {"d":>9}')
    maxdist = 0.0
    maxraw = 0.0
    for i in sorted(set(gl) & set(me)):
        g = gl[i][0]
        m = me[i][0]
        gp = (g[1], g[2], g[3])
        ev = (m[4], m[5], m[6])
        # GL logs a -1e19 sentinel once its marching loop has terminated (early
        # alpha saturation); those rows carry no position/raw and are skipped.
        if abs(gp[0]) > 1e18:
            continue
        d = ((gp[0] - ev[0]) ** 2 + (gp[1] - ev[1]) ** 2 +
             (gp[2] - ev[2]) ** 2) ** 0.5
        rd = abs(g[0] - m[7])
        maxdist = max(maxdist, d)
        maxraw = max(maxraw, rd)
        flag = ' <-- raw diff' if rd > 1e-4 else ''
        print(f'{i:>4} | {str(tuple(round(x, 5) for x in gp)):^26} | '
              f'{str(tuple(round(x, 5) for x in ev)):^26} | {d:>8.6f} | '
              f'{g[0]:>9.6f} {m[7]:>9.6f} {rd:>9.6f}{flag}')
    print(f'\nmax position dist: {maxdist:.6f}   max |raw| diff: {maxraw:.6f}')


if __name__ == '__main__':
    main()
