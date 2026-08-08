#!/usr/bin/env python3
"""Per-sample comparison + accumulation replay of GL and Metal ray-cast logs
for one screen pixel, plus a linear fit of the two sample-position traces.

This is the tool used to root-cause the NEAREST no-jitter camera-inside worst
pixel (Metal (372,131) == GL (372,380)) in
VolumeRayCastBackendComparisonFindingsUpdate16.md: exactly ONE sample (i=144)
flips the TF lookup (raw 0.0154270 vs 0.0181582 across the scalar-1150 knot),
driven by a linear sample-position drift between the backends of
~(1.2e-7, 2.4e-7, 6e-8) per sample (~0.018 texel of the 512^3 volume by i=144).

Pairing rule (same as compare_gl_metal_samples.py): Metal screenPos (top-left
origin) == GL glReadPixels (x, 511 - y). The GL_SAMPLE 8-channel dump prints
    GL_SAMPLE px=(Gx, Gy) i=N raw=.. pos=(..,..,..) color=(..,..,..) op=..
where color is PREMULTIPLIED (color.rgb = srcColor.a * TF color), so the TF
RGB is recovered as color/op. The Metal SAMPLE line prints the UNpremultiplied
rgb and the pre-integrated op, plus tex (raw texture coord) and eval (the
cell-to-point adjusted fetch coordinate actually used).

Both logs contain one dump per rendered frame (6 frames, deterministic per
backend); the last frame's values are used (matches how the logs were
developed). Sentinel samples (op < -1e18) are skipped.

Usage: python3 compare_gl_metal_accum.py [gl_samples.log] [metal_samples.log] [Mx] [My]
  gl_samples.log  : stderr of the GL run (VTK_GL_RAY_DUMP=1 VTK_GL_SAMPLE_DUMP=1 VTK_GL_SAMPLE_DUMP_PX=Gx,Gy)
  metal_samples.log: stderr of the Metal run (MTL_LOG_LEVEL=... MTL_LOG_TO_STDERR=1)
  Mx,My           : Metal pixel, default 372 131. GL pixel is read as (Mx, 511-My).

Reference capture for the worst pixel (NoJitter variant, 6 frames):

  # GL at (372,380)
  VTK_GL_RAY_DUMP=1 VTK_GL_SAMPLE_DUMP=1 VTK_GL_SAMPLE_DUMP_PX=372,380 \
    build_macos_metal/bin/vtkRenderingVolumeCxxTests \
      TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
      --vtk-factory-prefer RenderingBackend=OpenGL \
      -D build_macos_metal/ExternalData/Testing -T build_macos_metal/Testing/Temporary \
      -V /tmp/bc/dummy_baseline.png 2> gl372.log

  # Metal at (372,131)
  MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=16777216 MTL_LOG_TO_STDERR=1 \
    build_macos_metal/bin/vtkRenderingVolumeCxxTests \
      TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
      --vtk-factory-prefer RenderingBackend=Metal \
      -D build_macos_metal/ExternalData/Testing -T build_macos_metal/Testing/Temporary \
      -V /tmp/bc/dummy_baseline.png 2> metal_samples.log

Output:
  - a per-sample table (raw/op/rgb from both backends) for the region around
    the divergence, then the first diverging sample,
  - the front-to-back accumulation replay (accC, accA) for both backends at
    intervals plus the final composite colors,
  - a least-squares linear fit pos = pos0 + i*step over a clean window
    (default i=10..180, avoiding the exit-region clamp) for Metal evalPoint and
    GL g_dataPos, with the per-axis relative step difference.
"""
import argparse
import re
import sys

import numpy as np

SENTINEL = -8.99382e18


def parse_gl(path, px):
    pat = re.compile(
        r"GL_SAMPLE px=\((\d+), (\d+)\) i=(\d+) raw=([\d.e+-]+) "
        r"pos=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) "
        r"color=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) op=([\d.e+-]+)")
    out = {}
    for line in open(path):
        m = pat.search(line)
        if not m:
            continue
        gx, gy, i = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if (gx, gy) != px:
            continue
        op = float(m.group(11))
        if op < SENTINEL * 0.5:
            continue
        raw = float(m.group(4))
        pos = (float(m.group(5)), float(m.group(6)), float(m.group(7)))
        color = (float(m.group(8)), float(m.group(9)), float(m.group(10)))
        rgb = tuple(c / op for c in color) if op != 0.0 else (0.0, 0.0, 0.0)
        out[i] = dict(raw=raw, pos=pos, op=op, rgb=rgb)
    return out


def parse_metal(path, px):
    pat = re.compile(
        r"SAMPLE px=\((\d+), (\d+)\) i=(\d+) t=([\d.e+-]+) "
        r"tex=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) "
        r"eval=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\) "
        r"raw=([\d.e+-]+) norm=([\d.e+-]+) op=([\d.e+-]+) "
        r"mip=([\d.e+-]+) rgb=\(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)\)")
    out = {}
    for line in open(path):
        m = pat.search(line)
        if not m:
            continue
        mx, my, i = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if (mx, my) != px:
            continue
        op = float(m.group(13))
        if op < SENTINEL * 0.5:
            continue
        out[i] = dict(
            raw=float(m.group(11)),
            tex=(float(m.group(5)), float(m.group(6)), float(m.group(7))),
            eval=(float(m.group(8)), float(m.group(9)), float(m.group(10))),
            op=op,
            rgb=(float(m.group(15)), float(m.group(16)), float(m.group(17))))
    return out


def replay(samples):
    """Front-to-back accumulation from per-sample op/rgb."""
    accC = np.zeros(3)
    accA = 0.0
    rec = {}
    for i in sorted(samples):
        op = samples[i]["op"]
        rgb = np.array(samples[i]["rgb"])
        w = 1.0 - accA
        accC = accC + w * op * rgb
        accA = accA + w * op
        rec[i] = (accC.copy(), accA)
    return rec


def fit_trace(pos_by_i, lo, hi):
    idx = [i for i in range(lo, hi + 1) if i in pos_by_i]
    if len(idx) < 3:
        return None
    X = np.column_stack([np.ones(len(idx)), idx])
    Y = np.array([pos_by_i[i] for i in idx])
    coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    resid = np.abs(X @ coef - Y).max()
    return coef, resid, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gl_log", nargs="?", default="gl372.log")
    ap.add_argument("metal_log", nargs="?", default="metal_samples.log")
    ap.add_argument("mx", nargs="?", type=int, default=372)
    ap.add_argument("my", nargs="?", type=int, default=131)
    ap.add_argument("--lo", type=int, default=10)
    ap.add_argument("--hi", type=int, default=180)
    args = ap.parse_args()

    gl_px = (args.mx, 511 - args.my)
    metal_px = (args.mx, args.my)
    gl = parse_gl(args.gl_log, gl_px)
    mt = parse_metal(args.metal_log, metal_px)
    common = sorted(set(gl) & set(mt))
    if not common:
        print("no common samples -- wrong logs/pixels?", file=sys.stderr)
        return 1
    print(f"GL {len(gl)} samples at {gl_px}; Metal {len(mt)} samples at {metal_px}")

    # Find the first sample whose per-sample TF opacity or rgb diverges.
    flip = None
    for i in common:
        dop = abs(gl[i]["op"] - mt[i]["op"])
        drgb = max(abs(a - b) for a, b in zip(gl[i]["rgb"], mt[i]["rgb"]))
        if dop > 1e-4 or drgb > 1e-3:
            flip = i
            break

    lo = max(0, (flip or 0) - 14)
    hi = min(max(common), (flip or 0) + 12)
    print(f"\n=== per-sample region i={lo}..{hi} (first divergence i={flip}) ===")
    print("   i     raw_GL    raw_MT     op_GL     op_MT      rgb_GL            rgb_MT")
    for i in range(lo, hi + 1):
        if i not in gl or i not in mt:
            continue
        g, m = gl[i], mt[i]
        rgbg = tuple(f"{v:.5f}" for v in g["rgb"])
        rgbt = tuple(f"{v:.5f}" for v in m["rgb"])
        mark = " <--" if i == flip else ""
        print(f"{i:4d} {g['raw']:.7f} {m['raw']:.7f} {g['op']:.6f} {m['op']:.6f} "
              f"({rgbg[0]},{rgbg[1]},{rgbg[2]}) ({rgbt[0]},{rgbt[1]},{rgbt[2]}){mark}")

    # Accumulation replay.
    rg = replay(gl)
    rm = replay(mt)
    print("\n=== accumulation replay (first divergence + intervals) ===")
    print("   i     GL accC            MT accC            dAccC          GL aA   MT aA")
    for i in sorted(common):
        if i == flip or i % 50 == 0 or i == max(common):
            gc, ga = rg[i]
            mc, ma = rm[i]
            d = tuple(gc[j] - mc[j] for j in range(3))
            print(f"{i:4d} ({gc[0]:.6f},{gc[1]:.6f},{gc[2]:.6f}) "
                  f"({mc[0]:.6f},{mc[1]:.6f},{mc[2]:.6f}) "
                  f"({d[0]:+.4f},{d[1]:+.4f},{d[2]:+.4f})  {ga:.4f} {ma:.4f}")

    # Position drift fits.
    print("\n=== position linear fit pos = pos0 + i*step ===")
    gpos = {i: gl[i]["pos"] for i in gl}
    mpos = {i: mt[i]["eval"] for i in mt}
    for name, data in (("GL g_dataPos", gpos), ("MT evalPoint", mpos)):
        fit = fit_trace(data, args.lo, args.hi)
        if fit is None:
            print(f"{name}: no data in window")
            continue
        coef, resid, idx = fit
        print(f"{name} [i={idx[0]}..{idx[-1]}]:")
        print(f"  pos0=({coef[0,0]:.8f}, {coef[0,1]:.8f}, {coef[0,2]:.8f})")
        print(f"  step=({coef[1,0]:.8e}, {coef[1,1]:.8e}, {coef[1,2]:.8e})")
        print(f"  max|resid|={resid:.2e}")

    gfit = fit_trace(gpos, args.lo, args.hi)
    mfit = fit_trace(mpos, args.lo, args.hi)
    if gfit and mfit:
        gs, ms = gfit[0][1], mfit[0][1]
        rel = tuple((abs(gs[j]) - abs(ms[j])) / abs(ms[j]) * 100.0 for j in range(3))
        print(f"\nrelative |step| difference (GL-MT)/MT per axis: "
              f"x {rel[0]:+.4f}%  y {rel[1]:+.4f}%  z {rel[2]:+.4f}%")
        for i in (0, 49, 99, 143, 144, max(common)):
            if i in gpos and i in mpos:
                d = tuple(gpos[i][j] - mpos[i][j] for j in range(3))
                print(f"drift GL-MT at i={i:3d}: ({d[0]:+.1e}, {d[1]:+.1e}, {d[2]:+.1e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
