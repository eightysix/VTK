# divergent_tail_repro

Minimal self-contained reproduction (no VTK) of the Metal-vs-OpenGL
**divergent-tail deficit** in the j0 volume renderer at 2048x2048.

## What it shows

From the app measurements (`J0_FIXED_STEPS_DECOMP.md`):

| config     | GL ms/f | Metal ms/f | M/GL |
|------------|---------|------------|------|
| divergent  | 40.83   | 45.18      | 1.11 (Metal loses) |
| fixed(64)  | 19.28   | 16.93      | 0.88 (Metal wins)  |

This repro reproduces that signature on a 512x512x1794 synthetic volume
(470 MB, same size as the app's DICOM):

| config        | GL ms/f | Metal ms/f | M/GL |
|---------------|---------|------------|------|
| divergent     | 27.89   | 31.52      | 1.13 (Metal loses) |
| fixed(87)     | 6.92    | 6.11       | 0.88 (Metal wins)  |

The two modes march the **same rays with the same total sample budget**;
the only difference is the loop bound:

- **divergent**: bound is per-fragment from the ray-box exit (mean 87,
  max 283). Rays that miss the dense core run the full box; rays through
  it terminate early on opacity. The spread pins SIMT lanes.
- **fixed**: bound is capped at the frame mean (87), so no ray runs longer
  than the cap and the break-point spread is truncated.

Effective means (from the readback) match the app's too: divergent
83.4/83.5 (app: GL 86.5, Metal 81), fixed 60.0/60.1. Pixel coverage is
exact between backends (1912652 GL vs 1912192 Metal).

## Build

```
clang++ -std=c++17 -fobjc-arc -O2 -framework Metal -framework OpenGL \
        -framework Foundation -DGL_SILENCE_DEPRECATION \
        divergent_tail_repro.mm -o divergent_tail
```

## Run

```
./divergent_tail [rtSize=2048] [frames=30] [step=0.004] [volXY=512] \
                 [volZ=1794] [alphaMul=1.0] [fixedOverride=0]
```

M/GL > 1 means Metal loses. At 2048, divergent is ~1.10-1.14 (Metal loses)
and fixed is ~0.83-0.89 (Metal wins). At 1024 the gap shrinks but does not
fully tie (divergent ~1.09 vs the app's 0.97); the synthetic bimodal
distribution is more extreme than the real DICOM data.

## Key controls

- **alphaMul**: opacity-curve steepness; sets where the data-dependent
  break fires. Keep it high enough that some rays break early and others
  run the full geometric length.
- **fixedOverride**: cap for the fixed mode. 0 = frame geometric mean
  (~87). At 128/87 fixed hovers ~0.9-1.08; at 32/16 Metal wins
  ~0.91-0.92.
- **Structure of the volume is the key**: `makeVolume()` puts a dense
  sphere in the center (CT tissue) and air (0) everywhere else. This makes
  the break-point distribution bimodal (~10 vs ~283), which is what pins
  SIMT lanes. **Plain uniform noise cannot reproduce the deficit** — with
  random values every ray breaks at roughly the same count, the march is
  only weakly divergent, and Metal wins both modes.

## Caveats

- GL runs via a legacy 4.1 core-profile context on the same M-series GPU;
  Metal uses the default device. Both render to a 2048^2 RGBA8 target and
  are measured with MTLCreateSystemDefaultDevice-style implicit swap /
  glFinish after each frame.
- The fixed(87) mode still lets the opacity break fire (so its effective
  mean is 60, not 87) — matching the app's GL_STEPS behavior, where the
  clamp truncates the distribution but does not disable early termination.
- The repro's workload (28 ms/frame divergent) is lighter than the app's
  (40 ms/frame) because the synthetic volume's dense core is smaller than
  a real body slab; the ratio is what matters and it is stable across runs.