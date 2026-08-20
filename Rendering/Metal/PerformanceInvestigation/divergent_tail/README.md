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
                 [optContents=0] [kMax=288] [harness=0]
```

M/GL > 1 means Metal loses. At 2048, divergent is ~1.10-1.15 (Metal loses)
and fixed is ~0.75-0.98 (Metal wins). At 1024 the gap shrinks but does not
fully tie (divergent ~1.09 vs the app's 0.97); the synthetic bimodal
distribution is more extreme than the real DICOM data.

The binary runs a 6-variant sweep on the Metal side (each is a separately
compiled PSO, all producing bit-identical pixel coverage and mean steps —
parity is exact across every variant):

| variant | Metal shader shape                                  |
|---------|-----------------------------------------------------|
| V0      | baseline: implicit `sample()`, per-lane `for (i<steps)` + `break` |
| V1      | explicit `level(0.0)` sample (no implicit gradients) |
| V2      | `limit = simd_max(steps)` header, predicated body   |
| V3      | V2 + `if (!simd_any(alive)) break` whole-group exit |
| V4      | compile-time ceiling `kMax=288` function constant, `[[unroll(1)]]` |
| V5      | 32-step chunks, per-chunk `simd_max` bound + group exit |

GL is compared honestly: two separately compiled fragment programs (implicit
`texture()` for V0, baked-in `textureLod(...,0.0)` for V1-V5) so the fetch
mode is not polluted by a runtime uniform branch.

## Evaluation (variant sweep, 2048, optContents=NO)

Both measurement harnesses agree (GPU timestamps + DontCare vs wall-clock +
Clear): **no Metal loop-shape variant moves the divergent gap**. Metal is
pinned at ~31.2-31.5 ms/f across V0-V5 while GL sits at ~27.2-27.8 ms/f,
M/GL 1.12-1.15. The loop shape MSL emits is NOT the cause of the deficit:

```
                  divergent GL  Metal  M/GL   |  fixed GL  Metal  M/GL
V0 baseline         27.79  31.19  1.12  |   6.96   5.77  0.83
V1 explicit lod     27.13  31.20  1.15  |   5.89   5.80  0.98
V2 simd_max header  27.21  31.28  1.15  |   6.91   5.84  0.84
V3 +group exit      27.13  31.23  1.15  |   6.82   5.89  0.86
V4 kMax=288 const   27.18  31.16  1.15  |   7.40   5.81  0.78
V5 chunked reconv   27.23  31.23  1.15  |   6.51   5.94  0.91
```

Findings:

1. **The user's hypothesis is falsified.** "MSL loses to Apple's GLSL
   frontend on a high-max, per-lane trip count" — no. Every loop-shape
   rewrite (uniform `simd_max` header, predicated body, whole-group exit,
   compile-time ceiling, chunked reconvergence) produces identical Metal
   timing. The MSL frontend is not emitting a worse CFG; if it were, at
   least one of V2/V3/V4/V5 would have changed the time.
2. **The harness was not eating the gap.** GPU timestamps + DontCare vs
   wall-clock + Clear: same result (1.08-1.15 divergent both ways). The
   earlier 1.03 "improvement" was a measurement artifact — the runtime
   `uUseLod` uniform + ternary slowed GL by ~2.3 ms; baking the fetch into
   separate programs restored GL to ~27.2 ms and the gap to 1.15.
3. **`allowGPUOptimizedContents=YES` moves the gap (5-10%) but is an app
   constraint.** With YES, Metal divergent drops ~31.2 -> 28.7 ms and the
   gap collapses to M/GL 1.04-1.06; fixed drops 5.8 -> 3.9 ms (M/GL 0.6).
   The app forces NO for image-accuracy reasons, so the app's real gap
   (1.05-1.11) is the optContents=NO regime.
4. **Explicit LOD helps GL, not Metal.** V1's GL is slightly faster than
   V0's (27.13 vs 27.79) but Metal is flat (31.20 vs 31.19) — implicit
   derivatives are not the Metal-side footgun this shader.
5. **The deficit is a property of GPU execution of the long-tailed
   divergent loop**, not of MSL codegen: the same device runs the bounded
   fixed mode faster than GL (0.75-0.98) but the unbounded divergent mode
   slower (1.12-1.15), and no legal rewrite of the loop changes that. It
   tracks the per-lane trip-count distribution (mean 83.4, max 283) and
   resolution, exactly as the app's decomposition found.

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
- **optContents**: `allowGPUOptimizedContents` on the Metal volume texture.
  0 = NO (matches the app's image-accuracy constraint), 1 = YES. This is the
  only knob that measurably moves the divergent gap (~10%); it is the main
  residual after the loop-shape sweep, and the app cannot flip it.

## Caveats

- GL runs via a legacy 4.1 core-profile context on the same M-series GPU;
  Metal uses the default device. Both render to a 2048^2 RGBA8 target and
  are measured with GPU timestamps (Metal `GPUStartTime/GPUEndTime`, GL
  `GL_TIME_ELAPSED`), or wall clock + Clear when harness=1.
- The fixed(87) mode still lets the opacity break fire (so its effective
  mean is 60, not 87) — matching the app's GL_STEPS behavior, where the
  clamp truncates the distribution but does not disable early termination.
- The repro's workload (28 ms/frame divergent) is lighter than the app's
  (40 ms/frame) because the synthetic volume's dense core is smaller than
  a real body slab; the ratio is what matters and it is stable across runs.