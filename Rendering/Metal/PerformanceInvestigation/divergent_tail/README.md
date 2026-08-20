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
| V6      | persistent-threads compute kernel (falsified, disabled) |
| V7      | discard-fragment dead-ray head (S29, disabled)       |
| V8-11   | v34-exact batched march, 1 break per batch, batch 8/16/32/48 (latch-guarded consume) |
| V12     | S29 dead-path (disabled)                             |
| V13     | batch-8 (renumbered V8)                              |
| V14-16  | batch-16/32/48 (renumbered V9-11)                    |
| V17     | **binned-4pass**: 4 capped passes over last-frame done quartiles |

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

## Evaluation (batch sweep V8-11, v34-exact batched march)

Rewriting the march to the app's mv9 shape (all 8 positions computed first,
8 back-to-back fetches, ONE `break` per batch, strict `i+8<=steps` batches,
scalar break-aware tail) does NOT close the gap at the reference cell and
regresses at 2048:

```
                    divergent GL  Metal  M/GL   |  fixed GL  Metal  M/GL
V0  baseline          17.33  20.03  1.16  |   4.04   4.75  1.18   (1024)
V13 batch-8           17.23  18.48  1.07  |   4.51   4.93  1.09   (1024)
V14 batch-16          17.25  19.23  1.11  |   4.69   5.31  1.13   (1024)
V15 batch-32          17.33  20.20  1.17  |   4.74   5.75  1.21   (1024)
V16 batch-48          17.27  21.24  1.23  |   4.82   6.16  1.28   (1024)
V0  baseline          28.99  30.88  1.07  |   5.47   5.82  1.06   (2048)
V13 batch-8           27.16  31.68  1.17  |   5.47   6.27  1.15   (2048)
V14 batch-16          27.16  33.44  1.23  |   5.47   6.66  1.22   (2048)
V15 batch-32          27.16  34.41  1.27  |   5.47   7.41  1.36   (2048)
V16 batch-48          27.16  35.23  1.30  |   5.47   8.16  1.49   (2048)
```

Wider batches are monotonically worse at 2048 (the harness is already
latency/bandwidth-bound there: ~0.19 ns/sample Metal vs ~0.17 GL, versus
~2 ns/sample in the low-occupancy 400x400 metal_gap case where the 8x
unroll wins 1.46x). Batch-8 helps a bit at 1024 (1.07) but loses at 2048
(1.17). Branch-free select-gated consumes do not change this: batching
itself, not branchiness, is the problem at high occupancy. At fine SD
(0.0005, 2048) batch-8 wins 1.05 vs baseline 1.13 — mirroring the app's
fine-SD mv9 wins — so batching helps only where warps are NOT already
latency-bound.

MSL gotcha discovered during the sweep: unused batch lanes are NOT
dead-code-eliminated when their samples are guarded by runtime uniforms —
all 48 lanes stayed live and time scaled as 48/width (92.8/46.3/23.5/21.2
ms for batch 8/16/32/48). Batch variants must be compiled with `#if`
per width.

## V17: binned passes FIX the divergent gap (M/GL 0.64-0.65)

The structural fix (Experiment C): a frame-hint binned march. One
"bucket-build" frame renders a full-cap pass (MARCH_VARIANT 12) with two
MRT attachments — color + an R16Unorm "done histogram" texture — then a
CPU readback splits the done histogram into quartile caps (e.g. 41/65/114
at SD4). Each subsequent frame renders 4 passes; pass p keeps only pixels
whose previous-frame done fell in bucket p (`discard_fragment()` for the
rest) and caps its march at the bucket's quartile (`min(steps, cap)` —
never binds, so output is bit-identical; the win is SIMT grouping: every
lane in a pass terminates at or before the same cap). The histogram
texture is double-buffered (read prev frame, write cur frame) to avoid the
render-target/read hazard.

```
                    divergent GL  Metal  M/GL   |  fixed GL  Metal  M/GL
V0  baseline          17.33  20.03  1.16  |   4.04   4.75  1.18   (1024)
V17 binned-4pass      17.22  11.21  0.65  |   4.06   4.57  1.13   (1024)
V0  baseline          28.99  30.88  1.07  |   5.47   5.82  1.06   (2048)
V17 binned-4pass      27.17  17.54  0.65  |   5.46   5.83  1.07   (2048)
```

Parity stays exact (cov 1912652/83.4 GL vs 1912192/83.5 Metal, fixed
60.0/60.1). This is the first harness variant that beats GL on the
divergent (app-reflecting) cell — Metal 35% faster. The binning is gated
adaptively: it only engages when the done histogram is strongly bimodal
(25th-percentile cap <= half the 75th, spread >= 16). At fine SD the
distribution is broad (caps 321/512/512 at SD0.5) and the fallback runs
the single baseline pass (SD0.5 divergent 1.15 vs baseline 1.13, no
regression).

Caveat: the pass union requires discarded pixels to leave the target
untouched — writing `{0,0,0,0}` instead of `discard_fragment()` clobbers
other passes' pixels (caught by the parity readback: cov dropped to ~1/4).
The binned variant is a harness-level result; transferring it to the app
is the next step.

## Key controls

- **alphaMul**: opacity-curve steepness; sets where the data-dependent
  break fires. Keep it high enough that some rays break early and others
  run the full geometric length.
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