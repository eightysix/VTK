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

M/GL > 1 means Metal loses. At 2048, divergent is ~1.07-1.15 (Metal loses)
and fixed is ~1.04-1.07 (near-tie; see "Why fixed(87) flipped" below — the
early 0.88 reading was a GL-side measurement artifact). At 1024 divergent
is ~1.14-1.16.

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
| V17     | binned-4pass: 4 capped passes over last-frame done quartiles |
| V18     | manual trilinear via 8x `read()` (diagnostic)        |
| V19     | half-precision composite (diagnostic)                |
| V20     | binned-static: no history write, median split for fixed |

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

## RETRACTION: the binned-pass "win" was a measurement artifact

An earlier version of this section claimed V17 fixed the divergent gap at
M/GL 0.64-0.65. That claim was WRONG and is retracted. Two defects conspired:

1. **Cross-mode bin contamination.** The harness alternates divergent/fixed
   rounds within one variant. The fixed call's bucket-build overwrote the
   done-history texture, so later divergent rounds assigned buckets from
   FIXED-mode dones (all <= 87): a pixel with true done 283 stored as 60 was
   capped at 65 and kept reporting 65 — truncated marches, ~3x less work, and
   the per-variant average mixed one clean round (~31 ms) with contaminated
   ones (~10 ms) to produce the fake 17.5 ms. Parity was only read back in
   round 0, so it stayed green while timing was garbage. Worse, the warmup
   cannot be trusted to heal stale bins: each frame writes its own (already
   capped) done, so truncation is self-sustaining. The fix — rebuild bins on
   every mode switch, verified by dumping bin populations pre-timing — is in
   the code now.
2. **The corrected result kills the idea anyway.** With clean bins the
   binned variants measure: V17 divergent 31.09 (1.15), V20 30.29 (1.12),
   vs baseline 31.18 (1.11). Per-pass GPU spans explain why: pass0
   (done<=41) 0.66 ms, pass1 1.15, pass2 2.37, pass3 (done>114, uncapped)
   **26.9 ms alone**. `discard_fragment()` frees nothing — discarded lanes
   occupy their warp slots until the warp retires, and long-ray pixels are
   scattered across the screen, so nearly every 32-lane warp in the uncapped
   pass still contains one and runs the full ~283 steps. Binning partitions
   work but cannot re-GROUP it; the scattered tail sets the cost of every
   pass it touches. Beating it would require physically reordering fragments
   by trip count (sorted index buffers + indirect draws), not pass caps.

What survives: V20's fixed-mode median split ties GL (5.55 vs 5.53,
M/GL 1.00) because fixed mode has no long tail to scatter. The honest
harness state at 2048/SD4 remains: divergent M/GL ~1.11-1.15, fixed
~1.03-1.05.

## Why GL stays ahead, and what was falsified trying

With optContents=YES forbidden (a tax on incompressible data per
`../lag_repro/README.md`, and an app image-accuracy constraint), every
shader-level lever has now been measured:

| lever | result |
|---|---|
| loop shapes V0-V5 (uniform header, group exit, kMax const, chunks) | identical timing |
| batched march 8/16/32/48 (V13-16) | monotonically worse at 2048 |
| binned passes, MRT + static variants (V17/V20) | no effect (warp physics above) |
| manual trilinear via read() (V18) | 2.5x WORSE — sampler path is optimal |
| half-precision composite (V19) | no change |
| done-sorted tile quads, same order both APIs (V21) | +17% vertex/small-primitive tax, ratio unmoved — submission order does not steer warp composition |
| compute march over done-sorted indices, warp-homogeneous by construction (V22) | 2.8x WORSE — Apple's fragment/sampler engine crushes raw compute for this workload even with perfect grouping |
| NPOT depth 1794 vs 2048/1024 | ratio unchanged |
| axis permutation (march along x/y/z, argv[15]) | ratio unchanged — deficit is axis-independent |
| MSL languageVersion 3.2 + fast math | no change |

Two localization facts pin the residual: (a) with an 8 MB cache-resident
volume the DIVERGENT gap vanishes (Metal wins 0.95 at 2048, 0.79 at 1024)
— the divergent deficit is DRAM-path behavior under divergence, not loop
structure; (b) the FIXED deficit persists cache-resident (~1.04) and grows
as RT shrinks (1.13 at 512, 1.18 at 1024) — a small per-sample/scheduling
efficiency difference that no legal MSL rewrite has moved. GL samples its
own internally-tiled storage; Metal with allowGPUOptimizedContents=NO
marches an uncompressed private layout.

## V23: 2D-array two-tap march — parity with GL, same image

Replacing the single 3D trilinear tap with TWO explicit bilinear taps into
a `texture2D_array` (neighboring z slices, lerped in registers — identical
math order compiled for both APIs) removes whatever advantage GL's 3D
sampler path held:

```
                    divergent GL  Metal  M/GL   |  fixed GL  Metal  M/GL
V0  3D trilinear      28.25  29.60  1.05  |   5.60   5.68  1.01   (2048/SD4)
V23 arr two-tap       31.71  31.76  1.00  |   6.75   6.35  0.94   (2048/SD4)
V0  3D trilinear      18.13  19.41  1.07  |   4.07   4.66  1.14   (1024/SD4)
V23 arr two-tap       20.59  20.75  1.01  |   4.47   4.83  1.08   (1024/SD4)
V0  3D trilinear      63.55  73.20  1.15  |  23.61  22.73  0.96   (2048/SD0.5)
V23 arr two-tap       80.23  77.46  0.97  |  34.91  29.43  0.84   (2048/SD0.5)
```

Parity is exact in every cell (cov/mean match the 3D signature: GL-array
reads 1912652/83.4 — bit-identical to GL-3D — so the representation is
image-neutral, not a quality reduction).

The honest caveat: the array path is slower in ABSOLUTE terms for both
backends (GL +13%, Metal +7% at 2048/SD4). It satisfies M/GL <= 1 with
equal algorithms and equal data, but it converges partly by costing GL
more than Metal. Remaining >1 cells: 1024 fixed (1.08) and 1024 divergent
(1.01) — the small-frame scheduling tax documented above, which no
variant has ever moved.

## Why fixed(87) flipped from 0.88 (Metal wins) to ~1.05

The first published repro table showed fixed(87) at 6.92 GL / 6.11 Metal
(0.88). Two separate effects, neither a timing-methodology change:

1. **Honest-A/B correction (code change, before the sweep commit).** Early
   builds selected the fetch mode with a runtime `uUseLod` uniform +
   ternary inside ONE GL program. That dynamic branch cost GL ~2.3 ms
   divergent / ~1.3 ms fixed while Metal (separately compiled PSOs) paid
   nothing. Baking two GL programs (implicit `texture()` for V0,
   `textureLod(...,0.0)` otherwise) restored GL fixed to ~5.9 ms — most
   of the original "Metal wins fixed" margin was this GL handicap.
2. **Session drift (no code change).** The variant-sweep session read GL
   fixed 6.96 / Metal 5.77 (0.83) with byte-identical GL code to today's
   5.47-5.60 / 5.82-5.96 (1.04-1.07). Rebuilding the old commit's binary
   and running both back-to-back in one session ties them (old 5.60/5.85
   = 1.04, new 5.59/5.96 = 1.07), so the remaining shift is machine/
   driver state (clocks, shader cache), not code. Treat sub-10%
   fixed-mode ratios as session-sensitive; A/B in the same session.

**Timing is symmetric**: both backends are GPU-timed around the identical
single fullscreen draw (Metal `GPUStartTime/GPUEndTime` per command
buffer, GL `GL_TIME_ELAPSED` per frame), neither timed region contains a
clear or load, parity readback is outside timing for both, and the
wall-clock+Clear harness (`harness=1`) produces the same ratios. The
remaining asymmetry is the thing under test: legacy GLSL driver vs Metal
on the same GPU.

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