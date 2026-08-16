# minimal_gap - GL vs Metal volume-raycast microbenchmarks & harnesses

Standalone offscreen reproducers for the GL-vs-Metal volume-rendering gap in
the `vtkMetalGLVisualComparison` app. All three render 400x400 to an offscreen
FBO/texture, warm up 10 frames, and time N frames with a `glFinish` /
`WaitForCompletion` per frame. All consume `/tmp/app_gl_*.glsl` + the dumped
uniforms where noted, so run the app's shader/uniform dump step first.

## Files

| file | what it measures |
|---|---|
| `gl_gap.m` | GL bare-fetch march: the app's exact divergent camera rays, `GL_LINEAR` R8 512x512x1794 volume, one `texture()` + max-accumulate per iteration. The per-sample GL "raw fetch" cost. |
| `metal_gap.m` | Metal counterpart of `gl_gap.m`: same rays/volume (`R8Unorm`, trilinear), hardware sampler. The per-sample Metal "raw fetch" cost. |
| `gl_app_shader.m` | Runs the **app's exact composed GL shaders** (`/tmp/app_gl_frag.glsl`, `/tmp/app_gl_vert.glsl`) with the exact dumped uniforms and an R8 synthetic volume. The minimal reproduction of the app's GL render. |
| `gl_gap.ppm` / `metal_gap.ppm` / `gl_app_shader_iter.ppm` | sanity readback images (iteration counts / sample values), regenerated on every run. |

## Build

```sh
FLAGS="-O3 -DGL_SILENCE_DEPRECATION"
clang $FLAGS -framework AppKit -framework OpenGL gl_gap.m        -o gl_gap
clang $FLAGS -framework AppKit -framework OpenGL gl_app_shader.m -o gl_app_shader
clang $FLAGS -fobjc-arc -framework Metal -framework Foundation metal_gap.m -o metal_gap
```

(Objective-C / Metal on macOS. Host-code `-O` does not matter: the measured
region is GPU-bound. LSP/clangd diagnostics on `.m` files are a known
misconfiguration, not real errors.)

## Run

```sh
# GL bare fetch: [frames=100] [maxIter=8192] [noFetch=0] [useDepth=0] [fmt16=0]
./gl_gap 100 8192 0 0 0          # R8  (internal format prints 0x8229)
./gl_gap 100 8192 1 0 0          # noFetch (isolates fetch cost; ~1.4 ms)
./gl_gap 100 8192 0 0 1          # R16_SNORM variant (app does NOT use this)

# Metal bare fetch: [frames] [maxIter] [halfSampler] [depthMode] [compute] [lod0] [fastMath] [diag] [pipeline] [rtSize] [sampleDistMM] [dataMode]
./metal_gap 100 8192 0 0 0 0 1 0   # baseline: float sampler, depth attached+Store
./metal_gap 100 8192 1 0 0 0 1 0   # halfSampler=1: texture3d<half> + half accumulator
./metal_gap 100 8192 0 1 0 0 1 0   # depthMode=1: depth attached, MTLStoreActionDontCare
./metal_gap 100 8192 0 2 0 0 1 0   # depthMode=2: no depth attachment (matches gl_gap)
./metal_gap 100 8192 1 2 0 0 1 0   # half + no depth
./metal_gap 100 8192 0 0 1 0 1 0   # compute=1: same march as a compute kernel
./metal_gap 100 8192 0 0 0 1 1 0   # lod0=1: explicit level(0.0) sample (no effect)
./metal_gap 100 8192 0 2 0 0 0 0   # fastMath=0: ~8% FASTER (78-80 vs 85-87 ms)
./metal_gap 1 8192 0 0 0 0 1 1     # diag=1: shader writes rayDir*0.5+0.5 ->
#                                   #   metal_gap_diag.ppm; also prints invVP columns
./metal_gap 100 8192 0 2 0 0 1 0 8 # pipeline=8 (argv[9]): 8-way unroll, 8
#                                   #   independent in-flight fetches -> ~45.8 ms
#                                   #   (vs 90.0 baseline), 1.46x faster than GL
#                                   #   flipY=1 (67.1 ms); parity-exact.
# pipeline: 0=baseline 1=prefetch(loop-carried, SLOWER) 3=no-break >=2=N-way unroll
# All variants are statistically identical except fastMath=0 (~8% faster),
# compute (slower), and unroll (>=2, dramatically faster); see results below.
# diag is for ray-field parity checks.
# NOTE: the RT is BGRA8Unorm - PPM byte0=B(acc), byte1=G(iterHigh),
# byte2=R(iterLow). Iteration count = byte2 + 256*byte1.

# REPRODUCING THE APP'S COARSE-SD HIGH-RES LAG (2026-08):
# rtSize (argv[10]) + sampleDistMM (argv[11]) + dataMode (argv[12]) were added
# to both harnesses to reproduce the app's raw-path lag at 1024/2048 px with
# sample distances 2.0/4.0 mm. THE TRIGGER IS dataMode=1 (xorshift noise):
# synthetic gradient data hides the problem because every 4-slice slab is
# constant, so Metal's sampler cache stays warm. Real CT data has per-texel
# variation, which flips the ratio:
#   config             gradient M/GL   noise M/GL   GL degrade   Metal degrade
#   2048 x 2048, SD4   1.01            1.63-1.72    +15%         +86%
#   2048 x 2048, SD2   1.13            1.92         flat         +65%
#   1024 x 1024, SD4   0.96            1.68         flat         +80%
#   1024 x 1024, SD2   1.02            1.63         +12%         +77%
#   400 x 400,   SD0.5 1.22            1.56         +10%         +40%
# The 2048/SD4/noise cell (Metal ~78 ms vs GL ~45 ms = 1.72x) matches the app's
# raw-path 2048/SD4 (M 94 vs G 54 = 1.75x) almost exactly. Conclusion: the lag
# is a Metal 3D-texture sampling CACHE/LAYOUT deficit under coarse-stepped
# trilinear access (breaks the sample locality real data provides at fine SD),
# not a shader-scheduling deficit. GL barely reacts to data locality; Metal
# degrades ~65-86%. Layout hypothesis (slice-stack vs 3D Morton) tested below
# and REFUTED; remaining suspects: GL driver's internal 3D-texture tiling,
# sampler MSHR/cache-slice behavior, or driver-specific LOD/filter codegen.
./gl_gap 30 8192 0 0 0 1 2048 4.0 1     # GL: noise data, 2048 px, SD4 ~45 ms
./metal_gap 30 8192 0 2 0 0 1 0 0 2048 4.0 1   # Metal: noise, 2048 px, SD4 ~78 ms

# LAYOUT / TILING EXPERIMENT (layoutMode, argv[13] on metal_gap):
# 0 = MTLTextureType3D (app layout), 1 = texture2d_array of 1794 slices with
# manual Z trilinear (2 fetches + mix per sample, byte-identical rays:
# avgIter/meanB parity confirmed at 400 px). Tested the "3D Morton swizzle is
# cache-hostile for anisotropic NPOT volumes" hypothesis (GL may store 3D
# textures slice-oriented while Metal uses a fully interleaved 3D Morton curve).
# Result: REFUTED. At 2048 px noise, sliced is strictly worse than 3D:
#   SD4: 3D best 66.3 ms (half+unroll8) vs sliced best 73.1 ms (half+unroll2)
#   SD2: 3D best 92.1 ms (unroll8)        vs sliced best 110.7 ms (half+unroll2)
# Sliced issues 2 fetches/sample yet lands ~at 3D time (per-fetch ~2x more
# cache-efficient, but the doubled fetch count cancels it) -> Metal is bound by
# fetch/L2-miss throughput on noise data, and slice-oriented storage is NOT a
# fix. The GL-side advantage is not explainable by 3D-vs-2D-slice layout.
./metal_gap 30 8192 0 2 0 0 1 0 2 2048 4.0 1 1   # sliced pipeline2, SD4 ~74 ms
./metal_gap 30 8192 1 2 0 0 1 0 2 2048 4.0 1 1   # sliced half pipeline2 ~73 ms

# MECHANISM PINNED: CACHE / DRAM WORKING-SET CAPACITY (2026-08)
# Two more discriminators were run on the 2048 px noise lag cells:
#   filterMode (argv[14]): 0 = trilinear (app default), 1 = GL_NEAREST /
#       filter::nearest. The M/GL ratio PERSISTS with nearest (SD4 1.63x,
#       SD2 1.84x; both backends speed up ~35% but the gap does not collapse)
#       -> the trilinear 8-texel span is NOT the mechanism.
#   volDiv (argv[15]): 1/2/4 -> volume 512^3x1794 (470 MB) / 256^2x897 (58.7 MB)
#       / 128^2x448 (7.3 MB), SAME rays/SD/pixel grid (avgIter 36.2 identical,
#       verified by readback). Result at 2048/SD4 noise, linear:
#         volDiv=1 (470 MB): GL 44.1  Metal 78.9 (65.8 unroll8)  M/GL 1.49-1.80
#         volDiv=2 (58.7 MB): GL 5.65 Metal 8.39 (6.10 unroll8)  M/GL 1.08-1.48
#         volDiv=4 (7.3 MB):  GL 4.23 Metal 4.01                M/GL 0.95 (win)
#   The same ~151M samples cost Metal 79 ms at full footprint but 4-8 ms at
#   cache-sized footprint -> the lag is the working set exceeding the SoC's
#   shared cache (M2 SLC ~32-64 MB); Metal's 3D-texture read path degrades
#   ~9x as the footprint grows past capacity, GL only ~7x. NOT shader
#   scheduling (unroll floors), NOT layout (2D-array worse), NOT compression
#   (gap appears on incompressible noise = the opposite of compression),
#   NOT the trilinear span (nearest keeps ~1.5x).
#   Renderer implication: splitting the volume into cache-sized depth slabs
#   (K passes, front-to-back composite) shrinks each pass's working set toward
#   the volDiv=2/4 regime, which is exactly where Metal beats GL.
./metal_gap 30 8192 0 2 0 0 1 0 8 2048 4.0 1 0 1 1   # 3D, nearest, unroll8, SD4
./metal_gap 30 8192 0 2 0 0 1 0 0 2048 4.0 1 0 0 2   # 3D, linear, volDiv=2

# SLAB RENDERING VALIDATED - THE FIX (2026-08)
# Depth-slicing the march (argv[16]=numSlabs, argv[17]=slabIndex; argv[18]=1
# = t-clamped slab pass, real slab-rendering model) collapses the gap:
#   2048/SD4 noise, 8 passes, per-pass t-range = 1/8 of the depth (each pass
#   does 1/8 of the samples; sum of per-pass avgIter 36.7 == full 36.2):
#     Metal: 0.85/1.02/0.86/1.06/1.19/1.24/1.29/1.11 = 8.6 ms total (was 79)
#     GL:    1.13/1.18/1.23/1.32/1.29/1.40/1.54/1.49 = 10.6 ms total (was 44)
#   Metal slab-render wins over GL single-pass by ~5x and over GL slab by 1.2x.
#   Once the per-pass working set fits cache, Metal is FASTER than GL. This is
#   the mechanism (cache/DRAM working-set capacity) turned into a win: split
#   the volume depth into cache-sized slabs, render K front-to-back passes.
#   NOTE the Z-position effect (slab[0] fastest, slab[6] slowest for both
#   backends - swizzle/memory-order property, not a backend difference).
./metal_gap 30 8192 0 2 0 0 1 0 0 2048 4.0 1 0 0 1 8 0 1  # 3D slabT pass 0/8
./gl_gap 30 8192 0 0 0 1 2048 4.0 1 0 1 8 0 1              # GL slabT pass 0/8

# MULTI-PASS ACCUMULATION PROTOTYPE (2026-08, argv[19]=maccum)
# Renders numSlabs slabT passes per frame into ONE RT (pass 0 clears, passes
# 1+ loadAction=Load), accumulated via MTLBlendOperationMax, one commit/frame.
# Each pass's tStart is aligned UP to the global sample lattice (tStart + k*step,
# k = ceil((tlo - tStart)/stepSize)) so the union of slab sample positions
# equals the single-pass lattice exactly.
#   Correctness: numSlabs=1 (no split, just acc-out + blend) = pixel-exact vs
#   single-pass (0 mismatches). 8-slab split vs single-pass: 0.13% of marched
#   pixels differ by <=5/255 (slab-boundary lattice-epsilon artifact only).
#   2048/SD4 noise: single-pass 81.5 ms -> 8-slab maccum 13.5 ms (6.0x), and
#   vs GL single-pass 44 ms = 3.3x WIN. 2048/SD2: 18.5 ms vs GL 61 ms = 3.3x.
#   pipeline=8 (unroll) does NOT stack (16.3 ms): slab already fixes the cache
#   problem, unroll's win was single-pass-only. The ~13.5ms vs 8.6ms projection
#   gap is the lattice alignment + per-pass overhead; the app can recover it
#   (compute the slab clamp from uniforms instead of one PSO per slab).
./metal_gap 30 8192 0 2 0 0 1 0 0 2048 4.0 1 0 0 1 8 0 1 1  # maccum 8-slab SD4
./metal_gap 30 8192 0 2 0 0 1 0 8 2048 4.0 1 0 0 1 8 0 1 1  # maccum 8-slab + unroll8

# App-shader harness: [frames] [iterMode]
# iterMode=0: color output + frame timing
./gl_app_shader 30
# iterMode=1: encodes R=g_currentT/4096, G=g_terminatePointMax/2048,
#             prints avg iteration/termination counts, writes
#             gl_app_shader_iter.ppm
./gl_app_shader 3 1
```

## App-side dump pipeline (produces the harness inputs)

Run the real app once per dump (each dump is one-time per process):

```sh
APP=build_macos_metal/bin/vtkMetalGLVisualComparison
DICOM=/path/to/IMRToraceAddome
SCENE="--scene DICOMVolume --dicom $DICOM --frames 2"

env VTK_METAL_TEST_SAMPLE_DISTANCE=0.5 VTK_METAL_TEST_JITTER=1 \
    VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=0 \
    $APP $SCENE --backend gl          # writes /tmp/app_gl_frag.glsl, /tmp/app_gl_vert.glsl
env VTK_METAL_TEST_DUMP_UNIFORMS=1 $APP $SCENE --backend gl   # writes /tmp/app_gl_uniforms.txt
```

The harness reads `/tmp/app_gl_frag.glsl` + `/tmp/app_gl_vert.glsl` (the
uniforms are compiled in as constants, copied from the dump). Re-dump the
shaders whenever the scene/mapper config changes.

## Notes

- Reference GL numbers at the bench config (`sampleDistance 0.5`, camera
  (-637,-406,1831) vs box 426x426x717, jitter/minmax/accel on): app 48.97 ms,
  harness 28.7 ms, `gl_gap` 62.1 ms (all ~46 M samples; harness ~34 M at
  484 avg).
- The app's volume is **R8** (verified at runtime: requested/realized GL_R8,
  Rbits=8). Do not use the `fmt16` variant as an app baseline.
- `gl_gap` is *slower* per sample (1.35 ns) than the full app shader
  (1.06 ns) - a lone volume-fetch loop schedules badly on this driver
  (same artifact as section 9 in `PERFORMANCE_INVESTIGATION.md`). Use
  `gl_app_shader` as the app-side GL baseline; use `gl_gap`/`metal_gap` only
  for the same-pattern GL-vs-Metal ratio.

## Metal optimization experiment results (2026-08, M2 MBA, interleaved A/B)

All runs: fresh, short (30 timed frames), interleaved to cancel thermal drift.
Metal geometry was identical across every fragment variant (avgIter 287.8 over
69,861 marched pixels = 45.6 M samples - **exactly equal to GL/fp64**, see the
parity note below).

**Orientation-matched GL reference (`flipY=1`)**: GL's default window origin is
bottom-left, Metal's top-left, and `gl_gap` row order matters for GL's absolute
time (~6 ms, ~10%): flipY=0 is 62.1 ms, flipY=1 is 67.9 ms (per-pixel
identical to Metal's orientation, 159,990/160,000 pixels). The fair
app-normalized comparison is **GL flipY=1 ~67.9 ms vs Metal ~86.3 ms =
~1.27x / ~18 ms**, not the ~1.4x implied by the old flipY=0 baseline.

| variant | avg frame (rounds) |
|---|---|
| fragment, float sampler, depth+Store (legacy baseline) | 84.4 / 85.1 / 85.9 / 86.6 |
| fragment, **half sampler** + half acc | 83.1 / 84.2 / 84.2 / 85.4 / 85.9 |
| fragment, float, depth **DontCare** | 86.1 / 86.9 |
| fragment, float, **no depth attachment** | 85.3 / 86.1 / 86.4 / 86.9 |
| fragment, float, **LOD-0 sample** (`level(0.0)`) | 86.2 / 86.4 (no effect) |
| fragment, float, **`fastMathEnabled=NO`** | **77.8 / 78.7 / 79.4 / 79.6 / 79.9** |
| **compute kernel**, float | 97.9 / 98.0 / 99.2 |
| **compute kernel**, half | 95.3 / 95.8 / 96.7 |

**Non-power-of-two depth test (does NOT explain the gap):** raising `kD` from
1794 to 2048 (PoT) keeps the ratio ~identical: GL flipY=1 78.6 ms vs Metal
100.2 ms = **1.27x**, the same 1.27x as at 1794. Both backends scale ~linearly
with the +14% footprint (GL +16%, Metal +16%), so tiling/PoT padding is not
the source of the ratio.

**ILP / latency-hiding (the gap is a scheduling deficit — Metal wins with
unrolling):** `pipeline` arg selects the inner-loop structure (argv[9];
>=2 = N-way unroll). Identical per-pixel work across all variants (avgIter
287.8; unroll8 iter differs from baseline on only 1,531/160,000 silhouette
pixels by ±1, sample values identical). 30-frame interleaved rounds:

| variant | avg ms | ns/sample |
|---|---|---|
| GL flipY=1 (linear) | 67.1 | 1.47 |
| Metal baseline (0) | 90.0 | 1.97 |
| Metal prefetch-2 (1, loop-carried) | 89.7 | 1.97 |
| Metal 2x unroll | 72.1 | 1.58 |
| Metal **4x unroll** | 58.5 | 1.28 |
| Metal **8x unroll** | **45.8** | **1.00** |
| Metal 16x unroll | 51.6 | 1.13 |

The monotonic speedup to 8x (then a register-pressure regression at 16x)
confirms the harness shader was **TPU-latency-bound, not bandwidth- or
ALU-bound**: N independent 3-D-linear fetches in flight hide the sample issue
latency. Metal's straight loop left the latency exposed; GL's compiler already
software-pipelines it. The prefetch-with-loop-carried-dependency pattern
(variant 1) is *slower* than baseline and must be avoided. 8x-unrolled Metal
beats GL by 1.46x on identical work (1.00 vs 1.47 ns/sample).

### App transfer: the 8x unroll DOES transfer (PERFORMANCE_INVESTIGATION.md section 15)

The app's `marchVolumeUnified` loop already carried a loop-carried
prefetch-ahead-1 (the pattern 1 trap) plus min-max skip and gradient ILP, so a
literal 8x unroll of the full body was rejected. Instead a dedicated
`fc_marchVariant` 6 (8x chunked unroll: 8 independent `sampleVolumeScalar`
fetches per batch, latched exits, no loop-carried deps) and 7 (4x) were added,
guarded to the simplified composite config. Env `VTK_METAL_TEST_MARCH_VARIANT`
bakes the variant into the feature mask so each experiment gets its own
pipeline.

DICOM app bench, 3 interleaved rounds (baseline/variant6/variant7 per round),
30 frames x 3 reps, M2 MBA, min-max accel off for GL parity (DICOM scene has no
`ShadeOn`):

| variant | mean ms/f |
|---|---|
| 0 (baseline loop) | 105.0 |
| 7 (4x) | 88.4 |
| **6 (8x)** | **81.2** |

8x is **1.29x faster** than baseline with **thresholded error 0.000** vs GL on
every variant (real GL+Metal comparison - the `--backend metal`-only run skips
the diff). Variant 6 is now the default `fc_marchVariant`; the env var still
overrides for A/B.

The app is still slower than GL though: same-scene GL is 49.3 ms vs Metal 79-81
ms (~1.6x), even though the minimum-gap repro has Metal 1.46x faster. That
residual app gap is the next investigation - the harness models the bare-fetch
march, not pipeline setup / per-block dispatch / offscreen RT handling / the
app's actual geometry path.

Conclusions, addressing the four hypotheses:

1. **`texture3d<half>` sampling is not the gap.** Half vs float is within noise
   (~1-2%) in both fragment and compute. GL's 16-bit fast paths do not explain
   the GL/Metal ratio.
2. **Depth store action has no effect.** The pass writes no depth, so TBDR
   never touches the depth tile; DontCare vs Store vs no attachment is
   identical (the asymmetry between `gl_gap` no-depth and `metal_gap`
   depth+Store was real but *measurement-irrelevant*).
3. **Compute is *slower*, not faster**: ~13% worse than the fragment pass
   (98 vs 86 ms). On Apple's TBDR GPUs the fragment machinery is a win for a
   fullscreen march; a compute rewrite would be a regression here.
4. **`fastMathEnabled=NO` is ~8% faster in the harness** (78-80 ms vs 85-87 ms,
   five interleaved rounds), but **does NOT transfer to the full app shader**:
   the app A/B (`VTK_METAL_TEST_FAST_MATH=0` vs unset, 5 interleaved
   30-frame/3-rep benches) was 99.2-102.1 ms vs 100.4-101.1 ms - within noise
   and slightly slower on average. The ~8% is specific to the bare-fetch
   shader's tight fetch loop, not a general Metal-side lever.

Net: the GL/Metal bare-fetch ratio (~1.3x) is intrinsic to Metal's per-sample
fetch/loop throughput on this device, not to the fragment pipeline, depth
handling, precision, or the dispatch model. The harness-side
`fastMathEnabled=NO` gain (~8%) does not survive in the full app shader.

### Harness parity (readback channel bug, now 1:1)

Earlier notes claimed Metal over-runs per-ray by 4-7% (avgIter 306.4/701.7 vs
GL 287.8/659.2) and under-samples (meanB 0.210 vs 0.283). Both were artifacts
of a **readback bug in `metal_gap.m`**: the render target is `BGRA8Unorm`, so
memory byte order is B,G,R,A while the shader writes
`float4(R=iterLow, G=iterHigh, B=acc)`. The stats loop read byte0 as the
iteration low byte (it is actually the B/acc channel). GL's `RGBA8` readback
has no such trap.

With the corrected decode (`iter = byte2 + 256*byte1`) and the `diag` mode
(`diag=1`; shader writes `rayDir*0.5+0.5`), matching 1-frame runs confirm
**exactly 1:1 parity**:

- avgIter **287.8** on both sides (= fp64 287.8 / 659.2 per nonzero pixel);
  footprint 69,861 nonzero pixels both.
- per-pixel iteration counts: **159,998/160,000 identical** (GL bottom-first
  vs Metal top-first row flip applied); the 2 outliers differ by exactly 1
  (silhouette boundary).
- per-pixel sample values on marched pixels: identical (mean |d| = 0.00).

The residual ~30 ms (57.8 ms GL vs 84-87 ms Metal) is pure GPU execution cost
for identical work - no harness artifact and no algorithm difference.
