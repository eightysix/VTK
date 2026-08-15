# Volume ray-cast mapper: performance regression analysis

SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
SPDX-License-Identifier: BSD-3-Clause

## Summary

The Metal `vtkMetalGPUVolumeRayCastMapper` (measured through the
`TestMetalGLVisualComparison --bench` harness on the `VolumeRayCast` scene)
regressed from **0.60 ms/frame to 1.04 ms/frame** (~73% slower) across the
commit range `fd65034cf4` (early Metal volume work) to `d8a31e4aad` (branch
`metal-volume-parity-essential` tip). Git bisecting with the benchmark isolates
the regression to **two commits**:

| Commit | Change | ms/frame (Metal) | Delta |
|---|---|---|---|
| `b110f8e450` | render independent multi-component volumes in GPU ray-cast mapper | 0.60 → 0.76 | **+27%** |
| `a6bec091bb` | backport the 22 essential GL-parity fixes from `metal-ios` | 0.85 → 1.10 | **+29%** |

Both commits add substantial code to the per-sample ray-cast shader loop. The
interesting part is that **neither commit changes the behaviour of the
`VolumeRayCast` benchmark scene** — it is single-component, so the independent
multi-component path is inactive, and the parity fixes are pure visual
correctness. The cost is the *presence* of the new code, not its execution.

## Benchmark methodology

All numbers come from the standalone dual-backend harness
`Rendering/Metal/Testing/Cxx/TestMetalGLVisualComparison.cxx`:

- `vtkMetalGLVisualComparison --bench --scene VolumeRayCast --frames 60 --reps 5`
- Scene: `vtkRTAnalyticSource` (single-component, 33³) rendered 400×400 with a
  shaded transfer function.
- Both backends render into offscreen targets
  (`SetOffScreenRendering(true)`, Metal
  `VTK_METAL_ENABLE_OFFSCREEN_TARGET`) so the CAMetalLayer display-refresh
  pacing does not cap the timed loop; this makes the two backends symmetric.
- Each timed frame nudges the camera by `Azimuth(0.1)` so every frame does real
  work, and both backends are synchronized inside the timed region
  (`WaitForCompletion` / `glFinish`) so the wall clock covers GPU time.
- Each commit is rebuilt and measured with `--reps 5`: a fresh window per run,
  reported as mean ± σ of the 5 per-run averages.
- Hardware: Apple silicon (arm64), macOS, Release build.

Note: single `--reps 1` runs are noisy — the GL/Metal ratio shifts run-to-run
(documented in the harness header) — so all measurements here use 60×5, where
σ is ≤ 0.02 ms and the classification of a commit as fast vs slow is reliable.

## Results

### Full measurement table (VolumeRayCast, 60 frames × 5 reps)

| Commit | Description | Metal ms/f ± σ | Metal fps | M/GL |
|---|---|---|---|---|
| `fd65034cf4` | run the Volume test suite under the Metal backend (baseline) | 0.60 ± 0.02 | 1660 | 0.47 |
| `0d2a89565c` | restore cell-to-point texel-center sampling | 0.56 ± 0.02 | 1797 | 0.44 |
| `a1106f1f33` | uniform-grid blanking and ghost arrays | 0.57 ± 0.01 | 1757 | 0.44 |
| `2d5eb6ef59` | honor image-data direction matrices | 0.60 ± 0.01 | 1656 | 0.47 |
| **`b110f8e450`** | **render independent multi-component volumes** | **0.76 ± 0.01** | 1312 | 0.59 |
| `9444ff933b` | apply gradient opacity independent of shading | 0.82 ± 0.01 | 1217 | 0.63 |
| `c6b8cc1fd5` | camera-inside near-plane clip for ray-cast parity | 0.85 ± 0.01 | 1178 | 0.66 |
| **`a6bec091bb`** | **backport 22 essential GL-parity fixes** | **1.10 ± 0.01** | 911 | 0.85 |
| `d8a31e4aad` (HEAD) | add `--no-tests` flag to build script | 1.04 ± 0.01 | 964 | 0.91 |
| HEAD + fix (`fc_independentComponents`) | single-component pipeline specialization | **0.79 ± 0.01** | 1272 | 0.67 |
| HEAD + Fix 2 (7 feature-flag constants) | bake remaining feature-flag branches | 0.70 ± 0.02 | ~1428 | ~0.55 |
| HEAD + Fix 3 (cropping/blanking + uniform cleanup) | bake cropping/blanking, drop redundant re-checks | 0.64 ± 0.01 | ~1545 | ~0.50 |

GL stays essentially flat throughout (≈ 1.14–1.30 ms), so the drift is entirely
in the Metal backend. For reference, the earlier 30-frames×3-reps runs gave
1.11 ± 0.01 at HEAD and 0.92 ± 0.02 at `c6b8cc1fd5` — same conclusion, slightly
different absolute numbers.

### Bisect trace

Range 1 (`fd65034cf4` fast / `c6b8cc1fd5` slow, 37 commits):

| Step | Commit tested | Result | Decision |
|---|---|---|---|
| 1 | `9444ff933b` (mid) | 0.82 — slow | search first half |
| 2 | `0d2a89565c` (mid) | 0.56 — fast | search upper half |
| 3 | `b110f8e450` (mid) | 0.76 — slow | search first half |
| 4 | `a1106f1f33` (mid) | 0.57 — fast | search upper half |
| 5 | `2d5eb6ef59` (sole remaining) | 0.60 — fast | **culprit = `b110f8e450`** |

Range 2 (`c6b8cc1fd5` fast / HEAD slow, 9 commits):

| Step | Commit tested | Result | Decision |
|---|---|---|---|
| 1 | `a6bec091bb` (first commit in range) | 1.10 — already at HEAD level | **culprit = `a6bec091bb`** |

## Root cause: `b110f8e450` — independent multi-component volumes

The commit adds the OpenGL-parity independent-components rendering path
(per-component scalar normalization, per-component transfer-function lookup and
per-component blend accumulation). The benchmark scene is single-component
(`vtkRTAnalyticSource`, `VolumeNumComponents == 1`), so every new branch is
guarded by

```metal
const bool useIndependentPath =
  (volumeUniforms.useIndependentComponents > 0.5) &&
  !doTransfer2D && !(doMask && numLabels > 0.0);
```

which is `false` at runtime, and the shader-side CPU work (`compChanged`,
`doCompReload`, extra textures) is also inside `independentComps`. The 27%
slower single-component frame therefore comes from the code *existing*, not
running:

- **Register pressure inside the hot march loop.** The commit unconditionally
  allocates per-sample arrays in `marchVolumeUnified`:
  `half scalarNormComp[4]`, `half4 compColor[4]` (inside the loop), plus the
  function-scope accumulators `half mipMaxScalarComp[4]`,
  `minipMinScalarComp[4]`, `avgBlendSumComp[4]`, `int avgBlendCountComp[4]`,
  `additiveSumComp[4]`. Because `useIndependentPath` is a runtime uniform, the
  compiler cannot dead-code-eliminate the new branches, so it must keep
  registers live for the arrays and the retained control flow. Higher register
  usage lowers the shader's occupancy and increases local-memory pressure,
  which shows up as lower throughput for the same per-fragment work.
- **Shader bloat.** The shader grew ~500 lines (various hunk headers) with the
  march-loop code size approximately doubling around the per-sample body; the
  independent branches are uniform-coherent (cheap in isolation), but they force
  the register allocation above.

### Fix applied

The single-component vs independent path is now a **compile-time
specialization**: the mapper derives a new `VolumeFeature_IndependentComponents`
feature-mask bit (set only when the volume actually has independent components
and the 2D transfer-function / label-map fallbacks are inactive) and bakes it
into the pipeline via a new `fc_independentComponents` function constant
(`[[function_constant(19)]]`). The shader's `useIndependentPath` is now exactly
`fc_independentComponents`, so single-component pipelines compile the per-sample
arrays and branches out entirely and the independent pipeline pays only for what
it uses. Visual output is unchanged (all 97 RenderingVolumeCxx-Metal image
tests pass identically with and without the fix; the pre-existing environment
failures on this machine are unrelated).

Measured on `VolumeRayCast` (60 frames × 5 reps): **1.04 → 0.79 ms/frame**
(964 → 1272 fps). This recovers the independent-path register pressure in full
and also relieves the backport's added register load, since the two commits
compete for the same registers.

## Fix 2: specialize the remaining feature-flag branches via function constants

The single-component fast path still carried a set of runtime-uniform branches
in the hot march loop (dead for the benchmark configuration, but alive at
runtime so the compiler could not eliminate them):

- `doTransfer2D` / `doRectilinear` — 2D transfer-function sampling and
  rectilinear coordinate-curve remapping in the hot loop.
- The dependent multi-component RGBA / LA transfer-function sampling.
- The multi-light vs headlight lighting selection
  (`lightUniforms->defaultLighting == 0`) and the light-count loop bound.
- The RenderToImage first-opaque-sample tracking block.

All of these are now compile-time. Seven new function constants
(`fc_transfer2D`(20), `fc_rectilinear`(21), `fc_defaultLighting`(22),
`fc_lightCount`(23), `fc_dependentRGBA`(24), `fc_dependentLA`(25),
`fc_renderToTexture`(26)) are baked from the feature mask, which now also
encodes the light count (4-bit field) and the RenderToImage flag. Headlight
pipelines compile the entire multi-light accumulation loop out; non-TF_2D,
non-rectilinear, non-dependent, and non-RTT pipelines eliminate their
respective hot-loop branches. The `haveOpaquePos` RTT block is guarded by
`fc_renderToTexture`, so non-RTT pipelines drop the per-sample depth tracking.

One ordering caveat surfaced in the image tests: `uniforms.UseRectilinear` is
filled in *after* the feature mask is built, so the rectilinear bit is derived
from `this->RectilinearInput` (set during `EnsureEffectiveInput`) instead.

Measured on `VolumeRayCast` (60 frames × 5 reps): **0.79 → 0.70 ms/frame**
(1272 → ~1428 fps), M/GL ratio 0.67 → ~0.55. Output is unchanged — the
GL/Metal thresholded error stays 0.095 and all 97 `RenderingVolumeCxx-Metal`
image tests pass.

## Fix 3: bake cropping and blanking, drop redundant uniform re-checks

Fix 2 left two hot-loop branches that were still guarded by *runtime uniforms*
rather than function constants, so their code stayed alive (dead but present)
for every pipeline:

- **Cropping** (`doCropping = volumeUniforms.useCropping > 0.5`): the per-sample
  crop-region bitmask test (`computeCropRegion` + bitwise test) in the march
  loop.
- **Uniform-grid blanking** (`doBlanking = volumeUniforms.useBlanking > 0.5`):
  a block containing **seven texture fetches** per sample (label/blanking
  lookups plus the per-mode branching), all executed for every non-blanked
  pipeline.

Neither was encoded in the feature mask. Two new feature-mask bits
(`VolumeFeature_Cropping` = 1u<<22, `VolumeFeature_Blanking` = 1u<<23) are now
set from `uniforms.UseCropping` / `uniforms.UseBlanking` in both feature-mask
builders (the block/direct builder in `GPURender` and the fullscreen builder in
`DrawBlocksFullscreen`), and baked into the pipeline via two new function
constants (`fc_cropping`(27), `fc_blanking`(28)). Non-cropping, non-blanked
pipelines now compile those branches out entirely.

While there, the redundant runtime `uniform > 0.5` re-checks that duplicated an
already-baked feature flag were removed from the hot loop:
`doShading`/`doGradOp`/`doMask` no longer re-test
`useGradientShading`/`useGradientOpacity`/`useMask` (the feature bit is set iff
the flag is on), and the shading path no longer re-tests
`useComputeNormalFromOpacity` / `useNormalTexture` next to
`fc_computeNormalFromOpacity` / `fc_normalTexture`. These were uniform-coherent
branches already, so this only removes uniform loads — it cannot change output.

Measured on `VolumeRayCast` (60 frames × 5 reps): **0.65 ± 0.02 → 0.64 ± 0.01
ms/frame** (~1530 → ~1545 fps), M/GL ratio ~0.50 → ~0.50. The change is inside
run-to-run σ (a warm-cache run spiked to 0.70 before settling), i.e. the
remaining dead branches cost less than the independent-path bloat that
motivated Fix 1/2 — consistent with cropping/blanking being feature-limited
paths with modest register footprints. The win is therefore primarily
*compile-time* (branch-free hot loop, fewer uniform loads) rather than a
measurable frame-time change for this scene.
Output is unchanged: the GL/Metal thresholded error stays 0.095, all 97
`RenderingVolumeCxx-Metal` image tests pass, and the cropping
(`TestGPURayCastCropping*`) and ghost-array blanking
(`TestGPURayCastVolumeGhostArrays*`) Metal tests still pass with the new
specialized pipelines.

## Root cause: `a6bec091bb` — the 22 GL-parity fixes backport

This commit squashes 22 individual correctness fixes (camera-inside near-plane
clip, TF opacity correction, shading on every `alpha>0` sample, GL-aligned
composite gate/termination thresholds, blue-noise jitter, densified box
geometry, GL winding order, removed block-bounds march exit, etc.) into one
commit on top of the `c6b8cc1fd5` baseline. It cannot be bisected further with
git. Per-frame suspects, in order of plausibility:

1. **`e40517af16` — removed the non-GL block-bounds march exit.** An
   early-exit that skipped work for fully-outside-box segments was deleted in
   favour of GL-parity block-bounds behaviour; every fragment now walks the
   whole segment.
2. **`2f88ec35ac` — densified camera-outside box geometry to GL
   `vtkDensifyPolyData(2)`.** More vertices per frame raises vertex/geometry
   throughput on the box-quad pass that drives the ray-cast fragments.
3. **`c022b1f24a` — composite gate/termination thresholds aligned with GL.**
   A looser gate lets more samples accumulate before early termination, so the
   march runs longer per fragment.
4. **`5371e30652` / `970e84d6d5` / `ef4452a9bc` / `23e6c3d328`** — exact blue-noise
   jitter tile, unorm normalization division, per-fragment model-transpose
   normal, dataset-space anchor interpolation: per-sample arithmetic additions.
5. The shader diff itself grew `Rendering/Metal/Shaders/MetalShaders.metal` by
   512 lines and the mapper by 354 lines, inflating register usage of the march
   loop in the same way as `b110f8e450`.

Confirming the exact fix requires re-landing the 22 changes individually (they
exist as separate commits on `metal-ios`, referenced in the backport message)
and re-profiling each, or temporarily reverting the four suspects above.

## Side observations

- The GL/Metal visual thresholded error for this scene went from 0.0065
  (`fd65034cf4`) to 0.095 (`c6b8cc1fd5` and later) — both far below the
  `--threshold` fail level and visually indistinguishable; the parity work did
  not measurably regress image fidelity for this scene.
- A small additional drift (+0.06 ms, 0.76 → 0.82) sits between `b110f8e450`
  and `9444ff933b` (commits `f247fe427e` per-component materials/gradients,
  `aca0d236bc` multi-light alignment, `5989b3c032` ghost-array blanking). It was
  not bisected further.
- GL timing variance is consistently higher than Metal's (σ up to 0.18 on some
  scenes vs ≤ 0.03 for Metal), so GL/GL-ratio conclusions should always be
  drawn from `--reps > 1` runs.

## Fix 4: align the fullscreen feature-mask builder with GPURender

Two feature-mask builders disagreed about `VolumeFeature_NormalTexture`:
`GPURender`'s builder sets it whenever `GradientNormalTexture` exists
(mm:7120-7121), but `DrawBlocksFullscreen`'s builder did not (mm:6290-6318).
Consequence: with `UsePrecomputedNormals` enabled, the block/direct path
compiled `fc_normalTexture=true` while the camera-inside fullscreen path
compiled `fc_normalTexture=false`, so the latter fell back to the 6-texel
gradient even though the precomputed texture was bound (slot 7, mm:6152) and
the fullscreen shader consumes the constant (`marchVolume`).

Fix: mirror the check in `DrawBlocksFullscreen` (mm:6299-6300). Default scenes
(`UsePrecomputedNormals=false`) have `UseNormalTexture=0`, so the mask is
unchanged and behavior/perf is identical. Build clean; all 288 Metal tests
pass. Commit `ce8b652cb4`.

## Metal SDK and best-practices review: remaining levers and how they scale

### What the benchmark path actually executes

The `VolumeRayCast` scene uses all defaults, so `GPURender` dispatches to the
**direct-to-screen** branch — no offscreen intermediate:

- `RenderToImage` (RTT export) is off (mm:7337).
- `ImageSampleDistance == 1.0` (base-class default) → `useImageSampling` off
  (mm:7397). Offscreen-then-blit therefore only runs in the RTT and
  image-sampling modes, not for the benchmark.
- Camera outside, single block → `DrawBlocks` direct path (mm:7563) into the
  renderer's active encoder (BGRA8Unorm + Depth32Float).

The hot cost is per-sample: a 6-texel gradient (`computeGradientFast`,
MetalShaders.metal:3545) plus Phong lighting for every `alpha > 0` sample.

### SDK availability (macOS 15, deployment target 14.0)

Verified against the installed Metal SDK headers:

| Capability | Availability | Verdict for this mapper |
|---|---|---|
| `MTLBinaryArchive` | macOS 11+ | startup/PSO hitches only, not frame time |
| Classic ray tracing (`MTLAccelerationStructure`) | macOS 11+ | no win — per-sample march dominates |
| Mesh shaders (`MTLMeshRenderPipelineDescriptor`) | macOS 13+ | only for very high block counts |
| Classic argument buffers | macOS 10.13+ | noise until CPU-bound |
| `MTLStorageModeMemoryless` | macOS 11+ | legal on Apple Silicon Mac, but see below |
| Metal 4 (`MTL4*` headers) | macOS 26 only | not usable on this system |

### Memoryless attachments on Apple Silicon Macs (corrected)

Apple Silicon Macs (M1-M4) are **TBDR** — the same tile-memory architecture as
iOS/Vision — so `MTLStorageModeMemoryless` is fully supported on macOS. The
earlier framing that memoryless is "iOS/Vision mostly" was wrong; the actual
blocker here is **render-pass topology**, not GPU class:

- The offscreen `ImageSampleColorTexture` (RGBA16Float, mm:1586-1591) is written
  in one render pass (mm:7480-7510) and **read by a later blit pass**
  (mm:7512-7513). Memoryless textures have no backing store and are only valid
  within a single render pass, so a later pass cannot read them back.
- The benchmark direct path has no offscreen intermediate at all, so memoryless
  never applies to it.
- Legal memoryless uses here: the per-pass depth attachment, or an in-pass MSAA
  resolve. A meaningful win would require restructuring the volume + composite
  into a single pass via tile memory (`[[color(m)]]` programmable blending),
  which overlaps the compute/tile-march territory and forfeits the free
  rasterizer coverage + hardware depth handling (mm:5736-5744, 6084-6088).

### Lever table: benchmark (dense 33³) vs higher volume complexity

Larger or partitioned volumes change the cost mix: sample count per ray grows
with volume depth, sparse real-world data makes empty-space skipping matter,
and many blocks add geometry/CPU-side costs. Re-evaluating each lever:

| Lever | Benchmark (dense 33³) | Higher complexity | Verdict |
|---|---|---|---|
| Precomputed gradient normals (`UsePrecomputedNormals` / `fc_normalTexture`) | clean: no per-frame cost, ~144 KB texture, one-time build; 6→1 fetch | biggest: per-sample win scales with ray length; also relieves texture-cache pressure and register pressure | best lever; only matters when shading **or** gradient-opacity is on (`.w` carries grad magnitude, MetalShaders.metal:4555-4570); costs 4 B/voxel (512³ ≈ 512 MB) + one-time compute build (mm:2350-2354) |
| Min-max empty-space skipping (`fc_minmax`) | overhead — dense data has nothing to skip | dominant win on sparse/real volumes; macrocell size is a tunable | already active (`UseGPUMinMax`, mm:4066-4144); tune macrocell for large extents |
| Partitioned grid traversal | n/a (`Partitions={1,1,1}`) | essential for huge/sparse volumes; per-brick occupancy skip | already implemented (`fragment_volume_grid_traversal_main`, mm:5347) |
| Classic ray tracing | no win | marginal for many blocks | skip — the software march still runs |
| Compute/tile march + `[[color(m)]]` | no win | moderate at high res (occupancy, tile compositing) | forfeits coverage/depth; only if normals + min-max are insufficient |
| Mesh shaders | no | only for thousands of blocks (CPU-side geometry generation) | low |
| Binary archives | startup only | more PSO variants → more startup hitches | app polish, not frame time |
| Classic argument buffers | noise | CPU-bound scenes only | low |
| Sample distance / termination thresholds | parity trade | scales linearly with samples | gate behind a parity toggle |

### Precomputed-normals cost analysis

`EnsureGradientNormalTexture` (mm:2237) builds an RGBA8Unorm 3D texture via the
`volume_compute_normals` compute kernel, cached against
`VolumeUploadTime`/mapper MTime (mm:2257-2269). Costs:

- **Per-frame: none.** The stale check is cheap MTime compares; `Modified()` only
  fires on data upload or `SetPartitions` (mm:1499).
- **One-time build:** a compute pass over the whole volume per data upload —
  trivial at 33³, a real hitch at 512³+.
- **Memory:** 4 B/voxel (RGBA8Unorm), in addition to the 2 B/voxel R16Float
  volume — significant at 512³+.
- **Wasted if unused:** enabling the flag with shading *and* gradient-opacity
  both off pays build + memory for zero benefit (the gradient is never computed
  in that config anyway).
- **Quality:** 8-bit-quantized normals/magnitude vs the computed 6-fetch path —
  a parity concern the `--threshold` harness must verify, not a perf one.

## Recommendations

1. **Done:** recover the single-component fast path by specializing the
   ray-cast shader on the independent-components flag via a Metal function
   constant (`fc_independentComponents`). This addresses the register-pressure
   cost shared by both culprits without changing output.
2. **Done:** bake the remaining feature-flag branches (2D transfer function,
   rectilinear, dependent RGBA/LA, multi-light vs headlight, light count,
   RenderToImage depth tracking) into function constants so every pipeline
   compiles only the hot-loop paths it actually uses.
3. **Done (Fix 3):** bake cropping and uniform-grid blanking into function
   constants (`fc_cropping`(27), `fc_blanking`(28)) and remove the redundant
   runtime `uniform > 0.5` re-checks that duplicated already-baked feature
   flags. This relieves the last dead-but-present code in the hot loop for
   non-cropping, non-blanked pipelines. Frame time is unchanged within noise
   (0.65 ± 0.02 → 0.64 ± 0.01 ms/frame); the value is a smaller, branch-free hot
   loop and fewer per-sample uniform loads.
4. The remaining ~0.04 ms/frame vs the `fd65034cf4` baseline (0.64 → 0.60) is
   the inherent cost of the backport's *behavioural* GL-parity changes —
   primarily shading applied to every `alpha > 0` sample (a 6-texel gradient +
   Phong lighting per sample, `a4415d2329`), plus the tightened termination
   threshold and the removed block-bounds exit. These change the rendered image
   by design, so they can only be traded back for performance, not optimized
   away. If that trade is wanted, gate them behind a parity toggle.
5. If GL-parity output is not required for a given configuration, allow the
   parity march (block-bounds exit, gate thresholds, densified box, per-sample
   shading) to be toggled off.
6. Re-land `a6bec091bb`'s 22 fixes individually so regressions of this size can
   be attributed to a single change.
7. **Done (Fix 4):** align the `DrawBlocksFullscreen` feature-mask builder with
   `GPURender` so the precomputed-normal bit (`fc_normalTexture`) is set for the
   camera-inside path too (commit `ce8b652cb4`). No behavior change with the
   default (`UsePrecomputedNormals=false`); a latent specialization mismatch
   fixed.
8. **Next, the only remaining per-sample lever for the benchmark: precomputed
   gradient normals.** Prototype `UsePrecomputedNormals=true`, verify the
   8-bit-quantized normals/magnitude against `--threshold` (currently 0.095),
   and measure the fps/M-gl ratio. It is the one change that edits the hot-loop
   arithmetic (6 fetches → 1 per lit sample) and should deliver the largest
   measurable win for this scene. For large (512³+) volumes, weigh the 4 B/voxel
   texture memory against the gain and only enable when shading or
   gradient-opacity is active.
9. For higher-volume-complexity scenarios (larger/sparser extents, partitioned
   volumes), the levers above re-rank: min-max empty-space skipping and the
   grid-traversal path become the dominant wins (already implemented — tune the
   macrocell size), while ray tracing, mesh shaders, and argument buffers stay
   marginal. No additional Metal-API work is indicated.

## Reproduction

```
./macos_metal_build.sh --resume --tests          # builds the harness (tests must be ON)
./build_macos_metal/bin/vtkMetalGLVisualComparison --bench --scene VolumeRayCast --frames 60 --reps 5
```

To re-measure an arbitrary commit: `git checkout <sha>`, rebuild
`ninja -C build_macos_metal vtkMetalGLVisualComparison`, run the command above.
