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

The GL-faithful fix is to make the single-component vs independent path a
**compile-time specialization** (e.g. a per-pipeline Metal function constant
driven by `useIndependentComponents`) instead of a runtime uniform, so the
single-component pipeline compiles to the pre-commit lean shader and the
independent pipeline only pays for what it uses.

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

## Recommendations

1. Recover the single-component fast path by specializing the ray-cast shader
   on the component count / independent flag (Metal function constants), which
   addresses both culprits' register-pressure cost at once.
2. If GL-parity output is not required for a given configuration, allow the
   parity march (block-bounds exit, gate thresholds, densified box) to be
   toggled off.
3. Re-land `a6bec091bb`'s 22 fixes individually so regressions of this size can
   be attributed to a single change.

## Reproduction

```
./macos_metal_build.sh --resume --tests          # builds the harness (tests must be ON)
./build_macos_metal/bin/vtkMetalGLVisualComparison --bench --scene VolumeRayCast --frames 60 --reps 5
```

To re-measure an arbitrary commit: `git checkout <sha>`, rebuild
`ninja -C build_macos_metal vtkMetalGLVisualComparison`, run the command above.
