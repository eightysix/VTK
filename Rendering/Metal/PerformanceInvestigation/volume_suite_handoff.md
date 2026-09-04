# Volume Suite Handoff — RenderingVolumeCxx-Metal 96/97

Status as of `fac8ac80e4` on `metal-cleanup-1` (M2, `arm64 Release`,
`./macos_metal_build.sh --resume --tests`).

## Current state

| Suite | Result | Sole failure |
|---|---|---|
| `RenderingVolumeCxx-Metal`, default march (mv0) | 96/97 | `TestGPURayCastCameraInsideTransformation` 0.0912 (near-miss) |
| Same, `VTK_METAL_TEST_MARCH_VARIANT=9` | 96/97 | same test 0.0913 |
| NIFTI harness parity 512 | `2999 thr 0.000` exact | — |

Run it with:

```sh
python3 Rendering/Metal/Testing/metal_ctest_report.py -p RenderingVolumeCxx-Metal -j 8
# mv9 arm:
VTK_METAL_TEST_MARCH_VARIANT=9 python3 Rendering/Metal/Testing/metal_ctest_report.py -p RenderingVolumeCxx-Metal -j 8
```

The script inherits parent env (`Popen` without `env`), so perf knobs are a
prefix. Only `MARCH_VARIANT=9` is non-default-optimal; everything else optimal
is already default (dense auto, `W1PRE` ON, blocks ON, transpose argmin). Do NOT
add `VOLTRANSPOSE_AXIS=y` (workload knob, loses raw axis-y) or `FRAG_BATCH=16`
(regresses lean-coarse, see `perf_investigation_part2.md:26.9`).

## What got fixed this session (all landed)

1. **Depth/camera-inside PSO dead-strip** (`vtkMetalGPUVolumeRayCastMapper.mm`,
   `714fd2451f`): `fc_useDepthTexture`/`fc_useCameraInside` came from bare env
   presence (default OFF) while the uniforms said ON → volume+polydata and
   4× camera-inside tests failed. Now auto-derived per frame from live state
   (occlusion texture + actors present for depth; `IsCameraInside` for camera)
   with tri-state env override (`VTK_METAL_TEST_DEPTH`,
   `VTK_METAL_TEST_CAMERA_INSIDE`: unset=auto, bare/nonzero=on, `0`=off).
   Volume-only scenes keep the §18 diet exactly (NIFTI key `0x240004`,
   ABBA tie vs `DEPTH=0`).
2. **Auto-dense exclusions** (same commit): bypass never fires for 2D
   transfer functions or multivolume inputs (1D component-0 sampling cannot
   measure their occupancy). NIFTI still classifies dense (95.6%).
3. **mv9 inverted blanking** (`MetalShaders.metal` `MV9_COMPOSITE`/
   `MV9_COMPOSITE_LOOP`): read only `.r`, skipped when `< 0.5` — inverted
   polarity on the wrong channel. Now mirrors mv0 (mode-selected `.x`/`.y`,
   `> 0`). Fixed `UniformGridBlanking`, `GhostArrays`×2 and, unexpectedly,
   `Transfer2DYScalars` + `MultiBlockMapper` (both carry blanking).
4. **mv9 MIP/MinIP ignores binary mask**: mask skip existed only in the
   composite/additive/label branches; mv0 skips upstream. Added the binary-mask
   early skip for blend modes 1/2 in both macros. Fixed `MIPBinaryMask`.
5. **Gradient-opacity magnitude normalization** (`2233ef1273`): divisor
   `GradientOpacityMax` carried `/avgSpacing`, which GL does not have (GL folds
   spacing into its gradient aspect ratio). Invisible at unit spacing — which
   is why every green gf scene stayed green. This test: 0.191 → 0.103.
6. **1024-entry gradient-opacity LUT** (`fac8ac80e4`): match GL's default
   width (was 256). Zero metric movement (0.0912 both) — kept for parity.

## The residual: CameraInsideTransformation 0.09

**Scene** (unchanged since 2016, GL baseline passes at 0.0065): headsq/quarter
resampled to 512³ (USHORT, spacing 0.39/0.39/0.27), bone TF + gradient opacity
`(0,0),(90,0.5),(100,0.7)` + shade, volume rotated Rx180/Ry85/Rz55, camera
inside, one wheel-dolly replayed before compare.

**Exonerated** (each toggled in isolation, metric bit-identical): march
variant, minmax/dense/blocks (`DENSE=0/1`, `MM_BLOCKS=0` all 0.191 pre-fix),
W1PRE, transpose, GPU contents, jitter on/off, clipping-range refresh,
central-vs-forward vs nearest gradient taps (`GRAD4`, `GRAD_NEAREST`),
lighting direction (shade-on + gradop-off matches at 0.000%), TF cull (safe
by construction: culled samples carry <2% alpha), pow skip, resample filter
(reader-direct also diverges), rotation (64³ rotated matches at 0.015%),
interaction replay (final cameras bit-identical both backends).

**Narrowed to gradient-opacity magnitude on smooth sub-unit-spacing data**
(`0.19 → 0.09` across the `GradientOpacityMax` sweep: `0.5/avg` 0.191,
`0.5` 0.148, `0.25` 0.103, `0.125` **0.091**, `0.0625` 0.093) with lighting,
march, TF, texture content/width/sampling all verified matching. Gain scaling
is exhausted (broad floor at ~0.09); the remainder is offset/structural, not
gain — a pure scale error would zero out. Suspects, in order: (a) additive
offset in sampled magnitude, (b) sub-texel sampling-position difference in
the steep 90→100 band (2–9 entries depending on width), (c) half-precision
roundoff in `half sPX` gradient taps on smooth fields (visible only where
adjacent-texel differences approach half-ulp; `GRAD_FLOAT` was neutral, so
this ranks last).

**Recommended next step:** per-pixel magnitude-field comparison, not more
knobs — port the metal-ios update-63-style attribute dumps (GL_RAY vs STEP
`grad.w` fields) and diff the magnitude distributions directly; a constant
shift confirms (a), band-localized error confirms (b). The `metal-ios` branch
additionally carries the update-69/70 march-exit fixes (`e40517af16`,
`6032ed6015`, multi-thousand-pixel wins on constant-scalar volumes) NOT in
this branch — worth a look if the mag fields match but exits differ.

## Methodology pitfalls hit during this work (do not repeat)

- **Compare against the baseline, not metal-vs-GL on modified scenes.**
  Modified scenes (gf off/flat/linear, no rotation, no replay) can match
  across backends for vacuous reasons (uniform modulation, saturation,
  both-black renders — actually VIEW the PNGs).
- **Direct-binary runs lack `VTK_TESTING_IMAGE_COMPARE_METHOD=TIGHT_VALID`**,
  which ctest sets. Metrics from the two harnesses are incomparable
  (e.g. 5756 vs 0.38 on byte-identical PNGs).
- **Shared-TF probes can't separate magnitude from texture bugs** — both
  backends read the same table. Only saturation-free unsaturated TF shapes
  are informative, and this scene saturates hard.
- **Keep a clean 2×2 when bisecting**: several mid-investigation numbers were
  confounded by leftover test edits (once even flipping the conclusion).
  `git diff --stat -- Rendering/Volume/` before every measurement run.
- **Bit-identical metrics across runs are the norm** (deterministic stack);
  any wobble in the 4th decimal across identical configs is a red flag for
  stale binaries, not noise. Always confirm the rebuild compiled (`[1/2]
  Building`, not `no work to do`).

## Historical note (97/97 claim)

`d0f4a31066` (Aug 15) closes `Rendering/Metal/Testing/failures.txt` with
"97 pass / 0 fail / 0 abort", but the same file lists this test at 0.1676
with no documented fix afterward, and 11 of its 12 listed image failures
were genuinely fixed only later (this session included). The metal-ios
branch additionally carries an undiagnosed variant family for this exact
test (`...NoShade`, `...NoGradOp`, `...NoTransform`, `...NearPlaneTiny`,
`...SampleDist0_25/0_5`, `...MaxIP`, ...). Treat the Aug-15 line as
aspirational; the measured trajectory is 12 image failures → 1 (this test,
0.1676 → 0.09).
