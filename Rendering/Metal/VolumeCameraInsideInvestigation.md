# Metal volume ray-cast: camera-inside image-diff investigation

Status: **on-hold**. All concrete hypotheses raised during this investigation
have been tested and refuted or verified as parity-correct. The residual
camera-inside image diffs are well-characterized (obligueness-dependent,
sub-texel) but their root cause has not been isolated, in part because the
OpenGL backend renders black on this Metal-only build, so the checked-in
baselines are the only GL reference available.

This document is a record of the investigation: the committed state, every
experiment run (with methodology, data, and outcome), the verified
parity points, the error structure analysis, and concrete leads for whoever
picks this up next.

---

## 1. Objective

Match the Metal GPU volume ray-cast backend's output to the OpenGL
baselines for the camera-inside tests:

- `TestGPURayCastCameraInside`
- `TestGPURayCastCameraInsideSmallSpacing`
- `TestGPURayCastCameraInsideTransformation`
- `TestGPURayCastCameraInsideNonUniformScaleTransform`

"Camera inside" means the camera is dollied inside the volume bounding box,
which switches the renderer from the proxy-geometry path to a fullscreen
reconstruction path.

## 2. Committed state (reference)

- Commit `c6b8cc1fd5` (on `metal-ios`): near-plane clamp uniforms +
  `setupVolumeRay` clamp in `MetalShaders.metal` for OpenGL near-plane
  proxy-clip parity. Pushed.
- The `(M^-1)^T` normal convention is kept (the mathematically-correct `M^T`
  regressed Transformation 0.106 -> 0.485 and NonUniformScale 0.058 -> 0.138).
- Baseline suite state (committed code): **83 PASS / 10 FAIL / 4 ABORT**.
- Image-compare failures and metrics:
  - `RenderToTexture` 0.0559
  - `NonUniformScaleTransform` 0.0575
  - `ShadedClipping` 0.0645
  - `CameraInsideTransformation` 0.1059
  - `CameraInside` 0.1282
  - `ClippingUserTransform` 0.2436
  - `CameraInsideSmallSpacing` 0.4891
- Non-image failures: `ProjectedTetrahedra`, `ProjectedTetrahedraOffscreen`,
  `ProjectedTetrahedraTransform`, `ProjectedTetrahedraVectorComponent`,
  `ProjectedTetrahedraZoomIn`, `TestMultiBlockUnstructuredGridVolumeMapper`,
  `TestSmartVolumeMapperImplicitArray`.

## 3. Test configuration (what the failing tests do)

All four camera-inside tests share this configuration (in
`Rendering/Volume/Testing/Cxx/TestGPURayCastCameraInside*.cxx`):

- `vtkRTAnalyticSource` ("ironProt"), 68 x 68 x 68.
- `SetAutoAdjustSampleDistances(0)`.
- `SetSampleDistance(1.0)` (SmallSpacing uses `7e-6` with a tiny bounds
  `5e-4`).
- Linear interpolation, shading off, no jittering (default).
- Color TF: 0 -> (0,0,0), 64 -> (1,0,0), 128 -> (0,0,1), 192 -> (0,1,0),
  255 -> (0,0.2,0). Opacity TF ramps 0 at scalar 0 to 1 at 255.
- Interactor event log dollies the camera inside the volume; the final frame
  is compared against the baseline PNG.
- Window 301 x 300.

Consequences of the configuration:

- At `sampleDistance = 1.0`, every ray samples ~1 voxel apart, so aliasing /
  comb-phase effects are maximally visible (this is why these tests are the
  sensitive ones).
- The baseline `TestGPURayCastCameraInside.png` render is produced by the
  **clipped + densified proxy** path in the GL backend, not a fullscreen path.

## 4. Established parity facts (all verified against GL source)

These are the facts that were re-derived during the investigation and are
used as the frame of reference for every experiment.

### 4.1 Jitter fallback

GL with jittering off sets `g_rayJitter = g_dirStep` (one full step), not
zero (`vtkVolumeShaderComposer.h:459-464`). Metal's fallback
`1.0 * stepSize` is the GL parity value. Do **not** change it to `0.0`.

### 4.2 Entry / first sample

GL: `g_rayOrigin = ip_textureCoords` (proxy entry), then
`g_rayOrigin += g_rayJitter` (`g_dirStep`), then `g_dataPos = g_rayOrigin`;
the march samples `g_dataPos` first, then advances by `g_dirStep`
(`vtkVolumeShaderComposer.h:442-465`). So the first sample is
**entry + 1 step** in both backends. Metal: `firstT = jitter = stepSize`
and `currentPoint = rayOrigin + rayDir*tStart + rayDir*stepSize`
(`MetalShaders.metal:3698-3703`).

### 4.3 Texel step and cell-to-point

GL: `g_dirStep = (in_cellToPoint * in_inverseTextureDatasetMatrix) *
vec4(rayDir, 0) * in_sampleDistance` (`vtkVolumeShaderComposer.h:437-438`,
`ip_inverseTextureDataAdjusted` at :107). Metal: `texStep =
volumeToTexture * (rayDir * boundsSize) * stepSize`, `evalStep = texStep *
ctpScale`, `evalPoint = ctp(texLocalPos)` (`MetalShaders.metal:3646-3652`).
Since `cellToPoint` is affine, GL's "advance by ctp-scaled step" and Metal's
"advance raw, then ctp the position" are numerically equivalent:
`ctp(entry + k*texStep) = ctp(entry) + k*ctp(texStep)`.

Cell-to-point constants are identical: GL scale `(delta-1)/delta`, offset
`0.5/delta` for `delta = 68` (`vtkVolumeTexture.cxx:1210-1230`) == Metal
`ctpScale = (texelCount-1)/texelCount = 67/68`, `ctpOffset = 0.5/68`
(`MetalShaders.metal:3649-3651`).

### 4.4 Composite accumulation

GL single-component composite:
```
g_srcColor.a = computeOpacity(scalar);
g_srcColor = computeColor(scalar, g_srcColor.a);   // raw rgb
g_srcColor.rgb *= g_srcColor.a;                     // pre-multiply
g_fragColor = (1.0f - g_fragColor.a) * g_srcColor + g_fragColor;
```
(`vtkVolumeShaderComposer.h:2976-2996`). Metal (`MetalShaders.metal:4215-4294`):
```
weight = 1 - accumulatedOpacity;
accumulatedColor += weight * sampleColor * sampleOpacity;   // sampleColor raw
accumulatedOpacity += weight * sampleOpacity;
```
`sampleColor` comes from `colorOpacity.rgb`, the raw TF table
(`sampleTransferFunction` returns `half4(tfTex.sample(...))`,
`MetalShaders.metal:3037-3042`). Formulas and pre-multiplication are identical.

### 4.5 Near-plane geometry (camera-inside clip)

- GL computes the near frustum plane and pushes it into the volume by
  `offset = (far - near) * 0.001` with `minOffset = FLT_EPSILON * 1000`,
  then clips the proxy box against it (`vtkOpenGLGPUVolumeRayCastMapper.cxx:
  1140-1204`).
- Metal computes the identical plane and offset and uploads
  `CameraInsideNearPlaneOrigin/Normal` in normalized volume space
  (`vtkMetalGPUVolumeRayCastMapper.mm:6492-6556`); `setupVolumeRay` clamps
  `tStart = max(boxT, tNear)` (`MetalShaders.metal:3533-3541`).
- Normal convention: both keep `(M^-1)^T` (Metal) vs `dataToWorld^T`
  (GL); for pure rotation/identity model matrices these agree.
- Verified empirically via shader logging: the final camera-inside frame's
  `tStart` values equal `tNear`, and `entry` lies exactly on the uploaded
  near plane (`n dot (entry - origin) = 0.0000`).

### 4.6 Sample distance / step normalization

Metal uploads `SampleDistanceHalf = FloatToHalf(actualSampleDistance /
maxBoundsSize)` (`mm:97`, upload at mm:6594) and the shader converts back
with `physicalSampleStep = u.sampleDistance * maxBound / length(rayDir *
boundsSize)` (`MetalShaders.metal:3002-3009`), which exactly cancels the
CPU-side `/ maxBoundsSize` for the dominant axis. GL's `in_sampleDistance`
is the physical sample distance. Both produce the same physical step per
sample for cubic, identity-model volumes.

### 4.7 Empty-space / min-max skipping

GL has no empty-space/min-max skip. Min-max is Metal-only (`marchVolumeUnified`,
`MetalShaders.metal:3759`, gated by `fc_minmax` / `UseMinMaxAccel` mm:6849).
It does not fire in the camera-inside tests (single block).

## 5. Experiments run

### A/B #1 — jitter fallback `1.0 -> 0.0` (`MetalShaders.metal:4427, 4773, 5058-5060`)

Hypothesis: the GL-parity jitter fallback of one full step was itself wrong
and caused the oblique-ray comb phase to differ.

Result: **negative**. 5 new failures (e.g. `SampleDistance` 0.0523,
`TwoComponentsDependent` 0.0634, `Clipping` 0.0563); camera-inside barely
moved (SmallSpacing 0.4891 -> 0.4513, Transformation 0.1059 -> 0.1071).
Reverted. Confirms the one-step fallback is GL parity (section 4.1).

### A/B #2 — forced `UseMinMaxAccel = 0.0f` (mm:6849)

Hypothesis: Metal-only min-max empty-space skipping diverged from GL's
no-skip march.

Result: **negative**. All 97 tests identical to 4 decimals; camera-inside
0.1282 / 0.4891 / 0.1059 / 0.0575 unchanged. Reverted.

### A/B #3 — `UseFullscreenCameraInside` default flipped `true -> false` (header:267)

Hypothesis: the fullscreen path's per-fragment ray reconstruction
(`reconstructRayDir` via `ndcToVolume`, `MetalShaders.metal:3439-3446`)
differed from the densified-proxy interpolated entry that GL uses.

Result: **negative**. The CPU-proxy-clip path produced metrics that were
identical or worse than fullscreen: `CameraInside` 0.128233 (unchanged),
`NonUniformScaleTransform` 0.354816, `Transformation` 0.154896,
`SmallSpacing` 0.501972. Reverted.

This is an important negative: because the proxy path (which is structurally
what GL does) gives the same image as the fullscreen path, the difference is
**not** in the entry/proxy reconstruction. It must live in the march itself
(the comb, sampling, or accumulation) which both Metal paths share.

### A/B #4 — half -> float accumulation precision

Hypothesis: Metal accumulates color/opacity in `half`
(`MetalShaders.metal:3708-3709`, all weight math in `half`), while GL
accumulates in `float` (`vec4`); the multiplicative front-to-back opacity
accumulation (`acc += (1-acc)*a`) in half could terminate at a different
step (`acc >= 0.99h` early-out), shifting the output.

Result: **negative**. Two-phase:

1. First attempt changed accumulator types only. The runtime Metal shader
   compile **failed** with MSL strict-vector-conversion errors:
   ```
   error: implicit conversions between vector types ('float3' and 'half3') are not permitted
   error: no matching constructor for initialization of 'half4'
   ```
   The render came out all black, and the image metric read 0.81 (which is
   just "black vs baseline"). This is a caution: a 0.8x metric is a
   diagnostic smoke test for a broken shader, not a real result. The
   per-command-buffer error is printed by `vtkCocoaMetalRenderWindow.mm:156`
   ("Failed to compile shared shader library").
2. Fixed with explicit casts everywhere:
   `accumulatedColor += weight * float3(sampleColor) * float(sampleOpacity);`
   and `half4(half3(accumulatedColor), half(accumulatedOpacity));`.

   Result: metrics **identical** to the committed baseline:
   `CameraInside` 0.128234, `SmallSpacing` 0.489101, `Transformation`
   0.106193, `NonUniformScaleTransform` 0.057512.

   Half-vs-float accumulation is refuted. Reverted to the committed state.

MSL lesson recorded: MSL does **not** implicitly promote `halfN <-> floatN`
vectors in mixed binary ops or vector constructors; explicit casts are
required, and a compile error shows up as a silent black frame, not a build
failure (shaders compile at runtime via `newLibraryWithSource`).

### Shader-log validation (diagnostic, reverted)

A diagnostic `os_log` dump was added to `fragment_volume_fullscreen_main`
(gated to 8 fixed pixel positions, test-builds only via
`#if defined(VTK_METAL_ENABLE_LOGGING)`), logging `cameraPos`, `rayDir`,
`entryPoint`, `tStart`, `stepSize`, `totalBoxT`, the near-plane
origin/normal, `useCameraInsideNearClip`, `sampleDistance`, and volume
bounds for each frame of `TestGPURayCastCameraInside`. Run with:
```
MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=2097152 MTL_LOG_TO_STDERR=1 \
  <build>/bin/vtkRenderingVolumeCxxTests TestGPURayCastCameraInside --vtk-factory-prefer RenderingBackend=Metal ...
```

Data for the final camera-inside frame (camera at `(0.678174, 0.486826,
0.964175)`, inside `[0,1]^3`):

- `step = 0.014923`, `sd = 0.014923` (the fp16 rounding of `1/67`;
  see section 6).
- `nearOrg = (0.396744, -0.029335, 1.033587)`,
  `nearNrm = (-0.358232, 0.026488, -0.933257)`.
- `tStart ~= 0.0232` for **every** gated pixel with `nearClip=1`; all gated
  pixels were near-plane-clipped. `tStart` equals `tNear`; the `entry` point
  satisfies `n dot (entry - origin) = 0.0000` exactly. The near plane is
  ~0.023 normalized units (~1.6 voxels) in front of the eye.
- `totalBoxT` varies 0.976 .. 1.134 across the gated pixels (the box
  t-range remaining after the near-plane clamp).

Conclusion: the Metal comb in the camera-inside frame is internally
consistent and matches the GL formulas (sections 4.2, 4.3, 4.5). No comb
anomaly was found, including in the corner-patch pixels.

## 6. fp16 `sampleDistance` analysis (arithmetic, no experiment needed)

`SampleDistanceHalf` is the only fp16 quantity in `VolumeMapperUniforms`
(all other fields float32; `half sampleDistance` at `MetalShaders.metal:2695`,
uniform field at mm:97).

- SmallSpacing: ratio `7e-6 / 5e-4 = 0.014`. `0.014` is near-exactly
  representable in fp16 (exponent -7, mantissa `1.792`, error ~1e-8).
  **Zero** step error, yet SmallSpacing is the *worst* diff (0.4891).
- CameraInside: ratio `1/68 = 0.0147059` -> fp16 `0.0147095`, relative error
  ~2.4e-4 (~0.016 voxel drift over 68 steps).
- Transformation (x scaled by 2): ratio `1/136 = 0.00735` -> fp16 `0.007349`,
  relative error ~3.5e-4.

Conclusion: fp16 step rounding cannot produce the observed diffs (it is
zero for the worst case and sub-0.02-voxel elsewhere). The `half
sampleDistance` is not implicated.

## 7. Error structure analysis (current committed render vs baseline)

### CameraInside

- Interior: Metal is a smooth, uniform **+3..+5 (8-bit) brighter** than the
  baseline in the red channel along the center row, with smaller differences
  on the center column (0..6). Mean absolute RGB difference ~2-3. This is a
  small systematic offset, not banding.
- Corner patches: the metric is dominated by thin corner strips. Worst
  region top-left (y ~0..30, x ~40..60): baseline renders **dark blue**
  `(19, 22, 73)` while Metal renders **bright green** `(18, 128, 1)`.
  Using the test's color TF, blue ~= scalar ~128, green ~= scalar ~192, so
  Metal samples ~64 scalar units higher than the baseline there. Given
  ironProt's high-frequency content, this is consistent with a **spatial
  sampling offset of several voxels** in the corner region, not a color/opacity
  scaling issue. The earlier fully-black/constant misreads were GL-suite runs
  overwriting the shared `Testing/Temporary` filenames, not Metal renders.

### SmallSpacing

- Error is **smoothly radial**: mean |diff| ~6 at the image center, growing
  monotonically to ~31 near radius ~120, then falling off as rays leave the
  volume silhouette. Radial profile (mean |err| per 10-px annulus):
  `r 0-10: 5.9, 40-50: 16.4, 80-90: 25.9, 110-120: 30.8, 150-160: 20.3,
  190-200: 11.9`.
- This is a per-ray, **obliqueness-dependent** difference (rays near the
  volume perimeter are longest/most oblique). High-frequency banding
  (autocorrelation lag ~33 px) is present within the annuli.

### Interpretation

The radial growth with obliqueness, combined with the small uniform interior
offset and the corner patches, points at a per-ray sampling-position
difference that scales along the ray (not a global transform error, not a
comb-constant error, not accumulation precision — the comb is verified and
precision is refuted). See section 8, leads 1-4.

## 8. Extra leads to investigate (not yet tested)

None of these were run; they are ordered by how strongly the data supports
them.

1. **`volumeToTexture` matrix vs GL's texture-coordinate mapping.**
   The strongest remaining candidate. Metal's sampling coordinate is
   `texLocalPos = volumeToTexture * (boundsMin + currentPoint * boundsSize)`
   (`MetalShaders.metal:3738-3740`); GL's proxy `uvx` comes from
   `in_inverseTextureDatasetMatrix` baked into the proxy vertices. If the
   two mappings differ by a sub-texel offset (padding/alignment convention,
   a `+0.5` texel, or a different normalization), every sample shifts
   slightly. That produces exactly the observed signature: a small uniform
   interior offset, larger errors at sharp scalar gradients (corner
   patches), and obliqueness-dependent drift (radial SmallSpacing error).
   *How to test*: log the uploaded `volumeToTexture` matrix and the proxy
   bounds for a single block and compare numerically against
   `vtkVolumeTexture`'s GL matrices; or add a second comb to
   `fragment_volume_fullscreen_main` that computes the sample position with
   GL's exact `ctp * invTextureDataset` formula and diff the resulting
   image against the current one.

2. **Exit/termination handling at the last sample.**
   Metal breaks on `currentPoint` leaving the block bounds with a `1e-4`
   epsilon (`MetalShaders.metal:4318`) and composites the boundary voxel via
   the clamp-to-edge `seenInBounds` logic (`:3773-3780`); GL terminates on a
   fixed step count `g_terminatePointMax = length(g_terminatePos -
   g_dataPos) / length(g_dirStep)`. On oblique rays these can disagree by
   one sample at the exit, shifting the tail of the accumulation and the
   boundary ring of the image. A one-sample difference on long oblique rays
   is consistent with the radial error. *How to test*: log `maxSteps` and the
   last-sample position per ray and compare against GL's step count for the
   same geometry (computable on the CPU from the logged uniforms).

3. **Scalar normalization / TF lookup in half.**
   `scalarNorm` is computed in `half` (`MetalShaders.metal:3632-3633`, and
   per-component at :3946-3955). GL normalizes in float. Half ULP error on
   the TF coordinate shifts the color/opacity lookup by a sub-texel amount,
   which a steep opacity ramp (ramps 0..1 over the full scalar range here)
   turns into a small per-sample opacity difference — i.e. a smooth,
   accumulation-shaped error that grows with sample count (radial).
   *How to test*: compute `scalarNorm` in float and re-run the four tests.
   Cheap A/B, one rebuild.

4. **Fractional texel drift of the incremental position.**
   Both backends advance the sample position by repeated float adds
   (`texLocalPos += texStep` vs `g_dataPos += g_dirStep`). The rounding
   differs because the two increments are computed by different matrix
   chains, so over ~70 steps the combs can drift apart by a fraction of a
   texel in a direction-dependent way. This is the same signature as lead 1
   and would be tested by the same experiment (a numerically-exact GL comb
   in the Metal shader).

5. **A GL reference render.** On a machine with a working GL backend, render
   the four tests with GL and diff GL-vs-Metal directly (image-to-image).
   This machine's GL backend renders black (`vtkRenderingVolumeOpenGL2CxxTests`
   reports the same 0.809873 as a black image), so it was impossible here.
   Direct GL-vs-Metal comparison would remove all baseline-generation
   uncertainty and localize the error to a per-pixel spatial pattern.

## 9. Workflow notes (for reproducibility)

- Build: `./macos_metal_build.sh --resume --tests` (macOS) /
  `./ios_metal_build.sh --resume`.
- Targeted camera-inside metrics:
  `ctest --test-dir build_macos_metal -R "RenderingVolumeCxx-Metal-TestGPURayCastCameraInside..." -j 4`
  then `rg "Failed Image Test" build_macos_metal/Testing/Temporary/LastTest.log`.
- Full suite: `python3 Rendering/Metal/Testing/metal_ctest_report.py -p
  RenderingVolumeCxx-Metal -j 8`.
- The Metal and GL suites write **the same PNG filenames** into
  `Testing/Temporary/`; always copy renders apart or re-run immediately, and
  ignore apparently-constant images (they are GL-suite pollution).
- `Testing/Temporary/` is untracked; `git status` should be clean otherwise.
- Shader logging: see `Rendering/Metal/Testing/Cxx/TestMetalVolumeShaderLog.cxx`
  for the full recipe (MSL `os_log`, `MTL_LOG_*` env vars, format-specifier
  rules: string-literal format only, `%v4hlf` for vectors, no `%s`/`%n`).
- A failed MSL compile shows up as a **black frame**, not a build error;
  check stderr for `vtkCocoaMetalRenderWindow.mm:156` "Failed to compile
  shared shader library" before trusting any metric from a modified shader.
- A/B workflow discipline: one change at a time, rebuild after every change
  (including reverts), and record the committed baseline metrics
  (0.0575 / 0.1059 / 0.1282 / 0.4891) before starting.

## 10. Key file references

- `Rendering/Metal/Shaders/MetalShaders.metal` — `setupVolumeRay` ~3499-3560
  (near-plane clamp 3533-3541), `reconstructRayDir` 3439-3446,
  `marchVolume` 4413+, `marchVolumeUnified` 3707+ (accumulators 3708-3709,
  composite 4215-4294, termination 4311-4320), `physicalSampleStep`
  3002-3009, `firstT`/`evalStep` 3646-3704, `half sampleDistance` 2695,
  `sampleTransferFunction` 3037-3042, `cellToPointTextureCoord` 3087.
- `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm` — `SampleDistanceHalf`
  97, near-plane uniforms 6492-6556, `UseMinMaxAccel` 6849,
  `SetupBuffers` camera-inside proxy clip 5232-5380, `IsCameraInside` 4845.
- `Rendering/VolumeOpenGL2/vtkVolumeShaderComposer.h` — GL jitter
  `g_rayJitter = g_dirStep` 459-464, `g_dirStep` 437-438, composite
  2976-2996, ctp vertex transform 104-107.
- `Rendering/VolumeOpenGL2/vtkVolumeTexture.cxx` — `ComputeCellToPointMatrix`
  1196-1230.
- `Rendering/VolumeOpenGL2/vtkOpenGLGPUVolumeRayCastMapper.cxx` — near-plane
  clip 1140-1204.
- `Rendering/Volume/Testing/Cxx/TestGPURayCastCameraInside.cxx` (+
  SmallSpacing / Transformation / NonUniformScaleTransform) — the failing
  tests, ironProt 68^3.
- `Rendering/Metal/Testing/Cxx/TestMetalVolumeShaderLog.cxx` — shader
  logging recipe.
