# Metal vs OpenGL volume ray cast: promote the last two half-precision sites (per-component scalar ranges, float volume storage) to float32; NEAREST re-measurement is byte-identical, closing the half-precision work item (update 15)

Follow-up to all prior documents; read as an addendum to
[VolumeRayCastBackendComparisonFindingsUpdate14.md](VolumeRayCastBackendComparisonFindingsUpdate14.md).

Update 14 promoted every `half` arithmetic site in `MetalShaders.metal` to
`float`, but two `half` sites remained in the mapper/CPU path. This session
eliminated both, then re-measured the no-jitter NEAREST scene on genuine
backends: the result is **byte-identical** to update 14 (max 22 / mean 0.37,
642 masked px, same worst pixels) — as expected, because the single-component
uint16 test scene exercises neither site. This closes the half-precision work
item: every remaining `half` in the volume ray-cast path is either a
debug-logging cast, a now-default-off conversion kernel, or a variable name.

## 1. Site 1: per-component scalar ranges (committed `0882a41507`)

`VolumeMapperUniforms::ScalarMinCompHalf`/`ScalarMaxCompHalf` were packed as
`uint16_t` half (via `FloatToHalf`), and the shader declared
`half scalarMinComp[4]`/`half scalarMaxComp[4]` — diverging from OpenGL, which
passes `in_scalarsRange` as `float` vec2 (`vtkOpenGLGPUVolumeRayCastMapper.cxx`
line 3879, `RangeVec` is `float`). The per-component scalar range feeds the
independent-component normalization (`scalarNormComp`) and the dependent-LA
paths, so a half-rounded range would shift those components' transfer-function
lookups by up to half ulp.

Change: both the CPU struct and the Metal shader struct now store
`float scalarMinComp[4]`/`float scalarMaxComp[4]`. The layout is preserved
exactly — the old `half[4] + _pad[4]` pairs were 16 bytes per field, so
`float[4]` (16 bytes) keeps `scalarMinComp` at 1168, `scalarMaxComp` at 1184,
and `componentWeight` at 1200. The static_asserts were updated accordingly.

## 2. Site 2: volume texture format for float / large-int data (committed `0882a41507`)

`ChooseVolumeFormat` selected `R16Float`/`RG16Float`/`RGBA16Float` for
`VTK_FLOAT` and large-int (`VTK_INT`, `VTK_DOUBLE`, …) data whenever
`PreferHalfPrecision` (default `true`) was set and the scalar range fit in
[−65504, 65504]. OpenGL uploads the same data as `GL_R32F`/`GL_RG32F`/
`GL_RGBA32F` (`vtkVolumeTexture.cxx` lines 723–738 and 781–783), so the Metal
backend was silently quantizing float volumes to half storage.

Change: `PreferHalfPrecision` default flipped to `false`, so float and
large-int data now upload as full `float32` (R32F family), matching GL. The half
conversion kernels (`volume_convert_*_to_half`, the `Convert*ToHalfPipeline`s,
and the R16Float y-data path) remain in place but are off by default for anyone
who opts back in via `SetPreferHalfPrecision(true)`.

Note: `VTK_UNSIGNED_SHORT` data (the test scene) is unaffected on both sides —
Metal uses `R16Unorm`, GL uses the native ushort upload with scale/bias, and the
8-bit output was already float-parity (updates 9–13).

## 3. Re-measurement (procedures §2/§3) — byte-identical to update 14

Fresh `-T` per backend, black dummy baseline, GL engagement verified via
`GL_SAMPLING=12`. Scene:
`TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter`
(NEAREST interpolation, camera inside, no shading, no grad-op, no transform,
jittering off).

| metric | update 14 | update 15 (after fix) |
|---|---|---|
| max\|Δ\| | 22 | 22 |
| mean\|Δ\| | 0.37 | 0.37 |
| masked (≥5) px | 642 | 642 |
| exact-equal px | 177880 | 177880 |
| within-2 px | 259444 | 259444 |
| worst px | (372,131) Δ22, (422,92) Δ19, … | identical |
| R/G/B regression | ≈1.000·gl + 0.0 | identical |

As predicted, neither changed site is exercised by this scene: the test data is
single-component uint16 (site 1 only matters for independent/multi-component
volumes, site 2 only for float/int storage). The 642-px NEAREST residual is
therefore **independently confirmed** to be the sample-position offset
(update-12 §10), with zero contribution from any remaining half precision.

## 4. Status: half-precision work item closed

The full float32 parity checklist:

- Transfer-function LUTs: RGBA32F both sides ✓
- Composite accumulators (`accumulatedColor`/`accumulatedOpacity`): float ✓
- Scalar window/level normalization: float32 ✓
- Sample distance / evalStep: GL `g_dirStep` float32 chain ✓
- Gradient computation: float32 ✓
- Lighting (`computePhongLightingVolumeFast`, `computeVolumeLighting`): float ✓
- 2D transfer function lookup: float ✓
- Blend-mode accumulators / MIP/MinIP comparisons: float ✓
- **Per-component scalar ranges: float32 ✓ (this update)**
- **Float volume storage: R32F default ✓ (this update)**

Remaining `half` occurrences in `MetalShaders.metal` are: debug-logging casts
(GRADOP neighbor dumps, lines 4438–4443, under `VTK_METAL_ENABLE_LOGGING`),
the opt-in `volume_convert_*_to_half` kernels (5570–5572), and variable names
(`halfW` in geometry line shaders). None are in the default float32 render path.

## 5. Files changed

- `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm` — `ScalarMinCompHalf`/
  `ScalarMaxCompHalf` (uint16) → `ScalarMinComp`/`ScalarMaxComp` (float) at
  offsets 1168/1184; updated static_asserts; fill uses plain `float`.
- `Rendering/Metal/Shaders/MetalShaders.metal` — shader struct
  `scalarMinComp`/`scalarMaxComp` → `float[4]`.
- `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.h` — `PreferHalfPrecision`
  default `true` → `false`.
- `Rendering/Metal/VolumeRayCastBackendComparisonFindingsUpdate15.md` — this
  document.
