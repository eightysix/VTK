# Metal vs OpenGL volume ray cast: scalar window/level normalization promoted from half to float32 (update 6)

Follow-up to all prior documents; read as an addendum to
[VolumeRayCastBackendComparisonFindingsUpdate5.md](VolumeRayCastBackendComparisonFindingsUpdate5.md)
(whose §10 left the seed phase (B) and minmax-skip arithmetic (C) as the next
candidates) and its predecessors.

## 1. Hypothesis

The scalar → normalized-value (window/level) step that feeds the color/opacity
LUT lookups is computed in **half precision** in the Metal shader, while the
OpenGL shader computes the identical mapping in **full float32**. Because the
test volume's opacity ramp is steep at the skin/bone shells, a half-quantized
LUT coordinate shifts every ramp lookup and should appear as shell-edge deltas.

## 2. Evidence in the two backends

**OpenGL** (`Rendering/VolumeOpenGL2/`):

- `vtkVolumeTexture::GetScaleAndBias` (`vtkVolumeTexture.cxx:663-706`) computes
  the window/level scale/bias in double/float: for USHORT,
  `glRange[i] = ScalarRange[i] / (USHORT_MAX + 1)`,
  `scale = 1/(glRange[1]-glRange[0])`, `bias = -glRange[0]*scale`.
- The scale/bias are uploaded as float32 uniforms `in_volume_scale`/`in_volume_bias`
  (`vtkOpenGLGPUVolumeRayCastMapper.cxx:3653-3685`).
- The composite shader applies them in full float32 immediately after the raw
  texture fetch:
  `scalar.r = scalar.r * in_volume_scale[0].r + in_volume_bias[0].r`
  (`vtkVolumeShaderComposer.h:2697-2699`, also 2704-2705), then samples the
  opacity/color LUTs at that float32 value.

**Metal** (`Rendering/Metal/Shaders/MetalShaders.metal`):

- The `scalarMin`/`scalarMax` uniforms were `half` (offsets 244-251, written via
  `FloatToHalf` at `vtkMetalGPUVolumeRayCastMapper.mm:6659-6664`).
- The scale/bias were computed and stored in `half` (`:3731-3732`), and the
  LUT coordinate was `half`:
  `half scalarNorm = saturate(half(rawScalar) * scalarScale + scalarBias)`
  (`:4079`).

Half has an 11-bit mantissa: at `scalarMax ≈ 0.0667` (4371.9/65535) the ulp is
~3e-5 (≈0.05% of the value), and at the ramp norm values 0.2-0.3 the ulp is
~2e-4. Every LUT lookup therefore carries a per-sample coordinate quantization
that GL does not have.

## 3. The change (kept)

Promote the whole window/level chain to float32, structurally identical to GL:

- `VolumeMapperUniforms.scalarMin`/`scalarMax` → `float` (the two half pairs at
  offsets 244-251 become two floats; **all struct offsets and total size are
  unchanged** — `static_assert(sizeof(VolumeMapperUniforms) == 1712)` and the
  offset asserts in `vtkMetalGPUVolumeRayCastMapper.mm` still hold).
- `MetalShaders.metal:3733-3734`: `float scalarScale` / `float scalarBias`.
- `MetalShaders.metal:4081`: `float scalarNorm = saturate(rawScalar*scalarScale + scalarBias)`.
- `vtkMetalGPUVolumeRayCastMapper.mm:6657-6661`: write sites store float32.
- The single-component `scalarNormComp[4]` init now casts explicitly to half
  (`MetalShaders.metal:4086`); the per-component independent path still uses the
  `half scalarMinComp/scalarMaxComp` pair (not exercised by this test).

## 4. Results (fresh A/B, identical capture conditions)

Same variant as Update3's table
(`TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutside`,
default step, 512×512, `mean|Δ|` = per-pixel max-channel absolute delta,
"masked px" = `|Δ| ≥ 5/255`). **Both** backends were re-captured fresh with the
current build so the reference is not stale.

| metric | before (HEAD, half norm) | after (float32 norm) |
|---|---|---|
| center (GL vs M) | [254 175 134] | [254 175 134] (exact) |
| delta mean | −0.05 | −0.05 |
| mean\|Δ\| | 0.27 | **0.14** |
| max\|Δ\| | 8 | 8 |
| masked (≥5) px | 31 | **28** |

The half→float32 change roughly halves `mean|Δ|` and removes 3 of the 31
shell-edge pixels. `max|Δ|` (8) and the shell-edge footprint are otherwise
unchanged. Note: an initial claim of a 13501 → 28 masked-pixel collapse was
measured against a stale stored Metal capture and is **retracted**; the real,
reproducible A/B numbers are those in the table.

## 5. Interpretation

The half-quantized LUT coordinate was a real contributor to the shell-edge
deltas (halving the mean), but it is not the dominant remainder. The 28
remaining masked pixels (deltas 5-8/255, RGB deltas near-equal per channel)
are the boundary-phase signature already attributed in Update5 §6 to
boundary-crossing decisions: entry phase (`firstT`/jitter), opacity-threshold
termination, and the minmax-skip stepping. They sit exactly on the high-contrast
shell ring where a one-sample phase difference between the two lattices changes
the accumulated opacity at the surface.

## 6. Next steps (in order)

1. **Promote the Metal transfer-function textures from half (RGBA16Float) to
   float32** (OpenGL uploads RGBA32F). The shader already returns
   `half4(sampleTransferFunction(...))`; the texture format is the remaining
   quantization on the LUT output. Expected to shave the per-sample op/color
   ulp but likely not the 5-8 shell deltas.
2. **Seed-phase measurement (option B)**: diff Metal's entry texture coords /
   `firstT` / jitter against GL's interpolated `ip_textureCoords` at the 28
   masked pixels; a sub-step entry offset is the cleanest explanation of a
   uniform shell-edge phase shift.
3. **Minmax-skip measurement (option C)**: dump Metal `curCell`/`exactSkip`/
   `skipSteps` vs GL `g_skip` on the same rays.
4. Re-run the sample-distance sweep (`sd 0.0675→4.0`) to confirm the new floor.

## 7. Rejected / deferred

- Reverting the change: rejected — it is a strict improvement (mean|Δ| halved,
  zero cost, matches GL's arithmetic structure).
- Promoting the independent-components `scalarMinComp`/`scalarMaxComp` halves:
  deferred — not exercised by the comparison test (single component).
