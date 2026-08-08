# Metal vs OpenGL volume ray cast: transfer-function lookup promoted from half to float32 (update 9)

Follow-up to all prior documents; read as an addendum to
[VolumeRayCastBackendComparisonFindingsUpdate8.md](VolumeRayCastBackendComparisonFindingsUpdate8.md),
whose §5.3 claimed that "everything else (TF LUT contents, lookup convention, …)
has already been matched". That claim was true of the table *values* but not of
their **storage and sampling precision**: the Metal backend uploaded the shared
RGBA transfer function as an `RGBA16Float` (half, 11-bit mantissa) texture and
sampled/accumulated it in `half`, whereas the OpenGL backend uploads **float32**
tables (`RGB32F` color + `R32F` opacity) and composites in full `float`. This
update promotes the Metal transfer-function pipeline to float32 end to end and
measures the improvement across the whole comparison-test family.

## 1. CPU table fill: RGBA32F replicating `vtkOpenGLVolumeOpacityTable::InternalUpdate` byte-for-byte

`vtkMetalGPUVolumeRayCastMapper.mm` gains
`FillTransferFunctionRGBA32FWithPreIntegration()` (4-argument range form plus a
single-range wrapper used by dependent multi-component volumes). It mirrors the
OpenGL backend's `InternalUpdate` exactly:

- `vtkColorTransferFunction::GetTable`/`vtkPiecewiseFunction::GetTable` are fed
  `float*` buffers, so each double result is cast to float on store — the same
  value GL stores in its `RGB32F`/`R32F` textures.
- The composite pre-integration correction is applied per float entry:
  `a = (float)(1.0 - pow(1.0 - (double)a, factor))` gated on `a > 0.0001f`, and
  the additive correction is `a = (float)((double)a * factor)` — structurally
  identical to GL's loop.
- RGB channels are stored verbatim (no `clamp`, no quantization), matching GL's
  `vtkOpenGLVolumeRGBTable`.

Both the main single-path table and the four component tables switch to
`MTLPixelFormatRGBA32Float` (`bytesPerRow = width * 16`). The former 16F fill,
which quantized every entry through `FloatToHalf` before upload, is retained in
the file for reference but no longer used by the upload paths.

## 2. Shader: float32 sampling and accumulation in `marchVolumeUnified`

`MetalShaders.metal` promotes the transfer-function sampling chain to `float`:

- `sampleTransferFunction()` and `sampleComponentTransferFunction()` now return
  `float4` (the `half4(...)` wrap on the texture fetch is removed, so the
  bilinear-interpolated lookup keeps full float32 precision).
- The per-sample `colorOpacity` is `float4`, `sampleOpacity` is `float`, and the
  composite `sampleColor` is `float3`; the accumulation
  `accumulatedColor += weight * sampleColor * sampleOpacity` and
  `accumulatedOpacity += weight * sampleOpacity` now run entirely in float32.
- Dependent RGBA and LA paths sample the (now float32) table directly into
  `float`/`float4`; the 2D-TF and label-map paths keep their 16F sources but are
  wrapped in an explicit `float4(...)` at the assignment to `colorOpacity`.
- Shading still runs in `half3` (the lighting kernels and normals are untouched
  this update); the `float3` `sampleColor` is narrowed to `half3` at the
  `computeVolumeLighting`/`computePhongLightingVolumeFast` call sites and the
  result widened back with an explicit `float3(...)`.

### 2.1 Metal's implicit-conversion rule (compile constraint)

Metal does **not** implicitly convert between `half`/`float` *vector* types in
assignments or initializers (only scalars promote in expressions). The first
build passed the C++/encode step yet failed at *runtime* shader-library
compilation on exactly these mismatches (`assigning to 'half4' from incompatible
type 'float4'`, etc.). The runtime library build (`vtkCocoaMetalRenderWindow`)
is the authoritative check — a lesson recorded here: the `vtk_encode_string`
step embeds the source, so the C++ link can succeed while the MSL is broken.
All conversions are now explicit: independent-path `compColor`/MIP exit reads
keep their prior half precision via `half4(sampleTransferFunction(...))`, the
composite path consumes the float32 values directly.

## 3. MIP/MinIP accumulators promoted to float (exit-coordinate rounding fix)

While verifying the LUT change, `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformMaxIP`
surprised: switching the table to float32 alone **increased** its >4-LSB diff
(4090 → 9470 pixels). The root cause surfaced when inspecting the MIP exit: the
per-sample normalized scalar is float (`float scalarNorm =
saturate(rawScalar * scalarScale + scalarBias)`), but the MaxIP/MinIP ray
accumulators were `half mipMaxScalar` / `half minipMinScalar`. The exit re-sample
`sampleTransferFunction(tf, float2(float(mipMaxScalar), 0.5))` therefore used a
**half-rounded coordinate**, up to half-ULP (~0.0005 for scalars ≈ 0.26) off GL's
full-float `l_maxValue.w`. Near the steep opacity node (1150 HU, opacity
0.02 → 0.85 within a few texels) that 0.5-texel-class shift moves the sampled
opacity by tens of LSB. The 16F table masked this because its own half
quantization "snapped" the interpolation to coincidentally-close values; the
float32 table exposed it systematically.

Fix: `mipMaxScalar` and `minipMinScalar` are now `float` (updated from the
already-float `scalarNorm`), so the exit coordinate matches GL's float selection.

## 4. Verification: full comparison-test family, before vs after

All variants were run back-to-back with both factories
(`--vtk-factory-prefer RenderingBackend=Metal/OpenGL`) using the same capture
harness as Update8. **The OpenGL reference images are byte-identical across runs
for every variant** (determinism check, 0/15 changed), so the deltas below are
entirely attributable to the Metal changes.

"before" = prior 16F-LUT build (recap capture), "after" = this float32 build.
Metric: pixel count with max-channel |Δ| > 4 LSB against the OpenGL reference.

| variant | before >4 | after >4 | before max | after max |
|---|---|---|---|---|
| TestGPURayCastCameraInsideTransformation | 180 | 40 | 13 | 7 |
| …ConstGradOp | 316 | 39 | 13 | 13 |
| …NoGradOp | 510 | 68 | 15 | 15 |
| …NoShade | 189 | 27 | 13 | 13 |
| …NoShadeConstGradOp | 0 | 0 | 2 | 1 |
| …NoShadeLinGradOp | 200 | 38 | 13 | 12 |
| …NoShadeNoGradOp | 0 | 0 | 2 | 1 |
| …NoShadeNoGradOpNoTransform | 1175 | 646 | 22 | 22 |
| …NoShadeNoGradOpNoTransformCamOutside | 247 | 27 | 12 | 8 |
| …NoShadeNoGradOpNoTransformFineStep | 1175 | 646 | 22 | 22 |
| …NoShadeNoGradOpNoTransformMaxIP | 4090 | 2866 | 85 | 85 |
| …NoShadeNoGradOpNoTransformNearPlaneTiny | 1175 | 646 | 22 | 22 |
| …NoShadeNoGradOpNoTransformNearest | 1175 | 646 | 22 | 22 |
| …SampleDist0_25 | 180 | 40 | 13 | 7 |
| …SampleDist0_5 | 180 | 40 | 13 | 7 |

Every variant improves, roughly 2–5× by the >4-LSB count; the two
no-gradient-opacity no-shade variants are effectively pixel-perfect
(`NoShadeConstGradOp`/`NoShadeNoGradOp`, max Δ = 1). The standard shaded test
and both sample-distance variants drop max Δ from 13 to 7.

## 5. Current state and remaining residual

With the transfer-function contents, storage, sampling, and accumulation all
now float32 and matching OpenGL's, the per-sample composite color differs from
GL by at most the residual float32 sources:

1. **Gradient-opacity sampling**: `sampleGradientOpacity()` still returns
   `half` (the `R32F` gradient-opacity table is read through a `half(...)`
   cast) and `gradNormFactor` is still computed as `half`. The gradient-magnitude
   path is the dominant remaining contributor to the shaded/`GradOp` variants
   (27–68 pixels >4 LSB).
2. **Offscreen inter-block accumulation**: the volume renders in 8×8 blocks into
   an `RGBA16Float` offscreen texture, and the block results are combined in a
   second pass — each block boundary rounds through half. The `…NoTransform`
   family (camera-outside, fixed-step, 646 pixels >4 LSB, max 22) and MaxIP
   (2866 pixels >4, max 85) sit here; MaxIP's large-max pixels trace to the
   extreme-scalar selection being an extremum over many samples, which is
   sensitive to sub-ULP sample-position differences near localized features.
3. Update8 §6's proposal (seed Metal's first sample from GL-style interpolated
   texture coordinates) remains open and orthogonal to this change.

## 6. Files changed

- `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm` — `RGBA32F` fill
  functions + `RGBA16Float → RGBA32Float` texture format/bytesPerRow at the two
  upload sites.
- `Rendering/Metal/Shaders/MetalShaders.metal` — `sampleTransferFunction`/
  `sampleComponentTransferFunction` → `float4`; `colorOpacity`/`sampleOpacity`/
  `sampleColor` → float; explicit casts at the half-precision boundary
  (independent path, 2D TF, MIP/MinIP exit); `mipMaxScalar`/`minipMinScalar`
  → `float`.
