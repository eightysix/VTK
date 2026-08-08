# Metal vs OpenGL volume ray cast: the earlier byte-identical capture was a fluke; the re-capture reproduces the update-12 residual table; evalStep aligned to GL g_dirStep float32 chain; remaining half-precision entries catalogued (update 13)

Follow-up to all prior documents; read as an addendum to
[VolumeRayCastBackendComparisonFindingsUpdate12.md](VolumeRayCastBackendComparisonFindingsUpdate12.md).

Update 12 left the `…NoShadeNoGradOpNoTransform` residual at 646 pixels > 4 LSB,
root-caused to nearest volume interpolation amplifying sub-0.02-texel backend
position differences at bone-plateau boundaries (its §10). This session set out
to confirm the *byte-identical* capture that update 13 was intended to record,
then ship the seed-alignment change. The confirmation **failed**: the re-capture
reproduces the update-12 residual table exactly, so the earlier 0-gap run was a
capture artifact. This update records that result, the parity change that is
still worth keeping (evalStep now matches GL's `g_dirStep` float32 chain), and a
catalogue of the remaining Metal `half`-precision sites versus GL's all-float
pipeline, which is the next work item.

## 1. Confirmation attempt (procedures §2/§3): NOT byte-identical

After reverting the half→float lighting probe (so the tree held only the
evalStep/`SampleDistanceWorld` change), rebuilt
`vtkRenderingVolumeCxxTests` and re-captured all 18 variants on genuine
backends with `BackendComparisonTools/capture_variants.sh` (each OpenGL run
verified GL-engaged via `GL_SAMPLING`/`GL_OPTABLE`/`GL_TEX` stderr markers).

Result — this reproduces the update-12 residual table; it is **not**
byte-identical:

| variant | >4 LSB px | max\|Δ\| | update-12 expectation |
|---|---|---|---|
| `TestGPURayCastCameraInsideTransformation` | 40 | 7 | shaded 27–68 ✓ |
| `…ConstGradOp` | 38 | 13 | shaded 27–68 ✓ |
| `…NoGradOp` | 68 | 15 | shaded 27–68 ✓ |
| `…NoShade` | 27 | 13 | shaded 27–68 ✓ |
| `…NoShadeConstGradOp` | 0 | 1 | 0 ✓ |
| `…NoShadeLinGradOp` | 37 | 12 | (new) |
| `…NoShadeAmp` | 1459 | 222 | (new) |
| `…NoShadeNoGradOp` | 0 | 1 | 0 ✓ |
| `…NoShadeNoGradOpNoTransform` | **646** | 22 | **646** ✓ |
| `…NoShadeNoTransform` | 1321 | 32 | (new) |
| `…NoShadeNoGradOpNoTransformCamOutside` | 27 | 8 | (new) |
| `…NoShadeNoGradOpNoTransformNearest` | 646 | 22 | == NoTransform ✓ |
| `…NoShadeNoGradOpNoTransformMaxIP` | 2866 | 85 | (new) |
| `…SampleDist0_5` | 40 | 7 | == base ✓ |
| `…SampleDist0_25` | 40 | 7 | == base ✓ |
| `…NonUniformScaleTransform` | 32003 | 139 | (new) |
| `…SmallSpacing` | 0 | 4 | (new) |
| `…BlendModes` | 4454 | 89 | (new) |

The `…NoShadeNoGradOpNoTransform` variant returns exactly the update-12
residual: 646 pixels > 4 LSB, worst pixels (372, 131) Δ=22 and (422, 92) Δ=19
— the two samples whose raw combs straddle the bone-plateau edge (update-12 §2).

## 2. The earlier 0-gap capture was a fluke

The 10:28 `baseline_capture/` artifacts (all 18 variants `max|d|=0`) could not
be reproduced with the identical working tree. The likeliest mechanism is a
shared-output overwrite in that capture script: both backend runs pointed at the
same `-T` (Testing/Temporary) and wrote the rendered image to the same
`<test>.png` path, so one backend's file could be double-copied. Per the user's
directive this is not investigated further; the re-capture in §1 (which uses a
fresh `-T` dir per backend) is the reliable measurement, and it agrees with the
entire update-10→12 residual history.

## 3. Seed-alignment change (committed): evalStep == GL g_dirStep float32 chain

`Rendering/Metal/Shaders/MetalShaders.metal` `marchVolumeUnified` previously
built `evalStep` as `(adjustedLin * (p.rayDir * boundsSize)) * p.stepSize`,
which normalized in volume space and folded the sample distance across the
`SampleDistance/maxBoundsSize` uniform and `physicalSampleStep`. GL builds
`g_dirStep = (ip_inverseTextureDataAdjusted * normalize(vertexPos - eyePos)).xyz
* in_sampleDistance` with the normalize done in DATASET/OBJECT space
(`computeRayDirection`). The shader now replicates GL's float32 chain exactly:

```metal
float3 dirObj = normalize(p.rayDir * boundsSize);
float3 evalStep = (adjustedLin * dirObj) * volumeUniforms.sampleDistanceWorld;
```

`sampleDistanceWorld` is a new uniform (`VolumeMapperUniforms`, CPU struct offset
1244, `vtkMetalGPUVolumeRayCastMapper.mm:199`, set to `actualSampleDistance` at
`:6714`) mirroring GL's `in_sampleDistance` (not divided by `maxBoundsSize`).

No observable effect on the unit-bounds `…NoShadeNoGradOpNoTransform` scene:
`maxBounds = 1` makes `normalize(p.rayDir * boundsSize) == p.rayDir` and
`sampleDistanceWorld == stepSize`, so old and new forms are numerically
identical there (the 646-px residual is the nearest-interpolation boundary
flip of update-12 §10, independent of the seed). The change is a structural
parity improvement for transformed/scaled scenes (`NonUniformScaleTransform`,
`SmallSpacing`).

## 4. Half-precision catalogue: remaining Metal `half` vs GL `float` sites

The two backends are already float-equal on: transfer-function LUTs (RGBA32F
both sides), composite accumulators, scalar window/level normalization,
gradient computation, and the volume texture (uint16/R16Unorm). The following
Metal sites still compute in `half` where GL uses `float` (each sub-8-bit-LSB in
isolation, but they accumulate):

- **Lighting**: `computePhongLightingVolumeFast` (3342) and
  `computeVolumeLighting` (3372) — `half3` colors/normals/`half` `pow`, and the
  half `viewDirHalf`/`lightDirHalf`/material uniforms at 3804–3809; GL computes
  `float nDotL`, `vec3 r`, `pow(...)` (vtkVolumeShaderComposer.h:1297–1309).
- **`finalColor`**: half4 truncation of the float composite (4566, 4583–4654)
  and the half `wlScale`/`wlBias` window/level (4659–4661).
- **Blend-mode accumulators**: `avgBlendSum`/`additiveSum` (3865–3867) and
  `avgBlendSumComp`/`additiveSumComp` (3874–3876), plus the MIP/MinIP
  per-component comparisons (4284–4293).
- **`sampleTransferFunction2D`** returns half4 (3064).
- **`marchSegment`** grid path: `thread half3& accumulatedColor` / `thread half&
  accumulatedOpacity` (4735–4736).
- **Per-component independent path**: `scalarNormComp`/`compScale`/`compBias`
  (4112–4130), `compColor` (4148), `totalAlpha` (4267), `tmpRGB`/`tmpA`
  (4356–4357), `maskLabel` (4135), `secondNorm` (4169).

These are the entries to promote to `float` for precision parity with GL.

## 5. Next steps

1. Promote the §4 half-precision sites to `float` one at a time, re-capturing
   with procedures §2/§3 after each and keeping only changes that leave the
   8-bit output unchanged or closer to GL.
2. The nearest-interpolation residual (§1) is a separate, filter-setting
   artifact (update-12 §10): identical output under nearest would require
   bit-identical sample positions across backends.

## 6. Files changed

- `Rendering/Metal/Shaders/MetalShaders.metal` — `evalStep` built in GL
  `g_dirStep` float32 order; new `sampleDistanceWorld` uniform member.
- `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm` — CPU struct
  `SampleDistanceWorld` at offset 1244; set from `actualSampleDistance`.
- `Rendering/Metal/VolumeRayCastBackendComparisonFindingsUpdate13.md` — this
  document.
