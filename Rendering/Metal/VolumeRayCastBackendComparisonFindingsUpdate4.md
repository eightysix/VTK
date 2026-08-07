# Metal vs OpenGL volume ray cast: sample-lattice drift from chained evalPoint accumulators (update 4)

Follow-up to all prior documents, read as an addendum to:

- [VolumeRayCastBackendComparisonProcedures.md](VolumeRayCastBackendComparisonProcedures.md) —
  environment, capture/analyze tooling.
- [VolumeRayCastBackendComparisonFindings.md](VolumeRayCastBackendComparisonFindings.md) —
  sections 1–7, still valid.
- [VolumeRayCastBackendComparisonFindingsUpdate.md](VolumeRayCastBackendComparisonFindingsUpdate.md) —
  composite-gate and termination-threshold fixes.
- [VolumeRayCastBackendComparisonFindingsUpdate2.md](VolumeRayCastBackendComparisonFindingsUpdate2.md) —
  fp16-accumulation and MAX_RAY_STEPS fixes; §3.1 characterized the residual
  "grazing-ring-edge / sample phase" pixels at coarse steps.
- [VolumeRayCastBackendComparisonFindingsUpdate3.md](VolumeRayCastBackendComparisonFindingsUpdate3.md) —
  gradient-opacity LUT fix (1024×float32); collapsed the gf divergence family,
  leaving the ring-edge / sample-phase pixels as the common floor.

This document records the fourth root cause, established by code trace rather
than measurement: the two backends build the per-sample position with
**different accumulator architectures**. OpenGL advances one texture-space
accumulator by a fixed step; Metal advances three chained accumulators
(world → texture-space → cell-to-point) by a direction-normalized step. The
resulting sample lattice coherently drifts from OpenGL's by ~1 ulp per axis,
which is the same underlying cause as the grazing-ring-edge / crop-fence
pixels Update2 §3.1 and Update3 §2 left as the residual floor.

## 1. The two march-position architectures

### 1.1 OpenGL: one accumulator, one fixed step (texture space)

`Rendering/VolumeOpenGL2/vtkVolumeShaderComposer.h` and
`Rendering/VolumeOpenGL2/shaders/raycasterfs.glsl`:

```
// vertex-output texture coords already carry the cell-to-point shift:
g_rayOrigin = ip_textureCoords.xyz;                       // composer:418

// step = dataset->texture(rayDir) * sampleDistance, computed once per fragment:
g_dirStep = (ip_inverseTextureDataAdjusted *
             vec4(rayDir, 0.0)).xyz * in_sampleDistance;  // composer:437-438
g_lengthStep = length(g_dirStep);                         // composer:439

// marching:
g_dataPos = g_rayOrigin;                                  // composer:3307
...
g_dataPos += g_dirStep;                                   // raycasterfs.glsl:302
++g_currentT;                                             // integer counter
```

Properties:

- **Single accumulator** `g_dataPos` lives in cell-to-point-adjusted texture
  space; the shift is applied exactly once, at the vertex (`ip_textureCoords`).
- **Step is fixed per fragment**: `in_sampleDistance` is a uniform (world
  units), `ip_inverseTextureDataAdjusted` is a matrix uniform. No per-ray
  normalization — the step vector is a single `mat·vec` followed by a single
  scalar multiply.
- Termination compares `g_currentT` (integer) against
  `length(g_terminatePos - g_dataPos) / length(g_dirStep)` (composer:3310).

### 1.2 Metal: three chained accumulators, direction-normalized step

`Rendering/Metal/Shaders/MetalShaders.metal`, `marchVolumeUnified`:

```
// direction/anisotropy-dependent step (division by a per-ray length):
inline float physicalSampleStep(float3 rayDirNormSpace, constant VolumeMapperUniforms& u)
{
  float3 boundsSize = max(u.volumeBoundsMax.xyz - u.volumeBoundsMin.xyz, 1e-6);
  float  maxBound   = max(boundsSize.x, max(boundsSize.y, boundsSize.z));
  float  physPerNorm = length(rayDirNormSpace * boundsSize);
  return float(u.sampleDistance) * maxBound / max(physPerNorm, 1e-6);   // :3007-3014
}
...
float  stepSize = physicalSampleStep(rayDir, volumeUniforms);
float3 stepVec  = rayDir * stepSize;                        // world-space step
float3 texStep  = (v2t * (rayDir * boundsSize)).xyz * stepSize;  // dataset->texture, then *stepSize
float3 evalStep = texStep * ctpScale;                       // re-apply cell-to-point scale

float3 currentPoint = ...;                                  // world accumulator
float3 texLocalPos  = ...;                                  // texture-space accumulator
float3 evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);  // :3795
...
// in the march loop, every continue/advance path does all four at once:
currentPoint += stepVec;   // :3951
currentT     += p.stepSize;// :3952
texLocalPos  += texStep;   // :3953
evalPoint    += evalStep;  // :3954
```

`cellToPointTextureCoord` is `texCoord * scale + offset` (:3092), applied to
`texLocalPos` once to seed `evalPoint`, and then **again every step** through
`evalStep = texStep * ctpScale`.

## 2. Why this yields the ~1-ulp evalPoint difference

Three independent arithmetic deviations accumulate into the same observed
signature (sample-lattice comb at grazing ring edges, crop-fence shifts of
~half a texel, evalPoint values differing from the GL lattice by ~1 ulp per
axis):

1. **The step itself differs in the last ulp.** OpenGL computes
   `g_dirStep = mat(rayDir) · sampleDistance` (no normalization). Metal
   computes `stepSize = sampleDistance · maxBound / length(rayDir·boundsSize)`
   — a division by a per-ray, per-pixel `length`. Even for near-axis rays the
   two step vectors differ by ulps; for off-axis rays in a non-cubic volume the
   normalization is not merely an ulp change but a different step-length
   scaling. `length(rayDirNormSpace * boundsSize)` varies per pixel, so
   `stepSize` — and therefore every subsequent accumulator — carries a
   per-pixel ulp perturbation that OpenGL's step does not have.

2. **Three chained float accumulators instead of one.** OpenGL rounds once per
   step (`g_dataPos += g_dirStep`). Metal rounds three times per step
   (`currentPoint += stepVec`, then `texLocalPos += texStep`, then
   `evalPoint += evalStep`), and the three accumulators are seeded/advanced
   independently rather than derived from a single position. Repeated additions
   drift; because the incremental step differs from GL's in the last ulp, the
   drift is a coherent phase error per axis, not uncorrelated noise.

3. **Cell-to-point is re-applied per sample.** OpenGL applies the cell-to-point
   shift once, in the vertex shader (`ip_textureCoords`). Metal seeds
   `evalPoint` with `cellToPointTextureCoord(texLocalPos, ...)` and then adds
   `evalStep = texStep * ctpScale` — a second multiply-by-`ctpScale` per step
   that has no GL counterpart and adds its own rounding.

The crop-fence comb observed at Update2 §3.1 and the grazing-ring-edge pixels
are the same defect: the Metal sample lattice drifts coherently off the GL
`g_dataPos` lattice, and wherever the composed opacity is steeply varying in a
half-texel window (grazing ring edges, crop fences) that ~1-ulp lattice
difference becomes visible as a per-pixel delta.

## 3. Suggested fix direction

Collapse Metal's march to OpenGL's architecture:

- Seed a **single texture-space accumulator** from the same
  cell-to-point-shifted origin used by GL (the `ip_textureCoords` equivalent),
  instead of carrying world `currentPoint` and deriving `texLocalPos` /
  `evalPoint` from it.
- Advance it with `+= dataset→texture(rayDir) · sampleDistance`, i.e. drop the
  per-ray `length` normalization in `physicalSampleStep` (or re-derive it so
  the arithmetic is bit-identical to GL's `g_dirStep`), and stop re-multiplying
  by `ctpScale` per step (apply the shift once at the seed).
- Keep `currentT` as an integer counter (GL `++g_currentT`) so the step count
  and termination threshold match; the accumulated position then stays within
  one rounding of GL's `g_dataPos` at every sample index.

Note: the direction-normalization in `physicalSampleStep` (metal:3000-3006)
was introduced so the physical step stays `sampleDistance` for the
pre-integration factor. Any fix must keep the **physical step length** the same
as GL's while making the **accumulation arithmetic** identical — the two are
separable: fix the per-step vector to GL's `g_dirStep` form and, if the
pre-integration factor needs the normalized length, compute it as a separate
scalar rather than folding it into the accumulated step.

## 4. Current state

- No shader/code change is included in this document; this is a root-cause
  trace only.
- Working tree otherwise carries the Update3 gf-LUT change (uncommitted).
- All line references are to the current working tree:
  `vtkVolumeShaderComposer.h:418,437-439,3307,3310-3311`,
  `raycasterfs.glsl:302`, `MetalShaders.metal:3007-3014,3092,3795,3951-3954`.

## 5. Remaining work

1. Implement the single-accumulator fix per §3 and relink the test driver.
2. Re-capture the variant table (procedures doc *Image tests*) plus the GL
   deterministic re-capture check; expect the Update3 §2 residual floor
   (0.3–0.6 `mean|Δ|`, ≤360 masked px at default step) to drop, in particular
   the grazing-ring-edge and crop-fence comb pixels.
3. Confirm the pre-integration factor's step-length assumption still holds
   after de-normalizing the step (the §3 note).
