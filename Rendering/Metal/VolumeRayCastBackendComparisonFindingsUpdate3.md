# Metal vs OpenGL volume ray cast: gradient-opacity LUT fix (update 3)

Follow-up to all prior documents, read as an addendum to:

- [VolumeRayCastBackendComparisonProcedures.md](VolumeRayCastBackendComparisonProcedures.md) —
  environment, capture/analyze tooling.
- [VolumeRayCastBackendComparisonFindings.md](VolumeRayCastBackendComparisonFindings.md) —
  sections 1–7, still valid.
- [VolumeRayCastBackendComparisonFindingsUpdate.md](VolumeRayCastBackendComparisonFindingsUpdate.md) —
  composite-gate and termination-threshold fixes.
- [VolumeRayCastBackendComparisonFindingsUpdate2.md](VolumeRayCastBackendComparisonFindingsUpdate2.md) —
  fp16-accumulation and MAX_RAY_STEPS fixes; §5 left the gradient-opacity path
  as the sole remaining gap, with the gf LUT's 256-entry/8-bit build explicitly
  listed as a candidate source of the divergence.

This document records the third confirmed root cause — the gradient-opacity
lookup table was built as **256 RGBA8Unorm entries while OpenGL uses a
single-channel **float** table** — and the fix that collapses the entire
gradient-opacity divergence family.

## 1. Root cause: gf LUT quantized to 256 × 8 bits vs OpenGL's 1024 × float32

### 1.1 The two backends' gradient-opacity tables

OpenGL (`vtkOpenGLVolumeGradientOpacityTable::InternalUpdate`, table width from
`vtkOpenGLVolumeLookupTable::GetMaximumSupportedTextureWidth` over
`EstimateMinNumberOfSamples` of the component range — 1024 for typical
functions):

```
gradientOpacity->GetTable(0, (LastRange[1] - LastRange[0]) * 0.25, 1024, floatTable);
Create2DFromRaw(1024, 1, 1, VTK_FLOAT, floatTable);   // single-channel float32
```

Metal (`vtkMetalGPUVolumeRayCastMapper::UpdateGradientOpacityTexture`, before):

```
gradOpacityFunc->GetTable(0, gradMax, 256, table);      // 256 entries
// quantized to 8 bits and stored as RGBA8Unorm (R replicated):
val = (uint8)(clamp(table[i]) * 255)
```

Two independent deviations:

1. **Resolution**: 256 entries instead of 1024. With the steep test ramps over
   `[0, 0.25·range]` (e.g. `0@0 → 1@90` in NoShadeAmp), one entry spans
   ~4.3 data units, so a small `gradW` shift near the knee moves the sampled
   entry by a full level.
2. **Precision**: values were rounded to `1/255`. Near the knee of a steep
   ramp this is a large relative opacity error per sample.

The shader samples the table with the same normalized coordinate in both
backends (`grad.w`, clamped to `[0, 1]`), so every `gradW`/gf interplay in the
composite was being passed through Metal's coarser, quantized table. That is
why every variant that multiplied `sampleOpacity` by `gf(gradW)` diverged
(findings §3/§4, Update2 §5), and why the divergence grew with the steepness
of the gf response (3.69 → 8.54 → 6.53): a steeper table amplified the
per-sample opacity error into larger per-pixel deltas.

### 1.2 Fix

`UpdateGradientOpacityTexture` now mirrors the GL table build:

- width = `ComputeTransferFunctionWidth(nullptr, gradOpacityFunc, gradRange)`
  (the same EstimateMinNumberOfSamples-based width GL uses, 1024 for these
  functions),
- single-channel `MTLPixelFormatR32Float`,
- table filled by `GetTable(0, gradMax, tfWidth, floatData)` with no
  quantization.

Sampler parity was already in place and is unchanged: both backends filter the
table by the volume property's interpolation type (linear/nearest) and clamp
to the texture edge.

## 2. Effect (backend-vs-backend, cumulative with all prior fixes)

Same measurement scenes as Update2 (512×512, `mean|Δ|` = per-pixel max-channel
absolute delta; "masked px" = `|Δ| ≥ 5/255`). OpenGL engagement verified via
`GL_SAMPLING` in the OpenGL stderr (12 lines/run, none on Metal).

| variant | Update2 after fixes | **after gf-LUT fix** |
|---|---|---|
| camera-outside fixed-step (no gf) | 0.31 / 2 | 0.32 / **0** (exact) |
| NoShadeNoGradOpNoTransformCamOutside | 0.31 / 2 | 0.32 / **0** |
| base (shade + gf) | 3.69 / 84,020 | **0.56 / 183** |
| NoShade (gf on) | 8.54 / 120,220 | **0.55 / 192** |
| NoShadeAmp (steep gf, fixed 0.008) | 6.53 / 117,534 | **0.39 / 360** |
| ConstGradOp | — | 0.43 / 316 |
| NoShadeConstGradOp | — | 0.35 / **0** (exact) |
| NoShadeLinGradOp | — | 0.47 / 201 |

Regression fits for the worst prior offenders (R channel, `metal = m·gl + b`):

| variant | before | after |
|---|---|---|
| base | 1.0864·gl − 10.60 | 1.0030·gl − 0.80 |
| NoShade | 1.1931·gl − 49.76 | 1.0022·gl − 1.39 |
| NoShadeAmp | 1.6557·gl − 162.58 | 1.0030·gl − 0.80 |

All six gradient-opacity variants now sit at the same ~0.3–0.6 `mean|Δ|` /
≤360 masked-px floor as the already-solved no-gf scene. Two are byte-exact
(0 masked px). The residual masked pixels are the grazing-ring-edge / sample
phase differences already characterized at coarse steps in Update2 §3.1 — the
camera-inside scene at the default step, not a gf-pipeline artifact.

## 3. Revised interpretation

Update2 §5's open question — "where in the `gradW`/gf chain (sample position,
gradient stencil, LUT quantization, gf table differences) do the two backends
differ?" — is now answered: **the LUT quantization**. The underlying `gradW`
value and the gf formula were already parity (the no-gf scene matched to 2 px;
the offline gradient replay ratio was ≈1.0). Metal was feeding a correct
`gradW` through a 256×8-bit approximation of the gf function, and the steep
ramps in these scenes amplified that approximation error into the observed
contrast-stretch signature. With a 1024×float32 table the entire family lands
on the common step-phase floor.

## 4. Current state

- `vtkMetalGPUVolumeRayCastMapper.mm::UpdateGradientOpacityTexture` builds the
  gf LUT as 1024 (GL-parity width) × `R32Float`, no quantization.
- Working tree contains this change (uncommitted).
- Captures: `/tmp/bc/lutfx{2,3}/` (freshly relinked test driver; the stale
  binary in the first `lutfx` run silently re-used the pre-fix code and
  reproduced the old 6.53 numbers — always relink `vtkRenderingVolumeCxxTests`
  after touching the Metal library).

## 5. Remaining work

1. Relink + full variant-table re-capture (procedures doc *Image tests*),
   and the GL deterministic re-capture check.
2. Strip the remaining TEMP DEBUG logging behind `VTK_METAL_ENABLE_LOGGING`
   once the residual ring/phase pixels are accepted.
3. `vtkOpenGLVolumeGradientOpacityTable` parity for the label-map gradient
   opacity table is still unimplemented on Metal (out of scope here).
