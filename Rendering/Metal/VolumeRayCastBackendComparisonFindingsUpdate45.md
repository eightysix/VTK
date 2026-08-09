# Composite FMA-contraction test: explicit `fma()` in Metal's composite moves 8 px (net −2), the ±1 field is NOT the contraction either — composite arithmetic is now exhaustively ruled out; next candidate is the TF-lookup coordinate/filter convention (update 45)

**Date:** 2026-08-09
**Scope:** Test the update-44 hypothesis that clean GL's ±1 field is a compile-level asymmetry in the composite block. Update 36 already established this Apple GLSL driver contracts `a*b + c` chains as FMA (`mul, fma, fma, mul+add`), and GL's composite `g_fragColor = (1.0f - g_fragColor.a) * g_srcColor + g_fragColor` (all 4 channels, `vtkVolumeShaderComposer.h:2652`) is exactly such a chain. Metal, built with `fastMathEnabled = NO` (update 36), does **not** auto-contract, so its composite does 3 roundings/channel (premultiply, weight-multiply, add) where GL's contracted compile does 2 (premultiply + fused). If the ±1 field were the contraction asymmetry, forcing Metal to emit the fused form would collapse it.

**Follows:** [Update 44](VolumeRayCastBackendComparisonFindingsUpdate44.md), [Update 43](VolumeRayCastBackendComparisonFindingsUpdate43.md), [Update 36](VolumeRayCastBackendComparisonFindingsUpdate36.md).

---

## 1. The change (kept: semantically matches GL's compiled form)

`MetalShaders.metal:4794-4795` main composite path:

```metal
// before
accumulatedColor += weight * (sampleColor * sampleOpacity);
accumulatedOpacity += weight * sampleOpacity;
// after  — explicit fused multiply-add, matching GLSL's likely-contracted
//          (1.0f - g_fragColor.a) * g_srcColor + g_fragColor  (GLSL compiles a*b+c as fma)
accumulatedColor = fma(weight, sampleColor * sampleOpacity, accumulatedColor);
accumulatedOpacity = fma(weight, sampleOpacity, accumulatedOpacity);
```

Kept because: (a) with `fastMathEnabled=NO` the compiler cannot add the fusion on its own, so explicit `fma()` is the only way to match GL's compiled arithmetic; (b) it reduces Metal from 3 roundings/channel to 2 — the same rounding count as the written GL source; (c) net pixel effect is an improvement with no regression (see §2).

## 2. Measured effect: 8 px, net −2

Clean Metal capture with the fma composite (`/tmp/bc/u45_fma_metal.png`, no debug env) vs clean GL (`fix42_gl.png`, deterministic):

| metric | non-fma Metal | fma Metal |
|---|---|---|
| px differing vs clean GL | 63,692 | **63,690** |
| GL-higher (pure) | 63,649 | 63,647 |
| GL-lower (pure) | 39 | 39 |
| max \|d\| | 8 (397,110) | 8 (397,110, unchanged) |

The 8 px the fma moved (5 toward GL, 3 away):

```
(154,40) 175→176  TOWARD   (273,72) 167→166  away
(155,40) 175→176  TOWARD   (257,174) 154→153 away
(177,128) 158→159 TOWARD   (350,437) 172→171 away
(169,179) 162→163 TOWARD
(164,296) 158→159 TOWARD
```

Same class as update 43's ordering fix (3 px): a ±1-ULP boundary effect, not the ±1 field.

## 3. Ruled out so far (the ~3.5e-4 uniform GL>Metal bias)

- Composite multiply **ordering** `(w*c)*a` vs `w*(c*a)` (update 43): 3 px.
- Composite **FMA contraction** (this update): 8 px.
- Output pixel format (BGRA8Unorm direct path), volume texture (R16Unorm exact), TF table precision (both float32) — updates 43.

The composite block itself is now exhaustively covered: ordering × contraction, both directions measured, combined ~11 px. The systematic one-directional ~3.5e-4 bias (99.97% GL>Metal, uniform, 63,690 px) survives all of them, with per-sample raw/position equal to ≤1.3e-7/≤1e-6 (update 41/44). That forces the divergence into the **per-sample TF-lookup result** (equal scalar → different op/rgb), not the accumulation.

## 4. Next candidate: TF-lookup coordinate/filter convention

GL samples the color/opacity tables at `vec2(scalar.w, 0.0)` (`vtkVolumeShaderComposer.h:1910,1928`, `computeOpacity` likewise). Metal samples at `float2(float(scalarNorm), 0.5)` (`MetalShaders.metal:4454-4488`). Both tables are float32 RGBA 1024-wide (GL `GL_OPTABLE width=1024`; Metal `FillTransferFunctionRGBA32FWithPreIntegration`, 1024-texel floor at `vtkMetalGPUVolumeRayCastMapper.mm:1452-1464`), both sampled with linear filtering — but:

- the **v-coordinate** differs (GL `0.0` vs Metal `0.5`);
- the **u-coordinate convention** (texel-center vs edge offset) is unverified between the two backends' table uploads;
- the **normalization chain** (raw → scalarNorm) feeding u is verified equal to ≤1.3e-7 for raw but the *normalized* coordinate's float rounding is not yet compared (Metal's `DEBUG SAMPLE` logs `norm`).

A ~2e-6/sample systematic TF difference (uniform because it is data-independent w.r.t. the interpolation convention) accumulates to ~3.5e-4 over 170 samples — the observed bias. Next step: dump the exact TF sample coordinates + interpolated op/rgb from both backends per sample, normalize GL's `scalar.w` vs Metal's `norm` in float32, and compare the two `u` conventions; then make Metal's TF fetch match GL's `vec2(scalar, 0.0)` convention exactly (coordinate + v) and re-diff.

## Artifacts

- `/tmp/bc/u45_fma_metal.png`, `/tmp/bc/u45_fma_metal.log` (clean Metal, fma composite).
- Reference clean images: `/tmp/bc/fix42_gl.png`, `/tmp/bc/fix42_metal.png`.
