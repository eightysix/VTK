# Applied the composite multiply-ordering fix (GL `w·(c·a)` parity): measured effect is 3 px only — the 63,692-px ±1 field is NOT the multiply ordering; ruled out output pixel format (BGRA8Unorm direct path), volume texture (R16Unorm), and TF table precision (both backends float32) (update 43)

**Date:** 2026-08-09
**Scope:** Implement the composite-multiply ordering fix identified in [update 42](VolumeRayCastBackendComparisonFindingsUpdate42.md) (§2: GL composites `w·(c·a)` — premultiply sample color by opacity first — while Metal composites `(w·c)·a`), re-capture clean Metal, and measure. Result: the ordering difference is real and matches GL's written source semantics, but it moves only **3 pixels** (1 toward GL, 2 away) of the 63,692-px ±1 field. The bulk ±1 field has a different cause whose magnitude (~3.5e-4, ≈2^-11.5) implicates a per-sample precision source; this update rules out the three remaining precision-candidate paths (output pixel format, volume texture, TF table precision).

**Follows:** [Update 42](VolumeRayCastBackendComparisonFindingsUpdate42.md).

---

## 1. The fix

`MetalShaders.metal` main composite path (`MetalShaders.metal:4794`):

```metal
// before
accumulatedColor += weight * sampleColor * sampleOpacity;
// after  (OpenGL vtkVolumeShaderComposer.h:2651-2652 parity:
//         g_srcColor.rgb *= g_srcColor.a;  g_fragColor = (1-a) * g_srcColor + g_fragColor;)
accumulatedColor += weight * (sampleColor * sampleOpacity);
```

Left-associative MSL evaluated the old form as `(w·c)·a`; GL explicitly premultiplies `c·a` then scales by `w`. In float32 the two round differently; update-42's replay measured ~2.9e-6 divergence at (422,92), GL higher — matching the observed image-wide sign (GL > Metal). Alpha accumulation unchanged (single float multiply, same order in both backends).

## 2. Measured effect: 3 pixels

Clean re-captures after the fix (dummy-baseline method, both backends, no debug env):

| pixel | GL | Metal before fix | Metal after fix | effect |
|---|---|---|---|---|
| (316,151) | (240,187,153) | (239,187,153) | (240,187,153) | **+1, now == GL** |
| (177,128) | (244,158,117) | (244,158,117) | (244,158,116) | −1, now GL−1 |
| (169,179) | (243,162,122) | (243,162,122) | (243,162,121) | −1, now GL−1 |

Total differing pixels: 63,692 before ≈ 63,692 after (net 0). **The multiply ordering is NOT the source of the ±1 field.**

Notes:
- The direction is not monotone (1 helped, 2 hurt): MSL may still contract `w·(c·a)+acc` into an FMA (fused `w·(c·a)+acc`, single rounding) while GL's GLSL compiler may not, or vice versa — the residual is at the ±1-ULP boundary either way.
- Keep the fix: it makes Metal match the **written** GL composite formula, which is the right reference regardless of pixel count.

## 3. What the bias magnitude rules in / out

Observed: 69,855 flipped channels / 786,432 = 8.88% per channel, GL > Metal on 99.97%, uniform across the image ⇒ a one-directional accumulated-color offset of ~0.0888 byte ≈ **3.5e-4** (~2^-11.5) sits under the byte-quantization boundary.

- **Rules out — multiply ordering:** measured 2.9e-6, ~120× too small; random ULP accumulation (would be ±, not 99.97% one-sided).
- **Rules out — output pixel format:** the camera-inside path renders through `FullscreenDirect` → **BGRA8Unorm** (`vtkMetalGPUVolumeRayCastMapper.mm:6332`), the same 8-bit quantization GL applies; the RGBA16Float offscreen targets (lines 1708-1771, 5822) are used only for the partitioned / RenderToImage / downsampled paths, not this test.
- **Rules out — volume texture precision:** this test's scalars are USHORT 0..4370 → uploaded as **R16Unorm** (exact, `NormalizationFactor=65535`, no half conversion; `vtkMetalGPUVolumeRayCastMapper.mm:731-741`).
- **Rules out — transfer-function table precision:** both backends already use full float32 tables. Metal's `ColorOpacityTexture` is built by `FillTransferFunctionRGBA32FWithPreIntegration` and uploaded as **MTLPixelFormatRGBA32Float** (`vtkMetalGPUVolumeRayCastMapper.mm:1368-1413, 3643-3667`); GL's RGB/opacity tables upload via `Create2DFromRaw(..., VTK_FLOAT, ...)` (`vtkOpenGLVolumeRGBTable.cxx:34-35`). The half `RGBA16Float` TF-table variant (which the update-43 original draft listed as the remaining candidate) was already replaced by the RGBA32F variant precisely because it "quantized every sampled opacity/color to half precision (11-bit mantissa), shifting the accumulated color by a few LSB vs the OpenGL reference" (comment at `vtkMetalGPUVolumeRayCastMapper.mm:1371-1374`). So the ~3.5e-4 bias is **not** a TF-table half-quantization.
- The 13 hot px (worst 8 at (397,110)) are a separate, opposite-sign (GL lower) knife-edge issue; untouched by this fix.

## 4. Next steps

- Re-run the per-sample **accumulation-level** comparison (`compare_gl_metal_accum.py`) at a representative ±1 pixel and at a hot pixel, comparing accumulated color/alpha per sample — pin down whether the ~3.5e-4 divergence starts at sample 0 (a per-sample input/arithmeic drift) or accumulates late (termination / last-sample weight). With the ordering, output format, volume texture, and TF table all ruled out, the remaining candidates are per-sample arithmetic differences in the march itself (position → scalar → TF coordinate → premultiply chain).
- Re-run the same comparison at (422,92) to re-confirm the 2.9e-6 ordering replay still holds under the new composite order.
- Diff the compiled clean-GL vs debug-GL fragment shaders (update-41 doubt 2) for the composite block — the debug-instrumented GL is still 60k px closer to Metal than clean GL is.

## Artifacts

- `/tmp/bc/fix42_metal.png` (Metal, clean, post-fix), `/tmp/bc/clean_gl.png` (GL, clean, reference), `/tmp/bc/clean_metal.png` (Metal, pre-fix).
- Diff summary: 63,692 px ±1 (GL>Metal 99.97%), 13 px ≥2, worst (397,110) G+7 B+8.
