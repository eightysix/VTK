# Applied the composite multiply-ordering fix (GL `w·(c·a)` parity): measured effect is 3 px only — the 63,691-px ±1 field is NOT the multiply ordering; bias magnitude (~3.5e-4) points to a per-sample input precision difference, volume texture ruled out (update 43)

**Date:** 2026-08-09
**Scope:** Implement the composite-multiply ordering fix identified in [update 42](VolumeRayCastBackendComparisonFindingsUpdate42.md) (§2: GL composites `w·(c·a)` — premultiply sample color by opacity first — while Metal composites `(w·c)·a`), re-capture clean Metal, and measure. Result: the ordering difference is real and matches GL's written source semantics, but it moves only **3 pixels** (1 toward GL, 2 away) of the 63,692-px ±1 field. The bulk ±1 field has a different cause whose magnitude (~3.5e-4, ≈2^-11.5) implicates a per-sample precision source; the volume texture is ruled out (this test's USHORT data uploads as exact R16Unorm, not half).

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

- **Rules out:** the multiply ordering (measured 2.9e-6, ~120× too small); random ULP accumulation (would be ±, not 99.97% one-sided); the output pixel format (camera-inside path renders to BGRA8Unorm, same 8-bit quantization as GL; no RGBA16Float involved).
- **Consistent with half-precision somewhere per-sample:** half mantissa 10 bits ⇒ quantization ~2^-11 ≈ 4.9e-4, the right order. But the **volume texture is ruled out**: this test's scalars are USHORT 0..4370 → `vtkMetalGPUVolumeRayCastMapper.mm` uploads R16Unorm (exact, `NormalizationFactor=65535`, no half conversion). Candidate that remains: the **transfer-function table textures** (color/opacity 1D LUTs), which are sampled per sample as `half4` in Metal (`sampleTransferFunction`), vs whatever precision GL's TF table uses — a uniform ~2e-6/sample TF lookup difference would accumulate to ~3.5e-4 over 170 samples while leaving the raw-scalar match (≤1.3e-7, update-41) intact.
- The 13 hot px (worst 8 at (397,110)) are a separate, opposite-sign (GL lower) knife-edge issue; untouched by this fix.

## 4. Next steps

- Compare the **transfer-function table** precision/upload between backends (Metal TF textures as half4 vs GL table format + shader LUT fetch precision). Test by uploading Metal's TF tables as full float and re-diffing the ±1 field.
- Alternatively, re-run the per-sample accumulation-level comparison (`compare_gl_metal_accum.py`) at a representative ±1 pixel to confirm the divergence accumulates per-sample from the TF lookup rather than from the composite.

## Artifacts

- `/tmp/bc/fix42_metal.png` (Metal, clean, post-fix), `/tmp/bc/clean_gl.png` (GL, clean, reference), `/tmp/bc/clean_metal.png` (Metal, pre-fix).
- Diff summary: 63,692 px ±1 (GL>Metal 99.97%), 13 px ≥2, worst (397,110) G+7 B+8.
