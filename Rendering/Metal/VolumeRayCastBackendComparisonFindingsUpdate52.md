# Metal-side bisect of clean GL's compile divergence, step 3: full TF-table dump closes the pre-integration-factor last-ulp gap (190 alpha entries, RGB clean), but the 63,692-px ±1 field survives — the table was a real but immaterial bug (update 52)

**Date:** 2026-08-09
**Scope:** Continue update 51 §6.1 (TF table last-ulp) and §6.2 (one-extra-sample). This session (1) dumps the **full 1024-entry tables from both backends** (new env-gated `VTK_GL_OPTABLE_DUMP` logging in `vtkOpenGLVolumeOpacityTable`, `vtkOpenGLVolumeRGBTable`, and `vtkMetalGPUVolumeRayCastMapper`) and finds the RGB tables **byte-identical (0/1024)** but **190 alpha entries differing by exactly 1 ulp**; (2) root-causes the alpha gap to a **float32-vs-double sample-distance** in the pre-integration factor: GL casts `minWorldSpacing` to `float` before `factor = sampleDistance/unitDistance` (`vtkOpenGLGPUVolumeRayCastMapper.cxx:1710`), Metal kept the full double (`vtkMetalGPUVolumeRayCastMapper.mm:6544`); recovered ratios `f_gl/f_mt ≈ 1.00000009` confirm a ≤1-ulp factor shift; (3) **fixes Metal** to `static_cast<float>(minWorldSpacing)` — after rebuild the alpha tables are **0/1024 differing**; (4) but the rendered ±1 field is **unchanged (63,692 px)**, and only **9 pixels** moved in Metal's own image — the table bug was real but immaterial to the field; (5) re-confirms the update-48 float32 replay still reproduces the new Metal image 68/68. The ±1 field therefore lives in a **per-sample or store-level difference still unaccounted for** — the strongest remaining lead is the one-extra-tail-sample / opacity-break strictness + the store-rounding direction, since per-channel deltas at (93,201) (G +4e-5, B +1.6e-4, R <+0.0016) are far below any single-texel flip (~0.01).
**Target (unchanged):** Metal output must be bit-identical to **clean GL** (`RenderingBackend=OpenGL` without debug injection).

**Follows:** [Update 51](VolumeRayCastBackendComparisonFindingsUpdate51.md), [Update 50](VolumeRayCastBackendComparisonFindingsUpdate50.md), [Update 49](VolumeRayCastBackendComparisonFindingsUpdate49.md), [Update 48](VolumeRayCastBackendComparisonFindingsUpdate48.md).

---

## 1. Full-table dump: RGB identical, alpha off by 1 ulp on 190 entries

Extended the OPTABLE debug logging (now env-gated by `VTK_GL_OPTABLE_DUMP`, shared by both backends) to hex-dump all 1024 float32 entries:

- `vtkOpenGLVolumeOpacityTable.cxx` → `GL_OPTABLE_DUMP idx=… a=…` (post-correction, i.e. what is uploaded).
- `vtkOpenGLVolumeRGBTable.cxx` → `GL_RGBTABLE_DUMP idx=… rgb=…` (post-GetTable).
- `vtkMetalGPUVolumeRayCastMapper.mm` → `MTL_OPTABLE_DUMP idx=… rgba=…` (post-fill, what is uploaded).

Captured `/tmp/bc/opttab_gl.log` and `/tmp/bc/opttab_metal.log`:

- **RGB: 0/1024 mismatches** → `GetTable` output and upload are byte-identical between backends (also rules out any color-TF range/width/state difference).
- **Alpha: 190/1024 mismatches**, each exactly ±1 float ulp (e.g. `d58e4138` vs `d48e4138`). All mismatches are on **pre-integration-corrected** entries (raw `> 0.0001f`); the RGB channel has no correction and never differs.

## 2. Root cause: float32-vs-double sample distance in the pre-integration factor

Both backends pre-integrate with the same libc `pow` on the same CPU (identical formula `(float)(1 - pow(1 - (double)a, factor))`, identical raw inputs — RGB proof), so the only free variable is `factor = sampleDistance / unitDistance`:

- **GL:** `vtkOpenGLGPUVolumeRayCastMapper.cxx:1710` — `this->ActualSampleDistance = static_cast<float>(minWorldSpacing);` then the table is built with `factor = (double)ActualSampleDistance / unitDistance`.
- **Metal:** `vtkMetalGPUVolumeRayCastMapper.mm:6544` (pre-fix) — `actualSampleDistance = minWorldSpacing;` (full double kept) → `factor = minWorldSpacing / unitDistance`.

Recovered per-entry ratios `r = ln(1-α_GL)/ln(1-α_MT) = f_GL/f_MT`: all 190 are `> 1`, mean `1.0000000859`, max dev `1.25e-7` — consistent with `f_GL = f32(f_MT)` (single float ulp at 0.27 ≈ 2.2e-7 relative), direction = GL rounds up. This is the exact mechanism.

## 3. Fix applied and verified at the table level

```cpp
// vtkMetalGPUVolumeRayCastMapper.mm (autoAdjust branch)
actualSampleDistance = static_cast<float>(minWorldSpacing);
if (this->ReductionFactor < 1.0 && this->ReductionFactor != 0.0)
  actualSampleDistance /= static_cast<float>(this->ReductionFactor);
```

Rebuilt (`./macos_metal_build.sh --resume --tests`), re-dumped: **alpha entries differing after fix: 0/1024**. Metal's uploaded TF table is now byte-identical to GL's.

## 4. The rendered field is unchanged — the table was immaterial to it

- Metal-vs-GL rendered diff after fix: **63,692 px** (was 63,690; ±2 px noise), still GL ≥ Metal on the overwhelming majority, still ±1 LSB mostly (max channel diff 15 at isolated pixels).
- Metal old-vs-new image diff: only **9 pixels** changed → the factor bug was real but shifts almost nothing (the affected alpha deltas are ~1 ulp on low/medium entries and vanish into the round-half-even store).
- The update-48 CPU float32 replay still reproduces the **new** Metal image **68/68** (same lattice/TF/break model; the model's hardcoded factor 0.270059 is within the f32 value and indistinguishable at the stored LSB).

## 5. Conclusion

- TF tables are now **conclusively byte-identical** (RGB 0/1024, alpha 0/1024 after fix) → update 51 §6.1 closed.
- The divergence does **not** live in: scale, tables, single-step arithmetic, whole-loop reassociation, position/step/texel lattice (all closed, updates 48–52), or the pre-integration factor (now closed).
- The ±1 field needs a small *positive* per-sample or terminal bias (per-channel: (93,201) needs G +4e-5, B +1.6e-4, R < +0.0016) — far smaller than any single-texel flip (~0.01), so **not** a knife-edge/texel mechanism; a uniform bias is also excluded (update 51: uniform opacity scale only partially matches).

## 6. Current doubts / hypotheses (unresolved)

1. **Opacity-break strictness / one-extra-tail-sample (leading).** GL breaks on `g_fragColor.a > g_opacityThreshold` (strict, composer:3371); Metal on `accumulatedOpacity >= 1.0f - 1.0f/255.0f` (non-strict, MetalShaders.metal:4814). If at some ray GL's accA lands exactly on the threshold it continues for one more sample (and the extra tail sample's ~(0.0009,0.0008,0.0007) reproduces (93,201)=(247,171,131) exactly, update 51 §3); with strict-vs-non-strict, Metal stops one sample early. The 36-combo sweep (update 51 §2.1) pins the alpha chain, so this only fires when accA hits the threshold **exactly** in float32 — rare, but 63,692 field pixels is a lot of rays; needs a numerical check of how often `accA` lands within 1 ulp of 0.996078 on bright rays.
2. **Store rounding direction (update 48 fallback, now revived).** The per-channel deltas are all consistent with "same float value, different u8 conversion": if Metal's BGRA8Unorm store rounds half-even (as the 68/68 replay models) but GL's fixed-function conversion rounds half-**up** (ties away), then any float at `X.5` rounds to `X+1` in GL and `X` or `X+1` in Metal. With ~63k differing pixels all `GL = Metal+1`, this requires the float values to sit *at* `X.5` for a large fraction of pixels — implausible for a generic 4e-5-ulps field, but it is the only one-directional per-store mechanism not yet experimentally excluded (e.g. capture via an RGBA16F intermediate to lift the u8 quantization and compare the floats directly).
3. **Remaining shader-side structural differences (background).** GL composites `g_fragColor.rgb *= g_fragColor.a` then `(1-a)*src + frag` (composer:2651-52); Metal uses `fma(weight, sampleColor*sampleOpacity, accumulatedColor)` — the single-step variants were proven bit-identical on the CPU model (update 51 §2), but the *hardware* contraction of the GPU compiler is not directly observable for clean GL; kept only as a last resort.

## Artifacts

- Dump code: `VTK_GL_OPTABLE_DUMP` env-gated hex dumps in `vtkOpenGLVolumeOpacityTable.cxx` (a=…), `vtkOpenGLVolumeRGBTable.cxx` (rgb=…), `vtkMetalGPUVolumeRayCastMapper.mm` (rgba=…).
- Fix: `vtkMetalGPUVolumeRayCastMapper.mm:6544-6549` — `static_cast<float>(minWorldSpacing)` (autoAdjust branch).
- Captures: `/tmp/bc/opttab_gl.log`, `/tmp/bc/opttab_metal.log` (pre-fix), `/tmp/bc/opttab_metal2.log` (post-fix, 0/1024 alpha diffs).
- Images: `/tmp/bc/fix_metal.png`, `/tmp/bc/fix_gl.png` (post-fix comparison pair; 63,692 diff px).
- `/tmp/bc/u47_metal.png` overwritten with the post-fix Metal image; update-48 `replay_metal_accumulate.py` still 68/68.
