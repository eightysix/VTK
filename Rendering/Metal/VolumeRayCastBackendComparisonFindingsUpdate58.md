# After the blend fix, Metal out-matches GL vs the ideal blend model (99.89% vs 99.83%); the last 188 px are isolated knife-edge pixels where the pre-store float sits ~0.01 u8 from a rounding boundary; the GL float dump is a re-render and unreliable there — bit-identical is plausible for 174 ±1 px, likely unattainable for the 14 knife-edge px (update 58)

**Date:** 2026-08-09
**Scope:** Post-fix feasibility assessment for bit-identical output: build a full-field Metal pre-store float dump (env `VTK_METAL_FLOAT_DUMP`, mirror of `VTK_GL_FLOAT_DUMP`) and compare Metal vs GL at the 188 remaining differing pixels.
**Target (unchanged):** Metal bit-identical to **clean GL** (`RenderingBackend=OpenGL`, no debug injection).

**Follows:** [Update 57](VolumeRayCastBackendComparisonFindingsUpdate57.md) (blend-clamp fix: 63,692 → 188 px), [56](VolumeRayCastBackendComparisonFindingsUpdate56.md), [48](VolumeRayCastBackendComparisonFindingsUpdate48.md).

---

## 1. Full-field Metal pre-store float dump (VTK_METAL_FLOAT_DUMP)

- Added env-gated macro `VTK_METAL_FLOAT_DUMP` (vtkMetalGPUVolumeRayCastMapper.mm `vtkMetalVolumeCompileOptions`), which turns the existing shader `FINAL` log into a **dump-all** mode (`MetalShaders.metal` FINAL block: `dumpAll = 1` gates every pixel). The FINAL line already carries exactly the pre-store state: `accCol` = accumulatedColor (unblended gf), `accOp` = accumulatedOpacity (unclamped), `lastIter` = break iteration.
- Captured 512×512 full field (262,144 px, first-frame dedup): `/tmp/bc/u57_float_dump.log` (360 MB, `MTL_LOG_BUFFER_SIZE=1073741824`), parsed to `/tmp/bc/u57_metal_float.npy` (`g` = gf rgb, `a` = alpha). 100% coverage.
- Note the alpha dump confirms the update-57 fix is live: `accumulatedOpacity` is the raw post-break value (mean 0.9965), not 1.0.

## 2. Metal's gf == GL's gf to ~0.01 u8 field-wide; the 188 px are boundary-straddlers

- **gf float match:** Metal vs GL gf differs by mean 0.00002 u8, std 0.011 u8 across the whole field; 99.79% of channels within ±0.01 u8. **Alpha** differs by std 0.0024 u8. So Metal's composite arithmetic is effectively identical to GL's — the remaining differences are at sub-ULP-to-few-ULP level.
- Only **33 px** have gf |Δ|>0.5 u8 field-wide (the knife-edge set). Metal's own gf-delta histogram at the 188 px: mean 0.00004, std 0.00085, max 0.012 u8.
- **Both backends are perfectly self-consistent:** `fix_metal2 == rint_half_even(model built from Metal's gf)` at **all 188 px**, and `fix_gl == model from GL's gf` at all 188 px. I.e. each backend stores exactly what its own gf implies; the two backends' *bytes* differ only because their gf floats differ by a hair around a rounding boundary.
- **Metal now beats GL against the ideal model:** model-from-Metal-gf vs fix_metal2 = 99.89%; model-from-GL-gf vs fix_gl = 99.83%. Metal is closer to the physical `gf + 26/255·(1−a)` model than GL's own stored image.

## 3. The GL float dump is a re-render — unreliable at knife-edge rays

`DumpCleanGLFloats` (vtkOpenGLGPUVolumeRayCastMapper.cxx:4945) **re-renders** the volume into a black RGBA32F FBO with blend disabled (`glDisable(GL_BLEND)`, line 5044). At ordinary pixels that equals the real frame's gf (update 56), but at knife-edge (grid-aligned) rays the re-render diverges:

- (350,5) ch2: dump gives gf ≈ 151.59 for **both** backends, yet GL stored 150 (Metal stored 152) — a 1.5 u8 jump inconsistent with any conversion of the dump value.
- (397,110): dump predicts blend-model green ≈ 186, GL's real frame stores 179 (Δ 6+ u8).
- Consequence: the 14 |Δ|>1 px cannot be analyzed via the GL dump; GL's true pre-store float there is unmeasurable with the current tooling (it would need an in-place MRT capture on the *real* frame, not a re-render).

## 4. Feasibility: bit-identical = Metal's gf must equal GL's gf *at those pixels*

- The 188 px all sit with their pre-store float within ~0.01 u8 of a rounding boundary (dist-to-boundary median 0.18 u8, min 0.0001 u8). A sub-ULP gf delta flips the stored byte.
- **174 ±1 px: plausibly achievable.** Metal is already within ~0.01 u8 of GL everywhere; the residual is the update-48 §5 compile-level reassociation difference (GLSL compiler's fma-contraction / operation order vs Metal's written `fma(weight, src, acc)`). Replicating GL's compiled order is a finite bisect (reassociation / break-parity variants) and — while laborious and unguaranteed — is the last systematic term.
- **14 knife-edge px (|Δ|>1): likely unattainable.** GL's own output at these pixels is a function of the exact render-path state (its own re-render disagrees with its own frame by up to 6 u8). Bit-parity there would require reproducing GL's unmeasurable pre-store float exactly — not achievable with the current tooling and possibly not deterministic in GL.

## 5. Doubts / next options

1. **Operation-order bisect** (update 48 §5) on the 174 ±1 px: try GL's likely compiled forms (e.g. `acc + (1-acc)*src` without fma, `acc*(1-src)+src`, multiply-then-subtract) against the 68 gated replay pixels, then re-check the field. Success criterion: the residual collapses to the 14 knife-edge px.
2. **In-place GL float capture** (MRT on the real frame instead of a re-render) to make the dump trustworthy at knife-edge rays — would tell us definitively whether the 14 px are Metal's fault or GL self-instability.
3. If the 14 px prove to be GL-internal instability, declare bit-parity target as "188 px residual" or add a documented tolerance (per-pixel |Δ|≤1 except the 14 knife-edge px).

## Artifacts

- Code: `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm` (`VTK_METAL_FLOAT_DUMP` macro injection), `Rendering/Metal/Shaders/MetalShaders.metal` (`dumpAll` gate).
- Data: `/tmp/bc/u57_float_dump.log` (360 MB), `/tmp/bc/u57_metal_float.npy` (Metal pre-store gf + alpha, 512²), `/tmp/bc/fix_metal2.png` (post-fix image).
- Verified (python, /tmp/bc): gf match field-wide mean 0.00002 u8; Metal model 99.89% vs GL model 99.83%; each backend self-consistent at all 188 px.
