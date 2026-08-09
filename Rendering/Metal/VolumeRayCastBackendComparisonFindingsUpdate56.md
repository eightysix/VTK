# BREAKTHROUGH: ±1 field is the GL background blend (0.1 gray, u8→26/255, ONE/ONE_MINUS_SRC_ALPHA); clean-GL float == Metal float (d=0); float-FBO RGB=0 was self-inflicted texture clobber (update 56)

**Date:** 2026-08-09
**Scope:** Resolving update 55 §3 (float-FBO RGB=0), measuring clean GL's true pre-store float for the whole field, and identifying the true cause of the 63,692-px ±1 field.
**Target (unchanged):** Metal bit-identical to **clean GL** (`RenderingBackend=OpenGL`, no debug injection).

**Follows:** [Update 55](VolumeRayCastBackendComparisonFindingsUpdate55.md), [54](VolumeRayCastBackendComparisonFindingsUpdate54.md), [53](VolumeRayCastBackendComparisonFindingsUpdate53.md), [48](VolumeRayCastBackendComparisonFindingsUpdate48.md).

---

## 1. Update-55 float-FBO "RGB=0" mystery SOLVED — self-inflicted texture clobber

- `DumpCleanGLFloats` created its RGBA32F texture with a bare `glBindTexture(GL_TEXTURE_2D, tex)` on the **active texture unit (GL_TEXTURE1)**, silently replacing a shader sampler 2D texture that lived there — the **color transfer-function table**.
- Evidence: control-clear test (colored `glClear` into the FBO read back `(0.25,0.5,0.75,0.25)` → FBO RGB write/read path is fine); per-unit texture dump showed the active unit and the pre-existing 2D binding; with **save/restore of the active-unit 2D binding** the volume render now reads back `(0.933820307, 0.751776278, 0.621985435, 0.996922791)` at (422,419) — correct RGB **and** alpha. (Alpha survived the clobber in update 55 because the opacity table is on a different unit; RGB came back zero because the color table was replaced by the empty dump texture.)
- Result: full 512×512 clean-GL **pre-store float** dump, `VTK_GL_FLOAT_DUMP` (rows bottom-up; image is the y-flip). Store model note: `round_half_even(gl_float*255)` matches `fix_gl` only 75.5% — the GL store is *not* plain half-even of the raw composite (see §3).

## 2. CLEAN GL FINAL FLOAT == METAL MODEL FLOAT (d = 0)

- `round_half_even(gl_float*255)` vs `fix_metal.png`: **99.7% exact (785,397/786,432 channels)**, 99.99% any-channel. The remaining 0.3% are knife-edge outliers.
- ⇒ The update-54/55 hypothesis of a uniform additive excess **d ≈ 0.0984 u8 in Metal's float is WRONG**. `d=0` (Metal float == clean GL float) matches; `d=0.0984u8` matches only 73.7%.
- `center_u8 = fix_metal/255 − gl_float*255` is pure ±0.5 quantization noise (std 0.288, median ≈ 0.0016), i.e. **Metal stores the same float GL does, with round-half-even**.

## 3. THE ±1 FIELD IS THE GL BACKGROUND BLEND (root cause found)

- The test sets `ren->SetBackground(0.1, 0.1, 0.1)` (TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter.cxx:95).
- GL composites the volume with `GL_ONE, GL_ONE_MINUS_SRC_ALPHA` over the framebuffer, which was cleared to the background stored as **u8 = round(0.1*255) = 26** ⇒ dst.rgb = **26/255** (0.10196078…).
- So GL stores `gf.rgb + (26/255)·(1 − gf.a)` per channel, then GPU float→u8.
- Model check: `round_half_even((gf + bg·(1−a))·255)` with bg=26/255 matches `fix_gl.png` at **99.83% exact (785,841/786,432)**, 100% any-channel. A sweep found the optimum at **exactly bg = 26/255** (25.5/255 ⇒ 0.9879, 0.1 ⇒ 0.9930). No free parameters.
- This EXPLAINS all update-54/55 anomalies:
  - the "uniform d" interval [0.0955, 0.1012] u8 at the 15 gated px = `(26/255)·(1−a)` with a ≈ 0.9961–0.997;
  - E1's overshoots/undershoots at constant d = the alpha-dependence of the blend term (a ranges 0.979–0.998 field-wide ⇒ term ranges 0.06–0.53 u8);
  - value-dependent flip rates = frac distribution vs the (1−a)-scaled shift;
  - update 54's "input not arithmetic / constant not in pipeline" conclusion was misleading — the offset is the **blend against the background**, applied after the composite.

## 4. Metal does NOT apply the background blend (open)

- Despite the DirectScreen/FullscreenDirect pipelines declaring blend ON (`ONE, ONE_MINUS_SRC_ALPHA`, vtkMetalGPUVolumeRayCastMapper.mm:6047-6058) and vtkMetalRenderer clearing colorTarget to the background and the volume pass loading it, **Metal's stored bytes match the UNBLENDED gf (99.7%)**, not the blend model (75.7%).
- Why Metal's blend contributes nothing is **unresolved** (see §6). The practical consequence: the required Metal fix is either (a) replicate the blend term in the finalize (`finalColor.rgb += bg·(1−a)`, bg=26/255), or (b) find and fix why Metal's fixed-function blend isn't picking up the cleared background.

## 5. Residual vs fix_gl with the blend model

- 0.17% of channels still differ at bg=26/255: 251 (−1), 272 (+1), and **42 px with |Δ|>1** (knife-edge outliers: e.g. (397,110) +7/+8, (350,5) +1/+2, (54,6) +2/+2, (9,18) −2/−2, a run along y=64 with +2/+2).
- These are the same knife-edge set as updates 48/54/55 (grid-aligned rays), not part of the blend term.

## 6. Doubts / hypotheses (unresolved)

1. **Why does Metal's declared blend contribute nothing?** Candidates: (a) the volume actually renders through an offscreen RGBA16Float pipeline (blend OFF) that is then blitted *without* blending, so dst is never the background; (b) `colorTarget`/drawable at the volume pass does not contain the background (loads black); (c) the blend's effective src.a is 1.0. Needs a Metal-side probe (log which pipeline type is used + dump a pre/post-blend probe pixel), or simply the empirical fix (a) and a field diff.
2. **Exact GL float→u8 conversion** still open: `round_half_even` of the blend result matches 99.83% but the residual ±1 (523 channels) suggests a subtle conversion/rounding or float32-vs-double difference in the GPU blend path.
3. **The 42 knife-edge px** (|Δ|≥2) remain the only >1-byte deviations; Metal will need the same mechanism handled (update 48 grid-aligned-ray amplification) after the blend term is in.
4. **Fix design**: in-shader `finalColor.rgb += bg·(1−a)` bakes in the "pure background behind volume" assumption (correct for this test; GL's true dst is whatever is in the framebuffer). The proper fix is to make Metal's blend actually read the dst (fix (a)/(b) above) — equivalent result here, but general for volume-over-geometry scenes.

## Artifacts

- Code: `Rendering/VolumeOpenGL2/vtkOpenGLGPUVolumeRayCastMapper.cxx` (DumpCleanGLFloats now non-invasive: active-unit 2D binding save/restore, colored-clear control test, per-unit 2D/3D texture dump via `VTK_GL_FLOAT_DUMP_TEXTURES`, `glReadPixels` cross-check).
- Data: `/tmp/bc/u55c_gl_float.raw` + `.npy` (512×512 RGBA32F clean GL pre-store float, rows bottom-up; image = y-flip), `/tmp/bc/u55c_d_intervals.npz`, `/tmp/bc/u55c_analyze_d.py`.
- Baseline images: `/tmp/bc/fix_gl.png`, `/tmp/bc/fix_metal.png` (63,692 px, authoritative).
- Verified models (python, /tmp/bc): `round_half_even(gf·255)` ⇒ fix_metal 99.70%; `round_half_even((gf + 26/255·(1−a))·255)` ⇒ fix_gl 99.83% (bg sweep optimum = exactly 26/255).
