# Metal's compiled composite replays the written GLSL formula exactly; the clean-GL ±1 field is a compiler-level divergence — and the whole comparison runs under NEAREST interpolation (VTK default)

**Date:** 2026-08-09
**Scope:** After the scalarScale/`in_volume_scale` parity fix was shown to change 0 pixels (no doc written for that dead end), this update rebuilds with tests ON (enabling shader os_log), re-captures both backends, and runs update 44's prescribed diagnostic: CPU float32 replay of the written GLSL composite formula against Metal's per-sample rows.
**Target (confirmed):** Metal output must be bit-identical to **clean GL** (`RenderingBackend=OpenGL` without debug injection). The written source formula is the *mechanism*, not the target — if clean GL's compiled GLSL diverges from the formula, Metal must reproduce clean GL's divergent output, not the formula.

**Follows:** [Update 45](VolumeRayCastBackendComparisonFindingsUpdate45.md), [Update 44](VolumeRayCastBackendComparisonFindingsUpdate44.md).

---

## 1. Interpolation is NEAREST, not linear

- `vtkVolumeProperty.cxx:25` ctor sets `this->InterpolationType = VTK_NEAREST_INTERPOLATION;` and the test (`TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter.cxx`) never calls `SetInterpolationType`. So `fc_linearInterpolation = 0` (`vtkMetalGPUVolumeRayCastMapper.mm:7015` → 5929) and **both the volume and the transfer function are sampled with `sNearest`** (`MetalShaders.metal:3192-3197`, 3209-3214).
- Confirmed empirically: `raw` values in the `SAMPLE` log are exact texel values (e.g. `raw=0.001678` = 110/65535, `raw=0.016571` = 1086/65535), i.e. no trilinear blending. The TF table itself is RGBA32F, so `op/rgb` are still exact floats (nearest across table texels).

## 2. Rebuilt with tests, re-captured

- `./macos_metal_build.sh --resume --tests` → `VTK_BUILD_TESTING=ON`; full rebuild including shader library. Shader `os_log` (`VTK_METAL_ENABLE_LOGGING` gated) is now live: `u47_metal.log` carries ~54k `MARCH`/`SAMPLE` rows.
- `u47_metal.png` (13:07) and `u47_gl.png` (13:07), dummy-baseline trick, factory-prefer per backend.

## 3. Findings

1. **The scalarScale fix is live and GL-parity confirmed — and still changes nothing.** Metal shader now logs `scalarMax=0.066681` (= 4370/65536); GL logs `GL_TEX scale=(14.9967966,1,1,1) bias=(0,0,0,0)`. Both backends therefore compute the TF coordinate with the same `in_volume_scale`. `u47_metal` is byte-identical to the pre-fix `u45_fma_metal` (0 px). Scale parity is refuted as the field's cause by measurement.
2. **The field is unchanged and strongly asymmetric.** `u47_metal` vs `u47_gl`: 63,690 px differing, max|d| = 8, mean Σchannels = −1.095, Metal-lower on 63,647 px vs Metal-higher on 40. Center pixel identical (236,180,145). 78 of the 323 debug-logged pixels are in the field (all ±1).
3. **TF tables match.** Both RGBA32F 1024-wide over (0,4370), built identically (`GetTable` + COMPOSITE pre-integration, factor = 0.270059): `MTL_OPTABLE` and `GL_OPTABLE` agree. TF sampling coordinate is `scalarNorm = raw*scalarScale` on both sides.
4. **Metal's accumulation is faithful to the written formula.** Replaying `g_fragColor = (1.0 - fragColor.a) * g_srcColor + g_fragColor` (with `g_srcColor.rgb *= g_srcColor.a`) in float32 — float32 weight, separate inner product, outer `fma` — against Metal's `SAMPLE` rows (which turn out to log the PRE-update `accA/accC`) reproduces Metal's accumulation to log precision (residuals ≤ 1e-6, i.e. 6-decimal rounding).
5. **The per-sample accumulation is insensitive to compiler-legal FMA orders here.** V1 (`fma(w, rgb*op, acc)` = Metal), V2 (no-fma), V3 (`fma(w*rgb, op, acc)`), V4 (`fma(w*op, rgb, acc)`) all end at `accC×255 = (247.68, 161.41, 119.39)` for pixel (0,256) — identical to all printed decimals. The ~1.4e-3/channel brightness bias is NOT reproduced by any reordering of this one composite step.
6. **Reinforces update 44's direction — and now framed against the target:** Metal implements the written formula; clean GL diverges from it by the GLSL driver's compilation. Since the target is clean GL's output, the written-formula match is necessary but NOT sufficient: the divergence clean GL introduces must be identified and reproduced (or eliminated on the GL side) to close the ±1 field. The composite step in isolation is order-insensitive, so the divergence enters elsewhere (whole-loop arithmetic and/or per-sample inputs).

## 4. Current doubts / hypotheses (unresolved)

- The field's **systematic** Metal-dimmer direction (63,647 vs 40) is too asymmetric for rounding noise — a ~1 LSB systematic bias exists somewhere. Since the composite step alone is order-insensitive, the bias must enter via either (a) clean-GL's compile of the *whole* loop (register reuse across samples, contraction with the `g_fragColor` accumulate chain, or fused op in the sample-position/step math), or (b) slightly different per-sample **inputs** (raw/op/rgb) between clean GL and Metal. Per update 44, clean-GL per-sample rows are untrustworthy, and positions were only ever verified against debug-GL (which tracks Metal) — so (b) cannot be ruled out yet.
- **Nearest-interpolation implication:** with nearest volume sampling, per-sample inputs are quantized to texel values, so even a tiny ray/step divergence can flip a sample to a different texel → discrete opacity/color jumps. A sub-texel geometric divergence between clean GL and Metal could therefore produce exactly this kind of systematic field without any composite-arithmetic difference. This is now the leading hypothesis and should be tested by comparing clean-GL vs Metal sample positions/texels directly (needs a non-corrupting GL dump or a controlled CPU ray-march replay on both backends' exact uniforms/matrices).
- **Goal is closed; the path is not.** Target = clean GL's output. Metal matching the written formula is a checkpoint, not the destination: if the ±1 field is clean GL's compiler divergence, Metal must either reproduce that divergence (e.g. mirror the GLSL driver's contraction/reassociation in MSL) or the divergence must be eliminated on the GL side — a decision that now lives inside the work, not as an open question.

## Artifacts

- `/tmp/bc/u47_metal.png` (13:07), `/tmp/bc/u47_gl.png` (13:07).
- `/tmp/bc/u47_metal.log` (~54k shader `MARCH`/`SAMPLE` rows; `scalarMax=0.066681`, `MTL_OPTABLE`), `/tmp/bc/u47_gl.log` (`GL_TEX scale=(14.9967966,…)`, `GL_OPTABLE`, 19 GL markers).
- `u45_fma_metal.png` (pre-fix Metal, byte-identical to `u47_metal`), `fix42_gl.png` (clean-GL determinism reference, byte-identical to `u47_gl`).
- Replay scripts run inline (float32 via `struct`, `fma` emulation).
