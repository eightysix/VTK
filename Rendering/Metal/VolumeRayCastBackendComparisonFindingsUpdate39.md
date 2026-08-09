# Anchor parity (evalPoint = p.localPos + evalStep·jitterFrac) collapses the knife-edge 115→1 and fixes fetch parity, but exposes the accumulated step drift: 10039 px |d|≥2 (max 27), isolated to evalStep−g_dirStep ≈ 3e-8/axis (update 39)

**Date:** 2026-08-09
**Scope:** For the camera-inside proxy path, the Metal march's fetch position `evalPoint` was anchored on the ray-box near-plane entry (`CTP(texLocalPos)`), reproducing a systematic ~9.7e-5 z-offset vs GL's construction `g_rayOrigin = ip_textureCoords + g_rayJitter`. This update re-anchors `evalPoint` on the interpolated per-vertex texcoord plus the (jitter-scaled) step — `p.localPos + evalStep·jitterFrac` — at all three rebuild sites. Result: the knife-edge pixel (422,92)/(422,419) collapses **|d| = 115 → 1** and fetch positions match GL's `g_dataPos` to ~6e-7 at i=0, but the aggregate got worse (10039 px |d|≥2, max 27) because the underlying **per-step drift `evalStep` vs `g_dirStep` (~3e-8/axis)** is now unmasked. The drift is traced to the ray **direction**: `dirObj = normalize(anchorData − eyePosData)` vs GL's `normalize(ip_vertexPos − in_eyePosObjs[0])`, whose normalize inputs are the two backends' interpolated data-space anchors (~2.2e-5 apart, update-38 §4), not to the CTP step transform (`texStep` is *farther* from GL than `evalStep`).

**Follows:** [Update 38](VolumeRayCastBackendComparisonFindingsUpdate38.md).

---

## 1. The change (evalPoint anchor parity, `MetalShaders.metal`)

Three rebuild sites in `marchVolumeUnified` now branch on the camera-inside proxy path (`p.anchorIsData && useParallelProjection < 0.5`):

- **Init (~line 4045):** `evalPoint = (p.anchorIsData && !parallel) ? (p.localPos + evalStep * jitterFrac) : CTP(texLocalPos)`, with `jitterFrac = p.jitter / p.stepSize` (= 1.0 for NoJitter, matching GL `g_rayJitter = g_dirStep`). `p.localPos` carries the interpolated per-vertex CTP texcoord (`in.texcoord`, GL `ip_textureCoords` parity from update 38).
- **Out-of-bounds clamp re-sync (~line 4150):** `p.localPos + evalStep * (jitterFrac + float(currentT))`.
- **Min-max empty-cell re-sync (~line 4201):** same counter-based rebuild.

Legacy paths (camera outside, grid traversal) keep the entry-anchored `CTP(texLocalPos)`. The min-max skip does **not** fire in this test (t-sequence monotonic, step 0.001886), so the two re-sync edits are inert here but kept for consistency.

## 2. Measurement (same test, same last-frame camera `(102.122314, 102.122314, 61.5619835)`, fresh captures)

GL reference verified byte-identical to `u38c/gl.png` (max |d| = 0). `u38c/metal.png` is stale (== GL) — old-Metal numbers come from the update-38 table, not that file.

| metric | update-38 post-change | update-39 working tree |
|---|---|---|
| pixels differing | 64095 | 68881 |
| pixels \|d\| = 1 | 64094 | 58842 |
| pixels \|d\| > 1 | **1** | **10039** |
| pixels \|d\| ≥ 5 | 1 | 559 |
| worst pixel | 115 | 27 |

The ±1 population dropped (64094 → 58842) but the |d|≥2 population exploded (1 → 10039), i.e. the fix traded the single knife-edge for a broad, well-characterized drift. Worst pixel PNG `(384,151)`: GL `(234,177,142)` vs Metal `(233,165,128)`, |d| = 27. The knife-edge pixel (422,92)/(422,419): GL `(244,152,110)` vs Metal `(244,152,109)`, |d| = **1** (was 115).

## 3. Fetch-position parity (verified)

At (422,92)/(422,419) the fetch position now matches GL's `g_dataPos`:

- **i=0:** Metal `eval = (0.505428, 0.506534, 0.450738)` vs GL `pos = (0.505427599, 0.506533682, 0.450737983)` — ~6e-7 (was a 9.7e-5 z-offset).
- **i=128..170:** position diffs (Metal−GL) grow `(3.1e-6, 7.4e-6, 5e-7)` at i=128 → `(4.9e-6, 1.07e-5, 8.8e-7)` at i=170 — a steady ~2–7e-8/step growth.

## 4. The unmasked residual: a 2-sample raw flip at a thin structure + earlier opaque break

- Sample counts diverge: Metal 171 samples (i=0..170), GL 175 (i=0..174) at this pixel. Metal crosses the opaque threshold (`1 − 1/255 = 0.996078`) after i=170 (accA 0.9964); GL after i=172 (0.9972).
- Per-sample raw values agree to ~1e-7 for most samples, but at **i=132–133 GL's raw dips to 0.016281/0.016266 while Metal stays 0.017868/0.017609** (+0.0016), then both re-converge at i=134 — a thin tissue structure sampled on different sides due to the position drift.

## 5. Step drift quantified and localized to the direction

Last-frame STEP dump at the instrumented pixel (Metal `px=(422, 92)` = GL `(422,419)`):

```
rayDir = (-0.2396636903, -0.002679602941, 0.9708522558)   [GL semantic: normalize(ip_vertexPos - eye)]
dirObj = (-0.3392504752, -0.003795351367, 0.9406884313)   [Metal: normalize(anchorData - eyePosData)]
evalStep = (-4.535645712e-04, -5.074235560e-06, 1.837282092e-03)
texStep  = (-4.544391122e-04, -5.080938081e-06, 1.840884681e-03)
GL g_dirStep (from pos i5->6, 9-dec) = (-4.535910000e-04, -5.126000000e-06, 1.837253000e-03)
```

| per-step diff vs GL | x | y | z | drift @170 |
|---|---|---|---|---|
| evalStep − g_dirStep | +2.6e-8 | +5.2e-8 | +2.9e-8 | (4.5e-6, 8.8e-6, 4.9e-6) |
| texStep − g_dirStep | −8.5e-7 | +4.5e-8 | +3.6e-6 | — |

The observed position growth (x +2.6e-6, y +7.4e-6 over i=0..128, ~2–6e-8/step) matches the evalStep drift estimate, and **not** the texStep one — so the residual step error is **not** the CTP step transform (`adjustedLin` vs `ip_inverseTextureDataAdjusted`; that chain is already at GL op-order parity, closer than the texture-space variant). It is the **direction**: the normalize inputs `anchorData` and `ip_vertexPos` are the two backends' interpolated data-space anchors differing by ~2.2e-5 (update-38 §4, rasterizer 1–2 ULP on the giant cap triangles); a 2.2e-5-relative input shift → ~2e-5 rad direction shift → ~3e-8/step.

## 6. Phase 2: analytic ray direction in both backends (plan)

The interpolated anchors cannot be made bit-equal (rasterizer weight rounding, update-38 §5), so the only path to `evalStep == g_dirStep` bit-exactly is to stop using the interpolated anchor for the direction and unproject the pixel analytically in **both** backends:

- Metal already has `reconstructRayDir(screenPos, viewportSize, u)` (MetalShaders.metal:3603): `ndc = (screenPos/viewportSize)*2−1`, near/far volume positions `wn/wf = ndcToVolume * (ndc.x, −ndc.y, {0,1}, 1)` (divide by w), `normalize(wf − wn)`. Used on the fullscreen/parallel paths (5230, 5297, 5587).
- GL must compute the **same** quantity with the **same arithmetic**: a composed `in_ndcToVolume` uniform + the same near/far unprojection + normalize, replacing the perspective `computeRayDirection()` in `vtkVolumeShaderComposer.h:1731` (identity change for parallel, which already uses `in_projectionDirection`).
- Bit-parity requirements: (a) the composed `ndcToVolume` matrix must be CPU-built identically in both backends (same matrix product order, same transpose convention, float32); (b) the window→NDC mapping must agree (Metal's y-negation vs GL's `in_windowLowerLeftCorner`/`in_inverseWindowSize`; needs `windowLowerLeftCorner = (0,0)` or a shared transform); (c) identical near/far depth values (z = 0/1).
- Then verify `evalStep == g_dirStep` bit-exactly; the residual would shrink to the ~6e-8 anchor offset (`in.texcoord` vs `ip_textureCoords`), 100× below the ~1e-5 needed to flip samples. If sample counts and the opaque break still diverge, address termination/break parity next.

## Artifacts

- Captures: `/tmp/bc/phase1/OpenGL.png`, `/tmp/bc/phase1/Metal.png` (last frame; exit 1 = expected baseline mismatch, PNGs valid), `/tmp/bc/phase1/mt.log` (Metal SAMPLE/STEP dumps), `/tmp/bc/u38c/gl2.log` (GL_SAMPLE), `/tmp/bc/u38c/gl.png` (GL reference, byte-identical).
- Diff script output in this session: 68881 differing, 58842 |d|=1, 10039 |d|>1, 559 |d|≥5, max 27; GL-vs-GL maxd 0.
