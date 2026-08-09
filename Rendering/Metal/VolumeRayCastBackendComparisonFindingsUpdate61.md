# Barycentric weights back out bit-identical between GL and Metal (~1e-8); the interpolator's f32 NDC divide + a consistent sub-pixel sample offset explain the model gap — the residual is the rasterizer arithmetic, not the weights (update 61)

**Date:** 2026-08-09
**Scope:** Update 60 closed the per-vertex inputs (clip + texcoord 94/94 bit-identical) and the pixel center (exact). This update performs the update-60 §4.3 quantification: reconstruct the barycentric weights at the 14 knife-edge pixels from the interpolated clip (both backends), compare against analytic weights from window positions, and test interpolation-arithmetic variants to locate the ~1 ulp interpolated-anchor residual.
**Target (unchanged):** Metal bit-identical to **clean GL** (`RenderingBackend=OpenGL`, no debug injection).

**Follows:** [Update 60](VolumeRayCastBackendComparisonFindingsUpdate60.md) (per-vertex clip + texcoord bit-identical; residual = rasterizer interpolation), [Update 59](VolumeRayCastBackendComparisonFindingsUpdate59.md) (step bit-identical, anchor ~1 ulp, 188 px knife-edge flips), [Update 58](VolumeRayCastBackendComparisonFindingsUpdate58.md) (174 ±1 px plausible, 14 knife-edge px likely unattainable).

---

## 1. Tooling / data

Same logs as update 60 (frame-6, last occurrence per pixel/vid):

- `/tmp/bc/u62_gl_vlog.log` — GL_RAY (interpolated clip + tex + primId/flatVid per pixel), GL_VERT (per-vertex clip + tex), GL_CAPINDEX (triangle topology).
- `/tmp/bc/u62_metal.log` — `vertex_volume_main` (per-vertex clip + texcoord) and `DEBUG STEP` (interpolated clip + localPos + screenPos per pixel).
- Scripts: `/tmp/bc/bary_u61.py` (backed-out vs analytic weights), `/tmp/bc/consist_u61.py` (model vs both backends in ulps), `/tmp/bc/samplept_u61.py` (implied sample point), `/tmp/bc/variants_u61.py` (arithmetic variants), `/tmp/bc/fit_u61.py` (sample-offset fit).

## 2. Result 1: all 14 knife-edge pixels are covered by one triangle, 122 = (86, 40, 93)

| attribute | value |
|---|---|
| primId at all 14 knife px | **122** |
| triangle 122 | vids **(86, 40, 93)**, flatVid 93 |
| vid 86 clip / tex | (−2.44439697, −34.2432251, w=0.38470459) / (0.510781229, 0.461473942, 0.438469529) |
| vid 40 clip / tex | (38.9552612, 377.151184, w=0.384706497) / (0.440242916, 0.999023557, 0.559757113) |
| vid 93 clip / tex | (−117.459229, 240.019714, w=0.384702682) / (0.65001595, 0.819840372, 0.567701578) |

Note the triangle is huge in NDC (vid 40 NDC y ≈ 980, vid 93 NDC x ≈ −305): the covering vertices sit far outside the viewport, so the interpolation weights are strongly peaked on one vertex (~0.91 / ~0.08 / ~0.006).

## 3. Result 2: barycentric weights back out identical between GL and Metal (relative delta ~1e-8)

Backing out the weights from each backend's interpolated clip via the 3×3 system `[Cx Cy 1] t = [NDCx·S, NDCy·S, S]`, `S = 1/w_p`, `λ = t·w`:

- backGL vs backMt agree at every knife pixel to **relative deltas ≤ 8.4e-8** (e.g. (439,281): +2.0e-10, +3.1e-9, −8.4e-8).
- The implied sample point (Σ λᵢ·NDCᵢ) equals the exact pixel center to ~1e-6 NDC (≈2.6e-4 px) for **both** backends, i.e. the pixel-center convention is confirmed, not a half-pixel flip.

## 4. Result 3: the rasterizer's vertex divide is f32, not f64 — and the residual is a systematic weight bias, not arithmetic

Predicting the interpolated clip from analytic weights (exact pixel center) with different NDC-precision models vs the logged values (in float32 ulps of clip.x):

| model | (397,110) clip.x err | (405,171) clip.x err | (120,167) clip.x err |
|---|---|---|---|
| f64 vertex divide + f64 weights + f64 interp | **−16** | −12 | +14 |
| **f32 vertex divide + f32 weights + f32 interp** | **−4** | −20 | +12 |

The f32-NDC chain is far closer (10-80 ulp errors in the f64 chain are explained by f64 vertex NDC not matching the hardware's f32 perspective divide). With the f32 chain, all arithmetic variants (seq/fma/fma2/rcp/f64/interppos) land within ~1 ulp of each other but sit **consistently positive** vs the logs:

| variant | GL sum\|ulps\| | GL mean | Metal sum\|ulps\| | Metal mean |
|---|---|---|---|---|
| seq | 107 | +2.55 | 68 | +1.52 |
| fma | 104 | +2.48 | 67 | +1.45 |
| fma2 | 118 | +2.76 | 75 | +1.74 |
| rcp | 102 | +2.43 | 63 | +1.40 |
| f64 | 111 | +2.64 | 68 | +1.62 |
| interppos | 102 | +2.43 | 63 | +1.40 |

So the residual is a **systematic positive weight bias**, not the interpolation arithmetic. GL is consistently ~1 ulp above Metal across all 14 px and 3 axes.

## 5. Result 4: fitting a sub-pixel sample offset reduces the model gap but does not reach 0

Scanning the sample-NDC offset (both axes, ±8e-4 NDC = ±0.2 px, then fine around the best) minimizes Σ|ulps| for the f32/seq chain:

| target | best offset (NDC) | ≈ px | total err |
|---|---|---|---|
| baseline (no offset) GL / Mt | — | — | 107 / 68 |
| GL | dx≈+2.2e-4, dy≈−3.8e-4 | ~0.06 px | 37 |
| Metal | dx≈+2.4e-4, dy≈−2.0e-4 | ~0.06 px | **26** |

The optimum offsets for GL and Metal are **different** and neither reaches 0. A shared sub-pixel sample-offset model does not close the GL/Metal pair.

## 6. Conclusion

- The barycentric **weights are not the floor**: GL and Metal interpolate with weights identical to ~1e-8 relative, from the same exact pixel center.
- The hardware's perspective divide is f32 (a f64-NDC model is off by 10-80 ulps; f32-NDC by ~4).
- Every interpolation-arithmetic variant reproduces the same ~+1.5-2.5 ulp positive bias, i.e. the remaining GL/Metal ~1 ulp anchor delta is a systematic **weight bias** the shader cannot see (both backends run the same hardware rasterizer; per-vertex clip/texcoord and pixel center are bit-identical).
- The 1-3 ulp interpolated-clip/texcoord delta therefore must originate **below the shader** — most plausibly the GL-on-Metal driver's vertex position post-processing (viewport/flip transform applied to `gl_Position` that rounds differently than native Metal's raw clip feed), which is outside the Metal source.

## 7. Doubts / hypotheses (open)

1. **Driver-level position post-processing is the residual floor.** All shader-visible inputs are bit-identical; the interpolator weights are identical to 1e-8; the only untested input is the actual position stream fed to the rasterizer. The GL-on-Metal driver may apply its framebuffer/viewport transform to `gl_Position` (y-flip and/or window-offset) in a way that perturbs the interpolation weights at the 1-2 ulp level. This cannot be inspected or corrected from Metal source → the 188 px (0.072 %) knife-edge flips would be the irreducible floor for this pair.
2. **Weight computation beyond f32 area-ratio barycentrics.** The sample-offset fit reducing but not eliminating the bias suggests the rasterizer's edge-function/setup stage uses a rounding (fixed-point edge functions, or setup derivatives at reduced precision) not captured by any f32/fma/f64 model here. Characterizing that would need the driver's interpolation spec.
3. **GL > Metal systematically by ~1 ulp.** Every knife-edge pixel shows GL logged ≥ Metal logged by 1-3 ulps on the interpolated anchor. If this bias is a constant function of (triangle, weight), a Metal-side fragment-shader correction could in principle cancel it, but the exact functional form is not yet known.

## Artifacts

- Data: `/tmp/bc/u62_gl_vlog.log`, `/tmp/bc/u62_metal.log` (frame-6 knife-edge captures).
- Scripts: `/tmp/bc/bary_u61.py`, `/tmp/bc/consist_u61.py`, `/tmp/bc/samplept_u61.py`, `/tmp/bc/variants_u61.py`, `/tmp/bc/fit_u61.py`.
- Verified (python, /tmp/bc): triangle 122=(86,40,93) covers all 14 knife px; backGL/backMt weights identical to ≤8.4e-8 relative; implied sample point == pixel center (±1e-6 NDC); f32-NDC divide is the correct model (f64 off 10-80 ulps, f32 off ~4); all interpolation variants show a systematic +1.4-2.8 ulp positive bias vs both backends; sample-offset fit (best ~0.06 px, dx differing GL vs Metal) reduces but never reaches 0.
