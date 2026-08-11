# Update-76 §5's two next leads are both already explored offline: the sample-position probe (update 61) dead-ends, and the analytic-anchor concept was modeled exhaustively in updates 61/62/63 with largely negative verdicts — the only untried step is the shader implementation, which is the single experiment that settles the update-76 §4 vs update-63 contradiction (update 77)

**Date:** 2026-08-11
**Scope:** Update 76 §5 proposed two next experiments: (1) probe Metal's barycentrics / sample position by inverting the interpolation, and (2) the analytic-anchor experiment (bypass the interpolator with pixel-center barycentrics + per-vertex clip/texcoord). Before burning a build cycle, this update audits the historical record: both concepts were already explored *offline* (Python models against the logged dumps) in updates 61/62/63. The sample-position probe is refuted; the analytic-anchor models mostly failed to reproduce either backend at 0 ulps, with one recent single-pixel counter-signal (update 76 §4) whose tension with update 63 was never reconciled. Neither was ever implemented in the Metal shader, so the decisive experiment remains the shader change itself.
**Target (unchanged):** Metal bit-identical to **clean GL** (`RenderingBackend=OpenGL`, no debug injection) on `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter`.

**Follows:** [Update 76](VolumeRayCastBackendComparisonFindingsUpdate76.md) (per-vertex x/y texcoords 94/94 bit-identical; f64 reconstruction == GL exactly at worst pixel; Metal +1 ulp), [Update 63](VolumeRayCastBackendComparisonFindingsUpdate63.md) (no analytic pixel-center model hits 3/3 channels at 0 ulps on either backend), [Update 62](VolumeRayCastBackendComparisonFindingsUpdate62.md) (fitted per-pixel offsets: 9/14 px at 0 ulps, 5/14 residual 1-5 ulp on one channel), [Update 61](VolumeRayCastBackendComparisonFindingsUpdate61.md) (implied sample point == pixel center; all interpolation-arithmetic variants systematically +1.4-2.8 ulp positive vs both backends).

---

## 1. Lead 1 (probe the sample position) was already run in update 61 and dead-ends

Update 61 §4/§5 quantified exactly what update 76 §5 experiment 1 proposed:

- **Implied sample point == exact pixel center** for *both* backends (`samplept_u61.py`: Σλᵢ·NDCᵢ == pixel center to ~1e-6 NDC ≈ 2.6e-4 px).
- **The sample-offset fit does not close the pair** (`fit_u61.py`): scanning ±0.2 px, the best offset (~0.06 px) reduced the model error 107→37 (GL) and 68→26 (Metal) but **never reached 0**, and the optimal offsets were **different per backend** (dx≈+2.2e-4 GL vs +2.4e-4 Metal; dy −3.8e-4 vs −2.0e-4). No shared sub-pixel sample-offset model reproduces both backends.
- Update 63 §2 confirmed: backed-out effective weights are internally consistent (residual ~1e-15) but displaced identically in *both* backends (GL mean (+2.9e-4, −2.9e-4), Metal (+1.8e-4, −1.7e-4) NDC; GL−Metal ~(+0.03, −0.03) px).

**Verdict: the rasterizers do not evaluate at a different, discoverable sample point that could be emulated.** Lead 1 is closed.

## 2. Lead 2 (analytic anchor) was modeled exhaustively offline — mostly negative

The analytic-anchor idea (compute the interpolated anchor from pixel-center barycentrics + per-vertex clip/texcoord instead of the hardware interpolator) was already the subject of three update-series models:

| experiment | model(s) | result |
|---|---|---|
| update 61 §4 | analytic pixel-center weights × interp arithmetic (seq/fma/fma2/rcp/f64/interppos) vs logged clip | all variants land within ~1 ulp of each other but sit **consistently positive** vs both backends; none at 0 (GL sum 102-118 ulps, mean +2.4-2.8; Metal 63-75, mean +1.4-1.7) |
| update 62 (`solve_u62.py`, `plane_u62.py`) | plane-equation / derivative-based persp interp (Cramer, base-vertex 2×2, f32/f64); per-pixel **fitted** sample offset with f64-persp + f32-NDC weights | plane variants same error pattern as update 61; with a *free per-pixel* (dx,dy), 9/14 knife-edge px reach 0 ulps on all 3 channels, **5/14 still 1-5 ulp on one channel** — i.e. even a fitted offset cannot reproduce GL exactly at 5 px |
| update 63 (`model_u63.py`) | affine / persp `1/w` / persp `rcp(1/w)` / inverse-w at the pixel center | **no** analytic pixel-center model reproduces 3/3 channels at 0 ulps for *any* pixel on *either* backend; per-pixel 3/3 offset regions are broad and disjoint (no common offset); 5 px never reach 3/3 even with a fitted offset |

The categorical update-63 conclusion: the varying-path bias cannot be encoded in a fragment-shader interpolation recipe using pixel-center weights.

## 3. The one unresolved counter-signal: update 76 §4

Update 76 §4 reconstructed the worst-pixel covering triangle (primId=122) with **full-float64** perspective-correct interpolation (screen-space barycentrics from the bit-identical clip positions, no f32 NDC rounding) and reproduced **GL's interpolated y-texcoord exactly** (`0x3f01aa39`) while Metal logged `0x3f01aa3a` (+1 ulp).

This is in direct tension with update 63's categorical negative and update 61's "the f64-NDC chain is the wrong model (off by 10-80 ulps)". The three update-series models used **f32-NDC** weights (because update 61 showed the hardware's vertex divide is f32); update 76 §4 used **f64-NDC**. Only one axis at one pixel was checked. **The contradiction was never reconciled.**

## 4. Never implemented in the shader — that is the decisive experiment

`grep` of `MetalShaders.metal` confirms `anchorTex = in.texcoord;` (fragment entry points at ~5245/5346/5561) is still the hardware-interpolated value; there is no analytic/barycentric anchor computation anywhere in the fragment stage.

So the record is:

- Concept modeled offline: **exhaustive, mostly negative** (updates 61/62/63).
- Concept implemented in shader + image measured: **never done**.

Because the modeling results are contradictory (update 63 categorical-no vs update 76 §4 single-pixel-yes) and the difference between them (f32-NDC vs f64-NDC weights, 3-channel vs 1-axis) is exactly the kind of detail a real implementation would pin down, **the shader experiment is the only test that resolves the question**. If the analytic anchor collapses the residual, update 63's model space simply missed the correct recipe; if it does not, the floor is confirmed and documented.

## 5. Remaining decision / plan

Priority order agreed for the next session:

1. **Analytic-anchor shader experiment (decisive):** implement the anchor in Metal as `Σ(wᵢ·texᵢ/clip.wᵢ) / Σ(wᵢ/clip.wᵢ)` from pixel-center barycentrics + per-vertex clip/texcoord (f64 or f32-NDC variants as A/B), bypassing the interpolator for `anchorTex` only (camera-inside proxy path). Requires a CPU-built per-primitive vertex-data buffer (cap mesh has only ~94 vertices) indexed by `primId [[primitive_id]]`. Rebuild (`./macos_metal_build.sh --resume --tests`), compare the 178 px residual.
2. **If it closes:** pick the variant that hits 0 ulps, document, and extend the same fix to the parallel-projection path.
3. **If it does not:** quantify the interpolation floor across the full 8237-px gate (per-axis ulp histogram, not just the 19 residual pixels), frame-match, and document it as the irreducible bound with a quantified pixel budget.

## Artifacts

- Data: unchanged (update 76 §6 logs).
- Scripts: unchanged (updates 61/62/63 models referenced above: `samplept_u61.py`, `fit_u61.py`, `solve_u62.py`, `plane_u62.py`, `model_u63.py`).
- Verified (this session): `git grep` of `Rendering/Metal/Shaders/MetalShaders.metal` — no analytic/barycentric anchor path exists in the fragment stage; `anchorTex = in.texcoord` is the only assignment.
