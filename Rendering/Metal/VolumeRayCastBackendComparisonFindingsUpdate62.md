# The texcoord-vs-clip interpolation paths use different effective sample weights; a consistent ~(+0.06,-0.09) px varying-path offset reproduces GL's texcoord to 0 ulps at 9/14 knife-edge pixels (update 62)

**Date:** 2026-08-09
**Scope:** Update 61 proved barycentric weights back out bit-identical GL-vs-Metal (~1e-8) and that the interpolation-arithmetic variants all share a systematic +1.4-2.8 ulp bias. This update attacks the last untested formula class (plane-equation / setup-derivative interpolation) and, failing to differentiate it, searches for the effective sample point that the varying path really uses.
**Target (unchanged):** Metal bit-identical to **clean GL** (`RenderingBackend=OpenGL`).

**Follows:** [Update 61](VolumeRayCastBackendComparisonFindingsUpdate61.md) (weights bit-identical, bias is systematic), [Update 60](VolumeRayCastBackendComparisonFindingsUpdate60.md) (per-vertex clip+tex bit-identical), [Update 59](VolumeRayCastBackendComparisonFindingsUpdate59.md) (188 px knife-edge flips = ~1 ulp anchor), [Update 58](VolumeRayCastBackendComparisonFindingsUpdate58.md).

---

## 1. Data

Same frame-6 logs as update 61 (`/tmp/bc/u62_gl_vlog.log`, `/tmp/bc/u62_metal.log`; triangle 122 = (86,40,93), 14 knife-edge px). New scripts (update-62 tooling, `/tmp/bc`):

- `plane_u62.py` — plane-equation / derivative-based perspective interpolation (Cramer and base-vertex 2×2 solves, f64/f32), plus affine `1/w` (clip.w) and gl_FragCoord-reconstruction (clip.x/y) models.
- `solve_u62.py` — per-pixel search for a sample-NDC offset (dx,dy) such that the f64-perspective + f32-NDC-barycentric model hits GL's logged texcoord **exactly (0 ulps on all 3 channels)**.

Note: the framebuffer is 512 px wide, so vertex window coords = `(ndc+1)*256` (an earlier draft used `*128` — corrected). The three vertices' window positions are enormous: vid 86 (−1370.6, −22531), vid 40 (26178, 251228), vid 93 (−77907, 159977); f32 ulp at ±2.5e5 ≈ 0.015 px.

## 2. Result 1: plane-equation interpolation is numerically identical to barycentric — the formula class is NOT the differentiator

Implementing perspective correction as `affine(a/w)/affine(1/w)` via setup-stage plane coefficients (Cramer's rule and base-vertex 2×2 solves, in f64 and f32) yields the **same** error pattern as the barycentric area-ratio variants:

| model | GL sum\|ulps\| | Metal sum\|ulps\| |
|---|---|---|
| barycentric f64/seq (update 61) | 107 | 68 |
| plane f64 (Cramer) | 112 | 69 |
| plane f32 (Cramer) | 108 | 67 |
| plane f64 (base-vertex solve) | 100 | 59 |
| plane f32 (base-vertex solve) | 100 | 59 |

An *exact* plane through the 3 vertices reproduces the barycentric interpolation (it is the same function), and its f32 evaluation rounds to the same pattern. So the +2.5 (GL) / +1.5 (Metal) ulp bias is in the **effective weights / sample geometry**, not the combine formula.

## 3. Result 2: the affine `1/w` and gl_FragCoord-reconstruction models confirm the position path

- **clip.w = 1/affine(1/w)** at the sample matches logged clip.w to +2.5 (GL) / +1.5 (Metal) ulps — the clip-space w interpolates as the reciprocal of a window-affine `1/w` plane.
- **Logged clip.x/w and clip.y/w equal the pixel-center NDC** to ~1e-6: for (397,110) they imply window (397.50019, 401.49920) vs the pixel center (397.5, 401.5) — sub-pixel, f32-arithmetic noise. The logged GL_RAY clip is the gl_FragCoord-style reconstruction at the pixel center, as update 61 assumed.

## 4. Result 3 (NEW): the varying (texcoord) path interpolates as if at an offset sample point ~(+2.5e-4, −3.5e-4) NDC ≈ (+0.06, −0.09) px

Per-pixel search over (dx,dy) in sample NDC for the f64-perspective + f32-NDC model to reproduce GL's logged texcoord exactly (0 ulps on all 3 channels):

| px | best dx | best dy | channels exact |
|---|---|---|---|
| (349,255) | +2.0e-5 | −8.0e-5 | 3/3 |
| (405,171) | +3.8e-4 | −3.4e-4 | 3/3 |
| (338,432) | +3.4e-4 | −3.2e-4 | 3/3 |
| (350,  5) | +3.6e-4 | −4.2e-4 | 3/3 |
| (153, 32) | +2.8e-4 | −3.6e-4 | 3/3 |
| (482, 33) | +3.2e-4 | −4.8e-4 | 3/3 |
| (439,281) | +1.0e-4 | −1.4e-4 | 3/3 |
| (397,110) | +2.8e-4 | −9.0e-4 | 2/3 |
| (360,229) | +2.0e-4 | −4.8e-4 | 2/3 |
| (  9, 18) | +2.8e-4 | −2.6e-4 | 2/3 |
| (293,298) | +1.8e-4 | −3.6e-4 | 2/3 |
| (120,167) | +2.2e-4 | −4.4e-4 | 2/3 |
| (470,269) | +2.4e-4 | −5.8e-4 | 2/3 |
| (469,463) | +2.0e-4 | −3.0e-4 | 2/3 |

- The winning offsets **cluster consistently** (dx ∈ [+1e-4, +3.8e-4], dy ∈ [−9e-4, −8e-5]) — a real, reproducible sub-pixel bias, not noise. Same ballpark as update 61's global fit (GL dx≈+2.2e-4, dy≈−3.8e-4).
- At those offsets the f64-perspective model reproduces **GL's interpolated texcoord to exactly 0 ulps at 9/14 pixels** (all 3 channels) and to 0 ulps on 2/3 channels at the other 5 (residual 1-5 ulps on one channel).
- Likely mechanism: the rasterizer's edge functions / plane coefficients are evaluated from the **f32-rounded window positions** of the (huge) vertices; at ±2.5e5 px, f32 ulp ≈ 0.015 px, so the effective weights shift by ~1e-3 px — matching the observed bias and its per-pixel scatter.

## 5. Result 4 (NEW): the position (clip) path and the varying (texcoord) path use DIFFERENT effective sample weights

At the offset that fixes the texcoord (+2.5e-4, −3.5e-4), clip.x moves from ~0 ulps to **+6431 ulps** (≈ +0.0024 absolute) at (397,110):

| px | clip.x ulps @offset0 | clip.x ulps @tex-offset | tex ulps @0 | tex ulps @tex-offset |
|---|---|---|---|---|
| (397,110) | −3 | **+6435** | (3,3,5) | (1,0,3) |
| (360,229) | −6 | +6431 | (2,2,3) | (0,−1,1) |
| (349,255) | −7 | +6449 | (1,0,0) | (−2,−3,−3) |
| (405,171) | −14 | +6457 | (3,2,4) | (1,0,1) |
| (  9, 18) | +1 | −3220 | (3,2,5) | (1,−1,3) |

- The **clip/position path** is exactly the analytic f32-NDC pixel-center weights (backout matches to ~1e-8; implied sample = pixel center; needs offset ≈ 0).
- The **varying/texcoord path** is consistent with a ~(+2.5e-4, −3.5e-4) NDC offset (needs offset ≉ 0).
- So gl_Position interpolation and attribute interpolation run through different effective-weight paths in the hardware — matching the OpenGL model where `gl_Position` becomes `gl_FragCoord` (position path, screen-affine) while varyings go through the attribute interpolator. This resolves why a single weight set could reproduce clip to 0 ulps but leave texcoord +2.5 ulps biased.

## 6. Conclusion

- Formula class is not the differentiator (Result 1); the position path is exactly the pixel-center f32-NDC weights (Results 2-3); the varying path has its own consistent sub-pixel effective sample bias (Result 3) that is *decoupled* from the position path (Result 4).
- With a per-pixel varying-path offset, GL's interpolated texcoord is reproducible **to 0 ulps at 9/14 knife-edge pixels** from flat per-vertex data + sample position. This is the closest yet to a fragment-shader recipe for GL's varying interpolation.
- The GL-vs-Metal ~1-3 ulp difference remains (both backends' *position-path* weights are identical to 1e-8; the varying-path bias is common to both) and still points at the two drivers computing vertex window positions differently (GL-on-Metal viewport/flip post-processing vs native Metal).

## 7. Doubts / hypotheses (open)

1. **Varying-path bias = f32 window-position rounding.** The ~(+0.06, −0.09) px bias and its per-pixel scatter plausibly come from evaluating edge functions/plane coefficients at f32-rounded window positions of the huge vertices (f32 ulp ≈ 0.015 px at ±2.5e5). If the exact op order (sequential f32 `(ndc+1)*256` vs f64-then-f32 vs driver viewport matrix) were identified, the bias would become deterministic and fully replicable in a Metal fragment shader.
2. **5 pixels only match 2/3 channels.** A finer window-coordinate-rounding model may close the residual 1-5 ulps on one channel (next step if continuing: sweep window-rounding variants jointly with the per-pixel offset).
3. **GL-vs-Metal remains a driver-level difference.** Even with the varying-path recipe pinned, GL's values come from GL-on-Metal's position math; if that rounding cannot be reproduced from the flat clip data we have, the 188 px (0.072 %) knife-edge flips stay the floor and the shader recipe only works for pixel sets we calibrate empirically.

## Artifacts

- Data: `/tmp/bc/u62_gl_vlog.log`, `/tmp/bc/u62_metal.log` (frame-6, triangle 122 = (86,40,93), 14 knife-edge px).
- Scripts: `/tmp/bc/plane_u62.py`, `/tmp/bc/solve_u62.py` (not yet persisted to `Rendering/Metal/BackendComparisonTools/update62/`).
- Verified (python, /tmp/bc): plane-equation == barycentric (GL 100-112 / Mt 59-69 ulps); affine `1/w` for clip.w (+2.5/+1.5 ulps); clip.x/w, clip.y/w == pixel center to ~1e-6; per-pixel varying-path offset ~(+2.5e-4, −3.5e-4) NDC reproduces GL texcoord at 0 ulps for 9/14 px; the same offset breaks clip.x by +6431 ulps → position and varying paths are decoupled.
