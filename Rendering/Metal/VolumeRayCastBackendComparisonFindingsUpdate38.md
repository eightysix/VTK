# Per-vertex ip_textureCoords parity collapses the masked residual 130 → 1; the last pixel is at the rasterizer's 1–2 ULP interpolation floor (update 38)

**Date:** 2026-08-09
**Scope:** Ports GL's *per-vertex* `ip_textureCoords` computation (`in_cellToPoint * (in_inverseTextureDatasetMatrix * in_vertexPos)`) into the Metal vertex shader with the exact GLSL `mat4*vec4` contraction, replacing the fragment-time cell-to-point affine of the interpolated data. Result: pixels with `|d|≥5` drop from **130 → 1** and `|d|>1` from 1321 → 1; the remaining single knife-edge pixel `(422,92)/(422,419)` and the uniform 24% ±1 pattern are isolated to the **GL-vs-Metal rasterizer barycentric interpolation** (1–2 ULP on the interpolated value, ~1e-7 on the weights), amplified by the giant off-screen cap triangles. This update also invalidates two earlier cross-frame "inconsistency" analyses and re-verifies the same-frame per-vertex inputs are bit-identical.

**Follows:** [Update 37](VolumeRayCastBackendComparisonFindingsUpdate37.md).

---

## 1. The change (per-vertex texcoord parity)

GL computes the ray anchor's texture coordinate in the **vertex** shader and lets the rasterizer interpolate the per-vertex result:

```glsl
// vtkVolumeShaderComposer.h ComputeTextureCoordinates (lines 116-131)
uvx = in_inverseTextureDatasetMatrix * vec4(in_vertexPos, 1.0);
ip_textureCoords = (in_cellToPoint * vec4(uvx, 1.0)).xyz;
```

Metal previously reconstructed the same quantity in the **fragment** shader as an affine map of the *interpolated data-space* anchor, a different rounding path. This update moves it to the vertex stage with hand-written GLSL `mat4*vec4` contraction (`mul, fma, fma, mul+add`) over `v = float4(modelPos, 1)`:

- `MetalShaders.metal` `vertex_volume_main`: `uvx = volumeToTexture * v` (strict contraction); `out.texcoord = cellToPointScale * uvx + cellToPointOffset`; `VolumeVertexOut` gained `float3 texcoord`.
- All 3 fragment paths (base + two selection variants): `anchorTex = in.texcoord` under `cameraInsideProxy` (was a fragment-time affine of `in.localPos`).
- C++ `VolumeMapperUniforms` (+`CellToPointScale[4]`, +`CellToPointOffset[4]`, 1872 → 1904 bytes); CPU computes the float32 CTP scale/offset from input dims (`MTL_CTP` log == `GL_CTP`), and `VolumeToTexture` mirrors GL's `invert(transpose(texToVol))` with non-transposed store (update-33 `InvTexMatVec` parity).

Shader-build error fixed: the original edit declared a duplicate `float4 v` ("redefinition of 'v'"); reuses the existing `v` from the clip computation.

## 2. Definitive pre/post measurement (same test, same camera, fresh captures)

Both renders are the last frame (camera `(102.122314, 102.122314, 61.5619835)`); GL reference verified byte-identical to the update-37 capture (`u37/glref.png` ≡ `u38c/gl.png`, max |d| = 0). The pre-change measurement was taken from a **fresh HEAD build** (working tree stashed → built → captured → restored), not from stale artifacts.

| metric | HEAD (update 37, pre-change) | working tree (update 38, post-change) |
|---|---|---|
| pixels differing | 68934 | 64095 |
| pixels \|d\| = 1 | 67613 | 64094 |
| pixels \|d\| > 1 | 1321 | **1** |
| pixels \|d\| ≥ 5 | 130 | **1** |
| worst pixel | 14 | 115 |

The ±1 population is spatially uniform (quadrant fractions 0.225–0.255) — a global comb drift, not edge-specific. The single masked pixel is at PNG `(92,422)` = Metal screenPos `(422,92)` = GL `(422,419)`: GL `(238,176,140)` vs Metal `(216,61,76)`, |d| = 115 (knife-edge flip across a material boundary).

Earlier "0 masked" artifacts (`u36/dummy_baseline.png`) are stale — do not use them as a baseline; always stash/build/capture the exact code state.

## 3. Same-frame verification (invalidates two earlier cross-frame analyses)

The GL `GL_CAPVERTS` dump and Metal `vertex_volume_main` log were previously compared across different frames (different camera poses), producing bogus "degenerate triangle" and "24× weight inconsistency" conclusions. Same-frame checks:

- GL last-frame `GL_CAPVERT 40` = `(88.729393, 201.600006, 77.262619)` **== Metal last-frame vid=40** `modelPos` bit-for-bit; the last-frame `GL_CAPINDEX`/Metal vertex log agree the covering triangle at the probe is **#122 = (86, 40, 93)**.
- Reconstructed GL per-vertex clip `((P*V)*M)*v` from the last-frame `GL_CLIPMAT` P/V/M (column-major `<16f` hex) with the strict `mul,fma,fma,mul+add` emulation matches Metal's logged per-vertex clip to ≤ 0.1 ULP in x/y/w (z is the nearz convention: GL −1 vs Metal 0). A numpy `@` emulation initially "failed" (thousands of ULP off) — that was a **transposed-product bug in the Python emulation**, not a shader bug; the corrected emulation confirms the per-vertex clip chain (update 36) still holds for the current frame.
- Per-vertex data, clip x/y/w, and window size (512×512, pixel center exactly (422.5, 419.5)) are all bit-identical. The interpolated **fragment** values differ by 1–2 ULP: clip (x 2.9e-8, y 5.9e-8, w 2.9e-8), data anchor (x 2.2e-5, y 2.2e-5, z 1.9e-5), texcoord (x 6e-8, y 6e-8, z 3e-8).

## 4. Root cause: well-conditioned inputs, ill-conditioned weights

The covering triangle is **astronomically large in window space**: vertex windows ≈ (−1370, −22531), (26180, 251228), (−77906, 159980) — ~50,000× the screen area, with the pixel sitting on/near its edge. The barycentric weights are the ratio of nearly-cancelling edge functions:

- weights from same-frame data interpolation: Metal `(0.914352, 0.080206, 0.005441)`, GL `(0.914353, 0.080207, 0.005441)` — a **~1e-7 difference**.
- Both backends are **internally self-consistent**: each one's interpolated data, texcoord, and clip all agree with a single weight triple (z-texcoord earlier "inconsistency" was a wrong bounds constant: z-extent is 138, not 201.6). So the two backends simply use slightly different weights.
- The observed anchor diffs are quantitatively consistent: weight diff ~2e-7 × data spread 108.6 (y) ≈ 2.2e-5 (observed); weight diff ~2e-7 × texcoord spread 0.54 ≈ 1.1e-7 (observed 6e-8). No remaining inconsistency.

Conclusion: the interpolated-value differences are 1–2 ULP **from the rasterizer's barycentric interpolation** (GL-on-Metal driver vs Metal driver compute the edge-function ratios with a slightly different rounding/sub-pixel setup on the same silicon). This is outside shader control. The giant triangle makes the weight ratio numerically ill-conditioned, so the 1–2 ULP clip differences amplify into the observed data/texcoord anchor shifts.

## 5. Why the residual looks the way it does

- **64094 px uniform ±1:** every pixel is covered by a giant cap triangle → every pixel's interpolated anchor differs by ~1 ULP → the comb drifts ~1e-5 in data space over ~100 samples → the accumulated composite rounds ±1 on ~25% of pixels (those near an 8-bit rounding boundary).
- **1 knife-edge pixel (|d|=115):** the same ~1e-5 drift crosses a material boundary in the data, flipping many samples between bright and dark.

The step (evalStep vs g_dirStep) and anchor differences are dominated by the anchor's 2.2e-5 data-space shift (the direction's y-component is a near-cancelling difference with the camera at the volume axis).

## 6. Feasibility of going further

- The 1–2 ULP interpolation rounding **cannot be removed from the shader**: per-vertex clip/data are bit-identical and the difference lives in the rasterizer's weight evaluation.
- **Densifying/subdividing the cap mesh** (making each covering triangle ~pixel-sized, well-conditioned) is the only remaining lever: the weight *difference* would shrink proportionally with the triangle's attribute spread, plausibly collapsing both the ±1 and the knife-edge. It changes the geometry both backends render (GL stays the reference), is non-trivial (cap generation in `vtkOpenGLGPUVolumeRayCastMapper.cxx`), and is NOT guaranteed to reach bit-exactness if the two interpolators still round identically-shaped weights differently.

## Artifacts

- Fresh captures: `/tmp/bc/u38c/gl.png`, `/tmp/bc/u38c/dummy_baseline.png`, `/tmp/bc/u38c/mt.log`, `/tmp/bc/u38c/gl2.log` (last-frame `GL_CAPVERTS`/`GL_CAPINDEX`/`GL_CLIPMAT`), `/tmp/bc/u38c/step422_92.txt` (`STEP` last frame), `/tmp/bc/u38c/g419.txt`.
- HEAD baseline: `/tmp/bc/head_base/OpenGL.png`, `/tmp/bc/head_base/Metal.png` (+ logs). CPU emulation scripts inline in this session's analysis.
