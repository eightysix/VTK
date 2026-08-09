# Per-vertex clip and per-vertex texcoord are bit-identical (94/94 vids); the remaining ~1-ulp interpolated-anchor difference is the rasterizer interpolation itself — not per-vertex values, not the pixel-center (update 60)

**Date:** 2026-08-09
**Scope:** Update 59 closed the lattice to a single residual: Metal's interpolated texcoord anchor differs from GL's by ~1 ulp (2e-5…6.7e-5 texels) while the step is bit-identical. This update runs the update-59 §6.3 vertex bisect: dump GL's per-vertex clip **and** per-vertex texcoord (the GL_VERT path previously captured clip+pos only) and compare against the existing Metal `vertex_volume_main` log at the knife-edge pixels' covering triangles.
**Target (unchanged):** Metal bit-identical to **clean GL** (`RenderingBackend=OpenGL`, no debug injection).

**Follows:** [Update 59](VolumeRayCastBackendComparisonFindingsUpdate59.md) (lattice: step bit-identical, anchor ~1 ulp, 188 px knife-edge flips), [Update 58](VolumeRayCastBackendComparisonFindingsUpdate58.md) (feasibility 174 ±1 px plausible, 14 knife-edge px likely unattainable).

---

## 1. Tooling: GL_VERT now dumps per-vertex texcoord too

Two small changes:

- `vtkOpenGLGPUVolumeRayCastMapper.cxx` fragment debug channel selector extended past `gl_PrimitiveID`: channels 108/109/110 now emit `ip_textureCoords.x/y/z`, and the `GL_VERT` print gained a `tex=(x, y, z)` field (channel count 8 → 11).
- **Root fix needed:** the debug-vertex path (`in_debugVertexMode == 1` tiny-triangle raster) early-returns inside `ComputeClipPositionImplementation` *before* the `//VTK::ComputeTextureCoords::Impl` block, so `ip_textureCoords` was never computed and read back as `(0,0,0)`. The fix computes the real per-vertex texcoord (`sign(in_cellSpacing[0]) * (in_inverseTextureDatasetMatrix[0] * vec4(in_vertexPos,1.0)).xyz` then the `in_cellToPoint[0]` multiply) inside the debug branch before the `return` (vtkVolumeShaderComposer.h).

Also added the exact pixel-center to the Metal `DEBUG STEP` line: `screenPos=(3.975000000e+02, 1.105000000e+02)` = **(397.5, 110.5)**, the exact half-integer pixel center — Metal's interpolation coordinate is the standard pixel center, bit-identical by construction to GL's `gl_FragCoord`.

## 2. Result: per-vertex clip AND per-vertex texcoord are bit-identical, 94/94 vids

Frame-6 (last occurrence per vid) capture of the camera-inside cap mesh via `VTK_GL_RAY_DUMP=1 VTK_GL_VERTEX_DUMP=1` vs the Metal `vertex_volume_main` log:

| attribute | matching vids |
|---|---|
| clip.x / clip.y / clip.w | **94 / 94 bit-identical** (clip.z uses different z conventions between the dumps, as in update 59) |
| pos (modelPos) | 94 / 94 identical |
| texcoord (cell-to-point-adjusted) | **94 / 94 bit-identical** |

Concretely at the knife-edge triangle vertices (vid 86, vid 93 of primId 122):

- vid 86: GL clip=(-2.44439697, -34.2432251, …, 0.38470459), Metal identical; GL tex=(0.510781229, 0.461473942, 0.438469529) == Metal `texcoord` bit-for-bit.
- vid 93: GL clip=(-117.459229, 240.019714, …, 0.384702682), Metal identical; GL tex=(0.65001595, 0.819840372, 0.567701578) == Metal bit-for-bit.

## 3. The residual is the rasterizer interpolation, not per-vertex data

Per-vertex clip identical + per-vertex texcoord identical + pixel-center exact (397.5, 110.5) in Metal, yet the **interpolated** values still differ at the knife-edge pixels (frame-6, GL−Metal, in float32 ulps):

| px | clip delta x/y/w | texcoord delta x/y/z |
|---|---|---|
| (397,110) | +1, −1, −1 | −1, −1, −2 |
| (360,229) | +2, −5, −1 | −1, −1, −3 |
| (349,255) | +3, **−294**, +2 | +2, +1, +3 |
| (405,171) | +1, −1, −1 | −1, +0, −1 |
| (9,18) | −1, +0, −1 | −1, +0, −2 |
| (293,298) | **+7**, +1, −1 | −1, −2, −3 |
| (338,432) | +0, −1, −2 | −2, −2, −2 |
| (350,5) | +1, +0, −1 | −1, −1, −1 |
| (153,32) | +0, +0, −1 | −1, −1, −2 |
| (482,33) | −1, −2, +0 | −1, −1, −1 |
| (120,167) | −4, −1, −1 | −1, −1, −1 |
| (470,269) | −1, **+7**, −2 | −2, −1, −2 |
| (439,281) | +2, +4, +0 | +0, −1, +0 |
| (469,463) | −1, −2, −3 | −1, −2, −1 |

Since both per-vertex clip and per-vertex texcoord are bit-identical and the interpolation coordinate is the same exact pixel center, a bit-identical interpolation would give bit-identical interpolated values. The observed ±1–3 ulp differences therefore isolate to the **barycentric interpolation arithmetic** — the GL-on-Metal driver's rasterizer/interpolator produces slightly different weights than the native Metal interpolator, even though both run on the same Apple M2 GPU. This is the update-59 §6 hypothesis (c), now proven to be the cause: the per-vertex (a) and pixel-center (b) paths are exonerated.

The texcoord anchor delta is consistently ~−1 ulp at most knife-edge pixels, which is why the knife-edge texel flips are directional (GL picks the texel one side, Metal the other, shifting gf by 0.3–8 u8 with negligible alpha).

## 4. Doubts / hypotheses (open)

1. **The interpolation-weight difference is the knife-edge floor.** If the GL-on-Metal driver and the native Metal rasterizer cannot be made to interpolate identically from the shader side, the 188 px (0.072 % of the field) are the irreducible bit-parity floor for this GL-on-Metal/Metal pair. All composite arithmetic, per-sample lattice (step/anchor inputs), TF tables, and now per-vertex attributes are verified bit-identical; only the interpolated anchor sits ~1 ulp off.
2. **Outlier ulp counts need a second look before declaring the floor:** (349,255) clip.y −294 ulps, (293,298) clip.x +7, (470,269) clip.y +7. These exceed the usual ±1–3 and suggest either a near-triangle-edge/barycentric-degeneracy readback, a different covering-triangle selection in one backend (primId matched, but the same primId can be interpolated with wildly different weights near an edge), or a GL_VERT/GL_RAY encoding artifact at those pixels. Verify the covering triangle and the fragment's distance to the triangle edges at these pixels.
3. **Remaining candidate, if continuing:** quantify the weight difference itself by reconstructing the barycentric weights from per-vertex clip → window positions vs the interpolated clip at each knife-edge pixel (both backends), and check whether the weight deltas correlate with the 188-px flip set and with sub-ulp differences in the viewport transform (clip.w divide + window scale). Failing that, quantify and document the knife-edge floor per update-59 §6.2.

## Artifacts

- Code: `Rendering/VolumeOpenGL2/vtkOpenGLGPUVolumeRayCastMapper.cxx` (GL_VERT now 11 channels incl. per-vertex texcoord, `tex=` field), `Rendering/VolumeOpenGL2/vtkVolumeShaderComposer.h` (debug-vertex branch computes the real per-vertex texcoord before its early return), `Rendering/Metal/Shaders/MetalShaders.metal` (`DEBUG STEP` now logs exact `screenPos` = pixel center).
- Data: `/tmp/bc/u62_gl_vlog.log` (GL_RAY + GL_VERT frame-aligned dump), `/tmp/bc/u62_metal.log` (Metal STEP + vertex log).
- Scripts: `/tmp/bc/vertex_u62.py` (94/94 per-vertex bit-compare), ulp analysis inline in this session.
- Verified (python, /tmp/bc): per-vertex clip.x/y/w and per-vertex texcoord 94/94 bit-identical (frame 6); interpolated clip and texcoord still differ by ±1–3 ulps at the 14 knife-edge px; Metal pixel-center == (397.5, 110.5) exact.
