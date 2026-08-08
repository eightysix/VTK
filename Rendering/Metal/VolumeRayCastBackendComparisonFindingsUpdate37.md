# Fragment-stage residuals: interpolated anchor differs by 1–2 ULP despite bit-identical vertex stage (update 37)

**Date:** 2026-08-09
**Scope:** With the vertex stage closed out (update 36 — per-vertex clip x/y/w bit-identical, mesh/matrices bit-identical), this update isolates the remaining 130 masked-pixel residual to the **fragment-stage interpolation**: Metal's interpolated `in.localPos`/`in.clipPos` differ from GL's `ip_vertexPos`/`ip_debugClip` by 1–2 ULP at the same physical pixel and frame, even though every input to the interpolation is bit-identical. That small anchor shift is amplified into the per-step `g_dirStep` vs `evalStep` drift because the direction's y-component is the difference of two nearly-equal numbers (camera near the volume axis). The matrix, sampleDistance, and mat-vec dot-contraction hypotheses are refuted for this test.

**Follows:** [Update 36](VolumeRayCastBackendComparisonFindingsUpdate36.md).

---

## 1. Setup and frame alignment (corrections from earlier sloppy comparisons)

All comparisons below are **last-frame, same camera state**:

- GL capture (`gl_ray_fresh.log`, `VTK_ln_DUMP`): pixel (422,419) = glReadPixels coords (bottom-left origin). Two camera states exist across the 6 frames; the **last** frame (state B) has `cam=(102.122314, 102.122314, 61.5619835)`.
- Metal capture (`mt_full7.log`, MTL_LOG_TO_STDERR): pixel (422,92) = screenPos (top-left origin). Same physical pixel is GL (422, 511−92) = (422,419) for a 512×512 viewport. The last frame (00:38:15) has `cameraVol=(0.5065590739, …)` → `×201.6 = 102.122314`, i.e. the same camera.
- Frame-count cross-check: 6 STEP frames for (422,92) (3+2+1), 570 `vertex_volume_main` lines = 6 frames × 95 vertices, timestamps aligned 00:38:13 (285) / :14 (190) / :15 (95).

Last-frame values at the probe pixel:

| quantity | GL (422,419) | Metal (422,92) |
|---|---|---|
| interpolated clip | (0.250208169, 0.245701194, 0.00019452881, 0.384704709) | (0.250208557, 0.245698720, 0.192449659, 0.384704649) |
| interpolated anchor (data space) | vpos (101.987953, 102.120819, 61.9345016) | anchorData (101.9879684, 102.1208115, 61.93450546) |
| step | (−0.000453560555, −4.99691942e-6, 0.00183728477) | evalStep (−0.000453566, −5.07427e-6, 0.001837282) |

The interpolated clips differ (x by 3.9e-7, y by 2.5e-6, w by 6e-8) and the anchors differ (x by 1.5e-5 = 2 ULP of 102, y by 7.5e-6, z by 3.9e-6). `clip.w` differences matter because they enter the perspective divide → the barycentric weights differ slightly.

## 2. Hypotheses refuted for this test

1. **Matrix values** — `ip_inverseTextureDataAdjusted = cellToPoint * in_inverseTextureDatasetMatrix` (GL) vs Metal's `adjustedLin`/`VolumeToTextureMatrix`. The GL uniform dump and the CPU inversion (`vtkMatrix4x4::Invert` on a diagonal) produce identical float32 matrices; both are diagonal for the NoTransform test.
2. **sampleDistance** — identical on both backends (GL `GL_UNIFORMS sampleDist=0.270058721`, Metal `sampleDistanceWorld=0.270058721`).
3. **Dot contraction in the step** — for a diagonal matrix the mat-vec `(E * vec4(rayDir,0)).xyz` reduces to a single nonzero product per component, so plain-mul vs FMA patterns are all identical. Emulation confirms all three patterns give the same result.
4. **GL's own printed values are self-consistent only at the pixel level** — reconstructing GL's step from its printed `vpos`/`eye` with CPU float32 `normalize` does **not** reproduce its printed step (CPU `length`/`divide` association differs from the GPU's), so step-level comparisons must use per-component implications, not a CPU normalize.

Conclusion: the step difference is **entirely in rayDir**, i.e. `normalize(ip_vertexPos − eyePos)` (GL) vs `normalize(anchorData − eye)` (Metal), and the anchor difference (1–2 ULP) is amplified: the direction's y-component is `(102.1208 − 102.1223) ≈ −0.0015`, so an anchor shift of 7.5e-6 is a 0.5% relative change in that component, which propagates directly into the step's y.

## 3. Interpolation inputs are all bit-identical — so why do the outputs differ?

Verified bit-identical between backends:
- cap mesh: 95 float32 vertices (`GL_CAPVERT` ≡ `MTL_CAPVERT`, update 28/35), and `GL_CAPINDEX` ≡ `MTL_CAPINDEX` (126 triangles, same vertex order).
- per-vertex clip x/y/w (`P*V*M*v`, update 36, 94/94 signed-exact; Metal z uses the nearz convention, x/y/w identical).
- window size 512×512 (confirmed by inverting the pixel's interpolated clip through the viewport transform: both give exactly (422.5, 419.5)).

Because the same GPU interpolator runs both backends with the same per-vertex values, the same triangle order, and the same window coordinates, the interpolated values *should* be bit-identical — yet the logs show 1–2 ULP differences in both the interpolated clip and the interpolated anchor. The remaining candidates:

1. **Shared-edge / fill-rule triangle selection** — if the pixel center lies on a triangle edge, GL and Metal rasterizers may pick different covering triangles (different top-left rule or sub-pixel rounding), giving different barycentric weights.
2. **Interpolator precision / reference-vertex arithmetic** — a hardware/vendor-specific detail we cannot control from the shader.

Note the residual is confined to knife-edge pixels "on the near-face clip edges" (update 36 §5), consistent with a shared-edge mechanism rather than a global interpolation bias.

## 4. Tooling notes for continuing

- `findtri`-style CPU reconstruction of the covering triangle from the last-frame `vertex_volume_main` dumps is NOT yet reliable: the near-cap vertices (e.g. vid=86, mp=(102.98, 93.02, 60.49)) project to x/w ≈ −6.35 far off-screen, and the nearz convention means logged clip z is not directly comparable to a CPU `P*V*M*v` with the standard projection. X/Y/W of the logged Metal vertex clips do match a CPU recompute of GL's `P*V*M*v` (within double-vs-float32 precision).
- To get the covering triangle authoritatively, the fastest probe is a fragment dump of the flat `ip_vid` (GL) / vertex IDs (Metal) per probe pixel, or a debug rasterization that labels triangles.

## 5. Next probes

1. **Dump the covering triangle's vertex IDs per probe pixel** on both backends (GL `flat out int ip_vid` already exists in `raycastervs.glsl`/`raycasterfs.glsl`; Metal add a flat varying or use the existing per-vertex logging) to decide between shared-edge selection vs interpolator rounding.
2. **Compute the interpolated anchor analytically** in the Metal fragment shader from the flat vertex IDs + per-vertex values + window-space barycentrics and compare to the hardware interpolation, to quantify the interpolator's rounding.
3. If shared-edge is the cause: check whether the two candidate triangles give identical ray-geometry at the pixel (if the pixel is inside the volume interior, both triangles' interpolated anchors should be near-identical — only the sub-pixel rounding differs) — if so, the residual may be unfixable at the rasterizer level and must be handled by matching the fill rule or densifying the mesh along the offending edges.
4. Continue the u36 §6 agenda independently: bit-exact `evalStep` (4×4 in-shader compose) and termination parity (GL position-bounds loop vs precomputed `maxSteps`).

## Artifacts

- `/tmp/bc/u37/gl_ray_fresh.log` (GL, VTK_ln_DUMP), `/tmp/bc/u37/mt_full7.log` (Metal, last frame 00:38:15).
- CPU float32 emulations: `/tmp/bc/u37/emul_dot.py`, `/tmp/bc/u37/emul_gl.py`, `/tmp/bc/u37/findtri.py`.

## Reproduction

GL: `VTK_GL_RAY_DUMP=1 VTK_GL_SAMPLE_DUMP=1 VTK_GL_SAMPLE_DUMP_PX=422,419 build_macos_metal/bin/vtkRenderingVolumeCxxTests TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter --vtk-factory-prefer RenderingBackend=OpenGL -D build_macos_metal/ExternalData/Testing -T build_macos_metal/Testing/Temporary -V /tmp/bc/dummy_baseline.png 2> gl_ray_fresh.log`

Metal: `MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=16777216 MTL_LOG_TO_STDERR=1 … --vtk-factory-prefer RenderingBackend=Metal … 2> mt_full7.log`

Pixel diff: bare `-V <name>.png`; the render lands in `<Temporary>/<name>.png`; compare via the update-34-style `analyze.py`.
