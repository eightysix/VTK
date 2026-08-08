# Camera-inside: stage 4 — clip-chain parity and entry/anchor probes all inert; residual is a last-ulp per-sample march effect (update 30)

**Date:** 2026-08-08
**Scope:** Execute update 29's stage-4 candidates against the residual (max |Δ| 22, 307 px ≥5, knife-edge pixels (372,131)/(422,92)). All three probes produced **byte-identical output** (or a 2-pixel change), refuting the depth chain, the entry `tStart`, and the interpolated anchor as causes. Also corrects update 29's reading of the P-matrix diff: the two differing entries are both in the Z row and never touch XY/w, so the P difference is inert for barycentric interpolation.

**Follows:** [Update 29](VolumeRayCastBackendComparisonFindingsUpdate29.md).
**Status of the working tree:** clean (back at `d75c70452c`); the update-29 `GL_CLIPMAT`/`MTL_CLIPMAT` matrix dumps remain committed as TEMP DEBUG diagnostics. All stage-4 probe edits were reverted after measurement.

---

## 1. Correction of update 29's P-matrix reading

Decoding the actual bytes (not the summary text) for the two differing projection entries:

| Metal dump index (col,row) | vtk entry | GL | Metal | role |
|---|---|---|---|---|
| (2,2) | `P[2][2]` | `9a4180bf` = −1.002002 | `cd2080bf` = −1.001001 | Z scale `-(f+n)/(f-n)` |
| (3,2) | `P[2][3]` | `8143c5be` = −0.385280 | `814345be` = −0.192640 | Z offset `-(2fn)/(f-n)` |

The X scale (`P[0][0]`), Y scale (`P[1][1]`), the w-row (`P[3][2]` = −1, `P[3][3]` = 0) are **byte-identical**. The `w` row is what enters the perspective divide for XY: `w = -z_view` identically on both backends, and `X_clip = P[0][0]·x`, `Y_clip = P[1][1]·y` are identical. **Therefore the P difference cannot change window-space X/Y/w, the barycentric weights, or the interpolated anchor at all** — update 29's note that "one differing entry enters the perspective divide" was wrong; the MetalShaders.metal comment's conclusion ("irrelevant to XY/w") was correct. The Z row only feeds the depth texture and `ndcToVolume` far-termination reconstruction.

## 2. Probe 1 — feed Metal GL's exact P (nearz=-1/farz=1) + shader Z remap + depth readback conversion

Change: rebuilt `P` in the mapper's metalCamera branch via `GetProjectionTransformMatrix(tiledAspect, -1, 1)` (GL's exact call), uploaded it plus the recomputed CPU-composed `ViewProjectionMatrix`, added the vertex-shader clip-Z remap `z = 0.5·z + 0.5·w` (Metal [0,1] depth), and converted the two fragment-shader depth readbacks back to GL NDC (`depthSample*2-1`) before unprojecting. The fallback (non-metalCamera) branch was updated to match.

- `MTL_CLIPMAT` re-dump: **P byte-identical to GL** across all 16 floats (both `9a4180bf` and `8143c5be` present); V and M unchanged byte-identical.
- **Result: rendered output byte-identical to update 28/29** — max |Δ| 22, 307 px ≥5, same pixels, same values at (372,131)=(1,18,22) and (422,92)=(0,16,19).
- **Refuted.** The depth chain (stored depth value, `ndcToVolume` inverse, far-termination) has zero effect on this test's march — consistent with the depth-texture gate being effectively inactive (`depthSample` = clear value 1.0, or `tTerminateMax` ≥ box exit so `min()` picks the box exit).

## 3. Probe 2 — unconditional anchor-based entry (GL parity)

Change: in `marchVolume`'s camera-inside branch, replaced `tStart = max(dot(localPos-cameraPos,rayDir), analyticEntry)` with `tStart = dot(localPos-cameraPos,rayDir)` unconditionally, mirroring GL's `g_rayOrigin = ip_textureCoords + one step`.

- **Result: byte-identical output** — max |Δ| 22, 307 px, same pixels.
- **Refuted.** For the affected fragments the analytic near-plane entry and the anchor distance round to the same float32 `tStart` (or the difference lies below the march's sensitivity), so the `max()` was already a no-op. The entry is not the cause.

## 4. Probe 3 — perturb the interpolated anchor

Change: in `fragment_volume_main`'s camera-inside path, `localPos += float3(1e-7)` in [0,1] volume space (~2e-5 data units — the scale of the measured MT-GL anchor delta). `anchorData` (data-space, used for the GL-parity step direction) left unchanged.

- **Result: residual 307 → 309 px** (two pixels added); the two largest-Δ pixels are **unchanged at byte level** ((372,131) = (1,18,22), (422,92) = (0,16,19); max |Δ| stays 22).
- **Refuted.** A position perturbation 2–3× the measured anchor delta moves only 2 of 307 residual pixels. The knife-edge flips are not anchor-position-driven; perturbing the anchor does not relocate them.

## 5. Interpretation after stage 4

With **byte-identical** cap meshes (95 verts / 126 indices), **byte-identical** `V`/`M`, **byte-identical or inert** `P` (Z row only), **byte-identical** near-plane data (GL/METAL_NEARPLANE agree to the printed double), and the residual **insensitive to the depth chain, the entry tStart, and the anchor position**, the residual lives in the **per-sample march float32 chain**: the step-size/step-position accumulation, the far-side sample count (`maxSteps = ceil((tEnd - firstT)/stepSize)` vs GL's position-bounds march loop), or the scalar sampling/transfer lookup at individual samples.

This is consistent with the update-24 measurement (frame-aligned step agreement only ~0.006%, i.e. not bit-exact): over ~1150 samples a ~0.006% step-length difference accumulates to ~0.06 of a sample at the far end — enough to flip the far-side knife-edge sample at the scalar-1150 iso, and immune to anchor/entry/matrix corrections. The remaining candidates are:
1. **Sample count at the far side** — Metal precomputes `maxSteps` from `tEnd`/`stepSize`; GL iterates `g_dataPos += g_dirStep` and breaks on texture-space bounds. A one-sample count difference flips exactly these iso pixels.
2. **Per-sample step-position accumulation** — Metal accumulates in [0,1] volume space (`currentPoint += rayDir*stepSize`) then converts per sample; GL accumulates directly in texture space (`g_dataPos += g_dirStep`). Different float32 rounding per step.
3. **Scalar sampling** — trilinear-adjacent float32 differences in the sampled scalar near the 1150 iso.

## 6. What is now definitively excluded

- Vertex-attribute / cap-mesh differences (update 27/28: byte-identical after corner-order fix).
- Back-face culling / winding (update 28: CCW fix).
- Projection near/far convention in the clip chain (update 29/30 probe 1: P Z row inert for XY/w and for the depth chain in this test).
- Camera-inside entry `tStart` (probe 2: no-op).
- Interpolated anchor position (probe 3: 2/307 pixels).

## 7. Wrap-up notes / pending cleanup

- The matrix-dump instrumentation (`GL_CLIPMAT` in `vtkOpenGLGPUVolumeRayCastMapper.cxx`, `MTL_CLIPMAT` in `vtkMetalGPUVolumeRayCastMapper.mm`) is committed as TEMP DEBUG and should be removed when the investigation concludes, together with the update-26 in-shader `P*V*M*v` Metal change (which the stage-4 probes showed is not the deciding factor).
- The update-29 "one differing P entry enters the perspective divide" claim is superseded by section 1.
- If a fix is pursued later, section 5's candidate 1 (far-side sample-count parity: replace Metal's precomputed `maxSteps` with GL's position-bounds termination, or vice-versa) is the highest-value next probe and needs no new instrumentation — only the two existing images plus `cmp.py`.

## 8. Reproduction

```sh
# probe 1 (P parity): Metal P becomes byte-identical to GL; output unchanged
build_macos_metal/bin/vtkRenderingVolumeCxxTests \
  TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
  --vtk-factory-prefer RenderingBackend=Metal -D build_macos_metal/ExternalData/Testing \
  -T build_macos_metal/Testing/Temporary -V /tmp/bc/dummy_baseline.png
# compare vs the GL capture: python3 /tmp/bc/u30/cmp.py OpenGL.png Metal.png  ->  max|d| 22, 307 px
```

Artifacts: `/tmp/bc/u30/{OpenGL,Metal,Metal_c2,Metal_perturb}.png` + `.log` (u30 = probe 1, `_c2` = probe 2, `_perturb` = probe 3).
