# Camera-inside: bit-exact eye (in_eyePosObjs[0]) does NOT change the residual — the drift lives in the evalStep composition (update 33)

**Date:** 2026-08-08
**Scope:** Execute update 32's next probe 1 (bit-exact step parity) one step at a time, starting with the ray anchor: feed the Metal fragment shader GL's exact object-space eye (`in_eyePosObjs[0]`) as a uniform and compute `dirObj = normalize(anchorData − eyeData)` exactly like GL, then re-survey the camera-inside NoJitter residual.

**Result:** The eye is now bit-identical to GL's (both double and float32), but the image residual is **unchanged** (max|d| 22, 307 px ≥ 5, knife-edge flips intact). The `dirObj`/eye was therefore never the discriminator; the per-step accumulation drift measured in updates 24/31/32 persists and now localizes to the `evalStep` composition (Metal's 3×3 `adjustedLin` mat-vec vs GL's 4×4 `ip_inverseTextureDataAdjusted` mat-vec — mathematically equal, float32-different).

**Follows:** [Update 32](VolumeRayCastBackendComparisonFindingsUpdate32.md).

---

## 1. GL's eye is the object-space camera position, computed with a transposed view matrix

`vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::BindTransformations` computes the eye on the CPU as (vtkOpenGLGPUVolumeRayCastMapper.cxx:3790):

```cpp
dataToWorld->Transpose();
vtkMatrix4x4::Multiply4x4(dataToWorld, modelViewMat, dataToView);
dataToView->Invert();
eyePos[i] = dataToView->GetElement(3, i);        // cast to float32
```

`modelViewMat` here is the WCVCMatrix from `vtkOpenGLCamera::GetKeyMatrices` — which is **already transposed** (`vtkOpenGLCamera.cxx:130`), i.e. `transpose(GetModelViewTransformMatrix())`. With the NoTransform test's identity dataToWorld, this yields the camera position in object space (equal to world here): `GL_EYE double=(102.122314, 102.122314, 61.5619835) float=(102.122314, 102.122314, 61.561985)`.

The first Metal attempt used `cam->GetModelViewTransformMatrix()` **without** the transpose and produced `MTL_EYE (0,-0,-0)` (row 3 of the inverse of the non-transposed view matrix is the affine `(0,0,0,1)` row). Transposing the copy first fixes it: `MTL_EYE` is now byte-identical to `GL_EYE`.

The shader change (MetalShaders.metal, `dirObj` in the camera-inside branch) replaces

```metal
float3 cameraData = volumeUniforms.volumeBoundsMin.xyz + volumeUniforms.cameraVolumePos.xyz * boundsSize;
dirObj = normalize(p.anchorData - cameraData);
```

with a direct consumption of the new `EyePosData` uniform (`normalize(p.anchorData - volumeUniforms.eyePosData.xyz)`), mirroring GL's `normalize(ip_vertexPos - in_eyePosObjs[0])`.

## 2. Image survey: no change

Re-captured both backends (dummy-baseline trick, verified GL-engaged via `GL_EYE`/`GL_*` stderr, same last camera frame) and diffed:

| metric | update 32 | after eye parity |
|---|---|---|
| max |d| (of 255) | 22 | **22** |
| masked px (≥ 5) | 307 | **307** |
| (372,131) knife-edge | 22 (GL 238,160,121 vs MT 237,142,99) | **unchanged** |
| (422,92) knife-edge | 19 (GL 238,176,140 vs MT 238,192,159) | **unchanged** |

Conclusion: with the eye bit-identical, the anchor already matched GL to ~1e-5 (update 30 probe 3), so `dirObj` was already effectively GL-equal. The ~1e-4-per-step accumulation drift is **not** introduced by the eye/direction.

## 3. The drift localizes to the evalStep composition

Per-sample logs (Metal shader `os_log` STEP call site, `MTL_LOG_*` env vars; GL `GL_RAY` dump, `VTK_GL_RAY_DUMP=1`; y-flip pairing Metal `(x,y)` ↔ GL `(x, 511−y)`) at the two knife-edge pixels, last frame:

**pixel (372,·)** — Metal screenPos (372,131) / GL glReadPixels (372,380):

| quantity | value |
|---|---|
| GL step (g_dirStep) | (−3.90213e-4, −5.81052e-5, +1.866155e-3) |
| Metal evalStep | (−3.90200e-4, −5.80249e-5, +1.866164e-3) |

**pixel (422,·)** — Metal screenPos (422,92) / GL glReadPixels (422,419):

| quantity | value |
|---|---|
| GL step (g_dirStep) | (−4.53600e-4, −5.10010e-6, +1.837263e-3) |
| Metal evalStep | (−4.53565e-4, −5.02272e-6, +1.837282e-3) |

The two step vectors agree only to ~1e-11…1e-8 (i.e. 1–2 ulp at these magnitudes). Over a ~200-step march that accumulates to the measured ~5e-6…8e-3 texel drift (update 32 table), which is what flips the (422,·) texel-boundary crossing under nearest and shifts the far-side count.

Why they are not bit-identical — GL builds the step as (vtkVolumeShaderComposer.h:437, 107):

```glsl
g_dirStep = (ip_inverseTextureDataAdjusted * vec4(rayDir, 0.0)).xyz * in_sampleDistance;
// ip_inverseTextureDataAdjusted = in_cellToPoint[0] * in_inverseTextureDatasetMatrix[0];  // vertex shader, 4x4 product
```

Metal composes the equivalent linear map as a 3×3 (`adjustedLin = volumeToTexture rows × ctpScale`) and does `evalStep = (adjustedLin * dirObj) * sampleDistanceWorld`. Same matrix on paper, different float32 rounding path (3×3 vs 4×4 product + which matrix is transposed). `sampleDistanceWorld` and GL's `in_sampleDistance` are the same float32 (0.270058721, confirmed from `GL_UNIFORMS` vs Metal STEP logs).

## 4. Next probes

1. **Bit-exact g_dirStep**: make Metal's per-sample advance use GL's exact op order — CPU-compose `cellToPoint(4×4) * inverseTextureDataset(4×4)` the same way the GL vertex shader does, and in-shader compute `(adjustedMat * vec4(dirObj,0)).xyz * sampleDistanceWorld`. Goal: `evalPoint` bit-identical to GL's `g_dataPos += g_dirStep`.
2. **Termination parity** (update 32 probe 2, still pending): replace Metal's precomputed `maxSteps = ceil((tEnd−firstT)/stepSize)` and block-bounds checks with GL's position-bounds loop — `if(any(greaterThan(max(g_dirStep,0)*(g_dataPos − in_texMax[0]),0)) || any(greaterThan(min(g_dirStep,0)*(g_dataPos − in_texMin[0]),0))) break;` — plus `g_terminatePointMax = length(g_terminatePos − g_dataPos)/length(g_dirStep)` (GL termination from the depth-buffer termination point in adjusted texture space, g_terminatePos = rayTermination.xyz/w).

## 5. Reproduction

- Build: `./macos_metal_build.sh --resume --tests`.
- Capture + diff: the dummy-baseline procedure of `VolumeRayCastBackendComparisonProcedures.md` sections 2/3, `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter`, on both `RenderingBackend=OpenGL|Metal`. Verify GL engagement via `GL_EYE`/`GL_*` stderr.
- Per-sample step comparison: Metal STEP logs need `MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=10485760 MTL_LOG_TO_STDERR=1` (shader `os_log`, gated by `debugMarchGate` pxOkContained pixels); GL ray dump needs `VTK_GL_RAY_DUMP=1` (gate list includes the y-flipped pixels).

Artifacts: `/tmp/bc/u33_gl.png`, `/tmp/bc/u33_mt.png` (eye-parity captures), `/tmp/bc/u33_gl_ray.log` (GL_RAY dump), `/tmp/bc/u33_mt_full.log` (Metal STEP os_log dump).
