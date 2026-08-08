# Camera-inside: stage 3 — the clip chain's only non-bit-identical float32 input is the PROJECTION matrix's near/far rows (update 29)

**Date:** 2026-08-08
**Scope:** With the cap meshes byte-identical (update 28), byte-compare the three float32 matrices each backend feeds its vertex shader for `P*V*M*v` (GL: `in_projectionMatrix * in_modelViewMatrix * in_volumeMatrix[0]`; Metal: `projectionMatrix * modelViewMatrix * volumeToWorld`, MetalShaders.metal:2937). Added TEMP DEBUG matrix dumps on both sides (`GL_CLIPMAT` at vtkOpenGLGPUVolumeRayCastMapper.cxx:4015, `MTL_CLIPMAT` at vtkMetalGPUVolumeRayCastMapper.mm:7218). **Result: `V` and `M` are byte-identical across all 16 float32 entries; `P` differs in exactly two entries — both near/far-dependent, from GL using nearz=-1/farz=1 (`vtkOpenGLCamera.cxx:133`) vs Metal using nearz=0/farz=1 (`vtkMetalCamera.mm:50-51`).** This is the only remaining float32 difference in the clip chain and the prime candidate for the ~1e-5 anchor drift that drives the scalar-1150 knife-edge flips.

**Follows:** [Update 28](VolumeRayCastBackendComparisonFindingsUpdate28.md).
**Status of the working tree:** uncommitted; the two matrix-dump instrumentations (TEMP DEBUG), plus the committed update-28 fixes (boxSource corner order, CCW winding), the update-27 cap-mesh dumps, and the update-26 in-shader `P*V*M*v` Metal change.

---

## 1. What was dumped and how

Each backend prints the 64 float32 bytes of its three clip-chain matrices at the per-render uniform upload:

- **GL** (`SetCameraShaderParameters`): `P` = `in_projectionMatrix` (double→float via `SetUniformMatrix`, element `(r,c)`), `V` = `in_modelViewMatrix`, `M` = `in_volumeMatrix[0]` (first 16 floats of `this->VolMatVec`).
- **Metal** (`UpdateShaderParameters`, metalCamera branch): `P` = `uniforms.ProjectionMatrix`, `V` = `uniforms.ModelViewMatrix`, `M` = `uniforms.VolumeToWorldMatrix` — the exact bytes uploaded.

6 blocks per backend (per-frame uniform rebuild, same camera sequence).

## 2. Result of the byte-compare

| matrix | GL | Metal | byte-identical? |
|---|---|---|---|
| **V** (modelView / view) | `…` | `…` | **yes** (16/16 floats) |
| **M** (volumeToWorld) | `VolMatVec[0..15]` | `VolumeToWorldMatrix` | **yes** (16/16 floats) |
| **P** (projection) | `ecd96e40 00000000 00000000 00000000 ecd96e40 00000000 00000000 00000000 9a4180bf 000080bf 00000000 00000000 8143c5be 00000000 00000000 00000000` | `ecd96e40 … ecd96e40 … cd2080bf 000080bf 00000000 00000000 814345be …` | **no — exactly 2 of 16 entries differ** |

The two differing P entries (all frames, both camera phases):

| P entry | GL | Metal |
|---|---|---|
| near/far-dependent entry #1 | `9a4180bf` (−1.0029) | `cd2080bf` (−1.0010) |
| near/far-dependent entry #2 | `8143c5be` (−0.3855) | `814345be` (−0.1927) |

The X/Y scaling entries (`ecd96e40` ≈ 3.73, the `1/tan(θ/2)`-derived values) are **identical**. The differing entries are exactly the ones `vtkCamera::ComputeProjectionTransform` builds from the `nearz`/`farz` arguments (`vtkCamera.cxx:521-522`): `-(f+n)/(f-n)` and `-(2fn)/(f-n)`.

## 3. Interpretation

- **Update 25's hypothesis is now pinned to a concrete input.** The rasterizer interpolation, the mesh (95 verts / 126 tris byte-identical), the view matrix, and the model matrix are all bit-identical. The only float32 difference upstream of the window-space barycentric weights is the projection matrix's near/far rows.
- **The MetalShaders.metal:2863 comment ("nearz=0 vs -1 only changes the Z row, irrelevant to XY/w") is NOT fully accurate.** The dump shows two entries differ, not one; and the second differing entry (in the W/last row of the frustum) enters the perspective divide, so it is not strictly confined to Z. This needs a definitive check (stage 4) via actual shader-output window-space positions or a direct A/B of `P` swap.
- **Why it may still be a ~1-ulp-scale effect:** the visible images are near-identical (max |Δ| 22, 307 px ≥5), so the P difference is not grossly changing the proxy projection — it perturbs clip/w in the last ulps, which shifts window-space vertex positions by ~5e-5 px, perturbing barycentric weights → ~1e-6..1e-5 data-unit anchor shift → scalar-1150 knife-edge flips. Exactly the update-25 budget.

## 4. The fix direction (stage 4 candidates)

1. **Feed Metal GL's exact `P` (nearz=-1/farz=1) with a shader-side Z remap.** The XY/w rows then match GL's byte-for-byte; Z is remapped from GL's [-1,1] to Metal's [0,1] depth convention (`zc = (zndc+1)*0.5` or a small matrix factor) so Metal's depth/clip semantics stay intact. Direct A/B: if the residual knife-edge pixels drop to 0, the near/far rows were the cause.
2. **Single CPU-composed float32 `P*V*M` fed to both** (update-25 preferred fix): removes GPU FMA/order as a variable entirely; then both shaders do one `mat*vec`. Requires GL-side change (shader uses composed matrix) or a GLSL-equivalent; more invasive.
3. Verify with `compare_gl_metal_accum.py` at (422,92)/(372,131) and the whole-image survey (currently max |Δ| 22, 307 px ≥5): expect knife-edge flips → 0.

## 5. Next steps (stage 4)

1. Try fix candidate 1 (Metal P ← GL nearz=-1 values, shader Z remap), re-dump `MTL_CLIPMAT` to confirm P now byte-identical, then re-run the whole-image survey + per-sample compare at the knife-edge pixels.
2. If the residual persists with byte-identical P, the divergence is in the GPU shader float32 multiply itself (GLSL vs MSL FMA/order) → proceed to fix candidate 2 (CPU-composed P·V·M).
3. Clean up: remove the matrix-dump + cap-mesh TEMP DEBUG instrumentation; revert update-26's in-shader change if candidate 2 replaces it.

## 6. Reproduction

```sh
./macos_metal_build.sh --resume
build_macos_metal/bin/vtkRenderingVolumeCxxTests \
  TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
  --vtk-factory-prefer RenderingBackend=OpenGL -D build_macos_metal/ExternalData/Testing \
  -T build_macos_metal/Testing/Temporary -V /tmp/bc/u28/baseline.png >/dev/null 2>/tmp/bc/u28/gl/mat.log
build_macos_metal/bin/vtkRenderingVolumeCxxTests \
  TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
  --vtk-factory-prefer RenderingBackend=Metal -D build_macos_metal/ExternalData/Testing \
  -T build_macos_metal/Testing/Temporary -V /tmp/bc/u28/baseline.png >/dev/null 2>/tmp/bc/u28/metal/mat.log
# byte-compare: V and M SAME (16/16), P differs at the two near/far entries (GL_CLIPMAT vs MTL_CLIPMAT)
```

Artifacts: `/tmp/bc/u28/gl/mat.log`, `/tmp/bc/u28/metal/mat.log`.
