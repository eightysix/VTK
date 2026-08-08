# Camera-inside: update 26 — in-shader P·V·M·v (Metal-side) attempt does NOT close the anchor gap; the anchor delta is ~16× larger than the 1-ulp matrix budget (update 25 H1 refuted)

**Date:** 2026-08-08
**Scope:** Implemented the update-25 preferred fix **Metal-side only** (make the Metal vertex shader compute `P*V*M*v` in-shader exactly like GL, instead of feeding it the CPU-precomputed `viewProjection`), rebuilt, and re-surveyed. Result: **the per-sample anchor drift and the scalar-1150 knife-edge flip are unchanged.** Worse, the measured data-space anchor delta (~3–5e-5) is ~16× larger than update 25's 1-ulp budget (~2.5e-6), which refutes the "clip-position ulp from P·V rounding" hypothesis as the sole cause. The residual is likely a larger systematic difference (vertex-attribute values and/or window-space mapping), not a last-bit matrix rounding.

**Follows:** [Update 25](VolumeRayCastBackendComparisonFindingsUpdate25.md) (H1 attribution: `(P·V)` CPU-vs-shader rounding) and update 23 (Metal vertex path matches GL's interpolated data-space anchor "to float32", not bit-exact).

---

## 1. What was changed (Metal only; GL untouched)

Per the update-25 recommendation, the Metal backend now feeds the vertex shader the same three matrices GL uses and computes the clip position in-shader:

```cpp
// vtkMetalGPUVolumeRayCastMapper.mm — VolumeMapperUniforms, offsets 1712/1776
float ProjectionMatrix[16];   // 1712..1775
float ModelViewMatrix[16];    // 1776..1839  (struct 1712 -> 1840, static_asserts updated)
```
filled from `vtkMetalCamera` scene transforms (`memcpy` of the column-major float32 P/V; fallback branch casts `P4/V4` doubles identically), and

```metal
// MetalShaders.metal vertex_volume_main (was: viewProjection * volumeToWorld * v)
out.position = volumeUniforms.projectionMatrix * volumeUniforms.modelViewMatrix *
    volumeUniforms.volumeToWorld * float4(modelPos, 1.0);
```

This mirrors GL's `ComputeClipPositionImplementation` exactly:

```glsl
gl_Position = in_projectionMatrix * in_modelViewMatrix * in_volumeMatrix[0] *
  vec4(in_vertexPos.xyz, 1.0);   // vtkVolumeShaderComposer.h:87
```

`projectionMatrix` keeps Metal's nearz=0/farz=1 (only the Z row differs from GL's nearz=-1; rows 0,1,3 — which set clip.x, clip.y, clip.w — are bit-identical). `viewProjection`/`inverseViewProjection` stay for the fragment depth paths.

Verified active: `build_macos_metal/Rendering/Metal/vtkMetalShaders.cxx` contains the new expression; incremental build exited 0.

## 2. Re-survey (NoShadeNoGradOpNoTransformNoJitter, dummy-baseline capture)

Image delta (`analyze.py`), 512×512:

| metric | value |
|---|---|
| center pixel | GL == Metal `[236 180 145]` |
| delta mean / mean\|Δ\| / max\|Δ\| | −0.09 / 0.29 / **22** |
| per-channel fit | R 0.9997, G 0.9999, B 0.9998 |
| masked (≥5) px | **262** |
| mask histogram | 222∈[5,8), 26∈[8,12), 11∈[12,16), 3∈[16,22) |
| mask bbox | y 23..291, x 216..487; densest rows 81 (51 px), 103 (26 px) |

Per-sample at (422,92) Metal / (422,419) GL (`compare_gl_metal_accum.py`):

- first divergence **i=132**: GL raw 0.0162814 / op 0.127265 vs MT raw 0.0178680 / op 0.400904 — the sample still lands on the other side of the scalar-1150 knot.
- accumulation replay at i=132: GL accC (0.652, 0.465, 0.363) vs MT (0.742, 0.566, 0.457); final delta still driven by that flip.
- position linear fit (i=10..172): GL pos0=(0.50542762, 0.50653368, 0.45073802) step=(−4.5359e-4, −5.13e-6, 1.8373e-3); MT pos0=(0.50542840, 0.50653433, 0.45073820) step=(−4.5350e-4, −5.01e-6, 1.8373e-3); rel |step| diff x +0.019%, y +2.33%, z −0.003%.
- drift grows to (−1.6e-5, −2.1e-5, −1.0e-5) at i=170 — **indistinguishable from pre-fix (update 23/24 values)**.

**Conclusion: changing the Metal clip-space path changed nothing. The anchor divergence is not produced by the `(P·V)` CPU-vs-shader rounding.**

## 3. Direct anchor comparison (data space)

The GL `GL_RAY` dump's `vpos` is the interpolated `ip_vertexPos` (data space; debug channels 6–8). The Metal `STEP` log's `localPos` is the interpolated anchor in [0,1] volume space; with `boundsSize=(201.6, 201.6, 138.0)` (511·spacing, origin 0) from the same log:

| | x | y | z |
|---|---|---|---|
| GL vpos (data) | 101.987946 | 102.120789 | 61.934486 |
| MT localPos→data | 101.987991 | 102.120843 | 61.934515 |
| **Δ (MT−GL)** | **+4.5e-5** | **+5.4e-5** | **+2.8e-5** |

- This is ~**16×** update 25's budget for a single clip-space ulp (≈2.5e-6 data units over a ~200 px / ~10-unit cap triangle).
- It is consistent with the observed drift: `Δanchor/|eye−anchor| ≈ 4e-5/0.39 ≈ 1e-4` relative direction change × stepSize 1.9e-3 → ~1.2e-7/sample drift rate, matching the fits.

**Implication:** a few last-bit weight differences from matrix rounding cannot produce a 3–5e-5 data-unit anchor delta. Either the **vertex attribute values** uploaded by the two backends are not bit-identical, or the **window-space mapping** (view→viewport/pixel centers) differs by ≳1e-3 px, or the barycentric interpolation diverges systematically.

## 4. Ray-direction derivation (both backends, for reference)

- GL `computeRayDirection()` = `normalize(ip_vertexPos.xyz − in_eyePosObjs[0].xyz)` — **dataset space** (vtkVolumeShaderComposer.h:1716).
- Metal fragment main: `rayDir = normalize(localPos − cameraPos)` in [0,1] volume space, but for the camera-inside proxy path `dirObj = normalize(anchorData − cameraData)` in dataset space (MetalShaders.metal:3836-3845), then `evalStep = (adjustedLin * dirObj) * sampleDistanceWorld` replicating GL's `g_dirStep = (ip_inverseTextureDataAdjusted * normalize(vertexPos−eyePos)).xyz * in_sampleDistance`. The chain is already GL-parity; the divergence feeds in from the **anchor** (section 3), not the direction formula.

## 5. Next probes (ordered)

1. **Dump the cap-triangle vertex attributes** on both backends (the 3 densified/clipped vertex data-space positions after float32 conversion) — byte-compare. Rules out/in the interpolation input directly. (GL side allowed for probing per project convention; GL logic untouched.)
2. **Dump window-space vertex positions** (`gl_Position`-derived `gl_FragCoord` barycentrics vs `out.position`-derived) for the cap triangle to bound the window-space mismatch (is it ulp-scale or ≳1e-3 px?).
3. If both match: dump the **perspective-correct interpolation weights** at (422,92) on both backends (barycentric triple in float32) — a GPU-rasterizer parity check on the same hardware.
4. Re-check whether the **densify geometry** (`vtkClipConvexPolyData` + `vtkDensifyPolyData` output, double→float) is byte-identical between backends (update 23 assumed yes but it was never byte-compared).

## 6. Reproduction

```sh
# image survey (NoJitter variant), dummy-baseline capture
python3 -c "from PIL import Image; Image.new('RGB',(512,512),(0,0,0)).save('/tmp/bc/TestGPURayCastCameraInsideTransformation.png')"
for BE in OpenGL Metal; do
  build_macos_metal/bin/vtkRenderingVolumeCxxTests \
    TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=$BE \
    -D build_macos_metal/ExternalData/Testing -T build_macos_metal/Testing/Temporary \
    -V /tmp/bc/TestGPURayCastCameraInsideTransformation.png
done
python3 Rendering/Metal/BackendComparisonTools/analyze.py \
  /tmp/bc/gl/TestGPURayCastCameraInsideTransformation.png \
  /tmp/bc/metal/TestGPURayCastCameraInsideTransformation.png /tmp/bc/njit

# per-sample anchor (GL vpos vs MT localPos->data)
# GL: VTK_GL_RAY_DUMP=1 ... -> GL_RAY vpos=(...)
# Metal: MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=16777216 MTL_LOG_TO_STDERR=1 ... -> STEP localPos=
python3 Rendering/Metal/BackendComparisonTools/compare_gl_metal_accum.py \
  /tmp/bc/gl_samples.log /tmp/bc/metal_samples.log 422 92
```

## 7. Status of the working tree

Committed with this doc: the Metal-side in-shader `P*V*M*v` change (uniforms struct + vertex shader + static_asserts 1712→1840) as the update-26 exploration. **It is an unsuccessful fix attempt** (anchor drift unchanged); if probe #1/#2 attribute the gap elsewhere, this change should be reverted before the real fix lands.
