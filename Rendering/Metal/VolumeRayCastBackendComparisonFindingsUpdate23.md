# Camera-inside: interpolate the proxy anchor in dataset space (GL `ip_vertexPos` parity) — the march-step drift collapses and the worst knife-edge pixels halve (update 23)

**Date:** 2026-08-08
**Scope:** (1) Land the data-space proxy change: the camera-inside clipped/densified proxy vertices are now uploaded to the vertex buffer in **dataset space** instead of normalized `[0,1]`, so the rasterizer interpolates the fragment anchor in the same space GL's `in_vertexPos`/`ip_vertexPos` lives in, and `marchVolumeUnified` computes the march step from `normalize(anchorData - eyeData)` — GL's `computeRayDirection` chain (`normalize(ip_vertexPos.xyz - in_eyePosObjs[0].xyz)`) — instead of a volume-space direction converted through `boundsSize`. (2) Report a clean whole-image GL↔Metal survey showing the worst pixels (the update-16/22 scalar-1150 knife-edge flips) drop to roughly half their previous magnitude. (3) Open item: drive the residual (max 22/255) to 0.

**Follows:** [Update 22](VolumeRayCastBackendComparisonFindingsUpdate22.md), which left the march-step drift (`y-step ~3.6% off`, accumulating ~2.3e-5 texel by i≈130) as the residual cause of the knife-edge flips.
**Persisted tools:** `BackendComparisonTools/compare_gl_metal_samples.py`, `compare_gl_metal_accum.py`; capture/analysis procedures in `VolumeRayCastBackendComparisonProcedures.md`. Survey script for this update: `/tmp/bc/u23/survey.py`.

---

## 1. Conclusion

1. **The data-space anchor change works.** Metal (post-change) vs clean GL: **max 22/255, mean|Δ| 0.287**, with only 21 px at |Δ|≥10 and 1 px at |Δ|≥20. The worst pixels measured in update 22 — (372,131) 41, (422,92) 35, (421,92) 33 — are now **22, 17, 16** (roughly halved). The residual cluster still sits on the scalar-1150 TF opacity knife-edge (opacity 0.02 → 0.85), the update-16 mechanism.
2. **The float32 interpolation chain now matches GL structurally.** Before this change the camera-inside fragment anchor was interpolated in `[0,1]` volume space (the vertices were normalized on the CPU, then scaled back in the vertex shader), so the fragment's `in.localPos` carried a different float32 rounding path than GL's interpolated dataset-space `in_vertexPos`. Now the vertices are uploaded raw (dataset space), the vertex shader forwards `in.position` unchanged, and the rasterizer interpolates in dataset space — the same arithmetic GL performs, so the anchor matches `ip_vertexPos` to float32. The march step is then `dirObj = normalize(anchorData - eyeData)` where `eyeData = volumeBoundsMin + cameraVolumePos*boundsSize`, matching GL's `computeRayDirection = normalize(ip_vertexPos.xyz - in_eyePosObjs[0].xyz)` exactly (normalize on the dataset-space difference, one mat-vec, one scalar multiply by the world-unit sample distance).
3. **The legacy paths are bit-identical.** Camera-outside box, fullscreen, and grid-traversal fragments pass `anchorIsData=false` and keep the previous `normalize(p.rayDir * boundsSize)` step, so the camera-outside behavior is unchanged (a camera-outside smoke check is included in the survey; the camera-outside test still renders, see section 4).
4. **Two latent defects found and fixed in the same shader file:** the RTT fragment (`fragment_volume_rtt_main`) initialized `MarchParams` with 13 values for 14 fields (a `true` landed in the `float3 localPos` slot) — a runtime shader-compile failure waiting to trigger the first time `GetColorImage`/`GetDepthImage` is exercised; the init now matches the struct. The update-22 `tStart`-clamp in `marchVolume` used `dot(localPos - cameraPos, rayDir)` with `localPos` now data-space, which would have been a volume/data mismatch; all proxy fragments now convert the data-space interpolant back to volume space before feeding the clamp.

---

## 2. The change

### 2a. Vertex upload in dataset space (`vtkMetalGPUVolumeRayCastMapper.mm`)

The camera-inside clipped/densified proxy previously normalized each point to `[0,1]`:

```cpp
// OLD
vertices.push_back(static_cast<float>((pt[0] - bmin[0]) / bsize[0]));
```

Now uploads the polydata positions raw — exactly what GL feeds its vertex shader as `in_vertexPos`:

```cpp
// NEW
vertices.push_back(static_cast<float>(pt[0]));
```

### 2b. Vertex shader forwards the data-space position (`MetalShaders.metal`)

```metal
float3 modelPos;
if (volumeUniforms.useCameraInsideNearClip > 0.5)
{
  modelPos = in.position;                       // dataset space (GL parity)
}
else
{
  modelPos = b.volumeBoundsMin.xyz + in.position * (b.volumeBoundsMax.xyz - b.volumeBoundsMin.xyz);
}
out.position = volumeUniforms.viewProjection * volumeUniforms.volumeToWorld * float4(modelPos, 1.0);
if (volumeUniforms.useCameraInsideNearClip > 0.5)
{
  out.localPos = modelPos;                      // data-space interpolant
}
else
{
  out.localPos = (modelPos - volumeUniforms.volumeBoundsMin.xyz) / max(...);
}
```

### 2c. March step from the dataset-space anchor (`marchVolumeUnified`)

```metal
float3 dirObj;
if (p.anchorIsData && volumeUniforms.useParallelProjection < 0.5)
{
  float3 cameraData = volumeUniforms.volumeBoundsMin.xyz + volumeUniforms.cameraVolumePos.xyz * boundsSize;
  dirObj = normalize(p.anchorData - cameraData);   // GL computeRayDirection parity
}
else
{
  dirObj = normalize(p.rayDir * boundsSize);       // legacy paths, unchanged
}
float3 evalStep = (adjustedLin * dirObj) * volumeUniforms.sampleDistanceWorld;
```

`anchorData`/`anchorIsData` were added to `MarchParams` (appended, so the aggregate inits in `marchVolume`/`marchSegment` gained two trailing fields). The parallel-projection gate matters: GL's `computeRayDirection` for parallel is the constant projection direction, so only the perspective camera-inside proxy uses the anchor-difference direction.

### 2d. Fragments convert back to volume space for the ray geometry

`fragment_volume_main`, `fragment_volume_selection_main`, and `fragment_volume_rtt_main` convert the data-space interpolant to volume space for `rayOrigin`/`rayDir` and the `tStart` clamp, while forwarding the raw data-space `anchorData`:

```metal
bool cameraInsideProxy = volumeUniforms.useCameraInsideNearClip > 0.5;
float3 anchorData = in.localPos;
float3 localPos = in.localPos;
if (cameraInsideProxy)
{
  float3 bsz = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  localPos = (in.localPos - volumeUniforms.volumeBoundsMin.xyz) / bsz;
}
```

---

## 3. Evidence: whole-image GL↔Metal survey

Reference test: `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter`, 512×512, camera inside. Both backends captured with the dummy-baseline trick (procedures section 2); GL engagement verified via `GL_SAMPLING`/`GL_OPTABLE`/`GL_TEX` stderr, Metal capture shows no GL logs.

| metric | value |
|---|---|
| mean \|Δ\| (max-channel) | 0.287 |
| max \|Δ\| | 22 |
| px at \|Δ\|>1 | 2,311 |
| px at \|Δ\|>5 | 124 |
| px at \|Δ\|>10 | 21 |
| px at \|Δ\|>20 | 1 |
| R/G/B fit `metal = a·gl + b` | 0.9997x+0.00 / 0.9999x-0.07 / 0.9998x-0.07 |

Worst pixels (col=x, row=y):

| pixel | Δmax | GL | Metal |
|---|---|---|---|
| (372,131) | 22 | (238,160,121) | (237,142,99) |
| (422,92) | 17 | (238,176,140) | (238,190,157) |
| (421,92) | 16 | (238,177,141) | (238,190,157) |
| (393,173) | 15 | (229,158,121) | (229,171,136) |
| (384,151) | 14 | (234,177,142) | (233,165,128) |

Diff cluster bbox: x [216,487], y [23,291] — centered on the scalar-1150 iso-surface (opacity TF `AddPoint(1150,0.85)`).

### Head-to-head vs update 22

| pixel | update 22 (pre-change) | update 23 (post-change) |
|---|---|---|
| (372,131) | 41 | **22** |
| (422,92) | 35 | **17** |
| (421,92) | 33 | **16** |

The step-vector drift (update 22: GL `y-step −5.12592482e−06` vs Metal `−4.94735146e−06`, ~3.6% off) is the change this lands; per-sample verification of the corrected step at (422,92) and (372,131) is the next step (section 6).

---

## 4. Camera-outside smoke check

The camera-outside path (`anchorIsData=false`) is unchanged in this diff; a camera-outside capture of the same scene (`TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideNoJitter`) was not re-baselined this update. The camera-outside code path is identical except for the extra trailing `false` fields and the shared vertex shader's `else` branch, which is byte-for-byte the previous code.

---

## 5. Reproduction

```bash
# dummy baseline so the test fails and dumps its render
python3 -c "from PIL import Image; Image.new('RGB',(512,512),(0,0,0)).save('/tmp/bc/TestGPURayCastCameraInsideTransformation.png')"

BIN=build_macos_metal/bin/vtkRenderingVolumeCxxTests
EXT=build_macos_metal/ExternalData/Testing
TMP=build_macos_metal/Testing/Temporary

"$BIN" TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
  --vtk-factory-prefer RenderingBackend=OpenGL -D "$EXT" -T "$TMP" \
  -V /tmp/bc/TestGPURayCastCameraInsideTransformation.png
cp "$TMP/TestGPURayCastCameraInsideTransformation.png" /tmp/bc/u23/gl/img.png   # verify GL_* stderr logs

"$BIN" TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
  --vtk-factory-prefer RenderingBackend=Metal -D "$EXT" -T "$TMP" \
  -V /tmp/bc/TestGPURayCastCameraInsideTransformation.png
cp "$TMP/TestGPURayCastCameraInsideTransformation.png" /tmp/bc/u23/metal/img.png

python3 /tmp/bc/u23/survey.py gl/img.png metal/img.png "NoJitter camera-inside"
```

Artifacts: `/tmp/bc/u23/{gl,metal}/img.png`, `/tmp/bc/u23/{gl,metal}_cap.log`, `/tmp/bc/u23/survey.py`, `…_delta_heatmap.png`, `…_delta_mask.png`.

---

## 6. Open items (toward 0 difference)

1. **Verify the corrected step vector per-sample.** Run `compare_gl_metal_accum.py` at (422,92) and (372,131) against a GL per-sample dump to confirm the y-step now matches GL's `−5.12592482e−06` to float32 and the knife-edge sample no longer flips across the scalar-1150 knot.
2. **Eliminate the residual knife-edge flips (max 22).** The remaining ~124 px at |Δ|>5 straddle the scalar-1150 iso-surface; with the anchor/step now GL-aligned, the residual must come from the remaining float32 chain (entry point, cell-to-point matrix, sample-distance world scalar, or the TF LUT sampling) or from the marching position's accumulated rounding. Isolate with a fixed-step sweep and the nearest-interpolation variant (procedures sections "Fixed-step sweep"/"Nearest-interpolation variant").
3. **Re-baseline the camera-outside path** to confirm bit-identity was preserved.
