# WebGPU Volume Ray-Cast Mapper Jiggling Investigation Report

**Date:** 2026-07-03
**Platform:** iOS (WebGPU/Metal backend via Dawn)
**Status:** Root cause unidentified after extensive investigation

---

## Problem Description

When rotating a complex volume rendered with `vtkWebGPUGPUVolumeRayCastMapper` on iOS, the volume exhibits a visible "back-and-forth" jiggling artifact. The camera position itself is smooth (confirmed by per-frame logging), so the jiggling appears as a rendering artifact — the volume image shifts slightly between frames even during smooth camera motion.

**Key characteristics:**
- Visible with complex volumes at low FPS; not visible with simple volumes at high FPS
- Specific to the WebGPU backend — OpenGL volume renderer has no such issue
- The jiggling persists even when the flat-colored diagnostic box renders at 60fps without artifact
- Disabling jittering and changing sample distance do not fix it
- The model matrix is always identity (M00=1, M03=0, M30=0, M33=1)

---

## Architecture Overview

### Rendering Pipeline (per frame)

```
DeviceRender()
  └─ UpdateBuffers()                          [SyncDeviceResources stage]
  │    ├─ UpdateCamera() → CacheSceneTransforms()  (freezes view/projection matrices)
  │    ├─ UpdateGeometry() → volume GPURender(SyncDeviceResources)
  │    │    ├─ Upload textures if changed
  │    │    ├─ Create vertex/index/uniform buffers
  │    │    └─ Create pipeline
  │    ├─ WriteSceneTransformsBuffer()        (view/projection → GPU)
  │    └─ WriteLightsBuffer()
  ├─ ConfigureComputePipelines()
  ├─ PreRenderComputePipelines()
  └─ RecordRenderCommands()                   [RecordingCommands stage]
       ├─ BeginRecording()
       ├─ Clear() (background)
       └─ UpdateGeometry() → volume GPURender(RecordingCommands)
            ├─ Compute model/camera matrices
            ├─ queue.WriteBuffer() → uniform buffer
            └─ renderPass.DrawIndexed()
```

### Gesture Handling (iOS)

- No display link or timer — purely gesture-driven rendering
- `handleRotation:` is NEVER called on iOS (two-finger rotation gesture not used)
- `handlePan:` always uses trackball orbit mode (iOS `modifierFlags` never includes `UIKeyModifierShift`)
- Trackball orbit goes through: `LeftButtonPressEvent` → `MouseMoveEvent` → `LeftButtonReleaseEvent`
- `vtkInteractorStyleMultiTouchCamera` inherits from `vtkInteractorStyleTrackballCamera`
- `Rotate()` method: `camera->Azimuth(rxf)`, `camera->Elevation(ryf)`, `camera->OrthogonalizeViewUp()`

### Shader Architecture

- **Vertex shader:** Transforms model-space cube vertices to clip space via `projection * view * volumeToWorld * vertex`. Computes `localPos` = normalized model-space position in [0,1]³.
- **Fragment shader:** Computes ray direction from camera position to fragment, re-intersects bounding box via `intersectBox`, marches ray front-to-back accumulating color/opacity.
- Fragment shader uses perspective-correct interpolated `localPos` for ray direction computation.

---

## Attempts Made (Chronological)

### Attempt 1: Jitter Seed Fix
**Change:** Replaced `random(input.position.xy)` with `hash3(input.localPos)` in the WGSL shader.
**Rationale:** Screen-space `input.position.xy` changes frame-to-frame during rotation, causing different jitter patterns. Using `localPos` (model-space) makes jitter invariant under rotation.
**Result:** Did not fix jiggling.

### Attempt 2: Blend Factor Fix
**Change:** Changed `blend.color.srcFactor` from `SrcAlpha` to `One`.
**Rationale:** The shader outputs premultiplied alpha (`accumulatedColor, accumulatedOpacity`), but the blend state was using `SrcAlpha` which double-applies the alpha.
**Result:** Did not fix jiggling.

### Attempt 3: Uniform Write Restructuring
**Change:** Moved all volume uniform writes (model matrix, camera position, bounds, sample distance, scalar range) from `RecordingCommands` stage to `SyncDeviceResources` stage. RecordingCommands now only records the draw call.
**Rationale:** Eliminate any temporal gap between SceneTransformBuffer writes and volume uniform writes.
**Result:** Did not fix jiggling.

### Attempt 4: Cached Camera Position
**Change:** Added `vtkWebGPUCamera::GetCachedCameraWorldPosition()` that derives camera position from the cached view matrix in `CachedSceneTransforms`, guaranteeing bit-for-bit consistency with the SceneTransformBuffer.
**Rationale:** Camera position from `GetPosition()` might differ from what the view matrix implies due to floating-point precision.
**Result:** Did not fix jiggling.

### Attempt 5: CullMode::Back Test
**Change:** Changed `CullMode::Front` to `CullMode::Back` in the volume pipeline.
**Rationale:** With front-face rendering, `localPos` would be on the near face, matching the `intersectBox` entry point — eliminating the mismatch.
**Result:** Jiggling persisted. Volume was partially obscured at some angles (front-face rendering has different occlusion characteristics).

### Attempt 6: Direct Azimuth Rotation
**Change:** Bypassed `vtkInteractorStyleMultiTouchCamera` entirely. Computed rotation delta from gesture recognizer's accumulated angle and called `cam->Azimuth(delta)` directly.
**Rationale:** The interactor style's `Roll()` + `ApplyTransform()` path might introduce camera oscillation.
**Result:** Broke rotation — camera could not orbit. Reverted.

### Attempt 7: Pan Handler Fix (Forced Pan Mode)
**Change:** Modified `handlePan:` to always use pan mode instead of trackball orbit, eliminating the `LeftButtonPressEvent`/`MouseMoveEvent` path.
**Rationale:** The trackball orbit through the interactor style might conflict with other camera operations.
**Result:** Broke pan — camera behavior was wrong. Reverted.

### Attempt 8: Triple-Buffered SceneTransformBuffer
**Change:** Created 3 SceneTransformBuffers and 3 bind groups, cycling through them each frame.
**Rationale:** The SceneTransformBuffer (view/projection matrices) was the only GPU buffer written every frame without triple-buffering. On iOS/Metal, `queue.WriteBuffer()` might overlap with a still-executing command buffer reading the same buffer.
**Result:** Fixed garbling (bind group mismatch initially), but did not fix jiggling.

### Attempt 9: Back-to-Front Compositing Shader
**Change:** Eliminated `intersectBox` entirely. Used `localPos` directly as ray origin, marched toward camera with back-to-front compositing.
**Rationale:** The `intersectBox` re-computation might introduce per-frame numerical instability.
**Result:** Did not fix jiggling. Rendering appeared correct but artifact persisted.

### Attempt 10: MSAA Disable
**Change:** Set `SetMultiSamples(0)` on the render window.
**Rationale:** MSAA resolve might introduce per-frame visual artifacts.
**Result:** Did not fix jiggling.

### Attempt 11: Flat-Color Diagnostic (UseJittering=2.0)
**Change:** Shader outputs `localPos` as RGB color — no ray marching, no textures.
**Rationale:** Isolate whether the artifact is in the fragment shader or the vertex transform/presentation.
**Result:** NO jiggling visible at 60fps. But FPS was much higher than volume rendering.

### Attempt 12: Flat-Color with GPU Burn Loop (UseJittering=3.0)
**Change:** Added dummy 200-iteration loop to flat-color shader to slow down rendering.
**Rationale:** Force the flat-color rendering to match the volume's frame rate.
**Result:** Loop was too lightweight — still rendered at 60fps. Test inconclusive.

### Attempt 13: Screen-Space Ray Derivation
**Change:** Added `inverted_view` matrix to `SceneTransforms`. Fragment shader derives volume-space position from `@builtin(position)` via inverse projection → inverse view → world-to-volume transform. Eliminates dependence on interpolated `localPos`.
**Rationale:** Perspective-correct interpolation of `localPos` might introduce per-frame precision variations that get amplified through texture sampling.
**Result:** First attempt failed due to matrix transposition convention error (volume invisible). Fixed convention, volume renders correctly, but jiggling persists.

### Attempt 14: Uniform Write in SyncDeviceResources
**Change:** Moved ALL volume uniform computation and `queue.WriteBuffer()` from `RecordingCommands` to `SyncDeviceResources`. RecordingCommands now only does `SetPipeline`, `SetBindGroup`, `SetVertexBuffer`, `DrawIndexed`.
**Rationale:** `queue.WriteBuffer()` during command encoding might not synchronize properly with the render pass encoder's commands on Metal.
**Result:** Did not fix jiggling.

### Attempt 15: Nearest-Neighbor Volume Sampler
**Change:** Replaced the volume sampler's trilinear filtering (`FilterMode::Linear`) with nearest-neighbor (`FilterMode::Nearest`) for mag, min, and mipmap filters.
**Rationale:** If Metal/iOS trilinear filtering precision varies frame-to-frame for the same coordinates, switching to nearest-neighbor eliminates all interpolation and should stop the shimmer.
**Result:** Jiggling persists. Volume rendering looks blocky (expected) but artifact unchanged.

### Attempt 16: Depth Test Disabled
**Change:** Changed `depthCompare` from `LessEqual` to `Always`, so every volume fragment passes regardless of depth buffer contents.
**Rationale:** The render pass uses `clearDepth=false`. Even though the background fills the depth buffer, edge cases with stale depth values might cause volume fragments to flicker in/out between frames.
**Result:** Jiggling persists. Depth buffer interaction is not the root cause.

---

## Diagnostic Data Collected

### Camera Position Logs
Per-frame logging in `GPURender()` captured: frame number, dt (ms), camera world position, focal point, camera volume-space position, and model matrix elements. Analysis of all log captures showed:
- Camera follows smooth orbital paths around the focal point
- No rapid oscillation at per-frame granularity
- Touch input deltas (dx, dy) are consistent with smooth finger motion

### Pan Gesture Logs
Added logging to `handlePan:` showing: gesture state, trackball flag, touch delta (dx, dy), and camera position. Confirmed:
- All rotation goes through trackball orbit (one-finger drag)
- Touch deltas are smooth and consistent (1-50px per frame)
- No anomalous jumps in touch input

### Key Observations from Logs
- Duplicate frames occur (same camera position, dt≈2ms) — gesture fires Changed events with no finger movement
- Camera distance from focal point stays approximately constant (~1960-1970 units) — proper orbital motion
- Model matrix is always identity (volume at world origin, not transformed)

---

## Root Cause Analysis

### What's Been Ruled Out

| Hypothesis | Evidence Against |
|---|---|
| Camera oscillation from interaction code | Camera logs show smooth motion; touch deltas consistent |
| Buffer aliasing (CPU/GPU race) | Triple-buffered both SceneTransformBuffer and volume uniforms — no improvement |
| Shader intersectBox precision | Eliminated intersectBox entirely — jiggling persists |
| CullMode (back-face entry point mismatch) | Changed to front faces — jiggling persists |
| Perspective-correct interpolation of localPos | Screen-space derivation bypasses interpolation — jiggling persists |
| MSAA resolve artifacts | Disabled MSAA — jiggling persists |
| Blend state incorrectness | Changed blend factors — jiggling persists |
| Presentation latency | Flat-color at 60fps shows no jiggling |
| Interactor style Roll+ApplyTransform | Bypassed with direct Azimuth — broke rotation, suggests this isn't the path |
| Metal trilinear filtering precision | Nearest-neighbor sampler — jiggling persists |
| Texture sampling as root cause | Eliminated all texture reads (flat-color) — no jiggling; nearest-neighbor with textures still jiggles |
| Depth buffer interaction | depthCompare=Always — jiggling persists |

### What Remains

The jiggling persists despite ALL of the following being eliminated or tested:
1. Eliminating the interpolated `localPos` as ray direction source
2. Triple-buffering all per-frame GPU buffers
3. Moving uniform writes before command encoding
4. Disabling MSAA
5. Using the original interaction code
6. Switching to nearest-neighbor texture filtering

With nearest-neighbor filtering, the volume still jiggles. This eliminates trilinear filtering precision as the root cause.

**Remaining hypotheses:**

1. **Fragment shader execution ordering:** Metal might execute fragments in a non-deterministic order across frames. For the volume shader, different execution orders produce different accumulated colors due to floating-point non-associativity in the compositing loop. This would explain why flat-color (no compositing loop) doesn't jiggler but volume (heavy compositing) does. This is the strongest remaining theory.

2. **Dawn/WebGPU command buffer synchronization:** Despite triple-buffering, there may be a Dawn-specific issue where `queue.WriteBuffer()` doesn't fully synchronize with Metal command buffer execution.

3. **GPU thread scheduling differences between frames:** Metal's thread scheduler may assign different GPU cores or warps to fragments across frames, leading to different floating-point rounding in the compositing accumulation.

---

## Current State of Code

### Files Modified

| File | Changes |
|---|---|
| `vtkWebGPUGPUVolumeRayCastMapper.h` | Triple-buffered uniform buffers (`UniformBuffers[3]`, `BindGroups[3]`, `CurrentBufferIndex`) |
| `vtkWebGPUGPUVolumeRayCastMapper.cxx` | Triple-buffer logic, uniform writes in SyncDeviceResources, diagnostic logging |
| `vtkWebGPURenderer.h` | Triple-buffered SceneTransformBuffer (`SceneTransformBuffers[3]`, `SceneBindGroups[3]`, `CurrentSceneBufferIndex`), `SetupSceneBindGroups()` |
| `vtkWebGPURenderer.cxx` | Triple-buffer logic, buffer index advancement after RecordRenderCommands |
| `vtkWebGPUCamera.h` | Added `InvertedViewMatrix[4][4]` to `SceneTransforms` struct |
| `vtkWebGPUCamera.cxx` | Computes and stores inverted view matrix in CacheSceneTransforms |
| `vtkVolumeRayCastMapperShader.wgsl` | Screen-space ray derivation using `@builtin(position)` + inverted projection/view matrices |
| `VTKBaseViewController.mm` | Pan/rotation gesture logging (touch deltas, camera position) |

### Key Code Patterns

**Triple-buffer cycling (renderer):**
```cpp
// Write to buffer[current], advance after recording
this->WriteSceneTransformsBuffer();  // writes to SceneTransformBuffers[CurrentSceneBufferIndex]
this->CurrentSceneBufferIndex = (this->CurrentSceneBufferIndex + 1) % NUM_SCENE_BUFFERS;
```

**Screen-space ray derivation (shader):**
```wgsl
let ndcX = (input.position.x - viewport.x) / viewport.z * 2.0 - 1.0;
let ndcY = (input.position.y - viewport.y) / viewport.w * 2.0 - 1.0;
let clipPos = vec4<f32>(ndcX, ndcY, input.position.z, 1.0);
let viewPosH = sceneTransform.inverted_projection * clipPos;
let viewPos = viewPosH.xyz / viewPosH.w;
let worldPosH = sceneTransform.inverted_view * vec4<f32>(viewPos, 1.0);
let worldPos = worldPosH.xyz / worldPosH.w;
let modelPosH = volumeUniforms.worldToVolume * vec4<f32>(worldPos, 1.0);
let modelPos = modelPosH.xyz / modelPosH.w;
let fragVolumePos = (modelPos - volumeUniforms.volumeBoundsMin.xyz) / boundsSize;
```

---

## Recommendations for Further Investigation

1. **Capture per-pixel frame difference:** Render to an offscreen texture, read back consecutive frames, and compute the per-pixel difference. This would quantify the actual visual instability and confirm whether it's a compositing accumulation issue.

2. **Force deterministic fragment execution:** Add `uniform` qualifiers or use `storage` buffers for intermediate results to force sequential fragment processing. If jiggling disappears, non-deterministic execution order is confirmed.

3. **File a Dawn/Metal backend bug report:** With 15 attempts failing at the VTK level, the root cause likely lies in Dawn's Metal translation layer or in Metal's GPU execution model. A minimal reproducible case (simple volume, known shader) would be needed.

4. **Compare with native Metal volume renderer:** Write a minimal Metal volume renderer with the same algorithm (same shader logic, same buffer layout) to determine if the issue is Dawn-specific or Metal-specific.

5. **Profile with Metal System Trace:** Use Xcode's Metal System Trace to capture GPU command buffer timing, thread scheduling, and identify any synchronization gaps or non-deterministic execution patterns.
