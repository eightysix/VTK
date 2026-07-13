# Volume Rendering Feature Gap: Metal vs OpenGL

Status as of 2026-07-13. Compares `vtkMetalGPUVolumeRayCastMapper` against
`vtkOpenGLGPUVolumeRayCastMapper` (the reference OpenGL implementation).

## Performance Optimizations

| Optimization | OpenGL | Metal | Impact |
|---|:---:|:---:|---|
| **Early ray termination** (opacity threshold break) | Yes | Yes | Both terminate at ~95% accumulated opacity |
| **Adaptive sample distance** (`AutoAdjustSampleDistances`) | Yes | **Yes** | Dynamically reduces step count to meet frame-time target |
| **Image-space downsampling** (`ImageSampleDistance`) | Yes | No | Renders to lower-res FBO then upscales; cuts fragment count by up to 4x |
| **Lock sample distance to input spacing** | Yes | No | Adapts step size to voxel density for optimal quality/perf |
| **Depth buffer occlusion** (opaque geometry early-terminates rays) | Yes | No | Captures Z-buffer; ray stops at nearest opaque surface |
| **Two-pass contour + volume** (`UseDepthPass`) | Yes | No | Renders isosurface contours to depth FBO, then ray-marches behind them |
| **Volume partitioning** (`SetPartitions`) | Yes | No | Splits large volumes into blocks for 3D texture size limits |
| **Near-plane bounding box clipping** | Yes | No | Clips box geometry when camera is inside volume; fewer wasted fragments |
| **Gradient-based Phong shading** | Yes | No | Central-difference normals for lighting; visual quality, not perf |
| **2D transfer functions** (gradient opacity) | Yes | No | Uses gradient magnitude for edge/feature highlighting |
| **Cropping regions** (32-region mask) | Yes | No | Interactive ROI without data copy |
| **Clipping planes** (up to 8 arbitrary) | Yes | No | Cuts volume with arbitrary planes |
| **Multi-volume compositing** | Yes | No | Simultaneous rendering of multiple volumes |
| **Mask / label map** | Yes | No | Binary mask and label map with 2D TFs |
| **Double-stepped ILP loop** (2 samples/iter) | No | **Yes** | Exploits Apple GPU half-precision ALU at 2x throughput |

## Summary

The Metal mapper now has two unique performance features the OpenGL path lacks:
1. **Double-stepped sampling** exploiting Apple Silicon's half-precision ALU
2. **Adaptive sample distance** — dynamically adjusts step count frame-to-frame

However, the OpenGL mapper retains several high-impact adaptive features that
the Metal path is missing entirely:

### Critical gaps (largest performance impact)

1. **Image-space downsampling** — Renders at reduced resolution (e.g. 0.5x)
   during interaction, then full resolution when idle. Cuts fragment count by
   4x at 0.5x scale.

2. **Depth buffer occlusion** — When opaque geometry (e.g.骨骼 surface mesh)
   is rendered before the volume, the OpenGL path captures the depth texture
   and uses it to terminate rays early. Essential for mixed rendering scenes.

### Medium gaps (quality of life)

4. **Near-plane clipping** — When the camera is inside the bounding box, the
   OpenGL path clips the box against the near plane. The Metal path renders
   the full box, wasting fragments behind the camera.

5. **Gradient-based shading** — Central-difference normals computed in the
   shader enable Phong lighting and gradient-opacity transfer functions. The
   Metal path has flat color-only compositing.

6. **Cropping regions** — Interactive 32-region crop without re-uploading data.

### Low priority gaps (specialized use cases)

7. Multi-volume compositing
8. Mask / label map support
9. Clipping planes
10. Volume partitioning (only needed for textures exceeding hardware 3D limit)

## Recommended Implementation Order

For maximum performance improvement with minimum effort:

1. **Image-space downsampling** (Low effort, High impact)
   - Render to half-res offscreen texture, blit to screen
   - Triggered during interaction, full-res on idle

2. **Near-plane clipping** (Medium effort, Medium impact)
   - Clip bounding box geometry against near plane when camera is inside
   - Reference: `vtkOpenGLGPUVolumeRayCastMapper::RenderVolumeGeometry()`

3. **Depth buffer occlusion** (Medium effort, Medium impact)
   - Capture depth texture from earlier in render pass
   - Pass to volume shader; terminate ray when `tCurrent >= depthSample`

4. **Gradient-based shading** (High effort, Medium impact)
   - Compute central-difference gradients in fragment shader
   - Add Phong lighting model; enable gradient-opacity TFs
