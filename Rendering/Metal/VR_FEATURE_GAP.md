# Volume Rendering Feature Gap: Metal vs OpenGL

Status as of 2026-07-13. Compares `vtkMetalGPUVolumeRayCastMapper` against
`vtkOpenGLGPUVolumeRayCastMapper` (the reference OpenGL implementation).

## Performance Optimizations

| Optimization | OpenGL | Metal | Impact |
|---|:---:|:---:|---|
| **Early ray termination** (opacity threshold break) | Yes | Yes | Both terminate at ~95% accumulated opacity |
| **Adaptive sample distance** (`AutoAdjustSampleDistances`) | Yes | **Yes** | Dynamically reduces step count to meet frame-time target |
| **Image-space downsampling** (`ImageSampleDistance`) | Yes | **Yes** | Renders to lower-res FBO then upscales; cuts fragment count by up to 4x |
| **Lock sample distance to input spacing** | Yes | **Yes** | Adapts step size to voxel density for optimal quality/perf |
| **Depth buffer occlusion** (opaque geometry early-terminates rays) | Yes | **Yes** | Captures Z-buffer; ray stops at nearest opaque surface |
| **Two-pass contour + volume** (`UseDepthPass`) | Yes | No | Renders isosurface contours to depth FBO, then ray-marches behind them |
| **Volume partitioning** (`SetPartitions`) | Yes | **Yes** | Splits large volumes into blocks for 3D texture size limits |
| **Near-plane bounding box clipping** | Yes | **Yes** | Clips box geometry when camera is inside volume; fewer wasted fragments |
| **Gradient-based Phong shading** | Yes | **Yes** | Central-difference normals for lighting; visual quality, not perf |
| **Gradient opacity** (1D gradient opacity) | Yes | **Yes** | Uses gradient magnitude for edge/feature highlighting |
| **Cropping regions** (32-region mask) | Yes | No | Interactive ROI without data copy |
| **Clipping planes** (up to 8 arbitrary) | Yes | No | Cuts volume with arbitrary planes |
| **Multi-volume compositing** | Yes | No | Simultaneous rendering of multiple volumes |
| **Mask / label map** | Yes | No | Binary mask and label map with 2D TFs |
| **Double-stepped ILP loop** (2 samples/iter) | No | **Yes** | Exploits Apple GPU half-precision ALU at 2x throughput |

## Summary

The Metal mapper now has nine features matching or exceeding the OpenGL path:
1. **Double-stepped sampling** exploiting Apple Silicon's half-precision ALU
2. **Adaptive sample distance** — dynamically adjusts step count frame-to-frame
3. **Image-space downsampling** — renders at reduced resolution during interaction
4. **Lock sample distance to input spacing** — adapts step size to voxel density for optimal quality/perf
5. **Depth buffer occlusion** — samples scene depth to terminate rays at opaque surfaces
6. **Volume partitioning** — splits large volumes into blocks for 3D texture size limits
7. **Near-plane clipping** — clips bounding box geometry against near plane when camera is inside
8. **Gradient-based Phong shading** — central-difference normals for headlight Phong lighting
9. **Gradient opacity** — 1D gradient opacity transfer function for edge/feature highlighting

However, the OpenGL mapper retains several high-impact adaptive features that
the Metal path is missing entirely:

### Critical gaps (largest performance impact)

None remaining.

### Medium gaps (quality of life)

1. ~~**Gradient-based shading**~~ **IMPLEMENTED**
   - Central-difference normals computed in the fragment shader
   - Headlight Phong lighting model (ambient + diffuse + specular)
   - 1D gradient opacity transfer function support
   - Enabled via `vtkVolumeProperty::SetShade(1)` and `SetGradientOpacity()`

2. **Cropping regions** — Interactive 32-region crop without re-uploading data.

### Low priority gaps (specialized use cases)

4. Multi-volume compositing
5. Mask / label map support
6. Clipping planes
7. ~~Volume partitioning (only needed for textures exceeding hardware 3D limit)~~ **IMPLEMENTED**

## Recommended Implementation Order

For maximum performance improvement with minimum effort:

1. ~~**Image-space downsampling** (Low effort, High impact)~~ **IMPLEMENTED**
   - Render to half-res offscreen texture, blit to screen
   - Triggered during interaction, full-res on idle

2. ~~**Depth buffer occlusion** (Medium effort, Medium impact)~~ **IMPLEMENTED**
   - Sample scene depth texture during ray march
   - Unproject to volume-local space; terminate ray early at opaque surfaces
   - Handles MSAA via blit resolve before volume pass

3. ~~**Near-plane clipping** (Medium effort, Medium impact)~~ **IMPLEMENTED**
   - Clip bounding box geometry against near plane when camera is inside
   - Reference: `vtkOpenGLGPUVolumeRayCastMapper::RenderVolumeGeometry()`

4. ~~**Gradient-based shading** (High effort, Medium impact)~~ **IMPLEMENTED**
   - Central-difference gradient computation in fragment shader (7 texture samples per sample)
   - Headlight Phong lighting model (ambient + diffuse + specular)
   - 1D gradient opacity transfer function via 256x1 lookup texture
   - Per-block gradient step computation for partitioned volumes
