# VTK::RenderingMetal

## Description

This module contains the Metal backend for `RenderingCore` and
`RenderingVolume`. It registers its classes as factory overrides carrying a
`RenderingBackend=Metal` attribute so they can be selected at runtime with
`--vtk-factory-prefer RenderingBackend=Metal` (or the `VTK_FACTORY_PREFER`
environment variable) from an application that invokes
`vtkObjectFactory::InitializePreferencesFromCommandLineArgs(argc, argv)`.

> Note: the generic `vtk*CxxTests` drivers do **not** parse `--vtk-factory-prefer`,
> so a test built through those drivers always instantiates the registered
> default override. When `RenderingMetal` is linked, that override is the
> Metal implementation, regardless of the command-line flag. Use
> `vtkObjectFactory::SetPreferences()` in-process (as `vtkMetalGLVisualComparison`
> does) when a test must select a specific backend.

## Projected tetrahedra volume rendering

`vtkProjectedTetrahedraMapper` is overridden by `vtkMetalProjectedTetrahedraMapper`.
It follows the OpenGL backend's Shirley & Tuchman / Wylie 2002 algorithm
(segment-intersection projection, per-vertex attenuation and corrected depth).
Two implementation notes:

- All visibility-sort chunks are accumulated into a single set of vertex buffers
  and issued as one indexed draw. Reusing buffers per chunk is unsafe on Metal:
  `MTLResourceStorageModeShared` contents are read by the GPU at execution time,
  so later `memcpy`s would overwrite the data earlier draws still need.
- The CPU-side clip-space z (`[-1,1]`) is remapped to `[0,1]` when packing, so
  the written depth matches the scene's projection (nearz=0/farz=1).

All six `RenderingVolumeCxx-Metal` projected-tetrahedra tests pass.

## Volume rendering: nearest-vs-linear interpolation discrepancy

### Symptom

`TestSmartVolumeMapperWindowLevel` renders three MIP panels: left/center via the
GPU mapper (`vtkSmartVolumeMapper` in default mode) and right via the CPU
mapper (`SetRequestedRenderModeToRayCast`). The committed baseline PNG
(`Rendering/Volume/Testing/Data/Baseline/TestSmartVolumeMapperWindowLevel.png`)
shows **smooth** GPU-MIP panels. The Metal backend's GPU panels look visibly
pixellated/blocky next to the baseline; the CPU (right) panel matches.

### Root cause

The baseline was rendered with **linear** volume interpolation, while current
VTK renders volume MIPs with **nearest** interpolation:

- `vtkVolumeProperty`'s default interpolation has been
  `VTK_NEAREST_INTERPOLATION` since the 2012 tree modularization.
- Until 2019 the OpenGL volume texture kept its own default
  (`vtkTextureObject::Linear`, `vtkVolumeTexture::InterpolationType`),
  so GPU volume rendering was effectively linear.
- Upstream commit `ac76fe16364` (2019) made the GL mapper pass
  `property->GetInterpolationType()` into `vtkVolumeTexture::LoadVolume`,
  which sets the GL 3D texture's mag/min filter from the property. With the
  default property this makes GL sample the volume with nearest as well.
- The Metal mapper honors the property the same way:
  `fc_linearInterpolation` is driven by
  `property->GetInterpolationType() == VTK_LINEAR_INTERPOLATION`
  (`vtkMetalGPUVolumeRayCastMapper.mm`), and `sampleVolumeScalar`
  (`Rendering/Metal/Shaders/MetalShaders.metal`) picks `sNearest` otherwise.

A nearest-filtered MIP max is quantized to discrete voxel values, so adjacent
screen pixels share the same max voxel and form ~1-voxel-wide plateaus
(blockiness). A linear-filtered MIP max varies continuously.

### Measured facts

Blockiness metric = percentage of adjacent foreground pixel pairs whose luma
is exactly equal ("plateau fraction"), rows 1..h-1:

| panel        | baseline | Metal (nearest) | Metal (linear) |
|--------------|----------|-----------------|----------------|
| left  (GPU)  | 14.45%   | 65.11%          | 15.18%         |
| center (GPU) | 35.78%   | 69.27%          | 36.02%         |
| right (CPU)  | 59.33%   | 59.31%          | 59.31%         |

- Metal with nearest matches the OpenGL backend: 63.66% vs 64.26% plateau on
  `vtkMetalGLVisualComparison --scene VolumeRayCast`.
- Metal with **linear** volume sampling (temporarily forcing `sVolume` in
  `sampleVolumeScalar`) reproduces the committed baseline **byte-for-byte**
  (test reports `ImageError 0`; PNG byte-identical).
- The test passes with either interpolation because the legacy image
  comparison (threshold 20/255) swallows the ~4-5 luma-level per-pixel
  difference; the discrepancy is purely visual.

### Status

Accepted as-is: Metal is faithful to the current VTK semantics and to the
OpenGL reference backend. The `TestSmartVolumeMapperWindowLevel` baseline
predates GL honoring nearest interpolation (2019) and is therefore stale with
respect to both backends; the OpenGL test shows the same pre-existing
thresholded error (0.498).

### If pixel parity with the stale baseline is ever desired

- Restore linear volume sampling. The property default cannot reasonably change
  (≈31 GPU volume tests rely on it), so the change would be a Metal-local
  override of `sampleVolumeScalar`/`fc_linearInterpolation`. This would make
  Metal match the committed baseline but diverge from the GL backend for
  default (nearest) properties.
- Re-generating the baseline from a nearest render would instead keep Metal and
  GL aligned with each other.

## Volume rendering performance: composite slab tiling

`vtkMetalGPUVolumeRayCastMapper` splits each ray into **8 front-to-back slab
passes** by default. Each pass composites only a ray-length-fraction index
range `[ceil(j·maxSteps/K), ceil((j+1)·maxSteps/K))` starting from transparent;
the K partial composites are accumulated by the pipeline's existing
`(ONE, ONE_MINUS_SRC_ALPHA)` hardware blend. Because premultiplied front-to-back
`over` is associative, the combined result equals a single-pass composite up to
fp rounding. The per-slab working set is small enough to stay cache-resident on
Apple SoCs, which is decisive on the raw (minmax-off) coarse-sample-distance
path that previously lost to OpenGL by 1.3-1.8x:

- 400x400 minmax-on: 24.7 -> 15.1 ms (Metal/GL 0.30)
- 2048x2048 SD4 minmax-off: 87.7 -> 44.2 ms (Metal/GL 1.75x loss -> 0.87x win)

Controls and behavior:

- Slab tiling is disabled by default (single pass, bit-identical to the
  pre-slab build). `VTK_METAL_TEST_NUM_SLABS=N` (N >= 2) enables N slabs;
  `VTK_METAL_TEST_NUM_SLABS=0` re-enables the view-aligned adaptive choice
  (1 for near-axis views, 8 otherwise).
- Slabs apply only to the blended direct-render paths (proxy geometry and
  camera-inside fullscreen) in composite blend mode; offscreen render-to-image,
  grid-traversal, selection, and non-composite blend modes stay single-pass.
- Output stays within the existing GL-vs-Metal thresholded parity (0.000); the
  direct path's 8-bit drawable quantizes per slab, so Metal-vs-single-pass can
  differ by up to ~19/255 on a few percent of pixels.

See `PERFORMANCE_INVESTIGATION.md` ("App implementation: composite slab tiling")
for the full measured table and future performance/correctness leads (float
accumulation RT for byte-exactness, K scaling at high res, minmax+slab
stacking).
