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
