# Metal Volume Ray-Cast Mapper & Shader — General Guide

*Onboarding guide for `vtkMetalGPUVolumeRayCastMapper` + `MetalShaders.metal`. For NIFTI/DICOM perf deep-dives see `PerformanceInvestigation/perf_investigation_part2.md` and `PERFORMANCE_INVESTIGATION.md`.*

## 1. Overview

`vtkMetalGPUVolumeRayCastMapper` is the Metal override for `vtkVolumeMapper` (`Rendering/Volume` → `RenderingMetal`). It implements GPU ray-casting by proxy geometry (box) or fullscreen quad, marching each fragment through the volume texture and compositing via the transfer function. The Metal path mirrors the OpenGL path (`vtkOpenGLGPUVolumeRayCastMapper` + `vtkVolumeShaderComposer.h`) for parity.

```
CPU: vtkMetalGPUVolumeRayCastMapper.mm ──► PerBlockData + VolumeMapperUniforms ─┐
                                          volumeTexture, TF, minmax, ...       │
GPU: MetalShaders.metal  fragment_volume_main ──► marchVolume() ──► marchVolumeUnified() ──► half4 color
       + compute kernels (minmax, segment, transpose, bin, march)
```

## 2. Mapper (`vtkMetalGPUVolumeRayCastMapper.mm:1`)

### 2.1 Class & factory

* Header `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.h:1` subclasses `vtkGPUVolumeRayCastMapper`, registered as `RenderingBackend=Metal` override (`vtkMetalPolyDataMapper.mm` style). Selected via `vtkObjectFactory::SetPreferences()` or `--vtk-factory-prefer` in `vtkMetalGLVisualComparison` `Rendering/Metal/Testing/Cxx/TestMetalVolumeRayCast.cxx:1`.

### 2.2 Per-frame setup

| Step | File:line | What |
|------|-----------|------|
| Build DB | `vtkMetalGPUVolumeRayCastMapper.mm:3100` `BuildPerBlockData()` | `PerBlockData` `Shaders/MetalShaders.metal:3089` `volumeBoundsMin/Max`, `textureBoundsMin/Max`, `gradientStep`, `minMaxInfo`, `slabInfo` |
| Uniforms | `vtkMetalGPUVolumeRayCastMapper.mm:4000` `UpdateVolumeUniforms()` `Shaders/MetalShaders.metal:2833` `VolumeMapperUniforms` | `sampleDistance` `2841`, `scalarMin/Max` `2843`, `gradientStep` `2849`, `useJittering` `2847`, `croppingPlanes` `2859`, `mask`/`blanking`/`rectilinear`, `maxStepsFrame` `3021`, `maxBatchWidth` `3028` |
| Lights | `vtkMetalGPUVolumeRayCastMapper.mm:3500` `UpdateLightUniforms()` `Shaders/MetalShaders.metal:3070` `VolumeLightUniforms` | `MAX_LIGHTS 8`, `twoSidedLighting` |
| Textures | `vtkMetalGPUVolumeRayCastMapper.mm:6000` `CreateGlobalVolumeTexture()` `Shaders/MetalShaders.metal:3602` `sampleVolumeScalar` | `volumeTexture` `texture3d<float>`, `transferFunctionTexture` `texture2d<float>`, `gradientOpacityTexture`, `minMaxTexture` etc. Transpose `Shaders/MetalShaders.metal:2744` `fc_volTransposed`, RG8 `2742` |
| PSO | `vtkMetalGPUVolumeRayCastMapper.mm:8100` `BuildVolumePipelines()` `Shaders/MetalShaders.metal:2725` `fc_marchVariant` etc. | `MTLFunctionConstantValues` specialization (see §4) |

### 2.3 Draw

* Proxy geometry `vtkMetalGPUVolumeRayCastMapper.mm:7400` `Render()` builds box vertices, `fragment_volume_main` `Shaders/MetalShaders.metal:7384` rasterizes, or `fragment_volume_fullscreen_main` `Shaders/MetalShaders.metal:7584` for camera-inside (`useCameraInsideNearClip` `Shaders/MetalShaders.metal:2998`).
* Grid traversal `Shaders/MetalShaders.metal:7948` `fragment_volume_grid_traversal_main` for partitioned data (`GridTraversalUniforms` `Shaders/MetalShaders.metal:7835`).
* RTT `Shaders/MetalShaders.metal:7730` `fragment_volume_rtt_main` exports depth (`color1`).
* Selection `Shaders/MetalShaders.metal:7505` `fragment_volume_selection_main`.

## 3. Shader (`Rendering/Metal/Shaders/MetalShaders.metal:2690`)

### 3.1 Feature specialization (PSO)

All volume flags are `function_constant` (`[[function_constant(N)]]`) baked at pipeline creation via `VolumeShaderFeatureFlags` `vtkMetalGPUVolumeRayCastMapper.mm:700`. Dead branches are eliminated; lean `576` threads/TG vs heavy `32-wide` `37%` occupancy `PERFORMANCE_INVESTIGATION.md:39`.

| Constant | File:line | Env var |
|----------|-----------|---------|
| `fc_shading` `fc_gradientOpacity` | `Shaders/MetalShaders.metal:2666` | `VTK_METAL_TEST_SHADE` |
| `fc_minmax` `fc_mmBlocks` `fc_mmSuper` | `Shaders/MetalShaders.metal:2768` `35` `36` | `VTK_METAL_TEST_MINMAX` `MM_BLOCKS` |
| `fc_marchVariant` | `Shaders/MetalShaders.metal:2725` | `VTK_METAL_TEST_MARCH_VARIANT` (0 baseline, 6 8x, 7 4x, 8 harness, 9 adaptive 48) |
| `fc_fragBatch` `fc_cmBatch` | `Shaders/MetalShaders.metal:2810` `2796` | `VTK_METAL_TEST_FRAG_BATCH` `CM_BATCH` |
| `fc_cropping` `fc_mask` `fc_blanking` `fc_rectilinear` `fc_transfer2D` | `Shaders/MetalShaders.metal:2716` `2718` `2695` `2691` | `SetCropping`, mask, blanking |
| `fc_linearInterpolation` | `Shaders/MetalShaders.metal:2695` | `property->GetInterpolationType()` |
| `fc_slabMode` | `Shaders/MetalShaders.metal:2734` | `VTK_METAL_TEST_NUM_SLABS` |
| `fc_volTransposed` `fc_volRg8` | `Shaders/MetalShaders.metal:2744` `2742` | `VTK_METAL_TEST_VOLTRANSPOSE_AXIS` `RG8` |

`[fragpso] type= mask= extra= fragBatch=` log `vtkMetalGPUVolumeRayCastMapper.mm:8350` when `VTK_METAL_TEST_MARCH_DEBUG=1`.

### 3.2 Ray setup

* `computeVolumeBounds` `Shaders/MetalShaders.metal:4117` → `blockMinGlobal` etc.
* `setupVolumeRay` `Shaders/MetalShaders.metal:4132` → `RaySetup` `Shaders/MetalShaders.metal:4107` `entryPoint/exitPoint/totalBoxT/tTerminateMax`. Handles near-clip `Shaders/MetalShaders.metal:4157`, clipping planes `4172`, depth `4198`.
* `MarchParams` `Shaders/MetalShaders.metal:4214` `rayOrigin/Dir/tStart/tEnd/stepSize/jitter/tTerminateMax/checkBounds`.
* `marchVolume` wrapper `Shaders/MetalShaders.metal:7253` computes `jitter` `Shaders/MetalShaders.metal:3453` `sampleJitterNoise`/`sampleIGNJitter`, `physicalSampleStep` `Shaders/MetalShaders.metal:3520`, `tStart` `Shaders/MetalShaders.metal:7308`, calls `marchVolumeUnified`.

### 3.3 March (`marchVolumeUnified` `Shaders/MetalShaders.metal:4229`)

**Setup** `Shaders/MetalShaders.metal:4283` `rayDirTexLocal`, `texStep`, `texelCount` `4292` (`fc_volRg8`/`fc_volTransposedY` `4294`), `ctpScale/Offset` `4301`, `evalStep` `4303`, `viewDirHalf` `4314`. `firstT` `Shaders/MetalShaders.metal:4349` `p.checkBounds?jitter:jitter+ceil((tStart-jitter)/stepSize)*stepSize`, `maxSteps` `Shaders/MetalShaders.metal:4363` `max(1,ceil((tEnd-firstT)/stepSize))` (fix `b2e0286446` `5634`), `mainSteps` `4365` for variant 4/5. Slab tiling `Shaders/MetalShaders.metal:4413` partitions `[0,maxSteps)`.

**State** `Shaders/MetalShaders.metal:4550` `accumulatedColor/Opacity` `kExitAcc` `4554`, `MIP/MinIP` `4577`, `mmVisits` `4592`, `useMinMax` `Shaders/MetalShaders.metal:4643`.

**Loops** (mutually exclusive, compile-time):

* `fc_doExit` `Shaders/MetalShaders.metal:4709` back-edge latch (divergent_tail).
* `fc_marchVariant==9` `Shaders/MetalShaders.metal:5221` adaptive `48/32/16/8/4/2/1` via `batchCap` `Shaders/MetalShaders.metal:5568` `max(1,int(maxBatchWidth))` vs `fc_fragBatch`. `MV9_FETCH` `Shaders/MetalShaders.metal:5255` `MV9_COMPOSITE` `Shaders/MetalShaders.metal:5259` (handles `fc_cropping` `fc_blanking` `fc_independentComponents` `fc_shading` etc.), `MV9_ADVANCE` `Shaders/MetalShaders.metal:5554`. Preamble walk `Shaders/MetalShaders.metal:5734` `mmDimF` `Shaders/MetalShaders.metal:572` or `segHop` `Shaders/MetalShaders.metal:5643`.
* `fc_marchVariant==8` `Shaders/MetalShaders.metal:6117` harness-style 8-wide single advance.
* `fc_marchVariant>=6` `Shaders/MetalShaders.metal:5216` 8x/4x unroll `Shaders/MetalShaders.metal:6238` `PROC_UNROLL_SAMPLE` `Shaders/MetalShaders.metal:6254`.
* Baseline `for(i<maxSteps)` `Shaders/MetalShaders.metal:6416` divergent `latchExit` `Shaders/MetalShaders.metal:6436`, CTP bounds `Shaders/MetalShaders.metal:6474`, minmax `Shaders/MetalShaders.metal:6498`, cropping/mask/blanking, composite `Shaders/MetalShaders.metal:6934` `useIndependentPath`.

**Sampling** `sampleVolumeScalar` `Shaders/MetalShaders.metal:3602` (`sVolume`/`sNearest`, `fc_volTransposed` swizzle `Shaders/MetalShaders.metal:2823`, RG8 `Shaders/MetalShaders.metal:3572`), `sampleTransferFunction` `Shaders/MetalShaders.metal:3661`, `computeGradientFast` `Shaders/MetalShaders.metal:3861` `6 fetches`, `computeVolumeLighting` `Shaders/MetalShaders.metal:3988`.

### 3.4 Compute kernels

* `volume_compute_minmax` `Shaders/MetalShaders.metal:9376` (`MinMaxComputeUniforms` `Shaders/MetalShaders.metal:8138`), `volume_dilate_minmax` `Shaders/MetalShaders.metal:9434`, `volume_reduce_minmax_blocks` `Shaders/MetalShaders.metal:9469`, `volume_reduce_minmax_superblocks` `Shaders/MetalShaders.metal:9554`.
* `volume_segment_build` `Shaders/MetalShaders.metal:8399` + `fragment_volume_ray_atlas` `Shaders/MetalShaders.metal:8172` / `synthesizeAtlasRay` `Shaders/MetalShaders.metal:8303` → `marchRayFromAtlasCore` `Shaders/MetalShaders.metal:8689` (compute march).
* `volume_transpose_xz` `Shaders/MetalShaders.metal:9590`.
* `volume_convert_*` `Shaders/MetalShaders.metal:9661`.

## 4. Uniforms & Textures

* `VolumeMapperUniforms` `Shaders/MetalShaders.metal:2833` — `worldToVolume` `volumeBoundsMin/Max` `cameraVolumePos` `viewProjection` `sampleDistance` `scalarMin/Max` `gradientStep` `croppingPlanes` `clippingPlanes` `maskScale/Bias` `useDepthTexture` `transfer2D` `averageIP` `blanking` `textureToVolume` `scalarMinComp` etc.
* `PerBlockData` `Shaders/MetalShaders.metal:3089` — per-brick bounds + `gradientStep` + `minMaxInfo` + `slabInfo`.
* Textures: `volumeTexture` `texture3d<float>` (`R8Unorm`/`R16` etc.), `transferFunctionTexture` `texture2d<float>` (`256x1`), `gradientOpacityTexture`, `minMaxTexture` `R8` occupancy, `minMaxBlockTexture` 3-state, `normalTexture`, `blankingTexture`, `maskTexture`, `brickOccupancy` `Shaders/MetalShaders.metal:7964`.
* Samplers: `sVolume` `linear clamp_to_edge` `Shaders/MetalShaders.metal:16`, `sNearest` `nearest`, `sVolumeClampZero` `Shaders/MetalShaders.metal:18`.

## 5. Performance knobs

* `VTK_METAL_TEST_MARCH_VARIANT=9` default adaptive (since `b2e0286446`). `0` baseline, `6` 8x, `8` harness, `9` 48-wide. See `PerformanceInvestigation/perf_investigation_part2.md:7` `M>GL`.
* `VTK_METAL_TEST_FRAG_BATCH=1..48` `Shaders/MetalShaders.metal:2810` `light` `56%` occupancy vs `maxBatchWidth` `3028` `heavy` `37%`. `NIFTI 41 steps` best `f2 0.86`, `DICOM 400 steps` best `f16 0.31` `perf_investigation_part2.md:7`.
* `VTK_METAL_TEST_MINMAX=1` `VTK_METAL_TEST_ACCEL=1` `VTK_METAL_TEST_MM_BLOCKS` etc. `mmWarpMin` `Shaders/MetalShaders.metal:5694` `mmLeapLevel` `Shaders/MetalShaders.metal:5789`.
* `VTK_METAL_TEST_SAMPLE_DISTANCE=0.5/4` `physicalSampleStep` `Shaders/MetalShaders.metal:3520`.
* `VTK_METAL_TEST_JITTER=1` `VTK_METAL_TEST_IGN_JITTER=0` `Shaders/MetalShaders.metal:3453`.

## 6. Parity & Testing

* Harness `vtkMetalGLVisualComparison` `Rendering/Metal/Testing/Cxx/TestMetalVolumeRayCast.cxx:1` renders same scene with `gl` vs `metal` offscreen, compares `thresholded error` (`--bench` gives `ms/f` `glFinish` vs `WaitForCompletion` `TestMetalGLVisualComparison.cxx:379`). Reference `perf_investigation_part2.md:1` `§40.3` `thr <5%`.
* `visual_compare` `400x400` etc. `Dump` via `VTK_METAL_TEST_DUMP_UNIFORMS` `PERFORMANCE_INVESTIGATION.md:5.2`.
* Probe `PerformanceInvestigation/minimal_repro.mm:1` standalone `512x512x1794` `R8` `45 M samples` isolates filter cost.

## 7. Known issues & recent fixes

* Far-edge sliver `b2e0286446` `Shaders/MetalShaders.metal:5634` `if(i>0 && currentT>=p.tEnd-1e-6)` — grazing `firstT>tEnd` `maxSteps=1` now composites clamped boundary; before `0` samples `2024` pixels `512²`.
* `M>GL` slowdown `NIFTI SD0.5 f≥8` `§7` — short dense chord, wide fetch array `n*7` + `pow` + `I$` spill; fix `batchCap=min(fragBatch,maxSteps/4)` `Shaders/MetalShaders.metal:5600`.

See `CINEMATIC_METAL_PLAN.md` for cinematic, `VolumeRayCastPerfRegression.md` for history.
