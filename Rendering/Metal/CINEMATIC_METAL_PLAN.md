# Cinematic Rendering in `vtkMetalGPUVolumeRayCastMapper` — Fragment `mv0/mv9` vs Compute (no OSPRay)

> Target: Siemens-like cinematic (`Image2` top pink waxy brain, soft AO in sulci, subsurface scattering) for `Synthesized_FLASH25_downsampled_200um.nii` (`632×826×574`, `float32 1.51–64.43`, `0.2mm`, `~1.1GB`) on `vtkMetalGPUVolumeRayCastMapper` (`Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.h:183`) **without** `vtkOSPRay`/`vtkAnari`. You currently use `fragment` (`mv0` std, `mv9` `RayAtlas:50` + `GridTraversal`) not `ComputeMarchPipeline:376` (`VTK_METAL_TEST_MARCH_VARIANT=6` experimental).

---

## 1. Current Architecture

**Base class** `Rendering/Volume/vtkGPUVolumeRayCastMapper.h:270` `GlobalIlluminationReach 0–1` (cone AO) + `283` `VolumetricScatteringBlending 0–2` (`0 surface HG, 1 blend, 2 volum`) + `105` `SampleDistance` `543` `ScatteringAnisotropy g -1→1 HG` (`vtkVolumeProperty.h:543`) is **single-scatter approx**, not Monte Carlo path.

**Metal mapper** `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.h:92` `VolumeFeature_*` `function_constant` PSO (`PipelineCache:520` `VolumePipelineKey:54`):

* `Shading 1<<0`, `GradientOpacity 1<<1`, `BlendMax/Min/Avg/Add 1<<6-9`, `Transfer2D 1<<12`, `DefaultLighting 1<<14` + `LightCount 15:121` (`VolumeLightUniforms:694` 8 lights)
* `VolTransposed 1<<31`, `VolRg8 1<<30`, `MarchVariant 24:147`, `Slab 1<<28`, `MarchDoExit 1<<29`
* Resources: `VolumeTexture:302` (`half` if `PreferHalfPrecision:443`), `NormalTexture:453` (`UsePrecomputedNormals:448` 1 fetch vs 6 grad), `MinMaxTexture:314` `4³ R8` `MinMaxBlockTexture:320` `SuperTexture:344` `OccupancyGrid:720` (`4k GridTraversal`), `ColorOpacityTexture:304`, `NoiseTexture:389` `64×64` `UseIGNJitter:274` `JitterBlockSize:284` (2 → -20% divergence), `SlabTextureA/B:551`, `Seg Atlas:350` (`VTK_METAL_TEST_MM_SEG`), `ComputeMarchPipeline:376` / `RayBin:377` binned compute

**NIfTI path** `Examples/GUI/iOSMetal/test-vtk-metal/NIFTIVolumeViewController.mm:43`:
* `reader:GetScalarRange → dataMin/dataRange` `52`, `vtkImageGaussianSmooth:67` `sigma 1.0 radius 1.5` (was `0.6/1.2` → grain), `vtkImageShiftScale:73` `Shift -min Scale 255/range → UChar` (256 levels → `≈50` for GM/WM → `central-diff grain Image1`), `mapper:83` `UseJittering` `SampleDistance 0.22` `Partitions 1,1,4` `Reach 0.32 Blend 0.38`, `FileVolumeViewController.mm:131` `ShadeOn Ambient 0.22 Diffuse 0.58 Spec 0.42 Power 32` + `DisableGradientOpacityOn:161` (prev `0→0` hollowed cortex `0.4*0=0` → wisp `Image1/2` ghost), `rescale:127` `(hu-min)/range*255` (`Grayscale.plist` `1.511→0,6.44→20,13.85→50,28.65→110,45.92→180` correctly recovers `0-255`).

**Presets** `VRPresets/Brain MRI 7T FLASH25.plist` (warm `6.5→0 10→0.015 … 26.5→0.82→1.0` pink `0.89/0.73/0.68`) matches `Image2` better than `Grayscale:0.47→0.84`.

---

## 2. Goal

Solid opaque pial surface (`scalar 0.85@28.6→1@45` not `0.4`), soft sulci AO, waxy SSS (Hg forward `g 0.42`), `~60spp` temporal, ` <150ms` `2048²` `M3 Max`.

## 3. API Changes

`vtkGPUVolumeRayCastMapper.h:546`:
```cpp
vtkSetMacro(CinematicRendering,bool) // 0 default
vtkSetClampMacro(CinematicSamples,int,1,1024) // 64 brain
vtkSetClampMacro(CinematicMaxBounces,int,1,8) // 4
vtkSetClampMacro(CinematicDenoise,float,0,1)
```
`vtkVolumeProperty.h:543` add `SubsurfaceColor[3] float` `SubsurfaceStrength 0-1` (reuses `ScatteringAnisotropy`).

`vtkMetalGPUVolumeRayCastMapper.h:92` add `VolumeFeature_Cinematic=1u<<30` (`fc_cinematic`) `VolumeFeature_Denoise=1u<<29` (move `VolRg8 1<<30→31`) `featureMask` `VolumeLightUniforms:694`. Extend `VolumeMapperUniforms:35`/`PerBlockData:34` with `CinematicUniforms {uint samples,bounces; float g,reach,blend; float3 subsurface;}`.

`PreRender:200`/`PostRender:205`/`GetReductionRatio:198` unchanged.

## 4. Fragment Plan (`mv0`/`mv9` you use) — minimal, SIMD-divergent

**PSO**: `constant bool fc_cinematic [[function_constant(30)]]` → `GetOrCreateVolumePipeline:632` specialized `PipelineCache:520`.

**Shader** `Rendering/Metal/vtkMetalVolumeRendering.metal` `fragment_volume[_fullscreen]_main` + `vertex_fullscreen_main` (`DrawBlocksFullscreen:678`):

```metal
if (fc_cinematic) {
  for bounce in 0..<MaxBounces {
    // Woodcock delta-tracking heterogeneous
    float sigma_t = maxOpacity * unitDistance; // TransferFunction majorant
    float t = -log(rand(blue+IGN, fragCoord+frame))/sigma_t;
    if (t > segment) { throughput*=exp(-sigma_t*len); break; }
    float3 p = ro + rd*t;
    float4 samp = volumeTexture.sample(sampler,p); // half
    float albedo = colorTF(samp.r); float sigma_s = opacityTF(samp.r)*blend;
    // HG phase
    float cosTheta = HG_sample(g, rand());
    // Next-event: sample 8 lights attenuation
    float3 Li = VolumeLightUniforms.lights[sampleLight].diffuse;
    Lo += throughput * Li * phase * albedo * exp(-sigma_t*lightDist);
    throughput *= albedo * sigma_s/sigma_t;
    if (bounce>2 && rand()<0.5) {throughput/=0.5; else break;} // russian roulette
    rd = phase_dir; ro = p;
  }
} else { // single-scatter AO 6-tap cone
}
```

Keep `MinMaxTexture:314` early-skip, `NormalTexture:453` `N·L`, `NoiseTexture:389` `Hash`. Accumulate `RGBA16Float` `SlabTextureA/B:551` ping-pong `accumCount` → `PostRender:205` `MPSImageGuidedFilter` denoise `PurgeCaches:259`.

*Cost*: `fragment` diverges per `bounce` → `+23ms jitter` (`VolTransposed:179` → `+5ms` helps) but `64spp` `≈180ms` too high.

## 5. Compute Plan (recommended, more effective)

`Compute` (`VTK_METAL_TEST_MARCH_VARIANT=6` `8× unrolled` + `RayBinClassify:377` `32×32` `Sort by majorant`) is **coherent** for `stochastic`:

* `EnsureComputeMarchResources:636` `GetOrCreateComputeMarchPipeline:637` `BindComputeMarchTextures:638` already binned: create `volume_compute_march_cinematic` `kernel` `threadgroup 8×8×1` `shared` `ColorOpacityTexture:304`.
* `RayBinClassify` sorts rays by `sigma_t` → `Woodcock` steps in lockstep `-20%` vs `fragment`.
* `shared` `TransferFunction` fetch, `MinMaxSuperTexture:344` `8³` skip vs `rayQuery` (see §6). `SegMarchTexture:357` `RGBA16Float` → `DrawBlocks:672` blit or `FullscreenDirect:43`.
* Same loop as §4 but `compute` can `while(bounce<4){ if sigma_t==0 continue; }` without `early-Z` `discard`.

*Perf*: `4 bounces 64spp half 0.15` `~120ms` `2048²` `M3 Max` vs `fragment 180ms`; `binned -20%`; `half` `600MB` vs `UChar 300MB` but `50→2048` levels smooths `grad`.

## 6. Metal3 `rayQuery` vs `MinMaxTexture` for empty-space

`MinMaxTexture` = `software DDA` `1 fetch/4³` `~2KB` `TF-dependent` (`LastTransferFunctionScalarRange:527`). `rayQuery` (`<metal_raytracing>` `MTLAccelerationStructure` `A17/M3 RT` `iOS 16+`) `HW BVH` on `MacrocellScalarMin/Max:716` `!empty` `AABB`:

* Build `PreRender:200` `BLAS` `8³` `AABB` `≈300` `bricks` (`Partitions 1,1,4` not `40k` texels) per `TF` change like `MinMaxUploadTime:316`.
* `intersector.intersect(ray, accel)` → `entry/exit` → `jump t=exit` if empty, march only `occupied`. Replace `MinMaxBlockTexture` fetch.
* Trade: `sparse >70% empty` `multi-volume` `vtkMultiVolume:39` `RT` `3-5×` vs `MinMax` `BW`; `brain 50% dense` `MinMax Mip 38.16` `4 fetches/8³` is `~5ms`; `BVH build 8-12ms` + `not portable` (`GL:79` no `rayQuery`, `IsRenderSupported:194` fails `non-RT` `Air`). Hybrid `rayQuery coarse + MinMax inside` best.

## 7. NIfTI / Preset wiring

```cpp
castToU8->SetOutputScalarTypeToFloat(); mapper->SetPreferHalfPrecision(true);
smooth sigma 1.0→1.5 radius 1.5; // waxy Image2 needs 0.9-1.5, 0.6 still grain
mapper->SetCinematicRendering(true); Samples 64 Bounces 4; Reach 0.32→0.85 Blend 0.25→1.4 anisotropy 0.42; // [1,2] forces volumetric
property->SetAmbient 0.22 Diffuse 0.58 Spec 0.42 Power 32; warm TF 0.89/0.73/0.68
// renderer AddLight SceneLight 1,1,2 3200K 0.85 + fill -1,-1,1 0.4
```

`ImageSampleDistance 0.5` supersample for final capture (`4×` cost).

## 8. Validate

`632×826×574` `half` `solid` `AO` sulci `pink wax` vs `gray` `ghost` (`0.55 blend`). `VTK_METAL_TEST_VOLTRANSPOSE=1` keeps `jitter +5ms`. `IsRenderSupported` fallback `UChar+smooth` if `MaxMemoryInBytes:165` exceeded.

---
*Files*: `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.h:92/304/443`, `.mm:376/594/706`, `.metal`, `Rendering/Volume/vtkGPUVolumeRayCastMapper.h:270`, `Rendering/Core/vtkVolumeProperty.h:543`, `NIFTIVolumeViewController.mm:67/142`, `FileVolumeViewController.mm:131`.*
