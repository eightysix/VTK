// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkMetalGPUVolumeRayCastMapper
 * @brief   Metal implementation of volume rendering through ray-casting.
 */

#ifndef vtkMetalGPUVolumeRayCastMapper_h
#define vtkMetalGPUVolumeRayCastMapper_h

#include "vtkGPUVolumeRayCastMapper.h"
#include "vtkOverrideAttribute.h"
#include "vtkRenderingMetalModule.h" // For export macro
#include "vtkSmartPointer.h"         // For vtkSmartPointer
#include "vtkTimeStamp.h"            // For time stamp
#include "vtkWrappingHints.h"        // For VTK_MARSHALAUTO

#include <array>        // For std::array
#include <vector>       // For std::vector
#include <string>       // For std::string
#include <atomic>       // For std::atomic (MinMaxEmptyBlockFraction)
#include <unordered_map> // For pipeline cache
#include <functional>    // For std::hash

class vtkDataArray;
class vtkDataSet;
class vtkImageData;
class vtkPiecewiseFunction;
class vtkRenderWindow;
class vtkUnsignedCharArray;
class vtkVolume;

// Forward declarations for types defined in the .mm file.
struct PerBlockData;
struct VolumeMapperUniforms;

// Pipeline cache types for Phase 1B
enum class VolumePipelineType : uint32_t
{
  DirectScreen = 0,
  OffscreenLayer = 1,
  ImageSampleBlit = 2,
  FullscreenDirect = 3,      // Fullscreen ray-cast for camera-inside (BGRA8Unorm + depth)
  FullscreenOffscreen = 4,   // Fullscreen ray-cast for camera-inside (RGBA16Float, no depth)
  GridTraversalDirect = 5,   // Single-pass grid traversal fullscreen (BGRA8Unorm + depth)
  GridTraversalOffscreen = 6, // Single-pass grid traversal fullscreen (RGBA16Float, no depth)
  RenderToImage = 7,          // RenderToImage (RGBA16Float color + R32Float depth export)
  SelectionDirect = 8,        // Hardware-selection ray-cast (BGRA8Unorm + depth + RGBA32Uint ids)
  SelectionFullscreen = 9,    // Hardware-selection fullscreen ray-cast, camera inside (BGRA8Unorm + depth + RGBA32Uint ids)
  RayAtlas = 10               // §35.14 segment pre-pass: rasterizes per-pixel ray setup into 2x RGBA32Float

};

struct VolumePipelineKey
{
  uint32_t type;
  uint32_t colorFormat;
  uint32_t depthFormat;
  uint32_t sampleCount;
  uint32_t featureMask;
  // §29 transposed-orientation code (0=identity, 1=X-depth, 2=Y-depth); the
  // featureMask's VolumeFeature_VolTransposed bit only says "any transpose",
  // so the axis choice needs its own key component to get a specialized PSO.
  uint32_t featureMaskExtra;

  bool operator==(const VolumePipelineKey& other) const
  {
    return type == other.type && colorFormat == other.colorFormat &&
      depthFormat == other.depthFormat && sampleCount == other.sampleCount &&
      featureMask == other.featureMask &&
      featureMaskExtra == other.featureMaskExtra;
  }
};

struct VolumePipelineKeyHash
{
  size_t operator()(const VolumePipelineKey& k) const
  {
    size_t h = std::hash<uint32_t>()(k.type);
    h ^= std::hash<uint32_t>()(k.colorFormat) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<uint32_t>()(k.depthFormat) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<uint32_t>()(k.sampleCount) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<uint32_t>()(k.featureMask) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<uint32_t>()(k.featureMaskExtra) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

// Feature flags for volume shader specialization via function constants.
// Each flag enables a corresponding [[function_constant(n)]] in the Metal
// shader, allowing the compiler to eliminate dead code paths.
// Cinematic: VolumeFeature_Cinematic=1u<<30 (fc_cinematic 52) VolumeFeature_Denoise=1u<<29 (fc_denoise 53) — stored in featureMaskExtra bits 22/23 (all 32 featureMask bits used by VolRg8/Transposed).
// CinematicUniforms {uint samples,bounces; float g,reach,blend; float3 subsurface;} extends VolumeMapperUniforms:35/PerBlockData:34 (WAX brain g 0.42, Reach 0.85, Blend 1.4, 64spp, 4 bounces, denoise)
enum VolumeShaderFeatureFlags : uint32_t
{
  VolumeFeature_Shading        = 1u << 0,
  VolumeFeature_GradientOpacity = 1u << 1,
  VolumeFeature_Mask            = 1u << 2,
  VolumeFeature_MinMax          = 1u << 3,
  VolumeFeature_NormalTexture    = 1u << 4,
  VolumeFeature_LinearInterpolation = 1u << 5,
  // Blend modes (vtkVolumeMapper::BlendMode). Composite (0) uses no flag.
  VolumeFeature_BlendMaximumIntensity = 1u << 6,
  VolumeFeature_BlendMinimumIntensity = 1u << 7,
  VolumeFeature_BlendAverageIntensity = 1u << 8,
  VolumeFeature_BlendAdditive         = 1u << 9,
  VolumeFeature_ComputeNormalFromOpacity = 1u << 10,
  // Independent multi-component volume rendering (OpenGL independent-components
  // parity). Bakes the path into the pipeline via fc_independentComponents so
  // single-component pipelines compile it out entirely.
  VolumeFeature_IndependentComponents = 1u << 11,
  // 2D transfer-function (TF_2D) path. Baked via fc_transfer2D so non-TF_2D
  // pipelines compile the 2D lookup sampling out of the hot loop.
  VolumeFeature_Transfer2D = 1u << 12,
  // Rectilinear-grid path. Baked via fc_rectilinear so non-rectilinear
  // pipelines compile the coordinate-curve remapping out of the hot loop.
  VolumeFeature_Rectilinear = 1u << 13,
  // Default (single headlight) lighting. Baked via fc_defaultLighting so
  // headlight pipelines compile the multi-light loop out entirely.
  VolumeFeature_DefaultLighting = 1u << 14,
  // Active light count, encoded in 4 bits (values 0..8, MAX_LIGHTS). Baked via
  // fc_lightCount; only meaningful when VolumeFeature_DefaultLighting is clear.
  VolumeFeature_LightCountShift = 15,
  VolumeFeature_LightCountMask  = 0xFu << VolumeFeature_LightCountShift,
  // Dependent multi-component RGBA path. Baked via fc_dependentRGBA.
  VolumeFeature_DependentRGBA = 1u << 19,
  // Dependent multi-component LA path. Baked via fc_dependentLA.
  VolumeFeature_DependentLA = 1u << 20,
  // RenderToImage (depth-image export) path. Baked via fc_renderToTexture so
  // non-RTT pipelines compile the first-opaque-sample tracking out of the hot
  // loop.
  VolumeFeature_RenderToImage = 1u << 21,
  // Cropping regions. Baked via fc_cropping so non-cropping pipelines compile
  // the per-sample crop-region test out of the hot loop entirely.
  VolumeFeature_Cropping = 1u << 22,
  // Uniform-grid blanking (ghost arrays). Baked via fc_blanking so non-blanked
  // pipelines compile the seven per-sample blanking texture fetches out of the
  // hot loop entirely.
  VolumeFeature_Blanking = 1u << 23,
  // March-experiment selector (PERFORMANCE_INVESTIGATION.md section 9/10/14):
  // 4 bits encoded at bits 24-27, baked via fc_marchVariant so each experiment
  // gets its own specialized pipeline. Driven by VTK_METAL_TEST_MARCH_VARIANT;
  // 6 (8x unrolled march) is the default standard path. Experiments:
  //   1 = manual 8-tap trilinear (co-compiled volume samples)
  //   2 = clamp_to_zero volume sampler (in-shader clamp preserved)
  //   3 = predicated opacity exit, 4 = uniform frame-max loop, 5 = hybrid
  //   6 = 8x unrolled march (chunked independent fetches, latched exits)
  //   7 = 4x unrolled march
  VolumeFeature_MarchVariantShift = 24,
  VolumeFeature_MarchVariantMask  = 0xFu << VolumeFeature_MarchVariantShift,
  // Composite slab tiling (VTK_METAL_TEST_NUM_SLABS > 1). Baked via fc_slabMode
  // so non-slab pipelines compile the slab-index partition out of the hot loop.
  // Each slab pass composites only a ray-length-fraction index range from zero
  // and the mapper combines the partials with (ONE, ONE_MINUS_SRC_ALPHA)
  // blending; the associative front-to-back `over` makes the result equal to a
  // single-pass composite up to fp rounding (PERFORMANCE_INVESTIGATION /
  // minimal_gap phase-2). Only applies to the blended direct-render paths.
  VolumeFeature_Slab = 1u << 28,
  // V31 back-edge exit experiment (VTK_METAL_TEST_DOEXIT=1): baseline march
  // reshaped into a do-while with all exits folded into the back-edge
  // (divergent_tail_repro V31). Baked via fc_doExit so the reshaped loop gets
  // its own specialized pipeline; clear by default.
  VolumeFeature_MarchDoExit = 1u << 29,
  // RG8 pair-packed slice representation experiment (VTK_METAL_TEST_RG8=1):
  // R=slice 2z / G=slice 2z+1 over a halved-depth RG8 texture; the march's
  // trilinear z-blend is reconstructed in-shader from ~1.25 XY-bilinear taps
  // (divergent_tail V24/V32). Baked via fc_volRg8; clear by default.
  VolumeFeature_VolRg8 = 1u << 30,
  // Transposed volume representation (VTK_METAL_TEST_VOLTRANSPOSE, on by
  // default; =0 opts out): the volume uploads x<->z transposed (the slice axis
  // moves to the
  // texture's x extent) and every scalar fetch maps original-orientation
  // coordinates through .zyx. Root cause (2026-08-22): Metal's private 3D
  // tiling is strongly axis-biased — with the slice axis as texture depth,
  // trilinear z-pair fetches under per-pixel jitter phase scatter pay a huge
  // DRAM tax (jitter delta +23 ms vs GL +12 @2048 oblique); transposing
  // collapses it to +5 ms with byte-identical renders and halves j0.
  // Baked via fc_volTransposed; set by default. Mutually exclusive with
  // VolumeFeature_VolRg8 (pair indexing assumes the untransposed layout).
  VolumeFeature_VolTransposed = 1u << 31,
};

// Cinematic aliases for plan parity (real PSO bits are featureMaskExtra 22/23 -> fc_cinematic 52 / fc_denoise 53)
static constexpr uint32_t VolumeFeature_Cinematic_PlanAlias = 1u << 30; // fc_cinematic
static constexpr uint32_t VolumeFeature_Denoise_PlanAlias = 1u << 29; // fc_denoise
// Real compute-coherent bits (optimal variant: binned 8x8, shared TF, MinMaxSuper skip, temporal 64spp)
static constexpr uint32_t CinematicFeatureMaskExtra_Cinematic = 1u << 22;
static constexpr uint32_t CinematicFeatureMaskExtra_Denoise = 1u << 23;

VTK_ABI_NAMESPACE_BEGIN

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkMetalGPUVolumeRayCastMapper
  : public vtkGPUVolumeRayCastMapper
{
public:
  static vtkMetalGPUVolumeRayCastMapper* New();
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalGPUVolumeRayCastMapper, vtkGPUVolumeRayCastMapper);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  void GPURender(vtkRenderer* ren, vtkVolume* vol) override;

  int IsRenderSupported(vtkRenderWindow* window, vtkVolumeProperty* property) override;

  void ReleaseGraphicsResources(vtkWindow* window) override;

  void GetReductionRatio(double ratio[3]) override;

  void PreRender(vtkRenderer* ren, vtkVolume* vol, double datasetBounds[6],
    double scalarRange[2], int noOfComponents, unsigned int numberOfLevels) override;

  void RenderBlock(vtkRenderer* ren, vtkVolume* vol, unsigned int level) override;

  void PostRender(vtkRenderer* ren, int numberOfScalarComponents) override;

  // Image-space downsampling accessors (used by vtkMetalRenderer for blit)
  void* GetImageSampleColorTexture() const { return this->ImageSampleColorTexture; }
  int GetImageSampleWidth() const { return this->ImageSampleFBOWidth; }
  int GetImageSampleHeight() const { return this->ImageSampleFBOHeight; }

  // Depth buffer occlusion — set by vtkMetalRenderer before volume rendering
  void SetDepthTexture(void* depthTex) { this->DepthTextureOcclusion = depthTex; }

  /**
   * Set a fixed number of partitions in which to split the volume
   * during rendering. This will force by-block rendering without
   * trying to compute an optimum number of partitions.
   * Useful for volumes exceeding hardware 3D texture size limits.
   */
  void SetPartitions(unsigned short x, unsigned short y, unsigned short z);

  void SetUsePrecomputedNormals(bool val) { this->UsePrecomputedNormals = val; }
  bool GetUsePrecomputedNormals() const { return this->UsePrecomputedNormals; }

  // RenderToImage support (low-level color/depth texture export).
  void GetColorImage(vtkImageData*) override;
  void GetDepthImage(vtkImageData*) override;

  void SetPreferHalfPrecision(bool val) { this->PreferHalfPrecision = val; }
  bool GetPreferHalfPrecision() const { return this->PreferHalfPrecision; }

  void SetUseGPUMinMax(bool val) { this->UseGPUMinMax = val; }
  bool GetUseGPUMinMax() const { return this->UseGPUMinMax; }

  // Master switch for empty-space skipping via the min-max occupancy lattice.
  // When false, no occupancy lattice is ever built (GPU or CPU) and every
  // sample along the ray is fetched: a true unaccelerated march. This is the
  // apples-to-apples comparison for backends without min-max acceleration
  // (e.g. the OpenGL volume mapper), and it also isolates the cost of the
  // lattice machinery itself. When true (default), the lattice is built with
  // UseGPUMinMax selecting the GPU or CPU path.
  void SetUseMinMaxAcceleration(bool val) { this->UseMinMaxAcceleration = val; }
  bool GetUseMinMaxAcceleration() const { return this->UseMinMaxAcceleration; }

  // Drop the uploaded volume/min-max/gradient textures and derived grid
  // resources so the next render re-uploads them under the CURRENT layout and
  // feature settings (env-gated knobs like VTK_METAL_TEST_VOLTRANSPOSE are
  // re-read at upload time). The input dataset and all other mapper state are
  // untouched. Used by the test apps' runtime render-config toggles.
  void ForceResourceReupload();

  // §38.18.1: purge heavyweight Private-heap / PSO caches (SegPool 64 MB +
  // RayBin 4×W×H + ComputeMarchPipelineCache + SegBuild) that are otherwise
  // retained until ReleaseGraphicsResources/MTLDevice teardown. This is the
  // "reboot without reboot" — callable from ReleaseGraphicsResources and on
  // VTK_METAL_TEST_PURGE=1. qoff (window CB) default plus this purge removes
  // the reboot-only clog without cooldown (not DVFS).
  void PurgeCaches();

  // Phase 6: Fullscreen camera-inside path.
  // When true (default), camera-inside rendering uses a fullscreen ray-cast
  // fragment shader; setupVolumeRay clamps the entry to the near plane so the
  // eye->near-plane slab is skipped exactly like the OpenGL proxy mesh. Set
  // false to draw the CPU proxy geometry (ClipConvexPolyData + DensifyPolyData)
  // instead.
  void SetUseFullscreenCameraInside(bool val) { this->UseFullscreenCameraInside = val; }
  bool GetUseFullscreenCameraInside() const { return this->UseFullscreenCameraInside; }

  // Jitter-noise mode. When true the shader jitters sample starts with the
  // former Interleaved Gradient Noise (Jimenez 2014) instead of the GL-parity
  // blue-noise tile (kBlueNoise64) that replaced it. Off by default so renders
  // stay bit-identical to the OpenGL backend.
  void SetUseIGNJitter(bool val) { this->UseIGNJitter = val; }
  bool GetUseIGNJitter() const { return this->UseIGNJitter; }

  // Coherence block size for IGN jitter, in pixels. A per-pixel IGN offset
  // makes adjacent fragments take divergent min-max skip paths (SIMD
  // divergence + minmax-texture cache scatter); sampling the noise once per
  // 2x2 block keeps the stochastic anti-aliasing while marching in lockstep
  // (-20% at coarse sample distances, no change at the default 0.5 spacing).
  // Default 1 = legacy per-pixel behavior (bit-identical to pre-block renders);
  // opt in to a larger block via SetJitterBlockSize(2 or 4).
  void SetJitterBlockSize(int val) { this->JitterBlockSize = std::max(1, val); }
  int GetJitterBlockSize() const { return this->JitterBlockSize; }

  // No-op stubs: the instanced path was removed.  Kept so that any
  // external caller (test / UI) that references the setter still compiles.
  void SetDisableInstanceRendering(bool) {}
  bool GetDisableInstanceRendering() const { return false; }

protected:
  vtkMetalGPUVolumeRayCastMapper();
  ~vtkMetalGPUVolumeRayCastMapper() override;

private:
  vtkMetalGPUVolumeRayCastMapper(const vtkMetalGPUVolumeRayCastMapper&) = delete;
  void operator=(const vtkMetalGPUVolumeRayCastMapper&) = delete;

  // Metal pipeline objects (stored as void* to avoid Obj-C in header)
  void* PipelineState = nullptr;         // id<MTLRenderPipelineState>
  void* VolumeTexture = nullptr;         // id<MTLTexture>  (3D)

  void* ColorOpacityTexture = nullptr;   // id<MTLTexture>  (2D)
  void* GradientOpacityTexture = nullptr; // id<MTLTexture> (256x1 RGBA8Unorm)
  void* Transfer2DTexture = nullptr;     // id<MTLTexture> (2D RGBA16Float) — 2D transfer function image
  void* Transfer2DYAxisTexture = nullptr; // id<MTLTexture> (3D R16Float/R32Float) — Y-axis scalar array (e.g. "Temp")
  vtkTimeStamp Transfer2DUploadTime;
  vtkTimeStamp Transfer2DYAxisUploadTime;
  double Transfer2DYAxisRange[2] = { 0.0, 1.0 }; // value range of the Y-axis array
  std::string Transfer2DYAxisArrayName;        // cached array name to detect changes
  bool Transfer2DEnabled = false;        // TF_2D mode active and textures ready
  bool Transfer2DUseGradient = false;    // TF_2D without a Y-axis array: y-axis is gradient magnitude (OpenGL parity)
  void* MinMaxTexture = nullptr;         // id<MTLTexture> (3D) — 4x downsampled min-max accel
  void* MinMaxScratchTexture = nullptr;  // id<MTLTexture> — reusable scratch occupancy (R8Unorm 3D)
  int MinMaxDims[3] = {};               // dimensions of the min-max texture
  vtkTimeStamp MinMaxUploadTime;
  // Two-level occupancy summary (VTK_METAL_TEST_MM_BLOCKS): coarse R8 texture
  // marking whole-block all-empty regions of the dilated MinMaxTexture, so the
  // fragment walk leaps multiple macrocells per lattice fetch. Byte-identical
  // output (block emptiness derives from the exact per-cell semantics).
  void* MinMaxBlockTexture = nullptr;    // id<MTLTexture> (3D R8Unorm)
  int MinMaxBlockDims[3] = {};          // block-summary grid dims
  int MinMaxBlockSize = 0;              // block edge in fine-lattice cells (cache key)
  void* BlockReduceComputePipeline = nullptr; // id<MTLComputePipelineState> — volume_reduce_minmax_blocks
  // §38.16 (VTK_METAL_TEST_MM_MIP): same reduce writing into mip level
  // log2(blockSize) of the fine lattice (volume_reduce_minmax_mipblocks).
  void* MipBlockReduceComputePipeline = nullptr;
  // §38.17 MM_SEG_DEBUG staging (shared) for CPU-side pool/index dumps.
  void* SegDebugStageBuffer = nullptr;
  // §38.17 consume-state debug outlet (16 words shared, center ray).
  void* SegConsumeDbgBuffer = nullptr;
  size_t SegDebugStageBytes = 0;
  // Fraction of all-empty blocks in the last block-summary build, written by
  // the reduce dispatch's completion handler. When skipping cannot pay on a
  // transfer function (mostly-solid lattice), BuildPerBlockData clears the
  // walk-enable flag so the march matches raw cost instead of paying the
  // lattice walk for nothing.
  std::atomic<float> MinMaxEmptyBlockFraction{1.0f};
  std::size_t MinMaxEmptyBlockTotal = 0;  // blocks per build (denominator)
  void* MinMaxCountBuffer = nullptr;     // id<MTLBuffer> shared uint atomic counter
  // §35.5 headroom A/B (VTK_METAL_TEST_MM_SUPER): third occupancy level —
  // whole 8³-block groups of the block summary that are all-empty.
  void* MinMaxSuperTexture = nullptr;    // id<MTLTexture> (3D R8Unorm)
  int MinMaxSuperDims[3] = {};           // super-summary grid dims
  void* SuperReduceComputePipeline = nullptr; // volume_reduce_minmax_superblocks

  // §35.14 async segment pre-pass (VTK_METAL_TEST_MM_SEG=1): raster ray-atlas
  // + compute segment builder + fragment streaming consume. Default-off.
  void* SegAtlasATexture = nullptr;      // id<MTLTexture> 2D RGBA32Float: (evalPoint.xyz, steps)
  void* SegAtlasBTexture = nullptr;      // id<MTLTexture> 2D RGBA32Float: (evalStep.xyz, stepSize)
  void* SegAtlasCTexture = nullptr;      // id<MTLTexture> 2D RGBA32Float: (rayDir.xyz)
  void* SegIndexBuffer = nullptr;        // id<MTLBuffer> u32 per-pixel pool offset (UINT_MAX = none)
  void* SegPoolBuffer = nullptr;         // id<MTLBuffer> compacted gap records (word0=count, then (start<<16|end) u16 pairs)
  void* SegPoolCounterBuffer = nullptr;  // id<MTLBuffer> shared u32 atomic claim counter (CPU-zeroed per frame)
  void* SegDummyBuffer = nullptr;        // id<MTLBuffer> 16B zeros bound at slots 6/7 when disabled
  void* SegMarchTexture = nullptr;       // id<MTLTexture> 2D RGBA16Float offscreen march target
  void* SegBuildComputePipeline = nullptr; // id<MTLComputePipelineState> — volume_segment_build
  int SegAtlasWidth = 0;
  int SegAtlasHeight = 0;
  std::atomic<uint32_t> SegLastClaimWords{0}; // TEMP-DIAG §35.14
  std::size_t SegPoolCapWords = 0;
  bool SegActiveThisFrame = false;

  // §38.17 per-camera segment cache: builder runs once per camera pose;
  // static views amortize to zero, orbiting pays one build per frame.
  bool SegCacheValid = false;
  int SegCacheWidth = 0;
  int SegCacheHeight = 0;
  std::vector<uint8_t> SegCacheUniformBytes;
  std::vector<uint8_t> SegCachePbdBytes;
  size_t SegCachePoolCapWords = 0;
  vtkTimeStamp SegCacheMinMaxTime;

  // §38.6 / §36.4 Design B — Compute Marcher & Ray-Binned Marching
  void* ComputeMarchPipeline = nullptr; // id<MTLComputePipelineState> — volume_compute_march
  void* RayBinClassifyPipeline = nullptr; // id<MTLComputePipelineState> — volume_ray_bin_classify
  void* ComputeMarchBinnedPipeline = nullptr; // id<MTLComputePipelineState> — volume_compute_march_binned
  void* RayBinIndicesBuffer = nullptr; // id<MTLBuffer> uint32 packed UVs per bin
  void* RayBinCountersBuffer = nullptr; // id<MTLBuffer> uint32 atomic counters (CPU-zeroed per frame)
  size_t RayBinIndicesCapBytes = 0;
  std::unordered_map<VolumePipelineKey, void*, VolumePipelineKeyHash> ComputeMarchPipelineCache;
  std::unordered_map<VolumePipelineKey, void*, VolumePipelineKeyHash> ComputeMarchBinnedPipelineCache;
  void* ComputeMarchQueue = nullptr; // probe-selected fast-slot queue (§38.8)

  // Cinematic — shaded DVR (wax AO/SSS, front-to-back over, 1 spp)
  // Single compute variant; binned majorant path deleted (speckle at 1 spp, kernel removed).
  void* CinematicComputePipeline = nullptr; // volume_compute_march_cinematic
  std::unordered_map<VolumePipelineKey, void*, VolumePipelineKeyHash> CinematicComputePipelineCache;
  void* CinematicAccumTextureA = nullptr; // RGBA16Float ping-pong accumulation
  void* CinematicAccumTextureB = nullptr;
  int CinematicAccumWidth = 0;
  int CinematicAccumHeight = 0;
  uint32_t CinematicFrameSeed = 0;
  uint32_t CinematicAccumCount = 0;
  bool CinematicAccumValid = false;
  double CinematicLastCameraMTime = 0;
  double CinematicLastTransferMTime = 0;
  vtkTimeStamp CinematicLastVolumeTime;
  // Denoise via MPS (guided filter) when CinematicDenoise > 0
  void* CinematicDenoiseTexture = nullptr; // intermediate for MPS
  void* CinematicDenoisePipeline = nullptr; // cached pipeline for volume_cinematic_denoise

  void* DepthStencilState = nullptr;     // id<MTLDepthStencilState>
  void* DepthTextureOcclusion = nullptr; // id<MTLTexture> — scene depth for early ray termination
  void* DummyDepthTexture = nullptr;     // id<MTLTexture> — 1x1 R32Float(1.0) fallback when no depth available
  void* NoiseTexture = nullptr;          // id<MTLTexture> — 64x64 R8Unorm blue-noise tile (correlated jitter sampling)
  void* DummyVolumeTexture = nullptr;    // id<MTLTexture> — 1x1x1 R32Float fallback for nil volume tex
  void* DummyMaskTexture = nullptr;      // id<MTLTexture> — 1x1x1 R32Float fallback for nil mask tex
  void* DummyMinMaxTexture = nullptr;    // id<MTLTexture> — 1x1x1 R8Unorm fallback for nil minmax tex

  // Mask / label map support
  void* MaskTexture = nullptr;            // id<MTLTexture> (3D) — binary mask or label map
  void* LabelMapTransferTexture = nullptr; // id<MTLTexture> (2D) — label map transfer function
  void* LabelMapGradientOpacityTexture = nullptr; // id<MTLTexture> (2D) — label map gradient opacity
  vtkTimeStamp MaskUpdateTime;
  int LastLabelMapMaxLabel = -1;
  size_t LastLabelMapLabelCount = 0;

  // Blanking support (vtkUniformGrid / ghost arrays): a 3D texture at cell
  // granularity marking which cells/points are blanked. Mirrors the OpenGL
  // backend's blanking texture in vtkVolumeTexture.cxx.
  void* BlankingTexture = nullptr;        // id<MTLTexture> (3D) — RG8Unorm cell-centered blanking flags (.x=point, .y=cell)
  vtkTimeStamp BlankingUploadTime;
  vtkSmartPointer<vtkUnsignedCharArray> BlankingPoints; // cached point ghost array (detects changes)
  vtkSmartPointer<vtkUnsignedCharArray> BlankingCells;  // cached cell ghost array (detects changes)

  // Buffers
  void* UniformBuffers[3] = { nullptr, nullptr, nullptr }; // id<MTLBuffer>[3] — triple-buffered
  int UniformFrameIndex = 0;            // rotation index for triple-buffered uniforms
  void* FrameSemaphore = nullptr;       // dispatch_semaphore_t — gates in-flight frames
  void* VertexBuffer = nullptr;         // id<MTLBuffer>
  void* IndexBuffer = nullptr;          // id<MTLBuffer>
  int IndexCount = 0;
  void* RectCoordsBuffer = nullptr;     // id<MTLBuffer> — float3 per index, rectilinear coord curves
  void* DummyRectCoordsBuffer = nullptr; // id<MTLBuffer> — zeroed fallback for non-rectilinear inputs

  // Volume state
  double ModelBounds[6] = { 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 };
  double ScalarRange[2] = { 0.0, 1.0 };
  float ScalarNormalizationFactor = 1.0f;
  int VolumeNumComponents = 1;
  int CurrentSampleCount = 0;

  // Independent multi-component transfer-function textures (id<MTLTexture>, 2D
  // RGBA16F) for components 1..3; component 0 uses ColorOpacityTexture. Only
  // populated when the volume has > 1 component and vtkVolumeProperty is in
  // independent-components mode (OpenGL OpacityTables[i]/RGBTables[i] parity).
  void* ComponentTransferFunctionTexture1 = nullptr;
  void* ComponentTransferFunctionTexture2 = nullptr;
  void* ComponentTransferFunctionTexture3 = nullptr;
  vtkTimeStamp ComponentTransferFunctionUpdateTime;

  // Per-component scalar ranges (normalized units) for the independent path,
  // plus change detection for TF re-upload (OpenGL ScalarRange[n] parity).
  double ComponentScalarRange[4][2] = { { 0.0, 1.0 }, { 0.0, 1.0 }, { 0.0, 1.0 }, { 0.0, 1.0 } };
  double LastComponentScalarRange[4][2] = { { 0.0, 1.0 }, { 0.0, 1.0 }, { 0.0, 1.0 }, { 0.0, 1.0 } };
  int LastVolumeNumComponents = 1;
  bool LastIndependentComponents = false;

  bool PreferHalfPrecision = true;  // when true, prefer half-float (16-bit) for volume textures when the scalar range fits within [−65504, 65504]; covers native float and integer types
  // Enables a precomputed RGBA8Unorm normal texture to replace 6 gradient
  // fetches per sample with 1 normal texture fetch.  Adds ~4 bytes/voxel of
  // GPU memory and a one-time compute dispatch.  Provides a net benefit only
  // when shading is on (the gradient is unused otherwise).  Disabled by default;
  // set to true when shading is enabled for a ~5x reduction in texture-fetch
  // bandwidth per shaded sample.
  bool UsePrecomputedNormals = false;

  // Phase 4: Precomputed gradient/normal texture (replaces 6 gradient fetches with 1 normal fetch)
  void* GradientNormalTexture = nullptr; // id<MTLTexture> — RGBA8Unorm 3D (normal.xyz*0.5+0.5, gradMag)
  int NormalTextureDims[3] = {};        // dimensions of the normal texture
  void* NormalComputePipeline = nullptr; // id<MTLComputePipelineState>
  bool EnsureGradientNormalTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol);
  void ReleaseGradientNormalTexture();

  // Phase 6: Enables fullscreen ray-cast path when camera is inside the volume.
  // Defaults to true; setupVolumeRay clamps the entry to the near plane, so the
  // fullscreen path reproduces the OpenGL-parity proxy start. Set to false to
  // force the CPU proxy geometry path.
  bool UseFullscreenCameraInside = true;

  // When true, the shader uses Interleaved Gradient Noise (Jimenez 2014) for
  // sample jittering instead of the GL-parity blue-noise tile. Default false.
  bool UseIGNJitter = false;

  // IGN jitter coherence block size in pixels (see SetJitterBlockSize).
  int JitterBlockSize = 1;

  // Phase 5: GPU-based min/max acceleration generation.
  // When true, UpdateMinMaxTexture uses GPU compute kernels instead of CPU
  // vtkSMPTools to build the R8Unorm occupancy texture.
  bool UseGPUMinMax = true;

  // Master switch for the min-max occupancy lattice (see SetUseMinMaxAcceleration).
  bool UseMinMaxAcceleration = true;

  // Compute pipelines for GPU min-max generation.
  void* MinMaxComputePipeline = nullptr;  // id<MTLComputePipelineState> — volume_compute_minmax
  void* DilateComputePipeline = nullptr;  // id<MTLComputePipelineState> — volume_dilate_minmax

  // §28 GPU x<->z volume transpose (VTK_METAL_TEST_GPU_TRANSPOSE, on by
  // default; =0 forces the CPU repack): one-pass compute replacement for the
  // CPU blocked repack in the transposed-volume upload. volume_transpose_xz.
  void* TransposeComputePipeline = nullptr;

  // Ensure the two compute pipelines exist.
  bool EnsureMinMaxComputePipelines(void* mtlDevice);

  // Run GPU min/max generation after volume texture is uploaded.
  // Returns true on success, false on failure (caller falls back to CPU).
  bool ComputeMinMaxGPU(void* mtlDevice, void* mtlQueue, vtkVolume* vol,
    vtkImageData* input, vtkDataArray* scalars);

  // Phase 7: GPU compute kernels for data type conversion.
  // When true, UpdateVolumeTexture uses GPU compute kernels instead of CPU
  // vtkSMPTools loops to convert short/int/double/etc. to the target pixel format.
  bool UseGPUConversion = true;

  // Compute pipelines for GPU data type conversion.
  void* ConvertShortToHalfPipeline = nullptr;   // id<MTLComputePipelineState> — volume_convert_short_to_half
  void* ConvertShortToFloatPipeline = nullptr;  // id<MTLComputePipelineState> — volume_convert_short_to_float
  void* ConvertIntToHalfPipeline = nullptr;     // id<MTLComputePipelineState> — volume_convert_int_to_half
  void* ConvertIntToFloatPipeline = nullptr;    // id<MTLComputePipelineState> — volume_convert_int_to_float
  void* ConvertUIntToHalfPipeline = nullptr;    // id<MTLComputePipelineState> — volume_convert_uint_to_half
  void* ConvertUIntToFloatPipeline = nullptr;   // id<MTLComputePipelineState> — volume_convert_uint_to_float
  void* ConvertFloatToHalfPipeline = nullptr;   // id<MTLComputePipelineState> — volume_convert_float_to_half
  void* ConvertUShortToUCharPipeline = nullptr; // id<MTLComputePipelineState> — volume_convert_ushort_to_uchar
  // Ensure all conversion compute pipelines exist for the given (dataType, useHalf) pair.
  // Returns true on success, false on failure (caller falls back to CPU).
  bool EnsureConversionPipelines(void* mtlDevice);

  // Phase 1A: Cached shader library (avoid recompiling vtkMetalShaders)
  void* CachedShaderLibrary = nullptr; // id<MTLLibrary>
  bool EnsureShaderLibrary(void* mtlDevice);

  // Phase 1B: Pipeline state cache (keyed by format, sample count, feature mask)
  std::unordered_map<VolumePipelineKey, void*, VolumePipelineKeyHash> PipelineCache;

  // Phase 1C: Pipeline pre-warming — set to true after first pre-warm completes
  bool PipelinesPreWarmed = false;

  // Adaptive sample distance
  double ReductionFactor = 1.0;
  double LastTransferFunctionScalarRange[2] = { 0.0, 0.0 };
  // Dependent 2-component (LA) split the table: RGB over component 0's range,
  // A over the last component's range. Track the opacity-range half separately.
  double LastTransferFunctionOpacityScalarRange[2] = { 0.0, 0.0 };
  double LastTransferFunctionSampleDist = -1.0;
  int LastTransferFunctionBlendMode = vtkVolumeMapper::COMPOSITE_BLEND;
  double LastGradientOpacityScalarRange[2] = { 0.0, 0.0 };
  double LastLabelMapScalarRange[2] = { 0.0, 0.0 };
  void ComputeReductionFactor(double allocatedTime);

  // Image-space downsampling (ImageSampleDistance)
  void* ImageSampleColorTexture = nullptr;    // id<MTLTexture> — offscreen color at reduced res
  void* ImageSampleDepthTexture = nullptr;    // id<MTLTexture> — offscreen depth at reduced res
  void* ImageSamplePipeline = nullptr;        // id<MTLRenderPipelineState> — for blit pass
  int ImageSampleFBOWidth = 0;
  int ImageSampleFBOHeight = 0;
  int ImageSamplePixelFormat = 0;             // cached pixel format to detect changes
  bool EnsureImageSampleResources(void* device, int width, int height);
  void ReleaseImageSampleResources();

  // Composite-slab ping-pong (VTK_METAL_TEST_NUM_SLABS > 1): two full-size
  // BGRA8 textures the slab passes alternate writing into, each pass sampling
  // the other texture as the far-side composite. The final texture is exposed
  // to the renderer's Phase 3b blit via the ImageSample members.
  void* SlabTextureA = nullptr;       // id<MTLTexture> — ping-pong target A
  void* SlabTextureB = nullptr;       // id<MTLTexture> — ping-pong target B
  void* SlabDepthTexture = nullptr;   // id<MTLTexture> — Depth32Float, cleared per pass
  int SlabFBOWidth = 0;
  int SlabFBOHeight = 0;
  bool EnsureSlabResources(void* device, int width, int height);
  void ReleaseSlabResources();

  // RenderToImage (color/depth texture export, vtkGPUVolumeRayCastMapper RTT mode)
  void* RTTColorTexture = nullptr;   // id<MTLTexture> — window-sized RGBA8Unorm color
  void* RTTDepthTexture = nullptr;   // id<MTLTexture> — window-sized R32Float depth image
  int RTTWidth = 0;
  int RTTHeight = 0;
  int RTTDepthScalarType = -1;       // cached DepthImageScalarType to detect changes
  bool EnsureRTTResources(void* device, int width, int height, int depthScalarType);
  // §35.14 segment pre-pass resources (atlas textures + index/pool buffers +
  // compute pipeline), cached by size.
  bool EnsureSegResources(void* device, void* mtlQueue, int width, int height);
  void ReleaseRTTResources();


  // Cache/timestamps
  vtkTimeStamp VolumeUploadTime;
  // True while the uploaded VolumeTexture holds a transposed representation
  // (VTK_METAL_TEST_VOLTRANSPOSE). Compute kernels that sample the volume in
  // data space (min-max lattice, normals) read this to swizzle their texture
  // coordinates.
  bool VolumeTextureTransposed = false;
  // §29 orientation of the transposed representation: which ORIGINAL axis the
  // texture's DEPTH extent holds. 0 = identity (not transposed), 1 = X-depth
  // (texture holds z,y,x; fetch coords map via .zyx), 2 = Y-depth (texture
  // holds x,z,y; fetch coords map via .xzy). Chosen per upload by the argmin-
  // extent policy (shortest array dim to depth, ties prefer x — a 12-cell
  // Y-on-ties A/B lost raw axis-y +27%, doc §38) or forced via
  // VTK_METAL_TEST_VOLTRANSPOSE_AXIS=x|y|z (y is the fine-SD/axis-chord
  // opt-in: wins every sd<1.5 view, doc §38). Drives fc_volTransposedY and
  // the compute-kernel uniform code alongside VolumeTextureTransposed.
  int VolumeTextureAxisDepth = 0;
  vtkTimeStamp TransferFunctionUploadTime;
  vtkTimeStamp GradientOpacityUploadTime;
  vtkTimeStamp VertexBufferUploadTime;
  vtkTimeStamp NormalTextureTime;

  // Helper methods
  bool UpdateVolumeTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol);
  // Upload the cell/point ghost-array blanking flags into a cell-centered 3D
  // texture. Returns true on success (or when no blanking is present).
  bool UpdateBlankingTexture(void* mtlDevice, void* mtlQueue, vtkImageData* input);

  // Effective-input abstraction: converts vtkRectilinearGrid inputs and
  // cell-scalar inputs into an equivalent vtkImageData with point scalars so
  // the remainder of the pipeline (which only understands vtkImageData with
  // point data) works unchanged. Returns true on success.
  bool EnsureEffectiveInput();
  vtkSmartPointer<vtkImageData> EffectiveInput; // vtkImageData proxy (point scalars)
  vtkSmartPointer<vtkDataSet> EffectiveInputSource; // dataset the proxy was built from
  vtkTimeStamp EffectiveInputTime;

  // Rectilinear-grid support: per-axis coordinate curves (float3 per index, x/y/z
  // per axis, padded to the longest axis) that let the fragment shader remap the
  // uniform-spacing proxy sampling back to the real index space (OpenGL
  // in_coordTexs / in_coordsScale / in_coordsBias parity). Populated in
  // EnsureEffectiveInput; uploaded to RectCoordsBuffer on volume re-upload.
  bool RectilinearInput = false;
  std::vector<float> RectCoordsData;   // float3 per index, each axis GetScaleAndBias-normalized
  float RectCoordsSizes[3] = { 0.0f, 0.0f, 0.0f };
  float RectCoordsScale[3] = { 1.0f, 1.0f, 1.0f };
  float RectCoordsBias[3] = { 0.0f, 0.0f, 0.0f };
  void UpdateRectilinearCoordsBuffer(void* mtlDevice);

  bool UpdateTransferFunctionTexture(
    void* mtlDevice, void* mtlQueue, vtkVolume* vol,
    double actualSampleDistance);
  bool UpdateGradientOpacityTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol);
  // 2D transfer function mode (TF_2D): uploads the 2D lookup image and the
  // Y-axis scalar array used as its second coordinate.
  bool UpdateTransfer2DTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol);
  bool UpdateTransfer2DYAxisTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol,
    vtkImageData* input);
  bool UpdateMinMaxTexture(void* mtlDevice, vtkVolume* vol, vtkImageData* input, vtkDataArray* scalars, bool skipGlobalTexture = false);
  bool SetupBuffers(void* mtlDevice, vtkRenderer* ren, vtkVolume* vol, vtkImageData* input);
  bool SetupPipeline(void* mtlDevice, vtkRenderer* ren);
  void* GetOrCreateVolumePipeline(void* mtlDevice, uint32_t type,
    uint32_t colorFormat, uint32_t depthFormat, uint32_t sampleCount,
    uint32_t featureMask);
  bool EnsureComputeMarchResources(void* device, void* mtlQueue, int width, int height);
  void* GetOrCreateComputeMarchPipeline(void* mtlDevice, uint32_t featureMask, bool binned);
  void BindComputeMarchTextures(void* encoder, void* atlasA, void* atlasB, void* atlasC, void* outColor);
  // Cinematic — shaded DVR (wax AO/SSS, single 8x8, temporal, bilateral disabled at <4 spp)
  bool EnsureCinematicResources(void* device, int width, int height);
  void* GetOrCreateCinematicComputePipeline(void* mtlDevice, uint32_t featureMask);
  void ReleaseCinematicResources();
  bool DispatchCinematicCompute(void* device, void* queue, void* cmdBuf,
    vtkRenderer* ren, vtkVolume* vol, void* uniformBuf, const void* pbd,
    const void* lightUniforms, int width, int height);
  // §38.18.1: helper that releases all segment-pre-pass Private heaps and
  // invalidates the per-camera seg cache (called by PurgeCaches and
  // ReleaseGraphicsResources).
  void ReleaseSegmentResources();

  // Mask / label map helpers
  bool UpdateMaskTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol);
  bool UpdateLabelMapTransferTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol);
  void ReleaseMaskResources();
  void SetMaskUniforms(void* uniforms, vtkVolume* vol);

  // Near-plane clipping
  bool IsCameraInside(vtkRenderer* ren, vtkVolume* vol);
  bool CameraWasInsideInLastUpdate = false;

  // Clipping planes — up to 8 arbitrary clipping planes
  void SetClippingPlaneUniforms(void* uniforms, vtkRenderer* ren, vtkVolume* vol,
    vtkMatrix4x4* modelMatrix, vtkMatrix4x4* invModelMatrix);

  // Bind all volume fragment textures at fixed indices for the fullscreen paths.
  // volTex/minMaxTex/normalTex are per-block (or global for single-block).
  // The PerBlockData is bound at index 2 (vertex + fragment).
  // cullMode: MTL_CullModeBack or MTL_CullModeNone (layer composite uses none).
  void BindFullscreenTextures(void* encoder, void* uniformBuf,
    void* volTex, void* minMaxTex, void* normalTex,
    bool useDepth, const void* pbd, uint32_t cullMode);

  // Wait for all in-flight GPU frames to complete (safe teardown)
  void WaitForInFlightFrames();

  // Rendering helpers — shared between image-sampling and standard paths
  void BindEncoderResources(void* encoder, void* uniformBuf, void* pipelineState = nullptr,
    bool hasDepth = false);
  void DrawBlocks(void* encoder, void* uniformBuf, vtkRenderer* ren, vtkVolume* vol,
    void* uniforms, vtkMatrix4x4* invModelMatrix, int slabIndex = 0, int slabCount = 1);
  // Fullscreen camera-inside draw path.
  // Renders each non-empty brick using a fullscreen triangle (vertex_fullscreen_main +
  // fragment_volume_fullscreen_main) instead of proxy geometry. No vertex/index buffers
  // needed — the fullscreen vertex shader generates positions internally.
  void DrawBlocksFullscreen(void* encoder, void* uniformBuf, vtkRenderer* ren, vtkVolume* vol,
    void* uniforms, vtkMatrix4x4* invModelMatrix, bool useDirectPipeline,
    uint32_t lightingFeatureBits);

  // Build PerBlockData from global uniforms for single-block volumes.
  static void BuildPerBlockData(PerBlockData& pbd, const VolumeMapperUniforms* uniforms);

  // Volume light uniforms for multi-light shading support.
  struct VolumeLightData {
    float position[4];
    float direction[4];
    float ambientColor[4];
    float diffuseColor[4];
    float specularColor[4];
    float attenuation[4];
  };
  struct VolumeLightUniforms {
    VolumeLightData lights[8];
    int32_t lightCount;
    int32_t numPositionalLights;
    int32_t twoSidedLighting;
    int32_t defaultLighting;
    int32_t _pad[4];
  };
  void BuildVolumeLightUniforms(vtkRenderer* ren, vtkVolume* vol,
    vtkMatrix4x4* invModelMatrix, const double modelBounds[6],
    const double boundsSize[3], VolumeLightUniforms& out);

  // Feature bits for the lighting state (fc_defaultLighting / fc_lightCount),
  // derived from the built VolumeLightUniforms. Kept separate from the rest of
  // the feature mask because the fullscreen pipeline path
  // (DrawBlocksFullscreen) does not otherwise see the light uniforms.
  static uint32_t VolumeLightingFeatureBits(const VolumeLightUniforms& lights);

  unsigned short Partitions[3] = { 1, 1, 1 };

  // Per-macrocell scalar min/max — computed by UpdateMinMaxTexture and consumed
  // by EnsureGridTraversalResources to build the brick occupancy grid.
  std::vector<float> MacrocellScalarMin;
  std::vector<float> MacrocellScalarMax;

  // Grid traversal data for single-pass partitioned volume rendering
  void* OccupancyGridTexture = nullptr;     // id<MTLTexture> — R8Unorm 3D, dims = Partitions
  void* GridTraversalUniformBuffer = nullptr; // id<MTLBuffer> — GridTraversalUniforms
  int CachedGridDims[3] = {};
  bool GridTraversalResourcesValid = false;
  vtkTimeStamp GridTraversalUploadTime;
  void EnsureGridTraversalResources(void* mtlDevice, void* mtlQueue,
    vtkImageData* input, vtkVolume* vol);
  void ReleaseGridTraversalResources();
  bool CreateGlobalVolumeTexture(void* mtlDevice, void* mtlQueue,
    vtkImageData* input, vtkDataArray* scalars);
  void BindGridTraversalTextures(void* encoder, void* uniformBuf,
    void* volTex, void* minMaxTex, void* normalTex,
    bool useDepth, const void* pbd, uint32_t cullMode);
  void BuildGlobalPerBlockData(PerBlockData& pbd, vtkImageData* input);
};

#define vtkMetalGPUVolumeRayCastMapper_OVERRIDE_ATTRIBUTES \
  vtkMetalGPUVolumeRayCastMapper::CreateOverrideAttributes()

VTK_ABI_NAMESPACE_END
#endif
