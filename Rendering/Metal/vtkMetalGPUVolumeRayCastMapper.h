// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkMetalGPUVolumeRayCastMapper
 * @brief   Metal implementation of volume rendering through ray-casting.
 */

#ifndef vtkMetalGPUVolumeRayCastMapper_h
#define vtkMetalGPUVolumeRayCastMapper_h

#include "vtkGPUVolumeRayCastMapper.h"
#include "vtkRenderingMetalModule.h" // For export macro
#include "vtkTimeStamp.h"            // For time stamp
#include "vtkWrappingHints.h"        // For VTK_MARSHALAUTO

#include <array>        // For std::array
#include <vector>       // For std::vector
#include <unordered_map> // For pipeline cache
#include <functional>    // For std::hash

class vtkDataArray;
class vtkImageData;
class vtkPiecewiseFunction;
class vtkVolume;

// Pipeline cache types for Phase 1B
enum class VolumePipelineType : uint32_t
{
  DirectScreen = 0,
  OffscreenAccumulation = 1,
  OffscreenLayer = 2,
  LayerComposite = 3,
  ImageSampleBlit = 4,
  FullscreenDirect = 5,      // Fullscreen ray-cast for camera-inside (BGRA8Unorm + depth)
  FullscreenOffscreen = 6    // Fullscreen ray-cast for camera-inside (RGBA16Float, no depth)
};

struct VolumePipelineKey
{
  uint32_t type;
  uint32_t colorFormat;
  uint32_t depthFormat;
  uint32_t sampleCount;
  uint32_t featureMask;

  bool operator==(const VolumePipelineKey& other) const
  {
    return type == other.type && colorFormat == other.colorFormat &&
      depthFormat == other.depthFormat && sampleCount == other.sampleCount &&
      featureMask == other.featureMask;
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
    return h;
  }
};

// Feature flags for volume shader specialization via function constants.
// Each flag enables a corresponding [[function_constant(n)]] in the Metal
// shader, allowing the compiler to eliminate dead code paths.
enum VolumeShaderFeatureFlags : uint32_t
{
  VolumeFeature_Shading        = 1u << 0,
  VolumeFeature_GradientOpacity = 1u << 1,
  VolumeFeature_Mask            = 1u << 2,
  VolumeFeature_MinMax          = 1u << 3,
  VolumeFeature_NormalTexture    = 1u << 4,
};

VTK_ABI_NAMESPACE_BEGIN

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkMetalGPUVolumeRayCastMapper
  : public vtkGPUVolumeRayCastMapper
{
public:
  static vtkMetalGPUVolumeRayCastMapper* New();
  vtkTypeMacro(vtkMetalGPUVolumeRayCastMapper, vtkGPUVolumeRayCastMapper);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  void GPURender(vtkRenderer* ren, vtkVolume* vol) override;

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

  void SetUseGPUMinMax(bool val) { this->UseGPUMinMax = val; }
  bool GetUseGPUMinMax() const { return this->UseGPUMinMax; }

  // Phase 6: Fullscreen camera-inside path.
  // When true (default), camera-inside rendering uses a fullscreen ray-cast
  // fragment shader instead of CPU proxy geometry (ClipConvexPolyData +
  // DensifyPolyData + TriangleFilter). Eliminates CPU hitching when the
  // camera enters the volume.
  void SetUseFullscreenCameraInside(bool val) { this->UseFullscreenCameraInside = val; }
  bool GetUseFullscreenCameraInside() const { return this->UseFullscreenCameraInside; }

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
  void* AccumulationPipelineState = nullptr; // id<MTLRenderPipelineState> — for > MAX_LAYER_BRICKS fallback
  void* VolumeTexture = nullptr;         // id<MTLTexture>  (3D)

  void* ColorOpacityTexture = nullptr;   // id<MTLTexture>  (2D)
  void* GradientOpacityTexture = nullptr; // id<MTLTexture> (256x1 RGBA8Unorm)
  void* MinMaxTexture = nullptr;         // id<MTLTexture> (3D) — 4x downsampled min-max accel
  int MinMaxDims[3] = {};               // dimensions of the min-max texture
  vtkTimeStamp MinMaxUploadTime;
  void* DepthStencilState = nullptr;     // id<MTLDepthStencilState>
  void* DepthTextureOcclusion = nullptr; // id<MTLTexture> — scene depth for early ray termination
  void* DummyDepthTexture = nullptr;     // id<MTLTexture> — 1x1 R32Float(1.0) fallback when no depth available
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

  // Buffers
  void* UniformBuffers[3] = { nullptr, nullptr, nullptr }; // id<MTLBuffer>[3] — triple-buffered
  int UniformFrameIndex = 0;            // rotation index for triple-buffered uniforms
  void* FrameSemaphore = nullptr;       // dispatch_semaphore_t — gates in-flight frames
  void* VertexBuffer = nullptr;         // id<MTLBuffer>
  void* IndexBuffer = nullptr;          // id<MTLBuffer>
  int IndexCount = 0;

  // Volume state
  double ModelBounds[6] = { 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 };
  double ScalarRange[2] = { 0.0, 1.0 };
  float ScalarNormalizationFactor = 1.0f;
  int VolumeNumComponents = 1;
  int CurrentSampleCount = 0;

  bool PreferHalfPrecision = true;  // when true, prefer half-float for non-native data types
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
  // Defaults to true (recommended). Set to false to force the old CPU proxy geometry path.
  bool UseFullscreenCameraInside = true;

  // Phase 5: GPU-based min/max acceleration generation.
  // When true, UpdateMinMaxTexture uses GPU compute kernels instead of CPU
  // vtkSMPTools to build the R8Unorm occupancy texture.
  bool UseGPUMinMax = true;

  // Compute pipelines for GPU min-max generation.
  void* MinMaxComputePipeline = nullptr;  // id<MTLComputePipelineState> — volume_compute_minmax
  void* DilateComputePipeline = nullptr;  // id<MTLComputePipelineState> — volume_dilate_minmax

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
  double LastTransferFunctionSampleDistance = -1.0;
  double LastTransferFunctionScalarRange[2] = { 0.0, 0.0 };
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

  // Cache/timestamps
  vtkTimeStamp VolumeUploadTime;
  vtkTimeStamp TransferFunctionUploadTime;
  vtkTimeStamp GradientOpacityUploadTime;
  vtkTimeStamp VertexBufferUploadTime;
  vtkTimeStamp NormalTextureTime;

  // Helper methods
  bool UpdateVolumeTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol);
  bool UpdateTransferFunctionTexture(
    void* mtlDevice, void* mtlQueue, vtkVolume* vol, double actualSampleDistance);
  bool UpdateGradientOpacityTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol);
  bool UpdateMinMaxTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol, vtkImageData* input, vtkDataArray* scalars, bool skipGlobalTexture = false);
  bool SetupBuffers(void* mtlDevice, vtkRenderer* ren, vtkVolume* vol, vtkImageData* input);
  bool SetupPipeline(void* mtlDevice, vtkRenderer* ren);
  void* GetOrCreateVolumePipeline(void* mtlDevice, uint32_t type,
    uint32_t colorFormat, uint32_t depthFormat, uint32_t sampleCount,
    uint32_t featureMask);

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

  // Wait for all in-flight GPU frames to complete (safe teardown)
  void WaitForInFlightFrames();

  // Rendering helpers — shared between image-sampling and standard paths
  void BindEncoderResources(void* encoder, void* uniformBuf, void* pipelineState = nullptr,
    bool hasDepth = false);
  void DrawBlocks(void* encoder, void* uniformBuf, vtkRenderer* ren, vtkVolume* vol,
    void* uniforms, vtkMatrix4x4* invModelMatrix);
  // Fullscreen camera-inside draw path.
  // Renders each non-empty brick using a fullscreen triangle (vertex_fullscreen_main +
  // fragment_volume_fullscreen_main) instead of proxy geometry. No vertex/index buffers
  // needed — the fullscreen vertex shader generates positions internally.
  void DrawBlocksFullscreen(void* encoder, void* uniformBuf, vtkRenderer* ren, vtkVolume* vol,
    void* uniforms, vtkMatrix4x4* invModelMatrix, bool useDirectPipeline);

  // Volume partitioning — splits large volumes into blocks for 3D texture size limits
  struct VolumeBlock
  {
    void* Texture = nullptr; // id<MTLTexture> — 3D sub-texture for this block
    void* MinMaxTexture = nullptr; // id<MTLTexture> — per-block min-max accel (R8Unorm)
    void* NormalTexture = nullptr; // id<MTLTexture> — per-block precomputed normals (RGBA8Unorm)
    double BoundsMin[3] = {};
    double BoundsMax[3] = {};
    int Dims[3] = {};
    int MinMaxDims[3] = {}; // dimensions of the per-block min-max texture
    int Extents[6] = {};
    double Center[3] = {}; // world-space center for sorting
  };

  unsigned short Partitions[3] = { 1, 1, 1 };
  std::vector<VolumeBlock> Blocks;
  std::vector<int> SortedBlockOrder; // indices into Blocks, sorted back-to-front

  void ClearBlocks();
  void SortBlocksBackToFront(vtkRenderer* ren, vtkVolume* vol);
  bool UpdateBlockTextures(void* mtlDevice, void* mtlQueue, vtkVolume* vol,
    vtkImageData* input, vtkDataArray* scalars, int numComponents);

  // Per-block scalar min/max for empty-space skipping
  std::vector<std::array<double, 2>> BlockScalarRanges;
  bool IsBlockEmpty(double blockMin, double blockMax, vtkPiecewiseFunction* opacityFunc);

  // Per-macrocell scalar min/max — computed alongside the occupancy scan
  // in UpdateMinMaxTexture, consumed by UpdateBlockTextures to avoid a
  // redundant full-voxel walk for BlockScalarRanges.
  std::vector<float> MacrocellScalarMin;
  std::vector<float> MacrocellScalarMax;

  // --- Order-independent compositing: per-brick layer textures ---
  // Each brick renders into its own RGBA16Float slice of a 2D texture array;
  // a final composite pass reads the array, sorts the layers per-pixel by
  // ray-entry depth and folds front-to-back.
  // This eliminates the bright ring caused by framebuffer-fetch ordering.
  // Covered bricks: <= MAX_LAYER_BRICKS (8).  Volumes with more partitions
  // fall through to AccumulationPipelineState (>8 fallback, order-dependent).
  void* LayerTextureArray = nullptr;         // id<MTLTexture> — 2D array, RGBA16Float, <= MAX_LAYER_BRICKS slices
  int LayerTextureCapacity = 0;              // current number of slices in the array
  void* LayerPipelineState = nullptr;        // vertex_volume_main + fragment_volume_main, RGBA16Float
  void* CompositePipelineState = nullptr;    // vertex_fullscreen_main + fragment_layer_composite_main
  int LayerFBOWidth = 0;
  int LayerFBOHeight = 0;
  bool EnsureLayerResources(void* device, int w, int h, int neededSlices);
};

VTK_ABI_NAMESPACE_END
#endif
