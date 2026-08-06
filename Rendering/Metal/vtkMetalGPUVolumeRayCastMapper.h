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
  SelectionFullscreen = 9     // Hardware-selection fullscreen ray-cast, camera inside (BGRA8Unorm + depth + RGBA32Uint ids)
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
  VolumeFeature_LinearInterpolation = 1u << 5,
  // Blend modes (vtkVolumeMapper::BlendMode). Composite (0) uses no flag.
  VolumeFeature_BlendMaximumIntensity = 1u << 6,
  VolumeFeature_BlendMinimumIntensity = 1u << 7,
  VolumeFeature_BlendAverageIntensity = 1u << 8,
  VolumeFeature_BlendAdditive         = 1u << 9,
  VolumeFeature_ComputeNormalFromOpacity = 1u << 10,
};

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
  double LastTransferFunctionSampleDist = -1.0;
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

  // RenderToImage (color/depth texture export, vtkGPUVolumeRayCastMapper RTT mode)
  void* RTTColorTexture = nullptr;   // id<MTLTexture> — window-sized RGBA8Unorm color
  void* RTTDepthTexture = nullptr;   // id<MTLTexture> — window-sized R32Float depth image
  int RTTWidth = 0;
  int RTTHeight = 0;
  int RTTDepthScalarType = -1;       // cached DepthImageScalarType to detect changes
  bool EnsureRTTResources(void* device, int width, int height, int depthScalarType);
  void ReleaseRTTResources();


  // Cache/timestamps
  vtkTimeStamp VolumeUploadTime;
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
    void* uniforms, vtkMatrix4x4* invModelMatrix);
  // Fullscreen camera-inside draw path.
  // Renders each non-empty brick using a fullscreen triangle (vertex_fullscreen_main +
  // fragment_volume_fullscreen_main) instead of proxy geometry. No vertex/index buffers
  // needed — the fullscreen vertex shader generates positions internally.
  void DrawBlocksFullscreen(void* encoder, void* uniformBuf, vtkRenderer* ren, vtkVolume* vol,
    void* uniforms, vtkMatrix4x4* invModelMatrix, bool useDirectPipeline);

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
