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
#include <cstdint>       // For uint16_t, uint32_t

class vtkDataArray;
class vtkImageData;
class vtkPiecewiseFunction;
class vtkVolume;

// Forward declarations for types defined in the .mm file.
struct PerBlockData;

// Metal constant-address-space structs align float3 to 16 bytes (size 16),
// float4/float4x4 to 16 bytes, and float2 to 8 bytes.  This creates
// padding that plain C++ float[] arrays do not.  The layout below exactly
// mirrors the Metal compiler's computation (976 bytes total).
struct VolumeMapperUniforms
{
  float WorldToVolumeMatrix[16];     // 0..63
  float VolumeToWorldMatrix[16];     // 64..127
  float VolumeBoundsMin[4];          // 128..143
  float VolumeBoundsMax[4];          // 144..159
  float CameraVolumePos[4];          // 160..175
  float ViewProjectionMatrix[16];    // 176..239
  uint16_t SampleDistanceHalf;      // 240
  uint16_t OpacityPreIntegrationFactorHalf; // 242
  uint16_t ScalarMinHalf;           // 244
  uint16_t _padSM;                  // 246
  uint16_t ScalarMaxHalf;           // 248
  uint16_t _padSMax;                // 250
  float UseJittering;                // 252
  float InverseViewProjection[16];   // 256..319
  float ViewportSize[2];            // 320..327
  float _padViewport[2];            // 328..335
  float GradientStep[3];            // 336..347
  float _padGradStep;               // 348..351
  float UseGradientShading;         // 352
  float _padGradOpRange;            // 356..359
  float GradientOpacityMin;         // 360
  float GradientOpacityMax;         // 364
  float UseGradientOpacity;         // 368
  float _padAmbient[3];             // 372..383
  float AmbientColor[3];            // 384..395
  float _padAmb;                    // 396..399
  float DiffuseColor[3];            // 400..411
  float _padDiff;                   // 412..415
  float SpecularColor[3];           // 416..427
  float _padSpec;                   // 428..431
  float Shininess;                  // 432
  float _padLightDir[3];            // 436..447
  float LightDirection[3];          // 448..459
  float _padLight;                  // 460..463
  float _padEnd[4];                 // 464..479
  float CroppingPlanes[4];          // 480..495
  float CroppingPlanes2[4];         // 496..511
  uint32_t CroppingBitmask;         // 512..515
  float UsePreIntegratedTF;         // 516 (was _padCropFlags[0])
  float PreIntegStepFactor;         // 520 (was _padCropFlags[1])
  float _padCropFlags[29];          // 524..639
  float UseCropping;                // 640
  float UseClipping;                // 644
  float NumClippingPlanes;          // 648
  float _padClipping[2];            // 652..659
  float _padClipAlign[3];           // 660..671
  float ClippingPlane0Origin[4];    // 672..687
  float ClippingPlane0Normal[4];    // 688..703
  float ClippingPlane1Origin[4];    // 704..719
  float ClippingPlane1Normal[4];    // 720..735
  float ClippingPlane2Origin[4];    // 736..751
  float ClippingPlane2Normal[4];    // 752..767
  float ClippingPlane3Origin[4];    // 768..783
  float ClippingPlane3Normal[4];    // 784..799
  float ClippingPlane4Origin[4];    // 800..815
  float ClippingPlane4Normal[4];    // 816..831
  float ClippingPlane5Origin[4];    // 832..847
  float ClippingPlane5Normal[4];    // 848..863
  float ClippingPlane6Origin[4];    // 864..879
  float ClippingPlane6Normal[4];    // 880..895
  float ClippingPlane7Origin[4];    // 896..911
  float ClippingPlane7Normal[4];    // 912..927
  float UseMask;                  // 928
  float MaskBlendFactor;          // 932
  float MaskScale;                // 936
  float MaskBias;                 // 940
  float LabelMapNumLabels;        // 944
  float UseDepthTexture;          // 948
  float UseNormalTexture;         // 952
  float _padMask;                 // 956
  float UseMinMaxAccel;           // 960
  float MinMaxDimX;               // 964
  float MinMaxDimY;               // 968
  float MinMaxDimZ;               // 972
};

// RAII wrapper for Metal Obj-C resources stored as void*.
// Defined in .mm (injects release/retain via Obj-C bridging).
class VTKRENDERINGMETAL_EXPORT vtkMetalResource
{
  void* Obj = nullptr;

public:
  vtkMetalResource() = default;
  ~vtkMetalResource();
  vtkMetalResource(vtkMetalResource&& o) noexcept;
  vtkMetalResource& operator=(vtkMetalResource&& o) noexcept;
  vtkMetalResource(const vtkMetalResource&) = delete;
  vtkMetalResource& operator=(const vtkMetalResource&) = delete;

  vtkMetalResource& operator=(void* o);
  void reset();
  void take(void* o);
  void retain(void* o);
  void* get() const { return this->Obj; }
  operator void*() const { return this->Obj; }
};

// Pipeline cache types for Phase 1B
enum class VolumePipelineType : uint32_t
{
  DirectScreen = 0,
  OffscreenAccumulation = 1,
  OffscreenLayer = 2,
  LayerComposite = 3,
  ImageSampleBlit = 4,
  FullscreenDirect = 5,      // Fullscreen ray-cast for camera-inside (BGRA8Unorm + depth)
  FullscreenOffscreen = 6,   // Fullscreen ray-cast for camera-inside (RGBA16Float, no depth)
  FullscreenAccumulation = 7 // Fullscreen ray-cast with framebuffer fetch for multi-block accumulation
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
  VolumeFeature_PreIntegratedTF  = 1u << 5,
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
  void* GetImageSampleColorTexture() const { return this->ImageSampleColorTexture.get(); }
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

  void SetPreferHalfPrecision(bool val) { this->PreferHalfPrecision = val; }
  bool GetPreferHalfPrecision() const { return this->PreferHalfPrecision; }

  void SetUseGPUMinMax(bool val) { this->UseGPUMinMax = val; }
  bool GetUseGPUMinMax() const { return this->UseGPUMinMax; }

  // Phase 6: Fullscreen camera-inside path.
  void SetUseFullscreenCameraInside(bool val) { this->UseFullscreenCameraInside = val; }
  bool GetUseFullscreenCameraInside() const { return this->UseFullscreenCameraInside; }

  // No-op stubs: the instanced path was removed.
  void SetDisableInstanceRendering(bool) {}
  bool GetDisableInstanceRendering() const { return false; }

protected:
  vtkMetalGPUVolumeRayCastMapper();
  ~vtkMetalGPUVolumeRayCastMapper() override;

private:
  vtkMetalGPUVolumeRayCastMapper(const vtkMetalGPUVolumeRayCastMapper&) = delete;
  void operator=(const vtkMetalGPUVolumeRayCastMapper&) = delete;

  // Metal pipeline objects (RAII-managed void*)
  vtkMetalResource PipelineState;            // id<MTLRenderPipelineState>
  vtkMetalResource AccumulationPipelineState; // id<MTLRenderPipelineState>
  vtkMetalResource VolumeTexture;            // id<MTLTexture>  (3D)

  vtkMetalResource ColorOpacityTexture;     // id<MTLTexture> (2D)
  vtkMetalResource GradientOpacityTexture;  // id<MTLTexture> (256x1 RGBA8Unorm)
  vtkMetalResource MinMaxTexture;           // id<MTLTexture> (3D) — 4x downsampled min-max accel
  vtkMetalResource MinMaxScratchTexture;    // id<MTLTexture> — reusable scratch occupancy (R8Unorm 3D)
  int MinMaxDims[3] = {};                  // dimensions of the min-max texture
  vtkTimeStamp MinMaxUploadTime;
  vtkMetalResource DepthStencilState;       // id<MTLDepthStencilState>
  vtkMetalResource DepthTextureOcclusion;   // id<MTLTexture> — scene depth for early ray termination
  vtkMetalResource DummyDepthTexture;       // id<MTLTexture> — 1x1 R32Float(1.0) fallback
  vtkMetalResource DummyVolumeTexture;      // id<MTLTexture> — 1x1x1 R32Float fallback
  vtkMetalResource DummyMaskTexture;        // id<MTLTexture> — 1x1x1 R32Float fallback
  vtkMetalResource DummyMinMaxTexture;      // id<MTLTexture> — 1x1x1 R8Unorm fallback

  // Mask / label map support
  vtkMetalResource MaskTexture;             // id<MTLTexture> (3D) — binary mask or label map
  vtkMetalResource LabelMapTransferTexture;  // id<MTLTexture> (2D) — label map transfer function
  vtkMetalResource LabelMapGradientOpacityTexture; // id<MTLTexture> (2D) — label map gradient opacity
  vtkTimeStamp MaskUpdateTime;
  int LastLabelMapMaxLabel = -1;
  size_t LastLabelMapLabelCount = 0;

  // Buffers
  vtkMetalResource UniformBuffers[3];       // id<MTLBuffer>[3] — triple-buffered
  int UniformFrameIndex = 0;                // rotation index for triple-buffered uniforms
  vtkMetalResource FrameSemaphore;          // dispatch_semaphore_t — gates in-flight frames
  vtkMetalResource VertexBuffer;            // id<MTLBuffer>
  vtkMetalResource IndexBuffer;             // id<MTLBuffer>
  int IndexCount = 0;

  // Volume state
  double ModelBounds[6] = { 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 };
  double ScalarRange[2] = { 0.0, 1.0 };
  float ScalarNormalizationFactor = 1.0f;
  int VolumeNumComponents = 1;
  int CurrentSampleCount = 0;

  bool PreferHalfPrecision = true;
  bool UsePrecomputedNormals = false;

  // Phase 4: Precomputed gradient/normal texture
  vtkMetalResource GradientNormalTexture;    // id<MTLTexture> — RGBA8Unorm 3D
  int NormalTextureDims[3] = {};
  vtkMetalResource NormalComputePipeline;    // id<MTLComputePipelineState>
  bool EnsureGradientNormalTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol);
  void ReleaseGradientNormalTexture();

  bool UseFullscreenCameraInside = true;

  // Volume partitioning — splits large volumes into blocks for 3D texture size limits
  struct VolumeBlock
  {
    vtkMetalResource Texture;        // id<MTLTexture> — 3D sub-texture for this block
    vtkMetalResource MinMaxTexture;  // id<MTLTexture> — per-block min-max accel (R8Unorm)
    vtkMetalResource NormalTexture;  // id<MTLTexture> — per-block precomputed normals (RGBA8Unorm)
    double BoundsMin[3] = {};
    double BoundsMax[3] = {};
    int Dims[3] = {};
    int MinMaxDims[3] = {};
    int Extents[6] = {};
    double Center[3] = {};

    VolumeBlock() = default;
    VolumeBlock(const VolumeBlock& o)
      : BoundsMin{ o.BoundsMin[0], o.BoundsMin[1], o.BoundsMin[2] }
      , BoundsMax{ o.BoundsMax[0], o.BoundsMax[1], o.BoundsMax[2] }
      , Dims{ o.Dims[0], o.Dims[1], o.Dims[2] }
      , MinMaxDims{ o.MinMaxDims[0], o.MinMaxDims[1], o.MinMaxDims[2] }
      , Extents{ o.Extents[0], o.Extents[1], o.Extents[2], o.Extents[3], o.Extents[4], o.Extents[5] }
      , Center{ o.Center[0], o.Center[1], o.Center[2] }
    {
      Texture.retain(o.Texture.get());
      MinMaxTexture.retain(o.MinMaxTexture.get());
      NormalTexture.retain(o.NormalTexture.get());
    }
    VolumeBlock& operator=(const VolumeBlock& o)
    {
      if (this != &o)
      {
        Texture.retain(o.Texture.get());
        MinMaxTexture.retain(o.MinMaxTexture.get());
        NormalTexture.retain(o.NormalTexture.get());
        std::copy_n(o.BoundsMin, 3, BoundsMin);
        std::copy_n(o.BoundsMax, 3, BoundsMax);
        std::copy_n(o.Dims, 3, Dims);
        std::copy_n(o.MinMaxDims, 3, MinMaxDims);
        std::copy_n(o.Extents, 6, Extents);
        std::copy_n(o.Center, 3, Center);
      }
      return *this;
    }
  };

  // Phase 5: GPU min-max
  bool UseGPUMinMax = true;
  vtkMetalResource MinMaxComputePipeline;    // id<MTLComputePipelineState>
  vtkMetalResource DilateComputePipeline;    // id<MTLComputePipelineState>
  vtkMetalResource MinMaxDownsamplePipeline; // id<MTLComputePipelineState>
  bool EnsureMinMaxComputePipelines(void* mtlDevice);
  bool ComputeMinMaxGPU(void* mtlDevice, void* mtlQueue, vtkVolume* vol,
    vtkImageData* input, vtkDataArray* scalars);
  static int ComputeMinMaxMipLevels(int dimX, int dimY, int dimZ);
  static int ComputeShaderSkipLevels(int numMipLevels);
  // GPU min-max dispatch helper for per-block generation
  void DispatchBlockMinMaxGPU(void* device, void* mmEnc, void* blockTex, VolumeBlock& block,
    const int bdims[3], const float normFactor, const double scalarRange,
    const double opacityTable[256]);

  // Phase 7: GPU data type conversion
  bool UseGPUConversion = true;
  vtkMetalResource ConvertShortToHalfPipeline;
  vtkMetalResource ConvertShortToFloatPipeline;
  vtkMetalResource ConvertIntToHalfPipeline;
  vtkMetalResource ConvertIntToFloatPipeline;
  vtkMetalResource ConvertUIntToHalfPipeline;
  vtkMetalResource ConvertUIntToFloatPipeline;
  vtkMetalResource ConvertFloatToHalfPipeline;
  vtkMetalResource ConvertUShortToUCharPipeline;
  bool EnsureConversionPipelines(void* mtlDevice);

  // Phase 1A: Cached shader library
  vtkMetalResource CachedShaderLibrary;      // id<MTLLibrary>
  bool EnsureShaderLibrary(void* mtlDevice);

  // Phase 1B: Pipeline state cache
  std::unordered_map<VolumePipelineKey, void*, VolumePipelineKeyHash> PipelineCache;

  // Phase 1C: Pipeline pre-warming
  bool PipelinesPreWarmed = false;

  // Adaptive sample distance
  double ReductionFactor = 1.0;
  double LastTransferFunctionScalarRange[2] = { 0.0, 0.0 };
  double LastGradientOpacityScalarRange[2] = { 0.0, 0.0 };
  double LastLabelMapScalarRange[2] = { 0.0, 0.0 };
  void ComputeReductionFactor(double allocatedTime);

  // Image-space downsampling
  vtkMetalResource ImageSampleColorTexture;  // id<MTLTexture>
  vtkMetalResource ImageSampleDepthTexture;  // id<MTLTexture>
  vtkMetalResource ImageSamplePipeline;      // id<MTLRenderPipelineState>
  int ImageSampleFBOWidth = 0;
  int ImageSampleFBOHeight = 0;
  int ImageSamplePixelFormat = 0;
  bool EnsureImageSampleResources(void* device, int width, int height);
  void ReleaseImageSampleResources();

  // Pre-integrated transfer function (Phase 1A)
  vtkMetalResource PreIntegratedTFTexture; // id<MTLTexture> (256x256 RGBA16Float)
  vtkTimeStamp PreIntegratedTFUploadTime;
  double LastPreIntegScalarRange[2] = { 0.0, 0.0 };
  double LastPreIntegUnitDistance = -1.0;

  // Cache/timestamps
  vtkTimeStamp VolumeUploadTime;
  vtkTimeStamp TransferFunctionUploadTime;
  vtkTimeStamp GradientOpacityUploadTime;
  vtkTimeStamp VertexBufferUploadTime;
  vtkTimeStamp NormalTextureTime;

  // Helper methods
  bool UpdateVolumeTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol);
  bool UpdateTransferFunctionTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol);
  bool UpdateGradientOpacityTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol);
  bool UpdateMinMaxTexture(void* mtlDevice, vtkVolume* vol, vtkImageData* input,
    vtkDataArray* scalars, bool skipGlobalTexture = false);
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

  // Clipping planes
  void SetClippingPlaneUniforms(void* uniforms, vtkRenderer* ren, vtkVolume* vol,
    vtkMatrix4x4* modelMatrix, vtkMatrix4x4* invModelMatrix);

  // Pre-integrated TF texture generation
  bool UpdatePreIntegratedTFTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol);

  // Fragment texture binding (consolidated helper)
  void BindFragmentTextures(void* encoder, void* volTex, void* minMaxTex, void* normalTex);

  // Bind all volume fragment textures at fixed indices for fullscreen paths.
  void BindFullscreenTextures(void* encoder, void* uniformBuf,
    void* volTex, void* minMaxTex, void* normalTex,
    bool useDepth, const void* pbd, uint32_t cullMode);

  // Wait for all in-flight GPU frames to complete
  void WaitForInFlightFrames();

  // Rendering helpers
  void BindEncoderResources(void* encoder, void* uniformBuf, void* pipelineState = nullptr,
    bool hasDepth = false);
  void DrawBlocks(void* encoder, void* uniformBuf, vtkRenderer* ren, vtkVolume* vol,
    void* uniforms, vtkMatrix4x4* invModelMatrix);
  void DrawBlocksFullscreen(void* encoder, void* uniformBuf, vtkRenderer* ren, vtkVolume* vol,
    void* uniforms, vtkMatrix4x4* invModelMatrix, bool useDirectPipeline);

  static void BuildPerBlockData(PerBlockData& pbd,
    const VolumeBlock& block,
    const int fullExt[6], const double origin[3], const double spacing[3]);
  static void BuildPerBlockData(PerBlockData& pbd, const VolumeMapperUniforms* uniforms);

  unsigned short Partitions[3] = { 1, 1, 1 };
  std::vector<VolumeBlock> Blocks;
  std::vector<int> SortedBlockOrder;

  // Uniform caching to eliminate redundant per-frame work
  VolumeMapperUniforms CachedUniforms;
  vtkTimeStamp UniformsBuildTime;

  // Cached model/inverse matrices to avoid redundant vtkMatrix4x4::Invert
  double CachedModelMatrixData[16];
  double CachedInvModelMatrixData[16];
  vtkTimeStamp CachedMatrixBuildTime;

  void ClearBlocks();
  void SortBlocksBackToFront(vtkRenderer* ren, vtkVolume* vol);
  bool UpdateBlockTextures(void* mtlDevice, void* mtlQueue, vtkVolume* vol,
    vtkImageData* input, vtkDataArray* scalars, int numComponents);
  bool UpdateBlockMinMaxTextures(void* mtlDevice, void* mtlQueue, vtkVolume* vol,
    vtkImageData* input, vtkDataArray* scalars, int numComponents);

  // Per-block scalar min/max for empty-space skipping
  std::vector<std::array<double, 2>> BlockScalarRanges;
  bool IsBlockEmpty(double blockMin, double blockMax, vtkPiecewiseFunction* opacityFunc);

  // Per-macrocell scalar min/max
  std::vector<float> MacrocellScalarMin;
  std::vector<float> MacrocellScalarMax;

  // Order-independent compositing
  vtkMetalResource LayerTextureArray;        // id<MTLTexture> — 2D array
  int LayerTextureCapacity = 0;
  vtkMetalResource LayerPipelineState;       // id<MTLRenderPipelineState>
  vtkMetalResource CompositePipelineState;   // id<MTLRenderPipelineState>
  int LayerFBOWidth = 0;
  int LayerFBOHeight = 0;
  bool EnsureLayerResources(void* device, int w, int h, int neededSlices);
};

VTK_ABI_NAMESPACE_END
#endif
