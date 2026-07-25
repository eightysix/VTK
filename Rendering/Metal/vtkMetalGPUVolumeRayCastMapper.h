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

#include <array>  // For std::array
#include <vector> // For std::vector

class vtkDataArray;
class vtkImageData;
class vtkPiecewiseFunction;
class vtkVolume;

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
  void* VolumeSampler = nullptr;         // id<MTLSamplerState>

  void* ColorOpacityTexture = nullptr;   // id<MTLTexture>  (2D)
  void* ColorOpacitySampler = nullptr;   // id<MTLSamplerState>
  void* GradientOpacityTexture = nullptr; // id<MTLTexture> (256x1 RGBA8Unorm)
  void* GradientOpacitySampler = nullptr; // id<MTLSamplerState>
  void* MinMaxTexture = nullptr;         // id<MTLTexture> (3D) — 4x downsampled min-max accel
  void* MinMaxSampler = nullptr;         // id<MTLSamplerState> — nearest sampler for min-max
  int MinMaxDims[3] = {};               // dimensions of the min-max texture
  vtkTimeStamp MinMaxUploadTime;
  void* DepthStencilState = nullptr;     // id<MTLDepthStencilState>
  void* DepthTextureOcclusion = nullptr; // id<MTLTexture> — scene depth for early ray termination
  void* DummyDepthTexture = nullptr;     // id<MTLTexture> — 1x1 R32Float(1.0) fallback when no depth available
  void* DummyVolumeTexture = nullptr;    // id<MTLTexture> — 1x1x1 R8Unorm fallback when no volume texture
  void* DummyTransferFunctionTexture = nullptr; // id<MTLTexture> — 1x1 RGBA8Unorm fallback when no TF texture
  void* DepthSampler = nullptr;          // id<MTLSamplerState> — nearest sampler for depth texture

  // Mask / label map support
  void* MaskTexture = nullptr;            // id<MTLTexture> (3D) — binary mask or label map
  void* MaskSampler = nullptr;            // id<MTLSamplerState> — nearest sampler for mask
  void* LabelMapTransferTexture = nullptr; // id<MTLTexture> (2D) — label map transfer function
  void* LabelMapTransferSampler = nullptr; // id<MTLSamplerState> — nearest sampler for label map TF
  void* LabelMapGradientOpacityTexture = nullptr; // id<MTLTexture> (2D) — label map gradient opacity
  void* LabelMapGradientOpacitySampler = nullptr; // id<MTLSamplerState> — nearest sampler for label map grad op
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

  // Feature-valid flags — set each frame to prevent shaders from sampling partially updated resources
  bool GradientOpacityValid = false;
  bool MaskValid = false;
  bool LabelMapValid = false;
  bool MinMaxValid = false;
  bool BlocksValid = false;

  // Cached Metal queue for use during destruction
  void* CachedMetalQueue = nullptr;

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
  void* ImageSampleSampler = nullptr;         // id<MTLSamplerState> — linear sampler for blit
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

  // Helper methods
  bool EnsureDummyResources(void* mtlDevice);
  void WaitForGPUIdle(void* queueVoid);

  bool UpdateVolumeTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol);
  bool UpdateTransferFunctionTexture(
    void* mtlDevice, void* mtlQueue, vtkVolume* vol, double actualSampleDistance);
  bool UpdateGradientOpacityTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol);
  bool UpdateMinMaxTexture(void* mtlDevice, vtkVolume* vol, vtkImageData* input, vtkDataArray* scalars, bool skipGlobalTexture = false);
  bool SetupBuffers(void* mtlDevice, vtkRenderer* ren, vtkVolume* vol, vtkImageData* input);
  bool SetupPipeline(void* mtlDevice, vtkRenderer* ren);

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
    vtkMatrix4x4* invModelMatrix);

  // Rendering helpers — shared between image-sampling and standard paths
  void BindEncoderResources(void* encoder, void* uniformBuf, void* pipelineState = nullptr);
  void DrawBlocks(void* encoder, void* uniformBuf, vtkRenderer* ren, vtkVolume* vol,
    void* uniforms, vtkMatrix4x4* invModelMatrix);

  // Volume partitioning — splits large volumes into blocks for 3D texture size limits
  struct VolumeBlock
  {
    void* Texture = nullptr; // id<MTLTexture> — 3D sub-texture for this block
    void* MinMaxTexture = nullptr; // id<MTLTexture> — per-block min-max accel (R8Unorm)
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
  // Each brick renders into its own RGBA16Float layer; a final composite pass
  // sorts the layers per-pixel by ray-entry depth and folds front-to-back.
  // This eliminates the bright ring caused by framebuffer-fetch ordering.
  // Covered bricks: <= MAX_LAYER_BRICKS (8).  Volumes with more partitions
  // fall through to AccumulationPipelineState (>8 fallback, order-dependent).
  void* LayerColorTexture[8] = { nullptr };  // id<MTLTexture> — one per brick slot
  void* LayerPipelineState = nullptr;        // vertex_volume_main + fragment_volume_main, RGBA16Float
  void* CompositePipelineState = nullptr;    // vertex_fullscreen_main + fragment_layer_composite_main
  int LayerFBOWidth = 0;
  int LayerFBOHeight = 0;
  bool EnsureLayerResources(void* device, int w, int h);
};

VTK_ABI_NAMESPACE_END
#endif
