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

#include <vector> // For std::vector

class vtkDataArray;
class vtkImageData;

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

protected:
  vtkMetalGPUVolumeRayCastMapper();
  ~vtkMetalGPUVolumeRayCastMapper() override;

private:
  vtkMetalGPUVolumeRayCastMapper(const vtkMetalGPUVolumeRayCastMapper&) = delete;
  void operator=(const vtkMetalGPUVolumeRayCastMapper&) = delete;

  // Metal pipeline objects (stored as void* to avoid Obj-C in header)
  void* PipelineState = nullptr;         // id<MTLRenderPipelineState>
  void* VolumeTexture = nullptr;         // id<MTLTexture>  (3D)
  void* VolumeTextureView = nullptr;     // id<MTLTexture>  (alias, same object)
  void* VolumeSampler = nullptr;         // id<MTLSamplerState>

  void* ColorOpacityTexture = nullptr;   // id<MTLTexture>  (2D)
  void* ColorOpacityTextureView = nullptr; // id<MTLTexture>  (alias)
  void* ColorOpacitySampler = nullptr;   // id<MTLSamplerState>
  void* GradientOpacityTexture = nullptr; // id<MTLTexture> (256x1 RGBA8Unorm)
  void* GradientOpacitySampler = nullptr; // id<MTLSamplerState>
  void* DepthStencilState = nullptr;     // id<MTLDepthStencilState>
  void* DepthTextureOcclusion = nullptr; // id<MTLTexture> — scene depth for early ray termination
  void* DepthSampler = nullptr;          // id<MTLSamplerState> — nearest sampler for depth texture

  // Buffers
  void* UniformBuffer = nullptr;         // id<MTLBuffer>
  void* VertexBuffer = nullptr;          // id<MTLBuffer>
  void* IndexBuffer = nullptr;           // id<MTLBuffer>
  void* StagingBuffer = nullptr;         // id<MTLBuffer> (volume upload, kept alive for async blit)
  int IndexCount = 0;

  // Volume state
  double ModelBounds[6] = { 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 };
  double ScalarRange[2] = { 0.0, 1.0 };
  float ScalarNormalizationFactor = 1.0f;
  int VolumeNumComponents = 1;
  int CurrentSampleCount = 0;

  // Adaptive sample distance
  double ReductionFactor = 1.0;
  void ComputeReductionFactor(double allocatedTime);

  // Image-space downsampling (ImageSampleDistance)
  void* ImageSampleColorTexture = nullptr;    // id<MTLTexture> — offscreen color at reduced res
  void* ImageSampleDepthTexture = nullptr;    // id<MTLTexture> — offscreen depth at reduced res
  void* ImageSamplePipeline = nullptr;        // id<MTLRenderPipelineState> — for blit pass
  void* ImageSampleSampler = nullptr;         // id<MTLSamplerState> — linear sampler for blit
  int ImageSampleFBOWidth = 0;
  int ImageSampleFBOHeight = 0;
  bool EnsureImageSampleResources(void* device, int width, int height);
  void ReleaseImageSampleResources();

  // Cache/timestamps
  vtkTimeStamp VolumeUploadTime;
  vtkTimeStamp TransferFunctionUploadTime;
  vtkTimeStamp GradientOpacityUploadTime;
  vtkTimeStamp VertexBufferUploadTime;

  // Helper methods
  bool UpdateVolumeTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol);
  bool UpdateTransferFunctionTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol);
  bool UpdateGradientOpacityTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol);
  bool SetupBuffers(void* mtlDevice, vtkRenderer* ren, vtkVolume* vol, vtkImageData* input);
  bool SetupPipeline(void* mtlDevice, vtkRenderer* ren);

  // Near-plane clipping
  bool IsCameraInside(vtkRenderer* ren, vtkVolume* vol);
  bool CameraWasInsideInLastUpdate = false;

  // Clipping planes — up to 8 arbitrary clipping planes
  void SetClippingPlaneUniforms(void* uniforms, vtkRenderer* ren, vtkVolume* vol);

  // Volume partitioning — splits large volumes into blocks for 3D texture size limits
  struct VolumeBlock
  {
    void* Texture = nullptr; // id<MTLTexture> — 3D sub-texture for this block
    double BoundsMin[3] = {};
    double BoundsMax[3] = {};
    int Dims[3] = {};
    int Extents[6] = {};
    double Center[3] = {}; // world-space center for sorting
  };

  unsigned short Partitions[3] = { 1, 1, 1 };
  std::vector<VolumeBlock> Blocks;
  int SortedBlockOrder[64] = {}; // indices into Blocks, sorted back-to-front (max 64 blocks)

  void ClearBlocks();
  void SortBlocksBackToFront(vtkRenderer* ren, vtkVolume* vol);
  bool UpdateBlockTextures(void* mtlDevice, void* mtlQueue, vtkImageData* input,
    vtkDataArray* scalars, int numComponents);
};

VTK_ABI_NAMESPACE_END
#endif
