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
  void* DepthStencilState = nullptr;     // id<MTLDepthStencilState>

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
  vtkTimeStamp VertexBufferUploadTime;

  // Helper methods
  bool UpdateVolumeTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol);
  bool UpdateTransferFunctionTexture(void* mtlDevice, void* mtlQueue, vtkVolume* vol);
  bool SetupBuffers(void* mtlDevice, vtkVolume* vol, vtkImageData* input);
  bool SetupPipeline(void* mtlDevice, vtkRenderer* ren);
};

VTK_ABI_NAMESPACE_END
#endif
