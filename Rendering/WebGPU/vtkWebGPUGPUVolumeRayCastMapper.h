// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkWebGPUGPUVolumeRayCastMapper
 * @brief   WebGPU implementation of volume rendering through ray-casting.
 */

#ifndef vtkWebGPUGPUVolumeRayCastMapper_h
#define vtkWebGPUGPUVolumeRayCastMapper_h

#include "vtkGPUVolumeRayCastMapper.h"
#include "vtkRenderingWebGPUModule.h" // For export macro
#include "vtkTimeStamp.h"             // For time stamp
#include "vtkWrappingHints.h"         // For VTK_MARSHALAUTO
#include "vtk_wgpu.h"                 // For WebGPU

class vtkImageData;

VTK_ABI_NAMESPACE_BEGIN

class VTKRENDERINGWEBGPU_EXPORT VTK_MARSHALAUTO vtkWebGPUGPUVolumeRayCastMapper
  : public vtkGPUVolumeRayCastMapper
{
public:
  static vtkWebGPUGPUVolumeRayCastMapper* New();
  vtkTypeMacro(vtkWebGPUGPUVolumeRayCastMapper, vtkGPUVolumeRayCastMapper);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /**
   * Rendering volume on WebGPU
   */
  void GPURender(vtkRenderer* ren, vtkVolume* vol) override;

  /**
   * Release WebGPU resources
   */
  void ReleaseGraphicsResources(vtkWindow* window) override;

  /**
   * Get the reduction ratio (used for level-of-detail / interactive rendering)
   */
  void GetReductionRatio(double ratio[3]) override;

  /**
   * Pre-render method called by vtkGPUVolumeRayCastMapper
   */
  void PreRender(vtkRenderer* ren, vtkVolume* vol, double datasetBounds[6],
    double scalarRange[2], int noOfComponents, unsigned int numberOfLevels) override;

  /**
   * Render a block of volume data
   */
  void RenderBlock(vtkRenderer* ren, vtkVolume* vol, unsigned int level) override;

  /**
   * Post-render method called by vtkGPUVolumeRayCastMapper
   */
  void PostRender(vtkRenderer* ren, int numberOfScalarComponents) override;

protected:
  vtkWebGPUGPUVolumeRayCastMapper();
  ~vtkWebGPUGPUVolumeRayCastMapper() override;

private:
  vtkWebGPUGPUVolumeRayCastMapper(const vtkWebGPUGPUVolumeRayCastMapper&) = delete;
  void operator=(const vtkWebGPUGPUVolumeRayCastMapper&) = delete;

  // WebGPU pipeline objects
  wgpu::RenderPipeline Pipeline = nullptr;
  wgpu::BindGroupLayout BindGroupLayout = nullptr;
  wgpu::PipelineLayout PipelineLayout = nullptr;

  // Triple-buffered uniform buffer and bind groups to avoid GPU-CPU race
  // condition on Mailbox present mode: while the GPU executes frame N's
  // command buffer, the CPU can write frame N+1's uniforms into a
  // different buffer slot, preventing data corruption.
  static constexpr int NumUniformBuffers = 3;
  wgpu::Buffer UniformBuffers[NumUniformBuffers] = {};
  wgpu::BindGroup BindGroups[NumUniformBuffers] = {};
  int UniformBufferIndex = 0;
  int ActiveUniformSlot = 0;

  // Buffers
  wgpu::Buffer VertexBuffer = nullptr;
  wgpu::Buffer IndexBuffer = nullptr;
  int IndexCount = 0;

  // Textures and samplers
  wgpu::Texture VolumeTexture = nullptr;
  wgpu::TextureView VolumeTextureView = nullptr;
  wgpu::Sampler VolumeSampler = nullptr;

  wgpu::Texture ColorOpacityTexture = nullptr;
  wgpu::TextureView ColorOpacityTextureView = nullptr;
  wgpu::Sampler ColorOpacitySampler = nullptr;

  // Volume state
  double ModelBounds[6] = { 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 };
  float ScalarNormalizationFactor = 1.0f;
  int VolumeNumComponents = 1;

  // Cached device handle (set during SyncDeviceResources, used by CreateBindGroup)
  wgpu::Device CachedDevice = nullptr;

  // Cache/timestamps
  vtkTimeStamp VolumeUploadTime;
  vtkTimeStamp TransferFunctionUploadTime;

  // Helper methods
  bool UpdateVolumeTexture(wgpu::Device device, wgpu::Queue queue, vtkVolume* vol);
  bool UpdateTransferFunctionTexture(wgpu::Device device, wgpu::Queue queue, vtkVolume* vol);
  bool SetupBuffers(wgpu::Device device, vtkVolume* vol, vtkImageData* input);
  bool SetupPipeline(wgpu::Device device, vtkRenderer* ren);
  bool CreateBindGroup();
};

VTK_ABI_NAMESPACE_END
#endif
