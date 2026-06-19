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
  wgpu::BindGroup BindGroup = nullptr;
  
  // Buffers
  wgpu::Buffer UniformBuffer = nullptr;
  wgpu::Buffer VertexBuffer = nullptr;
  wgpu::Buffer IndexBuffer = nullptr;
  int IndexCount = 0;

  // Textures and samplers
  wgpu::Texture VolumeTexture = nullptr;
  wgpu::TextureView VolumeTextureView = nullptr;
  // Note: no VolumeSampler — the volume texture uses textureLoad() in the shader
  // (R32Float is UnfilterableFloat; a filtering sampler would require float32-filterable).

  wgpu::Texture ColorOpacityTexture = nullptr;
  wgpu::TextureView ColorOpacityTextureView = nullptr;
  wgpu::Sampler ColorOpacitySampler = nullptr;

  // Cache/timestamps
  vtkTimeStamp VolumeUploadTime;
  vtkTimeStamp TransferFunctionUploadTime;

  // Helper methods
  bool UpdateVolumeTexture(wgpu::Device device, wgpu::Queue queue, vtkVolume* vol);
  bool UpdateTransferFunctionTexture(wgpu::Device device, wgpu::Queue queue, vtkVolume* vol);
  bool SetupBuffers(wgpu::Device device, vtkVolume* vol);
  bool SetupPipeline(wgpu::Device device, wgpu::RenderPassEncoder renderPass, vtkRenderer* ren);
};

VTK_ABI_NAMESPACE_END
#endif
