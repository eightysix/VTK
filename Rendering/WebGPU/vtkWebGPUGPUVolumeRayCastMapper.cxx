// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
#include "vtkWebGPUGPUVolumeRayCastMapper.h"

#include "vtkColorTransferFunction.h"
#include "vtkImageData.h"
#include "vtkObjectFactory.h"
#include "vtkPiecewiseFunction.h"
#include "vtkPointData.h"
#include "vtkRenderer.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkWebGPURenderWindow.h"
#include "vtkWebGPURenderer.h"
#include "vtkCamera.h"
#include "vtkMatrix4x4.h"
#include "vtkVolumeRayCastMapperShader.h"

#include <algorithm>
#include <vector>

struct VolumeMapperUniforms
{
  float WorldToVolumeMatrix[16];
  float VolumeToWorldMatrix[16];
  float VolumeBoundsMin[4];
  float VolumeBoundsMax[4];
  float CameraVolumePos[4];
  float SampleDistance;
  float ScalarMin;   // scalar data range – used by the shader to normalise
  float ScalarMax;   // raw voxel values into [0,1] for the TF texture lookup
  float Padding;     // keep 16-byte alignment
};


VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkWebGPUGPUVolumeRayCastMapper);

//------------------------------------------------------------------------------
vtkWebGPUGPUVolumeRayCastMapper::vtkWebGPUGPUVolumeRayCastMapper()
{
  this->SampleDistance = 1.0f;
}

//------------------------------------------------------------------------------
vtkWebGPUGPUVolumeRayCastMapper::~vtkWebGPUGPUVolumeRayCastMapper()
{
  this->ReleaseGraphicsResources(nullptr);
}

//------------------------------------------------------------------------------
void vtkWebGPUGPUVolumeRayCastMapper::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//------------------------------------------------------------------------------
void vtkWebGPUGPUVolumeRayCastMapper::ReleaseGraphicsResources(vtkWindow* vtkNotUsed(window))
{
  this->Pipeline = nullptr;
  this->BindGroupLayout = nullptr;
  this->BindGroup = nullptr;
  this->UniformBuffer = nullptr;
  this->VertexBuffer = nullptr;
  this->IndexBuffer = nullptr;
  this->VolumeTexture = nullptr;
  this->VolumeTextureView = nullptr;
  this->ColorOpacityTexture = nullptr;
  this->ColorOpacityTextureView = nullptr;
  this->ColorOpacitySampler = nullptr;
}

//------------------------------------------------------------------------------
void vtkWebGPUGPUVolumeRayCastMapper::GetReductionRatio(double ratio[3])
{
  ratio[0] = 1.0;
  ratio[1] = 1.0;
  ratio[2] = 1.0;
}

//------------------------------------------------------------------------------
void vtkWebGPUGPUVolumeRayCastMapper::PreRender(vtkRenderer* vtkNotUsed(ren),
  vtkVolume* vtkNotUsed(vol), double vtkNotUsed(datasetBounds)[6],
  double vtkNotUsed(scalarRange)[2], int vtkNotUsed(noOfComponents),
  unsigned int vtkNotUsed(numberOfLevels))
{
}

//------------------------------------------------------------------------------
void vtkWebGPUGPUVolumeRayCastMapper::RenderBlock(vtkRenderer* vtkNotUsed(ren),
  vtkVolume* vtkNotUsed(vol), unsigned int vtkNotUsed(level))
{
}

//------------------------------------------------------------------------------
void vtkWebGPUGPUVolumeRayCastMapper::PostRender(vtkRenderer* vtkNotUsed(ren),
  int vtkNotUsed(numberOfScalarComponents))
{
}

//------------------------------------------------------------------------------
bool vtkWebGPUGPUVolumeRayCastMapper::UpdateVolumeTexture(wgpu::Device device, wgpu::Queue queue, vtkVolume* vol)
{
  vtkImageData* input = vtkImageData::SafeDownCast(this->GetInput());
  if (!input)
  {
    return false;
  }

  vtkDataArray* scalars = input->GetPointData()->GetScalars();
  if (!scalars)
  {
    return false;
  }

  bool doReload = (this->VolumeTexture == nullptr);
  doReload |= (input->GetMTime() > this->VolumeUploadTime.GetMTime());

  if (doReload)
  {
    int dims[3];
    input->GetDimensions(dims);

    wgpu::Extent3D extent = {
      static_cast<uint32_t>(dims[0]),
      static_cast<uint32_t>(dims[1]),
      static_cast<uint32_t>(dims[2])
    };

    wgpu::TextureFormat format = wgpu::TextureFormat::R8Unorm;
    if (scalars->GetDataType() == VTK_FLOAT)
    {
      format = wgpu::TextureFormat::R32Float;
    }
    else if (scalars->GetDataType() != VTK_UNSIGNED_CHAR)
    {
      vtkWarningMacro("Only unsigned char and float volumes are fully supported in WebGPU mapper currently.");
    }

    wgpu::TextureDescriptor texDesc = {};
    texDesc.label = "vtkWebGPUGPUVolumeRayCastMapper::VolumeTexture";
    texDesc.dimension = wgpu::TextureDimension::e3D;
    texDesc.size = extent;
    texDesc.format = format;
    texDesc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;

    this->VolumeTexture = device.CreateTexture(&texDesc);
    this->VolumeTextureView = this->VolumeTexture.CreateView();

    wgpu::TexelCopyTextureInfo destination = {};
    destination.texture = this->VolumeTexture;
    destination.mipLevel = 0;
    destination.origin = {0, 0, 0};
    destination.aspect = wgpu::TextureAspect::All;

    wgpu::TexelCopyBufferLayout dataLayout = {};
    dataLayout.offset = 0;
    dataLayout.bytesPerRow = dims[0] * scalars->GetDataTypeSize() * scalars->GetNumberOfComponents();
    dataLayout.rowsPerImage = dims[1];

    wgpu::Extent3D writeSize = extent;

    queue.WriteTexture(&destination, scalars->GetVoidPointer(0), scalars->GetDataSize() * scalars->GetDataTypeSize(), &dataLayout, &writeSize);

    this->VolumeUploadTime.Modified();
    this->BindGroup = nullptr;
  }

  return this->VolumeTexture != nullptr;
}

//------------------------------------------------------------------------------
bool vtkWebGPUGPUVolumeRayCastMapper::UpdateTransferFunctionTexture(wgpu::Device device, wgpu::Queue queue, vtkVolume* vol)
{
  vtkVolumeProperty* property = vol->GetProperty();
  if (!property)
  {
    return false;
  }

  vtkColorTransferFunction* colorFunc = property->GetRGBTransferFunction();
  vtkPiecewiseFunction* opacityFunc = property->GetScalarOpacity();
  if (!colorFunc || !opacityFunc)
  {
    return false;
  }

  bool doReload = (this->ColorOpacityTexture == nullptr);
  doReload |= (colorFunc->GetMTime() > this->TransferFunctionUploadTime.GetMTime());
  doReload |= (opacityFunc->GetMTime() > this->TransferFunctionUploadTime.GetMTime());

  if (doReload)
  {
    double range[2];
    vtkImageData* input = vtkImageData::SafeDownCast(this->GetInput());
    if (input && input->GetPointData()->GetScalars())
    {
      input->GetPointData()->GetScalars()->GetRange(range);
    }
    else
    {
      range[0] = 0.0;
      range[1] = 255.0;
    }

    unsigned char tfData[256 * 4];
    for (int i = 0; i < 256; ++i)
    {
      double val = range[0] + (range[1] - range[0]) * (i / 255.0);
      double rgb[3];
      colorFunc->GetColor(val, rgb);
      double opacity = opacityFunc->GetValue(val);
      tfData[i * 4 + 0] = static_cast<unsigned char>(rgb[0] * 255.0);
      tfData[i * 4 + 1] = static_cast<unsigned char>(rgb[1] * 255.0);
      tfData[i * 4 + 2] = static_cast<unsigned char>(rgb[2] * 255.0);
      tfData[i * 4 + 3] = static_cast<unsigned char>(opacity * 255.0);
    }

    wgpu::TextureDescriptor tfDesc = {};
    tfDesc.label = "vtkWebGPUGPUVolumeRayCastMapper::ColorOpacityTexture";
    tfDesc.dimension = wgpu::TextureDimension::e2D;
    tfDesc.size = { 256, 1, 1 };
    tfDesc.format = wgpu::TextureFormat::RGBA8Unorm;
    tfDesc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;

    this->ColorOpacityTexture = device.CreateTexture(&tfDesc);
    this->ColorOpacityTextureView = this->ColorOpacityTexture.CreateView();

    wgpu::TexelCopyTextureInfo destination = {};
    destination.texture = this->ColorOpacityTexture;
    destination.mipLevel = 0;
    destination.origin = {0, 0, 0};
    destination.aspect = wgpu::TextureAspect::All;

    wgpu::TexelCopyBufferLayout dataLayout = {};
    dataLayout.offset = 0;
    dataLayout.bytesPerRow = 256 * 4;
    dataLayout.rowsPerImage = 1;

    wgpu::Extent3D writeSize = { 256, 1, 1 };
    queue.WriteTexture(&destination, tfData, sizeof(tfData), &dataLayout, &writeSize);

    this->TransferFunctionUploadTime.Modified();
    this->BindGroup = nullptr;
  }

  return this->ColorOpacityTexture != nullptr;
}

//------------------------------------------------------------------------------
bool vtkWebGPUGPUVolumeRayCastMapper::SetupBuffers(wgpu::Device device, vtkVolume* vol)
{
  if (!this->UniformBuffer)
  {
    wgpu::BufferDescriptor bufDesc = {};
    bufDesc.label = "vtkWebGPUGPUVolumeRayCastMapper::UniformBuffer";
    bufDesc.size = sizeof(VolumeMapperUniforms);
    bufDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    this->UniformBuffer = device.CreateBuffer(&bufDesc);
  }

  double bounds[6];
  vol->GetBounds(bounds);

  if (!this->VertexBuffer || this->GetMTime() > this->VolumeUploadTime)
  {
    float boundsMin[3] = { static_cast<float>(bounds[0]), static_cast<float>(bounds[2]), static_cast<float>(bounds[4]) };
    float boundsMax[3] = { static_cast<float>(bounds[1]), static_cast<float>(bounds[3]), static_cast<float>(bounds[5]) };

    float vertices[] = {
      boundsMin[0], boundsMin[1], boundsMin[2],
      boundsMax[0], boundsMin[1], boundsMin[2],
      boundsMax[0], boundsMax[1], boundsMin[2],
      boundsMin[0], boundsMax[1], boundsMin[2],
      boundsMin[0], boundsMin[1], boundsMax[2],
      boundsMax[0], boundsMin[1], boundsMax[2],
      boundsMax[0], boundsMax[1], boundsMax[2],
      boundsMin[0], boundsMax[1], boundsMax[2],
    };

    unsigned int indices[] = {
      0, 2, 1,  0, 3, 2,
      4, 5, 6,  4, 6, 7,
      0, 7, 3,  0, 4, 7,
      1, 2, 6,  1, 6, 5,
      3, 6, 2,  3, 7, 6,
      0, 1, 5,  0, 5, 4,
    };

    this->IndexCount = sizeof(indices) / sizeof(unsigned int);

    {
      wgpu::BufferDescriptor bufDesc = {};
      bufDesc.label = "vtkWebGPUGPUVolumeRayCastMapper::VertexBuffer";
      bufDesc.size = sizeof(vertices);
      bufDesc.usage = wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst;
      this->VertexBuffer = device.CreateBuffer(&bufDesc);
      
      device.GetQueue().WriteBuffer(this->VertexBuffer, 0, vertices, sizeof(vertices));
    }

    {
      wgpu::BufferDescriptor bufDesc = {};
      bufDesc.label = "vtkWebGPUGPUVolumeRayCastMapper::IndexBuffer";
      bufDesc.size = sizeof(indices);
      bufDesc.usage = wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst;
      this->IndexBuffer = device.CreateBuffer(&bufDesc);
      
      device.GetQueue().WriteBuffer(this->IndexBuffer, 0, indices, sizeof(indices));
    }
  }

  return this->VertexBuffer && this->IndexBuffer && this->UniformBuffer;
}

//------------------------------------------------------------------------------
bool vtkWebGPUGPUVolumeRayCastMapper::SetupPipeline(wgpu::Device device, wgpu::RenderPassEncoder renderPass, vtkRenderer* ren)
{
  if (this->Pipeline && this->BindGroup)
  {
    return true;
  }

  wgpu::ShaderModule shaderModule = nullptr;
  {
    wgpu::ShaderModuleWGSLDescriptor wgslDesc = {};
    wgslDesc.code = vtkVolumeRayCastMapperShader;

    wgpu::ShaderModuleDescriptor shaderDesc = {};
    shaderDesc.nextInChain = &wgslDesc;
    shaderDesc.label = "vtkWebGPUGPUVolumeRayCastMapper::ShaderModule";
    shaderModule = device.CreateShaderModule(&shaderDesc);
  }

  if (!this->ColorOpacitySampler)
  {
    wgpu::SamplerDescriptor samplerDesc = {};
    samplerDesc.addressModeU = wgpu::AddressMode::ClampToEdge;
    samplerDesc.addressModeV = wgpu::AddressMode::ClampToEdge;
    samplerDesc.magFilter = wgpu::FilterMode::Linear;
    samplerDesc.minFilter = wgpu::FilterMode::Linear;
    this->ColorOpacitySampler = device.CreateSampler(&samplerDesc);
  }

  {
    // Binding layout must match the shader:
    //  0 - uniform buffer (vertex + fragment)
    //  1 - volumeTexture: texture_3d<f32> accessed via textureLoad (UnfilterableFloat,
    //      required for R32Float which lacks the float32-filterable feature)
    //  2 - transferFunctionTexture: texture_2d<f32>, RGBA8Unorm -> filterable Float
    //  3 - transferFunctionSampler: filtering sampler
    wgpu::BindGroupLayoutEntry entries[4] = {};

    entries[0].binding = 0;
    entries[0].visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    entries[0].buffer.type = wgpu::BufferBindingType::Uniform;

    entries[1].binding = 1;
    entries[1].visibility = wgpu::ShaderStage::Fragment;
    entries[1].texture.sampleType = wgpu::TextureSampleType::UnfilterableFloat;
    entries[1].texture.viewDimension = wgpu::TextureViewDimension::e3D;

    entries[2].binding = 2;
    entries[2].visibility = wgpu::ShaderStage::Fragment;
    entries[2].texture.sampleType = wgpu::TextureSampleType::Float;
    entries[2].texture.viewDimension = wgpu::TextureViewDimension::e2D;

    entries[3].binding = 3;
    entries[3].visibility = wgpu::ShaderStage::Fragment;
    entries[3].sampler.type = wgpu::SamplerBindingType::Filtering;

    wgpu::BindGroupLayoutDescriptor bglDesc = {};
    bglDesc.label = "vtkWebGPUGPUVolumeRayCastMapper::BindGroupLayout";
    bglDesc.entryCount = 4;
    bglDesc.entries = entries;
    this->BindGroupLayout = device.CreateBindGroupLayout(&bglDesc);
  }

  {
    wgpu::BindGroupEntry bgEntries[4] = {};

    bgEntries[0].binding = 0;
    bgEntries[0].buffer = this->UniformBuffer;
    bgEntries[0].size = sizeof(VolumeMapperUniforms);

    bgEntries[1].binding = 1;
    bgEntries[1].textureView = this->VolumeTextureView;

    bgEntries[2].binding = 2;
    bgEntries[2].textureView = this->ColorOpacityTextureView;

    bgEntries[3].binding = 3;
    bgEntries[3].sampler = this->ColorOpacitySampler;

    wgpu::BindGroupDescriptor bgDesc = {};
    bgDesc.label = "vtkWebGPUGPUVolumeRayCastMapper::BindGroup";
    bgDesc.layout = this->BindGroupLayout;
    bgDesc.entryCount = 4;
    bgDesc.entries = bgEntries;
    this->BindGroup = device.CreateBindGroup(&bgDesc);
  }

  auto* wgpuRenderer = vtkWebGPURenderer::SafeDownCast(ren);
  auto* wgpuRenderWindow = vtkWebGPURenderWindow::SafeDownCast(ren->GetRenderWindow());
  
  std::vector<wgpu::BindGroupLayout> layouts;
  wgpuRenderer->PopulateBindgroupLayouts(layouts);
  layouts.push_back(this->BindGroupLayout);

  wgpu::PipelineLayoutDescriptor layoutDesc = {};
  layoutDesc.label = "vtkWebGPUGPUVolumeRayCastMapper::PipelineLayout";
  layoutDesc.bindGroupLayoutCount = layouts.size();
  layoutDesc.bindGroupLayouts = layouts.data();
  wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&layoutDesc);

  // The renderer's render pass always has two color attachments:
  //   [0] BGRA8Unorm  - main color output
  //   [1] RGBA32Uint  - hardware selector IDs
  // Both must be declared in the pipeline or WebGPU raises an attachment-state error.
  wgpu::BlendState blend = {};
  blend.color.srcFactor = wgpu::BlendFactor::SrcAlpha;
  blend.color.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
  blend.color.operation = wgpu::BlendOperation::Add;
  blend.alpha.srcFactor = wgpu::BlendFactor::One;
  blend.alpha.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
  blend.alpha.operation = wgpu::BlendOperation::Add;

  wgpu::ColorTargetState colorTargets[2] = {};
  colorTargets[0].format = wgpuRenderWindow->GetPreferredSurfaceTextureFormat();
  colorTargets[0].blend = &blend;
  colorTargets[1].format = wgpuRenderWindow->GetPreferredSelectorIdsTextureFormat();
  // Selector target: no blending; volume rendering does not participate in picking.
  colorTargets[1].blend = nullptr;
  colorTargets[1].writeMask = wgpu::ColorWriteMask::None;

  wgpu::FragmentState fragmentState = {};
  fragmentState.module = shaderModule;
  fragmentState.entryPoint = "fragmentMain";
  fragmentState.targetCount = 2;
  fragmentState.targets = colorTargets;

  wgpu::VertexAttribute vertAttr = {};
  vertAttr.format = wgpu::VertexFormat::Float32x3;
  vertAttr.offset = 0;
  vertAttr.shaderLocation = 0;

  wgpu::VertexBufferLayout vertexBufferLayout = {};
  vertexBufferLayout.arrayStride = sizeof(float) * 3;
  vertexBufferLayout.stepMode = wgpu::VertexStepMode::Vertex;
  vertexBufferLayout.attributeCount = 1;
  vertexBufferLayout.attributes = &vertAttr;

  wgpu::RenderPipelineDescriptor pipelineDesc = {};
  pipelineDesc.label = "vtkWebGPUGPUVolumeRayCastMapper::RenderPipeline";
  pipelineDesc.layout = pipelineLayout;
  pipelineDesc.vertex.module = shaderModule;
  pipelineDesc.vertex.entryPoint = "vertexMain";
  pipelineDesc.vertex.bufferCount = 1;
  pipelineDesc.vertex.buffers = &vertexBufferLayout;
  pipelineDesc.fragment = &fragmentState;

  pipelineDesc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
  pipelineDesc.primitive.frontFace = wgpu::FrontFace::CCW;
  pipelineDesc.primitive.cullMode = wgpu::CullMode::None;

  wgpu::DepthStencilState depthStencil = {};
  depthStencil.format = wgpuRenderWindow->GetDepthStencilFormat();
  depthStencil.depthWriteEnabled = false;
  depthStencil.depthCompare = wgpu::CompareFunction::LessEqual;
  pipelineDesc.depthStencil = &depthStencil;

  this->Pipeline = device.CreateRenderPipeline(&pipelineDesc);

  return this->Pipeline != nullptr;
}

//------------------------------------------------------------------------------
void vtkWebGPUGPUVolumeRayCastMapper::GPURender(vtkRenderer* ren, vtkVolume* vol)
{
  auto* wgpuRenderer = vtkWebGPURenderer::SafeDownCast(ren);
  auto* wgpuRenderWindow = vtkWebGPURenderWindow::SafeDownCast(ren->GetRenderWindow());
  if (!wgpuRenderer || !wgpuRenderWindow)
  {
    return;
  }

  wgpu::Device device = wgpuRenderWindow->GetDevice();
  wgpu::Queue queue = device.GetQueue();

  switch (wgpuRenderer->GetRenderStage())
  {
    case vtkWebGPURenderer::RenderStageEnum::SyncDeviceResources:
    {
      if (!this->UpdateVolumeTexture(device, queue, vol))
      {
        return;
      }
      if (!this->UpdateTransferFunctionTexture(device, queue, vol))
      {
        return;
      }
      if (!this->SetupBuffers(device, vol))
      {
        return;
      }
      if (!this->SetupPipeline(device, nullptr, ren))
      {
        return;
      }
      break;
    }
    case vtkWebGPURenderer::RenderStageEnum::RecordingCommands:
    {
      wgpu::RenderPassEncoder renderPass = wgpuRenderer->GetRenderPassEncoder();
      if (!renderPass)
      {
        return;
      }

      if (!this->SetupPipeline(device, renderPass, ren))
      {
        return;
      }

      VolumeMapperUniforms uniforms;
      
      vtkNew<vtkMatrix4x4> modelMatrix;
      vol->GetModelToWorldMatrix(modelMatrix);
      
      vtkNew<vtkMatrix4x4> invModelMatrix;
      vtkMatrix4x4::Invert(modelMatrix, invModelMatrix);

      for (int r = 0; r < 4; ++r)
      {
        for (int c = 0; c < 4; ++c)
        {
          uniforms.VolumeToWorldMatrix[c * 4 + r] = modelMatrix->GetElement(r, c);
          uniforms.WorldToVolumeMatrix[c * 4 + r] = invModelMatrix->GetElement(r, c);
        }
      }

      double bounds[6];
      vol->GetBounds(bounds);

      // Bounds in world/model space (passed to shader for bounding-box geometry).
      uniforms.VolumeBoundsMin[0] = static_cast<float>(bounds[0]);
      uniforms.VolumeBoundsMin[1] = static_cast<float>(bounds[2]);
      uniforms.VolumeBoundsMin[2] = static_cast<float>(bounds[4]);
      uniforms.VolumeBoundsMin[3] = 1.0f;

      uniforms.VolumeBoundsMax[0] = static_cast<float>(bounds[1]);
      uniforms.VolumeBoundsMax[1] = static_cast<float>(bounds[3]);
      uniforms.VolumeBoundsMax[2] = static_cast<float>(bounds[5]);
      uniforms.VolumeBoundsMax[3] = 1.0f;

      // The vertex shader maps vertex positions to localPos in [0,1]³ texture
      // space.  The camera position must be expressed in the same space so that
      // the ray direction is consistent.
      double boundsSize[3] = {
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4]
      };
      // Guard against degenerate (zero-size) bounds.
      for (int k = 0; k < 3; ++k)
      {
        if (boundsSize[k] < 1e-10) boundsSize[k] = 1.0;
      }

      double* camPosWorld = ren->GetActiveCamera()->GetPosition();
      double camPosVolume[4] = { camPosWorld[0], camPosWorld[1], camPosWorld[2], 1.0 };
      invModelMatrix->MultiplyPoint(camPosVolume, camPosVolume);

      // Normalise camera position into [0,1]³ texture space.
      uniforms.CameraVolumePos[0] =
        static_cast<float>((camPosVolume[0] - bounds[0]) / boundsSize[0]);
      uniforms.CameraVolumePos[1] =
        static_cast<float>((camPosVolume[1] - bounds[2]) / boundsSize[1]);
      uniforms.CameraVolumePos[2] =
        static_cast<float>((camPosVolume[2] - bounds[4]) / boundsSize[2]);
      uniforms.CameraVolumePos[3] = 1.0f;

      // Normalise step size into [0,1]³ space.  Use the longest axis so the
      // step is isotropic in normalised coordinates.
      double maxBoundsSize = std::max({ boundsSize[0], boundsSize[1], boundsSize[2] });
      uniforms.SampleDistance =
        static_cast<float>(this->GetSampleDistance() / maxBoundsSize);

      // Scalar data range – needed by the shader to normalise raw voxel values
      // into [0,1] before indexing the transfer-function texture.
      {
        vtkImageData* inputImg = vtkImageData::SafeDownCast(this->GetInput());
        double scalarRange[2] = { 0.0, 1.0 };
        if (inputImg && inputImg->GetPointData()->GetScalars())
        {
          inputImg->GetPointData()->GetScalars()->GetRange(scalarRange);
        }
        uniforms.ScalarMin = static_cast<float>(scalarRange[0]);
        uniforms.ScalarMax = static_cast<float>(
          scalarRange[1] > scalarRange[0] ? scalarRange[1] : scalarRange[0] + 1.0);
      }

      queue.WriteBuffer(this->UniformBuffer, 0, &uniforms, sizeof(uniforms));

      renderPass.SetPipeline(this->Pipeline);
      renderPass.SetBindGroup(0, wgpuRenderer->GetSceneBindGroup());
      renderPass.SetBindGroup(1, this->BindGroup);

      renderPass.SetVertexBuffer(0, this->VertexBuffer);
      renderPass.SetIndexBuffer(this->IndexBuffer, wgpu::IndexFormat::Uint32);
      renderPass.DrawIndexed(this->IndexCount, 1, 0, 0, 0);
      break;
    }
    default:
      break;
  }
}

VTK_ABI_NAMESPACE_END
