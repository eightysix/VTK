// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalPolyDataMapper.h"

#include "vtkMetalRenderWindow.h"
#include "vtkMetalRenderer.h"
#include "vtkMetalCamera.h"
#include "vtkMetalShaders.h"
#include "vtkObjectFactory.h"
#include "vtkPolyData.h"
#include "vtkCellArray.h"
#include "vtkCellData.h"
#include "vtkPointData.h"
#include "vtkImageData.h"
#include "vtkProperty.h"
#include "vtkActor.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkInformation.h"
#include "vtkExecutive.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkUnsignedCharArray.h"
#include "vtkLight.h"
#include "vtkLightCollection.h"
#include "vtkCamera.h"
#include "vtkMatrix4x4.h"
#include "vtkTransform.h"
#include "vtkMath.h"

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <vector>
#include <unordered_map>

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalPolyDataMapper);

//------------------------------------------------------------------------------
struct vtkMetalPolyDataMapper::vtkMetalPolyDataMapperInternals
{
  id<MTLBuffer> VertexPositionBuffer = nil;
  id<MTLBuffer> VertexNormalBuffer = nil;
  id<MTLBuffer> IndexBuffer = nil;
  id<MTLBuffer> LineIndexBuffer = nil;

  id<MTLRenderPipelineState> TrianglePipeline = nil;
  id<MTLRenderPipelineState> LinePipeline = nil;

  id<MTLBuffer> SceneUniformBuffer = nil;
  id<MTLBuffer> MaterialUniformBuffer = nil;
  id<MTLBuffer> LightUniformBuffer = nil;

  vtkIdType TriangleVertexCount = 0;
  vtkIdType TriangleIndexCount = 0;
  vtkIdType LineIndexCount = 0;
  bool HasTriangles = false;
  bool HasLines = false;

  vtkIdType CachedInputMTime = 0;

  void ReleaseBuffers()
  {
    VertexPositionBuffer = nil;
    VertexNormalBuffer = nil;
    IndexBuffer = nil;
    LineIndexBuffer = nil;
    TrianglePipeline = nil;
    LinePipeline = nil;
    SceneUniformBuffer = nil;
    MaterialUniformBuffer = nil;
    LightUniformBuffer = nil;
    TriangleVertexCount = 0;
    TriangleIndexCount = 0;
    LineIndexCount = 0;
    HasTriangles = false;
    HasLines = false;
    CachedInputMTime = 0;
  }
};

//------------------------------------------------------------------------------
vtkMetalPolyDataMapper::vtkMetalPolyDataMapper()
  : Internals(new vtkMetalPolyDataMapperInternals)
{
}

vtkMetalPolyDataMapper::~vtkMetalPolyDataMapper() = default;

void vtkMetalPolyDataMapper::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

vtkMetalPolyDataMapper::MapperHashType vtkMetalPolyDataMapper::GenerateHash(vtkPolyData* polydata)
{
  if (!polydata)
  {
    return 0;
  }
  return static_cast<MapperHashType>(polydata->GetMTime());
}

void vtkMetalPolyDataMapper::ReleaseGraphicsResources(vtkWindow*)
{
  this->Internals->ReleaseBuffers();
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::RenderPiece(vtkRenderer* ren, vtkActor* act)
{
  vtkMetalRenderWindow* renWin = vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());
  if (!renWin || !renWin->GetMetalDevice())
  {
    return;
  }

  vtkPolyData* input = this->GetInput();
  if (!input)
  {
    return;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)renWin->GetMetalDevice();

    vtkIdType currentMTime = input->GetMTime();
    if (currentMTime != this->Internals->CachedInputMTime)
    {
      this->Internals->ReleaseBuffers();
      this->Internals->CachedInputMTime = currentMTime;
      this->BuildGeometryBuffers((__bridge void*)device, input);
    }

    if (this->Internals->TriangleVertexCount == 0 && this->Internals->LineIndexCount == 0)
    {
      return;
    }

    this->EnsurePipelineStates((__bridge void*)device);

    // Use the encoder already created by vtkMetalRenderer::DeviceRender().
    // Do NOT create a new render pass, command buffer, or drawable here.
    id<MTLRenderCommandEncoder> encoder =
      (__bridge id<MTLRenderCommandEncoder>)renWin->GetCurrentRenderCommandEncoder();
    if (!encoder)
    {
      vtkErrorMacro(<< "No active render command encoder. "
                    << "RenderPiece must be called within DeviceRender.");
      return;
    }

    vtkMetalCamera* metalCamera = vtkMetalCamera::SafeDownCast(ren->GetActiveCamera());
    if (metalCamera)
    {
      metalCamera->Render(ren);

      if (!this->Internals->SceneUniformBuffer)
      {
        this->Internals->SceneUniformBuffer = [device
          newBufferWithLength:vtkMetalCamera::GetSceneTransformsSize()
                     options:MTLResourceStorageModeShared];
      }
      memcpy([this->Internals->SceneUniformBuffer contents],
             metalCamera->GetCachedSceneTransforms(),
             vtkMetalCamera::GetSceneTransformsSize());
    }

    this->UpdateMaterialUniforms((__bridge void*)device, act);
    this->UpdateLightUniforms((__bridge void*)device, ren);

    if (this->Internals->HasTriangles && this->Internals->TrianglePipeline)
    {
      [encoder setRenderPipelineState:this->Internals->TrianglePipeline];
      [encoder setVertexBuffer:this->Internals->VertexPositionBuffer offset:0 atIndex:0];
      if (this->Internals->VertexNormalBuffer)
      {
        [encoder setVertexBuffer:this->Internals->VertexNormalBuffer offset:0 atIndex:1];
      }

      if (this->Internals->MaterialUniformBuffer)
      {
        [encoder setFragmentBuffer:this->Internals->MaterialUniformBuffer offset:0 atIndex:0];
      }
      if (this->Internals->LightUniformBuffer)
      {
        [encoder setFragmentBuffer:this->Internals->LightUniformBuffer offset:0 atIndex:1];
      }
      if (this->Internals->SceneUniformBuffer)
      {
        [encoder setVertexBuffer:this->Internals->SceneUniformBuffer offset:0 atIndex:2];
      }

      if (this->Internals->IndexBuffer)
      {
        [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                            indexCount:this->Internals->TriangleIndexCount
                             indexType:MTLIndexTypeUInt32
                           indexBuffer:this->Internals->IndexBuffer
                     indexBufferOffset:0];
      }
      else
      {
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0
                  vertexCount:this->Internals->TriangleVertexCount];
      }
    }

    if (this->Internals->HasLines && this->Internals->LinePipeline &&
        this->Internals->LineIndexBuffer)
    {
      [encoder setRenderPipelineState:this->Internals->LinePipeline];
      [encoder setVertexBuffer:this->Internals->VertexPositionBuffer offset:0 atIndex:0];
      if (this->Internals->VertexNormalBuffer)
      {
        [encoder setVertexBuffer:this->Internals->VertexNormalBuffer offset:0 atIndex:1];
      }

      if (this->Internals->MaterialUniformBuffer)
      {
        [encoder setFragmentBuffer:this->Internals->MaterialUniformBuffer offset:0 atIndex:0];
      }
      if (this->Internals->LightUniformBuffer)
      {
        [encoder setFragmentBuffer:this->Internals->LightUniformBuffer offset:0 atIndex:1];
      }
      if (this->Internals->SceneUniformBuffer)
      {
        [encoder setVertexBuffer:this->Internals->SceneUniformBuffer offset:0 atIndex:2];
      }

      [encoder drawIndexedPrimitives:MTLPrimitiveTypeLine
                          indexCount:this->Internals->LineIndexCount
                           indexType:MTLIndexTypeUInt32
                         indexBuffer:this->Internals->LineIndexBuffer
                   indexBufferOffset:0];
    }
  }
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::BuildGeometryBuffers(void* mtlDevice, vtkPolyData* polydata)
{
  if (!polydata || !mtlDevice)
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<uint32_t> triangleIndices;
  std::vector<uint32_t> lineIndices;

  vtkPointData* pd = polydata->GetPointData();
  vtkFloatArray* normalArray = nullptr;
  if (pd->GetNormals())
  {
    normalArray = vtkFloatArray::SafeDownCast(pd->GetNormals());
  }

  // Process triangles
  vtkCellArray* polys = polydata->GetPolys();
  if (polys && polys->GetNumberOfCells() > 0)
  {
    std::unordered_map<vtkIdType, uint32_t> pointMap;
    uint32_t nextPointId = 0;

    vtkIdType npts;
    const vtkIdType* pts;
    polys->InitTraversal();
    while (polys->GetNextCell(npts, pts) && npts > 0)
    {
      if (npts < 3)
      {
        continue;
      }
      for (vtkIdType i = 1; i < npts - 1; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          vtkIdType corner = (j == 0) ? pts[0] : (j == 1) ? pts[i] : pts[i + 1];
          if (pointMap.find(corner) == pointMap.end())
          {
            pointMap[corner] = nextPointId++;
            double pt[3];
            polydata->GetPoint(corner, pt);
            positions.push_back(static_cast<float>(pt[0]));
            positions.push_back(static_cast<float>(pt[1]));
            positions.push_back(static_cast<float>(pt[2]));
            if (normalArray)
            {
              double n[3];
              normalArray->GetTuple(corner, n);
              normals.push_back(static_cast<float>(n[0]));
              normals.push_back(static_cast<float>(n[1]));
              normals.push_back(static_cast<float>(n[2]));
            }
          }
          triangleIndices.push_back(pointMap[corner]);
        }
      }
    }
    this->Internals->TriangleVertexCount = nextPointId;
    this->Internals->TriangleIndexCount = triangleIndices.size();
    this->Internals->HasTriangles = !triangleIndices.empty();
  }

  // Process lines
  vtkCellArray* lines = polydata->GetLines();
  if (lines && lines->GetNumberOfCells() > 0)
  {
    std::unordered_map<vtkIdType, uint32_t> pointMap;
    uint32_t nextPointId = static_cast<uint32_t>(positions.size() / 3);

    vtkIdType npts;
    const vtkIdType* pts;
    lines->InitTraversal();
    while (lines->GetNextCell(npts, pts) && npts > 0)
    {
      for (vtkIdType i = 0; i < npts; ++i)
      {
        if (pointMap.find(pts[i]) == pointMap.end())
        {
          pointMap[pts[i]] = nextPointId++;
          double pt[3];
          polydata->GetPoint(pts[i], pt);
          positions.push_back(static_cast<float>(pt[0]));
          positions.push_back(static_cast<float>(pt[1]));
          positions.push_back(static_cast<float>(pt[2]));
          if (normalArray)
          {
            double n[3];
            normalArray->GetTuple(pts[i], n);
            normals.push_back(static_cast<float>(n[0]));
            normals.push_back(static_cast<float>(n[1]));
            normals.push_back(static_cast<float>(n[2]));
          }
        }
      }
      for (vtkIdType i = 0; i < npts - 1; ++i)
      {
        lineIndices.push_back(pointMap[pts[i]]);
        lineIndices.push_back(pointMap[pts[i + 1]]);
      }
    }
    this->Internals->LineIndexCount = lineIndices.size();
    this->Internals->HasLines = !lineIndices.empty();
  }

  if (!positions.empty())
  {
    this->Internals->VertexPositionBuffer = [device
      newBufferWithBytes:positions.data()
                 length:positions.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
  }
  if (!normals.empty())
  {
    this->Internals->VertexNormalBuffer = [device
      newBufferWithBytes:normals.data()
                 length:normals.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
  }
  else if (!positions.empty())
  {
    // Vertex descriptor always requires a buffer at index 1 for normals.
    // Fill with a default up-facing normal so the pipeline validates.
    normals.assign(positions.size(), 0.0f);
    for (size_t i = 1; i < normals.size(); i += 3)
    {
      normals[i] = 1.0f;
    }
    this->Internals->VertexNormalBuffer = [device
      newBufferWithBytes:normals.data()
                 length:normals.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
  }
  if (!triangleIndices.empty())
  {
    this->Internals->IndexBuffer = [device
      newBufferWithBytes:triangleIndices.data()
                 length:triangleIndices.size() * sizeof(uint32_t)
                options:MTLResourceStorageModeShared];
  }
  if (!lineIndices.empty())
  {
    this->Internals->LineIndexBuffer = [device
      newBufferWithBytes:lineIndices.data()
                 length:lineIndices.size() * sizeof(uint32_t)
                options:MTLResourceStorageModeShared];
  }
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::EnsurePipelineStates(void* mtlDevice)
{
  if (this->Internals->TrianglePipeline && this->Internals->LinePipeline)
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;

  NSError* error = nil;
  NSString* shaderSource = [NSString stringWithUTF8String:vtkMetalShaders];
  id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
  if (!library)
  {
    vtkErrorMacro(<< "Failed to compile Metal shader: "
                  << [[error localizedDescription] UTF8String]);
    return;
  }

  id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_main"];
  id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragment_main"];

  if (!vertexFunc || !fragmentFunc)
  {
    vtkErrorMacro(<< "Failed to find shader functions");
    return;
  }

  MTLVertexDescriptor* vertexDesc = [[MTLVertexDescriptor alloc] init];
  vertexDesc.attributes[0].format = MTLVertexFormatFloat3;
  vertexDesc.attributes[0].offset = 0;
  vertexDesc.attributes[0].bufferIndex = 0;
  vertexDesc.attributes[1].format = MTLVertexFormatFloat3;
  vertexDesc.attributes[1].offset = 0;
  vertexDesc.attributes[1].bufferIndex = 1;
  vertexDesc.layouts[0].stride = sizeof(float) * 3;
  vertexDesc.layouts[0].stepRate = 1;
  vertexDesc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
  vertexDesc.layouts[1].stride = sizeof(float) * 3;
  vertexDesc.layouts[1].stepRate = 1;
  vertexDesc.layouts[1].stepFunction = MTLVertexStepFunctionPerVertex;

  MTLRenderPipelineDescriptor* pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
  pipelineDesc.vertexFunction = vertexFunc;
  pipelineDesc.fragmentFunction = fragmentFunc;
  pipelineDesc.vertexDescriptor = vertexDesc;
  pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

  // Enable depth testing (matching WebGPU's depthCompare = Less, depthWriteEnabled = true)
  pipelineDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

  // Enable backface culling (matching WebGPU's default behavior)
  pipelineDesc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;

  if (!this->Internals->TrianglePipeline)
  {
    error = nil;
    this->Internals->TrianglePipeline =
      [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    if (!this->Internals->TrianglePipeline)
    {
      vtkErrorMacro(<< "Triangle pipeline: " << [[error localizedDescription] UTF8String]);
    }
  }

  if (!this->Internals->LinePipeline)
  {
    error = nil;
    this->Internals->LinePipeline =
      [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    if (!this->Internals->LinePipeline)
    {
      vtkErrorMacro(<< "Line pipeline: " << [[error localizedDescription] UTF8String]);
    }
  }
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::UpdateMaterialUniforms(void* mtlDevice, vtkActor* actor)
{
  if (!actor || !mtlDevice)
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;
  vtkProperty* prop = actor->GetProperty();

  // Flat layout matching Metal shader's MaterialUniforms byte-for-byte.
  // Metal float4 = 16 bytes aligned to 16.
  float mu[20]; // 4+4+4+4 color/ambient/diffuse/specular + opacity + specPow + 2 pad
  memset(mu, 0, sizeof(mu));

  double rgb[3];
  prop->GetColor(rgb);
  mu[0] = static_cast<float>(rgb[0]);
  mu[1] = static_cast<float>(rgb[1]);
  mu[2] = static_cast<float>(rgb[2]);
  mu[3] = 1.0f;

  mu[7] = static_cast<float>(prop->GetAmbient());  // ambient.w

  mu[11] = static_cast<float>(prop->GetDiffuse()); // diffuse.w

  mu[15] = static_cast<float>(prop->GetSpecular()); // specular.w

  mu[16] = static_cast<float>(prop->GetOpacity());
  mu[17] = static_cast<float>(prop->GetSpecularPower());

  if (!this->Internals->MaterialUniformBuffer)
  {
    this->Internals->MaterialUniformBuffer = [device
      newBufferWithLength:sizeof(mu)
                 options:MTLResourceStorageModeShared];
  }
  memcpy([this->Internals->MaterialUniformBuffer contents], mu, sizeof(mu));
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::UpdateLightUniforms(void* mtlDevice, vtkRenderer* ren)
{
  if (!mtlDevice || !ren)
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;

  struct LightData
  {
    float position[4];
    float direction[4];
    float color[4];
    float attenuation[4];
  };

  struct LightUniforms
  {
    LightData lights[8];
    int lightCount;
    float padding[3];
  };

  LightUniforms lu;
  memset(&lu, 0, sizeof(lu));

  // Get the camera's model-view transform for view-space conversion
  vtkCamera* cam = ren->GetActiveCamera();
  vtkTransform* viewTF = cam ? cam->GetModelViewTransformObject() : nullptr;

  vtkLightCollection* lc = ren->GetLights();
  vtkLight* light;
  int count = 0;

  lc->InitTraversal();
  while ((light = lc->GetNextItem()) && count < 8)
  {
    if (!light->GetSwitch())
    {
      continue;
    }

    LightData& ld = lu.lights[count];

    int lightType = light->GetLightType();
    if (lightType == VTK_LIGHT_TYPE_HEADLIGHT)
    {
      // Headlight: shader uses hardcoded (0,0,-1) direction in view space
      ld.position[3] = 0.0f;
      ld.position[0] = ld.position[1] = ld.position[2] = 0.0f;
    }
    else if (light->GetPositional())
    {
      ld.position[3] = (light->GetConeAngle() < 90.0) ? 3.0f : 2.0f;

      if (viewTF)
      {
        // Transform light position into view space
        double pos[3];
        light->GetPosition(pos);
        double tPos[3];
        viewTF->TransformPoint(pos, tPos);
        ld.position[0] = static_cast<float>(tPos[0]);
        ld.position[1] = static_cast<float>(tPos[1]);
        ld.position[2] = static_cast<float>(tPos[2]);

        // Transform direction into view space
        double* lfp = light->GetTransformedFocalPoint();
        double* lp = light->GetTransformedPosition();
        double lightDir[3];
        vtkMath::Subtract(lfp, lp, lightDir);
        vtkMath::Normalize(lightDir);
        double tDir[3];
        viewTF->TransformNormal(lightDir, tDir);
        ld.direction[0] = static_cast<float>(tDir[0]);
        ld.direction[1] = static_cast<float>(tDir[1]);
        ld.direction[2] = static_cast<float>(tDir[2]);
      }
      else
      {
        double pos[3];
        light->GetPosition(pos);
        ld.position[0] = static_cast<float>(pos[0]);
        ld.position[1] = static_cast<float>(pos[1]);
        ld.position[2] = static_cast<float>(pos[2]);
        ld.direction[0] = ld.direction[1] = 0.0f;
        ld.direction[2] = -1.0f;
      }
    }
    else
    {
      // Directional or camera light — transform direction to view space
      ld.position[3] = 1.0f;
      ld.position[0] = ld.position[1] = ld.position[2] = 0.0f;

      if (viewTF)
      {
        double* lfp = light->GetTransformedFocalPoint();
        double* lp = light->GetTransformedPosition();
        double lightDir[3];
        vtkMath::Subtract(lfp, lp, lightDir);
        vtkMath::Normalize(lightDir);
        double tDir[3];
        viewTF->TransformNormal(lightDir, tDir);
        ld.direction[0] = static_cast<float>(tDir[0]);
        ld.direction[1] = static_cast<float>(tDir[1]);
        ld.direction[2] = static_cast<float>(tDir[2]);
      }
      else
      {
        ld.direction[0] = ld.direction[1] = 0.0f;
        ld.direction[2] = -1.0f;
      }
    }

    ld.direction[3] = static_cast<float>(light->GetConeAngle());

    double diffColor[3];
    light->GetDiffuseColor(diffColor);
    float intensity = static_cast<float>(light->GetIntensity());
    ld.color[0] = static_cast<float>(diffColor[0]) * intensity;
    ld.color[1] = static_cast<float>(diffColor[1]) * intensity;
    ld.color[2] = static_cast<float>(diffColor[2]) * intensity;
    ld.color[3] = 1.0f;

    double attenValues[3];
    light->GetAttenuationValues(attenValues);
    ld.attenuation[0] = static_cast<float>(attenValues[0]);
    ld.attenuation[1] = static_cast<float>(attenValues[1]);
    ld.attenuation[2] = static_cast<float>(attenValues[2]);
    ld.attenuation[3] = static_cast<float>(light->GetExponent());

    count++;
  }

  lu.lightCount = count;

  if (count == 0)
  {
    // Headlight: at camera, type 0
    lu.lights[0].position[0] = 0.0f;
    lu.lights[0].position[1] = 0.0f;
    lu.lights[0].position[2] = 0.0f;
    lu.lights[0].position[3] = 0.0f;
    lu.lights[0].direction[0] = 0.0f;
    lu.lights[0].direction[1] = 0.0f;
    lu.lights[0].direction[2] = -1.0f;
    lu.lights[0].direction[3] = 0.0f;
    lu.lights[0].color[0] = lu.lights[0].color[1] = lu.lights[0].color[2] = 1.0f;
    lu.lights[0].color[3] = 1.0f;
    lu.lights[0].attenuation[0] = 1.0f;
    lu.lights[0].attenuation[1] = 0.0f;
    lu.lights[0].attenuation[2] = 0.0f;
    lu.lights[0].attenuation[3] = 0.0f;
    lu.lightCount = 1;
  }

  if (!this->Internals->LightUniformBuffer)
  {
    this->Internals->LightUniformBuffer = [device
      newBufferWithLength:sizeof(LightUniforms)
                 options:MTLResourceStorageModeShared];
  }
  memcpy([this->Internals->LightUniformBuffer contents], &lu, sizeof(LightUniforms));
}

VTK_ABI_NAMESPACE_END
