// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalPolyDataMapper.h"

#include "vtkMetalRenderWindow.h"
#include "vtkMetalRenderer.h"
#include "vtkMetalCamera.h"
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

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <vector>
#include <unordered_map>

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalPolyDataMapper);

//------------------------------------------------------------------------------
// Internal state for the mapper
//------------------------------------------------------------------------------
struct vtkMetalPolyDataMapper::vtkMetalPolyDataMapperInternals
{
  // Metal buffers
  id<MTLBuffer> VertexPositionBuffer = nil;
  id<MTLBuffer> VertexNormalBuffer = nil;
  id<MTLBuffer> IndexBuffer = nil;
  id<MTLBuffer> LineIndexBuffer = nil;

  // Pipeline states
  id<MTLRenderPipelineState> TrianglePipeline = nil;
  id<MTLRenderPipelineState> LinePipeline = nil;

  // Uniform buffers
  id<MTLBuffer> SceneUniformBuffer = nil;
  id<MTLBuffer> MaterialUniformBuffer = nil;
  id<MTLBuffer> LightUniformBuffer = nil;

  // Geometry counts
  vtkIdType TriangleVertexCount = 0;
  vtkIdType TriangleIndexCount = 0;
  vtkIdType LineIndexCount = 0;
  bool HasTriangles = false;
  bool HasLines = false;

  // Cache tracking
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

//------------------------------------------------------------------------------
vtkMetalPolyDataMapper::~vtkMetalPolyDataMapper() = default;

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//------------------------------------------------------------------------------
vtkMetalPolyDataMapper::MapperHashType vtkMetalPolyDataMapper::GenerateHash(
  vtkPolyData* polydata)
{
  if (!polydata)
  {
    return 0;
  }
  return static_cast<MapperHashType>(
    polydata->GetMTime() + this->GetInputAbstractArrayToIndexBufferVTK()->GetMTime());
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::ReleaseGraphicsResources(vtkWindow*)
{
  this->Internals->ReleaseBuffers();
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::RenderPiece(vtkRenderer* ren, vtkActor* act)
{
  vtkMetalRenderWindow* renWin =
    vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());
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

    // Check if we need to rebuild geometry
    vtkIdType currentMTime = input->GetMTime();
    if (currentMTime != this->Internals->CachedInputMTime)
    {
      this->Internals->ReleaseBuffers();
      this->Internals->CachedInputMTime = currentMTime;

      // Extract geometry from VTK polydata
      this->BuildGeometryBuffers(device, input);
    }

    if (this->Internals->TriangleVertexCount == 0 && this->Internals->LineIndexCount == 0)
    {
      return;
    }

    // Build pipeline states if needed
    this->EnsurePipelineStates(device);

    // Get the render encoder from the renderer's current render pass
    // We need to restructure slightly - the renderer should provide the encoder
    // For now, create a new encoder from the command buffer
    id<MTLCommandBuffer> commandBuffer = [(__bridge id<MTLCommandQueue>)
      [device newCommandQueue] commandBuffer];

    CAMetalLayer* layer = (__bridge CAMetalLayer*)renWin->GetMetalLayer();
    id<CAMetalDrawable> drawable = [layer nextDrawable];
    if (!drawable)
    {
      return;
    }

    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = drawable.texture;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    double bgColor[3];
    ren->GetBackground(bgColor);
    rpd.colorAttachments[0].clearColor = MTLClearColorMake(bgColor[0], bgColor[1], bgColor[2], 1.0);
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:rpd];

    // Set viewport
    int* size = ren->GetSize();
    double* viewport = ren->GetViewport();
    MTLViewport metalViewport;
    metalViewport.originX = viewport[0] * size[0];
    metalViewport.originY = viewport[1] * size[1];
    metalViewport.width = viewport[2] * size[0];
    metalViewport.height = viewport[3] * size[1];
    metalViewport.znear = 0.0;
    metalViewport.zfar = 1.0;
    [encoder setViewport:metalViewport];

    // SceneTransforms struct matching the Metal shader layout
    struct SceneTransforms
    {
      float ViewMatrix[4][4];
      float ProjectionMatrix[4][4];
      float NormalMatrix[3][4];
      float ModelMatrix[4][4];
      float Viewport[4];
      uint32_t Flags;
    };

    // Update camera uniforms
    vtkMetalCamera* metalCamera =
      vtkMetalCamera::SafeDownCast(ren->GetActiveCamera());
    if (metalCamera)
    {
      metalCamera->Render(ren);

      // Create or update scene uniform buffer
      if (!this->Internals->SceneUniformBuffer)
      {
        this->Internals->SceneUniformBuffer = [device
          newBufferWithLength:sizeof(SceneTransforms)
                     options:MTLResourceStorageModeShared];
      }
      memcpy([this->Internals->SceneUniformBuffer contents],
             metalCamera->GetCachedSceneTransforms(),
             sizeof(SceneTransforms));
    }

    // Update material uniforms
    this->UpdateMaterialUniforms(device, act);

    // Update light uniforms
    this->UpdateLightUniforms(device, ren);

    // Draw triangles
    if (this->Internals->HasTriangles && this->Internals->TrianglePipeline)
    {
      [encoder setRenderPipelineState:this->Internals->TrianglePipeline];

      // Set vertex buffers
      [encoder setVertexBuffer:this->Internals->VertexPositionBuffer offset:0 atIndex:0];
      if (this->Internals->VertexNormalBuffer)
      {
        [encoder setVertexBuffer:this->Internals->VertexNormalBuffer offset:0 atIndex:1];
      }
      else
      {
        // Provide a zero-normal buffer as fallback
        static id<MTLBuffer> zeroNormals = nil;
        static dispatch_once_t onceToken;
        dispatch_once(&onceToken, ^{
          zeroNormals = [device newBufferWithLength:sizeof(float) * 3 * 1024
                                           options:MTLResourceStorageModeShared];
          memset([zeroNormals contents], 0, zeroNormals.length);
        });
        [encoder setVertexBuffer:zeroNormals offset:0 atIndex:1];
      }

      // Set fragment buffers (material + lights)
      if (this->Internals->MaterialUniformBuffer)
      {
        [encoder setFragmentBuffer:this->Internals->MaterialUniformBuffer offset:0 atIndex:0];
      }
      if (this->Internals->LightUniformBuffer)
      {
        [encoder setFragmentBuffer:this->Internals->LightUniformBuffer offset:0 atIndex:1];
      }

      // Set scene uniforms
      if (this->Internals->SceneUniformBuffer)
      {
        [encoder setVertexBuffer:this->Internals->SceneUniformBuffer offset:0 atIndex:2];
      }

      // Draw indexed triangles
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

    // Draw lines
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

    [encoder endEncoding];
    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];
  }
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::BuildGeometryBuffers(id<MTLDevice> device,
                                                   vtkPolyData* polydata)
{
  if (!polydata || !device)
  {
    return;
  }

  // Collect all vertices (positions and normals)
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
    // Build a map from old point IDs to new sequential IDs
    std::unordered_map<vtkIdType, uint32_t> pointMap;
    uint32_t nextPointId = 0;

    vtkIdType npts;
    const vtkIdType* pts;
    vtkCellArray::Iterator it;
    for (polys->InitTraversal(it); polys->GetNextCell(it, npts, pts);)
    {
      if (npts < 3)
      {
        continue;
      }

      // Fan triangulation for polygons with >3 vertices
      for (vtkIdType i = 1; i < npts - 1; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          vtkIdType corner = (j == 0) ? pts[0] : (j == 1) ? pts[i] : pts[i + 1];

          if (pointMap.find(corner) == pointMap.end())
          {
            pointMap[corner] = nextPointId++;

            // Add position
            double pt[3];
            polydata->GetPoint(corner, pt);
            positions.push_back(static_cast<float>(pt[0]));
            positions.push_back(static_cast<float>(pt[1]));
            positions.push_back(static_cast<float>(pt[2]));

            // Add normal
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

  // Also process triangle strips (some sources produce strips)
  vtkCellArray* strips = polydata->GetStrips();
  if (strips && strips->GetNumberOfCells() > 0)
  {
    std::unordered_map<vtkIdType, uint32_t> pointMap;
    uint32_t nextPointId = static_cast<uint32_t>(positions.size() / 3);

    vtkIdType npts;
    const vtkIdType* pts;
    vtkCellArray::Iterator it;
    for (strips->InitTraversal(it); strips->GetNextCell(it, npts, pts);)
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

      // Convert strip to triangles
      for (vtkIdType i = 0; i < npts - 2; ++i)
      {
        if (i % 2 == 0)
        {
          triangleIndices.push_back(pointMap[pts[i]]);
          triangleIndices.push_back(pointMap[pts[i + 1]]);
          triangleIndices.push_back(pointMap[pts[i + 2]]);
        }
        else
        {
          triangleIndices.push_back(pointMap[pts[i + 1]]);
          triangleIndices.push_back(pointMap[pts[i]]);
          triangleIndices.push_back(pointMap[pts[i + 2]]);
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
    vtkCellArray::Iterator it;
    for (lines->InitTraversal(it); lines->GetNextCell(it, npts, pts);)
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

      // Add line segments
      for (vtkIdType i = 0; i < npts - 1; ++i)
      {
        lineIndices.push_back(pointMap[pts[i]]);
        lineIndices.push_back(pointMap[pts[i + 1]]);
      }
    }

    this->Internals->LineIndexCount = lineIndices.size();
    this->Internals->HasLines = !lineIndices.empty();
  }

  // Create Metal buffers
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
void vtkMetalPolyDataMapper::EnsurePipelineStates(id<MTLDevice> device)
{
  if (this->Internals->TrianglePipeline && this->Internals->LinePipeline)
  {
    return;
  }

  // Load the default Metal library
  id<MTLLibrary> library = [device newDefaultLibrary];
  if (!library)
  {
    vtkErrorMacro(<< "Failed to create Metal shader library");
    return;
  }

  id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_main"];
  id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragment_main"];

  if (!vertexFunc || !fragmentFunc)
  {
    vtkErrorMacro(<< "Failed to find shader functions");
    return;
  }

  // Triangle pipeline
  if (!this->Internals->TrianglePipeline)
  {
    MTLVertexDescriptor* vertexDesc = [[MTLVertexDescriptor alloc] init];

    // Attribute 0: position (float3)
    vertexDesc.attributes[0].format = MTLVertexFormatFloat3;
    vertexDesc.attributes[0].offset = 0;
    vertexDesc.attributes[0].bufferIndex = 0;

    // Attribute 1: normal (float3)
    vertexDesc.attributes[1].format = MTLVertexFormatFloat3;
    vertexDesc.attributes[1].offset = 0;
    vertexDesc.attributes[1].bufferIndex = 1;

    // Layout 0: position buffer (tight packing)
    vertexDesc.layouts[0].stride = sizeof(float) * 3;
    vertexDesc.layouts[0].stepRate = 1;
    vertexDesc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;

    // Layout 1: normal buffer (tight packing)
    vertexDesc.layouts[1].stride = sizeof(float) * 3;
    vertexDesc.layouts[1].stepRate = 1;
    vertexDesc.layouts[1].stepFunction = MTLVertexStepFunctionPerVertex;

    MTLRenderPipelineDescriptor* pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunc;
    pipelineDesc.fragmentFunction = fragmentFunc;
    pipelineDesc.vertexDescriptor = vertexDesc;
    pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

    NSError* error = nil;
    this->Internals->TrianglePipeline =
      [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    if (!this->Internals->TrianglePipeline)
    {
      vtkErrorMacro(<< "Failed to create triangle pipeline: "
                    << [[error localizedDescription] UTF8String]);
    }
  }

  // Line pipeline (same shaders, different primitive type)
  if (!this->Internals->LinePipeline)
  {
    // Reuse the same pipeline descriptor - the primitive type is set at draw time
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

    NSError* error = nil;
    this->Internals->LinePipeline =
      [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    if (!this->Internals->LinePipeline)
    {
      vtkErrorMacro(<< "Failed to create line pipeline: "
                    << [[error localizedDescription] UTF8String]);
    }
  }
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::UpdateMaterialUniforms(id<MTLDevice> device, vtkActor* actor)
{
  if (!actor || !device)
  {
    return;
  }

  vtkProperty* prop = actor->GetProperty();

  // Metal MaterialUniforms struct layout:
  // float4 color (rgba)
  // float4 ambient (rgb + intensity)
  // float4 diffuse (rgb + intensity)
  // float4 specular (rgb + intensity)
  // float opacity
  // float specularPower
  // float2 padding
  struct MaterialUniforms
  {
    float color[4];
    float ambient[4];
    float diffuse[4];
    float specular[4];
    float opacity;
    float specularPower;
    float padding[2];
  };

  MaterialUniforms mu;
  prop->GetColor(mu.color);
  mu.color[3] = 1.0f;

  double ambient = prop->GetAmbient();
  mu.ambient[0] = mu.ambient[1] = mu.ambient[2] = static_cast<float>(ambient);
  mu.ambient[3] = 1.0f;

  double diffuse = prop->GetDiffuse();
  mu.diffuse[0] = mu.diffuse[1] = mu.diffuse[2] = static_cast<float>(diffuse);
  mu.diffuse[3] = 1.0f;

  double specular = prop->GetSpecular();
  mu.specular[0] = mu.specular[1] = mu.specular[2] = static_cast<float>(specular);
  mu.specular[3] = 1.0f;

  mu.opacity = static_cast<float>(prop->GetOpacity());
  mu.specularPower = static_cast<float>(prop->GetSpecularPower());
  mu.padding[0] = mu.padding[1] = 0.0f;

  if (!this->Internals->MaterialUniformBuffer)
  {
    this->Internals->MaterialUniformBuffer = [device
      newBufferWithLength:sizeof(MaterialUniforms)
                 options:MTLResourceStorageModeShared];
  }

  memcpy([this->Internals->MaterialUniformBuffer contents], &mu, sizeof(MaterialUniforms));
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::UpdateLightUniforms(id<MTLDevice> device, vtkRenderer* ren)
{
  if (!device || !ren)
  {
    return;
  }

  // Metal LightUniforms struct layout:
  // Light lights[MAX_LIGHTS] (8 lights)
  // int lightCount
  // float3 padding
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

  vtkLightCollection* lc = ren->GetLights();
  vtkLight* light;
  int count = 0;

  lc->InitTraversal();
  while ((light = lc->GetNextLight()) && count < 8)
  {
    LightData& ld = lu.lights[count];

    if (light->GetPositional())
    {
      if (light->GetConeAngle() < 90.0)
      {
        ld.position[3] = 3.0f; // spot
      }
      else
      {
        ld.position[3] = 2.0f; // point
      }
      ld.position[0] = static_cast<float>(light->GetPosition()[0]);
      ld.position[1] = static_cast<float>(light->GetPosition()[1]);
      ld.position[2] = static_cast<float>(light->GetPosition()[2]);
    }
    else
    {
      // Directional or headlight
      if (light->GetLightType() == VTK_LIGHT_TYPE_HEADLIGHT)
      {
        ld.position[3] = 0.0f; // headlight
      }
      else
      {
        ld.position[3] = 1.0f; // directional
      }
      ld.position[0] = 0.0f;
      ld.position[1] = 0.0f;
      ld.position[2] = 0.0f;
    }

    ld.direction[0] = static_cast<float>(light->GetDirection()[0]);
    ld.direction[1] = static_cast<float>(light->GetDirection()[1]);
    ld.direction[2] = static_cast<float>(light->GetDirection()[2]);
    ld.direction[3] = static_cast<float>(light->GetConeAngle());

    light->GetDiffuseColor(ld.color);
    float intensity = static_cast<float>(light->GetIntensity());
    ld.color[0] *= intensity;
    ld.color[1] *= intensity;
    ld.color[2] *= intensity;
    ld.color[3] = 1.0f;

    ld.attenuation[0] = static_cast<float>(light->GetConstantAttenuation());
    ld.attenuation[1] = static_cast<float>(light->GetLinearAttenuation());
    ld.attenuation[2] = static_cast<float>(light->GetQuadraticAttenuation());
    ld.attenuation[3] = static_cast<float>(light->GetExponent());

    count++;
  }

  lu.lightCount = count;

  // If no lights, add a default headlight
  if (count == 0)
  {
    lu.lights[0].position[3] = 0.0f; // headlight
    lu.lights[0].color[0] = lu.lights[0].color[1] = lu.lights[0].color[2] = 1.0f;
    lu.lights[0].color[3] = 1.0f;
    lu.lights[0].diffuse[0] = lu.lights[0].diffuse[1] = lu.lights[0].diffuse[2] = 1.0f;
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
