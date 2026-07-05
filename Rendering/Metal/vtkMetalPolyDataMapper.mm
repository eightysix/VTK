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
#include "vtkMapper.h"

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <vector>
#include <unordered_map>
#include <cmath>

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalPolyDataMapper);

//------------------------------------------------------------------------------
struct vtkMetalPolyDataMapper::vtkMetalPolyDataMapperInternals
{
  // Triangle / line geometry
  id<MTLBuffer> VertexPositionBuffer = nil;
  id<MTLBuffer> VertexNormalBuffer = nil;
  id<MTLBuffer> IndexBuffer = nil;
  id<MTLBuffer> LineIndexBuffer = nil;

  id<MTLRenderPipelineState> TrianglePipeline = nil;
  id<MTLRenderPipelineState> LinePipeline = nil;

  // Point geometry — separate buffers so points can draw independently
  id<MTLBuffer> PointPositionBuffer = nil;   // float3 per point
  id<MTLBuffer> PointNormalBuffer = nil;     // float3 per point (from data or default)
  id<MTLBuffer> PointColorBuffer = nil;      // float4 per point (RGBA, from MapScalars)
  id<MTLBuffer> PointTangentBuffer = nil;    // float3 per point (from data or default)
  id<MTLBuffer> PointUVBuffer = nil;         // float2 per point (from data or default)
  id<MTLBuffer> PointColorUVBuffer = nil;    // float2 per point (from data or default)
  id<MTLBuffer> PointConnectivityBuffer = nil; // uint32 per entry (identity map)
  vtkIdType PointVertexCount = 0;             // number of points to draw

  id<MTLRenderPipelineState> PointPipeline = nil;       // basic 1px
  id<MTLRenderPipelineState> PointShapedPipeline = nil; // instanced quads

  // Uniforms
  id<MTLBuffer> SceneUniformBuffer = nil;
  id<MTLBuffer> MaterialUniformBuffer = nil;
  id<MTLBuffer> LightUniformBuffer = nil;
  id<MTLBuffer> CoincidentOffsetBuffer = nil;   // P1-5: polygon/line/point depth bias
  id<MTLBuffer> VertexColorBuffer = nil;        // P1-4: vertex visibility color
  id<MTLBuffer> ClipPlaneBuffer = nil;          // P1-6: clipping planes
  id<MTLBuffer> CellIdOffsetBuffer = nil;       // P2-7: homogeneous cell ID offset

  vtkIdType TriangleVertexCount = 0;
  vtkIdType TriangleIndexCount = 0;
  vtkIdType LineIndexCount = 0;
  bool HasTriangles = false;
  bool HasLines = false;

  vtkIdType CachedInputMTime = 0;
  int CachedRepresentation = -1;

  void ReleaseBuffers()
  {
    VertexPositionBuffer = nil;
    VertexNormalBuffer = nil;
    IndexBuffer = nil;
    LineIndexBuffer = nil;
    TrianglePipeline = nil;
    LinePipeline = nil;

    PointPositionBuffer = nil;
    PointNormalBuffer = nil;
    PointColorBuffer = nil;
    PointTangentBuffer = nil;
    PointUVBuffer = nil;
    PointColorUVBuffer = nil;
    PointConnectivityBuffer = nil;
    PointVertexCount = 0;
    PointPipeline = nil;
    PointShapedPipeline = nil;

    SceneUniformBuffer = nil;
    MaterialUniformBuffer = nil;
    LightUniformBuffer = nil;
    CoincidentOffsetBuffer = nil;
    VertexColorBuffer = nil;
    ClipPlaneBuffer = nil;
    CellIdOffsetBuffer = nil;
    TriangleVertexCount = 0;
    TriangleIndexCount = 0;
    LineIndexCount = 0;
    HasTriangles = false;
    HasLines = false;
    CachedInputMTime = 0;
    CachedRepresentation = -1;
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
    int representation = act->GetProperty()->GetRepresentation();
    if (currentMTime != this->Internals->CachedInputMTime ||
        representation != this->Internals->CachedRepresentation)
    {
      this->Internals->ReleaseBuffers();
      this->Internals->CachedInputMTime = currentMTime;
      this->Internals->CachedRepresentation = representation;
      this->BuildGeometryBuffers((__bridge void*)device, input, act);
    }

    bool hasGeometry = this->Internals->HasTriangles || this->Internals->HasLines ||
                       this->Internals->PointVertexCount > 0;
    if (!hasGeometry)
    {
      return;
    }

    if (representation == VTK_POINTS)
    {
      this->EnsurePointPipelineStates((__bridge void*)device);
    }
    else
    {
      this->EnsurePipelineStates((__bridge void*)device);
    }

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

      // Write point size and actor render options into SceneUniforms at known offsets.
      // Layout: ViewMatrix(64) + ProjectionMatrix(64) + NormalMatrix(48) +
      //         ModelMatrix(64) + Viewport(16) + Flags(256) + PointSize(260)
      char* buf = static_cast<char*>([this->Internals->SceneUniformBuffer contents]);
      float ptSize = static_cast<float>(act->GetProperty()->GetPointSize());
      *reinterpret_cast<float*>(buf + 260) = ptSize;

      // Merge actor render option flags into SceneUniforms flags (offset 256).
      // Bit 0: parallel projection (set by camera)
      // Bit 3: vertex visibility
      // Bit 5: RenderPointsAsSpheres
      // Bit 7: Point2DShape (0=Round, 1=Square)
      vtkProperty* prop = act->GetProperty();
      uint32_t actorFlags = 0;
      actorFlags |= (prop->GetVertexVisibility() ? 1u : 0u) << 3;
      actorFlags |= (prop->GetRenderPointsAsSpheres() ? 1u : 0u) << 5;
      actorFlags |= (static_cast<uint32_t>(prop->GetPoint2DShape())) << 7;
      *reinterpret_cast<uint32_t*>(buf + 256) |= actorFlags;
    }

    this->UpdateMaterialUniforms((__bridge void*)device, act);
    this->UpdateLightUniforms((__bridge void*)device, ren);
    this->UpdateCoincidentOffsetUniforms((__bridge void*)device, act);
    this->UpdateVertexColorUniforms((__bridge void*)device, act);
    this->UpdateClipPlaneUniforms((__bridge void*)device, ren);

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
      if (this->Internals->CoincidentOffsetBuffer)
      {
        [encoder setFragmentBuffer:this->Internals->CoincidentOffsetBuffer offset:0 atIndex:3];
      }
      if (this->Internals->ClipPlaneBuffer)
      {
        [encoder setFragmentBuffer:this->Internals->ClipPlaneBuffer offset:0 atIndex:5];
        [encoder setVertexBuffer:this->Internals->ClipPlaneBuffer offset:0 atIndex:5];
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
      if (this->Internals->CoincidentOffsetBuffer)
      {
        [encoder setFragmentBuffer:this->Internals->CoincidentOffsetBuffer offset:0 atIndex:3];
      }

      [encoder drawIndexedPrimitives:MTLPrimitiveTypeLine
                          indexCount:this->Internals->LineIndexCount
                           indexType:MTLIndexTypeUInt32
                         indexBuffer:this->Internals->LineIndexBuffer
                   indexBufferOffset:0];
    }

    // --- P1-4: Vertex visibility — draw vertex dots on top of surface/wireframe ---
    if (representation != VTK_POINTS && act->GetProperty()->GetVertexVisibility() &&
        this->Internals->PointVertexCount > 0 && this->Internals->PointPositionBuffer)
    {
      float ptSize = act->GetProperty()->GetPointSize();
      if (ptSize > 1.0f && this->Internals->PointShapedPipeline)
      {
        // Shaped vertex dots
        [encoder setRenderPipelineState:this->Internals->PointShapedPipeline];
        [encoder setVertexBuffer:this->Internals->PointPositionBuffer offset:0 atIndex:0];
        [encoder setVertexBuffer:this->Internals->PointConnectivityBuffer offset:0 atIndex:1];
        if (this->Internals->SceneUniformBuffer)
        {
          [encoder setVertexBuffer:this->Internals->SceneUniformBuffer offset:0 atIndex:2];
        }
        if (this->Internals->PointNormalBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointNormalBuffer offset:0 atIndex:3];
        }
        if (this->Internals->PointColorBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointColorBuffer offset:0 atIndex:4];
        }
        if (this->Internals->PointTangentBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointTangentBuffer offset:0 atIndex:6];
        }
        if (this->Internals->PointUVBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointUVBuffer offset:0 atIndex:7];
        }
        if (this->Internals->PointColorUVBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointColorUVBuffer offset:0 atIndex:8];
        }
        if (this->Internals->CellIdOffsetBuffer)
        {
          [encoder setVertexBuffer:this->Internals->CellIdOffsetBuffer offset:0 atIndex:9];
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
          [encoder setFragmentBuffer:this->Internals->SceneUniformBuffer offset:0 atIndex:2];
        }
        if (this->Internals->CoincidentOffsetBuffer)
        {
          [encoder setFragmentBuffer:this->Internals->CoincidentOffsetBuffer offset:0 atIndex:3];
        }
        if (this->Internals->VertexColorBuffer)
        {
          [encoder setFragmentBuffer:this->Internals->VertexColorBuffer offset:0 atIndex:4];
        }
        [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip
                    vertexStart:0
                  vertexCount:4
                instanceCount:this->Internals->PointVertexCount];
      }
      else if (this->Internals->PointPipeline)
      {
        // Basic 1px vertex dots
        [encoder setRenderPipelineState:this->Internals->PointPipeline];
        [encoder setVertexBuffer:this->Internals->PointPositionBuffer offset:0 atIndex:0];
        if (this->Internals->SceneUniformBuffer)
        {
          [encoder setVertexBuffer:this->Internals->SceneUniformBuffer offset:0 atIndex:1];
        }
        if (this->Internals->PointNormalBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointNormalBuffer offset:0 atIndex:2];
        }
        if (this->Internals->PointColorBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointColorBuffer offset:0 atIndex:3];
        }
        if (this->Internals->PointTangentBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointTangentBuffer offset:0 atIndex:6];
        }
        if (this->Internals->PointUVBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointUVBuffer offset:0 atIndex:7];
        }
        if (this->Internals->PointColorUVBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointColorUVBuffer offset:0 atIndex:8];
        }
        if (this->Internals->CellIdOffsetBuffer)
        {
          [encoder setVertexBuffer:this->Internals->CellIdOffsetBuffer offset:0 atIndex:9];
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
          [encoder setFragmentBuffer:this->Internals->SceneUniformBuffer offset:0 atIndex:2];
        }
        if (this->Internals->CoincidentOffsetBuffer)
        {
          [encoder setFragmentBuffer:this->Internals->CoincidentOffsetBuffer offset:0 atIndex:3];
        }
        if (this->Internals->VertexColorBuffer)
        {
          [encoder setFragmentBuffer:this->Internals->VertexColorBuffer offset:0 atIndex:4];
        }
        [encoder drawPrimitives:MTLPrimitiveTypePoint
                    vertexStart:0
                  vertexCount:this->Internals->PointVertexCount];
      }
    }

    // --- Draw points ---
    if (representation == VTK_POINTS && this->Internals->PointVertexCount > 0 &&
        this->Internals->PointPositionBuffer)
    {
      float ptSize = act->GetProperty()->GetPointSize();
      if (ptSize > 1.0f && this->Internals->PointShapedPipeline)
      {
        // Shaped points: instanced triangle-strip quads (4 verts × N instances)
        [encoder setRenderPipelineState:this->Internals->PointShapedPipeline];
        [encoder setVertexBuffer:this->Internals->PointPositionBuffer offset:0 atIndex:0];
        [encoder setVertexBuffer:this->Internals->PointConnectivityBuffer offset:0 atIndex:1];
        if (this->Internals->SceneUniformBuffer)
        {
          [encoder setVertexBuffer:this->Internals->SceneUniformBuffer offset:0 atIndex:2];
        }
        if (this->Internals->PointNormalBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointNormalBuffer offset:0 atIndex:3];
        }
        if (this->Internals->PointColorBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointColorBuffer offset:0 atIndex:4];
        }
        if (this->Internals->PointTangentBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointTangentBuffer offset:0 atIndex:6];
        }
        if (this->Internals->PointUVBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointUVBuffer offset:0 atIndex:7];
        }
        if (this->Internals->PointColorUVBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointColorUVBuffer offset:0 atIndex:8];
        }
        if (this->Internals->CellIdOffsetBuffer)
        {
          [encoder setVertexBuffer:this->Internals->CellIdOffsetBuffer offset:0 atIndex:9];
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
          [encoder setFragmentBuffer:this->Internals->SceneUniformBuffer offset:0 atIndex:2];
        }
        if (this->Internals->CoincidentOffsetBuffer)
        {
          [encoder setFragmentBuffer:this->Internals->CoincidentOffsetBuffer offset:0 atIndex:3];
        }
        if (this->Internals->VertexColorBuffer)
        {
          [encoder setFragmentBuffer:this->Internals->VertexColorBuffer offset:0 atIndex:4];
        }
        [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip
                    vertexStart:0
                  vertexCount:4
                instanceCount:this->Internals->PointVertexCount];
      }
      else if (this->Internals->PointPipeline)
      {
        // Basic 1px points
        [encoder setRenderPipelineState:this->Internals->PointPipeline];
        [encoder setVertexBuffer:this->Internals->PointPositionBuffer offset:0 atIndex:0];
        if (this->Internals->SceneUniformBuffer)
        {
          [encoder setVertexBuffer:this->Internals->SceneUniformBuffer offset:0 atIndex:1];
        }
        if (this->Internals->PointNormalBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointNormalBuffer offset:0 atIndex:2];
        }
        if (this->Internals->PointColorBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointColorBuffer offset:0 atIndex:3];
        }
        if (this->Internals->PointTangentBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointTangentBuffer offset:0 atIndex:6];
        }
        if (this->Internals->PointUVBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointUVBuffer offset:0 atIndex:7];
        }
        if (this->Internals->PointColorUVBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointColorUVBuffer offset:0 atIndex:8];
        }
        if (this->Internals->CellIdOffsetBuffer)
        {
          [encoder setVertexBuffer:this->Internals->CellIdOffsetBuffer offset:0 atIndex:9];
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
          [encoder setFragmentBuffer:this->Internals->SceneUniformBuffer offset:0 atIndex:2];
        }
        if (this->Internals->CoincidentOffsetBuffer)
        {
          [encoder setFragmentBuffer:this->Internals->CoincidentOffsetBuffer offset:0 atIndex:3];
        }
        if (this->Internals->VertexColorBuffer)
        {
          [encoder setFragmentBuffer:this->Internals->VertexColorBuffer offset:0 atIndex:4];
        }
        [encoder drawPrimitives:MTLPrimitiveTypePoint
                    vertexStart:0
                  vertexCount:this->Internals->PointVertexCount];
      }
    }
  }
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::BuildGeometryBuffers(void* mtlDevice, vtkPolyData* polydata, vtkActor* actor)
{
  if (!polydata || !mtlDevice)
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;
  std::vector<float> positions;
  std::vector<float> normals;
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
    vtkIdType npts;
    const vtkIdType* pts;
    polys->InitTraversal();
    while (polys->GetNextCell(npts, pts) && npts > 0)
    {
      if (npts < 3)
      {
        continue;
      }
      // Fan-triangulate polygon
      for (vtkIdType i = 1; i < npts - 1; ++i)
      {
        vtkIdType tri[3] = { pts[0], pts[i], pts[i + 1] };
        double p[3][3];
        for (int j = 0; j < 3; ++j)
        {
          polydata->GetPoint(tri[j], p[j]);
        }

        // Compute face normal
        float e1[3] = { (float)(p[1][0] - p[0][0]), (float)(p[1][1] - p[0][1]), (float)(p[1][2] - p[0][2]) };
        float e2[3] = { (float)(p[2][0] - p[0][0]), (float)(p[2][1] - p[0][1]), (float)(p[2][2] - p[0][2]) };
        float fn[3] = { 0.0f, 1.0f, 0.0f };

        if (normalArray)
        {
          double nn[3];
          normalArray->GetTuple(tri[0], nn);
          fn[0] = (float)nn[0]; fn[1] = (float)nn[1]; fn[2] = (float)nn[2];
        }
        else
        {
          float ne1 = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
          float ne2 = std::sqrt(e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]);
          if (ne1 > 1e-8f && ne2 > 1e-8f)
          {
            e1[0] /= ne1; e1[1] /= ne1; e1[2] /= ne1;
            e2[0] /= ne2; e2[1] /= ne2; e2[2] /= ne2;
          }
          fn[0] = e1[1] * e2[2] - e1[2] * e2[1];
          fn[1] = e1[2] * e2[0] - e1[0] * e2[2];
          fn[2] = e1[0] * e2[1] - e1[1] * e2[0];
          float nn = std::sqrt(fn[0] * fn[0] + fn[1] * fn[1] + fn[2] * fn[2]);
          if (nn > 1e-8f) { fn[0] /= nn; fn[1] /= nn; fn[2] /= nn; }
        }

        // Emit 3 vertices per triangle (no index buffer needed when computing normals)
        for (int j = 0; j < 3; ++j)
        {
          positions.push_back(static_cast<float>(p[j][0]));
          positions.push_back(static_cast<float>(p[j][1]));
          positions.push_back(static_cast<float>(p[j][2]));
          normals.push_back(fn[0]);
          normals.push_back(fn[1]);
          normals.push_back(fn[2]);
        }
      }
    }
    this->Internals->TriangleVertexCount = static_cast<uint32_t>(positions.size() / 3);
    this->Internals->TriangleIndexCount = 0;
    this->Internals->HasTriangles = !positions.empty();
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
  if (!lineIndices.empty())
  {
    this->Internals->LineIndexBuffer = [device
      newBufferWithBytes:lineIndices.data()
                 length:lineIndices.size() * sizeof(uint32_t)
                options:MTLResourceStorageModeShared];
  }

  // Build point buffers — all unique point positions with an identity connectivity map.
  vtkIdType numPts = polydata->GetNumberOfPoints();
  if (numPts > 0)
  {
    std::vector<float> pointPositions(numPts * 3);
    for (vtkIdType i = 0; i < numPts; ++i)
    {
      double pt[3];
      polydata->GetPoint(i, pt);
      pointPositions[i * 3] = static_cast<float>(pt[0]);
      pointPositions[i * 3 + 1] = static_cast<float>(pt[1]);
      pointPositions[i * 3 + 2] = static_cast<float>(pt[2]);
    }
    this->Internals->PointPositionBuffer = [device
      newBufferWithBytes:pointPositions.data()
                 length:pointPositions.size() * sizeof(float)
                options:MTLResourceStorageModeShared];

    // Point normals — from polydata if available, otherwise default (0,0,1).
    // Matches WebGPU: reads point_normals SSBO indexed by point_id.
    std::vector<float> pointNormals(numPts * 3);
    vtkFloatArray* ptNormalArray = nullptr;
    if (pd->GetNormals())
    {
      ptNormalArray = vtkFloatArray::SafeDownCast(pd->GetNormals());
    }
    if (ptNormalArray && ptNormalArray->GetNumberOfTuples() >= numPts)
    {
      for (vtkIdType i = 0; i < numPts; ++i)
      {
        double n[3];
        ptNormalArray->GetTuple(i, n);
        pointNormals[i * 3] = static_cast<float>(n[0]);
        pointNormals[i * 3 + 1] = static_cast<float>(n[1]);
        pointNormals[i * 3 + 2] = static_cast<float>(n[2]);
      }
    }
    else
    {
      // Default normal facing camera (0,0,1) — matches WebGPU fallback
      for (vtkIdType i = 0; i < numPts; ++i)
      {
        pointNormals[i * 3] = 0.0f;
        pointNormals[i * 3 + 1] = 0.0f;
        pointNormals[i * 3 + 2] = 1.0f;
      }
    }
    this->Internals->PointNormalBuffer = [device
      newBufferWithBytes:pointNormals.data()
                 length:pointNormals.size() * sizeof(float)
                options:MTLResourceStorageModeShared];

    // Point colors — from MapScalars (per-point RGBA) or default white.
    // Matches WebGPU: reads point_colors SSBO indexed by point_id.
    std::vector<float> pointColors(numPts * 4, 1.0f); // default white
    int cellFlag = 0;
    vtkUnsignedCharArray* mappedColors = nullptr;
    if (actor)
    {
      mappedColors = this->MapScalars(actor->GetProperty()->GetOpacity(), cellFlag);
    }
    if (mappedColors && cellFlag == 0 &&
        mappedColors->GetNumberOfTuples() >= numPts)
    {
      // Per-point colors — normalize unsigned char RGBA to float [0,1]
      const unsigned char* rgba = mappedColors->GetPointer(0);
      for (vtkIdType i = 0; i < numPts; ++i)
      {
        pointColors[i * 4] = rgba[i * 4] / 255.0f;
        pointColors[i * 4 + 1] = rgba[i * 4 + 1] / 255.0f;
        pointColors[i * 4 + 2] = rgba[i * 4 + 2] / 255.0f;
        pointColors[i * 4 + 3] = rgba[i * 4 + 3] / 255.0f;
      }
    }
    this->Internals->PointColorBuffer = [device
      newBufferWithBytes:pointColors.data()
                 length:pointColors.size() * sizeof(float)
                options:MTLResourceStorageModeShared];

    // Point tangents — from polydata if available, otherwise default (1,0,0).
    // Matches WebGPU: reads point_tangents SSBO indexed by point_id.
    std::vector<float> pointTangents(numPts * 3, 0.0f);
    vtkFloatArray* tangentArray = nullptr;
    if (pd->GetTangents())
    {
      tangentArray = vtkFloatArray::SafeDownCast(pd->GetTangents());
    }
    if (tangentArray && tangentArray->GetNumberOfTuples() >= numPts)
    {
      for (vtkIdType i = 0; i < numPts; ++i)
      {
        double t[3];
        tangentArray->GetTuple(i, t);
        pointTangents[i * 3] = static_cast<float>(t[0]);
        pointTangents[i * 3 + 1] = static_cast<float>(t[1]);
        pointTangents[i * 3 + 2] = static_cast<float>(t[2]);
      }
    }
    else
    {
      // Default tangent (1,0,0)
      for (vtkIdType i = 0; i < numPts; ++i)
      {
        pointTangents[i * 3] = 1.0f;
        pointTangents[i * 3 + 1] = 0.0f;
        pointTangents[i * 3 + 2] = 0.0f;
      }
    }
    this->Internals->PointTangentBuffer = [device
      newBufferWithBytes:pointTangents.data()
                 length:pointTangents.size() * sizeof(float)
                options:MTLResourceStorageModeShared];

    // Point UVs — from polydata if available, otherwise default (0,0).
    // Matches WebGPU: reads point_uvs SSBO indexed by point_id.
    std::vector<float> pointUVs(numPts * 2, 0.0f);
    vtkFloatArray* uvArray = nullptr;
    if (pd->GetTCoords())
    {
      uvArray = vtkFloatArray::SafeDownCast(pd->GetTCoords());
    }
    if (uvArray && uvArray->GetNumberOfTuples() >= numPts)
    {
      for (vtkIdType i = 0; i < numPts; ++i)
      {
        double uv[3];
        uvArray->GetTuple(i, uv);
        pointUVs[i * 2] = static_cast<float>(uv[0]);
        pointUVs[i * 2 + 1] = static_cast<float>(uv[1]);
      }
    }
    this->Internals->PointUVBuffer = [device
      newBufferWithBytes:pointUVs.data()
                 length:pointUVs.size() * sizeof(float)
                options:MTLResourceStorageModeShared];

    // Point color UVs — from polydata if available, otherwise default (0,0).
    // Matches WebGPU: reads point_color_uvs SSBO indexed by point_id.
    std::vector<float> pointColorUVs(numPts * 2, 0.0f);
    // Color UVs are typically the same as regular UVs unless a separate texture channel is used.
    // For now, use the same UV data.
    if (uvArray && uvArray->GetNumberOfTuples() >= numPts)
    {
      for (vtkIdType i = 0; i < numPts; ++i)
      {
        double uv[3];
        uvArray->GetTuple(i, uv);
        pointColorUVs[i * 2] = static_cast<float>(uv[0]);
        pointColorUVs[i * 2 + 1] = static_cast<float>(uv[1]);
      }
    }
    this->Internals->PointColorUVBuffer = [device
      newBufferWithBytes:pointColorUVs.data()
                 length:pointColorUVs.size() * sizeof(float)
                options:MTLResourceStorageModeShared];

    // Connectivity: identity map — vertex_index i maps to point i.
    std::vector<uint32_t> connectivity(numPts);
    for (vtkIdType i = 0; i < numPts; ++i)
    {
      connectivity[i] = static_cast<uint32_t>(i);
    }
    this->Internals->PointConnectivityBuffer = [device
      newBufferWithBytes:connectivity.data()
                 length:connectivity.size() * sizeof(uint32_t)
                options:MTLResourceStorageModeShared];

    // P2-7: Cell ID offset uniform — defaults to 0 for single-actor rendering.
    // For batched rendering, this would be set to the starting point index.
    uint32_t cellIdOffset = 0;
    this->Internals->CellIdOffsetBuffer = [device
      newBufferWithBytes:&cellIdOffset
                 length:sizeof(uint32_t)
                options:MTLResourceStorageModeShared];

    this->Internals->PointVertexCount = numPts;
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
  // P2-8: Second color attachment for picking IDs
  pipelineDesc.colorAttachments[1].pixelFormat = MTLPixelFormatR32Uint;

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
void vtkMetalPolyDataMapper::EnsurePointPipelineStates(void* mtlDevice)
{
  if (this->Internals->PointPipeline && this->Internals->PointShapedPipeline)
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;

  NSError* error = nil;
  NSString* shaderSource = [NSString stringWithUTF8String:vtkMetalShaders];
  id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
  if (!library)
  {
    vtkErrorMacro(<< "Failed to compile Metal shader for points: "
                  << [[error localizedDescription] UTF8String]);
    return;
  }

  // --- Basic 1px point pipeline ---
  // vertex_point_main takes position buffer at [[buffer(0)]], scene at [[buffer(1)]]
  // No vertex descriptor needed — the shader reads from raw buffers.
  if (!this->Internals->PointPipeline)
  {
    id<MTLFunction> vFunc = [library newFunctionWithName:@"vertex_point_main"];
    id<MTLFunction> fFunc = [library newFunctionWithName:@"fragment_point_main"];
    if (vFunc && fFunc)
    {
      MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
      desc.vertexFunction = vFunc;
      desc.fragmentFunction = fFunc;
      desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
      desc.colorAttachments[1].pixelFormat = MTLPixelFormatR32Uint;  // P2-8: picking IDs
      desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

      error = nil;
      this->Internals->PointPipeline =
        [device newRenderPipelineStateWithDescriptor:desc error:&error];
      if (!this->Internals->PointPipeline)
      {
        vtkErrorMacro(<< "Point pipeline: " << [[error localizedDescription] UTF8String]);
      }
    }
  }

  // --- Shaped point pipeline (instanced triangle-strip quads) ---
  // vertex_point_shaped_main takes position buffer at [[buffer(0)]],
  // connectivity at [[buffer(1)]], scene at [[buffer(2)]].
  if (!this->Internals->PointShapedPipeline)
  {
    id<MTLFunction> vFunc = [library newFunctionWithName:@"vertex_point_shaped_main"];
    id<MTLFunction> fFunc = [library newFunctionWithName:@"fragment_point_shaped_main"];
    if (vFunc && fFunc)
    {
      MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
      desc.vertexFunction = vFunc;
      desc.fragmentFunction = fFunc;
      desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
      desc.colorAttachments[1].pixelFormat = MTLPixelFormatR32Uint;  // P2-8: picking IDs
      desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
      // No backface culling for point quads
      desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;

      error = nil;
      this->Internals->PointShapedPipeline =
        [device newRenderPipelineStateWithDescriptor:desc error:&error];
      if (!this->Internals->PointShapedPipeline)
      {
        vtkErrorMacro(<< "Point shaped pipeline: " << [[error localizedDescription] UTF8String]);
      }
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
  // ambientColor: rgb + ambient_intensity
  // diffuseColor: rgb + diffuse_intensity
  // specularColor: rgb + specular_intensity
  // color: base color (unused in lighting)
  // opacity, specularPower, 2 pad
  float mu[20];
  memset(mu, 0, sizeof(mu));

  // ambientColor.rgb = property ambient color, .w = ambient intensity
  double ac[3];
  prop->GetAmbientColor(ac);
  mu[0] = static_cast<float>(ac[0]);
  mu[1] = static_cast<float>(ac[1]);
  mu[2] = static_cast<float>(ac[2]);
  mu[3] = static_cast<float>(prop->GetAmbient());

  // diffuseColor.rgb = property diffuse color, .w = diffuse intensity
  double dc[3];
  prop->GetDiffuseColor(dc);
  mu[4] = static_cast<float>(dc[0]);
  mu[5] = static_cast<float>(dc[1]);
  mu[6] = static_cast<float>(dc[2]);
  mu[7] = static_cast<float>(prop->GetDiffuse());

  // specularColor.rgb = property specular color, .w = specular intensity
  double sc[3];
  prop->GetSpecularColor(sc);
  mu[8] = static_cast<float>(sc[0]);
  mu[9] = static_cast<float>(sc[1]);
  mu[10] = static_cast<float>(sc[2]);
  mu[11] = static_cast<float>(prop->GetSpecular());

  // color = base actor color (unused in lighting shader)
  double rgb[3];
  prop->GetColor(rgb);
  mu[12] = static_cast<float>(rgb[0]);
  mu[13] = static_cast<float>(rgb[1]);
  mu[14] = static_cast<float>(rgb[2]);
  mu[15] = 1.0f;

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

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::UpdateCoincidentOffsetUniforms(void* mtlDevice, vtkActor* actor)
{
  if (!mtlDevice || !actor)
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;

  // CoincidentOffsetUniforms layout: 5 floats (20 bytes)
  // Matches Metal shader's CoincidentOffsetUniforms struct.
  float co[5];
  memset(co, 0, sizeof(co));

  const int resolveMode = vtkMapper::GetResolveCoincidentTopology();
  if (auto* vtkmap = vtkMapper::SafeDownCast(actor->GetMapper()))
  {
    if (resolveMode == VTK_RESOLVE_POLYGON_OFFSET)
    {
      double pgFactor = 0.0, pgUnits = 0.0;
      double lnFactor = 0.0, lnUnits = 0.0;
      double ptUnits = 0.0;
      vtkmap->GetCoincidentTopologyPolygonOffsetParameters(pgFactor, pgUnits);
      vtkmap->GetCoincidentTopologyLineOffsetParameters(lnFactor, lnUnits);
      vtkmap->GetCoincidentTopologyPointOffsetParameter(ptUnits);
      co[0] = static_cast<float>(pgFactor);
      co[1] = static_cast<float>(pgUnits);
      co[2] = static_cast<float>(lnFactor);
      co[3] = static_cast<float>(lnUnits);
      co[4] = static_cast<float>(ptUnits);
    }
    else if (resolveMode == VTK_RESOLVE_SHIFT_ZBUFFER)
    {
      const double zShift = vtkMapper::GetResolveCoincidentTopologyZShift();
      co[1] = static_cast<float>(zShift * 4.0);
    }
  }

  if (!this->Internals->CoincidentOffsetBuffer)
  {
    this->Internals->CoincidentOffsetBuffer = [device
      newBufferWithLength:sizeof(co)
                 options:MTLResourceStorageModeShared];
  }
  memcpy([this->Internals->CoincidentOffsetBuffer contents], co, sizeof(co));
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::UpdateVertexColorUniforms(void* mtlDevice, vtkActor* actor)
{
  if (!mtlDevice || !actor)
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;

  // VertexColorUniforms layout: float4 (16 bytes)
  // Default white; overridden when vertex visibility is on.
  float vc[4] = { 1.0f, 1.0f, 1.0f, 1.0f };

  if (actor->GetProperty()->GetVertexVisibility())
  {
    double vcol[3];
    actor->GetProperty()->GetVertexColor(vcol);
    vc[0] = static_cast<float>(vcol[0]);
    vc[1] = static_cast<float>(vcol[1]);
    vc[2] = static_cast<float>(vcol[2]);
    vc[3] = 1.0f;
  }

  if (!this->Internals->VertexColorBuffer)
  {
    this->Internals->VertexColorBuffer = [device
      newBufferWithLength:sizeof(vc)
                 options:MTLResourceStorageModeShared];
  }
  memcpy([this->Internals->VertexColorBuffer contents], vc, sizeof(vc));
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::UpdateClipPlaneUniforms(void* mtlDevice, vtkRenderer* ren)
{
  if (!mtlDevice || !ren)
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;

  // ClipPlaneUniforms layout: float4 planes[6] (96 bytes) + int numClipPlanes (4 bytes)
  // = 100 bytes total, padded to 16-byte alignment = 112 bytes
  // Using flat float array to avoid alignment issues.
  float cp[28]; // 7 floats × 4 components = 28 floats = 112 bytes
  memset(cp, 0, sizeof(cp));

  vtkCamera* cam = ren->GetActiveCamera();
  int numPlanes = 0;

  if (cam && cam->GetClippingRange())
  {
    // Get clip planes from the renderer (set by vtkClipPlanes or vtkAssembly)
    // For now, use the near/far clipping planes as defaults
    double nearClip = cam->GetClippingRange()[0];
    double farClip = cam->GetClippingRange()[1];

    // Near plane in view space: z = -nearClip (points behind camera are clipped)
    // In world space: dot(plane, point) >= 0 for visible points
    // The near plane normal in view space is (0, 0, -1), point on plane is (0, 0, -nearClip)
    // But we need to transform to world space... simplified: just pass identity planes
    // The actual clip planes come from vtkClipPlanes filter, not the camera
  }

  // Fill with identity-like planes (no clipping) by default
  // planes[0..3] = 4 clip planes (ax+by+cz+d format)
  // planes[4..5] = reserved
  // numClipPlanes at offset 24 (index 24 in float array)
  // Initialize all planes to (0,0,0,1) which never clips
  for (int i = 0; i < 24; ++i)
  {
    cp[i] = 0.0f;
  }
  // Set d=1 for each plane so dot(plane, (x,y,z,1)) = 1 > 0 always
  cp[3] = 1.0f;   // plane 0
  cp[7] = 1.0f;   // plane 1
  cp[11] = 1.0f;  // plane 2
  cp[15] = 1.0f;  // plane 3

  // numClipPlanes = 0 (no clipping by default)
  // This will be overridden when vtkClipPlanes is used
  reinterpret_cast<int*>(&cp[24])[0] = 0;

  if (!this->Internals->ClipPlaneBuffer)
  {
    this->Internals->ClipPlaneBuffer = [device
      newBufferWithLength:sizeof(cp)
                 options:MTLResourceStorageModeShared];
  }
  memcpy([this->Internals->ClipPlaneBuffer contents], cp, sizeof(cp));
}

VTK_ABI_NAMESPACE_END
