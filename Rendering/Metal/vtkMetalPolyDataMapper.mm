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

  // P1-1A/1B: per-vertex color for triangle/line surfaces (float4 RGBA per vertex)
  id<MTLBuffer> SurfaceColorBuffer = nil;
  bool HasSurfaceColors = false;

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

  // P2-8: Picking IDs
  id<MTLBuffer> TriangleCellIdBuffer = nil;    // GPU output: per-triangle-vertex cell IDs
  id<MTLBuffer> LineCellIdBuffer = nil;        // GPU output: per-line-vertex cell IDs
  id<MTLBuffer> PointCellIdBuffer = nil;       // GPU output: per-point cell IDs
  id<MTLBuffer> PropIdBuffer = nil;            // single uint32: actor prop ID
  id<MTLBuffer> PrimitiveToCellBuffer = nil;   // CPU→GPU: maps primitive index → cell index
  id<MTLComputePipelineState> CellToPrimitivePipeline = nil;
  vtkIdType TrianglePrimitiveCount = 0;        // number of triangles for compute dispatch
  vtkIdType LinePrimitiveCount = 0;            // number of line segments for compute dispatch

  vtkIdType TriangleVertexCount = 0;
  vtkIdType TriangleIndexCount = 0;
  vtkIdType LineIndexCount = 0;
  bool HasTriangles = false;
  bool HasLines = false;

  // P2-2A: Wireframe representation — polygon edges extracted as line segments
  // When representation == VTK_WIREFRAME, polygon edges are added to LineIndexBuffer
  // and HasTriangles is set to false so only line drawing occurs.

  // P2-2B: Edge visibility on surfaces — wireframe overlay on VTK_SURFACE
  id<MTLBuffer> EdgeVertexPositionBuffer = nil; // edge vertex positions (separate from triangle positions)
  id<MTLBuffer> EdgeVertexNormalBuffer = nil;   // edge vertex normals
  id<MTLBuffer> EdgeSurfaceColorBuffer = nil;   // edge vertex colors (float4 per vertex)
  id<MTLBuffer> EdgeIndexBuffer = nil;          // polygon edge indices for wireframe overlay
  vtkIdType EdgeIndexCount = 0;
  vtkIdType EdgeVertexCount = 0;
  bool HasEdgeOverlay = false;
  id<MTLBuffer> EdgeColorUniformBuffer = nil;   // edge color (float4 RGBA)
  id<MTLRenderPipelineState> EdgePipeline = nil; // pipeline for edge rendering

  // P2-2C: Triangle index buffers — deduplicated vertices + index buffer
  // IndexBuffer is populated when vertices can be deduplicated.

  vtkIdType CachedInputMTime = 0;
  int CachedRepresentation = -1;
  bool CachedEdgeVisibility = false;  // P2-2B: track edge visibility changes

  void ReleaseBuffers()
  {
    VertexPositionBuffer = nil;
    VertexNormalBuffer = nil;
    IndexBuffer = nil;
    LineIndexBuffer = nil;
    SurfaceColorBuffer = nil;
    HasSurfaceColors = false;
    TrianglePipeline = nil;
    LinePipeline = nil;

    // P2-2A/2B: Wireframe and edge overlay buffers
    EdgeVertexPositionBuffer = nil;
    EdgeVertexNormalBuffer = nil;
    EdgeSurfaceColorBuffer = nil;
    EdgeIndexBuffer = nil;
    EdgeIndexCount = 0;
    EdgeVertexCount = 0;
    HasEdgeOverlay = false;
    EdgeColorUniformBuffer = nil;
    EdgePipeline = nil;

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
    TriangleCellIdBuffer = nil;
    LineCellIdBuffer = nil;
    PointCellIdBuffer = nil;
    PropIdBuffer = nil;
    PrimitiveToCellBuffer = nil;
    CellToPrimitivePipeline = nil;
    TrianglePrimitiveCount = 0;
    LinePrimitiveCount = 0;
    TriangleVertexCount = 0;
    TriangleIndexCount = 0;
    LineIndexCount = 0;
    HasTriangles = false;
    HasLines = false;
    CachedInputMTime = 0;
    CachedRepresentation = -1;
    CachedEdgeVisibility = false;
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
    bool edgeVisibility = act->GetProperty()->GetEdgeVisibility();
    if (currentMTime != this->Internals->CachedInputMTime ||
        representation != this->Internals->CachedRepresentation ||
        edgeVisibility != this->Internals->CachedEdgeVisibility)
    {
      this->Internals->ReleaseBuffers();
      this->Internals->CachedInputMTime = currentMTime;
      this->Internals->CachedRepresentation = representation;
      this->Internals->CachedEdgeVisibility = edgeVisibility;
      this->BuildGeometryBuffers((__bridge void*)device, input, act);
    }

    bool hasGeometry = this->Internals->HasTriangles || this->Internals->HasLines ||
                       this->Internals->HasEdgeOverlay ||
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
      // Bit 8: has surface vertex colors (P1-1A/1B)
      vtkProperty* prop = act->GetProperty();
      uint32_t actorFlags = 0;
      actorFlags |= (prop->GetVertexVisibility() ? 1u : 0u) << 3;
      actorFlags |= (prop->GetRenderPointsAsSpheres() ? 1u : 0u) << 5;
      actorFlags |= (static_cast<uint32_t>(prop->GetPoint2DShape())) << 7;
      actorFlags |= (this->Internals->HasSurfaceColors ? 1u : 0u) << 8;
      *reinterpret_cast<uint32_t*>(buf + 256) |= actorFlags;
    }

    this->UpdateMaterialUniforms((__bridge void*)device, act);
    this->UpdateLightUniforms((__bridge void*)device, ren);
    this->UpdateCoincidentOffsetUniforms((__bridge void*)device, act);
    this->UpdateVertexColorUniforms((__bridge void*)device, act);
    this->UpdateClipPlaneUniforms((__bridge void*)device, ren);

    // P2-2A: Skip triangle drawing when in wireframe mode
    bool skipTriangles = (representation == VTK_WIREFRAME);

    if (!skipTriangles && this->Internals->HasTriangles && this->Internals->TrianglePipeline)
    {
      [encoder setRenderPipelineState:this->Internals->TrianglePipeline];
      [encoder setVertexBuffer:this->Internals->VertexPositionBuffer offset:0 atIndex:0];
      if (this->Internals->VertexNormalBuffer)
      {
        [encoder setVertexBuffer:this->Internals->VertexNormalBuffer offset:0 atIndex:1];
      }
      // P1-1A/1B: bind per-vertex color buffer at vertex buffer index 3
      if (this->Internals->SurfaceColorBuffer)
      {
        [encoder setVertexBuffer:this->Internals->SurfaceColorBuffer offset:0 atIndex:3];
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
        // P1-1A: bind scene uniforms to fragment buffer(2) so fragment shader can read flags
        [encoder setFragmentBuffer:this->Internals->SceneUniformBuffer offset:0 atIndex:2];
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

      // P2-8: Bind picking ID buffers for triangle rendering
      if (this->Internals->TriangleCellIdBuffer)
      {
        [encoder setVertexBuffer:this->Internals->TriangleCellIdBuffer offset:0 atIndex:6];
      }
      if (this->Internals->PropIdBuffer)
      {
        [encoder setVertexBuffer:this->Internals->PropIdBuffer offset:0 atIndex:7];
      }

      // P1-1C: set backface/frontface cull mode
      if (act->GetProperty()->GetBackfaceCulling())
      {
        [encoder setCullMode:MTLCullModeBack];
      }
      else if (act->GetProperty()->GetFrontfaceCulling())
      {
        [encoder setCullMode:MTLCullModeFront];
      }
      else
      {
        [encoder setCullMode:MTLCullModeNone];
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
      // P1-1A/1B: bind per-vertex color buffer at vertex buffer index 3
      if (this->Internals->SurfaceColorBuffer)
      {
        [encoder setVertexBuffer:this->Internals->SurfaceColorBuffer offset:0 atIndex:3];
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
        // P1-1A: bind scene uniforms to fragment buffer(2) so fragment shader can read flags
        [encoder setFragmentBuffer:this->Internals->SceneUniformBuffer offset:0 atIndex:2];
      }
      if (this->Internals->CoincidentOffsetBuffer)
      {
        [encoder setFragmentBuffer:this->Internals->CoincidentOffsetBuffer offset:0 atIndex:3];
      }

      // P2-8: Bind picking ID buffers for line rendering
      if (this->Internals->LineCellIdBuffer)
      {
        [encoder setVertexBuffer:this->Internals->LineCellIdBuffer offset:0 atIndex:6];
      }
      if (this->Internals->PropIdBuffer)
      {
        [encoder setVertexBuffer:this->Internals->PropIdBuffer offset:0 atIndex:7];
      }

      // P1-1C: set cull mode for lines
      [encoder setCullMode:MTLCullModeNone];

      [encoder drawIndexedPrimitives:MTLPrimitiveTypeLine
                          indexCount:this->Internals->LineIndexCount
                           indexType:MTLIndexTypeUInt32
                         indexBuffer:this->Internals->LineIndexBuffer
                   indexBufferOffset:0];
    }

    // P2-2B: Edge visibility — draw wireframe edges on top of surfaces
    if (representation == VTK_SURFACE && act->GetProperty()->GetEdgeVisibility() &&
        this->Internals->HasEdgeOverlay && this->Internals->EdgePipeline &&
        this->Internals->EdgeIndexBuffer)
    {
      // Ensure edge pipeline is created
      this->EnsureEdgePipelineState((__bridge void*)device);
      if (!this->Internals->EdgePipeline)
      {
        return;
      }

      // Update edge color uniform
      this->UpdateEdgeColorUniform((__bridge void*)device, act);

      [encoder setRenderPipelineState:this->Internals->EdgePipeline];
      // P2-2B: Use separate edge vertex buffers (not the main triangle position buffer)
      [encoder setVertexBuffer:this->Internals->EdgeVertexPositionBuffer offset:0 atIndex:0];
      if (this->Internals->EdgeVertexNormalBuffer)
      {
        [encoder setVertexBuffer:this->Internals->EdgeVertexNormalBuffer offset:0 atIndex:1];
      }
      // P1-1A/1B: bind per-vertex color buffer at vertex buffer index 3
      if (this->Internals->EdgeSurfaceColorBuffer)
      {
        [encoder setVertexBuffer:this->Internals->EdgeSurfaceColorBuffer offset:0 atIndex:3];
      }

      // Bind scene uniforms to vertex buffer(2) and fragment buffer(2)
      if (this->Internals->SceneUniformBuffer)
      {
        [encoder setVertexBuffer:this->Internals->SceneUniformBuffer offset:0 atIndex:2];
        [encoder setFragmentBuffer:this->Internals->SceneUniformBuffer offset:0 atIndex:2];
      }
      // Bind coincident offset to fragment buffer(3) for line offset
      if (this->Internals->CoincidentOffsetBuffer)
      {
        [encoder setFragmentBuffer:this->Internals->CoincidentOffsetBuffer offset:0 atIndex:3];
      }
      // Bind edge color to fragment buffer(4)
      if (this->Internals->EdgeColorUniformBuffer)
      {
        [encoder setFragmentBuffer:this->Internals->EdgeColorUniformBuffer offset:0 atIndex:4];
      }
      // Bind material to fragment buffer(0) for opacity
      if (this->Internals->MaterialUniformBuffer)
      {
        [encoder setFragmentBuffer:this->Internals->MaterialUniformBuffer offset:0 atIndex:0];
      }
      // Bind picking IDs for edge rendering (use line cell ID buffer)
      if (this->Internals->LineCellIdBuffer)
      {
        [encoder setVertexBuffer:this->Internals->LineCellIdBuffer offset:0 atIndex:6];
      }
      if (this->Internals->PropIdBuffer)
      {
        [encoder setVertexBuffer:this->Internals->PropIdBuffer offset:0 atIndex:7];
      }

      // No backface culling for edges
      [encoder setCullMode:MTLCullModeNone];

      [encoder drawIndexedPrimitives:MTLPrimitiveTypeLine
                          indexCount:this->Internals->EdgeIndexCount
                           indexType:MTLIndexTypeUInt32
                         indexBuffer:this->Internals->EdgeIndexBuffer
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
        if (this->Internals->PointCellIdBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointCellIdBuffer offset:0 atIndex:11];
        }
        if (this->Internals->PropIdBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PropIdBuffer offset:0 atIndex:12];
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
        if (this->Internals->PointCellIdBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointCellIdBuffer offset:0 atIndex:11];
        }
        if (this->Internals->PropIdBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PropIdBuffer offset:0 atIndex:12];
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
        if (this->Internals->PointCellIdBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointCellIdBuffer offset:0 atIndex:11];
        }
        if (this->Internals->PropIdBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PropIdBuffer offset:0 atIndex:12];
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
        if (this->Internals->PointCellIdBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PointCellIdBuffer offset:0 atIndex:11];
        }
        if (this->Internals->PropIdBuffer)
        {
          [encoder setVertexBuffer:this->Internals->PropIdBuffer offset:0 atIndex:12];
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
  std::vector<float> surfaceColors;  // P1-1A/1B: float4 per vertex
  std::vector<uint32_t> lineIndices;

  // P2-2B: Edge geometry for wireframe overlay on surfaces (separate vertex + index buffers)
  std::vector<float> edgePositions;
  std::vector<float> edgeNormals;
  std::vector<float> edgeColors;
  std::vector<uint32_t> edgeIndices;
  std::unordered_map<vtkIdType, uint32_t> edgeVertexMap;

  // P2-8: per-primitive cell ID mapping (primitive index → cell index)
  std::vector<uint32_t> trianglePrimToCell;
  std::vector<uint32_t> linePrimToCell;

  // Get representation for wireframe/edge handling
  int representation = actor ? actor->GetProperty()->GetRepresentation() : VTK_SURFACE;
  bool edgeVisibility = actor ? actor->GetProperty()->GetEdgeVisibility() : false;

  vtkPointData* pd = polydata->GetPointData();
  vtkFloatArray* normalArray = nullptr;
  if (pd->GetNormals())
  {
    normalArray = vtkFloatArray::SafeDownCast(pd->GetNormals());
  }

  // P1-1A/1B: MapScalars early so both triangle and line paths can use the result.
  int cellFlag = 0;
  vtkUnsignedCharArray* mappedColors = nullptr;
  if (actor)
  {
    mappedColors = this->MapScalars(actor->GetProperty()->GetOpacity(), cellFlag);
  }

  // Process polygons — handles VTK_WIREFRAME, VTK_SURFACE, and edge visibility
  vtkCellArray* polys = polydata->GetPolys();
  vtkIdType polyCellIdx = 0;
  // P2-2A: Vertex deduplication map for wireframe polygon edges
  std::unordered_map<vtkIdType, uint32_t> wireVertexMap;
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

      if (representation == VTK_WIREFRAME)
      {
        // P2-2A: Wireframe — extract edges from polygon as line segments.
        // For each polygon with vertices [v0, v1, ..., vn-1], emit line segments
        // (v0,v1), (v1,v2), ..., (vn-2,vn-1), (vn-1,v0) — closing the polygon.
        // This matches WebGPU's polygon_edges_to_lines compute shader.
        // Deduplicate vertices by point ID to avoid duplicates for shared edges.
        for (vtkIdType i = 0; i < npts; ++i)
        {
          vtkIdType v0 = pts[i];
          vtkIdType v1 = pts[(i + 1) % npts];  // wraps to close polygon

          // Add vertex v0 if not already added
          auto it0 = wireVertexMap.find(v0);
          uint32_t idx0;
          if (it0 == wireVertexMap.end())
          {
            idx0 = static_cast<uint32_t>(positions.size() / 3);
            wireVertexMap[v0] = idx0;

            double p[3];
            polydata->GetPoint(v0, p);
            positions.push_back(static_cast<float>(p[0]));
            positions.push_back(static_cast<float>(p[1]));
            positions.push_back(static_cast<float>(p[2]));
            if (normalArray)
            {
              double n[3];
              normalArray->GetTuple(v0, n);
              normals.push_back(static_cast<float>(n[0]));
              normals.push_back(static_cast<float>(n[1]));
              normals.push_back(static_cast<float>(n[2]));
            }
            // Color for wireframe vertex
            if (mappedColors)
            {
              const unsigned char* rgba = mappedColors->GetPointer(0);
              vtkIdType idx = (cellFlag == 0) ? v0 : polyCellIdx;
              surfaceColors.push_back(rgba[idx * 4] / 255.0f);
              surfaceColors.push_back(rgba[idx * 4 + 1] / 255.0f);
              surfaceColors.push_back(rgba[idx * 4 + 2] / 255.0f);
              surfaceColors.push_back(rgba[idx * 4 + 3] / 255.0f);
            }
            else
            {
              surfaceColors.push_back(1.0f);
              surfaceColors.push_back(1.0f);
              surfaceColors.push_back(1.0f);
              surfaceColors.push_back(1.0f);
            }
          }
          else
          {
            idx0 = it0->second;
          }

          // Add vertex v1 if not already added
          auto it1 = wireVertexMap.find(v1);
          uint32_t idx1;
          if (it1 == wireVertexMap.end())
          {
            idx1 = static_cast<uint32_t>(positions.size() / 3);
            wireVertexMap[v1] = idx1;

            double p[3];
            polydata->GetPoint(v1, p);
            positions.push_back(static_cast<float>(p[0]));
            positions.push_back(static_cast<float>(p[1]));
            positions.push_back(static_cast<float>(p[2]));
            if (normalArray)
            {
              double n[3];
              normalArray->GetTuple(v1, n);
              normals.push_back(static_cast<float>(n[0]));
              normals.push_back(static_cast<float>(n[1]));
              normals.push_back(static_cast<float>(n[2]));
            }
            // Color for wireframe vertex
            if (mappedColors)
            {
              const unsigned char* rgba = mappedColors->GetPointer(0);
              vtkIdType idx = (cellFlag == 0) ? v1 : polyCellIdx;
              surfaceColors.push_back(rgba[idx * 4] / 255.0f);
              surfaceColors.push_back(rgba[idx * 4 + 1] / 255.0f);
              surfaceColors.push_back(rgba[idx * 4 + 2] / 255.0f);
              surfaceColors.push_back(rgba[idx * 4 + 3] / 255.0f);
            }
            else
            {
              surfaceColors.push_back(1.0f);
              surfaceColors.push_back(1.0f);
              surfaceColors.push_back(1.0f);
              surfaceColors.push_back(1.0f);
            }
          }
          else
          {
            idx1 = it1->second;
          }

          lineIndices.push_back(idx0);
          lineIndices.push_back(idx1);
          linePrimToCell.push_back(static_cast<uint32_t>(polyCellIdx));
        }
      }
      else
      {
        // VTK_SURFACE: Fan-triangulate polygon for filled rendering
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

          // Emit 3 vertices per triangle
          for (int j = 0; j < 3; ++j)
          {
            positions.push_back(static_cast<float>(p[j][0]));
            positions.push_back(static_cast<float>(p[j][1]));
            positions.push_back(static_cast<float>(p[j][2]));
            normals.push_back(fn[0]);
            normals.push_back(fn[1]);
            normals.push_back(fn[2]);

            // P1-1A/1B: per-vertex color from scalar mapping
            if (mappedColors)
            {
              const unsigned char* rgba = mappedColors->GetPointer(0);
              vtkIdType idx = (cellFlag == 0) ? tri[j] : polyCellIdx;
              surfaceColors.push_back(rgba[idx * 4] / 255.0f);
              surfaceColors.push_back(rgba[idx * 4 + 1] / 255.0f);
              surfaceColors.push_back(rgba[idx * 4 + 2] / 255.0f);
              surfaceColors.push_back(rgba[idx * 4 + 3] / 255.0f);
            }
            else
            {
              surfaceColors.push_back(1.0f);
              surfaceColors.push_back(1.0f);
              surfaceColors.push_back(1.0f);
              surfaceColors.push_back(1.0f);
            }
          }
          trianglePrimToCell.push_back(static_cast<uint32_t>(polyCellIdx));

          // P2-2B: When edge visibility is on, also build edge index buffer from polygon edges
          if (edgeVisibility)
          {
            // Extract polygon edges for wireframe overlay.
            // For the first triangle in the fan, emit edges (v0,v1) and (v0,v2)
            // For middle triangles, emit edge (v0,v1) only (the diagonal)
            // For the last triangle, emit edges (v0,v1) and (v1,v2)
            // This hides interior edges shared between triangles in the fan.
            // Use edgeVertexMap to deduplicate vertices by original point ID.
            auto addEdgeVertex = [&](vtkIdType pointId) -> uint32_t {
              auto it = edgeVertexMap.find(pointId);
              if (it != edgeVertexMap.end())
              {
                return it->second;
              }
              // Add new vertex to separate edge buffers
              uint32_t idx = static_cast<uint32_t>(edgePositions.size() / 3);
              edgeVertexMap[pointId] = idx;

              double pt[3];
              polydata->GetPoint(pointId, pt);
              edgePositions.push_back(static_cast<float>(pt[0]));
              edgePositions.push_back(static_cast<float>(pt[1]));
              edgePositions.push_back(static_cast<float>(pt[2]));
              if (normalArray)
              {
                double n[3];
                normalArray->GetTuple(pointId, n);
                edgeNormals.push_back(static_cast<float>(n[0]));
                edgeNormals.push_back(static_cast<float>(n[1]));
                edgeNormals.push_back(static_cast<float>(n[2]));
              }
              if (mappedColors)
              {
                const unsigned char* rgba = mappedColors->GetPointer(0);
                vtkIdType idx2 = (cellFlag == 0) ? pointId : polyCellIdx;
                edgeColors.push_back(rgba[idx2 * 4] / 255.0f);
                edgeColors.push_back(rgba[idx2 * 4 + 1] / 255.0f);
                edgeColors.push_back(rgba[idx2 * 4 + 2] / 255.0f);
                edgeColors.push_back(rgba[idx2 * 4 + 3] / 255.0f);
              }
              else
              {
                edgeColors.push_back(1.0f);
                edgeColors.push_back(1.0f);
                edgeColors.push_back(1.0f);
                edgeColors.push_back(1.0f);
              }
              return idx;
            };

            if (i == 1)
            {
              // First triangle: emit edges (v0,v1) and (v0,v2) — two boundary edges
              edgeIndices.push_back(addEdgeVertex(tri[0]));
              edgeIndices.push_back(addEdgeVertex(tri[1]));
              edgeIndices.push_back(addEdgeVertex(tri[0]));
              edgeIndices.push_back(addEdgeVertex(tri[2]));
            }
            else if (i == npts - 2)
            {
              // Last triangle: emit edges (v0,v1) and (v1,v2) — two boundary edges
              edgeIndices.push_back(addEdgeVertex(tri[0]));
              edgeIndices.push_back(addEdgeVertex(tri[1]));
              edgeIndices.push_back(addEdgeVertex(tri[1]));
              edgeIndices.push_back(addEdgeVertex(tri[2]));
            }
            else
            {
              // Middle triangle: emit only edge (v0,v1) — the polygon boundary edge
              edgeIndices.push_back(addEdgeVertex(tri[0]));
              edgeIndices.push_back(addEdgeVertex(tri[1]));
            }
          }
        }
      }
      polyCellIdx++;
    }

    if (representation == VTK_WIREFRAME)
    {
      // Wireframe: no triangles, only lines
      this->Internals->TriangleVertexCount = 0;
      this->Internals->TriangleIndexCount = 0;
      this->Internals->HasTriangles = false;
    }
    else
    {
      this->Internals->TriangleVertexCount = static_cast<uint32_t>(positions.size() / 3);
      this->Internals->TriangleIndexCount = 0;
      this->Internals->HasTriangles = !positions.empty();
    }
  }

  // Process lines
  vtkCellArray* lines = polydata->GetLines();
  vtkIdType lineCellIdx = 0;
  if (lines && lines->GetNumberOfCells() > 0)
  {
    if (cellFlag != 0 && mappedColors)
    {
      // P1-1B: Cell coloring for lines — no deduplication, each segment endpoint
      // gets the cell's color (flat shading via duplicated per-vertex colors).
      uint32_t nextPointId = static_cast<uint32_t>(positions.size() / 3);

      vtkIdType npts;
      const vtkIdType* pts;
      lines->InitTraversal();
      while (lines->GetNextCell(npts, pts) && npts > 0)
      {
        const unsigned char* rgba = mappedColors->GetPointer(0);
        float cr = rgba[lineCellIdx * 4] / 255.0f;
        float cg = rgba[lineCellIdx * 4 + 1] / 255.0f;
        float cb = rgba[lineCellIdx * 4 + 2] / 255.0f;
        float ca = rgba[lineCellIdx * 4 + 3] / 255.0f;

        for (vtkIdType i = 0; i < npts; ++i)
        {
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
          surfaceColors.push_back(cr);
          surfaceColors.push_back(cg);
          surfaceColors.push_back(cb);
          surfaceColors.push_back(ca);
        }
        uint32_t base = nextPointId;
        nextPointId += npts;
        for (vtkIdType i = 0; i < npts - 1; ++i)
        {
          lineIndices.push_back(base + i);
          lineIndices.push_back(base + i + 1);
          linePrimToCell.push_back(static_cast<uint32_t>(lineCellIdx));
        }
        lineCellIdx++;
      }
    }
    else
    {
      // P1-1A: Point coloring (or no coloring) — deduplicate by point ID
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
            // P1-1A: point color
            if (mappedColors && cellFlag == 0)
            {
              const unsigned char* rgba = mappedColors->GetPointer(0);
              surfaceColors.push_back(rgba[pts[i] * 4] / 255.0f);
              surfaceColors.push_back(rgba[pts[i] * 4 + 1] / 255.0f);
              surfaceColors.push_back(rgba[pts[i] * 4 + 2] / 255.0f);
              surfaceColors.push_back(rgba[pts[i] * 4 + 3] / 255.0f);
            }
            else
            {
              surfaceColors.push_back(1.0f);
              surfaceColors.push_back(1.0f);
              surfaceColors.push_back(1.0f);
              surfaceColors.push_back(1.0f);
            }
          }
        }
        for (vtkIdType i = 0; i < npts - 1; ++i)
        {
          lineIndices.push_back(pointMap[pts[i]]);
          lineIndices.push_back(pointMap[pts[i + 1]]);
          linePrimToCell.push_back(static_cast<uint32_t>(lineCellIdx));
        }
        lineCellIdx++;
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

  // P2-2B: Create edge geometry buffers for wireframe overlay on surfaces
  if (!edgeIndices.empty() && !edgePositions.empty())
  {
    this->Internals->EdgeVertexPositionBuffer = [device
      newBufferWithBytes:edgePositions.data()
                 length:edgePositions.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
    if (!edgeNormals.empty())
    {
      this->Internals->EdgeVertexNormalBuffer = [device
        newBufferWithBytes:edgeNormals.data()
                   length:edgeNormals.size() * sizeof(float)
                  options:MTLResourceStorageModeShared];
    }
    if (!edgeColors.empty())
    {
      this->Internals->EdgeSurfaceColorBuffer = [device
        newBufferWithBytes:edgeColors.data()
                   length:edgeColors.size() * sizeof(float)
                  options:MTLResourceStorageModeShared];
    }
    this->Internals->EdgeIndexBuffer = [device
      newBufferWithBytes:edgeIndices.data()
                 length:edgeIndices.size() * sizeof(uint32_t)
                options:MTLResourceStorageModeShared];
    this->Internals->EdgeIndexCount = edgeIndices.size();
    this->Internals->EdgeVertexCount = edgePositions.size() / 3;
    this->Internals->HasEdgeOverlay = true;
  }

  // P1-1A/1B: Create surface color buffer (float4 per triangle/line vertex)
  if (!surfaceColors.empty())
  {
    this->Internals->SurfaceColorBuffer = [device
      newBufferWithBytes:surfaceColors.data()
                 length:surfaceColors.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
    this->Internals->HasSurfaceColors = (mappedColors != nullptr);
  }
  else if (!positions.empty())
  {
    // Always provide a white color buffer so the vertex shader can always read from buffer(3)
    std::vector<float> whiteColors(positions.size() / 3 * 4, 1.0f);
    this->Internals->SurfaceColorBuffer = [device
      newBufferWithBytes:whiteColors.data()
                 length:whiteColors.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
  }

  // P2-8: Create primitive-to-cell mapping buffers and cell ID output buffers
  if (!trianglePrimToCell.empty())
  {
    this->Internals->PrimitiveToCellBuffer = [device
      newBufferWithBytes:trianglePrimToCell.data()
                 length:trianglePrimToCell.size() * sizeof(uint32_t)
                options:MTLResourceStorageModeShared];
    this->Internals->TrianglePrimitiveCount = trianglePrimToCell.size();
    // Cell ID output buffer — same size as primitive-to-cell (one uint per primitive)
    this->Internals->TriangleCellIdBuffer = [device
      newBufferWithLength:trianglePrimToCell.size() * sizeof(uint32_t)
                 options:MTLResourceStorageModeShared];
  }
  if (!linePrimToCell.empty())
  {
    this->Internals->LineCellIdBuffer = [device
      newBufferWithLength:linePrimToCell.size() * sizeof(uint32_t)
                 options:MTLResourceStorageModeShared];
    this->Internals->LinePrimitiveCount = linePrimToCell.size();
    // Reuse PrimitiveToCellBuffer if only lines exist, otherwise we need a separate one
    if (trianglePrimToCell.empty())
    {
      this->Internals->PrimitiveToCellBuffer = [device
        newBufferWithBytes:linePrimToCell.data()
                   length:linePrimToCell.size() * sizeof(uint32_t)
                  options:MTLStorageModeShared];
    }
  }

  // Prop ID buffer — single uint32
  {
    uint32_t propId = 0;
    this->Internals->PropIdBuffer = [device
      newBufferWithBytes:&propId
                 length:sizeof(uint32_t)
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
    // Note: mappedColors and cellFlag are already set from the early MapScalars call above.
    std::vector<float> pointColors(numPts * 4, 1.0f); // default white
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

    // P2-8: Point cell IDs — identity mapping (point i → cell i)
    std::vector<uint32_t> pointCellIds(numPts);
    for (vtkIdType i = 0; i < numPts; ++i)
    {
      pointCellIds[i] = static_cast<uint32_t>(i);
    }
    this->Internals->PointCellIdBuffer = [device
      newBufferWithBytes:pointCellIds.data()
                 length:pointCellIds.size() * sizeof(uint32_t)
                options:MTLResourceStorageModeShared];
  }

  // P2-8: Create compute pipeline and dispatch cell-to-primitive mapping
  if (!this->Internals->CellToPrimitivePipeline)
  {
    NSString* shaderSource = [NSString stringWithUTF8String:vtkMetalShaders];
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
    if (library)
    {
      id<MTLFunction> kernelFunc = [library newFunctionWithName:@"cellToPrimitive"];
      if (kernelFunc)
      {
        this->Internals->CellToPrimitivePipeline =
          [device newComputePipelineStateWithFunction:kernelFunc error:&error];
      }
    }
  }

  // Dispatch compute kernel for triangle cell IDs
  if (this->Internals->CellToPrimitivePipeline &&
      this->Internals->TrianglePrimitiveCount > 0 &&
      this->Internals->TriangleCellIdBuffer &&
      this->Internals->PrimitiveToCellBuffer)
  {
    id<MTLCommandBuffer> cmdBuf = [(__bridge id<MTLCommandQueue>)
      [device newCommandQueue] commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:this->Internals->CellToPrimitivePipeline];
    [encoder setBuffer:this->Internals->TriangleCellIdBuffer offset:0 atIndex:0];
    [encoder setBuffer:this->Internals->PrimitiveToCellBuffer offset:0 atIndex:1];
    [encoder setBuffer:this->Internals->CellIdOffsetBuffer offset:0 atIndex:2];
    NSUInteger threadgroupSize = this->Internals->CellToPrimitivePipeline.maxTotalThreadsPerThreadgroup;
    if (threadgroupSize > static_cast<NSUInteger>(this->Internals->TrianglePrimitiveCount))
    {
      threadgroupSize = this->Internals->TrianglePrimitiveCount;
    }
    MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(this->Internals->TrianglePrimitiveCount), 1, 1);
    MTLSize tgSize = MTLSizeMake(threadgroupSize, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
  }

  // Dispatch compute kernel for line cell IDs
  if (this->Internals->CellToPrimitivePipeline &&
      this->Internals->LinePrimitiveCount > 0 &&
      this->Internals->LineCellIdBuffer &&
      this->Internals->PrimitiveToCellBuffer &&
      this->Internals->TrianglePrimitiveCount == 0)
  {
    // Only dispatch lines if we didn't already dispatch triangles
    // (they share PrimitiveToCellBuffer)
    id<MTLCommandBuffer> cmdBuf = [(__bridge id<MTLCommandQueue>)
      [device newCommandQueue] commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:this->Internals->CellToPrimitivePipeline];
    [encoder setBuffer:this->Internals->LineCellIdBuffer offset:0 atIndex:0];
    [encoder setBuffer:this->Internals->PrimitiveToCellBuffer offset:0 atIndex:1];
    [encoder setBuffer:this->Internals->CellIdOffsetBuffer offset:0 atIndex:2];
    NSUInteger threadgroupSize = this->Internals->CellToPrimitivePipeline.maxTotalThreadsPerThreadgroup;
    if (threadgroupSize > static_cast<NSUInteger>(this->Internals->LinePrimitiveCount))
    {
      threadgroupSize = this->Internals->LinePrimitiveCount;
    }
    MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(this->Internals->LinePrimitiveCount), 1, 1);
    MTLSize tgSize = MTLSizeMake(threadgroupSize, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
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
  pipelineDesc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;  // P2-8: picking IDs

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
      desc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;  // P2-8: picking IDs
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
      desc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;  // P2-8: picking IDs
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
void vtkMetalPolyDataMapper::EnsureEdgePipelineState(void* mtlDevice)
{
  if (this->Internals->EdgePipeline)
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;

  NSError* error = nil;
  NSString* shaderSource = [NSString stringWithUTF8String:vtkMetalShaders];
  id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
  if (!library)
  {
    vtkErrorMacro(<< "Failed to compile Metal shader for edges: "
                  << [[error localizedDescription] UTF8String]);
    return;
  }

  // Edge pipeline uses vertex_main + fragment_edge_main
  // vertex_main: transforms position, outputs vertex color
  // fragment_edge_main: outputs flat edge color from uniform
  id<MTLFunction> vFunc = [library newFunctionWithName:@"vertex_main"];
  id<MTLFunction> fFunc = [library newFunctionWithName:@"fragment_edge_main"];
  if (vFunc && fFunc)
  {
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

    MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = vFunc;
    desc.fragmentFunction = fFunc;
    desc.vertexDescriptor = vertexDesc;
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    desc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;  // P2-8: picking IDs
    desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassLine;

    error = nil;
    this->Internals->EdgePipeline =
      [device newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!this->Internals->EdgePipeline)
    {
      vtkErrorMacro(<< "Edge pipeline: " << [[error localizedDescription] UTF8String]);
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
void vtkMetalPolyDataMapper::UpdateEdgeColorUniform(void* mtlDevice, vtkActor* actor)
{
  if (!mtlDevice || !actor)
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;

  // EdgeColorUniforms layout: float4 (16 bytes)
  // RGB from vtkProperty::GetEdgeColor(), alpha = 1.0
  float ec[4] = { 0.0f, 0.0f, 0.0f, 1.0f };  // default black

  double edgeCol[3];
  actor->GetProperty()->GetEdgeColor(edgeCol);
  ec[0] = static_cast<float>(edgeCol[0]);
  ec[1] = static_cast<float>(edgeCol[1]);
  ec[2] = static_cast<float>(edgeCol[2]);
  ec[3] = 1.0f;

  if (!this->Internals->EdgeColorUniformBuffer)
  {
    this->Internals->EdgeColorUniformBuffer = [device
      newBufferWithLength:sizeof(ec)
                 options:MTLResourceStorageModeShared];
  }
  memcpy([this->Internals->EdgeColorUniformBuffer contents], ec, sizeof(ec));
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
