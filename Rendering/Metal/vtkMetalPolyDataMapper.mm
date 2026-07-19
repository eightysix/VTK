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
#include "vtkNew.h"
#include "vtkPlaneCollection.h"
#include "vtkPlane.h"
#include "vtkTexture.h"
#include "vtkDataObject.h"

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <vector>
#include <unordered_map>
#include <cmath>
#include <variant>

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

  // P5-5A: texture coordinates for triangles (float2 per vertex)
  id<MTLBuffer> TriangleUVBuffer = nil;

  id<MTLRenderPipelineState> TrianglePipeline = nil;
  id<MTLRenderPipelineState> LinePipeline = nil;

  // P5-5A: Actor texture and sampler for texture mapping
  id<MTLTexture> ActorTexture = nil;
  id<MTLSamplerState> ActorSampler = nil;
  id<MTLTexture> DefaultTexture = nil;   // 1x1 white fallback
  id<MTLSamplerState> DefaultSampler = nil;
  vtkIdType CachedTextureMTime = 0;

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

  // P6-6A: GPU tessellation — compute-based polygon → triangle / line conversion
  id<MTLComputePipelineState> PolygonToTrianglePipeline = nil;
  id<MTLComputePipelineState> PolyLineToLinePipeline = nil;
  id<MTLComputePipelineState> PolygonEdgesToLinesPipeline = nil;
  id<MTLBuffer> TessOutputConnectivityBuffer = nil; // GPU output: tessellated index buffer
  id<MTLBuffer> TessEdgeArrayBuffer = nil;          // GPU output: per-triangle edge visibility
  id<MTLBuffer> TessParamsBuffer = nil;             // uniform: numCells + cellIdOffset
  bool UseGPUTessellation = false;

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

  // P3-3A: Thick line pipeline — screen-space quad expansion for lineWidth > 1
  id<MTLRenderPipelineState> ThickLinePipeline = nil;
  id<MTLBuffer> ThickLineLineWidthBuffer = nil;  // float: line width in pixels
  vtkIdType ThickLineSegmentCount = 0;           // number of line segments for instanced draw

  // P3-3B: Round Cap + Round Join line pipeline — 36 verts per instance
  id<MTLRenderPipelineState> RoundCapLinePipeline = nil;
  vtkIdType RoundCapLineSegmentCount = 0;

  // P3-3C: Miter Join line pipeline — 4 verts per instance, miter offsets in shader
  id<MTLRenderPipelineState> MiterJoinLinePipeline = nil;
  vtkIdType MiterJoinLineSegmentCount = 0;
  id<MTLBuffer> MiterJoinSegmentCountBuffer = nil;  // uint32: total segment count for bounds check

  // P2-2C: Triangle index buffers — deduplicated vertices + index buffer
  // IndexBuffer is populated when vertices can be deduplicated.

  // 8B: Depth peeling pipeline states (same vertex shader, peeling fragment shaders)
  id<MTLRenderPipelineState> TriangleInitPeelPipeline = nil;  // init depth range
  id<MTLRenderPipelineState> TrianglePeelPipeline = nil;      // main peel pass
  id<MTLBuffer> PeelUniformBuffer = nil;                       // peel mode + pass index

  // 8D: Vertex attribute mapping — custom per-vertex buffers from user-mapped data arrays
  std::unordered_map<std::string, id<MTLBuffer>> ExtraAttributeBuffers;
  std::unordered_map<std::string, int> ExtraAttributeComponentCounts;

  vtkIdType CachedInputMTime = 0;
  int CachedRepresentation = -1;
  bool CachedEdgeVisibility = false;  // P2-2B: track edge visibility changes
  float CachedLineWidth = -1.0f;     // P3-3A: track line width changes
  int CachedSampleCount = 0;        // 8A: track MSAA sample count changes

  // 8C: Render bundle caching — pre-recorded encoder commands for static geometry
  // When geometry hasn't changed between frames, replay cached commands instead of
  // re-encoding all setVertexBuffer/setFragmentBuffer/drawPrimitives calls.
  // This eliminates CPU encoding overhead for static scenes.
  struct RenderBundleDrawCommand
  {
    enum Type
    {
      SetPipelineState,
      SetVertexBuffer,
      SetFragmentBuffer,
      SetFragmentTexture,
      SetFragmentSamplerState,
      SetCullMode,
      DrawPrimitives,
      DrawIndexedPrimitives
    };
    Type type;

    // Polymorphic params via variant
    struct SetPipelineStateParams
    {
      id<MTLRenderPipelineState> pipeline;
    };
    struct SetBufferParams
    {
      id<MTLBuffer> buffer;
      NSUInteger offset;
      NSUInteger index;
    };
    struct SetTextureParams
    {
      id<MTLTexture> texture;
      NSUInteger index;
    };
    struct SetSamplerParams
    {
      id<MTLSamplerState> sampler;
      NSUInteger index;
    };
    struct SetCullModeParams
    {
      MTLCullMode mode;
    };
    struct DrawPrimitivesParams
    {
      MTLPrimitiveType primitiveType;
      NSUInteger vertexStart;
      NSUInteger vertexCount;
      NSUInteger instanceCount; // 0 for non-instanced
    };
    struct DrawIndexedPrimitivesParams
    {
      MTLPrimitiveType primitiveType;
      NSUInteger indexCount;
      MTLIndexType indexType;
      id<MTLBuffer> indexBuffer;
      NSUInteger indexBufferOffset;
    };

    using Params = std::variant<SetPipelineStateParams, SetBufferParams, SetTextureParams,
      SetSamplerParams, SetCullModeParams, DrawPrimitivesParams, DrawIndexedPrimitivesParams>;
    Params params;
  };

  struct RenderBundle
  {
    std::vector<RenderBundleDrawCommand> Commands;
    bool Valid = false;

    void Invalidate()
    {
      Commands.clear();
      Valid = false;
    }
  };

  RenderBundle Bundle;

  // Bundle validity tracking — detects when the bundle needs rebuilding.
  // Bundle is valid only when ALL of these match the values at bundle creation time.
  vtkIdType BundleGeometryMTime = 0;
  int BundleRepresentation = -1;
  bool BundleEdgeVisibility = false;
  float BundleLineWidth = -1.0f;
  int BundleSampleCount = 0;
  int BundlePeelMode = 0;

  void InvalidateRenderBundle()
  {
    Bundle.Invalidate();
  }

  void ReleaseBuffers()
  {
    VertexPositionBuffer = nil;
    VertexNormalBuffer = nil;
    IndexBuffer = nil;
    LineIndexBuffer = nil;
    SurfaceColorBuffer = nil;
    HasSurfaceColors = false;
    TriangleUVBuffer = nil;
    ActorTexture = nil;
    ActorSampler = nil;
    DefaultTexture = nil;
    DefaultSampler = nil;
    CachedTextureMTime = 0;
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

    // P3-3A: Thick line pipeline
    ThickLinePipeline = nil;
    ThickLineLineWidthBuffer = nil;
    ThickLineSegmentCount = 0;

    // P3-3B/3C: Round cap and miter join pipelines
    RoundCapLinePipeline = nil;
    RoundCapLineSegmentCount = 0;
    MiterJoinLinePipeline = nil;
    MiterJoinLineSegmentCount = 0;
    MiterJoinSegmentCountBuffer = nil;

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

    // P6-6A: GPU tessellation buffers
    PolygonToTrianglePipeline = nil;
    PolyLineToLinePipeline = nil;
    PolygonEdgesToLinesPipeline = nil;
    TessOutputConnectivityBuffer = nil;
    TessEdgeArrayBuffer = nil;
    TessParamsBuffer = nil;
    UseGPUTessellation = false;
    TriangleVertexCount = 0;
    TriangleIndexCount = 0;
    LineIndexCount = 0;
    HasTriangles = false;
    HasLines = false;
    CachedInputMTime = 0;
    CachedRepresentation = -1;
    CachedEdgeVisibility = false;
    CachedLineWidth = -1.0f;

    // 8B: Depth peeling pipelines
    TriangleInitPeelPipeline = nil;
    TrianglePeelPipeline = nil;
    PeelUniformBuffer = nil;

    // 8D: Extra attribute buffers
    ExtraAttributeBuffers.clear();
    ExtraAttributeComponentCounts.clear();

    // 8C: Render bundle invalidation — geometry changed, cached commands are stale
    InvalidateRenderBundle();
    BundleGeometryMTime = 0;
    BundleRepresentation = -1;
    BundleEdgeVisibility = false;
    BundleLineWidth = -1.0f;
    BundleSampleCount = 0;
    BundlePeelMode = 0;
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
// 8D: Vertex attribute mapping — map VTK data arrays to generic vertex attributes.
// Follows the same pattern as vtkOpenGLPolyDataMapper: stores mappings and creates
// per-point GPU buffers that are bound at buffer indices 16+ for custom shaders.
//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::MapDataArrayToVertexAttribute(
  const char* vertexAttributeName,
  const char* dataArrayName,
  int fieldAssociation,
  int componentno)
{
  if (!vertexAttributeName)
  {
    return;
  }

  // Remove existing mapping for this attribute name
  this->RemoveVertexAttributeMapping(vertexAttributeName);
  if (!dataArrayName)
  {
    return;
  }

  ExtraAttributeValue aval;
  aval.DataArrayName = dataArrayName;
  aval.FieldAssociation = fieldAssociation;
  aval.ComponentNumber = componentno;

  this->ExtraAttributes.insert(std::make_pair(vertexAttributeName, aval));

  this->Internals->InvalidateRenderBundle();
  this->Modified();
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::RemoveVertexAttributeMapping(const char* vertexAttributeName)
{
  if (!vertexAttributeName)
  {
    return;
  }
  auto itr = this->ExtraAttributes.find(vertexAttributeName);
  if (itr != this->ExtraAttributes.end())
  {
    this->ExtraAttributes.erase(itr);
    this->Internals->InvalidateRenderBundle();
    this->Modified();
  }
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::RemoveAllVertexAttributeMappings()
{
  if (this->ExtraAttributes.empty())
  {
    return;
  }
  this->ExtraAttributes.clear();
  this->Internals->InvalidateRenderBundle();
  this->Modified();
}

//------------------------------------------------------------------------------
// 8C: Render bundle — replay cached encoder commands on the current render encoder.
// Uniform buffers (scene, material, light, etc.) are updated in-place each frame,
// so replaying the same buffer bindings reads the latest content automatically.
void vtkMetalPolyDataMapper::ReplayRenderBundle(void* mtlEncoder)
{
  id<MTLRenderCommandEncoder> encoder = (__bridge id<MTLRenderCommandEncoder>)mtlEncoder;
  using Cmd = vtkMetalPolyDataMapperInternals::RenderBundleDrawCommand;
  for (const auto& cmd : this->Internals->Bundle.Commands)
  {
    switch (cmd.type)
    {
      case Cmd::SetPipelineState:
        [encoder setRenderPipelineState:std::get<Cmd::SetPipelineStateParams>(cmd.params).pipeline];
        break;
      case Cmd::SetVertexBuffer:
      {
        const auto& p = std::get<Cmd::SetBufferParams>(cmd.params);
        [encoder setVertexBuffer:p.buffer offset:p.offset atIndex:p.index];
        break;
      }
      case Cmd::SetFragmentBuffer:
      {
        const auto& p = std::get<Cmd::SetBufferParams>(cmd.params);
        [encoder setFragmentBuffer:p.buffer offset:p.offset atIndex:p.index];
        break;
      }
      case Cmd::SetFragmentTexture:
      {
        const auto& p = std::get<Cmd::SetTextureParams>(cmd.params);
        [encoder setFragmentTexture:p.texture atIndex:p.index];
        break;
      }
      case Cmd::SetFragmentSamplerState:
      {
        const auto& p = std::get<Cmd::SetSamplerParams>(cmd.params);
        [encoder setFragmentSamplerState:p.sampler atIndex:p.index];
        break;
      }
      case Cmd::SetCullMode:
        [encoder setCullMode:std::get<Cmd::SetCullModeParams>(cmd.params).mode];
        break;
      case Cmd::DrawPrimitives:
      {
        const auto& p = std::get<Cmd::DrawPrimitivesParams>(cmd.params);
        if (p.instanceCount > 0)
        {
          [encoder drawPrimitives:p.primitiveType
                      vertexStart:p.vertexStart
                    vertexCount:p.vertexCount
                  instanceCount:p.instanceCount];
        }
        else
        {
          [encoder drawPrimitives:p.primitiveType
                      vertexStart:p.vertexStart
                    vertexCount:p.vertexCount];
        }
        break;
      }
      case Cmd::DrawIndexedPrimitives:
      {
        const auto& p = std::get<Cmd::DrawIndexedPrimitivesParams>(cmd.params);
        [encoder drawIndexedPrimitives:p.primitiveType
                            indexCount:p.indexCount
                             indexType:p.indexType
                           indexBuffer:p.indexBuffer
                     indexBufferOffset:p.indexBufferOffset];
        break;
      }
    }
  }
}

//------------------------------------------------------------------------------
// 8C: Render bundle — rebuild by recording all geometry-related encoder commands.
// This captures pipeline states, buffer bindings, and draw calls into a cache
// that can be replayed on subsequent frames when geometry hasn't changed.
void vtkMetalPolyDataMapper::RebuildRenderBundle(
  void* mtlEncoder, vtkRenderer* ren, vtkActor* act)
{
  id<MTLRenderCommandEncoder> encoder = (__bridge id<MTLRenderCommandEncoder>)mtlEncoder;
  using Cmd = vtkMetalPolyDataMapperInternals::RenderBundleDrawCommand;
  using PSParams = Cmd::SetPipelineStateParams;
  using BufParams = Cmd::SetBufferParams;
  using TexParams = Cmd::SetTextureParams;
  using SampParams = Cmd::SetSamplerParams;
  using CullParams = Cmd::SetCullModeParams;
  using DrawParams = Cmd::DrawPrimitivesParams;
  using IdxParams = Cmd::DrawIndexedPrimitivesParams;

  auto& commands = this->Internals->Bundle.Commands;
  commands.clear();

  // Helper lambdas to record encoder commands into the bundle
  auto recordPipeline = [&commands](id<MTLRenderPipelineState> pipeline) {
    Cmd cmd;
    cmd.type = Cmd::SetPipelineState;
    cmd.params = PSParams{ pipeline };
    commands.push_back(cmd);
  };
  auto recordVBuf = [&commands](id<MTLBuffer> buffer, NSUInteger offset, NSUInteger index) {
    Cmd cmd;
    cmd.type = Cmd::SetVertexBuffer;
    cmd.params = BufParams{ buffer, offset, index };
    commands.push_back(cmd);
  };
  auto recordFBuf = [&commands](id<MTLBuffer> buffer, NSUInteger offset, NSUInteger index) {
    Cmd cmd;
    cmd.type = Cmd::SetFragmentBuffer;
    cmd.params = BufParams{ buffer, offset, index };
    commands.push_back(cmd);
  };
  auto recordFTex = [&commands](id<MTLTexture> texture, NSUInteger index) {
    Cmd cmd;
    cmd.type = Cmd::SetFragmentTexture;
    cmd.params = TexParams{ texture, index };
    commands.push_back(cmd);
  };
  auto recordFSamp = [&commands](id<MTLSamplerState> sampler, NSUInteger index) {
    Cmd cmd;
    cmd.type = Cmd::SetFragmentSamplerState;
    cmd.params = SampParams{ sampler, index };
    commands.push_back(cmd);
  };
  auto recordCull = [&commands](MTLCullMode mode) {
    Cmd cmd;
    cmd.type = Cmd::SetCullMode;
    cmd.params = CullParams{ mode };
    commands.push_back(cmd);
  };
  auto recordDraw = [&commands](MTLPrimitiveType ptype, NSUInteger vstart, NSUInteger vcount,
                        NSUInteger icount = 0) {
    Cmd cmd;
    cmd.type = Cmd::DrawPrimitives;
    cmd.params = DrawParams{ ptype, vstart, vcount, icount };
    commands.push_back(cmd);
  };
  auto recordIdxDraw = [&commands](MTLPrimitiveType ptype, NSUInteger indexCount, MTLIndexType itype,
                           id<MTLBuffer> ibuf, NSUInteger offset) {
    Cmd cmd;
    cmd.type = Cmd::DrawIndexedPrimitives;
    cmd.params = IdxParams{ ptype, indexCount, itype, ibuf, offset };
    commands.push_back(cmd);
  };

  int representation = act->GetProperty()->GetRepresentation();
  float lineWidth = static_cast<float>(act->GetProperty()->GetLineWidth());
  bool skipTriangles = (representation == VTK_WIREFRAME);
  int peelMode = vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow())->DepthPeelingMode;

  // --- Triangle drawing ---
  if (!skipTriangles && this->Internals->HasTriangles && this->Internals->TrianglePipeline)
  {
    if (peelMode == 1 && this->Internals->TriangleInitPeelPipeline)
    {
      recordPipeline(this->Internals->TriangleInitPeelPipeline);
    }
    else if (peelMode == 2 && this->Internals->TrianglePeelPipeline)
    {
      recordPipeline(this->Internals->TrianglePeelPipeline);
    }
    else
    {
      recordPipeline(this->Internals->TrianglePipeline);
    }
    recordVBuf(this->Internals->VertexPositionBuffer, 0, 0);
    if (this->Internals->VertexNormalBuffer)
    {
      recordVBuf(this->Internals->VertexNormalBuffer, 0, 1);
    }
    if (this->Internals->SurfaceColorBuffer)
    {
      recordVBuf(this->Internals->SurfaceColorBuffer, 0, 3);
    }
    if (this->Internals->MaterialUniformBuffer)
    {
      recordFBuf(this->Internals->MaterialUniformBuffer, 0, 0);
    }
    if (this->Internals->LightUniformBuffer)
    {
      recordFBuf(this->Internals->LightUniformBuffer, 0, 1);
    }
    if (this->Internals->SceneUniformBuffer)
    {
      recordVBuf(this->Internals->SceneUniformBuffer, 0, 2);
      recordFBuf(this->Internals->SceneUniformBuffer, 0, 2);
    }
    if (this->Internals->CoincidentOffsetBuffer)
    {
      recordFBuf(this->Internals->CoincidentOffsetBuffer, 0, 3);
    }
    if (this->Internals->ClipPlaneBuffer)
    {
      recordFBuf(this->Internals->ClipPlaneBuffer, 0, 5);
      recordVBuf(this->Internals->ClipPlaneBuffer, 0, 5);
    }
    if (this->Internals->TriangleCellIdBuffer)
    {
      recordVBuf(this->Internals->TriangleCellIdBuffer, 0, 6);
    }
    if (this->Internals->PropIdBuffer)
    {
      recordVBuf(this->Internals->PropIdBuffer, 0, 7);
    }
    if (this->Internals->TriangleUVBuffer)
    {
      recordVBuf(this->Internals->TriangleUVBuffer, 0, 8);
    }
    // 8D: Bind extra attribute buffers at buffer indices 16+
    {
      NSUInteger extraIdx = 16;
      for (auto& eab : this->Internals->ExtraAttributeBuffers)
      {
        if (eab.second)
        {
          recordVBuf(eab.second, 0, extraIdx);
        }
        extraIdx++;
      }
    }
    {
      id<MTLTexture> texToBind = this->Internals->ActorTexture;
      id<MTLSamplerState> samplerToBind = this->Internals->ActorSampler;
      if (!texToBind)
      {
        texToBind = this->Internals->DefaultTexture;
        samplerToBind = this->Internals->DefaultSampler;
      }
      if (texToBind)
      {
        recordFTex(texToBind, 0);
      }
      if (samplerToBind)
      {
        recordFSamp(samplerToBind, 0);
      }
    }
    if (act->GetProperty()->GetBackfaceCulling())
    {
      recordCull(MTLCullModeBack);
    }
    else if (act->GetProperty()->GetFrontfaceCulling())
    {
      recordCull(MTLCullModeFront);
    }
    else
    {
      recordCull(MTLCullModeNone);
    }
    if (this->Internals->IndexBuffer)
    {
      recordIdxDraw(MTLPrimitiveTypeTriangle, this->Internals->TriangleIndexCount, MTLIndexTypeUInt32,
        this->Internals->IndexBuffer, 0);
    }
    else
    {
      recordDraw(MTLPrimitiveTypeTriangle, 0, this->Internals->TriangleVertexCount);
    }
  }

  // --- Line drawing ---
  if (this->Internals->HasLines && this->Internals->LineIndexBuffer)
  {
    auto lineJoinType = act->GetProperty()->GetLineJoin();
    bool useRoundCapLines = false;
    bool useMiterJoinLines = false;
    bool useThickLines = false;

    if (lineWidth > 1.0f)
    {
      if (lineJoinType == vtkProperty::LineJoinType::RoundCapRoundJoin &&
          this->Internals->RoundCapLineSegmentCount > 0 && this->Internals->RoundCapLinePipeline)
      {
        useRoundCapLines = true;
      }
      else if (lineJoinType == vtkProperty::LineJoinType::MiterJoin &&
               this->Internals->MiterJoinLineSegmentCount > 0 && this->Internals->MiterJoinLinePipeline)
      {
        useMiterJoinLines = true;
      }
      else if (lineJoinType == vtkProperty::LineJoinType::NoJoin &&
               this->Internals->ThickLineSegmentCount > 0 && this->Internals->ThickLinePipeline)
      {
        useThickLines = true;
      }
    }

    if (useRoundCapLines)
    {
      recordPipeline(this->Internals->RoundCapLinePipeline);
      recordVBuf(this->Internals->VertexPositionBuffer, 0, 0);
      recordVBuf(this->Internals->LineIndexBuffer, 0, 1);
      if (this->Internals->SceneUniformBuffer)
      {
        recordVBuf(this->Internals->SceneUniformBuffer, 0, 2);
      }
      if (this->Internals->SurfaceColorBuffer)
      {
        recordVBuf(this->Internals->SurfaceColorBuffer, 0, 3);
      }
      if (this->Internals->ThickLineLineWidthBuffer)
      {
        recordVBuf(this->Internals->ThickLineLineWidthBuffer, 0, 4);
      }
      if (this->Internals->LineCellIdBuffer)
      {
        recordVBuf(this->Internals->LineCellIdBuffer, 0, 5);
      }
      if (this->Internals->PropIdBuffer)
      {
        recordVBuf(this->Internals->PropIdBuffer, 0, 6);
      }
      if (this->Internals->MaterialUniformBuffer)
      {
        recordFBuf(this->Internals->MaterialUniformBuffer, 0, 0);
      }
      if (this->Internals->LightUniformBuffer)
      {
        recordFBuf(this->Internals->LightUniformBuffer, 0, 1);
      }
      if (this->Internals->SceneUniformBuffer)
      {
        recordFBuf(this->Internals->SceneUniformBuffer, 0, 2);
      }
      if (this->Internals->CoincidentOffsetBuffer)
      {
        recordFBuf(this->Internals->CoincidentOffsetBuffer, 0, 3);
      }
      recordCull(MTLCullModeNone);
      recordDraw(MTLPrimitiveTypeTriangleStrip, 0, 36, this->Internals->RoundCapLineSegmentCount);
    }
    else if (useMiterJoinLines)
    {
      recordPipeline(this->Internals->MiterJoinLinePipeline);
      recordVBuf(this->Internals->VertexPositionBuffer, 0, 0);
      recordVBuf(this->Internals->LineIndexBuffer, 0, 1);
      if (this->Internals->SceneUniformBuffer)
      {
        recordVBuf(this->Internals->SceneUniformBuffer, 0, 2);
      }
      if (this->Internals->SurfaceColorBuffer)
      {
        recordVBuf(this->Internals->SurfaceColorBuffer, 0, 3);
      }
      if (this->Internals->ThickLineLineWidthBuffer)
      {
        recordVBuf(this->Internals->ThickLineLineWidthBuffer, 0, 4);
      }
      if (this->Internals->LineCellIdBuffer)
      {
        recordVBuf(this->Internals->LineCellIdBuffer, 0, 5);
      }
      if (this->Internals->PropIdBuffer)
      {
        recordVBuf(this->Internals->PropIdBuffer, 0, 6);
      }
      if (this->Internals->MiterJoinSegmentCountBuffer)
      {
        recordVBuf(this->Internals->MiterJoinSegmentCountBuffer, 0, 7);
      }
      if (this->Internals->MaterialUniformBuffer)
      {
        recordFBuf(this->Internals->MaterialUniformBuffer, 0, 0);
      }
      if (this->Internals->LightUniformBuffer)
      {
        recordFBuf(this->Internals->LightUniformBuffer, 0, 1);
      }
      if (this->Internals->SceneUniformBuffer)
      {
        recordFBuf(this->Internals->SceneUniformBuffer, 0, 2);
      }
      if (this->Internals->CoincidentOffsetBuffer)
      {
        recordFBuf(this->Internals->CoincidentOffsetBuffer, 0, 3);
      }
      recordCull(MTLCullModeNone);
      recordDraw(MTLPrimitiveTypeTriangleStrip, 0, 4, this->Internals->MiterJoinLineSegmentCount);
    }
    else if (useThickLines)
    {
      recordPipeline(this->Internals->ThickLinePipeline);
      recordVBuf(this->Internals->VertexPositionBuffer, 0, 0);
      recordVBuf(this->Internals->LineIndexBuffer, 0, 1);
      if (this->Internals->SceneUniformBuffer)
      {
        recordVBuf(this->Internals->SceneUniformBuffer, 0, 2);
      }
      if (this->Internals->SurfaceColorBuffer)
      {
        recordVBuf(this->Internals->SurfaceColorBuffer, 0, 3);
      }
      if (this->Internals->ThickLineLineWidthBuffer)
      {
        recordVBuf(this->Internals->ThickLineLineWidthBuffer, 0, 4);
      }
      if (this->Internals->LineCellIdBuffer)
      {
        recordVBuf(this->Internals->LineCellIdBuffer, 0, 5);
      }
      if (this->Internals->PropIdBuffer)
      {
        recordVBuf(this->Internals->PropIdBuffer, 0, 6);
      }
      if (this->Internals->MaterialUniformBuffer)
      {
        recordFBuf(this->Internals->MaterialUniformBuffer, 0, 0);
      }
      if (this->Internals->LightUniformBuffer)
      {
        recordFBuf(this->Internals->LightUniformBuffer, 0, 1);
      }
      if (this->Internals->SceneUniformBuffer)
      {
        recordFBuf(this->Internals->SceneUniformBuffer, 0, 2);
      }
      if (this->Internals->CoincidentOffsetBuffer)
      {
        recordFBuf(this->Internals->CoincidentOffsetBuffer, 0, 3);
      }
      recordCull(MTLCullModeNone);
      recordDraw(MTLPrimitiveTypeTriangleStrip, 0, 4, this->Internals->ThickLineSegmentCount);
    }
    else
    {
      // Standard 1px lines
      recordPipeline(this->Internals->LinePipeline);
      recordVBuf(this->Internals->VertexPositionBuffer, 0, 0);
      if (this->Internals->VertexNormalBuffer)
      {
        recordVBuf(this->Internals->VertexNormalBuffer, 0, 1);
      }
      if (this->Internals->SurfaceColorBuffer)
      {
        recordVBuf(this->Internals->SurfaceColorBuffer, 0, 3);
      }
      if (this->Internals->MaterialUniformBuffer)
      {
        recordFBuf(this->Internals->MaterialUniformBuffer, 0, 0);
      }
      if (this->Internals->LightUniformBuffer)
      {
        recordFBuf(this->Internals->LightUniformBuffer, 0, 1);
      }
      if (this->Internals->SceneUniformBuffer)
      {
        recordVBuf(this->Internals->SceneUniformBuffer, 0, 2);
        recordFBuf(this->Internals->SceneUniformBuffer, 0, 2);
      }
      if (this->Internals->CoincidentOffsetBuffer)
      {
        recordFBuf(this->Internals->CoincidentOffsetBuffer, 0, 3);
      }
      if (this->Internals->LineCellIdBuffer)
      {
        recordVBuf(this->Internals->LineCellIdBuffer, 0, 6);
      }
      if (this->Internals->PropIdBuffer)
      {
        recordVBuf(this->Internals->PropIdBuffer, 0, 7);
      }
      if (this->Internals->TriangleUVBuffer)
      {
        recordVBuf(this->Internals->TriangleUVBuffer, 0, 8);
      }
      {
        id<MTLTexture> texToBind = this->Internals->ActorTexture;
        id<MTLSamplerState> samplerToBind = this->Internals->ActorSampler;
        if (!texToBind)
        {
          texToBind = this->Internals->DefaultTexture;
          samplerToBind = this->Internals->DefaultSampler;
        }
        if (texToBind)
        {
          recordFTex(texToBind, 0);
        }
        if (samplerToBind)
        {
          recordFSamp(samplerToBind, 0);
        }
      }
      recordCull(MTLCullModeNone);
      recordIdxDraw(MTLPrimitiveTypeLine, this->Internals->LineIndexCount, MTLIndexTypeUInt32,
        this->Internals->LineIndexBuffer, 0);
    }
  }

  // --- Edge overlay (wireframe on surface) ---
  if (representation == VTK_SURFACE && act->GetProperty()->GetEdgeVisibility() &&
      this->Internals->HasEdgeOverlay && this->Internals->EdgePipeline &&
      this->Internals->EdgeIndexBuffer)
  {
    recordPipeline(this->Internals->EdgePipeline);
    recordVBuf(this->Internals->EdgeVertexPositionBuffer, 0, 0);
    if (this->Internals->EdgeVertexNormalBuffer)
    {
      recordVBuf(this->Internals->EdgeVertexNormalBuffer, 0, 1);
    }
    if (this->Internals->EdgeSurfaceColorBuffer)
    {
      recordVBuf(this->Internals->EdgeSurfaceColorBuffer, 0, 3);
    }
    if (this->Internals->SceneUniformBuffer)
    {
      recordVBuf(this->Internals->SceneUniformBuffer, 0, 2);
      recordFBuf(this->Internals->SceneUniformBuffer, 0, 2);
    }
    if (this->Internals->CoincidentOffsetBuffer)
    {
      recordFBuf(this->Internals->CoincidentOffsetBuffer, 0, 3);
    }
    if (this->Internals->EdgeColorUniformBuffer)
    {
      recordFBuf(this->Internals->EdgeColorUniformBuffer, 0, 4);
    }
    if (this->Internals->MaterialUniformBuffer)
    {
      recordFBuf(this->Internals->MaterialUniformBuffer, 0, 0);
    }
    if (this->Internals->LineCellIdBuffer)
    {
      recordVBuf(this->Internals->LineCellIdBuffer, 0, 6);
    }
    if (this->Internals->PropIdBuffer)
    {
      recordVBuf(this->Internals->PropIdBuffer, 0, 7);
    }
    recordCull(MTLCullModeNone);
    recordIdxDraw(MTLPrimitiveTypeLine, this->Internals->EdgeIndexCount, MTLIndexTypeUInt32,
      this->Internals->EdgeIndexBuffer, 0);
  }

  // --- Vertex visibility (dots on surface) ---
  if (representation != VTK_POINTS && act->GetProperty()->GetVertexVisibility() &&
      this->Internals->PointVertexCount > 0 && this->Internals->PointPositionBuffer)
  {
    float ptSize = act->GetProperty()->GetPointSize();
    if (ptSize > 1.0f && this->Internals->PointShapedPipeline)
    {
      recordPipeline(this->Internals->PointShapedPipeline);
      recordVBuf(this->Internals->PointPositionBuffer, 0, 0);
      recordVBuf(this->Internals->PointConnectivityBuffer, 0, 1);
      if (this->Internals->SceneUniformBuffer)
      {
        recordVBuf(this->Internals->SceneUniformBuffer, 0, 2);
      }
      if (this->Internals->PointNormalBuffer)
      {
        recordVBuf(this->Internals->PointNormalBuffer, 0, 3);
      }
      if (this->Internals->PointColorBuffer)
      {
        recordVBuf(this->Internals->PointColorBuffer, 0, 4);
      }
      if (this->Internals->PointTangentBuffer)
      {
        recordVBuf(this->Internals->PointTangentBuffer, 0, 6);
      }
      if (this->Internals->PointUVBuffer)
      {
        recordVBuf(this->Internals->PointUVBuffer, 0, 7);
      }
      if (this->Internals->PointColorUVBuffer)
      {
        recordVBuf(this->Internals->PointColorUVBuffer, 0, 8);
      }
      if (this->Internals->CellIdOffsetBuffer)
      {
        recordVBuf(this->Internals->CellIdOffsetBuffer, 0, 9);
      }
      if (this->Internals->PointCellIdBuffer)
      {
        recordVBuf(this->Internals->PointCellIdBuffer, 0, 11);
      }
      if (this->Internals->PropIdBuffer)
      {
        recordVBuf(this->Internals->PropIdBuffer, 0, 12);
      }
      if (this->Internals->MaterialUniformBuffer)
      {
        recordFBuf(this->Internals->MaterialUniformBuffer, 0, 0);
      }
      if (this->Internals->LightUniformBuffer)
      {
        recordFBuf(this->Internals->LightUniformBuffer, 0, 1);
      }
      if (this->Internals->SceneUniformBuffer)
      {
        recordFBuf(this->Internals->SceneUniformBuffer, 0, 2);
      }
      if (this->Internals->CoincidentOffsetBuffer)
      {
        recordFBuf(this->Internals->CoincidentOffsetBuffer, 0, 3);
      }
      if (this->Internals->VertexColorBuffer)
      {
        recordFBuf(this->Internals->VertexColorBuffer, 0, 4);
      }
      recordDraw(MTLPrimitiveTypeTriangleStrip, 0, 4, this->Internals->PointVertexCount);
    }
    else if (this->Internals->PointPipeline)
    {
      recordPipeline(this->Internals->PointPipeline);
      recordVBuf(this->Internals->PointPositionBuffer, 0, 0);
      if (this->Internals->SceneUniformBuffer)
      {
        recordVBuf(this->Internals->SceneUniformBuffer, 0, 1);
      }
      if (this->Internals->PointNormalBuffer)
      {
        recordVBuf(this->Internals->PointNormalBuffer, 0, 2);
      }
      if (this->Internals->PointColorBuffer)
      {
        recordVBuf(this->Internals->PointColorBuffer, 0, 3);
      }
      if (this->Internals->PointTangentBuffer)
      {
        recordVBuf(this->Internals->PointTangentBuffer, 0, 6);
      }
      if (this->Internals->PointUVBuffer)
      {
        recordVBuf(this->Internals->PointUVBuffer, 0, 7);
      }
      if (this->Internals->PointColorUVBuffer)
      {
        recordVBuf(this->Internals->PointColorUVBuffer, 0, 8);
      }
      if (this->Internals->CellIdOffsetBuffer)
      {
        recordVBuf(this->Internals->CellIdOffsetBuffer, 0, 9);
      }
      if (this->Internals->PointCellIdBuffer)
      {
        recordVBuf(this->Internals->PointCellIdBuffer, 0, 11);
      }
      if (this->Internals->PropIdBuffer)
      {
        recordVBuf(this->Internals->PropIdBuffer, 0, 12);
      }
      if (this->Internals->MaterialUniformBuffer)
      {
        recordFBuf(this->Internals->MaterialUniformBuffer, 0, 0);
      }
      if (this->Internals->LightUniformBuffer)
      {
        recordFBuf(this->Internals->LightUniformBuffer, 0, 1);
      }
      if (this->Internals->SceneUniformBuffer)
      {
        recordFBuf(this->Internals->SceneUniformBuffer, 0, 2);
      }
      if (this->Internals->CoincidentOffsetBuffer)
      {
        recordFBuf(this->Internals->CoincidentOffsetBuffer, 0, 3);
      }
      if (this->Internals->VertexColorBuffer)
      {
        recordFBuf(this->Internals->VertexColorBuffer, 0, 4);
      }
      recordDraw(MTLPrimitiveTypePoint, 0, this->Internals->PointVertexCount);
    }
  }

  // --- Points (VTK_POINTS representation) ---
  if (representation == VTK_POINTS && this->Internals->PointVertexCount > 0 &&
      this->Internals->PointPositionBuffer)
  {
    float ptSize = act->GetProperty()->GetPointSize();
    if (ptSize > 1.0f && this->Internals->PointShapedPipeline)
    {
      recordPipeline(this->Internals->PointShapedPipeline);
      recordVBuf(this->Internals->PointPositionBuffer, 0, 0);
      recordVBuf(this->Internals->PointConnectivityBuffer, 0, 1);
      if (this->Internals->SceneUniformBuffer)
      {
        recordVBuf(this->Internals->SceneUniformBuffer, 0, 2);
      }
      if (this->Internals->PointNormalBuffer)
      {
        recordVBuf(this->Internals->PointNormalBuffer, 0, 3);
      }
      if (this->Internals->PointColorBuffer)
      {
        recordVBuf(this->Internals->PointColorBuffer, 0, 4);
      }
      if (this->Internals->PointTangentBuffer)
      {
        recordVBuf(this->Internals->PointTangentBuffer, 0, 6);
      }
      if (this->Internals->PointUVBuffer)
      {
        recordVBuf(this->Internals->PointUVBuffer, 0, 7);
      }
      if (this->Internals->PointColorUVBuffer)
      {
        recordVBuf(this->Internals->PointColorUVBuffer, 0, 8);
      }
      if (this->Internals->CellIdOffsetBuffer)
      {
        recordVBuf(this->Internals->CellIdOffsetBuffer, 0, 9);
      }
      if (this->Internals->PointCellIdBuffer)
      {
        recordVBuf(this->Internals->PointCellIdBuffer, 0, 11);
      }
      if (this->Internals->PropIdBuffer)
      {
        recordVBuf(this->Internals->PropIdBuffer, 0, 12);
      }
      if (this->Internals->MaterialUniformBuffer)
      {
        recordFBuf(this->Internals->MaterialUniformBuffer, 0, 0);
      }
      if (this->Internals->LightUniformBuffer)
      {
        recordFBuf(this->Internals->LightUniformBuffer, 0, 1);
      }
      if (this->Internals->SceneUniformBuffer)
      {
        recordFBuf(this->Internals->SceneUniformBuffer, 0, 2);
      }
      if (this->Internals->CoincidentOffsetBuffer)
      {
        recordFBuf(this->Internals->CoincidentOffsetBuffer, 0, 3);
      }
      if (this->Internals->VertexColorBuffer)
      {
        recordFBuf(this->Internals->VertexColorBuffer, 0, 4);
      }
      recordDraw(MTLPrimitiveTypeTriangleStrip, 0, 4, this->Internals->PointVertexCount);
    }
    else if (this->Internals->PointPipeline)
    {
      recordPipeline(this->Internals->PointPipeline);
      recordVBuf(this->Internals->PointPositionBuffer, 0, 0);
      if (this->Internals->SceneUniformBuffer)
      {
        recordVBuf(this->Internals->SceneUniformBuffer, 0, 1);
      }
      if (this->Internals->PointNormalBuffer)
      {
        recordVBuf(this->Internals->PointNormalBuffer, 0, 2);
      }
      if (this->Internals->PointColorBuffer)
      {
        recordVBuf(this->Internals->PointColorBuffer, 0, 3);
      }
      if (this->Internals->PointTangentBuffer)
      {
        recordVBuf(this->Internals->PointTangentBuffer, 0, 6);
      }
      if (this->Internals->PointUVBuffer)
      {
        recordVBuf(this->Internals->PointUVBuffer, 0, 7);
      }
      if (this->Internals->PointColorUVBuffer)
      {
        recordVBuf(this->Internals->PointColorUVBuffer, 0, 8);
      }
      if (this->Internals->CellIdOffsetBuffer)
      {
        recordVBuf(this->Internals->CellIdOffsetBuffer, 0, 9);
      }
      if (this->Internals->PointCellIdBuffer)
      {
        recordVBuf(this->Internals->PointCellIdBuffer, 0, 11);
      }
      if (this->Internals->PropIdBuffer)
      {
        recordVBuf(this->Internals->PropIdBuffer, 0, 12);
      }
      if (this->Internals->MaterialUniformBuffer)
      {
        recordFBuf(this->Internals->MaterialUniformBuffer, 0, 0);
      }
      if (this->Internals->LightUniformBuffer)
      {
        recordFBuf(this->Internals->LightUniformBuffer, 0, 1);
      }
      if (this->Internals->SceneUniformBuffer)
      {
        recordFBuf(this->Internals->SceneUniformBuffer, 0, 2);
      }
      if (this->Internals->CoincidentOffsetBuffer)
      {
        recordFBuf(this->Internals->CoincidentOffsetBuffer, 0, 3);
      }
      if (this->Internals->VertexColorBuffer)
      {
        recordFBuf(this->Internals->VertexColorBuffer, 0, 4);
      }
      recordDraw(MTLPrimitiveTypePoint, 0, this->Internals->PointVertexCount);
    }
  }

  // Mark bundle as valid and record the state that was used to create it
  this->Internals->Bundle.Valid = true;
  this->Internals->BundleGeometryMTime = this->Internals->CachedInputMTime;
  this->Internals->BundleRepresentation = representation;
  this->Internals->BundleEdgeVisibility = this->Internals->CachedEdgeVisibility;
  this->Internals->BundleLineWidth = lineWidth;
  this->Internals->BundleSampleCount = this->Internals->CachedSampleCount;
  this->Internals->BundlePeelMode = peelMode;
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

  // 8A: Invalidate all pipeline states when MSAA sample count changes
  int currentSampleCount = renWin->GetEffectiveSampleCount();
  if (currentSampleCount != this->Internals->CachedSampleCount)
  {
    this->Internals->TrianglePipeline = nil;
    this->Internals->LinePipeline = nil;
    this->Internals->PointPipeline = nil;
    this->Internals->PointShapedPipeline = nil;
    this->Internals->EdgePipeline = nil;
    this->Internals->ThickLinePipeline = nil;
    this->Internals->RoundCapLinePipeline = nil;
    this->Internals->MiterJoinLinePipeline = nil;
    this->Internals->TriangleInitPeelPipeline = nil;
    this->Internals->TrianglePeelPipeline = nil;
    this->Internals->CachedSampleCount = currentSampleCount;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)renWin->GetMetalDevice();

    vtkIdType currentMTime = input->GetMTime();
    int representation = act->GetProperty()->GetRepresentation();
    bool edgeVisibility = act->GetProperty()->GetEdgeVisibility();
    float lineWidth = static_cast<float>(act->GetProperty()->GetLineWidth());
    if (currentMTime != this->Internals->CachedInputMTime ||
        representation != this->Internals->CachedRepresentation ||
        edgeVisibility != this->Internals->CachedEdgeVisibility)
    {
      this->Internals->ReleaseBuffers();
      this->Internals->CachedInputMTime = currentMTime;
      this->Internals->CachedRepresentation = representation;
      this->Internals->CachedEdgeVisibility = edgeVisibility;
      this->BuildGeometryBuffers(
        (__bridge void*)device, renWin->GetMetalQueue(), input, act);
    }

    // P3-3A: Track line width changes (buffer updated at draw time)
    this->Internals->CachedLineWidth = lineWidth;

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

    // 8C: Ensure all pipeline states are created before render bundle recording.
    // These were previously created lazily inside inline draw code.
    if (representation != VTK_POINTS && representation != VTK_WIREFRAME &&
        this->Internals->HasEdgeOverlay)
    {
      this->EnsureEdgePipelineState((__bridge void*)device);
    }
    if (this->Internals->HasLines && this->Internals->LineIndexBuffer)
    {
      float lw = static_cast<float>(act->GetProperty()->GetLineWidth());
      if (lw > 1.0f)
      {
        auto lj = act->GetProperty()->GetLineJoin();
        if (lj == vtkProperty::LineJoinType::RoundCapRoundJoin &&
            this->Internals->RoundCapLineSegmentCount > 0)
        {
          this->EnsureRoundCapLinePipelineState((__bridge void*)device);
        }
        else if (lj == vtkProperty::LineJoinType::MiterJoin &&
                 this->Internals->MiterJoinLineSegmentCount > 0)
        {
          this->EnsureMiterJoinLinePipelineState((__bridge void*)device);
        }
        else if (lj == vtkProperty::LineJoinType::NoJoin &&
                 this->Internals->ThickLineSegmentCount > 0)
        {
          this->EnsureThickLinePipelineState((__bridge void*)device);
        }
      }
    }
    if (renWin->DepthPeelingMode != 0)
    {
      this->EnsurePeelPipelineStates((__bridge void*)device);
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
    this->UpdateClipPlaneUniforms((__bridge void*)device, act);
    this->UpdateActorTexture((__bridge void*)device, act);

    // P5-5A: Set texture flag (bit 9) in scene uniforms when actor has a texture
    if (this->Internals->ActorTexture)
    {
      char* buf = static_cast<char*>([this->Internals->SceneUniformBuffer contents]);
      *reinterpret_cast<uint32_t*>(buf + 256) |= (1u << 9);
    }

    // 8C: Update edge color uniform before bundle recording (per-frame update)
    if (representation == VTK_SURFACE && edgeVisibility && this->Internals->HasEdgeOverlay &&
        this->Internals->EdgeColorUniformBuffer)
    {
      this->UpdateEdgeColorUniform((__bridge void*)device, act);
    }

    // 8C: Ensure default texture and sampler exist before bundle recording.
    // These are used as fallbacks when the actor has no texture, and the bundle
    // needs valid objects at recording time.
    if (!this->Internals->DefaultTexture)
    {
      MTLTextureDescriptor* desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                   width:1
                                  height:1
                               mipmapped:NO];
      desc.usage = MTLTextureUsageShaderRead;
      desc.storageMode = MTLStorageModeShared;
      this->Internals->DefaultTexture = [device newTextureWithDescriptor:desc];
      unsigned char white[4] = { 255, 255, 255, 255 };
      MTLRegion region = MTLRegionMake2D(0, 0, 1, 1);
      [this->Internals->DefaultTexture replaceRegion:region
                                         mipmapLevel:0
                                           withBytes:white
                                         bytesPerRow:4];
    }
    if (!this->Internals->DefaultSampler)
    {
      MTLSamplerDescriptor* sDesc = [[MTLSamplerDescriptor alloc] init];
      sDesc.minFilter = MTLSamplerMinMagFilterLinear;
      sDesc.magFilter = MTLSamplerMinMagFilterLinear;
      this->Internals->DefaultSampler = [device newSamplerStateWithDescriptor:sDesc];
    }

    // 8C: Render bundle — check if cached encoder commands can be replayed.
    // Bundle is valid when geometry, representation, edge visibility, line width,
    // MSAA sample count, and depth peeling mode all match the values at bundle creation.
    // When valid, replay cached commands instead of re-encoding all draw calls.
    // Uniform buffers (scene, material, light, etc.) are updated in-place each frame,
    // so replaying the same buffer bindings reads the latest content automatically.
    int currentPeelMode = renWin->DepthPeelingMode;
    bool bundleValid = this->Internals->Bundle.Valid &&
                       this->Internals->BundleGeometryMTime == this->Internals->CachedInputMTime &&
                       this->Internals->BundleRepresentation == representation &&
                       this->Internals->BundleEdgeVisibility == edgeVisibility &&
                       this->Internals->BundleLineWidth == lineWidth &&
                       this->Internals->BundleSampleCount == currentSampleCount &&
                       this->Internals->BundlePeelMode == currentPeelMode;

    if (bundleValid)
    {
      this->ReplayRenderBundle((__bridge void*)encoder);
    }
    else
    {
      this->RebuildRenderBundle((__bridge void*)encoder, ren, act);
      this->ReplayRenderBundle((__bridge void*)encoder);
    }
  }
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::BuildGeometryBuffers(void* mtlDevice, void* mtlQueue, vtkPolyData* polydata, vtkActor* actor)
{
  if (!polydata || !mtlDevice)
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<float> surfaceColors;  // P1-1A/1B: float4 per vertex
  std::vector<float> triangleUVs;    // P5-5A: float2 per vertex
  std::vector<uint32_t> lineIndices;

  // P2-2C: Triangle index buffers — deduplicated vertices + index buffer
  std::vector<uint32_t> triangleIndices;
  std::unordered_map<vtkIdType, uint32_t> triVertexMap;

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

  // P5-5A: texture coordinate array
  vtkFloatArray* tcoordArray = nullptr;
  if (pd->GetTCoords())
  {
    tcoordArray = vtkFloatArray::SafeDownCast(pd->GetTCoords());
  }

  // P1-1A/1B: MapScalars early so both triangle and line paths can use the result.
  int cellFlag = 0;
  vtkUnsignedCharArray* mappedColors = nullptr;
  if (actor)
  {
    mappedColors = this->MapScalars(actor->GetProperty()->GetOpacity(), cellFlag);
  }

  // ---- P6-6A: GPU Tessellation Path ----
  // When per-point coloring (cellFlag == 0) with data normals, use compute shaders
  // for polygon → triangle fan tessellation, edge array generation, and line segment
  // extraction. Moves fan triangulation off the CPU and produces edge arrays for free.
  // Falls back to CPU for per-cell coloring, computed normals, or small geometries.
  vtkCellArray* polys = polydata->GetPolys();
  vtkIdType numPolyPts = polydata->GetNumberOfPoints();
  bool useGPUTess = (cellFlag == 0) && normalArray && (numPolyPts > 1000);
  bool gpuTessUsed = false;

  if (useGPUTess)
  {
    // Step 1: Build per-point vertex arrays from ALL polydata points.
    // This replaces the per-triangle vertex building in the CPU path.
    positions.reserve(numPolyPts * 3);
    normals.reserve(numPolyPts * 3);
    surfaceColors.reserve(numPolyPts * 4);
    triangleUVs.reserve(numPolyPts * 2);

    for (vtkIdType i = 0; i < numPolyPts; ++i)
    {
      double pt[3];
      polydata->GetPoint(i, pt);
      positions.push_back(static_cast<float>(pt[0]));
      positions.push_back(static_cast<float>(pt[1]));
      positions.push_back(static_cast<float>(pt[2]));

      double n[3];
      normalArray->GetTuple(i, n);
      normals.push_back(static_cast<float>(n[0]));
      normals.push_back(static_cast<float>(n[1]));
      normals.push_back(static_cast<float>(n[2]));

      if (mappedColors)
      {
        const unsigned char* rgba = mappedColors->GetPointer(0);
        surfaceColors.push_back(rgba[i * 4] / 255.0f);
        surfaceColors.push_back(rgba[i * 4 + 1] / 255.0f);
        surfaceColors.push_back(rgba[i * 4 + 2] / 255.0f);
        surfaceColors.push_back(rgba[i * 4 + 3] / 255.0f);
      }
      else
      {
        surfaceColors.push_back(1.0f);
        surfaceColors.push_back(1.0f);
        surfaceColors.push_back(1.0f);
        surfaceColors.push_back(1.0f);
      }

      if (tcoordArray && tcoordArray->GetNumberOfTuples() > i)
      {
        double uv[3];
        tcoordArray->GetTuple(i, uv);
        triangleUVs.push_back(static_cast<float>(uv[0]));
        triangleUVs.push_back(static_cast<float>(uv[1]));
      }
      else
      {
        triangleUVs.push_back(0.0f);
        triangleUVs.push_back(0.0f);
      }
    }

    // Step 2: Build polygon connectivity arrays for triangle tessellation.
    if (representation != VTK_WIREFRAME && polys && polys->GetNumberOfCells() > 0)
    {
      std::vector<uint32_t> polyConn;
      std::vector<uint32_t> polyOff;
      std::vector<uint32_t> polyPrimCounts;
      polyOff.push_back(0);
      vtkIdType numTris = 0;

      vtkIdType npts;
      const vtkIdType* pts;
      polys->InitTraversal();
      while (polys->GetNextCell(npts, pts) && npts >= 3)
      {
        for (vtkIdType i = 0; i < npts; ++i)
        {
          polyConn.push_back(static_cast<uint32_t>(pts[i]));
        }
        polyOff.push_back(polyOff.back() + static_cast<uint32_t>(npts));
        polyPrimCounts.push_back(static_cast<uint32_t>(numTris));
        numTris += (npts - 2);
      }
      polyPrimCounts.push_back(static_cast<uint32_t>(numTris));

      if (numTris > 0)
      {
        // Create compute pipeline if needed
        if (!this->Internals->PolygonToTrianglePipeline)
        {
          NSString* shaderSource = [NSString stringWithUTF8String:vtkMetalShaders];
          NSError* error = nil;
          id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
          if (library)
          {
            id<MTLFunction> func = [library newFunctionWithName:@"polygonToTriangle"];
            if (func)
            {
              this->Internals->PolygonToTrianglePipeline =
                [device newComputePipelineStateWithFunction:func error:&error];
            }
          }
        }

        if (this->Internals->PolygonToTrianglePipeline)
        {
          // Upload input buffers
          id<MTLBuffer> connBuf = [device
            newBufferWithBytes:polyConn.data()
                       length:polyConn.size() * sizeof(uint32_t)
                      options:MTLResourceStorageModeShared];
          id<MTLBuffer> offBuf = [device
            newBufferWithBytes:polyOff.data()
                       length:polyOff.size() * sizeof(uint32_t)
                      options:MTLResourceStorageModeShared];
          id<MTLBuffer> primBuf = [device
            newBufferWithBytes:polyPrimCounts.data()
                       length:polyPrimCounts.size() * sizeof(uint32_t)
                      options:MTLResourceStorageModeShared];

          // Allocate output buffers
          this->Internals->TessOutputConnectivityBuffer = [device
            newBufferWithLength:numTris * 3 * sizeof(uint32_t)
                       options:MTLResourceStorageModeShared];
          this->Internals->TessEdgeArrayBuffer = [device
            newBufferWithLength:numTris * sizeof(float)
                       options:MTLResourceStorageModeShared];
          this->Internals->TriangleCellIdBuffer = [device
            newBufferWithLength:numTris * sizeof(uint32_t)
                       options:MTLResourceStorageModeShared];

          // Upload params uniform
          struct { uint32_t numCells; uint32_t cellIdOffset; } tessParams;
          tessParams.numCells = static_cast<uint32_t>(polyOff.size() - 1);
          tessParams.cellIdOffset = 0;
          this->Internals->TessParamsBuffer = [device
            newBufferWithBytes:&tessParams
                       length:sizeof(tessParams)
                      options:MTLResourceStorageModeShared];

          // Dispatch polygonToTriangle compute kernel
          id<MTLCommandBuffer> cmdBuf = [(__bridge id<MTLCommandQueue>)
            mtlQueue commandBuffer];
          id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
          [enc setComputePipelineState:this->Internals->PolygonToTrianglePipeline];
          [enc setBuffer:this->Internals->TessOutputConnectivityBuffer offset:0 atIndex:0];
          [enc setBuffer:this->Internals->TessEdgeArrayBuffer offset:0 atIndex:1];
          [enc setBuffer:this->Internals->TriangleCellIdBuffer offset:0 atIndex:2];
          [enc setBuffer:connBuf offset:0 atIndex:3];
          [enc setBuffer:offBuf offset:0 atIndex:4];
          [enc setBuffer:primBuf offset:0 atIndex:5];
          [enc setBuffer:this->Internals->TessParamsBuffer offset:0 atIndex:6];

          NSUInteger tgMax = this->Internals->PolygonToTrianglePipeline.maxTotalThreadsPerThreadgroup;
          NSUInteger gridW = tessParams.numCells;
          MTLSize grid = MTLSizeMake(gridW, 1, 1);
          MTLSize tg = MTLSizeMake(std::min(tgMax, gridW), 1, 1);
          [enc dispatchThreads:grid threadsPerThreadgroup:tg];
          [enc endEncoding];
          [cmdBuf commit];
          [cmdBuf waitUntilCompleted];

          // Use compute output as triangle index buffer
          this->Internals->IndexBuffer = this->Internals->TessOutputConnectivityBuffer;
          this->Internals->TriangleVertexCount = numPolyPts;
          this->Internals->TriangleIndexCount = numTris * 3;
          this->Internals->HasTriangles = true;
          this->Internals->TrianglePrimitiveCount = numTris;
          gpuTessUsed = true;
        }
      }

      // Step 3: Build polygon edge connectivity for edge visibility
      if (edgeVisibility)
      {
        std::vector<uint32_t> eConn;
        std::vector<uint32_t> eOff;
        std::vector<uint32_t> ePrimCounts;
        eOff.push_back(0);
        vtkIdType numEdges = 0;

        vtkIdType npts;
        const vtkIdType* pts;
        polys->InitTraversal();
        while (polys->GetNextCell(npts, pts) && npts >= 3)
        {
          for (vtkIdType i = 0; i < npts; ++i)
          {
            eConn.push_back(static_cast<uint32_t>(pts[i]));
          }
          eOff.push_back(eOff.back() + static_cast<uint32_t>(npts));
          ePrimCounts.push_back(static_cast<uint32_t>(numEdges));
          numEdges += npts;
        }
        ePrimCounts.push_back(static_cast<uint32_t>(numEdges));

        if (numEdges > 0)
        {
          if (!this->Internals->PolygonEdgesToLinesPipeline)
          {
            NSString* shaderSource = [NSString stringWithUTF8String:vtkMetalShaders];
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
            if (library)
            {
              id<MTLFunction> func = [library newFunctionWithName:@"polygonEdgesToLines"];
              if (func)
              {
                this->Internals->PolygonEdgesToLinesPipeline =
                  [device newComputePipelineStateWithFunction:func error:&error];
              }
            }
          }

          if (this->Internals->PolygonEdgesToLinesPipeline)
          {
            id<MTLBuffer> eConnBuf = [device
              newBufferWithBytes:eConn.data()
                         length:eConn.size() * sizeof(uint32_t)
                        options:MTLResourceStorageModeShared];
            id<MTLBuffer> eOffBuf = [device
              newBufferWithBytes:eOff.data()
                         length:eOff.size() * sizeof(uint32_t)
                        options:MTLResourceStorageModeShared];
            id<MTLBuffer> ePrimBuf = [device
              newBufferWithBytes:ePrimCounts.data()
                         length:ePrimCounts.size() * sizeof(uint32_t)
                        options:MTLResourceStorageModeShared];

            id<MTLBuffer> edgeOutBuf = [device
              newBufferWithLength:numEdges * 2 * sizeof(uint32_t)
                         options:MTLResourceStorageModeShared];
            id<MTLBuffer> edgeCellIdBuf = [device
              newBufferWithLength:numEdges * sizeof(uint32_t)
                         options:MTLResourceStorageModeShared];

            struct { uint32_t numCells; uint32_t cellIdOffset; } eParams;
            eParams.numCells = static_cast<uint32_t>(eOff.size() - 1);
            eParams.cellIdOffset = 0;
            id<MTLBuffer> eParamsBuf = [device
              newBufferWithBytes:&eParams
                         length:sizeof(eParams)
                        options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cmdBuf = [(__bridge id<MTLCommandQueue>)
              mtlQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:this->Internals->PolygonEdgesToLinesPipeline];
            [enc setBuffer:edgeOutBuf offset:0 atIndex:0];
            [enc setBuffer:edgeCellIdBuf offset:0 atIndex:1];
            [enc setBuffer:eConnBuf offset:0 atIndex:2];
            [enc setBuffer:eOffBuf offset:0 atIndex:3];
            [enc setBuffer:ePrimBuf offset:0 atIndex:4];
            [enc setBuffer:eParamsBuf offset:0 atIndex:5];

            NSUInteger tgMax = this->Internals->PolygonEdgesToLinesPipeline.maxTotalThreadsPerThreadgroup;
            NSUInteger gridW = eParams.numCells;
            MTLSize grid = MTLSizeMake(gridW, 1, 1);
            MTLSize tg = MTLSizeMake(std::min(tgMax, gridW), 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            // Edge overlay uses the same per-point vertex buffers as triangles
            // (assigned after buffer creation below)
            this->Internals->EdgeIndexBuffer = edgeOutBuf;
            this->Internals->EdgeIndexCount = numEdges * 2;
            this->Internals->EdgeVertexCount = numPolyPts;
            this->Internals->HasEdgeOverlay = true;
          }
        }
      }
    }
    else if (representation == VTK_WIREFRAME && polys && polys->GetNumberOfCells() > 0)
    {
      // Wireframe mode: extract polygon edges as line segments via compute
      std::vector<uint32_t> wConn;
      std::vector<uint32_t> wOff;
      std::vector<uint32_t> wPrimCounts;
      wOff.push_back(0);
      vtkIdType numEdges = 0;

      vtkIdType npts;
      const vtkIdType* pts;
      polys->InitTraversal();
      while (polys->GetNextCell(npts, pts) && npts >= 3)
      {
        for (vtkIdType i = 0; i < npts; ++i)
        {
          wConn.push_back(static_cast<uint32_t>(pts[i]));
        }
        wOff.push_back(wOff.back() + static_cast<uint32_t>(npts));
        wPrimCounts.push_back(static_cast<uint32_t>(numEdges));
        numEdges += npts;
      }
      wPrimCounts.push_back(static_cast<uint32_t>(numEdges));

      if (numEdges > 0)
      {
        if (!this->Internals->PolygonEdgesToLinesPipeline)
        {
          NSString* shaderSource = [NSString stringWithUTF8String:vtkMetalShaders];
          NSError* error = nil;
          id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
          if (library)
          {
            id<MTLFunction> func = [library newFunctionWithName:@"polygonEdgesToLines"];
            if (func)
            {
              this->Internals->PolygonEdgesToLinesPipeline =
                [device newComputePipelineStateWithFunction:func error:&error];
            }
          }
        }

        if (this->Internals->PolygonEdgesToLinesPipeline)
        {
          id<MTLBuffer> wConnBuf = [device
            newBufferWithBytes:wConn.data()
                       length:wConn.size() * sizeof(uint32_t)
                      options:MTLResourceStorageModeShared];
          id<MTLBuffer> wOffBuf = [device
            newBufferWithBytes:wOff.data()
                       length:wOff.size() * sizeof(uint32_t)
                      options:MTLResourceStorageModeShared];
          id<MTLBuffer> wPrimBuf = [device
            newBufferWithBytes:wPrimCounts.data()
                       length:wPrimCounts.size() * sizeof(uint32_t)
                      options:MTLResourceStorageModeShared];

          id<MTLBuffer> wireOutBuf = [device
            newBufferWithLength:numEdges * 2 * sizeof(uint32_t)
                       options:MTLResourceStorageModeShared];
          id<MTLBuffer> wireCellIdBuf = [device
            newBufferWithLength:numEdges * sizeof(uint32_t)
                       options:MTLResourceStorageModeShared];

          struct { uint32_t numCells; uint32_t cellIdOffset; } wParams;
          wParams.numCells = static_cast<uint32_t>(wOff.size() - 1);
          wParams.cellIdOffset = 0;
          id<MTLBuffer> wParamsBuf = [device
            newBufferWithBytes:&wParams
                       length:sizeof(wParams)
                      options:MTLResourceStorageModeShared];

          id<MTLCommandBuffer> cmdBuf = [(__bridge id<MTLCommandQueue>)
            mtlQueue commandBuffer];
          id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
          [enc setComputePipelineState:this->Internals->PolygonEdgesToLinesPipeline];
          [enc setBuffer:wireOutBuf offset:0 atIndex:0];
          [enc setBuffer:wireCellIdBuf offset:0 atIndex:1];
          [enc setBuffer:wConnBuf offset:0 atIndex:2];
          [enc setBuffer:wOffBuf offset:0 atIndex:3];
          [enc setBuffer:wPrimBuf offset:0 atIndex:4];
          [enc setBuffer:wParamsBuf offset:0 atIndex:5];

          NSUInteger tgMax = this->Internals->PolygonEdgesToLinesPipeline.maxTotalThreadsPerThreadgroup;
          NSUInteger gridW = wParams.numCells;
          MTLSize grid = MTLSizeMake(gridW, 1, 1);
          MTLSize tg = MTLSizeMake(std::min(tgMax, gridW), 1, 1);
          [enc dispatchThreads:grid threadsPerThreadgroup:tg];
          [enc endEncoding];
          [cmdBuf commit];
          [cmdBuf waitUntilCompleted];

          this->Internals->LineIndexBuffer = wireOutBuf;
          this->Internals->LineIndexCount = numEdges * 2;
          this->Internals->HasLines = true;
          this->Internals->LinePrimitiveCount = numEdges;
          this->Internals->ThickLineSegmentCount = numEdges;
          this->Internals->RoundCapLineSegmentCount = numEdges;
          this->Internals->MiterJoinLineSegmentCount = numEdges;
          this->Internals->TriangleVertexCount = 0;
          this->Internals->TriangleIndexCount = 0;
          this->Internals->HasTriangles = false;
          gpuTessUsed = true;
        }
      }
    }

    // Step 4: Build line connectivity for polyline → line segment conversion
    vtkCellArray* lines = polydata->GetLines();
    if (!gpuTessUsed || (representation == VTK_SURFACE && !this->Internals->HasLines))
    {
      if (lines && lines->GetNumberOfCells() > 0 && cellFlag == 0)
      {
        std::vector<uint32_t> lConn;
        std::vector<uint32_t> lOff;
        std::vector<uint32_t> lPrimCounts;
        lOff.push_back(0);
        vtkIdType numLineSegs = 0;

        vtkIdType npts;
        const vtkIdType* pts;
        lines->InitTraversal();
        while (lines->GetNextCell(npts, pts) && npts >= 2)
        {
          for (vtkIdType i = 0; i < npts; ++i)
          {
            lConn.push_back(static_cast<uint32_t>(pts[i]));
          }
          lOff.push_back(lOff.back() + static_cast<uint32_t>(npts));
          lPrimCounts.push_back(static_cast<uint32_t>(numLineSegs));
          numLineSegs += (npts - 1);
        }
        lPrimCounts.push_back(static_cast<uint32_t>(numLineSegs));

        if (numLineSegs > 0)
        {
          if (!this->Internals->PolyLineToLinePipeline)
          {
            NSString* shaderSource = [NSString stringWithUTF8String:vtkMetalShaders];
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
            if (library)
            {
              id<MTLFunction> func = [library newFunctionWithName:@"polyLineToLine"];
              if (func)
              {
                this->Internals->PolyLineToLinePipeline =
                  [device newComputePipelineStateWithFunction:func error:&error];
              }
            }
          }

          if (this->Internals->PolyLineToLinePipeline)
          {
            id<MTLBuffer> lConnBuf = [device
              newBufferWithBytes:lConn.data()
                         length:lConn.size() * sizeof(uint32_t)
                        options:MTLResourceStorageModeShared];
            id<MTLBuffer> lOffBuf = [device
              newBufferWithBytes:lOff.data()
                         length:lOff.size() * sizeof(uint32_t)
                        options:MTLResourceStorageModeShared];
            id<MTLBuffer> lPrimBuf = [device
              newBufferWithBytes:lPrimCounts.data()
                         length:lPrimCounts.size() * sizeof(uint32_t)
                        options:MTLResourceStorageModeShared];

            id<MTLBuffer> lineOutBuf = [device
              newBufferWithLength:numLineSegs * 2 * sizeof(uint32_t)
                         options:MTLResourceStorageModeShared];
            id<MTLBuffer> lineCellIdBuf = [device
              newBufferWithLength:numLineSegs * sizeof(uint32_t)
                         options:MTLResourceStorageModeShared];

            struct { uint32_t numCells; uint32_t cellIdOffset; } lParams;
            lParams.numCells = static_cast<uint32_t>(lOff.size() - 1);
            lParams.cellIdOffset = 0;
            id<MTLBuffer> lParamsBuf = [device
              newBufferWithBytes:&lParams
                         length:sizeof(lParams)
                        options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cmdBuf = [(__bridge id<MTLCommandQueue>)
              mtlQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:this->Internals->PolyLineToLinePipeline];
            [enc setBuffer:lineOutBuf offset:0 atIndex:0];
            [enc setBuffer:lineCellIdBuf offset:0 atIndex:1];
            [enc setBuffer:lConnBuf offset:0 atIndex:2];
            [enc setBuffer:lOffBuf offset:0 atIndex:3];
            [enc setBuffer:lPrimBuf offset:0 atIndex:4];
            [enc setBuffer:lParamsBuf offset:0 atIndex:5];

            NSUInteger tgMax = this->Internals->PolyLineToLinePipeline.maxTotalThreadsPerThreadgroup;
            NSUInteger gridW = lParams.numCells;
            MTLSize grid = MTLSizeMake(gridW, 1, 1);
            MTLSize tg = MTLSizeMake(std::min(tgMax, gridW), 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            if (!this->Internals->HasLines)
            {
              this->Internals->LineIndexBuffer = lineOutBuf;
            }
            this->Internals->LineCellIdBuffer = lineCellIdBuf;
            this->Internals->LineIndexCount += numLineSegs * 2;
            this->Internals->HasLines = true;
            this->Internals->LinePrimitiveCount += numLineSegs;
            this->Internals->ThickLineSegmentCount += numLineSegs;
            this->Internals->RoundCapLineSegmentCount += numLineSegs;
            this->Internals->MiterJoinLineSegmentCount += numLineSegs;
            gpuTessUsed = true;
          }
        }
      }
    }
  }

  vtkIdType polyCellIdx = 0;
  if (!gpuTessUsed)
  {
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
            // P5-5A: UV for wireframe vertex
            if (tcoordArray && tcoordArray->GetNumberOfTuples() > v0)
            {
              double uv[3];
              tcoordArray->GetTuple(v0, uv);
              triangleUVs.push_back(static_cast<float>(uv[0]));
              triangleUVs.push_back(static_cast<float>(uv[1]));
            }
            else
            {
              triangleUVs.push_back(0.0f);
              triangleUVs.push_back(0.0f);
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
            // P5-5A: UV for wireframe vertex
            if (tcoordArray && tcoordArray->GetNumberOfTuples() > v1)
            {
              double uv[3];
              tcoordArray->GetTuple(v1, uv);
              triangleUVs.push_back(static_cast<float>(uv[0]));
              triangleUVs.push_back(static_cast<float>(uv[1]));
            }
            else
            {
              triangleUVs.push_back(0.0f);
              triangleUVs.push_back(0.0f);
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
        // P2-2C: When cellFlag == 0 (per-point coloring) AND normals come from
        // the data (normalArray), deduplicate vertices by point ID and build an
        // index buffer. Each unique point has identical position, normal, and color.
        // When normals are computed per-face, each vertex gets the face normal of
        // whichever triangle first emits it, so deduplication would produce incorrect
        // normals for vertices shared between faces with different orientations.
        // When cellFlag != 0 (per-cell coloring), vertices at the same point may
        // have different colors from different cells, so no deduplication is possible.
        bool useIndexBuffer = (cellFlag == 0) && normalArray;

        for (vtkIdType i = 1; i < npts - 1; ++i)
        {
          vtkIdType tri[3] = { pts[0], pts[i], pts[i + 1] };

          if (useIndexBuffer)
          {
            // Indexed path: deduplicate vertices by point ID
            for (int j = 0; j < 3; ++j)
            {
              auto it = triVertexMap.find(tri[j]);
              if (it != triVertexMap.end())
              {
                triangleIndices.push_back(it->second);
              }
              else
              {
                uint32_t vidx = static_cast<uint32_t>(positions.size() / 3);
                triVertexMap[tri[j]] = vidx;
                triangleIndices.push_back(vidx);

                double pt[3];
                polydata->GetPoint(tri[j], pt);
                positions.push_back(static_cast<float>(pt[0]));
                positions.push_back(static_cast<float>(pt[1]));
                positions.push_back(static_cast<float>(pt[2]));

                // useIndexBuffer requires normalArray, so it's always non-null here
                double nn[3];
                normalArray->GetTuple(tri[j], nn);
                normals.push_back(static_cast<float>(nn[0]));
                normals.push_back(static_cast<float>(nn[1]));
                normals.push_back(static_cast<float>(nn[2]));

                // P1-1A: per-vertex color from point scalar mapping
                if (mappedColors)
                {
                  const unsigned char* rgba = mappedColors->GetPointer(0);
                  surfaceColors.push_back(rgba[tri[j] * 4] / 255.0f);
                  surfaceColors.push_back(rgba[tri[j] * 4 + 1] / 255.0f);
                  surfaceColors.push_back(rgba[tri[j] * 4 + 2] / 255.0f);
                  surfaceColors.push_back(rgba[tri[j] * 4 + 3] / 255.0f);
                }
                else
                {
                  surfaceColors.push_back(1.0f);
                  surfaceColors.push_back(1.0f);
                  surfaceColors.push_back(1.0f);
                  surfaceColors.push_back(1.0f);
                }

                // P5-5A: texture coordinates for indexed triangle vertex
                if (tcoordArray && tcoordArray->GetNumberOfTuples() > tri[j])
                {
                  double uv[3];
                  tcoordArray->GetTuple(tri[j], uv);
                  triangleUVs.push_back(static_cast<float>(uv[0]));
                  triangleUVs.push_back(static_cast<float>(uv[1]));
                }
                else
                {
                  triangleUVs.push_back(0.0f);
                  triangleUVs.push_back(0.0f);
                }
              }
            }
          }
          else
          {
            // Non-indexed path: emit 3 unique vertices per triangle (cell coloring)
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

            for (int j = 0; j < 3; ++j)
            {
              positions.push_back(static_cast<float>(p[j][0]));
              positions.push_back(static_cast<float>(p[j][1]));
              positions.push_back(static_cast<float>(p[j][2]));
              normals.push_back(fn[0]);
              normals.push_back(fn[1]);
              normals.push_back(fn[2]);

              // P1-1B: per-vertex color from cell scalar mapping
              if (mappedColors)
              {
                const unsigned char* rgba = mappedColors->GetPointer(0);
                surfaceColors.push_back(rgba[polyCellIdx * 4] / 255.0f);
                surfaceColors.push_back(rgba[polyCellIdx * 4 + 1] / 255.0f);
                surfaceColors.push_back(rgba[polyCellIdx * 4 + 2] / 255.0f);
                surfaceColors.push_back(rgba[polyCellIdx * 4 + 3] / 255.0f);
              }
              else
              {
                surfaceColors.push_back(1.0f);
                surfaceColors.push_back(1.0f);
                surfaceColors.push_back(1.0f);
                surfaceColors.push_back(1.0f);
              }

              // P5-5A: texture coordinates for non-indexed triangle vertex
              if (tcoordArray && tcoordArray->GetNumberOfTuples() > tri[j])
              {
                double uv[3];
                tcoordArray->GetTuple(tri[j], uv);
                triangleUVs.push_back(static_cast<float>(uv[0]));
                triangleUVs.push_back(static_cast<float>(uv[1]));
              }
              else
              {
                triangleUVs.push_back(0.0f);
                triangleUVs.push_back(0.0f);
              }
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
      // P2-2C: When indexed, TriangleIndexCount is the number of indices (3 per triangle).
      this->Internals->TriangleIndexCount = static_cast<uint32_t>(triangleIndices.size());
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
          // P5-5A: UV for line vertex (cell coloring path)
          if (tcoordArray && tcoordArray->GetNumberOfTuples() > pts[i])
          {
            double uv[3];
            tcoordArray->GetTuple(pts[i], uv);
            triangleUVs.push_back(static_cast<float>(uv[0]));
            triangleUVs.push_back(static_cast<float>(uv[1]));
          }
          else
          {
            triangleUVs.push_back(0.0f);
            triangleUVs.push_back(0.0f);
          }
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
            // P5-5A: UV for line vertex (point coloring path)
            if (tcoordArray && tcoordArray->GetNumberOfTuples() > pts[i])
            {
              double uv[3];
              tcoordArray->GetTuple(pts[i], uv);
              triangleUVs.push_back(static_cast<float>(uv[0]));
              triangleUVs.push_back(static_cast<float>(uv[1]));
            }
            else
            {
              triangleUVs.push_back(0.0f);
              triangleUVs.push_back(0.0f);
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
    // P3-3A: number of line segments for instanced thick line drawing
    this->Internals->ThickLineSegmentCount = lineIndices.size() / 2;
    // P3-3B/3C: same segment count for round cap and miter join pipelines
    this->Internals->RoundCapLineSegmentCount = this->Internals->ThickLineSegmentCount;
    this->Internals->MiterJoinLineSegmentCount = this->Internals->ThickLineSegmentCount;
  }
  } // end if (!gpuTessUsed)

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

  // P2-2C: Create triangle index buffer for deduplicated geometry
  if (!triangleIndices.empty())
  {
    this->Internals->IndexBuffer = [device
      newBufferWithBytes:triangleIndices.data()
                 length:triangleIndices.size() * sizeof(uint32_t)
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

  // P5-5A: Create triangle UV buffer (float2 per triangle/line vertex)
  if (!triangleUVs.empty())
  {
    this->Internals->TriangleUVBuffer = [device
      newBufferWithBytes:triangleUVs.data()
                 length:triangleUVs.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
  }
  else if (!positions.empty())
  {
    // Always provide a zero UV buffer so the vertex shader can always read from buffer(8)
    std::vector<float> zeroUVs(positions.size() / 3 * 2, 0.0f);
    this->Internals->TriangleUVBuffer = [device
      newBufferWithBytes:zeroUVs.data()
                 length:zeroUVs.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
  }

  // 8D: Create extra attribute buffers from user-mapped data arrays.
  // Each attribute gets a per-point buffer (works correctly for indexed/deduplicated
  // rendering where vertex count == point count). Buffers are bound at indices 16+
  // in the render bundle for access by custom shaders.
  for (auto& itr : this->ExtraAttributes)
  {
    vtkDataArray* da = nullptr;
    if (itr.second.FieldAssociation == vtkDataObject::FIELD_ASSOCIATION_POINTS)
    {
      da = polydata->GetPointData()->GetArray(itr.second.DataArrayName.c_str());
    }
    else if (itr.second.FieldAssociation == vtkDataObject::FIELD_ASSOCIATION_CELLS)
    {
      da = polydata->GetCellData()->GetArray(itr.second.DataArrayName.c_str());
    }
    if (!da)
    {
      continue;
    }

    int numComps = da->GetNumberOfComponents();
    int comp = itr.second.ComponentNumber;
    int effectiveComps = (comp < 0) ? numComps : 1;
    vtkIdType numTuples = da->GetNumberOfTuples();

    std::vector<float> attrData(numTuples * effectiveComps);
    for (vtkIdType i = 0; i < numTuples; ++i)
    {
      if (comp < 0)
      {
        double* tuple = da->GetTuple(i);
        for (int c = 0; c < numComps; ++c)
        {
          attrData[i * numComps + c] = static_cast<float>(tuple[c]);
        }
      }
      else
      {
        attrData[i] = static_cast<float>(da->GetComponent(i, comp));
      }
    }

    this->Internals->ExtraAttributeBuffers[itr.first] = [device
      newBufferWithBytes:attrData.data()
                 length:attrData.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
    this->Internals->ExtraAttributeComponentCounts[itr.first] = effectiveComps;
  }

  // P6-6A: When GPU tessellation produced edge overlay, assign edge vertex buffers
  // to point to the same per-point vertex buffers as triangles (shared vertex data).
  if (gpuTessUsed && this->Internals->HasEdgeOverlay)
  {
    this->Internals->EdgeVertexPositionBuffer = this->Internals->VertexPositionBuffer;
    this->Internals->EdgeVertexNormalBuffer = this->Internals->VertexNormalBuffer;
    this->Internals->EdgeSurfaceColorBuffer = this->Internals->SurfaceColorBuffer;
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
      mtlQueue commandBuffer];
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
      mtlQueue commandBuffer];
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

  // 8A: Use cached sample count (set by RenderPiece before this call)
  int sampleCount = this->Internals->CachedSampleCount > 0 ? this->Internals->CachedSampleCount : 1;

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
  // 8A: Skip IDs attachment when MSAA is active — render pass only has 1 color attachment
  if (sampleCount <= 1)
  {
    pipelineDesc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;  // P2-8: picking IDs
  }

  // Enable depth testing (matching WebGPU's depthCompare = Less, depthWriteEnabled = true)
  pipelineDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

  // Enable backface culling (matching WebGPU's default behavior)
  pipelineDesc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;

  // 8A: Set sample count for MSAA
  pipelineDesc.rasterSampleCount = sampleCount;

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

  // 8A: Use cached sample count (set by RenderPiece before this call)
  int sampleCount = this->Internals->CachedSampleCount > 0 ? this->Internals->CachedSampleCount : 1;

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
      if (sampleCount <= 1)
      {
        desc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;  // P2-8: picking IDs
      }
      desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
      desc.rasterSampleCount = sampleCount;

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
      if (sampleCount <= 1)
      {
        desc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;  // P2-8: picking IDs
      }
      desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
      // No backface culling for point quads
      desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
      desc.rasterSampleCount = sampleCount;

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

  // 8A: Use cached sample count (set by RenderPiece before this call)
  int sampleCount = this->Internals->CachedSampleCount > 0 ? this->Internals->CachedSampleCount : 1;

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
    if (sampleCount <= 1)
    {
      desc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;  // P2-8: picking IDs
    }
    desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassLine;
    desc.rasterSampleCount = sampleCount;

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
void vtkMetalPolyDataMapper::EnsureThickLinePipelineState(void* mtlDevice)
{
  if (this->Internals->ThickLinePipeline)
  {
    return;
  }

  // 8A: Use cached sample count (set by RenderPiece before this call)
  int sampleCount = this->Internals->CachedSampleCount > 0 ? this->Internals->CachedSampleCount : 1;

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;

  NSError* error = nil;
  NSString* shaderSource = [NSString stringWithUTF8String:vtkMetalShaders];
  id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
  if (!library)
  {
    vtkErrorMacro(<< "Failed to compile Metal shader for thick lines: "
                  << [[error localizedDescription] UTF8String]);
    return;
  }

  id<MTLFunction> vFunc = [library newFunctionWithName:@"vertex_thick_line_main"];
  id<MTLFunction> fFunc = [library newFunctionWithName:@"fragment_thick_line_main"];
  if (vFunc && fFunc)
  {
    MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = vFunc;
    desc.fragmentFunction = fFunc;
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    if (sampleCount <= 1)
    {
      desc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;  // P2-8: picking IDs
    }
    desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    // Thick lines are rendered as triangle strips (quads)
    desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    desc.rasterSampleCount = sampleCount;

    error = nil;
    this->Internals->ThickLinePipeline =
      [device newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!this->Internals->ThickLinePipeline)
    {
      vtkErrorMacro(<< "Thick line pipeline: " << [[error localizedDescription] UTF8String]);
    }
  }
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::EnsureRoundCapLinePipelineState(void* mtlDevice)
{
  if (this->Internals->RoundCapLinePipeline)
  {
    return;
  }

  // 8A: Use cached sample count (set by RenderPiece before this call)
  int sampleCount = this->Internals->CachedSampleCount > 0 ? this->Internals->CachedSampleCount : 1;

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;

  NSError* error = nil;
  NSString* shaderSource = [NSString stringWithUTF8String:vtkMetalShaders];
  id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
  if (!library)
  {
    vtkErrorMacro(<< "Failed to compile Metal shader for round cap lines: "
                  << [[error localizedDescription] UTF8String]);
    return;
  }

  id<MTLFunction> vFunc = [library newFunctionWithName:@"vertex_round_cap_line_main"];
  id<MTLFunction> fFunc = [library newFunctionWithName:@"fragment_round_cap_line_main"];
  if (vFunc && fFunc)
  {
    MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = vFunc;
    desc.fragmentFunction = fFunc;
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    if (sampleCount <= 1)
    {
      desc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;  // P2-8: picking IDs
    }
    desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    // Round cap lines are rendered as triangle lists
    desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    desc.rasterSampleCount = sampleCount;

    error = nil;
    this->Internals->RoundCapLinePipeline =
      [device newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!this->Internals->RoundCapLinePipeline)
    {
      vtkErrorMacro(<< "Round cap line pipeline: " << [[error localizedDescription] UTF8String]);
    }
  }
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::EnsureMiterJoinLinePipelineState(void* mtlDevice)
{
  if (this->Internals->MiterJoinLinePipeline)
  {
    return;
  }

  // 8A: Use cached sample count (set by RenderPiece before this call)
  int sampleCount = this->Internals->CachedSampleCount > 0 ? this->Internals->CachedSampleCount : 1;

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;

  NSError* error = nil;
  NSString* shaderSource = [NSString stringWithUTF8String:vtkMetalShaders];
  id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
  if (!library)
  {
    vtkErrorMacro(<< "Failed to compile Metal shader for miter join lines: "
                  << [[error localizedDescription] UTF8String]);
    return;
  }

  id<MTLFunction> vFunc = [library newFunctionWithName:@"vertex_miter_join_line_main"];
  id<MTLFunction> fFunc = [library newFunctionWithName:@"fragment_miter_join_line_main"];
  if (vFunc && fFunc)
  {
    MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = vFunc;
    desc.fragmentFunction = fFunc;
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    if (sampleCount <= 1)
    {
      desc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;  // P2-8: picking IDs
    }
    desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    // Miter join lines are rendered as triangle strips (quads)
    desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    desc.rasterSampleCount = sampleCount;

    error = nil;
    this->Internals->MiterJoinLinePipeline =
      [device newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!this->Internals->MiterJoinLinePipeline)
    {
      vtkErrorMacro(<< "Miter join line pipeline: " << [[error localizedDescription] UTF8String]);
    }
  }
}

//------------------------------------------------------------------------------
// 8B: Create depth peeling pipeline states for triangle rendering.
// These use the same vertex shader (vertex_main) but peeling fragment shaders
// that output to 3 color attachments (backTemp, frontDest, depthDest).
//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::EnsurePeelPipelineStates(void* mtlDevice)
{
  if (this->Internals->TriangleInitPeelPipeline && this->Internals->TrianglePeelPipeline)
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;

  NSError* error = nil;
  NSString* shaderSource = [NSString stringWithUTF8String:vtkMetalShaders];
  id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
  if (!library)
  {
    vtkErrorMacro(<< "Failed to compile Metal shader for depth peeling: "
                  << [[error localizedDescription] UTF8String]);
    return;
  }

  id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_main"];
  if (!vertexFunc)
  {
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

  // --- Init peel pipeline ---
  // Outputs RG32Float (depth range) with MAX blend, depth test=Less
  if (!this->Internals->TriangleInitPeelPipeline)
  {
    id<MTLFunction> fragFunc = [library newFunctionWithName:@"fragment_peel_init"];
    if (fragFunc)
    {
      MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
      desc.vertexFunction = vertexFunc;
      desc.fragmentFunction = fragFunc;
      desc.vertexDescriptor = vertexDesc;
      // color(0): RG32Float (depth range) with MAX blend
      desc.colorAttachments[0].pixelFormat = MTLPixelFormatRG32Float;
      desc.colorAttachments[0].blendingEnabled = YES;
      desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationMax;
      desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationMax;
      desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
      desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
      desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
      desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
      // No IDs attachment during peeling
      desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
      desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;

      error = nil;
      this->Internals->TriangleInitPeelPipeline =
        [device newRenderPipelineStateWithDescriptor:desc error:&error];
      if (!this->Internals->TriangleInitPeelPipeline)
      {
        vtkErrorMacro(<< "Init peel pipeline: " << [[error localizedDescription] UTF8String]);
      }
    }
  }

  // --- Main peel pipeline ---
  // Outputs to 3 color attachments: RGBA8 (backTemp), RGBA8 (frontDest), RG32Float (depthDest)
  // frontDest and depthDest use MAX blend
  if (!this->Internals->TrianglePeelPipeline)
  {
    id<MTLFunction> fragFunc = [library newFunctionWithName:@"fragment_peel"];
    if (fragFunc)
    {
      MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
      desc.vertexFunction = vertexFunc;
      desc.fragmentFunction = fragFunc;
      desc.vertexDescriptor = vertexDesc;
      // color(0): RGBA8 (backTemp) — no blend, direct write
      desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
      desc.colorAttachments[0].blendingEnabled = NO;
      // color(1): RGBA8 (frontDest) — MAX blend
      desc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA8Unorm;
      desc.colorAttachments[1].blendingEnabled = YES;
      desc.colorAttachments[1].rgbBlendOperation = MTLBlendOperationMax;
      desc.colorAttachments[1].alphaBlendOperation = MTLBlendOperationMax;
      desc.colorAttachments[1].sourceRGBBlendFactor = MTLBlendFactorOne;
      desc.colorAttachments[1].destinationRGBBlendFactor = MTLBlendFactorOne;
      desc.colorAttachments[1].sourceAlphaBlendFactor = MTLBlendFactorOne;
      desc.colorAttachments[1].destinationAlphaBlendFactor = MTLBlendFactorOne;
      // color(2): RG32Float (depthDest) — MAX blend
      desc.colorAttachments[2].pixelFormat = MTLPixelFormatRG32Float;
      desc.colorAttachments[2].blendingEnabled = YES;
      desc.colorAttachments[2].rgbBlendOperation = MTLBlendOperationMax;
      desc.colorAttachments[2].alphaBlendOperation = MTLBlendOperationMax;
      desc.colorAttachments[2].sourceRGBBlendFactor = MTLBlendFactorOne;
      desc.colorAttachments[2].destinationRGBBlendFactor = MTLBlendFactorOne;
      desc.colorAttachments[2].sourceAlphaBlendFactor = MTLBlendFactorOne;
      desc.colorAttachments[2].destinationAlphaBlendFactor = MTLBlendFactorOne;
      // No depth attachment during peel passes
      desc.depthAttachmentPixelFormat = MTLPixelFormatInvalid;
      desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;

      error = nil;
      this->Internals->TrianglePeelPipeline =
        [device newRenderPipelineStateWithDescriptor:desc error:&error];
      if (!this->Internals->TrianglePeelPipeline)
      {
        vtkErrorMacro(<< "Peel pipeline: " << [[error localizedDescription] UTF8String]);
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
void vtkMetalPolyDataMapper::UpdateClipPlaneUniforms(void* mtlDevice, vtkActor* actor)
{
  if (!mtlDevice || !actor)
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;

  // ClipPlaneUniforms layout: float4 planes[6] (96 bytes) + int numClipPlanes (4 bytes)
  // = 100 bytes total, padded to 16-byte alignment = 112 bytes
  // Using flat float array to avoid alignment issues.
  float cp[28]; // 7 floats × 4 components = 28 floats = 112 bytes
  memset(cp, 0, sizeof(cp));

  int numPlanes = this->GetNumberOfClippingPlanes();

  if (numPlanes > 6)
  {
    vtkWarningMacro(<< "Too many clipping planes: " << numPlanes
                    << ", maximum supported is 6.");
    numPlanes = 6;
  }

  if (numPlanes > 0)
  {
    // Get the model-to-world matrix from the actor
    vtkNew<vtkMatrix4x4> modelToWorldMatrix;
    actor->GetModelToWorldMatrix(modelToWorldMatrix);

    for (int i = 0; i < numPlanes; ++i)
    {
      double planeEquation[4];
      this->GetClippingPlaneInDataCoords(modelToWorldMatrix, i, planeEquation);

      // Pack plane equation (ax+by+cz+d) into the flat float array.
      // Each plane occupies 4 floats; planes are stored at cp[0..23].
      int base = i * 4;
      cp[base + 0] = static_cast<float>(planeEquation[0]);
      cp[base + 1] = static_cast<float>(planeEquation[1]);
      cp[base + 2] = static_cast<float>(planeEquation[2]);
      cp[base + 3] = static_cast<float>(planeEquation[3]);
    }
  }

  // Set numClipPlanes at offset 24 (index 24 in float array)
  reinterpret_cast<int*>(&cp[24])[0] = numPlanes;

  if (!this->Internals->ClipPlaneBuffer)
  {
    this->Internals->ClipPlaneBuffer = [device
      newBufferWithLength:sizeof(cp)
                 options:MTLResourceStorageModeShared];
  }
  memcpy([this->Internals->ClipPlaneBuffer contents], cp, sizeof(cp));
}

//------------------------------------------------------------------------------
// P5-5A: Create Metal texture from actor's vtkTexture
void vtkMetalPolyDataMapper::UpdateActorTexture(void* mtlDevice, vtkActor* actor)
{
  if (!mtlDevice || !actor)
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;

  vtkTexture* texture = actor->GetTexture();
  if (!texture || !texture->GetInput())
  {
    this->Internals->ActorTexture = nil;
    this->Internals->ActorSampler = nil;
    return;
  }

  // Check if texture has changed
  vtkIdType texMTime = texture->GetMTime();
  if (this->Internals->ActorTexture && texMTime == this->Internals->CachedTextureMTime)
  {
    return; // texture hasn't changed
  }
  this->Internals->CachedTextureMTime = texMTime;

  vtkImageData* image = texture->GetInput();
  int extent[6];
  image->GetExtent(extent);
  int width = extent[1] - extent[0] + 1;
  int height = extent[3] - extent[2] + 1;
  int numComponents = image->GetNumberOfScalarComponents();

  if (width <= 0 || height <= 0)
  {
    this->Internals->ActorTexture = nil;
    this->Internals->ActorSampler = nil;
    return;
  }

  // Determine pixel format — convert to RGBA8Unorm for simplicity
  MTLPixelFormat pixelFormat = MTLPixelFormatRGBA8Unorm;

  // Create texture descriptor
  MTLTextureDescriptor* texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixelFormat
                                                                                     width:width
                                                                                    height:height
                                                                                 mipmapped:NO];
  texDesc.usage = MTLTextureUsageShaderRead;
  texDesc.storageMode = MTLStorageModeShared;

  this->Internals->ActorTexture = [device newTextureWithDescriptor:texDesc];
  if (!this->Internals->ActorTexture)
  {
    vtkErrorMacro(<< "Failed to create Metal texture");
    return;
  }

  // Convert image data to RGBA8 and upload
  unsigned char* rgbaData = new unsigned char[width * height * 4];
  vtkDataArray* scalars = image->GetPointData()->GetScalars();
  if (!scalars)
  {
    delete[] rgbaData;
    this->Internals->ActorTexture = nil;
    return;
  }

  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      int srcIdx = (y * width + x) * numComponents;
      int dstIdx = (y * width + x) * 4;
      unsigned char* dst = rgbaData + dstIdx;

      switch (numComponents)
      {
        case 1:
          dst[0] = dst[1] = dst[2] = static_cast<unsigned char*>(scalars->GetVoidPointer(0))[srcIdx];
          dst[3] = 255;
          break;
        case 2:
        {
          unsigned char* src = static_cast<unsigned char*>(scalars->GetVoidPointer(0)) + srcIdx;
          dst[0] = dst[1] = dst[2] = src[0];
          dst[3] = src[1];
          break;
        }
        case 3:
        {
          unsigned char* src = static_cast<unsigned char*>(scalars->GetVoidPointer(0)) + srcIdx;
          dst[0] = src[0];
          dst[1] = src[1];
          dst[2] = src[2];
          dst[3] = 255;
          break;
        }
        case 4:
        {
          unsigned char* src = static_cast<unsigned char*>(scalars->GetVoidPointer(0)) + srcIdx;
          dst[0] = src[0];
          dst[1] = src[1];
          dst[2] = src[2];
          dst[3] = src[3];
          break;
        }
        default:
          dst[0] = dst[1] = dst[2] = dst[3] = 255;
          break;
      }
    }
  }

  // Upload texture data
  MTLRegion region = MTLRegionMake2D(0, 0, width, height);
  [this->Internals->ActorTexture replaceRegion:region
                                mipmapLevel:0
                                  withBytes:rgbaData
                                bytesPerRow:width * 4];

  delete[] rgbaData;

  // Create sampler state
  MTLSamplerDescriptor* samplerDesc = [[MTLSamplerDescriptor alloc] init];
  samplerDesc.sAddressMode = MTLSamplerAddressModeRepeat;
  samplerDesc.tAddressMode = MTLSamplerAddressModeRepeat;
  samplerDesc.minFilter = MTLSamplerMinMagFilterLinear;
  samplerDesc.magFilter = MTLSamplerMinMagFilterLinear;
  this->Internals->ActorSampler = [device newSamplerStateWithDescriptor:samplerDesc];
}

VTK_ABI_NAMESPACE_END
