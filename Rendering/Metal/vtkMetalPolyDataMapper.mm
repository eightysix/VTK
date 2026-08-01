// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalPolyDataMapper.h"

#include "vtkMetalRenderWindow.h"
#include "vtkMetalRenderer.h"
#include "vtkMetalHardwareSelector.h"
#include "vtkMetalPickTypes.h"
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
#include "vtkScalarsToColors.h"
#include "vtkNew.h"
#include "vtkPlaneCollection.h"
#include "vtkPlane.h"
#include "vtkTexture.h"
#include "vtkDataObject.h"

#import <Metal/Metal.h>

#include <unordered_map>
#import <QuartzCore/CAMetalLayer.h>

#include <vector>
#include <unordered_map>
#include <cmath>
#include <variant>
#include <utility>

// MRC ownership helpers for Objective-C manual retain/release
#include "vtkMetalMRC.h"

namespace
{

struct EdgeKey
{
    vtkIdType A;
    vtkIdType B;

    bool operator==(const EdgeKey& other) const
    {
        return A == other.A && B == other.B;
    }
};

struct EdgeKeyHash
{
    size_t operator()(const EdgeKey& k) const
    {
        auto h1 = std::hash<vtkIdType>{}(k.A);
        auto h2 = std::hash<vtkIdType>{}(k.B);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

inline EdgeKey MakeEdgeKey(vtkIdType a, vtkIdType b)
{
    if (a > b)
    {
        std::swap(a, b);
    }
    return EdgeKey{ a, b };
}

bool CommitAndWaitForCompletion(id<MTLCommandBuffer> cmdBuf)
{
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    return cmdBuf.status == MTLCommandBufferStatusCompleted;
}

// Scene-uniform flag constants (offset 256 in SceneUniformBuffer)
[[maybe_unused]] constexpr uint32_t VTK_METAL_SCENE_FLAG_PARALLEL_PROJECTION = 1u << 0;
constexpr uint32_t VTK_METAL_SCENE_FLAG_VERTEX_VISIBILITY   = 1u << 3;
constexpr uint32_t VTK_METAL_SCENE_FLAG_SPHERE_POINTS       = 1u << 5;
constexpr uint32_t VTK_METAL_SCENE_FLAG_POINT_SHAPE         = 1u << 7;
constexpr uint32_t VTK_METAL_SCENE_FLAG_HAS_SURFACE_COLORS  = 1u << 8;
constexpr uint32_t VTK_METAL_SCENE_FLAG_HAS_ACTOR_TEXTURE   = 1u << 9;
constexpr uint32_t VTK_METAL_SCENE_FLAG_HAS_SURFACE_ALPHA   = 1u << 10;

constexpr uint32_t VTK_METAL_DYNAMIC_ACTOR_FLAG_MASK =
    VTK_METAL_SCENE_FLAG_VERTEX_VISIBILITY |
    VTK_METAL_SCENE_FLAG_SPHERE_POINTS |
    VTK_METAL_SCENE_FLAG_POINT_SHAPE |
    VTK_METAL_SCENE_FLAG_HAS_SURFACE_COLORS |
    VTK_METAL_SCENE_FLAG_HAS_ACTOR_TEXTURE |
    VTK_METAL_SCENE_FLAG_HAS_SURFACE_ALPHA;

id<MTLBuffer> CreateZeroBuffer(id<MTLDevice> device, size_t bytes)
{
  if (!device || bytes == 0)
  {
    return nil;
  }
  id<MTLBuffer> buffer =
      [device newBufferWithLength:bytes
                          options:MTLResourceStorageModeShared];
  if (buffer)
  {
    memset([buffer contents], 0, bytes);
  }
  return buffer;
}

// Shared GPU/CPU picking ID layout (see vtkMetalPickTypes.h). Written into
// PropIdBuffer before each draw; consumed by vertex shaders as buffer(7)/
// buffer(6)/buffer(12)/buffer(10).
using PickIds = vtkMetalPickIds;

} // namespace

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
  id<MTLBuffer> EdgeUniformBuffer = nil;        // per-frame single-pass edge uniforms (fragment buffer(4))
  id<MTLBuffer> VertexColorBuffer = nil;        // P1-4: vertex visibility color
  id<MTLBuffer> ClipPlaneBuffer = nil;          // P1-6: clipping planes
  // P2-8: Picking IDs
  id<MTLBuffer> TriangleCellIdBuffer = nil;    // GPU: per-triangle-vertex cell IDs
  id<MTLBuffer> LineCellIdBuffer = nil;        // GPU: per-line-index cell IDs (standard 1px lines, by vertex_id)
  id<MTLBuffer> LineSegmentCellIdBuffer = nil; // GPU: per-line-segment cell IDs (thick/round/miter lines, by instance_id)
  id<MTLBuffer> PointCellIdBuffer = nil;       // GPU: per-point cell IDs
  id<MTLBuffer> EdgeCellIdBuffer = nil;
  id<MTLBuffer> EdgeUVBuffer = nil;            // P1-2: edge overlay UV coordinates
  id<MTLBuffer> PropIdBuffer = nil;            // PickIds {propId, compositeIndex}: picking IDs

  // P2-4: separate alpha flag for opacity-only overrides
  bool HasSurfaceAlpha = false;

  // P1-3: zero fallback buffers — always provide valid bindings for shader-required slots
  id<MTLBuffer> ZeroTriangleCellIdBuffer = nil;
  id<MTLBuffer> ZeroLineCellIdBuffer = nil;
  id<MTLBuffer> ZeroEdgeCellIdBuffer = nil;
  id<MTLBuffer> ZeroTriangleUVBuffer = nil;
  id<MTLBuffer> ZeroEdgeUVBuffer = nil;

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

  // Single-pass surface edges: per-triangle-corner barycentrics + boundary flags
  id<MTLBuffer> TriangleBaryBuffer = nil;      // float3 per triangle corner (3*numTris)
  id<MTLBuffer> TriangleEdgeFlagBuffer = nil;  // uint   per triangle corner (3*numTris)
  id<MTLBuffer> TrianglePosBuffer = nil;       // float3[3] corner object positions per corner record (3*numTris)
  bool SurfaceUsesIndexedEntry = false;        // pipeline-selection key: edges folded into fragment
  bool CachedSurfaceUsesIndexedEntry = false;  // indexedEntry baked into cached pipelines; rebuild on flip

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
  id<MTLBuffer> EdgeTubeIndexBuffer = nil;      // per-polygon closed-loop segments for mitered edge tubes
  id<MTLBuffer> EdgeTubeCellIdBuffer = nil;     // per-segment cell id for edge tube loops
  id<MTLBuffer> EdgeTubeSegmentCountBuffer = nil; // uint32: total edge tube segment count
  vtkIdType EdgeTubeSegmentCount = 0;

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

  // 8C: Order-independent transparency accumulate pipeline. Same vertex shader,
  // fragment_main_oit outputs premultiplied color (RGBA16F) + revealage (R16F).
  id<MTLRenderPipelineState> TriangleOITPipeline = nil;

  // 8D: Vertex attribute mapping — custom per-vertex buffers from user-mapped data arrays
  std::unordered_map<std::string, id<MTLBuffer>> ExtraAttributeBuffers;
  std::unordered_map<std::string, int> ExtraAttributeComponentCounts;

  vtkIdType CachedInputMTime = 0;
  int CachedRepresentation = -1;
  bool CachedEdgeVisibility = false;  // P2-2B: track edge visibility changes
  float CachedLineWidth = -1.0f;     // P3-3A: track line width changes
  int CachedSampleCount = 0;        // 8A: track MSAA sample count changes
  vtkMTimeType CachedScalarMTime = 0;

  // Weak reference to the owning render window (set by RenderPiece).
  // Used by Ensure* methods to access the cached shader library.
  vtkMetalRenderWindow* CachedRenderWindow = nullptr;

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
  bool BundleOITActive = false;
  vtkMTimeType BundleTextureMTime = 0;
  bool BundleHasActorTexture = false;

  bool BundleVertexVisibility = false;
  float BundlePointSize = -1.0f;
  bool BundleRenderPointsAsSpheres = false;
  int BundlePoint2DShape = -1;

  int BundleLineJoin = -1;

  bool BundleBackfaceCulling = false;
  bool BundleFrontfaceCulling = false;

  vtkMTimeType BundleExtraAttributesMTime = 0;

  vtkMTimeType CachedExtraAttributesMTime = 0;

  // Override prop ID state (set by batched mapper for per-block picking)
  bool HasOverridePropId = false;
  uint32_t OverridePropId = 0;

  // Override composite index state (set by batched mapper for per-block picking)
  bool HasOverrideCompositeIndex = false;
  uint32_t OverrideCompositeIndex = 0;

  // Batch visual override state (set by batched mapper)
  bool UseBatchColor = false;
  double BatchColor[3] = { 1.0, 1.0, 1.0 };
  bool UseBatchOpacity = false;
  double BatchOpacity = 1.0;
  vtkMTimeType BatchOverrideMTime = 0;
  vtkMTimeType CachedBatchOverrideMTime = 0;
  vtkMTimeType BundleBatchOverrideMTime = 0;

  id<MTLCommandQueue> ComputeQueue = nil;

  id<MTLCommandQueue> EnsureComputeQueue(id<MTLDevice> device)
  {
    if (!this->ComputeQueue)
    {
      id<MTLCommandQueue> queue = [device newCommandQueue];
      vtkMetalMRC::AssignConsumed(this->ComputeQueue, queue);
    }
    return this->ComputeQueue;
  }

  void ReleasePipelines()
  {
    InvalidateRenderBundle();
    vtkMetalMRC::ReleaseAndNil(TrianglePipeline);
    vtkMetalMRC::ReleaseAndNil(LinePipeline);
    vtkMetalMRC::ReleaseAndNil(PointPipeline);
    vtkMetalMRC::ReleaseAndNil(PointShapedPipeline);
    vtkMetalMRC::ReleaseAndNil(EdgePipeline);
    vtkMetalMRC::ReleaseAndNil(ThickLinePipeline);
    vtkMetalMRC::ReleaseAndNil(RoundCapLinePipeline);
    vtkMetalMRC::ReleaseAndNil(MiterJoinLinePipeline);
    vtkMetalMRC::ReleaseAndNil(TriangleInitPeelPipeline);
    vtkMetalMRC::ReleaseAndNil(TrianglePeelPipeline);
    vtkMetalMRC::ReleaseAndNil(TriangleOITPipeline);

    vtkMetalMRC::ReleaseAndNil(PolygonToTrianglePipeline);
    vtkMetalMRC::ReleaseAndNil(PolyLineToLinePipeline);
    vtkMetalMRC::ReleaseAndNil(PolygonEdgesToLinesPipeline);
  }

  void InvalidateRenderBundle()
  {
    Bundle.Invalidate();
  }

  void ReleaseBuffers()
  {
    InvalidateRenderBundle();

    vtkMetalMRC::ReleaseAndNil(VertexPositionBuffer);
    vtkMetalMRC::ReleaseAndNil(VertexNormalBuffer);
    vtkMetalMRC::ReleaseAndNil(IndexBuffer);
    vtkMetalMRC::ReleaseAndNil(LineIndexBuffer);
    vtkMetalMRC::ReleaseAndNil(SurfaceColorBuffer);
    HasSurfaceColors = false;
    vtkMetalMRC::ReleaseAndNil(TriangleUVBuffer);

    vtkMetalMRC::ReleaseAndNil(ActorTexture);
    vtkMetalMRC::ReleaseAndNil(ActorSampler);
    vtkMetalMRC::ReleaseAndNil(DefaultTexture);
    vtkMetalMRC::ReleaseAndNil(DefaultSampler);
    CachedTextureMTime = 0;

    vtkMetalMRC::ReleaseAndNil(EdgeVertexPositionBuffer);
    vtkMetalMRC::ReleaseAndNil(EdgeVertexNormalBuffer);
    vtkMetalMRC::ReleaseAndNil(EdgeSurfaceColorBuffer);
    vtkMetalMRC::ReleaseAndNil(EdgeIndexBuffer);
    vtkMetalMRC::ReleaseAndNil(EdgeColorUniformBuffer);
    vtkMetalMRC::ReleaseAndNil(EdgeTubeIndexBuffer);
    vtkMetalMRC::ReleaseAndNil(EdgeTubeCellIdBuffer);
    vtkMetalMRC::ReleaseAndNil(EdgeTubeSegmentCountBuffer);
    EdgeTubeSegmentCount = 0;
    EdgeIndexCount = 0;
    EdgeVertexCount = 0;
    HasEdgeOverlay = false;

    vtkMetalMRC::ReleaseAndNil(ThickLineLineWidthBuffer);
    ThickLineSegmentCount = 0;
    vtkMetalMRC::ReleaseAndNil(MiterJoinSegmentCountBuffer);
    RoundCapLineSegmentCount = 0;
    MiterJoinLineSegmentCount = 0;

    vtkMetalMRC::ReleaseAndNil(PointPositionBuffer);
    vtkMetalMRC::ReleaseAndNil(PointNormalBuffer);
    vtkMetalMRC::ReleaseAndNil(PointColorBuffer);
    vtkMetalMRC::ReleaseAndNil(PointTangentBuffer);
    vtkMetalMRC::ReleaseAndNil(PointUVBuffer);
    vtkMetalMRC::ReleaseAndNil(PointColorUVBuffer);
    vtkMetalMRC::ReleaseAndNil(PointConnectivityBuffer);
    PointVertexCount = 0;

    vtkMetalMRC::ReleaseAndNil(SceneUniformBuffer);
    vtkMetalMRC::ReleaseAndNil(MaterialUniformBuffer);
    vtkMetalMRC::ReleaseAndNil(LightUniformBuffer);
    vtkMetalMRC::ReleaseAndNil(CoincidentOffsetBuffer);
    vtkMetalMRC::ReleaseAndNil(EdgeUniformBuffer);
    vtkMetalMRC::ReleaseAndNil(VertexColorBuffer);
    vtkMetalMRC::ReleaseAndNil(ClipPlaneBuffer);

    vtkMetalMRC::ReleaseAndNil(TriangleCellIdBuffer);
    vtkMetalMRC::ReleaseAndNil(LineCellIdBuffer);
    vtkMetalMRC::ReleaseAndNil(LineSegmentCellIdBuffer);
    vtkMetalMRC::ReleaseAndNil(PointCellIdBuffer);
    vtkMetalMRC::ReleaseAndNil(EdgeCellIdBuffer);
    vtkMetalMRC::ReleaseAndNil(EdgeUVBuffer);
    vtkMetalMRC::ReleaseAndNil(PropIdBuffer);

    vtkMetalMRC::ReleaseAndNil(ZeroTriangleCellIdBuffer);
    vtkMetalMRC::ReleaseAndNil(ZeroLineCellIdBuffer);
    vtkMetalMRC::ReleaseAndNil(ZeroEdgeCellIdBuffer);
    vtkMetalMRC::ReleaseAndNil(ZeroTriangleUVBuffer);
    vtkMetalMRC::ReleaseAndNil(ZeroEdgeUVBuffer);

    HasSurfaceAlpha = false;

    TrianglePrimitiveCount = 0;
    LinePrimitiveCount = 0;

    vtkMetalMRC::ReleaseAndNil(TessOutputConnectivityBuffer);
    vtkMetalMRC::ReleaseAndNil(TessEdgeArrayBuffer);
    vtkMetalMRC::ReleaseAndNil(TessParamsBuffer);
    vtkMetalMRC::ReleaseAndNil(TriangleBaryBuffer);
    vtkMetalMRC::ReleaseAndNil(TriangleEdgeFlagBuffer);
    vtkMetalMRC::ReleaseAndNil(TrianglePosBuffer);
    SurfaceUsesIndexedEntry = false;
    CachedSurfaceUsesIndexedEntry = false;
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

    for (auto& kv : ExtraAttributeBuffers)
    {
      [kv.second release];
    }
    ExtraAttributeBuffers.clear();
    ExtraAttributeComponentCounts.clear();

    ReleasePipelines();

    vtkMetalMRC::ReleaseAndNil(ComputeQueue);

    BundleGeometryMTime = 0;
    BundleRepresentation = -1;
    BundleEdgeVisibility = false;
    BundleLineWidth = -1.0f;
    BundleSampleCount = 0;
    BundlePeelMode = 0;
    BundleTextureMTime = 0;
    BundleHasActorTexture = false;
    BundleVertexVisibility = false;
    BundlePointSize = -1.0f;
    BundleRenderPointsAsSpheres = false;
    BundlePoint2DShape = -1;
    BundleLineJoin = -1;
    BundleBackfaceCulling = false;
    BundleFrontfaceCulling = false;
    BundleExtraAttributesMTime = 0;

    CachedExtraAttributesMTime = 0;
    CachedBatchOverrideMTime = 0;
    BundleBatchOverrideMTime = 0;
    CachedScalarMTime = 0;
  }

  ~vtkMetalPolyDataMapperInternals()
  {
    ReleaseBuffers();
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

void vtkMetalPolyDataMapper::ReleaseGraphicsResources(vtkWindow* w)
{
  this->Internals->ReleaseBuffers();
  this->Superclass::ReleaseGraphicsResources(w);
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
  this->ExtraAttributesMTime.Modified();
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
    this->ExtraAttributesMTime.Modified();
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
  this->ExtraAttributesMTime.Modified();
}

//------------------------------------------------------------------------------
// 8C: Render bundle — replay cached encoder commands on the current render encoder.
// Uniform buffers (scene, material, light, etc.) are updated in-place each frame,
// so replaying the same buffer bindings reads the latest content automatically.
void vtkMetalPolyDataMapper::ReplayRenderBundle(void* mtlEncoder)
{
  id<MTLRenderCommandEncoder> encoder = (id<MTLRenderCommandEncoder>)mtlEncoder;
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
  id<MTLRenderCommandEncoder> encoder = (id<MTLRenderCommandEncoder>)mtlEncoder;
  using Cmd = vtkMetalPolyDataMapperInternals::RenderBundleDrawCommand;
  using PSParams = Cmd::SetPipelineStateParams;
  using BufParams = Cmd::SetBufferParams;
  using TexParams = Cmd::SetTextureParams;
  using SampParams = Cmd::SetSamplerParams;
  using CullParams = Cmd::SetCullModeParams;
  using DrawParams = Cmd::DrawPrimitivesParams;
  using IdxParams = Cmd::DrawIndexedPrimitivesParams;

  bool recordedAnyDraw = false;
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
  auto recordDraw = [&commands, &recordedAnyDraw](MTLPrimitiveType ptype, NSUInteger vstart, NSUInteger vcount,
                        NSUInteger icount = 0) {
    Cmd cmd;
    cmd.type = Cmd::DrawPrimitives;
    cmd.params = DrawParams{ ptype, vstart, vcount, icount };
    commands.push_back(cmd);
    recordedAnyDraw = true;
  };
  auto recordIdxDraw = [&commands, &recordedAnyDraw](MTLPrimitiveType ptype, NSUInteger indexCount, MTLIndexType itype,
                           id<MTLBuffer> ibuf, NSUInteger offset) {
    Cmd cmd;
    cmd.type = Cmd::DrawIndexedPrimitives;
    cmd.params = IdxParams{ ptype, indexCount, itype, ibuf, offset };
    commands.push_back(cmd);
    recordedAnyDraw = true;
  };

  int representation = act->GetProperty()->GetRepresentation();
  float lineWidth = static_cast<float>(act->GetProperty()->GetLineWidth());

  const bool isSurface = (representation == VTK_SURFACE);
  const bool isWireframe = (representation == VTK_WIREFRAME);
  const bool isPoints = (representation == VTK_POINTS);

  int peelMode = vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow())->DepthPeelingMode;
  const bool peelPassActive = (peelMode != 0);
  vtkMetalRenderWindow* renWin =
      vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());
  const bool oitActive = renWin->OITActive;

  // During peel passes the relevant pipelines are TriangleInitPeelPipeline
  // (peelMode==1) and TrianglePeelPipeline (peelMode==2); fallback to
  // TrianglePipeline is invalid inside a peel pass.
  const bool standardLinePipelineAvailable = this->Internals->LinePipeline != nil;
  const bool thickLinesAvailable =
      lineWidth > 1.0f &&
      this->Internals->ThickLineSegmentCount > 0 &&
      this->Internals->ThickLinePipeline != nil;
  const bool miterLinesAvailable =
      lineWidth > 1.0f &&
      act->GetProperty()->GetLineJoin() == vtkProperty::LineJoinType::MiterJoin &&
      this->Internals->MiterJoinLineSegmentCount > 0 &&
      this->Internals->MiterJoinLinePipeline != nil;

  const bool trianglePipelineAvailable =
      this->Internals->TrianglePipeline ||
      (peelMode == 1 && this->Internals->TriangleInitPeelPipeline) ||
      (peelMode == 2 && this->Internals->TrianglePeelPipeline) ||
      (oitActive && this->Internals->TriangleOITPipeline);

  const bool drawTriangles =
      isSurface &&
      this->Internals->HasTriangles &&
      trianglePipelineAvailable;

  // Lines, edge overlays, and point dots are not drawn during depth-peel or
  // OIT accumulate passes — those pipelines write to different color
  // attachments than the pass provides.
  const bool drawLines =
      !peelPassActive && !oitActive &&
      (isSurface || isWireframe) &&
      this->Internals->HasLines &&
      this->Internals->LineIndexBuffer &&
      (standardLinePipelineAvailable || thickLinesAvailable || miterLinesAvailable);

  const bool drawEdgeOverlay =
      this->UseLegacyEdgeOverlay &&
      isSurface &&
      act->GetProperty()->GetEdgeVisibility() &&
      this->Internals->HasEdgeOverlay &&
      this->Internals->EdgeIndexBuffer &&
      (this->Internals->EdgePipeline ||
       (lineWidth > 1.0f &&
        this->Internals->EdgeTubeSegmentCount > 0 &&
        (this->Internals->MiterJoinLinePipeline || this->Internals->ThickLinePipeline)));

  const bool drawPointRepresentation =
      isPoints &&
      this->Internals->PointVertexCount > 0 &&
      this->Internals->PointPositionBuffer;

  const bool drawVertexVisibilityDots =
      !isPoints &&
      act->GetProperty()->GetVertexVisibility() &&
      this->Internals->PointVertexCount > 0 &&
      this->Internals->PointPositionBuffer;

  // --- Triangle drawing ---
  bool skipTriangleDraw = false;
  if (drawTriangles)
  {
    if (peelMode == 1)
    {
      if (this->Internals->TriangleInitPeelPipeline)
      {
        recordPipeline(this->Internals->TriangleInitPeelPipeline);
      }
      else
      {
        vtkWarningMacro(<< "Missing init peel pipeline; skipping triangle draw in peel pass.");
        skipTriangleDraw = true;
      }
    }
    else if (peelMode == 2)
    {
      vtkMetalRenderWindow* peelRenWin =
          vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());
      if (this->Internals->TrianglePeelPipeline &&
          peelRenWin && peelRenWin->PeelFrontTexture && peelRenWin->PeelDepthTexture)
      {
        recordPipeline(this->Internals->TrianglePeelPipeline);
        recordFTex((__bridge id<MTLTexture>)peelRenWin->PeelFrontTexture, 1);
        recordFTex((__bridge id<MTLTexture>)peelRenWin->PeelDepthTexture, 2);
      }
      else
      {
        vtkWarningMacro(<< "Missing peel pipeline or textures; skipping triangle draw in peel pass.");
        skipTriangleDraw = true;
      }
    }
    else if (oitActive)
    {
      if (this->Internals->TriangleOITPipeline)
      {
        recordPipeline(this->Internals->TriangleOITPipeline);
      }
      else
      {
        vtkWarningMacro(<< "Missing OIT accumulate pipeline; skipping triangle draw.");
        skipTriangleDraw = true;
      }
    }
    else
    {
      recordPipeline(this->Internals->TrianglePipeline);
    }
    if (!skipTriangleDraw)
    {
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
      // Single-pass surface edges: fragment_main (TrianglePipeline / peel variants)
      // reads EdgeUniforms at buffer(4). Bind explicitly here; a retained binding
      // from an earlier 1px-line draw in the same pass is not guaranteed.
      if (this->Internals->EdgeUniformBuffer)
      {
        recordFBuf(this->Internals->EdgeUniformBuffer, 0, 4);
      }
      if (this->Internals->ClipPlaneBuffer)
      {
        recordFBuf(this->Internals->ClipPlaneBuffer, 0, 5);
        recordVBuf(this->Internals->ClipPlaneBuffer, 0, 5);
      }
      // P1-3: Fallback-safe cell ID binding for triangles (vertex stage)
      if (this->Internals->TriangleCellIdBuffer)
      {
        recordVBuf(this->Internals->TriangleCellIdBuffer, 0, 6);
      }
      else if (this->Internals->ZeroTriangleCellIdBuffer)
      {
        recordVBuf(this->Internals->ZeroTriangleCellIdBuffer, 0, 6);
      }

      if (this->Internals->PropIdBuffer)
      {
        recordVBuf(this->Internals->PropIdBuffer, 0, 7);
      }

      // P1-3: Fallback-safe UV binding for triangles (vertex stage)
      if (this->Internals->TriangleUVBuffer)
      {
        recordVBuf(this->Internals->TriangleUVBuffer, 0, 8);
      }
      else if (this->Internals->ZeroTriangleUVBuffer)
      {
        recordVBuf(this->Internals->ZeroTriangleUVBuffer, 0, 8);
      }
      // 8D: Bind extra attribute buffers at buffer indices 16+
      {
        NSUInteger extraIdx = 16;
        for (const auto& attr : this->ExtraAttributes)
        {
          auto it = this->Internals->ExtraAttributeBuffers.find(attr.first);
          if (it != this->Internals->ExtraAttributeBuffers.end() && it->second)
          {
            recordVBuf(it->second, 0, extraIdx);
            ++extraIdx;
          }
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
      if (this->Internals->SurfaceUsesIndexedEntry)
      {
        if (this->Internals->IndexBuffer && this->Internals->TriangleBaryBuffer &&
            this->Internals->TriangleEdgeFlagBuffer && this->Internals->TrianglePosBuffer)
        {
          recordVBuf(this->Internals->IndexBuffer, 0, 9);
          recordVBuf(this->Internals->TriangleBaryBuffer, 0, 10);
          recordVBuf(this->Internals->TriangleEdgeFlagBuffer, 0, 11);
          recordVBuf(this->Internals->TrianglePosBuffer, 0, 12);
          recordDraw(MTLPrimitiveTypeTriangle, 0, this->Internals->TriangleIndexCount);
        }
        else
        {
          vtkWarningMacro(<< "Indexed surface entry buffers missing; skipping triangle draw.");
        }
      }
      else if (this->Internals->IndexBuffer)
      {
        recordIdxDraw(MTLPrimitiveTypeTriangle, this->Internals->TriangleIndexCount, MTLIndexTypeUInt32,
          this->Internals->IndexBuffer, 0);
      }
      else
      {
        recordDraw(MTLPrimitiveTypeTriangle, 0, this->Internals->TriangleVertexCount);
      }
    }
  }

  // --- Line drawing (skipped during depth-peel passes) ---
  if (!peelPassActive && drawLines)
  {
    auto lineJoinType = act->GetProperty()->GetLineJoin();
    bool useRoundCapLines = false;
    bool useMiterJoinLines = false;
    bool useThickLines = false;

    if (lineWidth > 1.0f)
    {
      // TODO: Re-enable round-cap lines once topology verification confirms
      // the CPU-side line-building pass produces correct round-cap vertices.
      if (lineJoinType == vtkProperty::LineJoinType::MiterJoin &&
               this->Internals->MiterJoinLineSegmentCount > 0 && this->Internals->MiterJoinLinePipeline)
      {
        useMiterJoinLines = true;
      }
      else if (this->Internals->ThickLineSegmentCount > 0 && this->Internals->ThickLinePipeline)
      {
        // Fallback: thick lines for round-cap, no-join, and all other non-miter joins
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
      if (this->Internals->LineSegmentCellIdBuffer)
      {
        recordVBuf(this->Internals->LineSegmentCellIdBuffer, 0, 5);
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
      if (this->Internals->LineSegmentCellIdBuffer)
      {
        recordVBuf(this->Internals->LineSegmentCellIdBuffer, 0, 5);
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
      if (this->Internals->LineSegmentCellIdBuffer)
      {
        recordVBuf(this->Internals->LineSegmentCellIdBuffer, 0, 5);
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
      if (this->Internals->EdgeUniformBuffer)
      {
        recordFBuf(this->Internals->EdgeUniformBuffer, 0, 4);
      }

      // P1-1: Bind clip planes for 1px lines (both vertex and fragment stages)
      if (this->Internals->ClipPlaneBuffer)
      {
        recordVBuf(this->Internals->ClipPlaneBuffer, 0, 5);
        recordFBuf(this->Internals->ClipPlaneBuffer, 0, 5);
      }

      // P1-3: Fallback-safe cell ID binding for lines (vertex stage)
      if (this->Internals->LineCellIdBuffer)
      {
        recordVBuf(this->Internals->LineCellIdBuffer, 0, 6);
      }
      else if (this->Internals->ZeroLineCellIdBuffer)
      {
        recordVBuf(this->Internals->ZeroLineCellIdBuffer, 0, 6);
      }

      if (this->Internals->PropIdBuffer)
      {
        recordVBuf(this->Internals->PropIdBuffer, 0, 7);
      }

      // P1-3: Fallback-safe UV binding for lines (vertex stage)
      if (this->Internals->TriangleUVBuffer)
      {
        recordVBuf(this->Internals->TriangleUVBuffer, 0, 8);
      }
      else if (this->Internals->ZeroTriangleUVBuffer)
      {
        recordVBuf(this->Internals->ZeroTriangleUVBuffer, 0, 8);
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

  // --- Edge overlay (wireframe on surface, skipped during depth-peel passes) ---
  if (!peelPassActive && !oitActive && drawEdgeOverlay)
  {
    // When lineWidth > 1, polygon edges render as thick tubes (matching GL's
    // fake-tube edge rendering). Per-polygon closed-loop segments let the miter
    // join shader connect edges into continuous tubes.
    const bool edgeTubes =
      lineWidth > 1.0f && this->Internals->EdgeTubeSegmentCount > 0 &&
      (this->Internals->MiterJoinLinePipeline || this->Internals->ThickLinePipeline);

    if (edgeTubes)
    {
      if (this->Internals->MiterJoinLinePipeline)
      {
        recordPipeline(this->Internals->MiterJoinLinePipeline);
      }
      else
      {
        recordPipeline(this->Internals->ThickLinePipeline);
      }
      recordVBuf(this->Internals->EdgeVertexPositionBuffer, 0, 0);
      recordVBuf(this->Internals->EdgeTubeIndexBuffer, 0, 1);
      if (this->Internals->SceneUniformBuffer)
      {
        recordVBuf(this->Internals->SceneUniformBuffer, 0, 2);
        recordFBuf(this->Internals->SceneUniformBuffer, 0, 2);
      }
      if (this->Internals->EdgeSurfaceColorBuffer)
      {
        recordVBuf(this->Internals->EdgeSurfaceColorBuffer, 0, 3);
      }
      if (this->Internals->ThickLineLineWidthBuffer)
      {
        recordVBuf(this->Internals->ThickLineLineWidthBuffer, 0, 4);
      }
      if (this->Internals->EdgeTubeCellIdBuffer)
      {
        recordVBuf(this->Internals->EdgeTubeCellIdBuffer, 0, 5);
      }
      if (this->Internals->PropIdBuffer)
      {
        recordVBuf(this->Internals->PropIdBuffer, 0, 6);
      }
      if (this->Internals->MiterJoinLinePipeline &&
        this->Internals->EdgeTubeSegmentCountBuffer)
      {
        recordVBuf(this->Internals->EdgeTubeSegmentCountBuffer, 0, 7);
      }
      if (this->Internals->MaterialUniformBuffer)
      {
        recordFBuf(this->Internals->MaterialUniformBuffer, 0, 0);
      }
      if (this->Internals->LightUniformBuffer)
      {
        recordFBuf(this->Internals->LightUniformBuffer, 0, 1);
      }
      if (this->Internals->CoincidentOffsetBuffer)
      {
        recordFBuf(this->Internals->CoincidentOffsetBuffer, 0, 3);
      }
      recordCull(MTLCullModeNone);
      recordDraw(MTLPrimitiveTypeTriangleStrip, 0, 4,
        this->Internals->EdgeTubeSegmentCount);
    }
    else
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
      // P1-3: Fallback-safe cell ID binding for edges (vertex stage)
      if (this->Internals->EdgeCellIdBuffer)
    {
      recordVBuf(this->Internals->EdgeCellIdBuffer, 0, 6);
    }
    else if (this->Internals->ZeroEdgeCellIdBuffer)
    {
      recordVBuf(this->Internals->ZeroEdgeCellIdBuffer, 0, 6);
    }

    if (this->Internals->PropIdBuffer)
    {
      recordVBuf(this->Internals->PropIdBuffer, 0, 7);
    }

    // P1-2: Bind edge UV buffer (vertex stage index 8)
    if (this->Internals->EdgeUVBuffer)
    {
      recordVBuf(this->Internals->EdgeUVBuffer, 0, 8);
    }
    else if (this->Internals->ZeroEdgeUVBuffer)
    {
      recordVBuf(this->Internals->ZeroEdgeUVBuffer, 0, 8);
    }

    // P1-1: Bind clip planes for edge overlay
    if (this->Internals->ClipPlaneBuffer)
    {
      recordVBuf(this->Internals->ClipPlaneBuffer, 0, 5);
      recordFBuf(this->Internals->ClipPlaneBuffer, 0, 5);
    }

      recordCull(MTLCullModeNone);
      recordIdxDraw(MTLPrimitiveTypeLine, this->Internals->EdgeIndexCount, MTLIndexTypeUInt32,
        this->Internals->EdgeIndexBuffer, 0);
    }
  }

  // --- Vertex visibility (dots on surface, skipped during depth-peel passes) ---
  if (!peelPassActive && !oitActive && drawVertexVisibilityDots)
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

  // --- Points (VTK_POINTS representation, skipped during depth-peel passes) ---
  if (!peelPassActive && !oitActive && drawPointRepresentation)
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

  // Collect bundle validation state
  bool hasActorTexture = (this->Internals->ActorTexture != nil);
  bool vertexVisibility = act->GetProperty()->GetVertexVisibility();
  float pointSize = static_cast<float>(act->GetProperty()->GetPointSize());
  bool renderPointsAsSpheres = act->GetProperty()->GetRenderPointsAsSpheres();
  int point2DShape = static_cast<int>(act->GetProperty()->GetPoint2DShape());
  int lineJoin = static_cast<int>(act->GetProperty()->GetLineJoin());
  bool backfaceCulling = act->GetProperty()->GetBackfaceCulling();
  bool frontfaceCulling = act->GetProperty()->GetFrontfaceCulling();
  vtkMTimeType extraMTime = this->ExtraAttributesMTime.GetMTime();
  vtkMTimeType batchOverrideMTime = this->Internals->BatchOverrideMTime;

  // Mark bundle as valid and record the state that was used to create it
  this->Internals->Bundle.Valid = recordedAnyDraw;
  this->Internals->BundleGeometryMTime = this->Internals->CachedInputMTime;
  this->Internals->BundleRepresentation = representation;
  this->Internals->BundleEdgeVisibility = this->Internals->CachedEdgeVisibility;
  this->Internals->BundleLineWidth = lineWidth;
  this->Internals->BundleSampleCount = this->Internals->CachedSampleCount;
  this->Internals->BundlePeelMode = peelMode;
  this->Internals->BundleOITActive = oitActive;
  this->Internals->BundleTextureMTime = this->Internals->CachedTextureMTime;
  this->Internals->BundleHasActorTexture = hasActorTexture;
  this->Internals->BundleVertexVisibility = vertexVisibility;
  this->Internals->BundlePointSize = pointSize;
  this->Internals->BundleRenderPointsAsSpheres = renderPointsAsSpheres;
  this->Internals->BundlePoint2DShape = point2DShape;
  this->Internals->BundleLineJoin = lineJoin;
  this->Internals->BundleBackfaceCulling = backfaceCulling;
  this->Internals->BundleFrontfaceCulling = frontfaceCulling;
  this->Internals->BundleExtraAttributesMTime = extraMTime;
  this->Internals->BundleBatchOverrideMTime = batchOverrideMTime;
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::EnsureRequiredBindingFallbacks(void* mtlDevice)
{
  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;

  const vtkIdType vertexCount =
      this->Internals->VertexPositionBuffer
          ? static_cast<vtkIdType>(
                [this->Internals->VertexPositionBuffer length] /
                (3 * sizeof(float)))
          : 0;

  const vtkIdType edgeVertexCount = this->Internals->EdgeVertexCount;

  // Zero cell ID buffers
  if (vertexCount > 0)
  {
    if (!this->Internals->ZeroTriangleCellIdBuffer)
    {
      id<MTLBuffer> buffer =
          CreateZeroBuffer(device, static_cast<size_t>(vertexCount) * sizeof(uint32_t));
      if (!buffer)
      {
        vtkErrorMacro(<< "Failed to allocate ZeroTriangleCellIdBuffer (" << vertexCount << " entries)");
      }
      vtkMetalMRC::AssignConsumed(this->Internals->ZeroTriangleCellIdBuffer, buffer);
    }

    if (!this->Internals->ZeroLineCellIdBuffer)
    {
      id<MTLBuffer> buffer =
          CreateZeroBuffer(device, static_cast<size_t>(vertexCount) * sizeof(uint32_t));
      if (!buffer)
      {
        vtkErrorMacro(<< "Failed to allocate ZeroLineCellIdBuffer (" << vertexCount << " entries)");
      }
      vtkMetalMRC::AssignConsumed(this->Internals->ZeroLineCellIdBuffer, buffer);
    }

    if (!this->Internals->ZeroTriangleUVBuffer)
    {
      id<MTLBuffer> buffer =
          CreateZeroBuffer(device, static_cast<size_t>(vertexCount) * 2 * sizeof(float));
      if (!buffer)
      {
        vtkErrorMacro(<< "Failed to allocate ZeroTriangleUVBuffer (" << vertexCount << " entries)");
      }
      vtkMetalMRC::AssignConsumed(this->Internals->ZeroTriangleUVBuffer, buffer);
    }
  }

  if (edgeVertexCount > 0)
  {
    if (!this->Internals->ZeroEdgeCellIdBuffer)
    {
      id<MTLBuffer> buffer =
          CreateZeroBuffer(device, static_cast<size_t>(edgeVertexCount) * sizeof(uint32_t));
      if (!buffer)
      {
        vtkErrorMacro(<< "Failed to allocate ZeroEdgeCellIdBuffer (" << edgeVertexCount << " entries)");
      }
      vtkMetalMRC::AssignConsumed(this->Internals->ZeroEdgeCellIdBuffer, buffer);
    }

    if (!this->Internals->ZeroEdgeUVBuffer)
    {
      id<MTLBuffer> buffer =
          CreateZeroBuffer(device, static_cast<size_t>(edgeVertexCount) * 2 * sizeof(float));
      if (!buffer)
      {
        vtkErrorMacro(<< "Failed to allocate ZeroEdgeUVBuffer (" << edgeVertexCount << " entries)");
      }
      vtkMetalMRC::AssignConsumed(this->Internals->ZeroEdgeUVBuffer, buffer);
    }
  }

  // Prop ID buffer should always exist if anything is drawable.
  if (!this->Internals->PropIdBuffer)
  {
    PickIds ids = { 0, 0 };
    id<MTLBuffer> buffer =
        [device newBufferWithBytes:&ids
                            length:sizeof(PickIds)
                           options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->PropIdBuffer, buffer);
  }

  // Clip plane buffer should always exist.
  if (!this->Internals->ClipPlaneBuffer)
  {
    float cp[28] = {};
    id<MTLBuffer> buffer =
        [device newBufferWithBytes:cp
                            length:sizeof(cp)
                           options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->ClipPlaneBuffer, buffer);
  }
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::RenderPiece(vtkRenderer* ren, vtkActor* act)
{
  vtkMetalRenderWindow* renWin = vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());
  if (!renWin || !renWin->GetMetalDevice())
  {
    return;
  }

  // Cache render window for use by Ensure* methods
  this->Internals->CachedRenderWindow = renWin;

  vtkPolyData* input = this->GetInput();
  if (!input)
  {
    return;
  }

  // 8A: Invalidate all pipeline states when MSAA sample count changes
  int currentSampleCount = renWin->GetEffectiveSampleCount();
  if (currentSampleCount != this->Internals->CachedSampleCount)
  {
    this->Internals->ReleasePipelines();
    this->Internals->CachedSampleCount = currentSampleCount;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (id<MTLDevice>)renWin->GetMetalDevice();

    vtkIdType currentMTime = input->GetMTime();
    vtkMTimeType extraMTime = this->ExtraAttributesMTime.GetMTime();
    int representation = act->GetProperty()->GetRepresentation();
    bool edgeVisibility = act->GetProperty()->GetEdgeVisibility();
    float lineWidth = static_cast<float>(act->GetProperty()->GetLineWidth());

    vtkMTimeType scalarMTime = this->GetMTime();
    if (this->GetLookupTable())
    {
      scalarMTime = std::max(scalarMTime,
        static_cast<vtkMTimeType>(this->GetLookupTable()->GetMTime()));
    }

    // P2-1: Include actor property MTime so visual property changes force geometry rebuild
    {
      vtkProperty* prop = act ? act->GetProperty() : nullptr;
      vtkMTimeType propertyMTime = prop ? prop->GetMTime() : 0;
      scalarMTime = std::max(scalarMTime, propertyMTime);
    }

    vtkMTimeType batchOverrideMTime = this->Internals->BatchOverrideMTime;

    if (currentMTime != this->Internals->CachedInputMTime ||
        representation != this->Internals->CachedRepresentation ||
        edgeVisibility != this->Internals->CachedEdgeVisibility ||
        extraMTime != this->Internals->CachedExtraAttributesMTime ||
        scalarMTime != this->Internals->CachedScalarMTime ||
        batchOverrideMTime != this->Internals->CachedBatchOverrideMTime)
    {
      this->Internals->ReleaseBuffers();
      this->Internals->CachedInputMTime = currentMTime;
      this->Internals->CachedRepresentation = representation;
      this->Internals->CachedEdgeVisibility = edgeVisibility;
      this->Internals->CachedExtraAttributesMTime = extraMTime;
      this->Internals->CachedScalarMTime = scalarMTime;
      this->Internals->CachedBatchOverrideMTime = batchOverrideMTime;
      this->BuildGeometryBuffers((void*)device, input, act);
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

    bool needPointPipelines =
      (representation == VTK_POINTS) ||
      (act->GetProperty()->GetVertexVisibility() &&
       this->Internals->PointVertexCount > 0);

    bool needTrianglePipeline =
      (representation == VTK_SURFACE) &&
      this->Internals->HasTriangles;

    bool needLinePipeline =
      (representation == VTK_SURFACE || representation == VTK_WIREFRAME) &&
      this->Internals->HasLines;

    bool needSurfacePipelines =
      needTrianglePipeline || needLinePipeline;

    if (needSurfacePipelines)
    {
      this->EnsurePipelineStates((void*)device);
    }

    if (needPointPipelines)
    {
      this->EnsurePointPipelineStates((void*)device);
    }

    // 8C: Ensure all pipeline states are created before render bundle recording.
    // These were previously created lazily inside inline draw code.
    if (representation == VTK_SURFACE &&
        this->Internals->HasEdgeOverlay)
    {
      this->EnsureEdgePipelineState((void*)device);
      if (act->GetProperty()->GetEdgeVisibility() &&
          act->GetProperty()->GetLineWidth() > 1.0f &&
          this->Internals->EdgeTubeSegmentCount > 0)
      {
        if (act->GetProperty()->GetLineJoin() == vtkProperty::LineJoinType::MiterJoin)
        {
          this->EnsureMiterJoinLinePipelineState((void*)device);
        }
        else
        {
          this->EnsureThickLinePipelineState((void*)device);
        }
      }
    }
    if (this->Internals->HasLines && this->Internals->LineIndexBuffer)
    {
      float lw = static_cast<float>(act->GetProperty()->GetLineWidth());
      if (lw > 1.0f)
      {
        auto lj = act->GetProperty()->GetLineJoin();
        if (lj == vtkProperty::LineJoinType::MiterJoin &&
            this->Internals->MiterJoinLineSegmentCount > 0)
        {
          this->EnsureMiterJoinLinePipelineState((void*)device);
        }
        else if (this->Internals->ThickLineSegmentCount > 0)
        {
          // Fallback: thick lines for round-cap and all other non-miter joins
          this->EnsureThickLinePipelineState((void*)device);
        }
      }
    }
    if (renWin->DepthPeelingMode != 0)
    {
      this->EnsurePeelPipelineStates((void*)device);
    }
    if (renWin->OITActive)
    {
      this->EnsureOITPipelineStates((void*)device);
    }

    // Use the encoder already created by vtkMetalRenderer::DeviceRender().
    // Do NOT create a new render pass, command buffer, or drawable here.
    id<MTLRenderCommandEncoder> encoder =
      (id<MTLRenderCommandEncoder>)renWin->GetCurrentRenderCommandEncoder();
    if (!encoder)
    {
      vtkErrorMacro(<< "No active render command encoder. "
                    << "RenderPiece must be called within DeviceRender.");
      return;
    }

    [encoder setFrontFacingWinding:MTLWindingCounterClockwise];

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

      // The cached camera transforms carry the camera's model matrix (identity).
      // The shader applies scene.modelMatrix to model-space vertices, so store the
      // actor's model-to-world matrix here (transposed like the camera matrices,
      // since Metal indexes matrices column-major).
      {
        vtkNew<vtkMatrix4x4> actorMatrix;
        act->GetModelToWorldMatrix(actorMatrix);
        float* modelMat = reinterpret_cast<float*>(buf + 176);
        for (int col = 0; col < 4; ++col)
        {
          for (int row = 0; row < 4; ++row)
          {
            modelMat[col * 4 + row] = static_cast<float>(actorMatrix->GetElement(row, col));
          }
        }
      }

      float ptSize = static_cast<float>(act->GetProperty()->GetPointSize());
      *reinterpret_cast<float*>(buf + 260) = ptSize;

      // Update actor texture before computing flags so the flag reflects current state
      this->UpdateActorTexture((void*)device, act);

      // Merge actor render option flags into SceneUniforms flags (offset 256).
      // Bit 0: parallel projection (set by camera)
      // Bit 3: vertex visibility
      // Bit 5: RenderPointsAsSpheres
      // Bit 7: Point2DShape (0=Round, 1=Square)
      // Bit 8: has surface vertex colors (P1-1A/1B)
      // Bit 9: has actor texture
      vtkProperty* prop = act->GetProperty();
      uint32_t flags = *reinterpret_cast<uint32_t*>(buf + 256);
      flags &= ~VTK_METAL_DYNAMIC_ACTOR_FLAG_MASK;

      uint32_t actorFlags = 0;

      if (prop->GetVertexVisibility())
      {
        actorFlags |= VTK_METAL_SCENE_FLAG_VERTEX_VISIBILITY;
      }

      if (prop->GetRenderPointsAsSpheres())
      {
        actorFlags |= VTK_METAL_SCENE_FLAG_SPHERE_POINTS;
      }

      if (static_cast<uint32_t>(prop->GetPoint2DShape()) != 0u)
      {
        actorFlags |= VTK_METAL_SCENE_FLAG_POINT_SHAPE;
      }

      if (this->Internals->HasSurfaceColors)
      {
        actorFlags |= VTK_METAL_SCENE_FLAG_HAS_SURFACE_COLORS;
      }

      if (this->Internals->HasSurfaceAlpha)
      {
        actorFlags |= VTK_METAL_SCENE_FLAG_HAS_SURFACE_ALPHA;
      }

      if (this->Internals->ActorTexture)
      {
        actorFlags |= VTK_METAL_SCENE_FLAG_HAS_ACTOR_TEXTURE;
      }

      flags |= actorFlags;
      *reinterpret_cast<uint32_t*>(buf + 256) = flags;
    }

    this->UpdateMaterialUniforms((void*)device, act);
    this->UpdateLightUniforms((void*)device, ren);
    this->UpdateCoincidentOffsetUniforms((void*)device, act);
    this->UpdateVertexColorUniforms((void*)device, act);
    this->UpdateClipPlaneUniforms((void*)device, act);

    // P1-2: Create fallback edge UV buffer if edge overlay is active but no UVs were built
    if (this->Internals->HasEdgeOverlay &&
        this->Internals->EdgeVertexCount > 0 &&
        !this->Internals->EdgeUVBuffer)
    {
      size_t uvCount = static_cast<size_t>(this->Internals->EdgeVertexCount) * 2;
      id<MTLBuffer> buffer = CreateZeroBuffer(device, uvCount * sizeof(float));
      vtkMetalMRC::AssignConsumed(this->Internals->EdgeUVBuffer, buffer);
    }

    // P1-3: Ensure fallback buffers exist for all shader-required bindings
    this->EnsureRequiredBindingFallbacks((void*)device);

    // Picking IDs: during a selection pass write the prop's per-render index
    // (from the hardware selector); otherwise fall back to overrides/0.
    this->UpdatePickUniforms(ren, act);

    // 8C: Update edge color uniform before bundle recording (per-frame update).
    // UpdateEdgeUniforms runs unconditionally so fragment buffer(4) is always
    // valid for every pipeline built on fragment_main (including line draws).
    this->UpdateEdgeUniforms((void*)device, act);
    if (this->UseLegacyEdgeOverlay && representation == VTK_SURFACE &&
        edgeVisibility &&
        this->Internals->HasEdgeOverlay)
    {
      this->UpdateEdgeColorUniform((void*)device, act);
    }

    // P3-3A: Create thick-line width and miter segment count buffers
    // These are required by the thick-line and miter-join shaders.
    {
      float lw = static_cast<float>(act->GetProperty()->GetLineWidth());
      if (!this->Internals->ThickLineLineWidthBuffer)
      {
        id<MTLBuffer> buffer =
          [device newBufferWithLength:sizeof(float)
                              options:MTLResourceStorageModeShared];
        vtkMetalMRC::AssignConsumed(this->Internals->ThickLineLineWidthBuffer, buffer);
      }
      if (this->Internals->ThickLineLineWidthBuffer)
      {
        memcpy([this->Internals->ThickLineLineWidthBuffer contents], &lw, sizeof(lw));
      }
    }
    {
      uint32_t segCount = static_cast<uint32_t>(this->Internals->MiterJoinLineSegmentCount);
      if (!this->Internals->MiterJoinSegmentCountBuffer)
      {
        id<MTLBuffer> buffer =
          [device newBufferWithLength:sizeof(uint32_t)
                              options:MTLResourceStorageModeShared];
        vtkMetalMRC::AssignConsumed(this->Internals->MiterJoinSegmentCountBuffer, buffer);
      }
      if (this->Internals->MiterJoinSegmentCountBuffer)
      {
        memcpy([this->Internals->MiterJoinSegmentCountBuffer contents], &segCount, sizeof(segCount));
      }
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
      [sDesc release];
    }

    // 8C: Render bundle — check if cached encoder commands can be replayed.
    // Bundle is valid when geometry, representation, edge visibility, line width,
    // MSAA sample count, and depth peeling mode all match the values at bundle creation.
    // When valid, replay cached commands instead of re-encoding all draw calls.
    // Uniform buffers (scene, material, light, etc.) are updated in-place each frame,
    // so replaying the same buffer bindings reads the latest content automatically.
    vtkProperty* prop = act->GetProperty();
    bool vertexVisibility = prop->GetVertexVisibility();
    float pointSize = static_cast<float>(prop->GetPointSize());
    int lineJoin = static_cast<int>(prop->GetLineJoin());
    bool backfaceCulling = prop->GetBackfaceCulling();
    bool frontfaceCulling = prop->GetFrontfaceCulling();
    bool renderPointsAsSpheres = prop->GetRenderPointsAsSpheres();
    int point2DShape = static_cast<int>(prop->GetPoint2DShape());

    bool hasActorTexture = (this->Internals->ActorTexture != nil);
    vtkMTimeType textureMTime = this->Internals->CachedTextureMTime;

    int currentPeelMode = renWin->DepthPeelingMode;
    bool currentOITActive = renWin->OITActive;

    bool bundleValid =
        this->Internals->Bundle.Valid &&
        this->Internals->BundleGeometryMTime == this->Internals->CachedInputMTime &&
        this->Internals->BundleRepresentation == representation &&
        this->Internals->BundleEdgeVisibility == edgeVisibility &&
        this->Internals->BundleLineWidth == lineWidth &&
        this->Internals->BundleSampleCount == currentSampleCount &&
        this->Internals->BundlePeelMode == currentPeelMode &&
        this->Internals->BundleOITActive == currentOITActive &&
        this->Internals->BundleTextureMTime == textureMTime &&
        this->Internals->BundleHasActorTexture == hasActorTexture &&
        this->Internals->BundleVertexVisibility == vertexVisibility &&
        this->Internals->BundlePointSize == pointSize &&
        this->Internals->BundleRenderPointsAsSpheres == renderPointsAsSpheres &&
        this->Internals->BundlePoint2DShape == point2DShape &&
        this->Internals->BundleLineJoin == lineJoin &&
        this->Internals->BundleBackfaceCulling == backfaceCulling &&
        this->Internals->BundleFrontfaceCulling == frontfaceCulling &&
        this->Internals->BundleExtraAttributesMTime == extraMTime &&
        this->Internals->BundleBatchOverrideMTime == batchOverrideMTime;

    // P1-4: Disable bundle caching during peel passes to avoid stale texture bindings
    const bool allowBundleCaching = (currentPeelMode == 0 && !currentOITActive);

    if (allowBundleCaching && bundleValid)
    {
      this->ReplayRenderBundle((void*)encoder);
    }
    else
    {
      this->RebuildRenderBundle((void*)encoder, ren, act);
      this->ReplayRenderBundle((void*)encoder);

      if (!allowBundleCaching)
      {
        this->Internals->Bundle.Valid = false;
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

  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;
  vtkProperty* bprop = actor ? actor->GetProperty() : nullptr;
  int rep = bprop ? bprop->GetRepresentation() : VTK_SURFACE;
  bool edgeVis = bprop ? bprop->GetEdgeVisibility() : false;
  // Single-pass edge pipeline-selection key: edges folded into fragment_main.
  this->Internals->SurfaceUsesIndexedEntry =
    edgeVis && rep == VTK_SURFACE && !this->UseLegacyEdgeOverlay;

  // The triangle/peel pipelines bake the indexed-entry vertex function. When
  // the key flips at runtime (e.g. the UseLegacyEdgeOverlay A/B switch), the
  // cached pipelines must be invalidated so the rebuild picks the right vertex
  // entry. This is the single point where the key is recomputed each frame,
  // so it is the correct place to release on a flip.
  if (this->Internals->SurfaceUsesIndexedEntry !=
      this->Internals->CachedSurfaceUsesIndexedEntry)
  {
    vtkMetalMRC::ReleaseAndNil(this->Internals->TrianglePipeline);
    vtkMetalMRC::ReleaseAndNil(this->Internals->TriangleInitPeelPipeline);
    vtkMetalMRC::ReleaseAndNil(this->Internals->TrianglePeelPipeline);
    vtkMetalMRC::ReleaseAndNil(this->Internals->TriangleOITPipeline);
    this->Internals->CachedSurfaceUsesIndexedEntry =
      this->Internals->SurfaceUsesIndexedEntry;
  }

  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<float> surfaceColors;  // P1-1A/1B: float4 per vertex
  std::vector<float> triangleUVs;    // P5-5A: float2 per vertex
  std::vector<uint32_t> lineIndices;

  // P2-2C: Triangle index buffers — deduplicated vertices + index buffer
  std::vector<uint32_t> triangleIndices;
  std::unordered_map<vtkIdType, uint32_t> triVertexMap;

  // Single-pass edges: per-triangle-corner bary + boundary flags (parallel to triangleIndices)
  std::vector<float> triangleBary;
  std::vector<uint32_t> triangleEdgeFlags;
  std::vector<float> trianglePos;   // float3[3] corner object positions per corner record
  static const float kBary[3][3] = { {1,0,0}, {0,1,0}, {0,0,1} };
  // Host mirror of the kernel's isBoundary: true iff (a,b) is a consecutive polygon pair.
  auto edgeIsBoundary = [&](vtkIdType a, vtkIdType b, vtkIdType npts, const vtkIdType* pts) -> bool {
    for (vtkIdType k = 0; k < npts; ++k)
    {
      vtkIdType x = pts[k];
      vtkIdType y = pts[(k + 1) % npts];
      if ((x == a && y == b) || (x == b && y == a)) return true;
    }
    return false;
  };
  // Bit order (c1c2, c2c0, c0c1): bit0 = edge opposite corner 0, bit1 = corner 1, bit2 = corner 2.
  auto packedEdgeFlags = [&](const vtkIdType tri[3], vtkIdType npts, const vtkIdType* pts) -> uint32_t {
    return (edgeIsBoundary(tri[1], tri[2], npts, pts) ? 1u : 0u)
         | (edgeIsBoundary(tri[2], tri[0], npts, pts) ? 2u : 0u)
         | (edgeIsBoundary(tri[0], tri[1], npts, pts) ? 4u : 0u);
  };

  // P2-2B: Edge geometry for wireframe overlay on surfaces (separate vertex + index buffers)
  std::vector<float> edgePositions;
  std::vector<float> edgeNormals;
  std::vector<float> edgeColors;
  std::vector<float> edgeUVs;              // P1-2: edge overlay UV coordinates (float2 per vertex)
  std::vector<uint32_t> edgeIndices;
  std::unordered_map<vtkIdType, uint32_t> edgeVertexMap;

  // P2-8: per-vertex cell IDs (replaces per-primitive cell ID mapping)
  std::vector<uint32_t> triangleVertexCellIds;
  std::vector<uint32_t> lineVertexCellIds;       // per-vertex, parallel to positions (standard 1px lines, read by vertex_id)
  std::vector<uint32_t> lineSegmentCellIds;      // per-segment (thick/round/miter lines, read by instance_id)
  std::vector<uint32_t> edgeVertexCellIds;
  std::vector<uint32_t> edgeTubeIndices;        // per-polygon closed-loop segment pairs for edge tubes
  std::vector<uint32_t> edgeTubeCellIds;        // per-segment cell id, parallel to edgeTubeIndices/2

  // P10-10A: Extra attribute arrays — parallel to positions (one value per rendered vertex)
  std::unordered_map<std::string, std::vector<float>> extraAttrArrays;
  for (const auto& attr : this->ExtraAttributes)
  {
    extraAttrArrays[attr.first] = std::vector<float>();
  }
  auto emitExtraAttrsForPoint = [&](vtkIdType pointId) {
    for (const auto& attr : this->ExtraAttributes)
    {
      // Each lambda is gated by field association so the point and cell paths
      // never double-emit for the same attribute (which would push one extra
      // zero per vertex and break the positions-parallel invariant).
      if (attr.second.FieldAssociation != vtkDataObject::FIELD_ASSOCIATION_POINTS)
      {
        continue;
      }
      vtkDataArray* da = polydata->GetPointData()->GetArray(attr.second.DataArrayName.c_str());
      if (da && pointId < da->GetNumberOfTuples())
      {
        int numComps = da->GetNumberOfComponents();
        int comp = attr.second.ComponentNumber;
        if (comp < 0)
        {
          double* tuple = da->GetTuple(pointId);
          for (int c = 0; c < numComps; ++c)
          {
            extraAttrArrays[attr.first].push_back(static_cast<float>(tuple[c]));
          }
        }
        else
        {
          extraAttrArrays[attr.first].push_back(
            static_cast<float>(da->GetComponent(pointId, comp)));
        }
      }
      else
      {
        // Zero-fill to stay parallel with positions (missing/OOB point array).
        int effectiveComps = (attr.second.ComponentNumber < 0)
          ? (da ? da->GetNumberOfComponents() : 1) : 1;
        for (int c = 0; c < effectiveComps; ++c)
        {
          extraAttrArrays[attr.first].push_back(0.0f);
        }
      }
    }
  };
  auto emitExtraAttrsForCell = [&](vtkIdType cellId) {
    for (const auto& attr : this->ExtraAttributes)
    {
      if (attr.second.FieldAssociation != vtkDataObject::FIELD_ASSOCIATION_CELLS)
      {
        continue;
      }
      vtkDataArray* da = polydata->GetCellData()->GetArray(attr.second.DataArrayName.c_str());
      if (da && cellId >= 0 && cellId < da->GetNumberOfTuples())
      {
        int numComps = da->GetNumberOfComponents();
        int comp = attr.second.ComponentNumber;
        if (comp < 0)
        {
          double* tuple = da->GetTuple(cellId);
          for (int c = 0; c < numComps; ++c)
          {
            extraAttrArrays[attr.first].push_back(static_cast<float>(tuple[c]));
          }
        }
        else
        {
          extraAttrArrays[attr.first].push_back(
            static_cast<float>(da->GetComponent(cellId, comp)));
        }
      }
      else
      {
        // Zero-fill to stay parallel with positions (missing/OOB cell array).
        int effectiveComps = (attr.second.ComponentNumber < 0)
          ? (da ? da->GetNumberOfComponents() : 1) : 1;
        for (int c = 0; c < effectiveComps; ++c)
        {
          extraAttrArrays[attr.first].push_back(0.0f);
        }
      }
    }
  };

  // P10-10B: Cell-associated extra attributes are per-primitive — a shared
  // vertex cannot carry two different cell values. When any is mapped, force the
  // CPU, non-indexed triangle path and disable GPU tessellation so every
  // rendered vertex is unique and can carry its own cell's value. (CPU line and
  // wireframe paths deduplicate vertices by point ID, so a point shared by two
  // cells carries the first cell's value — inherent to deduplication, and those
  // pipelines do not bind extra-attribute buffers anyway.)
  bool hasCellAssociatedExtraAttrs = false;
  for (const auto& attr : this->ExtraAttributes)
  {
    if (attr.second.FieldAssociation == vtkDataObject::FIELD_ASSOCIATION_CELLS)
    {
      hasCellAssociatedExtraAttrs = true;
      break;
    }
  }

  vtkIdType lineCellOffset = polydata->GetNumberOfVerts();
  vtkIdType polyCellOffset = lineCellOffset + polydata->GetNumberOfLines();

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
  // When batch color override is active, disable scalar coloring.
  int cellFlag = 0;
  vtkUnsignedCharArray* mappedColors = nullptr;
  if (actor && !this->Internals->UseBatchColor)
  {
    mappedColors = this->MapScalars(actor->GetProperty()->GetOpacity(), cellFlag);
  }

  // Helper to get override color or default actor color
  double actorOpacity = actor ? actor->GetProperty()->GetOpacity() : 1.0;
  auto getOverrideOrDefaultRGBA = [&](float rgba[4])
  {
    if (this->Internals->UseBatchColor)
    {
      rgba[0] = static_cast<float>(this->Internals->BatchColor[0]);
      rgba[1] = static_cast<float>(this->Internals->BatchColor[1]);
      rgba[2] = static_cast<float>(this->Internals->BatchColor[2]);
    }
    else if (actor)
    {
      double c[3];
      actor->GetProperty()->GetColor(c);
      rgba[0] = static_cast<float>(c[0]);
      rgba[1] = static_cast<float>(c[1]);
      rgba[2] = static_cast<float>(c[2]);
    }
    else
    {
      rgba[0] = rgba[1] = rgba[2] = 1.0f;
    }

    if (this->Internals->UseBatchOpacity)
    {
      rgba[3] = static_cast<float>(this->Internals->BatchOpacity);
    }
    else if (this->Internals->UseBatchColor)
    {
      // Color override without explicit opacity override: bake actor opacity
      // because material opacity is forced to 1.0 when batch overrides are active.
      rgba[3] = static_cast<float>(actorOpacity);
    }
    else
    {
      // Non-override default path: let material opacity apply once.
      // Thick-line shaders multiply vertexColor.a by material.opacity,
      // so this must remain 1.0 to avoid double multiplication.
      rgba[3] = 1.0f;
    }
  };

  // Helper: emit one vertex color (float4) into surfaceColors, respecting batch overrides.
  // When mappedColors is provided and no batch color override, use it (with optional opacity override).
  auto emitSurfaceColor = [&](vtkIdType idx, const unsigned char* colors)
  {
    if (colors && !this->Internals->UseBatchColor)
    {
      float r = colors[idx * 4] / 255.0f;
      float g = colors[idx * 4 + 1] / 255.0f;
      float b = colors[idx * 4 + 2] / 255.0f;
      float a = colors[idx * 4 + 3] / 255.0f;
      if (this->Internals->UseBatchOpacity)
      {
        a = static_cast<float>(this->Internals->BatchOpacity);
      }
      surfaceColors.push_back(r);
      surfaceColors.push_back(g);
      surfaceColors.push_back(b);
      surfaceColors.push_back(a);
    }
    else
    {
      float rgba[4];
      getOverrideOrDefaultRGBA(rgba);
      surfaceColors.push_back(rgba[0]);
      surfaceColors.push_back(rgba[1]);
      surfaceColors.push_back(rgba[2]);
      surfaceColors.push_back(rgba[3]);
    }
  };

  // ---- P6-6A: GPU Tessellation Path ----
  // When per-point coloring (cellFlag == 0) with data normals, use compute shaders
  // for polygon → triangle fan tessellation, edge array generation, and line segment
  // extraction. Moves fan triangulation off the CPU and produces edge arrays for free.
  // Falls back to CPU for per-cell coloring, computed normals, or small geometries.
  vtkCellArray* polys = polydata->GetPolys();
  vtkIdType numPolyPts = polydata->GetNumberOfPoints();
  bool useGPUTess = (cellFlag == 0) && normalArray && (numPolyPts > 1000) &&
    !hasCellAssociatedExtraAttrs;
  bool gpuTessUsed = false;

  if (useGPUTess)
  {
    // Shared command buffer for batching all GPU tessellation dispatches
    id<MTLCommandQueue> computeQueue = this->Internals->EnsureComputeQueue(device);
    id<MTLCommandBuffer> computeCmdBuf = [computeQueue commandBuffer];
    computeCmdBuf.label = @"VTK Geometry Tessellation";
    std::vector<id<MTLBuffer>> tempInputBuffers;
    bool encPolygonTess = false;
    bool encEdgeVis = false;
    bool encWireframe = false;
    vtkIdType numTris = 0;
    vtkIdType numEdges = 0;
    id<MTLBuffer> edgeOutBuf = nil, edgeCellIdBuf = nil;
    id<MTLBuffer> wireOutBuf = nil, wireCellIdBuf = nil;

    // Step 2: Build polygon connectivity arrays for triangle tessellation.
    if (representation != VTK_WIREFRAME && representation != VTK_POINTS && polys && polys->GetNumberOfCells() > 0)
    {
      std::vector<uint32_t> polyConn;
      std::vector<uint32_t> polyOff;
      std::vector<uint32_t> polyPrimCounts;
      polyOff.push_back(0);
      numTris = 0;

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
          id<MTLLibrary> library = (__bridge id<MTLLibrary>)
            this->Internals->CachedRenderWindow->GetSharedShaderLibrary();
          if (library)
          {
            id<MTLFunction> func = [library newFunctionWithName:@"polygonToTriangle"];
            if (func)
            {
              NSError* error = nil;
              this->Internals->PolygonToTrianglePipeline =
                [device newComputePipelineStateWithFunction:func error:&error];
              [func release];
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
          id<MTLBuffer> tessConnBuf = [device
            newBufferWithLength:numTris * 3 * sizeof(uint32_t)
                       options:MTLResourceStorageModeShared];
          vtkMetalMRC::AssignConsumed(this->Internals->TessOutputConnectivityBuffer, tessConnBuf);
          id<MTLBuffer> tessEdgeBuf = [device
            newBufferWithLength:numTris * sizeof(float)
                       options:MTLResourceStorageModeShared];
          vtkMetalMRC::AssignConsumed(this->Internals->TessEdgeArrayBuffer, tessEdgeBuf);
          id<MTLBuffer> triCellIdBuf = [device
            newBufferWithLength:numTris * sizeof(uint32_t)
                       options:MTLResourceStorageModeShared];
          vtkMetalMRC::AssignConsumed(this->Internals->TriangleCellIdBuffer, triCellIdBuf);

          // Single-pass edges: per-triangle-corner bary + boundary flags.
          // Allocated unconditionally because polygonToTriangle always writes them.
          {
            id<MTLBuffer> tessBaryBuf = [device
              newBufferWithLength:numTris * 3 * sizeof(float) * 3
                         options:MTLResourceStorageModeShared];
            vtkMetalMRC::AssignConsumed(this->Internals->TriangleBaryBuffer, tessBaryBuf);
            id<MTLBuffer> tessFlagBuf = [device
              newBufferWithLength:numTris * 3 * sizeof(uint32_t)
                         options:MTLResourceStorageModeShared];
            vtkMetalMRC::AssignConsumed(this->Internals->TriangleEdgeFlagBuffer, tessFlagBuf);
          }

          // Upload params uniform
          struct { uint32_t numCells; uint32_t cellIdOffset; } tessParams;
          tessParams.numCells = static_cast<uint32_t>(polyOff.size() - 1);
          tessParams.cellIdOffset = static_cast<uint32_t>(polyCellOffset);
          id<MTLBuffer> tessParamsBuf = [device
            newBufferWithBytes:&tessParams
                       length:sizeof(tessParams)
                      options:MTLResourceStorageModeShared];
          vtkMetalMRC::AssignConsumed(this->Internals->TessParamsBuffer, tessParamsBuf);

          // --- Encode polygonToTriangle ---
          {
            id<MTLComputeCommandEncoder> enc = [computeCmdBuf computeCommandEncoder];
            [enc setComputePipelineState:this->Internals->PolygonToTrianglePipeline];
            [enc setBuffer:this->Internals->TessOutputConnectivityBuffer offset:0 atIndex:0];
            [enc setBuffer:this->Internals->TessEdgeArrayBuffer offset:0 atIndex:1];
            [enc setBuffer:this->Internals->TriangleCellIdBuffer offset:0 atIndex:2];
            [enc setBuffer:this->Internals->TriangleBaryBuffer offset:0 atIndex:7];
            [enc setBuffer:this->Internals->TriangleEdgeFlagBuffer offset:0 atIndex:8];
            [enc setBuffer:connBuf offset:0 atIndex:3];
            [enc setBuffer:offBuf offset:0 atIndex:4];
            [enc setBuffer:primBuf offset:0 atIndex:5];
            [enc setBuffer:this->Internals->TessParamsBuffer offset:0 atIndex:6];
            NSUInteger tgMax = this->Internals->PolygonToTrianglePipeline.maxTotalThreadsPerThreadgroup;
            MTLSize grid = MTLSizeMake(tessParams.numCells, 1, 1);
            NSUInteger gW = static_cast<NSUInteger>(tessParams.numCells);
            MTLSize tg = MTLSizeMake(std::min(tgMax, gW), 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];

            tempInputBuffers.push_back(connBuf);
            tempInputBuffers.push_back(offBuf);
            tempInputBuffers.push_back(primBuf);
            encPolygonTess = true;
          }
        }
      }

      // Step 3: Build polygon edge connectivity for edge visibility (legacy overlay only)
      if (edgeVisibility && this->UseLegacyEdgeOverlay)
      {
        std::vector<uint32_t> eConn;
        std::vector<uint32_t> eOff;
        std::vector<uint32_t> ePrimCounts;
        eOff.push_back(0);
        numEdges = 0;

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
            id<MTLLibrary> library = (__bridge id<MTLLibrary>)
              this->Internals->CachedRenderWindow->GetSharedShaderLibrary();
            if (library)
            {
              id<MTLFunction> func = [library newFunctionWithName:@"polygonEdgesToLines"];
              if (func)
              {
                NSError* error = nil;
                this->Internals->PolygonEdgesToLinesPipeline =
                  [device newComputePipelineStateWithFunction:func error:&error];
                [func release];
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

            edgeOutBuf = [device
              newBufferWithLength:numEdges * 2 * sizeof(uint32_t)
                         options:MTLResourceStorageModeShared];
            edgeCellIdBuf = [device
              newBufferWithLength:numEdges * sizeof(uint32_t)
                         options:MTLResourceStorageModeShared];

            struct { uint32_t numCells; uint32_t cellIdOffset; } eParams;
            eParams.numCells = static_cast<uint32_t>(eOff.size() - 1);
            eParams.cellIdOffset = static_cast<uint32_t>(polyCellOffset);
            id<MTLBuffer> eParamsBuf = [device
              newBufferWithBytes:&eParams
                         length:sizeof(eParams)
                        options:MTLResourceStorageModeShared];

            // --- Encode polygonEdgesToLines (edge visibility) ---
            id<MTLComputeCommandEncoder> enc = [computeCmdBuf computeCommandEncoder];
            [enc setComputePipelineState:this->Internals->PolygonEdgesToLinesPipeline];
            [enc setBuffer:edgeOutBuf offset:0 atIndex:0];
            [enc setBuffer:edgeCellIdBuf offset:0 atIndex:1];
            [enc setBuffer:eConnBuf offset:0 atIndex:2];
            [enc setBuffer:eOffBuf offset:0 atIndex:3];
            [enc setBuffer:ePrimBuf offset:0 atIndex:4];
            [enc setBuffer:eParamsBuf offset:0 atIndex:5];
            NSUInteger tgMax = this->Internals->PolygonEdgesToLinesPipeline.maxTotalThreadsPerThreadgroup;
            MTLSize grid = MTLSizeMake(eParams.numCells, 1, 1);
            NSUInteger gW = static_cast<NSUInteger>(eParams.numCells);
            MTLSize tg = MTLSizeMake(std::min(tgMax, gW), 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];

            tempInputBuffers.push_back(eConnBuf);
            tempInputBuffers.push_back(eOffBuf);
            tempInputBuffers.push_back(ePrimBuf);
            tempInputBuffers.push_back(eParamsBuf);
            encEdgeVis = true;
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
      numEdges = 0;

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
          id<MTLLibrary> library = (__bridge id<MTLLibrary>)
            this->Internals->CachedRenderWindow->GetSharedShaderLibrary();
          if (library)
          {
            id<MTLFunction> func = [library newFunctionWithName:@"polygonEdgesToLines"];
            if (func)
            {
              NSError* error = nil;
              this->Internals->PolygonEdgesToLinesPipeline =
                [device newComputePipelineStateWithFunction:func error:&error];
              [func release];
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

          wireOutBuf = [device
            newBufferWithLength:numEdges * 2 * sizeof(uint32_t)
                       options:MTLResourceStorageModeShared];
          wireCellIdBuf = [device
            newBufferWithLength:numEdges * sizeof(uint32_t)
                       options:MTLResourceStorageModeShared];

          struct { uint32_t numCells; uint32_t cellIdOffset; } wParams;
          wParams.numCells = static_cast<uint32_t>(wOff.size() - 1);
          wParams.cellIdOffset = static_cast<uint32_t>(polyCellOffset);
          id<MTLBuffer> wParamsBuf = [device
            newBufferWithBytes:&wParams
                       length:sizeof(wParams)
                      options:MTLResourceStorageModeShared];

          // --- Encode polygonEdgesToLines (wireframe) ---
          id<MTLComputeCommandEncoder> enc = [computeCmdBuf computeCommandEncoder];
          [enc setComputePipelineState:this->Internals->PolygonEdgesToLinesPipeline];
          [enc setBuffer:wireOutBuf offset:0 atIndex:0];
          [enc setBuffer:wireCellIdBuf offset:0 atIndex:1];
          [enc setBuffer:wConnBuf offset:0 atIndex:2];
          [enc setBuffer:wOffBuf offset:0 atIndex:3];
          [enc setBuffer:wPrimBuf offset:0 atIndex:4];
          [enc setBuffer:wParamsBuf offset:0 atIndex:5];
          NSUInteger tgMax = this->Internals->PolygonEdgesToLinesPipeline.maxTotalThreadsPerThreadgroup;
          MTLSize grid = MTLSizeMake(wParams.numCells, 1, 1);
          NSUInteger gW = static_cast<NSUInteger>(wParams.numCells);
          MTLSize tg = MTLSizeMake(std::min(tgMax, gW), 1, 1);
          [enc dispatchThreads:grid threadsPerThreadgroup:tg];
          [enc endEncoding];

          tempInputBuffers.push_back(wConnBuf);
          tempInputBuffers.push_back(wOffBuf);
          tempInputBuffers.push_back(wPrimBuf);
          tempInputBuffers.push_back(wParamsBuf);
          encWireframe = true;
        }
      }
    }

    // --- Single commit + wait for all encoded dispatches ---
    [computeCmdBuf commit];
    [computeCmdBuf waitUntilCompleted];

    if (computeCmdBuf.status == MTLCommandBufferStatusCompleted && encPolygonTess)
    {
      // Use compute output as triangle index buffer
      vtkMetalMRC::AssignRetained(
          this->Internals->IndexBuffer,
          this->Internals->TessOutputConnectivityBuffer);
      this->Internals->TriangleVertexCount = numPolyPts;
      this->Internals->TriangleIndexCount = numTris * 3;
      this->Internals->HasTriangles = true;
      this->Internals->TrianglePrimitiveCount = numTris;

      // Expand per-triangle cell IDs to per-point for vertex indexing
      const uint32_t* triCellIds =
          (const uint32_t*)[this->Internals->TriangleCellIdBuffer contents];
      const uint32_t* connData =
          (const uint32_t*)[this->Internals->TessOutputConnectivityBuffer contents];
      std::vector<uint32_t> pointCellIds(numPolyPts, 0);
      std::vector<bool> pointAssigned(numPolyPts, false);
      for (vtkIdType t = 0; t < numTris; ++t)
      {
        uint32_t cid = triCellIds[t];
        for (int v = 0; v < 3; ++v)
        {
          uint32_t ptIdx = connData[t * 3 + v];
          if (ptIdx < (uint32_t)numPolyPts && !pointAssigned[ptIdx])
          {
            pointCellIds[ptIdx] = cid;
            pointAssigned[ptIdx] = true;
          }
        }
      }
      id<MTLBuffer> perPointCellIds = [device
          newBufferWithBytes:pointCellIds.data()
                     length:pointCellIds.size() * sizeof(uint32_t)
                    options:MTLResourceStorageModeShared];
      vtkMetalMRC::AssignConsumed(this->Internals->TriangleCellIdBuffer, perPointCellIds);
      gpuTessUsed = true;
    }

    if (computeCmdBuf.status == MTLCommandBufferStatusCompleted && encEdgeVis)
    {
      vtkMetalMRC::AssignConsumed(this->Internals->EdgeIndexBuffer, edgeOutBuf);

      const uint32_t* segEdgeCellIds = (const uint32_t*)[edgeCellIdBuf contents];
      const uint32_t* edgeConn = (const uint32_t*)[edgeOutBuf contents];
      std::vector<uint32_t> perPointEdgeCellIds(static_cast<size_t>(numPolyPts), 0);
      for (vtkIdType s = 0; s < numEdges; ++s)
      {
        uint32_t cid = segEdgeCellIds[s];
        uint32_t p0 = edgeConn[static_cast<size_t>(s) * 2];
        uint32_t p1 = edgeConn[static_cast<size_t>(s) * 2 + 1];
        if (p0 < (uint32_t)numPolyPts) perPointEdgeCellIds[p0] = cid;
        if (p1 < (uint32_t)numPolyPts) perPointEdgeCellIds[p1] = cid;
      }
      id<MTLBuffer> perPointEdgeBuf = [device
        newBufferWithBytes:perPointEdgeCellIds.data()
                    length:perPointEdgeCellIds.size() * sizeof(uint32_t)
                   options:MTLResourceStorageModeShared];
      vtkMetalMRC::AssignConsumed(this->Internals->EdgeCellIdBuffer, perPointEdgeBuf);
      [edgeCellIdBuf release];

      this->Internals->EdgeIndexCount = numEdges * 2;
      this->Internals->EdgeVertexCount = numPolyPts;
      this->Internals->HasEdgeOverlay = true;
    }

    if (computeCmdBuf.status == MTLCommandBufferStatusCompleted && encWireframe)
    {
      vtkMetalMRC::AssignConsumed(this->Internals->LineIndexBuffer, wireOutBuf);

      const uint32_t* segCellIds = (const uint32_t*)[wireCellIdBuf contents];
      vtkMetalMRC::AssignConsumed(this->Internals->LineSegmentCellIdBuffer, wireCellIdBuf);

      const uint32_t* wireConn = (const uint32_t*)[this->Internals->LineIndexBuffer contents];
      std::vector<uint32_t> perPointLineCellIds(static_cast<size_t>(numPolyPts), 0);
      for (vtkIdType s = 0; s < numEdges; ++s)
      {
        uint32_t cid = segCellIds[s];
        uint32_t p0 = wireConn[static_cast<size_t>(s) * 2];
        uint32_t p1 = wireConn[static_cast<size_t>(s) * 2 + 1];
        if (p0 < (uint32_t)numPolyPts) perPointLineCellIds[p0] = cid;
        if (p1 < (uint32_t)numPolyPts) perPointLineCellIds[p1] = cid;
      }
      id<MTLBuffer> perPointBuf = [device
        newBufferWithBytes:perPointLineCellIds.data()
                    length:perPointLineCellIds.size() * sizeof(uint32_t)
                   options:MTLResourceStorageModeShared];
      vtkMetalMRC::AssignConsumed(this->Internals->LineCellIdBuffer, perPointBuf);

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

    else if (computeCmdBuf.status != MTLCommandBufferStatusCompleted)
    {
      // One or more dispatches failed — release outputs that were never consumed
      if (encPolygonTess) { /* Internals already owns TessOutputConnectivityBuffer etc */ }
      if (encEdgeVis) { [edgeOutBuf release]; [edgeCellIdBuf release]; }
      if (encWireframe) { [wireOutBuf release]; [wireCellIdBuf release]; }
    }

    // Release all temporary input buffers after GPU work is complete
    for (id<MTLBuffer> buf : tempInputBuffers)
    {
      [buf release];
    }

  }

  // Step 4: Build line connectivity for polyline → line segment conversion.
  // This block runs independently of useGPUTess to ensure polyline-to-line
  // extraction also works for small geometries or datasets without normals.
  {
    vtkCellArray* lines = polydata->GetLines();
    if (representation != VTK_POINTS &&
        !hasCellAssociatedExtraAttrs &&
        (!gpuTessUsed || (representation == VTK_SURFACE && !this->Internals->HasLines)))
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
            id<MTLLibrary> library = (__bridge id<MTLLibrary>)
              this->Internals->CachedRenderWindow->GetSharedShaderLibrary();
            if (library)
            {
              id<MTLFunction> func = [library newFunctionWithName:@"polyLineToLine"];
              if (func)
              {
                NSError* error = nil;
                this->Internals->PolyLineToLinePipeline =
                  [device newComputePipelineStateWithFunction:func error:&error];
                [func release];
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
            lParams.cellIdOffset = static_cast<uint32_t>(lineCellOffset);
            id<MTLBuffer> lParamsBuf = [device
              newBufferWithBytes:&lParams
                         length:sizeof(lParams)
                        options:MTLResourceStorageModeShared];

            id<MTLCommandQueue> lineQueue = this->Internals->EnsureComputeQueue(device);
            id<MTLCommandBuffer> lineCmdBuf = [lineQueue commandBuffer];
            lineCmdBuf.label = @"VTK Line Extraction";

            id<MTLComputeCommandEncoder> enc = [lineCmdBuf computeCommandEncoder];
            [enc setComputePipelineState:this->Internals->PolyLineToLinePipeline];
            [enc setBuffer:lineOutBuf offset:0 atIndex:0];
            [enc setBuffer:lineCellIdBuf offset:0 atIndex:1];
            [enc setBuffer:lConnBuf offset:0 atIndex:2];
            [enc setBuffer:lOffBuf offset:0 atIndex:3];
            [enc setBuffer:lPrimBuf offset:0 atIndex:4];
            [enc setBuffer:lParamsBuf offset:0 atIndex:5];
            NSUInteger tgMax = this->Internals->PolyLineToLinePipeline.maxTotalThreadsPerThreadgroup;
            MTLSize grid = MTLSizeMake(lParams.numCells, 1, 1);
            NSUInteger gW = static_cast<NSUInteger>(lParams.numCells);
            MTLSize tg = MTLSizeMake(std::min(tgMax, gW), 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];

            [lineCmdBuf commit];
            [lineCmdBuf waitUntilCompleted];

            if (lineCmdBuf.status == MTLCommandBufferStatusCompleted)
            {
              if (!this->Internals->HasLines)
              {
                vtkMetalMRC::AssignConsumed(this->Internals->LineIndexBuffer, lineOutBuf);

                const uint32_t* segCellIds = (const uint32_t*)[lineCellIdBuf contents];
                vtkMetalMRC::AssignConsumed(this->Internals->LineSegmentCellIdBuffer, lineCellIdBuf);

                const uint32_t* lineConn =
                  (const uint32_t*)[this->Internals->LineIndexBuffer contents];
                std::vector<uint32_t> perPointLineCellIds(
                  static_cast<size_t>(numPolyPts), 0);
                for (vtkIdType s = 0; s < numLineSegs; ++s)
                {
                  uint32_t cid = segCellIds[s];
                  uint32_t p0 = lineConn[static_cast<size_t>(s) * 2];
                  uint32_t p1 = lineConn[static_cast<size_t>(s) * 2 + 1];
                  if (p0 < (uint32_t)numPolyPts) perPointLineCellIds[p0] = cid;
                  if (p1 < (uint32_t)numPolyPts) perPointLineCellIds[p1] = cid;
                }
                id<MTLBuffer> perPointBuf = [device
                  newBufferWithBytes:perPointLineCellIds.data()
                              length:perPointLineCellIds.size() * sizeof(uint32_t)
                             options:MTLResourceStorageModeShared];
                vtkMetalMRC::AssignConsumed(this->Internals->LineCellIdBuffer, perPointBuf);

                this->Internals->LineIndexCount = numLineSegs * 2;
                this->Internals->HasLines = true;
                this->Internals->LinePrimitiveCount = numLineSegs;
                this->Internals->ThickLineSegmentCount = numLineSegs;
                this->Internals->RoundCapLineSegmentCount = numLineSegs;
                this->Internals->MiterJoinLineSegmentCount = numLineSegs;
                gpuTessUsed = true;
              }
              else
              {
                [lineOutBuf release];
                [lineCellIdBuf release];
              }
            }
            else
            {
              [lineOutBuf release];
              [lineCellIdBuf release];
            }

            [lConnBuf release];
            [lOffBuf release];
            [lPrimBuf release];
            [lParamsBuf release];
          }
        }
      }
    }
  }

  // Build per-point vertex data from GPU tessellation outputs.
  // Moved outside if(useGPUTess) so this block runs even when Step 4
  // (line extraction) is the one that set gpuTessUsed = true.
  if (gpuTessUsed)
  {
    positions.reserve(numPolyPts * 3);
    surfaceColors.reserve(numPolyPts * 4);
    triangleUVs.reserve(numPolyPts * 2);

    for (vtkIdType i = 0; i < numPolyPts; ++i)
    {
      double pt[3];
      polydata->GetPoint(i, pt);
      positions.push_back(static_cast<float>(pt[0]));
      positions.push_back(static_cast<float>(pt[1]));
      positions.push_back(static_cast<float>(pt[2]));

      if (normalArray)
      {
        normals.reserve(numPolyPts * 3);
        double n[3];
        normalArray->GetTuple(i, n);
        normals.push_back(static_cast<float>(n[0]));
        normals.push_back(static_cast<float>(n[1]));
        normals.push_back(static_cast<float>(n[2]));
      }

      if (mappedColors)
      {
        emitSurfaceColor(i, mappedColors->GetPointer(0));
      }
      else
      {
        float defRGBA[4];
        getOverrideOrDefaultRGBA(defRGBA);
        surfaceColors.push_back(defRGBA[0]);
        surfaceColors.push_back(defRGBA[1]);
        surfaceColors.push_back(defRGBA[2]);
        surfaceColors.push_back(defRGBA[3]);
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
      emitExtraAttrsForPoint(i);
    }
  }

  // Single-pass edges (GPU tessellation path): build the per-triangle-corner
  // corner-position records from the compute connectivity + point positions.
  if (gpuTessUsed && this->Internals->TessOutputConnectivityBuffer &&
      this->Internals->TrianglePrimitiveCount > 0 && !positions.empty())
  {
    const uint32_t* conn = static_cast<const uint32_t*>(
        [this->Internals->TessOutputConnectivityBuffer contents]);
    vtkIdType numT = this->Internals->TrianglePrimitiveCount;
    std::vector<float> triPos;
    triPos.reserve(static_cast<size_t>(numT) * 27);
    for (vtkIdType t = 0; t < numT; ++t)
    {
      uint32_t c[3] = { conn[t * 3 + 0], conn[t * 3 + 1], conn[t * 3 + 2] };
      for (int r = 0; r < 3; ++r)
      {
        for (int k = 0; k < 3; ++k)
        {
          triPos.push_back(positions[c[k] * 3 + 0]);
          triPos.push_back(positions[c[k] * 3 + 1]);
          triPos.push_back(positions[c[k] * 3 + 2]);
        }
      }
    }
    id<MTLBuffer> posBuf = [device
      newBufferWithBytes:triPos.data()
                 length:triPos.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->TrianglePosBuffer, posBuf);
  }

  vtkIdType polyCellIdx = polyCellOffset;
  // Helper for emitting per-vertex wireframe colors with batch override support.
  // Defined here so it is available throughout the CPU geometry build.
  auto emitWireframeColor = [&](vtkIdType pointId, vtkIdType cellId)
  {
    const unsigned char* rgba = mappedColors ? mappedColors->GetPointer(0) : nullptr;
    vtkIdType idx = (cellFlag == 0) ? pointId : cellId;
    if (rgba && !this->Internals->UseBatchColor)
    {
      float a = rgba[idx * 4 + 3] / 255.0f;
      if (this->Internals->UseBatchOpacity)
        a = static_cast<float>(this->Internals->BatchOpacity);
      surfaceColors.push_back(rgba[idx * 4] / 255.0f);
      surfaceColors.push_back(rgba[idx * 4 + 1] / 255.0f);
      surfaceColors.push_back(rgba[idx * 4 + 2] / 255.0f);
      surfaceColors.push_back(a);
    }
    else
    {
      float defRGBA[4];
      getOverrideOrDefaultRGBA(defRGBA);
      surfaceColors.push_back(defRGBA[0]);
      surfaceColors.push_back(defRGBA[1]);
      surfaceColors.push_back(defRGBA[2]);
      surfaceColors.push_back(defRGBA[3]);
    }
  };

  if (!gpuTessUsed && representation != VTK_POINTS)
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
            emitWireframeColor(v0, polyCellIdx);
            emitExtraAttrsForPoint(v0);
            emitExtraAttrsForCell(polyCellIdx);
            lineVertexCellIds.push_back(static_cast<uint32_t>(polyCellIdx) + 1u);
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
            emitWireframeColor(v1, polyCellIdx);
            emitExtraAttrsForPoint(v1);
            emitExtraAttrsForCell(polyCellIdx);
            lineVertexCellIds.push_back(static_cast<uint32_t>(polyCellIdx) + 1u);
          }
          else
          {
            idx1 = it1->second;
          }

          lineIndices.push_back(idx0);
          lineIndices.push_back(idx1);
          lineSegmentCellIds.push_back(static_cast<uint32_t>(polyCellIdx) + 1u);
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
        bool useIndexBuffer = (cellFlag == 0) && normalArray && !hasCellAssociatedExtraAttrs;

        for (vtkIdType i = 1; i < npts - 1; ++i)
        {
          vtkIdType tri[3] = { pts[0], pts[i], pts[i + 1] };
          uint32_t packed = packedEdgeFlags(tri, npts, pts);

          if (useIndexBuffer)
          {
            // Single-pass edges: per-corner records carry the 3 corner object
            // positions (identical for all 3 corners) so the vertex shader can
            // build the triangle's window-space edge equations.
            double triPt[3][3];
            for (int j = 0; j < 3; ++j) polydata->GetPoint(tri[j], triPt[j]);
            for (int r = 0; r < 3; ++r)
            {
              for (int j = 0; j < 3; ++j)
              {
                trianglePos.push_back(static_cast<float>(triPt[j][0]));
                trianglePos.push_back(static_cast<float>(triPt[j][1]));
                trianglePos.push_back(static_cast<float>(triPt[j][2]));
              }
            }
            // Indexed path: deduplicate vertices by point ID
            for (int j = 0; j < 3; ++j)
            {
              auto it = triVertexMap.find(tri[j]);
              if (it != triVertexMap.end())
              {
                triangleIndices.push_back(it->second);
                triangleBary.push_back(kBary[j][0]);
                triangleBary.push_back(kBary[j][1]);
                triangleBary.push_back(kBary[j][2]);
                triangleEdgeFlags.push_back(packed);
                // No cellId push, no emitExtraAttrsForPoint — vertex already exists
              }
              else
              {
                uint32_t vidx = static_cast<uint32_t>(positions.size() / 3);
                triVertexMap[tri[j]] = vidx;
                triangleIndices.push_back(vidx);
                triangleBary.push_back(kBary[j][0]);
                triangleBary.push_back(kBary[j][1]);
                triangleBary.push_back(kBary[j][2]);
                triangleEdgeFlags.push_back(packed);
                triangleVertexCellIds.push_back(static_cast<uint32_t>(polyCellIdx) + 1u);

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
                emitSurfaceColor(tri[j], mappedColors ? mappedColors->GetPointer(0) : nullptr);

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
                emitExtraAttrsForPoint(tri[j]);
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

            // Compute face normal once (used when normalArray is null)
            float faceNormal[3] = { 0.0f, 1.0f, 0.0f };
            if (!normalArray)
            {
              float e1[3] = { (float)(p[1][0] - p[0][0]), (float)(p[1][1] - p[0][1]), (float)(p[1][2] - p[0][2]) };
              float e2[3] = { (float)(p[2][0] - p[0][0]), (float)(p[2][1] - p[0][1]), (float)(p[2][2] - p[0][2]) };
              float ne1 = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
              float ne2 = std::sqrt(e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]);
              if (ne1 > 1e-8f && ne2 > 1e-8f)
              {
                e1[0] /= ne1; e1[1] /= ne1; e1[2] /= ne1;
                e2[0] /= ne2; e2[1] /= ne2; e2[2] /= ne2;
              }
              float fn[3] = { 0.0f, 1.0f, 0.0f };
              fn[0] = e1[1] * e2[2] - e1[2] * e2[1];
              fn[1] = e1[2] * e2[0] - e1[0] * e2[2];
              fn[2] = e1[0] * e2[1] - e1[1] * e2[0];
              float normalLength = std::sqrt(fn[0] * fn[0] + fn[1] * fn[1] + fn[2] * fn[2]);
              if (normalLength > 1e-8f) { fn[0] /= normalLength; fn[1] /= normalLength; fn[2] /= normalLength; }
              faceNormal[0] = fn[0]; faceNormal[1] = fn[1]; faceNormal[2] = fn[2];
            }

            for (int j = 0; j < 3; ++j)
            {
              triangleBary.push_back(kBary[j][0]);
              triangleBary.push_back(kBary[j][1]);
              triangleBary.push_back(kBary[j][2]);
              triangleEdgeFlags.push_back(packed);

              // Single-pass edges: per-corner records carry the 3 corner object
              // positions (identical for all 3 corners).
              for (int r = 0; r < 3; ++r)
              {
                trianglePos.push_back(static_cast<float>(p[r][0]));
                trianglePos.push_back(static_cast<float>(p[r][1]));
                trianglePos.push_back(static_cast<float>(p[r][2]));
              }

              positions.push_back(static_cast<float>(p[j][0]));
              positions.push_back(static_cast<float>(p[j][1]));
              positions.push_back(static_cast<float>(p[j][2]));

              // P2-2: Per-vertex normals (use each vertex's own normal when available)
              if (normalArray)
              {
                double nn[3];
                normalArray->GetTuple(tri[j], nn);
                normals.push_back(static_cast<float>(nn[0]));
                normals.push_back(static_cast<float>(nn[1]));
                normals.push_back(static_cast<float>(nn[2]));
              }
              else
              {
                normals.push_back(faceNormal[0]);
                normals.push_back(faceNormal[1]);
                normals.push_back(faceNormal[2]);
              }

              // P1-1A/1B: per-vertex color — point index for per-point coloring
              // (cellFlag == 0), cell index for per-cell coloring.
              emitSurfaceColor((cellFlag == 0) ? tri[j] : polyCellIdx,
                mappedColors ? mappedColors->GetPointer(0) : nullptr);

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
              triangleVertexCellIds.push_back(static_cast<uint32_t>(polyCellIdx) + 1u);
              emitExtraAttrsForPoint(tri[j]);
              emitExtraAttrsForCell(polyCellIdx);
            }
          }

        }
      }
      polyCellIdx++;
    }

    // Single-pass edges: non-indexed surfaces still route through the indirection
    // vertex entry (vertex_main_indexed reads bary/flags by vertex_id), so build
    // an identity index buffer covering all emitted triangle vertices.
    if (this->Internals->SurfaceUsesIndexedEntry && triangleIndices.empty() && !positions.empty())
    {
      uint32_t n = static_cast<uint32_t>(positions.size() / 3);
      triangleIndices.reserve(n);
      for (uint32_t k = 0; k < n; ++k)
      {
        triangleIndices.push_back(k);
      }
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

    // P2-2B: Edge overlay — extract unique polygon boundary edges for VTK_SURFACE.
    // Legacy chord-depth overlay only; the new path bakes edges into fragment_main.
    struct EdgeInfo
    {
      vtkIdType A = 0;
      vtkIdType B = 0;
      uint32_t CellId = 0;
    };
    if (this->UseLegacyEdgeOverlay && representation == VTK_SURFACE && edgeVisibility)
    {
      std::unordered_map<EdgeKey, EdgeInfo, EdgeKeyHash> uniqueEdges;

      if (polys)
      {
        vtkIdType edgeCellIdx = 0;
        const vtkIdType* pts = nullptr;
        vtkIdType npts = 0;
        polys->InitTraversal();
        while (polys->GetNextCell(npts, pts))
        {
          if (npts < 3)
          {
            ++edgeCellIdx;
            continue;
          }
          for (vtkIdType i = 0; i < npts; ++i)
          {
            vtkIdType a = pts[i];
            vtkIdType b = pts[(i + 1) % npts];
            EdgeKey key = MakeEdgeKey(a, b);
            if (uniqueEdges.find(key) == uniqueEdges.end())
            {
              EdgeInfo info;
              info.A = a;
              info.B = b;
              info.CellId = static_cast<uint32_t>(edgeCellIdx);
              uniqueEdges[key] = info;
            }
          }
          ++edgeCellIdx;
        }
      }

      if (!uniqueEdges.empty())
      {
        edgePositions.clear();
        edgeNormals.clear();
        edgeColors.clear();
        edgeUVs.clear();
        edgeIndices.clear();
        edgeVertexMap.clear();

        auto addEdgeVertex = [&](vtkIdType pointId, vtkIdType cellId = 0) -> uint32_t {
          auto it = edgeVertexMap.find(pointId);
          if (it != edgeVertexMap.end())
          {
            return it->second;
          }
          uint32_t idx = static_cast<uint32_t>(edgePositions.size() / 3);
          edgeVertexMap[pointId] = idx;

          double p[3];
          polydata->GetPoint(pointId, p);
          edgePositions.push_back(static_cast<float>(p[0]));
          edgePositions.push_back(static_cast<float>(p[1]));
          edgePositions.push_back(static_cast<float>(p[2]));

          if (normalArray)
          {
            double n[3];
            normalArray->GetTuple(pointId, n);
            edgeNormals.push_back(static_cast<float>(n[0]));
            edgeNormals.push_back(static_cast<float>(n[1]));
            edgeNormals.push_back(static_cast<float>(n[2]));
          }

          // Edges always render with the property's edge color (matching
          // vtkOpenGLPolyDataMapper, where emix fully replaces the diffuse with
          // the edge color on edge fragments regardless of scalar mapping).
          double* ec = actor->GetProperty()->GetEdgeColor();
          double eo = actor->GetProperty()->GetEdgeOpacity();
          edgeColors.push_back(static_cast<float>(ec[0]));
          edgeColors.push_back(static_cast<float>(ec[1]));
          edgeColors.push_back(static_cast<float>(ec[2]));
          edgeColors.push_back(static_cast<float>(eo));

          // P1-2: Per-vertex texture coordinates for edge overlay
          if (tcoordArray && tcoordArray->GetNumberOfTuples() > pointId)
          {
            double uv[3];
            tcoordArray->GetTuple(pointId, uv);
            edgeUVs.push_back(static_cast<float>(uv[0]));
            edgeUVs.push_back(static_cast<float>(uv[1]));
          }
          else
          {
            edgeUVs.push_back(0.0f);
            edgeUVs.push_back(0.0f);
          }

          edgeVertexCellIds.push_back(static_cast<uint32_t>(cellId + polyCellOffset) + 1u);
          return idx;
        };

        for (const auto& kv : uniqueEdges)
        {
          vtkIdType a = kv.second.A;
          vtkIdType b = kv.second.B;
          uint32_t edgeCellId = kv.second.CellId;
          edgeIndices.push_back(addEdgeVertex(a, edgeCellId));
          edgeIndices.push_back(addEdgeVertex(b, edgeCellId));
        }

        // Edge tubes: emit each polygon's boundary as a closed loop so
        // consecutive segments share a vertex and cell id, letting the miter
        // join shader connect them into continuous tubes (matching GL's
        // fake-tube edge rendering, where coverage extends to shared vertices).
        vtkIdType polyLoopIdx = 0;
        polys->InitTraversal();
        while (polys->GetNextCell(npts, pts))
        {
          if (npts < 3)
          {
            ++polyLoopIdx;
            continue;
          }
          uint32_t loopCellId = static_cast<uint32_t>(polyLoopIdx + polyCellOffset) + 1u;
          // Emit the closing segment once more at the start of the loop so the
          // wrap-around vertex (pts[0], shared by the p_{n-1}->p_0 and p_0->p_1
          // segments) has a same-cellId neighbor on both sides. The miter shader
          // gates joins on instance_id > 0 / < segmentCount-1, so without this
          // duplicate the seam at pts[0] is never joined and shows a notch. The
          // duplicate quad exactly overlaps the loop's own closing segment, so
          // it is a no-op for opaque edges.
          edgeTubeIndices.push_back(addEdgeVertex(pts[npts - 1], polyLoopIdx));
          edgeTubeIndices.push_back(addEdgeVertex(pts[0], polyLoopIdx));
          edgeTubeCellIds.push_back(loopCellId);
          for (vtkIdType i = 0; i < npts; ++i)
          {
            uint32_t va = addEdgeVertex(pts[i], polyLoopIdx);
            uint32_t vb = addEdgeVertex(pts[(i + 1) % npts], polyLoopIdx);
            edgeTubeIndices.push_back(va);
            edgeTubeIndices.push_back(vb);
            edgeTubeCellIds.push_back(loopCellId);
          }
          ++polyLoopIdx;
        }
      }
    }
  }

  // Process lines
  vtkCellArray* lines = polydata->GetLines();
  vtkIdType lineCellIdx = lineCellOffset;
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
        if (this->Internals->UseBatchOpacity)
        {
          ca = static_cast<float>(this->Internals->BatchOpacity);
        }

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
          emitExtraAttrsForPoint(pts[i]);
          emitExtraAttrsForCell(lineCellIdx);
          lineVertexCellIds.push_back(static_cast<uint32_t>(lineCellIdx) + 1u);
        }
        uint32_t base = nextPointId;
        nextPointId += npts;
        for (vtkIdType i = 0; i < npts - 1; ++i)
        {
          lineIndices.push_back(base + i);
          lineIndices.push_back(base + i + 1);
          lineSegmentCellIds.push_back(static_cast<uint32_t>(lineCellIdx) + 1u);
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
            vtkIdType colorIdx = pts[i];
            emitSurfaceColor(colorIdx, (mappedColors && cellFlag == 0) ? mappedColors->GetPointer(0) : nullptr);
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
            emitExtraAttrsForPoint(pts[i]);
            emitExtraAttrsForCell(lineCellIdx);
            lineVertexCellIds.push_back(static_cast<uint32_t>(lineCellIdx) + 1u);
          }
        }
        for (vtkIdType i = 0; i < npts - 1; ++i)
        {
          lineIndices.push_back(pointMap[pts[i]]);
          lineIndices.push_back(pointMap[pts[i + 1]]);
          lineSegmentCellIds.push_back(static_cast<uint32_t>(lineCellIdx) + 1u);
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

  // Synchronize line state for wireframe polygon edges
  // that were added outside the explicit-lines block.
  if (!lineIndices.empty() && !this->Internals->HasLines)
  {
    this->Internals->LineIndexCount = lineIndices.size();
    this->Internals->HasLines = true;
    this->Internals->LinePrimitiveCount = lineSegmentCellIds.size();
    this->Internals->ThickLineSegmentCount = lineIndices.size() / 2;
    this->Internals->RoundCapLineSegmentCount = this->Internals->ThickLineSegmentCount;
    this->Internals->MiterJoinLineSegmentCount = this->Internals->ThickLineSegmentCount;
  }
  } // end if (!gpuTessUsed)

  float defaultRGBA[4];
  getOverrideOrDefaultRGBA(defaultRGBA);

  this->UploadVertexDataToMTLBuffers(mtlDevice, polydata, pd,
    mappedColors, cellFlag, gpuTessUsed, defaultRGBA,
    positions, normals, surfaceColors, triangleUVs, lineIndices,
    triangleIndices, triangleBary, triangleEdgeFlags, trianglePos,
    edgePositions, edgeNormals, edgeColors, edgeUVs,
    edgeIndices, triangleVertexCellIds, lineVertexCellIds,
    lineSegmentCellIds, edgeVertexCellIds, edgeTubeIndices,
    edgeTubeCellIds, extraAttrArrays);
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::UploadVertexDataToMTLBuffers(void* mtlDevice,
  vtkPolyData* polydata,
  vtkPointData* pd,
  vtkUnsignedCharArray* mappedColors,
  int cellFlag,
  bool gpuTessUsed,
  const float defaultRGBA[4],
  std::vector<float>& positions, std::vector<float>& normals,
  const std::vector<float>& surfaceColors, const std::vector<float>& triangleUVs,
  const std::vector<uint32_t>& lineIndices,
  const std::vector<uint32_t>& triangleIndices,
  const std::vector<float>& triangleBary,
  const std::vector<uint32_t>& triangleEdgeFlags,
  const std::vector<float>& trianglePos,
  const std::vector<float>& edgePositions,
  std::vector<float>& edgeNormals, const std::vector<float>& edgeColors,
  const std::vector<float>& edgeUVs, const std::vector<uint32_t>& edgeIndices,
  const std::vector<uint32_t>& triangleVertexCellIds,
  const std::vector<uint32_t>& lineVertexCellIds,
  const std::vector<uint32_t>& lineSegmentCellIds,
  const std::vector<uint32_t>& edgeVertexCellIds,
  const std::vector<uint32_t>& edgeTubeIndices,
  const std::vector<uint32_t>& edgeTubeCellIds,
  std::unordered_map<std::string, std::vector<float>>& extraAttrArrays)
{
  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;

  if (!positions.empty())
  {
    id<MTLBuffer> posBuffer = [device
      newBufferWithBytes:positions.data()
                 length:positions.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->VertexPositionBuffer, posBuffer);
  }
  if (!normals.empty())
  {
    id<MTLBuffer> normBuffer = [device
      newBufferWithBytes:normals.data()
                 length:normals.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->VertexNormalBuffer, normBuffer);
  }
  // Vertex descriptor always requires a buffer at index 1 for normals.
  else if (!positions.empty())
  {
    normals.assign(positions.size(), 0.0f);
    for (size_t i = 1; i < normals.size(); i += 3)
    {
      normals[i] = 1.0f;
    }
    id<MTLBuffer> normBuf2 = [device
      newBufferWithBytes:normals.data()
                 length:normals.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->VertexNormalBuffer, normBuf2);
  }
  if (!lineIndices.empty())
  {
    id<MTLBuffer> lineIdxBuf = [device
      newBufferWithBytes:lineIndices.data()
                 length:lineIndices.size() * sizeof(uint32_t)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->LineIndexBuffer, lineIdxBuf);
  }

  // P2-2C: Create triangle index buffer for deduplicated geometry
  if (!triangleIndices.empty())
  {
    id<MTLBuffer> triIdxBuf = [device
      newBufferWithBytes:triangleIndices.data()
                 length:triangleIndices.size() * sizeof(uint32_t)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->IndexBuffer, triIdxBuf);
  }

  // Single-pass edges: per-triangle-corner bary + boundary flags (CPU paths)
  if (!triangleBary.empty() && !triangleEdgeFlags.empty())
  {
    id<MTLBuffer> baryBuf = [device
      newBufferWithBytes:triangleBary.data()
                 length:triangleBary.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->TriangleBaryBuffer, baryBuf);

    id<MTLBuffer> flagBuf = [device
      newBufferWithBytes:triangleEdgeFlags.data()
                 length:triangleEdgeFlags.size() * sizeof(uint32_t)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->TriangleEdgeFlagBuffer, flagBuf);

    if (!trianglePos.empty())
    {
      id<MTLBuffer> posBuf = [device
        newBufferWithBytes:trianglePos.data()
                   length:trianglePos.size() * sizeof(float)
                  options:MTLResourceStorageModeShared];
      vtkMetalMRC::AssignConsumed(this->Internals->TrianglePosBuffer, posBuf);
    }
  }

  // P2-2B: Create edge geometry buffers for wireframe overlay on surfaces
  if (!edgeIndices.empty() && !edgePositions.empty())
  {
    if (edgeNormals.empty())
    {
      edgeNormals.assign(edgePositions.size(), 0.0f);
      for (size_t i = 1; i < edgeNormals.size(); i += 3)
      {
        edgeNormals[i] = 1.0f;
      }
    }

    id<MTLBuffer> posBuffer =
      [device newBufferWithBytes:edgePositions.data()
                          length:edgePositions.size() * sizeof(float)
                         options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->EdgeVertexPositionBuffer, posBuffer);

    id<MTLBuffer> normBuffer =
      [device newBufferWithBytes:edgeNormals.data()
                          length:edgeNormals.size() * sizeof(float)
                         options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->EdgeVertexNormalBuffer, normBuffer);

    if (!edgeColors.empty())
    {
      id<MTLBuffer> colorBuffer =
        [device newBufferWithBytes:edgeColors.data()
                            length:edgeColors.size() * sizeof(float)
                           options:MTLResourceStorageModeShared];
      vtkMetalMRC::AssignConsumed(this->Internals->EdgeSurfaceColorBuffer, colorBuffer);
    }

    // P1-2: Upload edge overlay UV coordinates (float2 per vertex)
    if (!edgeUVs.empty())
    {
      id<MTLBuffer> uvBuffer =
        [device newBufferWithBytes:edgeUVs.data()
                            length:edgeUVs.size() * sizeof(float)
                           options:MTLResourceStorageModeShared];
      vtkMetalMRC::AssignConsumed(this->Internals->EdgeUVBuffer, uvBuffer);
    }

    id<MTLBuffer> idxBuffer =
      [device newBufferWithBytes:edgeIndices.data()
                          length:edgeIndices.size() * sizeof(uint32_t)
                         options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->EdgeIndexBuffer, idxBuffer);

    this->Internals->EdgeIndexCount = edgeIndices.size();
    this->Internals->EdgeVertexCount = edgePositions.size() / 3;
    this->Internals->HasEdgeOverlay = true;

    if (!edgeVertexCellIds.empty())
    {
      id<MTLBuffer> cellIdBuf =
        [device newBufferWithBytes:edgeVertexCellIds.data()
                            length:edgeVertexCellIds.size() * sizeof(uint32_t)
                           options:MTLResourceStorageModeShared];
      vtkMetalMRC::AssignConsumed(this->Internals->EdgeCellIdBuffer, cellIdBuf);
    }

    // Edge tube loop segments + cell ids (mitered thick-edge overlay)
    if (!edgeTubeIndices.empty())
    {
      id<MTLBuffer> tubeIdxBuf =
        [device newBufferWithBytes:edgeTubeIndices.data()
                            length:edgeTubeIndices.size() * sizeof(uint32_t)
                           options:MTLResourceStorageModeShared];
      vtkMetalMRC::AssignConsumed(this->Internals->EdgeTubeIndexBuffer, tubeIdxBuf);

      if (!edgeTubeCellIds.empty())
      {
        id<MTLBuffer> tubeCellBuf =
          [device newBufferWithBytes:edgeTubeCellIds.data()
                              length:edgeTubeCellIds.size() * sizeof(uint32_t)
                             options:MTLResourceStorageModeShared];
        vtkMetalMRC::AssignConsumed(this->Internals->EdgeTubeCellIdBuffer, tubeCellBuf);
      }

      uint32_t tubeCount = static_cast<uint32_t>(edgeTubeIndices.size() / 2);
      id<MTLBuffer> tubeCountBuf =
        [device newBufferWithLength:sizeof(uint32_t)
                            options:MTLResourceStorageModeShared];
      memcpy([tubeCountBuf contents], &tubeCount, sizeof(tubeCount));
      vtkMetalMRC::AssignConsumed(this->Internals->EdgeTubeSegmentCountBuffer, tubeCountBuf);
      this->Internals->EdgeTubeSegmentCount = edgeTubeIndices.size() / 2;
    }
  }

  // P1-1A/1B: Create surface color buffer (float4 per triangle/line vertex)
  if (!surfaceColors.empty())
  {
    id<MTLBuffer> surfColorBuf = [device
      newBufferWithBytes:surfaceColors.data()
                 length:surfaceColors.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->SurfaceColorBuffer, surfColorBuf);
    this->Internals->HasSurfaceColors =
        (mappedColors != nullptr) || this->Internals->UseBatchColor;

    // P2-4: Detect whether any surface color has non-opaque alpha
    bool hasNonOpaqueAlpha = false;
    for (size_t ac = 3; ac < surfaceColors.size() && !hasNonOpaqueAlpha; ac += 4)
    {
      if (surfaceColors[ac] < 0.999f)
      {
        hasNonOpaqueAlpha = true;
      }
    }
    this->Internals->HasSurfaceAlpha =
        this->Internals->HasSurfaceColors ||
        this->Internals->UseBatchOpacity ||
        hasNonOpaqueAlpha;
  }
  // Always provide a white color buffer so the vertex shader can always read from buffer(3)
  else if (!positions.empty())
  {
    std::vector<float> whiteColors(positions.size() / 3 * 4, 1.0f);
    id<MTLBuffer> whiteColorBuf = [device
      newBufferWithBytes:whiteColors.data()
                 length:whiteColors.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->SurfaceColorBuffer, whiteColorBuf);
  }

  // P5-5A: Create triangle UV buffer (float2 per triangle/line vertex)
  if (!triangleUVs.empty())
  {
    id<MTLBuffer> uvBuf = [device
      newBufferWithBytes:triangleUVs.data()
                 length:triangleUVs.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->TriangleUVBuffer, uvBuf);
  }
  else if (!positions.empty())
  {
    // Always provide a zero UV buffer so the vertex shader can always read from buffer(8)
    std::vector<float> zeroUVs(positions.size() / 3 * 2, 0.0f);
    id<MTLBuffer> zeroUVBuf = [device
      newBufferWithBytes:zeroUVs.data()
                 length:zeroUVs.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->TriangleUVBuffer, zeroUVBuf);
  }

  // 8D validation: every active extra attribute must have exactly one value
  // tuple per rendered vertex, matching positions.size() / 3. Under the current
  // emission paths (point and cell lambdas gated by field association, cell
  // attrs forced onto the CPU non-indexed path) this holds by construction; a
  // mismatch here signals an emission bug, so log it loudly instead of silently
  // uploading a short buffer.
  {
    const vtkIdType vertexCount = static_cast<vtkIdType>(positions.size() / 3);
    for (const auto& attr : this->ExtraAttributes)
    {
      int comps = 1;
      vtkDataArray* da =
        (attr.second.FieldAssociation == vtkDataObject::FIELD_ASSOCIATION_CELLS)
          ? polydata->GetCellData()->GetArray(attr.second.DataArrayName.c_str())
          : polydata->GetPointData()->GetArray(attr.second.DataArrayName.c_str());
      if (da)
      {
        comps = (attr.second.ComponentNumber < 0) ? da->GetNumberOfComponents() : 1;
      }
      const size_t expected = static_cast<size_t>(vertexCount) * comps;
      const size_t actual = extraAttrArrays[attr.first].size();
      if (actual != expected)
      {
        vtkErrorMacro(<< "Extra attribute '" << attr.first
                      << "' size mismatch: expected " << expected
                      << " (" << vertexCount << " verts x " << comps << " comps)"
                      << ", got " << actual << ".");
      }
    }
  }

  // 8D: Create extra attribute buffers from per-vertex arrays built during emission.
  // Each attribute has exactly one value per rendered vertex, matching positions.size() / 3.
  for (const auto& attr : this->ExtraAttributes)
  {
    const auto& attrData = extraAttrArrays[attr.first];
    if (attrData.empty())
    {
      continue;
    }
    id<MTLBuffer> attrBuf = [device
      newBufferWithBytes:attrData.data()
                 length:attrData.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->ExtraAttributeBuffers[attr.first], attrBuf);

    int numComps = 1;
    vtkDataArray* da = nullptr;
    if (attr.second.FieldAssociation == vtkDataObject::FIELD_ASSOCIATION_POINTS)
    {
      da = polydata->GetPointData()->GetArray(attr.second.DataArrayName.c_str());
    }
    else if (attr.second.FieldAssociation == vtkDataObject::FIELD_ASSOCIATION_CELLS)
    {
      da = polydata->GetCellData()->GetArray(attr.second.DataArrayName.c_str());
    }
    if (da)
    {
      numComps = (attr.second.ComponentNumber < 0) ? da->GetNumberOfComponents() : 1;
    }
    this->Internals->ExtraAttributeComponentCounts[attr.first] = numComps;
  }

  // P6-6A: When GPU tessellation produced edge overlay, assign edge vertex buffers
  // to point to the same per-point vertex buffers as triangles (shared vertex data).
  if (gpuTessUsed && this->Internals->HasEdgeOverlay)
  {
    vtkMetalMRC::AssignRetained(
        this->Internals->EdgeVertexPositionBuffer,
        this->Internals->VertexPositionBuffer);

    vtkMetalMRC::AssignRetained(
        this->Internals->EdgeVertexNormalBuffer,
        this->Internals->VertexNormalBuffer);

    vtkMetalMRC::AssignRetained(
        this->Internals->EdgeSurfaceColorBuffer,
        this->Internals->SurfaceColorBuffer);

    if (this->Internals->TriangleUVBuffer)
    {
      vtkMetalMRC::AssignRetained(
          this->Internals->EdgeUVBuffer,
          this->Internals->TriangleUVBuffer);
    }
  }

  // P2-8: Create per-vertex cell ID buffer for triangles
  if (!triangleVertexCellIds.empty())
  {
    id<MTLBuffer> cellIdBuf =
      [device newBufferWithBytes:triangleVertexCellIds.data()
                          length:triangleVertexCellIds.size() * sizeof(uint32_t)
                         options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->TriangleCellIdBuffer, cellIdBuf);

    this->Internals->TrianglePrimitiveCount = triangleVertexCellIds.size() / 3;
  }

  // P2-8: Create per-vertex cell ID buffer for lines
  if (!lineVertexCellIds.empty())
  {
    id<MTLBuffer> cellIdBuf =
      [device newBufferWithBytes:lineVertexCellIds.data()
                          length:lineVertexCellIds.size() * sizeof(uint32_t)
                         options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->LineCellIdBuffer, cellIdBuf);

    this->Internals->LinePrimitiveCount = lineSegmentCellIds.size();
  }

  // P2-8: Create per-segment cell ID buffer for thick/round/miter lines
  if (!lineSegmentCellIds.empty())
  {
    id<MTLBuffer> segCellIdBuf =
      [device newBufferWithBytes:lineSegmentCellIds.data()
                          length:lineSegmentCellIds.size() * sizeof(uint32_t)
                         options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->LineSegmentCellIdBuffer, segCellIdBuf);
  }

  // Prop ID buffer — PickIds {propId, compositeIndex}
  {
    PickIds ids = { 0, 0 };
    id<MTLBuffer> propIdBuf = [device
      newBufferWithBytes:&ids
                 length:sizeof(PickIds)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->PropIdBuffer, propIdBuf);
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
    id<MTLBuffer> ptPosBuf = [device
      newBufferWithBytes:pointPositions.data()
                 length:pointPositions.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->PointPositionBuffer, ptPosBuf);

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
    id<MTLBuffer> ptNormBuf = [device
      newBufferWithBytes:pointNormals.data()
                 length:pointNormals.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->PointNormalBuffer, ptNormBuf);

    // Point colors — from MapScalars (per-point RGBA), batch override, or default white.
    // Matches WebGPU: reads point_colors SSBO indexed by point_id.
    // Note: mappedColors and cellFlag are already set from the early MapScalars call above.
    std::vector<float> pointColors(numPts * 4, 1.0f);
    if (this->Internals->UseBatchColor || this->Internals->UseBatchOpacity)
    {
      for (vtkIdType i = 0; i < numPts; ++i)
      {
        pointColors[i * 4] = defaultRGBA[0];
        pointColors[i * 4 + 1] = defaultRGBA[1];
        pointColors[i * 4 + 2] = defaultRGBA[2];
        pointColors[i * 4 + 3] = defaultRGBA[3];
      }
    }
    else if (mappedColors && cellFlag == 0 &&
        mappedColors->GetNumberOfTuples() >= numPts)
    {
      const unsigned char* rgba = mappedColors->GetPointer(0);
      // Per-point colors — normalize unsigned char RGBA to float [0,1]
      for (vtkIdType i = 0; i < numPts; ++i)
      {
        pointColors[i * 4] = rgba[i * 4] / 255.0f;
        pointColors[i * 4 + 1] = rgba[i * 4 + 1] / 255.0f;
        pointColors[i * 4 + 2] = rgba[i * 4 + 2] / 255.0f;
        pointColors[i * 4 + 3] = rgba[i * 4 + 3] / 255.0f;
      }
    }
    id<MTLBuffer> ptColorBuf = [device
      newBufferWithBytes:pointColors.data()
                 length:pointColors.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->PointColorBuffer, ptColorBuf);

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
    id<MTLBuffer> ptTangBuf = [device
      newBufferWithBytes:pointTangents.data()
                 length:pointTangents.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->PointTangentBuffer, ptTangBuf);

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
    id<MTLBuffer> ptUVBuf = [device
      newBufferWithBytes:pointUVs.data()
                 length:pointUVs.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->PointUVBuffer, ptUVBuf);

    // Point color UVs — from polydata if available, otherwise default (0,0).
    // Matches WebGPU: reads point_color_uvs SSBO indexed by point_id.
    std::vector<float> pointColorUVs(numPts * 2, 0.0f);
    // Color UVs are typically the same as regular UVs unless a separate texture channel is used.
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
    id<MTLBuffer> ptColUVBuf = [device
      newBufferWithBytes:pointColorUVs.data()
                 length:pointColorUVs.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->PointColorUVBuffer, ptColUVBuf);

    // Connectivity: identity map — vertex_index i maps to point i.
    std::vector<uint32_t> connectivity(numPts);
    for (vtkIdType i = 0; i < numPts; ++i)
    {
      connectivity[i] = static_cast<uint32_t>(i);
    }
    id<MTLBuffer> ptConnBuf = [device
      newBufferWithBytes:connectivity.data()
                 length:connectivity.size() * sizeof(uint32_t)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->PointConnectivityBuffer, ptConnBuf);

    this->Internals->PointVertexCount = numPts;

    // P2-8: Point cell IDs — identity mapping (point i → cell i)
    std::vector<uint32_t> pointCellIds(numPts);
    for (vtkIdType i = 0; i < numPts; ++i)
    {
      pointCellIds[i] = static_cast<uint32_t>(i) + 1u;
    }
    id<MTLBuffer> ptCellIdBuf = [device
      newBufferWithBytes:pointCellIds.data()
                 length:pointCellIds.size() * sizeof(uint32_t)
                options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->PointCellIdBuffer, ptCellIdBuf);
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

  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;

  if (!this->Internals->CachedRenderWindow)
  {
    vtkErrorMacro(<< "No render window available for shader library access");
    return;
  }

  id<MTLLibrary> library = (__bridge id<MTLLibrary>)
    this->Internals->CachedRenderWindow->GetSharedShaderLibrary();
  if (!library)
  {
    vtkErrorMacro(<< "No shared shader library available");
    return;
  }

  id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_main"];
  id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragment_main"];

  // Single-pass surface edges: when the indexed entry is active the triangle
  // pipeline reads deduplicated arrays through the index buffer (no stage_in).
  const bool indexedEntry = this->Internals->SurfaceUsesIndexedEntry;
  id<MTLFunction> triVertexFunc = indexedEntry
    ? [library newFunctionWithName:@"vertex_main_indexed"]
    : vertexFunc;

  if (!vertexFunc || !fragmentFunc || !triVertexFunc)
  {
    vtkErrorMacro(<< "Failed to find shader functions");
    [vertexFunc release];
    [fragmentFunc release];
    if (triVertexFunc != vertexFunc)
    {
      [triVertexFunc release];
    }
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
  pipelineDesc.vertexFunction = triVertexFunc;
  pipelineDesc.fragmentFunction = fragmentFunc;
  if (!indexedEntry)
  {
    pipelineDesc.vertexDescriptor = vertexDesc;
  }
  pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
  // 8A: Skip IDs attachment when MSAA is active — render pass only has 1 color attachment
  if (sampleCount <= 1)
  {
    pipelineDesc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;  // P2-8: picking IDs
  }

  // Enable alpha blending for transparency support
  pipelineDesc.colorAttachments[0].blendingEnabled = YES;
  pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
  pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
  pipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
  pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
  pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
  pipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;

  // Enable depth testing (matching WebGPU's depthCompare = Less, depthWriteEnabled = true)
  pipelineDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

  // Enable backface culling (matching WebGPU's default behavior)
  pipelineDesc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;

  // 8A: Set sample count for MSAA
  pipelineDesc.rasterSampleCount = sampleCount;

  if (!this->Internals->TrianglePipeline)
  {
    NSError* error = nil;
    this->Internals->TrianglePipeline =
      [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    if (!this->Internals->TrianglePipeline)
    {
      vtkErrorMacro(<< "Triangle pipeline: " << [[error localizedDescription] UTF8String]);
    }
  }

  // The line pipeline always keeps vertex_main + the attribute descriptor.
  pipelineDesc.vertexFunction = vertexFunc;
  pipelineDesc.vertexDescriptor = vertexDesc;
  pipelineDesc.inputPrimitiveTopology = MTLPrimitiveTopologyClassLine;

  if (!this->Internals->LinePipeline)
  {
    NSError* error = nil;
    this->Internals->LinePipeline =
      [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    if (!this->Internals->LinePipeline)
    {
      vtkErrorMacro(<< "Line pipeline: " << [[error localizedDescription] UTF8String]);
    }
  }

  [vertexFunc release];
  [fragmentFunc release];
  if (triVertexFunc != vertexFunc)
  {
    [triVertexFunc release];
  }
  [vertexDesc release];
  [pipelineDesc release];
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

  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;

  if (!this->Internals->CachedRenderWindow)
  {
    vtkErrorMacro(<< "No render window available for shader library access");
    return;
  }

  id<MTLLibrary> library = (__bridge id<MTLLibrary>)
    this->Internals->CachedRenderWindow->GetSharedShaderLibrary();
  if (!library)
  {
    vtkErrorMacro(<< "No shared shader library available for points");
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
      desc.colorAttachments[0].blendingEnabled = YES;
      desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
      desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
      desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
      desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
      if (sampleCount <= 1)
      {
        desc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;  // P2-8: picking IDs
      }
      desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
      desc.rasterSampleCount = sampleCount;

      NSError* error = nil;
      this->Internals->PointPipeline =
        [device newRenderPipelineStateWithDescriptor:desc error:&error];
      if (!this->Internals->PointPipeline)
      {
        vtkErrorMacro(<< "Point pipeline: " << [[error localizedDescription] UTF8String]);
      }
      [desc release];
    }
    [vFunc release];
    [fFunc release];
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
      desc.colorAttachments[0].blendingEnabled = YES;
      desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
      desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
      desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
      desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
      if (sampleCount <= 1)
      {
        desc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;  // P2-8: picking IDs
      }
      desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
      // No backface culling for point quads
      desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
      desc.rasterSampleCount = sampleCount;

      NSError* error = nil;
      this->Internals->PointShapedPipeline =
        [device newRenderPipelineStateWithDescriptor:desc error:&error];
      if (!this->Internals->PointShapedPipeline)
      {
        vtkErrorMacro(<< "Point shaped pipeline: " << [[error localizedDescription] UTF8String]);
      }
      [desc release];
    }
    [vFunc release];
    [fFunc release];
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

  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;

  if (!this->Internals->CachedRenderWindow)
  {
    vtkErrorMacro(<< "No render window available for shader library access");
    return;
  }

  id<MTLLibrary> library = (__bridge id<MTLLibrary>)
    this->Internals->CachedRenderWindow->GetSharedShaderLibrary();
  if (!library)
  {
    vtkErrorMacro(<< "No shared shader library available for edges");
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
    desc.colorAttachments[0].blendingEnabled = YES;
    desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    if (sampleCount <= 1)
    {
      desc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;  // P2-8: picking IDs
    }
    desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassLine;
    desc.rasterSampleCount = sampleCount;

    NSError* error = nil;
    this->Internals->EdgePipeline =
      [device newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!this->Internals->EdgePipeline)
    {
      vtkErrorMacro(<< "Edge pipeline: " << [[error localizedDescription] UTF8String]);
    }
    [desc release];
    [vertexDesc release];
  }
  [vFunc release];
  [fFunc release];
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

  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;

  if (!this->Internals->CachedRenderWindow)
  {
    vtkErrorMacro(<< "No render window available for shader library access");
    return;
  }

  id<MTLLibrary> library = (__bridge id<MTLLibrary>)
    this->Internals->CachedRenderWindow->GetSharedShaderLibrary();
  if (!library)
  {
    vtkErrorMacro(<< "No shared shader library available for thick lines");
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
    desc.colorAttachments[0].blendingEnabled = YES;
    desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    if (sampleCount <= 1)
    {
      desc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;  // P2-8: picking IDs
    }
    desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    // Thick lines are rendered as triangle strips (quads)
    desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    desc.rasterSampleCount = sampleCount;

    NSError* error = nil;
    this->Internals->ThickLinePipeline =
      [device newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!this->Internals->ThickLinePipeline)
    {
      vtkErrorMacro(<< "Thick line pipeline: " << [[error localizedDescription] UTF8String]);
    }
    [desc release];
  }
  [vFunc release];
  [fFunc release];
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

  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;

  if (!this->Internals->CachedRenderWindow)
  {
    vtkErrorMacro(<< "No render window available for shader library access");
    return;
  }

  id<MTLLibrary> library = (__bridge id<MTLLibrary>)
    this->Internals->CachedRenderWindow->GetSharedShaderLibrary();
  if (!library)
  {
    vtkErrorMacro(<< "No shared shader library available for round cap lines");
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
    desc.colorAttachments[0].blendingEnabled = YES;
    desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    if (sampleCount <= 1)
    {
      desc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;  // P2-8: picking IDs
    }
    desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    // Round cap lines are rendered as triangle lists
    desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    desc.rasterSampleCount = sampleCount;

    NSError* error = nil;
    this->Internals->RoundCapLinePipeline =
      [device newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!this->Internals->RoundCapLinePipeline)
    {
      vtkErrorMacro(<< "Round cap line pipeline: " << [[error localizedDescription] UTF8String]);
    }
    [desc release];
  }
  [vFunc release];
  [fFunc release];
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

  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;

  if (!this->Internals->CachedRenderWindow)
  {
    vtkErrorMacro(<< "No render window available for shader library access");
    return;
  }

  id<MTLLibrary> library = (__bridge id<MTLLibrary>)
    this->Internals->CachedRenderWindow->GetSharedShaderLibrary();
  if (!library)
  {
    vtkErrorMacro(<< "No shared shader library available for miter join lines");
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
    desc.colorAttachments[0].blendingEnabled = YES;
    desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    if (sampleCount <= 1)
    {
      desc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;  // P2-8: picking IDs
    }
    desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    // Miter join lines are rendered as triangle strips (quads)
    desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    desc.rasterSampleCount = sampleCount;

    NSError* error = nil;
    this->Internals->MiterJoinLinePipeline =
      [device newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!this->Internals->MiterJoinLinePipeline)
    {
      vtkErrorMacro(<< "Miter join line pipeline: " << [[error localizedDescription] UTF8String]);
    }
    [desc release];
  }
  [vFunc release];
  [fFunc release];
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

  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;

  if (!this->Internals->CachedRenderWindow)
  {
    vtkErrorMacro(<< "No render window available for shader library access");
    return;
  }

  id<MTLLibrary> library = (__bridge id<MTLLibrary>)
    this->Internals->CachedRenderWindow->GetSharedShaderLibrary();
  if (!library)
  {
    vtkErrorMacro(<< "No shared shader library available for depth peeling");
    return;
  }

  const bool indexedEntry = this->Internals->SurfaceUsesIndexedEntry;
  id<MTLFunction> vertexFunc = indexedEntry
    ? [library newFunctionWithName:@"vertex_main_indexed"]
    : [library newFunctionWithName:@"vertex_main"];
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
      if (!indexedEntry)
      {
        desc.vertexDescriptor = vertexDesc;
      }
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
      desc.rasterSampleCount = this->Internals->CachedSampleCount > 0
        ? this->Internals->CachedSampleCount : 1;

      NSError* error = nil;
      this->Internals->TriangleInitPeelPipeline =
        [device newRenderPipelineStateWithDescriptor:desc error:&error];
      if (!this->Internals->TriangleInitPeelPipeline)
      {
        vtkErrorMacro(<< "Init peel pipeline: " << [[error localizedDescription] UTF8String]);
      }
      [desc release];
    }
    [fragFunc release];
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
      if (!indexedEntry)
      {
        desc.vertexDescriptor = vertexDesc;
      }
      // color(0): RGBA8 (backTemp) — MAX blend, matching the GL reference
      // (vtkDualDepthPeelingPass uses glBlendEquation(GL_MAX) for all three
      // peel targets). Non-back fragments write backTemp = 0, and MAX blending
      // preserves previously peeled back fragments instead of overwriting them.
      desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
      desc.colorAttachments[0].blendingEnabled = YES;
      desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationMax;
      desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationMax;
      desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
      desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
      desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
      desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
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
      desc.rasterSampleCount = this->Internals->CachedSampleCount > 0
        ? this->Internals->CachedSampleCount : 1;

      NSError* error = nil;
      this->Internals->TrianglePeelPipeline =
        [device newRenderPipelineStateWithDescriptor:desc error:&error];
      if (!this->Internals->TrianglePeelPipeline)
      {
        vtkErrorMacro(<< "Peel pipeline: " << [[error localizedDescription] UTF8String]);
      }
      [desc release];
    }
    [fragFunc release];
  }

  [vertexFunc release];
  [vertexDesc release];
}

//------------------------------------------------------------------------------
// 8C: Create the order-independent transparency accumulate pipeline state for
// triangle rendering. Uses the same vertex shader (vertex_main) and a fragment
// shader (fragment_main_oit) that outputs premultiplied color to color(0)
// (RGBA16F) and revealage to color(1) (R16F).
//
// The blend configuration mirrors vtkOrderIndependentTranslucentPass in GL:
//   glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ZERO, GL_ONE_MINUS_SRC_ALPHA)
// applied to the shader output (C*a, a):
//   color(0).rgb = C*a + dst.rgb                    (ONE, ONE)
//   color(0).a   = dst.a * (1 - a)                  (ZERO, ONE_MINUS_SRC_ALPHA)
//   color(1).r   = a + dst.r                        (ONE, ONE)
//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::EnsureOITPipelineStates(void* mtlDevice)
{
  if (this->Internals->TriangleOITPipeline)
  {
    return;
  }

  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;

  if (!this->Internals->CachedRenderWindow)
  {
    vtkErrorMacro(<< "No render window available for shader library access");
    return;
  }

  id<MTLLibrary> library = (__bridge id<MTLLibrary>)
    this->Internals->CachedRenderWindow->GetSharedShaderLibrary();
  if (!library)
  {
    vtkErrorMacro(<< "No shared shader library available for OIT");
    return;
  }

  const bool indexedEntry = this->Internals->SurfaceUsesIndexedEntry;
  id<MTLFunction> vertexFunc = indexedEntry
    ? [library newFunctionWithName:@"vertex_main_indexed"]
    : [library newFunctionWithName:@"vertex_main"];
  id<MTLFunction> fragFunc = [library newFunctionWithName:@"fragment_main_oit"];
  if (!vertexFunc || !fragFunc)
  {
    vtkErrorMacro(<< "Failed to find OIT shader functions");
    [vertexFunc release];
    [fragFunc release];
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

  MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
  desc.vertexFunction = vertexFunc;
  desc.fragmentFunction = fragFunc;
  if (!indexedEntry)
  {
    desc.vertexDescriptor = vertexDesc;
  }
  // color(0): RGBA16F accumulate — RGB: (ONE, ONE) add, A: (ZERO, ONE_MINUS_SRC_ALPHA) add
  desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
  desc.colorAttachments[0].blendingEnabled = YES;
  desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
  desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
  desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
  desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
  desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorZero;
  desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
  // color(1): R16F revealage — (ONE, ONE) add
  desc.colorAttachments[1].pixelFormat = MTLPixelFormatR16Float;
  desc.colorAttachments[1].blendingEnabled = YES;
  desc.colorAttachments[1].rgbBlendOperation = MTLBlendOperationAdd;
  desc.colorAttachments[1].alphaBlendOperation = MTLBlendOperationAdd;
  desc.colorAttachments[1].sourceRGBBlendFactor = MTLBlendFactorOne;
  desc.colorAttachments[1].destinationRGBBlendFactor = MTLBlendFactorOne;
  desc.colorAttachments[1].sourceAlphaBlendFactor = MTLBlendFactorOne;
  desc.colorAttachments[1].destinationAlphaBlendFactor = MTLBlendFactorOne;
  // Depth test against the resolved opaque depth (read-only state set by the pass)
  desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
  desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
  // OIT runs without MSAA (matching depth peeling); the accumulate textures
  // are non-MSAA, so the sample count must stay 1.
  desc.rasterSampleCount = 1;

  NSError* error = nil;
  this->Internals->TriangleOITPipeline =
    [device newRenderPipelineStateWithDescriptor:desc error:&error];
  if (!this->Internals->TriangleOITPipeline)
  {
    vtkErrorMacro(<< "OIT accumulate pipeline: " << [[error localizedDescription] UTF8String]);
  }
  [desc release];
  [vertexDesc release];
  [vertexFunc release];
  [fragFunc release];
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::UpdateMaterialUniforms(void* mtlDevice, vtkActor* actor)
{
  if (!actor || !mtlDevice)
  {
    return;
  }

  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;
  vtkProperty* prop = actor->GetProperty();

  // Flat layout matching Metal shader's MaterialUniforms byte-for-byte.
  // ambientColor: rgb + ambient_intensity
  // diffuseColor: rgb + diffuse_intensity
  // specularColor: rgb + specular_intensity
  // color: base color (unused in lighting)
  // opacity, specularPower, 2 pad
  // then the same 20 floats again for the backface material.
  float mu[40];
  memset(mu, 0, sizeof(mu));

  // ambientColor.rgb = property ambient color, .w = ambient intensity
  // When batch color override is active, use the batch color for ambient too
  double ac[3];
  if (this->Internals->UseBatchColor)
  {
    ac[0] = this->Internals->BatchColor[0];
    ac[1] = this->Internals->BatchColor[1];
    ac[2] = this->Internals->BatchColor[2];
  }
  else
  {
    prop->GetAmbientColor(ac);
  }
  mu[0] = static_cast<float>(ac[0]);
  mu[1] = static_cast<float>(ac[1]);
  mu[2] = static_cast<float>(ac[2]);
  mu[3] = static_cast<float>(prop->GetAmbient());

  // diffuseColor.rgb = property diffuse color, .w = diffuse intensity
  double dc[3];
  if (this->Internals->UseBatchColor)
  {
    dc[0] = this->Internals->BatchColor[0];
    dc[1] = this->Internals->BatchColor[1];
    dc[2] = this->Internals->BatchColor[2];
  }
  else
  {
    prop->GetDiffuseColor(dc);
  }
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
  if (this->Internals->UseBatchColor)
  {
    rgb[0] = this->Internals->BatchColor[0];
    rgb[1] = this->Internals->BatchColor[1];
    rgb[2] = this->Internals->BatchColor[2];
  }
  else
  {
    prop->GetColor(rgb);
  }
  mu[12] = static_cast<float>(rgb[0]);
  mu[13] = static_cast<float>(rgb[1]);
  mu[14] = static_cast<float>(rgb[2]);
  mu[15] = 1.0f;

  // Option A: when batch override is active, opacity is baked into vertex colors,
  // so material opacity is 1.0 to avoid double multiplication.
  // Otherwise, use the actor property opacity.
  mu[16] = (this->Internals->UseBatchColor || this->Internals->UseBatchOpacity)
    ? 1.0f
    : static_cast<float>(prop->GetOpacity());
  mu[17] = static_cast<float>(prop->GetSpecularPower());

  // Backface material: mirror the front material, then override with the
  // actor's backface property when present (matches vtkOpenGLPolyDataMapper,
  // which substitutes the backface property's lighting params on backfaces).
  memcpy(&mu[20], &mu[0], 18 * sizeof(float));
  if (vtkProperty* backProp = actor->GetBackfaceProperty())
  {
    double bac[3], bdc[3], bsc[3], brgb[3];
    backProp->GetAmbientColor(bac);
    backProp->GetDiffuseColor(bdc);
    backProp->GetSpecularColor(bsc);
    backProp->GetColor(brgb);
    mu[20] = static_cast<float>(bac[0]);
    mu[21] = static_cast<float>(bac[1]);
    mu[22] = static_cast<float>(bac[2]);
    mu[23] = static_cast<float>(backProp->GetAmbient());
    mu[24] = static_cast<float>(bdc[0]);
    mu[25] = static_cast<float>(bdc[1]);
    mu[26] = static_cast<float>(bdc[2]);
    mu[27] = static_cast<float>(backProp->GetDiffuse());
    mu[28] = static_cast<float>(bsc[0]);
    mu[29] = static_cast<float>(bsc[1]);
    mu[30] = static_cast<float>(bsc[2]);
    mu[31] = static_cast<float>(backProp->GetSpecular());
    mu[32] = static_cast<float>(brgb[0]);
    mu[33] = static_cast<float>(brgb[1]);
    mu[34] = static_cast<float>(brgb[2]);
    mu[35] = 1.0f;
    mu[36] = static_cast<float>(backProp->GetOpacity());
    mu[37] = static_cast<float>(backProp->GetSpecularPower());
  }

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

  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;

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

  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;

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

  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;

  // VertexColorUniforms layout: float4 (16 bytes)
  // Default white; overridden when vertex visibility is on.
  float vc[4] = { 1.0f, 1.0f, 1.0f, 1.0f };

  if (actor->GetProperty()->GetVertexVisibility())
  {
    if (this->Internals->UseBatchColor)
    {
      vc[0] = static_cast<float>(this->Internals->BatchColor[0]);
      vc[1] = static_cast<float>(this->Internals->BatchColor[1]);
      vc[2] = static_cast<float>(this->Internals->BatchColor[2]);
    }
    else
    {
      double vcol[3];
      actor->GetProperty()->GetVertexColor(vcol);
      vc[0] = static_cast<float>(vcol[0]);
      vc[1] = static_cast<float>(vcol[1]);
      vc[2] = static_cast<float>(vcol[2]);
    }

    if (this->Internals->UseBatchOpacity)
    {
      vc[3] = static_cast<float>(this->Internals->BatchOpacity);
    }
    else if (this->Internals->UseBatchColor)
    {
      // Color override without explicit opacity: bake actor opacity
      vc[3] = static_cast<float>(actor->GetProperty()->GetOpacity());
    }
    else
    {
      vc[3] = 1.0f;
    }
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

  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;

  float ec[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

  if (this->Internals->UseBatchColor)
  {
    ec[0] = static_cast<float>(this->Internals->BatchColor[0]);
    ec[1] = static_cast<float>(this->Internals->BatchColor[1]);
    ec[2] = static_cast<float>(this->Internals->BatchColor[2]);
  }
  else
  {
    // Use edge color from actor's property
    vtkProperty* prop = actor->GetProperty();
    if (prop)
    {
      double ecDouble[3];
      prop->GetEdgeColor(ecDouble);
      ec[0] = static_cast<float>(ecDouble[0]);
      ec[1] = static_cast<float>(ecDouble[1]);
      ec[2] = static_cast<float>(ecDouble[2]);
    }
  }

  if (this->Internals->UseBatchOpacity)
  {
    ec[3] = static_cast<float>(this->Internals->BatchOpacity);
  }
  else if (this->Internals->UseBatchColor)
  {
    // Color override without explicit opacity: bake actor opacity
    ec[3] = static_cast<float>(actor->GetProperty()->GetOpacity());
  }

  if (!this->Internals->EdgeColorUniformBuffer)
  {
    id<MTLBuffer> buffer =
      [device newBufferWithLength:sizeof(ec)
                          options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->EdgeColorUniformBuffer, buffer);
  }

  memcpy([this->Internals->EdgeColorUniformBuffer contents], ec, sizeof(ec));
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::UpdateEdgeUniforms(void* mtlDevice, vtkActor* actor)
{
  if (!mtlDevice || !actor)
  {
    return;
  }

  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;

  // Must stay layout-compatible with EdgeUniforms in MetalShaders.metal.
  struct MetalEdgeUniforms
  {
    float edgeColor[4];
    float edgeWidth;
    uint32_t flags;
    float pad;
  };

  MetalEdgeUniforms e;
  memset(&e, 0, sizeof(e));
  e.edgeColor[3] = 1.0f;
  e.edgeWidth = 1.1f;

  vtkProperty* prop = actor->GetProperty();
  if (prop)
  {
    double ecDouble[3] = { 0.0, 0.0, 0.0 };
    prop->GetEdgeColor(ecDouble);
    e.edgeColor[0] = static_cast<float>(ecDouble[0]);
    e.edgeColor[1] = static_cast<float>(ecDouble[1]);
    e.edgeColor[2] = static_cast<float>(ecDouble[2]);
    e.edgeColor[3] = static_cast<float>(prop->GetEdgeOpacity());
    double w = prop->GetUseLineWidthForEdgeThickness() ? prop->GetLineWidth() : prop->GetEdgeWidth();
    if (w < 1.1)
    {
      w = 1.1;
    }
    e.edgeWidth = static_cast<float>(w);
    e.flags = (prop->GetEdgeVisibility() ? 1u : 0u) | (prop->GetRenderLinesAsTubes() ? 2u : 0u);
  }

  if (this->Internals->UseBatchColor)
  {
    e.edgeColor[0] = static_cast<float>(this->Internals->BatchColor[0]);
    e.edgeColor[1] = static_cast<float>(this->Internals->BatchColor[1]);
    e.edgeColor[2] = static_cast<float>(this->Internals->BatchColor[2]);
  }

  if (!this->Internals->EdgeUniformBuffer)
  {
    id<MTLBuffer> buffer =
      [device newBufferWithLength:sizeof(e)
                          options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->EdgeUniformBuffer, buffer);
  }

  memcpy([this->Internals->EdgeUniformBuffer contents], &e, sizeof(e));
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::UpdateClipPlaneUniforms(void* mtlDevice, vtkActor* actor)
{
  if (!mtlDevice || !actor)
  {
    return;
  }

  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;

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

  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;

  vtkTexture* texture = actor->GetTexture();

  if (!texture || !texture->GetInput())
  {
    if (this->Internals->ActorTexture != nil ||
        this->Internals->ActorSampler != nil)
    {
      vtkMetalMRC::ReleaseAndNil(this->Internals->ActorTexture);
      vtkMetalMRC::ReleaseAndNil(this->Internals->ActorSampler);
      this->Internals->CachedTextureMTime = 0;
      this->Internals->InvalidateRenderBundle();
    }
    return;
  }

  vtkMTimeType texMTime = texture->GetMTime();

  if (this->Internals->ActorTexture &&
      texMTime == this->Internals->CachedTextureMTime)
  {
    return;
  }

  // Texture changed.
  this->Internals->InvalidateRenderBundle();
  this->Internals->CachedTextureMTime = texMTime;

  vtkImageData* image = texture->GetInput();
  int extent[6];
  image->GetExtent(extent);
  int width = extent[1] - extent[0] + 1;
  int height = extent[3] - extent[2] + 1;
  int numComponents = image->GetNumberOfScalarComponents();

  if (width <= 0 || height <= 0)
  {
    vtkMetalMRC::ReleaseAndNil(this->Internals->ActorTexture);
    vtkMetalMRC::ReleaseAndNil(this->Internals->ActorSampler);
    this->Internals->CachedTextureMTime = 0;
    this->Internals->InvalidateRenderBundle();
    return;
  }

  // Determine pixel format — convert to RGBA8Unorm for simplicity
  MTLPixelFormat pixelFormat = MTLPixelFormatRGBA8Unorm;

  // Create texture descriptor
  MTLTextureDescriptor* texDesc = [[MTLTextureDescriptor alloc] init];
  texDesc.textureType = MTLTextureType2D;
  texDesc.pixelFormat = pixelFormat;
  texDesc.width = width;
  texDesc.height = height;
  texDesc.mipmapLevelCount = 1;
  texDesc.usage = MTLTextureUsageShaderRead;
  texDesc.storageMode = MTLStorageModeShared;

  id<MTLTexture> newTexture = [device newTextureWithDescriptor:texDesc];
  vtkMetalMRC::AssignConsumed(this->Internals->ActorTexture, newTexture);
  [texDesc release];
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
    vtkMetalMRC::ReleaseAndNil(this->Internals->ActorTexture);
    vtkMetalMRC::ReleaseAndNil(this->Internals->ActorSampler);
    this->Internals->CachedTextureMTime = 0;
    this->Internals->InvalidateRenderBundle();
    return;
  }

  // Guard: only unsigned char textures are supported
  if (scalars->GetDataType() != VTK_UNSIGNED_CHAR)
  {
    vtkErrorMacro(<< "vtkMetalPolyDataMapper: only unsigned char textures are currently supported");
    delete[] rgbaData;
    vtkMetalMRC::ReleaseAndNil(this->Internals->ActorTexture);
    vtkMetalMRC::ReleaseAndNil(this->Internals->ActorSampler);
    this->Internals->CachedTextureMTime = texMTime;
    this->Internals->InvalidateRenderBundle();
    return;
  }

  int xMin = extent[0];
  int yMin = extent[2];

  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      // Match vtkOpenGLTexture/vtkOpenGLPolyDataMapper: VTK image row 0 (min y)
      // is uploaded first, so texcoord (0,0) samples the bottom row exactly as
      // OpenGL renders it. No vertical flip.
      unsigned char* srcPtr = static_cast<unsigned char*>(image->GetScalarPointer(xMin + x, yMin + y, 0));
      int dstIdx = (y * width + x) * 4;
      unsigned char* dst = rgbaData + dstIdx;

      switch (numComponents)
      {
        case 1:
          dst[0] = dst[1] = dst[2] = srcPtr[0];
          dst[3] = 255;
          break;
        case 2:
          dst[0] = dst[1] = dst[2] = srcPtr[0];
          dst[3] = srcPtr[1];
          break;
        case 3:
          dst[0] = srcPtr[0];
          dst[1] = srcPtr[1];
          dst[2] = srcPtr[2];
          dst[3] = 255;
          break;
        case 4:
          dst[0] = srcPtr[0];
          dst[1] = srcPtr[1];
          dst[2] = srcPtr[2];
          dst[3] = srcPtr[3];
          break;
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

  // Create sampler state. Match vtkTexture's repeat/edge-clamp settings
  // (GL: GL_REPEAT, GL_CLAMP_TO_EDGE, or GL_CLAMP).
  MTLSamplerDescriptor* samplerDesc = [[MTLSamplerDescriptor alloc] init];
  MTLSamplerAddressMode addrMode =
    texture->GetRepeat() ? MTLSamplerAddressModeRepeat : MTLSamplerAddressModeClampToEdge;
  samplerDesc.sAddressMode = addrMode;
  samplerDesc.tAddressMode = addrMode;
  samplerDesc.minFilter = MTLSamplerMinMagFilterLinear;
  samplerDesc.magFilter = MTLSamplerMinMagFilterLinear;
  id<MTLSamplerState> newSampler = [device newSamplerStateWithDescriptor:samplerDesc];
  vtkMetalMRC::AssignConsumed(this->Internals->ActorSampler, newSampler);
  [samplerDesc release];
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::SetOverridePropId(uint32_t zeroBasedPropId)
{
  this->Internals->HasOverridePropId = true;
  this->Internals->OverridePropId = zeroBasedPropId;
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::SetOverridePropIdToNone()
{
  // UINT32_MAX is the explicit no-pick sentinel. The shader maps it to 0.
  this->Internals->HasOverridePropId = true;
  this->Internals->OverridePropId = UINT32_MAX;
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::ClearOverridePropId()
{
  this->Internals->HasOverridePropId = false;
  this->Internals->OverridePropId = 0;
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::SetOverrideCompositeIndex(uint32_t compositeIndex)
{
  this->Internals->HasOverrideCompositeIndex = true;
  this->Internals->OverrideCompositeIndex = compositeIndex;
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::ClearOverrideCompositeIndex()
{
  this->Internals->HasOverrideCompositeIndex = false;
  this->Internals->OverrideCompositeIndex = 0;
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::SetBatchVisualOverride(
    bool overrideColor,
    const double color[3],
    bool overrideOpacity,
    double opacity)
{
  bool changed = false;

  if (this->Internals->UseBatchColor != overrideColor)
  {
    this->Internals->UseBatchColor = overrideColor;
    changed = true;
  }

  if (overrideColor)
  {
    for (int i = 0; i < 3; ++i)
    {
      if (this->Internals->BatchColor[i] != color[i])
      {
        this->Internals->BatchColor[i] = color[i];
        changed = true;
      }
    }
  }

  if (this->Internals->UseBatchOpacity != overrideOpacity)
  {
    this->Internals->UseBatchOpacity = overrideOpacity;
    changed = true;
  }

  if (overrideOpacity)
  {
    if (this->Internals->BatchOpacity != opacity)
    {
      this->Internals->BatchOpacity = opacity;
      changed = true;
    }
  }

  if (changed)
  {
    this->Internals->BatchOverrideMTime++;
    this->Internals->InvalidateRenderBundle();
    this->Modified();
  }
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::ClearBatchVisualOverride()
{
  if (this->Internals->UseBatchColor || this->Internals->UseBatchOpacity)
  {
    this->Internals->UseBatchColor = false;
    this->Internals->UseBatchOpacity = false;
    this->Internals->BatchOverrideMTime++;
    this->Internals->InvalidateRenderBundle();
    this->Modified();
  }
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::UpdatePickUniforms(vtkRenderer* ren, vtkActor* act)
{
  uint32_t propId = 0;
  uint32_t compositeIndex = 0;

  if (this->Internals->HasOverridePropId && this->Internals->OverridePropId == UINT32_MAX)
  {
    // Explicit no-pick sentinel (shader maps UINT32_MAX + 1 to 0).
    propId = UINT32_MAX;
  }
  else if (vtkMetalHardwareSelector* sel =
      ren ? vtkMetalHardwareSelector::SafeDownCast(ren->GetSelector()) : nullptr)
  {
    // Per-render prop ID: the prop's index in the selector's visible PropArray.
    int id = sel->GetPropID(act);
    if (id >= 0)
    {
      propId = static_cast<uint32_t>(id);
    }
    else
    {
      // Defensive: an actor rendered outside the selector's PropArray (e.g. a
      // multi-renderer configuration) would otherwise resolve to PropArray[0].
      propId = UINT32_MAX;
    }
  }
  else if (this->Internals->HasOverridePropId)
  {
    propId = this->Internals->OverridePropId;
  }

  if (this->Internals->HasOverrideCompositeIndex)
  {
    compositeIndex = this->Internals->OverrideCompositeIndex;
  }

  if (this->Internals->PropIdBuffer)
  {
    PickIds ids = { propId, compositeIndex };
    memcpy([this->Internals->PropIdBuffer contents], &ids, sizeof(PickIds));
  }
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper::SetPropId(uint32_t propId)
{
  if (this->Internals->PropIdBuffer)
  {
    PickIds ids;
    memcpy(&ids, [this->Internals->PropIdBuffer contents], sizeof(PickIds));
    ids.PropId = propId;
    memcpy([this->Internals->PropIdBuffer contents], &ids, sizeof(PickIds));
  }
}

//------------------------------------------------------------------------------
uint32_t vtkMetalPolyDataMapper::GetOrCreatePropId(vtkActor* act)
{
  // DEPRECATED: Prop IDs are assigned per-render by the hardware selector.
  // This function returns 0 and should not be called externally.
  (void)act;
  vtkWarningMacro(<< "GetOrCreatePropId is deprecated. "
                  << "Prop IDs are assigned per-render during selection passes.");
  return 0;
}

VTK_ABI_NAMESPACE_END
