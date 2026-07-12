// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalPolyDataMapper2D.h"

#include "vtkMetalRenderWindow.h"
#include "vtkMetalShaders.h"
#include "vtkObjectFactory.h"
#include "vtkOverrideAttribute.h"
#include "vtkActor2D.h"
#include "vtkPolyData.h"
#include "vtkCellArray.h"
#include "vtkPointData.h"
#include "vtkProperty2D.h"
#include "vtkViewport.h"
#include "vtkCoordinate.h"
#include "vtkMatrix4x4.h"

#import <Metal/Metal.h>

#include <vector>
#include <cmath>
#include <cstdint>

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalPolyDataMapper2D);

//------------------------------------------------------------------------------
struct vtkMetalPolyDataMapper2D::vtkMetalPolyDataMapper2DInternals
{
  // Pipeline states for different primitive types
  id<MTLRenderPipelineState> TrianglePipeline = nil;
  id<MTLRenderPipelineState> LinePipeline = nil;
  id<MTLRenderPipelineState> PointPipeline = nil;

  // Geometry buffers
  id<MTLBuffer> PositionBuffer = nil;
  id<MTLBuffer> ColorBuffer = nil;

  // Uniform buffer for 2D state
  id<MTLBuffer> StateBuffer = nil;

  vtkIdType VertexCount = 0;
  bool HasTriangles = false;
  bool HasLines = false;
  bool HasPoints = false;

  // Cached state
  vtkIdType CachedInputMTime = 0;
  int CachedSampleCount = 0;

  void ReleaseBuffers()
  {
    PositionBuffer = nil;
    ColorBuffer = nil;
    StateBuffer = nil;
    VertexCount = 0;
    HasTriangles = false;
    HasLines = false;
    HasPoints = false;
  }
};

//------------------------------------------------------------------------------
vtkMetalPolyDataMapper2D::vtkMetalPolyDataMapper2D()
  : Internals(new vtkMetalPolyDataMapper2DInternals())
{
}

//------------------------------------------------------------------------------
vtkMetalPolyDataMapper2D::~vtkMetalPolyDataMapper2D() = default;

//------------------------------------------------------------------------------
vtkOverrideAttribute* vtkMetalPolyDataMapper2D::CreateOverrideAttributes()
{
  auto* renderingBackendAttribute =
    vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "Metal", nullptr);
  return renderingBackendAttribute;
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper2D::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper2D::RenderOverlay(vtkViewport* viewport, vtkActor2D* actor)
{
  vtkMetalRenderWindow* renWin =
    vtkMetalRenderWindow::SafeDownCast(viewport->GetVTKWindow());
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
    id<MTLRenderCommandEncoder> encoder =
      (__bridge id<MTLRenderCommandEncoder>)renWin->GetCurrentRenderCommandEncoder();
    if (!encoder)
    {
      return;
    }

    // Check if geometry needs rebuilding
    vtkIdType currentMTime = input->GetMTime();
    int sampleCount = renWin->GetEffectiveSampleCount();
    if (currentMTime != this->Internals->CachedInputMTime ||
        sampleCount != this->Internals->CachedSampleCount)
    {
      this->Internals->ReleaseBuffers();
      this->Internals->CachedInputMTime = currentMTime;
      this->Internals->CachedSampleCount = sampleCount;

      // Build geometry from polydata
      vtkPoints* points = input->GetPoints();
      if (!points || points->GetNumberOfPoints() == 0)
      {
        return;
      }

      // Transform points to viewport coordinates if TransformCoordinate is set
      vtkPoints* transformedPoints = points;
      vtkNew<vtkPoints> tempPoints;
      if (this->TransformCoordinate)
      {
        vtkIdType numPts = points->GetNumberOfPoints();
        tempPoints->SetNumberOfPoints(numPts);
        for (vtkIdType i = 0; i < numPts; i++)
        {
          this->TransformCoordinate->SetValue(points->GetPoint(i));
          int* p = this->TransformCoordinate->GetComputedViewportValue(viewport);
          tempPoints->SetPoint(i, p[0], p[1], 0.0);
        }
        transformedPoints = tempPoints;
      }

      // Upload position data
      vtkDataArray* posData = transformedPoints->GetData();
      if (posData)
      {
        vtkIdType numPts = posData->GetNumberOfTuples();
        this->Internals->VertexCount = numPts;

        // Create position buffer (float2 per vertex)
        std::vector<float> positions(numPts * 2);
        for (vtkIdType i = 0; i < numPts; i++)
        {
          double* pt = posData->GetTuple(i);
          positions[i * 2] = static_cast<float>(pt[0]);
          positions[i * 2 + 1] = static_cast<float>(pt[1]);
        }

        this->Internals->PositionBuffer = [device
          newBufferWithBytes:positions.data()
                     length:positions.size() * sizeof(float)
                    options:MTLResourceStorageModeShared];
      }

      // Create color buffer from actor property
      double r, g, b;
      actor->GetProperty()->GetColor(r, g, b);
      double opacity = actor->GetProperty()->GetOpacity();

      // Upload color data (float4 per vertex)
      std::vector<float> colors(numPts * 4);
      for (vtkIdType i = 0; i < numPts; i++)
      {
        colors[i * 4] = static_cast<float>(r);
        colors[i * 4 + 1] = static_cast<float>(g);
        colors[i * 4 + 2] = static_cast<float>(b);
        colors[i * 4 + 3] = static_cast<float>(opacity);
      }

      this->Internals->ColorBuffer = [device
        newBufferWithBytes:colors.data()
                   length:colors.size() * sizeof(float)
                  options:MTLResourceStorageModeShared];

      // Determine primitive types
      vtkCellArray* polys = input->GetPolys();
      vtkCellArray* lines = input->GetLines();
      vtkCellArray* verts = input->GetVerts();

      this->Internals->HasTriangles = (polys && polys->GetNumberOfCells() > 0);
      this->Internals->HasLines = (lines && lines->GetNumberOfCells() > 0);
      this->Internals->HasPoints = (verts && verts->GetNumberOfCells() > 0);
    }

    // Get sample count for MSAA
    int sampleCount = this->Internals->CachedSampleCount;

    // Compute WCVC (world-to-viewport-clip) matrix
    // For 2D overlay: orthographic projection from viewport pixel coordinates
    int* size = viewport->GetSize();
    double* vp = viewport->GetViewport();

    // Orthographic projection: maps [vp_x*size_w, (vp_x+vp_w)*size_w] x [vp_y*size_h, (vp_y+vp_h)*size_h] to [-1,1]
    float vpX = static_cast<float>(vp[0] * size[0]);
    float vpY = static_cast<float>(vp[1] * size[1]);
    float vpW = static_cast<float>(vp[2] * size[0]);
    float vpH = static_cast<float>(vp[3] * size[1]);

    // Build orthographic matrix: maps viewport coords to NDC [-1,1]
    // Metal NDC: x in [-1,1], y in [-1,1], z in [0,1]
    // VTK viewport: origin at bottom-left, Metal origin at top-left
    float ndcLeft = -1.0f;
    float ndcRight = 1.0f;
    float ndcBottom = -1.0f;
    float ndcTop = 1.0f;
    float ndcNear = 0.0f;
    float ndcFar = 1.0f;

    float worldLeft = vpX;
    float worldRight = vpX + vpW;
    float worldBottom = vpY;
    float worldTop = vpY + vpH;

    // Standard orthographic matrix
    float wcvc[16] = { 0 };
    wcvc[0] = 2.0f / (worldRight - worldLeft);
    wcvc[5] = 2.0f / (worldTop - worldBottom);
    wcvc[10] = 1.0f / (ndcFar - ndcNear);
    wcvc[12] = -(worldRight + worldLeft) / (worldRight - worldLeft);
    wcvc[13] = -(worldTop + worldBottom) / (worldTop - worldBottom);
    wcvc[14] = -ndcNear / (ndcFar - ndcNear);
    wcvc[15] = 1.0f;

    // Flip Y to convert from VTK bottom-left to Metal top-left origin
    // This is done by negating the Y row in the matrix
    wcvc[1] = -wcvc[1];
    wcvc[5] = -wcvc[5];
    wcvc[9] = -wcvc[9];
    wcvc[13] = -wcvc[13];

    // Create or update state buffer
    struct Mapper2DState {
      float wcvcMatrix[16];
      float color[4];
      float pointSize;
      float lineWidth;
      uint32_t flags;
      uint32_t padding;
    };

    Mapper2DState state = {};
    memcpy(state.wcvcMatrix, wcvc, sizeof(float) * 16);
    double r, g, b;
    actor->GetProperty()->GetColor(r, g, b);
    double opacity = actor->GetProperty()->GetOpacity();
    state.color[0] = static_cast<float>(r);
    state.color[1] = static_cast<float>(g);
    state.color[2] = static_cast<float>(b);
    state.color[3] = static_cast<float>(opacity);
    state.pointSize = static_cast<float>(actor->GetProperty()->GetPointSize());
    state.lineWidth = static_cast<float>(actor->GetProperty()->GetLineWidth());
    state.flags = 0;
    state.padding = 0;

    if (!this->Internals->StateBuffer)
    {
      this->Internals->StateBuffer = [device
        newBufferWithBytes:&state
                   length:sizeof(state)
                  options:MTLResourceStorageModeShared];
    }
    else
    {
      memcpy([this->Internals->StateBuffer contents], &state, sizeof(state));
    }

    // Create pipeline states if needed
    if (!this->Internals->TrianglePipeline || !this->Internals->LinePipeline ||
        !this->Internals->PointPipeline)
    {
      NSError* error = nil;
      NSString* shaderSource = [NSString stringWithUTF8String:vtkMetalShaders];
      id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
      if (!library)
      {
        vtkErrorMacro(<< "Failed to compile Metal shader for 2D mapper: "
                      << [[error localizedDescription] UTF8String]);
        return;
      }

      id<MTLFunction> vFunc = [library newFunctionWithName:@"vertex_2d_main"];
      id<MTLFunction> fFunc = [library newFunctionWithName:@"fragment_2d_main"];
      if (!vFunc || !fFunc)
      {
        vtkErrorMacro(<< "Failed to find 2D shader functions");
        return;
      }

      MTLVertexDescriptor* vertexDesc = [[MTLVertexDescriptor alloc] init];
      vertexDesc.attributes[0].format = MTLVertexFormatFloat2;
      vertexDesc.attributes[0].offset = 0;
      vertexDesc.attributes[0].bufferIndex = 0;
      vertexDesc.layouts[0].stride = sizeof(float) * 2;
      vertexDesc.layouts[0].stepRate = 1;
      vertexDesc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;

      // Triangle pipeline
      if (!this->Internals->TrianglePipeline)
      {
        MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
        desc.vertexFunction = vFunc;
        desc.fragmentFunction = fFunc;
        desc.vertexDescriptor = vertexDesc;
        desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
        desc.sampleCount = sampleCount;

        this->Internals->TrianglePipeline =
          [device newRenderPipelineStateWithDescriptor:desc error:&error];
        if (!this->Internals->TrianglePipeline)
        {
          vtkErrorMacro(<< "2D triangle pipeline: " << [[error localizedDescription] UTF8String]);
        }
      }

      // Line pipeline
      if (!this->Internals->LinePipeline)
      {
        MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
        desc.vertexFunction = vFunc;
        desc.fragmentFunction = fFunc;
        desc.vertexDescriptor = vertexDesc;
        desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassLine;
        desc.sampleCount = sampleCount;

        this->Internals->LinePipeline =
          [device newRenderPipelineStateWithDescriptor:desc error:&error];
        if (!this->Internals->LinePipeline)
        {
          vtkErrorMacro(<< "2D line pipeline: " << [[error localizedDescription] UTF8String]);
        }
      }

      // Point pipeline
      if (!this->Internals->PointPipeline)
      {
        MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
        desc.vertexFunction = vFunc;
        desc.fragmentFunction = fFunc;
        desc.vertexDescriptor = vertexDesc;
        desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        desc.sampleCount = sampleCount;

        this->Internals->PointPipeline =
          [device newRenderPipelineStateWithDescriptor:desc error:&error];
        if (!this->Internals->PointPipeline)
        {
          vtkErrorMacro(<< "2D point pipeline: " << [[error localizedDescription] UTF8String]);
        }
      }
    }

    // Set state buffer at buffer index 1
    [encoder setVertexBuffer:this->Internals->StateBuffer offset:0 atIndex:1];
    [encoder setFragmentBuffer:this->Internals->StateBuffer offset:0 atIndex:0];

    // Set position buffer at buffer index 0
    [encoder setVertexBuffer:this->Internals->PositionBuffer offset:0 atIndex:0];

    // Draw triangles (fan triangulation of polygons)
    if (this->Internals->HasTriangles && this->Internals->TrianglePipeline)
    {
      [encoder setRenderPipelineState:this->Internals->TrianglePipeline];

      vtkCellArray* polys = input->GetPolys();
      const vtkIdType* indices = nullptr;
      vtkIdType cellSize = 0;

      vtkIdType cellIndex = 0;
      for (polys->InitTraversal(); polys->GetNextCell(cellSize, indices); cellIndex++)
      {
        if (cellSize < 3)
        {
          continue;
        }

        // Fan triangulation: 0-1-2, 0-2-3, 0-3-4, ...
        std::vector<uint32_t> triIndices;
        for (vtkIdType i = 1; i < cellSize - 1; i++)
        {
          triIndices.push_back(static_cast<uint32_t>(indices[0]));
          triIndices.push_back(static_cast<uint32_t>(indices[i]));
          triIndices.push_back(static_cast<uint32_t>(indices[i + 1]));
        }

        if (!triIndices.empty())
        {
          [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                              indexCount:triIndices.size()
                               indexType:MTLIndexTypeUInt32
                             indexBuffer:this->Internals->PositionBuffer
                       indexBufferOffset:0
                           instanceCount:1
                                baseVertex:0
                              baseInstance:0];
        }
      }
    }

    // Draw lines
    if (this->Internals->HasLines && this->Internals->LinePipeline)
    {
      [encoder setRenderPipelineState:this->Internals->LinePipeline];

      vtkCellArray* lines = input->GetLines();
      const vtkIdType* indices = nullptr;
      vtkIdType cellSize = 0;

      vtkIdType vertexOffset = 0;
      for (lines->InitTraversal(); lines->GetNextCell(cellSize, indices); vertexOffset += cellSize)
      {
        if (cellSize < 2)
        {
          continue;
        }

        // Draw line segments
        [encoder drawPrimitives:MTLPrimitiveTypeLine
                    vertexStart:vertexOffset
                    vertexCount:cellSize];
      }
    }

    // Draw points
    if (this->Internals->HasPoints && this->Internals->PointPipeline)
    {
      [encoder setRenderPipelineState:this->Internals->PointPipeline];

      vtkCellArray* verts = input->GetVerts();
      const vtkIdType* indices = nullptr;
      vtkIdType cellSize = 0;

      vtkIdType vertexOffset = 0;
      for (verts->InitTraversal(); verts->GetNextCell(cellSize, indices); vertexOffset += cellSize)
      {
        [encoder drawPrimitives:MTLPrimitiveTypePoint
                    vertexStart:vertexOffset
                    vertexCount:cellSize];
      }
    }
  }
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper2D::ReleaseGraphicsResources(vtkWindow* w)
{
  this->Internals->ReleaseBuffers();
  this->Internals->TrianglePipeline = nil;
  this->Internals->LinePipeline = nil;
  this->Internals->PointPipeline = nil;
  this->Superclass::ReleaseGraphicsResources(w);
}

VTK_ABI_NAMESPACE_END
