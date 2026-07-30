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

#include "vtkMetalMRC.h"

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

  // Index buffers for indexed drawing
  id<MTLBuffer> TriangleIndexBuffer = nil;
  id<MTLBuffer> LineIndexBuffer = nil;
  id<MTLBuffer> PointIndexBuffer = nil;

  NSUInteger TriangleIndexCount = 0;
  NSUInteger LineIndexCount = 0;
  NSUInteger PointIndexCount = 0;

  // Uniform buffer for 2D state
  id<MTLBuffer> StateBuffer = nil;

  vtkIdType VertexCount = 0;
  bool HasTriangles = false;
  bool HasLines = false;
  bool HasPoints = false;

  // Cached state
  vtkIdType CachedInputMTime = 0;
  int CachedSampleCount = 0;
  MTLPixelFormat CachedDepthFormat = MTLPixelFormatInvalid;
  int CachedViewportSize[2] = { 0, 0 };
  double CachedViewportRect[4] = { 0.0, 0.0, 1.0, 1.0 };

  id<MTLDepthStencilState> OverlayDepthState = nil;

  void ReleasePipelines()
  {
    vtkMetalMRC::ReleaseAndNil(TrianglePipeline);
    vtkMetalMRC::ReleaseAndNil(LinePipeline);
    vtkMetalMRC::ReleaseAndNil(PointPipeline);
  }

  void ReleaseBuffers()
  {
    vtkMetalMRC::ReleaseAndNil(PositionBuffer);
    vtkMetalMRC::ReleaseAndNil(StateBuffer);

    vtkMetalMRC::ReleaseAndNil(TriangleIndexBuffer);
    vtkMetalMRC::ReleaseAndNil(LineIndexBuffer);
    vtkMetalMRC::ReleaseAndNil(PointIndexBuffer);

    TriangleIndexCount = 0;
    LineIndexCount = 0;
    PointIndexCount = 0;

    VertexCount = 0;
    HasTriangles = false;
    HasLines = false;
    HasPoints = false;

    CachedInputMTime = 0;
    CachedSampleCount = 0;
    CachedViewportSize[0] = 0;
    CachedViewportSize[1] = 0;
    CachedViewportRect[0] = 0.0;
    CachedViewportRect[1] = 0.0;
    CachedViewportRect[2] = 1.0;
    CachedViewportRect[3] = 1.0;
  }

  void ReleaseState()
  {
    vtkMetalMRC::ReleaseAndNil(OverlayDepthState);
  }

  ~vtkMetalPolyDataMapper2DInternals()
  {
    ReleaseBuffers();
    ReleaseState();
    ReleasePipelines();
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
    id<MTLDevice> device = (id<MTLDevice>)renWin->GetMetalDevice();
    id<MTLRenderCommandEncoder> encoder =
      (id<MTLRenderCommandEncoder>)renWin->GetCurrentRenderCommandEncoder();
    if (!encoder)
    {
      return;
    }

    // Check if geometry needs rebuilding
    vtkIdType currentMTime = input->GetMTime();
    int sampleCount = renWin->GetEffectiveSampleCount();
    bool sampleCountChanged = (sampleCount != this->Internals->CachedSampleCount);

    // Compute viewport-dependent state
    int* size = viewport->GetSize();
    double* vp = viewport->GetViewport();
    bool viewportChanged =
        size[0] != this->Internals->CachedViewportSize[0] ||
        size[1] != this->Internals->CachedViewportSize[1] ||
        vp[0] != this->Internals->CachedViewportRect[0] ||
        vp[1] != this->Internals->CachedViewportRect[1] ||
        vp[2] != this->Internals->CachedViewportRect[2] ||
        vp[3] != this->Internals->CachedViewportRect[3];

    bool geometryDirty =
        currentMTime != this->Internals->CachedInputMTime ||
        (this->TransformCoordinate && viewportChanged);

    // Release pipelines on sample-count or depth-format change
    MTLPixelFormat depthFormat = MTLPixelFormatInvalid;
    id<MTLTexture> depthTex = (id<MTLTexture>)renWin->GetDepthTexture();
    if (depthTex)
    {
      depthFormat = [depthTex pixelFormat];
    }
    bool depthFormatChanged =
        depthFormat != this->Internals->CachedDepthFormat;

    if (sampleCountChanged || depthFormatChanged)
    {
      this->Internals->ReleasePipelines();
      this->Internals->CachedSampleCount = sampleCount;
      this->Internals->CachedDepthFormat = depthFormat;
    }

    if (geometryDirty)
    {
      this->Internals->ReleaseBuffers();
      this->Internals->CachedInputMTime = currentMTime;
      this->Internals->CachedSampleCount = sampleCount;
      this->Internals->CachedViewportSize[0] = size[0];
      this->Internals->CachedViewportSize[1] = size[1];
      this->Internals->CachedViewportRect[0] = vp[0];
      this->Internals->CachedViewportRect[1] = vp[1];
      this->Internals->CachedViewportRect[2] = vp[2];
      this->Internals->CachedViewportRect[3] = vp[3];

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

        id<MTLBuffer> posBuffer = [device
          newBufferWithBytes:positions.data()
                     length:positions.size() * sizeof(float)
                    options:MTLResourceStorageModeShared];
        vtkMetalMRC::AssignConsumed(this->Internals->PositionBuffer, posBuffer);

      }

      // Build index buffers from cell arrays
      std::vector<uint32_t> triIndices;
      std::vector<uint32_t> lineIndices;
      std::vector<uint32_t> pointIndices;

      vtkCellArray* polys = input->GetPolys();
      vtkCellArray* lines = input->GetLines();
      vtkCellArray* verts = input->GetVerts();

      if (polys)
      {
        const vtkIdType* ids = nullptr;
        vtkIdType npts = 0;
        polys->InitTraversal();
        while (polys->GetNextCell(npts, ids))
        {
          if (npts < 3) continue;
          for (vtkIdType i = 1; i < npts - 1; ++i)
          {
            triIndices.push_back(static_cast<uint32_t>(ids[0]));
            triIndices.push_back(static_cast<uint32_t>(ids[i]));
            triIndices.push_back(static_cast<uint32_t>(ids[i + 1]));
          }
        }
      }

      if (lines)
      {
        const vtkIdType* ids = nullptr;
        vtkIdType npts = 0;
        lines->InitTraversal();
        while (lines->GetNextCell(npts, ids))
        {
          if (npts < 2) continue;
          for (vtkIdType i = 0; i < npts - 1; ++i)
          {
            lineIndices.push_back(static_cast<uint32_t>(ids[i]));
            lineIndices.push_back(static_cast<uint32_t>(ids[i + 1]));
          }
        }
      }

      if (verts)
      {
        const vtkIdType* ids = nullptr;
        vtkIdType npts = 0;
        verts->InitTraversal();
        while (verts->GetNextCell(npts, ids))
        {
          for (vtkIdType i = 0; i < npts; ++i)
          {
            pointIndices.push_back(static_cast<uint32_t>(ids[i]));
          }
        }
      }

      this->Internals->HasTriangles = !triIndices.empty();
      this->Internals->HasLines = !lineIndices.empty();
      this->Internals->HasPoints = !pointIndices.empty();

      if (!triIndices.empty())
      {
        id<MTLBuffer> buffer =
          [device newBufferWithBytes:triIndices.data()
                              length:triIndices.size() * sizeof(uint32_t)
                             options:MTLResourceStorageModeShared];
        vtkMetalMRC::AssignConsumed(this->Internals->TriangleIndexBuffer, buffer);
        this->Internals->TriangleIndexCount = triIndices.size();
      }

      if (!lineIndices.empty())
      {
        id<MTLBuffer> buffer =
          [device newBufferWithBytes:lineIndices.data()
                              length:lineIndices.size() * sizeof(uint32_t)
                             options:MTLResourceStorageModeShared];
        vtkMetalMRC::AssignConsumed(this->Internals->LineIndexBuffer, buffer);
        this->Internals->LineIndexCount = lineIndices.size();
      }

      if (!pointIndices.empty())
      {
        id<MTLBuffer> buffer =
          [device newBufferWithBytes:pointIndices.data()
                              length:pointIndices.size() * sizeof(uint32_t)
                             options:MTLResourceStorageModeShared];
        vtkMetalMRC::AssignConsumed(this->Internals->PointIndexBuffer, buffer);
        this->Internals->PointIndexCount = pointIndices.size();
      }
    }

    // Compute WCVC (world-to-viewport-clip) matrix
    // For 2D overlay: orthographic projection from viewport pixel coordinates
    // (size and vp are already captured above)
    // Orthographic projection: maps [vp_x*size_w, (vp_x+vp_w)*size_w] x [vp_y*size_h, (vp_y+vp_h)*size_h] to [-1,1]
    if (size[0] <= 0 || size[1] <= 0)
    {
      return;
    }
    float vpX = static_cast<float>(vp[0] * size[0]);
    float vpY = static_cast<float>(vp[1] * size[1]);
    float vpW = static_cast<float>((vp[2] - vp[0]) * size[0]);
    float vpH = static_cast<float>((vp[3] - vp[1]) * size[1]);
    if (vpW <= 0.0f || vpH <= 0.0f)
    {
      return;
    }

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

    // Color is passed via StateBuffer at index 1 (vertex) / 0 (fragment); no per-vertex ColorBuffer needed

    // Create pipeline states if needed
    if (!this->Internals->TrianglePipeline || !this->Internals->LinePipeline ||
        !this->Internals->PointPipeline)
    {
      NSError* error = nil;
      id<MTLLibrary> library = (__bridge id<MTLLibrary>)renWin->GetSharedShaderLibrary();
      if (!library)
      {
        vtkErrorMacro(<< "No shared shader library available for 2D mapper");
        return;
      }

      id<MTLFunction> vFunc = [library newFunctionWithName:@"vertex_2d_main"];
      id<MTLFunction> fFunc = [library newFunctionWithName:@"fragment_2d_main"];
      if (!vFunc || !fFunc)
      {
        vtkErrorMacro(<< "Failed to find 2D shader functions");
        [vFunc release];
        [fFunc release];
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
        desc.depthAttachmentPixelFormat = depthFormat;
        desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
        desc.rasterSampleCount = sampleCount;
        desc.colorAttachments[0].blendingEnabled = YES;
        desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

        this->Internals->TrianglePipeline =
          [device newRenderPipelineStateWithDescriptor:desc error:&error];
        if (!this->Internals->TrianglePipeline)
        {
          vtkErrorMacro(<< "2D triangle pipeline: " << [[error localizedDescription] UTF8String]);
        }
        [desc release];
      }

      // Line pipeline
      if (!this->Internals->LinePipeline)
      {
        MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
        desc.vertexFunction = vFunc;
        desc.fragmentFunction = fFunc;
        desc.vertexDescriptor = vertexDesc;
        desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        desc.depthAttachmentPixelFormat = depthFormat;
        desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassLine;
        desc.rasterSampleCount = sampleCount;
        desc.colorAttachments[0].blendingEnabled = YES;
        desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

        this->Internals->LinePipeline =
          [device newRenderPipelineStateWithDescriptor:desc error:&error];
        if (!this->Internals->LinePipeline)
        {
          vtkErrorMacro(<< "2D line pipeline: " << [[error localizedDescription] UTF8String]);
        }
        [desc release];
      }

      // Point pipeline
      if (!this->Internals->PointPipeline)
      {
        MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
        desc.vertexFunction = vFunc;
        desc.fragmentFunction = fFunc;
        desc.vertexDescriptor = vertexDesc;
        desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        desc.depthAttachmentPixelFormat = depthFormat;
        desc.rasterSampleCount = sampleCount;
        desc.colorAttachments[0].blendingEnabled = YES;
        desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

        this->Internals->PointPipeline =
          [device newRenderPipelineStateWithDescriptor:desc error:&error];
        if (!this->Internals->PointPipeline)
        {
          vtkErrorMacro(<< "2D point pipeline: " << [[error localizedDescription] UTF8String]);
        }
        [desc release];
      }

      [vertexDesc release];
      [vFunc release];
      [fFunc release];
    }

    // Create and bind overlay depth-stencil state (always pass, no write)
    if (!this->Internals->OverlayDepthState)
    {
      MTLDepthStencilDescriptor* dsDesc = [[MTLDepthStencilDescriptor alloc] init];
      dsDesc.depthCompareFunction = MTLCompareFunctionAlways;
      dsDesc.depthWriteEnabled = NO;
      this->Internals->OverlayDepthState =
          [device newDepthStencilStateWithDescriptor:dsDesc];
      [dsDesc release];
    }
    [encoder setDepthStencilState:this->Internals->OverlayDepthState];

    // Set state buffer at buffer index 1
    [encoder setVertexBuffer:this->Internals->StateBuffer offset:0 atIndex:1];
    [encoder setFragmentBuffer:this->Internals->StateBuffer offset:0 atIndex:0];

    // Set position buffer at buffer index 0
    [encoder setVertexBuffer:this->Internals->PositionBuffer offset:0 atIndex:0];

    // Draw triangles using index buffer
    if (this->Internals->HasTriangles &&
        this->Internals->TrianglePipeline &&
        this->Internals->TriangleIndexBuffer &&
        this->Internals->TriangleIndexCount > 0)
    {
      [encoder setRenderPipelineState:this->Internals->TrianglePipeline];

      [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                          indexCount:this->Internals->TriangleIndexCount
                           indexType:MTLIndexTypeUInt32
                         indexBuffer:this->Internals->TriangleIndexBuffer
                   indexBufferOffset:0];
    }

    // Draw lines using index buffer
    if (this->Internals->HasLines &&
        this->Internals->LinePipeline &&
        this->Internals->LineIndexBuffer &&
        this->Internals->LineIndexCount > 0)
    {
      [encoder setRenderPipelineState:this->Internals->LinePipeline];

      [encoder drawIndexedPrimitives:MTLPrimitiveTypeLine
                          indexCount:this->Internals->LineIndexCount
                           indexType:MTLIndexTypeUInt32
                         indexBuffer:this->Internals->LineIndexBuffer
                   indexBufferOffset:0];
    }

    // Draw points using index buffer
    if (this->Internals->HasPoints &&
        this->Internals->PointPipeline &&
        this->Internals->PointIndexBuffer &&
        this->Internals->PointIndexCount > 0)
    {
      [encoder setRenderPipelineState:this->Internals->PointPipeline];

      [encoder drawIndexedPrimitives:MTLPrimitiveTypePoint
                          indexCount:this->Internals->PointIndexCount
                           indexType:MTLIndexTypeUInt32
                         indexBuffer:this->Internals->PointIndexBuffer
                   indexBufferOffset:0];
    }
  }
}

//------------------------------------------------------------------------------
void vtkMetalPolyDataMapper2D::ReleaseGraphicsResources(vtkWindow* w)
{
  this->Internals->ReleaseBuffers();
  this->Internals->ReleaseState();
  this->Internals->ReleasePipelines();
  this->Superclass::ReleaseGraphicsResources(w);
}

VTK_ABI_NAMESPACE_END
