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
#include "vtkInformation.h"
#include "vtkProp.h"
#include "vtkUnsignedCharArray.h"

#import <Metal/Metal.h>

#include <vector>
#include <cmath>
#include <cstdint>

#include "vtkMetalMRC.h"

VTK_ABI_NAMESPACE_BEGIN

// Mapper2DState.flags bits. Must match the kFlagUse* constants in
// MetalShaders.metal.
constexpr uint32_t kFlagUseVertexColors = 1u;
constexpr uint32_t kFlagUseCellColors = 2u;

vtkStandardNewMacro(vtkMetalPolyDataMapper2D);

//------------------------------------------------------------------------------
struct vtkMetalPolyDataMapper2D::vtkMetalPolyDataMapper2DInternals
{
  // Pipeline states for different primitive types
  id<MTLRenderPipelineState> TrianglePipeline = nil;
  id<MTLRenderPipelineState> LinePipeline = nil;
  id<MTLRenderPipelineState> PointPipeline = nil;

  // Pipeline for textured geometry (text): samples the texture bound to the
  // actor's GENERAL_TEXTURE_UNIT and multiplies by the actor's color/opacity.
  id<MTLRenderPipelineState> TexturedPipeline = nil;

  // Geometry buffers
  id<MTLBuffer> PositionBuffer = nil;

  // Interleaved position + texture-coordinate buffer (float4 per vertex: xy
  // position, uv texcoord) used by the textured pipeline when the input
  // polydata carries TCoords (e.g. vtkTextMapper / vtkTextActor quads).
  id<MTLBuffer> TexCoordBuffer = nil;

  // Per-vertex color buffer (float4 per vertex, vertex buffer index 2). Used
  // when scalars map to per-vertex colors (point data), mirroring the OpenGL
  // "diffuseColor" vertex attribute. Always bound (as white when unused) so the
  // pipeline's color attribute never reads out of bounds.
  id<MTLBuffer> ColorBuffer = nil;

  // Per-primitive cell color buffers (float4 per primitive, fragment buffer
  // index 1). When cell scalars are present the shared corner vertices cannot
  // carry per-cell colors, so the fragment shader resolves them with
  // [[primitive_id]] -- the direct analog of GL's gl_PrimitiveID + textureC.
  // Each draw type starts its primitive id at 0, so each needs its own buffer.
  id<MTLBuffer> TriangleCellColorBuffer = nil;
  id<MTLBuffer> LineCellColorBuffer = nil;
  id<MTLBuffer> PointCellColorBuffer = nil;

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
  bool HasTextureCoords = false;
  bool HaveCellScalars = false;

  // Cached state
  vtkIdType CachedInputMTime = 0;
  vtkIdType CachedMapperMTime = 0;
  vtkIdType CachedActorMTime = 0;
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
    vtkMetalMRC::ReleaseAndNil(TexturedPipeline);
  }

  void ReleaseBuffers()
  {
    vtkMetalMRC::ReleaseAndNil(PositionBuffer);
    vtkMetalMRC::ReleaseAndNil(TexCoordBuffer);
    vtkMetalMRC::ReleaseAndNil(StateBuffer);
    vtkMetalMRC::ReleaseAndNil(ColorBuffer);
    vtkMetalMRC::ReleaseAndNil(TriangleCellColorBuffer);
    vtkMetalMRC::ReleaseAndNil(LineCellColorBuffer);
    vtkMetalMRC::ReleaseAndNil(PointCellColorBuffer);

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
    HasTextureCoords = false;
    HaveCellScalars = false;

    CachedInputMTime = 0;
    CachedMapperMTime = 0;
    CachedActorMTime = 0;
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
  this->GetInputAlgorithm()->Update();
  input = this->GetInput();

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
    vtkIdType mapperMTime = this->GetMTime();
    vtkIdType actorMTime = actor->GetMTime();
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
        mapperMTime != this->Internals->CachedMapperMTime ||
        actorMTime != this->Internals->CachedActorMTime ||
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
      this->Internals->CachedMapperMTime = mapperMTime;
      this->Internals->CachedActorMTime = actorMTime;
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

        // Upload texture coordinates (if any) as an interleaved
        // position+texcoord buffer for the textured pipeline. This mirrors the
        // OpenGL path, which uploads the polydata TCoords ("tcoordMC") and
        // samples the actor's bound texture when the texture unit is set.
        vtkDataArray* tcoords = input->GetPointData()->GetTCoords();
        if (tcoords && tcoords->GetNumberOfTuples() == numPts &&
          tcoords->GetNumberOfComponents() >= 2)
        {
          std::vector<float> posTc(numPts * 4);
          for (vtkIdType i = 0; i < numPts; i++)
          {
            double* pt = posData->GetTuple(i);
            double* tc = tcoords->GetTuple(i);
            posTc[i * 4] = static_cast<float>(pt[0]);
            posTc[i * 4 + 1] = static_cast<float>(pt[1]);
            posTc[i * 4 + 2] = static_cast<float>(tc[0]);
            posTc[i * 4 + 3] = static_cast<float>(tc[1]);
          }
          id<MTLBuffer> posTcBuffer = [device
            newBufferWithBytes:posTc.data()
                       length:posTc.size() * sizeof(float)
                      options:MTLResourceStorageModeShared];
          vtkMetalMRC::AssignConsumed(this->Internals->TexCoordBuffer, posTcBuffer);
          this->Internals->HasTextureCoords = true;
        }
        else
        {
          this->Internals->HasTextureCoords = false;
        }
      }

      // Map scalars to colors, mirroring vtkOpenGLPolyDataMapper2D::UpdateVBO.
      // this->Colors holds one RGBA per point (point data) or per cell (cell
      // data). Cell colors are indexed by VTK cell-id order (verts, lines,
      // polys, strips).
      this->MapScalars(actor->GetProperty()->GetOpacity());
      this->Internals->HaveCellScalars = false;
      if (this->ScalarVisibility)
      {
        if ((this->ScalarMode == VTK_SCALAR_MODE_USE_CELL_DATA ||
              this->ScalarMode == VTK_SCALAR_MODE_USE_CELL_FIELD_DATA ||
              this->ScalarMode == VTK_SCALAR_MODE_USE_FIELD_DATA ||
              !input->GetPointData()->GetScalars()) &&
          this->ScalarMode != VTK_SCALAR_MODE_USE_POINT_FIELD_DATA && this->Colors)
        {
          this->Internals->HaveCellScalars = true;
        }
      }

      // Per-vertex colors (point data) live in a float4 buffer indexed by vertex
      // id (vertex buffer index 2). A buffer is always uploaded (white when no
      // vertex colors are active) so the pipeline's color attribute never reads
      // out of bounds.
      {
        const vtkIdType numPts = this->Internals->VertexCount;
        std::vector<float> colors(numPts * 4, 1.0f);
        if (this->Colors && !this->Internals->HaveCellScalars)
        {
          const vtkIdType nColors = std::min(this->Colors->GetNumberOfTuples(), numPts);
          const int nComp = this->Colors->GetNumberOfComponents();
          for (vtkIdType i = 0; i < nColors; i++)
          {
            const unsigned char* c = this->Colors->GetPointer(i * nComp);
            colors[i * 4 + 0] = static_cast<float>(c[0]) / 255.0f;
            colors[i * 4 + 1] = static_cast<float>(nComp > 1 ? c[1] : c[0]) / 255.0f;
            colors[i * 4 + 2] = static_cast<float>(nComp > 2 ? c[2] : c[0]) / 255.0f;
            colors[i * 4 + 3] = static_cast<float>(nComp > 3 ? c[3] : 255) / 255.0f;
          }
        }
        id<MTLBuffer> colorBuffer = [device
          newBufferWithBytes:colors.data()
                     length:colors.size() * sizeof(float)
                    options:MTLResourceStorageModeShared];
        vtkMetalMRC::AssignConsumed(this->Internals->ColorBuffer, colorBuffer);
      }

      // Build index buffers from cell arrays
      std::vector<uint32_t> triIndices;
      std::vector<uint32_t> lineIndices;
      std::vector<uint32_t> pointIndices;
      std::vector<float> triCellColors;
      std::vector<float> lineCellColors;
      std::vector<float> pointCellColors;

      vtkCellArray* polys = input->GetPolys();
      vtkCellArray* lines = input->GetLines();
      vtkCellArray* verts = input->GetVerts();
      vtkCellArray* strips = input->GetStrips();

      const bool useCellColors = this->Internals->HaveCellScalars;
      unsigned char* cellColorData = useCellColors ? this->Colors->GetPointer(0) : nullptr;
      const int cellColorComp = useCellColors ? this->Colors->GetNumberOfComponents() : 0;
      auto pushCellColor = [&](std::vector<float>& buf, vtkIdType cellId) {
        float r = 1.0f, g = 1.0f, b = 1.0f, a = 1.0f;
        if (cellColorData && cellColorComp >= 3)
        {
          r = static_cast<float>(cellColorData[cellId * cellColorComp + 0]) / 255.0f;
          g = static_cast<float>(cellColorData[cellId * cellColorComp + 1]) / 255.0f;
          b = static_cast<float>(cellColorData[cellId * cellColorComp + 2]) / 255.0f;
          if (cellColorComp >= 4)
          {
            a = static_cast<float>(cellColorData[cellId * cellColorComp + 3]) / 255.0f;
          }
        }
        buf.push_back(r);
        buf.push_back(g);
        buf.push_back(b);
        buf.push_back(a);
      };

      // VTK numbers cell ids in order: verts, lines, polys, strips.
      const vtkIdType nVertsCells = verts ? verts->GetNumberOfCells() : 0;
      const vtkIdType nLineCells = lines ? lines->GetNumberOfCells() : 0;
      const vtkIdType nPolyCells = polys ? polys->GetNumberOfCells() : 0;
      vtkIdType cellCounter = 0;

      if (polys)
      {
        const vtkIdType* ids = nullptr;
        vtkIdType npts = 0;
        polys->InitTraversal();
        while (polys->GetNextCell(npts, ids))
        {
          const vtkIdType cellId = nVertsCells + nLineCells + cellCounter;
          if (npts >= 3)
          {
            for (vtkIdType i = 1; i < npts - 1; ++i)
            {
              triIndices.push_back(static_cast<uint32_t>(ids[0]));
              triIndices.push_back(static_cast<uint32_t>(ids[i]));
              triIndices.push_back(static_cast<uint32_t>(ids[i + 1]));
              if (useCellColors)
              {
                pushCellColor(triCellColors, cellId);
              }
            }
          }
          ++cellCounter;
        }
      }

      // P11-11A: Triangle strips — decompose with GL's winding
      // (tri_j = v_j, v_{j+1+j%2}, v_{j+1+(j+1)%2}), same as the 3D mapper.
      cellCounter = 0;
      if (strips)
      {
        const vtkIdType* ids = nullptr;
        vtkIdType npts = 0;
        strips->InitTraversal();
        while (strips->GetNextCell(npts, ids))
        {
          const vtkIdType cellId = nVertsCells + nLineCells + nPolyCells + cellCounter;
          if (npts >= 3)
          {
            for (vtkIdType j = 0; j < npts - 2; ++j)
            {
              triIndices.push_back(static_cast<uint32_t>(ids[j]));
              triIndices.push_back(static_cast<uint32_t>(ids[j + 1 + j % 2]));
              triIndices.push_back(static_cast<uint32_t>(ids[j + 1 + (j + 1) % 2]));
              if (useCellColors)
              {
                pushCellColor(triCellColors, cellId);
              }
            }
          }
          ++cellCounter;
        }
      }

      cellCounter = 0;
      if (lines)
      {
        const vtkIdType* ids = nullptr;
        vtkIdType npts = 0;
        lines->InitTraversal();
        while (lines->GetNextCell(npts, ids))
        {
          const vtkIdType cellId = nVertsCells + cellCounter;
          if (npts >= 2)
          {
            for (vtkIdType i = 0; i < npts - 1; ++i)
            {
              lineIndices.push_back(static_cast<uint32_t>(ids[i]));
              lineIndices.push_back(static_cast<uint32_t>(ids[i + 1]));
              if (useCellColors)
              {
                pushCellColor(lineCellColors, cellId);
              }
            }
          }
          ++cellCounter;
        }
      }

      cellCounter = 0;
      if (verts)
      {
        const vtkIdType* ids = nullptr;
        vtkIdType npts = 0;
        verts->InitTraversal();
        while (verts->GetNextCell(npts, ids))
        {
          const vtkIdType cellId = cellCounter;
          for (vtkIdType i = 0; i < npts; ++i)
          {
            pointIndices.push_back(static_cast<uint32_t>(ids[i]));
            if (useCellColors)
            {
              pushCellColor(pointCellColors, cellId);
            }
          }
          ++cellCounter;
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

      // Per-primitive cell color buffers (one float4 per primitive, matched to
      // the primitive-id order of each draw).
      if (useCellColors)
      {
        if (!triCellColors.empty())
        {
          id<MTLBuffer> buffer =
            [device newBufferWithBytes:triCellColors.data()
                                length:triCellColors.size() * sizeof(float)
                               options:MTLResourceStorageModeShared];
          vtkMetalMRC::AssignConsumed(this->Internals->TriangleCellColorBuffer, buffer);
        }
        if (!lineCellColors.empty())
        {
          id<MTLBuffer> buffer =
            [device newBufferWithBytes:lineCellColors.data()
                                length:lineCellColors.size() * sizeof(float)
                               options:MTLResourceStorageModeShared];
          vtkMetalMRC::AssignConsumed(this->Internals->LineCellColorBuffer, buffer);
        }
        if (!pointCellColors.empty())
        {
          id<MTLBuffer> buffer =
            [device newBufferWithBytes:pointCellColors.data()
                                length:pointCellColors.size() * sizeof(float)
                               options:MTLResourceStorageModeShared];
          vtkMetalMRC::AssignConsumed(this->Internals->PointCellColorBuffer, buffer);
        }
      }
    }

    // Compute WCVC (world-to-viewport-clip) matrix
    // For 2D overlay: orthographic projection from viewport pixel coordinates.
    // Mirrors vtkOpenGLPolyDataMapper2D::SetTransformShaderParameters: the
    // polydata points are in actor-local display coordinates, and the actor's
    // viewport-pixel position offsets the ortho range so local coords land at
    // the correct screen location. The renderer's own pixel size (GetSize) is
    // used as the range, so sub-viewport renderers map correctly too.
    if (size[0] <= 0 || size[1] <= 0)
    {
      return;
    }

    int* actorPos = actor->GetPositionCoordinate()->GetComputedViewportValue(viewport);
    float xoff = static_cast<float>(actorPos[0]);
    float yoff = static_cast<float>(actorPos[1]);

    // Build orthographic matrix: maps actor-local coords + position offset to
    // NDC [-1,1]. Metal NDC: x in [-1,1], y in [-1,1], z in [0,1].

    float worldLeft = -xoff;
    float worldRight = static_cast<float>(size[0]) - xoff;
    float worldBottom = -yoff;
    float worldTop = static_cast<float>(size[1]) - yoff;

    // Standard orthographic matrix. The 2D overlay renders with depth test
    // enabled (LessEqual, depth write on), so the Z row must encode the prop's
    // display location like vtkOpenGLPolyDataMapper2D::SetCameraShaderParameters:
    // foreground props at depth 0 (near), background props at depth 1 (far).
    // Metal's clip/NDC z range is [0,1], so the Z translate is 0 for
    // foreground and 1 for background. This keeps background overlays from
    // overwriting foreground text/geometry while letting foreground overlays
    // (later in render order) win at equal depth.
    float wcvc[16] = { 0 };
    wcvc[0] = 2.0f / (worldRight - worldLeft);
    wcvc[5] = 2.0f / (worldTop - worldBottom);
    wcvc[10] = 0.0f;
    wcvc[12] = -(worldRight + worldLeft) / (worldRight - worldLeft);
    wcvc[13] = -(worldTop + worldBottom) / (worldTop - worldBottom);
    wcvc[14] =
      actor->GetProperty()->GetDisplayLocation() == VTK_FOREGROUND_LOCATION ? 0.0f : 1.0f;
    wcvc[15] = 1.0f;

    // NOTE: no Y flip here. Metal's NDC maps +y to the top of the framebuffer,
    // and the VTK viewport origin is the bottom-left with y growing up, so the
    // identity y mapping (bottom → NDC -1, top → NDC +1) lands 2D geometry at
    // the same screen location as the 3D passes (whose camera already emits
    // Metal NDC). The color read-back additionally re-orders rows bottom-up,
    // so the regression PNG matches the on-screen image.

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
    state.flags = this->Internals->HaveCellScalars
      ? kFlagUseCellColors
      : (this->Colors && this->ScalarVisibility ? kFlagUseVertexColors : 0u);
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

    // Property color is passed via StateBuffer at index 1 (vertex) / 0
    // (fragment). Scalar colors override it: per-vertex colors arrive through
    // the vertex buffer at index 2, per-cell colors through the fragment
    // buffer at index 1 (indexed by [[primitive_id]]).

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

      // Per-vertex color attribute (float4, buffer index 2). A buffer is always
      // bound at index 2 so this attribute never reads out of bounds; the
      // shader only consumes it when kFlagUseVertexColors is set.
      vertexDesc.attributes[2].format = MTLVertexFormatFloat4;
      vertexDesc.attributes[2].offset = 0;
      vertexDesc.attributes[2].bufferIndex = 2;
      vertexDesc.layouts[2].stride = sizeof(float) * 4;
      vertexDesc.layouts[2].stepRate = 1;
      vertexDesc.layouts[2].stepFunction = MTLVertexStepFunctionPerVertex;

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

    // Create the textured pipeline (position+texcoord interleaved vertex
    // buffer, texture sampled in the fragment shader) if needed.
    if (!this->Internals->TexturedPipeline)
    {
      NSError* error = nil;
      id<MTLLibrary> library = (__bridge id<MTLLibrary>)renWin->GetSharedShaderLibrary();
      if (!library)
      {
        vtkErrorMacro(<< "No shared shader library available for 2D textured mapper");
        return;
      }

      id<MTLFunction> tvFunc = [library newFunctionWithName:@"vertex_2d_image_main"];
      id<MTLFunction> tfFunc = [library newFunctionWithName:@"fragment_2d_text_main"];
      if (!tvFunc || !tfFunc)
      {
        vtkErrorMacro(<< "Failed to find 2D textured shader functions");
        [tvFunc release];
        [tfFunc release];
        return;
      }

      MTLVertexDescriptor* texVertexDesc = [[MTLVertexDescriptor alloc] init];
      texVertexDesc.attributes[0].format = MTLVertexFormatFloat2;
      texVertexDesc.attributes[0].offset = 0;
      texVertexDesc.attributes[0].bufferIndex = 0;
      texVertexDesc.attributes[1].format = MTLVertexFormatFloat2;
      texVertexDesc.attributes[1].offset = sizeof(float) * 2;
      texVertexDesc.attributes[1].bufferIndex = 0;
      texVertexDesc.layouts[0].stride = sizeof(float) * 4;
      texVertexDesc.layouts[0].stepRate = 1;
      texVertexDesc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;

      MTLRenderPipelineDescriptor* tdesc = [[MTLRenderPipelineDescriptor alloc] init];
      tdesc.vertexFunction = tvFunc;
      tdesc.fragmentFunction = tfFunc;
      tdesc.vertexDescriptor = texVertexDesc;
      tdesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
      tdesc.depthAttachmentPixelFormat = depthFormat;
      tdesc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
      tdesc.rasterSampleCount = sampleCount;
      tdesc.colorAttachments[0].blendingEnabled = YES;
      tdesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
      tdesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      tdesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
      tdesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

      this->Internals->TexturedPipeline =
        [device newRenderPipelineStateWithDescriptor:tdesc error:&error];
      if (!this->Internals->TexturedPipeline)
      {
        vtkErrorMacro(<< "2D textured pipeline: " << [[error localizedDescription] UTF8String]);
      }
      [tdesc release];
      [texVertexDesc release];
      [tvFunc release];
      [tfFunc release];
    }

    // Create and bind overlay depth-stencil state (LessEqual, write on),
    // matching the OpenGL overlay depth state: the Z row of the wcvc matrix
    // encodes foreground (near) vs background (far), and equal-depth foreground
    // props resolve by render order.
    if (!this->Internals->OverlayDepthState)
    {
      MTLDepthStencilDescriptor* dsDesc = [[MTLDepthStencilDescriptor alloc] init];
      dsDesc.depthCompareFunction = MTLCompareFunctionLessEqual;
      dsDesc.depthWriteEnabled = YES;
      this->Internals->OverlayDepthState =
          [device newDepthStencilStateWithDescriptor:dsDesc];
      [dsDesc release];
    }
    [encoder setDepthStencilState:this->Internals->OverlayDepthState];

    // Set state buffer at buffer index 1
    [encoder setVertexBuffer:this->Internals->StateBuffer offset:0 atIndex:1];
    [encoder setFragmentBuffer:this->Internals->StateBuffer offset:0 atIndex:0];

    // Bind the per-vertex color buffer (white when unused) for the plain 2D
    // pipelines' color attribute at buffer index 2.
    [encoder setVertexBuffer:this->Internals->ColorBuffer offset:0 atIndex:2];

    // Textured geometry (e.g. vtkTextMapper / vtkTextActor): resolve the
    // texture registered for the actor's GENERAL_TEXTURE_UNIT and draw the
    // interleaved position+texcoord buffer, sampling the texture in the
    // fragment shader (multiplied by the actor's color/opacity). Mirrors the
    // OpenGL 2D mapper's "texture1" path.
    vtkInformation* info = actor->GetPropertyKeys();
    id<MTLTexture> boundTex = nil;
    bool texturedDraw = false;
    if (this->Internals->HasTextureCoords && this->Internals->TexturedPipeline &&
      this->Internals->TexCoordBuffer && this->Internals->HasTriangles && info &&
      info->Has(vtkProp::GENERAL_TEXTURE_UNIT()))
    {
      int tunit = info->Get(vtkProp::GENERAL_TEXTURE_UNIT());
      boundTex = (__bridge id<MTLTexture>)renWin->GetBoundTexture(tunit);
      texturedDraw = (boundTex != nil);
    }

    if (texturedDraw && this->Internals->TriangleIndexBuffer &&
      this->Internals->TriangleIndexCount > 0)
    {
      [encoder setRenderPipelineState:this->Internals->TexturedPipeline];
      [encoder setVertexBuffer:this->Internals->TexCoordBuffer offset:0 atIndex:0];
      [encoder setFragmentTexture:boundTex atIndex:0];

      [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                          indexCount:this->Internals->TriangleIndexCount
                           indexType:MTLIndexTypeUInt32
                         indexBuffer:this->Internals->TriangleIndexBuffer
                   indexBufferOffset:0];
    }

    // Set position buffer at buffer index 0
    [encoder setVertexBuffer:this->Internals->PositionBuffer offset:0 atIndex:0];

    // Draw triangles using index buffer
    if (!texturedDraw && this->Internals->HasTriangles &&
        this->Internals->TrianglePipeline &&
        this->Internals->TriangleIndexBuffer &&
        this->Internals->TriangleIndexCount > 0)
    {
      [encoder setRenderPipelineState:this->Internals->TrianglePipeline];
      [encoder setFragmentBuffer:(this->Internals->TriangleCellColorBuffer
            ? (id<MTLBuffer>)this->Internals->TriangleCellColorBuffer
            : (id<MTLBuffer>)this->Internals->ColorBuffer)
        offset:0
        atIndex:1];

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
      [encoder setFragmentBuffer:(this->Internals->LineCellColorBuffer
            ? (id<MTLBuffer>)this->Internals->LineCellColorBuffer
            : (id<MTLBuffer>)this->Internals->ColorBuffer)
        offset:0
        atIndex:1];

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
      [encoder setFragmentBuffer:(this->Internals->PointCellColorBuffer
            ? (id<MTLBuffer>)this->Internals->PointCellColorBuffer
            : (id<MTLBuffer>)this->Internals->ColorBuffer)
        offset:0
        atIndex:1];

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
