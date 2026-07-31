// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalGlyph3DMapper.h"

#include "vtkMetalRenderWindow.h"
#include "vtkMetalRenderer.h"
#include "vtkMetalHardwareSelector.h"
#include "vtkMetalPickTypes.h"
#include "vtkMetalCamera.h"
#include "vtkMetalShaders.h"
#include "vtkObjectFactory.h"
#include "vtkPolyData.h"
#include "vtkCellArray.h"
#include "vtkPointData.h"
#include "vtkProperty.h"
#include "vtkActor.h"
#include "vtkRenderer.h"
#include "vtkMatrix4x4.h"
#include "vtkMatrix3x3.h"
#include "vtkQuaternion.h"
#include "vtkMath.h"
#include "vtkOverrideAttribute.h"
#include "vtkFloatArray.h"
#include "vtkUnsignedCharArray.h"
#include "vtkBitArray.h"
#include "vtkCollection.h"
#include "vtkLight.h"
#include "vtkLightCollection.h"
#include "vtkNew.h"

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstring>

// Shared GPU/CPU picking ID layout (see vtkMetalPickTypes.h). Written into
// PropIdBuffer before each glyph draw; consumed by vertex shaders as
// buffer(10).
using PickIds = vtkMetalPickIds;

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalGlyph3DMapper);

// ---------------------------------------------------------------------------
// Metal vertex buffer slot assignments for glyph instanced rendering
// Slots 0-1: per-vertex source geometry
// Slots 2-5: per-instance glyph attributes (stepFunctionPerInstance)
// Slots 8-10: uniform buffers (shared vertex+fragment)
// ---------------------------------------------------------------------------
enum GlyphBufferSlot : int
{
  kGlyphSrcPosition = 0,      // float3 per vertex
  kGlyphSrcNormal = 1,        // float3 per vertex
  kGlyphTransform = 2,        // float4x4 per instance (4 x vec4 columns)
  kGlyphNormalTransform = 3,  // float3x3 per instance (3 x vec3 columns)
  kGlyphInstanceColor = 4,    // float4 RGBA per instance
  kGlyphPickId = 5,           // uint32 per instance
  kGlyphScene = 8,            // SceneUniforms
  kGlyphClipPlane = 9,        // ClipPlaneUniforms
  kGlyphPropId = 10,          // PickIds {propId, compositeIndex} (constant)
};

// ---------------------------------------------------------------------------
// Per-instance glyph data computed on CPU and uploaded to GPU
// ---------------------------------------------------------------------------
struct vtkMetalGlyph3DMapper::vtkMetalGlyph3DMapperInternals
{
  // Source geometry (cached from glyph source polydata)
  id<MTLBuffer> SrcPositionBuffer = nil;
  id<MTLBuffer> SrcNormalBuffer = nil;
  vtkIdType SrcVertexCount = 0;
  vtkIdType TriVertexCount = 0;
  vtkIdType LineVertexCount = 0;
  vtkIdType PtVertexCount = 0;
  bool HasTriangles = false;
  bool HasLines = false;
  bool HasPoints = false;
  vtkIdType CachedSourceMTime = 0;

  // Per-instance attribute buffers
  id<MTLBuffer> InstTransformBuffer = nil;
  id<MTLBuffer> InstNormalTransformBuffer = nil;
  id<MTLBuffer> InstColorBuffer = nil;
  id<MTLBuffer> InstPickIdBuffer = nil;
  uint32_t NumInstances = 0;
  vtkIdType CachedInputMTime = 0;

  // Pipeline states
  id<MTLRenderPipelineState> TriPipeline = nil;
  id<MTLRenderPipelineState> LinePipeline = nil;
  id<MTLRenderPipelineState> PtPipeline = nil;

  // Uniform buffers
  id<MTLBuffer> SceneBuffer = nil;
  id<MTLBuffer> MaterialBuffer = nil;
  id<MTLBuffer> LightBuffer = nil;
  id<MTLBuffer> CoincidentBuffer = nil;
  id<MTLBuffer> ClipPlaneBuffer = nil;
  id<MTLBuffer> PropIdBuffer = nil;

  int CachedSampleCount = 0;

  void ReleaseSourceBuffers()
  {
    [SrcPositionBuffer release];
    SrcPositionBuffer = nil;
    [SrcNormalBuffer release];
    SrcNormalBuffer = nil;
    SrcVertexCount = 0;
    TriVertexCount = LineVertexCount = PtVertexCount = 0;
    HasTriangles = HasLines = HasPoints = false;
    CachedSourceMTime = 0;
  }

  void ReleaseInstanceBuffers()
  {
    [InstTransformBuffer release];
    InstTransformBuffer = nil;
    [InstNormalTransformBuffer release];
    InstNormalTransformBuffer = nil;
    [InstColorBuffer release];
    InstColorBuffer = nil;
    [InstPickIdBuffer release];
    InstPickIdBuffer = nil;
    NumInstances = 0;
  }

  void ReleaseAll()
  {
    ReleaseSourceBuffers();
    ReleaseInstanceBuffers();
    [TriPipeline release];
    TriPipeline = nil;
    [LinePipeline release];
    LinePipeline = nil;
    [PtPipeline release];
    PtPipeline = nil;
    [SceneBuffer release];
    SceneBuffer = nil;
    [MaterialBuffer release];
    MaterialBuffer = nil;
    [LightBuffer release];
    LightBuffer = nil;
    [CoincidentBuffer release];
    CoincidentBuffer = nil;
    [ClipPlaneBuffer release];
    ClipPlaneBuffer = nil;
    [PropIdBuffer release];
    PropIdBuffer = nil;
    CachedSampleCount = 0;
  }
};

//------------------------------------------------------------------------------
vtkMetalGlyph3DMapper::vtkMetalGlyph3DMapper()
  : Internals(new vtkMetalGlyph3DMapperInternals())
{
}

//------------------------------------------------------------------------------
vtkMetalGlyph3DMapper::~vtkMetalGlyph3DMapper() = default;

//------------------------------------------------------------------------------
vtkOverrideAttribute* vtkMetalGlyph3DMapper::CreateOverrideAttributes()
{
  return vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "Metal", nullptr);
}

//------------------------------------------------------------------------------
void vtkMetalGlyph3DMapper::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//------------------------------------------------------------------------------
void vtkMetalGlyph3DMapper::ReleaseGraphicsResources(vtkWindow*)
{
  this->Internals->ReleaseAll();
}

// ---------------------------------------------------------------------------
// Build Metal buffers from glyph source polydata geometry.
// Extracts triangles (fan-triangulated polygons), line segments, and points.
// ---------------------------------------------------------------------------
static void BuildSourceGeometry(id<MTLDevice> device, vtkPolyData* src,
  vtkMetalGlyph3DMapper::vtkMetalGlyph3DMapperInternals* I)
{
  I->ReleaseSourceBuffers();
  if (!src || src->GetNumberOfPoints() == 0)
  {
    return;
  }

  vtkPointData* pd = src->GetPointData();
  vtkDataArray* normals = pd ? pd->GetNormals() : nullptr;

  std::vector<float> triPos, triNrm, linePos, lineNrm, ptPos, ptNrm;

  // Polygons → triangles via fan triangulation
  vtkCellArray* polys = src->GetPolys();
  if (polys)
  {
    const vtkIdType* pts = nullptr;
    vtkIdType npts = 0;
    for (polys->InitTraversal(); polys->GetNextCell(npts, pts);)
    {
      for (vtkIdType i = 1; i + 1 < npts; ++i)
      {
        vtkIdType tri[3] = { pts[0], pts[i], pts[i + 1] };
        for (int j = 0; j < 3; ++j)
        {
          double pt[3];
          src->GetPoint(tri[j], pt);
          triPos.push_back(static_cast<float>(pt[0]));
          triPos.push_back(static_cast<float>(pt[1]));
          triPos.push_back(static_cast<float>(pt[2]));
          if (normals)
          {
            double n[3];
            normals->GetTuple(tri[j], n);
            triNrm.push_back(static_cast<float>(n[0]));
            triNrm.push_back(static_cast<float>(n[1]));
            triNrm.push_back(static_cast<float>(n[2]));
          }
          else
          {
            triNrm.push_back(0.f);
            triNrm.push_back(1.f);
            triNrm.push_back(0.f);
          }
        }
      }
    }
  }

  // Lines
  vtkCellArray* lines = src->GetLines();
  if (lines)
  {
    const vtkIdType* pts = nullptr;
    vtkIdType npts = 0;
    for (lines->InitTraversal(); lines->GetNextCell(npts, pts);)
    {
      for (vtkIdType i = 0; i + 1 < npts; ++i)
      {
        for (vtkIdType j = i; j <= i + 1; ++j)
        {
          double pt[3];
          src->GetPoint(pts[j], pt);
          linePos.push_back(static_cast<float>(pt[0]));
          linePos.push_back(static_cast<float>(pt[1]));
          linePos.push_back(static_cast<float>(pt[2]));
          if (normals)
          {
            double n[3];
            normals->GetTuple(pts[j], n);
            lineNrm.push_back(static_cast<float>(n[0]));
            lineNrm.push_back(static_cast<float>(n[1]));
            lineNrm.push_back(static_cast<float>(n[2]));
          }
          else
          {
            lineNrm.push_back(0.f);
            lineNrm.push_back(1.f);
            lineNrm.push_back(0.f);
          }
        }
      }
    }
  }

  // Verts
  vtkCellArray* verts = src->GetVerts();
  if (verts)
  {
    const vtkIdType* pts = nullptr;
    vtkIdType npts = 0;
    for (verts->InitTraversal(); verts->GetNextCell(npts, pts);)
    {
      for (vtkIdType i = 0; i < npts; ++i)
      {
        double pt[3];
        src->GetPoint(pts[i], pt);
        ptPos.push_back(static_cast<float>(pt[0]));
        ptPos.push_back(static_cast<float>(pt[1]));
        ptPos.push_back(static_cast<float>(pt[2]));
        if (normals)
        {
          double n[3];
          normals->GetTuple(pts[i], n);
          ptNrm.push_back(static_cast<float>(n[0]));
          ptNrm.push_back(static_cast<float>(n[1]));
          ptNrm.push_back(static_cast<float>(n[2]));
        }
        else
        {
          ptNrm.push_back(0.f);
          ptNrm.push_back(1.f);
          ptNrm.push_back(0.f);
        }
      }
    }
  }

  // If no geometry, use points as fallback
  if (triPos.empty() && linePos.empty() && ptPos.empty())
  {
    vtkIdType n = src->GetNumberOfPoints();
    for (vtkIdType i = 0; i < n; ++i)
    {
      double pt[3];
      src->GetPoint(i, pt);
      ptPos.push_back(static_cast<float>(pt[0]));
      ptPos.push_back(static_cast<float>(pt[1]));
      ptPos.push_back(static_cast<float>(pt[2]));
      ptNrm.push_back(0.f);
      ptNrm.push_back(1.f);
      ptNrm.push_back(0.f);
    }
  }

  // Prefer triangles, then lines, then points for the shared source buffer
  const std::vector<float>* usePos = nullptr;
  const std::vector<float>* useNrm = nullptr;
  if (!triPos.empty())
  {
    usePos = &triPos;
    useNrm = &triNrm;
    I->HasTriangles = true;
    I->TriVertexCount = triPos.size() / 3;
  }
  else if (!linePos.empty())
  {
    usePos = &linePos;
    useNrm = &lineNrm;
    I->HasLines = true;
    I->LineVertexCount = linePos.size() / 3;
  }
  else
  {
    usePos = &ptPos;
    useNrm = &ptNrm;
    I->HasPoints = true;
    I->PtVertexCount = ptPos.size() / 3;
  }

  I->SrcPositionBuffer = [device newBufferWithBytes:usePos->data()
                                             length:usePos->size() * sizeof(float)
                                            options:MTLResourceStorageModeShared];
  I->SrcNormalBuffer = [device newBufferWithBytes:useNrm->data()
                                           length:useNrm->size() * sizeof(float)
                                          options:MTLResourceStorageModeShared];
  I->SrcVertexCount = usePos->size() / 3;
  I->CachedSourceMTime = src->GetMTime();
}

// ---------------------------------------------------------------------------
// Ensure pipeline states for glyph rendering (lazy creation).
// ---------------------------------------------------------------------------
static void EnsureGlyphPipelines(
  vtkMetalGlyph3DMapper::vtkMetalGlyph3DMapperInternals* I, id<MTLDevice> device, int sampleCount)
{
  if (I->TriPipeline && I->LinePipeline && I->PtPipeline)
  {
    return;
  }

  NSError* error = nil;
  NSString* src = [NSString stringWithUTF8String:vtkMetalShaders];
  id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&error];
  if (!lib)
  {
    return;
  }

  auto makeVertDesc = []() -> MTLVertexDescriptor*
  {
    MTLVertexDescriptor* vd = [[MTLVertexDescriptor alloc] init];

    // Per-vertex: position (float3)
    vd.attributes[0].format = MTLVertexFormatFloat3;
    vd.attributes[0].offset = 0;
    vd.attributes[0].bufferIndex = kGlyphSrcPosition;
    vd.layouts[kGlyphSrcPosition].stride = sizeof(float) * 3;
    vd.layouts[kGlyphSrcPosition].stepRate = 1;
    vd.layouts[kGlyphSrcPosition].stepFunction = MTLVertexStepFunctionPerVertex;

    // Per-vertex: normal (float3)
    vd.attributes[1].format = MTLVertexFormatFloat3;
    vd.attributes[1].offset = 0;
    vd.attributes[1].bufferIndex = kGlyphSrcNormal;
    vd.layouts[kGlyphSrcNormal].stride = sizeof(float) * 3;
    vd.layouts[kGlyphSrcNormal].stepRate = 1;
    vd.layouts[kGlyphSrcNormal].stepFunction = MTLVertexStepFunctionPerVertex;

    // Per-instance: 4x4 transform as 4 x float4 columns
    for (int i = 0; i < 4; ++i)
    {
      vd.attributes[2 + i].format = MTLVertexFormatFloat4;
      vd.attributes[2 + i].offset = i * (int)sizeof(float) * 4;
      vd.attributes[2 + i].bufferIndex = kGlyphTransform;
    }
    vd.layouts[kGlyphTransform].stride = sizeof(float) * 16;
    vd.layouts[kGlyphTransform].stepRate = 1;
    vd.layouts[kGlyphTransform].stepFunction = MTLVertexStepFunctionPerInstance;

    // Per-instance: 3x3 normal transform as 3 x float3 columns
    for (int i = 0; i < 3; ++i)
    {
      vd.attributes[6 + i].format = MTLVertexFormatFloat3;
      vd.attributes[6 + i].offset = i * (int)sizeof(float) * 3;
      vd.attributes[6 + i].bufferIndex = kGlyphNormalTransform;
    }
    vd.layouts[kGlyphNormalTransform].stride = sizeof(float) * 9;
    vd.layouts[kGlyphNormalTransform].stepRate = 1;
    vd.layouts[kGlyphNormalTransform].stepFunction = MTLVertexStepFunctionPerInstance;

    // Per-instance: color (float4)
    vd.attributes[9].format = MTLVertexFormatFloat4;
    vd.attributes[9].offset = 0;
    vd.attributes[9].bufferIndex = kGlyphInstanceColor;
    vd.layouts[kGlyphInstanceColor].stride = sizeof(float) * 4;
    vd.layouts[kGlyphInstanceColor].stepRate = 1;
    vd.layouts[kGlyphInstanceColor].stepFunction = MTLVertexStepFunctionPerInstance;

    // Per-instance: pick ID (uint32)
    vd.attributes[10].format = MTLVertexFormatUInt;
    vd.attributes[10].offset = 0;
    vd.attributes[10].bufferIndex = kGlyphPickId;
    vd.layouts[kGlyphPickId].stride = sizeof(uint32_t);
    vd.layouts[kGlyphPickId].stepRate = 1;
    vd.layouts[kGlyphPickId].stepFunction = MTLVertexStepFunctionPerInstance;

    return vd;
  };

  auto makePipeline = [&](NSString* vName, NSString* fName,
                           MTLPrimitiveTopologyClass topo) -> id<MTLRenderPipelineState>
  {
    id<MTLFunction> vf = [lib newFunctionWithName:vName];
    id<MTLFunction> ff = [lib newFunctionWithName:fName];
    if (!vf || !ff)
    {
      [vf release];
      [ff release];
      return nil;
    }

    MTLRenderPipelineDescriptor* d = [[MTLRenderPipelineDescriptor alloc] init];
    d.vertexFunction = vf;
    d.fragmentFunction = ff;
    d.vertexDescriptor = makeVertDesc();
    d.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    if (sampleCount <= 1)
    {
      d.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;
    }
    d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    d.inputPrimitiveTopology = topo;
    d.rasterSampleCount = sampleCount;

    error = nil;
    id<MTLRenderPipelineState> ps = [device newRenderPipelineStateWithDescriptor:d error:&error];
    if (!ps)
    {
      vtkGenericWarningMacro(<< "Glyph pipeline error: "
                             << [[error localizedDescription] UTF8String]);
    }
    [d release];
    [vf release];
    [ff release];
    return ps;
  };

  if (!I->TriPipeline)
  {
    I->TriPipeline = makePipeline(@"vertex_glyph_main", @"fragment_glyph_main",
                                  MTLPrimitiveTopologyClassTriangle);
  }
  if (!I->LinePipeline)
  {
    I->LinePipeline = makePipeline(@"vertex_glyph_line_main", @"fragment_glyph_line_main",
                                   MTLPrimitiveTopologyClassLine);
  }
  if (!I->PtPipeline)
  {
    I->PtPipeline = makePipeline(@"vertex_glyph_point_main", @"fragment_glyph_point_main",
                                 MTLPrimitiveTopologyClassPoint);
  }

  [lib release];
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------
void vtkMetalGlyph3DMapper::Render(vtkRenderer* ren, vtkActor* actor)
{
  auto* I = this->Internals.get();

  auto* inputDataObject = this->GetInputDataObject(0, 0);
  if (!inputDataObject)
  {
    return;
  }

  // Create default source if none set
  if (!this->UseSourceTableTree && this->GetSource(0) == nullptr)
  {
    vtkNew<vtkPolyData> defSrc;
    vtkNew<vtkPoints> pts;
    pts->InsertNextPoint(0, 0, 0);
    pts->InsertNextPoint(1, 0, 0);
    vtkNew<vtkCellArray> ln;
    ln->InsertNextCell({ 0, 1 });
    defSrc->SetLines(ln);
    this->SetSourceData(defSrc);
  }

  vtkPolyData* source = vtkPolyData::SafeDownCast(this->GetSource(0));
  if (!source || source->GetNumberOfPoints() == 0)
  {
    return;
  }

  vtkMetalRenderWindow* renWin = vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());
  if (!renWin || !renWin->GetMetalDevice())
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)renWin->GetMetalDevice();

  // MSAA invalidation
  int sampleCount = renWin->GetEffectiveSampleCount();
  if (sampleCount != I->CachedSampleCount)
  {
    I->TriPipeline = nil;
    I->LinePipeline = nil;
    I->PtPipeline = nil;
    I->CachedSampleCount = sampleCount;
  }

  // Rebuild source geometry if source changed
  if (source->GetMTime() != I->CachedSourceMTime)
  {
    BuildSourceGeometry(device, source, I);
  }
  if (I->SrcVertexCount == 0)
  {
    return;
  }

  // Get input dataset
  vtkDataSet* ds = vtkDataSet::SafeDownCast(inputDataObject);
  if (!ds)
  {
    return;
  }

  const vtkIdType numPoints = ds->GetNumberOfPoints();
  if (numPoints < 1)
  {
    return;
  }

  // Rebuild instance data if input changed
  vtkIdType inputMTime = ds->GetMTime();
  if (inputMTime != I->CachedInputMTime)
  {
    I->ReleaseInstanceBuffers();
    I->CachedInputMTime = inputMTime;

    auto* orientArr = this->GetOrientationArray(ds);
    auto* scaleArr = this->GetScaleArray(ds);
    auto* selArr = this->GetSelectionIdArray(ds);
    vtkBitArray* maskArr = this->Masking
      ? vtkArrayDownCast<vtkBitArray>(this->GetMaskArray(ds))
      : nullptr;

    // Validate orientation array
    if (orientArr)
    {
      int nc = orientArr->GetNumberOfComponents();
      if ((this->OrientationMode == ROTATION || this->OrientationMode == DIRECTION) && nc != 3)
      {
        vtkErrorMacro("Orientation array must have 3 components, got " << nc);
        return;
      }
      if (this->OrientationMode == QUATERNION && nc != 4)
      {
        vtkErrorMacro("Orientation array must have 4 components, got " << nc);
        return;
      }
    }

    // Map input scalars to colors
    vtkNew<vtkUnsignedCharArray> mappedColors;
    this->MapScalars(actor->GetProperty()->GetOpacity());
    if (this->Colors && this->Colors->GetNumberOfTuples() > 0)
    {
      mappedColors->DeepCopy(this->Colors);
    }

    // Count unmasked points
    vtkIdType instCount = 0;
    for (vtkIdType pid = 0; pid < numPoints; ++pid)
    {
      if (maskArr && maskArr->GetValue(pid) == 0)
        continue;
      ++instCount;
    }
    if (instCount == 0)
      return;

    double rangeSize = this->Range[1] - this->Range[0];
    if (rangeSize == 0.0)
      rangeSize = 1.0;

    double actorCol[4];
    actor->GetProperty()->GetColor(actorCol);
    actorCol[3] = actor->GetProperty()->GetOpacity();

    std::vector<vtkTypeFloat32> transforms(instCount * 16);
    std::vector<vtkTypeFloat32> normTransforms(instCount * 9);
    std::vector<vtkTypeFloat32> colors(instCount * 4);
    std::vector<vtkIdType> pickIds(instCount);

    vtkIdType idx = 0;
    for (vtkIdType pid = 0; pid < numPoints; ++pid)
    {
      if (maskArr && maskArr->GetValue(pid) == 0)
        continue;

      // Color
      float r = static_cast<float>(actorCol[0]);
      float g = static_cast<float>(actorCol[1]);
      float b = static_cast<float>(actorCol[2]);
      float a = static_cast<float>(actorCol[3]);
      if (mappedColors && mappedColors->GetNumberOfTuples() > pid)
      {
        unsigned char rgba[4];
        mappedColors->GetTypedTuple(pid, rgba);
        r = rgba[0] / 255.f;
        g = rgba[1] / 255.f;
        b = rgba[2] / 255.f;
        a = rgba[3] / 255.f;
      }
      colors[idx * 4 + 0] = r;
      colors[idx * 4 + 1] = g;
      colors[idx * 4 + 2] = b;
      colors[idx * 4 + 3] = a;

      // Scale
      double sx = 1.0, sy = 1.0, sz = 1.0;
      if (scaleArr)
      {
        double* t = scaleArr->GetTuple(pid);
        switch (this->ScaleMode)
        {
          case SCALE_BY_MAGNITUDE:
            sx = sy = sz = vtkMath::Norm(t, scaleArr->GetNumberOfComponents());
            break;
          case SCALE_BY_COMPONENTS:
            if (scaleArr->GetNumberOfComponents() == 3)
            {
              sx = t[0];
              sy = t[1];
              sz = t[2];
            }
            break;
          default:
            break;
        }
        if (this->Clamping && this->ScaleMode != NO_DATA_SCALING)
        {
          sx = vtkMath::ClampValue(sx, this->Range[0], this->Range[1]);
          sx = (sx - this->Range[0]) / rangeSize;
          sy = vtkMath::ClampValue(sy, this->Range[0], this->Range[1]);
          sy = (sy - this->Range[0]) / rangeSize;
          sz = vtkMath::ClampValue(sz, this->Range[0], this->Range[1]);
          sz = (sz - this->Range[0]) / rangeSize;
        }
      }
      sx *= this->ScaleFactor;
      sy *= this->ScaleFactor;
      sz *= this->ScaleFactor;

      // Transform (column-major)
      double tf[16];
      double ntf[9];
      vtkMatrix4x4::Identity(tf);
      vtkMatrix3x3::Identity(ntf);

      double pt[3];
      ds->GetPoint(pid, pt);
      tf[3] = pt[0];
      tf[7] = pt[1];
      tf[11] = pt[2];

      if (orientArr)
      {
        double orient[4];
        orientArr->GetTuple(pid, orient);
        double rot[3][3];
        vtkQuaterniond q;

        switch (this->OrientationMode)
        {
          case ROTATION:
          {
            double a1 = vtkMath::RadiansFromDegrees(orient[2]);
            vtkQuaterniond qz(cos(.5 * a1), 0, 0, sin(.5 * a1));
            a1 = vtkMath::RadiansFromDegrees(orient[0]);
            vtkQuaterniond qx(cos(.5 * a1), sin(.5 * a1), 0, 0);
            a1 = vtkMath::RadiansFromDegrees(orient[1]);
            vtkQuaterniond qy(cos(.5 * a1), 0, sin(.5 * a1), 0);
            q = qz * qx * qy;
            break;
          }
          case QUATERNION:
            q.Set(orient);
            break;
          default:
          {
            if (orient[1] == 0.0 && orient[2] == 0.0)
            {
              if (orient[0] < 0)
                q.Set(0.0, 0.0, 1.0, 0.0);
            }
            else
            {
              double vm = vtkMath::Norm(orient);
              double vn[3] = { (orient[0] + vm) / 2.0, orient[1] / 2.0, orient[2] / 2.0 };
              double f = 1.0 / sqrt(vn[0] * vn[0] + vn[1] * vn[1] + vn[2] * vn[2]);
              vn[0] *= f;
              vn[1] *= f;
              vn[2] *= f;
              q.Set(0.0, vn[0], vn[1], vn[2]);
            }
            break;
          }
        }
        q.ToMatrix3x3(rot);
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
          {
            tf[4 * i + j] = rot[i][j];
            ntf[3 * i + j] = rot[j][i];
          }
      }

      // Apply scale
      if (this->Scaling)
      {
        if (sx == 0.0) sx = 1e-10;
        if (sy == 0.0) sy = 1e-10;
        if (sz == 0.0) sz = 1e-10;
        for (int i = 0; i < 3; i++)
        {
          tf[4 * i] *= sx;
          ntf[i] /= sx;
          tf[4 * i + 1] *= sy;
          ntf[i + 3] /= sy;
          tf[4 * i + 2] *= sz;
          ntf[i + 6] /= sz;
        }
      }

      // Transpose to column-major for Metal and copy
      vtkTypeFloat32* m = &transforms[idx * 16];
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
          m[i * 4 + j] = static_cast<vtkTypeFloat32>(tf[j * 4 + i]);

      vtkTypeFloat32* nm = &normTransforms[idx * 9];
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          nm[i * 3 + j] = static_cast<vtkTypeFloat32>(ntf[i * 3 + j]);

      // Pick ID
      vtkIdType selId = pid;
      if (this->UseSelectionIds && selArr && selArr->GetNumberOfTuples() > 0)
      {
        selId = static_cast<vtkIdType>(*selArr->GetTuple(pid));
      }
      pickIds[idx] = selId;
      ++idx;
    }

    assert(idx == instCount);

    // Upload to GPU
    I->InstTransformBuffer = [device newBufferWithBytes:transforms.data()
                                                length:transforms.size() * sizeof(vtkTypeFloat32)
                                               options:MTLResourceStorageModeShared];
    I->InstNormalTransformBuffer = [device newBufferWithBytes:normTransforms.data()
                                                      length:normTransforms.size() * sizeof(vtkTypeFloat32)
                                                     options:MTLResourceStorageModeShared];
    I->InstColorBuffer = [device newBufferWithBytes:colors.data()
                                            length:colors.size() * sizeof(vtkTypeFloat32)
                                           options:MTLResourceStorageModeShared];

    std::vector<uint32_t> pick32(instCount);
    for (uint32_t i = 0; i < static_cast<uint32_t>(instCount); ++i)
      pick32[i] = static_cast<uint32_t>(pickIds[i]);
    I->InstPickIdBuffer = [device newBufferWithBytes:pick32.data()
                                             length:pick32.size() * sizeof(uint32_t)
                                            options:MTLResourceStorageModeShared];
    I->NumInstances = static_cast<uint32_t>(instCount);
  }

  if (I->NumInstances == 0)
  {
    return;
  }

  // Ensure pipelines
  EnsureGlyphPipelines(I, device, sampleCount);

  id<MTLRenderCommandEncoder> enc =
    (__bridge id<MTLRenderCommandEncoder>)renWin->GetCurrentRenderCommandEncoder();
  if (!enc)
  {
    return;
  }

  // Camera + scene uniforms
  vtkMetalCamera* cam = vtkMetalCamera::SafeDownCast(ren->GetActiveCamera());
  if (cam)
  {
    cam->Render(ren);
    if (!I->SceneBuffer)
    {
      I->SceneBuffer = [device newBufferWithLength:vtkMetalCamera::GetSceneTransformsSize()
                                          options:MTLResourceStorageModeShared];
    }
    memcpy([I->SceneBuffer contents], cam->GetCachedSceneTransforms(),
           vtkMetalCamera::GetSceneTransformsSize());
  }

  // Material
  if (!I->MaterialBuffer)
  {
    I->MaterialBuffer = [device newBufferWithLength:sizeof(float) * 18
                                           options:MTLResourceStorageModeShared];
  }
  {
    vtkProperty* p = actor->GetProperty();
    float* mb = static_cast<float*>([I->MaterialBuffer contents]);
    double amb[3], dif[3], spc[3];
    p->GetAmbientColor(amb);
    p->GetDiffuseColor(dif);
    p->GetSpecularColor(spc);
    mb[0] = amb[0]; mb[1] = amb[1]; mb[2] = amb[2]; mb[3] = p->GetAmbient();
    mb[4] = dif[0]; mb[5] = dif[1]; mb[6] = dif[2]; mb[7] = p->GetDiffuse();
    mb[8] = spc[0]; mb[9] = spc[1]; mb[10] = spc[2]; mb[11] = p->GetSpecular();
    mb[12] = mb[13] = mb[14] = mb[15] = 0.f;
    mb[16] = p->GetOpacity();
    mb[17] = p->GetSpecularPower();
  }

  // Lights
  if (!I->LightBuffer)
  {
    I->LightBuffer = [device newBufferWithLength:sizeof(float) * (16 * 8 + 4)
                                         options:MTLResourceStorageModeShared];
  }
  {
    char* buf = static_cast<char*>([I->LightBuffer contents]);
    vtkLightCollection* lc = ren->GetLights();
    vtkLight* light;
    vtkCollectionSimpleIterator cookie;
    int cnt = 0;
    lc->InitTraversal(cookie);
    while ((light = lc->GetNextLight(cookie)) && cnt < 8)
    {
      float* lb = reinterpret_cast<float*>(buf + cnt * 64);
      double* pos = light->GetPosition();
      double* fp = light->GetFocalPoint();
      double dir[3] = { fp[0] - pos[0], fp[1] - pos[1], fp[2] - pos[2] };
      double* col = light->GetDiffuseColor();
      lb[0] = pos[0]; lb[1] = pos[1]; lb[2] = pos[2];
      lb[3] = static_cast<float>(light->GetLightType());
      lb[4] = dir[0]; lb[5] = dir[1]; lb[6] = dir[2];
      lb[7] = light->GetConeAngle();
      lb[8] = col[0]; lb[9] = col[1]; lb[10] = col[2];
      lb[11] = light->GetIntensity();
      const double* att = light->GetAttenuationValues();
      lb[12] = att[0]; lb[13] = att[1]; lb[14] = att[2];
      lb[15] = light->GetExponent();
      ++cnt;
    }
    *reinterpret_cast<int*>(buf + 8 * 64) = cnt;
  }

  // Coincident offset
  if (!I->CoincidentBuffer)
  {
    I->CoincidentBuffer = [device newBufferWithLength:sizeof(float) * 5
                                             options:MTLResourceStorageModeShared];
    float* cb = static_cast<float*>([I->CoincidentBuffer contents]);
    cb[0] = 1.f;
    cb[1] = 1.f;
    cb[2] = 1.f;
    cb[3] = 1.f;
    cb[4] = -1.f;
  }

  // Clip planes
  if (!I->ClipPlaneBuffer)
  {
    I->ClipPlaneBuffer = [device newBufferWithLength:sizeof(float) * 4 * 6 + 16
                                            options:MTLResourceStorageModeShared];
    memset([I->ClipPlaneBuffer contents], 0, sizeof(float) * 4 * 6 + 16);
  }

  // Prop ID (PickIds {propId, compositeIndex})
  if (!I->PropIdBuffer)
  {
    I->PropIdBuffer = [device newBufferWithLength:sizeof(PickIds)
                                         options:MTLResourceStorageModeShared];
  }
  {
    // Per-render prop ID from the active hardware selector; 0 when not picking.
    uint32_t glyphPropId = 0;
    vtkMetalHardwareSelector* sel =
      vtkMetalHardwareSelector::SafeDownCast(ren->GetSelector());
    if (sel)
    {
      int id = sel->GetPropID(actor);
      if (id >= 0)
      {
        glyphPropId = static_cast<uint32_t>(id);
      }
    }
    PickIds ids = { glyphPropId, 0 };
    memcpy([I->PropIdBuffer contents], &ids, sizeof(PickIds));
  }

  // Cull mode
  if (actor->GetProperty()->GetBackfaceCulling())
    [enc setCullMode:MTLCullModeBack];
  else if (actor->GetProperty()->GetFrontfaceCulling())
    [enc setCullMode:MTLCullModeFront];
  else
    [enc setCullMode:MTLCullModeNone];

  // Helper lambda to bind all buffers and draw
  auto bindAndDraw = [&](id<MTLRenderPipelineState> pipeline,
                          MTLPrimitiveType primType,
                          vtkIdType vertCount)
  {
    if (!pipeline)
      return;
    [enc setRenderPipelineState:pipeline];
    [enc setVertexBuffer:I->SrcPositionBuffer offset:0 atIndex:kGlyphSrcPosition];
    [enc setVertexBuffer:I->SrcNormalBuffer offset:0 atIndex:kGlyphSrcNormal];
    [enc setVertexBuffer:I->InstTransformBuffer offset:0 atIndex:kGlyphTransform];
    [enc setVertexBuffer:I->InstNormalTransformBuffer offset:0 atIndex:kGlyphNormalTransform];
    [enc setVertexBuffer:I->InstColorBuffer offset:0 atIndex:kGlyphInstanceColor];
    [enc setVertexBuffer:I->InstPickIdBuffer offset:0 atIndex:kGlyphPickId];
    [enc setVertexBuffer:I->SceneBuffer offset:0 atIndex:kGlyphScene];
    [enc setVertexBuffer:I->ClipPlaneBuffer offset:0 atIndex:kGlyphClipPlane];
    [enc setVertexBuffer:I->PropIdBuffer offset:0 atIndex:kGlyphPropId];

    [enc setFragmentBuffer:I->MaterialBuffer offset:0 atIndex:0];
    [enc setFragmentBuffer:I->LightBuffer offset:0 atIndex:1];
    [enc setFragmentBuffer:I->SceneBuffer offset:0 atIndex:2];
    [enc setFragmentBuffer:I->CoincidentBuffer offset:0 atIndex:3];
    [enc setFragmentBuffer:I->ClipPlaneBuffer offset:0 atIndex:9];

    [enc drawPrimitives:primType
            vertexStart:0
          vertexCount:vertCount
        instanceCount:I->NumInstances];
  };

  // Draw
  if (I->HasTriangles)
  {
    bindAndDraw(I->TriPipeline, MTLPrimitiveTypeTriangle, I->TriVertexCount);
  }
  if (I->HasLines)
  {
    bindAndDraw(I->LinePipeline, MTLPrimitiveTypeLine, I->LineVertexCount);
  }
  if (I->HasPoints)
  {
    bindAndDraw(I->PtPipeline, MTLPrimitiveTypePoint, I->PtVertexCount);
  }
}

VTK_ABI_NAMESPACE_END
