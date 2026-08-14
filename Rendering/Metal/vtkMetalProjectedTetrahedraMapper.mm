// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-FileCopyrightText: Copyright 2003 Sandia Corporation
// SPDX-License-Identifier: LicenseRef-BSD-3-Clause-Sandia-USGov

#include "vtkMetalProjectedTetrahedraMapper.h"

#include "vtkAbstractMapper.h"
#include "vtkCamera.h"
#include "vtkCellArray.h"
#include "vtkCellData.h"
#include "vtkCellIterator.h"
#include "vtkFloatArray.h"
#include "vtkIdList.h"
#include "vtkIdTypeArray.h"
#include "vtkInformation.h"
#include "vtkMath.h"
#include "vtkMatrix3x3.h"
#include "vtkMatrix4x4.h"
#include "vtkMetalRenderWindow.h"
#include "vtkMetalShaders.h"
#include "vtkNew.h"
#include "vtkObjectFactory.h"
#include "vtkOverrideAttribute.h"
#include "vtkPointData.h"
#include "vtkRenderer.h"
#include "vtkSmartPointer.h"
#include "vtkTimerLog.h"
#include "vtkUnsignedCharArray.h"
#include "vtkUnstructuredGrid.h"
#include "vtkVisibilitySort.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkWindow.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <vector>

VTK_ABI_NAMESPACE_BEGIN

namespace
{
static const int tet_edges[6][2] = { { 0, 1 }, { 1, 2 }, { 2, 0 }, { 0, 3 }, { 1, 3 }, { 2, 3 } };

constexpr int SqrtTableSize = 2048;

// Release a Metal object held as a void* member (MRC helper).
inline void ReleaseMetalObject(void*& obj)
{
  if (obj)
  {
    [(__bridge id)obj release];
    obj = nullptr;
  }
}

// Takes ownership of a +1 Metal object into a void* member slot.
inline void AssignMetalObject(void*& slot, id obj)
{
  if (slot == (__bridge void*)obj)
  {
    return;
  }
  if (slot)
  {
    [(__bridge id)slot release];
  }
  slot = (__bridge void*)obj;
}

// Ensure a shared storage device buffer that can hold requiredBytes, growing
// geometrically. Returns the id<MTLBuffer> (or nil on failure).
inline id<MTLBuffer> EnsureBuffer(void*& slot, uint64_t& capacity, id<MTLDevice> device,
  uint64_t requiredBytes)
{
  if (slot && capacity >= requiredBytes)
  {
    return (__bridge id<MTLBuffer>)slot;
  }
  ReleaseMetalObject(slot);
  const uint64_t newCapacity = std::max<uint64_t>(requiredBytes, static_cast<uint64_t>(1) << 16);
  id<MTLBuffer> buffer = [device newBufferWithLength:newCapacity options:MTLResourceStorageModeShared];
  if (!buffer)
  {
    return nil;
  }
  AssignMetalObject(slot, buffer);
  capacity = newCapacity;
  return buffer;
}
}

//------------------------------------------------------------------------------
vtkStandardNewMacro(vtkMetalProjectedTetrahedraMapper);

//------------------------------------------------------------------------------
vtkOverrideAttribute* vtkMetalProjectedTetrahedraMapper::CreateOverrideAttributes()
{
  return vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "Metal", nullptr);
}

//------------------------------------------------------------------------------
vtkMetalProjectedTetrahedraMapper::vtkMetalProjectedTetrahedraMapper()
{
  this->TransformedPoints = vtkFloatArray::New();
  this->Colors = vtkUnsignedCharArray::New();
  this->LastProperty = nullptr;
  this->MaxCellSize = 0;
  this->GaveError = 0;
  this->SqrtTable = new float[SqrtTableSize];
  this->SqrtTableBias = 0.0;
}

//------------------------------------------------------------------------------
vtkMetalProjectedTetrahedraMapper::~vtkMetalProjectedTetrahedraMapper()
{
  this->ReleaseGraphicsResources(nullptr);
  this->TransformedPoints->Delete();
  this->Colors->Delete();
  delete[] this->SqrtTable;
}

//------------------------------------------------------------------------------
void vtkMetalProjectedTetrahedraMapper::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "VisibilitySort: " << this->VisibilitySort << endl;
}

//------------------------------------------------------------------------------
bool vtkMetalProjectedTetrahedraMapper::IsSupported(vtkRenderWindow* rwin)
{
  return (vtkMetalRenderWindow::SafeDownCast(rwin) != nullptr);
}

//------------------------------------------------------------------------------
void vtkMetalProjectedTetrahedraMapper::ReleaseGraphicsResources(vtkWindow* vtkNotUsed(win))
{
  ReleaseMetalObject(this->CachedLibrary);
  ReleaseMetalObject(this->PipelineState);
  ReleaseMetalObject(this->DepthStencilState);
  ReleaseMetalObject(this->PositionsBuffer);
  ReleaseMetalObject(this->ColorsBuffer);
  ReleaseMetalObject(this->AttenDepthBuffer);
  ReleaseMetalObject(this->IndexBuffer);
  this->CachedDevice = nullptr;
  this->CachedSampleCount = 0;

  this->Superclass::ReleaseGraphicsResources(nullptr);
}

//------------------------------------------------------------------------------
void vtkMetalProjectedTetrahedraMapper::Render(vtkRenderer* renderer, vtkVolume* volume)
{
  vtkMetalRenderWindow* renWin = vtkMetalRenderWindow::SafeDownCast(renderer->GetRenderWindow());
  if (!renWin)
  {
    vtkErrorMacro("Support for non-Metal render windows not implemented");
    return;
  }

  vtkUnstructuredGridBase* input = this->GetInput();
  if (!input)
  {
    return;
  }

  vtkVolumeProperty* property = volume->GetProperty();
  if (!property)
  {
    return;
  }

  // Check to see if input changed.
  if ((this->InputAnalyzedTime < this->MTime) || (this->InputAnalyzedTime < input->GetMTime()))
  {
    this->GaveError = 0;
    float max_cell_size2 = 0;

    if (input->GetNumberOfCells() == 0)
    {
      // Apparently, the input has no cells.  Just do nothing.
      return;
    }

    vtkSmartPointer<vtkCellIterator> cellIter =
      vtkSmartPointer<vtkCellIterator>::Take(input->NewCellIterator());
    for (cellIter->InitTraversal(); !cellIter->IsDoneWithTraversal(); cellIter->GoToNextCell())
    {
      vtkIdType npts = cellIter->GetNumberOfPoints();
      if (npts != 4)
      {
        if (!this->GaveError)
        {
          vtkErrorMacro("Encountered non-tetrahedra cell!");
          this->GaveError = 1;
        }
        continue;
      }
      vtkIdType* pts = cellIter->GetPointIds()->GetPointer(0);
      for (int j = 0; j < 6; j++)
      {
        double p1[3], p2[3];
        input->GetPoint(pts[tet_edges[j][0]], p1);
        input->GetPoint(pts[tet_edges[j][1]], p2);
        float size2 = (float)vtkMath::Distance2BetweenPoints(p1, p2);
        max_cell_size2 = std::max(size2, max_cell_size2);
      }
    }

    this->MaxCellSize = (float)sqrt(max_cell_size2);

    // Build a sqrt lookup table for measuring distances.  During perspective
    // modes we have to take a lot of square roots, and a table is much faster
    // than calling the sqrt function.
    this->SqrtTableBias = (SqrtTableSize - 1) / max_cell_size2;
    for (int i = 0; i < SqrtTableSize; i++)
    {
      this->SqrtTable[i] = (float)sqrt(i / this->SqrtTableBias);
    }

    this->InputAnalyzedTime.Modified();
  }

  if (renWin->CheckAbortStatus() || this->GaveError)
  {
    return;
  }

  // Check to see if we need to remap colors.
  if ((this->ColorsMappedTime < this->MTime) || (this->ColorsMappedTime < input->GetMTime()) ||
    (this->LastProperty != property) || (this->ColorsMappedTime < property->GetMTime()))
  {
    vtkDataArray* scalars = vtkAbstractMapper::GetScalars(input, this->ScalarMode,
      this->ArrayAccessMode, this->ArrayId, this->ArrayName, this->UsingCellColors);
    if (!scalars)
    {
      vtkErrorMacro(<< "Can't use projected tetrahedra without scalars!");
      return;
    }

    vtkProjectedTetrahedraMapper::MapScalarsToColors(this->Colors, property, scalars);

    this->ColorsMappedTime.Modified();
    this->LastProperty = property;
  }
  if (renWin->CheckAbortStatus())
  {
    return;
  }

  this->Timer->StartTimer();

  this->ProjectTetrahedra(renderer, volume);

  this->Timer->StopTimer();
  this->TimeToDraw = this->Timer->GetElapsedTime();
}

//------------------------------------------------------------------------------
float vtkMetalProjectedTetrahedraMapper::GetCorrectedDepth(float x, float y, float z1, float z2,
  const float inverse_projection_mat[16], int use_linear_depth_correction,
  float linear_depth_correction)
{
  if (use_linear_depth_correction)
  {
    float depth = linear_depth_correction * (z1 - z2);
    if (depth < 0)
      depth = -depth;
    return depth;
  }
  else
  {
    float eye1[3], eye2[3], invw;

    // This code does the same as the commented code above, but also collects
    // common arithmetic between the two matrix x vector operations.  An
    // optimizing compiler may or may not pick up on that.
    float common[4];

    common[0] =
      (inverse_projection_mat[0] * x + inverse_projection_mat[4] * y + inverse_projection_mat[12]);
    common[1] =
      (inverse_projection_mat[1] * x + inverse_projection_mat[5] * y + inverse_projection_mat[13]);
    common[2] = (inverse_projection_mat[2] * x + inverse_projection_mat[6] * y +
      inverse_projection_mat[10] * z1 + inverse_projection_mat[14]);
    common[3] =
      (inverse_projection_mat[3] * x + inverse_projection_mat[7] * y + inverse_projection_mat[15]);

    invw = 1 / (common[3] + inverse_projection_mat[11] * z1);
    eye1[0] = invw * (common[0] + inverse_projection_mat[8] * z1);
    eye1[1] = invw * (common[1] + inverse_projection_mat[9] * z1);
    eye1[2] = invw * (common[2] + inverse_projection_mat[10] * z1);

    invw = 1 / (common[3] + inverse_projection_mat[11] * z2);
    eye2[0] = invw * (common[0] + inverse_projection_mat[8] * z2);
    eye2[1] = invw * (common[1] + inverse_projection_mat[9] * z2);
    eye2[2] = invw * (common[2] + inverse_projection_mat[10] * z2);

    float dist2 = vtkMath::Distance2BetweenPoints(eye1, eye2);
    return this->SqrtTable[(int)(dist2 * this->SqrtTableBias)];
  }
}

//------------------------------------------------------------------------------
void vtkMetalProjectedTetrahedraMapper::ProjectTetrahedra(vtkRenderer* renderer, vtkVolume* volume)
{
  vtkMetalRenderWindow* window = vtkMetalRenderWindow::SafeDownCast(renderer->GetRenderWindow());
  if (!window)
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)window->GetMetalDevice();
  id<MTLRenderCommandEncoder> encoder =
    (__bridge id<MTLRenderCommandEncoder>)window->GetCurrentRenderCommandEncoder();
  if (!device || !encoder)
  {
    return;
  }

  // Ensure the Metal pipeline/depth state is built for this device and MSAA
  // sample count (the volume pass may run with or without multisampling).
  const int sampleCount = window->GetEffectiveSampleCount();
  if (this->CachedDevice != window->GetMetalDevice() || this->CachedSampleCount != sampleCount)
  {
    ReleaseMetalObject(this->CachedLibrary);
    ReleaseMetalObject(this->PipelineState);
    ReleaseMetalObject(this->DepthStencilState);
    this->CachedDevice = window->GetMetalDevice();
    this->CachedSampleCount = sampleCount;
  }

  if (!this->PipelineState)
  {
    @autoreleasepool
    {
      NSError* error = nil;
      if (!this->CachedLibrary)
      {
        NSString* shaderSource = [NSString stringWithUTF8String:vtkMetalShaders];
        id<MTLLibrary> library =
          [device newLibraryWithSource:shaderSource options:nil error:&error];
        if (!library)
        {
          vtkErrorMacro(<< "Failed to compile Metal shader library: "
                        << [[error localizedDescription] UTF8String]);
          return;
        }
        AssignMetalObject(this->CachedLibrary, library);
      }

      id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedLibrary;
      id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_projected_tetrahedra_main"];
      id<MTLFunction> fragmentFunc =
        [library newFunctionWithName:@"fragment_projected_tetrahedra_main"];
      if (!vertexFunc || !fragmentFunc)
      {
        [vertexFunc release];
        [fragmentFunc release];
        vtkErrorMacro("Failed to find projected tetrahedra shaders in the Metal library");
        return;
      }

      MTLRenderPipelineDescriptor* psoDesc = [[MTLRenderPipelineDescriptor alloc] init];
      psoDesc.vertexFunction = vertexFunc;
      psoDesc.fragmentFunction = fragmentFunc;
      psoDesc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
      psoDesc.rasterSampleCount = (NSUInteger)sampleCount;

      // The volume pass renders into BGRA8 with a Depth32Float attachment.
      psoDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
      psoDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

      // GL parity (vtkOpenGLProjectedTetrahedraMapper):
      // glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
      psoDesc.colorAttachments[0].blendingEnabled = YES;
      psoDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
      psoDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      psoDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
      psoDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
      psoDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      psoDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;

      id<MTLRenderPipelineState> pso =
        [device newRenderPipelineStateWithDescriptor:psoDesc error:&error];
      [psoDesc release];
      [vertexFunc release];
      [fragmentFunc release];

      if (!pso)
      {
        vtkErrorMacro(<< "Failed to create projected tetrahedra pipeline: "
                      << [[error localizedDescription] UTF8String]);
        return;
      }
      AssignMetalObject(this->PipelineState, pso);

      // Depth test Less (GL parity: the OpenGL mapper only enables GL_DEPTH_TEST,
      // which uses the default GL_LESS func), depth write off (translucent).
      // Using LessEqual would let tetrahedra that coincide with opaque geometry
      // (e.g. a contour surface) overdraw it, unlike the GL baseline.
      MTLDepthStencilDescriptor* dsDesc = [[MTLDepthStencilDescriptor alloc] init];
      dsDesc.depthCompareFunction = MTLCompareFunctionLess;
      dsDesc.depthWriteEnabled = NO;
      id<MTLDepthStencilState> ds = [device newDepthStencilStateWithDescriptor:dsDesc];
      [dsDesc release];
      AssignMetalObject(this->DepthStencilState, ds);
    }
  }

  if (!this->PipelineState)
  {
    return;
  }

  vtkUnstructuredGridBase* input = this->GetInput();
  if (!input || input->GetNumberOfCells() == 0)
  {
    return;
  }

  this->VisibilitySort->SetInput(input);
  this->VisibilitySort->SetDirectionToBackToFront();
  volume->GetModelToWorldMatrix(this->tmpMat);
  this->VisibilitySort->SetModelTransform(this->tmpMat);
  this->VisibilitySort->SetCamera(renderer->GetActiveCamera());
  this->VisibilitySort->SetMaxCellsReturned(1000);

  this->VisibilitySort->InitTraversal();

  if (window->CheckAbortStatus())
  {
    return;
  }

  // Build the projection and modelview matrices in the same column-major float
  // layout the OpenGL mapper feeds into TransformPoints.  With the nearz=0/
  // farz=1 projection above, clip-space Z is in [0, 1], which Metal writes to
  // the depth buffer as-is (matching GL's (z+1)/2 window depth for the GL
  // mapper's [-1, 1] clip Z).
  vtkCamera* cam = renderer->GetActiveCamera();
  vtkMatrix4x4* wcvc = cam->GetViewTransformMatrix();
  // Use the nearz=-1/farz=1 projection (GL convention) for the CPU-side
  // transforms so the opacity computation (GetCorrectedDepth) matches the GL
  // baseline bit-for-bit.  The GPU positions are remapped to Metal's [0,1]
  // clip Z when packed (0.5*z+0.5), which keeps the written depth identical to
  // the scene geometry (projected with nearz=0/farz=1 in vtkMetalCamera).
  vtkMatrix4x4* vcdc = cam->GetProjectionTransformMatrix(renderer->GetTiledAspectRatio(), -1, 1);

  float projection_mat[16];
  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      projection_mat[i * 4 + j] = static_cast<float>(vcdc->GetElement(j, i));
    }
  }

  float modelview_mat[16];
  if (!volume->GetIsIdentity())
  {
    volume->GetModelToWorldMatrix(this->tmpMat);
    this->tmpMat2->DeepCopy(wcvc);
    vtkMatrix4x4::Multiply4x4(this->tmpMat2, this->tmpMat, this->tmpMat);
    this->tmpMat->Transpose();
    for (int i = 0; i < 4; ++i)
    {
      for (int j = 0; j < 4; ++j)
      {
        modelview_mat[i * 4 + j] = static_cast<float>(this->tmpMat->GetElement(i, j));
      }
    }
  }
  else
  {
    for (int i = 0; i < 4; ++i)
    {
      for (int j = 0; j < 4; ++j)
      {
        modelview_mat[i * 4 + j] = static_cast<float>(wcvc->GetElement(j, i));
      }
    }
  }

  // Get the inverse projection matrix so that we can convert distances in
  // clipping space to distances in world or eye space.
  float inverse_projection_mat[16];
  float linear_depth_correction = 1;
  int use_linear_depth_correction;
  double tmp_mat[16];

  // VTK's matrix functions use doubles.
  std::copy(projection_mat, projection_mat + 16, tmp_mat);
  // VTK and OpenGL store their matrices differently.  Correct.
  vtkMatrix4x4::Transpose(tmp_mat, tmp_mat);
  // Take the inverse.
  vtkMatrix4x4::Invert(tmp_mat, tmp_mat);
  // Restore back to OpenGL form.
  vtkMatrix4x4::Transpose(tmp_mat, tmp_mat);
  // Copy back to float for faster computation.
  std::copy(tmp_mat, tmp_mat + 16, inverse_projection_mat);

  // Check to see if we can just do a linear depth correction from clipping
  // space to eye space.
  use_linear_depth_correction = ((projection_mat[3] == 0.0) && (projection_mat[7] == 0.0) &&
    (projection_mat[11] == 0.0) && (projection_mat[15] == 1.0));
  if (use_linear_depth_correction)
  {
    float pos1[3], *pos2;

    pos1[0] = inverse_projection_mat[8] + inverse_projection_mat[12];
    pos1[1] = inverse_projection_mat[9] + inverse_projection_mat[13];
    pos1[2] = inverse_projection_mat[10] + inverse_projection_mat[14];

    pos2 = inverse_projection_mat + 12;

    linear_depth_correction = sqrt(vtkMath::Distance2BetweenPoints(pos1, pos2));
  }
  // Transform all the points.
  vtkProjectedTetrahedraMapper::TransformPoints(
    input->GetPoints(), projection_mat, modelview_mat, this->TransformedPoints);
  float* points = this->TransformedPoints->GetPointer(0);

  if (window->CheckAbortStatus())
  {
    return;
  }

  float unit_distance = volume->GetProperty()->GetScalarOpacityUnitDistance();

  // tets have 4 points, 5th point here is used
  // to insert a point in case of intersections
  float tet_points[5 * 3] = { 0.0f };
  unsigned char tet_colors[5 * 3] = { 0 };
  float tet_texcoords[5 * 2] = { 0.0f };

  unsigned char* colors = this->Colors->GetPointer(0);
  vtkIdType totalnumcells = input->GetNumberOfCells();
  vtkIdType numcellsrendered = 0;
  vtkNew<vtkIdList> cellPointIds;

  std::vector<float> positions;
  positions.reserve(3 * 5 * totalnumcells);
  std::vector<unsigned char> packedColors;
  packedColors.reserve(4 * 5 * totalnumcells);
  std::vector<float> attenuationDepth;
  attenuationDepth.reserve(2 * 5 * totalnumcells);
  std::vector<unsigned int> indexArray;
  indexArray.reserve(3 * 4 * totalnumcells);

  // Bind the PT pipeline and depth state once per chunk loop.
  id<MTLRenderPipelineState> pso = (__bridge id<MTLRenderPipelineState>)this->PipelineState;
  id<MTLDepthStencilState> ds = (__bridge id<MTLDepthStencilState>)this->DepthStencilState;
  [encoder setRenderPipelineState:pso];
  [encoder setDepthStencilState:ds];

  // Let's do it!
  vtkIdType globalVertexOffset = 0;
  for (vtkIdTypeArray* sorted_cell_ids = this->VisibilitySort->GetNextCells();
       sorted_cell_ids != nullptr; sorted_cell_ids = this->VisibilitySort->GetNextCells())
  {
    if (window->CheckAbortStatus())
    {
      break;
    }
    vtkIdType* cell_ids = sorted_cell_ids->GetPointer(0);
    vtkIdType num_cell_ids = sorted_cell_ids->GetNumberOfTuples();

    float* posPtr = positions.data() + 3 * globalVertexOffset;
    unsigned char* colPtr = packedColors.data() + 4 * globalVertexOffset;
    float* attenPtr = attenuationDepth.data() + 2 * globalVertexOffset;
    int numPts = 0;

    for (vtkIdType i = 0; i < num_cell_ids; i++)
    {
      vtkIdType cell = cell_ids[i];
      input->GetCellPoints(cell, cellPointIds);
      int j;

      // Get the data for the tetrahedra.
      for (j = 0; j < 4; j++)
      {
        const float* p = points + 3 * cellPointIds->GetId(j);
        tet_points[j * 3 + 0] = p[0];
        tet_points[j * 3 + 1] = p[1];
        tet_points[j * 3 + 2] = p[2];

        const unsigned char* c;
        if (this->UsingCellColors)
        {
          c = colors + 4 * cell;
        }
        else
        {
          c = colors + 4 * cellPointIds->GetId(j);
        }

        tet_colors[j * 3 + 0] = c[0];
        tet_colors[j * 3 + 1] = c[1];
        tet_colors[j * 3 + 2] = c[2];

        tet_texcoords[j * 2 + 0] = static_cast<float>(c[3]) / 255.0f;
        tet_texcoords[j * 2 + 1] = 0;
      }

      // Do not render this cell if it is outside of the cutting planes.  For
      // most planes, cut if all points are outside.  For the near plane, cut if
      // any points are outside because things can go very wrong if one of the
      // points is behind the view.
      if (((tet_points[0 * 3 + 0] > 1.0f) && (tet_points[1 * 3 + 0] > 1.0f) &&
            (tet_points[2 * 3 + 0] > 1.0f) && (tet_points[3 * 3 + 0] > 1.0f)) ||
        ((tet_points[0 * 3 + 0] < -1.0f) && (tet_points[1 * 3 + 0] < -1.0f) &&
          (tet_points[2 * 3 + 0] < -1.0f) && (tet_points[3 * 3 + 0] < -1.0f)) ||
        ((tet_points[0 * 3 + 1] > 1.0f) && (tet_points[1 * 3 + 1] > 1.0f) &&
          (tet_points[2 * 3 + 1] > 1.0f) && (tet_points[3 * 3 + 1] > 1.0f)) ||
        ((tet_points[0 * 3 + 1] < -1.0f) && (tet_points[1 * 3 + 1] < -1.0f) &&
          (tet_points[2 * 3 + 1] < -1.0f) && (tet_points[3 * 3 + 1] < -1.0f)) ||
        ((tet_points[0 * 3 + 2] > 1.0f) && (tet_points[1 * 3 + 2] > 1.0f) &&
          (tet_points[2 * 3 + 2] > 1.0f) && (tet_points[3 * 3 + 2] > 1.0f)) ||
        ((tet_points[0 * 3 + 2] < -1.0f) || (tet_points[1 * 3 + 2] < -1.0f) ||
          (tet_points[2 * 3 + 2] < -1.0f) || (tet_points[3 * 3 + 2] < -1.0f)))
      {
        continue;
      }

      // The classic PT algorithm uses face normals to determine the
      // projection class and then do calculations individually.  However,
      // Wylie 2002 shows how to use the intersection of two segments to
      // calculate the depth of the thick part for any case.  Here, we use
      // face normals to determine which segments to use.  One segment
      // should be between two faces that are either both front facing or
      // back facing.  Obviously, we only need to test three faces to find
      // two such faces.  We test the three faces connected to point 0.
      vtkIdType segment1[2];
      vtkIdType segment2[2];

      float v1[2], v2[2], v3[3];
      v1[0] = tet_points[1 * 3 + 0] - tet_points[0 * 3 + 0];
      v1[1] = tet_points[1 * 3 + 1] - tet_points[0 * 3 + 1];
      v2[0] = tet_points[2 * 3 + 0] - tet_points[0 * 3 + 0];
      v2[1] = tet_points[2 * 3 + 1] - tet_points[0 * 3 + 1];
      v3[0] = tet_points[3 * 3 + 0] - tet_points[0 * 3 + 0];
      v3[1] = tet_points[3 * 3 + 1] - tet_points[0 * 3 + 1];

      float face_dir1 = v3[0] * v2[1] - v3[1] * v2[0];
      float face_dir2 = v1[0] * v3[1] - v1[1] * v3[0];
      float face_dir3 = v2[0] * v1[1] - v2[1] * v1[0];

      if ((face_dir1 * face_dir2 >= 0) &&
        ((face_dir1 != 0)       // Handle a special case where 2 faces
          || (face_dir2 != 0))) // are perpendicular to the view plane.
      {
        segment1[0] = 0;
        segment1[1] = 3;
        segment2[0] = 1;
        segment2[1] = 2;
      }
      else if (face_dir1 * face_dir3 >= 0)
      {
        segment1[0] = 0;
        segment1[1] = 2;
        segment2[0] = 1;
        segment2[1] = 3;
      }
      else // Unless the tet is degenerate, face_dir2*face_dir3 >= 0
      {
        segment1[0] = 0;
        segment1[1] = 1;
        segment2[0] = 2;
        segment2[1] = 3;
      }

#define VEC3SUB(Z, X, Y)                                                                           \
  do                                                                                               \
  {                                                                                                \
    (Z)[0] = (X)[0] - (Y)[0];                                                                      \
    (Z)[1] = (X)[1] - (Y)[1];                                                                      \
    (Z)[2] = (X)[2] - (Y)[2];                                                                      \
  } while (false)
#define P1 (tet_points + 3 * segment1[0])
#define P2 (tet_points + 3 * segment1[1])
#define P3 (tet_points + 3 * segment2[0])
#define P4 (tet_points + 3 * segment2[1])
#define C1 (tet_colors + 3 * segment1[0])
#define C2 (tet_colors + 3 * segment1[1])
#define C3 (tet_colors + 3 * segment2[0])
#define C4 (tet_colors + 3 * segment2[1])
#define T1 (tet_texcoords + 2 * segment1[0])
#define T2 (tet_texcoords + 2 * segment1[1])
#define T3 (tet_texcoords + 2 * segment2[0])
#define T4 (tet_texcoords + 2 * segment2[1])
      // Find the intersection of the projection of the two segments in the
      // XY plane.  This algorithm is based on that given in Graphics Gems
      // III, pg. 199-202.
      float A[3], B[3], C[3];
      // We can define the two lines parametrically as:
      //        P1 + alpha(A)
      //        P3 + beta(B)
      // where A = P2 - P1
      // and   B = P4 - P3.
      // alpha and beta are in the range [0,1] within the line segment.
      VEC3SUB(A, P2, P1);
      VEC3SUB(B, P4, P3);
      // The lines intersect when the values of the two parametric equations
      // are equal.  Setting them equal and moving everything to one side:
      //        0 = C + beta(B) - alpha(A)
      // where C = P3 - P1.
      VEC3SUB(C, P3, P1);
      // When we project the lines to the xy plane (which we do by throwing
      // away the z value), we have two equations and two unknowns.  The
      // following are the solutions for alpha and beta.
      float denominator = (A[0] * B[1] - A[1] * B[0]);
      if (denominator == 0)
        continue; // Must be degenerated tetrahedra.
      float alpha = (B[1] * C[0] - B[0] * C[1]) / denominator;
      float beta = (A[1] * C[0] - A[0] * C[1]) / denominator;

      if ((alpha >= 0) && (alpha <= 1))
      {
        // The two segments intersect.  This corresponds to class 2 in
        // Shirley and Tuchman (or one of the degenerate cases).

        // Make new point at intersection.
        tet_points[3 * 4 + 0] = P1[0] + alpha * A[0];
        tet_points[3 * 4 + 1] = P1[1] + alpha * A[1];
        tet_points[3 * 4 + 2] = P1[2] + alpha * A[2];

        // Find depth at intersection.
        float depth = this->GetCorrectedDepth(tet_points[3 * 4 + 0], tet_points[3 * 4 + 1],
          tet_points[3 * 4 + 2], P3[2] + beta * B[2], inverse_projection_mat,
          use_linear_depth_correction, linear_depth_correction);

        // Find color at intersection.
        tet_colors[3 * 4 + 0] = static_cast<unsigned char>(
          0.5f * (C1[0] + alpha * (C2[0] - C1[0]) + C3[0] + beta * (C4[0] - C3[0])));

        tet_colors[3 * 4 + 1] = static_cast<unsigned char>(
          0.5f * (C1[1] + alpha * (C2[1] - C1[1]) + C3[1] + beta * (C4[1] - C3[1])));

        tet_colors[3 * 4 + 2] = static_cast<unsigned char>(
          0.5f * (C1[2] + alpha * (C2[2] - C1[2]) + C3[2] + beta * (C4[2] - C3[2])));

        // Find the opacity at intersection.
        tet_texcoords[2 * 4 + 0] =
          0.5f * (T1[0] + alpha * (T2[0] - T1[0]) + T3[0] + alpha * (T4[0] - T3[0]));

        // Record the depth at the intersection.
        tet_texcoords[2 * 4 + 1] = depth / unit_distance;

        // Establish the order in which the points should be rendered.
        unsigned char indices[6];
        indices[0] = 4;
        indices[1] = segment1[0];
        indices[2] = segment2[0];
        indices[3] = segment1[1];
        indices[4] = segment2[1];
        indices[5] = segment1[0];
        // add the cells to the IBO
        for (int cellIdx = 0; cellIdx < 4; cellIdx++)
        {
          indexArray.push_back(indices[0] + numPts + globalVertexOffset);
          indexArray.push_back(indices[cellIdx + 1] + numPts + globalVertexOffset);
          indexArray.push_back(indices[cellIdx + 2] + numPts + globalVertexOffset);
        }
      }
      else
      {
        // The two segments do not intersect.  This corresponds to class 1
        // in Shirley and Tuchman.
        if (alpha <= 0)
        {
          // Flip segment1 so that alpha is >= 1.  P1 and P2 are also
          // flipped as are C1-C2 and T1-T2.  Note that this will
          // invalidate A.  B and beta are unaffected.
          std::swap(segment1[0], segment1[1]);
          alpha = 1 - alpha;
        }
        // From here on, we can assume P2 is the "thick" point.

        // Find the depth under the thick point.  Use the alpha and beta
        // from intersection to determine location of face under thick
        // point.
        float edgez = P3[2] + beta * B[2];
        float pointz = P1[2];
        float facez = (edgez + (alpha - 1) * pointz) / alpha;
        float depth = GetCorrectedDepth(P2[0], P2[1], P2[2], facez, inverse_projection_mat,
          use_linear_depth_correction, linear_depth_correction);

        // Fix color at thick point.  Average color with color of opposite
        // face.
        for (j = 0; j < 3; j++)
        {
          float edgec = C3[j] + beta * (C4[j] - C3[j]);
          float pointc = C1[j];
          float facec = (edgec + (alpha - 1) * pointc) / alpha;
          C2[j] = (unsigned char)(0.5f * (facec + C2[j]));
        }

        // Fix opacity at thick point.  Average opacity with opacity of
        // opposite face.
        float edgea = T3[0] + beta * (T4[0] - T3[0]);
        float pointa = T1[0];
        float facea = (edgea + (alpha - 1) * pointa) / alpha;
        T2[0] = 0.5f * (facea + T2[0]);

        // Record thickness at thick point.
        T2[1] = depth / unit_distance;

        // Establish the order in which the points should be rendered.
        unsigned char indices[5];
        indices[0] = segment1[1];
        indices[1] = segment1[0];
        indices[2] = segment2[0];
        indices[3] = segment2[1];
        indices[4] = segment1[0];

        // add the cells to the IBO
        for (int cellIdx = 0; cellIdx < 3; cellIdx++)
        {
          indexArray.push_back(indices[0] + numPts + globalVertexOffset);
          indexArray.push_back(indices[cellIdx + 1] + numPts + globalVertexOffset);
          indexArray.push_back(indices[cellIdx + 2] + numPts + globalVertexOffset);
        }
      }

      // Pack the points into the vertex buffers.  The CPU-side z's are in GL
      // NDC [-1,1]; remap to Metal NDC [0,1] so the written depth matches the
      // scene geometry (which is projected with nearz=0/farz=1).
      for (int ptIdx = 0; ptIdx < 5; ptIdx++)
      {
        *(posPtr++) = tet_points[ptIdx * 3];
        *(posPtr++) = tet_points[ptIdx * 3 + 1];
        *(posPtr++) = 0.5f * tet_points[ptIdx * 3 + 2] + 0.5f;

        *(colPtr++) = tet_colors[ptIdx * 3];
        *(colPtr++) = tet_colors[ptIdx * 3 + 1];
        *(colPtr++) = tet_colors[ptIdx * 3 + 2];
        *(colPtr++) = 255;

        *(attenPtr++) = tet_texcoords[ptIdx * 2];     // attenuation
        *(attenPtr++) = tet_texcoords[ptIdx * 2 + 1]; // depth
      }
      numPts += 5;
    }

    globalVertexOffset += numPts;

    const vtkIdType numVertices = numPts;
    const vtkIdType numIndices = static_cast<vtkIdType>(indexArray.size());
    if (numVertices <= 0 || numIndices <= 0)
    {
      continue;
    }

    numcellsrendered += num_cell_ids;
  }

  // Draw the accumulated tetrahedra in a single indexed draw.  All chunks are
  // packed into one set of buffers (with absolute vertex indices) so that each
  // draw reads its own data rather than the last chunk written to a shared
  // buffer.
  const vtkIdType totalVertices = globalVertexOffset;
  const vtkIdType totalIndices = static_cast<vtkIdType>(indexArray.size());
  if (totalVertices > 0 && totalIndices > 0)
  {
    id<MTLBuffer> posBuf = EnsureBuffer(
      this->PositionsBuffer, this->PositionsCapacity, device, sizeof(float) * 3 * totalVertices);
    id<MTLBuffer> colBuf = EnsureBuffer(
      this->ColorsBuffer, this->ColorsCapacity, device, sizeof(unsigned char) * 4 * totalVertices);
    id<MTLBuffer> attenBuf = EnsureBuffer(this->AttenDepthBuffer, this->AttenDepthCapacity, device,
      sizeof(float) * 2 * totalVertices);
    id<MTLBuffer> idxBuf = EnsureBuffer(
      this->IndexBuffer, this->IndexCapacity, device, sizeof(unsigned int) * totalIndices);
    if (posBuf && colBuf && attenBuf && idxBuf)
    {
      memcpy([posBuf contents], positions.data(), sizeof(float) * 3 * totalVertices);
      memcpy([colBuf contents], packedColors.data(), sizeof(unsigned char) * 4 * totalVertices);
      memcpy([attenBuf contents], attenuationDepth.data(), sizeof(float) * 2 * totalVertices);
      memcpy([idxBuf contents], indexArray.data(), sizeof(unsigned int) * totalIndices);

      [encoder setVertexBuffer:posBuf offset:0 atIndex:0];
      [encoder setVertexBuffer:colBuf offset:0 atIndex:1];
      [encoder setVertexBuffer:attenBuf offset:0 atIndex:2];
      [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:totalIndices
        indexType:MTLIndexTypeUInt32 indexBuffer:idxBuf indexBufferOffset:0];
    }
    else
    {
      vtkErrorMacro("Failed to allocate projected tetrahedra vertex buffers");
    }
  }

#undef VEC3SUB
#undef P1
#undef P2
#undef P3
#undef P4
#undef C1
#undef C2
#undef C3
#undef C4
#undef T1
#undef T2
#undef T3
#undef T4
}

VTK_ABI_NAMESPACE_END
