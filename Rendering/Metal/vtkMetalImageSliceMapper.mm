// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalImageSliceMapper.h"

#include "vtkMetalCamera.h"
#include "vtkMetalHardwareSelector.h"
#include "vtkMetalPolyDataMapper.h"
#include "vtkMetalRenderWindow.h"

#include "vtkActor.h"
#include "vtkCellArray.h"
#include "vtkDataArray.h"
#include "vtkFloatArray.h"
#include "vtkHardwareSelector.h"
#include "vtkImageData.h"
#include "vtkImageProperty.h"
#include "vtkImageSlice.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMapper.h"
#include "vtkMath.h"
#include "vtkMatrix3x3.h"
#include "vtkMatrix4x4.h"
#include "vtkNew.h"
#include "vtkObjectFactory.h"
#include "vtkOverrideAttribute.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkProperty.h"
#include "vtkRenderer.h"
#include "vtkScalarsToColors.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkTexture.h"
#include "vtkTrivialProducer.h"
#include "vtkUnsignedCharArray.h"

#include "vtkMetalMRC.h"

#import <Metal/Metal.h>

#include <cmath>

VTK_ABI_NAMESPACE_BEGIN
vtkStandardNewMacro(vtkMetalImageSliceMapper);

//------------------------------------------------------------------------------
class vtkMetalImageSliceMapper::vtkInternals
{
public:
  id<MTLDepthStencilState> OpaqueDepthState = nil;   // LessEqual, write=YES
  id<MTLDepthStencilState> ReadOnlyDepthState = nil; // LessEqual, write=NO

  // Hardware-selection pipeline (cell/point-ID picking)
  id<MTLRenderPipelineState> SelectionPipeline = nil;
  id<MTLBuffer> SelectionVertexBuffer = nil; // interleaved position+texCoord, 4 verts
  id<MTLBuffer> SelectionSceneBuffer = nil;  // SceneUniforms (camera transforms + model matrix)
  id<MTLBuffer> SelectionUniformBuffer = nil; // SliceSelectionUniforms

  void EnsureDepthStates(id<MTLDevice> device)
  {
    if (this->OpaqueDepthState && this->ReadOnlyDepthState)
    {
      return;
    }
    @autoreleasepool
    {
      if (!this->OpaqueDepthState)
      {
        MTLDepthStencilDescriptor* desc = [[MTLDepthStencilDescriptor alloc] init];
        desc.depthCompareFunction = MTLCompareFunctionLessEqual;
        desc.depthWriteEnabled = YES;
        this->OpaqueDepthState = [device newDepthStencilStateWithDescriptor:desc];
        [desc release];
      }
      if (!this->ReadOnlyDepthState)
      {
        MTLDepthStencilDescriptor* desc = [[MTLDepthStencilDescriptor alloc] init];
        desc.depthCompareFunction = MTLCompareFunctionLessEqual;
        desc.depthWriteEnabled = NO;
        this->ReadOnlyDepthState = [device newDepthStencilStateWithDescriptor:desc];
        [desc release];
      }
    }
  }

  void Release()
  {
    vtkMetalMRC::ReleaseAndNil(this->OpaqueDepthState);
    vtkMetalMRC::ReleaseAndNil(this->ReadOnlyDepthState);
    vtkMetalMRC::ReleaseAndNil(this->SelectionPipeline);
    vtkMetalMRC::ReleaseAndNil(this->SelectionVertexBuffer);
    vtkMetalMRC::ReleaseAndNil(this->SelectionSceneBuffer);
    vtkMetalMRC::ReleaseAndNil(this->SelectionUniformBuffer);
  }
};

//------------------------------------------------------------------------------
vtkMetalImageSliceMapper::vtkMetalImageSliceMapper()
  : Internals(new vtkMetalImageSliceMapper::vtkInternals())
{
  // setup the polygon mapper
  {
    vtkNew<vtkPolyData> polydata;
    vtkNew<vtkPoints> points;
    points->SetNumberOfPoints(4);
    polydata->SetPoints(points);

    vtkNew<vtkCellArray> tris;
    polydata->SetPolys(tris);

    vtkNew<vtkFloatArray> tcoords;
    tcoords->SetNumberOfComponents(2);
    tcoords->SetNumberOfTuples(4);
    polydata->GetPointData()->SetTCoords(tcoords);

    vtkNew<vtkTrivialProducer> prod;
    prod->SetOutput(polydata);
    vtkNew<vtkMetalPolyDataMapper> polyDataMapper;
    polyDataMapper->SetInputConnection(prod->GetOutputPort());
    this->PolyDataActor = vtkActor::New();
    this->PolyDataActor->SetMapper(polyDataMapper);
    vtkNew<vtkTexture> texture;
    texture->RepeatOff();
    this->PolyDataActor->SetTexture(texture);
  }

  // setup the backing polygon mapper
  {
    vtkNew<vtkPolyData> polydata;
    vtkNew<vtkPoints> points;
    points->SetNumberOfPoints(4);
    polydata->SetPoints(points);

    vtkNew<vtkCellArray> tris;
    polydata->SetPolys(tris);

    vtkNew<vtkTrivialProducer> prod;
    prod->SetOutput(polydata);
    vtkNew<vtkMetalPolyDataMapper> polyDataMapper;
    polyDataMapper->SetInputConnection(prod->GetOutputPort());
    this->BackingPolyDataActor = vtkActor::New();
    this->BackingPolyDataActor->SetMapper(polyDataMapper);
  }

  // setup the background polygon mapper
  {
    vtkNew<vtkPolyData> polydata;
    vtkNew<vtkPoints> points;
    points->SetNumberOfPoints(10);
    polydata->SetPoints(points);

    vtkNew<vtkCellArray> tris;
    polydata->SetPolys(tris);

    vtkNew<vtkTrivialProducer> prod;
    prod->SetOutput(polydata);
    vtkNew<vtkMetalPolyDataMapper> polyDataMapper;
    polyDataMapper->SetInputConnection(prod->GetOutputPort());
    this->BackgroundPolyDataActor = vtkActor::New();
    this->BackgroundPolyDataActor->SetMapper(polyDataMapper);
  }

  this->TextureSize[0] = 0;
  this->TextureSize[1] = 0;
  this->TextureBytesPerPixel = 1;

  this->LastOrientation = -1;
  this->LastSliceNumber = VTK_INT_MAX;
}

//------------------------------------------------------------------------------
vtkMetalImageSliceMapper::~vtkMetalImageSliceMapper()
{
  this->BackgroundPolyDataActor->UnRegister(this);
  this->BackingPolyDataActor->UnRegister(this);
  this->PolyDataActor->UnRegister(this);

  delete this->Internals;
  this->Internals = nullptr;
}

//------------------------------------------------------------------------------
vtkOverrideAttribute* vtkMetalImageSliceMapper::CreateOverrideAttributes()
{
  return vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "Metal", nullptr);
}

//------------------------------------------------------------------------------
// Release the graphics resources used by this mapper.
void vtkMetalImageSliceMapper::ReleaseGraphicsResources(vtkWindow* renWin)
{
  this->BackgroundPolyDataActor->ReleaseGraphicsResources(renWin);
  this->BackingPolyDataActor->ReleaseGraphicsResources(renWin);
  this->PolyDataActor->ReleaseGraphicsResources(renWin);

  this->Internals->Release();
  this->Modified();
}

//------------------------------------------------------------------------------
// Subdivide the image until the pieces fit into texture memory
void vtkMetalImageSliceMapper::RecursiveRenderTexturedPolygon(
  vtkRenderer* ren, vtkImageProperty* property, vtkImageData* input, int extent[6], bool recursive)
{
  int xdim, ydim;
  int imageSize[2];
  int textureSize[2];

  // compute image size and texture size from extent
  this->ComputeTextureSize(extent, xdim, ydim, imageSize, textureSize);

  // Check if we can fit this texture in memory
  if (this->TextureSizeOK(textureSize, ren))
  {
    // We can fit it - render
    this->RenderTexturedPolygon(ren, property, input, extent, recursive);
  }

  // If the texture does not fit, then subdivide and render
  // each half.  Unless the graphics card couldn't handle
  // a texture a small as 256x256, because if it can't handle
  // that, then something has gone horribly wrong.
  else if (textureSize[0] > 256 || textureSize[1] > 256)
  {
    int subExtent[6];
    subExtent[0] = extent[0];
    subExtent[1] = extent[1];
    subExtent[2] = extent[2];
    subExtent[3] = extent[3];
    subExtent[4] = extent[4];
    subExtent[5] = extent[5];

    // Which is larger, x or y?
    int idx = ydim;
    int tsize = textureSize[1];
    if (textureSize[0] > textureSize[1])
    {
      idx = xdim;
      tsize = textureSize[0];
    }

    // Divide size by two
    tsize /= 2;

    // Render each half recursively
    subExtent[idx * 2] = extent[idx * 2];
    subExtent[idx * 2 + 1] = extent[idx * 2] + tsize - 1;
    this->RecursiveRenderTexturedPolygon(ren, property, input, subExtent, true);

    subExtent[idx * 2] = subExtent[idx * 2] + tsize;
    subExtent[idx * 2 + 1] = extent[idx * 2 + 1];
    this->RecursiveRenderTexturedPolygon(ren, property, input, subExtent, true);
  }
}

//------------------------------------------------------------------------------
// Load the given image extent into a texture and render it
void vtkMetalImageSliceMapper::RenderTexturedPolygon(
  vtkRenderer* ren, vtkImageProperty* property, vtkImageData* input, int extent[6], bool recursive)
{
  // get the previous texture load time
  vtkMTimeType loadTime = this->LoadTime.GetMTime();

  // the Metal mapper uploads the texture itself, so no explicit load is needed

  // get information about the image
  int xdim, ydim; // orientation of texture wrt input image
  vtkImageSliceMapper::GetDimensionIndices(this->Orientation, xdim, ydim);

  // verify that the orientation and slice has not changed
  bool orientationChanged = (this->Orientation != this->LastOrientation);
  this->LastOrientation = this->Orientation;
  bool sliceChanged = (this->SliceNumber != this->LastSliceNumber);
  this->LastSliceNumber = this->SliceNumber;

  // get the mtime of the property, including the lookup table
  vtkMTimeType propertyMTime = 0;
  if (property)
  {
    propertyMTime = property->GetMTime();
    if (!this->PassColorData)
    {
      vtkScalarsToColors* table = property->GetLookupTable();
      if (table)
      {
        vtkMTimeType mtime = table->GetMTime();
        propertyMTime = std::max(mtime, propertyMTime);
      }
    }
  }

  // need to reload the texture
  if (this->Superclass::GetMTime() > loadTime || propertyMTime > loadTime ||
    input->GetMTime() > loadTime || orientationChanged || sliceChanged || recursive)
  {
    // get the data to load as a texture
    int xsize = this->TextureSize[0];
    int ysize = this->TextureSize[1];
    int bytesPerPixel = this->TextureBytesPerPixel;

    // whether to try to use the input data directly as the texture
    bool reuseData = true;
    bool reuseTexture = true;

    // generate the data to be used as a texture
    unsigned char* data = this->MakeTextureData((this->PassColorData ? nullptr : property), input,
      extent, xsize, ysize, bytesPerPixel, reuseTexture, reuseData);

    this->TextureSize[0] = xsize;
    this->TextureSize[1] = ysize;
    this->TextureBytesPerPixel = bytesPerPixel;

    vtkNew<vtkImageData> id;
    id->SetExtent(0, xsize - 1, 0, ysize - 1, 0, 0);
    vtkNew<vtkUnsignedCharArray> uca;
    uca->SetNumberOfComponents(bytesPerPixel);
    // Use size_t to avoid integer overflow for large images
    uca->SetArray(data,
      static_cast<vtkIdType>(static_cast<size_t>(xsize) * static_cast<size_t>(ysize) *
        static_cast<size_t>(bytesPerPixel)),
      reuseData, vtkAbstractArray::VTK_DATA_ARRAY_DELETE);
    id->GetPointData()->SetScalars(uca);

    this->PolyDataActor->GetTexture()->SetInputData(id);

    if (property->GetInterpolationType() == VTK_NEAREST_INTERPOLATION && !this->ExactPixelMatch)
    {
      this->PolyDataActor->GetTexture()->InterpolateOff();
    }
    else
    {
      this->PolyDataActor->GetTexture()->InterpolateOn();
    }

    this->PolyDataActor->GetTexture()->EdgeClampOn();

    // modify the load time to the current time
    this->LoadTime.Modified();
  }

  vtkPoints* points = this->Points;
  if (this->ExactPixelMatch && this->SliceFacesCamera)
  {
    points = nullptr;
  }

  this->RenderPolygon(this->PolyDataActor, points, extent, ren);

  if (this->Background)
  {
    double ambient = property->GetAmbient();
    double diffuse = property->GetDiffuse();

    double bkcolor[4];
    this->GetBackgroundColor(property, bkcolor);
    vtkProperty* pdProp = this->BackgroundPolyDataActor->GetProperty();
    pdProp->SetAmbient(ambient);
    pdProp->SetDiffuse(diffuse);
    pdProp->SetColor(bkcolor[0], bkcolor[1], bkcolor[2]);
    this->RenderBackground(this->BackgroundPolyDataActor, points, extent, ren);
  }
}

//------------------------------------------------------------------------------
// Render the polygon that displays the image data
void vtkMetalImageSliceMapper::RenderPolygon(
  vtkActor* actor, vtkPoints* points, const int extent[6], vtkRenderer* ren)
{
  bool textured = (actor->GetTexture() != nullptr);
  vtkPolyData* poly = vtkPolyDataMapper::SafeDownCast(actor->GetMapper())->GetInput();
  vtkPoints* polyPoints = poly->GetPoints();
  if (this->GetOutputPointsPrecision() == vtkAlgorithm::DOUBLE_PRECISION)
  {
    polyPoints->SetDataTypeToDouble();
  }
  vtkCellArray* tris = poly->GetPolys();
  vtkDataArray* polyTCoords = poly->GetPointData()->GetTCoords();

  // do we need to rebuild the cell array?
  int numTris = 2;
  if (points)
  {
    numTris = (points->GetNumberOfPoints() - 2);
  }
  if (tris->GetNumberOfConnectivityIds() != 3 * numTris)
  {
    tris->Initialize();
    tris->AllocateEstimate(numTris, 3);
    // this wacky code below works for 2 and 4 triangles at least
    for (vtkIdType i = 0; i < numTris; i++)
    {
      tris->InsertNextCell(3);
      tris->InsertCellPoint(numTris + 1 - (i + 1) / 2);
      tris->InsertCellPoint(i / 2);
      tris->InsertCellPoint((i % 2 == 0) ? numTris - i / 2 : i / 2 + 1);
    }
    tris->Modified();
  }

  // now rebuild the points/tcoords as needed
  if (!points)
  {
    double coords[12], tcoords[8];
    this->MakeTextureGeometry(extent, coords, tcoords);

    polyPoints->SetNumberOfPoints(4);
    if (textured)
    {
      polyTCoords->SetNumberOfTuples(4);
    }
    for (int i = 0; i < 4; i++)
    {
      polyPoints->SetPoint(i, coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]);
      if (textured)
      {
        polyTCoords->SetTuple(i, &tcoords[2 * i]);
      }
    }
    polyPoints->Modified();
    if (textured)
    {
      polyTCoords->Modified();
    }
  }
  else if (points->GetNumberOfPoints())
  {
    int xdim, ydim;
    vtkImageSliceMapper::GetDimensionIndices(this->Orientation, xdim, ydim);
    double* origin = this->DataOrigin;
    double* spacing = this->DataSpacing;
    double xshift = -(0.5 - extent[2 * xdim]) * spacing[xdim];
    double xscale = this->TextureSize[xdim] * spacing[xdim];
    double yshift = -(0.5 - extent[2 * ydim]) * spacing[ydim];
    double yscale = this->TextureSize[ydim] * spacing[ydim];
    vtkIdType ncoords = points->GetNumberOfPoints();
    double coord[3];
    double tcoord[2];
    double invDirection[9];

    polyPoints->DeepCopy(points);
    if (textured)
    {
      vtkMatrix3x3::Invert(this->DataDirection, invDirection);
      polyTCoords->SetNumberOfTuples(ncoords);
    }

    for (vtkIdType i = 0; i < ncoords; i++)
    {
      if (textured)
      {
        // convert points from 3D model coords to 2D texture coords
        points->GetPoint(i, coord);
        vtkMath::Subtract(coord, origin, coord);
        vtkMatrix3x3::MultiplyPoint(invDirection, coord, coord);
        tcoord[0] = (coord[0] - xshift) / xscale;
        tcoord[1] = (coord[1] - yshift) / yscale;
        polyTCoords->SetTuple(i, tcoord);
      }
    }
    if (textured)
    {
      polyTCoords->Modified();
    }
    polyPoints->Modified();
  }
  else // no polygon to render
  {
    return;
  }

  if (textured)
  {
    // no-op in the Metal backend (vtkMetalTexture); keeps the input pipeline
    // up to date for vtkMetalPolyDataMapper::UpdateActorTexture.
    actor->GetTexture()->Render(ren);
  }
  actor->GetMapper()->SetClippingPlanes(this->GetClippingPlanes());
  actor->GetMapper()->Render(ren, actor);
  if (textured)
  {
    actor->GetTexture()->PostRender(ren);
  }
}

//------------------------------------------------------------------------------
// Render a wide black border around the polygon, wide enough to fill
// the entire viewport.
void vtkMetalImageSliceMapper::RenderBackground(
  vtkActor* actor, vtkPoints* points, const int extent[6], vtkRenderer* ren)
{
  vtkPolyData* poly = vtkPolyDataMapper::SafeDownCast(actor->GetMapper())->GetInput();
  vtkPoints* polyPoints = poly->GetPoints();
  if (this->GetOutputPointsPrecision() == vtkAlgorithm::DOUBLE_PRECISION)
  {
    polyPoints->SetDataTypeToDouble();
  }
  vtkCellArray* tris = poly->GetPolys();

  static double borderThickness = 1e6;
  int xdim, ydim;
  vtkImageSliceMapper::GetDimensionIndices(this->Orientation, xdim, ydim);

  if (!points)
  {
    double coords[15], tcoords[10], center[3];
    this->MakeTextureGeometry(extent, coords, tcoords);
    coords[12] = coords[0];
    coords[13] = coords[1];
    coords[14] = coords[2];

    center[0] = 0.25 * (coords[0] + coords[3] + coords[6] + coords[9]);
    center[1] = 0.25 * (coords[1] + coords[4] + coords[7] + coords[10]);
    center[2] = 0.25 * (coords[2] + coords[5] + coords[8] + coords[11]);

    // render 4 sides
    tris->Initialize();
    polyPoints->SetNumberOfPoints(10);
    for (int side = 0; side < 4; side++)
    {
      tris->InsertNextCell(3);
      tris->InsertCellPoint(side);
      tris->InsertCellPoint(side + 5);
      tris->InsertCellPoint(side + 1);
      tris->InsertNextCell(3);
      tris->InsertCellPoint(side + 1);
      tris->InsertCellPoint(side + 5);
      tris->InsertCellPoint(side + 6);
    }

    for (int side = 0; side < 5; side++)
    {
      polyPoints->SetPoint(side, coords[3 * side], coords[3 * side + 1], coords[3 * side + 2]);

      double dx = coords[3 * side + xdim] - center[xdim];
      double sx = (dx >= 0 ? 1 : -1);
      double dy = coords[3 * side + ydim] - center[ydim];
      double sy = (dy >= 0 ? 1 : -1);
      coords[3 * side + xdim] += borderThickness * sx;
      coords[3 * side + ydim] += borderThickness * sy;

      polyPoints->SetPoint(side + 5, coords[3 * side], coords[3 * side + 1], coords[3 * side + 2]);
    }
  }
  else if (points->GetNumberOfPoints())
  {
    vtkIdType ncoords = points->GetNumberOfPoints();
    double coord[3], coord1[3];

    points->GetPoint(ncoords - 1, coord1);
    points->GetPoint(0, coord);
    double dx0 = coord[0] - coord1[0];
    double dy0 = coord[1] - coord1[1];
    double r = sqrt(dx0 * dx0 + dy0 * dy0);
    dx0 /= r;
    dy0 /= r;

    tris->Initialize();
    polyPoints->SetNumberOfPoints(ncoords * 2 + 2);

    for (vtkIdType i = 0; i < ncoords; i++)
    {
      tris->InsertNextCell(3);
      tris->InsertCellPoint(i * 2);
      tris->InsertCellPoint(i * 2 + 1);
      tris->InsertCellPoint(i * 2 + 2);
      tris->InsertNextCell(3);
      tris->InsertCellPoint(i * 2 + 2);
      tris->InsertCellPoint(i * 2 + 1);
      tris->InsertCellPoint(i * 2 + 3);
    }

    for (vtkIdType i = 0; i <= ncoords; i++)
    {
      polyPoints->SetPoint(i * 2, coord);

      points->GetPoint(((i + 1) % ncoords), coord1);
      double dx1 = coord1[0] - coord[0];
      double dy1 = coord1[1] - coord[1];
      r = sqrt(dx1 * dx1 + dy1 * dy1);
      dx1 /= r;
      dy1 /= r;

      double t;
      if (fabs(dx0 + dx1) > fabs(dy0 + dy1))
      {
        t = (dy1 - dy0) / (dx0 + dx1);
      }
      else
      {
        t = (dx0 - dx1) / (dy0 + dy1);
      }
      coord[0] += (t * dx0 + dy0) * borderThickness;
      coord[1] += (t * dy0 - dx0) * borderThickness;

      polyPoints->SetPoint(i * 2 + 1, coord);

      coord[0] = coord1[0];
      coord[1] = coord1[1];
      dx0 = dx1;
      dy0 = dy1;
    }
  }
  else // no polygon to render
  {
    return;
  }

  polyPoints->Modified();
  tris->Modified();
  actor->GetMapper()->SetClippingPlanes(this->GetClippingPlanes());
  actor->GetMapper()->Render(ren, actor);
}

//------------------------------------------------------------------------------
void vtkMetalImageSliceMapper::ComputeTextureSize(
  const int extent[6], int& xdim, int& ydim, int imageSize[2], int textureSize[2])
{
  // find dimension indices that will correspond to the
  // columns and rows of the 2D texture
  vtkImageSliceMapper::GetDimensionIndices(this->Orientation, xdim, ydim);

  // compute the image dimensions
  imageSize[0] = (extent[xdim * 2 + 1] - extent[xdim * 2] + 1);
  imageSize[1] = (extent[ydim * 2 + 1] - extent[ydim * 2] + 1);

  textureSize[0] = imageSize[0];
  textureSize[1] = imageSize[1];
}

//------------------------------------------------------------------------------
// Determine if a given texture size is supported by the device.
// Every Metal GPU (Apple family and beyond) guarantees 2D textures of at least
// 16384x16384, which is far larger than any vtkImageSlice this mapper renders,
// so subdividing (RecursiveRenderTexturedPolygon) practically never triggers.
bool vtkMetalImageSliceMapper::TextureSizeOK(const int size[2], vtkRenderer* ren)
{
  (void)ren;
  constexpr int kMaxMetalTextureSize = 16384;
  return size[0] <= kMaxMetalTextureSize && size[1] <= kMaxMetalTextureSize;
}

//------------------------------------------------------------------------------
// Render the slice for hardware selection (cell-ID picking).
void vtkMetalImageSliceMapper::RenderForSelection(
  vtkRenderer* ren, vtkImageSlice* prop, vtkHardwareSelector* selector)
{
  vtkMetalRenderWindow* renWin = vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());
  vtkImageData* input = this->GetInput();

  if (!renWin || !input)
  {
    return;
  }

  vtkMetalHardwareSelector* metalSelector = vtkMetalHardwareSelector::SafeDownCast(selector);
  if (!metalSelector)
  {
    return;
  }

  const int propId = metalSelector->GetPropID(prop);
  if (propId < 0)
  {
    return;
  }
  id<MTLDevice> device = (id<MTLDevice>)renWin->GetMetalDevice();
  id<MTLLibrary> library = (id<MTLLibrary>)renWin->GetSharedShaderLibrary();
  id<MTLRenderCommandEncoder> encoder =
    (id<MTLRenderCommandEncoder>)renWin->GetCurrentRenderCommandEncoder();
  if (!device || !library || !encoder)
  {
    return;
  }

  this->Internals->EnsureDepthStates(device);

  // Build the selection pipeline once.
  if (!this->Internals->SelectionPipeline)
  {
    NSError* error = nil;
    id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_slice_selection_main"];
    id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragment_slice_selection_main"];
    if (!vertexFunc || !fragmentFunc)
    {
      vtkErrorMacro(<< "Failed to find slice selection shader functions");
      [vertexFunc release];
      [fragmentFunc release];
      return;
    }

    MTLRenderPipelineDescriptor* pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunc;
    pipelineDesc.fragmentFunction = fragmentFunc;

    MTLVertexDescriptor* vertexDesc = [[MTLVertexDescriptor alloc] init];
    vertexDesc.attributes[0].format = MTLVertexFormatFloat3;
    vertexDesc.attributes[0].offset = 0;
    vertexDesc.attributes[0].bufferIndex = 0;
    vertexDesc.attributes[1].format = MTLVertexFormatFloat2;
    vertexDesc.attributes[1].offset = sizeof(float) * 3;
    vertexDesc.attributes[1].bufferIndex = 0;
    vertexDesc.layouts[0].stride = sizeof(float) * 5;
    vertexDesc.layouts[0].stepRate = 1;
    vertexDesc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    pipelineDesc.vertexDescriptor = vertexDesc;
    [vertexDesc release];

    pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    // 8A: the IDs attachment is only present when the selection forced sampleCount == 1.
    pipelineDesc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;
    pipelineDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    pipelineDesc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    pipelineDesc.rasterSampleCount = 1;

    this->Internals->SelectionPipeline =
      [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    if (!this->Internals->SelectionPipeline)
    {
      vtkErrorMacro(<< "Slice selection pipeline: " << [[error localizedDescription] UTF8String]);
    }
    [pipelineDesc release];
    [vertexFunc release];
    [fragmentFunc release];

    if (!this->Internals->SelectionPipeline)
    {
      return;
    }
  }

  // Allocate the reusable GPU buffers.
  if (!this->Internals->SelectionSceneBuffer)
  {
    this->Internals->SelectionSceneBuffer =
      [device newBufferWithLength:vtkMetalCamera::GetSceneTransformsSize()
                          options:MTLResourceStorageModeShared];
  }
  if (!this->Internals->SelectionVertexBuffer)
  {
    this->Internals->SelectionVertexBuffer =
      [device newBufferWithLength:4 * sizeof(float) * 5 options:MTLResourceStorageModeShared];
  }
  if (!this->Internals->SelectionUniformBuffer)
  {
    this->Internals->SelectionUniformBuffer =
      [device newBufferWithLength:sizeof(float) * 8 options:MTLResourceStorageModeShared];
  }
  if (!this->Internals->SelectionSceneBuffer || !this->Internals->SelectionVertexBuffer ||
    !this->Internals->SelectionUniformBuffer)
  {
    return;
  }

  // Scene transforms: copy the camera's cached matrices, then replace the model
  // matrix with the prop's model-to-world matrix (transposed into Metal's
  // column-major layout, matching vtkMetalPolyDataMapper).
  vtkMetalCamera* cam = vtkMetalCamera::SafeDownCast(ren->GetActiveCamera());
  if (!cam)
  {
    return;
  }
  char* sceneBuf = static_cast<char*>([this->Internals->SelectionSceneBuffer contents]);
  memcpy(sceneBuf, cam->GetCachedSceneTransforms(), vtkMetalCamera::GetSceneTransformsSize());
  {
    vtkNew<vtkMatrix4x4> actorMatrix;
    prop->GetMatrix(actorMatrix);
    float* modelMat = reinterpret_cast<float*>(sceneBuf + 176);
    for (int col = 0; col < 4; ++col)
    {
      for (int row = 0; row < 4; ++row)
      {
        modelMat[col * 4 + row] = static_cast<float>(actorMatrix->GetElement(row, col));
      }
    }
  }

  int dims[3];
  input->GetDimensions(dims);

  float idDims[2];
  if (selector->GetFieldAssociation() == vtkDataObject::FIELD_ASSOCIATION_CELLS)
  {
    // Cell Picking: Reduce dims by 1
    // Ensure we don't go negative if dims are 0 or 1
    idDims[0] = (dims[0] > 1) ? (float)(dims[0] - 1) : 1.0f;
    idDims[1] = (dims[1] > 1) ? (float)(dims[1] - 1) : 1.0f;
  }
  else
  {
    // Point Picking (default): Use full dims
    idDims[0] = (float)dims[0];
    idDims[1] = (float)dims[1];
  }

  int wholeExt[6];
  input->GetExtent(wholeExt);

  int* dispExt = this->DisplayExtent;

  double origin[3];
  double spacing[3];
  input->GetOrigin(origin);
  input->GetSpacing(spacing);

  double zVal = origin[2] + (dispExt[4] * spacing[2]);

  // Subtract 0.5 from min and add 0.5 to max to cover the full pixel area.
  // This ensures the picking quad aligns perfectly with the rendered pixels.
  float xMin = (float)(origin[0] + (dispExt[0] - 0.5) * spacing[0]);
  float xMax = (float)(origin[0] + (dispExt[1] + 0.5) * spacing[0]);
  float yMin = (float)(origin[1] + (dispExt[2] - 0.5) * spacing[1]);
  float yMax = (float)(origin[1] + (dispExt[3] + 0.5) * spacing[1]);

  // Map the visible "Display Extent" onto the quad.
  // Total dimensions of the full input image
  float wholeWidth = (float)(wholeExt[1] - wholeExt[0] + 1);
  float wholeHeight = (float)(wholeExt[3] - wholeExt[2] + 1);

  // Avoid division by zero
  wholeWidth = (wholeWidth < 1.0f) ? 1.0f : wholeWidth;
  wholeHeight = (wholeHeight < 1.0f) ? 1.0f : wholeHeight;

  float uMin = (float)(dispExt[0] - wholeExt[0]) / wholeWidth;
  float uMax = (float)(dispExt[1] - wholeExt[0] + 1) / wholeWidth;
  float vMin = (float)(dispExt[2] - wholeExt[2]) / wholeHeight;
  float vMax = (float)(dispExt[3] - wholeExt[2] + 1) / wholeHeight;

  // interleaved {position.xyz, texCoord.xy}, BL, BR, TL, TR
  const float verts[] = { xMin, yMin, (float)zVal, uMin, vMin,  xMax, yMin, (float)zVal,
    uMax, vMin, xMin, yMax, (float)zVal, uMin, vMax, xMax, yMax, (float)zVal, uMax, vMax };
  memcpy([this->Internals->SelectionVertexBuffer contents], verts, sizeof(verts));

  float* uniformBuf = static_cast<float*>([this->Internals->SelectionUniformBuffer contents]);
  uniformBuf[0] = idDims[0];
  uniformBuf[1] = idDims[1];
  uniformBuf[2] = static_cast<float>(propId);
  uniformBuf[3] = 0.0f; // composite index

  // Update selector with number of points/cells.
  int* inputExtent = this->GetInput()->GetExtent();
  unsigned int const numVoxels = (inputExtent[1] - inputExtent[0] + 1) *
    (inputExtent[3] - inputExtent[2] + 1) * (inputExtent[5] - inputExtent[4] + 1);
  selector->UpdateMaximumPointId(numVoxels);
  selector->UpdateMaximumCellId(numVoxels);

  [encoder setRenderPipelineState:this->Internals->SelectionPipeline];
  [encoder setDepthStencilState:this->Internals->ReadOnlyDepthState];
  [encoder setCullMode:MTLCullModeNone];
  [encoder setVertexBuffer:this->Internals->SelectionVertexBuffer offset:0 atIndex:0];
  [encoder setVertexBuffer:this->Internals->SelectionSceneBuffer offset:0 atIndex:1];
  [encoder setFragmentBuffer:this->Internals->SelectionUniformBuffer offset:0 atIndex:2];
  [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
}

//------------------------------------------------------------------------------
// Set the modelview transform and load the texture
void vtkMetalImageSliceMapper::Render(vtkRenderer* ren, vtkImageSlice* prop)
{
  vtkHardwareSelector* selector = ren->GetSelector();

  // The Metal selector renders IDs in a single pass (the RGBA32Uint color
  // attachment 1), so gate on the field association rather than the current
  // pass as vtkOpenGLImageSliceMapper does.
  if (selector != nullptr &&
    (selector->GetFieldAssociation() == vtkDataObject::FIELD_ASSOCIATION_CELLS ||
      selector->GetFieldAssociation() == vtkDataObject::FIELD_ASSOCIATION_POINTS))
  {
    // Hardware selection: render cell/point IDs using texture coordinates
    this->RenderForSelection(ren, prop, selector);
    // we are in selection mode, do not render anything else
    return;
  }

  // update the input information
  vtkImageData* input = this->GetInput();
  if (!input)
  {
    return;
  }
  input->GetSpacing(this->DataSpacing);
  vtkMatrix3x3::DeepCopy(this->DataDirection, input->GetDirectionMatrix());
  input->GetOrigin(this->DataOrigin);
  vtkInformation* inputInfo = this->GetInputInformation(0, 0);
  inputInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), this->DataWholeExtent);

  vtkMatrix4x4* matrix = this->GetDataToWorldMatrix();
  this->PolyDataActor->SetUserMatrix(matrix);
  this->BackingPolyDataActor->SetUserMatrix(matrix);
  this->BackgroundPolyDataActor->SetUserMatrix(matrix);
  if (prop->GetPropertyKeys())
  {
    this->PolyDataActor->SetPropertyKeys(prop->GetPropertyKeys());
    this->BackingPolyDataActor->SetPropertyKeys(prop->GetPropertyKeys());
    this->BackgroundPolyDataActor->SetPropertyKeys(prop->GetPropertyKeys());
  }

  // Depth write control (Metal equivalent of glDepthMask). The Metal renderer
  // binds LessEqual/write=YES for the opaque pass, so only swap the depth state
  // when the slice must not write depth, and restore it afterward.
  vtkMetalRenderWindow* renWin = vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());
  id<MTLRenderCommandEncoder> encoder = nullptr;
  id<MTLDevice> device = renWin ? (id<MTLDevice>)renWin->GetMetalDevice() : nullptr;
  bool boundReadOnly = false;
  if (renWin && device)
  {
    encoder = (id<MTLRenderCommandEncoder>)renWin->GetCurrentRenderCommandEncoder();
    if (!this->DepthEnable)
    {
      this->Internals->EnsureDepthStates(device);
      if (encoder)
      {
        [encoder setDepthStencilState:this->Internals->ReadOnlyDepthState];
        boundReadOnly = true;
      }
    }
  }

  // color and lighting related items
  vtkImageProperty* property = prop->GetProperty();
  double opacity = property->GetOpacity();
  double ambient = property->GetAmbient();
  double diffuse = property->GetDiffuse();
  vtkProperty* pdProp = this->PolyDataActor->GetProperty();
  pdProp->SetOpacity(opacity);
  pdProp->SetAmbient(ambient);
  pdProp->SetDiffuse(diffuse);

  // render the backing polygon
  int backing = property->GetBacking();
  double* bcolor = property->GetBackingColor();
  if (backing && (this->MatteEnable || (this->DepthEnable && !this->ColorEnable)))
  {
    // the backing polygon is always opaque
    pdProp = this->BackingPolyDataActor->GetProperty();
    pdProp->SetOpacity(1.0);
    pdProp->SetAmbient(ambient);
    pdProp->SetDiffuse(diffuse);
    pdProp->SetColor(bcolor[0], bcolor[1], bcolor[2]);
    this->RenderPolygon(this->BackingPolyDataActor, this->Points, this->DisplayExtent, ren);
    if (this->Background)
    {
      double bkcolor[4];
      this->GetBackgroundColor(property, bkcolor);
      pdProp = this->BackgroundPolyDataActor->GetProperty();
      pdProp->SetOpacity(1.0);
      pdProp->SetAmbient(ambient);
      pdProp->SetDiffuse(diffuse);
      pdProp->SetColor(bkcolor[0], bkcolor[1], bkcolor[2]);
      this->RenderBackground(this->BackgroundPolyDataActor, this->Points, this->DisplayExtent, ren);
    }
  }

  // render the texture. A depth-only pass (GL masks the color buffer off) has
  // no Metal equivalent since color writes are fixed per pipeline, so the
  // textured polygon is always rendered with color enabled here.
  if (this->ColorEnable || (!backing && this->DepthEnable))
  {
    this->RecursiveRenderTexturedPolygon(ren, property, input, this->DisplayExtent, false);
  }

  // restore the opaque depth state
  if (boundReadOnly && encoder)
  {
    [encoder setDepthStencilState:this->Internals->OpaqueDepthState];
  }

  this->TimeToDraw = 0.0001;
}

//------------------------------------------------------------------------------
void vtkMetalImageSliceMapper::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

VTK_ABI_NAMESPACE_END
