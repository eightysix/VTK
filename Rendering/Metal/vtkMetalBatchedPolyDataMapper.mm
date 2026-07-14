// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalBatchedPolyDataMapper.h"

#include "vtkMetalRenderWindow.h"
#include "vtkMetalRenderer.h"
#include "vtkMetalCamera.h"
#include "vtkObjectFactory.h"
#include "vtkCompositePolyDataMapper.h"
#include "vtkPolyData.h"
#include "vtkCellArray.h"
#include "vtkCellData.h"
#include "vtkPointData.h"
#include "vtkProperty.h"
#include "vtkActor.h"
#include "vtkRenderer.h"
#include "vtkFloatArray.h"
#include "vtkUnsignedCharArray.h"

#import <Metal/Metal.h>

#include <vector>
#include <map>

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalBatchedPolyDataMapper);

//------------------------------------------------------------------------------
vtkMetalBatchedPolyDataMapper::vtkMetalBatchedPolyDataMapper() = default;

//------------------------------------------------------------------------------
vtkMetalBatchedPolyDataMapper::~vtkMetalBatchedPolyDataMapper() = default;

//------------------------------------------------------------------------------
void vtkMetalBatchedPolyDataMapper::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Parent: " << this->Parent << '\n';
  os << indent << "Batch size: " << this->VTKPolyDataToBatchElement.size() << '\n';
}

//------------------------------------------------------------------------------
void vtkMetalBatchedPolyDataMapper::AddBatchElement(
  unsigned int flatIndex, BatchElement&& element)
{
  auto address = reinterpret_cast<std::uintptr_t>(element.PolyData);
  auto found = this->VTKPolyDataToBatchElement.find(address);

  this->FlatIndexToPolyData[flatIndex] = address;

  if (found == this->VTKPolyDataToBatchElement.end())
  {
    this->VTKPolyDataToBatchElement[address] =
      std::unique_ptr<BatchElement>(new BatchElement(std::move(element)));
    this->VTKPolyDataToBatchElement[address]->Marked = true;
    this->GeometryDirty = true;
    this->Modified();
  }
  else
  {
    auto& batchElement = found->second;
    batchElement->FlatIndex = flatIndex;
    batchElement->Marked = true;
  }
}

//------------------------------------------------------------------------------
vtkMetalBatchedPolyDataMapper::BatchElement*
vtkMetalBatchedPolyDataMapper::GetBatchElement(vtkPolyData* polydata)
{
  auto address = reinterpret_cast<std::uintptr_t>(polydata);
  auto found = this->VTKPolyDataToBatchElement.find(address);
  if (found != this->VTKPolyDataToBatchElement.end())
  {
    return found->second.get();
  }
  return nullptr;
}

//------------------------------------------------------------------------------
void vtkMetalBatchedPolyDataMapper::ClearBatchElements()
{
  this->VTKPolyDataToBatchElement.clear();
  this->FlatIndexToPolyData.clear();
  this->BatchPropertiesBuffer = nullptr;
  this->BatchPropertiesBufferSize = 0;
  this->GeometryDirty = true;
  this->Modified();
}

//------------------------------------------------------------------------------
std::vector<vtkPolyData*> vtkMetalBatchedPolyDataMapper::GetRenderedList() const
{
  std::vector<vtkPolyData*> result;
  result.reserve(this->VTKPolyDataToBatchElement.size());
  for (const auto& iter : this->VTKPolyDataToBatchElement)
  {
    result.emplace_back(iter.second->PolyData);
  }
  return result;
}

//------------------------------------------------------------------------------
void vtkMetalBatchedPolyDataMapper::SetParent(vtkCompositePolyDataMapper* parent)
{
  this->Parent = parent;
  this->SetInputDataObject(0, parent->GetInputDataObject(0, 0));
}

//------------------------------------------------------------------------------
void vtkMetalBatchedPolyDataMapper::UnmarkBatchElements()
{
  for (auto& iter : this->VTKPolyDataToBatchElement)
  {
    iter.second->Marked = false;
  }
}

//------------------------------------------------------------------------------
void vtkMetalBatchedPolyDataMapper::ClearUnmarkedBatchElements()
{
  for (auto iter = this->VTKPolyDataToBatchElement.begin();
       iter != this->VTKPolyDataToBatchElement.end();)
  {
    if (!iter->second->Marked)
    {
      this->VTKPolyDataToBatchElement.erase(iter++);
      this->GeometryDirty = true;
      this->Modified();
    }
    else
    {
      ++iter;
    }
  }
}

//------------------------------------------------------------------------------
vtkMTimeType vtkMetalBatchedPolyDataMapper::GetMTime()
{
  if (this->Parent)
  {
    return std::max(this->Superclass::GetMTime(), this->Parent->GetMTime());
  }
  return this->Superclass::GetMTime();
}

//------------------------------------------------------------------------------
void vtkMetalBatchedPolyDataMapper::ReleaseGraphicsResources(vtkWindow* w)
{
  // Release our batch-specific buffer (cast from void* to id<MTLBuffer>)
  if (this->BatchPropertiesBuffer)
  {
    CFRelease(this->BatchPropertiesBuffer);
    this->BatchPropertiesBuffer = nullptr;
  }
  this->BatchPropertiesBufferSize = 0;
  this->ResourcesSyncTimeStamp = vtkTimeStamp();
  this->GeometryDirty = true;
  this->Superclass::ReleaseGraphicsResources(w);
}

//------------------------------------------------------------------------------
void vtkMetalBatchedPolyDataMapper::UpdateBatchPropertiesBuffer(void* mtlDevice)
{
  if (this->VTKPolyDataToBatchElement.empty())
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;
  const size_t batchSize = this->VTKPolyDataToBatchElement.size();
  const size_t bufferSize = batchSize * AlignedPropertiesSize;

  if (this->BatchPropertiesBufferSize != bufferSize)
  {
    // Release old buffer
    if (this->BatchPropertiesBuffer)
    {
      CFRelease(this->BatchPropertiesBuffer);
    }
    id<MTLBuffer> newBuf = [device
      newBufferWithLength:bufferSize
                 options:MTLResourceStorageModeShared];
    this->BatchPropertiesBuffer = (__bridge void*)newBuf;
    CFRetain(this->BatchPropertiesBuffer);
    this->BatchPropertiesBufferSize = bufferSize;
  }

  id<MTLBuffer> propBuffer = (__bridge id<MTLBuffer>)this->BatchPropertiesBuffer;
  char* buf = static_cast<char*>([propBuffer contents]);
  uint32_t cellIdOffsetForSelector = 0;
  uint32_t cellIdOffsetForVerts = 0;
  uint32_t cellIdOffsetForLines = 0;
  uint32_t cellIdOffsetForPolys = 0;

  for (const auto& iter : this->VTKPolyDataToBatchElement)
  {
    const auto& batchElement = iter.second;
    if (!batchElement->Visibility)
    {
      continue;
    }

    CompositeDataProperties props = {};
    props.ApplyOverrideColors = batchElement->OverridesColor ? 1u : 0u;
    props.Opacity = static_cast<float>(batchElement->Opacity);
    props.CompositeId = batchElement->FlatIndex;
    props.Pickable = batchElement->Pickability ? 1u : 0u;
    props.Ambient[0] = static_cast<float>(batchElement->AmbientColor[0]);
    props.Ambient[1] = static_cast<float>(batchElement->AmbientColor[1]);
    props.Ambient[2] = static_cast<float>(batchElement->AmbientColor[2]);
    props.Diffuse[0] = static_cast<float>(batchElement->DiffuseColor[0]);
    props.Diffuse[1] = static_cast<float>(batchElement->DiffuseColor[1]);
    props.Diffuse[2] = static_cast<float>(batchElement->DiffuseColor[2]);

    vtkPolyData* pd = batchElement->PolyData;
    props.CellIdOffsetForVerts = cellIdOffsetForVerts;
    cellIdOffsetForVerts +=
      static_cast<uint32_t>(pd->GetNumberOfLines() + pd->GetNumberOfPolys());

    props.CellIdOffsetForLines = cellIdOffsetForLines;
    cellIdOffsetForLines +=
      static_cast<uint32_t>(pd->GetNumberOfVerts() + pd->GetNumberOfPolys());

    props.CellIdOffsetForPolys = cellIdOffsetForPolys;
    cellIdOffsetForPolys +=
      static_cast<uint32_t>(pd->GetNumberOfVerts() + pd->GetNumberOfLines());

    props.CellIdOffsetForSelector = cellIdOffsetForSelector;
    cellIdOffsetForSelector += static_cast<uint32_t>(pd->GetNumberOfCells());

    memcpy(buf, &props, sizeof(CompositeDataProperties));
    buf += AlignedPropertiesSize;
  }

  this->ResourcesSyncTimeStamp.Modified();
}

//------------------------------------------------------------------------------
void vtkMetalBatchedPolyDataMapper::BuildBatchedGeometryBuffers(void* mtlDevice)
{
  if (this->VTKPolyDataToBatchElement.empty())
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;

  // Accumulate geometry from all visible batch elements into combined arrays
  std::vector<float> allPositions;
  std::vector<float> allNormals;
  std::vector<float> allColors;
  std::vector<uint32_t> allTriIndices;
  std::vector<uint32_t> allLineIndices;

  for (const auto& iter : this->VTKPolyDataToBatchElement)
  {
    const auto& batchElement = iter.second;
    if (!batchElement->Visibility || !batchElement->PolyData)
    {
      continue;
    }

    vtkPolyData* pd = batchElement->PolyData;

    // Copy points as individual triangle vertices (fan-triangulated)
    vtkCellArray* polys = pd->GetPolys();
    if (polys && polys->GetNumberOfCells() > 0)
    {
      const vtkIdType* pts = nullptr;
      vtkIdType npts = 0;
      polys->InitTraversal();
      while (polys->GetNextCell(npts, pts))
      {
        for (vtkIdType i = 1; i + 1 < npts; ++i)
        {
          for (int j = 0; j < 3; ++j)
          {
            vtkIdType ptId = (j == 0) ? pts[0] : (j == 1 ? pts[i] : pts[i + 1]);
            double pt[3];
            pd->GetPoint(ptId, pt);
            allPositions.push_back(static_cast<float>(pt[0]));
            allPositions.push_back(static_cast<float>(pt[1]));
            allPositions.push_back(static_cast<float>(pt[2]));

            double normal[3] = { 0.0, 0.0, 1.0 };
            if (pd->GetPointData()->GetNormals())
            {
              pd->GetPointData()->GetNormals()->GetTuple(ptId, normal);
            }
            allNormals.push_back(static_cast<float>(normal[0]));
            allNormals.push_back(static_cast<float>(normal[1]));
            allNormals.push_back(static_cast<float>(normal[2]));

            allColors.push_back(1.0f);
            allColors.push_back(1.0f);
            allColors.push_back(1.0f);
            allColors.push_back(1.0f);
          }
          uint32_t base = static_cast<uint32_t>(allPositions.size() / 3 - 3);
          allTriIndices.push_back(base);
          allTriIndices.push_back(base + 1);
          allTriIndices.push_back(base + 2);
        }
      }
    }

    // Copy line segments
    vtkCellArray* lines = pd->GetLines();
    if (lines && lines->GetNumberOfCells() > 0)
    {
      const vtkIdType* pts = nullptr;
      vtkIdType npts = 0;
      lines->InitTraversal();
      while (lines->GetNextCell(npts, pts))
      {
        for (vtkIdType i = 0; i + 1 < npts; ++i)
        {
          for (int j = 0; j < 2; ++j)
          {
            vtkIdType ptId = pts[i + j];
            double pt[3];
            pd->GetPoint(ptId, pt);
            allPositions.push_back(static_cast<float>(pt[0]));
            allPositions.push_back(static_cast<float>(pt[1]));
            allPositions.push_back(static_cast<float>(pt[2]));

            allNormals.push_back(0.0f);
            allNormals.push_back(0.0f);
            allNormals.push_back(1.0f);

            allColors.push_back(1.0f);
            allColors.push_back(1.0f);
            allColors.push_back(1.0f);
            allColors.push_back(1.0f);
          }
          uint32_t base = static_cast<uint32_t>(allPositions.size() / 3 - 2);
          allLineIndices.push_back(base);
          allLineIndices.push_back(base + 1);
        }
      }
    }
  }

  if (allPositions.empty())
  {
    this->GeometryDirty = false;
    return;
  }

  // Create Metal buffers — use parent's protected methods by delegating
  // Since Internals is private, we store our own buffers and use the
  // parent's SetInputData/RenderPiece path instead.
  // For now, just mark as clean — actual buffer creation happens via
  // the parent's own geometry pipeline when we set input per mesh.
  this->GeometryDirty = false;
}

//------------------------------------------------------------------------------
void vtkMetalBatchedPolyDataMapper::RenderPiece(vtkRenderer* ren, vtkActor* act)
{
  vtkMetalRenderWindow* renWin = vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());
  if (!renWin || !renWin->GetMetalDevice())
  {
    return;
  }

  if (this->VTKPolyDataToBatchElement.empty())
  {
    return;
  }

  // Rebuild batch properties if needed
  vtkMTimeType currentMTime = this->GetMTime();
  if (currentMTime != this->ResourcesSyncTimeStamp)
  {
    this->UpdateBatchPropertiesBuffer(renWin->GetMetalDevice());
  }

  // Render each mesh in the batch individually via the parent class.
  // Each call to SetInputData + Superclass::RenderPiece handles geometry
  // building, pipeline state, and draw calls. The BatchPropertiesBuffer
  // provides per-mesh offsets for cell ID picking.
  this->CurrentDrawMeshId = 0;
  for (auto& iter : this->VTKPolyDataToBatchElement)
  {
    const auto& batchElement = iter.second;
    if (!batchElement->Visibility)
    {
      this->CurrentDrawMeshId++;
      continue;
    }

    // Temporarily set the input to this batch element's polydata
    this->SetInputData(batchElement->PolyData);

    // Delegate to parent class for actual rendering
    this->Superclass::RenderPiece(ren, act);

    this->CurrentDrawMeshId++;
  }
}

VTK_ABI_NAMESPACE_END
