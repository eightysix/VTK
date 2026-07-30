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
#include "vtkMetalPolyDataMapper.h"

#import <Metal/Metal.h>

#include <vector>
#include <map>
#include <algorithm>
#include <cmath>

#include "vtkMetalMRC.h"

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalBatchedPolyDataMapper);

//------------------------------------------------------------------------------
vtkMetalBatchedPolyDataMapper::vtkMetalBatchedPolyDataMapper() = default;

//------------------------------------------------------------------------------
vtkMetalBatchedPolyDataMapper::~vtkMetalBatchedPolyDataMapper()
{
  this->ReleaseChildMappers(nullptr);
}

//------------------------------------------------------------------------------
vtkSmartPointer<vtkMetalPolyDataMapper>
vtkMetalBatchedPolyDataMapper::GetChildMapper(vtkPolyData* polydata)
{
  auto address = reinterpret_cast<std::uintptr_t>(polydata);

  auto it = this->ChildMappers.find(address);
  if (it == this->ChildMappers.end())
  {
    vtkSmartPointer<vtkMetalPolyDataMapper> mapper =
      vtkSmartPointer<vtkMetalPolyDataMapper>::New();

    mapper->SetInputData(polydata);
    this->ConfigureChildMapper(mapper);
    this->ChildMappers[address] = mapper;
    return mapper;
  }

  if (it->second->GetInputDataObject(0, 0) != polydata)
  {
    it->second->SetInputData(polydata);
  }

  return it->second;
}

//------------------------------------------------------------------------------
void vtkMetalBatchedPolyDataMapper::ReleaseChildMappers(vtkWindow* w)
{
  for (auto& kv : this->ChildMappers)
  {
    if (kv.second)
    {
      kv.second->ReleaseGraphicsResources(w);
    }
  }

  this->ChildMappers.clear();
}

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

  // Remove stale FlatIndexToPolyData entries pointing to the same address
  for (auto it = this->FlatIndexToPolyData.begin();
       it != this->FlatIndexToPolyData.end();)
  {
    if (it->second == address && it->first != flatIndex)
    {
      it = this->FlatIndexToPolyData.erase(it);
    }
    else
    {
      ++it;
    }
  }

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
    auto updated = std::make_unique<BatchElement>(std::move(element));
    updated->Marked = true;
    found->second = std::move(updated);
    this->GeometryDirty = true;
    this->Modified();
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
  this->ReleaseChildMappers(nullptr);

  this->VTKPolyDataToBatchElement.clear();
  this->FlatIndexToPolyData.clear();

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
  bool changed = false;

  for (auto iter = this->VTKPolyDataToBatchElement.begin();
       iter != this->VTKPolyDataToBatchElement.end();)
  {
    if (!iter->second->Marked)
    {
      auto childIt = this->ChildMappers.find(iter->first);
      if (childIt != this->ChildMappers.end())
      {
        if (childIt->second)
        {
          childIt->second->ReleaseGraphicsResources(nullptr);
        }
        this->ChildMappers.erase(childIt);
      }

      iter = this->VTKPolyDataToBatchElement.erase(iter);
      changed = true;
    }
    else
    {
      ++iter;
    }
  }

  // Clean stale flat-index mappings
  for (auto fit = this->FlatIndexToPolyData.begin();
       fit != this->FlatIndexToPolyData.end();)
  {
    if (this->VTKPolyDataToBatchElement.find(fit->second) ==
        this->VTKPolyDataToBatchElement.end())
    {
      fit = this->FlatIndexToPolyData.erase(fit);
    }
    else
    {
      ++fit;
    }
  }

  if (changed)
  {
    this->GeometryDirty = true;
    this->Modified();
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
  this->ReleaseChildMappers(w);

  this->GeometryDirty = true;

  this->Superclass::ReleaseGraphicsResources(w);
}

//------------------------------------------------------------------------------
void vtkMetalBatchedPolyDataMapper::ConfigureChildMapper(
    vtkMetalPolyDataMapper* child)
{
    child->SetScalarVisibility(this->GetScalarVisibility());
    child->SetScalarMode(this->GetScalarMode());
    child->SetColorMode(this->GetColorMode());
    child->SetInterpolateScalarsBeforeMapping(
        this->GetInterpolateScalarsBeforeMapping());
    child->SetLookupTable(this->GetLookupTable());

    // Remove any stale mappings that may exist from a previous configuration
    child->RemoveAllVertexAttributeMappings();

    // Copy extra attribute mappings
    for (const auto& attr : this->ExtraAttributes)
    {
        child->MapDataArrayToVertexAttribute(
            attr.first.c_str(),
            attr.second.DataArrayName.c_str(),
            attr.second.FieldAssociation,
            attr.second.ComponentNumber);
    }
}

//------------------------------------------------------------------------------
void vtkMetalBatchedPolyDataMapper::RenderPiece(vtkRenderer* ren, vtkActor* act)
{
  vtkMetalRenderWindow* renWin =
    vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());

  if (!renWin || !renWin->GetMetalDevice())
  {
    return;
  }

  if (this->VTKPolyDataToBatchElement.empty())
  {
    return;
  }

  // Propagate parent state changes to existing child mappers
  vtkMTimeType childConfigMTime =
      std::max(this->GetMTime(), this->ExtraAttributesMTime.GetMTime());
  if (childConfigMTime != this->CachedChildConfigurationMTime)
  {
    for (auto& kv : this->ChildMappers)
    {
      if (kv.second)
      {
        this->ConfigureChildMapper(kv.second);
      }
    }
    this->CachedChildConfigurationMTime = childConfigMTime;
  }

  // Render in flat-index order, not pointer-address order.
  std::vector<const BatchElement*> visible;
  visible.reserve(this->VTKPolyDataToBatchElement.size());

  for (const auto& kv : this->VTKPolyDataToBatchElement)
  {
    const BatchElement* elem = kv.second.get();
    if (elem && elem->Visibility && elem->PolyData)
    {
      visible.push_back(elem);
    }
  }

  std::sort(visible.begin(), visible.end(),
    [](const BatchElement* a, const BatchElement* b)
    {
      return a->FlatIndex < b->FlatIndex;
    });

  // Hoist actor-level properties that do not change per-batch-element.
  double actorOpacity = act->GetProperty()->GetOpacity();
  bool actorPickable = act->GetPickable();

  for (const BatchElement* elem : visible)
  {
    vtkSmartPointer<vtkMetalPolyDataMapper> mapper =
      this->GetChildMapper(elem->PolyData);

    if (!mapper)
    {
      continue;
    }

    // Apply batch visual overrides for this element.
    // BatchElement::Opacity is the final desired opacity for the block.
    // When OverridesColor is true, the block opacity is always applied along
    // with the override color (the element's diffuse color serves as both
    // color and opacity override source).
    // Detect whether opacity is explicitly overridden by comparing against the
    // actor's baseline opacity, so we don't spuriously override when the element
    // inherits the actor value.
    bool overrideColor = elem->OverridesColor;
    bool overrideOpacity =
      overrideColor ||
      (std::abs(elem->Opacity - actorOpacity) > 1e-12);

    // NOTE: Opacity-only overrides switch the shader from material colors
    // (ambientColor, diffuseColor) to vertex colors. For the common case
    // where ambient/diffuse/color are synchronized this is fine, but actors
    // with custom ambient/diffuse colors may see a slight lighting color
    // change when only opacity is overridden.

    if (overrideColor || overrideOpacity)
    {
      double overrideColorArr[3] = {
        elem->DiffuseColor[0],
        elem->DiffuseColor[1],
        elem->DiffuseColor[2]
      };
      mapper->SetBatchVisualOverride(
        overrideColor, overrideColorArr, overrideOpacity, elem->Opacity);
    }
    else
    {
      mapper->ClearBatchVisualOverride();
    }

    // Per-block picking ID.
    // FlatIndex comes from the composite dataset's block index. These IDs
    // are unique within a single batched mapper, but may collide with IDs
    // from non-batched actors (which use GetOrCreatePropId's global counter)
    // or from other batched mappers. If hardware picking is used across
    // multiple mappers, batch elements should allocate from a shared ID
    // space or encode the mapper ID separately.
    bool pickable = actorPickable && elem->Pickability;
    if (pickable)
    {
      mapper->SetOverridePropId(elem->FlatIndex);
    }
    else
    {
      mapper->SetOverridePropIdToNone();
    }

    mapper->RenderPiece(ren, act);
  }
}

VTK_ABI_NAMESPACE_END
