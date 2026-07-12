// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalCompositePolyDataMapperDelegator.h"
#include "vtkCompositePolyDataMapper.h"
#include "vtkObjectFactory.h"
#include "vtkOverrideAttribute.h"
#include "vtkMetalBatchedPolyDataMapper.h"

VTK_ABI_NAMESPACE_BEGIN

//------------------------------------------------------------------------------
vtkStandardNewMacro(vtkMetalCompositePolyDataMapperDelegator);

//------------------------------------------------------------------------------
vtkMetalCompositePolyDataMapperDelegator::vtkMetalCompositePolyDataMapperDelegator()
{
  this->MetalDelegate = vtkMetalBatchedPolyDataMapper::New();
  this->Delegate = vtk::TakeSmartPointer(this->MetalDelegate);
}

//------------------------------------------------------------------------------
vtkMetalCompositePolyDataMapperDelegator::~vtkMetalCompositePolyDataMapperDelegator() = default;

//------------------------------------------------------------------------------
vtkOverrideAttribute* vtkMetalCompositePolyDataMapperDelegator::CreateOverrideAttributes()
{
  auto* renderingBackendAttribute =
    vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "Metal", nullptr);
  return renderingBackendAttribute;
}

//------------------------------------------------------------------------------
void vtkMetalCompositePolyDataMapperDelegator::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//------------------------------------------------------------------------------
void vtkMetalCompositePolyDataMapperDelegator::ShallowCopy(vtkCompositePolyDataMapper* cpdm)
{
  this->Superclass::ShallowCopy(cpdm);
}

//------------------------------------------------------------------------------
void vtkMetalCompositePolyDataMapperDelegator::ClearUnmarkedBatchElements()
{
  this->MetalDelegate->ClearUnmarkedBatchElements();
}

//------------------------------------------------------------------------------
void vtkMetalCompositePolyDataMapperDelegator::UnmarkBatchElements()
{
  this->MetalDelegate->UnmarkBatchElements();
}

//------------------------------------------------------------------------------
std::vector<vtkPolyData*> vtkMetalCompositePolyDataMapperDelegator::GetRenderedList() const
{
  return this->MetalDelegate->GetRenderedList();
}

//------------------------------------------------------------------------------
void vtkMetalCompositePolyDataMapperDelegator::SetParent(vtkCompositePolyDataMapper* mapper)
{
  this->MetalDelegate->SetParent(mapper);
}

//------------------------------------------------------------------------------
void vtkMetalCompositePolyDataMapperDelegator::Insert(BatchElement&& batchElement)
{
  this->MetalDelegate->AddBatchElement(batchElement.FlatIndex, std::move(batchElement));
}

//------------------------------------------------------------------------------
vtkCompositePolyDataMapperDelegator::BatchElement*
vtkMetalCompositePolyDataMapperDelegator::Get(vtkPolyData* polydata)
{
  return this->MetalDelegate->GetBatchElement(polydata);
}

//------------------------------------------------------------------------------
void vtkMetalCompositePolyDataMapperDelegator::Clear()
{
  this->MetalDelegate->ClearBatchElements();
}

VTK_ABI_NAMESPACE_END
