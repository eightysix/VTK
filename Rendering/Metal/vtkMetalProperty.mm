// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalProperty.h"
#include "vtkObjectFactory.h"
#include "vtkOverrideAttribute.h"

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalProperty);

//------------------------------------------------------------------------------
vtkOverrideAttribute* vtkMetalProperty::CreateOverrideAttributes()
{
  return vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "Metal", nullptr);
}

vtkMetalProperty::vtkMetalProperty() = default;
vtkMetalProperty::~vtkMetalProperty() = default;

void vtkMetalProperty::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

void vtkMetalProperty::Render(vtkActor*, vtkRenderer*)
{
}

void vtkMetalProperty::BackfaceRender(vtkActor*, vtkRenderer*)
{
}

void vtkMetalProperty::PostRender(vtkActor*, vtkRenderer*)
{
}

VTK_ABI_NAMESPACE_END
