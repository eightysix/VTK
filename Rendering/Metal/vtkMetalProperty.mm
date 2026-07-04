// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalProperty.h"
#include "vtkObjectFactory.h"

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalProperty);

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
