// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalProperty.h"

#include "vtkObjectFactory.h"
#include "vtkActor.h"

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalProperty);

//------------------------------------------------------------------------------
vtkMetalProperty::vtkMetalProperty() = default;

//------------------------------------------------------------------------------
vtkMetalProperty::~vtkMetalProperty() = default;

//------------------------------------------------------------------------------
void vtkMetalProperty::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//------------------------------------------------------------------------------
void vtkMetalProperty::Render(vtkActor*)
{
  // Material uniforms are pushed by the mapper during draw calls
  // This is called before the mapper's RenderPiece
}

//------------------------------------------------------------------------------
void vtkMetalProperty::BackfaceRender(vtkActor*)
{
  // No special backface handling needed for minimal impl
}

//------------------------------------------------------------------------------
void vtkMetalProperty::PostRender(vtkActor*)
{
  // No post-render work needed
}

VTK_ABI_NAMESPACE_END
