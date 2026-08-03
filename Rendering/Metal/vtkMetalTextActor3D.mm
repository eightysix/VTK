// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalTextActor3D.h"

#include "vtkObjectFactory.h"
#include "vtkOverrideAttribute.h"

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalTextActor3D);

//------------------------------------------------------------------------------
vtkOverrideAttribute* vtkMetalTextActor3D::CreateOverrideAttributes()
{
  return vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "Metal", nullptr);
}

//------------------------------------------------------------------------------
vtkMetalTextActor3D::vtkMetalTextActor3D() = default;

//------------------------------------------------------------------------------
vtkMetalTextActor3D::~vtkMetalTextActor3D() = default;

//------------------------------------------------------------------------------
void vtkMetalTextActor3D::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

VTK_ABI_NAMESPACE_END
