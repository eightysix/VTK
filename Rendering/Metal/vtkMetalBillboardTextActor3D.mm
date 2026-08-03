// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalBillboardTextActor3D.h"

#include "vtkObjectFactory.h"
#include "vtkOverrideAttribute.h"

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalBillboardTextActor3D);

//------------------------------------------------------------------------------
vtkOverrideAttribute* vtkMetalBillboardTextActor3D::CreateOverrideAttributes()
{
  return vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "Metal", nullptr);
}

//------------------------------------------------------------------------------
vtkMetalBillboardTextActor3D::vtkMetalBillboardTextActor3D() = default;

//------------------------------------------------------------------------------
vtkMetalBillboardTextActor3D::~vtkMetalBillboardTextActor3D() = default;

//------------------------------------------------------------------------------
void vtkMetalBillboardTextActor3D::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

VTK_ABI_NAMESPACE_END
