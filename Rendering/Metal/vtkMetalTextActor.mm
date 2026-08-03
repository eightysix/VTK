// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalTextActor.h"

#include "vtkObjectFactory.h"
#include "vtkOverrideAttribute.h"

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalTextActor);

//------------------------------------------------------------------------------
vtkOverrideAttribute* vtkMetalTextActor::CreateOverrideAttributes()
{
  return vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "Metal", nullptr);
}

//------------------------------------------------------------------------------
vtkMetalTextActor::vtkMetalTextActor() = default;

//------------------------------------------------------------------------------
vtkMetalTextActor::~vtkMetalTextActor() = default;

//------------------------------------------------------------------------------
void vtkMetalTextActor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

VTK_ABI_NAMESPACE_END
