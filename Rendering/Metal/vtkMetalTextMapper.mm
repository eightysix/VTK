// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalTextMapper.h"

#include "vtkObjectFactory.h"
#include "vtkOverrideAttribute.h"

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalTextMapper);

//------------------------------------------------------------------------------
vtkOverrideAttribute* vtkMetalTextMapper::CreateOverrideAttributes()
{
  return vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "Metal", nullptr);
}

//------------------------------------------------------------------------------
vtkMetalTextMapper::vtkMetalTextMapper() = default;

//------------------------------------------------------------------------------
vtkMetalTextMapper::~vtkMetalTextMapper() = default;

//------------------------------------------------------------------------------
void vtkMetalTextMapper::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

VTK_ABI_NAMESPACE_END
