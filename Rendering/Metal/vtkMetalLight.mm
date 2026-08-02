// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalLight.h"

#include "vtkObjectFactory.h"
#include "vtkOverrideAttribute.h"
#include "vtkLight.h"

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalLight);

//------------------------------------------------------------------------------
vtkOverrideAttribute* vtkMetalLight::CreateOverrideAttributes()
{
  return vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "Metal", nullptr);
}

//------------------------------------------------------------------------------
vtkMetalLight::vtkMetalLight() = default;

//------------------------------------------------------------------------------
vtkMetalLight::~vtkMetalLight() = default;

//------------------------------------------------------------------------------
void vtkMetalLight::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//------------------------------------------------------------------------------
void vtkMetalLight::Render(vtkRenderer*, int)
{
  // Light data is packed by the mapper into the LightUniforms buffer
  // during its draw call setup. No direct Metal calls needed here.
}

VTK_ABI_NAMESPACE_END
