// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalTexture.h"

#include "vtkObjectFactory.h"
#include "vtkOverrideAttribute.h"

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalTexture);

//------------------------------------------------------------------------------
vtkOverrideAttribute* vtkMetalTexture::CreateOverrideAttributes()
{
  return vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "Metal", nullptr);
}

vtkMetalTexture::vtkMetalTexture() = default;
vtkMetalTexture::~vtkMetalTexture() = default;

//------------------------------------------------------------------------------
void vtkMetalTexture::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//------------------------------------------------------------------------------
void vtkMetalTexture::Load(vtkRenderer*)
{
  // The Metal poly data mapper uploads and binds actor textures itself in
  // vtkMetalPolyDataMapper::UpdateActorTexture, so there is nothing to load.
}

//------------------------------------------------------------------------------
void vtkMetalTexture::PostRender(vtkRenderer*)
{
}

//------------------------------------------------------------------------------
void vtkMetalTexture::ReleaseGraphicsResources(vtkWindow*)
{
}

VTK_ABI_NAMESPACE_END
