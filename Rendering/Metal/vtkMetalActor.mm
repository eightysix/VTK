// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalActor.h"

#include "vtkObjectFactory.h"
#include "vtkMapper.h"
#include "vtkRenderer.h"

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalActor);

//------------------------------------------------------------------------------
vtkMetalActor::vtkMetalActor() = default;

//------------------------------------------------------------------------------
vtkMetalActor::~vtkMetalActor() = default;

//------------------------------------------------------------------------------
void vtkMetalActor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//------------------------------------------------------------------------------
void vtkMetalActor::Render(vtkRenderer* renderer, vtkMapper* mapper)
{
  // The mapper handles the actual Metal draw calls.
  // We just need to call the mapper's RenderPiece.
  if (mapper)
  {
    mapper->Render(renderer, this);
  }
}

//------------------------------------------------------------------------------
void vtkMetalActor::ReleaseGraphicsResources(vtkWindow* window)
{
  this->Superclass::ReleaseGraphicsResources(window);
}

VTK_ABI_NAMESPACE_END
