// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkWebGPUGPUInfoList.h"

#include "vtkGPUInfoListArray.h"
#include "vtkObjectFactory.h"
#include "vtkOverrideAttribute.h"
#include <cassert>

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkWebGPUGPUInfoList);

//------------------------------------------------------------------------------
void vtkWebGPUGPUInfoList::Probe()
{
  if (!this->Probed)
  {
    this->Probed = true;
    this->Array = new vtkGPUInfoListArray;
    this->Array->v.resize(0); // No GPU info available through this stub.
  }
  assert("post: probed" && this->IsProbed());
}

//------------------------------------------------------------------------------
vtkOverrideAttribute* vtkWebGPUGPUInfoList::CreateOverrideAttributes()
{
  auto* renderingBackendAttribute =
    vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "WebGPU", nullptr);
  return renderingBackendAttribute;
}

//------------------------------------------------------------------------------
vtkWebGPUGPUInfoList::vtkWebGPUGPUInfoList() = default;

//------------------------------------------------------------------------------
vtkWebGPUGPUInfoList::~vtkWebGPUGPUInfoList() = default;

//------------------------------------------------------------------------------
void vtkWebGPUGPUInfoList::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

VTK_ABI_NAMESPACE_END
