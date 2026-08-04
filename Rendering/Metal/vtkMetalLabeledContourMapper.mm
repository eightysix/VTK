// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalLabeledContourMapper.h"

#include "vtkActor.h"
#include "vtkMatrix4x4.h"
#include "vtkObjectFactory.h"
#include "vtkOverrideAttribute.h"
#include "vtkTextActor3D.h"

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalLabeledContourMapper);

//------------------------------------------------------------------------------
vtkOverrideAttribute* vtkMetalLabeledContourMapper::CreateOverrideAttributes()
{
  return vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "Metal", nullptr);
}

//------------------------------------------------------------------------------
vtkMetalLabeledContourMapper::vtkMetalLabeledContourMapper() = default;

//------------------------------------------------------------------------------
vtkMetalLabeledContourMapper::~vtkMetalLabeledContourMapper() = default;

//------------------------------------------------------------------------------
void vtkMetalLabeledContourMapper::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//------------------------------------------------------------------------------
bool vtkMetalLabeledContourMapper::CreateLabels(vtkActor* actor)
{
  if (!this->Superclass::CreateLabels(actor))
  {
    return false;
  }

  // The Metal backend passes each actor's matrix to the shader individually, so
  // a transform on this mapper's actor does not reach the labels (they are
  // independent vtkTextActor3D instances). Mirror the OpenGL override and fold
  // the actor's matrix into each label's user matrix.
  if (vtkMatrix4x4* actorMatrix = actor->GetMatrix())
  {
    for (vtkIdType i = 0; i < this->NumberOfUsedTextActors; ++i)
    {
      vtkMatrix4x4* labelMatrix = this->TextActors[i]->GetUserMatrix();
      vtkMatrix4x4::Multiply4x4(actorMatrix, labelMatrix, labelMatrix);
      this->TextActors[i]->SetUserMatrix(labelMatrix);
    }
  }

  return true;
}

VTK_ABI_NAMESPACE_END
