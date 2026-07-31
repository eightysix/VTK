// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalHardwareSelector.h"
#include "vtkMetalRenderWindow.h"

#include "vtkObjectFactory.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkUnsignedIntArray.h"
#include "vtkProp.h"
#include "vtkSelection.h"
#include "vtkSelectionNode.h"
#include "vtkIdTypeArray.h"
#include "vtkPointData.h"
#include "vtkCellData.h"
#include "vtkDataArray.h"

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalHardwareSelector);

//------------------------------------------------------------------------------
vtkMetalHardwareSelector::vtkMetalHardwareSelector() = default;

//------------------------------------------------------------------------------
vtkMetalHardwareSelector::~vtkMetalHardwareSelector()
{
  this->ReleasePixBuffers();
}

//------------------------------------------------------------------------------
void vtkMetalHardwareSelector::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//------------------------------------------------------------------------------
bool vtkMetalHardwareSelector::CaptureBuffers()
{
  if (!this->Renderer)
  {
    vtkErrorMacro("Renderer must be set before calling Select.");
    return false;
  }

  vtkMetalRenderWindow* renWin =
    vtkMetalRenderWindow::SafeDownCast(this->Renderer->GetRenderWindow());
  if (!renWin)
  {
    vtkErrorMacro("Cannot capture IDs: not a Metal render window.");
    return false;
  }

  // Hardware picking renders IDs to an RGBA32Uint attachment, which cannot be
  // multisampled. Force sampleCount == 1 for the duration of the selection
  // render so the IDs attachment is present and the readback is valid.
  // Mappers rebuild their pipelines at the forced sample count and rebuild
  // again when the original sample count is restored on the next regular frame.
  const int savedMultiSamples = renWin->GetMultiSamples();
  if (savedMultiSamples > 1)
  {
    renWin->SetMultiSamples(1);
  }

  this->BeginSelection();

  // Trigger a re-render so the IDs texture is populated.
  // For point field association, mappers check the selector and draw points.
  renWin->Render();

  // Read back the IDs texture into our buffer.
  renWin->GetIdsData(
    this->Area[0], this->Area[1], this->Area[2], this->Area[3], this->IdBuffer);

  this->BuildPropHitList(this->PixBuffer[ACTOR_PASS]);
  this->EndSelection();

  if (savedMultiSamples > 1)
  {
    renWin->SetMultiSamples(savedMultiSamples);
  }
  return true;
}

//------------------------------------------------------------------------------
void vtkMetalHardwareSelector::BeginSelection()
{
  this->Superclass::BeginSelection();

  // Build the visible prop array.
  vtkProp* aProp;
  int propArrayCount = 0;
  if (this->Renderer->GetViewProps()->GetNumberOfItems() > 0)
  {
    this->PropArray = new vtkProp*[this->Renderer->GetViewProps()->GetNumberOfItems()];
  }
  else
  {
    this->PropArray = nullptr;
  }

  vtkCollectionSimpleIterator pit;
  for (this->Renderer->GetViewProps()->InitTraversal(pit);
       (aProp = this->Renderer->GetViewProps()->GetNextProp(pit));)
  {
    if (aProp->GetVisibility())
    {
      this->PropArray[propArrayCount++] = aProp;
    }
  }

  // Initialize pixel buffers for each pass type.
  for (int i = MIN_KNOWN_PASS; i < MAX_KNOWN_PASS + 1; ++i)
  {
    unsigned char* iPtr = new unsigned char();
    *iPtr = i;
    this->PixBuffer[i] = iPtr;
  }

  this->PropCount = propArrayCount;
}

//------------------------------------------------------------------------------
void vtkMetalHardwareSelector::EndSelection()
{
  this->Superclass::EndSelection();
}

//------------------------------------------------------------------------------
vtkProp* vtkMetalHardwareSelector::GetPropFromID(int id)
{
  if (id >= 0 && this->PropArray && this->PropArray[id])
  {
    return this->PropArray[id];
  }
  return nullptr;
}

//------------------------------------------------------------------------------
int vtkMetalHardwareSelector::GetPropID(vtkProp* prop) const
{
  if (!prop || !this->PropArray)
  {
    return -1;
  }
  for (int i = 0; i < this->PropCount; ++i)
  {
    if (this->PropArray[i] == prop)
    {
      return i;
    }
  }
  return -1;
}

//------------------------------------------------------------------------------
void vtkMetalHardwareSelector::ReleasePixBuffers()
{
  this->IdBuffer->Reset();

  for (int i = MIN_KNOWN_PASS; i < MAX_KNOWN_PASS + 1; ++i)
  {
    delete this->PixBuffer[i];
    this->PixBuffer[i] = nullptr;
  }
  delete[] this->PropArray;
  this->PropArray = nullptr;
  this->PropCount = 0;
}

//------------------------------------------------------------------------------
int vtkMetalHardwareSelector::Convert(int xRelative, int yRelative, unsigned char* pb)
{
  if (!pb)
  {
    return 0;
  }
  if (this->IdBuffer->GetNumberOfValues() == 0)
  {
    vtkErrorMacro(<< "IDs are not captured!");
    return 0;
  }

  const int& x1 = this->Area[0];
  const int& x2 = this->Area[2];
  const int queryWidth = x2 - x1 + 1;
  const int pixelOffset = yRelative * queryWidth + xRelative;

  unsigned int rawIds[4] = {};
  this->IdBuffer->GetTypedTuple(pixelOffset, rawIds);

  const auto* ids = reinterpret_cast<const Ids*>(rawIds);

  if (*pb == vtkHardwareSelector::ACTOR_PASS)
  {
    return ids->PropId;
  }
  else if (*pb == vtkHardwareSelector::COMPOSITE_INDEX_PASS)
  {
    return ids->CompositeId;
  }
  else if (*pb == vtkHardwareSelector::POINT_ID_HIGH24 ||
           *pb == vtkHardwareSelector::POINT_ID_LOW24)
  {
    return ids->AttributeId;
  }
  else if (*pb == vtkHardwareSelector::PROCESS_PASS)
  {
    return ids->ProcessId;
  }
  else if (*pb == vtkHardwareSelector::CELL_ID_HIGH24 ||
           *pb == vtkHardwareSelector::CELL_ID_LOW24)
  {
    return ids->AttributeId;
  }
  return 0;
}

VTK_ABI_NAMESPACE_END
