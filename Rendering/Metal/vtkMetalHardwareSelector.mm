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

  // The render is submitted to the GPU asynchronously; wait for it to finish
  // before reading back the IDs texture (getBytes on a shared-storage texture
  // would otherwise race the GPU and yield stale or partial data).
  renWin->WaitForCompletion();

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

//------------------------------------------------------------------------------
vtkHardwareSelector::PixelInformation vtkMetalHardwareSelector::GetPixelInformation(
  const unsigned int inDisplayPosition[2], int maxDist, unsigned int outSelectedPosition[2])
{
  // Base case: single pixel, decoded from the single-pass ID buffer.
  const unsigned int maxDistance = (maxDist < 0) ? 0 : static_cast<unsigned int>(maxDist);
  if (maxDistance == 0)
  {
    outSelectedPosition[0] = inDisplayPosition[0];
    outSelectedPosition[1] = inDisplayPosition[1];
    if (inDisplayPosition[0] < this->Area[0] || inDisplayPosition[0] > this->Area[2] ||
      inDisplayPosition[1] < this->Area[1] || inDisplayPosition[1] > this->Area[3])
    {
      return PixelInformation();
    }

    const unsigned int displayPosition[2] = { inDisplayPosition[0] - this->Area[0],
      inDisplayPosition[1] - this->Area[1] };

    int actorId = this->Convert(displayPosition[0], displayPosition[1], this->PixBuffer[ACTOR_PASS]);
    if (actorId <= 0)
    {
      // the pixel did not hit any actor (the shader maps background to 0).
      return PixelInformation();
    }

    PixelInformation info;
    info.Valid = true;
    // The shader encodes the prop index as id + 1 (mapPropId), so undo the
    // offset to recover the 0-based index into PropArray.
    actorId -= 1;
    info.PropID = actorId;
    info.Prop = this->GetPropFromID(actorId);
    if (this->ActorPassOnly)
    {
      return info;
    }

    const int compositeId =
      this->Convert(displayPosition[0], displayPosition[1], this->PixBuffer[COMPOSITE_INDEX_PASS]);
    if (compositeId < 0 || compositeId > 0xffffff)
    {
      // the pixel did not hit any composite.
      return PixelInformation();
    }
    info.CompositeID = static_cast<unsigned int>(compositeId);

    // The shader encodes cell/point ids as id + 1, so undo the offset.
    const int attributeId =
      this->Convert(displayPosition[0], displayPosition[1], this->PixBuffer[CELL_ID_HIGH24]);
    if (attributeId > 0)
    {
      info.AttributeID = attributeId - 1;
    }

    // The Metal shaders always write 0 to the process-id channel, so leave
    // ProcessID unset (matching the base class "no process" semantics).
    return info;
  }

  // Iterate over successively growing boxes around the queried pixel.
  unsigned int dispPos[2] = { inDisplayPosition[0], inDisplayPosition[1] };
  unsigned int curPos[2] = { 0, 0 };
  PixelInformation info;
  info = this->GetPixelInformation(inDisplayPosition, 0, outSelectedPosition);
  if (info.Valid)
  {
    return info;
  }
  for (unsigned int dist = 1; dist < maxDistance; ++dist)
  {
    // Vertical sides of box.
    for (unsigned int y = ((dispPos[1] > dist) ? (dispPos[1] - dist) : 0);
         y <= dispPos[1] + dist; ++y)
    {
      curPos[1] = y;
      if (dispPos[0] >= dist)
      {
        curPos[0] = dispPos[0] - dist;
        info = this->GetPixelInformation(curPos, 0, outSelectedPosition);
        if (info.Valid)
        {
          return info;
        }
      }
      curPos[0] = dispPos[0] + dist;
      info = this->GetPixelInformation(curPos, 0, outSelectedPosition);
      if (info.Valid)
      {
        return info;
      }
    }
    // Horizontal sides of box.
    for (unsigned int x = ((dispPos[0] >= dist) ? (dispPos[0] - (dist - 1)) : 0);
         x <= dispPos[0] + (dist - 1); ++x)
    {
      curPos[0] = x;
      if (dispPos[1] >= dist)
      {
        curPos[1] = dispPos[1] - dist;
        info = this->GetPixelInformation(curPos, 0, outSelectedPosition);
        if (info.Valid)
        {
          return info;
        }
      }
      curPos[1] = dispPos[1] + dist;
      info = this->GetPixelInformation(curPos, 0, outSelectedPosition);
      if (info.Valid)
      {
        return info;
      }
    }
  }

  // nothing hit.
  outSelectedPosition[0] = inDisplayPosition[0];
  outSelectedPosition[1] = inDisplayPosition[1];
  return PixelInformation();
}

VTK_ABI_NAMESPACE_END
