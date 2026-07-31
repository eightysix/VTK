// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
//
// Shared verification helpers for the Metal rendering backend tests.
//
// Image-baseline regression is not supported yet (vtkMetalRenderWindow does not
// implement GetPixelData()/ReadPixels()), so tests verify rendering indirectly
// via the GPU picking IDs texture (GetIdsData), which is written every render
// pass. Tests must call RenderAndWait() before any read-back so the read does
// not race the GPU.

#ifndef TestMetalHelpers_h
#define TestMetalHelpers_h

#include "vtkCocoaMetalRenderWindow.h"
#include "vtkNew.h"
#include "vtkRenderWindow.h"
#include "vtkUnsignedIntArray.h"

#include <cstring>
#include <iostream>

namespace vtkMetalTesting
{

// Render a frame and block until the GPU finishes. Returns false if the
// drawable acquisition or the command buffer creation failed.
inline bool RenderAndWait(vtkRenderWindow* renWin)
{
  vtkCocoaMetalRenderWindow* metalWin = vtkCocoaMetalRenderWindow::SafeDownCast(renWin);
  if (!metalWin)
  {
    std::cerr << "Not a Metal render window" << std::endl;
    return false;
  }
  metalWin->Render();
  if (!metalWin->GetCurrentCommandBuffer())
  {
    std::cerr << "No command buffer was created (drawable acquisition failed?)" << std::endl;
    return false;
  }
  metalWin->WaitForCompletion();
  return true;
}

// Verify the window is backed by Metal and expose the expected Metal objects.
inline bool CheckBackend(vtkRenderWindow* renWin)
{
  vtkCocoaMetalRenderWindow* metalWin = vtkCocoaMetalRenderWindow::SafeDownCast(renWin);
  if (!metalWin)
  {
    std::cerr << "Not a Metal render window" << std::endl;
    return false;
  }
  if (std::strcmp(metalWin->GetRenderingBackend(), "Metal") != 0)
  {
    std::cerr << "Unexpected rendering backend: " << metalWin->GetRenderingBackend() << std::endl;
    return false;
  }
  if (!metalWin->GetMetalDevice())
  {
    std::cerr << "No Metal device was created" << std::endl;
    return false;
  }
  if (!metalWin->GetMetalLayer())
  {
    std::cerr << "No Metal layer was created" << std::endl;
    return false;
  }
  return true;
}

// Count pixels inside the given window region that were hit by geometry.
// A pixel is considered hit when the picking IDs texture holds a non-zero
// cell or prop ID (i.e. it is not background).
inline vtkIdType CountHitPixels(
  vtkRenderWindow* renWin, int x0, int y0, int x1, int y1)
{
  vtkCocoaMetalRenderWindow* metalWin = vtkCocoaMetalRenderWindow::SafeDownCast(renWin);
  if (!metalWin)
  {
    return 0;
  }
  vtkNew<vtkUnsignedIntArray> ids;
  metalWin->GetIdsData(x0, y0, x1, y1, ids);
  const vtkIdType numTuples = ids->GetNumberOfTuples();
  if (numTuples == 0)
  {
    return 0;
  }
  const unsigned int* raw = ids->GetPointer(0);
  vtkIdType count = 0;
  for (vtkIdType i = 0; i < numTuples; ++i)
  {
    if (raw[i * 4] != 0 || raw[i * 4 + 1] != 0)
    {
      ++count;
    }
  }
  return count;
}

// Count pixels inside the given window region whose composite-index channel
// is non-zero (used to verify per-block picking for composite datasets).
inline vtkIdType CountCompositePixels(
  vtkRenderWindow* renWin, int x0, int y0, int x1, int y1)
{
  vtkCocoaMetalRenderWindow* metalWin = vtkCocoaMetalRenderWindow::SafeDownCast(renWin);
  if (!metalWin)
  {
    return 0;
  }
  vtkNew<vtkUnsignedIntArray> ids;
  metalWin->GetIdsData(x0, y0, x1, y1, ids);
  const vtkIdType numTuples = ids->GetNumberOfTuples();
  if (numTuples == 0)
  {
    return 0;
  }
  const unsigned int* raw = ids->GetPointer(0);
  vtkIdType count = 0;
  for (vtkIdType i = 0; i < numTuples; ++i)
  {
    if (raw[i * 4 + 2] != 0)
    {
      ++count;
    }
  }
  return count;
}

// Verify that at least minPixels pixels in the region were hit by geometry.
inline bool VerifyRegionRendered(
  vtkRenderWindow* renWin, int x0, int y0, int x1, int y1, vtkIdType minPixels)
{
  const vtkIdType count = CountHitPixels(renWin, x0, y0, x1, y1);
  if (count < minPixels)
  {
    std::cerr << "Expected at least " << minPixels << " hit pixels in region [" << x0 << "," << y0
              << "]..[" << x1 << "," << y1 << "], got " << count << std::endl;
    return false;
  }
  return true;
}

}

#endif // TestMetalHelpers_h
