// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
//
// Shared verification helpers for the Metal rendering backend tests.
//
// Image-baseline regression is supported via vtkMetalRenderWindow's color
// read-back (GetPixelData()/GetRGBACharPixelData()): tests end with
// vtkRegressionTestImage() and compare against PNG baselines. Intermediate
// structural checks (VerifyRegionRendered/CountHitPixels) use the same color
// read-back and count pixels that differ from the renderer background, the
// same signal the OpenGL tests' image regression consumes. They deliberately
// do not use the GPU picking IDs texture (GetIdsData): the surface pipelines
// only write picking IDs during a hardware-selection pass, so that buffer is
// not populated by ordinary renders.
// Tests must call RenderAndWait() before any read-back so the read does not
// race the GPU.

#ifndef TestMetalHelpers_h
#define TestMetalHelpers_h

#include "vtkCocoaMetalRenderWindow.h"
#include "vtkNew.h"
#include "vtkRegressionTestImage.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkUnsignedCharArray.h"
#include "vtkUnsignedIntArray.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace vtkMetalTesting
{

// Map the vtkRegressionTestImage() return value to an exit code.
// NOT_RUN (no -V baseline passed, e.g. direct invocation) and DO_INTERACTOR
// are treated as success.
inline int RegressionExitCode(int retVal)
{
  if (retVal == vtkTesting::NOT_RUN || retVal == vtkTesting::DO_INTERACTOR)
  {
    return EXIT_SUCCESS;
  }
  return (retVal == vtkTesting::PASSED) ? EXIT_SUCCESS : EXIT_FAILURE;
}

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

// Tolerance (in [0,1] color units) below which a pixel is considered part of
// the background. It absorbs the gradient-background dither noise (half a
// quantum, ~0.002) and the sub-pixel difference between the sample position of
// the full-window gradient quad and the pixel's own location, while still
// detecting any geometry that differs from the background by more than a few
// quantization levels.
constexpr double kBackgroundTolerance = 0.02;

// Count pixels inside the given window region that were hit by geometry.
// A pixel is considered hit when its color differs from the renderer's
// background color at that location. Uniform and vertical-gradient
// backgrounds are supported (Background at the bottom, Background2 at the
// top, the layout vtkOpenGLRenderer/vtkMetalRenderer draw). This mirrors how
// the OpenGL tests verify rendered output — via the color read-back — rather
// than via the picking IDs texture, which ordinary renders do not populate.
inline vtkIdType CountHitPixels(
  vtkRenderWindow* renWin, vtkRenderer* renderer, int x0, int y0, int x1, int y1)
{
  if (!renWin || !renderer)
  {
    return 0;
  }
  const int xL = std::min(x0, x1);
  const int yL = std::min(y0, y1);
  const int xH = std::max(x0, x1);
  const int yH = std::max(y0, y1);
  const int width = xH - xL + 1;
  const int height = yH - yL + 1;
  if (width <= 0 || height <= 0)
  {
    return 0;
  }

  vtkNew<vtkUnsignedCharArray> rgba;
  renWin->GetRGBACharPixelData(xL, yL, xH, yH, 0, rgba);
  if (!rgba || rgba->GetNumberOfTuples() != static_cast<vtkIdType>(width * height))
  {
    return 0;
  }

  double bg[3], bg2[3];
  renderer->GetBackground(bg);
  renderer->GetBackground2(bg2);
  const bool gradient = renderer->GetGradientBackground();

  const int* winSize = renWin->GetSize();
  const double denom = (winSize && winSize[1] > 1) ? static_cast<double>(winSize[1] - 1) : 1.0;

  const unsigned char* raw = rgba->GetPointer(0);
  vtkIdType count = 0;
  for (int row = 0; row < height; ++row)
  {
    // VTK window coordinates are bottom-left origin (y up); the vertical
    // gradient interpolates Background at the bottom to Background2 at the top.
    const double t = gradient ? static_cast<double>(yL + row) / denom : 0.0;
    const double er = bg[0] + t * (bg2[0] - bg[0]);
    const double eg = bg[1] + t * (bg2[1] - bg[1]);
    const double eb = bg[2] + t * (bg2[2] - bg[2]);
    for (int col = 0; col < width; ++col)
    {
      const unsigned char* px = &raw[(row * width + col) * 4];
      if (std::fabs(px[0] / 255.0 - er) > kBackgroundTolerance ||
        std::fabs(px[1] / 255.0 - eg) > kBackgroundTolerance ||
        std::fabs(px[2] / 255.0 - eb) > kBackgroundTolerance)
      {
        ++count;
      }
    }
  }
  return count;
}

// Count pixels inside the given window region that were hit by composite
// geometry. The composite-index channel of the picking IDs buffer is only
// written during hardware selection, so this counts non-background pixels the
// same way as CountHitPixels, i.e. any geometry rendered by the composite
// mapper.
inline vtkIdType CountCompositePixels(
  vtkRenderWindow* renWin, vtkRenderer* renderer, int x0, int y0, int x1, int y1)
{
  return CountHitPixels(renWin, renderer, x0, y0, x1, y1);
}

// Verify that at least minPixels pixels in the region were hit by geometry.
inline bool VerifyRegionRendered(
  vtkRenderWindow* renWin, vtkRenderer* renderer, int x0, int y0, int x1, int y1,
  vtkIdType minPixels)
{
  const vtkIdType count = CountHitPixels(renWin, renderer, x0, y0, x1, y1);
  if (count < minPixels)
  {
    std::cerr << "Expected at least " << minPixels << " non-background pixels in region ["
              << x0 << "," << y0 << "]..[" << x1 << "," << y1 << "], got " << count << std::endl;
    return false;
  }
  return true;
}

}

#endif // TestMetalHelpers_h
