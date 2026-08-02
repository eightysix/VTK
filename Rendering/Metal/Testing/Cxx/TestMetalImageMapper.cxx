// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

// Test vtkMetalImageMapper (2D image display in an overlay pass).
//
// Two images are displayed with the Metal image mapper:
//  - 64x64 RGB unsigned-char image with quadrant colors and an identity
//    window/level (exercises the direct char fast path), placed at (0,0).
//  - 64x64 single-component unsigned-short gradient with a real window/level
//    (exercises the fixed-point short path), placed at (300,0).
// Pixel read-back verifies both the mapped colors and the vertical
// orientation (no flip), plus an image-baseline regression.

#include "TestMetalHelpers.h"

#include "vtkActor2D.h"
#include "vtkImageData.h"
#include "vtkMetalImageMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkNew.h"
#include "vtkPointData.h"
#include "vtkRenderer.h"

#include <cmath>
#include <cstring>
#include <iostream>

namespace
{

// 64x64 RGB uchar image with a distinct color per quadrant:
// bottom-left red, bottom-right green, top-left blue, top-right white.
vtkSmartPointer<vtkImageData> CreateQuadrantImage()
{
  constexpr int dim = 64;
  vtkNew<vtkImageData> image;
  image->SetDimensions(dim, dim, 1);
  image->AllocateScalars(VTK_UNSIGNED_CHAR, 3);

  unsigned char* ptr = static_cast<unsigned char*>(image->GetScalarPointer());
  for (int y = 0; y < dim; ++y)
  {
    for (int x = 0; x < dim; ++x)
    {
      unsigned char r = 0, g = 0, b = 0;
      const bool left = x < dim / 2;
      const bool bottom = y < dim / 2;
      if (left && bottom)
      {
        r = 255;
      }
      else if (!left && bottom)
      {
        g = 255;
      }
      else if (left && !bottom)
      {
        b = 255;
      }
      else
      {
        r = g = b = 255;
      }
      ptr[(y * dim + x) * 3 + 0] = r;
      ptr[(y * dim + x) * 3 + 1] = g;
      ptr[(y * dim + x) * 3 + 2] = b;
    }
  }
  return image;
}

// 64x64 single-component unsigned-short gradient 0..1000 along x+y.
vtkSmartPointer<vtkImageData> CreateGradientImage()
{
  constexpr int dim = 64;
  vtkNew<vtkImageData> image;
  image->SetDimensions(dim, dim, 1);
  image->AllocateScalars(VTK_UNSIGNED_SHORT, 1);

  unsigned short* ptr = static_cast<unsigned short*>(image->GetScalarPointer());
  for (int y = 0; y < dim; ++y)
  {
    for (int x = 0; x < dim; ++x)
    {
      ptr[y * dim + x] = static_cast<unsigned short>(1000 * (x + y) / (2 * (dim - 1)));
    }
  }
  return image;
}

// Read one RGB pixel from the window (VTK bottom-left origin).
void ReadPixel(vtkRenderWindow* renWin, int x, int y, unsigned char rgb[3])
{
  unsigned char* pixels = renWin->GetPixelData(x, y, x, y, 0);
  rgb[0] = pixels[0];
  rgb[1] = pixels[1];
  rgb[2] = pixels[2];
  delete[] pixels;
}

bool CheckColor(vtkRenderWindow* renWin, const char* label, int x, int y,
  const unsigned char expected[3], int tolerance)
{
  unsigned char rgb[3];
  ReadPixel(renWin, x, y, rgb);
  if (std::abs(rgb[0] - expected[0]) > tolerance || std::abs(rgb[1] - expected[1]) > tolerance ||
    std::abs(rgb[2] - expected[2]) > tolerance)
  {
    std::cerr << label << " at (" << x << "," << y << ") expected (" << (int)expected[0] << ","
              << (int)expected[1] << "," << (int)expected[2] << ") got (" << (int)rgb[0] << ","
              << (int)rgb[1] << "," << (int)rgb[2] << ")" << std::endl;
    return false;
  }
  return true;
}

} // namespace

int TestMetalImageMapper(int argc, char* argv[])
{
  (void)argc;
  (void)argv;
  vtkNew<vtkCocoaMetalRenderWindow> renWin;
  renWin->SetSize(600, 300);
  renWin->Initialize();
  if (!vtkMetalTesting::CheckBackend(renWin))
  {
    return EXIT_FAILURE;
  }

  vtkNew<vtkMetalRenderer> renderer;
  renderer->SetBackground(0.2, 0.2, 0.2);
  renWin->AddRenderer(renderer);

  // Quadrant RGB uchar image at (0,0), identity window/level (char path).
  vtkNew<vtkMetalImageMapper> mapper1;
  mapper1->SetInputData(CreateQuadrantImage());
  mapper1->SetColorLevel(127.5);
  mapper1->SetColorWindow(255.0);
  vtkNew<vtkActor2D> actor1;
  actor1->SetMapper(mapper1);
  actor1->SetPosition(0, 0);
  renderer->AddActor(actor1);

  // Unsigned-short gradient at (300,0), real window/level (short path).
  vtkNew<vtkMetalImageMapper> mapper2;
  mapper2->SetInputData(CreateGradientImage());
  mapper2->SetColorLevel(500.0);
  mapper2->SetColorWindow(1000.0);
  vtkNew<vtkActor2D> actor2;
  actor2->SetMapper(mapper2);
  actor2->SetPosition(300, 0);
  renderer->AddActor(actor2);

  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    std::cerr << "Render with image mapper failed." << std::endl;
    return EXIT_FAILURE;
  }

  // --- Structural pixel checks (window coords, VTK bottom-left origin) ---
  // --- Structural pixel checks (window coords, VTK bottom-left origin) ---

  const unsigned char red[3] = { 255, 0, 0 };
  const unsigned char green[3] = { 0, 255, 0 };
  const unsigned char blue[3] = { 0, 0, 255 };
  const unsigned char white[3] = { 255, 255, 255 };

  // Quadrant image occupies window x [0,64), y [0,64).
  if (!CheckColor(renWin, "bottom-left quadrant", 16, 16, red, 2))
    return EXIT_FAILURE;
  if (!CheckColor(renWin, "bottom-right quadrant", 48, 16, green, 2))
    return EXIT_FAILURE;
  // If the image were vertically flipped, this would read red instead of blue.
  if (!CheckColor(renWin, "top-left quadrant", 16, 48, blue, 2))
    return EXIT_FAILURE;
  if (!CheckColor(renWin, "top-right quadrant", 48, 48, white, 2))
    return EXIT_FAILURE;

  // Gradient occupies window x [300,364), y [0,64). Gradient maps 0..1000 to
  // 0..255, so (8,8) is dark and (56,56) is bright.
  unsigned char dark[3], bright[3];
  ReadPixel(renWin, 300 + 8, 8, dark);
  ReadPixel(renWin, 300 + 56, 56, bright);
  if (dark[0] > 80)
  {
    std::cerr << "gradient low end should be dark, got " << (int)dark[0] << std::endl;
    return EXIT_FAILURE;
  }
  if (bright[0] < 150)
  {
    std::cerr << "gradient high end should be bright, got " << (int)bright[0] << std::endl;
    return EXIT_FAILURE;
  }

  // Re-render: exercises the cached texture upload / pipeline path.
  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    std::cerr << "Re-render with image mapper failed." << std::endl;
    return EXIT_FAILURE;
  }
  if (!CheckColor(renWin, "bottom-left quadrant (re-render)", 16, 16, red, 2))
    return EXIT_FAILURE;
  if (!CheckColor(renWin, "top-left quadrant (re-render)", 16, 48, blue, 2))
    return EXIT_FAILURE;

  // Image-based regression against a baseline.
  return vtkMetalTesting::RegressionExitCode(vtkRegressionTestImage(renWin));
}
