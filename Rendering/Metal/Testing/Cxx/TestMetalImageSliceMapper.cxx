// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

// Test vtkMetalImageSliceMapper (3D image slice rendering).
//
// A 64x64 RGB unsigned-char quadrant image is displayed by a vtkImageSlice +
// vtkMetalImageSliceMapper under parallel projection. The image is centered in
// the square window; pixel read-back verifies the four quadrant colors and
// their orientation (bottom row of the image at the bottom of the window, no
// flip), plus an image-baseline regression.

#include "TestMetalHelpers.h"

#include "vtkDataObject.h"
#include "vtkIdTypeArray.h"
#include "vtkImageData.h"
#include "vtkImageProperty.h"
#include "vtkImageSlice.h"
#include "vtkInformation.h"
#include "vtkMetalCamera.h"
#include "vtkMetalHardwareSelector.h"
#include "vtkMetalImageSliceMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkNew.h"
#include "vtkPointData.h"
#include "vtkProp.h"
#include "vtkRenderer.h"
#include "vtkSelection.h"
#include "vtkSelectionNode.h"

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

int TestMetalImageSliceMapper(int argc, char* argv[])
{
  (void)argc;
  (void)argv;
  vtkNew<vtkCocoaMetalRenderWindow> renWin;
  renWin->SetSize(300, 300);
  renWin->Initialize();
  if (!vtkMetalTesting::CheckBackend(renWin))
  {
    return EXIT_FAILURE;
  }

  vtkNew<vtkMetalRenderer> renderer;
  renderer->SetBackground(0.2, 0.2, 0.2);
  renWin->AddRenderer(renderer);

  // The image data spans world x/y in [0,63] at z=0 (origin 0, spacing 1).
  // Parallel projection with the camera centered on the image and a parallel
  // scale larger than the image half-extent renders the 64x64 slice into the
  // center of the 300x300 window with a uniform margin.
  vtkNew<vtkMetalCamera> camera;
  renderer->SetActiveCamera(camera);
  camera->ParallelProjectionOn();
  camera->SetPosition(0.0, 0.0, 100.0);
  camera->SetFocalPoint(31.5, 31.5, 0.0);
  camera->SetViewUp(0.0, 1.0, 0.0);
  camera->SetParallelScale(40.0);
  renderer->ResetCameraClippingRange();

  vtkNew<vtkMetalImageSliceMapper> mapper;
  mapper->SetInputData(CreateQuadrantImage());

  vtkNew<vtkImageSlice> slice;
  slice->SetMapper(mapper);
  renderer->AddViewProp(slice);

  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    std::cerr << "Render with image slice mapper failed." << std::endl;
    return EXIT_FAILURE;
  }

  // The window is 300x300, the visible parallel half-height is 40 world units,
  // so 1 world unit = 300 / 80 = 3.75 px. The image spans 63 units, i.e.
  // ~236 px centered in the window; the quadrants are each ~118 px wide, so a
  // sample point 59 px from the window center lands at each quadrant's center.
  const unsigned char red[3] = { 255, 0, 0 };
  const unsigned char green[3] = { 0, 255, 0 };
  const unsigned char blue[3] = { 0, 0, 255 };
  const unsigned char white[3] = { 255, 255, 255 };

  const int q = 59;
  const int cx = 150;
  const int cy = 150;
  if (!CheckColor(renWin, "bottom-left quadrant", cx - q, cy - q, red, 3))
    return EXIT_FAILURE;
  if (!CheckColor(renWin, "bottom-right quadrant", cx + q, cy - q, green, 3))
    return EXIT_FAILURE;
  // If the image were vertically flipped, this would read red instead of blue.
  if (!CheckColor(renWin, "top-left quadrant", cx - q, cy + q, blue, 3))
    return EXIT_FAILURE;
  if (!CheckColor(renWin, "top-right quadrant", cx + q, cy + q, white, 3))
    return EXIT_FAILURE;

  // Re-render: exercises the cached texture upload / pipeline path.
  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    std::cerr << "Re-render with image slice mapper failed." << std::endl;
    return EXIT_FAILURE;
  }
  if (!CheckColor(renWin, "bottom-left quadrant (re-render)", cx - q, cy - q, red, 3))
    return EXIT_FAILURE;
  if (!CheckColor(renWin, "top-left quadrant (re-render)", cx - q, cy + q, blue, 3))
    return EXIT_FAILURE;

  // --- Hardware-selection checks (cell picking) ------------------------------
  // The 64x64 image with CELLS field association yields a 63x63 = 3969 cell
  // ID grid; the fragment shader encodes pixelId = j*63 + i.
  const vtkIdType gridDim = 63;
  const vtkIdType numCells = gridDim * gridDim;

  vtkNew<vtkMetalHardwareSelector> selector;
  selector->SetRenderer(renderer);
  selector->SetFieldAssociation(vtkDataObject::FIELD_ASSOCIATION_CELLS);

  // Select over the whole window: only the image slice prop is present, so
  // there must be exactly one selection node containing every cell id.
  selector->SetArea(0, 0, 300, 300);
  vtkSmartPointer<vtkSelection> selection;
  selection.TakeReference(selector->Select());
  if (!selection || selection->GetNumberOfNodes() != 1)
  {
    std::cerr << "Expected 1 selection node, got "
              << (selection ? selection->GetNumberOfNodes() : -1) << std::endl;
    return EXIT_FAILURE;
  }
  {
    vtkSelectionNode* node = selection->GetNode(0);
    vtkProp* prop =
      vtkProp::SafeDownCast(node->GetProperties()->Get(vtkSelectionNode::PROP()));
    if (prop != slice.Get())
    {
      std::cerr << "Selection node does not reference the image slice." << std::endl;
      return EXIT_FAILURE;
    }
    if (node->GetProperties()->Get(vtkSelectionNode::PROP_ID()) != 0)
    {
      std::cerr << "Expected image slice prop id 0." << std::endl;
      return EXIT_FAILURE;
    }
    vtkIdTypeArray* ids = vtkArrayDownCast<vtkIdTypeArray>(node->GetSelectionList());
    if (!ids || ids->GetNumberOfTuples() != numCells)
    {
      std::cerr << "Expected " << numCells << " cell ids, got "
                << (ids ? ids->GetNumberOfTuples() : -1) << std::endl;
      return EXIT_FAILURE;
    }
    if (ids->GetRange(0)[0] != 0 || ids->GetRange(0)[1] != numCells - 1)
    {
      std::cerr << "Cell ids out of expected range [0, " << numCells - 1 << "]: ["
                << ids->GetRange(0)[0] << ", " << ids->GetRange(0)[1] << "]" << std::endl;
      return EXIT_FAILURE;
    }
  }

  // A point query at the window center maps to world (31.5, 31.5), which the
  // selection shader resolves to texel (31, 31) -> cell id 31*63 + 31 = 1984.
  selector->SetArea(150, 150, 150, 150);
  selection.TakeReference(selector->Select());
  if (!selection || selection->GetNumberOfNodes() != 1)
  {
    std::cerr << "Expected 1 selection node for the center point." << std::endl;
    return EXIT_FAILURE;
  }
  {
    vtkIdTypeArray* ids =
      vtkArrayDownCast<vtkIdTypeArray>(selection->GetNode(0)->GetSelectionList());
    const vtkIdType expectedCellId = 31 * gridDim + 31;
    if (!ids || ids->GetNumberOfTuples() != 1 || ids->GetValue(0) != expectedCellId)
    {
      std::cerr << "Expected center cell id " << expectedCellId << ", got "
                << (ids && ids->GetNumberOfTuples() == 1 ? ids->GetValue(0) : -1) << std::endl;
      return EXIT_FAILURE;
    }
  }

  // Image-based regression against a baseline.
  return vtkMetalTesting::RegressionExitCode(vtkRegressionTestImage(renWin));
}
