// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

// Test the Metal glyph3D mapper (instanced rendering).

#include "TestMetalHelpers.h"

#include "vtkCamera.h"
#include "vtkElevationFilter.h"
#include "vtkMetalActor.h"
#include "vtkMetalCamera.h"
#include "vtkMetalGlyph3DMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkPlaneSource.h"
#include "vtkPolyData.h"
#include "vtkSphereSource.h"

#include <iostream>

int TestMetalGlyph3DMapper(int argc, char* argv[])
{
  (void)argc;
  (void)argv;
  vtkNew<vtkCocoaMetalRenderWindow> renWin;
  vtkNew<vtkMetalRenderer> renderer;
  renWin->AddRenderer(renderer);
  renWin->SetSize(600, 300);
  renderer->SetBackground(0.2, 0.2, 0.2);

  renWin->Initialize();
  if (!vtkMetalTesting::CheckBackend(renWin))
  {
    return EXIT_FAILURE;
  }

  // Input point set: a 4x4 plane (16 cells, 25 points).
  vtkNew<vtkPlaneSource> plane;
  plane->SetResolution(3, 3);

  // Per-instance colors.
  vtkNew<vtkElevationFilter> colors;
  colors->SetInputConnection(plane->GetOutputPort());
  colors->SetLowPoint(-1, -1, -1);
  colors->SetHighPoint(1, 1, 1);

  vtkNew<vtkSphereSource> squad;
  squad->SetPhiResolution(6);
  squad->SetThetaResolution(12);

  vtkNew<vtkMetalGlyph3DMapper> glypher;
  glypher->SetInputConnection(colors->GetOutputPort());
  glypher->SetSourceConnection(squad->GetOutputPort());
  glypher->SetScaleFactor(0.25);
  glypher->ScalarVisibilityOn();
  glypher->SetColorModeToMapScalars();
  glypher->SetScalarRange(0.0, 1.0);

  vtkNew<vtkMetalActor> glyphActor;
  glyphActor->SetMapper(glypher);

  renderer->AddActor(glyphActor);
  vtkNew<vtkMetalCamera> camera;
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
  renderer->GetActiveCamera()->Zoom(1.2);

  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }

  // The glyphs cover the center of the viewport: check a broad region.
  if (!vtkMetalTesting::VerifyRegionRendered(renWin, renderer, 150, 75, 450, 225, 500))
  {
    std::cerr << "Glyph rendering did not fill the expected viewport region." << std::endl;
    return EXIT_FAILURE;
  }

  // Second render with the same mapper: re-verifies instanced buffers reuse.
  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }

  if (!vtkMetalTesting::VerifyRegionRendered(renWin, renderer, 150, 75, 450, 225, 500))
  {
    std::cerr << "Glyph rendering changed after a re-render." << std::endl;
    return EXIT_FAILURE;
  }

  // Image-based regression against a baseline.
  return vtkMetalTesting::RegressionExitCode(vtkRegressionTestImage(renWin));
}
