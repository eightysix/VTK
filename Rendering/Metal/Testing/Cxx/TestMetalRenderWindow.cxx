// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
#include "TestMetalHelpers.h"

#include "vtkCocoaMetalRenderWindow.h"
#include "vtkMetalActor.h"
#include "vtkMetalCamera.h"
#include "vtkMetalPolyDataMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkNew.h"

#include "vtkConeSource.h"
#include "vtkRegressionTestImage.h"
#include "vtkRenderer.h"

#include <cstring>
#include <iostream>

int TestMetalRenderWindow(int argc, char* argv[])
{

  vtkNew<vtkCocoaMetalRenderWindow> renWin;
  renWin->SetSize(400, 400);
  renWin->SetMultiSamples(0);

  vtkNew<vtkMetalRenderer> renderer;
  renderer->SetBackground(0.1, 0.2, 0.4);
  renWin->AddRenderer(renderer);

  vtkNew<vtkMetalCamera> camera;
  renderer->SetActiveCamera(camera);

  vtkNew<vtkConeSource> cone;
  vtkNew<vtkMetalPolyDataMapper> mapper;
  mapper->SetInputConnection(cone->GetOutputPort());
  vtkNew<vtkMetalActor> actor;
  actor->SetMapper(mapper);
  renderer->AddActor(actor);

  renderer->ResetCamera();
  renWin->Render();

  if (std::strcmp(renWin->GetRenderingBackend(), "Metal") != 0)
  {
    std::cerr << "Unexpected rendering backend: " << renWin->GetRenderingBackend()
              << std::endl;
    return EXIT_FAILURE;
  }
  if (!renWin->GetMetalDevice())
  {
    std::cerr << "No Metal device was created" << std::endl;
    return EXIT_FAILURE;
  }
  if (!renWin->GetMetalLayer())
  {
    std::cerr << "No Metal layer was created" << std::endl;
    return EXIT_FAILURE;
  }
  if (!renWin->GetCurrentCommandBuffer())
  {
    std::cerr << "No command buffer was created (drawable acquisition failed?)"
              << std::endl;
    return EXIT_FAILURE;
  }
  renWin->WaitForCompletion();

  // A valid draw must put non-background pixels in the center of the cone.
  // Verified via the color read-back (the same signal vtkRegressionTestImage
  // compares) rather than the picking IDs texture, which the surface
  // pipelines only populate during a hardware-selection pass.
  if (!vtkMetalTesting::VerifyRegionRendered(renWin, renderer, 160, 160, 239, 239, 500))
  {
    std::cerr << "No geometry drawn (center region is all background)" << std::endl;
    return EXIT_FAILURE;
  }

  // Image-based regression against a baseline (the cone scene).
  const int retVal = vtkRegressionTestImage(renWin);
  if (retVal == vtkTesting::NOT_RUN || retVal == vtkTesting::DO_INTERACTOR)
  {
    return EXIT_SUCCESS;
  }
  return (retVal == vtkTesting::PASSED) ? EXIT_SUCCESS : EXIT_FAILURE;
}
