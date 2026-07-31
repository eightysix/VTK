// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
#include "vtkCocoaMetalRenderWindow.h"
#include "vtkMetalActor.h"
#include "vtkMetalCamera.h"
#include "vtkMetalPolyDataMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkNew.h"
#include "vtkUnsignedIntArray.h"

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

  // The IDs texture is written on every render pass, so a valid draw should
  // produce non-zero cell/prop IDs in the center of the cone.
  vtkNew<vtkUnsignedIntArray> ids;
  renWin->GetIdsData(0, 0, 399, 399, ids);
  if (ids->GetNumberOfTuples() == 0)
  {
    std::cerr << "GetIdsData returned no tuples" << std::endl;
    return EXIT_FAILURE;
  }

  bool foundRender = false;
  for (int y = 160; y < 240 && !foundRender; ++y)
  {
    for (int x = 160; x < 240; ++x)
    {
      vtkIdType index = y * 400 + x;
      if (ids->GetComponent(index, 0) != 0 || ids->GetComponent(index, 1) != 0)
      {
        foundRender = true;
        break;
      }
    }
  }
  if (!foundRender)
  {
    std::cerr << "No geometry drawn (all cell/prop IDs are zero)" << std::endl;
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
