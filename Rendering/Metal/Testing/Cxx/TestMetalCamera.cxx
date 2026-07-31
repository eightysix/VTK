// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
//
// Replicates the camera coverage of the OpenGL tests (TestCamera and friends)
// for the Metal backend: azimuth, elevation, roll, dolly, zoom and clipping
// range operations are exercised and the scene is re-rendered and verified
// after each camera change.

#include "TestMetalHelpers.h"

#include "vtkConeSource.h"
#include "vtkMetalActor.h"
#include "vtkMetalCamera.h"
#include "vtkMetalPolyDataMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkNew.h"

int TestMetalCamera(int argc, char* argv[])
{
  (void)argc;
  (void)argv;

  vtkNew<vtkCocoaMetalRenderWindow> renWin;
  renWin->SetSize(400, 400);
  renWin->SetMultiSamples(0);

  vtkNew<vtkMetalRenderer> renderer;
  renderer->SetBackground(0.1, 0.2, 0.4);
  renWin->AddRenderer(renderer);

  vtkNew<vtkConeSource> cone;
  vtkNew<vtkMetalPolyDataMapper> mapper;
  mapper->SetInputConnection(cone->GetOutputPort());
  vtkNew<vtkMetalActor> actor;
  actor->SetMapper(mapper);
  renderer->AddActor(actor);

  vtkNew<vtkMetalCamera> camera;
  renderer->SetActiveCamera(camera);
  camera->SetPosition(0.0, 0.0, 1.0);
  camera->SetFocalPoint(0.0, 0.0, 0.0);
  camera->SetViewUp(0.0, 1.0, 0.0);
  renderer->ResetCamera();

  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }

  camera->Azimuth(45.0);
  renderer->ResetCameraClippingRange();
  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }

  camera->Elevation(30.0);
  renderer->ResetCameraClippingRange();
  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }

  camera->Roll(25.0);
  camera->OrthogonalizeViewUp();
  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }

  camera->Dolly(1.5);
  renderer->ResetCameraClippingRange();
  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }

  camera->Zoom(1.5);
  renderer->ResetCameraClippingRange();
  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }

  camera->ParallelProjectionOn();
  camera->SetParallelScale(1.5);
  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }

  if (!vtkMetalTesting::VerifyRegionRendered(renWin, 0, 0, 399, 399, 100))
  {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
