// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
//
// Replicates the shaped-point and thick-line coverage of the OpenGL tests
// (TestSpherePoints) for the Metal backend: points rendered as spheres and
// thick edge lines are exercised and the scene is re-rendered and verified.

#include "TestMetalHelpers.h"

#include "vtkMetalActor.h"
#include "vtkMetalCamera.h"
#include "vtkMetalPolyDataMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkNew.h"
#include "vtkProperty.h"
#include "vtkSphereSource.h"

int TestMetalPointRender(int argc, char* argv[])
{
  (void)argc;
  (void)argv;

  vtkNew<vtkCocoaMetalRenderWindow> renWin;
  renWin->SetSize(400, 400);
  renWin->SetMultiSamples(0);

  vtkNew<vtkMetalRenderer> renderer;
  renderer->SetBackground(0.1, 0.2, 0.4);
  renWin->AddRenderer(renderer);

  vtkNew<vtkMetalCamera> camera;
  renderer->SetActiveCamera(camera);

  vtkNew<vtkSphereSource> sphere;
  vtkNew<vtkMetalPolyDataMapper> mapper;
  mapper->SetInputConnection(sphere->GetOutputPort());
  vtkNew<vtkMetalActor> actor;
  actor->SetMapper(mapper);
  actor->GetProperty()->SetDiffuseColor(1.0, 0.65, 0.7);
  actor->GetProperty()->SetSpecular(0.5);
  actor->GetProperty()->SetDiffuse(0.7);
  actor->GetProperty()->SetSpecularPower(20.0);
  actor->GetProperty()->RenderPointsAsSpheresOn();
  actor->GetProperty()->SetPointSize(10.0);
  actor->GetProperty()->SetRepresentationToPoints();
  renderer->AddActor(actor);

  renderer->ResetCamera();
  camera->Elevation(-45.0);
  camera->OrthogonalizeViewUp();
  camera->Zoom(1.5);
  renderer->ResetCameraClippingRange();

  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }
  if (!vtkMetalTesting::VerifyRegionRendered(renWin, 160, 160, 239, 239, 20))
  {
    return EXIT_FAILURE;
  }

  actor->GetProperty()->SetRepresentationToSurface();
  actor->GetProperty()->EdgeVisibilityOn();
  actor->GetProperty()->SetLineWidth(7.0);
  actor->GetProperty()->RenderLinesAsTubesOn();
  actor->GetProperty()->SetEdgeColor(1.0, 1.0, 1.0);
  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }
  if (!vtkMetalTesting::VerifyRegionRendered(renWin, 160, 160, 239, 239, 500))
  {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
