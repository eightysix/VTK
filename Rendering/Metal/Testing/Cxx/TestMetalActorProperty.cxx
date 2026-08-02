// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
//
// Replicates the actor-property coverage of the OpenGL tests (TestProperty,
// TestSpherePoints) for the Metal backend: material settings, opacity,
// backface properties, edge visibility and the surface/wireframe/points
// representations are exercised and re-rendered.

#include "TestMetalHelpers.h"

#include "vtkMetalActor.h"
#include "vtkMetalCamera.h"
#include "vtkMetalPolyDataMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkNew.h"
#include "vtkProperty.h"
#include "vtkSphereSource.h"

int TestMetalActorProperty(int argc, char* argv[])
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
  actor->GetProperty()->SetAmbient(0.1);
  actor->GetProperty()->SetDiffuse(0.8);
  actor->GetProperty()->SetSpecular(0.5);
  actor->GetProperty()->SetSpecularPower(30.0);
  actor->GetProperty()->SetDiffuseColor(0.4, 1.0, 1.0);
  actor->GetProperty()->SetAmbientColor(0.1, 0.2, 0.3);
  actor->GetProperty()->SetSpecularColor(1.0, 1.0, 1.0);
  renderer->AddActor(actor);

  vtkNew<vtkProperty> backProp;
  backProp->SetDiffuseColor(0.4, 0.65, 0.8);
  actor->SetBackfaceProperty(backProp);

  renderer->ResetCamera();

  // Surface representation with edges.
  actor->GetProperty()->SetRepresentationToSurface();
  actor->GetProperty()->EdgeVisibilityOn();
  actor->GetProperty()->SetEdgeColor(1.0, 1.0, 1.0);
  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }
  if (!vtkMetalTesting::VerifyRegionRendered(renWin, renderer, 160, 160, 239, 239, 500))
  {
    return EXIT_FAILURE;
  }

  // Wireframe representation.
  actor->GetProperty()->SetRepresentationToWireframe();
  actor->GetProperty()->EdgeVisibilityOff();
  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }
  if (!vtkMetalTesting::VerifyRegionRendered(renWin, renderer, 160, 160, 239, 239, 100))
  {
    return EXIT_FAILURE;
  }

  // Points representation.
  actor->GetProperty()->SetRepresentationToPoints();
  actor->GetProperty()->SetPointSize(6.0);
  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }
  if (!vtkMetalTesting::VerifyRegionRendered(renWin, renderer, 160, 160, 239, 239, 50))
  {
    return EXIT_FAILURE;
  }

  // Translucent surface (exercises the translucent geometry pass).
  actor->GetProperty()->SetRepresentationToSurface();
  actor->GetProperty()->SetOpacity(0.5);
  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }
  if (!vtkMetalTesting::VerifyRegionRendered(renWin, renderer, 160, 160, 239, 239, 500))
  {
    return EXIT_FAILURE;
  }

  // Image-based regression against a baseline (final translucent-surface state).
  return vtkMetalTesting::RegressionExitCode(vtkRegressionTestImage(renWin));
}
