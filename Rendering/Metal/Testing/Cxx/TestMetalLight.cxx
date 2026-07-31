// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
//
// Replicates the light coverage of the OpenGL tests (TestLight, TestLightKit)
// for the Metal backend: headlight, directional, point and spot lights are
// added to a renderer and the scene is rendered and verified.

#include "TestMetalHelpers.h"

#include "vtkConeSource.h"
#include "vtkLight.h"
#include "vtkLightCollection.h"
#include "vtkMetalActor.h"
#include "vtkMetalCamera.h"
#include "vtkMetalPolyDataMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkNew.h"
#include "vtkProperty.h"

int TestMetalLight(int argc, char* argv[])
{
  (void)argc;
  (void)argv;

  vtkNew<vtkCocoaMetalRenderWindow> renWin;
  renWin->SetSize(400, 400);
  renWin->SetMultiSamples(0);

  vtkNew<vtkMetalRenderer> renderer;
  renderer->SetBackground(0.1, 0.2, 0.4);
  renderer->AutomaticLightCreationOff();
  renWin->AddRenderer(renderer);

  vtkNew<vtkConeSource> cone;
  vtkNew<vtkMetalPolyDataMapper> mapper;
  mapper->SetInputConnection(cone->GetOutputPort());
  vtkNew<vtkMetalActor> actor;
  actor->SetMapper(mapper);
  actor->GetProperty()->SetAmbient(0.1);
  actor->GetProperty()->SetDiffuse(0.8);
  actor->GetProperty()->SetSpecular(0.4);
  actor->GetProperty()->SetSpecularPower(40.0);
  renderer->AddActor(actor);

  vtkNew<vtkMetalCamera> camera;
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();

  vtkNew<vtkLight> headlight;
  headlight->SetLightTypeToHeadlight();
  renderer->AddLight(headlight);

  vtkNew<vtkLight> directional;
  directional->SetLightTypeToSceneLight();
  directional->SetPositional(0);
  directional->SetFocalPoint(0.0, 0.0, 0.0);
  directional->SetPosition(1.0, 1.0, 1.0);
  renderer->AddLight(directional);

  vtkNew<vtkLight> point;
  point->SetLightTypeToSceneLight();
  point->SetPositional(1);
  point->SetPosition(2.0, 3.0, 4.0);
  point->SetConeAngle(180.0);
  renderer->AddLight(point);

  vtkNew<vtkLight> spot;
  spot->SetLightTypeToSceneLight();
  spot->SetPositional(1);
  spot->SetFocalPoint(0.0, 0.0, 0.0);
  spot->SetPosition(0.0, 0.0, 5.0);
  spot->SetConeAngle(25.0);
  spot->SetExponent(5.0);
  renderer->AddLight(spot);

  if (renderer->GetLights()->GetNumberOfItems() != 4)
  {
    std::cerr << "Expected 4 lights, got " << renderer->GetLights()->GetNumberOfItems()
              << std::endl;
    return EXIT_FAILURE;
  }

  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }
  if (!vtkMetalTesting::CheckBackend(renWin))
  {
    return EXIT_FAILURE;
  }

  if (!vtkMetalTesting::VerifyRegionRendered(renWin, 160, 160, 239, 239, 500))
  {
    return EXIT_FAILURE;
  }

  for (int i = 1; i <= 3; ++i)
  {
    renderer->GetLights()->RemoveItem(0);
    renderer->ResetCameraClippingRange();
    if (!vtkMetalTesting::RenderAndWait(renWin))
    {
      return EXIT_FAILURE;
    }
    if (!vtkMetalTesting::VerifyRegionRendered(renWin, 160, 160, 239, 239, 100))
    {
      std::cerr << "Rendering failed with " << renderer->GetLights()->GetNumberOfItems()
                << " lights left" << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
