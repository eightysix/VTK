// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
//
// Replicates the depth-peeling coverage of the OpenGL tests
// (TestDepthPeelingPass) for the Metal backend: several overlapping
// translucent actors are rendered both with and without depth peeling enabled
// and the scene is verified each time.

#include "TestMetalHelpers.h"

#include "vtkMetalActor.h"
#include "vtkMetalCamera.h"
#include "vtkMetalPolyDataMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkNew.h"
#include "vtkProperty.h"
#include "vtkSphereSource.h"

namespace
{

vtkNew<vtkMetalActor> MakeTranslucentSphere(vtkSphereSource* sphere, double x, double y, double z,
  double r, double g, double b, double opacity)
{
  vtkNew<vtkMetalPolyDataMapper> mapper;
  mapper->SetInputConnection(sphere->GetOutputPort());
  vtkNew<vtkMetalActor> actor;
  actor->SetMapper(mapper);
  actor->GetProperty()->SetAmbientColor(1.0, 0.0, 0.0);
  actor->GetProperty()->SetDiffuseColor(r, g, b);
  actor->GetProperty()->SetSpecular(0.0);
  actor->GetProperty()->SetDiffuse(0.5);
  actor->GetProperty()->SetAmbient(0.3);
  actor->GetProperty()->SetOpacity(opacity);
  actor->SetPosition(x, y, z);
  return actor;
}

}

int TestMetalDepthPeeling(int argc, char* argv[])
{
  (void)argc;
  (void)argv;

  vtkNew<vtkCocoaMetalRenderWindow> renWin;
  renWin->SetSize(400, 400);
  renWin->SetMultiSamples(0);

  vtkNew<vtkMetalRenderer> renderer;
  renderer->SetBackground(1.0, 1.0, 1.0);
  renderer->SetBackground2(0.3, 0.1, 0.2);
  renderer->GradientBackgroundOn();
  renWin->AddRenderer(renderer);

  vtkNew<vtkMetalCamera> camera;
  renderer->SetActiveCamera(camera);

  vtkNew<vtkSphereSource> sphere;
  sphere->SetThetaResolution(24);
  sphere->SetPhiResolution(24);

  renderer->AddActor(
    MakeTranslucentSphere(sphere, -0.5, 0.0, 0.0, 1.0, 0.8, 0.3, 0.35));
  renderer->AddActor(
    MakeTranslucentSphere(sphere, 0.0, 0.0, 0.2, 0.2, 1.0, 0.8, 0.2));
  renderer->AddActor(
    MakeTranslucentSphere(sphere, 0.5, 0.0, -0.2, 0.5, 0.65, 1.0, 0.35));

  renderer->ResetCamera();
  camera->Azimuth(15.0);
  camera->Zoom(1.5);
  renderer->ResetCameraClippingRange();

  // Standard alpha-blended translucent pass.
  renderer->SetUseDepthPeeling(false);
  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }
  if (!vtkMetalTesting::VerifyRegionRendered(renWin, 120, 120, 279, 279, 1000))
  {
    return EXIT_FAILURE;
  }

  // Depth-peeled translucent pass. The peel passes render to intermediate
  // textures that never carry picking IDs (the GPU picking buffer is cleared
  // by the opaque pass and only the standard translucent pass writes it), so
  // this frame is verified as a smoke test: it must render and complete.
  renderer->SetUseDepthPeeling(true);
  renderer->SetMaximumNumberOfPeels(20);
  renderer->SetOcclusionRatio(0.0);
  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }

  // Render a second frame to exercise the peel-texture reuse path.
  camera->Azimuth(30.0);
  renderer->ResetCameraClippingRange();
  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }

  // Image-based regression against a baseline (final depth-peeled frame).
  return vtkMetalTesting::RegressionExitCode(vtkRegressionTestImage(renWin));
}
