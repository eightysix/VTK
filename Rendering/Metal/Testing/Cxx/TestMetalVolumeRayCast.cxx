// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

// Test the Metal GPU volume ray-cast mapper.
//
// Note: the volume fragment shader only writes color (no picking IDs), so this
// test is a smoke test: it verifies the render pipeline runs to completion.

#include "TestMetalHelpers.h"

#include "vtkColorTransferFunction.h"
#include "vtkMetalCamera.h"
#include "vtkMetalGPUVolumeRayCastMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkNew.h"
#include "vtkPiecewiseFunction.h"
#include "vtkRTAnalyticSource.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"

#include <iostream>

int TestMetalVolumeRayCast(int argc, char* argv[])
{
  (void)argc;
  (void)argv;
  vtkNew<vtkCocoaMetalRenderWindow> renWin;
  vtkNew<vtkMetalRenderer> renderer;
  renWin->AddRenderer(renderer);
  renWin->SetSize(400, 400);
  renderer->SetBackground(0.1, 0.1, 0.2);

  renWin->Initialize();
  if (!vtkMetalTesting::CheckBackend(renWin))
  {
    return EXIT_FAILURE;
  }

  // A small analytic dataset so the test stays fast.
  vtkNew<vtkRTAnalyticSource> source;
  source->SetWholeExtent(0, 32, 0, 32, 0, 32);
  source->SetCenter(16, 16, 16);

  vtkNew<vtkColorTransferFunction> color;
  color->AddRGBPoint(0.0, 0.0, 0.0, 0.0);
  color->AddRGBPoint(0.25, 1.0, 0.0, 0.0);
  color->AddRGBPoint(0.75, 0.0, 1.0, 0.0);
  color->AddRGBPoint(1.0, 0.0, 0.0, 1.0);

  vtkNew<vtkPiecewiseFunction> opacity;
  opacity->AddPoint(0.0, 0.0);
  opacity->AddPoint(0.5, 0.15);
  opacity->AddPoint(1.0, 0.9);

  vtkNew<vtkVolumeProperty> property;
  property->SetColor(color);
  property->SetScalarOpacity(opacity);
  property->ShadeOn();
  property->SetAmbient(0.2);
  property->SetDiffuse(0.8);
  property->SetSpecular(0.3);

  vtkNew<vtkVolume> volume;
  volume->SetMapper(vtk::TakeSmartPointer(vtkMetalGPUVolumeRayCastMapper::New()));
  volume->SetProperty(property);

  renderer->AddVolume(volume);
  vtkNew<vtkMetalCamera> camera;
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
  renderer->GetActiveCamera()->Elevation(20);
  renderer->GetActiveCamera()->Azimuth(30);

  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    std::cerr << "Volume render failed." << std::endl;
    return EXIT_FAILURE;
  }

  // Move the camera and render again (exercises re-upload and re-build paths).
  renderer->GetActiveCamera()->Elevation(-40);
  renderer->GetActiveCamera()->Azimuth(-60);

  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    std::cerr << "Volume re-render failed." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
