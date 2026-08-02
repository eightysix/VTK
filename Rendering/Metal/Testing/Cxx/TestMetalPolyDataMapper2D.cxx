// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

// Test vtkMetalPolyDataMapper2D in an overlay.
//
// NOTE: the Metal 2D fragment shader only outputs color (no picking IDs), and
// vtkMetalRenderer::DeviceRender does not currently drive RenderOverlay(), so
// the overlay quad does not appear in the image. The test verifies that a 2D
// mapper/actor can be added to the scene, that the 3D geometry still renders,
// and uses image-baseline regression on the 3D output.

#include "TestMetalHelpers.h"

#include "vtkActor2D.h"
#include "vtkCellArray.h"
#include "vtkConeSource.h"
#include "vtkMetalActor.h"
#include "vtkMetalCamera.h"
#include "vtkMetalPolyDataMapper.h"
#include "vtkMetalPolyDataMapper2D.h"
#include "vtkMetalRenderer.h"
#include "vtkNew.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkProperty2D.h"

#include <iostream>

int TestMetalPolyDataMapper2D(int argc, char* argv[])
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

  // 3D scene: a cone filling the left half of the window.
  vtkNew<vtkConeSource> cone;
  cone->SetResolution(32);
  vtkNew<vtkMetalPolyDataMapper> coneMapper;
  coneMapper->SetInputConnection(cone->GetOutputPort());
  vtkNew<vtkMetalActor> coneActor;
  coneActor->SetMapper(coneMapper);
  coneActor->SetPosition(-1.5, 0, 0);
  renderer->AddActor(coneActor);
  vtkNew<vtkMetalCamera> camera;
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
  renderer->GetActiveCamera()->Zoom(1.4);

  // 2D overlay: a quad in display coordinates.
  vtkNew<vtkPoints> points;
  points->InsertNextPoint(450, 40, 0);
  points->InsertNextPoint(580, 40, 0);
  points->InsertNextPoint(580, 260, 0);
  points->InsertNextPoint(450, 260, 0);
  vtkNew<vtkCellArray> cellArray;
  vtkIdType quad[4] = { 0, 1, 2, 3 };
  cellArray->InsertNextCell(4, quad);
  vtkNew<vtkPolyData> quadData;
  quadData->SetPoints(points);
  quadData->SetPolys(cellArray);

  vtkNew<vtkMetalPolyDataMapper2D> quadMapper;
  quadMapper->SetInputData(quadData);
  vtkNew<vtkActor2D> quadActor;
  quadActor->SetMapper(quadMapper);
  quadActor->GetProperty()->SetColor(1.0, 0.5, 0.0);
  renderer->AddActor(quadActor);

  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    std::cerr << "Render with 2D overlay failed." << std::endl;
    return EXIT_FAILURE;
  }

  // The 3D cone (left half) must still render.
  if (!vtkMetalTesting::VerifyRegionRendered(renWin, renderer, 80, 80, 240, 220, 200))
  {
    std::cerr << "3D geometry did not render alongside the 2D overlay." << std::endl;
    return EXIT_FAILURE;
  }

  // A second frame with the overlay still attached must also render.
  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    std::cerr << "Re-render with 2D overlay failed." << std::endl;
    return EXIT_FAILURE;
  }

  // Image-based regression against a baseline (3D geometry; the overlay is not
  // driven by DeviceRender yet, so it does not appear in the image).
  return vtkMetalTesting::RegressionExitCode(vtkRegressionTestImage(renWin));
}
