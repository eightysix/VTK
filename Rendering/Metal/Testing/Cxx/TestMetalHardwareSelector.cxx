// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

// Test vtkMetalHardwareSelector: verifies per-prop and per-cell picking works
// with the Metal backend (single-pass picking IDs texture).

#include "TestMetalHelpers.h"

#include "vtkConeSource.h"
#include "vtkCoordinate.h"
#include "vtkIdTypeArray.h"
#include "vtkInformation.h"
#include "vtkMetalActor.h"
#include "vtkMetalCamera.h"
#include "vtkMetalHardwareSelector.h"
#include "vtkMetalPolyDataMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkNew.h"
#include "vtkSelection.h"
#include "vtkSelectionNode.h"
#include "vtkSphereSource.h"

#include <iostream>

int TestMetalHardwareSelector(int argc, char* argv[])
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

  vtkNew<vtkConeSource> cone;
  cone->SetResolution(32);
  vtkNew<vtkMetalActor> coneActor;
  vtkNew<vtkMetalPolyDataMapper> coneMapper;
  coneMapper->SetInputConnection(cone->GetOutputPort());
  coneActor->SetMapper(coneMapper);
  coneActor->SetPosition(-1.5, 0, 0);

  vtkNew<vtkSphereSource> sphere;
  sphere->SetPhiResolution(24);
  sphere->SetThetaResolution(24);
  vtkNew<vtkMetalActor> sphereActor;
  vtkNew<vtkMetalPolyDataMapper> sphereMapper;
  sphereMapper->SetInputConnection(sphere->GetOutputPort());
  sphereActor->SetMapper(sphereMapper);
  sphereActor->SetPosition(1.5, 0, 0);

  renderer->AddActor(coneActor);
  renderer->AddActor(sphereActor);
  vtkNew<vtkMetalCamera> camera;
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
  renderer->GetActiveCamera()->Zoom(1.4);

  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }

  vtkNew<vtkMetalHardwareSelector> selector;
  selector->SetRenderer(renderer);
  selector->SetFieldAssociation(vtkDataObject::FIELD_ASSOCIATION_CELLS);

  // Select over the whole window: both actors must be hit.
  selector->SetArea(0, 0, 600, 300);
  vtkSmartPointer<vtkSelection> selection;
  selection.TakeReference(selector->Select());

  if (selection->GetNumberOfNodes() != 2)
  {
    std::cerr << "Expected 2 selection nodes, got " << selection->GetNumberOfNodes() << std::endl;
    return EXIT_FAILURE;
  }

  int seenCone = 0, seenSphere = 0;
  for (vtkIdType i = 0; i < selection->GetNumberOfNodes(); ++i)
  {
    vtkSelectionNode* node = selection->GetNode(i);
    const int propId = node->GetProperties()->Get(vtkSelectionNode::PROP_ID());
    vtkProp* prop = vtkProp::SafeDownCast(node->GetProperties()->Get(vtkSelectionNode::PROP()));
    vtkIdTypeArray* ids = vtkArrayDownCast<vtkIdTypeArray>(node->GetSelectionList());
    if (!ids || ids->GetNumberOfTuples() == 0)
    {
      std::cerr << "Selection node " << i << " has no cell ids." << std::endl;
      return EXIT_FAILURE;
    }

    vtkActor* actor = vtkActor::SafeDownCast(prop);
    if (!actor)
    {
      std::cerr << "Selection node " << i << " has no actor prop." << std::endl;
      return EXIT_FAILURE;
    }
    if (actor == coneActor.Get())
    {
      seenCone = 1;
      if (propId != 0)
      {
        std::cerr << "Expected cone prop id 0, got " << propId << std::endl;
        return EXIT_FAILURE;
      }
      if (ids->GetRange(0)[1] >= cone->GetOutput()->GetNumberOfCells())
      {
        std::cerr << "Cone selection contains out-of-range cell ids." << std::endl;
        return EXIT_FAILURE;
      }
    }
    else if (actor == sphereActor.Get())
    {
      seenSphere = 1;
      if (propId != 1)
      {
        std::cerr << "Expected sphere prop id 1, got " << propId << std::endl;
        return EXIT_FAILURE;
      }
      if (ids->GetRange(0)[1] >= sphere->GetOutput()->GetNumberOfCells())
      {
        std::cerr << "Sphere selection contains out-of-range cell ids." << std::endl;
        return EXIT_FAILURE;
      }
    }
    else
    {
      std::cerr << "Unexpected actor in selection." << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (!seenCone || !seenSphere)
  {
    std::cerr << "Expected both actors to be selected." << std::endl;
    return EXIT_FAILURE;
  }

  // A point query at the cone's projected location must hit only the cone.
  vtkNew<vtkCoordinate> coord;
  coord->SetCoordinateSystemToWorld();
  const double* center = coneActor->GetCenter();
  coord->SetValue(center[0], center[1], center[2]);
  int* disp = coord->GetComputedDisplayValue(renderer);
  selector->SetArea(disp[0] - 30, disp[1] - 30, disp[0] + 30, disp[1] + 30);
  selection.TakeReference(selector->Select());
  if (selection->GetNumberOfNodes() != 1)
  {
    std::cerr << "Expected 1 selection node for the cone region, got "
              << selection->GetNumberOfNodes() << std::endl;
    return EXIT_FAILURE;
  }
  const int propId = selection->GetNode(0)->GetProperties()->Get(vtkSelectionNode::PROP_ID());
  if (propId != 0)
  {
    std::cerr << "Expected cone prop id 0 for the cone region, got " << propId << std::endl;
    return EXIT_FAILURE;
  }

  // Render a regular frame (selection ended) and compare against a baseline.
  return vtkMetalTesting::RegressionExitCode(vtkRegressionTestImage(renWin));
}
