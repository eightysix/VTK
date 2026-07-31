// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
//
// Replicates the composite-dataset coverage of the OpenGL tests
// (TestCompositePolyDataMapper*) for the Metal backend. The Metal delegator
// (vtkMetalCompositePolyDataMapperDelegator) is used through a small
// vtkCompositePolyDataMapper subclass that returns it from CreateADelegator(),
// then several blocks are batched-rendered and verified per block.

#include "TestMetalHelpers.h"

#include "vtkCompositePolyDataMapper.h"
#include "vtkCompositePolyDataMapperDelegator.h"
#include "vtkConeSource.h"
#include "vtkCubeSource.h"
#include "vtkMetalActor.h"
#include "vtkMetalCamera.h"
#include "vtkMetalCompositePolyDataMapperDelegator.h"
#include "vtkMetalRenderer.h"
#include "vtkNew.h"
#include "vtkObjectFactory.h"
#include "vtkPartitionedDataSetCollection.h"
#include "vtkSphereSource.h"

namespace
{

class MetalCompositePolyDataMapper : public vtkCompositePolyDataMapper
{
public:
  static MetalCompositePolyDataMapper* New();
  vtkTypeMacro(MetalCompositePolyDataMapper, vtkCompositePolyDataMapper);

  vtkCompositePolyDataMapperDelegator* CreateADelegator() override
  {
    return vtkMetalCompositePolyDataMapperDelegator::New();
  }

protected:
  MetalCompositePolyDataMapper() = default;
  ~MetalCompositePolyDataMapper() override = default;

private:
  MetalCompositePolyDataMapper(const MetalCompositePolyDataMapper&) = delete;
  void operator=(const MetalCompositePolyDataMapper&) = delete;
};

vtkStandardNewMacro(MetalCompositePolyDataMapper);

}

int TestMetalCompositePolyDataMapper(int argc, char* argv[])
{
  (void)argc;
  (void)argv;

  vtkNew<vtkPartitionedDataSetCollection> pdc;

  vtkNew<vtkConeSource> cone;
  cone->SetCenter(-2.0, 0.0, 0.0);
  cone->SetResolution(24);
  cone->Update();

  vtkNew<vtkSphereSource> sphere;
  sphere->SetCenter(0.0, 0.0, 0.0);
  sphere->SetThetaResolution(24);
  sphere->SetPhiResolution(24);
  sphere->Update();

  vtkNew<vtkCubeSource> cube;
  cube->SetCenter(2.0, 0.0, 0.0);
  cube->Update();

  pdc->SetPartition(0, 0, cone->GetOutput());
  pdc->SetPartition(1, 0, sphere->GetOutput());
  pdc->SetPartition(2, 0, cube->GetOutput());

  vtkNew<MetalCompositePolyDataMapper> mapper;
  mapper->SetInputDataObject(pdc);

  vtkNew<vtkCocoaMetalRenderWindow> renWin;
  renWin->SetSize(600, 300);
  renWin->SetMultiSamples(0);

  vtkNew<vtkMetalRenderer> renderer;
  renderer->SetBackground(0.2, 0.3, 0.4);
  renWin->AddRenderer(renderer);

  vtkNew<vtkMetalCamera> camera;
  renderer->SetActiveCamera(camera);

  vtkNew<vtkMetalActor> actor;
  actor->SetMapper(mapper);
  renderer->AddActor(actor);

  renderer->ResetCamera();
  renderer->GetActiveCamera()->Zoom(1.4);
  renderer->ResetCameraClippingRange();

  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }
  if (!vtkMetalTesting::CheckBackend(renWin))
  {
    return EXIT_FAILURE;
  }

  if (!vtkMetalTesting::VerifyRegionRendered(renWin, 80, 80, 199, 219, 300))
  {
    std::cerr << "First block (cone) not rendered" << std::endl;
    return EXIT_FAILURE;
  }
  if (!vtkMetalTesting::VerifyRegionRendered(renWin, 250, 80, 349, 219, 300))
  {
    std::cerr << "Second block (sphere) not rendered" << std::endl;
    return EXIT_FAILURE;
  }
  if (!vtkMetalTesting::VerifyRegionRendered(renWin, 400, 80, 519, 219, 300))
  {
    std::cerr << "Third block (cube) not rendered" << std::endl;
    return EXIT_FAILURE;
  }

  if (vtkMetalTesting::CountCompositePixels(renWin, 0, 0, 599, 299) == 0)
  {
    std::cerr << "No pixels with a non-zero composite index were found" << std::endl;
    return EXIT_FAILURE;
  }

  // Re-render to exercise the batched-mapper geometry-rebuild and reuse paths.
  renderer->GetActiveCamera()->Azimuth(30.0);
  renderer->ResetCameraClippingRange();
  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    return EXIT_FAILURE;
  }
  if (!vtkMetalTesting::VerifyRegionRendered(renWin, 80, 80, 519, 219, 500))
  {
    return EXIT_FAILURE;
  }

  // Image-based regression against a baseline (final rotated frame).
  return vtkMetalTesting::RegressionExitCode(vtkRegressionTestImage(renWin));
}
