// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

// Test actor texture support in the Metal poly data mapper.

#include "TestMetalHelpers.h"

#include "vtkImageData.h"
#include "vtkMetalActor.h"
#include "vtkMetalCamera.h"
#include "vtkMetalPolyDataMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkNew.h"
#include "vtkPlaneSource.h"
#include "vtkTexture.h"

#include <iostream>

namespace
{

// 64x64 RGB checkerboard image used as the actor texture.
vtkSmartPointer<vtkImageData> CreateCheckerboardImage()
{
  constexpr int dim = 64;
  vtkNew<vtkImageData> image;
  image->SetDimensions(dim, dim, 1);
  image->SetSpacing(1, 1, 1);
  image->SetOrigin(0, 0, 0);
  image->AllocateScalars(VTK_UNSIGNED_CHAR, 3);

  unsigned char* ptr = static_cast<unsigned char*>(image->GetScalarPointer());
  for (int y = 0; y < dim; ++y)
  {
    for (int x = 0; x < dim; ++x)
    {
      const bool white = ((x / 8) + (y / 8)) % 2 == 0;
      const unsigned char v = white ? 255 : 32;
      ptr[(y * dim + x) * 3 + 0] = v;
      ptr[(y * dim + x) * 3 + 1] = white ? 200 : 32;
      ptr[(y * dim + x) * 3 + 2] = white ? 100 : 32;
    }
  }
  return image;
}

}

int TestMetalTexture(int argc, char* argv[])
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

  vtkNew<vtkPlaneSource> plane;
  plane->SetResolution(8, 8);

  vtkNew<vtkMetalPolyDataMapper> mapper;
  mapper->SetInputConnection(plane->GetOutputPort());

  vtkNew<vtkTexture> texture;
  texture->InterpolateOn();
  texture->RepeatOff();
  texture->EdgeClampOn();
  texture->SetInputData(CreateCheckerboardImage());

  vtkNew<vtkMetalActor> actor;
  actor->SetMapper(mapper);
  actor->SetTexture(texture);

  renderer->AddActor(actor);
  vtkNew<vtkMetalCamera> camera;
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
  renderer->GetActiveCamera()->Azimuth(-20);
  renderer->GetActiveCamera()->Elevation(20);

  vtkMetalTesting::RenderAndWait(renWin);

  // The textured plane fills the viewport center.
  if (!vtkMetalTesting::VerifyRegionRendered(renWin, renderer, 150, 75, 450, 225, 2000))
  {
    std::cerr << "Textured plane did not fill the expected viewport region." << std::endl;
    return EXIT_FAILURE;
  }

  // Re-render: exercises the cached texture upload path.
  vtkMetalTesting::RenderAndWait(renWin);

  if (!vtkMetalTesting::VerifyRegionRendered(renWin, renderer, 150, 75, 450, 225, 2000))
  {
    std::cerr << "Textured plane rendering changed after a re-render." << std::endl;
    return EXIT_FAILURE;
  }

  // Image-based regression against a baseline (textured plane).
  return vtkMetalTesting::RegressionExitCode(vtkRegressionTestImage(renWin));
}
