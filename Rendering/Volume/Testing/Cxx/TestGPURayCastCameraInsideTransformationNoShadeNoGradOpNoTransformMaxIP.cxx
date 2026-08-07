// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

/// Description:
/// Same contained scene as NoShadeNoGradOpNoTransform but with MAXIMUM
/// INTENSITY blend mode on both backends. Removes opacity-weighted front-to-back
/// compositing from the picture: if the Metal vs OpenGL residual divergence
/// vanishes, it lives in the color/opacity accumulation path; if it persists,
/// it lives in the scalar fetch / ray domain.

#include "vtkCamera.h"
#include "vtkColorTransferFunction.h"
#include "vtkGPUVolumeRayCastMapper.h"
#include "vtkImageData.h"
#include "vtkImageResize.h"
#include "vtkInteractorStyleTrackballCamera.h"
#include "vtkNew.h"
#include "vtkPiecewiseFunction.h"
#include "vtkRegressionTestImage.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderer.h"
#include "vtkTestUtilities.h"
#include "vtkVolume.h"
#include "vtkVolume16Reader.h"
#include "vtkVolumeProperty.h"

#include <iostream>

static const char* TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformMaxIPLog =
  "# StreamVersion 1\n"
  "EnterEvent 298 27 0 0 0 0 0\n"
  "MouseWheelForwardEvent 200 142 0 0 0 0 0\n"
  "LeaveEvent 311 71 0 0 0 0 0\n";

int TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformMaxIP(int argc, char* argv[])
{
  // Load data
  vtkNew<vtkVolume16Reader> reader;
  reader->SetDataDimensions(64, 64);
  reader->SetImageRange(1, 93);
  reader->SetDataByteOrderToLittleEndian();
  char* fname = vtkTestUtilities::ExpandDataFileName(argc, argv, "Data/headsq/quarter");
  reader->SetFilePrefix(fname);
  delete[] fname;
  reader->SetDataSpacing(3.2, 3.2, 1.5);

  // Upsample data
  vtkNew<vtkImageResize> resample;
  resample->SetInputConnection(reader->GetOutputPort());
  resample->SetResizeMethodToOutputDimensions();
  resample->SetOutputDimensions(512, 512, 512);
  resample->Update();

  // Prepare TFs
  vtkNew<vtkColorTransferFunction> ctf;
  ctf->AddRGBPoint(0, 0.0, 0.0, 0.0);
  ctf->AddRGBPoint(500, 1.0, 0.5, 0.3);
  ctf->AddRGBPoint(1000, 1.0, 0.5, 0.3);
  ctf->AddRGBPoint(1150, 1.0, 1.0, 0.9);

  vtkNew<vtkPiecewiseFunction> pf;
  pf->AddPoint(0, 0.00);
  pf->AddPoint(500, 0.02);
  pf->AddPoint(1000, 0.02);
  pf->AddPoint(1150, 0.85);

  vtkNew<vtkPiecewiseFunction> gf;
  vtkNew<vtkVolumeProperty> volumeProperty;
  volumeProperty->SetScalarOpacity(pf);
  volumeProperty->SetColor(ctf);
  volumeProperty->ShadeOff();

  // Setup rendering context
  vtkNew<vtkRenderWindow> renWin;
  renWin->SetSize(512, 512);
  renWin->SetMultiSamples(0);

  vtkNew<vtkRenderer> ren;
  renWin->AddRenderer(ren);
  ren->SetBackground(0.1, 0.1, 0.1);

  vtkNew<vtkGPUVolumeRayCastMapper> mapper;
  mapper->SetInputConnection(resample->GetOutputPort());
  mapper->SetBlendModeToMaximumIntensity();

  vtkNew<vtkVolume> volume;
  volume->SetMapper(mapper);
  volume->SetProperty(volumeProperty);
  ren->AddVolume(volume);

  // Prepare the camera to be inside the volume (no vtkProp3D transformation)
  ren->ResetCamera();
  ren->GetActiveCamera()->SetPosition(102.4, 102.4, 60);

  // Initialize interactor
  vtkNew<vtkRenderWindowInteractor> iren;
  iren->SetRenderWindow(renWin);

  vtkNew<vtkInteractorStyleTrackballCamera> style;
  iren->SetInteractorStyle(style);

  renWin->Render();
  iren->Initialize();

  int rv = vtkTesting::InteractorEventLoop(
    argc, argv, iren, TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformMaxIPLog);
  return rv;
}
