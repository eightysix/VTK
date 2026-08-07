// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

/// Description:
/// Camera outside the volume (no near-plane clipping), no shading / no
/// gradient opacity / no vtkProp3D transform. Reads VTK_FIXED_SAMPLE_DISTANCE
/// (world units) and VTK_FIXED_AUTO_ADJUST (0/1) from the environment; when
/// the sample distance is set, AutoAdjustSampleDistances is disabled so the
/// requested step is honored by both backends, letting us sweep step sizes to
/// find a value where Metal and OpenGL converge. Sweep results are recorded in
/// Rendering/Metal/TestGPURayCastCameraInsideTransformation.md: no step size
/// makes the backends match, so the residual divergence is not a
/// sampling-phase artifact.

#include "vtkCamera.h"
#include "vtkColorTransferFunction.h"
#include "vtkGPUVolumeRayCastMapper.h"
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

#include <cstdlib>
#include <iostream>

static const char* TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideFixedStepLog =
  "# StreamVersion 1\n"
  "EnterEvent 298 27 0 0 0 0 0\n"
  "MouseWheelForwardEvent 200 142 0 0 0 0 0\n"
  "LeaveEvent 311 71 0 0 0 0 0\n";

int TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideFixedStep(
  int argc, char* argv[])
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

  const char* sd = std::getenv("VTK_FIXED_SAMPLE_DISTANCE");
  if (sd && atof(sd) > 0.0)
  {
    mapper->AutoAdjustSampleDistancesOff();
    mapper->SetSampleDistance(atof(sd));
  }
  const char* autoAdj = std::getenv("VTK_FIXED_AUTO_ADJUST");
  if (autoAdj && atoi(autoAdj) == 1)
  {
    mapper->AutoAdjustSampleDistancesOn();
  }
  std::cerr << "PROBE fixedSD=" << (sd ? sd : "(unset)") << " autoAdjust="
            << mapper->GetAutoAdjustSampleDistances() << std::endl;

  vtkNew<vtkVolume> volume;
  volume->SetMapper(mapper);
  volume->SetProperty(volumeProperty);
  ren->AddVolume(volume);

  // Prepare the camera to be outside the volume (no near-plane clipping)
  ren->ResetCamera();
  ren->GetActiveCamera()->SetPosition(102.4, 102.4, 400);

  // Initialize interactor
  vtkNew<vtkRenderWindowInteractor> iren;
  iren->SetRenderWindow(renWin);

  vtkNew<vtkInteractorStyleTrackballCamera> style;
  iren->SetInteractorStyle(style);

  renWin->Render();
  iren->Initialize();

  int rv = vtkTesting::InteractorEventLoop(argc, argv, iren,
    TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideFixedStepLog);
  return rv;
}
