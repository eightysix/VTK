// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

/// Description:
/// Same contained NoJitter scene as NoShadeNoGradOpNoTransformNoJitter but with
/// a single HARD STEP in both color and opacity at a sweepable scalar threshold.
/// Below the threshold every sample is color A / opacity a; above it color B /
/// opacity b. With nearest TF + volume sampling, the per-sample color is then a
/// pure function of which side of the step the sample's scalar lands on, so a
/// Metal vs OpenGL divergence can only come from a sample whose interpolated
/// scalar (interpolated-anchor / nearest-texel selection) crosses the step on
/// one backend and not the other. Sweeping the threshold (VTK_STEP_THRESHOLD)
/// moves the knife edge through the scalar domain and maps which texel-boundary
/// scalars are knife-edge-sensitive.
///
/// Modes (VTK_STEP_MODE):
///   0 (default): both color and opacity step at the threshold
///   1: only color steps (opacity constant, isolates the color-lookup path)
///   2: only opacity steps (color constant, isolates the opacity-lookup path)

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

#include <cstdlib>
#include <iostream>

static const char* TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformStepTFLog =
  "# StreamVersion 1\n"
  "EnterEvent 298 27 0 0 0 0 0\n"
  "MouseWheelForwardEvent 200 142 0 0 0 0 0\n"
  "LeaveEvent 311 71 0 0 0 0 0\n";

int TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformStepTF(
  int argc, char* argv[])
{
  // Sweepable step threshold in scalar units (default mid-range).
  double threshold = 2000.0;
  if (const char* env = std::getenv("VTK_STEP_THRESHOLD"))
  {
    threshold = std::atof(env);
  }
  int mode = 0;
  if (const char* env = std::getenv("VTK_STEP_MODE"))
  {
    mode = std::atoi(env);
  }
  int linear = 0;
  if (const char* env = std::getenv("VTK_STEP_LINEAR"))
  {
    linear = std::atoi(env);
  }

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

  // Hard step in color: A below threshold, B above.
  vtkNew<vtkColorTransferFunction> ctf;
  if (mode == 2)
  {
    // color constant (isolate opacity path)
    ctf->AddRGBPoint(0, 0.9, 0.7, 0.5);
    ctf->AddRGBPoint(4370, 0.9, 0.7, 0.5);
  }
  else
  {
    const double half = std::max(1.0, 0.001 * threshold);
    ctf->AddRGBPoint(0, 0.2, 0.2, 0.2);
    ctf->AddRGBPoint(threshold - half, 0.2, 0.2, 0.2);
    ctf->AddRGBPoint(threshold + half, 0.9, 0.9, 0.9);
    ctf->AddRGBPoint(4370, 0.9, 0.9, 0.9);
  }

  // Hard step in opacity: a below threshold, b above.
  vtkNew<vtkPiecewiseFunction> pf;
  if (mode == 1)
  {
    // opacity constant (isolate color path)
    pf->AddPoint(0, 0.1);
    pf->AddPoint(4370, 0.1);
  }
  else
  {
    const double half = std::max(1.0, 0.001 * threshold);
    pf->AddPoint(0, 0.02);
    pf->AddPoint(threshold - half, 0.02);
    pf->AddPoint(threshold + half, 0.85);
    pf->AddPoint(4370, 0.85);
  }

  vtkNew<vtkVolumeProperty> volumeProperty;
  volumeProperty->SetScalarOpacity(pf);
  volumeProperty->SetColor(ctf);
  volumeProperty->ShadeOff();
  if (linear)
  {
    volumeProperty->SetInterpolationTypeToLinear();
  }

  // Setup rendering context
  vtkNew<vtkRenderWindow> renWin;
  renWin->SetSize(512, 512);
  renWin->SetMultiSamples(0);

  vtkNew<vtkRenderer> ren;
  renWin->AddRenderer(ren);
  ren->SetBackground(0.1, 0.1, 0.1);

  vtkNew<vtkGPUVolumeRayCastMapper> mapper;
  mapper->SetInputConnection(resample->GetOutputPort());
  mapper->SetUseJittering(false);

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

  int rv = vtkTesting::InteractorEventLoop(argc, argv, iren,
    TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformStepTFLog);
  return rv;
}
