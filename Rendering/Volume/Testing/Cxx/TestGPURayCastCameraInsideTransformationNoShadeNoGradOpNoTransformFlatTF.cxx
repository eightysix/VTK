// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

/// Description:
/// Same contained NoJitter scene as NoShadeNoGradOpNoTransformNoJitter but with
/// a CONSTANT color and opacity transfer function over the whole scalar range.
/// With nearest TF sampling the per-sample color/opacity is then identical for
/// every sample and every ray, so the front-to-back composite becomes a pure
/// geometric-series accumulation of constant (rgb, opacity) over the ray's
/// sample count: any Metal vs OpenGL difference must come from the
/// accumulation arithmetic itself or from a differing sample count, never from
/// the scalar/TF lookup path.

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

static const char* TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformFlatTFLog =
  "# StreamVersion 1\n"
  "EnterEvent 298 27 0 0 0 0 0\n"
  "MouseWheelForwardEvent 200 142 0 0 0 0 0\n"
  "LeaveEvent 311 71 0 0 0 0 0\n";

int TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformFlatTF(int argc, char* argv[])
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

  // Constant color and opacity over the whole data range.
  vtkNew<vtkColorTransferFunction> ctf;
  ctf->AddRGBPoint(0, 0.9, 0.7, 0.5);
  ctf->AddRGBPoint(4370, 0.9, 0.7, 0.5);

  vtkNew<vtkPiecewiseFunction> pf;
  pf->AddPoint(0, 0.1);
  pf->AddPoint(4370, 0.1);

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

  int rv = vtkTesting::InteractorEventLoop(
    argc, argv, iren, TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformFlatTFLog);
  return rv;
}
