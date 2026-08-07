// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

/// Description:
/// Variant of TestGPURayCastCameraInsideTransformationNoShade that maximizes
/// the Metal/OpenGL divergence. Three amplifiers combined:
///   - ShadeOff: unshaded composite exposes the full dynamic range.
///   - Steep gradient-opacity ramp (0@0 -> 1@90 -> 1@2000): the gf table is
///     built on the CPU identically for both backends, so the ramp amplifies
///     any per-sample gradient-magnitude (gradW) difference between Metal and
///     OpenGL into large opacity swings. Placed over the interior sample
///     distribution's p10-p90 (gf input 1-90 data units).
///   - AutoAdjustSampleDistancesOff + SetSampleDistance(0.0675): a fixed fine
///     step, 4x the auto-adjusted default, which maximizes the per-sample
///     phase/comb divergence (the finest step in the fixed-step sweep was the
///     worst case). VTK_FIXED_SAMPLE_DISTANCE overrides the step (like the
///     CamOutsideFixedStep variant); VTK_FIXED_AUTO_ADJUST=1 re-enables
///     auto-adjust.
/// Camera is inside the axis-aligned volume so every ray traverses a long
/// interior path (many samples per ray).

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

#include <iostream>

static const char* TestGPURayCastCameraInsideTransformationNoShadeAmpLog =
  "# StreamVersion 1\n"
  "EnterEvent 298 27 0 0 0 0 0\n"
  "MouseWheelForwardEvent 200 142 0 0 0 0 0\n"
  "LeaveEvent 311 71 0 0 0 0 0\n";

int TestGPURayCastCameraInsideTransformationNoShadeAmp(int argc, char* argv[])
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

  // Steep gradient-opacity ramp over the interior sample distribution
  // (gf input p10-p90 = 1-90 data units): amplifies any per-sample gradW
  // difference between the backends into a large opacity swing.
  vtkNew<vtkPiecewiseFunction> gf;
  gf->AddPoint(0, 0.0);
  gf->AddPoint(90, 1.0);
  gf->AddPoint(2000, 1.0);

  vtkNew<vtkVolumeProperty> volumeProperty;
  volumeProperty->SetScalarOpacity(pf);
  volumeProperty->SetGradientOpacity(gf);
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

  // Fine fixed step: maximizes the per-sample phase/comb divergence. The
  // default (auto-adjusted ~0.27 world units) is overridden by this fixed
  // step; VTK_FIXED_SAMPLE_DISTANCE re-overrides for a sweep. 0.008 world
  // units is the finest step that keeps both backend renders structurally
  // comparable (finer steps drive the Metal composite toward collapse).
  const char* sd = std::getenv("VTK_FIXED_SAMPLE_DISTANCE");
  double fixedSd = (sd && atof(sd) > 0.0) ? atof(sd) : 0.008;
  mapper->AutoAdjustSampleDistancesOff();
  mapper->SetSampleDistance(fixedSd);
  const char* autoAdj = std::getenv("VTK_FIXED_AUTO_ADJUST");
  if (autoAdj && atoi(autoAdj) == 1)
  {
    mapper->AutoAdjustSampleDistancesOn();
  }
  std::cerr << "PROBE fixedSD=" << fixedSd << " autoAdjust="
            << mapper->GetAutoAdjustSampleDistances() << std::endl;

  vtkNew<vtkVolume> volume;
  volume->SetMapper(mapper);
  volume->SetProperty(volumeProperty);
  ren->AddVolume(volume);

  // Prepare the camera to be inside the volume (long rays, no vtkProp3D
  // transformation)
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
    argc, argv, iren, TestGPURayCastCameraInsideTransformationNoShadeAmpLog);
  return rv;
}
