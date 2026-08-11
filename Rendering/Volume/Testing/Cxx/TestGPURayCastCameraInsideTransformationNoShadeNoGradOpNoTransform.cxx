// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

/// Description:
/// Tests clipping of a volume (no vtkProp3D transformation) using the camera's
/// near plane while the camera is inside the volume.

#include "vtkCamera.h"
#include "vtkColorTransferFunction.h"
#include "vtkGPUVolumeRayCastMapper.h"
#include "vtkImageData.h"
#include "vtkImageResize.h"
#include "vtkPNGWriter.h"
#include "vtkPointData.h"
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
#include <cstring>
#include <iostream>

static const char* TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformLog =
  "# StreamVersion 1\n"
  "EnterEvent 298 27 0 0 0 0 0\n"
  "MouseWheelForwardEvent 200 142 0 0 0 0 0\n"
  "LeaveEvent 311 71 0 0 0 0 0\n";

int TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransform(int argc, char* argv[])
{
  // std::cout << "CTEST_FULL_OUTPUT (Avoid ctest truncation of output)" << std::endl;

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
  {
    vtkImageData* out = resample->GetOutput();
    vtkDataArray* sa = out->GetPointData()->GetScalars();
    double r[2];
    sa->GetRange(r);
    double fr[2];
    sa->GetFiniteRange(fr);
    const unsigned short* sp = static_cast<const unsigned short*>(sa->GetVoidPointer(0));
    std::cerr << "VTK_METAL_VOLUME_LOG DEBUG TEST_RESAMPLE dt=" << sa->GetDataType()
              << " dims=" << out->GetDimensions()[0] << "x" << out->GetDimensions()[1] << "x"
              << out->GetDimensions()[2] << " range=(" << r[0] << "," << r[1]
              << ") finite=(" << fr[0] << "," << fr[1] << ") first=" << sp[0] << " "
              << sp[1] << " " << sp[65536] << std::endl;
  }

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
    argc, argv, iren, TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformLog);

  // Raw front-buffer capture (frame-aligned backend comparison). Reads the
  // framebuffer of the last frame rendered (the W2IF perturbed camera copy for
  // both backends), bypassing the -V baseline PNG so backend render frames can
  // be diffed directly without harness frame-selection noise.
  if (const char* raw = std::getenv("VTK_STEP_RAW_CAPTURE"))
  {
    int* wsz = renWin->GetSize();
    unsigned char* px = renWin->GetRGBACharPixelData(0, 0, wsz[0] - 1, wsz[1] - 1, 1);
    if (px)
    {
      vtkNew<vtkImageData> img;
      img->SetDimensions(wsz[0], wsz[1], 1);
      img->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
      unsigned char* out = static_cast<unsigned char*>(img->GetScalarPointer());
      for (int i = 0; i < wsz[0] * wsz[1]; ++i)
      {
        out[3 * i + 0] = px[4 * i + 0];
        out[3 * i + 1] = px[4 * i + 1];
        out[3 * i + 2] = px[4 * i + 2];
      }
      delete[] px;
      vtkNew<vtkPNGWriter> png;
      png->SetFileName(raw);
      png->SetInputData(img);
      png->Write();
    }
  }

  return rv;
}
