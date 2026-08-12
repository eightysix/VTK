// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

/// Description:
/// Knob-sweep variant of TestGPURayCastCameraInsideNonUniformScaleTransform
/// (camera inside the bounding box of a volume with a NON-UNIFORM scale poke
/// matrix diag(3.2,3.2,1.5) + translate (200,100,40), large view angle 170).
/// Each knob is independently toggleable via environment variables so a
/// Metal-vs-OpenGL pixel diff can be attributed to one subsystem at a time:
///
///   VTK_NUS_JITTER  0 = SetUseJittering(false)   (default 1 = jitter on)
///   VTK_NUS_SHADE   0 = ShadeOff                 (default 1 = shading on)
///   VTK_NUS_GRADOP  0 = no gradient opacity      (default 1 = gradient opacity)
///   VTK_NUS_POKE    0 = no poke matrix           (default 1 = poke matrix)
///   VTK_NUS_RAW_CAPTURE <file> = raw front-buffer capture (frame aligned,
///                                bypassing the vtkWindowToImageFilter camera
///                                float32 view-angle round trip)

#include "vtkCamera.h"
#include "vtkColorTransferFunction.h"
#include "vtkGPUVolumeRayCastMapper.h"
#include "vtkImageData.h"
#include "vtkMatrix4x4.h"
#include "vtkNew.h"
#include "vtkPNGWriter.h"
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

int TestGPURayCastCameraInsideNonUniformScaleTransformKnobs(int argc, char* argv[])
{
  int jitter = 1;
  if (const char* env = std::getenv("VTK_NUS_JITTER"))
  {
    jitter = std::atoi(env);
  }
  int shade = 1;
  if (const char* env = std::getenv("VTK_NUS_SHADE"))
  {
    shade = std::atoi(env);
  }
  int gradOp = 1;
  if (const char* env = std::getenv("VTK_NUS_GRADOP"))
  {
    gradOp = std::atoi(env);
  }
  int poke = 1;
  if (const char* env = std::getenv("VTK_NUS_POKE"))
  {
    poke = std::atoi(env);
  }

  // Load data
  vtkNew<vtkVolume16Reader> reader;
  reader->SetDataDimensions(64, 64);
  reader->SetImageRange(1, 93);
  reader->SetDataByteOrderToLittleEndian();
  char* fname = vtkTestUtilities::ExpandDataFileName(argc, argv, "Data/headsq/quarter");
  reader->SetFilePrefix(fname);
  delete[] fname;
  reader->SetDataSpacing(1, 1, 1);

  double elements[16] = { 3.2, 0, 0, 200, 0, 3.2, 0, 100, 0, 0, 1.5, 40, 0, 0, 0, 1 };

  vtkNew<vtkMatrix4x4> matrix;
  matrix->DeepCopy(elements);

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
  gf->AddPoint(0, 0.0);
  gf->AddPoint(90, 0.5);
  gf->AddPoint(100, 0.7);

  vtkNew<vtkVolumeProperty> volumeProperty;
  volumeProperty->SetScalarOpacity(pf);
  if (gradOp)
  {
    volumeProperty->SetGradientOpacity(gf);
  }
  volumeProperty->SetColor(ctf);
  if (shade)
  {
    volumeProperty->ShadeOn();
  }

  // Setup rendering context
  vtkNew<vtkRenderWindow> renWin;
  renWin->SetSize(300, 300);
  renWin->SetMultiSamples(0);

  vtkNew<vtkRenderer> ren;
  renWin->AddRenderer(ren);
  ren->SetBackground(0.1, 0.1, 0.1);

  vtkNew<vtkGPUVolumeRayCastMapper> mapper;
  mapper->SetInputConnection(reader->GetOutputPort());
  mapper->SetUseJittering(jitter != 0);

  vtkNew<vtkVolume> volume;
  volume->SetMapper(mapper);
  volume->SetProperty(volumeProperty);
  if (poke)
  {
    volume->PokeMatrix(matrix);
  }
  ren->AddVolume(volume);

  // Prepare the camera to be inside the volume
  ren->ResetCamera();
  vtkCamera* cam = ren->GetActiveCamera();
  cam->SetViewAngle(170);
  cam->SetPosition(256.846, 168.853, 38.7375);
  cam->SetFocalPoint(178.423, 110.943, 142.038);
  cam->SetViewUp(-0.105083, 0.899357, 0.424399);
  ren->ResetCameraClippingRange();

  // Initialize interactor
  vtkNew<vtkRenderWindowInteractor> iren;
  iren->SetRenderWindow(renWin);

  renWin->Render();
  iren->Initialize();
  renWin->Render();

  if (const char* raw = std::getenv("VTK_NUS_RAW_CAPTURE"))
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

  int retVal = vtkRegressionTestImage(renWin);
  if (retVal == vtkRegressionTester::DO_INTERACTOR)
  {
    iren->Start();
  }

  return !retVal;
}
