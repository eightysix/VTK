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
/// Renders a SINGLE STILL FRAME (no interactor / no recorded-event replay) and
/// compares via vtkRegressionTestImage.
///
/// Modes (VTK_STEP_MODE):
///   0 (default): both color and opacity step at the threshold
///   1: only color steps (opacity constant, isolates the color-lookup path)
///   2: only opacity steps (color constant, isolates the opacity-lookup path)
///   3: constant color, opacity LINEAR RAMP over the full scalar range (no
///      step). If the mode-2 divergence survives this, it is a per-sample
///      varying-opacity value/accumulation difference, not step-crossing
///      sensitivity; if it collapses, the divergence is isosurface-classification
///      flips at the step boundary.
///   4: color LINEAR RAMP over the full scalar range, opacity CONSTANT at a
///      low non-saturating value (VTK_STEP_OPACITY) — the color-path mirror of
///      mode 3 (variant A).
///   5: constant color, opacity LINEAR RAMP confined to a scalar window
///      [VTK_STEP_WIN0, VTK_STEP_WIN1] (flat elsewhere) — moves the sensitive
///      dO/ds region through the scalar domain (variant C).
///
/// Additional variants (orthogonal env knobs):
///   VTK_STEP_DIMS     volume side dimension for the upsample (default 512;
///                     256 for half density, 0 = no upsample / raw 64x64x93).
///                     (variant D)
///   VTK_STEP_CONSTANT if set to a value, replaces the headsq data with a
///                     CONSTANT-scalar volume of that value (no data gradient,
///                     still varying ray length) — variant B control.
///   VTK_CAMERA_AXIS   if set to "z", points the camera straight down the
///                     volume z axis (rays parallel to z, axis-aligned
///                     interpolation direction) — variant E.
///   VTK_STEP_LINEAR   1 = linear volume interpolation (default nearest).
///   VTK_STEP_RAMP_MAX mode 3/5 ramp endpoint opacity (default 0.02).
///   VTK_STEP_OPACITY  mode 4 constant opacity (default 0.005).
///   VTK_STEP_WIN0/1   mode 5 ramp window in scalar units.

#include "vtkCamera.h"
#include "vtkAlgorithmOutput.h"
#include "vtkColorTransferFunction.h"
#include "vtkGPUVolumeRayCastMapper.h"
#include "vtkImageData.h"
#include "vtkImageResize.h"
#include "vtkInteractorStyleTrackballCamera.h"
#include "vtkNew.h"
#include "vtkPNGWriter.h"
#include "vtkPiecewiseFunction.h"
#include "vtkRegressionTestImage.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderer.h"
#include "vtkTestUtilities.h"
#include "vtkTrivialProducer.h"
#include "vtkVolume.h"
#include "vtkVolume16Reader.h"
#include "vtkVolumeProperty.h"
#include "vtkWindowToImageFilter.h"

#include <algorithm>
#include <cstdlib>

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
  double rampMax = 0.02;
  if (const char* env = std::getenv("VTK_STEP_RAMP_MAX"))
  {
    rampMax = std::atof(env);
  }
  double constOpacity = 0.005;
  if (const char* env = std::getenv("VTK_STEP_OPACITY"))
  {
    constOpacity = std::atof(env);
  }
  double win0 = 1500.0, win1 = 2500.0;
  if (const char* env = std::getenv("VTK_STEP_WIN0"))
  {
    win0 = std::atof(env);
  }
  if (const char* env = std::getenv("VTK_STEP_WIN1"))
  {
    win1 = std::atof(env);
  }
  int dims = 512;
  if (const char* env = std::getenv("VTK_STEP_DIMS"))
  {
    dims = std::atoi(env);
  }
  int constantScalar = 0; // 0 = use headsq data
  if (const char* env = std::getenv("VTK_STEP_CONSTANT"))
  {
    constantScalar = std::atoi(env);
  }
  int cameraAxis = 0; // 0 = reference camera, 1 = axis-aligned (z)
  if (const char* env = std::getenv("VTK_CAMERA_AXIS"))
  {
    cameraAxis = std::atoi(env);
  }

  // Load data
  vtkSmartPointer<vtkAlgorithmOutput> dataPort;
  vtkSmartPointer<vtkVolume16Reader> reader;
  vtkSmartPointer<vtkTrivialProducer> producer;
  if (constantScalar > 0)
  {
    // Variant B: constant-scalar volume (no data gradient). Same 64x64x93
    // extent/spacing as the headsq reader so the resize path is identical.
    vtkNew<vtkImageData> img;
    img->SetDimensions(64, 64, 93);
    img->SetSpacing(3.2, 3.2, 1.5);
    img->SetOrigin(0.0, 0.0, 0.0);
    img->AllocateScalars(VTK_UNSIGNED_SHORT, 1);
    vtkIdType n = img->GetNumberOfPoints();
    unsigned short* p = static_cast<unsigned short*>(img->GetScalarPointer());
    std::fill(p, p + n, static_cast<unsigned short>(constantScalar));
    producer = vtkSmartPointer<vtkTrivialProducer>::New();
    producer->SetOutput(img);
    dataPort = producer->GetOutputPort();
  }
  else
  {
    reader = vtkSmartPointer<vtkVolume16Reader>::New();
    reader->SetDataDimensions(64, 64);
    reader->SetImageRange(1, 93);
    reader->SetDataByteOrderToLittleEndian();
    char* fname = vtkTestUtilities::ExpandDataFileName(argc, argv, "Data/headsq/quarter");
    reader->SetFilePrefix(fname);
    delete[] fname;
    reader->SetDataSpacing(3.2, 3.2, 1.5);
    dataPort = reader->GetOutputPort();
  }

  // Upsample data
  vtkNew<vtkImageResize> resample;
  resample->SetInputConnection(dataPort);
  resample->SetResizeMethodToOutputDimensions();
  if (dims > 0)
  {
    resample->SetOutputDimensions(dims, dims, dims);
  }
  else
  {
    resample->SetOutputDimensions(64, 64, 93);
  }
  resample->Update();

  // Color transfer function.
  vtkNew<vtkColorTransferFunction> ctf;
  if (mode == 2 || mode == 3 || mode == 5)
  {
    // color constant (isolate opacity path)
    ctf->AddRGBPoint(0, 0.9, 0.7, 0.5);
    ctf->AddRGBPoint(4370, 0.9, 0.7, 0.5);
  }
  else if (mode == 4)
  {
    // color LINEAR RAMP over full scalar range (variant A)
    ctf->AddRGBPoint(0, 0.2, 0.2, 0.2);
    ctf->AddRGBPoint(4370, 0.9, 0.9, 0.9);
  }
  else
  {
    const double half = std::max(1.0, 0.001 * threshold);
    ctf->AddRGBPoint(0, 0.2, 0.2, 0.2);
    ctf->AddRGBPoint(threshold - half, 0.2, 0.2, 0.2);
    ctf->AddRGBPoint(threshold + half, 0.9, 0.9, 0.9);
    ctf->AddRGBPoint(4370, 0.9, 0.9, 0.9);
  }

  // Opacity transfer function.
  vtkNew<vtkPiecewiseFunction> pf;
  if (mode == 1)
  {
    // opacity constant (isolate color path)
    pf->AddPoint(0, 0.1);
    pf->AddPoint(4370, 0.1);
  }
  else if (mode == 4)
  {
    // opacity CONSTANT low (variant A; kept low to avoid alpha saturation)
    pf->AddPoint(0, constOpacity);
    pf->AddPoint(4370, constOpacity);
  }
  else if (mode == 3)
  {
    // opacity LINEAR RAMP over full scalar range (no step). The default max
    // (0.02) was chosen to avoid saturating the accumulated alpha to ~1
    // (a saturated constant-color image is byte-identical regardless of
    // accumulation, i.e. degenerate — see update 65). Sweep VTK_STEP_RAMP_MAX
    // to stay in a sensitive non-saturated range.
    pf->AddPoint(0, 0.0);
    pf->AddPoint(4370, rampMax);
  }
  else if (mode == 5)
  {
    // opacity LINEAR RAMP confined to a scalar window (variant C)
    pf->AddPoint(0, 0.0);
    pf->AddPoint(win0, 0.0);
    pf->AddPoint(win1, rampMax);
    pf->AddPoint(4370, rampMax);
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
  if (cameraAxis)
  {
    // Variant E: axis-aligned camera looking straight down the z axis
    ren->GetActiveCamera()->SetPosition(102.4, 102.4, 100.0);
    ren->GetActiveCamera()->SetFocalPoint(102.4, 102.4, 30.0);
    ren->GetActiveCamera()->SetViewUp(0.0, 1.0, 0.0);
  }
  else
  {
    ren->GetActiveCamera()->SetPosition(102.4, 102.4, 60);
  }

  // Initialize interactor
  vtkNew<vtkRenderWindowInteractor> iren;
  iren->SetRenderWindow(renWin);
  vtkNew<vtkInteractorStyleTrackballCamera> style;
  iren->SetInteractorStyle(style);

  renWin->Render();
  iren->Initialize();
  renWin->Render();

  // Replicate the reference's single MouseWheelForwardEvent without the
  // event loop: OnMouseWheelForward -> Dolly(pow(1.1, MotionFactor*0.2*
  // MouseWheelMotionFactor)) = Dolly(1.21) + ResetCameraClippingRange().
  if (std::getenv("VTK_STEP_WHEEL") != nullptr)
  {
    ren->GetActiveCamera()->Dolly(1.21);
    ren->ResetCameraClippingRange();
    renWin->Render();
  }

  if (const char* raw = std::getenv("VTK_STEP_RAW_CAPTURE"))
  {
    // Capture the front framebuffer of the frame that was just rendered,
    // bypassing vtkWindowToImageFilter's camera copy (vtkWindowToImageFilter.cxx
    // stores the view angle as float32 radians, re-deriving 30.0 -> 30.0000008,
    // so a W2IF render is not the frame this test drew and can flip knife-edge
    // samples ~4 ulp). The raw readback is byte-identical to the last render.
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

  if (const char* dump = std::getenv("VTK_STEP_DUMP"))
  {
    vtkNew<vtkWindowToImageFilter> w2i;
    w2i->SetInput(renWin);
    w2i->Update();
    vtkNew<vtkPNGWriter> png;
    png->SetFileName(dump);
    png->SetInputConnection(w2i->GetOutputPort());
    png->Write();
  }
  return vtkRegressionTestImage(renWin);
}
