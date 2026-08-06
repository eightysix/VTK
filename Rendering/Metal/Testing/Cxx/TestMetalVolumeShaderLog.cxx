// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

// Description
// Exercise the Metal volume ray-cast mapper's shader logging (os_log) support
// with a camera inside a small-spacing volume, mirroring the scenario of
// TestGPURayCastCameraInsideSmallSpacing. There is no image-baseline
// regression here: the test only drives the pipeline through both the proxy
// (camera outside) and the fullscreen (camera inside) paths so the volume
// shaders' "VTK_METAL_VOLUME_LOG" os_log messages are emitted, and verifies
// each frame completes.
//
// ---------------------------------------------------------------------------
// Metal shader logging (os_log) reference
// ---------------------------------------------------------------------------
//
// Mechanism
//   Shader logging (Metal 3.2 / macOS 15+, iOS 18+) compiles os_log calls in
//   MSL source and routes the emitted messages to the logging system. Enabling
//   it requires two things, both active in test builds only
//   (macos_metal_build.sh --tests, i.e. VTK_BUILD_TESTING=ON):
//     - the preprocessor macro VTK_METAL_ENABLE_LOGGING, added to
//       vtkRenderingMetal's test-only compile definitions in
//       Rendering/Metal/CMakeLists.txt, and
//     - MTLCompileOptions.enableLogging = YES with languageVersion =
//       MTLLanguageVersion3_2, produced by vtkMetalVolumeCompileOptions() in
//       vtkMetalGPUVolumeRayCastMapper.mm and passed to
//       newLibraryWithSource:options: in EnsureShaderLibrary().
//   The same MTLCompileOptions also defines VTK_METAL_ENABLE_LOGGING for the
//   shader source, so MetalShaders.metal gates its os_log call sites with
//   #if defined(VTK_METAL_ENABLE_LOGGING). In non-test builds the helper
//   returns nil and the call sites compile out, so the shaders behave exactly
//   as before.
//
// Current log call sites (all in Rendering/Metal/Shaders/MetalShaders.metal):
//     vertex_volume_main               every proxy-cube vertex (once per frame)
//     fragment_volume_main             center pixel, camera-outside (proxy) path
//     fragment_volume_fullscreen_main  center pixel, camera-inside (fullscreen) path
//   The fragment call sites are gated to the window center pixel
//   (|fragPos - 0.5 * viewportSize| < 1) and the proxy cube has only a handful
//   of vertices, so the volume logs a bounded number of messages per frame.
//
// Adding a new log call site
//   Wrap the call in #if defined(VTK_METAL_ENABLE_LOGGING) so production
//   shader libraries are unaffected, e.g.:
//     #if defined(VTK_METAL_ENABLE_LOGGING)
//     os_log_default.log_info("VTK_METAL_VOLUME_LOG <shader> myValue=%f", x);
//     #endif
//   Prefer os_log_default, or declare a custom logger to filter by subsystem /
//   category:
//     constant metal::os_log logger("org.vtk.metal", "volume");
//     logger.log_info("...");
//   Subsystem, category, and format string must each be shorter than 1024
//   characters.
//
// MSL format-specifier rules
//     - the format string must be a string literal (no runtime-built strings)
//     - %s and %n are not supported
//     - vectors print with %v<num><len><conv>, e.g. %v4hlf for a float4
//     - the hl length modifier is required for 4-byte element types in vectors
//     - scalar floats / ints use the usual %f / %d / %u specifiers
//
// Runtime configuration (set at process launch)
//     MTL_LOG_LEVEL=MTLLogLevelDebug  minimum level to buffer; without this the
//                                     messages are silently dropped
//     MTL_LOG_BUFFER_SIZE=<bytes>     capacity of the per-command-buffer log
//                                     buffer (default 1024, max 1 GB); the
//                                     buffer is drained only after the command
//                                     buffer finishes, so overflow discards
//                                     messages and ordering is not preserved
//     MTL_LOG_TO_STDERR=1             forward buffered messages to stderr
//   To run this test with the messages visible (verified form; the 8192-byte
//   buffer is not required for this particular test -- its ~1 KB of messages
//   also fits the 1024-byte default -- but a larger buffer is safer with
//   heavier call sites, which overflow and silently drop messages):
//     MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=8192 MTL_LOG_TO_STDERR=1 \
//       ctest -R TestMetalVolumeShaderLog
//   Or run the test binary directly (useful with rg to count the messages):
//     MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=8192 MTL_LOG_TO_STDERR=1 \
//       <build>/bin/vtkRenderingMetalCxxTests TestMetalVolumeShaderLog \
//       2>&1 | rg "VTK_METAL_VOLUME_LOG"
//
// Viewing the messages
//     - stderr (above) is the most reliable way with ctest
//     - /usr/bin/log stream / log show and Console.app need MTL_LOG_LEVEL set
//       and Console's Action > Include Info Messages / Include Debug Messages
//       enabled; delivery depends on the buffer being drained before the
//       process exits, so a long-running process is more reliable
//
// Caveats
//     - os_log calls execute per thread / fragment, so unguarded call sites
//       can flood the log buffer and drop messages (and hurt performance).
//       Keep them gated (center pixel, single thread, counter, ...).
//     - logging adds compile and runtime overhead and is therefore enabled
//       only in test builds, never in production builds.
// ---------------------------------------------------------------------------

#include "TestMetalHelpers.h"

#include "vtkColorTransferFunction.h"
#include "vtkImageChangeInformation.h"
#include "vtkImageData.h"
#include "vtkMetalCamera.h"
#include "vtkMetalGPUVolumeRayCastMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkNew.h"
#include "vtkPiecewiseFunction.h"
#include "vtkRTAnalyticSource.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"

#include <iostream>

int TestMetalVolumeShaderLog(int argc, char* argv[])
{
  (void)argc;
  (void)argv;
  std::cout << "CTEST_FULL_OUTPUT (Avoid ctest truncation of output)" << std::endl;

  vtkNew<vtkCocoaMetalRenderWindow> renWin;
  vtkNew<vtkMetalRenderer> renderer;
  renWin->AddRenderer(renderer);
  renWin->SetSize(301, 300);
  renWin->SetMultiSamples(0);
  renderer->SetBackground(0.1, 0.1, 0.2);

  renWin->Initialize();
  if (!vtkMetalTesting::CheckBackend(renWin))
  {
    return EXIT_FAILURE;
  }

  // Volume with a very small spacing, like the OpenGL camera-inside test: the
  // bounds collapse into a tiny box so the near-plane / camera-inside paths
  // are exercised.
  vtkNew<vtkRTAnalyticSource> source;
  source->SetWholeExtent(0, 32, 0, 32, 0, 32);
  source->SetCenter(16, 16, 16);

  int dims[3];
  source->Update();
  source->GetOutput()->GetDimensions(dims);

  double desiredBounds = 0.0005;
  double desiredSpacing[3];
  for (int i = 0; i < 3; ++i)
  {
    desiredSpacing[i] = desiredBounds / static_cast<double>(dims[i]);
  }

  vtkNew<vtkImageChangeInformation> imageChangeInfo;
  imageChangeInfo->SetInputConnection(source->GetOutputPort());
  imageChangeInfo->SetOutputSpacing(desiredSpacing);

  vtkNew<vtkColorTransferFunction> color;
  color->AddRGBPoint(0.0, 0.0, 0.0, 0.0);
  color->AddRGBPoint(0.25, 1.0, 0.0, 0.0);
  color->AddRGBPoint(0.75, 0.0, 1.0, 0.0);
  color->AddRGBPoint(1.0, 0.0, 0.0, 1.0);

  vtkNew<vtkPiecewiseFunction> opacity;
  opacity->AddPoint(0.0, 0.0);
  opacity->AddPoint(0.5, 0.15);
  opacity->AddPoint(1.0, 0.9);

  vtkNew<vtkVolumeProperty> property;
  property->SetColor(color);
  property->SetScalarOpacity(opacity);
  property->SetInterpolationTypeToLinear();
  property->ShadeOff();
  property->SetScalarOpacityUnitDistance(7e-6);

  vtkNew<vtkMetalGPUVolumeRayCastMapper> mapper;
  mapper->SetInputConnection(imageChangeInfo->GetOutputPort());
  mapper->SetAutoAdjustSampleDistances(0);
  mapper->SetSampleDistance(7e-6);

  vtkNew<vtkVolume> volume;
  volume->SetMapper(mapper);
  volume->SetProperty(property);

  renderer->AddVolume(volume);
  vtkNew<vtkMetalCamera> camera;
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();

  // Frame 1: camera outside the volume -> proxy path (vertex_volume_main +
  // fragment_volume_main log messages).
  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    std::cerr << "Proxy-path volume render failed." << std::endl;
    return EXIT_FAILURE;
  }

  // Dolly the camera toward the volume center until it is well inside the tiny
  // volume bounds, then render again -> fullscreen path
  // (fragment_volume_fullscreen_main log messages).
  for (int i = 0; i < 40; ++i)
  {
    camera->Dolly(1.25);
  }
  renderer->ResetCameraClippingRange();

  if (!vtkMetalTesting::RenderAndWait(renWin))
  {
    std::cerr << "Fullscreen (camera inside) volume render failed." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Shader logging test complete. The volume shaders emitted "
            << "\"VTK_METAL_VOLUME_LOG\" os_log messages for the proxy and "
            << "fullscreen paths; see them with:" << std::endl
            << "  MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=8192 "
            << "MTL_LOG_TO_STDERR=1 ctest -R TestMetalVolumeShaderLog" << std::endl;

  return EXIT_SUCCESS;
}
