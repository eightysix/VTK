// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
//
// Visual comparison harness: renders every Metal test scene with both the
// Metal and the OpenGL backends, writes a GL PNG, a Metal PNG and a diff PNG
// per scene, and prints per-scene image-difference metrics. It is meant to be
// inspected by eye (the images land in the --out directory) rather than to
// pass/fail.
//
// Usage:
//   vtkMetalGLVisualComparison [--out <dir>] [--threshold <value>]
//     [--scene <name>] [--backend gl|metal] [--bench] [--frames <n>]
//
// The default output directory is "visual_compare" under the current working
// directory. When --threshold is given, the process exits non-zero if any
// scene's thresholded error exceeds the value.
//
// With --bench, each enabled backend additionally times --frames renders of
// every scene (default 30) after a warmup render and prints per-scene
// average ms/frame, fps and the Metal/GL ratio. The camera is nudged slightly
// each frame so every timed render performs real work, and Metal frames are
// synchronized with WaitForCompletion inside the timed region so the
// wall-clock time includes GPU time.
//
// Note: the OpenGL backend needs the vtkShaderProgram object-factory override
// (vtkOpenGLShaderProgram), so vtkRenderingOpenGL2 and vtkRenderingVolumeOpenGL2
// are auto-initialized here. That autoinit would also make vtkProperty::New()
// and vtkTexture::New() return OpenGL classes, so the scene builders
// (TestMetalScenes.h) give every actor an explicit backend property and the
// Metal texture scene an explicit base-behavior texture.

#include "TestMetalScenes.h"

#include "vtkAutoInit.h"
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkRenderingVolumeOpenGL2);

#include "vtkCocoaMetalRenderWindow.h"
#include "vtkImageDifference.h"
#include "vtkImageExtractComponents.h"
#include "vtkPNGWriter.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkWindowToImageFilter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{

struct SceneSpec
{
  const char* Name;
  void (*Build)(vtkRenderer*, vtkMetalScenes::BackendKind);
  int Width;
  int Height;
};

const SceneSpec kScenes[] = {
  { "AP_OpaqueNoBF", vtkMetalScenes::BuildAP_OpaqueNoBF, 400, 400 },
  { "AP_OpaqueBF", vtkMetalScenes::BuildAP_OpaqueBF, 400, 400 },
  { "AP_TransNoBF", vtkMetalScenes::BuildAP_TransNoBF, 400, 400 },
  { "AP_TransBFbf05", vtkMetalScenes::BuildAP_TransBFbf05, 400, 400 },
  { "AP_TransBFbf10", vtkMetalScenes::BuildAP_TransBFbf10, 400, 400 },
  { "AP_OpFrTransBF", vtkMetalScenes::BuildAP_OpFrTransBF, 400, 400 },
  { "AP_Trans25BF05", vtkMetalScenes::BuildAP_Trans25BF05, 400, 400 },
  { "AP_Trans75BF05", vtkMetalScenes::BuildAP_Trans75BF05, 400, 400 },
  { "AP_FrontCull", vtkMetalScenes::BuildAP_FrontCull, 400, 400 },
  { "AP_BackCull", vtkMetalScenes::BuildAP_BackCull, 400, 400 },
  { "AP_GB05", vtkMetalScenes::BuildAP_GB05, 400, 400 },
  { "AP_GB10", vtkMetalScenes::BuildAP_GB10, 400, 400 },
  { "AP_RG05", vtkMetalScenes::BuildAP_RG05, 400, 400 },
  { "AP_GR25", vtkMetalScenes::BuildAP_GR25, 400, 400 },
  { "AP_GR7510", vtkMetalScenes::BuildAP_GR7510, 400, 400 },
  { "AP_GR7525", vtkMetalScenes::BuildAP_GR7525, 400, 400 },
  { "AP_GR2510", vtkMetalScenes::BuildAP_GR2510, 400, 400 },
  { "RenderWindow", vtkMetalScenes::BuildRenderWindowScene, 400, 400 },
  { "Camera", vtkMetalScenes::BuildCameraScene, 400, 400 },
  { "Light", vtkMetalScenes::BuildLightScene, 400, 400 },
  { "ActorProperty", vtkMetalScenes::BuildActorPropertyScene, 400, 400 },
  { "PointRender", vtkMetalScenes::BuildPointRenderScene, 400, 400 },
  { "DepthPeeling", vtkMetalScenes::BuildDepthPeelingScene, 400, 400 },
  { "CompositePolyDataMapper", vtkMetalScenes::BuildCompositeScene, 600, 300 },
  { "Glyph3DMapper", vtkMetalScenes::BuildGlyphScene, 600, 300 },
  { "HardwareSelector", vtkMetalScenes::BuildHardwareSelectorScene, 600, 300 },
  { "PolyDataMapper2D", vtkMetalScenes::BuildPolyDataMapper2DScene, 600, 300 },
  { "Texture", vtkMetalScenes::BuildTextureScene, 600, 300 },
  { "VolumeRayCast", vtkMetalScenes::BuildVolumeScene, 400, 400 },
};

// Render one scene with one backend and write an RGB PNG. Returns the image.
vtkSmartPointer<vtkImageData> RenderAndCapture(
  const SceneSpec& spec, vtkMetalScenes::BackendKind backend, const std::string& path)
{
  vtkSmartPointer<vtkRenderWindow> renWin = vtkMetalScenes::NewRenderWindow(backend);
  renWin->SetSize(spec.Width, spec.Height);
  renWin->SetMultiSamples(0);
  // Read from the back buffer, exactly like vtkTesting::RegressionTestAndCaptureOutput.
  renWin->SwapBuffersOff();

  vtkSmartPointer<vtkRenderer> renderer = vtkMetalScenes::NewRenderer(backend);
  renWin->AddRenderer(renderer);
  spec.Build(renderer, backend);

  renWin->Render();
  if (backend == vtkMetalScenes::BackendKind::Metal)
  {
    vtkCocoaMetalRenderWindow* metalWin = vtkCocoaMetalRenderWindow::SafeDownCast(renWin);
    if (!metalWin)
    {
      std::cerr << spec.Name << ": Metal window is not a vtkCocoaMetalRenderWindow"
                << std::endl;
      return nullptr;
    }
    metalWin->WaitForCompletion();
  }

  vtkNew<vtkWindowToImageFilter> w2i;
  w2i->SetInput(renWin);
  w2i->SetInputBufferTypeToRGBA();
  w2i->SetReadFrontBuffer(false);
  w2i->SetShouldRerender(false);
  w2i->Update();

  vtkNew<vtkImageExtractComponents> extract;
  extract->SetInputConnection(w2i->GetOutputPort());
  extract->SetComponents(0, 1, 2);
  extract->Update();

  vtkNew<vtkPNGWriter> writer;
  writer->SetFileName(path.c_str());
  writer->SetInputConnection(extract->GetOutputPort());
  writer->Write();

  return extract->GetOutput();
}

// Per-backend wall-clock render timing for one scene.
struct BenchStats
{
  double MinMs = 0.0;
  double AvgMs = 0.0;
  double MaxMs = 0.0;
};

// Build a fresh window/renderer for the scene, render one warmup frame (which
// compiles shaders / builds pipeline states), then time `frames` renders.
// The active camera is rotated slightly each frame so Render() does real work,
// and the Metal backend is synchronized with WaitForCompletion inside the timed
// region so the measurement covers GPU time, not just CPU submission.
BenchStats BenchmarkScene(
  const SceneSpec& spec, vtkMetalScenes::BackendKind backend, int frames)
{
  frames = std::max(1, frames);

  vtkSmartPointer<vtkRenderWindow> renWin = vtkMetalScenes::NewRenderWindow(backend);
  renWin->SetSize(spec.Width, spec.Height);
  renWin->SetMultiSamples(0);
  renWin->SwapBuffersOff();

  // The Metal backend blits every frame into a shared read-back texture (so
  // GetPixelData works) unless disabled; that blit is pure per-frame overhead
  // for a benchmark that never captures.
  if (backend == vtkMetalScenes::BackendKind::Metal)
  {
    vtkCocoaMetalRenderWindow::SafeDownCast(renWin)->SetColorReadbackEnabled(false);
  }

  vtkSmartPointer<vtkRenderer> renderer = vtkMetalScenes::NewRenderer(backend);
  renWin->AddRenderer(renderer);
  spec.Build(renderer, backend);

  renWin->Render();
  if (backend == vtkMetalScenes::BackendKind::Metal)
  {
    vtkCocoaMetalRenderWindow::SafeDownCast(renWin)->WaitForCompletion();
  }

  double minMs = std::numeric_limits<double>::max();
  double maxMs = 0.0;
  double sumMs = 0.0;
  for (int i = 0; i < frames; ++i)
  {
    renderer->GetActiveCamera()->Azimuth(0.1);
    const auto t0 = std::chrono::steady_clock::now();
    renWin->Render();
    if (backend == vtkMetalScenes::BackendKind::Metal)
    {
      vtkCocoaMetalRenderWindow::SafeDownCast(renWin)->WaitForCompletion();
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    minMs = std::min(minMs, ms);
    maxMs = std::max(maxMs, ms);
    sumMs += ms;
  }

  BenchStats stats;
  stats.MinMs = minMs;
  stats.AvgMs = sumMs / frames;
  stats.MaxMs = maxMs;
  return stats;
}

} // namespace

int main(int argc, char* argv[])
{
  std::string outDir = "visual_compare";
  double threshold = -1.0;
  std::string sceneFilter;
  std::string backendFilter;
  bool bench = false;
  int benchFrames = 30;
  for (int i = 1; i < argc; ++i)
  {
    std::string arg = argv[i];
    if (arg == "--out" && i + 1 < argc)
    {
      outDir = argv[++i];
    }
    else if (arg == "--threshold" && i + 1 < argc)
    {
      threshold = std::atof(argv[++i]);
    }
    else if (arg == "--scene" && i + 1 < argc)
    {
      sceneFilter = argv[++i];
    }
    else if (arg == "--backend" && i + 1 < argc)
    {
      backendFilter = argv[++i];
    }
    else if (arg == "--bench")
    {
      bench = true;
    }
    else if (arg == "--frames" && i + 1 < argc)
    {
      benchFrames = std::atoi(argv[++i]);
    }
    else
    {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return EXIT_FAILURE;
    }
  }

  std::error_code ec;
  std::filesystem::create_directories(outDir, ec);
  if (ec)
  {
    std::cerr << "Failed to create output directory " << outDir << ": " << ec.message()
              << std::endl;
    return EXIT_FAILURE;
  }

  // vtkImageDifference requires 3-component images and computes an error that
  // is easy to misinterpret; use the same thresholded error vtkTesting uses.
  double worst = 0.0;
  bool failed = false;

  std::cout << "scene                          error   thresholded error\n";
  std::cout << "-------------------------------------------------------------------\n";
  for (const SceneSpec& spec : kScenes)
  {
    if (!sceneFilter.empty() && sceneFilter != spec.Name)
    {
      continue;
    }

    const std::string glPath = outDir + "/" + spec.Name + ".gl.png";
    const std::string metalPath = outDir + "/" + spec.Name + ".metal.png";
    const std::string diffPath = outDir + "/" + spec.Name + ".diff.png";

    bool renderGl = backendFilter.empty() || backendFilter == "gl";
    bool renderMetal = backendFilter.empty() || backendFilter == "metal";

    vtkSmartPointer<vtkImageData> glImage;
    vtkSmartPointer<vtkImageData> metalImage;
    if (renderGl)
    {
      glImage = RenderAndCapture(spec, vtkMetalScenes::BackendKind::OpenGL, glPath);
    }
    if (renderMetal)
    {
      metalImage = RenderAndCapture(spec, vtkMetalScenes::BackendKind::Metal, metalPath);
    }
    if ((renderGl && !glImage) || (renderMetal && !metalImage))
    {
      failed = true;
      continue;
    }
    if (!renderGl || !renderMetal)
    {
      continue;
    }

    vtkNew<vtkImageDifference> diff;
    diff->SetInputData(glImage);
    diff->SetImageData(metalImage);
    diff->SetThreshold(20);
    diff->Update();

    const double error = diff->GetError();
    const double thresholdedError = diff->GetThresholdedError();
    worst = std::max(worst, thresholdedError);
    if (threshold >= 0.0 && thresholdedError > threshold)
    {
      failed = true;
    }

    char line[256];
    std::snprintf(line, sizeof(line), "%-30s %10.3f %14.3f\n", spec.Name, error, thresholdedError);
    std::cout << line;

    vtkNew<vtkPNGWriter> diffWriter;
    diffWriter->SetFileName(diffPath.c_str());
    diffWriter->SetInputConnection(diff->GetOutputPort());
    diffWriter->Write();
  }

  if (bench)
  {
    std::cout << "\nBenchmark (" << benchFrames << " frames after warmup):\n";
    std::cout << "scene                          "
                 "GL ms/f   GL fps   Metal ms/f  Metal fps    M/GL\n";
    std::cout << "-------------------------------------------------------------------\n";
    for (const SceneSpec& spec : kScenes)
    {
      if (!sceneFilter.empty() && sceneFilter != spec.Name)
      {
        continue;
      }
      const bool benchGl = backendFilter.empty() || backendFilter == "gl";
      const bool benchMetal = backendFilter.empty() || backendFilter == "metal";

      char line[256];
      if (benchGl && benchMetal)
      {
        const BenchStats gl = BenchmarkScene(spec, vtkMetalScenes::BackendKind::OpenGL, benchFrames);
        const BenchStats metal =
          BenchmarkScene(spec, vtkMetalScenes::BackendKind::Metal, benchFrames);
        std::snprintf(line, sizeof(line), "%-30s %8.2f %8.1f  %10.2f %10.1f  %6.2f\n", spec.Name,
          gl.AvgMs, 1000.0 / gl.AvgMs, metal.AvgMs, 1000.0 / metal.AvgMs, metal.AvgMs / gl.AvgMs);
      }
      else if (benchGl)
      {
        const BenchStats gl = BenchmarkScene(spec, vtkMetalScenes::BackendKind::OpenGL, benchFrames);
        std::snprintf(line, sizeof(line), "%-30s %8.2f %8.1f  %10s %10s  %6s\n", spec.Name,
          gl.AvgMs, 1000.0 / gl.AvgMs, "-", "-", "-");
      }
      else if (benchMetal)
      {
        const BenchStats metal =
          BenchmarkScene(spec, vtkMetalScenes::BackendKind::Metal, benchFrames);
        std::snprintf(line, sizeof(line), "%-30s %8s %8s  %10.2f %10.1f  %6s\n", spec.Name, "-",
          "-", metal.AvgMs, 1000.0 / metal.AvgMs, "-");
      }
      else
      {
        continue;
      }
      std::cout << line;
    }
    std::cout << "-------------------------------------------------------------------\n";
  }

  std::cout << "-------------------------------------------------------------------\n";
  std::cout << "worst thresholded error: " << worst << std::endl;
  std::cout << "Images written to " << outDir << "/" << std::endl;

  if (threshold >= 0.0)
  {
    std::cout << "threshold: " << threshold << std::endl;
  }
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
