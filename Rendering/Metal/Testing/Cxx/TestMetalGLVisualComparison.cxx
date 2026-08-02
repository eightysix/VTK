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
//     [--scene <name>] [--backend gl|metal] [--bench] [--frames <n>] [--reps <n>]
//     [--complexity] [--perframe] [--gpu-mem] [--host-mem]
//
// The default output directory is "visual_compare" under the current working
// directory. When --threshold is given, the process exits non-zero if any
// scene's thresholded error exceeds the value.
//
// With --bench, each enabled backend additionally times --frames renders of
// every scene (default 30) after a warmup render and prints per-scene
// average ms/frame, fps and the Metal/GL ratio. The camera is nudged slightly
// each frame so every timed render performs real work, and both backends are
// synchronized inside the timed region (Metal WaitForCompletion, OpenGL
// glFinish) so the wall-clock time covers GPU time for both. The bench is
// grouped by backend — the whole GL suite runs first, then the whole Metal
// suite — rather than interleaved per scene, so both backends are measured in
// the same sustained clock state (per-scene interleaving biased the ratio:
// GL landed in a fresh clock state while Metal inherited the throttled state
// right after GL's reps).
//
// --reps repeats the whole per-scene measurement (fresh window each run) the
// given number of times and reports the mean of the per-run averages plus the
// run-to-run standard deviation, which dilutes the noise (thermal/background
// load on laptops) that shows up in single runs.
//
// --complexity additionally benchmarks the kBenchScenes complexity-scaling
// scenes (GPU-bound geometry, CPU-bound draw-call count, depth-peel count, and
// volume size). These are bench-only and do not participate in the visual
// comparison, so the canonical --bench table is unchanged unless the flag is
// passed.
//
// --perframe prints every timed frame's ms, for diagnosing transient empty
// frames (near-zero ms) in the timed loop.
//
// --gpu-mem prints the Metal device's currentAllocatedSize (MB) after each
// bench run, and --host-mem prints this process's resident set size (MB), so
// a full --bench run doubles as a leak check: both should stay flat across a
// scene's reps and return to a low baseline between scenes.
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
#include "vtk_glad.h"
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
#include <cstdlib>

#include <mach/mach.h>
#include <mach/task_info.h>

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void* pool);
#include <filesystem>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

namespace
{

// Diagnostic: when set, BenchmarkScene prints every timed frame's ms to stdout
// so a transient empty frame (near-zero ms) is visible in the output.
bool gPerFrame = false;

// Diagnostic: when set, print the Metal device's currentAllocatedSize (MB)
// after each backend's bench run, to track GPU memory accumulation across
// scenes. Only meaningful for the Metal backend.
bool gGpuMem = false;

// Diagnostic: when set, print this process's resident set size (MB) after each
// backend's bench run, to track host-RAM accumulation across scenes. Unlike
// gpu-mem it is meaningful for both backends.
bool gHostMem = false;

// Process resident set size in bytes (host RAM footprint), queried via the
// mach task info API. This is the same "RSS" that ps reports for the process.
double GetProcessRSS()
{
  mach_task_basic_info info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
        reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS)
  {
    return static_cast<double>(info.resident_size);
  }
  return 0.0;
}

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
  { "ImageMapper", vtkMetalScenes::BuildImageMapperScene, 600, 300 },
  { "Texture", vtkMetalScenes::BuildTextureScene, 600, 300 },
  { "VolumeRayCast", vtkMetalScenes::BuildVolumeScene, 400, 400 },
  { "CellColor", [](vtkRenderer* r, vtkMetalScenes::BackendKind b) {
      vtkMetalScenes::BuildCellColorGridScene(r, b, 4, 30);
    }, 800, 800 },
};

// Complexity-scaling scenes, registered behind --complexity so the canonical
// --bench output (used to regenerate the doc table) is unchanged. Each scene
// grows one workload axis. The GPU-bound geometry scenes use the same 4x4 /
// 10x10 / 16x16 grid of vtkSphereSource spheres (merged into one polydata) with
// three coloring modes to isolate the fragment paths: CpxGeom* carry no scalars
// (lean opaque pipeline), CpxPoint* carry per-point scalars (color arrives as
// an interpolated varying), and CpxCell* carry per-cell scalars (color resolved
// per-primitive in the fragment shader — the cell-texture port). CpxActor*
// scale the CPU-bound draw-call count, CpxPeel* the depth-peel count,
// CpxVol* the volume size, and CpxImg* the image-mapper fill rate (a large
// RGB image drawn 1:1 by vtkImageMapper). With --complexity they are both
// benchmarked and captured (PNG + thresholded error like the visual scenes).
const SceneSpec kBenchScenes[] = {
  { "CpxGeomLo", [](vtkRenderer* r, vtkMetalScenes::BackendKind b) {
      vtkMetalScenes::BuildGeometryGridScene(r, b, 4, 30);
    }, 800, 800 },
  { "CpxGeomHi", [](vtkRenderer* r, vtkMetalScenes::BackendKind b) {
      vtkMetalScenes::BuildGeometryGridScene(r, b, 10, 60);
    }, 800, 800 },
  { "CpxCellLo", [](vtkRenderer* r, vtkMetalScenes::BackendKind b) {
      vtkMetalScenes::BuildCellColorGridScene(r, b, 4, 30);
    }, 800, 800 },
  { "CpxCellHi", [](vtkRenderer* r, vtkMetalScenes::BackendKind b) {
      vtkMetalScenes::BuildCellColorGridScene(r, b, 10, 60);
    }, 800, 800 },
  { "CpxPointLo", [](vtkRenderer* r, vtkMetalScenes::BackendKind b) {
      vtkMetalScenes::BuildPointColorGridScene(r, b, 4, 30);
    }, 800, 800 },
  { "CpxPointHi", [](vtkRenderer* r, vtkMetalScenes::BackendKind b) {
      vtkMetalScenes::BuildPointColorGridScene(r, b, 10, 60);
    }, 800, 800 },
  { "CpxGeomBig", [](vtkRenderer* r, vtkMetalScenes::BackendKind b) {
      vtkMetalScenes::BuildGeometryGridScene(r, b, 16, 60);
    }, 800, 800 },
  { "CpxPointBig", [](vtkRenderer* r, vtkMetalScenes::BackendKind b) {
      vtkMetalScenes::BuildPointColorGridScene(r, b, 16, 60);
    }, 800, 800 },
  { "CpxCellBig", [](vtkRenderer* r, vtkMetalScenes::BackendKind b) {
      vtkMetalScenes::BuildCellColorGridScene(r, b, 16, 60);
    }, 800, 800 },
  { "CpxActorLo", [](vtkRenderer* r, vtkMetalScenes::BackendKind b) {
      vtkMetalScenes::BuildActorGridScene(r, b, 8);
    }, 800, 800 },
  { "CpxActorHi", [](vtkRenderer* r, vtkMetalScenes::BackendKind b) {
      vtkMetalScenes::BuildActorGridScene(r, b, 32);
    }, 800, 800 },
  { "CpxPeel3", [](vtkRenderer* r, vtkMetalScenes::BackendKind b) {
      vtkMetalScenes::BuildPeelChainScene(r, b, 3);
    }, 400, 400 },
  { "CpxPeel12", [](vtkRenderer* r, vtkMetalScenes::BackendKind b) {
      vtkMetalScenes::BuildPeelChainScene(r, b, 12);
    }, 400, 400 },
  { "CpxVol64", [](vtkRenderer* r, vtkMetalScenes::BackendKind b) {
      vtkMetalScenes::BuildVolumeSceneSized(r, b, 64);
    }, 400, 400 },
  { "CpxVol128", [](vtkRenderer* r, vtkMetalScenes::BackendKind b) {
      vtkMetalScenes::BuildVolumeSceneSized(r, b, 128);
    }, 400, 400 },
  { "CpxImg1024", [](vtkRenderer* r, vtkMetalScenes::BackendKind b) {
      vtkMetalScenes::BuildImageSizeScene(r, b, 1024);
    }, 1024, 1024 },
  { "CpxImg2048", [](vtkRenderer* r, vtkMetalScenes::BackendKind b) {
      vtkMetalScenes::BuildImageSizeScene(r, b, 2048);
    }, 2048, 2048 },
};

// Render one scene with one backend and write an RGB PNG. Returns the image.
// If warmupFrames > 0, that many warmup frames (with a tiny camera nudge) are
// rendered before the captured frame, so stateful backends (e.g. the adaptive
// depth-peel count) converge before the image is captured.
vtkSmartPointer<vtkImageData> RenderAndCapture(
  const SceneSpec& spec, vtkMetalScenes::BackendKind backend, const std::string& path,
  int warmupFrames = 0)
{
  vtkSmartPointer<vtkRenderWindow> renWin = vtkMetalScenes::NewRenderWindow(backend);
  renWin->SetSize(spec.Width, spec.Height);
  renWin->SetMultiSamples(0);
  // Read from the back buffer, exactly like vtkTesting::RegressionTestAndCaptureOutput.
  renWin->SwapBuffersOff();

  vtkSmartPointer<vtkRenderer> renderer = vtkMetalScenes::NewRenderer(backend);
  renWin->AddRenderer(renderer);
  spec.Build(renderer, backend);

  for (int i = 0; i < std::max(0, warmupFrames); ++i)
  {
    renderer->GetActiveCamera()->Azimuth(0.1);
    renWin->Render();
    if (backend == vtkMetalScenes::BackendKind::Metal)
    {
      vtkCocoaMetalRenderWindow::SafeDownCast(renWin)->WaitForCompletion();
    }
    else
    {
      glFinish();
    }
  }

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
// and both backends are synchronized inside the timed region (Metal
// WaitForCompletion, OpenGL glFinish) so the measurement covers GPU time, not
// just CPU submission.
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

  // Render both backends into offscreen buffers for the timed loop. On recent
  // macOS the CAMetalLayer's nextDrawable is paced to the display refresh
  // cadence even for window-less layers, capping every Metal scene at the
  // refresh rate regardless of complexity; the test-only offscreen target
  // (VTK_METAL_ENABLE_OFFSCREEN_TARGET) skips drawable presentation so the
  // timed loop measures real GPU time. The GL backend renders into its standard
  // offscreen framebuffer (vtkOpenGLFramebufferObject), which skips the display
  // path the same way SwapBuffersOff() does. Enabling it for both backends
  // makes the two configurations symmetric.
  renWin->SetOffScreenRendering(true);

  vtkSmartPointer<vtkRenderer> renderer = vtkMetalScenes::NewRenderer(backend);
  renWin->AddRenderer(renderer);
  spec.Build(renderer, backend);

  renWin->Render();
  if (backend == vtkMetalScenes::BackendKind::Metal)
  {
    vtkCocoaMetalRenderWindow::SafeDownCast(renWin)->WaitForCompletion();
  }
  else
  {
    glFinish();
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
    else
    {
      // OpenGL submits asynchronously; block until the GPU finishes so the
      // measurement covers GPU time the same way WaitForCompletion does for
      // Metal. Without this the GL times are pure CPU-submit latency and the
      // comparison is apples-to-oranges.
      glFinish();
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (gPerFrame)
    {
      std::printf("[%s/%s] frame %3d: %.4f ms\n",
        backend == vtkMetalScenes::BackendKind::Metal ? "metal" : "gl", spec.Name, i, ms);
    }
    minMs = std::min(minMs, ms);
    maxMs = std::max(maxMs, ms);
    sumMs += ms;
  }

  BenchStats stats;
  stats.MinMs = minMs;
  stats.AvgMs = sumMs / frames;
  stats.MaxMs = maxMs;
  if (gGpuMem && backend == vtkMetalScenes::BackendKind::Metal)
  {
    vtkCocoaMetalRenderWindow* metalWin = vtkCocoaMetalRenderWindow::SafeDownCast(renWin);
    if (metalWin)
    {
      const double mb = static_cast<double>(metalWin->GetAllocatedSize()) / (1024.0 * 1024.0);
      std::printf("[gpu-mem/%s] %.1f MB\n", spec.Name, mb);
    }
  }
  if (gHostMem)
  {
    const double mb = GetProcessRSS() / (1024.0 * 1024.0);
    std::printf("[host-mem/%s] %.1f MB\n", spec.Name, mb);
  }
  return stats;
}

// Aggregate of a backend's timings across repeated BenchmarkScene runs.
struct BenchAggregate
{
  double MeanMs = 0.0;   // mean of the per-run averages (the reported number)
  double StdDevMs = 0.0; // sample std-dev of the per-run averages (0 if reps < 2)
};

// Run BenchmarkScene `reps` times (fresh window each run) and aggregate the
// per-run averages. Multiple runs dilute run-to-run noise; MeanMs is the
// headline, StdDevMs quantifies how much the per-run averages spread.
BenchAggregate RunBenchmark(
  const SceneSpec& spec, vtkMetalScenes::BackendKind backend, int frames, int reps)
{
  reps = std::max(1, reps);
  std::vector<double> avgs;
  avgs.reserve(reps);
  for (int r = 0; r < reps; ++r)
  {
    avgs.push_back(BenchmarkScene(spec, backend, frames).AvgMs);
  }

  double sum = 0.0;
  for (double v : avgs)
  {
    sum += v;
  }
  const double mean = sum / reps;

  double varSum = 0.0;
  for (double v : avgs)
  {
    varSum += (v - mean) * (v - mean);
  }

  BenchAggregate agg;
  agg.MeanMs = mean;
  agg.StdDevMs = (reps > 1) ? std::sqrt(varSum / (reps - 1)) : 0.0;
  return agg;
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
  int benchReps = 1;
  int warmupFrames = 0;
  bool complexity = false;
  int sizeW = 0;
  int sizeH = 0;
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
    else if (arg == "--reps" && i + 1 < argc)
    {
      benchReps = std::max(1, std::atoi(argv[++i]));
    }
    else if (arg == "--warmup" && i + 1 < argc)
    {
      warmupFrames = std::atoi(argv[++i]);
    }
    else if (arg == "--perframe")
    {
      gPerFrame = true;
    }
    else if (arg == "--gpu-mem")
    {
      gGpuMem = true;
    }
    else if (arg == "--host-mem")
    {
      gHostMem = true;
    }
    else if (arg == "--complexity")
    {
      complexity = true;
    }
    else if (arg == "--size" && i + 1 < argc)
    {
      // Override the window size for the benchmark scenes (e.g. "400x400"),
      // used to decompose per-vertex vs per-fragment cost.
      std::sscanf(argv[++i], "%dx%d", &sizeW, &sizeH);
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
  // Capture GL/Metal images for one scene, write the .gl/.metal/.diff PNGs, and
  // report the thresholded error (also tracking the worst across all scenes).
  const auto captureAndDiff = [&](const SceneSpec& spec) {
    if (!sceneFilter.empty() && sceneFilter != spec.Name)
    {
      return;
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
      glImage = RenderAndCapture(spec, vtkMetalScenes::BackendKind::OpenGL, glPath, warmupFrames);
    }
    if (renderMetal)
    {
      metalImage =
        RenderAndCapture(spec, vtkMetalScenes::BackendKind::Metal, metalPath, warmupFrames);
    }
    if ((renderGl && !glImage) || (renderMetal && !metalImage))
    {
      failed = true;
      return;
    }
    if (!renderGl || !renderMetal)
    {
      return;
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
  };

  for (const SceneSpec& spec : kScenes)
  {
    void* arPool = objc_autoreleasePoolPush();
    captureAndDiff(spec);
    objc_autoreleasePoolPop(arPool);
  }
  // With --complexity the scaling scenes are captured too, so their per-backend
  // PNGs and thresholded errors are available for inspection alongside the bench
  // table.
  if (complexity)
  {
    for (const SceneSpec& spec : kBenchScenes)
    {
      void* arPool = objc_autoreleasePoolPush();
      captureAndDiff(spec);
      objc_autoreleasePoolPop(arPool);
    }
  }

  if (bench)
  {
    if (benchReps > 1)
    {
      std::cout << "\nBenchmark (" << benchFrames << " frames x " << benchReps
                << " runs per scene, mean of run averages ± σ):\n";
    }
    else
    {
      std::cout << "\nBenchmark (" << benchFrames << " frames after warmup):\n";
    }
    if (benchReps > 1)
    {
      std::cout << "scene                          "
                   "GL ms/f   ±σ   GL fps   Metal ms/f   ±σ   Metal fps    M/GL\n";
    }
    else
    {
      std::cout << "scene                          "
                   "GL ms/f   GL fps   Metal ms/f  Metal fps    M/GL\n";
    }
    std::cout << "-------------------------------------------------------------------\n";
    std::vector<SceneSpec> benchScenes(std::begin(kScenes), std::end(kScenes));
    if (complexity)
    {
      benchScenes.insert(benchScenes.end(), std::begin(kBenchScenes), std::end(kBenchScenes));
    }
    if (sizeW > 0 && sizeH > 0)
    {
      for (SceneSpec& spec : benchScenes)
      {
        spec.Width = sizeW;
        spec.Height = sizeH;
      }
    }

    const bool benchGl = backendFilter.empty() || backendFilter == "gl";
    const bool benchMetal = backendFilter.empty() || backendFilter == "metal";

    std::vector<SceneSpec> toBench;
    for (const SceneSpec& spec : benchScenes)
    {
      if (!sceneFilter.empty() && sceneFilter != spec.Name)
      {
        continue;
      }
      toBench.push_back(spec);
    }

    // Group by backend instead of interleaving GL/Metal per scene: each backend
    // now runs its whole suite back-to-back in one sustained pass. The old
    // per-scene interleave measured GL first (fresh clock state) and Metal
    // immediately after GL's reps (inheriting the throttled state), which
    // biased the ratio — individual GL rows caught a clock boost and showed up
    // as "Metal slower" even when the clean single-backend numbers were at or
    // below parity. Grouping gives both backends the same sustained-state
    // comparison in one process.
    std::vector<BenchAggregate> glResults(toBench.size());
    std::vector<BenchAggregate> metalResults(toBench.size());
    for (size_t i = 0; i < toBench.size(); ++i)
    {
      void* arPool = objc_autoreleasePoolPush();
      if (benchGl)
      {
        glResults[i] = RunBenchmark(
          toBench[i], vtkMetalScenes::BackendKind::OpenGL, benchFrames, benchReps);
      }
      objc_autoreleasePoolPop(arPool);
    }
    for (size_t i = 0; i < toBench.size(); ++i)
    {
      void* arPool = objc_autoreleasePoolPush();
      if (benchMetal)
      {
        metalResults[i] =
          RunBenchmark(toBench[i], vtkMetalScenes::BackendKind::Metal, benchFrames, benchReps);
      }
      objc_autoreleasePoolPop(arPool);
    }

    char glCell[32], metalCell[32];
    const auto fmtMs = [benchReps](char* dst, size_t n, const BenchAggregate& a) {
      if (benchReps > 1)
      {
        std::snprintf(dst, n, "%.2f±%.2f", a.MeanMs, a.StdDevMs);
      }
      else
      {
        std::snprintf(dst, n, "%.2f", a.MeanMs);
      }
    };

    for (size_t i = 0; i < toBench.size(); ++i)
    {
      const BenchAggregate& gl = glResults[i];
      const BenchAggregate& metal = metalResults[i];
      char line[256];
      if (benchGl && benchMetal)
      {
        fmtMs(glCell, sizeof(glCell), gl);
        fmtMs(metalCell, sizeof(metalCell), metal);
        std::snprintf(line, sizeof(line), "%-30s %9s %6.1f  %11s %8.1f  %6.2f\n",
          toBench[i].Name, glCell, 1000.0 / gl.MeanMs, metalCell,
          1000.0 / metal.MeanMs, metal.MeanMs / gl.MeanMs);
      }
      else if (benchGl)
      {
        fmtMs(glCell, sizeof(glCell), gl);
        std::snprintf(line, sizeof(line), "%-30s %9s %6.1f  %11s %8s  %6s\n",
          toBench[i].Name, glCell, 1000.0 / gl.MeanMs, "-", "-", "-");
      }
      else if (benchMetal)
      {
        fmtMs(metalCell, sizeof(metalCell), metal);
        std::snprintf(line, sizeof(line), "%-30s %9s %6s  %11s %8.1f  %6s\n",
          toBench[i].Name, "-", "-", metalCell, 1000.0 / metal.MeanMs, "-");
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
