# VTK Metal Testing

Unit tests for the Metal rendering backend. Tests run as native macOS console
executables on the host machine (no iOS simulator or device required).

## Requirements

- A Mac with Metal support (Apple Silicon or discrete/Intel GPU with Metal).
- A GUI login session. The render window connects to the WindowServer
  (`MTLCreateSystemDefaultDevice`, `CAMetalLayer`, `nextDrawable`), so tests
  will not work over a headless SSH session.
- The `macos_metal_build.sh` build (see below).

## Building with tests

Enable the test suites when configuring the macOS Metal build:

```sh
./macos_metal_build.sh --tests
```

This sets both `VTK_BUILD_TESTING=ON` and `BUILD_TESTING=ON` (both are
required: `VTK_BUILD_TESTING` configures the module test targets while
`BUILD_TESTING` controls `enable_testing()`/ctest registration).

To reconfigure an existing build folder:

```sh
./macos_metal_build.sh --tests --resume
```

Build just the Metal test executable (instead of the whole tree):

```sh
ninja -C build_macos_metal vtkRenderingMetalCxxTests
```

## Running the tests

```sh
ctest --test-dir build_macos_metal -R RenderingMetalCxx --output-on-failure
```

or run a single test binary directly:

```sh
./build_macos_metal/bin/vtkRenderingMetalCxxTests TestMetalRenderWindow
```

## Visual comparison against the OpenGL backend

`vtkMetalGLVisualComparison` (a standalone executable, built with the test
suite) renders every Metal test scene a second time with the OpenGL backend and
writes side-by-side images plus a numeric difference metric. It is a manual
inspection tool, not a pass/fail test.

```sh
# All scenes (writes <Scene>.gl.png, <Scene>.metal.png, <Scene>.diff.png)
./build_macos_metal/bin/vtkMetalGLVisualComparison --out /tmp/visual_compare

# One scene, one backend, or a pass/fail threshold
./build_macos_metal/bin/vtkMetalGLVisualComparison --scene Texture
./build_macos_metal/bin/vtkMetalGLVisualComparison --backend gl
./build_macos_metal/bin/vtkMetalGLVisualComparison --threshold 10000
```

- `--out <dir>`: output directory (default `visual_compare` in the CWD).
- `--scene <name>`: restrict to one scene (e.g. `RenderWindow`,
  `DepthPeeling`, `VolumeRayCast`, ...).
- `--backend gl|metal`: render with only one backend (no diff).
- `--threshold <value>`: exit non-zero if any scene's thresholded error
  (`vtkImageDifference`, threshold 20, same metric `vtkTesting` uses) exceeds
  the value.

The scenes are the final rendered states of the 12 `TestMetal*.cxx` regression
tests, reproduced backend-agnostically in `TestMetalScenes.h`. As of writing,
`RenderWindow`, `Camera`, `Light`, `CompositePolyDataMapper` and
`HardwareSelector` match OpenGL to within a few hundredths of a percent of
pixels; the other scenes diverge where the Metal backend still renders
translucency/depth peeling, glyph instancing colors, 2D overlays, textures and
volume ray casting differently.

Notes:
- The harness auto-initializes `vtkRenderingOpenGL2` (the GL backend needs the
  `vtkShaderProgram` object-factory override), so it lives in its **own**
  executable. Linking OpenGL into the shared `vtkRenderingMetalCxxTests`
  executable would auto-init it for the regression tests too and make their
  `vtkProperty::New()`/`vtkTexture::New()` calls return OpenGL classes.
- For the same reason the scene builders set an explicit backend property on
  every actor (`vtkMetalProperty`/`vtkOpenGLProperty`) and give the Metal
  texture scene an explicit base-behavior texture.
- The harness reads the back buffer (`SwapBuffersOff`) and, for Metal, calls
  `vtkCocoaMetalRenderWindow::WaitForCompletion()` before capture, matching
  `vtkTesting::RegressionTestAndCaptureOutput`.

## Adding a new test

1. Add `Test<Name>.cxx` in `Rendering/Metal/Testing/Cxx/`.
   - Test sources must use the `.cxx` extension (required by
     `vtk_add_test_cxx`), so keep the test plain C++ and use the explicit
     Metal classes (`vtkCocoaMetalRenderWindow`, `vtkMetalRenderer`,
     `vtkMetalPolyDataMapper`, ...) rather than relying on object-factory
     resolution.
   - The file must define `int Test<Name>(int argc, char* argv[])`.
2. Register it in `Rendering/Metal/Testing/Cxx/CMakeLists.txt`:

   ```cmake
   vtk_add_test_cxx(vtkRenderingMetalCxxTests tests
     TestMetalRenderWindow.cxx,NO_DATA,NO_SERDES
     Test<Name>.cxx,NO_DATA,NO_SERDES)

   vtk_test_cxx_executable(vtkRenderingMetalCxxTests tests)
   ```

   Do **not** use `NO_VALID` for image-baseline regression (it disables the
   `-V` baseline argument). Keep `NO_SERDES`.
3. End the test with `vtkRegressionTestImage(renWin)` and map its return value
   with `vtkMetalTesting::RegressionExitCode()` (from `TestMetalHelpers.h`).
4. Reconfigure and rebuild: `./macos_metal_build.sh --tests --resume`
   (or `ninja -C build_macos_metal vtkRenderingMetalCxxTests`).
5. Generate the baseline:
   - Run `ctest --test-dir build_macos_metal -R RenderingMetalCxx --output-on-failure`.
     A missing baseline makes the test fail with an `ImageNotFound` measurement
     and writes the rendered image to
     `build_macos_metal/Testing/Temporary/<TestName>.png`.
   - Copy that PNG to `Rendering/Metal/Testing/Data/Baseline/<TestName>.png`
     and rebuild the data targets so ExternalData picks it up:
     `ninja -C build_macos_metal VTKData`. ExternalData converts the plain PNG
     into a tracked `.sha512` marker plus a content object in the repo
     `.ExternalData/SHA512/` store (this also happens automatically during a
     full `./macos_metal_build.sh` build). Re-run the test; it should pass.

## Baseline management

- Baselines follow VTK's ExternalData convention:
  - `Rendering/Metal/Testing/Data/Baseline/<TestName>.png.sha512` holds the
    content hash (tracked in git).
  - `.ExternalData/SHA512/<hash>` at the repo root holds the actual PNG content
    (tracked in git, laid out exactly as the VTK pre-commit hook does).
  - Raw `<TestName>.png` files and `.ExternalData_SHA512_*` content links are
    gitignored by the top-level `.gitignore`.
- `VTKData` is part of the default `all` target, so a full
  `./macos_metal_build.sh` build resolves baselines automatically. For a fresh
  clone, ExternalData finds the content in the committed `.ExternalData/`
  store, so no download URLs are needed.
- A test failing with `ImageNotFound` is the normal flow when a baseline is
  missing; it regenerates the image in `build_macos_metal/Testing/Temporary/`.
- If a renderer change intentionally alters the output, update the baseline
  deliberately: run the test, review the generated PNG, copy it into
  `Rendering/Metal/Testing/Data/Baseline/`, run `ninja -C build_macos_metal
  VTKData`, and commit the updated `.sha512` + store objects.

## Current limitations

- Image-baseline regression is supported: `vtkMetalRenderWindow` implements
  `GetPixelData()`/`GetRGBACharPixelData()` by blitting the final drawable to a
  CPU-readable color texture each frame (see `ColorCopyTexture` in
  `vtkMetalRenderWindow`). Tests compare against PNG baselines via
  `vtkRegressionTestImage` exactly like the OpenGL backend.
- The color read-back (blit, shared texture, `framebufferOnly=NO`) is compiled
  in **only when `VTK_BUILD_TESTING=ON`** (`VTK_METAL_ENABLE_COLOR_READBACK` in
  `Rendering/Metal/CMakeLists.txt`). Production builds of the library keep
  `framebufferOnly=YES` and do not pay for the per-frame copy.
- Tests render to the window's `CAMetalLayer` drawable and synchronize with
  `WaitForCompletion()` before reading data back. Do not drop the
  `WaitForCompletion()` call before a read-back, or the read may race the GPU.
