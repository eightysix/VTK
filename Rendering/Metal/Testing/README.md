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
     TestMetalRenderWindow.cxx,NO_DATA,NO_VALID,NO_SERDES
     Test<Name>.cxx,NO_DATA,NO_VALID,NO_SERDES)

   vtk_test_cxx_executable(vtkRenderingMetalCxxTests tests)
   ```

3. Reconfigure and rebuild: `./macos_metal_build.sh --tests --resume`
   (or `ninja -C build_macos_metal vtkRenderingMetalCxxTests`).

## Current limitations

- Image-baseline regression tests (`vtkRegressionTestImage`) are **not**
  supported yet: `vtkMetalRenderWindow` does not implement
  `GetPixelData()`/`ReadPixels()`, so `vtkWindowToImageFilter` cannot read the
  framebuffer. Tests therefore use `NO_VALID` and verify rendering indirectly
  (e.g. `GetIdsData()` reads the GPU-picking IDs texture that is written every
  render pass).
- Tests render to the window's `CAMetalLayer` drawable and synchronize with
  `WaitForCompletion()` before reading data back. Do not drop the
  `WaitForCompletion()` call before a read-back, or the read may race the GPU.
