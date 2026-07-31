# Metal vs OpenGL visual comparison notes

This file records the per-scene error between the Metal and OpenGL backends as
reported by the `vtkMetalGLVisualComparison` harness (see `README.md`). The
harness renders the final state of each `TestMetal*.cxx` regression test with
both backends and compares them with `vtkImageDifference` using the same
thresholded-error metric `vtkTesting` uses for baseline regression.

Numbers below were produced on Apple Silicon, macOS, `SetMultiSamples(0)`,
`SwapBuffersOff()` (back buffer read-back), threshold 20.

```
scene                          error   thresholded error
-------------------------------------------------------------------
RenderWindow                       39.322          0.000
Camera                             10.078          0.000
Light                              61.965          0.000
ActorProperty                   10456.472       7677.969
PointRender                      6901.410       3029.404
DepthPeeling                    49491.677      37757.492
CompositePolyDataMapper           241.004          0.000
Glyph3DMapper                   18812.837      15880.383
HardwareSelector                  307.776          0.000
PolyDataMapper2D                14604.102      12822.630
Texture                         11356.438       9219.059
VolumeRayCast                    2612.763        789.835
-------------------------------------------------------------------
worst thresholded error: 37757.5
```

## How to read this

- **`error`** is the raw summed image difference. **`thresholded error`**
  ignores pixels that differ by less than the threshold (20/255), so a value of
  `0.000` means *every* pixel is within 20 gray levels of the GL image —
  visually identical.
- `>= 10%-pixel fraction` below is the share of pixels whose channel
  difference exceeds 10% of full scale (from a per-panel pixel analysis of the
  same output).

## Per-scene results

### Visually identical (thresholded error 0.000)

- **RenderWindow** (`BuildRenderWindowScene`): single cone, default material.
  Difference is only anti-aliasing/rounding at edges; `>10%` pixels fraction
  <= 0.34%.
- **Camera** (`BuildCameraScene`): cone after the full camera-operation
  sequence (azimuth/elevation/roll/dolly/zoom), ending in parallel projection.
  Identical projection math.
- **Light** (`BuildLightScene`): cone lit by a single spot light (the test's
  final state). Identical lighting model.
- **CompositePolyDataMapper** (`BuildCompositeScene`): cone + sphere + cube as
  a `vtkPartitionedDataSetCollection` rendered through the composite poly data
  mapper. The Metal delegator path matches GL.
- **HardwareSelector** (`BuildHardwareSelectorScene`): cone + sphere side by
  side. Basic opaque geometry path, matches GL.

### Divergent scenes and likely causes

- **ActorProperty** — error 10456 / thresholded 7677, ~19.7% of pixels >10%.
  A translucent (opacity 0.5) sphere with a backface property. Metal does not
  perform depth peeling (see below); the translucent surface is blended with
  painter-order-only, so front/back-face ordering and overdraw differ from
  OpenGL's depth-peeled result.
- **PointRender** — error 6901 / thresholded 3029, ~21.6% of pixels >10%.
  Sphere with `EdgeVisibilityOn()`, `SetLineWidth(7)`, `RenderLinesAsTubesOn()`.
  Likely cause: the Metal mapper does not implement line/tube primitives with
  the same width/render style as OpenGL, so the edge strokes differ.
- **DepthPeeling** — error 49491 / thresholded 37757, ~86.2% of pixels >10%.
  Three overlapping translucent spheres with `SetUseDepthPeeling(true)` and 20
  peels. `vtkMetalRenderer` does not implement depth peeling; the flag is
  ignored and the spheres are blended in a single translucent pass, so the
  compositing is fundamentally different. This is the largest divergence and
  the main missing Metal feature this scene exercises.
- **Glyph3DMapper** — error 18812 / thresholded 15880, ~19.3% of pixels >10%.
  A 4x4 plane of instanced spheres with per-instance scalar colors
  (`vtkGlyph3DMapper`, `ColorModeToMapScalars`). The Metal glyph mapper
  instancing/color path diverges from GL — per-instance colors and lighting are
  not matched exactly.
- **PolyDataMapper2D** — error 14604 / thresholded 12822, ~10.7% of pixels >10%.
  A 3D cone plus an orange 2D overlay quad in display coordinates. The Metal
  backend does not drive `vtkRenderer::RenderOverlay`, so the quad is entirely
  absent from the Metal image. The differing pixels are exactly the quad area.
- **Texture** — error 11356 / thresholded 9219, ~16.1% of pixels >10%.
  A checkerboard-textured plane (`vtkTexture`, interpolation on, repeat off,
  edge clamp). The Metal sampling/filtering path (or how the base `vtkTexture`
  uploads) does not match the OpenGL texture unit behavior at edges/filtering.
- **VolumeRayCast** — error 2612 / thresholded 789, ~9.9% of pixels >10%.
  Analytic 32^3 volume through the GPU ray-cast mapper with a shaded transfer
  function (`ShadeOn`, ambient/diffuse/specular). The Metal volume mapper is a
  newer port; ray sampling/step size and shading terms differ slightly from the
  OpenGL shader, giving a moderate, smooth divergence (worst near the volume
  edges where classification changes fastest).

## Running the analysis yourself

```sh
./build_macos_metal/bin/vtkMetalGLVisualComparison --out /tmp/visual_compare
# then inspect /tmp/visual_compare/<Scene>.gl.png / .metal.png / .diff.png
```

`--scene <name>` isolates one scene, `--threshold <value>` turns the tool into
a pass/fail check on the thresholded error, `--backend gl|metal` renders a
single backend (no diff). Regenerating this table after a change to the Metal
backend is a one-command check that the fix moved the corresponding scene's
thresholded error toward zero.
