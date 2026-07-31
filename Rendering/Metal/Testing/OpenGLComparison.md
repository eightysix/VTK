# Metal vs OpenGL visual comparison notes

This file records the per-scene error between the Metal and OpenGL backends as
reported by the `vtkMetalGLVisualComparison` harness (see `README.md`). The
harness renders the final state of each `TestMetal*.cxx` regression test with
both backends and compares them with `vtkImageDifference` using the same
thresholded-error metric `vtkTesting` uses for baseline regression.

Numbers below were produced on Apple Silicon, macOS, `SetMultiSamples(0)`,
`SwapBuffersOff()` (back buffer read-back), threshold 20. The single-pass
surface-edge port (`applySurfaceEdges` mirroring `vtkPolyDataEdgesGS.glsl` +
`vtkOpenGLPolyDataMapper::ReplaceShaderEdges`) is active, so PointRender edges
are now rendered on the surface fragment and match GL. The run is fully
reproducible: rerunning the harness produces byte-identical images and
identical errors.

```
scene                          error   thresholded error
-------------------------------------------------------------------
RenderWindow                       39.322          0.000
Camera                             10.078          0.000
Light                              61.965          0.000
ActorProperty                    1949.314        167.807
PointRender                      1668.277          0.000
DepthPeeling                     4425.217        808.961
CompositePolyDataMapper           241.004          0.000
Glyph3DMapper                     444.277          0.000
HardwareSelector                  307.776          0.000
PolyDataMapper2D                  286.047          0.000
Texture                           177.318          0.000
VolumeRayCast                    2612.763        789.835
-------------------------------------------------------------------
worst thresholded error: 808.961
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
  ~0.2%.
- **Camera** (`BuildCameraScene`): cone after the full camera-operation
  sequence (azimuth/elevation/roll/dolly/zoom), ending in parallel projection.
  Identical projection math.
- **Light** (`BuildLightScene`): cone lit by a single spot light (the test's
  final state). Identical lighting model.
- **PointRender** (`BuildPointRenderScene`): sphere with `EdgeVisibilityOn()`,
  `SetLineWidth(7)`, `RenderLinesAsTubesOn()`. The single-pass edge port draws
  the GL-style fake-tube edges directly on the surface fragment (edge
  equations built per-triangle in the vertex entry from the three corner
  positions, evaluated at the fragment's window position), so the surface and
  tube shading match GL to within a few gray levels (only 15 pixels differ,
  max delta 3/255). The previous chord-depth flat-tube overlay is retired.
- **CompositePolyDataMapper** (`BuildCompositeScene`): cone + sphere + cube as
  a `vtkPartitionedDataSetCollection` rendered through the composite poly data
  mapper. The Metal delegator path matches GL.
- **HardwareSelector** (`BuildHardwareSelectorScene`): cone + sphere side by
  side. Basic opaque geometry path, matches GL.
- **Glyph3DMapper** (`BuildGlyphScene`): a 4x4 plane of instanced spheres with
  per-instance scalar colors (`vtkGlyph3DMapper`, `ColorModeToMapScalars`).
  Now matches GL exactly (`>10%` fraction 0.0%): the per-instance normal
  transform is padded to Metal's `float3x3` 16-byte-column layout, and the
  instance material/light buffers follow the `vtkMetalPolyDataMapper`
  conventions.
- **PolyDataMapper2D** (`BuildPolyDataMapper2DScene`): a 3D cone plus an
  orange 2D overlay quad in display coordinates. `vtkMetalRenderer` now drives
  `vtkRenderer::RenderOverlay()`, so the quad renders in Metal and matches GL
  (`>10%` fraction ~0.1%).
- **Texture** (`BuildTextureScene`): a checkerboard-textured plane
  (`vtkTexture`, interpolation on, repeat off, edge clamp). The Metal sampling
  path matches the OpenGL texture unit behavior (`>10%` fraction 0.0%).

### Divergent scenes and likely causes

- **ActorProperty** — error 1949 / thresholded 167.8, ~3.2% of pixels >10%.
  A translucent (opacity 0.5) sphere with a backface property. Backface
  materials are now implemented and match GL; the residual difference is a
  uniform slight darkening of the Metal sphere — every differing pixel is
  GL-brighter by 26-60 gray levels across the 210x210 sphere region — a subtle
  front/back-face translucency compositing or lighting rounding difference.
  Neither backend depth-peels this scene (the flag is off); both use
  painter-order blending.
- **DepthPeeling** — error 4425 / thresholded 808.9, ~17.2% of pixels >10%.
  Three overlapping translucent spheres with `SetUseDepthPeeling(true)` and 20
  peels. `vtkMetalRenderer` drives its depth peeler
  (`vtkMetalDepthPeeler`), which now tracks GL closely; residual differences
  concentrate in the sphere-overlap regions and are consistent with peel-count
  / front-to-back compositing rounding.
- **VolumeRayCast** — error 2612 / thresholded 789.8, ~28.5% of pixels >10%.
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
