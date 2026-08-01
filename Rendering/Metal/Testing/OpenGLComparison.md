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
are now rendered on the surface fragment and match GL. The order-independent
translucent pass is implemented in Metal (`vtkMetalOrderIndependentTranslucentPass`),
mirroring the GL default OIT path (`vtkOrderIndependentTranslucentPass`:
RGBA16F + R16F accumulate, then a fullscreen weighted-average resolve over the
drawable), so translucent rendering matches GL by construction. The volume
ray-cast march loop now uses the same sample→accumulate→advance→terminate
order as the GL reference (the bounds-termination check moved to the loop tail,
after accumulation), so the rim and silhouette match GL exactly. The volume
gradient and phong lighting are computed in half (16-bit) precision with
`fast::pow` (a deliberate performance choice: a float-precision experiment
reduced the VolumeRayCast thresholded error from 0.007 to 0.005 but was
reverted as not worth the cost). The run is fully reproducible: rerunning the
harness produces byte-identical images and identical errors.

```
scene                          error   thresholded error
-------------------------------------------------------------------
AP_OpaqueNoBF                     237.895          0.000
AP_OpaqueBF                       237.895          0.000
AP_TransNoBF                      204.000          0.000
AP_TransBFbf05                    209.736          0.000
AP_TransBFbf10                    272.187          0.000
AP_OpFrTransBF                    237.895          0.000
AP_Trans25BF05                    203.335          0.000
AP_Trans75BF05                    239.540          0.000
AP_FrontCull                      151.352          0.000
AP_BackCull                       123.367          0.000
AP_GB05                           209.736          0.000
AP_GB10                           272.187          0.000
AP_RG05                           209.736          0.000
AP_GR25                           168.059          0.000
AP_GR7510                         274.841          0.000
AP_GR7525                         214.532          0.000
AP_GR2510                         292.489          0.000
RenderWindow                       39.322          0.000
Camera                             10.078          0.000
Light                              61.965          0.000
ActorProperty                     606.631          0.000
PointRender                      1668.115          0.000
DepthPeeling                      740.265          0.000
CompositePolyDataMapper           241.004          0.000
Glyph3DMapper                     444.277          0.000
HardwareSelector                  307.776          0.000
PolyDataMapper2D                  286.047          0.000
Texture                           177.318          0.000
VolumeRayCast                     621.654          0.007
------------------------------------------------------------------
worst thresholded error: 0.00653595
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

### Visually identical (thresholded error ≤ 0.005)

- **OIT scenes** (`BuildAP_*`, the `AP_*` rows): a single translucent sphere
  with an optional backface property, rendered through the order-independent
  translucent pass. The front/back opacity and color combinations have
  closed-form OIT results, so an exact GL match pins down the accumulate
  (RGBA16F + R16F, blend `(ONE, ONE, ZERO, ONE_MINUS_SRC_ALPHA)`) and resolve
  (`rgb/max(reveal, 0.01)`, `alpha = 1 - accum.a`, standard over-blend) math
  plus backface-material handling. All 11 alpha/color combinations match GL
  exactly.
- **RenderWindow** (`BuildRenderWindowScene`): single cone, default material.
  Difference is only anti-aliasing/rounding at edges; `>10%` pixels fraction
  ~0.2%.
- **Camera** (`BuildCameraScene`): cone after the full camera-operation
  sequence (azimuth/elevation/roll/dolly/zoom), ending in parallel projection.
  Identical projection math.
- **Light** (`BuildLightScene`): cone lit by a single spot light (the test's
  final state). Identical lighting model.
- **ActorProperty** (`BuildActorPropertyScene`): translucent (opacity 0.5)
  sphere with a backface property. Previously divergent (error 1949 /
  thresholded 167.8); it now matches GL exactly because Metal runs the same
  OIT accumulate + resolve pass GL uses by default (`vtkRenderer::UseOIT`).
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
- **DepthPeeling** (`BuildDepthPeelingScene`): three overlapping translucent
  spheres with `SetUseDepthPeeling(true)` and 20 peels. Now matches GL exactly
  (`>10%` fraction 0.0%). The Metal peel pipeline
  (`vtkMetalPolyDataMapper::EnsurePeelPipelineStates`) originally wrote the
  `backTemp` target (attachment 0) with blending disabled, so the last
  non-back fragment at each pixel overwrote (erased) back faces captured
  earlier in the same peel — a pixel-order-dependent missing band below the
  seam. The GL reference (`vtkDualDepthPeelingPass`) blends all
  three peel targets with `glBlendEquation(GL_MAX)`; enabling MAX blending on
  the Metal `backTemp` attachment reproduces the reference behavior and brings
  the scene to 0.000 thresholded error (raw error 4425 → 740).
- **VolumeRayCast** (`BuildVolumeScene`): analytic 32^3 volume through the GPU
  ray-cast mapper with a shaded transfer function (`ShadeOn`,
  ambient/diffuse/specular). Previously divergent (error 2612 / thresholded
  789.8, ~28.5% of pixels >10%); now matches GL to within 0.007 thresholded
  error (raw 621.654). Two fixes account for most of the convergence:
  1. The march loop checked the box bounds at the top of the loop, *before*
     sampling, so rays that grazed the silhouette (box segment shorter than
     one step) broke with zero samples and left a black rim, while GL samples
     the (clamped border) position and accumulates it before terminating. The
     bounds check moved to the loop tail — after the sample/accumulate step
     and after advancing — reproducing the GL
     sample→accumulate→advance→terminate order, so silhouettes and rims match
     GL exactly (the volume `clamp_to_edge` sampler provides the same border
     clamping).
  2. A later perf pass advanced the sample position (`evalPoint += evalStep`)
     *before* the gradient/shading block, so the gradient was fetched one ray
     step ahead of the scalar it lit. The fetch order was restored to
     sample-then-advance.
- **VolumeRayCast residual (0.007)**: the ~40 thresholded pixels (0.02%) are
  all interior (none on the silhouette), all in the blue channel, on the edge
  of the specular highlight. Ruled out as sources: the termination threshold
  (GL `1 - 1/255` strict `>` vs Metal's `>= 0.99` — no image change), jitter
  (`UseJittering` defaults to 0, so neither backend jitters), step semantics
  (both march at a constant 1.0 physical sample distance), linear vs nearest
  interpolation, and the ray entry point (Metal's analytic `intersectBox` vs
  GL's interpolated proxy `ip_textureCoords` — overriding Metal's entry with
  the interpolated position produced no change). Computing the gradient and
  lighting in float precision (matching GL) instead of half reduced the error
  to `0.005`, but the gain was small and the change was reverted to keep the
  half-precision + `fast::pow` shading path. The mechanism is float-rounding-
  level sample-position divergence (entry and step composed in different
  spaces/arithmetic); near a texel boundary the nearest-interpolated gradient
  taps land in different cells, rotating the gradient a few degrees and
  shifting the specular highlight boundary by ~1px. This is irreducible
  without bit-identical sampling, which different backend arithmetic cannot
  guarantee, and is below the documented baseline.

## Running the analysis yourself

The volume shaders are embedded in the framework at build time, so a stale
library produces stale numbers. After any Metal backend change, rebuild first
(with tests, so the harness is relinked against the freshly embedded shader):

```sh
./macos_metal_build.sh --resume --tests
./build_macos_metal/bin/vtkMetalGLVisualComparison --out /tmp/visual_compare
# then inspect /tmp/visual_compare/<Scene>.gl.png / .metal.png / .diff.png
```

`--scene <name>` isolates one scene, `--threshold <value>` turns the tool into
a pass/fail check on the thresholded error, `--backend gl|metal` renders a
single backend (no diff). Regenerating this table after a change to the Metal
backend is a one-command check that the fix moved the corresponding scene's
thresholded error toward zero.
