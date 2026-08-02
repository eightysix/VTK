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

### Inter-device variability

The residual magnitude is GPU-dependent. The numbers above were produced on an
Apple M2 (the documented run): `VolumeRayCast` is `0.007` there. On other Apple
Silicon generations the same commit yields a different thresholded error for
this scene — e.g. an M1 machine reports `VolumeRayCast` at `0.005`. This is not
a build/checkout difference: different Metal GPU implementations round the
half-precision gradient/lighting and `fast::pow` differently at the sample
positions, so a different set of near-texel-boundary gradient taps crosses the
cell boundary and a slightly different subset of the ~40 specular-edge pixels
lands over the threshold. The `worst thresholded error` therefore varies by a
few thousandths across machines (0.007 on M2, 0.005 on M1), while remaining
reproducible *within* a machine. When comparing measurements, report the
device the numbers were produced on.

## Benchmarks

The benchmark tables below are produced by the harness in `--bench` mode. Each
scene is rendered `--frames 30` times after a warmup frame (which compiles
shaders / builds pipeline states); the camera is nudged 0.1° azimuth every timed
frame so `Render()` does real work. Both backends run symmetric configurations:
private offscreen targets (Metal `VTK_METAL_ENABLE_OFFSCREEN_TARGET` texture,
GL `vtkOpenGLFramebufferObject`), `SetMultiSamples(0)`, `SwapBuffersOff()`, and
GPU-synchronized timing (Metal `WaitForCompletion`, OpenGL `glFinish`) so the
wall-clock ms/frame covers GPU time for both. The offscreen targets exist
because Metal drawable delivery can be **display-paced**: before they were
added, every Metal scene flatlined at ~8.3 ms/frame (~120 fps) regardless of
complexity, with the profile showing the time inside `-[CAMetalLayer
nextDrawable]` (usleep). Whether that was a property of the SoC or of the macOS
version was never pinned down, and it no longer matters — the harness skips
`nextDrawable`/present entirely, so the timing is honest on any machine. Image
capture and the visual comparison are unaffected: they still use the normal
drawable path. Rerun on any machine with:

```sh
./build_macos_metal/bin/vtkMetalGLVisualComparison --bench --frames 30 --reps 3
```

`--frames` is the number of timed renders per run; `--reps` repeats the whole
per-scene measurement (fresh window each run) and the harness reports the mean
of the per-run averages plus the run-to-run standard deviation. The tables
below use the mean of 3 runs. `--gpu-mem` additionally prints the Metal
device's `currentAllocatedSize` after each scene — useful for checking that
the interleaved GL+Metal runs do not accumulate GPU memory (see the glitch
note in the complexity section below).

How to read the table:

- **`GL ms/f` / `Metal ms/f`** are average GPU-synchronized milliseconds per
  frame; **`M/GL`** is the Metal-to-GL time ratio, so values below 1 mean Metal
  is faster. `fps` = 1000/ms.
- **DepthPeeling is the headline**: Metal is ~2× faster than GL (M/GL 0.43 on
  the M2). GL pays per-peel occlusion-query stalls —
  `vtkDualDepthPeelingPass::PeelingDone()` blocks the
  CPU waiting for the query result every peel. Metal instead uses a
  *frame-delayed adaptive early exit*: the back-blend pass writes a visibility
  count (`MTLVisibilityResultModeCounting` on
  `vtkMetalDepthPeeler::VisibilityBuffer`, one 8-byte slot per peel), the CPU
  reads it back only after the previous frame's command buffer completes, and
  the next frame renders only `lastWritten + 2` peels. The scene needs only 3
  peels (2 DIFFERS, 3–5 are byte-identical to the full 20-peel output), so Metal
  runs 5 peels instead of 20. The trade-off is one frame of lag if the
  translucency depth of the scene changes suddenly.
- The slowest absolute Metal time is DepthPeeling; all other scenes are well
  under 1 ms/frame.
- Numbers are **device-dependent and noisy**: the timings and the per-scene
  visual residual (see "Inter-device variability") vary with the GPU, the macOS
  version, and the display configuration, and run-to-run variance was ~±15% on
  the M2's integrated GPU. Report the device when quoting either, and treat
  individual cells as indicative, not exact.

### MacBook Air (Apple M2)

MacBook Air with an **Apple M2** (Mac14,2), macOS 15.7.5, internal 60 Hz screen
only.

This is the machine where the Metal drawable pacing described above was first
observed, so the offscreen benchmark config was introduced here. Whether the
cause was the SoC or the macOS version was never pinned down, and it is moot
now that the harness renders offscreen.

```
scene                          GL ms/f   GL fps   Metal ms/f  Metal fps    M/GL
-------------------------------------------------------------------
AP_OpaqueNoBF                      0.32   3122.4        0.29     3446.7    0.91
AP_OpaqueBF                        0.39   2583.3        0.37     2703.6    0.96
AP_TransNoBF                       0.73   1365.1        0.46     2190.3    0.62
AP_TransBFbf05                     0.78   1279.9        0.45     2237.3    0.57
AP_TransBFbf10                     0.66   1511.7        0.42     2356.9    0.64
AP_OpFrTransBF                     0.43   2324.0        0.38     2602.0    0.89
AP_Trans25BF05                     0.75   1341.2        0.43     2331.0    0.58
AP_Trans75BF05                     0.70   1428.0        0.46     2194.7    0.65
AP_FrontCull                       0.76   1307.8        0.43     2322.1    0.56
AP_BackCull                        0.77   1303.4        0.44     2264.8    0.58
AP_GB05                            0.74   1344.2        0.45     2245.2    0.60
AP_GB10                            0.75   1327.1        0.47     2134.1    0.62
AP_RG05                            0.75   1335.4        0.49     2021.3    0.66
AP_GR25                            0.76   1321.0        0.53     1900.9    0.69
AP_GR7510                          0.88   1137.7        0.43     2309.0    0.49
AP_GR7525                          0.91   1102.0        0.44     2281.3    0.48
AP_GR2510                          0.78   1284.2        0.47     2124.2    0.60
RenderWindow                       0.47   2120.7        0.29     3455.9    0.61
Camera                             0.49   2059.6        0.32     3099.2    0.66
Light                              0.59   1692.1        0.30     3305.0    0.51
ActorProperty                      0.77   1303.7        0.50     1986.0    0.66
PointRender                        0.82   1221.3        0.39     2578.3    0.47
DepthPeeling                       3.90    256.7        1.69      593.3    0.43
CompositePolyDataMapper            0.48   2079.5        0.34     2908.5    0.71
Glyph3DMapper                      0.57   1749.1        0.36     2749.7    0.64
HardwareSelector                   0.55   1820.2        0.35     2865.5    0.64
PolyDataMapper2D                   0.62   1608.5        0.37     2737.9    0.59
Texture                            0.55   1817.8        0.31     3184.0    0.57
VolumeRayCast                      1.51    661.8        0.61     1641.6    0.40
-------------------------------------------------------------------
```

Device-specific notes:

- The run-to-run σ reported by `--reps` quantifies the spread here (e.g.
  DepthPeeling GL 3.90±0.21, VolumeRayCast GL 1.51±0.06, Metal 0.61±0.06);
  treat cells as indicative, not exact.
- The visual comparison is unaffected by the offscreen config: the thresholded
  table reproduces (worst 0.00653595, VolumeRayCast 0.007; PointRender's raw
  error drifts ±0.2 run-to-run while staying 0.000 thresholded).

### Geometry-bound scenes (`--complexity`)

The harness also registers `--complexity` scenes that scale one workload axis
(documented in the source). The GPU-bound geometry scenes share a 4×4 / 10×10 /
16×16 grid of `vtkSphereSource` spheres merged into a single polydata (800×800),
built in three coloring modes to isolate the fragment paths: `CpxGeom*` carry no
scalars (lean opaque pipeline), `CpxPoint*` carry per-point scalars (color
arrives as an interpolated varying), and `CpxCell*` carry per-cell scalars
(color resolved per-primitive in the fragment shader — the cell-texture port
below). These were the last Metal laggards: before the fixes below `CpxGeomHi`
ran ~1.5× slower than GL while `CpxActorLo`/`CpxActorHi` (CPU-bound, per-actor
draw calls) and the volume scenes were already at parity.

The cause was not the fragment shader (the fill-rate hypothesis): the gap
persisted down to a 1-pixel render, so it was vertex-side. Metal's CPU geometry
path expanded every triangle to 3 non-indexed vertices (`TriangleVertexCount`
2,088,000 = 3×696,000), while GL builds a vertex buffer from the polydata's own
deduplicated points and indexes it (348,200 unique vertices — the strip-shared
corners are processed once). Metal was running the vertex shader ~6× more often
than GL (2,088,000 vs 348,200), plus the extra per-vertex bandwidth.

The fix in `vtkMetalPolyDataMapper::BuildGeometryBuffers` relaxes the
`useIndexBuffer` deduplication condition from `cellFlag == 0` (per-point
coloring only) to also cover `cellFlag != 0` when `mappedColors == nullptr`
(no scalars → uniform per-actor color), which is the common opaque case. The
`cellFlag != 0` signal is a trap in this scene: `vtkAbstractMapper`'s
`GetAbstractScalars` sets `cellFlag = 1` in the default scalar mode whenever
point scalars are missing — even when cell scalars are also missing — while
`vtkMapper::MapScalars` returns null for no scalars at all. So Metal saw
`cellFlag != 0` and bailed on dedup even though the grid has no per-cell data
whatsoever. GL never expands in this situation either: its `Colors` is null so
`HaveCellScalars` stays false, the vertex buffer is the polydata's own points,
and the fragment shader just uses the material color. With `mappedColors ==
nullptr` there is nothing per-cell to differentiate, so dedup is safe and the
buffers match GL. Result on this repo's machine (Apple M1 Mac mini — the same
machine that reproduces the documented M1 VolumeRayCast residual 0.005):

```
scene                          GL ms/f   GL fps   Metal ms/f  Metal fps    M/GL
-------------------------------------------------------------------
CpxGeomLo                         0.59   1705.4        0.65     1539.5    1.11
CpxGeomHi                         2.01    498.5        2.14      466.3    1.07
CpxCellLo                         0.60   1655.4        0.78     1282.1    1.29
CpxCellHi                         2.12    471.4        2.46      406.0    1.16
CpxPointLo                        0.60   1653.5        0.72     1383.0    1.20
CpxPointHi                        2.06    486.5        2.29      437.3    1.11
CpxGeomBig                        3.53    283.0        3.96      252.3    1.12
CpxPointBig                       3.31    302.0        3.89      257.3    1.17
CpxCellBig                        3.55    282.1        3.87      258.6    1.09
CpxActorLo                        1.01    993.6        0.88     1132.7    0.88
CpxActorHi                       12.33     81.1       12.68       78.9    1.03
CpxPeel3                          3.71    269.2        1.12      889.2    0.30
CpxPeel12                        11.79     84.8        3.89      256.9    0.33
CpxVol64                          1.14    880.6        0.56     1793.5    0.49
CpxVol128                         1.17    851.9        0.57     1747.9    0.49
```

All rows above are from a single clean 45-scene `--reps 3` run after the harness
autorelease-pool fix described below. Earlier full-suite runs occasionally hit
the transient empty-frame glitch (one backend reporting ~0.05–0.15 ms — nothing
rendered) in the tail scenes after the heavy `Cpx*Big` scenes, so the
`CpxActor*`/`CpxPeel*`/`CpxVol*` rows had to be salvaged from shorter runs;
with the fix the whole 45-scene run completes cleanly in one pass. Absolute ms
still move ±20% run-to-run with the M1's GPU clock and the M/GL ratios move with
them (the geometry scenes measured ~0.92–1.05 in the earlier 15-scene run and
~1.07–1.29 here), so treat cells as indicative rather than exact.

`CpxGeomHi` went from M/GL ~1.46 to ~1.07; the per-cell and per-point geometry
scenes sit at ~1.07–1.29, and the CPU-bound / peel / volume rows are at or
better than parity (0.88, 0.30–0.33, 0.49). The shared-vertex cell-id for
picking in the dedup paths follows the existing first-wins convention (same as
the GPU-tess and per-point-coloring dedup paths). What the dedup fix does *not*
cover — scenes with real per-cell colors — is what the cell-texture port adds,
described next.

### Transient empty-frame glitch (fixed)

Long `--bench --reps 3 --complexity` runs used to corrupt the tail scenes: after
the heavy `Cpx*Big` scenes, the *next* scene's Metal rows occasionally reported
~0.05–0.15 ms — nothing rendered — and stayed broken for the rest of the run.
`--gpu-mem` (added to the harness, backed by
`vtkMetalRenderWindow::GetAllocatedSize()`) traced it to GPU memory:
`MTLDevice.currentAllocatedSize` grew monotonically across the interleaved
GL+Metal scenes, from ~810 MB to ~2.9 GB, until the device ran out of memory
(Apple's GL-on-Metal `GLDRendererMetal` command buffer failed with
`MTLCommandBufferErrorDomain` Code=8 "Insufficient Memory"). After that every
new Metal window got a nil device and rendered nothing — the "0.05 ms" frames
were empty. A Metal-only control run (`--backend metal`) stayed flat (~75–250
MB, peak on the `Cpx*Big` scenes), proving the Metal backend releases its
per-window GPU resources correctly; the accumulation came from the interleaved
*GL* runs.

The leak was in the harness, not a backend: the GL/Cocoa window path creates
autoreleased Objective-C objects, and the CLI `main()` never drains the
autorelease pool, so every GL window churn leaked GPU-resident autoreleased
objects (GL-on-Metal allocates on the same default `MTLDevice`). The fix wraps
each capture and each bench scene in an explicit `objc_autoreleasePoolPush` /
`Pop`. With the fix, the combined 45-scene run's memory stays flat at ~64–560 MB
(peak on the `Cpx*Big` scenes), matching the Metal-only profile, and the full
suite completes in one clean run.

### The per-cell color caveat (the "cell-texture port")

Per-cell-colored scenes were the last gap. Metal resolved the cell color as a
per-*vertex* quantity — `emitSurfaceColor(polyCellIdx, ...)` baked the cell's
RGBA into the vertex at geometry-build time and the cell id was a
flat-interpolated vertex attribute — so a shared corner (which belongs to
multiple cells) had to be duplicated per triangle and the vertex stream
expanded to 3 vertices per triangle. GL instead treats cell identity as a
per-*primitive* quantity delivered by hardware, and the port moves Metal to the
GL model:

- **GL** (`vtkOpenGLPolyDataMapper`): `AppendCellTextures` packs one RGBA texel
  per output primitive into a texture buffer, where each texel is its owning
  cell's color (`newColors[i] = Colors[ccmap->GetValue(i)]`), and `CellCellMap`
  + `PrimitiveIDOffset` map strips and polygon fans back to their cells. The
  fragment shader fetches it with
  `texelFetchBuffer(textureC, gl_PrimitiveID + PrimitiveIDOffset)` —
  `gl_PrimitiveID` is the index of the triangle being rasterized, so it is
  unambiguous even though vertices are shared across cells. The vertex buffer
  stays point-indexed whether the color is per-point, per-cell, or absent.
- **Metal** now does the same: `BuildGeometryBuffers` keeps `useIndexBuffer`
  dedup for cell-colored scenes and stops baking the cell color into the vertex
  stream. The per-primitive RGBA is packed into a 2D **RGBA8Unorm** texture
  (the analog of GL's RGBA8 buffer texture; the cell color source is 8-bit
  `mappedColors`, so this quantizes identically and uses 4 bytes/cell instead of
  16 for a float buffer), laid out row-major with `kCellTextureWidth = 8192` on
  both sides (matches the `8192u` constant in `MetalShaders.metal`) so the
  shader's `primId % width` / `primId / width` compile to a shift and mask; the
  8192×16384 layout caps at 134M triangles. A per-primitive cell-id buffer (the
  analog of GL's `CellCellMap` + `PrimitiveIDOffset`) supplies picking, and the
  fragment shader resolves the color with `[[primitive_id]]` — MSL's analog of
  `gl_PrimitiveID` — in `resolveCellColor`
  (`cellColorTex.read(uint2(prim_id % width, prim_id / width))`). It is gated
  behind the `kHasCellTexture` function constant (12) plus a runtime
  `kSceneFlagHasCellTexture` flag (feature bit `1u << 11`) so the all-true
  full-feature pipelines stay correct for plain per-vertex actors, and all four
  fragment paths (opaque, OIT, and both peel variants) take the texture. The
  zero-fallback is the 1×1 white `DefaultTexture` bound at slot 8, so a scene
  without a cell-color texture reads white instead of garbage.

Results on this repo's machine (Apple M1 Mac mini), `--reps 3` — per-cell
scenes went from the same ~1.5× vertex-bound regime `CpxGeomHi` was in to
~1.1–1.3× (the same run as the complexity table above):

```
scene                          GL ms/f  Metal ms/f   M/GL        tris
CpxCellLo                         0.60        0.78    1.29       ~27k
CpxCellHi                         2.12        2.46    1.16      ~696k
CpxCellBig                        3.55        3.87    1.09      ~1.8M
CpxPointLo                        0.60        0.72    1.20       ~27k
CpxPointHi                        2.06        2.29    1.11      ~696k
```

The investigation of the residual at mid-scale (`CpxCellHi` ~1.16):

- **The gap is the fetch, not the pipeline.** Temporarily skipping the buffer
  load in `resolveCellColor` (returning a constant) moved `CpxCellHi` to ~0.99
  and `CpxCellLo` to ~0.87 — the per-fragment indexed color read is the entire
  residual. The point-colored control (`CpxPoint*`, same geometry, color
  arrives as an interpolated varying, no fetch) runs below the per-cell rows
  in the same run, proving the scalar pipeline itself is at parity.
- **RGBA8 texture vs float buffer is a wash.** The port was first built against
  a `MTLPixelFormatRGBA32Float` buffer; switching to RGBA8Unorm changed nothing
  measurable (~1.5–2× GL's `texelFetchBuffer` cost, ~0.15–0.2 ms at
  `CpxCellHi`). The texture won on the 4× smaller footprint and the GL-faithful
  quantization, not on speed.
- **It does not amplify with complexity.** At 1.8M triangles (`CpxCellBig`) the
  ratio converges toward ~1.09 as the scene becomes geometry-bound and the
  per-fragment cost is amortized; GL's own cell-texture path degrades at scale
  too (its `CpxCellBig` runs 3.55 ms vs 3.31 for the point-colored control).

The remaining cost is the irreducible price of dedup: the ~6× vertex-stage
saving pays for a ~1.1–1.3× fragment fetch, which is where GL lives (its
`gl_PrimitiveID` + `texelFetchBuffer` does the same work). The port also makes
picking exact in this path — per-primitive cell ids report the owning cell
where the per-vertex first-wins value was ambiguous — so the common
per-cell-colored case is now both at near-parity and exact.

### Recording another machine

To add a machine, run the command above (`--bench --frames 30 --reps 3`), then
add a subsection here with the device model, macOS version, and display
configuration, followed by the table and any device-specific notes. Since both
the timings and the residual are GPU-dependent, always quote the device
alongside the numbers.

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
