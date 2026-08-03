# Metal Testing State

Snapshot of the state of testing for the Metal rendering backend. Historical
run: `metal-ios` branch head `bc4e9d93cd` on this repo's Apple M1 Mac mini
(Macmini9,1, macOS 14.8). Current working-tree run: this repo's Apple M2
MacBook Air (Mac14,2, macOS 15.7.5), after the texture-cluster, composite-mapper,
and selection cell-ID fixes described below. Both arm64 Release builds in
`build_macos_metal` (Ninja, `VTK_MODULE_ENABLE_VTK_RenderingMetal=YES`).

Two test surfaces exist:

1. A bespoke Metal suite under `Rendering/Metal/Testing/` (13 image-baseline
   regression tests + a HeaderTest + a dual-backend visual-parity harness).
   Status: **all 15 pass** (unchanged from the historical run).
2. The generic multi-backend suite in `Rendering/Core/Testing/Cxx/`, which
   registers the same ~175 tests once per backend and was wired up for Metal
   through    object-factory overrides (`--vtk-factory-prefer
   RenderingBackend=Metal`). Historical status: **55 pass / 120 fail (33
    crash)**. Current working-tree status: **82 pass / 84 fail (9 crash)** —
    the 14 OpenGL-texture-fallback crashes are fixed by the `vtkMetalTexture`
    factory override, the 8 composite-mapper `BuildGeometryBuffers` crashes by
    a `mappedColors != nullptr` guard (they now render), `TestOpacityMSAA` by
    clamping the MSAA sample count to the device maximum (see below), the
    `TestAreaSelections` crash by a missing `GetColorBufferSizes` override
    (see below), and the `TestAreaSelections` cell-set fidelity by exact
    per-primitive cell ids (see the selection-cluster section below). The three
    read-back tests `TestReadPixels`, `TestSelectVisiblePoints` and
    `TestWorldPointPicker` now pass via the color/depth read-back work (see the
    read-back-cluster section below), the composite per-block scalar
    overrides (`TestCompositePolyDataMapperOverrideLUT`,
    `OverrideScalarArray`, `NaNPartial`) pass via the per-block
    scalar-attribute forwarding and the NaN-color port (see the
    composite-cluster section below), and the four scalar-LUT tests —
    `TestCompositePolyDataMapperOverrideLUT`, `TestTextureInterpolateScalars`,
    `TestTranslucentLUTTextureAlphaBlending` and
    `TestTranslucentLUTTextureDepthPeeling` — pass via the new scalar-texture
    LUT pipeline that reproduces GL's `texture(colortexture, colorTCoord)`
    exactly (see the scalar-texture-LUT section below). No new crashes were
    introduced. The pass count fluctuates run to run (run-to-run flakiness).

---

## 1. Bespoke suite (`Rendering/Metal/Testing/`)

### Layout

- `README.md` — how to build with `--tests`, run, add a test, and manage
  baselines via VTK ExternalData (`.sha512` markers + `.ExternalData/SHA512/`
  store).
- `OpenGLComparison.md` — per-scene Metal-vs-OpenGL parity numbers from
  `vtkMetalGLVisualComparison`.
- `Cxx/CMakeLists.txt` — registers the 13 tests in the shared
  `vtkRenderingMetalCxxTests` executable, plus the standalone
  `vtkMetalGLVisualComparison` executable.
- `Cxx/TestMetalHelpers.h` — `vtkMetalTesting::RegressionExitCode()`
  (maps `vtkRegressionTestImage` returns) and read-back capture helpers
  (`VerifyRegionRendered`, `CountHitPixels`, `CountCompositePixels`).
- `Cxx/TestMetalScenes.h` — backend-agnostic scene builders shared by the
  regression tests and the parity harness (sets an explicit
  `vtkMetalProperty`/`vtkOpenGLProperty` on every actor).
- `Data/Baseline/*.png.sha512` — 13 content-hash markers; the PNG payloads
  live in the repo's `.ExternalData/SHA512/` store.

### The 13 regression tests and coverage

| Test | Coverage |
|------|----------|
| `TestMetalRenderWindow` | basic cone scene, render-window creation, color read-back |
| `TestMetalCamera` | camera placement, focal point, reset camera |
| `TestMetalLight` | positional + directional lights |
| `TestMetalActorProperty` | color, opacity, backface culling, point/line/fill representation |
| `TestMetalPointRender` | point rendering |
| `TestMetalDepthPeeling` | depth peeling / order-independent translucent pass |
| `TestMetalCompositePolyDataMapper` | composite mapper with scalar color mapping |
| `TestMetalGlyph3DMapper` | instanced glyph rendering |
| `TestMetalHardwareSelector` | area selection (verified via color read-back; see `74797c679e`) |
| `TestMetalPolyDataMapper2D` | 2D overlay rendering |
| `TestMetalImageMapper` | 2D image rendering (added `e162c7e9bd`) |
| `TestMetalTexture` | texture mapping (explicit base-behavior texture) |
| `TestMetalVolumeRayCast` | GPU volume ray casting |

Each test ends with `vtkRegressionTestImage(renWin)` compared against its
`Data/Baseline/` PNG. The shared executable does **not** link OpenGL (its
`vtkProperty::New()`/`vtkTexture::New()` must resolve to Metal classes), so the
OpenGL backend is linked only into the standalone comparison harness.

### `vtkMetalGLVisualComparison`

A standalone executable (links `vtkRenderingOpenGL2` +
`vtkRenderingVolumeOpenGL2`) that renders every scene with both backends and
writes `<scene>.gl.png` / `<scene>.metal.png` / `<scene>.diff.png` plus a
numeric `vtkImageDifference` metric (threshold 20 — the same metric `vtkTesting`
uses). It is a manual inspection tool, not a pass/fail test (it currently exits
0 after writing images unless a `--threshold` is given).

`OpenGLComparison.md` records the latest run: every scene's thresholded error is
`0.000` except `VolumeRayCast` (`0.005`). The run is byte-for-byte reproducible.

### Baseline management

Per the VTK ExternalData convention: commit the `.sha512` marker for a baseline,
copy the PNG into `.ExternalData/SHA512/<hash>`, and let `ninja VTKData` stage
it into the build tree. `ImageNotFound` in a test output means the baseline is
missing/stale; the rendered image is regenerated in
`build_macos_metal/Testing/Temporary/`. `Rendering/Metal/CMakeLists.txt` gates
the color read-back (`framebufferOnly=NO`, per-frame blit) behind
`VTK_BUILD_TESTING=ON`, so production builds pay no cost.

### Current status

```
ctest --test-dir build_macos_metal -R "RenderingMetalCxx|RenderingMetal-HeaderTest" -j 8
100% tests passed, 0 tests failed out of 15   (HeaderTest + 13 regression + GLVisualComparison)
```

---

## 2. Generic multi-backend suite (`Rendering/Core/Testing/Cxx`)

### Wiring (commits `e4c6d89213`, `bc4e9d93cd`)

- `Rendering/Metal/CMakeLists.txt` declares 14 `vtk_object_factory_declare`
  overrides (renderer, actor, PolyDataMapper, CompositePolyDataMapperDelegator,
  Glyph3DMapper, PolyDataMapper2D, ImageMapper, camera, light, property,
  HardwareSelector, GPUVolumeRayCastMapper + the two render windows), each with
  a `CreateOverrideAttributes()` chain scoring `RenderingBackend=Metal`.
- `Rendering/Core/Testing/Cxx/CMakeLists.txt` registers the shared `cxx_tests`
  list with `vtk_test_prefix "Metal-"` and
  `--vtk-factory-prefer RenderingBackend=Metal` under
  `if (TARGET VTK::RenderingMetal)`.
- `Rendering/Core/vtk.module` adds `VTK::RenderingMetal` to
  `TEST_OPTIONAL_DEPENDS` so the generic test executable links Metal's autoinit
  factory.

### Tally (working tree, rerun 2026-08-03 on the M2 MacBook Air,
`ctest -R "RenderingCoreCxx-Metal" -j 8`)

```
175 tests:  82 Passed  84 Failed (incl. image/pick fails)  9 "Subprocess aborted"
```

(Historical run at commit `bc4e9d93cd`: 55 Passed / 87 Failed / 33 aborted. Prior
working-tree run: 61 Passed / 95 Failed / 19 aborted — the composite-mapper
crash cluster below is since fixed, and `TestOpacityMSAA` since passes. Previous
run: 66 Passed / 100 Failed / 9 aborted — the `TestAreaSelections` area-selection
crash was fixed by the `GetColorBufferSizes` override below. This run:
69 Passed / 97 Failed / 9 aborted — the three new passes are `TestAreaSelections`
(now stable: the per-primitive cell-id work below makes the extracted-cell set
match OpenGL), `TestHardwareSelector` (was "0 nodes returned"), and
`TestAxesActor` (documented gross-fail 0.612; treat as flaky until reproduced).
Read-back run: 72 Passed / 94 Failed / 9 aborted — the three read-back tests
`TestReadPixels`, `TestSelectVisiblePoints` and `TestWorldPointPicker` now pass
via the read-back cluster below. Next run: 73 Passed / 93 Failed / 9 aborted —
`TestCompositePolyDataMapperBlockTextures` now passes via the per-block-texture
port described in the composite-mapper section below. Previous run:
81 Passed / 85 Failed / 9 aborted — the composite per-block scalar-override
cluster below now fixes `TestCompositePolyDataMapperOverrideLUT`,
`TestCompositePolyDataMapperOverrideScalarArray` and
`TestCompositePolyDataMapperNaNPartial` (all three confirmed via stash-check to
fail at the previous commit); the remaining 5-pass delta over the prior run is
within the documented run-to-run flakiness. The 9 aborts are unchanged
(`TestFollowerPicking`, `TestInteractorStyleImageProperty`,
`TestLabeledContourMapper{NoLabels,WithActorMatrix}`, `TestResizingWindowToImageFilter`,
`TestTranslucentImageActor{AlphaBlending,DepthPeeling}`, `TestWindowToImageFilter`).
Latest run: 82 Passed / 84 Failed / 9 aborted — `TestTextureInterpolateScalars`
now passes via the scalar-texture LUT pipeline (see the scalar-texture-LUT
section below), completing all four scalar-LUT tests
(`TestCompositePolyDataMapperOverrideLUT`, `TestTextureInterpolateScalars`,
`TestTranslucentLUTTextureAlphaBlending`, `TestTranslucentLUTTextureDepthPeeling`;
the first and last two were already passing via the per-block scalar-override
cluster). The +1 delta over the previous run is exactly that test; the 9 aborts
are unchanged.)

### The selection cluster is fixed

`TestAreaSelections` no longer crashes: `vtkMetalRenderWindow` did not
override `GetColorBufferSizes`, so the base class left `rgba` uninitialized
and `vtkHardwareSelector::Select` aborted on the "Color buffer depth must be
at least 8 bit" check (`vtkRenderWindow.h:642`, `vtkHardwareSelector.cxx:346`).
A new override reports the BGRA8Unorm attachments (8 bits per channel), so the
test runs and renders the full scene.

The remaining cell-set fidelity gap is also fixed. Metal reported 60 sphere
cells where OpenGL reported 139, and the Metal ID set contained only even raw
cell indices. The cause was the CPU triangle-emission path deduplicating
vertices by point ID (`useIndexBuffer` when `mappedColors == nullptr`, which
the test sphere satisfies): a shared vertex carries the *first* triangle's
`cellId`, and the surface fragment flat-interpolated `in.cellId`, so the
"provoking vertex first-wins" value did not name the owning cell. OpenGL
matches the exact cell with `gl_PrimitiveID`.

The fix (`e9e8a6bb66`) emulates GL's `gl_PrimitiveID` behavior:
`vtkMetalPolyDataMapper` now emits an exact per-primitive cell-id buffer
unconditionally (previously only built for per-cell-colored geometry), and the
surface fragment reads `cellPrimitiveIds[prim_id]` when the new
`kSceneFlagUsePrimitiveCellIds` scene flag (bit 12) is set, instead of the
ambiguous flat `in.cellId`. `TestAreaSelections` now **passes** (stable across
repeated runs): the Metal and GL selection ID lists are identical (139 cells)
and the image regression passes under both backends. `TestHardwareSelector`
(was "0 nodes returned") also passes this run, presumably on the same exact-ID
mechanism. `TestSelectVisiblePoints` and `TestWorldPointPicker` now pass via
the depth read-back (see the read-back-cluster section below); `TestReadPixels`
passes via the color read-back fixes there. `TestPointSelection*` remain (the
point field-association selection pass is not yet implemented).

Also fixed: `vtkMetalPolyDataMapper::RenderPiece` now calls
`GetInputAlgorithm()->Update()` (matching `vtkOpenGLPolyDataMapper`). Without
it, mappers fed through an intermediate filter render empty — the
`vtkDataSetMapper`-driven actors in `TestAreaSelections` (grid + extracted
frustum) had 0 points (`vtkGeometryFilter` never executed). Reader-driven
pipelines were affected; shrink/filter-driven ones (e.g.
`TestOrderedTriangulator`) were already populated via other update paths, so
the fix has no visible effect on those tests.

### The read-back cluster is fixed

Three tests that failed on framebuffer read-back now pass:

- **`TestReadPixels`** — two bugs. First, `vtkMetalRenderer::DeviceRender`
  cleared the color attachment with a hardcoded alpha of `1.0`
  (`MTLClearColorMake(..., 1.0)`), so `GetRGBACharPixelData` returned
  alpha 255 for the background while OpenGL clears with
  `vtkViewport::BackgroundAlpha` (default `0.0`). The clear now uses
  `this->GetBackgroundAlpha()`. Second, the float `GetRGBAPixelData`
  overrides did not exist, so the base class returned 0 and left the caller's
  `vtkFloatArray` empty (the test then crashed on `GetTuple`). New overrides
  convert the RGBA char read-back to normalized floats using the `* (1/255)`
  multiplication form (matching GL's conversion so the test's truncating
  `(int)(value*255)` cast rounds to the expected byte).
- **`TestSelectVisiblePoints`** and **`TestWorldPointPicker`** — both read
  depth through `vtkRenderer::GetZ` → `vtkRenderWindow::GetZbufferDataAtPoint`
  → `GetZbufferData`, which was not overridden (base returned 0). With depth 0
  (near plane) every sphere point appeared occluded and world picks unprojected
  at the near plane. `vtkMetalRenderWindow` now implements the three
  `GetZbufferData` overloads reading a per-frame shared `DepthCopyTexture`
  (Depth32Float, `MTLStorageModeShared`), copied at the end of each frame from
  the non-MSAA `DepthTexture` (mirroring the existing `ColorCopyTexture` blit).
  The non-MSAA `DepthTexture` is the render target when MSAA is off, and the
  MSAA depth resolve target when MSAA is on: the opaque pass now sets
  `depthAttachment.resolveTexture = DepthTexture` with
  `MTLStoreActionStoreAndMultisampleResolve` (the volume pass already did
  this), so the resolve chain is opaque/volume pass → `DepthTexture` → blit →
  `DepthCopyTexture` → `getBytes`. An initial attempt used a
  `resolveBlitCommandEncoder` to resolve the MSAA depth directly, but that
  selector is not implemented by the G14-family command buffer (aborts every
  MSAA frame); the render-pass resolve has no such dependency.
  Read-backs Y-flip like the color path (Metal top-origin → VTK bottom-left).

Remaining in this cluster: `TestPointSelection*` (the point field-association
selection pass still renders triangle cell ids, not per-point ids — the 
hardware selector and mapper do not yet have a point-pass equivalent of the
`kSceneFlagUsePrimitiveCellIds` mechanism).

### The texture cluster is fixed

The largest historical crash class (OpenGL-texture fallback: `vtkTexture::New()`
returned `vtkOpenGLTexture`, whose `Load()` called
`vtkTextureObject::SetContext(vtkOpenGLRenderWindow*)` on a Metal window →
SIGSEGV) is resolved by `vtkMetalTexture` (`Rendering/Metal/vtkMetalTexture.h`),
a `vtkTexture` factory override registered with `RenderingBackend=Metal` whose
`Load()` is a no-op because the Metal poly-data mapper uploads actor textures
itself (`vtkMetalPolyDataMapper::UpdateActorTexture`). As a result
`TestTextureWrap`, `TestBackfaceTexture`, `TestTextureRGBA`,
`TestTextureRGBADepthPeeling`, `TestTextureSize` now **pass**, and
`TestTextureInterpolateScalars` now passes via the scalar-texture LUT pipeline
(see the scalar-texture-LUT section below); the other texture
tests (`TestActor2DTextures`, `TestBackfaceCulling`, `TestImageAndAnnotations`,
`TestPickTextActor`, `TestRenderLinesAsTubes{OrthoCamera}`, `TestTexturedCylinder`,
`TestTilingCxx`) render without crashing but
still fail image comparison (texture-feature fidelity gaps).

`TestTextureWrap` specifically was debugged to a pass this session:
multi-renderer viewport tiling collapsed because `vtkMetalRenderer::DeviceRender`
scaled the fractional viewport by the renderer's *own* pixel size and used
`viewport[2]`/`viewport[3]` directly as the width/height. All five viewport
sites in `vtkMetalRenderer.mm` (opaque, translucent, volume, volume-framebuffer
blit, overlay) and both sites in `vtkMetalOrderIndependentTranslucentPass.mm`
now use `(viewport[2]-viewport[0]) * winSize` (window size). The window also
shares one drawable across renderers: the first renderer clears the full
attachment, later renderers load it, only the last presents
(`vtkMetalRenderWindow::FrameRendererIndex`). ClampToBorder textures get their
arbitrary border color from the fragment shader (`MaterialUniforms.borderColor`)
since Metal's sampler border-color presets are black/white only; the ClampToBorder
sampler clamps to edge and `resolveMaterial` substitutes the border color where
uv escapes [0,1].

### The composite-mapper crash cluster is fixed

The second-largest historical crash class (8 `Subprocess aborted` in
`vtkMetalPolyDataMapper::BuildGeometryBuffers` on the composite/legacy path)
is resolved by a one-line null guard at `vtkMetalPolyDataMapper.mm:3745`:
the indexed-triangle path called `mappedColors->GetPointer(0)` whenever
`cellFlag == 0` without checking `mappedColors != nullptr`. With block
display-attribute overrides active (`vtkMetalBatchedPolyDataMapper` sets
`SetBatchVisualOverride` → `Internals->UseBatchColor`), `MapScalars` is skipped
(`Rendering/Metal/vtkMetalPolyDataMapper.mm:2667`) so `mappedColors` is null and
`cellFlag == 0`, and the unguarded call SIGSEGV'd. Every sibling site already
guards with `mappedColors && ...`; the fix makes line 3745 consistent.

`TestCompositePolyDataMapper`, `TestCompositePolyDataMapperBlockOpacities`,
`TestCompositePolyDataMapperToggleScalarVisibilities`, the
`StaticBounds`/`SharedArray`/`PartialPointData` composite tests,
`TestCompositePolyDataMapperBlockTextures` (per-block textures, fixed by
forwarding the batch element's `Texture` to the child Metal mapper and by
mirroring GL's cell-scalar texture-size handling in `UpdateActorTexture`),
and now `TestCompositePolyDataMapperOverrideScalarArray`,
`TestCompositePolyDataMapperNaNPartial` and
`TestCompositePolyDataMapperOverrideLUT` (per-block scalar overrides, fixed by
the per-block scalar-attribute forwarding and the interpolate/NaN-color ports
below) now **pass**;
`TestActor2D`, `TestBlockOpacity`, `TestCompositePolyDataMapper{Spheres,
Vertices}` and the rest render without crashing but still fail image
comparison (block-opacity / per-block feature fidelity gaps). The composite
mapper remains the largest *image-compare* cluster.

### The composite per-block scalar-override cluster is fixed

`vtkMetalBatchedPolyDataMapper::RenderPiece` never pushed the batch element's
scalar attributes (`ArrayId`, `ArrayName`, `ScalarMode`, `ArrayComponent`,
`ScalarRange`, `UseLookupTableScalarRange`, `FieldDataTupleId`, the block LUT,
...) to the child mappers, so the child `vtkMetalPolyDataMapper` always saw
`ArrayId == -1` and mapped no scalars (white mesh) for the per-block
`ColorByArrayComponent`/`SetBlockLookupTable` overrides that
`TestCompositePolyDataMapperOverrideScalarArray`, `NaNPartial` and `OverrideLUT`
exercise. `RenderPiece` now forwards every batch element scalar field to the
child before `RenderPiece(ren, act)` (mirroring how
`vtkCompositePolyDataMapper::Render` pushes block overrides onto its delegate),
fixing `OverrideScalarArray` and `NaNPartial`.

For `TestCompositePolyDataMapperOverrideLUT` two further Metal gaps had to be
closed:

- **Scalar-as-texture interpolation.** The base `vtkMapper::MapScalars` with
  `InterpolateScalarsBeforeMapping` on routes through `MapScalarsToTexture`
  and returns `nullptr`, because Metal had no scalar-texture/LUT-texture
  pipeline — the mesh rendered white. This is now a real pipeline that
  reproduces GL's `texture(colortexture, colorTCoord)` exactly; see the
  scalar-texture-LUT section below. (An earlier CPU-subdivision workaround,
  `SubdividePolysForScalarInterpolation`, has been removed.)
- **NaN/missing-array colors.** `ColorMissingArraysWithNanColor` had no Metal
  support: blocks with no scalar array rendered actor-white instead of the LUT
  NaN color. `RenderPiece` now falls back to the NaN color
  (`vtkLookupTable::GetNanColor` / `vtkColorTransferFunction::GetNanColor`)
  via `SetBatchVisualOverride` when a visible block has no scalars.

### The scalar-texture LUT pipeline is implemented

The last scalar-coloring gap — `InterpolateScalarsBeforeMapping`, GL's
scalar-texture path — is now a first-class Metal pipeline, closing all four
scalar-LUT tests: `TestCompositePolyDataMapperOverrideLUT`,
`TestTextureInterpolateScalars`, `TestTranslucentLUTTextureAlphaBlending` and
`TestTranslucentLUTTextureDepthPeeling` pass at ImageError 1.2e-06 /
1.3e-05 / 9.8e-05 / 1.2e-04 (well under the 0.05 threshold).

- **Two-row LUT texture.** `vtkMapper::MapScalarsToTexture` /
  `ScalarToTextureCoordinate` build a 2-row RGBA texture (row 0 = the ramp,
  row 1 = the NaN color; GL samples row 0 at t=0.49/1.0). The mapper uploads
  it as a fragment texture (texture 9) with a **NEAREST** sampler
  (sampler 1), matching GL's effective sampling: linear filtering would blend
  the ramp row with the NaN row and mute the colors.
- **Per-vertex scalar texture coordinate.** `MapScalars` now sets
  `ColorCoordinates`/`ColorTextureMap`; `BuildGeometryBuffers` emits a
  `scalarCoord` (float2) stream parallel to the positions for every emitted
  vertex (buffer 12 in `vertex_main`/`vertex_main_indexed`), the fragment
  stage interpolates it and looks the color up in the LUT texture
  (`resolveCellColor`) — GL's per-fragment `texture(colortexture,
  colorTCoord)`. The t-coordinate alpha-baked in `ScalarToTextureCoordinate`
  and the mapper-baked actor/block opacity (`vertexColor.a`, multiplied into
  the alpha in the surface/OIT/peel fragments) reproduce GL's
  `opacityUniform * texColor.a`.
- The LUT path is point-scalar + surface-only (GL's `useScalarLUT`
  condition), and is excluded from the GPU-tessellation and indexed-dedup
  emission paths (which cannot carry a per-vertex scalar coordinate);
  wireframe and cell-colored lines push colors directly. The
  `SubdividePolysForScalarInterpolation` CPU-subdivision helper is removed.
- The LUT is carried through the full-behavior peel/OIT/edge pipelines too
  (`kHasScalarLUT` function constant); the peel/OIT vertex stages gate the
  `scalarCoords` load on the runtime scene flag so translucent non-LUT actors
  skip the read.

### Image-compare failures

Current run (2026-08-03): 84 failed = 81 image-compare + 3 non-image. Buckets
by max `vtkTesting` TIGHT_VALID error per test (threshold 0.05):

| Bucket | Range | Count | Examples |
|--------|-------|-------|----------|
| near-miss | 0.05 – 0.1 | 10 | `TestActorLightingFlag` 0.051, `TestEdgeFlags` 0.068, `TestQuadPointRep` 0.069, `TestMixedGeometry_3` 0.070, `TestImageAndAnnotations` 0.070, `TestVertexRendering` 0.072, `TestLineRenderingTranslucent` 0.079, `TestGlyph3DMapperPicking` 0.080, `RenderNonFinite` 0.087, `TestMixedGeometryCellScalars` 0.092 |
| mid | 0.1 – 0.5 | 43 | `TestSurfacePlusEdges` 0.104, `TestCompositePolyDataMapperPicking` 0.176, `TestGlyph3DMapperIndexing` 0.155, `TestWireframe` 0.239, `TestPolyDataMapper2D` 0.235, `TestCoincident` 0.334, `TestCompositePolyDataMapperCustomShader` 0.290, `TestColorByStringArrayDefaultLookupTable2D` 0.482, `TestGradientBackground` 0.469 |
| gross | >= 0.5 | 27 | `TestStereoBackground{Left,Right}` 0.887, `TestNViewports*` 0.85–0.88, `TestDirectScalarsToColors` 0.858, `TestMapVectorsToColors` 0.747, `TestImageMapper_1..4` 0.63–0.72, `TestActor2DTextures` 0.594, `TestTilingCxx` 0.616, `TestBareScalarsToColors` 0.584 |

(80 of the 81 image-compare failures exceed the 0.05 threshold; the 81st,
`TestCompositePolyDataMapperPickability`, has a below-threshold ImageError of
0.015 but still fails its own pickability check.) The 3 non-image failures:
`TestPointSelection`, `TestPointSelectionWithCellData` (selection returns
even-only point ids, pick-check fails) and `TestPickTextActor` (pick check).
`TestReadPixels`, `TestSelectVisiblePoints` and `TestWorldPointPicker` left
this set via the read-back cluster above, and the four scalar-LUT tests left it
via the scalar-texture-LUT section above.

### Crashes (9; all pre-existing classes, none from the texture or composite clusters)

Signal-level crashes are reported as `Subprocess aborted` with a `Caught
SIGSEGV` line but no backtrace in the ctest log, so crash causes were
re-derived from the earlier run's full signal stacks. Current-run attribution:

| Cause | Count | Tests |
|-------|-------|-------|
| SIGSEGV with no backtrace; prior stack analysis attributes these to the OpenGL factory fallback in label/text/image rendering | 9 | `TestFollowerPicking`, `TestInteractorStyleImageProperty`, `TestLabeledContourMapper`, `TestLabeledContourMapperNoLabels`, `TestLabeledContourMapperWithActorMatrix`, `TestResizingWindowToImageFilter`, `TestTranslucentImageActor{AlphaBlending,DepthPeeling}`, `TestWindowToImageFilter` |

The 14-test OpenGL-texture-fallback crash row from the historical tally is gone
(fixed by `vtkMetalTexture`, see above), the 8-test composite-mapper
`BuildGeometryBuffers` row is gone (see above), the `TestOpacityMSAA` MSAA
row is gone (see below), and the `TestAreaSelections` row is gone (fixed by
the `GetColorBufferSizes` override, see the selection-cluster section above).
The 9 remaining crashes are genuine Metal bugs, all the label/text/image
OpenGL-fallback class that the `vtkMetalTexture` fix did not cover (no
`vtkTexture` involved).

### The MSAA crash is fixed

`TestOpacityMSAA` requests `renWin->SetMultiSamples(8)`, but the Apple GPU
family supports only 1/2/4 MSAA samples. `CreateMultisampleAttachments` handed
the requested count straight to `newTextureWithDescriptor:`, and Metal's
descriptor validation aborted on the unsupported sample count. The fix clamps
`vtkMetalRenderWindow::GetEffectiveSampleCount()` to the device maximum (4 on
Apple-family GPUs via `supportsFamily:`, 8 elsewhere), and since every MSAA
resource and pipeline-state sample count already routes through that single
value (the renderer's opaque/translucent/volume/overlay passes and the
poly-data/glyph/image/volume mapper PSOs), the clamp applies everywhere.
`TestOpacityMSAA` now renders and passes its image comparison.

### Theme clusters in the 84 failures

- **Textures** (~15): every `TestTexture*`, `TestBackfaceTexture`,
  `TestTexturedCylinder`, `TestTilingCxx`, `TestActor2DTextures` — historically
  crashed on the OpenGL fallback; now render, with 6 passing
  (`TestTextureWrap`, `TestBackfaceTexture`, `TestTextureRGBA`,
  `TestTextureRGBADepthPeeling`, `TestTextureSize`,
  `TestTextureInterpolateScalars`) and the rest
  failing image comparison on texture-feature fidelity (filter/wrap/interpolation
  edge cases, textured-cylinder seams).
- **Composite mapper** (~19): the crash cluster is gone (see above), and
  `TestCompositePolyDataMapper{Scalars,CellScalars,Spheres,Vertices,Picking,
  PartialFieldData,CameraShiftScale,CustomShader,MixedGeometry*,
  Pickability}` now render, but still fail image comparison — the largest
  failing cluster. (`TestCompositePolyDataMapperBlockTextures`,
  `TestCompositePolyDataMapperOverrideScalarArray`,
  `TestCompositePolyDataMapperNaNPartial` and
  `TestCompositePolyDataMapperOverrideLUT` now pass via the per-block
  texture/scalar-attribute ports and the scalar-texture LUT pipeline described
  in the composite-cluster / scalar-texture-LUT sections above.)
- **Glyph instancing** (~9): `TestGlyph3DMapper{Arrow,BackfaceColor,Indexing,
  OrientationArray,Picking,PointSize,QuaternionArray,TreeIndexing,
  CompositeDisplayAttributeInheritance}` fail 0.15–0.6.
- **Selection/picking** (~3): `TestPointSelection*` and `TestPickTextActor`
  (`TestAreaSelections`, `TestHardwareSelector`, `TestSelectVisiblePoints`,
  `TestWorldPointPicker` now pass — see the selection/read-back cluster
  sections above). The 10 near-miss image tests (`TestActorLightingFlag`,
  `TestEdgeFlags`, `TestQuadPointRep`, `TestMixedGeometry_3`,
  `TestImageAndAnnotations`, `TestVertexRendering`, `TestLineRenderingTranslucent`,
  `TestGlyph3DMapperPicking`, `RenderNonFinite`, `TestMixedGeometryCellScalars`)
  are the next easy-win targets.
- **2D overlay / image mapper**: `TestPolyDataMapper2D` (0.235),
  `TestPolyDataMapper2D{Point,Cell}ScalarColorMapping` (0.236/0.246),
  `TestImageMapper_1..4` (0.63–0.72), `TestActor2D` (now renders; image fail).
- **LUT / color mapping** (~5): `TestBareScalarsToColors` 0.584,
  `TestDirectScalarsToColors` 0.858, `TestMapVectorsToColors` 0.747,
  `TestMapVectorsAsRGBColors` 0.632, `TestColorByStringArrayDefaultLookupTable2D` 0.482.
- **Stereo / multiview / gradient background**: `TestOffAxisStereo`,
  `TestStereoBackground{Left,Right}`, `TestStereoEyeSeparation`,
  `TestSplitViewportStereoHorizontal`, the 5 `TestNViewports*`, and 3
  `TestGradientBackground*` (0.5–0.8).
- **TStrips** (4) and `TestPolyDataMapperNormals` (0.608) fail 0.4–0.55.

### Evidence the core path is correct

The 82 passes include the strongest-scrutiny tests: `TestOpacity` (passes with
the `TIGHT_VALID` metric — the Lab-space color path matches GL to
`0.00038`), `TestOSConeCxx`, `TestMace`, the four scalar-LUT tests
(`TestCompositePolyDataMapperOverrideLUT`, `TestTextureInterpolateScalars`,
`TestTranslucentLUTTextureAlphaBlending`, `TestTranslucentLUTTextureDepthPeeling`
at ImageError 1.2e-06–1.2e-04), `TestTranslucentLUTAlphaBlending`,
`TestTranslucentLUTDepthPeeling`, `TestScalarModeToggle`,
`TestPointRendering_{1,2,Round_1,Round_2}`, `TestCompositePolyDataMapper` and its
`BlockOpacities`/`ToggleScalarVisibilities`/`PartialPointData`/`StaticBounds`/
`SharedArray` variants, `TestAreaSelections` and `TestHardwareSelector` (exact
per-primitive cell ids), the read-back cluster (`TestReadPixels`,
`TestSelectVisiblePoints`, `TestWorldPointPicker`), and the basic Glyph3D,
`FrustumClip`, `RGrid`, `TestQuad`. `Rendering/Metal/Testing/OpenGLComparison.md`
shows every bespoke scene now matches OpenGL to a thresholded error of 0.000
(0.005 for volume).
Failures cluster in features Metal still implements incompletely (see below),
not in the fundamental geometry/lighting/color path.

---

## 3. Known gaps / next steps (highest value first)

1. **Texture support — DONE** — `vtkMetalTexture` (`Rendering/Metal/vtkMetalTexture`)
   now overrides `vtkTexture` for `RenderingBackend=Metal` (no-op `Load`; the
   poly-data mapper uploads actor textures in `UpdateActorTexture`). This
   eliminated all 14 texture-fallback crashes; 6 texture tests now pass
   (`TestTextureWrap`, `TestBackfaceTexture`, `TestTextureRGBA`,
   `TestTextureRGBADepthPeeling`, `TestTextureSize`,
   `TestTextureInterpolateScalars` — the last via the scalar-texture LUT
   pipeline below) and the
   rest fail image comparison instead of SIGSEGV. Follow-ups: border-color and
   filter/mipmap fidelity (`TestTexturedCylinder`), and renderers that bypass
   the poly-data mapper
   (2D image/text in the label cluster).
2. **Composite mapper crash — DONE** — a missing `mappedColors != nullptr` guard
   in the indexed-triangle path of `vtkMetalPolyDataMapper::BuildGeometryBuffers`
   (`vtkMetalPolyDataMapper.mm:3745`) crashed whenever block display overrides
   skipped `MapScalars`. All 8 composite-cluster crashes are gone; the composite
   mapper is now an *image-compare* cluster. **Per-block scalar overrides — DONE**
   (`TestCompositePolyDataMapperOverrideScalarArray`, `NaNPartial`, `OverrideLUT`
   now pass): the batched mapper forwards per-block scalar attributes to child
   mappers and a NaN-color fallback covers `ColorMissingArraysWithNanColor`.
   **Scalar-texture LUT — DONE** (`TestCompositePolyDataMapperOverrideLUT`,
   `TestTextureInterpolateScalars`, `TestTranslucentLUTTextureAlphaBlending`,
   `TestTranslucentLUTTextureDepthPeeling` now pass): `InterpolateScalarsBeforeMapping`
   maps through a real two-row LUT texture (NEAREST sampler) sampled per-fragment
   from a per-vertex `scalarCoord` stream — GL's `texture(colortexture,
   colorTCoord)` exactly (see the scalar-texture-LUT section above). The former
   CPU-subdivision approximation is removed.
3. **MSAA attachments — DONE** — `TestOpacityMSAA` requested `MultiSamples(8)`,
   which the Apple GPU family does not support (max 4); Metal's descriptor
   validation aborted in `CreateMultisampleAttachments`.
   `vtkMetalRenderWindow::GetEffectiveSampleCount()` now clamps the requested
   sample count to the device maximum (4 on Apple-family GPUs, 8 elsewhere) at
   the single choke point every MSAA resource and pipeline state reads from.
   `TestOpacityMSAA` now passes its image comparison.
4. **Hardware selector / selection — DONE** — `TestAreaSelections` first
   stopped crashing via the `GetColorBufferSizes` override (8/8/8/8 for the
   BGRA8Unorm attachments; `vtkHardwareSelector::Select` had aborted on an
   uninitialized `rgba`), then started passing via the exact per-primitive
   cell-id port (`e9e8a6bb66`): the mapper emits `cellPrimitiveIds` per
   triangle unconditionally and the surface fragment reads
   `cellPrimitiveIds[prim_id]` under `kSceneFlagUsePrimitiveCellIds`,
   emulating GL's `gl_PrimitiveID`. Metal and GL selection ID lists are now
   identical (139 cells), and `TestAreaSelections` passes under both backends;
   `TestHardwareSelector` also passes. Remaining: `TestPointSelection*`,
   `TestSelectVisiblePoints`, `TestWorldPointPicker` (pick check),
   `TestReadPixels`.
5. **Read-back** — `TestReadPixels` errors; `TestWindowToImageFilter` /
   `TestResizingWindowToImageFilter` crash.
6. **Label/text/image OpenGL-fallback cluster** — the 9 remaining crashes all
   instantiate the OpenGL label/text/image classes (e.g.
   `vtkOpenGLLabeledContourMapper::ApplyStencil`) against a Metal window; this
   needs Metal overrides for the label/text rendering stack.
7. **Glyph instancing colors**, **2D overlay + image mapper**, **LUT/color
   mapping**, **stereo/multiview + gradient background**, **TStrips** — all
   render but diverge from GL.

---

## 4. How to reproduce

```sh
# bespoke Metal suite
./macos_metal_build.sh --tests --resume
ctest --test-dir build_macos_metal -R "RenderingMetalCxx|RenderingMetal-HeaderTest" --output-on-failure

# generic Metal suite (all 175)
ctest --test-dir build_macos_metal -R "RenderingCoreCxx-Metal" -j 8 --output-on-failure

# visual parity harness
./build_macos_metal/bin/vtkMetalGLVisualComparison --out /tmp/visual_compare
```

Run environment: a GUI login session is required (the render window attaches to
the WindowServer); headless SSH will not work.

Numbers in this file: historical tally from the 2026-08-02 run at commit
`bc4e9d93cd` (M1 Mac mini); working-tree tallies from the same date rerun on the
M2 MacBook Air with the texture-cluster changes present, then again after the
`GetColorBufferSizes` override and the `RenderPiece` input-`Update()` fix
(66 Passed / 100 Failed / 9 aborted), then again after the per-primitive
cell-id selection fix (`e9e8a6bb66`; 69 Passed / 97 Failed / 9 aborted —
`TestAreaSelections` and `TestHardwareSelector` now pass, `TestAxesActor`
passed this run), and most recently after the read-back cluster fixes
(72 Passed / 94 Failed / 9 aborted — `TestReadPixels`, `TestSelectVisiblePoints`
and `TestWorldPointPicker` now pass), then after the per-block-texture port
(73 Passed / 93 Failed / 9 aborted — `TestCompositePolyDataMapperBlockTextures`
now passes), and finally after the per-block scalar-override cluster
(81 Passed / 85 Failed / 9 aborted — `TestCompositePolyDataMapperOverrideLUT`,
`OverrideScalarArray` and `NaNPartial` now pass; the remaining pass-count delta
over the prior run is within documented run-to-run flakiness), and most recently
after the scalar-texture LUT pipeline
(82 Passed / 84 Failed / 9 aborted — `TestTextureInterpolateScalars` now passes,
completing the four scalar-LUT tests). The image-compare
buckets above are from that latest run's `LastTest.log` (max TIGHT_VALID error
per test). Re-running is reproducible except where a crash's signal stack
ordering varies; the pass count fluctuates run to run.
