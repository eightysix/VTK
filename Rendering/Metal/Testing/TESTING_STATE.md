# Metal Testing State

Snapshot of the state of testing for the Metal rendering backend. Historical
run: `metal-ios` branch head `bc4e9d93cd` on this repo's Apple M1 Mac mini
(Macmini9,1, macOS 14.8). Current working-tree run: this repo's Apple M2
MacBook Air (Mac14,2, macOS 15.7.5), after the texture-cluster, composite-mapper,
and selection cell-ID fixes described below. Both arm64 Release builds in
`build_macos_metal` (Ninja, `VTK_MODULE_ENABLE_VTK_RenderingMetal=YES`).

Two test surfaces exist:

1. A bespoke Metal suite under `Rendering/Metal/Testing/` (14 image-baseline
   regression tests + a HeaderTest + a dual-backend visual-parity harness).
   Status: **all 16 pass** (unchanged from the historical run except the 14th
   test `TestMetalImageSliceMapper` added by `3ec24ca9c5`).
2. The generic multi-backend suite in `Rendering/Core/Testing/Cxx/`, which
   registers the same ~175 tests once per backend and was wired up for Metal
     through    object-factory overrides (`--vtk-factory-prefer
       RenderingBackend=Metal`). Historical status: **55 pass / 120 fail (33
        crash)**. Current working-tree status: **133 pass / 39 fail (3 crash)** —
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
     exactly (see the scalar-texture-LUT section below). The glyph3D
     multi-source indexing tests `TestGlyph3DMapperIndexing` and
     `TestGlyph3DMapperTreeIndexing` now pass via the multi-source
     `SourceGeometry`/`SourceInstances` port (see the glyph3D multi-source
     indexing section below), and `TestTexturedBackground` plus the stereo
     backgrounds `TestStereoBackgroundLeft`/`TestStereoBackgroundRight` now pass
     via the textured-background implementation (see the textured-background
     section below). `TestNActorsOneMapper`/`TestNActorsNMappersOneInput` now pass
     via the per-actor edge-color fix (ambient pre-applied before the edge mix;
     see the per-actor edge-color section below), the flat-background
     alpha/dither fix restored `TestReadPixels`/`TestRemoveActors`, and the eight
     sub-viewport `vtkImageMapper` tests (`TestImageMapper_1..4`,
     `TestDirectScalarsToColors`, `TestBareScalarsToColors`,
      `TestMapVectorsAsRGBColors`, `TestMapVectorsToColors`) now pass via the 2D
      image-mapper viewport-WCVC fix (see the 2D image-mapper section below), and
      the four triangle-strip tests
      `TestTStrips{TCoords,NormalsTCoords,NormalsColorsTCoords,ColorsTCoords}` now
      pass via CPU-side strip decomposition plus the property-texture fallback
      (see the triangle-strip section below), and the point-rendering cluster
      (`TestPointRendering{,_Round}_3/4`, `TestVertexRendering`, `TestQuadPointRep`,
      `TestMixedGeometry_3`) now passes via the `HasVerts` gate and the hoisted
      `UpdateLightUniforms` call (see the point-rendering section below). No
      new crashes
      were introduced. The pass count
      fluctuates run to run (run-to-run flakiness).

---

## 1. Bespoke suite (`Rendering/Metal/Testing/`)

### Layout

- `README.md` — how to build with `--tests`, run, add a test, and manage
  baselines via VTK ExternalData (`.sha512` markers + `.ExternalData/SHA512/`
  store).
- `OpenGLComparison.md` — per-scene Metal-vs-OpenGL parity numbers from
  `vtkMetalGLVisualComparison`.
- `Cxx/CMakeLists.txt` — registers the 14 tests in the shared
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

### The 14 regression tests and coverage

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
| `TestMetalImageSliceMapper` | 2D image-slice rendering with GPU picking (added `3ec24ca9c5`) |
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
100% tests passed, 0 tests failed out of 16   (HeaderTest + 14 regression + GLVisualComparison)
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
175 tests:  133 Passed  39 Failed (incl. image/pick fails)  3 "Subprocess aborted"
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
Run after the 2D-overlay-mapper fix: 83 Passed / 83 Failed / 9 aborted —
`TestActor2D` now passes via the `vtkMetalPolyDataMapper2D` fix below
(`RenderOverlay` never called `GetInputAlgorithm()->Update()`, so 2D geometry
was not computed and the mapper returned early, and a stray Y-flip in the WCVC
orthographic matrix inverted the quad). The +1 delta over the previous run is
exactly that test; the 9 aborts are unchanged.
Run after the wireframe-lines fix (this run, 2026-08-03): 92 Passed / 80 Failed
/ 3 aborted out of 175 (analyzed with `analyze_metal_ctest_log.py` from a single
`ctest -R "RenderingCoreCxx-Metal" -j 8` run) — `TestWireframe` now passes via
the unlit-1px-line path below (`fragment_main_line` + `kSceneFlagLinesUnlit`),
which reproduces GL's NoLighting decision for line primitives without point
normals (was a mid-bucket 0.239 image fail; now white wires). The near-miss and
gross buckets are otherwise unchanged; the abort-count drop from 9 to 3 this run
(`TestFollowerPicking`, `TestInteractorStyleImageProperty`,
`TestTranslucentImageActor{AlphaBlending,DepthPeeling}`, `TestWindowToImageFilter`
ran clean and `TestResizingWindowToImageFilter` became a plain image fail) is the
documented run-to-run flakiness of the label/text/image abort class.
Run after the glyph3D multi-source indexing fix (this run, 2026-08-03):
96 Passed / 76 Failed / 3 aborted out of 175 (analyzed with
`analyze_metal_ctest_log.py` from a single `ctest -R "RenderingCoreCxx-Metal" -j 8`
run) — `TestGlyph3DMapperIndexing` and `TestGlyph3DMapperTreeIndexing` now pass
with a TIGHT_VALID ImageError of 0.0014 (was a mid-bucket 0.155 fail), the first
time they have passed; see the glyph3D multi-source indexing section below. The
remaining glyph image fails in this run (`TestGlyph3DMapperPicking` 0.080,
`TestGlyph3DMapper` 0.108, `TestGlyph3DMapperCompositeDisplayAttributeInheritance`
0.224, `TestGlyph3DMapperBackfaceColor` 0.291, `TestGlyph3DMapperPointSize` 0.460)
are the pre-existing non-indexing diff-only class, unchanged by this work. The
+4 pass delta over the previous run is exactly the two indexing tests plus the
two non-glyph gross-bucket deltas below; the 3 aborts are unchanged
(`TestLabeledContourMapper`, `TestLabeledContourMapperNoLabels`,
`TestLabeledContourMapperWithActorMatrix`). The regression check against the
documented passing cluster reports none.
Run after the textured-background fix (this run, 2026-08-03): 99 Passed /
73 Failed / 3 aborted out of 175 (analyzed with `analyze_metal_ctest_log.py` from
a single `ctest -R "RenderingCoreCxx-Metal" -j 8` run) — `TestTexturedBackground`
and the stereo backgrounds `TestStereoBackgroundLeft`/`TestStereoBackgroundRight`
now pass (ImageErrors 0.0104 / 0.0132, down from a gross-bucket ~0.8865), the
first time they have passed; see the textured-background section below. The +3
pass delta over the previous run is exactly those three tests (all formerly
gross-bucket fails); the 3 aborts are unchanged (`TestLabeledContourMapper`,
`TestLabeledContourMapperNoLabels`, `TestLabeledContourMapperWithActorMatrix`).
The regression check against the documented passing cluster reports none.
Run after the edge-color and flat-background fixes (this run, 2026-08-03):
111 Passed / 61 Failed / 3 aborted out of 175 (analyzed with
`analyze_metal_ctest_log.py` from a single `ctest -R "RenderingCoreCxx-Metal" -j 8`
run) — `TestNActorsOneMapper` and `TestNActorsNMappersOneInput` now pass
(TIGHT_VALID ImageErrors ~1.2e-05, were mid-bucket fails; see the per-actor
edge-color section below), and `TestReadPixels` + `TestRemoveActors` are restored
to passing (see the flat-background alpha/dither section below). The +12 pass
delta over the previous run is those four tests plus `TestNViewportsNActorsOneMapper`,
`TestNViewportsNActorsNMappersOneInput`, `TestNViewportsNActorsNMappersNInputs`,
`TestSurfacePlusEdges`, `TestTexturedCylinder`, `TestEdgeThickness`, `TestAxesActor`
(finally gone from the failure set; the multi-viewport NActors family had been
documented at gross 0.85-0.87 before the uncommitted multi-viewport background
work) and the read-back/wireframe classes. `TestReadPixels`/`TestRemoveActors` are
not in the analyzer's hardcoded regression set (`analyze_metal_ctest_log.py`
lines 113-123), so the script cannot flag them; worth adding. The 3 aborts are
unchanged (`TestLabeledContourMapper`, `TestLabeledContourMapperNoLabels`,
`TestLabeledContourMapperWithActorMatrix`). The regression check against the
documented passing cluster reports none.
Run after the 2D image-mapper viewport fix (this run, 2026-08-03): 119 Passed /
53 Failed / 3 aborted out of 175 (analyzed with `analyze_metal_ctest_log.py` from a
single `ctest -R "RenderingCoreCxx-Metal" -j 8` run; failures exported with
`export_image_compare.sh`) — `TestImageMapper_1..4` and the four scalar-LUT
image-mapper tests `TestDirectScalarsToColors`, `TestBareScalarsToColors`,
`TestMapVectorsAsRGBColors`, `TestMapVectorsToColors` now pass (TIGHT_VALID
ImageErrors ~1e-07-1e-06, all were gross- or mid-bucket fails), the first time
any of the eight have passed; see the 2D image-mapper viewport-WCVC section
below. The +8 pass delta over the previous run is exactly those eight tests; the
3 aborts are unchanged (`TestLabeledContourMapper`,
`TestLabeledContourMapperNoLabels`, `TestLabeledContourMapperWithActorMatrix`).
The regression check against the documented passing cluster reports none.
Run after the triangle-strip support fix (this run, 2026-08-03): 123 Passed /
49 Failed / 3 aborted out of 175 (analyzed with `analyze_metal_ctest_log.py`
from a single `ctest -R "RenderingCoreCxx-Metal" -j 8` run; failures exported
with `export_image_compare.sh`) — the four `TestTStrips*` tests
(`TestTStripsColorsTCoords`, `TestTStripsNormalsColorsTCoords`,
`TestTStripsNormalsTCoords`, `TestTStripsTCoords`) now pass, the first time any
have passed; they were gross/mid-bucket fails (0.4035/0.5062) that rendered only
the background because the mapper dropped `GetStrips()` geometry (see the
triangle-strip section below). The +4 pass delta over the previous run is exactly
those four tests; the 3 aborts are unchanged (`TestLabeledContourMapper`,
`TestLabeledContourMapperNoLabels`, `TestLabeledContourMapperWithActorMatrix`).
The regression check against the documented passing cluster reports none.
Run after the point-rendering lighting fix (this run, 2026-08-03): 130 Passed /
42 Failed / 3 aborted out of 175 (analyzed with `analyze_metal_ctest_log.py`
from a single `ctest -R "RenderingCoreCxx-Metal" -j 8` run; failures exported
with `export_image_compare.sh`) — the four point tests `TestPointRendering_3/_4`
and `TestPointRenderingRound_3/_4` plus the vertex-visibility tests
`TestVertexRendering`, `TestQuadPointRep` and `TestMixedGeometry_3` now pass
(ImageErrors 0.069-0.38 were mid/near-miss bucket fails; see the
point-rendering section below). The +7 pass delta over the previous run is
exactly those seven tests; the 3 aborts are unchanged
(`TestLabeledContourMapper`, `TestLabeledContourMapperNoLabels`,
`TestLabeledContourMapperWithActorMatrix`). The regression check against the
documented passing cluster reports none.
Run after the 2D overlay depth-ordering + text-texture fix (this run,
2026-08-03): 133 Passed / 39 Failed / 3 aborted out of 175 (analyzed with
`analyze_metal_ctest_log.py` from a single `ctest -R "RenderingCoreCxx-Metal" -j 8`
run; failures exported with `export_image_compare.sh`) — `TestImageAndAnnotations`
now passes (TIGHT_VALID 0.0461, was a 0.0607 near-miss fail), the first time it
has passed, and the two texture tests `TestActor2DTextures` (was a gross 0.7862
fail, now 3.9e-08) and `TestBackfaceCulling` (was a mid 0.1016 fail, now 0.0091)
also pass, the first time either has passed; see the 2D overlay depth-ordering
and text-texture section below. The +3 pass delta over the previous run is
exactly those three tests; the 3 aborts are unchanged (`TestLabeledContourMapper`,
`TestLabeledContourMapperNoLabels`, `TestLabeledContourMapperWithActorMatrix`).
The regression check against the documented passing cluster reports none.

### The point-rendering cluster is fixed

`TestPointRendering_3/_4` and `TestPointRenderingRound_3/_4` (round-point
rendering with point sizes 3.0/7.0) rendered entirely black, and the
vertex-visibility tests `TestVertexRendering`, `TestQuadPointRep` and
`TestMixedGeometry_3` were near-miss image fails, all from the same root cause:
`vtkMetalPolyDataMapper::RenderPiece` only created the light-uniform buffer in
the `needSurfacePipelines` branch, so a points-only/vertex-cell polydata (no
triangles or lines) had no `LightUniformBuffer` and `fragment_point_shaped_main`
read zero bytes from `lights [[buffer(1)]]` — the Phong fragment output black
points even though the point geometry and draw were correct.

The fix (`vtkMetalPolyDataMapper.mm`): the mapper now tracks vertex cells via a
`HasVerts` flag (set in `BuildGeometryBuffers` from `GetVerts()`, reset in
`ReleaseGeometryBuffers`), the point-representation gate becomes
`(representation == VTK_POINTS || HasVerts)`, and `UpdateLightUniforms` is
hoisted out of both pipeline branches so the light uniforms are created/refreshed
once per frame whenever any geometry (surface, line, or point) is drawn.
`TestPointRendering_1..4`, `TestPointRenderingRound_1..4`, `TestVertexRendering`,
`TestQuadPointRep` and `TestMixedGeometry_1..3` all pass. The remaining
`TestPointSelection`/`TestPointSelectionWithCellData` failures are separate
pick-check gaps (point field-association selection is not yet implemented).

### The per-actor edge-color cluster is fixed

`TestNActorsOneMapper` and `TestNActorsNMappersOneInput` painted black edges for
any actor whose property ambient coefficient was below 1.0 (the default 0). The
Metal flat-edge branch mixed `r.ambient` toward the full edge color, but the final
output was multiplied by the property ambient coefficient (`m.ambientColor.w`,
default 0), so edges went black. GL avoids this by pre-applying the intensity
then mixing toward the full edge color (`vtkOpenGLPolyDataMapper.cxx:713`).
Fix in `Rendering/Metal/Shaders/MetalShaders.metal`: pre-apply
`r.ambient = m.ambientColor.w * r.ambient` before `applySurfaceEdges`; the
function now takes an `ambientIntensity` parameter, the flat branch mixes toward
the full edge color `ec`, and the tube branch toward `ambientIntensity * ec`
(applied at all three call sites: `evaluateSurfaceFragment`,
`fragment_main_oit`, `fragment_peel`). Both NActors tests now pass with
TIGHT_VALID ImageErrors ~1.2e-05, matching the documented passing cluster;
fresh render captures show 0 black-edge pixels and all sampled edges exact-match
the baseline.

### The flat-background alpha and dither cluster is fixed

`TestReadPixels` (read-back of a flat-color background) regressed when the
uncommitted multi-viewport/gradient work made the flat background paint as a
quad with a hardcoded alpha of 1.0; GL paints the flat background via `glClear`
with `BackgroundAlpha` (default 0.0), so `TestReadPixels.cxx:60`
(`ucharTuple[3] == 0`) failed. And `TestRemoveActors` (read-back after removing
all actors, expecting pure white) failed with one `Unexpected pixel value 254`
because the flat background also ran the gradient shader, where
`DitherGradient` (default true, `vtkViewport.cxx:35`) added +/-0.5/255 noise;
GL only dithers the actual gradient overlay, never the flat clear.
Fixes in `vtkMetalRenderer.mm`: `fragment_gradient_background` now emits
`u.stopColors[0].a` instead of hardcoded 1.0, and the renderer sets stop-color
alpha to `GetBackgroundAlpha()` in flat mode (1.0 in gradient mode); dither is
now enabled only when `GetGradientBackground() && GetDitherGradient()`. Both
`TestReadPixels` and `TestRemoveActors` pass again; `TestGradientBackground*`
results are unchanged (gradient mode still dithers, matching GL).

### The glyph3D multi-source indexing cluster is fixed

`TestGlyph3DMapperIndexing` and `TestGlyph3DMapperTreeIndexing` (multi-source
glyphing via a `vtkMultiBlockDataSet` or a `vtkDataObjectTree`) crashed in
`GetSourceIndexArray` and were subsequently disabled. The causes were (a) the
`treeIndexing` overload of `GetSourceIndexArray` iterated `vtkDataObjectTree`
children without `GetChildDataObject`, so indexing sources beyond the first
were never selected and points for later sources were dropped, and (b) even with
indexing working, glyph3D source selection was a single global map
(`getSourceGlyphsBySource` in OpenGL), so a multi-source input picked only the
first source's geometry. `vtkMetalGlyph3DMapper` now mirrors OpenGL: per-source
`SourceGeometry`/`SourceInstances` (`std::vector<std::unique_ptr<...>>`),
source counting via `GetNumberOfInputConnections(1)` (or direct tree children
when `UseSourceTableTree`), per-point source selection from the indexed
`GetSourceIndexArray` clamped to the source count, per-source instance staging
(transforms, normals padded to float3x3, colors, pick ids) and a per-source draw
pass, with `kSceneFlagGlyphHasNormals` (bit 15) asserted per-draw. Both tests
now pass stably with ImageError 0.0014.

### The textured-background cluster is fixed

`TestTexturedBackground` and the stereo pair `TestStereoBackgroundLeft`/
`TestStereoBackgroundRight` (renderers using `SetBackgroundTexture` with
`SetViewport`/stereo eye views) failed image comparison at a gross ~0.8865.
The Metal renderer drew only the gradient background and never the textured
one. `vtkMetalRenderer` now mirrors `vtkOpenGLRenderer`:
`GetCurrentTexturedBackground()` (new, via a `vtkMetalRendererInternals` pimpl)
returns the effect only when the texture is set, the viewport matches, and the
stereo eye matches OpenGL's selection rules (left eye / both eyes), and a cached
`MTLTexture` is uploaded from the image data once per frame (kept in the pimpl
so it survives renderer re-creation). The background draw now uses a new
`fragment_textured_background` shader (with the depth-buffer-disabled blend /
clip-space quad pipeline from the gradient path). All three tests now pass with
TIGHT_VALID ImageErrors of 0.0104 / 0.0132 / 0.0104 (was 0.8865 gross).


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
`TestActor2DTextures`, `TestBackfaceCulling` and `TestImageAndAnnotations` have
since left this list — the first two via the textured-2D path and the last via
the overlay depth-ordering (see the 2D overlay depth-ordering and text-texture
section below).

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

### The triangle-strip cluster is fixed

All four `TestTStrips*` tests (`TestTStripsColorsTCoords`,
`TestTStripsNormalsColorsTCoords`, `TestTStripsNormalsTCoords`,
`TestTStripsTCoords`) rendered only the turquoise background because the Metal
poly-data mapper read `GetPolys()`/`GetLines()`/`GetVerts()` but never
`GetStrips()`: the `vtkStripper` output geometry was silently dropped, so the
beach-textured plane never drew (TIGHT_VALID 0.4035/0.5062).

`vtkMetalPolyDataMapper` now treats strips as first-class geometry, mirroring
GL's CPU-side decomposition in `vtkOpenGLIndexBufferObject::AppendStripIndexBuffer`
(`vtkOpenGLIndexBufferObject.cxx:409`):

- The surface path decomposes each strip with GL's winding
  `tri_j = (v_j, v_{j+1+j%2}, v_{j+1+(j+1)%2})`, routed through the same
  `emitSurfaceTriangle` helper as polygon fans (shared indexed dedup when
  normals/conditions allow, non-indexed 3-vertices-per-triangle otherwise, the
  per-primitive exact `cellPrimitiveIds` port, and the `useCellTexture` per-cell
  color port).
- The wireframe path emits GL's `AppendStripIndexBuffer(wireframe=true)` edge
  set — `(v0,v1)` then the two outer edges `(v_j,v_{j+2})`, `(v_{j+1},v_{j+2})`
  of each constituent triangle — through the shared wireframe vertex-dedup map.
- Strip cells are first-class in the polydata cell array: the per-primitive cell
  ids (cell colors, pick/ID pass) start at `polyCellOffset + numPolys`, and the
  legacy edge-overlay (`UseLegacyEdgeOverlay`) extracts each strip's boundary
  edges and emits edge tubes with the strip's absolute cell id for miter joins.
- The GPU-tessellation path (`useGPUTess`) now falls back to the CPU path
  whenever strips are present (`!hasStrips`), since the tess kernel only fans
  polys; polys-only meshes are unaffected.
- `vtkMetalPolyDataMapper2D` applies the same GL-winding decomposition to
  `GetStrips()`.

A second gap: the `TestTStrips*` suite sets its beach texture as a *named
property texture* (`actor->GetProperty()->SetTexture("mytexture", texture)`),
not `actor->SetTexture()`, so `UpdateActorTexture` saw no texture and the plane
rendered white. The mapper now falls back to the first non-PBR-slot named
property texture (skipping `albedoTex`/`normalTex`/`materialTex`/`brdfTex`/
`emissiveTex`/`anisotropyTex`/`coatNormalTex`/`colortexture`, which GL's BRDF
consumes rather than the diffuse sampler), mirroring how GL routes every bound
2D texture through the diffuse `tcolor` path.

All four tests now pass with the beach texture visible (TIGHT_VALID well under
0.05). Regression check: the strips change only re-routes geometry that was
previously dropped; the polys-only path produces identical vertex streams, and
the full-suite fail count dropped exactly by the four strip tests with no new
failures.

### Image-compare failures

Current run (2026-08-03, overlay-depth/texture run): 42 failed = 35 image-compare
(TIGHT_VALID >= 0.05) + 1 below-threshold pick-check + 3 non-image + 3 aborts.
Buckets by
max `vtkTesting` TIGHT_VALID error per test (threshold 0.05):

| Bucket | Range | Count | Examples |
|--------|-------|-------|----------|
| near-miss | 0.05 – 0.1 | 6 | `TestActorLightingFlag` 0.0513, `TestPolyDataMapper2D` 0.0664, `TestEdgeFlags` 0.0681, `TestLineRenderingTranslucent` 0.0790, `TestGlyph3DMapperPicking` 0.0800, `RenderNonFinite` 0.0870 |
| mid | 0.1 – 0.5 | 24 | `TestGlyph3DMapper` 0.1082, `TestMixedGeometryCellScalars` 0.1373, `TestCompositePolyDataMapperSpheres` 0.1499, `TestPolyDataMapperClipPlanes` 0.1526, `TestTransformCoordinateUseDouble` 0.1635, `TestCompositePolyDataMapperPicking` 0.1712, `TestGlyph3DMapperCompositeDisplayAttributeInheritance` 0.2241, `TestCoincident` 0.2343, `TestRenderLinesAsTubesOrthoCamera` 0.2349, `TestRenderLinesAsTubes` 0.2350, `TestStereoEyeSeparation` 0.2584, `TestCompositePolyDataMapperPartialFieldData` 0.2622, `TestPolyDataMapperNormals` 0.2698, `TestPolyDataMapper2DPointScalarColorMapping` 0.2861, `TestCompositePolyDataMapperVertices` 0.2891, `TestCompositePolyDataMapperCustomShader` 0.2897, `TestGlyph3DMapperBackfaceColor` 0.2914, `TestPolyDataMapper2DCellScalarColorMapping` 0.2943, `TestResetCameraScreenSpace` 0.3438, `TestGradientBackground` 0.3449, `TestCompositePolyDataMapperCameraShiftScale` 0.3601, `TestResizingWindowToImageFilter` 0.4130, `TestGlyph3DMapperPointSize` 0.4596, `TestColorByStringArrayDefaultLookupTable2D` 0.4821 |
| gross | >= 0.5 | 5 | `TestGradientBackgroundWithTiledViewport` 0.5063, `TestGradientBackgroundWithTiledViewports` 0.5856, `TestOffAxisStereo` 0.5921, `TestTilingCxx` 0.6159, `TestSplitViewportStereoHorizontal` 0.6816 |

(`TestNActors{OneMapper,NMappersOneInput}` left the mid bucket via the
per-actor edge-color fix above (passing at ~1.2e-05), and
`TestNViewportsNActors*` + `TestSurfacePlusEdges` + `TestTexturedCylinder` +
`TestEdgeThickness` + `TestAxesActor` left via the uncommitted multi-viewport/
per-actor-edge-uniform work (the NViewports family was documented at gross
0.85-0.87). This run the mid bucket dropped from 31 to 30 and gross from 15 to 8:
`TestImageMapper_1..4`, `TestBareScalarsToColors`, `TestMapVectorsAsRGBColors`
and `TestMapVectorsToColors` (all gross) plus `TestDirectScalarsToColors` (mid)
left via the 2D image-mapper viewport-WCVC fix below — the first time any of the
eight have passed (ImageErrors ~1e-07-1e-06).) This run the mid bucket dropped
from 30 to 28 and gross from 8 to 6: the four `TestTStrips*` tests (two at 0.4035
mid, two at 0.5062 gross) left via the triangle-strip section below — the first
time any have passed.) This run the near-miss bucket dropped from 11 to 7 and
mid from 28 to 25: the four point tests
(`TestPointRendering{,_Round}_3/4`), `TestVertexRendering`, `TestQuadPointRep`
and `TestMixedGeometry_3` left via the point-rendering section below — the first
time any of those seven have passed (ImageErrors 0.069-0.38).) The below-threshold
failure is `TestCompositePolyDataMapperPickability` (ImageError 0.0155 but fails
its own pickability check). The 3 non-image failures: `TestPointSelection`,
`TestPointSelectionWithCellData` (selection returns even-only point ids,
pick-check fails) and `TestPickTextActor` (pick check). `TestReadPixels`,
`TestSelectVisiblePoints` and `TestWorldPointPicker` left this set via the
read-back cluster above, the four scalar-LUT tests left it via the
scalar-texture-LUT section above, and the four `TestTStrips*` tests left it via
the triangle-strip section below.) This run the near-miss bucket dropped from 7
to 6 and gross from 6 to 5: `TestImageAndAnnotations` (near-miss 0.0607, now
passing at 0.0461), `TestActor2DTextures` (gross 0.7862, now passing at
3.9e-08) and `TestBackfaceCulling` (mid 0.1016, now passing at 0.0091) all left
via the 2D overlay depth-ordering and text-texture section below — the first
time any of the three have passed.

### Crashes (3; all pre-existing classes, none from the texture or composite clusters)

Signal-level crashes are reported as `Subprocess aborted` with a `Caught
SIGSEGV` line but no backtrace in the ctest log, so crash causes were
re-derived from the earlier run's full signal stacks. Current-run attribution:

| Cause | Count | Tests |
|-------|-------|-------|
| SIGSEGV with no backtrace; prior stack analysis attributes these to the OpenGL factory fallback in label/text/image rendering | 3 | `TestLabeledContourMapper`, `TestLabeledContourMapperNoLabels`, `TestLabeledContourMapperWithActorMatrix` |

The 14-test OpenGL-texture-fallback crash row from the historical tally is gone
(fixed by `vtkMetalTexture`, see above), the 8-test composite-mapper
`BuildGeometryBuffers` row is gone (see above), the `TestOpacityMSAA` MSAA
row is gone (see below), and the `TestAreaSelections` row is gone (fixed by
the `GetColorBufferSizes` override, see the selection-cluster section above).
The 6-test label/text/image class (`TestFollowerPicking`,
`TestInteractorStyleImageProperty`, `TestResizingWindowToImageFilter`,
`TestTranslucentImageActor{AlphaBlending,DepthPeeling}`, `TestWindowToImageFilter`)
that rounded out the 9 in the 2026-08-02 run has run clean since the
2026-08-03 wireframe-lines run and stays within the documented run-to-run
flakiness. The 3 remaining crashes are genuine Metal bugs, all the label/text
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

### The 2D overlay mapper is fixed

`TestActor2D` (a 3D plane with scalar-texture LUT plus a yellow
`vtkPolyDataMapper2D` quad at normalized viewport coords) now passes
(ImageError 0.027 vs. 0.05 threshold; was 0.276). Two bugs in
`vtkMetalPolyDataMapper2D::RenderOverlay`:

- It never called `this->GetInputAlgorithm()->Update()`, so the 2D source
  pipeline (e.g. `planeSource2`) never executed and the mapper returned early
  with 0 points. OpenGL's `vtkOpenGLPolyDataMapper2D.cxx` calls `Update()`
  before reading `GetInput()`; the Metal override now does the same.
- The WCVC orthographic matrix negated the viewport Y coords
  (`wcvc[1]/[5]/[9]/[13]`), mirroring an OpenGL-style flip. Metal's NDC
  `+y = framebuffer top` already matches VTK's y-up viewport, and the renderer's
  `ReadColorCopyData` emits rows bottom-up, so the flip inverted the quad. It is
  removed, matching the OpenGL matrix exactly.

Remaining 2D fidelity gaps (all pre-existing, now documented values): the 2D
vertex shader has no `[[point_size]]` output (`TestPolyDataMapper2D` 0.066) and
2D scalar color mapping is ignored (`TestPolyDataMapper2D{Point,Cell}
ScalarColorMapping` 0.286/0.294). The textured-2D gap is closed by the
2D overlay depth-ordering and text-texture section below
(`TestActor2DTextures` now passes at 3.9e-08).

### The 2D overlay depth-ordering and text-texture is fixed

Two Metal gaps left `TestImageAndAnnotations` (overlay text on top of images
with `DisplayLocation`-dependent ordering) failing at 0.0607, `TestActor2DTextures`
at a gross 0.7862, and `TestBackfaceCulling` at 0.1016. Both gaps live in the
2D overlay path (`vtkMetalPolyDataMapper2D` / `vtkMetalImageMapper` +
`vtkMetalTexture`):

- **No overlay depth ordering.** The Metal overlay pass ran with
  `MTLCompareFunctionAlways` / depth-write-off, so the GL semantics encoded in
  `vtkOpenGLPolyDataMapper2D::SetCameraShaderParameters` (foreground
  `DisplayLocation` props at wcvc Z = −1, background at +1, drawn into a
  depth-testing overlay pass) had no Metal equivalent: background images
  overwrote foreground text and foreground images covered text at equal depth.
  Both mappers now encode `DisplayLocation` into the wcvc Z translate (Metal's
  clip/NDC z range is [0,1], so foreground = 0, background = 1) and the overlay
  depth-stencil state is `LessEqual` with depth write on. Equal-depth foreground
  props still resolve by render order, matching GL.
- **No text texture path.** `vtkTextMapper`/`vtkTextActor` route their glyph
  raster through `vtkTexture` + the actor's `GENERAL_TEXTURE_UNIT` property key.
  The Metal `vtkTexture` was a no-op factory override and the 2D mapper never
  sampled a texture, so text quads rendered flat gray and textured 2D actors
  ignored their image. `vtkMetalTexture::Load` now uploads the input image to an
  `id<MTLTexture>` and registers it with the render window's per-unit registry
  (`vtkMetalRenderWindow::SetBoundTexture`/`GetBoundTexture`, mirroring GL's
  per-unit binding state; released on window teardown via
  `ReleaseBoundTextures`). `vtkMetalPolyDataMapper2D` builds an interleaved
  position+texcoord buffer and draws it through a new textured pipeline
  (`vertex_2d_image_main` + `fragment_2d_text_main`) that samples the bound
  texture and multiplies by the actor's color/opacity, exactly like
  `vtkPolyData2DFS.glsl`.
- **All textures collided on unit 0.** Core `vtkTexture::GetTextureUnit()`
  returns 0 unconditionally; the GL backend overrides it with a per-texture
  unit. Without the override every `vtkMetalTexture` registered under unit 0,
  so after the first frame's cache hit (`Registered && CachedMTime ==
  imageMTime` skips re-registration) every text quad resolved the *last*
  uploaded texture — the corner-annotation labels all rendered the same string
  ("foreground/transparent"). `vtkMetalTexture` now overrides `GetTextureUnit()`
  to allocate a unique, stable unit per instance (a static atomic counter),
  mirroring `vtkOpenGLTexture`.
- **Glyph gaps wrote depth.** `fragment_2d_text_main` now `discard_fragment()`s
  when the multiplied alpha is <= 0 (matching `vtkPolyData2DFS.glsl`'s
  `if (gl_FragData[0].a <= 0.0) discard;`), so a text quad's transparent
  bounding box no longer occludes background props in the depth-testing overlay
  pass.

`TestImageAndAnnotations` now passes at TIGHT_VALID 0.0461 (first time),
`TestActor2DTextures` at 3.9e-08 and `TestBackfaceCulling` at 0.0091 (both
first time). The remaining overlay-text fidelity gap is pure text anti-aliasing
(the residual ~0.046 is glyph edge AA vs the GL FreeType raster), unchanged by
this work. `TestPickTextActor` (pick check) remains a separate gap.

### The 2D image mapper viewport WCVC is fixed

All eight `vtkImageMapper` sub-viewport tests — `TestImageMapper_1..4`,
`TestDirectScalarsToColors`, `TestBareScalarsToColors`,
`TestMapVectorsAsRGBColors`, `TestMapVectorsToColors` — failed image comparison
with each partition showing a crop/zoomed-in slice of the baseline (ImageErrors
0.44-0.75). `vtkMetalImageMapper::DrawPixels` (`vtkMetalImageMapper.mm`) builds
the textured quad in renderer-local viewport-pixel coordinates (from
`actor->GetActualPositionCoordinate()->GetComputedViewportValue`, i.e. the range
`[0, viewport->GetSize()]`), but its wcvc orthographic matrix mapped that space to
Metal NDC using `viewport->GetViewport()` fractions multiplied by
`viewport->GetSize()` — and `GetSize()` already returns the renderer's pixel size
for a sub-viewport. The viewport fraction was thus applied twice, so the NDC
mapping compressed to `fraction * size` and only the bottom-left ~15% x ~39% of
each image was magnified to fill the partition. OpenGL maps `[0, size]` to NDC
directly (`vtkOpenGLPolyDataMapper2D.cxx`, ortho from `viewport->GetSize()` with
no fraction scaling). The fix builds the ortho from the renderer's own pixel size
only (`vpX = vpY = 0`, `vpW = size[0]`, `vpH = size[1]`). All eight tests now pass
(TIGHT_VALID ImageErrors ~1e-07-1e-06). `vtkMetalPolyDataMapper2D` shares the
same wcvc construction but is only masked because its tests use full-window
renderers; it should get the same size-only correction if sub-viewport 2D tests
are added.

### Wireframe / 1px lines render unlit like GL

`TestWireframe` (a single cone at `SetRepresentationToWireframe`, line width 1)
failed with pure-black wires (TIGHT_VALID 0.239): the Metal 1px-line pipeline
always ran the full Phong fragment with the default headlight, and the cone's
surface normals are near-silhouette so `df = max(N.z, eps)` collapsed to ~0.

`vtkOpenGLPolyDataMapper::GetNeedToRebuildShaders` computes, for line
primitives, `needLighting = (interpolation != VTK_FLAT && haveNormals)`; a
vtkConeSource has no point normals, so GL takes the NoLighting path and emits
the flat vertex color (`vtkGLSLModLight` case 0:
`gl_FragData[0] = vec4(ambientColor + diffuseColor, opacity)`), i.e. white
lines. The fix (`fragment_main_line` + `kSceneFlagLinesUnlit`, bit 14):

- The mapper sets `kSceneFlagLinesUnlit` per actor in `RenderPiece` when
  `!(prop->GetLighting() && prop->GetInterpolation() != VTK_FLAT && input-has-
  point-normals)` — GL's exact line decision.
- The `LinePipeline` now uses a dedicated `fragment_main_line` entry that passes
  that flag into `evaluateSurfaceFragment`; when set it skips
  `computePhongLighting` and outputs `ambientIntensity*ambientColor +
  diffuseIntensity*diffuseColor` (the flat vertex/material color), matching GL.
- Surface draws still use `fragment_main`/`fragment_main_nodepth` with
  `unlitLines = false`, so an actor without normals keeps full triangle
  lighting (GL lights tris regardless of normals), and thick-line/tube
  pipelines (`shadeLineFragment`) are untouched — GL always lights those.

`TestWireframe` now passes with white wires (1,360 white pixels vs the GL
baseline's 1,388 — a 28-pixel anti-aliasing delta, well under threshold). The
other line-image tests (`TestLineRenderingTranslucent`
0.079, `TestMixedGeometryCellScalars` 0.137,
`TestRenderLinesAsTubes` 0.326, `TestRenderLinesAsTubesOrthoCamera` 0.326) are
unchanged — their errors are separate fidelity gaps (translucent lines,
cell-scalar lines, thick-line/tube shading). `TestVertexRendering` 0.072 and
`TestMixedGeometry_3` 0.070 left the set via the point-rendering fix above
(vertex dots), and `TestSurfacePlusEdges` + `TestEdgeThickness` (edge
overlay) left via the per-actor edge-color fix above.

### Theme clusters in the 42 failures

- **Textures** (~13): every `TestTexture*`, `TestBackfaceTexture`,
  `TestTilingCxx`, `TestActor2DTextures` — historically
  crashed on the OpenGL fallback; now render, with 9 passing
  (`TestTextureWrap`, `TestBackfaceTexture`, `TestTextureRGBA`,
  `TestTextureRGBADepthPeeling`, `TestTextureSize`,
  `TestTextureInterpolateScalars`, `TestTexturedCylinder`,
  `TestActor2DTextures`, `TestBackfaceCulling`) and the rest
  failing image comparison on texture-feature fidelity (filter/wrap/interpolation
  edge cases, tiling).
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
- **Glyph instancing** (~8): `TestGlyph3DMapper{Arrow,BackfaceColor,Indexing,
  OrientationArray,Picking,PointSize,QuaternionArray,
  CompositeDisplayAttributeInheritance}` fail 0.15–0.6
  (`TestGlyph3DMapperTreeIndexing` passes).
- **Selection/picking** (~3): `TestPointSelection*` and `TestPickTextActor`
  (`TestAreaSelections`, `TestHardwareSelector`, `TestSelectVisiblePoints`,
  `TestWorldPointPicker`, `TestReadPixels`, `TestRemoveActors` now pass — see the
  selection/read-back cluster sections above). The 6 near-miss image tests
  (`TestActorLightingFlag`,
  `TestEdgeFlags`,
  `TestLineRenderingTranslucent`,
  `TestGlyph3DMapperPicking`, `RenderNonFinite`, `TestPolyDataMapper2D`)
  are the next easy-win targets (`TestImageAndAnnotations` left via the
  overlay depth/texture fix above).
- **Point rendering** — DONE: the four point tests `TestPointRendering_3/_4` +
  `TestPointRenderingRound_3/_4` plus the vertex-visibility tests
  `TestVertexRendering`, `TestQuadPointRep` and `TestMixedGeometry_3` now pass
  via the point-lighting fix (see the point-rendering section above).
- **2D overlay / image mapper**: `TestPolyDataMapper2D` (0.066; point size not
  in the 2D shader) and `TestPolyDataMapper2D{Point,Cell}ScalarColorMapping`
  (0.286/0.294; 2D scalar colors ignored). `TestActor2D` now passes via the
  2D-overlay section above, `TestImageMapper_1..4` now pass via the 2D
  image-mapper viewport-WCVC fix (see the 2D image-mapper section above), and
  `TestActor2DTextures` now passes via the 2D overlay depth-ordering and
  text-texture fix (see that section above).
- **LUT / color mapping** (~1): `TestColorByStringArrayDefaultLookupTable2D` 0.482.
  `TestBareScalarsToColors`, `TestDirectScalarsToColors`, `TestMapVectorsToColors`
  and `TestMapVectorsAsRGBColors` now pass via the 2D image-mapper viewport-WCVC
  fix (all four render `vtkImageMapper` in sub-viewports).
- **Stereo / multiview / gradient background**: `TestStereoBackground{Left,Right}`
  now pass via the textured-background implementation (see the textured-background
  section below), as do the 3 `TestNViewportsNActors*` multiview tests via the
  multi-viewport background work (see the flat-background section above;
  `TestNViewportsOneActor` passes too). Still failing: `TestOffAxisStereo`,
  `TestStereoEyeSeparation`,
  `TestSplitViewportStereoHorizontal`, and 3
  `TestGradientBackground*` (0.34–0.59).
- **Triangle strips** — DONE: the four `TestTStrips*` tests now pass via CPU-side
  strip decomposition in `vtkMetalPolyDataMapper`/`vtkMetalPolyDataMapper2D`
  (GL's `AppendStripIndexBuffer` winding) and the property-texture fallback in
  `UpdateActorTexture` (see the triangle-strip section above);
  `TestPolyDataMapperNormals` (0.270) remains in the mid bucket.

### Evidence the core path is correct

The 133 passes include the strongest-scrutiny tests: `TestOpacity` (passes with
the `TIGHT_VALID` metric — the Lab-space color path matches GL to
`0.00038`), `TestOSConeCxx`, `TestMace`, the four scalar-LUT tests
(`TestCompositePolyDataMapperOverrideLUT`, `TestTextureInterpolateScalars`,
`TestTranslucentLUTTextureAlphaBlending`, `TestTranslucentLUTTextureDepthPeeling`
at ImageError 1.2e-06–1.2e-04), `TestTranslucentLUTAlphaBlending`,
`TestTranslucentLUTDepthPeeling`, `TestScalarModeToggle`,
`TestPointRendering_{1,2,3,4,Round_1,Round_2,Round_3,Round_4}` (all eight point
tests), `TestVertexRendering`, `TestQuadPointRep`,
`TestCompositePolyDataMapper` and its
`BlockOpacities`/`ToggleScalarVisibilities`/`PartialPointData`/`StaticBounds`/
`SharedArray` variants, `TestAreaSelections` and `TestHardwareSelector` (exact
per-primitive cell ids), the read-back cluster (`TestReadPixels`,
`TestSelectVisiblePoints`, `TestWorldPointPicker`, `TestRemoveActors`),
`TestNActors{OneMapper,NMappersOneInput}` (exact edge colors),
`TestActor2D` (2D overlay
quad matches GL to 0.027), the textured-background cluster
(`TestTexturedBackground`, `TestStereoBackground{Left,Right}`), the
multi-viewport cluster (`TestNViewports*`), the eight sub-viewport
`vtkImageMapper` tests (`TestImageMapper_1..4`, `TestDirectScalarsToColors`,
`TestBareScalarsToColors`, `TestMapVectorsAsRGBColors`, `TestMapVectorsToColors`
at ImageError ~1e-07-1e-06), the four triangle-strip tests
(`TestTStrips{TCoords,NormalsTCoords,NormalsColorsTCoords,ColorsTCoords}`, the
first time any strip-geometry test has passed), and the basic
Glyph3D,
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
   eliminated all 14 texture-fallback crashes; 7 texture tests now pass
   (`TestTextureWrap`, `TestBackfaceTexture`, `TestTextureRGBA`,
   `TestTextureRGBADepthPeeling`, `TestTextureSize`,
   `TestTextureInterpolateScalars` — the last via the scalar-texture LUT
   pipeline — and `TestTexturedCylinder`) and the
   rest fail image comparison instead of SIGSEGV. Follow-ups: border-color and
   filter/mipmap fidelity (tiling), and renderers that bypass
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
    `TestSelectVisiblePoints`, `TestWorldPointPicker` (pick check).
 5. **Read-back** — the read-back cluster (`TestReadPixels`, `TestRemoveActors`,
    `TestWindowToImageFilter`, `TestSelectVisiblePoints`, `TestWorldPointPicker`)
    now passes; `TestResizingWindowToImageFilter` (0.413, mid bucket) remains the
    read-back-class gap.
 6. **Label/text/image OpenGL-fallback cluster** — the 3 remaining crashes all
    instantiate the OpenGL label/text/image classes (e.g.
   `vtkOpenGLLabeledContourMapper::ApplyStencil`) against a Metal window; this
   needs Metal overrides for the label/text rendering stack.
 7. **Glyph instancing colors**, **2D overlay (image mapper done)**, **LUT/color
      mapping**, **stereo/multiview + gradient background** — all
      render but diverge from GL. (The glyph3D multi-source indexing tests
      `TestGlyph3DMapperIndexing`/`TreeIndexing` now pass — see the glyph3D
      multi-source indexing section above; `TestActor2D` now passes — see the
      2D-overlay section above; the sub-viewport `vtkImageMapper` tests
      `TestImageMapper_1..4`, `TestDirectScalarsToColors`, `TestBareScalarsToColors`,
      `TestMapVectorsAsRGBColors`, `TestMapVectorsToColors` now pass via the 2D
      image-mapper viewport-WCVC fix above; `TestPolyDataMapper2D` is at 0.066,
      the closest 2D miss; the textured/stereo backgrounds `TestTexturedBackground`
      and `TestStereoBackground{Left,Right}` now pass — see the textured-background
      section above.)
 8. **Triangle strips — DONE** — `vtkMetalPolyDataMapper` and
    `vtkMetalPolyDataMapper2D` now decompose `GetStrips()` on the CPU exactly like
    GL's `AppendStripIndexBuffer` (shared indexed/non-indexed surface emit,
    wireframe edges, strip cell ids after polys, legacy edge-overlay tubes), the
    GPU-tess path falls back to CPU only when strips are present, and
    `UpdateActorTexture` falls back to the property's named textures (the
    `TestTStrips*` suite's `mytexture` beach texture) — all four `TestTStrips*`
    tests now pass (see the triangle-strip section above).
 9. **Point rendering — DONE** — `vtkMetalPolyDataMapper` now creates the
    light-uniform buffer when only points/vertex cells are drawn (previously only
    the surface branch did, so point fragments read an empty `lights` buffer and
    rendered black) and tracks vertex cells via a `HasVerts` flag so the
    `VTK_POINTS`/vertex-representation gate is exact. All four point tests
    `TestPointRendering{,_Round}_3/4` and the vertex-visibility tests
    `TestVertexRendering`, `TestQuadPointRep`, `TestMixedGeometry_3` now pass
    (see the point-rendering section above). Remaining point-adjacent:
    `TestPointSelection*` (pick-check; point field-association selection not yet
    implemented).

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
completing the four scalar-LUT tests), and then after the 2D-overlay-mapper fix
(83 Passed / 83 Failed / 9 aborted — `TestActor2D` now passes via the
`vtkMetalPolyDataMapper2D` `Update()` / no-Y-flip fix), and then after the
wireframe-lines fix (92 Passed / 80 Failed / 3 aborted — `TestWireframe` now
passes via the `fragment_main_line` / `kSceneFlagLinesUnlit` unlit-line path),
and then after the glyph3D multi-source indexing fix
(96 Passed / 76 Failed / 3 aborted — `TestGlyph3DMapperIndexing` and
`TestGlyph3DMapperTreeIndexing` now pass), and then after the
textured-background fix
(99 Passed / 73 Failed / 3 aborted — `TestTexturedBackground` and
`TestStereoBackground{Left,Right}` now pass), and then after the edge-color and
flat-background fixes
(111 Passed / 61 Failed / 3 aborted — `TestNActors{OneMapper,NMappersOneInput}`
now pass via the ambient-pre-applied edge shader, `TestReadPixels` and
`TestRemoveActors` are restored via the flat-background alpha/dither fix, and
`TestNViewportsNActors*`, `TestSurfacePlusEdges`, `TestTexturedCylinder`,
`TestEdgeThickness`, `TestAxesActor` are gone from the failure set), and then
 after the 2D image-mapper viewport-WCVC fix
 (119 Passed / 53 Failed / 3 aborted — `TestImageMapper_1..4`,
 `TestDirectScalarsToColors`, `TestBareScalarsToColors`, `TestMapVectorsAsRGBColors`
 and `TestMapVectorsToColors` now pass via the size-only WCVC ortho in
 `vtkMetalImageMapper::DrawPixels`), and then after the triangle-strip support fix
 (123 Passed / 49 Failed / 3 aborted — the four `TestTStrips*` tests now pass via
 CPU-side strip decomposition in `vtkMetalPolyDataMapper`/`vtkMetalPolyDataMapper2D`
 matching GL's `AppendStripIndexBuffer`, plus the property-texture fallback in
 `UpdateActorTexture`), and then after the point-rendering lighting fix
 (130 Passed / 42 Failed / 3 aborted — `TestPointRendering{,_Round}_3/4`,
 `TestVertexRendering`, `TestQuadPointRep` and `TestMixedGeometry_3` now pass via
 the `HasVerts` gate and the hoisted `UpdateLightUniforms` call in
 `vtkMetalPolyDataMapper::RenderPiece`).
 The image-compare buckets above are from that latest run's `LastTest.log` (max
TIGHT_VALID error per test), analyzed with `analyze_metal_ctest_log.py`.
Re-running is reproducible except where a crash's signal stack
ordering varies; the pass count fluctuates run to run.
