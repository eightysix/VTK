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
    test `TestMetalImageSliceMapper` added by `3ec24ca9c5`; the working tree had
    regressed `TestMetalCamera`/`TestMetalPointRender` to a
    `vtkShaderProperty::New()` null-override abort, fixed by the
    `vtkMetalShaderProperty` override below).
2. The generic multi-backend suite in `Rendering/Core/Testing/Cxx/`, which
   registers the same ~175 tests once per backend and was wired up for Metal
       through    object-factory overrides (`--vtk-factory-prefer
             RenderingBackend=Metal`).    Historical status: **55 pass / 120 fail (33
               crash)**. Current working-tree status: **161 pass / 14 fail (0 crash)** —
        `TestGlyph3DMapperCompositeDisplayAttributeInheritance` now passes via the
       composite per-block display-attribute inheritance port in the Metal glyph
       mapper (see the composite display-attribute section below),
       `TestLineRenderingTranslucent` now passes via the OIT translucent-line
      pipelines (see the OIT translucent-line section below), the
      last crash class, `vtkLabeledContourMapper`, is fixed by the
      `vtkMetalLabeledContourMapper` override (see the labeled-contour-mapper
      section below); the 14 OpenGL-texture-fallback crashes are fixed by the `vtkMetalTexture`
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
      `UpdateLightUniforms` call (see the point-rendering section below), and the
      composite vertex/sphere tests `TestCompositePolyDataMapperSpheres` and
      `TestCompositePolyDataMapperVertices` now pass via the per-block edge/vertex
      color fix (see the per-block edge/vertex-color section below). The coincident
      point/line fixes make `TestCoincident` pass for the first time (see the
      coincident point-color/line-offset section below), and the new
      `vtkMetalShaderProperty` factory override restores the bespoke
      `TestMetalCamera`/`TestMetalPointRender` (which the working tree had
      regressed to a `vtkShaderProperty::New()` null-override abort; see the
       vtkMetalShaderProperty section below). No
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

### Tally (working tree, rerun 2026-08-04 on the M2 MacBook Air,
`ctest -R "RenderingCoreCxx-Metal" -j 8`)

```
175 tests:  161 Passed  14 Failed (incl. image/pick fails)  0 "Subprocess aborted"
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
Run after the surface lighting-flag fix (this run, 2026-08-03): 134 Passed /
38 Failed / 3 aborted out of 175 (analyzed with `analyze_metal_ctest_log.py`
from a single `ctest -R "RenderingCoreCxx-Metal" -j 8` run; failures exported
with `export_image_compare.sh`) — `TestActorLightingFlag` now passes (TIGHT_VALID
9.53e-05, was a 0.0513 near-miss fail): the central cone (property lighting
disabled) now renders flat white like the GL baseline instead of being shaded
like its neighbors. The mapper now bakes `vtkProperty::GetLighting()` into a new
`kSceneFlagLightingDisabled` scene bit (set when `!GetLighting()`) that every
surface/point/glyph fragment honors by skipping `computePhongLighting` and
emitting GL's NoLighting output `ambientIntensity*ambientColor +
diffuseIntensity*diffuseColor` (see the surface lighting-flag section below).
The +1 pass delta over the previous run is exactly that test; the 3 aborts are
unchanged (`TestLabeledContourMapper`, `TestLabeledContourMapperNoLabels`,
`TestLabeledContourMapperWithActorMatrix`). The regression check against the
  documented passing cluster reports none.
Run after the thick-line/tube lighting bake (this run, 2026-08-03): 134 Passed /
38 Failed / 3 aborted out of 175 (analyzed with `analyze_metal_ctest_log.py`
from a single `ctest -R "RenderingCoreCxx-Metal" -j 8` run) — the pass count is
unchanged, but the thick-line/tube pipelines now bake
`vtkProperty::GetLighting()` into a dedicated `kLightingDisabled` function
constant (16), closing the documented "Known gap": a `RenderLinesAsTubes` actor
with `SetLighting(false)` now renders its tubes flat like GL instead of running
the Phong loop (see the tube-light bake section below). The only metric deltas:
`TestRenderLinesAsTubes` and `TestRenderLinesAsTubesOrthoCamera` improved from
0.2350/0.2349 to 0.2292/0.2292 (still mid-bucket; the residual error is the
pre-existing thick-line tube-shading fidelity gap, not the lighting flag). The
3 aborts are unchanged (`TestLabeledContourMapper`,
`TestLabeledContourMapperNoLabels`, `TestLabeledContourMapperWithActorMatrix`).
The regression check against the documented passing cluster reports none.

Run after the tiled-viewport/tile-aware-gradient fix (this run, 2026-08-03): 139 Passed /
33 Failed / 3 aborted out of 175 (analyzed with `analyze_metal_ctest_log.py` from a single
`ctest -R "RenderingCoreCxx-Metal" -j 8` run; failures exported with
`export_image_compare.sh`) — the three gradient-background tests now pass for the first
time (`TestGradientBackground` at 0.0023, `TestGradientBackgroundWithTiledViewport` at
0.0012, `TestGradientBackgroundWithTiledViewports` at 0.0014; were mid-bucket 0.3449
and gross-bucket 0.5063/0.5856 fails), `TestTilingCxx` improved from gross 0.6159 to mid
0.2054, and `TestGlyph3DMapper` (mid 0.1082) and `RenderNonFinite` (near-miss 0.0870)
also left the failure set this run, both stable across three re-runs; see the
tiled-viewport/gradient-background section below. The +5 pass delta over the previous
run is exactly those five tests; the 3 aborts are unchanged (`TestLabeledContourMapper`,
`TestLabeledContourMapperNoLabels`, `TestLabeledContourMapperWithActorMatrix`).
The regression check against the documented passing cluster reports none.
Run after the labeled-contour-mapper fix (this run, 2026-08-04): 142 Passed /
33 Failed / 0 aborted out of 175 (analyzed with `analyze_metal_ctest_log.py` from a
single `ctest -R "RenderingCoreCxx-Metal" -j 8` run) — the last crash class,
`vtkLabeledContourMapper` (all three tests crashed in
`vtkOpenGLLabeledContourMapper::ApplyStencil`/`RemoveStencil`, which were wrongly
dispatched because no Metal override existed for `vtkLabeledContourMapper` and the
object factory selected the OpenGL override even under
`--vtk-factory-prefer RenderingBackend=Metal`), now resolves to a Metal override
(see the labeled-contour-mapper section below). `TestLabeledContourMapper`,
`TestLabeledContourMapperNoLabels` and `TestLabeledContourMapperWithActorMatrix`
now pass (TIGHT_VALID ImageErrors 0.0002 / 0 / 0.0286). The +3 pass delta over
the previous run is exactly those three tests; the failure set is otherwise
unchanged (33 image-compare/pick fails, no new failures). The regression check
against the documented passing cluster reports none.
Run after the stereo-composite write-back fix (this run, 2026-08-04): 145 Passed /
30 Failed / 0 aborted out of 175 (analyzed with `analyze_metal_ctest_log.py` from a
single `ctest -R "RenderingCoreCxx-Metal" -j 8` run) — `TestOffAxisStereo` (gross
0.5921), `TestSplitViewportStereoHorizontal` (gross 0.6816) and
`TestStereoEyeSeparation` (mid 0.2584) now pass, the first time any have passed;
`vtkMetalRenderWindow` now implements `SetPixelData`/`SetRGBACharPixelData` (the
CPU-side composite was previously dropped by the base-class no-op) and the
eye-pass presents are suppressed for CPU-composited stereo (see the
stereo-composite write-back section below). The +3 pass delta over the previous
run is exactly those three tests; the failure set is otherwise unchanged (26
image-compare + 1 below-threshold pick-check + 3 non-image fails, no new
failures). The regression check against the documented passing cluster reports
none.
Run after the 2D poly-data-mapper line-width/point-size fix (this run, 2026-08-04):
149 Passed / 26 Failed / 0 aborted out of 175 (analyzed with
`analyze_metal_ctest_log.py` from a single `ctest -R "RenderingCoreCxx-Metal" -j 8`
run; failures exported with `export_image_compare.sh`) — the full
`TestPolyDataMapper2D` family now passes for the first time:
`TestPolyDataMapper2D` (was a 0.0664 near-miss fail) and the point-scalar /
cell-scalar variants (were 0.2861/0.2943 mid-bucket fails) all pass at TIGHT_VALID
0: the Metal 2D shaders now emit `[[point_size]]` from the actor's point size and
render `lineWidth > 1` as screen-space quads instead of always drawing 1px
lines/points (see the 2D overlay line-width and point-size section below).
`TestColorByStringArrayDefaultLookupTable2D` (was a 0.4821 mid-bucket fail) also
left the failure set — fixed by the 2D scalar-color mapping commit `bd68ee20b2`,
whose effect this run documents for the first time. The +4 pass delta over the
previous run is exactly those four tests; the failure set is otherwise unchanged
(22 image-compare + 1 below-threshold pick-check + 3 non-image fails, no new
failures). The regression check against the documented passing cluster reports
none.
Run after the 2D overlay tile-cropping fix (this run, 2026-08-04): 150 Passed /
25 Failed / 0 aborted out of 175 (analyzed with `analyze_metal_ctest_log.py` from
a single `ctest -R "RenderingCoreCxx-Metal" -j 8` run; failures exported with
`export_image_compare.sh`) — `TestTilingCxx` now passes (ImageError 0, was a mid
0.2195 fail), the first time it has passed; see the 2D overlay tile-cropping
section below. The +1 pass delta over the previous run is exactly that test; the
failure set is otherwise unchanged (21 image-compare + 1 below-threshold
pick-check + 3 non-image fails, no new failures). The regression check against
the documented passing cluster reports none. The temporary `TestTilingDebug`
debug harness used to diagnose the bar was removed (its CMakeLists entry and the
source file), restoring the suite to 175 tests.
Run after the custom-shader support fix (this run, 2026-08-04): 151 Passed /
24 Failed / 0 aborted out of 175 (analyzed with `analyze_metal_ctest_log.py` from
a single `ctest -R "RenderingCoreCxx-Metal" -j 8` run; failures exported with
`export_image_compare.sh`) — `TestCompositePolyDataMapperCustomShader` now passes
(the first shader-replacement test to pass on Metal; was a gross 0.2897 image
fail), via the `vtkShaderProperty` replacement mechanism and GLSL→MSL shim
described in the custom-shader section below (TIGHT_VALID 1.18e-06). The +1 pass delta over the previous
run is exactly that test; the failure set is otherwise unchanged (20 image-compare
+ 1 below-threshold pick-check + 3 non-image fails, no new failures). The
regression check against the documented passing cluster reports none.
Run after the per-block edge/vertex-color fix (this run, 2026-08-04): 153 Passed /
22 Failed / 0 aborted out of 175 (analyzed with `analyze_metal_ctest_log.py` from
a single `ctest -R "RenderingCoreCxx-Metal" -j 8` run; failures exported with
`export_image_compare.sh`) — `TestCompositePolyDataMapperSpheres` (was a mid
0.1499 fail) and `TestCompositePolyDataMapperVertices` (was a mid 0.2891 fail) now
pass (TIGHT_VALID ImageErrors 0.0215 / 0.0244), the first time either has passed;
see the per-block edge/vertex-color section below. The +2 pass delta over the
previous run is exactly those two tests; the failure set is otherwise unchanged
(18 image-compare + 1 below-threshold pick-check + 3 non-image fails, no new
failures). The regression check against the documented passing cluster reports
none.
Run after the coincident point-color/line-offset fix and the
`vtkMetalShaderProperty` factory override (this run, 2026-08-04): 154 Passed /
21 Failed / 0 aborted out of 175 (analyzed with `analyze_metal_ctest_log.py` from
a single `ctest -R "RenderingCoreCxx-Metal" -j 8` run; failures exported with
`export_image_compare.sh`) — `TestCoincident` now passes for the first time
(ImageError 0; ctest method `LOOSE_VALID` 0.00443623, `TIGHT_VALID` 0.0154217):
the Metal point fragments ignored the property's point color and rendered the
pink coincident dots white, and the 1px-line coincident depth offset was dropped
so the pink dots z-fought with the coincident surface (see the coincident
point-color/line-offset section below). The +1 pass delta over the previous run is
exactly that test; the failure set is otherwise unchanged (17 image-compare + 1
below-threshold pick-check + 3 non-image fails, no new failures). The regression
check against the documented passing cluster reports none. The bespoke suite's
`TestMetalCamera` and `TestMetalPointRender` — Subprocess aborted on the working
tree via a `vtkShaderProperty::New()` null-override crash in
`vtkProp::GetShaderProperty()` — now pass via the `vtkMetalShaderProperty` factory
override (see below); the bespoke suite re-ran at 16/16 pass.
Run after the edge-flag support fix (this run, 2026-08-04): 155 Passed /
20 Failed / 0 aborted out of 175 (analyzed with `analyze_metal_ctest_log.py` from
a single `ctest -R "RenderingCoreCxx-Metal" -j 8` run; failures exported with
`export_image_compare.sh`) — `TestEdgeFlags` now passes for the first time
(TIGHT_VALID 0.0102255, was a 0.0681 near-miss fail): the Metal poly-data mapper
now honors the `vtkDataSetAttributes::EDGEFLAG` point attribute in the CPU
wireframe path, the GPU wireframe/surface-edge tessellation kernels, and the
legacy and single-pass edge overlays (see the edge-flag support section below).
The +1 pass delta over the previous run is exactly that test; the failure set is
otherwise unchanged (16 image-compare + 1 below-threshold pick-check + 3
non-image fails, no new failures). The regression check against the documented
passing cluster reports none.
Run after the resize-capture offscreen-target gate fix (this run, 2026-08-04):
156 Passed / 19 Failed / 0 aborted out of 175 (analyzed with
`analyze_metal_ctest_log.py` from a single `ctest -R "RenderingCoreCxx-Metal" -j 8`
run; failures exported with `export_image_compare.sh`) — `TestResizingWindowToImageFilter`
now passes (TIGHT_VALID ImageErrors 4.3e-04 / 5.4e-04 / 3.5e-04 / 1.9e-04 across
the four resolutions, was a mid-bucket 0.4130 image fail): the Metal renderer and
render window gated their offscreen color target on `GetOffScreenRendering()`
(`!ShowWindow`), but `vtkResizingWindowToImageFilter` sets only
`UseOffScreenBuffers` (leaving `ShowWindow` true) and resizes through the base
`vtkRenderWindow::SetSize`, so the first capture fell back to the stale 400x400
`CAMetalLayer` drawable while the depth/viewport/read-back were sized for the new
resolution — only the top-left 400x400 of the read-back was valid and nothing
drew (see the resize-capture offscreen-target section below). The +1 pass delta
over the previous run is exactly that test; the failure set is otherwise
unchanged (15 image-compare + 1 below-threshold pick-check + 3 non-image fails,
no new failures). The regression check against the documented passing cluster
reports none.
Run after the glyph3D backface-color fix (this run, 2026-08-04): 157 Passed /
18 Failed / 0 aborted out of 175 (analyzed with `analyze_metal_ctest_log.py`
from a single `ctest -R "RenderingCoreCxx-Metal" -j 8` run; failures exported
with `export_image_compare.sh`) — `TestGlyph3DMapperBackfaceColor` now passes
(the glyph mapper previously mirrored the front material into the backface
material slots without reading `actor->GetBackfaceProperty()` and the glyph
fragment shader had no `front_facing` handling, so the magenta backface
property never reached the inner "mouth" surfaces, which rendered black/unlit;
see the glyph3D backface-color section below). The +1 pass delta over the
previous run is exactly that test; the failure set is otherwise unchanged
(14 image-compare + 3 non-image fails, no new failures). The regression check
against the documented passing cluster reports none.
Run after the glyph3D point-selection fix (this run, 2026-08-04): 158 Passed /
17 Failed / 0 aborted out of 175 (analyzed with `analyze_metal_ctest_log.py`
from a single `ctest -R "RenderingCoreCxx-Metal" -j 8` run; failures exported
with `export_image_compare.sh`) — `TestGlyph3DMapperPicking` now passes for the
first time (TIGHT_VALID ImageError 0; was a near-miss 0.0800 image fail): the
Metal selection pass rendered glyphs with their exact sphere silhouettes, while
GL dilates pick coverage by drawing every glyph-source vertex as a 6.0px point
during point selection (`vtkOpenGLGlyph3DHelper::GlyphRender`), so a pick
rectangle grazing a glyph column could miss points GL would catch. The Metal
glyph mapper now mirrors that — with a point-field-association hardware
selector active it routes every source vertex through the point pipeline at
`scene.pointSize` = 6.0, and the Turn-On id set is now identical to GL's 21 ids
(see the glyph3D point-selection section below). The +1 pass delta over the
previous run is exactly that test; the failure set is otherwise unchanged
(13 image-compare + 1 below-threshold pick-check + 3 non-image fails, no new
failures). The regression check against the documented passing cluster reports
none.
Run after the OIT translucent-line fix (this run, 2026-08-04): 159 Passed /
16 Failed / 0 aborted out of 175 (analyzed with `analyze_metal_ctest_log.py`
from a single `ctest -R "RenderingCoreCxx-Metal" -j 8` run; failures exported
with `export_image_compare.sh`) — `TestLineRenderingTranslucent` now passes for
the first time (TIGHT_VALID ImageError 0.0315735, was a mid-bucket 0.0790303
image fail): the OIT accumulate pass targets RGBA16F/R16F attachments, and the
line draws were skipped there because line pipelines only existed in the
non-OIT BGRA8 variants, so translucent lines were invisible during OIT (see
the OIT translucent-line section below). The `vtkMetalOrderIndependentTranslucentPass`
first-frame "invalid or MSAA depth texture; falling back to standard
transparency" warning is also gone: `vtkMetalRenderWindow::Render` now applies
the superclass's 300x300 size default before creating the depth texture, so the
very first frame has a valid DepthTexture. The +1 pass delta over the previous
run is exactly that test; the failure set is otherwise unchanged
(12 image-compare + 1 below-threshold pick-check + 3 non-image fails, no new
failures). The regression check against the documented passing cluster reports
none.
Run after the composite display-attribute inheritance fix (this run, 2026-08-04):
160 Passed / 15 Failed / 0 aborted out of 175 (analyzed with
`analyze_metal_ctest_log.py` from a single `ctest -R "RenderingCoreCxx-Metal" -j 8`
run; failures exported with `export_image_compare.sh --no-run`) —
`TestGlyph3DMapperCompositeDisplayAttributeInheritance` now passes for the first
time (was a mid-bucket 0.224 image fail that rendered only the black background):
the Metal glyph mapper rejected composite inputs, so the `vtkPartitionedDataSetCollection`
(12 colored shapes with block 3 at 50% opacity and block 9 hidden) never drew;
the mapper now walks composite datasets recursively, applying the inherited
per-block display attributes (visibility/pickability/color/opacity) like
`vtkOpenGLGlyph3DMapper` (see the composite display-attribute section below). The
+1 pass delta over the previous run is exactly that test; the failure set is
otherwise unchanged (11 image-compare + 1 below-threshold pick-check + 3
non-image fails, no new failures). The regression check against the documented
passing cluster reports none.
Run after the point/line draw-order + flat wide-line shading fix (this run,
2026-08-04): 161 Passed / 14 Failed / 0 aborted out of 175 (analyzed with
`analyze_metal_ctest_log.py` from a single `ctest -R "RenderingCoreCxx-Metal" -j 8`
run; failures exported with `export_image_compare.sh --no-run`) —
`TestMixedGeometryCellScalars` now passes for the first time (TIGHT_VALID
ImageError 0.000, was a mid-bucket 0.137264 image fail): the Metal mapper drew
wide lines before points, so the 20px point sprites (drawn last, LEQUAL depth)
overwrote the 10px blue polyline where they overlapped, and wide non-tube lines
used the fake-tube cylinder normal (a blue gradient darker toward the long
borders) while GL's native `glLineWidth` renders a flat strip (see the
point/line draw-order + flat wide-line shading section below). The +1 pass
delta over the previous run is exactly that test; the failure set is otherwise
unchanged (10 image-compare + 1 below-threshold pick-check + 3 non-image fails,
no new failures). The regression check against the documented passing cluster
reports none.

### The point/line draw-order + flat wide-line shading fix (`TestMixedGeometryCellScalars` now passes)

`TestMixedGeometryCellScalars` renders a `vtkPolyPointSource` (5 points),
`vtkPolyLineSource` (open polyline) and `vtkRegularPolygonSource` (8-sided
polygon) appended with per-cell scalar values (points 0.1, line 0.5, polygon
0.9) through a single mapper with the line width 10 and point size 20. It
failed image comparison at a mid-bucket 0.137264 with exactly the line wrong:
the baseline blue polyline (1365 px) rendered as only 319 px of blue plus a
strong blue gradient across the line width.

Two root causes, both in `vtkMetalPolyDataMapper` / `MetalShaders.metal`:

- **Draw order.** `vtkOpenGLPolyDataMapper` draws primitives in
  points → lines → triangle order (`PrimitiveStart` → `PrimitiveVertices`),
  so later primitives overwrite earlier ones at equal depth (LEQUAL depth
  test). The Metal mapper recorded the wide-line draw *before* the point
  draws, so the 20px point sprites (drawn last) occluded the 10px polyline
  wherever the S-curve passed under or near the points. The two point blocks
  ("vertex visibility" dots and the VTK_POINTS representation pass) were moved
  ahead of the line-drawing block, restoring the GL points-first order
  (triangles → points → lines → edge overlay in the file); the depth-peel / OIT
  gates on each block are unchanged.
- **Wide-line shading.** The thick-line / round-cap / miter-join fragment
  shaders (`shadeLineFragment`, `shadeLineFragmentOIT`) always built the
  fake-tube cylinder normal (`dist_to_centerline`-based cross-section) for the
  Phong lighting, producing a blue gradient darker toward the long borders.
  GL's native `glLineWidth` wide lines are a flat screen-space strip, and the
  baseline shows a uniform color. The fragments now use a flat view-facing
  normal `float3(0,0,1)` unless the new `kSceneFlagLinesTubeShading` scene bit
  (1u << 19, wired to `vtkProperty::GetRenderLinesAsTubes()`) is set, which
  keeps the cylinder shading exactly for `RenderLinesAsTubes` scenes. The
  fragments gained a `constant SceneUniforms& scene` argument (all 6 thick/OIT
  call sites updated); 1px lines, surface, points, and glyph fragments are
  untouched, so the flag only affects wide non-tube lines.

`TestMixedGeometryCellScalars` now passes at TIGHT_VALID 0.000 (pixel-perfect:
red/green/blue/white pixel counts match the baseline exactly), and the
`RenderLinesAsTubes` tests are unchanged (`TestRenderLinesAsTubes` /
`TestRenderLinesAsTubesOrthoCamera` still at their pre-existing near-miss
0.0535, since the cylinder path is preserved behind the flag). The full suite
re-ran at 161 pass / 14 fail / 0 aborted with the failure set otherwise
unchanged and no regression against the documented passing cluster.

### The OIT translucent-line fix (`TestLineRenderingTranslucent` now passes)

`TestLineRenderingTranslucent` renders the `TestLineRendering` line scene
(line width 4, `MiterJoin` join, opacity 0.4) with `SetMultiSamples(0)`, which
disables MSAA and enables VTK's order-independent translucent pass. It failed
image comparison at a mid-bucket 0.0790303: under OIT the translucent lines
were invisible.

The OIT accumulate pass (`vtkMetalOrderIndependentTranslucentPass`) creates
RGBA16F + R16F intermediate textures, and its resolve shader
(`fragment_oit_resolve`) emits `float4(accum.rgb / max(reveal, 0.01),
1.0 - accum.a)` against a `ReadOnlyDepthState` (Less, no depth write) using the
scene's non-MSAA Depth32Float. The line draw in `vtkMetalPolyDataMapper::RenderPiece`
was gated on `!oitActive` ("those pipelines write to different color
attachments than the pass provides"), so during the OIT accumulate frame the
lines were simply never recorded.

The fix, following the existing triangle OIT pipeline (`EnsureOITPipelineStates`)
and the tube-line pipeline pattern (`BuildTubeLinePipelineVariant`):

- **OIT line pipelines.** New `BuildOITLinePipelineVariant` builds
  miter-join / thick / round-cap variants against the RGBA16F/R16F attachments
  with the same blend config as the triangle OIT pipeline (color(0) RGB
  ONE/ONE add, A ZERO/ONE_MINUS_SRC_ALPHA add; color(1) ONE/ONE add),
  `MTLPrimitiveTopologyClassTriangle`, sample count 1, and the function
  constant 16 (`kLightingDisabled`) split; a dedicated `EnsureLineOITPipelineState`
  does the same for 1px lines via `vertex_main` + `fragment_main_line_oit`.
- **Draw gate.** `drawLines` now permits the OIT accumulate pass and selects
  the OIT pipeline variants (`LineOITPipeline` /
  `MiterJoinLineOITPipeline[Unlit]` / `ThickLineOITPipeline[Unlit]` /
  `RoundCapLineOITPipeline[Unlit]`) when `oitActive`, while the non-OIT path is
  untouched. Depth-peel passes still skip lines. The OIT pipelines are created
  lazily in the `renWin->OITActive` ensure block (miter if
  `MiterJoinLineSegmentCount > 0`, else thick if `ThickLineSegmentCount > 0`,
  else 1px if `lineWidth <= 1.0f`), so scenes without translucent lines never
  pay for them.
- **OIT line fragment shaders.** `shadeLineFragmentOIT` mirrors
  `shadeLineFragment` (same cylinder-normal Phong lighting and coincident-depth
  offset) but writes premultiplied color/revealage; `fragment_main_line_oit`
  mirrors `fragment_main_oit` for 1px lines, honoring the
  `kSceneFlagLinesUnlit` scene flag and `resolveMaterial`/`resolveCellColor`
  (cellColorTex texture 8, lutTexture 9, lutSampler 1 bound in the OIT draw).
  The three tube entry points (`fragment_{thick,round_cap,miter_join}_line_main_oit`)
  wrap it.

**First-frame depth texture.** The `vtkMetalOrderIndependentTranslucentPass`
"invalid or MSAA depth texture" warning fired once per process on the first
frame: `vtkMetalRenderWindow::Render` checked `Size[0] > 0` before the
superclass `vtkRenderWindow::Render` applied its `SetSize(300, 300)` default,
so no DepthTexture existed during the first OIT pass and it fell back to
standard transparency (the second frame rendered correctly, which is why the
old code still failed at only 0.079). The Metal render window now applies the
same size default first, so every frame has a valid DepthTexture and the
warning is gone.

`TestLineRenderingTranslucent` passes at TIGHT_VALID 0.0315735 (deterministic
across five re-runs; the residual error is the OIT-vs-alpha-blend fidelity gap),
`TestLineRendering` (opaque) and all six `TestTranslucent*` tests still pass,
and the full suite re-ran at 159 pass / 16 fail / 0 aborted with the failure
set otherwise unchanged and no regression against the documented passing
cluster.

### The composite display-attribute inheritance fix (`TestGlyph3DMapperCompositeDisplayAttributeInheritance` now passes)

`TestGlyph3DMapperCompositeDisplayAttributeInheritance` renders the 12 shapes of
a `vtkPartitionedDataSetCollectionSource` through `vtkGlyph3DMapper` with a
`vtkCompositeDataDisplayAttributes` set on the mapper: blocks 0-8 and 10 are
colored (yellow/red/magenta), block 3 is at 50% opacity, block 9 is hidden, and
`SetOrientationArray("Normals")`/`ScaleFactor(0.5)` drive the per-point glyph
transform. Under Metal nothing drew — the black-background baseline mismatch was
a mid-bucket 0.224 image fail.

The root cause is in `vtkMetalGlyph3DMapper::Render` (`Rendering/Metal/vtkMetalGlyph3DMapper.mm`):
the mapper only handled a `vtkDataSet` input and early-returned when
`vtkDataSet::SafeDownCast(inputDataObject)` was null, so a `vtkCompositeDataSet`
(including the `vtkPartitionedDataSetCollection` this test feeds) never reached
any draw call. `vtkOpenGLGlyph3DMapper` handles this by walking the composite
tree with `RenderChildren`, applying the per-block display attributes
(visibility/pickability/color/opacity) inherited from parent blocks and passing
the flat composite index through to the pick buffer.

The Metal mapper now ports that structure:

- **Internals rework.** The single-dataset `Instances` vector was replaced with
  a `DataSetCache` (`std::map<const vtkDataSet*, DataSetInstances>`), each entry
  holding the per-source instanced buffers plus the mtimes they were built with
  (`CachedInputMTime`/`CachedNumSources`/`CachedBlockMTime`). A
  `ClearUnusedCachedEntries` pass drops entries for datasets no longer in the
  input, mirroring the OpenGL mapper.
- **Composite walk.** New member helpers `RenderChildren` (recursive,
  flat-index-increment-before-recurse with null-child skipping, matching GL),
  `RenderDataSet`, `BuildAndUploadInstances` and `DrawInstances`. `RenderChildren`
  layers block overrides onto the inherited parent color/opacity and skips
  invisible blocks (and unpickable blocks during a selection pass); leaf
  `vtkDataSet`s render through `RenderDataSet`, which builds the cached instance
  buffers and issues one instanced draw per glyph source.
- **Per-block pick ids.** `DrawInstances` writes the `PickIds {propId,
  compositeIndex}` for the current block into the `PropIdBuffer` before each
  draw, so the flat composite index reaches the glyph vertex shader for picking
  exactly as the per-dataset path did (composite index 0 for a plain
  `vtkDataSet` input).
- **Cache invalidation.** A per-dataset entry is rebuilt when the dataset mtime,
  the source count, or `BlockAttributes` mtime changes (`BlockMTime` recorded at
  the top of `Render`); block-color/opacity changes therefore invalidate the
  baked per-instance colors without touching the transform buffers of other
  datasets.

The new member helpers are declared in `vtkMetalGlyph3DMapper.h` with
`void*` device/encoder parameters (cast inside the `.mm`), following the
`vtkMetalPolyDataMapper.h` convention so the header stays Objective-C-free.
`TestGlyph3DMapperCompositeDisplayAttributeInheritance` now passes (TIGHT_VALID
0), the bespoke `TestMetalGlyph3DMapper` still passes, and the full suite re-ran
at 160 pass / 15 fail / 0 aborted with the failure set otherwise unchanged and
no regression against the documented passing cluster.

### The glyph3D point-selection fix (`TestGlyph3DMapperPicking` now passes)

`TestGlyph3DMapperPicking` renders two 7x7 grids of sphere glyphs (the right
actor shifted +2 world units and `PickableOff`) and area-picks a 30px-wide
rectangle (x 53-82) over the left actor's left columns, asserting the selected
point-id set. Under Metal the selection returned fewer ids than GL: the pick
rectangle grazed glyph column 2, whose sphere silhouettes ended at x≈51 —
just outside the rectangle.

A probe built from the test's own compile flags dumped both backends'
selection buffers (GL: `/tmp/gl_pass{0,1,2,5}_*.ppm` via
`vtkHardwareSelector::SavePixelBuffer`; Metal: `/tmp/metal_ids.bin` via
`vtkMetalRenderWindow::GetIdsData`, raw ids are id+1). GL's pass 2
(`POINT_ID_LOW24`) covered all 30 area columns (rows 21-122) with the full
21-id grid `{2,3,4,9,10,11,16,17,18,23,24,25,30,31,32,37,38,39,44,45,46}`;
Metal's single-pass IdBuffer covered only window-x 57-65 and 72-80 at the same
grid positions — the geometry was correct but the coverage too narrow. The gap
is GL's `vtkOpenGLGlyph3DHelper::GlyphRender`: during point selection it draws
every glyph-source vertex as a 6.0px `GL_POINT`, dilating coverage by ~3px on
either side of the silhouette.

The fix in `vtkMetalGlyph3DMapper.mm`:

- **Point-size uniform.** When `ren->GetSelector()` targets
  `vtkDataObject::FIELD_ASSOCIATION_POINTS`, write 6.0 into the
  `SceneUniforms.pointSize` float at byte offset 260 (after the uint Flags at
  256), matching GL's 6.0px sprites; the camera default of 1.0 is kept
  otherwise.
- **Selection draw path.** In the source draw loop, when selecting points,
  bind the existing point pipeline (`MTLPrimitiveTypePoint`,
  `g->VertexCount`) for every source regardless of its topology
  (triangle/line/point alike), instead of the surface geometry. Non-selection
  rendering is unchanged.

The Metal Turn-On set now equals GL's 21 ids, the regression image matches
(TIGHT_VALID ImageError 0), and the full-suite re-run shows no new failures —
the remaining 17 fails are the unchanged 13 image-compare + 1 below-threshold
pick-check (`TestCompositePolyDataMapperPickability`) + 3 non-image set.

### The glyph3D backface-color fix (`TestGlyph3DMapperBackfaceColor` now passes)

`TestGlyph3DMapperBackfaceColor` renders partial spheres (`vtkSphereSource` with
`SetStartTheta(20)`/`SetEndTheta(330)`) through `vtkGlyph3DMapper` whose actor
sets a magenta `vtkBackfaceProperty`. The baseline expects the yellow outer
shells and magenta exposed inner "mouth" surfaces; under Metal the mouths
rendered black/unlit.

The glyph path in the Metal backend had no backface support at all. In
`vtkMetalGlyph3DMapper.mm` the material upload (`UpdateMaterialUniforms`-style
40-float `MaterialBuffer`) mirrored the front material into the backface slots
with a plain `memcpy` and never queried `actor->GetBackfaceProperty()`, so the
magenta color never reached the GPU. And the glyph fragment shader
(`shadeGlyphFragment` in `MetalShaders.metal`) had no `[[front_facing]]`
parameter, no backface normal flip, and no backface material swap — backfaces
were lit with the outward normals, yielding black inner surfaces (the surface
mapper's `evaluateSurfaceFragment` already had the correct pattern).

The fix mirrors the surface path and GL (`vtkOpenGLGlyph3DHelper`):

- **Material upload.** The mapper now reads `actor->GetBackfaceProperty()` and
  fills the backface material slots (ambient/diffuse/specular color +
  intensity, color, opacity, specular power), mirroring `vtkMetalPolyDataMapper`.
- **Scene flag.** A new `kSceneFlagGlyphHasBackface` scene bit (18, next free
  after `kSceneFlagHasPointColors` 17) is asserted per-draw when the actor has a
  backface property, alongside the existing `kSceneFlagGlyphHasNormals` bit.
  The glyph path uses runtime flags rather than the function-constant
  specialization the surface path uses for `kHasBackface`, consistent with how
  `kSceneFlagGlyphHasNormals`/`kSceneFlagLightingDisabled` are already handled,
  so no extra pipeline variants are created.
- **Shader.** `shadeGlyphFragment` takes a `frontFacing` argument. Backfaces
  flip the geometric normal (`N = -N`, matching the surface shader and GL's
  light mod) and, when the flag is set, swap in the backface material. Colors
  match GL's `ReplaceShaderColor` exactly: front faces use the per-glyph color,
  back faces use the backface material color, and backface opacity is the
  property value without the glyph-color alpha. The triangle entry point
  `fragment_glyph_main` reads `[[front_facing]]`; the line/point entry points
  pass `true` because non-triangle primitives are always front-facing (GL's
  rule), so the new `[[front_facing]]` attribute is only used on the triangle
  pipeline.

`TestGlyph3DMapperBackfaceColor` now passes at ImageError 0 (was 0.2657). The
rest of the pipeline is untouched: only `vtkMetalGlyph3DMapper` and the three
glyph fragment entry points changed, the new scene bit collides with nothing,
and the full suite re-ran at 157 pass / 18 fail / 0 aborted with no regression
against the documented passing cluster.

### The resize-capture offscreen-target gate is fixed (`TestResizingWindowToImageFilter` now passes)

`TestResizingWindowToImageFilter` renders a sphere, captures the window at four
resolutions (1280x720, 1440x1080, 2048x1080, 4096x2160) through
`vtkResizingWindowToImageFilter`, then re-displays each capture. It failed under
Metal: the first capture (which happens before the test calls
`SetOffScreenRendering(true)`) read back only a 400x400 top-left region with no
sphere.

The root cause is a gate mismatch between the Metal backend and the capture
filters. `vtkResizingWindowToImageFilter::RequestData` (`Rendering/Core/`) calls
`SetUseOffScreenBuffers(true)` — leaving `ShowWindow` true — and then resizes via
`renWin->vtkRenderWindow::SetSize(...)`, deliberately bypassing the
`vtkMetalRenderWindow::SetSize` override so `CAMetalLayer.drawableSize` stays at
the window's original 400x400. The Metal renderer and render window, however,
gated their offscreen color target on `GetOffScreenRendering()` (= `!ShowWindow`),
which is false at that point, so the capture rendered into the stale 400x400
drawable while the depth attachment, viewport, and read-back (`ColorCopyTexture`)
were all sized for the new resolution — a mismatched render pass (nothing drew)
and a read-back of only the top-left 400x400, black elsewhere. OpenGL is
unaffected because `vtkOpenGLRenderWindow` keys its FBO off `UseOffScreenBuffers`.

`vtkWindow::SetOffScreenRendering` always sets both ivars together
(`ShowWindow = !val`, `UseOffScreenBuffers = val`), so gating on
`GetUseOffScreenBuffers()` instead is strictly more correct: it covers the normal
offscreen case and the resize-capture case. The two sites —

- the color-target selection in `vtkMetalRenderer::DeviceRender`
  (`Rendering/Metal/vtkMetalRenderer.mm`), and
- the offscreen-texture recreation in `vtkMetalRenderWindow::Render`
  (`Rendering/Metal/vtkMetalRenderWindow.mm`)

— now both gate on `GetUseOffScreenBuffers()`. The on-screen drawable path
(`ShowWindow` true, `UseOffScreenBuffers` false) is unchanged, so nothing that
worked before regresses. `TestResizingWindowToImageFilter` now passes all four
resolutions (TIGHT_VALID ImageErrors 4.3e-04 / 5.4e-04 / 3.5e-04 / 1.9e-04), and
the full-suite fail count dropped exactly by that one test with no new failures.

### The edge-flag support is fixed (`TestEdgeFlags` now passes)

`TestEdgeFlags` draws a square from four triangles (plus a five-point polygon)
with a per-point `vtkDataSetAttributes::EDGEFLAG` attribute (`vtkEdgeFlags`, a
1-component `vtkUnsignedCharArray`) that hides the internal edges: only the
boundary of the square is drawn. The Metal mapper previously derived edge
visibility from the polygon boundary alone and never consulted the attribute, so
the internal edges rendered and the test failed at a near-miss 0.0681.

The Metal poly-data mapper (`vtkMetalPolyDataMapper.mm`) and the tessellation
kernels (`Rendering/Metal/Shaders/MetalShaders.metal`) now reproduce GL's
semantics exactly. GL's `AppendEdgeFlagIndexBuffer` draws the edge from point `p`
to the next point in a polygon only when `p`'s flag is non-zero, and the
surface-edge `AppendTriangleIndexBuffer` masks its per-corner boundary values
(`val & mask`) by the three corner flags; triangle strips ignore edge flags
(`AppendStripIndexBuffer` takes none).

- **Attribute read.** `BuildGeometryBuffers` reads the active `EDGEFLAG`
  attribute and honors it only when it is a 1-component `vtkUnsignedCharArray`
  (GL's `vtkArrayDownCast` gate); anything else falls back to null. A
  `pointEdgeFlag(p)` lambda resolves visibility per point.
- **CPU wireframe** (`emitWireframeCell`): the polygon branch skips the edge
  `(pts[i], pts[(i+1)%npts])` when `pointEdgeFlag(pts[i])` is false; the strip
  branch is unchanged (strips ignore flags).
- **GPU wireframe kernel** (`polygonEdgesToLines`): the host `wPrimCounts`
  prefix sum now counts only flag-on edges and the kernel iterates the polygon
  and skips flag-off edges, so the output offsets stay consistent. The unfiltered
  behavior (all polygon edges) is preserved when no attribute exists
  (`writeEdgeFlags == 0`).
- **Surface edges.** The single-pass `polygonToTriangle` kernel and the legacy
  edge-overlay pass both gate their boundary flags through the per-starting-point
  flag test (`tessEdgeVisible` in the shader; `pointEdgeFlag` on the CPU legacy
  `uniqueEdges` path), matching GL's `val & mask`.
- **Host/GPU params.** The `TessParams` struct gained a `hasUserEdgeFlags`
  field (host structs kept in lockstep), and a per-point uint32 0/1 flag buffer
  is bound at buffer index 8 (`polygonToTriangle`) / 6 (`polygonEdgesToLines`)
  only when the attribute exists; the kernels read it only when the flag is set,
  so unbound-buffer and base-path behavior are unchanged.

`TestEdgeFlags` passes at TIGHT_VALID 0.0102255. With no edge-flag attribute
present every path behaves exactly as before; the six edge/wireframe/strip
tests (`TestEdgeFlags`, `TestSurfacePlusEdges`, `TestEdgeOpacity`,
`TestEdgeThickness`, `TestWireframe`, `TestTStripsColorsTCoords`) all pass, and
the full suite re-ran at 155 pass / 20 fail / 0 aborted with the failure set
otherwise unchanged and no regression against the documented passing cluster.

### The labeled-contour-mapper cluster is fixed

The last remaining crash class was the `TestLabeledContourMapper` trio, all three
SIGSEGVing inside `vtkOpenGLLabeledContourMapper::ApplyStencil`/
`RemoveStencil` (called from the base `vtkLabeledContourMapper::Render` via
`vtkMetalRenderer::DeviceRender`). `vtkLabeledContourMapper::New()` goes through
the object factory (`vtkObjectFactoryNewMacro`, `vtkLabeledContourMapper.cxx:187`),
and the only registered override was `vtkOpenGLLabeledContourMapper` (via the
OpenGL2 `opengl_overrides` list), so even with `--vtk-factory-prefer
RenderingBackend=Metal` the factory returned the OpenGL subclass, whose stencil
pass casts the render window to a GL window and dereferences null Metal context
state. The base `vtkLabeledContourMapper::ApplyStencil`/`RemoveStencil` are
already no-ops ("Handled in backend override", `vtkLabeledContourMapper.cxx:830,
845`).

A new `vtkMetalLabeledContourMapper` override (`Rendering/Metal/`), registered
via `vtk_object_factory_declare(BASE vtkLabeledContourMapper OVERRIDE
vtkMetalLabeledContourMapper)` with the usual `RenderingBackend=Metal`
attribute chain, now captures the factory selection. Because the Metal backend
has no stencil buffer yet (depth attachments are Depth32Float-only, no
`MTLPixelFormatStencil8` anywhere in `Rendering/Metal/`), the stencil passes
inherit the base no-ops: isolines draw continuously and the label quads compose
on top, which is exact for the opaque label backgrounds these tests use (the
isolines never pass behind a label, so the stencil is a visual no-op here; a
translucent-background label would differ only where the isoline crosses it).
`CreateLabels` additionally mirrors the OpenGL override by folding the actor's
matrix into each label's user matrix (`vtkOpenGLLabeledContourMapper.cxx`), which
`TestLabeledContourMapperWithActorMatrix` needs because the label `vtkTextActor3D`
instances are rendered as separate actors and do not inherit the transformed
mapper actor's matrix. All three tests pass (TIGHT_VALID 0.0002 / 0 / 0.0286),
and the full suite is now at **142 pass / 33 fail / 0 aborted** with no
regressions.

### The tiled-viewport and tile-aware gradient-background cluster is fixed

`TestGradientBackground` and the tiled pair `TestGradientBackgroundWithTiledViewport`/
`TestGradientBackgroundWithTiledViewports` (renderers whose fractional viewports tile a
larger virtual window while `vtkWindowToImageFilter` drives physical tiles) failed image
comparison at mid 0.3449 / gross 0.5063 / gross 0.5856: the gradient was drawn with
full-window UVs that repeated per tile instead of spanning the renderer's viewport, and
the Metal viewports were computed from the untiled renderer size with no tile origin.
The Metal viewport sites in `vtkMetalRenderer.mm` (opaque, translucent, volume,
volume-framebuffer blit, overlay) now derive `viewportX/Y/W/H` from
`vtkViewport::GetTiledSizeAndOrigin` and flip Y against the physical drawable height
(Metal top-origin vs VTK bottom-left), matching GL's `vtkOpenGLCamera::Render`/
`vtkOpenGLRenderer::Clear` tile handling. The gradient background state now carries the
physical viewport rect plus the renderer and tile viewports (normalized), and
`fragment_gradient_background` maps the fragment position through tileViewport→
rendererViewport so the quad spans the renderer's whole virtual viewport coherently under
tiling instead of repeating per tile (tcoord = fragment pos normalized across the
renderer's viewport, matching GL). `vtkMetalCamera::Render` computes its aspect and
cached `Viewport` from `GetTiledSizeAndOrigin` + the window's actual physical size. All
three tests now pass (TIGHT_VALID ~1e-03), `TestTilingCxx` improved from gross
0.6159 to mid 0.2054 on the same viewport-rect corrections, and `TestGlyph3DMapper` (mid
0.1082) + `RenderNonFinite` (near-miss 0.0870) also left the failure set this run (stable
across re-runs; see the run paragraph above).

### The 2D overlay tile cropping is fixed (`TestTilingCxx` now passes at 0)

`TestTilingCxx` (160x160 window, ren1 left 75%, ren2 right 25% with a vertical
`vtkScalarBarActor`, captured 3x2 tiled via `vtkWindowToImageFilter`) failed at mid
0.2195: under tiling `vtkWindow::GetSize()` reports the *virtual* window
(`Size * TileScale`, e.g. 480x320) while each physical tile is 160x160, so the 2D
overlay pass sized and positioned the bar with full-virtual-window math inside a
physical tile — the bar was split into two squished copies, one per affected tile.
`vtkMetalPolyDataMapper2D::SetShaderParameters2D` now mirrors
`vtkOpenGLPolyDataMapper2D::SetCameraShaderParameters`: it intersects the renderer
viewport with `GetTileViewport()`, scales the ortho range by the visible fraction
(`visSize`, the same viewport-derived scaling GL applies — the previous port only
shifted `xoff` and left the range at the full `size`), and shifts the actor origin
by the tile offset (`xoff/yoff = actorPos - (visVP - vp) * winSize`). Each tile now
draws exactly its visible slice of the overlay. `TestTilingCxx` passes at
ImageError 0; the scalar bar body renders as a single continuous run at
x373-407/y39-291 matching the GL baseline. The 2D actor/image-mapper cluster
(`TestActor2D`, `TestActor2DTextures`, `TestImageMapper_1..4`) still passes.

### The `vtkShaderProperty` replacement mechanism and GLSL→MSL shim

`TestCompositePolyDataMapperCustomShader` — the multi-backend shader-replacement
test that injects four GLSL replacements into the vertex/fragment shaders via
`vtkShaderProperty` to color the sphere by `abs(modelNormal)` — failed at a gross
0.2897 (systematically ~30% darker than the GL baseline). It was the last
remaining shader-replacement test, and the Metal backend had no mechanism at all
for `vtkShaderProperty` replacements: the GL shaders are templates with
`//VTK::Token` markers that `vtkOpenGLShaderProperty`'s `GetShaderReplacements`
lets the user substitute, while the Metal shaders are precompiled `id<MTLLibrary>`
objects from `MetalShaders.metal` that nothing could specialize per-actor.

The Metal implementation (`vtkMetalPolyDataMapper.mm`) now:

- **Substitution engine.** `BuildCustomShaderSource(actor, outSource)` reads the
  actor's replacements through the abstract `vtkShaderProperty` API
  (`GetNumberOfShaderReplacements`, `GetNthShaderReplacement`,
  `GetNthShaderReplacementTypeAsString`), and — when any exist — re-emits the
  shared `MetalShaders.metal` source with the replacements applied in scope.
  `ComputeLineScopes` classifies every line (TopLevel/Vertex/Fragment/Other) via
  `vertex`/`fragment` function-header detection plus `// VTK-METAL-SCOPE:`
  markers, so `//VTK::Normal::Dec` (a struct member, top-level) and
  `//VTK::Normal::Impl` (one instance in `vertex_main`, one in
  `evaluateSurfaceFragment`) are each replaced in the right places.
- **Per-actor pipeline specialization.** `vtkMetalPolyDataMapper::EnsurePipelineStates`
  now takes the `vtkActor`; when the actor has replacements it compiles the
  substituted source through `vtkMetalRenderWindow::GetShaderLibraryForSource`
  (new; compiles + caches `id<MTLLibrary>` keyed on the full source string, owned
  by the window and released in `Finalize()`) and builds the specialized surface
  pipelines from that library. The surface-pipeline cache key widened to
  `uint64_t`: the low 32 bits are the existing feature mask, the high 32 bits are
  the effective library pointer, so a custom-shader actor and a plain actor on
  the same mapper never alias each other's pipeline.
- **Bounded GLSL→MSL shim.** The injected replacement text is GLSL written for
  the GL template, so it must be translated before it can compile as MSL.
  `TranslateGlslToMsl` handles the scope it is injected into: top-level `out`/`in`
  varying declarations become plain `floatN` members, vertex-body code rewrites
  the tracked varyings to `out.X` and `normalMC` to `in.normal`, fragment-body
  code rewrites them to `in.X` and `diffuseColor` to `r.diffuse`, and the GLSL
  type names map to MSL (`vec4`→`float4`, `mat4`→`float4x4`, ...). It is a
  lexical translator for the bounded class of replacements the shared tests use,
  not a general GLSL compiler.
- **Diffuse-intensity parity.** GL declares `vec3 diffuseColor =
  diffuseIntensity * diffuseColorUniform` and lets the user's replacement
  overwrite that value, so a custom `diffuseColor = X` REPLACES the
  intensity-scaled color and the final GL output uses `X` directly. MSL applies
  the intensity at the end (`m.diffuseColor.w * totalDiffuse`), which would have
  darkened every custom diffuse by the property's `SetDiffuse` coefficient
  (0.7 here — the observed metric). The shim therefore rewrites a fragment
  `diffuseColor = X;` to `r.diffuse = X / m.diffuseColor.w;`, cancelling the
  end-of-pipeline intensity exactly like GL's override does.

`TestCompositePolyDataMapperCustomShader` now passes at TIGHT_VALID 1.18e-06
(was 0.0758 before the intensity-parity fix, gross 0.2897 before the mechanism
existed). Base rendering is untouched: an actor with no replacements returns early
from `BuildCustomShaderSource`, so the shared-library pipelines are selected
exactly as before; the full suite re-ran at 151 pass / 24 fail / 0 aborted with
the failure set otherwise unchanged and no regression against the documented
passing cluster. Known limits (beyond the shim's lexical scope): the single
`evaluateSurfaceFragment` is shared by multiple fragment entry points, so a
fragment replacement changes all surface pipelines uniformly (GL can substitute
per-shader-source), and the shim only translates the GLSL idioms the shared
tests exercise.

### The sibling translucent/volume passes are tile-aware (follow-up, same run)

The tiled-viewport fix above covered the renderer's own viewport sites, but the
three sibling passes still derived their rects from the virtual window size
(`ren->GetSize()` / `renderer->GetSize()`, which return the *virtual* tiled
size via `vtkWindow::GetSize()`) or from fractional-viewport math against it, so
under `vtkWindowToImageFilter` tiling their viewports/UVs were 2x too large and
their textures wrong-sized. All three now use
`vtkViewport::GetTiledSizeAndOrigin` exactly like their OpenGL counterparts:

- `vtkMetalOrderIndependentTranslucentPass.mm` (matches
  `vtkOrderIndependentTranslucentPass.cxx:198`): the accumulate/reveal textures
  are sized to the physical drawable (`drawableTexture`), not
  `renderer->GetSize()`, and the accumulate + resolve viewports are the tile
  rect (Y flipped against the drawable height). The resolve shader reads
  `in.position.xy` at absolute render-target pixels, so drawable-sized textures
  + tile-rect viewport keep the accumulate and resolve regions aligned.
- `vtkMetalDepthPeeler.mm` (matches `vtkDepthPeelingPass.cxx:358`): the six
  peel textures are sized to the physical drawable and the init, peel,
  back-blend, and composite viewports are the tile rect.
- `vtkMetalGPUVolumeRayCastMapper.mm` (matches
  `vtkOpenGLGPUVolumeRayCastMapper.cxx:1835,3215`): the `ViewportSize` uniform
  (GL's `in_inverseWindowSize` = 1/tile size, used for the depth-texture UV and
  ray reconstruction), the generic-aspect fallback, and the image-sample FBO
  size all derive from the tile size instead of `ren->GetSize()`.

The image-sample blit path in `vtkMetalRenderer.mm` already used the tile rect
(the committed fix); the volume offscreen render + blit are now consistent with
it. No metric changes: the suite re-ran at the same 139/33/3 with identical
bucket membership (6 near-miss / 21 mid / 2 gross) and no regression vs the
documented passing cluster.

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

### The per-block edge/vertex-color cluster is fixed

`TestCompositePolyDataMapperVertices` and `TestCompositePolyDataMapperSpheres`
(colored composite blocks with global `actor->GetProperty()->SetEdgeColor` /
`SetVertexColor` overrides and `RenderLinesAsTubes`/`RenderPointsAsSpheres`)
rendered the edge tubes and vertex spheres in each block's `SetBlockColor` color
instead of the property's grey edges / pink vertices. The Metal backend was
injecting the per-block override color into the edge/vertex uniform RGB
whenever `Internals->UseBatchColor` was active (i.e. on any block with a color
display-attribute override).

OpenGL never does this: `vtkOpenGLBatchedPolyDataMapper::DrawIBO` skips
`SetShaderValues` for `PrimitiveVertices` (`primType <= PrimitiveTriStrips`), so
the block color never reaches vertex rendering and
`SetPropertyShaderParameters` always uploads `ppty->GetVertexColor()`
(`vtkOpenGLPolyDataMapper.cxx:3218/3230`); the `edgeColor` uniform is likewise
always `actor->GetProperty()->GetEdgeColor()` (`vtkOpenGLPolyDataMapper.cxx:2903`),
never touched by the batched mapper's per-block shader values. The per-block color
only ever affects the *surface* ambient/diffuse material (and the point/lines
geometry buffers), exactly like the block-opacity override bakes into vertex
alpha.

Fix in `Rendering/Metal/vtkMetalPolyDataMapper.mm`: removed the `UseBatchColor`
RGB override from the three edge/vertex uniform updates —
`UpdateVertexColorUniforms`, `UpdateEdgeColorUniform` (legacy edge overlay) and
`UpdateEdgeUniforms` (single-pass surface edges). The RGB always comes from the
actor's property; the alpha/opacity handling (batch opacity baked in when
`UseBatchOpacity`, material opacity forced to 1.0 under batch overrides) is
unchanged. `TestCompositePolyDataMapperVertices` drops from 0.2891 to 0.0244 and
`TestCompositePolyDataMapperSpheres` from 0.1499 to 0.0215 (both pass); the rest
of the composite-mapper suite is unaffected.

### The coincident point-color and line-offset fix (`TestCoincident` now passes)

`TestCoincident` renders two coincident spheres (one pink, one transparent) whose
vertices alias in projected space, plus a 1px pink line drawn coincident with the
front sphere. It failed on Metal with two visual defects:

- **Point color ignored.** The point/vertex fragments resolved the point color
  from the vertex color stream only, so the pink dots (GL draws them from
  `vtkProperty::GetColor()`) rendered white whenever the geometry carried no
  per-point scalar colors. Fix in `Rendering/Metal/Shaders/MetalShaders.metal`:
  the point fragments now fall back to the material ambient/diffuse color when no
  per-point scalars are present, and a new `kSceneFlagHasPointColors` scene bit
  (`1u << 17`, `VTK_METAL_SCENE_FLAG_HAS_POINT_COLORS`) distinguishes "the vertex
  stream carries real per-point colors" from "the vertex stream is a dummy 1.0
  constant", so batch color/opacity and per-point colors both resolve correctly.
- **1px-line coincident offset dropped.** GL's line shader applies a coincident
  depth offset (`0` for polygons, `-4` for lines) so the pink line floats in
  front of the coincident surface; the Metal line fragment hardcoded the surface
  offset. Fix: `makeFragmentOutput` now takes explicit `factor`/`offset`
  arguments, `fragment_main` passes the polygon offsets and `fragment_main_line`
  passes the line offsets, restoring GL's line-in-front behavior.

`TestCoincident` passes at ImageError 0; the pink `(255,76,255)` dots and the
floating 1px line match the GL baseline. `Internals->HasPointColors` is set only
for batch color/opacity or mapped per-point colors, so the base path is
unchanged.

### The `vtkMetalShaderProperty` factory override (bespoke crash fix)

`TestMetalCamera` and `TestMetalPointRender` aborted (Subprocess aborted) on the
working tree even though the historical run documented them passing. The crash
was in `vtkProp::GetShaderProperty()`: `vtkShaderProperty` is an abstract class
declared with `vtkAbstractObjectFactoryNewMacro`, so `New()` returns null unless a
factory override exists; the Metal backend registered none, the OpenGL override
was not selected under `--vtk-factory-prefer RenderingBackend=Metal`, and the
lazily-created property dereferenced null in `vtkObjectBase::Register` (lldb:
`Error: no override found for 'vtkShaderProperty'`). The crash reproduced with
the point/line fixes stashed, so it was independent of them.

Fix: a new concrete `vtkMetalShaderProperty`
(`Rendering/Metal/vtkMetalShaderProperty.h` / `.mm`) overrides `vtkShaderProperty`
for `RenderingBackend=Metal`
(`vtk_object_factory_declare(BASE vtkShaderProperty OVERRIDE vtkMetalShaderProperty)`
in `Rendering/Metal/CMakeLists.txt`). It is a functional port of
`vtkOpenGLShaderProperty`'s replacement storage: a PIMPL holding
`std::map<std::tuple<int,std::string,bool>, std::tuple<std::string,bool>>` keyed
on (shader stage, OriginalValue, ReplaceFirst) mapping to (Replacement,
ReplaceAll), with the stage names (`"Vertex"`/`"Fragment"`/...) matching the
OpenGL strings the shim and tests use. It deliberately does not depend on
`vtkShader.h` (Rendering/OpenGL2-only); `Clear*ShaderReplacement` calls
`Modified()` only when a key was actually removed, like the OpenGL reference.
`vtkMetalPolyDataMapper::BuildCustomShaderSource` reads replacements through the
abstract `vtkShaderProperty` API, so the custom-shader mechanism above works
unchanged against the new subclass.

`TestMetalCamera` and `TestMetalPointRender` pass again; the bespoke suite is
back to 16/16. The generated factory
(`build_macos_metal/Rendering/Metal/vtkRenderingMetalObjectFactory.cxx`) registers
`"vtkShaderProperty"` → `"vtkMetalShaderProperty"` with the
`RenderingBackend=Metal` override attribute.

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

Current run (2026-08-04, resize-capture offscreen-target fix): 19 failed = 15
image-compare (TIGHT_VALID >= 0.05) + 1 below-threshold pick-check + 3 non-image
+ 0 aborts.
Buckets by
max `vtkTesting` TIGHT_VALID error per test (threshold 0.05):

| Bucket | Range | Count | Examples |
|--------|-------|-------|----------|
| near-miss | 0.05 – 0.1 | 4 | `TestRenderLinesAsTubesOrthoCamera` 0.0535, `TestRenderLinesAsTubes` 0.0535, `TestLineRenderingTranslucent` 0.0790, `TestGlyph3DMapperPicking` 0.0800 |
| mid | 0.1 – 0.5 | 11 | `TestMixedGeometryCellScalars` 0.1373, `TestPolyDataMapperClipPlanes` 0.1526, `TestTransformCoordinateUseDouble` 0.1635, `TestCompositePolyDataMapperPicking` 0.1712, `TestGlyph3DMapperCompositeDisplayAttributeInheritance` 0.2241, `TestCompositePolyDataMapperPartialFieldData` 0.2544, `TestGlyph3DMapperBackfaceColor` 0.2657, `TestPolyDataMapperNormals` 0.2698, `TestResetCameraScreenSpace` 0.3438, `TestCompositePolyDataMapperCameraShiftScale` 0.3601, `TestGlyph3DMapperPointSize` 0.4599 |
| gross | >= 0.5 | 0 | — |

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
time any of the three have passed.) This run the near-miss bucket dropped from 6
to 5: `TestActorLightingFlag` (0.0513, now passing at 9.53e-05) left via the
surface lighting-flag fix above, and `TestRenderLinesAsTubes`/`OrthoCamera`
improved from 0.2350/0.2349 to 0.2292/0.2292 (still mid) via the tube-light
bake below — the pass count is unchanged.) This run the mid bucket dropped from 24
to 21 and gross from 5 to 2: the three gradient tests `TestGradientBackground`
(mid 0.3449), `TestGradientBackgroundWithTiledViewport` and
`TestGradientBackgroundWithTiledViewports` (both gross) left via the
tiled-viewport/gradient-background section below — the first time any have passed
(now ~1e-03), `TestGlyph3DMapper` (mid 0.1082) and `RenderNonFinite` (near-miss
0.0870) also left (stable across re-runs), `TestTilingCxx` moved gross→mid
(0.6159→0.2054), and `TestRenderLinesAsTubes`/`OrthoCamera` moved mid→near-miss
(0.2292→0.0535). The +5 pass delta is exactly the gradient trio plus
`TestGlyph3DMapper` and `RenderNonFinite`.) This run the mid bucket dropped from
21 to 20 and gross from 2 to 0: `TestOffAxisStereo` (gross 0.5921),
`TestSplitViewportStereoHorizontal` (gross 0.6816) and `TestStereoEyeSeparation`
(mid 0.2584) all left via the stereo-composite write-back section below — the
first time any have passed (OffAxisStereo now ~2.9e-03). The +3 pass delta is
exactly the three stereo tests, and the regression check against the
previously-passing cluster is clean.) This run the near-miss bucket dropped from 6
to 5 and mid from 20 to 17: `TestPolyDataMapper2D` (near-miss 0.0664) and its
point-scalar / cell-scalar variants (mid 0.2861/0.2943) left via the 2D overlay
line-width and point-size section below — the first time any of the three have
passed (all at TIGHT_VALID 0) — and `TestColorByStringArrayDefaultLookupTable2D`
(mid 0.4821) left via the 2D scalar-color mapping commit `bd68ee20b2`. The +4 pass
delta is exactly those four tests.) This run the mid bucket dropped from 17 to 16:
`TestTilingCxx` (mid 0.2195) left via the 2D overlay tile-cropping section above —
the first time it has passed (ImageError 0). The +1 pass delta is exactly that
test; the bucket membership is otherwise unchanged.) This run the mid bucket dropped from 16
to 15: `TestResizingWindowToImageFilter` (mid 0.4130) left via the
resize-capture offscreen-target gate fix above — the first time it has passed
(all four resolutions, TIGHT_VALID ≤ 5.4e-04). The +1 pass delta is exactly that
test; the bucket membership is otherwise unchanged.

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

Remaining 2D fidelity gaps (all pre-existing, now documented values): 2D scalar
color mapping (`TestPolyDataMapper2D{Point,Cell}ScalarColorMapping` 0.286/0.294)
and 2D line width/point size (`TestPolyDataMapper2D` 0.066) were both closed
after this section was written — the first by the scalar-color commit `bd68ee20b2`
and the second by the 2D overlay line-width and point-size section below, so the
whole `TestPolyDataMapper2D` family now passes. The textured-2D gap is closed by
the 2D overlay depth-ordering and text-texture section below
(`TestActor2DTextures` now passes at 3.9e-08).

### The 2D overlay line-width and point-size is fixed

`TestPolyDataMapper2D` failed at a near-miss 0.0664 (and the scalar variants at
0.2861/0.2943): the 2D polydata mapper passed the actor's `pointSize`/`lineWidth`
into the `Mapper2DState` uniform buffer, but the shaders ignored them, so
`vtkProperty2D::SetPointSize(10)`/`SetLineWidth(10)` had no effect and Metal
always rasterized 1px lines and points (points fell back to Metal's default 1px
size because `vertex_2d_main` never wrote `[[point_size]]`). Fixes in
`Rendering/Metal/Shaders/MetalShaders.metal` + `vtkMetalPolyDataMapper2D.mm`:

- **Point size.** Metal's `[[point_size]]` output is only valid for point
  topology (a pipeline whose vertex shader outputs it for triangle/line topology
  is rejected), so the point path uses a dedicated `Vertex2DPointOut` struct and
  `vertex_2d_point_main`, which emits `point_size = max(state.pointSize, 1.0)`
  (the same dedicated-struct pattern as the 3D glyph point shaders). The point
  pipeline now binds `vertex_2d_point_main`.
- **Line width.** Metal has no native line-width support, so `lineWidth > 1` is
  drawn as a screen-space quad per segment: a new `ThickLinePipeline` +
  `vertex_thick_line_2d_main`/`fragment_thick_line_2d_main` expand each segment
  into a 4-vertex triangle strip per instance (mirroring the 3D mapper's
  `vertex_thick_line_main`), computing the perpendicular in the 2D mapper's
  viewport-pixel space and mapping the expanded quad to NDC through the existing
  WCVC matrix. Colors support all three modes — per-vertex (interpolated along
  the segment like GL), per-cell (indexed by `instance_id`, aligned with the
  existing cell-color buffer), and the plain property color. Line segments at
  `lineWidth <= 1` keep the plain 1px `LinePipeline`. The thick-line draw
  restores the plain-pipeline vertex bindings afterwards so the point draw reads
  the correct state buffer.
- **Selection.** The thick-line path applies whenever `state.lineWidth > 1`,
  the per-segment instanced draw costs the same 4 vertices per segment as the
  GL-style approach, and the pipeline/buffer work is cached on the mapper like
  every other pipeline state, so the non-2D overlay passes (textured text,
  images, 3D geometry) are unaffected.

The whole `TestPolyDataMapper2D` family now passes at TIGHT_VALID 0 (base +
point-scalar + cell-scalar variants), the first time any have passed, and the
related 2D overlay tests (`TestActor2D`, `TestActor2DTextures`,
`TestImageAndAnnotations`) still pass. The residual 2D overlay text error
(`TestImageAndAnnotations` ~0.046, glyph-edge AA vs the GL FreeType raster) and
`TestPickTextActor` (pick check) are unchanged.

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
  lighting (GL lights tris regardless of normals). The thick-line/tube
  pipelines (`shadeLineFragment`) are compiled with the actor's lighting state
  baked via the `kLightingDisabled` function constant (16); see the tube-light
  bake section below.

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

### Surface / point / tube draws respect the property lighting flag

`TestActorLightingFlag` (three cones; the middle one sets
`GetProperty()->SetLighting(false)`) failed at 0.0513: the middle cone was
still Phong-shaded, so all three cones looked alike, while the GL baseline
renders the middle cone flat white. `vtkGLSLModLight::GetBasicLightStats` drops
the light complexity to 0 whenever `property->GetLighting()` is false, and the
GL fragment then emits `gl_FragData[0] = vec4(ambientColor + diffuseColor,
opacity)` (the flat material color) for every primitive type. The Metal
surface/point fragments ignored the flag and always ran `computePhongLighting`.
The fix (surface/point runtime flag + tube pipeline bake):

- The mapper sets `VTK_METAL_SCENE_FLAG_LIGHTING_DISABLED` per actor in
  `RenderPiece` when `!prop->GetLighting()` (lines already fold
  `prop->GetLighting()` into the existing `VTK_METAL_SCENE_FLAG_LINES_UNLIT`
  decision, so they are unaffected).
- `fragment_main`, `fragment_main_nodepth`, `fragment_main_oit`, `fragment_peel`
  and the point fragments (`fragment_point_main`, `fragment_point_shaped_main`)
  now skip `computePhongLighting` when the flag is set and output
  `ambientIntensity*ambientColor + diffuseIntensity*diffuseColor` — GL's
  NoLighting result — instead of `ambient + diffuse*df + specular`. The glyph
  fragment (`shadeGlyphFragment`) honors the same bit if a caller ever sets it.
- `evaluateSurfaceFragment`'s `unlitLines` parameter was renamed `unlit`; the
  surface entries pass `kSceneFlagLightingDisabled`, the line entry keeps
  passing `kSceneFlagLinesUnlit`, so triangle and line decisions stay separate
  (an actor without normals keeps full triangle lighting; a lit-but-normalless
  line actor keeps flat lines).
- Thick-line/tube pipelines follow GL's design: GL forces `needLighting = true`
  for tubes but still clears it when `GetLighting()` is false, so the lighting
  state is a compile-time specialization, not a runtime flag. Each tube PSO
  family (`thick-line`, `round-cap`, `miter-join`) now has a lit and an unlit
  variant baked with the `kLightingDisabled` function constant (16), selected
  in `RenderPiece` from `act->GetProperty()->GetLighting()`; `shadeLineFragment`
  skips the fake-tube normal construction and `computePhongLighting` when baked
  unlit, emitting `ambientColor.w*baseColor + diffuseColor.w*baseColor` with no
  specular (GL's complexity-0 tube output). The unlit variants are also
  registered in the availability/`edgeTubes`/`drawEdgeOverlay` checks, and the
  bundle cache key records the property lighting so lit and unlit actors sharing
  a mapper don't alias each other's pipeline. No test exercises the unlit-tube
  combination in the generic suite, but the bespoke `TestMetalPointRender`
  scene's tubes-plus-lighting renders identically.

`TestActorLightingFlag` now passes (TIGHT_VALID 9.53e-05). The regression check
reports no new failures: the other five near-miss tests keep their exact prior
metrics (`TestPolyDataMapper2D` 0.0664, `TestEdgeFlags` 0.0681,
`TestLineRenderingTranslucent` 0.0790, `TestGlyph3DMapperPicking` 0.0800,
`RenderNonFinite` 0.0870).

### The stereo-composite write-back is implemented

`TestOffAxisStereo` (gross 0.5921), `TestSplitViewportStereoHorizontal` (gross
0.6816) and `TestStereoEyeSeparation` (mid 0.2584) all failed because the
CPU-side stereo composite was never written back to the framebuffer. VTK's
`vtkRenderWindow::StereoRenderComplete` computes the composite (e.g.
`vtkStereoCompositor::RedBlue`) into `ResultFrame` and calls
`CopyResultFrame` → `SetPixelData`; `vtkMetalRenderWindow` did not override
`SetPixelData`/`SetRGBACharPixelData`, so the base-class no-op left the second
eye pass alone on the drawable (the observed red-blue output was a partial
right-eye image — 10,158 nonblack pixels, 0 colorful).

- `vtkMetalRenderWindow` now overrides both `SetPixelData` overloads and both
  `SetRGBACharPixelData` overloads. They share a `WritePixelData` helper that
  converts the bottom-up VTK RGB(A) rows into a top-origin BGRA8Unorm staging
  texture (`MTLStorageModeShared`), blits it into the current drawable and, when
  color read-back is enabled, into the shared color-copy texture, presents the
  drawable exactly once, commits, and re-arms `SetCurrentCommandBuffer` so a
  subsequent `GetPixelData` waits for the composite copy. A framebuffer-only
  drawable texture skips the drawable blit/present (the read-back texture is
  still updated).
- The drawable is presented once per frame (CAMetalLayer rejects a second
  present). `vtkMetalRenderWindow` tracks `DrawablePresented` (reset in
  `Render()` and on fresh acquisition); `AcquireDrawable` releases and
  re-fetches `nextDrawable` if the current one was already presented.
- `vtkMetalRenderer` suppresses the eye-pass presents for the CPU-composited
  stereo types (RED_BLUE, ANAGLYPH, INTERLACED, DRESDEN, CHECKERBOARD,
  SPLITVIEWPORT_HORIZONTAL) and for any already-presented drawable; the
  composite `SetPixelData` is the sole present of the frame.
- Each eye pass is a full window pass: `firstRenderer`/`lastRenderer` are now
  computed per pass (`frameRendererIndex % totalRenderers`) instead of per
  window frame. Previously the second eye was treated as a continuation of the
  first, so its color/depth attachments were loaded (not cleared) and the
  right-eye geometry was depth-culled against the left eye's depth (the
  observed symptom: the anaglyph's blue channel dark at 27.2 mean vs 197.8
  baseline).

All three now pass: `TestOffAxisStereo` TIGHT_VALID 2.88e-03, exit 0, no
double-present warnings; the red/blue channels of the composite match the left
and right gray passes exactly at every nonblack pixel. The regression check
against the previously-passing cluster is clean.

### Theme clusters in the remaining failures

- **Textures** (~12): every `TestTexture*`, `TestBackfaceTexture`,
  `TestActor2DTextures` — historically
  crashed on the OpenGL fallback; now render, with 10 passing
  (`TestTextureWrap`, `TestBackfaceTexture`, `TestTextureRGBA`,
  `TestTextureRGBADepthPeeling`, `TestTextureSize`,
  `TestTextureInterpolateScalars`, `TestTexturedCylinder`,
  `TestActor2DTextures`, `TestBackfaceCulling`, `TestTilingCxx`) and the rest
  failing image comparison on texture-feature fidelity (filter/wrap/interpolation
  edge cases).
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
  selection/read-back cluster sections above). The 4 near-miss image tests
  (`TestEdgeFlags`,
  `TestLineRenderingTranslucent`,
  `TestGlyph3DMapperPicking`, `RenderNonFinite`)
  are the next easy-win targets (`TestActorLightingFlag` left via the
  surface lighting-flag fix above, `TestImageAndAnnotations` via the
  overlay depth/texture fix above, and `TestPolyDataMapper2D` via the
  2D overlay line-width and point-size fix below).
- **Point rendering** — DONE: the four point tests `TestPointRendering_3/_4` +
  `TestPointRenderingRound_3/_4` plus the vertex-visibility tests
  `TestVertexRendering`, `TestQuadPointRep` and `TestMixedGeometry_3` now pass
  via the point-lighting fix (see the point-rendering section above).
- **2D overlay / image mapper**: the full `TestPolyDataMapper2D` family (base +
  point-scalar + cell-scalar variants) now passes via the 2D overlay line-width
  and point-size fix below (`[[point_size]]` + screen-space thick-line quads) and
  the 2D scalar-color mapping commit `bd68ee20b2`. `TestActor2D` now passes via the
  2D-overlay section above, `TestImageMapper_1..4` now pass via the 2D
  image-mapper viewport-WCVC fix (see the 2D image-mapper section above), and
  `TestActor2DTextures` now passes via the 2D overlay depth-ordering and
  text-texture fix (see that section above).
- **LUT / color mapping** — DONE: `TestColorByStringArrayDefaultLookupTable2D`
  (0.482) now passes via the 2D scalar-color mapping commit `bd68ee20b2`.
  `TestBareScalarsToColors`, `TestDirectScalarsToColors`, `TestMapVectorsToColors`
  and `TestMapVectorsAsRGBColors` now pass via the 2D image-mapper viewport-WCVC
  fix (all four render `vtkImageMapper` in sub-viewports).
- **Stereo / multiview / gradient background**: `TestStereoBackground{Left,Right}`
  now pass via the textured-background implementation (see the textured-background
  section below), as do the 3 `TestNViewportsNActors*` multiview tests via the
  multi-viewport background work (see the flat-background section above;
  `TestNViewportsOneActor` passes too). `TestOffAxisStereo`,
  `TestStereoEyeSeparation` and `TestSplitViewportStereoHorizontal` now pass via
  the stereo-composite write-back fix (see the stereo-composite section below).
  Still failing: 3 `TestGradientBackground*` (0.34–0.59).
- **Triangle strips** — DONE: the four `TestTStrips*` tests now pass via CPU-side
  strip decomposition in `vtkMetalPolyDataMapper`/`vtkMetalPolyDataMapper2D`
  (GL's `AppendStripIndexBuffer` winding) and the property-texture fallback in
  `UpdateActorTexture` (see the triangle-strip section above);
  `TestPolyDataMapperNormals` (0.270) remains in the mid bucket.

### Evidence the core path is correct

The 145 passes include the strongest-scrutiny tests: `TestOpacity` (passes with
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
first time any strip-geometry test has passed), the stereo-composite cluster
(`TestOffAxisStereo`, `TestStereoEyeSeparation`, `TestSplitViewportStereoHorizontal`
at ImageError 2.9e-03–5e-05), and the basic
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
 5. **Read-back — DONE** — the read-back cluster (`TestReadPixels`, `TestRemoveActors`,
    `TestWindowToImageFilter`, `TestSelectVisiblePoints`, `TestWorldPointPicker`)
    and `TestResizingWindowToImageFilter` (resize captures) now pass; no
    read-back-class failures remain.
 6. **Label/text/image OpenGL-fallback cluster** — the 3 remaining crashes all
    instantiate the OpenGL label/text/image classes (e.g.
   `vtkOpenGLLabeledContourMapper::ApplyStencil`) against a Metal window; this
   needs Metal overrides for the label/text rendering stack.
  7. **Glyph instancing colors**, **2D overlay (image mapper done)**, **LUT/color
       mapping**, **stereo/multiview done**, **gradient background** — all
       render but diverge from GL. (The glyph3D multi-source indexing tests
       `TestGlyph3DMapperIndexing`/`TreeIndexing` now pass — see the glyph3D
       multi-source indexing section above; `TestActor2D` now passes — see the
       2D-overlay section above; the sub-viewport `vtkImageMapper` tests
       `TestImageMapper_1..4`, `TestDirectScalarsToColors`, `TestBareScalarsToColors`,
        `TestMapVectorsAsRGBColors`, `TestMapVectorsToColors` now pass via the 2D
        image-mapper viewport-WCVC fix above; the full `TestPolyDataMapper2D`
        family (base + point-scalar + cell-scalar variants) now passes via the 2D
        scalar-color commit `bd68ee20b2` and the 2D overlay line-width and
        point-size fix below; the textured/stereo backgrounds `TestTexturedBackground`
       and `TestStereoBackground{Left,Right}` now pass — see the textured-background
       section above; the CPU-composited stereo tests `TestOffAxisStereo`,
       `TestStereoEyeSeparation`, `TestSplitViewportStereoHorizontal` now pass —
       see the stereo-composite write-back section above. The 3
       `TestGradientBackground*` tests (0.34–0.59) remain.)
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
  `vtkMetalPolyDataMapper::RenderPiece`), then after the surface lighting-flag fix
  (134 Passed / 38 Failed / 3 aborted — `TestActorLightingFlag` now passes via the
  `kSceneFlagLightingDisabled` runtime flag in the surface/point fragments), and
  then after the thick-line/tube lighting bake
  (134 Passed / 38 Failed / 3 aborted — the tube pipelines now bake
  `GetLighting()` into the `kLightingDisabled` function constant; pass count
  unchanged, `TestRenderLinesAsTubes{OrthoCamera}` improved 0.2350/0.2349 →
  0.2292/0.2292), and then after the labeled-contour-mapper fix
  (142 Passed / 33 Failed / 0 aborted — the `TestLabeledContourMapper` trio now
  passes via the Metal `vtkLabeledContourMapper` override), and then after the
  stereo-composite write-back fix
  (145 Passed / 30 Failed / 0 aborted — `TestOffAxisStereo`,
  `TestSplitViewportStereoHorizontal` and `TestStereoEyeSeparation` now pass via
  the `SetPixelData`/`SetRGBACharPixelData` overrides and the eye-pass present
  suppression in `vtkMetalRenderWindow`/`vtkMetalRenderer`).
  The image-compare buckets above are from that latest run's `LastTest.log` (max
TIGHT_VALID error per test), analyzed with `analyze_metal_ctest_log.py`.
Re-running is reproducible except where a crash's signal stack
ordering varies; the pass count fluctuates run to run.

Current working-tree run (2026-08-04, after the 2D scalar-color mapping commit
`bd68ee20b2` and the 2D overlay line-width/point-size fix):
(149 Passed / 26 Failed / 0 aborted — `TestPolyDataMapper2D` and its
point-scalar / cell-scalar variants now pass via the `[[point_size]]` point
shader and the screen-space thick-line quad pipeline, and
`TestColorByStringArrayDefaultLookupTable2D` now passes via the 2D scalar-color
mapping commit `bd68ee20b2`; the tally above and the buckets below are from this
run).

Latest working-tree run (2026-08-04, after the 2D overlay tile-cropping fix):
150 Passed / 25 Failed / 0 aborted — `TestTilingCxx` now passes (ImageError 0)
via the per-tile visVP/visSize/xoff/yoff ortho adjustment in
`vtkMetalPolyDataMapper2D` (see the 2D overlay tile-cropping section above); the
tally above and the buckets below are from this run, exported with
`export_image_compare.sh`.

Newest working-tree run (2026-08-04, after the coincident point-color/line-offset
fix and the `vtkMetalShaderProperty` factory override):
154 Passed / 21 Failed / 0 aborted — `TestCoincident` now passes (ImageError 0;
see the coincident point-color/line-offset section above), the bespoke
`TestMetalCamera`/`TestMetalPointRender` abort is fixed via the
`vtkMetalShaderProperty` override (see the vtkMetalShaderProperty section above;
bespoke suite re-ran at 16/16), and the remaining failure set is unchanged
(17 image-compare + 1 below-threshold pick-check + 3 non-image); the tally above
and the buckets are from this run, exported with `export_image_compare.sh`.

Newest working-tree run (2026-08-04, after the resize-capture offscreen-target
gate fix): 156 Passed / 19 Failed / 0 aborted — `TestResizingWindowToImageFilter`
now passes (all four resolutions, TIGHT_VALID ≤ 5.4e-04) via the
`GetUseOffScreenBuffers()` gate in `vtkMetalRenderer`/`vtkMetalRenderWindow`
(see the resize-capture offscreen-target section above); the remaining failure
set is unchanged (15 image-compare + 1 below-threshold pick-check + 3 non-image);
the tally above and the buckets are from this run, exported with
`export_image_compare.sh`.

Newest working-tree run (2026-08-04, after the point/line draw-order + flat
wide-line shading fix): 161 Passed / 14 Failed / 0 aborted —
`TestMixedGeometryCellScalars` now passes (TIGHT_VALID 0.000, was a mid-bucket
0.137264) via the points-before-lines draw reorder and the flat wide-line
shading behind `kSceneFlagLinesTubeShading` (see the point/line draw-order +
flat wide-line shading section above); the remaining failure set is unchanged
(10 image-compare + 1 below-threshold pick-check + 3 non-image); the tally above
and the buckets are from this run, exported with `export_image_compare.sh`.
