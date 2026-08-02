# Metal Testing State

Snapshot of the state of testing for the Metal rendering backend. Historical
run: `metal-ios` branch head `bc4e9d93cd` on this repo's Apple M1 Mac mini
(Macmini9,1, macOS 14.8). Current working-tree run: this repo's Apple M2
MacBook Air (Mac14,2, macOS 15.7.5), after the texture-cluster fixes described
below. Both arm64 Release builds in `build_macos_metal` (Ninja,
`VTK_MODULE_ENABLE_VTK_RenderingMetal=YES`).

Two test surfaces exist:

1. A bespoke Metal suite under `Rendering/Metal/Testing/` (13 image-baseline
   regression tests + a HeaderTest + a dual-backend visual-parity harness).
   Status: **all 15 pass** (unchanged from the historical run).
2. The generic multi-backend suite in `Rendering/Core/Testing/Cxx/`, which
   registers the same ~175 tests once per backend and was wired up for Metal
   through object-factory overrides (`--vtk-factory-prefer
   RenderingBackend=Metal`). Historical status: **55 pass / 120 fail (33
   crash)**. Current working-tree status: **61 pass / 95 fail (19 crash)** — the
   14 OpenGL-texture-fallback crashes are fixed by the `vtkMetalTexture` factory
   override (they now render; 5 of them pass outright), and no new crashes were
   introduced. The pass count measures 55–61 across runs (run-to-run flakiness).

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

### Tally (working tree, rerun 2026-08-02 on the M2 MacBook Air,
`ctest -R "RenderingCoreCxx-Metal" -j 8`)

```
175 tests:  61 Passed  95 Failed (incl. image/pick fails)  19 "Subprocess aborted"
```

(Historical run at commit `bc4e9d93cd`: 55 Passed / 87 Failed / 33 aborted.)

### The texture cluster is fixed

The largest historical crash class (OpenGL-texture fallback: `vtkTexture::New()`
returned `vtkOpenGLTexture`, whose `Load()` called
`vtkTextureObject::SetContext(vtkOpenGLRenderWindow*)` on a Metal window →
SIGSEGV) is resolved by `vtkMetalTexture` (`Rendering/Metal/vtkMetalTexture.h`),
a `vtkTexture` factory override registered with `RenderingBackend=Metal` whose
`Load()` is a no-op because the Metal poly-data mapper uploads actor textures
itself (`vtkMetalPolyDataMapper::UpdateActorTexture`). As a result
`TestTextureWrap`, `TestBackfaceTexture`, `TestTextureRGBA`,
`TestTextureRGBADepthPeeling`, `TestTextureSize` now **pass**; the other texture
tests (`TestActor2DTextures`, `TestBackfaceCulling`, `TestImageAndAnnotations`,
`TestPickTextActor`, `TestRenderLinesAsTubes{OrthoCamera}`, `TestTexturedCylinder`,
`TestTextureInterpolateScalars`, `TestTilingCxx`) render without crashing but
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

### Image-compare failures

Current run: 95 image/pick failures = the historical 87 (below) plus 8
texture-cluster tests that used to crash and now render with image differences.
Of the 87: 81 have a `vtkTesting` `ImageError`; 6 fail without an image compare
(see below). Buckets by thresholded error (threshold 0.05):

| Bucket | Range | Count | Examples |
|--------|-------|-------|----------|
| near-miss | 0.05 – 0.1 | 8 | `TestActorLightingFlag` 0.051, `TestEdgeFlags` 0.068, `TestQuadPointRep` 0.069, `TestMixedGeometry_3` 0.070, `TestVertexRendering` 0.072, `TestLineRenderingTranslucent` 0.079, `TestGlyph3DMapperPicking` 0.080, `TestMixedGeometryCellScalars` 0.092 |
| mid | 0.1 – 0.5 | 44 | `TestSurfacePlusEdges` 0.104, `TestGlyph3DMapperIndexing` 0.155, `TestCompositePolyDataMapperPicking` 0.176, `TestWireframe` 0.239, `TestPolyDataMapper2D` 0.235, `TestCoincident` 0.334, `TestCompositePolyDataMapperCustomShader` 0.385, `TestColorByStringArrayDefaultLookupTable2D` 0.482 |
| gross | >= 0.5 | 29 | `TestMapVectorsToColors` 0.962, `TestBareScalarsToColors` 0.925, `TestImageMapper_1..4` 0.86–0.92, `RenderNonFinite` 0.913, `TestStereoBackground{Left,Right}` 0.887, `TestGradientBackground*` 0.51–0.79, `TestAxesActor` 0.612, `TestPolyDataMapperNormals` 0.552 |

The 6 non-image failures are all selection/read-back checks:
`TestHardwareSelector`, `TestPointSelection`, `TestPointSelectionWithCellData`,
`TestSelectVisiblePoints` (selection results wrong), `TestWorldPointPicker`
(image matches, pick check fails), `TestReadPixels` (read-back reports an
error; the test's `ERR|` regex matched).

### Crashes (19; all pre-existing classes, none from the texture cluster)

Signal-level crashes are reported as `Subprocess aborted` with a `Caught
SIGSEGV` line but no backtrace in the ctest log, so crash causes were
re-derived from the earlier run's full signal stacks. Current-run attribution:

| Cause | Count | Tests |
|-------|-------|-------|
| `vtkMetalPolyDataMapper::BuildGeometryBuffers` on the composite/legacy path | 8 | `TestActor2D`, `TestBlockOpacity`, `TestCompositePolyDataMapper`, `TestCompositePolyDataMapper{BlockOpacities,Pickability,Spheres,ToggleScalarVisibilities,Vertices}` |
| SIGSEGV with no backtrace; prior stack analysis attributes these to the OpenGL factory fallback in label/text/image rendering | 9 | `TestFollowerPicking`, `TestInteractorStyleImageProperty`, `TestLabeledContourMapper`, `TestLabeledContourMapperNoLabels`, `TestLabeledContourMapperWithActorMatrix`, `TestResizingWindowToImageFilter`, `TestTranslucentImageActor{AlphaBlending,DepthPeeling}`, `TestWindowToImageFilter` |
| `CreateMultisampleAttachments` (MSAA) | 1 | `TestOpacityMSAA` |
| `vtkMetalHardwareSelector` selection path (`SelectionChanged`, "Color buffer depth must be at least 8 bit") | 1 | `TestAreaSelections` |

The 14-test OpenGL-texture-fallback crash row from the historical tally is gone
(fixed by `vtkMetalTexture`, see above). ~8 of the remaining 19 crashes are
genuine Metal bugs (composite-mapper buffer building, MSAA attachments, area
selection); the other ~9 are the label/text/image OpenGL-fallback class that the
`vtkMetalTexture` fix did not cover (no `vtkTexture` involved).

### Theme clusters in the 95 failures

- **Textures** (~16): every `TestTexture*`, `TestBackfaceTexture`,
  `TestTexturedCylinder`, `TestTilingCxx`, `TestActor2DTextures` — historically
  crashed on the OpenGL fallback; now render, with 5 passing and the rest
  failing image comparison on texture-feature fidelity (filter/wrap/interpolation
  edge cases, textured-cylinder seams).
- **Composite mapper** (~20): 8 crashes in `BuildGeometryBuffers` +
  `TestCompositePolyDataMapper{Scalars,CellScalars,Picking,PartialFieldData,
  OverrideScalarArray,OverrideLUT,CameraShiftScale,CustomShader,NaNPartial,
  MixedGeometry*,BlockTextures}` — the largest failing cluster.
- **Glyph instancing** (~9): `TestGlyph3DMapper{Arrow,BackfaceColor,Indexing,
  OrientationArray,Picking,PointSize,QuaternionArray,TreeIndexing,
  CompositeDisplayAttributeInheritance}` fail 0.15–0.6.
- **Selection/picking** (~6): `TestHardwareSelector`, `TestPointSelection*`,
  `TestSelectVisiblePoints`, `TestAreaSelections` (crash), `TestWorldPointPicker`.
- **2D overlay / image mapper**: `TestPolyDataMapper2D` (0.235),
  `TestPolyDataMapper2D{Point,Cell}ScalarColorMapping` (0.236/0.246),
  `TestImageMapper_1..4` (0.86–0.92), `TestActor2D` (crash).
- **LUT / color mapping** (~5): `TestBareScalarsToColors` 0.925,
  `TestDirectScalarsToColors` 0.696, `TestMapVectorsToColors` 0.962,
  `TestMapVectorsAsRGBColors` 0.899, `TestColorByStringArrayDefaultLookupTable2D` 0.482.
- **Stereo / multiview / gradient background**: `TestOffAxisStereo`,
  `TestStereoBackground{Left,Right}`, `TestStereoEyeSeparation`,
  `TestSplitViewportStereoHorizontal`, the 5 `TestNViewports*`, and 3
  `TestGradientBackground*` (0.51–0.79).
- **TStrips** (4) and `TestPolyDataMapperNormals` (0.552) fail 0.4–0.55.

### Evidence the core path is correct

The 61 passes include the strongest-scrutiny tests: `TestOpacity` (passes with
the `TIGHT_VALID` metric — the Lab-space color path matches GL to
`0.00038`), `TestOSConeCxx`, `TestMace`, `TestTranslucentLUTAlphaBlending`,
`TestTranslucentLUTDepthPeeling`, `TestScalarModeToggle`,
`TestPointRendering_{1,2,Round_1,Round_2}`, and the basic Glyph3D,
`FrustumClip`, `RGrid`, `TestQuad`, and composite-mapper partial-point-data
tests. `Rendering/Metal/Testing/OpenGLComparison.md` shows every bespoke scene
now matches OpenGL to a thresholded error of 0.000 (0.005 for volume). Failures
cluster in features Metal still implements incompletely (see below), not in the
fundamental geometry/lighting/color path.

---

## 3. Known gaps / next steps (highest value first)

1. **Texture support — DONE** — `vtkMetalTexture` (`Rendering/Metal/vtkMetalTexture`)
   now overrides `vtkTexture` for `RenderingBackend=Metal` (no-op `Load`; the
   poly-data mapper uploads actor textures in `UpdateActorTexture`). This
   eliminated all 14 texture-fallback crashes; 5 texture tests now pass and the
   rest fail image comparison instead of SIGSEGV. Follow-ups: border-color and
   filter/mipmap fidelity (`TestTextureInterpolateScalars`,
   `TestTexturedCylinder`), and renderers that bypass the poly-data mapper
   (2D image/text in the label cluster).
2. **Composite mapper crash** — `vtkMetalPolyDataMapper::BuildGeometryBuffers`
   crashes on the composite/legacy path (`vtkCompositePolyDataMapperDelegator`
   interaction); ~8 crash tests plus ~12 image-fails.
3. **MSAA attachments** — `CreateMultisampleAttachments`; `TestOpacityMSAA`.
4. **Hardware selector** — selection path (`TestAreaSelections` crash,
   `TestHardwareSelector`/`TestPointSelection*`/`TestSelectVisiblePoints`
   wrong results, `TestWorldPointPicker` pick check).
5. **Read-back** — `TestReadPixels` errors; `TestWindowToImageFilter` /
   `TestResizingWindowToImageFilter` crash.
6. **Glyph instancing colors**, **2D overlay + image mapper**, **LUT/color
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
`bc4e9d93cd` (M1 Mac mini); working-tree tally from the same date rerun on the
M2 MacBook Air with the texture-cluster changes present. The image-compare
buckets and theme clusters below are preserved from the historical analysis of
the 87 image/pick failures (that set is unchanged, minus the 8 texture tests
now counted there). Re-running is reproducible except where a crash's signal
stack ordering varies; the pass count fluctuates 55–61 run to run.
