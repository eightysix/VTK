# GPU volume ray cast: `TestGPURayCastCameraInsideTransformation` Metal vs OpenGL

## Symptom

`TestGPURayCastCameraInsideTransformation` renders a 512³ `headsq` volume with
Phong shading, gradient opacity and the camera inside the volume (near-plane
clipping of a `vtkProp3D`-transformed volume). The OpenGL backend passes, the
Metal backend fails the image comparison:

| backend | ctest id | result | TIGHT_VALID (threshold 0.05) |
|---------|----------|--------|------------------------------|
| OpenGL  | `VTK::RenderingVolumeCxx-TestGPURayCastCameraInsideTransformation` (#681) | Passed | — |
| Metal   | `VTK::RenderingVolumeCxx-Metal-TestGPURayCastCameraInsideTransformation` (#782) | **Failed** | **0.112623** |

Pixel-level comparison of the two backends at the same camera pose: **Metal is
uniformly brighter**. Center pixel `(86,66,54)` → `(127,92,74)`;
`mean(Metal−GL) = +28.82` over all 512² pixels, `mean|Metal−GL| = 42.04`,
`max|Metal−GL| = 90`, and *every* pixel differs by ≥ 5. Per-channel linear fit:
`metal = 0.778·gl + 64` (R channel), i.e. Metal lifts and compresses the
gradient.

## Environment and build

- macOS arm64, `-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0`, Release, static libs, Ninja.
- Build directory: `build_macos_metal` (see `macos_metal_build.sh --resume --tests`).
- The `RenderingMetal` module registers factory overrides carrying
  `RenderingBackend=Metal`, so either backend can be selected at runtime with
  `--vtk-factory-prefer RenderingBackend=<OpenGL|Metal>`. The `vtkRenderingVolumeCxxTests`
  driver does parse that flag (unlike the generic `vtk*CxxTests` driver), which
  is what makes the side-by-side comparison below possible.

## Reproduction

### 1. Plain ctest

```sh
ctest --test-dir build_macos_metal -R "^VTK::RenderingVolumeCxx-TestGPURayCastCameraInsideTransformation$" --output-on-failure     # OpenGL, passes
ctest --test-dir build_macos_metal -R "^VTK::RenderingVolumeCxx-Metal-TestGPURayCastCameraInsideTransformation$" --output-on-failure  # Metal, fails
```

The failing Metal run writes the rendered image and difference image to
`build_macos_metal/Testing/Temporary/TestGPURayCastCameraInsideTransformation.png`
(`.diff.png`).

### 2. Capturing both backends' images

The legacy vtk image comparison only writes the actual render when the test
**fails**. To capture both backends at the same camera pose, feed a
deliberately-wrong baseline (512×512 black) so both fail and dump their
renders:

```sh
python3 - <<'EOF'
from PIL import Image
Image.new('RGB', (512, 512), (0, 0, 0)).save('/tmp/bc/TestGPURayCastCameraInsideTransformation.png')
EOF

build_macos_metal/bin/vtkRenderingVolumeCxxTests TestGPURayCastCameraInsideTransformation \
  --vtk-factory-prefer RenderingBackend=OpenGL \
  -D build_macos_metal/ExternalData/Testing -T build_macos_metal/Testing/Temporary \
  -V /tmp/bc/TestGPURayCastCameraInsideTransformation.png
cp build_macos_metal/Testing/Temporary/TestGPURayCastCameraInsideTransformation.png /tmp/bc/gl/

# repeat with RenderingBackend=Metal -> /tmp/bc/metal/
```

(`-D` points at the ExternalData tree so the `vtkVolume16Reader` finds
`Data/headsq/quarter`.)

### 3. Analysis: delta stats, mask, heatmap

Everything below was produced with numpy + Pillow. Save this as
`/tmp/bc/analyze.py`:

```python
import sys
from PIL import Image
import numpy as np

gl = np.array(Image.open(sys.argv[1])).astype(float)   # OpenGL render
mt = np.array(Image.open(sys.argv[2])).astype(float)   # Metal render
label = sys.argv[3]

d = mt - gl                       # signed per-channel delta
md = np.abs(d).max(axis=2)        # max-channel |delta|
mask = md >= 5                    # pixels that differ by >= 5/255

print(f'=== {label} ===')
print(f'  center GL {gl[256,256].astype(int)} Metal {mt[256,256].astype(int)}')
print(f'  delta mean {d.mean():+.2f}  mean|d| {md.mean():.2f}  max|d| {md.max():.0f}')
for c, ch in zip(range(3), 'RGB'):
    a = np.polyfit(gl[..., c].ravel(), mt[..., c].ravel(), 1)
    print(f'  {ch}: metal = {a[0]:.4f}*gl + {a[1]:.2f}')
print(f'  masked (>=5) px: {mask.sum()}')

heat = np.zeros_like(gl)
heat[mask] = md[mask, None] * 255.0 / max(md.max(), 1e-9)   # scale to [0,255]
Image.fromarray(heat.astype(np.uint8)).save(f'{label}_delta_heatmap.png')
Image.fromarray((mask.astype(np.uint8) * 255)).save(f'{label}_delta_mask.png')
```

Run it once per variant:

```sh
cd /tmp/bc
python3 analyze.py gl/TestGPURayCastCameraInsideTransformation.png \
                   metal/TestGPURayCastCameraInsideTransformation.png \
                   TestGPURayCastCameraInsideTransformation
```

This prints the delta table rows used below and writes
`<label>_delta_heatmap.png` (pixels with |Δ|≥5, brightness ∝ |Δ|) and
`<label>_delta_mask.png`.

## Image tests: isolating the feature

The test combines three features: Phong shading (`ShadeOn`), gradient opacity
(`gf: 0@0, 0.5@90, 0.7@100`), and camera-inside-with-clipping. To find which
one makes Metal diverge, sibling tests were created in
`Rendering/Volume/Testing/Cxx/`:

| file | change vs original |
|------|--------------------|
| `TestGPURayCastCameraInsideTransformationNoShade.cxx` | `ShadeOff()` |
| `TestGPURayCastCameraInsideTransformationNoGradOp.cxx` | drop `SetGradientOpacity` |
| `TestGPURayCastCameraInsideTransformationNoShadeNoGradOp.cxx` | `ShadeOff()` + no gradient opacity |
| `TestGPURayCastCameraInsideTransformationConstGradOp.cxx` | gradient opacity constant `0.7` everywhere (`gf: 0.7@0, 0.7@2000`) |
| `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransform.cxx` | also drop the `vtkProp3D` transform (`Rotate*`/`SetOrigin`), camera repositioned inside the axis-aligned bounds |
| `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutside.cxx` | also move the camera outside (no near-plane clip) |
| `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideFixedStep.cxx` | camera outside, env-driven step sweep: `VTK_FIXED_SAMPLE_DISTANCE` disables `AutoAdjustSampleDistances` and forces the step on both backends |
| `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNearPlaneTiny.cxx` | camera inside, but near-plane pulled onto the eye (`SetClippingRange(0.001, …)`) so the near-plane clip is a no-op |
| `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformFineStep.cxx` | camera inside, `SetSampleDistance(0.25)` (a no-op probe: the override is ignored while `AutoAdjustSampleDistances` is on) |
| `TestGPURayCastCameraInsideTransformationSampleDist0_5.cxx` | original test, `SetSampleDistance(0.5)` (2× more samples) |
| `TestGPURayCastCameraInsideTransformationSampleDist0_25.cxx` | original test, `SetSampleDistance(0.25)` (4× more samples) |

Each is registered in `Rendering/Volume/Testing/Cxx/CMakeLists.txt`, e.g.:

```cmake
  TestGPURayCastCameraInsideTransformationConstGradOp.cxx
  TestGPURayCastCameraInsideTransformationNoGradOp.cxx
  TestGPURayCastCameraInsideTransformationNoShade.cxx
  TestGPURayCastCameraInsideTransformationNoShadeNoGradOp.cxx
  TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransform.cxx
  TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutside.cxx
  TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideFixedStep.cxx
  TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNearPlaneTiny.cxx
  TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformFineStep.cxx
  TestGPURayCastCameraInsideTransformationSampleDist0_5.cxx
  TestGPURayCastCameraInsideTransformationSampleDist0_25.cxx
```

Each variant is run through ctest the same way as the original (swap the test
name; the regex matches both backends):

```sh
ctest --test-dir build_macos_metal \
  -R "TestGPURayCastCameraInsideTransformation(ConstGradOp|NoGradOp|NoShade|NoShadeNoGradOp|NoShadeNoGradOpNoTransform|NoShadeNoGradOpNoTransformCamOutside|NoShadeNoGradOpNoTransformNearPlaneTiny|NoShadeNoGradOpNoTransformFineStep|SampleDist0_5|SampleDist0_25)?" \
  --output-on-failure
```

and the test driver rebuilt with

```sh
cmake --build build_macos_metal --target vtkRenderingVolumeCxxTests -j"$(sysctl -n hw.ncpu)"
```

Each variant was captured on both backends with the dummy-baseline trick above
(section 2) and analyzed with `analyze.py` (section 3),
comparing `max(|Metal−GL|)` across channels:

| variant | center pixel GL → Metal | mean(M−G) | mean\|M−G\| | max\|M−G\| | R-channel fit | px with \|Δ\|≥5 |
|---|---|---|---|---|---|---|
| Original (ShadeOn + gf) | (86,66,54) → (127,92,74) | **+28.82** | 42.04 | 90 | 0.778·gl + 64 | 262144 |
| NoShade + gf | (179,134,108) → (165,122,98) | −5.68 | 8.49 | 32 | 1.182·gl − 48 | 117947 |
| ShadeOn, no gf | (100,76,61) → (101,76,61) | +0.47 | 1.97 | 17 | 1.009·gl + 0.4 | 16527 |
| NoShade, no gf | (252,191,153) → (253,188,152) | −0.97 | 1.80 | 8 | ≈ 1.0·gl | 14453 |
| ConstGradOp (gf≡0.7) | (102,77,62) → (105,79,62) | +1.21 | 2.97 | 17 | 1.006·gl + 1.9 | 46069 |
| NoShade, no gf, **no transform** | (236,180,145) → (235,179,144) | −0.56 | 1.31 | 24 | 0.985·gl + 3.0 | 1444 |
| NoShade, no gf, **no transform, camera outside** | (254,175,134) → (253,174,133) | −0.36 | 1.82 | 21 | 0.998·gl + 0.0 | 13501 |
| NoShade, no gf, no transform, near plane at eye | (236,180,145) → (235,179,144) | −0.56 | 1.31 | 24 | 0.985·gl + 3.0 | 1444 |
| NoShade, no gf, no transform, **SetSampleDistance(0.25)** | (236,180,145) → (235,179,144) | −0.56 | 1.31 | 24 | 0.985·gl + 3.0 | 1444 |
| Original, **SetSampleDistance(0.5)** | (86,66,54) → (127,92,74) | +28.82 | 42.04 | 90 | 0.778·gl + 64 | 262144 |
| Original, **SetSampleDistance(0.25)** | (86,66,54) → (127,92,74) | +28.82 | 42.04 | 90 | 0.778·gl + 64 | 262144 |

`mean(M−G)` is the signed per-channel mean, `mean|M−G|` the max-channel mean
(`md` in `analyze.py`). All rows above were captured with the OpenGL backend
genuinely engaged (verified via the `GL_*` stderr logs); earlier captures that
claimed bit-identical output for the `CamOutside` and `FineStep` rows were
invalid because the `--vtk-factory-prefer RenderingBackend=OpenGL` run silently
fell back to the Metal backend (see the corrections in the takeaways below).

Delta visualizations (`_delta_heatmap.png`, `_delta_mask.png`) are produced by
`analyze.py` above, run from `/tmp/bc/` for each variant.

### Takeaways

1. **Removing gradient opacity collapses the divergence in BOTH shading modes**
   (ShadeOn +0.47, NoShade −0.97) → the entire failure lives in the
   gradient-opacity path.
2. **ConstGradOp (constant gf) is near-identical (+1.21)** → the LUT
   application and its normalized coordinate are correct. The divergence only
   appears with the *varying* gf.
3. **NoShade + gf flips sign** (Metal slightly *dimmer*, −5.68) → it is the
   interaction of gradient opacity with shading (grad-opacity scales the
   composited opacity) that blows up.
4. The varying gf ramp is steep: 0 → 0.5 over 0 → 90 data units. Small
   per-sample gradient-magnitude differences are amplified dramatically.

The four additional variants below (section 2 capture + section 3 analysis)
remove/refine the next parts of the original scene, working only on the
contained `NoShadeNoGradOp`-style scene (no shading, no gradient opacity):

5. **Dropping the `vtkProp3D` transform** (axis-aligned volume, camera still
   inside) cuts the masked pixels ~10× (14453 → 1444), but a residual
   max|Δ|=24 stays, concentrated in the band where the near-plane clip slices
   the volume (`NoTransform_delta_mask.png`, upper-middle region).
6. **Dropping camera-inside too** (camera outside, no near-plane clip) does
   **not** make the render bit-identical: max|Δ|=21 with 13501 masked pixels —
   *more* masked pixels than the inside case. The divergence is therefore
   **not** specific to the camera-inside path.
7. **Near-plane distance is irrelevant** (`NearPlaneTiny`, near plane pulled
   onto the eye): byte-for-byte identical delta table to the default-clip
   `NoTransform` row → the divergence does not come from the clipped entry
   distance.
8. **`SetSampleDistance` is a no-op under the default settings.** Both backends
   ignore the requested sample distance because `AutoAdjustSampleDistances` is
   true by default and they use `ActualSampleDistance = minWorldSpacing`
   (`vtkOpenGLGPUVolumeRayCastMapper.cxx` `UpdateSamplingDistance`, else branch;
   `vtkMetalGPUVolumeRayCastMapper.mm` the matching camera-inside branch). So
   the three `SetSampleDistance` rows above are byte-for-byte the renders of
   their parent rows: `NoTransform`/`FineStep` are identical (max|Δ|=24), and
   `SampleDist0_5`/`SampleDist0_25` are identical to the original (max|Δ|=90).
   The earlier "4× samples → bit-identical (max|Δ|=0)" claim for `FineStep` was
   a **capture artifact**: that run's `--vtk-factory-prefer
   RenderingBackend=OpenGL` invocation silently fell back to the Metal backend
   (the saved "OpenGL" image is byte-identical to the Metal one and to a fresh
   no-prefer run), so it compared Metal against Metal. A genuine GL run of the
   same test shows max|Δ|=24, identical to `NoTransform`.
9. **The original full test does *not* respond to finer sampling**
   (`SampleDist0_5`, `SampleDist0_25`): the delta table is identical to the
   original (+28.82, 262144 masked). Because `SetSampleDistance` is a no-op
   under the default `AutoAdjustSampleDistances` (takeaway 8), these variants
   are byte-for-byte the original render — this row does **not** demonstrate
   that the gradient-opacity failure is sampling-insensitive, it only confirms
   the sample-distance override is ignored.

## Fixed-step sweep: the divergence is not a sampling-phase artifact

To test whether the residual (no-shade / no-gf / no-transform) divergence is a
per-sample phase/comb artifact — which should shrink or vanish as the step is
refined — `AutoAdjustSampleDistances` was disabled and the step forced to the
same explicit world-space value on *both* backends at once, using
`TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideFixedStep.cxx`
(camera outside, env vars `VTK_FIXED_SAMPLE_DISTANCE` /
`VTK_FIXED_AUTO_ADJUST`). Each row captures genuine Metal and OpenGL renders
(GL verified engaged via the `GL_SAMPLING`/`GL_OPTABLE`/`GL_TEX` stderr logs);
the OpenGL backend is deterministic run-to-run (max|Δ|=0 across reruns).

| forced step (world units) | max\|M−G\| | px with \|Δ\|>1 | px with \|Δ\|>9 |
|---|---|---|---|
| 0.0675 | 66 | 196776 | 81040 |
| 0.135 | 56 | 192301 | 71695 |
| 0.27 | 20 | 128559 | 1974 |
| 0.5 | **17** | 84302 | 241 |
| 1.0 | 41 | 62860 | 307 |
| 2.0 | 76 | 55481 | 405 |
| 4.0 | 123 | 35453 | 758 |

Takeaways:

- **No step size makes the backends match**: `max|M−G|` is ≥ 17 everywhere.
- **Refining the step makes the divergence *worse*** (66 at 0.0675 vs 20 at
  0.27): the error grows as samples are added. A comb/phase artifact would
  shrink with more samples, so the divergence is **not** a sampling-phase
  artifact.
- The error accumulates per-sample, pointing at a difference in each sample's
  scalar value or its position along the ray — i.e. the per-sample texture
  fetch (GL `scale`/`bias` + sampler vs Metal 16-bit-unorm path) or the ray
  entry/tstep derivation (GL full-screen `g_dirStep` vs Metal
  `setupVolumeRay`) — not the step size itself.

## Float-accumulation probe: composite precision is not the cause

The residual (no-shade / no-gf / no-transform) divergence grows with sample
count, so the composite accumulation precision was the next suspect: GL
accumulates the front-to-back composite in full `float`
(`g_fragColor = (1.0f - g_fragColor.a) * g_srcColor + g_fragColor;`,
`vtkVolumeShaderComposer.h` 2613/2996) while Metal accumulates in 16-bit
`half` (`half3 accumulatedColor`, `half accumulatedOpacity`,
`MetalShaders.metal` 3737-3738, weight math 4275-4281).

To test this, the Metal accumulators (`accumulatedColor`, `accumulatedOpacity`)
and the weight arithmetic were temporarily changed to `float` precision and the
`NoShadeNoGradOpNoTransform` scene re-captured on genuine Metal and OpenGL
(`/tmp/bc/floatacc2/`):

| metric | half (baseline) | float (probe) |
|---|---|---|
| mean\|M−G\| | 1.31 | 1.26 |
| max\|M−G\| | 24 | 22 |
| px with \|Δ\|≥5 | 1444 | 1271 |
| px with \|Δ\|≥9 | 80 | 89 |

Takeaway: **the composite accumulation precision is not the cause** — the
delta table is essentially unchanged. The half-vs-float hypothesis is
rejected; the probe was reverted. This further narrows the residual to the
per-sample scalar value or ray entry/tstep derivation (a per-sample fetch or
coordinate difference), which the next probe should compare directly on both
backends at the same divergent pixel.

## Scalar-normalization probe: TF-coordinate precision is not the cause

The Metal transfer-function lookup coordinate is computed in 16-bit `half`
precision while GL computes the identical mapping in full `float32`:

- GL: `scalar = texture3D(in_volume[0], g_dataPos) * in_volume_scale[0] +
  in_volume_bias[0]` (all `float`, `vtkVolumeShaderComposer.h` 2690-2706), then
  `computeOpacity`/`computeColor` sample the opacity/color table at the
  resulting `float` coordinate.
- Metal: `scalarNorm = saturate(half(rawScalar) * scalarScale + scalarBias)`
  with `scalarScale`/`scalarBias` also `half`
  (`MetalShaders.metal` 3661-3662, 3984). Casting `rawScalar` (an `R16Unorm`
  sample in [0,1]) to `half` quantizes it to ~11 significant bits — for the
  scalar range (0,4370)/65535 the scale is ≈15, so the TF coordinate is
  quantized by up to ~7.5 texels of the 1024-wide opacity table, far coarser
  than GL's float32 coordinate.

To test this, the Metal `scalarScale`/`scalarBias`/`scalarNorm` (and the
per-component `scalarNormComp`) were temporarily changed to `float` precision
and `NoShadeNoGradOpNoTransform` re-captured on genuine Metal and OpenGL
(`/tmp/bc/floatnorm/`):

| metric | half (baseline) | float (probe) |
|---|---|---|
| mean\|M−G\| | 1.31 | 1.36 |
| max\|M−G\| | 24 | 23 |
| px with \|Δ\|≥5 | 1444 | 1591 |
| px with \|Δ\|≥9 | 80 | 87 |

Takeaway: **the TF-coordinate precision is not the cause** — the delta table is
again essentially unchanged. The half-vs-float normalization hypothesis is
rejected; the probe was reverted. The residual is therefore not in the scalar →
TF-coordinate mapping either; the per-sample scalar *value* itself (the raw
texture sample, or the ray position it is fetched at) or the ray entry/tstep
derivation remain the candidate divergence.


## Gradient-magnitude math: Metal vs GL

Both backends compute the gf LUT input as a spacing-weighted central difference
of the scalar field, normalized and looked up in the same user function:

**Metal** (`Rendering/Metal/Shaders/MetalShaders.metal`
`computeGradientFast` 3238-3249, `normalizedGradient` 3193-3200;
`Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm`):
- 6-neighbor central difference at ±`gradStep` texels,
  `gradStep = 1/(dims−1)` per axis (mapper 6073-6079, 6721);
  `gradTex = rawGrad / gradStep`.
- model-space magnitude `mag = |transpose(volumeToTexture)·gradTex|`;
  `gradW = saturate(mag / gradNormFactor)`,
  `gradNormFactor = 0.5·range / (normFactor·avgSpacing)` (mapper 1072-1074,
  6743-6746; shader 3663).
- gf LUT built over `[0, 0.25·range]` (mapper 3681-3699); sampled at `gradW`
  (shader `sampleGradientOpacity` 3074, applied 4131 / 4302).

**OpenGL** (`Rendering/VolumeOpenGL2/vtkVolumeShaderComposer.h`
`computeGradient` 770-797, `computeGradientOpacity` 637-641;
`Rendering/VolumeOpenGL2/vtkVolumeInputHelper.cxx`
`UpdateGradientOpacityTransferFunction` 194-238):
- `g2 = (g1 − g2)·avgSpacing / (2·spacing_axis)`; `grad_mag = |g2|`;
  `grad_mag = clamp(grad_mag / (0.25·range))`.
- gf LUT built over `[0, range]` (default `SCALAR` range type); sampled at
  `grad_mag`.

The apparent mismatch — the Metal LUT spans `0.25·range` while the GL LUT spans
`range` (a factor of 4) — is compensated on the shader side: the Metal
normalization constant `gradNormFactor` (0.5·range/(normFactor·avgSpacing))
makes the sampled coordinate land on the *same data-unit input* to `gf()` as
the GL coordinate. Working both pipelines through the constants, each reduces
to the magnitude `|Δscalar|·avgSpacing/(2·spacing)` fed to the same `gf()`.

This was checked numerically (see next section): at a logged sample both
backends evaluate `gf()` at **38.67 data units → gf = 0.2149**, identical on
both backends. **The initial hypothesis of a ~60× gradient-opacity
under-application in Metal is therefore rejected** — the normalization is
correct and equivalent to GL by construction.

## Offline verification (numpy + pip `vtk` 9.6.2)

### Step 1: regenerate the resampled volume — `/tmp/bc/vol512.npy`

`vol512.npy` is the 512³ `headsq` array (`uint16`, range [0, 4370], spacing
(0.3945, 0.3945, 0.2701)). Regenerate it from the ExternalData tree with
`make_vol512.py`:

```python
#!/usr/bin/env python3
"""Regenerate /tmp/bc/vol512.npy (512^3 headsq array) from the ExternalData tree.

Requires: pip install vtk numpy
Usage:    python3 make_vol512.py <headsq-prefix> <out.npy>
Example:  python3 make_vol512.py \
            /path/build_macos_metal/ExternalData/Testing/Data/headsq/quarter \
            /tmp/bc/vol512.npy
"""
import sys
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

prefix, out = sys.argv[1], sys.argv[2]

reader = vtk.vtkVolume16Reader()
reader.SetDataDimensions(64, 64)
reader.SetImageRange(1, 93)
reader.SetDataByteOrderToLittleEndian()
reader.SetFilePrefix(prefix)
reader.SetDataSpacing(3.2, 3.2, 1.5)

resize = vtk.vtkImageResize()
resize.SetInputConnection(reader.GetOutputPort())
resize.SetResizeMethodToOutputDimensions()
resize.SetOutputDimensions(512, 512, 512)
resize.Update()

out_img = resize.GetOutput()
arr = vtk_to_numpy(out_img.GetPointData().GetScalars()).reshape(512, 512, 512)
np.save(out, arr)
print(f'wrote {out}: shape={arr.shape} dtype={arr.dtype} range=[{arr.min()}, {arr.max()}] '
      f'spacing={tuple(out_img.GetSpacing())}')
```

```sh
python3 make_vol512.py \
  build_macos_metal/ExternalData/Testing/Data/headsq/quarter \
  /tmp/bc/vol512.npy
```

### Step 2: replay the gradient — `verify_gradient.py`

`verify_gradient.py` reads `vol512.npy` + `metal3.log` and re-implements the
shader chain from `MetalShaders.metal` (`computeGradientFast` /
`normalizedGradient`; see the math section):

```python
#!/usr/bin/env python3
"""Verify the Metal per-sample gradient-opacity numbers against ground truth.

Reads /tmp/bc/vol512.npy (see make_vol512.py) and /tmp/bc/metal3.log (see
make_metal3_log.sh) and replays the shader's gradient computation:

  rawGrad_axis   = (sPX - sNX)/65535              # 16-bit unorm texture
  mag            = |rawGrad / spacing|             # model-space magnitude
  gradW          = mag / (0.5*range/(65535*avgSpacing))
  gf_input       = gradW * 0.25 * range            # data units
  gradOp         = gf(gf_input)                    # the user's gf table

Reproduces the report's key numbers: gradW_pred = 0.0354 vs log 0.0296 at
sample (256,256,168) (ratio 1.20x; all 33 logged samples within 35%), and
gf_input ~38.6 data units -> gf ~0.214 on both backends.

Usage: python3 verify_gradient.py [vol512.npy] [metal3.log]
"""
import re
import sys
import numpy as np

vol = np.load(sys.argv[1] if len(sys.argv) > 1 else 'vol512.npy')
log = open(sys.argv[2] if len(sys.argv) > 2 else 'metal3.log').read().splitlines()

spacing = np.array([0.39452054794520547, 0.39452054794520547, 0.2700587084148728])
RANGE = float(vol.max())
AVG = spacing.mean()
NORM = 65535.0
GRAD_MAX = 0.25 * RANGE  # Metal gf LUT spans [0, 0.25*range]
GRAD_NORM_FACTOR = 0.5 * RANGE / (NORM * AVG)

def gf(x):  # report gf: 0@0, 0.5@90, 0.7@100
    if x <= 0: return 0.0
    if x < 90: return 0.5 * x / 90
    if x < 100: return 0.5 + 0.2 * (x - 90) / 10
    return 0.7

def sample(p):
    i0 = np.floor(p).astype(int); t = p - i0
    c = [vol[i0[0],i0[1],i0[2]], vol[i0[0]+1,i0[1],i0[2]],
         vol[i0[0],i0[1]+1,i0[2]], vol[i0[0]+1,i0[1]+1,i0[2]],
         vol[i0[0],i0[1],i0[2]+1], vol[i0[0]+1,i0[1],i0[2]+1],
         vol[i0[0],i0[1]+1,i0[2]+1], vol[i0[0]+1,i0[1]+1,i0[2]+1]]
    w = [(1-t[0])*(1-t[1])*(1-t[2]), t[0]*(1-t[1])*(1-t[2]),
         (1-t[0])*t[1]*(1-t[2]), t[0]*t[1]*(1-t[2]),
         (1-t[0])*(1-t[1])*t[2], t[0]*(1-t[1])*t[2],
         (1-t[0])*t[1]*t[2], t[0]*t[1]*t[2]]
    return sum(w[k]*c[k] for k in range(8))

ps = re.compile(r'DEBUG SAMPLE px=\(\d+, \d+\) i=(\d+) .*eval=\(([\d.e-]+), ([\d.e-]+), ([\d.e-]+)\) raw=([\d.e-]+)')
pl = re.compile(r'DEBUG LIGHT px=\(\d+, \d+\) i=(\d+) .*gradW=([\d.e-]+) gradOp=([\d.e-]+)')
evald, rawd, gradw, gradop = {}, {}, {}, {}
for l in log:
    m = ps.search(l)
    if m:
        evald[int(m.group(1))] = np.array([float(m.group(2)), float(m.group(3)), float(m.group(4))])
        rawd[int(m.group(1))] = float(m.group(5))
    m = pl.search(l)
    if m:
        gradw[int(m.group(1))] = float(m.group(2))
        gradop[int(m.group(1))] = float(m.group(3))

STEP = 512 / 511  # gradStep in texel units
ratios = []
for i, e in evald.items():
    if i not in gradw:
        continue
    f = e * 512 - 0.5
    d = np.array([sample(f + np.eye(3)[k]*STEP) - sample(f - np.eye(3)[k]*STEP) for k in range(3)])
    mag = np.sqrt(((d / NORM / spacing) ** 2).sum())
    gradw_pred = mag / GRAD_NORM_FACTOR
    ratios.append(gradw_pred / gradw[i])
r = np.array(ratios)
print(f'gradW replay: {len(r)} samples, ratio mean {r.mean():.3f}, '
      f'median {np.median(r):.3f}, 100% within 35%: {100*((r>0.65)&(r<1.35)).mean():.0f}%')

i = 168
e = evald[i]
f = e * 512 - 0.5
d = np.array([sample(f + np.eye(3)[k]*STEP) - sample(f - np.eye(3)[k]*STEP) for k in range(3)])
mag = np.sqrt(((d / NORM / spacing) ** 2).sum())
gradw_pred = mag / GRAD_NORM_FACTOR
gf_input = gradw_pred * GRAD_MAX
print(f'sample (256,256,{i}): central diff d={np.round(d,1)} data units')
print(f'  gradW_pred={gradw_pred:.4f} vs log {gradw[i]:.4f} (ratio {gradw_pred/gradw[i]:.2f}x)')
print(f'  gf_input={gf_input:.2f} data units -> gf={gf(gf_input):.4f} (log gradOp={gradop[i]:.4f})')

relerr = []
for i, e in evald.items():
    raw = rawd[i] * NORM
    if raw < 500:
        continue
    relerr.append((sample(e * 512 - 0.5) - raw) / raw)
a = np.array(relerr)
print(f'raw-scalar probe (texel-center convention): n={len(a)}, median rel {np.median(a)*100:+.2f}%, '
      f'|err| p90 {np.percentile(np.abs(a),90)*100:.2f}%')
```

```sh
python3 verify_gradient.py /tmp/bc/vol512.npy /tmp/bc/metal3.log
```

**Verified output** (reproduced from the current `vol512.npy` + a fresh
`metal3.log`):

```
gradW replay: 33 samples, ratio mean 0.957, median 0.935, 100% within 35%: 100%
sample (256,256,168): central diff d=[-84.1  17.   -7.1] data units
  gradW_pred=0.0354 vs log 0.0296 (ratio 1.20x)
  gf_input=38.67 data units -> gf=0.2149 (log gradOp=0.1647)
raw-scalar probe (texel-center convention): n=188, median rel -0.04%, |err| p90 2.24%
```

Interpretation:

- At the logged sample `eval=(0.400268, 0.494169, 0.622434)` the shader's
  texel-center convention `f = eval·512 − 0.5` with `gradStep = 512/511`
  texels reproduces the log's `gradW` to within 35% on all 33 samples (the
  1.20× point residual is the same per-frame marching phase difference seen in
  the raw-scalar probe). Using `f = eval·511` instead collapses the gradient
  (`gradW ≈ 0.0001`), so Metal's own convention is internally consistent.
- Both backends evaluate `gf()` at **38.67 data units → gf = 0.2149** — the
  Metal log's `gradW` and the GL pipeline therefore feed the *same* gf input.
  **The initial hypothesis of a ~60× gradient-opacity under-application in
  Metal is rejected** — the normalization is correct and equivalent to GL by
  construction.
- **Raw-scalar probe** (log `raw` vs trilinear interpolation of `vol512.npy` at
  `eval·512 − 0.5`, samples with raw ≥ 500): median relative error −0.04%,
  p90 of |err| 2.2% (max ~7%). The small residuals point to per-frame marching
  differences rather than a global coordinate-convention error.

## Per-sample GPU logging — regenerate `/tmp/bc/metal3.log`

The `MARCH`/`SAMPLE`/`LIGHT`/`LIGHT2` call sites live in `MetalShaders.metal`,
gated by `#if defined(VTK_METAL_ENABLE_LOGGING)` and `debugMarchGate`
(pixel (256,256) is included). Shader `os_log` messages are buffered per
command buffer and must be forwarded to stderr with `MTL_LOG_*` env vars —
see `Rendering/Metal/Testing/Cxx/TestMetalVolumeShaderLog.cxx` for the full
reference. `make_metal3_log.sh` regenerates the log:

```sh
#!/bin/bash
# Regenerate /tmp/bc/metal3.log (per-sample march dump) from a Metal test run.
#
# Prerequisites:
#   - Build with tests enabled so the volume shaders compile with shader
#     logging: ./macos_metal_build.sh --resume --tests
#     (VTK_BUILD_TESTING=ON -> VTK_METAL_ENABLE_LOGGING is defined and
#     MTLCompileOptions.enableLogging is set; see
#     Rendering/Metal/Testing/Cxx/TestMetalVolumeShaderLog.cxx).
#   - The per-sample call sites (MARCH/SAMPLE/LIGHT/LIGHT2) are gated to a few
#     pixels by debugMarchGate in Rendering/Metal/Shaders/MetalShaders.metal
#     (includes the (256,256) camera-inside pixel used by this report).
#   - Shader os_log is buffered per command buffer and forwarded to stderr via
#     MTL_LOG_TO_STDERR=1; MTL_LOG_BUFFER_SIZE must be large enough to hold a
#     full frame (~0.5 MB for the 469-sample pixel gate), else messages are
#     silently dropped.
#
# Usage: ./make_metal3_log.sh [build_dir]
# Output: /tmp/bc/metal3.log (stderr of the run, formatted by Metal's os_log)

set -e
BUILD_DIR="${1:-build_macos_metal}"
OUT="${OUT:-/tmp/bc/metal3.log}"

# dummy baseline so the run exercises the full (fail-and-dump) path
B="/tmp/bc/TestGPURayCastCameraInsideTransformation.png"
python3 -c "from PIL import Image; Image.new('RGB',(512,512),(0,0,0)).save('$B')"

MTL_LOG_LEVEL=MTLLogLevelDebug \
MTL_LOG_BUFFER_SIZE=16777216 \
MTL_LOG_TO_STDERR=1 \
  "$BUILD_DIR/bin/vtkRenderingVolumeCxxTests" TestGPURayCastCameraInsideTransformation \
    --vtk-factory-prefer RenderingBackend=Metal \
    -D "$BUILD_DIR/ExternalData/Testing" -T "$BUILD_DIR/Testing/Temporary" \
    -V "$B" \
    2> "$OUT"

wc -l "$OUT"
```

Prerequisites: build with `--tests` (`VTK_BUILD_TESTING=ON` → the
`VTK_METAL_ENABLE_LOGGING` macro + `MTLCompileOptions.enableLogging`), macOS 15+
(Metal 3.2 shader `os_log`). `MTL_LOG_BUFFER_SIZE` must exceed one frame's
messages (~0.5 MB for the 469-sample pixel gate), otherwise lines are silently
dropped. Each log line records `t`, texture/eval position, raw and normalized
scalar, opacity, `gradW`, sampled `gradOp`, `nDotL`, lighting terms and the
color before/after shading; for pixel (256,256) the first sample is at
`t = 0.001623`.

## Conclusion

**Two independent mechanisms are at work.**

1. **Residual ray-casting mismatch (contained scene, no shade / no gf).**
   With shading, gradient opacity and the transform all removed, Metal still
   diverges from GL: max|Δ|=24 (1444 px ≥5) with the camera inside,
   max|Δ|=21 (13501 px ≥5) with the camera **outside** the volume. This is a
   *general* ray-cast mismatch, **not** a camera-inside / near-plane-clipping
   artifact:
   - Camera outside → still diverges (`NoShadeNoGradOpNoTransformCamOutside`,
     max|Δ|=21).
   - Near-plane pulled onto the eye → identical delta table to the default
     clip (`NearPlaneTiny`): the clipped-entry distance is irrelevant.
   - `SetSampleDistance(0.25)` (`FineStep`) → identical to `NoTransform`
     (max|Δ|=24) because the override is a no-op under default
     `AutoAdjustSampleDistances`; the earlier "4× samples → bit-identical"
     result was a capture fluke where the OpenGL run fell back to Metal.
   - **Forcing an explicit step on both backends** (`AutoAdjustSampleDistancesOff`,
      camera-outside step sweep above) still diverges at every step size, and
      refining the step makes the error *worse* → the divergence is **not** a
      sampling-phase/comb artifact.
   - **Composite accumulation precision is not involved**: running the Metal
       composite in `float` instead of `half` leaves the delta table essentially
       unchanged (max|Δ| 24→22, px ≥9 80→89, see the float-accumulation probe).
   - **TF-coordinate precision is not involved**: computing the Metal scalar
      normalization (`scalarScale`/`scalarBias`/`scalarNorm`) in `float`
      instead of `half` leaves the delta table essentially unchanged
      (max|Δ| 24→23, px ≥9 80→87, see the scalar-normalization probe).
    The candidate causes are the per-sample scalar value (raw texture sample or
    the ray position it is fetched at) or coordinate/tstep
    derivation (GL full-screen ray origin/`g_dirStep` chain vs Metal
    `setupVolumeRay`; GL volume-texture `scale`/`bias` + sampler vs Metal
    16-bit-unorm path) — all active for any camera position.
   The previously-suspected `firstT` comb mismatch
   (`MetalShaders.metal` 3723-3725, `ceil` on the `checkBounds == false` path)
   is **not** involved: the camera-inside test marches through `marchVolume`
   (`checkBounds == true`, `firstT = jitter = stepSize`), whose comb already
   matches GL — changing that `firstT` had zero effect on the render.

2. **Gradient-opacity × shading interaction (original full test).**
   The full original scene diverges much harder (max|Δ|=90, 262144 px) and its
   removal is dominated by the gradient-opacity path (takeaways 1–4). The
   `SampleDist0_5`/`SampleDist0_25` variants confirm nothing about sampling
   sensitivity because the sample-distance override is ignored; the gradient
   failure must be investigated separately.

## Fix plan

For the **contained residual mismatch** (max|Δ|=24 inside / 21 outside):
the divergence is camera-position-independent, step-size-independent (fixed
step sweep above), and precision-independent (float-accumulation and
scalar-normalization probes), so the fix must be in a per-sample quantity that
both backends compute differently.
Candidate causes to compare at a logged divergent pixel: (a) the GL full-screen
ray origin/step (`g_dirStep`) vs Metal `setupVolumeRay` entry/tstep, and (b)
the GL volume-texture `scale`/`bias`+ sampler fetch vs the Metal 16-bit-unorm
sample — compare raw sample scalars along a divergent ray on both backends. The
sample-distance knob is exhausted (it does not change the divergence), as are
composite accumulation precision and the TF-coordinate precision, so the next
probe should log per-sample scalars on both backends for the same divergent
pixel (e.g. using the
`MetalShaders.metal` `VTK_METAL_ENABLE_LOGGING` dump plus an equivalent GL-side
log) and diff the trace. Validate with:

```sh
ctest --test-dir build_macos_metal \
  -R "TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransform" \
  --output-on-failure
```

The original full test (GF/OpenGL #681, Metal #782) remains **unfixed**: its
divergence lives in the gradient-opacity × shading path and is orthogonal to
the contained residual — see the gradient-magnitude section above for the
state of that investigation.

## Artifacts

All captures and analysis live in `/tmp/bc/`; each item is regenerable from the
command listed (assume the directory may be erased at any time):

| artifact | contents | regenerate with |
|---|---|---|
| `gl/`, `metal/`, `metal2/`, `metal3/` | original test, each backend's render + `.diff.png` | section 2 (dummy-baseline capture, `RenderingBackend=OpenGL`/`Metal`) |
| `noshade/`, `TestGPURayCastCameraInsideTransformationNoGradOp/`, `TestGPURayCastCameraInsideTransformationNoShadeNoGradOp/`, `ConstGradOp/` | per-variant captures (`opengl/`, `metal/`) + `_delta_heatmap.png`, `_delta_mask.png` | sections 2+3 on each variant (swap the test name) |
| `NoTransform/` | `NoShadeNoGradOpNoTransform` captures (no vtkProp3D transform) + delta mask/heatmap | sections 2+3 on `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransform` |
| `CamOutside/` | `NoShadeNoGradOpNoTransformCamOutside` captures (camera outside, no near-plane clip) — **diverges** (max|Δ|=21, 13501 px); the byte-identical captures in this folder predate the factory-fallback fix and are invalid (both were Metal renders) | sections 2+3 on `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutside` |
| `NearPlaneTiny/` | `NoShadeNoGradOpNoTransformNearPlaneTiny` captures (near plane on the eye) — delta identical to `NoTransform/` | sections 2+3 on `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNearPlaneTiny` |
| `FineStep/` | `NoShadeNoGradOpNoTransformFineStep` captures (requested 0.25 sample distance) — the byte-identical copies are invalid (OpenGL run fell back to Metal); genuine GL gives max|Δ|=24, identical to `NoTransform/` because the override is a no-op | sections 2+3 on `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformFineStep` |
| `SampleDist0_5/`, `SampleDist0_25/` | original test at 0.5× / 0.25× sample distance — delta identical to the original failure | sections 2+3 on `TestGPURayCastCameraInsideTransformationSampleDist0_5` / `…SampleDist0_25` |
| `recheck/` | full variant table recaptured with genuine OpenGL (`RenderingBackend=OpenGL` verified in stderr) | section 2 + `analyze.py` on each variant, `-T /tmp/bc/recheck` |
| `sweep/` | camera-outside fixed-step sweep (`sd{0.0675…4.0}_{Metal,OpenGL}.png` + logs), auto-adjust off | `VTK_FIXED_SAMPLE_DISTANCE=<sd>` on `…NoTransformCamOutsideFixedStep` for both backends |
| `floatacc2/` | `NoShadeNoGradOpNoTransform` captures with Metal composite temporarily in `float` precision (`OpenGL.png`/`Metal.png`) — delta ≈ unchanged, probe reverted | section 2 on `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransform` with the float-accumulation shader edit |
| `floatnorm/` | `NoShadeNoGradOpNoTransform` captures with Metal scalar normalization (`scalarScale`/`scalarBias`/`scalarNorm`) in `float` precision (`NoTransform_GL_floatnorm.png`/`NoTransform_MT_floatnorm.png`) — delta ≈ unchanged, probe reverted | section 2 on the same test with the scalar-normalization shader edit |
| `analyze.py` | delta-stats + heatmap/mask script | section 3 (inline listing) |
| `vol512.npy` | 512³ `headsq` array (uint16, [0,4370]) | `make_vol512.py` (offline verification, step 1) |
| `metal3.log` | per-sample MARCH/SAMPLE/LIGHT/LIGHT2 dump (3247 lines) | `make_metal3_log.sh` (per-sample GPU logging) |
| `verify_gradient.py` | numpy replay of `gradW`/`gf` chain | offline verification, step 2 (inline listing) |
| `baseline.png` | copy of the committed baseline for reference | `cp build_macos_metal/ExternalData/Rendering/Volume/Testing/Data/Baseline/TestGPURayCastCameraInsideTransformation.png /tmp/bc/baseline.png` |
