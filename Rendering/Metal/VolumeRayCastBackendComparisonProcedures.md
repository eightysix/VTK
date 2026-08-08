# Metal vs OpenGL volume ray cast: artifact production and analysis

Procedural cookbook for producing and analyzing artifacts that compare the
Metal and OpenGL GPU volume ray-cast backends, using
`TestGPURayCastCameraInsideTransformation` as the base scene. The same
procedures apply to other volume-rendering image tests and to the togglable
variants below.

## Environment and build

- macOS arm64, `-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0`, Release, static libs, Ninja.
- Build directory: `build_macos_metal` (see `macos_metal_build.sh --resume --tests`).
- The `RenderingMetal` module registers factory overrides carrying
  `RenderingBackend=Metal`, so either backend can be selected at runtime with
  `--vtk-factory-prefer RenderingBackend=<OpenGL|Metal>`. The
  `vtkRenderingVolumeCxxTests` driver does parse that flag (unlike the generic
  `vtk*CxxTests` driver), which is what makes side-by-side backend captures
  possible.

## Reproduction

### 1. Plain ctest

```sh
ctest --test-dir build_macos_metal -R "^VTK::RenderingVolumeCxx-TestGPURayCastCameraInsideTransformation$" --output-on-failure     # OpenGL
ctest --test-dir build_macos_metal -R "^VTK::RenderingVolumeCxx-Metal-TestGPURayCastCameraInsideTransformation$" --output-on-failure  # Metal
```

A failing run writes the rendered image and difference image to
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

**Capture validity:** a `--vtk-factory-prefer RenderingBackend=OpenGL`
invocation can silently fall back to the Metal backend, which makes the saved
"OpenGL" render byte-identical to the Metal one. Verify each OpenGL capture is
genuinely GL-engaged via the `GL_*` stderr logs before analyzing it.

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

This prints per-channel delta statistics (center pixel, signed/absolute/max
delta, per-channel linear fit `metal = a·gl + b`, masked-pixel count) and writes
`<label>_delta_heatmap.png` (pixels with |Δ|≥5, brightness ∝ |Δ|) and
`<label>_delta_mask.png`.

## Image tests: isolating the feature

The base test combines Phong shading (`ShadeOn`), gradient opacity
(`gf: 0@0, 0.5@90, 0.7@100`), and camera-inside-with-clipping. To isolate which
feature differs between backends, create sibling tests in
`Rendering/Volume/Testing/Cxx/` by toggling those features:

| file | change vs original |
|------|--------------------|
| `TestGPURayCastCameraInsideTransformationNoShade.cxx` | `ShadeOff()` |
| `TestGPURayCastCameraInsideTransformationNoGradOp.cxx` | drop `SetGradientOpacity` |
| `TestGPURayCastCameraInsideTransformationNoShadeNoGradOp.cxx` | `ShadeOff()` + no gradient opacity |
| `TestGPURayCastCameraInsideTransformationConstGradOp.cxx` | gradient opacity constant `0.7` everywhere (`gf: 0.7@0, 0.7@2000`) |
| `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransform.cxx` | also drop the `vtkProp3D` transform (`Rotate*`/`SetOrigin`), camera repositioned inside the axis-aligned bounds |
| `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutside.cxx` | also move the camera outside (no near-plane clip) |
| `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideFixedStep.cxx` | camera outside, env-driven step sweep: `VTK_FIXED_SAMPLE_DISTANCE` disables `AutoAdjustSampleDistances` and forces the step on both backends |
| `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideNoJitter.cxx` | camera outside with `SetUseJittering(false)` on the mapper (both backends then offset the first sample by exactly one step), ruling out the random-noise source |
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
  TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideNoJitter.cxx
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

Each variant is then captured on both backends with the dummy-baseline trick
(section 2) and analyzed with `analyze.py` (section 3), comparing
`max(|Metal−GL|)` across channels. Delta visualizations
(`_delta_heatmap.png`, `_delta_mask.png`) are produced by `analyze.py` run from
`/tmp/bc/` for each variant.

## Fixed-step sweep

To test whether a divergence is a per-sample phase/comb artifact — which should
shrink or vanish as the step is refined — disable `AutoAdjustSampleDistances`
and force the step to the same explicit world-space value on *both* backends at
once, using
`TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideFixedStep.cxx`
(camera outside, env var `VTK_FIXED_SAMPLE_DISTANCE` in world units;
`VTK_FIXED_AUTO_ADJUST=1` re-enables auto-adjust). Each row should capture
genuine Metal and OpenGL renders; verify the OpenGL backend is genuinely
engaged via the `GL_SAMPLING`/`GL_OPTABLE`/`GL_TEX` stderr logs. The OpenGL
backend is deterministic run-to-run, so reruns can be compared to validate a
capture.

```sh
mkdir -p /tmp/bc/sweep
python3 -c "from PIL import Image; Image.new('RGB',(512,512),(0,0,0)).save('/tmp/bc/TestGPURayCastCameraInsideTransformation.png')"
for sd in 0.0675 0.135 0.27 0.5 1.0 2.0 4.0; do
  for backend in OpenGL Metal; do
    VTK_FIXED_SAMPLE_DISTANCE=$sd \
      build_macos_metal/bin/vtkRenderingVolumeCxxTests \
        TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideFixedStep \
        --vtk-factory-prefer RenderingBackend=$backend \
        -D build_macos_metal/ExternalData/Testing \
        -T build_macos_metal/Testing/Temporary \
        -V /tmp/bc/TestGPURayCastCameraInsideTransformation.png
    cp build_macos_metal/Testing/Temporary/TestGPURayCastCameraInsideTransformation.png \
       /tmp/bc/sweep/sd${sd}_${backend}.png
  done
done
```

(Each run prints `PROBE fixedSD=<sd> autoAdjust=0` to stderr — confirm
`autoAdjust=0` so the requested step is honored.)

## Precision probes

To test whether a divergence is caused by Metal computing in `half` precision
where GL uses `float`, temporarily change the relevant Metal precision in
`Rendering/Metal/Shaders/MetalShaders.metal`, rebuild
(`cmake --build build_macos_metal --target vtkRenderingVolumeCxxTests -j"$(sysctl -n hw.ncpu)"`),
re-capture on genuine Metal and OpenGL (section 2), and compare with
`analyze.py` (section 3). Revert the probe afterwards.

- **Float-accumulation probe**: change the Metal composite accumulators
  (`accumulatedColor`, `accumulatedOpacity`) and the weight arithmetic from
  `half` to `float`.
- **Scalar-normalization probe**: change the Metal `scalarScale`/`scalarBias`/
  `scalarNorm` (and the per-component `scalarNormComp`) from `half` to `float`.

## Nearest-interpolation variant

`SetInterpolationTypeToNearest()` on both backends removes the trilinear
sampler (cell-to-point texel offsets, R16Unorm/`texture3D` filtering) from the
scene: with nearest-neighbor lookup each sample reads one raw texel, so any
interpolation-related backend difference disappears.

Test: `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNearest`
(the contained scene + `SetInterpolationTypeToNearest()`), captured on genuine
Metal and OpenGL and analyzed as in sections 2+3.

## Maximum-intensity variant

`SetBlendModeToMaximumIntensity()` removes the opacity-weighted front-to-back
compositing from the scene: only the per-sample scalar is tracked (MIP keeps
the max normalized scalar) and the color TF is sampled once at the end, so the
composite path is bypassed entirely.

Test: `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformMaxIP`
(the contained scene + `SetBlendModeToMaximumIntensity()`), captured on genuine
Metal and OpenGL and analyzed as in sections 2+3.

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
`normalizedGradient`), so logged per-sample values can be compared against
ground truth:

```python
#!/usr/bin/env python3
"""Verify the Metal per-sample gradient computation against ground truth.

Reads /tmp/bc/vol512.npy (see make_vol512.py) and /tmp/bc/metal3.log (see
make_metal3_log.sh) and replays the shader's gradient computation:

  rawGrad_axis   = (sPX - sNX)/65535              # 16-bit unorm texture
  mag            = |rawGrad / spacing|             # model-space magnitude
  gradW          = mag / (0.5*range/(65535*avgSpacing))
  gf_input       = gradW * 0.25 * range            # data units
  gradOp         = gf(gf_input)                    # the user's gf table

Prints a gradW replay ratio over all logged samples, the gf input/value at a
specific logged sample, and a raw-scalar probe relative-error distribution.

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

The script prints a `gradW` replay ratio over all logged samples, the `gf`
input/value at a specific logged sample, and a raw-scalar probe
relative-error distribution, so logged Metal values can be compared against
the numpy ground truth.

## Verified logging invocations (cheatsheet)

Both backends have per-sample dump hooks gated behind env vars. The exact,
verified invocation for each is below; copy these verbatim — the args are easy
to get wrong (the GL sample dump silently requires the RAY dump gate, and the
Metal shader `os_log` messages are silently dropped without the `MTL_LOG_*`
vars).

Common bits (substitute per run):

```sh
BIN=build_macos_metal/bin/vtkRenderingVolumeCxxTests
EXT=build_macos_metal/ExternalData/Testing
TMP=build_macos_metal/Testing/Temporary
BASELINE=/tmp/bc/dummy_baseline.png   # wrong baseline -> test fails and dumps its render
# e.g. python3 -c "from PIL import Image; Image.new('RGB',(512,512),(0,0,0)).save('$BASELINE')"
```

**Metal** — per-sample MARCH/SAMPLE dump at the GL-matched pixel (422,92)
(shader `os_log` is buffered and dropped unless all three `MTL_LOG_*` vars are
set; see `TestMetalVolumeShaderLog.cxx`):

```sh
MTL_LOG_LEVEL=MTLLogLevelDebug \
MTL_LOG_BUFFER_SIZE=16777216 \
MTL_LOG_TO_STDERR=1 \
  "$BIN" TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=Metal \
    -D "$EXT" -T "$TMP" -V "$BASELINE" 2> metal_samples.log
# verify: rg -c 'SAMPLE px=\(422, 92\)' metal_samples.log   (expect ~1000+)
#   MARCH: rg -c 'MARCH px=\(422, 92\)' metal_samples.log   (expect 6)
```

The per-pixel gate is `debugMarchGate` in `MetalShaders.metal`; it currently
also dumps (422,92) unconditionally (camera-agnostic) via `pxOkAlways`. This
Metal pixel is GL `(422,419)` (y-flip), so pair it with
`VTK_GL_SAMPLE_DUMP_PX=422,419` on the GL side.

**OpenGL** — per-sample raw dump at the matched pixel. BOTH env vars are
required (the sample dump block sits under the RAY dump gate). The dump pixel
is in `glReadPixels` coords (bottom origin), so for the Metal-matched pixel
use the y-flipped value: Metal `(422,92)` == GL `(422,419)`. (Using
`VTK_GL_SAMPLE_DUMP_PX=422,92` captures a *different* physical pixel — see the
pixel-pairing note under `compare_gl_metal_samples.py`.)

```sh
VTK_GL_RAY_DUMP=1 \
VTK_GL_SAMPLE_DUMP=1 \
VTK_GL_SAMPLE_DUMP_PX=422,419 \
  "$BIN" TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=OpenGL \
    -D "$EXT" -T "$TMP" -V "$BASELINE" 2> gl_samples.log
# verify: rg -c 'GL_SAMPLE px=\(422, 419\)' gl_samples.log   (expect 175 * frames)
```

The GL dump lives in `vtkOpenGLGPUVolumeRayCastMapper.cxx` (`DoGPURender`,
`if (getenv("VTK_GL_SAMPLE_DUMP"))`), printing every sample's channel-encoded
scalar (16-bit float pack) for the pixel named by `VTK_GL_SAMPLE_DUMP_PX`.

## Per-sample GL↔Metal comparison — `compare_gl_metal_samples.py`

`Rendering/Metal/BackendComparisonTools/compare_gl_metal_samples.py` diffs the
per-sample dumps of both backends for one physical pixel, printing per-sample
positions and raw scalars side by side plus the max position/raw divergence.

**Pixel pairing (important):** the Metal `MARCH`/`SAMPLE` logs use Metal
`screenPos` (origin top-left) while the `GL_SAMPLE` dump uses `glReadPixels`
coords (origin bottom-left). For the 512×512 viewport the same physical pixel is
Metal `(x, y)` == GL `(x, 511 - y)`. So to compare against Metal pixel
`(422, 92)`, the GL log must be captured with `VTK_GL_SAMPLE_DUMP_PX=422,419`
— *not* `422,92`. Capturing GL at `(422,92)` and Metal at `(422,92)` compares
two different physical pixels, and the sample positions then diverge
systematically (a false positive). The `GL_RAY` gate list already carries the
flipped `(422, 419)` entry, and its `origin`/`step` match the Metal `MARCH`
ray to ~1e-5, confirming the pairing.

Usage (logs produced by the cheatsheet invocations below):

```sh
python3 Rendering/Metal/BackendComparisonTools/compare_gl_metal_samples.py \
  gl_samples.log metal_samples.log 422 92
```

The tool was developed and validated against the NoJitter variant
(`TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter`,
6 frames, deterministic per backend); its docstring embeds the exact GL and
Metal capture commands so the inputs are reproducible. For other tests, capture
the same test name on both backends with the cheatsheet invocations (swap the
pixel if the gate differs).

Expected result for the NoJitter test: positions agree to ≤1e-5 and raw scalars
to ≤1e-6 across all samples, with only a handful of larger raw diffs
(i=30, 134, 167) that are frame-ordering / camera-animation artifacts rather
than backend differences.

## Per-sample GPU logging — regenerate `/tmp/bc/metal3.log`

The `MARCH`/`SAMPLE`/`LIGHT`/`LIGHT2` call sites live in `MetalShaders.metal`,
gated by `#if defined(VTK_METAL_ENABLE_LOGGING)` and `debugMarchGate`. Shader
`os_log` messages are buffered per command buffer and must be forwarded to
stderr with `MTL_LOG_*` env vars — see
`Rendering/Metal/Testing/Cxx/TestMetalVolumeShaderLog.cxx` for the full
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
#     (includes the (256,256) camera-inside pixel).
#   - Shader os_log is buffered per command buffer and forwarded to stderr via
#     MTL_LOG_TO_STDERR=1; MTL_LOG_BUFFER_SIZE must be large enough to hold a
#     full frame, else messages are silently dropped.
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
messages, otherwise lines are silently dropped. Each log line records `t`,
texture/eval position, raw and normalized scalar, opacity, `gradW`, sampled
`gradOp`, `nDotL`, lighting terms and the color before/after shading for the
gated pixels.

## Artifacts

All captures and analysis live in `/tmp/bc/`; each item is regenerable from the
command listed (assume the directory may be erased at any time). Tool scripts are
persisted in `Rendering/Metal/BackendComparisonTools/` (the inline listings below
are copies for reader convenience; the persisted copies parameterize the build
dir, defaulting to `<repo-root>/build_macos_metal`):

| artifact | contents | regenerate with |
|---|---|---|
| `gl/`, `metal/`, `metal2/`, `metal3/` | original test, each backend's render + `.diff.png` | section 2 (dummy-baseline capture, `RenderingBackend=OpenGL`/`Metal`) |
| `noshade/`, `TestGPURayCastCameraInsideTransformationNoGradOp/`, `TestGPURayCastCameraInsideTransformationNoShadeNoGradOp/`, `ConstGradOp/` | per-variant captures (`opengl/`, `metal/`) + `_delta_heatmap.png`, `_delta_mask.png` | sections 2+3 on each variant (swap the test name) |
| `NoTransform/` | `NoShadeNoGradOpNoTransform` captures (no vtkProp3D transform) + delta mask/heatmap | sections 2+3 on `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransform` |
| `CamOutside/` | `NoShadeNoGradOpNoTransformCamOutside` captures (camera outside, no near-plane clip) | sections 2+3 on `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutside` |
| `NearPlaneTiny/` | `NoShadeNoGradOpNoTransformNearPlaneTiny` captures (near plane on the eye) | sections 2+3 on `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNearPlaneTiny` |
| `FineStep/` | `NoShadeNoGradOpNoTransformFineStep` captures (requested 0.25 sample distance) | sections 2+3 on `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformFineStep` |
| `SampleDist0_5/`, `SampleDist0_25/` | original test at 0.5× / 0.25× sample distance | sections 2+3 on `TestGPURayCastCameraInsideTransformationSampleDist0_5` / `…SampleDist0_25` |
| `recheck/` | full variant table recaptured, OpenGL engagement verified in stderr | section 2 + `analyze.py` on each variant, `-T /tmp/bc/recheck` |
| `sweep/` | camera-outside fixed-step sweep (`sd{0.0675…4.0}_{Metal,OpenGL}.png` + logs), auto-adjust off | fixed-step sweep section for both backends |
| `floatacc2/` | `NoShadeNoGradOpNoTransform` captures with Metal composite temporarily in `float` precision (`OpenGL.png`/`Metal.png`), probe reverted | section 2 on `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransform` with the float-accumulation shader edit |
| `floatnorm/` | `NoShadeNoGradOpNoTransform` captures with Metal scalar normalization (`scalarScale`/`scalarBias`/`scalarNorm`) in `float` precision, probe reverted | section 2 on the same test with the scalar-normalization shader edit |
| `nearest/` | `NoShadeNoGradOpNoTransformNearest` captures (`OpenGL_probe.png`/`Metal_probe.png`) | section 2 on `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNearest` |
| `maxip/` | `NoShadeNoGradOpNoTransformMaxIP` captures (`OpenGL_probe.png`/`Metal_probe.png`) | section 2 on `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformMaxIP` |
| `analyze.py` | delta-stats + heatmap/mask script | section 3 (inline listing) |
| `vol512.npy` | 512³ `headsq` array (uint16) | `make_vol512.py` (offline verification, step 1) |
| `metal3.log` | per-sample MARCH/SAMPLE/LIGHT/LIGHT2 dump | `make_metal3_log.sh` (per-sample GPU logging) |
| `metal_noshade.log` | NoShade SAMPLE + GRADOP dump | `make_metal_noshade_log.sh` (findings doc section 5) |
| `verify_gradient.py` | numpy replay of `gradW`/`gf` chain | offline verification, step 2 (inline listing) |
| `verify_gradient_noshade.py` | numpy replay of `gradW`/`gradOp` from GRADOP lines | findings doc section 5 |
| `replay_422_92.py`, `finestep_sim.py` | historical px (422,92) trace replays (default-vs-4x composite) | persisted `BackendComparisonTools/` |
| `compare_gl_metal_samples.py` | per-sample GL↔Metal log diff for one pixel (y-flipped pairing) | `Rendering/Metal/BackendComparisonTools/compare_gl_metal_samples.py gl_samples.log metal_samples.log 422 92` |
| `capture_variants.sh` | both-backend capture for a variant set (OpenGL/Metal) + GL_SAMPLING check | `Rendering/Metal/BackendComparisonTools/capture_variants.sh` |
| `capture_sweep.sh` | camera-outside fixed-step sweep, both backends | `Rendering/Metal/BackendComparisonTools/capture_sweep.sh` |
| `baseline.png` | copy of the committed baseline | `cp build_macos_metal/ExternalData/Rendering/Volume/Testing/Data/Baseline/TestGPURayCastCameraInsideTransformation.png /tmp/bc/baseline.png` |
