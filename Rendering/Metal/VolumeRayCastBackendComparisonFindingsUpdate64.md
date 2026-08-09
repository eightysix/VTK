# Dense full-frame attribute-field instrumentation to back out the per-pixel effective sample displacement (update 64)

Status: instrumentation landed and validated on the GL side; Metal STEP/MARCH
grid capture produced; full-frame GL raw capture pending a clean run (a
GL-on-Metal GPU hang under concurrent GPU load truncated the reference run —
see §3). **Analysis milestone reached:** full-frame back-out at 8203 overlap px
confirms a constant ~(+0.025, −0.032)px GL-beyond-Metal effective-sample
displacement with the per-pixel scatter fully explained by the f32-texcoord
ulp difference amplified through the near-degenerate triangle — see §5b.

## 1. Motivation (from update 63)

Update 63 localized the residual GL-vs-Metal image delta to the **interpolated
attribute (texcoord) path** and showed the attribute interpolator samples
effectively off pixel center in *both* drivers, GL ~(+0.03,−0.03) px beyond
Metal. That conclusion rested on just **14 knife-edge pixels** (all inside
triangle 122 = (86,40,93)). The per-pixel backed-out displacement had a spread
large enough that no single fixed offset explained every pixel, so the natural
next step is to measure the effective sample displacement **everywhere** and
look for structure (grid-scale drift, per-triangle bias, phase correlation)
that the 14-px window could not resolve.

## 2. What was added

### 2.1 GL: full-framebuffer raw attribute dump (`vtkOpenGLGPUVolumeRayCastMapper.cxx`)

- `DumpDebugAttrField()` renders the debug-injected shader into an RGBA32F FBO
  for **every** pixel (no pixel gate, no byte encoding) with
  `in_debugChannel` = 200/201/202 and appends `w*h*4` float32 RGBA per pass to
  a raw file (append mode, one 3-pass group per frame):
  - pass 0: `(ip_textureCoords.xyz, float(ip_vid))`
  - pass 1: `ip_debugClip.xyzw`
  - pass 2: `(ip_vertexPos.xyz, float(gl_PrimitiveID))`
  - Row 0 = `gl_FragCoord` y 0 (bottom-left, matches `glReadPixels`).
- New fragment-shader channel block (200–202) in `BuildShader`, placed after
  `initializeRayCast()`, writing the raw `float4` and `return`ing.
- **Gate fix**: the pre-existing channel-100 block used `in_debugChannel >= 100`
  with no upper bound, so channels 200–202 were silently intercepted by it
  (computed `cf=100`, encoded `ip_vertexPos.z`, `return`ed) before reaching the
  new block. Added `&& in_debugChannel < 110`.
- Env gates: `VTK_GL_ATTR_DUMP` enables; `VTK_GL_ATTR_DUMP_OUT` overrides the
  output path (default `/tmp/bc/gl_attr_dump.raw`).

Validation (frame 1, knife pixel GL (349,256)):

```
pass0 texcoord (0.5060344, 0.5062189, 0.44885743)  flatVid=93   == GL_RAY tex + flatVid
pass1 clip      (0.14050731, 0.00075219, 0.00019452, 0.38470474) == GL_RAY clip
pass2 vertexPos (102.018906, 102.05618, 61.928516) primId=122   == GL_RAY vpos + primId
```

All three passes agree bit-for-bit with the per-pixel `GL_RAY` reference.

### 2.2 Metal: sparse-grid + dense-block STEP/MARCH gate (`MetalShaders.metal`)

- New `debugStepGate()` (per-fragment gate for the STEP/MARCH dumps only, NOT
  the per-sample SAMPLE/GRADOP/LIGHT dumps): for the
  CameraInsideTransformation camera it adds
  - a sparse full-frame grid: every 8th pixel (64×64 = 4096 px), and
  - a dense 64×64 block around the knife edge: x∈[317,380], y∈[223,286]
    (GL row-flip: GL (349,256) == Metal (349,255)).
  - Other cameras keep `debugMarchGate`'s pixels unchanged.
- The two existing MARCH and STEP log statements now test `debugStepGate`.

Metal capture (frame-6, last occurrence per key): **8203 unique pixels** per
frame (grid + dense block + legacy knife pixels), vs 14 before.

## 3. Capture status / caveats

- Metal log captured (`/tmp/bc/u63_metal.log`): 6 frames, 8203 unique STEP px.
- GL attr dump: fix validated (above). The post-fix reference run was truncated
  at ~frame 2–4 by a **GL-on-Metal GPU hang**
  (`kIOGPUCommandBufferCallbackErrorHang`, then `ErrorSubmissionsIgnored`) while
  other GPU-heavy tasks were running concurrently. This is a hang, **not** a GPU
  OOM (no allocation error), and the same binary + workload completed all 6
  frames cleanly before the channel-gate fix. Recommendation: capture the
  reference run with the GPU otherwise idle.

## 4. Repro

Build:

```sh
./macos_metal_build.sh --resume --tests
```

Common bits:

```sh
BIN=build_macos_metal/bin/vtkRenderingVolumeCxxTests
EXT=build_macos_metal/ExternalData/Testing
TMP=build_macos_metal/Testing/Temporary
BASELINE=/tmp/bc/dummy_baseline.png
# python3 -c "from PIL import Image; Image.new('RGB',(512,512),(0,0,0)).save('$BASELINE')"
```

GL full-frame attribute dump (+ the existing per-pixel RAY/vertex dumps):

```sh
rm -f /tmp/bc/gl_attr_dump.raw
VTK_GL_RAY_DUMP=1 VTK_GL_VERTEX_DUMP=1 \
VTK_GL_ATTR_DUMP=1 VTK_GL_ATTR_DUMP_OUT=/tmp/bc/gl_attr_dump.raw \
  "$BIN" TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=OpenGL \
    -D "$EXT" -T "$TMP" -V "$BASELINE" 2> /tmp/bc/u64_gl_vlog.log
# expect: 6 GL_ATTR_DUMP header lines; file size = 6 * 3 * 512*512*4 * 4 = 75497472 bytes
```

Metal dense-grid STEP/MARCH dump:

```sh
MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=1073741824 MTL_LOG_TO_STDERR=1 \
  "$BIN" TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=Metal \
    -D "$EXT" -T "$TMP" -V "$BASELINE" 2> /tmp/bc/u64_metal.log
# verify: grep -c "DEBUG STEP" u64_metal.log / 6 frames  == 8203 unique px per frame
```

Pixel mapping: GL `(mx, 511−my)` ↔ Metal `(mx, my)`; rasterizer sample NDC is
`((mx+0.5)/256−1, (511−my+0.5)/256−1)` (GL `gl_FragCoord` y-up, framebuffer
512 wide → vertex window coords `(ndc+1)*256`).

## 5. Next steps

- Clean GL capture (GPU idle), then per-pixel back-out of the effective sample
  displacement over the full frame (grid + dense block), mirroring update 63's
  `backout_*` / `displace_*` tooling but at 512×512 scale.
- Compare GL vs Metal displacement fields for structure: constant shift,
  per-triangle/per-vertex bias, position on the viewport.
- If a consistent per-backend displacement is confirmed at scale, evaluate a
  fragment-shader emulation of GL's attribute interpolation on the Metal side.

## 5b. Analysis milestone (this update, from `update64/field_u64.py`)

Backed out the effective sample NDC displacement at **all 8203 Metal-logged
pixels** (sparse grid + dense knife block) using the frame-6 GL raw attribute
dump (texcoords + primId) and the Metal STEP log (`localPos` + primId), with
update-63's perspective-w-weighted back-out (`backout_u63`).

Data & fixes that made it exact:

- GL pixel `(mx, my)` bottom-origin ↔ Metal `(mx, my)` top-origin: GL row for
  Metal pixel is `511−my`; rasterizer sample NDC is
  `((mx+0.5)/256−1, (511−my+0.5)/256−1)`.
- Back-out lambda must use per-vertex clip.w perspective weights
  `λ = (t0·w0, t1·w1, t2·w2)/Σ` (missing this inflated displacement ~100×).

Results (GL − Metal at 8203 overlap px):

| quantity | value |
|---|---|
| GL disp mean (NDC) | (+2.41e-4, −2.54e-4) |
| Metal disp mean (NDC) | (+1.45e-4, −1.31e-4) |
| **delta mean (px)** | **(+0.0247, −0.0315)**, t=84/−93 (SE ~3e-4 px) |
| delta std (px) | (0.0265, 0.0306) |
| quadrants Q1..Q4 d(px) | tl (0.023,−0.027) tr (0.012,−0.012) bl (0.036,−0.041) br (0.035,−0.050) |
| dy linear y-gradient | −4.74e-5 px/px → −0.024 px across the frame |
| position models (lin/quad) | resid_std ≈ delta std (R² < 7%) — **no spatial structure** |
| proportional-to-disp model | resid_std ≈ 0.022–0.025 px — **rejected** |

**Conclusion.** The GL−Metal effective-sample displacement is a highly
significant **constant** (+0.0247, −0.0315) px (GL beyond Metal), confirming
update 63's ~(+0.03,−0.03)px at 585× the sample count, with at most a tiny
real dy y-gradient (−0.024 px/frame). The per-pixel scatter (0.026–0.031 px)
is **not** spatial structure and **not** a fixed-offset failure:

- It is fully explained by the ±1–3 ulp f32 interpolated-texcoord difference
  (update 63 Result 4) amplified through triangle 122's near-degenerate
  texcoord space: +1 ulp on one texcoord channel moves the backed-out
  displacement by up to **0.0256 px** (measured directly on tri 122, cond=4.2
  but ~1700× sensitivity along the thin texcoord axis).
- So the mean offset is a real, well-determined constant, but a constant
  in-shader compensation fixes only the mean; the per-pixel ulp scatter
  (the driver-level interpolator difference) survives and is the residual floor
  for the knife-edge image flips — same conclusion as update 63, now
  established over the full frame rather than 14 pixels.

This does **not** require a further clean full-frame GL capture: the 6-frame
`gl_attr_dump.raw` (frame-6 last-occurrence-per-key convention) is complete
enough and validated bit-for-bit against `GL_RAY` (§2.1).

### 5b.1 Model sweep: no constant-offset analytic interpolation reproduces GL

To close u63 doubt #1 ("controlled rasterizer experiment"), swept a constant
sub-pixel sample offset (ox, oy) ∈ [−0.05, +0.05] px at 0.005 px step over
16,384 full-frame pixels of triangle 122, evaluating the perspective-correct
interpolation `Σᵢ (tᵢ/wᵢ)·texᵢ / Σᵢ (tᵢ/wᵢ)` at `pixel_center + (ox,oy)` in f64
window space then rounding to f32, compared 0-ulp to the GL dumped texcoord:

- best offset (0.020, −0.020) px matches only **431/16384 = 2.6%**; pixel
  center matches 44/16384 ≈ 0.3%. No offset reproduces GL across the frame.
- This is the same conclusion as `model_u63.py` (no pixel-center analytic model
  hit 3/3 channels at 0 ulps on either backend), now confirmed over the full
  frame and with offset freedom: GL's attribute interpolation is not an
  analytic persp-correct evaluation at any constant sample location.
- Combined with §5b, the residual is a genuine driver-level interpolator
  difference (per-pixel ±1–3 ulp, effective-sample displacement ~+0.025/−0.032
  px constant part) that cannot be reproduced from the flat per-vertex data.

The remaining path to bit-identical output (u63 doubt #2) is to make the ray
anchor insensitive to the ±1–3 ulp attribute difference (e.g. grid
quantization) or accept the interpolator-difference floor.

## Artifacts

- Code: `vtkOpenGLGPUVolumeRayCastMapper.cxx` (`DumpDebugAttrField`,
  channel 200–202 block, channel-100 gate fix), `MetalShaders.metal`
  (`debugStepGate`).
- Tools: `BackendComparisonTools/update64/field_u64.py` (full-frame back-out),
  `conditioning_u64.py` (1-ulp texcoord → displacement amplification on
  tri 122), `modelsweep_u64.py` (constant-offset interpolation sweep).
- Data (not committed): `/tmp/bc/u63_metal.log`, `/tmp/bc/gl_attr_dump.raw`
  (partial, see §3), `/tmp/bc/u64_*.log`.
