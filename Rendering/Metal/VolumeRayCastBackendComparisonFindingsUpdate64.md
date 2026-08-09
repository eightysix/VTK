# Dense full-frame attribute-field instrumentation to back out the per-pixel effective sample displacement (update 64)

Status: instrumentation landed and validated on the GL side; Metal STEP/MARCH
grid capture produced; full-frame GL raw capture pending a clean run (a
GL-on-Metal GPU hang under concurrent GPU load truncated the reference run —
see §3).

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

## Artifacts

- Code: `vtkOpenGLGPUVolumeRayCastMapper.cxx` (`DumpDebugAttrField`,
  channel 200–202 block, channel-100 gate fix), `MetalShaders.metal`
  (`debugStepGate`).
- Data (not committed): `/tmp/bc/u63_metal.log`, `/tmp/bc/gl_attr_dump.raw`
  (partial, see §3), `/tmp/bc/u64_*.log`.
