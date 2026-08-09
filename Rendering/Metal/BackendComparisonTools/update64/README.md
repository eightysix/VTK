# Update 64 — dense full-frame attribute-field instrumentation

Persistent README behind
[`VolumeRayCastBackendComparisonFindingsUpdate64.md`](../../VolumeRayCastBackendComparisonFindingsUpdate64.md).

## What this milestone adds

Two new capture hooks to replace the 14-px knife-edge window of updates 61–63
with a full-frame, per-pixel measurement of the interpolated attributes, so the
effective sample displacement (update 63) can be backed out at scale:

1. **GL full-frame raw attribute dump** (`VTK_GL_ATTR_DUMP`,
   `VTK_GL_ATTR_DUMP_OUT`) in
   `Rendering/VolumeOpenGL2/vtkOpenGLGPUVolumeRayCastMapper.cxx`:
   `DumpDebugAttrField()` renders the debug-injected shader into an RGBA32F FBO
   with `in_debugChannel` = 200/201/202 and appends 3 raw float4 planes per
   frame (texcoords+flatVid, clip, vertexPos+primId). Requires the new
   channel-200–202 fragment-shader block and the channel-100 gate fix
   (`< 110`).
2. **Metal dense STEP/MARCH gate** (`debugStepGate`) in
   `Rendering/Metal/Shaders/MetalShaders.metal`: adds a sparse 64×64 grid
   (every 8th pixel) plus a dense 64×64 block around the knife edge
   (x∈[317,380], y∈[223,286]) to the STEP/MARCH dump pixels for the
   CameraInsideTransformation camera. Per-sample dumps (SAMPLE/GRADOP/LIGHT)
   are unchanged.

## Data inventory (not committed)

| file | size | source |
|---|---|---|
| `u63_metal.log` | 97 MB | Metal run: `DEBUG STEP` at 8203 unique px/frame (grid + dense + legacy knife) |
| `gl_attr_dump.raw` | partial | GL raw attribute dump (6 frames = 75,497,472 bytes when complete) |

Both are captures of
`TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter`
(frame-6 last-occurrence-per-key convention).

## Regeneration

Build with `./macos_metal_build.sh --resume --tests`; capture GL with
`VTK_GL_RAY_DUMP=1 VTK_GL_VERTEX_DUMP=1 VTK_GL_ATTR_DUMP=1
VTK_GL_ATTR_DUMP_OUT=/tmp/bc/gl_attr_dump.raw` and Metal with
`MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=1073741824
MTL_LOG_TO_STDERR=1`, both against a dummy baseline (test fails-and-dumps).
Exact commands in the update-64 findings doc (§4).

Caveat: the GL-on-Metal reference run can hang (`kIOGPUCommandBufferCallback
ErrorHang`) under concurrent GPU load — capture with the GPU otherwise idle.

## Planned analysis (not yet persisted)

Per-pixel back-out of the effective sample NDC displacement at full frame
(512×512), GL and Metal side by side, mirroring `backout_u63.py` /
`displace_u63.py` but over the full grid, plus structure search (constant
shift / per-triangle bias / viewport-phase correlation) between the two
displacement fields.

## Commit anchors

- This update: work in progress on `metal-ios`.
- Prior: update 63 `73a1b52cab`, update 62 `e1e312445a`/`c75fc18c1a`,
  update 61 `7919acbb98`/`4c250a654d`, update 60 `8109341d27`.
- Capture cheatsheet: `Rendering/Metal/VolumeRayCastBackendComparisonProcedures.md`.
