# Update 61 — barycentric weight reconstruction, arithmetic-variant search, and sample-point fit

Persistent copies of the scripts behind
[`VolumeRayCastBackendComparisonFindingsUpdate61.md`](../../VolumeRayCastBackendComparisonFindingsUpdate61.md)
(commit `7919acbb98`, branch `metal-ios`), which quantified the update-60
residual (per-vertex clip + texcoord already bit-identical) by reconstructing
the rasterizer's barycentric weights at the 14 knife-edge pixels:

- **All 14 knife-edge px are covered by one triangle**, 122 = (86, 40, 93)
  (flatVid 93). The triangle is huge in NDC (vid 40 y≈980, vid 93 x≈−305):
  weights are strongly peaked (~0.91 / ~0.08 / ~0.006), so the backout is
  ill-conditioned and near-zero clip.y ulp counts are magnitude artifacts.
- **Backed-out weights are identical GL vs Metal** to ≤ 8.4e-8 relative
  (`bary_u61.py`); the implied sample point equals the exact pixel center to
  ±1e-6 NDC (`samplept_u61.py`) — the pixel-center convention is confirmed for
  both backends, no half-pixel flip.
- **The rasterizer's vertex divide is f32, not f64.** A f64-NDC weight model is
  off by 10-80 ulps while the f32-NDC model is within ~4 (`consist_u61.py`,
  `arith_u61.py`).
- **Every interpolation-arithmetic variant shows a systematic positive bias**
  (GL +2.55, Metal +1.52 mean ulps; `variants_u61.py`): the residual is a
  weight bias, not the interpolation arithmetic.
- **A fitted sub-pixel sample offset reduces but never zeroes the gap**
  (`fit_u61.py`): GL err 107→37 ulps at dx≈+2.2e-4/dy≈−3.8e-4, Metal 68→26 ulps
  at dx≈+2.4e-4/dy≈−2.0e-4 — and the two backends' best offsets differ.

Conclusion: per-vertex clip/texcoord, pixel center, and barycentric weights are
all exonerated; the 1-3 ulp interpolated-anchor delta originates below the
shader (GL-on-Metal driver position post-processing), outside the Metal source.

## What is persisted here

| file | role |
|---|---|
| `vertex_u62.py` | Per-vertex bit-compare GL_VERT vs Metal `vertex_volume_main` across all 94 vids (clip x/y/w, texcoord x/y/z) — both bit-identical (update 60 re-check) |
| `bary_u61.py` | Backs out barycentric weights from each backend's interpolated clip (`backout_weights`), compares backGL/backMt, analytic f32-NDC and f64-NDC weights, and predicts interpolated clip/tex in ulps (update 61 §3-4) |
| `samplept_u61.py` | Backs out the implied sample NDC from the interpolated clip and compares to the exact pixel-center NDC (update 61 §3) |
| `consist_u61.py` | f64-NDC weight model vs both backends' interpolated clip in ulps — demonstrates the f64 model is off by 10-80 ulps (motivates f32 divide) |
| `arith_u61.py` | seq/fma/f64 interpolation variants vs both backends' logged texcoords per pixel/axis (update 61 §4) |
| `variants_u61.py` | Exhaustive 6-variant arithmetic search (seq, fma, fma2, rcp, f64, interppos) with sum\|ulps\|/max/mean per variant (update 61 §4 table) |
| `fit_u61.py` | Scans sub-pixel sample-NDC offsets (coarse ±8e-4 @ 2e-5, fine @ 2e-6) minimizing Σ\|ulps\| for the f32/seq chain, GL and Metal separately (update 61 §5) |

The **input data is NOT committed** (large; `u62_gl_vlog.log` 44 MB,
`u62_metal.log` 27 MB). It lives in `/tmp/bc` on the machine where the findings
were produced and can be regenerated with the commands below. Point the scripts
at any copy with `BC_DATA=/path/to/data`.

## Data inventory (not committed — regenerate or keep in `/tmp/bc`)

| file | size | source |
|---|---|---|
| `u62_gl_vlog.log` | 44 MB | debug-injected GL run: `GL_VERT` (94 vids, 11 channels incl. per-vertex texcoord), `GL_RAY` (interpolated clip+tex+flatVid+primId at the gated px), `GL_CAPINDEX` (tri topology) |
| `u62_metal.log` | 27 MB | Metal run: `vertex_volume_main` (94 vids, clip+uvx+texcoord+modelPos) and `DEBUG STEP` (interpolated clip+localPos+screenPos at the gated px) |

Both are frame-6 (last-occurrence-per-key) captures of
`TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter`.

## Regeneration instructions

### 0. Build with tests

```sh
./macos_metal_build.sh --resume --tests
BIN=build_macos_metal/bin/vtkRenderingVolumeCxxTests
EXT=build_macos_metal/ExternalData/Testing
TMP=build_macos_metal/Testing/Temporary
BASELINE=/tmp/bc/dummy_baseline.png
python3 -c "from PIL import Image; Image.new('RGB',(512,512),(0,0,0)).save('$BASELINE')"
```

The wrong-baseline trick makes the test fail-and-dump its render, so the saved
output images and the shader logs come out of the same invocation.

### 1. GL vertex + ray + topology dump (`u62_gl_vlog.log`)

Gated in `vtkOpenGLGPUVolumeRayCastMapper.cxx`: `GL_VERT` by
`VTK_GL_VERTEX_DUMP`, `GL_RAY` by `VTK_GL_RAY_DUMP` (29 px: 15 legacy + 14
knife-edge), `GL_CAPINDEX` by the capmesh log. Debug-injected → geometry only,
not the stored image.

```sh
VTK_GL_RAY_DUMP=1 VTK_GL_VERTEX_DUMP=1 \
  "$BIN" TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=OpenGL \
    -D "$EXT" -T "$TMP" -V "$BASELINE" 2> /tmp/bc/u62_gl_vlog.log
```

### 2. Metal vertex + STEP dump (`u62_metal.log`)

`DEBUG STEP` (interpolated clip + localPos + screenPos) gated in
`MetalShaders.metal` (`debugMarchGate`, `pxOkKnife`); `vertex_volume_main`
(vid, modelPos, clip, uvx, texcoord) is part of the same log.

```sh
MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=1073741824 MTL_LOG_TO_STDERR=1 \
  "$BIN" TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=Metal \
    -D "$EXT" -T "$TMP" -V "$BASELINE" 2> /tmp/bc/u62_metal.log
```

## Running the scripts

```sh
cd Rendering/Metal/BackendComparisonTools/update61
BC_DATA=/tmp/bc python3 vertex_u62.py      # -> clip x/y/w bit-identical 94/94, tex 94/94
BC_DATA=/tmp/bc python3 bary_u61.py        # -> backGL vs backMt weights, rel dG-dM <= ~8e-8; clipU anF32 ~ -4 ulps vs anF64 ~ -16
BC_DATA=/tmp/bc python3 samplept_u61.py    # -> implied sample NDC == pixel center to ~1e-6
BC_DATA=/tmp/bc python3 consist_u61.py     # -> f64-NDC model off 10-80 ulps (motivates f32 divide)
BC_DATA=/tmp/bc python3 arith_u61.py       # -> per-px/axis seq|fma|f64 ulp table
BC_DATA=/tmp/bc python3 variants_u61.py    # -> 6-variant sum|ulps|: GL 102-118, Mt 63-75 (systematic positive bias)
BC_DATA=/tmp/bc python3 fit_u61.py         # -> GL err 107->37 @ dx+2.2e-4/dy-3.8e-4; Mt 68->26 @ dx+2.4e-4/dy-2.0e-4
```

Requires `numpy` (and `ctypes` libm `fmaf` on macOS for the fma variants). All
scripts read the two log files from `BC_DATA` (default `/tmp/bc`), parse
last-occurrence-per-key (frame 6), and convert GL keys `(mx, 511-my)` ↔ Metal
`(mx, my)` internally; the rasterizer sample NDC is
`((mx+0.5)/256−1, (511−my+0.5)/256−1)` (GL gl_FragCoord y-up convention).

## Commit / history anchors

- Update 61 doc + tooling: commit `7919acbb98` — `Rendering/Metal/VolumeRayCastBackendComparisonFindingsUpdate61.md`
- Prior milestone: update 60 (`8109341d27`), update 59 tooling
  (`BackendComparisonTools/update59/`), update 58 (`651e3ea4ae`),
  update 57 (`8f991da45b`), update 56 (`fdd7281d07`).
- Canonical capture-procedure cheatsheet:
  `Rendering/Metal/VolumeRayCastBackendComparisonProcedures.md`.
