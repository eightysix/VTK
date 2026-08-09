# Update 48 — replay tooling, data, and regeneration instructions

Persistent copies of the scripts behind
[`VolumeRayCastBackendComparisonFindingsUpdate48.md`](../../VolumeRayCastBackendComparisonFindingsUpdate48.md)
(commit `430946b931`, branch `metal-ios`), which closed the Metal side of the
volume-raycast comparison:

- **Metal's pipeline is reproduced to the stored LSB** on all 68 gated pixels
  by a CPU float32 replay of the accumulate lattice
  (`evalPoint₀ = f32(localPos+evalStep)`, nearest texel, validated TF tables,
  exact-fma composite, `accA ≥ 1−1/255` break, round-half-even store).
- **GL's own debug-dumped lattice reproduces clean GL only where
  clean-GL == clean-Metal** (10/14 shared pixels); at `(93,201)` it recomposes
  to *Metal* while clean GL is one channel higher — confirming update 44's
  "debug-GL tracks Metal, clean GL diverges at compile level".
- **Lattice params agree to <1e-6** (anchor ≤ 6.6e-7, step ≤ 1.6e-7),
  ruling out geometry for the 63,690-px ±1 field, and a new knife-edge
  mechanism (grid-aligned rays) explains the 15 gated ±1 pixels.

## What is persisted here

| file | role |
|---|---|
| `replay_metal_accumulate.py` | 68/68 Metal byte-match + the 15 gated clean-GL-differs pixels (update 48 §1–2) |
| `replay_gl_lattice.py` | GL `GL_RAY` lattice replay on the 14 shared pixels + lattice-param bounds (update 48 §2–3) |
| `knife_edge_422_92.py` | (422,92) knife-edge amplification, frame-1 vs frame-6 step (update 48 §4) |

The **input data is NOT committed** (large; see sizes below). It lives in
`/tmp/bc` on the machine where the findings were produced and can be
regenerated with the commands below. Point the scripts at any copy with
`BC_DATA=/path/to/data`.

## Data inventory (not committed — regenerate or keep in `/tmp/bc`)

| file | size | source |
|---|---|---|
| `vol512.npy` | 1074 MB | regenerated from ExternalData (`make_vol512.py` below); float64 512³, integer texel values in [0, 4370] |
| `u47_metal.log` | 22 MB | Metal test run, shader `os_log` dump (STEP/MARCH/SAMPLE/FINAL rows), 6 deterministic renders per gated pixel |
| `u47_metal.png` | 112 KB | Metal's stored image (512×512) |
| `u47_gl.png` | 111 KB | clean GL's stored image (512×512) |
| `gl372.log` | 4 MB | debug-injected GL run (`VTK_GL_RAY_DUMP=1`), `GL_RAY` lines for 15 pixels |

## Regeneration instructions

### 0. Build with tests (shader logging) and create the dummy baseline

```sh
./macos_metal_build.sh --resume --tests
BIN=build_macos_metal/bin/vtkRenderingVolumeCxxTests
EXT=build_macos_metal/ExternalData/Testing
TMP=build_macos_metal/Testing/Temporary
BASELINE=/tmp/bc/dummy_baseline.png
python3 -c "from PIL import Image; Image.new('RGB',(512,512),(0,0,0)).save('$BASELINE')"
```

The wrong-baseline trick makes the test fail-and-dump its render, so the
saved output images and the shader log both come out of the same invocation.

### 1. Volume (`vol512.npy`)

```sh
python3 ../make_vol512.py \
  build_macos_metal/ExternalData/Testing/Data/headsq/quarter \
  /tmp/bc/vol512.npy
```

(script: `Rendering/Metal/BackendComparisonTools/make_vol512.py`; output range
should be [0, 4370], dtype float64 — the replay scripts index `vol[x][y][z]`.)

### 2. Metal capture (`u47_metal.log` + `u47_metal.png`)

Shader `os_log` is buffered per command buffer and silently dropped unless all
three `MTL_LOG_*` vars are set (see
`Rendering/Metal/Testing/Cxx/TestMetalVolumeShaderLog.cxx`).

```sh
MTL_LOG_LEVEL=MTLLogLevelDebug \
MTL_LOG_BUFFER_SIZE=16777216 \
MTL_LOG_TO_STDERR=1 \
  "$BIN" TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=Metal \
    -D "$EXT" -T "$TMP" -V "$BASELINE" 2> /tmp/bc/u47_metal.log
cp "$TMP/dummy_baseline.png" /tmp/bc/u47_metal.png
# verify: rg -c 'DEBUG STEP px=' /tmp/bc/u47_metal.log   (expect 68 unique gated px)
```

### 3. Clean GL capture (`u47_gl.png`)

```sh
"$BIN" TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
  --vtk-factory-prefer RenderingBackend=OpenGL \
  -D "$EXT" -T "$TMP" -V "$BASELINE" > /dev/null 2>&1
cp "$TMP/dummy_baseline.png" /tmp/bc/u47_gl.png
```

Clean GL is deterministic across runs (update 44: 0 px between two captures).
Two clean-GL captures must be byte-identical.

### 4. Debug-injected GL capture (`gl372.log`)

The `GL_RAY` dump lives in
`Rendering/VolumeOpenGL2/vtkOpenGLGPUVolumeRayCastMapper.cxx`
(`if (getenv("VTK_GL_RAY_DUMP"))`). It dumps the exact `g_rayOrigin` /
`g_dirStep` for a fixed list of 15 pixels (including `(256,256)`, `(422,419)`,
`(93,310)`, `(480,111)`, …) on every frame, without changing the ray geometry
for the other pixels. **Note:** the injected GLSL compiles differently from
clean GL (update 44) — use `gl372.log` only for the *ray geometry*, never as a
clean-image reference.

```sh
VTK_GL_RAY_DUMP=1 \
  "$BIN" TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=OpenGL \
    -D "$EXT" -T "$TMP" -V "$BASELINE" 2> /tmp/bc/gl372.log
# verify: rg -c 'GL_RAY px=' /tmp/bc/gl372.log   (expect 15 unique px * 6 frames)
```

## Running the replay scripts

```sh
cd Rendering/Metal/BackendComparisonTools/update48
BC_DATA=/tmp/bc python3 replay_metal_accumulate.py   # -> metal-match: 68/68
BC_DATA=/tmp/bc python3 replay_gl_lattice.py         # -> 10/14 GLOK split + lattice bounds
BC_DATA=/tmp/bc python3 knife_edge_422_92.py         # -> (238,192,159) vs (238,190,157) vs (238,176,140)
```

Requires `numpy` + `Pillow`. The TF/accumulator model in each script is
self-contained (opacity pre-integration with factor 0.270059, composite over
(0,4370), nearest sampling, exact-fp64-product fma, `accA ≥ 1−1/255` break,
round-half-even store).

## Commit / history anchors

- Update 48 doc: commit `430946b931` — `Rendering/Metal/VolumeRayCastBackendComparisonFindingsUpdate48.md`
- Prior milestones: update 47 (`042b712836`), update 46 (`4fe520b585`,
  `b057f910ee`, `970e84d6d5`), update 45 (`8083e26aca`), update 44
  (`6296a3ae93`).
- Canonical capture-procedure cheatsheet:
  `Rendering/Metal/VolumeRayCastBackendComparisonProcedures.md`
  ("Verified logging invocations" section) — the same `MTL_LOG_*` /
  `VTK_GL_RAY_DUMP` invocation patterns are documented there.
