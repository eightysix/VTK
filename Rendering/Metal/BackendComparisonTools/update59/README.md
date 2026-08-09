# Update 59 — replay tooling, data, and regeneration instructions

Persistent copies of the scripts behind
[`VolumeRayCastBackendComparisonFindingsUpdate59.md`](../../VolumeRayCastBackendComparisonFindingsUpdate59.md)
(commit `696bbb9be3`, branch `metal-ios`), which localized the last 188-px
Metal-vs-GL residual to **knife-edge nearest-texel flips driven by a ~1-ulp
interpolated-anchor difference**:

- The GL float dump was frame-1 while the stored image is frame-6 (camera
  animates), hiding the real gf delta. Fix: dump every frame, last write =
  frame 6.
- The **frame-6-aligned** GL gf dump reproduces clean GL at 262,141/262,144 px
  (3 ultra-boundary px, none in the 188 Metal-diff set) → the aligned dump is
  trustworthy at every pixel we care about.
- True Metal-vs-GL gf delta at the 188 px: mean 0.31 u8 / max 8.25 u8 with
  alpha |Δ| ≤ 0.0015 → large color, negligible alpha = **texel-selection flip**,
  not composite arithmetic (operation-order bisect is dead: mul+add moved 4 px).
- Frame-6 lattice comparison at the 14 big-delta px: `evalStep` vs `g_dirStep`
  bit-identical to ~5e-8 texels; **anchor** (`g_rayOrigin` vs `localPos+evalStep`)
  differs by 2e-5…6.7e-5 texels (~1 ulp of the interpolated texcoord);
  interpolated clip matches to ~6e-8 (x/y/w; z conventions differ between dumps).

## What is persisted here

| file | role |
|---|---|
| `aligned_gl_gf_validate.py` | Parse the frame-6-aligned GL gf dump, flip rows, verify it reproduces clean GL via `round_half_even((gf + 26/255·(1−a))·255)` at 262,141/262,144 px (update 59 §1) |
| `metal_vs_gl_gf.py` | Metal vs GL gf/alpha comparison at the 188 stored-diff px + the 14 big-|Δ| px table (update 59 §2–3) |
| `lattice_u61.py` | GL_RAY vs Metal STEP lattice comparison at the 14 knife-edge px: step ~bit-identical, anchor ~1 ulp, clip x/y/w ~6e-8 (update 59 §4) |
| `parse_metal_final.py` | Parse the Metal full-field pre-store FINAL dump log into a numpy .npy (last frame wins), regenerating `u59_metal_float.npy` (update 58 §1) |

The **input data is NOT committed** (large; see sizes below). It lives in
`/tmp/bc` on the machine where the findings were produced and can be
regenerated with the commands below. Point the scripts at any copy with
`BC_DATA=/path/to/data`.

## Data inventory (not committed — regenerate or keep in `/tmp/bc`)

| file | size | source |
|---|---|---|
| `u60_gl_float.raw` | 4 MB | frame-6-aligned GL pre-store gf dump (RGBA float32, 512×512×4, row 0 = gl_FragCoord y 0). Produced by `DumpCleanGLFloats` with the update-59 per-frame change |
| `u60_gl_clean.png` | 112 KB | clean GL stored image (== `u59_gl.png` byte-identical; GL is deterministic) |
| `u59_metal.png` | 112 KB | Metal stored image |
| `u59_metal_float.npy` | 16 MB | Metal frame-6 pre-store gf + alpha + lastIter (parsed from the FINAL dump log) |
| `u61_gl.log` | 44 MB | debug-injected GL run, `GL_RAY` lines for 29 px (15 legacy + 14 knife-edge) |
| `u61_metal.log` | 27 MB | Metal run, `STEP` lines (MTL_LOG capture) for the gated pixels |

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

### 1. Clean GL image (`u60_gl_clean.png`)

```sh
"$BIN" TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
  --vtk-factory-prefer RenderingBackend=OpenGL \
  -D "$EXT" -T "$TMP" -V "$BASELINE" > /dev/null 2>&1
cp "$TMP/dummy_baseline.png" /tmp/bc/u60_gl_clean.png
```

### 2. Frame-6-aligned GL gf dump (`u60_gl_float.raw`)

Env-gated in `vtkOpenGLGPUVolumeRayCastMapper.cxx` (`VTK_GL_FLOAT_DUMP`). With
the update-59 change the dump runs on **every frame** and overwrites the file,
so the final content is frame 6. **Caveat:** the dump re-render corrupts the
same run's stored image (64k px vs clean), so pair the dump with a *separate*
clean image (step 1), never with the dump-run image.

```sh
VTK_GL_FLOAT_DUMP=1 VTK_GL_FLOAT_DUMP_OUT=/tmp/bc/u60_gl_float.raw \
  "$BIN" TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=OpenGL \
    -D "$EXT" -T "$TMP" -V "$BASELINE" 2> /tmp/bc/u60_gl_stderr.log
# verify: rg -c 'GL_FLOAT_DUMP wrote' /tmp/bc/u60_gl_stderr.log  (expect 6)
```

### 3. Metal stored image + pre-store float dump

Metal float dump (`VTK_METAL_FLOAT_DUMP`, update 58) turns the FINAL log into a
dump-all mode; parse with last-frame-wins (below).

```sh
VTK_METAL_FLOAT_DUMP=1 MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=1073741824 \
MTL_LOG_TO_STDERR=1 \
  "$BIN" TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=Metal \
    -D "$EXT" -T "$TMP" -V "$BASELINE" 2> /tmp/bc/u59_metal_float.log
cp "$TMP/dummy_baseline.png" /tmp/bc/u59_metal.png
BC_DATA=/tmp/bc python3 parse_metal_final.py
```

### 4. Lattice capture (14 knife-edge px in both dump gates)

```sh
# Metal STEP log (uses debugMarchGate pxOkKnife, MetalShaders.metal)
MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=1073741824 MTL_LOG_TO_STDERR=1 \
  "$BIN" TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=Metal \
    -D "$EXT" -T "$TMP" -V "$BASELINE" 2> /tmp/bc/u61_metal.log

# GL GL_RAY log (VTK_GL_RAY_DUMP, 29 px; debug-injected -> geometry only)
VTK_GL_RAY_DUMP=1 \
  "$BIN" TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=OpenGL \
    -D "$EXT" -T "$TMP" -V "$BASELINE" 2> /tmp/bc/u61_gl.log
```

## Running the scripts

```sh
cd Rendering/Metal/BackendComparisonTools/update59
BC_DATA=/tmp/bc python3 aligned_gl_gf_validate.py   # -> 262141/262144 px, 3 ultra-boundary px NOT in the 188 set
BC_DATA=/tmp/bc python3 metal_vs_gl_gf.py           # -> 188 px, gf |d| mean 0.31 / max 8.25 u8, alpha <= 0.0015
BC_DATA=/tmp/bc python3 lattice_u61.py              # -> step ~5e-8 texels, anchor 2e-5..6.7e-5 texels (~1 ulp)
```

Requires `numpy` + `Pillow`. The TF/accumulator model for the gf→byte
conversion is self-contained: `round_half_even((gf + 26/255·(1−a))·255)` where
26/255 = the test's `SetBackground(0.1,0.1,0.1)` stored as u8 (update 56).

## Commit / history anchors

- Update 59 doc + tooling: commit `696bbb9be3` — `Rendering/Metal/VolumeRayCastBackendComparisonFindingsUpdate59.md`
- Prior milestone: update 58 (`651e3ea4ae`), update 57 (`8f991da45b`),
  update 56 (`fdd7281d07`), update 48 tooling (`1d1fe1562e`).
- Canonical capture-procedure cheatsheet:
  `Rendering/Metal/VolumeRayCastBackendComparisonProcedures.md`.
