# Full debug/clean 2x2 capture matrix: debug-injected GLSL is bit-identical to Metal; clean GL residuals now 183 px (update 72)

**Date:** 2026-08-10
**Status:** **milestone (measurement).** Fresh 512x512 captures of all four
states with the correct `-V $TMP/<unique>.png` invocation (render lands in
`$TMP/<basename>.png`; stale `$TMP/TestGPURayCastCameraInsideTransformation.png`
is frozen at update-34 md5 and must never be used). The matrix proves:

| state | invocation | md5 |
|---|---|---|
| clean Metal | `--vtk-factory-prefer RenderingBackend=Metal -V $TMP/mtA.png` | `5e723c7d384a8ab4458c0ea0920f1d58` |
| clean Metal (2nd) | same, `mtB.png` | `5e723c7d384a8ab4458c0ea0920f1d58` |
| clean GL | `--vtk-factory-prefer RenderingBackend=OpenGL2 -V $TMP/glA.png` | `dc4bab2eb48d8f894babc6fd801193b1` |
| clean GL (2nd) | same, `glB.png` | `dc4bab2eb48d8f894babc6fd801193b1` |
| debug Metal | `MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=33554432 MTL_LOG_TO_STDERR=1` + Metal + `-V $TMP/mtD.png` | `5e723c7d384a8ab4458c0ea0920f1d58` |
| debug GL | `VTK_GL_RAY_DUMP=1` (only, no sample/final dump) + GL + `-V $TMP/glD.png` | `5e723c7d384a8ab4458c0ea0920f1d58` |

Copies: `/tmp/bc/u72/{clean_metal,clean_gl,debug_metal,debug_gl}.png`.

## 1. Pairwise pixel diffs (all 512x512, RGB)

```
clean_Metal   vs clean_Metal_2 :      0 px  (deterministic)
clean_GL      vs clean_GL_2    :      0 px  (deterministic)
clean_Metal   vs debug_Metal   :      0 px
clean_Metal   vs debug_GL      :      0 px
debug_Metal   vs debug_GL      :      0 px
clean_Metal   vs clean_GL      :    183 px, max channel delta 8
```

1. **Debug logging never perturbs Metal:** the `MTL_LOG_*` run (`mtD`) is
   byte-identical to the clean Metal run — the embedded debug logging in
   `MetalShaders.metal` compiles out to the same arithmetic (reconfirms update 22).
2. **Debug-injected GLSL == Metal bit-for-bit.** `VTK_GL_RAY_DUMP=1` changes
   only the compiled GLSL (shader source injection gated on `DebugRayDump`,
   `vtkOpenGLGPUVolumeRayCastMapper.cxx:3001`), and that compile produces an
   **entirely identical 512x512 image to Metal** (0 px, identical md5). This
   confirms update 44's conclusion at whole-image level: the GLSL driver's
   optimization of the *clean* source is the odd one out, not Metal.
3. **The gated 29-ray re-render loop does not corrupt the `-V` capture.** With
   only `VTK_GL_RAY_DUMP=1` set (no `VTK_GL_SAMPLE_DUMP` / `VTK_GL_FINAL_DUMP`),
   the per-pixel readback loop (`vtkOpenGLGPUVolumeRayCastMapper.cxx:4687`) leaves
   the presented framebuffer as the real render — `glD` has zero corrupted pixels
   (it is exactly Metal). Sample/final dump env vars do overwrite the window with
   encoded bytes and are for per-sample logs only, not image captures.
4. **Residual clean-GL vs Metal field is down to 183 px, max channel delta 8.**
   This is a fresh-binary measurement (the update-44-era 63,691 px field was
   measured on the old binary). Both backends are independently deterministic, so
   the 183 px is a stable arithmetic residual.

## 2. The residual pixels (clean GL minus Metal), worst first

```
(397,110) GL-Metal=(0,-7,-8)  Metal=(239,186,151) GL=(239,179,143)
(360,229) GL-Metal=(0, 3, 4)  Metal=(225,174,141) GL=(225,177,145)
(405,171) GL-Metal=(0,-2,-3)  Metal=(230,168,133) GL=(230,166,130)
(349,255) GL-Metal=(0,-2,-3)  Metal=(226,182,151) GL=(226,180,148)
(482, 33) GL-Metal=(0,-1,-2)  Metal=(242,163,123) GL=(242,162,121)
(470,269) GL-Metal=(0, 1, 2)  Metal=(228,145,105) GL=(228,146,107)
(469,463) GL-Metal=(0,-1,-2)  Metal=(242,171,133) GL=(242,170,131)
(439,281) GL-Metal=(0, 1, 2)  Metal=(229,166,130) GL=(229,167,132)
(350,  5) GL-Metal=(0,-1,-2)  Metal=(246,188,152) GL=(246,187,150)
(338,432) GL-Metal=(0, 1, 2)  Metal=(241,167,128) GL=(241,168,130)
(293,298) GL-Metal=(0,-2,-1)  Metal=(231,158,120) GL=(231,156,119)
(153, 32) GL-Metal=(0, 2, 1)  Metal=(247,183,146) GL=(247,185,147)
(120,167) GL-Metal=(0, 2, 2)  Metal=(245,177,139) GL=(245,179,141)
(  9, 18) GL-Metal=(0, 2, 2)  Metal=(248,193,157) GL=(248,195,159)
...remaining 169 px are single-channel ±1.
```

Delta histogram (GL-Metal, RGB): `(0,+1,0)` x27, `(0,0,+1)` x27,
`(0,+1,+1)` x25, `(0,0,-1)` x23, `(0,-1,-1)` x16, `(-1,0,0)` x16,
`(0,-1,0)` x15, `(+1,0,0)` x14 — dominated by single-channel ±1 rounding.
Red channel is almost always equal (GL-Metal red is 0 for ~153/183 px).

## 3. Invocation gotchas (to keep results valid)

- Always capture with `-V $TMP/<unique-name>.png`; the render lands in
  `$TMP/<basename>.png`. The default-filename image
  `$TMP/TestGPURayCastCameraInsideTransformation.png` is **stale** (update-34
  md5 `becc616…`) and must not be diffed.
- Both backends are deterministic across captures (0 px self-diff), so any nonzero
  delta is backend arithmetic, not jitter.
- For image captures: `VTK_GL_RAY_DUMP=1` alone; for per-sample logs add
  `VTK_GL_SAMPLE_DUMP=1 VTK_GL_SAMPLE_DUMP_PX=x,y` but expect the window buffer to
  be overwritten with encoded bytes.

## 4. Artifacts

- `/tmp/bc/u72/clean_metal.png`, `/tmp/bc/u72/clean_gl.png`,
  `/tmp/bc/u72/debug_metal.png`, `/tmp/bc/u72/debug_gl.png` (all fresh, 512x512).
