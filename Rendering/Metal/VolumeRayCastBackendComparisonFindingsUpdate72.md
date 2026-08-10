# Corrected clean/debug 2x2 matrix: debug Metal == clean Metal; clean GL vs Metal = 183 px; the "debug GL == Metal" result was a wrong-flag artifact (update 72, rev 2)

**Date:** 2026-08-10
**Status:** **milestone (measurement, corrected).** Update 72 rev-1's headline
("debug-injected GLSL is bit-identical to Metal") was **wrong**: the GL captures
used `--vtk-factory-prefer RenderingBackend=OpenGL2`, which matches **no**
registered override attribute (the OpenGL2 render window registers
`RenderingBackend=OpenGL`, `vtkOpenGLRenderWindow.cxx:680`; Metal registers
`RenderingBackend=Metal`, `vtkMetalRenderWindow.mm:35`). An unmatched prefer
falls back to the default factory, which on this build is **Metal** — so both
"GL" runs actually rendered with Metal (confirmed: they printed
`VTK_METAL_VOLUME_LOG DEBUG METAL_CAM/MTL_INVPVM` and produced the Metal md5).
The 0-px "debug GL == Metal" was Metal == Metal.

Re-ran everything with the correct flag `--vtk-factory-prefer
RenderingBackend=OpenGL` (see Rendering/Volume/Testing/Cxx/CMakeLists.txt:125):

| state | invocation | md5 |
|---|---|---|
| clean Metal | `--vtk-factory-prefer RenderingBackend=Metal` | `5e723c7d384a8ab4458c0ea0920f1d58` |
| clean Metal (2nd) | same | `5e723c7d384a8ab4458c0ea0920f1d58` |
| clean GL | `--vtk-factory-prefer RenderingBackend=OpenGL` | `dc4bab2eb48d8f894babc6fd801193b1` |
| clean GL (2nd) | same | `dc4bab2eb48d8f894babc6fd801193b1` |
| debug Metal | `MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=33554432 MTL_LOG_TO_STDERR=1` + Metal | `5e723c7d384a8ab4458c0ea0920f1d58` |
| debug GL | `VTK_GL_RAY_DUMP=1` + OpenGL | `6487b74deed0cd7db822ca302fd9f469` |

(rev-1's `glA/glB` md5 `dc4bab2e` is confirmed genuine GL: a fresh
`RenderingBackend=OpenGL` capture is byte-identical.)

## 1. Corrected pairwise pixel diffs (all 512x512, RGB)

```
clean_Metal   vs debug_Metal   :      0 px  (debug logging compiles out — valid)
clean_Metal   vs clean_GL      :    183 px, max channel delta 8   <- the real residual
clean_GL      vs clean_GL(2nd) :      0 px  (deterministic)
debug_GL      vs debug_GL(2nd) :      0 px  (deterministic)
clean_GL      vs debug_GL      :  64090 px, max channel delta 2
clean_Metal   vs debug_GL      :  64138 px, max channel delta 8
```

1. **Debug Metal == clean Metal (0 px)** — the embedded `VTK_METAL_VOLUME_LOG`
   instrumentation compiles out to identical arithmetic. Confirmed again.
2. **Debug-injected GLSL is NOT clean GLSL.** With the correct backend flag,
   `VTK_GL_RAY_DUMP=1` changes the compiled GLSL such that **64,090 px** (of
   262,144) shift vs the clean GL compile, all ±1 LSB-ish (max delta 2). The
   injected debug blocks (extra uniforms `in_debugSample`/`in_debugTexel`/
   `in_debugPixel`, extra branches `g_dbgIter++`, `if (in_debugSample>=0 ...)`,
   `if (in_debugPixel == gl_FragCoord.xy ...)`) perturb register allocation /
   FMA contraction / constant folding, so the per-sample GL dump rows and the
   debug-GL image are **not** a faithful sample of clean-GL arithmetic. This
   re-affirms update 44's warning, now with the correct invocation and the
   current (march-fixed) binary.
3. **update 44's "debug GL ≈ Metal" (694 px) does not replicate** on the current
   binary: debug GL is now 64,138 px from Metal. That earlier coincidence was on
   the old binary where clean GL was itself 63,691 px from Metal; the debug
   compile simply lands at another point in the same ~1-ulp rounding cloud. It is
   not a stable property of the backend pair.
4. **The genuine residual to attack is clean GL vs clean Metal: 183 px, max
   channel delta 8**, deterministic on both sides. This is the same 183-px field
   that updates 59-71 progressively attributed to the attribute-interpolator
   mechanism (update 62-64), after updates 67-71 made the march geometry
   bit-identical (this residual is NOT ray formation: update 71 verified
   nearP/farP/dirObj/evalStep bit-identical at every pixel).

## 2. Residual pixels (clean GL minus Metal), worst first

```
(397,110) GL-Metal=(0,-7,-8)  Metal=(239,186,151) GL=(239,179,143)   <- largest, color-opacity path
(360,229) GL-Metal=(0, 3, 4)  Metal=(225,174,141) GL=(225,177,145)
(405,171) GL-Metal=(0,-2,-3)  Metal=(230,168,133) GL=(230,166,130)
(349,255) GL-Metal=(0,-2,-3)  Metal=(226,182,151) GL=(226,180,148)
(482, 33) GL-Metal=(0,-1,-2)  Metal=(242,163,123) GL=(242,162,121)
...
(4 px have |d|>=2; the other 179 px are single-channel ±1.)
```

Delta histogram: `(0,+1,0)` x27, `(0,0,+1)` x27, `(0,+1,+1)` x25,
`(0,0,-1)` x23, `(0,-1,-1)` x16, `(-1,0,0)` x16, `(0,-1,0)` x15,
`(+1,0,0)` x14 — dominated by single-channel ±1 rounding; red is almost always
equal (equal for ~153/183 px).

## 3. Full commands used (run from the repo root)

Env for all runs (dedupe as needed):

```sh
TMP=build_macos_metal/Testing/Temporary
BIN=build_macos_metal/bin/vtkRenderingVolumeCxxTests
EXT=build_macos_metal/ExternalData/Testing
DUMMY() { python3 -c "from PIL import Image; Image.new('RGB',(512,512),(0,0,0)).save('$1')"; }
```

Clean Metal (`mtA.png` = md5 `5e723c7d…`):

```sh
DUMMY "$TMP/mtA.png"
$BIN TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
  --vtk-factory-prefer RenderingBackend=Metal -D "$EXT" -T "$TMP" -V "$TMP/mtA.png"
```

Clean GL (`f_glA.png` = md5 `dc4bab2e…`; **flag is `OpenGL`, not `OpenGL2`**):

```sh
DUMMY "$TMP/f_glA.png"
$BIN TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
  --vtk-factory-prefer RenderingBackend=OpenGL -D "$EXT" -T "$TMP" -V "$TMP/f_glA.png"
```

Debug Metal (`mtD.png` = md5 `5e723c7d…`, identical to clean Metal):

```sh
DUMMY "$TMP/mtD.png"
MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=33554432 MTL_LOG_TO_STDERR=1 \
  $BIN TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
  --vtk-factory-prefer RenderingBackend=Metal -D "$EXT" -T "$TMP" -V "$TMP/mtD.png"
```

Debug GL (`f_glDA.png` = md5 `6487b74d…`; takes ~23 s because of the 29-px x
46-channel CPU re-render loop):

```sh
DUMMY "$TMP/f_glDA.png"
VTK_GL_RAY_DUMP=1 \
  $BIN TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
  --vtk-factory-prefer RenderingBackend=OpenGL -D "$EXT" -T "$TMP" -V "$TMP/f_glDA.png"
```

Diff helper (Pillow, 512x512 RGB, reports px count + max channel delta):

```sh
python3 - <<'EOF'
from PIL import Image
A=Image.open('A.png').convert('RGB'); B=Image.open('B.png').convert('RGB')
pa,pb=A.load(),B.load(); diff=0; maxd=0
for y in range(A.height):
  for x in range(A.width):
    ca,cb=pa[x,y],pb[x,y]
    if ca!=cb: diff+=1; maxd=max(maxd,max(abs(ca[i]-cb[i]) for i in range(3)))
print(diff, 'px differ, max channel delta', maxd)
EOF
```

Sanity check that the backend is actually the one requested: a Metal render
prints `VTK_METAL_VOLUME_LOG DEBUG METAL_CAM …` and `MTL_INVPVM=…`; a GL render
prints `VTK_METAL_VOLUME_LOG DEBUG GL_CAM …` / `GL_NEARPLANE` (the log macro is
shared, so look at the prefix `METAL_*` vs `GL_*`, or check
`(397,110) == (239,186,151)` for Metal vs `(239,179,143)` for GL).

## 4. Invocation rules (learned the hard way)

- **Backend flag must be `RenderingBackend=OpenGL` for GL and
  `RenderingBackend=Metal` for Metal.** `RenderingBackend=OpenGL2` matches
  nothing (registered values are `OpenGL`/`Metal`) and silently falls back to the
  default backend (Metal on this build), producing Metal output while looking
  like a "GL" capture. Always confirm the actual backend from the log prefix.
- Capture with `-V $TMP/<unique>.png`; the image is written to
  `$TMP/<basename>.png` only when the test fails (vtkTesting.cxx:904). A passing
  run leaves the dummy in place — always pre-create the dummy black PNG so a
  pass-vs-fail ambiguity shows up as a black file.
- Both backends are deterministic across repeated captures (0 px self-diff), so
  any nonzero delta is arithmetic, not jitter.
- Always run from the repo root so `-D "$EXT"` resolves the ExternalData
  baseline (the run fails with `ImageNotFound` otherwise, which is fine for a
  capture but pollutes the log).

## 5. Artifacts

- `/tmp/bc/u72/clean_metal.png`, `/tmp/bc/u72/clean_gl.png` (corrected; GL re-run
  with `RenderingBackend=OpenGL`), plus `debug_gl.png` (corrected) and
  `debug_metal.png`.
