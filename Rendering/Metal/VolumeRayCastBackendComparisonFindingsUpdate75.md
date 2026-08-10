# The i=132 texel flip is a 1-2 ulp difference in the interpolated anchor: Metal `localPos` is systematically ABOVE GL `ip_textureCoords` (y +1 ulp in state A; x,y,z +1/+1/+2 in state B), and after 132 byte-identical `evalStep` additions that z-ulp lands Metal exactly ON the 356/512 = 0.6953125 texel boundary while GL sits 1 ulp below (update 75)

**Date:** 2026-08-10
**Status:** **milestone (root cause of the last worst-pixel flip identified, corrected).**
With update 74's two codegen fixes, `nearP`/`farP`/`d`/`dirObj`/`evalStep` are all
byte-identical to GL, yet the worst pixel `(397,110)`↔`(397,401)` still shows
Metal `(239,186,151)` vs GL `(239,179,143)`. Update 74 §5 doubt 2 (tex-vs-eval
lattice) is **resolved** from source: the scalar fetch at MetalShaders.metal:4401
uses `rectEvalPoint == evalPoint` (the `g_dataPos` lattice), and `texLocalPos`
in the debug dump is only the raw [0,1] position used for bounds clamping — no
lattice split. The real cause, proven by reconstructing both float32 chains at
the worst pixel: **Metal's interpolated per-vertex texcoord `p.localPos` is
systematically higher than GL's interpolated `ip_textureCoords`** — y +1 ulp in
state A, and x +1 / y +1 / z +2 ulp in state B. With `evalStep` byte-identical
(update 74 §3), that anchor offset is the *entire* mechanism: in state B the
z-ulp carries Metal to `0x3f320000` = 0.6953125 exactly (the 356/512 texel
boundary) while GL stays at `0x3f31ffff` = 0.69531244, flipping the nearest
texel pick 356 vs 355 at i=132.

**Follows:** [Update 74](VolumeRayCastBackendComparisonFindingsUpdate74.md) §5
doubt 2, and §3/§4 (evalStep byte-identical, i=132 flip is the sole residual).
**Test:** `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter`
(6 frames), worst pair Metal (397,110) ↔ GL (397,401), `max|d| = 8`.
**Logs:** `/tmp/bc/u74c_metal.log` (Metal STEP/SAMPLE), `/tmp/bc/u74_gls.log`
(GL `VTK_GL_RAY_DUMP=1 VTK_GL_SAMPLE_DUMP=1 VTK_GL_SAMPLE_DUMP_PX=397,401`).
**Scripts:** `/tmp/bc/u74_reconstruct{2,3}.py`, `/tmp/bc/u74_anchor.py`,
`/tmp/bc/u75_exact.py`.

---

## 1. The two camera states (W2IF perturbation, updates 19-20)

The 6-frame harness renders frames 1-3 with the original camera (view angle
30°) and frames 4-6 with the W2IF copy (30.0000008°). Both backends' anchor
coordinates shift at the frame 3→4 boundary. The anchor is Metal's interpolated
`p.localPos` (= interpolated vertex `texcoord`, MetalShaders.metal:4195/5245) vs
GL's interpolated `ip_textureCoords` (`tex=` in the GL_RAY dump):

```
              GL ip_textureCoords (tcoord)      Metal localPos (STEP)         delta
state A (f1-3): (3f0184b9, 3f01aa39, 3ee5d6d4)   (3f0184b9, 3f01aa3a, 3ee5d6d4)   y +1 ulp
state B (f4-6): (3f0184b7, 3f01aa36, 3ee5d6d0)   (3f0184b8, 3f01aa37, 3ee5d6d2)   x +1, y +1, z +2 ulp
```

- **State A: Metal y is 1 ulp HIGHER** (`3f01aa3a` vs GL `3f01aa39`); x and z
  are bit-identical.
- **State B: Metal is higher in ALL three axes** — x +1 ulp (`3f0184b8` vs
  `3f0184b7`), y +1 ulp (`3f01aa37` vs `3f01aa36`), z +2 ulp (`3ee5d6d2` vs
  `3ee5d6d0`). Both backends also moved down from state A (the W2IF view-angle
  response), but Metal's shift is slightly *smaller* per axis (x 1 vs 2 ulp, y
  3 vs 3, z 2 vs 4), so the two backends' anchors do NOT stay aligned.

So the anchor discrepancy is **not** a state-B coincidence: Metal's
interpolated texcoord is consistently ~1-2 ulp above GL's in the components that
matter. The first version of this update incorrectly reported state-B anchors
as bit-identical (the STEP log prints `localPos` at `%0.9e`, ~9 significant
digits, which hides the last ulp). This revision uses the GL_RAY `tex=` prints
(`setprecision(9)`, an exact float32 round-trip) and reconstructs the Metal
bits from the incremental chain (verified: `localPos + evalStep*1.0` reproduces
the logged `evalPoint@0` for state A and B).

## 2. The 1-ulp propagates to the texel boundary at i=132

Reconstructing both backends' sample positions as
`anchor + evalStep * i` with **incremental float32 adds** (the shader does
`evalPoint += evalStep` per iteration) at the worst pixel, with the byte-
identical step `evalStepBits=b9dd56a9b7f560333af2d668`:

```
state A:  Metal eval@132 z = 0x3f320001 = 0.69531256  (texel 356)
          GL    pos@132 z = 0x3f320001 = 0.69531256  (texel 356)   <-- SAME, no flip
state B:  Metal eval@132 z = 0x3f320000 = 0.6953125  (texel 356)   <-- ON the boundary
          GL    pos@132 z = 0x3f31ffff = 0.69531244 (texel 355)   <-- 1 ulp below boundary
```

The texel boundary is `356/512 = 0.6953125`. In state A both backends land
identically on texel 356 — consistent with the logs (frames 1-3: both raw
≈0.017487, op=0.371981). In state B, Metal lands **exactly on the boundary**
(nearest rounds to texel 356) while GL is **1 ulp below** (texel 355). That is
precisely the observed flip: GL frames 4-6 `raw=0.0163272 op=0.127265` (texel
355), Metal all frames `raw=0.0174870 op=0.371981` (texel 356).

## 3. Why state A didn't flip and state B did

The decision texel is chosen by the **z** coordinate at i=132 (z ≈ 0.6953),
which sits within ~1 ulp of the 356/512 boundary.

- **State A:** Metal's z == GL's z bit-for-bit at the anchor (`3ee5d6d4` both)
  and `evalStep` is identical → every i is identical, no flip. The y-ulp
  (`3f01aa3a` vs `3f01aa39`) is real but never selects a texel.
- **State B:** Metal's z anchor is **+2 ulp** (`3ee5d6d2` vs `3ee5d6d0`); after
  132 byte-identical adds that offset lands Metal exactly on `0x3f320000`
  (0.6953125 = boundary) while GL sits 1 ulp below at `0x3f31ffff`. Nearest
  rounds Metal up to texel 356, GL down to texel 355.

There is **no** mystery ulp entering inside the loop: the two backends' step is
byte-identical and the incremental add is deterministic, so the i=132 split is
entirely the anchor offset. (The first version of this update built a
"bit-identical anchors + bit-identical step ⇒ the ulp must enter in the loop"
contradiction on the assumption that state-B anchors matched; the exact GL_RAY
bits prove they do not.)

## 4. Where the anchor ulp comes from (open)

The 1-2 ulp lives in the *interpolated per-vertex texcoord*, i.e. between
Metal's interpolated `in.texcoord` and GL's interpolated `ip_textureCoords`.
Two candidate origins, both consistent with the evidence so far:

1. **Per-vertex rounding in the vertex shader.** GL computes
   `ip_textureCoords = (in_cellToPoint[0] * vec4(uvx, 1.0)).xyz` — a full
   mat4×vec4 (four-term dot with fma contraction) — while Metal computes
   `out.texcoord = cellToPointScale * uvx + cellToPointOffset` — a single
   mul+add per component (MetalShaders.metal:3077). If the GLSL driver
   contracts the four-term dot differently than Metal's one-fma form, the
   per-vertex values can differ by 1 ulp per component. The vertex logs
   (`vertex_volume_main` on Metal, `VTK_GL_VERTEX_DUMP` on GL) must be
   compared bit-for-bit to confirm or rule out.
2. **Rasterizer interpolation rounding.** Update 59's doubt (c): the fragment
   interpolator itself may round the same per-vertex floats differently
   (Metal vs GL on the same M2). If per-vertex texcoords are bit-identical but
   the interpolated fragment anchor differs, this is the irreducible
   interpolation floor and no shader-side change can close it.

Note the proxy triangle is the same (`primId=122` both) but the provoking
vertex differs (GL `flatVid=93` vs Metal `flatVid=86`), i.e. the index/vertex
ordering of the proxy mesh is not byte-identical between backends — worth
checking whether the three per-vertex texcoords feeding the interpolation are
the same three floats in each backend.

## 5. Verification data (all reproduced from logs)

- **Anchors (state A)**: GL `tcoord` = `3f0184b9 3f01aa39 3ee5d6d4`; Metal
  `localPos` = `3f0184b9 3f01aa3a 3ee5d6d4`. y differs by exactly 1 ulp
  (5.96e-8). x/z bit-identical.
- **Anchors (state B)**: GL `tcoord` = `3f0184b7 3f01aa36 3ee5d6d0`; Metal
  `localPos` = `3f0184b8 3f01aa37 3ee5d6d2` — x +1, y +1, z +2 ulp.
- **evalPoint@0 = anchor + evalStep** (state B): GL `g_rayOrigin` =
  `3f01690c 3f01a84b 3ee6c9a6` (== logged `origin=`); Metal `localPos+evalStep`
  = `3f01690d 3f01a84c 3ee6c9a8` — +1 ulp per axis.
- **evalStep** (update 74 §3): Metal `evalStepBits=b9dd56a9b7f560333af2d668`
  == GL `step0=b9dd56a9...` at (397,110) — byte-identical.
- **i=132 (state B)**: Metal z `0x3f320000` (0.6953125, texel 356, raw
  0.0174870); GL z `0x3f31ffff` (0.69531244, texel 355, raw 0.0163272).
- **i=132 (state A)**: both z `0x3f320001` (0.69531256, texel 356, raw
  0.0174868/0.0174870) — no flip.
- **GL per-sample (6 frames)**: frames 1-3 `raw=0.0174868 op=0.371981` (texel
  356), frames 4-6 `raw=0.0163272 op=0.127265` (texel 355).
- **Metal per-sample (5 logged frames)**: all `raw=0.017487 op=0.371981`
  (texel 356), eval z flips 0.695312/0.695313 at the same frame boundary as GL.

## 6. Next experiments (in priority order)

1. **Compare per-vertex texcoords bit-for-bit.** Run GL with
   `VTK_GL_VERTEX_DUMP=1` (vtkOpenGLGPUVolumeRayCastMapper.cxx:5134) and diff
   `tex=` per `vid` against Metal's `vertex_volume_main` log (MetalShaders.metal:3082)
   for the `primId=122` triangle vertices. Identical per-vertex ⇒
   rasterizer-interpolation origin (update 59 doubt c); differing per-vertex ⇒
   fix the Metal `cellToPointScale * uvx + cellToPointOffset` form to reproduce
   GL's four-term dot contraction.
2. **If per-vertex matches:** confirm the rasterizer origin by computing the
   interpolated value from the three per-vertex texcoords and the pixel-center
   barycentrics on both sides (update 59's candidate bisect). If Metal's
   interpolator rounds differently, quantify the irreducible floor.
3. **If per-vertex differs:** reorder the Metal vertex formula and re-run; the
   reference test should collapse if the anchor offset was the sole residual.
4. Otherwise, document the interpolation floor with a frame-matched
   comparison (Metal frame N vs GL frame N, update 74 §5 doubt 1).

## 7. Files / commands

- Sources consulted: `Rendering/Metal/Shaders/MetalShaders.metal` (fetch at
  4401, `rectEvalPoint == evalPoint` for non-rectilinear; anchor at 4195/5245;
  vertex texcoord at 3077; loop add at 4925),
  `Rendering/VolumeOpenGL2/vtkOpenGLGPUVolumeRayCastMapper.cxx` (GL_RAY/SAMPLE
  dumps at 4611/4955, GL vertex dump at 5134),
  `vtkVolumeShaderComposer.h` (g_rayOrigin at 441-493, vertex texcoord at
  128-136).
- Logs: `/tmp/bc/u74c_metal.log`, `/tmp/bc/u74_gls.log`.
- Scripts: `/tmp/bc/u74_reconstruct2.py`, `/tmp/bc/u74_reconstruct3.py`,
  `/tmp/bc/u74_anchor.py`, `/tmp/bc/u75_exact.py`.
- Images: `build_macos_metal/Testing/Temporary/{glA_u74,mtA_u74c}.png`
  (178 px diff, max Δ 8, worst `(397,110)` (0,-7,-8)).
