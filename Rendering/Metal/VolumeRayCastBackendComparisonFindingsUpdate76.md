# The B-residual pixel lists are expressed in Metal PNG coordinates and were being sampled by GL at the y-flipped physical pixel; with the correct GL-window ↔ Metal-PNG pairing the interpolated ray anchor matches to 0-4 ulp (33/114 exact) and dirObj is 114/114 bit-identical, and per-vertex texcoord x/y are 94/94 bit-identical — so the update-75 anchor +1 ulp is ruled out as per-vertex rounding and must enter in the rasterizer interpolator itself (update 76)

**Date:** 2026-08-11
**Status:** **milestone (update-75 §4 candidate 1 ruled out; the anchor ulp enters in the rasterizer interpolator, candidate 2).** Update 75 proved the last worst-pixel flip (i=132 texel 355↔356) is *entirely* a 1-2 ulp offset in Metal's interpolated per-vertex texcoord (`p.localPos` vs GL `ip_textureCoords`), with `evalStep` byte-identical. Update 75 §4 left two candidate origins for that anchor offset: (1) per-vertex rounding in the vertex shader's cell-to-point formula, or (2) rasterizer interpolation rounding. This update closes candidate 1:

- The 19 "B residual" pixels (update 69) are listed in **Metal PNG coordinates** but the GL dump (`VTK_GL_RAY_DUMP` with `VTK_GL_RESID_DUMP=1`) was calling `glReadPixels(gx, gy)` with `gy = py` directly, i.e. sampling the **y-flipped physical pixel**. The correct GL-window ↔ Metal-PNG pairing is `GL(px, py) ↔ Metal(px, 511-py)`. All 114 residual rows re-paired this way give **dirObj 114/114 bit-identical** and interpolated anchor texcoord **0-4 ulp (33/114 exact)**, i.e. sub-ulp interpolation agreement across all residual pixels — the earlier apparent "hundreds-of-ulp tex mismatch" in the preceding session was purely the flipped-pixel artifact.
- Per-vertex texcoords **x/y are bit-identical for 94/94** vertices (GL `VTK_GL_VERTEX_DUMP` vs Metal `vertex_volume_main`, joined on clip-space x/y/w; z must be excluded from the join key because the depth-convention [0,1] vs [-1,1] splits clip.z, and the GL near-cap z-texcoord readback is an encoding artifact — `tex.z == tex.y` and `pos.z == vid` for the near-cap rows). So the vertex shader's `cellToPointScale * uvx + cellToPointOffset` (MetalShaders.metal:3077) already reproduces GL's four-term mat4×vec4 contraction bit-for-bit on x/y.
- Exact float64 perspective-correct interpolation of the three (bit-identical) per-vertex y-texcoords of covering triangle primId=122, using the (bit-identical) clip w's and the pixel-center barycentrics, reproduces **GL's interpolated y exactly** (`0x3f01aa39`) while Metal logs `0x3f01aa3a` (+1 ulp) from the same inputs. Same per-vertex values, same clip, same pixel → the +1 ulp is created by Metal's fragment interpolator (sample-position or interpolation-precision), i.e. update-75 §4 candidate 2.

**Follows:** [Update 75](VolumeRayCastBackendComparisonFindingsUpdate75.md) §4/§6 (anchor-ulp origin, per-vertex comparison experiment 1), §5 (state-A/B anchors).
**Test:** `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter`
(6 frames), worst pair Metal (397,110) ↔ GL (397,401), `max|d| = 8`.
**Logs:** `/tmp/bc/gl_resid.log` (GL `VTK_GL_RAY_DUMP=1 VTK_GL_RESID_DUMP=1`),
`/tmp/bc/metal_step2.log` (Metal STEP, resid+grid+dense gate),
`/tmp/bc/vtx_gl.log` (GL `VTK_GL_VERTEX_DUMP=1`), `/tmp/bc/u74c_metal.log`
(Metal `vertex_volume_main` + STEP at the worst pair).
**Scripts:** `/tmp/bc/join_verts2.py` (per-vid tex join), session ad-hoc Python
(resid re-pairing, interpolation reconstruction).

---

## 1. The B-residual pixel list is in Metal PNG coordinates; GL was sampling the flipped pixel

The resid pixel list in vtkOpenGLGPUVolumeRayCastMapper.cxx:4648 is documented
as "(Metal PNG coords (px, py), GL window coords (px, 511 - py))" but the dump
loop reads `gy = static_cast<GLint>(py)` (line 4666) — **not** `511 - py`.
`glReadPixels` uses GL window coordinates (origin bottom-left), so for the same
*pixel label* the two backends were reporting different physical pixels:

```
GL_RAY  px=(140, 505)   glReadPixels y=505  (window, bottom-left origin)
Metal   px=(140, 505)   screenPos y=505     (PNG, top-left origin)
                       ⇒ 505 (Metal PNG) = 511-505 = 6 (GL window)
```

Concretely, Metal's STEP at `(140, 6)` reproduces GL's GL_RAY at `(140, 505)`
to every printed digit on `dirObj`/`evalStep`/`clip.x`/`clip.w`:

```
GL_RAY  (140,505): step=(-7.86901146e-05, 1.09005414e-04, 1.94322516e-03)
                   rd  =(-0.0588574633, 0.0815322474, 0.994931281)
Metal   (140,  6): evalStep=(-7.86901146e-05, 1.09005414e-04, 1.94322516e-03)
                   dirObj  =(-5.88574633e-02, 8.15322474e-02, 9.94931281e-01)
```

while Metal at `(140, 505)` is a *different* physical pixel (clip.x differs by
~22 ulp). The preceding session compared same-labeled pixels and concluded
"hundreds to thousands of ulp" in the interpolated anchor; that comparison was
physically meaningless.

## 2. Re-paired residual comparison: dirObj 114/114, anchor 0-4 ulp

Re-pairing each of the 114 GL_RAY rows (19 pixels × 6 frames) with its Metal
twin `(px, 511-py)`:

- **`dirObj` (GL `rd=`, Metal `dirObj=`): 114/114 bit-identical** (0 ulp on all
  three axes). The analytic ray is byte-identical GL↔Metal at every residual
  pixel — confirms update 74's fixes hold off the single worst pixel too.
- **Interpolated anchor (GL `tex=`, Metal `localPos=`): 0-4 ulp, 33/114 exact.**
  Full histogram (per-axis ulp deltas, 114 rows):

```
(0,0,0) 33    (0,0,1)  9    (0,1,1)  9    (0,1,3)  3
(1,1,1)  3    (1,1,2) 12    (1,1,3)  3    (1,2,2)  6
(1,2,3)  3    (1,2,4)  6    (2,2,3)  3    (2,2,4)  3
(2,3,3)  3    (2,3,4)  9    (3,3,4)  6    (3,4,4)  3
```

  No row exceeds 4 ulp per axis. The anchor offset is therefore a small,
  everywhere-present interpolation difference — not a special-case of the worst
  pixel — and it is on the order of the "interpolation floor" of update-59 doubt
  (c).

## 3. Per-vertex texcoords x/y are bit-identical 94/94 (update-75 experiment 1)

Joining `GL_VERT` (vtx_gl.log) with Metal `vertex_volume_main` (u74c_metal.log)
by clip-space **x/y/w** (z excluded: Metal's depth convention maps clip.z to
[0,1] while GL keeps [-1,1], so clip.z can never be an equality key):

- 94/94 common vertices; **x and y per-vertex texcoords are bit-identical on
  all 94** (`du = [0,0,·]` on every row).
- The z per-vertex texcoord is NOT comparable from the GL dump: for the
  near-cap rows GL reports `tex.z == tex.y` and `pos.z == vid` (encoding
  artifact of the channel readback), e.g. `vid=86 pos=(102.977745, 93.0179443,
  86)` where 86 == vid and `tex=(0x3f02c28f, 0x3eec4650, 0x3eec4650)` has
  `tex.z == tex.y`. Metal's modelPos.z for the same clip is the genuine
  near-plane intersection (`z=60.4921799`). GL's near-cap proxy vertices are
  box-grid positions with a z-texcoord that the dump mis-encodes; only the
  interpolated fragment value (GL_RAY `tex.z`, verified 0-ulp at the worst
  pair, update 75 §5) is meaningful, and it matches Metal.

Combined with §2 (interpolated anchor ≤4 ulp), the conclusion is: **the vertex
shader cell-to-point texcoord is already bit-identical (x/y)** — update-75 §4
candidate 1 is ruled out for the x/y axes that drive the z-flip mechanism (the
i=132 decision is made by the **z** texcoord, but the z per-vertex data feeds
the same interpolator and the interpolated z matches to 0-2 ulp at the residual
pixels).

## 4. Reconstruction: identical inputs + identical clip ⇒ GL's interpolator is "exact", Metal rounds +1 ulp

At the worst pixel the covering triangle is primId=122 (both backends). Its
three per-vertex y-texcoords and clip w's are bit-identical (§3):

```
vertex   y-texcoord         clip.w
 86      0x3eec4650 (0.461473942)   0.38470459
 40      0x3f7fc002 (0.999023557)   0.384706497
 93      0x3f51e10f (0.819840372)   0.384702682
```

The pixel center NDC is the same on both sides (`x=0.552734375`, `y=0.568359375`,
GL window (397,401) == Metal PNG (397,110)). Perspective-correct interpolation
`Σ(w_i · attr_i / clip.w_i) / Σ(w_i / clip.w_i)` computed in float64 with the
screen-space barycentrics derived from the (bit-identical) clip positions:

```
float64 reconstruction   →  0x3f01aa39  (0.50650364…)
GL   logged ip_textureCoords.y  = 0x3f01aa39   ← EXACT match
Metal logged localPos.y         = 0x3f01aa3a   ← +1 ulp
```

Same per-vertex values, same clip, same pixel center; a correct interpolation
reproduces GL bit-for-bit, Metal is +1 ulp. The offset is therefore created in
Metal's fragment rasterizer interpolation — either a sub-pixel difference in
the effective sample position (GL vs Metal viewport transform) or a different
interpolation arithmetic/precision. This is update-75 §4 candidate 2.

## 5. Remaining open question and next experiments

The anchor +1 ulp is real, systematic (Metal always at-or-above GL: state A
y+1, state B x+1/y+1/z+2, and the resid histogram has no negative-delta rows),
and now localized to the rasterizer interpolator. The two sub-hypotheses:

1. **Sub-pixel sample position.** Metal and GL rasterizers may evaluate at
   slightly different fragment centers (viewport transform rounding), shifting
   the barycentrics by ~1e-7 and the interpolated value by 1 ulp. This would be
   *removable* by making Metal evaluate the anchor at the exact same sample
   point GL uses (or by computing the anchor analytically in the fragment
   shader from the pixel center + per-vertex clip/texcoord, bypassing the
   interpolator).
2. **Interpolation precision.** The GPU interpolator's internal arithmetic
   (fixed-point plane evaluation, fma contraction) differs between the two API
   layers on the same M2. If this is the cause, the offset is a hard
   interpolation floor and the only bit-identical route is to replace the
   interpolated anchor with an analytically reconstructed value in *both*
   backends.

Next experiments (priority order):

1. **Probe the sample position.** Reconstruct Metal's barycentrics by inverting
   the interpolation of the (known) per-vertex values against the logged
   `localPos` and compare with the exact pixel-center barycentrics; a
   systematic sub-ulp shift would confirm sub-hypothesis 1.
2. **Analytic-anchor experiment.** Temporarily compute the anchor in Metal as
   `Σ(w_i·tex_i/clip.w_i)/Σ(w_i/clip.w_i)` from the pixel-center barycentrics
   and the per-vertex clip/texcoord (uniforms), i.e. bypass the interpolator,
   and re-run the reference test. If the residual collapses, sub-hypothesis 1
   is confirmed and made removable.
3. If neither, quantify the interpolation floor across all 8237 gated pixels
   (histogram of per-axis ulp deltas over the full STEP gate, not just the 19
   residual pixels) and document it as the irreducible bound with a
   frame-matched comparison.

## 6. Files / commands

- Sources: `Rendering/VolumeOpenGL2/vtkOpenGLGPUVolumeRayCastMapper.cxx`
  (resid list 4648, GL_RAY dump 4611-4822, GL_VERT dump 5134-5198),
  `Rendering/VolumeOpenGL2/vtkVolumeShaderComposer.h` (vertex texcoord 128-136),
  `Rendering/Metal/Shaders/MetalShaders.metal` (vertex texcoord 3060-3086,
  anchor `anchorTex = in.texcoord` 5231-5246, STEP log 4261).
- Logs: `/tmp/bc/gl_resid.log`, `/tmp/bc/metal_step2.log`,
  `/tmp/bc/vtx_gl.log`, `/tmp/bc/u74c_metal.log`.
- Scripts: `/tmp/bc/join_verts2.py`; ad-hoc: resid re-pairing + histogram,
  per-vertex join, interpolation reconstruction.
- Images: `build_macos_metal/Testing/Temporary/{glA_u74,mtA_u74c}.png`
  (178 px diff, max Δ 8, worst `(397,110)` (0,-7,-8)).
