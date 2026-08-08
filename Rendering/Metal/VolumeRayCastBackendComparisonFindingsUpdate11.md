# Metal vs OpenGL volume ray cast: sample-distance and step arithmetic are parity; the camera-inside NoTransform ray origin is the surviving candidate (update 11)

Follow-up to all prior documents; read as an addendum to
[VolumeRayCastBackendComparisonFindingsUpdate10.md](VolumeRayCastBackendComparisonFindingsUpdate10.md).
Update 10 left a single candidate for the `…NoShadeNoGradOpNoTransform` residual
(646 pixels > 4 LSB): the camera-inside ray *seed* — Metal's fullscreen path
reconstructs `rayDir` from the pixel and finds the entry analytically, while
OpenGL starts each ray at the interpolated near-plane-clipped proxy surface.
This update takes that candidate apart on the Metal side: the sample distance is
byte-for-byte parity, the march step arithmetic is parity *by construction*, and
the near-plane clamp and the box entry coincide on the worst pixel. The only
remaining unknown is OpenGL's actual `g_rayOrigin`, and the scaffolding to read
it for the exact worst pixel is now in place.

## 1. Sample distance is parity (0.270059 in both backends, reduction = 1)

Fresh cross-backend logs for the same `…NoShadeNoGradOpNoTransform` scene:

- OpenGL (`DEBUG GL_SAMPLING_RESULT`):
  `autoAdjust actual=0.270059 minWorldSpacing=0.270059 reduction=1`
- Metal (`DEBUG MTL_OPTABLE`):
  `range=(0,4370) width=1024 sampleDist=0.270059 unitDist=1 preIntFactor=0.270059`

Both backends therefore build the opacity table with the same
`preIntegrationFactor = sampleDistance / unitDistance = 0.270059` and step the
same physical distance per march sample. A scratch observation from earlier
sessions that the Metal *world* step (≈1.59 model units) disagreed with
`sampleDist` was a misreading: the model extent here is (201.6, 201.6, 138),
not the full-res (1635.2, 1635.2, 766.5) of the earlier ramp tests, and the
observed world step *is* `actualSampleDistance` once the correct extents are
used (verified in §3).

## 2. The near-plane clamp and the box entry coincide on the worst pixel

The NoTransform camera sits at model (102.122, 102.122, 61.562) = volume
(0.506559, 0.506559, 0.446101) with the camera near plane behind it along the
view (plane origin (−4.28865, −4.28865, 24.1236), normal (−0.172412, −0.172412,
0.969819); `DEBUG METAL_NEARPLANE`). For pixel (422, 92):

- the ray is `rayDir = (-0.239570, -0.002670, 0.970876)`, unit in [0,1] volume
  space (`DEBUG MARCH`);
- the analytic near-plane intersection distance is `tNear ≈ 0.002783`
  (computed from the exact uniforms), and the logged post-clamp
  `tStart = 0.002781` — the clamp wins over the box entry (`max(tBox, 0) = 0`
  since the camera is inside), and `tNear` sits essentially on the box entry
  surface. `useClip = 0` in the log refers to user clip planes, not the
  camera-inside near-plane clamp.

So Metal starts this ray at `camera + rayDir * tNear`, i.e. on the camera near
plane at the box boundary. OpenGL starts it at the interpolated
near-plane-clipped proxy vertex for the same plane (built with the identical
`0.001 * (far - near)` precision offset; `vtkOpenGLGPUVolumeRayCastMapper.cxx`
line 1214 vs `vtkMetalGPUVolumeRayCastMapper.mm` line 6646). The two agree in
the limit; the float-level difference is the open question.

## 3. March step arithmetic is parity by construction; the earlier "0.2% step" reading compared the wrong quantity

`marchVolumeUnified` maintains two per-sample increments
(`Rendering/Metal/Shaders/MetalShaders.metal`):

1. `texStep = (volumeToTexture * (rayDir * boundsSize)) * stepSize` (line 3757)
   — advances `texLocalPos`, the raw [0,1] texture-space trace used only for
   the debug `tex` field and the min/max cell-cull bookkeeping;
2. `evalStep = (adjustedLin * (rayDir * boundsSize)) * stepSize` (line 3776)
   — advances `evalPoint`, the cell-to-point-adjusted position where the
   volume scalar, transfer function and gradients are actually sampled.

`adjustedLin` folds the cell-to-point scale `(d-0.5)/d - 0.5/d` into
`volumeToTexture`, reproducing OpenGL's
`ip_inverseTextureDataAdjusted = in_cellToPoint * in_inverseTextureDatasetMatrix`
(comment at lines 3765–3771). A per-axis expansion for the axis-aligned
single-block case (rayDir already unit in volume space) gives:

```
Metal evalStep_i = ctpScale_i * rayDir_i * stepSize,   stepSize = sd / physPerNorm
GL   g_dirStep_i = ctpScale_i * (rayDir_model_i / extent_i) * sd
```

with `physPerNorm = |rayDir * boundsSize| = 142.42` and
`rayDir_model = rayDir * boundsSize / physPerNorm`; since
`rayDir_model_i / extent_i = rayDir_i / physPerNorm`, the two are the same
expression: **`Metal evalStep == GL g_dirStep`**. Numbers for pixel (422, 92):

| component | Metal `evalStep` (predicted) | GL `g_dirStep` (predicted) |
|---|---|---|
| x | −0.00045334 | −0.00045337 |
| y | −0.00000505 | −0.00000505 |
| z | +0.00183719 | +0.00183732 |

Agreement to ~1e-7 (float order), consistent with the observed per-sample eval
advance from the `DEBUG SAMPLE` stream:
`eval(1)-eval(0) = (-0.000453, -0.000005, +0.001838)`.

The "0.2% longer Metal step" flagged during this session compared `texStep`
(which is *not* ctp-scaled) against `g_dirStep`. That is the debug-trace
advance, not the sampling advance, so it is not a residual mechanism for the
pure-composite path (min/max culling is inactive in this test — the march runs
consecutive `i` with no skips). It is corrected here so the record is not
confused by it.

## 4. March termination parity re-affirmed (already recorded in update 10 §3)

For completeness at (422, 92): `maxSteps = 299`, the march reaches `i = 171`
with `accA = 0.995886` and sample 172 crosses
`accumulatedOpacity >= 1.0 - 1.0/255.0` — parity-correct early termination,
not a premature stop. Metal's final `accC ≈ (0.9328, 0.7353, 0.6025)`
(G ≈ 187/255) vs GL ≈ 176/255 accrues across the bone plateau
(`op ≈ 0.4009 = 1 - pow(1 - 0.85, 0.270059)`, flat `rgb = (1, 1, 0.9)`), i.e.
the *positions* differ, not the table values.

## 5. The surviving candidate, refined

With §1–§3, the step size, the step vector, the pre-integration factor, the
near-plane definition and the termination rule are all parity. What remains
unmeasured is the **seed**: Metal's analytic `camera + rayDir * tNear` versus
OpenGL's interpolated proxy vertex (the `ip_textureCoords` value that becomes
`g_rayOrigin`). The two agree to ~1e-5 on paper; the residual (646 pixels,
uniform interior deltas) requires them to differ by a sub-texel amount that
changes where the comb enters and exits the bone plateau.

Supporting asymmetry from the residual table (update 10 §5): the *with-transform*
camera-inside variants of the pure-composite family are pixel-perfect (0 px),
while the no-transform variant is 646 px. Both use the same Metal fullscreen
path and the same GL clipped-proxy path; the difference is geometric — with a
rotated volume, the near plane clearly cuts the box interior (`tNear > 0`
dominates the entry), while in the no-transform layout `tNear ≈ box entry`
to ~1e-5, so the analytic-vs-interpolated origin difference is maximally
amplified relative to the step.

### Next step (the lead)

Capture OpenGL's actual `g_rayOrigin` / `g_dirStep` for the exact worst pixel
and compare against Metal's `entry` / `evalStep`:

- the GL-side `VTK_GL_RAY_DUMP` scaffolding (update 8) already emits
  origin/step; the pixel list has been extended with `(422.5, 419.5)` — GL's
  `gl_FragCoord` space for Metal screenPos `(422, 92)` (`Metal (x, y) == GL
  (x, 511 - y)`), see `DumpDebugRays` in
  `vtkOpenGLGPUVolumeRayCastMapper.cxx` (this change is uncommitted at the time
  of writing);
- run `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransform`
  with `VTK_GL_RAY_DUMP=1` and the Metal logging build, then compare
  `GL origin + g_dirStep*i` against Metal `eval(i)` for `i = 0..172`.

If the combs agree to <0.05 texel, the camera-inside seed is exonerated and the
residual must move to the sampled scalar/TF value at identical positions; if
they are comb-shifted (≥ ~0.1 texel), the shift is the mechanism and the fix is
to align the Metal entry with GL's interpolated proxy origin.

## 6. Current residual state (unchanged since update 10; no production-code change in this update)

| variant | >4 LSB pixels |
|---|---|
| Pure composite, transform camera-inside | 0 (pixel-perfect) |
| Shaded (default / ConstGradOp / NoGradOp / NoShade) | 27–68 |
| **NoShadeNoGradOpNoTransform** | **646** |

## 7. Files changed

- `Rendering/VolumeOpenGL2/vtkOpenGLGPUVolumeRayCastMapper.cxx` —
  `DumpDebugRays` pixel list: added `(422.5, 419.5)` (Metal's `(422, 92)`) and
  bumped the loop bound 14 → 15 (uncommitted).
- This document.
