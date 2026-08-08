# Metal vs OpenGL volume ray cast: §5.1 gradient-opacity half-precision fix is inert; offscreen-block §5.2 refuted; camera-inside ray geometry is the remaining NoTransform candidate (update 10)

Follow-up to all prior documents; read as an addendum to
[VolumeRayCastBackendComparisonFindingsUpdate9.md](VolumeRayCastBackendComparisonFindingsUpdate9.md),
whose §5 listed three remaining residual sources. This update closes two of
them with direct measurements — the §5.1 gradient-opacity half-precision chain
and the §5.2 offscreen `RGBA16Float` inter-block accumulation — and shows that
neither is responsible for the residual it was blamed for. The surviving
candidate for the `…NoTransform` family (646 pixels >4 LSB) is §5.3's
camera-inside ray seed: the Metal fullscreen camera-inside path derives
`rayDir` from the pixel position and finds the entry point analytically, while
OpenGL starts each ray at the interpolated near-plane-clipped proxy surface.

## 1. §5.1 evaluated: promoting the gradient-opacity chain to float changes nothing

Update9 §5.1 blamed two remaining `half` sites in the gradient-magnitude path:

1. `sampleGradientOpacity()` returning `half` (the `R32F` gradient-opacity
   table read through an explicit `half(...)` cast), and
2. the `marchVolumeUnified` local `half gradNormFactor` narrowing the
   normalized-magnitude divisor before it feeds the (already-float)
   `normalizedGradient`/`computeGradientFast` kernels.

Both were promoted to `float`:

- `sampleGradientOpacity` now returns `float` and drops the `half(...)` casts
  (`Rendering/Metal/Shaders/MetalShaders.metal`, formerly line 3076).
- `gradNormFactor = max(1e-8f, volumeUniforms.gradientOpacityRange.y)` is now
  `float` (formerly `half(...)` at line 3748).

The call sites are unaffected: `compColor[c].a *= sampleGradientOpacity(...)`
is a scalar `half *= float` compound assignment (MSL allows scalar
conversion), and the main path already did `sampleOpacity *= float(...)` where
`sampleOpacity` is `float`.

### 1.1 Rebuilt and re-measured: zero change

Back-to-back capture with the same harness as Update9 (max-channel |Δ| > 4 LSB
against the OpenGL reference):

| variant | Update9 "after" | this build |
|---|---|---|
| TestGPURayCastCameraInsideTransformation | 40 | 40 |
| …ConstGradOp | 39 | 38 |
| …NoGradOp | 68 | 68 |
| …NoShade | 27 | 27 |
| …NoShadeNoGradOpNoTransform | 646 | 646 |

### 1.2 The whole chain was already float except the two sites

Auditing the rest of the path confirms nothing else is `half`:

- `normalizedGradient` (line 3197) and `computeGradientFast` (line 3241) are
  all-`float`; the six neighbor samples read through `sampleVolumeScalar`
  (float), the magnitude is `length(correctedGrad)` in float, and
  `w = saturate(mag / gradNormFactor)` is float.
- The gradient-opacity LUT build matches
  `vtkOpenGLVolumeGradientOpacityTable::InternalUpdate` exactly:
  `gradOpacityFunc->GetTable(0.0, scalarRange * 0.25, tfWidth, float*)`
  uploaded as a single-channel `R32F` texture
  (`vtkMetalGPUVolumeRayCastMapper.mm`, `UpdateGradientOpacityTexture`,
  line 3816). GL applies no pre-integration `pow` to this table, and Metal
  does not either. The shader lookup coordinate is the normalized magnitude
  `grad.w` in both backends.

Conclusion: §5.1's two `half` sites are not the bottleneck. The GradOp
residual (27–68 pixels) is not limited by gradient-opacity *precision*.

## 2. §5.2 refuted: the offscreen `RGBA16Float` inter-block accumulation is not the NoTransform residual

Update9 §5.2 blamed the 646-pixel `…NoShadeNoGradOpNoTransform` residual on the
volume rendering in 8×8 blocks into an `RGBA16Float` offscreen, "each block
boundary rounds through half". If that were the mechanism, the >4-LSB pixels
would concentrate on the 8-pixel block seams. They do not.

Distribution of the 646 diff pixels over the pixel position modulo the block
size (rows = `x mod 8`, columns = `y mod 8`, from
`…NoShadeNoGradOpNoTransform` Metal vs OpenGL reference):

```
       y0   y1   y2   y3   y4   y5   y6   y7
xmod0   5   15    7    9    7    5    8   13
xmod1   6   14    8   12   11    8    4   12
xmod2  11   22   11   10   13    9   11   16
xmod3   8   24   12    9   10   10    5    6
xmod4   4   21    8    6    6    2    4    9
xmod5   6   18   18   13   17   11   10   11
xmod6   8   16    7    7    5    5    2    8
xmod7   8   23   10   12   10    9    7   14
```

- 120 of 646 (19%) sit on an exact block-boundary line (`x mod 8 == 0` or
  `y mod 8 == 0`); a uniform spread would give ~121. No enrichment.
- The interior 64-pixel-grid clustering observed earlier (upper-left region,
  y 64–320, x 192–384) is a *feature-location* effect (the bone structure
  under the camera), not a seam effect — the diffs are spread across all
  (x, y) mod 8 residues within that region.

Conclusion: §5.2's attribution is refuted. The NoTransform residual is
per-pixel and uniform, consistent with a per-sample *ray* difference rather
than a compositing/rounding artifact.

## 3. March termination at (422,92) confirmed parity-correct

A fresh full-frame log (`DEBUG SAMPLE` now carries `accA`/`maxSteps`/`termMax`)
for the same pixel previously diagnosed as "stopping at sample ~168":

- `maxSteps = 299`, `termMax ≈ 1e30` (no depth pass → depth termination off).
- The march runs to `i = 171` with `accumulatedOpacity = 0.995886`, i.e. just
  below the OpenGL threshold `1 - 1/255 = 0.996078`; sample 172 crosses it and
  hits the `accumulatedOpacity >= 1.0 - 1.0/255.0` break (line 4537). This is
  parity-correct early termination, not a premature stop.
- The earlier "stops at 168" reading came from the older log format whose last
  printed sample was `i=167`; the composite in fact continued.

So the NoTransform residual at (422,92) is not a sample-count/termination
difference. Metal's final accumulation there is `accC ≈ (0.9328, 0.7353,
0.6025)` (G ≈ 187/255 rendered) versus GL ≈ 176/255 — the delta accrues across
the bone plateau, i.e. the sampled *positions* differ, not the table values.

## 4. Per-sample transfer-function parity in the bone plateau confirmed

- Metal logs `sampleDist=0.270059 unitDist=1 preIntFactor=0.270059` for the
  opacity table (`DEBUG MTL_OPTABLE`, range (0,4370), width 1024).
- The bone-plateau samples log `op ≈ 0.4009`, matching
  `1 - pow(1 - 0.85, 0.270059) = 0.4008` — the same pre-integration correction
  `vtkOpenGLVolumeOpacityTable::InternalUpdate` applies.
- The plateau color is flat `rgb = (1, 1, 0.9)` (last CTF node), so the
  half-texel-class TF-sampling offset observed in the ramp region does not
  affect the plateau that carries the bulk of the accumulated opacity. The
  nearest-vs-linear TF filter is also parity: both backends filter the
  transfer-function textures with the volume property's interpolation type
  (default nearest; `vtkVolumeInputHelper.cxx:129`).

## 5. Current state and the surviving candidate

With §5.1 and §5.2 closed out:

1. **Pure composite** (`NoShadeNoGradOp`, `NoShadeConstGradOp`) remains
   pixel-perfect (max Δ = 1).
2. **Shading precision** (`NoGradOp` 68 px, standard 40 px) is the residual
   for the shaded variants; the lighting kernels and the
   `half3(sharedGrad.xyz)` normal still run in `half` (Update9 §5 item 1 for
   shading, untouched by this update).
3. **Camera-inside ray geometry** (`…NoShadeNoGradOpNoTransform` 646 px) is
   the largest remaining residual. Key asymmetry: the *with-transform*
   camera-inside variant is pixel-perfect (0 px) while the *no-transform*
   variant is 646 px. Both use the same compositing path, so the difference is
   in how the entry point / ray direction is established:
   - Metal's `UseFullscreenCameraInside` path skips the clipped-proxy vertex
     buffer entirely, draws a fullscreen triangle, and computes
     `rayDir = reconstructRayDir(in.position.xy, …)` from the pixel position;
     `setupVolumeRay` then intersects the box analytically and applies the
     near-plane clamp (`CameraInsideNearPlaneOrigin/Normal`, which the CPU
     computes identically to GL).
   - OpenGL renders the near-plane-clipped and densified proxy and starts the
     ray at the *interpolated* `ip_textureCoords` of that surface, so the ray
     origin is exactly on the clip plane rather than reconstructed from the
     pixel and re-intersected.
   The two agree in the limit (the clip surface is the near plane, and a
   perspective ray through a pixel meets it at the same point) but differ at
   the float-precision level in where the first sample lands — the exact
   mechanism Update8 §5.3 / Update9 §5 item 3 proposed, never yet measured for
   the camera-inside case.

Next step: capture the Metal fullscreen path's `reconstructRayDir` + analytic
entry and compare against GL's interpolated `ip_textureCoords`/`g_rayOrigin`
at the worst NoTransform pixels (the GL-side `VTK_GL_RAY_DUMP` scaffolding
from Update8 already emits these; the Metal side logs the entry/t in
`DEBUG MARCH`). A sub-texel comb shift at the bone plateau would reproduce the
uniform interior deltas.

## 6. Files changed

- `Rendering/Metal/Shaders/MetalShaders.metal` — `sampleGradientOpacity`
  returns `float` (drops the `half(...)` casts); `marchVolumeUnified`'s
  `gradNormFactor` is `float` instead of `half`.
