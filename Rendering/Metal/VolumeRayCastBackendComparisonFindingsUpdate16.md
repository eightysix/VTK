# Metal vs OpenGL volume ray cast: NEAREST no-jitter camera-inside residual root-caused to a single flipped TF sample (i=144) via linear sample-position drift (update 16)

**Date:** 2026-08-08
**Scope:** Worst pixel of the NEAREST no-jitter camera-inside residual, Metal `(372,131)` == GL `(372,380)`, `max|d|=22`, from the `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter` test (6 frames, deterministic per backend).
**Follows:** [Update 15](VolumeRayCastBackendComparisonFindingsUpdate15.md) (which attributed the residual to a "<=0.02 texel sample-position offset"). This update confirms that hypothesis, pins the exact mechanism to ONE diverging sample, and quantifies the underlying position drift.
**Persisted tool:** `BackendComparisonTools/compare_gl_metal_accum.py` (per-sample table + accumulation replay + position-trace fit). Captured logs used to develop this update: `/tmp/bc/gl372.log` (GL) and `/tmp/bc/true/metal_samples.log` (Metal).

---

## 1. Conclusion

At the worst residual pixel, both backends march 236 samples (i=0..235) with
**identical** per-sample raw scalars, TF opacities and TF colors at every single
sample **except i=144**. There the backends sample raw values on opposite sides
of the scalar-1150 opacity knot:

| i | GL raw | Metal raw | GL op | Metal op | GL rgb | Metal rgb |
|---|---|---|---|---|---|---|
| 143 | 0.0155337 | 0.0155340 | 0.031651 | 0.031651 | (1.000, 0.556, 0.367) | same |
| **144** | **0.0181582** | **0.0154270** | **0.400904** | **0.017987** | **(1.000, 1.000, 0.900)** | **(1.000, 0.527, 0.333)** |
| 145 | 0.0159609 | 0.0159610 | 0.084118 | 0.084118 | (1.000, 0.655, 0.486) | same |

raw threshold = 1150 / 65535 = 0.017548. GL (0.01816) is just above the knot
(sampled value 1190, full bone-plateau color rgb=(1,1,0.9) at op 0.401); Metal
(0.01543) is just below it (sampled value 1011, pre-knot ramp color at op 0.018).
One sample crossing the boundary flips the entire accumulated composite:

```
final composite:  GL (0.9341, 0.6293, 0.4749) -> PNG [238,160,121]
                  MT (0.9307, 0.5575, 0.3898) -> PNG [237,142,99]
```

The GL composite is brighter in G/B (dG +0.072, dB +0.085) because GL
composites the saturated bone color at 0.40 opacity instead of 0.018.

The i=144 flip is NOT a TF lookup difference and NOT a sampling-density
difference. It is a **linear sample-position drift between the backends**: a
least-squares fit of the logged positions over i=10..180 gives

```
GL g_dataPos : pos0=(0.50559104, 0.50640416, 0.45077306)  step=(-3.90201694e-4, -5.81144383e-5, +1.86616183e-3)
MT evalPoint : pos0=(0.50559210, 0.50640469, 0.45077338)  step=(-3.90082646e-4, -5.78758658e-5, +1.86622120e-3)
max|resid|   : 5.08e-7 (GL), 5.01e-7 (MT)   <- both traces are exactly linear
```

The per-axis step magnitudes differ by **(GL-MT)/MT = x +0.0305%, y +0.4122%,
z -0.0032%** -- i.e. NOT a uniform scale, so the sample distance is not the
cause. The drift grows linearly and reaches

```
drift (GL - MT position): i=49  (-7.0e-6, -1.2e-5, -3.0e-6)
                          i=99  (-1.3e-5, -2.4e-5, -6.0e-6)
                          i=143 (-1.8e-5, -3.4e-5, -9.0e-6)
                          i=144 (-1.8e-5, -3.5e-5, -9.0e-6)
```

~(-1.8e-5, -3.5e-5, -9e-6) of a unit volume is ~(0.018, 0.035, 0.009) texels of
the 512^3 volume -- exactly Update 15's "<=0.02 texel" estimate. The crossing
happens at i=144 because the ray is skimming the bone surface where the raw
scalar has a near-step (0.0155 -> 0.0182 across the boundary), so the tiny
offset flips the sampled texel.

**Why per-sample lookups agree but composites differ:** they do NOT agree --
there is exactly one divergent sample, and it is the one that matters, because
it sits on a near-step of the data and right at the opacity-knot edge of the TF.

---

## 2. Method: per-sample dump + accumulation replay

### 2.1 Captures

GL per-sample 8-channel dump (uncommitted debug code in
`vtkOpenGLGPUVolumeRayCastMapper.cxx`, `VTK_GL_SAMPLE_DUMP` block) at the
glReadPixels pixel, one `GL_SAMPLE` line per sample:

```
VTK_GL_RAY_DUMP=1 VTK_GL_SAMPLE_DUMP=1 VTK_GL_SAMPLE_DUMP_PX=372,380 \
  build_macos_metal/bin/vtkRenderingVolumeCxxTests \
    TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=OpenGL \
    -D build_macos_metal/ExternalData/Testing -T build_macos_metal/Testing/Temporary \
    -V /tmp/bc/dummy_baseline.png 2> /tmp/bc/gl372.log
```

Metal per-sample `SAMPLE` log (existing `debugMarchGate` in
`Rendering/Metal/MetalShaders.metal`, gated for screenPos (372.5,131.5)):

```
MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=16777216 MTL_LOG_TO_STDERR=1 \
  build_macos_metal/bin/vtkRenderingVolumeCxxTests \
    TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=Metal \
    -D build_macos_metal/ExternalData/Testing -T build_macos_metal/Testing/Temporary \
    -V /tmp/bc/dummy_baseline.png 2> /tmp/bc/true/metal_samples.log
```

Line formats (see section 5):

```
GL:    GL_SAMPLE px=(372, 380) i=0 raw=0.00195315 pos=(0.505592, 0.506404, 0.450773)
       color=(0.000332157, 0.000166078, 9.9647e-05) op=0.00134063
       (color is PREMULTIPLIED; TF rgb = color / op)

MT:    SAMPLE px=(372, 131) i=0 t=0.001911 tex=(0.505603, 0.506417, 0.450677)
       eval=(0.505592, 0.506404, 0.450773) raw=0.001953 norm=0.029291 op=0.001341
       mip=0.000000 rgb=(0.247761, 0.123881, 0.074328) w=1.000000
       accA=0.000000 accC=(0.000000, 0.000000, 0.000000)
       (tex = raw texture coord; eval = cell-to-point adjusted fetch coord;
        rgb is UNpremultiplied)
```

### 2.2 Analysis

`python3 BackendComparisonTools/compare_gl_metal_accum.py /tmp/bc/gl372.log /tmp/bc/true/metal_samples.log 372 131`

reproduces the tables in this document (per-sample region, accumulation replay,
position fit). The script:

1. parses both logs for the paired pixel (Metal (Mx,My) == GL (Mx, 511-My)),
   skipping sentinel samples (op < -8.99e18, logged after ray termination),
2. finds the first sample where op or TF rgb diverges (>1e-4 / >1e-3),
3. replays the front-to-back accumulation `accC += (1-accA)*op*rgb`,
   `accA += (1-accA)*op` for both backends,
4. least-squares fits `pos = pos0 + i*step` over i=10..180 for both traces.

### 2.3 Key correctness checks

- Both marches are exactly linear: max residuals ~5e-7, i.e. at the 7-decimal
  print precision of the logs. The full 0..235 window adds exit-region clamp
  artifacts (GL deviates after the ray leaves the volume); the 10..180 window
  isolates the clean linear region.
- The y-flip pairing (Metal (372,131) == GL (372,380)) is required; comparing
  the same numeric pixel on both backends compares different physical pixels
  (see `compare_gl_metal_samples.py` header).
- The GL `color` channel is premultiplied (`g_srcColor.rgb *= a` in
  `//VTK::Shading::Impl`); the raw TF rgb is recovered by dividing by `op`.
  Confirmed by identical plateau rgb=(1,1,0.9) at i=144 and the pre-knot ramp.

---

## 3. The single flipped sample (i=144) in detail

- GL samples value 1190 (`0.0181582 * 65535`), above the 1150 knot -> the
  pre-integrated TF hits the bone plateau: op 0.400904, rgb (1,1,0.9).
- Metal samples value 1011 (`0.0154270 * 65535`), below the 1150 knot -> op
  0.017987, rgb (1,0.527,0.333) (the linear ramp between 1000 and 1150).
- The two sample positions at i=144 differ by only (-1.8e-5, -3.5e-5, -9e-6)
  (unit-volume coords), yet straddle the near-step of the raw data (raw jumps
  0.0155 -> 0.0182 there). This is why the residual is pixel-localized and
  why the y component of the drift is the dominant driver (y-step diff is
  0.41% vs x 0.03%).

Accumulation replay divergence:

```
i     GL accC                MT accC                 GL aA    MT aA
0     (0.0003, 0.0002, ...)  (0.0003, 0.0002, ...)   0.0013   0.0013
50    (0.1082, 0.0541, ...)  (0.1082, 0.0541, ...)   0.1425   0.1425
100   (0.2931, 0.1466, ...)  (0.2931, 0.1466, ...)   0.3369   0.3369
144   (0.6298, 0.4251, ...)  (0.4320, 0.2229, ...)   0.6904   0.4926   <- first divergence
150   (0.6688, 0.4490, ...)  (0.4959, 0.2620, ...)   0.7295   0.5565
200   (0.8947, 0.6017, ...)  (0.8661, 0.5123, ...)   0.9587   0.9322
235   (0.9341, 0.6293, 0.4749) (0.9307, 0.5575, 0.3898) 0.9981 0.9969
```

Final colors match the observed PNGs: GL [238,160,121], Metal [237,142,99].

---

## 4. Underlying cause: linear sample-position drift

Both backends advance a position pointer by a per-sample step:

```
GL:  g_dataPos += g_dirStep        (raycasterfs.glsl, g_dirStep built in BaseInit)
MT:  evalPoint += evalStep         (MetalShaders.metal debugMarch + march loop)
```

The fitted steps (window i=10..180, ~1e-7 residual => step resolution ~1e-9):

```
GL step = (-3.90201694e-4, -5.81144383e-5, +1.86616183e-3)
MT step = (-3.90082646e-4, -5.78758658e-5, +1.86622120e-3)
|step| diff (GL-MT)/MT: x +0.0305%, y +0.4122%, z -0.0032%
```

Because the three axes do not scale together, the sample distance / stepSize is
NOT the cause (a uniform stepSize change would shift all three axes by the same
percentage). The signature (y-dominant, roughly 7x the x relative error, z ~0)
is consistent with a tiny ray-direction difference between the two backends
dominated by the small y component of the ray. Numerically: absolute step diff
is (GL-MT) = (-1.19e-7, -2.39e-7, -5.9e-8); for a step length ~0.0019 that
corresponds to a direction difference ~(-6e-5, -1.25e-4, -3e-5), i.e. a ~0.007
degree tilt mostly in y. The ray is nearly parallel to z (rayDir
(-0.2045, -0.0304, 0.9784)), so the small y component carries the relative
error.

Confirmed hypothesis: Update 15's "sample positions differ by up to ~0.02
texel" is exactly what happens by sample ~144 (drift x -1.8e-5 unit volume =
0.018 texel of 512^3; y -3.5e-5 = 0.035 texel). The observable symptom is not
a gradual color difference but a single binary TF flip at the near-step.

Caveat: the fitted step difference is below the 7-decimal print precision of
the individual positions; it is only resolvable because ~170 samples are
averaged (resolution ~1e-9). The fit is robust (both max residuals 5e-7, both
axes consistent). Direct full-precision step logging is the confirmed next step
(see section 6).

---

## 5. Reproduction

### 5.1 Logs (as in section 2.1)

### 5.2 Analysis tool

`Rendering/Metal/BackendComparisonTools/compare_gl_metal_accum.py`:

```
python3 compare_gl_metal_accum.py [gl_samples.log] [metal_samples.log] [Mx] [My]
```

Prints the per-sample region around the divergence, the accumulation replay,
and the position fit + drift (all tables in this document). Reference run:

```
python3 BackendComparisonTools/compare_gl_metal_accum.py \
  /tmp/bc/gl372.log /tmp/bc/true/metal_samples.log 372 131
```

### 5.3 Reference artifacts

- `/tmp/bc/gl372.log` - GL 8-channel dump at (372,380), 236 samples/frame x 6 frames.
- `/tmp/bc/true/metal_samples.log` - Metal SAMPLE log at (372,131), 236 samples/frame x 6 frames.
- PNGs: GL baseline `[238,160,121]` at (372,380); Metal `[237,142,99]` at (372,131).

---

## 6. Status and next step

Open question (unchanged from Update 15, now narrowed): why do the two
backends compute slightly different per-sample steps, with the difference
concentrated in the small y direction component? Candidate sources in code:

- GL `rayDir = normalize(ip_vertexPos - in_eyePosObjs[0].xyz)` (per-vertex
  interpolated quad) vs Metal `dirObj = normalize(p.rayDir * boundsSize)` with
  `p.rayDir` computed per-pixel in `computeRayDirection` from camera position
  and screen ray.
- Matrix composition order / adjustedLin vs ip_inverseTextureDataAdjusted.

Next step: log `g_dirStep` (GL) and `evalStep` / `dirObj` / `camera-relative
ray` (Metal) at full precision (%.9e) for the same pixel and frame, and diff
them directly. That will identify whether the ~1.25e-4 y-tilt is a ray-
direction interpolation difference or a step/scale composition difference.

Status of uncommitted working-tree changes relevant to this update: the
`VTK_GL_SAMPLE_DUMP` 8-channel dump block in
`Rendering/VolumeOpenGL2/vtkOpenGLGPUVolumeRayCastMapper.cxx` and the
`debugMarchGate` / `SAMPLE` log in `Rendering/Metal/MetalShaders.metal` remain
uncommitted (kept for the next-step full-precision logging).
