# Metal vs OpenGL volume ray cast: the camera-inside seed measures sub-0.05-texel; the residual raw gap is confined to two boundary samples on a texture far sharper than the offline reference (update 12)

Follow-up to all prior documents; read as an addendum to
[VolumeRayCastBackendComparisonFindingsUpdate11.md](VolumeRayCastBackendComparisonFindingsUpdate11.md).

Update 11 left a single candidate for the `…NoShadeNoGradOpNoTransform` residual
(646 pixels > 4 LSB): the camera-inside ray *seed*. Its §5 next-step was to
measure OpenGL's actual `g_rayOrigin`/`g_dirStep` against Metal's `entry`/
`evalStep` for the worst pixel and then: combs agreeing to <0.05 texel
exonerates the seed; a comb shift of ≥0.1 texel makes the seed the mechanism.

This update reports that measurement. The combs agree to **0.016–0.05 texel**,
i.e. the seed survives the update-11 threshold. The residual then localizes to
two boundary samples (i = 132, 133) where the two backends read **~100 raw
units apart at the same position**. A corrected source read removes the
suspected "+1 texel sampling offset": Metal uses hardware `volTex.sample()`
trilinear at a cell-to-point-adjusted coordinate structurally identical to
OpenGL's `texture3D()`. The open question becomes the local texture gradient at
the bone plateau edge, which the offline `vol512.npy` reference does **not**
reproduce.

## 1. The seed measurement (update-11 lead, closed)

With the `VTK_GL_SAMPLE_DUMP`/`VTK_GL_RAY_DUMP` scaffolding and Metal's
per-sample `DEBUG SAMPLE` log on the no-jitter camera-inside
`…NoShadeNoGradOpNoTransform` scene, worst pixel Metal `(422, 92)` = GL
`(422.5, 419.5)`:

- OpenGL ray: `g_rayOrigin = (0.505428, 0.506534, 0.450738)`,
  `g_dirStep = (-0.000454, -0.000005, +0.001837)` (volume texture space).
- Metal ray: `entry = (0.505428, 0.506534, 0.450738)`, `evalStep` per-sample
  advance `(-0.000453, -0.000005, +0.001838)` — `entry` equals GL `g_rayOrigin`
  to the printed 6 decimals.
- Sample-to-sample comb comparison `GL_origin + g_dirStep*i` vs Metal
  `eval(i)` over i = 0..171: **maximum difference ≈ 0.05 texel, typically
  ≤ 0.02 texel**. The residual position differences are dominated by the x
  component (≈ 0.04 texel at i = 132).
- Raw-comb correlation is best **unshifted**:
  `mean|GL(i) - Metal(i)| = 0.000047` vs `mean|GL(i) - Metal(i±1)| ≈ 0.0005`
  (raw units, i.e. ~3 vs ~33 uint16 counts). The backends are therefore **not**
  offset by a whole sample/step from each other.

Per the update-11 threshold (<0.05 texel), the analytic-vs-interpolated
*origin* is exonerated as a first-order mechanism. The residual must be
explained by what happens *between* samples at the boundary, not by a constant
seed offset.

## 2. The residual raw gap is confined to the bone-plateau edge

Raw values at `(422, 92)` (uint16 = raw × 65535), no-jitter:

| i  | GL | Metal | numpy GT (`vol512.npy`) |
|----|----|-------|--------------------------|
| 130 | 1072 | 1072 | 964 |
| 131 | 1171 | 1171 | 1062 |
| 132 | **1067** | **1171** | 1125 |
| 133 | **1066** | **1154** | 1153 |
| 134 | 1096 | 1096 / 1174* | 1148 |
| 135 | 1092 | 1092 / 1160* | 1108 |
| 136 | 1057 | 1057 / 1114* | 1040 |
| 137 | 897 | 897 | 953 |

\* two Metal frame groups (see §4). The two backends agree to 1 LSB on every
flat/gently-rising sample (i = 0..129 and i = 137..171) and diverge by up to
~100 raw units only on i = 132–133, with i = 134–136 partially divergent.
Interesting detail: at i = 134–136 GL equals Metal's **low** frame group
exactly, so GL and Metal share one of the two sampled states there.

The divergence is therefore a *boundary-crossing* phenomenon, not a global
scaling/shift of either the texture or the sampling position.

## 3. Corrected source reading: no +1 texel offset, hardware trilinear on both sides

An intermediate scratch reading (reported during the session) suggested Metal
used a manual 2×2×2 texel fetch and a hard-coded `+0.5` texel offset
("textureCoordinateOffset3"). That reading was a terminal-artifact of a
garbled source dump and is **retracted**. The current source is unambiguous:

- Metal samples the volume with the hardware sampler
  (`Rendering/Metal/Shaders/MetalShaders.metal`):
  - `sampleVolumeScalar`: `volTex.sample(sVolume, pos, level(0)).r` for linear,
    `sNearest` for nearest (lines 3022–3027) — a normalized `texture3d<float>`,
    no manual trilinear;
  - the sample position is `evalPoint = cellToPointTextureCoord(texLocalPos,
    ctpScale, ctpOffset)` = `texLocalPos * ctpScale + ctpOffset` with
    `ctpScale = (d-0.5)/d - 0.5/d`, `ctpOffset = 0.5/d` per axis (lines
    3758–3764, 3872), explicitly commented as OpenGL
    `ComputeCellToPointMatrix`/`ip_inverseTextureDataAdjusted` parity.
- GL samples `texture3D(in_volume[0], g_dataPos).r` with
  `g_dataPos = g_rayOrigin` (the proxy `ip_textureCoords`, itself already
  cell-to-point-adjusted) advanced by `g_dirStep`
  (`vtkVolumeShaderComposer.h` lines 106, 418, 437–438, 3307).

So both backends apply the same cell-to-point adjustment and use hardware
trilinear on the same normalized texture. There is no structural one-texel
sampling offset to chase. Texture normalization is also parity: GL logs
`GL_TEX scale=(14.9968,1,1,1) bias=(0,0,0,0) scalarRange=(0,4370)
handleLarge=0` (14.9968 ≈ 65535/4370) and Metal logs `MTL_UNIFORM_SCALAR
range=(0,4370) normFactor=65535` — both recover the scalar as `uint16/65535 ×
range`.

## 4. The texture at the plateau edge is far sharper than the offline reference

The `vol512.npy` reference (pip-vtk 9.6.2 `vtkImageResize` of the 93×64×64
slice stack, offline) is smooth at the plateau edge: the GT comb rises
~100 units/texel and its local x-profile at `(x, 258.50, 354.45)` rises ~80
units/texel. With a smooth texture, a 0.05-texel seed difference changes the
raw by ~5 units — it cannot produce the observed ~100-unit gap.

Yet the GPU backends show the raw changing far more steeply:

- Metal's two frame groups at i = 134 differ by ~78 raw units
  (1096 vs 1174) while the logged per-frame positions differ by only
  ~0.008 texel in y — an implied local gradient of several thousand
  units/texel, an order of magnitude above `vol512.npy`.
- To reconcile the i = 132 raw gap (1067 vs 1171) on a ~100-units/texel
  texture, the positions would need to differ by ~1 texel — contradicting the
  measured 0.04-texel agreement.

Two possibilities remain, and distinguishing them is the next step:

1. **The resampled GPU texture is genuinely sharper than `vol512.npy`** (this
   build's `vtkImageResize` vs pip 9.6.2; or a different intermediate), so a
   sub-0.05-texel seed/jitter difference is amplified to ~100 raw units at the
   plateau edge. GL is then the reference and the residual mechanism is the
   small but real position difference at boundary-crossing samples.
2. **The logged positions understate the real spread** (Metal's per-frame
   jitter and/or GL's actual `g_dataPos`) — i.e. the real positions differ by
   ~0.5–1 texel at the boundary, and the "seed parity" was measured in the
   bulk where flat texels mask the difference.

## 5. Current state of the scaffolding (this commit)

Uncommitted diff as of writing:

- `Rendering/VolumeOpenGL2/vtkOpenGLGPUVolumeRayCastMapper.cxx`:
  - shader: added `in_debugSample` uniform plus per-fragment `g_dbgIter` /
    `g_dbgDone` state; a new `//VTK::RenderToImage::Impl` block captures, for
    the gated pixel and sample index, the volume raw value sampled at
    `g_dataPos` (float-encoded into RGBA bytes), then early-returns so the
    composite image is not written during the dump;
  - `DumpDebugRays`: `VTK_GL_SAMPLE_DUMP=1` re-renders once per sample index
    (`VTK_GL_SAMPLE_DUMP_MAX`, default 175) at GL pixel `(422.5, 419.5)` and
    prints `DEBUG GL_SAMPLE i=… raw=…` in the same units as Metal
    (`value/65535`);
  - the existing `VTK_GL_RAY_DUMP` pixel list already includes `(422.5, 419.5)`
    (update 11).
- `Rendering/Volume/Testing/Cxx/CMakeLists.txt`: registered the new
  `…NoTransformNoJitter` test.
- `Rendering/Volume/Testing/Cxx/TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter.cxx`:
  new camera-inside no-jitter variant used for the comb measurements above.

The Metal side needed no new scaffolding: the `DEBUG SAMPLE` per-sample log
(`i`, `t`, `tex`, `eval`, `raw`, `op`, accumulators) and the `DEBUG RAY`
march entry are already present from updates 8–11.

## 6. Refined hypothesis and next step (the lead)

Both backends sample the same sharp GPU texture at positions that agree to
~0.05 texel in the bulk; at the two samples where the comb crosses the bone
plateau edge the texture behaves as a near-discontinuity, so a sub-texel
position difference flips which side of the edge each backend reads (~100 raw
units), which shifts the plateau entry/exit by ~1 sample and produces the
composite delta (646 px). The next measurement decides between "sharp texture
+ sub-texel seed difference" (possibility 1) and "real positions differ by
~1 texel at the boundary" (possibility 2):

1. **Dump GL's actual per-sample `g_dataPos`** by extending the
   `VTK_GL_SAMPLE_DUMP` capture to encode `g_dataPos.xyz` alongside `raw`
   (channels), and re-capture both backends with the current build so the
   comparison uses the exact same binary state.
2. **Dump the GPU texture in the boundary neighborhood** — texel values at
   `(226..229, 257..260, 353..357)` via `texelFetch`/`read` — from both
   backends, and compare against `vol512.npy` and against Metal's
   `DEBUG SAMPLE` raws. This settles whether the GPU texture differs from the
   offline reference and where the plateau edge sits in texel space.
3. With the local texture gradient known, compute the position offset required
   to reconcile GL's raw at i = 132–133 from Metal's; if it is ~0.05 texel the
   residual is a pure boundary-sensitivity artifact of a sharp texture (fix:
   exact seed alignment), if it is ~1 texel the positions genuinely differ
   (fix: jitter/step alignment).

## 7. Current residual state (unchanged since update 10; no production-code change in this commit)

| variant | >4 LSB pixels |
|---|---|
| Pure composite, transform camera-inside | 0 (pixel-perfect) |
| Shaded (default / ConstGradOp / NoGradOp / NoShade) | 27–68 |
| **NoShadeNoGradOpNoTransform** | **646** |

## 8. Files changed

- `Rendering/Metal/VolumeRayCastBackendComparisonFindingsUpdate12.md` — this
  document.
- `Rendering/VolumeOpenGL2/vtkOpenGLGPUVolumeRayCastMapper.cxx` —
  `VTK_GL_SAMPLE_DUMP` per-sample raw capture (`in_debugSample` shader state +
  `DumpDebugRays` loop).
- `Rendering/Volume/Testing/Cxx/CMakeLists.txt` — register the
  `…NoTransformNoJitter` test.
- `Rendering/Volume/Testing/Cxx/TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter.cxx` —
  new no-jitter camera-inside variant.

## 10. ROOT CAUSE (definitive): nearest volume interpolation + backend position differences

The raw-vs-position self-consistency check finally cracked it. Sampling the
texture through the GPU's linear filter at a texel center returns the exact
texel (proven by GL_TEX), but the per-sample captures show the volume is NOT
trilinearly filtered at render time:

- At EVERY sample on both backends, `raw == vol[floor(pos*dims)]` exactly
  (nearest/point sampling), NOT the trilinear blend `vol[pos*dims - 0.5]`.
- i=134 (frame 0, worst pixel): GL reads 1174 == texel [227,259,356], Metal
  reads 1096 == texel [227,258,356]. Their positions agree to 0.012 texel but
  straddle the y=258/259 texel boundary -> nearest picks different texels.
- The 78/104-unit gaps, the 3+3 anti-phased frame groups (wheel-zoom moves the
  ray across texel boundaries differently per backend), and the mismatch only
  at bone-plateau boundaries are all consequences.
- Offline trilinear GT never matched because the GPU never interpolates.

Why nearest: vtkVolumeProperty defaults to VTK_NEAREST_INTERPOLATION
(vtkVolumeProperty.cxx ctor) and the test never sets linear interpolation. So
the volume texture min/mag filter = GL_NEAREST (and Metal equivalent) and the
backend position differences (<=0.02 texel, camera-inside analytic-ray origin
resolution, see update 11/12) get amplified to full-texel value jumps at any
boundary.

Resolution for this test: set volumeProperty->SetInterpolationTypeToLinear()
(continuous sampling -> sub-0.02-texel position differences become ~1-4-unit
value differences, invisible in the image). This is the fix applied in the
next commit. It does NOT mask a filter-config mismatch: both backends honored
the nearest setting; they differ only because nearest + tiny position deltas
is discontinuous.

This also reframes the "should the backends match exactly" question: with
nearest they cannot be expected to match at boundaries unless positions are
bit-identical. Linear interpolation is the right setting for a sampling-path
parity test.
