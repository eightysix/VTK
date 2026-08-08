# Metal vs OpenGL volume ray cast: jittering eliminated as a candidate (update 7)

Follow-up to all prior documents; read as an addendum to
[VolumeRayCastBackendComparisonFindingsUpdate6.md](VolumeRayCastBackendComparisonFindingsUpdate6.md),
whose §5 attributed the remaining 28 shell-edge masked pixels to "boundary-phase
signature" including *entry phase (`firstT`/jitter)*. This update tests that
specific hypothesis with a jitter-free variant of the comparison test.

## 1. Hypothesis under test

If the residual were driven by the per-pixel random jitter (`jitter ∈ [0, step)`
applied to the first sample offset), then disabling jittering on **both**
backends should either collapse the masked set or shift it significantly, and
the residual would be a *jitter-pattern* artifact rather than a *boundary-phase*
artifact.

## 2. The no-jitter test

New test
`Rendering/Volume/Testing/Cxx/TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideNoJitter.cxx`
(registered in `CMakeLists.txt`) — identical to the comparison test but calls
`mapper->SetUseJittering(false)`. Both backends were captured with it, and the
per-sample Metal `DEBUG SAMPLE`/`DEBUG MARCH` logs were collected for pixel
(307, 8).

Jitter-off is verified on both sides:

- **Metal** (`MetalShaders.metal:4696`):
  `jitter = (useJittering > 0.5 ? volume_random(...) : 1.0) * stepSize`, i.e. the
  no-jitter branch sets `jitter` to exactly one step. With the camera outside
  (`checkBounds`), `firstT = jitter = stepSize` (`:3822-3824`), so the first
  sample is offset by exactly one step — deterministic, no randomness.
- **OpenGL**: `g_rayOrigin += g_dirStep` under the same `useJittering == false`
  branch. The no-jitter paths therefore advance both lattices by exactly one
  step at the seed.

## 3. Results (no-jitter A/B)

Same variant/metrics as Update6 §4 (default step, 512×512,
`mean|Δ|` = per-pixel max-channel absolute delta, masked = `|Δ| ≥ 5/255`).

| metric | Update6 float32 (jitter on) | no-jitter |
|---|---|---|
| center (GL vs M) | [254 175 134] exact | [254 175 134] exact |
| delta mean | −0.05 | 0.05 |
| mean\|Δ\| | 0.14 | 0.14 |
| max\|Δ\| | 8 | 8 |
| masked (≥5) px | 28 | **28** |

The masked set is **identical** in size, location, and magnitude to the jitter-on
float32 run. Jittering is therefore **not** the source of the residual.

## 4. Masked-pixel structure (no-jitter)

The 28 pixels split into two clear groups by channel signature:

- **Top-of-head ring (307, 7-9)**: GL brighter than Metal in all three channels
  (e.g. (307,8): GL=[147,85,58] vs M=[139,79,54], d=[8,6,4]) — GL accumulates
  more opacity.
- **Lower shell ring** (e.g. (227,85) d=[0,4,5], (205,91) d=[0,-4,-5], (211,98)
  d=[0,-5,-6]): R matches to 0/1 but G,B differ by ±4-6, in either sign.

## 5. Per-sample audit of pixel (307,8) — what does *not* differ

Using the no-jitter Metal log for (307,8), each per-sample ingredient was
reconstructed offline and compared against both backends:

1. **Lattice**: 527 samples, `tStart`, `tEnd`, `stepSize` — Metal's logged
   `evalPoint` chain is a uniform advance; count matches GL's `maxSteps`.
2. **Opacity table**: width 1024, range [0, 4370], factor 0.270059,
   `op[i] = 1-(1-f(i*4370/1023))^0.270059` — identical content in both backends
   (same `vtkPiecewiseFunction::GetTable` call).
3. **Lookup convention**: Metal's logged `norm`→`op` pair matches the standard
   texture mapping `x = norm*1024 - 0.5` with linear interpolation between texel
   centers (reconstruction ratio 1.0011 ± 0.013), *not* `norm*(1024-1)`. Both
   backends fetch `texture2D(lut, vec2(norm, 0.5))`, so the convention is shared.
4. **Raw scalar source**: Metal's logged `rawScalar` (with only 206/527 samples
   showing the non-0.5 color-ratio anomaly, all <0.3%) matches a numpy trilinear
   sample of the volume at `evalPoint` to ≤0.1% median.

None of the table/lookup/lattice ingredients explains the GL-vs-Metal gap at this
pixel.

## 6. Step-scale sweep (uniform phase model)

A uniform expansion of Metal's lattice (`evalPoint → entry + (1+ε)·step·m`) was
swept for ε ∈ [−0.06, 0.06] to see whether GL is simply Metal with a slightly
larger step. The best fit was ε = +0.03 → [147,81,54], still 8/255 from GL's
[147,85,58]; ε = 0 (Metal's own lattice) reconstructs [143,78,51] vs the actual
Metal capture [139,79,54], i.e. the offline replay itself carries ±5/255
uncertainty at this opacity level. A **uniform** step/phase difference therefore
does not fit GL's output; the residual needs a non-uniform per-sample
explanation (or a model the replay does not capture).

## 7. Remaining candidates

In Update6 §6 order, minus the seed-phase/jitter item now resolved:

1. **Seed-phase entry difference (non-uniform)**: Metal's `evalPoint` is seeded
   from an analytic box hit + `firstT`, GL's `g_dataPos` from interpolated
   `ip_textureCoords` + `g_dirStep`. Both are first-sample exact when the
   lattice is identical, but a sub-step **entry** offset (not a step scale) at
   the grazing top-of-head rays — where `tStart` is near the box exit and the
   skin dome rises steeply — remains the best explanation for the (307,7-9)
   cluster.
2. **206/527 non-0.5 color-ratio anomaly**: the small set of Metal samples whose
   reconstructed 4-slab color ratio is not 0.5 deserves a direct
   `evalPoint`-vs-ray-linearity dump; if Metal occasionally samples off the ray
   by a fraction of a texel, that is a per-sample scalar error GL does not have.
3. **Promote Metal TF textures RGBA16Float → RGBA32F** (Update6 §6.1): still
   open; expected to shave per-sample LUT-output ulp, not the 5-8 shell deltas.
4. **Minmax-skip measurement** (Update6 §6.3): still open; `g_skip` vs
   `exactSkip`/`skipSteps` on the same rays.

## 8. Rejected / deferred

- Jittering as the source of the residual: **rejected** — no-jitter run
  reproduces the exact 28-pixel masked set.
- A uniform step-scale/phase offset between the lattices: **rejected** as a full
  explanation — best-fit ε sweep leaves 8/255 residual and the per-sample
  lattice audit shows the two lattices agree at the sample level.
