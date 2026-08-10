# Finetuned diagnostic tests isolate the residual to the per-sample scalar/opacity fetch and its accumulation (update 65)

Status: three new finetuned tests built and captured GL-vs-Metal on the
current binary (`build_macos_metal`). The Metal-vs-GL residual for the reference
composite test is confirmed **unchanged at 188 px** (14 px |Δ|>1), and the
new tests **falsify the "accumulation arithmetic is the culprit" hypothesis
while amplifying a distinct, much larger opacity-path divergence to 3711 px.**

## 0. Re-verification of the reference (correcting the 02:02 false positive)

The 02:02 "byte-identical, 0 diff px" result was a **stale-copy artifact**:
`/tmp/bc/rec_metal_fresh.png` carried GL's md5 (`dc4bab2e`, 110938 B). The
genuine Metal render from that run is `build_macos_metal/Testing/Temporary/
dummy_baseline.png` = `a67b9eb9` (110941 B) — byte-identical to
`u59_metal.png`. Fresh re-captures confirm:

| backend | stored image md5 | size |
|---|---|---|
| OpenGL | `dc4bab2eb48d8f894babc6fd801193b1` | 110938 B |
| Metal   | `a67b9eb9b90612b1dc00333e7423ffbd` | 110941 B |

diff = **188 px** (14 px |Δ|>1, max 8), exactly u59's number. No change.

## 1. Motivation

Update 64 concluded the residual is a driver-level attribute-interpolator
floor (effective sample displacement ~(+0.025,−0.032) px GL-beyond-Metal,
scatter explained by f32-texcoord ulp amplification). Before accepting a
driver floor, we want to know *which stage* the divergence lives in:

1. accumulation arithmetic (front-to-back composite of per-sample rgba), or
2. the per-sample scalar → TF (color/opacity) lookup path.

FlatTF (constant TF) already implied accumulation is fine for constant rgba
(geometric series, byte-identical). MaxIP (no accumulation, same gradient TF)
still diverges (102 px) → divergence is *upstream* of the composite. The new
tests sharpen this.

## 2. New tests added

- `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformLinear.cxx`
  — reference scene with `SetInterpolationTypeToLinear()` (default is nearest).
- `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformStepTF.cxx`
  — single hard step in color and/or opacity at env-controlled threshold
  (`VTK_STEP_THRESHOLD`); mode `VTK_STEP_MODE`: 0 = both step, 1 = color-only
  step (opacity constant), 2 = opacity-only step (color constant);
  `VTK_STEP_LINEAR=1` switches to linear volume interpolation.

Both registered in `Rendering/Volume/Testing/Cxx/CMakeLists.txt`.

## 3. Results

All captures: 512×512, camera inside volume, no jitter, fail-and-dump via a
checkerboard dummy baseline (VTK's image compare has a pass threshold, so a
flat/black dummy can silently pass near-uniform renders — use a high-contrast
pattern, see §4).

| test / setting | diff px | \|Δ\|>1 | max Δ | notes |
|---|---|---|---|---|
| composite NoJitter (nearest, reference) | 188 | 14 | 8 | unchanged (u59 floor) |
| MaxIP (gradient TF, no accumulation) | 102 | 102 | 27 | diverges w/o accumulation |
| FlatTF (both channels constant) | **0** | 0 | 0 | byte-identical |
| Linear interp (gradient TF) | 356 | **0** | **1** | divergence deltas all collapse to ±1 LSB |
| StepTF m1 color-only step (opacity const 0.1) | **0** | 0 | 0 | byte-identical |
| StepTF m2 opacity-only step (color const) | **3711** | 23 | 15 | huge, full-frame |
| StepTF m2 + linear interpolation | **3765** | 24 | 9 | NOT collapsed by linear |
| StepTF m0 both step @1600 | 125 | 20 | 20 | |
| StepTF sweep (m0) T=300/950/1600/2250+ | 41 / 18 / 125 / 100 | | | ≥2250 renders are ~uniform (2 unique vals) |

Pixel-set analysis (m2 nearest vs m2 linear): **3667 / 3711 px overlap** — the
opacity-step divergence is essentially the same set under linear interpolation.

Overlap of m2 (nearest) with the reference composite 188-px set: **2 px** —
nearly disjoint.

## 4. Findings

1. **Accumulation arithmetic is exonerated (again, more strongly).**
   FlatTF = 0 and color-only-step (m1) = 0: with constant (or only-color-varying)
   rgba, GL and Metal composite to byte-identical output even with a hard TF
   discontinuity present in color.
2. **The divergence is in the per-sample opacity path.**
   m2 (opacity-only step, color constant) diverges at **3711 px** — ~20× the
   reference's 188 px. The only difference vs FlatTF is that opacity now varies
   through the hard step, so the residual lives where opacity is fetched /
   accumulated per sample.
3. **Linear interpolation does NOT fix it.**
   m2+linear ≈ m2 (3765 vs 3711, 97% set overlap). This falsifies the
   hypothesis that the opacity-path divergence is a nearest-texel-selection
   flip that linear filtering would smooth away. (Contrast: the reference
   gradient-TF test collapses its 14 big-|Δ| px to ±1 under linear — there the
   knife-edge flips *are* nearest-texel flips.)
4. **The m2 set is full-frame and disjoint from the reference knife-edge set**
   (2/3711 overlap) — it is a *different* phenomenon, not the same 188-px
   knife-edge amplified.
5. **VTK image-compare pass threshold gotcha:** a near-uniform render passes
   against a flat/black dummy baseline (ImageError 0) — the earlier sweeps at
   thresholds ≥2250 silently produced uniform images and "passed". All
   captures in this update use a 16-px red/green checkerboard dummy so any
   render fails and dumps.

## 5. Doubts / hypotheses (open)

- Is m2's 3711-px divergence driven by (a) a per-sample opacity *value*
  difference (scalar fetch → pf lookup), (b) the accumulation of a hard
  opacity step (rounding of `1-opacity` chains), or (c) a step-crossing /
  sample-count difference near the isosurface? m1=0 vs m2=3711 with the same
  step threshold and same scalar fetch argues it is specific to the opacity
  channel, but which sub-stage is unresolved.
- Why does m2 not collapse under linear interpolation while the reference
  gradient-TF knife-edge flips do? If m2 were the same anchor/nearest-texel
  mechanism, linear should have reduced it like it did the reference test.
- The m2 divergence bounding box is the full frame (x 0–511, y 0–511) with a
  high unique-value count on both backends — not a localized knife edge. Its
  spatial structure is not yet characterized (is it aligned to the opacity
  isosurface? per-ray depth?).
- Does m2's divergence exist in GL *itself* run-to-run (i.e., is it a stable
  GL-vs-Metal delta or a self-inconsistent GL)? Not yet checked.
- The reference 188-px floor (u59–64: driver interpolator) is untouched and
  consistent with findings 2–3: it is a *different, smaller* phenomenon than
  the m2 opacity divergence.
