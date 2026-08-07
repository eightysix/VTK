# Metal vs OpenGL volume ray cast: findings

Results of isolating, measuring, and amplifying the Metal vs OpenGL GPU
volume ray-cast divergence, built on the procedures in
[VolumeRayCastBackendComparisonProcedures.md](VolumeRayCastBackendComparisonProcedures.md).
That document defines the environment, the capture/analysis tooling
(`analyze.py`, dummy-baseline captures, `GL_*` engagement checks, the
fixed-step and per-sample-logging machinery). This document records *what
the measurements show* and references the procedures instead of repeating
them.

All figures are Metal minus OpenGL per-channel deltas at 512×512,
captured at identical camera pose on genuinely GL-engaged runs (GL
deterministic run-to-run, verified byte-identical across re-captures).
`mean|Δ|` is the per-pixel max-channel absolute delta averaged over the
image; "masked px" counts pixels with `|Δ| ≥ 5/255`.

---

## 1. Variant isolation: which scene features diverge

The base scene (`TestGPURayCastCameraInsideTransformation`: Phong shading +
gradient opacity `gf: 0@0, 0.5@90, 0.7@100` + camera inside a rotated
volume) and every toggled sibling from the procedures doc's *Image tests*
section were captured on both backends and analyzed:

| variant | mean\|Δ\| | masked px | max\|Δ\| | note |
|---|---|---|---|---|
| base | 3.54 | 80,358 | 15 | `metal ≈ 1.08·gl − 10` (R) |
| NoShade | **8.49** | 117,947 | 32 | `metal ≈ 1.18·gl − 48` (R) — largest |
| NoGradOp | 1.01 | 591 | 15 | |
| ConstGradOp | 1.22 | 563 | 14 | |
| NoShadeNoGradOp | 1.80 | 14,453 | 8 | |
| NoTransform (=NearPlaneTiny=FineStep=Nearest) | 1.31 | 1,444 | 24 | four variants byte-identical |
| CamOutside | 1.82 | 13,501 | 21 | |
| MaxIP | 0.20 | 4,090 | 85 | tiny mean, isolated spikes |
| SampleDist0_5 / SampleDist0_25 | — | — | — | byte-identical to base |

Reads:

- **ShadeOff maximizes divergence.** Removing shading (NoShade) raises
  mean|Δ| from 3.54 to 8.49, *more* than the base. Shading dims the image
  (base center pixel 86 vs NoShade 179), which compresses absolute color
  deltas; the unshaded composite exposes the full dynamic range. Lighting
  is therefore not the divergent feature — it masks a compositing-path
  difference.
- **Gradient opacity is the amplifier.** Dropping gradient opacity
  (NoShadeNoGradOp) collapses 8.49 → 1.80. See section 3 for the
  follow-up isolates that confirm the gradient-opacity table is the
  amplification mechanism.
- **Transform / near-plane clip is not a driver.** NoTransform ≈ NoShade
  (7.72 vs 8.49, section 3); removing the transform only helps once
  gradient opacity is gone.
- **Several probes are inert.** SampleDist0_5/0_25 are byte-identical to
  base: `SetSampleDistance` is ignored while `AutoAdjustSampleDistances`
  is on (the procedures doc flags the FineStep variant as a no-op probe).
  NearPlaneTiny, FineStep and Nearest are byte-identical to NoTransform:
  near-plane clip position, the (ignored) sample-distance override, and
  interpolation type are all non-sources.
- **MIP bypasses the composite and mostly matches.** MaxIP's tiny mean|Δ|
  (0.20) but 85/255 spikes at ~4k pixels show the transfer-function and
  per-sample scalar tracking are fine; the composite path is where the
  divergence lives.

## 2. Fixed-step sweep: divergence is a per-sample artifact

The camera-outside, no-gradient-opacity scene
(`…NoShadeNoGradOpNoTransformCamOutsideFixedStep`) was rendered on both
backends with `AutoAdjustSampleDistancesOff` and the step forced to the
same world value (procedures doc, *Fixed-step sweep*):

| step | 0.0675 | 0.135 | 0.27 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|---|
| mean\|Δ\| | 8.00 | 6.78 | 1.82 | 1.27 | 0.98 | 0.82 | 0.61 |
| masked px | 111,294 | 100,897 | 13,490 | 2,875 | 1,386 | 2,083 | 4,080 |
| Δ mean | +2.41 | +2.06 | −0.36 | −0.83 | −0.60 | −0.45 | −0.23 |

Divergence is maximal at the finest step and falls monotonically as the
step coarsens — a per-sample phase/comb signature (a fixed phase offset
between the backends accumulates more error across more samples), and it
exists **without gradient opacity**. It is step-dependent, so it is not a
half-vs-float accumulation-precision effect (that would be step-
independent).

## 3. New NoShade isolates: the gradient-opacity LUT is the amplifier

Three new variants toggling the gradient-opacity function with shading off
(created in `Rendering/Volume/Testing/Cxx/`, registered in
`CMakeLists.txt`):

| variant | change vs NoShade | mean\|Δ\| | masked px | max\|Δ\| |
|---|---|---|---|---|
| NoShadeConstGradOp | gf constant `0.7` everywhere | 2.61 | 40,768 | 11 |
| NoShadeLinGradOp | gf gentle linear `0@0→1@2000` | 7.69 | 240,531 | 20 |
| NoShadeNoTransform | gf unchanged, transform dropped | 7.72 | 116,398 | 47 |
| (NoShade, ref) | — | 8.49 | 117,947 | 32 |
| (NoShadeNoGradOp, ref) | gf dropped | 1.80 | 14,453 | 8 |

With `ShadeOff` the shader computes the gradient only to feed the
gradient-opacity lookup (`MetalShaders.metal`, `doGradOp` block: `sampleOpacity
*= gf(gradW)`). Flattening gf while still computing the gradient drops
divergence 8.49 → 2.61 (3.3×); a gentle linear gf keeps it large (7.69);
removing the transform does nothing (7.72). Conclusion: **any
gradient-scaled opacity diverges between the backends**, and the steeper
the table's response to `gradW`, the larger the amplification. The
residual floor (~1.8–2.6) is the gf-independent phase/comb artifact of
section 2.

## 4. Offline gradient verification: Metal's gradient is correct

Using the procedures doc's *Per-sample GPU logging* machinery on NoShade
(`/tmp/bc/metal_noshade.log`, 11,130 SAMPLE + 10,674 GRADOP dumps; the
NoShade run logs the gradient-opacity block's `GRADOP` lines since the
shading `LIGHT` lines only fire when shading is on — parsed by the new
`/tmp/bc/verify_gradient_noshade.py`), Metal's per-sample `gradW` was
replayed against numpy ground truth (`vol512.npy`, procedures doc *Offline
verification*). 1752 samples with `gradW > 0`:

| gradW bin | n | replay ratio median | p10–p90 |
|---|---|---|---|
| 0.001–0.01 | 698 | 0.982 | 0.853–1.132 |
| 0.01–0.03 | 307 | 0.985 | 0.859–1.163 |
| 0.03–0.1 | 572 | 0.999 | 0.914–1.087 |
| 0.1–0.3 | 56 | 0.996 | 0.963–1.019 |

**Metal's gradient magnitude matches ground truth at its own sample
positions** (median ratio ≈ 1.0 at every magnitude; tightest for the
strongest gradients). No systematic half-precision bias in the gradient
path. The observed gf LUT quantization (~256 entries over
`[0, 0.25·range]`, step ≈ 4.2 data units; 71% of samples produce nonzero
`gradOp`) plus the steep `0.5@90→0.7@100` knee — only ~2.4 LUT bins wide —
means a ~5% `gradW` shift near the knee flips a full LUT level (Δgf ≈
0.094, ~13% opacity swing). This quantifies exactly why gradient opacity
amplifies small gradient differences.

Caveat: the replay is self-consistent (it evaluates numpy at Metal's own
logged positions), so it validates the gradient *formula*, not the
sample *positioning*. It cannot by itself distinguish a Metal sample-phase
offset from an OpenGL one — but it rules out a Metal gradient-formula or
precision error.

## 5. Amplifier variant: `NoShadeAmp`

`TestGPURayCastCameraInsideTransformationNoShadeAmp.cxx` combines the three
amplifiers in one scene, camera inside the axis-aligned volume (long rays):
`ShadeOff`, a steep gf ramp `0@0 → 1@90 → 1@2000` (20× steeper than the
base table below 90, positioned over the interior samples' gf-input range
p10–p90 ≈ 1–90 data units), and `AutoAdjustSampleDistancesOff` with a
fixed fine step (default 0.008 world units; `VTK_FIXED_SAMPLE_DISTANCE`
overrides it for a sweep). Both backends honor the forced step
(`PROBE fixedSD=… autoAdjust=0`).

Step sweep (Metal vs GL):

| step | mean\|Δ\| | masked px | max\|Δ\| | Δ mean | Metal sat. |
|---|---|---|---|---|---|
| 0.0675 | 9.60 | 211,884 | 60 | +1.9 | 65% |
| 0.03375 | 28.04 | 261,864 | 79 | +12.0 | 66% |
| 0.016875 | 30.27 | 261,915 | 82 | +11.2 | 62% |
| **0.008 (default)** | **45.83** | **260,773** | **168** | −6.6 | 35% |
| 0.004 | 169.10 | 262,144 | 225 | −126.2 | 0% (collapsed) |

At the default step the divergence is **5.4× the NoShade baseline**
(mean|Δ| 45.83 vs 8.49) with ~99.5% of pixels differing and
`metal ≈ 2.53·gl − 397`.

**Side finding — OpenGL is step-invariant at fine steps, Metal is not.**
Across the whole Amp sweep OpenGL's render barely moves (mean luminance
188.5, 26% of pixels saturated at *every* step), consistent with correct
opacity pre-integration scaling the per-sample opacity by the step. Metal's
render shifts dramatically with step (mean luminance 190 → 62; saturation
65% → 0%). At 0.004 the Metal composite collapses to near-black — a
degenerate breakdown, which is why the test defaults to 0.008 (maximal
divergence while both renders remain structurally comparable). This
asymmetry is the strongest single clue for the root cause: **Metal's
composite does not step-compensate opacity the way OpenGL's does.**

## 6. Synthesis

The evidence chains:

1. The divergence lives in the per-sample composite/accumulation path
   (MIP and the interpolate/clip/transform probes are inert; shading only
   masks it).
2. It is step-dependent (section 2, 5): a phase/comb artifact that grows
   with sample count.
3. Metal's gradient formula is ground-truth-correct (section 4), so the
   divergent input is the sample evaluation position/phase, not the
   gradient math or half-precision rounding in the gradient.
4. The steep gradient-opacity LUT amplifies that small per-sample
   difference into large opacity swings (section 1, 3, 4) — the reason
   `NoShade` is the worst simple variant and `NoShadeAmp` the worst
   overall.
5. Metal's composite is not step-invariant under fine fixed steps
   (section 5), pointing at a step/pre-integration asymmetry between the
   backends' opacity handling.

Open hypotheses for the root cause, in order of likelihood given the
above:

- **Metal applies a different per-sample opacity step-compensation (or
  none) than OpenGL**, so the integrated opacity changes with step on Metal
  and is stable on OpenGL (section 5 side finding).
- **A fixed sampling-position/phase offset between the backends' rays**,
  amplified by sample count (sections 2, 5) and by the gf LUT (section 4).

Next steps (see the procedures doc's *Precision probes* and *Per-sample GPU
logging*): run the float-accumulation and scalar-normalization probes on
`NoShadeAmp` to rule half-precision accumulation in or out at the extreme
step; and instrument OpenGL-side per-sample values (or compare both
backends' logged entry/first-sample positions) to measure the phase offset
directly rather than infer it.

## 7. New artifacts and files (not in the procedures doc)

| artifact | contents |
|---|---|
| `Rendering/Volume/Testing/Cxx/TestGPURayCastCameraInsideTransformationNoShadeConstGradOp.cxx` | NoShade + constant gf 0.7 |
| `Rendering/Volume/Testing/Cxx/TestGPURayCastCameraInsideTransformationNoShadeLinGradOp.cxx` | NoShade + linear gf 0→1 over [0,2000] |
| `Rendering/Volume/Testing/Cxx/TestGPURayCastCameraInsideTransformationNoShadeNoTransform.cxx` | NoShade + no vtkProp3D transform |
| `Rendering/Volume/Testing/Cxx/TestGPURayCastCameraInsideTransformationNoShadeAmp.cxx` | amplifier (section 5), default fixed step 0.008 |
| `CMakeLists.txt` entries | all four above registered in `Rendering/Volume/Testing/Cxx/` |
| `/tmp/bc/make_metal_noshade_log.sh` | regenerates `/tmp/bc/metal_noshade.log` (NoShade variant of the procedures doc's `make_metal3_log.sh`) |
| `/tmp/bc/verify_gradient_noshade.py` | numpy replay of `gradW`/`gradOp` from GRADOP log lines (shade-off analog of the procedures doc's `verify_gradient.py`) |
| `/tmp/bc/caprun/` | fresh per-variant captures, logs, delta heatmaps/masks: `TestGPURayCastCameraInsideTransformation*`, `sweep/` (camera-outside step sweep), `TestGPURayCastCameraInsideTransformationNoShadeAmp*` (amplifier + fine-step sweep `sd0.03375…0.004`) |

All captures were verified GL-engaged (`GL_SAMPLING`/`GL_OPTABLE`/`GL_TEX`
stderr logs present; Metal runs show none).
