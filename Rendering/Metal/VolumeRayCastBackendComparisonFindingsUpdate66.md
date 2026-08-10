# Linear-ramp opacity test falsifies step-crossing classification; the residual scales with TF-curve sensitivity (update 66)

Status: **milestone.** A non-saturating linear opacity ramp (no step anywhere)
still produces thousands of near-±1-LSB GL-vs-Metal pixel deltas, growing
monotonically with ramp steepness. Combined with the mode-2 (hard opacity step)
results from update 65, this localizes the residual to the **per-sample
varying-opacity value path** — i.e. the ~1-ulp interpolated-scalar difference
(driver attribute-interpolator floor, updates 59–64) mapped through the opacity
curve dO/ds — and falsifies step-crossing/isosurface-classification as the
mechanism.

## 1. Motivation (from update 65)

Update 65's mode-2 (opacity-only hard step, color constant) test blew up to
**3711 diff px, full-frame, ~20× the 188-px reference**, and — critically —
was **not collapsed by linear interpolation** (3765 px, 97% set overlap). That
left two competing hypotheses for the mode-2 amplification:

1. **Step-crossing sensitivity**: GL and Metal interpolate the scalar to within
   ~1 ulp; samples straddling the opacity step's threshold get classified onto
   different sides, flipping per-sample opacity 0.02↔0.85 → huge accumulated
   delta.
2. **Per-sample varying-opacity value difference**: wherever dO/ds ≠ 0, a
   1-ulp interpolated-scalar difference maps to a per-sample opacity value
   difference that accumulates to a small (±1 LSB) output delta across the
   whole sensitive region.

Update 65 noted both mode 1 (color-only step) and FlatTF were degenerate
(saturated to alpha≈1 → uniform image, 2–3 unique values) and therefore
non-diagnostic.

## 2. What was added

`TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformStepTF.cxx`
grew **mode 3**: constant color, **linear opacity ramp** 0→`VTK_STEP_RAMP_MAX`
(default 0.02) over the full scalar range, no step. Ramp max is sweepable via
`VTK_STEP_RAMP_MAX`. A smooth ramp has **no** isosurface, so hypothesis 1
predicts collapse to ~0 px while hypothesis 2 predicts thousands of ±1-LSB
deltas proportional to dO/ds.

The earlier mode-3 attempt with ramp 0.02→0.85 saturates (alpha→1 over the
long in-volume ray) → uniform byte-identical image (md5 == FlatTF
`1cbf2e8a`), i.e. degenerate; default lowered to 0.02 with the range kept
non-saturating.

## 3. Results

All captures 512×512, checkerboard dummy baseline, threshold 1600, mode 3.

| VTK_STEP_RAMP_MAX | GL md5 | MT md5 | diff px | \|Δ\|>1 | max Δ | uniq GL | GL range |
|---|---|---|---|---|---|---|---|
| 0.005 | `7c338507` | `106b2d47` | **1309** | 0 | 1 | 16 | 31–46 |
| 0.02  | `396f584d` | `3ea29539` | **4923** | 246 | 2 | 52 | 45–96 |
| 0.1   | `8c6130f2` | `6d07b6ed` | **5619** | 1363 | 3 | 109 | 91–206 |

- **The ramp does not collapse the divergence**: even the gentlest ramp
  (0→0.005) yields 1309 diff px, all |Δ|=1.
- **Divergence grows monotonically with ramp steepness** (1309→4923→5619),
  i.e. with dO/ds — exactly what a 1-ulp interpolated-scalar difference mapped
  through a steeper opacity curve predicts.
- Spatial structure (ramp 0.02): diff bbox = full frame (x 0–511, y 0–511,
  512 rows / 511 cols), mean |Δ| among diff ≈ 1.05, and diff pixels are
  **~4× enriched in top-10% image-gradient areas** (23.3% vs 5.6% of non-diff)
  — the divergence tracks where the TF is steep, not where an isosurface sits.
- **m2 GL self-consistency confirmed**: mode-2 (step, color constant) rendered
  twice on GL → byte-identical (0 px). The 3711-px m2 delta is a stable,
  backend-to-backend difference, not GL run-to-run noise.

## 4. Findings

1. **Step-crossing classification is falsified.** A smooth opacity ramp with no
   threshold and no isosurface still diverges at thousands of pixels, so the
   mode-2 amplification is not "samples classified onto different sides of a
   step."
2. **The residual lives in the per-sample varying-opacity value path.**
   The divergence is a monotone function of dO/ds (ramp steepness) and
   concentrates in high-gradient regions. A ~1-ulp interpolated-scalar
   difference (attribute-interpolator floor, u59–64) mapped through the opacity
   curve yields a per-sample opacity difference that accumulates into a
   ±1-LSB output delta everywhere the curve is steep.
3. **This unifies all observations so far:**
   - FlatTF / color-only-step "0 px" results are degenerate (saturated to
     alpha≈1) and were never diagnostic — not evidence that accumulation is
     bit-identical.
   - Gradient-TF reference (188 px, 14 big) = the same mechanism sampled at a
     smooth opacity curve (few steep regions → fewer, mostly small deltas).
   - m2 hard step (3711 px, 23 big, max 15) = the same mechanism at a
     near-delta-function opacity curve (big dO/ds → many deltas + a handful of
     large ones from near-boundary samples).
   - Linear-interp variants (356 px all ±1 on the gradient TF; 3765 px on m2)
     keep the ±1-LSB mass because the interpolated-scalar ulp difference is
     independent of interpolation mode; linear only removes the amplified
     nearest-texel-selection flips.
4. **Root cause remains the u59–64 attribute-interpolator floor**: GL and Metal
   produce interpolated per-sample scalars that differ by ~1 ulp; the output
   delta is the image of that ulp through the local TF slope. No
   accumulation-arithmetic bug is implicated.

## 5. Doubts / hypotheses (open)

- The exact ulp relationship (how many interpolated-scalar ulps differ per
  sample, where, and whether it is GL-beyond-Metal bias or per-sample scatter)
  was characterized at knife-edge pixels only (u63/u64). Whether the ±1-LSB
  ramp mass is consistent with the same ~(+0.025,−0.032)-px displacement at
  every pixel is unverified.
- Whether the divergence is *fully* explained by scalar ulps, or whether a
  residual per-sample opacity-table lookup difference (pf→RGBA table
  interpolation) contributes, is untested. A constant-scalar / flat-region
  probe could separate "scalar ulp → opacity" from "opacity lookup itself".
- Whether GL and Metal agree run-to-run on the ramp tests (GL self-consistency
  was checked for m2 only) is unverified.
- The reference 188-px floor's 14 big-|Δ| pixels remain the u59–64 knife-edge
  set; under this unified model they are simply the steepest dO/ds region of
  the gradient TF.
