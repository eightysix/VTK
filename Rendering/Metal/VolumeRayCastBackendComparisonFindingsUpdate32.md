# Camera-inside: the 78-unit raw "spike" is a nearest-texel boundary straddle from accumulated x-drift, not a scalar-sampling bug (update 32)

**Date:** 2026-08-08
**Scope:** Execute update 31's probe A (classify the 78-unit raw spike at the iso-1150 crossing) and probe B (sample-count parity). Probe A is conclusive from the update-31 per-sample dumps alone: the spike at (422,·) `i=167` is a **nearest-interpolation texel-boundary straddle** produced by the accumulated per-step position drift between the two marches, and it disappears the moment the sample is ≥0.01 texel from a texel boundary. Combined with the default interpolation being `VTK_NEAREST_INTERPOLATION` (confirmed, `vtkVolumeProperty.cxx:25`), this explains why a ~0.008-texel position offset (irrelevant under trilinear) flips the sample under nearest.

**Follows:** [Update 31](VolumeRayCastBackendComparisonFindingsUpdate31.md).

---

## 1. Confirmed: default interpolation is NEAREST

`vtkVolumeProperty.cxx:25` sets `this->InterpolationType = VTK_NEAREST_INTERPOLATION` in the constructor. `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter.cxx` never calls `SetInterpolationType`, so both backends run the test volume in **nearest** mode. This reframes everything downstream: a position offset that is harmless under trilinear (a few millitexels smear to nothing) is **amplified to a full texel's scalar difference** under nearest whenever the two marches land on opposite sides of a texel boundary.

## 2. Probe A result — the spike is a texel-boundary straddle

The 78-unit raw difference at (422,·) `i=167` (GL 0.018494 vs MT 0.017304) has a simple geometric cause. Reconstructing both marches' x-texel position at the crossing (cellToPoint convention, texel index = `floor(pos*512)`, boundary between texels 219/220 at x = 220.0):

| i | GL x (texel) | MT x (texel) | GL-to-boundary | MT-to-boundary | rawGL | rawMT |
|---|---|---|---|---|---|---|
| 166 | 220.2273 | 220.2348 | 0.77 | 0.76 | 0.016983 | 0.016983 |
| **167** | **219.9951** | **220.0028** | **0.005** | **0.003** | **0.018494** | **0.017304** |
| 168 | 219.7628 | 219.7704 | 0.24 | 0.23 | 0.018662 | 0.018662 |

At `i=167` GL sits 0.005 texel **below** the 219/220 boundary and Metal 0.003 texel **above** it. Under nearest sampling GL rounds to texel 219 (raw 0.018494) and Metal rounds to texel 220 (raw 0.017304) — the 78-unit difference. At every other sample the two marches are ≥0.02 texel from a boundary on the same side, so they agree to <0.03 units. The "single scalar spike" of update 31 is therefore **not** a scalar-sampling bug; it is the nearest-mode amplification of a ~0.008-texel position straddle.

### Where the straddle comes from: accumulated x-drift

The two marches' x positions diverge linearly as they advance (per-step drift `(MTx−GLx)` growing ~0.000045 texel/step in x for (422,·)):

| i | dx = MTx − GLx (texel) |
|---|---|
| 0 | +0.00021 |
| 50 | +0.00254 |
| 100 | +0.00437 |
| 140 | +0.00624 |
| 160 | +0.00718 |
| 167 | +0.00776 |

By `i=167` the drift is ~0.0078 texel — just enough to straddle the texel boundary that the (422,·) ray crosses at that sample. The (372,·) ray's crossing samples (i=215–232) all land ≥0.08 texel from a boundary, so it shows **no** raw spike (raw agrees to <0.03 units through the crossing) — its knife-edge is instead decided by the far-side sample-count difference (below). The drift itself is the ~0.006% step-length disagreement measured in update 24 (metal step length vs GL `g_dirStep` not bit-exact), now shown to matter only through nearest-mode boundary straddles.

## 3. Probe B — far-side sample count diverges per-pixel (already in update-31 dumps)

- (372,·): GL 233 real samples vs Metal 236 → Metal marches 3 further.
- (422,·): GL 180 vs Metal 170 → Metal stops 10 earlier.

This is consistent with the same drift + different termination predicates (Metal's precomputed `maxSteps = ceil((tEnd−firstT)/stepSize)` vs GL's position-bounds loop `g_dataPos` vs `in_texMin/Max`). The (372,·) flip is decided by this count difference at the far side; the (422,·) flip is decided by the crossing straddle. Both stem from the marches not being bit-identical.

## 4. Corrected interpretation vs update 31

- Update 31 proposed the spike could be "nearest-vs-trilinear mode difference, a border/tile fetch difference, or a float32 weight difference". None of these: both backends use the same (nearest) mode; the spike is a **position boundary straddle**.
- Update 31's "two independent drivers" (scalar spike + sample count) are better stated as **one driver with two manifestations**: the marches' positions are not bit-identical (per-step accumulation drift + different termination predicate), and under nearest mode that position difference produces (a) raw flips where the ray crosses a texel boundary and (b) a different far-side sample count.
- Position agreement in update 31 (≤0.006 texel) was correctly measured but the conclusion that "step accumulation is not a significant divergence source" was wrong at nearest precision: a 0.008-texel offset is fatal at a texel boundary.

## 5. Next probes (if pursued)

1. **Bit-exact step parity**: make Metal's per-step advance bit-identical to GL's `g_dirStep` accumulation (`g_dataPos += g_dirStep` in cellToPoint-adjusted texture space; Metal currently accumulates `evalPoint += evalStep` where `evalStep` is built from `normalize(anchorData−cameraData)` folded through `adjustedLin * sampleDistanceWorld`). If both marches then straddle boundaries identically, the residual should drop to ~0.
2. **Termination parity**: replace Metal's precomputed `maxSteps` with GL's position-bounds loop (`g_dataPos` vs `in_texMin[0]`/`in_texMax[0]`) to equalize the far-side count.
3. Verify with the two existing images + `cmp.py` (no new instrumentation needed for 1/2).

## 6. Reproduction

Analysis only — no production code changed. Inputs: `/tmp/bc/u31/{GL_372.log,GL_422.log,Metal.log}` from the update-31 captures. The straddle table (section 2) and drift table are reproduced by diffing the last-frame per-sample x positions ×512 with the boundary condition `floor` split at `220.0`.

Artifacts: `/tmp/bc/u31/` (unchanged from update 31).
