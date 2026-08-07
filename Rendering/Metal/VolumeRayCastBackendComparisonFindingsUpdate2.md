# Metal vs OpenGL volume ray cast: accumulation-precision and ray-length fixes (update 2)

Follow-up to the two existing documents, read as an addendum to all three:

- [VolumeRayCastBackendComparisonProcedures.md](VolumeRayCastBackendComparisonProcedures.md) —
  environment, capture/analyze tooling (`analyze.py`, dummy-baseline captures,
  `GL_*` engagement checks, fixed-step and per-sample-logging machinery).
- [VolumeRayCastBackendComparisonFindings.md](VolumeRayCastBackendComparisonFindings.md) —
  sections 1–7, all still valid.
- [VolumeRayCastBackendComparisonFindingsUpdate.md](VolumeRayCastBackendComparisonFindingsUpdate.md) —
  the composite-gate (0.001h) and termination-threshold (0.99h) fixes; its
  probe-campaign conclusions in §1.1/§1.2 are **corrected** here (§4).

This document records two further confirmed root causes found and fixed in
`Rendering/Metal/Shaders/MetalShaders.metal`, the proof each was the mechanism,
the updated backend-vs-backend numbers, and the remaining gap.

Measurement scenes (identical camera pose, both backends, genuinely
GL-engaged — verified via the `GL_SAMPLING`/`GL_OPTABLE`/`GL_TEX` stderr logs,
25 lines on GL runs, none on Metal):

- `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideFixedStep`
  (camera outside, no transform, no shading, no gradient opacity, fixed step
  forced via `VTK_FIXED_SAMPLE_DISTANCE`, `autoAdjust=0` confirmed by the
  `PROBE fixedSD=… autoAdjust=0` stderr line).
- `TestGPURayCastCameraInsideTransformationNoShadeAmp` (camera inside,
  `ShadeOff`, steep gradient-opacity ramp `0@0 → 1@90 → 1@2000`, fixed step;
  the highest-delta test from findings §5).

All figures are Metal minus OpenGL per-channel deltas at 512×512. `mean|Δ|` is
the per-pixel max-channel absolute delta; "masked px" counts pixels with
`|Δ| ≥ 5/255`. Unless noted, "after" numbers are the cumulative state: both
fixes below **plus** the earlier gate/threshold fixes from Update1.

---

## 1. Confirmed root cause #1: fp16 composite accumulation stalls and drops the volume tail

### 1.1 Mechanism

The front-to-back composite accumulators were declared `half`:

```
half3 accumulatedColor = initialColor;
half accumulatedOpacity = initialOpacity;
...
half weight = 1.0h - accumulatedOpacity;
accumulatedColor += weight * sampleColor * sampleOpacity;
accumulatedOpacity += weight * sampleOpacity;
```

Once `accumulatedOpacity` grows past ~0.82, each subsequent per-sample
increment `(1 − accA) · op` falls below the fp16 mantissa ulp at that
magnitude (ulp ≈ 0.00049 at 0.8). Each addition rounds to zero, so the shader
**silently stops compositing the volume tail** even though the march continues
and the samples are evaluated. The far side of any opacity that has already
reached ~0.82 therefore never reaches the framebuffer, and interior pixels
render systematically darker than the float32 GL reference.

### 1.2 Proof (per-sample GPU log vs fp16 replay)

At the residual ring pixel (32,346) of the camera-outside 0.0675 scene, the
GPU's own logged accumulation ends at

```
DEBUG FINAL px=(32, 346) lastIter=2111 accOp=0.824219 accCol=(0.824219, 0.488281, 0.322754)
```

A float64 replay of the **same logged per-sample op/rgb values** composites
through `accA += (1−accA)·op` to `accA = 0.979` → pixel ≈ [248,143,97] =
**exactly GL's [249,144,98]**. The log is internally inconsistent: the GPU
accumulated only 0.8242 of opacity while its own logged opacities integrate
to 0.979. Replaying the same samples with **fp16** arithmetic reproduces the
GPU state byte-for-byte:

| | accA | accC (R,G,B) | predicted pixel (bg-blended) |
|---|---|---|---|
| GPU log (measured) | 0.824219 | (0.824219, 0.488281, 0.322754) | [215,129,87] |
| fp16 replay of logged op | 0.824219 | (0.8242, 0.4883, 0.3225) | [214,129,86] |
| float64 replay of logged op | 0.9790 | (0.973, 0.562, 0.381) | [248,143,97] |

The inferred per-sample opacity actually composited (`(accA[i+1]−accA[i]) /
(1−accA[i])`) is 0 for every sample after i ≈ 250 while the logged opacity
stays 0.00136 — the fp16 stall, exactly.

### 1.3 Fix (commit `38343bdb4b`)

Promote the composite accumulators, the weight arithmetic, and the opacity
termination threshold to `float`; the per-sample TF values stay `half`. MSL
rejects implicit `float3`/`half3` conversion, so the accumulation lines use
explicit casts:

```
float3 accumulatedColor = float3(initialColor);
float accumulatedOpacity = float(initialOpacity);
...
float weight = 1.0f - accumulatedOpacity;
accumulatedColor += weight * float3(sampleColor) * float(sampleOpacity);
accumulatedOpacity += weight * sampleOpacity;
```

### 1.4 Effect

Camera-outside fixed step 0.0675 (cumulative: gate + threshold + float acc):

| metric | Update1 (gate+threshold) | after float acc |
|---|---|---|
| mean\|Δ\| | 2.37 | **0.31** |
| masked px | 47,578 | **2** |
| max\|Δ\| | 34 | **6** |
| ring px (32,346) | Metal [215,129,87] vs GL [249,144,98] | Metal **[249,144,98]** (exact) |
| ring px (45,113) | [192,118,84] | [198,119,84] vs GL [198,120,84] |
| per-channel fit | metal = 1.0182·gl − 4.71 (R) | metal = 1.0000·gl − 0.03 (R), 0.9999·gl − 0.06 (G/B) |

## 2. Confirmed root cause #2: the fixed `MAX_RAY_STEPS = 8192` clamp truncates long rays

### 2.1 Mechanism

- **OpenGL** (`raycasterfs.glsl`): `while (!g_exit)` — an unbounded loop. It
  terminates only when the sample position passes the exit face
  (`g_currentT >= g_terminatePointMax`) or the opacity threshold is reached.
  There is no sample-count cap.
- **Metal**: `maxSteps = min(max(1, int(ceil((p.tEnd - firstT) / p.stepSize))), MAX_RAY_STEPS)`
  clamped the per-ray sample count to 8192.

For short rays the clamp is never hit, but **camera-inside rays at fine fixed
steps exceed it**. The NoShadeAmp ray to the far corner is ~165 world units
long; at step 0.008 that is ~20,600 samples > 8192. Metal composited only the
first 8192 · 0.008 ≈ 65 world units (~40% of the ray) and rendered the far
end dimmer than GL.

This is the same mechanism as findings §5's "Metal is not step-invariant / the
composite collapses at fine steps": the truncation distance is `8192 · step`,
so halving the step halves the composited ray length. At 0.004 only ~20% of
the ray was composited → the near-black collapse.

### 2.2 Proof (A/B: remove the clamp, rebuild, re-capture)

NoShadeAmp step sweep, Metal vs GL:

| step | cap on (current before fix) | cap removed (probe) |
|---|---|---|
| 0.008 | mean\|Δ\| **19.13**, masked 136,429, lum_M 175.1 | mean\|Δ\| **6.53**, masked 117,534, lum_M 186.9 |
| 0.016875 | 6.50 | 6.50 |
| 0.03375 | 6.53 | 6.53 |
| 0.0675 | 6.55 | 6.55 |

Removing the clamp alone turns the 0.008 outlier into the same ~6.5 floor as
every other step, and Metal's mean luminance (186.9) snaps to GL's (188.6).
The step-dependence is gone; the divergence is now step-independent.

### 2.3 Fix (commit `e66a38ff60`)

Compute `maxSteps` from the ray geometry alone — always finite for a bounded
volume — and delete the `MAX_RAY_STEPS` constant:

```
int maxSteps = max(1, int(ceil((p.tEnd - firstT) / p.stepSize)));
```

The march still terminates unconditionally via the existing checks (exit face
`currentT >= p.tEnd`, opacity threshold `1 − 1/255`, `tTerminateMax`, and the
block-bounds exit), so unbounded sample counts cannot loop forever.

## 3. Updated backend-vs-backend numbers (both fixes applied)

### 3.1 Camera-outside fixed-step sweep (isolation scene)

| step | original (findings §2) | after both fixes |
|---|---|---|
| 0.0675 | 8.00 / 111,294 | **0.31 / 2** |
| 0.135 | 6.78 / 100,897 | **0.32 / 29** |
| 0.27 | 1.82 / 13,490 | **0.33 / 264** |
| 0.5 | 1.27 / 2,875 | **0.34 / 600** |
| 1.0 | 0.98 / 1,386 | **0.35 / 1,029** |
| 2.0 | 0.82 / 2,083 | **0.35 / 1,769** |
| 4.0 | 0.61 / 4,080 | **0.34 / 3,378** |

(`mean|Δ| / masked px`.) The scene is effectively solved: at every step the
per-channel fit is `metal = 1.0000·gl + ε` with |ε| ≤ 0.06, and the ring pixel
is exact. The few hundred masked pixels at coarse steps are grazing-ring-edge
pixels where the sample phase (a fraction of a step, comparable to the step
itself at 2.0–4.0 world units) lands on different material — hence the growing
single-pixel `max|Δ|` (39/76/124) with a still-tiny mean.

### 3.2 NoShadeAmp step sweep (amplifier scene)

| step | findings §5 | Update1 (§4.2) | after both fixes |
|---|---|---|---|
| 0.008 | 45.83 / 260,773 / −6.6 | 33.43 / 251,405 / −15.67 | **6.53 / 117,534 / −1.64** |
| 0.016875 | 30.27 / 261,915 / +11.2 | — | **6.50 / 116,865 / −1.65** |
| 0.03375 | 28.04 / 261,864 / +12.0 | — | **6.53 / 117,129 / −1.66** |
| 0.0675 | 9.60 / 211,884 / +1.9 | — | **6.55 / 117,094 / −1.68** |

(`mean|Δ| / masked px / Δ mean`.) The step-sensitivity and the fine-step
collapse are gone. Center pixel is now essentially exact: GL [247,200,166] vs
Metal [247,201,167].

### 3.3 Base and NoShade variants (unchanged — separate root cause)

| variant | findings §1 | after both fixes |
|---|---|---|
| base (shade + gf) | 3.54 / 80,358 | 3.69 / 84,020 |
| NoShade (gf on) | 8.49 / 117,947 | 8.54 / 120,220 |

Both are statistically unchanged: their divergence is the gradient-opacity
amplification path (findings §3/§4), which neither fix addresses.

## 4. Corrections to Update1 probe conclusions

Two "inert" probe results from `VolumeRayCastBackendComparisonFindingsUpdate.md`
were conclusions drawn from probes that never exercised the mechanism:

- **§1.2 "half → float composite accumulation — byte-identical, inert".**
  Wrong. As written, that probe cannot compile: MSL rejects implicit
  `float3`/`half3` conversion (`implicit conversions between vector types are
  not permitted`), so the render silently fell back to the cached library and
  appeared byte-identical. The float promotion in §1 changes the output
  decisively once the explicit casts are added (proof in §1.2 above).
- **§1.1 "MAX_RAY_STEPS 8192 → 65536 — inert".** Tested on the
  camera-outside ray (~1,100 samples), where the cap is never reached — inert
  for the wrong reason. The cap only bites on long camera-inside rays
  (section 2), which the probe never exercised.

## 5. Remaining gap and interpretation

After all four fixes (gate, threshold, float accumulation, cap removal) the
residual is step-independent and concentrated in the gradient-opacity scenes:

| scene | mean\|Δ\| | masked px | d mean | fit (R) |
|---|---|---|---|---|
| camera-outside fixed-step (no gf) | 0.31 | 2 | −0.06 | 1.0000·gl − 0.03 |
| base (shade + gf) | 3.69 | 84,020 | −0.93 | 1.0864·gl − 10.60 |
| NoShade (gf on) | 8.54 | 120,220 | −4.91 | 1.1931·gl − 49.76 |
| NoShadeAmp (steep gf) | 6.53 | 117,534 | −1.64 | 1.6557·gl − 162.58 |

Reads:

1. **The composite path is solved.** The scene with no gradient opacity —
   the one that isolates the front-to-back composite — matches GL to within
   2 masked px. The interior dimming (fp16 stall), the border (0.001h gate),
   the early-termination (0.99h threshold), and the fine-step truncation
   (8192 cap) were all accumulation-pipeline artifacts and are all fixed.
2. **What remains is the gradient-opacity path.** Every variant that feeds
   `sampleOpacity *= gf(gradW)` still diverges; the larger the gf response to
   `gradW`, the larger the divergence (3.69 → 8.54 → 6.53 in this family),
   and all of it is Metal-dimmer (negative d mean) with the slope>1/negative-
   intercept contrast-stretch signature. This matches findings §3/§4 exactly:
   the gf LUT (~256 entries over `[0, 0.25·range]`, step ≈ 4.2 data units)
   turns a few-percent `gradW` difference near its steep knee into a full LUT
   level — a large opacity swing per sample.
3. **The underlying `gradW` difference is still unmeasured between backends.**
   The offline check (findings §4) validated Metal's gradient *formula* at
   Metal's own sample positions (median replay ratio ≈ 1.0) but could not
   compare the two backends' sample/gradient-stencil *positions*. The observed
   signature (Metal slightly dimmer everywhere, small constant slope offset)
   is consistent with a small per-sample position/phase offset feeding the
   gradient stencil, amplified by the gf LUT — but it is no longer the
   one-step-entry-offset hypothesis of Update1 §5, since the no-gf scene now
   matches to 2 pixels and would show that offset too. The open question is
   specifically: where in the `gradW`/gf chain (sample position, gradient
   stencil, LUT quantization, gf table differences) do the two backends
   differ?

## 6. Current state

Commits on `metal-ios` (pushed to the fork `origin`):

| commit | change |
|---|---|
| `c022b1f24a` | composite gate `sampleOpacity > 0.001h` → `> 0.0h`; termination `>= 0.99h` → `>= 1.0h − 1.0h/255.0h` (Update1) |
| `38343bdb4b` | composite accumulators + weight + termination threshold `half` → `float` (§1) |
| `e66a38ff60` | remove `MAX_RAY_STEPS` clamp; `maxSteps` from ray geometry alone (§2) |

`debugMarchGate`/SAMPLE/FINAL log extensions remain behind
`VTK_METAL_ENABLE_LOGGING`. Working tree clean.

Artifacts (regenerable via the procedures doc section 2 / fixed-step sweep):

- `/tmp/bc/fix3/` — camera-outside fixed-step sweep after the float fix.
- `/tmp/bc/fix3amp/` — NoShadeAmp step sweep after the float fix (cap still on).
- `/tmp/bc/fixcap/` — NoShadeAmp + camera-outside sweeps after the cap removal
  (= current state).

## 7. Next steps

1. Measure the `gradW`/gf input difference directly between the backends:
   log GL-side per-sample `gradW` (or both backends' gradient-stencil sample
   positions) for the same pixel and compare against Metal's logged GRADOP
   lines.
2. If a position/phase offset is confirmed in the gradient stencil, align it
   and re-measure — the NoShade/NoShadeAmp numbers should collapse the way the
   composite scenes did.
3. Re-verify the four kept parity changes against the full variant table
   (procedures doc *Image tests*) and strip the TEMP DEBUG gate, then run the
   GL deterministic re-capture check.
