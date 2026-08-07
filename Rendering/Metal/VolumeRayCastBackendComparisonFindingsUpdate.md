# Metal vs OpenGL volume ray cast: composite-path probe campaign (update)

Follow-up to the two existing documents, read as an addendum to both:

- [VolumeRayCastBackendComparisonProcedures.md](VolumeRayCastBackendComparisonProcedures.md) —
  environment, capture/analyze tooling (`analyze.py`, dummy-baseline captures,
  `GL_*` engagement checks, fixed-step and per-sample-logging machinery).
- [VolumeRayCastBackendComparisonFindings.md](VolumeRayCastBackendComparisonFindings.md) —
  sections 1–7, all still valid.

This document records the probe campaign against the two open hypotheses of the
findings doc (opacity step-compensation asymmetry; sample phase offset), the two
confirmed shader-level divergences that were found and changed in
`Rendering/Metal/Shaders/MetalShaders.metal`, and the updated backend-vs-backend
numbers measured after those changes.

Measurement scene (identical camera pose, both backends, genuinely
GL-engaged — verified via the `GL_SAMPLING`/`GL_OPTABLE`/`GL_TEX` stderr logs,
25 lines on the GL run, none on the Metal run):
`TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideFixedStep`
(camera outside, no transform, no shading, no gradient opacity, fixed step forced
via `VTK_FIXED_SAMPLE_DISTANCE`). `AutoAdjustSampleDistancesOff` confirmed by the
`PROBE fixedSD=… autoAdjust=0` stderr line.

All figures are Metal minus OpenGL per-channel deltas at 512×512. `mean|Δ|` is
the per-pixel max-channel absolute delta; "masked px" counts pixels with
`|Δ| ≥ 5/255`.

---

## 1. Probes that ruled hypotheses in/out (all reverted)

### 1.1 `MAX_RAY_STEPS` 8192 → 65536 — inert

`constant int MAX_RAY_STEPS` (`MetalShaders.metal:2944`). The center
camera-outside ray at step 0.0675 needs ~1100 samples — far below either cap, so
the cap is never hit. Rebuild + re-capture at 0.0675: no change. Reverted
(8192 restored).

### 1.2 Half → float composite accumulation — byte-identical, inert

Changed `accumulatedColor` / `accumulatedOpacity` and the weight arithmetic from
`half` to `float` in both composite paths. Re-capture at camera-outside 0.0675:
**byte-identical output to the half build (0 masked-px diff)**. Half-precision
accumulation is not the cause of the divergence. Reverted.

### 1.3 Scalar-normalization float probe (camera-inside scene) — minor, inert

`scalarScale`/`scalarBias`/`scalarNorm` promoted to `float` on the
camera-inside `NoShadeNoGradOpNoTransform` scene: masked px 1,444 → 1,271
(~12%), no structural change. Not the driver. Reverted.

## 2. Confirmed divergence #1: Metal early-termination threshold ≠ GL

- **GL:** `g_opacityThreshold = 1.0 - 1.0/255.0` ≈ 0.99608
  (`vtkVolumeShaderComposer.h:3224`); composite loop breaks when
  `g_fragColor.a > g_opacityThreshold` (line 3332).
- **Metal:** `accumulatedOpacity >= 0.99h` (`MetalShaders.metal:4440`) — a lower
  bar that stops compositing ~1.6/255 of opacity early.
- **Change:** Metal now terminates at `1.0h - 1.0h/255.0h` (GL parity).

Effect (camera-outside 0.0675): the center pixel became exact — GL and Metal
both render **[254, 175, 133]**. Overall masked px rose 111,294 → 121,297
(more leading samples now accumulate, consistent with GL's higher bar). This is
correct parity, but it is not the main divergence.

## 3. Confirmed divergence #2 (root cause of the border band and much of the ring): the `0.001h` composite opacity gate

- **Metal:** both composite paths gated compositing on `sampleOpacity > 0.001h`
  (`MetalShaders.metal:4237` independent path, `4294` main path).
- **GL:** composites every sample with positive alpha (`g_srcColor.a > 0.0`,
  `vtkVolumeShaderComposer.h` composite block). **There is no 0.001 threshold.**

Mechanism (from the per-sample log at border pixel (0,256),
`/tmp/bc/camout_ring.log`, 2119 samples): each sample raw ≈ 0.0015
(scalar ≈ 103, numpy-replayed exactly against `vol512.npy`), opacity ≈ 0.0003 —
**every sample below the gate** — so Metal's `FINAL accumulatedOpacity =
0.000000` and it renders the flat background **[26,26,26]** while GL composites
all of them into a faint warm ramp **[42,28,22]**. The gate silently drops the
entire leading low-opacity ramp that GL counts.

**Change:** gate relaxed to `sampleOpacity > 0.0h` in both paths (lines 4237,
4294).

Effect (camera-outside 0.0675, GL vs Metal):

| metric | threshold-only | after gate fix |
|---|---|---|
| masked px | 121,297 | **47,578** (−61%) |
| mean\|Δ\| | 8.11 | **2.37** |
| max\|Δ\| | 66 | **34** |
| border px (0,256) | [26,26,26] | **[41,27,21] vs GL [42,28,22]** |
| ring px (113,45) | Metal [244,149,103] vs GL [198,120,84] (d=46) | Metal [192,118,84] (d=6) |
| center (256,256) | exact | exact |

Replay confirmation: compositing Metal's *own logged* (0,256) samples without
the gate yields **[28,14,8]** vs gate-on **[0,0,0]**; GL renders [42,28,22].
Direction and magnitude confirmed; the residual offset is sample phase
(section 5).

## 4. Updated backend-vs-backend numbers (after sections 2 + 3)

### 4.1 Camera-outside fixed step 0.0675 (isolation scene)

| metric | original (0.99h + 0.001h) | threshold-only | after gate fix |
|---|---|---|---|
| mean\|Δ\| | 8.00 | 8.11 | **2.37** |
| masked px | 111,294 | 121,297 | **47,578** |
| Δ mean | +2.41 | — | **−0.38** |

Per-channel fits after the fix: `metal = 1.0182·gl − 4.71` (R),
`1.0084·gl − 1.87` (G), `0.9997·gl + 0.27` (B) — slope ≈ 1 with tiny negative
offsets. The border band is essentially clean: 3,668 masked px in the outer
8-px band (mean|Δ| 2.78); the remaining 43,910 masked px are interior.

### 4.2 `NoShadeAmp` (default fixed step 0.008) — the highest-delta test

| metric | before (findings §5) | after |
|---|---|---|
| mean\|Δ\| | 45.83 | **33.43** (−27%) |
| masked px | 260,773 | **251,405** |
| max\|Δ\| | 168 | **153** |
| Δ mean | −6.6 | −15.67 |

Per-channel fits after the fix: `metal = 2.2583·gl − 330.19` (R);
center GL [247,200,166] vs Metal [251,209,174].

The two shader fixes are necessary but **not sufficient** for the amplifier
scene: the gradient-opacity LUT amplification and Metal's step-sensitivity
(findings doc §5) remain the dominant residual. GL was re-captured and is
unchanged (deterministic); the improvement is entirely Metal-side.

## 5. Remaining divergence and interpretation

At 0.0675 what is left is an interior ring (concentrated upper-left in the
16×16 block map), max |Δ| = 34 at px (32,346): GL [249,144,98] vs Metal
[215,129,87]. The opaque core and the border now match.

The remaining deltas at the ring are negative (Metal dimmer) with slope ≈ 1 and
small intercept — a **small sample-phase/position offset between the backends**,
not an opacity-scale mismatch. The phase hypothesis from the findings doc §6
still stands, and its predicted signature is exactly what is observed: Metal's
first sample sits one full step inside the entry face (`firstT = jitter =
stepSize`, `checkBounds = true`); GL's first sample is at the entry face
(`t = 0`, `g_dataPos = g_rayOrigin`). Fixing the 0.001h gate fixed the border
and much of the ring because both were caused by dropping/starting those
low-opacity leading samples; the residual is the one-step positional offset
itself.

## 6. Current working-tree state (uncommitted)

`Rendering/Metal/Shaders/MetalShaders.metal`:

| line | change | status |
|---|---|---|
| 4237 | `sampleOpacity > 0.001h` → `> 0.0h` (independent composite path) | keep (GL parity) |
| 4294 | `sampleOpacity > 0.001h` → `> 0.0h` (main composite path) | keep (GL parity) |
| 4440 | `>= 0.99h` → `>= 1.0h - 1.0h/255.0h` | keep (GL parity) |
| 3631–3638 | `debugMarchGate` `pxOkCamOut` (ring + border pixels) | keep — behind `VTK_METAL_ENABLE_LOGGING` (same as f1ec8e6/1ce3977); required by `make_camout_log.sh` |

Everything else from the probe campaign is reverted: `MAX_RAY_STEPS` back to
8192, half-precision accumulation restored.

Artifacts: `/tmp/bc/co_OpenGL.png`, `/tmp/bc/co_Metal.png` (camera-outside
0.0675), `/tmp/bc/amp_OpenGL.png`, `/tmp/bc/amp_Metal.png` (NoShadeAmp 0.008),
`/tmp/bc/camout_ring.log` (per-sample border+ring dump). All re-captures
regenerate via the procedures doc section 2.

New tools in `Rendering/Metal/BackendComparisonTools/` (persisted with this
update): `make_camout_log.sh` (camera-outside per-sample log), `replay_camout_log.py`
(gate A/B replay of a logged pixel), `image_delta_profile.py` (pixel/border/block
delta profile).

## 7. Next steps

1. Measure the phase offset directly: log GL-side entry/first-sample positions
   (or reuse the logged Metal entry vs the analytical ray-box intersection) and
   confirm it equals one step.
2. If confirmed, align Metal's first sample to the entry face (or GL's to the
   step-offset position) and re-measure — the interior ring should collapse the
   way the border did after the gate fix.
3. Re-verify the two kept parity changes against the full variant table
   (procedures doc *Image tests*) and, if they hold, strip the TEMP DEBUG gate
   and re-run the GL deterministic check.
