# Accumulation chains are bit-exact on both backends; the 183-px residual is fully explained by per-sample nearest-texel selection flips at texel-boundary evalPoints (update 73)

**Date:** 2026-08-10
**Status:** **milestone (root-cause reduction).** Replayed each backend's own
per-sample debug data (op + premultiplied color) through the sequential float32
composite formula in Python and reproduced **both** finals to the printed digit.
The accumulation chains are bit-exact on both sides; **every remaining channel
delta maps to a per-sample op/raw difference**, and the worst pixel's divergence
is a single i=132 nearest-texel selection flip at the texel boundary z ≈ 178/256.

## 1. The replay (self-consistency proof)

For the reference test's worst pixel (Metal top-left (397,110) ↔ GL
(397,401), the (0,-7,-8) pair from update 72 §2), each backend's debug log
dumps per-sample `op` and premultiplied `color`:

```
GL_SAMPLE ... i=.. raw=.. color=(cr, cg, cb) op=..          (VTK_GL_RAY_DUMP=1)
DEBUG SAMPLE ... i=.. raw=.. op=.. rgb=(r, g, b) w=.. ..    (Metal VTK_METAL_VOLUME_LOG)
```

Replaying GL's rows with GL's sequential composite (`w = 1 - accA`;
`accC = w*col*op + accC`; `accA = w*op + accA`, each step in float32) yields:

| source | replay result | backend final |
|---|---|---|
| GL per-sample rows | accC=(0.937398, 0.700823, 0.559768) accA=0.997589 | (0.937398, 0.700823, 0.559768, 0.997589) — **exact** |
| Metal per-sample rows | accC=(0.937106, 0.727694, 0.592101) accA=0.997105 | (0.937106, 0.727694, 0.592101, 0.997105) — **exact** |

Both reproduce their own final exactly, so the sequential compositing (the
`w*src+acc` float32 chain) is not a divergence source and the per-sample logs
faithfully capture the per-sample state. **The residual must be entirely in the
per-sample (op, raw, color) values.**

## 2. Per-sample diff at the worst pixel: exactly one material divergence

Comparing GL's last frame vs Metal's last frame (i = 0..179), the ops match to
≤ 4e-7 (raw to ≤ 1e-6, the log print precision) at every index **except one**:

```
 i   GL raw      M raw       GL op      M op
131  0.0163272   0.0163270   0.127265   0.127265
132  0.0163272   0.0174870   0.127265   0.371981   <- the only |Δop|>1e-6 sample
133  0.0174868   0.0174870   0.371981   0.371981
```

Both backends log the **same** evalPoint to 6 decimals at i=132:
pos/eval z = **0.695312** = 178/256 = 0.6953125 — exactly a texel boundary.
The test uses default **nearest** interpolation (vtkVolumeProperty default;
see update 65 §1: `...NoTransformLinear.cxx` sets linear explicitly "default is
nearest", and `fc_linearInterpolation` is false → `sampleVolumeScalar` →
`sNearest`). At a nearest-texel boundary the raw is decided by which side of
178/256 the evalPoint falls:

- GL i=132 → texel 177 → raw 0.0163272, op 0.127265 (matches its own i=131)
- Metal i=132 → texel 178 → raw 0.0174870, op 0.371981 (matches its own i=133)

So Metal's evalPoint.z is a hair ≥ 178/256, GL's a hair below — the two
evalSteps (update 71 §2.4: Metal evalStep vs GL g_dirStep within 1-10 ulp)
have drifted by ≤ 1e-7 over 132 steps, and the boundary straddle flips the
texel selection. One flipped sample (op 0.127 vs 0.372, color (1, 0.726, 0.572)
vs (1, 0.983, 0.879)) fully accounts for the pixel's ΔG = −7, ΔB = −8.

## 3. Consequences

1. **GL is self-consistent end-to-end.** The VTK_GL_RAY_DUMP per-sample dump
   reproduces the debug-GL final exactly (this does not contradict update 72 §2:
   the *image* shifts ±1 LSB under the injected debug GLSL, but the per-sample
   rows still reconstruct the same debug run's final — i.e. the dump is a valid
   sample of the arithmetic it ran).
2. **Metal is also self-consistent**; the two backends only disagree in the
   per-sample scalar/opacity they feed identical accumulation.
3. **The residual is the nearest-texel-boundary sensitivity of `evalPoint`.**
   Since evalPoint is accumulated as `evalPoint + evalStep` per sample, a
   sub-ulp-per-step `evalStep` difference (the 1–10 ulp residual of update 71
   §2.4, traced to round-off amplification through
   `normalize → adjustedLin * dirObj → * sampleDistanceWorld`) accumulates to a
   ~1e-7 shift at i≈132 that straddles the nearest threshold.
4. **This unifies the reference test's 183-px field** (update 59-66 attribute-
   interpolator floor + the 14 knife-edge |Δ|≥2 px, worst at (397,110)) as a
   single mechanism: evalPoint's few-ulp drift flipping nearest-texel picks at
   exactly-half-voxel crossings, plus the ±1-LSB interpolated-scalar family
   (update 66) at the same sub-ulp positions.

## 4. Next target (next milestone)

Make `evalStep` **bit-identical** to GL's `g_dirStep`:

- GL: `g_dirStep = (ip_inverseTextureDataAdjusted * normalize(vertexPos - eyePos)).xyz * in_sampleDistance`,
  `ip_inverseTextureDataAdjusted = in_cellToPoint * in_inverseTextureDatasetMatrix`
  (vtkVolumeShaderComposer), normalize done in dataset space.
- Metal (`marchVolumeUnified`, MetalShaders.metal:4009-4075):
  `adjustedLin = rows(volumeToTexture) * ctpScale` (folded 3x3, comment claims
  GL parity), `dirObj = normalize(farP - nearP)`,
  `evalStep = (adjustedLin * dirObj) * sampleDistanceWorld`.

### 4.1 Status check against the prior findings (read 2026-08-10)

Two of the three proposed verification steps are **already answered**, so the
remaining budget should go entirely to step (b):

- **(a) `adjustedLin` vs `ip_inverseTextureDataAdjusted` — DONE, bit-identical**
  (update 17: `adjustedLin` == GL `invTexDataset*cellToPoint` to 3.1e-12 at
  `%.9e`). No folding-order ulp to fix in the matrix itself.
- **(c) `sampleDistanceWorld` vs `in_sampleDistance` — DONE, bit-identical**
  (update 17: equal to 1e-14 relative). Not a divergence source.
- **(b) the normalize input — THE live thread.** Update 71 §2.4 already
  quantified it: `nearP` is bit-identical at every physical pixel, but
  `farP.x` differs by **1 ulp** (Metal 90.3353195 vs GL 90.3353271),
  `dirObj` vs GL `rd` ≈ 1e-7 (few ulp), and `evalStep.x` ≈ 10 ulp — consistent
  with round-off amplification of that single ulp through
  `normalize → adjustedLin * dirObj` (update-70 §4c). So "make evalStep
  byte-identical" reduces to: **match Metal's `farP` float32 chain to GL's**.

Also relevant for the fix decision: the structural `evalStep` = GL `g_dirStep`
chain is already in place (update 13 §3), the march loop already advances on
`evalStep`/`g_dirStep` (update 69), and the near-plane ray-ORIGIN difference
(Metal fullscreen `entryPoint` vs GL proxy `vpos`, ~4e-7 vol units) was
explored via the proxy path in updates 16-20 — proxy-Metal matched GL's
pre-flip geometry byte-for-byte (update 18), and update 20 attributed GL's
larger response to near-eye anchor-geometry amplification of the harness W2IF
view-angle perturbation. That origin difference is below the farP.nearP ulp
level of update 71 and is a **separate** mechanism from the 1-ulp `farP.x`
here.

### 4.2 Concrete next experiments

1. **Dump full float32 `farP`/`dirObj`/`evalStep`** on both backends at the
   i=132 pixel of the worst pair (Metal (397,110) ↔ GL (397,401)) at `%.9e`,
   and diff the exact chains: GL `farP` via the fragment-shader inversePVM
   t-lambda/near-far interpolation vs Metal `unprojectPoint`
   (MetalShaders.metal:4082-4095). If Metal's `farP` uses a different float32
   chain than GL's `inversePVM`, reorder Metal to match; if they are already
   chain-identical, the 1 ulp is a uniform/vertex-input difference upstream.
2. Re-run the update-73 replay after the fix: with per-sample evalPoints
   identical, i=132 (and every texel-boundary sample in the 183-px field)
   should pick the same texel and the reference test should collapse.
3. If the farP chain cannot be matched without invading GL's proxy-mesh path,
   re-open the proxy-path routing decision from updates 16-20 §6 as the
   alternative production route.

## 5. Artifacts / scripts

- Python replay + per-sample diff used here:
  `/tmp/bc/glreplay2.py`, `/tmp/bc/finddiff.py`, `/tmp/bc/knife.py`,
  `/tmp/bc/precise.py`, `/tmp/bc/zeroop.py` (ephemeral; key results in §1-§2).
- Logs: `/tmp/bc/u73_gl.log` (VTK_GL_RAY_DUMP=1, GL px=(397,401)) and
  `/tmp/bc/u73_metal.log` (Metal DEBUG SAMPLE px=(397,110)).
- Capture commands identical to update 72 §3 (`RenderingBackend=OpenGL` for GL,
  `RenderingBackend=Metal` for Metal; verify backend via log prefix).
