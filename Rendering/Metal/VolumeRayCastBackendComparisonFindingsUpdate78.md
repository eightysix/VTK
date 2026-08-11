# The analytic-anchor experiment is wrong-direction (mode-1 429 px, mode-2 469 px vs GL 178 px baseline): the float64 anchor matches GL frame-A exactly on x/y but is +1 ulp above on z, and GL's own interpolated tex is not frame-invariant (frame-A/B spread 1-2 ulp) (update 78)

**Date:** 2026-08-11
**Status:** **milestone (experiment conclusively negative).** `VTK_METAL_ANALYTIC_ANCHOR`
modes 1 (float32 weights) and 2 (float64 weights) both make the reference-test
image **worse** than the mode-0 baseline (update 74 = 178 px). The diff is
definitively active (mode-1 flips 275 px, mode-2 flips 297 px vs mode-0) but
points the wrong way (mode-1 429 px, mode-2 469 px vs clean GL). Root cause of
the negative result, established at the worst pixel Metal (397,110) ↔ GL
(397,401):

- GL's *interpolated texcoord is not frame-invariant*. Across its 6 frames the
  same physical pixel logs **two distinct tex values 1-2 ulp apart**
  (frame-A `(0x3f0184b9, 0x3f01aa39, 0x3ee5d6d4)` vs frame-B
  `(0x3f0184b7, 0x3f01aa36, 0x3ee5d6d0)`) — the update-19 W2IF 30° →
  30.0000008° frame-3→4 perturbation, now shown to perturb the *interpolated
  anchor itself*, not just the texel pick.
- Metal's **hardware** interpolator (mode-0) sits exactly at the **midpoint of
  the two GL frames** on all three axes
  (`(0x3f0184b8, 0x3f01aa37, 0x3ee5d6d2)`) — already the best single
  deterministic value, and the source of the update-76 "+1 ulp, Metal always
  at-or-above GL" (it is at-or-above only the *lower* GL frame).
- The mode-2 (float64) analytic anchor reproduces **frame-A exactly on x/y**
  (`0x3f0184b9`, `0x3f01aa39`) and is **+1 ulp above frame-A on z**
  (`0x3ee5d6d5`). z is the axis that drives the i=132 texel pick (update 74 §4),
  so rendering with it tips knife-edge pixels the wrong way.
- The mode-1 (float32) analytic anchor is +1..+2 ulp *above* the float64 value
  on x/y at the shared pixels (rounded up through the float32 weight chain),
  i.e. further above frame-A, and is also worse for the image (429 px).

Conclusion: the analytic-anchor reconstruction (both precisions) is the wrong
lever. The interpolation mechanism is *not* the residual cause — Metal's
hardware interpolator and the analytic reconstruction both agree with one of
GL's frames to ≤1 ulp — and the 178-px knife-edge field is decided at the
**z-lattice / frame-selection** level, where GL itself is not deterministic.

## 1. Image diffs (512x512 RGB, vs clean GL `dc4bab2e…`)

| state | vs clean GL | vs mode-0 Metal |
|---|---|---|
| mode-0 (current build, hardware interp) | **178 px**, max Δ 8 | — |
| mode-1 (`VTK_METAL_ANALYTIC_ANCHOR=1`, float32 weights) | **429 px**, max Δ 8 | 275 px |
| mode-2 (`VTK_METAL_ANALYTIC_ANCHOR=2`, float64 weights) | **469 px**, max Δ 8 | 297 px |

mode-0 of the *current* build reproduces update-74's 178-px baseline, so the
worsening is attributable entirely to the anchor replacement. The changed
pixels are scattered over the whole frame (no mesh-edge clustering), consistent
with small (≤3 ulp) anchor shifts tipping knife-edge texel picks.

## 2. Anchor values at the worst pixel (Metal (397,110) ↔ GL (397,401))

```
GL frame-A  tex = (0x3f0184b9, 0x3f01aa39, 0x3ee5d6d4)     (3 of 5 dump rows)
GL frame-B  tex = (0x3f0184b7, 0x3f01aa36, 0x3ee5d6d0)     (2 of 5 dump rows)
Metal interp   = (0x3f0184b8, 0x3f01aa37, 0x3ee5d6d2)      midpoint of A/B (x,y,z)
Metal mode-2   = (0x3f0184b9, 0x3f01aa39, 0x3ee5d6d5)      == frame-A x/y, A+1 z
Metal mode-1   = (0x3f0184ba, 0x3f01aa3a, 0x3ee5d6d6)      == frame-A+1 x/y, A+2 z
```

- The float64 reconstruction of update 76 §4 (which "reproduces GL exactly")
  reproduces **frame-A**; the earlier +2..+61-ulp readings were a comparison
  artifact (the dump stores multiple per-pixel rows and the "last row" picked
  frame-B on some pixels).
- On z, every Metal configuration is at-or-above frame-A, i.e. 1-5 ulp above
  frame-B — the wrong side for matching the texel-177 frames the compared GL
  image holds (update 74 §4).

## 3. Why both analytic modes lose

1. The analytic value is the *exact* perspective-correct interpolation of the
   (bit-identical, update 76 §3) per-vertex x/y texcoords. It can match frame-A
   but **cannot match frame-B**, because frame-B's tex is not exact — it is the
   perturbed-camera value 1-2 ulp lower. No deterministic per-fragment
   reconstruction of the anchor can track a frame-dependent quantity.
2. z is the knife-edge lever and the z per-vertex texcoord is the one axis that
   is *not* bit-identical GL↔Metal (update 76 §3: GL's near-cap z readback is an
   encoding artifact, Metal's modelPos.z is a genuine intersection). So the z
   interpolation can be off by 1 ulp regardless of mechanism, and that 1 ulp
   decides the residual pixels.
3. The float32 weight chain (mode-1) rounds *up* relative to float64 on x/y at
   the sampled pixels, so it is strictly further from frame-A there; its slightly
   better image score (429 vs 469) is knife-edge noise, not signal.

## 4. The residual is a frame-selection / z-lattice artifact, not interpolation

- Metal's hardware interpolator already equals the *midpoint* of GL's frame-A/B
  spread on all three axes — there is no interpolation defect left to fix.
- The remaining 178 px are the texel picks at the z-lattice boundary where GL
  flips between its own frames (update 74 §4: i=132 texel 177/178) and Metal
  deterministically lands on one side.
- The compared GL image holds a specific single frame; Metal's deterministic
  output can only match it if its anchor z rounds to the *same side* of the
  nearest-texel boundary, which is a frame-dependent property.

## 5. Next experiment: per-frame, frame-aligned image comparison

Update 74 §5.1 (frame alignment) is now the only open lever, and the frame-A/B
anchor spread measured here gives it a concrete target:

1. Capture GL **and** Metal per-frame images (frames 1-6; frames 1-3 unperturbed
   30.0°, frames 4-6 perturbed 30.0000008°, update 19) and diff Metal vs each GL
   frame. A single GL frame that is 0 px from Metal would prove the residual is
   pure frame-selection; the harness's `-V` capture picks one of the perturbed
   frames (update 19), so frame 4-6 is the expected match target.
2. If no single GL frame matches, measure the z-anchor side (frame-A vs frame-B
   tex at each of the 178 residual pixels) to bound how many are
   frame-inherently undecidable vs fixable by a deterministic z rounding rule.

## 6. Files / commands

- Anchor switch: `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm:7689-7706`
  (`VTK_METAL_ANALYTIC_ANCHOR`, `BuildTriangleAnchorBuffer`),
  `Rendering/Metal/Shaders/MetalShaders.metal:5312-5435`
  (`analyticAnchorTexcoord`, modes 1/2), fragment override at
  `MetalShaders.metal:5487-5497` (and 5602-5612, 5832-5842).
- Runs (update 72 §3): Metal with
  `VTK_METAL_ANALYTIC_ANCHOR={1,2} MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=33554432 MTL_LOG_TO_STDERR=1`
  + `RenderingBackend=Metal`; clean GL with `RenderingBackend=OpenGL`.
- Logs (this session): mode-2 `…/u78/mt_an2b.log` (1049 `ANALYTIC_OUT64` rows,
  265 unique px), mode-1 `…/u78/mt_an1.log`, GL dump `…/u78/gl_dbg.log` (145 rows,
  29 px × multiple frames; **multiple rows per pixel, distinct tex frames**).
- Images (this session): `…/u78/{mt_mode0,mt_an1,mt_an2,f_gl_clean}.png`:
  178 / 429 / 469 px vs clean GL.
- Key reproduced values (worst pixel): GL frames
  `(0x3f0184b9,0x3f01aa39,0x3ee5d6d4)` / `(0x3f0184b7,0x3f01aa36,0x3ee5d6d0)`;
  Metal interp `(0x3f0184b8,0x3f01aa37,0x3ee5d6d2)`; mode-2
  `(0x3f0184b9,0x3f01aa39,0x3ee5d6d5)`; mode-1
  `(0x3f0184ba,0x3f01aa3a,0x3ee5d6d6)`.
