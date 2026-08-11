# Per-vertex inputs are bit-identical GL↔Metal on all three texcoord axes (94/94, three capture generations): update 79 §4.1's z-input lever is refuted, and the interpolator-model sweep is negative on current data (update 80)

**Date:** 2026-08-11
**Status:** **milestone (update-79 §4.1 lever executed, refuted).** The one
remaining input-side hypothesis — that the near-cap **z per-vertex texcoord**
is not bit-identical GL↔Metal and is the +1-ulp anchor source (update 79 §4,
update 76 §3) — is now tested on the frame-aligned W2IF capture set and is
**negative**: the per-vertex `texcoord` matches **94/94 on x, y AND z**, and
`clip.x/y/w` and `modelPos` also match 94/94. The z input is bit-identical, so
the ±1-ulp anchor-z offset is produced by the **rasterizer interpolator**, not
by any per-vertex input. A fresh interpolation-model sweep against the current
(three-capture-consistent) logs finds **no model reproduces either backend's
interpolated anchor**: best case 8/303 vs Metal's `interp` field (shader-exact
float32 chain at pixel-center barycentrics), 0/28 vs GL's frame-6 anchors.

## 1. Method

- GL per-vertex dump: `…/u78/nj_vtx_gl.log` (`GL_VERT` lines, VTK_GL_VERTEX_DUMP,
  update-76 §3 fixed channel: `ip_vertexPos.z`).
- Metal per-vertex dump: `Testing/Temporary/nj_anchor_rec4.txt` (`ANCHOR_REC`,
  `VTK_METAL_ANCHOR_REC_DUMP`), joined to GL on `vi`; 94 shared vertices.
- Metal fragment log: `Testing/Temporary/nj_anchor_rec{2,3,4}.log` — three
  independent capture generations (18:44/18:46/18:52), all with the **same**
  303-pixel set (28 overlapping the GL_RAY gate, `ANALYTIC_IN` lines carrying
  the hardware `interp` field + the 21-float per-primitive `rec` from
  `BuildTriangleAnchorBuffer`, MetalShaders.metal:5312).
- GL fragment log: `Testing/Temporary/nj_gl_raydump.log` (`GL_RAY`, 29 px × 6
  frames, `VTK_GL_RAY_DUMP`).
- Pairing: GL window (px,py) ↔ Metal PNG (px, 511-py); GL_RAY `states[-1]` is
  the final frame (all 29 gate pixels have exactly 6 states).

## 2. Per-vertex join (update-79 §4.1 lever)

| quantity | result |
|---|---|
| `clip.x/y/w` bit-match | **94/94** |
| `modelPos` bit-match | **94/94** |
| `texcoord` bit-match per axis | **(94, 94, 94)**, all-3 94/94 |
| `clip.z` | depth-only; near-cap readback is the update-76 §3 encoding artifact (u78 log); the clean newer GL dump matched `(z+w)/2` to ±1 ulp at 92/94 (vi=4,6) |

`clip.z` is not consumed by the raycast (no `gl_FragCoord.z`/`ip_vid` use in
`vtkVolumeShaderComposer.h`; the near/far entry is texcoord-driven), so it is
irrelevant to the image. The **z texcoord — the axis that drives the i=132
texel pick (update 74 §4) — is bit-identical 94/94.** Conclusion: update 79
§4.1 ("capture the near-cap z per-vertex texcoord GL↔Metal … confirm the z
input, not the interpolator") is refuted; the +1-ulp is interpolator-created.

## 3. Interpolator-model sweep (current, three-capture-consistent data)

Two representative models over the 303-pixel `rec`/`interp` set and the 28-pixel
GL gate overlap (off = pixel-center +0.5):

| model | vs Metal `interp` (303 px) | vs GL frame-6 (28 px) |
|---|---|---|
| f64 barycentric weights, f32 q/1w/Q/W/divide | 3/303 | 0/28 |
| shader-exact f32 chain (MetalShaders.metal mode-1 recipe) | 8/303 | 0/28 |

Nearby variants (fma/mul-add order, Q*rcp(W) with trunc/round/lowered
reciprocals, mantissa-quantized intermediates, result fixed-point grids) do not
exceed this. **No offline model hits 0 ulp on either backend**, consistent with
update 77's historical record (updates 61-63: "no pixel-center model hits 3/3
channels at 0 ulps"; sub-pixel offsets reduce but never reach 0).

> Correction for the record: an earlier in-session reading of a stale
> `nj_anchor_rec4.log` generation reported "f32-model == Metal interp 28/28".
> That does not reproduce on the current three-capture-consistent logs (8/303
> max) and is retracted.

## 4. Conclusion

- All per-vertex inputs that feed the interpolator (texcoord x/y/z, clip x/y/w,
  modelPos) are **bit-identical**; the ray anchor difference is produced inside
  the rasterizer interpolator.
- The interpolator arithmetic is **not reproducible** by any f32/f64
  barycentric/perspective model on either backend, and — per updates 78/79 —
  Metal's hardware interpolator already sits at the midpoint of GL's own
  frame-A/B anchor spread, so no anchor substitution can track GL's
  frame-dependent values (mode-1 429 px, mode-2 469 px vs 178 px baseline).
- The 178-px residual is therefore **bounded below by the GL-driver vs Metal
  interpolator arithmetic difference** at knife-edge nearest-texel picks, with
  all inputs proven bit-identical.

## 5. Remaining levers (no input-side mechanism remains)

1. The barycentric-inversion probe (update 76 §5 exp. 1) was **already run and
   closed**: update 61 (implied sample point == exact pixel center on both
   backends; the per-pixel fitted sample offset reduces error but never reaches
   0 and is backend-specific) and update 63 (backed-out effective weights
   internally consistent to ~1e-15 but displaced identically in both backends),
   with update 77 §1's verdict: "the rasterizers do not evaluate at a different,
   discoverable sample point that could be emulated. Lead 1 is closed." It is
   **not** a remaining lever.
2. The only genuinely-open lever is the full-gate interpolation-floor
   quantification (update 77 §5 item 3, update 76 §5 exp. 3): measure the
   per-axis ulp histogram across the full 8237-px STEP gate (not just the 28
   GL-gate pixels), frame-matched. If it confirms the floor, accept the 178-px
   residual as the hardware floor and switch the acceptance metric to the
   frame-aligned `VTK_STEP_RAW_CAPTURE` diff (update 79 §5), documenting the
   residual as irreducible rather than chasing bit-identity through the anchor.

## 6. Files

- Logs: `…/u78/nj_vtx_gl.log`; `Testing/Temporary/nj_anchor_rec{2,3,4}.{log,txt}`;
  `Testing/Temporary/nj_gl_raydump.log`.
- Instrumentation: `Rendering/Metal/Shaders/MetalShaders.metal:5312-5435`
  (`analyticAnchorTexcoord`, ANALYTIC_IN/OUT debug),
  `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm:6413-6530`
  (`BuildTriangleAnchorBuffer`), `…/u78/nj_vtx_gl.log` producer
  (`VTK_GL_VERTEX_DUMP`, OpenGL2 path).
- Prior: update 79 (frame-aligned 178-px same-camera), update 78 (analytic
  anchor A/B negative), update 76 §3 (z artifact), update 75 rev2 (anchor
  offset = entire i=132 mechanism), update 77 (interpolation-model dead ends,
  incl. update-61/63 barycentric-inversion probe closed).
