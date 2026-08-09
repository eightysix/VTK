# Metal-side bisect, step 5: GL's true final float == Metal's model float; clean GL = Metal + uniform 3.857e-4 (confirmed on all 204 gated channels) — the ±1 field is a uniform additive float excess, mechanism unknown (update 54)

**Date:** 2026-08-09
**Scope:** New `VTK_GL_FINAL_DUMP` capability (true float32 capture of GL's final `g_fragColor` pre- and post-finalize via shader channels 60-67), first measurements on the 15 gated pixels, and the uniform-excess test that closes the store/arithmetic leads and pins clean GL to **Metal's float + 3.857e-4**.
**Target (unchanged):** Metal bit-identical to **clean GL** (`RenderingBackend=OpenGL`, no debug injection).

**Follows:** [Update 53](VolumeRayCastBackendComparisonFindingsUpdate53.md), [52](VolumeRayCastBackendComparisonFindingsUpdate52.md), [51](VolumeRayCastBackendComparisonFindingsUpdate51.md).

---

## 1. New capability: `VTK_GL_FINAL_DUMP`

- Extended the debug injection in `vtkOpenGLGPUVolumeRayCastMapper.cxx`:
  - Shader: after `castRay(...)` save `g_dbgPreFinal = g_fragColor`, then `finalizeRayCast()`, then channels **60-67** encode true float32 `g_fragColor`: 60-63 = pre-finalize rgba, 64-67 = post-finalize rgba (the exact value stored, after scale/bias).
  - Pre-loop dump block gated to `in_debugChannel < 60` so the new codes do not short-circuit.
  - Mapper: `VTK_GL_FINAL_DUMP=1` + `VTK_GL_SAMPLE_DUMP_PX=gx,gy` re-renders once per channel, reads back the float bits, prints `DEBUG GL_FINAL px=(...) c=N v=...`. Requires `VTK_GL_RAY_DUMP=1` too (the injection block is gated on `DebugRayDump`).

## 2. GL's true final float == Metal's model float, exactly

- Run: `VTK_GL_RAY_DUMP=1 VTK_GL_FINAL_DUMP=1 VTK_GL_SAMPLE_DUMP_PX=93,310 ...` (GL px (93,310) == Metal px (93,201)).
- **GL final float = (0.968990088, 0.668587208, 0.511607766, 0.996370196)** — pre- and post-finalize **identical** (runtime confirmation that `in_scale=1.0`, `in_bias=0.0` are identity; the finalize scale/bias lead is now closed at runtime, not just by inspection).
- This is **exactly** the update-48/53 CPU-model value for Metal at (93,201). `0.668587208*255 = 170.4897` → round-half-even → **170 = Metal's byte**, not clean GL's 171.
- Clean GL stores 171, which requires its float ≥ 0.668627451 — **+4.0e-5 above GL's own dump**. Since 170.49 needs +0.51 u8, no rounding mode explains 171. **The store conversion is definitively exonerated** (shared hardware also produced Metal's byte for the same float in the injected render).

## 3. Injected GL is a third compile variant (confirmed at image level)

- Debug-injected GL (`VTK_GL_RAY_DUMP=1` only) stored image:
  - vs **Metal**: **byte-identical on all 15 gated pixels** (matches Metal, not clean GL), but 683 px differ elsewhere (up=152, down=747 — mixed direction).
  - vs **clean GL**: 70,334 px differ.
- So the injection changes clean GL's compiled arithmetic enough to (at the 15 gated pixels) track Metal exactly, but is a distinct variant elsewhere. Clean GL's divergence is in its **compiled loop only**; the fragment main is textually identical (`initializeRayCast(); castRay(-1.0,-1.0); finalizeRayCast();`).

## 4. Uniform-excess hypothesis CONFIRMED on all 204 gated channels

- `uniform_excess_test.py` (update53/): for the 68 logged gated pixels, model float `accC` (68/68 metal-match, and == the true Metal/injected-GL floats per §2), test `round_half_even(accC*255 + d)` against clean GL's byte per channel:
  - **20 flipped channels** (GL byte = Metal byte + 1) require `d ≥ (m+0.5) − v*255`.
  - **184 unchanged channels** require `d < (m+0.5) − v*255`.
  - **Feasible interval [0.095527, 0.101195) u8 — non-empty.** Midpoint `d = 0.098361 u8 = 3.857287e-4` float reproduces clean GL on **all 204 channels**.
- Additive, not multiplicative: required excess per channel shows no correlation with channel value ((0,256) G needs 9× (93,201) G at a slightly lower value).
- Gated-channel flip rate 20/204 = 9.8% ≈ d, confirming the gated frac distribution is uniform and d ≈ 0.0984 u8.
- Magnitude: 3.857e-4 ≈ 6,400 ulps at 0.67 — **impossible for arithmetic reassociation** (all CPU fma/muladd/mixed variants are byte-identical, update 53 §3). The excess is an **input/constant difference** in clean GL's compiled loop.

## 5. Field flip-rate reconciliation (open)

- Whole field: 63,692/262,144 px differ (24.3%), always GL ≥ Metal by exactly +1 LSB. Under a uniform `d=0.0984` u8 and uniform fractional parts, only ~9.8% would flip. Observed 24.3% ⇒ either the field's frac(`v*255`) distribution is non-uniform (mass clustered near the .5 boundary) or `d` varies across the field. The 68 gated pixels are NOT a representative sample (they are the shader-logged pixels).

## 6. Conclusion

- Closed: store conversion (runtime float == Metal + shared hardware → exonerated), finalize scale/bias (runtime identity), composite operation order (unchanged), opacity-break strictness and extra-sample (update 53).
- New fact: **clean GL's final float is Metal's + a uniform additive 3.857e-4** over all 204 tested channels; the ±1 field is this excess passing through round-half-even.

## 7. Doubts / hypotheses (unresolved)

1. **Source of the +3.857e-4 in clean GL's compiled loop.** Candidates, in order of suspicion:
   - A per-sample input drift in clean GL's compiled shader (lattice position, scalar normalization, or TF-table lookup coordinate) of ~3.7e-6/sample over ~105 samples summing systematically to 3.86e-4.
   - A small TF-table index/coordinate offset in clean GL's compiled `computeOpacity`/`computeColor` (injected GL samples the same tables and matches Metal to the digit, so this would be a compile-level difference).
   - A subtle constant (e.g. 0.1 u8 ≈ 3.92e-4; note the interval contains the clean value 0.1 u8 = 1/2550) in the compiled loop path.
2. **Is d field-wide constant?** The 24.3% vs 9.8% flip-rate discrepancy is unresolved; needs a full-field test (lattice interpolation across pixels, or a direct clean-GL float readback via a float-format FBO/readback) to confirm d is the same everywhere.
3. **Fix candidates (not yet decided):** if d is uniform field-wide, adding the equivalent +d to Metal's finalize (or removing it from GL) achieves byte-parity — but the honest fix requires finding where d originates.

## Artifacts

- Code: `vtkOpenGLGPUVolumeRayCastMapper.cxx` — shader channels 60-67 + `VTK_GL_FINAL_DUMP` block; `in_debugChannel < 60` gate.
- Scripts: `Rendering/Metal/BackendComparisonTools/update53/uniform_excess_test.py`, `field_flip_rate_test.py`.
- Images: `/tmp/bc/dbg_gl.png` (injected GL), `/tmp/bc/fix_metal.png`, `/tmp/bc/fix_gl.png` (63,692 px).
- Final-float dump values for all 15 gated pixels captured in-session (§2 pattern).
