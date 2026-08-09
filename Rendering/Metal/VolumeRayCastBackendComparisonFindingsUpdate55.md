# Metal-side bisect, step 6: uniform-d validated by flip-rate statistics; Metal+d collapses the field 63,692 → 6,812 (d varies ~±3%); clean-GL float-FBO readback returns RGB=0 (unresolved) (update 55)

**Date:** 2026-08-09
**Scope:** Resolving update 54 §5 (24.3% vs 9.8%), the decisive Metal+d experiment, and the first attempt to measure clean GL's true final float via an RGBA32F FBO readback.
**Target (unchanged):** Metal bit-identical to **clean GL** (`RenderingBackend=OpenGL`, no debug injection).

**Follows:** [Update 54](VolumeRayCastBackendComparisonFindingsUpdate54.md), [53](VolumeRayCastBackendComparisonFindingsUpdate53.md), [48](VolumeRayCastBackendComparisonFindingsUpdate48.md).

---

## 1. Update 54 §5 discrepancy RESOLVED — d IS uniform field-wide (statistically)

- Per-channel flip rate by value band (fix_gl vs fix_metal):
  - ch2 v[60,120): **9.89%**, ch1 v[120,180): **9.37%**, ch0 v[180,240): **9.28%**, ch0 v[240,256): **7.67%**, ch1 v[180,240): **7.82%**, ch2 v[120,180): **8.86%**.
- Under uniform fractional parts, per-channel flip rate ≈ d. The field's per-channel rates ≈ 8–10% ≈ **d = 0.0984 u8**. The "24.3%" in update 54 was the **pixel-level** rate (any of 3 channels): 1−(1−p)³ with p≈0.094 ≈ 24.3%. **No non-uniformity needed.**
- The 43 GL<Metal pixels and 13 |d|>1 pixels (e.g. (397,110) G −7/B −8, (360,229) G +3/B +4) are **separate knife-edge outliers** (grid-aligned-ray amplification, update 48 mechanism), not part of the uniform-d population.

## 2. Decisive experiment: Metal + uniform d collapses the field 63,692 → 6,812

- Added env-gated bias to Metal's finalize: `finalColor.rgb += VTK_METAL_CLEAN_BIAS_F` (MetalShaders.metal:4921), macro injected by `vtkMetalVolumeCompileOptions()` from `VTK_METAL_CLEAN_BIAS_F` (vtkMetalGPUVolumeRayCastMapper.mm). Applied post window/level, pre store.
- Run with `VTK_METAL_CLEAN_BIAS_F=0.0003857287` (d = 0.098361 u8 midpoint of update-54 band): `u55_metal_bias.png`.
- **Field diff vs clean GL: 63,692 → 6,812 (89.3% collapse).** Residual: 479 GL>biased (+1, still under) + 6,338 GL<biased (−1, overshoot).
- Interpretation: d is *approximately* uniform but **not exactly**: the gated-pixel band midpoint overshoots on 2.4% of field pixels (their true d < 0.098361), while 479 pixels need d > 0.098361 (up to the 0.1012 edge). So d varies over roughly the update-54 band [0.0955, 0.1012) across the field — a ±3% spread around ~0.098 u8.
- Overshoots (2.4%) far exceed the ≈0.3% expected if d were the constant 0.098361 everywhere ⇒ per-pixel d is a distribution, not a single constant.

## 3. Clean-GL float readback: FBO works, RGB reads back ZERO (unresolved)

- Added `DumpCleanGLFloats` (vtkOpenGLGPUVolumeRayCastMapper.cxx, env `VTK_GL_FLOAT_DUMP`, out `VTK_GL_FLOAT_DUMP_OUT`, default `/tmp/bc/gl_float_dump.raw`): re-renders the volume via `RenderVolumeGeometry` into an RGBA32F FBO (depth renderbuffer, blend off after dst=black, viewport restored, one-shot per process). **Deliberately independent of `DebugRayDump`** so the shader is the clean (non-injected) variant.
- Result: FBO `GL_FRAMEBUFFER_COMPLETE`, no GL errors, **alpha reads back correctly (0.996922791 at (422,419))** but **RGB reads back 0** for both `glReadPixels(GL_RGBA, GL_FLOAT)` and `glGetTexImage`.
- The render clearly targets the FBO (alpha is the accumulated opacity, not the 0.0 clear) and the composite clearly ran (opacity 0.997 only accrues from volume samples) — yet the RGB channels come back zero.
- **Unresolved.** Candidate causes (in suspicion order):
  1. The volume shader's output declaration (`//VTK::Output::Dec` substitution; the shader writes `gl_FragData[0] = g_fragColor`) interacting with the float FBO on Apple's GL (GLSL 1.50 + RGBA32F draw target) — possibly a driver quirk that only honors the alpha channel.
  2. The setup-time `glBindTexture(GL_TEXTURE_2D, tex)` on texture unit 0 overriding the volume texture *if* the volume is on unit 0 — but then opacity would also be 0, which contradicts the 0.997 alpha, so this is unlikely unless the opacity/color tables live on other units.
  3. The named-output mapping (`glBindFragDataLocation`) not covering attachment 0 for the float target.

## 4. Conclusion

- d ≈ 0.0984 u8 is field-uniform at the 10% level (per-channel flip-rate statistics), confirmed as the dominant ±1 mechanism; **Metal + uniform d collapses the field 89%** (to the 43 knife-edge + 13 big-diff + the ~2.4% overshoot/undershoot set).
- d varies within ~[0.0955, 0.1012) u8 across the field — a ~±3% spread — so a single constant cannot reach byte-parity on all 63,692 pixels.
- Clean GL's true final float is still unmeasured (the float-FBO RGB readback is broken); the exact d distribution and its value-dependence remain open.

## 5. Doubts / hypotheses (unresolved)

1. **Why does the RGBA32F FBO read RGB=0 with correct alpha?** Needs either (a) binding the volume texture to a non-0 unit before sampling, (b) a second-color-attachment/`glBindFragDataLocation` test, or (c) an RGB16F/RGBA16F target to see if the quirk is 32F-specific. Until solved, clean GL's true floats are unavailable.
2. **The exact d distribution.** Is d truly per-pixel-varying (⇒ no single-constant fix; a per-pixel correction would need the mechanism), or bimodal (e.g. one value inside the volume, another at edges)? The 479 undershoots + 6,338 overshoots at d=0.098361 need mapping to values/locations once clean floats are readable.
3. **Is the ~8%→10% flip-rate value-dependence (lower at saturation) real or a frac-quantization artifact?** Settled by the float readback.
4. **Source of d** (unchanged from update 54 §7): clean GL's compiled loop input drift vs a constant — the float readback was meant to measure it exactly.

## Artifacts

- Code: `Rendering/Metal/Shaders/MetalShaders.metal` (VTK_METAL_CLEAN_BIAS_F block at finalize), `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm` (compile-options macro), `Rendering/VolumeOpenGL2/vtkOpenGLGPUVolumeRayCastMapper.cxx` (DumpCleanGLFloats + VTK_GL_FLOAT_DUMP, diagnostics).
- Data: `/tmp/bc/u55_metal_bias.png` (Metal + 0.098361 u8, 6,812 px vs clean GL), `/tmp/bc/u55_gl_float.raw` (RGBA32F readback, RGB=0), `/tmp/bc/u55_gl_float_stdout.log`, `/tmp/bc/u55_metal_bias_stdout.log`.
- Baseline images: `/tmp/bc/fix_gl.png`, `/tmp/bc/fix_metal.png` (63,692 px, authoritative).
