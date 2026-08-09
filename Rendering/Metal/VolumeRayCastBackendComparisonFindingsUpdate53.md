# Metal-side bisect of clean GL's compile divergence, step 4: strictness and one-extra-sample are quantitatively DEAD; GL's own per-sample values composite to Metal's byte — the ±1 field is a terminal/store-level difference (update 53)

**Date:** 2026-08-09
**Scope:** Close update 52 §6.1 (opacity-break strictness / one-extra-tail-sample) numerically, re-test operation-order (update 52 §6.3) and the final scale/bias, then dump GL's **true float32 per-sample values** (`VTK_GL_SAMPLE_DUMP`, y-flipped pixel convention) for the gated pixel (93,201). Key result: **GL's own per-sample op/color/pos match Metal's to the digit, and compositing them with the fma chain reproduces Metal's byte (247,170,130), NOT GL's (247,171,131)** — a direct contradiction that removes per-sample values, operation order, break strictness, and extra samples as causes, and isolates the ±1 field to a terminal/store-level step still unaccounted for.
**Target (unchanged):** Metal output must be bit-identical to **clean GL** (`RenderingBackend=OpenGL` without debug injection).

**Follows:** [Update 52](VolumeRayCastBackendComparisonFindingsUpdate52.md), [Update 51](VolumeRayCastBackendComparisonFindingsUpdate51.md), [Update 50](VolumeRayCastBackendComparisonFindingsUpdate50.md).

---

## 1. Opacity-break strictness is quantitatively dead

- **Hypothesis (update 52 §6.1):** Metal breaks on `accA >= 1 - 1/255` (non-strict, MetalShaders.metal:4814), GL on `g_fragColor.a > g_opacityThreshold` (strict, composer:3371). If a ray's `accA` lands *exactly* on `0.9960784316`, GL continues one more sample.
- **Numerical check (all 1,938 logged FINAL rays, u47_metal.log):** threshold `f32(1-1/255) = 0.9960784316062927`. FINAL logged `accOp` (half-rounded print, so ±4.9e-4 fuzz): min `0.992977` (geometry-exit rays, e.g. (240,176)), max `1.0` (saturated). **0 rows within 1e-3 of the threshold, 0 exact.** Half-print fuzz cannot hide a true exact-tie hit across 1,938 rays combined with the prior ~3e-5/ray hit probability (~8 expected ties in 63,692 field rays) — and even a true tie would flip only ~8 pixels, not the 63,692-px one-directional field. **Hypothesis closed.**

## 2. One-extra-tail-sample is dead

- **Hypothesis (update 52 §6.1):** a single extra tail sample (the ~(0.0009,0.0008,0.0007) contribution at (93,201)) reproduces GL's byte.
- **Numerical check:** the update-48 replay forced one extra sample (no opacity break) on all 68 gated pixels. Result: **0/15** GL-diff gated pixels turn into GL's byte (all 15 stay Metal or neither); across all 68, the extra-sample variant equals GL's image on 53 pixels (coincidence — most pixels are unaffected by a sub-LSB tail sample). A one-sample-count difference cannot explain the field. **Hypothesis closed.**

## 3. Composite operation-order variants are byte-immaterial (re-confirmed)

- Tested on all 68 gated pixels: `fma(w, f32(op*rgb), accC)` / `f32(accC + f32(w*f32(op*rgb)))` / color-muladd+alpha-fma / color-fma+alpha-muladd. **All four give identical stored bytes** on every gated pixel (matches-Metal 68/68, breaks none of the 53 no-diff pixels). The GPU-contraction ordering question (update 52 §6.3) cannot be resolved at the byte level because no ordering crosses a u8 boundary here.

## 4. Final scale/bias and TF-sampling filter re-verified

- **finalizeRayCast scale/bias (composer/raycasterfs.glsl:327-329 vs MetalShaders.metal:4919-4921):** both compute `rgb * scale + bias * alpha` with `scale = f32(1/FinalColorWindow)` = 1.0 and `bias = f32(0.5 - level/window)` = 0.0 (defaults window=1, level=0.5) → identity on both backends. Closed.
- **TF texture filter:** Metal picks the sampler via the `fc_linearInterpolation` function constant (MetalShaders.metal:3209-3214), driven by the volume property's interpolation (default NEAREST → `sNearest`); GL's TF tables follow the same property-derived filter. For this test the floor-index NEAREST lookup model stands.

## 5. GL true float32 per-sample dump (new capability + pixel-convention caveat)

- The existing `VTK_GL_RAY_DUMP=1 VTK_GL_SAMPLE_DUMP=1 VTK_GL_SAMPLE_DUMP_PX=Gx,Gy` machinery (vtkOpenGLGPUVolumeRayCastMapper.cxx:4609-4669) re-renders once per sample index and reads back the **true float32** `raw`, `g_dataPos`, `g_srcColor.rgb`, `g_srcColor.a` byte-encoded in an RGBA8 attachment.
- **Convention caveat:** GL reads back in glReadPixels coords (bottom-left origin), Metal in top-left. The SAME physical pixel is GL `(Gx, Gy)` == Metal `(Gx, 511-Gy)` for a 512×512 viewport (compare_gl_metal_samples.py docstring). Earlier naive GL (422,92) vs Metal (422,92) comparisons were physically wrong pixels.
- **GL default framebuffer:** RGBA8 (colorSize 32, alphaSize 8) per the pixel-format descriptor printed by the GL run → GL's fragment float output is fixed-function converted float→u8 (round-half-even) before readback.

## 6. Central new contradiction: GL's own per-sample values composite to Metal's byte

- Metal gated (93,201) → GL dump pixel **(93, 310)**. Captured all 320 samples; last valid sample i=104 (`op=0.250333279`, matching Metal's tail op 0.250333).
- GL per-sample `pos`/`raw` match Metal's SAMPLE log (u47_metal.log) to 6 decimals everywhere; `op` matches the CPU float32 table; dumped `color` is **premultiplied** (`f32(op*rgb)`, verified per sample; e.g. i=7 colR=2.274224e-4 = op 0.001109 × rgb 0.205044).
- Compositing GL's OWN dumped `op`/`color` with the fma chain (or any operation order, update 51 §2 / §3 above):
  `accC = (0.968990, 0.668587, 0.511608) → u8 (247, 170, 130)` — **exactly Metal's byte.**
  GL's rendered byte is **(247, 171, 131)**, requiring `accC.G ≥ 0.6686275`, i.e. a +4e-5 float excess the same per-sample values cannot produce.
- **Consequence:** per-sample values are bit-identical between backends, and compositing them identically yields Metal's byte. GL therefore reaches a higher float (or converts to u8 differently) at a step the CPU model still does not contain: the **terminal break**, the **post-loop finalize**, or the **store/readback conversion**.

## 7. Conclusion

- Closed this session: opacity-break strictness, one-extra-tail-sample, composite operation order, final scale/bias, TF filter.
- New fact: the ±1 field cannot come from per-sample opacity/color/lattice/arithmetic — GL's own per-sample values prove Metal's byte. The residual is isolated to terminal/store behavior.
- Required GL float excess per gated pixel (from update-52/53 replay thresholds): +0.0037..+0.0955 u8 (G +4e-5 at (93,201)) — consistent with a small one-directional store/terminal conversion difference, not a sample or ulp effect.

## 8. Current doubts / hypotheses (unresolved, after findings)

1. **Store / readback conversion (leading).** Metal's byte equals `round-half-even(accC*255)` (68/68 replay). If GL's fixed-function float→RGBA8 conversion, or its `vtkWindowToImageFilter` readback, rounds half-**up** (ties-away) or otherwise biases one LSB upward, every pixel whose `accC*255` fractional part is near `X.5` flips `X → X+1` in GL only. The one-directional 63,692-px field requires the fractional parts to cluster near `.5` for many pixels — testable only by capturing GL's **true final float** (`g_fragColor` after the loop, and after finalizeRayCast) via an extension of the byte-encode dump machinery and comparing against the model float per gated pixel. This is the same experiment as update 52 §6.2, now with the per-sample path excluded.
2. **Terminal break beyond strictness.** GL's `g_currentT >= g_terminatePointMax` (composer:3371-72) vs Metal's `firstT + currentT*stepSize >= tTerminateMax` + bounds+1e-4 checks (MetalShaders.metal:4818-4823); also the OOB-vs-opacity check order. Interaction with the last sample could add/remove one contribution for rays that saturate near the exit. Not yet numerically excluded for the 15 gated pixels (extra-sample test only added a sample after Metal's break, not after GL's own break rule).
3. **GPU contraction with unmodeled precision (last resort).** If GL's GPU evaluates `(1.0f - a) * g_srcColor + g_fragColor` in a fused form whose intermediate differs from all CPU f32 orderings, it could shift a few LSB — but the CPU order variants are byte-identical, so this would need a >100-ulp hardware behavior to reach +4e-5; implausible.
4. **Sample-dump re-render drift.** The dump re-renders the full frame once per sample index (320 renders); the captured per-sample values matched the model, but the single real render's sequence is not directly captured — low risk, kept for completeness.

## Artifacts

- Dump run (GL, Metal pixel (93,201) → GL px (93,310)):
  `VTK_GL_RAY_DUMP=1 VTK_GL_SAMPLE_DUMP=1 VTK_GL_SAMPLE_DUMP_PX=93,310 VTK_GL_SAMPLE_DUMP_MAX=320 build_macos_metal/bin/vtkRenderingVolumeCxxTests TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter --vtk-factory-prefer RenderingBackend=OpenGL -D build_macos_metal/ExternalData/Testing -T build_macos_metal/Testing/Temporary -V /tmp/bc/dummy_baseline.png 2> /tmp/bc/gl_dump_93_310.log`
- Logs: `/tmp/bc/gl_dump_93_310.log` (3.8 MB, 320 samples × channels), `/tmp/bc/u47_metal.log` (1,938 FINAL rows), `/tmp/bc/u47_gl.log` (no per-sample rows).
- Images: `/tmp/bc/fix_metal.png`, `/tmp/bc/fix_gl.png` (63,692 diff px).
- Replay models: `BackendComparisonTools/update48/replay_metal_accumulate.py` (68/68), session analyses inline.
