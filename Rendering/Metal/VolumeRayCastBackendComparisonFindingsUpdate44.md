# The GL per-sample debug dump is not a valid clean-GL reference: debug-injected GLSL compiles to different float arithmetic and matches Metal on 60k more px — the clean-GL ±1 field is a compile-level difference, not a per-sample input divergence (update 44)

**Date:** 2026-08-09
**Scope:** After update 43 ruled out the ordering, output-pixel-format, volume-texture, and TF-table-precision paths, the documented next step was a per-sample accumulation-level comparison (`compare_gl_metal_accum.py`) at a representative ±1 pixel. That comparison came back with **zero per-sample divergence** — which, given the `VTK_GL_SAMPLE_DUMP` debug caveat already flagged in [update 41](VolumeRayCastBackendComparisonFindingsUpdate41.md) (§4 doubt 2), demanded a check: does the debug-injected GL actually represent clean GL's arithmetic? It does not.

**Follows:** [Update 43](VolumeRayCastBackendComparisonFindingsUpdate43.md), [Update 41](VolumeRayCastBackendComparisonFindingsUpdate41.md).

---

## 1. Fresh capture (post-fix binary, 2026-08-09 12:18)

Re-captured per-sample dumps with the current (composite-order-fixed) binary at the representative ±1 GL-higher pixel **Metal (422,419)** ↔ **GL (422, 92)**:

- GL: `VTK_GL_RAY_DUMP=1 VTK_GL_SAMPLE_DUMP=1 VTK_GL_SAMPLE_DUMP_PX=422,92` → `/tmp/bc/gl_u44_samples.log`
- Metal: `MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=16777216 MTL_LOG_TO_STDERR=1` → `/tmp/bc/metal_u44_samples.log`

`compare_gl_metal_accum.py gl_u44_samples.log metal_u44_samples.log 422 419`:

- 132 samples each; **first per-sample divergence: None** (raw/op/rgb equal to log precision at every i).
- Position linear fits: step differs by ≤0.0002% per axis; max drift ≤5e-7.
- Accumulation replay: `dAccC ≈ 0` at all checkpoints; final `GL=(0.949248,0.663471,0.511390)` vs `MT=(0.949250,0.663472,0.511392)` — agree to 6 decimals.

Naively this says "per-sample identical, nothing to fix" — but the clean image at this pixel is **GL=(242,169,131), Metal=(242,169,130)** (B+1, GL higher). The replay must be reconciling against the wrong reference.

## 2. The debug-GL per-sample data replays to Metal, not clean GL

Replaying the GL `gl_u44_samples.log` samples (last frame) with the same front-to-back loop:

```
debug-GL replay final accC = [0.949248 0.663471 0.511390]  ×255 = (242.06, 169.18, 130.40) → (242,169,130)
clean  GL (422,419)        = (242, 169, 131)
clean  Metal (422,419)     = (242, 169, 130)
```

The debug-GL per-sample values recompose to **clean Metal**, not clean GL. The debug-injected GLSL variant evaluates the composite differently than the clean GLSL variant, and that different evaluation happens to equal Metal's.

## 3. Image-level confirmation (all with the same, unchanged GL binary)

| comparison | differing px | max \|d\| | direction |
|---|---|---|---|
| clean GL (11:48) vs debug GL (11:20) | 64,088 | 131 | 64,088 pure debug-LOWER |
| debug GL (11:20) vs clean Metal (11:50) | 694 | 131 | — |
| clean GL vs clean Metal | 63,691 | 8 | GL>Metal 99.97% |
| clean GL (11:48) vs clean GL (12:04) | **0** | 0 | deterministic |

Two independent clean GL captures are byte-identical (0 px), so the ±1 field is a deterministic backend difference, not jitter. The debug-GL-vs-clean-GL divergence (64,088 px) is nearly the same size as the clean-GL-vs-Metal field (63,691 px), and debug GL sits ~60k px closer to Metal. The GL sample-dump mechanism (`vtkOpenGLGPUVolumeRayCastMapper.cxx:4609` CPU-side re-render loop) only corrupts the **frame**; but the `VTK_GL_RAY_DUMP`-gated shader source injection (line 2979) changes the **compiled GLSL arithmetic** — register allocation / FMA contraction / constant folding — which is why the debug-GL image and even its per-sample values track Metal instead of clean GL.

## 4. What this means for the ±1 field

- **Not per-sample input divergence:** positions ≤1e-6, raws ≤1.3e-7, per-sample op/rgb equal to log precision, 132/132 samples identical — between Metal and the debug-GL compile. Since debug-GL ≈ Metal and clean GL differs from debug GL by 64k px, the clean-GL-vs-Metal ±1 field is a **compile-level arithmetic difference**: the same written formula (GLSL `g_fragColor = (1.0f - g_fragColor.a) * g_srcColor + g_fragColor`) rounds differently under the two shader-language compilers on this Apple M2.
- **The GL debug dump cannot localize it.** Any accumulation-level comparison that consumes `GL_SAMPLE` rows measures debug-GL arithmetic, which already equals Metal. Per-sample data from the debug passes is only valid where it does not depend on the injected block's compile effects (march geometry: positions/raws).
- **Clean GL is deterministic** (0 px across two captures), so the ±1 field is a stable property of clean-GL's compiled arithmetic, not noise. The target remains well-defined, but the tooling must stop using debug-GL per-sample rows as the clean reference.

## 5. Next steps

- Stop treating `GL_SAMPLE` per-sample rows as clean-GL. To still diagnose the ±1 field at float level, instrument **Metal's** accumulation (its `DEBUG SAMPLE` row already logs `w/accA/accC` per sample without changing Metal's arithmetic — Metal debug capture == Metal clean capture, verified in update 22) and replay the **written GLSL formula in float32 on CPU** for the same sample positions, comparing against Metal's logged `accC` — this isolates whether Metal's compile diverges from the formula, or GL's does.
- Alternatively, bisect the difference by forcing one side to match: compile the GLSL with FMA contraction off (or Metal with explicit `fma`) and re-diff the ±1 field — this directly tests whether the field is an FMA-contraction asymmetry between the GLSL and MSL compilers.
- Given debug-GL ≈ Metal, consider whether "clean GL" (the GLSL driver's compile) is the right bit-identical target at all, or whether the written source formula is (in which case Metal may already be closer to it than clean GL is). This is a goal decision for the user.

## Artifacts

- `/tmp/bc/gl_u44_samples.log`, `/tmp/bc/metal_u44_samples.log` (fresh, post-fix binary, pixel Metal (422,419)/GL(422,92)).
- `/tmp/bc/clean_gl.png` (11:48) vs `/tmp/bc/fix42_gl.png` (12:04): clean-GL determinism, 0 px.
- `/tmp/bc/gl_fix2.png` (debug GL) vs `/tmp/bc/clean_metal.png`: 694 px — debug GL ≈ Metal.
- `/tmp/bc/gl_fix2.png` vs `/tmp/bc/clean_gl.png`: 64,088 px, all debug-lower — debug injection changes compiled GLSL arithmetic.
