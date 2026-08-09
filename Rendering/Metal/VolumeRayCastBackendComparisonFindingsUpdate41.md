# Correction: the (422,92) "131-level" residual was a GL debug-dump artifact; clean GL == Metal at (422,92) — real residual is 63,691 px at ±1 plus 13 px (max 8) (update 41)

**Date:** 2026-08-09
**Scope:** Re-capture GL and Metal **without** the `VTK_GL_SAMPLE_DUMP`/`VTK_GL_RAY_DUMP` debug env (the per-sample re-render passes corrupt the captured GL frame — already documented for the update-22 baselines). With clean captures, (1) the hot pixel (422,92) is **bit-identical** GL vs Metal, (2) the update-40 "correction" claim of a "GL 175 vs Metal 170 sample step gap" is **wrong** — both backends march exactly 170 real samples (GL's extra i=170..174 log rows are terminated-march sentinels), and (3) the true residual is ±1 ULP on 63,691 px plus 13 px with a channel ≥2 (worst 8 at (397,110)).

**Follows:** [Update 40](VolumeRayCastBackendComparisonFindingsUpdate40.md).

---

## 1. The (422,92) pixel was never a rendering difference — it is the GL debug-dump pixel

The GL sample dump (`vtkOpenGLGPUVolumeRayCastMapper.cxx:4609`, `VTK_GL_SAMPLE_DUMP`) re-renders the volume geometry once per sample index (default max 175) into the **live framebuffer** before the frame capture. At the gated pixel the final framebuffer holds the byte-encoded debug value, not the composite. Update-22 already measured this for the Standard baseline (debug GL read `(216,61,76)` where clean read `(238,176,140)`); the same mechanism hits this test's gated pixel.

Clean re-captures (no dump env, same binary, same test, last frame):

| source | (422,92) RGB |
|---|---|
| GL clean | **(238, 192, 159)** |
| Metal clean | **(238, 192, 159)** |
| GL debug-instrumented | (216, 61, 76) |

Both marches at this pixel (GL pixel (422,419), Metal (422,92)) recompose to **GL=(238.1,191.7,158.6) / Metal=(238.1,191.7,158.6)** — matching the clean captures exactly. The image-level diff never represented a real backend difference at this pixel.

## 2. The update-40 "175 vs 170 sample gap" claim is wrong

Update-40's appended correction claimed GL marches 175 samples to Metal's 170. Re-reading the GL dump:

- GL logs rows i=0..174, but **i=170..174 all read the terminated-march sentinel −1.03674172e19** (the shader's `if (g_dbgDone) return;` short-circuits after the march terminates and the read-back byte-decode of the leftover pixel gives the sentinel). GL's real march is **i=0..169 = 170 samples**, identical to Metal (i=0..169).
- Per-sample positions ≤1e-6, raws ≤1.3e-7 for **all of i=0..169**, all 6 frames static.

So the step parity is 170 = 170; the "5 extra GL steps" explanation is void.

## 3. Clean GL vs clean Metal — the real residual

| metric | clean GL vs clean Metal |
|---|---|
| pixels differing (any channel) | 63,691 / 262,144 (24.3%) |
| pixels with sum-of-diffs == 1 | 57,711 |
| pixels with a channel ≥ 2 | **13** |
| max per-channel \|d\| | **8** at (397,110) |
| (422,92) | 0 (identical) |

The 13 pixels with a channel ≥2 (GL → Metal):

```
(397,110)  GL=(239,179,143) Metal=(239,186,151)  G+7 B+8
(360,229)  GL=(225,177,145) Metal=(225,174,141)  G-3 B-4
(405,171)  GL=(230,166,130) Metal=(230,168,133)  G+2 B+3
(349,255)  GL=(226,180,148) Metal=(226,182,151)  G+2 B+3
(439,281)  GL=(229,167,132) Metal=(228,166,130)  -1,-1,-2
(120,167)  GL=(245,179,141) Metal=(245,177,139)  G-2 B-2
(9,18)     GL=(248,195,159) Metal=(248,193,157)  G-2 B-2
(482,33)   GL=(242,162,121) Metal=(241,162,123)  R-1 B+2
(470,269)  GL=(228,146,107) Metal=(228,145,105)  -1,-2
(469,463)  GL=(242,170,131) Metal=(242,171,133)  G+1 B+2
(338,432)  GL=(241,168,130) Metal=(241,167,128)  -1,-2
(293,298)  GL=(231,156,119) Metal=(231,158,120)  G+2 B+1
(153,32)   GL=(247,185,147) Metal=(247,183,146)  G-2 R-1
```

All 13 are interior volume pixels near opacity knife-edges; none is the (422,92) gated pixel.

## 4. Doubts / problems to solve (recorded honestly, not yet answered)

1. **±1 ULP on 63,691 px — what is the source?** The per-sample raws/positions match to ≤1.3e-7, so a few ULP of accumulated float difference can flip the final 8-bit rounding boundary by ±1. Two hypotheses, both untested: (a) final float→uint8 rounding at a boundary; (b) genuine per-sample accumulation ULP divergence. Needs a float-accumulation-level comparison at a ±1 pixel (compare accumulated color/alpha per sample, not just raw), e.g. via `compare_gl_metal_accum.py`.
2. **debug-GL is *closer* to Metal (694 px) than clean-GL is (63,691 px).** The debug-instrumented GL shader differs from the clean GL shader only by the injected debug block, yet matches Metal 60k px better. Suspect GLSL compiler optimization differences (FMA / reassociation / constant folding) between the two shader variants rather than anything semantic — worth confirming by diffing the compiled debug vs clean GL fragment sources. If true, "clean GL" is the only valid reference and the ±1 field is the target to attack.
3. **The 13 px with channel ≥2.** Likely alpha-threshold/termination knife-edge pixels (like update-22's 1150 scalar edge). Next step: per-sample GL/Metal dumps at each, with the correct GL pixel pairing `(x, 511−y)` (GL dump uses `VTK_GL_SAMPLE_DUMP_PX=x,511-y` for a Metal pixel `(x,y)`), and compare step counts + last composited sample.
4. **Clean capture determinism not established.** Two clean GL runs should be compared to prove run-to-run stability of the ±1 field before treating it as backend divergence rather than jitter/timing.
5. **Goal definition.** With (422,92) now bit-identical and only 13 px at ≥2, define whether "bit-identical" means the full 512² at 8-bit (needs the ±1 field resolved) or float-accumulation parity (need the float-level comparison to pass first).

## 5. Next steps

- Clean re-captures are at `/tmp/bc/clean_gl.png`, `/tmp/bc/clean_metal.png` (no dump env).
- Verify doubt 2 (compiled shader diff / debug-vs-clean GL ±1 field).
- Per-pixel float accumulation comparison at a ±1 pixel and at the 13 hot pixels.
- Then decide the bit-identical target and iterate.

## Artifacts

- `/tmp/bc/clean_gl.png` (GL, clean, last frame), `/tmp/bc/clean_metal.png` (Metal, clean, last frame).
- Debug-instrumented frames `/tmp/bc/gl_fix2.png`, `/tmp/bc/metal_fix2.png` — **do not use for image diffs** (debug GL is ±1-ULP contaminated on 64k px and corrupted at (422,92); Metal is clean).
- Sample logs `/tmp/bc/gl_fix2.log`, `/tmp/bc/metal_fix2.log` — still valid for per-sample values (march i=0..169, 170 samples, matches).
