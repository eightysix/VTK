# B residual 18 → 3 px: the block-bounds exit (GL has none) fired 1 sample early (update 70)

**Date:** 2026-08-10
**Status:** **milestone.** The remaining 18 B-survivors of update 69 (all
n_MT = n_GL − 1) are eliminated by removing the legacy block-bounds exit
(`currentPoint` vs `blockMinGlobal/blockMaxGlobal` ±1e-4) from the camera-inside
march loop. OpenGL `TerminationImplementation` has no such check, so it was
firing one sample *before* the CTP bounds test on GL's `g_dataPos` lattice. After
the fix, Metal takes the **same sample count as GL at every pixel** of the StepTF
reference test; the image residual drops 18 → **3 px, all exactly ±1 LSB**, all at
knife-edge pixels where GL's own interpolated anchor drifts ~4 ulp between renders
or the lattice sits within 1 ulp of the exit plane.

## 1. How the cause was isolated (this session)

1. **Killed the 1-ulp CTP-plane hypothesis (update 69 §4a).** The plane values are
   bit-identical and exactly representable in float32 on both backends:
   `ctpOffset = 0.0009765625 = 2^-10`, `ctpScale = 0.998046875`, so
   `adjTexMax = ctpOffset + ctpScale = 0.9990234375 = 1 − 2^-10` exactly
   (`vtkVolumeTexture.cxx` `AdjustedTexMin/Max` vs in-shader `ctpOffset + ctpScale`
   agree bit-for-bit; GL_CTP dump confirms).
2. **Python float32 lattice simulation** (`/tmp/bc/analyze_exit.py`) of both
   exit lattices from the 9-sig-digit STEP/GL_RAY dumps showed n_MT == n_GL for
   every pixel — but n_MT was *one more* than the observed Metal `lastIter + 1`.
   The Metal loop was stopping one sample earlier than its own lattice allowed.
3. **Break-reason instrumentation** (breakWhy at each break site) proved it: for
   16/18 pixels the loop broke with **breakWhy=5 — the block-bounds exit**
   (`p.checkBounds && (any(currentPoint < p.blockMinGlobal − 1e-4) ||
   any(currentPoint > p.blockMaxGlobal + 1e-4))`), not the CTP test. Only
   (312,183) and (435,480) broke on the CTP test (breakWhy=2).
4. `currentPoint` is a **separate lattice** (anchored at `cameraPos + rayDir *
   tStart`, advanced by `rayDir * stepSize`) from `evalPoint` (anchored at the
   interpolated `localPos + evalStep`, advanced by `evalStep`). The two drift; the
   block-bounds plane (volume z ≈ 1.0001) was crossed at the iteration *before*
   the CTP plane (texture z ≈ 0.99902) on the `evalPoint` lattice — exactly one
   sample short.
5. **GL has no such exit.** `TerminationImplementation`
   (`vtkVolumeShaderComposer.h:3366`) breaks only on the CTP test, the opacity
   threshold, and `g_currentT >= g_terminatePointMax`. The block-bounds check is a
   Phase-6 leftover (commit 999a6238f3, the original camera-inside exit) that
   predates the CTP parity work. `p.checkBounds` is `true` **only** for the
   camera-inside proxy (line 5098); the legacy marchSegment/grid callers pass
   `false`, so the removed line was camera-inside-only.

## 2. The fix (Rendering/Metal/Shaders/MetalShaders.metal, marchVolumeUnified)

Removed the loop-end block-bounds break; termination now matches GL exactly:
CTP test on `evalPoint` (line 4276), opacity threshold (line 4898),
`tTerminateMax` (line 4903), and the `i < maxSteps` loop bound. The `currentPoint`
variable remains (used by the min-max empty-cell skip path).

## 3. Verification

- Both backends deterministic (Metal 3 runs md5-identical; GL 2 runs md5-identical).
- StepTF reference test (VTK_STEP_MODE=3 VTK_STEP_CONSTANT=2000
  VTK_STEP_RAMP_MAX=0.02 VTK_STEP_WHEEL=1): **diff 18 → 3 px, max Δ 1**
  (was max Δ 1 before too, but 18 px; all were n_MT = n_GL − 1).
- The 3 survivors, all ±1 LSB, single channel:
  - **(312,183)** and **(435,480)** — GL's *own* interpolated anchor shifts ~4 ulp
    between render 2 and render 3 of the run (`GL_RAY tex` at (312,328):
    z 0.448911905 → 0.448911786; origin 0.450802296 → 0.450802177), flipping the
    knife-edge sample count from 290 → 291. **Metal matches GL's first-two-render
    anchor bit-for-bit** (9-sig-digit round-trip). The PNG captures GL's *last*
    render, hence 1 LSB. Both backends are individually deterministic.
  - **(71,424)** — adjacent to the fixed (70/74/75,424) group; GL stably takes 298
    samples (α 0.522899568), Metal stably takes 299 (α 0.524083). A pure sub-ulp
    knife-edge: the last lattice position sits within ~1 ulp of the exit plane.
- Per-sample cross-check: Metal FINAL accOp at (140,6) etc. now equals GL FINAL
  c=67 to the last printed digit (e.g. 0.504793 / 0.504792571; 0.524083 /
  0.524082899; 0.519332 / 0.519331932).

## 4. Doubts / hypotheses (open)

- The 3 survivors are the update-69 §4b/c family: (b) a ~1–4 ulp difference in
  the interpolated anchor (`localPos` vs `ip_textureCoords`) — for 2 of the 3
  pixels this is **GL's own render-pass drift** (Metal matches GL's render 1-2
  value); (c) `evalStep` vs `g_dirStep` per-axis ~1e-10..1e-8 (1–50 ulp)
  differences (measured this session: anchor 1–3 ulp, step 0–50 ulp across the
  residual pixels).
- Whether the reference test's 169 ±1 px floor (u59–64 attribute-interpolator
  family) changed — not measured this session (different test binary).
- Whether the legacy camera-outside / grid-traversal loops need the same
  block-bounds removal or the CTP-lattice bound — unchanged this session.

## 5. Reproducibility

```
env VTK_STEP_MODE=3 VTK_STEP_CONSTANT=2000 VTK_STEP_RAMP_MAX=0.02 VTK_STEP_WHEEL=1 \
  build_macos_metal/bin/vtkRenderingVolumeCxxTests \
  TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformStepTF \
  --vtk-factory-prefer RenderingBackend=Metal \
  -D build_macos_metal/ExternalData/Testing -T build_macos_metal/Testing/Temporary \
  -V /tmp/bc/metal_new.png
```

(OpenGL variant: `RenderingBackend=OpenGL`.) Diff: numpy per-channel max over the
two PNGs. Captures: `/tmp/bc/metal_new.png`, `/tmp/bc/metal_run2/3.png`,
`/tmp/bc/gl_new.png`, `/tmp/bc/gl_run2.png`; break-reason logs
`/tmp/bc/brk_mt_raw.log` (before fix) and `/tmp/bc/brk_mt2_raw.log` (after fix).
