# B (constant-scalar volume) residual root-caused: ±1 sample-count flip from a 0.2% step-length mismatch (update 68)

**Date:** 2026-08-10
**Status:** **milestone.** The u67 constant-scalar-volume divergence (B = 5530 px)
is now pinned at float precision: at every diverging pixel GL marches exactly
**one more sample** than Metal (n_GL = n_MT + 1.000 exactly), with **bit-identical
per-sample opacity**. The extra sample comes from a systematic ~0.2% shorter
step in GL's `g_dirStep` vs Metal's march step. The 100%-one-sided GL>MT sign
(u67 §5) is a direct consequence: one extra sample of positive opacity always
makes GL brighter.

## 1. Methodology fix: y-inversion in the GL dumps

GL `glReadPixels` / `gl_FragCoord` are **bottom-left origin**; Metal `screenPos`
is top-left. The GL FINAL/SAMPLE dump reads `glReadPixels(gx,gy)`, so top-left
`(x, y)` requires **`gy = 511 - y`**.

- First GL run used `VTK_GL_SAMPLE_DUMP_PX=439,281` → actually top-left
  **(439,230)**, which does NOT diverge (GL α=0.531121433, Metal
  α=0.531121433 same). The apparent "Metal darker by 0.0012" in the first
  read was a pixel-mismatch artifact.
- Correct pair for top-left (439,281): GL `VTK_GL_SAMPLE_DUMP_PX=439,230`.

## 2. Decisive float-precision comparison at a diverging pixel (439,281)

B variant = `VTK_STEP_MODE=3 VTK_STEP_CONSTANT=2000 VTK_STEP_RAMP_MAX=0.02
VTK_STEP_WHEEL=1` (constant-scalar 2000 volume, opacity ramp 0→0.02).

Pre-finalize accumulated alpha (GL channel 63 / Metal `accOp`):

| backend | α (float32) | backout n | per-sample op |
|---|---|---|---|
| GL | 0.534601569 | **308.000021** | 0.00248023518 |
| Metal | 0.533444000 | **306.999675** | 0.00248023518 |

- Per-sample opacity is **identical** (0.00248023518 = pre-integrated
  1−(1−0.00915332)^0.270059). GL op dump channel 7 == Metal SAMPLE `op=`.
- Sample **positions** match bit-exactly through i=0..306 (GL pos
  `(0.360522777, 0.426111341, 0.997226834)` == Metal `eval`
  `(0.360523, 0.426111, 0.997227)` at i=306).
- GL marches a **308th sample** (i=307, `pos=(0.360049, 0.425851,
  0.999013)` — still in volume, z<1). Metal stops at i=306
  (`tex=(0.360206, 0.425936, 0.998284)`).

## 3. Verified across all gated pixels: dn is exactly +1 at diverging pixels, 0 elsewhere

Backout uses `n = ln(1−α)/ln(1−a)`, a = 0.00248023518:

| pixel (top-left) | GL n | MT n | dn | 8-bit diff |
|---|---|---|---|---|
| (439,281) | 308.000 | 307.000 | +1.000 | GL (135,108,80) vs MT (135,107,80) |
| (32,346)  | 292.000 | 291.000 | +1.000 | G+1 |
| (373,466) | 319.000 | 318.000 | +1.000 | B+1 |
| (422,419) | 318.000 | 318.000 | 0.000 | identical |
| (349,255) | 297.000 | 297.000 | 0.000 | identical |
| (256,256) | 290.000 | 290.000 | 0.000 | identical |

dn is **exactly +1.000** (integer) at all three divergent pixels and
**exactly 0** at matching pixels. This fully explains B's 5530-px field and
the one-sided GL>MT sign: one extra sample of α≈0.00248 adds 0.00116 to α ≈
0.3 LSB, flipping channels whose rounding sits at the boundary.

## 4. Root cause: GL `g_dirStep` is ~0.2% shorter than Metal's march step

At (439,281):

- GL `g_dirStep` (GL_RAY dump): `(-4.73408145e-4, -2.60755536e-4, 1.78641989e-3)`, |step| = **0.00186639**
- Metal `texStep` (STEP log, what the march loop actually advances by):
  `(-4.74400847e-4, -2.61333276e-4, 1.78985752e-3)`, |step| = **0.00187017**
- Metal also computes `evalStep` = `(-4.73408232e-4, -2.60755594e-4, 1.78642000e-3)`
  which is **bit-exact GL `g_dirStep`** — but the ray loop advances with
  `texStep`, not `evalStep`.
- Sample distance is identical on both backends (GL `GL_SAMPLING_RESULT`
  actual 0.270058721 == Metal `sampleDistanceWorld` 2.700587213e-01) — the
  step mismatch is purely the **step formula**, not the distance.

Step-size ratios: |texStep|/|g_dirStep| = 0.00187017/0.00186639 = **1.00202**,
so Metal's step is 0.202% longer. Over ~300 samples that accumulates to
~0.6 of a step, flipping the exit sample count (307 vs 308) on the ~2% of rays
whose exit boundary lands in the last-step window.

Why the formulas differ (from `MetalShaders.metal` comments, lines 3167-3180,
3997-3999, 4192):

- **GL** `g_dirStep = (inverseTextureDataAdjusted * normalize(vertexPos -
  eyePos)).xyz * in_sampleDistance` — normalizes the ray direction in
  volume/vertex space **before** the non-uniform texture transform, so the
  physical step is direction-dependent (per-axis −0.03%/+0.41%/−0.003% vs
  Metal).
- **Metal** `physicalSampleStep()` returns `sampleDistance * maxBound /
  length(rayDirNormSpace * boundsSize)` — a *constant physical* sample
  distance along the ray (deliberately introduced in an earlier update so the
  pre-integration factor, which assumes a full sampleDistance per step, does
  not over-accumulate opacity).

This is a deliberate, documented behavioral divergence, not an accident — and
it is now proven to be the direct cause of B's sample-count flips.

## 5. Doubts / hypotheses (open)

- **Does switching Metal's march loop from `texStep` to `evalStep` (the
  bit-exact GL `g_dirStep`) eliminate B entirely (5530 → 0)?** This is the
  obvious next experiment. It may resurrect the pre-integration
  over-accumulation concern the `physicalSampleStep` change was introduced to
  fix (u<16-era), so both B and the reference test must be re-checked.
- **Is the same 0.2% step-length mismatch the dominant driver of the other
  variants (A 742, C 286, D 9215/12620, E 13578)?** Those have larger max
  deltas (2-5) and diverge in more channels; the reference test (188 px) may
  still have a second, smaller mechanism on top (u59-64's 14 knife-edge
  attribute-interpolator displacement).
- **Why only ~2% of rays flip?** With a 0.202% step difference accumulating to
  ~0.6 of a step over a full ray, the flip probability depends on where exit
  boundaries fall relative to sample-grid phase; needs a quantitative model to
  confirm the 5530/262144 ≈ 2.1% fraction.
- Whether the *reference* test (with scalar variation) has the same ±1-sample
  mechanism as B, or whether B is its cleanest instance (constant volume
  makes n the only free variable).
- Whether GL's normalize-in-volume-space g_dirStep should be treated as the
  stable reference at all, given it is "physically" direction-dependent —
  the choice of reference is what the whole bit-identical exercise assumes.

## 6. Reproducibility

```
# GL: float dump at glReadPixels(439,230) = top-left (439,281)
env VTK_GL_RAY_DUMP=1 VTK_STEP_MODE=3 VTK_STEP_CONSTANT=2000 \
    VTK_STEP_RAMP_MAX=0.02 VTK_STEP_WHEEL=1 VTK_GL_FINAL_DUMP=1 \
    VTK_GL_SAMPLE_DUMP_PX=439,230 \
  build_macos_metal/bin/vtkRenderingVolumeCxxTests \
    TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformStepTF \
    --vtk-factory-prefer RenderingBackend=OpenGL \
    -D build_macos_metal/ExternalData/Testing \
    -T build_macos_metal/Testing/Temporary -V /tmp/bc/ramp_bl.png 2>&1 | rg GL_FINAL

# Metal: per-pixel FINAL log (os_log → stderr)
env VTK_STEP_MODE=3 VTK_STEP_CONSTANT=2000 VTK_STEP_RAMP_MAX=0.02 \
    VTK_STEP_WHEEL=1 MTL_LOG_LEVEL=MTLLogLevelDebug \
    MTL_LOG_BUFFER_SIZE=536870912 MTL_LOG_TO_STDERR=1 \
  build_macos_metal/bin/vtkRenderingVolumeCxxTests \
    TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformStepTF \
    --vtk-factory-prefer RenderingBackend=Metal \
    -D build_macos_metal/ExternalData/Testing \
    -T build_macos_metal/Testing/Temporary -V /tmp/bc/ramp_bl.png 2>&1 | \
  rg "FINAL px=\(439, 281\)"
```

Captures: `/tmp/bc/b3gl_final_439_230.out`, `/tmp/bc/b3gl_sample_439_230_full.out`,
`/tmp/bc/b3mt_gate2.out`, `/tmp/bc/b3gl_sample_439_230.out`.
