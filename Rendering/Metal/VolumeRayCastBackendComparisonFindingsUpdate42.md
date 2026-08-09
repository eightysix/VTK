# Decomposing the ±1 field: 63,691 px is a systematic GL>Metal accumulated-float bias (~3.5e-4), not ULP noise; composite multiply ordering accounts for ~1% of it; hot 13 px are separate (update 42)

**Date:** 2026-08-09
**Scope:** Quantitatively decompose the 63,691-px ±1 field reported in [update 41](VolumeRayCastBackendComparisonFindingsUpdate41.md). The result: (1) the ±1 field is a **one-directional systematic bias** (GL > Metal on 99.9% of flipped channels, uniform across the image), not random ULP noise; (2) per-channel flip rate 8.88% ⇒ GL's accumulated float color is ~0.0888 byte ≈ **3.5e-4 higher than Metal's**; (3) the composite-multiply ordering difference found earlier sims to only **2.9e-6** (~1% of the bias); (4) the composite accumulators in Metal are already full `float` (not half), so half-precision loss is **not** the cause; (5) the 13 px with a channel ≥2 flip the *opposite* direction (GL lower) and are a separate knife-edge issue.

**Follows:** [Update 41](VolumeRayCastBackendComparisonFindingsUpdate41.md).

---

## 1. The ±1 field is systematic and one-directional

From the clean-vs-clean diff field (`/tmp/bc/clean_gl.png` vs `/tmp/bc/clean_metal.png`, 262,144 px, 3 channels):

| metric | value |
|---|---|
| pixels differing (any channel) | 63,691 / 262,144 = 24.3% |
| channel flips, total | 69,855 / 786,432 = 8.88% per channel |
| sign GL > Metal | **69,874 of 69,855 = 99.97%** |
| sign GL < Metal | 21 (all within the 13 hot px + 8 stragglers) |
| row-band distribution | uniform: 7.8k–10.3k per 64-row band |
| pixels with a channel ≥2 | 13 (separate issue, §4) |

Three facts follow:

1. **99.97% one-directional is not rounding noise.** Random ULP divergence would split roughly 50/50. A one-sided bias means GL's accumulated float color is genuinely *higher* than Metal's by a small, image-wide constant.
2. **Bias magnitude.** A byte flips when the underlying float sits within the bias of a rounding boundary. Per-channel flip rate 8.88% ⇒ the offset is **0.0888 byte ≈ 3.5e-4** in accumulated float color (assuming uniform fractional parts, which smooth gradients approximate). A 2.9e-6 offset would flip only ~0.1% of channels; we see 8.88% — the real bias is ~120× larger.
3. **Uniformity rules out data-dependent transfer-function sampling** as the primary cause: a per-scalar LUT error would concentrate at specific scalar values / regions, not smear uniformly over the whole volume.

## 2. Composite multiply ordering: real, sign-correct, but ~100× too small

The two backends apply the front-to-back composite with different multiply associativity:

- **GL** (`vtkVolumeShaderComposer.h`, `OpacityAndColorTermination`/composite):
  `g_srcColor.rgb *= g_srcColor.a;` then `g_fragColor = (1.0f - g_fragColor.a) * g_srcColor + g_fragColor;`
  → `w · (c·a)` — premultiplies the sample first, then scales by the weight.
- **Metal** (`vtkMetalGPUVolumeRayCastMapper.mm:4794`): `accumulatedColor += weight * sampleColor * sampleOpacity;`
  → `(w·c)·a` — left-associative; the weight multiplies the color *before* the opacity.

Float32 is non-associative; the two orderings round differently. Simulated with the real per-sample values from the (422,92) march (i=0..169):

| ordering | final (422,92) recompose |
|---|---|
| GL `w·(c·a)` | R 0.933820307, G 0.751744628, B 0.622474551, A 0.996922791 |
| Metal `(w·c)·a` | R 0.933817387, G 0.751741469, B 0.622471571, A 0.996922791 |

Per-channel color divergence ≈ **2.9e-6, GL higher** — the correct sign, and it rounds to the same byte at (422,92) (which is why that pixel is bit-identical). But 2.9e-6 is ~120× smaller than the 3.5e-4 bias the flip rate implies. **The ordering difference explains ~1% of the ±1 field.** Fixing it is necessary but not sufficient.

## 3. Ruled out: half-precision accumulation

Hypothesis: a `half` intermediate in the Metal accumulation would contribute a systematic ~1e-3 relative error (half mantissa ~10 bits ⇒ ulp ~4.9e-4 at 1.0), matching the observed bias magnitude. **Checked and ruled out** — `MetalShaders.metal:4058-4065` promotes all composite accumulators and weight arithmetic to `float`:

```
float3 accumulatedColor = float3(initialColor);
float accumulatedOpacity = float(initialOpacity);
```

Only the per-sample TF values stay `half` (unavoidable: the TF textures are half4), and those feed into the float composite as reads, not accumulated state. So the 3.5e-4 bias is not Metal accumulating in low precision.

## 4. The 13 hot pixels are a different (opposite-sign) phenomenon

The worst 13 px flip GL **lower** than Metal (up to −8 on G/B), e.g. `(397,110) GL=(239,179,143) Metal=(239,186,151)`. The bulk field is GL **higher** by ±1. Two opposite-signed mechanisms are at work:

- bulk ±1: the ~3.5e-4 systematic bias (§1);
- hot 13 px: a larger, localized divergence — all sit at opacity knife-edges (update-22's 1150 scalar class). Per-sample dumps at these pixels (with the correct `VTK_GL_SAMPLE_DUMP_PX = x, 511−y` pairing) should show step-count or last-sample divergence, not ULP smearing.

## 5. Open hypotheses for the remaining ~3.4e-4

The bias is uniform, one-directional, ~3.5e-4, and not half-precision. Remaining candidates, cheapest first:

1. **GLSL compiler reassociation / FMA contraction in clean GL.** Debug-instrumented GL is *closer* to Metal (694 px, update-41 §4 doubt 2) than clean GL is (63,691 px). If the clean GLSL variant compiles the composite with an FMA (fused `a*b+c`) while Metal's MSL does not, the clean-GL accumulate rounds only once per step and drifts ~0.5 ulp/step → ~1e-4–4e-4 over 170 steps, exactly the right size and sign. **Test: diff the compiled clean-GL vs debug-GL fragment shader (SPIR-V/MTL) for the composite block.**
2. **GL `+=` on g_fragColor vs Metal `+=` on accumulatedColor with the 1.0-alpha clamp at termination.** Metal sets `accumulatedOpacity = 1.0f` at break (`:4815`); GL leaves `g_fragColor.a` at the pre-break value (~0.9969). This changes only alpha, not RGB, so it should not move the color — unless a downstream step consumes alpha.
3. **The last composited sample differs by one** at some pixels. Same march depth verified at (422,92) only; needs the per-sample accumulation-level comparison at a representative ±1 pixel (not the knife-edge ones).

## 6. Next steps

- Run `compare_gl_metal_accum.py` at a representative ±1 pixel (e.g. a mid-image, mid-gradient pixel) and at one hot pixel, comparing accumulated color/alpha per sample — pin down whether divergence starts at sample 0 (per-sample input drift) or accumulates late (FMA/termination).
- Diff the compiled clean-GL vs debug-GL fragment shaders for the composite block (doubt 1).
- Measure clean-GL run-to-run determinism of the ±1 field (two clean captures) to confirm it's backend arithmetic, not jitter.

## Artifacts

- `/tmp/bc/clean_gl.png`, `/tmp/bc/clean_metal.png` — the diffed pair.
- Previous sample logs `/tmp/bc/gl_fix2.log`, `/tmp/bc/metal_fix2.log` — per-sample raw/op/color for (422,92) marches.
