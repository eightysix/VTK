# Metal-side bisect of clean GL's compile divergence, step 2: `in_volume_scale` bit-identity confirmed, and a 36-combination whole-loop reassociation sweep proves the alpha chain is exactly Metal's — a single extra tail sample reproduces (93,201), but the mechanism behind it (and the TF-table last-ulp, unverified) remains open (update 51)

**Date:** 2026-08-09
**Scope:** Continue update 50 §6's Metal-side bisect plan. This session (1) closes doubt #1 — `in_volume_scale` is **bit-identical** between backends (both `0xe1f26f41`); (2) CPU-bisects every plausible whole-loop composite reassociation of GL's written form `(1.0f - g_fragColor.a) * g_srcColor + g_fragColor` across all 68 gated pixels — **none reproduces clean GL, and any deviation in the alpha chain diverges on all 68 pixels**, pinning the alpha chain to be exactly Metal's `fma(1-accA, op, accA)`; (3) shows via store-threshold analysis that **one extra tail sample reproduces (93,201) exactly** (G needs only +4e-5, B +1.6e-4, R stays < +0.0016), but the required accA gap (+0.00029 below Metal's break value) is too large for ulp rounding — pointing at a per-sample opacity or sample-count difference rather than arithmetic; (4) flags that the update-46 §3.3 "TF tables match" claim was validated against only **2 of 1024 entries**, so a table last-ulp difference at specific idx is not ruled out.
**Target (unchanged):** Metal output must be bit-identical to **clean GL** (`RenderingBackend=OpenGL` without debug injection).

**Follows:** [Update 50](VolumeRayCastBackendComparisonFindingsUpdate50.md), [Update 49](VolumeRayCastBackendComparisonFindingsUpdate49.md), [Update 48](VolumeRayCastBackendComparisonFindingsUpdate48.md), [Update 46](VolumeRayCastBackendComparisonFindingsUpdate46.md).

---

## 1. Doubt #1 closed: `in_volume_scale` is bit-identical

Update 50 §6.1 asked whether GL's CPU-computed scale and Metal's in-shader scale differ in the last ulp. They do not:

- GL (`vtkOpenGLGPUVolumeRayCastMapper.cxx:3958-3993`, via `vtkVolumeTexture::GetScaleAndBias`): for `VTK_UNSIGNED_SHORT`, `glScale = 1/(65535+1)`, `glRange[1] = 4370 * glScale`, then `scale = float(1.0 / (glRange[1] - glRange[0]))` = `float32(65536/4370 in double)`.
- Metal (`MetalShaders.metal:3922`): `scalarScale = 1.0f / max((scalarMax - scalarMin), 1e-4f)` with `scalarMax = float32(4370/65536) = 0.066680908203125` (exact, dyadic). IEEE float32 division `1.0f / 0.066680908203125f` is correctly rounded.

Both yield **`0xe1f26f41` = 14.996796607971191** (verified in float32 via exact rational arithmetic). No scale divergence; the ±1 field is not caused by a scale last-ulp.

## 2. Whole-loop reassociation bisect (update 50 §6.2): all variants, 68 gated pixels

Replay harness extended from update 48 (`whole_loop_variants.py`, new in this update) — reuses the exact update-48 model (`fma(w, f32(op*rgb), accC)`, `/65536`, `floor(norm*1024)`, `accA ≥ 1-1/255`, round-half-even store), which reproduces Metal 68/68.

Single-step composite reassociations (update 50 §4 reconfirm):

| variant | metal-match |
|---|---|
| `fma(w, op*rgb, accC)` (Metal written) | 68/68 |
| `muladd` (`f32(w*op*rgb)+accC`) | 68/68 |
| `fma(f32(w*op), rgb, accC)` | 68/68 |
| `fma(f32(w*rgb), op, accC)` | 68/68 |

Back-to-front / GL-written-with-src-weight forms (tested but **wrong weight source**):

| variant | metal | GL | note |
|---|---|---|---|
| `w = 1-op; accC = fma(w, accC, src)` | 0 | 0 | gross overshoot, e.g. (93,201) → (254,219,187) |
| `w = 1-op; muladd` | 0 | 0 | same |

So GL does **not** reassociate the composite to a src-alpha-weighted form.

### 2.1 The decisive sweep: 36 combinations of the real reassociation set

GL's written composite is literally `g_fragColor = (1.0f - g_fragColor.a) * g_srcColor + g_fragColor` (composer lines 2651-2652) — the weight is the **accumulated** alpha `1 - accA`, same as Metal. The plausible compiler rewrites of `(1-a)*src + frag` are:

- `fma(1-a, src, frag)` (Metal/vec-fma),
- `fma(-a, src, frag) + src`,
- `fma(-a, src, src+frag)`,
- muladd `round(round((1-a)*src) + frag)`,
- reassociated fma `fma(1-a, frag, src)` and muladd `round(round((1-a)*frag) + src)`.

Sweeping all 6 × 6 (color × alpha) over all 68 gated pixels: **every combination except the exact Metal form misses on all 68 pixels** (0 metal-match, 0 GL-match). Consequences:

- The **alpha chain is provably exactly `accA = fma(1-accA, op, accA)`** in both backends — any alternative alpha rounding shifts the break sample and diverges every saturated pixel (all gated rays break at `accA ≥ 1-1/255`).
- The color chain is likewise pinned: only the exact fma form reproduces Metal, and GL is *closer* to Metal (15/68 differ) than to any variant (68/68 differ).

→ The divergence is **not** in the accumulate arithmetic. It must live in per-sample *inputs* or the sample *count*.

## 3. Store-threshold analysis at (93,201): one extra tail sample reproduces GL exactly

Metal accC = (0.96899, 0.66858715, 0.51160765) → stored (247,170,130). GL = (247,171,131). With round-half-even store the required deltas are:

| channel | Metal | needed for +1 | required delta |
|---|---|---|---|
| R (must stay 247) | 0.96899 | < 0.97059 | margin +0.0016 |
| G (→171) | 0.668587 | > 0.668627 (170.5/255) | +4e-5 |
| B (→131) | 0.511608 | > 0.51176 (130.5/255) | +1.6e-4 |

A single extra tail sample (i=105, `w≈0.0036, op=0.250333, rgb=(1, 0.883089, 0.759707)`) adds ≈ (0.0009, 0.0008, 0.0007) — **R stays 247, G→171, B→131 exactly**. Consistent with update 50's whole-loop/pre-saturation bias: GL must simply accumulate ~1 sample more than Metal before the opacity break.

**But the required accA gap is too large for ulp rounding.** Metal's logged accA at i=104 (pre-sample) is 0.995158; after i=104 it is ~0.99637, which crosses the threshold 0.996078. For GL to run one more sample, its accA after i=104 must be < 0.996078 — **0.00029 lower**, i.e. ~0.1% lower per tail sample (idx 260/261 op ≈ 0.25), or a real sample-count difference. That is not rounding noise; it is either a per-sample opacity difference at specific TF idx, or a termination/step-count mechanism that adds a sample.

## 4. Inputs re-verified; the one input still not fully validated: the TF table

- **Accumulators are float** (`MetalShaders.metal:4060-4061`); the `w`/`accC` fields in the `SAMPLE` rows are half-rounded debug prints only (`float(1.0h - accumulatedOpacity)`, line 4505) — not the real values.
- **Break threshold parity**: GL `g_opacityThreshold = 1.0 - 1.0/255.0` (composer 3263), break `g_fragColor.a > g_opacityThreshold` (strict, 3371); Metal `accumulatedOpacity >= 1.0f - 1.0f/255.0f` (non-strict, 4814). Identical value; strict-vs-non-strict only matters if accA lands exactly on the threshold.
- **Position / step / texel knife-edge**: sweeping (93,201) across ±1-ulp step per axis, ±1-ulp start, and x-offsets up to 2e-5 texture units (the i=78/79 knife-edge margins are ≈1.8e-5) produced **no** GL match. The i=76 knife-edge (frac 0.00024) is color-inert (val 935↔910 both map to op=0.0054).
- **Uniform opacity scale**: ~0.998 reproduces (93,201) but only matches 6/15 GL-diff gated pixels while breaking metal-matches — a uniform opacity scale is not the answer.
- **TF table**: the update-46 §3.3 "tables match" claim compares `MTL_OPTABLE`/`GL_OPTABLE` logs that print only **metadata + 2 entries** (`table[25%]=0.537743`, `table[50%]=0.85`). The full 1024-entry RGBA32F tables were never compared entry-by-entry, and the two backends build them through different code paths. A last-ulp (or even ~0.1% at specific idx) table difference at the tail idx values is **not ruled out** and is the leading open input.

## 5. Conclusion

- `in_volume_scale`: bit-identical → doubt #1 closed.
- Whole-loop composite reassociation: all 36 real combinations diverge; alpha chain is exactly Metal's → doubt #2's "compiler reassociated the accumulate chain" is effectively closed for the composite arithmetic.
- The divergence reduces to a **per-sample opacity/color input difference or a one-sample tail-count difference** near the opacity break; (93,201) needs only ~1 extra sample's worth of color.
- The remaining unvalidated input is the **TF table's full 1024 entries** (last-ulp per backend build path).

## 6. Current doubts / hypotheses (unresolved)

1. **TF table last-ulp (leading).** Metal and GL may build the 1024-entry RGBA32F opacity/color tables through different CPU paths (e.g. different `GetTable`/COMPOSITE pre-integration, float vs double intermediates), differing at specific idx in the last ulp or more. **Next: dump and compare all 1024 entries from both backends (extend the OPTABLE logging to full-table), and independently rebuild each backend's table on the CPU to diff bit-level.**
2. **One-extra-sample tail mechanism.** (93,201) is reproduced by processing i=105; this requires either a ~0.1% lower opacity at idx 260/261 or a sample-count/termination difference that adds a trailing sample. The accA gap (+0.00029) is too large for rounding — so this is a *symptom* of #1 or of a step/termination input, not an independent arithmetic cause.
3. **Store rounding (retained as fallback).** Metal round-half-even vs GL fixed-function u8 conversion is random-sign per pixel (update 48), so it cannot explain one-directionality; retained only per update 48.
4. **Position chain (revisit only if #1 fails).** The debug-injected GL lattice tracks Metal, so clean GL's true compiled lattice remains unobservable; the sweep found no perturbation that reproduces GL, but a larger-than-swept systematic difference (e.g. a step last-ulp accumulation over ~100 samples, or a different cell-to-point anchor) is not fully excluded.

## Artifacts

- `/tmp/bc/u47_metal.png`, `/tmp/bc/u47_gl.png` (13:07) — comparison pair (unchanged).
- New tooling in `Rendering/Metal/BackendComparisonTools/update50/`:
  - `whole_loop_variants.py` — 68-gated-pixel replay with the M / G / G2 / M2 / M3 / M4 composite variants.
  - (ad-hoc) the 6×6 reassociation sweep and the store-threshold/extra-sample analysis were run as inline `BC_DATA=/tmp/bc python3` scripts; fold into `whole_loop_variants.py` on the next iteration if useful.
- Scale bit-identity check (inline, exact rational + float32): GL `float(1.0/0.066680908203125)` = Metal `1.0f/max(0.066680908203125, 1e-4f)` = `0xe1f26f41`.
