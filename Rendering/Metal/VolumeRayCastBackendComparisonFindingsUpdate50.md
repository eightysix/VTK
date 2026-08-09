# Metal-side bisect of clean GL's compile divergence, step 1: the ±1 field is confined to bright saturated pixels, and CPU bisect of the sole shared divergent pixel (93,201) rules out every single-step arithmetic, TF-index, norm-divisor, and position-chain variant — the ~+0.002 bias is a whole-loop, pre-saturation accumulation effect (update 50)

**Date:** 2026-08-09
**Scope:** Begin update 48 §6's Metal-side bisect plan ("reproduce clean GL's compile-level arithmetic divergence in MSL"). This session (1) documents the exact GL vs Metal termination structures and shows the loop-level termination differences are inert for this test because every divergent pixel saturates opacity before reaching the exit; (2) confirms the ±1 field is confined to bright (≥100/255) pixels with GL ≥ Metal on 63,647 of 63,690; (3) runs a CPU float32 bisect on **(93,201)** — the only pixel shared between the gated logs and the ±1 field, where clean GL=(247,171,131) vs Metal=(247,170,130) — against every plausible single-site arithmetic variant. **No tested variant reproduces clean GL**, and the divergence is a ~+0.002 uniform positive color accumulation entering before the opacity break.
**Target (unchanged):** Metal output must be bit-identical to **clean GL** (`RenderingBackend=OpenGL` without debug injection).

**Follows:** [Update 49](VolumeRayCastBackendComparisonFindingsUpdate49.md), [Update 48](VolumeRayCastBackendComparisonFindingsUpdate48.md), [Update 46](VolumeRayCastBackendComparisonFindingsUpdate46.md).

---

## 1. Termination structures: GL and Metal differ, but the loop-level difference is inert for this test

GL (`vtkVolumeShaderComposer.h`):

- `g_terminatePointMax = length(g_terminatePos - g_dataPos) / length(g_dirStep)` — a texture-space **float step count** computed once (line 3349-3350); break when the integer counter `g_currentT >= g_terminatePointMax` (line 3372).
- Out-of-bounds break is **strict, no epsilon** against the last-texel-center `in_texMax = AdjustedTexMax` (line 3361-3362).

Metal (`MetalShaders.metal` marchVolume):

- `maxSteps = max(1, int(ceil((p.tEnd - firstT) / p.stepSize)))` (line 4052) — a world-space ceiling.
- Out-of-bounds break with `+1e-4` slack against `blockMaxGlobal` (the box face) (line 4821).
- `tTerminateMax` (depth-texture terminator) = `1e30` for this test (no depth occlusion), so it never fires.

For camera-inside rays the per-ray *effective* terminator is the opacity saturation break (`accA >= 1 - 1/255`, line 4814), which matches GL's `g_fragColor.a > g_opacityThreshold` (line 3371) to the sample. Analysis of the debug log: of 1938 logged FINAL rows, **1932 terminate with accOp=1.000000** (opacity saturation); only pixel (240,176) terminates via the bounds/maxSteps path (accOp=0.993, and it is byte-identical between GL and Metal anyway). **Consequence:** the termination-formula and out-of-bounds-epsilon differences between GL and Metal cannot produce the bulk field — every saturated pixel has stopped accumulating color before reaching the exit, so a "one extra sample at the exit" effect contributes nothing there. This is consistent with (and tightens) update 48 §3's sample-count refutation.

## 2. Field geometry: all 63,690 differing pixels are bright, GL ≥ Metal on 63,647

Using `u47_metal.png` vs `u47_gl.png` (13:07, both 512×512):

```
differing px: 63690 of 262144
diff px bbox: full image (x 0..511, y 0..511)
GL mean brightness at diff px: 176.0
GL >= Metal at diff px: 63647 / 63690   (99.93% one-directional, GL brighter)
diff in dark (<100/255) interior: 0 of 63690
```

The entire ±1 field lives on bright pixels (the rendered head/tissue, not the dark background) and is one-directionally GL-brighter. Combined with §1, the ~+3.5e-4 float-level bias (update 42) accumulates **before opacity saturation**, i.e. inside the front-to-back composite of the saturated rays, not at their termination.

## 3. Per-sample input parity re-checked

- **Volume texture**: Metal uses `MTLPixelFormatR16Unorm` (`vtkMetalGPUVolumeRayCastMapper.mm:709-733`), so the sampler returns `u16/65535` — identical to GL's normalized `GL_UNSIGNED_SHORT` texture. No `65535` vs `65536` divisor split between backends.
- **TF tables**: match (update 46 §3.3); both RGBA32F 1024-wide, nearest.
- **Composite written form**: GL `g_fragColor = (1.0f - g_fragColor.a) * g_srcColor + g_fragColor` with `g_srcColor.rgb *= g_srcColor.a` (composer lines 2651-2652) == Metal's `fma(w, op*rgb, accC)` (update 46 §3.4).

## 4. CPU bisect at (93,201): the one divergent pixel with a logged lattice

(93,201) is the only pixel in the ±1 field with a Metal `DEBUG STEP` lattice (`localPos=(0.50653356, 0.50632495, 0.44901463)`, `evalStep=(-8.75e-6, -1.53e-4, +1.94e-3)`). Metal stores (247,170,130); clean GL stores (247,171,131) — G/B each one LSB higher. The update-48 float32 replay model (`fma(w, f32(op*rgb), accC)`, `/65536`, `floor(norm*1024)`, round-half-even store) reproduces **Metal** byte-for-byte here, validating the bisect harness. To match GL, G must reach ≥ 0.67059 and B ≥ 0.51373 vs Metal's 0.66859 / 0.51161 — a **+0.002 uniform positive bias** in the accumulated color, far above per-sample rounding noise.

Variants tested (all vs Metal-ref):

| category | variants | result |
|---|---|---|
| composite reassociation | `fma`, `muladd`, `fma(w*op,rgb)`, `fma(w*rgb,op)` | all bit-identical to Metal (reconfirms update 46 §3.5 at this pixel) |
| TF index width | `floor(norm*1024)` vs `floor(norm*1023)` | 1023 goes the **wrong** direction (−0.0014 G) |
| norm divisor | stored/65536 vs stored/65535 | identical to Metal |
| weight precision | f32(1−accA) vs double-precision | identical to Metal |
| position chain | incremental `p+=step` vs analytic `p0+f32(n*step)`, `fma(step,n,p0)`, double-exact | all identical to Metal |

Additionally: **no sample of this ray sits near a TF-bin edge** (all `norm*1024` coords ≥ 0.02 from an integer), so a last-ulp norm/scale difference cannot flip a TF bin here; and matching GL would need ~4 *extra trailing samples* to cross 171/131, which §1 rules out. One live knife-edge exists: sample **i=76** sits 0.00024 texel from a texel boundary in x (a 4.7e-7 texture-space shift flips texel 258↔259 in the steep opacity ramp), but none of the position variants flipped it, so the compiled position chain is not (yet) implicated.

## 5. Conclusion

- Metal-side single-step arithmetic, TF indexing, norm normalization, and position accumulation are **all closed** at the one pixel that distinguishes clean GL from Metal — none reproduces clean GL's +0.002.
- The divergence is a **whole-loop, pre-saturation accumulation bias** (clean GL accumulates ~0.3% more color along saturated rays), consistent with the bright-only, one-directional 63,690-px field.
- This is the first clean update-48 §6 step; the bisect now narrows to loop-level structure.

## 6. Current doubts / hypotheses (unresolved)

1. **`in_volume_scale` bit-identity (untested).** GL computes `in_volume_scale` on the CPU (`ScaleVec`, `vtkOpenGLGPUVolumeRayCastMapper.cxx:3958-3993`) = 14.9967966; Metal computes `scalarScale = 1.0f / max(scalarMax - scalarMin, 1e-4f)` in-shader (MetalShaders.metal:3922). If the two differ in the last ulp, every sample's norm shifts by 1 ulp — flipping TF bins at near-edge norms and producing exactly the systematic bright-region field (though it does *not* explain (93,201), whose norms are far from bin edges). **Next: verify bit-identity.**
2. **Whole-loop reassociation of the accumulate chain.** The fma chain `accC = fma(w, src, accC)` across ~105 samples could be reassociated by clean GL's compiler (register reuse, software pipelining, or factoring the `(1-a)` weight). The single step is order-insensitive (update 46), so the divergence must live in cross-sample structure — the one category not yet probed. **Next: emit an alternative whole-loop composite form in MSL (e.g. accumulate `accC` in the GL *written* `(1-a)*src + acc` mul+add form with the weight chain, or pre-scale src and defer the weight).**
3. **Store rounding (fallback).** Metal's round-half-even store vs GL's fixed-function u8 conversion is random-sign per pixel, so it cannot explain one-directionality; retained only per update 48 §"store rounding".
4. **Position knife-edge at i=76.** A 4.7e-7 position divergence flips a texel in the steep ramp; the analytic variants didn't flip it, but clean GL's compiled position chain remains unobservable — revisit only if loop reassociation variants fail.

## Artifacts

- `/tmp/bc/u47_metal.png`, `/tmp/bc/u47_gl.png` (13:07) — comparison pair.
- `/tmp/bc/u47_metal.log` (per-sample SAMPLE rows, per-gated-pixel STEP rows), `/tmp/bc/u47_gl.log` (GL uniforms/optable).
- New tooling in `Rendering/Metal/BackendComparisonTools/update48/`: `bisect_93_201.py` (Metal-ref validation + 4 quick variants), `inspect_93_201.py` (per-sample trace at (93,201)), `sweep_93_201.py` (composite × index-width × norm-divisor grid). All take `BC_DATA=/tmp/bc`.
