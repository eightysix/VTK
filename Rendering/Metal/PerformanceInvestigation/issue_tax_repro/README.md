# issue_tax_repro

Minimal repro of the RESIDUAL Metal-vs-GL gap that remains after the
do-while back-edge fix: at **1024x1024, SD 0.5** (`step=0.0005`), the plain
3D do-while march runs ~1-3% slower on Metal than GL. This is the last
reproducible >1 cell in the `../divergent_tail` matrix.

## What it contains

Two shader pairs (GLSL + MSL, identical math), fair wall-clock timing
(per-frame drain on both sides), interleaved rounds:

| pair | what it isolates | expected M/GL |
|------|------------------|---------------|
| `march31` | the shipped do-while march on the plain 3D path | ~1.01-1.03 |
| `l1fetch` | frozen coordinate — every tap hits one L1-resident texel, zero DRAM streaming, uniform trip counts | ~1.03-1.06 |

## Findings (bisecting the residual)

- ALU-only loop (no fetch): Metal FASTER (0.84) — see `../divergent_tail`
  V40. ALU exonerated.
- L1-resident fetches: Metal +3-6% — the deficit survives with ZERO DRAM
  traffic, so it is a per-tap SAMPLER ISSUE tax, not a memory-system or
  exit-divergence effect.
- Full streaming march: +1-3% — the issue tax partially hidden under DRAM
  latency.
- Consequence: halving tap count halves the tax — which is exactly why
  RG8 pair-packing (`../divergent_tail` V24/V32) wins at high resolutions.
  On the plain 3D path one tap per sample is the minimum, so this residual
  marks the floor of source-level optimization.

## Build / run

```
clang++ -std=c++17 -fobjc-arc -O2 -framework Metal -framework OpenGL \
      -framework Foundation -DGL_SILENCE_DEPRECATION \
      issue_tax_repro.mm -o issue_tax_repro
./issue_tax_repro [rt=1024] [frames=15] [step=0.0005]
```

Example output (M2 MBA):

```
rt=1024 frames=15 step=0.00050
pair                      GL ms/f  Metal ms/f    M/GL
march31 (residual)          40.72       40.99    1.01   parity 477960/229.4 vs 478044/229.4
l1fetch (issue tax)          8.51        8.83    1.04
```

Parity is exact (cov/mean match across backends).

## Repeatability

argv[4] toggles the main harness's compile options
(`MTLLanguageVersion3_2 + MTLMathModeFast`) to A/B them. Three
interleaved-round repetitions per setting:

| setting | march31 M/GL | l1fetch M/GL |
|---------|--------------|--------------|
| knobs off | 1.00 / 1.00 / 0.98 | 1.04 / 1.03 / 1.04 |
| knobs on  | 0.99 / 1.02 / 1.00 | 1.03 / 1.03 / 1.04 |

Compile options have no measurable effect. An earlier single-shot run
that read 0.96 did not reproduce (fluke). The l1fetch issue tax is the
stable signal: +3-4% in every run regardless of settings.
