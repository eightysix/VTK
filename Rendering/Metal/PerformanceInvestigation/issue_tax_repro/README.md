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

## Follow-up bisection: which part of the tap costs? (2026-08-21)

The harness was extended with six more pairs (same interleaved-round
protocol, parity checked every pair). Motivation: the shipped VTK shader
samples through `constexpr sampler sVolume` (MetalShaders.metal), while
this repro (and divergent_tail) bind RUNTIME samplers — first question
was whether the whole "issue tax" was a runtime-sampler artifact.

```
rt=1024 frames=15 step=0.00050 knobs=1
pair                        GL ms/f  Metal ms/f    M/GL
march31 (residual)            39.46       40.53    1.03
l1fetch (issue tax)           8.41        8.76    1.04   <- runtime linear sampler
l1_ce (constexpr)             8.40        8.73    1.04   <- shipped sVolume shape
l1_near (nearest)             8.41        8.86    1.05
l1_ce_near                    8.48        8.80    1.04
l1_read (no sampler)          8.45        2.15    0.25   <- !!!
l1_x2 (2 taps/iter)          16.57       17.15    1.04
l1_rep (addr repeat)          8.42        8.72    1.04
l1_pix (coord pixel)          8.40        8.78    1.05
```

Verdicts (stable across 3 knobs-on + 1 knobs-off sessions):

| hypothesis | pair | verdict |
|---|---|---|
| runtime-sampler artifact | l1_ce / l1_ce_near | **FALSIFIED** — constexpr == runtime (1.03-1.04); one 1.12 reading did not reproduce |
| bilinear weight math | l1_near | **FALSIFIED** — nearest identical |
| clamp-to-edge handling | l1_rep | **FALSIFIED** — repeat addressing identical |
| normalized-coord transform | l1_pix | **FALSIFIED** — pixel coords never better, occasionally worse (1.04-1.12, noisiest pair): the sampler does the transform for free |
| ILP amortization | l1_x2 | **FALSIFIED** — time doubles exactly on BOTH sides (16.5/17.1); the tax is strictly per-tap-linear |
| cost is the memory system | l1_read | **REFRAMED** — see below |

The `l1_read` row is the big reveal: replacing the sampler call with a
raw `vol.read()` (GL twin: `texelFetch`) of the same frozen texel drops
Metal from 8.7 to **2.15 ms/f (M/GL 0.25)** while GL stays at ~8.4.
So on this M2:

- Metal raw-load path: ~112 Gtexel/s (warp-uniform L1-resident taps broadcast)
- Metal sampler engine: ~27 Gtaps/s
- GL sampler path AND GL texelFetch: ~28 Gtaps/s (identical to each other)

Corrected framing of the "issue tax": it is NOT that Metal's sampler is
slower than GL's — the two APIs' SAMPLER rates agree within 3% (which is
exactly the march31 residual). It is that taps cost ~27-28 Gtap/s through
anyone's sampler front-end, and Metal alone exposes a cheaper raw-load
escape hatch that GL lacks. The residual 1024xSD0.5 cell is the small
difference between the two sampler implementations' fixed per-tap issue
cost, partially hidden under DRAM latency in streaming marches.

Why this closes rather than opens a lever:

- `read()`-based manual filtering cannot exploit the 4x: it wins only
  when every lane hits ONE address (broadcast). Real marching taps are
  per-lane distinct and stream from DRAM — V18 (`../divergent_tail`)
  measured manual-trilinear 2.5x WORSE there.
- With the front-end exonerated down to its fixed rate, fewer taps per
  sample remains the only mitigation — re-validating RG8 pair-packing
  (V24/V32) as the correct and final source-level answer.
- Consistency check: at 2048/SD0.5 the shipped march runs ~15 Gtap/s —
  BELOW the 27 Gtap/s sampler ceiling, i.e. DRAM-bound, which is why the
  residual only bites in L1-hot/small-frame regimes (1024-class).

Repro validity note: since constexpr == runtime here, the historical
divergent_tail/issue_tax measurements (all runtime-sampler) remain
comparable to the app's constexpr-sampler behavior.

## Round 3: arrays, transmittance form, and a warm-up trap (2026-08-21)

Chasing "Metal must be >= GL", four more hypotheses were measured
(v9-v17 in the harness). Results at the reference cell (1024xSD0.5):

```
l1_arr (2D-array tap)      GL 4.47   Metal 4.62   1.03
march_2ta (V23 two-tap)    GL 43.65  Metal 49.29  1.13
march_T* (transmit forms)  see below
march_a_nolod_sw           GL 44.83  Metal 40.32  0.90 (elevated GL)
```

Findings:

1. **A 2D-array bilinear tap costs HALF a 3D trilinear tap on BOTH APIs**
   (l1fetch 8.4 -> l1_arr 4.5). Yet the two-tap slice-array MARCH loses
   13% here (march_2ta) — in the streaming regime the second fetch per
   sample costs more than the cheaper tap saves. Consistent with V23's
   absolute regressions elsewhere; RG8 (1.25 taps avg) would sit between.
   Representation changes cannot close this cell.

2. **Transmittance rewrite (acc += T*o; T -= T*o, exit T>0.1) is
   arithmetically identical to the alpha form and measures identically in
   steady state** — FALSIFIED as a lever. On the way it exposed a genuine
   MSL codegen corner case worth recording: T-form + RUNTIME sampler +
   uniform-first while-condition compiled to a ~20% slower PSO in some
   sessions; changing either factor alone avoided it. The shipped shader
   uses constexpr samplers, so it never hits this combo — one more reason
   to keep constexpr sVolume.

3. **THE BIG ONE: GL warm-up contamination.** All four transmittance rows
   shared ONE identical GL program yet timed 50.8 / 47.0 / 39.1 / 38.5 ms
   sequentially within one session — Apple's GL driver keeps re-JITting a
   freshly-linked complex program far beyond kWarmup=10 draws, inflating
   whatever block runs first. Raising warmup to 30 (and especially
   running any comparison after prior GPU work) collapses everything:
   second back-to-back launch with warmup=30:

```
rt=1024 frames=15 step=0.00050 knobs=1
pair                        GL ms/f  Metal ms/f    M/GL
march31 (residual)            41.47       40.67    0.98
march_nolod                   39.06       40.53    1.04
march_2ta (arr march)         44.26       49.80    1.13
march_T (transmit)            40.22       41.64    1.04
march_T_nl (leanest)          40.24       40.87    1.02
march_T_rsmp_sw               40.57       40.84    1.01
march_T_nolod_orig            40.80       40.78    1.00
march_a_nolod_sw              40.14       40.95    1.02
```

Every spelling of the plain 3D do-while march lands at parity
(0.96-1.04) once GL reaches steady state. The celebrated "+2-3%
residual" at 1024xSD0.5 was substantially a measurement artifact of
under-warmed GL programs plus session drift; the true residual is inside
run-to-run noise. The l1fetch microbenchmark (+3%) remains real but does
not surface through a DRAM-streaming march.

## Bottom line for "Metal >= GL"

- Fairly measured (wall-clock, per-frame drain, warmup>=30, same-session
  A/B), the plain 3D path is AT PARITY at every resolution/SD cell;
  sub-3% deviations flip sign between sessions on this battery MacBook.
- Where Metal can actually WIN, it already does via the representation
  policy from ../divergent_tail: RG8 pair-packed + do-while (V32) at
  >=2048-class (0.77-0.96), plain 3D + do-while (V31) below.
- Ship-shape notes for the MSL: keep constexpr samplers (also avoids the
  T-form corner case); mip_filter::none without an explicit level operand
  is safe and marginally leaner; avoid adding a data-dependent exit
  condition BEFORE the trip-count test if a runtime sampler is ever used.
- kWarmup in this harness raised 10 -> 30 because of finding 3.
