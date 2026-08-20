# Harness vs app: the jitter gap discrepancy (2026-08-20)

Prep doc for the next investigation phase. The `jitter_gap_repro` harness
concluded Metal's absolute jitter cost is **equal** to GL's on identical rays,
and M/GL j1 <= ~1.0 everywhere. The app still shows M/GL j1 ~1.29 on the
pathologic cell. This doc pins down exactly where the harness and app diverge
and what to check next.

## 1. Current results

### 1.1 Harness (`jitter_gap_repro/`) — reproduce

```
clang -fobjc-arc -lstdc++ -framework Metal -framework Foundation \
      -framework OpenGL -framework QuartzCore \
      Rendering/Metal/PerformanceInvestigation/jitter_gap_repro/jitter_gap_repro.mm \
      -o /tmp/jitter_gap_repro

/tmp/jitter_gap_repro 2048 4 10 Rendering/Metal/PerformanceInvestigation/dicom.u8
```

Averages at 2048/SD4 (8 rounds, means ± stdev; GL carries thermal variance,
Metal is stable):

| metric | GL | Metal |
|---|---|---|
| j0 (ms) | 50.30 ± 4.97 | 40.90 ± 0.86 |
| j1 (ms) | 75.49 ± 5.99 | 65.64 ± 1.45 |
| jitter delta (ms) | +25.19 ± 3.38 | +24.74 ± 1.76 |
| jitter delta (%) | +50.5 | +60.6 |
| M/GL j0 / j1 | — | 0.82 ± 0.08 / 0.88 ± 0.07 |

Paired per round, MetalΔ − GLΔ = −0.45 ± 3.04 ms: statistically zero. The
harness's "gap" (M/GL j1 1.28-1.35, prior sessions) was the `float3` vs
`packed_float3` vertex-stride bug — fixed, coverage now 1,831,193 vs GL
1,831,510 px. Hypothesis verdicts in `jitter_gap_repro/RESULTS.md`:
`allowGPUOptimizedContents=YES` (~50% slower), lattice re-alignment (same cost
class), and the 4 KiB `kBlue64` constant (texture2d-identical) are all
refuted; `constphase` (+1.8%) proves the cost is the per-pixel phase
divergence, not the tap.

rt x sd matrix (single rounds): M/GL j1 0.70-1.04, Metal never behind.

### 1.2 App (`vtkMetalGLVisualComparison`) — reproduce

Pathologic cell = single-pass oblique raw march (slabs=1), 2048x2048, SD4,
minmax/accel off:

```
VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 \
VTK_METAL_TEST_MINMAX=0 VTK_METAL_TEST_ACCEL=0 VTK_METAL_TEST_NUM_SLABS=1 \
VTK_METAL_TEST_JITTER=0|1 build_macos_metal/bin/vtkMetalGLVisualComparison \
  --bench --backend gl --scene DICOMVolume \
  --dicom /Users/macair/Public/IMR/CTIMR/IMRToraceAddome \
  --frames 30 --reps 1 --size 2048x2048
```

Swap `--backend gl` for `--backend metal`. Grep the `DICOMVolume` row.

Fresh measurement (2026-08-20, battery):

| | GL | Metal | M/GL |
|---|---|---|---|
| j0 | 42.07 | 43.76 | 1.04 |
| j1 | 54.32 | 70.32 | **1.29** |
| jitter delta | +12.25 ms / +29.1% | +26.56 ms / +60.7% | — |

Reproduces the documented 1.31-1.37x (SLAB_BENCHMARKS.md §7.2) and Metal's
+57-59% jitter penalty. NOTE: the app's march variant is currently pinned to
mv0 (TEMP-REPRO in `VolumeMarchVariant()`; revert to 9 before landing).

## 2. The discrepancy, pinpointed

Side-by-side jitter increments (absolute ms and % of the backend's own j0):

| | harness | app |
|---|---|---|
| GL | +25.19 ms / +50.5% | **+12.25 ms / +29.1%** |
| Metal | +24.74 ms / +60.6% | +26.56 ms / +60.7% |
| M/GL j1 | 0.88 | 1.29 |

**Metal is consistent harness-vs-app (~+25-27 ms, ~+60%). The harness GL
over-charges jitter by ~2x (+25.2 vs +12.3 ms).** So:

- The harness's "Metal pays the same as GL" is a harness-GL artifact: the app
  GL jitter is much cheaper than the harness GL jitter.
- In-app, Metal really does pay ~2x GL's jitter cost (+26.6 vs +12.3 ms) —
  the asymmetry the prior sessions described is real, but the harness's
  equalized-field A/B cannot reproduce it because harness GL already pays the
  high rate.

The harness is faithful on the Metal side and wrong on the GL side.

## 3. Candidate causes (harness GL over-pays jitter)

H1. **GL jitter semantics mismatch.** The harness GL does origin-shift
`tStart += noise*stepSize`, and its j0 adds a full `stepSize` (VTK
`g_rayJitter = g_dirStep` parity). Verify against the app's composed GL source
(`vtkVolumeShaderComposer.h`): exact shift formula, whether j0 shifts at all,
and the noise scale. A different j0 baseline inflates/deflates the increment
arithmetically.

H2. **Noise field/state mismatch.** Harness GL: 64x64 R32F, NEAREST + REPEAT,
sampled at `gl_FragCoord.xy / textureSize(...)`. App: `GetNoiseTextureUnit`
float tile, same NEAREST+REPEAT — but confirm the actual sampler state, tile
size/scale, and that the harness's `textureSize`-per-fragment fetch compiles to
the same thing. Also: the harness GL fetches the noise even on the j0 path? (It
does not — `if (uJitter > 0)` only. App shader may sample unconditionally,
changing j0's cache footprint.)

H3. **Iteration-count interaction.** Early-exit (`accOp > 0.996`) and
`maxSteps` computation under the per-pixel phase spread; if the app's TF
reaches saturation sooner, the phase spread matters less. The harness uses the
app's "Airways II" LUT (dumped), so the TF should match — re-verify.

H4. **March prologue differences.** The app GL prologue is heavier (lighting,
minmax hooks off but prologue remains); heavier per-step work dilutes the
fetch-divergence share. The harness GL is a stripped march — if the harness's
inner loop is dominated by the fetch, the jitter penalty concentrates.

H5. **Field asymmetry in the app (Metal side).** The app Metal jitter may
use a sharp per-pixel field (IGN hash) or lattice semantics while app GL uses
the block-constant 64px tile — `jitter_lag_repro` notes describe a sharp
per-pixel IGN hash class for the app Metal. The harness intentionally
equalized the field; the app may not be equalized. This would explain Metal's
in-app penalty with zero harness discrepancy. Check which branch
`MetalShaders.metal` actually compiles for the current pipeline.

## 4. Plan

1. **Dump the app's composed GL fragment source** (TEMP hook in
   `vtkOpenGLGPUVolumeRayCastMapper`/`vtkOpenGLShaderProgram`) and diff against
   the harness GL fs: jitter formula, j0 shift, noise sampler, TF. Falsify or
   confirm H1/H2/H3/H4 on the GL side.
2. **Check the app Metal jitter field/semantics** actually compiled
   (`MetalShaders.metal`: tile vs IGN, origin-shift vs lattice) — H5. This is
   the highest-leverage check: if the app's fields differ, the "Metal pays
   2x" reduces to a field-shape effect the harness deliberately removed.
3. **Make the harness GL faithful to the app** (expected: harness GL jitter
   increment drops from +25.2 to ~+12 ms, M/GL j1 -> ~1.29, reproducing the
   app cell).
4. Only if the app asymmetry survives a field-matched harness: investigate the
   Metal read path under the app's exact field (sharp vs tile) with the
   harness's A/B knobs (`lattice`, `NOISE_TEX`, `constphase`) — the
   explanation space is prepared in `jitter_gap_repro/RESULTS.md`.

## 5. Files

- `jitter_gap_repro/` — harness + `RESULTS.md` (build/run/knobs, verdicts).
- `jitter_gap_repro/RESULTS.md` — full experiment record.
- `../SLAB_BENCHMARKS.md` — app bench recipe §1, pathological cell §7.2.
- `../jitter_lag_repro/` — sharp-field (IGN) vs tile-field A/B harness.
- App: `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm`,
  `Rendering/Metal/Shaders/MetalShaders.metal`,
  `Rendering/VolumeOpenGL2/vtkOpenGLGPUVolumeRayCastMapper.cxx`.
