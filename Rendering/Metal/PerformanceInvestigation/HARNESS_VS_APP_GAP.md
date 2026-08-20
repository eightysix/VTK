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

> RESOLVED (2026-08-20 evening session): refuted as the *cause* of the app gap.
> Both backends are per-pixel fields in the target config (below); the gap
> survives the field match. Details in §6.

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

## 6. Update (2026-08-20 evening session)

Phase-1 checks executed. New facts that change the picture:

### 6.1 Field mapping corrected: GL's noise sampling is per-pixel, not block-constant

`gl_FragCoord.xy / vec2(textureSize(in_noiseSampler, 0))` with NEAREST maps
`texel = floor(gl_FragCoord.xy / 64 * 64) = integer pixel coords mod 64` —
**each pixel gets its own noise value, tiled every 64 px** (app GL,
`vtkVolumeShaderComposer.h`; harness GL identical). The harness's earlier
"block-constant 32 px" reading was an arithmetic slip.

`VTK_METAL_TEST_JITTER_PARITY=1` does **NOT** reproduce GL's field: it
quantizes the noise to `viewportH/64`-px blocks (32 px at 2048), a *different*
field that renders with gross 32-px banding GL never shows. The flag is a
timing-parity hack, not visual parity. **Do not use it for A/B.**

### 6.2 The app benchmark A/B is field-mismatched on Metal by default

`TestMetalScenes.h:1242-1249`: with `VTK_METAL_TEST_JITTER=1` and no
`VTK_METAL_TEST_IGN_JITTER`, the Metal mapper is set to `UseIGNJitter = true`
→ Metal jitters with per-pixel Interleaved Gradient Noise, GL with per-pixel
blue noise. Both are per-pixel scattered, so the IGN-vs-blue choice is
cost-neutral — but it is NOT the config that matches GL's field.

### 6.3 Measured this session (back-to-back, 2048/SD4, 30 frames)

| config | j0 | j1 | jitter Δ | M/GL j1 |
|---|---|---|---|---|
| GL (per-pixel blue noise, app native) | 48.78 | 60.67 (52.64 rerun) | +12.0 ms / +24% | — |
| Metal per-pixel blue noise (`IGN_JITTER=0`) | 43.96 | 68.35 | +24.4 ms / +55% | 1.30 |
| Metal per-pixel IGN (bench default) | 43.96 | 69.87 | +25.9 ms / +59% | 1.30 |
| Metal GL-parity 32-px block (`JITTER_PARITY=1`) | 43.96 | 44.34 | +0.4 ms / +1% | 0.84 |

Visual parity (GL vs Metal per-pixel blue noise): mean abs diff 0.064/255,
9.5% of pixels differ, max 122 — the renders are equivalent.

### 6.4 What survives

- The app Metal jitter penalty (+24-26 ms) matches the harness Metal
  (+24.7 ms) on a per-pixel field — the harness is faithful on Metal.
- The app GL jitter penalty (+12 ms) is ~2x cheaper than the harness GL
  (+25.2 ms) on an identical per-pixel field. The harness over-charges GL —
  §2's conclusion stands, and it is NOT a field-shape effect.
- The app M/GL j1 ~1.29-1.30 persists with matched per-pixel fields on both
  backends. The asymmetry is real, sits on the GL side (app GL jitter is
  anomalously cheap, or the harness GL march is heavy), and the harness's
  equalized field cannot reproduce it.

### 6.5 Target configuration for reproduction (visual parity to GL)

The config that renders identically to GL is the **per-pixel blue-noise**
field on both backends:

```
# GL side (app native): per-pixel 64-tile blue noise
VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 \
VTK_METAL_TEST_MINMAX=0 VTK_METAL_TEST_ACCEL=0 VTK_METAL_TEST_NUM_SLABS=1 \
VTK_METAL_TEST_JITTER=1 build_macos_metal/bin/vtkMetalGLVisualComparison \
  --bench --backend gl --scene DICOMVolume \
  --dicom /Users/macair/Public/IMR/CTIMR/IMRToraceAddome \
  --frames 30 --reps 1 --size 2048x2048

# Metal side (app default blue-noise, IGN off — NOT the bench default)
VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 \
VTK_METAL_TEST_MINMAX=0 VTK_METAL_TEST_ACCEL=0 VTK_METAL_TEST_NUM_SLABS=1 \
VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_IGN_JITTER=0 \
  build_macos_metal/bin/vtkMetalGLVisualComparison \
  --bench --backend metal --scene DICOMVolume \
  --dicom /Users/macair/Public/IMR/CTIMR/IMRToraceAddome \
  --frames 30 --reps 1 --size 2048x2048
```

Do NOT set `VTK_METAL_TEST_JITTER_PARITY` (block-quantized; gross, not
GL-parity). Do NOT leave `IGN_JITTER` unset (forces IGN).

### 6.6 Next

The remaining open question is the GL side: why is app GL jitter ~+12 ms while
harness GL (same field, same march semantics, same TF, same early-exit) is
~+25 ms? Candidates: GL thermal variance is huge (±15% run-to-run; j1 measured
60.67 vs 52.64 in the same session), the app's depth-buffer-driven
`g_terminatePointMax`/discard path, or the per-step texture-space vs
ray-parameter coordinate differences. Instrument the harness GL with the app's
`g_currentT`-to-red iteration encoding (`VTK_METAL_TEST_GL_ITER=1`) and compare
iteration distributions.

## 7. Resolution scaling (2026-08-20)

Single-run, 30 frames, SD4, minmax/accel off, slabs=1. Metal side uses
per-pixel blue noise (`IGN_JITTER=0`), matching GL's field.

### 7.1 Raw timing (ms/frame)

400-2048: single-run, sequential. 4096/8192: interleaved GL/Metal to
eliminate thermal ordering bias.

| Resolution | GL j0 | GL j1 | Metal j0 | Metal j1 |
|---|---|---|---|---|
| 400x400 | 14.98 | 19.80 | 13.91 | 19.18 |
| 800x800 | 21.75 | 31.58 | 23.62 | 38.84 |
| 1024x1024 | 27.24 | 36.63 | 27.41 | 47.85 |
| 2048x2048 | 40.32 | 54.14 | 44.15 | 69.48 |
| 4096x4096 | 75.65 | 90.16 | 65.01 | 85.91 |
| 8192x8192 | 232.66 | 264.37 | 188.65 | 199.84 |

### 7.2 Jitter delta (j1 − j0)

| Resolution | GL Δ ms | GL Δ % | Metal Δ ms | Metal Δ % | Metal/GL Δ ratio |
|---|---|---|---|---|---|
| 400x400 | +4.82 | +32.2% | +5.27 | +37.9% | 1.09 |
| 800x800 | +9.83 | +45.2% | +15.22 | +64.4% | 1.55 |
| 1024x1024 | +9.39 | +34.5% | +20.44 | +74.6% | 2.18 |
| 2048x2048 | +13.82 | +34.3% | +25.33 | +57.4% | 1.83 |
| 4096x4096 | +14.51 | +19.2% | +20.90 | +32.1% | 1.44 |
| 8192x8192 | +31.71 | +13.6% | +11.19 | +5.9% | 0.35 |

### 7.3 Backend ratios

| Resolution | M/GL j0 | M/GL j1 | j1−j0 spread |
|---|---|---|---|
| 400x400 | 0.93 | 0.97 | +0.04 |
| 800x800 | 1.09 | 1.23 | +0.14 |
| 1024x1024 | 1.01 | 1.31 | +0.30 |
| 2048x2048 | 1.10 | 1.28 | +0.18 |
| 4096x4096 | 0.86 | 0.95 | +0.09 |
| 8192x8192 | 0.81 | 0.76 | −0.05 |

### 7.4 Analysis

**M/GL j1 tracks M/GL j0.** Every resolution where Metal is slower at j0 is
also slower at j1, and vice versa. The "jitter gap" is not purely a jitter
phenomenon — it is two stacked effects:

1. **Base overhead (j0 gap).** At 1024-2048, Metal j0 is ~1.01-1.10x GL j0.
   This baseline carries through to j1 arithmetically: adding the same
   absolute jitter to a larger base inflates the ratio.

2. **Jitter asymmetry (spread).** The spread (j1 ratio − j0 ratio) isolates
   the pure jitter effect. It peaks at +0.30 (1024x1024) and is +0.18 at
   2048. Metal's jitter costs ~2x GL's in absolute ms at these resolutions.

Decomposition at the pathologic cell (2048, M/GL j1 ≈ 1.28):
- ~0.10 from base Metal overhead (j0 gap)
- ~0.18 from jitter asymmetry (spread)
- Total ≈ 1.28

**Scaling regime boundaries (confirmed interleaved at 4096/8192):**
- **≤400:** Negligible gap. Both backends are lightweight; jitter costs are
  small in absolute ms, Metal slightly faster overall.
- **800-2048:** Gap regime. Metal j0 ≈ GL j0, but Metal jitter Δ is 1.5-2.2x
  GL's. This is the production rendering range and where the gap is real.
- **≥4096:** Inversion. Metal becomes faster than GL at both j0 and j1. At
  8192, M/GL j0 = 0.81, M/GL j1 = 0.76. The crossover is real (not a
  thermal artifact — confirmed with interleaved back-to-back runs on the same
  machine). At extreme pixel counts, Metal's pipeline handles the load more
  efficiently; GL degrades faster as resolution grows.

**Metal jitter Δ scales sub-linearly.** Goes +5 → +15 → +20 → +25 ms then
drops to +21 (4096) and +11 (8192). At extreme resolutions the per-pixel
jitter divergence is amortized by other fixed costs in the pipeline.

## 5. Files

- `jitter_gap_repro/` — harness + `RESULTS.md` (build/run/knobs, verdicts).
- `jitter_gap_repro/RESULTS.md` — full experiment record.
- `../SLAB_BENCHMARKS.md` — app bench recipe §1, pathological cell §7.2.
- `../jitter_lag_repro/` — sharp-field (IGN) vs tile-field A/B harness.
- App: `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm`,
  `Rendering/Metal/Shaders/MetalShaders.metal`,
  `Rendering/VolumeOpenGL2/vtkOpenGLGPUVolumeRayCastMapper.cxx`.
