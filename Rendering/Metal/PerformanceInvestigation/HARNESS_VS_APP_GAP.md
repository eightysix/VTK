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

1. **j0 gap (small).** At 2048, interleaved 8-round M/GL j0 = 1.046 ± 0.029
   (~+1.9 ms). At 1024, M/GL j0 = 1.027 ± 0.025 (tied). The old 1.10 / +4 ms
   was a sequential thermal artifact (GL 40.3 vs interleaved 41.7; Metal barely
   moved). The +1.9 ms is unlabeled — envelope vs coherent-path vs noise. It
   is not large enough to manufacture Metal Δ = 2× GL Δ.

2. **Jitter asymmetry (spread).** The spread (j1 ratio − j0 ratio) isolates
   the pure jitter effect. It peaks at +0.30 (1024x1024) and is +0.18 at
   2048. Metal's jitter costs ~2x GL's in absolute ms at these resolutions.

**Frozen metrics (replaces earlier decomposition):**

```
M/GL j0  1024: 1.03 ± 0.03   (tied — headline cell)
M/J j0  2048: 1.05 ± 0.03   (~+1.9 ms; do not re-litigate; do not subtract)

Jitter metrics (only):
  GL Δ ms, Metal Δ ms
  spread = (M/GL j1) − (M/GL j0)
```

An additive j0 gap cancels in Δ (j1−j0) by algebra. The 2× jitter penalty
is purely a j1 problem regardless of what the +1.9 ms is.

**Scaling regime boundaries (confirmed interleaved at 4096/8192):**
- **≤400:** Negligible gap. Both backends are lightweight; jitter costs are
  small in absolute ms, Metal slightly faster overall.
- **800-2048:** Gap regime. Metal j0 ≈ GL j0, but Metal jitter Δ is 1.5-2.2x
  GL's. This is the production rendering range and where the gap is real.
- **≥4096:** Inversion. Metal becomes faster than GL at both j0 and j1. At
  8192, M/GL j0 = 0.81, M/GL j1 = 0.76. The crossover is real (confirmed
  with interleaved back-to-back runs). At extreme pixel counts, Metal's
  pipeline handles the load more efficiently; GL degrades faster as resolution
  grows.

**Metal jitter Δ scales sub-linearly.** Goes +5 → +15 → +20 → +25 ms then
drops to +21 (4096) and +11 (8192). At extreme resolutions the per-pixel
jitter divergence is amortized by other fixed costs in the pipeline.

## 8. j0 gap: small, not the 2× story (2026-08-20)

Interleaved 8-round j0 at 2048: M/GL = 1.046 ± 0.029 (~+1.9 ms).
1024 j0: M/GL = 1.027 ± 0.025 (tied within noise).

The old 1.10 was a sequential thermal artifact: GL ran hotter in sequential
ordering (40.3 ms) than interleaved (41.7 ms); Metal barely moved (43.6 ms).
The +1.9 ms leftover is real but small — not large enough to explain the 2×
jitter gap. Left unlabeled (envelope vs coherent-path vs noise). Do not
re-litigate.

**MARCH_STEPS probe — status unknown.** The fixed-steps data showed Metal
constant at ~44.5ms from steps=1 to 1000, which is internally contradictory:
unconstrained Metal Δ (j1−j0) is ~+25 ms, so the march MUST add measurable
time. Likely the cap did not bite (wrong uniform, pipeline not rebuilt, or
forced steps already below the computed cap). Needs verification via GL_ITER
PPM (red×4096 at steps=1: if still hundreds, cap never entered the loop).
Do not interpret the table until the PPM confirms the cap works.

Full data in `J0_GAP_DUMP.txt` (note: §3 interpretation is provisional).

## 9. Jitter investigation: interleaved j1 + sample-count PPMs (2026-08-20)

### 9.1 Interleaved j1 (8 rounds, GL↔Metal, JITTER=1, IGN_JITTER=0)

| Cell | GL Δ (ms) | Mtl Δ (ms) | Mtl/GL Δ | spread |
|---|---|---|---|---|
| 1024 | 11.00 ± 2.02 | 18.49 ± 0.55 | 1.68× | 0.190 |
| 2048 | 10.27 ± 1.60 | 25.53 ± 0.35 | 2.48× | 0.285 |

Metal jitter is 6-9× more stable than GL (stdev 0.18-0.55 vs 1.6-2.0 ms).
The 2× penalty scales with resolution: 1.68× at 1024, 2.48× at 2048.

### 9.2 App GL sample-count PPMs (GL_ITER at 1024)

| | Covered px | % total | Mean iter | Median | P95 |
|---|---|---|---|---|---|
| GL j0 | 421,828 | 40.2% | 86.5 | 80.3 | 192.8 |
| GL j1 | 423,564 | 40.4% | 86.7 | 80.3 | 192.8 |

**Iteration counts are identical between j0 and j1.** Jitter does NOT add
iterations — the +11 ms jitter cost is entirely a per-sample cost change
(divergent memory access, cache behavior, phase interaction with early-exit).

### 9.3 MARCH_STEPS cap verification

`VTK_METAL_TEST_MARCH_STEPS=1` + GL_ITER: mean iterations = 86.5 (not 1).
The cap did not bite — MARCH_STEPS is dead on GL. The fixed-steps data in
§8/J0 dump is uninterpretable for per-sample cost.

### 9.4 Key findings

1. **Jitter cost is per-sample, not per-iteration.** Same trip count, different
   $/sample. The 2× penalty is entirely in the inner-loop cost under phase
   scatter.

2. **The 2× lives in the 800-2048 window.** At 1024 (j0 tied), Metal pays
   1.68× GL's jitter. At 2048, 2.48×. Next question: why does Metal's
   per-pixel phase cost ~half of GL's? (H4/H7 — inner-loop shape, $/divergent
   sample.)

3. **Harness GL over-charges jitter.** Harness GL Δ ≈ +25 ms vs app GL Δ ≈
   +11 ms. The harness march is heavier per-step, so jitter divergence costs
   more. Harness GL PPMs not available (no GL_ITER in harness).

Full data in `JITTER_DUMP.txt`.

## 10. App Metal sample-count PPMs — real mapper, not harness (2026-08-20)

The harness Metal iter PPM (§9.5-style, `jitter_gap_repro` `GL_ITER`) showed
the same trip counts as GL, but that only proves the *harness* shader. To
answer "does the real Metal mapper march more samples?" the production shader
was instrumented:

- `MetalShaders.metal`: `marchIter` counter in `fragment_volume_main` (variant-0
  march), captured at the loop's end; when the `_padCropFlags[0]` uniform flag
  is set (`METAL_ITER=1` env → `vtkMetalGPUVolumeRayCastMapper.mm:7055`), the
  fragment returns `half4(marchIter/256, 0, 0, 1)` instead of the final color.
  One int add per iteration in all pipelines; output gated by the flag.
- `xcrun -sdk macosx metal -c` compiles the edited shader clean.
- App command (same target config as §6.5, `JITTER=0|1`):

```
VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 \
VTK_METAL_TEST_MINMAX=0 VTK_METAL_TEST_ACCEL=0 VTK_METAL_TEST_NUM_SLABS=1 \
VTK_METAL_TEST_JITTER=0|1 VTK_METAL_TEST_IGN_JITTER=0 \
METAL_ITER=1 build_macos_metal/bin/vtkMetalGLVisualComparison \
  --bench --backend metal --scene DICOMVolume \
  --dicom /Users/macair/Public/IMR/CTIMR/IMRToraceAddome \
  --frames 5 --reps 1 --size 1024x1024
```

(GL side: same invocation with `--backend gl` + `VTK_METAL_TEST_GL_ITER=1`,
writes `/tmp/app_gl_iter.ppm`.)

### 10.1 Results (1024x1024, SD4, minmax/accel off, slabs=1)

| | Covered px | % total | Mean iter | Median | P95 | Max |
|---|---|---|---|---|---|---|
| App GL j0 | 421,828 | 40.2% | 86.5 | 80.3 | 192.8 | 225 |
| App GL j1 | 423,564 | 40.4% | 86.7 | 80.3 | 192.8 | — |
| App Metal j0 | 457,876 | 43.7% | **81.1** | **72.3** | 187.7 | 222 |
| App Metal j1 | 457,876 | 43.7% | 81.6 | 72.3 | 187.7 | 223 |

### 10.2 Findings

1. **Metal does NOT march more samples than GL — it marches fewer.** Mean 81.1
   vs 86.5 (−5.4), median 72.3 vs 80.3 (−8). The 1.68-2.48x jitter Δ gap is
   NOT a trip-count difference in the real mapper either. Per-sample cost is
   the whole story (consistent with §9.4 on GL; now proven on Metal directly).

2. **Jitter does not change Metal's trip count** (j0/j1 identical, same as GL
   §9.2). j1 mean 81.6 vs j0 81.1 — the phase scatter costs $/sample, not
   iterations.

3. **Metal renders ~36K extra pixels GL discards** (43.7% vs 40.2% coverage).
   All in the 0-16 iter bin (57,128 px vs 0): grazing rays through proxy
   corners/edges with 1-16 samples. GL's ray-box setup rejects these
   (discard/early-out); Metal's variant-0 march runs them. Negligible time
   (short rays) but a real geometric divergence — worth checking if
   `tEnter >= tExit` handling or the proxy-box CTP test differs (H1/H4 side
   effect, not the jitter story).

4. **Harness verdict holds in-app**: the harness's "same trip count" was not a
   harness artifact. The equal-field harness correctly predicted the app's
   per-sample jitter cost model; its GL-side over-charge (§2) remains the only
   discrepancy.

Full data: `JITTER_DUMP.txt` (app GL), this section (app Metal).

## 11. Harness GL A/B knob results (2026-08-20)

Goal: find which app-GL shader structure makes its jitter Δ cheap (~11 ms
interleaved) vs the harness's ~22 ms. All knobs in `jitter_gap_repro.mm`
(compile-time shader-string switches), 3 interleaved rounds each at 1024,
mean Δ = j1−j0.

| knob | GL Δ (ms) | Metal Δ (ms) | samples |
|---|---|---|---|
| baseline (current shader) | 22.3 | 20.0 | 88 / 81 |
| A: `LOD=1` textureLod (H8: kill implicit dFdx/dFdy) | 20.5 | 20.8 | same |
| B: `SPLIT_TF=1` separate color/opacity LUT + a>0 gate (H4) | **18.2** | 21.2 | same |
| C: `WHILE=1` + box-exit + accumulated evalPoint (H4 compiler shape) | 19.1 | 18.8 | same (11 px fewer, box no-op) |

App-side check — `VTK_METAL_TEST_GL_NOTF=1` (skip TF fetches, fixed alpha
0.005, keeps volume fetch + full loop), interleaved pairs at 1024:

| app GL config | j0 (ms) | j1 (ms) | Δ (ms) |
|---|---|---|---|
| baseline (no NOTF) | 27.2 | 37.2 | **10.0** |
| NOTF | 30.2 | 43.5 | **13.3** |

### 11.1 Findings

1. **H8 refuted** — removing the implicit gradient (textureLod) does not move
   the harness GL Δ (20.5 vs 22.3, within noise). The dFdx/dFdy on the
   per-pixel-jittered evalPoint is not the cost.
2. **H4 (split TF) gives only a partial move** — Knob B: 22.3 → 18.2 ms
   (borderline vs noise; B's spread 15.8-20.2 overlaps baseline's 20.6-24.8).
   Knob C (while+box-exit, the app's loop shape) adds nothing (19.1). The
   harness GL cannot be made as cheap as the app's ~11 ms with any loop/TF
   structure.
3. **NOTF does NOT help the app GL** — removing the TF fetches *worsens* Δ
   (10.0 → 13.3). The TF-fetch path is not what makes app GL jitter cheap.
4. **Volume-texture upload path is identical** — GL GL_R8 512x512x1794 vs
   Metal MTLPixelFormatR8Unorm same dims (VTK_METAL_TEST_DUMP_VOLTEX).
   Texture-format A/B is moot.
5. **Net**: the cheap app-GL jitter is not explained by loop shape, TF shape,
   gradient mode, or texture format. The harness over-charge persists (~2x
   app GL Δ) with identical rays, counts, and field; remaining candidates are
   harness-level (CGL context/swap behavior, FBO vs drawable, driver state)
   rather than shader structure.

## 12. Same-rays + app-strip + pass A/B (2026-08-20, continuation of §11)

All knobs env-gated (default-off), 1024, j0/j1 interleaved pairs. Success metric
is Δ = j1−j0 with mean samples logged (must stay ~86–88 or the run is discarded).

### 12.1 §1 — same rays? (can invalidate every A/B so far)

Dumped once each, 1024, j0: first-hit texcoord PPM (app `g_rayOrigin` vs harness
`evalBase`) and step-in-texels PPM (app `g_dirStep*textureSize` vs harness
`evalStep*uTexelCount`). Camera constants verified identical against the app's
`VTK_METAL_TEST_DUMP_UNIFORMS` dump (`in_eyePosObjs`/bounds, `in_sampleDistance`=4,
`in_cellSpacing` (0.834,0.834,0.4), `in_cellToPoint` = harness ctpScale/ctpOffset).

| measure | app GL | harness GL | Δ |
|---|---|---|---|
| first-hit mean XYZ (texcoord) | .3158 .3827 .7754 | .3156 .3826 .7768 | 0.0002/0.0001/0.0013 |
| step mean XYZ (texels) | 2.305 1.698 (Z clamped, −dir) | 2.305 1.699 (Z clamped, −dir) | ~0 |
| first-hit per-pixel | 454,204 covered | 457,875 covered | — |

After correcting the harness PPM's Y-flip: mean |Δ| = 1.0/0.9/3.6 texels,
std ≈ mean → **within 8-bit PPM quantization (1 LSB = 2 texels XY, 7 texels Z)**.
Both march −Z (step Z clamps to 0 in both). **Verdict: rays match.** Same line
through the anisotropic brick; the "identical march" claim survives.

### 12.2 §2 — strip the app toward the harness (invert the A/B)

| app GL config | j0 (ms) | j1 (ms) | Δ (ms) | mean samples |
|---|---|---|---|---|
| baseline | 28.2±1.0 | 39.3±0.5 | 11.0±1.5 | 86.5 |
| `GL_NODEPTH` (no depth fetch/discard, AABB term) | 27.4 | 37.7 | **10.3** | 86.5 |
| `GL_NOCLIP` (skip AdjustSampleRangeForClipping) | 32.5 | 47.8 | 15.3 | 86.5 |
| `GL_NOBOX` (drop 6-way texMin/Max break) | — | — | — | 183 (**discarded**) |
| `GL_MINIMAL` (all three + NOTF) | — | — | — | 288 (**discarded**) |
| `GL_NOTF` (TF fetches → fixed alpha) | 30.2 | 43.5 | 13.3 | 87.6 |

1. **NODEPTH does not move Δ** (~10.3 vs baseline 11.0). The depth-sampler
   fetch / discard / pass split is **not** the hiding factor.
2. **NOCLIP makes it worse** (Δ 15.3, j0 base +4 ms at constant 86.5 samples):
   removing the clip prologue changes the composer codegen/occupancy, and the
   jitter tax rises toward the Metal plateau — but not to it.
3. **NOTF keeps samples** (87.6 vs 86.5) — the +3.3 ms is a real H4 effect
   (removing TF fetches raises jitter cost), bounded at ~3 ms, consistent with
   harness split-TF (22→18).
4. **NOBOX/MINIMAL break terminate** (samples 183/288, ray walks past the box
   exit to the far plane) — discarded per protocol.

### 12.3 §3 — pass / FBO structure on the harness

| harness GL config | GL Δ (ms) | Metal Δ (ms) | samples |
|---|---|---|---|
| baseline (this session) | 31.3 | 20.0 | 88 |
| `DEPTH=1` (depth FBO attachment + dummy depth fetch at FS start) | 28.8 | 20.0 | 88 |

The dummy depth texture fetch + depth attachment does not collapse harness GL
Δ toward the app's ~10 ms. Pass structure is **not** the hiding factor either.

### 12.4 Net

With §1 confirming identical rays, and every shader-structure/pass knob
(LOD, split-TF, while+box, NODEPTH, NOCLIP, DEPTH, NOTF) failing to move app GL
below ~10 ms or harness GL toward it, the residual ~8–12 ms gap is **not encoded
in the loop text, TF shape, gradient mode, texture format, or pass structure**.
Per the plan, the next step is GL texture/FBO *state* (wrap/filter/immutable/
swizzle/max-level/compare) or occupancy profiling (Instruments) — not more GLSL.

## 13. V31 back-edge exit ported to the app — NEUTRAL (2026-08-21)

The divergent_tail root cause (MSL mid-body-exit CFG loses to GLSL->Air under
data-dependent trip counts) was ported to the production march as
`VTK_METAL_TEST_DOEXIT=1` → `VolumeFeature_MarchDoExit` (1u<<29) →
`fc_doExit` (function_constant 31): the baseline loop reshaped into a do-while
with every exit folded into the back-edge, `maxSteps > 0` entry guard, gated to
the lean composite config (blendMode 0, no crop/mask/blanking/rect/tf2d/
independent/RGBA/LA), baseline text untouched (separate `else if` branch).

Parity: `METAL_ITER` dumps byte-identical AND color renders byte-identical vs
baseline at the 1024 bench config.

| paired j0 Δ (DOEXIT−base) | value | verdict |
|---|---|---|
| oblique 2048 | +0.88 ± 0.94 ms | neutral |
| 1024 | +0.27 ± 0.45 ms | neutral |
| 800 | −0.25 ± 0.12 ms | neutral |
| j1 probe 2048 (JITTER=1, IGN=0) | −0.37 ± 0.18 ms on ~70 ms | neutral |

Baseline M/GL j0 reproduced the frozen metric exactly (1.054 ± 0.034 at 2048).
Verdict: same as the jitter_gap_repro port — the repro's codegen deficit does
not transfer to the production march's instruction mix (interleaved TF taps +
prefetch-ahead). Baseline j0 at 1024/800 measured ~1.05 this session (doc had
1024 "tied"); session-sensitive.

## 14. Jitter dose-response: the tax ∝ phase-diversity magnitude (JSCALE)

New probe `_padCropFlags[1]` (`VTK_METAL_TEST_JSCALE=s`): the per-pixel phase
offset becomes `mix(1, noise, s)`, shrinking lane spread toward the coherent
j0 lattice while keeping the noise tap, ALU and trip counts identical.
Validated: s=1 byte-identical to native j1; s=0 byte-identical color AND
per-pixel traversal to real j0 (`JITTER=0`).

2048/SD4 raw, interleaved order-alternated:

| config | Metal jitter Δ |
|---|---|
| native (s=1) | +23.74 ± 1.44 ms |
| s=0.5 | +13.50 ± 0.17 ms |
| s=0.25 | +6.82 ± 0.41 ms |
| GL reference | +11.27 ± 0.65 ms |

**Metal's tax scales near-linearly with phase-spread magnitude; GL sits on
Metal's own curve at s≈0.45.** The noise tap/ALU contribute ~nothing (s→0 ≡
j0 cost). `optContents=YES` refuted as a fix: Δ doubles (+50.8 ± 1.1 ms;
lag_repro's swizzle read-tax amplified by scatter). Quality: any s<1 converges
toward the no-jitter image (mean|d| ~4.4/255 vs GL j1 — same magnitude as
turning jitter off), so JSCALE is diagnostic-only, never shippable.

## 15. NOPREFETCH refuted; NEAREST convergence pins trilinear-z

- **NOPREFETCH** (`_padCropFlags[2]`, drop the prefetch-ahead pipeline so the
  march issues one fetch per iteration like GL's composed loop): j0 neutral,
  Δ +22.48 vs +23.23 ms — refuted.
- **NEAREST interpolation both backends** (`VTK_METAL_TEST_GL_NEAREST=1`,
  1 texel/sample): GL Δ +13.83, Metal Δ +17.69 → **M/GL j1 = 1.02, tied**
  (Metal faster at nearest-j0: 30.3 vs 33.3). Linear: +9.90 vs +23.23 (1.29).

**The entire pre-RG8 2× lives in hardware trilinear-z handling under
cross-lane phase scatter.** Not trip counts, not loop CFG, not TF structure,
not pass structure, not prefetch, not derivatives (`level(0)` already),
not layout flag (§14). Counter-intuitively GL's nearest Δ *exceeds* its linear Δ
(+13.8 vs +9.9): more taps made GL cheaper — the sampler feeds z-pairs from
storage differently than Metal's uncompressed private layout does.

## 16. Production configuration: Metal already wins everywhere measured

With minmax acceleration ON (the app default path), 2048/SD4/slabs=1:

| config | j0 | j1 | jitter Δ |
|---|---|---|---|
| GL | 39.62 ± 0.32 | 51.25 ± 0.42 | +11.63 |
| Metal mv9 + minmax | 24.79 ± 0.06 | 28.92 ± 0.15 | +4.12 |
| **Metal mv0 + minmax** | **25.12 ± 0.11** | **26.60 ± 0.12** | **+1.48** |

M/GL j0 0.63, j1 0.52–0.56. The acceleration removes exactly the scattered
samples the tax feeds on. Note for the planned TEMP-REPRO revert: **mv0+minmax
ties mv9 at j0 and wins at j1 on this study** — do not blindly revert to 9.

## 17. RG8 pair-packed slices ported to the app — flips the sign (RG8 knob)

`VTK_METAL_TEST_RG8=1`: volume repacked R=slice 2z / G=slice 2z+1 over a
halved-depth RG8Unorm texture (both upload functions patched; the live one is
`UpdateVolumeTexture`), shader reconstructs trilinear-z from ~1.25 XY-bilinear
pair-taps (`sampleVolumeScalarRG8Pair`, function_constant 32, feature bit
1u<<30). Two bugs found and fixed en route:

1. **fp32 ulp swallows numerator-baked nudges**: at pairs=897 one ulp is
   ~6e-5 texels; a center offset baked into `(p+0.500002)/pairs` rounds away
   and floor() flips to the neighboring pair at full weight. Fix: post-divide
   nudge `(p+0.5)/pairs + 1e-3/pairs` (18× above ulp, bleeds 0.1% of the
   adjacent pair).
2. **`ctpScale`/`ctpOffset` derive from `volumeTexture.get_depth()`** — after
   repack that is the PAIR count, silently re-basing every ray's z lattice
   (found via a fixed-coordinate probe that proved the sampler exact while
   the march diverged). Fix: `texelCountZ = depth × 2` under fc_volRg8.

Image parity after both fixes (j0, 512²): mean|d|=0.0178/255, >1LSB 0.0065%
of pixels, max Δ2. Probe sweep z=0.05..0.95 (both parities, all fz):
base ≡ RG8 exactly.

2048/SD4 oblique final matrix (interleaved, order-alternated):

| config | j0 | j1 | jitter Δ |
|---|---|---|---|
| GL | 40.10 ± 0.36 | 52.83 ± 1.40 | +12.73 |
| Metal 3D trilinear | 43.26 ± 0.07 | 66.28 ± 0.18 | +23.02 ± 0.20 |
| **Metal RG8 pair-tap** | **33.20 ± 0.39** | **43.07 ± 0.75** | **+9.87 ± 0.49** |

M/GL j1 **0.82** — Metal beats GL by 18% on the cell where it previously lost
1.29×. Visual exports indistinguishable (/tmp/vis_*, /tmp/vis2k_*; GL-vs-RG8
mean|d|=0.066/255 ≈ GL-vs-native 0.064).

## 18. RG8 regression matrix — NOT shippable as default

Same protocol swept across sampling density and orientation:

| cell (2048²) | base j0 | RG8 j0 | RG8-vs-base j0/j1 |
|---|---|---|---|
| oblique SD4 | 43.31 | 33.56 | **−22.5% / −34.3%** ✅ |
| oblique SD0.5 | 118.03 | 159.50 | **+35.1% / +36.3%** ❌ |
| axial z SD4 | 11.43 | 15.54 | +36.1% / +4.7% ❌ |
| coronal y SD4 | 9.54 | 14.02 | +46.9% / +47.2% ❌ |
| sagittal x SD4 | 9.65 | 14.00 | +45.1% / +45.2% ❌ |

Plain 3D Metal already beats GL in all those cells (M/GL j0 0.69–0.76, j1
0.68–0.82; SD0.5 matches SLAB_BENCHMARKS §7's 0.74×). The pair-tap pays only
where the march is DRAM-bandwidth-bound under heavy scatter; issue/latency-bound
cells eat the extra ALU and dual taps as pure overhead — matching
divergent_tail's small-frame warning. Decision: **keep `VTK_METAL_TEST_RG8`
as a documented opt-in knob; do not default it.**

## 19. Azimuth sweep: the pre-RG8 lag is a continuum (2026-08-21)

CAM_AZ orbits the camera around world-Y over the fixed oblique base
(2048/SD4/raw/jitter, blue-noise):

| AZ | GL Δ | Metal Δ | M/GL j0 | M/GL j1 |
|---|---|---|---|---|
| 0° | +9.6 | +18.3 | 0.90 | 1.08 |
| 45° | +2.8 | +2.8 | 0.91 | **0.92** |
| 90° | +4.9 | +7.1 | 1.06 | 1.11 |
| 135° | +23.6 | **+43.8** | 0.95 | **1.32** |
| 180° | +10.2 | +19.3 | 0.93 | 1.13 |
| 225° | +2.9 | +2.5 | 0.95 | **0.94** |
| 270° | +6.5 | +7.1 | 1.14 | 1.14 |
| 315° | +23.3 | **+43.8** | 0.89 | **1.28** |

No oblique view escapes the tax (best case ties at 0.92); peaks come in
opposite pairs (135/315 slow, 45/225 fast) — sign-independent, axis-dependent:
the slow azimuths are the ones where rays cross slice planes steepest,
matching the distinct-z-pair-per-warp model of §14/§15.

## 20. Harness GL knobs: context profile + immutable storage — refuted

Chasing why harness-GL pays +22–26 ms while app-GL pays +10–12 ms on proven-
identical rays (§12), two remaining state candidates were tested in
`jitter_gap_repro.mm` (new env knobs `GL41`, `GLSTORAGE`; single runs,
2048/SD4):

| harness config | GL j0 | GL j1 | GL Δ |
|---|---|---|---|
| baseline (3.2-core, mutable) | 47.63 | 71.22 | +23.60 |
| GL **4.1 core** profile | 43.42 | 70.23 | +26.82 |
| **immutable** storage (`glTexStorage3D`) | 45.70 | 71.08 | +25.38 |
| both | 60.96 | 98.86 | +37.90 (worse) |

Both refuted (Metal inert throughout ✓). The app-GL cheapness now survives:
loop text, TF layout, gradient mode, LOD, split-TF, NODEPTH, NOCLIP, NOTF,
depth/FBO pass structure, context profile, storage class. Remaining candidates
are surface/drawable presence (NSOpenGLView vs headless CGL+FBO), blending
state, uniform plumbing (UBO vs glUniform), or driver-internal caching keyed
on something we have not enumerated. See HANDOFF (§22).

## 21. Mechanism synthesis (2026-08-21)

Proven: the tax is per-sample cost inflation under cross-lane phase diversity
(identical trip counts; ∝ spread via JSCALE); it lives in trilinear z-slice-
pair handling (NEAREST ties; RG8 pair-tap beats GL); it is view-angle
dependent via slice-crossing rate; absent in every accelerated config measured.

Inferred (strongly constrained): GL's opaque 3D tiling keeps warp footprints
line-coherent under sub-voxel scatter; Metal's uncompressed layout doesn't and
its swizzled alternative taxes reads of CT data. Same sampler silicon, different
feeds.

Unknowable from source: which driver layer owns the difference — that requires
Apple. NOTE: `jitter_gap_repro` alone cannot be the report exhibit (there
Metal WINS, 0.89×); the gap claim needs the app binary at CAM_AZ 135/315
(1.28–1.32×) with matched fields, plus the harness as mechanism exhibit
(dose-response, NEAREST convergence, RG8 flip).

## 22. HANDOFF — next work: replicate app-GL performance in the harness

Goal: make harness-GL's jitter Δ reach app-GL's ~+11 ms (it sits at +22–26 ms
on identical rays/field/trip counts, §12.1). Success would (a) yield the
missing minimal Apple-report repro boundary, (b) prove the tax is driver-state,
not physics, and possibly (c) hint at an MSL-reachable equivalent.

Refuted so far (do NOT retry): loop shape (LOD/split-TF/while+box individually),
NODEPTH, NOCLIP, DEPTH attach, NOTF, NOBOX/MINIMAL (break terminate — discard),
context profile 3.2↔4.1, mutable↔immutable storage, optContents, prefetch
removal, DOEXIT CFG, JSCALE<1 (changes the image).

Untested candidates, ranked:

1. **Drawable/window-backed rendering instead of headless CGL+FBO.** The app
   renders into an NSOpenGLContext with a real surface; the harness is pure
   offscreen FBO. Test: build a minimal NSWindow+NSOpenGLView variant of the
   harness (hidden window is fine) or CGL with `kCGLPFAWindow` + surface
   attachment; rerun the standard interleaved pairs.
2. **Blending state**: the app composites with `GL_BLEND (ONE,
   ONE_MINUS_SRC_ALPHA)` enabled during the volume pass (read-modify-write
   keeps the RT in tile memory); the harness writes opaque. One-line knob:
   enable blend around the timed draw.
3. **Combined structural knobs**: §11 tested SPLIT_TF (22.3→18.2) and WHILE
   (19.1) separately; the COMBINATION was never run. Try `SPLIT_TF=1 WHILE=1`
   together, then + BLEND.
4. **Uniform plumbing**: VTK may use UBOs/incremental glUniform patterns vs
   the harness's per-draw glUniform* — low prior but free to A/B once the FS
   is instrumented.
5. **Occupancy shaping**: the app FS is much larger (lower occupancy, more
   independent loads in flight per thread — consistent with NOTF making app-GL
   WORSE and split-TF helping harness). If 1–4 fail, try padding the harness
   FS with dead-but-unremovable independent work (uniform-fed ALU chains +
   dummy bound-texture reads outside the hot path) to emulate app register
   pressure, and see if Δ falls toward +11.
6. **Instruments/GPU counters on both contexts** at AZ 315: DRAM read
   amplification would confirm/refute the line-coalescing model directly.

Protocol reminders: interleaved order-alternated j0/j1 pairs, mean samples
must stay ~86–88 (discard otherwise), same-session GL references, battery
state affects absolute ms only. Raw logs this session: /tmp/appv31, /tmp/jsmat,
/tmp/matrix, /tmp/azsweep, /tmp/rg8t, /tmp/np, /tmp/near, /tmp/pzsweep.

TEMP debug edits still in the tree when this section was written (flagged
TEMP in comments; revert before landing anything production-facing):
METAL_ITER return encodes fzDebug/parity into G/B channels; probe early-return
in fragment_volume_main gated on _padCropFlags[2] (reuses NOPREFETCH env);
`[march] fc_doExit/fc_volRg8` and `[RG8]` stderr lines (env-gated).

## 23. HANDOFF ablation complete (2026-08-22): all state candidates refuted; protocol upgraded

§22's ranked candidates were implemented as env knobs in `jitter_gap_repro`
and ablated at 1024/SD4 under a tightened protocol (`WARMUP=30` default,
`ROUNDS` order-alternated interleaved rounds with mean±sd, `SELECT`/`ONLYGL`
per-cell fresh-process runs, full table in jitter_gap_repro/RESULTS.md):

| candidate | verdict |
|---|---|
| 1. drawable/window-backed context (`SURFACE=1`) | refuted (+24.5±4.9) |
| 2. blending ONE/1−SRC_ALPHA (`BLEND=1`) | REFUTED, worse (+38.3) |
| 3. combined structural knobs (`SPLIT_TF=1 WHILE=1`) | no stacking (+17.2) |
| 4. uniform plumbing via UBO (`UBO=1`) | refuted (+23.4) |
| 5. occupancy shaping (`PAD=1`) | refuted (+24.6) |
| clip-prologue port (`CLIP=1`, byte-identical image) | refuted (+20.1) |
| duty cycle / DVFS (`GAPMS`, fresh-process per cell) | refuted (+18.7 in-process; +20.6±1.0 fresh-process) |
| per-frame camera orbit (`AZSTEP=0.1`, app bench does this) | +11.2±1.5 BUT confounded by ~47% sample shedding across the sweep; coverage-preserving doses show nothing. NOT evidence. |
| **warm-up contamination (from issue_tax R3, commit 8dd24927a5)** | **CONFIRMED & FIXED**: WARMUP 5→30 halves Δ (+25.9±10.7 → +18.2±2.8) and cuts variance 4x |

Live re-anchor same hour: app binary j0 29.32 / j1 37.98 → Δ +8.66 @1024;
fresh-process harness j0 35.5±0.1 / j1 55.9±1.1 → Δ +20.6±1.0. The
harness now measures GL as reproducibly as Metal — the residual is real,
structural, and NOT: JIT/warm-up, surface, blending, uniforms, storage,
profile, pass/depth structure, loop/TF shape, clip prologue, occupancy
padding, thermal/duty cycle.

Also root-caused this session: `appgl_parity`'s march truncates to ~1
sample/ray (dirStep oversized ~4x → terminatePointMax median 21 iters);
its earlier "Δ +1.5 ms" was an artifact of a shortened march and is
retracted. Fix its matrix chain before using it as a composition probe.

Next steps, in order:
1. Fix appgl_parity's VS/object-space chain until coverage ≈40% and
   meanIter ≈86; then its verbatim composed FS is the valid test of
   whether shader composition alone carries the cheapness.
2. If composition is exonerated: inverse-transplant — render the lean
   harness FS inside the app process (TEMP bench mode in
   TestMetalGLVisualComparison). +11 there ⇒ process-level driver state
   (Metal coexistence is the leading suspect); +20 ⇒ mapper internals.
3. Instruments GPU counters on both contexts (DRAM read amplification
   would settle the line-coalescing model of §21 directly).

## 24. RESOLVED (2026-08-22 night): the composed shader carries the cheap jitter

The "next steps" of §23 completed within hours:

1. `appgl_parity` root-caused and FIXED: its `kInvProj[11]` constant was
   transcribed as 0 instead of -1 (verified against a fresh
   glGetUniformfv dump via VTK_METAL_TEST_DUMP_UNIFORMS). The broken
   inverse-projection w-row collapsed g_rayTermination near each ray's
   entry, capping every march at g_terminatePointMax ~21 samples (vs
   ~86-225 real). All earlier timings from that tool were artifacts.
2. With the fix, appgl_parity — the VERBATIM composed FS in the identical
   headless CGL+FBO harness context — now reads:

| @1024/SD4 | GL j0 | GL j1 | jitter Δ |
|---|---|---|---|
| app binary (live) | 29.32 | 37.98 | +8.66 |
| fixed appgl_parity | 25.53±1.7 | 33.24±0.9 | **+7.71 ±2.1** |
| lean harness FS (WARMUP=30) | 35.5 | 55.9 | **+20.6** |

Absolutes reconcile: live frame total minus parity GPU-only ≈ 3.8 ms of
VTK CPU overhead on both j0 and j1.

**CONCLUSION: §22 is resolved.** The app-vs-harness GL difference was in
the shader composition all along; every context/state/duty-cycle knob had
to be refuted first only because no one had ever run the composed shader
in the harness with a correct march. The §21 "driver-state" suspicion is
dead: two shaders, one context, one geometry, 13 ms of jitter delta.

§21's inferred mechanism needs revision: it is not GL-vs-Metal tiling
mystique — the LEAN GL shader pays the full tax while the COMPOSED GL
shader does not, so the tax is a property of the per-sample instruction
mix under phase scatter, on both APIs (Metal pays an analogous premium:
lean-harness Metal Δ ~+20 vs app Metal Δ ~+18.5 at 1024).

Remaining refinement (optional): bisect which ingredient(s) of the
composed FS transfer the cheapness into the lean FS (conditional color
fetch, R32F opacity LUT shape, scale/bias, sign-gated OOB test,
currentT/tPM break form, or their combination). INCR=1 (incremental
position accumulation alone) already refuted (+22.8).

## 25. HANDOFF — next session: port the composed-shader cheapness to Metal (2026-08-22 night)

### 25.1 State

Replication target met on the GL side (§24): fixed `appgl_parity` =
verbatim composed FS in a headless CGL+FBO harness reads jitter Δ
**+7.71 ±2.1** @1024/SD4 vs live app **+8.66** (which includes ~3.8 ms of
VTK CPU/frame); the lean reconstruction FS pays **+20.6** in the identical
context. Mechanism (revised §21): per-sample instruction mix under
cross-lane phase scatter, living in trilinear z-slice-pair handling
(NEAREST ties M/GL 1.02 §15), ∝ phase spread (JSCALE §14), modulated by
slice-crossing rate (azimuth §19).

Tooling ready: `appgl_parity` (valid reference, probes uDbg 14/21/22/23/
25/26), `jitter_gap_repro` knob harness (WARMUP/ROUNDS/SELECT protocol,
fresh-process runs repeatable to ±1 ms), in-app `METAL_ITER`/`GL_ITER`
PPM instrumentation, `VTK_METAL_TEST_RG8`/`DOEXIT` feature bits.

### 25.2 Corrections to carry forward (do not cite stale claims)

1. **§16 is not a fair backend win**: GL has no minmax implementation.
   "Metal already wins in production" means Metal has an algorithmic
   acceleration GL lacks. The fair comparison is raw-march vs raw-march:
   composed-GL Δ ≈ +8–12 ms vs Metal raw Δ ≈ +18–25 ms at 800–2048 px →
   the Metal deficit is REAL and still open on equal footing.
2. **RG8 is not a general answer (§18)**: −22.5%/−34.3% ONLY on oblique
   SD4; +35% at SD0.5, +36–47% on axial/coronal/sagittal SD4. Opt-in
   only; do not propose it as a global default again.

### 25.3 Goal

Make Metal's raw-march jitter Δ reach composed-GL level (~+9–12 ms at
the 800–2048 oblique cells) by identifying the compositional
ingredient(s) that kill the tax and mirroring them in
`MetalShaders.metal`.

### 25.4 Plan A — ingredient bisection (GL harness first, then MSL)

Each probe = one env knob in `jitter_gap_repro.mm`
(`loopShape`/`stepBody` string variants), measured against fixed
appgl_parity's +7.7 reference at 1024/SD4, ROUNDS≥2:

1. **OPACITY-GATED COLOR FETCH**: skip the color LUT tap when a≈0
   (composer behavior). Prior: SPLIT_TF moved 22.3→18.2 (§11) — right
   family, partial alone; expect stacking.
2. **TF LUT SHAPE**: separate R32F 1024-wide opacity table (+ RGB32F
   color table) instead of one RGBA8 256 fetch.
3. **SCALAR FLOW**: per-sample scale/bias (`in_volume_scale`) +
   vec4 swizzle path around the sample.
4. **BREAK-FORM COMBO**: sign-gated texMax/texMin OOB test +
   `g_currentT >= g_terminatePointMax` break pair INSIDE the loop
   (CLIP=1 prologue alone was inert; the in-loop break form was never
   isolated).
5. **COMBINATIONS** of whatever moves (§22 lesson: singles understate
   stacking).

Success criteria per probe: GL Δ falls materially below ~18 toward ≤10;
mean samples stay 86–88; image byte-identical to baseline (footprint +
optional RAYS=1 dump diff). Discard runs violating sample counts.

Then port the winning structure to `MetalShaders.metal` behind a
function_constant (patterns: `fc_doExit` bit 1u<<29, `fc_volRg8`
bit 1u<<30): byte-parity via METAL_ITER dumps + color renders, then
interleaved j0/j1 pairs at 1024 & 2048 oblique PLUS the §18 regression
cells (axis views, SD0.5).

Expectation calibration: full success closes the gap to composed-GL
level (~Δ+11–13 at these cells); it does NOT make jitter free. If an
ingredient transfers on GL but not MSL, that itself localizes the
remaining true API difference (sampler feed, §15/§21) — record it.

### 25.5 DEPRIORITIZED — conditional RG8 gating (wrong shape for interactive use)

An earlier draft proposed gating `fc_volRg8` on ray-vs-slice-plane
geometry (the slow azimuths 135°/315° are where RG8 pays, §19).
Deprioritized: the product target is INTERACTIVE rendering, and a
representation flip driven by live camera geometry is exactly the wrong
shape for that —

- engaging RG8 means REPACKING the volume texture into halved-depth
  RG8Unorm (a ~450 MB blit); toggling it mid-orbit stalls frames;
- function-constant flips rebuild PSOs at the worst moment;
- frame cost that swings with view angle yields unpredictable frame
  budgets precisely while the user is interacting — worse than a uniform
  moderate cost.

Status: keep `VTK_METAL_TEST_RG8` as a manual opt-in knob only. Any
future revival needs a STATIC per-dataset/per-deployment justification
(a dataset class where plain 3D loses in every orientation), never a
runtime gate. Plan A (§25.4) is the only active path.

### 25.6 Refuted — do NOT retry

Everything in §22/§23 tables, plus: INCR=1 (+22.8), BLEND=1 (+38),
SURFACE=1, UBO=1, PAD=1, CLIP-prologue alone, GAPMS/duty-cycle,
AZSTEP-as-evidence (~47% sample shedding confound), optContents,
JSCALE<1 (changes the image), NOPREFETCH, DOEXIT-on-production (neutral),
NOBOX/MINIMAL (breaks termination), kInvProj-era appgl_parity timings
(retracted).

### 25.7 Protocol reminders

WARMUP=30 now default-baked; ROUNDS≥2 order-alternated;
`SELECT=g0`/`SELECT=g1` fresh-process runs are the tightest GL
measurement (±1 ms); always verify march stats (covered ≈40%, meanIter
≈86) before trusting any delta; same-session references; battery state
affects absolute ms only. TEMP debug edits still in tree (METAL_ITER G/B
encoding, probe early-return gated on `_padCropFlags[2]`, `[march]`/
`[RG8]` stderr lines) — revert before any production-facing landing.
INVOCATION: app-bench wrappers MUST use `eval "env $C ... $B"`-style
expansion — zsh does not word-split unquoted `$VARS` and silent
default-config runs fake "convergence"; see §29 for the verbatim recipe
and canary numbers.

## 26. RESOLVED (2026-08-22 night): transposed volume upload kills the Metal jitter tax — root cause found, production port byte-parity

### 26.1 Root cause

**Metal's private 3D texture tiling is strongly axis-biased.** With the slice
axis (the 1794-deep extent) as texture DEPTH, trilinear z-pair fetches under
per-pixel jitter phase scatter pay a huge DRAM tax; uploading the volume
x<->z TRANSPOSED (slice axis in the texture's x extent) collapses it. This
is the mechanism §21 could only infer: not GL-vs-Metal "tiling mystique"
(revised §24), not request shape — pure memory layout.

Proof ladder (all same-day, interleaved, order-alternated):

1. `jitter_gap_repro` new knob `TRANSPOSE=1` (CPU transpose + coord swizzle,
   byte-identical renders): Metal @1024/SD4 j0 29.7→16.7, **jitter Δ
   +22.28±1.12 → +1.78±0.18 ms**. Also @2048/SD4 (+23.06→+5.23), SD0.5
   (+7.27→+0.98), SD8 (wins). NO cell regresses.
2. Request-shape restructuring does NOT reach it: `APSTEP` (composed-loop
   port to MSL: two-tap float LUTs + gate + incremental pos + OOB/tPM
   exits) only −2.7 ms; `MANTRI` (manual z-split trilinear: two XY-bilinear
   taps at slice centers + lerp, image max Δ=0 vs baseline) only −1.6 ms.
   MSL is structure-insensitive where GLSL swings 2× (see 26.3).
3. Production port (`VTK_METAL_TEST_VOLTRANSPOSE=1`, fc_volTransposed):
   byte-identical renders (j0 AND j1, max Δ=0) and:

| production config (@2048/SD4 oblique) | j0 | j1 | jitter Δ |
|---|---|---|---|
| baseline mv0 raw | 43.3 | 66.1 | +22.8 |
| **transposed raw** | **20.2** | **22.2** | **+2.0** |
| §17 GL reference (raw) | 40.10 | 52.83 | +12.73 |
| baseline mv0+minmax | 25.14 | 27.18 | +2.04 |
| **transposed mv0+minmax** | **23.94** | **24.67** | **+0.73** |

M/GL j1 on the pathological cell goes from ~1.29 to **~0.42** — Metal beats
GL by 2.4× absolutely, with bit-identical output.

4. Full compass sweep (CAM_AZ 0..315, @1024 raw): transposed beats baseline
   ABSOLUTELY at every azimuth (tr-j1/base-j1 = 0.41–0.62). The tax
   redistributes (worst transposed Δ +9.6 at AZ45/225 — the azimuths that
   were previously cheapest) but every absolute frame time improves.
5. SD0.5 @1024: Δ+0.18, absolutes 71/78 → 38.8/39.0 (1.8× faster).

### 26.2 The GL-side composition hunt (Plan A) — results

The §25.4 bisection ran first and REFUTED the "port the composed structure"
path, which forced the layout experiment:

- Lean-harness singles all cluster ~Δ+17–21 (target ~+8): SPLIT_TF +19.9,
  SCALARFLOW +17.4 (byte-exact), EXITAPP=1 +19.2, TFSHAPE +19.4,
  TFSHAPE+GATE +19.7. Nothing approaches appgl_parity's +7.8.
- Inverse direction (appgl_parity ablations, PAR_* knobs): NOGATE +6.6,
  NOSCALE +8.2, NOINCR +7.2 — gate/scale/incremental-position are inert.
  LEANLOOP (swap loop body for lean mechanics) jumps to +12.6±0.2 — the
  exit/break structure carries real weight ON GL — but ONETAP/FULL variants
  were march-length-confounded (TF_FACTOR=4 in parity's LUT saturates rays:
  meanIter 81.3 vs harness 86.5). Treat those numbers as upper bounds.
- **MSL non-transfer**: the composed structure barely moves Metal
  (APSTEP +19.6, MANTRI +20.7 vs lean +22.3) while GL swings +7.8↔+15.6
  inside one program. The cheapness is substantially a GLSL-codegen
  phenomenon plus a layout effect GL gets for free from its opaque tiling.
  This is the §25.4 expectation-calibration outcome ("if an ingredient
  transfers on GL but not MSL...") — recorded.

NOTE: divergent_tail's earlier "axis permutation: ratio unchanged" (argv[15])
was a CONTROL (volume AND eye/dir permuted together = relabeling); today's
experiment moves DATA while fixing the camera — no contradiction.

Also fixed en route: env collision with TestMetalScenes' pre-existing scene-
level `VTK_METAL_TEST_TRANSPOSE` vtkImagePermute hook → production knob is
`VTK_METAL_TEST_VOLTRANSPOSE`. The two must not be combined.

### 26.3 Implementation (production)

- `vtkMetalGPUVolumeRayCastMapper.h`: `VolumeFeature_VolTransposed = 1u<<31`,
  member `VolumeTextureTransposed`.
- Mapper: CPU blocked x<->z transpose of the U8 single-component staging
  bytes (both upload paths), swapped upDims/upBytesPerRow/upBytesPerImage,
  blit strides keyed on `(rg8Pair || volTransposed)`; featureMask bit at both
  builder sites; `fc_volTransposed` constant; minmax/normals kernel uniforms
  carry `volTransposed` and swizzle their data-space sample coords.
- `MetalShaders.metal`: `fc_volTransposed [[function_constant(33)]]`;
  `sampleVolumeScalar` pre-swizzles `pos=pos.zyx` (covers all interpolation
  variants incl mv1 8-tap; RG8Pair excluded — mutual with transpose);
  texelCount un-swizzled so ctpScale/ctpOffset/evalStep stay original-space;
  rawScalar4 sites swizzle; minmax/normals kernels swizzle via uniform flag.
- Mutually exclusive with `VTK_METAL_TEST_RG8` (pair indexing assumes
  untransposed depth). Upload cost: one blocked CPU transpose (~seconds,
  upload-time only).

TEMP debug instrumentation added (env-gated, inert by default; revert before
any production-facing landing per §25.7): `[TR]`/`[TEXCRE]` stderr prints,
TR_DUMP/TR_GPU slice-dump probes (CPU + GPU readback), PROBE_RAW raw-plane
probe branch, mask hex print in GetOrCreateVolumePipeline.

New harness knobs this session: `TRANSPOSE`, `AZ0` (static camera azimuth;
GL+Metal), `APSTEP`, `MANTRI`, `TFSHAPE`, `GATE`, `SCALARFLOW`, `EXITAPP`,
`FSDUMP`; appgl_parity `PAR_NOGATE/PAR_NOSCALE/PAR_NOINCR/PAR_LEANTF/
PAR_LEANLOOP/PAR_LN_*/PAR_LN_FULL`.

### 26.5 Regression matrix completed (2026-08-22 late night): ONE repeatable regression found

All cells @2048/SD4 raw (MINMAX=0/ACCEL=0/slabs=1/IGN_JITTER=0), interleaved
order-alternated pairs, same camera both arms:

| cell | baseline j0/j1 (Δ) | transposed j0/j1 (Δ) | verdict |
|---|---|---|---|
| oblique SD4 | 43.3/66.1 (+22.8) | 20.2/22.2 (+2.0) | ✅ wins big |
| oblique SD0.5 | 118.4/122.6 (+4.2) | 102.1/102.4 (+0.25) | ✅ wins (RG8 paid +35% here) |
| 4096² SD4 | 66.7/84.8 (+18.1) | 51.7/54.1 (+2.4) | ✅ wins |
| axial z (CAM_AXIS) | 42.1/103.0 (**+60.9**) | 42.7/43.7 (+1.1) | ✅ kills hidden catastrophic tax (2.35× faster j1) |
| coronal y (CAM_AXIS) | 31.4/33.1 (+1.7) | 31.3/32.0 (+0.7) | ✅ tie/better |
| **sagittal x (CAM_AXIS)** | 31.9/32.8 (+0.9) | 30.9/36.2 (**+5.2**) | ❌ **j1 +10%, repeatable** |

The sagittal regression is the predicted tax relocation: marching along
original-X now runs along texture-depth' (512 extent), the disfavored
direction post-transpose — same mechanism as the AZ45/225 redistribution.
Net user-visible cost at worst case: +3.4 ms/frame on x-only views, against
−60 ms on axial and −44 ms on the oblique interactive path. Note also the
baseline's own hidden axial tax (+61 ms!) that no prior benchmark table had
caught (§18's "axial z 11.43 ms" used a different geometry convention).

New env: `VTK_METAL_TEST_CAM_AXIS=x|y|z` (TEMP-DIAG in TestMetalScenes.h,
exact axis-aligned views for §18-style cells).

GL references for the same cells (@2048/SD4 raw, --backend gl, 2 rounds each;
GL has no VOLTRANSPOSE equivalent):

| axis | GL j0 | GL j1 | GL Δ |
|---|---|---|---|
| axial z | 51.47 | 95.41 | **+43.94** |
| coronal y | 41.57 | 44.11 | +2.54 |
| sagittal x | 42.39 | 44.99 | +2.60 |

Three-way j1 standing per axis cell:

| axis | GL | base Metal | tr Metal |
|---|---|---|---|
| axial z | 95.4 | 103.0 (**M/GL 1.08** — was losing!) | **43.8 (0.46)** |
| coronal y | 44.1 | 33.1 (0.75) | **32.0 (0.73)** |
| sagittal x | 45.0 | 32.8 (0.73) | 36.2 (**0.80**, despite regression) |

Even at the sagittal worst case transposed stays 20% under GL; on axial it
converts a LOSING cell into a 2.2× win.

Visual verification (12 exports @2048² jittered, /tmp/png_review/ sheets):
M-transposed ≡ M-baseline pixel-identical at all four views (max Δ=0);
GL-vs-Metal residuals unchanged vs pre-port (oblique mean|d|=0.064 ==
§6.3's exact value). Minmax-ON render parity: near-identical (mean|d|=0.0005,
nine >1LSB px of 480K, max Δ9 — macrocell edge-rounding).

### 26.6 HANDOFF — next work

1. **Default-enable decision**: `VTK_METAL_TEST_VOLTRANSPOSE` needs wider
   dataset coverage before becoming default (other studies/orientations/
   presets). Validation so far is single-dataset IMRToraceAddome U8.
2. **Sagittal/x-march residual (+10% j1)**: root cause = march along
   original-X runs along texture-depth' post-transpose (relocated tax).
   Options: accept+document (+3.4 ms worst case vs −60 ms axial gain);
   Instruments DRAM counters to confirm directly (§22 item 6, still open);
   ask Apple about depth-extent tiling. Do NOT runtime-gate representations
   (§25.5 reasoning applies).
3. **Variant/config coverage**: all validation ran on mv0 (the TEMP-REPRO
   pin). Other march variants route through `sampleVolumeScalar` so they
   should inherit the fix, but smoke-run mv9 + shading-on configs before
   defaulting. Shading-on exercises the normals-kernel swizzle path
   (code-reviewed, not yet image-diffed).
4. **GPU transpose pass**: the CPU blocked transpose costs seconds at load
   for ~450 MB volumes; a compute-kernel transpose would remove it (nice
   for large datasets / reloads).
5. **RG8 interplay**: mutual exclusion documented (pair indexing assumes
   untransposed depth). With transpose winning obliquely AND axially, RG8's
   original use case is superseded; keep opt-in or consider retirement.
6. **TEMP instrumentation inventory** (env-gated, revert before any
   production-facing landing): `[TR]`/`[TEXCRE]` prints, TR_DUMP/TR_GPU
   slice probes, PROBE_RAW branch + mapper negation, MARCH_DEBUG mask print,
   CAM_AXIS env (useful diagnostics — decide keep vs revert).
7. **Apple-report exhibit**: two uploads, bit-identical renders, 10×
   jitter-delta difference + the axial hidden-tax discovery. The §25 Plan-A
   composition hunt is superseded/moot (kept in git history).


### 26.4 Verdict

The task is closed on equal footing: with matched bytes and matched march,
Metal's raw-march jitter Δ drops from +22.8 to +2.0 ms @2048 oblique —
BELOW composed-GL (+12.7) — and wins or ties every measured cell absolutely
EXCEPT one: sagittal/x-axis views pay a repeatable ~+10% j1 regression
(§26.5) — the relocated tax, bounded at +3.4 ms/frame against −60 ms on
axial and −44 ms on the oblique interactive path. The interactive deficit
was never shader physics: it was a texture-layout choice. Remaining
follow-ups (optional): GPU-side transpose pass instead of CPU; axis-view
(pitch) bench cells; decide whether VOLTRANSPOSE should ship
enabled-by-default after wider dataset coverage; Apple-report exhibit is
now trivial
(two uploads, one bit-identical render, 10× jitter delta difference).

## 27. §26.6 items 1+3 complete (2026-08-22): variant/config + dataset coverage validated

### 27.1 Variant/config coverage (§26.6 item 3) — all pass

All @1024 or as noted, oblique, JITTER=1, IGN_JITTER=0, IMRToraceAddome:

| config | baseline | transposed | parity |
|---|---|---|---|
| mv9 raw (`VTK_METAL_TEST_MARCH_VARIANT=9`, MINMAX off) j1 | 43.20±0.48 | **13.13±0.24** (3.3×) | **byte-identical** |
| mv9+minmax+accel @2048 j1 | 27.18 (§26.1 mv0 base) | 21.73±0.01 | (mv0-tr ref 24.67; mv9 faster) |
| SHADE=1 raw (normals kernel) j1 | 45.08 | **19.52** (2.3×) | **byte-identical** |
| production minmax+accel+SHADE j1 | 13.82 | **9.63** (1.4×) | near-identical: mean|d|=0.0004, 8/1048576 px >1LSB, max 7 (= §26.5's known minmax macrocell edge-rounding, NOT the swizzle) |

The normals-kernel `volTransposed` swizzle (NormalComputeUniforms /
MinMaxComputeUniforms paths) is now image-verified, closing §26.6 item 3's
"code-reviewed, not yet image-diffed" gap. mv9's lean march fetches through
`sampleVolumeScalar` (MetalShaders.metal MV9_FETCH), so it inherits the entry
swizzle — confirmed by byte parity.

New TEMP-DIAG env: `VTK_METAL_TEST_SHADE` in TestMetalScenes.h
(BuildDICOMVolumeScene; ShadeOn + 0.2/0.8/0.3 ADS). The DICOM scene previously
had no shading coverage at all. Env-gated, default off.

### 27.2 Dataset coverage (§26.6 item 1)

Available-study survey (`VTK_METAL_TEST_TR_DUMP` texture dims): of 16 study
dirs under /Users/macair/Public/IMR, only 4 are real 3D volumes — the rest
are 1–3-slice degenerate stacks or unreadable series. Valid set (all CT U8,
512² in-plane, deep slice axis — same anisotropy class):

| dataset @2048/SD4 raw oblique | base j0/j1 (Δ) | transposed j0/j1 (Δ) |
|---|---|---|
| IMRToraceAddome ×1794 (§26.5 ref) | 43.3/66.1 (+22.8) | 20.2/22.2 (+2.0) |
| **IMRTA4** ×1714 | 44.41/66.35 (+21.9) | 19.34/21.98 (+2.6) |
| **FGFegatoIMR** ×1084 | 19.17/32.89 (+13.7) | 13.54/15.33 (+1.8) |
| **ACOvaioIMR** ×1654 | 36.00/49.34 (+13.3) | 15.04/18.58 (+3.5) |

Jittered-render parity: pixel-identical on all three new studies
(mean|d|=0, max 0).

Axis cells replicate the §26.5 pattern (IMRTA4 @1024 j1, renders identical):
x/sagittal 15.86→19.12 (**+20%**, the relocated tax); y/coronal
15.59→11.83 (−24%); z/axial 69.58→15.95 (**−77%**). Net worst case +3.3 ms
on x-only views against −54 ms axial and −14 to −45 ms oblique.

TF diversity: BoneSkinII preset (multi-color TF, app runs it shadeless)
@1024 on IMRTA4: 16.91→6.40 (2.6×), render pixel-identical.

### 27.3 Verdict

§26.6 items 1 and 3 are done. VOLTRANSPOSE is correct and strictly faster on
every 3D dataset available (4/4 studies), every orientation except the
documented bounded x-march tax, both TF presets, shading on/off, minmax
on/off, mv0/mv9. Nothing regresses beyond the known sagittal pattern.
Scope caveat for any default-enable decision: validation remains
single-modality (CT U8), single in-plane size class (512²), one GPU
(M1-class Apple Silicon). Multi-component/independent (RGBA/LA) inputs and
16-bit uploads route through different upload functions — code-reviewed but
not exercised here.

Remaining §26.6 items unchanged: GPU transpose pass (load-time CPU cost),
sagittal residual accept/document (now replicated on a second dataset),
RG8 retirement consideration, TEMP inventory sweep, Apple-report exhibit.

## 28. §26.6 item 4 done (2026-08-22): GPU transpose pass — 13× faster load, byte-identical

### 28.1 Implementation

`VTK_METAL_TEST_GPU_TRANSPOSE=1` (env-gated; CPU repack remains the default
until wider validation): skips the CPU blocked x<->z transpose entirely.
Staging keeps the ORIGINAL layout (plain memcpy of scalars); a one-pass
compute kernel `volume_transpose_xz` (MetalShaders.metal) reads the staging
bytes (`device const uchar*`, coalesced along source-x) and writes the
swapped-dims volume texture directly at `(z,y,x)` — no transStorage
allocation, no blit, no intermediate texture. One `MTLComputeCommandEncoder`
on the usual upload command buffer; same queue ordering as the blit it
replaces. R8Unorm round-trip is byte-exact. Pipeline cached in
`TransposeComputePipeline` (header member); texture created with
`ShaderRead|ShaderWrite` under the knob. Threadgroup 8³ (4³ fallback via
`maxTotalThreadsPerThreadgroup`).

### 28.2 Measured (IMRToraceAddome 512×512×1794, 470 MB)

| stage | CPU repack | GPU kernel |
|---|---|---|
| transpose itself | 1049–1087 ms ×2 uploads | **73–86 ms** per upload |
| wall-clock speedup | 1× | **~13×** |

Frame times identical to the CPU-repack path in every cell re-measured:
oblique @2048 raw j1 21.99 ms/f (= §27's 21.98 reference), axial z @2048
42.53, production minmax+accel+shade @1024 9.18 (vs 9.93 CPU-repack arm).

### 28.3 Verification ladder

1. Full-volume FNV-1a of the uploaded texture (TR_DUMP + TR_GPU shared
   storage readback) ≡ CPU `transStorage` hash byte-for-byte
   (`5082437852801311472` both paths). Staging hash ≡ dicom.u8 ground truth.
2. Render parity: jittered raw @1024 PNG **byte-identical** to CPU-repack;
   production minmax+accel+shade PNG **byte-identical**.
3. Occupancy scan along the transposed axis (TEMP-DIAG) uniform ~0.87 as
   expected from source content.

### 28.4 Debug war story (for future Metal compute work)

The first working version produced partially-corrupted output whose root
cause cost most of a session to isolate: **the kernel declared
`dst [[texture(1)]]` while the dispatch bound `[enc setTexture:tex
atIndex:0]`** — writes to an unbound slot are silently discarded and the
command buffer still reports Completed/error=0. Symptoms were actively
misleading: the readback showed *banded partial data* (~87% occupancy in
[x',512..1024)+[1664,1794) slabs, zeros elsewhere) that did NOT change when
the kernel body was swapped for a position-encoded constant writer — i.e.,
the visible content was unrelated to anything the kernel wrote. Lessons:
(a) always cross-check binding indices against the [[texture(n)]]/
[[buffer(n)]] attributes; (b) an unbound write target fails silently, not
with a validation error; (c) MSL printf never emitted here, so "no printf"
was NOT evidence of non-execution; (d) the legacy getBytes selector on this
runtime is `getBytes:bytesPerRow:bytesPerImage:fromRegion:mipmapLevel:
slice:` — the modern `from:` variant raises unrecognized-selector on
AGXG14GFamilyTexture.

Also fixed en route: the pre-existing TR_GPU probe's `fromRegion:` call had
the same latent selector bug (never exercised before); TR_DUMP now prints a
full-volume FNV-1a for both CPU and GPU paths plus a per-slab occupancy
profile (all env-gated TEMP-DIAG).

Verdict: §26.6 item 4 closed. The knob is ready to fold into the
default-enable decision once dataset coverage questions (§27.3 caveats)
are settled — GPU transpose removes the only remaining runtime cost of the
transposed representation (seconds → tens of ms at load), so
`VOLTRANSPOSE` + `GPU_TRANSPOSE` together have no load-time penalty worth
gating on.

## 29. Benchmark invocation guide (2026-08-22): the zsh word-splitting trap

A full afternoon was lost to phantom "regressions" caused purely by shell
quoting: **zsh does not word-split unquoted `$VARS`**. An invocation like
`env $C ... $B` (with `C`/`B` config strings) passes the ENTIRE string as ONE
argv element, so `env` sets a single garbage variable whose NAME contains
spaces and the app silently runs with DEFAULTS — minmax/accel ON, default
sample distance, adaptive slabs. Those runs look plausible (~24-33 ms
oblique j0) and layout-insensitive (baseline ≈ transposed in every cell,
jitter Δ ≈ 0 or even negative), i.e. exactly a "convergence" that does not
exist. The doc's frozen numbers only reproduce when the config string is
re-parsed by the shell (`eval`) or written out literally.

Rules for any wrapper around this bench:

1. Build the command with `eval "env $C ... $B"` (re-splits correctly in zsh
   AND bash), or write every VAR=VAL literally on the command line.
2. `env` consumes leading `-`-prefixed args as ITS OWN options — every app
   flag must come AFTER the binary path ("env: --backend: No such file or
   directory" means you got this wrong).
3. Sanity-check the engaged config at least once per session:
   `VTK_METAL_TEST_MARCH_DEBUG=1` prints `[march] mask=0x…` per pipeline
   (VolTransposed = bit 31 = 0x80000000; slab = bit 28), and METAL_ITER/GL_ITER
   march stats must read meanIter ≈ 81-86 @1024 before trusting any delta.
4. Extraction: the bench row is `scene GLms GLfps Metalms Metalfps M/GL` —
   GL ms/f is `$2`, Metal ms/f is `$4`.

Verbatim, fully working example (repository root, zsh/bash; pathological
cell @2048/SD4 raw oblique, blue-noise jitter, slabs=1):

```
C="VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 \
VTK_METAL_TEST_MINMAX=0 VTK_METAL_TEST_ACCEL=0 VTK_METAL_TEST_NUM_SLABS=1 \
VTK_METAL_TEST_IGN_JITTER=0"
B="build_macos_metal/bin/vtkMetalGLVisualComparison --bench --backend metal \
--scene DICOMVolume --dicom /Users/macair/Public/IMR/CTIMR/IMRToraceAddome \
--frames 30 --reps 1 --size 2048x2048"

# untransposed baseline (explicit — VOLTRANSPOSE defaults ON since §29 era):
eval "env $C VTK_METAL_TEST_JITTER=0 VTK_METAL_TEST_VOLTRANSPOSE=0 $B"
eval "env $C VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_VOLTRANSPOSE=0 $B"
# transposed (GPU kernel is also default-ON; =0 forces the CPU repack):
eval "env $C VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_VOLTRANSPOSE=1 $B"
```

Expected @2048/SD4 raw oblique (§26.5 anchor): baseline j0/j1 ≈ 43/66
(Δ +22 ms), transposed ≈ 20/22 (Δ +2 ms). Axis cells via
`VTK_METAL_TEST_CAM_AXIS=x|y|z` inside the same eval pattern (axial-z
baseline j1 ≈ 103 is the canary — if it reads ~75-80 with baseline ≈
transposed everywhere, your env vars did not reach the app).

## 30. Orientation-general VOLTRANSPOSE: argmin-dims depth selection (2026-08-22 night)

Closes the anisotropy-assumption hole in the unconditional x<->z swap: on a
volume whose long axis is NOT Z, blind swapping would put the long extent
into texture DEPTH and recreate the catastrophe class (measured here: oblique
j1 64.7 ms/f and axial-z 102.5 @2048 raw on the standard study rendered
identity-layout — exactly what a long-axis-X dataset would suffer).

### 30.1 Design

`VolumeTransposedAxisDepth(dims)` policy: transpose iff Z is strictly greater
than an in-plane extent; the SHORTER in-plane axis becomes texture depth
(ties prefer X, matching every §26.5/§27 cell); Z already shortest → identity
upload, no repack at all. `VTK_METAL_TEST_VOLTRANSPOSE_AXIS=x|y|z` forces the
orientation for A/B. Y-depth support end-to-end:

- Shader: `fc_volTransposedY [[function_constant(34)]]` +
  `volumeFetchSwizzle()` (.xzy for Y-depth); texelCount un-swizzle 3-way;
  rawScalar4 sites via the helper; minmax/normals kernels take an axis CODE
  (0/1/2) in the existing `volTransposed` uniform; `volume_transpose_xz`
  kernel takes a mode constant (buffer 3) and writes `(z,y,x)` or `(x,z,y)`.
- Mapper: per-axis upDims/strides/CPU-repack formulas at both upload sites;
  `VolumeTextureAxisDepth` member; PSO key extended with
  `featureMaskExtra` (= axis code — the featureMask bit alone cannot
  distinguish orientations); `fc_volTransposedY` bound from the member.
  Feature-mask gates now read the POLICY result recorded by this frame's
  upload, not the raw env (a policy no-op must NOT select swizzled pipelines).
- RG8 mutual exclusion unchanged.

### 30.2 Validation

Byte parity @1024 oblique jittered vs the long-standing X-depth reference:
X-depth and Z-identity **byte-identical**; auto policy picks X on this
dataset and matches. Y-depth: ±1 LSB on 29 of 1,048,576 px (max Δ1) —
hardware trilinear depth-split rounding order differs when original-Y plays
the depth role; visually nil, documented as near-exact.

Orientation × view matrix (@2048/SD4 raw j1 ms/f):

| view | depth=x | depth=y | depth=z (identity) |
|---|---|---|---|
| oblique | **23.0** | 24.5 | 64.7 ☠️ |
| cam-x march | 36.4 | **34.6** | 33.1 |
| cam-y march | **32.7** | 36.3 | 32.9 |
| cam-z march | 45.5 | **43.7** | **102.5 ☠️** |

The tax relocates symmetrically with the chosen depth axis (cam-x pays under
X-depth, cam-y under Y-depth, bounded +1–3.5 ms) while both transposed
orientations kill the catastrophic identity-layout cells (−41 ms oblique,
−57–59 ms axial). Argmin picks X here → optimal interactive cell.

Production config (minmax+accel+shade): fresh runs are **byte-identical** to
the morning's pre-generalization references on BOTH arms; transposed-vs-base
residual stays at §26.5's minmax edge-rounding (8 px >1LSB, max 7); GL
cross-check mean|d|=0.0498 (§6.3 equivalence class).

### 30.3 Debug post-mortem (two session traps, one real bug)

The AXIS=z A/B initially appeared nondeterministic ("sometimes honored,
sometimes transposes as axis-x"). Root cause was MY SITE-2 BUG, not shell
quoting: the UpdateVolumeTexture branch set `volTransposed = true`
unconditionally and its if/else treated orientation code 0 as X-depth — so a
policy no-op still built an X-transposed texture while the feature gates
(correctly reading the 0) compiled UNSWIZZLED pipelines against it = garbage.
The [TRPOLICY] print (env-gated) settled it in one run: `axis env 'z' -> 0`
followed by a transpose print. Site 1 had the guard; site 2 didn't. Lesson:
when a decision function gains a third outcome, audit EVERY consumer's
assumption that it returns the old domain.

§29's zsh traps bit twice more en route (unsplit `$CFG` in a for-loop →
"Unknown argument"; a missing `SAMPLE_DISTANCE` in one hand-built invocation
made SD default 0.5 and faked a 2.4-ms-mean render "regression" — caught by
re-running both arms fresh and diffing against same-day references before
believing any delta). The §29 protocol (eval-env wrappers + fresh-pair A/B +
same-session references) converted a would-be wild-goose chase into a
10-minute bisection.

Verdict: §26.6 item 1's scope caveat is answered for orientation; VOLTRANSPOSE
is safe-by-construction across volume shapes now, not just across the
datasets that happen to be Z-longest.

## 31. Session 2026-08-22 night: mv0/mv9 recheck on post-transpose code + NEW ISSUE — tr×mm catastrophe at fine SD

Context: HEAD `d1724e10de` plus `3e352942ff` (mapper `VolumeMarchVariant()`
now reads `VTK_METAL_TEST_MARCH_VARIANT` LIVE per feature-mask build instead of
once-per-process, and test-vtk-metal's macOS Rendering menu gained an mv9
toggle, ⌘⌥V — same setenv+rebuild pattern as the transpose toggle). All runs:
battery-powered (absolute ms carry session drift; relative pairs measured
back-to-back), single 30-frame rounds per cell with interleaved
order-alternated / combo-rotated arms; verification passes replicated key
cells within ~2–5%. Logs: `/tmp/mvmatrix/`, `/tmp/mvmatrix_az/`,
`/tmp/recheck_tr_mm/`. Sanity canaries matched frozen anchors (2048 oblique j1
21.0–24.0 vs §30.2's 23.0; MARCH_DEBUG mask prints confirmed engaged config).

### 31.1 mv0(default) vs mv9, MINMAX=0/ACCEL=0, blue-noise field

**Resolution × SD × axis grid (96 cells @800/1024/2048):**

Oblique is a dead tie everywhere (mv9/def 0.97–1.03) at both SDs and both
jitters; jitter Δ ≈ 0 for BOTH variants (±0.7 ms) — VOLTRANSPOSE has erased
the old interactive-path tax entirely.

Axis views SD4 (def/mv9): x-march = the ONLY losing class anywhere (mv9 wins
j0 −7…−13% but LOSES j1 +3…+14%; ≤~2 ms); y/z mv9 wins −11…−32%. SD0.5 axes:
mv9 wins every cell by 33–45% (z@2048 j0 342→189). Jitter Δ ≈ 0 on y/z/x at
SD0.5; x@SD4 still carries the relocated tax (+5–13 ms, BOTH variants).

**Azimuth compass @1024 SD4** (`CAM_AZ` = offset over base Azimuth 30°; NOTE
the bench-default oblique takes the else branch = net −30°, so it is NOT AZ0):

| AZ | j0 def/mv9 | j1 def/mv9 | Δj def/mv9 |
|---|---|---|---|
| 0° | 19.31/**14.01** | 22.66/**14.81** | +3.4/+0.8 |
| 45° | 8.44/8.43 | 18.47/**17.99** | +10.0/+9.6 |
| 90° | 10.97/10.66 | 18.58/**16.81** | +7.6/+6.2 |
| 135° | 18.09/**10.78** | 21.02/**11.82** | +2.9/+1.0 |
| 180° | 19.62/**13.96** | 22.34/**14.92** | +2.7/+1.0 |
| 225° | 8.54/8.35 | 18.75/18.60 | +10.2/+10.3 |
| 270° | 10.66/10.37 | 18.44/**16.88** | +7.8/+6.5 |
| 315° | 17.24/**10.52** | 20.01/**11.31** | +2.8/+0.8 |

Two regimes matching §26's relocated-tax model (opposite-phase pairs): cheap
class (0/135/180/315) — def pays +2.7–3.4 ms jitter, mv9 only +0.8–1.0 → mv9
wins 25–43% on j0 AND j1; expensive class (45/225 worst, 90/270 mid) — both
pay the full tax → ties. mv9 never loses a compass cell. SD0.5 compass: mv9
wins every azimuth ~34–40%.

**Verdict**: reverting TEMP-REPRO to 9 is net-positive on the raw path —
neutral where it used to be contested (oblique), large wins elsewhere; sole
regression is jittered sagittal views (+3–14% j1). §16's "do not blindly
revert" caution was written pre-transpose against minmax-on numbers and does
not extend to today's code. The earlier "oblique tie" was a single-camera
artifact (net −30° sits in the cheap class near a variant-neutral spot).

### 31.2 NEW ISSUE — tr×mm interaction @2048² SD0.5 (default variant)

48 runs: VOLTRANSPOSE{0,1} × MINMAX/ACCEL{0,1} × jitter{0,1} × orientations
(oblique default, CAM_AZ 45/135, CAM_AXIS x/y/z). ms/f below = mean(j0,j1);
**jitter Δ ≤ 2 ms in ALL 24 cells** (jitter is free at SD0.5 post-transpose):

| view | tr0+mm0 | tr0+mm1 | tr1+mm0 | tr1+mm1 |
|---|---|---|---|---|
| oblique def | 120.4 | 128.5 | **101.6** | 122.3 |
| AZ45 | 108.3 | **99.2** | 99.3 | ☠️ 172.2 (+74%) |
| AZ135 | 122.9 | 134.9 | 104.3 | **99.5** |
| axis x | 229.4 | **213.4** | 231.9 | ☠️ **421.4 (+97%)** |
| axis y | 224.6 | **213.9** | 227.3 | 220.3 |
| axis z | 329.3 | 416.1 | 315.7 | **238.0** |

Findings:

1. **Transpose alone (mm0)**: −15% oblique/AZ135, −4% z, neutral x/y.
   Consistent with §26.5/§30 anchors at this SD.
2. **Minmax alone is NOT safe at fine SD**: +8–11% WORSE on oblique/AZ135,
   +26% worse on axial-z (lattice-walk overhead exceeds skipping gains when
   marches are dense), while helping az45/x/y by 4–10%. Prior minmax
   validation (§16, §27.1) was SD4-only and never re-checked post-VOLTRANSPOSE.
3. **The tr×mm INTERACTION is view-class-dependent and sometimes
   catastrophic**: combined ON is best-in-row at z (238 vs tr1mm0's 316, −25%)
   and AZ135 (99.5), but catastrophic at axis-x (+82% vs tr1mm0) and AZ45
   (+73%). The pathological views are those whose rays travel along / cross
   steeply the transposed depth' axis — prime suspect: the swizzled minmax
   kernel/lattice path (§26.3 code-reviewed only) or lattice-walk fetches
   scattering along the disfavored axis at fine SD.
4. **Robustness ranking @SD0.5**: `tr1+mm0` never loses >~9% anywhere and wins
   the interactive oblique path; the current production default (`tr1+mm1`)
   can hit 170–420 ms frames where 100–240 is achievable in the same cells.

### 31.3 HANDOFF — next work

> STATUS (2026-08-22 late night): items 1 and 2 superseded by §32 — the tr×mm
> catastrophe was a build-side dims bug in `ComputeMinMaxGPU` (lattice built
> from texture extents instead of data dims) and is FIXED; §31.2's
> "minmax helps" cells (AZ45-adjacent az135/z) were corrupted-lattice artifacts.
> The fine-SD minmax penalty itself is real and layout-independent (§32.3).

1. **Root-cause the tr×mm catastrophe** (axis-x / AZ45 @SD0.5):
   - Isolate build vs walk: force the minmax lattice to be built from the
     UNTRANSPOSED data (CPU path or kernel-swizzle off) while keeping the
     transposed volume upload, and vice versa. Whichever arm collapses the
     172/421 ms cells owns the bug.
   - METAL_ITER-style PPM of samples-skipped per pixel per view: confirm the
     walk visits/skips the same macrocells as tr0+mm1 (thrash vs geometry).
   - Instruments GPU counters on the two pathological cells (still-open §22
     item 6): DRAM read amplification would settle scattered-lattice-fetch
     directly.
2. **Map the SD-dependence**: rerun the 24-cell grid at SD4 (expect clean per
   §16/§27) and SD1/SD2 to find whether a threshold exists. If yes: gate
   `SetUseMinMaxAcceleration` off below it (mirroring mv9's MaxBatchWidth SD
   mapping) — a STATIC quality-based gate, NOT runtime view gating (§25.5
   reasoning applies).
3. **mv9 under mm1 at SD0.5** (this sweep ran the default variant only):
   mv9 carries its own fc_minmax lattice walk — does the catastrophe compound,
   transfer, or vanish?
4. **TEMP-REPRO revert decision**: supported for the raw path by §31.1;
   before flipping the default, repeat the compass/axis A/B UNDER the
   minmax-on production config, since §31.2 shows mm changes view-class
   behavior.
5. Protocol unchanged (§29/§25.7): eval-env wrappers, MARCH_DEBUG mask sanity,
   fresh-pair A/Bs, same-session references, verify march stats before
   trusting deltas.

### 31.4 Reproduction (commit ref + verbatim scripts)

**Code state**: results were measured on `d1724e10de` plus the working-tree
mapper change (live `MARCH_VARIANT` env read) minutes later committed as
**`3e352942ff`** — checkout ≥ `3e352942ff`. The live-read edit is timing-inert
for these sweeps (each process sets the env once at launch), so any later
commit without march/minmax/transpose changes reproduces too. Build:

```
./macos_metal_build.sh --resume     # produces build_macos_metal/bin/vtkMetalGLVisualComparison
```

Dataset: `/Users/macair/Public/IMR/CTIMR/IMRToraceAddome` (CT U8 512×512×1794).
Extraction convention (§29): Metal ms/f is `$4` of the `DICOMVolume` row.
The three generator scripts below are embedded verbatim (they lived in
`/tmp/opencode/*.zsh`, which is volatile); save each as a file and invoke as
shown. Battery vs AC changes absolute ms (~±5–10%), never the A/B ordering.

**Script A — mv0-vs-mv9 resolution × SD × axis grid (§31.1, 96 cells).**
Usage: `RESLIST="800 1024 2048" SEEDN=0 zsh mvmatrix.zsh` (one shot) or per-
resolution chunks `RESLIST=800 SEEDN=0`, then `1024 SEEDN=16`, `2048 SEEDN=32`.

```zsh
#!/bin/zsh
emulate -L zsh
BIN="/Users/macair/Public/VTK-Source/mine/VTK/build_macos_metal/bin/vtkMetalGLVisualComparison"
DICOM="/Users/macair/Public/IMR/CTIMR/IMRToraceAddome"
OUT="/tmp/mvmatrix"; mkdir -p "$OUT"; RES_FILE="$OUT/results.txt"
[[ -f "$RES_FILE" ]] || touch "$RES_FILE"

bench() {
  local R=$1 SD=$2 AX=$3 J=$4 V=$5
  local extra=""
  [[ "$V" == "mv9" ]] && extra="VTK_METAL_TEST_MARCH_VARIANT=9 "
  [[ -n "$AX" ]] && extra="${extra}VTK_METAL_TEST_CAM_AXIS=$AX "
  local C="VTK_METAL_TEST_SAMPLE_DISTANCE=$SD VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_MINMAX=0 VTK_METAL_TEST_ACCEL=0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=$J $extra"
  local tag="r${R}_sd${SD}_${AX:-obl}_j${J}_${V}"
  local log="$OUT/$tag.log"
  eval "env $C $BIN --bench --backend metal --scene DICOMVolume --dicom $DICOM --frames 30 --reps 1 --size ${R}x${R}" >"$log" 2>&1
  local ms=$(grep DICOMVolume "$log" | awk '{print $4}')
  [[ -z "$ms" ]] && ms="PARSE_FAIL"
  printf "%s %s\n" "$tag" "$ms" >> "$RES_FILE"
  echo "$tag $ms"
}
cell() {
  local R=$1 SD=$2 AX=$3 J=$4
  CELLN=$((CELLN+1))
  if (( CELLN % 2 )); then bench $R $SD $AX $J def; bench $R $SD $AX $J mv9
  else bench $R $SD $AX $J mv9; bench $R $SD $AX $J def; fi
}
CELLN=${SEEDN:-0}
for R in ${RESLIST:-800 1024 2048}; do
  for SD in 4 0.5; do
    for AX in "" x y z; do
      for J in 0 1; do cell $R $SD "$AX" $J; done
    done
  done
done
echo "DONE CELLN=$CELLN"
```

**Script B — azimuth compass @1024 SD{4,0.5} × jitter × variant (§31.1).**
Usage: `SEEDN=0 zsh azmatrix.zsh` (64 runs; §31.1 table quotes the SD4 rows).

```zsh
#!/bin/zsh
emulate -L zsh
BIN="/Users/macair/Public/VTK-Source/mine/VTK/build_macos_metal/bin/vtkMetalGLVisualComparison"
DICOM="/Users/macair/Public/IMR/CTIMR/IMRToraceAddome"
OUT="/tmp/mvmatrix_az"; mkdir -p "$OUT"; RES_FILE="$OUT/results.txt"
[[ -f "$RES_FILE" ]] || touch "$RES_FILE"

bench() {
  local R=$1 SD=$2 AZ=$3 J=$4 V=$5
  local extra="VTK_METAL_TEST_CAM_AZ=$AZ "
  [[ "$V" == "mv9" ]] && extra="${extra}VTK_METAL_TEST_MARCH_VARIANT=9 "
  local C="VTK_METAL_TEST_SAMPLE_DISTANCE=$SD VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_MINMAX=0 VTK_METAL_TEST_ACCEL=0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=$J $extra"
  local tag="r${R}_az${AZ}_sd${SD}_j${J}_${V}"
  local log="$OUT/$tag.log"
  eval "env $C $BIN --bench --backend metal --scene DICOMVolume --dicom $DICOM --frames 30 --reps 1 --size ${R}x${R}" >"$log" 2>&1
  local ms=$(grep DICOMVolume "$log" | awk '{print $4}')
  [[ -z "$ms" ]] && ms="PARSE_FAIL"
  printf "%s %s\n" "$tag" "$ms" >> "$RES_FILE"
  echo "$tag $ms"
}
cell() {
  local R=$1 SD=$2 AZ=$3 J=$4
  CELLN=$((CELLN+1))
  if (( CELLN % 2 )); then bench $R $SD $AZ $J def; bench $R $SD $AZ $J mv9
  else bench $R $SD $AZ $J mv9; bench $R $SD $AZ $J def; fi
}
CELLN=${SEEDN:-0}
for AZ in 0 45 90 135 180 225 270 315; do
  for SD in 4 0.5; do
    for J in 0 1; do cell 1024 $SD $AZ $J; done
  done
done
echo "DONE CELLN=$CELLN"
```

**Script C — tr×mm × orientation grid @2048 SD0.5 (§31.2, 48 runs).**
Chunks as run: `ORIENTS=":: :45: :135:" SEEDN=0`, `ORIENTS="x::" SEEDN=3`,
`ORIENTS="y:: z::" SEEDN=5` (SEEDN only sets combo-rotation parity — thermal
ordering, not results; any values work).

```zsh
#!/bin/zsh
emulate -L zsh
BIN="/Users/macair/Public/VTK-Source/mine/VTK/build_macos_metal/bin/vtkMetalGLVisualComparison"
DICOM="/Users/macair/Public/IMR/CTIMR/IMRToraceAddome"
OUT="/tmp/recheck_tr_mm"; mkdir -p "$OUT"; RES_FILE="$OUT/results.txt"
[[ -f "$RES_FILE" ]] || touch "$RES_FILE"

bench() {
  local AX=$1 AZ=$2 TR=$3 MM=$4 J=$5
  local extra=""
  [[ -n "$AX" ]] && extra="${extra}VTK_METAL_TEST_CAM_AXIS=$AX "
  [[ -n "$AZ" ]] && extra="${extra}VTK_METAL_TEST_CAM_AZ=$AZ "
  local C="VTK_METAL_TEST_SAMPLE_DISTANCE=0.5 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 \
VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 \
VTK_METAL_TEST_MINMAX=$MM VTK_METAL_TEST_ACCEL=$MM \
VTK_METAL_TEST_VOLTRANSPOSE=$TR VTK_METAL_TEST_GPU_TRANSPOSE=$TR \
VTK_METAL_TEST_JITTER=$J $extra"
  local tag="${AX:-${AZ:+az$AZ}-obl}"
  tag="sd05_2048_${tag}_tr$TR\_mm$MM\_j$J"
  local log="$OUT/$tag.log"
  eval "env $C $BIN --bench --backend metal --scene DICOMVolume --dicom $DICOM --frames 30 --reps 1 --size 2048x2048" >"$log" 2>&1
  local ms=$(grep DICOMVolume "$log" | awk '{print $4}')
  [[ -z "$ms" ]] && ms="PARSE_FAIL"
  printf "%s %s\n" "$tag" "$ms" >> "$RES_FILE"
  echo "$tag $ms"
}
orient() {
  local AX=$1 AZ=$2
  local combos=("0 0" "0 1" "1 0" "1 1")
  if (( CELLN % 2 )); then combos=("1 1" "1 0" "0 1" "0 0"); fi
  for J in 0 1; do
    for c in $combos; do
      set -- $=c
      bench "$AX" "$AZ" $1 $2 $J
    done
  done
  CELLN=$((CELLN+1))
}
CELLN=${SEEDN:-0}
for spec in ${ORIENTS:-"::" ":45:" ":135:" "x::" "y::" "z::"}; do
  IFS=':' read -r AX AZ _ <<< "$spec"
  orient "$AX" "$AZ"
done
echo "DONE CELLN=$CELLN"
```

**Spot-check canaries** (single cells, expect ±5% battery permitting):

```
B="build_macos_metal/bin/vtkMetalGLVisualComparison --bench --backend metal \
--scene DICOMVolume --dicom /Users/macair/Public/IMR/CTIMR/IMRToraceAddome \
--frames 30 --reps 1 --size 2048x2048"

# oblique best cell ~101 ms (§31.2): transposed, minmax OFF:
eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=0.5 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 \
VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_MINMAX=0 \
VTK_METAL_TEST_ACCEL=0 VTK_METAL_TEST_VOLTRANSPOSE=1 VTK_METAL_TEST_GPU_TRANSPOSE=1 \
VTK_METAL_TEST_JITTER=0 $B"

# catastrophe canary ~420 ms (§31.2 axis-x tr1+mm1): add VTK_METAL_TEST_CAM_AXIS=x,
# MINMAX=1 ACCEL=1:
eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=0.5 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 \
VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_MINMAX=1 \
VTK_METAL_TEST_ACCEL=1 VTK_METAL_TEST_VOLTRANSPOSE=1 VTK_METAL_TEST_GPU_TRANSPOSE=1 \
VTK_METAL_TEST_JITTER=0 VTK_METAL_TEST_CAM_AXIS=x $B"

# compass cheap-class canary ~11.8 ms j1 (§31.1 az135 mv9): @1024, MINMAX off,
# JITTER=1, CAM_AZ=135, MARCH_VARIANT=9
```

If the axial-z identity-layout baseline reads ~75–80 instead of ~315–330, or
oblique reads layout-insensitive ~24–33, your env did not reach the app
(§29 failure signature).

## 32. RESOLVED (2026-08-22 late night): tr×mm catastrophe root-caused and fixed — GPU minmax lattice was built in the wrong coordinate space

### 32.1 Root cause

`ComputeMinMaxGPU` derived its lattice dimensions from the **volume TEXTURE
extents** (`volTex.width/height/depth`). Under VOLTRANSPOSE those are the
axis-SWAPPED extents, while both other ends of the pipeline live in DATA space:

- `volume_compute_minmax` expects `u.volDim*` in data dims — it computes
  `pos = (voxel+0.5)/volDims` and applies the data→texture swizzle
  (`.zyx`/`.xzy`) itself. Fed texture dims, the normalized→texel conversion
  rescales by mismatched extents (512↔1794 ⇒ ~3.5×), so every cell's min/max
  came from scattered wrong voxels, and occupancy was written on a swapped
  grid.
- The fragment walk (`mmPos * mmDimF`, sample at `mmPos`) and the
  `MinMaxInfo` uniform (`BuildGlobalPerBlockData` ← `MinMaxDims`) consume the
  lattice in DATA space.

Net under transpose: scrambled occupancy + grid geometry stretched ~3.5×/0.29×
per axis. Skipping decisions became view-dependent garbage: catastrophic at
fine SD where marches are dense (axis-x +97%, AZ45 +74% — §31.2), mildly
wrong elsewhere. Identity layout masked everything (texture dims == data
dims). The CPU minmax path (`UpdateMinMaxTexture`) always used input dims and
was never affected.

### 32.2 Fix

In `ComputeMinMaxGPU`: recover DATA dims by inverse-swapping the texture
extents per `VolumeTextureAxisDepth` (1=X-depth: swap[0]<->[2];
2=Y-depth: swap[1]<->[2]), cross-checked against `input->GetDimensions()`
(vtkErrorMacro + fallback on mismatch). mmDims, uniforms, `MinMaxDims`
member, lattice texture creation and dispatch all become data-space; kernel
and walk untouched. The crosscheck is skipped under `VolumeRg8PairActive()`
(RG8 keeps axisDepth==0 but halves texture depth; its minmax conventions were
never defined — pre-fix behavior preserved verbatim there). New env-gated
diagnostics: `[TRMM]` (lattice geometry per build) and `[TRMMCACHE]`
(cache-check inputs) on `VTK_METAL_TEST_TR_DUMP`/`TR_BENCH`.

### 32.2.1 Safety sweep (final build)

- **Byte-parity tr-mm ≡ identity-mm**: max Δ=0 at az135 SD0.5, oblique SD4,
  AND axial-z SD0.5 (4.2M px each) — same lattice semantics regardless of
  layout, the fix's core claim.
- **Timing anchors** (final build vs earlier-today/session references):
  SD4 tr+mm j0/j1 24.94/25.75 (ref 23.94/24.67, battery drift);
  catastrophe canaries 229.6 axis-x / 107.3 AZ45 (pre-fix 421/172; raw arms
  227/97).
- **Identity layout untouched**: code path identical when axisDepth==0;
  today's ABBA id-arm timings reproduced §31.2-class values.
- **RG8 guard**: RG8=1 MINMAX=1 runs clean, lattice over pair dims exactly
  as HEAD (no error spam); RG8×minmax remains semantically undefined but
  unchanged.

### 32.3 Results (@2048² SD0.5 raw-class config unless noted)

Catastrophes eliminated (fixed build, verified twice):

| view | pre-fix tr+mm | post-fix tr+mm | post-fix tr+raw |
|---|---|---|---|
| axis-x | ☠️ 421.4 | **212.8** | 227 |
| AZ45 | ☠️ 172.2 | **98.6** | 96.6 |

SD4 production anchor unchanged: tr mv0+minmax j0/j1 = 23.65/24.49 (§26.1
reference 23.94/24.67).

**New parity proof**: post-fix, tr+mm vs identity-mm renders are
BYTE-IDENTICAL at az135 SD0.5 (max Δ=0, 4.2M px) — same data-space lattice ⇒
same skipping decisions regardless of layout. Pre-fix this comparison could
not even be formulated.

ABBA-balanced minmax deltas at SD0.5 (order-alternated rounds; drift-immune):

| view / layout | raw | +minmax | Δ |
|---|---|---|---|
| oblique, transposed | ~99.5 | ~123.2 | +24 ms (+24%) |
| az135, transposed | ~102 | ~129 | +27 ms (+26%) |
| axial-z, transposed | ~360–385ᵈ | ~470ᵈ | ~+90 ms |
| az135, identity | ~122.9 | ~134.7 | +12 ms |
| axial-z, identity | ~356 | ~460 | ~+100 ms |

Post-fix there is NO residual transpose-specific interaction: minmax at fine
SD costs about the same in both layouts (§31.2 finding #2 is the whole
remaining story — poor skip efficiency when marching is DRAM-bound; DS=2
cells over CT fine structure stay mostly non-empty after dilation).

### 32.4 The pre-fix "minmax helps" cells were corrupted-lattice artifacts

The §31.2 cells where tr+mm looked GOOD (az135 99.5, z 238) were re-measured
by building HEAD and fixed binaries side-by-side (/tmp/vtkPreFixMM,
/tmp/vtkFixedMM) and running interleaved order-alternated pairs:

- The pre-fix binary reproduced §31.2's numbers exactly (az135 96.7–97.2,
  z 232–234) in BOTH round orderings — binary identity, not thermal position.
- Its images are still near-identical to correct renders (pre vs raw:
  mean|d|=0.029, 0.18% px >1LSB, max 36): the scrambled field marked many
  low-opacity-content cells empty, skipping near-invisible contributors —
  accidentally fast because wrong. Not shippable behavior; the fix removes
  the speedup along with the catastrophes.

### 32.5 Protocol additions (§29-class traps hit today)

- **Stale-binary trap**: after `git stash → build → stash pop`, the build dir
  still held the PRE-FIX binary while sources were fixed; a full matrix ran
  against it before the discrepancy surfaced. Verify a code marker whenever
  binaries are swapped/stash-danced: `strings -a BIN | grep MARKER` plus
  `cmp` against saved copies. Saved reference binaries caught it in one step.
- **Arm-ordering bias**: the first sweep ran mm0 before mm1 in every view;
  several "+cost" readings needed ABBA confirmation. axz additionally drifted
  317→386 ms within one sweep — paired within-round deltas stayed valid,
  absolutes did not.

### 32.6 HANDOFF status after this session

§31.3 item 1 (root-cause tr×mm) DONE — build/walk isolation resolved it as a
build-side dims bug; no Instruments needed. Remaining:

1. **Fine-SD minmax gate**: minmax hurts at SD≤~0.5 in BOTH layouts now
   (+12..+100 ms by view). Map SD-dependence at SD1/SD2 (item 2 of §31.3)
   and consider a STATIC sample-distance gate for
   `SetUseMinMaxAcceleration` — layout-independent decision now.
2. **mv9-under-mm1 at SD0.5** (§31.3 item 3) still untested.
3. TEMP-REPRO revert decision (§31.3 item 4): mm behavior is now
   layout-independent, so compass/axis A/Bs under mm-on reduce to the raw
   ones plus the known fine-SD penalty; repeat once under the production
   config before flipping defaults.
4. Latent same-class bug noted: `volume_compute_normals` +
   `EnsureGradientNormalTexture` also mix texture dims with data-space math
   (mapper:2926 region) — currently DEAD CODE (`UsePrecomputedNormals=false`,
   no setter calls anywhere); fix before ever enabling precomputed normals
   under transpose.
5. Partitioned-volume path (`usePartitions` + GPU minmax) reuses the same
   kernel with per-block uniforms — audit its dim conventions too if
   partitions+transpose ever combine.

Logs: /tmp/trmm_fix/ (first matrix, ordering-biased — superseded),
/tmp/trmm_ab/ (binary A/B), /tmp/trmm_abba/ + /tmp/trmm_abba_tr/ (clean ABBA),
/tmp/trmmimg*/ (parity PNGs). Reference binaries: /tmp/vtkPreFixMM,
/tmp/vtkFixedMM.

## 33. HANDOFF — next work: attack the minmax penalty at fine sample distance (2026-08-22 late night)

Decision: pursue the MECHANISM (make the walk cheap or skipping smarter)
before falling back to a static SD-gate. This section scopes that work.

### 33.1 State of knowledge (all post-§32-fix, layout-independent)

Penalty being attacked (@2048² SD0.5, tr layout, ABBA): +24 ms oblique,
+27 az135, ~+90 axial-z (identity: +12 az135, ~+100 axial-z). At SD4 minmax
WINS everywhere (§16/§26/§27). The lattice is now provably correct
(tr-mm ≡ id-mm byte-identical, §32.2.1).

Mechanism model to falsify: per-sample walk overhead × poor skip yield.
- Walk cost per sample: clamp + `int3(mmPos*mmDimF)` + cell-change compare;
  on crossing, one R8 lattice fetch; when empty, the overshoot skip math.
- Yield: DS=2 cells (GPU path, sd<1.5) over CT fine structure stay mostly
  non-empty after dilation ⇒ few skips ⇒ pure tax. Axial-z hurts most because
  its rays traverse the full 1794-slice depth = longest marches = most
  samples taxed.

Headroom exists: the corrupted-lattice accident (§32.4) skipped
near-zero-alpha contributors with ≤0.18% px >1LSB image change and up to
40% time savings (az135/z) — an upper bound on what smarter emptiness could
harvest legitimately.

### 33.2 Plan, ranked

1. **Decompose before building** (METAL_ITER-style probe, ~1 session):
   extend the march instrumentation to count per pixel: samples visited,
   cell crossings, lattice fetches, empty-cell skips taken; dump PPMs for
   mm1 vs raw trip counts at SD{0.5,1,2,4} × {oblique, AZ45, axis-z}.
   Splits the penalty into walk-overhead vs missed-skip-opportunity and
   picks which of 3/4 matters. Uniform channel: `_padCropFlags[3]` is free
   ([0]=METAL_ITER, [1]=JSCALE, [2]=probe); feature bits 1u<<28..31 taken.
2. **ε-contribution emptiness tier** (the principled accident):
   today a cell is empty iff NO scalar in [cellMin,cellMax] has nonzero
   opacity — one barely-opaque voxel pins it solid. Change build-time
   semantics to "max achievable per-cell alpha < ε ⇒ skippable":
   `volume_compute_minmax` already has cellMin/cellMax; iterate the TF table
   range [idxMin,idxMax] (≤256 entries, GPU-cheap) for the max opacity and
   encode it (second channel RG8, or 3-state in R8). Walk test unchanged.
   ε calibration protocol: sweep ε; per view record Δ(ms) AND image diff vs
   mm0 reference; find the knee where image stays sub-visible while Δ→≤0.
   MUST be env-gated (`VTK_METAL_TEST_MM_EPS=s`) + feature bit; document as
   an approximation (today's U8 emptiness is exact); never default without
   the visual matrix incl. §18-style cells.
3. **Cheaper walk** (image must stay byte-identical):
   - Amortized DDA cell stepper (Amanatides–Woo t-next-axis) replacing
     per-sample position→cell conversion. Unknown: MSL/Air codegen behavior
     — inspect disassembly first (divergent_tail precedent).
   - Two-level lattice: coarsened summary texture (one fetch covers 8³
     cells) for early-out deep in uniform regions; tiny extra memory/build.
   - Verify codegen isn't hoisting the empty-cell skip math into the hot
     path when curCellEmpty is false — read Air before blaming the algorithm.
4. **DS retune at fine SD**: DS=2 was tuned @400px pre-transpose ("DS 2 best
   at sd<=1" comment at ComputeMacrocellDownsample). Re-validate DS 2 vs 4
   at 2048 post-transpose at SD0.5/1 — fewer cell crossings vs coarser
   skips; env-gate the function temporarily for the A/B.
5. **Fallback**: static SD-gate for SetUseMinMaxAcceleration if mechanism
   work stalls — but only after item 1 quantifies the crossover (SD1/SD2
   cells are still unmeasured).

### 33.3 Success criteria

mm1 ≤ mm0 (Δ ≤ 0 ± noise, ABBA) at every view class at SD0.5 AND no SD4
regression; byte-identical images for walk-shape fixes; for the ε tier, the
calibrated bound documented with the same rigor as §17's parity ladder.
Mean march stats sanity (§29) before trusting any delta.

### 33.4 Refuted / do NOT retry

Runtime or view-dependent gating of representations (§25.5 reasoning);
JSCALE<1-class phase shrinking (changes the image); citing §31.2's
"minmax helps" rows as evidence (corrupted-lattice artifacts, §32.4);
sequential-run deltas (§32.5); RG8×minmax as a baseline (semantically
undefined, §32.2 guard note).

### 33.5 Reproduction anchors

Scripts A/B/C of §31.4 remain valid patterns (eval-env wrappers mandatory);
ABBA order-alternation required for any mm0-vs-mm1 claim (§32.5). Canaries
on the fixed build: oblique SD4 tr+mm j1 ≈ 24–26; catastrophe cells axis-x
≈ 213–230 / AZ45 ≈ 98–107 (raw arms 227/97); identity-layout code inert
when VOLTRANSPOSE=0. [TRMM]/[TRMMCACHE] diagnostics gate on TR_DUMP/TR_BENCH.


## 34. RESOLVED (2026-08-23): fine-SD minmax penalty fixed — two-level occupancy summary (block leaps)

§33's plan executed. The penalty is gone: minmax+blocks now BEATS raw at
SD0.5 in 4/6 view classes and ties the other two, while SD>=1.5 stays
byte-identical HEAD.

### 34.1 Probe first — §33.1's mechanism model was half wrong

MM_PROBE probe (_padCropFlags[4] + METAL_ITER; R*32=visits, G*8=lattice
crossings, B*32=skipped steps; encode = count/(scale*255) — the first
attempt divided by the scale alone and saturated at 64 counts, costing two
matrix reruns): at SD0.5 oblique the walk SKIPS 569 of 665 raw samples
(85% yield — NOT "poor skip yield" as §33.1 assumed) yet costs +24 ms,
because crossings/ray ~= 395 ~= 0.92/iteration: DS=2 cells turn empty-space
traversal into a per-cell grind of R8 fetches + skip math + full main-loop
iterations (CTP tests, resync) per ~1.7-step hop. Overhead, not yield.

### 34.2 DS retune (cheap lever, tested first)

VTK_METAL_TEST_MM_DS forced the macrocell downsample at fine SD:
DS=4 cut the penalty to +3..-45 by view; DS=8 flipped most cells to
winning (axz -74). But no single static DS dominated (axx/axy preferred 4,
obl/az135/axz preferred 8) — leap granularity vs skip yield tradeoff.
Superseded by 34.3, which decouples them.

### 34.3 Two-level occupancy summary (the fix)

`VTK_METAL_TEST_MM_BLOCKS=1`: a coarse R8 texture marks whole 8³-cell
blocks of the DILATED fine lattice whose cells are ALL empty
(`volume_reduce_minmax_blocks` kernel, dispatched after dilation on the same
encoder). The baseline walk then derives its block from the CELL index
(`newCell/8`, clamped), fetches the block texel at its CENTER once per
block change, short-circuits the fine fetch while block-empty, and LEAPS to
the block's true boundary plane instead of hopping cell edges. Output is
unchanged by construction (block-empty => every covered cell empty => the
composited sample set is identical); landing points stay on the step lattice.

Three bugs found on the way, each caught by bisects the protocol made cheap:

1. **Block index from mmPos product** (`floor(mmPos*blkDim)`): agrees with
   `cellIndex/8` only when fineDim/blockSize is an exact integer. On z
   (897/8=112.125) the mappings disagree almost everywhere -> blocks marked
   empty covered solid cells (12,129 px of runtime disagreements measured
   with a counter probe; CPU-side texture consistency check read 0
   violations because the TEXTURES were consistent — the WALK's mapping was
   wrong). Fix: integer-divide the cell index. Lesson: when a coarse grid
   tiles a fine grid, derive coarse indices FROM fine indices, never from
   independent normalized-coordinate products.
2. **Leap target in block-fraction space**: `fract(mmPos*blockDimF)` assumes
   uniform block width 1/blockDim — same non-integer drift (~3.5 voxels
   mid-range on z). Fix: target `min((b+1)*8,fineDim)/fineDim` directly.
3. Residual ±1-step landing differences vs the per-cell chain are inherent
   fp-order effects (accumulated ceils + 1e-4 fudges cannot be replicated
   by a single hop). NOT fixable without simulating the chain; accepted as
   bounded quantization (34.5).

fc_mmBlocks [[function_constant(35)]] specializes pipelines (featureMaskExtra
bit 2): non-block pipelines keep byte-identical HEAD codegen; runtime-uniform
gating cost ~+30..+90 ms in an intermediate build (hot-path branch defeated
codegen) — compile-time gating is mandatory for march-loop features.

### 34.4 Results (@2048², 30-frame bench, eval-env ABBA-rotated arms)

minmax+blocks vs HEAD minmax (same binary, fc-specialized):

| view | SD0.5 j0/j1 | SD4 j0/j1 |
|---|---|---|
| obl | −32/−34 | +1.2/+1.4 |
| AZ45 | −20/−21 | −0.7/−0.3 |
| AZ135 | −34/−34 | +1.4/+0.8 |
| axx | −43/−44 | −5.4/−3.9 |
| axy | −50/−51 | −4.5/−3.7 |
| axz | **−145/−148** | −2.0/+0.2 |

vs RAW (mm off): az45 −11, axx −46..−48, axy −48..−50, axz −29..−32 WIN;
obl +0.7..+2.5, az135 +4.5/+5.2 tie-class. @1024 SD0.5 blk beats raw in all
six views. Production config spot checks: SHADE+mm1 @1024 44.7→36.8 (−18%);
mv9 unaffected (its preamble walk has no block path; separate PSOs).

### 34.5 Fine-SD gate + image deltas

Blocks are gated to sampleDistance < 1.5 (the DS=2 tier;
VolumeMinMaxBlocksWanted unifies build/fc/PSO-key gates): at SD4 the
bookkeeping cost more than leaps saved (+~1 ms obliques) and the coarser
lattice widened quantization, so SD>=1.5 is byte-identical HEAD again
(verified: max Δ=0 at SD4 az135/axz, timings tied).

Residual blocks-vs-HEAD image delta at fine SD (skip-landing quantization,
same perceptual class as §26.5's accepted macrocell rounding): 2.5–11% of
px differ by exactly 1 LSB; >1LSB px 4–155 of 1M @1024 / up to 9.8K of 4.2M
@2048 (worst axy SD4-class cells before the gate); max Δ <= 19 observed at
fine SD; mean|d| <= 0.041. mm-off paths byte-untouched (walk dead-code-
eliminated; raw-arm anchors matched §26.5 within battery drift).

### 34.6 mv9 preamble adopts block leaps (2026-08-23)

§33 item "mv9 preamble walk could adopt block leaps" DONE. The variant-9
preamble walk now carries the same two-level fast path (cell-derived block
index, center-texel fetch, true fine-cell-unit boundary planes, per-pass
block state cache, reciprocal-multiplied planes); fc_mmBlocks eliminates it
entirely for non-block pipelines.

Results (@2048² SD0.5, order-balanced pairs):

| view | mv9 raw | mv9 mm HEAD | mv9 mm +BLOCKS |
|---|---|---|---|
| oblique | 97.0 / 95.8 | 132.5 / 131.7 | **95.5 / 95.4** |
| axial-z | 307.3 / 307.2 | 426.0 / 424.9 | **276.5 / 276.7** |

mv9+mm flips from a +36/+118 ms penalty to **−1.5/−31 ms vs mv9-raw** —
minmax is now strictly worth running under mv9 at fine SD. The quantization
class is TIGHTER than the baseline's (max Δ=1, ZERO px >1LSB on obl+axz:
the step-parametrized leap avoids the baseline's matrix-resync drift), and
blocks-off mv9 output is byte-identical to the pre-change binary while the
fine-SD gate keeps SD4 byte-identical too (19.2/19.5 ms tied).

### 34.7 Dense-TF counterattack: mv9 three-state WIN, baseline codegen cliff, gate refuted (2026-08-23)

App testing flagged two issues; both chased to ground:

1. **Xcode-Debug launch crashed** (`volumeUniforms ... has space for 1732
   bytes, but argument has a length(1744)`): MSL rounds the uniform struct to
   its 16-byte alignment (float3/float4 members) while C++ sizeof stays 1732.
   All FIELD offsets verified identical on both sides (runtime-offset kernel
   vs C++ offsetof table) — pure trailing padding, silently tolerated in
   Release, fatal under Metal validation. Fix: allocate uniform buffers at
   round16(1732)=1744 (`VolumeUniformBufferSize`); `MTL_DEBUG_LAYER=1` bench
   runs now pass assertion-free.
2. **Dense presets (Bone + Skin II) still slower than mm-off at 2048**
   (+7 ms): tried a three-state block summary (all-SOLID blocks suspend
   per-cell lattice work). Results split by pipeline:
   - **mv9 preamble: huge win** — all-solid state lets batches composite
     dense terrain with near-zero preamble cost: mv9+mm+blocks 49.7–51.0 ms
     @2048 SD0.5 j1 = 2x under mv9-raw (~97) on Airways too.
   - **Baseline walk: CODEGEN CLIFF** — any added live state (solidRun/
     curBlockSolid), however minimal (even fetch-gating only), regresses the
     whole walk ~96→116-120 ms, losing more than solid-handling gains.
     Reverted to the proven two-state form; byte-output verified identical
     via framework-snapshot A/B (see protocol note).
3. **Static empty-fraction gate REFUTED**: fractions measured Airways 0.67 /
   BoneSkinII 0.46, but the benefit flips with RESOLUTION, not just TF:
   BoneSkinII mm+blocks WINS at 800 (−7.8%) and 1024 (−3.5%) SD0.5 and LOSES
   at 2048 SD{0.5→+12%, SD1 +39%}. A resolution-blind threshold would throw
   away real wins, so the BuildPerBlockData gate was removed; the fraction
   remains as a [TRMM] diagnostic. Residual dense-TF cost at 2048 stands at
   ~+7 ms over raw (still −18 vs HEAD-mm); at <=1024 minmax+blocks wins.

PROTOCOL LESSON (§32.5 extension): saved EXECUTABLE copies do not pin behavior
— the shader source + mapper live in vtk.framework, so A/B across versions
must snapshot `build_macos_metal/frameworks/vtk.framework/Versions/A/vtk`
alongside the binary (git-stash rebuild cycles otherwise compare identical
code).

### 34.8 HANDOFF status

§33 items done: probe decomposition (34.1), DS retune data (34.2),
mechanism fix (34.3), mv9 preamble adoption (34.6). Remaining:

1. ε-contribution emptiness tier (§33.2 item 2) remains open — orthogonal
   to blocks; would multiply skippable volume (fat/air near-zero-alpha).
2. TEMP inventory: MM_PROBE counters + [TRMM] block-consistency readback are
   env-gated diagnostics (keep-or-revert decision before any production
   default flip of MM_BLOCKS/VOLTRANSPOSE).
3. Default-enable decision for MM_BLOCKS needs multi-dataset coverage
   (single IMRToraceAddome so far, like §26.6 item 1's original caveat).

Logs: /tmp/mmprobe (probe matrix), /tmp/mmds (DS sweep), /tmp/mmblk +
/tmp/final (timing matrices), /tmp/blkpar+/tmp/blkfix+/tmp/blkfix2+/tmp/fc3
(parity ladder), /tmp/vtkBlkLeap (reference binary).

## 35. Preset × orientation matrix: mv9+mm+blocks vs mv9-raw @2048 SD0.5 (2026-08-23)

Follow-up to §34.6/§34.7 (which covered only oblique + axial-z on Airways II
and spot checks): full matrix of **mv9-raw** (`MINMAX=0 ACCEL=0`) vs
**mv9+mm+blocks** (`MINMAX=1 ACCEL=1 MM_BLOCKS=1`), both
`MARCH_VARIANT=9`, @2048², SD0.5, blue-noise jitter, slabs=1, VOLTRANSPOSE +
GPU_TRANSPOSE default-on, IMRToraceAddome. Four orientations (oblique,
CAM_AXIS x/y/z) × five TF presets — Airways II (default) plus the four app
presets newly wired into the bench: `VTK_METAL_TEST_PRESET=DarkBone`,
`SkinOnBlue`, `BoneSkin`, `BoneSkinII` (verbatim from VRPresets/*.plist,
rescaled like BoneSkinII was; bench stays shadeless per harness convention —
note Skin On Blue's plist sets useShading=true, untested here).

Protocol: ABBA order-alternated arm order per cell, 30-frame rounds,
j0/j1 both measured. Battery 100%, no thermal flags.

### 35.1 Results — mean(j0,j1) ms/frame, Δ = blocks vs raw

| preset | oblique | axis-x | axis-y | axis-z |
|---|---|---|---|---|
| Airways II | 64.4 → 50.4 (**−21.7%**) | 133.3 → 91.5 (**−31.3%**) | 130.5 → 89.3 (**−31.6%**) | 173.7 → 133.1 (**−23.4%**) |
| Dark Bone | 64.2 → 51.1 (**−20.4%**) | 132.6 → 92.3 (**−30.4%**) | 129.5 → 89.2 (**−31.2%**) | 175.5 → 133.9 (**−23.8%**) |
| Skin On Blue | 64.5 → 51.3 (**−20.5%**) | 131.6 → 91.9 (**−30.2%**) | 129.5 → 89.2 (**−31.1%**) | 174.8 → 133.9 (**−23.4%**) |
| Bone + Skin | 64.7 → 50.7 (**−21.6%**) | 131.4 → 91.5 (**−30.3%**) | 131.0 → 89.8 (**−31.4%**) | 173.1 → 132.8 (**−23.3%**) |
| Bone + Skin II | 63.7 → 50.5 (**−20.7%**) | 131.8 → 92.1 (**−30.1%**) | 128.7 → 88.9 (**−30.9%**) | 174.3 → 133.2 (**−23.6%**) |

Raw per-arm numbers (j0/j1) in `/tmp/preset_matrix/results.txt`; generator
script preserved at `/var/folders/.../T/opencode/preset_matrix.zsh`
(eval-env wrapper pattern, §29-compliant).

### 35.2 Findings

1. **mv9+mm+blocks wins every one of the 20 cells**, by a view-class margin
   that is TF-independent: oblique −20..−22%, axis-x −30..−31%, axis-y
   −31%, axis-z −23..−24%. The three-state mv9 block summary
   (`44abc454f1`) has erased §34.7's dense-TF residual: Bone + Skin II no
   longer pays "+7 ms over raw" in this protocol — it is −13 ms UNDER raw
   at oblique (50.5/51.3 vs 63.4/64.0). The earlier +7 reading did not
   reproduce (different session/camera state; not re-litigated).
2. **TF choice is cost-neutral here**: all five presets land within ~1–2%
   of each other in every cell/arm (raw oblique spans 63.4–64.9 across
   presets). Verified the preset knob actually engages: 512² Metal renders
   of all five presets are visually distinct with distinct hashes
   (/tmp/preset_matrix/presets_png/). At SD0.5 frame cost is dominated by
   traversal-to-saturation, which these CT presets reach at similar depth
   despite different opacity ramps.
3. **Jitter is free at SD0.5 in every cell and both arms** (|Δ| ≤ ~2 ms,
   one +4 outlier) — replicates §31.2's post-transpose finding, now across
   presets and all four orientations.
4. Orientation profile matches §31.1/§26.5: raw-arm z-march is the most
   expensive (~175), x/y ~130; blocks compresses the spread
   (~89 y / ~92 x / ~134 z). The x-vs-y asymmetry (91.5 vs 89.3 blocks;
   X-depth layout) is present in every preset identically.
5. Cross-check: an earlier same-day run (broken arg-passing, axis cells
   only) replicated these axis values within 1–3%; obliques were
   re-measured under the fixed protocol.

Session-drift caveat: absolute raw-oblique reads ~64 ms today vs §34.6's
~97 anchor (same binary family; battery healthy). Relative A/B within
session is the deliverable, per protocol.

### 35.3 Reconciliation with the §34-era invocation: those "mv9" rows were mv0 (2026-08-23)

The §34.6/§34.7-era spot-check script omits `MARCH_VARIANT`, and
`VolumeMarchVariant()` defaults to **0** (the TEMP-REPRO pin) — so its
"raw"/"blk" rows measure **mv0**, only its trailing "mv9 blk" row is really
mv9. Verbatim re-run of that script on today's build (same session as
§35.1):

| cell | §34-era ref | repro | true variant |
|---|---|---|---|
| Airways sd0.5 raw | 96.63 | 101.53 | mv0 raw |
| Airways sd0.5 blk | 96.22 | 95.83 | mv0 + mm+blocks |
| BoneSkinII sd0.5 raw | 59.75 | 58.79 | mv0 raw |
| BoneSkinII sd0.5 blk | 67.08 | 65.45 | mv0 + mm+blocks |
| "mv9 blk" | 51.01 | 50.79 | mv9 + mm+blocks (env set here) |
| MTL_DEBUG_LAYER 1024² | exit 0, 0 assertions | exit 0, 0 assertions | uniform-buffer fix holds |

Blk cells reproduce within ±0.4–2.4%; Airways-raw ±5% (that arm is the
noisiest — see below). Consequences:

1. **The mv0 picture differs sharply from mv9**: mv0-raw is strongly
   TF-dependent (Airways II ~97–102 — its 0.25-max-opacity ramp never
   saturates rays, so marches run the full volume; Bone + Skin II ~59 —
   dense ramp saturates fast), and mm+blocks on mv0 ties Airways (−5%)
   but LOSES +11% on the dense preset — §34.7's "+7 ms" dense-TF
   counterattack is real **on mv0 only**. Under mv9 both arms flatten
   (~64 raw / ~51 blk on every preset) and blocks win everywhere (§35.1).
2. **§34.6's oblique table rows labeled "mv9 ..." carry mv0-class values**
   (97.0/95.8/95.5 ≈ today's mv0 101.5/95.8; explicit mv9 measures 64/50),
   i.e. that session's A/B arms most likely also ran without
   `MARCH_VARIANT`. Its axz triple (307/426/276) likewise matches the
   §31.2 mv0-class z column. Treat §34.6's table as an **mv0** result.
3. Net: "minmax+blocks worth it under mv9 at fine SD" (§34.6 verdict)
   stands and extends to all presets/orientations (§35.1); the dense-TF
   regression applies only to the baseline-variant walk, whose own
   codegen cliff (§34.7 item 2) already blocks adding state there.
4. Any future A/B MUST print or set `VTK_METAL_TEST_MARCH_VARIANT`
   explicitly — the TEMP-REPRO default silently switches the variant
   class being measured.

### 35.4 Azimuth × preset compass on mv9 (2026-08-23): refines §35.2 finding 2

§35.1 tested one fixed oblique camera; this compass sweeps CAM_AZ
0..315 over the same base (@2048 SD0.5, explicit `MARCH_VARIANT=9`, ABBA
order-alternated, j0/j1 both measured, 5 presets × 8 azimuths ×
{raw, mm+blocks}). Full data `/tmp/preset_az/results.txt`.

Blocks-vs-raw Δ% by azimuth (mean(j0,j1)) — per-preset columns show the A/B
is azimuth-shaped but preset-independent (per-AZ spread across presets
≤ ~1.5%):

| AZ | Airways | DarkBone | SkinOnBlue | BoneSkin | BoneSkinII |
|---|---|---|---|---|---|
| 0° | 68.4/52.8 (−22.8%) | 64.8/51.0 (−21.3%) | 64.5/50.9 (−21.1%) | 65.5/51.9 (−20.9%) | 64.9/50.9 (−21.7%) |
| 45° | 61.9/45.8 (−26.0%) | 60.7/43.9 (−27.7%) | 60.3/43.8 (−27.4%) | 61.0/44.3 (−27.3%) | 61.6/43.9 (−28.7%) |
| 90° | 64.9/49.5 (−23.7%) | 63.0/48.6 (−22.8%) | 62.5/48.0 (−23.2%) | 63.3/48.4 (−23.5%) | 63.0/48.7 (−22.7%) |
| 135° | 67.2/57.1 (−15.0%) | 64.2/53.9 (−16.0%) | 63.8/53.0 (−16.9%) | 64.2/53.4 (−16.9%) | 64.3/53.9 (−16.2%) |
| 180° | 66.3/53.7 (−19.1%) | 64.6/51.6 (−20.1%) | 64.5/52.1 (−19.3%) | 64.6/52.1 (−19.4%) | 65.4/52.3 (−20.1%) |
| 225° | 63.2/46.2 (−26.9%) | 61.1/44.3 (−27.5%) | 60.3/44.3 (−26.6%) | 61.1/45.2 (−26.1%) | 61.9/44.9 (−27.4%) |
| 270° | 65.5/48.2 (−26.4%) | 62.7/46.9 (−25.2%) | 62.8/47.0 (−25.1%) | 62.8/46.9 (−25.4%) | 62.9/47.0 (−25.4%) |
| 315° | 65.5/53.6 (−18.2%) | 63.5/51.0 (−19.7%) | 63.5/50.7 (−20.2%) | 63.9/51.1 (−20.0%) | 63.9/51.1 (−20.0%) |

(cells = raw-mean/blocks-mean ms/f; note this fine-SD transposed azimuth
profile differs from §31.1's SD4 compass regime: here az45/225 are the
CHEAPEST cells and az135 the most expensive.)

Findings (refining §35.2):

1. **DarkBone / SkinOnBlue / BoneSkin / BoneSkinII are mutually
   cost-neutral at every azimuth** (cluster spread ≤ ~1.5% both arms) —
   §35.2's "TF choice is cost-neutral" holds across the full compass for
   these four, not just the one camera.
2. **Airways II is NOT free, though**: it sits +1.7..+6.6% above the
   four-preset cluster in BOTH arms at every azimuth (largest at
   az135/315/0). Mechanism: its opacity tops out at 0.25, so rays never
   reach the 0.996 early-exit threshold and march the full volume; the
   other presets saturate and trim ray length. The TF lever that matters
   on mv9 is SATURATION DEPTH (march length), not ramp content/color.
3. The minmax+blocks win therefore generalizes: −15..−17% (az135) to
   −26..−29% (az45/225) with negligible preset interaction. Jitter stays
   free (|Δ| ≤ ~2 ms, one +3 outlier at az135-blk).

### 35.5 Headroom A/B round 1 (2026-08-23): forced Y-depth wins under blocks; DS retune dead

Quick env-only variants @2048² mv9 (single 30-frame runs unless noted;
ABBA where stated). Logs `/tmp/headroom/`.

1. **`VOLTRANSPOSE_AXIS=y` beats the auto-X tie-break under mm+blocks**,
   everywhere measured (jittered, mm1+blk1):
   ToraceAddome obl 50.5→45.0 (−11%), az45 44.6→39.3 (−12%), az135
   54.0→48.5 (−10%), axz 134.1→122.4 (−9%) [ABBA-confirmed, replicates
   ±0.5]; FGFegatoIMR obl 43.2→39.1 (−9.5%), axz 59.8→54.6 (−8.6%);
   IMRTA4 obl 51.0→45.1 (−11.7%), axz 126.3→115.6 (−8.4%). RAW arms are
   neutral (±2%). Render parity X-vs-Y under blocks @512²: ZERO px >1LSB,
   max Δ=1 (same class as §30.2's near-exact rounding note).
   Caveat: §30.2's SD4/mv0-era matrix ranked X first on oblique
   (23.0 vs 24.5); today mv9 ranks Y first at SD4 too (raw 19.8 vs 22.1,
   blk 20.1 vs 20.4). The ranking is pipeline-dependent → do NOT
   blind-flip the tie-break; give it §27-class breadth first. The env
   knob makes it deployable per-deployment today.
2. **MM_DS retune under blocks: refuted** — ds2/ds4/ds8 within 0.7% of
   each other on obl/az45/az135/axz (blocks already decouple leap
   granularity from cell size; §34.2's tradeoff is gone).
3. **Crossing probe (baseline walk, MM_PROBE) post-blocks**: mean
   visits/crossings per covered ray drop from 428/394 → 205/172 (obl),
   318/269 → 167/118 (az45), 663/617 → 296/251 (axz). Residual crossings
   are the remaining walk-overhead pool; a third summary level
   ("super-blocks", all-empty runs of blocks) is the next byte-exact
   mechanism candidate — bounded by these counts, likely ~5–15% of frame
   if it halves crossings again. Not built yet; mv9's three-state
   preamble (§34.7) would absorb it more safely than the baseline walk
   (codegen-cliff history).
4. ε-contribution emptiness tier (§33.2 item 2) remains the largest
   theoretical lever (~40% in the §32.4 accident) but changes the image —
   still open, needs its calibration protocol.

Verdict: the immediate, already-validated headroom is the Y-depth
orientation under blocks (−8..−12%, three datasets, image-equivalent);
everything else needs either breadth validation or new code.

### 35.6 Headroom round 2 (2026-08-23): ε-tier and super-blocks BOTH REFUTED

§33.2 items 2 and the §35.5 item-3 follow-up implemented behind env knobs
and A/B'd (`VTK_METAL_TEST_MM_EPS=<f>`, `VTK_METAL_TEST_MM_SUPER=1`;
both default-off, byte-inert when unset — verified: new binary renders
BYTE-IDENTICAL to the pre-change binary at eps=0).

Implementation notes (for future work in this area):
- ε tier: MinMaxComputeUniforms carries the 256-entry opacity LUT +
  threshold; `volume_compute_minmax` marks a cell empty when
  max achievable opacity in [idxMin,idxMax] <= eps (eps=0 keeps exact
  prefix-sum semantics). Walk untouched — zero codegen risk.
- Super level: `volume_reduce_minmax_superblocks` reduces the BLOCK
  summary into all-empty 8³-block groups (fc_mmSuper, function_constant
  36, featureMaskExtra bit 3, fragment texture slot 17);
  `marchVolumeUnified`/`marchVolume`/`marchSegment` thread the texture;
  the mv9 preamble derives super indices from BLOCK indices (integer
  divide), caches state, and leaps to true fine-cell-unit boundaries.
  Shader plumbing lesson RE-LEARNED: the march lives in inline helpers
  with forwarded texture params — entry-point signature edits alone fail
  at RUNTIME library compile ("undeclared identifier"), and duplicated
  parameters fail with "redefinition" (black frames either way; stderr
  names the line).

Results (@2048² SD0.5 mv9 mm+blocks, j1; same-session references):

| variant | obl | az45 | axz |
|---|---|---|---|
| blocks-only reference | 50.6 | 45.1 | 134.6 |
| +SUPER | 61.5 (**+21.6%**) | 53.2 (**+17.8%**) | 165.0 (**+22.6%**) |
| EPS 0.005 | 50.6 (±0) | 44.7 (−1.0%) | 134.5 (±0) |
| EPS 0.01 | 50.8 (+0.5%) | 44.6 (−1.2%) | 135.6 (+0.7%) |
| EPS 0.02 | 50.3 (−0.6%) | 44.6 (−1.2%) | 134.9 (±0) |
| EPS 0.05 | 51.9 (+2.6%) | 44.7 (−0.9%) | 136.4 (+1.3%) |

Image parity ladder (@512², jittered): old-binary≡new-binary eps0
(max Δ=0); blk≡blk+SUPER (max Δ=0 — leaps land on identical samples);
eps 0.005/0.01 inert, 0.02 max Δ=1, 0.05 → 5 px >1LSB of 262K, max Δ=15.

Findings:

1. **Super-blocks REFUTED**: +18–23% REGRESSION despite byte-identical
   output. The probe-predicted residual crossings (§35.5: 118–251/ray)
   live in MIXED block territory, so the third level pays 2 summary
   fetches + index math per block change and almost never fires; the air
   gaps that motivated it were already consumed wholesale by block leaps.
   Two levels capture the lattice's real structure. Do NOT retry a third
   level on this pipeline shape.
2. **ε-tier REFUTED as a perf lever** on this dataset/TF class: timing
   flat within ±1% across 0.005–0.05, worse at 0.05 on obl/axz. The
   exact-emptiness + dilation + blocks stack already harvests everything
   reachable: near-zero-opacity cells form thin boundary slivers that are
   off the critical path. This also retro-explains §32.4: the corrupted
   lattice's "40%" came from SCATTERED min/max wrongly emptying dense
   interior cells — not replicable by legitimate ε semantics. Keep MM_EPS
   as a diagnostic knob; never default.
3. Protocol: the §29 zsh trap bit AGAIN (`for ARM in $ORD` does not
   word-split — only the second arm ran; caught because blk rows were
   missing from output and sup timings looked anomalous vs anchors).
   Use `${=ORD}` or arrays.

Net after rounds 1+2: the minmax+blocks stack is at its architectural
floor for this pipeline; the validated remaining headroom is the Y-depth
orientation policy (§35.5) and anything outside the walk (batch width,
axis-z specifics). §33's plan is fully dispositioned.

### 35.7 Why is Y-depth faster? Mechanism probe (2026-08-23)

Discriminating experiment: re-run X-vs-Y under GL_NEAREST (kills the
depth'-pair fetch whose handling owned every prior layout effect,
§15/§26). @2048² SD0.5 mv9 oblique j1, order-alternated:

| arm | X-depth | Y-depth | Δ |
|---|---|---|---|
| NEAREST raw | 58.25 | 60.26 | +3.5% |
| NEAREST blocks | 48.77 | 44.26 | **−9.2%** |

The Y win SURVIVES NEAREST undiminished → the trilinear-z-pair
locality model (which explained the pre-transpose tax) does NOT explain
this one. Additional evidence against role-based models: under Y-depth
the axy cell (marching along Y-depth's OWN depth' axis) reads 83.0 ms vs
X-depth's 89.3 — if depth'-axis marches were intrinsically disfavored
(the §26.5 sagittal story), Y should LOSE that cell; it wins.

Refined conclusion: the effect is (a) tied to the BLOCKS regime
(scattered post-leap composite bursts; raw streams are neutral), (b)
governed by the PHYSICAL placement of the long 1794 extent in the
tiler's dimension order (texture width 1794×h512×d512 vs w512×h1794×d512),
not by which logical axis plays depth', and (c) a point-fetch locality
property of burst access patterns, not interpolation semantics. Walk
work is provably identical between orientations (data-space math,
byte-identical outputs), so the delta lives entirely in volume-texture
fetch behavior during composite bursts. Which tiling rule produces it is
hardware/driver knowledge — settling it needs Instruments GPU counters
(DRAM read amplification, §22 item 6, still open) or Apple input.
Practically: the finding stands on its own measurements — validate
breadth (§27-class) and consider flipping the tie-break policy.

### 35.8 Y-depth confirmation sweep (2026-08-23): wins 11/11 views

Full ABBA per cell (x,y,y,x order-alternated by cell parity), @2048²
SD0.5 mv9 mm+blocks j1, IMRToraceAddome. Closes §35.5's coverage gaps:
the whole azimuth compass plus all three axes now measured interleaved
(previously only obl/az45/az135/axz were ABBA'd; axx/axy were single-run).

| view | X mean | Y mean | Δ |
|---|---|---|---|
| az0° | 52.55 | 46.84 | −10.9% |
| az45° | 45.36 | 40.14 | −11.5% |
| az90° | 49.68 | 43.98 | −11.5% |
| az135° | 56.44 | 49.72 | −11.9% |
| az180° | 53.57 | 47.61 | −11.1% |
| az225° | 47.41 | 40.70 | −14.2% |
| az270° | 47.85 | 43.00 | −10.1% |
| az315° | 52.71 | 47.25 | −10.4% |
| axis-x | 94.48 | 83.85 | −11.2% |
| axis-y | 91.28 | 84.66 | −7.3% |
| axis-z | 136.02 | 124.44 | −8.5% |

Within-arm replicates ≤ ±0.7 ms (worst: az135-x 55.69–57.19). Verdict:
Y-depth is uniformly −7..−14% under mm+blocks at this config across every
oblique angle and all axis views — no view class escapes or inverts it.
Combined with §35.5's second/third-dataset cells and render parity
(§35.5: zero px >1LSB), the finding is breadth-confirmed on this dataset;
remaining scope caveats for a policy flip: other datasets have only
obl+axz interleaved-less cells, SDs beyond {0.5, 4-obl} unswept, single
GPU/dataset-modality class (§27.3 caveats apply verbatim).

### 35.9 Y-depth × sample distance (2026-08-23): the win is FINE-SD specific

§35.8's remaining caveat closed with an ABBA sweep @2048² SD4 mv9 j1
(6 views × {raw, mm} — note blocks are gated OFF at SD>=1.5, so "mm" here
is minmax-without-blocks), plus an SD1 disambiguation pass:

SD4 means (X / Y / Y-delta):

| view | raw | mm |
|---|---|---|
| obl | 21.63 / 19.90 (−8.0%) | 20.41 / 20.10 (−1.5%) |
| az45 | 24.69 / 16.38 (**−33.7%**) | 20.00 / 18.68 (−6.6%) |
| az135 | 18.61 / 20.43 (+9.8% X) | 21.28 / 21.39 (+0.5% ≈tie) |
| axis-x | 37.02 / 28.64 (**−22.6%**) | 40.53 / 38.86 (−4.1%) |
| axis-y | 28.90 / 35.82 (**+23.9% X**) | 36.57 / 37.81 (+3.4% X) |
| axis-z | 38.92 / 38.14 (−2.0%) | 54.39 / 54.46 (±0) |

SD1 (inside the blocks/fine-SD tier): Y wins uniformly again —
obl 33.73→30.07 (−10.9%), az135 34.26→31.32 (−8.6%), axy 56.82→52.87
(−7.0%) — including exactly the views that preferred X at SD4.

Conclusions:

1. The Y advantage is a property of the FINE-SD regime (DS=2 lattice
   tier, dense marches), not universal: at SD4 the ranking is view-
   dependent with large swings both ways (az45 −34% vs axy +24% raw),
   and under SD4-mm it collapses to ±0–7% noise.
2. Policy consequence: do NOT flip the global tie-break. If adopted,
   gate Y on the SAME static fine-SD tier the blocks feature already uses
   (sampleDistance < 1.5 in VolumeMinMaxBlocksWanted terms) — a static
   quality-based decision (§25.5-compliant: no runtime/view gating).
   Implementation would extend VolumeTransposedAxisDepth's policy input
   with SampleDistance; breadth caveats of §35.8 still apply before any
   default change.
3. §30.2's SD4-era "X wins oblique" was measured pre-mv9-pin; today's
   mv9 oblique reads slightly Y-favoring while az135/axy favor X — the
   coarse-SD ranking remains pipeline-dependent and is best left to the
   existing policy/env.

### 35.10 NEW FINDING + blocks-at-SD4 probe (2026-08-23): mm LOSES to raw at SD4 on axes; blocks recover most of it

Interleaved raw-vs-mm pairs @2048² SD4 mv9 X-depth j1 (first time mm-on
is A/B'd against mm-off on current code — §16's "minmax wins at SD4"
compared against GL, which has no minmax; §25.2's caveat finally bit):

| view | raw | mm | mm penalty |
|---|---|---|---|
| obl | 21.3 | 20.7 | none (mm −2%) |
| axis-y | 29.0 | 36.7 | **+27%** |
| axis-z | 39.0 | 54.6 | **+40%** |

Confirmed by a second interleaved pass. The "minmax wins at SD4"
standing assumption is dead on transposed+mv9 code: at SD4 the lattice
walk taxes more than skipping pays except on obliques. Blocks probe
(`VTK_METAL_TEST_MM_BLOCKS_ANY_SD=1`, env-gated lift of the <1.5 gate;
three-arm ABBA rotation, X-depth):

| view | raw | mm | mm+blocks (forced) |
|---|---|---|---|
| obl | 21.79 | 20.75 | **19.57** (−10% vs raw, best) |
| az135 | 18.38 | 21.18 | 18.79 (≈raw) |
| axis-y | 29.13 | 36.71 | 31.58 (+8% vs raw) |
| axis-z | 39.05 | 54.42 | 47.40 (+21% vs raw) |

Verdicts:

1. Blocks at SD4 make minmax SAFE (mm+blk <= mm everywhere, often <),
   but on axis views mm+blocks still loses to plain RAW — the walk's
   fixed per-crossing cost exceeds skip yield for short coarse-SD
   marches. Raw remains the best SD4 config on axes; mm+blocks wins
   only the oblique class.
2. Image caveat for any SD4-blocks default: block leaps introduce the
   ±1-step fp-drift quantization class (§34.5: up to ~10K px >1LSB
   @2048 in the worst pre-gate cells). NOT a DS effect — the emptiness
   test is exact for U8, so plain-lattice DS changes are output-
   identical (correction of the first draft, which blamed DS=4).
 3. Unexplored: DS interplay at SD4 (MM_DS=2 with blocks — finer cells
    might improve axz skip yield further; §34.2's DS data was fine-SD).
 4. Policy options for SD>=1.5: (a) keep mm off (best on axes, −10% on
    oblique vs mm+blk), or (b) mm+blocks everywhere (uniform behavior,
    +21% worst case). Either is a static SD-tier decision; (a) is
    currently the performance-optimal read on this dataset.

### 35.11 Why mm loses at SD4 — history resolved + DS grid: COARSER lattice is the coarse-SD win (2026-08-23)

Q: "previous reports said minmax was faster at SD4?" A: it never was,
Metal-vs-Metal. The belief came from two unfair comparisons:
(i) §16 compared Metal-mm vs GL, which has no minmax implementation
(§25.2's caveat); (ii) the pre-transpose mv0 era looked like a −40% mm
win (baseline raw 43.3/66.1 vs baseline+mm 25.1/27.2 @2048 SD4) but that
raw arm was paying the axis-bias tiling tax transpose later removed.
§26.1's own table already contained the flip unremarked — transposed
mv0-RAW 20.2/22.2 vs transposed mv0+mm 23.94/24.67 — i.e. mm was ~10-15%
SLOWER at SD4 oblique the day transpose landed. mv0-vs-mv9 is NOT the
cause: the flip predates mv9. mm didn't regress; the raw march improved
out from under it, and the walk became net tax except where skip yield
is high (az45-x still −19% for mm at SD4).

Improvement A/B — DS × blocks grid at SD4 (@2048² mv9 X-depth j1, ABBA;
DS forced via VTK_METAL_TEST_MM_DS, blocks via MM_BLOCKS_ANY_SD):

| view | raw | mm DS4 (default) | mm DS8 | blk DS8 | mm DS2 | blk DS2 |
|---|---|---|---|---|---|---|
| axz | **39.0** | 54.4 | 51.4 | 46.0 | 61.9 | 66.1 |
| obl | 21.8 | 20.75 | **19.85** | 20.3 | 23.9 | 23.9 |
| az45 | 24.7 | 20.0 | **18.0** | 21.6* | 21.1 | 20.1 |
(*b2/b8 shown where run; blk-DS4 from §35.10: obl 19.57, axz 47.40)

Findings:

1. At SD4 the OPTIMAL DS flips to COARSE (DS=8), matching the
   ComputeMacrocellDownsample comment's own rule ("coarse sampling wants
   coarser cells so one skip advances several steps") — the mirror of
   §34.2's fine-SD result. mm-DS8 beats today's default mm-DS4 by −4%
   (obl) / −10% (az45) and trims axz to +32%-over-raw (from +40%).
   Blocks add nothing once DS is right at SD4 (b8 ≈ ds8 obl; worse
   az45) — the fine-SD gate stays correct.
2. Output parity: plain-lattice DS changes are pixel-exact (exact U8
   emptiness test per ComputeMacrocellDownsample comment; §34.5 verified
   gated-SD4 renders max Δ=0). DS8-at-SD4 would be a byte-clean pure
   timing change; formal snapshot spot-check listed as follow-up.
3. Residual structural gap: even best-mm config loses to raw by ~18% on
   axis views at SD4 — the per-crossing walk cost exceeds skip yield on
   body-interior rays regardless of granularity. Untunable by lattice
   shape; candidate future fix is an early walk bail-out after N
   fruitless crossings (runtime-adaptive — needs §25.5 review) or
   simply keeping minmax off at coarse SD (policy (a)).
 4. Refined policy read for SD>=1.5: if minmax stays on there, retune
   ComputeMacrocellDownsample to return 8 in the coarse tier
   (`useGPUMinMax && sd >= 1.5 ? 8 : ...`); if it goes off, DS is moot.
   Either way blocks stay fine-SD-only.

### 35.12 Residual walk-tax decomposition: three fixes attempted, all refuted; tax localized (2026-08-23)

Goal: close the +18..40% mm-vs-raw gap at SD4 axis views. All experiments
@2048² mv9 X-depth j1, ABBA/rotation-balanced, reverted after refutation
(HEAD binary re-verified: axz-ds4 56.6 ≈ pre-experiment).

1. **Pipeline cost is ZERO** — new diagnostic `VTK_METAL_TEST_MM_NOWALK`
   (runs the fc_minmax pipeline, lattice built+bound, walk skipped):
   axz 38.4 vs raw 40.8 vs mm 56.7; axy 28.7 / 30.4 / 38.2. The entire
   residual tax lives in WALK EXECUTION. (Diagnostic reverted with the
   rest; re-add if this line of work resumes.)
2. **Walk-window clamp REFUTED**: preamble look-ahead was hardcoded 48
   steps while the batch dispatch consumes maxBatchWidth=8 at SD4 (cap
   added later — integration gap). Clamping the window after a
   compositing batch changed nothing (axz clamp 56.0 / full48 55.7;
   az45 20.7 / 20.7): leap count within the consumed span is invariant.
3. **Cell-solid memoization REFUTED (codegen cliff #2)**: caching the last
   sampled cell index + verdict to skip re-samples made things WORSE:
   memo axz-ds4 75.1 / nomemo 69.1 / pre-change HEAD 54.4 (+38% absolute).
   Matches §34.7's baseline-walk lesson: ANY added live state on the
   serial preamble path costs more than saved work. Reverted.
4. **Tax localized by elimination**: not pipeline structure (1), not
   iteration count (2), not sample count alone (2,3) — it is the
   per-empty-cell LEAP MATH (fract/div/ceil boundary solve) plus branch/
   state overhead on a serially-dependent chain that stalls each batch's
   volume-fetch flight. Consistent with blocks helping most (one leap per
   block, cached state) and with DS8 > DS4 (fewer cells = fewer solves).
5. Remaining options, in ascending cost: accept policy (a)/DS8 retune
   (§35.11); DDA-style incremental traversal (adds/comparisons instead of
   div+ceil — NOT byte-clean: ulp drift changes skip landings, same class
   as blocks' accepted drift but on a path currently held byte-exact);
   warp-cooperative or async-compute segment pre-passes (structural,
   out of scope for quick A/B). No default changes made.

### 35.13 In-fragment segment pre-walk implemented and REFUTED; occupancy-cliff law generalized (2026-08-23)

Implemented the cheap variant of the async-segment design: run the walk
ONCE per fragment before the march (same thread => bit-identical math),
record skipped-step gaps, consume with integer tests between batches.
Full plumbing: VolumeFeature_MMSegments (1u<<27), fc_marchSeg
(function_constant 37), PSO-key bit, VTK_METAL_TEST_MM_SEG gate.
Three storage encodings A/B'd at SD4 @2048² (obl / axz, legacy = HEAD):

| encoding | mm legacy | mm+seg |
|---|---|---|
| function-scope int arrays [24] | (decls alone regressed legacy 22→44!) | ~45 / ~112 |
| int arrays [4] | 21.85 / 55.9 | 45.2 / 116.6 |
| packed ulong registers (K=8) | 22.0 / 56.3 | **49.1 / 124.0** |

Findings:

1. **Occupancy-cliff law (third instance, now generalized)**: ANY
   meaningful added state or upfront work in this fragment — arrays
   (even fc-dead: decls alone 2×'d legacy via scratch), tiny live arrays,
   or ~10 extra live registers + a serial probe chain — costs ~2× frame
   time. §34.7 (baseline-walk cliff), §35.12 (memo cliff), §35.13 (this):
   the mv9 fragment sits exactly at the register/occupancy edge; there is
   NO headroom for in-shader structural additions. Future designs MUST
   move work into separate passes, not the march shader.
2. The upfront build's serial probe chain does not overlap batch fetch
   flights (unlike HEAD's interleaved per-batch probes, whose latency
   partially hides behind compositing ALU of the previous batch).
3. Tooling gotchas fixed en route: `getenv("X")` treats `X=0` as ON
   (VolumeMMSegWanted added); function-scope array decls are NOT free
   even when dead.
4. Verdict: the true async pre-pass must be a separate GPU pass (ray-atlas
   raster pre-pass writing origin/dir/t-range from the REAL interpolated
   varyings — kernels cannot reproduce them bit-exactly — then compute
   kernel builds segments, main pass consumes). Multi-day infrastructure
   project; patch archived at
   PerformanceInvestigation/mm_segment_prewalk_refuted.patch (195 lines,
   applies to 621cc2f3d8). Tree reverted; HEAD re-verified
   (axz-ds4 57.0 ≈ anchor).

### 35.14 TRUE multi-pass segment pre-pass BUILT and verified pixel-exact — walk cost eliminated; the wall is now the consume pipeline itself (2026-08-23)

Implemented §35.13's "true design" end-to-end, default-off
(VTK_METAL_TEST_MM_SEG=1, value-parsed so =0 stays OFF):

- **Stage 1 — ray atlas** (`fragment_volume_ray_atlas`, new
  VolumePipelineType::RayAtlas=10, three RGBA32Float attachments):
  replicates the fragment_volume_main prologue + marchVolume jitter/tStart
  + marchVolumeUnified setup head VERBATIM on the same interpolated
  varyings and writes A=(evalPoint.xyz, steps), B=(evalStep.xyz,
  stepSize), C=(rayDir.xyz). Same geometry/vertex fn as the main pass ⇒
  bit-identical inputs.
- **Stage 2 — builder** (`volume_segment_build` compute, one thread/pixel):
  walks the lattice ONCE per ray (preamble leap chain verbatim: same
  clamp, boundary solve, +1e-4, ceil), records skipped-step gaps as u16
  (start,end) pairs into an ATOMICALLY COMPACTED pool (word0=count, then
  pairs; segIndexMap[pixel]=offset or UINT_MAX on overflow → composite-
  all fallback). Gap storage = packed uint4 registers (16 gaps) appended
  via static select chains — a dynamically indexed local array spills to
  device memory per append. Guard valve bounds the walk (8192 iters).
- **Stage 3 — consume**: mv9 preamble compiles OUT under fc_segHop
  (function_constant(37), featureMaskExtra bit 16); the loop streams the
  gap list with integer tests, holding only (recOff,cnt,idx,gS,gE) and
  pulling the next pair from the pool on gap crossing. Direct path follows
  the slab ping-pong precedent (end drawable encoder → pre-passes →
  offscreen RGBA16Float march → Phase 3b blit).

Results @2048² SD4 mv9 X-depth j1 (warmup-clean, see tooling note):

| view | mm HEAD (direct) | seg full |
|---|---|---|
| axz | 55.08 | 112.44 |
| obl | 20.66 | 47.82 |

Decomposition (axz): offscreen+blit floor 58.29 (= direct + ~3 ms blit);
builder kernel ≈ FREE (~0 ms measurable at 14.7M claimed pool words);
atlas pass floor 1.7 ms (empty body) / ~2.7 ms full; **the consume
pipeline costs ≈ +51 ms EVEN WHEN IT HOPS NOTHING** (SKIP_BUILD run:
112.72 ≈ full 112.44) — i.e. pure pipeline overhead from ~6 live scalars
+ two device-buffer params, with the legacy preamble compiled out.

Findings:

1. **The async walk SUCCEEDED**: the entire preamble-walk tax moved out
   of the fragment for free, and output is PIXEL-EXACT vs HEAD-mm
   (0 bytes differ @axz 2048² after the fixes below).
2. **Occupancy-cliff law, finest instance yet (fourth)**: the mv9
   fragment rejects not just added WORK (§35.12/§35.13) but ~6 registers
   of added STATE even when equal work is REMOVED (preamble dead).
   Skipping itself is free (hops == no-hops timing); hosting the skip
   list is not. Any future fix must keep the march's register budget
   byte-identical — e.g. consuming segments in a SEPARATE lightweight
   pass that writes a pruned sample-interval texture the march reads via
   its EXISTING bindings, or restructuring the march entirely.
3. **Tooling lesson that poisoned hours of measurements**: bench defaults
   to --warmup 0, so every NEW pipeline variant paid seconds of PSO
   compilation amortized into short --frames windows (fake 2× slowdowns;
   contradictory stage attributions like "atlas costs 60ms"). Baselines
   looked stable only because Metal's persistent shader cache had their
   variants warm. ALWAYS bench with --warmup >= 5.
4. Crash mitigation after a system overload during early testing:
   meta.w maxGaps now EQUALS the 16-slot register capacity (48 silently
   corrupted gaps 17..48 into shared lanes), pool capped 64 MiB
   (overflow → safe composite-all), builder guard valve guarantees
   termination.
5. Investigation controls kept (all env-gated, default-off):
   VTK_METAL_TEST_MM_SEG_SKIP_ATLAS / _SKIP_BUILD / _SKIP_MARCH /
   _NOCONSUME (+ MM_SEG_DEBUG claims telemetry). Robustness notes:
   OffscreenLayer pipelines are in the seg key-bit gate (bind slots 6/7
   always, dummies when inactive) because the direct-path march runs
   through one; RTT branch carries the same pre-pass for completeness.

## 36. PLAN — restructuring mv9 to close the residual coarse-SD minmax gap (2026-08-23)

Design section distilled from §35.9–35.14. No code written yet; every path
below carries its own gate, parity bar, and kill criteria.

### 36.1 Constraint set (what the measurements force)

The target: at SD4 @2048² the accelerated march still pays ~+10 ms vs raw on
axis views through lung (mm-ds8 axz 51.4 / blocks-anySD 47.4 vs raw ~40.8),
while oblique already wins (19.85 / 19.57 vs 21.3). Fine SD is solved by
blocks (§34). The fix must live inside these proven bounds:

1. **Walk work must leave the fragment** — §35.12 (NOWALK ≈ raw: the tax is
   walk execution), §35.13 (in-fragment pre-walk 2×), §35.14 (compute builder
   ≈ free).
2. **The march must gain zero persistent state** — §35.14's consume pipeline
   cost +51 ms from ~6 loop-carried scalars + two device params, even with
   the preamble compiled out and zero hops executed ("occupancy-cliff law",
   fourth and finest instance).
3. **Skipping itself is free** — hops == no-hops timing; empty samples
   composite as exact zeros (verified by the occupancy prefix table), so any
   superset of HEAD's skipped set is output-safe up to the accepted ±1-step
   fp-landing class.
4. **Bench protocol**: --warmup ≥ 5 always (§35.14 tooling lesson); ABBA
   order-alternated; anchors drift ±5% with battery.

Rejected up front (do not revisit): warp-cooperative pre-passes (divergence
tax + complexity, same in-fragment state problem); DDA-only traversal
(constant-factor shrink of a walk we can already make free; loses byte-exact
landings on a path currently held exact); async overlap with other passes
(nothing to overlap within this renderer's serial frame).

### 36.2 Path P — de-risking probes (~half day, run first)

Two cheap A/Bs that determine which resource actually cliffs, and therefore
how much headroom any in-march design has:

- **P1 ladder shrink.** Rebuild mv9's width dispatch with {16,8,4,2,1}
  instead of {48,32,16,8,4,2,1} (48-wide only pays at fine SD where blocks
  already win; coarse SD caps at 8 anyway). Env-gate as
  `VTK_METAL_TEST_MARCH_LADDER=16` decoded next to `maxBatchWidth`
  (`VolumeShaderFeatureFlags`-free: uniform-driven like maxBatchWidth, no new
  fc). If the §35.14 consume penalty shrinks materially under the small
  ladder, the cliff is code-size/I-cache rather than registers — mv9 has
  unaccounted headroom, and Design A gets riskier while Design B less
  urgent.
- **P2 pointer-vs-index.** Rerun the §35.14 consume with the two
  `device uint*` params folded into ONE pointer (map+pool in a single
  buffer; fragment still pre-offsets). If +51 ms collapses, the culprit is
  device-pointer parameter handling (scalarization/address-space setup), not
  register count — then even the existing segment design may be salvageable,
  or consumable via an R32UI texture instead of buffers.

Both probes are measurement-only; nothing lands.

### 36.3 Design A — "tap-and-leap": delete the walk AND its state (recommended)

Invert the skipping philosophy. Today's preamble walks forward maintaining
cached summaries (mv9Blk/mv9BlkState/mv9Sb/mv9SbEmpty/solidRun — many
loop-carried registers) and re-walks between batches. Instead, certify large
empty spans with AT MOST one or two transient taps per batch, and otherwise
just march: compositing empties is free-and-exact (constraint 3).

Per outer iteration, before the width dispatch:

```
// all values transient — die at issue like MV9_FETCH addresses
superCell = cellCoord at batch far end (clamp(ep + es*(float)min(i+W,steps)-ε))
ssv = minMaxSuperTexture.sample(sNearest, (sbIdx+0.5)/mmSbDimF, level(0)).r
if (ssv > 0.5f && entryCellEmpty)          // whole span certified empty
    leap to super far face along the ray   // existing distToEdge/tToEdge/
                                           // exactSkip(+1e-4)/ceil math
else if (block tap affordable)             // optional second level
    ... block-summary variant, 8³ cells ...
else
    dispatch the batch unchanged           // empties composite as zeros
```

Why it satisfies the constraint set:

- **Zero persistent state.** Nothing survives the iteration; the leap math
  is the preamble's own formulas reused inline. Net registers go NEGATIVE:
  mv9Blk/mv9BlkState/mv9Sb/mv9SbEmpty/solidRun are deleted outright.
- **Worst case benign.** Mixed terrain ⇒ one/two R8 taps per batch (~56/ray)
  plus normal batches ⇒ degenerates toward raw + ε, i.e. strictly better
  than today's mm on the views where mm currently loses.
- **Best case is the whole prize.** Chest supers tile 64³ fine cells;
  airway/lung regions leap entire supers per tap.
- **Output class**: skipped set ⊆ HEAD's skipped set (only certified-empty
  spans leap). Accumulation stays bit-exact where we composite zeros that
  HEAD leaped; positions advance linearly (raw-canonical) instead of HEAD's
  leap arithmetic ⇒ ±1LSB-class vs HEAD-mm, identical to the accepted
  blocks class.

Implementation sketch:

- Shader: replace the `else if (useMinMax)` preamble block in the mv9 branch
  (MetalShaders.metal, marchVolumeUnified, variant-9 sub-branch) with the
  tap-leap block above. Keep the legacy preamble compiled behind the SAME
  function constants for A/B (fc_mmBlocks/fc_mmSuper stay; add
  `fc_mmTap [[function_constant(38)]]`).
- Mapper: `VTK_METAL_TEST_MM_TAP=1` → featureMaskExtra bit 32u (next free
  after seg's 16u) in `GetOrCreateVolumePipeline`; requires
  MinMaxSuperTexture present (same readiness pattern as mmSuper).
- Mostly-solid gate: reuse the EXISTING signal — BuildPerBlockData already
  clears the walk-enable flag via MinMaxEmptyBlockFraction when skipping
  cannot pay. Extend it: when empty-fraction < threshold, tap-leap compiles
  to plain batches so solid views keep today's ds8 win exactly.
- Optional refinement: adaptive tap depth (super → block → give up) chosen
  by ONE static per-frame constant from the CPU (empty-fraction buckets),
  never per-ray branching.

Expected outcomes / kill criteria:

| view | today | expected | verdict bar |
|---|---|---|---|
| axz | 51.4 | 42–46 | must beat 47.4 (blocks-anySD) |
| obl | 19.85 | 19.5–21 | within ±1 ms of ds8 |
| az45 | 18.0 | ~unchanged | no regression |
| SD0.5 obl/axz | blocks win | gated off or complementary | no regression |

Kill criteria: if taps alone can't beat blocks-anySD at axz, combine
(tap-leap for spans + blocks retained for mixed terrain) before abandoning.
Parity bar: snapshot diff vs HEAD-mm ≤ ±1LSB class quantified per view
(expect far fewer differing pixels than blocks, since fewer skips occur).

Effort: ~1 day including ABBA sweep + parity snapshots.

### 36.4 Design B — full compute marcher (the definitive restructure)

§35.14 quietly built two-thirds of this. The atlas already yields per-pixel
evalPoint/evalStep/rayDir/steps bit-exactly (rasterized varyings); the blit
handoff is measured (+3 ms floor); the builder proves compute occupancy is
generous enough to run the full walk for free. Move the ENTIRE mv9 batch
scheduler into a compute kernel:

- Kernel = one thread/pixel: read atlas planes, rebuild the few remaining
  setup quantities (texStep/ctp already derivable; add plane C.w =
  s.tTerminateMax from setupVolumeRay so scene-depth early-out survives),
  run MV9_FETCH/MV9_COMPOSITE + width ladder verbatim (registers free!),
  write half4 into an RGBA16Float target, Phase-3b blit (exists).
- Compute kernels sample 3D textures with constexpr samplers today
  (volume_compute_minmax et al.) — TF/gradient/shading bindings port
  directly.
- Hybrid strategy: keep the raster path for hybrid polygonal scenes
  (depth-composite correctness) and hardware selection; route volume-only
  interactive frames through compute (mapper-level decision, static per
  scene).
- Payoff: every occupancy-cliff instance (§34.7, §35.12, §35.13, §35.14)
  becomes structurally impossible; unlocks DDA, wavefront scheduling, and
  non-composite blend modes without fragment-path fear. Expected landing:
  near-raw on ALL views with acceleration intact.
- Effort/risk: multi-day; plumbing breadth (lighting uniforms, mask/blanking
  variants) is the cost, not the march itself.
- Decision criterion: build ONLY if (a) Design A leaves >5 ms on the table
  at axz, or (b) the wavefront/DDA roadmap justifies the platform anyway.
  Otherwise it is over-engineering for the residual 10 ms.

### 36.5 Execution order

1. P1 + P2 probes (half day) — they price Design A's risk.
2. Design A tap-and-leap behind VTK_METAL_TEST_MM_TAP (1 day): parity
   snapshots + ABBA {axz,axy,az45,obl} × {raw, mm-ds8, tap} + fine-SD spot.
3. Decide Design B on §36.4's criterion with A's measured numbers.
4. Whatever lands: doc section here, default-off env gate, warmup ≥ 5
   protocol, anchors table updated.

## 37. EXECUTED (2026-08-23 afternoon): the §36 premise collapsed — no minmax
penalty exists on a healthy machine; 2-level blocks win EVERYWHERE and are
the real fix (2026-08-23)

§36's execution order was followed until the measurements invalidated its
target. Full matrix, 2048² SD4 mv9, --warmup 8 frames 30, order-alternated
passes agreeing within ±2%:

### 37.1 The penalty did not reproduce

| view | jitter | RAW9 | MM8 (HEAD preamble) | BLK anySD | BLK+SUPER | SEG c8 |
|---|---|---|---|---|---|---|
| axz | j0 | 39.0 | 38.3 | **32.3** | — | 70.4 |
| axz | j1 | 45.0 | 43.6 | **35.0** | 42.5 | 91.6 |
| axy | j0 | 36.1 | 35.7 | **31.8** | — | — |
| axy | j1 | 37.7 | 36.6 | **31.3** | — | — |
| az45 | j1 | 22.3 | 21.3 | **19.8** | — | — |
| obl | j0 | 21.7 | 20.0 | **18.7** | — | — |
| obl | j1 | 22.4 | 21.0 | **19.7** | — | 47.8 |

**mm beats raw on every view at both jitter settings today.** The historical
"+10 ms mm-vs-raw at SD4 axes" anchor (51.4 vs 40.8) does not exist on the
current machine state; §35.x-session absolutes (55.08 mm-axz etc.) were
measured adjacent to the system-overload incident and are degraded-machine
artifacts. Relative orderings were stable then and now; absolutes shift up
to ±20% with machine state. All future anchors must record battery/thermal
state.

### 37.2 P1 ladder sweep — code-size cliff REFUTED, direction inverted

VTK_METAL_TEST_MARCH_CAP sweep on the §35.14 seg-consume path (which carries
the +48 ms pipeline penalty), axz j0: cap 8 → 70.4, cap 4 → 83.1, cap 2 →
118.6, cap 1 → 171.1. MM without seg shows the same direction (cap 4 → 47.1
vs cap 8 → 38.3). **The consume penalty GROWS as the batch width shrinks**
(≈ ∝ outer-iteration count), so it is per-iteration interaction, not
code-size/I-cache. §36.2's "smaller ladder frees headroom" hypothesis is
dead; keep cap ≥ 8 at coarse SD.

### 37.3 Blocks-anySD: uniformly fastest, but NOT byte-identical at DS=4

With VTK_METAL_TEST_MM_BLOCKS=1 VTK_METAL_TEST_MM_BLOCKS_ANY_SD=1 (2-level;
the SD<1.5 gate lifted by the probe env), blocks beat plain mm by 6-20% on
EVERY view and both jitters — including the obliques whose "~+1 ms" loss was
the original reason for the coarse-SD gate (§33). That rationale is refuted
on current machine state.

Parity vs HEAD-mm at SD4 (PNG pixel diff, j0):
- axz: 0.44% of pixels differ, mean |d| 1.03 LSB on differing px, max 21
- obl: 0.14% differ, mean 1.06 LSB, max 73 (2 px > 8)

This is the accepted ±1-step fp-landing class, NOT byte-identical: §33's
"byte-identical" claim holds only for the fine-SD DS=2 lattice where cell-
and block-boundary arithmetic coincide. At DS=4 the block-leap landing
arithmetic differs from the cell-walk's (+1e-4-per-cell chain), producing
rare single-pixel outliers on silhouettes.

BLK+SUPER (3-level) is WORSE than 2-level at coarse SD (axz j1: 42.5 vs
35.0) — super taps add per-batch work without payoff when block leaps
already cover the empty spans. Keep super off at coarse SD.

### 37.4 §36 path statuses after execution

- **P1**: done (env-only, no code needed — MARCH_CAP already existed).
  Refuted code-size cliff; §37.2.
- **P2 (pointer-vs-index)**: MOOTED. Its purpose was pricing Design A; the
  target vanished (§37.1) and P1 showed the residual seg-penalty is
  iteration-driven rather than fixed-cost, which P2's fixed-param swap would
  not have resolved anyway.
- **Design A (tap-and-leap)**: PREMISE DEAD. Its justification was
  "persistent state always cliffs" — but BLK adds cached block-state on top
  of mm and wins everywhere on this machine state. And its success bar is
  now BLK's 35.0/19.7, which transient taps (≥1 tap/batch, no caching)
  cannot beat against amortized <1 tap/batch cached lookups.
- **Design B (compute marcher)**: trigger unmet (>5 ms left on the table —
  there isn't, relative to BLK).

The seg pre-pass (§35.14) stays default-off investigation code; its consume
pipeline remains ~+32-48 ms and is now understood to be iteration-count-
driven, not register/code-size — do not resurrect it for skipping.

### 37.5 Recommendation (pending sign-off — image-class tradeoff)

Make the 2-level block summary the DEFAULT whenever GPU minmax is active,
for ALL sample distances (delete both the env requirement and the SD<1.5
gate; keep VTK_METAL_TEST_MM_SUPER opt-in off at coarse SD). Expected:
6-20% frame-time win over shipped mm on every view class at SD4, larger in
empty-heavy scenes; cost: sub-LSB-class deltas on <0.5% of pixels with rare
single-pixel outliers at coarse SD (fine SD stays byte-identical per §33).
If byte-exactness at coarse SD is non-negotiable, the alternative is gating
default-on to sampleDistance < 1.5 only (status quo ante) — but then SD4
keeps paying ~15% vs achievable.

### 37.6 EXECUTED same day: blocks ungated as recommended — and the
"byte-identical at fine SD" claim is also FALSE (2026-08-23 evening)

VolumeMinMaxBlocksWanted now returns true whenever the GPU minmax lattice is
active, all SDs (env requirement + SD<1.5 gate deleted;
VTK_METAL_TEST_MM_BLOCKS=0 is the kill switch; MM_BLOCKS_ANY_SD removed;
VTK_METAL_TEST_MM_SUPER stays opt-in). Verification:

| cell | blocks-off | default (blocks) |
|---|---|---|
| SD4 axz j1 | 41.4 ms | **36.0 ms (−13%)** |
| SD0.5 axz j0 | 135.6 ms | **98.2 ms (−28%)** |

Default renders are byte-identical to the explicit-probe-env renders (SD4
axz/obl), so §37.3's parity quantification carries over.

**Correction**: the §33 "byte-identical" claim fails at the FINE tier too.
SD0.5 axz j0, default vs MM_BLOCKS=0: 8276 px differ (0.20%), mean 1.26 LSB,
max 22, 128 px >8LSB — the same ±1-step landing class as DS=4, not zero.
Block emptiness is exact per-cell; what differs is the LANDING arithmetic of
block leaps vs the cell-walk's +1e-4-per-cell chain, at every DS tier. The
change therefore alters output everywhere within this accepted class; no
tier is byte-exact anymore. Kill switch (MM_BLOCKS=0) restores bit-exact
legacy output for byte-diff regression tests.

### 37.7 Same-binary re-measurement resolves the §35.10/§35.11 conflict —
those findings were machine-state artifacts too (2026-08-23 night)

Challenge from the conflicting doc record: §35.10/§35.11 (this same day,
morning) measured mm LOSING to raw at SD4 on axis cameras (axis-z raw
39.05 / mm 54.42, +40%; axis-y +27%) and concluded a "residual structural
gap untunable by lattice shape". §37.1 (evening) measured mm winning
everywhere. Settled by re-running the §35.10 cells on the ACTUAL
9c91995253 binary (=1e0660cf, docs-only delta) checked out and rebuilt
tonight, back-to-back with HEAD:

| cam-axis z, SD4 mv9 j1 | RAW | plain mm | verdict |
|---|---|---|---|
| 9c91995253, morning (doc) | 39.05 | 54.42 | mm +40% LOSS |
| **9c91995253, tonight** | **61.77** | **58.38** | mm −5.5% WIN |
| HEAD tonight (blocks off via MM_BLOCKS=0) | 61.92 | 58.48 | identical |
| HEAD tonight (default = blocks on) | 61.92 | 51.06 | −17% vs raw |

cam-axis x confirms (old binary tonight): raw 44.3 / mm 43.6 — where the
morning doc had mm losing +9% (§35.9).

Findings:

1. **The §35.10 ordering flip does not survive its own binary.** Same
   code, same protocol, same day: mm loses +40% in the morning session,
   wins −5.5% at night. The "mm loses at SD4 axes" family (§35.10's
   penalty table, §35.11's "residual structural gap", policy option
   "(a) keep mm off at SD>=1.5") is machine-state artifact, refuted.
2. Old-binary and HEAD timings agree within noise tonight — none of the
   code deltas between 9c91995253 and HEAD moved the raw/mm arms; only
   the blocks ungate (03dd74b048) improves them further (−12% more on
   axis-z).
3. Machine state swung absolute numbers ±30–60% WITHIN one day (raw-z
   39→62), large enough to invert relative orderings when two configs
   sit within ~10% of each other. Protocol upgrade, mandatory:
   **record battery/thermal state with every anchor table; never compare
   anchors across sessions for verdicts — re-measure both arms
   interleaved in the SAME session, order-alternated.**
4. Standing results that DO hold across all sessions regardless of
   machine state: blocks beat plain mm wherever both were measured
   together (morning §35.10 forced-blocks AND tonight §37); VOLTRANSPOSE;
   the occupancy-cliff law's existence (though §37.2 re-priced its
   ladder dependence); §35.14's seg consume pipeline cost.

## 5. Files

- `JITTER_DUMP.txt` — jitter investigation dump (interleaved j1, sample-count PPMs).
- `J0_GAP_DUMP.txt` — j0 investigation dump (interleaved rounds, fixed-steps, timer audit).
- `jitter_gap_repro/` — harness + `RESULTS.md` (build/run/knobs, verdicts; now also GL41/GLSTORAGE knobs).
- `jitter_gap_repro/RESULTS.md` — full experiment record.
- `../SLAB_BENCHMARKS.md` — app bench recipe §1, pathological cell §7.2.
- `../jitter_lag_repro/` — sharp-field (IGN) vs tile-field A/B harness.
- `../divergent_tail/README.md` — V31/V24/V32 root-cause record (source of DOEXIT/RG8 ports).
- Session 2026-08-21 logs: `/tmp/appv31` (DOEXIT j0 A/B), `/tmp/jsmat` (JSCALE dose matrix),
  `/tmp/np`, `/tmp/near`, `/tmp/pzsweep`, `/tmp/azsweep` (orientation/azimuth sweeps),
  `/tmp/matrix` (RG8 regression matrix + GL knob tests), `/tmp/rg8t` (final oblique matrix),
  `/tmp/vis_*`, `/tmp/vis2k_*` (visual exports), `/tmp/vis2k_diff_heat.png`.
- Session 2026-08-22 night logs (§31): `/tmp/mvmatrix/` (mv0-vs-mv9 grid),
  `/tmp/mvmatrix_az/` (azimuth compass), `/tmp/recheck_tr_mm/` (tr×mm SD0.5
  grid); generator scripts embedded verbatim in §31.4.
- Session 2026-08-22 late-night logs (§32): `/tmp/trmm_fix/`, `/tmp/trmm_ab/`,
  `/tmp/trmm_abba/`, `/tmp/trmm_abba_tr/` (timing matrices), `/tmp/trmmimg/`,
  `/tmp/trmmimg2/` (parity PNG exports); reference binaries
  `/tmp/vtkPreFixMM` + `/tmp/vtkFixedMM` (volatile — rebuild per §32.2 to
  reproduce).
- App: `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm`,
  `Rendering/Metal/Shaders/MetalShaders.metal`,
  `Rendering/VolumeOpenGL2/vtkOpenGLGPUVolumeRayCastMapper.cxx`.
