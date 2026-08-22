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
- App: `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm`,
  `Rendering/Metal/Shaders/MetalShaders.metal`,
  `Rendering/VolumeOpenGL2/vtkOpenGLGPUVolumeRayCastMapper.cxx`.
