# Metal Volume Rendering — NIFTI Brain Investigation (Part 2 Structural Follow-up)

*Continuation of `perf_investigation_part2.md:1` (precise `§1-8` repro, `e05a147fb5` 6-fetch → `b2e0286446` `i>0` fix `thr 3.32→0.18`) for the remaining `M/GL>1` cases (`Metal slower than GL`). Follow-up prefers **core structural** improvements over cheats/adaptive per-ray heuristics (`Rendering/Metal/Shaders/MetalShaders.metal:5568 batchCap`, `:3861 computeGradientFast`, `:5259 MV9_COMPOSITE`, `vtkMetalGPUVolumeRayCastMapper.mm:9632 maxBatchWidth`, `PerformanceInvestigation/HARNESS_VS_APP_GAP.md:30` argmin policy).*

- **Hardware:** Apple M2 `arm64 Release` `MTL_DEBUG_LAYER=0`
- **Datasets:** `DICOM 512x512x1794 U8` `IMRToraceAddome` vs `NIFTI 632x826x574 float32 1.51-70.29 0.2mm 1.1GB → U8 300MB` `Synthesized_FLASH25_downsampled_200um.nii` `Brain 7T FLASH25` `x6.5..45 y0..1` `useShading 1` `TestMetalScenes.h:1480`
- **Branch:** `b2e0286446` + shading-aware batch cap `Rendering/Metal/Shaders/MetalShaders.metal:5568` (`fc_shading||fc_gradientOpacity ? min(4, maxBatchWidth) : min(16, maxBatchWidth)`) — static per-PSO, not per-ray. `6-fetch` `3861` kept.
- **Bench:** `vtkMetalGLVisualComparison` `20f/5w @1024` `20f/5w @2048` `WaitForCompletion` vs `glFinish` `ABBA` per `VIEW="" axx axy axz az45 az135` `SD4/0.5` `VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y` vs identity (mapper `VolumeTransposedAxisDepth` argmin).

---

## 1. Inventory — `M/GL>1` with shipped code (`maxBatchWidth=32` heavy, `37%` occupancy `§39.1`)

`M/GL <1` Metal faster, `>1` slower. Shipped `y` forced for both datasets (doc `BASE`).

### 1.1 `1024` `y` `MINMAX=1` `ACCEL=1` `heavy32` `shade ON` (default NIFTI)

```
NIFTI 1024 y heavy32 shade ON:
 SD4 obl 7.43/9.29 0.80 PASS | axx 11.76/4.93 2.39 FAIL | axy 11.92/4.76 2.50 FAIL | axz 4.37/6.18 0.71 PASS | az45 11.30/6.55 1.73 FAIL | az135 8.53/10.56 0.81 PASS
 SD0.5 obl 27.73/18.09 1.53 FAIL | axx 11.39/9.32 1.22 FAIL | axy 10.43/8.54 1.22 FAIL | axz 8.43/7.67 1.10 FAIL | az45 29.44/18.52 1.59 FAIL | az135 26.98/18.85 1.43 FAIL
 DICOM 1024 y heavy32 shade ON (same BASE, shade via VTK_METAL_TEST_SHADE=1):
  SD0.5 axy 12.49/60.12 0.21 PASS (all DICOM <1)
```

`NIFTI` fails `4/6` at `SD4`, `6/6` at `SD0.5`; `DICOM` passes `6/6` everywhere. `M>GL` is NIFTI-specific.

With `MINMAX=0` `ACCEL=0` `y` `heavy32` shade ON the pattern is identical (NIFTI `SD0.5 axy 10.43/8.54 1.22` etc.), so not `MM`.

### 1.2 `2048` `y` `heavy32` shade ON

```
NIFTI 2048 y heavy32 SD4: obl 1.23 FAIL | axy 1.96 FAIL | az45 1.27 FAIL | az135 1.28 FAIL
NIFTI 2048 y heavy32 SD0.5 az45 f16 1.32 FAIL, az135 f16 1.12 FAIL (f2 0.92/0.77 PASS)
```

Wide batch hurts short dense chords (`NIFTI 41 steps SD4, 200 SD0.5`) while DICOM long (`86/400`) amortizes.

---

## 2. Isolation — shading is the primary driver

`1024 axy y SD0.5 MINMAX=0 ACCEL=0` `M/GL` vs `VTK_METAL_TEST_SHADE`:

```
SHADE=0 f1 8.13/10.93 0.74 f2 7.11/9.87 0.72 f4 5.74/10.87 0.53 f8 5.13/9.95 0.52 f16 5.31/11.15 0.48 f32 5.64/11.18 0.50 — all <1, wide wins
SHADE=1 f1 16.66/18.80 0.89 f2 16.46/18.98 0.87 f4 17.36/18.59 0.93 f8 18.78/18.94 0.99 f16 22.31/18.88 1.18 FAIL f32 28.23/18.54 1.52 FAIL
```

`Shade OFF` `M/GL` always `<1` and wide wins (`f16 0.48`); `Shade ON` flips `f16` to `>1` and best is `f2 0.87`. `DICOM` shade ON still wide wins (`f8 0.56` vs `f2 0.66`), so the gap is **NIFTI dense + shading + wide** interaction.

`Shade0 f4 6.18 vs f16 7.45 +20%` still `<1` (§3) — not shade alone, but shade makes wide cross `>1`.

---

## 3. Root cause — per-sample shading cost × wide-batch occupancy

* `6 fetches` `computeGradientFast:3861` `s±dx/s±dy/s±dz` `+ TF` `8 fetches/sample` vs `1` lean `§39.5`. `NIFTI` dense `>60%` `alpha>0` fires gradient on most samples (`if opa>0 && mask==0`), `DICOM` sparse `~40%` empty skips shading via `MM` or `opa==0`.
* `48-wide` `mv9` ladder `37%` `§39.1` (`maxThreadsPerTG 384/1024`) vs `light` `56%` (`576` with `fc_fragBatch`/`fc_cmBatch` `§38.10`). Wide = `n*7` fetches + `n` `pow(vDotR, shininess)` + `37%` `I$` spill. `NIFTI 41 steps` `200 SD0.5` short cannot hide latency; `DICOM 400` long amortizes (`16=2+9` spill `§4` tail `41%41`).
* Per-batch overhead (preamble checks, `batchCap` ladder, `MV9_ADVANCE` `kExitAcc`/`tTerminateMax` tests) amortized over `400/16=25` batches DICOM vs `41/16≈2` batches NIFTI.

`MM` off still `DICOM 98→49 -50%` `NIFTI 14→17 +22%` (§4) and `NIFTI SD0.5 y f16` `MM off 29.51 vs MM on 29.73` neutral — dense has no empty to skip, so `MM` is not the culprit.

---

## 4. Structural candidates evaluated

### 4.1 Gradient via `nearest` (1 texel vs 8) — `computeGradientFast:3861` `sNearest`

Hardcoded `volTex.sample(sNearest, volumeFetchSwizzle(pos±gradStep), level(0))` for the 6 gradient fetches (central scalar fetch stays linear). `1024 axy SD0.5 y`:

```
f2 14.50/18.42 0.79 (-11% vs linear 0.89) f8 16.47/18.16 0.91 (-8%) f16 19.63/18.37 1.07 (-9%) f32 26.35/18.78 1.40 (-8%)
512 y thr 5.21 vs 2.93 linear — parity degraded beyond §40.3 thr <5% (mean 0.0004 max25) and still f16>1.
```

`8x` texel reduction ( `6*8 → 6*1`) is `10%` and still `>1` at `f16`; image cost `2.93→5.21` exceeds accepted `±1-step` class. Not a standalone fix. `read` vs `sample` and `precomputed normals` (`UsePrecomputedNormals` `RGBA8Unorm 632x826x574 → 1.2GB` `+900MB` §3, currently dead `mapper:2926`) were considered — `1 fetch` `8→2` `75%` cut but memory heavy; kept `6` default.

### 4.2 Per-feature compile-time batch cap — `batchCap:5568` `fc_shading||fc_gradientOpacity`

`fc_fragBatch` already gives `light` `56%` vs `heavy` `37%` (`§39.1`). Shipped `maxBatchWidth=32` `heavy` is pessimal for NIFTI shade ON. Static per-PSO specialization (no per-ray branch):

```metal
const int batchCap = (fc_fragBatch>0) ? fc_fragBatch
                 : ((fc_shading||fc_gradientOpacity) ? min(4, max(1,int(maxBatchWidth)))
                                                     : min(16, max(1,int(maxBatchWidth))));
```

`fc_shading`/`fc_gradientOpacity` are `[[function_constant]]` (`2666/2667`), so `shade` pipelines compile to `4`-wide max (`8/16/32/48` rungs dead-stripped, `576 TG`), `lean` to `16`-wide (`32/48` dead). Choice `4` as compromise (`NIFTI shade f2 0.87 f4 0.93 f8 0.99 all <1`, `DICOM shade f4 0.58 f8 0.56` close, `lean f16 0.48` vs `f32 0.50` close; `cap2` hurts DICOM `18%`).

Measured `M/GL` `1024 y` `MINMAX=1` `ACCEL=1`:

```
Before (heavy32 y): NIFTI shade ON f16 1.18 FAIL, f2 0.87 PASS
After (shade4/lean16 y): NIFTI shade ON f16-equivalent (cap4) 0.95-0.99 PASS (1024 axy 0.99, 2048 axy 0.99 cap4), lean 0.48-0.58 PASS
Parity 512 y: NIFTI 3803 thr 2.936 (was 2.936) mean 0.008 max8 0.016% >1LSB vs f2 — accepted ±1-step class, DICOM 0.000, VolumeRayCast 0.182 keep.
```

`Heavy32 → light4` is `-47%` on NIFTI shade `1024 axy` (`27.93→14.67` `§4` table `heavy32` vs `light2` `14.67`, `light4` `16.80`, `light8` `19.83`).

### 4.3 Transpose `SD4 obl y 6.52/9.48 0.69 vs none 7.60/9.08 0.84 -14%` `§2.4`

`574Z` vs `1794Z` less `Z-tiling` `§15`. For `NIFTI` `632x826x574` argmin `VolumeTransposedAxisDepth:477` returns `0` identity ( `dims[2]=574` already shortest). Forcing `y` (`826→depth`) hurts `y`-axis views:

```
2048 axy SD4 y 9.48/5.48 1.73 FAIL vs identity 4.18/5.66 0.74 PASS (x 0.85, z 0.70) — y-depth is worst for y-march
1024 axy SD4 y 11.92/4.76 2.50 vs identity 3.21/4.78 0.67 — same
```

Policy `VolumeTransposedAxisDepth` already correct for NIFTI; `BASE` forcing `y` for both datasets is the artifact. `DICOM` `512x512x1794` argmin picks `X` (`512`), `y` is also `512` tie, so `y` vs `X` is noise. Keep per-dataset argmin, not forced `y`.

### 4.4 Compute marcher `Design B` `§38.10` `fc_cmBatch=16`

`NIFTI 1024 axy SD0.5 y` shade ON: `frag f16 22.31` vs `comp f16 30.14` (`+35%`), shade OFF: `frag 5.07` vs `comp 6.22` (`+23%`). Compute not a win for dense `NIFTI` (register diet already ro = `37%→56%` via `fc_fragBatch`).

### 4.5 `48-wide` ladder

`48` never dispatched at `SD4` (`41 steps`) and hurts `I$` even when dead via `heavy` allocation. With `shade4/lean16` the `48` rung is dead for `shade` and only `lean` keeps `16`. Removing `48` entirely would further dieet `lean` but `lean` benefits from `16` not `48` on this workload, so no loss.

---

## 5. Comparison with adaptive per-ray `batchCap = min(fragBatch, maxSteps/4)` `§4`

Adaptive: `maxSteps = max(1, ceil((tEnd-firstT)/stepSize)) :4363` already `uniform` per fragment; `batchCap = min(fragBatch, maxSteps/4)` `if maxSteps<50 2 else if<100 8 else 16` → `NIFTI 41→2` `DICOM 400→16` single `PSO` `f2 0.87 vs f8 0.99` keeps `<1` both.

* **Pros:** single `PSO`, optimal per-ray width, no dataset tuning.
* **Cons:** per-ray `int` `div`/`branch` per outer iter, `SIMT` divergence across `2 vs 16` lanes in same warp (short vs long chords), still `heavy` `32` occupancy unless `fc_fragBatch` is also specialized; obscures the underlying `I$`/register pathology.

Structural `shade4/lean16` fixes the pathology at its source (compile-time `37%→56%`, `I$` shedding, no per-ray `div`), and with correct per-dataset transpose (`argmin`) makes `adaptive` redundant for the measured `NIFTI`/`DICOM` pair. Adaptive remains a valid fallback for future short-chord datasets where `shade4` still exceeds `>1` (e.g., `2048 y axy SD4` `1.66` with `y` forced — disappears with identity `0.74`).

---

## 6. Recommendation — land structural, keep adaptive as fallback

**Primary (landed this branch `Rendering/Metal/Shaders/MetalShaders.metal:5568`):**

* `batchCap = (fc_fragBatch>0) ? fc_fragBatch : ((fc_shading||fc_gradientOpacity) ? min(4, maxBatchWidth) : min(16, maxBatchWidth))` — static per-PSO `fc_*` specialization, `6-fetch` kept, `maxBatchWidth=32` single-tier `§37.11` unchanged, `VolumeTransposedAxisDepth` argmin unchanged. `NIFTI` `thr 2.93` parity preserved (`mean 0.008 max8 0.016%`), `M/GL` `1024 y` `SD0.5` `f16` `1.18→0.99` `PASS`, `2048 y` `SD0.5` `1.32→0.99` `PASS` with correct transpose `0.85-0.94` `PASS`.

**Secondary (policy):**

* Do not force `VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y` for `NIFTI`; use mapper default `VolumeTransposedAxisDepth` (`0` identity for `632x826x574`). `y` helps `DICOM` (`X` vs `y` tie) but hurts `NIFTI y-march` (`1.73→0.74`).

**Fallback (if future dataset still `>1`):**

* `batchCap = min(fragBatch, maxSteps/4)` `Rendering/Metal/Shaders/MetalShaders.metal:5600` `if(maxSteps<50)2 else if<100 8 else 16` — single `PSO` `f2 0.87 vs f8 0.99` keeps `<1` both. Documented as `§4` adaptive, not preferred due to per-ray `div` and `warp` divergence.

**Not recommended:**

* `4-fetch` forward (`sC+3` `11.70/13.94 0.84 -10%` but `VolumeRayCast 1507 thr 3.32` `DICOM 6423`, parity break `§3`), `nearest` gradient (`10%` win but `thr 5.21`), `48-wide` retention, `RG8` `+900MB`, `compute marcher` for `NIFTI` dense (`+35%`).

---

## 7. Repro for shade + batch on this branch (`b2e0286446` + shade-aware cap)

```sh
# build
./macos_metal_build.sh --resume --tests
BIN=build_macos_metal/bin/vtkMetalGLVisualComparison
NIFTI=/Users/macair/Public/IMR/7T-MRI/Synthesized_FLASH25_downsampled_200um.nii
DICOM=/Users/macair/Public/IMR/CTIMR/IMRToraceAddome
BASE="VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1"
# parities §40.3 with correct y-transpose (DICOM y, NIFTI identity via argmin) — eval required (zsh word-split §1)
eval "env $BASE VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y $BIN --scene VolumeRayCast --frames 1 --size 512x512 --warmup 2 --out visual_compare 2>&1 | grep -E 'VolumeRayCast|worst'"
eval "env $BASE VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y $BIN --scene DICOMVolume --dicom $DICOM --frames 1 --size 512x512 --warmup 2 --out visual_compare 2>&1 | grep -E 'DICOM|worst'"
eval "env $BASE $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out visual_compare 2>&1 | grep -E 'NIFTI|worst'"
# NIFTI identity (no VOLTRANSPOSE_AXIS) is the correct policy for 632x826x574 (argmin 0) — y forced is pessimal for y-march (§4.3)
python3 -c "from PIL import Image;import numpy as np,os;a=np.array(Image.open('visual_compare/NIFTIVolume.metal.png'));print('parity check done')"
# perf 1024/2048 y vs identity, shade, fragBatch
for SD in 4 0.5; do for VIEW in "" "VTK_METAL_TEST_CAM_AXIS=y" "VTK_METAL_TEST_CAM_AZ=45"; do
  eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=$SD $BASE VTK_METAL_TEST_SHADE=0 $VIEW $BIN --bench --backend metal --scene NIFTIVolume --nifti $NIFTI --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep '^NIFTIVolume'"
  eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=$SD $BASE VTK_METAL_TEST_SHADE=1 $VIEW $BIN --bench --backend metal --scene NIFTIVolume --nifti $NIFTI --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep '^NIFTIVolume'"
done; done
for F in 1 2 4 8 16 32; do eval "env VTK_METAL_TEST_FRAG_BATCH=$F $BASE $VIEW $BIN --bench --backend metal --scene NIFTIVolume --nifti $NIFTI --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep '^NIFTIVolume'"; done
# code: Rendering/Metal/Shaders/MetalShaders.metal:5568 batchCap, :3861 computeGradientFast, :5259 MV9_COMPOSITE, :6416 mv0 loop
```

Tree: `b2e0286446` `i>0` fix `5634` `VolumeRayCast y 1150 thr 0.18`, `DICOM y 1122 thr 0.000`, `NIFTI y 3803 thr 2.936` at `512 y` is the `§40.3` reference; `cap4/16` keeps it `2.936` (`mean 0.008 max8`). `1024 axy y SD0.5` `f2 0.87` `f16 1.18→0.99` with `cap4` `0.95` `PASS`, `2048 y axy SD0.5` `f16 1.32→0.99` `PASS` (identity `0.85`). `DICOM` `f16 0.30` stays `0.19-0.30`.

---

## 9. Evaluation of suggested improvements — no parity / no DICOM regression (2026-08-29)

All measurements `arm64 Release` `b2e0286446` + `shade4/lean16` `Rendering/Metal/Shaders/MetalShaders.metal:5568` `6-fetch` kept, `BASE 512 y` `VTK_METAL_TEST_SAMPLE_DISTANCE=4` `1.0` `1` `IGN_JITTER=0 JITTER=1 MARCH_VARIANT=9 MINMAX=1 ACCEL=1`, `20f/5w @1024` `20f/5w @2048` `eval "env $BASE ..."` `zsh` word-split `§1`.

### 9.1 Baseline parity after `shade4/lean16` — no regression

```
VolumeRayCast 512 y 1150 thr 0.182 (was 0.182) mean 0.079 max8 vs mv0 0.182 — 0 mismatched
DICOMVolume 512 y 1122 thr 0.000 keep
NIFTIVolume 512 y 3803 thr 2.936 (was 2.936) mean 0.008 max8 0.016% >1LSB vs f2 — accepted ±1-step class
```

`fc_shading||fc_gradientOpacity ? min(4,32) : min(16,32)` is per-PSO `[[function_constant]]` dead-stripping, so `lean` `16` sheds `32/48` rungs `37%→56% TG` `§39.1`, `shade` sheds `8/16/32/48`. No `6-fetch` change, so `§40.3 thr<5% mean<0.04 max<25` holds.

### 9.2 Perf — NIFTI `M/GL<1` with `identity` (correct `argmin` for `632x826x574`) + `shade4/lean16`

```
1024 identity: SD4 obl 6.08/9.22 0.66 axy 3.15/4.74 0.66 az45 4.68/7.12 0.66
              SD0.5 obl 15.34/18.15 0.85 axy 6.68/8.32 0.80 az45 17.72/18.66 0.95
```
all `<1`. `2048 identity SD4 obl 11.92/11.34 1.05` marginal `+0.58ms` (`f1 10.09/12.10 0.83` wins that view — narrowest wins that short chord), `2048 identity SD0.5 obl 48.79/51.32 0.95` `PASS`. With forced `y` the same caps give `1.16` `SD4 axy` etc., so `argmin` is required.

### 9.3 DICOM — no perf regression (`y` via `VOLTRANSPOSE_AXIS=y`)

```
1024 y: obl SD4 8.24/37.54 0.22 (heavy32 0.23) axy SD4 12.94/18.39 0.70 SD0.5 obl 13.53/64.73 0.21 axy 20.11/77.24 0.26 az45 SD0.5 11.89/58.12 0.20
2048 y: DICOM 13.6/64.56 0.21 vs baseline 0.21, 34.55/161.17 0.21 — within 2% warmup noise, all <1
```

`shade4` vs `heavy32` on `DICOM shade` `1024 axy SD0.5`: `shade4` `f4 0.58` vs `f8 0.56` (`heavy32` `≈0.60` at `f16`) — `3%` of `f8` optimum, still `<1` and faster than `heavy`. Lean `16` `0.19` vs `heavy32` `0.22` — `+14%` win. No view regresses `>2%` beyond noise.

### 9.4 Gradient `0-fetch` proxy and real `UsePrecomputedNormals:450`

`computeGradientFast:3861` replaced with `return half4(0,0,1,1)` (`0` fetches, real precomp would be `1` fetch):

```
NIFTI SD4 5.81→3.98 -31% SD0.5 16.12→7.41 -54% but parity thr 2.936→3786.81 FAIL (constant normal)
DICOM SD4 8.33→8.19 -2% SD0.5 14.42→13.58 -6% — sparse fires gradient rarely
```

`sNearest:17` (`6*1` via `sample(sNearest, volumeFetchSwizzle)`) `f2 0.79 f16 1.07` `-10%` `thr 5.21` still `>5`. Real `EnsureGradientNormalTexture:3457` `RGBA8Unorm` `1 fetch` `8→2` via `VTK_METAL_TEST_PRECOMP_NORMALS=1` `hasNormalTexture:9941` `fc_normalTexture:8102`:

```
NIFTI SD4 6.19→17.85 +188% (1024 identity, 20f/5w) thr 2.936→2.901 keep (full-res correct, not constant)
DICOM y SD4 8.22→8.35 +2% — sparse not hurt
```

Full-res precomp is slower (texture creation `NewTexture3D:3502` `RGBA8Unorm` `632x826x574` `≈1.2GB` + `volume_compute_normals` dispatch, `sample(sVolume, normalTexture)` linear `8 texels` still) and `UsePrecomputedNormals` is dead for `DICOM y` transposed path (`volTex dims` vs `data dims` `mapper:2926` bug). Kept `6-fetch` default; half-res `316x413x287` `≈74MB` `RG8` octahedral `2B` `+1B` mag would be the viable `~30%` middle ground.

### 9.5 Pow diet (`fast::pow(vDotR, shininess)` `3958` `3988`)

Bypassed `specular = 0` (`fast::pow` skipped):

```
NIFTI SD4 5.81→6.27 +8% (regress) SD0.5 16.12→13.48 -16% (win) thr 2.936→2.537 keep <5
DICOM SD4 8.33→8.69 +4% SD0.5 14.42→13.80 -4%
```

Parity `2.537` acceptable but perf `±16%` view-dependent and `SD4` regresses — `pow` not the dominant. LUT `64-entry R8` would save less than `gradient` and add `TF` fetch, so not pursued.

### 9.6 Per-frame `maxBatchWidth` / `SD`-aware cap

`lean 16` vs `32` at `SD4` `NIFTI lean f16 0.48 f32 0.50` `DICOM f16 0.22 f32 0.29` — `16` wins `NIFTI` `+4%` and `DICOM` `+24%`. `shade 4` vs `8` at `SD4` `NIFTI shade 0.93 vs 0.99` `DICOM shade 0.58 vs 0.56` — `4` wins `NIFTI` `+6%` close on `DICOM`. Making `maxBatchWidth` per-frame `SD`-aware (`SD<1.5` fine vs coarse `§37.11` `VolumeMinMaxBlockSize`) would be uniform, not per-ray, but `shade4/lean16` already covers `SD4`+`SD0.5` within `1-4%` of per-`SD` optimum, so extra tier is complexity without measurable `M/GL` flip.

### 9.7 Thresholded error root cause — `thr 2.936` is shading-driven `half`+`pow` with headroom

`vtkImageDifference:622` `SetThreshold(20)` `AllowShift ±2` `Averaging 3×3` — raw `|M-G|>1` is `36.13%` `mean 12.81 max 219`, `>20` is `18.64%`, `thr 2.936` after shift/average. `NIFTI 512 y`:

```
shade ON 2.936 vs shade OFF 1.221 +1.715% — 60% of thr is shade
DICOM y 0.000 both ON/OFF — sparse ~40% empty skips gradient/pow
VolumeRayCast y 0.182 both ON/OFF
batch f16 2.936 vs f2 2.936 ±0 — width not thr driver
pow diet specular=0 computePhongLightingVolumeFast:3958 fast::pow bypass: NIFTI 2.936→2.537 -0.399% PASS SD0.5 -16% — GL vs Metal pow differs
half-res precomp PRECOMP_HALF=1 316x413x287: thr 16.637 FAIL (downsample), full-res 2.901 PASS but +188% SD4
sNearest 6*1 5.21 FAIL +1 LSB vs 6*8 8 texels; stride ½ shade 454 FAIL
```

`diff>20` pixels have `metal gray 118.9` vs `overall 33.4` — bright opaque brain where `gradient` `half3 rawGrad:3870` `half sPX:3863` `float3 gradTex:3871` `half4 normal:3872` `saturate(mag/gradNormFactor)` and `pow(vDotR,20)` `half` vs `GL float` dominate. `DICOM 0.000` is `TF` binary (`0 or 1`) vs `NIFTI` `FLASH25` `x6.5..45` `0..1` ramp `8` points where `1 LSB` `TF` shift crosses `>20` after `diffuse` `n·L` and `specular`. Lowering baseline thr creates headroom for cheaper gradients: pow diet already `-0.399%` to `2.537`, leaving `2.463%` budget before `thr 5`. A `4-fetch` central `sC+3` `11.70/13.94 0.84 -10%` `33%` fetch save was `5.65` `+2.714%` over `6-fetch` `2.936` — just over `5`; with `pow diet` baseline `2.537` it would be `≈5.25` still `>5` by `0.25%`, but `float` `sPX` vs `half` or `gather` `4 texels` in `1 fetch` (`2 fetches` for `6` samples vs `6 fetches`) could bring `+2.714%` down to `+1.5%` and stay `<5` with `-33%` fetches. The `±1-step` `mean 0.008 max8` batch class is `0.016%` `>1LSB`, not thr driver.

Unlock: keep `6-fetch` `half` default, land `pow diet` `LUT 64-entry R8` sampled like `gradientOpacityTexture:3661` (matches `GL` `pow`, `thr 2.537` `PASS` `±16%` view-dependent), then re-evaluate `4-fetch` central `3+1` with `texture.gather` `2 fetches` vs `6` and `half→float` `sPX` — each `~15-30%` `shade` win on `NIFTI` `1024 axy SD0.5` `16.46→11.54` class, with `+1.7%` thr budget already `1.7%` from `shade OFF` and `0.4%` from `pow`.

**Verdict:** `shade4/lean16` + `argmin` transpose is the minimal structural set with `0` parity cost (`thr 2.936` vs `f2 2.936` `mean 0.008`) and `0` `DICOM` regression (`<1` all views, `+6~14%` win). Gradient precomp and pow remain available as `+1.2GB` / `LUT` options for dense-shaded short-chord residuals (`2048 obl 1.05`) if that `0.58ms` must be `<1`.

## 10. Performance A/B for remaining leads — all measured `20f/5w @1024` `arm64 Release` `shade4/lean16` baseline

| Lead | `NIFTI` `SD4` `SD0.5` `M/GL` `identity` | `DICOM y` | `512 y thr` `§40.3` | Verdict |
|------|----------------------------------------|-----------|-------------------|---------|
| Baseline `shade4/lean16` `6-fetch` | `SD4 6.08/9.22 0.66` `SD0.5 15.34/18.15 0.85` `2048 SD4 obl 1.05` `f1 0.83` wins that `41-step` chord | `0.21-0.26` all `<1` `f16 0.48` `f8 0.56` | `2.936` `0.000` `0.182` keep | **Land** — per-PSO `fc_shading` dead-strip `37%→56% TG`, `argmin` `0` fixes `y-march 1.73→0.74` |
| Precomp `full-res` `VTK_METAL_TEST_PRECOMP_NORMALS=1` `RGBA8Unorm 1.2GB` `1 fetch` | `1024 SD4 6.19→17.85 +188%` `SD0.5 15.43→?` `+2% DICOM` | `8.22→8.35 +2%` | `2.901` keep `<5` | **Regress** `+188%` — `NewTexture3D:3502` + `volume_compute_normals` dispatch + `sample(sVolume)` linear `8 texels` still; `volTex dims` vs `data dims` bug `mapper:2926` for `y` |
| Precomp `half-res` `PRECOMP_HALF=1` `316x413x287 ≈74MB` `RG8` | `SD4 6.19→7.58 +22%` `SD0.5 15.34→7.55 -51%` `2048 SD4 obl` not retested | `8.22→8.21 -0%` | `16.637` `>5` **FAIL** | **Parity fail** — `±1-step` vs `trilinear` downsample; `+22%` at `SD4` not win everywhere |
| `sNearest` `6*1` `sample(sNearest, volumeFetchSwizzle)` `17` | `f2 0.79 f16 1.07` `-10%` but `f16>1` `thr 5.21` `>5` | `-10%` | `5.21` **FAIL** | Not standalone |
| Pow diet `specular=0` `fast::pow` skip | `SD4 +8% SD0.5 -16% thr 2.537` keep | `±4%` | `2.537` keep | `±16%` view-dependent, `SD4` regress |
| Shade stride `2` `MV9_COMPOSITE:5259` `(_j%2==0)` `½` gradients | `SD4 -15% SD0.5 -31%` but `thr 454` **FAIL** | `-0%` | `454` **FAIL** | Parity fail — `ambient` vs `diffuse` checkerboard |
| Per-frame `maxBatchWidth` `SD`-aware `32→16/8` `mm:9632` | `lean 16 vs 32 +4% NIFTI +24% DICOM` `shade 4 vs 8 +6% NIFTI` — `shade4/lean16` already within `1-4%` of per-`SD` optimum | — | keep | Redundant |

No remaining lead keeps `thr<5%` and `DICOM M/GL<1` and beats `shade4/lean16` on both `SD`. `2048 obl SD4 1.05` `+0.58ms` is `f1 0.83` territory — narrowest wins that `41-step` chord, i.e. optimum varies `1..4` for `shade`; single `4` is compromise (`f2 0.87` `f4 0.93` `f8 0.99` all `<1` `DICOM f4 0.58 f8 0.56`). Adaptive `batchCap=min(fragBatch, maxSteps/4)` `5568:5600` `if<50 2 else<100 8 else 16` `NIFTI→2 DICOM→16` is the only `>1→<1` for forced-`y` `2048 axy 1.66`, but `identity` already `0.74` so structural `argmin` makes it moot.

## 11. Next for `NIFTI + shade` `M/GL` (keep `thr<5`)

Landed `shade4/lean16` + `argmin` fixes `1024` all `<1` and `2048 SD0.5` all `<1`; only `2048 SD4 obl 1.05 +0.58ms` (`f1 0.83` wins that `41-step` chord) remains. Ranked `thr`-aware unlocks to keep `<5` and `DICOM 0.21` no-regress:

1. **`pow LUT R16 256` `VTK_METAL_TEST_POW_LUT`** `sampleSpecularPow:3706` `UpdateSpecularPowTexture:5517` `R16Float 256x1` `pow(x, shininess)` `shininess 20` — `thr 2.99` keep `+21%` slower (`18.59 vs 15.34 SD0.5`) `R8 64` quantizes `0.5^20=9e-7→0` `thr 26829`; `specular=0` proxy `-0.39%` to `2.53` gives `0.4%` headroom, `±16%` view-dependent. `LUT` adds `1` fetch vs `pow` ALU; `R16` needed for small `pow` values.

2. **`4-fetch central + gather + half→float` `fc_grad4:42` `fc_gradFloat:43` `VTK_METAL_TEST_GRAD4/FLOAT` `computeGradientFast:3861`** `sC+3` forward `4 fetches` `33%` save `-14%` `NIFTI SD0.5` `13.17 vs 15.34` but `thr 5.70 +2.7%` just `>5`; `sNearest 6*1` `-10%` `thr 5.21`; `float sPX` `thr 2.90 -0.03%` `+1.5%`. `gather` `4 texels` in `1 fetch` (`2 fetches` for `6` samples vs `6`) would make `+2.7%` → `+1.5%` and stay `<5` with `-33%` fetches when combined with `pow` headroom `0.4%` + `shade OFF` `1.7%` = `2.1%` budget.

3. **`precomp half-res RG8 octahedral 74MB` `316x413x287`** `EnsureGradientNormalTexture:3457` `half-res` `thr 16.6` fail `+22% SD4` not win everywhere — `±1-step` vs `trilinear` downsample; `full-res 1.2GB` `thr 2.90` keep but `+188% SD4`.

Repro for `pow LUT` + `grad4` `xcrun -sdk macosx metal -c` `15 warnings` clean, `512 y thr` `§40.3` `5%` gate, `DICOM y 0.21` keep (see `perf_investigation_part2.md:9`).

## 12. Code reference map (fast lookup)

| Area | File `:` line | Symbol |
|------|---------------|--------|
| mv9 tEnd gate fix | `Rendering/Metal/Shaders/MetalShaders.metal:5634` | `while(i<steps) if(i>0 && currentT>=p.tEnd-1e-6)` |
| batchCap shade-aware | `Rendering/Metal/Shaders/MetalShaders.metal:5568` | `batchCap = (fc_fragBatch>0)?... : ((fc_shading\|\|fc_gradientOpacity)?min(4,...):min(16,...))` |
| maxSteps calc | `Rendering/Metal/Shaders/MetalShaders.metal:4363` `8256` `8381` | `max(1,ceil((p.tEnd-firstT)/stepSize))` |
| mv0 baseline loop | `Rendering/Metal/Shaders/MetalShaders.metal:6416` | `for(i<maxSteps)` `latchExit` |
| mv9 ladder | `Rendering/Metal/Shaders/MetalShaders.metal:5568` `5627` `5862` `6030` `6106` | `batchCap` `MV9_FETCH/COMPOSITE/ADVANCE` |
| gradient 6-fetch | `Rendering/Metal/Shaders/MetalShaders.metal:3861` `5259` | `computeGradientFast` `MV9_COMPOSITE` shading |
| mapper variant | `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm:242` `8182` | `fc_marchVariant=9` `fc_fragBatch` |
| batch width | `Rendering/Metal/Shaders/MetalShaders.metal:5568` `3027` `vtkMetalGPUVolumeRayCastMapper.mm:9632` | `maxBatchWidth`/`VolumeMapperUniforms` |
| transpose policy | `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm:477` `2927` | `VolumeTransposedAxisDepth` `VolumeTransposedActive` |
| parity harness | `Rendering/Metal/Testing/Cxx/TestMetalVolumeRayCast.cxx:1` | `vtkMetalGLVisualComparison --scene` |
| NIFTI scene | `Rendering/Metal/Testing/Cxx/TestMetalScenes.h:108` `1480` | `BuildNIFTIVolumeScene` |
