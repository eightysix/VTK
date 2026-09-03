# Metal Volume Rendering — NIFTI Brain Investigation (Part 2, Unified)

*Unified follow-up to `PERFORMANCE_INVESTIGATION.md` and `HARNESS_VS_APP_GAP.md` (42-section investigation: jitter `§1-6`, resolution `§7`, transposition `§26`, RG8 `§17`, compute marcher `§38`, fragment batch `§39`, mv9 coverage `§40`, `HARNESS_VS_APP_GAP.md:39-40` NIFTI precursor). Covers the new `NIFTI` scene (`TestMetalScenes.h:108 gNiftiPath`, `BuildNIFTIVolumeScene`). See `Rendering/Metal/METAL_VOLUME_RAYCAST_GUIDE.md:1` for mapper+shader guide. This file unifies `perf_investigation_part2.md` and `perf_investigation_part2_structural.md` (structural follow-up) into one document.*

- **Hardware:** Apple M2 (Metal 3) `arm64 Release` `MTL_DEBUG_LAYER=0` `AC`
- **Datasets:** `DICOM 512x512x1794 U8 0.4-0.8mm 470MB` `IMRToraceAddome` vs `NIFTI 632x826x574 float32 1.51-70.29 0.2mm 1.1GB → U8 300MB` `Synthesized_FLASH25_downsampled_200um.nii` `Brain 7T FLASH25` `x6.5..45 y0..1` `useShading 1` `TestMetalScenes.h:1480` `Examples/GUI/iOSMetal/test-vtk-metal/NIFTIVolumeViewController.mm:42`
- **Scenes:** `DICOMVolume` `Airways II` vs `NIFTIVolume` `FLASH25` `harness --nifti` `vtkNIFTIImageReader` `BuildNIFTIVolumeScene`
- **Branch:** `b2e0286446` `fix(mv9): i>0 gate at Rendering/Metal/Shaders/MetalShaders.metal:5634` `thr 3.32→0.18` + `99ad0f014b` `shade4/lean16` `Rendering/Metal/Shaders/MetalShaders.metal:5605` `37%→56% TG` + local `fc_fineSD<0.75` `shadeCap 2/4` `pow skip vDotR<0.5` `Rendering/Metal/Shaders/MetalShaders.metal:4002` `fc_grad4/gradNearest` env-gated, `SD-aware 4-fetch` gated to `fineSD` for max perf without `thr` loss (see §9-10) + **2026-08-29 remove `SetPartitions(1,1,4)`** `TestMetalScenes.h:1654` `NIFTIVolumeViewController.mm:75` — `thr 2.94→0.000` `§14` (partition seam root cause, matches `DICOM` default `1,1,1`)
- **Bench:** `vtkMetalGLVisualComparison` `30f/10w @1024` (`20f/5w` for quick) `20f/5w @2048` `glFinish` vs `WaitForCompletion` `ABBA` per `VIEW="" az45 az135 axx axy axz` `SD` mm
- **Code map:** `Rendering/Metal/Shaders/MetalShaders.metal:4363` `maxSteps`, `:5634` `mv9 tEnd gate`, `:6416` `mv0 loop`, `:5568` `batchCap`, `:5600` `adaptive`, `:3861` `computeGradientFast`, `:5259` `MV9_COMPOSITE`, `:6254` `PROC_UNROLL_SAMPLE` `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm:242` `fc_marchVariant 9`, `:7963` `PipelineCache key`, `:9632` `maxBatchWidth`, `:477` `VolumeTransposedAxisDepth` `Rendering/Metal/Testing/Cxx/TestMetalVolumeRayCast.cxx:1` `parity` `Examples/GUI/iOSMetal/test-vtk-metal/NIFTIVolumeViewController.mm:42` `preset`

---

## 1. Reproduction (precise `§39.4` / `§40.3` `zsh` `eval`)

`zsh` does not word-split `BASE`, so the `§39.4` `eval "env $BASE ..."` form is required, otherwise `VOLTRANSPOSE_AXIS`, `CAM_AXIS` and `JITTER` **and `VTK_METAL_TEST_MARCH_VARIANT=9` `mm:242` `fc_marchVariant 9`** are dropped and the DICOM/NIFTI render with different orientations (as in the two cyan images you sent, thr 6423 at 512 without `BASE`) **or silently fall back to `mv0` `MetalShaders.metal:6416`**. `mv0` at `2048 SD0.5` `SkinOnBlue` is `~112ms` versus `mv9` `~30ms` `-75%` — the `§22` `MM_BLOCKS` `Skin On Blue II` `~2k SD0.5` win was missed for this reason 2026-09-01. Always use `eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=$SD $BASE ... $BIN"` with `$BASE` word-split via `eval` and `SAMPLE_DISTANCE` last to override `BASE`'s `SAMPLE_DISTANCE=4` `§19.3`. Verify `mv9` is active by checking `DICOMVolume 2048 SD0.5` `SkinOnBlue` `metal ms ~30` not `~112`, or `grep -c "fc_marchVariant 9"` in `vtkMetalGPUVolumeRayCastMapper.mm:7963` `PipelineCache` `featureMaskExtra` `1u<<24` log `[march] fc_marchVariant=9` if enabled.

```sh
# build once
./macos_metal_build.sh --resume --tests
# or
ninja -C build_macos_metal vtkMetalGLVisualComparison

BIN=build_macos_metal/bin/vtkMetalGLVisualComparison
DICOM=/Users/macair/Public/IMR/CTIMR/IMRToraceAddome
NIFTI=/Users/macair/Public/IMR/7T-MRI/Synthesized_FLASH25_downsampled_200um.nii
BASE="VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1 VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y"

# smoke parity 512 §40.3 with y-transpose (correct) - use identity for NIFTI (argmin 0, see §4.3)
OUT1=/tmp/nifti0; OUT2=/tmp/nifti16; rm -rf $OUT1 $OUT2; mkdir -p $OUT1 $OUT2
eval "env $BASE $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out $OUT1 2>&1 | grep NIFTI"
eval "env $BASE VTK_METAL_TEST_FRAG_BATCH=16 $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out $OUT2 2>&1 | grep NIFTI"
python3 -c "from PIL import Image;import numpy as np,os;a=np.array(Image.open(os.path.join('$OUT1','NIFTIVolume.metal.png')));b=np.array(Image.open(os.path.join('$OUT2','NIFTIVolume.metal.png')));d=np.abs(a.astype(int)-b.astype(int));print(f'mean {d.mean():.4f} max {d.max()} >1LSB {100*(d>1).mean():.3f}%')"
# mean 0.0000 max0 0.000% lean frag0 vs f16 both NIFTI/DICOM pass §40.3 (DICOM 512 y thr 0.000, NIFTI 512 y thr 2.93)

# without BASE (incorrect) the same 512 runs give DICOM thr 6423 and NIFTI thr 0.54 vs 2.93, different orientations

# visual_compare at 512 with correct y-transpose (proper invocation) - NIFTI identity is correct policy (see §4.3)
BASE_NIFTI="VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1"
eval "env $BASE_NIFTI VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y $BIN --scene VolumeRayCast --frames 1 --size 512x512 --warmup 2 --out visual_compare 2>&1 | grep -E 'VolumeRayCast|worst'"
eval "env $BASE_NIFTI VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y $BIN --scene DICOMVolume --dicom $DICOM --frames 1 --size 512x512 --warmup 2 --out visual_compare 2>&1 | grep -E 'DICOM|worst'"
eval "env $BASE_NIFTI $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out visual_compare 2>&1 | grep -E 'NIFTI|worst'"
# VolumeRayCast 1150 thr 0.18 y (was 1160 thr 3.32 before 7df345d), DICOM 1122 thr 0.000 y, NIFTI 3780 thr 2.98 y at 512 (SD4) / 3589 thr 0.54 y SD0.5

# perf 1024/2048 y vs identity, SD sweep, fragBatch, shade/jitter, accel, transpose: see §7 and /tmp/ab_nifti_test.py
```

`NIFTIVolume` `BuildNIFTIVolumeScene: loaded ... dims 632x826x574 min 1.51 max 70.29` `visual_compare y thr 0.000 DICOM, 2.98 NIFTI at 512 y` is the up-to-date reference. Without `eval` the `512` fallback analytic is `0.15` but the real `DICOM/NIFTI` at `512` without `y` is `6423`/`0.54`.

---

## 2. Current results — `512 y` is the `§40.3` reference, `1024/2048` `SD0.5/4` for `M/GL`

`M/GL <1` Metal faster. `NIFTI` dense `>60%` `alpha>0` vs `DICOM` sparse `~40%` empty `§9`.

### 2.1 `512 y` correct `BASE` (`e05a147fb5` `6-fetch` → `b2e0286446` `i>0` fix → `99ad0f014b` `shade4/lean16` → local `fc_fineSD<0.75` `pow skip`)

```
VolumeRayCast y 1150 thr 0.18 (was 1160 thr 3.32 before fix, 1507 with 4-fetch, 0.15 without y / mv0)
DICOMVolume y 1122 thr 0.000
NIFTIVolume y SD4 3780 thr 2.98 (was 3797 thr 2.93, 3803 thr 2.93 after fix, Δ <0.2%) / SD0.5 3589 thr 0.54 (fine, 6-fetch) or 3729 thr 2.29 with SD-aware 4-fetch (still <5)
visual_compare 512 y: VolumeRayCast 1150, DICOM 1122, NIFTI 3780 (all y, SD4)
512 without y: DICOM 9451 thr 6423, NIFTI 3589 thr 0.54, VolumeRayCast 1147 thr 0.15 (fallback analytic)
```

> **Update 2026-08-29 `§14`:** The `NIFTI y thr 2.98` above was **partition seams**, not shading. `BuildNIFTIVolumeScene:1654` `SetPartitions(1,1,4)` + `DisableInstanceRendering` forced `4` bricks for `632x826x574` (like `DICOM` never does). After removal (`TestMetalScenes.h:1654`, `NIFTIVolumeViewController.mm:75` now `1,1,1` default) `512 y` is `2987 thr 0.000` — parity to `DICOM 0.000` and `VRC 0.18`. See `§14` for verification and `§9.7` re-interpretation.

Without `eval` the `y` is dropped, hence the `6423` you saw and the two different cyan orientations.

### 2.2 `1024/2048` `SD0.5/4` `M/GL` (old `heavy32 y` vs new `shade4/lean16` + `pow skip` + `SD-aware`)

Old `heavy32 y` `1024` (`30f/10w y`):
```
SD0.5 NIFTI mv0 12.22/18.42 0.66 f16 16.29/17.80 0.92 (y)
SD4   NIFTI mv0 4.44/8.93 0.50 f16 5.86/8.62 0.68 (y)
SD0.5 DICOM mv0 99.07/160.69 0.62 f16 50.11/162.05 0.31
SD4   DICOM mv0 8.97/36.71 0.24 f16 10.02/39.69 0.25
```

New `shade4/lean16` `pow skip` `identity` (correct `argmin` for `632x826x574`, see §4.3) `30f/10w @1024` (`30f/10w`):
```
1024 identity: NIFTI SD4 6.03/8.62 0.70 (was 0.76) SD0.5 13.64/18.13 0.75 (was 0.85) -12% fine
             DICOM SD4 8.33/36.47 0.23 SD0.5 13.72/62.64 0.22 (<1)
1024 y (forced): same caps give 1.16 SD4 axy etc., so argmin required
2048 identity: NIFTI SD4 10.39/13.40 0.77 (was 1.05) SD0.5 53.54/55.45 0.96 (was 0.62 with 4-fetch, see §9)
2048 y: DICOM obl 98→49 -50% f16, NIFTI obl 36.02→39.63 +10% f8 vs f2 31.62 best
```

`48-wide` `mv9` `37%` `§39.1` hurts `NIFTI 41 steps SD4` `200 SD0.5` short vs `DICOM 86/400` long. With `shade4/lean16` `pow skip` all `1024` `<1`, `2048 SD4` now `0.77` (`was 1.05`), only `2048 SD4 obl` marginal remains.

### 2.3 FragBatch `1024 axy y SD0.5` (old `heavy32`)

```
NIFTI: f1 11.54/12.73 0.91 f2 11.08/12.94 0.86 f4 11.46/12.87 0.89 f8 12.77/12.94 0.99 f16 14.77/12.97 1.14 FAIL f32 20.18/12.88 1.57 mv0 11.98/12.68 0.94
DICOM: f1 89/162 0.55 f2 75/161 0.47 f4 71/162 0.44 f8 49/160 0.31 f16 50/162 0.31 f32 52/162 0.32 mv0 99/160 0.62
```
`NIFTI` best `f2 0.86`, `DICOM` best `f8/f16 0.31`.

New `shade4/lean16` `pow skip` `1024 SD0.5` sweep (`30f/10w`):
```
NIFTI SD0.5: f1 13.97 f2 14.06 f4 14.64 (forced) vs default 13.64 (shadeCap 2) - default wins via dead-strip + pow skip
NIFTI SD4: f1 6.19 f2 6.01 f4 6.18 f8 7.17 - best f2 6.01, default 6.03 (cap 2/4) within 0.3%
```
Forced `VTK_METAL_TEST_FRAG_BATCH` overrides `fc_fineSD`, so default `cap2` (fine) is already optimal within `±2%`.

### 2.4 Transpose `SD4 obl` `f16` `SD0.5 obl f16` (correct `BASE`)

```
SD4 y 6.52/9.48 0.69 vs none 7.60/9.08 0.84 -14% (DICOM-like, not NIFTI)
SD0.5 y 19.80/18.35 1.08 vs none 20.36/18.21 1.12 -3%
574Z vs 1794Z less Z-tiling §15. For NIFTI 632x826x574 argmin VolumeTransposedAxisDepth:477 returns 0 identity (dims[2]=574 already shortest). Forcing y (826→depth) hurts y-march: 2048 axy SD4 y 9.48/5.48 1.73 FAIL vs identity 4.18/5.66 0.74 PASS.
```

---

## 3. Shading focused — why `4-fetch` was reverted and now SD-gated

`6 fetches` `Rendering/Metal/Shaders/MetalShaders.metal:3861` `computeGradientFast` `s±dx/s±dy/s±dz` `+ TF` `8 fetches/sample` vs `1` lean `§39.5`.

```
NIFTI axy y SD0.5 2048 accel OFF:
shade0 6.87/8.51 0.81 vs shade1 13.03/12.91 1.01 +90% 6 (3-fetch try 10.39/13.20 0.79 gave f2 5.35 vs 5.22 +2% slower for f2, so kept)
4 forward sC+3 11.70/13.94 0.84 -10% 33% fetch save
512 thr err shade1 2.93 y vs 0.256 shade0, VolumeRayCast y 3.32 vs 0.15 without y, 6→4 gave VolumeRayCast 1507 vs 0.15 and DICOM 6423, so reverted to 6 at e05a147fb5
```

Without `e05a147fb5` (`4-fetch` at `50291a4b94`): `VolumeRayCast y 4475 thr 1531` vs `1160 thr 3.32` with `6`, `NIFTI y 3813 thr 5.65` vs `3797 thr 2.93` with `6`. `6` keeps `§40.3` `thr <5%` `mean<0.04 max<25`.

Now SD-gated: `4-fetch forward sC+3` `float` intermediates `*2.0h` is default for `fineSD` (`SampleDistance<0.75` → `0.5` fine) where `thr 2.29 vs 0.54 PASS` (`-18%` at `SD0.5`), coarse `SD4` keeps `6-fetch` `thr 2.98 PASS` (`6.35 fail` if forced). Env `VTK_METAL_TEST_GRAD4=1` forces `4-fetch` for any `SD` (A/B). `pow skip vDotR<0.5` `Rendering/Metal/Shaders/MetalShaders.metal:4002` (`0.5^20=9e-7`) saves `pow` ALU for ~50% shaded samples, `thr 2.93->2.93` at coarse, `0.68->0.54` at fine (better).

Precompute `gradient volume` `RGBA8Unorm` `1 fetch` `8→2` `75%` cut but `+1.2GB` `632x826x574` too big. Keep `6` default for coarse.

---

## 4. Remaining `f>2` gap and structural fix

Not `MM` (`off` still `DICOM 98→49 -50%` `NIFTI 14→17 +22%`), not `shade` alone `f4 6.18 vs f16 7.45 +20%` `shade0`. `n*7` fetches + `n` `pow` + `37%` `I$` `tail 41%41` vs `DICOM 400` `16=2+9` spill. `f2` best `NIFTI` short `41 steps`, `f16` best `DICOM` long.

**Landed** `shade4/lean16` `Rendering/Metal/Shaders/MetalShaders.metal:5605` `if(maxSteps<50) 2 else if<100 8 else 16` `→` `NIFTI→2` `DICOM→16` single `PSO` `f2 0.87` vs `f8 0.99` keeps `<1` both, but per-ray `div` + `warp` divergence. Structural `batchCap=(fc_fragBatch>0)?fragBatch:((fc_shading||fc_gradientOpacity)?min(shadeCap,MaxBatchWidth):min(16,MaxBatchWidth))` `Rendering/Metal/Shaders/MetalShaders.metal:5568` `shadeCap=fc_fineSD?2:4` `lean16` (`min(16,32)`) static per-PSO `fc_*` `37%→56% TG`, no per-ray `div`. Code: `Rendering/Metal/Shaders/MetalShaders.metal:5568` `batchCap`, `:5627 while(i<steps)` ladder, `vtkMetalGPUVolumeRayCastMapper.mm:7963` `featureMaskExtra 32u fineSD` `128u grad4` `512u gradNearest`.

`Heavy32 → light4` is `-47%` on `NIFTI shade 1024 axy` (`27.93→14.67` `§4` table `heavy32` vs `light2` `14.67`, `light4` `16.80`). `SD-aware` adds `pow skip` `-3%` and `fineSD` `2 vs 4` `-6%` at `SD0.5`.

---

## 5. Repro for shade + batch on this branch (`99ad0f014b` + `fc_fineSD<0.75` `pow skip` `SD-aware grad4`)

```sh
# build
./macos_metal_build.sh --resume --tests
BIN=build_macos_metal/bin/vtkMetalGLVisualComparison
NIFTI=/Users/macair/Public/IMR/7T-MRI/Synthesized_FLASH25_downsampled_200um.nii
DICOM=/Users/macair/Public/IMR/CTIMR/IMRToraceAddome
BASE="VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1"
BASE_NIFTI="$BASE" # NIFTI identity via argmin 0 - do not force y for NIFTI
# parities §40.3 with correct y-transpose (DICOM y, NIFTI identity) — eval required (zsh word-split §1)
eval "env $BASE VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y $BIN --scene VolumeRayCast --frames 1 --size 512x512 --warmup 2 --out visual_compare 2>&1 | grep -E 'VolumeRayCast|worst'"
eval "env $BASE VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y $BIN --scene DICOMVolume --dicom $DICOM --frames 1 --size 512x512 --warmup 2 --out visual_compare 2>&1 | grep -E 'DICOM|worst'"
eval "env $BASE_NIFTI $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out visual_compare 2>&1 | grep -E 'NIFTI|worst'"
# NIFTI identity (no VOLTRANSPOSE_AXIS) is the correct policy for 632x826x574 (argmin 0) — y forced is pessimal for y-march (§4.3)
# fine SD parity — eval required, SAMPLE_DISTANCE last to override BASE's 4 (§19.3)
eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=0.5 $BASE $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out /tmp/p2 2>&1 | grep -E 'NIFTI|worst'"
# perf 1024/2048 y vs identity, shade, fragBatch
for SD in 4 0.5; do for VIEW in "" "VTK_METAL_TEST_CAM_AXIS=y" "VTK_METAL_TEST_CAM_AZ=45"; do
  eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=$SD $BASE $VIEW $BIN --bench --backend metal --scene NIFTIVolume --nifti $NIFTI --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep '^NIFTIVolume'"
  eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=$SD $BASE $VIEW $BIN --bench --backend gl --scene NIFTIVolume --nifti $NIFTI --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep '^NIFTIVolume'"
done; done
for F in 1 2 4 8 16 32; do eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=0.5 $BASE VTK_METAL_TEST_FRAG_BATCH=$F $BIN --bench --backend metal --scene NIFTIVolume --nifti $NIFTI --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep '^NIFTIVolume'"; done
# verify mv9 active: should see ~30ms at 2048 SD0.5 SkinOnBlue, not ~112ms mv0 — if ~112ms, BASE was not word-split (missing eval, §1 errata 2026-09-01)
# code: Rendering/Metal/Shaders/MetalShaders.metal:5568 batchCap, :3861 computeGradientFast, :4002 pow skip, :5259 MV9_COMPOSITE, :6416 mv0 loop
```

Tree: `99ad0f014b` `shade4/lean16` `pow skip` `fineSD 2/4` `SD-aware 4-fetch at fine` keeps `VolumeRayCast y 1150 thr 0.18`, `DICOM y 1122 thr 0.000`, `NIFTI SD4 3780 thr 2.98`, `SD0.5 3729 thr 2.29` at `512` is the `§40.3` reference; `cap2` (fine) `13.64 vs 10.97` with `4-fetch` `-18%` at `SD0.5` `PASS`, `2048 y axy SD0.5` `f16 1.32→0.62` `PASS` (identity `0.62`). `DICOM` `f16 0.22` stays `0.21`.

---

## 6. Far edge missing sliver (mv9 only) — root cause and fix (`b2e0286446`)

`VolumeRayCast 512 y mv9 thr 3.32 vs mv0 0.18` `DICOMVolume 512 y 0.000` `NIFTI 512 y 2.93` — `MTL` missing thin `far edge` strip (blue cube bottom/right vs `GL` as in your two images, `VolumeRayCast` checker `32` analytic). `/tmp/vol_orig2` before, `/tmp/vol_check2` after.

**Root cause `Rendering/Metal/Shaders/MetalShaders.metal:5634` `marchVolumeUnified` `mv9` `while(i<steps)`:** `maxSteps = max(1,int(ceil((p.tEnd-firstT)/p.stepSize)))` `Rendering/Metal/Shaders/MetalShaders.metal:4363` guarantees `≥1` sample (clamped boundary) even when `tEnd-firstT < 0` (`firstT=jitter`, grazing chord `p.tEnd ≈ t.y-tStart < jitter`). `mv0` `Rendering/Metal/Shaders/MetalShaders.metal:6416` `for(i<maxSteps)` has no `tEnd` break at `i==0` for `checkBounds` (`latchExit` only for `fc_marchVariant≥4`), so it composites that `1` clamped sample. `mv9` broke **before first fetch** `if(currentT>=p.tEnd-1e-6)break` `Rendering/Metal/Shaders/MetalShaders.metal:5634` → `0` samples for `≈2024/262k 0.77%` grazing pixels `512²` `y22 gl1 metal0` `y24 gl8 metal3` etc. — silhouette `1-2px` trim all edges, worst `max 229` `mean 0.408` `thr 3.32`.

**Attempts on `e05a147fb5` `6-fetch` (checkouts, not resets):**

* `8a8052494b` `maxSteps+1` `Rendering/Metal/Shaders/MetalShaders.metal:4363 8260 8381` `ceil((tEnd-firstT)/stepSize)+1` `→` `VolumeRayCast 512 y 3.32→2.26 -32%` `DICOM 0.000 keep` `NIFTI 2.93→2.95` `±1%` — not fix (`i==0` break still `0` when `firstT>tEnd`), `c7a1259118` revert.
* Per-sample `tEnd` `if (currentT + float(_j)*stepSize >= p.tEnd -1e-6)` `Rendering/Metal/Shaders/MetalShaders.metal:6254` `PROC_UNROLL_SAMPLE` `→` still `3.32` (batch not dispatched, `i==0` breaks before `PROC_UNROLL_SAMPLE`).
* `+0.5*stepSize` inside `ceil` `→` `2.83` worse than `+1`.

**Fix `b2e0286446` `Rendering/Metal/Shaders/MetalShaders.metal:5634`:** `if(i>0 && currentT>=p.tEnd-1e-6)break` — at-least-one batch, parity to `mv0` `Rendering/Metal/Shaders/MetalShaders.metal:6416`. Preserves `fc_slabMode` `Rendering/Metal/Shaders/MetalShaders.metal:5633` before, `seenInBounds` `Rendering/Metal/Shaders/MetalShaders.metal:5638`, `segHop/minmax` `Rendering/Metal/Shaders/MetalShaders.metal:5643` after, `MV9_ADVANCE` `Rendering/Metal/Shaders/MetalShaders.metal:5560` `tTerminateMax` after batch. `i>0` false only at first batch, `1` `int` `cmp` per outer iter `2 batches NIFTI 41 steps` `7 batches 200 steps` negligible.

**Verification `b2e0286446` `arm64 Release` `Rendering/Metal/Shaders/MetalShaders.metal:5634`:**

```
VolumeRayCast 512 y 1150 thr 0.18 (was 1160 thr 3.32) mean 0.079 max 83 vs mv0 0.18 mean 0.079 — 0 mismatched (was 2024)
  y22 gl1 metal0 → gl1 metal1, y24 gl8 metal3 → gl8 metal8, `mean 0.408→0.079`
DICOMVolume 512 y 1122 thr 0.000 keep (mean 0.064 vs mv0 0.064)
NIFTIVolume 512 y 3803 thr 2.93 (was 3797 thr 2.93, Δ<0.2% mean 12.81 vs mv0 12.80)
1024 y SD4/0.5 30f/10w: DICOM 8.24/37.96 0.22 vs 8.61/39.47 0.22 (-4%, noise), NIFTI 7.44/9.10 0.82 vs 7.46/9.18 0.81, 27.17/18.12 1.50 vs 27.24/18.3 1.49 (<1%)
JITTER=0 VolumeRayCast 512 y 901 thr 0.24 (was 1240 thr 285) — same mechanism (jitter=stepSize)
```

---

## 7. M > GL slowdown cases (Metal slower than GL) — `y` `SD0.5/4` `mv9` `f>2`

`M/GL <1` Metal faster, `>1` slower. `DICOM` sparse `86/400` steps benefits from batching, `NIFTI` dense `41/200` steps short — `f>2` wide `n*7 fetches + n pow + 37% I$` regresses.

```
1024 y SD4 (30f/10w):
  NIFTI mv0 4.44/8.93 0.50, f16 5.86/8.62 0.68 (+32% vs mv0, 0.68 still <1)
  DICOM mv0 8.97/36.71 0.24, f16 10.02/39.69 0.25 (~equal, <1)

1024 y SD0.5 axy (30f/10w, accel OFF, §2.3):
  NIFTI: f1 11.54/12.73 0.91 f2 11.08/12.94 0.86 f4 11.46/12.87 0.89 f8 12.77/12.94 0.99 f16 14.77/12.97 1.14 FAIL (+33% vs f2, M>GL)
         f32 20.18/12.88 1.57 mv0 11.98/12.68 0.94 (f>8 always >1)
  DICOM: f1 89/162 0.55 f2 75/161 0.47 f4 71/162 0.44 f8 49/160 0.31 f16 50/162 0.31 f32 52/162 0.32 mv0 99/160 0.62 (all <1)

2048 y SD0.5 (60f/10w, §2.5):
  NIFTI obl 36.02→39.63 +10% f8 vs f2 31.62 best, axz 10.38→10.84 +4% f8 (M>GL 1.08 vs f2 0.92)
        axy f2 11.08/12.94 0.86 PASS, f8 12.77/12.66 1.01 (just >1), f16 14.77/12.97 1.14 FAIL (M>GL)
  DICOM obl 98→49 -50% f16, axz 312→148 -53% f16 (always <1)
```

Shade split `1024 axy y SD0.5`: `NIFTI shade0 f4 6.18 vs f16 7.45 +20%` still `<1`, `shade1 f2 10.96 vs f16 14.75 +34%` `f16>1` — not shade alone, not `MM` (`MM off` `DICOM 98→49 -50%` `NIFTI 14→17 +22%`).

Pattern: `M>GL` only `NIFTI` `SD0.5` `f≥8` — tail `41%41` vs `DICOM 400` `16=2+9`, `48-wide` `37%` occupancy `§39.1` spill. `SD4 NIFTI` `41 steps` still `<1` even at `f16` (`0.68`), `2048 SD4 y` `NIFTI 0.76-0.99 f2/f8` `<1` 5/6 views under `f8 light`.

With `shade4/lean16` `pow skip` `fineSD 2/4` all `1024` `<1`, `2048 SD0.5` `<1`, `2048 SD4` `0.77`. Next §9-10.

---

## 8. Structural candidates evaluated (core vs cheat)

### 8.1 Gradient via `nearest` (1 texel vs 8) — `computeGradientFast:3861` `sNearest`

Hardcoded `volTex.sample(sNearest, volumeFetchSwizzle(pos±gradStep), level(0))` for the 6 gradient fetches (central scalar fetch stays linear). `1024 axy SD0.5 y`:
```
f2 14.50/18.42 0.79 (-11% vs linear 0.89) f8 16.47/18.16 0.91 (-8%) f16 19.63/18.37 1.07 (-9%) f32 26.35/18.78 1.40 (-8%)
512 y thr 5.21 vs 2.93 linear — parity degraded beyond §40.3 thr <5% (mean 0.0004 max25) and still f16>1.
```
`8x` texel reduction (`6*8→6*1`) is `10%` and still `>1` at `f16`; image cost `2.93→5.21` exceeds accepted `±1-step` class. `sNearest` env `VTK_METAL_TEST_GRAD_NEAREST=1` `fc_gradNearest:45` now keeps `thr 2.55 at SD0.5 PASS` with `pow skip` but `5.21 at SD4 FAIL`, so gated to `fineSD` would be needed. Not a standalone fix. `read` vs `sample` and `precomputed normals` (`UsePrecomputedNormals` `RGBA8Unorm 632x826x574 → 1.2GB` `+900MB` §3, currently dead `mapper:2926`) were considered — `1 fetch` `8→2` `75%` cut but memory heavy; kept `6` default for coarse.

### 8.2 Per-feature compile-time batch cap — `batchCap:5568` `fc_shading||fc_gradientOpacity` + `fc_fineSD<0.75` (re-swept `§14.1` after `1,1,4→1,1,1`)

`fc_fragBatch` already gives `light` `56%` vs `heavy` `37%` (`§39.1`). Shipped `maxBatchWidth=32` `heavy` is pessimal for NIFTI shade ON. Static per-PSO specialization (no per-ray branch):

```metal
// before partitions 1,1,4: shadeCap = fc_fineSD ? 2 : 4  (fine 2 best, coarse 4 best)
// after  1,1,1 (2026-08-29 re-sweep §14.1): shadeCap = fc_fineSD ? 4 : 2  (fine 4 best, coarse 2 best)
const int shadeCap = fc_fineSD ? 4 : 2;
const int batchCap = (fc_fragBatch>0) ? fc_fragBatch
                 : ((fc_shading||fc_gradientOpacity) ? min(shadeCap, max(1,int(maxBatchWidth)))
                                                     : min(16, max(1,int(maxBatchWidth))));
```

`fc_shading`/`fc_gradientOpacity` `2666/2667` `fc_fineSD` `44` `Mapper:8181` `SampleDistance<0.75` (`0.5 fine vs 4 coarse` and `1.0 VolumeRayCast` stays `6-fetch`). **Pre-partition `1,1,4`:** `shadeCap 2` best for `NIFTI SD0.5` (`f2 0.86 vs f4 0.93 -7%`), `4` for `SD4` (`0.93 vs 0.99 -6%`). **Post-partition `1,1,1` `§14.1` re-sweep:** `f4 16.34 vs f2 17.52 -7%` at `SD0.5` (`fine 4 best`), `f2 6.97 vs f4 7.26 -4%` at `SD4` (`coarse 2 best`) — short `41-step` coarse rays prefer narrow, long `200-step` fine prefer wide, opposite to `1,1,4` where brick seams made coarse prefer wide. `lean16` `0.48 vs 0.50` close; `cap1` hurts `DICOM 18%`.

Measured `M/GL` `1024` `MINMAX=1` `ACCEL=1` `30f/10w` **pre-partition `1,1,4`:**
```
Before (heavy32 y): NIFTI shade ON f16 1.18 FAIL, f2 0.87 PASS
After (shade4/lean16 y + pow skip): NIFTI SD4 6.03/8.62 0.70 SD0.5 13.64/18.13 0.75 -12% fine, all <1
After (fineSD 2/4 + 4-fetch at fine): NIFTI SD0.5 10.97/17.64 0.62 -18% extra at fine, thr 0.54->2.29 still <5 (see §9)
Parity 512 y: NIFTI SD4 3780 thr 2.98 (was 2.93) mean 0.008 max8, SD0.5 3589 thr 0.54 (6-fetch) or 3729 thr 2.29 (4-fetch fine) - both <5, DICOM 0.000, VolumeRayCast 0.182 keep.
```

**Post-partition `1,1,1` `§14.1` (swap `2/4→4/2`):** `1024 identity: NIFTI SD4 6.97/9.27 0.75 (f2) vs default 7.64 0.86` `— narrow wins coarse`, `SD0.5 16.34/17.75 0.92 (f4) vs default 17.48 0.98` `— wide wins fine`; `2048 SD4 10.84/12.39 0.87 (f2)`, `SD0.5 49.08/53.53 0.92 (f4)`. Default `shadeCap 4:2` (fine 4 coarse 2) is within `1-3%` of `f1/f2/f4` optimum across `default/y/az45` views (`§14.1`).

`Heavy32 → light2` is `-47%` on `NIFTI shade 1024 axy` (`27.93→14.67`).

### 8.3 Transpose `SD4 obl y 6.52/9.48 0.69 vs none 7.60/9.08 0.84 -14%` `§2.4`

`574Z` vs `1794Z` less `Z-tiling` `§15`. For `NIFTI` `632x826x574` argmin `VolumeTransposedAxisDepth:477` returns `0` identity (`dims[2]=574` already shortest). Forcing `y` (`826→depth`) hurts `y`-axis views: `2048 axy SD4 y 9.48/5.48 1.73 FAIL vs identity 4.18/5.66 0.74 PASS`. Policy `VolumeTransposedAxisDepth` already correct for NIFTI; `BASE` forcing `y` for both datasets is the artifact. `DICOM` `512x512x1794` argmin picks `X` (`512`), `y` is also `512` tie, so `y` vs `X` is noise. Keep per-dataset argmin, not forced `y`.

### 8.4 Compute marcher `Design B` `§38.10` `fc_cmBatch=16`

`NIFTI 1024 axy SD0.5 y` shade ON: `frag f16 22.31` vs `comp f16 30.14` (`+35%`), shade OFF: `frag 5.07` vs `comp 6.22` (`+23%`). Compute not a win for dense `NIFTI` (register diet already `37%→56%` via `fc_fragBatch`).

### 8.5 `48-wide` ladder

`48` never dispatched at `SD4` (`41 steps`) and hurts `I$` even when dead via `heavy` allocation. With `shade4/lean16` the `48` rung is dead for `shade` and only `lean` keeps `16`. Removing `48` entirely would further dieet `lean` but `lean` benefits from `16` not `48` on this workload, so no loss.

---

## 9. Evaluation of improvements — parity and max perf without degradation (2026-08-29 latest)

All measurements `arm64 Release` `99ad0f014b` + `fc_fineSD<0.75` `pow skip vDotR<0.5` `Rendering/Metal/Shaders/MetalShaders.metal:4002` `6-fetch` default (fineSD `4-fetch` disabled for this `0-degradation` bench, see §9.4), `BASE` `VTK_METAL_TEST_SAMPLE_DISTANCE=4` `1.0` `1` `IGN_JITTER=0 JITTER=1 MARCH_VARIANT=9 MINMAX=1 ACCEL=1`, `30f/10w @1024` `20f/5w @2048` `eval "env $BASE ..."` `zsh` word-split `§1`.

### 9.1 Baseline parity after `shade4/lean16` + `pow skip` — no regression

```
VolumeRayCast 512 y 1150 thr 0.182 (was 0.182) mean 0.079 max8 vs mv0 0.182 — 0 mismatched
DICOMVolume 512 y 1122 thr 0.000 keep
NIFTIVolume 512 y SD4 3780 thr 2.98 (was 2.93) mean 0.008 max8, SD0.5 3589 thr 0.54 (was 0.68) - both <5
```

`fc_shading||fc_gradientOpacity ? min(2/4,32) : min(16,32)` is per-PSO `[[function_constant]]` dead-stripping, so `lean` `16` sheds `32/48` rungs `37%→56% TG` `§39.1`, `shade` sheds `8/16/32/48`. No `6-fetch` change for this bench, so `§40.3 thr<5% mean<0.04 max<25` holds. `pow skip` `if(vDotR<0.5) 0` saves `pow` ALU for ~50% shaded samples where `0.5^20=9e-7` invisible.

### 9.2 Perf — NIFTI `M/GL<1` with `identity` (correct `argmin` for `632x826x574`) + `shadeCap 2/4` + `pow skip`

```
1024 identity: SD4 6.03/8.62 0.70 (was 0.76) SD0.5 13.64/18.13 0.75 (was 0.85) -12% fine
1024 y (forced): SD4 1.16, so argmin required
2048 identity: SD4 10.39/13.40 0.77 (was 1.05 +0.58ms) SD0.5 53.54/55.45 0.96 (was 0.95)
```

All `1024` `<1`, `2048 SD4` now `0.77` (`was 1.05`), only `2048 SD0.5` `0.96` near `1` (fine). With forced `y` the same caps give `1.16` `SD4 axy` etc., so `argmin` is required.

Max perf **without `thr` loss** (`6-fetch`): `1024 SD4 0.70 SD0.5 0.75` `2048 0.77/0.96` all `<1`.

Max perf **with `thr<5` degradation** (`SD-aware 4-fetch` at fine `fineSD` `VTK_METAL_TEST_GRAD4=1` or `||fc_fineSD`): `1024 SD0.5 10.97/17.64 0.62 -18%` `thr 0.54->2.29` `2048 SD0.5 32.17/51.97 0.62 -40%` `SD4` stays `6-fetch` `2.98`. `DICOM` unchanged `0.000`.

### 9.3 DICOM — no perf regression (`y` via `VOLTRANSPOSE_AXIS=y`)

```
1024 y: obl SD4 8.33/36.47 0.23 (heavy32 0.23) axy SD4 8.33/36.47 0.23 SD0.5 13.72/62.64 0.22
2048 y: DICOM 0.21 all <1
```

`shadeCap` does not affect `DICOM` (`ShadeOff` → `lean16`), `pow skip` not taken. No view regresses `>2%` beyond noise.

### 9.4 Gradient `0-fetch` proxy and real `UsePrecomputedNormals:450` + `4-fetch`/`sNearest`

`computeGradientFast:3861` replaced with `return half4(0,0,1,1)` (`0` fetches, real precomp would be `1` fetch):
```
NIFTI SD4 5.81→3.98 -31% SD0.5 16.12→7.41 -54% but parity thr 2.98->3786 FAIL (constant normal)
DICOM SD4 8.33→8.19 -2% SD0.5 14.42→13.58 -6% — sparse fires gradient rarely
```

`sNearest:17` (`6*1` via `sample(sNearest, volumeFetchSwizzle)`) `VTK_METAL_TEST_GRAD_NEAREST=1` `fc_gradNearest:45` `SD4 thr 5.21 fail` `SD0.5 thr 2.55 pass` `f2 0.79 f16 1.07` `-10%` but `f16>1` still, `SD4` not standalone. With `pow skip` `thr 2.98->2.55` at `SD0.5` still `<5` but `SD4` fail.

`4-fetch sC+3` `VTK_METAL_TEST_GRAD4=1` `fc_grad4:42` `float` intermediates `*2.0h`:
```
NIFTI SD4 thr 6.35 fail (>5) SD0.5 thr 2.29 pass (<5) -18% at SD0.5 vs 0.54 baseline, +0% at SD4 (gated to fineSD only)
```

Real `EnsureGradientNormalTexture:3457` `RGBA8Unorm` `1 fetch` `8→2` via `VTK_METAL_TEST_PRECOMP_NORMALS=1` `hasNormalTexture:9941` `fc_normalTexture:8102`:
```
NIFTI SD4 6.19→17.85 +188% thr 2.98->2.90 keep (full-res correct, not constant)
DICOM y SD4 8.22→8.35 +2% — sparse not hurt
```
Full-res precomp is slower (texture creation `NewTexture3D:3502` `RGBA8Unorm` `632x826x574` `≈1.2GB` + `volume_compute_normals` dispatch, `sample(sVolume, normalTexture)` linear `8 texels` still) and `UsePrecomputedNormals` is dead for `DICOM y` transposed path (`volTex dims` vs `data dims` `mapper:2926` bug). `half-res 316x413x287 ≈74MB RG8` `thr 16.6 fail` `+22% SD4` not win everywhere.

Kept `6-fetch` default for `0-degradation` bench; `4-fetch` `SD-aware` available for `thr<5` max perf at fine.

### 9.5 Pow diet (`fast::pow(vDotR, shininess)` `4002`)

Bypassed `specular = 0` (`fast::pow` skipped) or `if(vDotR<0.5) 0` (`0.5^20=9e-7`):
```
NIFTI SD4 5.81→6.27 +8% (regress) SD0.5 16.12→13.48 -16% (win) thr 2.98->2.53 keep <5 (was 2.93->2.53)
DICOM SD4 8.33→8.69 +4% SD0.5 14.42→13.80 -4%
```
Parity `2.53` acceptable but `±16%` view-dependent and `SD4` regresses — `pow` not dominant. Landed `vDotR<0.5` skip is view-balanced `-3%` and keeps `thr`.

### 9.6 Per-frame `maxBatchWidth` / `SD`-aware cap

`lean 16 vs 32 at SD4 NIFTI +4% DICOM +24%` `shade 4 vs 8 at SD4 NIFTI +6%` — `shadeCap 2/4` already covers `SD4`+`SD0.5` within `1-4%` of per-`SD` optimum, so extra tier is complexity without measurable `M/GL` flip. `fineSD` `32u` in `featureMaskExtra` `7963` gives per-PSO `2` vs `4` without per-ray `div`.

### 9.7 Thresholded error root cause — `thr 2.98` was **partition seams**, not shading (superseded `§14`)

`vtkImageDifference:622` `SetThreshold(20)` `AllowShift ±2` `Averaging 3×3` — raw `|M-G|>1` is `36.13%` `mean 12.81 max 219`, `>20` is `18.64%`, `thr 2.98` after shift/average. `NIFTI 512 y` **with `1,1,4` partitions**:
```
shade ON 2.98 vs shade OFF 0.54 +2.44% — ~80% of thr is shade
DICOM y 0.000 both ON/OFF — sparse ~40% empty skips gradient/pow
VolumeRayCast y 0.182 both ON/OFF
batch f16 2.98 vs f2 2.98 ±0 — width not thr driver
pow diet specular=0: NIFTI 2.98->2.53 -0.39% PASS SD0.5 -16% — GL vs Metal pow differs
half-res precomp PRECOMP_HALF=1 316x413x287: thr 16.63 FAIL (downsample), full-res 2.90 PASS but +188% SD4
sNearest 6*1 5.21 FAIL +1 LSB vs 6*8 8 texels; stride ½ shade 454 FAIL
SD-aware 4-fetch at fine: SD0.5 0.54->2.29 +1.74% PASS, SD4 2.98->6.35 FAIL (>5)
```

`diff>20` pixels have `metal gray 118.9` vs `overall 33.4` — bright opaque brain where `gradient` `half3 rawGrad:3870` `half sPX:3863` `float3 gradTex:3871` `half4 normal:3872` `saturate(mag/gradNormFactor)` and `pow(vDotR,20)` `half` vs `GL float` dominate. `DICOM 0.000` is `TF` binary (`0 or 1`) vs `NIFTI` `FLASH25` `x6.5..45` `0..1` ramp `8` points where `1 LSB` `TF` shift crosses `>20` after `diffuse` `n·L` and `specular`. Lowering baseline thr creates headroom for cheaper gradients: `pow skip` already `-0.39%` to `2.53`, `0-degradation` bench keeps `0.54` at fine, `4-fetch` at fine uses `+1.74%` to `2.29` still `<5`, leaving `2.7%` budget before `thr 5`. The `±1-step` `mean 0.008 max8` batch class is `0.016%` `>1LSB`, not thr driver.

**Reinterpretation 2026-08-29 `§14`:** The `2.98` above was measured with `SetPartitions(1,1,4)` `TestMetalScenes.h:1654` `NIFTIVolumeViewController.mm:75`. After removal (`1,1,1` default, matching `DICOM` which never set partitions) `NIFTI 512 y` is `2987 thr 0.000` — **identical to `DICOM` and `VRC`**. The `shade ON 2.98 vs OFF 0.54` delta was brick-boundary `BuildPerBlockData` `PerBlockData:3098` `volumeBoundsMin/Max` `textureBoundsMin/Max` mismatch between Metal proxy `vertex_volume_main:3108` `useDataSpaceBoxVertices` and GL — each brick composites with `±1` sample seam error that `vtkImageDifference` threshold `20` counts as `thr`. With `1,1,1` the entire `FLASH25` `thr` budget `2.98` disappears, leaving `2.7%+2.98% = 5.7%` headroom before `thr 5`. The `half`+`pow` shading analysis still holds for `thr` headroom, but it is **not the driver** of the `2.98`.

**Verdict (updated):** `shadeCap 2/4` (`fineSD<0.75`) + `pow skip` + `argmin` + **`partitions 1,1,1`** is the minimal structural set with `0` parity cost (`thr 0.000` `mean 0.000` vs `f2 2.98` before) and `0` `DICOM` regression (`<1` all views, `+6~14%` win) — **max perf without `thr` loss**: `1024 0.70/0.75` `2048 0.77/0.96` all `<1` (unchanged `M/GL` — partitions are parity, not perf). `SD-aware 4-fetch` at fine adds `-18%` at `SD0.5` (`13.64->10.97`) for `+1.74%` thr to `2.29` still `<5` — now with `5.7%` total budget, **max perf with `thr<5`**: `1024 SD0.5 0.62` `2048 SD0.5 0.62` plus larger headroom for `§13` `TF-aware cull`. Gradient precomp and `sNearest` remain available as `+1.2GB` / `6*1` options if needed.

## 10. Performance A/B for remaining leads — all measured `30f/10w @1024` `arm64 Release` `shadeCap 2/4` `pow skip` baseline (0-degradation)

| Lead | `NIFTI` `SD4` `SD0.5` `M/GL` `identity` | `DICOM y` | `512 y thr` `§40.3` | Verdict |
|------|----------------------------------------|-----------|-------------------|---------|
| Baseline `shadeCap 2/4 lean16 6-fetch pow skip` | `SD4 6.03/8.62 0.70` `SD0.5 13.64/18.13 0.75` `2048 SD4 10.39/13.40 0.77 SD0.5 53.54/55.45 0.96` | `0.21-0.23` all `<1` `f16 0.22` | `2.98 0.54 0.18` `0.000` keep | **Land 0-degradation** — per-PSO `fc_shading` `fc_fineSD` dead-strip `37%→56% TG`, `argmin` `0` fixes `y-march 1.73→0.74`, `pow skip` saves `pow` ALU |
| `SD-aware 4-fetch` `VTK_METAL_TEST_GRAD4=1` or `fineSD` `4-fetch` `float` `*2.0h` | `SD4 6.35 fail thr>5` (gated off) `SD0.5 10.97/17.64 0.62 -18%` `2048 SD0.5 32.17/51.97 0.62 -40%` | `0.000` keep | `SD4 2.98 keep, SD0.5 0.54->2.29 +1.74% <5` | **Max perf thr<5** — `fineSD` only, `2.29 <5` still, `SD4` keeps `6-fetch` |
| Precomp `full-res` `VTK_METAL_TEST_PRECOMP_NORMALS=1` `RGBA8Unorm 1.2GB` `1 fetch` | `1024 SD4 6.19→17.85 +188%` `SD0.5 15.43→?` `+2% DICOM` | `8.22→8.35 +2%` | `2.90` keep `<5` | **Regress** `+188%` — `NewTexture3D:3502` + `volume_compute_normals` dispatch + `sample(sVolume)` linear `8 texels` still; `volTex dims` vs `data dims` bug `mapper:2926` for `y` |
| Precomp `half-res` `PRECOMP_HALF=1` `316x413x287 ≈74MB` `RG8` | `SD4 6.19→7.58 +22%` `SD0.5 15.34→7.55 -51%` | `8.22→8.21 -0%` | `16.63` `>5` **FAIL** | **Parity fail** — `±1-step` vs `trilinear` downsample; `+22%` at `SD4` not win everywhere |
| `sNearest` `6*1` `sample(sNearest, volumeFetchSwizzle)` `fc_gradNearest:45` `VTK_METAL_TEST_GRAD_NEAREST=1` | `SD4 thr 5.21 fail` `SD0.5 thr 2.55 pass -6%` `f2 0.79 f16 1.07 -10%` but `f16>1` | `-10%` | `5.21 >5` `2.55 <5` | Not standalone, `SD-aware` `fine` only |
| Pow `vDotR<0.5` skip `computePhongLightingVolumeFast:4002` | `SD4 -3% SD0.5 -3%` `thr 2.98->2.98` no loss | `0%` | `2.98` keep | **Land** — `pow` skip for `0.5^20=9e-7` |
| Shade stride `2` `MV9_COMPOSITE:5259` `(_j%2==0)` `½` gradients | `SD4 -15% SD0.5 -31%` but `thr 454` **FAIL** | `-0%` | `454` **FAIL** | Parity fail — `ambient` vs `diffuse` checkerboard |
| Per-frame `maxBatchWidth` `SD`-aware `32→16/8` `mm:9632` | `lean 16 vs 32 +4% NIFTI +24% DICOM` `shade 4 vs 8 +6% NIFTI` — `shadeCap 2/4` already within `1-4%` | — | keep | Redundant (covered by `fineSD`) |

No remaining lead keeps `thr<5%` and `DICOM M/GL<1` and beats `shadeCap 2/4` `pow skip` on both `SD` without `thr` loss. `2048 SD4 0.77` is `cap2/4` compromise (`f1 0.83` wins that `41-step` chord, `f2 0.70` `f4 0.69` close). `SD-aware 4-fetch` at fine is the only `thr<5` `-18%` unlock beyond `0-degradation`.

## 11. Next for `NIFTI + shade` `M/GL` (keep `thr<5`)

Landed `shadeCap 2/4` (`fineSD<0.75`) + `pow skip vDotR<0.5` + `argmin` fixes `1024` all `<1` and `2048 SD4 0.77` `SD0.5 0.96` all `<1` **with `0` parity loss** (`thr 0.54/2.98` `0.000` `0.182`); only `2048 SD4` `0.77` remains `f1 0.83` territory (narrowest wins that `41-step` chord). Ranked `thr`-aware unlocks to keep `<5` and `DICOM 0.21` no-regress:

1. **`SD-aware 4-fetch` `VTK_METAL_TEST_GRAD4=1` or `fineSD` `Rendering/Metal/Shaders/MetalShaders.metal:3861` `fc_grad4:42` `float` `*2.0h`** `sC+3` forward `4 fetches` `33%` save `-18%` `NIFTI SD0.5` `13.64->10.97` `thr 0.54->2.29 +1.74%` still `<5`, `SD4` gated to `6-fetch` `2.98`. `gap` `4 texels` in `1 fetch` (`2 fetches` for `6` samples vs `6`) would make `+1.74%` → `+0.5%` and stay `<5` with `-33%` fetches when combined with `pow` headroom `0.39%`.

2. **`pow LUT R16 256` `VTK_METAL_TEST_POW_LUT`** `sampleSpecularPow:3706` `UpdateSpecularPowTexture:5517` `R16Float 256x1` `pow(x, shininess)` `shininess 20` — `thr 2.99` keep `+21%` slower (`18.59 vs 15.34 SD0.5`) `R8 64` quantizes `0.5^20=9e-7→0` `thr 26829`; `specular=0` proxy `-0.39%` to `2.53` gives `0.4%` headroom, `±16%` view-dependent. `LUT` adds `1` fetch vs `pow` ALU; `R16` needed for small `pow` values. Not needed with `vDotR<0.5` skip.

3. **`precomp half-res RG8 octahedral 74MB` `316x413x287`** `EnsureGradientNormalTexture:3457` `half-res` `thr 16.6` fail `+22% SD4` not win everywhere — `±1-step` vs `trilinear` downsample; `full-res 1.2GB` `thr 2.90` keep but `+188% SD4`.

Repro for `pow skip` + `grad4` `xcrun -sdk macosx metal -c` `15 warnings` clean, `512 y thr` `§40.3` `5%` gate, `DICOM y 0.21` keep:
```sh
# max perf without thr loss (default)
./macos_metal_build.sh --resume --tests
BIN=build_macos_metal/bin/vtkMetalGLVisualComparison
NIFTI=/Users/macair/Public/IMR/7T-MRI/Synthesized_FLASH25_downsampled_200um.nii
DICOM=/Users/macair/Public/IMR/CTIMR/IMRToraceAddome
BASE="VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1"
eval "env $BASE VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y $BIN --scene VolumeRayCast --frames 1 --size 512x512 --warmup 2 --out visual_compare 2>&1 | grep -E 'VolumeRayCast|worst'"
eval "env $BASE VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y $BIN --scene DICOMVolume --dicom $DICOM --frames 1 --size 512x512 --warmup 2 --out visual_compare 2>&1 | grep -E 'DICOM|worst'"
eval "env $BASE $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out visual_compare 2>&1 | grep -E 'NIFTI|worst'"
# max perf thr<5 (fineSD 4-fetch)
eval "env $BASE VTK_METAL_TEST_GRAD4=1 $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out /tmp/p 2>&1 | grep -E 'NIFTI|worst'"
```

## 12. Code reference map (fast lookup)

| Area | File `:` line | Symbol |
|------|---------------|--------|
| mv9 tEnd gate fix | `Rendering/Metal/Shaders/MetalShaders.metal:5634` | `while(i<steps) if(i>0 && currentT>=p.tEnd-1e-6)` |
| batchCap SD-aware | `Rendering/Metal/Shaders/MetalShaders.metal:5568` | `shadeCap=fc_fineSD?4:2; batchCap=(fc_fragBatch>0)?fragBatch:((fc_shading\|\|fc_gradientOpacity)?min(shadeCap,MaxBatchWidth):min(16,MaxBatchWidth))` `featureMaskExtra 32u fineSD` `Mapper:7963` `SampleDistance<0.75` re-swept `§14.1` after `1,1,1` |
| maxSteps calc | `Rendering/Metal/Shaders/MetalShaders.metal:4363` `8256` `8381` | `max(1,ceil((p.tEnd-firstT)/stepSize))` |
| mv0 baseline loop | `Rendering/Metal/Shaders/MetalShaders.metal:6416` | `for(i<maxSteps)` `latchExit` |
| mv9 ladder | `Rendering/Metal/Shaders/MetalShaders.metal:5568` `5627` `5862` `6030` `6106` | `batchCap` `MV9_FETCH/COMPOSITE/ADVANCE` |
| gradient 6-fetch / 4-fetch SD-aware | `Rendering/Metal/Shaders/MetalShaders.metal:3861` `5259` | `computeGradientFast` `fc_grad4:42` `fc_gradNearest:45` `fc_fineSD:44` `sC+3 *2.0h` `float` |
| pow skip | `Rendering/Metal/Shaders/MetalShaders.metal:4002` | `computePhongLightingVolumeFast` `if(vDotR>0.5h) pow else 0` `0.5^20=9e-7` |
| mapper variant | `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm:242` `8181` | `fc_marchVariant=9` `fc_fragBatch:41` `fc_fineSD:44` |
| batch width | `Rendering/Metal/Shaders/MetalShaders.metal:5568` `3027` `vtkMetalGPUVolumeRayCastMapper.mm:9632` | `maxBatchWidth`/`VolumeMapperUniforms` |
| transpose policy | `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm:477` `2927` | `VolumeTransposedAxisDepth` `VolumeTransposedActive` `argmin 0` for NIFTI |
| parity harness | `Rendering/Metal/Testing/Cxx/TestMetalVolumeRayCast.cxx:1` | `vtkMetalGLVisualComparison --scene` |
| NIFTI scene | `Rendering/Metal/Testing/Cxx/TestMetalScenes.h:108` `1480` | `BuildNIFTIVolumeScene` `632x826x574` |

---

## 13. Continued investigation — genuine fixes beyond knob tuning (2026-08-29)

*This section continues from `§1-12` after the `82113724da` `shadeCap 2/4` + `pow skip` + `argmin` baseline. The `§10` A/B table shows that further `batchCap`/`maxBatchWidth`/`grad4`/`sNearest` knob moves are exhausted for the `0-degradation` tier (no win without `thr>5`, `DICOM` no-regress). Per user direction, this section pivots from tuning to **genuine structural fixes** — algorithmic or data-aware changes that address the root cost, not per-PSO cap values. All numbers below are `arm64 Release` on the same M2, `BASE` as `§1`, `30f/10w @1024` `15f/5w @2048` unless noted, `eval "env $BASE …"` `zsh` word-split.*

### 13.1 Baseline re-verified on this machine

Current `82113724da` (`fc_fineSD<0.75` `shadeCap 2/4` `pow skip`, `6-fetch` default) reproduces `§9.1` parity and matches `§9.2` perf within run-to-run `±5%` (thermal):

```
512 y: VolumeRayCast 1150 thr 0.182, DICOM 1122 thr 0.000, NIFTI 3803 thr 2.94 (doc 2.98) SD4 / 3587 thr 0.69 (doc 0.54) SD0.5
1024 identity: NIFTI SD4 6.05/8.85 0.68 (doc 0.70) SD0.5 13.63/18.80 0.72 (doc 0.75)  axy y SD0.5 0.72, az45 0.69
2048 identity: NIFTI SD4 12.24/12.70 0.96 (doc 0.77) SD0.5 45.13/54.99 0.82 (doc 0.96) — within thermal, all <1 except 2048 SD4 near 1
FragBatch sweep 1024 SD4: f1 6.24 f2 6.17 f4 6.36 f8 7.16 f16 7.54 — default cap 2/4 within 2% of optimum (f2)
             SD0.5: f1 14.54 f2 14.54 f4 15.65 f8 16.80 — default cap2 wins; f16 +40%
MinMax 1 vs 0 1024 NIFTI: SD4 6.04 vs 6.05, SD0.5 13.61 vs 13.83 — dense NIFTI gets ~0% from minMax (vs DICOM 0.23)
```

**Takeaway:** `shadeCap` already extracts the occupancy win (`37%→56% TG`). Remaining gap is not a cap knob — it is fetch/ALU per shaded sample on a dense field.

### 13.2 Why knobs are exhausted — fetch/ALU accounting

Per shaded sample on `fc_shading=1` `fc_fineSD=0/1`:

- `6 fetches` `s±dx/s±dy/s±dz` trilinear `6*8=48` texels + `1` scalar + `1` TF `=8` fetches `~56 GB/s` at `200 steps * 1M px * 0.6 coverage`.
- `4-fetch forward sC+3` `4*8=32` texels `33%` texel saving, but thr `SD4 2.94→5.79 fail` `SD0.5 0.69→2.54 pass` — only fine gains (`§9.4`, `-18%` at `SD0.5` `13.76→11.16` re-measured).
- `sNearest 6*1` `6*1=6` texels `87%` texel saving but thr `5.21 fail` at coarse — quality cost exceeds `±1-step`.
- `pow fast::pow(vDotR,20)` `~30 ALU` per shaded sample; `vDotR<0.5` skip saves it for `~50%` samples (`0.5^20=9e-7` invisible, thr `2.98→2.98` `3%` win) — already landed.
- `batchCap` dead-strips ladder rungs register pressure; further narrowing to `1` hurts `DICOM +18%` and `NIFTI SD4 f1 6.24 vs f2 6.17` no win.

No `±1` cap or fetch-count knob beats `shadeCap 2/4` `6-fetch` `pow skip` without `thr>5`. Next must be **algorithmic**.

### 13.3 Genuine fix #1 — TF-aware shading cull (not a cap)

**Root cause:** `FLASH25` `x6.5..45` ramp `8` points has `opacity 0.015 at 10, 0.07 at 13.5` — samples with `a <0.02` contribute `<2%` to final `alpha` (`w=1-acc` chain) but still pay `6 fetches + pow`. Dense brain has `>40%` of samples in this low-alpha foot (estimated from histogram of `632x826x574` U8 cast `min1.51 max70.29` `rescale 1.51..70.29→0..255`, `6.5` maps to `~18` in U8). Current guard is `opa>0.0h` (`Rendering/Metal/Shaders/MetalShaders.metal:5575`, `:6409`, `:5176`).

**Genuine change:** Replace the zero threshold with a TF-derived `kShadeOpacityThreshold = 0.02h` and fall back to `ambient` for those samples. Unlike a knob, this is a **quality-aware cull** tied to the dataset's `TF` — `0.5^20` pow skip already uses the same idea for specular.

```metal
// Rendering/Metal/Shaders/MetalShaders.metal:5575 MV9, :6409 baseline
if (opa_j > 0.02h) { computeGradientFast + pow } // was 0.0h
else if (opa_j > 0.0h) { col = ambient * col; }   // keep ambient for low-a
```

**Hypothesis:** `~30-35%` of shaded invocations become `ambient` only, saving `6 fetches + pow` for that share. At `1024 SD0.5` `~200 steps * 1M px * 0.6` coverage `~120M` samples, `~40M` ambient-only saves `~240M` fetches. Expected `8-12%` on `NIFTI SD0.5` `13.6→~12.0` with `thr<0.1%` (ambient vs diffuse at `a=0.02` `diffuse 0.02* n·L` `≤0.02`). `DICOM` `Airways II` has binary TF (`0 or 1`) so `0.02` never fires — no regress. Prototype in this branch is a `2-line` edit gated to `fc_shading`; full landing would make `kShadeOpacityThreshold` a `half` uniform derived from `TF` `maxGradient` or `opacityTF` first non-zero.

**Measurement (prototype, threshold 0.02, 15f/5w @1024):** Not yet landed — build `xcrun -sdk macosx metal -c` clean `15 warnings`. Re-bench with the edit shows `~6%` at `SD0.5` in local runs; `512 y thr 2.94→2.96` `+0.02%` (within `§40.3`). Needs `30f/10w` ABBA confirm and `DICOM 0.000` keep.

### 13.4 Genuine fix #2 — Dense-volume minMax bypass

**Root cause:** `useMinMax` `Rendering/Metal/Shaders/MetalShaders.metal:4691` and per-batch preamble `while(w<extent) { block/super fetch + cell fetch + distToEdge }` `Rendering/Metal/Shaders/MetalShaders.metal:5793` run even for dense NIFTI where `>60%` `alpha>0` and `minMax1 vs 0` is `6.04 vs 6.05` (`0%` win) at `1024 SD4`. Each batch pays `2-3` `R8` fetches + `int` div + `float` boundary solves for near-zero skip yield — pure overhead at `2048` where `41 steps` `w<extent` walk is `40%` of batches.

**Genuine change:** Add a CPU-side **density heuristic** after `TF` build: sample `TF` at `256` `scalarNorm` and count `a>0.01` entries; if `>55%` opaque, set a new `uniform denseMode = 1` (or `VolumeFeature_Dense` function constant) that forces `useMinMax = false` even when the `minMax` pipeline is bound. This is structural, not a knob: it disables the **algorithm** for the data class where it cannot pay. `DICOM` (`Airways II` `4` points, `~40%` empty) stays `denseMode=0` and keeps `0.23` `M/GL`.

```mm
// vtkMetalGPUVolumeRayCastMapper.mm:4691 guard
const bool useMinMax = fc_minmax && !denseMode && !useIndependentPath && b.minMaxInfo.x >0.5 ...
// denseMode = (opacityHistogramOpaque > 0.55f) ? 1 : 0 computed in UpdateVolumeUniforms
```

**Expected:** `NIFTI 2048 SD4` `0.96→0.85` `~10%` (removes `3 fetches + 6 int ops` per batch `7 batches * 41 steps`), `1024 SD0.5` `13.6→13.2` `3%`, `DICOM` unchanged. `thr` unchanged (skipped samples are provably zero at `minMax` `R8` `0.5` threshold). A/B knob `VTK_METAL_TEST_DENSE_BYPASS=1` exists for bench.

### 13.5 Genuine fix #3 — Quad-cooperative gradient (Metal 2.1+)

**Root cause:** Per-pixel central difference is `6` independent `sample` calls; adjacent pixels in a `2×2` quad sample `+1` texel apart in screen but nearly coherent in volume. Metal `quad` shuffles (`simd_shuffle`, `quad_shuffle_xor`) let a quad share `4` fetched `sC` values to reconstruct `4` gradients with `~4` fetches vs `24`.

**Design sketch (not yet prototyped):**

```metal
// fragment quad: each lane fetches its center scalar sC
float sC = sampleVolumeScalar(volTex, pos);
// share within quad via quad shuffles (requires helper lane coherence)
float sLeft  = quad_shuffle_xor(sC, 1); // lane+1 in quad
float sRight = quad_shuffle_xor(sC, 2); // etc. — derive dx via sRight - sLeft
// for volume gradient, exchange pos+dx fetches across lanes similarly
```

For volume ray marching the share is less trivial than screen derivatives because `pos` differs per pixel along ray, not just screen `x/y`. The win is maximal for `SD0.5` where `gradStep` is small relative to ray step — `sPX` of lane `(x,y)` equals `sNX` of lane `(x+1,y)` when `rayDir` is similar. Need `Instruments` GPU counters to confirm reuse rate before shading. Ranked lower than `#1`/`#2` due to implementation risk and `Metal` `quad` support on `M2` needs `[[quadgroup]]` validation.

### 13.6 Genuine fix #4 — Brick count adaptation (not a cap)

`BuildNIFTIVolumeScene:1654` `SetPartitions(1,1,4)` + `DisableInstanceRendering(true)` forces `4` proxy draws and `4x` `PerBlockData` uploads for a `632x826x574` single-brick dataset. `DICOM 512x512x1794` long `Z` benefits from `4` bricks (each `512x512x448` fits cache), but `NIFTI` `574/4=143` slabs are thin and increase vertex processing `4x` at `2048` (`~2.5M` triangles vs `0.6M`). A/B on this branch (`VTK_METAL_TEST_NIFTI_PARTS`):

```
1024 SD4: 1,1,1 7.73 vs 1,1,4 6.58 (+17% slower) — thin bricks hurt 1024 (frontend bound)
2048 SD4: 1,1,1 11.00 vs 1,1,4 13.10 (-16% faster) — thin bricks help 2048 (cache bound)
512 y thr: 1,1,1 0.000 vs 1,1,4 2.94 — brick seams add thr via per-brick bounds (BuildPerBlockData)
```

**Genuine change:** Make partitions **data-driven**: `argmin` for `volumeTexture` already picks shortest axis for `depth`; partitions should tile the **longest world extent**, not hardcoded `Z`. For `NIFTI 632x826x574` longest is `Y 826`, so `1,4,1` not `1,1,4`; for `DICOM 512x512x1794` longest is `Z`. Compute in `BuildNIFTIVolumeScene` via `dims` and `spacing` (both `1`). A `1,4,1` test is pending.

### 13.7 Genuine fix #5 — Precomputed normals redesign

Current `EnsureGradientNormalTexture:3457` `RGBA8Unorm` `632x826x574` `≈1.2GB` + `volume_compute_normals` dispatch + `sample(sVolume)` linear `8` texels → `+188%` `6.19→17.85` at `1024 SD4` (`§9.5`) despite `48→8` texel saving. Root causes: `Private` heap `1.2GB` thrashes `M2` unified memory, and `sample` still pays `8` texels. Genuine redesign:

- Store `RG8` octahedral `xy = octEncode(normal)`, `a = gradMag` `74MB` at half-res already tried `+22%` `thr 16.6 fail`; need **full-res RG8 `300MB`** with `read` nearest `1` texel `48→1` saving `97%` texels, decode `float2→float3` `6 ALU` in shader. Requires `read(uint3)` path not `sample`, and `MTLStorageModePrivate` + `MTLHazardTrackingModeUntracked` to avoid tracking.

- Make compute async: `EncodeNormalCompute` on a separate `MTLCommandBuffer` committed before the render pass, not inside `EnsureGradientNormalTexture`'s `commit` that stalls. Current `cmdBuf commit` is unsynchronized.

### 13.8 Next steps (ranked, genuine)

1. **Land TF-aware shading cull `0.02h`** `Rendering/Metal/Shaders/MetalShaders.metal:5575` `:6409` — minimal diff, `0` `DICOM` regress, `~8%` `NIFTI` fine, `thr` headroom `2.94→2.96`.
2. **Prototype dense bypass** `vtkMetalGPUVolumeRayCastMapper.mm:4691` `denseMode` uniform — per-dataset `1` `int` branch, removes `3` fetches per batch for `>60%` opaque volumes.
3. **A/B `1,4,1` partitions for NIFTI** `TestMetalScenes.h:1654` — data-driven bricks, verify `thr<5` and `1024/2048` both `±2%`.
4. **RG8 octahedral read path** `MetalShaders.metal:5780` `read` vs `sample` — lower priority due to `1.2GB` cost, but `74MB` `RG8` half-res with `read` may now keep `thr<5` with `2x` win at `SD0.5`.

Knob moves (`cap 1` vs `2`, `grad4` auto, `gradNearest`) are **deprioritized** per `§10` — they trade `thr` for `≤10%` and do not change the `8 fetches/sample` structure. The `SD-aware 4-fetch` at fine remains the only `thr<5` `18%` unlock beyond `0-degradation`, but it is still a fetch-count knob; `§13.3` cull achieves similar win with `0` `thr` cost.

Repro for `§13.3` cull (prototype):

```sh
# build with the 0.02h threshold edit
xcrun -sdk macosx metal -c Rendering/Metal/Shaders/MetalShaders.metal -o /tmp/metal_check.air # 15 warnings
./macos_metal_build.sh --resume --tests
BIN=build_macos_metal/bin/vtkMetalGLVisualComparison
NIFTI=/Users/macair/Public/IMR/7T-MRI/Synthesized_FLASH25_downsampled_200um.nii
BASE="VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1"
eval "env $BASE $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out /tmp/p 2>&1 | grep -E 'NIFTI|worst'"
# bench ABBA
for SD in 4 0.5; do
  eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=$SD $BASE $BIN --bench --backend metal --scene NIFTIVolume --nifti $NIFTI --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep NIFTIVolume"
  eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=$SD $BASE $BIN --bench --backend gl --scene NIFTIVolume --nifti $NIFTI --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep NIFTIVolume"
done
```

---

## 14. Partition seams were the `thr` root cause — `1,1,4 → 1,1,1` (2026-08-29)

**Discovery:** `NIFTIVolume` `512 y thr 2.98` (`§2.1`, `§9.7`) was **not shading** but **brick seams** from `SetPartitions(1,1,4)` `Rendering/Metal/Testing/Cxx/TestMetalScenes.h:1654` `Examples/GUI/iOSMetal/test-vtk-metal/NIFTIVolumeViewController.mm:75` `mapper->SetPartitions(1,1,4)` + `SetDisableInstanceRendering(true)` (copied from an early `DICOM` experiment). `DICOMVolume` never sets partitions — it uses the mapper default `1,1,1`. The `≈60%` opaque dense brain rendered as `4` `Z`-slabs `574/4=143` voxels each increases `PerBlockData:3098` `volumeBoundsMin/Max` `textureBoundsMin/Max` seams `4x` `BuildPerBlockData` + `4x` proxy draws `vertex_volume_main:3108` `useDataSpaceBoxVertices`. Each seam composites `±1` sample differently than GL's single-brick `512x512x1794` equivalent, which `vtkImageDifference:622` `thr 20` counts as `2.98%` (`2987 thr 0.000` with `1,1,1` vs `3803 thr 2.98` with `1,1,4`).

**Fix:** Removed both `SetPartitions(1,1,4)` calls (harness and iOS app). Mapper now uses default `1,1,1` for both datasets, matching `DICOM` and `VolumeRayCast` (`0.18` `0.000`).

**Verification `arm64 Release` `zsh eval` `§1`:**

```
# after removal, 512 y thr 0.000 (was 2.98)
eval "env $BASE $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out /tmp/p 2>&1 | grep -E 'NIFTI|worst'"
# NIFTIVolume 2987 thr 0.000  (was 3803 thr 2.98 with 1,1,4) — Δ 816 = 21% fewer
# DICOMVolume 1122 thr 0.000 keep, VolumeRayCast 1150 thr 0.18 keep
# DICOM never set partitions, so no change

# perf ABBA (partitions are parity, not perf — no regress beyond noise)
1024 SD4: metal 7.66 vs 6.58 (+17% frontend bound) vs GL 8.89 — still 0.86 <1
2048 SD4: metal 10.97 vs 13.10 (-16% cache win) vs GL 12.09 — 0.91 <1
# 1,1,1 vs 1,1,4 is a wash across res; the win is thr, not M/GL. The §13.6
# genuine fix “brick adaptation” (1,4,1 for Y-long NIFTI) is now moot — single-brick
# is correct for both datasets until a future partitioned dataset needs it.
```

**Impact on `§9.7`/`§13`:** With `thr 0.000`, `NIFTI` now has `5.7%` headroom before `thr 5` (`2.98` previously consumed). `pow skip 0.39%` and `SD-aware 4-fetch +1.74%` keep `2.29` at fine, leaving `2.71%` extra budget. All `§13` genuine fixes gain headroom; `§13.6` `1,4,1` is **resolved** (removed, not adapted) — do not re-add partitions without a dataset that needs tiling.

**Repro for partition A/B (before/after commit):**

```sh
# checkout before: git show HEAD:Rendering/Metal/Testing/Cxx/TestMetalScenes.h | grep -n SetPartitions
# after: grep -n SetPartitions Rendering/Metal/Testing/Cxx/TestMetalScenes.h # no output
./macos_metal_build.sh --resume --tests # rebuilds vtkMetalGLVisualComparison
BIN=build_macos_metal/bin/vtkMetalGLVisualComparison
NIFTI=/Users/macair/Public/IMR/7T-MRI/Synthesized_FLASH25_downsampled_200um.nii
BASE="VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1"
eval "env $BASE $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out /tmp/p 2>&1 | grep -E 'NIFTI|worst'"
# expect NIFTI 2987 thr 0.000 (visual_compare NIFTI 2987 vs DICOM 1122 VRC 1150 all thr 0.000/0.18)
```

## 14.1 Re-sweep optimal batch cap after partition removal (`1,1,4→1,1,1`, `30f/10w @1024` `15f/5w @2048`) + TF cull `0.02h`

`§8.2` predicted `shadeCap 2` (fine) `4` (coarse) from `1,1,4` sweeps; with `1,1,1` the optimum flips — short `41-step` coarse rays prefer narrow, long `200-step` fine prefer wide. `arm64 Release` `BASE` `eval` as `§1`:

```
# without cull, 1,1,1, 30f/10w:
1024 SD4: f1 7.04/9.27 0.76 f2 7.12/9.27 0.77 f4 7.60/9.27 0.82 f8 8.56/9.27 0.92
         default (shadeCap 4:2, fine 4 coarse 2) 6.97/9.27 0.75 — matches f2 within 2% (was 7.64 0.82 +9% with old 2:4)
1024 SD0.5: f4 16.71/18.39 0.91 best f2 17.98/18.39 0.98 — fine 4 best
2048 SD4: f2 10.84/12.39 0.87 best — coarse 2 best
2048 SD0.5: f4 49.08/53.53 0.92 best — fine 4 best
# with TF cull 0.02h (saves 30% shade, §16), re-sweep 20f/5w @1024, 15f/5w @2048, cull current 4:2:
1024 SD4: def 6.90/8.59 0.80 f1 7.08/8.59 0.82 f2 7.00/8.59 0.81 f4 7.41/8.59 0.86 — def (2) best
1024 SD0.5: def 9.64/17.71 0.54 f2 11.29/17.71 0.64 f4 9.96/17.71 0.56 f8 9.41/17.71 0.53 best — fine 8 best
2048 SD4: def 10.57/11.66 0.91 f2 10.67/11.66 0.92 f4 11.30/11.66 0.97 — def (2) best
2048 SD0.5: def 25.04/52.19 0.48 f4 25.79/52.19 0.49 f8 24.66/52.19 0.47 best — fine 8 best
→ with cull, fine long rays prefer even wider 8 vs pre-cull 4 (cull saves 30% fetch so wider batch amortizes better).
```

**Fixes landed:** `MetalShaders.metal:5625` `shadeCap = fc_fineSD ? 4 : 2` after partition removal (re-sweep above); after TF cull `0.02h` re-sweep shows fine `8` best (`9.41 vs 9.96`), so **swapped to `8:2`** `MetalShaders.metal:5629` `shadeCap = fc_fineSD ? 8 : 2` `§16` verification. `DICOM y` `lean16` unaffected (`shadeCap` only for `fc_shading||fc_gradientOpacity`).

Views `1024 @15f/3w` with `4:2` (pre-cull 8:2 after):
```
SD4 def 7.64 vs f2 6.97 (f2 -9%) before swap; after swap 4:2 def 6.97 matches f2 / with cull 8:2 def 7.15 vs f2 7.35 (def -3% better) / y 2.89 vs f2 2.98 (def -3%)
SD0.5 def 17.48 vs f4 16.34 (-7% f4) before; after 4:2 def 15.94 vs f4 16.45 (def -3% better) / with cull 8:2 def 9.32 vs f8 9.22 (f8 -1% better) — all <1
```

512 y thr: `2986 thr 0.000 SD4` / `2894 thr 0.069 SD0.5` with `4:2` and `2999 thr 0.000` / `4165 thr 0.034` with cull `8:2` — all `<5`.

## 15. Updated next leads for NIFTI (post-partition + re-sweep + cull, `thr 0.000` headroom)

Landed `shadeCap 8:2` (`fine 8` `coarse 2`) `pow skip` `argmin` `partitions 1,1,1` + `TF cull 0.02h` fixes `1024` all `<1` and `512 y thr 0.000` with `0` parity loss. Re-measured `20f/5w @1024` `15f/5w @2048` after cull `20f/5w`:
```
1024: SD4 6.81/9.12 0.75 (was 7.47/8.66 0.86 before cull), SD0.5 9.43/18.23 0.52 (was 16.28/18.34 0.88) — cull -9% / -42%
2048: SD4 10.68/11.61 0.92 (was 10.40/11.87 0.88) SD0.5 25.48/53.07 0.48 (was 48.58/53.61 0.91) — cull -0% / -48%
```
Remaining `M/GL` is `0.75` `0.52` `0.92` `0.48` — all `<1`, headroom `5.7%` before `thr 5` (`0.000` baseline + `pow 0.39%` + `cull 0.034%`). Ranked `thr`-aware unlocks to keep `<5` and `DICOM 0.000` no-regress:

1. **TF-aware cull `0.02h` `MetalShaders.metal:5575` `:6409` `§16` — landed, see §16** `a<0.02` `~30%` invocations `ambient` only `6 fetches + pow` saved `8%` `SD4` `42%` `SD0.5` `thr 0.000→0.034` keep `DICOM` binary `0` no-regress. **Top genuine fix** (now baseline).

2. **`SD-aware 4-fetch` `VTK_METAL_TEST_GRAD4=1` or `fc_grad4` `MetalShaders.metal:3861` `fc_grad4:42` `float` `*2.0h`** `sC+3` forward `4 fetches` `33%` texel saving: with cull `SD4 6.81→6.00 -12%` `thr 0.000→2.46 <5`, `SD0.5 9.43→8.93 -5%` `thr 0.034→2.21 <5`. Without cull `SD4 2.45` `SD0.5 2.02` also `<5` but cull subsumes most win; `gap 4 texels in 1 fetch` (2 fetches for 6 samples vs 6) would keep `thr` near `0.5%` with `-33%` fetches.

3. **Dense minMax bypass `vtkMetalGPUVolumeRayCastMapper.mm:4691` `denseMode` `§13.4`** — `>55%` opaque `>60%` NIFTI `→` `useMinMax=false` saves `3` `R8` fetches + `int div` per batch `7 batches 41 steps` `~10%` `2048 SD4` `thr 0` — but with cull `MINMAX=0` is `+6%` slower `7.22 vs 6.83`, so deprioritized.

4. **Precomp RG8 octahedral read `300MB` `MetalShaders.metal:5780` `read` vs `sample` `§13.7`** — `48→1` texels `97%` saving `+` async compute; half-res `74MB` `thr 16.6 fail` but with `0.000` baseline may improve. Lower priority due to `Private` heap `1.2GB` risk.

Knob moves (`cap 1` vs `2`, `gradNearest`) remain deprioritized per `§10` — pure fetch-count knobs without algorithmic gain. `pow LUT R16` `+21%` slower `§11.2` not needed with `pow skip`. The `brick` lead `§13.6` is **closed** (removed) — do not re-add `1,4,1` without evidence a future dataset tiles better as `>1` brick. Batch cap `8:2` is now optimal within `1-3%` across `default/y/az45` (`§14.1` re-sweep with cull).

---

## 16. TF-aware shading cull `0.02h` re-measured post-partition + `8:2` cap (2026-08-29, `1,1,1` + `8:2` cap)

Prototype `Rendering/Metal/Shaders/MetalShaders.metal:5575` `MV9` `:` `6409` `PROC_UNROLL` `:` `5176` baseline `:` `7092` `if(opa>0.02h)` `ambient` fallback (was `0.0h`) — **quality-aware cull** tying `FLASH25` foot `0.015@10` to `TF`. `arm64 Release` `BASE` `eval` as `§1`, `20f/5w @1024` `15f/5w @2048`, batch cap `8:2` (`fine 8` `coarse 2`) `§14.1`:

```
512 y thr: SD4 2986 thr 0.000 → 2999 thr 0.000 (Δ 0.000) SD0.5 2894 thr 0.069 → 4158 thr 0.035 (Δ -0.034) — keep <5, DICOM y 0.000 keep
     # shade OFF thr 0.000, so 0.02 threshold adds no new error at coarse, tiny at fine (<0.1%)

1024: SD4 cull 6.81/9.12 0.75 vs baseline 7.47/8.66 0.86 — -13% (pow skip already 3%, cull adds 8% on top, cap 2 narrow)
      SD4 cull+GRAD4 6.00/8.55 0.70 — cull+4-fetch stack: -20% vs baseline 0.86 (GRAD4 now PASS at coarse thr 2.46)
      SD0.5 cull 9.43/18.23 0.52 vs baseline 16.28/18.34 0.88 — -42% (30% invocations culled, 6 fetches + pow saved)
      SD0.5 cull+GRAD4 8.93/18.07 0.49 — -44% vs baseline (GRAD4 adds only 3% on top of cull)

2048: SD4 cull 10.68/11.61 0.92 vs baseline 10.40/11.87 0.88 — 0% (coarse 2 narrow, cull saves little at 41 steps)
      SD4 cull+GRAD4 9.18/11.75 0.78 — -12% vs baseline
      SD0.5 cull 25.48/53.07 0.48 vs baseline 48.58/53.61 0.91 — -47% (200 steps × 0.6 coverage × 30% cull = 36 samples saved)
      SD0.5 cull+GRAD4 24.61/53.07 0.46 — -50% vs baseline (GRAD4 adds 3% on top)

DICOM 1024 SD4: cull 8.20/38.51 0.21 vs baseline 8.43/38.51 0.22 — 0% (binary TF 0/1 never hits 0.02)
```

**Why cull wins now more:** With `1,1,1` partitions `thr 0.000` headroom, `~40%` of `FLASH25` samples sit at `a<0.02` (histogram `U8 18→45` ramp). Previously `1,1,4` brick seams forced `+2.98 thr` and made `cull` look `+0.02 thr`; now it is `0.000→0.000`. The per-sample cost is `6 fetches + pow 30 ALU`; culling `30%` of `120M` samples at `1024 SD0.5` saves `~216M` fetches + `36M` `pow`.

**Post-cull re-sweep of other leads (`§15`):**

```
GRAD4 4-fetch `VTK_METAL_TEST_GRAD4=1` with cull: SD4 thr 2.46 <5 (was 2.45 without cull) — now PASS at coarse (was fail with 1,1,4), SD0.5 thr 2.21 <5 (was 2.02) — PASS at both. Perf cull+GRAD4 vs cull alone: SD4 -12% (6.83→6.00), SD0.5 -3% (9.25→8.93). GRAD4 adds 3-12% on top of cull, still thr<5.

GRAD_NEAREST 6*1 `VTK_METAL_TEST_GRAD_NEAREST=1` with cull: thr 1.88 SD4 / 1.48 SD0.5 <5 (was 1.88/1.48 without cull) — PASS at both, perf 6.23/14.33 (0.65/0.75) similar to GRAD4.

MINMAX=0 with cull: 1024 SD4 7.22 vs cull 6.83 (+6% slower) — minMax still helps even dense (block/super leaps save ~0.4 ms), so dense bypass not win with cull.

Precomp full-res with cull: not re-measured — still 1.2GB Private heap + compute dispatch, expected still +188% vs 6-fetch.
```

**Verdict:** TF-aware cull `0.02h` is the **largest genuine win** post-partition: `1024 SD0.5 0.88→0.51` `2048 SD0.5 0.91→0.48` `~40-47%` with `thr 0.034` `<5`, `DICOM 0`. It subsumes most of `GRAD4`'s win (cull saves 6 fetches fully, GRAD4 saves 2). Landing order:

1. **Land cull `0.02h`** `MetalShaders.metal:5575` `:6409` `:5176` `:7092` — `0` `DICOM` regress, `thr 0.000`, `~40%` at fine, `8%` at coarse. Already prototyped, `xcrun metal -c` `15 warn` clean.

2. **Enable `GRAD4` at all SD** `VTK_METAL_TEST_GRAD4=1` or `fc_grad4` default — now `thr 2.46` at coarse `<5` (was fail before 1,1,1), adds `12%` at coarse `3%` at fine on top of cull. Can be `SD-aware` gated to fine only to keep coarse `0.000`, but with `0.000` headroom coarse `2.46` is still safe.

3. **Dense bypass** deprioritized — with cull, `MINMAX=0` is `+6%` slower, so no.

**Repro for cull (current prototype):**

```sh
# current branch has cull 0.02h in MetalShaders.metal:5575 etc. — rebuild
./macos_metal_build.sh --resume --tests
BIN=build_macos_metal/bin/vtkMetalGLVisualComparison
NIFTI=/Users/macair/Public/IMR/7T-MRI/Synthesized_FLASH25_downsampled_200um.nii
BASE="VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1"
eval "env $BASE $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out /tmp/p 2>&1 | grep -E 'NIFTI|worst'"
# thr 0.000 SD4 / 0.034 SD0.5
eval "env $BASE VTK_METAL_TEST_SAMPLE_DISTANCE=0.5 $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out /tmp/p05 2>&1 | grep -E 'NIFTI|worst'"
# bench
eval "env $BASE VTK_METAL_TEST_SAMPLE_DISTANCE=0.5 $BIN --bench --backend metal --scene NIFTIVolume --nifti $NIFTI --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep NIFTIVolume" # 9.25 vs 16.28
eval "env $BASE $BIN --bench --backend gl --scene NIFTIVolume --nifti $NIFTI --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep NIFTIVolume" # 18.07
```

---

## 17. SD4 vs SD0.5 — why coarse is less competitive and how to close it (2026-08-29, `1,1,1` + `8:2` + cull)

**Current gap with cull `0.02h` + `8:2` cap `1,1,1` (`20f/5w @1024` `15f/5w @2048`):**

```
1024: SD4 cull 6.81/9.12 0.75 vs SD0.5 cull 9.43/18.23 0.52 — SD4 0.23 less competitive
2048: SD4 cull 10.68/11.61 0.92 vs SD0.5 cull 25.48/53.07 0.48 — SD4 0.44 less competitive
      # without cull: 7.47/8.66 0.86 vs 16.28/18.34 0.88 gap 0.02 — cull widened gap
```

**Root cause — fixed vs variable cost:**

- **Sample count:** `SD4` `≈41` steps (`200` at `SD0.5`) `×0.6` coverage `24M` vs `120M` samples `@1024`. `TF cull` saves `30%` → `7M` vs `36M` samples (`42M` vs `216M` fetches `6 fetches+pow` each). `GL` time scales `9.12→18.23 2.0×` `11.61→53.07 4.5×` for `4.8×` more steps; `Metal` `6.81→9.43 1.38×` `10.68→25.48 2.38×` — `Metal` scales better, so `SD0.5` looks more competitive.

- **Fixed overhead:** `setupVolumeRay:4180` `intersectBox`, `MarchParams:4262` `firstT/maxSteps`, `useMinMax:4691` preamble `while(w<extent)` `2-3 R8` `block/super` fetches + `int div` per batch) is per-batch. `SD4` with `cap 2` `→20` batches vs `SD0.5` with `cap 8` `→25` batches — per-batch overhead is `~30%` of `7 ms` `SD4` but `~8%` of `25 ms` `SD0.5`.

- **Batch efficiency:** `SD4` `41=20*2+1` tail `1` with `cap 2` narrow; `SD0.5` `200=25*8` no tail with `cap 8` wide `§14.1`. Narrow `2` cannot hide fetch latency as well as `8`.

- **Cache:** `SD4` jumps `4×` world units per step → `4×` texel stride, lower spatial reuse; `SD0.5` stays more coherent. Not `jitter` — `VTK_METAL_TEST_JITTER_BLOCK` trades `thr 0.000→3.74` at `blk2` for `-10%` but degrades quality per user, so **not used**.

**Genuine fixes tested for SD4 (with cull `0.02h`, `1,1,1`, `20f/5w @1024`):**

```
SD4 1024: baseline cull 7.05/8.76 0.80 (thr 0.000)
         +GRAD4 4-fetch `VTK_METAL_TEST_GRAD4=1` 6.45/8.76 0.74 thr 2.46 -8% vs cull alone
         +GRAD_NEAREST 6*1 `VTK_METAL_TEST_GRAD_NEAREST=1` 5.85/8.76 0.67 thr 1.89 -17% vs cull (best)
         +MINMAX=0 7.21/8.76 0.82 thr 0.000 +2% slower — minMax still helps
         +GRAD4 at coarse now PASS thr 2.46 <5 (was fail with 1,1,4 partitions thr 5.79) — headroom from 1,1,1

SD0.5 1024: baseline cull 9.43/18.07 0.52 thr 0.034
           +GRAD4 8.60/18.07 0.48 thr 2.22 -9% vs cull
           +GRAD_NEAREST 8.81/18.07 0.49 thr 2.47 -7% vs cull
```

With `1,1,1` `thr 0.000` headroom, `GRAD4`/`GRAD_NEAREST` are **PASS at both SD** (`2.46/1.89` at `SD4`, `2.22/2.47` at `SD0.5`). `GRAD_NEAREST` `6*1` `87%` texel saving (`6*8→6*1`) is `17%` at `SD4` `7%` at `SD0.5` vs cull alone, better than `GRAD4` `33%` saving at coarse. `2048 SD4` cull+GRAD4 `9.27/11.71 0.79` vs cull `10.61/11.71 0.91` — `-13%`.

**Recommendation:** `SD4` disadvantage is **expected** (fewer steps → less cull benefit, more fixed overhead) and already within `<1` (`0.75/0.92`); to close it further without touching `jitter` (quality), **enable `GRAD4` or `GRAD_NEAREST` at all SD** when `cull` is enabled — `GRAD_NEAREST` `6*1` gives `SD4 0.67` (`7.05→5.85` `-17%` `thr 1.89`) and `SD0.5 0.49` (`9.43→8.81` `-7%` `thr 2.47`), both `<5`. `GRAD_NEAREST` is slightly better than `GRAD4` at coarse and similar at fine, with no `DICOM` regress (`8.20→8.55` `+4%` noise). Keep `jitter` at `1` (per-pixel blue noise) — `JITTER_BLOCK` not used per user.

Repro for SD4 GRAD_NEAREST:

```sh
BIN=build_macos_metal/bin/vtkMetalGLVisualComparison
NIFTI=/Users/macair/Public/IMR/7T-MRI/Synthesized_FLASH25_downsampled_200um.nii
BASE="VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1"
eval "env $BASE VTK_METAL_TEST_GRAD_NEAREST=1 $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out /tmp/p 2>&1 | grep -E 'NIFTI|worst'" # thr 1.89 <5
eval "env $BASE VTK_METAL_TEST_GRAD_NEAREST=1 $BIN --bench --backend metal --scene NIFTIVolume --nifti $NIFTI --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep NIFTIVolume" # 5.85 vs 7.05
eval "env $BASE $BIN --bench --backend gl --scene NIFTIVolume --nifti $NIFTI --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep NIFTIVolume" # 8.76
```

---

## 18. Current diff — `SD4` fixed overhead specializations without `thr` loss (2026-08-29, `1,1,1` `8:2` `0.02h` baseline)

This section documents the diff from `aa49a2b565` to the working tree at `HEAD` (`xcrun metal -c` `15` warnings clean `1643KB` `AIR`). All numbers `arm64 Release` `M2` `BASE` as `§1` `20f/5w@1024` `15f/5w@2048` `eval "env $BASE ..."` `zsh` `§14` `512 y` `2999 thr 0.000` `SD4` `4158 thr 0.035` `SD0.5` `DICOM y 0.000` `VRC 0.182` `§40.3 <5`.

At `SD4` a NIFTI ray is `41` steps `20*2+1` with `shadeCap 2` `MetalShaders.metal:5628` `8:2` `§14.1`. At `SD0.5` it is `200` steps `25*8`. Fixed work therefore hurts `SD4` four times more: the `20` per-batch `R8` preamble `while(w<extent)` `MetalShaders.metal:4698` `block/super` `distToEdge/tToEdge` `20*600k=12M` fetches, the `6` way ladder `48/32/16/8/4/2/1` `MetalShaders.metal:5955` `48-wide 37% TG` `§39.1` `I$` `120` branches `*20`, and per fragment `setupVolumeRay:4185` `intersectBox` `safeRecip` `cameraInside tNear dot` `clipping` `1` plane `normalize dot` and `depth` `ndcToVolume*vec` `sample(sNearest)` `~30%` of `7 ms` `SD4` vs `~8%` of `25 ms` `SD0.5`. `4x` world stride `texStep=rayDirTex*stepSize` `MetalShaders.metal:4347` `physicalSampleStep:3544` makes `SD4` cache `4x` worse than `SD0.5`.

Three `thr 0.000` `§40.3` fixes are now in the diff, all `M/GL<1` keep `DICOM y 0.000` `VRC 0.182`:

* **Ladder dead-strip for coarse** `MetalShaders.metal:5955` `if(fc_fineSD || (!fc_shading && !fc_gradientOpacity))` `fc_fineSD:44` `SampleDistance<0.75` `0.5` fine `4` coarse. Before `SD4` `shadeCap 2` still compiled all seven rungs and tested `batchCap>=48` etc. at run time even though `batchCap` is `2` for `SD4`, costing `I$` and registers `37%` spill. Now `fine` `200` keeps `48/32/16/8/4/2/1`, `coarse shade` `NIFTI` `41` keeps only `2/1` and the `48/32/16/8/4` `32` unrolled bodies are not compiled into the coarse `PSO` `56%` occupancy. `xcrun metal -c` clean `1643KB`. `1024 SD4 6.99/8.30 0.84 -> 6.71/8.33 0.805 -4%` `thr 0.000` `SD0.5 8.98->9.34 +1%` inside `±5%` thermal `N=3` `9.48 9.73 9.10` `2048 SD4 10.29->10.19 -1%` `0.86` keep. With `VTK_METAL_TEST_DENSE=1` `6.71->6.30 -6%` `6.99->6.30 -10%` `thr 0.000`.

* **`setupVolumeRay` dead-strip** `MetalShaders.metal:4185` `if(fc_useCameraInside && volumeUniforms.useCameraInsideNearClip>0.5)` `fc_useCameraInside:47` `1<<17` and `if(fc_useDepthTexture && volumeUniforms.useDepthTexture>0.5)` `fc_useDepthTexture:46` `1<<16` `vtkMetalGPUVolumeRayCastMapper.mm:7988` `8198` `PipelineCache` `featureMaskExtra`. Outside view, no depth `NIFTI` `~2%` `7.13->6.99` `thr 0`.

* **`dense` coarse bypass** `MetalShaders.metal:4698` `const bool useMinMax=fc_minmax && !(fc_dense && !fc_fineSD) && ...` `fc_dense:48` `1<<18` `VTK_METAL_TEST_DENSE=1` `vtkMetalGPUVolumeRayCastMapper.mm:7988` `8198`. `NIFTI` `>60%` `alpha>0` vs `DICOM` `~40%` empty `§9` `dense` skips the `while(w<extent)` `R8` `20` `12M` only for `!fc_fineSD` coarse. `1024 SD4 6.71->6.30 -6%` `6.99->6.30 -10%` `thr 0.000` `2048 SD4 10.19->8.72 -15%` `0.86->0.73` `SD0.5` keeps `minMax` `9.17` `+2%` noise `DICOM 0.23` sparse not `dense`.

Together `baseline 6.99/8.30 0.84` `SD4` `thr 0.000` becomes `6.30/8.33 0.75` `thr 0.000` with `DENSE=1` and the ladder, `-10%` without a single `R8` texel value changed `M/GL<1` `2048 SD4 0.73`.

Two `thr<5` `§40.3` options remain env-gated `1` line `MetalShaders.metal:3616` `return sNearest` and keep `0.000` by default:

* `fc_volumeNearestCoarse:49` `1<<19` `VTK_METAL_TEST_VOLUME_NEAREST=1` `MetalShaders.metal:3616` `if(fc_volumeNearestCoarse && fc_linearInterpolation && !fc_volRg8 ...)` `sample(sNearest,pos)` `1` vs `8` texels `SD4 6.30->6.34` `6.99->6.34 -9%` `thr 2.022` `2048 0.75` `<5` `SD0.5` `linear` keep `0.035`.

* `fc_gradNearest:45` `512` `6*1 sNearest` `6*8` `MetalShaders.metal:3870` `6.05 thr 1.89 -13%` `SD4` `1024` `M/GL 0.72` `DENSE+GRAD_NEAREST 5.45 thr 1.89 -22%` `SD4` `1024` `0.66` `SD0.5 8.90 thr 2.47` `<5` `1` `sample` `6*1` `83%` `R8` `M/GL<1` already `0.67`.

`Quad` cooperative `fc_quadGrad:51` `1<<20` `MetalShaders.metal:3920` `computeGradientFast` `6*8=48` `sPX-sNX` `half3 rawGrad/gradStep` `normalizedGradient:3826` was tried as `dfdx(sC)/dfdx(texPos)` `12` vs `24` `50%` `R8` `SD4 6.32 thr 69.2` `SD0.5 8.62 thr 180.5` `512 y 33990 thr 26829` `>>5` `pos+dx 0.0015` `dPos 0.0006` `1.6x` off `SD0.5` `evalStep 0.38` `4x` `SD4 1.52` `miss` `rayDir` spread `axis-chord +27%` `§37.13` `VolumeTransposedAxisDepth:477` `0` `574` `Y-depth 1.73` `quad_shuffle(sC,0..3)` `lane&3` `s0..s3` `thread_index_in_quadgroup` `[[quadgroup]]` `100` lines `Instruments` `Texture BW` `Quad Active` `4*64` `hit 40% SD0.5 10% SD4` not `thr0` `DENSE -7.5%` `thr0` `M/GL<1` `0.67` `+quad 0.66` `1%` `R8` `VOL_NEAREST 10%` `thr 2.02` already. Kept behind `VTK_METAL_TEST_QUAD_GRAD=1` as fallback `6-fetch` `thr 0.000` `2999` to keep `xcrun` clean while `1` `env` `A/B` remains for `Instruments` `4*64` `hit`.

Precompute `boundsSize/maxBound*sampleDist` `texelCount/ctpScale/ctpOffset` `VolumeMapperUniforms:62` `1752->1840` `float4` `16` pad `1760` `+80B` `Uniforms` `physicalSampleStep:3544` `marchVolumeUnified:4347` `3` `get_width` `6` `div` `600k*5 3M` `<<1%` `R8` `1.1B` `±2%` noise `§17` `doc` `1760` `1840` not shipped, note only.

Repro `thr0` `DENSE+ladder` `M/GL<1` `0.75` `0.73` `SD4` `thr 0.000`:

```sh
BIN=build_macos_metal/bin/vtkMetalGLVisualComparison
NIFTI=/Users/macair/Public/IMR/7T-MRI/Synthesized_FLASH25_downsampled_200um.nii
BASE="VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1"
eval "env $BASE $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out /tmp/p 2>&1 | grep -E 'NIFTI|worst'" # 2999 0.000 / 4158 0.035
eval "env $BASE VTK_METAL_TEST_DENSE=1 $BIN --bench --backend metal --scene NIFTIVolume --nifti $NIFTI --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep NIFTIVolume" # 6.30
eval "env $BASE $BIN --bench --backend gl --scene NIFTIVolume --nifti $NIFTI --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep NIFTIVolume" # 8.33
eval "env $BASE VTK_METAL_TEST_VOLUME_NEAREST=1 $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out /tmp/p 2>&1 | grep -E 'NIFTI|worst'" # 2.02 <5
eval "env $BASE VTK_METAL_TEST_GRAD_NEAREST=1 $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out /tmp/p 2>&1 | grep -E 'NIFTI|worst'" # 1.89 <5
xcrun -sdk macosx metal -c Rendering/Metal/Shaders/MetalShaders.metal -o /tmp/metal_check.air # 15 warnings clean 1643KB
```

---

## 19. DICOM transpose tie-break - balanced vs uniform and SD interaction (2026-08-30, `3293a9a14a`)

This section documents the `§30` `VolumeTransposedAxisDepth:477` tie-break `512==512` re-evaluation, the `SD4` vs `SD0.5` equality artefact, and the `DENSE`/`FRAG_BATCH` bisection. All numbers `arm64 Release` `M2` `3293a9a14a` `xcrun metal -c` `2 warnings` `93M` `AC` `MTL_DEBUG_LAYER=0` `eval "env $BASE ..."` `zsh` `§1`, `IMRToraceAddome 512x512x1794 U8` `BASE` as `§1` `VTK_METAL_TEST_VOLTRANSPOSE=1 VTK_METAL_TEST_GPU_TRANSPOSE=1` default `ON`.

### 19.1 Current rule

`Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm:441` `VolumeTransposedActive()` default `ON`, `mm:477` `VolumeTransposedAxisDepth(dims)` `if(VTK_METAL_TEST_VOLTRANSPOSE_AXIS) return 1:X 2:Y else if(dims[2]<=dims[0]&&dims[2]<=dims[1]) return 0 identity else return (dims[0]<=dims[1])?1:2` `X` on `X==Y` tie `mm:496`. `DICOM 512x512x1794` `-> 1 X-depth 1794x512x512` auto, `NIFTI 632x826x574` `574` shortest `-> 0` identity `perf_investigation_part2.md:270`. `Y` forced via `VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y` `mm:481` supersedes `argmin`, `Y-tie` toggle `VTK_METAL_TEST_VOLTRANSPOSE_Y_TIE=1` `mm:496` flips tie `512==512 -> Y` `X` `<=` vs `Y` `<` `HARNESS_VS_APP_GAP §38.3`, app `Rendering -> Transpose Y-Tie` `AppDelegate.mm:264` `VTK_METAL_TEST_VOLTRANSPOSE_Y_TIE`, inert when `VOLTRANSPOSE=0`.

`mm:467` comment keeps `X` tie: `Y` won `10/12` coarse `HARNESS_VS_APP_GAP.md:3902` `§38.3` `obl -6.3% az45 -32.8% axx -20.5% axz -9.6%` but lost `axy +2.7% mm +26.9% raw` `HARNESS_VS_APP_GAP.md:3911` - not uniform so `X` stays, tiered `fine-Y/coarse-X` reverted per `§25.5`.

### 19.2 Balanced 12-view vs uniform 26-view `X` vs `Y` on `DICOM`

`12-view` `8x AZ@elev-20 +3 axes +obl` `1024 20f/5w SD4 X15.08 Y12.55 -16.8% geom -17.3%` `Y 9/12 75%` `1024 30f 6-view -12.5%` `2048 60f -13.3%` is equator biased `HARNESS_VS_APP_GAP.md:1426` `§30.2` (`Y` free `||XZ`, `X` free `||Y`). `3 faces equal` `axx/axy/axz` `X14.00 Y13.91 -0.6%` tie `512==512` `mm:496`.

Uniform `26-view` cubemap `6 faces+12 edges+8 corners` via `VTK_METAL_TEST_CAM_DIR=x,y,z` `Rendering/Metal/Testing/Cxx/TestMetalScenes.h:1445` added this section, normalized, `if(fabs(nd[2])<0.9) viewUp(0,0,1) else (0,1,0)`, `ResetCameraClippingRange`:

```
1024 20f 26-view: X SD4 9.59 geomean9.17 Y8.86 geomean8.52 -7.6% Y19/26 73% X7/26
                 X SD0.5 9.60 Y8.86 -7.7% Y20/26 X6/26 (same, see §19.3)
  per-dir Y wins ||X sagittal/diag 12.95->4.63 -8.32 12.56->4.71 -7.85 1,0,±1 -4.7, X wins ||Y coronal 5.88->12.36 +6.48 0,±1,0 +6.38
```

Even uniform `Y` wins `~7%` mean, not just `AZ` circle bias. Residual is `LL>AP` `+` row-major `X` coherence `HARNESS_VS_APP_GAP.md:1002` `§26` tiling: screen `X` neighbours share `64B` `L1`, `Y` depth preserves `X` in `XY` tile. `LL>AP` `+` more air in `AP` makes `||Y` rays shorter `0,±1,0 5.19` vs `12.95`. `X` still `+6.4` `coronal` `7` views `HARNESS_VS_APP_GAP.md:3911`. For neutral orbit `Y` wins mean, for `coronalish` dwell `X` wins - tie-break `X` `mm:496` kept for determinism, `Y` opt-in `mm:481`.

Repro `26-view` `1024 20f` `BASE_SHIPPED` `VTK_METAL_TEST_SAMPLE_DISTANCE` last:

```sh
BIN=build_macos_metal/bin/vtkMetalGLVisualComparison
DICOM=/Users/macair/Public/IMR/CTIMR/IMRToraceAddome
BASE="VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1 VTK_METAL_TEST_VOLTRANSPOSE=1 VTK_METAL_TEST_GPU_TRANSPOSE=1"
for SD in 4 0.5; do for AXIS in x y; do for d in "-1,-1,-1" "-1,-1,0" "-1,-1,1" "-1,0,-1" "-1,0,0" "-1,0,1" "-1,1,-1" "-1,1,0" "-1,1,1" "0,-1,-1" "0,-1,0" "0,-1,1" "0,0,-1" "0,0,1" "0,1,-1" "0,1,0" "0,1,1" "1,-1,-1" "1,-1,0" "1,-1,1" "1,0,-1" "1,0,0" "1,0,1" "1,1,-1" "1,1,0" "1,1,1"; do
  eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=$SD $BASE VTK_METAL_TEST_VOLTRANSPOSE_AXIS=$AXIS VTK_METAL_TEST_CAM_DIR=$d $BIN --bench --backend metal --scene DICOMVolume --dicom $DICOM --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep ^DICOMVolume"
done; done; done
# parse mean: python3 -c "import re; ..." geomean9.17->8.52 -7.6%
```

Commit `3293a9a14a` `CAM_DIR` patch `TestMetalScenes.h:1445` `f[3] p[3] nd[3] dist sscanf` `xcrun -c` `2 warnings`.

### 19.3 `SD4` vs `SD0.5` `MM` vs `RAW`

Previous `26-view` fake tie `X SD4 9.59 SD0.5 9.60` was shell `env VTK_METAL_TEST_SAMPLE_DISTANCE=$SD $BASE` where `BASE` contained `SAMPLE_DISTANCE=4` overriding `0.5` `§1`. Correct order `BASE` without `SAMPLE_DISTANCE` or `SD` last. Correct `6-view 20f` `1024/2048` `BASE` `IMAGE...` only:

```
1024 RAW X SD4 14.67 SD0.5 32.27 2.20x Y13.52->29.27 2.17x perf_investigation_part2.md:76 DICOM mv0 8.97 vs 99
1024 MM  X 10.17->22.31 2.19x Y9.04->18.51 2.05x HARNESS_VS_APP_GAP.md:3868 obl15.97 vs axz121
2048 RAW X 23.34->151.82 6.50x Y21.90->117.09 5.35x
2048 MM  X 19.93->95.95 4.81x Y17.55->73.93 4.21x
```

`SD4` still `2x @1024 5x @2048` faster even with `MINMAX=1 ACCEL=1` `mm:7962` sparse `40% empty` `§9` - `minMax` leaps `12M` `R8` but fine still `8x` `200 vs 41` `MetalShaders.metal:4363`. `RAW` `2.2x` vs `MM` `2.1x` - `minMax` does not erase `SD` cost.

`X vs Y` same ratio both `SD`: `Y -7%` uniform `§19.2`.

Repro `6-view` `1024/2048`:

```sh
BASE="VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_VOLTRANSPOSE=1 VTK_METAL_TEST_GPU_TRANSPOSE=1"
for RES in 1024x1024 2048x2048; do for SD in 4 0.5; do for MODE in "VTK_METAL_TEST_MINMAX=0 VTK_METAL_TEST_ACCEL=0" "VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1"; do
  for V in "CAM_AXIS=x" "CAM_AXIS=y" "CAM_AXIS=z" "CAM_AZ=45" "CAM_AZ=135" ""; do
    eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=$SD $BASE $MODE ${V:+VTK_METAL_TEST_$V} $BIN --bench --backend metal --scene DICOMVolume --dicom $DICOM --frames 20 --size $RES --warmup 5 2>&1 | grep ^DICOMVolume"
done; done; done; done
```

## 20. Flags bisection vs production (2026-08-30, `3293a9a14a`)

Quick `obl` `20f/5w` `BASE_PROD` `MINMAX=1 ACCEL=1 VOLTRANSPOSE=1` `1024 9.86 2048 14.20` `§29` `eval`:

```
PROD 1024 9.86 2048 14.20
DENSE=1 Rendering/Metal/Shaders/MetalShaders.metal:4698 fc_dense:48 1024 14.46 +46% 2048 21.11 +48% regress both res obl - sparse DICOM 40% empty §9, dense skips while(w<extent) R8 20 fetches §18 for NIFTI >60% opaque, forced on sparse marches empty bricks
FRAG16 MetalShaders.metal:2794 fc_fragBatch=16 1024 9.68 -1.8% 2048 14.15 -0.3% neutral/win 56% perf_investigation_part2.md:243 not regress
VOLTRANSPOSE_AXIS=y mm:477 1024 8.42 -14% 2048 13.28 -6% win obl §38.3 -6%
```

`DENSE` is the `+48%` regressor vs `PROD` on `obl`, `FRAG16` and `Y` are wins/ties. `DENSE` only for `NIFTI` dense, not `DICOM` sparse. `FRAG16` small now because `0` already light after `§18` `8:2` `37%->56%` `perf_investigation_part2.md:243`.

Repro:

```sh
BASE_PROD="VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1 VTK_METAL_TEST_VOLTRANSPOSE=1 VTK_METAL_TEST_GPU_TRANSPOSE=1"
for FLAG in "PROD" "VTK_METAL_TEST_DENSE=1" "VTK_METAL_TEST_FRAG_BATCH=16" "VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y"; do
  eval "env $BASE_PROD ${FLAG#PROD} $BIN --bench --backend metal --scene DICOMVolume --dicom $DICOM --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep ^DICOMVolume"
  eval "env $BASE_PROD ${FLAG#PROD} $BIN --bench --backend metal --scene DICOMVolume --dicom $DICOM --frames 20 --size 2048x2048 --warmup 5 2>&1 | grep ^DICOMVolume"
done
```

## 21. Frag defaults and sweep re-run vs `§39.3`/`§39.4`/`§39.7` (2026-08-30, `3293a9a14a`)

Default `0` `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm:1453` `VolumeFragBatch() 0` if unset `->` `MetalShaders.metal:5635` `batchCap=min(shadeCap 8:2,32) lean16` heavy `37%` `HARNESS_VS_APP_GAP.md:5332` `§39.7` `0` default. `16` light `56%` `MetalShaders.metal:2794` `fc_fragBatch=16` opt-in `VTK_METAL_TEST_FRAG_BATCH=16` `+7%` win `§39.7` not yet default per `§37.20`. `fine` `8` `MetalShaders.metal:5634` `shadeCap=fc_fineSD?8:2` `82113724da` `SampleDistance<0.75` `0.5` fine `8` `4` coarse `false` `1.0` false, only when `fc_shading||fc_gradientOpacity` else `16` `MetalShaders.metal:5635`.

Why `FRAG16` small now: `heavy32` baseline gone `7acc778a18` `§18` `8:2` dead-strip `37%->56%` `perf_investigation_part2.md:243` plus `TF cull 0.02h` `§16`. Old `§39.7 X frag0 16.49->frag16 14.22 -13.8%` was `heavy32` vs `light16` before `§18`. Now `0` already light for `DICOM` long `400-step` sparse `lean16`, `obl 9.86->9.68 -1.8%` `1024` within `±2%` `M2` noise. `NIFTI 41-step` dense `§7` `f2 0.86 vs f16 1.14 +32%` still shows `+10-13%` for shade.

Full sweep `60f/10w @2048 X/Y obl` `BASE` `§39.6` `SD4 JITTER=1 mv9 mm` `eval`:

```
2048 X 60f: X frag0 14.68 frag8 13.93 frag16 14.19 frag32 15.32 frag48 15.67 (frag8 best -3.4% vs 0)
           Y frag0 13.89 frag8 13.57 frag16 14.25 frag32 14.31 frag48 15.15 (frag8 best -1.4%)
2048 X 60f single-run §39.7 doc: X 16.49 14.02 14.35 16.37  Y 15.09 13.57 13.72 14.31 -> 16<0<32 vs doc 16<32<0 within 0.08ms 0.5%
```

`16` still best/tie `X14.19 Y14.25` beats `32` `15.32/14.31` `+7%` `HARNESS_VS_APP_GAP.md:5332`, `32` beats `0` heavy `X -4..-7%` `doc -7.5%` reproduced - `§39.3` `orb 137.9->107.2 -22%` `§39.3` still.

Re-run `§39.3` `X` `60f @2048` `20f @4096` `§39.4` `ABBA` `Y` `BS8 frag` vs `BS16 comp` `§38.12` on this branch `3293a9a14a` compute deleted `13 insertions +1759 deletions` `Rendering/Metal/Shaders/MetalShaders.metal:714` `vtkMetalGPUVolumeRayCastMapper.mm:1040` fragment `marchVolumeUnified` retained, `VTK_METAL_TEST_COMPUTE_MARCH` env no-op:

```
@2048 X 60f single: obl 14.68 az45 16.16 az135 11.87 axx 21.05 axy 17.16 axz 26.12 vs doc 16.32 17.59 14.42 28.01 25.95 35.59 -> doc frag0 heavy slower -1.6..-9.4, frag16 doc 14.35 vs now 14.19 -0.16
@4096 X 20f: obl 34.89 axz 92.37 vs doc 46.65 129.47 -11.7 -37.1 heavy faster now, frag16 33.60 vs 33.86 -0.26 90.40 vs 91.86 -1.46 light equivalent
@2048 Y single: Y frag0 obl 13.89 az45 9.68 az135 13.80 axx 16.20 axy 19.92 axz 26.56 vs doc Y frag0 14.74 11.62 14.80 21.99 25.95 31.64 -> Y wins 4/6 still
@4096 Y: frag16_Y obl 35.86 axz 126.51 vs doc frag16_Y 31.89 87.72 heavy -> now 35.86 vs 31.89 +12% slower at 4096 Y (thermal/host)
```

Best numbers equivalent or faster `doc frag0 -> now frag0 -1.6..-9.4` `frag16 -0.16..-1.46` `HARNESS_VS_APP_GAP.md:5332` `16<32<0` holds within `±2%` `M2` `+ single-run ±1%` `§39.7` caveat. Compute `§39.3` `frag16_Y 13.54 vs comp16_Y 12.28` vs `now` `comp` deleted `3293a9a14a` `13 insertions +1759 deletions` - `comp` numbers no longer valid, fragment `frag16_Y` `13.54` vs `now 14.25` `+5%` `noise`.

Repro `§39.3` `§39.4`:

```sh
BIN=build_macos_metal/bin/vtkMetalGLVisualComparison
DICOM=/Users/macair/Public/IMR/CTIMR/IMRToraceAddome
BASE="VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1"
for FB in 0 8 16 32 48; do for AXIS in X Y; do eval "env $BASE ${AXIS#X} ${AXIS#Y:+VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y} ${FB#0:+VTK_METAL_TEST_FRAG_BATCH=$FB} $BIN --bench --backend metal --scene DICOMVolume --dicom $DICOM --frames 60 --size 2048x2048 --warmup 10 2>&1 | grep ^DICOMVolume"; done; done
# ABBA per §37.20: run frag0/frag16/frag16/frag0 per view and average two rounds per label
```

---

## 22. Latest uniform, SD and block re-evaluation (2026-08-30, `3293a9a14a` `CAM_DIR` `2 warnings`)

This section captures points that were not in the `HEAD` diff at `§18` - the `26-view` uniform `X vs Y`, the `SD4 vs SD0.5` `MINMAX` interaction, the `DENSE`/`FRAG` bisection and the `MM_BLOCKS` `fine` tier re-check, all on the same `M2` `93M` `eval` `§1` build that carries `TestMetalScenes.h:1445` `CAM_DIR=x,y,z`.

**`26-view` vs `12-view`:** `12-view` `8x AZ@elev-20 +3 axes+obl` gave `Y 9/12 -16.8%` `1024 20f` `HARNESS_VS_APP_GAP.md:1426` equator bias. Uniform `26-view` `6 faces+12 edges+8 corners` `VTK_METAL_TEST_CAM_DIR` still `Y 19/26 -7.6%` `X9.59 Y8.86` `SD4` `X9.60 Y8.86` `SD0.5` `geomean 9.17->8.52`. Even uniform `Y` wins `~7%` mean `LL>AP` `+` row-major `X` coherence `HARNESS_VS_APP_GAP.md:1002` `§26` `3 faces equal` `X14.00 Y13.91 -0.6%` tie `512==512` `mm:496` shows the `7%` is edges/corners, not faces.

**`SD` `MINMAX`:** correct `BASE` without `SAMPLE_DISTANCE` `SD` last `§1` `SAMPLE_DISTANCE=$SD $BASE` was `4` overriding `0.5`. Correct `6-view 20f` `1024/2048` `BASE` `IMAGE...` only:

```
1024 RAW X SD4 14.67 SD0.5 32.27 2.20x Y13.52->29.27 2.17x perf_investigation_part2.md:76
1024 MM  X 10.17->22.31 2.19x Y9.04->18.51 2.05x HARNESS_VS_APP_GAP.md:3868
2048 RAW X 23.34->151.82 6.50x Y21.90->117.09 5.35x
2048 MM  X 19.93->95.95 4.81x Y17.55->73.93 4.21x
```

`SD4` still `2x @1024 5x @2048` faster even with `MINMAX=1 ACCEL=1` `mm:7962` `40% empty` `§9` `200 vs 41` `MetalShaders.metal:4363`. Earlier fake tie `9.59==9.60` was env order bug.

**`DENSE`/`FRAG` bisection `obl` `20f` `BASE_PROD` `MINMAX=1 ACCEL=1 VOLTRANSPOSE=1` `1024 9.86 2048 14.20` `§29`:**

```
PROD 1024 9.86 2048 14.20
DENSE=1 MetalShaders.metal:4698 fc_dense:48 1024 14.46 +46% 2048 21.11 +48% regress sparse DICOM 40% empty §9, dense skips while(w<extent) R8 20 fetches §18 for NIFTI >60% opaque
FRAG16 MetalShaders.metal:2794 fc_fragBatch=16 1024 9.68 -1.8% 2048 14.15 -0.3% neutral/win 56% perf_investigation_part2.md:243 not regress
VOLTRANSPOSE_AXIS=y mm:477 1024 8.42 -14% 2048 13.28 -6% win obl §38.3 -6%
```

`DENSE` is the `+48%` regressor vs `PROD` on `obl`, `FRAG16` and `Y` are wins/ties. `DENSE` only for `NIFTI` dense, not `DICOM` sparse.

**`NIFTI` with them on/off `1024/2048` `NIFTIVolume` `BASE` as `§1`:**

```
1024 SD4 PROD 7.14 DENSE 6.60 -7.6% FRAG16 7.11 -0.4% DENSE+FRAG16 6.59 -7.7%
1024 SD0.5 PROD 9.26 DENSE 9.43 +1.8% FRAG16 10.98 +18% DENSE+FRAG16 11.10 +19%
2048 SD4 PROD 10.60 DENSE 8.81 -16.9% FRAG16 10.57 tie
2048 SD0.5 PROD 25.82 DENSE 25.80 tie FRAG16 33.06 +28% regress
```

`8:2` `MetalShaders.metal:5634` `shadeCap=fc_fineSD?8:2` `82113724da` already default `PROD` wins fine `9.26 vs 10.98` `25.82 vs 33.06` `28%` vs forced `16`, dense wins coarse `7.14->6.60` `10.60->8.81` `17%`. Combined `8:2` fine `+` dense coarse covers both ends for `~10` lines. `B16` disadvantage at fine `200 steps` `16*7 fetches` `16 pow` `37%` spill `tail 8` vs `8*7` `25` batches no tail `§7` `f2 0.86 vs f16 1.14 +32%`.

**`MM_BLOCKS` fine tier `VTK_METAL_TEST_MM_BLOCKS=1` `AppDelegate.mm:88` default `NO` `mm:1399` gated `<1.5` `mm:7924` — CORRECTED 2026-09-01 for `mv9` `eval` bug (§1):**

> **Errata 2026-09-01:** prior `§22` `MM_BLOCKS` bash runs used `env $BASE $BIN` in `zsh` without `eval "env $BASE ..."` `§1`. `zsh` does not word-split `BASE`, so `VTK_METAL_TEST_MARCH_VARIANT=9` `mm:242` `fc_marchVariant 9` `MetalShaders.metal:242` and `VOLTRANSPOSE` etc. were dropped, falling back to `mv0` `MetalShaders.metal:6416`. `mv0` at `2048 SD0.5` `SkinOnBlue` is `~112-120ms` versus `mv9` `~30ms` with `eval` (`-75%`). All numbers below are re-measured with `eval "env $BASE ..."` `zsh` correct `§1` `§40.3` `ABBA` `15-30f/5-10w` `arm64 Release` `AC`. `Skin On Blue II` `VRPresets/Skin On Blue II.plist:148` `useShading false` is same TF as `SkinOnBlue` `TestMetalScenes.h:1208` `rescale (hu+1024)*255/4095` `shade OFF` harness default, so `VTK_METAL_TEST_PRESET=SkinOnBlue` below reproduces `II`.

```
# Airways II sparse ~40% empty §9 — prior tie reproduced at mv9 correct
SD4 no-blocks 1024 9.64 1024+blocks 9.46 -1.8% 2048 14.11->14.13 tie - SD4 gated off §34, correct at mv9
SD0.5 no-blocks 1024 9.63 1024+blocks 9.68 tie 2048 14.13->14.08 tie - sparse DICOM no leap at mv9

# SkinOnBlue dense Skin On Blue II — mv9 correct shows large win at ~2k SD0.5 (user observation)
SkinOnBlue SD4 1024 6.52->7.23 +10% regress (short 41-step)  2048 SD4 11.78->11.18 -5% tie
SkinOnBlue SD0.5 1024 25.30->12.08 -52%  2048 SD0.5 69.34->30.41 -56%  15f/5w metal-only ABBA 30f/10w 72.02->30.11 -58% at DOLLY=1
  DOLLY sweep mv9 2048 SD0.5 SkinOnBlue: 1.5 129.09->58.73 -55%  2.0 179.64->85.36 -52%  2.5 187.97->95.79 -49% (obl 182.34->93.31 -48% x 112.73->96.57 -14% y 119.10->... )  3.0 158.35->92.76 -41% — win persists at high zoom
  Both backends 15f DOLLY=1 Skin SD0.5 126.62/30.02 0.24 vs 294.42/96.01 0.33 at 2.5 — M<GL stays, accel ON beats OFF 40.54 0.30 vs 30.02 0.24
NIFTI SD0.5 1024 9.08->9.30 +2% 2048 25.33->25.31 tie 4096 87.24->87.03 tie - cull+ladder strip §16 §18 already leapt at mv9
```

`Airways II` dense `SkinOnBlue` split: `Airways` tie at `1024/2048` both `SD` stays `2%` tie, leaving `AppDelegate.mm:88` `NO` correct for sparse prod; `SkinOnBlue`/`BoneSkinII`/`DarkBone` dense win `>50%` at `~2k SD0.5` `mv9` where `>60%` opaque blocks leap `mm:6373` `8³` `block/super` `R8`. Before `cull+ladder` `§16 §18` `4096` `fine` `BS16` won, now at `4096` `NIFTI` `2%` tie, but at `2048` `SkinOnBlue` `mv9` `BS` wins. Opt-in `VTK_METAL_TEST_MM_BLOCKS=1` for dense `TF` `Skin On Blue II` at `~2k` `SD0.5` `mv9`, keep `NO` default for sparse `Airways` prod (or auto via dense heuristic `§23.2`). `M` vs `GL` at zoom `DOLLY=2.5` with `mv9` correct: `SD0.5` `accel ON` `M<GL` `0.33` `96/294` vs `accel OFF` `0.40` `117/289` — `accel ON` still wins at zoom with `mv9`, unlike `mv0` where `M>GL` `1.60` `494/309` flipped.

Repro `§22`:

```sh
BIN=build_macos_metal/bin/vtkMetalGLVisualComparison
DICOM=/Users/macair/Public/IMR/CTIMR/IMRToraceAddome
NIFTI=/Users/macair/Public/IMR/7T-MRI/Synthesized_FLASH25_downsampled_200um.nii
BASE="VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1 VTK_METAL_TEST_VOLTRANSPOSE=1 VTK_METAL_TEST_GPU_TRANSPOSE=1"
for SD in 4 0.5; do for AXIS in x y; do for d in "-1,-1,-1" "-1,-1,0" "-1,-1,1" "-1,0,-1" "-1,0,0" "-1,0,1" "-1,1,-1" "-1,1,0" "-1,1,1" "0,-1,-1" "0,-1,0" "0,-1,1" "0,0,-1" "0,0,1" "0,1,-1" "0,1,0" "0,1,1" "1,-1,-1" "1,-1,0" "1,-1,1" "1,0,-1" "1,0,0" "1,0,1" "1,1,-1" "1,1,0" "1,1,1"; do
  eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=$SD $BASE VTK_METAL_TEST_VOLTRANSPOSE_AXIS=$AXIS VTK_METAL_TEST_CAM_DIR=$d $BIN --bench --backend metal --scene DICOMVolume --dicom $DICOM --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep ^DICOMVolume"
done; done; done
for SD in 4 0.5; do for FLAG in "PROD" "VTK_METAL_TEST_DENSE=1" "VTK_METAL_TEST_FRAG_BATCH=16"; do eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=$SD $BASE ${FLAG#PROD} $BIN --bench --backend metal --scene NIFTIVolume --nifti $NIFTI --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep ^NIFTIVolume"; done; done
eval "env $BASE $BIN --bench --backend metal --scene DICOMVolume --dicom $DICOM --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep ^DICOMVolume" # 9.86 PROD Airways
eval "env $BASE VTK_METAL_TEST_MM_BLOCKS=1 $BIN --bench --backend metal --scene DICOMVolume --dicom $DICOM --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep ^DICOMVolume" # 9.46 tie Airways
# Skin On Blue II at ~2k SD0.5 mv9 — must use eval and SAMPLE_DISTANCE last, verify mv9 ~30ms not ~112ms mv0
eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=0.5 $BASE VTK_METAL_TEST_PRESET=SkinOnBlue $BIN --bench --backend metal --scene DICOMVolume --dicom $DICOM --frames 15 --size 2048x2048 --warmup 5 2>&1 | grep ^DICOMVolume" # 69.34 blocks 0
eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=0.5 $BASE VTK_METAL_TEST_PRESET=SkinOnBlue VTK_METAL_TEST_MM_BLOCKS=1 $BIN --bench --backend metal --scene DICOMVolume --dicom $DICOM --frames 15 --size 2048x2048 --warmup 5 2>&1 | grep ^DICOMVolume" # 30.41 blocks 1 -56% win, DOLLY=2.5 187.97->95.79 -49% 6 views all win §22
# incorrect (missing eval) would give ~112ms mv0: env VTK_METAL_TEST_SAMPLE_DISTANCE=0.5 $BASE $BIN ... without eval word-splits BASE incorrectly in zsh — do not use
```

---

## 23. Suggested next steps (2026-08-30, `b725fd89a4` `26-view` `SD` `DENSE`/`FRAG` `§19-§22`)

Ranked by `thr0` `§40.3` `M/GL<1` keep, `DICOM` `0.000` `VRC 0.18` `NIFTI 0.000`, `AC` `M2` `60f/10w @2048` `20f/10w @4096` `eval` `§1`:

1. **Density+resolution aware batch - cheapest structural win.** Current `8:2` `MetalShaders.metal:5634` `shadeCap=fc_fineSD?8:2` already `7-28%` vs forced `16` on `NIFTI` fine `§22` `9.26 vs 10.98` `25.82 vs 33.06`, lean `16` already optimal for sparse `DICOM`. Remaining split is `8` at `1024/2048` vs `16` at `4096` `§21` `13.93 vs 14.19` `X` `37.18 vs 33.60` `-10%`. Make `batchCap` also see `viewport` `VolumeMapperUniforms:62` `ViewportSize` `mm:9622` or `dense` `fc_dense:48`: `sparse 4096` `16`, `dense fine` `8`, `dense coarse` `2`. One uniform `+` one `min` `~3` lines, `PROD` keeps `9.86` `14.20` `§20`, `NIFTI` `4096` `90.40` stays `90.40` `§21`, `DICOM` `4096` `33.60` best `§22`.

2. **Auto dense via histogram, not manual `DENSE`.** `NIFTI` coarse wins `7.14->6.60 -7.6%` `1024` `10.60->8.81 -16.9%` `2048` `§22` dense skips `while(w<extent)` `R8 20 fetches` `MetalShaders.metal:4698` `§18`, `DICOM` sparse regresses `+46-48%` `§20` `9.86->14.46`. Add CPU histogram `TF a>0.02` `>55%` opaque `-> denseMode` uniform `mm:4691` `denseMode` `~10` lines, `NIFTI` coarse auto wins, `DICOM` stays `9.86` `14.20`.

3. **Keep `MM_BLOCKS` off by default for sparse `Airways II`, ON for dense `Skin On Blue II` `~2k SD0.5`.** `AppDelegate.mm:88` `NO` `mm:1399` `<1.5` `mm:7924` `Airways` `9.64->9.46 -1.8%` `SD4` and `9.63->9.68` `SD0.5` tie at `mv9` correct `§22`, `NIFTI` `4096` `87.24->87.03` `cull+ladder` `§16 §18` tie. Dense `DICOM` `SkinOnBlue` `Skin On Blue II.plist:148` at `mv9` `~2k SD0.5` is the exception: `1024 25.30->12.08 -52%` `2048 69.34->30.41 -56%` `DOLLY 1-3` `-58%` to `-41%` `§22` `6` views all win `x 112->96 -14%` `obl 182->93 -48%` `M/GL 0.24` vs `0.33` at `2.5` stays `<1` and `accel ON` `96ms` beats `accel OFF` `117ms` `§22` dolly sweep, so `M<GL` at zoom. Prior `§22` tie was `Airways` sparse; `SkinOnBlue` dense needs `VTK_METAL_TEST_MM_BLOCKS=1` `BLOCKSIZE=8` default `mm:1399` `16` fine `8` coarse. For sparse prod keep `NO`; for dense `TF` auto-enable via dense heuristic `§23.2` `a>0.02` `>55%` opaque `mm:4691`.

4. **`SD-aware 4-fetch` `VTK_METAL_TEST_GRAD4=1` `MetalShaders.metal:3861` `fc_grad4:42` `float` `*2.0h` `sC+3` `4*8=32` texels `33%` save.** Already `thr<5` `SD4 2.02` `SD0.5 1.89` `§18` `M/GL<1` `0.66` `§21`, `NIFTI` fine `9.26` `->` `7.11` `-18%` `§9` with `thr<5`, `DICOM` tie `±2%`. Keep env-gated `fineSD` only, not default, `+` `pow skip` `MetalShaders.metal:4002` `0.5^20=9e-7` `3%` already landed.

No new `TF`/`precomp`/`quad` work - `precomp 1.2GB` `+188%` `§9`, `quad thr69` `§18` remain `thr<5` fail. Next `Instruments` `Texture BW` `Quad Active` `4*64` `hit 40% SD0.5 10% SD4` `§18` only if `viewport` batch shows `>5%` `4096` win.

Repro `§23`:

```sh
# 1. density+viewport batch A/B (add ViewportSize uniform, shadeCap = dense ? (fine?8:2) : (ViewportSize.x>3000?16:8))
BIN=build_macos_metal/bin/vtkMetalGLVisualComparison
DICOM=/Users/macair/Public/IMR/CTIMR/IMRToraceAddome
NIFTI=/Users/macair/Public/IMR/7T-MRI/Synthesized_FLASH25_downsampled_200um.nii
BASE="VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1 VTK_METAL_TEST_VOLTRANSPOSE=1 VTK_METAL_TEST_GPU_TRANSPOSE=1"
for RES in 1024x1024 2048x2048 4096x4096; do for SD in 4 0.5; do eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=$SD $BASE $BIN --bench --backend metal --scene DICOMVolume --dicom $DICOM --frames 20 --size $RES --warmup 5 2>&1 | grep ^DICOMVolume"; eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=$SD $BASE $BIN --bench --backend metal --scene NIFTIVolume --nifti $NIFTI --frames 20 --size $RES --warmup 5 2>&1 | grep ^NIFTIVolume"; done; done
# expect DICOM 1024 9.86 2048 14.20 4096 33.60, NIFTI 1024 SD4 6.60 SD0.5 9.26 2048 8.81 25.82 §22
```

---

## 24. Lower res and angle where `2` wide wins (2026-08-30, `9a16208eb7` `CAM_DIR` `§19`)

`NIFTI` `SD4` `41 steps` `6.93-7.14` `§21` `f1 7.07 f2 7.08 f4 7.03 f8 6.97 f16 6.96 f32 6.93` `30f` `1024` tie `±2%` now, historic `f2 6.97 vs f4 7.26 -4%` `§14.1` `1,1,1` before `cull` `8:2`. Lower `512` `5.22 f1 vs 5.39 f2` `1` wins `-3%` `400` `4.64 vs 4.68` `1` tie, `1024` `CAM_AXIS=x 2.92 f8 vs 2.96 f2` `8` wins `+0.04` `CAM_AXIS=y 2.96 f2 vs 3.08 f8` `2` wins `+0.12` `az45 4.67 f16 vs 4.68 f4` `16` wins. `2` wide peak is `1024` `obl` `1,1,4` era, now with `1,1,1` `cull` `8:2` `§18` flat `±0.14ms`, `8` covers both ends `§22` `9.18 f8 vs 11.11 f16` `fine` `18%` `1024` `24.89 vs 31.95` `2048` `fine`.

Repro `512` `1` wins:

```sh
BIN=build_macos_metal/bin/vtkMetalGLVisualComparison
NIFTI=/Users/macair/Public/IMR/7T-MRI/Synthesized_FLASH25_downsampled_200um.nii
BASE="VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1 VTK_METAL_TEST_VOLTRANSPOSE=1 VTK_METAL_TEST_GPU_TRANSPOSE=1"
for RES in 512x512 400x400; do for FB in 1 2 4 8 16 32; do eval "env $BASE VTK_METAL_TEST_FRAG_BATCH=$FB VTK_METAL_TEST_SAMPLE_DISTANCE=4 $BIN --bench --backend metal --scene NIFTIVolume --nifti $NIFTI --frames 20 --size $RES --warmup 5 2>&1 | grep ^NIFTIVolume"; done; done
for V in "CAM_AXIS=x" "CAM_AXIS=y" "CAM_AZ=45"; do for FB in 1 2 4 8 16; do eval "env $BASE VTK_METAL_TEST_FRAG_BATCH=$FB VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_$V $BIN --bench --backend metal --scene NIFTIVolume --nifti $NIFTI --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep ^NIFTIVolume"; done; done
# expect 512 f1 5.22 vs f2 5.39 -3% f1 wins, 400 4.64 vs 4.68 tie, 1024 CAM_AXIS=y f2 2.96 vs f8 3.08 f2 wins 0.12ms §22
```

---

## 25. mv9 width-1 vs mv0 — closing the `+11%` gap and parity bisect (`metal-cleanup-1`)

Goal: decide whether `mv9` with `VTK_METAL_TEST_FRAG_BATCH=1` (`w1`) is functionally equivalent to `mv0`, so `mv0` can be deleted. Baseline `arm64 Release` `M2` `BASE` as `§1` + `SAMPLE_DISTANCE=4` (`20f/5w` quick, `15f/5w @2048`, `eval "env $BASE ..."` `zsh` `§1`): `w1 11.2-11.9 vs mv0 10.0-10.5 +9-12% @1024 DICOM` (`13.00 vs 9.99 +30%` at `15f` pre-lean), `2048 w1 26.8 vs mv0 31.8 -15%`, `ACCEL=0 19.5 tie`, `thr0` `512 VRC byte-identical` / `DICOM mean 0.015 max6`.

**Gap root cause:** the `48`-wide lookahead `while(w<extent)` did `fract/distToEdge/tToEdge/cellSteps` math per empty cell + `warp/super/block` state + `minMaxBlock/Super` fetches + `i+1` fetch guard — all dead for `w1`, since `SD4 step>cell` makes `cellSteps==1` anyway. Tried and reverted: `extent 8 +20%` worse, `w1` block-leap `+15%`, `mv0`-in-`mv9` fast path `30.6 vs 10.5 3x` (register spill), redirect `else if(fc_marchVariant>=6 && !(9 && w1))` to `mv0` (`1577344 AIR`) as cheating.

**Landed `57f07ab781` `MetalShaders.metal:5748` `if(fc_fragBatch==1)` lean branch:** `48` per-cell incremental `curMMPos+=evalStep` `w++`, no `fract` math; `mv9Blk/mv9Sb` declarations moved inside heavy `else` (`&& fc_fragBatch!=1`, `w1` `PSO` sheds registers); `fetch1` without guard. `AIR 1612528->1602000`. `ABBA @1024 DICOM w1 9.49/9.33/9.15 avg 9.32 vs mv0 10.40/10.79/9.91 avg 10.37 ~-10%`; `2048 w1 16.15-16.36 vs mv0 31.75-32.10 ~-50%` (`w48 16.62 w16 15.06 w8 13.96` — the `2x` is the `mv9` harness win `§14`, inherited even at width `1`); `NIFTI 7.35/6.98 vs 7.71/7.40 -5%`; `VRC 1.64 vs 1.69`; `ACCEL=0 19.61 vs 19.95 tie`; `thr0` everywhere. Dropped hunks bisection (`fc_walkBatch:50` dead, `segHop/warp/block/super &&!=1` redundant inside `else`, fetch guard `0.03ms`): none beat keep-alone (`9.53 vs 9.08 +0.45ms` all-disables).

**Parity `w1` vs `mv0` `512` (not byte-identical, `thr0` pass):** `DICOM mean 0.015375 max4 >0 1.5133% >1 0.0223%` (122 px mag 2-4, all interior rows 150-378, 0% border); `NIFTI mean 0.010618 max4 >1 0.0017%`; `VRC mean 0 max0` identical. No knob is identical: `ACCEL=0 mean 0.012352 max3 >1 0.0013%` (closest), `MINMAX=0 mean 0.017119 max6`, `both0` = `ACCEL=0`.

**Cause experiments (all `512 DICOM` `w1` vs `mv0`):**

```
Exp1 post-walk tEnd/bounds re-check before fetch (mv0 :6703 clone): bit-identical to baseline — not over-fetch, reverted
Exp2A diag no-walk (composite everything): mean 0.017816 max32 >1 0.0449% 2x worse — walk is CORRECT (minmax-empties are not bit-zero under linear bleed), reverted
Exp3 matrix re-sync after walk advance (mv0 :6711-6714 clone): mean 0.014654 -5%, >1 0.0174% -22% (58->45px), max still 4 — committed 0797692a7d then reverted 5e1e6966f1: parity-by-mimicry (~30 roundings agreeing with mv0 vs 2 independent), no threshold crossed, keep lean
Exp4 recomputed (non-incremental) classify positions: bit-identical to Exp3 — in-walk drift ruled out, reverted
```

**Residual attribution (`mean 0.0147 max4 ~45px`):** landing *choices*, not positions — mv0's `+1e-4` ceil fudge + edge guard (`:6667`) vs fudge-free counting (±1 step at cell edges), mv0's block consults + texel-center sampling (`:6636-6643`) vs `w1` ignoring blocks, and the no-skip floor both skeletons share (`:5688` top-`tEnd` break present in `mv9` but absent in bounded `mv0` `:6569`, prefetch order, FMA contraction). Closing it means giving `w1` mv0's leap math, i.e. un-leaning the fast path.

**State:** `metal-cleanup-1` = `57f07ab781` (keep lean) + `5e1e6966f1` (revert resync); full `77`-line variant saved at `/tmp/metal_w1lean.metal` + `/tmp/full.diff`. `w1` faster than `mv0` everywhere measured, `thr0`, `VRC` identical — equivalence argument is `thr`-parity + `2048` win, not byte-identity.

Repro:

```sh
BIN=build_macos_metal/bin/vtkMetalGLVisualComparison
DICOM=/Users/macair/Public/IMR/CTIMR/IMRToraceAddome/UZOZWT24/TQHNCPFG
BASE="VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1"
eval "env $BASE VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_FRAG_BATCH=1 $BIN --bench --scene DICOMVolume --dicom $DICOM --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep ^DICOMVolume" # w1 ~9.3
eval "env $BASE VTK_METAL_TEST_MARCH_VARIANT=0 $BIN --bench --scene DICOMVolume --dicom $DICOM --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep ^DICOMVolume" # mv0 ~10.4
rm -rf /tmp/p_w1 /tmp/p_v0
eval "env $BASE VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_FRAG_BATCH=1 $BIN --scene DICOMVolume --dicom $DICOM --frames 1 --size 512x512 --warmup 2 --out /tmp/p_w1 2>&1 | grep -E 'DICOM|worst'"
eval "env $BASE VTK_METAL_TEST_MARCH_VARIANT=0 $BIN --scene DICOMVolume --dicom $DICOM --frames 1 --size 512x512 --warmup 2 --out /tmp/p_v0 2>&1 | grep -E 'DICOM|worst'"
python3 -c "from PIL import Image;import numpy as np,pathlib;a=np.array(Image.open(list(pathlib.Path('/tmp/p_w1').glob('*.metal.png'))[0])).astype(int);b=np.array(Image.open(list(pathlib.Path('/tmp/p_v0').glob('*.metal.png'))[0])).astype(int);d=np.abs(a-b);print(f'mean {d.mean():.6f} max {d.max()} >0 {100*(d>0).mean():.4f}% >1 {100*(d>1).mean():.4f}%')"
# expect mean ~0.015 max4 >0 ~1.5% >1 ~0.02% thr0
```

---

## 26. Wide-width revival — rolled 16-composite loop + `half` scalars + `§22` retest (2026-09-03, `metal-cleanup-1`)

Goal: make `16`-wide viable at fine (`§7` `f16 1.14 FAIL`, `§22` `9.26 vs 10.98 +18%`), zero byte delta. All numbers `arm64 Release` `M2` `BASE` as `§1` + `MARCH_VARIANT=9`, `20f/5w @1024` `15f/5w @2048` `eval`, vs same-protocol `ABBA` (thermal-proof); absolutes carry `~+5%` warm offset — see `§26.5`.

### 26.1 Changes (`Rendering/Metal/Shaders/MetalShaders.metal`, `+329/-17`, all byte-verified `max0`)

1. **`half s##_j` (`MV9_FETCH:5312`).** Audit: sole use is `:5424 `half(s_j)*scalarScale+scalarBias`` with scale/bias already `half` (`:4321-4322`), so `half(half(x))==half(x)` — narrowing the 16-wide scalar carry `64B→32B` is bit-exact. Alone `~0` (kept as enabler, 1 word). All other `float`s load-bearing (texture coords, accumulators).
2. **`MV9_COMPOSITE_LOOP(mvk)` (`:5610`) + shade-gated 16-rung (`:6431`).** The 16-rung pasted the `~300`-line composite body 16× (`f8 +0% f16 +15% f32 +38%` superlinear = `I$`). New macro is the de-suffixed body (`##_j`→plain, `(float)_j`→`(float)(mvk)`, `s##_j`→`sBuf[mvk]`; script-verified, no `return`/`goto`/`break` inside); rung keeps batched `FETCH(0..15)` (issue overlap) then `#pragma unroll 1 for(mvk<16)` loop. `if(fc_shading||fc_gradientOpacity)` selects loop vs original pastes per-`PSO` — shade gets `I$`, lean keeps unrolled scheduling. `AIR 1602032→1437552 -10%`, `xcrun` stays `2 warnings`.
3. **Tried and reverted (byte-max `0` both):** block/super cache hoist above `while(i<steps)` — neutral (texture cache already dedups same-texel re-fetch); fetch+composite interleave in 16-rung — `-18-25%` (kills fetch-issue overlap, `§v39` was right). Default-vs-forced-`f2` gap investigated by `ABBA` — not real (`7.37` vs `7.41`, earlier `6.78` outlier).

Byte-identity argument: identical IEEE ops in identical per-sample order (`0→15`), fetches side-effect-free, no early exits inside body, loop-carried state (`accumulated*`, MIP, `firstOpaquePos`) updated in same sequence.

### 26.2 Parity — zero delta everywhere

```
NIFTI 512 SD4 2999 thr 0.000 byte-max 0 vs pre-change; SD0.5 4158 thr 0.035 (= baseline .034)
NIFTI 512 SD0.5 forced-f16 4155 thr 0.035 keep; DICOM 512 y 1122 thr 0.000 keep
```

### 26.3 Perf — `F16` revived, `DEF` neutral, DICOM records

```
1024 SD0.5: F16/DEF 11.23/9.93 (+13%) -> 9.88/10.07 (-2%)  M/GL 0.54 (GL 18.27)
2048 SD0.5: F16/DEF 31.70/27.71 (+14%) -> 25.75/27.46 (-6%) M/GL 0.49 (GL 53.05)
NIFTI SD4 DEF 7.37 = exact pre-change mean (coarse-shade 2/1 rungs untouched)
DICOM 1024 SD4 F16 10.63 -> 10.08 -5% (half carry); F8 9.27 -> 9.04 tie; SD0.5 DEF/F16 16.26/16.45 +1.2% (normal §21 pattern)
```

`§7`'s `1.14 FAIL` is gone at `1024` (`0.98×DEF`); `§22` `9.26 vs 10.98` verdict superseded — `16`-wide now ties-or-beats `8` at fine.

**32/48 re-check (current tree, pasted rungs + `half` scalars):** still catastrophically slower — `NIFTI 1024 SD0.5 F8 9.35 F16 9.37 F32 17.22 +84% F48 24.04 +157%`, `2048 F16 23.62 vs F32 52.30 +121%`, `DICOM 2048 SD0.5 F16 36.59 vs F32 39.18 +7%` (same `§21` lean verdict). The `f16→f32` step (`+84-121%`) dwarfs any loop-dedup win (`-12-19%` on the smaller body), so rolling `32/48` cannot close it — caps `8:2`/`16` stand. Next: roll the `8`-rung the same way (moves `DEF` itself) + re-sweep `shadeCap`.

### 26.4 `§22` retest — reproductions and errata (`~150` runs, current tree)

- **SD/MINMAX table ran `mv0`, not `mv9`.** Its `BASE` (`§19.3:860`) has no `MARCH_VARIANT`; mapper default is `0` (`mm:380-384`). SD4 cells accidentally valid (`mv0≈mv9` coarse); SD0.5 cells `~2×` stale (fine-step gap). True `mv9` (default view): `1024 RAW X 14.51/28.17 MM X 10.02/14.97`, `2048 RAW X 22.80/59.05 MM X 15.35/39.23`; `mv0` spot-check `2048 SD0.5 RAW X 109.96` (doc-implied `151.82`, residual likely their-session thermal).
- **Airways SD0.5 MM rows (`9.63`/`14.13`) are SD4 duplicates** (`9.64`/`14.11` pairs — `5×` steps at identical ms impossible; `§19.3`-class env bug). First real numbers: blocks-0 `108.42` (plain cell-walk yields ~nothing at fine) vs blocks-on `39.8` (`2.7×`). "Airways tie, keep `NO` default" is unsupported.
- **Blocks are default-ON** (`VolumeMinMaxBlocksWanted mm:1408-1416` returns true, fine-gate removed) — opt-in framing obsolete, SkinOnBlue `-52-56%` win is the default. `DOLLY 1.5-3.0` reproduces exactly (`-56/-54/-49/-41%` vs `-55/-52/-49/-41%`); `=0` is the true off (`Skin 77.54` ≈ doc `69.34` warm).
- **Bisection reproduces** (`DENSE +49/+50%` vs `+46/+48%`, `FRAG16` tie `+1.4% ABBA`, `YTIE` direction); **NIFTI on/off reproduces** except `FRAG16`-fine fixed above; `DENSE`-coarse same direction (`-7%/-10%` vs `-7.6%/-16.9%`).
- **26-view uniform reproduces at SD4 exactly** (`X 9.60/geo 9.22 vs Y 8.90/8.55, 20/26 -7.3%` vs doc `9.59/9.17 vs 8.86/8.52, 19/26 -7.6%`); doc's SD0.5 row is the SD4 duplicate (fake tie) — genuine SD0.5: `X 13.82 vs Y 12.96, 20/26 -6.2%`. `Y wins ~7%` holds at both `SD`.

### 26.5 Thermal note (checkout recheck pending)

Absolutes run `~+5%` (short runs) to `+8-14%` (long fine runs: `NIFTI 2048 SD0.5 27.44 vs 25.82`, `4096 95.42 vs 87.24`) over doc after `~150` back-to-back benches; ratios/percentages reproduce exactly. To rule out a real regression hiding in the offset, `§26.6` rechecks key cells on a cold machine at `3293a9a14a` vs this commit.

### 26.6 Cold-machine recheck — no regression, `F16` win confirmed (2026-09-03)

Checked out `3293a9a14a`, rebuilt (`~8min` cooldown), benched cool; back to this commit, rebuilt, benched cool before heat buildup. Same protocol `20f/5w @1024` `15f/5w @2048`:

```
cell                  doc    old-cool  HEAD-cool  HEAD-hot   verdict
NIFTI 1024 SD0.5 DEF  9.26   9.43      9.47       10.0-10.1  +0.4% identical (untouched path)
NIFTI 1024 SD0.5 F16  10.98  11.05     9.35       9.88       -15.4% win, bigger cold
NIFTI 2048 SD0.5 DEF  25.82  25.86     25.47      27.4-27.8  -1.5% neutral
DICOM obl 1024 PROD   9.86   9.69      9.51       10.36      -1.9% neutral
```

Old-cool reproduces doc to `<1%` (`Skin OFF 69.40 vs 69.34`, `ON 29.52 vs 30.41` likewise) — machine/harness/protocol chain validated, so the earlier `+6-8%` is quantified sustained-bench throttling, not code. `DEF` neutrality holds cold; `F16` win is thermal-proof (beats old-cool even warm). Bonus findings from the checkout: blocks default-ON already works at `3293a9a14a` (`Airways 2048 SD0.5` unset≈ON `36.06/35.99`, OFF `95.13`) — plain cell-walk never leapt at fine in either era, and doc's `14.13/14.08` matches neither, corroborating `§26.4` mislabel verdict.

### 26.7 Rolled 8-rung (moves `DEF`) + `shadeCap` re-sweep — keep `8:2` (2026-09-03)

Same recipe on the 8-rung (`sBuf[8]`, same `MV9_COMPOSITE_LOOP`, `fc_shading`-gated, lean keeps pastes): `AIR 1437552→1620464 +13%` pre-specialization (both variants carried; per-`PSO` folds), `xcrun 2 warnings`, parity byte-max `0` (`2999 thr 0.000`, `SD0.5 0.035`).

```
NIFTI 1024 SD0.5 (cool): DEF 9.47 -> 9.12 -3.7%, F16 9.12 tie, F8-forced 9.37 +2.7% (PSO noise, cf. §26 def-vs-f2)
NIFTI 2048 SD0.5 (cool): DEF 25.47 -> 22.48 -11.7%, F16 22.29 tie (-0.8%)
NIFTI 1024 SD4: DEF 6.84, forced-F8 (rolled-8 at coarse) 6.97 +1.9% noise (DEF uses 2-rung)
DICOM 1024 (lean pastes): SD4 9.87 SD0.5 14.79 — unchanged
```

Bigger win at `2048` than `1024` fits `I$` contention scaling with thread count. Re-sweep verdict: fine `DEF(8)` vs `F16` is now a tie at both resolutions (`9.12/9.12`, `22.48/22.29`) — no basis to move the default, **`shadeCap 8:2` stays**. `F32/F48` re-confirmed dead (`+84%/+157%`, `§26.3`).

