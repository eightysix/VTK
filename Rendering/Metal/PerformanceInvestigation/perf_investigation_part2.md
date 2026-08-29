# Metal Volume Rendering — NIFTI Brain Investigation (Part 2, Unified)

*Unified follow-up to `PERFORMANCE_INVESTIGATION.md` and `HARNESS_VS_APP_GAP.md` (42-section investigation: jitter `§1-6`, resolution `§7`, transposition `§26`, RG8 `§17`, compute marcher `§38`, fragment batch `§39`, mv9 coverage `§40`, `HARNESS_VS_APP_GAP.md:39-40` NIFTI precursor). Covers the new `NIFTI` scene (`TestMetalScenes.h:108 gNiftiPath`, `BuildNIFTIVolumeScene`). See `Rendering/Metal/METAL_VOLUME_RAYCAST_GUIDE.md:1` for mapper+shader guide. This file unifies `perf_investigation_part2.md` and `perf_investigation_part2_structural.md` (structural follow-up) into one document.*

- **Hardware:** Apple M2 (Metal 3) `arm64 Release` `MTL_DEBUG_LAYER=0` `AC`
- **Datasets:** `DICOM 512x512x1794 U8 0.4-0.8mm 470MB` `IMRToraceAddome` vs `NIFTI 632x826x574 float32 1.51-70.29 0.2mm 1.1GB → U8 300MB` `Synthesized_FLASH25_downsampled_200um.nii` `Brain 7T FLASH25` `x6.5..45 y0..1` `useShading 1` `TestMetalScenes.h:1480` `Examples/GUI/iOSMetal/test-vtk-metal/NIFTIVolumeViewController.mm:42`
- **Scenes:** `DICOMVolume` `Airways II` vs `NIFTIVolume` `FLASH25` `harness --nifti` `vtkNIFTIImageReader` `BuildNIFTIVolumeScene`
- **Branch:** `b2e0286446` `fix(mv9): i>0 gate at Rendering/Metal/Shaders/MetalShaders.metal:5634` `thr 3.32→0.18` + `99ad0f014b` `shade4/lean16` `Rendering/Metal/Shaders/MetalShaders.metal:5605` `37%→56% TG` + local `fc_fineSD<0.75` `shadeCap 2/4` `pow skip vDotR<0.5` `Rendering/Metal/Shaders/MetalShaders.metal:4002` `fc_grad4/gradNearest` env-gated, `SD-aware 4-fetch` gated to `fineSD` for max perf without `thr` loss (see §9-10)
- **Bench:** `vtkMetalGLVisualComparison` `30f/10w @1024` (`20f/5w` for quick) `20f/5w @2048` `glFinish` vs `WaitForCompletion` `ABBA` per `VIEW="" az45 az135 axx axy axz` `SD` mm
- **Code map:** `Rendering/Metal/Shaders/MetalShaders.metal:4363` `maxSteps`, `:5634` `mv9 tEnd gate`, `:6416` `mv0 loop`, `:5568` `batchCap`, `:5600` `adaptive`, `:3861` `computeGradientFast`, `:5259` `MV9_COMPOSITE`, `:6254` `PROC_UNROLL_SAMPLE` `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm:242` `fc_marchVariant 9`, `:7963` `PipelineCache key`, `:9632` `maxBatchWidth`, `:477` `VolumeTransposedAxisDepth` `Rendering/Metal/Testing/Cxx/TestMetalVolumeRayCast.cxx:1` `parity` `Examples/GUI/iOSMetal/test-vtk-metal/NIFTIVolumeViewController.mm:42` `preset`

---

## 1. Reproduction (precise `§39.4` / `§40.3` `zsh` `eval`)

`zsh` does not word-split `BASE`, so the `§39.4` `eval "env $BASE ..."` form is required, otherwise `VOLTRANSPOSE_AXIS`, `CAM_AXIS` and `JITTER` are dropped and the DICOM/NIFTI render with different orientations (as in the two cyan images you sent, thr 6423 at 512 without `BASE`).

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
# fine SD parity
env VTK_METAL_TEST_SAMPLE_DISTANCE=0.5 $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out /tmp/p2 2>&1 | grep -E 'NIFTI|worst'
# perf 1024/2048 y vs identity, shade, fragBatch
for SD in 4 0.5; do for VIEW in "" "VTK_METAL_TEST_CAM_AXIS=y" "VTK_METAL_TEST_CAM_AZ=45"; do
  eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=$SD $BASE $VIEW $BIN --bench --backend metal --scene NIFTIVolume --nifti $NIFTI --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep '^NIFTIVolume'"
  eval "env VTK_METAL_TEST_SAMPLE_DISTANCE=$SD $BASE $VIEW $BIN --bench --backend gl --scene NIFTIVolume --nifti $NIFTI --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep '^NIFTIVolume'"
done; done
for F in 1 2 4 8 16 32; do eval "env VTK_METAL_TEST_FRAG_BATCH=$F $BASE VTK_METAL_TEST_SAMPLE_DISTANCE=0.5 $BIN --bench --backend metal --scene NIFTIVolume --nifti $NIFTI --frames 20 --size 1024x1024 --warmup 5 2>&1 | grep '^NIFTIVolume'"; done
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

### 8.2 Per-feature compile-time batch cap — `batchCap:5568` `fc_shading||fc_gradientOpacity` + `fc_fineSD<0.75`

`fc_fragBatch` already gives `light` `56%` vs `heavy` `37%` (`§39.1`). Shipped `maxBatchWidth=32` `heavy` is pessimal for NIFTI shade ON. Static per-PSO specialization (no per-ray branch):

```metal
const int shadeCap = fc_fineSD ? 2 : 4;
const int batchCap = (fc_fragBatch>0) ? fc_fragBatch
                 : ((fc_shading||fc_gradientOpacity) ? min(shadeCap, max(1,int(maxBatchWidth)))
                                                     : min(16, max(1,int(maxBatchWidth))));
```

`fc_shading`/`fc_gradientOpacity` `2666/2667` `fc_fineSD` `44` `Mapper:8181` `SampleDistance<0.75` (`0.5 fine vs 4 coarse` and `1.0 VolumeRayCast` stays `6-fetch`). `shadeCap 2` best for `NIFTI SD0.5` (`f2 0.86 vs f4 0.93 -7%`), `4` for `SD4` (`0.93 vs 0.99 -6%`), `lean16` `0.48 vs 0.50` close; `cap1` hurts `DICOM 18%`.

Measured `M/GL` `1024` `MINMAX=1` `ACCEL=1` `30f/10w`:
```
Before (heavy32 y): NIFTI shade ON f16 1.18 FAIL, f2 0.87 PASS
After (shade4/lean16 y + pow skip): NIFTI SD4 6.03/8.62 0.70 SD0.5 13.64/18.13 0.75 -12% fine, all <1
After (fineSD 2/4 + 4-fetch at fine): NIFTI SD0.5 10.97/17.64 0.62 -18% extra at fine, thr 0.54->2.29 still <5 (see §9)
Parity 512 y: NIFTI SD4 3780 thr 2.98 (was 2.93) mean 0.008 max8, SD0.5 3589 thr 0.54 (6-fetch) or 3729 thr 2.29 (4-fetch fine) - both <5, DICOM 0.000, VolumeRayCast 0.182 keep.
```

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

### 9.7 Thresholded error root cause — `thr 2.98` is shading-driven `half`+`pow` with headroom

`vtkImageDifference:622` `SetThreshold(20)` `AllowShift ±2` `Averaging 3×3` — raw `|M-G|>1` is `36.13%` `mean 12.81 max 219`, `>20` is `18.64%`, `thr 2.98` after shift/average. `NIFTI 512 y`:
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

**Verdict:** `shadeCap 2/4` (`fineSD<0.75`) + `pow skip` + `argmin` transpose is the minimal structural set with `0` parity cost (`thr 2.98 vs f2 2.98` `mean 0.008`) and `0` `DICOM` regression (`<1` all views, `+6~14%` win) — **max perf without `thr` loss**: `1024 0.70/0.75` `2048 0.77/0.96` all `<1`. `SD-aware 4-fetch` at fine adds `-18%` at `SD0.5` (`13.64->10.97`) for `+1.74%` thr to `2.29` still `<5` — **max perf with `thr<5`**: `1024 SD0.5 0.62` `2048 SD0.5 0.62`. Gradient precomp and `sNearest` remain available as `+1.2GB` / `6*1` options if needed.

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
| batchCap SD-aware | `Rendering/Metal/Shaders/MetalShaders.metal:5568` | `shadeCap=fc_fineSD?2:4; batchCap=(fc_fragBatch>0)?fragBatch:((fc_shading\|\|fc_gradientOpacity)?min(shadeCap,MaxBatchWidth):min(16,MaxBatchWidth))` `featureMaskExtra 32u fineSD` `Mapper:7963` `SampleDistance<0.75` |
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
