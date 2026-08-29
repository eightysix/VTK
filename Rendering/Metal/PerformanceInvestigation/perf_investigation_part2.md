# Metal Volume Rendering — NIFTI Brain Investigation (Part 2)

*Follow-up to `PERFORMANCE_INVESTIGATION.md` and `HARNESS_VS_APP_GAP.md:39-40` for the new `NIFTI` scene (`TestMetalScenes.h:108 gNiftiPath`, `BuildNIFTIVolumeScene`).*

- **Hardware:** Apple M2 (Metal 3) `arm64 Release` `MTL_DEBUG_LAYER=0` `AC`
- **Datasets:** `DICOM 512x512x1794 U8 0.4-0.8mm 470MB` `IMRToraceAddome` vs `NIFTI 632x826x574 float32 1.51-70.29 0.2mm 1.1GB → U8 300MB` `Synthesized_FLASH25_downsampled_200um.nii` `Brain 7T FLASH25` `x6.5..45 y0..1` `useShading 1` `TestMetalScenes.h:1480` `Examples/GUI/iOSMetal/test-vtk-metal/NIFTIVolumeViewController.mm:42`
- **Scenes:** `DICOMVolume` `Airways II` vs `NIFTIVolume` `FLASH25` `harness --nifti` `vtkNIFTIImageReader` `BuildNIFTIVolumeScene`
- **Branch:** `metal-volume-parity-essential e05a147fb5` `6-fetch central` `thr fallback analytic 0.15` `y-transpose` default off for `512` fallback, `BASE` below default on for `512` real volumes → `7df345d` `fix(mv9): i>0 gate at MetalShaders.metal:5634` `thr 3.32→0.18` `VolumeRayCast`
- **Bench:** `vtkMetalGLVisualComparison` `30f/10w @1024` `60f/10w @2048` `glFinish` vs `WaitForCompletion` `§39.3` `ABBA` per `VIEW="" az45 az135 axx axy axz` `SD` mm

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

# smoke parity 512 §40.3 with y-transpose (correct)
OUT1=/tmp/nifti0; OUT2=/tmp/nifti16; rm -rf $OUT1 $OUT2; mkdir -p $OUT1 $OUT2
eval "env $BASE $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out $OUT1 2>&1 | grep NIFTI"
eval "env $BASE VTK_METAL_TEST_FRAG_BATCH=16 $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out $OUT2 2>&1 | grep NIFTI"
python3 -c "from PIL import Image;import numpy as np,os;a=np.array(Image.open(os.path.join('$OUT1','NIFTIVolume.metal.png')));b=np.array(Image.open(os.path.join('$OUT2','NIFTIVolume.metal.png')));d=np.abs(a.astype(int)-b.astype(int));print(f'mean {d.mean():.4f} max {d.max()} >1LSB {100*(d>1).mean():.3f}%')"
# mean 0.0000 max0 0.000% lean frag0 vs f16 both NIFTI/DICOM pass §40.3 (DICOM 512 y thr 0.000, NIFTI 512 y thr 2.93)

# without BASE (incorrect) the same 512 runs give DICOM thr 6423 and NIFTI thr 0.54 vs 2.93, different orientations

# visual_compare at 512 with correct y-transpose (proper invocation)
eval "env $BASE $BIN --scene VolumeRayCast --frames 1 --size 512x512 --warmup 2 --out visual_compare 2>&1 | grep -E 'VolumeRayCast|worst'"
eval "env $BASE $BIN --scene DICOMVolume --dicom $DICOM --frames 1 --size 512x512 --warmup 2 --out visual_compare 2>&1 | grep -E 'DICOM|worst'"
eval "env $BASE $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out visual_compare 2>&1 | grep -E 'NIFTI|worst'"
# VolumeRayCast 1150 thr 0.18 y (was 1160 thr 3.32 before 7df345d), DICOM 1122 thr 0.000 y, NIFTI 3797 thr 2.93 y at 512

# perf 1024/2048 y-transpose
for VIEW in "" "VTK_METAL_TEST_CAM_AXIS=x" "y" "z" "VTK_METAL_TEST_CAM_AZ=45" "VTK_METAL_TEST_CAM_AZ=135"; do
  eval "env $BASE $VIEW $BIN --bench --backend metal --scene NIFTIVolume --nifti $NIFTI --frames 30 --size 1024x1024 --warmup 10 2>&1 | grep NIFTIVolume"
  eval "env $BASE $VIEW $BIN --bench --backend gl --scene NIFTIVolume --nifti $NIFTI --frames 30 --size 1024x1024 --warmup 10 2>&1 | grep NIFTIVolume"
done
# SD sweep, fragBatch, shade/jitter, accel, transpose: see /tmp/ab_nifti_test.py and /tmp/test_frag1_vs_mv0.py
```

`NIFTIVolume` `BuildNIFTIVolumeScene: loaded ... dims 632x826x574 min 1.51 max 70.29` `visual_compare 08:27 y thr 0.000 DICOM 124k, 2.93 NIFTI 288k at 512 y` is the up-to-date reference. Without `eval` the `512` fallback analytic is `0.15` but the real `DICOM/NIFTI` at `512` without `y` is `6423`/`0.54`.

---

## 2. Current results — `512 y` is the `§40.3` reference, `1024/2048 y SD0.5/4 accel OFF` for `M/GL`

`M/GL <1` `Metal` faster. `NIFTI` dense `>60%` `alpha>0` vs `DICOM` sparse `~40%` empty `§9`.

### 2.1 `512 y` correct `BASE` (`e05a147fb5` `6-fetch` → `7df345d` `i>0` fix)

```
VolumeRayCast y 1150 thr 0.18 (was 1160 thr 3.32 before fix, 1507 with 4-fetch, 0.15 without y / mv0)
DICOMVolume y 1122 thr 0.000
NIFTIVolume y 3797 thr 2.93 (3803 thr 2.93 after fix, Δ <0.2%)
visual_compare 512 y: VolumeRayCast 08:27 170k, DICOM 08:27 124k, NIFTI 08:27 288k (all y)
512 without y: DICOM 9451 thr 6423, NIFTI 3589 thr 0.54, VolumeRayCast 1147 thr 0.15 (fallback analytic, mv0 0.18 y-parity after fix)
```

Without `eval` the `y` is dropped, hence the `6423` you saw and the two different cyan orientations.

### 2.2 `1024/2048 y SD0.5/4 accel OFF` (`30f/10w y`)

```
SD0.5 NIFTI mv0 12.22/18.42 0.66 f16 16.29/17.80 0.92 (y)
SD4   NIFTI mv0 4.44/8.93 0.50 f16 5.86/8.62 0.68 (y)
SD0.5 DICOM mv0 99.07/160.69 0.62 f16 50.11/162.05 0.31
SD4   DICOM mv0 8.97/36.71 0.24 f16 10.02/39.69 0.25
```

`48-wide` `mv9` `37%` `§39.1` hurts `NIFTI 41 steps SD4` `200 SD0.5` short vs `DICOM 86/400` long.

### 2.3 FragBatch `1024 axy y SD0.5 accel OFF`

```
NIFTI: f1 11.54/12.73 0.91 f2 11.08/12.94 0.86 f4 11.46/12.87 0.89 f8 12.77/12.94 0.99 f16 14.77/12.97 1.14 f32 20.18/12.88 1.57 mv0 11.98/12.68 0.94
DICOM: f1 89/162 0.55 f2 75/161 0.47 f4 71/162 0.44 f8 49/160 0.31 f16 50/162 0.31 f32 52/162 0.32 mv0 99/160 0.62
```
`NIFTI` best `f2 0.86`, `DICOM` best `f8/f16 0.31` — `n>2` penalty `+20%` `shade0` `+34%` `shade1` `f4→f16`.

### 2.4 Transpose `SD4 obl` `f16` `SD0.5 obl f16` (correct `BASE`)

```
SD4 y 6.52/9.48 0.69 vs none 7.60/9.08 0.84 -14%
SD0.5 y 19.80/18.35 1.08 vs none 20.36/18.21 1.12 -3%
```
`574Z` vs `1794Z` less `Z-tiling` `§15`.

### 2.5 `2048 y SD0.5 accel OFF`

```
DICOM obl 98→49 -50% f16, axz 312→148 -53% f16
NIFTI obl 36.02→39.63 +10% f8 vs f2 31.62 best, axz 10.38→10.84 +4% f8
2048 y axy f8 NIFTI 12.77/12.66 1.01 f2 11.08/12.94 0.86 PASS, f16 14.77/12.97 1.14 FAIL
1024 axy y shade0 f4 6.18 vs f16 7.45 +20%, shade1 f2 10.96 vs f16 14.75 +34%
```

Single `f8 light y SD4` `M/GL<1` 5/6 views `NIFTI` `0.76-0.99` `f2` `0.86` best `axy`, `DICOM` `0.31` within `2%` `f16`.

---

## 3. Shading focused — why `4-fetch` was reverted

`6 fetches` `computeGradientFast:3861` `s±dx/s±dy/s±dz` `+ TF` `8 fetches/sample` vs `1` lean `§39.5`.

```
NIFTI axy y SD0.5 2048 accel OFF:
shade0 6.87/8.51 0.81 vs shade1 13.03/12.91 1.01 +90% 6 (3-fetch try 10.39/13.20 0.79 gave f2 5.35 vs 5.22 +2% slower for f2, so kept)
4 forward sC+3 11.70/13.94 0.84 -10% 33% fetch save
512 thr err shade1 2.93 y vs 0.256 shade0, VolumeRayCast y 3.32 vs 0.15 without y, 6→4 gave VolumeRayCast 1507 vs 0.15 (fallback 0.15) and DICOM 6423, so reverted to 6 at e05a147fb5
```

Without `e05a147fb5` (`4-fetch` at `50291a4b94`): `VolumeRayCast y 4475 thr 1531` vs `1160 thr 3.32` with `6`, `NIFTI y 3813 thr 5.65` vs `3797 thr 2.93` with `6`. `6` keeps `§40.3` `thr <5%` `mean<0.04 max<25`.

Precompute `gradient volume` `RGBA8Snorm` `1 fetch` `→` `8→2` `75%` cut but `+900MB` `632x826x574` too big. Keep `6` default.

---

## 4. Remaining `f>2` gap

Not `MM` (`off` still `DICOM 98→49 -50%` `NIFTI 14→17 +22%`), not `shade` alone `f4 6.18 vs f16 7.45 +20%` `shade0`. `n*7` fetches + `n` `pow` + `37%` `I$` `tail 41%41` vs `DICOM 400` `16=2+9` spill. `f2` best `NIFTI` short `41 steps`, `f16` best `DICOM` long.

Cut: `maxStepsFrame=ceil(maxChord/SD)` `§4` already `uniform` → `batchCap=min(fragBatch, maxSteps/4)` `MetalShaders.metal:5600` `if(maxSteps<50) 2 else if<100 8 else 16` `→` `NIFTI→2` `DICOM→16` single `PSO` `f2 0.87` vs `f8 0.99` keeps `<1` both.

---

## 5. Repro for shade + batch on this branch (`e05a147fb5` `6-fetch` `clean on 8ef6b9a68d`)

```sh
# shade bisect axy y SD0.5 1024/2048 (checkouts, not resets, as requested)
git checkout 50291a4b94 -- Rendering/Metal/Shaders/MetalShaders.metal # 4-fetch
ninja -C build_macos_metal vtkMetalGLVisualComparison
# vs
git checkout e05a147fb5 -- Rendering/Metal/Shaders/MetalShaders.metal # 6-fetch
ninja -C build_macos_metal vtkMetalGLVisualComparison
# then with correct BASE
BIN=build_macos_metal/bin/vtkMetalGLVisualComparison
BASE="VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1 VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y"
for S in 0 1; do for F in 1 2 4 8 16 32; do eval "env VTK_METAL_TEST_SHADE=$S VTK_METAL_TEST_FRAG_BATCH=$F $BASE $BIN --bench --backend metal --scene NIFTIVolume --nifti $NIFTI --frames 30 --size 1024x1024 --warmup 10 2>&1 | grep NIFTIVolume"; done; done
# 6 vs 4 fetch: git show HEAD~1:Shaders/MetalShaders.metal | grep -A5 computeGradientFast
```

Tree: `metal-volume-parity-essential e05a147fb5` `6-fetch` `light` `56%` `visual_compare 512 y thr 0.000 DICOM 124k, 2.93 NIFTI 288k at 08:27` is the `§40.3` reference (`7df345d` `VolumeRayCast 0.18` after fix). `branch` not reset since `95045c5a3d`, only `checkout` as requested. `NIFTI 2048 axy y f2 11.03/12.94 0.86 PASS` vs `f16 1.14` (see §7 M>GL).

---

## 6. Far edge missing sliver (mv9 only) — root cause and fix (`7df345d`)

`VolumeRayCast 512 y mv9 thr 3.32 vs mv0 0.18` `DICOMVolume 512 y 0.000` `NIFTI 512 y 2.93` — `MTL` missing thin `far edge` strip (blue cube bottom/right vs `GL` as in your two images, `VolumeRayCast` checker `32` analytic). `/tmp/vol_orig2` before, `/tmp/vol_check2` after.

**Root cause `MetalShaders.metal:5634` `marchVolumeUnified` `mv9` `while(i<steps)`:** `maxSteps = max(1,int(ceil((p.tEnd-firstT)/p.stepSize)))` `MetalShaders.metal:4363` guarantees `≥1` sample (clamped boundary) even when `tEnd-firstT < 0` (`firstT=jitter`, grazing chord `p.tEnd ≈ t.y-tStart < jitter`). `mv0` `MetalShaders.metal:6416` `for(i<maxSteps)` has no `tEnd` break at `i==0` for `checkBounds` (`latchExit` only for `fc_marchVariant≥4`), so it composites that `1` clamped sample. `mv9` broke **before first fetch** `if(currentT>=p.tEnd-1e-6)break` `MetalShaders.metal:5634` → `0` samples for `≈2024/262k 0.77%` grazing pixels `512²` `y22 gl1 metal0` `y24 gl8 metal3` etc. — silhouette `1-2px` trim all edges, worst `max 229` `mean 0.408` `thr 3.32`.

**Attempts on `e05a147fb5` `6-fetch` (checkouts, not resets):**

* `8a8052494b` `maxSteps+1` `MetalShaders.metal:4363 8260 8381` `ceil((tEnd-firstT)/stepSize)+1` `→` `VolumeRayCast 512 y 3.32→2.26 -32%` `DICOM 0.000 keep` `NIFTI 2.93→2.95` `±1%` — not fix (`i==0` break still `0` when `firstT>tEnd`), `c7a1259118` revert.
* Per-sample `tEnd` `if (currentT + float(_j)*stepSize >= p.tEnd -1e-6)` `§6254` `PROC_UNROLL_SAMPLE` `→` still `3.32` (batch not dispatched, `i==0` breaks before `PROC_UNROLL_SAMPLE`).
* `+0.5*stepSize` inside `ceil` `→` `2.83` worse than `+1`.

**Fix `7df345d` `MetalShaders.metal:5634`:** `if(i>0 && currentT>=p.tEnd-1e-6)break` — at-least-one batch, parity to `mv0`. Preserves `fc_slabMode` `5633` before, `seenInBounds` `5638`, `segHop/minmax` `5643` after, `MV9_ADVANCE` `5560` `tTerminateMax` after batch. `i>0` false only at first batch, `1` `int` `cmp` per outer iter `2 batches NIFTI 41 steps` `7 batches 200 steps` negligible.

**Verification `7df345d` `arm64 Release`:**

```
VolumeRayCast 512 y 1150 thr 0.18 (was 1160 thr 3.32) mean 0.079 max 83 vs mv0 0.18 mean 0.079 — 0 mismatched (was 2024)
  y22 gl1 metal0 → gl1 metal1, y24 gl8 metal3 → gl8 metal8, `mean 0.408→0.079`
DICOMVolume 512 y 1122 thr 0.000 keep (mean 0.064 vs mv0 0.064)
NIFTIVolume 512 y 3803 thr 2.93 (was 3797 thr 2.93, Δ<0.2% mean 12.81 vs mv0 12.80)
1024 y SD4/0.5 30f/10w: DICOM 8.24/37.96 0.22 vs 8.61/39.47 0.22 (-4%, noise), NIFTI 7.44/9.10 0.82 vs 7.46/9.18 0.81, 27.17/18.12 1.50 vs 27.24/18.3 1.49 (<1%)
JITTER=0 VolumeRayCast 512 y 901 thr 0.24 (was 1240 thr 285) — same mechanism (jitter=stepSize)
```

```sh
# repro for far edge missing sliver (mv9 only) — before e05a147fb5 vs after 7df345d
BIN=build_macos_metal/bin/vtkMetalGLVisualComparison
BASE="VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1 VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y"
eval "env $BASE $BIN --scene VolumeRayCast --frames 1 --size 512x512 --warmup 2 --out /tmp/vol_orig2 2>&1 | grep -E 'VolumeRayCast|worst'"
# before: VolumeRayCast 1160 thr 3.32 vs mv0 thr 0.15 (no y 0.15), DICOM 0.000, NIFTI 2.93; after: 1150 thr 0.18 vs mv0 0.18
# check: python3 -c "from PIL import Image;import numpy as np;a=np.array(Image.open('/tmp/vol_check2/VolumeRayCast.metal.png'));b=np.array(Image.open('/tmp/vol_check2/VolumeRayCast.gl.png'));print((np.abs(a.astype(int)-b.astype(int))>1).mean())"
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

Shade split 1024 axy y SD0.5: NIFTI shade0 f4 6.18 vs f16 7.45 +20% (shade0 still <1), shade1 f2 10.96 vs f16 14.75 +34% (f16 >1) — not shade alone (§3), not MM (MM off still NIFTI 14→17 +22% vs DICOM 98→49 -50%).

Pattern: `M>GL` only `NIFTI` `SD0.5` (dense, short chord `~200 steps`) `f≥8 (f16/f32)` — tail `41%41` vs DICOM `400` `16=2+9`, `48-wide` `37%` occupancy `§39.1` spill. `SD4 NIFTI` `41 steps` still `<1` even at `f16` (`0.68`), `2048 SD4 y` `NIFTI 0.76-0.99 f2/f8` `<1` 5/6 views under `f8 light`.

Custom `VTK_METAL_TEST_FRAG_BATCH` vs `mv0`:
  NIFTI 1024 axy y SD0.5 f2 0.86 best vs mv0 0.94 (-8%), f16 1.14 +21% vs mv0 — wide fetch array `n*7` hurts short rays.
  DICOM f16 0.31 vs mv0 0.62 (-50%) — long rays amortize fetch.

Next: §4 `batchCap=min(fragBatch,maxSteps/4)` adaptive `MetalShaders.metal:5600` `if(maxSteps<50)2 else if<100 8 else 16` → `NIFTI→2 DICOM→16` single `PSO` `f2 0.87 vs f8 0.99` keeps `<1` both (see §4).
```

---
