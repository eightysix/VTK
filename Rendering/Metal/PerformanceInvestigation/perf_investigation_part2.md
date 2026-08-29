# Metal Volume Rendering — NIFTI Brain Investigation (Part 2)

*Follow-up to `PERFORMANCE_INVESTIGATION.md` and `HARNESS_VS_APP_GAP.md:39-40` for the new `NIFTI` scene.*

- **Hardware:** Apple M2 (Metal 3) `arm64 Release` `MTL_DEBUG_LAYER=0` `AC`
- **Datasets:** `DICOM 512x512x1794 U8 0.4-0.8mm 470MB` `IMRToraceAddome` vs `NIFTI 632x826x574 float32 1.51-70.29 0.2mm 1.1GB → U8 300MB` `Synthesized_FLASH25_downsampled_200um.nii` `Brain 7T FLASH25` `x6.5..45 y0..1` `useShading 1` `TestMetalScenes.h:1480`
- **Scenes:** `DICOMVolume` `512x512x1794` `Airways II` vs `NIFTIVolume` `632x826x574` `FLASH25` `BuildNIFTIVolumeScene` `harness --nifti`
- **Bench:** `vtkMetalGLVisualComparison` `30f/10w @1024` `60f/10w @2048` `glFinish` vs `WaitForCompletion` `§39.3` `ABBA` per `VIEW="" az45 az135 axx axy axz` `SD` mm `y-transpose` default

---

## 1. Reproduction

```sh
# build once
./macos_metal_build.sh --resume --tests
# or
ninja -C build_macos_metal vtkMetalGLVisualComparison

BIN=build_macos_metal/bin/vtkMetalGLVisualComparison
NIFTI=/Users/macair/Public/IMR/7T-MRI/Synthesized_FLASH25_downsampled_200um.nii
DICOM=/Users/macair/Public/IMR/CTIMR/IMRToraceAddome
BASE="VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1 VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y"

# smoke parity 512 §40.3
OUT1=/tmp/nifti0; OUT2=/tmp/nifti16; rm -rf $OUT1 $OUT2; mkdir -p $OUT1 $OUT2
env $BASE $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out $OUT1
env $BASE VTK_METAL_TEST_FRAG_BATCH=16 $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out $OUT2
python3 -c "from PIL import Image;import numpy as np,os;a=np.array(Image.open(os.path.join('$OUT1','NIFTIVolume.metal.png')));b=np.array(Image.open(os.path.join('$OUT2','NIFTIVolume.metal.png')));d=np.abs(a.astype(int)-b.astype(int));print(f'mean {d.mean():.4f} max {d.max()} >1LSB {100*(d>1).mean():.3f}%')"
# mean 0.0000 max0 0.000% lean frag0 vs f16 both NIFTI/DICOM pass §40.3

# perf 1024/2048 y-transpose
for VIEW in "" "VTK_METAL_TEST_CAM_AXIS=x" "y" "z" "VTK_METAL_TEST_CAM_AZ=45" "VTK_METAL_TEST_CAM_AZ=135"; do
  eval "env $BASE $VIEW $BIN --bench --backend metal --scene NIFTIVolume --nifti $NIFTI --frames 30 --size 1024x1024 --warmup 10 2>&1 | grep NIFTIVolume"
  eval "env $BASE $VIEW $BIN --bench --backend gl --scene NIFTIVolume --nifti $NIFTI --frames 30 --size 1024x1024 --warmup 10 2>&1 | grep NIFTIVolume"
done
# SD sweep, fragBatch, shade/jitter, accel, transpose: see /tmp/ab_nifti_test.py and /tmp/test_frag1_vs_mv0.py
```

`NIFTIVolume` `BuildNIFTIVolumeScene: loaded ... dims 632x826x574 min 1.51 max 70.29` `thr err 512 0.39-2.93` `400 0.44` `GL/Metal` parity.

---

## 2. Current results — NIFTI vs DICOM `1024 30f y SD0.5/4 accel OFF` `2048 30f y SD0.5 accel OFF`

`M/GL <1` `Metal` faster. `NIFTI` dense `>60%` `alpha>0` vs `DICOM` sparse `~40%` empty `§9`.

### 2.1 SD sweep `obl y` `mv9 f16` vs `mv0`

```
SD0.5 NIFTI mv0 12.22/18.42 0.66 f16 16.29/17.80 0.92
SD4   NIFTI mv0 4.44/8.93 0.50 f16 5.86/8.62 0.68
SD0.5 DICOM mv0 99.07/160.69 0.62 f16 50.11/162.05 0.31
SD4   DICOM mv0 8.97/36.71 0.24 f16 10.02/39.69 0.25
```
`48-wide` `mv9` `37%` `§39.1` hurts `NIFTI 41 steps SD4` `200 SD0.5` short vs `DICOM 86/400` long.

### 2.2 FragBatch `1024 axy y SD0.5 accel OFF`

```
NIFTI: f1 11.54/12.73 0.91 f2 11.08/12.94 0.86 f4 11.46/12.87 0.89 f8 12.77/12.94 0.99 f16 14.77/12.97 1.14 f32 20.18/12.88 1.57 mv0 11.98/12.68 0.94
DICOM: f1 89/162 0.55 f2 75/161 0.47 f4 71/162 0.44 f8 49/160 0.31 f16 50/162 0.31 f32 52/162 0.32 mv0 99/160 0.62
```
`NIFTI` best `f2 0.86`, `DICOM` best `f8/f16 0.31` — `n>2` penalty `+20%` `shade0` `+34%` `shade1` `f4→f16`.

### 2.3 Transpose `SD4 obl` `f16` `SD0.5 obl f16`

```
SD4 y 6.52/9.48 0.69 vs none 7.60/9.08 0.84 -14%
SD0.5 y 19.80/18.35 1.08 vs none 20.36/18.21 1.12 -3%
```
`574Z` vs `1794Z` less `Z-tiling` `§15`.

### 2.4 2048 `SD0.5 y accel OFF`

```
DICOM obl 98→49 -50% f16, axz 312→148 -53% f16
NIFTI obl 36.02→39.63 +10% f8 vs f2 31.62 best, axz 10.38→10.84 +4% f8
2048 y axy f8 NIFTI 12.77/12.66 1.01 f2 11.08/12.94 0.86 PASS, f16 14.77/12.97 1.14 FAIL
1024 axy y shade0 f4 6.18 vs f16 7.45 +20%, shade1 f2 10.96 vs f16 14.75 +34%
```

Single `f8 light y SD4` `M/GL<1` 5/6 views `NIFTI` `0.76-0.99` `f2` `0.86` best `axy`, `DICOM` `0.31` within `2%` `f16`.

---

## 3. Shading focused

`6 fetches` `computeGradientFast:3861` `s±dx/s±dy/s±dz` `+ TF` `8 fetches/sample` vs `1` lean `§39.5`.

```
NIFTI axy y SD0.5 2048 accel OFF:
shade0 6.87/8.51 0.81 vs shade1 13.03/12.91 1.01 +90% 6
4 forward sC+3 11.70/13.94 0.84 -10% 33% fetch save
3 forward sC reuse half(rawScalar)/half(s##_j) 10.39/13.20 0.79 -11% vs 6, +2% vs 4 f2 5.35 vs 5.22
512 thr err shade1 2.93→2.26 <5% §40.3 mean<0.04 max<25 vs shade0 0.256
```

`6→4` kept `8aaddf2dc6`, `4→3` `+2%` `f2` slower `5.35>5.22` not kept. `f16` shade `8*7=112` fetches/batch `+` serial `w` chain `+` `pow` `→` `+83%` `f2→f32` `shade1` vs `+20%` `shade0`.

Precompute `gradient volume` `RGBA8Snorm` `1 fetch` `→` `8→2` fetches/sample `75%` cut but `+900MB` `632x826x574` too big. Keep `4` default.

---

## 4. Remaining `f>2` gap

Not `MM` (`off` still `DICOM 98→49 -50%` `NIFTI 14→17 +22%`), not `shade` alone `f4 6.18 vs f16 7.45 +20%` `shade0`. `n*7` fetches + `n` `pow` + `37%` `I$` `tail 41%41` vs `DICOM 400` `16=2+9` spill. `f2` best `NIFTI` short `41 steps`, `f16` best `DICOM` long.

Cut: `maxStepsFrame=ceil(maxChord/SD)` `§4` already `uniform` → `batchCap=min(fragBatch, maxSteps/4)` `MetalShaders.metal:5600` `if(maxSteps<50) 2 else if<100 8 else 16` `→` `NIFTI→2` `DICOM→16` single `PSO` `f2 0.87` vs `f8 0.99` keeps `<1` both.

---

## 5. Repro for shade + batch

```sh
# shade bisect axy y SD0.5 1024/2048
for S in 0 1; do for F in 1 2 4 8 16 32; do env VTK_METAL_TEST_SHADE=$S VTK_METAL_TEST_FRAG_BATCH=$F VTK_METAL_TEST_SAMPLE_DISTANCE=0.5 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y VTK_METAL_TEST_CAM_AXIS=y $BIN --bench --backend metal --scene NIFTIVolume --nifti $NIFTI --frames 30 --size 1024x1024 --warmup 10 | grep NIFTIVolume; done; done
# 4 vs 6 fetch: git show HEAD~1:Shaders/MetalShaders.metal | grep -A5 computeGradientFast
```

Tree: `metal-volume-parity-essential 8aaddf2dc6` `4-fetch` `light` `56%`, `xcrun metal -c 15 warnings`, `NIFTI 2048 axy f2 11.03/12.94 0.86 PASS`.
