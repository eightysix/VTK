# Metal Volume Rendering — NIFTI Brain Investigation (Part 2 Structural Follow-up) — UNIFIED

*This file is now unified into `perf_investigation_part2.md` (see that file for the full §1-12 investigation). Kept as a stub for history.*

*Continuation of `perf_investigation_part2.md:1` (precise `§1-8` repro, `e05a147fb5` 6-fetch → `b2e0286446` `i>0` fix `thr 3.32→0.18`) for the remaining `M/GL>1` cases (`Metal slower than GL`). Follow-up prefers **core structural** improvements over cheats/adaptive per-ray heuristics (`Rendering/Metal/Shaders/MetalShaders.metal:5568 batchCap`, `:3861 computeGradientFast`, `:5259 MV9_COMPOSITE`, `vtkMetalGPUVolumeRayCastMapper.mm:9632 maxBatchWidth`, `PerformanceInvestigation/HARNESS_VS_APP_GAP.md:30` argmin policy).*

- **Hardware:** Apple M2 `arm64 Release` `MTL_DEBUG_LAYER=0`
- **Datasets:** `DICOM 512x512x1794 U8` `IMRToraceAddome` vs `NIFTI 632x826x574 float32 1.51-70.29 0.2mm 1.1GB → U8 300MB` `Synthesized_FLASH25_downsampled_200um.nii` `Brain 7T FLASH25` `x6.5..45 y0..1` `useShading 1` `TestMetalScenes.h:1480`
- **Branch:** `b2e0286446` + `99ad0f014b` `shade4/lean16` + local `fc_fineSD<0.75` `shadeCap 2/4` `pow skip vDotR<0.5` `Rendering/Metal/Shaders/MetalShaders.metal:4002` `6-fetch` default ( `4-fetch` `fc_grad4` `fineSD`-gated for `thr<5` max perf, see main doc §9)
- **Bench:** `vtkMetalGLVisualComparison` `30f/10w @1024` `20f/5w @2048` `WaitForCompletion` vs `glFinish` `ABBA` per `VIEW="" axx axy axz az45 az135` `SD4/0.5`

**Unified:** All inventory (§1 `M/GL>1` with `heavy32`), isolation (§2 shading primary driver), root cause (§3 `6-fetch` `37%` occupancy), structural candidates (§4 `sNearest` `batchCap` `transpose` `compute marcher` `48-wide`), adaptive comparison (§5), recommendation (§6 `shadeCap 2/4` `pow skip` `argmin`), repro (§7), evaluation (§9 `0-degradation` `shadeCap 2/4` `pow skip` `M/GL 0.70/0.75` `thr 0.54/2.98` vs `4-fetch` `thr<5` `0.62`), A/B table (§10), next (§11 `4-fetch` `pow LUT` `precomp`), code map (§12) are now in `perf_investigation_part2.md` `§1-12` (unified).

**Repro for latest `0-degradation` max perf (see main doc §5):**

```sh
./macos_metal_build.sh --resume --tests
BIN=build_macos_metal/bin/vtkMetalGLVisualComparison
NIFTI=/Users/macair/Public/IMR/7T-MRI/Synthesized_FLASH25_downsampled_200um.nii
DICOM=/Users/macair/Public/IMR/CTIMR/IMRToraceAddome
BASE="VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_NUM_SLABS=1 VTK_METAL_TEST_IGN_JITTER=0 VTK_METAL_TEST_JITTER=1 VTK_METAL_TEST_MARCH_VARIANT=9 VTK_METAL_TEST_MINMAX=1 VTK_METAL_TEST_ACCEL=1"
eval "env $BASE VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y $BIN --scene VolumeRayCast --frames 1 --size 512x512 --warmup 2 --out visual_compare 2>&1 | grep -E 'VolumeRayCast|worst'"
eval "env $BASE VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y $BIN --scene DICOMVolume --dicom $DICOM --frames 1 --size 512x512 --warmup 2 --out visual_compare 2>&1 | grep -E 'DICOM|worst'"
eval "env $BASE $BIN --scene NIFTIVolume --nifti $NIFTI --frames 1 --size 512x512 --warmup 2 --out visual_compare 2>&1 | grep -E 'NIFTI|worst'"
# NIFTI identity (no VOLTRANSPOSE_AXIS) is correct policy for 632x826x574 (argmin 0)
```

**Code map:** See `perf_investigation_part2.md:12`.

---
