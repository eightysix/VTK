# App slab-tiling benchmarks: exact invocations and key results (2026-08)

Focused companion to `PERFORMANCE_INVESTIGATION.md` (sections 14-20) and the
`minimal_gap/` harnesses. Records the exact command lines and environment used
to measure composite slab tiling in the app (`vtkMetalGPUVolumeRayCastMapper`)
and the harnesses, plus the results and the pitfalls that made earlier
readings look contradictory. Machine: M2 MBA, on battery — absolute times are
inflated ~10% vs wall power; ratios (M/GL, win/loss) are reliable.

## 1. App benchmark: `vtkMetalGLVisualComparison`

### 1.1 Canonical recipe (PERFORMANCE_INVESTIGATION.md §5.4)

```sh
./macos_metal_build.sh --resume --tests     # MUST relink the test binary after
                                            # .metal edits (stale-binary trap, §17.2)

VTK_METAL_TEST_SAMPLE_DISTANCE=0.5 VTK_METAL_TEST_JITTER=1 \
VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_MINMAX=1 \
VTK_METAL_TEST_ACCEL=0 \
build_macos_metal/bin/vtkMetalGLVisualComparison \
  --bench --backend gl --scene DICOMVolume \
  --dicom /Users/macair/Public/IMR/CTIMR/IMRToraceAddome \
  --frames 30 --reps 1
```

Swap `--backend gl` for `--backend metal`. GL is timed with `glFinish`, Metal
with `WaitForCompletion`, so numbers are comparable (GPU time).

### 1.2 Raw-path (minmax off) / slab-era recipe

The slab investigation measures the raw march with min-max acceleration off:

```sh
VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_JITTER=1 \
VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_MINMAX=0 \
VTK_METAL_TEST_ACCEL=0 \
[VTK_METAL_TEST_NUM_SLABS=1]          # 1 = true single pass; 8 = forced 8 slabs
[VTK_METAL_TEST_MARCH_VARIANT=0]      # 0 = raw scalar march; unset = 9 (default)
[VTK_METAL_TEST_CAM_AXIS=z]           # axis views; unset = oblique preset
build_macos_metal/bin/vtkMetalGLVisualComparison \
  --bench --backend metal --scene DICOMVolume \
  --dicom /Users/macair/Public/IMR/CTIMR/IMRToraceAddome \
  --frames 30 --reps 1 --size 2048x2048
```

### 1.3 Environment reference (all `VTK_METAL_TEST_*`)

| var | default | effect |
|---|---|---|
| `SAMPLE_DISTANCE` | 0.5 | mm per march step (4 = coarse SD cell) |
| `IMAGE_SAMPLE_DISTANCE` | 1.0 | image-space downsampling (1.0 = off) |
| `MINMAX` | 1 | min-max occupancy lattice skip (0 = raw path) |
| `ACCEL` | 1 | master switch for the above |
| `JITTER` | 1 | per-pixel ray phase jitter (0 = coherent rays) |
| `JITTER_BLOCK` | 1 | block-coherent jitter block size |
| `MARCH_VARIANT` | 9 | 0=raw scalar, 6=8x unroll, 7=4x, 8=8-wide harness sched, 9=48-wide inline |
| `MARCH_STEPS` | 0 | fixed iteration count (0 = frame-max bound) |
| `MARCH_CAP` | SD-based | max batch width {48,16,8} by {<2, 2-3, >=3} |
| `NUM_SLABS` | 0 | 0 = adaptive (see 3.4); else forced count, clamped [1,32] |
| `SLAB_ALIGN` | 0.95 | adaptive threshold (max\|axis dot\| >= t -> 1 slab) |
| `CAM_AXIS` | unset | `x`/`y`/`z` look along that world axis; unset = oblique preset |
| `CAM_AZ` | -60 | extra azimuth for the oblique preset |
| `CAM_DOLLY` | unset | camera dolly factor |

CLI: `--bench --frames N --reps N --warmup N --size WxH --scene DICOMVolume
--backend gl|metal --dicom DIR --perframe --gpu-mem --host-mem`.

## 2. Harness: `minimal_gap/metal_gap`

Named flags take values (`--composite 1`, not `--composite`):

```sh
# oblique (camera 0 = the app's real oblique camera, dumped from the app):
./metal_gap --camera 0 --rt 2048 --sd 4 --composite 1
./metal_gap --camera 0 --rt 2048 --sd 4 --composite 1 --slabs 8      # tiled
./metal_gap --camera 0 --rt 2048 --sd 4 --composite 1 --maccum 1 --slabs 8

# axial (camera 1 = along z):
./metal_gap --camera 1 --rt 2048 --sd 4 --composite 1 [--slabs 8] [--maccum 1]

# CT-like data instead of gradient (--data 1 = xorshift noise, the cache-hostile
# case): add --data 1
```

Camera presets: 0 = app oblique (eye (-1.495,-0.952,2.553), bounds
426.166x426.166x717.2 mm — the actual `DICOMVolume` camera dump), 1 = axial(z),
2 = coronal(y), 3 = sagittal(x), 4 = oblique45. Build:
`clang -fobjc-arc -framework Metal -framework Foundation metal_gap.m -o metal_gap`.

## 3. Key results

All app rows: DICOM IMRToraceAddome, 2048x2048, SD=4, MINMAX=0, ACCEL=0,
IMAGE_SAMPLE_DISTANCE=1.0, 30 frames x 1 rep, unless noted. "v9" = current
default march (48-wide inline); "mv0" = raw scalar march.

### 3.1 Oblique view (default camera) — the slab win is real

| env | GL | Metal single (slabs=1) | Metal slabs=8 | M/GL single | M/GL slabs8 |
|---|---|---|---|---|---|
| JITTER=1 | 52.38 | 102.08 (v9) | 38.40 (v9) | 1.95x loss | **0.73x win** |
| JITTER=1 | 52.38 | 101.31 (mv0) | 38.17 (mv0) | 1.93x loss | **0.73x win** |
| JITTER=0 | 40.60 | 62.09 (v9) | 28.59 (v9) | 1.53x loss | **0.70x win** |

- Slabs8 cut the raw-path time 2.17x (jitter off) / 2.66x (jitter on) and flip
  the M/GL ratio from ~1.5-2x loss to ~0.7x win. This is the effect the slab
  commit (a2fb556) reported as 87.7 -> 44.2 (M/GL 1.75x -> 0.87x); our
  jitter=1 numbers (102.08/38.40) reproduce that regime within battery state.
- mv0 and v9 converge under slabs at SD4 (101.31/38.17 vs 102.08/38.40): the
  slab cache fix subsumes march-scheduling differences at coarse SD.
- Jitter alone costs 1.64x on the single-pass raw path (102.08 vs 62.09) but
  only 1.34x on slabs8 (38.40 vs 28.59) — per-pixel ray phases destroy
  warp-coherent fetches; slabs restore locality.

### 3.2 Axis views (CAM_AXIS=z / y / x), jitter=0 — near-axis regression

| view | GL | Metal single | Metal slabs8 (forced) |
|---|---|---|---|
| axial z | 16.63 | 11.55 (v9) | 18.10 (**0.64x regression**) |
| axial z | — | 6.13 (v8) | — |
| y | — | 9.04 (v9) | 17.18 (**0.53x**) |
| x | — | 9.15 (v9) | 17.18 (**0.53x**) |

Near-axis views are already cache-friendly; forced slabs8 adds 8x per-pass
plumbing (depth attach, `setFragmentBytes`, draw setup) for zero cache gain.

### 3.3 Harness: tiled vs maccum (2048, SD=4, composite)

| camera | single | tiled `--slabs 8` | `--maccum 1 --slabs 8` |
|---|---|---|---|
| 0 oblique | 41.07 | 5.91 (**6.95x win**) | 50.71 (1.23x loss) |
| 1 axial | 5.84 | 3.82 (**1.53x win**) | 32.16 (**5.5x loss**) |
| 1 axial, `--data 1` noise | 9.95 | 4.37 (**2.27x win**) | — |

- The tiled mode (the app's implementation: per-pass `kStart/kEnd` re-anchor,
  no warmup) wins at every angle in the harness — even axial.
- The maccum mode (per-pass lattice warmup re-advancing the march) regresses
  everywhere, badly at axial: that warmup is per-pass overhead, the same
  category of cost that hurts the app on axis views. The harness's tiled mode
  does NOT reproduce the app's axial regression — the app's per-pass plumbing
  (depth attach, per-draw bytes, PSO state) is the missing term.

### 3.4 Size scaling of the oblique slabs8 path (jitter=0, adaptive slabs8)

512x512 17.27 / 1024x1024 20.40 / 2048x2048 29.81 / 4096x4096 70.06 ms — fits
`16.4 ms + 3.19 ms/Mpx`. The 16.4 ms floor is 8 passes of per-pass overhead;
at small sizes slabs8 is overhead-dominated (this is also why the "baseline"
and "slabs8" runs looked identical — see 4.1).

## 4. Pitfalls (why earlier readings contradicted the commit notes)

1. **`NUM_SLABS` unset = adaptive, not single-pass.** For the oblique camera
   (align 0.82 < 0.95) `AdaptiveVolumeSlabCount` returns 8, so a "baseline"
   run was already slabs8 (29.81 ms) and forcing slabs8 changed nothing
   (29.29 ms) — this produced the false "slabs do nothing" reading. True
   single-pass requires `VTK_METAL_TEST_NUM_SLABS=1` (62.09 ms).
2. **`JITTER` defaults to 1.** At coarse SD the raw path is 1.64x slower with
   jitter (102 vs 62 ms). The canonical recipe keeps jitter on; jitter-free
   A/B needs `VTK_METAL_TEST_JITTER=0`.
3. **`SAMPLE_DISTANCE` defaults to 0.5.** A "defaults" run at 2048x2048
   silently reverts to SD 0.5 (8x samples): GL 40.9 -> 159.9 ms. Always export
   `SAMPLE_DISTANCE` explicitly.
4. **zsh does not word-split unquoted variables** in loops — use `${=var}`
   (the "args not relayed" mystery; the C CLI parsing was correct all along).
5. **Stale test binary.** After editing `.metal` sources, rebuild with
   `./macos_metal_build.sh --resume --tests` (PERFORMANCE_INVESTIGATION.md
   §17.2 note); otherwise the binary embeds the old shader source.
6. **Battery power.** All numbers above measured on battery; ratios transfer,
   absolute ms do not (GL 52.38 here vs 51.0 in the doc's session, both
   jitter=1).

## 5. Open question: replacing the adaptive slab count

`AdaptiveVolumeSlabCount` (mm:384-400) avoids the near-axis regression by
dropping to 1 slab when `align >= 0.95` — but it is a view-dependent
compromise, not a fix (it gives up the win precisely on the cache-friendly
views where slabs are cheap, and it still regresses if forced). The harness
data shows the real lever: **per-pass overhead**. The tiled harness mode
(win at every angle, incl. 1.53x axial) differs from the app only in the
per-pass cost. Candidate directions:

- Cut the app's per-pass cost (depth attachment, per-draw `setFragmentBytes`,
  PSO state) so slabs8 pays everywhere; the harness's 5.91 ms oblique floor
  vs the app's 28.59 ms quantifies the available headroom.
- Float accumulation RT (RGBA16Float) to also remove the per-slab 8-bit
  quantization (correctness lead from the slab section of
  PERFORMANCE_INVESTIGATION.md).