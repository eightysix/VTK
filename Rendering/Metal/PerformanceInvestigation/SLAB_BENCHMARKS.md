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

### 3.3 Harness: single-band probe vs full composites (2048, SD=4, composite)

| camera | single | `--slabs 8` (band-0 probe) | `--maccum 1 --slabs 8` (full 8-band) |
|---|---|---|---|
| 0 oblique | 41.07 | 5.91 (**24.8 G/s**) | 50.71 (1.23x loss) |
| 1 axial | 5.84 | 3.82 | 32.16 (**5.5x loss**) |
| 1 axial, `--data 1` noise | 9.95 | 4.37 | — |

- **The non-maccum `--slabs 8` mode is NOT a composite — it is a band-0 probe.**
  `ApplySlab` bakes `clamp(zfrac, 0/8, 1/8)` for band 0 only; the shader still
  marches the full ray (avgIter 28.5 / sumIter 148M = the single-pass counts),
  with the fetch's texture-z pinned into band 0. It measures the *coherence
  ceiling* (24.8 G/s vs 3.5 G/s single-pass) on one band's worth of image
  content — 5.91 ms is 1/8 of the composite.
- **The full 8-band z-clamp composite (maccum) re-marches the full ray per
  band: 8x the samples (1.18G @ 2048 oblique) → 50.71 ms, a loss vs
  single-pass.** The z-clamp does not reduce sample count; it only makes each
  band's fetches hot. It is also an approximation (each band samples the
  band's z-slice along the ray's xy path, not the true ray), so it is not a
  faithful composite at any count.
- **The exact spatial split (`--slabt`, baked per-band constants, true short
  marches: avgIter 3.4/band) costs 0.778 ms/band @512 and 3.76 ms/band @2048**
  (lean harness shader) — short marches are latency-bound (~1-4 G/s), NOT
  cache-hot-fast. 8 bands ≈ 6.2 ms @512 / 30 ms @2048 (lean): at 2048 that is
  ~the app's current exact split; at 512 it is ~3x better (see §5).
- The `--uniformslab` variant only updates the slab uniforms in the maccum
  path; in the non-maccum path it silently runs the full march (41.9 ms @2048
  = single-pass) — harness quirk, not a measurement.
- The axial exact split (`--slabt 1 --slabs 8 --maccum 1`, measured
  2026-08-18) is 13.75 ms vs 6.21 single = **2.2x loss** — the §5.2 "regression
  gone" estimate (12-16) is falsified in the lean harness: on axial the
  single-pass fetch set is already coherent (8.3 G/s), so the split only adds
  short-march latency and per-pass fragment overhead, which are invariant to
  the split geometry. The app reproduces it (spatial slabs8 axial 19.58 vs
  single 11.56).

### 3.4 Size scaling of the oblique slabs8 path (jitter=0, adaptive slabs8)

512x512 17.27 / 1024x1024 20.40 / 2048x2048 29.81 / 4096x4096 70.06 ms — fits
`16.4 ms + 3.19 ms/Mpx`.

The 16.4 ms "floor" is NOT per-pass plumbing: the harness single-pass (zero
plumbing — one draw, no depth attach, pure GPU time) shows the same floor at
small sizes (16.29 @400, 18.42 @512), and axial has none (1.83 @512, 3.82
@1024, 11.55 @2048 — clean linear scaling). The floor is **cold-fetch DRAM
latency**: at small RTs the oblique rays' fetch set spans the whole 470 MB
volume with no L2 reuse (0.5 G/s), while at 2048 the neighboring rays' wedges
overlap enough to hit L2 (2.35 G/s) and axial rays are inherently coherent
(8.3 G/s). Latency-bound stalls, not draw-call overhead; the app's per-draw
CPU cost (`setFragmentBytes` + 3-vertex draw x8) is ~0.05-0.2 ms total.

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
7. **Harness readback = iteration bytes, not the image.** `meanB`/`avgIter`/
   `true` decode the iteration counter the shader stores in the RT (B,G,R
   bytes); `meanB` is the count's low byte, so a band-0 probe and a full
   composite show identical readbacks as long as the march length matches.
   `--slabs 8` non-maccum keeping avgIter 28.5 (the single-pass count) is the
   tell that it is a 1-band probe, not a true split; `--slabt`'s avgIter 3.4
   is the true split's short-march signature.
8. **Visual diff requires `JITTER=0`.** GL and Metal draw independent
   per-pixel jitter phases, so with jitter on the thresholded GL-vs-Metal
   error is ~0.82 regardless of the composite's correctness; with jitter off
   every slab config (ray-fraction and spatial) measures 0.000. The
   "thresholded error stays 0.000 in all configs" claims in
   PERFORMANCE_INVESTIGATION.md §19 hold for jitter-free runs.

## 5. Problem and solution: replacing the ray-fraction slab split

### 5.1 Problem

The current exact split (`fc_slabMode`: per-fragment sample-index partition
`kStart/kEnd`, mm:4162-4173) is bit-exact but its passes are **cache-cold at
small RTs and on near-axis views**:

- The per-pass fetch set is a ray-space *wedge* (each ray's z-range differs
  across the frustum), so at 512 the passes' fetches still span the whole
  volume → 0.54 G/s, barely better than single-pass (0.40 G/s): slabs8 17.27
  vs single 23.35 (1.35x only, vs 2.17x at 2048).
- The split shortens every march to ~maxSteps/8 ≈ 4 steps; short marches are
  latency-bound (~5 G/s ceiling) → the near-axis regression (axial 11.55 ->
  18.10, y/x 9.04/9.15 -> 17.18): axial single-pass is already coherent
  (8.3 G/s) and the split only adds per-pass latency.
- `AdaptiveVolumeSlabCount` (mm:384-400) papers over the regression by
  dropping to 1 slab at align >= 0.95 — a view-dependent compromise, not a
  fix (it gives up the win exactly where slabs are cheap).

The earlier "per-pass overhead" framing (28.59 app vs 5.91 harness = ~2.8
ms/pass) was an artifact: 5.91 is a 1-band probe (§3.3). The real per-pass
plumbing is negligible; the real problems are fetch locality (512/1024) and
short-march latency (axial).

### 5.2 Solution: exact spatial slabs (uniform world-plane bounds)

Replace the per-fragment index partition with **uniform world-z plane bounds**
(the harness's `--slabt`/ApplySlabT scheme, exact): per pass p the fragment
marches only `t ∈ [t_lo, t_hi]` where t_lo/t_hi are the ray-parameter
intersections with the planes `z = zmin + (p/8)(zmax-zmin)` (normalized
volume space), anchored with the same ceil-lattice arithmetic as the current
split so the index union still tiles the full ray bit-exactly (the harness
kEndT alignment: "consecutive ceil() calls can never disagree by an index").

Every pass' fetch set is then a thin flat band of the volume (all rays'
samples in the pass have z inside the same interval) → L2/L1-hot at any RT
size, and near-axis views become the best case instead of the worst (their
bands are the thinnest in ray terms). SlabInfo gains the plane basis (axis +
start/end, or the 8 plane positions); single PSO + per-pass
`setFragmentBytes` stays.

Measured expectations (lean harness shader, SD=4, composite, jitter=0):

| view | single-pass | current split | spatial bands (est) |
|---|---|---|---|
| oblique 512 | 23.35 | 17.27 | ~6-9 |
| oblique 1024 | 37.27 | 20.40 | ~12-18 |
| oblique 2048 | 62.97 | 28.59 | ~30 (par; short marches latency-bound) |
| axial 2048 | 11.55 | 18.10 (regress) | ~12-16 (regression gone) |

**The axial estimate was falsified (2026-08-18).** The app's spatial split on
axial 2048 measures 19.58 ms vs 11.56 single-pass (0.58x), and the lean
harness's exact split is 13.75 vs 6.21 (2.2x loss, §3.3): short-march latency
and per-pass fragment overhead are invariant to the split geometry, and on
near-axis views single-pass fetch locality is already perfect — there is no
cache win to harvest. No K>1 keeps axial at single-pass speed (the penalty is
monotonic in K: 1.13x@2, 1.34x@4, 1.65x@8 in the app, §6). The spatial mode's
remaining promise — the oblique small-RT win (512/1024) — is unverified:
only the 2048 oblique ray-fraction and axial spatial cases have been run.

The 2048 oblique stays par with the current split (both ~28-30); the wins are
at small RTs and the near-axis views — the exact-space ceiling is ~5 G/s for
split marches, and the 24.8 G/s coherence ceiling belongs to the (non-exact)
z-clamp probe, not to any faithful composite.

## 6. Correctness fix and fixed-K design (2026-08-18)

The ping-pong slab path had a composite-alpha bug: `accumulatedOpacityOwn`
was only accumulated in the baseline march loop, so the default v9 (and
v8/v6/v7) paths emitted `finalColor.a = slabFar.a` (or 0 on fullscreen
passes) — the pass's own opacity was dropped and Phase 3b's over-blend left
the background un-attenuated (washed-out image). Fixed by using the global
over-chain alpha (`accumulatedOpacity`) as the composite alpha. Also fixed:
`slabInfo.w == 1.0` (the documented spatial selector) fell into a debug
return — every value in (0.5, 4] was a debug encoding, so the clean spatial
split was unreachable; w == 1.0 now falls through to the march. The spatial
entry guard now fires only for genuinely grazing rays (`tEnd <= firstT`),
removing a double-composite of sample 0 at band boundaries, and the slab path
uses a 1-sample pipeline independent of window MSAA.

Re-measurement (same session, 2048x2048, SD4, minmax off, accel off;
thresholded GL-vs-Metal error 0.000 in every row — jitter must be 0 for the
visual diff, pitfall §4.8):

| config | GL | single | slabs=2 | slabs=4 | slabs=8 | slabs=16 |
|---|---|---|---|---|---|---|
| oblique (adaptive = 8) | 42.74 | 60.92 | 37.27 | 25.25 | 24.55 | 31.29 |
| axial z (forced) | 15.04 | 11.56 | 13.11 | 15.53 | 19.03 | — |
| oblique, jitter=1 | 57.81 | — | — | — | 30.88 | — |

Design conclusions:

- **Oblique bottoms at K=8; K=16 regresses** (per-pass overhead exceeds the
  locality gain once the working set is L2-resident). **K=4 captures the full
  oblique win** (25.25 vs 24.55, within noise).
- **The axial penalty is monotonic in K** (1.13x@2, 1.34x@4, 1.65x@8): no
  fixed K>1 keeps axial at single-pass speed. **K=4 puts axial at GL parity**
  (15.53 vs 15.04) while keeping the oblique win — the best "never loses to
  GL" fixed point. K=8 has the tightest all-orientation frame band
  (19.0-24.6, 1.3x spread) — the most pacing-constant choice.
- **Adaptive slab count (align >= 0.95 -> 1 slab) removes the axial
  regression but creates a frame-time discontinuity at the threshold during
  rotation** (24.6 <-> 11.6 ms jumps). A fixed K trades the axial cost for
  constant pacing; the choice is K=4 (axial GL parity) vs K=8 (tightest
  band).
- The app pays ~5 ms/frame of fixed plumbing vs the lean harness (the delta
  is the same at K=1 and K=8); recoverable via a back-to-front single-RT slab
  path (no ping-pong feedback fetch) and box geometry instead of a fullscreen
  quad (kills out-of-footprint fragments). Estimated axial K=8 19 -> ~13-15,
  i.e. near single-pass parity; below parity is impossible (sample count and
  fetch behavior are unchanged by splitting on axial).