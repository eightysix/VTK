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
| `GPU_OPTIMIZED_CONTENTS` | 0 | volume `MTLTextureDescriptor.allowGPUOptimizedContents` (0 = NO = the lag_repro root-cause fix, default; 1 = legacy YES) |

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
### 6.1 Next steps: cutting the axial per-fragment cost

Why axial slabs8 is slow (19.03 vs 11.56): splitting does not change the
total sample count (identical marches, 1.25 samples/pass at K=8) and axial
fetch locality is already perfect (8.3 G/s single-pass), so slabs add only
per-pass fragment overhead. Each slab fragment pays, per pass: ray-setup
prologue (arithmetic), a **ping-pong feedback texture fetch** (DRAM-latency
round trip — the 16.8 MB RT exceeds L2), the short latency-bound march, and
depth test + RT write. The lean harness (13.75 ms, no ping-pong at all) shows
~5 ms of this is app plumbing.

Cuttable, in order of expected value:

1. **Back-to-front single-RT blending instead of ping-pong.** Draw far-to-near
   with `(ONE, ONE_MINUS_SRC_ALPHA)` over-blend into one cleared offscreen RT
   (the harness and the pre-ping-pong `DrawBlocksFullscreen` design; the blend
   unit does the composite, no `texture(15)` fetch, half the texture traffic).
   Trade-off: loses the front-to-back opacity latch, which currently makes
   oblique passes 2-7 nearly free; on axial the latch never fires (passes stay
   transparent) so this is pure win on axial, a small loss on oblique.
   **Measured 2026-08-18 (harness `--feedback`: in-shader read of the previous
   pass's RT on alternating textures — the app's exact ping-pong fetch +
   combine, vs `--maccum 1`'s hardware single-RT blend; interleaved, 2048²,
   SD 4, jitter 0, GPU throttled mid-session so absolute numbers shifted, the
   paired deltas are the signal):**

   | config | single-RT blend (maccum) | ping-pong (feedback) | delta |
   |---|---|---|---|
   | axial slabt8 | 15.26 / 15.22 / 15.77 | 16.42 / 16.19 / 16.88 | **+1.1 ms** |
   | oblique slabt8 | 86.4 / 87.0 / 86.4 | 87.2 / 87.8 / 85.7 | ~0 (noise) |

   The fetch itself is cheap (~1 ms axial, ~0 oblique — the RT load/store
   traffic per pass is identical in both designs). So the §6.1 "~5 ms app
   plumbing" (app 19.03 vs harness 13.75) is **not** the feedback fetch: it is
   the rest of the app plumbing (per-pass uniform/viewport setup, MSAA PSO
   switching, the box-geometry draws, depth). Swapping the app to single-RT
   blending is worth ~1 ms of the 7.5 ms axial regression — not worth the
   surgery on its own. (The latch-loss concern is moot: the latch is an
   in-pass `acc` saturation break, identical in both designs — oblique
   maccum ≈ oblique feedback within noise.)
2. **Box geometry instead of the fullscreen quad.** The direct path
   rasterizes all 33M fragments per pass; the volume's screen footprint is
   only ~60-80%, so the out-of-footprint fragments run the full prologue and
   write 0, K times. Six back-face triangles cull them at the rasterizer.
   **Measured 2026-08-18 (harness `--scissor`, footprint-probe A/B, interleaved
   min of 2, 2048², SD 4, jitter 0): the invalid fragments were already nearly
   free — culling them gains ~0-4%:**

   | config | no scissor | scissor | footprint |
   |---|---|---|---|
   | axial single | 5.795 | 5.802 | 23.0% |
   | axial slabt8 maccum | 12.903 | 12.644 | 23.0% |
   | oblique single | 41.706 | 40.158 | 57.8% |
   | oblique slabt8 maccum | 21.172 | 20.643 | 57.8% |

   (Footprint rects measured by a clean single-pass probe readback, 983² at
   +530+535 axial, 1621x1495 at +303+145 oblique — the AABB of the marched
   pixels, which is what the box geometry would rasterize.) So this is **not**
   the axial regression's cause: the regression is per-pass fixed work on the
   *valid* fragments (prologue + box test + RT write × K on ~1M marched
   pixels), which geometry culling cannot touch.
3. **Lean slab entry point** (skip minmax/lighting setup the slab passes
   don't use). Unmeasured in the harness (the harness has no minmax/lighting
   to skip); the app-side delta is part of the ~5 ms non-fetch plumbing above.

All three §6.1 levers are now measured (box geometry ~0-4%, single-RT ~1 ms,
lean entry unquantified but bounded by the same plumbing analysis). Together
they cannot close the 7.5 ms axial K=8 regression: the residual is the
per-pass fixed fragment cost × K on the valid fragments — structural to the
split. The design decision therefore stands on §6: **fixed K=4** (axial GL
parity, full oblique win) or **fixed K=8** (tightest frame band); adaptive is
rejected for frame-pacing.

Verification: re-run the §6 table (oblique + axial, K=8, jitter=0) after each
app change, keeping thresholded error 0.000.

## 7. The single-pass root cause, ported to the app: `allowGPUOptimizedContents = NO` (2026-08-18)

`lag_repro` pinned the raw single-pass lag to the volume texture's
`MTLTextureDescriptor.allowGPUOptimizedContents` (default YES: the GPU's
lossless re-swizzle of private 3D textures taxes incompressible per-texel data
~1.8x; NO = uncompressed layout, GL parity — see `lag_repro/README.md`). The
mapper now sets **NO by default** on the volume texture
(`VolumeGPUOptimizedContents()`, all four creation sites in
`CreateGlobalVolumeTexture`/`UpdateVolumeTexture`), with
`VTK_METAL_TEST_GPU_OPTIMIZED_CONTENTS=1` restoring the legacy YES for A/B.

In-app A/B (DICOM IMRToraceAddome, 2048x2048, battery, interleaved; opt1 =
legacy YES, opt0 = default NO). NOTE: the code default march variant is
**mv0** (raw scalar march) right now — `VolumeMarchVariant()` returns 0 with a
TEMP-REPRO comment (revert to 9); unset `MARCH_VARIANT` rows below are mv0,
and the mv9 row is pinned explicitly. The flag's effect is march-independent:

| config | march | GL | opt1 | opt0 | gain | M/GL opt0 |
|---|---|---|---|---|---|---|
| single oblique, SD4 (jitter=0) | mv0 | 41.66 | 61.5 | **43.9** | **1.40x** | 1.05x |
| single oblique, SD4 (jitter=0) | mv9 | 41.66 | 60.8 | **46.5** | **1.31x** | 1.12x |
| single oblique, SD0.5 (jitter=0) | mv0 | 154.77 | 294.8 | **115.2** | **2.56x** | **0.74x win** |
| single axial, SD4 (jitter=0) | mv0 | 15.75 | 11.4 | 11.0 | 1.04x | 0.70x |
| slabs8 adaptive, SD4 (jitter=0) | mv0 | 39.31 | 24.4 | 25.2 | ~neutral | 0.64x |
| production defaults (SD0.5/jitter=1/minmax=1) | mv0 | 161.33 | 194.5 | 187.6 | 1.035x | 1.16x |
| 400x400 SD0.5 (jitter=0) | mv0 | — | 39.01 | 23.16 | 1.68x | — |

- The flag is the single-pass root cause in-app too: SD4 1.40x, SD0.5 2.56x —
  matching the harness `--noopt` matrix (noise sd4 1.8x, sd0.5 2.9x on the
  lean shader; the app's minmax/slab plumbing shaves some of the headroom).
- With the flag off, the oblique single-pass raw path goes from 1.47x loss to
  **1.05x parity at SD4 and a 0.74x win at SD0.5** — the deficit is gone
  without slabs.
- The flag is **neutral where the working set is already small**: axial
  (coherent single-pass, ~4%) and the slabs8 path (per-pass sets are
  cache-resident; within noise, opt1 slightly ahead by ~2-3%). It is the
  single-pass/large-footprint lever, not a general one.
- The production path (minmax + jitter + adaptive slabs at SD0.5) gains only
  ~3.5%: the slab tiling already keeps the working set resident, and minmax
  skips cut the samples — the layout tax applies to the full-footprint
  passes only.
- Visual parity: GL-vs-Metal thresholded error is byte-identical before and
  after the flag change (16.033 @400/SD0.5 — a pre-existing baseline
  divergence of the current slab/ping-pong path, unchanged by the layout;
  the flag itself is lossless).
- The layout flag is the right default for CT/DICOM data (incompressible).
  `VTK_METAL_TEST_GPU_OPTIMIZED_CONTENTS=1` keeps the legacy swizzle for
  compressible payloads (smooth atlases, distance fields) where the YES tax
  does not exist.

### 7.1 Low-res recheck: the 400x400 "gap" was a march-variant artifact (2026-08-18)

The 400x400 raw cell looks like a loss under the current mv0 default, but is a
**win under the production mv9 march** — the earlier "1.19-1.24x loss" readings
were the TEMP-REPRO mv0 pin, not Metal. 400x400 raw (ACCEL=0/MINMAX=0,
SD0.5, jitter=1, opt off, slabs off, 3 interleaved rounds):

| march | Metal | vs GL ~49.4 |
|---|---|---|
| mv0 (temporary default) | ~61.1 | 1.24x loss |
| mv9 (production 48-wide) | ~30.9 | **0.63x win** |

mv0's serial fetch→consume loop exposes the DRAM-latency floor at small RTs;
mv9's 48-wide in-flight batches hide it (~2x at this cell). At 2048 the roles
invert slightly (mv0 raw SD4 43.9 vs mv9 46.5, jitter=0). Same story at 400/SD4:
mv9 17.8 vs GL 19.0 (0.93x) vs mv0 21.6 (1.13x).

### 7.2 Jitter costs Metal ~2x what it costs GL

Jitter=1 inflates the 2048/SD4 single-pass raw cell asymmetrically (same run
session): GL +27% (41.7 -> 53.1), Metal +57-59% (mv0 43.9 -> 69.8, mv9 46.5 ->
72.8). Jitter scatters per-warp fetch addresses, multiplying the effective
working set past SLC capacity; Metal's read path degrades more than GL's
driver-internal tiling. Hence parity (jitter=0) vs 1.31-1.37x (jitter=1) at
that cell, and why the visual diff and the harness parity cells are jitter=0.
Full fair matrix: PERFORMANCE_INVESTIGATION.md §21.

## 8. Revisit: slab win is gone on the current stack (2026-09-05)

Re-ran the §1.2 raw recipe plus production/minmax-on and axis views at
`8a99a9ceb6` (density-gated fine shade cap + partition seam fix; i.e. post
layout-NO, transpose-default, W1PRE/dense, lean/shade caps, cull removal).
M2, **AC power** (absolutes ~10% below the battery-era docs; ratios transfer),
`--frames 20 --warmup 5`, ABBA/repeat-confirmed (first-run-of-new-config rows
can carry PSO-compile inflation — deltas below are repeat-confirmed).
Thresholded error 0.000 in every row.

Recipe (swap `NUM_SLABS` 1↔8; axial adds `VOLTRANSPOSE_AXIS=y` + `CAM_AXIS=z`):

```sh
R="VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 \
VTK_METAL_TEST_MINMAX=0 VTK_METAL_TEST_ACCEL=0 VTK_METAL_TEST_MARCH_VARIANT=9 \
VTK_METAL_TEST_JITTER=0"
env $R VTK_METAL_TEST_NUM_SLABS=1 build_macos_metal/bin/vtkMetalGLVisualComparison \
  --bench --backend metal --scene DICOMVolume \
  --dicom /Users/macair/Public/IMR/CTIMR/IMRToraceAddome \
  --frames 20 --warmup 5 --size 2048x2048
```

### 8.1 Raw path (SD4, MINMAX=0, DICOM IMRToraceAddome) — inversion

| config | single (K=1) | slabs=8 | delta | GL | M/GL single | M/GL slabs8 |
|---|---|---|---|---|---|---|
| 2048 oblique J0 (mv9; mv0 identical) | 32–33 | 48–50 | **+48–50% loss** | 54.3 | 0.59 win | 0.88 |
| 2048 oblique J1 | 33.1 | 49.0 | **+48% loss** | 75.8 | 0.44 win | 0.65 |
| 1024 oblique J0 | 10.2 | 14.3 | +40% loss | — | — | — |
| 1024 axial-z J0 (y-transpose) | 26.3 | 38.8 | +48% loss | — | — | — |
| 1024 coronal-y J0 (y-transpose) | 19.2 | 30.8 | +60% loss | — | — | — |

§3.1-era (battery): single 102.1 / slabs8 38.4 J1 oblique (2.66x slab win).
Now inverted on both marches (mv0 rows: single 31.8/32.4 vs slabs 47.6/49.1 —
march-independent, jitter now +2–3% on Metal vs +40% on GL).

K-sweep, 1024 oblique J0, default transpose: K=1 9.98 / K=2 11.92 /
K=4 13.06 / K=8 14.29 / K=16 16.91 — **monotonic loss, no K>1 wins**
(was: bottom at K=8, K=4 keeps the full win). Spatial mode
(`SLAB_SPATIAL=1`) 14.66 vs ray-fraction 14.25 — same loss, no rescue.

### 8.2 Production path (SD0.5, MINMAX=1, J1, 1024, mv9)

DICOM single 40.5 vs slabs8 73.2 (**+81% loss**); adaptive `NUM_SLABS=0`
75.7 (picks 8 on oblique, inherits the loss). NIFTI FLASH25 single 14.9 vs
slabs8 16.4 (+10% loss).

### 8.3 Mechanism and guidance

The single-pass cache pathology slabs fixed is now fixed at the source —
layout-NO + transpose short-axis-in-depth killed Metal's jitter tax
(single J0→J1 now +3% vs §7.2's +57–59%), leaving only slabs' per-pass
overhead (ping-pong passes, depth clears, feedback traffic, prologue ×K)
with no cache gain to fund it. Bisect at 1024: transpose helps single −31%
vs slabs −19% (contributory, not sole — transpose-off is still single 9.9
vs slabs 14.3); layout flag and `W1PRE=0` now neutral. The two latest
commits are orthogonal: the density-gated fine cap applies to single and
slab passes alike (no retune needed), and the seam fix is `SetPartitions`
grid traversal, a different mechanism from `NUM_SLABS` compositing.

Guidance: default (`NUM_SLABS` unset → 1, single pass) is already optimal
on all measured configs — no code change. `NUM_SLABS=0` adaptive currently
picks 8 on oblique and pays +40–80%; keep the path as an opt-in fallback
for unmeasured huge-volume cases, do not default to it.

Confound note: with the default X-depth transpose, world-z axial single is
89 ms vs 26 ms with `VOLTRANSPOSE_AXIS=y` — axis-view absolutes now depend
on transpose orientation (part2 §38 territory). Slabs lose under either.