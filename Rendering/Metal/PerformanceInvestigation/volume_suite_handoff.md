# Volume Suite Handoff — RenderingVolumeCxx-Metal 96/97

Status as of `fac8ac80e4` on `metal-cleanup-1` (M2, `arm64 Release`,
`./macos_metal_build.sh --resume --tests`).

## Current state

| Suite | Result | Sole failure |
|---|---|---|
| `RenderingVolumeCxx-Metal`, default march (mv0) | 96/97 | `TestGPURayCastCameraInsideTransformation` 0.0912 (near-miss) |
| Same, `VTK_METAL_TEST_MARCH_VARIANT=9` | 96/97 | same test 0.0913 |
| NIFTI harness parity 512 | `2999 thr 0.000` exact | — |

Run it with:

```sh
python3 Rendering/Metal/Testing/metal_ctest_report.py -p RenderingVolumeCxx-Metal -j 8
# mv9 arm:
VTK_METAL_TEST_MARCH_VARIANT=9 python3 Rendering/Metal/Testing/metal_ctest_report.py -p RenderingVolumeCxx-Metal -j 8
```

The script inherits parent env (`Popen` without `env`), so perf knobs are a
prefix. Only `MARCH_VARIANT=9` is non-default-optimal; everything else optimal
is already default (dense auto, `W1PRE` ON, blocks ON, transpose argmin). Do NOT
add `VOLTRANSPOSE_AXIS=y` (workload knob, loses raw axis-y) or `FRAG_BATCH=16`
(regresses lean-coarse, see `perf_investigation_part2.md:26.9`).

## What got fixed this session (all landed)

1. **Depth/camera-inside PSO dead-strip** (`vtkMetalGPUVolumeRayCastMapper.mm`,
   `714fd2451f`): `fc_useDepthTexture`/`fc_useCameraInside` came from bare env
   presence (default OFF) while the uniforms said ON → volume+polydata and
   4× camera-inside tests failed. Now auto-derived per frame from live state
   (occlusion texture + actors present for depth; `IsCameraInside` for camera)
   with tri-state env override (`VTK_METAL_TEST_DEPTH`,
   `VTK_METAL_TEST_CAMERA_INSIDE`: unset=auto, bare/nonzero=on, `0`=off).
   Volume-only scenes keep the §18 diet exactly (NIFTI key `0x240004`,
   ABBA tie vs `DEPTH=0`).
2. **Auto-dense exclusions** (same commit): bypass never fires for 2D
   transfer functions or multivolume inputs (1D component-0 sampling cannot
   measure their occupancy). NIFTI still classifies dense (95.6%).
3. **mv9 inverted blanking** (`MetalShaders.metal` `MV9_COMPOSITE`/
   `MV9_COMPOSITE_LOOP`): read only `.r`, skipped when `< 0.5` — inverted
   polarity on the wrong channel. Now mirrors mv0 (mode-selected `.x`/`.y`,
   `> 0`). Fixed `UniformGridBlanking`, `GhostArrays`×2 and, unexpectedly,
   `Transfer2DYScalars` + `MultiBlockMapper` (both carry blanking).
4. **mv9 MIP/MinIP ignores binary mask**: mask skip existed only in the
   composite/additive/label branches; mv0 skips upstream. Added the binary-mask
   early skip for blend modes 1/2 in both macros. Fixed `MIPBinaryMask`.
5. **Gradient-opacity magnitude normalization** (`2233ef1273`): divisor
   `GradientOpacityMax` carried `/avgSpacing`, which GL does not have (GL folds
   spacing into its gradient aspect ratio). Invisible at unit spacing — which
   is why every green gf scene stayed green. This test: 0.191 → 0.103.
6. **1024-entry gradient-opacity LUT** (`fac8ac80e4`): match GL's default
   width (was 256). Zero metric movement (0.0912 both) — kept for parity.

## The residual: CameraInsideTransformation 0.09

**Scene** (unchanged since 2016, GL baseline passes at 0.0065): headsq/quarter
resampled to 512³ (USHORT, spacing 0.39/0.39/0.27), bone TF + gradient opacity
`(0,0),(90,0.5),(100,0.7)` + shade, volume rotated Rx180/Ry85/Rz55, camera
inside, one wheel-dolly replayed before compare.

**Exonerated** (each toggled in isolation, metric bit-identical): march
variant, minmax/dense/blocks (`DENSE=0/1`, `MM_BLOCKS=0` all 0.191 pre-fix),
W1PRE, transpose, GPU contents, jitter on/off, clipping-range refresh,
central-vs-forward vs nearest gradient taps (`GRAD4`, `GRAD_NEAREST`),
lighting direction (shade-on + gradop-off matches at 0.000%), TF cull (safe
by construction: culled samples carry <2% alpha), pow skip, resample filter
(reader-direct also diverges), rotation (64³ rotated matches at 0.015%),
interaction replay (final cameras bit-identical both backends).

**Narrowed to gradient-opacity magnitude on smooth sub-unit-spacing data**
(`0.19 → 0.09` across the `GradientOpacityMax` sweep: `0.5/avg` 0.191,
`0.5` 0.148, `0.25` 0.103, `0.125` **0.091**, `0.0625` 0.093) with lighting,
march, TF, texture content/width/sampling all verified matching. Gain scaling
is exhausted (broad floor at ~0.09); the remainder is offset/structural, not
gain — a pure scale error would zero out. Suspects, in order: (a) additive
offset in sampled magnitude, (b) sub-texel sampling-position difference in
the steep 90→100 band (2–9 entries depending on width), (c) half-precision
roundoff in `half sPX` gradient taps on smooth fields (visible only where
adjacent-texel differences approach half-ulp; `GRAD_FLOAT` was neutral, so
this ranks last).

**Recommended next step:** per-pixel magnitude-field comparison, not more
knobs — port the metal-ios update-63-style attribute dumps (GL_RAY vs STEP
`grad.w` fields) and diff the magnitude distributions directly; a constant
shift confirms (a), band-localized error confirms (b). The `metal-ios` branch
additionally carries the update-69/70 march-exit fixes (`e40517af16`,
`6032ed6015`, multi-thousand-pixel wins on constant-scalar volumes) NOT in
this branch — worth a look if the mag fields match but exits differ.

## Methodology pitfalls hit during this work (do not repeat)

- **Compare against the baseline, not metal-vs-GL on modified scenes.**
  Modified scenes (gf off/flat/linear, no rotation, no replay) can match
  across backends for vacuous reasons (uniform modulation, saturation,
  both-black renders — actually VIEW the PNGs).
- **Direct-binary runs lack `VTK_TESTING_IMAGE_COMPARE_METHOD=TIGHT_VALID`**,
  which ctest sets. Metrics from the two harnesses are incomparable
  (e.g. 5756 vs 0.38 on byte-identical PNGs).
- **Shared-TF probes can't separate magnitude from texture bugs** — both
  backends read the same table. Only saturation-free unsaturated TF shapes
  are informative, and this scene saturates hard.
- **Keep a clean 2×2 when bisecting**: several mid-investigation numbers were
  confounded by leftover test edits (once even flipping the conclusion).
  `git diff --stat -- Rendering/Volume/` before every measurement run.
- **Bit-identical metrics across runs are the norm** (deterministic stack);
  any wobble in the 4th decimal across identical configs is a red flag for
  stale binaries, not noise. Always confirm the rebuild compiled (`[1/2]
  Building`, not `no work to do`).

## Resolution (97/97, Sep 5 2026)

`TestGPURayCastCameraInsideTransformation` now passes at **0.00716**
(GL itself: 0.00646 — interpolator floor). Suite is **97/97 on mv0 and
mv9**. Root cause was TWO compounding bugs, not gradient magnitude:

1. **0.02 shading cull** (`MetalShaders.metal`, 4 sites): Metal gave
   ambient-only light to samples with opacity ≤ 0.02 (perf opt from
   `perf_investigation_part2.md §13.3`, designed for NIFTI FLASH25).
   GL lights every positive-alpha sample (`a > 0.0`). Dimmed all
   dim-but-visible boundary samples to ambient — isolated by
   env-gated `NOGRADOP` (matches, 0.72) vs `NOSHADE` (diverges, 33)
   probes plus a dim-TF probe (16.2 → 0.14 mean|d| with the cull off).
   Removed (GL parity).
2. **Gradient divisor** (`vtkMetalGPUVolumeRayCastMapper.mm`): `2233ef1273`
   removed the `/avgSpacing` term per a sweep optimum (0.125) that was
   **confounded by bug 1** — over-opacity masked cull-darkness. CPU
   replica on ground-truth data plus dumped uniforms proved the parity
   form `0.5·range/(norm·avgSpacing)` equalizes magnitudes exactly
   (ratio 11.330 constant); with the cull fixed it passes. Either fix
   alone still fails (0.199 / 0.089). Restored (reverts `2233ef1273`'s
   formula; keeps its sweep record above as a confounding warning).

Also ported along the way (principled GL parity, inert on this metric
but kept): gradient-opacity LUT `RGBA8Unorm→R32Float` (metal-ios update 3),
gradient taps + TF sampling in float32 (updates 9/10). Rejected after
measurement: march-exit 69/70 (already present here — CTP bounds exit in
all loops, counts match ±8/460), fullscreen-vs-proxy path (identical to
0.07), R32F-vs-8-bit LUT width (zero movement twice).

## Addendum: float promotions reverted (half suffices)

The float32 items above were reverted after a direct experiment: the
half-precision stack (8-bit grad LUT, half TF/taps/sampling) with ONLY the
two root-cause fixes passes the target at **0.00693** (vs 0.00716 float,
GL 0.00646) and the suite at **97/97 on mv0 and mv9**. The promotions are
not load-bearing here, and half is the codebase norm (accumulation/color
intermediates; volumes stay `R16Unorm`/`R32Float`, gradient magnitudes
stay float). Kept reverted for consistency; revisit only with a failing
metric in hand (steep-ramp amplifier scenes outside the suite remain the
known hazard class for 8-bit LUT quantization).

## Perf recheck (cull removal vs `part2 §13.3`)

NIFTI 1024 mv9 bench (`vtkMetalGLVisualComparison`, FLASH25):
with-cull 8.52 ms/f (8.92/8.37/8.26) vs without-cull 8.55 (8.05/8.58/9.03)
at SD0.5; SD4 9.38 vs 8.78; 512 thr 3.29 vs 3.30 (budget <5). The §13.3
~8% win evaporated under later march restructuring (W1PRE/dense/rolled
loops) — on the current stack the cull is perf-neutral within DVFS noise
and correctness-negative. DICOM is provably unaffected (binary 0/1 TF
never enters the culled band). Do not re-add a shading threshold without
re-measuring the volume suite.

## Historical note (97/97 claim)

`d0f4a31066` (Aug 15) closes `Rendering/Metal/Testing/failures.txt` with
"97 pass / 0 fail / 0 abort", but the same file lists this test at 0.1676
with no documented fix afterward, and 11 of its 12 listed image failures
were genuinely fixed only later (this session included). The metal-ios
branch additionally carries an undiagnosed variant family for this exact
test (`...NoShade`, `...NoGradOp`, `...NoTransform`, `...NearPlaneTiny`,
`...SampleDist0_25/0_5`, `...MaxIP`, ...). Treat the Aug-15 line as
aspirational; the measured trajectory is 12 image failures → 1 (this test,
0.1676 → 0.09).

## Follow-up session (Sep 5 2026, `metal-cleanup-1`): cull-removal fallout

Recheck after the two root-cause fixes: suite still **97/97 on mv0 and
mv9** (`CameraInsideTransformation` mv0 0.00693 / mv9 0.00696, GL floor
0.00646). Main-matrix perf vs the thr-clean baseline: NIFTI 1024 SD4
6.93/8.64 0.80, SD0.5 16.40/18.28 0.90; 2048 SD4 9.52/11.75 0.81, SD0.5
48.69/52.53 0.93; DICOM-y 0.14/0.20; all `<1`. Fine NIFTI returns to the
pre-cull level by design (the −42% came from skipping work GL does);
coarse is neutral-to-better, DICOM provably untouched (binary TF).

### 1. Density-gated fine shade cap (landed, zero margin)

Cull removal restored full 6-fetch+pow shade cost, flipping the fine
optimum back to narrow (forced-F1 stable best-of-round; `F2/F8/F16` tie
±3%). Flat `16→2` ABBA-tied, so the landed form gates on the occupancy
auto-flag instead — `shadeCap = fine ? (dense ? 1 : 16) : 2`
(`MetalShaders.metal`, no new PSOs, both bits already key the cache):
NIFTI-fine 14.49 (−5%, equals forced-F1), coarse/DICOM-lean ties,
Skin-fine-shaded-sparse 12.56 vs F16 12.21 (+2.9%, 2v2, watch item).
Parities identical-class (2986/0.000, 2884/0.067, DICOM 0.000); suite
97/97 ×2; bit-exact on mv0 (0/96 moved — caps live only in the mv9
ladder). Rationale in situ: wide amortizes the 48-walk preamble where
leaps fire (sparse), narrow wins where the march is raw (dense bypass).

### 2. `GRAD_NEAREST` vs `GRAD4` delta recheck (measured, not landed)

NIFTI 512: `GRAD_NEAREST` SD4 1.889 / SD0.5 1.523, `GRAD4` 2.458 / 2.042
(all `<5`); stacked F1+`GRAD_NEAREST` 1.542 (no stacking penalty). Full
suite: `GRAD_NEAREST` **97/97** (23 movers, max|d| 0.034, mean 0.0016 —
but `Cropping` 0.0468, `OrientedVolume` 0.0432, `VolumeUpdate` 0.0420 sit
within 0.01 of the line: landable, creates watched tests) vs `GRAD4`
**93/97** (4 near-miss fails 0.052–0.091, 32 movers, mean 3.5× nearest:
not landable as a default; fine-gating wouldn't save it — failures are
not fine-only scenes). Target test: nearest bit-identical (0.00693),
`GRAD4` 0.00786. Deferred: margin-eating by definition, after zero-margin
items.

### 3. Partition seam fix (landed, `thr 2.93 → 0.000`)

`§14`'s ±1-sample theory was wrong: single-vs-partitioned metal-metal
diff is 48% px global, RAW (`MINMAX=0/ACCEL=0`) is bit-identical
3803.779/2.927 (occupancy exonerated), partitioned `JITTER=0` explodes to
**87.97** (shade-off: 1.22). Mechanism: `marchSegment` passed the raw
camera-relative jitter with `checkBounds=false`, anchoring the lattice at
the camera (`jitter+k·step`) while every other path (and GL) anchors at
the volume entry (`entry+jitter+k·step`) — up to a full step systematic
shift, dithered to 2.93 by jitter. Fix: `latticePhase = entry+jitter`
per pixel + entry-guard for the J0 `ceil(−1.0)=−1` knife-edge
(`MetalShaders.metal` `marchSegment`/grid loop/`firstT`; sole
`checkBounds=false` user, single-brick paths byte-identical).
Validation: `1,1,4` and `1,4,1` both `0.000` (J1, J0, shade-off);
suite 97/97 ×2; partitioned perf intact (5.94 vs 8.74 single at 1024).
Residual err deltas (3015 vs 2986) are fp accumulation-order noise.
Per-brick `viewDir` needs no action (shaded J1 exactly 0.000).
`SetPartitions` doc now records this; default-`1,1,1` guidance stays
(perf wash), but partitioning is supported tiling, not a known-bad path.

### 4. Special paths recheck (same session)

Camera-inside fullscreen (`DOLLY` 1/2/3/5/8): `M/GL` 0.74/0.76/0.60/
0.59/0.49, all `<1`, inside more competitive; suite covers 5 such tests.
Batch×partitioned: narrow-or-tie (F1 6.30 … F16 6.39, one F4 8.16 run
shown thermal by repeat). Selection (`VolumePicking` 0.23 s,
`HardwareSelector` 1.0 s): pass, no perf contract.

### 5. Partition mode post-fix: batch optimum, NIFTI wash, DICOM 2× loss

Batch sweep on the grid path post-fix (1024 SD4, `1,1,4`): `DEF`
6.02/6.05 best-or-tied (F1 6.25/6.18, F2 6.29, F4 6.23, F8 6.39, F16
6.23/6.20); at fine `DEF(dense→1)` 15.19/14.40 vs F16 17.27 (−14%).
Default needs no retune on partitioned — same narrow-or-tie rule.

Partitioned vs single post-fix (NIFTI metal): 1024 SD4 6.06 vs 7.00
(−13% partitioned); 1024 SD0.5 14.96 vs 14.53 (tie); 2048 SD4 11.09 vs
9.41 (single −15%). Same directions as pre-fix: the lattice fix moved
correctness only. Wash across resolutions — default-`1,1,1` stands.

DICOM sparse is a different story. Parity is clean (1133/0.000 vs
1122/0.000 — the lattice fix generalizes to minmax-active segments) but
perf is catastrophic: 1024 SD4-y 12.2 vs 5.8 (+108%), 2048 ~19.8 vs
10.5 (+90%), tight interleaved repeats both. Mechanism is leap
fragmentation, proven by raw-march inversion (`MINMAX=0/ACCEL=0`:
partitioned 11.6 vs single 13.2, −12%): the gap exists ONLY when
skipping is on. Single-brick W1PRE/minmax leaps fly over brick joints
in one probe; grid segments must stop, re-probe and re-walk at each of
the 3 joints, fragmenting exactly the long empty runs sparse DICOM
lives on (cell-walk 108 ms vs leaps 40 ms at fine) — `§26.9`
skip-resolution at brick scale. Fullscreen-background and program-size
suspects are exonerated by the same inversion (both would persist raw).

Segment-count scaling (2048 SD4 NIFTI: `1,1,1` 9.47 / `1,1,2` 10.62 /
`1,1,4` 11.13) is sublinear — ~70% of the gap arrives with merely
taking the grid path (walker/DDA/occupancy/kernel footprint), ~0.26
ms per extra brick. That kills the prologue-hoist idea (would recover
~0.5 of 1.66 ms at refactor-scale risk to shared `marchVolumeUnified`).
The dense-grid-bypass sketch from the prior session is unaffected (it
targets dense, where there is nothing to fragment); sparse keeps the
grid it pays for.

Guidance, now measured: never partition sparse volumes (they fit
single-brick anyway); partitioning stays a dense/huge-volume tiling
fallback (0.000, wash-or-better). Fix shapes exist (cross-joint leap
carry = the shelved segHop vehicle; fewer joints) but are ROI-negative
with zero users — revisit only with a volume that requires tiling AND
a capture-backed profile.
