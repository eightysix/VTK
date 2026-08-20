# j0 gap decomposition with a fixed-steps probe (2026-08-20)

The `J0_GAP_DUMP.txt` §3 "envelope tax" conclusion was built on a **dead
probe**: `VTK_METAL_TEST_MARCH_STEPS` only capped march variants 4/5, but the
app is pinned to variant 0 (`VolumeMarchVariant()` returns 0, TEMP-REPRO). So
the fixed-steps data in that dump measured nothing — the cap never bit.

This doc fixes both backends' probes, verifies they bite, and re-runs the
fixed-steps sweep. Result: the j0 deficit is **not** an envelope cost.

## 1. Why the old probe was dead

- `VolumeMarchVariant()` (`vtkMetalGPUVolumeRayCastMapper.mm:340-348`) returns
  `0` (baseline divergent march).
- The shader only read `volumeUniforms.maxStepsFrame` inside the
  `fc_marchVariant == 4` / `== 5` branches. For variant 0 the cap was dead
  code; `MARCH_STEPS` env had no effect (verified: uniform dump showed
  `MaxStepsFrame@1724 = 0.0` even with `MARCH_STEPS=1`).

## 2. Fixes

### 2.1 Metal probe now caps any variant

- `MetalShaders.metal` (~4168-4185): added
  `else if (volumeUniforms.maxStepsFrame > 0.5) { maxSteps = min(maxSteps, int(maxStepsFrame)); }`
  after the variant 4/5 branches. Only fires when the probe env var is set;
  `MaxStepsFrame` stays 0 in production for variants 0/6-9.
- `vtkMetalGPUVolumeRayCastMapper.mm` (~7358-7386): the MARCH_STEPS branch now
  runs for every variant (was gated on variant 4/5); added a
  `VTK_METAL_TEST_MARCH_DEBUG` stderr line.

### 2.2 GL probe

- `vtkVolumeShaderComposer.h` `TerminationInit`: injects
  `g_terminatePointMax = min(g_terminatePointMax, N.0)` from
  `VTK_METAL_TEST_GL_STEPS`.
- The first attempt clamped in `TerminationInit` only — but a later
  `ClippingInit` block re-assigns `g_terminatePointMax`, overwriting the clamp.
  Fixed by clamping at the **march exit condition** in
  `TerminationImplementation`: `g_currentT >= min(g_terminatePointMax, N.0)`.

### 2.3 Stale test binary

- `build_macos_metal/bin/vtkMetalGLVisualComparison` was stale (15:20) and not
  in `build.ninja` (tests were OFF). A manual clang++ relink produced a broken
  binary that rendered black on **all** Metal scenes (volume AND geometry),
  even though the embedded shader was byte-identical to the committed source.
- Fix: `cmake -S . -B build_macos_metal -DVTK_BUILD_TESTING=ON -DBUILD_TESTING=ON`
  then `ninja -C build_macos_metal vtkMetalGLVisualComparison`. Also had to
  `touch vtkVolumeShaderComposer.h` to force `vtkRenderingVolumeOpenGL2`
  (the library that actually includes the composer header) to rebuild.

## 3. Probe verification (iteration images)

Metal `METAL_ITER=1` (red = marchIter/256), GL `VTK_METAL_TEST_GL_ITER=1`
(red = g_currentT/4096):

| config | covered | mean iter | max iter |
|---|---|---|---|
| Metal baseline | 457,876 | 81.08 | 221.9 |
| Metal `MARCH_STEPS=1` | 457,876 | 1.00 | 1.00 |
| GL baseline | 421,828 | 86.50 | 224.9 |
| GL `GL_STEPS=16` | 421,828 | 16.06 | 16.06 |
| GL `GL_STEPS=64` | 421,828 | 52.26 | 64.25 |

GL max tracks the cap exactly; the mean trails it because pixels still exit
early on opacity. Metal `MARCH_STEPS=1` hits mean 1.00. Both caps bite.

## 4. j0 fixed-steps sweep (ms, mean of 3-4 runs)

Env (j0, minmax/accel off — minmax is a Metal-only feature, GL has none):

```
VTK_METAL_TEST_SAMPLE_DISTANCE=4 VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 \
VTK_METAL_TEST_MINMAX=0 VTK_METAL_TEST_ACCEL=0 VTK_METAL_TEST_NUM_SLABS=1 \
VTK_METAL_TEST_JITTER=0 VTK_METAL_TEST_IGN_JITTER=0
```

| steps | 2048 M | 2048 GL | M/GL | 1024 M | 1024 GL | M/GL |
|---|---|---|---|---|---|---|
| 1 | 1.41 | 3.75 | **0.38** | 0.99 | 2.11 | **0.47** |
| 4 | 1.82 | 4.12 | 0.44 | — | — | — |
| 16 | 4.46 | 7.24 | 0.62 | 3.27 | 5.30 | 0.62 |
| 64 | 16.93 | 19.28 | 0.88 | 11.87 | 14.59 | 0.81 |
| uncapped | 45.18 | 40.83 | **1.11** | 27.12 | 27.90 | 0.97 |

## 5. Interpretation

1. **The envelope is cheaper in Metal.** At steps=1 (pure dispatch/fixed cost)
   Metal is 1.41 ms vs GL 3.75 ms at 2048 — Metal is 2.6x *faster*.
2. **Per-step cost is identical in lockstep.** Between steps 1 and 64 both
   backends cost 0.246 ms/step at 2048 (Metal: (16.93−1.41)/63; GL:
   (19.28−3.75)/63).
3. **The deficit appears only in the divergent regime.** From steps=64 to
   uncapped the marginal cost is 1.66 ms/step (Metal) vs 0.96 ms/step (GL) at
   2048 — and Metal does *fewer* samples (mean 81.08 vs GL 86.50) yet still
   ends up slower.
4. At 1024 the uncapped case ties (0.97), matching the earlier "1024 tied"
   observation — the deficit scales with resolution.

### Conclusion

The old "envelope tax" classification (`J0_GAP_DUMP.txt` §3/§8) is **dead**.
Metal's fixed cost is lower and its per-sample cost matches GL; the j0 gap is a
**divergent-march tail cost** (SIMT lane lock on the longest rays, or a
threadgroup-level loop bound) that only shows up when rays exit at different
lengths and at high resolution.

## 6. Cross-variant sweep: does the march variant close the gap?

The divergent-tail hypothesis predicts that scheduling variants (unrolled
fetches 6/7/8/9, latched exits 3, uniform frame-max 4) might change the tail
cost. Ran the same j0 sweep (2048, uncapped + fixed steps 1/16/64) for every
march variant via `VTK_METAL_TEST_MARCH_VARIANT`. GL reference repeated per
loop (GL itself has no variants).

| variant | uncapped | M/GL | steps=1 | M/GL | steps=16 | M/GL | steps=64 | M/GL |
|---|---|---|---|---|---|---|---|---|
| 0 (baseline divergent) | 43.22 | 1.05 | 2.00 | 0.49 | 4.52 | 0.68 | 16.95 | 0.91 |
| 3 (latch exits) | 43.62 | 1.09 | 1.41 | 0.36 | 4.57 | 0.64 | 17.10 | 0.92 |
| 4 (uniform frame-max) | 43.65 | 1.07 | 1.75 | 0.42 | 4.59 | 0.67 | 18.30 | 0.97 |
| 5 (hybrid uniform+tail) | 43.54 | 1.09 | 43.90* | 9.56 | 44.15* | 6.21 | 46.12* | 2.44 |
| 6 (8x unroll) | 43.42 | 1.08 | 1.54 | 0.35 | 4.73 | 0.72 | 17.21 | 0.91 |
| 7 (4x unroll) | 43.65 | 1.11 | 1.46 | 0.36 | 4.73 | 0.72 | 17.65 | 0.96 |
| 8 (harness w48 schedule) | 43.56 | 1.10 | 1.40 | 0.37 | 3.46 | 0.49 | 15.07 | 0.81 |
| 9 (adaptive-width w48) | 43.50 | 1.06 | 2.22 | 0.53 | 4.25 | 0.64 | 15.80 | 0.83 |

\* Variant 5's `MARCH_STEPS` is a **floor**, not a cap: the shader computes
`maxSteps = max(per-fragment, mainSteps)`, so a 1-step/16-step bound still
marches the full per-fragment length (~43.9 ms). The fixed-steps rows for
variant 5 are therefore uncapped (expected; variant 5 is the hybrid).

### 6.1 Interpretation

1. **No variant fixes the uncapped deficit.** Every variant lands uncapped at
   ~43.2-43.7 ms, M/GL 1.05-1.11. The gap is not scheduling, unrolling, fetch
   latency, or latch policy — all of those are already ruled out.
2. **Variant 8/9 (the production harness-scheduled march) is the best at high
   fixed steps** (steps=64: 15.07/15.80 vs 16.95-18.30 for others) but the
   uncapped time is identical to baseline — the scheduling helps only when the
   step count is uniform, i.e. it does not attack the divergent tail.
3. The deficit is therefore insensitive to *how* the march is scheduled. It
   tracks the **distribution of ray lengths** (mean 81, max ~222 at 2048) and
   the resolution (1024 ties, 2048 loses) — pointing at threadgroup/occupancy
   effects (long-ray lanes pinning threadgroups) rather than the loop body.

### Next step

Attack the divergent-tail cost at the threadgroup level: investigate how long
rays (up to ~222 steps vs mean 81) pin Metal threadgroups / SIMD groups while
GL warps drain earlier, and whether re-packing rays by length (or a frame-max
uniform bound with latched exits, variant-4 semantics at the *threadgroup*
level rather than per-fragment) drains the tail earlier.