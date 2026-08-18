# lag_repro — minimal GL-vs-Metal 3D texture sampling lag repro

Self-contained reproduction of the "Metal lags GL on the raw single-pass
volume march" case (2026-08-18). One file, no VTK, no DICOM, no external
dependencies — only the system frameworks. Same GPU (Apple silicon), same
camera, same ray lattice, same shader math, MIP max-accumulate, offscreen RT.

## Build

```
clang -fobjc-arc -framework Metal -framework Foundation -framework OpenGL \
      -framework QuartzCore -lc++ -Wno-deprecated-declarations \
      lag_repro.mm -o lag_repro
```

## Run

```
./lag_repro [gl|metal|both] [rt 2048] [sd 4.0] [frames 30] [smooth 0]
```

- Default `both` prints each backend's avg frame time and the M/GL ratio.
- `smooth 1` swaps the xorshift noise volume for a smooth ramp — the control.
- The volume is generated in-process: 512x512x1794 R8 (~448 MB), matching the
  DICOM reference scene's footprint and aspect.

## What it shows (M2 MBA, battery, 30 frames, 2048x2048, SD 4 mm)

| data | GL | Metal | M/GL |
|---|---|---|---|
| noise (per-texel variation) | ~51 ms | ~80 ms | **~1.6x — the lag** |
| smooth (gradient ramp) | ~46 ms | ~44 ms | ~0.97x — parity |

Identical marched footprint both backends (1831504/4194304 pixels — the same
count as `minimal_gap`'s gl_gap/metal_gap and the app's volume projection),
so the rays and sample addresses are identical; only the data content flips
the ratio. On per-texel-varying data the Metal 3D texture read path degrades
as the working set exceeds the SoC cache, GL does not — a driver-level
texture-layout/cache behavior, not a shader or API-pipeline property.

## Root cause: `allowGPUOptimizedContents` (2026-08-18, verified)

A/B on the M2 MBA (interleaved, 30 frames, 2048x2048, SD 4 mm, noise data):

| Metal volume upload | ms/frame |
|---|---|
| default (private blit, `allowGPUOptimizedContents` YES) | **~79.5** |
| `vd.allowGPUOptimizedContents = NO` | **~44.2** |

The flag is worth ~1.8x on noise and ~nothing on smooth (default 42.8 vs
noopt 44.1): the lossless GPU-optimized layout is a tax on incompressible
data only, and it is the entire M/GL gap — with the flag off, Metal matches
or beats GL (~0.9x) even under the more favorable pipelined Metal clock.

Follow-ups that did NOT help, measured:

- `MTLStorageModeShared` + `replaceRegion` (no private blit): 83.6 ms — the
  layout flag governs shared textures too; the blit was never the unlock.
- Depth padding 1794 -> 2048: 96.8 ms — worse, not better; NPOT depth is
  not the pathological part.
- Nearest instead of trilinear: 47.1 ms — halves the tax (8 incompressible
  taps multiply the DRAM cost) but does not eliminate it.

Actionable rule: for DICOM/CT/MR/entropy-heavy volumes, set
`allowGPUOptimizedContents = NO` on the volume `MTLTextureDescriptor`
(keep private storage and the upload blit). Leave the default YES only for
compressible data (smooth atlases, distance fields).

Context and full investigation: `../SLAB_BENCHMARKS.md` and
`../minimal_gap/README.md` (the composite-path decomposition section).