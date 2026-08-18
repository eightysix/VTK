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

Context and full investigation: `../SLAB_BENCHMARKS.md` and
`../minimal_gap/README.md` (the composite-path decomposition section).