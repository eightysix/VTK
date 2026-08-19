# jitter_lag_repro — minimal GL-vs-Metal jitter-cost gap repro

Self-contained reproduction of the app's GL-vs-Metal **jitter** gap (2026-08-19):
GL's jitter is nearly free, Metal's costs +50-65%. One file, no VTK, no DICOM,
no external dependencies — only the system frameworks. Same GPU (Apple
silicon), same camera, same ray lattice, same march math, MIP max-accumulate,
offscreen RT.

## Build

```
clang -fobjc-arc -framework Metal -framework Foundation -framework OpenGL \
      -framework QuartzCore -lc++ -Wno-deprecated-declarations \
      jitter_lag_repro.mm -o jitter_lag_repro
```

## Run

```
./jitter_lag_repro [rt 2048] [sd 4.0] [frames 30]
```

Runs all four cells (GL/Metal x jitter off/on) interleaved and prints the
jitter deltas and M/GL ratios:

```
GL    j0   47.936   j1   53.263   jitter  +11.1%
METAL j0   45.564   j1   69.598   jitter  +52.7%
M/GL  j0     0.95   j1     1.31  <- GAP REPRODUCED
```

The volume is generated in-process: 512x512x1794 R8 z-slice gradient
(~448 MB) — the DICOM-like cell from `minimal_gap` (`--data 0` / `lag_repro`
smooth).

## What it shows (M2 MBA, battery, 30 frames, 2048x2048, SD 4 mm)

| | j0 | j1 (jitter on) | jitter cost |
|---|---|---|---|
| GL | ~47 ms | ~52-54 ms | **+9-22%** |
| Metal | ~44-46 ms | ~65-70 ms | **+52-66%** |
| M/GL | ~0.95-1.0 | **~1.30-1.41** | |

The app's DICOM single-pass at the same cell: GL 51.9 vs Metal 74.1 at j1
(M/GL 1.36-1.42x; j0 1.03-1.08x).

## Why the backends diverge (the jitter path, not the march)

Both shaders shift the ray start to a per-fragment lattice phase
`tStart' = jitterF + ceil((tStart - jitterF)/stepSize)*stepSize` with
`jitterF = noise * stepSize` — but the noise differs:

- **GL samples a bilinear-filtered noise texture** (128x128 sawtooth, LINEAR +
  REPEAT — the app's `in_noiseSampler` analog). Adjacent fragments in a warp
  get nearly identical `jitterF`, so the warp marches coherently: same
  termination, same fetch set, ~free.
- **Metal uses a per-pixel sharp hash** (independent `jitterF` per fragment —
  the app's blue-noise tile behaves identically; harness A/B verified
  +57-63% either way). Independent phases diverge the lanes and shift the
  fetch lattice relative to the data, which on z-slice-correlated data costs
  +50-65% (fetch-set change, not termination spread).

Swap the Metal hash for a smooth field and the gap collapses to ~0 (verified
in `minimal_gap`: Metal correlated-smooth jitter is ~free) — the noise path,
not the pipeline, is the entire gap.

Context and full investigation: `../minimal_gap/README.md`,
`../PERFORMANCE_INVESTIGATION.md` (§22), and `../lag_repro/README.md` (the
sibling no-jitter data-path lag repro).
