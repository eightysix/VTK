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
./jitter_lag_repro [rt 2048] [sd 4.0] [frames 30] [mode all|sharp|point|texture]
```

Runs all six cells (GL j0/j1, Metal j0, Metal j1 per noise mode) interleaved and
prints the jitter deltas and M/GL ratios:

```
GL    j0   44.609   j1   53.044   jitter  +18.9%
METAL j0   44.030   j1(sharp)   69.543   jitter  +57.9%
METAL j0   44.030   j1(point)   53.925   jitter  +22.5%
METAL j0   44.030   j1(texture)  45.775   jitter   +4.0%
M/GL  j0     0.99   sharp  1.31   point  1.02   texture  0.86
```

The volume is generated in-process: 512x512x1794 R8 z-slice gradient
(~448 MB) — the DICOM-like cell from `minimal_gap` (`--data 0` / `lag_repro`
smooth).

## What it shows (M2 MBA, battery, 30 frames, 2048x2048, SD 4 mm)

| | j0 | j1 (jitter on) | jitter cost | M/GL @j1 |
|---|---|---|---|---|
| GL (correlated texture) | ~44-47 ms | ~53-55 ms | +18% | — |
| Metal **sharp** (per-pixel IGN / blue-noise class) | ~44 ms | ~70 ms | **+58-62%** | **~1.30-1.31** |
| Metal **point** (Fix 1: GL sawtooth, point-sampled) | ~44 ms | ~54 ms | +22-24% | ~0.99-1.02 |
| Metal **texture** (Fix 2: GL tile, LINEAR+REPEAT) | ~44 ms | ~46 ms | +4% | ~0.84-0.86 |

The app's DICOM single-pass at the same cell: GL 51.9 vs Metal 74.1 at j1
(M/GL 1.36-1.42x; j0 1.03-1.08x).

## Result: the jitter field, not the march, is the gap

Same snap, same march, same volume — only the Metal noise source changes:

- **sharp** (per-pixel independent phase) → +58-62%, M/GL ~1.3: lane-divergent
  fetch sets, ~10 voxels/step apart.
- **point** — the GL 128-sawtooth tile sampled point-wise (`i*13%256`,
  `fmod(x*13 + y*128, 256)/255`) → +22-24%, M/GL ~1.0: neighbors keep nearby
  lattice phases, the 3D fetches stay in the same voxels.
- **texture** — the GL tile itself via `texture2d`, LINEAR + REPEAT, UV
  `(x, rt-y)/128` (GL's y-flip) → +4%, M/GL ~0.85: the bilinear field is even
  smoother than point-sampled (pixel centers sit between texels), so it beats
  GL's own cost.

All three modes keep the footprint probe in the same ballpark (1328-1332/3025
probes) — the jitter is live in every mode, only its correlation changes.

## Why the backends diverge (the jitter path, not the march)

Both shaders shift the ray start to a per-fragment lattice phase
`tStart' = jitterF + ceil((tStart - jitterF)/stepSize)*stepSize` with
`jitterF = noise * stepSize` — but the noise differs:

- **GL samples a bilinear-filtered noise texture** (128x128 sawtooth, LINEAR +
  REPEAT — the app's `in_noiseSampler` analog). Adjacent fragments in a warp
  get nearly identical `jitterF`, so the warp marches coherently: same
  termination, same fetch set, ~free. `jitterF` *is* the lattice phase; on
  this cell (sd 4 mm, ~10 voxels/step) a phase delta δ is a 3D address delta
  of ~10δ voxels.
- **Metal sharp** uses an independent per-pixel source (IGN hash — the app's
  blue-noise tile is the same class: high-frequency, independent per
  fragment; harness A/B verified +57-63% either way). Independent phases
  diverge the lanes and randomize the fetch set (a warp pulls ~32 distinct
  z-planes out of the 447 MB NPOT texture), which on z-slice-correlated data
  costs +58-62%.
- **Metal point/texture** use the GL sawtooth field (spatially correlated):
  neighbors keep nearly the same phase, the fetch set stays coherent, the
  cost collapses to GL's level or below.

The noise path, not the pipeline or the march, is the entire gap.

Context and full investigation: `../minimal_gap/README.md`,
`../PERFORMANCE_INVESTIGATION.md` (§22), and `../lag_repro/README.md` (the
sibling no-jitter data-path lag repro).
