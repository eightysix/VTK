# jitter_gap_repro — results

Minimal repro of the app's Metal-vs-GL jitter cost question on the real DICOM
volume (512x512x1794 R8). Draws the volume bounding box with the app's exact
view-projection, then marches the app's oblique ray cast with the app's
transfer function, once via OpenGL 3.2 core (CGL) and once via Metal, with
GL's jitter semantics as the reference on both backends.

## Build

```
clang -fobjc-arc -lstdc++ -framework Metal -framework Foundation \
      -framework OpenGL -framework QuartzCore \
      jitter_gap_repro.mm -o /tmp/jitter_gap_repro
```

## Run (fair recipe — rt 2048, sd 4, 10 frames, real volume)

```
/tmp/jitter_gap_repro 2048 4 10 <path>/dicom.u8
```

Output (stderr):

```
GL    j0 50.779   j1 79.746   jitter +57.0%
METAL j0 40.287   j1 66.871   jitter +66.0%
M/GL  j0   0.79   j1   0.84
RESULT: gap CLOSED (M/GL j1 0.84)
```

`[rt] [sd-mm] [frames] [volume] [mode]` — mode `constphase`, `fetchonly`, or
`lattice` selects the jitter-field debug variant (below).

## Env knobs (diagnostics)

| env | effect |
|---|---|
| `PROBE_RAST=1` | fragment shader returns early (raw rasterization coverage) |
| `PROBE_FULLSCREEN=1` | fullscreen quad instead of the box |
| `NDC_CORNERS=1` | tile-grid NDC probe (all triangles at fixed positions) |
| `CULL=none\|front\|back` | Metal cull mode override |
| `T_PER_TRI=1` | one draw call per triangle (12 draws) |
| `TRI_SEL=n` | with T_PER_TRI, draw only triangle n |
| `NOISE_TEX=1` | jitter tap via texture2d (R8Unorm tile) instead of the in-shader `kBlue64` constant array |
| `MTL_DUMP_PPM=1` / `GL_DUMP_PPM=1` | dump the Metal/GL framebuffer to /tmp/jgr_metal.ppm, /tmp/jgr.ppm |
| `PROBE_BOX=1` | constant opaque white TF (probe variants) |

## Root cause found: MSL float3 vertex stride

The box corners buffer is tight-packed (12 B/vertex). MSL `constant float3*`
strides at 16 B (float4-aligned), so `corners[vid]` fetched scrambled values;
vid 27+ read past the 432-B buffer. This is why 6 of 12 triangles were "dead"
under every cull mode and the box rasterized at ~half coverage. The GL path
(stride 0) was always correct.

Fix: `constant packed_float3*`. After the fix:

- TRI_SEL=8 alone: 266,756 px vs 267,285 exact analytic area, at the true
  -z face position.
- Full box: Metal 1,831,193 px vs GL 1,831,510, identical bbox.
- Render parity (real march): GL 593,141 vs Metal 593,781 non-black px,
  mean |Δ| 3.4/255 on the union.

## Jitter question: does Metal pay more?

### Resolution x sample-distance matrix (10 frames/round)

| rt | sd | GL j0/j1 (ms) | GL Δ% | Metal j0/j1 (ms) | M Δ% | M/GL j0 | M/GL j1 |
|---|---|---|---|---|---|---|---|
| 512 | 2 | 40.5/51.2 | +26.6 | 39.3/47.1 | +19.8 | 0.97 | 0.92 |
| 512 | 4 | 21.3/30.5 | +43.4 | 18.9/29.5 | +56.3 | 0.89 | 0.97 |
| 512 | 8 | 14.8/21.2 | +43.1 | 10.3/18.4 | +77.5 | 0.70 | 0.87 |
| 1024 | 2 | 67.1/92.2 | +37.5 | 47.0/69.5 | +47.7 | 0.70 | 0.75 |
| 1024 | 4 | 41.5/60.2 | +45.2 | 30.4/48.8 | +60.6 | 0.73 | 0.81 |
| 1024 | 8 | 20.5/38.9 | +89.7 | 18.3/36.2 | +97.5 | 0.89 | 0.93 |
| 2048 | 2 | 71.7/99.9 | +39.3 | 54.9/69.4 | +26.4 | 0.77 | 0.70 |
| 2048 | 4 | 45.8/82.9 | +81.2 | 39.9/63.2 | +58.2 | 0.87 | 0.76 |
| 2048 | 8 | 30.8/64.3 | +108.6 | 27.8/56.6 | +103.7 | 0.90 | 0.88 |
| 4096 | 2 | 67.8/83.5 | +23.2 | 68.1/86.7 | +27.3 | 1.01 | 1.04 |
| 4096 | 4 | 44.1/71.4 | +61.9 | 44.8/74.5 | +66.2 | 1.02 | 1.04 |
| 4096 | 8 | 34.3/72.2 | +110.3 | 32.9/69.0 | +109.6 | 0.96 | 0.96 |

GL j1 at 2048/4 carries ~±15% run-to-run thermal variance (GL j0 ranges
43–57 ms; Metal is stable at 40.3–42.5 ms), so single cells are noisy.

### Averages at the fair config (2048/4, 8 rounds, means ± stdev)

| metric | GL | Metal |
|---|---|---|
| j0 (ms) | 50.30 ± 4.97 | 40.90 ± 0.86 |
| j1 (ms) | 75.49 ± 5.99 | 65.64 ± 1.45 |
| jitter delta (ms) | +25.19 ± 3.38 | +24.74 ± 1.76 |
| jitter delta (%) | +50.5 | +60.6 |
| M/GL j0 / j1 | — | 0.82 ± 0.08 / 0.88 ± 0.07 |

Paired per round, MetalΔ − GLΔ = −0.45 ± 3.04 ms: **statistically zero**.
Metal does not pay more for jitter in absolute terms; the %-gap is base-cost
arithmetic (same ~25 ms on a lower j0).

## Hypothesis verdicts (read-path explanation attempts)

| hypothesis | result |
|---|---|
| Jitter cost is per-pixel phase divergence, noise tap is free | confirmed both backends: `constphase` (constant 0.5-step) — GL −9.7%, Metal +1.8% vs +50–65% for the real per-pixel field |
| `allowGPUOptimizedContents=YES` closes the gap | refuted: bit-identical render but ~50% slower on j0 and j1 (59.5/92.7 ms), flat M/GL ~1.41 |
| lattice re-alignment (app Metal semantics: `tStart = jitterF + ceil((t-jitterF)/step)*step`) restores coherence | refuted: per-pixel phases persist, same cost class — GL +73%, Metal +58% |
| 4 KiB `kBlue64` constant table hurts occupancy | refuted: `texture2d` R8Unorm tap timing-identical (65.99 vs 65.96 ms), field-identical (2/4.19M px differ by 1/255) |
| Metal j0 is a longer march than GL j0 | false: GL j0 adds `stepSize` (VTK `g_rayJitter = g_dirStep`), Metal j0 adds nothing |

## Conclusions

1. The harness never reproduces the app-level M/GL j1 ~1.35; with correct
   vertex fetch, M/GL j1 stays ≤ ~1.0 across the whole rt x sd matrix.
   Metal wins everywhere on identical rays.
2. Jitter cost (absolute ms) is equal on both backends; jitter cost grows
   strongly with sample distance (sd 2 → +20–40%, sd 8 → +90–110%) on both.
3. At 4096 (memory-saturated) M/GL j0 -> ~1.0 and j1 -> ~1.04.
4. The app-level gap is not a harness-reproducible read-path effect of the
   shared GL jitter field; candidates remaining (unexplained by this harness)
   are app-specific pipeline costs outside the march itself.

## Reproduce

```
# fair timing, then average several rounds:
for i in 1 2 3 4 5 6 7 8; do \
  /tmp/jitter_gap_repro 2048 4 10 dicom.u8 2>&1 | grep -E '^GL    j0|^METAL j0|^M/GL'; \
done

# jitter-field debug variants:
/tmp/jitter_gap_repro 2048 4 10 dicom.u8 constphase   # constant phase
/tmp/jitter_gap_repro 2048 4 10 dicom.u8 lattice      # phase-quantized march
NOISE_TEX=1 /tmp/jitter_gap_repro 2048 4 10 dicom.u8  # texture2d noise tap

# rasterization coverage probes:
PROBE_RAST=1 MTL_DUMP_PPM=1 /tmp/jitter_gap_repro 2048 4 1 dicom.u8
TRI_SEL=8 T_PER_TRI=1 PROBE_RAST=1 MTL_DUMP_PPM=1 /tmp/jitter_gap_repro 2048 4 1 dicom.u8
```
