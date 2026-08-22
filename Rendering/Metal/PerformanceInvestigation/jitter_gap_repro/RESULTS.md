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


## V31 port: back-edge exit (`DOEXIT=1`, 2026-08-21)

Ported the divergent_tail V31 fix (all exit conditions moved into the loop
BACK-EDGE — one branch per iteration instead of two around the fetch) to
both the GL and Metal marches via `DOEXIT=1`. Entry is guarded on
`maxSteps > 0` (an unguarded do-while composited one bogus sample on
~33 corner-grazer rays — caught by the GL footprint counter before it
could poison timings). Guarded footprints match the baseline exactly
(795455 / 796419), i.e. bit-identical traversal.

Result on the fair recipe (rt 2048, sd 4, real DICOM): NEUTRAL.

| | GL j0/j1 | Metal j0/j1 | M/GL |
|---|---|---|---|
| baseline | 51.4 / 91.7 | 42.9 / 66.7 | 0.84 / 0.73 |
| DOEXIT=1 | 50.5 / 92.5 | 40.8 / 67.5 | 0.81 / 0.73 |

The divergent_tail codegen deficit does not transfer to this march: the
interleaved TF-LUT taps change the instruction mix, and this repro's Metal
was already well ahead of GL (0.73-0.84). Knob kept for future A/B;
default output is unchanged.

## GL context knobs: profile + storage class — refuted (2026-08-21)

New env knobs on the GL side (`jitter_gap_repro.mm`): `GL41=1` requests the
4.1 core pixel format instead of 3.2 core; `GLSTORAGE=1` uploads the volume via
immutable `glTexStorage3D`+`glTexSubImage3D` instead of mutable
`glTexImage3D`. Both are hypotheses for why harness-GL pays +22–26 ms of
jitter tax while app-GL pays +10–12 ms on identical rays (HARNESS_VS_APP_GAP
§12). Single runs, 2048/SD4:

| config | GL j0 | GL j1 | GL Δ |
|---|---|---|---|
| baseline (3.2-core, mutable) | 47.63 | 71.22 | +23.60 |
| `GL41=1` | 43.42 | 70.23 | +26.82 |
| `GLSTORAGE=1` | 45.70 | 71.08 | +25.38 |
| both | 60.96 | 98.86 | +37.90 |

Both refuted (Metal inert throughout, Δ +21.6–24.5 ✓). Remaining candidates:
drawable/window-backed surface vs headless FBO, blending state, uniform
plumbing, occupancy shaping — see HARNESS_VS_APP_GAP §22 HANDOFF.

## HANDOFF ablation complete + protocol upgrade (2026-08-22 session)

Every remaining §22 HANDOFF candidate was implemented as an env knob in
`jitter_gap_repro.mm` and measured at 1024/SD4, ROUNDS=2-4, WARMUP=30.
New knobs: `WARMUP`, `ROUNDS` (order-alternated interleaved rounds,
mean±sd), `SURFACE=1` (NSWindow+NSOpenGLView drawable context),
`BLEND=1` (ONE, ONE_MINUS_SRC_ALPHA), `UBO=1` (std140 uniform block),
`PAD=1` (dead-but-unremovable prologue ALU chains), `AZSTEP=d`
(per-frame camera orbit replicating the app bench's `Azimuth(0.1)`),
`CLIP=1` (the composer's AdjustSampleRangeForClipping prologue ported,
plane a no-op exactly like the DICOMVolume scene's), `GAPMS=n` (idle
before each block), `SELECT=g0,g1,m0,m1` / `ONLYGL=1` (per-cell runs).

### Verdicts

| knob | GL jitter Δ (ms) |
|---|---|
| legacy (WARMUP=5) | +25.87 ± 10.71 |
| **WARMUP=30** (default now) | **+18.24 ± 2.81** |
| SPLIT_TF=1 WHILE=1 combo | +17.17 ± 2.75 |
| BLEND=1 | +38.31 ± 4.64 — REFUTED, worse |
| SURFACE=1 | +24.47 ± 4.91 — refuted |
| UBO=1 | +23.43 ± 3.22 — refuted |
| PAD=1 | +24.63 ± 10.01 — refuted |
| CLIP=1 | +20.05 ± 7.73 — refuted (image byte-identical ✓) |
| AZSTEP=0.1 | +11.18 ± 1.49 — CONFOUNDED, see below |
| GAPMS=2000 in-process | +18.67 ± 8.10 — refuted |
| fresh-process per cell (`SELECT=g0` / `SELECT=g1`) | **+20.6 ± 1.0** (j0 35.5±0.1 / j1 55.9±1.1) |
| app binary (live, same machine/hour) | **+8.66** (j0 29.32 / j1 37.98) |

Key findings:

1. **The warm-up fix from issue_tax_repro R3 transfers** (commit
   8dd24927a5): Apple's GL driver keeps re-JITting freshly linked complex
   programs past 10 draws; the harness's old 5-frame warm-up left JIT
   residue inflating GL blocks and dominating variance. WARMUP=30 halves
   the measured Δ and cuts its spread 4x. All historical single-shot
   harness numbers carry this contamination.

2. **Duty cycle / thermal state is NOT the mechanism**: the app-shaped
   protocol (fresh process per jitter value, seconds of idle, 30-frame
   bursts) is rock-stable and still pays +20.6 vs the app's +8.7 measured
   minutes earlier.

3. **AZSTEP (rotation) is not admissible evidence**: at the dose that
   reaches app-level Δ (+11), coverage falls 198k→92k px across the sweep
   (meanIter on the last frame: 113.7 over 177k px vs static 88.0 over
   427k → ~47% fewer total samples). At coverage-preserving doses
   (AZSTEP≤0.02) there is no improvement (+21.4±10.8). The app bench does
   rotate its camera, but our orbit sheds grazing rays much faster than
   the app's documented coverage; do not use rotation to claim parity.

4. **appgl_parity timing is INVALID** — root-caused: its march terminates
   after ~1 sample/ray (new probes uDbg==21 lastPos, ==22
   terminatePointMax [median 21 iters, should be ~86-200], ==23 dirStep
   [oversized ≥8 texels vs harness 2.9]). Its j0 6.9 ms / Δ +1.5 ms
   reflect a ~5x-shortened march (stride bug in the matrix chain feeding
   g_dirStep/g_terminatePointMax), NOT shader composition. Do not cite
   them. The TF_FACTOR knob added there is secondary.

5. Net: with rays/trip counts/field/uniforms proven identical (§12.1)
   and every context/state/duty-cycle candidate now refuted under tight
   protocols, the remaining harness-vs-app GL difference (~+12 ms of
   jitter Δ; also j0 absolute: harness 35.5 pure-GPU vs app 29.3
   including VTK CPU work) lives inside the app mapper's shader/mapper
   execution itself. Next steps: fix appgl_parity's matrix chain until
   its march stats match (coverage ~40%, meanIter ~86), then its verbatim
   FS becomes the valid composition probe; or inverse-transplant the
   lean harness FS into the app process to test process-level driver
   state (e.g., Metal coexistence).

## RESOLVED (same session, later): appgl_parity fixed — the composed shader carries the cheap jitter

The truncation root-caused above was ONE mis-transcribed constant:
`kInvProj[11]` held 0 instead of -1 (the inverse-projection w-row lost its
-z_eye term, so g_rayTermination collapsed near the ray entry and every
march stopped at min(tPM~21, saturation)). After correcting [11] to -1
(verbatim from today's glGetUniformfv dump):

| measurement @1024/SD4 | GL j0 | GL j1 | jitter D |
|---|---|---|---|
| app binary (live, incl VTK CPU) | 29.32 | 37.98 | **+8.66** |
| **fixed appgl_parity** (3 interleave pairs) | 25.53 | 33.24 | **+7.71 +-2.1** |
| lean jitter_gap_repro FS (WARMUP=30) | 35.5 | 55.9 | +20.6 |

- Iteration stats now match the live app: covered 43%, meanIters ~81+
  (was 21 before the fix).
- Absolutes reconcile: app frame total - parity GPU-only ~= 3.8 ms of
  VTK per-frame CPU overhead, consistent across j0/j1.

CONCLUSION: the app's cheap jitter lives in the COMPOSED SHADER TEXT ITSELF.
In the identical headless CGL+FBO harness context, the verbatim composed FS
pays +7.7 ms where the lean reconstruction pays +20.6 ms on proven-identical
rays/trip counts/field. All context/state/duty-cycle knobs were refuted
first because they were not the mechanism. The earlier §11/§12 conclusion
("not encoded in loop text") failed because individual knobs never
reproduced the composition as a whole.

New knob: `INCR=1` ports the composer's incremental position accumulation
(g_dataPos += g_dirStep) into the lean loop — byte-identical image, NO
effect on D (+22.8+-9.5): refuted as the single ingredient.

Remaining (optional) work: bisect WHICH compositional ingredient(s)
transfer the cheapness — candidates: conditional color fetch gated on
opacity, 1024xR32F opacity LUT shape, per-sample scale/bias, sign-gated
texMax OOB test, currentT/tPM break form, or simply their combination.
For the Apple-report purpose the boundary is established: two shaders,
one context, one geometry, 13 ms of jitter-delta difference.
