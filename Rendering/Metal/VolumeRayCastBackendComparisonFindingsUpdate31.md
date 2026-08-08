# Camera-inside: per-sample march dumps at the knife-edge pixels — positions agree to ~0.005 texel, raw agrees to <0.03 units everywhere except one 78-unit spike at the iso-1150 crossing; sample counts diverge (update 31)

**Date:** 2026-08-08
**Scope:** Execute update 30's highest-value next probe: replace Metal's precomputed `maxSteps` with GL's position-bounds march termination. Before touching the march structure, instrument **per-sample** dumps on both backends at the two knife-edge pixels to see exactly where the sample streams diverge. Result: GL and Metal sample positions track each other to ≤0.006 texel and the scalar to <0.03 units (of 65535) for the whole march **except** a single ~78-unit scalar spike exactly at the iso-1150 crossing; and the two backends march **different total sample counts** at the far side (GL 233 vs Metal 236 at (372,·); GL 180 vs Metal 170 at (422,·)).

**Follows:** [Update 30](VolumeRayCastBackendComparisonFindingsUpdate30.md).

---

## 1. Instrumentation

Per-sample logging on both backends, then a frame-aligned diff at the two worst pixels:

- **GL** (`vtkOpenGLGPUVolumeRayCastMapper.cxx`): `g_dbgIter`/`g_dbgPx` set from a uniform; in the fragment shader's march loop, at every iteration emit
  `GL_SAMPLE px=(u,v) i=<iter> raw=<g_dataValue> pos=<g_dataPos xyz> color=<rgb> op=<g_framebufferColor.a>`.
  One log line per `i` per affected fragment per frame; capture frames across a couple of seconds and align by `i` sequence (the GL backend re-renders on window events, so a multi-frame log contains several `i=0..N` blocks).
- **Metal** (`vtkMetalGPUVolumeRayCastMapper.mm` / `MetalShaders.metal`): a `DebugKernel` (locked to the two pixels) prints the equivalent fields
  `DEBUG SAMPLE px=(u,v) i=<iter> t=<> tex=<evalT voxel> eval=<evalPoint data> raw=<v> norm=<normalized> op=<opacity so far> mip=<> rgb=<> w=<> accA=<> accC=<> maxSteps=<n> termMax=<>` for every `i` every frame.
- Comparison: split the Metal log and the GL log into per-frame `i`-sequences, take the **last frame of each**, and compare per `i`:
  - `pos` (GL) vs `eval` (Metal), both in [0,1] volume space — reported as texel deltas ×512;
  - `raw` scalar — reported in units of 65535 (×65535).

Files: `/tmp/bc/u31/GL_372.log` (pixel (372,380), 240 lines), `/tmp/bc/u31/GL_422.log` (pixel (422,419), 180 lines), `/tmp/bc/u31/Metal.log` (both Metal pixels (372,131) and (422,92), 236 + 170 samples/frame).

## 2. Result — per-sample agreement and the single crossing spike

### 2.1 Positions track each other to ≤0.006 texel everywhere

Across the whole march, `eval_MT − pos_GL` is at most ~0.006 texel per axis:

- (372,·): |Δz| ≤ 0.0008 texel for the shared samples (mean ~0.0005).
- (422,·): |Δz| ≈ 0.005 texel (0.0046–0.0053), constant through the march; Δx/Δy the same scale.

### 2.2 Scalar agrees to <0.03 units except one spike

For both pixels, `raw` is **identical to <0.03 units (of 65535)** for every shared sample — this is far below any iso-crossing sensitivity — **with exactly one exception**:

- (422,·) `i=167`: `rawGL = 0.018494` vs `rawMT = 0.017304` → **Δraw = −78 units**, while |Δpos| is still ~0.005 texel. This is precisely the sample where the cumulative opacity jumps (op GL 0.4009 vs MT 0.3126), i.e. the scalar-1150 iso crossing.
- (372,·) `i=215..232`: agreement is ≤0.03 units and op matches to 4 decimals through the crossing (op 0.0180→0.2114 identical both backends); the knife-edge flip for this pixel is decided by the far-side **sample-count** difference (below), not by a scalar spike.

### 2.3 Far-side sample count diverges — in opposite directions

| pixel | GL marched | Metal marched | notes |
|---|---|---|---|
| (372,·) | 233 real samples (`i=0..232`; `i=233..239` are post-loop garbage printed by the debug re-render) | 236 | Metal marches **+3** further |
| (422,·) | 180 | 170 | Metal stops **−10** earlier |

So neither backend simply stops "one sample short": the two termination mechanisms (GL's `g_dataPos` texture-space bounds loop vs Metal's precomputed `maxSteps = ceil((tEnd−firstT)/stepSize)`) resolve to a different sample count at the far side, and the difference is per-pixel.

## 3. Interpretation

- The step-length/step-position accumulation is **not** a significant divergence source here: positions agree to ~0.005 texel at every shared index. The prior ~0.006% step-agreement (update 24) does not translate into observable position drift at these sample counts (1150 samples × 0.006% × ~1/512 texel ≈ 1e-4 texel — below the dump precision).
- The knife-edge flip has **two independent drivers**, both at the far side:
  1. **(422,·) a single scalar spike of 78 units at `i=167`, exactly on the iso** — large enough to flip the transfer-function crossing (Δop 0.088 here) even though every other sample agrees to <0.03 units. With |Δpos| still ~0.005 texel at that sample, a ~78-unit raw jump is more than a pure 0.005-texel trilinear round-trip explains (~30 units for a local gradient of ~6k units/texel) — it points at a genuinely different sampled scalar at the crossing (nearest-vs-trilinear, a different texel picked, or a float32 difference in the trilinear weights near the texture border).
  2. **(372,·) a far-side sample-count difference (236 vs 233)** — the two marches end on a different last sample, so the final composited remnant/termination flips the edge even though every shared sample agrees.
- Both drivers live in the per-sample march chain (update 30 section 5), and update 30's Probe 3 (anchor perturb) is further confirmed inert: a position shift moves only 2/307 pixels, while a *scalar* change at the crossing is what actually flips the edge.

## 4. Next probes (highest value first)

1. **Scalar spike at the crossing**: dump the raw texture fetches (base + gradient texels, i.e. the 2/4/8 neighbors and their weights) at (422,·) `i=167` on both backends. Determines whether the 78-unit delta is a nearest/tri-linear mode difference, a border/tile fetch difference, or a float32 weight difference. No march-structure change needed.
2. **Sample-count parity**: replace Metal's precomputed `maxSteps` with GL's `while(pos in bounds)` termination (or clamp Metal's count to GL's) and re-diff — the (372,·) flip should resolve if the count is the sole driver there.
3. If probe 1 shows the spike is position-scale (~30 units), re-check the (422,·) constant 0.005-texel position offset (double the (372,·) offset) as the residual cause of the spike.

## 5. Reproduction

```sh
# GL per-sample dumps (capture frames; align by i-sequence, take last frame)
build_macos_metal/bin/vtkRenderingVolumeCxxTests \
  TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
  --vtk-factory-prefer RenderingBackend=OpenGL -D build_macos_metal/ExternalData/Testing \
  -T build_macos_metal/Testing/Temporary -V /tmp/bc/u31/GL.png
# Metal per-sample dumps (DebugKernel prints every i every frame)
# compare: last-frame i-aligned diff of pos/raw/op  ->  /tmp/bc/u31/GL_372.log GL_422.log Metal.log
```

Artifacts: `/tmp/bc/u31/{GL_372.log,GL_422.log,Metal.log}` + `cmp.py`; no production code changed (instrumentation reverted after capture).
