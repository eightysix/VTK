# Metal vs OpenGL: near-parity / slower scenes analysis

SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
SPDX-License-Identifier: BSD-3-Clause

## Summary

The `vtkMetalGLVisualComparison --bench` harness measures every Metal test scene
under both the Metal and the OpenGL backends and reports a per-scene M/GL
ms/frame ratio (Metal ms / GL ms). Most scenes are Metal-faster (depth peeling
down to 0.34, volumes 0.54, transparency ~0.65). A handful of scenes
consistently sit at or above parity in the full-suite run:

| Scene (suite, 30f × 3r) | GL ms/f | Metal ms/f | M/GL |
|---|---|---|---|
| CellColor | 0.58 | 0.71 | 1.23 |
| Glyph3DMapper | 0.38 | 0.46 | 1.21 |
| CpxActorHi | 8.80 | 9.25 | 1.05 |
| CpxCellHi | 1.93 | 2.00 | 1.04 |
| Texture | 0.35 | 0.36 | 1.03 |
| CpxGeomLo | 0.60 | 0.60 | 1.01 |

This document records (a) what the investigation found when each suspect scene
was re-measured in isolation, (b) which readings are real backend differences vs
measurement artifacts, and (c) the optimization opportunities identified in the
Metal backend (with the ones actually implemented below).

## Benchmark methodology

Same harness and methodology as `VolumeRayCastPerfRegression.md`:
`Rendering/Metal/Testing/Cxx/TestMetalGLVisualComparison.cxx` with
`--bench --complexity`, offscreen targets on both backends, camera nudged per
timed frame, `WaitForCompletion` / `glFinish` inside the timed region so the
wall clock covers GPU time. Isolation runs add `--scene <name> --reps 3`.
Hardware: Apple silicon (arm64), macOS, Release build.

## Findings

### 1. Most of the "near parity / slower" group is small-frame measurement noise

Re-measuring every suspect scene in isolation (fresh process, only that scene)
does not reproduce the suite ratios:

| Scene | M/GL in suite | M/GL isolated | Identical scene? |
|---|---|---|---|
| CellColor | 1.23 | 1.00 / 1.15 | = CpxCellLo |
| CpxCellLo | 0.63 | 1.01 | = CellColor |
| Glyph3DMapper | 1.13–1.21 | 0.87 | — |
| CpxGeomLo | 1.01 | 0.74 | — |
| Texture | 1.03 | 0.80 | — |
| CpxCellHi | 0.86–1.04 | 0.76 | — |
| CpxGeomHi | 0.77 | 0.80 | — |
| **CpxActorHi** | **1.05** | **1.03** | — |

The decisive control is that **`CellColor` and `CpxCellLo` are the same scene**
(both `BuildCellColorGridScene(r, b, 4, 30)` at 800×800), yet the suite reads
1.23 and 0.63 for them — a backend difference cannot produce two different
ratios for one scene. On sub-millisecond frames the M/GL ratio is dominated by
run-to-run noise: Metal is the *more stable* backend (σ ≈ 0.01–0.04 ms) while
GL's small-frame σ is 2–5× larger (e.g. 0.62±0.08, 0.41±0.11), and the ratio
swings between 0.7 and 1.2 depending on suite position / thermal / clock state.

### 2. The one reproducible Metal-slower scene: CpxActorHi (draw-call bound)

The 32×32 = 1024-actor scene reproduces M/GL 1.03–1.05 with tight σ on both
backends, both in the suite and in isolation:

- suite: GL 8.80±0.06 vs Metal 9.25±0.06
- isolated: GL 9.07±0.17 vs Metal 9.32±0.12

At ~9 µs/draw for both backends, Metal is ~5% higher per draw. The mechanism is
Metal's per-draw CPU cost: every draw replays pipeline state plus vertex/fragment
buffer sets on the `MTLRenderCommandEncoder` (e.g.
`vtkMetalPolyDataMapper::ReplayRenderBundle`), while GL's cached-VAO path
reaches the driver with a single `glDrawElements`. `CpxActorLo` (8 actors) is
still Metal-faster (0.89), so the overhead only crosses parity at high
draw-call counts.

### 3. Metal's advantage grows with workload

Across the complexity scenes, Metal's relative advantage widens as the scene
gets heavier: CpxGeomLo 0.74 → CpxGeomHi 0.80 → CpxGeomBig 0.87, CpxCellHi 0.76,
depth-peel 0.34–0.45, volumes 0.54–0.58. The sub-millisecond scenes are where
the ratio is noise; the only genuine regression is the per-draw CPU path.

## Optimization opportunities

The Metal backend already follows the main Apple guidance: function-constant
shader specialization (dead-code elimination), render-bundle replay of
pre-recorded encoder commands (Apple's "encoding indirect command buffers on the
CPU" pattern), batched `setVertexBuffers:`/`setFragmentBuffers:`, empty passes
skipped (volume pass only when volumes exist), camera transforms computed once
per frame rather than per actor. The remaining opportunities, in impact order:

1. **Always-on picking-ID color attachment.** `vtkMetalRenderWindow` keeps a
   window-sized RGBA32Uint `IdsTexture`, every render pass binds it as
   `colorAttachments[1]`, and it is cleared + stored every frame — ~2.5 MB of
   DRAM store traffic per frame at 800×800 — even when no `vtkHardwareSelector`
   is active (GL only enables its ID target during selection passes). The
   `out.ids` write itself is already dead-code-eliminated outside selection by
   the `kEmitIds` function constant, but the clear/store is unconditional.

2. **Flat background drawn as a full-screen triangle every frame.** For a flat
   non-gradient background the renderer still runs a full-screen gradient
   pipeline pass with depth-stencil state switches every frame; GL just uses the
   clear color ("GL only draws the overlay when gradient mode is on"). For
   single-renderer frames the first renderer's `MTLLoadActionClear` already
   paints the whole attachment, so the pass is redundant.

3. **Per-actor double-precision matrix churn.** Each actor recomputes the
   transposed model matrix, a view×model multiply and a 3×3 matrix invert in
   doubles every frame (`vtkMetalPolyDataMapper::RenderPiece`). ×1024 actors
   this is a real chunk of the draw-call-bound frame. It only changes when the
   actor / camera / VBO shift-scale changes, so it is cacheable.

4. **Over-wide batched bind range.** `ReplayRenderBundle` binds vertex +
   fragment buffer indices 0–12 (26 slots) on every draw even though most are
   nil / unchanged; the Metal CPU cost scales with the range. The batching
   analysis is also recomputed on every replay instead of once per bundle build.

5. **Indirect command buffers (`MTLIndirectCommandBuffer`).** Apple's headline
   mechanism for many-draw scenes: move the per-draw pipeline/buffer state into
   GPU-addressable memory and replay it with `executeCommandsInBuffer:`,
   removing the per-draw `set*` calls from the CPU entirely. This is the
   architectural lever for draw-call-bound content (CpxActorHi). *Not
   implemented here* — it is an invasive rewrite of the mapper's replay path and
   would need its own GL-parity verification pass.

6. **Minor.** Static read-only geometry/instance buffers use
   `MTLResourceStorageModeShared`; `Private` + blit upload is the documented
   leaner path for immutable geometry (difference is small on Apple silicon
   unified memory). The cell-color texture `read()` path is already fine.

## Implemented changes

All changes are confined to `Rendering/Metal`.

- **Skip the picking-ID attachment's per-frame clear+store outside selection**
  (`vtkMetalRenderer::DeviceRender`): every render pass that binds the RGBA32Uint
  IDs attachment now uses `MTLLoadActionDontCare` / `MTLStoreActionDontCare` when
  no `vtkHardwareSelector` is active, and the full Clear/Load/Store chain only
  during a selection render. The attachment stays present so the scene/2D/
  gradient pipelines (which declare `colorAttachments[1]` whenever MSAA is off)
  stay consistent with the pass; the fragment `out.ids` write is already
  dead-code-eliminated by the `kEmitIds` function constant outside selection, so
  nothing writes it. (See `feat(#1)`.)
- **Skip the flat-background pass for single-renderer flat frames**
  (`vtkMetalRenderer::DeviceRender`): the redundant full-screen triangle is no
  longer drawn when one renderer already cleared the attachment with the same
  color. (See `feat(#2)`.)
- **Cache the per-actor model/normal matrices** (`vtkMetalPolyDataMapper`):
  the double-precision matrix math (model transpose, view×model 3×3 multiply,
  3×3 invert) is cached per (renderer, actor) and recomputed only when the
  actor matrix / VBO shift-scale (model matrix) or the camera view rotation
  (normal matrix) actually changes — compared against the view-rotation floats,
  not a camera MTime, which bumps every frame. (See `feat(#3)`.)
- **Precompute and tighten the batched uniform-bind range**
  (`vtkMetalPolyDataMapper::ReplayRenderBundle` / `RebuildRenderBundle`): the
  uniform-bind batching tables are built once when the render bundle is
  recorded instead of rescanning the command list on every replay, and the
  `set*Buffers:withRange:` calls span only the live uniform indices instead of
  the full 0–12 range. (See `feat(#4)`.)

## Reproduction

```
./macos_metal_build.sh --resume --tests          # builds the harness (tests must be ON)
./build_macos_metal/bin/vtkMetalGLVisualComparison --bench --complexity --reps 3
./build_macos_metal/bin/vtkMetalGLVisualComparison --scene CpxActorHi --bench --reps 3
```

The full-suite run verifies the visual comparison (worst thresholded error stays
0.095, the pre-existing VolumeRayCast value) and re-measures the M/GL table; the
isolated scene runs separate real backend differences from measurement noise.

The Metal ctest image-comparison suites also verify the changes:
`Rendering/Metal/Testing/metal_ctest_report.py` (RenderingCoreCxx-Metal) and
`-p RenderingVolumeCxx-Metal` both stay fully green (175/175 and 97/97) with the
implemented changes, matching the pre-change baseline.

The benchmark cannot resolve the implemented savings: CpxActorHi's M/GL ratio
swings 0.98–1.08 run-to-run (thermal/clock noise of several percent on both
backends), which is larger than the combined ~1–2% of frame time these changes
remove. The savings are real fixed per-frame / per-actor work (a window-sized
RGBA32Uint clear+store, a redundant full-screen triangle, ~1024 double-precision
matrix computations, and a per-replay command scan) but each is a small fraction
of the ~9 µs/draw frame, so they do not move the wall clock measurably.
