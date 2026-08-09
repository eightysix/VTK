# THE ±1 FIELD IS GONE: Metal clamped accumulatedOpacity to 1.0 at the opacity break, zeroing the background blend term dst*(1-a) that GL keeps; removing the clamp collapsed 63,692 px → 188 px (99.7%) (update 57)

**Date:** 2026-08-09
**Scope:** Implementing the fix suggested by update 56 §4/§6.1 candidate (c) — and finding the true cause: Metal's *alpha clamp*, not a missing blend.
**Target (unchanged):** Metal bit-identical to **clean GL** (`RenderingBackend=OpenGL`, no debug injection).

**Follows:** [Update 56](VolumeRayCastBackendComparisonFindingsUpdate56.md), [49](VolumeRayCastBackendComparisonFindingsUpdate49.md), [48](VolumeRayCastBackendComparisonFindingsUpdate48.md).

---

## 1. Root cause of Metal's missing background blend

Update 56 established the ±1 field is GL's `ONE, ONE_MINUS_SRC_ALPHA` blend over the background (`26/255`): GL stores `gf + (26/255)·(1−a)`, Metal stored plain `gf`. Update 56 left *why* Metal's declared blend (DirectScreen/FullscreenDirect pipelines, vtkMetalGPUVolumeRayCastMapper.mm:6047-6058) contributed nothing as an open question.

**Answer: Metal's shader clamped `accumulatedOpacity = 1.0f` at the opacity break** (`MetalShaders.metal:4814-4815`):

```metal
// BEFORE (the bug):
if (accumulatedOpacity >= 1.0f - 1.0f / 255.0f) {
  accumulatedOpacity = 1.0f;   // <-- makes 1-src.a == 0 at blend time
  break;
}

// AFTER (OpenGL parity):
if (accumulatedOpacity > 1.0f - 1.0f / 255.0f) {
  break;
}
```

- **GL** (`vtkVolumeShaderComposer.h` `TerminationImplementation`, line 3371): `if((g_fragColor.a > g_opacityThreshold) || ...) break;` — breaks **without** clamping. `g_fragColor.a` stays ≈ 0.9969 → blend term `dst·(1−a) ≈ 26/255 · 0.0031` survives into the framebuffer.
- **Metal** clamped alpha to exactly 1.0 → `1−src.a = 0` → the fixed-function blend correctly computed `src·1 + dst·0 = gf`. The blend machinery was fine all along; the *source alpha* was wrong. This also explains why the Metal output was deterministic and why the blend model (with the u8-quantized 26/255 background) failed at 75.7% while unblended gf matched 99.7%.
- The comparison operators also now match: GL uses strict `>` (break only when alpha strictly exceeds threshold), Metal previously `>=`.

The composite formula itself was already at parity (`g_fragColor = (1.0 - g_fragColor.a)*g_srcColor + g_fragColor`, vtkVolumeShaderComposer.h:2652 ≡ `accumulatedColor/Opacity = fma(weight, src, acc)` MetalShaders.metal:4794-4795), so only the clamp had to go.

## 2. Result: 63,692 px → 188 px (99.70% of the field difference eliminated)

| | px differing from fix_gl |
|---|---|
| before fix (update 56, fix_metal) | **63,692** |
| after fix (fix_metal2) | **188** (262,144 total) |

- `fix_metal2` vs `fix_gl`: **188 px / 250 channels differ**; 174 px are pure ±1 LSB, 14 px have |Δ|>1.
- `fix_metal2` vs the ideal blend model `rint_half_even((gf + 26/255·(1−a))·255)` (from the clean-GL float dump): **99.89% exact** — *better* than fix_gl itself (99.83%). I.e. Metal now tracks the physical model more closely than GL does.
- Metal is deterministic: two consecutive runs byte-identical (262,144/262,144 px).

## 3. The remaining 188 px are the known knife-edge set

All remaining differences are isolated single pixels spread across the field — the **grid-aligned-ray knife-edge set** from updates 48/54/55, *not* a systematic term:

- 14 px with |Δ|>1 (max +7/+8 at (397,110)) — these are exactly the pixels where **clean GL itself disagrees with its own float dump**: e.g. at (397,110) the dump gives gf·255 = (238.963, 185.562, 150.986) → blend model predicts green ≈ 186, yet fix_gl stores green = **179**. The knife-edge mechanism (update 48 §4: grid-aligned rays amplify tiny position/arithmetic differences into multi-LSB swings) explains this; the model's own failure set is 266 px that BOTH backends miss.
- 174 px ±1 — near-boundary pixels where Metal's float differs from GL's float by a hair around a rounding boundary.

## 4. Remaining work to bit-parity

1. **Metal's float must equal GL's float at the knife-edge pixels.** Update 48 closed Metal's side (Metal replays its written composite 68/68 on the gated pixels) but clean GL diverges from its own debug-injected form at 4/14 shared pixels ("clean GL ≠ debug GL ≠ Metal; clean GL diverges at compile level"). The residual 188 px are the pixels where clean GL's *compiled* arithmetic (the exact fused operations/rounding the GLSL compiler emits) differs from Metal's `fma`-form composite — amplified by grid alignment.
2. Recommended continuation per update 48 §5: Metal-side bisect of clean GL's compile-level composite (reassociation / break-parity variants) to find the operation order that reproduces GL's knife-edge floats. The alpha is no longer the differentiator (both now emit the same `g_fragColor.a` at break); the RGB float at the break iteration is.

## Artifacts

- Code: `Rendering/Metal/Shaders/MetalShaders.metal` (opacity-break parity fix, §1) — **the only change** this update.
- Data: `/tmp/bc/fix_metal2.png` (post-fix Metal capture), `/tmp/bc/fix_metal3.png` (determinism re-run, byte-identical).
- Baseline images: `/tmp/bc/fix_gl.png` (clean GL), `/tmp/bc/u55c_gl_float.npy` (clean-GL float dump).
- Verified (python, /tmp/bc): fix_gl vs blend model 99.834%, fix_metal2 vs blend model 99.891%; fix_metal2↔fix_gl 188 px.
