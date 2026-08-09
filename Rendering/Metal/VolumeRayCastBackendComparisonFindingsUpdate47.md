# Metal's composite accumulation is proven self-consistent and arithmetic-association-insensitive; the ±1 field is a per-sample input (op/rgb) divergence on the GL side, localized to a few texels per ray — GL per-sample dump needed

**Date:** 2026-08-09
**Scope:** Today's session closed three open items by reading both backends' actual code (TF texture layout, TF table fill, TF/MIP/composite formulas) and then ran a full CPU float32 replay of Metal's accumulation from its own logged per-sample rows against the captured images. The replay is exact (15/15 gated differing pixels), and — critically — the accumulated-color difference vs clean GL is at the **float level (±0.001–0.002/channel)** with all arithmetic-reassociation variants producing identical results, which **rules out the accumulation arithmetic entirely** and pins the divergence on per-sample inputs.
**Target (confirmed):** Metal output must be bit-identical to **clean GL** (`RenderingBackend=OpenGL` without debug injection).

**Follows:** [Update 46](VolumeRayCastBackendComparisonFindingsUpdate46.md), [Update 45](VolumeRayCastBackendComparisonFindingsUpdate45.md).

---

## 1. Ruled out: TF v-coordinate, TF table contents, TF texel selection, MIP/Composite formula parity (source-level proof)

- **Metal TF texture is height 1** (`NewTexture2D(device, MTLPixelFormatRGBA32Float, tfWidth, 1, …)`), so the `float2(norm, 0.5)` v-coordinate selects row 0 exactly like GL's `vec2(norm, 0.0)` on its height-1 texture. The v-coordinate is a non-issue.
- **Table contents are bit-identical by construction.** Metal's `FillTransferFunctionRGBA32FWithPreIntegration` (`vtkMetalGPUVolumeRayCastMapper.mm:1378-1413`) and GL's `vtkOpenGLVolumeOpacityTable::InternalUpdate` (`vtkOpenGLVolumeOpacityTable.cxx:17-72`) are textually equivalent: same `GetTable`, same `> 0.0001f` gate, same `float(1.0 - pow(1.0 - double(a), factor))` COMPOSITE pre-integration with `factor = sampleDistance/unitDistance`, same ADDITIVE multiply. RGB tables are a raw `GetTable` on both sides (`vtkOpenGLVolumeRGBTable.cxx:21-36`). GL splits the table into 1-comp (opacity, sampled `.r`) and 3-comp (color) textures; Metal packs RGBA. All width-1024, RGBA32F, NEAREST, clamp-to-edge → `floor(norm*1024)` selects the same entry on both.
- **MIP output formula identical:** GL `ShadingExit` = `g_fragColor.rgb = g_srcColor.rgb * g_srcColor.a; g_fragColor.a = g_srcColor.a;` (`vtkVolumeShaderComposer.h:3156-3160`) == Metal `finalColor = float4(c.rgb * c.a, c.a)` (`MetalShaders.metal:4888-4889`). Both premultiply.
- **MIP tracking identical:** GL `if (l_maxValue.w < scalar.x || l_firstValue) l_maxValue.w = scalar.x;` (1-comp; `scalar = vec4(scalar.r)` after scale, so `.w == .x` — same value) vs Metal `if (firstBlendSample || mipMaxScalar < scalarNorm) mipMaxScalar = scalarNorm;`. Same strict-`<`, same first-value init, both track the scaled norm. Metal verified self-consistent from logs: final `MIPFINAL mip` == max over per-sample `scalarNorm` for all 68 gated rays.
- **Composite accumulate identical:** Metal `accumulatedColor = fma(weight, sampleColor*sampleOpacity, accumulatedColor); accumulatedOpacity = fma(weight, sampleOpacity, accumulatedOpacity)` (`MetalShaders.metal:4794-4795`), gate `sampleOpacity > 0.0f`, termination `>= 1 - 1/255` with clamp-to-1, `weight = 1 - accumulatedOpacity` — mirrors GL's `g_fragColor = (1.0f - g_fragColor.a) * g_srcColor + g_fragColor` with `g_srcColor.rgb *= g_srcColor.a` (`vtkVolumeShaderComposer.h:2651-2652`).

## 2. Metal's accumulation is exactly reproducible from its own log (float32 CPU replay)

- The u47 composite log carries per-sample `op`, `rgb`, and the running accumulator for gated pixels. Note: **each gated pixel is logged 6 times (6 deterministic renders, byte-identical traces)** — earlier "sort by i" replays interleaved renders and were wrong; file-order replay with the break at `accA >= 1-1/255` is correct.
- Replaying `accC = fma(1-accA, rgb*op, accC); accA = fma(1-accA, op, accA)` with an **exact float32 fma emulation** (fp64-exact product + round-to-fp32) from Metal's logged per-sample inputs reproduces Metal's stored image **15/15** for the gated pixels that differ, and matches Metal's logged final `accCol` to all printed decimals.
- The break sample is stable: e.g. pixel (319,25) crosses the opacity threshold at i=91 (accA 0.996440 > 0.996078) with a large margin, so the break cannot shift by rounding.

## 3. The accumulated-color difference vs GL is a float-level, not rounding-level, divergence — and not from the accumulate arithmetic

- For every gated pixel, Metal's replay float and GL's uint8 define GL's float bin. GL's accumulated color is **above** Metal's by 0.0003–0.0023 in the differing channels (e.g. (319,25) R: Metal 0.966324 vs GL ≥ 0.968628), and within ±~0.002 of Metal's (mixed sign) in the non-differing channels. This is a real ~1e-3 float divergence, not a store-rounding artifact.
- **Arithmetic reassociation is ruled out:** `fma`, `muladd`, `distrib` (`(1-a)*s+a` as `s-a*s+a`), and `fma(w*rgb, op)` variants all produce byte-identical accumulator values from the same per-sample inputs. A systematic ~1e-3 bias cannot come from ~1-ULP reordering of one multiply-add per sample.
- **"One extra sample before the opacity break" is refuted:** replaying Metal's rows with one additional sample accumulated matches GL 0/15 pixels.

## 4. Conclusion / where the divergence must live

- Since the tables, TF coordinates, formulas, and accumulation arithmetic are all proven identical, and Metal's own pipeline is self-consistent to the LSB, the remaining explanation is that **clean GL's per-sample inputs differ from Metal's** — i.e. GL selects a different volume texel (`raw`) or TF coordinate for some samples on some rays. Each such flip changes `op`/`rgb` discretely (nearest interpolation), and a few flips per ray produce exactly the observed ±0.001–0.002 accumulated float spread, which at the final rounding bin boundary manifests as the asymmetric 63,647-down / 40-up field.
- This contradicts the earlier assumption that clean-GL geometry == debug-GL geometry == Metal geometry: the debug-injected GL is not clean GL (update 44), so a sub-texel geometric divergence between **clean** GL and Metal is still live and is now the leading mechanism.
- **MIP (102 px) and composite (63,690 px) are separate phenomena:** MIP never accumulates opacity (max-scalar only), so the composite opacity-bias mechanism cannot affect it; its larger per-pixel diffs (2–24 LSB, all at the skin/bone surface) point to a per-ray max-scalar texel difference, consistent with the same underlying clean-GL-vs-Metal sample-position/texel divergence.

## 5. Next step (decisive experiment)

- Capture **clean GL's per-sample positions, raw texels, and TF coordinate** for the gated pixels **without corrupting its output**. The previous shader-injection dump corrupted clean GL (update 44); a non-interfering route is to dump GL's uniforms/step lattice at the host (GL_TEX lines already exist) and **CPU-replay GL's exact `g_rayOrigin + g_dirStep*g_currentT` lattice** against the volume texels and TF tables, comparing GL's replayed final float to its stored image (as done here for Metal). The Metal replay already proves the method closes to the LSB on the Metal side; applying the same replay to GL's lattice decides whether GL's geometry/texels differ from Metal's (and where), giving the exact per-sample flip to chase.

## Artifacts

- `/tmp/bc/u47_metal.log` (~54k SAMPLE/MARCH/FINAL rows, 6 deterministic renders per gated pixel), `/tmp/bc/u47_metal.png`, `/tmp/bc/u47_gl.png`.
- Replay scripts run inline: exact float32 fma emulation via fp64-exact product + fp32 round; `int(round(f*255))` store model matches Metal's image.
