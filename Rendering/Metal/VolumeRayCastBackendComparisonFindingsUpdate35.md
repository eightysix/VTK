# Per-vertex clip byte-comparison: cap mesh, P/V/M matrices, and vertex clip are all bit-identical **except** the projection Z row (nearz convention) and a ≤1-ULP mat4-product rounding difference (update 35)

**Date:** 2026-08-08
**Scope:** Complete the per-vertex clip-space byte-comparison (GL vs Metal) for `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter` (camera-inside, 512×512, NoJitter): dump the uploaded cap mesh, the three clip-chain matrices (P, V, M), and each vertex's exact `P*V*M*v` clip from both backends, and compare bit-for-bit. This is the vertex-level grounding for the interpolated-anchor discrepancy chased in updates 30/34.

**Result:** After fixing two capture bugs (Metal `os_log` line truncation at ~213 chars; `%0.8g` = 8 significant digits, which does **not** round-trip float32 and collapsed distinct vertices), the comparison is:
- **Cap mesh is bit-identical** between GL and Metal: all 95 uploaded float32 positions (`GL_CAPVERT` ≡ `MTL_CAPVERT`). The earlier "GL has 94 / Metal 95" and "2-ULP position difference" observations were artifacts of a broken GL strip `pos` field and the `%0.8g` collapse.
- **`V` (modelView) and `M` (volumeToWorld) are byte-identical** (64 bytes each). **`P` (projection) differs in exactly 2 of 16 floats**, both in the Z row — `Element[2][2]` and `Element[3][2]` — matching the documented nearz convention (GL nearz = −1, Metal nearz = 0). Rows 0, 1, 3 match as designed.
- **Per-vertex clip (`P*V*M*v`):** Metal's x/y are a global sign negation of GL's (coordinate convention); magnitudes differ by ≤1 ULP on the majority of vertices (x-magnitude bit-exact on 15/94, y on 14/94, w signed bit-exact on 9/94); z is dominated by the nearz offset and is not directly comparable. Since mesh, V, M, and P-rows-0/1/3 are all bit-identical, the ≤1 ULP x/y/w differences point to a **rounding-order (FMA vs mul+add) difference in the GLSL vs MSL matrix products** — a candidate source for the ~1e-5 interpolated-anchor class at the vertex stage, before any interpolation.

**Follows:** [Update 34](VolumeRayCastBackendComparisonFindingsUpdate34.md).

---

## 1. Capture-infrastructure fixes (both were silently corrupting data)

1. **Metal `os_log` truncation.** Lines retrieved via `log show`/`log stream` (MTL_LOG_TO_STDERR) are cut at ~213 chars. The original `modelPos=(%0.8g, %0.8g, %0.8g) clip=(%0.8e, %0.8e, %0.8e, %0.8e)` clipped the tail (clip.w printed as `3.`). Shortened the format so the whole line fits.
2. **`%0.8g` = 8 significant digits is lossy for float32.** `%0.8g` prints 8 sig figs; float32 needs 9 for a guaranteed round-trip. Distinct mesh vertices whose float32 positions differ by 1–2 ULP collapsed to the same 8-digit decimal, so the `modelPos`→mesh-index map was wrong for a large fraction of vertices (only 61 distinct `modelPos` among 95 invocations). Fixed by:
   - adding `uint vertexId [[vertex_id]]` to `vertex_volume_main` and logging it — the vertex index is now authoritative, and
   - switching to `%.9g` (9 sig figs, round-trips) for `modelPos`/`clip`.

   Current Metal log line (all 95 last-frame invocations parse cleanly, `vid` 0..94):
   ```
   vertex_volume_main vid=%u modelPos=(%.9g, %.9g, %.9g) clip=(%.9g, %.9g, %.9g, %.9g)
   ```

## 2. GL strip-dump correctness audit (the strip was partially broken)

The GL per-vertex dump (`VTK_GL_VERTEX_DUMP=1`) renders each cap vertex k as a tiny triangle at a fixed strip position (`baseX = −0.998 + k*(5/256)`, i.e. pixel `x ≈ 5k+1`, rows ~20–22) and reads back encoded floats. With pixel coordinates added to the log (`GL_VERT <vid> px=(x,y) clip=(…) pos=(…)`):

- **`clip` per pixel is reliable**: each pixel's clip equals the clip of the triangle under it (verified by matching to Metal `vid` clips). The cap mesh has **95 indices but only 65 distinct positions** (shared corners of the densified/clipped box), so identical clip values legitimately repeat across distant triangles.
- **`pos` field is garbage**: the debug override `return`s early out of `ComputeClipPositionImplementation` (see `vtkVolumeShaderComposer.h`), which skips `ip_vertexPos = in_vertexPos` in `main()`. The 5–7 channels decode undefined/interpolator-leftover data. Do not use `pos` for alignment.
- **`vid` label is unreliable**: decoded `ip_vid` is correct for triangle 1 but off-by-one for k≥2. Do not use `vid` for alignment. **Align by pixel position instead: `k ≈ round((px − 1) / 5)`.**

Both faults are cosmetic for the comparison (clip is what matters) but must not be misused. The earlier "GL emits only 93 of 95 vertices" is a decode artifact: `vid` 0 encodes to 0.0 and is filtered by `vals[0] != 0`, and the off-by-one shifts the tail.

## 3. Cap mesh is bit-identical (GL_CAPVERT ≡ MTL_CAPVERT)

GL dumps the uploaded `BBoxPolyData` float array; Metal dumps its vertex buffer. All 95 vertices match byte-for-byte, e.g. vertex 0 = `0x4349999a 0x42b17573 0x429a8676` in both, vertex 1 = `0x4349999a 0x00000000 0x430a0000` in both. The two backends rasterize the **same** proxy geometry, so any clip difference is not input geometry.

## 4. Clip-chain matrices (GL_CLIPMAT vs MTL_CLIPMAT)

Both backends dump P, V, M as 16 little-endian float32 bytes (GL replicates `SetUniformMatrix`'s `data[i] = Element[i/4][i%4]`). Comparison of the last frame:

| matrix | GL | Metal | result |
|---|---|---|---|
| `V` modelView | `in_modelViewMatrix` | `ModelViewMatrix` (from `vtkMetalCamera::GetCachedSceneTransforms`) | **IDENTICAL (64 bytes)** |
| `M` volume→world | `in_volumeMatrix[0]` (`VolMatVec`) | `VolumeToWorldMatrix` | **IDENTICAL (64 bytes)** |
| `P` projection | `in_projectionMatrix` (`cam->GetKeyMatrices`) | `ProjectionMatrix` | **2/16 floats differ, both Z row** |

`P` differences:
- `Element[2][2]`: GL `0xbf80419a` ≈ −1.0007, Metal `0xbf8020cd` ≈ −1.00025
- `Element[3][2]`: GL `0xbec54381` ≈ −0.3853, Metal `0xbe454381` ≈ −0.1926 (≈ half)

These are exactly the nearz convention: GL maps near → −1, Metal near → 0 (Metal clip Z in [0,1]). Rows 0, 1, 3 (which drive NDC x, y, w) are identical, matching the design comment in the shader. The Z-row difference is expected and does not affect the XY/w barycentric weights.

## 5. Per-vertex clip (94 of 95 vertices compared; GL `vid` 0 absent by decode filtering)

Aligned by pixel-derived k (GL) and `[[vertex_id]]` (Metal), 94 vertices common:

| component | GL vs Metal |
|---|---|
| x | sign negated (Metal convention), magnitude bit-exact on **15/94**, else ≤1 ULP |
| y | sign negated (Metal convention), magnitude bit-exact on **14/94**, else ≤1 ULP |
| w | signed bit-exact on **9/94**, else ≤1–2 ULP |
| z | dominated by the nearz offset (0.00019 vs 0.192 for the near-cap vertices); not comparable |

Example (vertex k=2, position (201.6, 0, 61.4885)): GL clip.x `0x43b6bcff` vs Metal `0xc3b6bcff` — **magnitude bit-identical**, only sign differs. Vertex k=1: GL `0x43cfba6a` vs Metal magnitude `0x43cfba6b` — 1 ULP apart. The mix of exact and ≤1 ULP matches across vertices, with all inputs bit-identical, is the signature of a **rounding-order difference (FMA vs mul+add) in the matrix-matrix/matrix-vector products** between the GLSL and MSL compilers, not an input mismatch.

**Consequence for the anchor problem (updates 30/34):** the interpolated anchor is a linear blend of the vertex clips via barycentric weights. A ≤1 ULP clip difference at the vertices feeds a ≤~1e-5 relative data-space anchor difference after interpolation/denormalization — the same magnitude class as the knife-edge driver identified in update 34. The vertex-stage rounding difference is therefore a live candidate alongside the `evalStep` accumulation drift (updates 31/32) and must be discriminated before concluding which one dominates.

## 6. Next probes

1. **Confirm the FMA/rounding hypothesis for `(P*V)*M*v`:** emulate the float32 chain on the CPU with both plain mul+add and fused FMA against the dumped matrices/positions; whichever matches the GPU clip bit-for-bit identifies the compiler behavior. (GLSL and MSL both compile down to native; either may emit FMA.)
2. **Dump the combined `(P*V*M)` matrix** from each backend's shader (one extra matrix product per vertex) to see whether the difference originates in the matrix-matrix multiply (`P*V`) or the final matrix-vector product.
3. Re-run update 34 probe 1 (step parity at (372,131)) with fresh `-V` captures now that the capture procedure and the vertex logs are clean.

## Files touched this session

- `Rendering/VolumeOpenGL2/vtkOpenGLGPUVolumeRayCastMapper.cxx`: GL per-vertex strip driver (~4625), `GL_VERT` print with pixel coords, new `GL_CLIPMAT` matrix dump in `SetCameraShaderParameters` (~4104).
- `Rendering/VolumeOpenGL2/shaders/raycastervs.glsl`: debug outputs (`ip_debugClip`, `ip_debugClipFlat`, `ip_vid`, `in_debugVertexMode/Count`).
- `Rendering/VolumeOpenGL2/vtkVolumeShaderComposer.h`: `ComputeClipPositionImplementation` debug override (strip positions, `ip_debugClipFlat`, `ip_vid`).
- `Rendering/Metal/Shaders/MetalShaders.metal`: `vertex_volume_main` gains `[[vertex_id]]` and full-precision `%.9g` log (~2965).
- Artifacts: `/tmp/bc/u35/gl_vert.log` (GL mesh/matrix/strip dumps), `/tmp/bc/u35/mt_full.log` (Metal mesh/matrix/vertex logs).
