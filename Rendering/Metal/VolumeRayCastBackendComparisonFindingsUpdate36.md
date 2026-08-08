# Vertex clip is now bit-identical to GL: root cause was the matrix-vector multiply rounding order + FMA contraction, not the inputs (update 36)

**Date:** 2026-08-09
**Scope:** Close out the vertex-stage hypothesis from update 35 (≤1 ULP clip = rounding-order/FMA difference in the GLSL vs MSL matrix products) by enumerating every rounding configuration (multiplication order permutation × per-term FMA bitmask) in a CPU float32 emulation against the dumped P/V matrices and both backends' per-vertex clips, then make the Metal vertex shader reproduce GL's exact arithmetic.

**Result:** The emulation pinpoints GL's clip codegen as `(P*V)*W` with **`[0,1,2,3]` summation in both the matrix product and the final vector multiply**, and the vector multiply contracted as `mul, fma, fma, mul+add`. The Metal shader now hand-writes exactly that pattern (with reassociation and contraction disabled via `#pragma clang fp`), and all **94/94** common cap vertices match GL **bit-for-bit on x/y/w** (sign convention accounted; z excluded by the nearz convention), across all 6 frames × 95 invocations (570 `vertex_volume_main` lines). Pixel-level, the fix collapses the interpolated-anchor-driven knife edges from update 34: masked (≥5) pixels drop **307 → 130**, max |d| **22 → 14**, and the (372,131) knife-edge goes **22 → 0** (bit-equal), (422,92) **19 → 1**.

**Follows:** [Update 35](VolumeRayCastBackendComparisonFindingsUpdate35.md).

---

## 1. The emulation setup and a critical tooling caveat

Emulation (`/tmp/bc/u36/emul5.c`): load `P` and `V` (the `MTL_CLIPMAT`/`GL_CLIPMAT` dumps; `M` = identity for the NoTransform test), enumerate all 24 multiplication-order permutations for the matrix product `P*V` and the vector multiply, and for each term ≥ 1 a per-term FMA bitmask (terms 0 and the first product are plain multiplies by construction). The verdict per vertex: GL compare is **signed** bit-exact on the dumped GL clip for k ∈ {0,1,3}; MT compare is **magnitude** bit-exact (sign bit masked) on the dumped Metal clip. `v.w` is always `1.0f`.

**CRITICAL tooling caveat:** the emulator's `dotx()` loop body `t = t + a[i]*v[i]` must be compiled with `-ffp-contract=off`. Clang's default `-ffp-contract=on` (plain `cc -O2`) FMA-contracts that expression inside the loop, so the `fm`/`fm2` bitmasks were meaningless — every "fm=00 = no FMA" run was silently FMA'd, which is why the first pass (emul6, `cc -O2`) reported that plain `[0,1,2,3]` reproduced GL 94/94. Honest rebuilds (`emul5h`, `emul8_nc`, compiled with `-ffp-contract=off`) show that **GL's clip actually uses FMA** — a strict no-FMA emulation only reproduces GL 33/94.

## 2. GL's exact arithmetic (honest, `-ffp-contract=off`)

With the FMA bitmask honored, the search finds GL reproduced 94/94 **only** by:

- **matrix product** `P*V` (and hence `(P*V)*W`): permutation **0 = [0,1,2,3]**; FMA bitmask irrelevant (all 16 values win on this data — the sparse clip-chain matrices make the matrix-product rounding benign here).
- **vector multiply**: permutation **0 = [0,1,2,3]** with per-term FMA bitmask ∈ **{0x06, 0x07, 0x0e, 0x0f}** — i.e. bits 1 and 2 always set (terms 1 and 2 are fused), bit 3 (last term) is a don't-care.

Equivalent codegen for the dot (`c = m[0][r]*v.x + m[1][r]*v.y + m[2][r]*v.z + m[3][r]*v.w`):

```
t = m[0][r] * v.x;
t = fma(m[1][r], v.y, t);   // term 1 fused
t = fma(m[2][r], v.z, t);   // term 2 fused
t = t + m[3][r] * v.w;      // term 3 plain mul+add (≡ fma here, see §5)
```

This is the textbook contraction of a straight-line dot: first term plain multiply, middle terms fused, last term an add. The old fast-math Metal shader's builtin `*` produced `[3,0,1,2]` (reassociated), which is why the Metal clip was off by ≤1 ULP on most vertices.

## 3. Metal compiler behavior (verified in AIR)

- `MTLCompileOptions.fastMathEnabled` **defaults to YES**. With it on, even a hand-written `[0,1,2,3]` multiply is reassociated back to `[3,0,1,2]` and recontracted (first fix attempt, `mt_full.log`: still 3/94; `xcrun metal` AIR disassembly confirmed the reassociation).
- `#pragma clang fp reassociate(off)` + `#pragma clang fp contract(off)` **are honored by the Metal compiler even under `-ffast-math`**: the AIR for the instrumented functions shows strict `[0,1,2,3]` accumulation, no `fast` flags, and no compiler-generated `ffma`. (Pragma alone → 76/94: x/y all exact, 18 near-face `w` residuals were the still-contracted middle terms.)
- `options.fastMathEnabled = NO` additionally disables fast math for the whole library; with the pragmas plus the hand-written `fma()` calls the result is stable and byte-reproducible.

## 4. The fix

1. **`Rendering/Metal/Shaders/MetalShaders.metal`** (`vertex_volume_main`):
   - `#pragma clang fp reassociate(off)` + `#pragma clang fp contract(off)` before the clip code (line ~2929).
   - `matrixMulStrict()`: strict `[0,1,2,3]` 4×4 multiply, first term plain, remaining terms explicit `fma()` — replaces the builtin `*` for `(P*V)*W`.
   - Hand-written vector multiply `mul, fma, fma, mul+add` per row.
2. **`Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm`**: `vtkMetalVolumeCompileOptions()` now **always** returns an options object with `fastMathEnabled = NO` (previously returned `nil` in production). Shader-logging flags (`enableLogging`, `languageVersion`, `VTK_METAL_ENABLE_LOGGING` macro) remain gated behind `#ifdef VTK_METAL_ENABLE_LOGGING`.

Build: `./macos_metal_build.sh --resume` (shader is embedded — `MetalShaders.metal` → generated `vtkMetalShaders.cxx` → framework).

## 5. Verification

**Vertex clips (the comparison metric used throughout updates 33–35):** `compare.py` (signed on x/y/w, magnitude variant) on the dumped last-frame clips — **signed-exact 94/94, magnitude-exact 94/94**. All 6 frames × 95 invocations parse cleanly. Result is stable across the two final shader variants (`mt_full6` = fma-mv only, `mt_full7` = fma-mv + all-fma matrix product; both 94/94).

**Robustness note on the "don't-care" last term:** in the vector multiply, the last term is always `m[3][r] * v.w` with `v.w = 1.0f` exactly, so `fma(m[3][r], 1.0, t) ≡ t + m[3][r]` always — the term-3 fma/add choice can never diverge. For the matrix product, the all-fma variant was chosen (matches the same GLSL contraction pattern and remains in the winning set on this data).

**Pixel level** (genuine fresh captures, `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter`, Metal vs GL):

| metric | update 34 baseline | now |
|---|---|---|
| masked px (≥5) | 307 | **130** |
| max \|d\| | 22 | **14** |
| (372,131) | d=22 | **0** |
| (422,92) | d=19 | **1** |
| top residual | (151,384),(141,381),(92,422) d=14 | symmetric about the camera diagonal, on the near-face clip edges |

GL render md5 `e67e54…` unchanged (genuine, reproducible). The (372,131) pixel — the knife-edge whose response to `VTK_METAL_ANCHOR_PERTURB` implicated the interpolated anchor — is now **bit-equal**, confirming the vertex-stage rounding difference was the driver of that edge. The remaining 130 px are fragment-stage residuals (see §6).

**Fragment still healthy under `fastMathEnabled=NO`:** debug-pixel logging intact (1026 SAMPLE, 6 MARCH lines), no shader compile errors. **Note:** `fastMathEnabled=NO` changes numerics library-wide; every earlier fragment-stage result measured under fast math (e.g. `evalStep` drift, updates 31/32/33) is no longer valid as-is and must be re-measured.

## 6. Next probes

1. **`evalStep` bit-exactness** (update 34 6.3, update 33 probe 1): CPU-compose GL's `cellToPoint * inverseTextureDataset` 4×4 and compute `(M * vec4(dirObj,0)).xyz * sampleDistanceWorld` in-shader instead of Metal's CPU-precomposed uniform — now measurable with a clean vertex stage.
2. **Termination parity** (update 34 6.4, update 32/33 probe 2, still pending): replace Metal's precomputed `maxSteps` with GL's position-bounds loop.
3. **Re-measure** the fragment drift numbers from updates 31/32/33 under the new strict compile options before porting.

## Files touched this session

- `Rendering/Metal/Shaders/MetalShaders.metal`: pragmas, `matrixMulStrict()`, hand-written clip multiply (~2922–3001).
- `Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm`: `vtkMetalVolumeCompileOptions()` (~76–95).
- Artifacts: `/tmp/bc/u36/gl_vert3.log` (GL reference clips), `/tmp/bc/u36/emul_mat_t.txt`, `/tmp/bc/u36/emul.csv` (94-vertex CSV), `/tmp/bc/u36/emul5.c` + honest rebuilds `emul5h`/`emul8_nc` (`-ffp-contract=off`), `/tmp/bc/u37/mt_full.log` (attempt 1, 3/94) … `mt_full5.log` (33/94) → `mt_full6.log`/`mt_full7.log` (94/94), `/tmp/bc/u37/compare*.py`, `/tmp/bc/u37/glref.png` (GL) + `/tmp/bc/u37/mtfix.png` (fixed Metal) for the pixel diff.

## Reproduction

- Vertex capture: `MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=16777216 MTL_LOG_TO_STDERR=1 $BIN <Test> --vtk-factory-prefer RenderingBackend=Metal -D <ExternalData>/Testing -T <Temporary> -V dummy.png 2> log`; expect 570 `vertex_volume_main` lines; compare x/y/w vs `gl_vert3.log`.
- Pixel capture: use a bare `-V <name>.png`; the render is written to `<Temporary>/<name>.png` (the `-V` absolute path is only the (missing) baseline reference). Diff with the u34 `analyze.py` style.
