# Camera-inside: a capture bug invalidated the "perturb has no effect" runs — the anchor DOES drive the knife-edge, and Metal/GL are both reproducible (update 34)

**Date:** 2026-08-08
**Scope:** Execute update 33's probe 1 (bit-exact step parity) via the anchor-perturbation debug uniform: perturb the interpolated anchor by a controlled amount and measure how the knife-edge residual responds. Also audit the capture procedure after a byte-identical outcome suggested the perturb was inert.

**Result:** The "perturb has no effect" result was **an artifact of a stale capture**: the test writes its render to the `-V` path, not to `Testing/Temporary/TestGPURayCastCameraInsideTransformation.png`, which was stale since 19:04. Every "byte-identical" comparison (gl_p0/mt_p0, mt_r1–r3, gl_r1–r3, the ab2/ab3 sweeps, the "GL silently falls back to Metal" conclusion) was comparing the **same stale image** (md5 `becc616…`). With fresh `-V` captures:
- **Metal is reproducible**: `cmp_base.png` ≡ `u33_mt.png` (md5 `4f858a…`).
- **GL is genuine and reproducible**: `cmp_gl.png` ≡ `u33_gl.png` (md5 `e67e54…`). GL does **not** fall back to Metal.
- **Real Metal-vs-GL baseline** (NoJitter camera-inside): 307 px ≥ 5, max |d| = 22 — matches u33.
- **The anchor perturb works**: `VTK_METAL_ANCHOR_PERTURB=0.5,0,0` changes the whole image (mean |d| 20, max 71).
- **The anchor hypothesis is confirmed**: at the knife-edge (372,131) d=22, perturb `1e-4,0,0` or `2e-5,0,0` drives d → 0 (exactly matches GL), while `-1e-5,0,0` reduces the total masked count 307 → 272. The ~1e-5 anchor difference is therefore live, but it is **per-pixel position-dependent**, so no single uniform perturb is a fix (the two zeroing cases raise the global count to 944/431).

**Follows:** [Update 33](VolumeRayCastBackendComparisonFindingsUpdate33.md).

---

## 1. The capture bug: renders land in the `-V` path, not in the test-name PNG

The test binary writes its final render to the file given after `-V` (the dummy baseline) inside the `-T` Temporary dir, not to `TestGPURayCastCameraInsideTransformation.png`. That file last changed at **Aug 8 19:04** and stayed frozen through every run in this session (21:27–21:48). Copying it after each run therefore always produced the same bytes.

Audit of `/tmp/bc/u34/*.png` (md5):

| group | md5 | verdict |
|---|---|---|
| gl_p0, mt_p0, mt_r1–3, gl_r1–3, Metal_t1/t2, OpenGL_t1/t2, mt_perturb, gl_perturb, mt_big_*×3, mt_ctrl_05, pre_ctrl | `becc616bc4b46cd755e6dfc4d9f0c671` | **all stale copies** of the 19:04 render |
| cmp_base, ab_0n0n0, u33_mt | `4f858a0a61b055ee4e60de13f6889e4f` | genuine Metal render |
| cmp_gl, u33_gl | `e67e54ab011da6cda893f9131b1751dd` | genuine GL render |
| cmp_p05 (perturb 0.5,0,0) | `1f4bd9b01162d84724d76acffdc46b41` | genuine perturbed Metal |
| ab_1ep5n0n0 / ab_p1ep5n0n0 / ab_1ep4n0n0 / ab_2ep5n0n0 / ab_0n1ep5n0 / ab_0n0n1ep5 | distinct | genuine perturbed Metal (sweep) |

Consequences: the "GL falls back to Metal" conclusion (gl_p0 ≡ mt_p0, "no MTL_EYE in gl log") was wrong — both files were the stale image. The GL logs still contained genuine `GL_*` lines because GL was really engaged. The earlier "within-session determinism t1==t2, r1==r2==r3" results are meaningless (same stale bytes). Session-to-session drift observations (mt_r1 vs u33_mt) were stale-vs-genuine comparisons.

## 2. Correct capture procedure

```
$BIN <Test> ... -V <Temporary>/<unique>.png    # render lands in <unique>.png
```

Always use a fresh `-V` filename per run (e.g. `cmp_base.png`, `cmp_p05.png`) and copy that file; never rely on `TestGPURayCastCameraInsideTransformation.png`. Note the earlier "GL-silent-fallback" tests should be re-run with a fresh `-V` name before trusting any GL-vs-Metal byte equality.

## 3. Genuine baseline: Metal vs GL (NoJitter camera-inside, last frame)

| metric | value |
|---|---|
| max |d| (of 255) | 22 |
| masked px (≥ 5) | 307 |
| delta mean | +0.087 |
| mean |d| | 0.117 |
| top px | (372,131) d=22 MT=[237,142,99] GL=[238,160,121]; (422,92) d=19 MT=[238,192,159] GL=[238,176,140]; (422,91) d=18; (393,173) d=15 |

This reproduces u33/update-33's numbers exactly (307 px, max 22, same knife-edges), confirming both backends are deterministic given a correct capture.

## 4. The anchor perturb is live and confirms the anchor hypothesis

The shader debug uniform `anchorPerturbData` (added via `VTK_METAL_ANCHOR_PERTURB`) is applied only in the camera-inside branch: `dirObj = normalize((anchorData + perturb) − eyePosData)` (MetalShaders.metal:3848). Fresh captures:

| perturb | ke (372,131) d vs GL | px ≥ 5 vs GL | max |d| |
|---|---|---|---|---|
| 0 (base) | 22 | 307 | 22 |
| +1e-4,0,0 | **0** | 944 | 19 |
| +2e-5,0,0 | **0** | 431 | 19 |
| +1e-5,0,0 | 22 | 349 | 22 |
| −1e-5,0,0 | 22 | **272** | 22 |
| 0,+1e-5,0 | 22 | 336 | 22 |
| 0,0,+1e-5 | 22 | 325 | 22 |
| 0.5,0,0 | — (global) | ~251417 | 71 |

Interpretation: an x-perturb of +1e-5…1e-4 (i.e. exactly the ~1e-5 data-space anchor discrepancy from update 30 probe 3) collapses the (372,131) knife-edge to bit-equal with GL, and the negative direction reduces the total divergent count. The anchor difference is **real and live** — update 30 probe 3's "perturb +1e-7 → residual 307→309 px" was measuring the same stale-capture noise and must be re-validated. Because the interpolated-anchor error varies with barycentric phase per-pixel, no single uniform perturb zeros all pixels at once; the fix must reproduce GL's `ip_vertexPos` interpolation itself (or its exact float32 chain), not add a constant offset.

## 5. Implication for the evalStep direction (update 33 probe 1)

The perturb path perturbs the *anchor*, not the *eye* — so it discriminates the anchor contribution to `dirObj`, which update 33 concluded was "already GL-equal to ~1e-5". The knife-edge response above shows that ~1e-5 anchor error is exactly at the sensitivity threshold: it flips the (372,131) edge. So the residual can be driven by the anchor at 1e-5 **and** by the evalStep accumulation drift (update 31/32, ~1e-4/step). Both candidate classes remain live and must be addressed by bit-exact porting (GL `ip_vertexPos` anchor interpolation; GL 4×4 `g_dirStep` composition; GL position-bounds termination), not by tuning a uniform.

## 6. Next probes

1. **Re-validate update 30 probe 3** (anchor perturb +1e-7) and update 33's "no change" survey with fresh `-V` captures before trusting any null result.
2. **Anchor parity**: dump Metal's `p.anchorData` and GL's `ip_vertexPos` at (372,131) last frame; diff to get the per-pixel anchor error sign/magnitude and confirm it matches the +x direction implied by the sweep.
3. **evalStep bit-exactness** (update 33 probe 1): CPU-compose GL's `cellToPoint * inverseTextureDataset` 4×4 and in-shader compute `(M * vec4(dirObj,0)).xyz * sampleDistanceWorld`.
4. **Termination parity** (update 32/33 probe 2, still pending): replace Metal's precomputed `maxSteps` with GL's position-bounds loop.

## 7. Reproduction

- Build: `./macos_metal_build.sh --resume --tests`.
- Capture: `$BIN TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter --vtk-factory-prefer RenderingBackend=Metal|OpenGL -D <ExternalData>/Testing -T <Temporary> -V <Temporary>/<unique>.png`; copy `<unique>.png`. Exit code 1 (baseline mismatch) is expected with the dummy baseline.
- Perturb sweep: set `VTK_METAL_ANCHOR_PERTURB=x,y,z` (comma list, `sscanf("%f,%f,%f")`), e.g. `0.5,0,0` or `1e-4,0,0`.
- Diff: the u34 analyze script (`/tmp/bc/u34/analyze.py`-style: delta mean, mean|d|, max|d|, masked ≥5 count, top pixels).

Artifacts: `/tmp/bc/u34/cmp_base.png` (genuine Metal), `/tmp/bc/u34/cmp_gl.png` (genuine GL), `/tmp/bc/u34/cmp_p05.png` (perturb 0.5,0,0), `/tmp/bc/u34/ab_*.png` (sweep), `/tmp/bc/u34/*.log` (capture logs). Stale group md5 `becc616…` retained for reference only.
