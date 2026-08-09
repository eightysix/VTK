# The last 188 px are knife-edge texel-selection flips driven by a ~1-ulp interpolated-anchor difference: the frame-aligned GL float dump now proves Metal's gf genuinely differs from GL's gf (mean 0.31 u8, max 8.25 u8) with negligible alpha deltas, and the frame-6-aligned GL lattice matches Metal's step bit-for-bit (~5e-8 texels) but the anchor differs by ~5e-5 texels — the residual is rasterizer-interpolation hair, not composite arithmetic (update 59)

**Date:** 2026-08-09
**Scope:** After update 58, the operation-order bisect on the 174 px failed (mul+add moved only 4 px) and the GL float dump was known to be frame-1 vs a frame-6 image. This update: (a) fixes the GL float dump to be frame-aligned (dump every frame, last write = stored image frame), (b) measures the true Metal-vs-GL gf delta at the 188 px, (c) compares the frame-6-aligned lattice (anchor/step/clip) between the backends at the 14 largest-delta pixels, and (d) identifies the residual as a ~1-ulp interpolated-anchor difference causing nearest-texel flips at grid-aligned rays.
**Target (unchanged):** Metal bit-identical to **clean GL** (`RenderingBackend=OpenGL`, no debug injection).

**Follows:** [Update 58](VolumeRayCastBackendComparisonFindingsUpdate58.md) (feasibility: 174 ±1 px plausible, 14 knife-edge px likely unattainable), [Update 57](VolumeRayCastBackendComparisonFindingsUpdate57.md) (blend-clamp fix: 63,692 → 188 px).

---

## 1. Tooling fix: the GL float dump is now frame-aligned

`DumpCleanGLFloats` (vtkOpenGLGPUVolumeRayCastMapper.cxx:4945) had `static bool dumped` — it ran **once (frame 1)** while the stored image is the **last frame (6)** and the camera animates between frames. Knife-edge rays are hypersensitive to the camera pose, so the frame-1 dump disagreed with the frame-6 image by up to 6+ u8 (update 58 §3) — hiding the real gf difference.

- **Change:** removed the `dumped` gate; the dump now re-renders and overwrites the output file on **every frame**, so the final file content is frame 6, aligned with the stored image.
- **Caveat discovered:** running the dump **corrupts the real frame** in the same run (u60 dump-run image differs from clean GL at 64k px; clean GL is otherwise deterministic, 0 px vs u59). So the dump-run's stored image is unusable — use the dump for gf only, and pair it with a separate **clean** image (u60_gl_clean.png == u59_gl.png byte-identical).
- **Validation of the aligned dump:** `GL clean image == round_half_even((gf_dump + 26/255·(1−a))·255)` at **262,141 / 262,144** px. The 3 failing px ((323,225),(203,278),(400,467)) are ultra-boundary straddlers (gf sits ~0.4 u8 from the rounding boundary, so a sub-ulp re-render jitter flips them) — and **none of the 3 is in the 188 Metal-diff set**, so the aligned dump is trustworthy at every pixel we care about.

## 2. The 188 px gf differences are real (update 58's near-zero estimate was a frame artifact)

Metal gf vs **frame-aligned** GL gf at the 188 diff px (stored-image diff between clean GL and Metal):

| metric | value |
|---|---|
| gf |Δ| mean | **0.31 u8** |
| gf |Δ| median | 0.07 u8 |
| gf |Δ| max | **8.25 u8** |
| channels >0.5 u8 | 111 |
| channels >1 u8 | 50 |
| channels >2 u8 | 9 |
| alpha |Δ| mean / max | 0.000141 / 0.001548 |

This is 100× larger than update 58's estimate (mean 0.00004 u8), which was computed against the **frame-1** GL dump. The gf difference is genuine.

## 3. Signature: large color delta + negligible alpha delta = knife-edge texel selection flips

At the 14 big-diff px (|stored Δ|>1), gf deltas reach 8.25 u8 while alpha deltas stay ≤0.00086:

- (397,110): gf ch2 8.25 u8, ch1 6.85 u8, alpha Δ 0.00048, Metal lastIter 179
- (360,229): gf 2.78–3.35 u8, alpha Δ 0.00024
- (338,432): gf ch2 1.76 u8, alpha Δ **0.00000**
- (9,18): gf 1.95–2.36 u8, alpha Δ 0.00019

Color changing by ~0.03 (8 u8) while opacity changes by <0.0005 is **not** a composite-arithmetic difference (the composite chains are verified bit-exact at the gated pixels, updates 51/53) and **not** a sample-count break (that would move alpha proportionally). It is the signature of a **nearest-texel selection flip** at a grid-aligned (knife-edge) ray: adjacent texels share opacity but differ in color (e.g. a tissue boundary), and a sub-ulp lattice hair moves one backend's sample across the texel boundary.

## 4. Frame-6-aligned lattice comparison at the 14 big-delta pixels

Added the 14 pixels to both dump gates (Metal `debugMarchGate` `pxOkKnife`; GL `GL_RAY` list 15 → 29) and captured frame 6 from each backend:

- **Step (Metal `evalStep` vs GL `g_dirStep`):** essentially **bit-identical** — deltas ≤ ~6e-8 **texels** (≈1e-10 in [0,1]). Update 48's 1.6e-7 step bound was from the debug-GL compile; the aligned clean-step matches to below 1 ulp.
- **Anchor (GL `g_rayOrigin` vs Metal first sample `localPos + evalStep`):** differs by **2e-5 … 6.7e-5 texels** (~1e-7 in [0,1] ≈ **~1 ulp** of the float32 interpolated texcoord).
- **Interpolated clip:** x/y/w match to ≤ ~6e-8; z is not comparable across dumps (GL_RAY `ip_debugClip.z` vs Metal `out.position.z` use different conventions — clip.z reads 0.00019 vs 0.192).

So the entire remaining lattice difference is a **~1-ulp interpolated-anchor offset**, and at knife-edge rays the ~5e-5-texel offset crosses a nearest-texel boundary, flipping the per-sample texel and shifting gf by 0.3–8 u8.

## 5. The operation-order bisect (update 58 §5.1) is dead

Switching Metal's composite color chain from `fma(1−accA, src, accC)` to explicit `mul+add` moved only **4 px** field-wide (185 → 188-style residual unchanged). With fast-math off and fp-contract off, the MSL genuinely executed mul+add — so GL is not doing plain mul+add; the composite is not the residual. Alpha and color chains were already pinned to the exact fma forms by the 68-gated-pixel sweep (updates 51/53).

## 6. Doubts / hypotheses (open)

1. **The ~1-ulp anchor difference is rasterizer interpolation hair.** Metal's vertex shader already mirrors GL's exact contraction for both clip (`matrixMulStrict` + the GLSL [mul,fma,fma,mul+add] dot-product pattern) and per-vertex texcoord (`cellToPointScale*uvx + cellToPointOffset`). GL and Metal both rasterize on the same Apple M2, so if the per-vertex values AND the pixel-center (gl_FragCoord vs in.position.xy) were bit-identical, the interpolated anchor should be bit-identical too. The residual suggests one of: (a) a per-vertex clip or texcoord still off by 1 ulp (an un-observed contraction pattern in the GLSL compile), (b) a 1-ulp difference in the pixel-center interpolation coordinate, or (c) a rasterizer-interpolation arithmetic difference that is not reproducible from the shader side.
2. **If the 1-ulp anchor cannot be closed, the 188 px (0.072% of the field) are the irreducible bit-parity floor for this GL-on-Metal/Metal pair.** All evidence says both backends are self-consistent, deterministic, and arithmetically correct; the differences are sub-ulp ray-lattice flips at grid-aligned rays.
3. **Next candidate bisect if continuing:** dump the per-vertex clip/texcoord (Metal `vertex_volume_main` log already exists) vs GL's per-vertex values for the same proxy triangles at the knife-edge pixels, and check the pixel-center (in.position.xy) vs gl_FragCoord at 1-ulp level. Failing that, quantify and document the knife-edge floor.

## Artifacts

- Code: `Rendering/VolumeOpenGL2/vtkOpenGLGPUVolumeRayCastMapper.cxx` (DumpCleanGLFloats now per-frame → frame-aligned; GL_RAY list 15 → 29 px), `Rendering/Metal/Shaders/MetalShaders.metal` (`pxOkKnife` pixel group in `debugMarchGate`).
- Data: `/tmp/bc/u60_gl_float.raw` (frame-6 aligned GL gf dump), `/tmp/bc/u60_gl_clean.png` (clean GL image), `/tmp/bc/u61_gl.log` (GL_RAY lattice), `/tmp/bc/u61_metal.log` (Metal STEP lattice).
- Scripts: `/tmp/bc/compare_gf_u60.py`, `/tmp/bc/check_gl_dump_u60.py`, `/tmp/bc/decide_u60.py`, `/tmp/bc/lattice_u61.py`, `/tmp/bc/clip_u61.py`.
- Verified (python, /tmp/bc): aligned GL dump reproduces clean image at 262,141/262,144 px (3 ultra-boundary px, none in the 188 set); gf |Δ| mean 0.31 u8 / max 8.25 u8 at the 188 px with alpha |Δ| ≤0.0015; step bit-identical to ~5e-8 texels while the anchor differs by ~2e-5–6.7e-5 texels (~1 ulp).
