# Sample-count field eliminated: march-loop bounds exit moved to the GL g_dirStep lattice (update 69)

**Date:** 2026-08-10
**Status:** **milestone.** Implemented the fix hypothesized in update 68 §5: the
march-loop bounds-exit now tests the OpenGL `g_dirStep` lattice (Metal
`evalPoint`) instead of the raw `texStep` lattice. The u67 constant-scalar-volume
field **B collapses 5530 → 18 px** (all ±1 LSB); the full StepTF variant matrix
collapses 3–200× (m2 3711→21, A 742→6, E 13578→60, flat 314→5). The reference
test's 188-px floor is **unchanged** and is now confirmed to be the separate
u59–64 attribute-interpolator/knife-edge mechanism (all 14 |Δ|≥2 px are exactly
the known knife-edge pixels).

## 1. The fix (Rendering/Metal/Shaders/MetalShaders.metal, marchVolumeUnified)

Three surgical changes, all aligning exit decisions to GL's lattice:

1. **Bounds-exit test (line 4245)** — replaced `any(texLocalPos < 0) ||
   any(texLocalPos > 1)` (plain [0,1] texture space, `texStep` lattice, 0.202%
   longer step) with GL `TerminationImplementation` parity
   (`vtkVolumeShaderComposer.h` line 3366):
   ```metal
   // exit when the g_dirStep lattice position leaves the cell-to-point
   // adjusted bounds along any axis of travel
   const float3 adjTexMin = ctpOffset;
   const float3 adjTexMax = ctpOffset + ctpScale;
   if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
       any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f)))
   ```
   `evalPoint` is already bit-exact GL `g_dataPos` (verified through i=0..306 at
   the divergent pixel in update 68 §2); `adjTexMin/Max` replicate GL's
   `in_texMin/Max = AdjustedTexMin/Max` (cell-to-point of (0,0,0,1)/(1,1,1,1)).
   The old and new tests cross the **same exit planes**; only the lattice
   changes (texStep → evalStep), so grazing/corner behavior is untouched.
2. **`maxSteps` (line 4099)** — now `ceil((tEnd - firstT) / length(evalStep))`
   (was `/ p.stepSize`), matching GL `g_terminatePointMax =
   length(g_terminatePos - g_rayOrigin) / length(g_dirStep)`.
3. **`tTerminateMax` break (line 4881)** — `firstT + currentT * length(evalStep)
   >= p.tTerminateMax` (was `* p.stepSize`).

## 2. Fluke-proof verification protocol

Because the update-65 stale-md5 and update-67 zsh word-splitting incidents, every
number below was re-verified with: (a) outputs written to unique filenames and
re-confirmed present with correct size before use; (b) each backend run twice and
required byte-identical md5 across runs; (c) an early matrix run that silently
rendered background-only (env not applied) caught and redone. The StepTF
background-only trap (1 unique pixel, all images byte-identical) is easy to hit
and was hit once during this session.

## 3. Results (all 512×512, checkerboard dummy, VTK_STEP_WHEEL=1)

### B constant-scalar volume (mode 3, VTK_STEP_CONSTANT=2000, ramp 0→0.02)

| metric | before (u67) | after fix |
|---|---|---|
| diff px | 5530 | **18** |
| \|Δ\|≥2 | 794 | **0** |
| max Δ | 2 | **1** |
| sign | 100% GL>MT | 100% GL>MT (18× −1 LSB) |

- Both backends deterministic (GL1==GL2, MT1==MT2 md5-identical).
- Constant scalar ⇒ per-sample opacity identical ⇒ any delta is a **sample-count
  difference**; the 18 survivors are still n_MT = n_GL − 1 pixels, now only at
  exit-crossing margins within ~1 ulp of the bounds plane (18/262144 ≈ 0.007% ≈
  ulp-width/step-width fraction).
- Formerly-divergent gate pixels (439,281), (32,346), (373,466) now byte-identical.

### Reference test TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransform

| metric | before (u65 floor) | after fix |
|---|---|---|
| diff px | 188 | 183 |
| \|Δ\|≥2 | 14 | 14 |
| max Δ | 8 | 8 |

- **Unchanged** — the u59–64 floor survives the step fix.
- **All 14 |Δ|≥2 pixels are exactly the known u59–64 knife-edge set** ((397,110)
  max 8, (360,229) 4, (349,255) 3, (405,171) 3, plus the other 10 at ±2), and the
  remaining 169 px are all ±1. The reference residual is therefore the
  attribute-interpolator displacement family, NOT the sample-count field.
- Deterministic on both backends (md5-identical across two runs each).

### StepTF variant matrix (StepTF test, all VTK_STEP_WHEEL=1)

| variant | before (u67) | after fix | \|Δ\|≥2 now | max Δ now |
|---|---|---|---|---|
| m2 opacity-step (mode 2) | 3711 | **21** | 0 | 1 |
| A color-ramp const-op (mode 4, op 0.005) | 742 | **6** | 0 | 1 |
| C window-limited ramp (mode 5) | 286 | **187** | 0 | 1 |
| D 256³ (mode 3, VTK_STEP_DIMS=256) | 9215 | **4338** | 0 | 1 |
| D raw 64³ (mode 3, VTK_STEP_DIMS=0) | 12620 | **10281** | 1484 | 2 |
| E axis camera (mode 3, VTK_CAMERA_AXIS=z) | 13578 | **60** | 0 | 1 |
| flat near-zero (mode 3, ramp 0.0005) | 314 | **5** | 0 | 1 |

The sample-count field that dominated every variant is gone (everything ≤60
except the coarse-volume D rows). Remaining fields are ±1 LSB except D64's 1484
±2 pixels.

## 4. Doubts / hypotheses (open)

- **B's 18 survivors** are still ±1 sample flips. Candidate causes, in
  likelihood order: (a) a 1-ulp difference in the bounds plane — GL passes
  `in_texMin/Max` computed from the CPU `CellToPointMatrix`
  (`AdjustedTexMin/Max`, vtkVolumeTexture.cxx:1235-1236) while Metal recomputes
  `ctpOffset`/`ctpOffset+ctpScale` in-shader; the CTP matrix construction
  (formula/precision) is still unverified when this update was frozen; (b) a
  1-ulp difference in the interpolated anchor (`localPos` vs GL `ip_textureCoords`)
  shifting the whole lattice; (c) `evalStep` vs `g_dirStep` per-axis ~1e-9 diffs
  accumulating to ~3e-7 over 300 steps. Any of these shift the exit crossing by
  ~1 ulp, and 18 px ≈ the fraction of crossing margins within 1 ulp.
- **D64 still 10281 px (±1, 1484 at ±2)** — with 64³ texels the per-axis step /
  bounds geometry is ~8× coarser in step-index terms; whether this is the same
  1-ulp plane/anchor family amplified by short rays, or a second entry-side
  effect (non-anchored `evalPoint` drift: it starts as `cellToPoint(texLocalPos)`
  but advances by `evalStep` while `texLocalPos` advances by `texStep`) is open.
- Whether the reference's 169 ±1 px contain any residual sample-count component
  or are purely the interpolator floor.
- Whether switching to `g_dataPos`-lattice bounds for the **non-anchored**
  (camera-outside, grid-traversal) legacy march loops is needed for the CamOutside
  tests (not part of this update's scope).

## 5. Reproducibility

```
# B (constant-scalar) — both backends, twice each for determinism
env VTK_STEP_MODE=3 VTK_STEP_CONSTANT=2000 VTK_STEP_RAMP_MAX=0.02 VTK_STEP_WHEEL=1 \
  build_macos_metal/bin/vtkRenderingVolumeCxxTests \
  TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformStepTF \
  --vtk-factory-prefer RenderingBackend=Metal \
  -D build_macos_metal/ExternalData/Testing -T build_macos_metal/Testing/Temporary \
  -V /tmp/bc/x.png

# reference test
build_macos_metal/bin/vtkRenderingVolumeCxxTests \
  TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransform \
  --vtk-factory-prefer RenderingBackend=OpenGL \
  -D build_macos_metal/ExternalData/Testing -T build_macos_metal/Testing/Temporary \
  -V /tmp/bc/y.png
```

Compare with `Rendering/Metal/BackendComparisonTools/image_delta_profile.py` (for
≥5 mask) or numpy per-channel max-diff for full residual. Captures:
`/tmp/bc/bchk_gl1/2.png`, `/tmp/bc/bchk_mt1/2.png` (B, deterministic);
`/tmp/bc/ref_gl1/2.png`, `/tmp/bc/ref_mt1/2.png` (reference, deterministic);
`/tmp/bc/mtx_{m2,A,C,D256,D64,E,flat}_{OpenGL,Metal}.png` (variant matrix).
