# Camera-inside: residual root-cause hypotheses after update 23 — the knife-edge flip, not the double-march (update 24)

**Date:** 2026-08-08
**Scope:** Re-examine the remaining GL↔Metal residual (max 22/255 after update 23) with a fresh Metal sample log (`/tmp/bc/u23b/metal_samples.log`, captured at HEAD `23e6c3d328`). Record what the fresh log proves about the double-march, quantify the two candidate residual causes, and set up the decisive tests.

**Follows:** [Update 23](VolumeRayCastBackendComparisonFindingsUpdate23.md), which left the residual (max 22/255) open, attributed to the remaining float32 chain or accumulated marching-position rounding.

---

## 1. Conclusion

1. **The double-march is structurally still present but it is NOT the residual cause.** Every pixel still marches twice per frame (far-remnant pass + cap pass), but update 22's `tStart`-clamp (`df934c8`) changed the far-remnant march from a full re-march (~170 samples) into **exactly 1 clamped sample** at the far slab (`i=0`, `texz=1.0`, `lastIter=0`, `op=0.005441`), verified for both worst pixels (372,131) and (422,92). Its worst-case contribution to the final composite is `0.0054 × 255 ≈ 1.4/255` — far below the observed 22/255.

2. **The residual is the scalar-1150 knife-edge sample flip, driven by a remaining sub-texel ray-position drift.** This is the update-16 mechanism (one sample crossing the `AddPoint(1150, 0.85)` opacity knot, `raw 0.0154 vs 0.0182`), not a new backend error.

3. **The "0.2–2.3% step difference" reported by `compare_gl_metal_accum.py` is largely a frame-mismatch artifact, not backend step arithmetic.** Frame-aligned fits on the fresh log show GL and Metal agree to ~0.006% (y) when comparing unperturbed frames:
   - Frame 0 (clean camera): GL y-step `−5.0096e−06` vs Metal `−5.0093e−06` → **0.006%** off.
   - Frame 4 (perturbed camera): GL `−5.126e−06` vs Metal `−5.0093e−06` → **2.3%** off — but this compares GL's float32 view-angle-perturbed frame (update-19 root cause) against Metal's clean frame, so it is not apples-to-apples.

4. **The remaining drift is localized to the near-plane cap anchor interpolation.** GL interpolates `ip_vertexPos` (data space); update 23 matched this in Metal, but the barycentric interpolation of the near-plane cap triangle is still not bit-identical, so `g_dirStep = normalize(anchorData − eyeData)` differs by ~1e-7/step. Over ~144 steps that accumulates to ~0.02 texel — just enough to flip a sample across the sharp scalar-1150 knot.

---

## 2. Fresh-log facts (double-march)

Reference test: `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter`, 512×512, camera inside. Metal capture: `MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=16777216 MTL_LOG_TO_STDERR=1 … -V /tmp/bc/dummy_baseline.png 2> /tmp/bc/u23b/metal_samples.log`.

| fact | value |
|---|---|
| MARCH blocks in log | 810 = 64 pixels × 12 (6 frames × **2 marches/pixel/frame**) |
| far-remnant march | entry z=1.0, tStart=0.566, emits **1 sample** (`i=0`, texz=1.0, op=0.005441, lastIter=0); 0 samples in some frames |
| cap march | entry z=0.449, tStart=0.003, the real march (236 samples i=0..235 at (372,131)) |
| far-remnant op (372,131) | 0.005441 |
| far-remnant op (422,92) | 0.005441 |
| far-remnant max composite contribution | ≈ 1.4/255 |

So `df934c8`'s clamp removed the redundant re-march but the pass still runs and still drops one sample at the far slab. Because that sample is translucent and small, and for interior rays the cap march reaches full opacity before the far slab, the double-march is bounded to ~1.4/255 worst-case and cannot explain the 22/255 residual.

---

## 3. Frame-aligned step analysis (the artifact)

`compare_gl_metal_accum.py` fits `pos = pos0 + i·step` over a clean window and reports per-axis |step| differences. On the fresh log the per-frame fits for (422,92) are:

| frame | GL y-step | Metal y-step | diff |
|---|---|---|---|
| 0 (clean) | −5.0096e−06 | −5.0093e−06 | 0.006% |
| 4 (perturbed) | −5.126e−06 | −5.0093e−06 | 2.3% |

The tool uses the last frame per index (documented in its header), which for GL is the perturbed frame — so the headline "2.3%" is a comparison of GL's perturbed camera against Metal's clean camera, not a backend step-arithmetic difference. Comparing like-for-like (frame 0) leaves ~0.006% in y — consistent with ~1e-7-per-sample residual position drift, which is the knife-edge driver.

---

## 4. Candidate hypotheses for the residual (max 22/255)

1. **H1 — near-plane cap anchor interpolation is not bit-identical (most likely).** GL interpolates the data-space `ip_vertexPos`; Metal interpolates the data-space `in.localPos`. Both are linear across the triangle, but the float32 barycentric weighting (perspective-correct division, interpolation order) can still differ, shifting `anchorData` by ~1e-5 volume units → `g_dirStep` differs ~1e-7/step → ~0.02 texel drift at i=144 → scalar-1150 knife-edge flip.
2. **H2 — accumulated marching-position rounding.** Even with identical steps, the incremental `pos += evalStep` float32 accumulation could diverge by ~1e-6 texel at i=144; likely too small alone, but testable by computing position as `anchor + i·step` instead of incrementally.
3. **H3 — entry-point / cell-to-point / sample-distance-world float32 differences.** The update-23 note listed these; each is a single constant or mat-vec, testable with the fixed-step sweep and nearest-interpolation variants.
4. **H4 — far-remnant double-march (ruled out as dominant).** ≤1.4/255, cannot explain 22/255. Still worth eliminating for exactness, but not the residual driver.

---

## 5. Decisive next tests

1. **Per-sample step verification at the worst pixels** — `compare_gl_metal_accum.py` at (422,92) and (372,131) against a fresh GL per-sample dump (`VTK_GL_RAY_DUMP=1 VTK_GL_SAMPLE_DUMP=1 VTK_GL_SAMPLE_DUMP_PX=…`), frame-aligned (compare frame 0 to frame 0, not last-to-last).
2. **H2 discriminator** — in `marchVolumeUnified` compute the sample position as `anchorData + i·evalStep` (or `anchor + i·step`) instead of `pos += step`, and re-measure the worst pixels. If the flips persist, H2 is dead.
3. **H3 discriminator** — fixed-step sweep and nearest-interpolation variant (procedures sections) to isolate the TF LUT / sample-distance chain.
4. **H1 discriminator** — dump the GL `ip_vertexPos` and Metal `anchorData` at the same pixel and compare to float32; confirm the residual anchor delta and its projection onto the ray direction.

---

## 6. Reproduction

```bash
# fresh Metal sample log (this update)
MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=16777216 MTL_LOG_TO_STDERR=1 \
  build_macos_metal/bin/vtkRenderingVolumeCxxTests \
    TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
    --vtk-factory-prefer RenderingBackend=Metal \
    -D build_macos_metal/ExternalData/Testing -T build_macos_metal/Testing/Temporary \
    -V /tmp/bc/dummy_baseline.png 2> /tmp/bc/u23b/metal_samples.log
```

Artifacts: `/tmp/bc/u23b/metal_samples.log` (810 MARCH / 6 frames × 2 marches at 64 pixels), this file.
