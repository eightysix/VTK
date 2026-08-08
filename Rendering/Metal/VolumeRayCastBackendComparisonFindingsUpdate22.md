# Camera-inside: implement GL's anchor-march for Metal's far remnants (clamp tStart to the anchor) — and retract the contaminated GL-reference numbers: all prior "131-level / 118-level / 78,675 px" comparisons were artifacts of the GL debug sample-dump instrumentation corrupting the captured frame. Against a clean GL capture Metal is within 22 levels; the residual divergence is the update-16 TF knife-edge sample flip driven by a ~2e-5 texel march-step drift (update 22)

**Date:** 2026-08-08
**Scope:** (1) Land the anchor-march fix in the Metal camera-inside proxy fragment so far-face remnants stop re-marching the volume (they now clamp `tStart` to the anchor distance and produce a single clamped far-slab sample, matching GL's zero-contribution far fragments). (2) **Methodology correction:** every GL-reference image used in updates 17–21 was captured with the `VTK_GL_SAMPLE_DUMP` instrumentation enabled, whose per-sample re-render passes corrupt the final captured frame. Re-captured cleanly, GL matches the Standard baseline exactly (ImageError 0), Metal is within 22 levels of it everywhere, and the clean cull on/off probe is ≤1 level (restoring update-21's original H2: GL's back/far cells anchor-march from z>1 and contribute nothing). (3) Root-cause the residual divergence: the worst pixels are samples sitting on the TF opacity knife-edge (scalar 1150, opacity 0.02→0.85) that flip because the GL and Metal march-step vectors differ slightly (e.g. y-step ~3.6% off), accumulating ~2e-5 texel by i≈130 — the update-16 mechanism.
**Follows:** [Update 21](VolumeRayCastBackendComparisonFindingsUpdate21.md), which left the far-remnant re-march as the actionable bug. This update implements that fix and retracts update-21 section 4.1's cull-probe conclusion (its 78,675-px / 118-level numbers were contamination).
**Persisted tools:** `BackendComparisonTools/compare_gl_metal_samples.py` (per-sample pos/raw compare), `compare_gl_metal_accum.py` (TF op/rgb + accumulation replay + step fits). Capture env documented in `VolumeRayCastBackendComparisonProcedures.md`.

---

## 1. Conclusion

1. **The anchor-march fix works.** `MetalShaders.metal` `marchVolume` now clamps `tStart` to the interpolated anchor distance for camera-inside proxy fragments (`useCameraInsideNearClip` && !parallel), whenever the anchor lies beyond the recomputed near-plane entry. Cap fragments are unchanged (their anchor coincides with the entry), and the fullscreen path is a no-op (`localPos == entryPoint`). After the fix the far remnant at (422,92) marches **1 sample** (the clamped far slab, `tex.z = 0.999`) instead of re-marching all 170: per-i fragment counts drop from 12/frame (far + cap) to i=0:12, i≥1:6. The far remnant now behaves like GL's (starts beyond the volume, contributes nothing under the cap).
2. **The prior GL-reference numbers were contaminated, not divergences.** The GL captures in updates 17–21 were taken with `VTK_GL_SAMPLE_DUMP` set, which makes `vtkOpenGLGPUVolumeRayCastMapper.cxx:4364` re-render the volume geometry once per sample index (maxSample × 8 channels) into the live framebuffer before the frame capture. Comparison of the **debug-instrumented** GL image against a **clean** GL run: **64,095 px differ, max 115/255**; at the gated far-remnant pixel (422,92) the debug image read `(216,61,76)` while the clean image reads `(238,176,140)`. Update-21's "131-level divergence at (422,92)" and "Metal vs GL 23,327 px (max 131)" were therefore artifacts of comparing Metal against a corrupted GL frame.
3. **Clean GL == Standard baseline.** A clean GL run of the NoJitter camera-inside test reports `ImageError = 0` against the Standard baseline (the debug-instrumented run is the *only* thing that makes GL "fail" against itself at 64k pixels). Per-sample raw/position logs are unaffected by the contamination (the debug channels are re-read per sample before the composite), so all per-sample comparisons in updates 16–21 remain valid.
4. **The clean cull probe reverses update-21's section 4.1.** Re-running the GL cull on/off probe with **no** dump env vars: cull-on vs cull-off differ by **max 1/255, mean 0.1155** (72,015 px at ±1 level) — un-culling GL's back cells changes the image by at most one level, and both variants still pass the Standard baseline with ImageError 0. Update-21's "78,675 px differ, max 118" was contamination. This restores update-21's original H2: GL's back/far cells are rasterized but their anchor-march starts at `z ≈ 1.0 + step` (outside the volume), so they contribute nothing. (Update-21's separate finding that the original byte-identical probe was zero-data remains correct.)
5. **Corrected divergence picture for Metal.** Metal (post-fix) vs clean GL: **max 22/255, 14,368 diff px, mean 0.127**. The far-fragment fix reduces 15,157 → 14,368 diff px (≈5%) and mean 0.1316 → 0.1274. The test's `ImageError = 72,134` is the sum-of-squared-error over these ~14k small-diff pixels.
6. **Residual divergence root cause: TF knife-edge sample flips from a march-step drift.** At the worst pixel (422,92)↔(422,419) the samples match to ≤1.4e-5 position and raws to ~1e-6 up to i=131; at **i=132** the raw flips across the scalar-1150 knot (opacity 0.02→0.85): GL `raw=0.0162814, op=0.127, rgb=(1.0,0.726,0.572)` vs Metal `raw=0.0178680, op=0.4009, rgb=(1.0,1.0,0.9)`. The linear position fits show slightly different step vectors: GL `(−4.53591606e−4, −5.12592482e−6, 1.83725358e−3)` vs Metal `(−4.53499941e−4, −4.94735146e−6, 1.83731304e−3)` (y-step ~3.6% off, others <0.1%), accumulating ~2e-5 texel by i≈130. This is the update-16 mechanism (worst pixel (372,131), sample i=144); the residual diff pixels cluster near the scalar-1150 iso-surface (top diffs: (372,131) 41, (422,92) 35, (421,92) 33).

---

## 2. The fix (`MetalShaders.metal`)

```metal
  float jitter = (volumeUniforms.useJittering > 0.5 ? volume_random(screenPos) : 1.0) * stepSize;
  float tStart = dot(entryPoint - cameraPos, rayDir);
  // OpenGL camera-inside parity (update 22): GL ignores the box/near-plane entry for
  // camera-inside proxy fragments and marches from the interpolated anchor
  // (g_rayOrigin = ip_textureCoords + one step), so far-face fragments start at z>1 and
  // only composite the clamped far slab instead of re-marching the volume from the near
  // plane. setupVolumeRay recomputes the near-plane entry for those same fragments; clamp
  // tStart to the anchor distance whenever the anchor lies beyond the computed entry.
  if (volumeUniforms.useCameraInsideNearClip > 0.5 &&
      volumeUniforms.useParallelProjection < 0.5)
  {
    float tStartAnchor = dot(localPos - cameraPos, rayDir);
    if (tStartAnchor > tStart)
    {
      tStart = tStartAnchor;
    }
  }
```

- Cap fragments: `localPos` (anchor) sits on the recomputed near-plane entry, so `tStartAnchor ≈ tStart` and the clamp never fires.
- Far remnants: `localPos.z = 1.0` (anchor beyond the entry) → `tStartAnchor > tStart` → `tStart` jumps to the anchor, so the march starts at the far slab and the block-bounds exit stops it after one sample.
- Fullscreen (camera-outside) path: `localPos == entryPoint`, no-op.

Verification at (422,92), 6 frames:
- Pre-fix: 12 SAMPLE rows per i (far + cap both re-march all 170).
- Post-fix: i=0 has 12 (cap + far's single boundary sample), i≥1 has 6 (cap only); max i = 169 (170 samples). The far remnant's only sample reads the clamped far slab (`tex.z = 0.999`, `raw≈0.0124`), then the loop exits.
- Image delta Metal(fixed) vs Metal(orig): 892 px, max 5/255 — small, because the cap (drawn last) dominates and the far remnant's single boundary sample is buried under the cap's near-opaque accumulation.

---

## 3. Contamination evidence (debug GL vs clean GL)

Same binary, same test, same frame; only the env vars differ (`VTK_GL_RAY_DUMP=1 VTK_GL_SAMPLE_DUMP=1 ...` for the debug run):

| capture | max | diff px (>0.5) | mean |
|---|---|---|---|
| debug-GL vs clean-GL | 115 | 64,095 | 0.090 |
| clean-GL vs Standard baseline | 0 (ImageError 0) | — | — |
| clean-GL cull-on vs cull-off | 1 | 72,015 | 0.1155 |
| Metal(fixed) vs clean-GL | 22 | 14,368 | 0.1274 |
| Metal(orig) vs clean-GL | 22 | 15,157 | 0.1316 |

At the gated pixel (422,92): debug-GL `(216,61,76)`, clean-GL `(238,176,140)`, Metal(fixed) `(238,192,159)`.

Why the contamination happens: the sample-dump block (`vtkOpenGLGPUVolumeRayCastMapper.cxx:4364-4424`) clears and re-renders the volume geometry once per (sample × 8 channels) with `in_debugSample`/`in_debugChannel` uniforms, leaving the color/depth buffers in a debug state that the subsequent frame capture partially reflects. The per-sample *values* read back by `glReadPixels` during those passes are still meaningful (that is why the raw/pos logs remain trustworthy), but the **final captured PNG is not**.

**Retraction scope:** update-21 section 4.1's "Metal vs GL 23,327 px (max 131)" and "cull-on vs cull-off 78,675 px (max 118)" are retracted. The original byte-identical-probe artifact finding (zero-data) stands. Per-sample findings (winding, mesh reproduction, fragment census, step/ray geometry) are unaffected.

---

## 4. Residual divergence: the TF knife-edge flip (update-16 mechanism)

`compare_gl_metal_accum.py cullon.log metal_fixed.log 422 92` (GL dump at (422,419), Metal at (422,92)):

- Samples i=0..131: positions ≤1e-5 (fit resid ~5e-7), raws agree to ~1e-6, per-sample op/rgb identical.
- **i=132 (first divergence):** GL `raw 0.0162814 → op 0.127265, rgb (1.0, 0.726, 0.572)`; Metal `raw 0.0178680 → op 0.400904, rgb (1.0, 1.0, 0.9)`. The raw straddles the scalar-1150 knot where the opacity TF jumps 0.02 → 0.85 (`pf->AddPoint(0,0.00); AddPoint(500,0.02); AddPoint(1000,0.02); AddPoint(1150,0.85)`).
- Accumulation replay diverges from there: at i=132 GL accC `(0.652,0.465,0.363)` vs Metal `(0.742,0.566,0.457)`; final GL `(0.924,0.682,0.541)` vs Metal `(0.934,0.752,0.622)` ≈ Metal PNG (422,92) `(238,192,159)`.

Position fits (i=10..169/172):

```
GL  pos0=(0.50542762, 0.50653368, 0.45073802) step=(-4.53591606e-04, -5.12592482e-06, 1.83725358e-03)
MT  pos0=(0.50542724, 0.50653411, 0.45073814) step=(-4.53499941e-04, -4.94735146e-06, 1.83731304e-03)
|step| diff (GL-MT)/MT: x +0.02%  y +3.6%  z -0.003%
```

The y-step differs by ~3.6% (absolute 1.8e-7/step), accumulating to ~2.3e-5 texel of the 512³ volume by i≈130 — enough to push samples across the knife-edge. Entry points agree to ~4e-7, so the drift is in the step vector, not the entry. Same mechanism as update-16's (372,131) pixel (there i=144; the residual's top pixels cluster x=370–422, y=92–173).

---

## 5. Reproduction

```bash
# clean GL reference (== Standard baseline, ImageError 0)
build_macos_metal/bin/vtkRenderingVolumeCxxTests \
  TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
  --vtk-factory-prefer RenderingBackend=OpenGL \
  -D build_macos_metal/ExternalData/Testing -T <tmp> -V <out>.png 2> gl_clean.log

# clean GL cull-off (VTK_GL_NO_CULL_PROBE was compiled in the probe binary only;
# source change was reverted after the probe)
VTK_GL_NO_CULL_PROBE=1 build_macos_metal/bin/vtkRenderingVolumeCxxTests ... (same)

# Metal (fixed), capture image + per-sample log
MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=16777216 MTL_LOG_TO_STDERR=1 \
  build_macos_metal/bin/vtkRenderingVolumeCxxTests \
  TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
  --vtk-factory-prefer RenderingBackend=Metal \
  -D build_macos_metal/ExternalData/Testing -T <tmp> -V <out>.png 2> metal_fixed.log

python3 Rendering/Metal/BackendComparisonTools/compare_gl_metal_accum.py \
  gl_samples.log metal_fixed.log 422 92
```

Artifacts: `/tmp/bc/update21/cullprobe/` — `T_gl_clean/`, `T_nocull_clean/` (clean GL cull on/off), `T_metal_fixed/`, `gl_clean.log`, `nocull_clean.log`, `metal_fixed.log`, `cullon.log` (GL per-sample dump, used only for the per-sample compare).

**Data-dir gotcha (from update 21):** `-D` is the test **data** search directory (`build_macos_metal/ExternalData/Testing`), not an output directory; passing an output dir silently zeroes the resampled volume (`TEST_RESAMPLE range=(0,0)`).

---

## 6. Open items

1. **Eliminate the march-step drift** (this update's residual): align Metal's camera-inside step vector with GL's `g_dirStep` (investigate GL's `g_dirStep` construction in `vtkVolumeShaderComposer.h:418,464` / `computeRayDirection()` at :1716 vs Metal's `setupVolumeRay` step) so no sample crosses the scalar-1150 knife-edge; this should collapse the 14k-diff-pixel / ImageError 72,134 residual toward the pass threshold.
2. **Re-baseline the debug-tooling workflow:** treat any GL image captured with `VTK_GL_SAMPLE_DUMP`/`VTK_GL_RAY_DUMP` as unusable for pixel comparisons; use clean captures for image diffs and the debug passes only for per-sample values.
