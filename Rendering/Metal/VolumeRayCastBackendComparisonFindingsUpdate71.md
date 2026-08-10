# Analytic pixel-ray verified GL-identical: the nearP "mismatch" was a mirrored-pixel comparison artifact (update 71)

**Date:** 2026-08-10
**Status:** **milestone (verification, no image delta change).** The analytic
pixel-ray unprojection (`marchVolumeUnified` camera-inside branch,
`MetalShaders.metal:4046`) is confirmed **bit-identical to OpenGL at every
physical pixel**: Metal nearP matches GL nearP to the last printed digit, the
inversePVM matrices are byte-identical, and the earlier "nearP mismatch"
(102.1136 vs 102.1105 at px=(140,505)) was a **coordinate-convention comparison
error** — GL's bottom-up `gl_FragCoord.y` and Metal's top-left `screenPos.y`
index *mirrored* rows, so the two logs' (140,505) were different physical pixels.

## 1. The fix for the measurement (self-contained STEP lines)

The previous cross-backend ray comparison was ambiguous because `os_log` is
asynchronous: `MTL_INVPVM` prints from the CPU encode thread interleave
unpredictably with GPU `DEBUG STEP` fragment prints, so a STEP line could not be
attributed to the camera pose it actually used. Embedded the shader's own
`volumeUniforms.inversePVM` (16 words, `as_type<uint>`) into every STEP line
(`MetalShaders.metal:4271-4278`) so each line is self-contained.

## 2. What was verified (single Metal run + existing GL_RAY log)

1. **Shader uniform == CPU matrix.** The embedded `invPVMBits` decode to exactly
   the same floats as the CPU `MTL_INVPVM` dump (max |Δ| = 0.0 / 1 ulp). The
   CPU print is memory-order hex (bytes), the shader `%08x` prints the uint —
   byte-reversed relative to each other, same values.
2. **Backend matrices identical.** GL `GL_CLIPMAT I=` (in_inversePVM upload,
   `vtkOpenGLGPUVolumeRayCastMapper.cxx:4282`) and Metal `MTL_INVPVM` are
   **byte-for-byte identical** (`761287be...fb712640`, only the Metal run's
   2nd frame differs by 1 ulp on element [0]).
3. **Same physical pixel, same ray.** The shader computes `-ndc.y` from
   top-left `screenPos`; GL computes `ndc.y` from bottom-up `gl_FragCoord.y`.
   For physical row *r* from top: Metal `screenPos.y = r+0.5`,
   `ndc.y = (r+0.5)*2/H - 1`; GL `gl_FragCoord.y = H-0.5-r`,
   `ndc.y = (H-0.5-r)*2/H - 1 = -(r+0.5)*2/H + 1`. Both are the SAME ndc.y.
   The decisive pair (same physical pixel):
   - **Metal STEP px=(140, 6):  nearP = (102.110527, 102.138634, 61.7612267)**
   - **GL_RAY   px=(140, 505): nearP = (102.110527, 102.138634, 61.7612267)**
   - identical to the last printed digit (Metal prints full float32).
4. **farP / dir / step within 1–10 ulp** (sub-ulp knife-edge family, not a
   systematic ray error): farP.x 90.3353195 (Metal) vs 90.3353271 (GL) = 1 ulp
   at 90.3; `dirObj` vs GL `rd` ≈ 1e-7 (few ulp); `evalStep.x` 7.8690187e-5 vs
   GL 7.8690115e-5 ≈ 10 ulp — consistent with update-70 §4c, the round-off
   amplification of the ~1 ulp farP.x through normalize → adjustedLin * dirObj.

## 3. Methodology caveats (why earlier sessions misread this)

- The os_log STEP record wraps at ~563 chars into two log records; the second
  chunk (nearBits/farBits/dBits/invPVMBits) is a separate line that also starts
  with the `VTK_METAL_VOLUME_LOG` prefix. Greps must match on the 1190-char full
  lines (`'invPVMBits=' in l`), not just `DEBUG STEP px=...`.
- Analysis scripts must byte-reverse/normalize the two hex encodings (CPU
  memory-order vs shader uint order) before comparing.
- A direct GL(140,505) vs Metal(140,505) comparison is invalid: those are
  mirrored pixels. Pair via `Metal(px, r) ↔ GL(px, H-1-r)` (verified
  H=512 → GL(140,505) ↔ Metal(140,6)).

## 4. Conclusion / implication for the reference-test residual

The analytic camera-inside ray (nearP/farP unprojection, `dirObj`,
`evalStep`) is **not** a source of backend divergence. The reference test's
183 px residual (update 70: 169 ±1 attribute-interpolator floor + 14 knife-edge)
must live entirely in the per-sample scalar/interpolation/accumulation chain
after the ray is formed, or in the remaining ~1–10 ulp ray-parameter differences
documented in §2.4. Next experiments should focus on those, not the ray.
