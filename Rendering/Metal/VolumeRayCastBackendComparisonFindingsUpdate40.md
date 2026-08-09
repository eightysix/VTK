# Root cause of the flat render: `in_inversePVM` is CPU-composed with the matrix product in the WRONG ORDER in BOTH backends — `inv(V)·inv(P)` instead of `inv(P)·inv(V)·inv(M)` (update 40)

**Date:** 2026-08-09
**Scope:** The analytic camera-inside ray (`computeRayDirection` in `vtkVolumeShaderComposer.h`, ported to Metal's `reconstructRayDir` camera-inside proxy path) unprojects the fragment with a CPU-composed `in_inversePVM` uniform. Both backends built that uniform as `inverseVolumeMatrix * inverseModelViewMatrix * inverseProjectionMatrix` and were therefore **byte-identical to each other — but both wrong** (`vtkMatrix4x4` stores row-major, GLSL/Metal read the uniform column-major, so the shader effectively applied the transpose; the product order must be reversed). This single bug was the flat/noise-free render blocking bit-parity. Fixed in both backends; per-pixel ray samples now match GL vs Metal to ≤1e-6 position / 0 raw, and the full 512² images differ in only **694/262144 pixels (0.26%)**, 13 with any channel ≥2 (worst 131 at the known (422,92) frame-ordering artifact).

**Follows:** [Update 39](VolumeRayCastBackendComparisonFindingsUpdate39.md).

---

## 1. The bug (verified by CPU float32 simulation before touching code)

Dumped clip-chain matrices for the last-frame camera `(102.122314, 102.122314, 61.5619835)` (GL_CLIPMAT / MTL_INVPVM, byte-identical across backends):

- `P` (projection), `V` (modelview), `M` (volume/identity in this NoTransform test), `I` = composed `inversePVM` bytes.
- The composed `I` equals `inv(V)·inv(P)` (max |I − inv(V)·inv(P)| = 1.1e-5, M = identity), i.e. the row-major `vtkMatrix4x4` product `InverseVolumeMat · InverseModelViewMat · InverseProjectionMat`.

Simulating the exact GLSL column-major mat4-vec4 multiply with those float32 bytes and unprojecting pixel (422,92) at z = −1/+1:

| uploaded bytes | ray misses eye by | direction |
|---|---|---|
| current `I` (as-is) | **6.5e+01** | (−0.703, −0.711, 0.023) |
| `inv(P)·inv(V)` (fixed) | **7.2e-06** | (−0.338, −0.004, 0.941) |
| `inv(V)·inv(P)` (current, sanity) | 6.5e+01 | (−0.703, −0.711, 0.023) |

The fixed direction matches the update-39 interpolated GL ray `(−0.339, −0.004, 0.941)` — the analytic ray now lands on the reference ray. `I == inv(V)·inv(P)` confirmed in numpy (`np.allclose(..., atol=1e-4)` True), `I == inv(P)·inv(V)` False.

**Why:** the shader receives the uniform bytes as a column-major `mat4`, i.e. `m[col][row] = bytes[col*4+row]`. `vtkMatrix4x4` stores row-major, so the *effective* matrix is the transpose of the stored one; composing `inv(V)·inv(P)` in row-major storage yields the effective `(inv(P))ᵀ·(inv(V))ᵀ`, which is `transpose(inv(P)·inv(V))` — not the unprojection needed. The correct stored row-major product is `inv(P)·inv(V)·inv(M)` (projection inverse applied first).

## 2. The fix

- **GL:** `vtkOpenGLGPUVolumeRayCastMapper.cxx` `SetCameraShaderParameters` (~line 4106) — swap the multiply order to `Multiply4x4(InverseProjectionMat, InverseModelViewMat, tmp)` then `Multiply4x4(tmp, InverseVolumeMat, InversePVMMat)`.
- **Metal:** `vtkMetalGPUVolumeRayCastMapper.mm` (~line 7424) — same swap: `Multiply4x4(invProj, invMV, temp)` then `Multiply4x4(temp, invVol, composed)`.
- Both now upload byte-identical fixed matrices (GL `I=` == Metal `MTL_INVPVM`, verified in the fresh runs; both differ from the old wrong bytes in the off-diagonal Z columns/rows).

## 3. Measurement (fresh captures, same test, exit 1 = expected baseline mismatch, PNGs valid)

Per-pixel sample comparison at the knife-edge pixel (422,92)/(422,419) via `compare_gl_metal_samples.py`:

| metric | value |
|---|---|
| max position |GL − Metal eval| | 0.000001 (was ~1e-5 drift in update 39) |
| max |GL raw − Metal raw| | 0.000000 (frame 0, all i 0..169) |
| GL i range / Metal i range | 0..174 / 0..169 |

Full 512² image diff (GL vs Metal, last frame):

| metric | value |
|---|---|
| pixels differing | 694 / 262144 (0.26%) |
| pixels |d| = 1 (any channel) | 496 |
| pixels with a channel ≥ 2 | 13 |
| pixels |d| ≥ 10 | 2 |
| max per-channel |d| | 131 (pixel (422,92)) |
| combined RMS / normalized | 0.1481 / 0.00092 |

The residual is concentrated at (422,92) (the known frame-ordering/camera-animation artifact flagged in the compare tool header at i = 30, 134, 167) plus a handful of ±1 ULP-level pixels. Both new backends sit at exactly the same residual vs the stored `u38c/gl.png` reference (combined RMS 0.1481, max 10.77) — i.e. GL and Metal are mutually consistent, and their shared residual vs the old interpolated reference is the expected analytic-vs-interpolated ray difference (direction now matches to ~1e-5 as shown above).

## 4. Next

- The remaining (422,92) hot pixel and the few ±1 pixels: confirm they are the documented frame-ordering/camera-animation artifacts (frames beyond #0 diverge only in sum/ordering, frame-0 rays are bit-consistent) before chasing further.
- Then re-run the full accumulation test (`compare_gl_metal_accum.py`) and the other NoTransform/NoJitter variants to confirm the fix generalizes beyond the single camera pose.

## Artifacts

- Captures: `/tmp/bc/gl_fix2.png` (GL, last frame), `/tmp/bc/metal_fix2.png` (Metal, last frame), `/tmp/bc/gl_fix2.log` (GL_SAMPLE/STEP), `/tmp/bc/metal_fix2.log` (Metal SAMPLE/STEP), `/tmp/bc/u38c/gl.png` (stored reference).
- GL_CLIPMAT `I=` / MTL_INVPVM after fix: `761287beb0afdd9f421a40bd00000000b97b04bcb422873efd4d3a3d000000009f8784c39f8784c3e8c81fc3d91c26c07ab584437ab5844312132143fb712640`.
