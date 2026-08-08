# Why GL's frame 3→4 response is ~50× larger: the ray anchor sits 11.7× closer to the camera eye in GL than in proxy-Metal (`rayDir = normalize(anchor − eye)` sensitivity = `δ_perp / |anchor − eye|`), verified to <2% on GL and within print noise on Metal (update 20)

**Date:** 2026-08-08
**Scope:** Close Update 19's remaining option 3 — "investigate GL's larger interpolation sensitivity (why GL's per-step delta is ~50× proxy-Metal's)". The cause is a purely geometric **anchor-distance amplification**: both backends reconstruct the per-pixel ray as `rayDir = normalize(anchor − eye)` where `anchor` is the interpolated fragment position; a position shift `δ` becomes a direction change `δ_perp / |anchor − eye|`. GL's anchor sits on the near-plane-clipped proxy surface ~0.0048 (normalized volume units) from the eye; proxy-Metal's anchors sit 0.056 (pass A) and 0.566 (pass B) away. The ~12× distance ratio plus the fact that GL's shift is ~97% perpendicular to the ray (while Metal's is ~95% parallel) reproduces the observed ~40-50× per-step ratio quantitatively.
**Follows:** [Update 19](VolumeRayCastBackendComparisonFindingsUpdate19.md), which established the W2IF view-angle perturbation (30 → 30.0000008°) as the common cause and left the amplitude asymmetry as option 3.
**Persisted tool:** none new — reuses the existing captures `/tmp/bc/update20/metal_proxy.log` (Metal, `VTK_METAL_FULLSCREEN_CAMERA_INSIDE=0`), `/tmp/bc/update19/gl.log` (GL), and the update20 gated pixel `(372,131)`.

---

## 1. Conclusion

1. **The amplitude asymmetry is fully explained by ray-anchor geometry, not by per-backend interpolation accuracy.** Both backends compute `rayDir = normalize(anchor − eye)` in float32, where `anchor` is the fragment's interpolated position (GL `g_rayOrigin` from `ip_textureCoords`; Metal `in.localPos` from the vertex shader, or `s.entryPoint` on the fullscreen path). A tiny anchor shift `δ` therefore changes the ray direction by ≈ `δ_perp / |anchor − eye|`. The measured anchors are:

   | backend | anchor | |anchor − eye| (normalized) | |shift δ| at frame 3→4 |
   |---|---|---|---|
   | GL | ray origin on near-plane clip surface | **0.00477** | 3.69e-7 |
   | Metal pass A | interpolated position | **0.05584** | 2.40e-7 |
   | Metal pass B | far face (z = 1.0) | **0.566** | 4.2e-8 |

2. **The direction of the shift is the second amplifier.** Decomposing `δ` into parallel/perpendicular to the ray:
   - GL: `δ_perp = 3.58e-7` (97% of `δ`) → `ΔrayDir ≈ 3.58e-7 / 0.00477 = 7.5e-5` — matches the observed step-derived value `1.457e-7 / 0.001911 = 7.6e-5` to **1.4%**.
   - Metal pass A: `δ_perp = 7.8e-8` (only 33% of `δ`; 95% of the shift is along the ray, which does not change the direction) → `ΔrayDir ≈ 7.8e-8 / 0.0558 = 1.4e-6` — matches the observed `|ΔrayDir| = 1.9e-6` within print resolution of the y component.
   - Ratio: `(0.05584/0.00477) × (3.58e-7/7.8e-8) = 11.7× × 4.6× = 54×` — the observed ~40-50× (GL `1.457e-7` vs Metal `3.7e-9` per-step).
3. **Why the anchors differ.** GL's camera-inside path renders the near-plane-clipped proxy mesh, so the fragment that survives at the gated pixel is on the near-plane cap, essentially touching the eye (0.005). Metal's camera-inside default path is the fullscreen triangle (vtkMetalGPUVolumeRayCastMapper.mm:7486-7491 — no proxy geometry), and even the forced-proxy capture (`VTK_METAL_FULLSCREEN_CAMERA_INSIDE=0`) yields anchors 0.056/0.566 from the eye. The proxy MESH construction is otherwise identical (same `ClipConvexPolyData` near-plane + `SetNumberOfSubdivisions(2)` + triangle filter; the earlier "GL 5 vs Metal 2 subdivisions" hypothesis was a misreading and is retracted).
4. **Correction to Update 19 section 6.** Metal's interpolated position shift is NOT ~10× smaller than GL's — `|δ|` is comparable (3.7e-7 vs 2.4e-7). The asymmetry is the **anchor distance** (11.7×) times the **perpendicular fraction** (4.6×). Metal's dominant shift component is along the ray (z), which rotates the direction almost not at all; GL's is sideways (x/y), which rotates it fully.
5. **Update 19's option 3 is closed.** The i=144 residual is a harness-injected 8.3e-7° view-angle perturbation, amplified by each backend's anchor geometry into a ray-direction change; GL's is large enough (7.6e-5) to drift sample i=144 across the scalar-1150 opacity knot over 144 steps (~1.4e-5), proxy-Metal's is not (~3e-7). This is a **numerical-robustness property of GL's near-eye proxy-mesh ray origin**, not a correctness or nondeterminism difference.

---

## 2. The verified arithmetic (all from the logs)

### 2.1 Anchor distances

```
eye (cameraVol, shared)        = (0.506559074, 0.506559074, 0.446101338)

GL origin pre  = (0.505591691, 0.5064044, 0.450773209)   |anchor-eye| = 0.00477
GL origin post = (0.505591452, 0.506404161, 0.45077306)  delta = (-2.39e-7, -2.39e-7, -1.49e-7)

MT-A localPos pre  = (0.4951383173, 0.5048617125, 0.5007355213)  |anchor-eye| = 0.05584
MT-A localPos post = (0.4951383471, 0.5048617125, 0.5007357597)  delta = (+2.98e-8, 0, +2.38e-7)

MT-B localPos pre  = (0.3907720745, 0.4893507063, 1.0)   |anchor-eye| = 0.566
MT-B localPos post = (0.3907721043, 0.4893507361, 1.0)   delta = (+2.98e-8, +2.98e-8, 0)
```

Consistency check: `normalize(localPos − eye)` reproduces the STEP `rayDir` exactly to printed precision in both Metal passes, confirming `rayDir = normalize(anchor − eye)`.

### 2.2 Predicted vs observed ray-direction change

```
                     delta_perp   /  |anchor-eye|   = predicted   observed
GL   (origin)        3.58e-7      /  0.00477        = 7.51e-5    7.62e-5   (|dstep|/stepSize, 1.4% match)
MT-A (localPos)      7.83e-8      /  0.05584        = 1.40e-6    1.86e-6   (|drayDir|, y-delta below 9-decimal print)
MT-B (localPos)      ~3.0e-8      /  0.566          = ~5.3e-8    7.9e-8    (within 1.5x, noise-dominated)
```

The GL side is an independent confirmation: two measurements of `ΔrayDir` agree to 1.4% — `|δ_perp|/r` from the interpolated origin, and `|Δstep|/stepSize` from the march step. Metal's smaller shift is consistent within the print resolution of its y component.

### 2.3 Why the shift directions differ

- GL's anchor is on the near-plane cap, a surface nearly **perpendicular to the ray**; interpolation error there moves the anchor **sideways** (97% of `δ` perpendicular) → full directional effect.
- Metal's pass-A shift is dominated by z (2.38e-7 of 2.40e-7), and the ray is 0.978 along z → 95% of `δ` is **parallel** → only 7.8e-8 remains to rotate the direction.

### 2.4 Per-step accumulation (reproduces the observed flip)

| quantity | GL | Metal proxy |
|---|---|---|
| per-step ray change | 7.6e-5 (rayDir) / 1.457e-7 (step) | 1.9e-6 (rayDir) / 3.7e-9 (evalStep) |
| accumulated drift at i=144 | ~1.4e-5 → crosses scalar-1150 knot → `raw` 0.0154269 → 0.0181582 | ~3e-7 → stays below knot → `raw` 0.015427 constant |

---

## 3. Structural root of the anchor positions

- **GL camera-inside** (vtkOpenGLGPUVolumeRayCastMapper.cxx:1160+): proxy box clipped against the offset near plane, densified, rendered; the nearest surface at the gated pixel is the near-plane cap, so the interpolated origin sits ~0.005 from the eye. GL's ray origin is exactly where the near-eye robustness problem lives.
- **Metal camera-inside default** (vtkMetalGPUVolumeRayCastMapper.mm:7486-7491): fullscreen triangle, no proxy mesh; `anchor` = interpolated fullscreen position / analytic entry, ~0.06-0.57 from the eye.
- **Metal forced-proxy** (`VTK_METAL_FULLSCREEN_CAMERA_INSIDE=0`): the clipped mesh is built, but the measured pass-A anchor (0.495, 0.505, 0.501) is not on any clipped-box surface plane (verified: `n·(v − planeOrigin) = 7.4` world units off the cap; no coordinate at 0/1), and pass-B is the far face. **Open sub-detail:** the exact draw/triangle that produces the interior pass-A anchor in the proxy capture is not resolved from static reading (two fragment invocations per frame at the pixel; the winning fragment is not the depth-nearest cap fragment). This does not affect the residual explanation — both Metal anchors are 12-118× farther from the eye than GL's, which is all the sensitivity argument needs.

The proxy-mesh construction is otherwise identical between the backends (same `ClipConvexPolyData` near plane with the same offset formula `(far−near)*0.001`, same `SetNumberOfSubdivisions(2)`, Metal adds a redundant `vtkTriangleFilter`). The earlier "GL densifies 5 vs Metal 2" claim (from a misread log) is retracted.

---

## 4. Net

The i=144 residual and its ~50× amplitude asymmetry are now fully accounted for:

1. W2IF perturbs the view angle 30 → 30.0000008° (Update 19), changing the projection by ~3e-8 relative.
2. Each backend's interpolated ray anchor shifts by a comparable amount (`|δ| ≈ 2.4-3.7e-7`).
3. `rayDir = normalize(anchor − eye)` turns that into `ΔrayDir ≈ δ_perp/|anchor − eye|`; GL's anchor is 11.7× closer to the eye and its shift is ~4.6× more perpendicular → ~54× larger `ΔrayDir` (observed ~40-50×).
4. Over 144 steps that direction error drifts GL's sample ~1.4e-5, crossing the scalar-1150 opacity knot; proxy-Metal drifts ~3e-7 and stays put.

This is a **GL-side numerical-robustness characteristic of starting the ray on a surface essentially touching the camera**, exposed only by the harness's perturbation at the W2IF comparison frames. It is not a Metal-vs-GL correctness difference and not nondeterminism.

---

## 5. Reproduction

Same captures as Update 19 (no source changes this update; all numbers read from `/tmp/bc/update20/metal_proxy.log` and `/tmp/bc/update19/gl.log`):

```sh
# Metal pass-A STEP (pre/post boundary):
rg 'STEP px=\(372, 131\)' /tmp/bc/update20/metal_proxy.log | sed 's/.*localPos=/localPos=/' | sort -u

# GL GL_RAY origin/step (pre/post boundary):
rg 'GL_RAY px=\(372, 380\)' /tmp/bc/update19/gl.log | sed 's/.*origin=/origin=/' | sed 's/ ip=.*//' | sort -u

# i=144 constancy:
rg 'SAMPLE px=\(372, 131\) i=144 ' /tmp/bc/update20/metal_proxy.log | sed 's/.*raw=/raw=/' | sort -u
rg 'GL_SAMPLE px=\(372, 380\) i=144 ' /tmp/bc/update19/gl.log
```

---

## 6. Status and remaining open items

- Updates 16-18: residual pinned to one flipped TF sample i=144; drift root-caused to a ray-origin/setup tilt; proxy path matched GL's pre-flip frames.
- Update 19: the tilt source = W2IF float32 view-angle round-trip perturbing the camera at frame 3→4 in both backends.
- **Update 20 (this):** the ~50× amplitude asymmetry = ray-anchor geometry (`δ_perp/|anchor − eye|`; GL anchor 0.0048 vs Metal 0.056/0.566 from the eye, GL shift 97% perpendicular vs Metal 95% parallel), verified to 1.4% on GL. Option 3 of Update 19 is closed.

Open items:

1. **Pass A/B identity in the forced-proxy capture** — the interior pass-A anchor is not on any clipped-box surface; identify which draw/fragment produces it (this is a logging curiosity; it does not affect the residual explanation).
2. **Optionally harden GL** — because the sensitivity is `∝ 1/|anchor − eye|`, starting GL's camera-inside ray from the analytic near-plane entry (as Metal's fullscreen path does) instead of the interpolated mesh origin would remove GL's ~12× disadvantage. Not required for the Metal-vs-GL comparison.
3. **The two untracked PNG artifacts** (`NoJitter_delta_heatmap.png`, `NoJitter_delta_mask.png`) in the working tree predate this update and can be committed or removed separately.
