# MaxIP 6-px residual is a knife-edge argmax pick-flip (not accumulation, not a rounding artifact): Metal's camera-A render at (501,480) bit-matches GL's W2IF frame, and the (370,266)/(428,270) candidate values swap between backends (update 81)

**Date:** 2026-08-11
**Status:** **MaxIP experiment closed.** The 6-pixel MaxIP (blend mode 1)
Metal↔GL residual is **not** a color/opacity-accumulation bug and **not** an
8-bit rounding artifact: both backends are individually self-consistent
(Metal PNG = Metal MIPFINAL logs; GL PNG = GL RGBA32F dump), but at the W2IF
frame the exact floats **differ** at all 6 pixels because each backend's ray
picks a different near-tied max texel. All 6 pixels are knife-edge
(`mip−mip2` in [0.00046, 0.00206]); at (501,480) Metal's own **camera-A
render is bit-identical to GL's W2IF-frame value**, then a later render flips
to a different (higher) argmax; at (370,266)/(428,270) the two backends
**swap** their two candidate values. This is the same sub-texel pick-flip
driven by the interpolator-floor ray difference as the reference-test diffs —
the MaxIP residual therefore lives in the scalar-fetch / ray domain, exactly
what the test was designed to discriminate (see test file header).

Secondary correction: the frame-alignment "mystery" from the MaxIP session was
a **y-orientation error**, not a missing frame. The Metal PNG row == Metal
`screenPos` row (**unflipped** readback), while the GL PNG row == GL window
row `511−y`. The run-5 MIPFINAL gates were therefore at the wrong rows
(`511−png_y`); the "all renders saturated yet the PNG is mid-ramp"
contradiction disappears once the gates are placed at the PNG rows.

## 1. Method / captures

- Metal: `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformMaxIP`
  with `--vtk-factory-prefer RenderingBackend=Metal`, `MTL_LOG_*` env for
  shader os_log. Gate pixels (`pxOkMaxIP` in `debugMarchGate`,
  MetalShaders.metal:3921) set to the **PNG rows** of the 6 diff pixels.
  Output: `/tmp/bc/maxip/metal_run6.log` (608 `MIPFINAL`, 47 542 `MARCH`).
- Metal PNG (same run, `/tmp/bc/maxip_mtl_tmp/dummy.png`) is **bit-identical
  to the suite `mt_MaxIP.png`** (max_delta = 0, 0 diff px), so gating does not
  perturb output and the capture is reproducible.
- GL: `VTK_GL_FLOAT_DUMP` → `/tmp/bc/gl_float_dump.raw` (RGBA32F, 512×512,
  row 0 = gl_FragCoord y 0), verified against the clean suite `gl_MaxIP.png`
  (≤1 LSB at the 6 pixels).
- Pairing (consistent with update 76 §4): same world content at
  `Metal screenPos (x, y) ↔ GL window (x, 511 − y) ↔ PNG (x, y)` for GL, and
  `Metal screenPos (x, y) ↔ PNG (x, y)` directly for Metal. Run-5 gated Metal
  pixels (68,282), (24,119), (370,245), (428,241), (449,137), (501,31) were
  the `511−png_y` rows and are **not** the PNG pixels.

## 2. Y-orientation correction (why run 5 "lost" the PNG frame)

- Metal FINAL/MIPFINAL logs at `(68,229)`, `(24,392)`, `(370,266)`,
  `(428,270)`, `(449,374)`, `(501,480)` (screenPos) reproduce the Metal PNG at
  those exact rows after `round((out + 0.1*(1−out.r))*255)`.
- GL dump rows `(68,282)`, `(24,119)`, `(370,245)`, `(428,241)`, `(449,137)`,
  `(501,31)` reproduce the GL PNG at rows 229/392/266/270/374/480.
- The two PNGs are orientation-aligned (direct sum of |Δ| = 109 vs
  `flipud` sum = 15,043,079), so the W2IF readback flips GL and does **not**
  flip Metal — both PNGs land in the same (top-origin) orientation. Metal
  `screenPos.y` is mirrored vs GL `gl_FragCoord.y` for the same content.

## 3. Bit-exact W2IF-frame floats at the 6 diff pixels

`out = (R,G,B)` = pre-background-blend fragment value (`c.rgb*c.a, c.a`,
a = out.r since c.r = 1). 8-bit = `round((out + 0.1*(1−a))*255)`.

| PNG px | Metal mip (camera B) | Metal out | Metal 8-bit (= MT PNG) | GL out (GL row 511−y) | GL 8-bit (= GL PNG) |
|---|---|---|---|---|---|
| (68,229) | 0.259042859 | (0.750476, 0.705481, 0.621435) | (198, 186, 165) ✓ | (0.726839, 0.672912, 0.589443) | (192, 179, 157) ✓ |
| (24,392) | 0.252177775 | (0.585017, 0.491631, 0.414453) | (160, 136, 116) ✓ | (0.561380, 0.463774, 0.388115) | (154, 129, 110) ✓ |
| (370,266) | 0.264306098 | (0.850000, 0.850000, 0.765000) | (221, 221, 199) ✓ | (0.845024, 0.842491, 0.757482) | (219, 219, 197) ✓ |
| (428,270) | 0.262704253 | (0.845024, 0.842491, 0.757482) | (219, 219, 197) → PNG 220 (±1) | (0.850000, 0.850000, 0.765000) | (221, 221, 199) ✓ |
| (449,374) | 0.257212192 | (0.703202, 0.641016, 0.558258) | (187, 171, 150) ✓ | (0.655928, 0.579242, 0.498313) | (176, 156, 136) ✓ |
| (501,480) | 0.260873556 | (0.797750, 0.772640, 0.687843) | (209, 202, 181) ✓ | (0.774113, 0.738724, 0.654235) | (203, 194, 173) ✓ |

Not bit-identical at the W2IF frame. GL ±1 LSB vs PNG at (24,392) is the known
`DumpCleanGLFloats` re-render knife-edge caveat (update 58); all 8-bit cols
match their own PNG to ≤1 LSB, i.e. **each backend is self-consistent**.

## 4. The differences are argmax pick-flips at knife-edge rays

- **Near-tied maxima everywhere.** `mip2` (runner-up) sits within
  [0.00046, 0.00206] of the max at every pixel: (68,229) 0.00046, (24,392)
  0.00046, (370,266) 0.00160, (428,270) 0.00114, (449,374) 0.00206,
  (501,480) 0.00092.
- **(501,480): Metal flips within its own run, and its camera-A value
  bit-matches GL.** Metal camera-A renders log `mip = 0.259958208` with
  `mip2 = 0.259958208` — a **true tie** — and `out = (0.774113, 0.738724,
  0.654235)`, **bit-identical to GL's W2IF-frame value**. A later render
  (W2IF camera) flips to `mip = 0.260873556`, `out = (0.797750, 0.772640,
  0.687843)` = the Metal PNG. So at the *same* W2IF camera Metal and GL pick
  different texels, while Metal's camera-A pick equals GL's exactly.
- **(370,266)/(428,270): candidate values swap.** Metal(370,266) = 0.85 =
  GL(428,270), and Metal(428,270) = 0.845024 = GL(370,266). Same two
  candidates, opposite picks — the fingerprint of a knife-edge boundary, not
  a wrong-value computation.
- **The flip sensitivity is smaller than the backend ray difference.** Metal
  MARCH shows the W2IF view-angle perturbation (30° → 30.0000008°) moves the
  ray by only ~2–5e-5 in a normalized dir component, e.g. (501,480)
  (−0.295154, −0.284952, 0.911969) → (−0.295112, −0.284932, 0.911988), yet
  that alone flips Metal's argmax. The Metal↔GL ray difference from the
  interpolator floor (updates 1–2, 16–19, 76; anchors within 0–4 ulp,
  dirObj 114/114 bit-identical) is of the same order and flips it at the same
  camera.

## 5. Conclusion

The MaxIP 6-px residual is the **same interpolator-floor / argmax pick-flip
mechanism** as the reference-test diffs, in the **scalar-fetch / ray domain**:
each backend marches a slightly different ray, and at these 6 knife-edge
pixels the near-tied MIP max flips to a different texel. Removing opacity-
weighted compositing did **not** remove the residual (5/6 pixels differ by
1–16 in 8-bit), so the residual cannot live in the color/opacity accumulation
path; and both backends reproduce their own outputs bit-exactly, so no backend
computes a wrong value. Consistent with the test's design question, the MaxIP
residual pins the divergence to the ray/scalar-fetch domain.

No fix is implied for the residual itself; the result is diagnostic (matches
the reference-test picture). The TEMP DEBUG instrumentation touched this
session: `pxOkMaxIP` gate pixels corrected to the PNG rows
(MetalShaders.metal:3921), duplicate gate definition removed, `|| pxOkMaxIP`
de-duplicated in `debugMarchGate`'s return (MetalShaders.metal:3949).
