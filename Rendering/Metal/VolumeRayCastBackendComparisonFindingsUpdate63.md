# Window-position f32 rounding is REFUTED as the varying-path bias; the interpolated-texcoord residual is localized to a rasterizer-level sample displacement that differs per backend (~+0.03,-0.03 px) and is reproducible by no analytic pixel-center model (update 63)

**Date:** 2026-08-09
**Scope:** Update 62 left three open hypotheses: (1) the varying (texcoord) path bias = f32 rounding of the enormous vertex window positions, (2) a finer window-rounding model would close the 5 pixels that only match 2/3 channels, (3) the GL-vs-Metal ~1-3 ulp delta is a driver-level floor. This update tests all three from the existing logs.
**Target (unchanged):** Metal bit-identical to **clean GL** (`RenderingBackend=OpenGL`).

**Follows:** [Update 62](VolumeRayCastBackendComparisonFindingsUpdate62.md) (plane≡barycentric, varying vs clip paths decoupled), [Update 61](VolumeRayCastBackendComparisonFindingsUpdate61.md) (weights bit-identical, bias systematic), [Update 60](VolumeRayCastBackendComparisonFindingsUpdate60.md) (per-vertex clip+tex bit-identical), [Update 59](VolumeRayCastBackendComparisonFindingsUpdate59.md) (188 px knife-edge flips).

---

## 1. Data

Same frame-6 logs as updates 61-62 (`/tmp/bc/u62_gl_vlog.log`, `/tmp/bc/u62_metal.log`; triangle 122 = (86,40,93), 14 knife-edge px). New scripts (update-63 tooling, `/tmp/bc`):

- `window_u63.py` — sweeps 5 window-coordinate rounding variants for the 3 vertices (exact f64 `(ndc+1)*256`, sequential f32 mul-add, sequential f32 add-mul, f64 product then f32, half-framebuffer f32 `ndc*256+256`) and reconstructs the **implied sample-NDC offset** `Σλᵢ·refᵢ` from the rounded-window barycentric at each pixel center.
- `refine_u63.py` — fine per-pixel scan (step 1e-6, window ±6e-4) mapping the full span of (dx,dy) that reproduce GL's texcoord on 3/3 channels, plus the pixel-phase (mx%2, my%2).
- `backout_u63.py` — least-squares back-out of the effective perspective weights `λ = normalize(tᵢ·wᵢ)` from the 3 logged texcoord channels (per backend), with the fit residual.
- `backout_both_u63.py`, `displace_u63.py` — per-pixel weight back-out for BOTH backends and conversion of each weight set to an **effective sample displacement** from the pixel center (NDC and px).
- `texcompare_u63.py` — direct per-pixel comparison of GL's interpolated `tex=` vs Metal's interpolated `localPos=` (verified to be the interpolated texcoord, `anchorTex = in.texcoord`).
- `model_u63.py` — analytic pixel-center interpolation (affine / persp `1/w` / persp `rcp(1/w)` / inverse-w) vs logged texcoord, 3/3 zero-ulp counts.

## 2. Result 1 (REFUTED): window-position f32 rounding is NOT the varying-path bias

The five rounding variants produce **byte-identical window coordinates** at 6 sig figs and **identical implied sample-NDC offsets** at every pixel (e.g. (397,110): `(−1.09e-7, −4.82e-6)`; (349,255): `(+4.03e-7, +2.03e-6)`; (482,33): `(−3.78e-7, −1.44e-6)`):

| px | implied offset (dx,dy) NDC — all 5 variants identical | texcoord ulps vs GL |
|---|---|---|
| (397,110) | (−1.09e-7, −4.82e-6) | +5/+5 |
| (349,255) | (+4.03e-7, +2.03e-6) | +1/+1 |
| (482, 33) | (−3.78e-7, −1.44e-6) | +4/+4 |

The f32 rounding of the huge window coordinates (f32 ulp ≈ 0.015 px at ±2.5e5) shifts the barycentric weights by only ~1e-7 NDC — **~2000× smaller** than the ~(+2.5e-4, −3.5e-4) NDC offset update 62 needed. So the varying-path bias is a **real sub-pixel effective-sample displacement**, not a window-coordinate rounding artifact. This refutes update-62 doubt #1 and makes doubt #2 moot (no finer rounding model can produce 1e-4-scale effects).

## 3. Result 2: no single sample offset explains all pixels — the "offset" is a per-pixel stand-in, not a fixed displacement

`refine_u63.py` (step 1e-6, window ±6e-4) maps the full region of (dx,dy) giving 3/3 zero-ulp matches per pixel:

| px | 3/3 dx span | 3/3 dy span | #3/3 pts | phase |
|---|---|---|---|---|
| (349,255) | [+1.00e-6 .. +1.19e-4] | [−1.02e-4 .. +3.40e-5] | 10174 | (1,1) |
| (405,171) | [+3.62e-4 .. +3.98e-4] | [−3.62e-4 .. −3.09e-4] | 776 | (1,1) |
| (338,432) | [+3.06e-4 .. +3.49e-4] | [−3.39e-4 .. −2.87e-4] | 919 | (0,0) |
| (350,  5) | [+3.41e-4 .. +4.59e-4] | [−4.38e-4 .. −3.02e-4] | 9044 | (0,1) |
| (153, 32) | [+2.70e-4 .. +3.44e-4] | [−3.77e-4 .. −2.94e-4] | 2870 | (1,0) |
| (482, 33) | [+3.15e-4 .. +4.35e-4] | [−5.07e-4 .. −3.78e-4] | 12430 | (0,1) |
| (439,281) | [+9.20e-5 .. +2.11e-4] | [−1.56e-4 .. −2.70e-5] | 8357 | (1,1) |
| 5 others | **no 3/3 point in ±6e-4** | | 0 | — |

- The winning regions are **broad** (the f32-quantized texcoord is insensitive to the offset within ~1e-4) and **do not intersect** across pixels: (349,255) needs dx ≤ +1.19e-4 while (482,33) needs dx ≥ +3.15e-4. **No fixed sample displacement reproduces GL's texcoord at all pixels.** The cluster of update 62 was the union of these per-pixel regions, not a single offset.
- Phase (mx%2, my%2) does not predict the offset magnitude (both the smallest and largest regions are phase (1,1)). No parity/quad pattern.

## 4. Result 3: per-pixel effective weights are internally consistent (one weight set per pixel) but displaced identically in both backends

`backout_u63.py` fits a single weight set to the 3 texcoord channels; residuals are ~1e-15 (f64) for every pixel — the 3 channels are perfectly consistent with ONE weight set, on both backends. Those weight sets differ from the analytic pixel-center f32-NDC weights by ~1e-6 relative (l2) / ~1e-7 absolute (l0).

`displace_u63.py` converts each backed-out weight set into an effective sample position `Σλᵢ·ndcᵢ`:

| px | disp GL (dx,dy) NDC | disp Metal (dx,dy) NDC | GL−Metal (px) |
|---|---|---|---|
| (397,110) | (+3.54e-4, −3.94e-4) | (+2.19e-4, −2.62e-4) | (+0.035, −0.034) |
| (349,255) | (+5.68e-5, −2.87e-5) | (+3.17e-4, −1.66e-4) | (−0.067, +0.035) |
| (482, 33) | (+3.77e-4, −4.41e-4) | (+2.55e-4, −3.23e-4) | (+0.031, −0.030) |
| (439,281) | (+1.56e-4, −9.53e-5) | (+1.60e-4, +1.63e-5) | (−0.001, −0.029) |
| **mean over 14** | **(+2.91e-4, −2.89e-4)** | **(+1.81e-4, −1.74e-4)** | **(+0.028, −0.030)** |

- **Both** backends' varying interpolators sample effectively toward the same off-center direction (+x, −y NDC); **GL is systematically further** by ~(+0.03, −0.03) px at most pixels, with a few 2× outliers (349,255, 338,432, 470,269, 293,298, 439,281). The outliers are within the f32-texcoord-quantization noise of the back-out (~1.6e-4 NDC), so the underlying bias may be near-constant; but the refine result (Result 2) says no single offset is exact at all pixels.

## 5. Result 4 (NEW): the residual GL-vs-Metal delta is localized to the interpolated texcoord — it differs by −3..+3 ulps per pixel

`texcompare_u63.py`: GL's interpolated `tex=` (debug channels 9-11 = `ip_textureCoords`) vs Metal's `localPos=` (DEBUG STEP; verified = `anchorTex = in.texcoord`, the interpolated per-vertex texcoord):

| px | ulp diff (GL − Metal) | px | ulp diff |
|---|---|---|---|
| (397,110) | (−1, −1, −2) | (120,167) | (−1, −1, −1) |
| (360,229) | (−1, −1, −3) | (470,269) | (−2, −1, −2) |
| (349,255) | **(+2, +1, +3)** | (439,281) | (0, −1, 0) |
| (405,171) | (−1, 0, −1) | (469,463) | (−1, −2, −1) |

- Per-vertex clip/texcoord bit-identical (update 60), position-path weights bit-identical (update 61), yet the **interpolated varying attribute differs by 1-3 ulps** (mostly GL 1-2 ulps below Metal). This is the last interpolation-level difference standing between the two backends; it propagates directly into the ray anchor and drives the 188-px knife-edge image delta.
- Combined with Result 1-3: both drivers' attribute interpolators are displaced from the pixel center by a few e-4 NDC in the same direction, and GL-on-Metal is displaced ~(+0.03, −0.03) px further than native Metal. That difference is a rasterizer/driver-level property of the varying (attribute) interpolation path, not reproducible from the flat per-vertex data we have.

`model_u63.py` closes the door on shader-side reproduction: at the pixel center, **no** analytic model — affine, perspective `1/w`, perspective `rcp(1/w)`, or inverse-w — reproduces 3/3 channels at 0 ulps for **any** pixel on **either** backend. So the varying-path bias cannot be encoded in a fragment-shader interpolation recipe using pixel-center weights.

## 6. Conclusion

- Update-62's leading hypothesis (f32 window-position rounding) is **refuted** (Result 1): the bias is a real ~(+2e-4, −3e-4) NDC effective-sample displacement, common to both backends, **GL ~(+0.03,−0.03) px beyond Metal**.
- The bias is **per-pixel** — no single offset reproduces all pixels, no phase pattern, no analytic pixel-center model reproduces either backend at 0 ulps (Results 2-4).
- The GL-vs-Metal ~1-3 ulp interpolated-texcoord difference is a **driver-level attribute-interpolator difference** (GL-on-Metal vs native Metal), outside the Metal shader source. It is the residual floor for the 188-px knife-edge flips unless Metal can emulate GL's attribute interpolation exactly — which the analytics show is not achievable from pixel-center weights.

## 7. Doubts / hypotheses (open)

1. **The interpolator displacement is a hardware sub-pixel sample convention** (e.g. the attribute plane evaluated at a slightly offset sample, or at a reduced-precision sample location) present on Apple GPUs, differing subtly between the GL-on-Metal and native-Metal drivers. Characterization requires a **controlled rasterizer experiment** (e.g. a tiny deliberately-degenerate triangle with simple attributes, dumped per-pixel) to measure the effective displacement field across the framebuffer — the knife-edge data alone is too sparse to distinguish a fixed displacement from a per-pixel one.
2. **Compensating in-shader is feasible only if the displacement is constant.** Since Result 2 shows it is not (at 0-ulp tolerance), the remaining option to reach bit-identical images is to make the ray anchor (and only the anchor) insensitive to the 1-3 ulp attribute difference — e.g. quantizing the anchor to a grid before use, if the pipeline tolerates it — or to accept the 188-px floor.
3. **GL-on-Metal position post-processing** (update 61/62 doubt) remains the explanation for the position-path GL-vs-Metal ~1 ulp split; this update shows the same class of driver difference also affects the varying path and is ~3 ulps there.

## Artifacts

- Data: `/tmp/bc/u62_gl_vlog.log`, `/tmp/bc/u62_metal.log` (frame-6, triangle 122 = (86,40,93), 14 knife-edge px).
- Scripts: `/tmp/bc/window_u63.py`, `/tmp/bc/refine_u63.py`, `/tmp/bc/backout_u63.py`, `/tmp/bc/backout_both_u63.py`, `/tmp/bc/displace_u63.py`, `/tmp/bc/texcompare_u63.py`, `/tmp/bc/model_u63.py`.
- Verified (python, /tmp/bc): window-rounding variants all give identical implied offsets ~1e-7 NDC (~2000× below the needed 2.5e-4) — hypothesis refuted; per-pixel 3/3 offset regions are broad and disjoint (no common offset; 5 px never reach 3/3); backed-out effective weights are internally consistent (res ~1e-15) and displaced identically in both backends (GL mean (+2.9e-4,−2.9e-4), Metal (+1.8e-4,−1.7e-4) NDC, GL−Metal ~(+0.03,−0.03) px); interpolated texcoord GL vs Metal differ −3..+3 ulps; no pixel-center analytic model hits 3/3 on either backend.
