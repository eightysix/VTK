# Constant-scalar volume diverges: the residual survives zero scalar variation (update 67)

Status: **milestone.** The StepTF test was rewritten as a true single-still-frame
render (no event loop; the reference `MouseWheelForwardEvent` is replicated by
`camera->Dolly(1.21)` + `ResetCameraClippingRange()`, which reproduces the
event-loop framing exactly — m2 still-frame = 3711 px = event-loop 3711 px) and
extended with variants A–E. The decisive new result: **a volume with constant
scalar (no data gradient anywhere) still diverges at 5530 px.** Since the
interpolated scalar is then identical on both backends by construction, this
falsifies the u66 "interpolated-scalar ulp through dO/ds" model as the *primary*
mechanism and redirects the search to the ray-march sample-count / accumulation
path.

## 0. CRITICAL methodology fix: zsh word-splitting bug invalidated the first matrix run

The first still-frame matrix was run via `env $envs $BIN ...` in zsh. zsh does
NOT word-split unquoted parameters, so the entire env string was passed as a
single argument and **no env var was ever set** — every test silently ran with
defaults (no wheel → background-only render). All eight rows of that first
matrix were garbage (all identical, 1 unique value). Fix: `${=envs}` (zsh
forced splitting). Everything below is from the **fixed** run.

## 1. What was added

`TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformStepTF.cxx`
rewritten:

- **Single still frame**: `renWin->Render()` → `iren->Initialize()` → second
  `Render()` → (optional wheel) → `vtkRegressionTestImage`. No event log.
- **`VTK_STEP_WHEEL`**: replicates the reference log's single
  `MouseWheelForwardEvent` (`Dolly(pow(1.1, 10.0*0.2*1.0))` =
  `Dolly(1.21)` + `ResetCameraClippingRange()`) — **required** for the volume
  to be in frame (without it the render is pure background).
- **Modes**: 0 both-step, 1 color-only-step, 2 opacity-only-step, 3 opacity
  ramp, 4 color-ramp/const-opacity, 5 window-limited opacity ramp.
- **Variants**: `VTK_STEP_DIMS` (256 / 0=raw 64³), `VTK_STEP_CONSTANT`
  (constant-scalar volume), `VTK_CAMERA_AXIS` (z-axis camera).
- Env-gated `VTK_STEP_DUMP` PNG writer for direct output inspection.

## 2. Results (all with VTK_STEP_WHEEL=1, 512×512, checkerboard dummy)

| test / setting | diff px | \|Δ\|>1 | max Δ | uniq |
|---|---|---|---|---|
| m2 opacity-step (still-frame) | 3711 | 704 | 35 | 98 |
| m0 both-step | 125 | 125 | 60 | 80 |
| A color-ramp, const-op 0.005 | 742 | 742 | 3 | 8 |
| **B constant-scalar volume, opacity ramp** | **5530** | 794 | 2 | 35 |
| C window-limited opacity ramp | 286 | 8 | 2 | 14 |
| D volume 256³, opacity ramp | 9215 | 1820 | 3 | 52 |
| D raw 64³ (no upsample), opacity ramp | 12620 | 11299 | 5 | 53 |
| E axis-aligned camera (z), opacity ramp | 13578 | 520 | 2 | 58 |
| m1 color-step only | 0 | — | — | 2 (saturated/degenerate) |
| flat near-zero alpha (ramp 0.0005) | 314 | 0 | 1 | 3 |

Cross-checks:
- **Still-frame m2 == event-loop m2**: 3711 px both — the wheel replication
  reproduces the reference framing exactly.
- **B GL self-consistent** (run twice, byte-identical) — 5530 px is a stable
  backend delta, not GL noise.

## 3. Findings

1. **Constant-scalar volume still diverges (B = 5530 px).** With uniform data
   the interpolated per-sample scalar is identical on both backends, so the
   divergence cannot come from an interpolated-scalar ulp difference. The
   output of B is a function of the per-ray **sample count n** (constant
   per-sample opacity → accumulated alpha = 1−(1−a)ⁿ → color), so B isolates
   the ray-march geometry / sample-count / accumulation path.
2. **B's diff concentrates 16× in high-gradient regions** (50.1% of diff px in
   top-10% image-gradient vs 3.0% of non-diff), full-frame bbox, max Δ=2 — the
   signature of ±1 sample-count or accumulation-rounding deltas where the ray
   length (sample count) changes steeply (volume silhouette / entry-exit).
3. **The u66 model is at best secondary.** Where scalar variation exists
   (A–E), the deltas are larger than B's, but B proves the mechanism does not
   *require* scalar variation. The dominant factor across the matrix is ray
   geometry/sample count:
   - fewer samples/ray (raw 64³: 12620; 256³: 9215 vs 512³: ~5530) → more
     divergence;
   - axis-aligned camera (rays parallel to z, long uniform runs: 13578) → most
     divergence;
   - window-limited ramp (C: 286) → least (dO/ds active over a narrow scalar
     band only).
4. **Color path is NOT clean**: A (color ramp, const low opacity) diverges at
   742 px (all >1). The earlier m1 "0 px" is a saturation artifact (2 unique
   values, alpha→1) — same trap as FlatTF in u65/u66.
5. **Even a nearly-flat opacity ramp diverges** (314 px, all exactly ±1) —
   the divergence is present at the ±1-LSB floor of accumulation even when
   opacity is essentially constant.

## 4. Doubts / hypotheses (open)

- Is B's 5530-px delta **sample-count (n) differences** between backends, or
  **accumulation-arithmetic rounding** at identical n, or **ray-geometry
  (start/end) ulps** from attribute interpolation changing n? B's structure
  (16× gradient enrichment, max 2) is consistent with ±1 sample differences at
  silhouette boundaries but does not yet separate "n differs" from "same n,
  different rounding".
- How does B reconcile with u59–64's attribute-interpolator displacement at the
  14 knife-edge pixels? Those 14 px may be a *separate* thin effect on top of
  this sample-count floor.
- Whether the divergence scales with ray length (more samples → more rounding
  exposure) or with silhouette length (more n-change events) is untested.
- E (axis camera) at 13578 px vs B 5530: is the amplification from longer
  axis-aligned rays or from the different camera clip geometry?
