# The frame-aligned comparison (update 78 §5) refutes pure frame-selection: the 178-px residual is a genuine, deterministic, same-camera Metal-vs-GL divergence (update 79)

**Date:** 2026-08-11
**Status:** **milestone (experiment executed, hypothesis refuted).** Update 78 §5's
proposed next experiment — per-frame, frame-aligned image comparison — is now
done via a raw front-buffer capture that bypasses the `-V` baseline PNG. For the
reference test (jitter on) AND its NoJitter variant, Metal's and GL's **own
rendered frames** for the **identical** W2IF camera copy differ at the **same
178 px** as the harness comparison (max Δ 8 at (397,110), which is unchanged).
GL is internally deterministic for the perturbed camera (GL raw == GL w2if,
0 px) and Metal likewise (Metal raw == Metal w2if, 0 px), so there is **no GL
frame that Metal fails to match by frame-selection** — the residual is a real
backend arithmetic/geometry divergence at knife-edge texel picks, present with
identical camera pose and with jitter disabled.

## 1. Tooling: raw front-buffer capture (`VTK_STEP_RAW_CAPTURE`)

Added to three tests (all read the front buffer via `vtkRenderWindow::
GetRGBACharPixelData(0,0,w-1,h-1,1)` after the harness finishes, write an RGB
PNG — same readback path W2IF uses, so orientation is VTK top-left):

- `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransform.cxx`
- `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter.cxx`
- `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformStepTF.cxx`

Because `vtkTestingInteractor::Start()` runs the W2IF regression *before* the
test's `InteractorEventLoop` returns, the captured frame is each backend's final
**W2IF-perturbed** frame (30.0000008°, update 19) — the exact same camera for
both backends, since the perturbation is a CPU-side float32 view-angle
round-trip shared by both. `raw == w2if` (0 px) on each backend confirms the
capture is the W2IF render. The comparison is therefore a same-camera,
frame-aligned, deterministic backend diff.

## 2. Results

| comparison | jitter-on (reference) | NoJitter variant |
|---|---|---|
| Metal(raw) vs GL(raw) | **178 px**, max Δ 8 | **178 px**, max Δ 8 |
| Metal(w2if) vs GL(w2if) (harness) | 178 px, max Δ 8 | 178 px, max Δ 8 |
| GL raw vs GL w2if (GL self-drift) | 0 px | 0 px |
| Metal raw vs Metal w2if (Metal self-drift) | 0 px | 0 px |
| jitter vs NoJitter residual sets | identical (symmetric diff 0) | — |
| Metal NoJitter run1 vs run2 | — | 0 px |

- The jitter and NoJitter residual sets are **identical** (symmetric diff 0),
  so jitter is not the mechanism.
- Delta histogram (NoJitter): 164 px at ±1, 10 at ±2, 2 at ±3, 1 at ±4,
  1 at ±8. Worst pixel (397,110): Metal (239,186,151) vs GL (239,179,143) —
  Δ (0,7,8), the same worst pixel as updates 72-74. (422,92) matches
  (Metal == GL).
- All runs deterministic (0 px drift).

## 3. What this refutes / confirms

- **Refutes update 74/78's "pure frame-selection" framing.** Update 78 §5's
  decisive test ("a single GL frame that is 0 px from Metal would prove the
  residual is pure frame-selection") is negative: the compared GL frame (the
  deterministic W2IF render) already differs from Metal at the same 178 px, so
  no frame choice would fix it. The frame-A/B anchor spread (update 78 §2) is
  real but it is a property of GL's *unperturbed→perturbed* transition
  (frame 3 → 4); within the perturbed set GL is deterministic.
- **Confirms the residual is decided at knife-edge texel picks** under an
  identical camera — i.e. a per-pixel ulp-scale backend difference that
  crosses a sampling boundary. This is consistent with update 75 rev2 (anchor
  offset = entire i=132 mechanism), update 76 (interpolated anchor 0-4 ulp,
  z per-vertex texcoord not bit-identical GL↔Metal), and update 78 (analytic
  z = frame-A + 1 ulp regardless of precision).
- The analytic-anchor experiment (update 78) remains correctly negative: the
  analytic z is *above* GL's interpolated z, so it tips knife-edge picks the
  same wrong way as Metal's hardware interpolator.

## 4. Live mechanism and next lever

The remaining mechanism is the **z per-vertex texcoord of the near-cap** (update
76 §3: GL's near-cap z readback is an encoding artifact; Metal's modelPos.z is a
genuine intersection), propagated through the rasterizer to a systematic ±1-ulp
anchor-z offset that decides the 178 texel picks. Next experiments, in order:

1. Capture the near-cap z per-vertex texcoord GL↔Metal for the perturbed
   camera and diff at bit level (the update-76 §3 artifact) to confirm the z
   input, not the interpolator, is the +1-ulp source.
2. If the z input is the source, make Metal's near-cap z texcoord bit-identical
   to GL's (replicate the encoding artifact) and re-measure the aligned
   residual.
3. If the residual drops toward 0, the alignment metric (`VTK_STEP_RAW_CAPTURE`
   raw diff) is the stable acceptance criterion for the bit-identical goal; the
   `-V` W2IF path can then never be the metric (it inherits the frame-3→4
   selection choice).

## 5. Files / commands

- Raw capture knob: the three test files above (guarded by
  `VTK_STEP_RAW_CAPTURE`, reads front buffer, RGB PNG).
- Capture run (per backend):
  `VTK_STEP_RAW_CAPTURE=<out>.png <test binary> TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter --vtk-factory-prefer RenderingBackend={Metal,OpenGL} -D <DataRoot> -T <Tmp> -V <baseline>.png`
- Images (this session): `…/u78/{nj_gl_raw,nj_mt_raw,nj_mt_raw2,ref_gl_raw,ref_mt_raw}.png`;
  residual set `…/u78/nj_resid_set.txt` (178 px).
- Prior anchors: update 78 (analytic-anchor negative), update 76 §3 (z per-vertex
  texcoord artifact), update 75 rev2 (anchor offset mechanism), update 19 (W2IF
  view-angle perturbation).
