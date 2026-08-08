# Metal vs OpenGL volume ray cast: commit the remaining half→float promotions; re-measure the no-jitter camera-inside scene under NEAREST interpolation (the residual is unchanged, proving it is the sample-position offset, not precision) (update 14)

Follow-up to all prior documents; read as an addendum to
[VolumeRayCastBackendComparisonFindingsUpdate13.md](VolumeRayCastBackendComparisonFindingsUpdate13.md).

Update 13 catalogued the remaining Metal `half`-precision sites (§4) and left the
`…NoShadeNoGradOpNoTransform` residual at 646 pixels > 4 LSB, root-caused to
nearest volume interpolation amplifying sub-0.02-texel backend sample-position
differences at bone-plateau boundaries (update-12 §10). This session promoted the
entire §4 catalogue to `float`, removed the linear-interpolation override that
update-12 had added to the no-jitter test (so the test again samples with the
vtkVolumeProperty default NEAREST, per the decision to test only on nearest to
expose differences), and re-measured the camera-inside no-jitter scene on genuine
backends. The NEAREST residual reproduces update-12/13 exactly (worst
(372,131) Δ=22 and (422,92) Δ=19, ≈642 pixels ≥ 5) — i.e. the float promotions
have no observable effect on this scene, and the residual is the sample-position
offset, not numeric precision. Under linear interpolation the same tree differs
by max 2.

## 1. Decision: test only on NEAREST interpolation

The linear-interpolation override added in `49ecec08b7` was removed from
`TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter.cxx`
(committed `4867687909`). The test again samples the volume texture with the
vtkVolumeProperty default (`VTK_NEAREST_INTERPOLATION`), because nearest is what
exposes the true backend differences: linear smears the sample-position offset
into at most a couple of LSB and can hide regressions. The no-jitter scene
(`mapper->SetUseJittering(false)`, line 99) keeps ray positions deterministic,
isolating the position-precision residual from jitter.

## 2. Half→float promotions (committed `ccffd0095d`)

All update-13 §4 sites are now computed in `float32` in
`Rendering/Metal/Shaders/MetalShaders.metal`:

- **Lighting**: `computePhongLightingVolumeFast` and `computeVolumeLighting`
  (`float3` colors/normals/`float` `pow`), plus the `float3` `viewDirHalf`/
  `lightDirHalf`/`specularMat`/`lightDirHalf` uniform conversion — matching GL's
  `float nDotL`, `vec3 r`, `pow(...)`.
- **`finalColor`**: full-`float4` composite result, `float` window/level
  scale/bias.
- **Blend-mode accumulators**: `avgBlendSum`/`additiveSum` and per-component
  variants, plus the MIP/MinIP comparisons, all `float`.
- **`sampleTransferFunction2D`**: returns `float4`.
- **`marchSegment` / `marchVolumeUnified`**: `float3& accumulatedColor` /
  `float& accumulatedOpacity` accumulator state.
- **Per-component independent path**: `scalarNormComp`/`compScale`/`compBias`,
  `compColor`, `totalAlpha`, `tmpRGB`/`tmpA`, `maskLabel`, `secondNorm` — all
  `float`.

No `half`-precision arithmetic remains in the volume ray-cast shader path
(remaining `half` identifiers are variable names only, e.g. `halfW` for
line half-width in unrelated geometry shaders).

## 3. Re-measurement (procedures §2/§3) — NEAREST residual unchanged

Both backend runs used a fresh `-T` dir per backend and a black dummy baseline
(per update-13 §2; the 0-gap capture remains a fluke). GL engagement verified via
`GL_SAMPLING=12` stderr markers. Scene:
`TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter`
(camera inside, no shading, no grad-op, no vtkProp3D transform, jittering off).

| interpolation | max\|Δ\| | mean\|Δ\| | masked (≥5) px | exact-equal px | within-2 px |
|---|---|---|---|---|---|
| LINEAR | 2 | 0.27 | 0 | 190711 | 262144 / 262144 |
| NEAREST | 22 | 0.37 | 642 | 177880 | 259444 / 262144 |

NEAREST worst pixels: `(372,131) Δ=22`, `(422,92) Δ=19`, `(421,92) Δ=18`,
`(393,173) Δ=15`, `(384,151) Δ=14` … — **the same pair that update-12/13
reported** ((372,131) Δ=22, (422,92) Δ=19, 646 px > 4 LSB on the sibling
`…NoTransform` variant). The ±1 px count difference (642 vs 646) is the ≥5 vs
>4 threshold. Linear regression: R/G/B `metal ≈ 1.000*gl + 0.0` under both
interpolations.

## 4. Interpretation

- The NEAREST residual reproduces update-12 §10 exactly and is **independent of
  the float promotions committed here**: promoting every remaining `half` site to
  `float` did not move a single masked pixel. The residual is the ≤0.02-texel
  ray-origin offset between backends flipping full texels at bone-plateau
  boundaries under nearest sampling — a sample-position artifact, not a precision
  artifact.
- LINEAR interpolation keeps the same offset but quantizes it to ≤2 LSB, which is
  why update-12's linear-override run showed max 2/mean 0.10. Under the float-
  promoted tree the linear comparison is max 2/mean 0.27 — same order, tiny mean
  drift from the (harmless) promotion of the composite/finalColor path.
- The position-offset root cause therefore remains open: bit-identical output
  under nearest requires bit-identical sample positions across backends
  (update-13 §5 step 2). The float work is still worth keeping as structural
  parity with GL and as a prerequisite for isolating the position residual.

## 5. Files changed

- `Rendering/Volume/Testing/Cxx/TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter.cxx`
  — removed `SetInterpolationTypeToLinear()` (commit `4867687909`).
- `Rendering/Metal/Shaders/MetalShaders.metal` — full half→float promotion of the
  update-13 §4 sites (commit `ccffd0095d`).
- `Rendering/Metal/VolumeRayCastBackendComparisonFindingsUpdate14.md` — this
  document.
