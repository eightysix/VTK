# TestGPURayCastRenderToTexture: the Metal RTT pass blends over the white clear, OpenGL's does not — the 47,878-px contour-concentrated residual (update 84)

**Date:** 2026-08-12
**Status:** **Root cause found, proven per-pixel, and FIXED.** Blending on the
Metal `RenderToImage` pipeline color attachment 0 was disabled
(`blendingEnabled = NO`, `GetOrCreateVolumePipeline`'s `RenderToImage` branch,
applied 2026-08-12). Post-fix, the final-frame Metal↔GL diff on
`TestGPURayCastRenderToTexture` dropped from **47,878 px / max_d 65 (29.9 %)**
to **1,471 px (0.9 %) / max_d 1** — no deltas ≥2, all ±1 rounding confined to
the head-projection bbox (x 65–336, y 79–310), i.e. the predicted
knife-edge/rounding floor (LINEAR + ShadeOn + sharp TF ramp on coarse 64³
data). Both backends byte-deterministic across runs. On
`TestGPURayCastRenderToTexture` (401×399, headsq/quarter 64³ CT, LINEAR
interpolation, `ShadeOn`, opacity ramp 0→0.15 at scalar 900), the Metal↔GL
final-frame diff is **47,878 px / max_d 65** (29.9 %), and the RTT color
texture differs on **144,852 px** with **max_d 255** while its **alpha field
is essentially bit-identical (61 px, max_d 1)**. The difference is the Metal
RenderToImage pipeline blending its shader output over the cleared white RTT
background with `ONE, ONE_MINUS_SRC_ALPHA`, while OpenGL's RenderToImage pass
writes the raw (unblended) shader output. Per-pixel proof: every drawn RTT
pixel satisfies `Metal_RGB == GL_RGB + (1 − α)·255` (to ≤1 u8), e.g. at
(138,23) α=111: GL=(72,51,28), Metal=(216,195,172), Δ=(144,144,144) =
(1−111/255)·255. The final frame residual is that injected white halo
surfacing at the volume contour via the image actor's linear filtering, which
is why the diff looks "mostly in volume contour".

## 1. The comparison, reproducibly

Harness as in `BackendComparisonTools/run_pixel_diff_suite.sh` (checkerboard
dummy so the `-V` regression fails and dumps; `--vtk-factory-prefer
RenderingBackend=OpenGL` vs `=Metal`, same `vtkRenderingVolumeCxxTests`
binary). 401×399 (159,999 px). `VTK_RTT_DUMP=<file>` (stash hook in the test
source) dumps the `GetColorImage` vtkImageData; the `-V` fail-dump is the
final frame (image actor displaying the RTT image over black). Both captures
were reproduced byte-identical across three independent capture sets.

| capture | diff px | \|Δ\|≥2 | \|Δ\|≥5 | max_d |
|---|---|---|---|---|
| Final frame (the regression image) | **47,878** (29.9 %) | 6,875 | 6,053 | **65** |
| RTT color texture (raw `GetColorImage` RGB) | 144,852 (90.5 %) | 57,045 | 55,598 | 255 |
| RTT **alpha** | **61** | — | — | **1** |

The Δ-magnitude structure of the frame diff: Δ=1 → 41,003 px (86 % of the
diff), Δ 2–4 → 822, Δ 5–29 → 1,789, Δ 30–65 → 4,264. Every diff pixel lies
inside the head projection (bbox x 64–336, y 78–315); the background is
clean. The large deltas (≥30) sit on the head's surface/edge region — the
"volume contour".

## 2. Why the alpha is the tell

The RTT color texture is RGBA8 cleared to opaque white (1,1,1,0) on both
backends (`vtkOpenGLGPUVolumeRayCastMapper.cxx`:
`vtkglClearColor(1.0,1.0,1.0,0.0)`; Metal:
`MTLClearColorMake(1.0,1.0,1.0,0.0)`). The shaders accumulate the same
front-to-back composite (the recap proved the composite/accumulation is
bit-identical for the on-screen family), so the shader's raw output `src` and
`srcAlpha` match between backends — confirmed here by the alpha channel
matching to 61 px / max_d 1 (blending alpha with dstAlpha=0 leaves
`alpha_result == srcAlpha`, so both alphas are equal).

With the two backends writing the *same* raw `src`:

- **OpenGL, unblended:** `GL_RTT = src` (raw premultiplied color).
- **Metal, blended `ONE, ONE_MINUS_SRC_ALPHA` over the white clear:**
  `Metal_RTT = src + (1 − α)·255`.

Hence `Metal_RTT − GL_RTT = (1 − α)·255`, exact per drawn pixel. Verified on
every drawn pixel across the frame; the only exceptions (12,832 px) are
pure-background pixels where neither backend draws (both stay clear white) and
the relationship is trivially inapplicable. Concrete samples along the contour
row y=23:

| px | α | GL_rgb | Metal_rgb | Δ | (1−α)·255 |
|---|---|---|---|---|---|
| (138,23) | 111 | (72,51,28) | (216,195,172) | (144,144,144) | 144 |
| (141,23) | 123 | (78,54,30) | (210,187,162) | (132,133,132) | 132 |
| (143,23) | 132 | (82,57,30) | (205,180,154) | (123,123,124) | 123 |

The maximum RTT Δ of 255 occurs in the low-α contour shell (where
`(1−α)·255` is large), and the opaque interior (α≥200) differs by only
4,919 px with |Δ|≥2 (max 56) — exactly the `(1−α)·255` residue. The frame
diff is smaller (max_d 65) because the image actor draws the RGBA image with
alpha applied over black; the residual concentrates at the volume contour
because that is where the white halo is largest and where the image actor's
linear filtering spreads the near-white texels into the opaque edge.

## 3. Root cause in the code

Metal's `RenderToImage` pipeline enables blending on color attachment 0,
mirroring the on-screen composite path, but the OpenGL RTT pass does not:

```
Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm
  GetOrCreateVolumePipeline, VolumePipelineType::RenderToImage branch:
    pipelineDesc.colorAttachments[0].blendingEnabled = YES;
    sourceRGBBlendFactor = MTLBlendFactorOne;
    destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
```

`vtkVolumeStateRAII`'s `ONE, ONE_MINUS_SRC_ALPHA` setup does run for the GL
volume draw, yet the empirical evidence is unambiguous that GL's RTT color
attachment stores the raw shader output (no `(1−α)·255` term), while Metal's
stores the blended result. The mismatch is backend behavior on the
render-to-image target, not shader output (the composite is bit-identical).

**Fix:** disable blending on the Metal `RenderToImage` pipeline color
attachment 0 (`blendingEnabled = NO`), so Metal writes the same raw
premultiplied shader output as OpenGL.

The `git stash` entry `stash@{0}` (WIP on metal-ios, top of `d8a8efda23`)
contains a comment block documenting exactly this diagnosis ("The OpenGL RTT
pass is UNBLENDED … Blending here with ONE/ONE_MINUS_SRC_ALPHA instead
injected (1-alpha)*255 into every RTT pixel, which was the contour-concentrated
47,878-px GL-vs-Metal residual") plus the `VTK_RTT_DUMP` test hook — but it
only adds the comment; it does **not** change the pipeline. This document is
the numeric proof; the code change is applied in the follow-up.

## 4. Relationship to the recap's residual

This is a *structural* RTT bug, distinct from the 512² reference family's
178-px interpolator-floor residual (recap updates 78–81). The recap's on-screen
family never renders to a texture; this test is the first one exercising
`RenderToImage` in the comparison effort, so the blended-RTT defect was latent.
Once the blend is removed, whatever remains on this test is expected to be the
usual knife-edge/rounding floor (this test amplifies it via LINEAR + ShadeOn +
sharp TF ramp on coarse 64³ data), to be quantified after the fix. **Post-fix
(2026-08-12):** that floor is **1,471 px / max_d 1** — all ±1 rounding, no
|Δ|≥2 — confined to the head-projection bbox (x 65–336, y 79–310), confirming
the prediction.

## 5. Reproduction

```
WORK=/tmp/bc/rtt_check   # GL/Metal -V fail-dump + VTK_RTT_DUMP captures
./macos_metal_build.sh --resume --tests
# GL:  VTK_RTT_DUMP=gl_rtt.png  bin/vtkRenderingVolumeCxxTests \
#        TestGPURayCastRenderToTexture --vtk-factory-prefer RenderingBackend=OpenGL \
#        -D $EXT -T gl -V dummy.png
# Metal: same with RenderingBackend=Metal, mt_rtt.png, -T mt
# frame = tmp/dummy.png (the -V fail-dump); RTT = the VTK_RTT_DUMP output
```
