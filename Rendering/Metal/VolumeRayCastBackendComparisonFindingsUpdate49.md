# The ±1 field is not a capture artifact: the image-capture pipelines of GL and Metal are consistent to the byte (both render to 8-bit BGRA drawables, blend identically, and read back the same 8-bit framebuffer through backend-agnostic harness code) (update 49)

**Date:** 2026-08-09
**Scope:** Answer the remaining open question from updates 42/44/48 — could the 63,690-px ±1 field be an artifact of the *image-capture* pipeline (the off-screen/framebuffer-readback path used only when producing the comparison PNGs), rather than a genuine framebuffer-content difference from the volume rendering arithmetic? This update walks both backends' full capture paths line-by-line. Conclusion: **No.** For `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter`, Metal renders the volume directly into an 8-bit BGRA drawable (no half-float intermediate), blends over the background with the same fixed-function factors GL uses, and its readback is a byte-identical BGRA8Unorm→BGRA8Unorm blit of that same drawable read by the same backend-agnostic `vtkWindowToImageFilter` harness GL uses. The differing bytes are therefore genuine framebuffer content — the volume arithmetic itself — consistent with update 42's inferred float-level ~3.5e-4 one-directional bias.
**Target (unchanged):** Metal output must be bit-identical to **clean GL** (`RenderingBackend=OpenGL` without debug injection).

**Follows:** [Update 48](VolumeRayCastBackendComparisonFindingsUpdate48.md), [Update 44](VolumeRayCastBackendComparisonFindingsUpdate44.md), [Update 43](VolumeRayCastBackendComparisonFindingsUpdate43.md).

---

## 1. Render target: both backends write the volume directly to an 8-bit BGRA drawable

For this test (camera inside the volume, single block, `RenderToImage` off, `ImageSampleDistance = 1.0`), the Metal mapper dispatches to the standard path and, because `cameraInside` is true, to `DrawBlocksFullscreen(..., useDirectPipeline=true)` (`vtkMetalGPUVolumeRayCastMapper.mm:7768-7773`) → pipeline type `FullscreenDirect` with `colorFormat = MTLPixelFormatBGRA8Unorm` (`mm:6332`).

The RGBA16Float offscreen targets are **not** involved for this test:

- `ImageSampleColorTexture` (RGBA16Float, `mm:1695-1771`) is used only under `useImageSampling`, i.e. image-space downsampling (`mm:7608-7612`), which is inactive at the default `ImageSampleDistance = 1.0`.
- `OffscreenLayer`/`GridTraversalOffscreen`/`RenderToImage` (RGBA16Float/RGBA8 targets, `mm:7548-7725`) require `RenderToImage` or partitioned volumes — both off here.
- The `ImageSampleBlit` pipeline (`mm:1745-1774`) composites the downsampled image back to screen; never used in this path.

This confirms and extends update 43's earlier ruling: the volume fragment output goes **straight to the BGRA8Unorm drawable**, the same 8-bit quantization GL's default framebuffer applies.

## 2. Background blend: identical fixed-function factors

- **Metal** `FullscreenDirect` (and `DirectScreen`, `GridTraversalDirect`): `src*1 + dst*(1−src.a)` for both RGB and A (`mm:6018-6031`).
- **GL** volume pass: `glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)` (`vtkOpenGLGPUVolumeRayCastMapper.cxx:1975`, also `:4808`).

Same factors, same add operation, same 8-bit framebuffer precision. Interior saturated pixels (alpha = 1 after the opacity break) collapse both sides to `src + 0`, so the blend cannot shape the image-wide field regardless.

## 3. Drawable formats: both 8-bit per channel, non-sRGB

- **Metal** CAMetalLayer: `MTLPixelFormatBGRA8Unorm` (`vtkMetalRenderWindow.mm:99`). Non-sRGB (a sRGB layer would apply a gamma transfer and produce a gross visible difference, not ±1 LSB).
- **GL** `vtkCocoaRenderWindow`: `NSOpenGLPixelFormat` attributes 3.2 Core profile, `NSOpenGLPFADepthSize` 32, 32-bit color (RGBA 8+8+8+8), double-buffered, accelerated; **no sRGB attribute** (`vtkCocoaRenderWindow.mm:964-988`). 8-bit per channel, linear.

The channel order differs (BGRA vs RGBA) but that is only a byte-order swap handled identically by each backend's readback; precision is the same 8-bit.

## 4. Readback: Metal's is a byte-identical copy of the presented drawable

**Metal** (`vtkMetalRenderer.mm:1264-1288`): at end of frame, before present, on the same command buffer, a blit copies the final resolved color target (`colorTarget`, which is the BGRA8Unorm drawable at `MSAA=0`) into the shared `ColorCopyTexture` (also BGRA8Unorm, `vtkMetalRenderWindow.mm:438-446`). The format is unchanged, so the copy is byte-identical. `vtkMetalRenderWindow::GetPixelData` then `getBytes`s that texture (waiting on the frame via `WaitForCompletion`, `mm:1032-1046`) and emits rows bottom-up with a BGRA→RGB channel swap (`mm:1050-1067`).

**GL**: `vtkOpenGLRenderWindow::GetPixelData` → `glReadPixels` from the back buffer of the double-buffered context (the very buffer the volume rendered into).

Both return the exact 8-bit bytes the volume wrote; no intermediate quantization, no re-render, no format conversion beyond the channel-order swap.

## 5. Capture harness: the same backend-agnostic code path

`vtkTesting::RegressionTest` (`Testing/Rendering/vtkTesting.cxx:397-420`) builds a `vtkWindowToImageFilter` bound to the **same live render window**, disables swap, performs one extra `Render()`, reads the back buffer, and diffs. This is identical code for both backends; the only backend-specific surface is `vtkRenderWindow::GetPixelData`, whose two implementations both return the live 8-bit framebuffer (sections 3-4).

## 6. Conclusion

The capture pipeline cannot produce the ±1 field:

1. Both backends render the volume fragment output directly to an 8-bit BGRA drawable (no RGBA16F intermediate for this test).
2. Both blend over the background with `src + (1−src.a)·dst`.
3. Both read back the same 8-bit bytes of that drawable through the same harness.

The differing bytes are genuine framebuffer-content differences from the volume rendering arithmetic — i.e. the one-directional float-level offset inferred in update 42 (§1: 8.88% flip rate ⇒ ~3.5e-4), which update 47/48 localized to clean GL's compiled arithmetic (clean GL ≠ debug GL ≠ Metal; Metal's own written formula replays 68/68). The **only** physical asymmetry left on the output side is the final fragment-float→u8 conversion (GL's fixed-function framebuffer conversion vs Metal's BGRA8Unorm store): per-pixel rounding, random sign, so it cannot by itself produce the 99.97% one-directional field, but it is the one step that remains untestable from the CPU replay (update 48's replay used round-half-even and matched Metal's store; GL's exact u8 conversion is not observable). This is retained only as a final fallback per update 48 §"store rounding".

**Next:** proceed with the Metal-side bisect of clean GL's compile-level arithmetic divergence (first variant: reproduce GL's written composite form in Metal, then whole-loop reassociation and break-condition parity variants), per update 48's §5 plan.
