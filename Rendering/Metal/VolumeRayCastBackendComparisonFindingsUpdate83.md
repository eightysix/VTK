# Jitter noise parity: replace Interleaved Gradient Noise with GL's exact blue-noise tile (kBlueNoise64), composing the vtkJPEGReader bottom-up decode with the GL/Metal y-origin flip (update 83)

**Date:** 2026-08-12
**Status:** **Jittered renders now bit-match GL's noise field; the last Metal
shader that used a *different* noise source from GL is retired.** On the only
jittered comparison test (`TestGPURayCastCameraInsideNonUniformScaleTransform`,
300², `SetUseJittering(1)`), the Metal↔GL pixel diff collapses from
**45,840 px / max_d 139** (IGN noise, different per-pixel lattice phase) to
**29 px / max_d 13** (the same interpolator floor as the 512² reference); with
the poke matrix removed the whole jittered pipeline is **bit-identical (0 px)**.
The 512² reference family is untouched because those tests never enable
jitter (`vtkGPUVolumeRayCastMapper::UseJittering` defaults to 0) — which also
corrects the recap's misleading "Reference (jitter on)" label.

## 1. The diff, file by file

```
Rendering/Metal/Shaders/MetalShaders.metal          (modified, +336/-36)
  - remove volume_random() (Interleaved Gradient Noise, Jimenez 2014)
  + add constant uchar kBlueNoise64[4096]           (the 64² luminance tile)
  + add inline float sampleJitterNoise(float2 st, float viewportH)
  - three call sites changed to sampleJitterNoise:
      marchVolume                         (fullscreen / cap pass)
      fragment_volume_rtt_main            (render-to-texture pass)
      fragment_volume_grid_traversal_main (grid pass; also dropped the
        old +0.5 half-pixel shift in in.position.xy)
  + corrected comment block documenting the vtkJPEGReader row flip and the
    GL-vs-Metal y-origin composition

Rendering/Volume/Testing/Cxx/CMakeLists.txt          (modified, +1)
  + TestGPURayCastCameraInsideNonUniformScaleTransformKnobs.cxx

Rendering/Volume/Testing/Cxx/
  TestGPURayCastCameraInsideNonUniformScaleTransformKnobs.cxx   (new)
  - knob-sweep variant of TestGPURayCastCameraInsideNonUniformScaleTransform
    (300², camera inside the bbox of a volume with poke matrix
    diag(3.2,3.2,1.5)+t(200,100,40), view angle 170, jitter ON by default).
    Independent env knobs so a Metal↔GL diff can be attributed to one
    subsystem at a time:
      VTK_NUS_JITTER  0 = SetUseJittering(false)
      VTK_NUS_SHADE   0 = ShadeOff
      VTK_NUS_GRADOP  0 = no gradient opacity
      VTK_NUS_POKE    0 = no poke matrix
      VTK_NUS_RAW_CAPTURE <file> = raw front-buffer capture (frame aligned,
        bypassing the vtkWindowToImageFilter float32 view-angle round trip)
  - 300 is NOT a multiple of 64, so it is the test that exposes the noise
    row-flip (a 512² or 256² window masks it: (py_mt − H) mod 64 collapses
    to py_mt mod 64).

Rendering/Metal/VolumeRayCastBackendComparisonRecap.md    (modified, +18)
  - new section 4c "Update 83" summarizing the finding + A/B numbers
```

## 2. Root cause: two vertical flips compose into the GL noise row

GL's volume shaders sample a real blue-noise texture for jitter:

- `vtkOpenGLRenderWindow::GetNoiseTextureUnit()` (`vtkOpenGLRenderWindow.cxx:3045`)
  decodes the embedded `BlueNoiseTexture64x64.jpg` with **vtkJPEGReader**,
  uploads component 0 / 255 as a 64×64 float texture, REPEAT wrap, NEAREST.
- `vtkVolumeShaderComposer.h:481` samples it as
  `texture2D(in_noiseSampler, gl_FragCoord.xy / vec2(textureSize(...))).x`.

The old Metal shader used Interleaved Gradient Noise at the same pixel
coordinate. IGN is a hash of the coordinate; GL's tile is a pre-computed
blue-noise table. The two fields differ at essentially every pixel, so every
*jittered* Metal render marched a sample lattice with a different phase than
GL's at every pixel — a 45,840-px blow-up on the one test that enables jitter.

**Flip 1 — the data.** `vtkJPEGReader` writes its output rows **bottom-up**:

```
vtkJPEGReader.cxx:344-351
  long destLine = cinfo.output_height - cinfo.output_scanline;
  ...
    memcpy(outPtr2, row_pointers[linesRead - i - 1] + ..., outSize);
```

i.e. output row `y` holds JPEG row `63 − y`. Verified empirically: the reader's
decode of `Rendering/OpenGL2/textures/BlueNoiseTexture64x64.jpg` satisfies
`vtk[y][x] == pil[63-y][x]` for **4096/4096** values vs PIL's decode
(`/tmp/bc/vtkjpeg_vals.txt` vs `/tmp/bc/blue_noise_64.npy`). GL's texture row
`r` therefore holds JPEG row `(63 − r) mod 64`.

**Flip 2 — the coordinate.** GL `gl_FragCoord.y` counts from the bottom of the
viewport; Metal `in.position.y` counts from the top. For the same physical
pixel in Metal top-down row `py_mt`, GL sees row `py_gl = H − 1 − py_mt`.

**Composition.** GL samples texture row `py_gl mod 64` = JPEG row
`(63 − py_gl) mod 64` = `(63 − (H − 1 − py_mt)) mod 64` = `(py_mt − H) mod 64`.

So GL's effective JPEG row for a Metal top-down pixel `py_mt` is
**`(py_mt − H) mod 64`** — the exact formula the new shader uses. When `H` is a
multiple of 64 this collapses to `py_mt mod 64` (why the 512² reference family,
even if it *had* jitter, would look right), and a 300² window does not.

## 3. The fix

`kBlueNoise64[4096]` is the luminance tile in the **JPEG's top-down (PIL)
byte order** (verified: equals `PIL.Image.open(BlueNoiseTexture64x64.jpg)`
4096/4096, equals vtkJPEGReader's decode flipped). The value `byte / 255.0f`
is therefore exactly what GL's texture contains *after* the row flip is
accounted for by the index, not at the raw index — the two candidate
orientations are deliberately separated: the array stores JPEG orientation and
the sampling index applies the flip.

```metal
inline float sampleJitterNoise(float2 st, float viewportH) {
  int2 t = int2(floor(st.x), floor(st.y - viewportH)) & 63;
  return float(kBlueNoise64[t.y * 64 + t.x]) / 255.0f;
}
```

- `st` is the pixel center: Metal `in.position.xy == (pixel + 0.5)` for all
  three passes (fullscreen, RTT, grid).
- `floor(st.y - viewportH) & 63` = `(py_mt − H) mod 64` (two's-complement
  `& 63` ≡ mod 64 for negative operands; `floor` not truncation, so the +0.5
  half-pixel cancels exactly against the integer `H`).
- `floor(st.x) & 63` = `px mod 64` (GL's NEAREST/REPEAT column pick).
- The grid-traversal pass previously sampled at `in.position.xy + (0.5,0.5)`;
  GL has no separate grid pass and samples at `gl_FragCoord.xy`, so the +0.5
  desynchronized the Metal grid pass from its own composite passes and from
  GL. Removed.

### Verification that the formula equals GL's sampler

Simulated against the *actual* GL-side data (vtkJPEGReader's decode as the
texture, gl_FragCoord semantics) and the shader's array at every pixel:

```
H=512: 262144/262144   H=256: 65536/65536   H=128: 16384/16384
H= 65: 4225/4225       H=105: 11025/11025   H= 63: 3969/3969
H= 66: 4356/4356       H= 24: 576/576
```

100 % match at every tested viewport height — including heights that are not
multiples of 64, where the flip actually matters.

## 4. A/B: IGN (HEAD) vs kBlueNoise64 (this diff)

`TestGPURayCastCameraInsideNonUniformScaleTransformKnobs`, 300², raw
front-buffer capture, same explicit camera on both backends
(`VTK_NUS_RAW_CAPTURE`; this bypasses the W2IF float32 camera perturbation, so
the two frames are the exact same camera):

| config | IGN (pre-fix, HEAD) | blue-noise + flip (post-fix) | IGN delta contribution |
|---|---|---|---|
| default (jitter on, all knobs on) | 45,840 px / max_d 139 | **29 px / max_d 13** | 45,811 px (99.94 %) |
| `VTK_NUS_POKE=0` (no poke matrix) | 33 px / max_d 94 | **0 px** (bit-identical) | 33 px |
| `VTK_NUS_JITTER=0` (jitter off) | 31 px / max_d 49 | 31 px / max_d 49 | 0 px |
| `VTK_NUS_SHADE=0` | 41,553 px / max_d 150 | 12 px / max_d 1 | 41,541 px |
| `VTK_NUS_GRADOP=0` | 43,782 px / max_d 197 | 8 px / max_d 19 | 43,774 px |
| `VTK_NUS_POKE=0` + `VTK_NUS_JITTER=0` | 0 px | 0 px | 0 px |
| `VTK_NUS_POKE=0` + `VTK_NUS_SHADE=0` | 31 px / max_d 51 | 0 px | 31 px |
| `VTK_NUS_POKE=0` + `VTK_NUS_GRADOP=0` | 33 px / max_d 136 | 0 px | 33 px |

**IGN delta contribution = pre-fix − post-fix px**: the number of the pre-fix
diff pixels that were *caused by the IGN noise-field mismatch* (each backend
marched a lattice with a different per-pixel phase), as opposed to the
interpolator floor that both noise implementations share. On the reference
config the IGN mismatch contributed **45,811 of 45,840 px (99.94 %)** of the
pre-fix diff — only 29 px (0.06 %) survive the noise fix, i.e. the floor.

The contribution vanishes exactly when jitter is off (`nojitter`: 0 px, both
variants bit-identical on the same pixels) and when there is nothing to be
flipped by a phase shift (`nopoke`+`nojitter` under IGN: already 0 px). With
jitter on, IGN contributes ~33 px (no-poke configs) to ~46 k px (poke configs);
the poke-transform geometry spreads each noise-driven lattice-phase shift into
far more knife-edge texel-pick flips.

Key points:

- **`nopoke → 0 px`** is the cleanest proof: with the poke matrix removed, the
  *entire jittered* pipeline (noise field, lattice phase, anchor, composite)
  is bit-identical between Metal and GL. The 29-px poke residual is the same
  interpolator-floor knife-edge mechanism as the 512² reference, on the
  non-uniform scale transform — not noise.
- The `noshade` and `nogradop` rows confirm the floor shrinks as subsystems
  are disabled, down to bit-exactness when the non-uniform poke transform is
  removed.
- Deterministic: `RUNS=2`-style byte-compare of the captures on each backend
  shows no self-drift (same protocol as update 69 §2).

## 5. The 512² reference family is untouched — and the "jitter on" label was wrong

The comparison suite's 512² family never enables jitter:

- `vtkGPUVolumeRayCastMapper.cxx:47` — `this->UseJittering = 0;`
- None of the `...TransformationNoShadeNoGradOpNoTransform*` tests call
  `SetUseJittering` (grep confirms only
  `TestGPURayCastCameraInsideNonUniformScaleTransform` and its Knobs variant
  set it to 1 in this family).

Re-run of the acceptance gate on the fixed build is **unchanged**:

```
Reference (jitter-off): diff=178 |d|>=2=14 |d|>=5=1 max_d=8
NoJitter:               diff=178 |d|>=2=14 |d|>=5=1 max_d=8
FlatTF:                 diff=  0
```

So the recap's baseline-table label "Reference (jitter on)" applied to no test
in the family — those rows are jitter-off, which is why the noise fix cannot
move them. The only jittered test is the 300² non-uniform one, and that is
where the 45,840 → 29 px improvement lives.

## 6. Residual-consistent with the interpolator floor

At 29 px, the jittered 300² non-uniform test sits right at the same
interpolator hardware floor that bounds the 512² reference (updates 76–81):
per-vertex inputs, matrices, `evalStep`, the anchor, and now the noise field
are all bit-identical or GL-driven; the ±1–2 ulp rasterizer-interpolator
rounding of the ray anchor flips nearest-texel picks at knife-edge rays. The
`nopoke → 0 px` result is the strongest evidence yet that *every shader-visible
input* is now bit-identical and only the poke-transform geometry amplifies the
interpolator difference.

## 7. Edge case not covered (documented, not fixed)

The formula uses `H = volumeUniforms.viewportSize.y` (the render target / tile
height, `vtkMetalGPUVolumeRayCastMapper.mm:7593-7594`) and assumes the
viewport spans the render target from origin 0 (the standard single-renderer,
full-window layout of every test here). For a *tiled* viewport with a nonzero
origin, Metal `in.position.y` and GL `gl_FragCoord.y` each include their own
(oppositely-measured) origin offset and the exact formula would need an
additional `(framebufferHeight − H) mod 64` term. Out of scope for the current
test family; GL-parity in that layout is not exercised.

## 8. Artifacts / reproduction

```
WORK=/tmp/bc/nus_jit
BIN=build_macos_metal/bin/vtkRenderingVolumeCxxTests
EXT=build_macos_metal/ExternalData/Testing
env VTK_NUS_RAW_CAPTURE=$WORK/gl.png "$BIN" \
    TestGPURayCastCameraInsideNonUniformScaleTransformKnobs \
    --vtk-factory-prefer RenderingBackend=OpenGL -D "$EXT" -T $WORK/t_gl -V $WORK/dummy.png
env VTK_NUS_RAW_CAPTURE=$WORK/mt.png "$BIN" \
    TestGPURayCastCameraInsideNonUniformScaleTransformKnobs \
    --vtk-factory-prefer RenderingBackend=Metal  -D "$EXT" -T $WORK/t_mt -V $WORK/dummy.png
python3 -c "import numpy as np;from PIL import Image;g=np.array(Image.open('$WORK/gl.png').convert('RGB')).astype(int);m=np.array(Image.open('$WORK/mt.png').convert('RGB')).astype(int);d=np.abs(m-g).max(2);print('diff',int((d>=1).sum()),'max_d',int(d.max()))"
```

Pre-fix IGN numbers: the same two invocations against a build of HEAD
(`git checkout HEAD -- Rendering/Metal/Shaders/MetalShaders.metal`, rebuild,
run, restore).

Captures used for this update:
- post-fix: `/tmp/bc/nus_final/{gl,mt}.png` (default), `/tmp/bc/nus_final/{gl0,mt0}.png` (nopoke), `/tmp/bc/pixdiff_final/` (512² gate)
- pre-fix:  `/tmp/bc/nus_ign/{gl,mt}.png` (IGN default), `/tmp/bc/nus_ign/{gl0,mt0}.png` (IGN nopoke)
- verification: `/tmp/bc/vtkjpeg_vals.txt`, `/tmp/bc/blue_noise_64.npy`,
  `/tmp/bc/MetalShaders.metal.fixed`
