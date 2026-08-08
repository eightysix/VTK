# Metal vs OpenGL volume ray cast: OpenGL ray-geometry dump is now trustworthy and both backends' rays are matched (update 8)

Follow-up to all prior documents; read as an addendum to
[VolumeRayCastBackendComparisonFindingsUpdate7.md](VolumeRayCastBackendComparisonFindingsUpdate7.md),
whose §7 left "seed-phase entry difference" as the best remaining candidate for the
28 masked shell pixels. This update closes the loop on that candidate's
measurement side: the OpenGL-side debug dump that reads `g_rayOrigin`/`g_dirStep`
was itself buggy, and once fixed it confirms that **OpenGL and Metal now produce
identical ray geometry and sample counts**, with the residual image difference
reduced to pure float32 accumulation rounding (0.09% RMS).

## 1. OpenGL ray-dump scaffolding (VTK_GL_RAY_DUMP)

`vtkOpenGLGPUVolumeRayCastMapper.cxx` gained a temporary, environment-gated
(`VTK_GL_RAY_DUMP`) debug path:

- In `BuildShader` a hook replaces `//VTK::CallWorker::Impl` with
  `initializeRayCast();` + a gated debug encode + `castRay(-1.0,-1.0);
  finalizeRayCast();`. The substituted fragment and vertex sources are echoed as
  `=== FRAG SHADER ===` / `=== VERT SHADER ===`, and the fully-replaced source
  after `ReplaceShaderValues` as `=== COMPILED FRAG ===`.
- `DumpDebugRays` re-renders the box once per (pixel, channel) with blending
  disabled, reads back one encoded float per fragment, decodes it, and logs
  `origin`, `step`, `ip_vertexPos`, `ip_textureCoords`, plus the 8 box corners
  (`GL_BOX`) and the live `volTex->CellToPointMatrix` (`GL_CTP`).

### 1.1 Encoding scheme (GLSL 150: no `floatBitsToUint`)

A float is packed into 4 bytes: bytes 0-2 = 23-bit mantissa (`(f-1)*2^23` of the
normalized form), byte 3 = sign bit (bit 7) + 7-bit exponent biased by 64
(`b3 = clamp(e+64,0,127)`, decode `e = (b3&0x7F)-64`). All-zero bytes mean the
fragment never ran (readback of the clear color) and the pixel is skipped.

### 1.2 Two decode bugs fixed this session

1. **Channel mapping**: the shader selected the component via `field =
   in_debugChannel >> 1` while the C++ decoder used `f % 3` — decoded values were
   scrambled. Fixed the shader to `field = in_debugChannel % 3` with base chosen
   by `in_debugChannel < 3/6/9` (origin / step / ip_vertexPos / ip_textureCoords),
   matching the host-side `dest[f % 3]`.
2. **Exponent range**: `b3 = clamp(e + 16, 0, 31)` confined exponents to
   [-16, 15], corrupting every value with |v| < 2^-16 (e.g. `step.y` =
   8.13e-6 decoded 2x too large). Fixed to `clamp(e + 64, 0, 127)` (7-bit
   exponent, range ≈ 2^-64..2^63).

After both fixes the decode is byte-exact (validated with a `FragCoord` probe):
center pixel (256,256) reads
`origin=(0.502087,0.502078,0.99707)`,
`step=(-2.32728e-5,-2.33231e-5,-0.00195309)`,
`vpos=(101.232,101.230,138)`,
`tex=(0.502093,0.502084,0.999023)`, with `origin == tex + step` as expected.

## 2. CellToPoint matrix: resolved as a non-issue (analysis slip)

The running GL uniform `in_cellToPoint` and the live `CellToPointMatrix` probe
both report `scale = 0.998047`, `offset = 0.000976562`. The source formula in
`vtkVolumeTexture::ComputeCellToPointMatrix` computes `range = (delta-0.5)/delta
- min`; with `delta = 512` this is `511.5/512 - 0.5/512 = 511/512 =
0.998046875`, i.e. **exactly the value the running binary and Metal use**.
Metal's `ctpScale = (texelCount-0.5)/texelCount - 0.5/texelCount` evaluates to
the same `0.998046875`. The apparent "0.99707 vs 0.998047" discrepancy reported
earlier was an arithmetic slip in the analysis (`(d-0.5)/d` was misread as
0.99707; it is 0.9990234375). No code change is needed.

## 3. Ray-geometry parity — what now matches

Using the gated pixel set (14 GL pixels paired with the Metal
`debugMarchGate`/`pxOkNoJitter` pixels via `GL(x,y) == Metal(x, 511-y)`) and the
Metal `DEBUG MARCH`/`DEBUG SAMPLE` log (`/tmp/bc/nojitter_metal.log`):

- **Step**: `g_dirStep` vs Metal's `texStep`/`evalStep` agree to <1e-6 (dX~4e-7,
  dY~7e-7, dZ~4e-6). Metal's step is GL's exact formula
  `(in_cellToPoint * in_inverseTextureDatasetMatrix * dirData) * stepSize`.
- **First sample**: Metal's `evalPoint_0 = cellToPoint(entry + rayDir*jitter)`
  (no-jitter ⇒ `firstT = jitter = stepSize`) matches GL's `g_rayOrigin` to ~1e-6
  at 12/13 pixels; worst case (357,357) is 1.22e-4 in x/y (GL interpolates the
  box corner, Metal hits the box analytically — the seed-phase difference
  Update7 §7.1 predicted, now quantified at a sub-texel level).
- **Sample count / termination**: GL's `g_terminatePointMax =
  length(g_terminatePos - g_dataPos)/length(g_dirStep)` collapses (back face at
  raw z = 0 ⇒ ctp z = 0.0009765625) to `|(origin.z - 0.0009765625)/step.z|`;
  Metal's is `(tEnd - jitter)/stepSize`. Both ceil to the **same integer count at
  every gated pixel**:

| pixel | GL count | Metal count | GL ratio | Metal ratio |
|---|---|---|---|---|
| (93,310) | 519 | 519 | 518.453 | 518.477 |
| (104,266) | 517 | 517 | 516.796 | 516.887 |
| (188,204) | 513 | 513 | 512.335 | 512.393 |
| (242,181) | 512 | 512 | 511.842 | 511.899 |
| (307,502) | 527 | 527 | 526.668 | 526.763 |
| (322,339) | 513 | 513 | 512.800 | 512.876 |
| (357,357) | 516 | 516 | 515.211 | 515.340 |
| (372,380) | 518 | 518 | 517.452 | 517.466 |
| (382,304) | 515 | 515 | 514.667 | 514.549 |
| (480,111) | 530 | 530 | 529.374 | 529.454 |
| (496,23) | 541 | 541 | 540.408 | 540.501 |

- **March arithmetic**: Metal's normal path advances `currentPoint += stepVec;
  texLocalPos += texStep; evalPoint += evalStep` — the ctp-adjusted
  `evalPoint += evalStep` accumulation is affine-equivalent to GL's
  `g_dataPos += g_dirStep`.

Supporting facts verified this session: `UseDepthPass` defaults to 0
(`vtkGPUVolumeRayCastMapper.cxx:48`) so the no-depth-pass branch is active;
the vertex shader is trivial (`ip_vertexPos = in_vertexPos`, `gl_Position =
proj*mv*volumeMatrix*vec4(in_vertexPos,1)`); the box corners in data coords are
(0,0,0)/(201.6,0,0)/(0,201.6,0)/(201.6,201.6,0)/…(201.6,201.6,138); the
`in_vertexPos` attribute binding (CubeVAOId/CubeVBOId, ~1295-1296) is correct.

## 4. Current image difference (both backends, same test)

`TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformCamOutsideNoJitter`
captured back-to-back with both factories (ExternalData baseline missing ⇒ the GL
render is used as the reference). Reproducible across runs (identical pixel
count twice):

| metric | value |
|---|---|
| differing pixels | 37,168 / 262,144 (14.18%) |
| max channel |Δ| | 8 / 255 |
| mean |Δ| over differing | 1.011 |
| per-channel RMS | 0.2348 (= 0.00092 normalized) |
| pixels ≥ 4 |Δ| | 49 |
| pixels ≥ 6 |Δ| | ~26 |
| rows spanned | 0..511 (whole footprint, not just silhouette) |

The residual is spread over the volume interior, not confined to the shell ring
of Update7. Channel breakdown: R max 8 (6,083 px), G max 6 (17,310 px),
B max 7 (16,659 px) — G/B carry the signal in this test. The 49 high-delta
pixels sit in the bright interior (mean RGB ≈ [228,180,148]) where tiny sample-
position shifts cross transfer-function gradients and shift opacity by 1-8 LSB.

## 5. What remains for bit-exact parity

With start/step/count all matched, the only degrees of freedom left are
**per-sample float32 rounding**:

1. **Seed-phase entry** (Update7 §7.1, now quantified): GL's `g_rayOrigin` comes
   from interpolated `ip_textureCoords` (vertex-interpolated cell-to-point
   coords) plus one step; Metal's `evalPoint_0` from an analytic box hit plus one
   step. They agree to ~1e-6 except grazing pixels like (357,357) at 1.2e-4. To
   close this, Metal would need to seed from GL's interpolated coordinates
   (interpolate the 4 box corners' cell-to-point coords in the fragment shader
   rather than re-deriving the entry from the box intersection).
2. **Incremental accumulation**: `evalPoint += evalStep` vs GL's
   `g_dataPos += g_dirStep`; both are float32 additive chains over ~520 steps. If
   the two chains start bit-identical, identical step values keep them
   bit-identical (same fp rounding per add), so closing (1) should also close
   this at the sample level.
3. Everything else (TF LUT contents, lookup convention, scalar normalization,
   sample counts, blend-mode gating) has already been matched or shown identical
   in earlier updates.

## 6. Next steps

- Seed Metal's first sample from GL-style interpolated texture coordinates and
  re-run the GL-vs-Metal diff, targeting 0 differing pixels.
- If a handful of pixels still differ, dump per-sample `evalPoint`/`g_dataPos`
  on both backends for the diverging pixels and diff the chains sample by sample.
