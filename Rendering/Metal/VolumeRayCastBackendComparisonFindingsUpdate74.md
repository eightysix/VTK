# Two GLSL-vs-Metal codegen divergences in the analytic pixel ray are now fixed: GL divides by w as rcp+mul and normalizes with an approximate rsqrt; dirObj/evalStep are now byte-identical, yet the image residual persists (183 -> 178 px) (update 74)

**Date:** 2026-08-10
**Status:** **milestone (two root causes fixed, residual not collapsed).** Update 73 §4
targeted making `evalStep` bit-identical to GL `g_dirStep`. Two previously
undocumented GLSL **codegen** behaviors were the actual 1-ulp sources:

1. **GL `X /= X.w` compiles to multiply-by-reciprocal** (`X * (1.0/X.w)`), not
   IEEE division. Metal used a true division.
2. **GLSL `normalize(d)` compiles to `d * inversesqrt(dot(d,d))`** where
   `inversesqrt` is the GPU's **approximate** reciprocal-sqrt instruction, which
   is 1 ulp above the correctly-rounded `metal::rsqrt` that Metal's `normalize`
   uses.

After both fixes, at the worst pixel `(397,110)` (Metal) == `(397,401)` (GL),
`nearP`, `farP`, `d`, `dirObj` and `evalStep` are **byte-identical** to GL. But
the image residual **did not collapse**: 183 -> 181 -> **178 px**, max channel
delta 8, worst pixel unchanged `(239,186,151)` vs GL `(239,179,143)` (ΔG=-7,
ΔB=-8). The per-sample evidence shows the remaining divergence is the same
single i=132 nearest-texel flip of update 73 — and now **GL itself flips that
sample across its own 6 frames** (3 frames texel 177, 3 frames texel 178),
Metal consistently picks texel 178 and matches GL's texel-178 frames
bit-for-bit. Doubts in §5.

## 1. Divergence 1: `X /= X.w` — GL is rcp+mul, Metal was IEEE division

Update 71 §2.4 reported `farP.x` 1 ulp apart (Metal 90.3353195 vs GL
90.3353271) while `nearP` was bit-identical. Decomposing GL's debug dump at
`(140,505)` (`nj_gl_rcp.log`, same camera/step as the Metal pair):

```
GL farPRaw = (0x3ef05400, 0x3f1d9000, 0x3fad7700, w=0x3baa4400)   (pre-division mat-vec)
GL rcpF    = 0x434073c6  = 192.45223999023438 = float32(1.0/w)    (correctly rounded)
GL farP.x  = 0x42b4abb0  = 90.3353271484375 == raw.x * rcpF       (bit-exact)
np.f32 division raw.x/w  = 0x42b4abaf  = 90.33531951904297        (1 ulp lower)
```

The Metal shader computed `farP /= farP.w` as an IEEE division (0x42b4abaf);
GL's compiled GLSL multiplies the raw result by a precomputed reciprocal
(0x42b4abb0). The nearP match was coincidence: `raw*rcp` and `raw/w` happen to
round identically for nearP's operands but differ for farP's.

**Fix** (MetalShaders.metal, both analytic unproject sites): replicate GL's
rcp+mul exactly:

```metal
nearP *= (1.0f / nearP.w);   // was nearP /= nearP.w
farP  *= (1.0f / farP.w);    // was farP  /= farP.w
```

Same for `reconstructRayDir` (wn/wf, lines 3640-3641). Verified post-fix:
Metal `farBits=421ffc3e42c3a2b04378310e` == GL `farP0=421ffc3e ...` at
`(397,401)`. **farP is now byte-identical.**

## 2. Divergence 2: `normalize()` — GL is approximate rsqrt, Metal was IEEE

With farP fixed, `d = farP.xyz - nearP.xyz` became byte-identical to GL
(`dBits=c2784162c0899ba0433a7169` == GL `dir0=c2784162...`), but `dirObj` was
still 1 ulp low (`dirBits=bea1ac5f...` vs GL `rd0=bea1ac60`). Decomposing GL's
debug fields at `(397,401)`:

```
d2 = dot(d,d) = 0x4716e769   (Metal dot also = 0x4716e769, both orders agree)
GL inv = inversesqrt(d2) = 0x3ba6b788  (GPU approximate rsqrt, 1 ulp HIGH)
IEEE  float32(1/sqrt(d2)) = 0x3ba6b787 (Metal normalize's rsqrt)
rd.x  = d.x * inv          = 0xbea1ac60 == GL   (d.x * 0x3ba6b787 = 0xbea1ac5f = old Metal)
```

GLSL `normalize(d)` lowers to `d * inversesqrt(dot(d,d))` and GLSL
`inversesqrt` maps to the hardware approximate reciprocal-sqrt (documented
~1-ulp error, here +1 ulp). Metal's `normalize` uses the correctly-rounded
`metal::rsqrt`.

**Fix** (MetalShaders.metal:4075): replace `normalize(d)` with GL's explicit
form using the fast (approximate) instruction:

```metal
dirObj = d * fast::rsqrt(dot(d, d));
```

Verified post-fix at `(397,110)`:
`dirBits=bea1ac60bcb33b223f72d668` == GL `rd0=bea1ac60`, and
`evalStepBits=b9dd56a9b7f560333af2d668` == GL `step0=b9dd56a9`.
**dirObj and evalStep are now byte-identical to GL.**

## 3. Image result: 183 -> 181 -> 178 px, worst pixel unchanged

| state | vs clean GL |
|---|---|
| before (update 73) | 183 px, max Δ 8 |
| after rcp+mul fix | 181 px, max Δ 8 |
| after + fast::rsqrt fix | **178 px, max Δ 8** |

Worst pixel `(397,110)` is still `(239,186,151)` vs GL `(239,179,143)`, Δ =
(0,-7,-8). So the update-73 hypothesis (make evalStep byte-identical -> the
texel-boundary flip disappears) is **falsified at the worst pixel**: evalStep IS
now byte-identical and the flip persists.

## 4. Per-sample at i=132 after the fixes: Metal matches GL's texel-178 frames

`DEBUG SAMPLE` (Metal) and `GL_SAMPLE` (GL, `VTK_GL_SAMPLE_DUMP=1
VTK_GL_SAMPLE_DUMP_PX=397,401`) at i=132:

```
GL   (397,401): 3 frames  raw=0.0163272  op=0.127265  (texel 177)  pos.z=0.695312
                3 frames  raw=0.0174868  op=0.371981  (texel 178)  pos.z=0.695313
Metal(397,110): all frames raw=0.0174870  op=0.371981  (texel 178)  eval.z=0.695313
```

- Metal's sample now equals GL's texel-178 frames **to every printed digit**
  (raw 0.0174870 vs 0.0174868 at 7dp, op 0.371981 == 0.371981).
- **GL itself is not frame-invariant at this sample**: 3 frames pick texel 177,
  3 pick texel 178 — the update-18/19 W2IF frame 3->4 view-angle perturbation
  (30 -> 30.0000008 degrees) crosses the 178/256 nearest boundary. Metal is
  deterministic and sits on the texel-178 side, matching GL's perturbed frames.
- Metal eval.z (0.695313) == GL pos.z for the texel-178 frames (0.695313) to
  print precision; GL's texel-177 frames are at 0.695312.

So the remaining image delta at this pixel is consistent with **Metal == GL's
texel-178 (perturbed) frame while the compared GL image holds its texel-177
value** — i.e. the residual may now be entirely the frame-selection / W2IF
perturbation artifact, not an arithmetic divergence. This requires a
frame-aligned image comparison to confirm.

## 5. Doubts / open questions

1. **Frame alignment.** The 6-frame dump shows GL flips i=132 between frames;
   which GL frame's image does the regression comparison actually use, and is
   Metal's deterministic texel-178 output comparable to any single GL frame?
   Need a frame-1..6 image capture on each backend (updates 18-20 had this;
   not re-run here) and a per-frame diff.
2. **`tex` vs `eval` in the SAMPLE dump.** Metal logs `tex=(..., 0.695793)` and
   `eval=(..., 0.695313)` at i=132 — the two lattices differ by ~0.0005 in z
   (~0.13 texel). Which one feeds the actual `sampleVolumeScalar` texture
   coordinate? If Metal samples at `texLocalPos` while GL samples at
   `g_dataPos` (the eval lattice), the nearest-texel pick could legitimately
   differ at boundaries even with byte-identical `evalStep`. Check the sample
   call site; update 69 switched the *loop advance* to evalStep but the sample
   coordinate may still be texLocalPos.
3. **Why did the image improve 183 -> 178 but not collapse.** Three pixels
   flipped in the right direction under these two fixes, but the knife-edge
   family (update 70) did not move; the mechanism at the remaining pixels may
   be the tex/eval lattice split of doubt 2 rather than the ray/step chain now
   known to be byte-identical.
4. **`normalize` on the legacy (non-analytic) paths.** Only the analytic
   camera-inside branch (anchorIsData && perspective) was changed to
   `fast::rsqrt`. The else-branch `normalize(p.rayDir * boundsSize)` (line
   4079) and `reconstructRayDir`'s `normalize` (line 3642) still use the
   correctly-rounded rsqrt. If GL reaches those paths via `computeRayDirection`
   with the same approximate inversesqrt, they may carry a 1-ulp dirObj that
   the analytic path no longer has. (Reference test uses the analytic branch,
   so this did not affect the 178-px field.)
5. **rcp+mul is a codegen property, not a language guarantee.** On a different
   GL driver/GPU the `/w` may lower to true division and the approximate rsqrt
   ulp sign may flip; the fixes hard-code the current macOS GL behavior.

## 6. Files / commands

- Fixes: `Rendering/Metal/Shaders/MetalShaders.metal` — `*= (1.0f / w)` at
  lines 3640-3641 and 4069/4071; `d * fast::rsqrt(dot(d,d))` at line 4075.
- Build: `ninja -C build_macos_metal vtkRenderingVolumeCxxTests`
- Capture (update 72 §3 with these adds):
  - Metal STEP/SAMPLE: `MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=33554432
    MTL_LOG_TO_STDERR=1` + `RenderingBackend=Metal` -> `/tmp/bc/u74c_metal.log`
  - GL RAY dump: `VTK_GL_RAY_DUMP=1` + `RenderingBackend=OpenGL` -> `/tmp/bc/u74_gl.log`
  - GL per-sample: `VTK_GL_RAY_DUMP=1 VTK_GL_SAMPLE_DUMP=1
    VTK_GL_SAMPLE_DUMP_PX=397,401` -> `/tmp/bc/u74_gls.log`
- Images: `build_macos_metal/Testing/Temporary/{glA_u74,mtA_u74b,mtA_u74c}.png`;
  diff 178 px / max Δ 8 (worst `(397,110)` (0,-7,-8)).
- Key reproduced values:
  - GL farP.x = raw.x*rcpF = 90.3353271484375 (0x42b4abb0); division = 0x42b4abaf.
  - GL inv = inversesqrt(0x4716e769) = 0x3ba6b788; IEEE rsqrt = 0x3ba6b787.
  - Post-fix Metal dirObj = GL rd = 0xbea1ac60bcb33b223f72d668;
    evalStep = GL step = 0xb9dd56a9b7f560333af2d668 at (397,110).
