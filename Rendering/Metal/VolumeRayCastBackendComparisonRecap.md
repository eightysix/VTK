# Metal ↔ GL GPU Volume Raycast — Full Recap (updates 1–84)

**Goal:** Metal output **bit-identical** to **clean GL** (reference, `RenderingBackend=OpenGL`, no debug injection) on `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter` (512², camera-inside, NEAREST interpolation, 6-frame animated camera).

**Current status:** **178 px differ / 262,144 (0.068 %), max channel delta 8** at the worst pair `(397,110)` ↔ `(397,401)`. All shader-visible inputs are bit-identical — **including per-vertex `texcoord` on all three axes (94/94, update 80)** — and the residual is a ~1–2 ulp offset in the **rasterizer-interpolated** ray anchor (Metal vs GL), causing nearest-texel selection flips at grid-aligned (knife-edge) rays. The decisive analytic-anchor experiment was run and is **conclusively negative** (mode-1 429 px / mode-2 469 px, update 78); the frame-aligned raw capture refutes frame-selection (update 79); the per-vertex z-input lever is refuted (update 80). The 178 px is therefore **bounded below by the GL-driver vs Metal interpolator arithmetic difference at knife-edge picks**, with every input proven bit-identical. The single remaining lever is the full-gate interpolation-floor quantification; if it confirms the floor, accept 178 px as the irreducible hardware bound and switch the acceptance metric to the frame-aligned `VTK_STEP_RAW_CAPTURE` diff.

Two further structural fixes landed since: **update 82** densified the camera-outside proxy box to GL's exact `vtkDensifyPolyData(2)` centroid-fan geometry (uploaded in dataset space), cutting CamOutside 1384 → **250 px** (the remainder is the same interpolator floor on the larger box triangles); **update 84** found and fixed the RenderToImage pipeline blending its shader output over the cleared white RTT background with `ONE, ONE_MINUS_SRC_ALPHA` (GL's RTT pass is unblended) — on `TestGPURayCastRenderToTexture` that injected `(1−α)·255` into every RTT pixel (47,878-px / max_d 65 final-frame diff), collapsing to **1,471 px / max_d 1** after disabling the blend. The remaining 1,471 px are all ±1 rounding on the head contour — the same interpolator floor, amplified by LINEAR + ShadeOn + the sharp TF ramp.

---

## Baseline results (commit `8bb5c370b6`, 2026-08-11)

Full pixel-diff suite via `BackendComparisonTools/run_pixel_diff_suite.sh` (both backends, 512², checkerboard dummy baseline so the `-V` regression always fails and dumps; frame-aligned captures via `VTK_STEP_RAW_CAPTURE`). Metrics: `diff` px = any-channel \|Δ\|≥1, `\|Δ\|≥2` = max-channel ≥2, `\|Δ\|≥5` = max-channel ≥5, `max_d` = max channel delta.

| test / config | diff px | \|Δ\|≥2 | \|Δ\|≥5 | max_d |
|---|---|---|---|---|
| Reference (jitter on) | 178 | 14 | 1 | 8 |
| NoJitter (acceptance test) | 178 | 14 | 1 | 8 |
| StepTF (no env, no wheel) | 0 | 0 | 0 | 0 |
| CamOutside | 1509 | 191 | 31 | 8 |
| CamOutsideFixedStep | 1509 | 191 | 31 | 8 |
| CamOutsideNoJitter | 1509 | 191 | 31 | 8 |
| FineStep | 178 | 14 | 1 | 8 |
| FlatTF | 0 | 0 | 0 | 0 |
| Linear | 343 | 0 | 0 | 1 |
| MaxIP | 6 | 6 | 4 | 14 |
| Nearest | 178 | 14 | 1 | 8 |
| NearPlaneTiny | 178 | 14 | 1 | 8 |
| StepTF m0 (both-step) | 0 | 0 | 0 | 0 |
| StepTF m1 (color-step) | 0 | 0 | 0 | 0 |
| StepTF m2 (opacity-step) | 5 | 0 | 0 | 1 |
| StepTF m3a (ramp 0.005) | 6 | 0 | 0 | 1 |
| StepTF m3b (ramp 0.02) | 16 | 0 | 0 | 1 |
| StepTF m3c (ramp 0.1) | 42 | 0 | 0 | 1 |
| StepTF m4 (color-ramp) | 1 | 0 | 0 | 1 |
| StepTF m5 (win-ramp) | 181 | 0 | 0 | 1 |
| StepTF B (constant-scalar) | 3 | 0 | 0 | 1 |
| StepTF D256 | 25 | 0 | 0 | 1 |
| StepTF D64 (raw 64³) | 19 | 0 | 0 | 1 |
| StepTF E (axis camera) | 16 | 0 | 0 | 1 |
| StepTF m0+linear | 0 | 0 | 0 | 0 |
| StepTF m2+linear | 5 | 0 | 0 | 1 |

Notes:
- The acceptance test (NoJitter) and the reference reproduce the documented **178 px / max Δ 8** exactly; all runs deterministic.
- The `\|Δ\|≥5` column is 1 for most residual cases (the single known knife-edge worst pair) except CamOutside\* (31) and MaxIP (4, max_d 14).
- Run twice per backend and compared byte-identical (`RUNS=2` determinism mode) for the NoJitter/FlatTF quick gate.
- Reproduction: `WORK=/tmp/bc/pixdiff_base ./BackendComparisonTools/run_pixel_diff_suite.sh` (raw captures + `summary.txt` left in `$WORK`).
- **Stale rows (superseded):** the CamOutside\* rows above (1509 px) predate update 82; the camera-outside proxy is now GL's exact densified geometry, so the current CamOutside value is **250 px / max_d 8** (update 82, see §4c). The suite table is retained as the 2026-08-11 baseline.

---

## 1. What is bit-identical (proven closed)

| Item | Evidence |
|---|---|
| Composite accumulation (fma chain) | updates 51/53/73: 68-gated-pixel sweeps + full per-sample float32 replay reproduce both backends' finals exactly |
| Background blend (`ONE, ONE_MINUS_SRC_ALPHA`, bg `26/255`) | update 56/57: root cause was Metal clamping `alpha=1` at the opacity break, zeroing GL's blend term; removing it collapsed 63,692 → 188 px |
| TF tables (1024×float32 RGBA) | updates 52: 0/1024 rgb diff after the pre-integration `float(minWorldSpacing)` cast fix; alpha 190→0 ulps |
| Composite gate / termination threshold | update 1: `0.001h` gate and `0.99h` break fixed to `alpha>0` / `1−1/255` strict |
| fp16 stalls / MAX_RAY_STEPS clamp | update 2: half→float everywhere; maxSteps from ray geometry |
| Scalar w/l normalization, per-component ranges, float volume storage | updates 6, 9, 14, 15 |
| Gradient-opacity LUT | update 3: 256×8-bit → 1024×float32 |
| Analytic ray (`inversePVM`, `nearP`/`farP`, `dirObj`) | updates 40, 71, 74: wrong product order fixed in both backends; `/=w`→rcp+mul; `normalize`→approx rsqrt; nearP/dirObj/evalStep byte-identical |
| March sample-count lattice (`g_dirStep`-lattice bounds exit, maxSteps, tTerminateMax) | updates 68–70: constant-scalar volume 5530→3 px |
| Per-vertex clip (x/y/w), `modelPos`, and per-vertex texcoord (x/y/z) | updates 36, 60, 76, 80: 94/94 bit-identical on all axes; z included (update 80) |
| Pixel center (GL `gl_FragCoord` == Metal `in.position.xy`) | update 60: exact `(397.5, 110.5)` |
| Proxy cap mesh (verts/indices), V/M matrices, `in_volume_scale`, sample distance | updates 27–29, 33, 50: byte-identical |
| GLSL↔MSL mat-mul FMA contraction | update 36: `(P*V)*W`, `mul,fma,fma,mul+add`, `fastMathEnabled=NO` — collapsed 307→130 px |
| Step (`evalStep` == `g_dirStep`), `dirObj` 114/114, dirObj lattice | updates 17, 59, 69, 74, 76: bit-identical |

## 2. The residual and its proof of origin (updates 75–80)

- The 188→183→178 px residual is **nearest-texel selection flips** at texel boundaries (e.g. `z=0.6953125 = 178/256`), driven by a **~1–2 ulp difference in the interpolated anchor texcoord** (Metal systematically at-or-above GL: state A y+1, state B x+1/y+1/z+2 ulp). Color deltas up to 8 u8 with negligible alpha deltas. `evalStep` is byte-identical — the ulp is not in the step.
- **Per-vertex data exonerated — now on all three axes:** per-vertex clip x/y/w, `modelPos`, and texcoord x/y/z are 94/94 bit-identical (updates 60, 76, 80). The +1 ulp is created **inside the rasterizer interpolator**: the update-76 §4 f64 perspective-correct interpolation of the bit-identical inputs reproduces GL's frame-A exactly (`0x3f01aa39`) while Metal logs `0x3f01aa3a`.
- **Update 78 reconciles the f64-vs-f32 contradiction:** the "exact" f64 value is *frame-A*; GL's own interpolated texcoord is not frame-invariant (frame-A `0x3ee5d6d4` vs frame-B `0x3ee5d6d0`), and the earlier +2..+61-ulp readings were a dump artifact (multi-row per-pixel records; reading the "last row" silently picked frame-B on some pixels).
- **Update 79 exonerates frame-selection:** a frame-aligned raw front-buffer capture (`VTK_STEP_RAW_CAPTURE`) at the identical W2IF-perturbed camera shows Metal(raw) vs GL(raw) = the same 178 px, with 0 px self-drift on each backend and identical jitter/NoJitter residual sets. No GL frame choice would fix it.
- **The rasterizers do not evaluate at a discoverable sample point:** implied sample point == exact pixel center for both backends; fitted sub-pixel offsets reduce but never reach 0 and differ per backend (updates 61–64). Effective GL-beyond-Metal sample displacement ~(+0.025, −0.032) px with all scatter explained by ±1–3 ulp f32 amplification through triangle 122's near-degenerate texcoord space (update 64).
- **No analytic pixel-center model reproduces either backend at 0 ulps** — affine / persp `1/w` / `rcp(1/w)` / inverse-w (update 63), and the update-80 sweep (f64 weights and shader-exact f32 chain, best 8/303 vs Metal, 0/28 vs GL).

## 3. The decisive lead — analytic-anchor — is closed negative; residual pinned to the interpolator (updates 78–80)

- **Update 78 — analytic-anchor implemented, both modes lose.** `VTK_METAL_ANALYTIC_ANCHOR` mode-1 (f32 weights) → **429 px**, mode-2 (f64 weights) → **469 px**, vs the 178-px mode-0 baseline (mode-0 unchanged, so the worsening is purely the anchor swap). At the worst pixel `(397,110)`↔`(397,401)`: the f64 reconstruction matches GL frame-A exactly on x/y (`0x3f0184b9`,`0x3f01aa39`) but is **+1 ulp above frame-A on z** (`0x3ee5d6d5`); z drives the i=132 texel pick, so it tips knife-edge pixels the wrong way. Metal's hardware interpolator already sits exactly at the **midpoint of GL's frame-A/B spread** on all three axes — the best deterministic value — and no deterministic anchor can track GL's frame-dependent quantity.
- **Update 79 — frame-aligned comparison refutes pure frame-selection.** Raw front-buffer capture on the reference test and its NoJitter variant: Metal(raw) vs GL(raw) = **178 px** for the identical W2IF-perturbed camera; GL raw==GL w2if (0 px) and Metal raw==Metal w2if (0 px). The compared GL frame already differs from Metal at the same 178 px, so the residual is a genuine same-camera backend divergence at knife-edge texel picks, not a frame-selection artifact.
- **Update 80 — per-vertex z-input lever refuted.** The near-cap z per-vertex texcoord (the axis driving the i=132 texel pick) is bit-identical 94/94, so the ±1-ulp anchor-z offset is produced **inside the rasterizer interpolator**. A fresh interpolation-model sweep finds no offline f32/f64 barycentric/perspective model reproduces either backend (best 8/303 vs Metal, 0/28 vs GL), consistent with updates 61–63.
- **Remaining lever:** the full-gate interpolation-floor quantification (per-axis ulp histogram over the full 8237-px STEP gate, frame-matched, update 80 §5.2). If it confirms the floor: accept 178 px as the hardware floor and switch the acceptance metric to the frame-aligned `VTK_STEP_RAW_CAPTURE` diff (update 79 §5), documenting the residual as irreducible.

## 4. Secondary / parallel leads

1. **W2IF frame perturbation / frame alignment (update 19/74/78/79):** GL *itself* flips the i=132 texel between its frame groups (3× texel 177, 3× 178) because `vtkWindowToImageFilter` perturbs the view angle 30 → 30.0000008° at frame 3→4, and the perturbed-camera interpolated anchor moves 1–2 ulp (update 78 frame-A/B spread). **Resolved by update 79:** within the perturbed set GL is deterministic and Metal still differs at the same 178 px, so this is *not* a frame-selection artifact — but the `-V` W2IF path inherits the frame-3→4 selection choice and can never be the bit-identity metric; use the frame-aligned `VTK_STEP_RAW_CAPTURE` diff instead.
2. **Finetuned-test matrix residues (updates 65–70):** C (window-limited ramp) 187 px, D256 4338 px, D64 10281 px (coarse-volume, ±2 LSB), E (axis-camera) 60 px — these mostly share the same per-sample mechanism; B (constant-scalar) is 3 px, m2 21 px. Not blockers for the reference test but relevant if the suite is re-run.
3. **Legacy paths:** camera-outside loops, label-map gradient-opacity parity, offscreen RGBA16F accumulation for downsampled — out of scope for the reference test, structural parity only. RenderToImage is no longer a blind spot: `TestGPURayCastRenderToTexture` exposed the blended-RTT defect, now fixed (update 84, §4e).

## 4b. Update 81 — StepTF m4 1-px diff at `(290,330)`: a byte-rounding-boundary coincidence, same interpolator floor

The StepTF m4 config (`VTK_STEP_MODE=4 VTK_STEP_WHEEL=1`: color linear ramp, constant opacity 0.005, raw-capture metric) leaves exactly **1 px** GL-vs-Metal diff: `(290,330)` GL=45 / Metal=44, max_d 1, reproducible (3/3 runs, dump and capture both stable). This is the same irreducible floor as the reference residual, but surfacing through a **byte-rounding boundary** instead of a nearest-texel pick:

- **Inputs bit-identical:** clean-GL `VTK_GL_FLOAT_DUMP` readback at `(290,181)` = `accCol 0.106299959, accOp 0.331048012`; Metal exact `%.9g` FINAL log (precise-print rebuild) = `accCol 0.106299959` (**bit-identical**), `accOp 0.331048042` (9th significant digit only).
- **Blend formula confirmed for BOTH backends:** the float model `src + (1−a)·26/255` reproduces each backend's *own* capture at 99.997 % (GL 7/262144, Metal 3/170633 mismatches, all 1 LSB at rounding boundaries). Ruled out as wrong models: 8-bit-quantized src/alpha blend variants (5.6k–64k mismatches). So bg `26/255` and the `ONE, ONE_MINUS_SRC_ALPHA` blend are exactly right (update 56/57 holds).
- **The value sits on the round threshold:** model composite at `(290,330)` = **44.4992**, 0.0008 px below the 44.5 round-up byte boundary. Neighbors in the same column are brighter (`accCol` 0.106309/0.106331 → 44.5016/44.5095) and **both** backends render them 45 — `(290,330)` is a 1-px-wide accCol dip exactly at the boundary.
- **Why the flip is not a bug:** for GL to store 45 its real composite must be ≥44.5, i.e. its real `accCol ≥ ~0.106303` (≈3e-6 = one knife-edge sample's color contribution) or its 8-bit store rounds up — while GL's own RGBA32F re-render dump reads 0.106299959. GL's real render therefore diverges from its dump re-render at this single knife-edge pixel — the same GL-internal interpolator bistability as update 78's frame-A/B ±1–2 ulp anchor spread, here translated by the composite's byte-rounding threshold. The 3e-8 `accOp` log difference is the same floor's fingerprint in the per-sample TF-opacity lookup (color accumulation still rounds to the same 0.106299959).
- **Matrix-wide consistency:** every StepTF matrix residue has max_d = 1 (baseline table), i.e. all are 1-LSB — consistent with this rounding/knife-edge class, not a magnitude-8 texel-pick flip like the reference test's 178 px. Accept as hardware floor; the frame-aligned `VTK_STEP_RAW_CAPTURE` metric (update 79/80) already measures it.

## 4c. Update 82 — CamOutside 1384 → 250: densify the camera-outside proxy box to GL's exact `vtkDensifyPolyData(2)` geometry, in dataset space

GL renders its camera-outside proxy box **densified**: `vtkOpenGLGPUVolumeRayCastMapper.cxx` runs the box through `vtkDensifyPolyData` with `SetNumberOfSubdivisions(2)`, a **centroid fan** that splits each of the 12 box triangles into 3 twice — 108 triangles / 56 vertices — uploaded as float32, while Metal used a coarse 12-triangle box with a different face triangulation. Over the large triangles the Metal interpolator rounded the ray anchor up to ~28 ulps off GL's small-triangle value, and with NEAREST + the sharp scalar-500 TF step every shift flipped knife-edge samples (1384 px).

**Fix (two steps, both required):**
1. **Densify to GL's exact geometry** — the camera-outside box is built as a `vtkPolyData` with GL's 8-corner `DataGeometry` order `{000,100,010,110,001,101,011,111}`, GL's `tris[36]` set, and GL's 0-2-1 winding, then run through the identical `vtkDensifyPolyData(2)` — byte-identical to GL's `BBoxPolyData` VBO (108 tris, float32 positions + uint32 indices).
2. **Upload in dataset space, not the unit cube** — the old `boundsMin + in.position * size` vertex math double-rounded each centroid ~1 ulp off GL's single `float32(double centroid)`; the box now uploads model-space positions exactly like GL (new `useDataSpaceBoxVertices` uniform, `_padAnalyticAnchor[0]`), with `vertex_volume_main` forwarding `in.position` unchanged. `localPos` stays normalized [0,1] so fragment semantics are unchanged.

| Test | before | after |
|---|---|---|
| CamOutside (+FixedStep, +NoJitter) | 1384 | **250** |
| Reference / FineStep / Nearest / NearPlaneTiny | 178 | 178 |
| FlatTF / Linear / MaxIP / StepTF m0..m5lin | 0 / 343 / 6 / 0..181 | unchanged |

The 250 remaining pixels: 116 at \|d\|=1, 68 at \|d\|=2 (184/250 within one 8-bit LSB per channel), rest knife-edge clusters at tissue boundaries — the same ~1 ulp interpolator floor as the camera-inside cap, amplified because the densified box triangles are ~1/108 of the screen area (so their rounding is a little larger: 250 vs 178). With per-vertex values bit-identical, no densification can reduce it further: a different subdivision than GL would break the per-vertex inputs, and the interpolator hardware difference is irreducible at the geometry level.

## 4d. Update 83 — jitter noise parity: vtkJPEGReader decodes the blue-noise tile bottom-up, and GL samples it bottom-up too

The Metal shader's jitter used **Interleaved Gradient Noise (IGN)** while GL samples a real blue-noise texture, so every *jittered* render had a different sample-lattice phase per pixel. Only one test in the comparison family enables jitter — `TestGPURayCastCameraInsideNonUniformScaleTransform` (300², `SetUseJittering(1)`) — and it showed **45,840 px / max_d 139** Metal-vs-GL. (The 512² reference family never enables jitter — `vtkGPUVolumeRayCastMapper::UseJittering` defaults to 0 — so the "Reference (jitter on)" label in the baseline table is wrong; those runs are jitter-off and were never affected by this.)

- **Finding:** `vtkJPEGReader` writes its output rows **bottom-up** (`vtkJPEGReader.cxx`: `destLine = output_height - output_scanline` + reversed `row_pointers`), so its decode is the JPEG's vertical flip: verified 4096/4096 `vtk[y][x] == pil[63-y][x]` against PIL's decode of `Rendering/OpenGL2/textures/BlueNoiseTexture64x64.jpg`. GL's noise texture (`vtkOpenGLRenderWindow::GetNoiseTextureUnit`) is that decode, and the volume shader samples it at `gl_FragCoord.xy/64` (`vtkVolumeShaderComposer.h`) — row `py_gl` from the bottom, REPEAT/NEAREST.
- **Two flips compose:** vtkJPEGReader's row flip (texture row y = JPEG row 63−y) plus the GL-vs-Metal y-origin flip (`py_gl = H−1−py_mt`). For a Metal top-down pixel `py_mt`, GL samples **JPEG row `(py_mt − H) mod 64`**. This collapses to `py_mt mod 64` only when `H` is a multiple of 64 (why a 512² or 256² window looks right and a 300² window does not).
- **Fix:** `MetalShaders.metal` now embeds the exact 64² luminance tile in the JPEG's top-down orientation (`kBlueNoise64[4096]`, PIL/libjpeg byte order, verified equal to PIL's decode) and samples `kBlueNoise64[(floor(st.y) − H) mod 64][floor(st.x) mod 64]` in all three jitter call sites (fullscreen, RTT, grid-traversal — the last also dropped its old `+0.5` half-pixel shift). The flip formula is verified numerically to equal GL's `texture2D(in_noiseSampler, gl_FragCoord.xy/64)` at **every pixel for every viewport height** (262144/262144 @512, 4225/4225 @65, 576/576 @24, …).
- **A/B on `TestGPURayCastCameraInsideNonUniformScaleTransformKnobs` (300², raw capture, frame-aligned):**

  | config | IGN (pre-fix) | blue-noise+flip (post-fix) | IGN delta contribution |
  |---|---|---|---|
  | default (jitter on) | 45,840 px / max_d 139 | **29 px / max_d 13** | **45,811 px (99.94 %)** |
  | VTK_NUS_POKE=0 (no poke matrix) | 33 px / max_d 94 | **0 px** (bit-identical) | 33 px |
  | VTK_NUS_JITTER=0 | 31 px / max_d 49 | 31 px / max_d 49 | 0 px |

  `nopoke → 0 px` proves that with the poke matrix removed, the whole jittered pipeline is now bit-identical — the 29-px poke residual is the same interpolator floor as the 512² reference, not noise. The **IGN delta contribution** (pre-fix − post-fix px = the pixels that were caused purely by the IGN noise mismatch) is **99.94 % of the pre-fix diff** (45,811/45,840) on the reference config; only 0.06 % (29 px) is the shared interpolator floor, and the contribution is 0 px with jitter off.
- **Framing correction:** the recap's "Reference (jitter on)" label applies to no test in the family (the only jittered test is the 300² non-uniform one); the 512² reference and NoJitter rows are both jitter-off, which is why update 83 leaves them at exactly 178/178/0.

## 4e. Update 84 — RenderToImage was blended over the white clear; GL's RTT pass is unblended (structural RTT bug, fixed)

`TestGPURayCastRenderToTexture` (401×399, headsq/quarter 64³ CT, LINEAR, `ShadeOn`, opacity ramp 0→0.15 at scalar 900) was the first suite test exercising `RenderToImage`, and it exposed a **structural RTT bug** distinct from the interpolator floor: the Metal `RenderToImage` pipeline blended its shader output over the cleared white RTT background with `ONE, ONE_MINUS_SRC_ALPHA`, while OpenGL's RenderToImage pass writes the **raw (unblended)** shader output. Pre-fix: final-frame diff **47,878 px / max_d 65 (29.9 %)**, and the raw RTT color texture differed on 144,852 px / max_d 255 **while its alpha field was essentially bit-identical** (61 px, max_d 1).

- **Per-pixel proof:** both backends composite the same (the recap's composite/accumulation is bit-identical, and the RTT alpha channel matches to 61 px), so with `GL_RTT = src` and `Metal_RTT = src + (1−α)·255`, every drawn RTT pixel satisfies `Metal_RGB == GL_RGB + (1−α)·255` to ≤1 u8 — e.g. at (138,23) α=111, GL=(72,51,28), Metal=(216,195,172), Δ=(144,144,144) = (1−111/255)·255. The max RTT Δ of 255 sits in the low-α contour shell; the opaque interior (α≥200) differs by 4,919 px with |Δ|≥2 (max 56). The final-frame residual is that injected white halo surfacing at the volume contour via the image actor's linear filtering.
- **Fix:** disabled blending on the Metal `RenderToImage` pipeline color attachment 0 (`blendingEnabled = NO`, `GetOrCreateVolumePipeline`'s `RenderToImage` branch), matching the OpenGL RTT framebuffer (unblended write over `vtkglClearColor(1.0,1.0,1.0,0.0)`).
- **Post-fix (2026-08-12, commit `fe81a78ebb`):** final-frame diff **1,471 px (0.9 %) / max_d 1** — all ±1 rounding (687 R / 535 G / 224 B single-channel flips, few 2-channel), no |Δ|≥2, confined to the head-projection bbox (x 65–336, y 79–310). That is the predicted knife-edge/rounding floor, amplified by LINEAR + ShadeOn + the sharp TF ramp on coarse 64³ data — the same irreducible mechanism as the reference family's 178 px. Both backends byte-deterministic across runs. The `VTK_RTT_DUMP` capture hook used for the diagnosis lives in the stash entry (`stash@{0}`, comment only); it is not part of the committed fix.

## 5. Methodology debt / landmines (do not repeat)

- **GL debug dumps corrupt the captured frame and the compiled arithmetic** — debug-GL is a *third* compile variant ≈ Metal, never a clean-GL reference (updates 22, 44, 59). Pair every GL dump with a separate clean capture.
- **Pixel pairing:** GL dumps are bottom-left origin → pair `GL(px, py) ↔ Metal(px, 511−py)`; the B-residual list bug (update 76) silently compared flipped pixels.
- **Frame alignment:** the GL float dump must be frame-aligned (per-frame overwrite, final = frame 6); camera animates 6 frames (update 59).
- **Stale binaries/images:** `-V` filename staleness (update 34), shared `-T` overwrite (update 13), stale-copy md5 mixups (update 65), `OpenGL2`≠`OpenGL` backend flag fallback (update 72).
- **`fastMathEnabled=NO` changed numerics library-wide at update 36** — all pre-36 fragment measurements were re-baselined.
- **os_log truncation (~213/563 chars), `%0.8g` lossy float print** — use `%.9g` / `setprecision(9)`, join multi-line records (updates 35, 71, 75).
- **zsh word-splitting of `$envs`** silently ran a garbage matrix (update 67); use `${=envs}`.
- **Multi-row per-pixel GL dumps:** GL logs several rows per pixel (one per frame) with distinct tex values; reading the "last row" silently picks frame-B on some pixels — this produced the spurious +2..+61-ulp "exact f64 reproduction" readings (update 78 §2) and the retracted "f32-model == Metal 28/28" claim from a stale log generation (update 80 §3). Join on pixel + frame, and only ever compare frame-aligned.
- z-texcoord per-vertex readback is an encoding artifact on GL's near-cap rows (`tex.z==tex.y`, `pos.z==vid`) — only the interpolated fragment value is meaningful (update 76); the clean newer GL dump matched `(z+w)/2` to ±1 ulp at 92/94 and `clip.z` is unused by the raycast (update 80 §2).

## 6. Instrumentation / debug scaffolding still in the tree (evaluate for cleanup)

Env-gated: `VTK_GL_RAY_DUMP`/`VTK_GL_VERTEX_DUMP`/`VTK_GL_SAMPLE_DUMP`/`VTK_GL_OPTABLE_DUMP`/`VTK_GL_FINAL_DUMP`/`VTK_GL_FLOAT_DUMP`/`VTK_GL_CAP_DUMP`/`VTK_GL_SWEEP_DUMP`/`VTK_GL_RESID_DUMP`, `VTK_METAL_FLOAT_DUMP`/`VTK_METAL_CLEAN_BIAS_F`/`VTK_METAL_ANCHOR_PERTURB`/`VTK_METAL_FULLSCREEN_CAMERA_INSIDE`/`VTK_METAL_ANALYTIC_ANCHOR`/`VTK_METAL_ANCHOR_REC_DUMP`, `VTK_STEP_RAW_CAPTURE` (front-buffer frame-aligned capture), plus the `DEBUG STEP`/`vertex_volume_main`/`fragment_volume_main` os_log blocks in `MetalShaders.metal`. None affect clean output (update 72: debug Metal == clean Metal, 0 px), but they should be removed or gated behind the test-build macro before production. Update 81 also bumped the `FINAL` log's `accOp`/`accCol` from `%f` to `%0.9g` (inside `VTK_METAL_ENABLE_LOGGING` only) to make the byte-boundary analysis possible — per the recap's own `%0.8g`/os_log advice (update 35/71/75).

## 7. Test / build commands

```
./macos_metal_build.sh --resume --tests
# run: -V $TMP/<unique>.png  --vtk-factory-prefer RenderingBackend=OpenGL (GL) / =Metal (Metal)
# acceptance metric (updates 79/80): VTK_STEP_RAW_CAPTURE=<out>.png on the NoJitter test, diff Metal(raw) vs GL(raw) — frame-aligned, deterministic
```

## 8. Chained reference (in order, read these for depth)

Updates **1–19** (pipeline: gates, fp16, TF tables, half→float, NEAREST insight, W2IF) → **20–40** (proxy mesh, matrix/FMA parity, 307→130→694 px, inversePVM order) → **41–59** (composite exonerated, background-blend root cause 63,692→188 px, frame-aligned dumps, step vs anchor) → **60–64** (per-vertex bit-identity, barycentric weights ~1e-8, interpolation floor quantified) → **65–70** (finetuned matrix, sample-count lattice 5530→3) → **71–77** (ray byte-identity, rcp/rsqrt codegen, anchor +1 ulp proven, analytic-anchor audit) → **78** (analytic-anchor negative: mode-1 429 / mode-2 469 px; GL frame-A/B anchor spread) → **79** (frame-aligned raw capture refutes frame-selection: same-camera 178 px) → **80** (per-vertex texcoord z 94/94, z-input lever refuted, interpolator-model sweep negative; residual = interpolator hardware floor) → **81** (StepTF m4 1-px diff at `(290,330)` = byte-rounding-boundary coincidence at composite 44.4992; inputs/accumulation bit-identical, blend model confirmed per-backend; same interpolator floor) → **82** (CamOutside densify 1384→250: GL's vtkDensifyPolyData(2) centroid-fan geometry uploaded in dataset space) → **83** (jitter noise parity: vtkJPEGReader bottom-up decode composed with the GL/Metal y-origin flip; IGN replaced by the exact blue-noise tile; jittered 300² test 45,840→29 px, nopoke 0 px) → **84** (RenderToImage blended over the white clear while GL's RTT pass is unblended: Metal injected (1−α)·255 into every RTT pixel; disabling the blend on the RTT pipeline collapsed TestGPURayCastRenderToTexture from 47,878 px / max_d 65 to 1,471 px / max_d 1, the same interpolator floor).

---

## 9. Commit-by-commit changelog (`c6b8cc1fd5..HEAD`, code changes)

Every non-doc commit in the range, oldest first, grouped by the recap's phases. Pure documentation commits (findings/recap markdown), the analysis-tool persistence commits (the `BackendComparisonTools/updateNN/*.py` + README landings — 7b4a788f0c, 1d1fe1562e, 4c250a654d, c75fc18c1a, 73a1b52cab, 2b7e219c8d, 7a64609682, plus 8bb5c370b6's `run_pixel_diff_suite.sh`), and config-only tweaks (df97bd5336 `opencode.json`, 6bae321a2a parser) are omitted. All listed commits touch runtime code: the Metal/GL shaders, mappers, GL volume-mapper helpers, tests, or build scripts.

**Phase 0 — harness: shader logging, GL-as-reference, test scaffolding (updates 1–12)**

- `b1798cd6ad` feat(Metal): test-only Metal 3.2 shader logging for volume ray-cast — env-gated `os_log` STEP/MARCH dumps in `MetalShaders.metal` behind `VTK_METAL_ENABLE_LOGGING`, `TestMetalVolumeShaderLog`, plumbing in `vtkMetalGPUVolumeRayCastMapper.mm` + CMake.
- `f658845983` fix(OpenGL): register `RenderingBackend=OpenGL` attribute for the volume mappers (`vtkOpenGLGPUVolumeRayCastMapper`, `vtkOpenGLRayCastImageDisplayHelper`) — makes `--vtk-factory-prefer RenderingBackend=OpenGL` actually select GL, usable as the clean reference.
- `6767cb0493` fix(Metal): report 32-bit depth size — render-window depth pixel format matches GL's depth buffer for the camera-inside near-plane parity.
- `ec477e9df7` fix(Metal): clip the volume ray-cast march to the clipped exit point.
- `6a3b2c6849` test(Volume): camera-inside gradient-opacity divergence tests — `ConstGradOp` / `NoGradOp` / `NoShade` isolates.
- `b4a9087663` test(Volume): no-transform + camera-outside variants (`...NoTransform`, `...NoTransformCamOutside`).
- `c41f975d5e` test(Volume): sampling-artifact variants + sample-distance probes (`FineStep`, `NearPlaneTiny`, `SampleDist0_25`, `SampleDist0_5`).
- `a831b07ca2` test(Volume): fixed-step sweep probe for camera-outside (`...CamOutsideFixedStep`).
- `f9bc0103ab` test(Volume): nearest-interpolation and MaxIP variants.
- `fa52bf3773` fix(Metal): apply blend-mode-specific opacity correction to the TF tables, matching GL.
- `a4415d2329` fix(Metal): apply shading to every `alpha>0` sample like OpenGL.
- `f1ec8e6697` debug(Metal): gradient-opacity debug logging behind `VTK_METAL_ENABLE_LOGGING`.
- `1ce3977260` fix(Metal): bounds-check `MTL_FIRSTVALS` probe indices in GPURender.
- `bc36ecc468` test(Volume): NoShade gradient-opacity isolates + amplifier.
- `c022b1f24a` fix(Metal): align the volume composite gate and termination with OpenGL (`alpha>0` gate, strict break/termination thresholds).
- `e808ebe5f6` build(macos): preserve the test flag when resuming the build.

**Phase 1 — float32 sweep: half → float everywhere (updates 2–15)**

- `38343bdb4b` fix(Metal): accumulate the volume composite in full float (was half).
- `e66a38ff60` fix(Metal): remove the fixed `MAX_RAY_STEPS` clamp on the march.
- `86a45a729d` fix(Metal): build the gradient-opacity LUT as 1024×R32Float (GL parity).
- `f7c1c814c9` fix(Metal): compute volume gradients in full float.
- `1732d6bf25` fix(Metal): march with an integer step counter and GL `g_dirStep` arithmetic.
- `8c50593b8d` fix(Metal): promote the sample-distance uniform to full float32 (GL step parity).
- `6256696218` fix(Metal): compute scalar window/level normalization in full float32.
- `71b22c376a` docs(Metal): ray-geometry parity verification + no-jitter test + OpenGL ray dump.
- `8f308ce936` fix(Metal): promote transfer-function lookup to float32.
- `0994a59606` fix(Metal): promote gradient-opacity sampling to float32.
- `e82046f22e` debug(GL): extend the GL ray dump to (422,419).
- `d42c4aa545` debug(GL): per-sample volume raw capture at the worst pixel + camera-inside no-jitter test.
- `49ecec08b7` fix(volume): linear interpolation in the no-jitter test (root-cause probe).
- `5ab0b12eb4` debug: GL sample-dump pixel select + Metal camera-agnostic dump gate.
- `1efb0c56e7` fix(Metal): build `evalStep` in OpenGL `g_dirStep` float32 order (`sampleDistanceWorld`).
- `4867687909` test(volume): NEAREST interpolation in the camera-inside no-jitter test.
- `ccffd0095d` fix(Metal): promote the remaining half-precision lighting/interpolation sites to float32.
- `0882a41507` fix(Metal): promote per-component scalar ranges to float32 and default volume textures to float (R32F).

**Phase 2 — camera-inside proxy & analytic ray (updates 16–40)**

- `f79e93a5a6` debug(volume): widen the GL per-sample dump to 8 channels (op + premultiplied rgb) + composite readback.
- `9d9698164d` docs(Metal)+tools: step/ray/camera compare tool; update-17 findings (near-plane ray-origin root cause).
- `be9d152fcd` debug(Metal): `VTK_METAL_FULLSCREEN_CAMERA_INSIDE` override — proxy path deterministic, matches GL pre-flip geometry.
- `c7a10f508a` debug(Metal): print `METAL_CAM` at 9 sig figs (W2IF view-angle perturbation discovery).
- `7c663464e0` fix(Metal): default camera-inside to the OpenGL-parity proxy path; drop the redundant `vtkTriangleFilter`.
- `06d0619341` fix(Metal): set front-facing winding Clockwise so proxy culling matches GL's `GL_BACK`.
- `df934c8ad2` fix(Metal): clamp the camera-inside proxy `tStart` to the anchor so far remnants stop re-marching.
- `23e6c3d328` fix(Metal): interpolate the camera-inside proxy anchor in dataset space (GL `ip_vertexPos` parity) and compute `g_dirStep` from `normalize(anchorData-eyeData)`; also fixed a latent RTT `MarchParams` init (13 values / 14 fields).
- `6a157e4cef` fix(Metal): in-shader `P*V*M*v` clip attempt (GL `ComputeClipPosition` parity) — unsuccessful, retracted.
- `9d02f8b32c` fix(Metal): TEMP DEBUG cap-mesh dumps on both backends — vertex sets differ (box corner ordering).
- `53df0dd93f` fix(Metal): reorder the camera-inside `boxSource` corners to GL `ijkCorners` order + flip proxy back-face cull CW→CCW — cap meshes now byte-identical.
- `d75c70452c` fix(Metal): TEMP DEBUG clip-chain matrix bit dumps — P differs only in the near/far Z row.
- `9669e703c7` fix(Metal): bit-exact eye — pass GL's object-space eye (`in_eyePosObjs[0]`) as a uniform; camera-inside ray dir now `normalize(anchorData - eyePosData)`.
- `11b25d7a0f` docs(Metal)+debug: Metal `vertex_volume_main` vid/modelPos/clip logs + GL `GL_VERT` per-pixel + `GL_CLIPMAT` matrix dumps (update 35).
- `029e8be510` fix(Metal): bit-exact vertex clip — hand-write `(P*V)*W` and the vector multiply in GL's `[0,1,2,3]` FMA-contracted order + disable fast math (update 36).
- `a5809c683c` fix(Metal): per-vertex `ip_textureCoords` parity — masked residual 130 → 1 (update 38).
- `6ef754fed7` fix(Metal): anchor `evalPoint` on the interpolated texcoord (GL `g_rayOrigin` parity) — knife-edge 115 → 1 (update 39).
- `f597ef686c` fix(Metal): compose `in_inversePVM` as `inv(P)*inv(V)*inv(M)` in both backends (was `inv(V)*inv(P)`) — analytic rays match GL (update 40).

**Phase 3 — composite & background blend (updates 42–58)**

- `a5a59cb1b4` fix(Metal): composite in GL order `w*(c*a)` (update 43).
- `8083e26aca` fix(Metal): composite via explicit `fma()` to match GLSL's compiled contraction of `(1-a)*c+frag` (update 45).
- `970e84d6d5` fix(Metal): `in_volume_scale` parity for UCHAR/USHORT unorm — upload ScalarMin/Max divided by `(normFactor+1)` (update 46).
- `a25e73dd15` fix(Metal)+GL: TF-table pre-integration factor float32-vs-double — cast `actualSampleDistance` to float like GL (`vtkOpenGLGPUVolumeRayCastMapper.cxx:1710`), plus the env-gated `VTK_GL_OPTABLE_DUMP` (update 52).
- `5d7f30b333` docs+debug(GL): `VTK_GL_FINAL_DUMP` channels 60–67 — true float32 final capture at 15 gated px (update 54).
- `dd93d95341` docs+debug: Metal `VTK_METAL_CLEAN_BIAS_F` uniform-bias probe + GL `VTK_GL_FLOAT_DUMP` RGBA32F readback (update 55).
- `fdd7281d07` docs+debug(GL): fix the float-FBO readback RGB=0 — save/restore the active-unit 2D texture binding the bare `glBindTexture` was clobbering (update 56).
- `8f991da45b` **fix(Metal,volume) (update 57):** remove the `accumulatedOpacity=1.0` clamp at the opacity break (`MetalShaders.metal:4814`) and make the break `>=` → strict `>` — the root cause of the whole 63,692-px ±1 field (the clamp zeroed the `dst*(1-a)` background blend term GL keeps); field collapsed 63,692 → 188 px.
- `fea09bf24a` chore(Metal,volume): remove the now-dead `VTK_METAL_CLEAN_BIAS_F` debug bias + its compile-option injection.
- `651e3ea4ae` docs+debug(Metal): full-field pre-store float dump `VTK_METAL_FLOAT_DUMP` (262,144-px FINAL log) for the post-blend-fix feasibility analysis (update 58).

**Phase 4 — the interpolator floor (updates 59–80)**

- `696bbb9be3` docs+debug: fix `DumpCleanGLFloats` to dump every frame (the frame-1-vs-frame-6 misalignment had hidden the true gf delta) + add the evalStep/anchor/clip lattice to both dump gates (update 59).
- `8109341d27` docs+debug: extend the `VTK_GL_VERTEX_DUMP` to 11 channels incl. per-vertex texcoord (108/109/110); fix the debug-vertex branch to compute the real texcoord before its early return; Metal logs the exact screenPos pixel-center (update 60).
- `08d00f434d` docs+debug: GL `DumpDebugAttrField` renders channels 200–202 into an RGBA32F FBO (full-frame attribute planes); FIX the channel-100 block upper bound that was silently intercepting channels 200–202; Metal `debugStepGate()` sparse 64×64 grid + dense knife-edge block → 8203 unique px/frame (update 64).
- `0fc7e84852` docs+tests: add `Linear` and `StepTF` diagnostic tests (update 65).
- `c6ea74d513` docs+tests: StepTF mode-3 non-saturating ramp sweep (`VTK_STEP_RAMP_MAX`) (update 66).
- `68bac8533d` docs+tests: rewrite StepTF as a true single-still-frame + variants A–E (update 67).
- `6032ed6015` **fix(metal,volume) (update 69):** march-loop bounds exit on the GL `g_dirStep` lattice — directional per-axis bounds test against cell-to-point adjusted bounds (TerminationImplementation parity), `maxSteps = ceil((tEnd-firstT)/length(evalStep))`, tTerminateMax in step units; B constant-scalar volume 5530 → 18 px.
- `e40517af16` fix(metal,volume) (update 70): remove the legacy block-bounds exit (GL has none) — B 18 → 3 px.
- `5f8f1f5629` docs+debug: STEP lines self-contained with shader inversePVM bits + GL debug channels 43–47 (`rcpNear/rcpFar/q0mul/sweepA/B`) and `VTK_GL_CAP_DUMP`/`VTK_GL_SWEEP_DUMP` guards (update 71).
- `4def5f95c3` **fix(metal,volume) (update 74):** two GLSL-vs-MSL codegen divergences in the analytic pixel ray — GL `X/=w` compiles to multiply-by-reciprocal (`raw*(1.0/w)`) and `normalize` lowers to `d*fast::rsqrt(dot(d,d))` with the GPU's approximate rsqrt — `dirObj`/`evalStep` now byte-identical to GL `g_dirStep`; residual 183 → 178 px.
- `8c5d361922` docs+debug: fix the GL_VERT dump channel 107 to read `ip_vertexPos.z` instead of `gl_PrimitiveID` (the near-cap z==y/vid artifact source) (update 76).
- `2b89fc6049` **fix(metal,volume) (updates 78–80):** the decisive analytic-anchor experiment — `TriangleAnchorBuffer` (fragment buffer 3) + `BuildTriangleAnchorBuffer` strict-fma per-vertex clip/texcoord parity, `AnalyticAnchorMode` uniforms (f32/f64 A/B), and `VTK_STEP_RAW_CAPTURE` front-buffer capture hooks in three GPUVolumeRayCast tests. Result: negative (mode-1 429 px / mode-2 469 px vs the 178-px baseline); residual pinned to the rasterizer interpolator hardware floor.

**Phase 5 — final fixes & the acceptance suite (updates 81–84)**

- `df9191cd4f` docs+debug: bump the Metal FINAL log `accOp`/`accCol` from `%f` to `%0.9g`; add StepTF m4/m2 gate pixels + `VTK_GL_STEPTF_DUMP` debug block (recap §4b).
- `8696d5d3ef` docs+debug: MaxIP gate `pxOkMaxIP` corrected to PNG rows + `mipMaxPos`/`mip2` tracking in `marchVolumeUnified` (update 81).
- `2f88ec35ac` **fix(Metal):** densify the camera-outside proxy box to GL's exact `vtkDensifyPolyData(2)` centroid-fan geometry (108 tris / 56 verts, float32) and upload it in dataset space via the new `UseDataSpaceBoxVertices` uniform (forwarding `in.position` unchanged) — CamOutside 1384 → 250 px (update 82).
- `ef4452a9bc` **fix(Metal):** camera-inside near-plane normal transform — use the transpose `M^T` of the model matrix instead of the inverse-transpose for world→data normals, bit-matching `vtkOpenGLGPUVolumeRayCastMapper::RenderVolumeGeometry`; fixes `TestGPURayCastCameraInsideNonUniformScaleTransform` (47,776 → 45,837 px; cap mesh byte-identical to GL).
- `5371e30652` **fix(Metal) (update 83):** replace Interleaved Gradient Noise jitter with GL's exact blue-noise tile (`kBlueNoise64[4096]`, JPEG top-down byte order) sampled `(floor(st.y)−H) mod 64 / floor(st.x) mod 64` in all three jitter sites (fullscreen, RTT, grid-traversal, the last dropping its old `+0.5` half-pixel shift); adds `TestGPURayCastCameraInsideNonUniformScaleTransformKnobs` — jittered 300² test 45,840 → 29 px.
- `fe81a78ebb` **fix(Metal) (update 84):** disable blending on the RenderToImage pipeline color attachment 0 (GL's RTT pass writes unblended over `vtkglClearColor(1.0,1.0,1.0,0.0)`) — `TestGPURayCastRenderToTexture` 47,878 px / max_d 65 → 1,471 px / max_d 1.
