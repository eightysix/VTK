# Metal ↔ GL GPU Volume Raycast — Full Recap (updates 1–80)

**Goal:** Metal output **bit-identical** to **clean GL** (reference, `RenderingBackend=OpenGL`, no debug injection) on `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter` (512², camera-inside, NEAREST interpolation, 6-frame animated camera).

**Current status:** **178 px differ / 262,144 (0.068 %), max channel delta 8** at the worst pair `(397,110)` ↔ `(397,401)`. All shader-visible inputs are bit-identical — **including per-vertex `texcoord` on all three axes (94/94, update 80)** — and the residual is a ~1–2 ulp offset in the **rasterizer-interpolated** ray anchor (Metal vs GL), causing nearest-texel selection flips at grid-aligned (knife-edge) rays. The decisive analytic-anchor experiment was run and is **conclusively negative** (mode-1 429 px / mode-2 469 px, update 78); the frame-aligned raw capture refutes frame-selection (update 79); the per-vertex z-input lever is refuted (update 80). The 178 px is therefore **bounded below by the GL-driver vs Metal interpolator arithmetic difference at knife-edge picks**, with every input proven bit-identical. The single remaining lever is the full-gate interpolation-floor quantification; if it confirms the floor, accept 178 px as the irreducible hardware bound and switch the acceptance metric to the frame-aligned `VTK_STEP_RAW_CAPTURE` diff.

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
3. **Legacy paths:** camera-outside loops, label-map gradient-opacity parity, offscreen RGBA16F accumulation for downsampled/RenderToImage — out of scope for this test, structural parity only.

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

Env-gated: `VTK_GL_RAY_DUMP`/`VTK_GL_VERTEX_DUMP`/`VTK_GL_SAMPLE_DUMP`/`VTK_GL_OPTABLE_DUMP`/`VTK_GL_FINAL_DUMP`/`VTK_GL_FLOAT_DUMP`/`VTK_GL_CAP_DUMP`/`VTK_GL_SWEEP_DUMP`/`VTK_GL_RESID_DUMP`, `VTK_METAL_FLOAT_DUMP`/`VTK_METAL_CLEAN_BIAS_F`/`VTK_METAL_ANCHOR_PERTURB`/`VTK_METAL_FULLSCREEN_CAMERA_INSIDE`/`VTK_METAL_ANALYTIC_ANCHOR`/`VTK_METAL_ANCHOR_REC_DUMP`, `VTK_STEP_RAW_CAPTURE` (front-buffer frame-aligned capture), plus the `DEBUG STEP`/`vertex_volume_main`/`fragment_volume_main` os_log blocks in `MetalShaders.metal`. None affect clean output (update 72: debug Metal == clean Metal, 0 px), but they should be removed or gated behind the test-build macro before production.

## 7. Test / build commands

```
./macos_metal_build.sh --resume --tests
# run: -V $TMP/<unique>.png  --vtk-factory-prefer RenderingBackend=OpenGL (GL) / =Metal (Metal)
# acceptance metric (updates 79/80): VTK_STEP_RAW_CAPTURE=<out>.png on the NoJitter test, diff Metal(raw) vs GL(raw) — frame-aligned, deterministic
```

## 8. Chained reference (in order, read these for depth)

Updates **1–19** (pipeline: gates, fp16, TF tables, half→float, NEAREST insight, W2IF) → **20–40** (proxy mesh, matrix/FMA parity, 307→130→694 px, inversePVM order) → **41–59** (composite exonerated, background-blend root cause 63,692→188 px, frame-aligned dumps, step vs anchor) → **60–64** (per-vertex bit-identity, barycentric weights ~1e-8, interpolation floor quantified) → **65–70** (finetuned matrix, sample-count lattice 5530→3) → **71–77** (ray byte-identity, rcp/rsqrt codegen, anchor +1 ulp proven, analytic-anchor audit) → **78** (analytic-anchor negative: mode-1 429 / mode-2 469 px; GL frame-A/B anchor spread) → **79** (frame-aligned raw capture refutes frame-selection: same-camera 178 px) → **80** (per-vertex texcoord z 94/94, z-input lever refuted, interpolator-model sweep negative; residual = interpolator hardware floor).
