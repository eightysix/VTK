# Metal ↔ GL GPU Volume Raycast — Full Recap (updates 1–77)

**Goal:** Metal output **bit-identical** to **clean GL** (reference, `RenderingBackend=OpenGL`, no debug injection) on `TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter` (512², camera-inside, NEAREST interpolation, 6-frame animated camera).

**Current status:** **178 px differ / 262,144 (0.068 %), max channel delta 8** at the worst pair `(397,110)` ↔ `(397,401)`. All shader-visible inputs are bit-identical; the residual is a ~1–2 ulp offset in **Metal's *interpolated* ray anchor** (`in.texcoord`) vs GL's `ip_textureCoords`, causing nearest-texel selection flips at grid-aligned (knife-edge) rays. The one decisive, never-run experiment is the **analytic-anchor shader implementation**.

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
| Per-vertex clip (x/y/w) and per-vertex texcoord (x/y) | updates 36, 60, 76: 94/94 bit-identical, joined on clip x/y/w |
| Pixel center (GL `gl_FragCoord` == Metal `in.position.xy`) | update 60: exact `(397.5, 110.5)` |
| Proxy cap mesh (verts/indices), V/M matrices, `in_volume_scale`, sample distance | updates 27–29, 33, 50: byte-identical |
| GLSL↔MSL mat-mul FMA contraction | update 36: `(P*V)*W`, `mul,fma,fma,mul+add`, `fastMathEnabled=NO` — collapsed 307→130 px |
| Step (`evalStep` == `g_dirStep`), `dirObj` 114/114, dirObj lattice | updates 17, 59, 69, 74, 76: bit-identical |

## 2. The residual and its proof of origin (updates 75–77)

- The 188→183→178 px residual is **nearest-texel selection flips** at texel boundaries (e.g. `z=0.6953125 = 178/256`), driven by a **~1–2 ulp difference in the interpolated anchor texcoord** (Metal systematically at-or-above GL: state A y+1, state B x+1/y+1/z+2 ulp). Color deltas up to 8 u8 with negligible alpha deltas. `evalStep` is byte-identical — the ulp is not in the step.
- **Per-vertex data exonerated:** per-vertex clip and per-vertex texcoord x/y are 94/94 bit-identical (update 60/76). The +1 ulp is created **inside the rasterizer interpolator** (update 76 §4): a full-float64 perspective-correct interpolation of the bit-identical inputs reproduces **GL exactly** (`0x3f01aa39`) while Metal logs `0x3f01aa3a`.
- **The rasterizers do not evaluate at a discoverable sample point:** implied sample point == exact pixel center for both backends; fitted sub-pixel offsets reduce but never reach 0 and differ per backend (updates 61–64). Effective GL-beyond-Metal sample displacement ~(+0.025, −0.032) px with all scatter explained by ±1–3 ulp f32 amplification through triangle 122's near-degenerate texcoord space (update 64).
- **No analytic pixel-center model reproduces either backend at 0 ulps** (affine / persp `1/w` / `rcp(1/w)` / inverse-w, update 63) — **except** update 76 §4's single-pixel f64 result, whose f64-NDC weights contradict update 61's "hardware divide is f32" finding. This contradiction is **unreconciled**.

## 3. The decisive remaining lead: analytic-anchor shader experiment

Never implemented (`anchorTex = in.texcoord` is the only fragment-stage assignment). Implement the anchor as `Σ(wᵢ·texᵢ/clip.wᵢ) / Σ(wᵢ/clip.wᵢ)` from pixel-center barycentrics + per-vertex clip/texcoord, bypassing the interpolator for `anchorTex` only (camera-inside proxy path).

- **Data availability:** cap mesh is only ~94 vertices / 126 triangles; a CPU-built per-primitive (clip[3], texcoord[3]) buffer indexed by `primId [[primitive_id]]` is cheap.
- **A/B variants:** (A) f64 weights, (B) f32-NDC weights — this is exactly the f64-NDC-vs-f32-NDC axis that separates the contradictory update 76 §4 vs update 61/63 results.
- **If it closes:** pick the 0-ulp variant, extend to the parallel-projection path, re-verify the 178 px and the finetuned matrix.
- **If it does not:** quantify the interpolation floor across the full 8237-px gate (per-axis ulp histogram), frame-match, document as the irreducible bound.

## 4. Secondary / parallel leads

1. **W2IF frame perturbation / frame alignment (update 19/74):** GL *itself* flips the i=132 texel between its frame groups (3× texel 177, 3× 178) because `vtkWindowToImageFilter` perturbs the view angle 30 → 30.0000008° at frame 3→4. The reference comparison must state which GL frame it pairs against; frame 1–3 (unperturbed camera) is the fair target. Metal matches GL's texel-178 frames to the digit.
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
- z-texcoord per-vertex readback is an encoding artifact on GL's near-cap rows (`tex.z==tex.y`, `pos.z==vid`) — only the interpolated fragment value is meaningful (update 76).

## 6. Instrumentation / debug scaffolding still in the tree (evaluate for cleanup)

Env-gated: `VTK_GL_RAY_DUMP`/`VTK_GL_VERTEX_DUMP`/`VTK_GL_SAMPLE_DUMP`/`VTK_GL_OPTABLE_DUMP`/`VTK_GL_FINAL_DUMP`/`VTK_GL_FLOAT_DUMP`/`VTK_GL_CAP_DUMP`/`VTK_GL_SWEEP_DUMP`/`VTK_GL_RESID_DUMP`, `VTK_METAL_FLOAT_DUMP`/`VTK_METAL_CLEAN_BIAS_F`/`VTK_METAL_ANCHOR_PERTURB`/`VTK_METAL_FULLSCREEN_CAMERA_INSIDE`, plus the `DEBUG STEP`/`vertex_volume_main`/`fragment_volume_main` os_log blocks in `MetalShaders.metal`. None affect clean output (update 72: debug Metal == clean Metal, 0 px), but they should be removed or gated behind the test-build macro before production.

## 7. Test / build commands

```
./macos_metal_build.sh --resume --tests
# run: -V $TMP/<unique>.png  --vtk-factory-prefer RenderingBackend=OpenGL (GL) / =Metal (Metal)
```

## 8. Chained reference (in order, read these for depth)

Updates **1–19** (pipeline: gates, fp16, TF tables, half→float, NEAREST insight, W2IF) → **20–40** (proxy mesh, matrix/FMA parity, 307→130→694 px, inversePVM order) → **41–59** (composite exonerated, background-blend root cause 63,692→188 px, frame-aligned dumps, step vs anchor) → **60–64** (per-vertex bit-identity, barycentric weights ~1e-8, interpolation floor quantified) → **65–70** (finetuned matrix, sample-count lattice 5530→3) → **71–77** (ray byte-identity, rcp/rsqrt codegen, anchor +1 ulp proven, analytic-anchor audit).
