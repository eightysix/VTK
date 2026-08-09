# Metal's pipeline is closed to the LSB on all 68 gated rays: a full float32 accumulate-lattice replay (evalPoint = f32(localPos+evalStep), nearest texel, TF tables, fma composite, opacity break, round-half-even store) reproduces Metal's stored image 68/68 — and GL's own debug-dumped lattice reproduces clean GL only where clean-GL == clean-Metal, reproducing **Metal** at a real ±1 pixel (93,201). The ±1 field is clean GL's compile-level arithmetic divergence, not geometry (update 48)

**Date:** 2026-08-09
**Scope:** Update 47's §5 next step ("CPU-replay GL's exact `g_rayOrigin + g_dirStep·g_currentT` lattice against the volume texels and TF tables") executed in full. This session (1) validated the entire Metal side end-to-end — transfer function, sample lattice, composite, break, store — against Metal's stored image on **all 68 gated pixels** (extending update 47's 15/15 on differing pixels), and (2) replayed GL's exact logged `g_rayOrigin`/`g_dirStep` from the debug-dump run `gl372.log` on the 14 gated pixels that overlap the GL dump. At (93,201) — the only shared pixel in the ±1 field — GL's dumped lattice recomposes to **Metal** in both frames while clean GL is one channel higher; at the other 13 shared pixels (all clean-GL == clean-Metal) the dump reproduces clean GL except at knife-edge pixels where the dump's own frame-to-frame ~1e-7 step jitter (not clean GL) swings the result. That is a direct, self-contained confirmation of update 44's "debug-GL ≈ Metal, clean GL diverges at compile level" and localizes the entire remaining field to clean GL's compiled GLSL arithmetic.
**Target (confirmed):** Metal output must be bit-identical to **clean GL** (`RenderingBackend=OpenGL` without debug injection).

**Follows:** [Update 47](VolumeRayCastBackendComparisonFindingsUpdate47.md), [Update 46](VolumeRayCastBackendComparisonFindingsUpdate46.md).

---

## 1. The full Metal replay: exact model, 68/68 gated pixels

The replay consumes only Metal's own `STEP` log rows (`localPos`, `evalStep`) plus the volume data (`/tmp/bc/vol512.npy`) and reproduces Metal's stored image **byte-for-byte on all 68 gated pixels**:

```
p₀ = f32(localPos + evalStep)                 # evalPoint₀ = localPos + evalStep·jitterFrac, jitterFrac = 1.0 (update 39 anchor)
loop:  texel = clamp(floor(p·512), 0, 511)    # nearest, volume coords [0,1]
       raw = vol[texel]
       stored = f32(raw/65536); norm = f32(stored·65536/4370)
       idx = floor(clamp(norm,0,1)·1024); op = OP_TABLE[idx]; rgb = COL_TABLE[idx]
       if op > 0:
           w = f32(1 − accA)
           accC = fma(w, f32(op·rgb), accC)     # exact fp64-product fma
           accA = fma(w, op, accA)
           break when accA ≥ f32(1 − 1/255), clamp accA → 1
       p = f32(p + evalStep)
store: u8 = round(clamp(accC,0,1)·255)          # round-half-even; matches Metal's store on all 68
```

- **TF model validated to print precision** before the lattice run: predicted `norm`/`op`/`rgb` for the (38,448) first sample equal the SAMPLE log rows (norm 0.025172, op 0.001155, rgb (0.213587, 0.106794, 0.064076) — all within 6-decimal rounding of the log).
- **The `+evalStep` first-sample anchor is load-bearing:** replaying with the first sample *at* `localPos` (dropping the `+step`, as a first script did) matches only 39/68 — the first sample is the near-plane entry, and `localPos` vs `localPos+evalStep` is a ~1-texel shift that flips results on boundary rays. Metal's true `evalPoint₀ = localPos + evalStep` (update 39) is what closes to 68/68.
- **Termination is validated implicitly:** the `accA ≥ 1−1/255` break, the break sample, and the final clamp all reproduce; no pixel needs a different break or a one-extra-sample variant (consistent with update 47 §3's refutation).
- **Store model (round-half-even `×255`) is validated** — update 47 only replayed float `accCol`; this closes the store as well.

Combined with update 46 §3.4 (Metal's accumulation replays the written GLSL formula), this means **every stage of Metal's pipeline that can be observed is now reproduced exactly from its own inputs** — the Metal side contains no unknown.

## 2. GL's own debug-dumped lattice does not reproduce clean GL where it matters

`gl372.log` (debug-injected GL run, `GL_RAY` lines carry the exact `g_rayOrigin`/`g_dirStep`) overlaps 14 of the 68 gated Metal pixels (y-flipped: GL `(x, 511−y)`). Replaying GL's exact `(g_rayOrigin, g_dirStep)` with the *same* accumulator/TF/store model:

| result | count (first/last frame) | pixels | clean GL vs Metal there |
|---|---|---|---|
| GL replay == clean GL (`u47_gl.png`) | 12/14 · 10/14 | (104,245), (188,307), (242,330), (307,7/8/9), (322,172), (357,154), (382,207), (496,488), and (372,131)+(480,400) on first frame only | all clean-GL == clean-Metal |
| GL replay ≠ clean GL | 2/14 · 4/14 | (93,201) both frames; (422,92) both frames; (372,131), (480,400) last frame only | (93,201) is the only shared pixel in the ±1 field; the rest clean-GL == clean-Metal |

- **Decisive case (93,201):** the *only* shared pixel in the ±1 field. GL's dumped lattice replays to `(247,170,130)` — **Metal's value** — in **both** frames, while clean GL stores `(247,171,131)`. Same accumulator, same TF, same store; the only difference between the replay and clean GL is that the replay uses clean GL's *logged* `g_rayOrigin`/`g_dirStep`, which were captured under the debug-injected compile. So the debug-injected GL evaluates the ray at the sample level like Metal, and clean GL's compiled evaluation is what is one channel higher. This is update 44's conclusion in a single pixel, produced without any shader modification: **clean GL ≠ debug GL ≠ Metal, and the ±1 field is clean GL's compile-level divergence.**
- **The other 13 shared pixels all have clean-GL == clean-Metal**, and in **no** case does the dump ever replay toward "clean GL ≠ Metal". Where the replay misses clean GL it misses *both* (knife-edge sensitivity, below): the dump's own frame-to-frame step jitter (~1e-7) swings the replay by up to ~15 LSB, so those pixels are below the dump's reproducibility precision — not evidence about clean GL. The debug-GL lattice is therefore *not* clean GL's lattice (update 44), and it only looks like clean GL's lattice on the pixels where clean GL and Metal already agree.
- **Cross-run sensitivity of the dump is real but secondary:** gl372's `(422,419)` step differs between its frames by ~1e-7 (e.g. `-5.0e-6` vs `-5.1e-6` in y) and the two frames replay to `(238,190,157)` vs `(238,176,140)` — a 14–17-LSB swing on a knife-edge pixel (below). Frame choice does not rescue (93,201): neither frame reproduces clean GL there.

## 3. GL-vs-Metal lattice parameters agree to <1e-6 — geometry is ruled out for the 63,690-px field

For all 14 shared pixels the GL and Metal lattice inputs are equal to within float rounding:

- `tex(GL) − localPos(Metal)` ≤ **5.4e-7** (anchor)
- `g_rayOrigin − f32(localPos+evalStep)` ≤ **4.8e-7** (first-frame dump) / **6.6e-7** (last-frame dump) — first sample point
- `g_dirStep − evalStep` ≤ **9.4e-8** (first frame) / **1.6e-7** (last frame) per axis — step

These offsets are far below the 1/512 ≈ 2e-3 texel spacing. A ~1e-6 seed growing by ~1e-7/sample can flip at most a handful of texel selections per ray, and only when the ray grazes a boundary — far too rare to produce a uniform 63,690-px one-directional field. **The ±1 field is not a geometric/texel-input divergence at this precision.**

## 4. Knife-edge amplification at grid-aligned rays explains the *gated* ±1 pixels (and why the GL dump flips so hard)

At near-grid-aligned rays the y-step is ~1e-5 texel/sample, so y is nearly constant for the whole ray and sits within ~0.4 texel of a boundary. A ~1e-7 step difference changes *which sample* crosses the boundary, after which the entire ray tail samples a different texel column — a discrete, amplified color jump from a sub-ULP step difference:

- (422,92): Metal steps `evalStep = (−4.5347e-4, −4.95e-6, +1.8373e-3)`; the two gl372 frames' steps differ by ~1e-7 and replay to `(238,190,157)` (171 samples) vs `(238,176,140)`; Metal's own lattice gives `(238,192,159)` (170 samples). Same TF, same accumulator — pure lattice-vs-boundary alignment.
- This is the mechanism behind the **15 gated pixels** in the ±1 field (all GL one channel higher), but its requirement (ray ~parallel to a grid axis and grazing a boundary) is why it cannot scale to the whole 63,690-px field.

## 5. Conclusion

- **Metal side closed:** 68/68 gated rays reproduced to the stored LSB from `localPos`/`evalStep` + volume + TF tables. Metal implements the written formula exactly (updates 45-47).
- **GL side split confirmed:** clean GL's output at the ±1 pixels is produced by none of (a) Metal's lattice, (b) GL's own debug-dumped lattice, (c) the written formula — it is produced by clean GL's compiled GLSL arithmetic alone. Debug-GL tracks Metal, not clean GL (update 44), now demonstrated at a real ±1 pixel (93,201) via lattice replay without any shader edit.
- The remaining task is unchanged and now fully pinned: **reproduce clean GL's compile-level arithmetic divergence in MSL (or eliminate it on the GL side).** Geometry, TF, accumulation, termination, and store are all closed.

## 6. Next steps

1. **Bisect the compile-level divergence by forcing compiler behavior.** Compile the GLSL fragment shader with FMA contraction off / strict-float (`-ffp-contract=off` or `#pragma`/`precision` variations) or Metal with explicit non-fma `mul+add`, re-diff the ±1 field (update 44 §5's suggested bisect, never executed). If some variant makes Metal's or GL's field disappear, the exact contraction/reassociation site is identified.
2. **Whole-loop, not single-step, reassociation.** Update 46 §4.4 showed the single composite step is FMA-order-insensitive; the divergence therefore lives in clean GL's compiled *loop* (register reuse across samples, position-update chain contraction, or the `scalar*scale` chain). Enumerate the few candidate chains and test each by emitting the alternative in MSL.
3. **Revisit store rounding on the GL side.** The field's one-directionality (GL higher) is consistent with Metal round-half-even vs GL truncation *combined* with the float-level bias, but update 42 measured a float-level ~3.5e-4 bias independent of the store — keep this as the fallback check after (1)/(2).

## Artifacts

- `/tmp/bc/u47_metal.log` (~54k SAMPLE/MARCH/FINAL rows, 6 deterministic renders per gated pixel), `/tmp/bc/u47_metal.png`, `/tmp/bc/u47_gl.png` (13:07).
- `/tmp/bc/vol512.npy` — the 512³ volume (`x` fastest) extracted from the test's reader; note vol must be indexed `[x][y][z]`.
- `/tmp/bc/gl372.log` — debug-injected GL run with `GL_RAY` origin/step per dumped pixel (15 px; 14 overlap the gated set after y-flip).
- Replay scripts (persisted in `Rendering/Metal/BackendComparisonTools/update48/`, see `README.md` there for regeneration and input data): exact float32 fma via fp64-exact product + fp32 round; nearest texel `floor(clamp(p,0,1)·512)`; break `accA ≥ 1−1/255`; store `round(f·255)`; Metal match 68/68. `replay_metal_accumulate.py` (68/68 + the 15 gated ±1 pixels), `replay_gl_lattice.py` (<1e-6 lattice agreement + GL dump replay), `knife_edge_422_92.py` (knife-edge amplification).
