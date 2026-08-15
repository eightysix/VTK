# Metal Volume Rendering Performance Investigation

Investigation of why the Metal GPU volume ray-caster is slower than the OpenGL
back-end for the DICOM volume scene, and a proposed fix.

- **Branch:** `metal-volume-parity-essential`
- **Hardware:** Apple M2 (Metal 3)
- **Scene:** DICOM volume 512x512x1794, R8 u8, ~470 MB, `--scene DICOMVolume`
  (render target 400x400, ~69,862 box pixels)
- **Bench config:** `VTK_METAL_TEST_SAMPLE_DISTANCE=0.5`
  `VTK_METAL_TEST_JITTER=1` `VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0`
  `VTK_METAL_TEST_MINMAX=1` `VTK_METAL_TEST_ACCEL=0`

---

> **Update (follow-up investigation):** after the transpose / X-march layout
> change and the AIR-level (Metal ISA) check, the findings in sections 1-4 were
> refined. See **section 7** for the latest numbers, the camera-dependence
> analysis, and the final conclusion: the residual 2x gap is the Metal 3-D
> linear filter cost, which is a single texture op at the AIR level (not a
> compiler expansion) and is **not** emulatable from Metal shader code.
>
> **Update (minimal repro):** **section 8** reproduces the Metal-vs-GL linear
> gap (~2-2.7x filter add-on) in a ~480-line standalone program with a trivial
> raymarch shader and synthetic volume, proving it is a texture-unit property,
> not a VTK-path artifact.

---

## 1. Headline results

All times are GPU times measured identically for both back-ends (GL: `glFinish`
after each frame; Metal: `WaitForCompletion` + `GPUEndTime - GPUStartTime`).
At the bench sample distance of 0.5 mm the march performs ~46 M texture samples
per frame (69,862 pixels x ~659 avg steps; verified by an on-GPU counter:
`total_iters = 46,049,650`).

| Back-end / mode                          | ms    | ns/sample |
|------------------------------------------|-------|-----------|
| GL full shader, linear (baseline target) | 48.9  | 1.06      |
| GL full shader, nearest                  | 44.8  | 0.97      |
| **Metal real shader (current, divergent)**| **103.0** | **2.24** |
| Metal linear, fixed count, clamp_to_edge | 72.7  | 1.58      |
| Metal linear, fixed count, no break (select) | 72.2 | 1.57   |
| Metal linear, fixed count, clamp_to_zero | 64.3  | 1.40      |
| Metal nearest, fixed count               | 46.4  | 1.01      |

Interpretation:

1. **Nearest-sampling throughput is identical** between GL and Metal
   (~0.97 vs ~1.01 ns/sample). The baseline texture path is not the problem.
2. **Metal's 3-D linear sample is ~1.4-1.6 ns/sample, GL's is ~1.06 ns/sample.**
   Metal pays +0.4 to +0.6 ns/sample for trilinear; GL pays only +0.2 ns.
   No experiment (sampler address mode, manual trilinear, 2-D array slice-lerp,
   software pipelining) closed this gap.
3. **Metal pays a ~30 ms divergence penalty** for the data-dependent loop
   `break`s (103 ms -> 72 ms when the break is replaced by predication). GL pays
   no such penalty. This is the largest actionable lever.

---

## 2. What was measured and how

### 2.1 Sample-distance scaling (both back-ends, `--backend gl|metal`)

Varying `VTK_METAL_TEST_SAMPLE_DISTANCE` (1.0 / 0.5 / 0.25 mm):

| SD (mm) | GL ms | Metal ms |
|---------|-------|----------|
| 1.0     | 43.24 | 70.87    |
| 0.5     | 50.6  | 101.62   |
| 0.25    | 55.63 | 154.97   |

Both are sub-linear in sample count (latency-bound), Metal much steeper.
At `sampleDistance=5000` (~1 sample/ray): GL 1.20 ms, Metal 0.39 ms; GL with
`CellColor` (no volume): 0.54 ms -> no fixed per-frame overhead in either path.
GL `imageSampleDistance` 1.0/0.5/0.25 -> 48.07/21.75/1.81 ms (GL is
fragment-bound / pixel-bound, not sample-bound).

### 2.2 GL iteration-count cap experiment

`VTK_GL_ITER_CAP` was injected into `raycasterfs.glsl` and referenced from
`vtkVolumeShaderComposer.h`'s `TerminationImplementation` as
`g_currentT >= min(g_terminatePointMax, VTK_GL_ITER_CAP)`:

| GL cap | ms |
|--------|-----|
| 50     | 4.58 |
| 659 / 1e6 | 50.9 |

Linear fit: GL ~= 0.8 ms fixed + **1.09 ns/sample**, i.e. the GL march really
does ~46 M samples at ~1.1 ns each. GL's march is iteration-bound; the 50 ms is
not explained by anything else (filters, depth pass, uploads).

### 2.3 Filter cost isolation

GL nearest vs linear (same scene): 44.8 vs 53.9 ms -> GL trilinear adds
~9 ms over 46 M samples (~0.2 ns/sample, i.e. native and cheap).
Metal nearest vs linear (probe, fixed count): 46.4 vs 72.7 ms -> Metal
trilinear adds ~26 ms (~0.57 ns/sample).

### 2.4 Divergence isolation (probe)

The real shader's loop (`marchVolumeUnified`) breaks at a per-fragment,
data-dependent iteration (per-fragment `maxSteps`, CTP-bounds exit,
`tTerminateMax`, opacity threshold). A probe variant that runs a **fixed,
uniform** iteration count for all fragments (no break) drops the march from
~103 ms (divergent) to ~72.7 ms (fixed). A variant that keeps the uniform count
but uses predication (`select`) instead of a break - so the result is
image-correct and the fetch stays unconditional - measures **72.2 ms**.

This proves ~30 ms is pure SIMT divergence on the data-dependent exit, and that
predication recovers it with no image change.

### 2.5 Sampler / texture experiments (all fixed-count, ~46 M samples)

| Sampler / strategy               | ms    | note |
|----------------------------------|-------|------|
| `clamp_to_edge` (current) linear | 72.7  | baseline |
| `clamp_to_zero` linear           | 64.3  | best linear, keep in-shader clamp |
| `repeat` linear                  | 95.8  | worse - clamp handling is not the cost |
| manual trilinear (8 nearest taps)| 96.8  | native linear cheaper than 8 taps |
| 2-D array slice-lerp (2 samples) | 77.7  | per-sample 0.84 ns but x2 samples |
| software pipeline 2-deep / 3-deep| 76.4 / 73.7 | no gain vs 72.7 |

The 2-D-array slice-lerp result per *sample* (0.84 ns) is notably cheaper than
native 3-D linear (1.56 ns), but trilinear needs two of them (1.68 ns/result),
so it loses overall. This is the closest evidence that Metal's 3-D swizzled
layout locality is the root cause of the linear-sample gap (GL's slice-oriented
3-D layout streams along the dominant Z-march); it could not be exploited to a
win because a single native 3-D sample is already cheaper than two 2-D ones.

### 2.6 Invalid measurements (for the record)

- `fragment_fixedpoint_linear/nearest`, `fragment_march_xdir_linear`: measured
  1.3-2.6 ms regardless of 46 M iterations. The compiler constant-propagated /
  CSE'd the invariant coordinates (loop degenerated), and the X-direction sweep
  is cache-hot (all fragments traverse the same thin slab). **Do not trust these
  as texture-unit throughput.**
- Air disassembly (`minitest.metal`): the march loop is **not unrolled** and
  issues one `sample_texture_3d` per iteration (LLVM IR level); no GPU ISA was
  obtained.

---

## 3. Conclusions

- Metal's raw path at the bench config is **103 ms vs GL's 48.9 ms**.
- The 54 ms gap decomposes roughly as:
  - ~30 ms SIMT divergence on the data-dependent loop exit (fixable).
  - ~20-25 ms Metal 3-D linear-sample overhead vs GL (not yet fixed;
    strong evidence it is swizzled-layout locality; the 2-D-array per-sample
    advantage (0.84 ns) hints at the fix shape but needs a 1-sample/result
    design to win).
  - the rest is measurement/loop noise.
- Applying the divergence fix + `clamp_to_zero` should bring the app to
  **~64-75 ms** (image-identical). Reaching GL's ~49 ms requires solving the
  linear-sample cost, which is not yet solved.

---

## 4. Proposed fix (non-divergent march)

Target: `marchVolumeUnified` in `Rendering/Metal/Shaders/MetalShaders.metal`
(currently lines ~4095-4674).

### 4.1 Current structure (divergent)

```metal
int maxSteps = max(1, int(ceil((p.tEnd - firstT) / p.stepSize))); // per-fragment
for (int i = 0; i < maxSteps; i++) {
  if (!p.checkBounds && currentT >= p.tEnd - 1e-6) break;            // break 1
  // CTP directional bounds check:
  if (any(max(evalStep,0) * (evalPoint - adjTexMax) > 0) ||
      any(min(evalStep,0) * (evalPoint - adjTexMin) > 0)) {
    if (seenInBounds) break;                                          // break 2
    texLocalPos = clamp(...);
  }
  // ... sample, transfer function, composite ...
  if (accumulatedOpacity > 1.0h - 1.0h/255.0h) break;                 // break 3
  if (currentT >= p.tTerminateMax) break;                             // break 4
  // one-step-ahead prefetch (hides latency behind the data-dependent break)
  prefetchScalar = sampleVolumeScalar(...); prefetchValid = true;
}
```

### 4.2 Proposed structure (non-divergent)

1. **Uniform loop bound.** Compute a per-frame scalar on the CPU
   (`maxStepsFrame = ceil(maxChordMM / sampleDistanceMM)`, `maxChordMM` = longest
   ray-box chord for the current camera, cheap from the 8 box corners) and pass
   it as a new uniform. Every fragment runs the same iteration count, so SIMT
   lanes stay locked.
   - To avoid the ~30 % wasted fetches of a pure frame-max loop, use a
     **hybrid**: a uniform main loop at the frame-average chord (~659) with
     predicated composition, plus a short divergent tail loop for the ~15-20 %
     of rays longer than that. (Estimated ~47.5 M fetches, mostly non-divergent
     -> ~64-75 ms with `clamp_to_zero`.)
2. **Replace all four `break`s with predication.** Keep the texture fetch
   unconditional; gate only the accumulation:
   ```metal
   bool active = inside && (accumulatedOpacity < 1.0h - 1.0h/255.0h);
   accumulatedColor += active ? weight * (sampleColor * sampleOpacity) : 0.0h;
   accumulatedOpacity += active ? weight * sampleOpacity : 0.0h;
   ```
   Preserve the CTP directional bounds test and the `seenInBounds` grazing-ray
   clamp semantics exactly (these are GL-parity requirements, see the comments
   at MetalShaders.metal:4098-4113).
3. **Drop the manual one-step prefetch**; with no data-dependent break the
   compiler/hardware can keep the texture pipeline full on its own.
4. **Swap the volume sampler to `clamp_to_zero`** (a new constexpr sampler);
   keep the existing in-shader `clamp(texLocalPos, 0, 1)` so edge behavior is
   unchanged. Worth ~8 ms.

### 4.3 Verification

- Image parity: render before/after with the same env config and diff via
  `visual_compare/`; the DICOM scene must be pixel-identical (or within the
  existing float-epsilon tolerance used for the GL comparison).
- Time: `--bench --backend metal` must drop from ~103 ms toward ~64-75 ms.
- All five blend modes, the independent multi-component path, the rectilinear
  and minmax paths must still be exercised (the restructure touches the shared
  loop).

---

## 5. Reproduction instructions

### 5.1 Generate the DICOM volume texture

```
cd Rendering/Metal/PerformanceInvestigation
python3 dump_dicom.py              # needs numpy + pydicom
# writes ./dicom.u8 (512x512x1794, raw u8). Override inputs with:
#   DICOM_SRC=/path/to/series  DICOM_OUT=/path/to/dicom.u8
```

The app's texture is `castToU8` with shift +1024 and scale 255/4095 (see
`TestMetalScenes.h`); `dump_dicom.py` replicates that mapping.

### 5.2 Capture the per-frame uniform/buffer set (one-time)

Add the following **temporary** diagnostics to
`Rendering/Metal/vtkMetalGPUVolumeRayCastMapper.mm` (they are intentionally not
committed - revert after capturing):

```objc
// near the vertex buffer build (~line 5455):
if (getenv("VTK_METAL_TEST_DUMP_UNIFORMS") && !getenv("VTK_METAL_TEST_DUMP_VERT_DONE"))
{
  setenv("VTK_METAL_TEST_DUMP_VERT_DONE", "1", 1);
  FILE* f = fopen("/tmp/app_verts.bin", "wb");
  fwrite(vertices.data(), 1, vertices.size() * sizeof(float), f);
  fclose(f);
  f = fopen("/tmp/app_idxs.bin", "wb");
  fwrite(indices.data(), 1, indices.size() * sizeof(unsigned int), f);
  fclose(f);
}

// near the uniform buffer update (~line 7397):
if (getenv("VTK_METAL_TEST_DUMP_UNIFORMS") && !getenv("VTK_METAL_TEST_DUMP_UNIFORMS_DONE"))
{
  setenv("VTK_METAL_TEST_DUMP_UNIFORMS_DONE", "1", 1);
  FILE* f = fopen("/tmp/app_uniforms.bin", "wb");
  fwrite(&uniforms, 1, sizeof(uniforms), f);
  fclose(f);
  PerBlockData pbdTmp = {};
  BuildPerBlockData(pbdTmp, &uniforms);
  f = fopen("/tmp/app_pbd.bin", "wb");
  fwrite(&pbdTmp, 1, sizeof(pbdTmp), f);
  fclose(f);
  f = fopen("/tmp/app_light.bin", "wb");
  fwrite(&lightUniforms, 1, sizeof(lightUniforms), f);
  fclose(f);
}
```

Rebuild, then run the test binary once (any GL or Metal frame) with
`VTK_METAL_TEST_DUMP_UNIFORMS=1`; this writes `/tmp/app_uniforms.bin`,
`/tmp/app_pbd.bin`, `/tmp/app_light.bin`, `/tmp/app_verts.bin`,
`/tmp/app_idxs.bin`.

### 5.3 Build and run the probe

```
cd Rendering/Metal/PerformanceInvestigation
./build_probe.sh          # produces ./probe7b
PROBE_BASE=$(pwd) ./probe7b <variant>        # variants 0..19
PROBE_BASE=$(pwd) PROBE_FIXED_N=659 ./probe7b 7
PROBE_SAMPLE_DISTANCE_MM=0.5 ./probe7b 0     # override sample distance
```

- `PROBE_BASE` defaults to the executable's directory (used for the .metal
  sources and `dicom.u8`).
- `PROBE_FIXED_N` overrides the fixed iteration count for fixed-count variants.
- Variants and their meanings (see `probe7_extra.metal`):

| v | function                  | purpose |
|---|---------------------------|---------|
| 0 | `fragment_volume_main`    | full real shader (reference) |
| 1 | `fragment_march_only`     | (not present - removed during experiments) |
| 2 | `fragment_march_no_fetch` | loop without volume fetch |
| 3 | `fragment_march_only_nearest` | divergent nearest march |
| 4 | `fragment_count_steps`    | counts iterations/pixels (use for the 46 M figure) |
| 5 | `fragment_debug_steps`    | per-pixel step histogram (avg 659, max ~850) |
| 6 | `fragment_march_linear_implicit` | divergent linear march, implicit LOD |
| 7 | `fragment_march_linear_fixedN` | linear, fixed count, clamp_to_edge |
| 8 | `fragment_march_nearest_fixedN` | nearest, fixed count |
| 9 | `fragment_march_linear_pipe2` | linear fixed, 2-deep software pipeline |
| 10| `fragment_march_linear_pipe3` | linear fixed, 3-deep pipeline |
| 11| `fragment_march_manual_trilinear` | 8 nearest taps + lerp |
| 12| `fragment_march_linear_clampZero` | linear fixed, clamp_to_zero sampler |
| 13| `fragment_march_linear_2Darray` | slice-lerp on a 2-D array texture |
| 14| `fragment_march_linear_repeat` | linear fixed, repeat sampler (timing only) |
| 15| `fragment_fixedpoint_linear` | **invalid** - CSE'd (see 2.6) |
| 16| `fragment_fixedpoint_nearest` | **invalid** - CSE'd (see 2.6) |
| 17| `fragment_march_linear_select` | fixed count, no break, predicated accumulate |
| 18| `fragment_march_xdir_linear` | **invalid** - cache-hot (see 2.6) |
| 19| `fragment_march_xdir_linear_counted` | v18 + iteration counter |

`probe7b` reads the DICOM volume from `dicom.u8` (in `PROBE_BASE`) and the app
dumps from `/tmp/app_*.bin`. The fragment shader sources are `probe6lib.metal`
(a copy of `MetalShaders.metal`) concatenated with `probe7_extra.metal`; the
vertex shader is `probe6vert.metal`. The `U`/`Block`/`Lights` mirror structs in
`probe7b.m` must match the mapper's layouts (offsets verified; `_Static_assert`s
at the top).

### 5.4 App benchmarks

```
./macos_metal_build.sh --resume --tests

VTK_METAL_TEST_SAMPLE_DISTANCE=0.5 VTK_METAL_TEST_JITTER=1 \
VTK_METAL_TEST_IMAGE_SAMPLE_DISTANCE=1.0 VTK_METAL_TEST_MINMAX=1 \
VTK_METAL_TEST_ACCEL=0 \
build_macos_metal/bin/vtkMetalGLVisualComparison \
  --bench --backend gl --scene DICOMVolume \
  --dicom /path/to/IMR/CTIMR/IMRToraceAddome \
  --frames 30 --reps 1
```

Swap `--backend gl` for `--backend metal` for the Metal side. `--perframe` gives
per-frame times. GL is timed with `glFinish`, Metal with `WaitForCompletion`
(TestMetalGLVisualComparison.cxx ~379-406), so the numbers are comparable.

---

## 6. Files

- `PERFORMANCE_INVESTIGATION.md` - this report.
- `probe7b.m` - probe harness (variants 0-19, env overrides).
- `probe7_extra.metal` - experiment fragment shaders.
- `probe6lib.metal` - copy of `Rendering/Metal/Shaders/MetalShaders.metal` at
  the time of the experiments.
- `probe6vert.metal` - volume vertex shader.
- `dump_dicom.py`, `parse_dicom.py` - DICOM series -> `dicom.u8` conversion.
- `minitest.metal` - minimal march loop for AIR disassembly.
- `build_probe.sh` - builds `probe7b`.

Note: `dicom.u8` (~470 MB) and the `/tmp/app_*.bin` capture are **not**
committed; regenerate per section 5.

---

## 7. Follow-up investigation (X-march orientation, camera dependence, ISA)

Later experiments marched along the volume's **longest axis** instead (the
transposed / permute-210 layout: 1794 samples along texture-Z becomes 1794
along texture-X), which reduced the memory-latency penalty of the long march
for **both** back-ends. All numbers in this section share one
camera/orientation/data set and the same sample count (~45.6 M); times are
per-frame GPU ms.

### 7.1 App-level nearest/linear isolation (X-march, 45.6 M samples)

| Back-end / mode | ms | ns/sample |
|---|---|---|
| GL nearest | 22.15 | 0.49 |
| GL linear | 25.55 | 0.56 |
| Metal nearest | 26.94 | 0.59 |
| Metal linear | 46.35 | 1.02 |

Interpretation:

- **Metal nearest ~= GL linear.** The entire 2x gap is the trilinear add-on:
  **+0.43 ns/sample in Metal vs +0.07 ns/sample in GL** on the same GPU.
- Confirmed in the probe with shader branches compiled out (fixed count):
  nearest 25.06 ms, linear 41.97 ms -> the filter cost is intrinsic to the
  sample operation, not a shader-branch artifact.

### 7.2 The 2x gap is real (not measurement)

- **GL iteration-count cap sweep** (rebuilt OpenGL2 lib per cap):
  50 -> 3.83 ms, 100 -> 5.44 ms, 659 -> 19.86 ms, uncapped -> 24.40 ms.
  GL genuinely iterates ~45 M samples at ~0.5-0.55 ns/sample.
- Images are **bit-identical** between back-ends (compare error 0), so both do
  the same samples with the same filter; the difference is purely cost/sample.

### 7.3 Camera dependence

Sweeping the camera azimuth (Metal X-layout vs GL Z-layout): Metal 28.1-59.9 ms,
GL 23.7-49.7 ms. Concrete points: az -60 -> Metal 44.7 vs GL 49.7
(Metal wins); az -120 -> Metal 53.6 vs GL 23.7 (GL wins 2.3x).

- **Both back-ends are camera-dependent**, slowing when rays march along
  texture-Z. Any reorientation (permute 210/021/120) drops Metal 101 -> ~44-51 ms
  and GL 48.6 -> ~25 ms.
- **No fixed layout wins for all cameras.** Orientation alone cannot deliver
  "Metal <= GL everywhere".

### 7.4 Sampler / substitute experiments at the better orientation

| Strategy | ms | note |
|---|---|---|
| v7 linear fixedN | 41.97 | Metal 3-D linear baseline |
| v8 nearest fixedN | 25.06 | filter add-on ~= 17 ms |
| v12 clampZero (divergent) | 37.60 | divergent real shader, clamp_to_zero |
| v13 2-D-array slice-lerp | 43.23 | no longer cheaper than 3-D (vs section 2.5) |
| v11 manual 8-tap trilinear | 52.90 | worse |
| v3 real nearest (divergent) | 32.34 | divergent nearest |

- At the better orientation the 2-D-array advantage seen in section 2.5
  disappears (43.23 vs 41.97 ms) - layout, not texture type, was the driver.
- App-level `clamp_to_zero` gave no win (X-march 45.02 vs 44.13;
  Z-march 101.03 vs 101.32) and was **reverted**; MetalShaders.metal is back to
  `clamp_to_edge`. The earlier probe-level clamp_to_zero gain (72.7 -> 64.3 ms)
  did not reproduce at app level.

### 7.5 ISA / AIR evidence (answers "can we emulate GL's operation?")

Compiling a minimal fragment shader with linear and nearest 3-D sampling and
disassembling the AIR (`xcrun -sdk macosx metal-objdump`) shows **both filters
emit the identical instruction stream: a single `air.sample_texture_3d.v4f32`
call**. The only difference is the sampler descriptor (filter bits, 0x2000).

- Metal's compiler does **not** expand the 3-D trilinear sample into multiple
  fetches/lerps at the AIR level - that hypothesis is disproven.
- The whole linear-vs-nearest cost is realized inside the GPU texture unit,
  below the ISA we can inspect; it is the same silicon GL uses.
- Combined with section 7.4 (every shader-side substitute is equal-or-worse),
  **there is no Metal texture op or shader construct that reproduces GL's
  cheaper trilinear.** GL's cost comes from its driver-internal texture
  representation/filtering path, which the Metal API cannot reach.

### 7.6 Updated conclusions

- Metal's raw path at the bench config is **~44-51 ms** at a favorable
  orientation vs GL's **~24-25 ms** for the identical march; the residual ~2x is
  the Metal 3-D linear filter cost (+0.43 ns/sample vs GL's +0.07 ns/sample).
- This cost is a hardware/texture-unit property (single op in both AIR and GL),
  not a shader-compiler artifact, and no Metal-side emulation exists.
- The image-exact levers that remain:
  1. **Orientation / layout** (march along the long axis): ~2x for both
     back-ends, but camera-robustness-limited (no fixed layout wins for all
     cameras; section 7.3).
  2. **Non-divergent march restructure** (section 4): recovers only the SIMT
     divergence portion; it does not touch the per-sample filter cost.
- The earlier 103 ms -> 72 ms divergence finding (section 3) still stands for
  the original orientation; the follow-up shows that once the layout is fixed
  the residual gap is the sample filter.

---

## 8. Self-contained minimal repro (no VTK)

`Rendering/Metal/PerformanceInvestigation/minimal_repro.mm` is a ~480-line
standalone Objective-C++ program (no VTK, no vtkModule) that recreates the
comparison with synthetic data:

- Same footprint: R8 512x512x1794 volume (470 MB) in both `GL_TEXTURE_3D` and
  `MTLTexture3D` (`R8Unorm`, mipLevel 1, clamp-to-edge, `Shared` storage).
- Minimal raymarch fragment shaders (GLSL 410 core / MSL) doing only the
  volume sample loop: 400x400 frags x 604 steps = 45.68 M samples/frame, no
  transfer function, no compositing, no divergence, no jitter.
- Axis-lockstep march in both Z and X directions, nearest/linear x both
  back-ends, 30 warmup + 120 timed frames, `glFinish` / `waitUntilCompleted`
  per frame.

### 8.1 Results (Apple M2, current build, 120 frames x 2 rounds)

| march | gl nearest | metal nearest | gl linear | metal linear |
|-------|-----------:|--------------:|----------:|-------------:|
| Z     | 2.40 ms    | 2.66 ms       | 10.27 ms  | 18.96 ms     |
| X     | 2.56 ms    | 4.10 ms       | 6.06 ms   | 13.60 ms     |

Linear-vs-nearest filter add-on:

| march | gl add-on | metal add-on | ratio |
|-------|----------:|-------------:|------:|
| Z     | +7.87 ms  | +16.30 ms    | 2.07x |
| X     | +3.50 ms  | +9.50 ms     | 2.71x |

Readbacks (~9.64M, avg px ~128) match on both back-ends for every config,
confirming all 604 texture samples execute; results are round-stable.

### 8.2 What this proves

- The Metal-vs-GL linear gap is **not a VTK artifact**: it reproduces in a
  trivial self-contained shader with perfect cache behavior, no divergence and
  no per-iteration ALU. Metal linear is ~1.85-2.2x GL linear in the app and
  the same in the repro; the filter add-on ratio (2.0-2.7x) matches.
- The nearest base gap is march-dependent (Z: Metal +0.01 ns, ~parity; X:
  Metal +0.03 ns, +60%), consistent with the app's layout-driven base gap
  (section 7): the texture representation costs only ~0.01-0.03 ns/sample in
  the best case, but the Metal filter adds ~2-2.7x GL's per-sample filter cost
  regardless of layout.
- The absolute numbers are ~5x faster than the app's same-config numbers
  because the repro shader has none of the app's per-iteration work (TF taps,
  compositing, bounds, minmax, jitter); the *relative* Metal/GL filter gap is
  unchanged, so the comparison is valid for isolating the texture-unit cost.

### 8.3 Build / run

```
clang++ -std=c++17 -fobjc-arc -O2 -DGL_SILENCE_DEPRECATION \
  -framework Metal -framework OpenGL -framework Foundation \
  minimal_repro.mm -o minimal_repro
./minimal_repro [frames]
```

### 8.4 Notes

- `replaceRegion` on a `MTLStorageModePrivate` 3-D texture with this size
  faults inside Apple's lossless texture compressor (`AGXMetalG14G
  ::AppleCompressionGen2::Compressor::compressMacroblock`, EXC_BAD_ACCESS).
  VTK avoids this path by writing the volume through a compute shader
  (vtkMetalGPUVolumeRayCastMapper.mm `CreateGlobalVolumeTexture`); the repro
  uses `Shared` storage to stay in the simple path.
- Timing uses host-clock per-frame with a per-frame GPU sync on both back-ends
  (identical to the app's GL `glFinish` protocol; Metal sync cost is identical
  across filter configs since only the sampler state differs).
