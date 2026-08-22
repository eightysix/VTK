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
>
> **Update (scheduling discovery):** **section 9** shows the Z-march half of
> that gap is a **Metal compiler scheduling artifact**, not a texture-unit
> property: co-compiling >= 4 extra volume samples in the shader (or using
> separate loops per filter path) drops Metal's 3-D linear Z-march from
> ~19 ms to ~5.8 ms - **faster than GL's 10.3 ms**. The X-march gap (~1.9x)
> is structural and not scheduling-fixable.

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

---

## 9. The 3-D linear Z-march gap was a Metal compiler scheduling artifact

Follow-up on the minimal repro (section 8). The repro's Z-march (march along
the volume's long axis) originally showed Metal 3-D linear ~2x slower than GL
(Metal ~19 ms vs GL ~10 ms, i.e. a 0.42 ns/sample filter add-on). Sweeping the
MSL shader structure shows this is **not** a hardware property: the same
`vol.sample(smp, coord)` call runs 3.3x faster depending on how the loop is
compiled.

### 9.1 Results (all on the identical R8 512x512x1794 volume, 45.68 M
samples/frame, Metal backend, Z-march linear, 120 frames)

| Variant | Structure | Metal linear Z |
|---|---|---|
| S1/HEAD | single bare loop `acc += vol.sample(...)` | 19.3 ms (0.42 ns/sample) |
| S6 | loop with in-loop `if strategy==-1/0` branch | 19.7 ms |
| S9 | `#pragma unroll 4` | 19.9 ms |
| S14 | two accumulators (even/odd) | 18.5 ms |
| S17 | manual software-prefetch (vNext/v) | 20.6 ms |
| S23 | `const bool linear` branch in loop | 18.8 ms |
| S25 | + 1 extra manual tap in an else path | 20.2 ms |
| S30_3 | + 3 extra taps in an else path | 11.2 ms (partial) |
| S30_4 | + 4 extra taps in an else path | 5.86 ms |
| S24 | + 8 extra taps (no lerps) in an else path | 5.8 ms |
| S7 | full strategy chain incl. manual 8-tap path | 5.87 ms |
| **S29** | **three separate loops, branch hoisted out** | **5.74-5.92 ms (0.06 ns/sample)** |
| S33 | S29 + explicit `level(0.0)` on every sample | 5.67-5.79 ms |
| S34 | single bare loop + explicit `level(0.0)` | 18.5-18.6 ms (still slow) |
| S35 | S29 compiled with `fastMathEnabled=NO` | 5.79-5.85 ms |

GL linear Z in the same runs: **10.26 ms**. So with the S29 structure Metal's
3-D linear Z-march is **~1.8x faster than GL** (gap closed and inverted).

Readbacks are identical across every variant, so all 604 samples execute and
the output is the same.

### 9.2 Mechanism

- The bare accumulation loop compiles to a schedule that **serializes** each
  texture fetch (fetch latency exposed per iteration -> 0.42 ns/sample).
- When the same function co-compiles >= 4 independent extra `vol.sample`
  calls (another loop or a dead path), the compiler emits a schedule that
  **overlaps** the fetches (0.06 ns/sample).
- Not explainable by unrolling, prefetching, accumulator splitting, or
  branch hoisting alone - it is the co-compilation of >= ~4 extra samples in
  the same shader that flips the backend scheduling.
- Data entropy is irrelevant: random/gradient/zero volumes measure the same
  (section 8 follow-up), ruling out the lossless-compression hypothesis.
- Explicit-LOD is irrelevant: adding `level(0.0)` to every `vol.sample` call
  (S33) changes nothing. S29 linear Z stays ~5.7-5.8 ms with or without it,
  and the **single-loop bare shader stays slow (18.5-18.6 ms) even with
  explicit `level(0.0)`** - so the MSL spec's implicit-LOD-derivative rule for
  fragment functions is not the trigger. The serialization is purely a
  fetch-scheduling artifact of a lone sample loop.
- Fast-math/reassociation is irrelevant: `fastMathEnabled=NO`
  (`-fno-fast-math`, the spec's "safe" math mode) leaves S29 linear Z at
  5.79-5.85 ms and the bare loop at ~18.2-18.3 ms. Only minor side-benefit:
  Metal nearest X improves ~7% (3.2 -> 2.97 ms).
- The X-march gap is **not** fixed by this: Metal linear X ~11.3 ms vs GL
  ~6.0 ms (1.9x) in all scheduling variants. A dual-accumulator loop helps X
  (11.7 -> 8.4 ms) but re-breaks Z (5.8 -> 22.2 ms). The X gap is structural
  (3-D texture tiling across the XY plane), not scheduling.

### 9.3 Implication for the app

- The app's default camera marches roughly along the volume's long axis, the
  analog of the repro's Z-march. If the app shader's volume-sampling loop can
  be compiled in the fast regime, that filter component could drop from ~0.42
  to ~0.06 ns/sample.
- Caveat: the app's total per-sample cost is ~2.3 ns at the bench config,
  dominated by TF lookups, compositing, divergence and min/max - not the
  volume filter. Closing the filter component does not by itself close the
  app gap; it must be combined with the section-4 divergence restructure.
- The exact S29 trick (separate loop per filter path, or any co-compiled
  multi-sample path) must be verified inside the real shader; the effect is
  backend-scheduling-dependent and may not transfer verbatim.

### 9.4 Driver bug found along the way

`replaceRegion` on a `MTLStorageModePrivate` 3-D texture of this size faults
inside Apple's lossless texture compressor
(`AGXMetalG14G::AppleCompressionGen2::Compressor::compressMacroblock`).
VTK never hits this because the volume is written through a compute shader
(vtkMetalGPUVolumeRayCastMapper.mm `CreateGlobalVolumeTexture`). The repro
uses `Shared` storage to stay on the simple path.

---

## 10. Runtime verification: app volume format, sample count, and the "GL app
faster than the bare-fetch microbench" anomaly

Instrumented the real GL app (`--scene DICOMVolume`, bench config) with an
on-GPU iteration counter (`VTK_METAL_TEST_GL_ITER`, encoded as
`g_currentT/4096` in the R channel of a one-time framebuffer dump) and a
volume-texture introspection dump (`VTK_METAL_TEST_DUMP_VOLTEX`).

### 10.1 The app's volume texture is plain GL_R8, not 16-bit

At runtime: `scalarType=3` (unsigned char), `ncomp=1`, requested
`internalFormat=0x8229`, realized `0x8229`, `Rbits=8`, dims 512x512x1794.
The earlier suspicion that the app used a 16-bit / Apple-private format was a
**print artifact**: `vtkOStreamWrapper` has no `operator<<` overload for the
`std::hex`/`std::dec` manipulators, so they are converted to `bool` (printed
as `"1"`) and all subsequent integers print in decimal with a spurious digit
glued on (`0x8229` -> `0x1333211` in the log). Using `snprintf` gives clean
`0x8229`. The app's volume is R8, identical to `gl_gap`'s microbenchmark.

### 10.2 The 659 avg / ~46 M sample count is confirmed

The GL_ITER bench-frame dump decodes to `avg 659.0` iterations over 69,199
marching pixels (~46.0 M samples), matching the probe-based 46,049,650 figure
(section 2). The earlier "avg 200" reading came from a non-bench first frame
and was wrong. The report's per-sample figures stand.

### 10.3 Standalone GL harness reproduces the app's shader cost

`minimal_gap/gl_app_shader.m` runs the app's exact composed shaders + dumped
uniforms (window 400x400, off-axis camera, R8 synthetic volume, linear filter,
TF/compositing path enabled, low-alpha TF so the loop runs to the geometry
limit). Results (identical measurement to the app: `glFinish` per frame):

| Shader (GL, R8 volume) | avg iters | ms | ns/sample |
|---|---|---|---|
| **app** (real window, real CT data) | 659 x 69,199 px | 48.97 | 1.06 |
| **harness** (app shader + uniforms, FBO, synthetic ramp) | 484 x 69,831 px | 28.7 | 0.85 |
| **gl_gap** (bare volume fetch, same camera) | 659 x 69,861 px | 62.1 | 1.35 |

`-O3` vs `-O0` host code changes nothing (GPU-bound; verified 43.07 vs
43.39 ms on the R16 harness build). The harness confirms the app's GL
per-sample cost (~1 ns/sample) is real and reproducible outside the app; the
remaining app-vs-harness gap (1.06 vs 0.85) is the real CT data + onscreen
target, not the shader.

### 10.4 The "app faster than the bare-fetch microbench" anomaly is a GL-side
shader-structure artifact

The bare-fetch loop (`acc = max(acc, texture(...).r)` + a few FMAs per
iteration, `noFetch` drops it 62 ms -> 1.44 ms, so it is ~97% fetch-bound) is
*slower* per sample (1.35 ns) than the app's full shader (1.06 ns) or the
harness (0.85 ns), even though the full shader does strictly more work (an
extra 2-D opacity-TF fetch, scale/bias, compositing branch per iteration).
This is the same effect the Metal side hit in section 9: a lone volume-sample
loop schedules poorly on this driver (per-iteration fetch latency exposed);
interleaving extra fetch/ALU work lets the 3-D fetches pipeline. So the GL
microbenchmark is a *valid same-pattern GL-vs-Metal comparison* but **not** a
valid absolute baseline for the app; the app was never magically cheap per
sample. The 6x spread between `gl_gap` (1.35) and the section-9 repro's
axis-aligned GL Z-march (0.22 ns/sample) is the access pattern: axis-aligned
rays through a column of texels are fully coherent, the app's off-axis camera
rays are not.

### 10.5 Implication for the M/GL comparison

- Bare-fetch raw 3-D linear sample (same pattern, same data, same camera):
  GL 1.35 vs Metal 1.92 ns/sample (~1.4x) - but both numbers are inflated by
  the lone-loop scheduling artifact.
- Real app shader: GL 1.06 vs Metal 2.24 ns/sample (**2.1x**). The Metal
  penalty over GL is concentrated in the divergent data-dependent loop +
  TF/composite work, consistent with sections 4 and 9.
- The app's GL side cannot be optimized further by "simpler" sampling; the
  actionable levers remain the Metal divergence restructure (section 4) and
  the S29-style scheduling fix verified inside the real shader (section 9.3).

### 10.6 Instrumentation reference (kept in the tree for future investigations)

The debug hooks below are **deliberately left in place**; see
`minimal_gap/README.md` for the standalone harnesses that consume their
output. All are env-gated and one-shot (first frame/load only), so they are
inert unless the variable is set:

| Env var | Location | Effect |
|---|---|---|
| `VTK_METAL_TEST_GL_ITER` | `vtkOpenGLGPUVolumeRayCastMapper.cxx` (GetShaderTemplate Substitute + RenderSingleInput FBO dump) | rewrites `gl_FragData[0] = g_fragColor;` to `vec4(g_currentT/4096.0, 0, 0, 1)`, dumps the first frame to `/tmp/app_gl_iter.ppm` (R channel = iterations). Decode: `R * 4096/255`. |
| `VTK_METAL_TEST_DUMP_VOLTEX` | `vtkVolumeTexture.cxx` | one-shot runtime probe of the volume texture: requested/realized internal format, dims, scalar type, R/G/A bit sizes. |
| `VTK_METAL_TEST_DUMP_UNIFORMS` | `vtkOpenGLGPUVolumeRayCastMapper.cxx` | dumps all active uniforms to `/tmp/app_gl_uniforms.txt` (harness's constants). |
| `VTK_METAL_TEST_GL_NOTF` | `vtkVolumeShaderComposer.h` (BaseImplementation) | replaces the TF-fetch/computeColor path with a fixed color+alpha to isolate the TF-fetch cost. |
| (always on) | `vtkOpenGLGPUVolumeRayCastMapper.cxx` BuildShader | dumps the composed shaders to `/tmp/app_gl_frag.glsl` + `/tmp/app_gl_vert.glsl` (harness inputs). |

Caution: when printing GL enums through `vtkErrorMacro`, `std::hex` is
silently dropped by `vtkOStreamWrapper` (no manipulator overload), producing
corrupted hex values; use `snprintf` (see `DBG volume tex`).

## 11. Metal microbenchmark optimization hypotheses: all four ruled out

Four hypotheses for the Metal side of the gap were tested in `metal_gap.m`
(interleaved A/B on the M2 MBA, fresh short runs to cancel thermal drift):
half-precision (`texture3d<half>`) sampling, depth store-action parity,
`fastMathEnabled`, and a compute-kernel rewrite. Full tables and run syntax
are in `minimal_gap/README.md`.

| variant | avg frame |
|---|---|
| fragment float depth+Store (baseline) | 84-87 ms |
| fragment half sampler | 83-86 ms |
| fragment float, depth DontCare / no depth | 85-87 ms |
| compute kernel float / half | 95-99 ms |

Results, per hypothesis:

1. **Half sampling: no measurable effect (~1-2%, noise).** GL's 16-bit fast
   paths are not the source of the ratio.
2. **Depth store action: no effect.** The pass never writes depth; TBDR keeps
   it in tile memory. The `gl_gap`(no-depth)/`metal_gap`(depth+Store)
   asymmetry was real but measurement-irrelevant.
3. **Compute kernel: ~13% slower**, not faster (98 vs 86 ms). On Apple TBDR
   the fragment pipeline is a win for a fullscreen march; a compute rewrite
   would regress the app's Metal backend.
4. **`fastMathEnabled=NO` is reproducibly ~8% FASTER in the bare-fetch
   harness** (78-80 ms vs 85-87 ms, five interleaved 30-frame rounds). The MSL
   default is `YES`, and the old "no-op by construction" claim was wrong:
   toggling the flag off changes the compiled shader and measurably helps the
   tight fetch loop.
   **The gain does NOT transfer to the full app shader.** The app A/B
   (`VTK_METAL_TEST_FAST_MATH=0` env knob added to
   `vtkMetalGPUVolumeRayCastMapper.mm:1608`, five interleaved 30-frame/3-rep
   benches) was 99.2-102.1 ms vs 100.4-101.1 ms - within noise and slightly
   slower on average. The ~8% is specific to the bare-fetch shader, so it is
   **not** an app-level lever; the knob is left in place (env-gated, no
   behavior change when unset) for future A/Bs.

Concluded: the ~1.3x GL/Metal bare-fetch ratio is intrinsic to Metal's
per-sample fetch/loop throughput on this device. The actionable Metal levers
remain the divergence restructure (section 4) and the S29-style scheduling fix
(section 9.3), not precision, depth, math flags, or dispatch model.

## 12. Harness parity: Metal and GL are exactly 1:1

The earlier avgIter comparison (Metal 701.7 vs GL/fp64 659.2, "Metal over-runs
~4-7% per ray" and "Metal under-samples, meanB 0.210 vs 0.283") was an
**artifact of a readback channel bug in `metal_gap.m`**, not a real harness
difference.

Cause: `metal_gap.m`'s render target is `MTLPixelFormatBGRA8Unorm`, so memory
byte order is B,G,R,A while the shader writes `float4(R=iterLow, G=iterHigh,
B=acc)`. The old stats loop treated byte0 as the iteration low byte (it is
actually the `acc`/B channel), inflating the reported avgIter. GL's `RGBA8`
readback has no such trap.

Verified with the correct decode (`iter = byte2 + 256*byte1`), matching
1-frame runs, and the `diag` mode (`metal_gap.m ... diag=1`, shader writes
`rayDir*0.5+0.5`; note the same BGRA8 swap applies when reading the PPM):

- avgIter: Metal **287.8** = GL 287.8 = fp64 287.8 (659.2 per nonzero pixel).
- footprint: 69,861 nonzero pixels on both sides.
- per-pixel iteration counts: **159,998 / 160,000 identical** after the
  GL bottom-first -> Metal top-first row flip; the 2 remaining pixels differ
  by exactly 1 (silhouette boundary).
- per-pixel sample values on marched pixels: identical (mean |d| = 0.00).

So the harnesses trace identical rays with identical step counts and identical
fetches; the residual ~30 ms (57.8 ms GL vs 84-87 ms Metal) is pure GPU
execution cost for the same work, and the fastMath result in section 11 is not
a measurement artifact.

## 13. Orientation-matched GL baseline and the non-PoT-depth test

Two follow-up tests prompted by an external review of section 12:

**GL row order matters for GL's absolute time (~10%).** `gl_gap`'s `flipY`
only permutes which pixel traces which ray, so the ray multiset (and avgIter,
287.8) is invariant. But the *spatial coherence* of adjacent pixels' marches
through the 448 MB volume changes, and GL pays ~6 ms (62.1 vs 67.9 ms,
5 interleaved rounds) when tracing Metal's row order. Per-pixel comparison
confirms `flipY=1` reproduces Metal's exact orientation (159,990/160,000
pixels identical; the row-flipped mapping is garbage). The fair, app-normalized
comparison is therefore **GL `flipY=1` ~67.9 ms vs Metal ~86.3 ms = ~1.27x /
~18 ms**, narrower than the ~1.4x implied by the old `flipY=0` baseline but
still substantial. All Metal-side experiment conclusions in section 11 are
unaffected (they never touch the GL reference).

**Non-power-of-two depth (D=1794) is not the cause.** The harness uses the
app's real `kD=1794`. Raising it to `kD=2048` (PoT) keeps the same-orientation
ratio essentially unchanged: GL 78.6 ms vs Metal 100.2 ms = **1.27x**, matching
the 1.27x at 1794. Both backends scale proportionally with the +14% footprint
(GLM +16%, Metal +16%), so 3-D tiling / PoT padding does not explain the gap.

Together with section 11's null results (half sampler, LOD, depth action,
compute, fastMath-in-app), the remaining candidates for the fixed ~1.27x
per-sample factor are Metal-side instruction scheduling / divergence handling
(sections 4 and 9), not memory bandwidth: GL and Metal share the same L2 and
Unified Memory on the M2, so a DRAM-bound workload would show identical times,
and nearest-sampling throughput is already at parity (GL 0.97 vs Metal
1.01 ns/sample).

## 14. The gap is a fetch-latency-hiding deficit: 8x unroll makes Metal 1.46x FASTER than GL

New `pipeline` variants in `metal_gap.m` (argv[9]) restructure the march inner
loop without changing its semantics. All variants are parity-exact vs the
baseline (avgIter 287.8; the 8x-unrolled version's iteration counts differ
from baseline on only 1,531/160,000 silhouette pixels by +/-1, sample values
identical - the `i+n <= steps` chunk bound guarantees chunks never cross
`tExit`, so no over-march).

30-frame interleaved rounds on the M2 MBA:

| variant | avg ms | ns/sample | note |
|---|---|---|---|
| GL flipY=1 (linear) | 67.1 | 1.47 | orientation-matched reference |
| Metal baseline (serial fetch->max) | 90.0 | 1.97 | the historical "Metal is slower" number |
| Metal prefetch-2 (loop-carried `cur`) | 89.7 | 1.97 | fetch-ahead-then-consume, **slower** |
| Metal 2x unroll | 72.1 | 1.58 | |
| Metal **4x unroll** | 58.5 | 1.28 | already faster than GL |
| Metal **8x unroll** | **45.8** | **1.00** | **1.46x faster than GL** |
| Metal 16x unroll | 51.6 | 1.13 | register-pressure / occupancy regression |

Interpretation:

1. **The harness gap was a scheduling deficit, not a hardware cost.** The
   monotonic 1x->2x->4x->8x speedup (90 -> 45.8 ms) means the serial
   fetch->max chain exposed the full 3-D-linear issue latency; N independent
   in-flight fetches hide it. GL's driver compiler already pipelines its loop
   (hence GL at 1.47 ns/sample vs Metal's exposed 1.97).
2. **The prefetch-with-loop-carried-dependency pattern is a trap**: fetching
   step i+1 into a register consumed next iteration keeps only one fetch in
   flight (the carried dependency serializes it) and adds register traffic -
   it is ~2 ms *slower* than baseline, not faster.
3. **8x is the sweet spot on this device**; 16x regresses (register pressure
   cuts occupancy / latency hiding).

## 15. The 8x unroll transfers to the app shader: variant 6 is the new default

The app's `marchVolumeUnified` loop (~640 lines) is not a bare fetch: it
already carries a loop-carried prefetch-ahead-1 (`prefetchScalar`/`prefetchMask`
- the exact pattern the harness measured as slower) plus min-max/empty-cell
skip, gradient ILP, TF lookup, lighting, and latched exits. A literal 8x unroll
of the full body was therefore rejected; instead a dedicated `fc_marchVariant`
experiment restructures the march as a chunked N-way unroll: 8 independent
`sampleVolumeScalar` fetches per batch (all issued back-to-back, hiding 3-D-
linear latency), latched exits, and no loop-carried dependencies.

The env selector `VTK_METAL_TEST_MARCH_VARIANT` bakes the variant into the
feature mask (bits 24-27) so each variant gets its own specialized pipeline.
Variant 6 = 8x unroll, variant 7 = 4x unroll. The unrolled branch is guarded to
the simplified composite config (no shading/gradient-op/min-max/cropping/mask/
rectilinear/2D-TF/independent-path); other configs fall through to the baseline
loop.

Measured on the DICOM app benchmark (`vtkMetalGLVisualComparison --bench
--scene DICOMVolume`, 30 frames x 3 runs x 3 interleaved rounds, M2 MBA,
sample-distance 0.5, jitter on, min-max accel disabled for GL parity - the
DICOM scene has no `ShadeOn`, so the timed workload is simplified composite):

| variant | round 1 | round 2 | round 3 | mean |
|---|---|---|---|---|
| 0 (baseline loop) | 107.14 | 103.12 | 104.77 | **105.0** ms |
| 7 (4x unroll) | 87.00 | 89.22 | 88.95 | **88.4** ms |
| 6 (**8x unroll**) | 82.69 | 81.65 | 79.40 | **81.2** ms |

Variant 6 is **1.29x faster than baseline** (105.0 -> 81.2 ms) on the app, but
the app is *still* slower than GL: same-scene GL is 49.3 ms (48.73-49.91 across
two interleaved rounds) vs Metal 79-81 ms - a residual **1.6x app gap** even
after the unroll, in the opposite direction of the minimum-gap repro (where
8x-unrolled Metal beat GL by 1.46x). GL+Metal image comparison (`--scene
DICOMVolume` with both backends enabled, vtkTesting-style thresholded error)
reports **0.000 for all three variants** - byte-identical output within
tolerance (the ~741 raw `vtkImageDifference` error is the pre-existing backend
difference, identical across variants). Note: running the comparison with
`--backend metal` only *skips* the diff (TestMetalGLVisualComparison.cxx:599),
so "worst thresholded error: 0" is vacuous unless both backends render.

**Conclusion**: the harness's 8x-unroll win does transfer to the real app
shader. Variant 6 was the default at this point (`VolumeMarchVariant()` returned
6 unless `VTK_METAL_TEST_MARCH_VARIANT` is set); variants remain selectable for
A/B. Variant 8 (harness-style scheduling, section 17) later superseded 6 as the
default, cutting the gap from 1.6x to 1.27x.
The **residual app gap (Metal 1.6x slower than GL despite the harness showing
Metal 1.46x faster)** is the next question: the minimum-gap repro (45.8 vs
67.1 ms) models the bare-fetch march with the app's real camera/uniforms, so
the extra ~2.3x of app-side Metal cost lives in what the harness does not model
- pipeline/PSO setup, per-block dispatch overhead, offscreen render-target
handling, the fullscreen quad vs the app's actual ray setup, or
shading/min-max/2D-TF-enabled configs.

## 16. Correction: the decomp "GL parity" was a mis-attributed MIP probe

Earlier drafts claimed `fragment_march_real_decomp` (probe v22) hit GL parity
at 44-47 ms. That was **wrong**: the 47 ms measurement belonged to probe v22 =
`fragment_march_nearest_clamp`, a trivial *nearest-clamp MIP* probe, not the
decomp. The real decomp (`fragment_march_real_decomp`, probe v20) runs the same
lean prefetch-1 loop (prefetch-ahead-1, bounds-clamp+refetch, TF, composite,
direct breaks) at **~104 ms** in the same harness - *slower* than the batch-8
unroll, not at GL parity. The "45 ms" figure was a fragment-index mix-up between
v20 and v22 (fragNames array positions in probe7b.m). The conclusions of the
original section 16 (batch-8 vs decomp structural gap, fc_probeSlim context
blocker, dead-array trap) were all built on that phantom baseline and are
invalidated.

### 16.1 Corrected measurements (probe7b, BGRA8 direct, no plumbing)

| probe | loop structure | time |
|---|---|---|
| v20 `fragment_march_real_decomp` | prefetch-1, lean consume, breaks | **104 ms** |
| v22 `fragment_march_nearest_clamp` | trivial MIP (nearest/clamp) | **47 ms** (phantom) |
| v25 `fragment_march_decomp_unrolled` | batch-8 + unrolled machinery | 76 ms |
| v34 `fragment_march_phase_batch_sched` | harness-style (positions-first, all fetches back-to-back, single advance, scalar tail) | **56 ms** |
| real mv=6 | batch-8, full `fragment_volume_main` | 75 ms |

### 16.2 What is actually true

1. **The batch-8 unrolled consume is not the problem.** v25 (batch-8, full
   machinery) = 76 ms beats the lean prefetch-1 decomp (v20) = 104 ms. The
   unrolled structure's ~30 ms advantage over the baseline loop is real; the
   prefetch-1 "pipelining win" was an artifact of the mis-attributed MIP probe.
2. **The `float bs[8]` array is not a 2.3x trap in general.** The dead-array
   probes (v27/v28, ~104 ms) showed a penalty only in the prefetch-1 *context*
   where the array forces local memory; in the batch-8 context the array is
   register-scalarized (scalar batch v33 vs array batch v32: 70 vs 83 ms - the
   array costs ~13 ms there, but the production function's register pressure
   already scalarizes it, so replacing it gains nothing in-app).
3. **The production fragment's context matters, but only via scheduling.** The
   decisive win is *not* the consume body - it is the loop **scheduling**
   harness (v34): compute all 8 positions first, issue all 8 volume fetches and
   all 8 TF fetches back-to-back (independent, in flight), composite serially,
   advance once per batch (`evalPoint += evalStep * 8`, a 1-op loop-carried
   chain instead of 8 serial adds), and use a scalar tail loop. v34 = 56 ms vs
   the batch-8 consume's 76 ms in the same shader context.

### 16.3 The v34 scheduling decomposition (probe7_extra.metal)

| probe | structure | time |
|---|---|---|
| v32 `fragment_march_phase_batch` | positions-first, but TF via runtime-indexed `co[8]` array | 83 ms |
| v33 `fragment_march_phase_batch_scalar` | positions-first, scalar s0..s7/c0..c7, phase-separated | 70 ms |
| v34 `fragment_march_phase_batch_sched` | v33 + single advance per batch, one break per batch, scalar tail | **56 ms** |

The v32->v33 delta (~13 ms) is the runtime-indexed `co[8]` array forcing local
memory; the v33->v34 delta (~14 ms) is the scheduling: one loop-carried advance
per batch and one break check per batch instead of per-sample bookkeeping.
Bounds-clamping is ~free (v34 mode 0 vs 16 within noise). Mode bits barely
matter; the *shape* is the win.

### 16.4 Direction

The production `fragment_volume_main` should adopt the v34 scheduling. Because
minmax/shading/gradient-op/renderToTexture are `function_constant`s, the lean
combo (all off) can compile to a dedicated harness-scheduled loop, with any
feature-enabled config falling back to the existing batch-8 consume (which is
correct for all flags). This is implemented as **fc_marchVariant 8** and
documented in section 17.

## 17. Variant 8: the harness-style scheduling is ported to the app shader (new default)

### 17.1 The port

`fc_marchVariant 8` in `MetalShaders.metal` ports probe v34's loop shape into
`fragment_volume_main`:

- `for (; i + unrollN <= steps; i += unrollN)` batch loop with **one** break
  check at the top (`currentT >= p.tEnd - 1e-6f`).
- All 8 positions computed first (`p0..p7 = evalPoint + evalStep*k`), all 8
  volume fetches back-to-back, all 8 TF fetches back-to-back - independent, in
  flight.
- Serial composite chain (`w0..w7`, no arrays, no runtime-indexed state).
- **One advance per batch**: `currentPoint += stepVec * 8` etc. - a 1-op
  loop-carried chain instead of 8 serial adds.
- Scalar tail loop for the `steps - i < unrollN` remainder.
- Bounds clamp preserved (mode 0 semantics: clamp-and-continue on the entry
  side, break after `seenInBounds`, matching the GL clamp-to-edge parity).

The branch is gated at compile time on the lean combo (`fc_marchVariant == 8 &&
!fc_minmax && !fc_shading && !fc_gradientOpacity && !fc_renderToTexture`); any
other feature combination keeps the existing batch-8 consume (variant 6/7
semantics), which is correct for all flags. So the fast path never compromises
minmax/shading/gradient-op/renderToTexture rendering.

### 17.2 In-app measurement (DICOM app benchmark, M2 MBA, interleaved A/B)

`vtkMetalGLVisualComparison --bench --scene DICOMVolume --dicom ... --frames 30
--reps 1`, sample-distance 0.5, jitter on, min-max accel disabled for GL
parity:

| variant | round 1 | round 2 | mean |
|---|---|---|---|
| GL | 46.97 | 48.89 | **47.9** ms |
| 6 (batch-8 consume) | 75.93 | 77.18 | **76.6** ms |
| 8 (**harness scheduling**) | 62.61 | 59.45 | **61.0** ms |

Thresholded GL-vs-Metal image error: **0.000** (byte-identical within
tolerance) for both variants. Variant 8 cuts the Metal-vs-GL gap from 1.60x to
**1.27x**, matching the probe's v34 result (56 ms probe + ~6 ms app plumbing).

**Note on the earlier flat result**: an initial in-app run of variant 8 showed
no improvement because the test binary had not been relinked (only the framework
was rebuilt with `--resume`, not `--tests`). The stale `bin/vtkMetalGLVisualComparison`
did not contain the new shader source. Always rebuild with `--resume --tests`
after editing `.metal` sources before benchmarking in-app.

### 17.3 Status

Variant 8 is now the app default (`VolumeMarchVariant()` returns 8;
`VTK_METAL_TEST_MARCH_VARIANT` still overrides for A/B). The residual 1.27x gap
vs GL is dominated by the app's ~6 ms plumbing plus the shader's remaining
per-sample machinery that the lean harness probe does not model. With minmax
acceleration enabled (the app's default when not forcing GL parity) the gap
would be even smaller; the numbers above use min-max off to match GL exactly.

## 18. Wider inline-address batches: probe w16/w32/w48/w64 and variant 9

### 18.1 Motivation

Section 17's variant 8 (8-wide batch) left the app at **1.27x** vs GL. The
user's thesis: the app is nowhere near the harness floor, and the remaining
deficit is fetch-latency hiding, not machinery. The harness's 16x unroll had
regressed (45.8 -> 51.6 ms) only because it kept 16 live `float3` position
registers alongside the results. Probe v36 tests the fix: pass each sample
address **inline** to `sampleVolumeScalar` so the address dies at issue and only
the 16 scalar results stay live - wider batches without the register spill.

### 18.2 Probe results (512x512x1794 R8, 400x400 RT, full DVR, mode 0)

Fragment functions in `probe7_extra.metal` (`fragment_march_phase_batch_w16`,
`..._w32`, `..._w48`, `..._w64`; probe7b variants 33/34/35/36), interleaved runs:

| batch width | r1 | r2 | r3 | mean |
|---|---|---|---|---|
| v34 w8 (sched, index 31) | 57.86 | 55.57 | 54.35 | **55.9** ms |
| v36 w16 (index 33) | 49.99 | 47.86 | 47.68 | **48.5** ms |
| v38 w32 (index 34) | 39.22 | 38.87 | 37.04 | **38.4** ms |
| v39 w48 (index 35) | 38.06 | 38.71 | 35.69 | **37.5** ms |
| v40 w64 (index 36) | 42.33 | 42.05 | 38.25 | **40.9** ms |

Notes:

- w16 already cuts ~7 ms off w8; w32 nearly halves the w8 time. w48 is the best
  (35.7-38.7 ms); w64 regresses (~2.4 ms) - register pressure returns.
- **w48's full-DVR time (37.5 ms) is below the harness's bare-fetch 8x-unroll
  time (45.8 ms)**: with a 48-wide batch the TF fetch + composite + clamp costs
  are fully hidden under the volume-fetch latency. This is direct evidence that
  the app was nowhere near a structural floor.
- A first w24 macro variant was measured but is **invalid**: the shared macro
  fetched/composited 32 samples while advancing 24 (double-counting). It is not
  reported here and was removed; w32/w48/w64 were written with exact sample
  counts.
- All w-fragments keep v34's semantics (bounds clamp-and-continue until
  `seenInBounds`, then break; `currentT`/`tTerminateMax`/opacity exits; scalar
  tail loop) so per-sample output is identical to w8.

### 18.3 The port: fc_marchVariant 9 (48-wide inline batch)

`MetalShaders.metal` now has, in the `fc_marchVariant >= 6` guard, an
`if (fc_marchVariant == 9 && !fc_shading && !fc_gradientOpacity &&
!fc_renderToTexture)` branch exactly like variant 8 but with `unrollN = 48` and
inline sample addresses (`sampleVolumeScalar(volumeTexture, evalPoint + evalStep
* k)`), plus a minmax lattice skip (section 19). Shading / gradient-opacity /
render-to-texture combinations fall through to the feature-complete batch-8
consume, so correctness is preserved for all flags. Variant 9 is the default
(`VolumeMarchVariant()` returns 9) and now carries the minmax acceleration that
the production app enables by default.

### 18.4 In-app measurement (DICOM app benchmark, M2 MBA, interleaved A/B)

| variant | r1 | r2 | r3 | mean | M/GL |
|---|---|---|---|---|---|
| GL | 51.34 / 51.04 / 48.01 / 48.35 | | | **49.7** ms | |
| 8 (harness 8-wide) | 66.20 / 65.74 / 60.22 / 60.96 | | | **63.3** ms | 1.25 |
| 9 (**48-wide inline**) | 53.05 / 51.83 / 50.55 / 49.80 | | | **51.3** ms | **1.03** |

(Each row: interleaved runs, 30 frames each. GL and Metal ms/f are from the same
bench run per round.)

Thresholded GL-vs-Metal image error stays **0.000** for variant 9 - the 48-wide
batch composites exactly the same samples as variant 8, so parity is preserved.

### 18.5 Status and residual gap

Variant 9 (lean, minmax off) closed the app gap from 1.25x to ~1.0x vs GL at
fine sample distance - see section 19 for the full 3x3 grid and the minmax +
adaptive-cap combination that makes Metal clearly beat GL everywhere.

## 19. Variant 9 + minmax acceleration: Metal clearly beats GL at every SD

### 19.1 Minmax lattice skip inside the 48-wide batch

The production app runs `UseMinMaxAcceleration` (occupancy lattice). Variant 9
now handles the minmax path itself instead of falling back to the broken
batch-8 consume (which took 150-580 ms). A fragment pre-pass walks the lattice
cells the ray will cover in its next batch: if every cell in the walk's extent
is empty (occupancy R8 `> 0.5`), the whole extent is skipped bit-neutrally;
otherwise the walk stops at the first solid cell and the batch dispatches after
it, so the first sampled voxel is always inside a solid cell. The walk extent
is `min(48, steps - i)` and runs *before* the width dispatch - an earlier
per-width walk over-marched past `tTerminateMax` into boundary voxels (error
up to 2.485 @SD4); the preamble walk keeps error exactly at the shared GL/Metal
baseline floor.

Lattice geometry: `ComputeMacrocellDownsample(sd, gpuMinmax)` = 2 for `sd<1.5`
else 4, so a cell is 2 voxels at SD 0.5 and 4 voxels at SD >= 1.5.

### 19.2 Adaptive batch-width cap (new uniform `MaxBatchWidth`)

A fixed 48-wide batch regressed at coarse SD: with the lattice skipping empty
cells, the remaining solid runs are short, so 48-wide batches waste up to 47
slots each. The CPU now sets a per-frame `MaxBatchWidth` uniform from the
sample distance (env override `VTK_METAL_TEST_MARCH_CAP`); the shader dispatch
chain {48, 32, 16, 8, 4, 2, 1} only allows widths at or below the cap:

| sample distance | cap |
|---|---|
| `< 2.0` (fine, SD 0.5) | 48 |
| `2.0 .. 3.0` | 16 |
| `>= 3.0` (coarse, SD 4) | 8 |

Sweeps (400x400, DICOM): SD 2.0 best cap 12-24 (~0.49-0.50 M/GL), SD 4.0 best
cap 4-12 (~0.53); the mapping above picks the middle of each band. The cap is
uniform across the warp (no lane divergence) and leaves the bit-exact output
untouched.

### 19.3 Full grid (M2 MBA, minmax ON, reps 3, ms/f and M/GL)

| size | SD | GL | Metal | M/GL | prev (cap-48) |
|---|---|---|---|---|---|
| 400 | 0.5 | 49.7 | 24.7 | **0.50** | 0.48 |
| 400 | 2.0 | 29.6 | 14.3 | **0.48** | 0.55 |
| 400 | 4.0 | 19.3 | 10.2 | **0.53** | 0.77 |
| 1024 | 0.5 | 62.9 | 51.2 | **0.81** | 0.85 |
| 1024 | 2.0 | 44.2 | 27.9 | **0.63** | 0.75 |
| 1024 | 4.0 | 37.7 | 21.4 | **0.57** | 0.83 |
| 2048 | 0.5 | 161.6 | 128.7 | **0.80** | 0.77 |
| 2048 | 2.0 | 65.8 | 54.9 | **0.83** | 0.86 |
| 2048 | 4.0 | 53.1 | 41.1 | **0.77** | 1.02 |

Metal wins GL at all nine cells (0.48-0.83); the coarse-SD regression from the
fixed 48-wide batch (up to 1.02x at 2048/SD4) is gone. The default production
path (no env vars) measures 0.52x @400/SD4 and 0.78x @2048/SD4.

Thresholded error stays at the shared GL/Metal baseline floor: **0.000 @ SD0.5,
0.080 @ SD2, 0.814 @ SD4** (identical for mv0 lean, mv0+minmax, and mv9+minmax;
the coarse-SD floor is a pre-existing GL-vs-Metal divergence, not caused by the
minmax work).

Residual: at coarse SD the mv0 baseline loop (per-cell granular skip) is still
~0.1 better than mv9 at 1024/2048 (mv0 0.45-0.68 vs mv9 0.57-0.77). The lean
raw march (minmax off) still loses to GL at coarse SD (1.12-1.74x) - the raw
path, unchanged; production uses minmax on.



====================================================================
RAW-PATH COARSE-SD LAG: MECHANISM NAILED + VALIDATED FIX (2026-08)
====================================================================
Context: the raw (minmax off) march at high res + high sample distance loses to
GL (2048/SD4 1.75x in-app). A standalone repro (minimal_gap/metal_gap.m +
gl_gap.m, pixel-verified parity) reproduced the lag exactly with noise data
(dataMode=1): 2048/SD4 Metal ~79ms vs GL ~44ms = 1.8x.

MECHANISM: cache / DRAM working-set capacity.
  Every discriminator is now tested:
  - Trilinear 8-texel span: REFUTED. filter::nearest keeps 1.63-1.84x.
  - Shader scheduling (gradients/unroll): REFUTED. lod0=0, half, unroll8 only
    reach 1.41x on noise and never close it.
  - Texture layout (3D vs 2D-array sliced): REFUTED. Sliced is strictly worse.
  - Lossless texture compression: REFUTED. Gap appears on incompressible noise
    (the opposite of compression); GL runs ~26 GB/s, far under DRAM bandwidth.
  - Working set vs cache: CONFIRMED. Shrinking the volume (volDiv, same sample
    count, avgIter 36.2 verified) collapses the gap:
      470MB: M/GL 1.80x   ->   58.7MB: 1.08-1.48x   ->   7.3MB: 0.95x (Metal wins)
    Same 151M samples cost Metal 79ms at full footprint but 4-8ms at
    cache-sized footprint (M2 SoC shared cache ~32-64MB); Metal degrades ~9x
    past capacity, GL only ~7x.

VALIDATED FIX: depth-sliced (slab) rendering.
  - Real slab model (per-pass t-range = 1/K of depth, K=8): Metal 8.6ms total
    vs 79ms full; GL 10.6 vs 44ms. Metal slab beats GL single-pass ~5x and GL
    slab by 1.2x.
  - Multi-pass accumulation prototype (maccum in metal_gap.m): 8 passes/frame
    into one RT, MTLBlendOperationMax. Correctness is now essentially
    PIXEL-EXACT (deterministic, 2048/SD4 noise): numSlabs=1 = 0 / 4,194,304
    mismatches; 8 slabs = 1 / 4,194,304 pixel differing by exactly +-1/255.
    Four bugs were found and fixed to get there:
      1. MSL string buffers (64 B slack) overflowed once the slab clamp +
         warmup text was inserted -> 8-slab SIGSEGV / infinite re-insertion
         hang (strstr restarted from the buffer head, never advancing past
         the insert, and the warmup contains its own needle).
      2. Aligning tStart via tStart + kStartF*stepSize (a fused mul-add)
         seeded the warmup ~1 ulp off the single-pass accumulated lattice ->
         ~4.9% of marched pixels differed (max 88/255). Fixed by seeding
         currentT/texLocal/evalPoint from the RAW tStart and reproducing the
         exact += stepSize sequence in a kPass warmup loop.
      3. Interior slabs' ceil-aligned tExit could overshoot the box exit and
         sample one index past single-pass's break. Fixed by breaking on
         min(tExit, tExitRaw); the per-slab index-set union now equals
         single-pass exactly (verified with an fp32 CPU simulator of the
         lattice/count/break logic).
      4. fastMath=1 is REQUIRED: with fastMath=0 the alignment disagrees on
         thousands of boundary rays.
    Performance at 2048/SD4 noise: 80.5ms single-pass -> 23.8ms 8-slab
    maccum = 3.4x, vs GL 44ms = 1.85x WIN. The kPass warmup costs ~7ms (it
    re-advances every pass's start along the lattice). Phase-2 targets: one
    uniform-driven PSO instead of 8 PSOs; kill the warmup by indexing
    positions as tStart + float(i)*stepSize (mul-add) in both single-pass and
    slab shaders, making exactness free.
  - unroll (pipeline=8) does NOT stack with slabs: the cache fix subsumes it.

====================================================================
APP IMPLEMENTATION: COMPOSITE SLAB TILING (2026-08, DONE)
====================================================================
The harness fix is now ported into vtkMetalGPUVolumeRayCastMapper as an
env-gated composite-compatible tiling (the app composites front-to-back, it
does not max-project, so the harness's max-blend maccum does not transfer).

Design:
  - Composite-compatible: premultiplied front-to-back `over` is associative, so
    each slab pass composites only its interval starting from transparent and
    the K passes are combined by the existing (ONE, ONE_MINUS_SRC_ALPHA)
    hardware blend on the direct path (DirectScreen/FullscreenDirect pipelines
    already carry exactly that blend). Result equals a single-pass composite up
    to fp rounding, with no extra combine pass.
  - Slab partition: per-fragment ray-length-fraction index ranges via shared
    ceilings, kStart = ceil(j*maxSteps/K), kEnd = ceil((j+1)*maxSteps/K). The
    union of per-slab index sets equals single-pass's [0, maxSteps) exactly
    (no skipped/duplicated samples, no camera/axis/order math, always
    front-to-back). The march is re-anchored to firstT + kStart*stepSize and
    maxSteps is re-set to kEnd-kStart.
  - Plumbing: PerBlockData gained a slabInfo float4; the march splits in a
    fc_slabMode-gated block (function_constant 30) placed after the variant-4/5
    overrides and dead-code-eliminated when clear (so numSlabs=1 compiles the
    old code bit-for-bit). Mapper: VolumeFeature_Slab (bit 28) decoded into
    fc_slabMode; static VolumeSlabCount() reads VTK_METAL_TEST_NUM_SLABS
    (originally default 8, clamped [1,32]; now default-off — unset = 1); the
    direct proxy-geometry and camera-inside
    fullscreen draws loop over K slab passes. Offscreen/RTT/grid-traversal/
    selection pipelines stay single-pass (no blend there), as do non-composite
    blend modes (max/min/avg/additive are not `over`-associative in one RT).
    Variants 4/5 (uniform frame-max bound) are excluded in the shader.

Measured (vtkMetalGLVisualComparison --bench, DICOM IMRToraceAddome, M2):

  | config                        | GL    | slabs=1 | slabs=8 | M/GL  |
  |-------------------------------|-------|---------|---------|-------|
  | 400x400, minmax on (default)  | 49.7  | 24.7    | 15.1    | 0.30  |
  | 1024x1024, SD2, minmax off    | 44.4  | 68.5    | 29.6    | 0.67  |
  | 2048x2048, SD2, minmax off    | 66.2  | 108.8   | 51.3    | 0.77  |
  | 1024x1024, SD4, minmax off    | 35.9  | 51.8    | 25.6    | 0.71  |
  | 2048x2048, SD4, minmax off    | 51.0  | 87.7    | 44.2    | 0.87  |

  Every previously Metal-losing regime (raw path, coarse SD, high res — the
  cache/DRAM working-set cells) flips from M/GL 1.35-1.75x to 0.67-0.89x.
  Thresholded GL-vs-Metal image error stays 0.000 in all configs; the absolute
  error (741-743) is unchanged, i.e. the slab output is as close to GL as the
  single-pass output was.

Correctness caveat (direct path only):
  - The direct path renders to an 8-bit BGRA drawable, so each slab's partial
    composite is quantized to 8-bit before the blend accumulates it. Metal-vs-
    Metal single-pass comparison shows up to 19/255 difference on ~8% of pixels
    (same order as the pre-existing single-pass-vs-GL spread; thresholded error
    unchanged). numSlabs=1 remains byte-identical to the pre-change build
    (verified 0 diff), and the harness's float-accumulation results were
    pixel-exact (1/4,194,304 pixel at K=8), confirming the 8-bit per-slab
    quantization — not the partition math — is the only new error source.

FUTURE LEADS (performance):
  - The harness reached ~5x (Metal slab 8.6ms vs GL single-pass 44ms) with a
    lean fma-only shader and a float slab RT; the app currently gets 1.1-3.3x.
    The residual gap is the full app shader machinery (TF sampling, minmax
    walk, lighting plumbing) plus the ~6ms app/framework plumbing overhead per
    frame (section 17.2).
  - Float accumulation RT (offscreen RGBA16Float + blend, then one present
    pass) would remove the per-slab 8-bit quantization (see caveat above) and
    is the cleanest path to byte-exact single-pass parity at K>1. The offscreen
    image-sampling path's machinery could be reused.
  - Z-position cost gradient (slab[0] fastest, slab[6] slowest, both backends)
    is a swizzle/memory-order property; front-to-back composite order is fixed,
    but a max-blend/MIP slab pass could choose a cheaper slab order.
  - Higher K: flat at 400x400 beyond 8; untested at 2048+ where the per-pass
    working set is larger — K=16/32 may extend the win at very high res.
  - minmax + slabs may stack further on very large volumes (the minmax walk
    already skips empty macrocells; slabs additionally shrink the per-pass
    working set). Not yet measured.
  - The phase-2 harness finding that indexing sample positions as
    tStart + float(i)*stepSize (mul-add) makes exactness free still applies;
    variant 9 already uses mul-add-style inline sampling, so the slab re-anchor
    is cheap (~no warmup in the app).

FUTURE LEADS (correctness):
  - Byte-exact single-pass parity at K>1 requires the float accumulation RT
    (above); with the 8-bit direct path the only mismatch is the per-slab
    quantization.
  - fastMath=1 is required for lattice alignment across slab boundaries
    (harness bug #4); the app shader already compiles with fast math enabled.
  - The residual absolute error (741-743, thresholded 0.000) is the fp16
    accumulator-vs-GL ordering spread; a full fp32 composite accumulator would
    close it at some throughput cost, independent of slab tiling.

## 20. Slab tiling is view-dependent: adaptive slab count

Composite slab tiling (`VTK_METAL_TEST_NUM_SLABS` > 1) splits each ray into N
ray-length-fraction index ranges and composites the partials back-to-front with
(ONE, ONE_MINUS_SRC_ALPHA) blending (bit-identical to single-pass up to fp
rounding). Section 19-era measurements showed 8 slabs winning decisively on the
oblique "angled" bench camera at 2048x2048 (mv0, minmax ON, accel OFF, DICOM
512x512x1794):

| view | SD | s1 ms/f | s8 ms/f | s8 vs s1 |
|---|---|---|---|---|
| oblique | 0.5 | 330.4 | 150.7 | **2.2x faster** |
| oblique | 4.0 | 102.5 | 38.9 | **2.6x faster** |

But the benefit is orientation-dependent: on the DICOM app's axis-aligned
(`ResetCamera`) views the per-pass working set is already cache-friendly and
tiling is pure pass-count overhead:

| view | SD | s1 ms/f | s8 ms/f | s8 vs s1 |
|---|---|---|---|---|
| coronal (camAxis=y) | 0.5 | 60.7 | 68.5 | 1.13x slower |
| coronal (camAxis=y) | 4.0 | 9.5 | 18.0 | **1.9x slower** |
| sagittal (camAxis=x) | 0.5 | 60.1 | 67.6 | 1.12x slower |
| sagittal (camAxis=x) | 4.0 | 9.5 | 19.2 | **2.0x slower** |
| axial (camAxis=z) | 0.5 | 103.7 | 96.5 | ~tie (7% faster) |
| axial (camAxis=z) | 4.0 | 19.1 | 19.4 | ~tie |

The discriminator is view-to-axis alignment, not ray length (coronal SD0.5 has
~902 max steps/ray and prefers s1; oblique SD4 has ~223 and prefers s8). Head-on
rays sweep the texture one coherent slice at a time, so the single pass is
cache-friendly; oblique rays fan across the texture and the depth-sliced passes
keep the working set resident.

### 20.1 Fix: adaptive slab count from the volume-space view direction

`ResolveNumSlabs()` in vtkMetalGPUVolumeRayCastMapper.mm picks the count per
frame from the max |dot| between the volume-space view direction (camera ->
box center, in `[0,1]` volume space) and the volume axes: aligned
(>= `VTK_METAL_TEST_SLAB_ALIGN`, default 0.95 ~ 18 deg) -> 1 slab, otherwise 8.
An explicit `VTK_METAL_TEST_NUM_SLABS` still wins (0 = adaptive; adaptive is
no longer the default — unset now means 1, slab tiling off).
The choice is purely a performance trade-off: every count composites
bit-identically, so the visual_compare thresholded error is unchanged (0).

| view | SD | s1 | s8 | adaptive |
|---|---|---|---|---|
| oblique | 0.5 | 330.4 | 150.7 | **148.4** |
| oblique | 4.0 | 102.5 | 38.9 | **38.1** |
| coronal (y) | 4.0 | 9.5 | 18.0 | **9.6** |
| sagittal (x) | 4.0 | 9.5 | 19.2 | **9.6** |
| axial (z) | 4.0 | 19.1 | 19.4 | **19.5** |

Adaptive never exceeds the best of s1/s8 beyond run noise (axial SD0.5 measures
88.4 vs 96.5 s8 best, within the ~22 ms run-to-run sigma of that cell). The app's
axis-aligned `ResetCamera` view now renders single-pass (no slab penalty), and
oblique/rotated views automatically fall back to the cache-resident tiling.

## 21. Fair raw matrix and the one remaining loss (2026-08-18)

### 21.1 GL has no minmax — the fair comparison is raw vs raw

The harness env vars only reach the Metal backend (TestMetalScenes.h:1233-1234),
and `Rendering/OpenGL2` contains **no** `UseMinMaxAcceleration` at all — the GL
backend has no min-max skipping, period. So minmax must be excluded from any
fair A/B: the comparison is raw march vs raw march, and minmax is a Metal-only
advantage (when it is on, Metal wins 0.62-0.79x at 2048/SD0.5 and 0.37-0.49x at
400 — §19).

### 21.2 Final raw matrix (DICOM app, 30 frames, battery, opt off, ACCEL=0/MINMAX=0, NUM_SLABS=1, jitter=1, single run)

| cell | GL | mv0 | mv9 |
|---|---|---|---|
| 2048, SD0.5 | 160.2 | 119.1 (0.74x) | **90.4 (0.56x win)** |
| 2048, SD4 | 53.1 | 69.8 (1.31x) | 72.8 (1.37x) |
| 400, SD0.5 | 50.7 | 60.5 (1.19x) | **30.4 (0.60x win)** |
| 400, SD4 | 19.0 | 21.6 (1.13x) | **17.8 (0.93x)** |

With the layout flag off (allowGPUOptimizedContents = NO, §20-era lag_repro
root cause), the fair raw regime is: **Metal wins or ties 3 of 4 cells under
the production mv9 march; the single residual loss is 2048/SD4 with jitter=1
(1.31-1.37x)**.

### 21.3 Where the residual 2048/SD4+jitter loss comes from

- Not the layout flag (opt-off throughout): jitter=0 same cells are parity
  (GL 41.7 vs mv0 43.9 / mv9 46.5 — §7-era numbers).
- Jitter cost is asymmetric: GL +27% (41.7 -> 53.1), Metal +57-59% (43.9 ->
  69.8, 46.5 -> 72.8) at this exact cell. Jitter randomizes per-pixel march
  phases, scattering each warp's fetch addresses across the 470 MB volume and
  multiplying the effective working set past the M2 SLC (~32-64 MB); past
  capacity Metal's 3D read path degrades ~9x vs GL's ~7x (volDiv discriminator,
  minimal_gap README). At jitter=0 the fetch set is warp-coherent and
  cache-friendly, so parity returns.
- Confirmed shader-side impossible: unroll, layout (2D-array worse), filter
  (nearest keeps it), compression (incompressible noise), ISA (identical) all
  ruled out. Same conclusion as lag_repro: driver-internal texture-read-path
  behavior.
- Recovery options (both excluded from the fair raw comparison by the user's
  constraint or by design): slabs (restore per-pass cache residency — §20) and
  minmax (Metal-only, skips empty lattice cells — §19).

### 21.4 Harness confirmation of the port

`metal_gap --camera 0 --rt 2048 --sd 4 --composite 1 --data 1`, 3 rounds,
interleaved with GL (composite-path parity cell of §0-era, now with `--noopt`):

| | GL | Metal opt-on | Metal --noopt |
|---|---|---|---|
| 2048/SD4 composite noise | 41.8 / 44.0 / 41.8 (~42.5) | 76.7 / 76.2 / 77.4 (~76.7) | 42.9 / 43.5 / 42.2 (~42.9) |

M/GL 1.80x -> **1.01x**, readback byte-identical (meanB 0.142, sumIter
151697660 in all three Metal runs). The harness loss was the layout flag, and
the port removes it in both the harness and the app.

### 21.5 Low-res recheck (why §19-era 400px numbers showed a win)

The 400x400 raw cell is a **win under mv9** (0.60-0.93x, above) and only a loss
under the temporary mv0 default (1.13-1.24x) — mv0's serial loop exposes the
DRAM-latency floor at small RTs, mv9's 48-wide in-flight batches hide it. The
documented §19.3 grid (400/SD0.5 minmax-on Metal 24.7 vs GL 49.7 = 0.50x) and
the §7.3 az-60 row (Metal 44.7 vs GL 49.7) are consistent with this: Metal is
not slower at 400x400 when the march is mv9.

## 22. Jitter equalization: the harness says parity is achievable, the app does not (2026-08-18)

### 22.1 The question

At 2048/SD4 raw single-pass (jitter=1), GL pays +27% for jitter (41.7 -> 53.1)
while Metal pays +57-60% (mv9 46.5 -> 72.8): the 1.12x j0 residual becomes the
1.31-1.37x j1 gap (§21). Goal: close it without touching quality (per-pixel
jitter required; block-coherent jitter rejected visually, see 22.4).

### 22.2 Harness: equal jitter gives exact parity

Both harnesses implement the SAME IGN ceil-lattice jitter (`--jitter`, the
app-Metal math). 2048 composite noise, noopt, 3 interleaved rounds:

| | GL | Metal | M/GL |
|---|---|---|---|
| SD4 j1 | 73.5 / 71.9 / 70.8 (~72.1) | 68.6 / 68.9 / 68.7 (~68.7) | **0.95x** |
| SD0.5 j1 | 106.5 / 106.9 (~106.7) | 107.4 / 106.2 (~106.8) | **1.00x** |

Both backends pay ~+60-63% over j0 in the harness. So jitter parity IS
achievable when both sides run the same mechanism — the app asymmetry is
app-side, not a fundamental Metal read-path property.

### 22.3 App: equalizing the noise source does nothing

An env-gated IGN jitter for the app GL shader (vtkVolumeShaderComposer.h,
`VTK_METAL_TEST_GL_IGN_JITTER`, since reverted) replaced GL's blue-noise tile
with the same IGN formula Metal uses. Interleaved with Metal at j1:

| cell | GL blue-noise | GL IGN | Metal IGN |
|---|---|---|---|
| 2048/SD4 | 53.1 | 52.9 | 73.7-78.4 |
| 2048/SD0.5 | 160.2 | 158.3 | 87-93 |
| 400/SD4 | 19.0 | 21.1 | 18.2 |
| 400/SD0.5 | 50.7 | 47.3 | 30.0 |

The noise source is irrelevant on GL (blue-noise == IGN in-app, within noise);
Metal is unchanged. The app asymmetry (GL +27%, Metal +57-60%) survives equal
noise.

### 22.4 Block-coherent jitter closes the perf gap but fails quality

`JitterBlockSize` (IGN) + a new block quantization for the blue-noise tile
(sampleJitterNoise, MetalShaders.metal) let pixels march in lockstep. 2048/SD4
j1 mv9, Metal:

| block | Metal | vs GL 54.4 | jitter cost |
|---|---|---|---|
| 1 (per-pixel) | ~73.1 | 1.34x | +57% |
| 2 | ~65.5 | 1.20x | +41% |
| 4 | ~58.1 | **1.07x** | +25% |
| 8 | ~52.0 | **0.96x** | +12% |

SD0.5 is block-neutral (87-90 ms, 0.55x win, per the header's "no change at
the default 0.5 spacing"). **Visually rejected**: block8 shows obvious block
structure, and block4's 4px-grain is clearly worse than GL's per-pixel grain
(user evaluation, 2048/SD4 PNG set in `visual_compare/jitterblocks/`). Quality
is the binding constraint; block coherence is not a solution.

### 22.5 The harness CAN reproduce the app GL's cheap jitter (2026-08-19)

`minimal_gap/gl_app_shader.m` runs the app's exact dumped GL composite shader
(fragment dump from the VolumeRayCast GL scene, `inverseOriginalWindowSize`,
`glReadPixels` and the PPM header fixed to the actual rt). At 2048/SD4
composite-noise data, interleaved rounds:

| shader | j0 | j1 | jitter cost |
|---|---|---|---|
| app GL shader (exact), ramp-128 LINEAR noise | ~40.6 | ~47.0 | +16% |
| app GL shader, **app's real 64x64 blue tile, NEAREST** | ~41.3 | ~44.0 | **+6.5%** |
| app GL shader, same tile LINEAR | ~43.2 | ~44.4 | +3% |
| lean harness GL (gl_gap), same tile NEAREST | ~43.4 | ~69.7 | +61% |
| lean harness GL, IGN hash | ~45.2 | ~69.8 | +54% |

The 2026-08-18 "unreachable from the shader side" conclusion was wrong: the
app GL shader's cheap jitter is a *shader structure* property, reproducible
in the harness (the earlier harness runs used the lean gl_gap shader, which
never carried the app's body).

### 22.6 The mechanism: SIMT divergence from uncorrelated per-pixel jitter

The jitter cost is NOT march length and NOT extra instructions — every
variant marches the same total samples: j0 sumIter 151697660 vs j1
(sharp, smooth, blue, IGN all) 150781xxx (~0.6% less — the ±1 lattice step
variation). The +55-64% appears only when the per-pixel jitter values are
*uncorrelated* between adjacent lanes. Evidence (2048/SD4 composite noise):

| jitter field | harness GL | harness Metal |
|---|---|---|
| IGN hash (sharp, uncorrelated) | +54% | +63% |
| blue tile NEAREST (sharp, uncorrelated) | +61% | +64-66% |
| bilinear blue tile (smooth, correlated) | +61% | **+2-4%** |
| linear-ramp 128 texture (smooth) | +22% | — |

Metal distinguishes the fields: a correlated (smooth) field keeps SIMT lanes
in lockstep and jitter becomes ~free (+2-4%). GL's lean body is saturated —
it pays ~+60% even for the smooth field (the +22% ramp shows GL *can* react
to very low-frequency fields). App-shader bisects (gradient removal, box-exit
removal, alpha-break removal, noise source, origin-shift vs lattice) never
broke the app GL shader's robustness (+6.5-13% throughout): its heavier body
hides the lane divergence — no single feature is the absorber. The app GL's
DICOM cost (+27% vs the noise cell's +6.5%) is the real-scene divergence
(alpha-breaks scatter lane exits more than uniform noise).

### 22.7 GL-parity jitter is implemented, correct, and NOT the DICOM lever (2026-08-19)

The 2026-08-18 harness finding that GL's jitter is *block-coherent* (GL samples
the 64x64 tile at `gl_FragCoord.xy/64` with NEAREST → texel-sized blocks,
32 px @2048, 8 px @512 — not per-pixel) led to a GL-exact parity mode:
`sampleJitterNoise` gains a `blockSize < 0.5` branch (texel = viewportH/64,
index = `floor(st/texel) & 63`), selected by the CPU via
`VTK_METAL_TEST_JITTER_PARITY=1` → `uniforms.JitterBlockSize = 0` (the mapper
env hook also accepts `VTK_METAL_TEST_JITTER_BLOCK_SIZE` for any block size).
The parity field is GL-exact by construction (same tile, same blocks, same
flips).

Harness (gradient data, 2048/SD4): block32 ≈ per-pixel (+51% vs +52% — the
2026-08-18 +4-5% was noise-data only). App (DICOM, 2048/SD4, raw single-pass,
interleaved rounds): j0 45.9 ms, j1 per-pixel 74.1, parity 74.6 — the parity
field costs the same +62% as per-pixel. The jitter gap on real data is NOT
lane divergence from the noise values: block-constant fields keep warps in
lockstep yet still cost +60%. The remaining common factor is the lattice shift
itself (first-sample phase → alpha-gate termination spread → warp-length
divergence); even the predicated fixed-iteration march (`MARCH_STEPS` 32/64)
keeps the +60%. On the cache-hostile noise cell parity stays ~free (+4-5%),
so the lever is data-dependent — a dead end for the DICOM gap (M/GL 1.36-1.42x
at 2048/SD4 j1, vs 1.03-1.08x at j0).

Two real bugs fixed while wiring this (both live in the committed shader):

1. `sampleIGNJitter` divided by the raw blockSize → with nSize 0 the
   fast-math-speculated arm produced NaN and **blacked the whole render**
   (1.2 ms background-only frames, silent). Now clamps `blockSize` to ≥ 1.
2. `sampleJitterNoise`'s block path had the same div-by-zero shape; both now
   use `max(blockSize, 1.0f)`.

`TestMetalScenes.h` `BuildVolumeScene` now honors `TempJitter()` like the DICOM
builder (jitter on by default for the noise scenes; was always off — the
in-app A/B used DICOM only). `metal_gap` gained `METAL_GAP_BLUE_PARITY` (the
exact app parity index, for harness A/B).

### 22.8 Status

Raw single-pass 2048/SD4 j1 stays 1.36-1.42x (j0 1.03-1.08x). Mechanism
refined: the DICOM jitter cost is the lattice-shift termination spread (data-
dependent; the noise-cell cost is pure lane divergence and is fully cured by
the GL-parity field). Per the user's direction (2026-08-19) the hunt
continues; the parity lever, the predicated march (`MARCH_STEPS`), and the
smooth-field lever all fail on DICOM. The only verified way to soften the
DICOM jitter cost remains the slab path (+34% on slabs8 vs +62% single) and
the minmax path. Metal jitter defaults are unaffected: blue-noise tile
per-pixel (default), GL-parity only via `VTK_METAL_TEST_JITTER_PARITY=1`.

### 22.9 The GL-vs-Metal gap is now reproducible in the minimal harness (2026-08-19)

The user's question: how to reproduce the app's GL-vs-Metal jitter gap in the
minimal repro. Two findings:

1. **The gradient harness cell reproduces the *absolute* jitter cost but not
   the backend gap** — with the sharp per-pixel hash, GL jitter costs
   +40-55% (65-67 vs 42-44), nearly as much as Metal (+58-66%): the
   hand-rolled harness GL shader was *not* app-like.
2. **The app GL's jitter is cheap because it uses the bilinear-filtered noise
   texture (smooth correlated field)** — the same reason Metal's
   "correlated smooth" mode is ~free. With `GL_GAP_TEXNOISE=1` (now the
   default; `GL_GAP_SHARPHASH=1` restores the sharp hash), the harness GL
   jitter drops to +9-22% and the pair reproduces the app gap:

   | round | GL j0 | GL j1 (smooth) | Metal j0 | Metal j1 | M/GL j1 |
   |-------|-------|----------------|----------|----------|---------|
   | 1     | 40.7  | 48.7  (+19%)   | 41.3     | 65.1 (+58%) | 1.34x |
   | 2     | 40.1  | 47.4  (+18%)   | 41.6     | 66.8 (+61%) | 1.41x |
   | 3     | 45.8  | 50.0  (+9%)    | 41.5     | 68.9 (+66%) | 1.38x |

   App reference: GL +28%, Metal +62%, M/GL 1.36-1.42x. Recipe:
   `./gl_gap --camera 0 --rt 2048 --sd 4 --composite 1 --jitter 1` vs
   `./metal_gap` same flags (interleave; battery drift).

   Reproduces the DICOM finding: the Metal *lattice-shift* cost (gradient
   data) is backend-specific; GL's filtered noise field doesn't pay it.

While repairing `gl_gap.m` for the default flip, the committed source was
found broken (the 07:23 binary predated the corruption; the committed
`uUnbounded` experiment replaced the bounded march with a gated `while(true)`
that (a) never runs when `uUnbounded=0`, (b) is unbalanced in two variants,
(c) uses a GLSL-invalid `?: (void)0` ternary; plus a duplicated
`uniform sampler2D uNoise;` in `kFragSrc`). All three march variants restored
to the canonical `for (i < min(uMaxIter, maxSteps))` loop; `GL_GAP_UNBOUNDED`
is now a no-op.
