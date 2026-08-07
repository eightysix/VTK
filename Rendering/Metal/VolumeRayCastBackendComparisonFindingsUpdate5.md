# Metal vs OpenGL volume ray cast: sample-lattice re-verification — update 4's drift hypothesis refuted by direct step measurement (update 5)

Follow-up to all prior documents, read as an addendum to:

- [VolumeRayCastBackendComparisonProcedures.md](VolumeRayCastBackendComparisonProcedures.md) —
  environment, capture/analyze tooling.
- [VolumeRayCastBackendComparisonFindings.md](VolumeRayCastBackendComparisonFindings.md) —
  sections 1–7, still valid.
- [VolumeRayCastBackendComparisonFindingsUpdate.md](VolumeRayCastBackendComparisonFindingsUpdate.md) —
  composite-gate and termination-threshold fixes.
- [VolumeRayCastBackendComparisonFindingsUpdate2.md](VolumeRayCastBackendComparisonFindingsUpdate2.md) —
  fp16-accumulation and MAX_RAY_STEPS fixes; §3.1 characterized the residual
  "grazing-ring-edge / sample phase" pixels at coarse steps.
- [VolumeRayCastBackendComparisonFindingsUpdate3.md](VolumeRayCastBackendComparisonFindingsUpdate3.md) —
  gradient-opacity LUT fix (1024×float32); collapsed the gf divergence family,
  leaving the ring-edge / sample-phase pixels as the common floor.
- [VolumeRayCastBackendComparisonFindingsUpdate4.md](VolumeRayCastBackendComparisonFindingsUpdate4.md) —
  hypothesized that Metal's three chained accumulators and direction-normalized
  step make the sample lattice coherently drift from OpenGL's `g_dataPos`, and
  that this drift is the residual floor.

Update 4 was a code-trace root-cause candidate, not a measurement. The
single-accumulator / `g_dirStep`-arithmetic fix it proposed in §3 has since been
implemented (commit `1732d6bf25`), and this document records what direct
measurement of the post-fix sample lattice shows: the lattice does **not**
coherently drift from OpenGL's, Update 4's "different step-length scaling for
off-axis rays" is algebraically cancelled by the ray-direction normalization
spaces, and the measurable residual is two orders of magnitude too small to be
the observed delta floor. The floor is re-attributed to boundary-crossing phase
behaviour.

## 1. What was measured

The Metal shader was instrumented (`VTK_METAL_ENABLE_LOGGING`) to dump, per
gated fragment, the MARCH parameters (camera, `rayDir`, `tStart`, `tEnd`,
`stepSize`, `firstT`, `jitter`, entry point, texel counts) and the per-sample
cell-to-point-adjusted position `eval=(x,y,z)` for every march iteration
(`MetalShaders.metal:3836-3853, 4163-4174`). A capture of the fixed-step
camera-outside variant (512³ head volume, non-uniform spacing
`boundsSize = (201.6, 201.6, 138)`, `sampleDistance = 0.5`, axis-aligned) was
parsed, and the per-step advance was recovered as the median of successive
`eval` diffs along each ray.

The comparison target is OpenGL's per-step advance, computed from the composer
formula that drives the reference renderer
(`Rendering/VolumeOpenGL2/vtkVolumeShaderComposer.h:437-439`):

```
g_dirStep = (ip_inverseTextureDataAdjusted * vec4(rayDir, 0.0)).xyz * in_sampleDistance
```

with `ip_inverseTextureDataAdjusted = in_cellToPoint * in_inverseTextureDatasetMatrix`
and `in_sampleDistance` a float32 uniform equal to the actual sample distance
(0.5). For the axis-aligned test volume `inverseTextureDataset = diag(1/201.6, 1/201.6, 1/138)`
and `in_cellToPoint` is the diagonal `ctpScale` computed with GL's two-stage
float32 expression `(d-0.5)/d - 0.5/d` (= 0.998046875 for d = 512).

## 2. The key correction: ray directions live in different normalization spaces

Update 4 §2 item 1 argued that Metal's per-ray
`length(rayDir·boundsSize)` normalization changes the step-length scaling for
off-axis rays in a non-cubic volume. That argument overlooks that the two
backends normalize the *same physical ray line* in different spaces:

- OpenGL's `rayDir` is the unit **physical/object-space** direction (the proxy
  box is positioned at the volume's physical bounds), i.e. the normalized
  physical-space vector `rd_gl = (rd_box · boundsSize) / |rd_box · boundsSize|`.
- Metal's `rayDir` (`MarchParams.rayDir`) is the unit **normalized-volume
  (proxy-box) space** direction `rd_box`.

The physical advance per step is 0.5 world units along the line in both
backends, but written in each backend's own frame:

- OpenGL texture-space step:
  `inverseTextureDataset · rd_gl · sampleDistance`
- Metal texture-space step (`MetalShaders.metal:3691-3717`):
  `volumeToTexture · (rd_box · boundsSize) · stepSize`
  with `stepSize = sampleDistance · maxBound / |rd_box · boundsSize|`.

Because `(rd_box · boundsSize) / |rd_box · boundsSize| = rd_gl` and
`volumeToTexture = inverseTextureDataset`, the two step vectors are
**algebraically identical**:

```
volumeToTexture · (rd_box · boundsSize) · (sampleDistance / |rd_box · boundsSize|)
  = inverseTextureDataset · rd_gl · sampleDistance
```

The `length` normalization and the `(rd_box · boundsSize)` pre-multiply exactly
cancel the normalization-space difference. There is no per-pixel step-length
scaling discrepancy for off-axis rays. (An earlier comparison that *appeared* to
show a ~1.4× off-axis step divergence was an analysis bug: it fed the
box-space-normalized `rd_box` into the physical-space GL formula. Corrected with
`rd_gl`, the steps match.)

## 3. Measured result: Metal's evalStep equals GL's g_dirStep to print precision

For every sampled pixel the recovered per-step advance reproduces GL's
`g_dirStep` to the six-decimal print precision of the log (residual ≤
6e-4 voxels/step, i.e. the log's own truncation floor):

```
px      GL g_dirStep                      Metal evalStep (log AP)      drift (voxels/step)
(480,508) [ 0.00053709 -0.00062803 -0.00340865] [ 0.000537 -0.000628 -0.003408]  ~5e-4
(201,13)  [-0.00014824  0.00059646 -0.00350287] [-0.000148  0.000596 -0.003502]  ~4e-4
(45,113)  [-0.00053860  0.00034522 -0.00349325] [-0.000538  0.000345 -0.003492]  ~6e-4
(372,131) [ 0.00028535  0.00030576 -0.00356412] [ 0.000285  0.000306 -0.003563]  ~6e-4
(256,256) [-0.00001067 -0.00001326 -0.00361603] [-0.000011 -0.000013 -0.003615]  ~5e-4
```

(256,256) is the near-axis ray, the rest are strongly off-axis diagonal rays —
exactly the case Update 4 §2 item 1 predicted should show a step-length scaling
mismatch. It does not.

## 4. The true ulp-level residual: the half-precision SampleDistanceHalf uniform

An exact float32 simulation of the two shaders' arithmetic order pins the
residual that the log's print precision hides. For `(480,508)`:

```
GL   g_dirStep = [ 0.00053709 -0.00062803 -0.00340865]
Metal evalStep = [ 0.00053696 -0.00062788 -0.00340781]
per-step diff   = ~1.3e-7 .. 8.3e-7 texels   (~6.7e-5 .. 4.3e-4 voxels/step)
relative        = ~2.44e-4
```

The Metal shader arithmetic (including `physicalSampleStep`,
`MetalShaders.metal:3007-3014`) reproduces the logged values exactly, so the
model is the shader. The entire ~2.44e-4 relative residual comes from one
source: the march step uniform is stored in **half precision**
(`SampleDistanceHalf`, `vtkMetalGPUVolumeRayCastMapper.mm:97`, written at
`:6649-6650` as `FloatToHalf(actualSampleDistance / maxBoundsSize)`), which
quantizes `0.5/201.6` to half ulp ~2.4e-4 relative. OpenGL passes
`in_sampleDistance` as full float32.

This is the residual class Update 4 described as "~1 ulp per axis", but its
magnitude matters: a fully coherent accumulation of the step error over a
440-sample ray is ≤ 0.03–0.19 voxels of lattice offset depending on axis — one
to two orders of magnitude below the observed delta floor (`max|Δ|` up to 22 at
the default sample distance).

## 5. Reconciliation with Update 4 §2

- **§2 item 1 (step differs in last ulp / off-axis step-length scaling):**
  *Refuted as a divergence source.* The normalization is algebraically cancelled
  by the box-vs-physical ray normalization (this document §2). The only real
  step difference is the half-vs-float sample-distance uniform (§4), worth
  ~2.4e-4 relative.
- **§2 item 2 (three chained float accumulators):** *Moot.* Both backends
  advance the *sample* position with a single float32 add per step
  (`g_dataPos += g_dirStep`; `evalPoint += evalStep`). Metal's `currentPoint`
  and `texLocalPos` are side accumulators for t bookkeeping / boundary checks,
  not the sample lattice. The committed fix (single accumulator, integer
  `currentT` counter, `MetalShaders.metal:4462-4465`) already matches GL's
  architecture.
- **§2 item 3 (cell-to-point re-applied per sample):** *Resolved by the fix.*
  `evalStep` is now one mat-vec (`adjustedLin`, the volumeToTexture rows folded
  with `ctpScale`) times one scalar, structurally identical to GL's
  `ip_inverseTextureDataAdjusted · rayDir · sampleDistance`
  (`MetalShaders.metal:3713-3717`).
- **§2 headline (lattice drift is the residual floor):** *Not supported by
  measurement.* The post-fix lattice matches GL's to ≤ ~2.4e-4 relative
  (half-uniform quantization), accumulating to ≤ 0.2 voxels over a full ray.

## 6. The residual floor is boundary-crossing phase, not lattice

The variants/sample-distance sweeps are consistent with a boundary-sampling
mechanism rather than a lattice mechanism:

- `mean|Δ|` is flat (~0.3–0.35) while `max|Δ|` and the masked-pixel count grow
  strongly with step size (`sd 0.0675→4.0`: `max|Δ|` 6→124, masked px 2→3378,
  `/tmp/bc/recap/sweep_table.txt`). A lattice-offset mechanism would grow with
  the number of samples (smaller steps → more samples → more drift), the
  opposite of what is observed; a boundary-crossing mechanism grows with step
  size.
- Deltas concentrate in a small mask at high-contrast surfaces (skin/bone
  shells) while centers and flat regions match to <1
  (`/tmp/bc/recap/variants_table.txt`): the signature of two sample sequences
  that coincide almost everywhere and differ only where the accumulated opacity
  is steep within a step's window.
- With the lattice verified identical, the only remaining mechanisms that can
  shift a surface crossing by O(step size) are the *phase* decisions around
  boundaries: minmax-skip stepping vs GL's `g_skip` (which cell of an empty
  minmax block the ray exits from, and how the skip distance snaps to the step
  lattice), the opacity-threshold termination
  (`accumulatedOpacity >= 1 - 1/255`), and the entry/first-sample offset
  (`firstT`, jitter).

## 7. Current state

- Commit `1732d6bf25` ("fix(Metal): march with integer step counter and GL
  g_dirStep arithmetic") implements Update 4 §3's single-accumulator fix.
- Working tree (uncommitted) carries only capture instrumentation: the
  `debugMarchGate` extra gated pixels (`MetalShaders.metal:3645`) and a TEMP
  DEBUG GL-uniform dump in
  `Rendering/VolumeOpenGL2/vtkOpenGLGPUVolumeRayCastMapper.cxx:3600-3616`.
- Line references above are to the current working tree.

## 8. Remaining work

1. **Promote `SampleDistanceHalf` to float32** (and audit the other
   precision-sensitive uniforms that feed per-step arithmetic) so Metal's step
   matches GL's `g_dirStep` to ~1 ulp instead of ~2.4e-4 relative. This is the
   correct "last ulp" item — Update 4's suggested de-normalization is *not*
   needed, since the normalization is already exact.
2. **Test the boundary-phase candidates** at a known high-delta pixel: dump the
   minmax-skip stepping (Metal `curCell` / `exactSkip` / `skipSteps`,
   `MetalShaders.metal:3879-3924`) against GL's `g_skip` on the same ray, then
   the opacity-threshold termination and the `firstT`/jitter entry offset.
   Expect the common floor (0.3–0.6 `mean|Δ|`, ≤ ~360 masked px at default
   step, Update 3 §2) to drop when the skip stepping and entry phase match.
3. Re-capture the variant table (procedures doc *Image tests*) plus the GL
   deterministic re-capture check after each of the above.

## 9. Decision: eliminate the gap without touching OpenGL (options, by effort)

The single-source shared-core approach (one GLSL raycast core compiled to MSL via
SPIR-V so both backends execute the same program) would eliminate the gap by
construction, but it is **deferred**: the requirement is to reach bit-parity
without modifying the OpenGL backend.

### The enabling realization

OpenGL's lattice is pure IEEE-754 float32 accumulation of a constant step from a
constant seed (`g_dataPos += g_dirStep`, round-to-nearest-even, one add per
step). IEEE adds are deterministic and identical given identical operands, so
Metal does **not** need a shared core or a canonical
`pos(k) = fma(float(k), step, origin)` recompute to be bit-identical — that
recompute would in fact *break* parity with GL's accumulating `g_dataPos`
(different rounding sequence). Metal only needs:

1. a **bit-identical step** (same fp32 vector as `g_dirStep`), and
2. a **bit-identical seed** (same cell-to-point-shifted entry + jitter),

after which Metal's own `evalPoint += evalStep` accumulation matches GL's
`g_dataPos` bit-for-bit at every sample index. Both goals are achievable with
Metal-side-only changes.

### Options, ordered from easiest to hardest

- **A. Float32-only decision path.** Promote `SampleDistanceHalf` to float32 and
  audit every other half uniform that feeds step/seed/skip arithmetic. Removes
  the entire §4 residual (~2.4e-4 relative from half quantization of the step).
  After A, the only step-level difference left is the mat-vec folding order
  (Metal `adjustedLin·(rayDir·boundsSize)·stepSize` vs GL
  `ip_inverseTextureDataAdjusted·rayDir·sampleDistance`), worth ~1 ulp,
  accumulating to ~5e-5 voxels over a full ray — below any decision threshold.
  This is the recommended first step.
- **B. Bit-identical seed.** Ensure Metal's entry texture coords, `tStart`/`tEnd`,
  jitter and `firstT` derive from the same CPU-side math as GL's, so the
  accumulation starting value matches bit-for-bit. Medium effort; without it the
  whole lattice carries a constant phase offset.
- **C. Bit-identical minmax-skip arithmetic.** Metal's `exactSkip`/`skipSteps`
  (`MetalShaders.metal:3879-3924`) must produce the same integer step count as
  GL's `g_skip`, otherwise the lattice desyncs after every traversed empty
  block. This is the boundary-phase class from §6 and the likeliest remaining
  source of the `max|Δ|` outliers. Medium-hard.
- **D. Decision-dump harness.** Log the per-sample `(k, position, cell, skip)`
  sequence from both backends for the same pixel and diff them; gate renders on
  `max|Δ| == 0` over float channels before 8-bit quantization. OpenGL is touched
  only by a TEMP DEBUG position dump (same spirit as the existing GL_UNIFORMS
  dump in the tree); the algorithm is untouched. Cheap and high-leverage for
  locating whatever A–C miss.
- **E. Fixed-point integer lattice** (radical end). Integer cell-index / skip
  math is identical across compilers and drivers by construction. Only needed if
  A–D leave driver-level float behavior (non-IEEE contraction, flush-to-zero)
  that flips discrete decisions; a robustness fallback, not the primary plan.

### Recommended order

`A` (foundation, smallest) → `D`-light (measure where the residual actually is)
→ `B` (seed) → `C` (skip) → `E` only if a measured residual remains. Each step is
verified by re-running the lattice check (§3) and the variant/sample-distance
captures before moving on.

## 10. Status: option A implemented

`A` is done. `SampleDistance` was promoted from `half` to full `float32`
(`MetalShaders.metal` struct field and `vtkMetalGPUVolumeRayCastMapper.mm`
mirror, offset 240; the unused `opacityPreIntegrationFactor` half that shared
the 4-byte slot was dropped, so **all struct offsets are unchanged** —
`static_assert(sizeof(VolumeMapperUniforms) == 1712)` still holds). The write
site stores `static_cast<float>(actualSampleDistance / maxBoundsSize)` and
`physicalSampleStep` reads it directly.

Re-measured on the fixed-step camera-outside capture (sd = 0.5, same pixels as
§3): the logged `stepSize` moved from the half-quantized `0.003513` to
`0.003514`, and the per-step lattice drift against GL's `g_dirStep` dropped to
the log's print-precision floor (≤ 6e-4 voxels/step, i.e. the `%f` truncation).
An exact float32 simulation now gives:

```
GL   g_dirStep = [ 0.00053709 -0.00062803 -0.00340865]
Metal evalStep = [ 0.00053709 -0.00062803 -0.00340865]
per-step diff   = ~5e-11 texels   (relative ~1.1e-7)
coherent accumulation over a 440-sample ray: <= 1e-4 voxels
```

The only step-level residual left is the mat-vec folding order (Metal
`adjustedLin·(rayDir·boundsSize)·stepSize` vs GL
`ip_inverseTextureDataAdjusted·rayDir·sampleDistance`), worth one ulp. Option A
therefore removes the entire half-quantization residual; the next candidate for
measurement is the seed phase (B) and the minmax-skip arithmetic (C).
