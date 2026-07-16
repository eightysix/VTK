# Metal Volume Renderer Performance Analysis

I went through both files carefully. For a 1 GB thin-slice CT, the axial-view drop (10–15 fps vs 20–30 fps coronal/sagittal) is the classic "long ray through the long axis" problem. Below are concrete, prioritized changes — grouped by ROI — that don't touch visual-quality knobs.

---

## 1. Critical correctness + perf bug — `MAX_RAY_STEPS = 2000` ✅ DONE

Raised `MAX_RAY_STEPS` from 2000 to 8192 in `MetalShaders.metal:1306`. The existing adaptive computation `int(ceil(totalDist/stepSize))` already governs for normal cases; the constant only acts as a safety cap. For thin-slice CT (512×512×2000+), the old cap silently truncated the ray, cutting off the back of the volume. The new cap of 8192 is high enough that only pathological cases hit it.

---

## 2. Biggest GPU win — make min-max sampling cell-boundary–triggered ✅ DONE

Right now every loop iteration issues a `minMaxTexture.sample(...)` regardless of whether we've moved to a new macrocell. In dense tissue (the slow axial case), that's a wasted texture fetch + filter on every step. Track the current cell and only re-sample when the ray crosses into a new one:

```cpp
// Before the loop:
int3  curCell     = int3(-1);
bool  curCellEmpty = false;
float3 mmDimF     = float3(volumeUniforms.minMaxDimX,
                            volumeUniforms.minMaxDimY,
                            volumeUniforms.minMaxDimZ);

// Inside the loop, replacing the existing useMinMaxAccel block:
if (volumeUniforms.useMinMaxAccel > 0.5) {
  float3 mmPos = clamp(currentPoint, float3(0.0), float3(1.0));
  int3  newCell = int3(mmPos * mmDimF);
  if (any(newCell != curCell)) {
    curCell      = newCell;
    curCellEmpty = minMaxTexture.sample(minMaxSampler, mmPos, level(0)).r > 0.5;
  }
  if (curCellEmpty) {
    // … existing DDA skip logic …
    // After skip, invalidate the cached cell so we re-sample next iter:
    curCell = int3(-1);
    continue;
  }
}
```

In the dense patient interior this typically cuts the min-max fetch count by ~4× (DS=4 means each cell covers 4 samples). Combined with the texture cache staying warm for `volumeTexture`, this alone often buys back the axial-vs-coronal gap.

---

## 3. Auto-partition large volumes along the long axis ✅ DONE

`Partitions` defaults to `{1,1,1}`, so the user is hitting the single-texture path for a 1 GB volume. The block pipeline already exists and already does:
- Empty-block skipping via `IsBlockEmpty`
- Back-to-front sorting
- Per-block ERT

Add an auto-partition heuristic in `UpdateVolumeTexture` (or in `SetInputData`):

```cpp
// Heuristic: if any dimension > 384 voxels, partition that axis so each
// block is <= ~384 voxels on its longest side.
if (!usePartitions) {
  int dims[3]; input->GetDimensions(dims);
  for (int axis = 0; axis < 3; ++axis) {
    this->Partitions[axis] = std::max(1, (dims[axis] + 383) / 384);
  }
  // Re-enter the partitioned branch by recomputing usePartitions
  usePartitions = (this->Partitions[0] > 1 || this->Partitions[1] > 1 || this->Partitions[2] > 1);
}
```

For your 1 GB CT this typically yields `1×1×4` or `1×1×8`, which:
- Lets empty slabs (above the head / below the feet) be skipped at the draw-call level — zero fragment work for those z-ranges.
- Shrinks each 3D texture so it fits better in the GPU texture cache (Apple GPUs have ~96–384 KB per-texture L1).
- Lets ERT fire per-block instead of needing to walk 2000+ slices.

---

## 4. Per-block min-max texture

When partitioning is on, `UpdateMinMaxTexture` still builds *one* global occupancy volume at DS=4. For a 1 GB CT partitioned 1×1×8, each block's slice range is ~250 slices; a per-block min-max at DS=2 would have the same memory footprint as the current global DS=4 but ~4× finer granularity. Inside `UpdateBlockTextures`, after computing each block's scalar range, also build a small `R8Unorm` 3D texture for that block and store it on `VolumeBlock`. Bind it at fragment texture index 8 (you'll need to add it to the shader signature and to the fallback bind in `BindEncoderResources`).

This is the single biggest "architectural" win for the axial case after auto-partitioning.

---

## 5. CPU-side: parallelize the dilation pass in `UpdateMinMaxTexture`

The dilation is currently a serial triple-nested loop over every macrocell — for a 512×512×2000 CT at DS=4 that's 128×128×500 ≈ 8 M cells. On one thread this is ~50–150 ms. Replace it with a gather-style stencil (read-only on `rawMinMax`, write-only on `minMaxData`), which is embarrassingly parallel:

```cpp
std::vector<uint8_t> minMaxData(numCells, 255);
vtkSMPTools::For(0, numCells, [&](vtkIdType begin, vtkIdType end) {
  for (vtkIdType cellIdx = begin; cellIdx < end; ++cellIdx) {
    const int gx = static_cast<int>(cellIdx % mmDims0);
    const int gy = static_cast<int>((cellIdx / mmDims0) % mmDims1);
    const int gz = static_cast<int>(cellIdx / (mmDims0 * mmDims1));

    const int x0 = std::max(0, gx - 1), x1 = std::min(mmDims0 - 1, gx + 1);
    const int y0 = std::max(0, gy - 1), y1 = std::min(mmDims1 - 1, gy + 1);
    const int z0 = std::max(0, gz - 1), z1 = std::min(mmDims2 - 1, gz + 1);

    bool solid = false;
    for (int nz = z0; nz <= z1 && !solid; ++nz)
      for (int ny = y0; ny <= y1 && !solid; ++ny)
        for (int nx = x0; nx <= x1 && !solid; ++nx)
          if (rawMinMax[(nz * mmDims1 + ny) * mmDims0 + nx] == 0) solid = true;

    minMaxData[cellIdx] = solid ? 0 : 255;
  }
});
```

Same memory access pattern, but now uses all cores. This typically drops the dilation step from ~80 ms to ~5 ms on an M-series chip.

---

## 6. CPU-side: stop iterating voxels twice

`UpdateMinMaxTexture` walks every voxel to compute per-cell min/max. Then `UpdateBlockTextures` walks every voxel *again* per block to compute `BlockScalarRanges`. For a 1 GB volume, that's ~500 M voxels walked twice on the CPU.

Hoist the block-range computation into the min-max pass: while scanning macrocells, also accumulate block-level min/max (you know each block's extent). Or have `UpdateBlockTextures` consume the already-built macrocell mins/maxes (reduction over `(blockExtents/DS)³` cells per block — typically ~16³ = 4096 reductions vs. 250³ voxels).

---

## 7. Shader: stop unconditionally prefetching `volumeTexture` in skip iterations — ✅ DONE

In the skip branch:
```cpp
if (i + 1 < maxSteps) {
  prefetchScalar = volumeTexture.sample(volumeSampler, currentPoint, level(0)).r;
  if (doMask) prefetchMask = maskTexture.sample(maskSampler, currentPoint, level(0)).r;
}
continue;
```

When the ray is skipping through many empty macrocells in a row (typical for the air around the patient in axial view), every skip iteration re-issues a `volumeTexture` fetch that is immediately thrown away on the next iteration when the cell is still empty. Defer the prefetch until the cell is confirmed non-empty:

```cpp
// In the skip branch: do NOT prefetch. Just invalidate:
prefetchScalar = as_type<float>(0x7fc00000u); // NaN sentinel
continue;

// In the regular branch, before using prefetchScalar:
bool needsFetch = (as_type<uint>(prefetchScalar) == 0x7fc00000u);
float rawScalar = needsFetch
  ? volumeTexture.sample(volumeSampler, evalPoint, level(0)).r
  : prefetchScalar;
float rawMask = (doMask && needsFetch)
  ? maskTexture.sample(maskSampler, evalPoint, level(0)).r
  : prefetchMask;
```

Alternatively, do a single min-max fetch in the skip branch for the next cell, and only prefetch volume if that cell is non-empty. Either way, this is a meaningful bandwidth saving for the axial case.

---

## 8. Shader: pack `minMaxDim` and cache other uniforms in scalars

```cpp
float3 mmDim = float3(volumeUniforms.minMaxDimX, volumeUniforms.minMaxDimY, volumeUniforms.minMaxDimZ);
```
is fetched three times per iteration from the constant buffer. Hoist it above the loop (you already do this for many things — just add `mmDim`). Also consider packing the three dims into the `.rgb` of an existing float4 in the uniforms struct (e.g., reuse a padding slot) — minor, but every constant-buffer fetch in the hot loop adds up.

---

## 9. Per-block uniform upload overhead

`DrawBlocks` calls `setVertexBytes` + `setFragmentBytes` with the full 976-byte `VolumeMapperUniforms` struct per block. For an 8-block partition that's 8 × 976 = ~7.8 KB of memcpy + 16 API calls per frame, plus the cost of re-binding the block texture. Two options:

- **Quick win**: use `setVertexBuffer:offset:atIndex:` with a single shared `MTLBuffer` containing an array of `VolumeBlockUniforms` (just the per-block bits: `volumeBoundsMin/Max`, `cameraVolumePos`, `gradientStep`, block texture index). The fragment shader indexes by `gl_InstanceID`/`[[instance_id]]`. Drops 976 → ~64 bytes per block.
- **Better win**: render all blocks in one `drawIndexedPrimitives:instanceCount:` call with instancing. The vertex shader picks the block by instance ID. This collapses 8 draw calls into 1.

---

## 10. Fix the `currentT >= t.y` termination check

```cpp
if (any(currentPoint < float3(0.0)) || any(currentPoint > float3(1.0)) || currentT >= t.y) {
  break;
}
```

`currentT` is measured from `entryPoint`, but `t.y` is the ray parameter from `cameraPos`. When the camera is outside the volume (`tStart > 0`), this check never triggers (so it's just dead code), but when the camera is inside, `t.y` is the correct exit distance *only because* `tStart = 0` was clamped. Make it explicit and correct:

```cpp
float exitDist = t.y - tStart;   // distance from entryPoint to exitPoint
...
if (... || currentT >= exitDist) break;
```

Not a perf win directly, but it makes the skip loop's correctness obvious and lets you trust larger skips.

---

## 11. Inter-block opacity propagation (medium-effort, big axial-view win)

Currently, even with back-to-front block sorting, each block's fragment shader starts `accumulatedOpacity = 0`. So if block N (farthest) deposits 0.7 opacity, block N-1 still ray-marches all ~250 of its slices even though only 0.3 weight remains. Solutions:

- **Front-to-back with screen-space opacity buffer**: maintain an R8/R16 opacity texture. Before block K, bind it as a readable texture; the fragment shader reads `prevOpacity` and uses `1 - prevOpacity` as the initial `weight`. After block K, blend-write the new opacity. This is the standard "multi-pass volume" trick.
- **Or**: render all blocks in a single fragment-shader pass by using a texture array (`MTLTextureType3DArray`) for the block volumes and a per-block bounds buffer; the shader marches each ray through all blocks in order. This is the cleanest solution and gives you global ERT for free. Requires shader rewrite but is conceptually simple.

Either of these typically gives 1.5–2× on the axial view alone, because most of the patient's interior is dense enough to ERT well before the ray exits the last block.

---

## 12. Optional: precomputed gradient volume (only if shading is the bottleneck)

If profiling shows `computeGradientFast` (6 samples per shaded voxel) is dominant, precompute a `MTLPixelFormatRGBA8Unorm` or `RGBA16Float` 3D texture holding `(normal.xyz, gradientMagnitude)` once on CPU/GPU upload. The shading path becomes 1 fetch instead of 6. Memory cost is 4–8× the scalar volume, so for a 1 GB CT that's 4–8 GB — only viable on higher-end Macs. Worth gating behind a property flag.

A cheaper variant: store gradients at half resolution (DS=2) and bilinearly upsample in the shader. 1 GB → 0.5 GB extra and still removes 5 of the 6 fetches per shaded voxel.

---

## 13. Small things worth doing

- **`fragment_volume_main`**: the `intersectBox` uses `1.0 / (dir + float3(1e-8))`. Adding `1e-8` to all three components biases the inverse along axes that may not need it. Use `select(1.0/huge, 1.0/dir, abs(dir) > 1e-8)` instead — clearer and a hair faster.
- **`computeGradientFast`**: the `level(0)` argument on `volTex.sample` is a no-op for non-mipmapped textures; remove it to shave an instruction.
- **`volume_random`**: `fract(sin(dot(...)) * 43758.5453)` is fine, but `hash(…)` via `as_type`/interleave is faster and statistically better. Minor.
- **`BindEncoderResources` fallbacks**: when min-max is disabled, you bind `volTex` at index 7. The shader still samples it (then ignores the result). Add `if (useMinMaxAccel > 0.5)` around the min-max block in the shader (already there — good) but also short-circuit the volume fetch path when possible.
- **`SetPartitions`**: when partitions change, you currently don't release the single-texture `VolumeTexture` — make sure both paths can't be active simultaneously.

---

## Suggested implementation order

1. ~~Item **#1** (MAX_RAY_STEPS)~~ ✅ — done. ~~Item **#7** (skip-branch prefetch)~~ ✅ — done. ~~Item **#2** (cell-boundary min-max)~~ ✅ — done. ~~Item **#3** (auto-partition)~~ ✅ — done.
2. Item **#5** (parallel dilation) and **#6** (dedup voxel scans) — 1 hour, cuts first-frame and re-upload times.
3. Item **#11** (inter-block opacity) — half day, the remaining axial-view gap.
4. Item **#4** (per-block min-max) — half day, on top of #3.
5. Items **#9, #10, #12** — polish / optional.

Expect #1–#3 alone to push axial view from 10–15 fps into the 20–25 fps range; #5, #6, #11 should get you to parity with the coronal/sagittal numbers.
