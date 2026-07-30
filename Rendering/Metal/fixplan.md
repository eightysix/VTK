# Stage 1 Implementation Guide: Critical Correctness Fixes

## Scope

This guide covers all critical and high-priority correctness fixes for the VTK Metal polydata mapper pipeline. Each fix is self-contained with exact code changes, ordered by dependency.

**Files modified:**

| File | Fixes |
|---|---|
| `vtkMetalDepthPeeler.h/.mm` | #1, #12, #13, #14 |
| `vtkMetalPolyDataMapper.mm` | #3, #4, #5, #6, #7, #8, #9, #10, #11 |
| `vtkMetalRenderer.mm` | #5, #6 |
| `MetalShaders.metal` (shader string) | #3, #4 |
| `vtkMetalRenderWindow.h` | #2 (accessor) |

**Estimated effort:** 3–5 days for one engineer familiar with Metal and VTK.

---

## Fix #1: Depth Peeler `InitDepthPipeline` Shader Mismatch

### Problem

`vtkMetalDepthPeeler::CreatePipelines()` creates `InitDepthPipeline` using `vertex_fullscreen_main` + `fragment_peel_init`. But `fragment_peel_init` expects `VertexOut` (from `vertex_main`), not `FullscreenVertexOut`. Metal validates vertex-output/fragment-input compatibility at pipeline creation, so this pipeline returns nil.

Since `PipelinesCreated` requires all three pipelines to be non-nil:

```cpp
this->PipelinesCreated = (this->InitDepthPipeline && this->CompositePipeline &&
                          this->BackBlendPipeline);
```

`PipelinesCreated` is always false, and `RenderTranslucentGeometry()` returns 0 immediately. **Depth peeling is entirely non-functional.**

The `InitDepthPipeline` is never actually used in the render loop — the mapper's `TriangleInitPeelPipeline` handles geometry init via `DepthPeelingMode=1`. The depth peeler's own init pipeline is dead code.

### File: `vtkMetalDepthPeeler.mm`

### Change

Remove `InitDepthPipeline` from the pipeline creation and the `PipelinesCreated` check.

**In `CreatePipelines()`, delete the entire init depth pipeline block:**

```cpp
// DELETE THIS ENTIRE BLOCK:
// --- Init depth pipeline ---
// Fullscreen pass: reads nothing, outputs RG32Float with MAX blend
{
    id<MTLFunction> vFunc = [library newFunctionWithName:@"vertex_fullscreen_main"];
    id<MTLFunction> fFunc = [library newFunctionWithName:@"fragment_peel_init"];
    if (vFunc && fFunc)
    {
        MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
        // ... entire block ...
        [desc release];
    }
    [vFunc release];
    [fFunc release];
}
```

**Change the `PipelinesCreated` check:**

```cpp
// BEFORE:
this->PipelinesCreated = (this->InitDepthPipeline && this->CompositePipeline &&
                          this->BackBlendPipeline);

// AFTER:
this->PipelinesCreated = (this->CompositePipeline && this->BackBlendPipeline);
```

### File: `vtkMetalDepthPeeler.h`

**Remove the dead member:**

```cpp
// DELETE:
id<MTLRenderPipelineState> InitDepthPipeline = nil;
```

**In `Release()`, delete:**

```cpp
// DELETE:
[this->InitDepthPipeline release];
this->InitDepthPipeline = nil;
```

### Testing

- Enable depth peeling: `renderer->SetUseDepthPeeling(true)`
- Render a scene with two overlapping translucent spheres
- Verify `RenderTranslucentGeometry()` returns > 0
- Verify correct front-to-back compositing (no sorting artifacts)

---

## Fix #2: Peel Pass Missing Texture Bindings

### Problem

The peel fragment shader reads two textures:

```metal
fragment PeelPassOutput fragment_peel(
    ...
    texture2d<float, access::read> prevFrontTex [[texture(1)]],
    texture2d<float, access::read> prevDepthTex [[texture(2)]])
```

The depth peeler sets these on the render window:

```cpp
renWin->PeelFrontTexture = (__bridge void*)frontSrc;
renWin->PeelDepthTexture = (__bridge void*)depthSrc;
```

But `RebuildRenderBundle()` never binds them. The peel mode 2 path only records the pipeline:

```cpp
else if (peelMode == 2 && this->Internals->TrianglePeelPipeline)
{
    recordPipeline(this->Internals->TrianglePeelPipeline);
}
```

**Impact:** Metal validation error / crash on the first peel iteration.

### File: `vtkMetalPolyDataMapper.mm` — `RebuildRenderBundle()`

### Change

After the pipeline selection block for triangles, add texture bindings when in peel mode 2.

**Find this block in `RebuildRenderBundle()`:**

```cpp
if (peelMode == 1 && this->Internals->TriangleInitPeelPipeline)
{
    recordPipeline(this->Internals->TriangleInitPeelPipeline);
}
else if (peelMode == 2 && this->Internals->TrianglePeelPipeline)
{
    recordPipeline(this->Internals->TrianglePeelPipeline);
}
else
{
    recordPipeline(this->Internals->TrianglePipeline);
}
```

**Replace with:**

```cpp
if (peelMode == 1 && this->Internals->TriangleInitPeelPipeline)
{
    recordPipeline(this->Internals->TriangleInitPeelPipeline);
}
else if (peelMode == 2 && this->Internals->TrianglePeelPipeline)
{
    recordPipeline(this->Internals->TrianglePeelPipeline);

    // Bind previous-frame peel textures required by fragment_peel
    vtkMetalRenderWindow* peelRenWin =
        vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());
    if (peelRenWin)
    {
        id<MTLTexture> prevFront =
            (__bridge id<MTLTexture>)peelRenWin->PeelFrontTexture;
        id<MTLTexture> prevDepth =
            (__bridge id<MTLTexture>)peelRenWin->PeelDepthTexture;
        if (prevFront)
        {
            recordFTex(prevFront, 1);
        }
        if (prevDepth)
        {
            recordFTex(prevDepth, 2);
        }
    }
}
else
{
    recordPipeline(this->Internals->TrianglePipeline);
}
```

### File: `vtkMetalRenderWindow.h`

The `PeelFrontTexture` and `PeelDepthTexture` members are already declared as `void*` in the protected section, and `vtkMetalPolyDataMapper` is already a friend. No header change needed.

### Testing

- After Fix #1, enable depth peeling with multiple translucent layers
- Run with Metal validation layer enabled (`MTL_DEBUG_LAYER=1`)
- Verify no "texture not bound" validation errors
- Verify peel iterations produce correct progressive refinement

### Dependency

Requires Fix #1 (depth peeler must actually run to reach this code path).

---

## Fix #3: Cell ID Buffer — Per-Primitive vs Per-Vertex

### Problem

The vertex shader reads cell IDs by `vertex_id`:

```metal
out.cellId = cellIds[vertex_id];
```

But `TriangleCellIdBuffer` has one entry per **triangle** (from `trianglePrimToCell`), not per vertex. For non-indexed drawing, `vertex_id` ranges 0 to `numTriangles*3 - 1`, but the buffer only has `numTriangles` entries → **out-of-bounds GPU read**.

For indexed drawing, `vertex_id` is the deduplicated vertex index, which has no relationship to triangle count → **wrong cell IDs**.

The same bug affects `LineCellIdBuffer` (one entry per segment, indexed by vertex_id).

### Solution

Expand cell IDs to per-vertex during geometry build. Remove the `cellToPrimitive` compute dispatch entirely — it is unnecessary if per-vertex IDs are built on the CPU.

### File: `vtkMetalPolyDataMapper.mm` — `BuildGeometryBuffers()`

### Change A: Triangle cell IDs (CPU path)

**In the non-indexed triangle path**, after the inner `for (int j = 0; j < 3; ++j)` loop that emits positions/normals/colors, the code currently does:

```cpp
trianglePrimToCell.push_back(static_cast<uint32_t>(polyCellIdx));
```

This is outside the `for (int j = 0; j < 3; ++j)` loop, so it pushes one entry per triangle.

**Replace with per-vertex cell IDs.** Add a new vector alongside `trianglePrimToCell`:

At the top of `BuildGeometryBuffers()`, add:

```cpp
std::vector<uint32_t> triangleVertexCellIds;  // per-vertex cell IDs for picking
```

**In the indexed path** (inside `if (useIndexBuffer)`), after emitting each vertex:

```cpp
// AFTER: triangleIndices.push_back(vidx);  (or it->second for cached)
// ADD:
triangleVertexCellIds.push_back(static_cast<uint32_t>(polyCellIdx));
```

This must be added in **both** branches (new vertex and cached vertex):

```cpp
if (it != triVertexMap.end())
{
    triangleIndices.push_back(it->second);
    triangleVertexCellIds.push_back(static_cast<uint32_t>(polyCellIdx));
}
else
{
    uint32_t vidx = static_cast<uint32_t>(positions.size() / 3);
    triVertexMap[tri[j]] = vidx;
    triangleIndices.push_back(vidx);
    triangleVertexCellIds.push_back(static_cast<uint32_t>(polyCellIdx));
    // ... emit position, normal, color, uv ...
}
```

**In the non-indexed path**, inside the `for (int j = 0; j < 3; ++j)` loop:

```cpp
for (int j = 0; j < 3; ++j)
{
    // ... emit position, normal, color, uv ...
    triangleVertexCellIds.push_back(static_cast<uint32_t>(polyCellIdx));
}
```

**Remove the old per-triangle push:**

```cpp
// DELETE:
trianglePrimToCell.push_back(static_cast<uint32_t>(polyCellIdx));
```

### Change B: Line cell IDs (CPU path)

**In the cell-coloring line path**, replace:

```cpp
linePrimToCell.push_back(static_cast<uint32_t>(lineCellIdx));
```

with per-vertex IDs. Add at the top:

```cpp
std::vector<uint32_t> lineVertexCellIds;
```

In the segment emission loop:

```cpp
for (vtkIdType i = 0; i < npts - 1; ++i)
{
    lineIndices.push_back(base + i);
    lineIndices.push_back(base + i + 1);
    lineVertexCellIds.push_back(static_cast<uint32_t>(lineCellIdx));
    lineVertexCellIds.push_back(static_cast<uint32_t>(lineCellIdx));
}
```

**In the point-coloring line path**, same change:

```cpp
for (vtkIdType i = 0; i < npts - 1; ++i)
{
    lineIndices.push_back(pointMap[pts[i]]);
    lineIndices.push_back(pointMap[pts[i + 1]]);
    lineVertexCellIds.push_back(static_cast<uint32_t>(lineCellIdx));
    lineVertexCellIds.push_back(static_cast<uint32_t>(lineCellIdx));
}
```

### Change C: Wireframe edge cell IDs (CPU path)

In the wireframe polygon edge loop:

```cpp
lineIndices.push_back(idx0);
lineIndices.push_back(idx1);
// REPLACE:
// linePrimToCell.push_back(static_cast<uint32_t>(polyCellIdx));
// WITH:
lineVertexCellIds.push_back(static_cast<uint32_t>(polyCellIdx));
lineVertexCellIds.push_back(static_cast<uint32_t>(polyCellIdx));
```

### Change D: Edge overlay cell IDs (CPU path)

Add at top:

```cpp
std::vector<uint32_t> edgeVertexCellIds;
```

In the edge emission loop:

```cpp
for (const auto& kv : uniqueEdges)
{
    // ...
    edgeIndices.push_back(addEdgeVertex(a, edgeCellId));
    edgeIndices.push_back(addEdgeVertex(b, edgeCellId));
    // REPLACE:
    // edgePrimToCell.push_back(edgeCellId);
    // WITH:
    edgeVertexCellIds.push_back(edgeCellId);
    edgeVertexCellIds.push_back(edgeCellId);
}
```

### Change E: Buffer creation

**Replace the triangle buffer creation block:**

```cpp
// BEFORE:
if (!trianglePrimToCell.empty())
{
    id<MTLBuffer> primToCell = [device newBufferWithBytes:trianglePrimToCell.data() ...];
    vtkMetalMRC::AssignConsumed(this->Internals->TrianglePrimitiveToCellBuffer, primToCell);
    id<MTLBuffer> cellIdOut = [device newBufferWithLength:trianglePrimToCell.size() * sizeof(uint32_t) ...];
    vtkMetalMRC::AssignConsumed(this->Internals->TriangleCellIdBuffer, cellIdOut);
    this->Internals->TrianglePrimitiveCount = trianglePrimToCell.size();
}

// AFTER:
if (!triangleVertexCellIds.empty())
{
    id<MTLBuffer> cellIdBuf = [device
        newBufferWithBytes:triangleVertexCellIds.data()
        length:triangleVertexCellIds.size() * sizeof(uint32_t)
        options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->TriangleCellIdBuffer, cellIdBuf);
}
```

**Same for lines:**

```cpp
// BEFORE:
if (!linePrimToCell.empty()) { ... }

// AFTER:
if (!lineVertexCellIds.empty())
{
    id<MTLBuffer> cellIdBuf = [device
        newBufferWithBytes:lineVertexCellIds.data()
        length:lineVertexCellIds.size() * sizeof(uint32_t)
        options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->LineCellIdBuffer, cellIdBuf);
}
```

**Same for edges:**

```cpp
if (!edgeVertexCellIds.empty())
{
    id<MTLBuffer> cellIdBuf = [device
        newBufferWithBytes:edgeVertexCellIds.data()
        length:edgeVertexCellIds.size() * sizeof(uint32_t)
        options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->EdgeCellIdBuffer, cellIdBuf);
}
```

### Change F: Remove compute dispatches

**Delete all three `DispatchCellToPrimitive` calls** at the end of `BuildGeometryBuffers()`:

```cpp
// DELETE ALL THREE:
if (this->Internals->CellToPrimitivePipeline && this->Internals->TrianglePrimitiveCount > 0 && ...) { ... }
if (this->Internals->CellToPrimitivePipeline && this->Internals->LinePrimitiveCount > 0 && ...) { ... }
if (this->Internals->CellToPrimitivePipeline && this->Internals->EdgeCellIdBuffer && ...) { ... }
```

### Change G: GPU tessellation path

The GPU `polygonToTriangle` kernel writes one cell ID per triangle into `TriangleCellIdBuffer`. The shader then reads `cellIds[vertex_id]`. Since the GPU path uses indexed drawing (vertex_id = point index, not triangle index), this is also wrong.

**For the GPU tess path**, the cell ID buffer must be expanded to per-point. After the compute dispatch completes, expand on CPU:

```cpp
if (cmdBuf.status == MTLCommandBufferStatusCompleted)
{
    // ... existing index buffer assignment ...

    // Expand per-triangle cell IDs to per-vertex (per-point) cell IDs
    // The GPU wrote numTris cell IDs; we need numPolyPts cell IDs
    // For indexed drawing, vertex_id = point index, so we need
    // a per-point cell ID. Use the first triangle that references each point.
    const uint32_t* triCellIds =
        (const uint32_t*)[this->Internals->TriangleCellIdBuffer contents];
    const uint32_t* connData =
        (const uint32_t*)[this->Internals->TessOutputConnectivityBuffer contents];

    std::vector<uint32_t> pointCellIds(numPolyPts, 0);
    std::vector<bool> pointAssigned(numPolyPts, false);
    for (vtkIdType t = 0; t < numTris; ++t)
    {
        uint32_t cid = triCellIds[t];
        for (int v = 0; v < 3; ++v)
        {
            uint32_t ptIdx = connData[t * 3 + v];
            if (ptIdx < (uint32_t)numPolyPts && !pointAssigned[ptIdx])
            {
                pointCellIds[ptIdx] = cid;
                pointAssigned[ptIdx] = true;
            }
        }
    }

    id<MTLBuffer> perPointCellIds = [device
        newBufferWithBytes:pointCellIds.data()
        length:pointCellIds.size() * sizeof(uint32_t)
        options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->TriangleCellIdBuffer, perPointCellIds);

    gpuTessUsed = true;
}
```

### Change H: Add `+1` offset in CPU-built cell IDs

The shader expects 1-based cell IDs (0 = background):

```metal
out.cellId = cellIds[vertex_id];
// ...
out.ids = uint4(in.cellId, in.propId, 1u, 0u);
```

The `cellToPrimitive` kernel added `+1`:

```metal
cellIds[gid] = primitiveToCell[gid] + cellIdOffset + 1u;
```

Since we removed the kernel, add `+1` in the CPU vectors:

```cpp
triangleVertexCellIds.push_back(static_cast<uint32_t>(polyCellIdx) + 1u);
lineVertexCellIds.push_back(static_cast<uint32_t>(lineCellIdx) + 1u);
edgeVertexCellIds.push_back(edgeCellId + 1u);
```

### Testing

- Render a scene with known cell IDs
- Use `GetIdsData()` to read back the picking texture
- Verify cell IDs match expected values for triangles, lines, and edges
- Verify no GPU validation errors (enable `MTL_DEBUG_LAYER=1`)
- Test with both indexed and non-indexed triangle paths
- Test with GPU tessellation path (large mesh with normals, >1000 points)

---

## Fix #4: Edge Shader Missing `LightUniforms` Binding

### Problem

`fragment_edge_main` declares `constant LightUniforms& lights [[buffer(1)]]` but the edge render bundle never binds `LightUniformBuffer` at fragment index 1. Metal requires all declared buffers to be bound.

### Solution

Either bind the buffer, or remove the unused parameter from the shader.

**Option A (preferred — remove unused parameter):**

### File: `MetalShaders.metal` (shader string)

```metal
// BEFORE:
fragment FragmentOutput fragment_edge_main(VertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant float4& edgeColor [[buffer(4)]])

// AFTER:
fragment FragmentOutput fragment_edge_main(VertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant float4& edgeColor [[buffer(4)]])
```

The function body does not use `lights`, so no other changes needed.

**Option B (bind the buffer):**

### File: `vtkMetalPolyDataMapper.mm` — `RebuildRenderBundle()`, edge overlay section

Add after the `MaterialUniformBuffer` binding:

```cpp
if (this->Internals->MaterialUniformBuffer)
{
    recordFBuf(this->Internals->MaterialUniformBuffer, 0, 0);
}
// ADD:
if (this->Internals->LightUniformBuffer)
{
    recordFBuf(this->Internals->LightUniformBuffer, 0, 1);
}
```

### Testing

- Render a surface with `SetEdgeVisibility(true)`
- Verify no Metal validation errors
- Verify edges render with correct color

---

## Fix #5: Missing `setFrontFacingWinding`

### Problem

Metal's default front-facing winding is `MTLWindingClockwise`. VTK uses counter-clockwise front faces. Without explicit winding, backface culling is inverted — front faces are culled and back faces are rendered.

### File: `vtkMetalRenderer.mm` — `DeviceRender()`

### Change

In the opaque pass, after setting the depth stencil state, add winding:

```cpp
[encoder setDepthStencilState:sOpaqueDepthState];

// ADD:
[encoder setFrontFacingWinding:MTLWindingCounterClockwise];
```

Do the same in the translucent fallback pass and the volume pass.

### File: `vtkMetalPolyDataMapper.mm` — `RebuildRenderBundle()`

The render bundle records cull mode but not winding. Add a new command type or set winding before replay.

**Simplest approach:** Set winding in `RenderPiece()` before replaying the bundle:

```cpp
// In RenderPiece(), before the bundleValid check:
[encoder setFrontFacingWinding:MTLWindingCounterClockwise];
```

This is set once per mapper per frame, which is negligible overhead.

### Testing

- Render a closed mesh (e.g., sphere) with backface culling enabled
- Verify the outside surface is visible and the inside is culled
- Compare with OpenGL backend rendering

---

## Fix #6: Main Pipelines Missing Blending

### Problem

The triangle, line, edge, and point pipelines do not enable blending. Actor opacity < 1 renders as opaque. The translucent fallback path in the renderer also uses these pipelines, so it renders opaque too.

### Solution

Enable alpha blending on all surface pipelines. For correct transparency, you need separate opaque and transparent pipeline variants, but as a first pass, enable blending on all pipelines.

### File: `vtkMetalPolyDataMapper.mm` — `EnsurePipelineStates()`

### Change

After setting the pixel formats and before creating the pipeline:

```cpp
pipelineDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
pipelineDesc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
pipelineDesc.rasterSampleCount = sampleCount;

// ADD: Enable alpha blending for transparency support
pipelineDesc.colorAttachments[0].blendingEnabled = YES;
pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
pipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
pipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
```

Apply the same blending configuration to:

- `EnsurePointPipelineStates()` — both `PointPipeline` and `PointShapedPipeline`
- `EnsureEdgePipelineState()`
- `EnsureThickLinePipelineState()`
- `EnsureRoundCapLinePipelineState()`
- `EnsureMiterJoinLinePipelineState()`

### Note on depth writing

With blending enabled, transparent objects should ideally not write to the depth buffer. However, the depth-stencil state is set by the renderer, not the pipeline. The renderer already uses `sReadOnlyDepthState` (depth write disabled) for the translucent pass. For the opaque pass, depth write is enabled, which is correct for opaque objects.

If an actor has opacity < 1 but is rendered in the opaque pass (because `HasTranslucentPolygonalGeometry()` returns false), it will write depth and blend. This is acceptable as a first pass.

### Testing

- Render a translucent sphere (opacity 0.5) over an opaque background
- Verify the background shows through
- Render two overlapping translucent objects
- Verify blending is visible (may have sorting artifacts — that's expected without depth peeling)

---

## Fix #7: Wireframe-Only Polygons Don't Set Line Draw State

### Problem

When `representation == VTK_WIREFRAME` and the polydata has only polygons (no explicit lines), the CPU path appends polygon edges to `lineIndices` but only sets `HasLines`, `LineIndexCount`, etc. inside the `if (lines && lines->GetNumberOfCells() > 0)` block. If there are no explicit lines, these fields remain at their defaults (false/0), and wireframe doesn't draw.

### File: `vtkMetalPolyDataMapper.mm` — `BuildGeometryBuffers()`

### Change

After the `if (lines && lines->GetNumberOfCells() > 0)` block closes, and before the closing brace of `if (!gpuTessUsed)`, add a synchronization block:

```cpp
} // end if (lines && lines->GetNumberOfCells() > 0)

// ADD: Synchronize line state for wireframe polygon edges
// that were added outside the explicit-lines block.
if (!lineIndices.empty() && !this->Internals->HasLines)
{
    this->Internals->LineIndexCount = lineIndices.size();
    this->Internals->HasLines = true;
    this->Internals->LinePrimitiveCount = lineVertexCellIds.size() / 2;
    this->Internals->ThickLineSegmentCount = lineIndices.size() / 2;
    this->Internals->RoundCapLineSegmentCount = this->Internals->ThickLineSegmentCount;
    this->Internals->MiterJoinLineSegmentCount = this->Internals->ThickLineSegmentCount;
}

} // end if (!gpuTessUsed)
```

### Testing

- Create a `vtkCubeSource` (polys only, no lines)
- Set representation to wireframe
- Verify edges render
- Set line width > 1 and verify thick wireframe renders

---

## Fix #8: Scalar/LUT Changes Don't Rebuild Geometry

### Problem

`RenderPiece()` checks `input->GetMTime()` for geometry dirty detection but does not check mapper-level scalar state (scalar visibility, scalar mode, color mode, lookup table, color array name). Changing these does not trigger a geometry rebuild, so colors remain stale.

### File: `vtkMetalPolyDataMapper.mm` — `RenderPiece()`

### Change

Add a cached scalar MTime to `vtkMetalPolyDataMapperInternals`:

```cpp
// ADD to vtkMetalPolyDataMapperInternals:
vtkMTimeType CachedScalarMTime = 0;
```

Reset it in `ReleaseBuffers()`:

```cpp
CachedScalarMTime = 0;
```

In `RenderPiece()`, compute the scalar MTime and include it in the dirty check:

```cpp
vtkIdType currentMTime = input->GetMTime();
vtkMTimeType extraMTime = this->ExtraAttributesMTime.GetMTime();

// ADD: Compute scalar configuration MTime
vtkMTimeType scalarMTime = this->GetMTime();
if (this->GetLookupTable())
{
    scalarMTime = std::max(scalarMTime,
        static_cast<vtkMTimeType>(this->GetLookupTable()->GetMTime()));
}

int representation = act->GetProperty()->GetRepresentation();
bool edgeVisibility = act->GetProperty()->GetEdgeVisibility();

if (currentMTime != this->Internals->CachedInputMTime ||
    representation != this->Internals->CachedRepresentation ||
    edgeVisibility != this->Internals->CachedEdgeVisibility ||
    extraMTime != this->Internals->CachedExtraAttributesMTime ||
    scalarMTime != this->Internals->CachedScalarMTime)  // ADD
{
    this->Internals->ReleaseBuffers();
    this->Internals->CachedInputMTime = currentMTime;
    this->Internals->CachedRepresentation = representation;
    this->Internals->CachedEdgeVisibility = edgeVisibility;
    this->Internals->CachedExtraAttributesMTime = extraMTime;
    this->Internals->CachedScalarMTime = scalarMTime;  // ADD
    this->BuildGeometryBuffers((void*)device, input, act);
}
```

### Testing

- Render a mesh with scalar coloring
- Call `mapper->SetScalarVisibility(false)` → verify colors change to actor color
- Call `mapper->SetLookupTable(newLUT)` → verify colors update
- Call `mapper->SetColorModeToDirectScalars()` → verify colors update

---

## Fix #9: Cell ID Offsets for Mixed Cell Types

### Problem

In `BuildGeometryBuffers()`, `polyCellIdx` starts at 0 for polygons. But in `vtkPolyData`, cell IDs are ordered: verts → lines → polys → strips. If the dataset has verts or lines, polygon cell IDs should be offset by `numVerts + numLines`.

Similarly, `lineCellIdx` should be offset by `numVerts`.

### File: `vtkMetalPolyDataMapper.mm` — `BuildGeometryBuffers()`

### Change

Compute offsets at the top of the function:

```cpp
// ADD after getting polydata:
vtkIdType vertCellOffset = 0;
vtkIdType lineCellOffset = polydata->GetNumberOfVerts();
vtkIdType polyCellOffset = lineCellOffset + polydata->GetNumberOfLines();
```

**In the CPU polygon path**, change:

```cpp
// BEFORE:
vtkIdType polyCellIdx = 0;

// AFTER:
vtkIdType polyCellIdx = polyCellOffset;
```

**In the CPU line path**, change:

```cpp
// BEFORE:
vtkIdType lineCellIdx = 0;

// AFTER:
vtkIdType lineCellIdx = lineCellOffset;
```

**In the GPU tessellation paths**, change `cellIdOffset`:

```cpp
// BEFORE:
tessParams.cellIdOffset = 0;

// AFTER:
tessParams.cellIdOffset = static_cast<uint32_t>(polyCellOffset);
```

```cpp
// BEFORE (wireframe):
wParams.cellIdOffset = 0;

// AFTER:
wParams.cellIdOffset = static_cast<uint32_t>(polyCellOffset);
```

```cpp
// BEFORE (polyline):
lParams.cellIdOffset = 0;

// AFTER:
lParams.cellIdOffset = static_cast<uint32_t>(lineCellOffset);
```

```cpp
// BEFORE (edge overlay):
eParams.cellIdOffset = 0;

// AFTER:
eParams.cellIdOffset = static_cast<uint32_t>(polyCellOffset);
```

### Testing

- Create a polydata with verts + lines + polys
- Enable picking
- Pick a polygon → verify cell ID = numVerts + numLines + polyIndex
- Pick a line → verify cell ID = numVerts + lineIndex

---

## Fix #10: Extra Attribute Buffer Bounds

### Problem

Extra attribute buffers are created with the original data array size (`numTuples`), but the rendered vertex count may differ due to deduplication or duplication. The vertex shader indexes by `vertex_id`, which can exceed the buffer size.

### Solution

Build extra attribute arrays in parallel with `positions`, emitting one value per rendered vertex.

### File: `vtkMetalPolyDataMapper.mm` — `BuildGeometryBuffers()`

### Change

This is a larger refactor. The key principle: wherever the code emits a vertex (pushes to `positions`), it must also emit the corresponding extra attribute value.

**Step 1:** At the top of `BuildGeometryBuffers()`, create parallel vectors:

```cpp
// ADD:
std::unordered_map<std::string, std::vector<float>> extraAttrArrays;
for (const auto& attr : this->ExtraAttributes)
{
    extraAttrArrays[attr.first] = std::vector<float>();
}
```

**Step 2:** Create a helper lambda to emit extra attribute values for a point:

```cpp
auto emitExtraAttrsForPoint = [&](vtkIdType pointId) {
    for (const auto& attr : this->ExtraAttributes)
    {
        vtkDataArray* da = nullptr;
        if (attr.second.FieldAssociation == vtkDataObject::FIELD_ASSOCIATION_POINTS)
        {
            da = polydata->GetPointData()->GetArray(attr.second.DataArrayName.c_str());
        }
        if (da && pointId < da->GetNumberOfTuples())
        {
            int numComps = da->GetNumberOfComponents();
            int comp = attr.second.ComponentNumber;
            if (comp < 0)
            {
                double* tuple = da->GetTuple(pointId);
                for (int c = 0; c < numComps; ++c)
                {
                    extraAttrArrays[attr.first].push_back(static_cast<float>(tuple[c]));
                }
            }
            else
            {
                extraAttrArrays[attr.first].push_back(
                    static_cast<float>(da->GetComponent(pointId, comp)));
            }
        }
        else
        {
            int effectiveComps = (attr.second.ComponentNumber < 0)
                ? (da ? da->GetNumberOfComponents() : 1) : 1;
            for (int c = 0; c < effectiveComps; ++c)
            {
                extraAttrArrays[attr.first].push_back(0.0f);
            }
        }
    }
};
```

**Step 3:** Call `emitExtraAttrsForPoint(pointId)` at every site that emits a vertex. For cell-associated attributes, emit the cell's value for each vertex of that cell.

**Step 4:** Replace the old extra attribute buffer creation block at the end:

```cpp
// REPLACE the entire "for (auto& itr : this->ExtraAttributes)" block with:
for (const auto& attr : this->ExtraAttributes)
{
    const auto& attrData = extraAttrArrays[attr.first];
    if (attrData.empty())
    {
        continue;
    }
    id<MTLBuffer> attrBuf = [device
        newBufferWithBytes:attrData.data()
        length:attrData.size() * sizeof(float)
        options:MTLResourceStorageModeShared];
    vtkMetalMRC::AssignConsumed(this->Internals->ExtraAttributeBuffers[attr.first], attrBuf);

    int numComps = 1;
    vtkDataArray* da = nullptr;
    if (attr.second.FieldAssociation == vtkDataObject::FIELD_ASSOCIATION_POINTS)
    {
        da = polydata->GetPointData()->GetArray(attr.second.DataArrayName.c_str());
    }
    else if (attr.second.FieldAssociation == vtkDataObject::FIELD_ASSOCIATION_CELLS)
    {
        da = polydata->GetCellData()->GetArray(attr.second.DataArrayName.c_str());
    }
    if (da)
    {
        numComps = (attr.second.ComponentNumber < 0) ? da->GetNumberOfComponents() : 1;
    }
    this->Internals->ExtraAttributeComponentCounts[attr.first] = numComps;
}
```

### Testing

- Map a point-associated array to a vertex attribute
- Render with cell coloring (non-indexed path) → verify no GPU fault
- Render with point coloring (indexed path) → verify correct attribute values
- Map a cell-associated array → verify values are expanded per-vertex

---

## Fix #11: PropId Always 0

### Problem

`PropIdBuffer` is created with value 0 and never updated. The shader outputs `propId + 1u`, so all actors pick as propId = 1.

### File: `vtkMetalPolyDataMapper.mm`

### Change

In `RenderPiece()`, after the `PropIdBuffer` is guaranteed to exist (it's created in `BuildGeometryBuffers`), update it with the actor's pick ID:

```cpp
// ADD in RenderPiece(), after UpdateActorTexture() call:
if (this->Internals->PropIdBuffer)
{
    // Use the actor's address as a unique prop ID, or a registered pick ID
    // VTK uses vtkProp::GetPickable() and composite IDs for picking
    uint32_t propId = 0;
    if (act)
    {
        // Use a simple hash of the actor pointer as prop ID
        // In a full implementation, this would use vtkProp's registered ID
        propId = static_cast<uint32_t>(
            reinterpret_cast<std::uintptr_t>(act) & 0xFFFFFFFF);
    }
    memcpy([this->Internals->PropIdBuffer contents], &propId, sizeof(uint32_t));
}
```

For the batched mapper, the prop ID should come from the `BatchElement::FlatIndex`. In `vtkMetalBatchedPolyDataMapper::RenderPiece()`, before calling `mapper->RenderPiece()`:

```cpp
// Set per-element prop ID
if (mapper->Internals->PropIdBuffer)  // requires friend or accessor
{
    uint32_t propId = elem->FlatIndex;
    memcpy([mapper->Internals->PropIdBuffer contents], &propId, sizeof(uint32_t));
}
```

Since `Internals` is private, add a public accessor to `vtkMetalPolyDataMapper`:

```cpp
// ADD to vtkMetalPolyDataMapper.h public section:
void SetPropId(uint32_t propId);
```

```cpp
// ADD to vtkMetalPolyDataMapper.mm:
void vtkMetalPolyDataMapper::SetPropId(uint32_t propId)
{
    if (this->Internals->PropIdBuffer)
    {
        memcpy([this->Internals->PropIdBuffer contents], &propId, sizeof(uint32_t));
    }
}
```

### Testing

- Render two actors
- Pick each one
- Verify different prop IDs are returned
- In batched mode, verify composite IDs match flat indices

---

## Fix #12: Peel Pipeline Missing Sample Count

### Problem

`EnsurePeelPipelineStates()` does not set `rasterSampleCount`. If MSAA is active, the peel pipelines default to sample count 1 and mismatch the render pass.

### File: `vtkMetalPolyDataMapper.mm` — `EnsurePeelPipelineStates()`

### Change

Add sample count to both pipeline descriptors:

```cpp
// In the init peel pipeline block, after setting inputPrimitiveTopology:
desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
// ADD:
desc.rasterSampleCount = this->Internals->CachedSampleCount > 0
    ? this->Internals->CachedSampleCount : 1;

// In the main peel pipeline block, after setting inputPrimitiveTopology:
desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
// ADD:
desc.rasterSampleCount = this->Internals->CachedSampleCount > 0
    ? this->Internals->CachedSampleCount : 1;
```

### Testing

- Enable MSAA (`renWin->SetMultiSamples(4)`)
- Enable depth peeling
- Verify no pipeline/render pass mismatch validation errors

---

## Fix #13: Depth Peeler Texture Leak on Resize

### Problem

`CreateTextures()` overwrites texture pointers without releasing old textures. Every window resize leaks 6 textures.

### File: `vtkMetalDepthPeeler.mm` — `CreateTextures()`

### Change

Add release calls at the top:

```cpp
void vtkMetalDepthPeeler::CreateTextures(id<MTLDevice> device, int width, int height)
{
    // ADD: Release old textures before creating new ones
    [this->FrontPeelA release];   this->FrontPeelA = nil;
    [this->FrontPeelB release];   this->FrontPeelB = nil;
    [this->BackPeelTemp release]; this->BackPeelTemp = nil;
    [this->BackAccum release];    this->BackAccum = nil;
    [this->DepthPeelA release];   this->DepthPeelA = nil;
    [this->DepthPeelB release];   this->DepthPeelB = nil;

    this->CurrentWidth = width;
    this->CurrentHeight = height;

    // ... rest of creation code unchanged ...
```

### Testing

- Resize the window repeatedly
- Monitor memory usage → should not grow
- Run with Metal validation → no double-release errors

---

## Fix #14: Depth Peeler Missing Viewport in Fullscreen Passes

### Problem

The back-blend and composite fullscreen passes do not set a viewport. Metal's default viewport is (0,0,0,0), so these passes render nothing.

### File: `vtkMetalDepthPeeler.mm` — `RenderTranslucentGeometry()`

### Change

**In the back-blend pass**, after creating the encoder:

```cpp
id<MTLRenderCommandEncoder> encoder =
    [commandBuffer renderCommandEncoderWithDescriptor:rpd];
encoder.label = @"VTK Depth Peeling - Back Blend";

// ADD:
MTLViewport vp;
vp.originX = 0; vp.originY = 0;
vp.width = width; vp.height = height;
vp.znear = 0.0; vp.zfar = 1.0;
[encoder setViewport:vp];

[encoder setRenderPipelineState:this->BackBlendPipeline];
```

**In the composite pass**, same addition:

```cpp
id<MTLRenderCommandEncoder> encoder =
    [commandBuffer renderCommandEncoderWithDescriptor:rpd];
encoder.label = @"VTK Depth Peeling - Composite";

// ADD:
MTLViewport vp;
vp.originX = 0; vp.originY = 0;
vp.width = width; vp.height = height;
vp.znear = 0.0; vp.zfar = 1.0;
[encoder setViewport:vp];

[encoder setRenderPipelineState:this->CompositePipeline];
```

### Testing

- Enable depth peeling
- Verify the final composite shows translucent geometry blended over opaque
- Verify back-blend accumulation is visible (multiple layers)

---

## Implementation Order

```
Fix #1  (depth peeler pipeline)     ← no dependencies
Fix #14 (depth peeler viewport)     ← no dependencies
Fix #13 (depth peeler texture leak) ← no dependencies
Fix #5  (winding order)             ← no dependencies
Fix #6  (blending)                  ← no dependencies
Fix #4  (edge light binding)        ← no dependencies
Fix #7  (wireframe line state)      ← no dependencies
Fix #9  (cell ID offsets)           ← no dependencies
Fix #8  (scalar dirty tracking)     ← no dependencies
Fix #11 (prop ID)                   ← no dependencies
Fix #3  (cell ID per-vertex)        ← depends on #9 (offsets)
Fix #10 (extra attribute bounds)    ← depends on #3 (parallel emit)
Fix #2  (peel texture bindings)     ← depends on #1 (peeler must run)
Fix #12 (peel sample count)         ← depends on #1 (peeler must run)
```

Recommended order for a single engineer:

1. **Day 1:** Fixes #1, #14, #13, #5, #6 (independent, quick wins)
2. **Day 2:** Fixes #4, #7, #9, #8, #11 (independent, moderate)
3. **Day 3–4:** Fix #3 (large refactor, per-vertex cell IDs)
4. **Day 4–5:** Fixes #10, #2, #12 (depend on #3 and #1)

---

## Verification Checklist

After all fixes:

- [ ] Opaque rendering matches OpenGL backend (winding, culling)
- [ ] Translucent rendering shows blending (opacity < 1 works)
- [ ] Depth peeling produces correct order-independent transparency
- [ ] Picking returns correct cell IDs for triangles, lines, edges, points
- [ ] Picking returns distinct prop IDs for different actors
- [ ] Wireframe mode works for polygon-only meshes
- [ ] Edge visibility works without validation errors
- [ ] Scalar visibility / LUT changes update colors immediately
- [ ] Mixed cell-type datasets (verts + lines + polys) have correct cell IDs
- [ ] Extra vertex attributes don't cause GPU faults with cell coloring
- [ ] Window resize doesn't leak depth peeler textures
- [ ] MSAA + depth peeling doesn't produce pipeline mismatch errors
- [ ] Metal validation layer shows zero errors for all test scenes
