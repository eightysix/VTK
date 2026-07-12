# Metal Backend: Feature Parity with WebGPU — Implementation Plan

Feature-by-feature plan for bringing `vtkMetalPolyDataMapper` to full parity with `vtkWebGPUPolyDataMapper`.

**Reference**: WebGPU implementation at `Rendering/WebGPU/vtkWebGPUPolyDataMapper.{h,cxx}`
**Target files**: `Rendering/Metal/vtkMetalPolyDataMapper.{h,mm}`, `Rendering/Metal/Shaders/MetalShaders.metal`

---

## Phase 1: Foundation — Per-Vertex Color & Cull Mode

### 1A. Per-Vertex Color for Surfaces

**Gap**: Triangle/line vertex descriptor only has position + normal. `MapScalars()` is called but colors are only used in the point path. Surfaces cannot be colored by scalar arrays.

**Files**:
- `vtkMetalPolyDataMapper.mm` — `BuildGeometryBuffers()`, `EnsurePipelineStates()`
- `MetalShaders.metal` — `vertex_main`, `fragment_main`

**Implementation**:
1. In `BuildGeometryBuffers()`, after building `VertexPositionBuffer`/`VertexNormalBuffer`, call `this->MapScalars(actor->GetProperty()->GetOpacity(), cellFlag)`.
2. If `cellFlag == 0` (point-associated scalars), normalize the `unsigned char` RGBA colors to `float [0,1]` and create `VertexColorBuffer` (float4 per vertex, duplicated per triangle vertex just like normals).
3. Update `EnsurePipelineStates()` vertex descriptor: add `MTLVertexFormatFloat4` at `bufferIndex = 2` (color attribute).
4. Update `MetalShaders.metal` `VertexIn` struct: add `float4 color [[attribute(2)]]`.
5. Update `VertexOut` struct: add `float4 vertexColor`.
6. In `vertex_main`: pass `in.color` through to `out.vertexColor`.
7. In `fragment_main`: when per-vertex color is available, use `in.vertexColor` instead of `material.diffuseColor.rgb` for the base color.
8. Add a uniform flag (or use a bit in `SceneUniforms.flags`) to tell the fragment shader whether per-vertex color is active.

**WebGPU reference**: `DeducePointCellAttributeAvailability()` (line ~1469), `ReplaceVertexShaderColors()` (line ~3444), `ReplaceFragmentShaderColors()` (line ~3818).

### 1B. Cell Data Coloring

**Gap**: When `cellFlag != 0` from `MapScalars()`, colors are discarded. No `CellColorBuffer` exists.

**Files**:
- `vtkMetalPolyDataMapper.mm` — internals struct, `BuildGeometryBuffers()`
- `MetalShaders.metal` — fragment shader

**Implementation**:
1. Add `id<MTLBuffer> CellColorBuffer` to `vtkMetalPolyDataMapperInternals`.
2. In `BuildGeometryBuffers()`, when `cellFlag != 0`, iterate cells, look up each cell's mapped color, and duplicate the color for each vertex of that cell (fan-triangulated triangles get the cell color for all 3 vertices).
3. Create a separate vertex buffer for cell colors or pack into the existing color buffer with a flag.
4. Fragment shader reads cell color when cell coloring is active.

**WebGPU reference**: `DeducePointCellAttributeAvailability()` lines 1500-1530, `UploadAttributeToGPUBuffer()` for `CELL_COLORS`.

### 1C. Backface/Frontface Culling

**Gap**: Metal only sets `inputPrimitiveTopology` but no explicit cull mode.

**Files**:
- `vtkMetalPolyDataMapper.mm` — `EnsurePipelineStates()`
- `MetalShaders.metal` — no change needed

**Implementation**:
1. In `EnsurePipelineStates()`, after creating the pipeline descriptor, check `actor->GetProperty()->GetBackfaceCulling()` and `GetFrontfaceCulling()`.
2. Set `descriptor.inputPrimitiveTopology` and enable backface culling via the pipeline's `MTLRenderPipelineDescriptor` — specifically, the rasterizer state doesn't have cull mode directly; it's set on `MTLDepthStencilState` or implicit. Actually in Metal, cull mode is set on the render command encoder: `[encoder setCullMode:]`.
3. In `RenderPiece()`, before issuing draw calls, set `[encoder setCullMode:MTLCullModeBack]` or `MTLCullModeFront` based on actor property.
4. Add `CachedActorBackfaceCulling`/`CachedActorFrontfaceCulling` to detect changes.

**WebGPU reference**: `SetupGraphicsPipelines()` lines 2093-2100, `CacheActorRendererProperties()` lines 416-417.

---

## Phase 2: Edge & Wireframe Rendering

### 2A. Wireframe Representation

**Gap**: No `VTK_WIREFRAME` code path. Lines only come from explicit `GetLines()` cells.

**Files**:
- `vtkMetalPolyDataMapper.mm` — `BuildGeometryBuffers()`, `RenderPiece()`
- `MetalShaders.metal` — no new shaders needed

**Implementation**:
1. In `BuildGeometryBuffers()`, when `representation == VTK_WIREFRAME`, extract edges from polygon cells: for each polygon with vertices `[v0, v1, ..., vn]`, emit line segments `(v0,v1), (v1,v2), ..., (vn-1,vn)`. This is the same line deduplication already done for `GetLines()`.
2. In `RenderPiece()`, when representation is `VTK_WIREFRAME`, skip triangle drawing entirely and only draw lines.
3. Store a `CachedRepresentation` and rebuild geometry buffers when representation changes (already partially done).

**WebGPU reference**: WebGPU handles this via `TopologyBindGroupInfos[TOPOLOGY_SOURCE_POLYGON_EDGES]` — the `CellToPrimitiveConverter` extracts polygon edges for wireframe mode.

### 2B. Edge Visibility on Surfaces

**Gap**: No edge overlay rendering when `GetEdgeVisibility()` is true in `VTK_SURFACE` mode.

**Files**:
- `vtkMetalPolyDataMapper.mm` — `BuildGeometryBuffers()`, `RenderPiece()`
- `MetalShaders.metal` — new shaders or modifications

**Implementation**:
1. When `representation == VTK_SURFACE && GetEdgeVisibility()`, after drawing filled triangles, draw wireframe edges on top with a slight depth bias.
2. In `BuildGeometryBuffers()`, when edge visibility is on, also build an edge index buffer from polygon edges (same as wireframe path).
3. Add a dedicated edge pipeline that renders lines with the edge color from `vtkProperty::GetEdgeColor()`.
4. Apply coincident topology offset to push edges slightly forward (already have `CoincidentOffsetUniforms` with `lineFactor`/`lineOffset`).
5. In fragment shader, use anti-aliased line rendering (alpha from distance to edge).

**WebGPU reference**: `ReplaceVertexShaderEdges()` (line 3226), `ReplaceFragmentShaderEdges()` (line 3666) — computes per-vertex edge distances and blends edge color in fragment shader.

### 2C. Triangle Index Buffers

**Gap**: `IndexBuffer` field exists but is never populated. Triangles are non-indexed (3 unique verts per tri).

**Files**:
- `vtkMetalPolyDataMapper.mm` — `BuildGeometryBuffers()`

**Implementation**:
1. Build a shared vertex array (deduplicate positions+normals) and an index buffer referencing shared vertices.
2. Use `[encoder drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:]`.
3. This reduces memory by ~50% for typical meshes and improves vertex cache hit rate.

**WebGPU reference**: WebGPU uses `CellToPrimitiveConverter` which produces indexed geometry via compute shaders.

---

## Phase 3: Line Rendering Variants

### 3A. Thick Lines (No Join)

**Gap**: Lines are always 1px regardless of `GetLineWidth()`.

**Files**:
- `MetalShaders.metal` — new `vertex_thick_line_main` / `fragment_thick_line_main`
- `vtkMetalPolyDataMapper.mm` — new pipeline state, new `id<MTLBuffer> ThickLineVertex/FragmentBuffer`

**Implementation**:
1. For each line segment, emit a screen-space quad (4 vertices per segment). The quad is oriented along the line direction and expanded perpendicular by `lineWidth / 2`.
2. Vertex shader: transform both endpoints to clip space, compute line direction, expand perpendicular in screen space, emit 4 vertices per instance.
3. Fragment shader: simple color output (same as line color).
4. Create `ThickLinePipeline` with `MTLPrimitiveTypeTriangle` topology, no backface culling.
5. In `RenderPiece()`, when `lineWidth > 1` and `lineJoin == NoJoin`, use thick line pipeline instead of basic line pipeline.

**WebGPU reference**: `GFX_PIPELINE_LINES_THICK` — uses instanced quads (4 verts × N instances), `GetDrawCallArgs()` returns `vertexCount=4, instanceCount=vertexCount/2`.

### 3B. Round Cap + Round Join Lines

**Files**:
- `MetalShaders.metal` — new shaders
- `vtkMetalPolyDataMapper.mm` — new pipeline

**Implementation**:
1. For each line segment, emit a quad body (4 verts) + caps (semicircles at each end, approximated with triangle fans, ~14 verts per cap). Total: ~32 verts per segment, instanced.
2. At joints between consecutive segments, emit additional geometry to fill the gap (round join).
3. Fragment shader computes distance from line center for anti-aliasing.
4. Pipeline uses `MTLPrimitiveTypeTriangle`.

**WebGPU reference**: `GFX_PIPELINE_LINES_ROUND_CAP_ROUND_JOIN` — 36 verts per segment instance.

### 3C. Miter Join Lines

**Files**: Same as 3B.

**Implementation**:
1. Similar to thick lines but at joints, extend the quads along the miter direction.
2. Miter limit check: if miter angle is too acute, fall back to bevel.

**WebGPU reference**: `GFX_PIPELINE_LINES_MITER_JOIN` — 4 verts per segment, instanced.

---

## Phase 4: Clipping Planes (Full Implementation)

### 4A. Activate Clipping Plane Support

**Gap**: `ClipPlaneUniforms` buffer is allocated but `numClipPlanes` is always 0.

**Files**:
- `vtkMetalPolyDataMapper.mm` — `UpdateClipPlaneUniforms()`
- `MetalShaders.metal` — already has clip plane logic in shaders

**Implementation**:
1. In `UpdateClipPlaneUniforms()`, actually read clip planes from `vtkRenderer::GetClippingPlanes()` or `vtkPlaneCollection`.
2. Transform each plane equation by the actor's inverse transpose matrix (matching WebGPU).
3. Pack up to 6 planes into `ClipPlaneUniforms` and set `numClipPlanes` to the actual count.
4. The vertex/fragment shaders already compute `clipDistances` and discard — they just need real plane data.

**WebGPU reference**: `UpdateClippingPlanesBuffer()` (line ~2184) reads from `vtkPlanes` and transforms by shift/scale + model-view inverse transpose.

---

## Phase 5: Texture Mapping

### 5A. Complete UV Pipeline with Sampling

**Gap**: UV buffers exist and are passed through vertex shaders but never sampled.

**Files**:
- `vtkMetalPolyDataMapper.mm` — new texture binding code, `BuildGeometryBuffers()`
- `MetalShaders.metal` — new texture sampling in fragment shaders

**Implementation**:
1. In `BuildGeometryBuffers()`, when `pd->GetTCoords()` exists, the UV buffers are already created.
2. Add texture creation/binding: when the actor has a texture (`vtkActor::GetTexture()`), create an `MTLTexture` and `MTLSamplerState`.
3. Add a new buffer binding (or repurpose an existing one) for the texture + sampler.
4. In the fragment shader, add `texture2d<float>` and `sampler` arguments.
5. When texture is present, multiply the fragment color by the sampled texture color.

**WebGPU reference**: `ReplaceFragmentShaderColors()` handles texture coordinate lookup and sampling, `ColorTextureHostResource` manages the GPU texture.

---

## Phase 6: GPU Tessellation (Compute Shader)

### 6A. Cell-to-Primitive Compute Pipeline

**Gap**: CPU fan-triangulation in `BuildGeometryBuffers()`. WebGPU uses a compute shader for polygon → triangle conversion and edge array generation.

**Files**:
- `MetalShaders.metal` — expand `cellToPrimitive` kernel or add new kernel
- `vtkMetalPolyDataMapper.mm` — new compute dispatch code

**Implementation**:
1. Extend the existing `cellToPrimitive` compute kernel to also:
   - Perform polygon → triangle tessellation (fan triangulation)
   - Generate line indices for edges (for wireframe/edge visibility)
   - Generate edge arrays (per-triangle edge visibility flags)
2. Use Metal compute buffers to store tessellated output.
3. Replace CPU-side `BuildGeometryBuffers()` polygon processing with compute dispatch.
4. Keep CPU fallback for simple cases (< threshold vertices).

**WebGPU reference**: `vtkWebGPUCellToPrimitiveConverter` — full compute-based tessellation with edge array output.

---

## Phase 7: Additional Mappers & Infrastructure

### 7A. 2D Mapper

**Files**: New `vtkMetalPolyDataMapper2D.{h,mm}`

**Implementation**:
1. Create `vtkMetalPolyDataMapper2D` inheriting from `vtkPolyDataMapper2D`.
2. Implement `RenderOverlay()` and `RenderOpaqueOverlay()` / `RenderTranslucentOverlay()`.
3. Use 2D viewport coordinates (no perspective projection).
4. Simple shader: position → screen space transform via 2D actor transform.

**WebGPU reference**: `vtkWebGPUPolyDataMapper2D` + `Private/vtkWebGPUPolyDataMapper2DInternals`.

### 7B. Batched Mapper

**Files**: New `vtkMetalBatchedPolyDataMapper.{h,mm}`

**Implementation**:
1. Accumulate multiple actors' geometry into shared vertex/index buffers.
2. Use `CellIdOffsetBuffer` (already exists, hardcoded to 0) to offset cell IDs per actor.
3. Single draw call per batch instead of one per actor.
4. Requires uniform buffer array for per-actor transforms/materials.

**WebGPU reference**: `vtkWebGPUBatchedPolyDataMapper`.

### 7C. Composite Mapper Delegator

**Files**: New `vtkMetalCompositePolyDataMapperDelegator.{h,mm}`

**Implementation**:
1. Delegate geometry processing to the batched mapper.
2. Handle composite/LOD rendering.

**WebGPU reference**: `vtkWebGPUCompositePolyDataMapperDelegator`.

### 7D. Glyph3D Mapper

**Files**: New `vtkMetalGlyph3DMapper.{h,mm}`

**Implementation**:
1. Instanced rendering of glyph geometry.
2. Per-instance transform buffer.
3. Point/glyph attribute passing.

**WebGPU reference**: `vtkWebGPUGlyph3DMapper`.

---

## Phase 8: Advanced Features

### 8A. MSAA

**Files**:
- `vtkMetalRenderWindow.mm` — create multisample textures
- `vtkMetalRenderer.mm` — configure multisample render pass
- `vtkMetalPolyDataMapper.mm` — set `sampleCount` on pipeline descriptors

**Implementation**:
1. Query `MTLDevice` for `maxSampleCount`.
2. Create `MTLTexture` with `textureType = MTLTextureType2DMultisample`.
3. Set `descriptor.sampleCount` on all pipeline descriptors.
4. Configure resolve texture in render pass descriptor.

### 8B. Depth Peeling / Correct Translucency

**Files**:
- `vtkMetalRenderer.mm` — multi-pass rendering
- `MetalShaders.metal` — peeling shaders
- New `vtkMetalDepthPeeler` or equivalent

**Implementation**:
1. Implement OIT (Order-Independent Transparency) via depth peeling.
2. Multiple render passes, each peeling the closest translucent layer.
3. Stencil buffer to track peeled fragments.
4. Final composite pass blends all layers back-to-front.

### 8C. Render Bundles

**Files**:
- `vtkMetalPolyDataMapper.mm` — command buffer caching

**Implementation**:
1. Metal doesn't have an exact equivalent of WebGPU render bundles, but command buffers can be pre-recorded and replayed.
2. Use `MTLCommandBuffer` caching for static geometry.
3. Detect when geometry hasn't changed and replay cached command buffer.

### 8D. Vertex Attribute Mapping

**Files**:
- `vtkMetalPolyDataMapper.h/.mm` — `MapDataArrayToVertexAttribute()`, `RemoveVertexAttributeMapping()`

**Implementation**:
1. Maintain a map of custom vertex attribute names → data arrays.
2. In `BuildGeometryBuffers()`, create additional vertex buffers for custom attributes.
3. Extend vertex descriptor with additional buffer slots.
4. Pass custom attributes through to shaders via a generic mechanism.

**WebGPU reference**: `MapDataArrayToVertexAttribute()` stores mappings, `UploadAttributeToGPUBuffer()` handles upload.

---

## Priority Order

| Priority | Phase | Effort | Impact |
|----------|-------|--------|--------|
| 1 | 1A — Per-vertex color | Small | Critical — surfaces can't be colored |
| 2 | 1B — Cell data coloring | Small | High — cell-based scalars broken |
| 3 | 1C — Cull mode | Trivial | Medium — incorrect face culling |
| 4 | 4A — Clipping planes | Small | Medium — planes exist but do nothing |
| 5 | 2C — Triangle index buffers | Small | Medium — memory/perf improvement |
| 6 | 2A — Wireframe representation | Medium | High — SetRepresentationToWireframe() broken |
| 7 | 2B — Edge visibility | Medium | High — edge overlay missing |
| 8 | 3A — Thick lines | Medium | Medium — line width ignored |
| 9 | 5A — Texture mapping | Medium | Medium — UVs buffered but unused |
| 10 | 3B/3C — Round/miter joins | Large | Low — niche line styles |
| 11 | 6A — GPU tessellation | Large | Medium — moves work off CPU |
| 12 | 8A — MSAA | Medium | Medium — anti-aliasing |
| 13 | 7A — 2D mapper | Medium | Low — 2D overlay support |
| 14 | 7B — Batched mapper | Large | Low — performance optimization |
| 15 | 8B — Depth peeling | Large | Low — correct translucency |
| 16 | 7C — Composite delegator | Large | Low — LOD support |
| 17 | 7D — Glyph3D mapper | Large | Low — glyph instancing |
| 18 | 8C — Render bundles | Large | Low — perf optimization |
| 19 | 8D — Vertex attribute mapping | Medium | Low — custom attributes |

---

## Testing Strategy

For each phase:
1. **Unit visual test**: Create a VTK test program exercising the specific feature.
2. **Comparison screenshot**: Render the same scene with WebGPU and Metal, compare side-by-side.
3. **Regression check**: Ensure existing features (phong lighting, picking, coincident offset) still work.
4. **Performance baseline**: Time `BuildGeometryBuffers()` + `RenderPiece()` before/after.

Key test programs to create/modify:
- `TestVertexColors.cxx` — scalar-mapped surface coloring
- `TestWireframe.cxx` — VTK_WIREFRAME representation
- `TestEdgeVisibility.cxx` — edge overlay on surfaces
- `TestThickLines.cxx` — line width > 1
- `TestClippingPlanes.cxx` — clip plane activation
- `TestTextureMapping.cxx` — texture on polygonal geometry
- `TestBackfaceCulling.cxx` — cull mode toggling
