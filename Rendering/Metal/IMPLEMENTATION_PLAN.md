# Metal Backend: Feature Parity with WebGPU — Implementation Plan

Feature-by-feature plan for bringing `vtkMetalPolyDataMapper` to full parity with `vtkWebGPUPolyDataMapper`.

**Reference**: WebGPU implementation at `Rendering/WebGPU/vtkWebGPUPolyDataMapper.{h,cxx}`
**Target files**: `Rendering/Metal/vtkMetalPolyDataMapper.{h,mm}`, `Rendering/Metal/Shaders/MetalShaders.metal`

---

## Phase 1: Foundation — Per-Vertex Color & Cull Mode ✅ COMPLETED

### 1A. Per-Vertex Color for Surfaces ✅

**Status**: Implemented. `MapScalars()` called early in `BuildGeometryBuffers()`. `SurfaceColorBuffer` (float4 per vertex) created for triangles and lines. Vertex shader reads from `[[buffer(3)]]`. Fragment shader uses `vertexColor` when flags bit 8 is set, replacing `material.ambientColor.rgb` and `material.diffuseColor.rgb`.

### 1B. Cell Data Coloring ✅

**Status**: Implemented. When `cellFlag != 0`: triangle vertices get cell color via `polyCellIdx`; line vertices get cell color with no point deduplication (flat shading via duplicated per-vertex colors).

### 1C. Backface/Frontface Culling ✅

**Status**: Implemented. `[encoder setCullMode:]` set from `GetBackfaceCulling()`/`GetFrontfaceCulling()` before triangle draw calls. Lines always use `MTLCullModeNone`.

---

## Phase 2: Edge & Wireframe Rendering ✅ COMPLETED

### 2A. Wireframe Representation ✅

**Status**: Implemented. When `representation == VTK_WIREFRAME`, polygon edges are extracted as line segments with vertex deduplication. `RenderPiece()` skips triangle drawing and only draws lines. Cache invalidation tracks representation changes.

**WebGPU reference**: WebGPU handles this via `TopologyBindGroupInfos[TOPOLOGY_SOURCE_POLYGON_EDGES]` — the `CellToPrimitiveConverter` extracts polygon edges for wireframe mode.

### 2B. Edge Visibility on Surfaces ✅

**Status**: Implemented. When `representation == VTK_SURFACE && GetEdgeVisibility()`, separate edge geometry is built from polygon edges (with interior edge hiding for fan triangulation). Edge overlay is drawn after triangles with edge color from `vtkProperty::GetEdgeColor()` and coincident topology offset. Dedicated `EdgePipeline` with `fragment_edge_main` shader outputs flat edge color.

**WebGPU reference**: `ReplaceVertexShaderEdges()` (line 3226), `ReplaceFragmentShaderEdges()` (line 3666) — computes per-vertex edge distances and blends edge color in fragment shader.

### 2C. Triangle Index Buffers ✅

**Status**: Implemented. When `cellFlag == 0` (per-point coloring) and normals come from the data, vertices are deduplicated by point ID into a shared vertex array. An index buffer references shared vertices, reducing memory by ~50% for typical meshes and improving vertex cache hit rate. Falls back to non-indexed rendering when per-cell coloring is active (different colors per cell prevent deduplication) or when normals are computed per-face (different face normals per shared vertex).

---

## Phase 3: Line Rendering Variants

### 3A. Thick Lines (No Join) ✅

**Status**: Implemented. When `lineWidth > 1` and `lineJoin == NoJoin`, line segments are rendered as screen-space quads using `vertex_thick_line_main` / `fragment_thick_line_main`. The vertex shader reads both endpoints from the line index buffer, transforms to screen space, and expands perpendicular to the line direction by `lineWidth`. The fragment shader applies tube-like shading by modifying the normal based on distance from the centerline. `ThickLinePipeline` uses `MTLPrimitiveTypeTriangle` topology with instanced drawing (4 verts × N segments).

**Files**:
- `MetalShaders.metal` — new `vertex_thick_line_main` / `fragment_thick_line_main`
- `vtkMetalPolyDataMapper.mm` — new `ThickLinePipeline`, `ThickLineLineWidthBuffer`, `EnsureThickLinePipelineState()`
- `vtkMetalPolyDataMapper.h` — `EnsureThickLinePipelineState()` declaration

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

### 4A. Activate Clipping Plane Support ✅

**Status**: Implemented. `UpdateClipPlaneUniforms()` reads clip planes via `GetNumberOfClippingPlanes()` and `GetClippingPlaneInDataCoords()`, transforms them by the actor's model-to-world matrix, and packs up to 6 planes into `ClipPlaneUniforms` with `numClipPlanes` set to the actual count.

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

| Priority | Phase | Effort | Impact | Status |
|----------|-------|--------|--------|--------|
| 1 | 1A — Per-vertex color | Small | Critical | ✅ Done |
| 2 | 1B — Cell data coloring | Small | High | ✅ Done |
| 3 | 1C — Cull mode | Trivial | Medium | ✅ Done |
| 4 | 4A — Clipping planes | Small | Medium | ✅ Done |
| 5 | 2C — Triangle index buffers | Small | Medium | ✅ Done |
| 6 | 2A — Wireframe representation | Medium | High | ✅ Done |
| 7 | 2B — Edge visibility | Medium | High | ✅ Done |
| 8 | 3A — Thick lines | Medium | Medium | ✅ Done |
| 9 | 5A — Texture mapping | Medium | Medium | — UVs buffered but unused |
| 10 | 3B/3C — Round/miter joins | Large | Low | — niche line styles |
| 11 | 6A — GPU tessellation | Large | Medium | — moves work off CPU |
| 12 | 8A — MSAA | Medium | Medium | — anti-aliasing |
| 13 | 7A — 2D mapper | Medium | Low | — 2D overlay support |
| 14 | 7B — Batched mapper | Large | Low | — performance optimization |
| 15 | 8B — Depth peeling | Large | Low | — correct translucency |
| 16 | 7C — Composite delegator | Large | Low | — LOD support |
| 17 | 7D — Glyph3D mapper | Large | Low | — glyph instancing |
| 18 | 8C — Render bundles | Large | Low | — perf optimization |
| 19 | 8D — Vertex attribute mapping | Medium | Low | — custom attributes |

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
