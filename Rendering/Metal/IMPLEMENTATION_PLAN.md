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

### 3B. Round Cap + Round Join Lines ✅

**Status**: Implemented. When `lineWidth > 1` and `lineJoin == RoundCapRoundJoin`, line segments are rendered as a 36-vertex triangle strip template: 6 body quad vertices + 15 left semicircle (5 triangle fan segments × 3 verts) + 15 right semicircle (5 triangle fan segments × 3 verts). The third coordinate (`p_coord.z`) maps to interpolation along the line: `z=0` → p0 (left cap), `z=1` → p1 (right cap). Fragment shader applies tube-like shading based on distance from centerline. `RoundCapLinePipeline` uses `MTLPrimitiveTypeTriangleStrip` with instanced drawing (36 verts × N segments).

**Files**:
- `MetalShaders.metal` — `vertex_round_cap_line_main` / `fragment_round_cap_line_main`, `RoundCapLineVertexOut`
- `vtkMetalPolyDataMapper.mm` — `RoundCapLinePipeline`, `EnsureRoundCapLinePipelineState()`
- `vtkMetalPolyDataMapper.h` — `EnsureRoundCapLinePipelineState()` declaration

**WebGPU reference**: `GFX_PIPELINE_LINES_ROUND_CAP_ROUND_JOIN` — 36 verts per segment instance.

### 3C. Miter Join Lines ✅

**Status**: Implemented. When `lineWidth > 1` and `lineJoin == MiterJoin`, line segments use the same 4-vertex triangle strip quad as thick lines, but the vertex shader examines adjacent segments to compute miter offsets at shared endpoints. At each shared vertex, the miter direction is computed as the normalized sum of edge normals from adjacent segments. The quad corner is offset along the miter direction. Miter limit check: if the miter offset exceeds 2× line width, falls back to bevel (no extension). `MiterJoinLinePipeline` uses `MTLPrimitiveTypeTriangleStrip` with instanced drawing (4 verts × N segments). A `segmentCount` uniform buffer provides bounds for the next-segment adjacency check.

**Files**:
- `MetalShaders.metal` — `vertex_miter_join_line_main` / `fragment_miter_join_line_main`, `MiterJoinLineVertexOut`
- `vtkMetalPolyDataMapper.mm` — `MiterJoinLinePipeline`, `MiterJoinSegmentCountBuffer`, `EnsureMiterJoinLinePipelineState()`
- `vtkMetalPolyDataMapper.h` — `EnsureMiterJoinLinePipelineState()` declaration

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

### 5A. Complete UV Pipeline with Sampling ✅

**Status**: Implemented. TriangleUVBuffer (float2 per vertex) created from `pd->GetTCoords()` for all geometry types. Vertex shader reads UVs from `[[buffer(8)]]` and passes through `VertexOut.uv`. Fragment shader accepts `texture2d<float>` at `[[texture(0)]]` and `sampler` at `[[sampler(0)]]`. When actor has texture (`vtkActor::GetTexture()`), `MTLTexture` and `MTLSamplerState` are created from `vtkImageData`. Default 1x1 white texture used as fallback. Texture color multiplied with ambient/diffuse colors and opacity (modulate blending). Scene flags bit 9 indicates texture presence.

**Files**:
- `MetalShaders.metal` — `VertexOut.uv`, `vertex_main` UV buffer, `fragment_main` texture/sampler arguments and sampling
- `vtkMetalPolyDataMapper.mm` — `TriangleUVBuffer`, `ActorTexture`, `ActorSampler`, `UpdateActorTexture()`, texture binding in `RenderPiece()`
- `vtkMetalPolyDataMapper.h` — `UpdateActorTexture()` declaration

**Implementation**:
1. In `BuildGeometryBuffers()`, when `pd->GetTCoords()` exists, UV data is collected alongside positions at all vertex addition points (wireframe, indexed, non-indexed, line paths).
2. `TriangleUVBuffer` created from collected UV data, with zero-UV fallback when no UVs exist.
3. `UpdateActorTexture()` reads `vtkActor::GetTexture()`, converts `vtkImageData` to RGBA8 `MTLTexture`, creates `MTLSamplerState` with linear filtering and repeat addressing.
4. Fragment shader samples texture when scene flags bit 9 is set, multiplying sampled color with base ambient/diffuse colors and opacity.
5. Default 1x1 white texture and sampler created lazily as fallback when no actor texture is present.

**WebGPU reference**: `ReplaceFragmentShaderColors()` handles texture coordinate lookup and sampling, `ColorTextureHostResource` manages the GPU texture.

---

## Phase 6: GPU Tessellation (Compute Shader)

### 6A. Cell-to-Primitive Compute Pipeline ✅

**Status**: Implemented. Three new Metal compute kernels (`polygonToTriangle`, `polyLineToLine`, `polygonEdgesToLines`) replace CPU fan-triangulation for large meshes (>1000 points). When per-point coloring (`cellFlag == 0`) with data normals, `BuildGeometryBuffers()` builds per-point vertex arrays from all polydata points, constructs connectivity/offsets/primitiveCounts arrays, and dispatches compute kernels that produce triangle index buffers, line segment index buffers, and edge visibility arrays on the GPU. The GPU-produced edge array encodes which internal fan edges to hide (-1=all visible, 0/1/2=specific edge hidden), matching WebGPU's `polygon_to_triangle` shader. Wireframe mode uses `polygonEdgesToLines` to extract polygon boundary edges. CPU fallback retained for per-cell coloring, computed normals, and small meshes.

**Files**:
- `MetalShaders.metal` — new `polygonToTriangle`, `polyLineToLine`, `polygonEdgesToLines` kernels with `TessParams` uniform struct
- `vtkMetalPolyDataMapper.mm` — GPU tessellation path in `BuildGeometryBuffers()`, new pipeline states (`PolygonToTrianglePipeline`, `PolyLineToLinePipeline`, `PolygonEdgesToLinesPipeline`), output buffers (`TessOutputConnectivityBuffer`, `TessEdgeArrayBuffer`, `TessParamsBuffer`)

**WebGPU reference**: `vtkWebGPUCellToPrimitiveConverter` — full compute-based tessellation with edge array output. Metal kernels mirror the WGSL `polygon_to_triangle`, `poly_line_to_line`, and `polygon_edges_to_lines` shaders.

---

## Phase 7: Additional Mappers & Infrastructure

### 7A. 2D Mapper ✅

**Status**: Implemented. `vtkMetalPolyDataMapper2D` inherits from `vtkPolyDataMapper2D` and overrides `RenderOverlay()`. Uses orthographic projection to transform polydata points from viewport pixel coordinates to Metal NDC. Three pipeline states (triangle, line, point) with `vertex_2d_main` / `fragment_2d_main` shaders. Supports `TransformCoordinate` for coordinate system conversion. Geometry is rebuilt when input MTime changes. Pipeline states support MSAA sample count.

**Files**:
- `vtkMetalPolyDataMapper2D.h` — class declaration
- `vtkMetalPolyDataMapper2D.mm` — `RenderOverlay()`, orthographic WCVC matrix, fan triangulation
- `MetalShaders.metal` — `vertex_2d_main`, `fragment_2d_main`, `Mapper2DState` struct
- `CMakeLists.txt` — added to classes list

**WebGPU reference**: `vtkWebGPUPolyDataMapper2D` + `Private/vtkWebGPUPolyDataMapper2DInternals`.

### 7B. Batched Mapper ✅

**Status**: Implemented. `vtkMetalBatchedPolyDataMapper` accumulates multiple actors' geometry into shared vertex/index buffers. Uses `CompositeDataProperties` uniform buffer with 256-byte alignment per entry to store per-actor properties (opacity, ambient/diffuse colors, cell ID offsets, pickability, composite ID). `BatchPropertiesBuffer` stores all mesh properties in a single GPU buffer. `AddBatchElement()`/`GetBatchElement()`/`ClearBatchElements()` manage the batch. `RenderPiece()` iterates over batch elements and calls the parent class for each visible mesh. `GetMTime()` returns max of parent and batch MTime.

**Files**:
- `vtkMetalBatchedPolyDataMapper.h` — class declaration with `CompositeDataProperties` struct, `BatchElement` typedef
- `vtkMetalBatchedPolyDataMapper.mm` — `AddBatchElement()`, `BuildBatchedGeometryBuffers()`, `UpdateBatchPropertiesBuffer()`, `RenderPiece()`
- `CMakeLists.txt` — added to classes list

**WebGPU reference**: `vtkWebGPUBatchedPolyDataMapper` — same architecture with storage buffer instead of uniform buffer.

### 7C. Composite Mapper Delegator ✅

**Status**: Implemented. `vtkMetalCompositePolyDataMapperDelegator` inherits from `vtkCompositePolyDataMapperDelegator` and delegates to `vtkMetalBatchedPolyDataMapper`. Trampolines all virtual methods (`Insert()`, `Get()`, `Clear()`, `SetParent()`, `GetRenderedList()`, etc.) to the Metal batched mapper. `CreateOverrideAttributes()` sets `RenderingBackend` to `"Metal"`.

**Files**:
- `vtkMetalCompositePolyDataMapperDelegator.h` — class declaration
- `vtkMetalCompositePolyDataMapperDelegator.mm` — trampoline implementations
- `CMakeLists.txt` — added to classes list

**WebGPU reference**: `vtkWebGPUCompositePolyDataMapperDelegator` — same pattern.

### 7D. Glyph3D Mapper ✅

**Status**: Implemented. `vtkMetalGlyph3DMapper` inherits from `vtkGlyph3DMapper` and overrides `Render()` to perform Metal instanced rendering. Source geometry (triangles, lines, points) is extracted from the glyph source polydata into per-vertex position/normal buffers. Per-instance glyph attributes (4×4 transform, 3×3 normal transform, RGBA color, pick ID) are computed on the CPU from input point data, orientation arrays, scale arrays, and selection arrays, then uploaded to per-instance Metal buffers with `stepFunctionPerInstance`. Three dedicated pipeline states (triangle, line, point) use `vertex_glyph_main`/`fragment_glyph_main` shaders that read source geometry from buffer slots 0–1 and instance data from slots 2–5. The glyph transform is composed in the vertex shader: `modelMatrix * glyphTransform * position`. Source geometry is cached and only rebuilt when the source polydata MTime changes; instance data is cached and only rebuilt when the input dataset MTime changes. Supports all orientation modes (DIRECTION, ROTATION, QUATERNION), scale modes (BY_MAGNITUDE, BY_COMPONENTS, NO_DATA_SCALING), clamping, masking, and selection IDs. MSAA sample count changes invalidate pipeline states.

**Files**:
- `vtkMetalGlyph3DMapper.h` — class declaration
- `vtkMetalGlyph3DMapper.mm` — `Render()`, source geometry extraction, instance buffer management, pipeline creation
- `MetalShaders.metal` — `vertex_glyph_main`/`fragment_glyph_main`, `vertex_glyph_line_main`/`fragment_glyph_line_main`, `vertex_glyph_point_main`/`fragment_glyph_point_main`, `GlyphVertexOut`, `GlyphLineVertexOut`, `GlyphPointVertexOut` structs
- `CMakeLists.txt` — added to classes list

**WebGPU reference**: `vtkWebGPUGlyph3DMapper` — same architecture with shader substitution; Metal uses dedicated shader functions instead.

---

## Phase 8: Advanced Features

### 8A. MSAA ✅

**Status**: Implemented. `vtkMetalRenderWindow` creates `MTLTextureType2DMultisample` color and depth textures via `CreateMultisampleAttachments()` when `MultiSamples > 1`. `GetEffectiveSampleCount()` returns the active sample count. `vtkMetalRenderer::DeviceRender()` uses multisample textures as render targets with `MTLStoreActionMultisampleResolve` to resolve to the drawable. IDs attachment (RGBA32Uint) is skipped when MSAA is active since RGBA32Uint doesn't support multisampling. All 8 pipeline types in `vtkMetalPolyDataMapper` set `sampleCount` on their `MTLRenderPipelineDescriptor`. Pipeline states are invalidated in `RenderPiece()` when sample count changes.

**Files**:
- `vtkMetalRenderWindow.h` — `MultisampleColorTexture`, `MultisampleDepthTexture`, `GetEffectiveSampleCount()`, `CreateMultisampleAttachments()`, `DestroyMultisampleAttachments()`
- `vtkMetalRenderWindow.mm` — MSAA texture creation/destruction, `Render()` recreates MSAA textures on size change
- `vtkMetalRenderer.mm` — render pass uses MSAA textures with resolve, skips IDs when MSAA active
- `vtkMetalPolyDataMapper.mm` — `sampleCount` on all pipeline descriptors, `CachedSampleCount` invalidation in `RenderPiece()`

### 8B. Depth Peeling / Correct Translucency ✅

**Status**: Implemented. Dual depth peeling for order-independent transparency. `vtkMetalDepthPeeler` manages the multi-pass algorithm with 6 ping-pong textures (FrontPeelA/B, BackPeelTemp, BackAccum, DepthPeelA/B). Algorithm:
1. Opaque pass renders with depth write=Yes.
2. Init peel pass renders translucent with MAX blending on RG32Float depth range, depth test=Less against opaque depth.
3. Peel loop (up to 8 iterations): each pass renders translucent with 3 color outputs (backTemp, frontDest, depthDest). Four-zone depth comparison: outside (pass through), inside (mark for future peel), on front (under-blend into front accumulation), on back (premultiply alpha into backTemp). BackTemp blended into BackAccum via premultiplied over-blend. Ping-pong front/depth buffers swapped each iteration.
4. Final composite: fullscreen pass blends front accumulation (alpha stored as 1-alpha) and back accumulation onto drawable with premultiplied over-blend.

**Files**:
- `vtkMetalDepthPeeler.h` — class declaration with texture/pipeline management
- `vtkMetalDepthPeeler.mm` — multi-pass orchestration, texture creation, fullscreen pipelines
- `MetalShaders.metal` — `fragment_peel_init`, `fragment_peel`, `fragment_peel_alpha_blend`, `fragment_peel_composite`, `fragment_peel_back_blend`, `vertex_fullscreen_main`
- `vtkMetalRenderer.mm` — split DeviceRender into opaque pass + depth peeling or fallback alpha blending
- `vtkMetalRenderer.h` — `DepthPeeler` member, `HasTranslucentPolygonalGeometry()`
- `vtkMetalRenderWindow.h` — peeling state fields (`DepthPeelingMode`, `PeelFrontTexture`, `PeelDepthTexture`, `PeelIndex`)
- `vtkMetalPolyDataMapper.mm` — `EnsurePeelPipelineStates()`, peeling pipeline selection in `RenderPiece()`, peeling texture/uniform binding
- `vtkMetalPolyDataMapper.h` — `EnsurePeelPipelineStates()` declaration
- `CMakeLists.txt` — added `vtkMetalDepthPeeler`

**Key design decisions**:
- No stencil buffer needed — min-max depth buffer (RG32Float) with MAX blending handles all tracking.
- Front accumulation alpha stored as (1-alpha) so MAX blending correctly picks nearest fragment.
- No occlusion queries — fixed maximum peel count (default 8).
- Mapper creates separate peeling pipeline states (fragment_peel_init for init pass, fragment_peel for peel passes) with 3 color attachments and appropriate MAX blend modes.
- Depth peeling enabled via `vtkRenderer::SetUseDepthPeeling(1)` (base class API).

**WebGPU reference**: Not yet implemented in WebGPU backend. Algorithm based on `vtkDualDepthPeelingPass` (OpenGL2).

### 8C. Render Bundles ✅

**Status**: Implemented. Render bundle caching pre-records encoder commands (pipeline states, buffer bindings, draw calls) into a `RenderBundle` on first frame, then replays them directly on subsequent frames when geometry hasn't changed. This eliminates CPU encoding overhead for static scenes.

**Mechanism**:
1. `RenderBundleDrawCommand` stores individual encoder operations (set pipeline, bind buffer, draw, etc.) via a `std::variant`-based tagged union.
2. `RenderBundle` holds a `std::vector<RenderBundleDrawCommand>` of the full command sequence for a mapper's draw calls.
3. `RebuildRenderBundle()` records all geometry-related encoder commands (triangles, lines, edge overlay, vertex visibility, points) into the bundle. It mirrors the exact encoder command sequence that was previously inline in `RenderPiece()`.
4. `ReplayRenderBundle()` iterates through cached commands and issues them on the current `MTLRenderCommandEncoder`. Since Metal buffer objects persist across frames and uniform buffers are updated in-place, replaying the same buffer bindings reads the latest content automatically.
5. `RenderPiece()` checks bundle validity by comparing current geometry state (MTime, representation, edge visibility, line width, MSAA sample count, peel mode) against values at bundle creation. If valid, only `ReplayRenderBundle()` is called. If invalid, `RebuildRenderBundle()` + `ReplayRenderBundle()` are called.

**Invalidation triggers**: Geometry MTime change, representation change, edge visibility toggle, line width change, MSAA sample count change, depth peeling mode change, or `ReleaseGraphicsResources()`.

**Files**:
- `vtkMetalPolyDataMapper.mm` — `RenderBundleDrawCommand`, `RenderBundle` structs, `ReplayRenderBundle()`, `RebuildRenderBundle()`, bundle validity tracking fields, modified `RenderPiece()` to use bundle

### 8D. Vertex Attribute Mapping ✅

**Status**: Implemented. `MapDataArrayToVertexAttribute()` stores mappings in an `ExtraAttributes` map (attribute name → data array name, field association, component number). `RemoveVertexAttributeMapping()` and `RemoveAllVertexAttributeMappings()` clear entries and invalidate the render bundle. In `BuildGeometryBuffers()`, each mapped data array is looked up from point/cell data, converted to float, and uploaded as a per-point `MTLBuffer`. Extra attribute buffers are bound at buffer indices 16+ in `RebuildRenderBundle()`, making them accessible to custom Metal shaders via `[[buffer(N)]]`.

**Files**:
- `vtkMetalPolyDataMapper.h` — `ExtraAttributeValue` struct, `ExtraAttributes` map, `MapDataArrayToVertexAttribute()`, `RemoveVertexAttributeMapping()`, `RemoveAllVertexAttributeMappings()` declarations
- `vtkMetalPolyDataMapper.mm` — method implementations, extra attribute buffer creation in `BuildGeometryBuffers()`, buffer binding in `RebuildRenderBundle()` at indices 16+, cleanup in `ReleaseBuffers()`

**WebGPU reference**: `MapDataArrayToVertexAttribute()` stores mappings, `UploadAttributeToGPUBuffer()` handles upload. WebGPU implementation is currently a stub.

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
| 9 | 5A — Texture mapping | Medium | Medium | ✅ Done |
| 10 | 3B/3C — Round/miter joins | Large | Low | ✅ Done |
| 11 | 6A — GPU tessellation | Large | Medium | ✅ Done |
| 12 | 8A — MSAA | Medium | Medium | ✅ Done |
| 13 | 7A — 2D mapper | Medium | Low | ✅ Done |
| 14 | 7B — Batched mapper | Large | Low | ✅ Done |
| 15 | 7C — Composite delegator | Large | Low | ✅ Done |
| 16 | 8B — Depth peeling | Large | Low | ✅ Done |
| 17 | 7D — Glyph3D mapper | Large | Low | ✅ Done |
| 18 | 8C — Render bundles | Large | Low | ✅ Done |
| 19 | 8D — Vertex attribute mapping | Medium | Low | ✅ Done |

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
