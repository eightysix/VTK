Below is a detailed implementation guide for the **Priority 1 correctness blockers** identified earlier.

This guide intentionally **does not modify the volume ray-cast mapper shaders**. Any shader changes described here are limited to the polydata/line/point/depth-peeling portions.

---

# Priority 1 Implementation Guide

## Scope

Priority 1 items:

1. Fix `VTK_POINTS` representation rendering triangles/lines incorrectly.
2. Fix sticky scene-uniform flags.
3. Fix prop ID off-by-one / inconsistent ID convention.
4. Verify and fix line-width scaling.
5. Verify and fix round-cap line topology.
6. Make the batched mapper actually apply per-batch visual overrides, or remove the unused batch-property buffer.
7. Fix stale flat-index / batch-element mappings.
8. Guard depth peeling against MSAA incompatibility.
9. Ensure depth-peel textures are always bound when required.

Recommended implementation order:

```text
1. Representation gating
2. Sticky scene flags
3. Prop ID convention
4. Line width scaling
5. Round-cap line safety/fix
6. Batch mapper overrides
7. Batch element lifetime / flat-index staleness
8. Depth peeling MSAA guard
9. Depth-peel texture binding safety
```

---

# 0. Pre-work: Enable Validation and Capture

Before making changes, enable Metal validation and configure a few debug helpers.

## 0.1 Enable Metal API validation

In Xcode:

```text
Product → Scheme → Edit Scheme → Run → Options → Metal API Validation: On
```

Also enable:

```text
Metal Shader Validation
GPU Frame Capture
```

## 0.2 Add a small debug assertion helper

In an internal Metal utility header, add something like:

```cpp
#if defined(VTK_METAL_DEBUG)
#define VTK_METAL_ASSERT_BOUND(obj) \
  do { if (!(obj)) { vtkErrorMacro("Metal object not bound: " #obj); } } while (0)
#else
#define VTK_METAL_ASSERT_BOUND(obj)
#endif
```

This will help catch missing buffers/textures during bundle recording.

---

# 1. Fix `VTK_POINTS` Representation Rendering Triangles/Lines

## Problem

`RebuildRenderBundle` currently uses:

```cpp
bool skipTriangles = (representation == VTK_WIREFRAME);
```

This means:

- `VTK_SURFACE` draws triangles: correct.
- `VTK_WIREFRAME` skips triangles: correct.
- `VTK_POINTS` also draws triangles: incorrect.

Line drawing is also not sufficiently gated by representation. If the polydata contains lines, they may draw even when representation is `VTK_POINTS`.

## Desired behavior

For standard VTK representation semantics:

```text
VTK_SURFACE:
  - draw triangles
  - draw explicit lines
  - draw edge overlay if EdgeVisibility is on

VTK_WIREFRAME:
  - draw wireframe lines from polygons
  - draw explicit lines
  - do not draw filled triangles
  - do not draw edge overlay

VTK_POINTS:
  - draw points only
  - do not draw triangles
  - do not draw lines
  - do not draw edge overlay
```

Vertex visibility dots are separate and may still draw for surface/wireframe if `VertexVisibility` is enabled.

---

## 1.1 Update representation predicates in `RebuildRenderBundle`

Replace the current `skipTriangles` logic with explicit predicates.

### Before

```cpp
int representation = act->GetProperty()->GetRepresentation();
float lineWidth = static_cast<float>(act->GetProperty()->GetLineWidth());
bool skipTriangles = (representation == VTK_WIREFRAME);
```

### After

```cpp
int representation = act->GetProperty()->GetRepresentation();
float lineWidth = static_cast<float>(act->GetProperty()->GetLineWidth());

const bool isSurface = (representation == VTK_SURFACE);
const bool isWireframe = (representation == VTK_WIREFRAME);
const bool isPoints = (representation == VTK_POINTS);

const bool drawTriangles =
    isSurface &&
    this->Internals->HasTriangles &&
    this->Internals->TrianglePipeline;

const bool drawLines =
    (isSurface || isWireframe) &&
    this->Internals->HasLines &&
    this->Internals->LineIndexBuffer;

const bool drawEdgeOverlay =
    isSurface &&
    act->GetProperty()->GetEdgeVisibility() &&
    this->Internals->HasEdgeOverlay &&
    this->Internals->EdgePipeline &&
    this->Internals->EdgeIndexBuffer;

const bool drawPointRepresentation =
    isPoints &&
    this->Internals->PointVertexCount > 0 &&
    this->Internals->PointPositionBuffer;

const bool drawVertexVisibilityDots =
    !isPoints &&
    act->GetProperty()->GetVertexVisibility() &&
    this->Internals->PointVertexCount > 0 &&
    this->Internals->PointPositionBuffer;
```

Then use these booleans everywhere.

---

## 1.2 Use the predicates for draw recording

### Triangle block

Replace:

```cpp
if (!skipTriangles && this->Internals->HasTriangles && this->Internals->TrianglePipeline)
```

with:

```cpp
if (drawTriangles)
```

### Line block

Replace:

```cpp
if (this->Internals->HasLines && this->Internals->LineIndexBuffer)
```

with:

```cpp
if (drawLines)
```

### Edge overlay block

Replace:

```cpp
if (representation == VTK_SURFACE && act->GetProperty()->GetEdgeVisibility() &&
    this->Internals->HasEdgeOverlay && this->Internals->EdgePipeline &&
    this->Internals->EdgeIndexBuffer)
```

with:

```cpp
if (drawEdgeOverlay)
```

### Vertex visibility block

Replace:

```cpp
if (representation != VTK_POINTS && act->GetProperty()->GetVertexVisibility() &&
    this->Internals->PointVertexCount > 0 && this->Internals->PointPositionBuffer)
```

with:

```cpp
if (drawVertexVisibilityDots)
```

### Point representation block

Replace:

```cpp
if (representation == VTK_POINTS && this->Internals->PointVertexCount > 0 &&
    this->Internals->PointPositionBuffer)
```

with:

```cpp
if (drawPointRepresentation)
```

---

## 1.3 Prevent unnecessary pipeline creation

In `RenderPiece`, pipeline creation should also respect representation.

### Before

```cpp
bool needSurfacePipelines =
    (representation != VTK_POINTS);
```

### After

```cpp
bool needTrianglePipeline =
    (representation == VTK_SURFACE) &&
    this->Internals->HasTriangles;

bool needLinePipeline =
    (representation == VTK_SURFACE || representation == VTK_WIREFRAME) &&
    this->Internals->HasLines;

bool needSurfacePipelines =
    needTrianglePipeline || needLinePipeline;
```

Then:

```cpp
if (needSurfacePipelines)
{
    this->EnsurePipelineStates((void*)device);
}
```

Edge pipeline:

```cpp
if (representation == VTK_SURFACE &&
    act->GetProperty()->GetEdgeVisibility() &&
    this->Internals->HasEdgeOverlay)
{
    this->EnsureEdgePipelineState((void*)device);
}
```

Point pipelines:

```cpp
bool needPointPipelines =
    (representation == VTK_POINTS) ||
    (act->GetProperty()->GetVertexVisibility() &&
     this->Internals->PointVertexCount > 0);
```

---

## 1.4 Optional but recommended: skip triangle/line generation for `VTK_POINTS`

In `BuildGeometryBuffers`, avoid building triangle and line geometry when representation is points.

Near the top after determining representation:

```cpp
int representation = actor ? actor->GetProperty()->GetRepresentation() : VTK_SURFACE;
bool edgeVisibility = actor ? actor->GetProperty()->GetEdgeVisibility() : false;

const bool buildSurfaceGeometry = (representation == VTK_SURFACE);
const bool buildWireframeGeometry = (representation == VTK_WIREFRAME);
const bool buildPointGeometryOnly = (representation == VTK_POINTS);
```

Then guard polygon triangulation:

```cpp
if (!buildPointGeometryOnly && polys && polys->GetNumberOfCells() > 0)
{
    ...
}
```

Guard explicit line processing:

```cpp
if (!buildPointGeometryOnly && lines && lines->GetNumberOfCells() > 0)
{
    ...
}
```

Guard edge overlay:

```cpp
if (buildSurfaceGeometry && edgeVisibility)
{
    ...
}
```

The point buffers at the end should still always be built because they are needed for both `VTK_POINTS` and vertex visibility dots.

---

## 1.5 Validation

Test cases:

1. Sphere with representation `VTK_SURFACE`:
   - should render filled surface.

2. Sphere with representation `VTK_WIREFRAME`:
   - should render wireframe only.

3. Sphere with representation `VTK_POINTS`:
   - should render points only.
   - no triangles.
   - no lines.
   - no edge overlay.

4. Polydata containing triangles plus explicit lines:
   - surface: triangles + lines.
   - wireframe: polygon wireframe + explicit lines.
   - points: points only.

5. Surface with edge visibility:
   - surface: triangles + edges.
   - wireframe: no separate edge overlay.
   - points: no edges.

---

# 2. Fix Sticky Scene-Uniform Flags

## Problem

The scene-uniform flags are currently updated with bitwise OR:

```cpp
*reinterpret_cast<uint32_t*>(buf + 256) |= actorFlags;
```

and later:

```cpp
*reinterpret_cast<uint32_t*>(buf + 256) |= (1u << 9);
```

This means flags can remain set after they should be cleared.

Examples:

- Turn off vertex visibility: bit 3 may remain set.
- Remove actor texture: bit 9 may remain set.
- Disable surface colors: bit 8 may remain set.
- Change point shape: old bit may remain set.

## Required behavior

Dynamic per-actor flags must be cleared and rewritten every frame.

---

## 2.1 Define explicit flag constants

Add constants near the top of `vtkMetalPolyDataMapper.mm`:

```cpp
namespace
{
constexpr uint32_t VTK_METAL_SCENE_FLAG_PARALLEL_PROJECTION = 1u << 0;
constexpr uint32_t VTK_METAL_SCENE_FLAG_VERTEX_VISIBILITY   = 1u << 3;
constexpr uint32_t VTK_METAL_SCENE_FLAG_SPHERE_POINTS       = 1u << 5;
constexpr uint32_t VTK_METAL_SCENE_FLAG_POINT_SHAPE         = 1u << 7;
constexpr uint32_t VTK_METAL_SCENE_FLAG_HAS_SURFACE_COLORS  = 1u << 8;
constexpr uint32_t VTK_METAL_SCENE_FLAG_HAS_ACTOR_TEXTURE   = 1u << 9;

constexpr uint32_t VTK_METAL_DYNAMIC_ACTOR_FLAG_MASK =
    VTK_METAL_SCENE_FLAG_VERTEX_VISIBILITY |
    VTK_METAL_SCENE_FLAG_SPHERE_POINTS |
    VTK_METAL_SCENE_FLAG_POINT_SHAPE |
    VTK_METAL_SCENE_FLAG_HAS_SURFACE_COLORS |
    VTK_METAL_SCENE_FLAG_HAS_ACTOR_TEXTURE;
}
```

If `Point2DShape` eventually needs more than one bit, expand the mask accordingly.

---

## 2.2 Replace OR-only flag updates

In `RenderPiece`, after copying the camera scene transforms:

### Before

```cpp
uint32_t actorFlags = 0;
actorFlags |= (prop->GetVertexVisibility() ? 1u : 0u) << 3;
actorFlags |= (prop->GetRenderPointsAsSpheres() ? 1u : 0u) << 5;
actorFlags |= (static_cast<uint32_t>(prop->GetPoint2DShape())) << 7;
actorFlags |= (this->Internals->HasSurfaceColors ? 1u : 0u) << 8;
*reinterpret_cast<uint32_t*>(buf + 256) |= actorFlags;
```

### After

```cpp
uint32_t flags = *reinterpret_cast<uint32_t*>(buf + 256);

flags &= ~VTK_METAL_DYNAMIC_ACTOR_FLAG_MASK;

uint32_t actorFlags = 0;

if (prop->GetVertexVisibility())
{
    actorFlags |= VTK_METAL_SCENE_FLAG_VERTEX_VISIBILITY;
}

if (prop->GetRenderPointsAsSpheres())
{
    actorFlags |= VTK_METAL_SCENE_FLAG_SPHERE_POINTS;
}

// Currently shader only checks bit 7 for round vs square.
// If more point shapes are added, this must be widened.
if (static_cast<uint32_t>(prop->GetPoint2DShape()) != 0u)
{
    actorFlags |= VTK_METAL_SCENE_FLAG_POINT_SHAPE;
}

if (this->Internals->HasSurfaceColors)
{
    actorFlags |= VTK_METAL_SCENE_FLAG_HAS_SURFACE_COLORS;
}

flags |= actorFlags;

*reinterpret_cast<uint32_t*>(buf + 256) = flags;
```

Then remove the later standalone texture OR:

### Remove

```cpp
if (this->Internals->ActorTexture)
{
    char* buf = static_cast<char*>([this->Internals->SceneUniformBuffer contents]);
    *reinterpret_cast<uint32_t*>(buf + 256) |= (1u << 9);
}
```

### Replace with integrated update

Immediately after the flag block above, or before writing flags back:

```cpp
if (this->Internals->ActorTexture)
{
    flags |= VTK_METAL_SCENE_FLAG_HAS_ACTOR_TEXTURE;
}
else
{
    flags &= ~VTK_METAL_SCENE_FLAG_HAS_ACTOR_TEXTURE;
}

*reinterpret_cast<uint32_t*>(buf + 256) = flags;
```

Final structure:

```cpp
char* buf = static_cast<char*>([this->Internals->SceneUniformBuffer contents]);

uint32_t flags = *reinterpret_cast<uint32_t*>(buf + 256);
flags &= ~VTK_METAL_DYNAMIC_ACTOR_FLAG_MASK;

uint32_t actorFlags = 0;

if (prop->GetVertexVisibility())
{
    actorFlags |= VTK_METAL_SCENE_FLAG_VERTEX_VISIBILITY;
}

if (prop->GetRenderPointsAsSpheres())
{
    actorFlags |= VTK_METAL_SCENE_FLAG_SPHERE_POINTS;
}

if (static_cast<uint32_t>(prop->GetPoint2DShape()) != 0u)
{
    actorFlags |= VTK_METAL_SCENE_FLAG_POINT_SHAPE;
}

if (this->Internals->HasSurfaceColors)
{
    actorFlags |= VTK_METAL_SCENE_FLAG_HAS_SURFACE_COLORS;
}

if (this->Internals->ActorTexture)
{
    actorFlags |= VTK_METAL_SCENE_FLAG_HAS_ACTOR_TEXTURE;
}

flags |= actorFlags;

*reinterpret_cast<uint32_t*>(buf + 256) = flags;
```

---

## 2.3 Validation

Test cases:

1. Actor with texture, then remove texture:
   - texture sampling must stop.

2. Turn vertex visibility on/off:
   - vertex dots must appear/disappear.

3. Switch `RenderPointsAsSpheres` on/off:
   - point shading must update immediately.

4. Switch point shape between round/square:
   - no stale shape bits.

5. Toggle scalar coloring:
   - surface color branch must update correctly.

---

# 3. Fix Prop ID Off-by-One / Inconsistent Convention

## Problem

CPU side currently creates prop IDs starting at `1`:

```cpp
static uint32_t nextId = 1;
```

Shader side then adds one:

```metal
out.propId = propId + 1u;
```

So the first actor produces prop ID `2`.

Cell IDs are already written as 1-based on the CPU:

```cpp
triangleVertexCellIds.push_back(static_cast<uint32_t>(polyCellIdx) + 1u);
```

and shader does not add one:

```metal
out.cellId = cellIds[vertex_id];
```

This inconsistency is likely to break picking.

---

## Recommended convention

Use this convention:

```text
CPU buffer stores 0-based prop IDs.
Shader outputs propId + 1.
Background remains 0.
```

This requires the smallest change and avoids modifying shaders.

Cell IDs can remain 1-based on CPU for now, but document the convention clearly.

---

## 3.1 Change CPU prop ID allocation to zero-based

In `vtkMetalPolyDataMapper::GetOrCreatePropId`:

### Before

```cpp
static uint32_t nextId = 1;
```

### After

```cpp
// Buffer stores zero-based prop IDs.
// Shaders output propId + 1, so the first actor becomes 1.
static uint32_t nextId = 0;
```

No shader change is required for this convention.

---

## 3.2 Document the ID convention

Add comments near `SetPropId` and `GetOrCreatePropId`:

```cpp
// Picking ID convention:
// - GPU prop ID buffer stores zero-based IDs.
// - vertex shaders output propId + 1.
// - 0 in the picking buffer means background/no prop.
//
// Cell IDs are currently written as 1-based on the CPU and passed through
// unchanged by the shader. This should eventually be unified with prop IDs.
```

---

## 3.3 Optional: unify cell and prop ID conventions later

Longer term, prefer one convention everywhere:

Option A:

```text
All CPU ID buffers are zero-based.
All shaders output id + 1.
```

Option B:

```text
All CPU ID buffers are one-based.
Shaders output id unchanged.
```

Option B is cleaner for supporting unpickable objects with ID `0`, but requires shader edits. Since Priority 1 is correctness stabilization, use Option A now.

---

## 3.4 Validation

Test cases:

1. Render one actor and pick it:
   - expected prop ID should be `1`, not `2`.

2. Render two actors:
   - expected prop IDs should be `1` and `2`.

3. Pick background:
   - expected prop ID `0`.

4. Pick cells:
   - verify cell IDs are still nonzero where expected.

---

# 4. Verify and Fix Line-Width Scaling

## Problem

The thick-line shader uses `lineWidth` as a half-width-like offset:

```metal
float w = max(lineWidth, 1.0);
float2 adjusted_p0 = p0_screen + p_coord.x * x_basis + p_coord.y * y_basis * w;
```

Since `p_coord.y` ranges from `-1` to `1`, the total width is approximately `2 * lineWidth`.

VTK/OpenGL conventions usually treat `lineWidth` as the full width in pixels.

---

## 4.1 Fix `vertex_thick_line_main`

Use an explicit half-width and generate a simple quad.

### Replace the thick-line vertex transformation with:

```metal
float halfWidth = max(lineWidth, 1.0) * 0.5;

// p_coord.x: -1 = start, +1 = end
// p_coord.y: -1 = left, +1 = right
float t = (p_coord.x + 1.0) * 0.5;
float side = p_coord.y;

float2 center = mix(p0_screen, p1_screen, t);
float2 p = center + side * y_basis * halfWidth;

float4 p_DC = mix(p0_DC, p1_DC, t);
```

Then:

```metal
out.position = float4(p_DC.w * ((2.0 * p) / resolution - 1.0), p_DC.z, p_DC.w);
```

Also:

```metal
out.dist_to_centerline = side;
```

### Full replacement sketch

Inside `vertex_thick_line_main`:

```metal
const float2 tri_verts[4] = {
    float2(-1, -1),
    float2( 1, -1),
    float2(-1,  1),
    float2( 1,  1)
};

float2 p_coord = tri_verts[vertex_id];

uint p0_idx = lineIndices[instance_id * 2];
uint p1_idx = lineIndices[instance_id * 2 + 1];

float3 p0_MC = positions[p0_idx];
float3 p1_MC = positions[p1_idx];

float4 p0_DC = scene.projectionMatrix * scene.viewMatrix * scene.modelMatrix * float4(p0_MC, 1.0);
float4 p1_DC = scene.projectionMatrix * scene.viewMatrix * scene.modelMatrix * float4(p1_MC, 1.0);

float2 resolution = scene.viewport.zw;

float2 p0_screen = resolution * (0.5 * p0_DC.xy / p0_DC.w + 0.5);
float2 p1_screen = resolution * (0.5 * p1_DC.xy / p1_DC.w + 0.5);

float2 delta = p1_screen - p0_screen;
float segLen = length(delta);

float2 x_basis = segLen < 0.001 ? float2(1.0, 0.0) : (delta / segLen);
float2 y_basis = float2(-x_basis.y, x_basis.x);

float halfWidth = max(lineWidth, 1.0) * 0.5;

float t = (p_coord.x + 1.0) * 0.5;
float side = p_coord.y;

float2 center = mix(p0_screen, p1_screen, t);
float2 p = center + side * y_basis * halfWidth;

float4 p_DC = mix(p0_DC, p1_DC, t);

ThickLineVertexOut out;

out.position = float4(p_DC.w * ((2.0 * p) / resolution - 1.0), p_DC.z, p_DC.w);

out.viewPos =
    (scene.viewMatrix * scene.modelMatrix *
     float4(mix(p0_MC, p1_MC, t), 1.0)).xyz;

out.viewNormal = scene.normalMatrix * float3(0.0, 0.0, 1.0);
out.vertexColor = vertexColors[p0_idx];
out.dist_to_centerline = side;
out.cellId = cellIds[instance_id];
out.propId = propId + 1u;

return out;
```

This produces butt-cap quads with exact full width equal to `lineWidth`.

---

## 4.2 Fix miter-join shader scaling

In `vertex_miter_join_line_main`, replace:

```metal
float w = max(lineWidth, 1.0);
```

with:

```metal
float halfWidth = max(lineWidth, 1.0) * 0.5;
```

Then replace all uses of `w` with `halfWidth`.

In particular:

```metal
float2 offset = p_coord.x * x_basis + p_coord.y * y_basis * halfWidth;
```

and miter offset:

```metal
float miterOffset = halfWidth / dot(miter, float2(-x_basis.y, x_basis.x));
```

Do this in both the start-join and end-join blocks.

---

## 4.3 Fragment normal adjustment

The current normal pseudo-lighting uses:

```metal
N.z = 1.0 - 2.0 * abs(in.dist_to_centerline);
```

With `dist_to_centerline` in `[-1, 1]`, this goes negative near the edges.

For better stability, use:

```metal
float d = saturate(abs(in.dist_to_centerline));
N.z = sqrt(max(0.0, 1.0 - d * d));
N = normalize(N);
```

Apply this in:

- `fragment_thick_line_main`
- `fragment_round_cap_line_main`
- `fragment_miter_join_line_main`

This is optional for correctness but improves lighting consistency.

---

## 4.4 Validation

Test cases:

1. Set line width to `1`:
   - should render approximately 1 pixel.

2. Set line width to `5`:
   - should measure approximately 5 pixels wide, not 10.

3. Compare against OpenGL backend for:
   - width 1
   - width 2
   - width 5
   - width 10

4. Test diagonal lines:
   - width should remain visually constant.

5. Test thick lines with vertex colors and scalar colors.

---

# 5. Verify and Fix Round-Cap Line Topology

## Problem

The round-cap pipeline records:

```cpp
recordDraw(MTLPrimitiveTypeTriangleStrip, 0, 36, this->Internals->RoundCapLineSegmentCount);
```

But the vertex shader appears to generate vertices as independent triangles or triangle-list-style groups.

If the vertex generation was designed for a triangle list, drawing as a triangle strip will produce broken geometry.

---

## 5.1 Short-term safety fallback

Because round-cap geometry is easy to get wrong, the safest immediate fix is to disable round-cap lines until the shader is verified or rewritten.

In both `RenderPiece` and `RebuildRenderBundle`, force:

```cpp
bool useRoundCapLines = false;
```

or remove the round-cap branch temporarily.

Example:

```cpp
if (lineWidth > 1.0f)
{
    // Temporarily disabled until round-cap topology is verified.
    // if (lineJoinType == vtkProperty::LineJoinType::RoundCapRoundJoin && ...)
    // {
    //     useRoundCapLines = true;
    // }
    if (lineJoinType == vtkProperty::LineJoinType::MiterJoin &&
        this->Internals->MiterJoinLineSegmentCount > 0 &&
        this->Internals->MiterJoinLinePipeline)
    {
        useMiterJoinLines = true;
    }
    else if (this->Internals->ThickLineSegmentCount > 0 &&
             this->Internals->ThickLinePipeline)
    {
        useThickLines = true;
    }
}
```

This ensures users still get thick lines, just without round caps.

---

## 5.2 Proper fix: verify primitive topology

Use Xcode GPU capture:

1. Render a scene using `RoundCapRoundJoin`.
2. Inspect the draw call.
3. Check:
   - pipeline primitive topology class
   - actual draw primitive type
   - vertex shader vertex_id logic
   - rendered geometry

If the shader generates independent triangles, change the draw to:

```cpp
recordDraw(
    MTLPrimitiveTypeTriangle,
    0,
    36,
    this->Internals->RoundCapLineSegmentCount);
```

The pipeline descriptor already uses:

```cpp
desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
```

so triangle list is compatible.

---

## 5.3 Proper fix: rewrite round-cap vertex generation as triangle list

If you choose to fully fix round caps, rewrite `vertex_round_cap_line_main` to generate a true triangle list.

A robust structure is:

```text
36 vertices per instance:
  6 vertices for center rectangle = 2 triangles
  15 vertices for start cap = 5 triangles
  15 vertices for end cap = 5 triangles
```

Use local coordinates:

```text
u: along segment, 0 at p0, 1 at p1
s: perpendicular side, [-1, 1]
```

For caps, generate fan triangles around the endpoint centers.

High-level pseudocode:

```metal
struct RoundCapVertex
{
    float t;     // 0 at p0, 1 at p1
    float side;  // -1..1 perpendicular offset
    float capU;  // optional cap-local x
    float capV;  // optional cap-local y
};
```

Then compute:

```metal
float2 center = mix(p0_screen, p1_screen, v.t);
float2 p = center + v.side * y_basis * halfWidth;
```

For caps:

```metal
float angle = ...;
float2 capOffset =
    x_basis * cos(angle) * halfWidth +
    y_basis * sin(angle) * halfWidth;

float2 p = endpointScreen + capOffset;
```

Important:

- Use `halfWidth = lineWidth * 0.5`.
- Do not use full `lineWidth` as the radial offset.
- Keep `dist_to_centerline` in a normalized range for lighting.

---

## 5.4 Validation

Test cases:

1. Single horizontal line with round caps:
   - caps should be semicircles.
   - total width should equal `lineWidth`.

2. Single vertical line:
   - caps should not be distorted.

3. Polyline with multiple segments:
   - joins should not produce gaps or spikes.

4. Compare against OpenGL round joins if available.

5. GPU capture:
   - confirm triangle list vs triangle strip.
   - confirm 36 vertices per instance.
   - confirm no degenerate/broken triangles.

---

# 6. Make Batched Mapper Apply Per-Batch Overrides or Remove Unused Buffer

## Problem

`vtkMetalBatchedPolyDataMapper::UpdateBatchPropertiesBuffer` fills a Metal buffer with per-batch properties:

```cpp
props.ApplyOverrideColors = ...
props.Opacity = ...
props.CompositeId = ...
props.Pickable = ...
props.Ambient[...]
props.Diffuse[...]
```

But this buffer is never bound or consumed by child draws.

Therefore:

- per-element color overrides are ignored,
- per-element opacity overrides are ignored,
- per-element composite IDs are ignored,
- pickability is ignored.

---

## Recommended approach

For Priority 1, implement a **pragmatic per-child override mechanism** that does not require a true batched argument-buffer architecture.

The idea:

- Each child mapper receives per-batch visual override state.
- Overrides are baked into that child mapper’s geometry/uniforms.
- The unused `CompositeDataProperties` buffer is removed or disabled.

This is not the final high-performance batching solution, but it makes overrides functionally correct.

---

## 6.1 Add override state to `vtkMetalPolyDataMapper`

Add public methods:

```cpp
void SetBatchVisualOverride(
    bool overrideColor,
    const double color[3],
    bool overrideOpacity,
    double opacity);

void ClearBatchVisualOverride();
```

In the private internals:

```cpp
bool UseBatchColor = false;
double BatchColor[3] = { 1.0, 1.0, 1.0 };

bool UseBatchOpacity = false;
double BatchOpacity = 1.0;

vtkMTimeType BatchOverrideMTime = 0;
vtkMTimeType CachedBatchOverrideMTime = 0;
```

---

## 6.2 Implement override setter

```cpp
void vtkMetalPolyDataMapper::SetBatchVisualOverride(
    bool overrideColor,
    const double color[3],
    bool overrideOpacity,
    double opacity)
{
    bool changed = false;

    if (this->Internals->UseBatchColor != overrideColor)
    {
        this->Internals->UseBatchColor = overrideColor;
        changed = true;
    }

    if (overrideColor)
    {
        for (int i = 0; i < 3; ++i)
        {
            if (this->Internals->BatchColor[i] != color[i])
            {
                this->Internals->BatchColor[i] = color[i];
                changed = true;
            }
        }
    }

    if (this->Internals->UseBatchOpacity != overrideOpacity)
    {
        this->Internals->UseBatchOpacity = overrideOpacity;
        changed = true;
    }

    if (overrideOpacity)
    {
        if (this->Internals->BatchOpacity != opacity)
        {
            this->Internals->BatchOpacity = opacity;
            changed = true;
        }
    }

    if (changed)
    {
        this->Internals->BatchOverrideMTime++;
        this->Internals->InvalidateRenderBundle();
        this->Modified();
    }
}
```

And:

```cpp
void vtkMetalPolyDataMapper::ClearBatchVisualOverride()
{
    if (this->Internals->UseBatchColor || this->Internals->UseBatchOpacity)
    {
        this->Internals->UseBatchColor = false;
        this->Internals->UseBatchOpacity = false;
        this->Internals->BatchOverrideMTime++;
        this->Internals->InvalidateRenderBundle();
        this->Modified();
    }
}
```

---

## 6.3 Include override state in rebuild invalidation

In `RenderPiece`, add override MTime to the dirty check.

### Before

```cpp
if (currentMTime != this->Internals->CachedInputMTime ||
    representation != this->Internals->CachedRepresentation ||
    edgeVisibility != this->Internals->CachedEdgeVisibility ||
    extraMTime != this->Internals->CachedExtraAttributesMTime ||
    scalarMTime != this->Internals->CachedScalarMTime)
```

### After

```cpp
vtkMTimeType batchOverrideMTime = this->Internals->BatchOverrideMTime;

if (currentMTime != this->Internals->CachedInputMTime ||
    representation != this->Internals->CachedRepresentation ||
    edgeVisibility != this->Internals->CachedEdgeVisibility ||
    extraMTime != this->Internals->CachedExtraAttributesMTime ||
    scalarMTime != this->Internals->CachedScalarMTime ||
    batchOverrideMTime != this->Internals->CachedBatchOverrideMTime)
```

Then store:

```cpp
this->Internals->CachedBatchOverrideMTime = batchOverrideMTime;
```

Also include it in bundle validity:

```cpp
bool bundleValid =
    this->Internals->Bundle.Valid &&
    ...
    this->Internals->BundleExtraAttributesMTime == extraMTime &&
    this->Internals->BundleBatchOverrideMTime == batchOverrideMTime;
```

Add:

```cpp
vtkMTimeType BundleBatchOverrideMTime = 0;
```

and set it at the end of `RebuildRenderBundle`:

```cpp
this->Internals->BundleBatchOverrideMTime = batchOverrideMTime;
```

---

## 6.4 Apply overrides in `BuildGeometryBuffers`

The simplest correct behavior:

- If batch color override is active, ignore scalar colors and use the override color.
- If batch opacity override is active, replace alpha with the override opacity.
- If no scalar colors are present but an override is active, create vertex colors so shaders use the override.

At the top of `BuildGeometryBuffers`, after `mappedColors` is computed:

```cpp
vtkUnsignedCharArray* mappedColors = nullptr;
if (actor && !this->Internals->UseBatchColor)
{
    mappedColors = this->MapScalars(actor->GetProperty()->GetOpacity(), cellFlag);
}
else
{
    cellFlag = 0;
}
```

This disables scalar coloring when a batch color override is active.

Add helper lambdas:

```cpp
double actorOpacity = actor ? actor->GetProperty()->GetOpacity() : 1.0;

auto getOverrideOrDefaultRGBA = [&](float rgba[4])
{
    if (this->Internals->UseBatchColor)
    {
        rgba[0] = static_cast<float>(this->Internals->BatchColor[0]);
        rgba[1] = static_cast<float>(this->Internals->BatchColor[1]);
        rgba[2] = static_cast<float>(this->Internals->BatchColor[2]);
    }
    else if (actor)
    {
        double c[3];
        actor->GetProperty()->GetColor(c);
        rgba[0] = static_cast<float>(c[0]);
        rgba[1] = static_cast<float>(c[1]);
        rgba[2] = static_cast<float>(c[2]);
    }
    else
    {
        rgba[0] = rgba[1] = rgba[2] = 1.0f;
    }

    if (this->Internals->UseBatchOpacity)
    {
        rgba[3] = static_cast<float>(this->Internals->BatchOpacity);
    }
    else
    {
        rgba[3] = static_cast<float>(actorOpacity);
    }
};
```

Then replace default white-color emission.

For example, where code currently does:

```cpp
surfaceColors.push_back(1.0f);
surfaceColors.push_back(1.0f);
surfaceColors.push_back(1.0f);
surfaceColors.push_back(1.0f);
```

replace with:

```cpp
float rgba[4];
getOverrideOrDefaultRGBA(rgba);
surfaceColors.push_back(rgba[0]);
surfaceColors.push_back(rgba[1]);
surfaceColors.push_back(rgba[2]);
surfaceColors.push_back(rgba[3]);
```

For mapped colors, if opacity override is active, replace alpha:

```cpp
if (mappedColors)
{
    const unsigned char* rgba = mappedColors->GetPointer(0);

    float r = rgba[idx * 4 + 0] / 255.0f;
    float g = rgba[idx * 4 + 1] / 255.0f;
    float b = rgba[idx * 4 + 2] / 255.0f;
    float a = rgba[idx * 4 + 3] / 255.0f;

    if (this->Internals->UseBatchOpacity)
    {
        a = static_cast<float>(this->Internals->BatchOpacity);
    }

    surfaceColors.push_back(r);
    surfaceColors.push_back(g);
    surfaceColors.push_back(b);
    surfaceColors.push_back(a);
}
```

Do this consistently for:

- triangle vertices,
- wireframe line vertices,
- explicit line vertices,
- edge vertex colors if used,
- fallback color buffers.

---

## 6.5 Force surface-color flag when overrides are active

At the end of `BuildGeometryBuffers`, when creating the surface color buffer:

```cpp
if (!surfaceColors.empty())
{
    id<MTLBuffer> surfColorBuf = [device
        newBufferWithBytes:surfaceColors.data()
        length:surfaceColors.size() * sizeof(float)
        options:MTLResourceStorageModeShared];

    vtkMetalMRC::AssignConsumed(this->Internals->SurfaceColorBuffer, surfColorBuf);

    bool hasMappedColors = (mappedColors != nullptr);
    bool hasOverrideColors =
        this->Internals->UseBatchColor || this->Internals->UseBatchOpacity;

    this->Internals->HasSurfaceColors =
        hasMappedColors || hasOverrideColors;
}
```

If positions exist but `surfaceColors` is empty and an override is active, create override colors:

```cpp
else if (!positions.empty())
{
    float rgba[4];
    getOverrideOrDefaultRGBA(rgba);

    std::vector<float> colors(positions.size() / 3 * 4);

    for (size_t i = 0; i < colors.size(); i += 4)
    {
        colors[i + 0] = rgba[0];
        colors[i + 1] = rgba[1];
        colors[i + 2] = rgba[2];
        colors[i + 3] = rgba[3];
    }

    id<MTLBuffer> colorBuf = [device
        newBufferWithBytes:colors.data()
        length:colors.size() * sizeof(float)
        options:MTLResourceStorageModeShared];

    vtkMetalMRC::AssignConsumed(this->Internals->SurfaceColorBuffer, colorBuf);

    this->Internals->HasSurfaceColors =
        this->Internals->UseBatchColor ||
        this->Internals->UseBatchOpacity;
}
```

---

## 6.6 Apply overrides to point colors

When building point colors:

```cpp
std::vector<float> pointColors(numPts * 4, 1.0f);

if (this->Internals->UseBatchColor || this->Internals->UseBatchOpacity)
{
    float rgba[4];
    getOverrideOrDefaultRGBA(rgba);

    for (vtkIdType i = 0; i < numPts; ++i)
    {
        pointColors[i * 4 + 0] = rgba[0];
        pointColors[i * 4 + 1] = rgba[1];
        pointColors[i * 4 + 2] = rgba[2];
        pointColors[i * 4 + 3] = rgba[3];
    }
}
else if (mappedColors && cellFlag == 0 &&
         mappedColors->GetNumberOfTuples() >= numPts)
{
    const unsigned char* rgba = mappedColors->GetPointer(0);

    for (vtkIdType i = 0; i < numPts; ++i)
    {
        pointColors[i * 4 + 0] = rgba[i * 4 + 0] / 255.0f;
        pointColors[i * 4 + 1] = rgba[i * 4 + 1] / 255.0f;
        pointColors[i * 4 + 2] = rgba[i * 4 + 2] / 255.0f;
        pointColors[i * 4 + 3] = rgba[i * 4 + 3] / 255.0f;
    }
}
```

---

## 6.7 Adjust material opacity to avoid double multiplication

Some shaders multiply vertex color alpha by material opacity, especially thick-line shaders.

To avoid `opacity * opacity`, when overrides bake alpha into vertex colors, set material opacity to `1`.

In `UpdateMaterialUniforms`:

```cpp
float effectiveOpacity = 1.0f;

if (this->Internals->UseBatchOpacity)
{
    // Alpha is baked into vertex/point colors.
    effectiveOpacity = 1.0f;
}
else
{
    effectiveOpacity = static_cast<float>(prop->GetOpacity());
}

mu[16] = effectiveOpacity;
```

If batch color override is active, also set ambient/diffuse colors to the override color:

```cpp
if (this->Internals->UseBatchColor)
{
    mu[0] = static_cast<float>(this->Internals->BatchColor[0]);
    mu[1] = static_cast<float>(this->Internals->BatchColor[1]);
    mu[2] = static_cast<float>(this->Internals->BatchColor[2]);

    mu[4] = mu[0];
    mu[5] = mu[1];
    mu[6] = mu[2];
}
else
{
    double ac[3];
    prop->GetAmbientColor(ac);
    mu[0] = static_cast<float>(ac[0]);
    mu[1] = static_cast<float>(ac[1]);
    mu[2] = static_cast<float>(ac[2]);

    double dc[3];
    prop->GetDiffuseColor(dc);
    mu[4] = static_cast<float>(dc[0]);
    mu[5] = static_cast<float>(dc[1]);
    mu[6] = static_cast<float>(dc[2]);
}
```

---

## 6.8 Apply overrides to edge color uniform

In `UpdateEdgeColorUniform`:

```cpp
float ec[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

if (this->Internals->UseBatchColor)
{
    ec[0] = static_cast<float>(this->Internals->BatchColor[0]);
    ec[1] = static_cast<float>(this->Internals->BatchColor[1]);
    ec[2] = static_cast<float>(this->Internals->BatchColor[2]);
}
else
{
    vtkProperty* prop = actor->GetProperty();
    if (prop)
    {
        double c[3];
        prop->GetEdgeColor(c);
        ec[0] = static_cast<float>(c[0]);
        ec[1] = static_cast<float>(c[1]);
        ec[2] = static_cast<float>(c[2]);
    }
}

if (this->Internals->UseBatchOpacity)
{
    ec[3] = static_cast<float>(this->Internals->BatchOpacity);
}
else
{
    ec[3] = 1.0f;
}
```

---

## 6.9 Apply overrides from the batched mapper

In `vtkMetalBatchedPolyDataMapper::RenderPiece`, before rendering each child:

```cpp
for (const BatchElement* elem : visible)
{
    vtkSmartPointer<vtkMetalPolyDataMapper> mapper =
        this->GetChildMapper(elem->PolyData);

    if (!mapper)
    {
        continue;
    }

    double overrideColor[3] =
    {
        elem->DiffuseColor[0],
        elem->DiffuseColor[1],
        elem->DiffuseColor[2]
    };

    double actorOpacity = act->GetProperty()->GetOpacity();

    bool overrideOpacity =
        (elem->Opacity != actorOpacity);

    mapper->SetBatchVisualOverride(
        elem->OverridesColor,
        overrideColor,
        overrideOpacity,
        elem->Opacity);

    mapper->RenderPiece(ren, act);
}
```

If `BatchElement` later gains an explicit `OverridesOpacity` flag, use that instead of comparing against actor opacity.

---

## 6.10 Remove or disable the unused `CompositeDataProperties` buffer

Since the new approach applies overrides through child mapper state, remove the unused buffer to avoid confusion.

Remove or comment out:

```cpp
void UpdateBatchPropertiesBuffer(void* mtlDevice);
void* BatchPropertiesBuffer;
size_t BatchPropertiesBufferSize;
void SetBatchPropertiesBufferConsumed(void* buffer);
void ReleaseBatchPropertiesBuffer();
struct CompositeDataProperties;
```

Also remove calls:

```cpp
this->UpdateBatchPropertiesBuffer(renWin->GetMetalDevice());
```

If you want to preserve the code for future true batching, wrap it in:

```cpp
#if 0
...
#endif
```

But for Priority 1, removing dead code is preferable.

---

## 6.11 Validation

Test cases:

1. Composite dataset with one block overriding color:
   - that block should render with the override color.

2. Composite dataset with one block overriding opacity:
   - that block should render with the override opacity.

3. Toggle override color on/off:
   - block should rebuild and display correctly.

4. Scalar coloring enabled:
   - non-overridden blocks use scalars.
   - color-overridden blocks use override color.

5. Edge visibility:
   - overridden blocks use override edge color.

6. Points representation:
   - overridden blocks use override point color.

---

# 7. Fix Stale Flat-Index / Batch-Element Mappings

## Problem

The batched mapper currently keys elements by raw pointer address:

```cpp
std::map<std::uintptr_t, std::unique_ptr<BatchElement>> VTKPolyDataToBatchElement;
```

Issues:

1. If a `vtkPolyData` is deleted and a new one is allocated at the same address, stale state may be reused.
2. If a flat index is reassigned to a different polydata, the old element may remain.
3. `Parent` is a raw pointer and may dangle.

---

## 7.1 Store weak references to polydata

Add a parallel map:

```cpp
std::map<std::uintptr_t, vtkWeakPointer<vtkPolyData>> PolyDataRefs;
```

Include:

```cpp
#include "vtkWeakPointer.h"
```

When adding an element:

```cpp
auto address = reinterpret_cast<std::uintptr_t>(element.PolyData);
```

Check whether the existing address refers to a dead or different object:

```cpp
auto refIt = this->PolyDataRefs.find(address);
if (refIt != this->PolyDataRefs.end())
{
    vtkPolyData* old = refIt->second;

    if (old == nullptr || old != element.PolyData)
    {
        // Stale address reuse. Remove old element and child mapper.
        this->VTKPolyDataToBatchElement.erase(address);

        auto childIt = this->ChildMappers.find(address);
        if (childIt != this->ChildMappers.end())
        {
            if (childIt->second)
            {
                childIt->second->ReleaseGraphicsResources(nullptr);
            }
            this->ChildMappers.erase(childIt);
        }

        this->PolyDataRefs.erase(refIt);
        this->GeometryDirty = true;
        this->Modified();
    }
}
```

Then store:

```cpp
this->PolyDataRefs[address] = element.PolyData;
```

---

## 7.2 Validate weak references before rendering

In `RenderPiece`, when building the visible list:

```cpp
for (const auto& kv : this->VTKPolyDataToBatchElement)
{
    const BatchElement* elem = kv.second.get();

    if (!elem || !elem->Visibility || !elem->PolyData)
    {
        continue;
    }

    auto refIt = this->PolyDataRefs.find(kv.first);
    if (refIt == this->PolyDataRefs.end() || refIt->second == nullptr)
    {
        continue;
    }

    visible.push_back(elem);
}
```

Optionally, periodically purge dead references:

```cpp
for (auto it = this->PolyDataRefs.begin(); it != this->PolyDataRefs.end();)
{
    if (it->second == nullptr)
    {
        auto addr = it->first;

        this->VTKPolyDataToBatchElement.erase(addr);

        auto childIt = this->ChildMappers.find(addr);
        if (childIt != this->ChildMappers.end())
        {
            if (childIt->second)
            {
                childIt->second->ReleaseGraphicsResources(nullptr);
            }
            this->ChildMappers.erase(childIt);
        }

        it = this->PolyDataRefs.erase(it);
        changed = true;
    }
    else
    {
        ++it;
    }
}
```

---

## 7.3 Fix flat-index reassignment

When inserting a new element for a flat index, remove the previous element that claimed that flat index.

In `AddBatchElement`:

```cpp
auto oldFlatIt = this->FlatIndexToPolyData.find(flatIndex);
if (oldFlatIt != this->FlatIndexToPolyData.end())
{
    std::uintptr_t oldAddress = oldFlatIt->second;

    if (oldAddress != address)
    {
        auto oldElemIt = this->VTKPolyDataToBatchElement.find(oldAddress);

        if (oldElemIt != this->VTKPolyDataToBatchElement.end())
        {
            // Only remove if that element still believes it owns this flat index.
            if (oldElemIt->second->FlatIndex == flatIndex)
            {
                auto oldChildIt = this->ChildMappers.find(oldAddress);
                if (oldChildIt != this->ChildMappers.end())
                {
                    if (oldChildIt->second)
                    {
                        oldChildIt->second->ReleaseGraphicsResources(nullptr);
                    }
                    this->ChildMappers.erase(oldChildIt);
                }

                this->PolyDataRefs.erase(oldAddress);
                this->VTKPolyDataToBatchElement.erase(oldElemIt);
            }
        }
    }
}
```

Then keep the existing cleanup that removes other flat-index entries pointing to the same address:

```cpp
for (auto it = this->FlatIndexToPolyData.begin();
     it != this->FlatIndexToPolyData.end();)
{
    if (it->second == address && it->first != flatIndex)
    {
        it = this->FlatIndexToPolyData.erase(it);
    }
    else
    {
        ++it;
    }
}
```

Then assign:

```cpp
this->FlatIndexToPolyData[flatIndex] = address;
```

---

## 7.4 Make `Parent` a weak pointer

Replace:

```cpp
vtkCompositePolyDataMapper* Parent = nullptr;
```

with:

```cpp
vtkWeakPointer<vtkCompositePolyDataMapper> Parent;
```

Then:

```cpp
void vtkMetalBatchedPolyDataMapper::SetParent(vtkCompositePolyDataMapper* parent)
{
    this->Parent = parent;

    if (parent)
    {
        this->SetInputDataObject(0, parent->GetInputDataObject(0, 0));
    }
    else
    {
        this->SetInputDataObject(0, nullptr);
    }
}
```

And:

```cpp
vtkMTimeType vtkMetalBatchedPolyDataMapper::GetMTime()
{
    if (this->Parent)
    {
        return std::max(this->Superclass::GetMTime(), this->Parent->GetMTime());
    }

    return this->Superclass::GetMTime();
}
```

---

## 7.5 Validation

Test cases:

1. Add batch elements, render, clear unmarked:
   - no stale elements remain.

2. Reuse the same flat index with a different `vtkPolyData`:
   - old polydata should not render.

3. Delete a `vtkPolyData` while batch mapper still alive:
   - no crash.
   - weak reference becomes null.

4. Force allocator address reuse:
   - create polydata A,
   - delete A,
   - create B likely at same address,
   - ensure B does not inherit A’s child mapper state incorrectly.

---

# 8. Guard Depth Peeling Against MSAA Incompatibility

## Problem

The depth peeler creates ordinary single-sample textures:

```cpp
MTLTextureDescriptor texture2DDescriptorWithPixelFormat:...
```

But mapper pipelines may be created with MSAA sample count:

```cpp
desc.rasterSampleCount = sampleCount;
```

If `sampleCount > 1`, rendering multisample pipelines into single-sample textures is invalid.

---

## Recommended Priority 1 fix

Disable depth peeling when MSAA is active.

Full MSAA-compatible depth peeling is a larger feature requiring:

- multisample peel textures,
- resolve passes,
- sample-aware peeling shaders,
- renderer pass graph changes.

That is not Priority 1.

---

## 8.1 Add MSAA guard in depth peeler

In `vtkMetalDepthPeeler::RenderTranslucentGeometry`, after obtaining `renWin`:

```cpp
int sampleCount = renWin->GetEffectiveSampleCount();

if (sampleCount > 1)
{
    vtkWarningMacro(
        "vtkMetalDepthPeeler: depth peeling is not supported with MSAA. "
        "Falling back to standard translucent rendering.");

    renWin->DepthPeelingMode = 0;
    renWin->PeelFrontTexture = nullptr;
    renWin->PeelDepthTexture = nullptr;
    renWin->PeelIndex = 0;

    // Caller should render translucent geometry normally.
    return 0;
}
```

The caller, likely `vtkMetalRenderer`, should handle `return 0` by executing the standard translucent pass.

Example caller logic:

```cpp
if (useDepthPeeling)
{
    int peels = depthPeeler->RenderTranslucentGeometry(...);

    if (peels == 0)
    {
        this->RenderTranslucentGeometry();
    }
}
else
{
    this->RenderTranslucentGeometry();
}
```

---

## 8.2 Force peel pipelines to single-sample

Even if the main window has MSAA enabled, peel pipelines should be single-sample because peel textures are single-sample.

In `EnsurePeelPipelineStates`, replace:

```cpp
desc.rasterSampleCount = this->Internals->CachedSampleCount > 0
    ? this->Internals->CachedSampleCount : 1;
```

with:

```cpp
// Depth peeling currently uses single-sample textures.
desc.rasterSampleCount = 1;
```

Do this for both:

- `TriangleInitPeelPipeline`
- `TrianglePeelPipeline`

---

## 8.3 Validation

Test cases:

1. Disable MSAA:
   - depth peeling should work.

2. Enable MSAA:
   - depth peeling should not start.
   - no Metal validation errors.
   - translucent geometry should fall back to normal blending.

3. Toggle MSAA at runtime:
   - no stale peel pipelines.
   - no mismatched sample counts.

---

# 9. Ensure Depth-Peel Textures Are Always Bound When Required

## Problem

For peel mode 2, the peel fragment shader expects:

```metal
texture2d<float, access::read> prevFrontTex [[texture(1)]],
texture2d<float, access::read> prevDepthTex [[texture(2)]]
```

But the mapper currently binds them only if non-nil:

```cpp
if (prevFront)
{
    recordFTex(prevFront, 1);
}
if (prevDepth)
{
    recordFTex(prevDepth, 2);
}
```

If either is nil, shader validation may fail.

---

## 9.1 Guarantee textures in the depth peeler

In `vtkMetalDepthPeeler::RenderTranslucentGeometry`, after creating textures:

```cpp
if (!this->FrontPeelA ||
    !this->FrontPeelB ||
    !this->BackPeelTemp ||
    !this->BackAccum ||
    !this->DepthPeelA ||
    !this->DepthPeelB)
{
    vtkErrorMacro("vtkMetalDepthPeeler: failed to create peel textures.");
    return 0;
}
```

Before each peel pass:

```cpp
if (!frontSrc || !frontDst || !depthSrc || !depthDst || !this->BackPeelTemp)
{
    vtkErrorMacro("vtkMetalDepthPeeler: invalid peel textures.");
    break;
}
```

---

## 9.2 Guard mapper bundle recording

In `RebuildRenderBundle`, for peel mode 2:

```cpp
else if (peelMode == 2 && this->Internals->TrianglePeelPipeline)
{
    vtkMetalRenderWindow* peelRenWin =
        vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());

    id<MTLTexture> prevFront = nil;
    id<MTLTexture> prevDepth = nil;

    if (peelRenWin)
    {
        prevFront = (__bridge id<MTLTexture>)peelRenWin->PeelFrontTexture;
        prevDepth = (__bridge id<MTLTexture>)peelRenWin->PeelDepthTexture;
    }

    if (!prevFront || !prevDepth)
    {
        vtkWarningMacro(
            "vtkMetalPolyDataMapper: missing depth-peel textures. "
            "Skipping peel draw for this pass.");

        // Do not record triangle draws for this peel pass.
    }
    else
    {
        recordPipeline(this->Internals->TrianglePeelPipeline);
        recordFTex(prevFront, 1);
        recordFTex(prevDepth, 2);

        // Continue recording buffers/draws...
    }
}
```

You may need to wrap the rest of the triangle draw recording in a boolean:

```cpp
bool peelBindingsValid = true;
```

Then:

```cpp
if (peelBindingsValid)
{
    recordVBuf(...);
    ...
    recordDraw(...);
}
```

---

## 9.3 Do not fallback to non-peel pipeline inside a peel render pass

Important: if the current render pass was created by the depth peeler with three color attachments, you cannot simply bind the normal triangle pipeline.

The normal triangle pipeline expects:

```text
color0: BGRA8Unorm
color1: RGBA32Uint
depth: Depth32Float
```

The peel pass expects:

```text
color0: RGBA8Unorm
color1: RGBA8Unorm
color2: RG32Float
no depth attachment
```

Therefore, if peel textures are missing, the safest action inside the mapper is:

```cpp
skip drawing for this pass
```

and report a warning.

The higher-level fix is to ensure the depth peeler never starts a peel pass without valid textures.

---

## 9.4 Validation

Test cases:

1. Normal depth peeling:
   - front and depth textures bound.
   - no validation errors.

2. Simulate missing peel texture:
   - mapper should skip draw and warn.
   - no crash.

3. Run peel loop with maximum peels:
   - ping-pong textures remain valid.
   - no nil bindings after swaps.

---

# Regression Test Matrix

After implementing all Priority 1 items, run this matrix.

## Representation tests

| Geometry | Representation | Expected |
|---|---:|---|
| Sphere | Surface | filled triangles |
| Sphere | Wireframe | polygon edges only |
| Sphere | Points | points only |
| Sphere + lines | Surface | triangles + lines |
| Sphere + lines | Wireframe | wireframe + lines |
| Sphere + lines | Points | points only |
| Sphere + edges | Surface + EdgeVisibility | triangles + edges |
| Sphere + edges | Wireframe + EdgeVisibility | wireframe only |
| Sphere + edges | Points + EdgeVisibility | points only |

---

## Flag tests

| Action | Expected |
|---|---|
| Toggle vertex visibility | dots appear/disappear immediately |
| Toggle texture | texture appears/disappears immediately |
| Toggle scalar colors | color source changes immediately |
| Toggle sphere points | point shading changes immediately |
| Toggle point shape | shape changes immediately |

---

## Picking tests

| Scene | Expected |
|---|---|
| One actor | prop ID 1 |
| Two actors | prop IDs 1 and 2 |
| Background | prop ID 0 |
| Triangle cell pick | nonzero cell ID |
| Line cell pick | nonzero cell ID |
| Point pick | nonzero point/cell ID |

---

## Line tests

| Line width | Expected |
|---:|---|
| 1 | approximately 1 px |
| 2 | approximately 2 px |
| 5 | approximately 5 px |
| 10 | approximately 10 px |

Additional:

- diagonal lines maintain width,
- miter joins do not spike excessively,
- round caps are disabled or verified correct,
- line colors respect scalar/override colors.

---

## Batch mapper tests

| Case | Expected |
|---|---|
| Many blocks, no overrides | renders correctly |
| One block override color | block uses override color |
| One block override opacity | block uses override opacity |
| Toggle visibility | block appears/disappears |
| Reassign flat index | old block does not remain |
| Delete polydata | no crash |
| Clear unmarked | stale child mappers released |

---

## Depth peeling tests

| Case | Expected |
|---|---|
| MSAA off, peeling on | correct OIT |
| MSAA on, peeling requested | fallback, no validation error |
| Many translucent layers | stable blending |
| Missing peel texture | warning/skip, no crash |

---

# Suggested Commit Sequence

To make review easier, split the work into separate commits.

## Commit 1: Representation gating

- Fix `VTK_POINTS` drawing triangles.
- Gate line drawing by representation.
- Gate edge overlay by representation.
- Optional: skip building unused geometry.

## Commit 2: Scene flag clearing

- Add flag constants.
- Clear dynamic actor flags every frame.
- Integrate texture flag.

## Commit 3: Prop ID convention

- Make prop IDs zero-based on CPU.
- Document convention.
- Update picking tests.

## Commit 4: Line width scaling

- Fix thick-line half-width.
- Fix miter half-width.
- Improve line normal fallback.

## Commit 5: Round-cap safety

- Disable round-cap path by default, or
- switch to triangle list and verify/rewrite shader.

## Commit 6: Batch visual overrides

- Add child mapper override state.
- Bake override colors/opacity into child geometry/uniforms.
- Remove unused batch property buffer.

## Commit 7: Batch lifetime and flat-index hygiene

- Add weak polydata references.
- Clean stale flat-index mappings.
- Make parent weak.

## Commit 8: Depth peeling safety

- Disable peeling under MSAA.
- Force peel pipelines to single-sample.
- Validate peel textures.
- Skip draws if peel bindings are invalid.

---

# Notes for Future Work

These are not Priority 1, but they should follow:

1. **True batched rendering**
   - combined vertex/index buffers,
   - indirect draws,
   - argument buffers,
   - per-batch IDs and properties without rebuilding child mappers.

2. **Unified picking ID convention**
   - all IDs zero-based on CPU,
   - shaders output `id + 1`,
   - explicit support for unpickable objects.

3. **Native Metal render bundles**
   - replace custom command replay with `MTLRenderBundle` where possible.

4. **Shader binding table**
   - replace magic buffer indices with shared constants or argument buffers.

5. **MSAA-compatible depth peeling**
   - multisample peel textures,
   - resolve passes,
   - sample-aware peeling.

6. **Round-cap/join line rewrite**
   - mathematically consistent caps,
   - consistent half-width convention,
   - consistent lighting normals.

---

This completes the detailed Priority 1 implementation plan while leaving the volume ray-cast mapper shaders untouched.
