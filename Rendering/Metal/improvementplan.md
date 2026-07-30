Below is a detailed, implementation-oriented guide for the **P1 + P2 + P3** items from the review.

This guide assumes the codebase you pasted is the working tree and focuses on:

- **P1:** Prevent GPU crashes / validation errors
- **P2:** Fix visual correctness
- **P3:** Fix picking correctness

I have organized this as a sequence of concrete tasks with:

- what to change
- where to change it
- why it matters
- example code
- validation criteria

---

# 0. Recommended implementation order

Do the work in this order:

1. **P1 first**, because these issues can cause undefined behavior or GPU crashes.
2. **P2 second**, because these are rendering correctness issues visible to users.
3. **P3 last**, because picking correctness is important but usually less catastrophic than GPU crashes or visual bugs.

Also enable Metal validation during development:

- Xcode Metal API validation
- GPU frame capture
- shader validation
- runtime assertions

---

# 1. Shared preparatory work

Before implementing P1–P3, add a few shared helpers and fields. These will make the later fixes cleaner.

---

## 1.1. Add new internal fields to `vtkMetalPolyDataMapperInternals`

In `vtkMetalPolyDataMapper.mm`, extend the internals struct.

Add at least:

```cpp
id<MTLBuffer> EdgeUVBuffer = nil;

bool HasSurfaceAlpha = false;

// P3: primitive-cell-ID buffers for accurate picking
id<MTLBuffer> TrianglePrimitiveCellIdBuffer = nil;
id<MTLBuffer> EdgePrimitiveCellIdBuffer = nil;

// Optional but useful for fallback bindings
id<MTLBuffer> ZeroTriangleCellIdBuffer = nil;
id<MTLBuffer> ZeroLineCellIdBuffer = nil;
id<MTLBuffer> ZeroEdgeCellIdBuffer = nil;
id<MTLBuffer> ZeroTriangleUVBuffer = nil;
id<MTLBuffer> ZeroEdgeUVBuffer = nil;
```

Release them in `ReleaseBuffers()`:

```cpp
vtkMetalMRC::ReleaseAndNil(EdgeUVBuffer);

vtkMetalMRC::ReleaseAndNil(TrianglePrimitiveCellIdBuffer);
vtkMetalMRC::ReleaseAndNil(EdgePrimitiveCellIdBuffer);

vtkMetalMRC::ReleaseAndNil(ZeroTriangleCellIdBuffer);
vtkMetalMRC::ReleaseAndNil(ZeroLineCellIdBuffer);
vtkMetalMRC::ReleaseAndNil(ZeroEdgeCellIdBuffer);
vtkMetalMRC::ReleaseAndNil(ZeroTriangleUVBuffer);
vtkMetalMRC::ReleaseAndNil(ZeroEdgeUVBuffer);

HasSurfaceAlpha = false;
```

---

## 1.2. Add small buffer-creation helpers

Inside `vtkMetalPolyDataMapper.mm`, add file-local helpers:

```cpp
namespace
{

id<MTLBuffer> CreateZeroBuffer(id<MTLDevice> device, size_t bytes)
{
  if (!device || bytes == 0)
  {
    return nil;
  }

  id<MTLBuffer> buffer =
      [device newBufferWithLength:bytes
                          options:MTLResourceStorageModeShared];

  if (buffer)
  {
    memset([buffer contents], 0, bytes);
  }

  return buffer;
}

id<MTLBuffer> CreateFloatFillBuffer(id<MTLDevice> device, size_t count, float value)
{
  if (!device || count == 0)
  {
    return nil;
  }

  std::vector<float> data(count, value);

  return [device newBufferWithBytes:data.data()
                             length:data.size() * sizeof(float)
                            options:MTLResourceStorageModeShared];
}

} // namespace
```

These are useful for fallback UVs, cell IDs, and default attributes.

---

# 2. P1: Prevent GPU crashes / validation errors

These are the highest-priority fixes.

---

# P1-1. Bind `ClipPlaneBuffer` in the standard 1px line path

## Problem

The standard line path uses:

- `vertex_main`
- `fragment_main`

Both shaders declare clip-plane bindings:

```metal
constant ClipPlaneUniforms& clipPlanes [[buffer(5)]]
```

But the standard line section of `RebuildRenderBundle()` does not bind `ClipPlaneBuffer`.

This can cause:

- Metal validation errors
- undefined reads
- GPU crashes
- broken clipping behavior

---

## Where to change

File:

```text
vtkMetalPolyDataMapper.mm
```

Function:

```cpp
vtkMetalPolyDataMapper::RebuildRenderBundle(...)
```

Section:

```cpp
// Standard 1px lines
```

---

## Implementation

Find the standard line path:

```cpp
else
{
  // Standard 1px lines
  recordPipeline(this->Internals->LinePipeline);
  recordVBuf(this->Internals->VertexPositionBuffer, 0, 0);
  ...
}
```

After the scene/coincident bindings, add:

```cpp
if (this->Internals->ClipPlaneBuffer)
{
  recordFBuf(this->Internals->ClipPlaneBuffer, 0, 5);
  recordVBuf(this->Internals->ClipPlaneBuffer, 0, 5);
}
```

A good location is immediately after:

```cpp
if (this->Internals->CoincidentOffsetBuffer)
{
  recordFBuf(this->Internals->CoincidentOffsetBuffer, 0, 3);
}
```

So the standard line path becomes:

```cpp
if (this->Internals->CoincidentOffsetBuffer)
{
  recordFBuf(this->Internals->CoincidentOffsetBuffer, 0, 3);
}

if (this->Internals->ClipPlaneBuffer)
{
  recordFBuf(this->Internals->ClipPlaneBuffer, 0, 5);
  recordVBuf(this->Internals->ClipPlaneBuffer, 0, 5);
}

if (this->Internals->LineCellIdBuffer)
{
  recordVBuf(this->Internals->LineCellIdBuffer, 0, 6);
}
```

---

## Extra safety

`UpdateClipPlaneUniforms()` already creates `ClipPlaneBuffer` when an actor exists.

However, for defensive programming, ensure it exists before bundle recording.

In `RenderPiece()`, after:

```cpp
this->UpdateClipPlaneUniforms((void*)device, act);
```

add:

```cpp
if (!this->Internals->ClipPlaneBuffer)
{
  float cp[28] = {};
  id<MTLBuffer> buffer =
      [device newBufferWithBytes:cp
                          length:sizeof(cp)
                         options:MTLResourceStorageModeShared];

  vtkMetalMRC::AssignConsumed(this->Internals->ClipPlaneBuffer, buffer);
}
```

---

## Validation criteria

- Render a dataset with lines and clipping planes enabled.
- Enable Metal validation.
- Confirm no missing-buffer validation errors.
- Confirm clipping works for 1px lines.

---

# P1-2. Create and bind an edge UV buffer for the edge overlay path

## Problem

The edge overlay path uses `vertex_main`.

`vertex_main` unconditionally reads:

```metal
constant float2* triangleUVs [[buffer(8)]]
...
out.uv = triangleUVs[vertex_id];
```

But the edge overlay recording does not bind anything at vertex buffer index 8.

This is unsafe.

---

## Where to change

Files:

```text
vtkMetalPolyDataMapper.mm
MetalShaders.metal
```

Primary C++ functions:

```cpp
BuildGeometryBuffers(...)
RebuildRenderBundle(...)
ReleaseBuffers()
```

---

## Implementation plan

You need an `EdgeUVBuffer` whose length matches `EdgeVertexCount`.

There are three cases:

1. CPU-built edge overlay
2. GPU-built edge overlay
3. Fallback zero UV buffer

---

## Step 1: Add `EdgeUVBuffer` to internals

Already described in Section 1.1.

---

## Step 2: Populate UVs for CPU edge overlay

In `BuildGeometryBuffers()`, locate the CPU edge overlay section:

```cpp
if (representation == VTK_SURFACE && edgeVisibility)
{
  ...
}
```

Add a local vector:

```cpp
std::vector<float> edgeUVs;
```

Near:

```cpp
std::vector<float> edgePositions;
std::vector<float> edgeNormals;
std::vector<float> edgeColors;
std::vector<uint32_t> edgeIndices;
```

Then modify `addEdgeVertex`.

Current logic:

```cpp
auto addEdgeVertex = [&](vtkIdType pointId, vtkIdType cellId = 0) -> uint32_t {
  ...
};
```

After normal emission and color emission, add UV emission:

```cpp
if (tcoordArray && tcoordArray->GetNumberOfTuples() > pointId)
{
  double uv[3];
  tcoordArray->GetTuple(pointId, uv);
  edgeUVs.push_back(static_cast<float>(uv[0]));
  edgeUVs.push_back(static_cast<float>(uv[1]));
}
else
{
  edgeUVs.push_back(0.0f);
  edgeUVs.push_back(0.0f);
}
```

So `addEdgeVertex` becomes conceptually:

```cpp
auto addEdgeVertex = [&](vtkIdType pointId, vtkIdType cellId = 0) -> uint32_t
{
  auto it = edgeVertexMap.find(pointId);
  if (it != edgeVertexMap.end())
  {
    return it->second;
  }

  uint32_t idx = static_cast<uint32_t>(edgePositions.size() / 3);
  edgeVertexMap[pointId] = idx;

  double p[3];
  polydata->GetPoint(pointId, p);
  edgePositions.push_back(static_cast<float>(p[0]));
  edgePositions.push_back(static_cast<float>(p[1]));
  edgePositions.push_back(static_cast<float>(p[2]));

  if (normalArray)
  {
    double n[3];
    normalArray->GetTuple(pointId, n);
    edgeNormals.push_back(static_cast<float>(n[0]));
    edgeNormals.push_back(static_cast<float>(n[1]));
    edgeNormals.push_back(static_cast<float>(n[2]));
  }

  if (mappedColors)
  {
    const unsigned char* rgba = mappedColors->GetPointer(0);
    vtkIdType idx2 = (cellFlag == 0) ? pointId : cellId;

    edgeColors.push_back(rgba[idx2 * 4 + 0] / 255.0f);
    edgeColors.push_back(rgba[idx2 * 4 + 1] / 255.0f);
    edgeColors.push_back(rgba[idx2 * 4 + 2] / 255.0f);
    edgeColors.push_back(rgba[idx2 * 4 + 3] / 255.0f);
  }
  else
  {
    edgeColors.push_back(1.0f);
    edgeColors.push_back(1.0f);
    edgeColors.push_back(1.0f);
    edgeColors.push_back(1.0f);
  }

  if (tcoordArray && tcoordArray->GetNumberOfTuples() > pointId)
  {
    double uv[3];
    tcoordArray->GetTuple(pointId, uv);
    edgeUVs.push_back(static_cast<float>(uv[0]));
    edgeUVs.push_back(static_cast<float>(uv[1]));
  }
  else
  {
    edgeUVs.push_back(0.0f);
    edgeUVs.push_back(0.0f);
  }

  edgeVertexCellIds.push_back(static_cast<uint32_t>(cellId + polyCellOffset) + 1u);

  return idx;
};
```

---

## Step 3: Upload CPU edge UV buffer

Later in the edge-buffer upload block:

```cpp
if (!edgeIndices.empty() && !edgePositions.empty())
{
  ...
}
```

After creating `EdgeSurfaceColorBuffer`, add:

```cpp
if (!edgeUVs.empty())
{
  id<MTLBuffer> uvBuffer =
      [device newBufferWithBytes:edgeUVs.data()
                          length:edgeUVs.size() * sizeof(float)
                         options:MTLResourceStorageModeShared];

  vtkMetalMRC::AssignConsumed(this->Internals->EdgeUVBuffer, uvBuffer);
}
else
{
  size_t uvCount = static_cast<size_t>(this->Internals->EdgeVertexCount) * 2;
  id<MTLBuffer> uvBuffer = CreateZeroBuffer(device, uvCount * sizeof(float));
  vtkMetalMRC::AssignConsumed(this->Internals->EdgeUVBuffer, uvBuffer);
}
```

---

## Step 4: Handle GPU edge overlay

In the GPU tessellation path, edge vertices usually alias the main point vertices.

After this block:

```cpp
if (gpuTessUsed && this->Internals->HasEdgeOverlay)
{
  vtkMetalMRC::AssignRetained(
      this->Internals->EdgeVertexPositionBuffer,
      this->Internals->VertexPositionBuffer);

  vtkMetalMRC::AssignRetained(
      this->Internals->EdgeVertexNormalBuffer,
      this->Internals->VertexNormalBuffer);

  vtkMetalMRC::AssignRetained(
      this->Internals->EdgeSurfaceColorBuffer,
      this->Internals->SurfaceColorBuffer);
}
```

Add:

```cpp
if (this->Internals->TriangleUVBuffer)
{
  vtkMetalMRC::AssignRetained(
      this->Internals->EdgeUVBuffer,
      this->Internals->TriangleUVBuffer);
}
```

Full block:

```cpp
if (gpuTessUsed && this->Internals->HasEdgeOverlay)
{
  vtkMetalMRC::AssignRetained(
      this->Internals->EdgeVertexPositionBuffer,
      this->Internals->VertexPositionBuffer);

  vtkMetalMRC::AssignRetained(
      this->Internals->EdgeVertexNormalBuffer,
      this->Internals->VertexNormalBuffer);

  vtkMetalMRC::AssignRetained(
      this->Internals->EdgeSurfaceColorBuffer,
      this->Internals->SurfaceColorBuffer);

  if (this->Internals->TriangleUVBuffer)
  {
    vtkMetalMRC::AssignRetained(
        this->Internals->EdgeUVBuffer,
        this->Internals->TriangleUVBuffer);
  }
}
```

---

## Step 5: Add fallback creation before render-bundle recording

In `RenderPiece()`, before bundle validity checking, add:

```cpp
if (this->Internals->HasEdgeOverlay &&
    this->Internals->EdgeVertexCount > 0 &&
    !this->Internals->EdgeUVBuffer)
{
  size_t uvCount = static_cast<size_t>(this->Internals->EdgeVertexCount) * 2;
  id<MTLBuffer> buffer = CreateZeroBuffer(device, uvCount * sizeof(float));
  vtkMetalMRC::AssignConsumed(this->Internals->EdgeUVBuffer, buffer);
}
```

---

## Step 6: Bind edge UV buffer in `RebuildRenderBundle()`

Locate the edge overlay section:

```cpp
if (!peelPassActive && drawEdgeOverlay)
{
  recordPipeline(this->Internals->EdgePipeline);
  ...
}
```

After prop ID binding, add:

```cpp
if (this->Internals->EdgeUVBuffer)
{
  recordVBuf(this->Internals->EdgeUVBuffer, 0, 8);
}
else if (this->Internals->TriangleUVBuffer)
{
  // Less safe fallback; prefer EdgeUVBuffer.
  recordVBuf(this->Internals->TriangleUVBuffer, 0, 8);
}
```

Also bind clip planes for safety:

```cpp
if (this->Internals->ClipPlaneBuffer)
{
  recordVBuf(this->Internals->ClipPlaneBuffer, 0, 5);
}
```

So the edge overlay section includes:

```cpp
if (this->Internals->PropIdBuffer)
{
  recordVBuf(this->Internals->PropIdBuffer, 0, 7);
}

if (this->Internals->EdgeUVBuffer)
{
  recordVBuf(this->Internals->EdgeUVBuffer, 0, 8);
}

if (this->Internals->ClipPlaneBuffer)
{
  recordVBuf(this->Internals->ClipPlaneBuffer, 0, 5);
}

recordCull(MTLCullModeNone);
```

---

## Validation criteria

- Render a surface with edge visibility enabled.
- Enable Metal validation.
- Confirm no missing vertex buffer index 8 error.
- Confirm edges render correctly.
- Confirm textured surfaces with edges do not crash.

---

# P1-3. Ensure all shader-required buffers are bound even when optional

## Problem

Several shaders unconditionally read buffers:

For example, `vertex_main` expects:

```metal
constant float4* vertexColors [[buffer(3)]]
constant ClipPlaneUniforms& clipPlanes [[buffer(5)]]
constant uint* cellIds [[buffer(6)]]
constant uint& propId [[buffer(7)]]
constant float2* triangleUVs [[buffer(8)]]
```

Some C++ paths conditionally bind these only if non-null.

If a buffer is missing, Metal may validate or crash.

---

## Required binding policy

For any draw path that uses `vertex_main` and `fragment_main`, guarantee these bindings:

### Vertex stage

| Index | Purpose |
|---:|---|
| 0 | positions |
| 1 | normals |
| 2 | scene uniforms |
| 3 | vertex colors |
| 5 | clip planes |
| 6 | cell IDs |
| 7 | prop ID |
| 8 | UVs |

### Fragment stage

| Index | Purpose |
|---:|---|
| 0 | material |
| 1 | lights |
| 2 | scene |
| 3 | coincident offset |
| 5 | clip planes |
| 6 | primitive cell IDs, after P3 |
| texture 0 | actor texture or default texture |
| sampler 0 | actor sampler or default sampler |

---

## Implementation strategy

Instead of sprinkling conditionals everywhere, add a helper that creates fallback buffers before bundle rebuild.

---

## Step 1: Add a fallback-buffer helper method

Add a protected method declaration in `vtkMetalPolyDataMapper.h`:

```cpp
void EnsureRequiredBindingFallbacks(void* mtlDevice);
```

Implement in `vtkMetalPolyDataMapper.mm`:

```cpp
void vtkMetalPolyDataMapper::EnsureRequiredBindingFallbacks(void* mtlDevice)
{
  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;

  const vtkIdType vertexCount =
      this->Internals->VertexPositionBuffer
          ? static_cast<vtkIdType>(
                [this->Internals->VertexPositionBuffer length] /
                (3 * sizeof(float)))
          : 0;

  const vtkIdType edgeVertexCount = this->Internals->EdgeVertexCount;

  // Zero cell ID buffers
  if (vertexCount > 0)
  {
    if (!this->Internals->ZeroTriangleCellIdBuffer)
    {
      id<MTLBuffer> buffer =
          CreateZeroBuffer(device, static_cast<size_t>(vertexCount) * sizeof(uint32_t));
      vtkMetalMRC::AssignConsumed(this->Internals->ZeroTriangleCellIdBuffer, buffer);
    }

    if (!this->Internals->ZeroLineCellIdBuffer)
    {
      id<MTLBuffer> buffer =
          CreateZeroBuffer(device, static_cast<size_t>(vertexCount) * sizeof(uint32_t));
      vtkMetalMRC::AssignConsumed(this->Internals->ZeroLineCellIdBuffer, buffer);
    }

    if (!this->Internals->ZeroTriangleUVBuffer)
    {
      id<MTLBuffer> buffer =
          CreateZeroBuffer(device, static_cast<size_t>(vertexCount) * 2 * sizeof(float));
      vtkMetalMRC::AssignConsumed(this->Internals->ZeroTriangleUVBuffer, buffer);
    }
  }

  if (edgeVertexCount > 0)
  {
    if (!this->Internals->ZeroEdgeCellIdBuffer)
    {
      id<MTLBuffer> buffer =
          CreateZeroBuffer(device, static_cast<size_t>(edgeVertexCount) * sizeof(uint32_t));
      vtkMetalMRC::AssignConsumed(this->Internals->ZeroEdgeCellIdBuffer, buffer);
    }

    if (!this->Internals->ZeroEdgeUVBuffer)
    {
      id<MTLBuffer> buffer =
          CreateZeroBuffer(device, static_cast<size_t>(edgeVertexCount) * 2 * sizeof(float));
      vtkMetalMRC::AssignConsumed(this->Internals->ZeroEdgeUVBuffer, buffer);
    }
  }

  // Prop ID buffer should always exist if anything is drawable.
  if (!this->Internals->PropIdBuffer)
  {
    uint32_t propId = 0;
    id<MTLBuffer> buffer =
        [device newBufferWithBytes:&propId
                            length:sizeof(uint32_t)
                           options:MTLResourceStorageModeShared];

    vtkMetalMRC::AssignConsumed(this->Internals->PropIdBuffer, buffer);
  }

  // Clip plane buffer should always exist.
  if (!this->Internals->ClipPlaneBuffer)
  {
    float cp[28] = {};
    id<MTLBuffer> buffer =
        [device newBufferWithBytes:cp
                            length:sizeof(cp)
                           options:MTLResourceStorageModeShared];

    vtkMetalMRC::AssignConsumed(this->Internals->ClipPlaneBuffer, buffer);
  }
}
```

---

## Step 2: Call it before bundle rebuild

In `RenderPiece()`, after uniform updates and before bundle validity checking:

```cpp
this->EnsureRequiredBindingFallbacks((void*)device);
```

Place it near:

```cpp
this->UpdateClipPlaneUniforms((void*)device, act);
```

---

## Step 3: Prefer fallback buffers in `RebuildRenderBundle()`

For triangle path:

```cpp
if (this->Internals->TriangleCellIdBuffer)
{
  recordVBuf(this->Internals->TriangleCellIdBuffer, 0, 6);
}
else if (this->Internals->ZeroTriangleCellIdBuffer)
{
  recordVBuf(this->Internals->ZeroTriangleCellIdBuffer, 0, 6);
}
```

For UVs:

```cpp
if (this->Internals->TriangleUVBuffer)
{
  recordVBuf(this->Internals->TriangleUVBuffer, 0, 8);
}
else if (this->Internals->ZeroTriangleUVBuffer)
{
  recordVBuf(this->Internals->ZeroTriangleUVBuffer, 0, 8);
}
```

For standard lines:

```cpp
if (this->Internals->LineCellIdBuffer)
{
  recordVBuf(this->Internals->LineCellIdBuffer, 0, 6);
}
else if (this->Internals->ZeroLineCellIdBuffer)
{
  recordVBuf(this->Internals->ZeroLineCellIdBuffer, 0, 6);
}
```

For edges:

```cpp
if (this->Internals->EdgeCellIdBuffer)
{
  recordVBuf(this->Internals->EdgeCellIdBuffer, 0, 6);
}
else if (this->Internals->ZeroEdgeCellIdBuffer)
{
  recordVBuf(this->Internals->ZeroEdgeCellIdBuffer, 0, 6);
}

if (this->Internals->EdgeUVBuffer)
{
  recordVBuf(this->Internals->EdgeUVBuffer, 0, 8);
}
else if (this->Internals->ZeroEdgeUVBuffer)
{
  recordVBuf(this->Internals->ZeroEdgeUVBuffer, 0, 8);
}
```

---

## Validation criteria

- Metal validation should report no missing vertex/fragment buffer bindings.
- Rendering should not crash when optional arrays are absent.
- Edge overlay, lines, and triangles should all render with fallback data.

---

# P1-4. Fix depth-peeling render-bundle invalidation

## Problem

The render bundle records actual depth-peeling textures:

```cpp
recordFTex((__bridge id<MTLTexture>)peelRenWin->PeelFrontTexture, 1);
recordFTex((__bridge id<MTLTexture>)peelRenWin->PeelDepthTexture, 2);
```

But bundle validity only checks:

```cpp
this->Internals->BundlePeelMode == currentPeelMode
```

It does not check whether the peel textures changed.

If textures are recreated, for example after resize, the cached bundle may bind stale textures.

---

## Short-term fix: disable bundle reuse during peel passes

This is the safest immediate fix.

In `RenderPiece()`, locate bundle validity logic:

```cpp
bool bundleValid = ...;
```

Change execution to:

```cpp
const bool allowBundleCaching = (currentPeelMode == 0);

if (allowBundleCaching && bundleValid)
{
  this->ReplayRenderBundle((void*)encoder);
}
else
{
  this->RebuildRenderBundle((void*)encoder, ren, act);
  this->ReplayRenderBundle((void*)encoder);

  if (!allowBundleCaching)
  {
    this->Internals->Bundle.Valid = false;
  }
}
```

This prevents stale peel bundles from being reused.

---

## Longer-term fix: track peel texture identities

If you want bundle reuse during peeling, add fields:

```cpp
void* BundlePeelFrontTexture = nullptr;
void* BundlePeelDepthTexture = nullptr;
```

Reset them in `ReleaseBuffers()`.

In `RebuildRenderBundle()`, store:

```cpp
if (peelMode == 2)
{
  vtkMetalRenderWindow* peelRenWin =
      vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());

  this->Internals->BundlePeelFrontTexture =
      peelRenWin ? peelRenWin->PeelFrontTexture : nullptr;

  this->Internals->BundlePeelDepthTexture =
      peelRenWin ? peelRenWin->PeelDepthTexture : nullptr;
}
else
{
  this->Internals->BundlePeelFrontTexture = nullptr;
  this->Internals->BundlePeelDepthTexture = nullptr;
}
```

Then include them in `bundleValid`:

```cpp
this->Internals->BundlePeelFrontTexture == renWin->PeelFrontTexture &&
this->Internals->BundlePeelDepthTexture == renWin->PeelDepthTexture &&
```

For now, I recommend the short-term disable.

---

## Validation criteria

- Render translucent geometry with depth peeling.
- Resize window.
- Confirm no stale texture binding.
- Confirm peel passes continue to work after resize.
- Confirm no crash during repeated peel passes.

---

# P1-5. Add explicit function constants for volume pipelines

## Problem

The volume shaders declare function constants:

```metal
constant bool fc_shading [[function_constant(0)]];
constant bool fc_gradientOpacity [[function_constant(1)]];
constant bool fc_mask [[function_constant(2)]];
constant bool fc_minmax [[function_constant(3)]];
constant bool fc_normalTexture [[function_constant(4)]];
```

If these are not set during pipeline creation, pipeline creation may fail or behave unpredictably.

---

## Where to change

The volume mapper pipeline creation code is not fully included in the pasted files, but the rule is:

Any time you create a function for:

```metal
fragment_volume_main
fragment_volume_fullscreen_main
fragment_volume_grid_traversal_main
```

you must provide function constant values.

---

## Implementation

When creating the fragment function, use `MTLFunctionConstantValues`.

Example:

```objc
MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];

bool fcShading = useGradientShading;
bool fcGradientOpacity = useGradientOpacity;
bool fcMask = useMask;
bool fcMinMax = useMinMaxAcceleration;
bool fcNormalTexture = useNormalTexture;

[constants setConstantValue:&fcShading
                       type:MTLDataTypeBool
                    atIndex:0];

[constants setConstantValue:&fcGradientOpacity
                       type:MTLDataTypeBool
                    atIndex:1];

[constants setConstantValue:&fcMask
                       type:MTLDataTypeBool
                    atIndex:2];

[constants setConstantValue:&fcMinMax
                       type:MTLDataTypeBool
                    atIndex:3];

[constants setConstantValue:&fcNormalTexture
                       type:MTLDataTypeBool
                    atIndex:4];

NSError* error = nil;
id<MTLFunction> fragmentFunction =
    [library newFunctionWithName:@"fragment_volume_main"
                  constantValues:constants
                           error:&error];

[constants release];
```

Do this for every volume fragment entry point.

---

## Pipeline cache key

Because function constants create different pipeline variants, your pipeline cache key must include them.

Example bitmask:

```cpp
uint32_t variantMask = 0;

if (useGradientShading)     variantMask |= 1u << 0;
if (useGradientOpacity)     variantMask |= 1u << 1;
if (useMask)                variantMask |= 1u << 2;
if (useMinMaxAcceleration)  variantMask |= 1u << 3;
if (useNormalTexture)       variantMask |= 1u << 4;
```

Cache pipelines by:

```cpp
struct VolumePipelineKey
{
  int sampleCount;
  uint32_t variantMask;
  MTLPixelFormat colorFormat;
  MTLPixelFormat depthFormat;
};
```

---

## Validation criteria

- Volume rendering pipelines create successfully.
- Switching shading, gradient opacity, masking, min/max acceleration, or normal texture does not fail.
- No pipeline creation errors in Metal validation.

---

# 3. P2: Fix visual correctness

---

# P2-1. Include actor property MTime in geometry invalidation

## Problem

`BuildGeometryBuffers()` bakes colors and opacity into GPU buffers.

But `RenderPiece()` invalidates geometry using:

```cpp
vtkMTimeType scalarMTime = this->GetMTime();
```

plus lookup table MTime.

It does not include actor property changes such as:

- color
- opacity
- edge color
- vertex color
- scalar-related property state

As a result, changing actor color or opacity may not update baked vertex colors.

---

## Where to change

File:

```text
vtkMetalPolyDataMapper.mm
```

Function:

```cpp
RenderPiece(...)
```

---

## Short-term implementation

After:

```cpp
vtkMTimeType scalarMTime = this->GetMTime();
```

add:

```cpp
vtkProperty* prop = act ? act->GetProperty() : nullptr;
vtkMTimeType propertyMTime = prop ? prop->GetMTime() : 0;

scalarMTime = std::max(scalarMTime, propertyMTime);
```

So:

```cpp
vtkMTimeType scalarMTime = this->GetMTime();

if (this->GetLookupTable())
{
  scalarMTime = std::max(
      scalarMTime,
      static_cast<vtkMTimeType>(this->GetLookupTable()->GetMTime()));
}

vtkProperty* prop = act ? act->GetProperty() : nullptr;
vtkMTimeType propertyMTime = prop ? prop->GetMTime() : 0;

scalarMTime = std::max(scalarMTime, propertyMTime);
```

---

## Important note

Use **property MTime**, not actor MTime.

Actor MTime can change due to transform changes, which should not require geometry rebuilds.

Property MTime changes when visual properties change.

---

## Longer-term improvement

Property MTime may change for properties that do not affect geometry, such as point size or line width.

For better performance, track only relevant state:

```cpp
struct VisualPropertyCache
{
  double Color[3] = {1.0, 1.0, 1.0};
  double Opacity = 1.0;
  double Ambient = 0.0;
  double Diffuse = 1.0;
  double Specular = 0.0;
  double EdgeColor[3] = {0.0, 0.0, 0.0};
  bool EdgeVisibility = false;
  bool VertexVisibility = false;
  double VertexColor[3] = {1.0, 1.0, 1.0};
};
```

Then compare current values to cached values and rebuild only when relevant values change.

But for correctness, property MTime is an acceptable first fix.

---

## Validation criteria

- Change actor color without calling `Modified()` on the mapper.
- Confirm rendered color updates.
- Change actor opacity.
- Confirm transparency updates.
- Confirm no unnecessary rebuilds when only camera changes.

---

# P2-2. Fix non-indexed triangle normal usage

## Problem

In the non-indexed triangle path, when a normal array exists, the code does:

```cpp
normalArray->GetTuple(tri[0], nn);
fn[0] = (float)nn[0]; fn[1] = (float)nn[1]; fn[2] = (float)nn[2];
```

Then uses that same normal for all three vertices.

This is incorrect when per-vertex normals exist.

It can cause:

- flat-looking shading
- incorrect smooth shading
- visual artifacts on cell-colored surfaces with point normals

---

## Where to change

File:

```text
vtkMetalPolyDataMapper.mm
```

Function:

```cpp
BuildGeometryBuffers(...)
```

Section:

```cpp
// Non-indexed path: emit 3 unique vertices per triangle
```

---

## Implementation

Replace the normal logic.

Current approximate code:

```cpp
float fn[3] = { 0.0f, 1.0f, 0.0f };

if (normalArray)
{
  double nn[3];
  normalArray->GetTuple(tri[0], nn);
  fn[0] = (float)nn[0]; fn[1] = (float)nn[1]; fn[2] = (float)nn[2];
}
else
{
  // compute face normal
}

for (int j = 0; j < 3; ++j)
{
  ...
  normals.push_back(fn[0]);
  normals.push_back(fn[1]);
  normals.push_back(fn[2]);
  ...
}
```

Change to:

```cpp
float faceNormal[3] = { 0.0f, 1.0f, 0.0f };

if (!normalArray)
{
  float e1[3] = {
      static_cast<float>(p[1][0] - p[0][0]),
      static_cast<float>(p[1][1] - p[0][1]),
      static_cast<float>(p[1][2] - p[0][2])
  };

  float e2[3] = {
      static_cast<float>(p[2][0] - p[0][0]),
      static_cast<float>(p[2][1] - p[0][1]),
      static_cast<float>(p[2][2] - p[0][2])
  };

  float ne1 = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
  float ne2 = std::sqrt(e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]);

  if (ne1 > 1e-8f && ne2 > 1e-8f)
  {
    e1[0] /= ne1; e1[1] /= ne1; e1[2] /= ne1;
    e2[0] /= ne2; e2[1] /= ne2; e2[2] /= ne2;
  }

  faceNormal[0] = e1[1] * e2[2] - e1[2] * e2[1];
  faceNormal[1] = e1[2] * e2[0] - e1[0] * e2[2];
  faceNormal[2] = e1[0] * e2[1] - e1[1] * e2[0];

  float nn = std::sqrt(
      faceNormal[0] * faceNormal[0] +
      faceNormal[1] * faceNormal[1] +
      faceNormal[2] * faceNormal[2]);

  if (nn > 1e-8f)
  {
    faceNormal[0] /= nn;
    faceNormal[1] /= nn;
    faceNormal[2] /= nn;
  }
}

for (int j = 0; j < 3; ++j)
{
  positions.push_back(static_cast<float>(p[j][0]));
  positions.push_back(static_cast<float>(p[j][1]));
  positions.push_back(static_cast<float>(p[j][2]));

  if (normalArray)
  {
    double nn[3];
    normalArray->GetTuple(tri[j], nn);

    normals.push_back(static_cast<float>(nn[0]));
    normals.push_back(static_cast<float>(nn[1]));
    normals.push_back(static_cast<float>(nn[2]));
  }
  else
  {
    normals.push_back(faceNormal[0]);
    normals.push_back(faceNormal[1]);
    normals.push_back(faceNormal[2]);
  }

  emitSurfaceColor(polyCellIdx, mappedColors ? mappedColors->GetPointer(0) : nullptr);

  if (tcoordArray && tcoordArray->GetNumberOfTuples() > tri[j])
  {
    double uv[3];
    tcoordArray->GetTuple(tri[j], uv);
    triangleUVs.push_back(static_cast<float>(uv[0]));
    triangleUVs.push_back(static_cast<float>(uv[1]));
  }
  else
  {
    triangleUVs.push_back(0.0f);
    triangleUVs.push_back(0.0f);
  }

  triangleVertexCellIds.push_back(static_cast<uint32_t>(polyCellIdx) + 1u);

  emitExtraAttrsForPoint(tri[j]);
  emitExtraAttrsForCell(polyCellIdx);
}
```

---

## Validation criteria

- Load a smooth mesh with per-point normals.
- Enable cell coloring.
- Confirm smooth shading is preserved.
- Confirm no unintended flat shading.

---

# P2-3. Verify and correct texture Y flipping

## Problem

VTK image data typically uses a lower-left origin.

Metal texture coordinate `(0,0)` is top-left.

If texture upload does not flip Y, textures may appear vertically mirrored.

---

## Where to change

File:

```text
vtkMetalPolyDataMapper.mm
```

Function:

```cpp
UpdateActorTexture(...)
```

---

## Implementation

Current upload loop:

```cpp
for (int y = 0; y < height; ++y)
{
  for (int x = 0; x < width; ++x)
  {
    unsigned char* srcPtr =
        static_cast<unsigned char*>(image->GetScalarPointer(xMin + x, yMin + y, 0));

    int dstIdx = (y * width + x) * 4;
    ...
  }
}
```

Change to flip Y:

```cpp
for (int y = 0; y < height; ++y)
{
  int srcY = yMin + (height - 1 - y);

  for (int x = 0; x < width; ++x)
  {
    int srcX = xMin + x;

    unsigned char* srcPtr =
        static_cast<unsigned char*>(image->GetScalarPointer(srcX, srcY, 0));

    int dstIdx = (y * width + x) * 4;
    unsigned char* dst = rgbaData + dstIdx;

    switch (numComponents)
    {
      case 1:
        dst[0] = dst[1] = dst[2] = srcPtr[0];
        dst[3] = 255;
        break;

      case 2:
        dst[0] = dst[1] = dst[2] = srcPtr[0];
        dst[3] = srcPtr[1];
        break;

      case 3:
        dst[0] = srcPtr[0];
        dst[1] = srcPtr[1];
        dst[2] = srcPtr[2];
        dst[3] = 255;
        break;

      case 4:
        dst[0] = srcPtr[0];
        dst[1] = srcPtr[1];
        dst[2] = srcPtr[2];
        dst[3] = srcPtr[3];
        break;

      default:
        dst[0] = dst[1] = dst[2] = dst[3] = 255;
        break;
    }
  }
}
```

---

## Optional: make flipping configurable

If some texture sources are already top-left oriented, add an internal flag:

```cpp
bool FlipTextureY = true;
```

Then:

```cpp
int srcY = this->Internals->FlipTextureY
    ? (yMin + (height - 1 - y))
    : (yMin + y);
```

Default it to `true` unless testing proves otherwise.

---

## Validation criteria

- Render a textured quad with known orientation.
- Confirm text or asymmetric image is not vertically flipped.
- Compare against OpenGL/WebGPU backend if available.

---

# P2-4. Fix opacity-only override lighting behavior

## Problem

Currently, opacity-only overrides force:

```cpp
HasSurfaceColors = true;
```

Then the shader uses vertex color RGB instead of material RGB:

```metal
float3 ambientColor = hasVertexColors ? in.vertexColor.rgb : material.ambientColor.rgb;
float3 diffuseColor = hasVertexColors ? in.vertexColor.rgb : material.diffuseColor.rgb;
```

This can change lighting color when only opacity was supposed to change.

---

## Desired behavior

Separate two concepts:

1. **Use vertex RGB colors**
2. **Use vertex alpha / opacity**

---

## Step 1: Add a new scene flag

In the anonymous namespace in `vtkMetalPolyDataMapper.mm`, add:

```cpp
constexpr uint32_t VTK_METAL_SCENE_FLAG_HAS_SURFACE_ALPHA = 1u << 10;
```

Update the dynamic mask:

```cpp
constexpr uint32_t VTK_METAL_DYNAMIC_ACTOR_FLAG_MASK =
    VTK_METAL_SCENE_FLAG_VERTEX_VISIBILITY |
    VTK_METAL_SCENE_FLAG_SPHERE_POINTS |
    VTK_METAL_SCENE_FLAG_POINT_SHAPE |
    VTK_METAL_SCENE_FLAG_HAS_SURFACE_COLORS |
    VTK_METAL_SCENE_FLAG_HAS_ACTOR_TEXTURE |
    VTK_METAL_SCENE_FLAG_HAS_SURFACE_ALPHA;
```

---

## Step 2: Add `HasSurfaceAlpha`

Already added in Section 1.1.

---

## Step 3: Update color-buffer logic

In `BuildGeometryBuffers()`, locate:

```cpp
this->Internals->HasSurfaceColors =
    (mappedColors != nullptr) ||
    this->Internals->UseBatchColor ||
    this->Internals->UseBatchOpacity;
```

Replace with:

```cpp
this->Internals->HasSurfaceColors =
    (mappedColors != nullptr) ||
    this->Internals->UseBatchColor;

this->Internals->HasSurfaceAlpha =
    (mappedColors != nullptr) ||
    this->Internals->UseBatchColor ||
    this->Internals->UseBatchOpacity;
```

---

## Step 4: Update scene flags in `RenderPiece()`

Find:

```cpp
if (this->Internals->HasSurfaceColors)
{
  actorFlags |= VTK_METAL_SCENE_FLAG_HAS_SURFACE_COLORS;
}
```

Add:

```cpp
if (this->Internals->HasSurfaceAlpha)
{
  actorFlags |= VTK_METAL_SCENE_FLAG_HAS_SURFACE_ALPHA;
}
```

---

## Step 5: Update shaders

In `MetalShaders.metal`, update `fragment_main`.

Current:

```metal
bool hasVertexColors = (scene.flags & (1u << 8)) != 0u;

float3 ambientColor = hasVertexColors ? in.vertexColor.rgb : material.ambientColor.rgb;
float3 diffuseColor = hasVertexColors ? in.vertexColor.rgb : material.diffuseColor.rgb;

float baseOpacity = hasVertexColors ? in.vertexColor.a : material.opacity;
```

Change to:

```metal
bool hasVertexColors = (scene.flags & (1u << 8)) != 0u;
bool hasVertexAlpha  = (scene.flags & (1u << 10)) != 0u;

float3 ambientColor = hasVertexColors ? in.vertexColor.rgb : material.ambientColor.rgb;
float3 diffuseColor = hasVertexColors ? in.vertexColor.rgb : material.diffuseColor.rgb;

float baseOpacity = hasVertexAlpha ? in.vertexColor.a : material.opacity;
```

Apply the same change in:

```metal
fragment_peel
fragment_peel_alpha_blend
```

where similar logic exists.

---

## Step 6: Ensure alpha-only vertex buffers still exist

For opacity-only overrides, `HasSurfaceColors` may be false but `HasSurfaceAlpha` true.

You still need a `SurfaceColorBuffer` containing valid alpha.

The existing code already creates surface colors when batch opacity is active because `emitSurfaceColor` / `getOverrideOrDefaultRGBA` fills colors.

Just make sure the buffer creation condition does not depend only on `HasSurfaceColors`.

In `BuildGeometryBuffers()`, buffer creation already uses:

```cpp
if (!surfaceColors.empty())
{
  ...
}
```

That is fine.

---

## Validation criteria

- Create an actor with custom ambient/diffuse colors.
- Apply an opacity-only batch override.
- Confirm RGB lighting does not change.
- Confirm opacity changes.
- Confirm scalar-colored geometry still works.

---

# P2-5. Ensure 2D mapper pipelines match the active render pass depth format

## Problem

`vtkMetalPolyDataMapper2D` creates pipelines with:

```cpp
desc.depthAttachmentPixelFormat = MTLPixelFormatInvalid;
```

If the overlay is rendered into a render pass that has a depth attachment, this may be invalid.

---

## Where to change

File:

```text
vtkMetalPolyDataMapper2D.mm
```

Function:

```cpp
RenderOverlay(...)
```

---

## Short-term fix

Assume the overlay is rendered in the main depth-bearing pass.

Set:

```cpp
desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
```

for all three 2D pipelines:

- triangle
- line
- point

Example:

```cpp
desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
```

---

## Add an overlay depth-stencil state

2D overlays usually should draw on top.

Add to internals:

```cpp
id<MTLDepthStencilState> OverlayDepthState = nil;
```

Release in destructor / `ReleaseBuffers()`.

Create once:

```cpp
if (!this->Internals->OverlayDepthState)
{
  MTLDepthStencilDescriptor* dsDesc = [[MTLDepthStencilDescriptor alloc] init];
  dsDesc.depthCompareFunction = MTLCompareFunctionAlways;
  dsDesc.depthWriteEnabled = NO;

  this->Internals->OverlayDepthState =
      [device newDepthStencilStateWithDescriptor:dsDesc];

  [dsDesc release];
}
```

Before drawing:

```cpp
[encoder setDepthStencilState:this->Internals->OverlayDepthState];
```

---

## Longer-term fix

Add a render-window query:

```cpp
MTLPixelFormat GetCurrentDepthPixelFormat();
```

or:

```cpp
bool CurrentRenderPassHasDepth();
```

Then create 2D pipelines using the actual depth format.

If overlay passes can sometimes have no depth, use pipeline variants:

- depth format `Depth32Float`
- depth format `Invalid`

---

## Remove unused per-vertex color buffer

The 2D shader uses:

```metal
out.color = state.color;
```

The per-vertex `ColorBuffer` is bound but unused.

For now, remove it:

- remove `ColorBuffer`
- remove color-buffer creation
- remove `[encoder setVertexBuffer:ColorBuffer ...]`

This reduces confusion.

If per-vertex 2D colors are needed later, add a proper vertex attribute.

---

## Add point size support

The 2D shader currently ignores point size.

Update shader:

```metal
struct Vertex2DOut
{
  float4 position [[position]];
  float4 color;
  float pointSize [[point_size]];
};
```

Vertex shader:

```metal
vertex Vertex2DOut vertex_2d_main(
    Vertex2DIn in [[stage_in]],
    constant Mapper2DState& state [[buffer(1)]])
{
  Vertex2DOut out;
  out.position = state.wcvcMatrix * float4(in.position, 0.0, 1.0);
  out.color = state.color;
  out.pointSize = state.pointSize;
  return out;
}
```

Line width cannot be implemented natively in Metal for arbitrary widths. If wide 2D lines are required, reuse a thick-line expansion approach.

---

## Validation criteria

- 2D overlay renders without pipeline validation errors.
- 2D overlay appears on top when expected.
- 2D point size changes take effect.
- No unused buffer warnings.

---

# 4. P3: Fix picking correctness

---

# P3-1. Use a unified prop-ID space

## Problem

Currently:

- non-batched actors use `GetOrCreatePropId()`
- batched elements use `FlatIndex`

These ID spaces can collide.

---

## Desired design

Use one global picking-ID allocator for:

- actors
- batched blocks

Reserve `0` for background/no prop.

Shader convention remains:

- GPU buffer stores zero-based ID
- shader outputs `propId + 1`
- picking buffer value `0` means background

---

## Step 1: Add a global allocator

In `vtkMetalPolyDataMapper.mm`, add:

```cpp
#include <atomic>

namespace
{

std::atomic<uint32_t> VTK_METAL_NEXT_PICKING_ID{0};

uint32_t vtkMetalAllocatePickingId()
{
  uint32_t id = VTK_METAL_NEXT_PICKING_ID.fetch_add(1);

  if (id == UINT32_MAX)
  {
    vtkGenericWarningMacro(
        << "vtkMetalPolyDataMapper: picking ID space exhausted; "
        << "returning unpickable sentinel.");
    return UINT32_MAX;
  }

  return id;
}

} // namespace
```

---

## Step 2: Replace actor pointer map with actor information key

Remove the static pointer map in `GetOrCreatePropId()`.

Add includes:

```cpp
#include "vtkInformation.h"
#include "vtkInformationIntegerKey.h"
```

Define a key:

```cpp
namespace
{

vtkInformationIntegerKey* GetMetalPickingIdKey()
{
  static vtkInformationIntegerKey key(
      "VTK_METAL_PICKING_ID",
      "vtkMetalPolyDataMapper");

  return &key;
}

} // namespace
```

Implement:

```cpp
uint32_t vtkMetalPolyDataMapper::GetOrCreatePropId(vtkActor* act)
{
  if (!act)
  {
    return UINT32_MAX;
  }

  vtkInformation* info = act->GetInformation();
  if (!info)
  {
    return UINT32_MAX;
  }

  vtkInformationIntegerKey* key = GetMetalPickingIdKey();

  if (info->Has(key))
  {
    return static_cast<uint32_t>(info->Get(key));
  }

  uint32_t id = vtkMetalAllocatePickingId();

  if (id != UINT32_MAX)
  {
    info->Set(key, static_cast<int>(id));
  }

  return id;
}
```

This avoids stale pointer maps and pointer-reuse problems.

---

## Step 3: Allocate batched picking IDs from the same space

In `vtkMetalBatchedPolyDataMapper.h`, add:

```cpp
std::map<unsigned int, uint32_t> PickingIdsByFlatIndex;
```

Add helper declaration:

```cpp
uint32_t GetOrCreateBatchPickingId(unsigned int flatIndex);
```

Implement in `vtkMetalBatchedPolyDataMapper.mm`:

```cpp
uint32_t vtkMetalBatchedPolyDataMapper::GetOrCreateBatchPickingId(
    unsigned int flatIndex)
{
  auto it = this->PickingIdsByFlatIndex.find(flatIndex);
  if (it != this->PickingIdsByFlatIndex.end())
  {
    return it->second;
  }

  uint32_t id = vtkMetalAllocatePickingId();
  this->PickingIdsByFlatIndex[flatIndex] = id;

  return id;
}
```

You will need to expose `vtkMetalAllocatePickingId()` in a shared internal header or duplicate a small internal allocator.

Better: create:

```text
vtkMetalPickingIdAllocator.h
vtkMetalPickingIdAllocator.mm
```

with:

```cpp
uint32_t vtkMetalAllocatePickingId();
```

Then use it from both mapper classes.

---

## Step 4: Use batch picking ID instead of flat index

In `vtkMetalBatchedPolyDataMapper::RenderPiece()`, replace:

```cpp
mapper->SetOverridePropId(elem->FlatIndex);
```

with:

```cpp
mapper->SetOverridePropId(this->GetOrCreateBatchPickingId(elem->FlatIndex));
```

Full block:

```cpp
bool pickable = actorPickable && elem->Pickability;

if (pickable)
{
  uint32_t pickingId = this->GetOrCreateBatchPickingId(elem->FlatIndex);
  mapper->SetOverridePropId(pickingId);
}
else
{
  mapper->SetOverridePropIdToNone();
}
```

---

## Validation criteria

- Pick non-batched actors and batched blocks in the same scene.
- Confirm no ID collisions.
- Confirm background remains ID 0.
- Confirm deleting actors does not cause ID reuse hazards.

---

# P3-2. Avoid actor pointer reuse in prop-ID creation

This is solved by the actor information key approach above.

Do not use:

```cpp
static std::unordered_map<const vtkActor*, uint32_t> propIds;
```

That map:

- grows forever
- can suffer pointer reuse
- is not tied to actor lifetime

The information-key approach stores the ID with the actor itself.

---

# P3-3. Improve cell-ID accuracy for shared/indexed vertices

This is the most involved P3 item.

There are two possible implementations:

---

## Option A: Simple but slower — disable deduplication when accurate cell picking is needed

### Idea

If cell picking must be accurate, do not deduplicate triangle vertices.

Then each triangle has its own vertices and its own per-vertex cell ID.

### Where

In `BuildGeometryBuffers()`:

```cpp
bool useIndexBuffer = (cellFlag == 0) && normalArray;
```

Change to something like:

```cpp
bool useIndexBuffer =
    (cellFlag == 0) &&
    normalArray &&
    !this->Internals->RequireAccurateCellPicking;
```

Add a mapper flag:

```cpp
bool RequireAccurateCellPicking = false;
```

You can expose:

```cpp
void SetAccurateCellPicking(bool);
bool GetAccurateCellPicking() const;
```

Or automatically enable it during picking passes if the render window exposes that state.

### Pros

- simple
- correct cell IDs
- minimal shader changes

### Cons

- more vertices
- more memory
- lower performance

---

## Option B: Preferred — use `[[primitive_id]]`-based picking

This is the cleaner long-term solution.

---

# P3-4. Implement primitive-ID-based picking

## Overview

Instead of storing cell IDs per vertex, store cell IDs per primitive.

Then the fragment shader uses:

```metal
uint primitiveId [[primitive_id]]
```

to look up the correct cell ID.

This works for:

- triangles
- lines
- edge overlay lines

For thick lines, you already use `instance_id`, which is fine.

---

## Step 1: Add primitive cell-ID buffers

Already added in Section 1.1:

```cpp
id<MTLBuffer> TrianglePrimitiveCellIdBuffer = nil;
id<MTLBuffer> EdgePrimitiveCellIdBuffer = nil;
```

For lines, you can reuse:

```cpp
LineSegmentCellIdBuffer
```

as the primitive cell-ID buffer.

---

## Step 2: Populate triangle primitive cell IDs

### CPU path

In the polygon triangulation loop, add:

```cpp
std::vector<uint32_t> trianglePrimitiveCellIds;
```

For each emitted triangle, push one cell ID.

In the fan loop:

```cpp
for (vtkIdType i = 1; i < npts - 1; ++i)
{
  trianglePrimitiveCellIds.push_back(static_cast<uint32_t>(polyCellIdx) + 1u);

  vtkIdType tri[3] = { pts[0], pts[i], pts[i + 1] };
  ...
}
```

Do this for both indexed and non-indexed paths.

After geometry building:

```cpp
if (!trianglePrimitiveCellIds.empty())
{
  id<MTLBuffer> buffer =
      [device newBufferWithBytes:trianglePrimitiveCellIds.data()
                          length:trianglePrimitiveCellIds.size() * sizeof(uint32_t)
                         options:MTLResourceStorageModeShared];

  vtkMetalMRC::AssignConsumed(
      this->Internals->TrianglePrimitiveCellIdBuffer,
      buffer);
}
```

---

### GPU tessellation path

The compute kernel already produces per-triangle cell IDs:

```cpp
id<MTLBuffer> triCellIdBuf = ...;
vtkMetalMRC::AssignConsumed(this->Internals->TriangleCellIdBuffer, triCellIdBuf);
```

Later you replace `TriangleCellIdBuffer` with per-point cell IDs.

Change this so the original per-triangle buffer is preserved as the primitive buffer.

Before replacement:

```cpp
vtkMetalMRC::AssignRetained(
    this->Internals->TrianglePrimitiveCellIdBuffer,
    this->Internals->TriangleCellIdBuffer);
```

Then create per-point cell IDs as before.

Conceptually:

```cpp
// Keep per-triangle IDs for primitive-id picking.
vtkMetalMRC::AssignRetained(
    this->Internals->TrianglePrimitiveCellIdBuffer,
    this->Internals->TriangleCellIdBuffer);

// Then replace TriangleCellIdBuffer with per-point IDs for vertex shader compatibility.
vtkMetalMRC::AssignConsumed(this->Internals->TriangleCellIdBuffer, perPointCellIds);
```

---

## Step 3: Populate edge primitive cell IDs

### CPU edge overlay

Add:

```cpp
std::vector<uint32_t> edgePrimitiveCellIds;
```

In the edge emission loop:

```cpp
for (const auto& kv : uniqueEdges)
{
  vtkIdType a = kv.second.A;
  vtkIdType b = kv.second.B;
  uint32_t edgeCellId = kv.second.CellId;

  edgeIndices.push_back(addEdgeVertex(a, edgeCellId));
  edgeIndices.push_back(addEdgeVertex(b, edgeCellId));

  edgePrimitiveCellIds.push_back(
      static_cast<uint32_t>(edgeCellId + polyCellOffset) + 1u);
}
```

Upload:

```cpp
if (!edgePrimitiveCellIds.empty())
{
  id<MTLBuffer> buffer =
      [device newBufferWithBytes:edgePrimitiveCellIds.data()
                          length:edgePrimitiveCellIds.size() * sizeof(uint32_t)
                         options:MTLResourceStorageModeShared];

  vtkMetalMRC::AssignConsumed(this->Internals->EdgePrimitiveCellIdBuffer, buffer);
}
```

---

### GPU edge path

The compute kernel produces:

```cpp
edgeCellIdBuf
```

Currently you expand it to per-point IDs and release it.

Instead, keep it:

```cpp
vtkMetalMRC::AssignConsumed(this->Internals->EdgePrimitiveCellIdBuffer, edgeCellIdBuf);
```

Then do not release `edgeCellIdBuf` separately.

---

## Step 4: Use line segment cell IDs as primitive cell IDs

For standard 1px lines, the primitive count is:

```cpp
LineIndexCount / 2
```

You already have:

```cpp
LineSegmentCellIdBuffer
```

which is per segment.

Use that as the fragment-stage primitive cell-ID buffer.

---

## Step 5: Update shaders to use `primitive_id`

### Update `fragment_main`

Add:

```metal
fragment FragmentOutput fragment_main(
    VertexOut in [[stage_in]],
    uint primitiveId [[primitive_id]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(5)]],
    constant uint* primitiveCellIds [[buffer(6)]],
    texture2d<float> actorTexture [[texture(0)]],
    sampler actorSampler [[sampler(0)]])
```

Then replace:

```metal
out.ids = uint4(in.cellId, in.propId, 1u, 0u);
```

with:

```metal
uint cellId = primitiveCellIds[primitiveId];
out.ids = uint4(cellId, in.propId, 1u, 0u);
```

You may keep `in.cellId` in `VertexOut` for compatibility, but it is no longer used for picking.

---

### Update `fragment_edge_main`

Add:

```metal
fragment FragmentOutput fragment_edge_main(
    VertexOut in [[stage_in]],
    uint primitiveId [[primitive_id]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant float4& edgeColor [[buffer(4)]],
    constant uint* primitiveCellIds [[buffer(6)]])
```

Then:

```metal
uint cellId = primitiveCellIds[primitiveId];
out.ids = uint4(cellId, in.propId, 1u, 0u);
```

---

## Step 6: Bind primitive cell-ID buffers in `RebuildRenderBundle()`

### Triangle path

Add fragment binding:

```cpp
if (this->Internals->TrianglePrimitiveCellIdBuffer)
{
  recordFBuf(this->Internals->TrianglePrimitiveCellIdBuffer, 0, 6);
}
else if (this->Internals->TriangleCellIdBuffer)
{
  // Temporary fallback only; not correct for shared vertices.
  recordFBuf(this->Internals->TriangleCellIdBuffer, 0, 6);
}
```

Keep the vertex-stage binding as well:

```cpp
if (this->Internals->TriangleCellIdBuffer)
{
  recordVBuf(this->Internals->TriangleCellIdBuffer, 0, 6);
}
else if (this->Internals->ZeroTriangleCellIdBuffer)
{
  recordVBuf(this->Internals->ZeroTriangleCellIdBuffer, 0, 6);
}
```

---

### Standard line path

Add:

```cpp
if (this->Internals->LineSegmentCellIdBuffer)
{
  recordFBuf(this->Internals->LineSegmentCellIdBuffer, 0, 6);
}
```

Keep vertex binding:

```cpp
if (this->Internals->LineCellIdBuffer)
{
  recordVBuf(this->Internals->LineCellIdBuffer, 0, 6);
}
else if (this->Internals->ZeroLineCellIdBuffer)
{
  recordVBuf(this->Internals->ZeroLineCellIdBuffer, 0, 6);
}
```

---

### Edge overlay path

Add:

```cpp
if (this->Internals->EdgePrimitiveCellIdBuffer)
{
  recordFBuf(this->Internals->EdgePrimitiveCellIdBuffer, 0, 6);
}
```

Keep vertex binding:

```cpp
if (this->Internals->EdgeCellIdBuffer)
{
  recordVBuf(this->Internals->EdgeCellIdBuffer, 0, 6);
}
else if (this->Internals->ZeroEdgeCellIdBuffer)
{
  recordVBuf(this->Internals->ZeroEdgeCellIdBuffer, 0, 6);
}
```

---

## Step 7: Verify primitive counts

For triangles:

```cpp
TrianglePrimitiveCellIdBuffer length == TriangleIndexCount / 3
```

for indexed drawing, or:

```cpp
TrianglePrimitiveCellIdBuffer length == TriangleVertexCount / 3
```

for non-indexed drawing.

For lines:

```cpp
LineSegmentCellIdBuffer length == LineIndexCount / 2
```

For edges:

```cpp
EdgePrimitiveCellIdBuffer length == EdgeIndexCount / 2
```

Add debug assertions during development:

```cpp
VTK_ASSERT(static_cast<size_t>(this->Internals->TriangleIndexCount / 3) ==
           trianglePrimitiveCellIds.size());
```

---

## Validation criteria

- Pick a triangle on a shared vertex.
- Confirm the correct cell ID is returned.
- Pick lines and edges.
- Confirm correct cell IDs.
- Confirm picking still works with indexed and non-indexed geometry.
- Confirm GPU-tessellated geometry returns correct triangle cell IDs.

---

# 5. Suggested commit sequence

To keep changes reviewable, I recommend separate commits:

---

## Commit 1: P1 buffer-binding safety

- bind clip planes for lines
- add edge UV buffer
- add fallback buffers
- bind required buffers consistently

---

## Commit 2: P1 depth-peeling bundle safety

- disable bundle reuse during peel passes
- optionally track peel texture identities

---

## Commit 3: P1 volume function constants

- specialize volume fragment functions
- add pipeline variant cache key

---

## Commit 4: P2 visual correctness

- property MTime invalidation
- non-indexed normal fix
- texture Y flip
- opacity-only override flag

---

## Commit 5: P2 2D mapper correctness

- depth format
- overlay depth state
- remove unused color buffer
- point size support

---

## Commit 6: P3 unified picking IDs

- global picking allocator
- actor information key
- batched picking IDs

---

## Commit 7: P3 primitive-ID picking

- primitive cell-ID buffers
- shader `primitive_id`
- fragment-stage cell-ID bindings
- validation

---

# 6. Final acceptance checklist

## P1

- [ ] Metal validation reports no missing buffer bindings.
- [ ] Standard lines work with clipping planes.
- [ ] Edge overlay does not crash or validate due to missing UVs.
- [ ] All `vertex_main` paths bind buffers 0, 1, 2, 3, 5, 6, 7, 8.
- [ ] Depth-peeling passes do not reuse stale bundles.
- [ ] Volume pipelines explicitly set function constants.

## P2

- [ ] Changing actor color updates rendering without mapper modification.
- [ ] Changing actor opacity updates rendering.
- [ ] Cell-colored smooth meshes shade correctly.
- [ ] Textures are oriented correctly.
- [ ] Opacity-only overrides do not alter RGB lighting.
- [ ] 2D overlay pipelines match render pass depth format.
- [ ] 2D point size works.

## P3

- [ ] Picking IDs are unique across batched and non-batched props.
- [ ] Actor pointer reuse cannot cause wrong picking IDs.
- [ ] Triangle cell picking is correct for shared vertices.
- [ ] Line cell picking is correct.
- [ ] Edge cell picking is correct.
- [ ] Picking background remains ID 0.
- [ ] Shader outputs remain 1-based for prop and cell IDs.

---

# 7. Minimum viable implementation if time is constrained

If you need the smallest safe patch set, implement these first:

1. Bind clip planes in standard line path.
2. Create and bind edge UV buffer.
3. Disable render-bundle reuse during depth peeling.
4. Include property MTime in geometry invalidation.
5. Fix non-indexed triangle normals.
6. Replace actor pointer prop-ID map with actor information key.
7. Use unified picking IDs for batched elements.

Those seven changes remove the most dangerous correctness and stability issues.
