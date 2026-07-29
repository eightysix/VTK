Below is a detailed, MRC-specific implementation guide for the Priority 0 items from the review.

This guide assumes:

- Your Objective-C++ files are compiled **without ARC**, i.e. `-fno-objc-arc`.
- Metal objects returned from `new...` / `alloc/init` / `copy` methods are **+1 owned** and must be explicitly released.
- Assigning an Objective-C object pointer to `nil` does **not** release it.
- Convenience constructors such as `[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:...]` usually return **autoreleased** objects and should **not** be released unless retained.
- You should avoid `__bridge` casts in MRC. Use normal C-style or Objective-C casts:
  ```cpp
  id<MTLDevice> device = (id<MTLDevice>)mtlDevice;
  void* p = (void*)buffer;
  id<MTLBuffer> b = (id<MTLBuffer>)p;
  ```

I will refer to the main files as:

- `vtkMetalPolyDataMapper.h/.mm`
- `vtkMetalBatchedPolyDataMapper.h/.mm`
- `vtkMetalPolyDataMapper2D.h/.mm`
- `vtkMetalCompositePolyDataMapperDelegator.h/.mm`

---

# 0. Pre-work: Establish MRC ownership helpers

Before fixing individual features, introduce consistent MRC ownership helpers. This is essential because many later fixes involve creating/replacing Metal objects.

Add a small internal header or put these helpers at the top of each `.mm` file, for example:

```cpp
namespace vtkMetalMRC
{

inline void ReleaseAndNil(id& obj)
{
    if (obj)
    {
        [obj release];
        obj = nil;
    }
}

// Use when you have an existing retained object and want to store it.
// This retains src and releases dst.
template <typename T>
inline void AssignRetained(T& dst, T src)
{
    if (dst != src)
    {
        [src retain];
        [dst release];
        dst = src;
    }
}

// Use when src is already +1 owned, usually from new.../alloc/init/copy.
// This transfers ownership to dst without an extra retain.
template <typename T>
inline void AssignConsumed(T& dst, T src)
{
    if (dst != src)
    {
        [dst release];
        dst = src;
    }
    else
    {
        [src release];
    }
}

} // namespace vtkMetalMRC
```

Then use:

```cpp
id<MTLBuffer> buf = [device newBufferWithLength:size
                                        options:MTLResourceStorageModeShared];

vtkMetalMRC::AssignConsumed(this->Internals->SomeBuffer, buf);
```

For releasing:

```cpp
vtkMetalMRC::ReleaseAndNil(this->Internals->SomeBuffer);
```

For containers:

```cpp
for (auto& kv : this->Internals->ExtraAttributeBuffers)
{
    [kv.second release];
}
this->Internals->ExtraAttributeBuffers.clear();
```

---

# P0-1. Fix Objective-C/MRC ownership across all mappers

This is the foundational fix. Without this, later changes may leak or crash.

## 1.1. Remove ARC-style bridging casts

Replace:

```cpp
id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDevice;
id<MTLRenderCommandEncoder> encoder = (__bridge id<MTLRenderCommandEncoder>)mtlEncoder;
id<MTLBuffer> propBuffer = (__bridge id<MTLBuffer>)this->BatchPropertiesBuffer;
```

with MRC-style casts:

```cpp
id<MTLDevice> device = (id<MTLDevice>)mtlDevice;
id<MTLRenderCommandEncoder> encoder = (id<MTLRenderCommandEncoder>)mtlEncoder;
id<MTLBuffer> propBuffer = (id<MTLBuffer>)this->BatchPropertiesBuffer;
```

`__bridge` is an ARC concept. It may compile in MRC, but it obscures ownership.

---

## 1.2. Add a destructor to `vtkMetalPolyDataMapperInternals`

Your current `ReleaseBuffers()` assigns many Objective-C object pointers to `nil` without releasing them. Under MRC, that leaks.

Change the internals struct to release everything.

First, add a helper to release pipelines separately, because sample-count changes should release pipelines without necessarily releasing geometry.

Inside `vtkMetalPolyDataMapperInternals`:

```cpp
void ReleasePipelines()
{
    vtkMetalMRC::ReleaseAndNil(TrianglePipeline);
    vtkMetalMRC::ReleaseAndNil(LinePipeline);
    vtkMetalMRC::ReleaseAndNil(PointPipeline);
    vtkMetalMRC::ReleaseAndNil(PointShapedPipeline);
    vtkMetalMRC::ReleaseAndNil(EdgePipeline);
    vtkMetalMRC::ReleaseAndNil(ThickLinePipeline);
    vtkMetalMRC::ReleaseAndNil(RoundCapLinePipeline);
    vtkMetalMRC::ReleaseAndNil(MiterJoinLinePipeline);
    vtkMetalMRC::ReleaseAndNil(TriangleInitPeelPipeline);
    vtkMetalMRC::ReleaseAndNil(TrianglePeelPipeline);

    vtkMetalMRC::ReleaseAndNil(PolygonToTrianglePipeline);
    vtkMetalMRC::ReleaseAndNil(PolyLineToLinePipeline);
    vtkMetalMRC::ReleaseAndNil(PolygonEdgesToLinesPipeline);
    vtkMetalMRC::ReleaseAndNil(CellToPrimitivePipeline);
}
```

Then modify `ReleaseBuffers()`:

```cpp
void ReleaseBuffers()
{
    // Invalidate first so cached commands do not reference released objects.
    InvalidateRenderBundle();

    vtkMetalMRC::ReleaseAndNil(VertexPositionBuffer);
    vtkMetalMRC::ReleaseAndNil(VertexNormalBuffer);
    vtkMetalMRC::ReleaseAndNil(IndexBuffer);
    vtkMetalMRC::ReleaseAndNil(LineIndexBuffer);
    vtkMetalMRC::ReleaseAndNil(SurfaceColorBuffer);
    vtkMetalMRC::ReleaseAndNil(TriangleUVBuffer);

    vtkMetalMRC::ReleaseAndNil(ActorTexture);
    vtkMetalMRC::ReleaseAndNil(ActorSampler);

    // Default resources can be released here, but ideally they should live
    // longer than geometry. See performance notes later.
    vtkMetalMRC::ReleaseAndNil(DefaultTexture);
    vtkMetalMRC::ReleaseAndNil(DefaultSampler);

    CachedTextureMTime = 0;

    vtkMetalMRC::ReleaseAndNil(EdgeVertexPositionBuffer);
    vtkMetalMRC::ReleaseAndNil(EdgeVertexNormalBuffer);
    vtkMetalMRC::ReleaseAndNil(EdgeSurfaceColorBuffer);
    vtkMetalMRC::ReleaseAndNil(EdgeIndexBuffer);
    vtkMetalMRC::ReleaseAndNil(EdgeColorUniformBuffer);

    EdgeIndexCount = 0;
    EdgeVertexCount = 0;
    HasEdgeOverlay = false;

    vtkMetalMRC::ReleaseAndNil(ThickLineLineWidthBuffer);
    ThickLineSegmentCount = 0;

    vtkMetalMRC::ReleaseAndNil(MiterJoinSegmentCountBuffer);
    RoundCapLineSegmentCount = 0;
    MiterJoinLineSegmentCount = 0;

    vtkMetalMRC::ReleaseAndNil(PointPositionBuffer);
    vtkMetalMRC::ReleaseAndNil(PointNormalBuffer);
    vtkMetalMRC::ReleaseAndNil(PointColorBuffer);
    vtkMetalMRC::ReleaseAndNil(PointTangentBuffer);
    vtkMetalMRC::ReleaseAndNil(PointUVBuffer);
    vtkMetalMRC::ReleaseAndNil(PointColorUVBuffer);
    vtkMetalMRC::ReleaseAndNil(PointConnectivityBuffer);
    PointVertexCount = 0;

    vtkMetalMRC::ReleaseAndNil(SceneUniformBuffer);
    vtkMetalMRC::ReleaseAndNil(MaterialUniformBuffer);
    vtkMetalMRC::ReleaseAndNil(LightUniformBuffer);
    vtkMetalMRC::ReleaseAndNil(CoincidentOffsetBuffer);
    vtkMetalMRC::ReleaseAndNil(VertexColorBuffer);
    vtkMetalMRC::ReleaseAndNil(ClipPlaneBuffer);
    vtkMetalMRC::ReleaseAndNil(CellIdOffsetBuffer);

    vtkMetalMRC::ReleaseAndNil(TriangleCellIdBuffer);
    vtkMetalMRC::ReleaseAndNil(LineCellIdBuffer);
    vtkMetalMRC::ReleaseAndNil(PointCellIdBuffer);
    vtkMetalMRC::ReleaseAndNil(PropIdBuffer);
    vtkMetalMRC::ReleaseAndNil(PrimitiveToCellBuffer);

    TrianglePrimitiveCount = 0;
    LinePrimitiveCount = 0;

    vtkMetalMRC::ReleaseAndNil(TessOutputConnectivityBuffer);
    vtkMetalMRC::ReleaseAndNil(TessEdgeArrayBuffer);
    vtkMetalMRC::ReleaseAndNil(TessParamsBuffer);
    UseGPUTessellation = false;

    TriangleVertexCount = 0;
    TriangleIndexCount = 0;
    LineIndexCount = 0;
    HasTriangles = false;
    HasLines = false;

    CachedInputMTime = 0;
    CachedRepresentation = -1;
    CachedEdgeVisibility = false;
    CachedLineWidth = -1.0f;

    vtkMetalMRC::ReleaseAndNil(PeelUniformBuffer);

    for (auto& kv : ExtraAttributeBuffers)
    {
        [kv.second release];
    }
    ExtraAttributeBuffers.clear();
    ExtraAttributeComponentCounts.clear();

    ReleasePipelines();

    BundleGeometryMTime = 0;
    BundleRepresentation = -1;
    BundleEdgeVisibility = false;
    BundleLineWidth = -1.0f;
    BundleSampleCount = 0;
    BundlePeelMode = 0;
}
```

Add a destructor:

```cpp
~vtkMetalPolyDataMapperInternals()
{
    ReleaseBuffers();
}
```

---

## 1.3. Fix sample-count pipeline invalidation

Current code does:

```cpp
this->Internals->TrianglePipeline = nil;
this->Internals->LinePipeline = nil;
...
```

Under MRC, that leaks.

Replace with:

```cpp
if (currentSampleCount != this->Internals->CachedSampleCount)
{
    this->Internals->ReleasePipelines();
    this->Internals->CachedSampleCount = currentSampleCount;
}
```

---

## 1.4. Fix temporary Metal object ownership

You currently create many temporary buffers for compute dispatches:

```cpp
id<MTLBuffer> connBuf = [device newBufferWithBytes:...];
```

Under MRC, these leak unless released.

You have two good options.

### Option A: autorelease temporaries

If the code is inside an `@autoreleasepool`, which `RenderPiece()` is, you can do:

```cpp
id<MTLBuffer> connBuf = [[device newBufferWithBytes:polyConn.data()
                                             length:polyConn.size() * sizeof(uint32_t)
                                            options:MTLResourceStorageModeShared] autorelease];
```

Then you do not release it manually.

### Option B: release explicitly after command buffer commit

Example:

```cpp
id<MTLBuffer> connBuf = [device newBufferWithBytes:polyConn.data()
                                            length:polyConn.size() * sizeof(uint32_t)
                                           options:MTLResourceStorageModeShared];

...

[enc setBuffer:connBuf offset:0 atIndex:3];

...

[cmdBuf commit];
[cmdBuf waitUntilCompleted];

[connBuf release];
[offBuf release];
[primBuf release];
[lParamsBuf release];
```

For performance-critical code, explicit release is often clearer.

---

## 1.5. Stop creating command queues inline

This pattern leaks a command queue under MRC and is inefficient:

```cpp
id<MTLCommandBuffer> cmdBuf = [(__bridge id<MTLCommandQueue>)
    [device newCommandQueue] commandBuffer];
```

Add a cached queue to internals:

```cpp
id<MTLCommandQueue> ComputeQueue = nil;
```

Release it in `ReleaseBuffers()`:

```cpp
vtkMetalMRC::ReleaseAndNil(ComputeQueue);
```

Add helper:

```cpp
id<MTLCommandQueue> EnsureComputeQueue(id<MTLDevice> device)
{
    if (!this->ComputeQueue)
    {
        id<MTLCommandQueue> queue = [device newCommandQueue];
        vtkMetalMRC::AssignConsumed(this->ComputeQueue, queue);
    }
    return this->ComputeQueue;
}
```

Then use:

```cpp
id<MTLCommandQueue> queue = this->Internals->EnsureComputeQueue(device);
id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
```

Do not release `cmdBuf` if it is autoreleased. Do not release the queue except in `ReleaseBuffers()`.

---

## 1.6. Fix descriptor ownership

Audit all descriptor creation.

### `MTLSamplerDescriptor`

This is correct if using `alloc/init`:

```cpp
MTLSamplerDescriptor* sDesc = [[MTLSamplerDescriptor alloc] init];
...
this->Internals->DefaultSampler = [device newSamplerStateWithDescriptor:sDesc];
[sDesc release];
```

### `MTLTextureDescriptor`

This convenience constructor returns an autoreleased object:

```cpp
MTLTextureDescriptor* texDesc =
    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                       width:width
                                                      height:height
                                                   mipmapped:NO];
```

Do **not** release it:

```cpp
// Wrong under MRC if texDesc is autoreleased:
// [texDesc release];
```

Your current `UpdateActorTexture()` does:

```cpp
[texDesc release];
```

That may overrelease. Remove it unless you switch to:

```cpp
MTLTextureDescriptor* texDesc = [[MTLTextureDescriptor alloc] init];
...
[texDesc release];
```

### `MTLVertexDescriptor` and `MTLRenderPipelineDescriptor`

Your current code uses:

```cpp
MTLVertexDescriptor* vertexDesc = [[MTLVertexDescriptor alloc] init];
MTLRenderPipelineDescriptor* pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
```

Those must be released, and your code already does that in most places. Keep that pattern.

---

## 1.7. Fix `vtkMetalBatchedPolyDataMapper` buffer ownership

The batched mapper stores:

```cpp
void* BatchPropertiesBuffer = nullptr;
```

Under MRC, implement explicit helpers.

In `vtkMetalBatchedPolyDataMapper.mm`:

```cpp
void vtkMetalBatchedPolyDataMapper::ReleaseBatchPropertiesBuffer()
{
    if (this->BatchPropertiesBuffer)
    {
        [(id)this->BatchPropertiesBuffer release];
        this->BatchPropertiesBuffer = nullptr;
        this->BatchPropertiesBufferSize = 0;
    }
}
```

Add declaration in the private section of the header:

```cpp
void ReleaseBatchPropertiesBuffer();
void SetBatchPropertiesBufferConsumed(id<MTLBuffer> buffer);
```

But because the header should not contain Objective-C types, either:

### Preferred: use PIMPL for batched Metal resources

Replace `void* BatchPropertiesBuffer` with:

```cpp
struct vtkMetalBatchedPolyDataMapperInternals;
std::unique_ptr<vtkMetalBatchedPolyDataMapperInternals> BatchInternals;
```

Then store `id<MTLBuffer>` inside the `.mm` file.

### Tactical minimal fix: keep `void*`

If you want minimal changes, keep `void*`, but add:

```cpp
void vtkMetalBatchedPolyDataMapper::SetBatchPropertiesBufferConsumed(id<MTLBuffer> buffer)
{
    void* newPtr = (void*)buffer;

    if (this->BatchPropertiesBuffer != newPtr)
    {
        this->ReleaseBatchPropertiesBuffer();
        this->BatchPropertiesBuffer = newPtr;
        this->BatchPropertiesBufferSize = buffer ? (size_t)[buffer length] : 0;
    }
    else
    {
        [buffer release];
    }
}
```

Then in `UpdateBatchPropertiesBuffer()`:

```cpp
id<MTLBuffer> newBuf = [device newBufferWithLength:bufferSize
                                           options:MTLResourceStorageModeShared];

this->SetBatchPropertiesBufferConsumed(newBuf);
```

Do not manually release `newBuf` after that.

In `ClearBatchElements()`:

```cpp
this->ReleaseBatchPropertiesBuffer();
```

In `ReleaseGraphicsResources()`:

```cpp
this->ReleaseBatchPropertiesBuffer();
this->Superclass::ReleaseGraphicsResources(w);
```

In destructor:

```cpp
vtkMetalBatchedPolyDataMapper::~vtkMetalBatchedPolyDataMapper()
{
    this->ReleaseBatchPropertiesBuffer();
}
```

---

# P0-2. Fix the 2D mapper index-buffer bug

The 2D mapper currently does this:

```cpp
[encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                    indexCount:triIndices.size()
                     indexType:MTLIndexTypeUInt32
                   indexBuffer:this->Internals->PositionBuffer
             indexBufferOffset:0
                 instanceCount:1
                    baseVertex:0
                 baseInstance:0];
```

This is invalid because `PositionBuffer` contains `float2` positions, not `uint32_t` indices.

You must build real index buffers.

---

## 2.1. Add index buffers to 2D internals

In `vtkMetalPolyDataMapper2DInternals`:

```cpp
id<MTLBuffer> TriangleIndexBuffer = nil;
id<MTLBuffer> LineIndexBuffer = nil;
id<MTLBuffer> PointIndexBuffer = nil;

NSUInteger TriangleIndexCount = 0;
NSUInteger LineIndexCount = 0;
NSUInteger PointIndexCount = 0;
```

Update `ReleaseBuffers()`:

```cpp
void ReleaseBuffers()
{
    vtkMetalMRC::ReleaseAndNil(PositionBuffer);
    vtkMetalMRC::ReleaseAndNil(ColorBuffer);
    vtkMetalMRC::ReleaseAndNil(StateBuffer);

    vtkMetalMRC::ReleaseAndNil(TriangleIndexBuffer);
    vtkMetalMRC::ReleaseAndNil(LineIndexBuffer);
    vtkMetalMRC::ReleaseAndNil(PointIndexBuffer);

    TriangleIndexCount = 0;
    LineIndexCount = 0;
    PointIndexCount = 0;

    VertexCount = 0;
    HasTriangles = false;
    HasLines = false;
    HasPoints = false;
}
```

Add pipeline release helper:

```cpp
void ReleasePipelines()
{
    vtkMetalMRC::ReleaseAndNil(TrianglePipeline);
    vtkMetalMRC::ReleaseAndNil(LinePipeline);
    vtkMetalMRC::ReleaseAndNil(PointPipeline);
}
```

Destructor:

```cpp
~vtkMetalPolyDataMapper2DInternals()
{
    ReleaseBuffers();
    ReleasePipelines();
}
```

Update `ReleaseGraphicsResources()`:

```cpp
void vtkMetalPolyDataMapper2D::ReleaseGraphicsResources(vtkWindow* w)
{
    this->Internals->ReleaseBuffers();
    this->Internals->ReleasePipelines();
    this->Superclass::ReleaseGraphicsResources(w);
}
```

---

## 2.2. Rebuild pipelines when sample count changes

Current code releases buffers when sample count changes, but not pipelines.

Change:

```cpp
if (currentMTime != this->Internals->CachedInputMTime ||
    sampleCount != this->Internals->CachedSampleCount)
{
    this->Internals->ReleaseBuffers();
    this->Internals->ReleasePipelines();

    this->Internals->CachedInputMTime = currentMTime;
    this->Internals->CachedSampleCount = sampleCount;

    // rebuild geometry...
}
```

---

## 2.3. Build proper triangle, line, and point index buffers

Replace the current geometry upload section with something like this.

After transforming points and uploading `PositionBuffer`:

```cpp
std::vector<uint32_t> triIndices;
std::vector<uint32_t> lineIndices;
std::vector<uint32_t> pointIndices;

vtkCellArray* polys = input->GetPolys();
vtkCellArray* lines = input->GetLines();
vtkCellArray* verts = input->GetVerts();

if (polys)
{
    const vtkIdType* ids = nullptr;
    vtkIdType npts = 0;

    polys->InitTraversal();

    while (polys->GetNextCell(npts, ids))
    {
        if (npts < 3)
        {
            continue;
        }

        for (vtkIdType i = 1; i < npts - 1; ++i)
        {
            triIndices.push_back(static_cast<uint32_t>(ids[0]));
            triIndices.push_back(static_cast<uint32_t>(ids[i]));
            triIndices.push_back(static_cast<uint32_t>(ids[i + 1]));
        }
    }
}

if (lines)
{
    const vtkIdType* ids = nullptr;
    vtkIdType npts = 0;

    lines->InitTraversal();

    while (lines->GetNextCell(npts, ids))
    {
        if (npts < 2)
        {
            continue;
        }

        for (vtkIdType i = 0; i < npts - 1; ++i)
        {
            lineIndices.push_back(static_cast<uint32_t>(ids[i]));
            lineIndices.push_back(static_cast<uint32_t>(ids[i + 1]));
        }
    }
}

if (verts)
{
    const vtkIdType* ids = nullptr;
    vtkIdType npts = 0;

    verts->InitTraversal();

    while (verts->GetNextCell(npts, ids))
    {
        for (vtkIdType i = 0; i < npts; ++i)
        {
            pointIndices.push_back(static_cast<uint32_t>(ids[i]));
        }
    }
}

this->Internals->HasTriangles = !triIndices.empty();
this->Internals->HasLines = !lineIndices.empty();
this->Internals->HasPoints = !pointIndices.empty();

if (!triIndices.empty())
{
    id<MTLBuffer> buffer =
        [device newBufferWithBytes:triIndices.data()
                            length:triIndices.size() * sizeof(uint32_t)
                           options:MTLResourceStorageModeShared];

    vtkMetalMRC::AssignConsumed(this->Internals->TriangleIndexBuffer, buffer);
    this->Internals->TriangleIndexCount = triIndices.size();
}

if (!lineIndices.empty())
{
    id<MTLBuffer> buffer =
        [device newBufferWithBytes:lineIndices.data()
                            length:lineIndices.size() * sizeof(uint32_t)
                           options:MTLResourceStorageModeShared];

    vtkMetalMRC::AssignConsumed(this->Internals->LineIndexBuffer, buffer);
    this->Internals->LineIndexCount = lineIndices.size();
}

if (!pointIndices.empty())
{
    id<MTLBuffer> buffer =
        [device newBufferWithBytes:pointIndices.data()
                            length:pointIndices.size() * sizeof(uint32_t)
                           options:MTLResourceStorageModeShared];

    vtkMetalMRC::AssignConsumed(this->Internals->PointIndexBuffer, buffer);
    this->Internals->PointIndexCount = pointIndices.size();
}
```

---

## 2.4. Draw using the real index buffers

Replace the triangle draw code with:

```cpp
if (this->Internals->HasTriangles &&
    this->Internals->TrianglePipeline &&
    this->Internals->TriangleIndexBuffer &&
    this->Internals->TriangleIndexCount > 0)
{
    [encoder setRenderPipelineState:this->Internals->TrianglePipeline];

    [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                        indexCount:this->Internals->TriangleIndexCount
                         indexType:MTLIndexTypeUInt32
                       indexBuffer:this->Internals->TriangleIndexBuffer
                 indexBufferOffset:0];
}
```

Lines:

```cpp
if (this->Internals->HasLines &&
    this->Internals->LinePipeline &&
    this->Internals->LineIndexBuffer &&
    this->Internals->LineIndexCount > 0)
{
    [encoder setRenderPipelineState:this->Internals->LinePipeline];

    [encoder drawIndexedPrimitives:MTLPrimitiveTypeLine
                        indexCount:this->Internals->LineIndexCount
                         indexType:MTLIndexTypeUInt32
                       indexBuffer:this->Internals->LineIndexBuffer
                 indexBufferOffset:0];
}
```

Points:

```cpp
if (this->Internals->HasPoints &&
    this->Internals->PointPipeline &&
    this->Internals->PointIndexBuffer &&
    this->Internals->PointIndexCount > 0)
{
    [encoder setRenderPipelineState:this->Internals->PointPipeline];

    [encoder drawIndexedPrimitives:MTLPrimitiveTypePoint
                        indexCount:this->Internals->PointIndexCount
                         indexType:MTLIndexTypeUInt32
                       indexBuffer:this->Internals->PointIndexBuffer
                 indexBufferOffset:0];
}
```

Remove the old `vertexStart += cellSize` logic. It assumes contiguous identity connectivity and is incorrect for general VTK cells.

---

## 2.5. Fix viewport matrix calculation

Current code:

```cpp
float vpX = static_cast<float>(vp[0] * size[0]);
float vpY = static_cast<float>(vp[1] * size[1]);
float vpW = static_cast<float>(vp[2] * size[0]);
float vpH = static_cast<float>(vp[3] * size[1]);
```

This is wrong unless viewport is `[0, 0, 1, 1]`.

Use:

```cpp
float vpX = static_cast<float>(vp[0] * size[0]);
float vpY = static_cast<float>(vp[1] * size[1]);
float vpW = static_cast<float>((vp[2] - vp[0]) * size[0]);
float vpH = static_cast<float>((vp[3] - vp[1]) * size[1]);
```

Guard against zero size:

```cpp
if (size[0] <= 0 || size[1] <= 0 || vpW <= 0.0f || vpH <= 0.0f)
{
    return;
}
```

---

## 2.6. Enable blending for translucent 2D actors

For 2D overlays, you probably want alpha blending.

When creating the 2D pipelines:

```cpp
desc.colorAttachments[0].blendingEnabled = YES;
desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
```

---

# P0-3. Stop batched mapper from rebuilding geometry every frame

The current batched mapper does:

```cpp
this->SetInputData(batchElement->PolyData);
this->Superclass::RenderPiece(ren, act);
```

for every mesh every frame. That can cause full geometry rebuilds for every mesh every frame.

There are two ways to fix this:

1. **Tactical fix:** use one cached child `vtkMetalPolyDataMapper` per batch element.
2. **Correct long-term fix:** build combined vertex/index buffers and render from shared GPU resources.

For Priority 0, I recommend implementing the tactical fix first to stop the bleeding, then schedule the true batching refactor.

---

## 3.1. Tactical fix: child mapper cache

Add to `vtkMetalBatchedPolyDataMapper.h`:

```cpp
#include <vtkSmartPointer.h>

class vtkMetalPolyDataMapper;
```

Private members:

```cpp
std::map<std::uintptr_t, vtkSmartPointer<vtkMetalPolyDataMapper>> ChildMappers;

vtkSmartPointer<vtkMetalPolyDataMapper> GetChildMapper(vtkPolyData* polydata);
void ReleaseChildMappers(vtkWindow* w);
```

Implement:

```cpp
vtkSmartPointer<vtkMetalPolyDataMapper>
vtkMetalBatchedPolyDataMapper::GetChildMapper(vtkPolyData* polydata)
{
    auto address = reinterpret_cast<std::uintptr_t>(polydata);

    auto it = this->ChildMappers.find(address);
    if (it == this->ChildMappers.end())
    {
        vtkSmartPointer<vtkMetalPolyDataMapper> mapper =
            vtkSmartPointer<vtkMetalPolyDataMapper>::New();

        mapper->SetInputData(polydata);
        this->ChildMappers[address] = mapper;
        return mapper;
    }

    if (it->second->GetInputDataObject(0, 0) != polydata)
    {
        it->second->SetInputData(polydata);
    }

    return it->second;
}
```

Release helpers:

```cpp
void vtkMetalBatchedPolyDataMapper::ReleaseChildMappers(vtkWindow* w)
{
    for (auto& kv : this->ChildMappers)
    {
        if (kv.second)
        {
            kv.second->ReleaseGraphicsResources(w);
        }
    }

    this->ChildMappers.clear();
}
```

Update `ReleaseGraphicsResources()`:

```cpp
void vtkMetalBatchedPolyDataMapper::ReleaseGraphicsResources(vtkWindow* w)
{
    this->ReleaseChildMappers(w);
    this->ReleaseBatchPropertiesBuffer();

    this->ResourcesSyncTimeStamp = vtkTimeStamp();
    this->GeometryDirty = true;

    this->Superclass::ReleaseGraphicsResources(w);
}
```

Update `ClearBatchElements()`:

```cpp
void vtkMetalBatchedPolyDataMapper::ClearBatchElements()
{
    this->ReleaseChildMappers(nullptr);

    this->VTKPolyDataToBatchElement.clear();
    this->FlatIndexToPolyData.clear();

    this->ReleaseBatchPropertiesBuffer();

    this->GeometryDirty = true;
    this->Modified();
}
```

Update `ClearUnmarkedBatchElements()`:

```cpp
void vtkMetalBatchedPolyDataMapper::ClearUnmarkedBatchElements()
{
    bool changed = false;

    for (auto iter = this->VTKPolyDataToBatchElement.begin();
         iter != this->VTKPolyDataToBatchElement.end();)
    {
        if (!iter->second->Marked)
        {
            auto childIt = this->ChildMappers.find(iter->first);
            if (childIt != this->ChildMappers.end())
            {
                if (childIt->second)
                {
                    childIt->second->ReleaseGraphicsResources(nullptr);
                }
                this->ChildMappers.erase(childIt);
            }

            iter = this->VTKPolyDataToBatchElement.erase(iter);
            changed = true;
        }
        else
        {
            ++iter;
        }
    }

    if (changed)
    {
        this->GeometryDirty = true;
        this->Modified();
    }
}
```

Update `RenderPiece()`:

```cpp
void vtkMetalBatchedPolyDataMapper::RenderPiece(vtkRenderer* ren, vtkActor* act)
{
    vtkMetalRenderWindow* renWin =
        vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());

    if (!renWin || !renWin->GetMetalDevice())
    {
        return;
    }

    if (this->VTKPolyDataToBatchElement.empty())
    {
        return;
    }

    // Optional: keep batch properties updated.
    // This is still needed if you implement true batched shading/picking later.
    vtkMTimeType currentMTime = this->GetMTime();
    if (currentMTime != this->ResourcesSyncTimeStamp)
    {
        this->UpdateBatchPropertiesBuffer(renWin->GetMetalDevice());
    }

    // Render in flat-index order, not pointer-address order.
    std::vector<const BatchElement*> visible;
    visible.reserve(this->VTKPolyDataToBatchElement.size());

    for (const auto& kv : this->VTKPolyDataToBatchElement)
    {
        const BatchElement* elem = kv.second.get();
        if (elem && elem->Visibility && elem->PolyData)
        {
            visible.push_back(elem);
        }
    }

    std::sort(visible.begin(), visible.end(),
        [](const BatchElement* a, const BatchElement* b)
        {
            return a->FlatIndex < b->FlatIndex;
        });

    for (const BatchElement* elem : visible)
    {
        vtkSmartPointer<vtkMetalPolyDataMapper> mapper =
            this->GetChildMapper(elem->PolyData);

        if (!mapper)
        {
            continue;
        }

        mapper->RenderPiece(ren, act);
    }
}
```

This prevents changing the parent mapper input every frame and prevents the parent mapper from rebuilding geometry for each mesh.

### Important limitation

This tactical fix does **not** fully implement per-element property overrides. The child mapper still uses the passed `vtkActor` property. To fully support batch element opacity/color overrides, you need the true batching path or a per-mapper property mechanism.

---

## 3.2. Long-term correct fix: combined buffers

For production batched rendering, implement combined geometry buffers.

### Data structures

In batched internals:

```cpp
struct BatchDrawRecord
{
    unsigned int FlatIndex = 0;
    uint32_t PropertyIndex = 0;

    uint32_t TriangleIndexOffset = 0;
    uint32_t TriangleIndexCount = 0;

    uint32_t LineIndexOffset = 0;
    uint32_t LineIndexCount = 0;

    uint32_t EdgeIndexOffset = 0;
    uint32_t EdgeIndexCount = 0;
};
```

CPU accumulation arrays:

```cpp
std::vector<float> Positions;
std::vector<float> Normals;
std::vector<float> Colors;
std::vector<float> UVs;

std::vector<uint32_t> TriangleIndices;
std::vector<uint32_t> LineIndices;
std::vector<uint32_t> EdgeIndices;

std::vector<BatchDrawRecord> DrawRecords;
```

GPU buffers:

```cpp
id<MTLBuffer> CombinedPositionBuffer = nil;
id<MTLBuffer> CombinedNormalBuffer = nil;
id<MTLBuffer> CombinedColorBuffer = nil;
id<MTLBuffer> CombinedUVBuffer = nil;

id<MTLBuffer> CombinedTriangleIndexBuffer = nil;
id<MTLBuffer> CombinedLineIndexBuffer = nil;
id<MTLBuffer> CombinedEdgeIndexBuffer = nil;

id<MTLBuffer> BatchPropertiesBuffer = nil;
```

### Build process

When batch geometry is dirty:

1. Clear CPU arrays.
2. Sort visible batch elements by `FlatIndex`.
3. For each element:
   - Record starting offsets:
     ```cpp
     uint32_t vertexOffset = Positions.size() / 3;
     uint32_t triangleIndexOffset = TriangleIndices.size();
     uint32_t lineIndexOffset = LineIndices.size();
     uint32_t edgeIndexOffset = EdgeIndices.size();
     ```
   - Append geometry from that `vtkPolyData`.
   - When appending indices, add `vertexOffset`:
     ```cpp
     TriangleIndices.push_back(localIndex + vertexOffset);
     ```
   - Create a `BatchDrawRecord` with counts and offsets.
4. Upload combined buffers.
5. Build `BatchPropertiesBuffer` indexed by either:
   - flat index, or
   - packed draw index.

### Property buffer layout fix

Your current `CompositeDataProperties` may not match GPU layout because `float[3]` alignment differs from shader `float3`.

Use padded `float4` fields:

```cpp
struct CompositeDataProperties
{
    uint32_t ApplyOverrideColors = 0;
    float Opacity = 1.0f;
    uint32_t CompositeId = 0;
    uint32_t Pickable = 1;

    float Ambient[4] = { 1.0f, 1.0f, 1.0f, 0.0f };

    uint32_t CellIdOffsetForVerts = 0;
    uint32_t CellIdOffsetForLines = 0;
    uint32_t CellIdOffsetForPolys = 0;
    uint32_t CellIdOffsetForSelector = 0;

    float Diffuse[4] = { 1.0f, 1.0f, 1.0f, 0.0f };
};
```

Add:

```cpp
static_assert(sizeof(CompositeDataProperties) <= 256,
    "CompositeDataProperties must fit in aligned batch slot");
```

### Fix cell ID offsets

Use a single global cell offset:

```cpp
uint32_t globalCellOffset = 0;

for (each visible pd in flat-index order)
{
    uint32_t numVerts = static_cast<uint32_t>(pd->GetNumberOfVerts());
    uint32_t numLines = static_cast<uint32_t>(pd->GetNumberOfLines());
    uint32_t numPolys = static_cast<uint32_t>(pd->GetNumberOfPolys());
    uint32_t numStrips = static_cast<uint32_t>(pd->GetNumberOfStrips());

    props.CellIdOffsetForVerts = globalCellOffset;
    props.CellIdOffsetForLines = globalCellOffset + numVerts;
    props.CellIdOffsetForPolys = globalCellOffset + numVerts + numLines;
    props.CellIdOffsetForSelector = globalCellOffset;

    globalCellOffset += numVerts + numLines + numPolys + numStrips;
}
```

If invisible meshes should still occupy flat-index slots, allocate by `maxFlatIndex + 1` and write zeroed/non-pickable entries for invisible meshes.

### Rendering

For each draw record:

```cpp
[encoder setVertexBuffer:CombinedPositionBuffer offset:0 atIndex:0];
[encoder setVertexBuffer:CombinedNormalBuffer offset:0 atIndex:1];
[encoder setVertexBuffer:CombinedColorBuffer offset:0 atIndex:3];

[encoder setVertexBuffer:BatchPropertiesBuffer
                  offset:record.PropertyIndex * AlignedPropertiesSize
                 atIndex:PROP_BUFFER_INDEX];

[encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                    indexCount:record.TriangleIndexCount
                     indexType:MTLIndexTypeUInt32
                   indexBuffer:CombinedTriangleIndexBuffer
             indexBufferOffset:record.TriangleIndexOffset * sizeof(uint32_t)];
```

This gives you:

- no per-mesh input switching,
- no per-mesh geometry rebuild,
- stable picking offsets,
- a path toward indirect draws or instancing.

---

# P0-4. Fix line pipeline topology

In `EnsurePipelineStates()`, you currently create both triangle and line pipelines from the same descriptor with:

```cpp
pipelineDesc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
```

The line pipeline must use line topology.

Change to:

```cpp
pipelineDesc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;

if (!this->Internals->TrianglePipeline)
{
    error = nil;
    this->Internals->TrianglePipeline =
        [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];

    if (!this->Internals->TrianglePipeline)
    {
        vtkErrorMacro(<< "Triangle pipeline: "
                      << [[error localizedDescription] UTF8String]);
    }
}

pipelineDesc.inputPrimitiveTopology = MTLPrimitiveTopologyClassLine;

if (!this->Internals->LinePipeline)
{
    error = nil;
    this->Internals->LinePipeline =
        [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];

    if (!this->Internals->LinePipeline)
    {
        vtkErrorMacro(<< "Line pipeline: "
                      << [[error localizedDescription] UTF8String]);
    }
}
```

If the line rendering path eventually uses a different fragment shader, create a separate descriptor/functions for it. But at minimum, the topology class must be correct.

---

# P0-5. Fix edge overlay extraction

The current CPU edge overlay code emits fan diagonals, not polygon boundary edges.

You need a dedicated polygon-edge extraction helper.

---

## 5.1. Add edge key helper

```cpp
namespace
{

inline uint64_t MakeEdgeKey(vtkIdType a, vtkIdType b)
{
    if (a > b)
    {
        std::swap(a, b);
    }

    return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

} // namespace
```

---

## 5.2. Build unique polygon boundary edges

For surface edge visibility:

```cpp
std::unordered_map<uint64_t, std::pair<vtkIdType, vtkIdType>> uniqueEdges;

if (polys)
{
    const vtkIdType* pts = nullptr;
    vtkIdType npts = 0;

    polys->InitTraversal();

    while (polys->GetNextCell(npts, pts))
    {
        if (npts < 3)
        {
            continue;
        }

        for (vtkIdType i = 0; i < npts; ++i)
        {
            vtkIdType a = pts[i];
            vtkIdType b = pts[(i + 1) % npts];

            uint64_t key = MakeEdgeKey(a, b);

            if (uniqueEdges.find(key) == uniqueEdges.end())
            {
                uniqueEdges[key] = std::make_pair(a, b);
            }
        }
    }
}
```

Then build edge vertex buffers:

```cpp
std::vector<float> edgePositions;
std::vector<float> edgeNormals;
std::vector<float> edgeColors;
std::vector<uint32_t> edgeIndices;

std::unordered_map<vtkIdType, uint32_t> edgeVertexMap;

auto addEdgeVertex = [&](vtkIdType pointId) -> uint32_t
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

    // Edge color buffer is optional if fragment_edge_main uses a uniform.
    // If you keep per-vertex edge colors, push defaults or mapped colors here.
    edgeColors.push_back(1.0f);
    edgeColors.push_back(1.0f);
    edgeColors.push_back(1.0f);
    edgeColors.push_back(1.0f);

    return idx;
};

for (const auto& kv : uniqueEdges)
{
    vtkIdType a = kv.second.first;
    vtkIdType b = kv.second.second;

    edgeIndices.push_back(addEdgeVertex(a));
    edgeIndices.push_back(addEdgeVertex(b));
}
```

---

## 5.3. Always create an edge normal buffer if the edge pipeline expects one

Your edge pipeline vertex descriptor expects:

- buffer 0: position
- buffer 1: normal

If `edgeNormals` is empty, create a default normal buffer:

```cpp
if (!edgeIndices.empty() && !edgePositions.empty())
{
    id<MTLBuffer> posBuffer =
        [device newBufferWithBytes:edgePositions.data()
                            length:edgePositions.size() * sizeof(float)
                           options:MTLResourceStorageModeShared];

    vtkMetalMRC::AssignConsumed(this->Internals->EdgeVertexPositionBuffer, posBuffer);

    if (edgeNormals.empty())
    {
        edgeNormals.assign(edgePositions.size(), 0.0f);

        for (size_t i = 1; i < edgeNormals.size(); i += 3)
        {
            edgeNormals[i] = 1.0f;
        }
    }

    id<MTLBuffer> normalBuffer =
        [device newBufferWithBytes:edgeNormals.data()
                            length:edgeNormals.size() * sizeof(float)
                           options:MTLResourceStorageModeShared];

    vtkMetalMRC::AssignConsumed(this->Internals->EdgeVertexNormalBuffer, normalBuffer);

    if (!edgeColors.empty())
    {
        id<MTLBuffer> colorBuffer =
            [device newBufferWithBytes:edgeColors.data()
                                length:edgeColors.size() * sizeof(float)
                               options:MTLResourceStorageModeShared];

        vtkMetalMRC::AssignConsumed(this->Internals->EdgeSurfaceColorBuffer, colorBuffer);
    }

    id<MTLBuffer> indexBuffer =
        [device newBufferWithBytes:edgeIndices.data()
                            length:edgeIndices.size() * sizeof(uint32_t)
                           options:MTLResourceStorageModeShared];

    vtkMetalMRC::AssignConsumed(this->Internals->EdgeIndexBuffer, indexBuffer);

    this->Internals->EdgeIndexCount = edgeIndices.size();
    this->Internals->EdgeVertexCount = edgePositions.size() / 3;
    this->Internals->HasEdgeOverlay = true;
}
```

---

## 5.4. Use the same edge extraction for CPU wireframe

For `VTK_WIREFRAME`, do not use triangle fan edges. Use the same polygon boundary edge extraction as above, then write into `lineIndices` instead of `edgeIndices`.

---

# P0-6. Fix GPU tessellation fallback corruption

The current GPU tessellation path fills CPU vertex arrays before confirming that GPU tessellation succeeded. If GPU tessellation fails, the CPU path appends more vertices and corrupts the geometry.

---

## 6.1. Do not fill vertex arrays until GPU success

Restructure like this:

```cpp
bool gpuTessUsed = false;

if (useGPUTess)
{
    bool gpuOK = false;

    // Run compute tessellation first.
    // Do not append to positions/normals/colors yet.
    gpuOK = this->TryBuildGPUTriangleTessellation(device, polydata, polys);

    if (gpuOK)
    {
        // Now build per-point vertex arrays.
        this->BuildPerPointVertexArrays(
            device,
            polydata,
            normalArray,
            tcoordArray,
            mappedColors,
            cellFlag,
            positions,
            normals,
            surfaceColors,
            triangleUVs);

        gpuTessUsed = true;
    }
}
```

If GPU tessellation fails, leave the arrays empty and fall back to CPU.

---

## 6.2. Check command buffer status

After dispatch:

```cpp
[cmdBuf commit];
[cmdBuf waitUntilCompleted];

bool ok = (cmdBuf.status == MTLCommandBufferStatusCompleted);
```

If not completed successfully, treat as failure.

---

## 6.3. Release temporary compute buffers

Example:

```cpp
id<MTLBuffer> connBuf = [device newBufferWithBytes:polyConn.data()
                                            length:polyConn.size() * sizeof(uint32_t)
                                           options:MTLResourceStorageModeShared];

id<MTLBuffer> offBuf = [device newBufferWithBytes:polyOff.data()
                                           length:polyOff.size() * sizeof(uint32_t)
                                          options:MTLResourceStorageModeShared];

id<MTLBuffer> primBuf = [device newBufferWithBytes:polyPrimCounts.data()
                                            length:polyPrimCounts.size() * sizeof(uint32_t)
                                           options:MTLResourceStorageModeShared];

...

[cmdBuf commit];
[cmdBuf waitUntilCompleted];

[connBuf release];
[offBuf release];
[primBuf release];
```

---

## 6.4. Disable GPU wireframe fallback when explicit lines exist

The GPU wireframe path currently processes polygon edges but ignores explicit `vtkPolyData::GetLines()`.

For Priority 0, use this rule:

```cpp
bool hasExplicitLines = lines && lines->GetNumberOfCells() > 0;

if (representation == VTK_WIREFRAME && hasExplicitLines)
{
    useGPUTess = false;
}
```

Then CPU wireframe can correctly combine polygon edges and explicit lines.

Later, you can implement a GPU/CPU merged line buffer.

---

# P0-7. Fix missing edge color uniform creation

Current code:

```cpp
if (representation == VTK_SURFACE &&
    edgeVisibility &&
    this->Internals->HasEdgeOverlay &&
    this->Internals->EdgeColorUniformBuffer)
{
    this->UpdateEdgeColorUniform((__bridge void*)device, act);
}
```

This prevents `UpdateEdgeColorUniform()` from ever creating the buffer.

Change to:

```cpp
if (representation == VTK_SURFACE &&
    edgeVisibility &&
    this->Internals->HasEdgeOverlay)
{
    this->UpdateEdgeColorUniform((void*)device, act);
}
```

Also ensure this happens **before** render bundle rebuilding.

Inside `UpdateEdgeColorUniform()`, use MRC-safe creation:

```cpp
if (!this->Internals->EdgeColorUniformBuffer)
{
    id<MTLBuffer> buffer =
        [device newBufferWithLength:sizeof(ec)
                            options:MTLResourceStorageModeShared];

    vtkMetalMRC::AssignConsumed(this->Internals->EdgeColorUniformBuffer, buffer);
}

memcpy([this->Internals->EdgeColorUniformBuffer contents], ec, sizeof(ec));
```

---

# P0-8. Ensure point pipelines exist for vertex visibility

Current code only ensures point pipelines when representation is `VTK_POINTS`:

```cpp
if (representation == VTK_POINTS)
{
    this->EnsurePointPipelineStates((__bridge void*)device);
}
else
{
    this->EnsurePipelineStates((__bridge void*)device);
}
```

But vertex visibility on surfaces also draws points.

Replace with:

```cpp
bool needPointPipelines =
    (representation == VTK_POINTS) ||
    (act->GetProperty()->GetVertexVisibility() &&
     this->Internals->PointVertexCount > 0);

bool needSurfacePipelines =
    (representation != VTK_POINTS);

if (needSurfacePipelines)
{
    this->EnsurePipelineStates((void*)device);
}

if (needPointPipelines)
{
    this->EnsurePointPipelineStates((void*)device);
}
```

If representation is surface and vertex visibility is on, both triangle and point pipelines will be available.

---

# P0-9. Fix render bundle staleness

The render bundle currently tracks too little state. It can replay stale draw commands when actor state changes.

---

## 9.1. Add more bundle validity fields

In `vtkMetalPolyDataMapperInternals`, add:

```cpp
vtkMTimeType BundleTextureMTime = 0;
bool BundleHasActorTexture = false;

bool BundleVertexVisibility = false;
float BundlePointSize = -1.0f;
bool BundleRenderPointsAsSpheres = false;
int BundlePoint2DShape = -1;

int BundleLineJoin = -1;

bool BundleBackfaceCulling = false;
bool BundleFrontfaceCulling = false;

vtkMTimeType BundleExtraAttributesMTime = 0;
```

Reset them in `ReleaseBuffers()` and `InvalidateRenderBundle()` as appropriate.

---

## 9.2. Add extra attribute mapping MTime

In `vtkMetalPolyDataMapper.h`, protected section:

```cpp
vtkTimeStamp ExtraAttributesMTime;
```

In the internals, add:

```cpp
vtkMTimeType CachedExtraAttributesMTime = 0;
```

Update in mapping functions:

```cpp
void vtkMetalPolyDataMapper::MapDataArrayToVertexAttribute(...)
{
    ...
    this->ExtraAttributesMTime.Modified();
    this->Internals->InvalidateRenderBundle();
    this->Modified();
}
```

Same for:

```cpp
RemoveVertexAttributeMapping()
RemoveAllVertexAttributeMappings()
```

---

## 9.3. Rebuild geometry when extra attribute mappings change

In `RenderPiece()`:

```cpp
vtkMTimeType currentMTime = input->GetMTime();
vtkMTimeType extraMTime = this->ExtraAttributesMTime.GetMTime();

if (currentMTime != this->Internals->CachedInputMTime ||
    representation != this->Internals->CachedRepresentation ||
    edgeVisibility != this->Internals->CachedEdgeVisibility ||
    extraMTime != this->Internals->CachedExtraAttributesMTime)
{
    this->Internals->ReleaseBuffers();

    this->Internals->CachedInputMTime = currentMTime;
    this->Internals->CachedRepresentation = representation;
    this->Internals->CachedEdgeVisibility = edgeVisibility;
    this->Internals->CachedExtraAttributesMTime = extraMTime;

    this->BuildGeometryBuffers((void*)device, input, act);
}
```

---

## 9.4. Invalidate bundle when texture changes

In `UpdateActorTexture()`, detect changes and invalidate:

```cpp
vtkTexture* texture = actor->GetTexture();

if (!texture || !texture->GetInput())
{
    if (this->Internals->ActorTexture != nil ||
        this->Internals->ActorSampler != nil)
    {
        vtkMetalMRC::ReleaseAndNil(this->Internals->ActorTexture);
        vtkMetalMRC::ReleaseAndNil(this->Internals->ActorSampler);
        this->Internals->CachedTextureMTime = 0;
        this->Internals->InvalidateRenderBundle();
    }
    return;
}

vtkMTimeType texMTime = texture->GetMTime();

if (this->Internals->ActorTexture &&
    texMTime == this->Internals->CachedTextureMTime)
{
    return;
}

// Texture changed.
this->Internals->InvalidateRenderBundle();
this->Internals->CachedTextureMTime = texMTime;
```

Also include texture state in bundle validity:

```cpp
bool hasActorTexture = (this->Internals->ActorTexture != nil);
vtkMTimeType textureMTime = this->Internals->CachedTextureMTime;
```

---

## 9.5. Expand bundle validity check

In `RenderPiece()`:

```cpp
vtkProperty* prop = act->GetProperty();

int representation = prop->GetRepresentation();
bool edgeVisibility = prop->GetEdgeVisibility();
bool vertexVisibility = prop->GetVertexVisibility();
float lineWidth = static_cast<float>(prop->GetLineWidth());
float pointSize = static_cast<float>(prop->GetPointSize());
int lineJoin = prop->GetLineJoin();
bool backfaceCulling = prop->GetBackfaceCulling();
bool frontfaceCulling = prop->GetFrontfaceCulling();
bool renderPointsAsSpheres = prop->GetRenderPointsAsSpheres();
int point2DShape = static_cast<int>(prop->GetPoint2DShape());

bool hasActorTexture = (this->Internals->ActorTexture != nil);
vtkMTimeType textureMTime = this->Internals->CachedTextureMTime;
vtkMTimeType extraMTime = this->ExtraAttributesMTime.GetMTime();

int currentPeelMode = renWin->DepthPeelingMode;

bool bundleValid =
    this->Internals->Bundle.Valid &&
    this->Internals->BundleGeometryMTime == this->Internals->CachedInputMTime &&
    this->Internals->BundleRepresentation == representation &&
    this->Internals->BundleEdgeVisibility == edgeVisibility &&
    this->Internals->BundleLineWidth == lineWidth &&
    this->Internals->BundleSampleCount == currentSampleCount &&
    this->Internals->BundlePeelMode == currentPeelMode &&
    this->Internals->BundleTextureMTime == textureMTime &&
    this->Internals->BundleHasActorTexture == hasActorTexture &&
    this->Internals->BundleVertexVisibility == vertexVisibility &&
    this->Internals->BundlePointSize == pointSize &&
    this->Internals->BundleRenderPointsAsSpheres == renderPointsAsSpheres &&
    this->Internals->BundlePoint2DShape == point2DShape &&
    this->Internals->BundleLineJoin == lineJoin &&
    this->Internals->BundleBackfaceCulling == backfaceCulling &&
    this->Internals->BundleFrontfaceCulling == frontfaceCulling &&
    this->Internals->BundleExtraAttributesMTime == extraMTime;
```

Then after rebuilding:

```cpp
this->Internals->BundleGeometryMTime = this->Internals->CachedInputMTime;
this->Internals->BundleRepresentation = representation;
this->Internals->BundleEdgeVisibility = edgeVisibility;
this->Internals->BundleLineWidth = lineWidth;
this->Internals->BundleSampleCount = currentSampleCount;
this->Internals->BundlePeelMode = currentPeelMode;
this->Internals->BundleTextureMTime = textureMTime;
this->Internals->BundleHasActorTexture = hasActorTexture;
this->Internals->BundleVertexVisibility = vertexVisibility;
this->Internals->BundlePointSize = pointSize;
this->Internals->BundleRenderPointsAsSpheres = renderPointsAsSpheres;
this->Internals->BundlePoint2DShape = point2DShape;
this->Internals->BundleLineJoin = lineJoin;
this->Internals->BundleBackfaceCulling = backfaceCulling;
this->Internals->BundleFrontfaceCulling = frontfaceCulling;
this->Internals->BundleExtraAttributesMTime = extraMTime;
```

---

## 9.6. Bind extra attributes deterministically

Current code iterates:

```cpp
for (auto& eab : this->Internals->ExtraAttributeBuffers)
```

`ExtraAttributeBuffers` is an `unordered_map`, so binding order is nondeterministic.

Use the ordered `ExtraAttributes` map instead:

```cpp
NSUInteger extraIdx = 16;

for (const auto& attr : this->ExtraAttributes)
{
    auto it = this->Internals->ExtraAttributeBuffers.find(attr.first);
    if (it != this->Internals->ExtraAttributeBuffers.end() && it->second)
    {
        recordVBuf(it->second, 0, extraIdx);
        ++extraIdx;
    }
}
```

---

# P0-10. Fix line picking when both triangles and lines exist

Current code uses one shared `PrimitiveToCellBuffer` and skips line picking when triangles exist.

Fix by using separate buffers.

---

## 10.1. Add separate primitive-to-cell buffers

In internals:

```cpp
id<MTLBuffer> TrianglePrimitiveToCellBuffer = nil;
id<MTLBuffer> LinePrimitiveToCellBuffer = nil;
id<MTLBuffer> EdgePrimitiveToCellBuffer = nil;
```

Release in `ReleaseBuffers()`:

```cpp
vtkMetalMRC::ReleaseAndNil(TrianglePrimitiveToCellBuffer);
vtkMetalMRC::ReleaseAndNil(LinePrimitiveToCellBuffer);
vtkMetalMRC::ReleaseAndNil(EdgePrimitiveToCellBuffer);
```

You can keep or remove the old `PrimitiveToCellBuffer`, but separate buffers are clearer.

---

## 10.2. Upload triangle and line primitive-to-cell buffers separately

After geometry building:

```cpp
if (!trianglePrimToCell.empty())
{
    id<MTLBuffer> primToCell =
        [device newBufferWithBytes:trianglePrimToCell.data()
                            length:trianglePrimToCell.size() * sizeof(uint32_t)
                           options:MTLResourceStorageModeShared];

    vtkMetalMRC::AssignConsumed(this->Internals->TrianglePrimitiveToCellBuffer, primToCell);

    id<MTLBuffer> cellIdOut =
        [device newBufferWithLength:trianglePrimToCell.size() * sizeof(uint32_t)
                            options:MTLResourceStorageModeShared];

    vtkMetalMRC::AssignConsumed(this->Internals->TriangleCellIdBuffer, cellIdOut);

    this->Internals->TrianglePrimitiveCount = trianglePrimToCell.size();
}
```

Lines:

```cpp
if (!linePrimToCell.empty())
{
    id<MTLBuffer> primToCell =
        [device newBufferWithBytes:linePrimToCell.data()
                            length:linePrimToCell.size() * sizeof(uint32_t)
                           options:MTLResourceStorageModeShared];

    vtkMetalMRC::AssignConsumed(this->Internals->LinePrimitiveToCellBuffer, primToCell);

    id<MTLBuffer> cellIdOut =
        [device newBufferWithLength:linePrimToCell.size() * sizeof(uint32_t)
                            options:MTLResourceStorageModeShared];

    vtkMetalMRC::AssignConsumed(this->Internals->LineCellIdBuffer, cellIdOut);

    this->Internals->LinePrimitiveCount = linePrimToCell.size();
}
```

---

## 10.3. Dispatch both triangle and line cell-ID compute passes

Add a helper:

```cpp
void vtkMetalPolyDataMapper::DispatchCellToPrimitive(
    id<MTLDevice> device,
    id<MTLBuffer> outputBuffer,
    id<MTLBuffer> primitiveToCellBuffer,
    vtkIdType primitiveCount)
{
    if (!this->Internals->CellToPrimitivePipeline ||
        !outputBuffer ||
        !primitiveToCellBuffer ||
        primitiveCount <= 0)
    {
        return;
    }

    id<MTLCommandQueue> queue = this->Internals->EnsureComputeQueue(device);
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    [enc setComputePipelineState:this->Internals->CellToPrimitivePipeline];
    [enc setBuffer:outputBuffer offset:0 atIndex:0];
    [enc setBuffer:primitiveToCellBuffer offset:0 atIndex:1];
    [enc setBuffer:this->Internals->CellIdOffsetBuffer offset:0 atIndex:2];

    NSUInteger maxThreads =
        this->Internals->CellToPrimitivePipeline.maxTotalThreadsPerThreadgroup;

    NSUInteger threadgroupSize =
        std::min<NSUInteger>(maxThreads, static_cast<NSUInteger>(primitiveCount));

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(primitiveCount), 1, 1);
    MTLSize tg = MTLSizeMake(threadgroupSize, 1, 1);

    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}
```

Then call:

```cpp
this->DispatchCellToPrimitive(
    device,
    this->Internals->TriangleCellIdBuffer,
    this->Internals->TrianglePrimitiveToCellBuffer,
    this->Internals->TrianglePrimitiveCount);

this->DispatchCellToPrimitive(
    device,
    this->Internals->LineCellIdBuffer,
    this->Internals->LinePrimitiveToCellBuffer,
    this->Internals->LinePrimitiveCount);
```

Do not skip lines when triangles exist.

---

## 10.4. Add edge picking IDs

Edge overlay currently binds `LineCellIdBuffer`, which is not correct for separately built edge geometry.

Add:

```cpp
id<MTLBuffer> EdgeCellIdBuffer = nil;
```

Release it in `ReleaseBuffers()`.

For CPU edge extraction, build:

```cpp
std::vector<uint32_t> edgePrimToCell;
```

If you want edge picking to return the originating polygon cell ID, store that when extracting edges. For unique shared edges, you must decide which cell ID to report. Common choices:

- first adjacent cell,
- last adjacent cell,
- invalid ID for shared edges,
- separate edge ID space.

For Priority 0, first adjacent cell is simplest.

Then create:

```cpp
id<MTLBuffer> edgeCellIdBuffer =
    [device newBufferWithBytes:edgePrimToCell.data()
                        length:edgePrimToCell.size() * sizeof(uint32_t)
                       options:MTLResourceStorageModeShared];

vtkMetalMRC::AssignConsumed(this->Internals->EdgeCellIdBuffer, edgeCellIdBuffer);
```

In the edge overlay bundle section, bind `EdgeCellIdBuffer` instead of `LineCellIdBuffer`.

---

# Suggested implementation order

I recommend implementing the fixes in this order:

## Phase 1: MRC stability

1. Add MRC helpers.
2. Fix `vtkMetalPolyDataMapperInternals` release/destructor.
3. Fix `vtkMetalPolyDataMapper2DInternals` release/destructor.
4. Fix batched `BatchPropertiesBuffer` ownership.
5. Replace inline `newCommandQueue` with cached queue.
6. Audit temporary buffers and descriptors.

## Phase 2: Basic rendering correctness

1. Fix line pipeline topology.
2. Fix edge color uniform creation.
3. Ensure point pipelines for vertex visibility.
4. Fix render bundle staleness.
5. Fix edge overlay extraction.
6. Fix GPU tessellation fallback.

## Phase 3: Picking correctness

1. Add separate triangle/line primitive-to-cell buffers.
2. Dispatch both triangle and line cell ID passes.
3. Add edge cell ID buffer.
4. Update prop ID buffer if actor picking identity is available.

## Phase 4: 2D mapper

1. Add real index buffers.
2. Remove contiguous vertex-offset draws.
3. Fix viewport matrix.
4. Release pipelines on sample count change.
5. Enable blending.

## Phase 5: Batched mapper

1. Implement child mapper cache as tactical P0 fix.
2. Render in flat-index order.
3. Remove `SetInputData()` on the parent mapper during render.
4. Plan and implement true combined-buffer batching.
5. Fix `CompositeDataProperties` layout and cell ID offsets.
6. Bind batch properties buffer in shaders.

---

# Validation checklist

After implementing these Priority 0 items, test the following with Metal API validation enabled:

## MRC/memory

- Run Instruments Allocations and Leaks.
- Repeatedly toggle:
  - geometry visibility
  - representation
  - edge visibility
  - MSAA sample count
  - depth peeling mode
- Confirm no unbounded growth in Metal buffers/pipelines.

## Geometry

- Render a cube with quads.
- Render a concave polygon.
- Render polygons with holes if supported by your data.
- Toggle:
  - surface
  - wireframe
  - points
  - edge visibility
- Confirm edge overlay shows polygon boundaries, not fan diagonals.

## Lines

- Render explicit polyline cells with non-contiguous point IDs.
- Render line width 1 and line width > 1.
- Test line join modes.
- Confirm lines appear correctly.

## Points

- Test vertex visibility on surface representation.
- Test point size 1 and point size > 1.
- Test `VTK_POINTS` representation.

## Picking

- Render dataset with both triangles and lines.
- Pick triangles.
- Pick lines.
- Confirm both return correct cell IDs.
- Test batched picking after true batched implementation.

## 2D

- Render 2D annotations in a non-full viewport.
- Resize window.
- Test polygons, lines, and points.
- Confirm no GPU validation errors from indexed draws.

## Batched

- Insert many batch elements.
- Hide/show elements.
- Change flat indices.
- Confirm no per-frame geometry rebuild.
- Confirm stable render order by flat index.

---

# Summary of the most important MRC-specific rules

For every Metal object:

| Creation method | Ownership | Action |
|---|---:|---|
| `new...` | +1 | release or consume |
| `alloc/init` | +1 | release or consume |
| `copy...` | +1 | release or consume |
| convenience class method | autoreleased | do not release unless retained |
| object from window/device that you did not create | not owned | do not release |

For member assignment:

```cpp
id<MTLBuffer> obj = [device newBufferWithLength:size options:opts];
vtkMetalMRC::AssignConsumed(member, obj);
```

For cleanup:

```cpp
vtkMetalMRC::ReleaseAndNil(member);
```

For containers:

```cpp
for (auto& kv : container)
{
    [kv.second release];
}
container.clear();
```

If you apply the MRC ownership model consistently first, the remaining Priority 0 fixes become much safer to implement.
