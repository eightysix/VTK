# Metal Backend — Feature Status & Progress

Track the implementation status of the VTK Metal rendering backend compared to the WebGPU reference implementation.

Last updated: 2026-07-11

---

## Completed Features

### Core Rendering
- **Triangle rendering** — CPU fan-triangulation in `BuildGeometryBuffers()`, `TrianglePipeline`
- **Line rendering** — CPU index buffer via `pointMap` deduplication, `LinePipeline`
- **Point rendering (1px)** — `PointPipeline` with `MTLPrimitiveTypePoint`
- **Point rendering (shaped)** — `PointShapedPipeline` with instanced triangle-strip quads (4 verts × N instances)
- **Phong lighting** — headlight, directional, point, spot lights; matches WebGPU backend
- **Material uniforms** — ambient, diffuse, specular, opacity, specular power
- **Light uniforms** — up to 8 lights with position, direction, color, attenuation

### P1 Features (Priority 1)
- **P1-4 Vertex visibility** — draws vertex dots on top of surface/wireframe when `GetVertexVisibility()` is true. Supports both 1px (`PointPipeline`) and shaped (`PointShapedPipeline`) based on point size.
- **P1-5 Coincident topology offset** — `CoincidentOffsetUniforms` buffer with polygon/line/point factors. Polygon offset uses `dfdx`/`dfdy` depth derivatives for slope-scale bias. Matches WebGPU behavior.
- **P1-6 Clipping planes** — up to 6 clip planes via `ClipPlaneUniforms`. Clip distances computed in vertex shader, fragments discarded in fragment shader.

### P2 Features (Priority 2)
- **P2-8 GPU picking IDs** — full single-pass picking pipeline mirroring WebGPU:
  - `IdsTexture` (RGBA32Uint, MTLStorageModeShared) on `vtkMetalRenderWindow`
  - Attached as `colorAttachments[1]` in render pass, cleared to zeros
  - `cellToPrimitive` compute kernel maps primitives → cell IDs
  - Flat-interpolated `cellId` and `propId` passed through all vertex/fragment shaders
  - `vtkMetalHardwareSelector` for single-pass capture + readback
  - `GetIdsData()` reads back texture with Y-flip

---

## Partially Implemented

### P2-7 Homogeneous Cell ID Offset
- `CellIdOffsetBuffer` exists and is bound to shaders
- Hardcoded to 0 for single-actor rendering
- Infrastructure for batched rendering cell ID offsetting is in place but unused

### P2-9 Tangent in View Space
- Tangent buffers created from polydata `GetTangents()` or default (1,0,0)
- Passed through point vertex shaders (`PointVertexOut.tangent`)
- Not consumed by fragment shaders for lighting calculations

### P2-10 Texture Coordinates
- UV buffers created from `pd->GetTCoords()`
- Passed through point vertex shaders (`PointVertexOut.uv`, `PointVertexOut.lut_uv`)
- Never sampled in fragment shaders — no `texture2d` or sampler declarations exist

---

## Not Implemented

### Per-Vertex Color for Surfaces
- `MapScalars()` is called but colors are only used for the point rendering path
- Triangle/line vertex descriptor only has position + normal (no color attribute)
- Fragment shader always uses `material.diffuseColor.rgb` — no per-vertex color input
- **Impact**: Surfaces cannot be colored by scalar arrays

### Cell Data Coloring
- `cellFlag != 0` from `MapScalars()` causes colors to be discarded
- No `CellColorBuffer` exists in the mapper internals
- **Impact**: Cell-based scalar arrays produce no visible coloring

### Wireframe / Edge Rendering
- No `VTK_WIREFRAME` representation code path
- Line rendering only handles explicit line cells from `GetLines()`
- No triangle-to-edge conversion for wireframe overlay
- **Impact**: `SetRepresentationToWireframe()` has no effect on surfaces

### Triangle Index Buffers
- `IndexBuffer` field exists but is never populated for polygon geometry
- `TriangleIndexCount` is always 0 — triangles drawn non-indexed
- Every triangle emits 3 unique vertices (shared vertices are duplicated)
- **Impact**: Higher memory usage and bandwidth vs indexed rendering

### MSAA (Multisampling)
- No `sampleCount` set on pipeline descriptors or textures
- No `MTLMultisampleStateDescriptor` usage
- No resolve texture or multisample texture creation
- **Impact**: No anti-aliasing — edges appear jagged

### Depth Peeling / Correct Translucency
- Renderer calls `UpdateTranslucentPolygonalGeometry()` (back-to-front sort)
- No multi-pass depth peeling, no stencil buffer, no peel layers
- Depth stencil state is simple `Less` with write enabled
- No alpha blending configured on pipeline descriptors
- **Impact**: Translucent geometry rendering is incorrect (painter's algorithm only)

### Thick Lines
- WebGPU has 4 line pipeline variants: 1px, thick no-join, round-cap round-join, miter-join
- Metal has single `MTLPrimitiveTypeLine` pipeline
- **Impact**: Lines are always 1px regardless of line width setting

### Batched Rendering
- No `vtkMetalBatchedPolyDataMapper` equivalent
- Each actor renders independently with its own buffers and draw calls
- `CellIdOffsetBuffer` placeholder exists but is always 0
- **Impact**: Higher CPU overhead for scenes with many actors

### Glyph3D Mapper
- No `vtkMetalGlyph3DMapper` equivalent
- WebGPU has `vtkWebGPUGlyph3DMapper` with instanced glyph rendering

### 2D Mapper
- No `vtkMetalPolyDataMapper2D` equivalent
- WebGPU has `vtkWebGPUPolyDataMapper2D` for 2D annotations/overlays

### GPU Tessellation
- WebGPU uses `CellToPrimitiveConverter` compute shader for polygon → triangle mapping
- Metal does CPU fan-triangulation in `BuildGeometryBuffers()`
- WebGPU compute also produces edge arrays for wireframe; Metal has no equivalent
- **Impact**: CPU-bound tessellation, no edge data for wireframe

### Actor Integration
- `vtkMetalActor` is a thin passthrough (calls `mapper->Render()`)
- `vtkWebGPUActor` manages property state, wireframe mode, texture binding
- All property queries done directly by mapper via `act->GetProperty()`

---

## File Inventory

| File | Lines | Role |
|------|-------|------|
| `vtkMetalRenderWindow.h/.mm` | 143/445 | Device, layer, depth/IDs textures, readback |
| `vtkMetalRenderer.mm` | 182 | Render pass setup, encoder, camera |
| `vtkMetalPolyDataMapper.mm` | 1651 | Geometry build, pipeline states, draw calls |
| `Shaders/MetalShaders.metal` | 540 | Vertex/fragment/compute shaders |
| `vtkMetalActor.mm` | 43 | Thin passthrough to mapper |
| `vtkMetalCamera.mm` | — | Scene transforms, view/projection matrices |
| `vtkMetalProperty.mm` | — | Property state (color, opacity, representation) |
| `vtkMetalLight.mm` | — | Light state |
| `vtkMetalHardwareSelector.h/.mm` | 83/184 | Single-pass GPU picking |
| `vtkIOSMetalRenderWindow.h/.mm` | — | iOS-specific window (UIKit integration) |
| `CMakeLists.txt` | 67 | Build configuration |

---

## Recommended Priority Order for Closing Gaps

See `IMPLEMENTATION_PLAN.md` for the full feature-by-feature implementation plan with file locations, code changes, and WebGPU references.

1. **Per-vertex color for surfaces** — most visible gap; surfaces can't be colored by scalar arrays
2. **Cell data coloring** — needed for cell-based scalar visualization
3. **Wireframe rendering** — needed for `VTK_WIREFRAME` representation
4. **Triangle index buffers** — reduces memory and improves cache efficiency
5. **MSAA** — straightforward Metal feature, high visual impact
6. **Depth peeling** — needed for correct translucent rendering
7. **GPU tessellation** — moves polygon→triangle conversion off CPU
8. **Thick lines** — needed for line width settings
9. **Batched rendering** — reduces CPU overhead for many-actor scenes
10. **Texture mapping** — complete the P2-10 UV plumbing with actual sampling
