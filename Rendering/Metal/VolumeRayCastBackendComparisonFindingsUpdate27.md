# Camera-inside: stage 1 of probe #1 — the cap-triangle vertex attributes are NOT bit-identical; the box-corner upload ORDER differs and produces different densify geometry (update 27)

**Date:** 2026-08-08
**Scope:** First stage of update-26 probe #1: byte-compare the cap-triangle vertex attributes uploaded by the two backends. Instrumented both `UpdateGeometry` camera-inside paths to dump the uploaded float32 vertices + indices, found a dump bug (`GetPointer` is value-indexed), fixed it, and got a decisive result: **the uploaded vertex arrays differ** — same 95 vertices / 126 indices, indices byte-identical, but the vertex *sets* differ (65 unique in each, only 22 in common; near-plane cap 17 vs 17, only 5 in common). Root-cause candidate identified by code inspection: **the boxSource corner ordering differs between the backends**, so the same `tris[36]` index list triangulates the box faces with different diagonals, and `vtkClipConvexPolyData` + `vtkDensifyPolyData` then produce different cap geometry.

**Follows:** [Update 26](VolumeRayCastBackendComparisonFindingsUpdate26.md) (probe #1: "Dump the cap-triangle vertex attributes … byte-compare").
**Status of the working tree:** two source files instrumented with `TEMP DEBUG` dumps (uncommitted); update-26's in-shader `P*V*M*v` Metal change still committed (still an unsuccessful fix, see section 6).

---

## 1. What was instrumented

Both backends now print, at the camera-inside proxy build, the exact float32 bytes uploaded for the cap mesh plus the index list:

- **GL** (`vtkOpenGLGPUVolumeRayCastMapper.cxx`, `RenderVolumeGeometry`): `GL_CAPMESH` (camera state), `GL_CAPVERTS` (count, data type, components), `GL_CAPVERT i 0x… 0x… 0x…` (per-vertex float32 bits), `GL_CAPINDICES n`, `GL_CAPINDEX i a b c`.
- **Metal** (`vtkMetalGPUVolumeRayCastMapper.mm`, `SetupBuffers` camera-inside branch): `MTL_CAPMESH` / `MTL_CAPVERTS` / `MTL_CAPVERT` / `MTL_CAPINDICES` / `MTL_CAPINDEX` with identical formatting.

Dumps repeat per geometry rebuild (once per frame while the camera is inside); blocks are paired by the camera state printed in the `*_CAPMESH` line. 95 verts / 126 cells in every block on both backends.

## 2. Debug-logging bug found and fixed (important)

The first GL dump used `vtkAOSDataArrayTemplate<float>::FastDownCast(points->GetData())->GetPointer(i)` with `i` the point index and read 3 floats — but **`vtkAOSDataArrayTemplate::GetPointer` is value-indexed, not tuple-indexed** (`vtkAOSDataArrayTemplate.txx:452`, returns `Array + valueIdx`). Point `i`'s tuple starts at value index `i*3`. The buggy read produced impossible coordinates (e.g. `(88.7, 77.26, 201.6)` with z beyond the box), in a cyclic-rotation pattern. Fixed to `GetPointer(0) + i*3` (same for the `polys`/index array). Metal's dump used a `std::vector<float>` stepped by 3 and was correct. **All numbers below are from the fixed dumps.**

## 3. Results of the byte-compare (fixed dumps, perturbed frame)

| item | GL | Metal | byte-identical? |
|---|---|---|---|
| near plane origin/normal (from `GL_NEARPLANE`/`METAL_NEARPLANE`) | `(-4.28864881,-4.28864881,24.1236496)` / `(-0.172412191,-0.172412191,0.969818577)` | same | **yes** |
| camera (pos/focal/viewAngle/clipRange) | same | same | **yes** |
| vertex count | 95 | 95 | yes |
| index list | 126 tris | 126 tris | **yes** |
| points data type | float32 | float32 (double→float roundtrip) | yes |
| **vertex arrays (in order)** | 95 | 95 | **no** |
| **vertex sets** | 65 unique | 65 unique | **no — 22 common, 43 GL-only, 43 MT-only** |
| near-plane (cap) points | 17 | 17 | **no — 5 common, 12 GL-only, 12 MT-only** |

So probe #1's first question is answered: **the vertex-attribute values uploaded by the two backends are not identical**, and the difference is not a last-bit rounding — the cap point *sets* are genuinely different.

## 4. What the differing points are (and what is shared)

The near-plane cap polygon is the same quadrilateral in both (corners, all four common):

- A `(0, 0, 25.6485)`, B `(0, 201.6, 61.4885)`, C `(201.6, 201.6, 97.3285)`, D `(201.6, 0, 61.4885)` — plus the 4 z=138 face corners and the z=138-face densify points are also common.

The **differing** points (43 each) are the `vtkDensifyPolyData` subdivision points of the clipped mesh:

- **GL-only** near-plane points sit on the cap edges at quarter/fractional positions, e.g. `(0, 50.61, 34.65)` (1/4 point of edge A–B), `(201.6, 88.73, 77.26)` (edge C–D region).
- **MT-only** near-plane points sit on interior cap diagonals, e.g. `(77.97, 77.97, 53.37)`, `(127.46, 126.29, 70.76)`.

Both sets of points are on the same plane, but they are **different points** — so at a given fragment pixel the two backends interpolate the anchor across **different triangles with different float32 vertex triples**.

## 5. Root-cause candidate: box-corner upload ORDER

Code inspection (no code change made this stage):

- **GL** box (`vtkOpenGLGPUVolumeRayCastMapper.cxx:1126`): `points->InsertNextPoint(geometry + i*3)` where `geometry = block->VolumeGeometry`, whose 8 corners are filled from `ijkCorners = {000, 100, 010, 110, 001, 101, 011, 111}` (`vtkVolumeTexture.cxx:1030-1043`).
- **Metal** box (`vtkMetalGPUVolumeRayCastMapper.mm:5444-5453`): an explicit literal with order `{000, 100, 110, 010, 001, 101, 111, 011}` — i.e. **corner pairs (2,3) and (6,7) are swapped relative to GL**.

Both backends feed the *same* `tris[36]` list into the same clip+densify pipeline. With different corner indexing, the box faces are triangulated with different diagonals (e.g. the z0 face diagonal is `100-010` in GL vs `100-110` in Metal). `vtkClipConvexPolyData` + `vtkDensifyPolyData` (2 subdivisions) then produce different edge/cap subdivision points — exactly the 43-vs-43 set difference observed.

**Why this yields the ~4e-5 anchor delta:** even though the cap polygon region and its planar function are the same, the float32 anchor at a pixel is a barycentric-weighted sum of the *containing triangle's* three vertex values. With different triangles (different vertex float32 triples), the two weighted sums round differently — ~1 ulp of a ~100-magnitude coordinate ≈ 1.2e-5, consistent with the measured 3–5e-5. This also explains why update 26's in-shader `P*V*M*v` change (making the clip matrix chain identical) changed nothing: the divergence is upstream, in the interpolated anchor itself.

## 6. Where this leaves the earlier work

- Update 23 (data-space anchor) and update 26 (in-shader P·V·M·v) are still structurally correct improvements but cannot remove the residual on their own: the anchor input differs at the mesh level.
- The update-26 `P*V*M*v` Metal change is **unnecessary for this bug** and should likely be reverted before the real fix lands (it is committed as an exploration; see update 26 §7).
- The two `TEMP DEBUG` dumps added this stage are temporary and should be removed before landing the real fix.

## 7. Next steps (stage 2)

1. **Fix candidate:** reorder Metal's `boxSource` corner literal to GL's `ijkCorners` order `{000,100,010,110,001,101,011,111}` (swap literal entries 2↔3 and 6↔7) and re-dump. Expected: vertex arrays and index list byte-identical; cap interpolation triangles identical.
2. Verify the two backends' *box double inputs* are also bit-identical (`block->VolumeGeometry` vs `ComputeVolumeBounds`/`ModelBounds`) — if not, the corner-order fix alone may not be enough.
3. Re-run the whole-image survey and the per-sample `compare_gl_metal_accum.py` at (422,92): residual knife-edge flips should drop to 0 if the anchor is bit-identical.
4. Re-baseline camera-outside to confirm no regression.
5. Remove the TEMP DEBUG dumps; decide on reverting update-26's in-shader change.

## 8. Reproduction

```sh
# GL capture (dummy-baseline trick) — new GL_CAPMESH/GL_CAPVERT/GL_CAPINDEX lines in stderr
python3 -c "from PIL import Image; Image.new('RGB',(512,512),(0,0,0)).save('/tmp/bc/u27/baseline.png')"
build_macos_metal/bin/vtkRenderingVolumeCxxTests \
  TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
  --vtk-factory-prefer RenderingBackend=OpenGL -D build_macos_metal/ExternalData/Testing \
  -T build_macos_metal/Testing/Temporary -V /tmp/bc/u27/baseline.png 2>/tmp/bc/u27/gl/out.log

# Metal capture
MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_BUFFER_SIZE=16777216 MTL_LOG_TO_STDERR=1 \
build_macos_metal/bin/vtkRenderingVolumeCxxTests \
  TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
  --vtk-factory-prefer RenderingBackend=Metal -D build_macos_metal/ExternalData/Testing \
  -T build_macos_metal/Testing/Temporary -V /tmp/bc/u27/baseline.png 2>/tmp/bc/u27/metal/out.log

# byte-compare (blocks paired by camera state; indices identical, vertex sets differ 22/65)
python3 /tmp/bc/u27/cmp_mesh.py
```

Renders for visual review: `/tmp/bc/u27/gl/img.png`, `/tmp/bc/u27/metal/img.png`, `/tmp/bc/u27/montage.png`, `/tmp/bc/u27/heatmap.png`. Visual result is near-identical (max |Δ| 22, 262 px ≥ 5), consistent with a sub-tolerance anchor-level difference rather than a gross mismatch.
