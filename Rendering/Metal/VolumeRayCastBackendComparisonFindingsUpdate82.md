# CamOutside 1384 → 250: densify the camera-outside proxy box to GL's exact vtkDensifyPolyData(2) geometry and upload it in dataset space (update 82)

**Date:** 2026-08-12
**Status:** **Major CamOutside improvement, geometry now matches GL bit-for-bit.**
The remaining 250-px diff (vs the 178-px camera-inside floor) is the
rasterizer's ~1 ulp interpolator rounding on the densified box triangles —
the same hardware residual as the near-plane cap, amplified because the box
triangles are larger. No regression on any other suite test (camera-inside
floor unchanged at 178; FlatTF 0; StepTF 0–181 unchanged).

## 1. Root cause

GL renders its proxy box **densified** even when the camera is outside:

```
vtkOpenGLGPUVolumeRayCastMapper.cxx:1272-1276
  densifyPolyData->SetInputData(boxSource);          // camera outside (no clip)
  densifyPolyData->SetNumberOfSubdivisions(2);
```

`vtkDensifyPolyData(2)` is a **centroid fan**: each of the 12 box triangles is
split into 3 at its centroid `(v0+v1+v2)/3` (double precision), and each of
those 3 again — 12 × 9 = **108 triangles, 56 vertices** (8 corners + 12 level-1
centroids + 36 level-2 centroids). GL uploads those double positions as float32
(`in_vertexPos`) and the rasterizer interpolates the per-vertex
`ip_textureCoords`.

Metal's camera-outside proxy was a **coarse 12-triangle box** with a different
face triangulation (different face diagonals). Over the large triangles the
Metal interpolator rounds the ray anchor up to **~28 ulps** off GL's
small-triangle value (measured: x/y up to 1.7e-6, z identical), and with the
nearest-neighbor resampling + sharp TF step at scalar 500 every small anchor
shift flips knife-edge samples (1384 px).

## 2. Fix (two steps, both required)

**Step 1 — densify to GL's exact geometry.** The camera-outside box is now
built as a `vtkPolyData` with the 8 corners in GL's `DataGeometry` order
`{000,100,010,110,001,101,011,111}`, GL's `tris[36]` set, and GL's 0-2-1
winding swap, then run through the identical `vtkDensifyPolyData` filter with
`SetNumberOfSubdivisions(2)`. Result uploaded as float32 positions + uint32
indices (108 triangles) — byte-identical to GL's `BBoxPolyData` VBO.

**Step 2 — upload in dataset space, not the unit cube.** With the unit-cube
convention the vertex shader computed `modelPos = boundsMin + in.position *
(boundsMax - boundsMin)`, which rounds each centroid twice (the float32 `size`
subtraction and the multiply) and lands ~1 ulp off GL's single
`float32(double centroid)`. Combined with the densification this left 880 px.
So the camera-outside box now uploads **model-space positions** exactly like
GL (and like the camera-inside cap already did), and a new uniform
`useDataSpaceBoxVertices` (reusing `_padAnalyticAnchor[0]`, offset 1972)
makes `vertex_volume_main` forward `in.position` unchanged:

```
MetalShaders.metal vertex_volume_main
  if (useCameraInsideNearClip > 0.5 || useDataSpaceBoxVertices > 0.5)
    modelPos = in.position;               // dataset space, GL in_vertexPos parity
  else
    modelPos = boundsMin + in.position * (boundsMax - boundsMin);  // retired
```

`out.localPos` stays normalized [0,1] for camera-outside (`(modelPos -
boundsMin)/size`), so the fragment's `localPos`/`anchorData` semantics are
unchanged; the lattice anchor is `in.texcoord` (interpolated), which now has
per-vertex values bit-identical to GL's.

## 3. Results

| Test | before | after |
|---|---|---|
| CamOutside (+FixedStep, +NoJitter) | 1384 | **250** |
| Reference (camera-inside floor) | 178 | 178 |
| FineStep / Nearest / NearPlaneTiny | 178 | 178 |
| FlatTF | 0 | 0 |
| Linear | 343 | 343 |
| MaxIP | 6 | 6 |
| StepTF m0..m5lin | 0..181 | 0..181 |

The 250 remaining CamOutside pixels: 116 are |d|=1, 68 are |d|=2 (184 of 250
at or below a single 8-bit LSB-per-channel); the rest are knife-edge clusters
at tissue boundaries (e.g. a diagonal cluster x∈[196,349], y∈[162,315] and a
bottom-left cluster y∈[480,497]) where a ~1 ulp anchor difference flips the
nearest-neighbor pick across the scalar-500 TF step.

## 4. Why it stops at 250 (the floor)

The lattice is `evalPoint = anchorTex + evalStep*(jitterFrac + currentT)`, and
`evalStep` is bit-exact vs GL's `g_dirStep`, so the residual is purely the
anchor: each backend's interpolator rounds the same small triangle to within a
few ulp of the true value, and Apple's Metal interpolator is ~1 ulp off the GL
driver's (the +1 ulp measured on the camera-inside cap, update 76 §4). The
cap's triangles are far smaller (near-plane clip strip), so its floor is 178;
the densified box triangles are ~1/108 of the box's screen area, so their
rounding — and the knife-edge flips it produces — is a little larger (250).
With matching per-vertex values, no densification can reduce this further:
using a *different* subdivision than GL would break the per-vertex inputs, and
the interpolator hardware difference is irreducible at the geometry level.
