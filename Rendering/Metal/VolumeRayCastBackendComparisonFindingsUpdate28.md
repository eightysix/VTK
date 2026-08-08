# Camera-inside: stage 2 of probe #1 — box-corner reorder makes the cap meshes byte-identical, but exposes a winding/cull mismatch (CW→CCW) and does NOT remove the knife-edge residual (update 28)

**Date:** 2026-08-08
**Scope:** Applied the update-27 fix candidate (reorder Metal's `boxSource` corner literal to GL's `ijkCorners` order) and verified it directly: the uploaded cap-mesh vertex arrays + index lists are now **byte-identical** between the backends. Two follow-on findings: (1) the reorder **flips the winding** of the clip/densify output, so Metal's proxy back-face cull (previously `MTLWindingClockwise` to compensate for the old geometry) must now be `MTLWindingCounterClockwise` to match GL's `GL_CCW/GL_BACK` — without it the byte-identical mesh renders fully culled (dark image, max |Δ| 222, whole image wrong). (2) Even with byte-identical meshes, the camera-inside residual persists (max |Δ| 22, 307 px ≥5; knife-edge pixels (422,92) and (372,131) still flip ~17–22 levels). So the vertex-attribute difference was necessary but NOT sufficient — the residual lives in the fragment-stage anchor/barycentric or step computation, consistent with update-25's window-space P·V·M·v hypothesis.

**Follows:** [Update 27](VolumeRayCastBackendComparisonFindingsUpdate27.md).
**Status of the working tree:** uncommitted; the box-corner reorder (mm:5444), the winding flip (mm:6079, now the confirmed fix, not a temp), and the two `TEMP DEBUG` cap-mesh dumps from update 27.

---

## 1. The fix candidate, applied

Update 27 hypothesized that Metal's `boxSource` literal (`{000,100,110,010,001,101,111,011}`) triangulated the box faces with different diagonals than GL's `ijkCorners` (`{000,100,010,110,001,101,011,111}`), so the same clip+densify pipeline produced different cap geometry. This stage swapped Metal's literal entries 2↔3 and 6↔7 to GL's order (mm:5444). The `tris[36]` index list and the `(t0, t2, t1)` cell insertion were already identical to GL (verified line-by-line), and the rest of the camera-inside clip pipeline (near-plane extraction → `vtkClipConvexPolyData` → `vtkDensifyPolyData` 2 subdivisions → take first-3-vertices-per-cell) is line-equivalent.

## 2. Direct verification: meshes now byte-identical

Re-captured with the update-27 cap-mesh dumps (`cmp_mesh.py`):

- GL blocks=6, Metal blocks=5 (frame-phase; GL re-renders the proxy one extra time).
- At every paired camera state: **95 vertices and 126 indices byte-identical** (float32 bits match exactly). The single "CAM MISMATCH" block is a frame-phase artifact (viewAngle `30` vs `30.0000008`), not a data difference.
- So probe #1's root cause (vertex-attribute sets differing) is **confirmed and eliminated**: the two backends now feed identical triangle soup to their vertex shaders.

## 3. Finding 1: winding/cull must flip CW → CCW

With the byte-identical mesh the Metal render broke completely: max |Δ| 222, every pixel wrong, dark + "weird shapes" (bright rows 101–511 only, corners black). Root cause: the corner reorder changes the *order* of the box vertices, which changes the winding of every face triangulation. The clip/densify output inherits the flipped winding, so Metal's proxy cull — previously `MTLWindingClockwise` chosen to match the *old* geometry — now culls the cap/front faces.

Fix (AB-tested in one rebuild): `setFrontFacingWinding:MTLWindingCounterClockwise` (mm:6079), matching GL's default `GL_CCW` front face with `glCullFace(GL_BACK)` (vtkVolumeStateRAII.h:47-49). Result:

| capture | max |Δ| | px diff>0 | px ≥5 |
|---|---|---|---|---|
| GL vs MT camera-inside, old build (update 27) | 22 | — | 262 |
| GL vs MT camera-inside, fixed+CCW | **22** | 70947 | **307** |
| MT old vs MT fixed (same GL input) | 13 | 13222 | 127 |
| GL vs MT camera-outside (CamOutsideNoJitter), fixed | 8 | 36096 | 27 |

Camera-outside unchanged (its box/winding was already consistent; no regression). The fix's own delta vs the old build is bounded (127 px ≥5) — it moved the Metal image, did not regress it.

## 4. Finding 2: the knife-edge residual persists

Despite byte-identical mesh inputs, the camera-inside residual is essentially unchanged:

| pixel | GL | MT (fixed) | |Δ| | update-23 value |
|---|---|---|---|---|
| (422,92) | (238,176,140) | (238,192,159) | 19 | 17 (update 23: 35→17) |
| (372,131) | (238,160,121) | (237,142,99) | 22 | 22 (update 23: 41→22) |

So update 27's "vertex sets differ → per-triangle float32 anchor sums differ" mechanism, though real, was **not the driver of the ~1e-5 anchor drift / knife-edge flips**. The mesh is now identical; the residual must come from the fragment stage after rasterization:

- **update-25 hypothesis (strengthened):** the window-space vertex coordinate `P*V*M*v` still differs between backends at the 1-ulp level — GL computes it in-shader on GPU (FMA), Metal precomputes `(P*V)*M*v` on CPU and feeds `useCameraInsideNearClip` with fixed non-FMA float32 sums (mm:7157). That 1-ulp window-space difference → last-bit barycentric weights → ~1e-6..1e-5 data-unit anchor shift → knife-edge flip.
- update-26's in-shader `P*V*M*v` Metal change (committed) reportedly "changed nothing"; with the mesh now identical it is worth re-verifying that change actually reached both shaders and that no other window-space input differs (viewport transform, NDC z, vertex shader output position).

## 5. Status vs update-27 next steps

- [x] 1. Reorder Metal boxSource corners to GL order → meshes byte-identical.
- [ ] 2. Verify box *double inputs* (`block->VolumeGeometry` vs `ModelBounds`) bit-identical → superseded: the byte-identical float32 uploads prove the double→float path matches for this test; still worth confirming for the transformed/spacing cases.
- [ ] 3. Residual knife-edge flips → **did NOT drop to 0** (section 4) → next probe: window-space vertex positions / P·V·M·v byte-compare.
- [x] 4. Camera-outside re-baseline → no regression (max 8).
- [ ] 5. Remove TEMP DEBUG dumps; decide on reverting update-26's in-shader change (now less likely to matter, but see section 4 note).

## 6. Next steps (stage 3)

1. Byte-compare the **window-space vertex positions** (post-`P*V*M*v`) emitted by both vertex shaders for the identical mesh — the decisive test for update-25's hypothesis. GL: dump `gl_Position` after the transform; Metal: dump the same after the equivalent transform. Both at the same 3-4 vertices of one densified cell.
2. If they differ: make the two computations bit-identical (single CPU-composed float32 P·V·M fed to both, or both in-shader GPU FMA). Update 26 tried one direction; re-verify with the now-identical mesh.
3. Re-survey after the P·V·M·v fix: knife-edge flip pixels (422,92) and (372,131) should drop to 0.
4. Clean up: remove cap-mesh `TEMP DEBUG` dumps, remove update-26 in-shader change if it stays ineffective, keep corner-order + winding fixes.

## 7. Reproduction

```sh
# build with the two fixes (corner order mm:5444, winding mm:6079)
./macos_metal_build.sh --resume

# GL clean render (dummy baseline trick)
build_macos_metal/bin/vtkRenderingVolumeCxxTests \
  TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
  --vtk-factory-prefer RenderingBackend=OpenGL -D build_macos_metal/ExternalData/Testing \
  -T build_macos_metal/Testing/Temporary -V /tmp/bc/u28/baseline.png
cp build_macos_metal/Testing/Temporary/baseline.png /tmp/bc/u28/gl/clean.png

# Metal clean render
build_macos_metal/bin/vtkRenderingVolumeCxxTests \
  TestGPURayCastCameraInsideTransformationNoShadeNoGradOpNoTransformNoJitter \
  --vtk-factory-prefer RenderingBackend=Metal -D build_macos_metal/ExternalData/Testing \
  -T build_macos_metal/Testing/Temporary -V /tmp/bc/u28/baseline.png
cp build_macos_metal/Testing/Temporary/baseline.png /tmp/bc/u28/metal/clean_ccw.png

# cap-mesh byte-compare (same as update 27): python3 /tmp/bc/u28/cmp_mesh.py
# => VERTS identical, INDICES identical at every paired camera state
```

Artifacts: `/tmp/bc/u28/gl/clean.png`, `/tmp/bc/u28/metal/clean_ccw.png`, `/tmp/bc/u28/gl/out_camout.png`, `/tmp/bc/u28/metal/out_camout_ccw.png`.
