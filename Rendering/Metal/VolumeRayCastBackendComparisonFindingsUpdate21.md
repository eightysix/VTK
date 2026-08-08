# Camera-inside proxy mesh reproduction: the far face has genuinely mixed winding, so no cull convention separates it from the cap — Metal still rasterizes a far-face remnant (localPos z=1.0) at ~every gated pixel while GL logs only cap fragments; root cause narrows to GL's anchor-march vs Metal's recomputed-entry (update 21)

**Date:** 2026-08-08
**Scope:** Establish the winding/cull facts about the camera-inside proxy mesh with a full CPU-side reproduction (boxSource + near-plane clip + densify(2) + first-3-verts), and re-examine the far-face fragment discrepancy after the `setFrontFacingWinding:MTLWindingClockwise` change in `vtkMetalGPUVolumeRayCastMapper.mm`. The winding change is kept (it makes Metal keep the true near-plane cap, cell 122, matching GL's origin), but it does **not** remove the far-face fragments: cells 21/22/23/29 share the cap's winding class, so no uniform front/back rule can keep the cap while dropping the far face. The remaining discrepancy is that Metal's proxy fragment recomputes the ray entry (`setupVolumeRay` box + near-plane intersection) while GL marches from the interpolated anchor (`g_rayOrigin = ip_textureCoords`), so Metal's far remnants re-march the whole volume and can overwrite the cap with a slightly different ray.
**Follows:** [Update 20](VolumeRayCastBackendComparisonFindingsUpdate20.md), which closed the ~50× amplitude asymmetry (ray-anchor distance) and left pass A/B identity open. This update closes the "pass A interior anchor" question: post-fix the cap fragment is the true near-plane cap triangle (cell 122) and the "interior" fragment is gone.
**Persisted tools:** `Rendering/Metal/BackendComparisonTools/dump_proxy_mesh.cxx` (reproduces the proxy mesh and dumps verts/indices/matrices), `winding.py` (projects to Metal FB space and classifies per-pixel winding/coverage), `verify_fragments.py` (maps logged STEP fragments to mesh cells).

---

## 1. Conclusion

1. **The proxy mesh is fully reproduced on the CPU (95 verts / 126 cells), matching the app's `vertex_volume_main` logs.** `dump_proxy_mesh.cxx` rebuilds the exact chain used by both backends' `UpdateGeometry` camera-inside path: `vtkBoxSource` triangle list inserted with the "0 2 1" wedge convention, near-plane clip (`vtkClipConvexPolyData`, plane = frustum near plane pushed in by `(far-near)*0.001`), `vtkDensifyPolyData` with `SetNumberOfSubdivisions(2)`, first-3-verts index list. Projecting with the dumped V/P matrices and testing pixel-center coverage reproduces the logged fragments.
2. **The far face has genuinely mixed winding.** The six densified far-face triangles split by their Metal-FB (y-down) signed area: cells **21, 22, 23, 29** are `A_FB < 0` (counter-clockwise in the y-down window = "front" under `MTLWindingClockwise`, so the current cull keeps them), cells **36, 37** are `A_FB > 0` (back, culled). This mirrors the raw `vtkBoxSource` far face, whose two triangles {5,7,4} and {4,7,6} have opposite normals (+z and −z).
3. **The cap shares the kept far cells' winding class.** The near-plane cap triangle, cell 122 (verts 86, 40, 93, all at clip `w = 0.3847`, the offset near plane), has `A_FB = -1.32e10` — the same sign as kept far cells 21/22/23/29. **Therefore no uniform raw-triangle front/back rule can keep the cap (which GL renders) while culling the far face (which GL does not): any convention that keeps cell 122 also keeps cells 21-29.** GL's absent far fragments must be explained by something other than the raw-triangle cull rule.
4. **After the `MTLWindingClockwise` fix, Metal rasterizes one cap fragment + one far-face remnant per gated pixel per frame.** Over the whole post-fix log the gated pixels split 1:1: ~66 of 68 gated pixels show `far(z=1.0)` and `cap(z≈0.4486)` in equal counts (e.g. 6+6 over 6 frames; (422,92) is 5+5). Only (45,113) has no far remnant. The far remnant's `localPos.z = 1.0`; in the top-right gated pixels its xy sits on the shared far-face vertex 51 (0.3333, 0.3333, 1.0), i.e. these are **frustum-clipped remnants** of far-face triangles straddling the right/top clip planes (vertex 8 projects to NDC (4.36, 5.71); vertex 31 to (-5.6, -4.5)) — an unclipped barycentric match fails for them, which is expected.
5. **The far remnants re-march the entire volume from the near plane.** At (422,92) all 10 fragment invocations (5 frames × 2) share the identical first sample `tex=(0.505438, 0.506547, 0.450642)`, and 163/171 sample indices have >1 distinct tex value across fragments — the far and cap fragments march the same path but with slightly different rays (`rayDir` differs in the ~6th decimal because `rayDir = normalize(anchor − eye)` and the anchors are the cap vs far positions, per Update 20). So the far remnant is **not** a no-op: it re-composites the volume and, depending on rasterization order, overwrites the cap's color with a perturbed one.
6. **GL never shows far fragments.** All 90 `GL_RAY` lines across the gated pixels have origin z = 0.450/0.451 (the cap); zero far origins. Caveat: the `GL_RAY` readback (`vtkOpenGLGPUVolumeRayCastMapper.cxx:4290-4339`) renders one encoded channel per 12-pass batch and reads back the **last** fragment at the pixel, so it cannot distinguish "far fragment not rasterized" from "far fragment rasterized earlier and overwritten / empty".
7. **The structural divergence is GL's anchor-march vs Metal's recomputed entry.** GL's camera-inside fragment shader sets `g_rayOrigin = ip_textureCoords.xyz` (the interpolated anchor) then `g_rayOrigin += g_dirStep` (one full step, no-jitter test) and marches (`vtkVolumeShaderComposer.h:418,464`; `computeRayDirection() = normalize(ip_vertexPos − eyePos)` at :1716). A GL far fragment would start at `z ≈ 1.0 + step` and never enter `[0,1]³` — zero contribution. Metal's proxy fragment (`fragment_volume_main` → `setupVolumeRay`, MetalShaders.metal:3515-3550) instead recomputes the entry from the camera ray vs box + offset near plane, so far remnants re-enter at `z ≈ 0.4486` and fully re-march.
8. **Why the winding fix should have aligned the cull sets — and the consequence.** With `MTLWindingClockwise` + cull-Back, Metal's front/back classification equals GL's `GL_CCW`/`GL_BACK` classification for the same polygons (the Metal framebuffer is the GL window y-reflected, which flips orientation for every triangle equally). So if GL genuinely back-culled the far remnants, Metal would too; the fact that Metal keeps them is *consistent with GL also rasterizing them but producing nothing* (anchor-march). This favors hypothesis H2 below over H1.

---

## 2. Verified facts

### 2.1 Mesh reproduction (95 verts / 126 cells)

```
NEARPLANE origin=(-4.288648899, -4.288648899, 24.123650870) normal=(-0.172412186, -0.172412186, 0.969818579)
verts=95 cells=126   w>0: 95/95   ndc.z range=(0.500, 0.999)
```

The cap cell 122 vertices all sit on the offset near plane:

| vert | normalized volume pos | clip.w | ndc |
|---|---|---|---|
| 86 | (0.5101, 0.4926, 0.4463) | 0.3847 | (-6.8, -27.0, 0.5002) |
| 40 | (0.6019, 1.0000, 0.6019) | 0.3847 | (-220.1, 980.4, 0.5002) |
| 93 | (0.3706, 0.8309, 0.4979) | 0.3847 | (249.6, 644.6, 0.5002) |

(ndc.x/y of ±100-1000: cell 122 is a huge near-plane triangle covering the whole viewport after frustum clipping.)

### 2.2 Winding table (Metal FB y-down; A_win(y-up) = −A_FB)

| cell | verts | A_FB (y-down) | Metal Clockwise+Back | GL CCW on −A_FB |
|---|---|---|---|---|
| 122 (cap) | 86, 40, 93 | −1.32e10 | keep | keep |
| 21 (far) | 7, 8, 52 | −3.61e5 | keep | keep |
| 22 (far) | 8, 51, 52 | −3.12e5 | keep | keep |
| 23 (far) | 51, 7, 52 | −3.12e5 | keep | keep |
| 29 (far) | 51, 9, 54 | −1.86e5 | keep | keep |
| 36 (far) | 12, 10, 58 | +4.85e5 | cull | cull |
| 37 (far) | 10, 55, 58 | +2.59e5 | cull | cull |
| 9 | 3, 4, 47 | +2.09e6 | cull | cull |
| 114 | 38, 39, 91 | +4.60e8 | cull | cull |

The cap (122) and far cells 21/22/23/29 share the same class under both conventions; 36/37 are the opposite half of the far face. Confirms conclusion 3.

### 2.3 Fragment counts (post-fix log `/tmp/bc/update20/metal_proxy_cullfix.log`)

- Per gated pixel per frame: **1 cap fragment (localPos z≈0.4486, from cell 122) + 1 far-face remnant (localPos z=1.0)** — 6+6 per 6 frames at ~66/68 gated pixels; (45,113) is cap-only. At (422,92): 10 invocations over 5 frames (5+5).
- All 10 invocations at (422,92) share `SAMPLE i=0 tex=(0.505438, 0.506547, 0.450642)`: the far remnants re-march from the recomputed near-plane entry, not from their own anchor.
- 163/171 sample indices at (422,92) have >1 distinct tex value across fragments: the far ray differs from the cap ray in the ~6th decimal (`rayDir = normalize(anchor − eye)`, anchors on cap vs far face), so the double-march is not bit-identical and can perturb the final pixel.

### 2.4 GL side (update21 `nojitter_gl.log`, 90 `GL_RAY` lines)

```
origin z distribution: 12 @ 0.450, 78 @ 0.451   (all cap plane; zero far)
```

GL pixel ↔ Metal pixel (y-flip): GL(480,111)=Metal(480,400), GL(422,419)=Metal(422,92), GL(372,380)=Metal(372,131), GL(496,23)=Metal(496,488).

### 2.5 GL↔Metal per-sample comparison (`compare_out.txt`, pixel (422,419)/(422,92))

```
GL   per-i counts: [6], i range 0..174
Metal per-i counts: [12], i range 0..170
```

GL: 1 fragment/frame → 6 samples per i over 6 frames. Metal: 2 fragments/frame (cap + far) → 12 per i. Positions/raws agree to ≤1e-5 / ≤1e-6 against the first Metal fragment; the far remnant's extra 6 samples are invisible to the tool (it only keeps `me[i][0]`).

---

## 3. Structural root of the remaining discrepancy

| | OpenGL | Metal (proxy path) |
|---|---|---|
| ray origin | `g_rayOrigin = ip_textureCoords.xyz` (interpolated anchor) | `s.entryPoint = cameraPos + rayDir * tStart` (recomputed box/near-plane entry) |
| first sample | `g_rayOrigin += g_dirStep` (one full step; no-jitter test) | `entryPoint + rayDir * stepSize` |
| far remnant (z=1.0) | starts at z≈1.0+step → never enters cube → **empty contribution** | re-enters at z≈0.4486 → **full re-march, can overwrite cap** |
| source | vtkVolumeShaderComposer.h:418, 423, 437-465 | MetalShaders.metal:3529-3550 (`setupVolumeRay`), 4804-4867 (`fragment_volume_main`) |

The cap fragments agree between backends because for them the recomputed entry ≈ the interpolated anchor (both land on the offset near plane); the divergence appears exactly at the far remnants.

---

## 4. Open question and the decisive experiment

**Q: does GL rasterize the far-face remnants at all?**

- **H1 — GL back-culls them.** Requires the frustum clip to flip the clipped polygon's window winding (GL window y-up + `GL_CCW`) while the same clip in the y-down Metal window stays front (`Clockwise`). Rejected by the reasoning in conclusion 8: a pure coordinate reflection flips orientation for every triangle equally, so a cull set that differs between backends cannot come from the winding rule alone.
- **H2 — GL rasterizes them but they are empty (anchor-march).** The `GL_RAY` last-fragment-wins readback cannot see a far remnant that draws before the cap; even if it drew last, its zero contribution leaves the pixel unchanged. Consistent with all evidence.
- **Decisive test (cheap, no rebuild):** temporarily toggle GL's cull off (`glCullFace(GL_NONE)` or `vtkVolumeStateRAII` no-cull) and re-capture the NoJitter test. If H1, the far-face region gains contributions (pixels near shared vertex 51 change); if H2, the image is unchanged.
- **Decisive test (the likely fix):** make Metal's proxy camera-inside fragment march from its interpolated anchor (`in.localPos`) with GL's jitter semantics instead of `setupVolumeRay`'s recomputed entry. Far remnants then start at z≈1.0 and contribute nothing (matching H2's GL), the per-i counts drop from 12 to 6, and the far-remnant overwrite of the cap disappears. This is the next step.

---

## 5. Reproduction

### 5.1 Regenerate the mesh + winding tables

```sh
# 1. Build dump_proxy_mesh.cxx against the installed VTK build:
mkdir -p /tmp/bc/meshdump/build && cat > /tmp/bc/meshdump/CMakeLists.txt <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(meshdump CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(VTK REQUIRED PATHS <repo>/build_macos_metal/install NO_DEFAULT_PATH)
include(${VTK_USE_FILE})
add_executable(meshdump dump_proxy_mesh.cxx)
target_link_libraries(meshdump ${VTK_LIBRARIES})
EOF
cmake -S /tmp/bc/meshdump -B /tmp/bc/meshdump/build 2>/dev/null || true
cmake --build /tmp/bc/meshdump/build -j"$(sysctl -n hw.ncpu)"
# (copy Rendering/Metal/BackendComparisonTools/dump_proxy_mesh.cxx into /tmp/bc/meshdump/ first)

# 2. Dump verts.txt / indices.txt / matrices.txt:
/tmp/bc/meshdump/build/meshdump /tmp/bc/meshdump   # prints NEARPLANE + "verts=95 cells=126"

# 3. Winding / coverage tables for the gated pixels:
#    (winding.py was fixed during this update: its GL CCW column used A_FB
#    instead of A_win=-A_FB and was exactly inverted; it now reports the same
#    keep-set as Metal, per conclusion 8.)
python3 Rendering/Metal/BackendComparisonTools/winding.py /tmp/bc/meshdump 490 484 480 400

# 4. Map logged STEP fragments onto mesh cells (needs the post-fix Metal log):
python3 Rendering/Metal/BackendComparisonTools/verify_fragments.py /tmp/bc/meshdump \
  /tmp/bc/update20/metal_proxy_cullfix.log
```

### 5.2 Extract the fragment-count / far-remnant facts from a post-fix Metal log

```sh
# Distinct localPos.z classes per gated pixel (1:1 far/cap split):
rg "STEP px=" /tmp/bc/update20/metal_proxy_cullfix.log \
  | grep -o "localPos=([-0-9.e, +]*)" | sort | uniq -c

# Far remnant re-marches from the near plane (identical first sample):
rg "SAMPLE px=\(422, 92\) i=0 " /tmp/bc/update20/metal_proxy_cullfix.log \
  | grep -o "tex=([-0-9.e, +]*)" | sort | uniq -c

# GL never logs far origins (all cap-plane z):
rg "GL_RAY" /tmp/bc/update21/nojitter_gl.log \
  | sed -E 's/.*origin=\([-0-9.e]+, [-0-9.e]+, ([0-9.e-]+)\).*/\1/' \
  | awk '{printf "%.3f\n",$1}' | sort -n | uniq -c

# GL vs Metal per-i fragment counts at the matched pixel:
python3 Rendering/Metal/BackendComparisonTools/compare_gl_metal_samples.py \
  /tmp/bc/update21/nojitter_gl.log /tmp/bc/update21/nojitter_metal.log 422 92
```

### 5.3 Captures referenced

| artifact | contents |
|---|---|
| `/tmp/bc/meshdump/` | verts.txt (95), indices.txt (126), matrices.txt, dump_proxy_mesh.cxx, winding.py, verify_fragments.py, build/ |
| `/tmp/bc/update20/metal_proxy.log` | Metal NoJitter capture **before** the `MTLWindingClockwise` change |
| `/tmp/bc/update20/metal_proxy_cullfix.log` | Metal NoJitter capture **after** the winding change (5-6 frames, 68 gated pixels) |
| `/tmp/bc/update21/nojitter_gl.log` | GL NoJitter capture (90 `GL_RAY`, 6/frame `GL_SAMPLE` at (422,419)) |
| `/tmp/bc/update21/nojitter_metal.log` | Metal NoJitter capture (12/frame samples at (422,92)) |
| `/tmp/bc/update21/compare_out.txt` | per-i GL↔Metal table for (422,419)/(422,92) |

Capture commands: the NoJitter variant run through `vtkRenderingVolumeCxxTests` with `--vtk-factory-prefer RenderingBackend=<OpenGL|Metal>`, `MTL_LOG_*`/`VTK_GL_*` dump env vars as documented in `VolumeRayCastBackendComparisonProcedures.md` (cheatsheet section).

---

## 6. Status and remaining open items

- Updates 16-20: residual pinned to sample i=144 → ray-setup tilt (W2IF view-angle perturbation) → anchor-distance amplification; the proxy path was committed (7c663464e0).
- **Update 21 (this):** proxy mesh reproduced (95/126) and matched to logs; far-face winding shown to be genuinely mixed (21/22/23/29 share the cap's class, 36/37 opposite) — no raw-triangle rule separates cap from far; Metal still rasterizes one far remnant per pixel per frame post-fix and re-marches it from the near plane; GL's anchor-march vs Metal's recomputed-entry is the structural divergence.

Open items:

1. **Confirm H2 vs H1 with the GL cull-off probe** (section 4): if the far-face region is unchanged with cull disabled, GL rasterizes empty far remnants → H2.
2. **Implement the anchor-march proxy path in Metal** so far remnants contribute nothing (matches GL, drops per-i 12→6, removes the far-remnant overwrite). This is the intended next step after this update's commit.
3. **Pass A/B identity from Update 20 is now closed**: post-fix the cap fragment is cell 122; the earlier "interior anchor" was the pre-fix cap fragment from a different (interior) triangle.
