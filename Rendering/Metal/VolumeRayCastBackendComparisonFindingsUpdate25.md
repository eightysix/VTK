# Camera-inside: what is NOT bit-identical in the cap-triangle barycentric interpolation — the clip-space matrix chain, not the rasterizer (update 25)

**Date:** 2026-08-08
**Scope:** Answer the update-24 open item H1 ("near-plane cap anchor interpolation is not bit-identical") precisely: reviewing commit `23e6c3d328`, the perspective-correct barycentric interpolation itself is identical in both backends; the divergence is upstream, in the float32 matrix chain that produces the window-space vertex coordinates (`gl_Position` vs `out.position`) which fix the interpolation weights.

**Follows:** [Update 24](VolumeRayCastBackendComparisonFindingsUpdate24.md), which proposed H1 (cap-anchor interpolation, "most likely") without pinpointing the exact non-bit-identical step.

---

## 1. Conclusion

1. **The rasterizer's interpolation is NOT the problem.** Both GL and Metal perform the same perspective-correct barycentric interpolation of the triangle vertex attribute in the fixed-function rasterizer. Given identical inputs (3 vertex attributes + 3 window-space vertex positions + same pixel center), the interpolated anchor is bit-identical on the same GPU.

2. **The non-bit-identical input is the projected window coordinate of each vertex**, which is derived from the clip-space position. GL computes it as `P * V * M * v` with **three float32 uniforms multiplied in the shader**; Metal precomputes `P*V` **on the CPU in float32** and the shader does `(P*V) * M * v`. Even with bit-identical P, V, M, the two `(P·V)` products round differently in the last ulp (CPU fixed-order non-FMA sum vs GPU FMA-contracted shader product).

3. **That 1-ulp clip-position difference is enough.** It shifts window coordinates by ~5e-5 px, perturbing the last bit of the barycentric weights, which moves the interpolated data-space anchor by ~1e-6..1e-5 data units across a ~10-unit cap triangle — matching the observed ~1e-7-per-sample `g_dirStep` drift (0.006% frame-aligned y-step) and the residual scalar-1150 knife-edge flips.

4. **Fix: compose `(P·V·M)` once in float32 on the CPU** and have both backends do a single `mat * vec` — or make GL consume a CPU-precomposed `P·V` exactly as Metal does, so the intermediate rounding is identical.

---

## 2. The two clip-space chains side by side

### 2a. GL vertex shader (`Rendering/VolumeOpenGL2/shaders/raycastervs.glsl` + `vtkVolumeShaderComposer.h`)

```glsl
// ComputeClipPositionImplementation (vtkVolumeShaderComposer.h:87-89)
gl_Position = in_projectionMatrix * in_modelViewMatrix * in_volumeMatrix[0] *
  vec4(in_vertexPos.xyz, 1.0);
```

- `in_projectionMatrix`, `in_modelViewMatrix`, `in_volumeMatrix[0]` are **three separate float32 uniforms**.
- The whole `P*V*M*v` product is evaluated in the shader; the `(P·V)` and subsequent products round at the GPU's chosen order/FMA contraction.
- Source matrices: `vtkOpenGLCamera::GetKeyMatrices` (`vtkOpenGLCamera.cxx:119-140`): `WCVCMatrix = transpose(GetModelViewTransformMatrix())`, `VCDCMatrix = transpose(GetProjectionTransformMatrix(aspect, -1, 1))`; `in_volumeMatrix[0]` = volume's model matrix. Uploaded via `prog->SetUniformMatrix` (double→float conversion, `vtkOpenGLGPUVolumeRayCastMapper.cxx:3965-3971`).

### 2b. Metal vertex shader (`Rendering/Metal/Shaders/MetalShaders.metal`)

```metal
// MetalShaders.metal:2921
out.position = volumeUniforms.viewProjection * volumeUniforms.volumeToWorld * float4(modelPos, 1.0);
```

- `viewProjection` = `P·V` **precomputed on the CPU** in float32 with a fixed non-FMA dot-product sum:

```cpp
// vtkMetalGPUVolumeRayCastMapper.mm:7157-7159
uniforms.ViewProjectionMatrix[c * 4 + r] = P[0 * 4 + r] * V[c * 4 + 0] +
  P[1 * 4 + r] * V[c * 4 + 1] + P[2 * 4 + r] * V[c * 4 + 2] +
  P[3 * 4 + r] * V[c * 4 + 3];
```

- `P`, `V` come from `vtkMetalCamera::GetCachedSceneTransforms` (float32 `static_cast<float>` of the same double `vtkCamera` matrices, `vtkMetalCamera.mm:54-65`).
- The shader then multiplies `viewProjection * volumeToWorld * float4(...)` — the `(P·V)` matrix is already rounded; the remaining `* M * v` happens in the shader.

### 2c. What is bit-identical (checked)

| item | GL | Metal | identical? |
|---|---|---|---|
| vertex attribute values (cap data-space positions) | `points->GetData()` float array (`vtkOpenGLGPUVolumeRayCastMapper.cxx:1292`) | `static_cast<float>(pt)` from same densify output (`vtkMetalGPUVolumeRayCastMapper.mm:5567`) | yes |
| cap geometry (clip + densify, near-plane offset) | `vtkClipConvexPolyData` + `vtkDensifyPolyData` subdivisions=2 (`vtkOpenGLGPUVolumeRayCastMapper.cxx:1159-1250`) | same pipeline (`vtkMetalGPUVolumeRayCastMapper.mm:5530-5550`) | yes |
| P, V float values | double→float via `SetUniformMatrix` | double→float via `static_cast<float>` | yes (Z row differs by design: GL nearz=-1/farz=1, Metal nearz=0/farz=1 — irrelevant to XY/w) |
| interpolation kernel | perspective-correct barycentric (fixed-function) | perspective-correct barycentric (fixed-function) | yes |

### 2d. The one difference

| step | GL | Metal |
|---|---|---|
| `P·V` computation | **in shader**, float32, GPU FMA/order | **on CPU**, float32, fixed non-FMA sum (`mm:7157`) |
| resulting intermediate | `(P·V)_gpu` | `(P·V)_cpu` |
| final product | `(P·V)_gpu * M * v` | `(P·V)_cpu * M * v` |

`(P·V)_gpu` and `(P·V)_cpu` differ by ≤1 ulp per element. This propagates into `gl_Position`/`out.position`, then the viewport transform, then the barycentric weights, then the interpolated anchor.

---

## 3. Magnitude budget

- 1 ulp of clip coords (~|clip| ≈ 10..100) → window coord shift ≈ `2^-23 · clip · (0.5·viewport/clip.w)` ≈ **5e-5 px**.
- Barycentric weight delta over a ~200 px cap triangle ≈ `5e-5/200` ≈ **2.5e-7**.
- Interpolated anchor span across the ~10 data-unit triangle → anchor delta ≈ `2.5e-7 · 10` ≈ **~2.5e-6 data units** (bounded 1e-6..1e-5).
- `g_dirStep = normalize(anchorData − eyeData)` over a ~79-unit eye-anchor distance → direction delta ≈ `1e-5/79` ≈ **1.3e-7 per sample** — matching the measured ~1e-7-per-sample drift and 0.006% frame-aligned y-step.

---

## 4. The fix (toward bit-identical anchors)

1. **Preferred:** compute `MVP = P·V·M` once in float32 on the CPU (one canonical non-FMA order) and upload a single matrix; both shaders do `clip = MVP * vec4(v, 1.0)`. The intermediate rounding is then identical by construction.
2. **Alternative:** make GL consume a CPU-precomposed `P·V` (and `P·V·M`) exactly like Metal does, replacing the shader-side `P*V*M*v` product — the two backends then share the same float32 intermediate regardless of GPU FMA behavior.
3. Verify with `compare_gl_metal_accum.py` at (422,92)/(372,131): the per-sample anchor trace should agree to float32 and the scalar-1150 knife-edge flip should vanish.

---

## 5. Reproduction

```bash
# capture GL + Metal per-sample logs (procedures: dummy-baseline trick)
python3 Rendering/Metal/BackendComparisonTools/compare_gl_metal_accum.py \
  gl_samples.log metal_samples.log 422 92
```

Expected before the fix: anchor drift ~1e-7/sample, one knife-edge sample flips across scalar 1150. After the fix: per-sample anchors bit-identical, residual knife-edge pixels drop to 0.
