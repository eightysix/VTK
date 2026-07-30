# Yes — Significant Reduction Is Possible

The code has substantial structural duplication that can be eliminated without touching entry-point names, buffer/texture indices, struct memory layouts, or any arithmetic. I count roughly **400–450 lines** (≈30%) removable. Here are the concrete refactorings, ordered by impact:

---

## 1. Unify `marchVolume` and `marchSegment` (~150 lines saved)

The comment in the code itself acknowledges this: *"Any bug fix applied to the march loop body must be applied to BOTH functions."* They differ only in:

| Aspect | `marchVolume` | `marchSegment` |
|--------|--------------|----------------|
| Jitter | Computed internally | Received as parameter |
| Start position | `entryPoint + rayDir*jitter` | `rayOrigin + rayDir*(jitter + ceil(…)*step)` |
| Loop bound | `totalBoxT` / boundary check | `t1` parameter |
| Output | Returns `half4` | Writes to `thread half3&` / `thread half&` |
| Tex-local mapping | `(pt - texMinGlobal) * invTexSizeGlobal` | Identity (global texture) |

Merge into one function:

```metal
struct MarchParams {
    float3 rayOrigin;
    float3 rayDir;
    float  tStart;        // absolute t along ray where marching begins
    float  tEnd;          // absolute t where marching stops
    float  stepSize;
    float  jitter;        // pre-computed (0 if disabled)
    float  tTerminateMax;
    float3 blockMinGlobal;
    float3 blockMaxGlobal;
    float3 texMinGlobal;  // == float3(0) for grid traversal
    float3 texMaxGlobal;  // == float3(1) for grid traversal
    bool   checkBounds;   // true = marchVolume, false = marchSegment
};

inline half4 marchVolumeUnified(
    MarchParams p,
    half3 initialColor, half initialOpacity,
    constant VolumeMapperUniforms& volumeUniforms,
    constant PerBlockData& b,
    texture3d<float> volumeTexture,
    texture2d<float> transferFunctionTexture,
    texture2d<float> gradientOpacityTexture,
    texture3d<float> maskTexture,
    texture2d<float> labelMapTransferTexture,
    texture3d<float> minMaxTexture,
    texture3d<float> normalTexture,
    constant VolumeLightUniforms* lightUniforms)
{
    // ... shared setup (scalarScale, gradScale, materials, crop, etc.) ...

    float3 texSizeGlobal = max(p.texMaxGlobal - p.texMinGlobal, 1e-6);
    float3 invTexSizeGlobal = 1.0 / texSizeGlobal;

    float firstT = p.checkBounds
        ? p.jitter
        : p.jitter + ceil((p.tStart - p.jitter) / p.stepSize) * p.stepSize;

    float3 currentPoint = p.rayOrigin + p.rayDir * (p.checkBounds ? p.tStart : 0.0)
                        + p.rayDir * firstT;
    float currentT = firstT;
    float totalDist = p.tEnd - p.tStart;
    int maxSteps = min(max(1, int(ceil(totalDist / p.stepSize))), MAX_RAY_STEPS);

    // ... single loop body (identical logic) ...
    // Boundary check guarded by: if (p.checkBounds && any(currentPoint < ...)) break;

    return half4(accumulatedColor, accumulatedOpacity);
}
```

Both call sites become thin wrappers:

```metal
// In fragment_volume_main / fragment_volume_fullscreen_main:
half4 result = marchVolumeUnified({cameraPos, rayDir, 0.0, s.totalBoxT,
    stepSize, jitter, s.tTerminateMax, blockMinGlobal, blockMaxGlobal,
    texMinGlobal, texMaxGlobal, /*checkBounds=*/true},
    half3(0), 0.0h, volumeUniforms, b, ...);

// In fragment_volume_grid_traversal_main (per segment):
half4 seg = marchVolumeUnified({cameraPos, rayDir, segmentT0, segmentT1,
    stepSize, jitter, tTerminateMax, float3(0), float3(1),
    float3(0), float3(1), /*checkBounds=*/false},
    color, opacity, volumeUniforms, b, ...);
color = seg.xyz; opacity = seg.w;
```

---

## 2. Merge Identical Line Structs + Fragment Shaders (~60 lines saved)

`ThickLineVertexOut`, `RoundCapLineVertexOut`, and `MiterJoinLineVertexOut` are field-for-field identical. The three corresponding fragment shaders are character-for-character identical.

```metal
// One struct replaces three:
struct LineVertexOut {
  float4 position [[position]];
  float3 viewPos;
  float3 viewNormal;
  float4 vertexColor;
  float dist_to_centerline;
  uint cellId;
  uint propId;
};

// One shared implementation:
inline FragmentOutput shadeLineFragment(LineVertexOut in,
    constant MaterialUniforms& material,
    constant LightUniforms& lights,
    constant CoincidentOffsetUniforms& coinOffset)
{
  FragmentOutput out;
  float3 baseColor = in.vertexColor.rgb;
  float baseAlpha = in.vertexColor.a * material.opacity;
  float3 N = normalize(in.viewNormal);
  N.z = 1.0 - 2.0 * abs(in.dist_to_centerline);
  N = normalize(N);

  float3 totalDiffuse = float3(0.0), totalSpecular = float3(0.0);
  computePhongLighting(N, in.viewPos, baseColor, material.specularColor.rgb,
      material.specularColor.w, material.specularPower, lights, totalDiffuse, totalSpecular);

  out.color = float4(material.ambientColor.w * baseColor
                   + material.diffuseColor.w * totalDiffuse + totalSpecular, baseAlpha);
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);
  float cscale = length(float2(dfdx(in.position.z), dfdy(in.position.z)));
  out.depth = in.position.z + coinOffset.lineFactor * cscale + coinOffset.lineOffset / 65000.0;
  return out;
}

// Entry points remain (interface preserved), but are one-liners:
fragment FragmentOutput fragment_thick_line_main(LineVertexOut in [[stage_in]],
    constant MaterialUniforms& m [[buffer(0)]], constant LightUniforms& l [[buffer(1)]],
    constant SceneUniforms& s [[buffer(2)]], constant CoincidentOffsetUniforms& c [[buffer(3)]]) {
  return shadeLineFragment(in, m, l, c);
}
// ... same for fragment_round_cap_line_main, fragment_miter_join_line_main
```

---

## 3. Merge Glyph Structs + Shaders (~80 lines saved)

`GlyphVertexOut`, `GlyphLineVertexOut`, `GlyphPointVertexOut` differ only by the presence of `point_size`. The three vertex shaders differ only in whether they set `point_size`. The three fragment shaders differ only in the depth offset term.

```metal
struct GlyphVertexOut {
  float4 position [[position]];
  float3 viewPos;
  float3 viewNormal;
  float4 glyphColor;
  float3 modelPos;
  uint cellId;
  uint propId;
  float point_size [[point_size]];  // harmless when unused (rasterizer ignores for tris/lines)
};

inline GlyphVertexOut computeGlyphVertex(
    uint vertex_id, uint instance_id,
    constant float3* positions, constant float3* normals,
    constant float4x4* glyphTransforms, constant float3x3* glyphNormalTransforms,
    constant float4* glyphColors, constant uint* glyphPickIds,
    constant SceneUniforms& scene, constant uint& propId,
    float pointSize)
{
  GlyphVertexOut out;
  float4 worldPos = scene.modelMatrix * glyphTransforms[instance_id] * float4(positions[vertex_id], 1.0);
  out.viewPos = (scene.viewMatrix * worldPos).xyz;
  out.position = scene.projectionMatrix * float4(out.viewPos, 1.0);
  out.viewNormal = scene.normalMatrix * glyphNormalTransforms[instance_id] * normals[vertex_id];
  out.glyphColor = glyphColors[instance_id];
  out.cellId = glyphPickIds[instance_id] + 1u;
  out.propId = (propId == 0xFFFFFFFFu) ? 0u : (propId + 1u);
  out.point_size = pointSize;
  out.modelPos = positions[vertex_id];
  return out;
}

inline FragmentOutput shadeGlyphFragment(GlyphVertexOut in,
    constant MaterialUniforms& material, constant LightUniforms& lights,
    constant ClipPlaneUniforms& clipPlanes, float depthBias)
{
  if (isClipped(in.modelPos, clipPlanes)) discard_fragment();
  float3 N = normalize(in.viewNormal);
  float3 totalDiffuse = float3(0.0), totalSpecular = float3(0.0);
  computePhongLighting(N, in.viewPos, in.glyphColor.rgb, material.specularColor.rgb,
      material.specularColor.w, material.specularPower, lights, totalDiffuse, totalSpecular);
  FragmentOutput out;
  out.color = float4(material.ambientColor.w * in.glyphColor.rgb
                   + material.diffuseColor.w * totalDiffuse + totalSpecular,
                   in.glyphColor.a * material.opacity);
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);
  out.depth = in.position.z + depthBias;
  return out;
}
```

---

## 4. Remove `PointFragmentOutput` (~5 lines saved)

It is field-for-field identical to `FragmentOutput`. Replace all uses of `PointFragmentOutput` with `FragmentOutput`.

---

## 5. Macro-ify Volume Conversion Kernels (~80 lines saved)

Seven kernels share identical structure, differing only in source type and destination element type:

```metal
#define DEFINE_CONVERT_KERNEL(SRC_TYPE, DST_TYPE, SUFFIX)                        \
kernel void volume_convert_##SUFFIX(                                             \
    device const SRC_TYPE* src [[buffer(0)]],                                    \
    texture3d<DST_TYPE, access::write> dst [[texture(0)]],                       \
    constant VolumeConvertUniforms& u [[buffer(1)]],                             \
    uint3 gid [[thread_position_in_grid]])                                       \
{                                                                                \
    if (any(gid >= uint3(u.dimX, u.dimY, u.dimZ))) return;                      \
    uint srcIdx = (gid.z * u.dimY + gid.y) * u.dimX + gid.x;                    \
    DST_TYPE##4 val;                                                             \
    val.x = DST_TYPE(src[srcIdx * u.numComponents + 0]);                         \
    val.y = u.numComponents > 1 ? DST_TYPE(src[srcIdx * u.numComponents + 1])   \
                                : DST_TYPE(0);                                   \
    val.z = u.numComponents > 2 ? DST_TYPE(src[srcIdx * u.numComponents + 2])   \
                                : DST_TYPE(0);                                   \
    val.w = u.numComponents > 3 ? DST_TYPE(src[srcIdx * u.numComponents + 3])   \
                                : DST_TYPE(0);                                   \
    dst.write(val, gid);                                                         \
}

DEFINE_CONVERT_KERNEL(short, half,  short_to_half)
DEFINE_CONVERT_KERNEL(short, float, short_to_float)
DEFINE_CONVERT_KERNEL(int,   half,  int_to_half)
DEFINE_CONVERT_KERNEL(int,   float, int_to_float)
DEFINE_CONVERT_KERNEL(uint,  half,  uint_to_half)
DEFINE_CONVERT_KERNEL(uint,  float, uint_to_float)
DEFINE_CONVERT_KERNEL(float, half,  float_to_half)

#undef DEFINE_CONVERT_KERNEL
```

The `ushort_to_uchar` kernel stays separate (different logic: clamping + normalization).

---

## 6. Extract Shared Material Resolution (~30 lines saved)

The block that resolves `ambientColor`, `diffuseColor`, `baseOpacity` from flags/vertex-color/texture is copy-pasted in `fragment_main`, `fragment_peel`, and `fragment_peel_alpha_blend`:

```metal
struct ResolvedMaterial {
    float3 ambient;
    float3 diffuse;
    float  opacity;
};

inline ResolvedMaterial resolveMaterial(
    constant MaterialUniforms& material,
    constant SceneUniforms& scene,
    float4 vertexColor, float2 uv,
    texture2d<float> actorTexture, sampler actorSampler)
{
    bool hasVC = (scene.flags & (1u << 8)) != 0u;
    ResolvedMaterial r;
    r.ambient = hasVC ? vertexColor.rgb : material.ambientColor.rgb;
    r.diffuse = hasVC ? vertexColor.rgb : material.diffuseColor.rgb;
    r.opacity = (scene.flags & (1u << 10)) != 0u ? vertexColor.a : material.opacity;
    if ((scene.flags & (1u << 9)) != 0u) {
        float4 tex = actorTexture.sample(actorSampler, uv);
        r.ambient *= tex.rgb;
        r.diffuse *= tex.rgb;
        r.opacity *= tex.a;
    }
    return r;
}
```

---

## 7. Extract Shared Ray Reconstruction (~15 lines saved)

`fragment_volume_fullscreen_main` and `fragment_volume_grid_traversal_main` both reconstruct the ray from `inverseViewProjection` identically:

```metal
inline float3 reconstructRayDir(float2 screenPos, float2 viewportSize,
    constant VolumeMapperUniforms& u)
{
    float2 ndc = (screenPos / viewportSize) * 2.0 - 1.0;
    float4 wn = u.inverseViewProjection * float4(ndc.x, -ndc.y, 0.0, 1.0); wn.xyz /= wn.w;
    float4 wf = u.inverseViewProjection * float4(ndc.x, -ndc.y, 1.0, 1.0); wf.xyz /= wf.w;
    float3 bsz = max(u.volumeBoundsMax.xyz - u.volumeBoundsMin.xyz, 1e-6);
    float3 lN = ((u.worldToVolume * float4(wn.xyz, 1.0)).xyz - u.volumeBoundsMin.xyz) / bsz;
    float3 lF = ((u.worldToVolume * float4(wf.xyz, 1.0)).xyz - u.volumeBoundsMin.xyz) / bsz;
    return normalize(lF - lN);
}
```

---

## 8. Minor: Extract `propId` Mapping (~10 lines saved)

The expression `(propId == 0xFFFFFFFFu) ? 0u : (propId + 1u)` appears 8 times:

```metal
inline uint mapPropId(uint raw) { return (raw == 0xFFFFFFFFu) ? 0u : (raw + 1u); }
```

---

## Summary

| Refactoring | Lines saved | Risk |
|---|---|---|
| Unify `marchVolume`/`marchSegment` | ~150 | Low – parameterize 5 differences |
| Merge line structs + fragment | ~60 | None – identical code |
| Merge glyph structs + shaders | ~80 | None – trivial delta |
| Remove `PointFragmentOutput` | ~5 | None – identical struct |
| Macro-ify convert kernels | ~80 | None – mechanical |
| Extract `resolveMaterial` | ~30 | None – pure refactor |
| Extract ray reconstruction | ~15 | None – pure refactor |
| `mapPropId` helper | ~10 | None |
| **Total** | **~430** | |

All entry-point names, `[[buffer(N)]]`/`[[texture(N)]]` indices, struct field offsets, function-constant indices, and arithmetic remain untouched. The Metal compiler sees the same IR after inlining, so **runtime performance is identical**.
