// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
//
// Metal shaders for VTK Metal rendering backend.
//

#include <metal_stdlib>
using namespace metal;

// Constexpr samplers: avoids per-draw sampler binding overhead.
// sVolume — linear min/mag, clamp-to-edge (volume data, transfer function, gradient opacity)
// sNearest — nearest min/mag, clamp-to-edge (depth, mask, label map, min-max occupancy)
constexpr sampler sVolume(filter::linear, address::clamp_to_edge);
constexpr sampler sNearest(filter::nearest, address::clamp_to_edge);

// Maximum number of lights
#define MAX_LIGHTS 8

// Scene-level uniforms (camera transforms, viewport)
struct SceneUniforms {
  float4x4 viewMatrix;
  float4x4 projectionMatrix;
  float3x3 normalMatrix;       // inverse of view matrix 3x3 (matching WebGPU)
  float4x4 modelMatrix;
  float4 viewport;             // x, y, width, height
  uint flags;
  float pointSize;
};

// Coincident topology offset (P1-5)
struct CoincidentOffsetUniforms {
  float polygonFactor;
  float polygonOffset;
  float lineFactor;
  float lineOffset;
  float pointOffset;
};

// Vertex color override (P1-4)
struct VertexColorUniforms {
  float4 color;
};

// Clipping planes (P1-6)
struct ClipPlaneUniforms {
  float4 planes[6];            // up to 6 clip planes (ax+by+cz+d)
  int numClipPlanes;
};

// Per-material uniforms
struct MaterialUniforms {
  float4 ambientColor;         // rgb + ambient_intensity
  float4 diffuseColor;         // rgb + diffuse_intensity
  float4 specularColor;        // rgb + specular_intensity
  float4 color;                // base color (unused in lighting)
  float opacity;
  float specularPower;
  float2 _padding;
};

// Light data
struct Light {
  float4 position;             // xyz + type (0=headlight, 1=directional, 2=point, 3=spot)
  float4 direction;            // xyz + cone_angle
  float4 color;                // rgb + intensity
  float4 attenuation;          // constant, linear, quadratic, spot_exponent
};

struct LightUniforms {
  Light lights[MAX_LIGHTS];
  int lightCount;
  float _padding[3];
};

// Vertex input attributes
struct VertexIn {
  float3 position  [[attribute(0)]];
  float3 normal    [[attribute(1)]];
};

// Vertex output / fragment input
struct VertexOut {
  float4 position [[position]];
  float3 viewPos;
  float3 viewNormal;
  float4 vertexColor;    // P1-1A: per-vertex color from scalar mapping
  float2 uv;             // P5-5A: texture coordinates
  float3 modelPos;       // Optimized 6-plane clip validation
  uint cellId;           // P2-8: flat-interpolated cell ID (1-based, 0=background)
  uint propId;           // P2-8: flat-interpolated prop ID (1-based, 0=background)
  uint compositeIndex;   // P2-8: flat-interpolated composite index (0 = no composite)
};

// Per-draw picking identity. propId is the renderer's PropArray index assigned
// per-render during selection passes; compositeIndex carries the composite
// dataset block index (flat index) for batched blocks.
struct PickIds {
  uint propId;
  uint compositeIndex;
};

// Fragment output with explicit depth
struct FragmentOutput {
  float4 color [[color(0)]];
  uint4 ids [[color(1)]];
  float depth [[depth(any)]];
};


// ---------------------------------------------------------------------------
// Shared Shader Helper Functions
// ---------------------------------------------------------------------------

// Check clipping dynamically to support full 6 planes without bloating Varyings
inline bool isClipped(float3 modelPos, constant ClipPlaneUniforms& clipPlanes) {
  if (clipPlanes.numClipPlanes > 0) {
    for (int i = 0; i < clipPlanes.numClipPlanes && i < 6; ++i) {
      if (dot(float4(modelPos, 1.0), clipPlanes.planes[i]) < 0.0) {
        return true;
      }
    }
  }
  return false;
}

// Unified fast lighting calculation preventing duplicated loops / logic across fragments
inline void computePhongLighting(
    float3 N, float3 viewPos, float3 diffuseColor, float3 specularColor, 
    float specularIntensity, float specularPower,
    constant LightUniforms& lights,
    thread float3& totalDiffuse, thread float3& totalSpecular) {
  
  float3 viewDir = normalize(-viewPos);
  
  for (int i = 0; i < lights.lightCount && i < MAX_LIGHTS; ++i) {
    Light L = lights.lights[i];
    int lightType = int(L.position.w);
    float3 lightColor = L.color.rgb * L.color.w;
    float attenuation = 1.0;
    float3 toLight;
    
    if (lightType == 0) { // Headlight
      toLight = float3(0.0, 0.0, 1.0);
    } else if (lightType == 1) { // Directional
      toLight = normalize(-L.direction.xyz);
    } else { // Point / Spot
      toLight = L.position.xyz - viewPos;
      float dist = length(toLight);
      toLight = dist > 0.00001 ? toLight / dist : float3(0,0,1);
      attenuation = 1.0 / (L.attenuation.x + L.attenuation.y * dist + L.attenuation.z * dist * dist);
      
      if (lightType == 3) { // Spot specifics
        float spotCos = dot(-toLight, normalize(L.direction.xyz));
        float spotCutoff = cos(L.direction.w * (M_PI_F / 180.0));
        attenuation *= select(0.0f, pow(max(spotCos, 0.0f), L.attenuation.w), spotCos > spotCutoff);
      }
    }

    float NdotL = max(dot(N, toLight), 0.0);
    float df = (lightType == 0) ? max(N.z, 0.000001f) : NdotL;
    
    totalDiffuse += df * diffuseColor * lightColor * attenuation;
    
    // Only calculate reflect direction / specular if illuminated and valid specular intensity
    if (NdotL > 0.0 && specularIntensity > 0.0) {
      float3 reflDir = reflect(-toLight, N);
      float sf = pow(max(dot(viewDir, reflDir), 0.0), specularPower);
      totalSpecular += sf * specularIntensity * specularColor * lightColor * attenuation;
    }
  }
}

inline uint mapPropId(uint raw) { return (raw == 0xFFFFFFFFu) ? 0u : (raw + 1u); }

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

// ---------------------------------------------------------------------------
// Vertex shader
// ---------------------------------------------------------------------------
vertex VertexOut vertex_main(uint vertex_id [[vertex_id]],
                             VertexIn in [[stage_in]],
                             constant SceneUniforms& scene [[buffer(2)]],
                             constant float4* vertexColors [[buffer(3)]],
                             constant ClipPlaneUniforms& clipPlanes [[buffer(5)]],
                             constant uint* cellIds [[buffer(6)]],
                             constant PickIds& pickIds [[buffer(7)]],
                             constant float2* triangleUVs [[buffer(8)]]) {
  VertexOut out;

  float4 worldPos = scene.modelMatrix * float4(in.position, 1.0);
  float4 viewPos = scene.viewMatrix * worldPos;
  out.viewPos = viewPos.xyz;
  out.position = scene.projectionMatrix * viewPos;
  out.viewNormal = scene.normalMatrix * in.normal;
  out.vertexColor = vertexColors[vertex_id];
  out.uv = triangleUVs[vertex_id];
  out.modelPos = in.position; // Direct pass for unbounded planes evaluation
  out.cellId = cellIds[vertex_id];
  out.propId = mapPropId(pickIds.propId);
  out.compositeIndex = pickIds.compositeIndex;

  return out;
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------
fragment FragmentOutput fragment_main(VertexOut in [[stage_in]],
                              constant MaterialUniforms& material [[buffer(0)]],
                              constant LightUniforms& lights [[buffer(1)]],
                              constant SceneUniforms& scene [[buffer(2)]],
                              constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
                              constant ClipPlaneUniforms& clipPlanes [[buffer(5)]],
                              texture2d<float> actorTexture [[texture(0)]],
                              sampler actorSampler [[sampler(0)]]) {
  if (isClipped(in.modelPos, clipPlanes)) discard_fragment();

  float3 N = normalize(in.viewNormal);

  ResolvedMaterial r = resolveMaterial(material, scene, in.vertexColor, in.uv, actorTexture, actorSampler);

  float3 totalAmbient = material.ambientColor.w * r.ambient;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);

  computePhongLighting(N, in.viewPos, r.diffuse, material.specularColor.rgb, material.specularColor.w, material.specularPower, lights, totalDiffuse, totalSpecular);

  FragmentOutput out;
  out.color = float4(totalAmbient + material.diffuseColor.w * totalDiffuse + totalSpecular, r.opacity);
  out.ids = uint4(in.cellId, in.propId, in.compositeIndex, 0u);
  
  float cscale = length(float2(dfdx(in.position.z), dfdy(in.position.z)));
  out.depth = in.position.z + coinOffset.polygonFactor * cscale + coinOffset.polygonOffset / 65000.0;
  return out;
}

fragment FragmentOutput fragment_edge_main(VertexOut in [[stage_in]],
                                   constant MaterialUniforms& material [[buffer(0)]],
                                   constant SceneUniforms& scene [[buffer(2)]],
                                   constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
                                   constant float4& edgeColor [[buffer(4)]],
                                   constant ClipPlaneUniforms& clipPlanes [[buffer(5)]]) {
  if (isClipped(in.modelPos, clipPlanes)) discard_fragment();
  FragmentOutput out;
  out.color = float4(edgeColor.rgb, edgeColor.a * material.opacity);
  out.ids = uint4(in.cellId, in.propId, in.compositeIndex, 0u);

  float cscale = length(float2(dfdx(in.position.z), dfdy(in.position.z)));
  out.depth = in.position.z + coinOffset.lineFactor * cscale + coinOffset.lineOffset / 65000.0;
  return out;
}

// ---------------------------------------------------------------------------
// Point shaders
// ---------------------------------------------------------------------------
struct PointVertexOut {
  float4 position [[position]];
  float point_size [[point_size]];
  float3 viewPos;
  float3 viewNormal;
  float4 pointColor;
  float3 tangent;
  float2 uv;
  float2 lut_uv;
  uint cellId;
  uint propId;
  uint compositeIndex;
};

vertex PointVertexOut vertex_point_main(
    uint vertex_id [[vertex_id]],
    constant float3* point_positions [[buffer(0)]],
    constant SceneUniforms& scene [[buffer(1)]],
    constant float3* point_normals [[buffer(2)]],
    constant float4* point_colors [[buffer(3)]],
    constant float3* point_tangents [[buffer(6)]],
    constant float2* point_uvs [[buffer(7)]],
    constant float2* point_color_uvs [[buffer(8)]],
    constant uint* pointCellIds [[buffer(11)]],
    constant PickIds& pickIds [[buffer(12)]]) {
  PointVertexOut out;
  float3 pos = point_positions[vertex_id];
  float4 worldPos = scene.modelMatrix * float4(pos, 1.0);
  float4 viewPos = scene.viewMatrix * worldPos;
  
  out.viewPos = viewPos.xyz;
  out.position = scene.projectionMatrix * viewPos;
  out.viewNormal = scene.normalMatrix * point_normals[vertex_id];
  out.point_size = 1.0;
  out.pointColor = point_colors[vertex_id];
  out.tangent = scene.normalMatrix * point_tangents[vertex_id];
  out.uv = point_uvs[vertex_id];
  out.lut_uv = point_color_uvs[vertex_id];
  out.cellId = pointCellIds[vertex_id];
  out.propId = mapPropId(pickIds.propId);
  out.compositeIndex = pickIds.compositeIndex;
  return out;
}

fragment FragmentOutput fragment_point_main(PointVertexOut in [[stage_in]],
                                    constant MaterialUniforms& material [[buffer(0)]],
                                    constant LightUniforms& lights [[buffer(1)]],
                                    constant SceneUniforms& scene [[buffer(2)]],
                                    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
                                    constant VertexColorUniforms& vertexColorUniform [[buffer(4)]]) {
  float3 N = normalize(in.viewNormal);

  bool showVertices = (scene.flags & (1u << 3)) != 0u;
  float3 baseColor = showVertices ? vertexColorUniform.color.rgb : in.pointColor.rgb;
  float baseAlpha = showVertices ? vertexColorUniform.color.a : in.pointColor.a;

  float3 totalAmbient = material.ambientColor.w * baseColor;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);
  
  computePhongLighting(N, in.viewPos, baseColor, material.specularColor.rgb, material.specularColor.w, material.specularPower, lights, totalDiffuse, totalSpecular);

  FragmentOutput out;
  out.color = float4(totalAmbient + material.diffuseColor.w * totalDiffuse + totalSpecular, baseAlpha * material.opacity);
  out.ids = uint4(in.cellId, in.propId, in.compositeIndex, 0u);
  out.depth = in.position.z + coinOffset.pointOffset / 65000.0;
  return out;
}

// -----------------------------------------------------------------------
// Shaped point shaders
// -----------------------------------------------------------------------
struct PointShapedVertexOut {
  float4 position [[position]];
  float3 viewPos;
  float3 viewNormal;
  float2 p_coord;
  float4 pointColor;
  float3 tangent;
  float2 uv;
  float2 lut_uv;
  uint cellId;
  uint propId;
  uint compositeIndex;
};

vertex PointShapedVertexOut vertex_point_shaped_main(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    constant float3* point_positions [[buffer(0)]],
    constant uint* connectivity [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant float3* point_normals [[buffer(3)]],
    constant float4* point_colors [[buffer(4)]],
    constant float3* point_tangents [[buffer(6)]],
    constant float2* point_uvs [[buffer(7)]],
    constant float2* point_color_uvs [[buffer(8)]],
    constant uint* shapedCellIds [[buffer(11)]],
    constant PickIds& pickIds [[buffer(12)]]) {
  
  const float2 tri_verts[4] = { float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1) };
  uint point_id = connectivity[instance_id];
  float3 pos = point_positions[point_id];

  float4 worldPos = scene.modelMatrix * float4(pos, 1.0);
  float4 viewPos = scene.viewMatrix * worldPos;
  float4 clipPos = scene.projectionMatrix * viewPos;

  float2 resolution = scene.viewport.zw;
  float2 screenPos = resolution * (0.5 * clipPos.xy / clipPos.w + 0.5);
  float2 corner = tri_verts[vertex_id];
  float2 expanded = screenPos + 0.5 * scene.pointSize * corner;

  PointShapedVertexOut out;
  out.position = float4(clipPos.w * ((2.0 * expanded) / resolution - 1.0), clipPos.z, clipPos.w);
  out.viewPos = viewPos.xyz;
  out.viewNormal = scene.normalMatrix * point_normals[point_id];
  out.p_coord = corner;
  out.pointColor = point_colors[point_id];
  out.tangent = scene.normalMatrix * point_tangents[point_id];
  out.uv = point_uvs[point_id];
  out.lut_uv = point_color_uvs[point_id];
  out.cellId = shapedCellIds[point_id];
  out.propId = mapPropId(pickIds.propId);
  out.compositeIndex = pickIds.compositeIndex;
  return out;
}

fragment FragmentOutput fragment_point_shaped_main(
    PointShapedVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant VertexColorUniforms& vertexColorUniform [[buffer(4)]]) {
  FragmentOutput out;

  float d = length(in.p_coord);
  bool drawSpheres = (scene.flags & (1u << 5)) != 0u;
  bool isRound = ((scene.flags >> 7) & 1u) == 0u;

  if ((isRound || drawSpheres) && d > 1.0) discard_fragment();

  float3 N;
  if (drawSpheres && d <= 1.0) {
    N = normalize(float3(in.p_coord, 1.0));
    N.z = sqrt(1.0 - d * d);

    float pointSize = clamp(scene.pointSize, 1.0, 100000.0);
    float r = pointSize / (scene.viewport.z * scene.projectionMatrix[0][0]);
    bool parallel = (scene.flags & 1u) != 0u;
    if (parallel) {
      out.depth = in.position.z + N.z * r * scene.projectionMatrix[2][2];
    } else {
      float s = -scene.projectionMatrix[2][2];
      out.depth = (s - in.position.z) / (N.z * r - 1.0) + s;
    }
  } else {
    N = normalize(in.viewNormal);
    out.depth = in.position.z;
  }

  bool showVertices = (scene.flags & (1u << 3)) != 0u;
  float3 baseColor = showVertices ? vertexColorUniform.color.rgb : in.pointColor.rgb;
  float baseAlpha = showVertices ? vertexColorUniform.color.a : in.pointColor.a;

  float3 totalAmbient = material.ambientColor.w * baseColor;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);
  
  computePhongLighting(N, in.viewPos, baseColor, material.specularColor.rgb, material.specularColor.w, material.specularPower, lights, totalDiffuse, totalSpecular);

  out.color = float4(totalAmbient + material.diffuseColor.w * totalDiffuse + totalSpecular, baseAlpha * material.opacity);
  out.ids = uint4(in.cellId, in.propId, in.compositeIndex, 0u);
  out.depth += coinOffset.pointOffset / 65000.0;
  return out;
}

// ---------------------------------------------------------------------------
// Line shaders (thick line, round cap, miter join — shared struct + fragment)
// ---------------------------------------------------------------------------
struct LineVertexOut {
  float4 position [[position]];
  float3 viewPos;
  float3 viewNormal;
  float4 vertexColor;
  float dist_to_centerline;
  uint cellId;
  uint propId;
  uint compositeIndex;
};

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
  out.ids = uint4(in.cellId, in.propId, in.compositeIndex, 0u);
  float cscale = length(float2(dfdx(in.position.z), dfdy(in.position.z)));
  out.depth = in.position.z + coinOffset.lineFactor * cscale + coinOffset.lineOffset / 65000.0;
  return out;
}

vertex LineVertexOut vertex_thick_line_main(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    constant float3* positions [[buffer(0)]],
    constant uint* lineIndices [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant float4* vertexColors [[buffer(3)]],
    constant float& lineWidth [[buffer(4)]],
    constant uint* cellIds [[buffer(5)]],
    constant PickIds& pickIds [[buffer(6)]]) {
  
  const float2 tri_verts[4] = { float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1) };
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

  float t = (p_coord.x + 1.0) * 0.5;
  float side = p_coord.y;

  float halfW = max(lineWidth, 1.0) * 0.5;

  float2 center = mix(p0_screen, p1_screen, t);
  float2 p = center + side * y_basis * halfW;

  float4 p_DC = mix(p0_DC, p1_DC, t);

  LineVertexOut out;
  out.position = float4(p_DC.w * ((2.0 * p) / resolution - 1.0), p_DC.z, p_DC.w);
  float3 pos_MC = mix(p0_MC, p1_MC, t);
  out.viewPos = (scene.viewMatrix * scene.modelMatrix * float4(pos_MC, 1.0)).xyz;
  out.viewNormal = scene.normalMatrix * float3(0.0, 0.0, 1.0);
  out.vertexColor = mix(vertexColors[p0_idx], vertexColors[p1_idx], t);
  out.dist_to_centerline = side;
  out.cellId = cellIds[instance_id];
  out.propId = mapPropId(pickIds.propId);
  out.compositeIndex = pickIds.compositeIndex;
  return out;
}

fragment FragmentOutput fragment_thick_line_main(
    LineVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]]) {
  return shadeLineFragment(in, material, lights, coinOffset);
}

// ---------------------------------------------------------------------------
// Round Cap Line Shaders 
// ---------------------------------------------------------------------------
vertex LineVertexOut vertex_round_cap_line_main(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    constant float3* positions [[buffer(0)]],
    constant uint* lineIndices [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant float4* vertexColors [[buffer(3)]],
    constant float& lineWidth [[buffer(4)]],
    constant uint* cellIds [[buffer(5)]],
    constant PickIds& pickIds [[buffer(6)]]) {

  float3 p_coord;
  const int CAP_SEGMENTS = 5;
  const float PI = 3.14159265358979;

  if (vertex_id < 6) {
    const float3 body_verts[6] = {
      float3(-0.5, 0.0, 0.0), float3(-0.5, 0.0, 1.0), float3( 0.5, 0.0, 0.0),
      float3( 0.5, 0.0, 1.0), float3(-0.5, 1.0, 0.0), float3(-0.5, 1.0, 1.0)
    };
    p_coord = body_verts[vertex_id];
  } else if (vertex_id < 21) {
    int local = vertex_id - 6;
    float theta0 = PI * 0.5 + (float(local / 3) * PI) / float(CAP_SEGMENTS);
    float theta1 = PI * 0.5 + (float(local / 3 + 1) * PI) / float(CAP_SEGMENTS);
    p_coord = (local % 3 == 0) ? float3(0.0) : 
              ((local % 3 == 1) ? float3(0.5 * cos(theta0), 0.5 * sin(theta0), 0.0) :
                                  float3(0.5 * cos(theta1), 0.5 * sin(theta1), 0.0));
  } else {
    int local = vertex_id - 21;
    float theta0 = 1.5 * PI + (float(local / 3) * PI) / float(CAP_SEGMENTS);
    float theta1 = 1.5 * PI + (float(local / 3 + 1) * PI) / float(CAP_SEGMENTS);
    p_coord = (local % 3 == 0) ? float3(0.0, 1.0, 1.0) : 
              ((local % 3 == 1) ? float3(0.5 * cos(theta0), 0.5 * sin(theta0) + 1.0, 1.0) :
                                  float3(0.5 * cos(theta1), 0.5 * sin(theta1) + 1.0, 1.0));
  }

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

  float w = max(lineWidth, 1.0);
  float2 adjusted_p0 = p0_screen + (p_coord.x * x_basis + p_coord.y * y_basis) * w;
  float2 adjusted_p1 = p1_screen + (p_coord.x * x_basis + p_coord.y * y_basis) * w;
  float2 p = mix(adjusted_p0, adjusted_p1, p_coord.z);

  float4 p_DC = mix(p0_DC, p1_DC, p_coord.z);

  LineVertexOut out;
  out.position = float4(p_DC.w * ((2.0 * p) / resolution - 1.0), p_DC.z, p_DC.w);
  out.viewPos = (scene.viewMatrix * scene.modelMatrix * float4(mix(p0_MC, p1_MC, p_coord.z), 1.0)).xyz;
  out.viewNormal = scene.normalMatrix * float3(0.0, 0.0, 1.0);
  out.vertexColor = mix(vertexColors[p0_idx], vertexColors[p1_idx], p_coord.z);
  out.dist_to_centerline = p_coord.y;
  out.cellId = cellIds[instance_id];
  out.propId = mapPropId(pickIds.propId);
  out.compositeIndex = pickIds.compositeIndex;
  return out;
}

fragment FragmentOutput fragment_round_cap_line_main(
    LineVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]]) {
  return shadeLineFragment(in, material, lights, coinOffset);
}

// ---------------------------------------------------------------------------
// Miter Join Line Shaders
// ---------------------------------------------------------------------------
vertex LineVertexOut vertex_miter_join_line_main(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    constant float3* positions [[buffer(0)]],
    constant uint* lineIndices [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant float4* vertexColors [[buffer(3)]],
    constant float& lineWidth [[buffer(4)]],
    constant uint* cellIds [[buffer(5)]],
    constant PickIds& pickIds [[buffer(6)]],
    constant uint& segmentCount [[buffer(7)]]) {
  
  const float2 tri_verts[4] = { float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1) };
  float2 p_coord = tri_verts[vertex_id];

  uint p0_idx = lineIndices[instance_id * 2];
  uint p1_idx = lineIndices[instance_id * 2 + 1];

  float4 p0_DC = scene.projectionMatrix * scene.viewMatrix * scene.modelMatrix * float4(positions[p0_idx], 1.0);
  float4 p1_DC = scene.projectionMatrix * scene.viewMatrix * scene.modelMatrix * float4(positions[p1_idx], 1.0);

  float2 resolution = scene.viewport.zw;
  float2 p0_screen = resolution * (0.5 * p0_DC.xy / p0_DC.w + 0.5);
  float2 p1_screen = resolution * (0.5 * p1_DC.xy / p1_DC.w + 0.5);

  float2 delta = p1_screen - p0_screen;
  float segLen = length(delta);
  float2 x_basis = segLen < 0.001 ? float2(1.0, 0.0) : (delta / segLen);
  float2 y_basis = float2(-x_basis.y, x_basis.x);

  float w = max(lineWidth, 1.0);
  float halfW = w * 0.5;
  float t = (p_coord.x + 1.0) * 0.5;
  float side = p_coord.y;
  float2 offset = side * y_basis * halfW;

  if (p_coord.x == -1.0 && instance_id > 0 && cellIds[instance_id - 1] == cellIds[instance_id]) {
    float4 prev_p0_DC = scene.projectionMatrix * scene.viewMatrix * scene.modelMatrix * float4(positions[lineIndices[(instance_id - 1) * 2]], 1.0);
    float2 prev_p0_screen = resolution * (0.5 * prev_p0_DC.xy / prev_p0_DC.w + 0.5);
    
    float2 prev_delta = p0_screen - prev_p0_screen;
    float prev_len = length(prev_delta);
    float2 prev_dir = prev_len < 0.001 ? float2(1.0, 0.0) : (prev_delta / prev_len);
    
    float2 miter = float2(-prev_dir.y, prev_dir.x) + float2(-x_basis.y, x_basis.x);
    float denom = dot(miter, y_basis);

    if (abs(denom) > 1e-3) {
      float miterOffset = halfW / denom;
      miterOffset = clamp(miterOffset, -4.0 * halfW, 4.0 * halfW);
      if (sign(dot(p_coord.y * y_basis, miter)) == sign(dot(float2(0.0, 1.0), miter))) {
        offset = miter * miterOffset;
      }
    }
  }

  if (p_coord.x == 1.0 && instance_id < segmentCount - 1 && cellIds[instance_id + 1] == cellIds[instance_id]) {
    float4 next_p1_DC = scene.projectionMatrix * scene.viewMatrix * scene.modelMatrix * float4(positions[lineIndices[(instance_id + 1) * 2 + 1]], 1.0);
    float2 next_p1_screen = resolution * (0.5 * next_p1_DC.xy / next_p1_DC.w + 0.5);
    
    float2 next_delta = next_p1_screen - p1_screen;
    float next_len = length(next_delta);
    float2 next_dir = next_len < 0.001 ? float2(1.0, 0.0) : (next_delta / next_len);
    
    float2 miter = float2(-x_basis.y, x_basis.x) + float2(-next_dir.y, next_dir.x);
    float denom = dot(miter, y_basis);

    if (abs(denom) > 1e-3) {
      float miterOffset = halfW / denom;
      miterOffset = clamp(miterOffset, -4.0 * halfW, 4.0 * halfW);
      if (sign(dot(p_coord.y * y_basis, miter)) == sign(dot(float2(0.0, 1.0), miter))) {
        offset = miter * miterOffset;
      }
    }
  }

  float2 p = mix(p0_screen, p1_screen, t) + offset;
  float4 p_DC = mix(p0_DC, p1_DC, t);

  LineVertexOut out;
  out.position = float4(p_DC.w * ((2.0 * p) / resolution - 1.0), p_DC.z, p_DC.w);
  out.viewPos = (scene.viewMatrix * scene.modelMatrix * float4(mix(positions[p0_idx], positions[p1_idx], t), 1.0)).xyz;
  out.viewNormal = scene.normalMatrix * float3(0.0, 0.0, 1.0);
  out.vertexColor = mix(vertexColors[p0_idx], vertexColors[p1_idx], t);
  out.dist_to_centerline = side;
  out.cellId = cellIds[instance_id];
  out.propId = mapPropId(pickIds.propId);
  out.compositeIndex = pickIds.compositeIndex;
  return out;
}

fragment FragmentOutput fragment_miter_join_line_main(
    LineVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]]) {
  return shadeLineFragment(in, material, lights, coinOffset);
}

// ---------------------------------------------------------------------------
// Compute Kernels (Tessellation mapping)
// ---------------------------------------------------------------------------
struct TessParams { uint numCells; uint cellIdOffset; };

kernel void polygonToTriangle(
    device uint* outConnectivity [[buffer(0)]],
    device float* edgeArray [[buffer(1)]],
    device uint* cellIds [[buffer(2)]],
    constant uint* connectivity [[buffer(3)]],
    constant uint* offsets [[buffer(4)]],
    constant uint* primitiveCounts [[buffer(5)]],
    constant TessParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.numCells) return;

  uint numTriangles = primitiveCounts[gid + 1u] - primitiveCounts[gid];
  uint outputOffset = primitiveCounts[gid] * 3u;
  uint inputOffset = offsets[gid];

  for (uint i = 0u; i < numTriangles; i++) {
    uint triangleId = primitiveCounts[gid] + i;
    edgeArray[triangleId] = (numTriangles == 1u) ? -1.0 : (i == 0u ? 2.0 : (i == numTriangles - 1u ? 0.0 : 1.0));
    cellIds[triangleId] = gid + params.cellIdOffset + 1u;

    outConnectivity[outputOffset] = connectivity[inputOffset];
    outConnectivity[outputOffset + 1u] = connectivity[inputOffset + i + 1u];
    outConnectivity[outputOffset + 2u] = connectivity[inputOffset + i + 2u];
    outputOffset += 3u;
  }
}

kernel void polyLineToLine(
    device uint* outConnectivity [[buffer(0)]],
    device uint* cellIds [[buffer(1)]],
    constant uint* connectivity [[buffer(2)]],
    constant uint* offsets [[buffer(3)]],
    constant uint* primitiveCounts [[buffer(4)]],
    constant TessParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.numCells) return;

  uint numLines = primitiveCounts[gid + 1u] - primitiveCounts[gid];
  uint outputOffset = primitiveCounts[gid] * 2u;
  uint inputOffset = offsets[gid];

  for (uint i = 0u; i < numLines; i++) {
    cellIds[primitiveCounts[gid] + i] = gid + params.cellIdOffset + 1u;
    outConnectivity[outputOffset] = connectivity[inputOffset + i];
    outConnectivity[outputOffset + 1u] = connectivity[inputOffset + i + 1u];
    outputOffset += 2u;
  }
}

kernel void polygonEdgesToLines(
    device uint* outConnectivity [[buffer(0)]],
    device uint* cellIds [[buffer(1)]],
    constant uint* connectivity [[buffer(2)]],
    constant uint* offsets [[buffer(3)]],
    constant uint* primitiveCounts [[buffer(4)]],
    constant TessParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.numCells) return;

  uint numEdges = primitiveCounts[gid + 1u] - primitiveCounts[gid];
  uint outputOffset = primitiveCounts[gid] * 2u;
  uint inputOffset = offsets[gid];

  for (uint i = 0u; i < numEdges; i++) {
    cellIds[primitiveCounts[gid] + i] = gid + params.cellIdOffset + 1u;
    outConnectivity[outputOffset] = connectivity[inputOffset + i];
    outConnectivity[outputOffset + 1u] = connectivity[inputOffset + (i + 1u) % numEdges];
    outputOffset += 2u;
  }
}

// ---------------------------------------------------------------------------
// Volume gradient/normal precomputation kernel (Phase 4)
// ---------------------------------------------------------------------------
struct NormalComputeUniforms {
    uint dimX, dimY, dimZ;
    float gsX, gsY, gsZ;
    float scalarScale;
    float scalarBias;
    float gradNormFactor;
    float invBoundsX, invBoundsY, invBoundsZ;
};
static_assert(sizeof(NormalComputeUniforms) == 48, "NormalComputeUniforms must be 48 bytes");

// Each thread computes one voxel's gradient from 6 neighbors in the volume
// texture (linear clamp, safe at borders) and stores encoded normal + magnitude.
// Output format: RGBA8Unorm. RGB = normal * 0.5 + 0.5, A = normalized gradMag.
kernel void volume_compute_normals(
    texture3d<float, access::sample> volume [[texture(0)]],
    texture3d<float, access::write> normalTex [[texture(1)]],
    constant NormalComputeUniforms& u [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint3 dims = uint3(u.dimX, u.dimY, u.dimZ);
    if (any(gid >= dims)) return;

    float3 pos = (float3(gid) + 0.5) / float3(dims);
    float3 gs = float3(u.gsX, u.gsY, u.gsZ);

    // Central differences (6 texel fetches — same as computeGradientFast)
    float sPX = volume.sample(sVolume, pos + float3(gs.x, 0, 0), level(0)).r;
    float sNX = volume.sample(sVolume, pos - float3(gs.x, 0, 0), level(0)).r;
    float sPY = volume.sample(sVolume, pos + float3(0, gs.y, 0), level(0)).r;
    float sNY = volume.sample(sVolume, pos - float3(0, gs.y, 0), level(0)).r;
    float sPZ = volume.sample(sVolume, pos + float3(0, 0, gs.z), level(0)).r;
    float sNZ = volume.sample(sVolume, pos - float3(0, 0, gs.z), level(0)).r;

    float3 rawGrad = float3(sPX - sNX, sPY - sNY, sPZ - sNZ);
    float3 dimsF = float3(float(u.dimX), float(u.dimY), float(u.dimZ));
    float3 invBounds = float3(u.invBoundsX, u.invBoundsY, u.invBoundsZ);
    float3 physGrad = rawGrad * dimsF * invBounds;
    float mag = length(physGrad);

    // Normalize and encode to [0, 1]
    float3 normal = mag > 1e-6 ? physGrad / mag : float3(0.0, 0.0, 1.0);
    float3 encoded = normal * 0.5 + 0.5;
    float gradMagNorm = saturate(mag / max(u.gradNormFactor, 1e-6));

    normalTex.write(float4(encoded, gradMagNorm), gid);
}

// ---------------------------------------------------------------------------
// 2D Mapper shaders
// ---------------------------------------------------------------------------
struct Mapper2DState { float4x4 wcvcMatrix; float4 color; float pointSize; float lineWidth; uint flags; };
struct Vertex2DIn { float2 position [[attribute(0)]]; };
struct Vertex2DOut { float4 position [[position]]; float4 color; };

vertex Vertex2DOut vertex_2d_main(Vertex2DIn in [[stage_in]], constant Mapper2DState& state [[buffer(1)]]) {
  Vertex2DOut out;
  out.position = state.wcvcMatrix * float4(in.position, 0.0, 1.0);
  out.color = state.color;
  return out;
}

fragment float4 fragment_2d_main(Vertex2DOut in [[stage_in]]) {
  return in.color;
}

// ---------------------------------------------------------------------------
// Depth Peeling Shaders
// ---------------------------------------------------------------------------
struct PeelUniforms { uint mode; uint peelPass; float2 viewportSize; };
struct FullscreenVertexOut { float4 position [[position]]; float2 texCoord; };

vertex FullscreenVertexOut vertex_fullscreen_main(uint vertex_id [[vertex_id]]) {
  const float2 positions[3] = { float2(-1, -1), float2( 3, -1), float2(-1,  3) };
  const float2 texCoords[3] = { float2(0, 1), float2(2, 1), float2(0, -1) };
  FullscreenVertexOut out;
  out.position = float4(positions[vertex_id], 0, 1);
  out.texCoord = texCoords[vertex_id];
  return out;
}

struct PeelInitOutput { float2 depthRange [[color(0)]]; };
struct PeelPassOutput { float4 backTemp  [[color(0)]]; float4 frontDest [[color(1)]]; float2 depthDest [[color(2)]]; };

fragment PeelInitOutput fragment_peel_init(
    VertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(5)]],
    texture2d<float> actorTexture [[texture(0)]],
    sampler actorSampler [[sampler(0)]]) {
  
  if (isClipped(in.modelPos, clipPlanes)) discard_fragment();

  PeelInitOutput out;
  out.depthRange = float2(-in.position.z, in.position.z);
  return out;
}

fragment PeelPassOutput fragment_peel(
    VertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(5)]],
    texture2d<float> actorTexture [[texture(0)]],
    sampler actorSampler [[sampler(0)]],
    texture2d<float, access::read> prevFrontTex [[texture(1)]],
    texture2d<float, access::read> prevDepthTex [[texture(2)]]) {
  
  if (isClipped(in.modelPos, clipPlanes)) discard_fragment();

  uint2 texSize = uint2(prevFrontTex.get_width(), prevFrontTex.get_height());
  uint2 pixel = min(uint2(in.position.xy), texSize - 1);
  float4 prevFront = prevFrontTex.read(pixel);
  float2 prevDepth = prevDepthTex.read(pixel).rg;
  
  float minDepth = -prevDepth.x;
  float maxDepth = prevDepth.y;
  float fragDepth = in.position.z;
  float epsilon = 0.0000001;

  PeelPassOutput out;
  out.backTemp = float4(0.0);
  out.frontDest = prevFront;
  out.depthDest = float2(-1.0, -1.0);

  // Early depth out to prevent highly expensive texture / lighting recalculations for overlapping zones
  if (fragDepth < minDepth - epsilon || fragDepth > maxDepth + epsilon) return out;
  if (fragDepth > minDepth + epsilon && fragDepth < maxDepth - epsilon) {
    out.depthDest = float2(-fragDepth, fragDepth);
    return out;
  }

  float3 N = normalize(in.viewNormal);
  ResolvedMaterial r = resolveMaterial(material, scene, in.vertexColor, in.uv, actorTexture, actorSampler);
  float3 totalAmbient = material.ambientColor.w * r.ambient;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);

  computePhongLighting(N, in.viewPos, r.diffuse, material.specularColor.rgb, material.specularColor.w, material.specularPower, lights, totalDiffuse, totalSpecular);

  float3 fragRGB = totalAmbient + material.diffuseColor.w * totalDiffuse + totalSpecular;

  if (fragDepth >= minDepth - epsilon && fragDepth <= minDepth + epsilon) {
    float prevAlpha = 1.0 - prevFront.a;
    out.frontDest.rgb = prevAlpha * r.opacity * fragRGB + prevFront.rgb;
    out.frontDest.a = 1.0 - (prevAlpha * (1.0 - r.opacity));
  } else if (fragDepth >= maxDepth - epsilon && fragDepth <= maxDepth + epsilon) {
    out.backTemp = float4(fragRGB * r.opacity, r.opacity);
  }

  return out;
}

fragment float4 fragment_peel_alpha_blend(
    VertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(5)]],
    texture2d<float> actorTexture [[texture(0)]],
    sampler actorSampler [[sampler(0)]],
    texture2d<float, access::read> prevDepthTex [[texture(2)]]) {
  
  if (isClipped(in.modelPos, clipPlanes)) discard_fragment();

  uint2 texSize = uint2(prevDepthTex.get_width(), prevDepthTex.get_height());
  uint2 pixel = min(uint2(in.position.xy), texSize - 1);
  float2 prevDepth = prevDepthTex.read(pixel).rg;
  float fragDepth = in.position.z;
  float epsilon = 0.0000001;

  if (fragDepth < -prevDepth.x - epsilon || fragDepth > prevDepth.y + epsilon) discard_fragment();

  float3 N = normalize(in.viewNormal);
  ResolvedMaterial r = resolveMaterial(material, scene, in.vertexColor, in.uv, actorTexture, actorSampler);
  float3 totalAmbient = material.ambientColor.w * r.ambient;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);

  computePhongLighting(N, in.viewPos, r.diffuse, material.specularColor.rgb, material.specularColor.w, material.specularPower, lights, totalDiffuse, totalSpecular);

  float3 fragRGB = totalAmbient + material.diffuseColor.w * totalDiffuse + totalSpecular;
  return float4(fragRGB * r.opacity, r.opacity);
}

fragment float4 fragment_peel_composite(
    FullscreenVertexOut in [[stage_in]],
    texture2d<float, access::read> frontTex [[texture(0)]],
    texture2d<float, access::read> backTex [[texture(1)]]) {
  
  uint2 pixel = uint2(in.position.xy);
  float4 front = frontTex.read(pixel);
  float4 back = backTex.read(pixel);

  float frontAlpha = 1.0 - front.a;
  return float4(front.rgb + back.rgb * frontAlpha, 1.0 - frontAlpha * (1.0 - back.a));
}

fragment float4 fragment_peel_back_blend(
    FullscreenVertexOut in [[stage_in]],
    texture2d<float, access::read> backTempTex [[texture(0)]]) {
  
  float4 backTemp = backTempTex.read(uint2(in.position.xy));
  if (backTemp.a < 0.001) discard_fragment();
  return backTemp;
}

// ---------------------------------------------------------------------------
// Glyph3D Shaders
// ---------------------------------------------------------------------------
struct GlyphVertexOut {
  float4 position [[position]];
  float3 viewPos;
  float3 viewNormal;
  float4 glyphColor;
  float3 modelPos;
  uint cellId;
  uint propId;
  uint compositeIndex;
};

// Point glyphs write [[point_size]], which is only valid for point topology.
// Metal rejects a pipeline whose vertex shader outputs point_size with a
// triangle/line primitive topology, so the point path uses a separate struct.
struct GlyphPointVertexOut {
  float4 position [[position]];
  float3 viewPos;
  float3 viewNormal;
  float4 glyphColor;
  float3 modelPos;
  uint cellId;
  uint propId;
  uint compositeIndex;
  float point_size [[point_size]];
};

template <typename T>
inline T computeGlyphVertex(
    uint vertex_id, uint instance_id,
    constant float3* positions, constant float3* normals,
    constant float4x4* glyphTransforms, constant float3x3* glyphNormalTransforms,
    constant float4* glyphColors, constant uint* glyphPickIds,
    constant SceneUniforms& scene, constant PickIds& pickIds)
{
  T out;
  float3 pos = positions[vertex_id];
  float4 worldPos = scene.modelMatrix * glyphTransforms[instance_id] * float4(pos, 1.0);
  out.viewPos = (scene.viewMatrix * worldPos).xyz;
  out.position = scene.projectionMatrix * float4(out.viewPos, 1.0);
  out.viewNormal = scene.normalMatrix * glyphNormalTransforms[instance_id] * normals[vertex_id];
  out.glyphColor = glyphColors[instance_id];
  out.cellId = glyphPickIds[instance_id] + 1u;
  out.propId = mapPropId(pickIds.propId);
  out.compositeIndex = pickIds.compositeIndex;
  out.modelPos = pos;
  return out;
}

template <typename T>
inline FragmentOutput shadeGlyphFragment(T in,
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
  out.ids = uint4(in.cellId, in.propId, in.compositeIndex, 0u);
  out.depth = in.position.z + depthBias;
  return out;
}

vertex GlyphVertexOut vertex_glyph_main(
    uint vertex_id [[vertex_id]], uint instance_id [[instance_id]],
    constant float3* positions [[buffer(0)]], constant float3* normals [[buffer(1)]],
    constant float4x4* glyphTransforms [[buffer(2)]], constant float3x3* glyphNormalTransforms [[buffer(3)]],
    constant float4* glyphColors [[buffer(4)]], constant uint* glyphPickIds [[buffer(5)]],
    constant SceneUniforms& scene [[buffer(8)]], constant ClipPlaneUniforms& clipPlanes [[buffer(9)]],
    constant PickIds& pickIds [[buffer(10)]]) {
  return computeGlyphVertex<GlyphVertexOut>(vertex_id, instance_id, positions, normals,
      glyphTransforms, glyphNormalTransforms, glyphColors, glyphPickIds, scene, pickIds);
}

fragment FragmentOutput fragment_glyph_main(
    GlyphVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(9)]]) {
  return shadeGlyphFragment(in, material, lights, clipPlanes, 0.0);
}

vertex GlyphVertexOut vertex_glyph_line_main(
    uint vertex_id [[vertex_id]], uint instance_id [[instance_id]],
    constant float3* positions [[buffer(0)]], constant float3* normals [[buffer(1)]],
    constant float4x4* glyphTransforms [[buffer(2)]], constant float3x3* glyphNormalTransforms [[buffer(3)]],
    constant float4* glyphColors [[buffer(4)]], constant uint* glyphPickIds [[buffer(5)]],
    constant SceneUniforms& scene [[buffer(8)]], constant ClipPlaneUniforms& clipPlanes [[buffer(9)]],
    constant PickIds& pickIds [[buffer(10)]]) {
  return computeGlyphVertex<GlyphVertexOut>(vertex_id, instance_id, positions, normals,
      glyphTransforms, glyphNormalTransforms, glyphColors, glyphPickIds, scene, pickIds);
}

fragment FragmentOutput fragment_glyph_line_main(
    GlyphVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(9)]]) {
  return shadeGlyphFragment(in, material, lights, clipPlanes, 0.0);
}

vertex GlyphPointVertexOut vertex_glyph_point_main(
    uint vertex_id [[vertex_id]], uint instance_id [[instance_id]],
    constant float3* positions [[buffer(0)]], constant float3* normals [[buffer(1)]],
    constant float4x4* glyphTransforms [[buffer(2)]], constant float3x3* glyphNormalTransforms [[buffer(3)]],
    constant float4* glyphColors [[buffer(4)]], constant uint* glyphPickIds [[buffer(5)]],
    constant SceneUniforms& scene [[buffer(8)]], constant ClipPlaneUniforms& clipPlanes [[buffer(9)]],
    constant PickIds& pickIds [[buffer(10)]]) {
  GlyphPointVertexOut out = computeGlyphVertex<GlyphPointVertexOut>(
      vertex_id, instance_id, positions, normals,
      glyphTransforms, glyphNormalTransforms, glyphColors, glyphPickIds, scene, pickIds);
  out.point_size = scene.pointSize;
  return out;
}

fragment FragmentOutput fragment_glyph_point_main(
    GlyphPointVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(9)]]) {
  return shadeGlyphFragment(in, material, lights, clipPlanes, coinOffset.pointOffset / 65000.0);
}


// Function constants for volume shader specialization.
// These are set via MTLFunctionConstantValues at pipeline creation time,
// allowing the Metal compiler to eliminate dead code for unused features.
// Each constant controls whether a specific expensive feature path is
// compiled into the fragment shader at all.
constant bool fc_shading [[function_constant(0)]];
constant bool fc_gradientOpacity [[function_constant(1)]];
constant bool fc_mask [[function_constant(2)]];
constant bool fc_minmax [[function_constant(3)]];
constant bool fc_normalTexture [[function_constant(4)]];

// ============================================================================
// Volume Ray Casting Mapper
// ============================================================================

struct VolumeMapperUniforms {
  float4x4 worldToVolume;
  float4x4 volumeToWorld;
  float4 volumeBoundsMin;
  float4 volumeBoundsMax;
  float4 cameraVolumePos;
  float4x4 viewProjection;
  half sampleDistance;
  half opacityPreIntegrationFactor; // unused; pre-integration baked into TF on CPU. Kept for struct layout.
  half scalarMin;
  half _pdSM;             // padding — was upper half of float scalarMin
  half scalarMax;
  half _pdSMax;           // padding — was upper half of float scalarMax
  float useJittering;
  float4x4 inverseViewProjection;
  float2 viewportSize;
  float3 gradientStep;
  float useGradientShading;
  float2 gradientOpacityRange;
  float useGradientOpacity;
  float4 ambientColor;
  float4 diffuseColor;
  float4 specularColor;
  float shininess;
  float3 lightDirection;
  float _pad2;
  float4 croppingPlanes;
  float4 croppingPlanes2;
  uint croppingBitmask;
  float _padCropFlags[31];
  float useCropping;
  float useClipping;
  float numClippingPlanes;
  float _padClipping[2];
  float4 clippingPlane0Origin;
  float4 clippingPlane0Normal;
  float4 clippingPlane1Origin;
  float4 clippingPlane1Normal;
  float4 clippingPlane2Origin;
  float4 clippingPlane2Normal;
  float4 clippingPlane3Origin;
  float4 clippingPlane3Normal;
  float4 clippingPlane4Origin;
  float4 clippingPlane4Normal;
  float4 clippingPlane5Origin;
  float4 clippingPlane5Normal;
  float4 clippingPlane6Origin;
  float4 clippingPlane6Normal;
  float4 clippingPlane7Origin;
  float4 clippingPlane7Normal;
  float useMask;
  float maskBlendFactor;
  float maskScale;
  float maskBias;
  float labelMapNumLabels;
  float useDepthTexture;
  float useNormalTexture;
  float _padMask;
  // Min-max acceleration texture
  float useMinMaxAccel;
  float minMaxDimX;
  float minMaxDimY;
  float minMaxDimZ;
};

struct VolumeLight {
    float4 position;      // xyz = position (positional lights), w = type (0=directional, 1=positional)
    float4 direction;     // xyz = direction (directional) or focal-point direction (spot), w = cone angle (degrees)
    float4 ambientColor;  // rgb * intensity
    float4 diffuseColor;  // rgb * intensity
    float4 specularColor; // rgb * intensity
    float4 attenuation;   // x=constant, y=linear, z=quadratic, w=spot exponent
};

struct VolumeLightUniforms {
    VolumeLight lights[MAX_LIGHTS]; // 8 * 96 = 768 bytes
    int lightCount;                 // 4 bytes
    int numPositionalLights;        // 4 bytes (informational, not used in shader)
    int twoSidedLighting;           // 4 bytes
    int defaultLighting;            // 4 bytes
    int _pad[4];                    // 16 bytes; total = 800 (must match C++ VolumeLightUniforms)
};

struct VolumeVertexOut {
  float4 position [[position]];
  float3 localPos;
  uint instanceID [[flat]];
};

struct VolumeVertexIn {
  float3 position [[attribute(0)]];
};

struct PerBlockData {
  float4 volumeBoundsMin;
  float4 volumeBoundsMax;
  float4 textureBoundsMin;
  float4 textureBoundsMax;
  float4 gradientStep;  // xyz + pad
  float4 minMaxInfo;    // useMinMax, dimX, dimY, dimZ
};

vertex VolumeVertexOut vertex_volume_main(
    VolumeVertexIn in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]]) {
  VolumeVertexOut out;

  // in.position is a unit cube [0,1]. Scale it to the block's model-space bounds.
  float3 modelPos = b.volumeBoundsMin.xyz + in.position * (b.volumeBoundsMax.xyz - b.volumeBoundsMin.xyz);
  out.position = volumeUniforms.viewProjection * volumeUniforms.volumeToWorld * float4(modelPos, 1.0);
  out.localPos = (modelPos - volumeUniforms.volumeBoundsMin.xyz) / max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  out.instanceID = 0;
  return out;
}

struct VolumeFragmentOut { float4 color [[color(0)]]; };

constant int MAX_RAY_STEPS = 8192;

inline float volume_random(float2 st) {
  // Interleaved Gradient Noise (Jimenez 2014). Smooth, low‑discrepancy,
  // deterministic per pixel, no sin‑hash streaks, no texture required.
  // For a regular sampling grid this shifts the grid phase smoothly across
  // the screen, which breaks banding *without* adding white‑noise grain —
  // usually the most pleasing look for single‑sample (no‑TAA) volume rendering.
  //
  // OpenGL equivalent (vtkOpenGLGPUVolumeRayCastMapper): samples a pre-filled
  // tiled noise texture (in_noiseSampler bound from win->GetNoiseTextureUnit())
  // via fragment_coord / texture_size.  A tiled field avoids the directional
  // correlation and GPU precision drift of a sin-based hash.  IGN matches the
  // quality of a texture at zero asset cost.
  //
  // To match GL exactly, replace this function with a tiled noise texture:
  //
  //   constexpr sampler sNoise(filter::nearest, address::repeat);
  //   inline float sampleJitterNoise(texture2d<float> t, float2 st) {
  //     float2 uv = st / float2(t.get_width(), t.get_height());
  //     return t.sample(sNoise, uv, level(0)).r;
  //   }
  //
  // Shader side: add noiseTexture [[texture(9)]] to the three volume fragment
  // entry points and pass to marchVolume. Replace the sample sites:
  //   marchVolume:    sampleJitterNoise(noiseTexture, screenPos)
  //   grid traversal: sampleJitterNoise(noiseTexture, in.position.xy)
  // Then delete this function.
  //
  // .mm side: create 128×128 R16Unorm in SetupPipeline (deterministic mt19937,
  // seed 0xC0FFEE), bind at index 9 in BindEncoderResources and
  // BindFullscreenTextures, release in ReleaseGraphicsResources. Provide a 1×1
  // fallback (0.5) so nil binds never occur.
  return fract(52.9829189 * fract(dot(st, float2(0.06711056, 0.00583715))));
}

inline float safeRecip(float x) {
  return 1.0 / (abs(x) < 1e-8 ? copysign(1e-8, x) : x);
}

inline float2 intersectBox(float3 orig, float3 dir, float3 boxMin, float3 boxMax) {
  float3 invDir = float3(safeRecip(dir.x), safeRecip(dir.y), safeRecip(dir.z));
  float3 tbot = invDir * (boxMin - orig);
  float3 ttop = invDir * (boxMax - orig);
  float3 tmin = min(ttop, tbot);
  float3 tmax = max(ttop, tbot);
  return float2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

// rayDirNormSpace is normalized in [0,1] volume space. Returns the normalized-space
// step that corresponds to a constant *physical* sample distance along that ray.
// Without this, the physical step is direction-dependent (<= sampleDistance), so the
// pre-integration factor (which assumes a full sampleDistance per step) over-accumulates
// opacity and the volume renders less translucent than the OpenGL backend.
// NOTE: boundsSize clamps degenerate axes to 1e-6 (matching computeVolumeBounds),
// whereas the CPU clamps vb.Size to 1.0. This never matters in practice because
// maxBound selects the largest axis (always >= 1). The tiny clamp on degenerate axes
// ensures physPerNorm stays non-zero for the division.
inline float physicalSampleStep(float3 rayDirNormSpace,
                                constant VolumeMapperUniforms& u)
{
  float3 boundsSize = max(u.volumeBoundsMax.xyz - u.volumeBoundsMin.xyz, 1e-6);
  float  maxBound   = max(boundsSize.x, max(boundsSize.y, boundsSize.z));
  float  physPerNorm = length(rayDirNormSpace * boundsSize);
  return float(u.sampleDistance) * maxBound / max(physPerNorm, 1e-6);
}

// Optimized: Gradient fetch with direction correction for anisotropic spacing.
// gradScale = 1 / (gradientStep * texSizeGlobal) converts raw central-difference
// components from texture-local to normalized-volume space.
inline half4 computeGradientFast(texture3d<float> volTex, float3 pos,
                                 float3 gradStep, half3 gradScale, half gradNormFactor) {
  half sPX = half(volTex.sample(sVolume, pos + float3(gradStep.x, 0, 0), level(0)).r);
  half sNX = half(volTex.sample(sVolume, pos - float3(gradStep.x, 0, 0), level(0)).r);
  half sPY = half(volTex.sample(sVolume, pos + float3(0, gradStep.y, 0), level(0)).r);
  half sNY = half(volTex.sample(sVolume, pos - float3(0, gradStep.y, 0), level(0)).r);
  half sPZ = half(volTex.sample(sVolume, pos + float3(0, 0, gradStep.z), level(0)).r);
  half sNZ = half(volTex.sample(sVolume, pos - float3(0, 0, gradStep.z), level(0)).r);

  half3 rawGrad = half3(sPX - sNX, sPY - sNY, sPZ - sNZ);

  half3 correctedGrad = rawGrad * gradScale;
  half mag = length(correctedGrad);
  half3 normal = mag > 0.0h ? correctedGrad / mag : half3(0.0h);

  return half4(normal, saturate(mag / gradNormFactor));
}

// Optimized: Pure FP16 math and fast::pow
inline half3 computePhongLightingVolumeFast(half3 sampleColor, half3 normal, half3 lightDir, half3 viewDir,
                                            half3 ambientMat, half3 diffuseMat, half3 specularMat, half shininess,
                                            bool twoSided = false) {
  half nDotL = dot(normal, -lightDir);
  if (nDotL < 0.0h && twoSided) {
    nDotL = -nDotL;
  }
  if (nDotL > 0.0h) {
    half3 diffuse = nDotL * diffuseMat * sampleColor;
    half3 r = normal * (2.0h * nDotL) + lightDir;
    half vDotR = max(dot(r, -viewDir), 0.0h);
    // fast::pow utilizes M2 hardware approximations, significantly faster than pow()
    half3 specular = fast::pow(vDotR, shininess) * specularMat;
    return ambientMat * sampleColor + diffuse + specular;
  }
  return ambientMat * sampleColor;
}

// Full multi-light volume shading. Loops over all active lights, accumulating
// ambient + diffuse + specular contributions. Handles directional, positional,
// and spot lights with attenuation. Matches OpenGL's ComputeLightingDeclaration.
inline half3 computeVolumeLighting(
    half3 sampleColor,
    half3 normal,
    half3 viewDir,           // normalized, pointing toward camera (in volume space)
    half3 ambientMat,
    half3 diffuseMat,
    half3 specularMat,
    half shininess,
    constant VolumeLightUniforms& lightUniforms,
    float3 fragPosVolume)    // current sample position in [0,1] volume space
{
    half3 totalAmbient  = half3(0.0h);
    half3 totalDiffuse  = half3(0.0h);
    half3 totalSpecular = half3(0.0h);

    int numLights = lightUniforms.lightCount;
    bool twoSided = lightUniforms.twoSidedLighting != 0;

    for (int i = 0; i < numLights && i < MAX_LIGHTS; ++i) {
        constant VolumeLight& L = lightUniforms.lights[i];

        half3 lightAmbient  = half3(L.ambientColor.rgb);
        half3 lightDiffuse  = half3(L.diffuseColor.rgb);
        half3 lightSpecular = half3(L.specularColor.rgb);

        half3 toLight;
        half attenuation = 1.0h;

        if (L.position.w < 0.5) {
            // Directional light: direction is pre-normalized in volume space
            toLight = half3(-L.direction.xyz);
        } else {
            // Positional light: compute direction from fragment to light
            half3 lightPos = half3(L.position.xyz);
            half3 delta = lightPos - half3(fragPosVolume);
            half dist = length(delta);
            toLight = dist > 0.0001h ? delta / dist : half3(0.0h, 0.0h, 1.0h);

            // Attenuation: 1 / (constant + linear*d + quadratic*d^2)
            half attenDenom = half(L.attenuation.x)
                            + half(L.attenuation.y) * dist
                            + half(L.attenuation.z) * dist * dist;
            attenuation = attenDenom > 0.0h ? 1.0h / attenDenom : 0.0h;

            // Spot light cone check
            if (L.direction.w <= 90.0) {
                half spotCos = dot(-toLight, half3(normalize(L.direction.xyz)));
                half spotCutoff = half(cos(float(L.direction.w) * (M_PI_F / 180.0)));
                if (spotCos < spotCutoff) {
                    attenuation = 0.0h;
                } else {
                    attenuation *= fast::pow(spotCos, half(L.attenuation.w));
                }
            }
        }

        // Diffuse
        half nDotL = dot(normal, toLight);
        if (nDotL < 0.0h && twoSided) {
            nDotL = -nDotL;
        }
        if (nDotL > 0.0h) {
            totalDiffuse += nDotL * lightDiffuse * attenuation;

            // Phong reflection vector (matches OpenGL's ComputeLightingDeclaration
            // and computePhongLightingVolumeFast)
            half3 r = normalize(normal * (2.0h * nDotL) - toLight);
            half vDotR = dot(viewDir, r);
            if (vDotR < 0.0h && twoSided) {
                vDotR = -vDotR;
            }
            if (vDotR > 0.0h) {
                totalSpecular += fast::pow(vDotR, shininess) * lightSpecular * attenuation;
            }
        }

        // Ambient (always accumulates)
        totalAmbient += lightAmbient;
    }

    return ambientMat * totalAmbient * sampleColor
         + diffuseMat * totalDiffuse * sampleColor
         + specularMat * totalSpecular;
}

// Branchless, fast crop region evaluator (returns 0..26 matching VTK region bit convention)
inline int computeCropRegion(float3 cropMin, float3 cropMax, float3 pos) {
  int3 r = int3(step(cropMin, pos)) + int3(step(cropMax, pos));
  return r.x + r.y * 3 + r.z * 9;
}

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

struct RaySetup {
    float3 entryPoint;
    float3 exitPoint;
    float3 rayDir;
    float totalDist;
    float tTerminateMax;
    float totalBoxT;
    bool valid;
};

inline void computeVolumeBounds(
    constant PerBlockData& b,
    constant VolumeMapperUniforms& volumeUniforms,
    thread float3& blockMinGlobal,
    thread float3& blockMaxGlobal,
    thread float3& texMinGlobal,
    thread float3& texMaxGlobal)
{
    float3 bsz = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
    blockMinGlobal = (b.volumeBoundsMin.xyz - volumeUniforms.volumeBoundsMin.xyz) / bsz;
    blockMaxGlobal = (b.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz) / bsz;
    texMinGlobal = (b.textureBoundsMin.xyz - volumeUniforms.volumeBoundsMin.xyz) / bsz;
    texMaxGlobal = (b.textureBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz) / bsz;
}

inline RaySetup setupVolumeRay(
    float3 cameraPos,
    float3 rayDir,
    float3 blockMinGlobal,
    float3 blockMaxGlobal,
    float2 screenPos,
    float2 viewportSize,
    constant VolumeMapperUniforms& volumeUniforms,
    texture2d<float> depthTexture)
{
    RaySetup s;
    s.rayDir = rayDir;
    s.valid = false;

    float2 t = intersectBox(cameraPos, rayDir, blockMinGlobal, blockMaxGlobal);
    float tStart = max(t.x, 0.0);
    if (tStart >= t.y) return s;

    s.entryPoint = cameraPos + rayDir * tStart;
    s.exitPoint = cameraPos + rayDir * t.y;
    s.totalDist = length(s.exitPoint - s.entryPoint);
    s.totalBoxT = t.y - tStart;

    if (volumeUniforms.useClipping > 0.5) {
        int numClipPlanes = int(volumeUniforms.numClippingPlanes);
        float4 planeOrigins[8] = { volumeUniforms.clippingPlane0Origin, volumeUniforms.clippingPlane1Origin, volumeUniforms.clippingPlane2Origin, volumeUniforms.clippingPlane3Origin, volumeUniforms.clippingPlane4Origin, volumeUniforms.clippingPlane5Origin, volumeUniforms.clippingPlane6Origin, volumeUniforms.clippingPlane7Origin };
        float4 planeNormals[8] = { volumeUniforms.clippingPlane0Normal, volumeUniforms.clippingPlane1Normal, volumeUniforms.clippingPlane2Normal, volumeUniforms.clippingPlane3Normal, volumeUniforms.clippingPlane4Normal, volumeUniforms.clippingPlane5Normal, volumeUniforms.clippingPlane6Normal, volumeUniforms.clippingPlane7Normal };

        #pragma unroll
        for (int cp = 0; cp < numClipPlanes; cp++) {
            float3 planeOrigin = planeOrigins[cp].xyz;
            float3 planeNormal = normalize(planeNormals[cp].xyz);
            float startDistance = dot(planeNormal, planeOrigin - s.entryPoint);
            float stopDistance = dot(planeNormal, planeOrigin - s.exitPoint);

            if (startDistance > 0.0 && stopDistance > 0.0) return s;

            float rayDotNormal = dot(rayDir, planeNormal);
            if (rayDotNormal > 0.0 && startDistance > 0.0) s.entryPoint += (startDistance / rayDotNormal) * rayDir;
            if (rayDotNormal <= 0.0 && stopDistance > 0.0) s.exitPoint += (stopDistance / rayDotNormal) * rayDir;
        }
        s.totalDist = length(s.exitPoint - s.entryPoint);
        if (s.totalDist < 1e-6) return s;
    }

    s.tTerminateMax = 1e30;
    if (volumeUniforms.useDepthTexture > 0.5) {
        float depthSample = depthTexture.sample(sNearest, screenPos / viewportSize).r;
        if (depthSample < 1.0) {
            float2 ndcXY = (screenPos / viewportSize) * 2.0 - 1.0;
            float4 worldTermination = volumeUniforms.inverseViewProjection * float4(ndcXY.x, -ndcXY.y, depthSample, 1.0);
            worldTermination.xyz /= worldTermination.w;
            float3 terminationLocal = (worldTermination.xyz - volumeUniforms.volumeBoundsMin.xyz) / max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
            s.tTerminateMax = dot(terminationLocal - s.entryPoint, rayDir);
            if (s.tTerminateMax <= 0.0) return s;
        }
    }

    s.valid = true;
    return s;
}

struct MarchParams {
    float3 rayOrigin;
    float3 rayDir;
    float  tStart;
    float  tEnd;
    float  stepSize;
    float  jitter;
    float  tTerminateMax;
    float3 blockMinGlobal;
    float3 blockMaxGlobal;
    float3 texMinGlobal;
    float3 texMaxGlobal;
    bool   checkBounds;
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
  const bool doShading = fc_shading && (volumeUniforms.useGradientShading > 0.5);
  const bool doGradOp = fc_gradientOpacity && (volumeUniforms.useGradientOpacity > 0.5);
  const bool doCropping = volumeUniforms.useCropping > 0.5;
  const bool doMask = fc_mask && (volumeUniforms.useMask > 0.5);

  half scalarScale = half(1.0 / max((volumeUniforms.scalarMax - volumeUniforms.scalarMin), 1e-4h));
  half scalarBias  = half(-volumeUniforms.scalarMin) * scalarScale;

  half gradNormFactor = half(max(1e-8f, volumeUniforms.gradientOpacityRange.y));

  float3 texSizeGlobal = max(p.texMaxGlobal - p.texMinGlobal, 1e-6);
  float3 invTexSizeGlobal = 1.0 / texSizeGlobal;
  float3 rayDirTexLocal = p.rayDir * invTexSizeGlobal;
  float3 dt = max(b.gradientStep.xyz, 1e-8);
  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz
                        - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  half3 gradScale = half3(1.0 / (dt * texSizeGlobal * boundsSize));

  half3 viewDirHalf  = half3(normalize((p.rayOrigin + p.rayDir * p.tStart) - volumeUniforms.cameraVolumePos.xyz));
  half3 lightDirHalf = half3(normalize(volumeUniforms.lightDirection));
  half3 ambientMat   = half3(volumeUniforms.ambientColor.rgb);
  half3 diffuseMat   = half3(volumeUniforms.diffuseColor.rgb);
  half3 specularMat  = half3(volumeUniforms.specularColor.rgb);
  half shininessMat  = half(volumeUniforms.shininess);

  float maskScale = volumeUniforms.maskScale;
  float maskBias  = volumeUniforms.maskBias;
  float numLabels = volumeUniforms.labelMapNumLabels;

  float3 cropMin = float3(volumeUniforms.croppingPlanes.x, volumeUniforms.croppingPlanes.z, volumeUniforms.croppingPlanes2.x);
  float3 cropMax = float3(volumeUniforms.croppingPlanes.y, volumeUniforms.croppingPlanes.w, volumeUniforms.croppingPlanes2.y);

  uint cropBitmask = volumeUniforms.croppingBitmask;

  float firstT = p.checkBounds
      ? p.jitter
      : p.jitter + ceil((p.tStart - p.jitter) / p.stepSize) * p.stepSize;
  float3 stepVec = p.rayDir * p.stepSize;
  float3 currentPoint = p.rayOrigin + p.rayDir * (p.checkBounds ? p.tStart : 0.0)
                      + p.rayDir * firstT;
  float currentT = firstT;

  int maxSteps = min(max(1, int(ceil((p.tEnd - firstT) / p.stepSize))), MAX_RAY_STEPS);

  half3 accumulatedColor = initialColor;
  half accumulatedOpacity = initialOpacity;

  float3 texLocalPos0 = (currentPoint - p.texMinGlobal) * invTexSizeGlobal;
  float3 evalPoint0 = texLocalPos0;
  float prefetchScalar = volumeTexture.sample(sVolume, evalPoint0, level(0)).r;
  float prefetchMask = doMask ? maskTexture.sample(sNearest, evalPoint0, level(0)).r : 0.0;
  bool prefetchValid = true;
  int3  curCell     = int3(-1);
  bool  curCellEmpty = false;
  float3 mmDimF     = b.minMaxInfo.yzw;
  const bool useMinMax = fc_minmax &&
    b.minMaxInfo.x > 0.5 &&
    b.minMaxInfo.y > 0.5 &&
    b.minMaxInfo.z > 0.5 &&
    b.minMaxInfo.w > 0.5;

  for (int i = 0; i < maxSteps; i++) {
    if (p.checkBounds && (any(currentPoint < p.blockMinGlobal - 1e-4) || any(currentPoint > p.blockMaxGlobal + 1e-4))) break;
    if (!p.checkBounds && currentT >= p.tEnd - 1e-6) break;

    if (useMinMax) {
      float3 texLocalPos = (currentPoint - p.texMinGlobal) * invTexSizeGlobal;
      float3 mmPos = clamp(texLocalPos, float3(0.0), float3(1.0));
      int3 newCell = min(int3(mmPos * mmDimF), int3(mmDimF) - 1);
      if (any(newCell != curCell)) {
        curCell      = newCell;
        curCellEmpty = minMaxTexture.sample(sNearest, mmPos, level(0)).r > 0.5;
      }

      if (curCellEmpty) {
        float3 cellCoord = mmPos * mmDimF;
        float3 fractCoord = fract(cellCoord);

        float3 distToEdge;
        distToEdge.x = p.rayDir.x > 0.0 ? (1.0 - fractCoord.x) : fractCoord.x;
        distToEdge.y = p.rayDir.y > 0.0 ? (1.0 - fractCoord.y) : fractCoord.y;
        distToEdge.z = p.rayDir.z > 0.0 ? (1.0 - fractCoord.z) : fractCoord.z;
        distToEdge = mix(distToEdge, float3(1.0), float3(distToEdge <= 1e-5));

        float3 tToEdge;
        tToEdge.x = abs(rayDirTexLocal.x) > 1e-5 ? distToEdge.x / abs(rayDirTexLocal.x * mmDimF.x) : 1e30;
        tToEdge.y = abs(rayDirTexLocal.y) > 1e-5 ? distToEdge.y / abs(rayDirTexLocal.y * mmDimF.y) : 1e30;
        tToEdge.z = abs(rayDirTexLocal.z) > 1e-5 ? distToEdge.z / abs(rayDirTexLocal.z * mmDimF.z) : 1e30;

        float exactSkip = min(min(tToEdge.x, tToEdge.y), tToEdge.z);
        exactSkip += 1e-4;
        float skipDist = ceil(exactSkip / p.stepSize) * p.stepSize;
        skipDist = max(p.stepSize, skipDist);

        currentPoint += p.rayDir * skipDist;
        currentT += skipDist;

        if (p.checkBounds && (any(currentPoint < p.blockMinGlobal - 1e-4) || any(currentPoint > p.blockMaxGlobal + 1e-4) || currentT >= p.tEnd)) {
          break;
        }

        prefetchValid = false;
        curCell = int3(-1);
        continue;
      }
    }

    float3 texLocalPos = (currentPoint - p.texMinGlobal) * invTexSizeGlobal;
    float3 evalPoint = texLocalPos;
    bool needsFetch = !prefetchValid;
    float rawScalar = needsFetch
      ? volumeTexture.sample(sVolume, evalPoint, level(0)).r
      : prefetchScalar;
    float rawMask = (doMask && needsFetch)
      ? maskTexture.sample(sNearest, evalPoint, level(0)).r
      : prefetchMask;

    float3 lastPoint = currentPoint;
    currentPoint += stepVec;
    currentT += p.stepSize;

    if (i + 1 < maxSteps) {
      float3 nextTexLocalPos = (currentPoint - p.texMinGlobal) * invTexSizeGlobal;
      float3 nextEvalPoint = nextTexLocalPos;
      prefetchScalar = volumeTexture.sample(sVolume, nextEvalPoint, level(0)).r;
      if (doMask) {
        prefetchMask = maskTexture.sample(sNearest, nextEvalPoint, level(0)).r;
      }
      prefetchValid = true;
    }

    if (doCropping && ((cropBitmask & (1u << computeCropRegion(cropMin, cropMax, lastPoint))) == 0u)) {
      continue;
    }

    half scalarNorm = saturate(half(rawScalar) * scalarScale + scalarBias);

    half4 colorOpacity;
    half maskLabel = 0.0h;

    if (doMask) {
      float maskVal = rawMask * maskScale + maskBias;
      if (numLabels > 0.0) {
        float label = floor(maskVal + 0.5);
        if (label > 0.0) {
          label = clamp(label, 1.0, numLabels - 1.0);
          maskLabel = half(label);
          float labelY = (label + 0.5) / numLabels;
          colorOpacity = half4(labelMapTransferTexture.sample(sNearest, float2(float(scalarNorm), labelY), level(0)));
        } else {
          colorOpacity = half4(transferFunctionTexture.sample(sVolume, float2(float(scalarNorm), 0.5), level(0)));
        }
      } else {
        colorOpacity = half4(transferFunctionTexture.sample(sVolume, float2(float(scalarNorm), 0.5), level(0)));
      }
    } else {
      colorOpacity = half4(transferFunctionTexture.sample(sVolume, float2(float(scalarNorm), 0.5), level(0)));
    }

    half sampleOpacity = colorOpacity.a;
    // Opacity pre-integration is baked into the transfer function texture
    // on the CPU at TF-build time (matches OpenGL backend).

    if (sampleOpacity > 0.001h) {
      half3 sampleColor = colorOpacity.rgb;
      half weight = 1.0h - accumulatedOpacity;

      if (sampleOpacity < 0.01h) {
        accumulatedColor += weight * sampleColor * sampleOpacity;
        accumulatedOpacity += weight * sampleOpacity;
      } else {

      if (doShading && maskLabel == 0.0h && (sampleOpacity * weight > 0.002h)) {

        half3 normal;
        half gradMag;

        if (fc_normalTexture && volumeUniforms.useNormalTexture > 0.5) {
          half4 nrmSample = half4(normalTexture.sample(sVolume, evalPoint, level(0)));
          normal = normalize(nrmSample.xyz * 2.0h - 1.0h);
          gradMag = nrmSample.w;
        } else {
          half4 grad = computeGradientFast(volumeTexture, evalPoint, b.gradientStep.xyz, gradScale, gradNormFactor);
          normal = grad.xyz;
          gradMag = grad.w;
        }

        if (lightUniforms != nullptr && lightUniforms->defaultLighting == 0) {
          sampleColor = computeVolumeLighting(sampleColor, normal, -viewDirHalf,
              ambientMat, diffuseMat, specularMat, shininessMat,
              *lightUniforms, evalPoint);
        } else {
          bool twoSided = (lightUniforms != nullptr && lightUniforms->twoSidedLighting != 0);
          sampleColor = computePhongLightingVolumeFast(sampleColor, normal, lightDirHalf, viewDirHalf,
              ambientMat, diffuseMat, specularMat, shininessMat, twoSided);
        }

        if (doGradOp) {
          sampleOpacity *= half(gradientOpacityTexture.sample(sVolume, float2(float(gradMag), 0.5), level(0)).r);
        }
      } else if (doShading) {
        sampleColor = ambientMat * sampleColor;
      }

      accumulatedColor += weight * sampleColor * sampleOpacity;
      accumulatedOpacity += weight * sampleOpacity;
      }
    }

    if (accumulatedOpacity >= 0.99h) {
      accumulatedOpacity = 1.0h;
      break;
    }
    if (currentT >= p.tTerminateMax) {
      break;
    }
  }

  return half4(accumulatedColor, accumulatedOpacity);
}

inline half4 marchVolume(
    float3 entryPoint,
    float3 exitPoint,
    float totalDist,
    float tTerminateMax,
    float3 rayDir,
    float3 blockMinGlobal,
    float3 blockMaxGlobal,
    float3 texMinGlobal,
    float3 texMaxGlobal,
    float3 cameraPos,
    float stepSize,
    float totalBoxT,
    float2 screenPos,
    half3 initialColor,
    half initialOpacity,
    constant VolumeMapperUniforms& volumeUniforms,
    constant PerBlockData& b,
    texture3d<float> volumeTexture,
    texture2d<float> transferFunctionTexture,
    texture2d<float> depthTexture,
    texture2d<float> gradientOpacityTexture,
    texture3d<float> maskTexture,
    texture2d<float> labelMapTransferTexture,
    texture3d<float> minMaxTexture,
    texture3d<float> normalTexture,
    constant VolumeLightUniforms* lightUniforms)
{
  (void)exitPoint;
  (void)totalDist;
  float jitter = volumeUniforms.useJittering > 0.5 ? volume_random(screenPos) * stepSize : 0.0;
  float tStart = dot(entryPoint - cameraPos, rayDir);
  MarchParams p = {cameraPos, rayDir, tStart, totalBoxT, stepSize, jitter, tTerminateMax,
      blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal, true};
  return marchVolumeUnified(p, initialColor, initialOpacity,
      volumeUniforms, b, volumeTexture, transferFunctionTexture,
      gradientOpacityTexture, maskTexture, labelMapTransferTexture,
      minMaxTexture, normalTexture, lightUniforms);
}

inline void marchSegment(
    float3 rayOrigin,
    float3 rayDir,
    float t0,
    float t1,
    float stepSize,
    float jitter,
    float tTerminateMax,
    thread half3& accumulatedColor,
    thread half& accumulatedOpacity,
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
  float3 zero = float3(0.0);
  float3 one = float3(1.0);
  MarchParams p = {rayOrigin, rayDir, t0, t1, stepSize, jitter, tTerminateMax,
      zero, one, zero, one, false};
  half4 result = marchVolumeUnified(p, accumulatedColor, accumulatedOpacity,
      volumeUniforms, b, volumeTexture, transferFunctionTexture,
      gradientOpacityTexture, maskTexture, labelMapTransferTexture,
      minMaxTexture, normalTexture, lightUniforms);
  accumulatedColor = result.xyz;
  accumulatedOpacity = result.w;
}

fragment VolumeFragmentOut fragment_volume_main(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> transferFunctionTexture [[texture(1)]],
    texture2d<float> depthTexture [[texture(2)]],
    texture2d<float> gradientOpacityTexture [[texture(3)]],
    texture3d<float> maskTexture [[texture(4)]],
    texture2d<float> labelMapTransferTexture [[texture(5)]],
    texture3d<float> minMaxTexture [[texture(6)]],
    texture3d<float> normalTexture [[texture(7)]],
    constant VolumeLightUniforms& volumeLights [[buffer(4)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  float3 rayDir = in.localPos - cameraPos;
  float dirLength = length(rayDir);
  if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
  rayDir /= dirLength;

  RaySetup s = setupVolumeRay(cameraPos, rayDir, blockMinGlobal, blockMaxGlobal,
      in.position.xy, volumeUniforms.viewportSize, volumeUniforms, depthTexture);
  if (!s.valid) { output.color = float4(0.0); return output; }

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  half4 _marchResult = marchVolume(s.entryPoint, s.exitPoint, s.totalDist, s.tTerminateMax, rayDir,
      blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal, cameraPos,
      stepSize, s.totalBoxT, in.position.xy,
      half3(0.0), 0.0h, volumeUniforms, b,
      volumeTexture, transferFunctionTexture, depthTexture, gradientOpacityTexture,
      maskTexture, labelMapTransferTexture, minMaxTexture, normalTexture,
      &volumeLights);
  output.color = float4(float3(_marchResult.xyz), float(_marchResult.w));
  return output;
}

// Fullscreen volume ray-cast shader: used when the camera is inside the volume.
// Reconstructs the ray from screen UV using inverseViewProjection (same approach
// as the composite shader) instead of relying on proxy-geometry vertices.
// This eliminates the CPU-heavy ClipConvexPolyData + DensifyPolyData + TriangleFilter
// pipeline that the proxy-based path requires for camera-inside rendering.
// The march loop is identical to fragment_volume_main.
fragment VolumeFragmentOut fragment_volume_fullscreen_main(
    FullscreenVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> transferFunctionTexture [[texture(1)]],
    texture2d<float> depthTexture [[texture(2)]],
    texture2d<float> gradientOpacityTexture [[texture(3)]],
    texture3d<float> maskTexture [[texture(4)]],
    texture2d<float> labelMapTransferTexture [[texture(5)]],
    texture3d<float> minMaxTexture [[texture(6)]],
    texture3d<float> normalTexture [[texture(7)]],
    constant VolumeLightUniforms& volumeLights [[buffer(4)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  float3 rayDir = reconstructRayDir(in.position.xy, volumeUniforms.viewportSize, volumeUniforms);

  RaySetup s = setupVolumeRay(cameraPos, rayDir, blockMinGlobal, blockMaxGlobal,
      in.position.xy, volumeUniforms.viewportSize, volumeUniforms, depthTexture);
  if (!s.valid) { output.color = float4(0.0); return output; }

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  half4 _marchResult = marchVolume(s.entryPoint, s.exitPoint, s.totalDist, s.tTerminateMax, rayDir,
      blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal, cameraPos,
      stepSize, s.totalBoxT, in.position.xy,
      half3(0.0), 0.0h, volumeUniforms, b,
      volumeTexture, transferFunctionTexture, depthTexture, gradientOpacityTexture,
      maskTexture, labelMapTransferTexture, minMaxTexture, normalTexture,
      &volumeLights);
  output.color = float4(float3(_marchResult.xyz), float(_marchResult.w));
  return output;
}

// ============================================================================
// Grid traversal for partitioned volume rendering.
// Single-pass front-to-back brick grid traversal using 3D DDA.
// ============================================================================

struct GridTraversalUniforms {
    int gridDimsX;          // nx
    int gridDimsY;          // ny
    int gridDimsZ;          // nz
    int _pad;
};

// 3D DDA grid walker for axis-aligned brick grid traversal.
// Operates in normalized [0,1] volume space. Each grid cell
// occupies (1/nx) x (1/ny) x (1/nz) in that space.
struct GridWalker {
    int3 cell;              // current brick index
    float tEntry;           // ray t at entry into current cell
    float tExit;            // ray t at exit from current cell
    int3 step;              // traversal direction (+1 or -1) per axis
    float3 tMax;            // ray t of next boundary crossing per axis
    float3 tDelta;          // ray t delta to cross one cell per axis
    int3 gridDims;
    bool valid;
};

inline GridWalker initGridWalker(
    float3 rayOrigin,
    float3 rayDir,
    float tStart,
    float tEndRel,
    int3 gridDims)
{
    GridWalker w;
    w.gridDims = gridDims;

    float3 pos = rayOrigin + rayDir * tStart;
    float3 cellF = clamp(pos, 0.0, 1.0) * float3(gridDims);

    // Nudge into the correct cell when pos sits exactly on a boundary
    // and the ray heads in the negative direction (floor puts us in the
    // wrong cell otherwise).
    float3 nudge = float3(
        rayDir.x < 0.0 ? 1e-6 : 0.0,
        rayDir.y < 0.0 ? 1e-6 : 0.0,
        rayDir.z < 0.0 ? 1e-6 : 0.0);
    w.cell = clamp(int3(floor(cellF - nudge)), int3(0), gridDims - 1);

    // Work in relative coordinates: tEntry = 0 means "at pos right now".
    if (tEndRel <= 1e-8) {
        w.valid = false;
        return w;
    }

    w.tEntry = 0.0;

    float3 invGridDims = 1.0 / float3(gridDims);

    for (int a = 0; a < 3; ++a)
    {
        float d = rayDir[a];
        if (abs(d) < 1e-8)
        {
            w.step[a]   = 0;
            w.tMax[a]   = 1e30;
            w.tDelta[a] = 1e30;
        }
        else
        {
            w.step[a] = (d > 0.0) ? 1 : -1;

            float boundary = (d > 0.0)
                ? float(w.cell[a] + 1) * invGridDims[a]
                : float(w.cell[a])     * invGridDims[a];

            w.tMax[a]   = (boundary - pos[a]) / d;
            w.tDelta[a] = invGridDims[a] / abs(d);

            if (w.tMax[a] <= 0.0)
                w.tMax[a] = w.tDelta[a] * 1e-6;
        }
    }

    float minT = min(min(w.tMax.x, w.tMax.y), w.tMax.z);
    w.tExit = min(minT, tEndRel);
    w.valid = w.tExit > w.tEntry + 1e-8;

    return w;
}

inline void advanceGridWalker(thread GridWalker& w, float tEnd)
{
    int axis = 0;
    if (w.tMax.y < w.tMax.x) axis = 1;
    if (w.tMax.z < w.tMax[axis]) axis = 2;

    if (w.step[axis] == 0) { w.valid = false; return; }

    w.tEntry = w.tMax[axis];
    w.cell[axis] += w.step[axis];

    if (any(w.cell < int3(0)) || any(w.cell >= w.gridDims))
    {
        w.valid = false;
        return;
    }

    w.tMax[axis] += w.tDelta[axis];
    if (w.tMax[axis] <= w.tEntry)
        w.tMax[axis] = w.tEntry + w.tDelta[axis] * 1e-6;

    float minT = min(min(w.tMax.x, w.tMax.y), w.tMax.z);
    w.tExit = min(minT, tEnd);
    w.valid = w.tExit > w.tEntry;
}

// Single-pass front-to-back grid traversal shader for partitioned volumes.
// Replaces the per-brick layer composite with a direct traversal of the
// brick grid along each pixel ray, marching active bricks in true geometric order.
fragment VolumeFragmentOut fragment_volume_grid_traversal_main(
    FullscreenVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    constant GridTraversalUniforms& grid [[buffer(3)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> transferFunctionTexture [[texture(1)]],
    texture2d<float> depthTexture [[texture(2)]],
    texture2d<float> gradientOpacityTexture [[texture(3)]],
    texture3d<float> maskTexture [[texture(4)]],
    texture2d<float> labelMapTransferTexture [[texture(5)]],
    texture3d<float> minMaxTexture [[texture(6)]],
    texture3d<float> normalTexture [[texture(7)]],
    texture3d<float> brickOccupancy [[texture(8)]],
    constant VolumeLightUniforms& volumeLights [[buffer(4)]])
{
    VolumeFragmentOut output;

    float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;

    // Reconstruct ray in normalized [0,1] volume space
    float3 rayDir = reconstructRayDir(in.position.xy, volumeUniforms.viewportSize, volumeUniforms);

    // Intersect with full volume bounds [0,1]
    float3 volMin = float3(0.0);
    float3 volMax = float3(1.0);
    float2 tVol = intersectBox(cameraPos, rayDir, volMin, volMax);
    float tStart = max(tVol.x, 0.0);
    float tEnd = tVol.y;

    if (tStart >= tEnd - 1e-8) {
        output.color = float4(0.0);
        return output;
    }

    // Apply clipping planes
    if (volumeUniforms.useClipping > 0.5) {
        int numClipPlanes = int(volumeUniforms.numClippingPlanes);
        float4 planeOrigins[8] = {
            volumeUniforms.clippingPlane0Origin, volumeUniforms.clippingPlane1Origin,
            volumeUniforms.clippingPlane2Origin, volumeUniforms.clippingPlane3Origin,
            volumeUniforms.clippingPlane4Origin, volumeUniforms.clippingPlane5Origin,
            volumeUniforms.clippingPlane6Origin, volumeUniforms.clippingPlane7Origin
        };
        float4 planeNormals[8] = {
            volumeUniforms.clippingPlane0Normal, volumeUniforms.clippingPlane1Normal,
            volumeUniforms.clippingPlane2Normal, volumeUniforms.clippingPlane3Normal,
            volumeUniforms.clippingPlane4Normal, volumeUniforms.clippingPlane5Normal,
            volumeUniforms.clippingPlane6Normal, volumeUniforms.clippingPlane7Normal
        };

        float3 entryPoint = cameraPos + rayDir * tStart;
        float3 exitPoint = cameraPos + rayDir * tEnd;

        bool valid = true;
        for (int cp = 0; cp < numClipPlanes && cp < 8; cp++) {
            float3 planeOrigin = planeOrigins[cp].xyz;
            float3 planeNormal = normalize(planeNormals[cp].xyz);
            float startDistance = dot(planeNormal, planeOrigin - entryPoint);
            float stopDistance = dot(planeNormal, planeOrigin - exitPoint);

            if (startDistance > 0.0 && stopDistance > 0.0) { valid = false; break; }

            float rayDotNormal = dot(rayDir, planeNormal);
            if (rayDotNormal > 0.0 && startDistance > 0.0) {
                entryPoint += (startDistance / rayDotNormal) * rayDir;
            }
            if (rayDotNormal <= 0.0 && stopDistance > 0.0) {
                exitPoint += (stopDistance / rayDotNormal) * rayDir;
            }
        }

        if (!valid) {
            output.color = float4(0.0);
            return output;
        }

        // Recompute t from clipped entry/exit
        float3 d = exitPoint - entryPoint;
        tStart = dot(entryPoint - cameraPos, rayDir);
        tEnd = tStart + length(d);
    }

    // Depth occlusion termination
    float tTerminateMax = 1e30;
    if (volumeUniforms.useDepthTexture > 0.5) {
        float2 uv = in.position.xy / volumeUniforms.viewportSize;
        float depthSample = depthTexture.sample(sNearest, uv).r;
        if (depthSample < 1.0) {
            float2 ndc = uv * 2.0 - 1.0;
            float4 worldTermination = volumeUniforms.inverseViewProjection * float4(ndc.x, -ndc.y, depthSample, 1.0);
            worldTermination.xyz /= worldTermination.w;
            float3 terminationLocal = (worldTermination.xyz - volumeUniforms.volumeBoundsMin.xyz) / max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
            float tDepth = dot(terminationLocal - (cameraPos + rayDir * tStart), rayDir);
            if (tDepth <= 0.0) {
                output.color = float4(0.0);
                return output;
            }
            tTerminateMax = tStart + tDepth;
        }
    }

    tEnd = min(tEnd, tTerminateMax);
    if (tStart >= tEnd - 1e-8) {
        output.color = float4(0.0);
        return output;
    }

    // Global sample schedule
    float stepSize = physicalSampleStep(rayDir, volumeUniforms);
    float jitter = volumeUniforms.useJittering > 0.5
        ? volume_random(in.position.xy + float2(0.5, 0.5)) * stepSize
        : 0.0;

    // Grid traversal loop
    half3 color = 0.0h;
    half opacity = 0.0h;

    int3 gridDims = int3(grid.gridDimsX, grid.gridDimsY, grid.gridDimsZ);
    float tEndRel = tEnd - tStart;
    GridWalker walker = initGridWalker(cameraPos, rayDir, tStart, tEndRel, gridDims);

    int maxCells = gridDims.x + gridDims.y + gridDims.z + 3;
    int cellsVisited = 0;

    while (walker.valid && opacity < 0.99h && cellsVisited < maxCells) {
        ++cellsVisited;
        int3 cell = walker.cell;

        float3 occUV = (float3(cell) + 0.5) / float3(gridDims);
        float occ = brickOccupancy.sample(sNearest, occUV, level(0)).r;

        if (occ > 0.5) {
            float segmentT0 = tStart + max(walker.tEntry, 0.0);
            float segmentT1 = tStart + min(walker.tExit, tEndRel);

            if (segmentT1 > segmentT0 + 1e-8) {
                marchSegment(
                    cameraPos, rayDir,
                    segmentT0, segmentT1,
                    stepSize, jitter, tTerminateMax,
                    color, opacity,
                    volumeUniforms, b,
                    volumeTexture, transferFunctionTexture,
                    gradientOpacityTexture, maskTexture, labelMapTransferTexture,
                    minMaxTexture, normalTexture,
                    &volumeLights);
            }
        }

        advanceGridWalker(walker, tEndRel);
    }

    output.color = float4(float3(color), float(opacity));
    return output;
}

// ============================================================================
// GPU Min-Max Acceleration: compute kernels
// Phase 5: GPU-based empty-space skipping generation.
// Generates an R8Unorm occupancy texture where >0.5 means empty,
// 0.0 means solid. The fragment shader uses this to skip empty space.
// ============================================================================

struct MinMaxComputeUniforms {
  uint  mmDimX, mmDimY, mmDimZ;   // macrocell grid dimensions
  uint  volDimX, volDimY, volDimZ; // full volume voxel dimensions
  float ds;                       // downsampling factor (typically 4.0)
  float scalarMin;                // ScalarRange[0]
  float scalarScale;              // 255.0 / (ScalarRange[1] - ScalarRange[0])
  float _pad;
  uint  opacityPrefix[257];       // prefix sum: opacityPrefix[i] = count of non-zero opacity entries for indices < i
};

kernel void volume_compute_minmax(
    texture3d<float, access::sample> volume [[texture(0)]],
    texture3d<float, access::write>  occupancy [[texture(1)]],
    constant MinMaxComputeUniforms& u [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]])
{
  if (any(gid >= uint3(u.mmDimX, u.mmDimY, u.mmDimZ))) return;

  uint ds = uint(u.ds);
  uint3 start = gid * ds;
  uint3 end = min(start + ds, uint3(u.volDimX, u.volDimY, u.volDimZ));

  float cellMin = INFINITY;
  float cellMax = -INFINITY;
  float3 volDims = float3(u.volDimX, u.volDimY, u.volDimZ);

  for (uint z = start.z; z < end.z; z++) {
    for (uint y = start.y; y < end.y; y++) {
      for (uint x = start.x; x < end.x; x++) {
        float3 pos = (float3(x, y, z) + 0.5) / volDims;
        float v = volume.sample(sNearest, pos, level(0)).r;
        cellMin = min(cellMin, v);
        cellMax = max(cellMax, v);
      }
    }
  }

  // Check emptiness via opacity prefix table
  if (cellMin <= cellMax) {
    int iMin = int(floor((cellMin - u.scalarMin) * u.scalarScale));
    int iMax = int(floor((cellMax - u.scalarMin) * u.scalarScale));
    uint idxMin = uint(clamp(iMin, 0, 255));
    uint idxMax = uint(clamp(iMax, 0, 255));
    bool empty = (u.opacityPrefix[idxMax + 1] == u.opacityPrefix[idxMin]);
    occupancy.write(empty ? 1.0 : 0.0, gid);
  } else {
    occupancy.write(1.0, gid);
  }
}

kernel void volume_dilate_minmax(
    texture3d<float, access::read> src [[texture(0)]],
    texture3d<float, access::write> dst [[texture(1)]],
    uint3 gid [[thread_position_in_grid]])
{
  uint3 dims = uint3(dst.get_width(), dst.get_height(), dst.get_depth());
  if (any(gid >= dims)) return;

  bool solid = false;
  int3 g = int3(gid);

  for (int dz = -1; dz <= 1 && !solid; dz++) {
    for (int dy = -1; dy <= 1 && !solid; dy++) {
      for (int dx = -1; dx <= 1 && !solid; dx++) {
        int3 n = g + int3(dx, dy, dz);
        if (all(n >= 0) && all(n < int3(dims))) {
          if (src.read(uint3(n)).r < 0.5) {
            solid = true;
          }
        }
      }
    }
  }

  dst.write(solid ? 0.0 : 1.0, gid);
}

fragment float4 fragment_image_sample_blit(
    FullscreenVertexOut in [[stage_in]],
    texture2d<float> offscreenColor [[texture(0)]]) {
  return offscreenColor.sample(sVolume, in.texCoord);
}

// --- PHASE 7: GPU COMPUTE KERNELS FOR DATA TYPE CONVERSION ---
// Reads raw scalar data from a device buffer and writes the converted
// result directly into a 3D texture.  Replaces the CPU vtkSMPTools loop
// with a GPU compute dispatch for 5-10x speedup on short/int/double data.

struct VolumeConvertUniforms {
    uint dimX, dimY, dimZ;
    uint numComponents;
    uint outputComponents;
    uint _pad;
};

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

// ushort -> normalized half with clamping to [0,255] (for ushort→uchar normalization).
// Writes normalized values to a Unorm texture; the shader samples them as [0,1] and
// reconstructs the original by multiplying by ScalarNormalizationFactor (255.0).
kernel void volume_convert_ushort_to_uchar(
    device const ushort* src [[buffer(0)]],
    texture3d<half, access::write> dst [[texture(0)]],
    constant VolumeConvertUniforms& u [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]])
{
    if (any(gid >= uint3(u.dimX, u.dimY, u.dimZ))) return;
    uint srcIdx = (gid.z * u.dimY + gid.y) * u.dimX + gid.x;
    half4 val;
    val.x = half(min(src[srcIdx * u.numComponents + 0], (ushort)255)) / 255.0h;
    val.y = u.numComponents > 1 ? half(min(src[srcIdx * u.numComponents + 1], (ushort)255)) / 255.0h : 1.0h;
    val.z = u.numComponents > 2 ? half(min(src[srcIdx * u.numComponents + 2], (ushort)255)) / 255.0h : 1.0h;
    val.w = u.numComponents > 3 ? half(min(src[srcIdx * u.numComponents + 3], (ushort)255)) / 255.0h : 1.0h;
    dst.write(val, gid);
}

// NOTE: double -> half/float kernels are not provided because Metal does not
// support 'double' in device address space. VTK_DOUBLE data falls back to CPU.