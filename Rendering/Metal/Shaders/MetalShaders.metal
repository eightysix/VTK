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

// Cell ID offset (P2-7)
struct CellIdOffsetUniform {
  uint offset;                 // added to instance_id to get global cell ID
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

// ---------------------------------------------------------------------------
// Vertex shader
// ---------------------------------------------------------------------------
vertex VertexOut vertex_main(uint vertex_id [[vertex_id]],
                             VertexIn in [[stage_in]],
                             constant SceneUniforms& scene [[buffer(2)]],
                             constant float4* vertexColors [[buffer(3)]],
                             constant ClipPlaneUniforms& clipPlanes [[buffer(5)]],
                             constant uint* cellIds [[buffer(6)]],
                             constant uint& propId [[buffer(7)]],
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
  out.propId = propId + 1u;

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

  bool hasVertexColors = (scene.flags & (1u << 8)) != 0u;
  float3 ambientColor = hasVertexColors ? in.vertexColor.rgb : material.ambientColor.rgb;
  float ambientIntensity = material.ambientColor.w;
  float3 diffuseColor = hasVertexColors ? in.vertexColor.rgb : material.diffuseColor.rgb;
  float diffuseIntensity = material.diffuseColor.w;
  float3 specularColor = material.specularColor.rgb;
  float specularIntensity = material.specularColor.w;
  float baseOpacity = hasVertexColors ? in.vertexColor.a : material.opacity;

  bool hasTexture = (scene.flags & (1u << 9)) != 0u;
  if (hasTexture) {
    float4 texColor = actorTexture.sample(actorSampler, in.uv);
    ambientColor *= texColor.rgb;
    diffuseColor *= texColor.rgb;
    baseOpacity *= texColor.a;
  }

  float3 totalAmbient = ambientIntensity * ambientColor;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);

  computePhongLighting(N, in.viewPos, diffuseColor, specularColor, specularIntensity, material.specularPower, lights, totalDiffuse, totalSpecular);

  FragmentOutput out;
  out.color = float4(totalAmbient + diffuseIntensity * totalDiffuse + totalSpecular, baseOpacity);
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);
  
  float cscale = length(float2(dfdx(in.position.z), dfdy(in.position.z)));
  out.depth = in.position.z + coinOffset.polygonFactor * cscale + coinOffset.polygonOffset / 65000.0;
  return out;
}

fragment FragmentOutput fragment_edge_main(VertexOut in [[stage_in]],
                                   constant MaterialUniforms& material [[buffer(0)]],
                                   constant LightUniforms& lights [[buffer(1)]],
                                   constant SceneUniforms& scene [[buffer(2)]],
                                   constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
                                   constant float4& edgeColor [[buffer(4)]]) {
  FragmentOutput out;
  out.color = float4(edgeColor.rgb, edgeColor.a * material.opacity);
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);

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
    constant uint& pointPropId [[buffer(12)]]) {
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
  out.propId = pointPropId + 1u;
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
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);
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
    constant uint& shapedPropId [[buffer(12)]]) {
  
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
  out.propId = shapedPropId + 1u;
  return out;
}

struct PointFragmentOutput {
  float4 color [[color(0)]];
  uint4 ids [[color(1)]];
  float depth [[depth(any)]];
};

fragment PointFragmentOutput fragment_point_shaped_main(
    PointShapedVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant VertexColorUniforms& vertexColorUniform [[buffer(4)]]) {
  PointFragmentOutput out;

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
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);
  out.depth += coinOffset.pointOffset / 65000.0;
  return out;
}

// ---------------------------------------------------------------------------
// Thick line shaders
// ---------------------------------------------------------------------------
struct ThickLineVertexOut {
  float4 position [[position]];
  float3 viewPos;
  float3 viewNormal;
  float4 vertexColor;
  float dist_to_centerline;
  uint cellId;
  uint propId;
};

vertex ThickLineVertexOut vertex_thick_line_main(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    constant float3* positions [[buffer(0)]],
    constant uint* lineIndices [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant float4* vertexColors [[buffer(3)]],
    constant float& lineWidth [[buffer(4)]],
    constant uint* cellIds [[buffer(5)]],
    constant uint& propId [[buffer(6)]]) {
  
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

  float w = max(lineWidth, 1.0);
  float2 adjusted_p0 = p0_screen + p_coord.x * x_basis + p_coord.y * y_basis * w;
  float2 adjusted_p1 = p1_screen + p_coord.x * x_basis + p_coord.y * y_basis * w;
  float2 p = mix(adjusted_p0, adjusted_p1, p_coord.x);

  float4 p_DC = mix(p0_DC, p1_DC, p_coord.x);

  ThickLineVertexOut out;
  out.position = float4(p_DC.w * ((2.0 * p) / resolution - 1.0), p_DC.z, p_DC.w);
  out.viewPos = (scene.viewMatrix * scene.modelMatrix * float4(mix(p0_MC, p1_MC, p_coord.x), 1.0)).xyz;
  out.viewNormal = scene.normalMatrix * float3(0.0, 0.0, 1.0);
  out.vertexColor = vertexColors[p0_idx];
  out.dist_to_centerline = p_coord.y;
  out.cellId = cellIds[instance_id];
  out.propId = propId + 1u;
  return out;
}

fragment FragmentOutput fragment_thick_line_main(
    ThickLineVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]]) {
  FragmentOutput out;

  float3 baseColor = in.vertexColor.rgb;
  float baseAlpha = in.vertexColor.a * material.opacity;

  float3 N = normalize(in.viewNormal);
  N.z = 1.0 - 2.0 * abs(in.dist_to_centerline);
  N = normalize(N);

  float3 totalAmbient = material.ambientColor.w * baseColor;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);

  computePhongLighting(N, in.viewPos, baseColor, material.specularColor.rgb, material.specularColor.w, material.specularPower, lights, totalDiffuse, totalSpecular);

  out.color = float4(totalAmbient + material.diffuseColor.w * totalDiffuse + totalSpecular, baseAlpha);
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);

  float cscale = length(float2(dfdx(in.position.z), dfdy(in.position.z)));
  out.depth = in.position.z + coinOffset.lineFactor * cscale + coinOffset.lineOffset / 65000.0;
  return out;
}

// ---------------------------------------------------------------------------
// Round Cap Line Shaders 
// ---------------------------------------------------------------------------
struct RoundCapLineVertexOut {
  float4 position [[position]];
  float3 viewPos;
  float3 viewNormal;
  float4 vertexColor;
  float dist_to_centerline;
  uint cellId;
  uint propId;
};

vertex RoundCapLineVertexOut vertex_round_cap_line_main(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    constant float3* positions [[buffer(0)]],
    constant uint* lineIndices [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant float4* vertexColors [[buffer(3)]],
    constant float& lineWidth [[buffer(4)]],
    constant uint* cellIds [[buffer(5)]],
    constant uint& propId [[buffer(6)]]) {

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

  RoundCapLineVertexOut out;
  out.position = float4(p_DC.w * ((2.0 * p) / resolution - 1.0), p_DC.z, p_DC.w);
  out.viewPos = (scene.viewMatrix * scene.modelMatrix * float4(mix(p0_MC, p1_MC, p_coord.z), 1.0)).xyz;
  out.viewNormal = scene.normalMatrix * float3(0.0, 0.0, 1.0);
  out.vertexColor = mix(vertexColors[p0_idx], vertexColors[p1_idx], p_coord.z);
  out.dist_to_centerline = p_coord.y;
  out.cellId = cellIds[instance_id];
  out.propId = propId + 1u;
  return out;
}

fragment FragmentOutput fragment_round_cap_line_main(
    RoundCapLineVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]]) {
  FragmentOutput out;

  float3 baseColor = in.vertexColor.rgb;
  float baseAlpha = in.vertexColor.a * material.opacity;

  float3 N = normalize(in.viewNormal);
  N.z = 1.0 - 2.0 * abs(in.dist_to_centerline);
  N = normalize(N);

  float3 totalAmbient = material.ambientColor.w * baseColor;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);
  
  computePhongLighting(N, in.viewPos, baseColor, material.specularColor.rgb, material.specularColor.w, material.specularPower, lights, totalDiffuse, totalSpecular);

  out.color = float4(totalAmbient + material.diffuseColor.w * totalDiffuse + totalSpecular, baseAlpha);
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);

  float cscale = length(float2(dfdx(in.position.z), dfdy(in.position.z)));
  out.depth = in.position.z + coinOffset.lineFactor * cscale + coinOffset.lineOffset / 65000.0;
  return out;
}

// ---------------------------------------------------------------------------
// Miter Join Line Shaders
// ---------------------------------------------------------------------------
struct MiterJoinLineVertexOut {
  float4 position [[position]];
  float3 viewPos;
  float3 viewNormal;
  float4 vertexColor;
  float dist_to_centerline;
  uint cellId;
  uint propId;
};

vertex MiterJoinLineVertexOut vertex_miter_join_line_main(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    constant float3* positions [[buffer(0)]],
    constant uint* lineIndices [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant float4* vertexColors [[buffer(3)]],
    constant float& lineWidth [[buffer(4)]],
    constant uint* cellIds [[buffer(5)]],
    constant uint& propId [[buffer(6)]],
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
  float2 offset = p_coord.x * x_basis + p_coord.y * y_basis * w;

  if (p_coord.x == -1.0 && instance_id > 0 && cellIds[instance_id - 1] == cellIds[instance_id]) {
    float4 prev_p0_DC = scene.projectionMatrix * scene.viewMatrix * scene.modelMatrix * float4(positions[lineIndices[(instance_id - 1) * 2]], 1.0);
    float2 prev_p0_screen = resolution * (0.5 * prev_p0_DC.xy / prev_p0_DC.w + 0.5);
    
    float2 prev_delta = p0_screen - prev_p0_screen;
    float prev_len = length(prev_delta);
    float2 prev_dir = prev_len < 0.001 ? float2(1.0, 0.0) : (prev_delta / prev_len);
    
    float2 miter = float2(-prev_dir.y, prev_dir.x) + float2(-x_basis.y, x_basis.x);
    float miter_len = length(miter);

    if (miter_len > 0.001 && miter_len < 4.0) {
      miter = miter / miter_len;
      float miterOffset = w * 0.5 / dot(miter, float2(-x_basis.y, x_basis.x));
      if (sign(dot(p_coord.y * y_basis, miter)) == sign(dot(float2(0.0, 1.0), miter))) {
        offset = p_coord.x * x_basis + miter * miterOffset;
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
    float miter_len = length(miter);

    if (miter_len > 0.001 && miter_len < 4.0) {
      miter = miter / miter_len;
      float miterOffset = w * 0.5 / dot(miter, float2(-x_basis.y, x_basis.x));
      if (sign(dot(p_coord.y * y_basis, miter)) == sign(dot(float2(0.0, 1.0), miter))) {
        offset = p_coord.x * x_basis + miter * miterOffset;
      }
    }
  }

  float2 p = p0_screen + offset + (p1_screen - p0_screen) * 0.5 * (p_coord.x + 1.0);
  float4 p_DC = mix(p0_DC, p1_DC, p_coord.x);

  MiterJoinLineVertexOut out;
  out.position = float4(p_DC.w * ((2.0 * p) / resolution - 1.0), p_DC.z, p_DC.w);
  out.viewPos = (scene.viewMatrix * scene.modelMatrix * float4(mix(positions[p0_idx], positions[p1_idx], p_coord.x), 1.0)).xyz;
  out.viewNormal = scene.normalMatrix * float3(0.0, 0.0, 1.0);
  out.vertexColor = mix(vertexColors[p0_idx], vertexColors[p1_idx], p_coord.x);
  out.dist_to_centerline = p_coord.y;
  out.cellId = cellIds[instance_id];
  out.propId = propId + 1u;
  return out;
}

fragment FragmentOutput fragment_miter_join_line_main(
    MiterJoinLineVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]]) {
  FragmentOutput out;

  float3 baseColor = in.vertexColor.rgb;
  float baseAlpha = in.vertexColor.a * material.opacity;

  float3 N = normalize(in.viewNormal);
  N.z = 1.0 - 2.0 * abs(in.dist_to_centerline);
  N = normalize(N);

  float3 totalAmbient = material.ambientColor.w * baseColor;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);
  
  computePhongLighting(N, in.viewPos, baseColor, material.specularColor.rgb, material.specularColor.w, material.specularPower, lights, totalDiffuse, totalSpecular);

  out.color = float4(totalAmbient + material.diffuseColor.w * totalDiffuse + totalSpecular, baseAlpha);
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);

  float cscale = length(float2(dfdx(in.position.z), dfdy(in.position.z)));
  out.depth = in.position.z + coinOffset.lineFactor * cscale + coinOffset.lineOffset / 65000.0;
  return out;
}

// ---------------------------------------------------------------------------
// Compute Kernels (Tessellation mapping)
// ---------------------------------------------------------------------------
kernel void cellToPrimitive(
    device uint* cellIds [[buffer(0)]],
    constant uint* primitiveToCell [[buffer(1)]],
    constant uint& cellIdOffset [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  cellIds[gid] = primitiveToCell[gid] + cellIdOffset + 1u;
}

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
    cellIds[triangleId] = gid + params.cellIdOffset;

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
    cellIds[primitiveCounts[gid] + i] = gid + params.cellIdOffset;
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
    cellIds[primitiveCounts[gid] + i] = gid + params.cellIdOffset;
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
};

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
    float rawMag = length(rawGrad);

    // Normalize and encode to [0, 1]
    float3 normal = rawMag > 1e-6 ? rawGrad / rawMag : float3(0.0, 0.0, 1.0);
    float3 encoded = normal * 0.5 + 0.5;
    float gradMagNorm = saturate(rawMag / max(u.gradNormFactor, 1e-6));

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

  uint2 pixel = uint2(in.position.xy) - uint2(scene.viewport.xy);
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
  bool hasVertexColors = (scene.flags & (1u << 8)) != 0u;
  float3 ambientColor = hasVertexColors ? in.vertexColor.rgb : material.ambientColor.rgb;
  float3 diffuseColor = hasVertexColors ? in.vertexColor.rgb : material.diffuseColor.rgb;
  float baseOpacity = hasVertexColors ? in.vertexColor.a : material.opacity;

  if ((scene.flags & (1u << 9)) != 0u) {
    float4 texColor = actorTexture.sample(actorSampler, in.uv);
    ambientColor *= texColor.rgb;
    diffuseColor *= texColor.rgb;
    baseOpacity *= texColor.a;
  }

  float3 totalAmbient = material.ambientColor.w * ambientColor;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);

  computePhongLighting(N, in.viewPos, diffuseColor, material.specularColor.rgb, material.specularColor.w, material.specularPower, lights, totalDiffuse, totalSpecular);

  float3 fragRGB = totalAmbient + material.diffuseColor.w * totalDiffuse + totalSpecular;

  if (fragDepth >= minDepth - epsilon && fragDepth <= minDepth + epsilon) {
    float prevAlpha = 1.0 - prevFront.a;
    out.frontDest.rgb = prevAlpha * baseOpacity * fragRGB + prevFront.rgb;
    out.frontDest.a = 1.0 - (prevAlpha * (1.0 - baseOpacity));
  } else if (fragDepth >= maxDepth - epsilon && fragDepth <= maxDepth + epsilon) {
    out.backTemp = float4(fragRGB * baseOpacity, baseOpacity);
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

  uint2 pixel = uint2(in.position.xy) - uint2(scene.viewport.xy);
  float2 prevDepth = prevDepthTex.read(pixel).rg;
  float fragDepth = in.position.z;
  float epsilon = 0.0000001;

  if (fragDepth < -prevDepth.x - epsilon || fragDepth > prevDepth.y + epsilon) discard_fragment();

  float3 N = normalize(in.viewNormal);
  bool hasVertexColors = (scene.flags & (1u << 8)) != 0u;
  float3 ambientColor = hasVertexColors ? in.vertexColor.rgb : material.ambientColor.rgb;
  float3 diffuseColor = hasVertexColors ? in.vertexColor.rgb : material.diffuseColor.rgb;
  float baseOpacity = hasVertexColors ? in.vertexColor.a : material.opacity;

  if ((scene.flags & (1u << 9)) != 0u) {
    float4 texColor = actorTexture.sample(actorSampler, in.uv);
    ambientColor *= texColor.rgb;
    diffuseColor *= texColor.rgb;
    baseOpacity *= texColor.a;
  }

  float3 totalAmbient = material.ambientColor.w * ambientColor;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);

  computePhongLighting(N, in.viewPos, diffuseColor, material.specularColor.rgb, material.specularColor.w, material.specularPower, lights, totalDiffuse, totalSpecular);

  float3 fragRGB = totalAmbient + material.diffuseColor.w * totalDiffuse + totalSpecular;
  return float4(fragRGB * baseOpacity, baseOpacity);
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
};

vertex GlyphVertexOut vertex_glyph_main(
    uint vertex_id [[vertex_id]], uint instance_id [[instance_id]],
    constant float3* positions [[buffer(0)]], constant float3* normals [[buffer(1)]],
    constant float4x4* glyphTransforms [[buffer(2)]], constant float3x3* glyphNormalTransforms [[buffer(3)]],
    constant float4* glyphColors [[buffer(4)]], constant uint* glyphPickIds [[buffer(5)]],
    constant SceneUniforms& scene [[buffer(8)]], constant ClipPlaneUniforms& clipPlanes [[buffer(9)]],
    constant uint& propId [[buffer(10)]]) {
  
  GlyphVertexOut out;
  float3 pos = positions[vertex_id];
  float4 worldPos = scene.modelMatrix * glyphTransforms[instance_id] * float4(pos, 1.0);
  
  out.viewPos = (scene.viewMatrix * worldPos).xyz;
  out.position = scene.projectionMatrix * float4(out.viewPos, 1.0);
  out.viewNormal = scene.normalMatrix * glyphNormalTransforms[instance_id] * normals[vertex_id];
  out.glyphColor = glyphColors[instance_id];
  out.cellId = glyphPickIds[instance_id] + 1u;
  out.propId = propId + 1u;
  out.modelPos = pos;
  return out;
}

fragment FragmentOutput fragment_glyph_main(
    GlyphVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(9)]]) {
  
  if (isClipped(in.modelPos, clipPlanes)) discard_fragment();

  float3 N = normalize(in.viewNormal);
  float3 totalAmbient = material.ambientColor.w * in.glyphColor.rgb;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);

  computePhongLighting(N, in.viewPos, in.glyphColor.rgb, material.specularColor.rgb, material.specularColor.w, material.specularPower, lights, totalDiffuse, totalSpecular);

  FragmentOutput out;
  out.color = float4(totalAmbient + material.diffuseColor.w * totalDiffuse + totalSpecular, in.glyphColor.a * material.opacity);
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);
  out.depth = in.position.z;
  return out;
}

struct GlyphLineVertexOut {
  float4 position [[position]];
  float3 viewPos;
  float3 viewNormal;
  float4 glyphColor;
  float3 modelPos;
  uint cellId;
  uint propId;
};

vertex GlyphLineVertexOut vertex_glyph_line_main(
    uint vertex_id [[vertex_id]], uint instance_id [[instance_id]],
    constant float3* positions [[buffer(0)]], constant float3* normals [[buffer(1)]],
    constant float4x4* glyphTransforms [[buffer(2)]], constant float3x3* glyphNormalTransforms [[buffer(3)]],
    constant float4* glyphColors [[buffer(4)]], constant uint* glyphPickIds [[buffer(5)]],
    constant SceneUniforms& scene [[buffer(8)]], constant ClipPlaneUniforms& clipPlanes [[buffer(9)]],
    constant uint& propId [[buffer(10)]]) {
  
  GlyphLineVertexOut out;
  float3 pos = positions[vertex_id];
  float4 worldPos = scene.modelMatrix * glyphTransforms[instance_id] * float4(pos, 1.0);
  
  out.viewPos = (scene.viewMatrix * worldPos).xyz;
  out.position = scene.projectionMatrix * float4(out.viewPos, 1.0);
  out.viewNormal = scene.normalMatrix * glyphNormalTransforms[instance_id] * normals[vertex_id];
  out.glyphColor = glyphColors[instance_id];
  out.cellId = glyphPickIds[instance_id] + 1u;
  out.propId = propId + 1u;
  out.modelPos = pos;
  return out;
}

fragment FragmentOutput fragment_glyph_line_main(
    GlyphLineVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(9)]]) {
  
  if (isClipped(in.modelPos, clipPlanes)) discard_fragment();

  float3 N = normalize(in.viewNormal);
  float3 totalAmbient = material.ambientColor.w * in.glyphColor.rgb;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);

  computePhongLighting(N, in.viewPos, in.glyphColor.rgb, material.specularColor.rgb, material.specularColor.w, material.specularPower, lights, totalDiffuse, totalSpecular);

  FragmentOutput out;
  out.color = float4(totalAmbient + material.diffuseColor.w * totalDiffuse + totalSpecular, in.glyphColor.a * material.opacity);
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);
  out.depth = in.position.z;
  return out;
}

struct GlyphPointVertexOut {
  float4 position [[position]];
  float3 viewPos;
  float3 viewNormal;
  float4 glyphColor;
  float3 modelPos;
  uint cellId;
  uint propId;
  float point_size [[point_size]];
};

vertex GlyphPointVertexOut vertex_glyph_point_main(
    uint vertex_id [[vertex_id]], uint instance_id [[instance_id]],
    constant float3* positions [[buffer(0)]], constant float3* normals [[buffer(1)]],
    constant float4x4* glyphTransforms [[buffer(2)]], constant float3x3* glyphNormalTransforms [[buffer(3)]],
    constant float4* glyphColors [[buffer(4)]], constant uint* glyphPickIds [[buffer(5)]],
    constant SceneUniforms& scene [[buffer(8)]], constant ClipPlaneUniforms& clipPlanes [[buffer(9)]],
    constant uint& propId [[buffer(10)]]) {
  
  GlyphPointVertexOut out;
  float3 pos = positions[vertex_id];
  float4 worldPos = scene.modelMatrix * glyphTransforms[instance_id] * float4(pos, 1.0);
  
  out.viewPos = (scene.viewMatrix * worldPos).xyz;
  out.position = scene.projectionMatrix * float4(out.viewPos, 1.0);
  out.viewNormal = scene.normalMatrix * glyphNormalTransforms[instance_id] * normals[vertex_id];
  out.glyphColor = glyphColors[instance_id];
  out.cellId = glyphPickIds[instance_id] + 1u;
  out.propId = propId + 1u;
  out.point_size = scene.pointSize;
  out.modelPos = pos;
  return out;
}

fragment FragmentOutput fragment_glyph_point_main(
    GlyphPointVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(9)]]) {
  
  if (isClipped(in.modelPos, clipPlanes)) discard_fragment();

  float3 N = normalize(in.viewNormal);
  float3 totalAmbient = material.ambientColor.w * in.glyphColor.rgb;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);

  computePhongLighting(N, in.viewPos, in.glyphColor.rgb, material.specularColor.rgb, material.specularColor.w, material.specularPower, lights, totalDiffuse, totalSpecular);

  FragmentOutput out;
  out.color = float4(totalAmbient + material.diffuseColor.w * totalDiffuse + totalSpecular, in.glyphColor.a * material.opacity);
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);
  out.depth = in.position.z + coinOffset.pointOffset / 65000.0;
  return out;
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
  float sampleDistance;
  float scalarMin;
  float scalarMax;
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
  float4 croppingFlagsRow0;
  float4 croppingFlagsRow1;
  float4 croppingFlagsRow2;
  float4 croppingFlagsRow3;
  float4 croppingFlagsRow4;
  float4 croppingFlagsRow5;
  float4 croppingFlagsRow6;
  float4 croppingFlagsRow7;
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
  float frameIndex;
  // Min-max acceleration texture
  float useMinMaxAccel;
  float minMaxDimX;
  float minMaxDimY;
  float minMaxDimZ;
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

inline float interleavedGradientNoise(float2 p) {
  return fract(52.9829189 * fract(dot(p, float2(0.06711056, 0.00583715))));
}

inline float2 intersectBox(float3 orig, float3 dir, float3 boxMin, float3 boxMax) {
  float3 invDir = 1.0 / select(dir, float3(1e-8), abs(dir) < 1e-8);
  float3 tbot = invDir * (boxMin - orig);
  float3 ttop = invDir * (boxMax - orig);
  float3 tmin = min(ttop, tbot);
  float3 tmax = max(ttop, tbot);
  return float2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

// Compute the ray skip distance to exit the current cell for empty-space skipping.
// Used by both coarse and fine min-max levels in the hierarchical accelerator.
inline float computeMinMaxSkip(float3 mmPos, float3 cellDims, float3 rayDir,
                                float3 rayDirTexLocal, float stepSize) {
  float3 cellCoord = mmPos * cellDims;
  float3 fractCoord = fract(cellCoord);
  float3 distToEdge;
  distToEdge.x = rayDir.x > 0.0 ? (1.0 - fractCoord.x) : fractCoord.x;
  distToEdge.y = rayDir.y > 0.0 ? (1.0 - fractCoord.y) : fractCoord.y;
  distToEdge.z = rayDir.z > 0.0 ? (1.0 - fractCoord.z) : fractCoord.z;
  distToEdge = mix(distToEdge, float3(1.0), float3(distToEdge <= 1e-5));
  float3 tToEdge;
  tToEdge.x = abs(rayDirTexLocal.x) > 1e-5 ? distToEdge.x / abs(rayDirTexLocal.x * cellDims.x) : 1e30;
  tToEdge.y = abs(rayDirTexLocal.y) > 1e-5 ? distToEdge.y / abs(rayDirTexLocal.y * cellDims.y) : 1e30;
  tToEdge.z = abs(rayDirTexLocal.z) > 1e-5 ? distToEdge.z / abs(rayDirTexLocal.z * cellDims.z) : 1e30;
  float exactSkip = min(min(tToEdge.x, tToEdge.y), tToEdge.z);
  exactSkip += 1e-4;
  float skipDist = ceil(exactSkip / stepSize) * stepSize;
  return max(stepSize, skipDist);
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
  half rawMag = length(rawGrad);

  half3 correctedGrad = rawGrad * gradScale;
  half3 normal = length(correctedGrad) > 0.0h ? normalize(correctedGrad) : half3(0.0h);

  return half4(normal, saturate(rawMag / gradNormFactor));
}

// Optimized: Pure FP16 math and fast::pow
inline half3 computePhongLightingVolumeFast(half3 sampleColor, half3 normal, half3 lightDir, half3 viewDir,
                                            half3 ambientMat, half3 diffuseMat, half3 specularMat, half shininess) {
  half nDotL = dot(normal, -lightDir);
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

// Branchless, fast crop region evaluator (returns 0..26 matching VTK region bit convention)
inline int computeCropRegion(float3 cropMin, float3 cropMax, float3 pos) {
  int3 r = int3(step(cropMin, pos)) + int3(step(cropMax, pos));
  return r.x + r.y * 3 + r.z * 9;
}

fragment VolumeFragmentOut fragment_volume_main(
    VolumeVertexOut in [[stage_in]],
    bool isFrontFace [[front_facing]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> transferFunctionTexture [[texture(1)]],
    texture2d<float> depthTexture [[texture(2)]],
    texture2d<float> gradientOpacityTexture [[texture(3)]],
    texture3d<float> maskTexture [[texture(4)]],
    texture2d<float> labelMapTransferTexture [[texture(5)]],
    texture3d<float> minMaxTexture [[texture(6)]],
    texture3d<float> normalTexture [[texture(7)]]) {

  if (!isFrontFace) discard_fragment();

  VolumeFragmentOut output;
  float3 blockMinGlobal = (b.volumeBoundsMin.xyz - volumeUniforms.volumeBoundsMin.xyz) / max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 blockMaxGlobal = (b.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz) / max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);

  float3 texMinGlobal = (b.textureBoundsMin.xyz - volumeUniforms.volumeBoundsMin.xyz) / max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 texMaxGlobal = (b.textureBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz) / max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);

  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float stepSize = volumeUniforms.sampleDistance;

  float3 rayDir = in.localPos - cameraPos;
  float dirLength = length(rayDir);
  if (dirLength < 0.0001) discard_fragment();

  rayDir /= dirLength;
  float2 t = intersectBox(cameraPos, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) discard_fragment();

  float3 entryPoint = cameraPos + rayDir * tStart;
  float3 exitPoint = cameraPos + rayDir * t.y;
  float totalDist = length(exitPoint - entryPoint);

  if (volumeUniforms.useClipping > 0.5) {
    int numClipPlanes = int(volumeUniforms.numClippingPlanes);
    float4 planeOrigins[8] = { volumeUniforms.clippingPlane0Origin, volumeUniforms.clippingPlane1Origin, volumeUniforms.clippingPlane2Origin, volumeUniforms.clippingPlane3Origin, volumeUniforms.clippingPlane4Origin, volumeUniforms.clippingPlane5Origin, volumeUniforms.clippingPlane6Origin, volumeUniforms.clippingPlane7Origin };
    float4 planeNormals[8] = { volumeUniforms.clippingPlane0Normal, volumeUniforms.clippingPlane1Normal, volumeUniforms.clippingPlane2Normal, volumeUniforms.clippingPlane3Normal, volumeUniforms.clippingPlane4Normal, volumeUniforms.clippingPlane5Normal, volumeUniforms.clippingPlane6Normal, volumeUniforms.clippingPlane7Normal };

    for (int cp = 0; cp < numClipPlanes; cp++) {
      float3 planeOrigin = planeOrigins[cp].xyz;
      float3 planeNormal = normalize(planeNormals[cp].xyz);
      float startDistance = dot(planeNormal, planeOrigin - entryPoint);
      float stopDistance = dot(planeNormal, planeOrigin - exitPoint);

      if (startDistance > 0.0 && stopDistance > 0.0) discard_fragment();
      float rayDotNormal = dot(rayDir, planeNormal);

      if (rayDotNormal > 0.0 && startDistance > 0.0) entryPoint += (startDistance / rayDotNormal) * rayDir;
      if (rayDotNormal <= 0.0 && stopDistance > 0.0) exitPoint += (stopDistance / rayDotNormal) * rayDir;
    }
    totalDist = length(exitPoint - entryPoint);
    if (totalDist < 1e-6) discard_fragment();
  }

  float tTerminateMax = 1e30;
  if (volumeUniforms.useDepthTexture > 0.5) {
    float depthSample = depthTexture.sample(sNearest, in.position.xy / volumeUniforms.viewportSize).r;
    if (depthSample < 1.0) {
      float2 ndcXY = (in.position.xy / volumeUniforms.viewportSize) * 2.0 - 1.0;
      float4 worldTermination = volumeUniforms.inverseViewProjection * float4(ndcXY.x, -ndcXY.y, depthSample, 1.0);
      worldTermination.xyz /= worldTermination.w;
      float3 terminationLocal = (worldTermination.xyz - volumeUniforms.volumeBoundsMin.xyz) / max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
      tTerminateMax = dot(terminationLocal - entryPoint, rayDir);
      if (tTerminateMax < 0.0) tTerminateMax = 1e30;
    }
  }

  // --- LOCAL CACHE WARM-UP ---
  const bool doShading = fc_shading && (volumeUniforms.useGradientShading > 0.5);
  const bool doGradOp = fc_gradientOpacity && (volumeUniforms.useGradientOpacity > 0.5);
  const bool doCropping = volumeUniforms.useCropping > 0.5;
  const bool doMask = fc_mask && (volumeUniforms.useMask > 0.5);

  half scalarScale = half(1.0 / (volumeUniforms.scalarMax - volumeUniforms.scalarMin));
  half scalarBias  = half(-volumeUniforms.scalarMin) * scalarScale;

  half gradNormFactor = half(max(1e-8f, volumeUniforms.gradientOpacityRange.y));

  float3 texSizeGlobal = max(texMaxGlobal - texMinGlobal, 1e-6);
  float3 dt = max(volumeUniforms.gradientStep, 1e-8);
  half3 gradScale = half3(1.0 / (dt * texSizeGlobal));

  half3 viewDirHalf  = half3(normalize(entryPoint - cameraPos));
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

  uint cropBitmask = 0;
  if (doCropping) {
    float4 cropF[8] = { volumeUniforms.croppingFlagsRow0, volumeUniforms.croppingFlagsRow1, volumeUniforms.croppingFlagsRow2, volumeUniforms.croppingFlagsRow3, volumeUniforms.croppingFlagsRow4, volumeUniforms.croppingFlagsRow5, volumeUniforms.croppingFlagsRow6, volumeUniforms.croppingFlagsRow7 };
    for (int j = 0; j < 32; j++) {
      if (cropF[j / 4][j % 4] > 0.0) cropBitmask |= (1u << j);
    }
  }

  // --- SOFTWARE PIPELINING INITIALIZATION ---
  float jitter = volumeUniforms.useJittering > 0.5
    ? interleavedGradientNoise(in.position.xy + volumeUniforms.frameIndex) * stepSize
    : 0.0;
  float3 stepVec = rayDir * stepSize;
  float3 currentPoint = entryPoint + (rayDir * jitter);
  float currentT = jitter;

  int maxSteps = min(max(1, int(ceil(totalDist / stepSize))), MAX_RAY_STEPS);

  half3 accumulatedColor = half3(0.0);
  half accumulatedOpacity = 0.0;

  // PREFETCH the very first samples before the loop starts
  float3 texLocalPos0 = (currentPoint - texMinGlobal) / max(texMaxGlobal - texMinGlobal, 1e-6);
  float3 evalPoint0 = texLocalPos0;
  float prefetchScalar = volumeTexture.sample(sVolume, evalPoint0, level(0)).r;
  float prefetchMask = doMask ? maskTexture.sample(sNearest, evalPoint0, level(0)).r : 0.0;
  bool prefetchValid = true;
  int3  curCell     = int3(-1);
  int3  curCoarseCell = int3(-1);
  bool  curCoarseEmpty = false;
  bool  curCellEmpty = false;
  float3 mmDimF     = b.minMaxInfo.yzw;

  // --- THE RAYMARCHING LOOP ---
  for (int i = 0; i < maxSteps; i++) {
    // Strict check to completely suppress smearing outside partitioned edges
    if (any(currentPoint < blockMinGlobal - 1e-4) || any(currentPoint > blockMaxGlobal + 1e-4)) break;

    // 0. MIN-MAX ACCELERATION (hierarchical, two-level)
    // fc_minmax gates the entire empty-space skipping code at compile time.
    // If min-max is not in use, this entire block is eliminated by the compiler.
    // Coarse level (mip level 1, 2x reduction) is checked first: large empty
    // regions are skipped 8x faster than iterating each fine macrocell.
    if (fc_minmax &&
        b.minMaxInfo.x > 0.5 &&
        b.minMaxInfo.y > 0.5 &&
        b.minMaxInfo.z > 0.5 &&
        b.minMaxInfo.w > 0.5) {
      float3 texLocalPos = (currentPoint - texMinGlobal) / max(texMaxGlobal - texMinGlobal, 1e-6);
      float3 mmPos = clamp(texLocalPos, float3(0.0), float3(1.0));
      float3 rayDirTexLocal = rayDir / max(texMaxGlobal - texMinGlobal, 1e-6);

      // --- COARSE LEVEL CHECK (mip level 1, 2x coarser) ---
      // Coarse level exists only when b.minMaxInfo.x > 1.5 (set by C++).
      // The dimension-based OR is not the canonical check because C++ may
      // create the texture with only 1 mip level (no coarse mip).
      float3 coarseDimF = max(mmDimF * 0.5, 1.0);
      bool hasCoarse = b.minMaxInfo.x > 1.5;

      if (hasCoarse) {
        int3 newCoarse = min(int3(mmPos * coarseDimF), int3(coarseDimF) - 1);
        if (any(newCoarse != curCoarseCell)) {
          curCoarseCell = newCoarse;
          curCoarseEmpty = minMaxTexture.sample(sNearest, mmPos, level(1)).r > 0.5;
        }

        if (curCoarseEmpty) {
          float skipDist = computeMinMaxSkip(mmPos, coarseDimF, rayDir, rayDirTexLocal, stepSize);
          currentPoint += rayDir * skipDist;
          currentT += skipDist;
          if (any(currentPoint < blockMinGlobal - 1e-4) || any(currentPoint > blockMaxGlobal + 1e-4) || currentT >= t.y - tStart) break;
          prefetchValid = false;
          curCell = int3(-1);
          curCoarseCell = int3(-1);
          continue;
        }
      }

      // --- FINE LEVEL CHECK (mip level 0) ---
      int3 newCell = min(int3(mmPos * mmDimF), int3(mmDimF) - 1);
      if (any(newCell != curCell)) {
        curCell      = newCell;
        curCellEmpty = minMaxTexture.sample(sNearest, mmPos, level(0)).r > 0.5;
      }

      if (curCellEmpty) {
        float skipDist = computeMinMaxSkip(mmPos, mmDimF, rayDir, rayDirTexLocal, stepSize);
        currentPoint += rayDir * skipDist;
        currentT += skipDist;
        if (any(currentPoint < blockMinGlobal - 1e-4) || any(currentPoint > blockMaxGlobal + 1e-4) || currentT >= t.y - tStart) break;
        prefetchValid = false;
        curCell = int3(-1);
        curCoarseCell = int3(-1);
        continue;
      }
    }

    // 1. Claim prefetched data
    float3 texLocalPos = (currentPoint - texMinGlobal) / max(texMaxGlobal - texMinGlobal, 1e-6);
    float3 evalPoint = texLocalPos;
    bool needsFetch = !prefetchValid;
    float rawScalar = needsFetch
      ? volumeTexture.sample(sVolume, evalPoint, level(0)).r
      : prefetchScalar;
    float rawMask = (doMask && needsFetch)
      ? maskTexture.sample(sNearest, evalPoint, level(0)).r
      : prefetchMask;

    // 2. Advance ray trackers
    float3 lastPoint = currentPoint;
    currentPoint += stepVec;
    currentT += stepSize;

    // 3. LAUNCH PREFETCH FOR NEXT ITERATION
    if (i + 1 < maxSteps) {
      float3 nextTexLocalPos = (currentPoint - texMinGlobal) / max(texMaxGlobal - texMinGlobal, 1e-6);
      float3 nextEvalPoint = nextTexLocalPos;
      prefetchScalar = volumeTexture.sample(sVolume, nextEvalPoint, level(0)).r;
      if (doMask) {
        prefetchMask = maskTexture.sample(sNearest, nextEvalPoint, level(0)).r;
      }
      prefetchValid = true;
    }

    // 4. MATH & EVALUATION
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

    if (sampleOpacity > 0.001h) {
      half3 sampleColor = colorOpacity.rgb;
      half weight = 1.0h - accumulatedOpacity;

      // Ghost lighting bypass: skip expensive gradient/lighting for near-transparent voxels
      if (sampleOpacity < 0.01h) {
        accumulatedColor += weight * sampleColor * sampleOpacity;
        accumulatedOpacity += weight * sampleOpacity;
      } else {

      // Visual Significance Threshold:
      // If the voxel's actual contribution to the screen is less than 0.002
      // it is invisible on an 8-bit monitor. Do not waste memory bandwidth shading it.
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

        sampleColor = computePhongLightingVolumeFast(sampleColor, normal, lightDirHalf, viewDirHalf, ambientMat, diffuseMat, specularMat, shininessMat);

        if (doGradOp) {
          sampleOpacity *= half(gradientOpacityTexture.sample(sVolume, float2(float(gradMag), 0.5), level(0)).r);
        }
      } else if (doShading) {
        // Fallback for "invisible" fuzz/noise to maintain baseline brightness
        sampleColor = ambientMat * sampleColor;
      }

      accumulatedColor += weight * sampleColor * sampleOpacity;
      accumulatedOpacity += weight * sampleOpacity;
      } // end ghost lighting bypass
    }

    // Early Ray Termination (ERT)
    if (accumulatedOpacity >= 0.99h) {
      accumulatedColor /= max(accumulatedOpacity, 1e-4h);
      accumulatedOpacity = 1.0h;
      break;
    }
    if (currentT >= tTerminateMax) {
      break;
    }
  }

  output.color = float4(float3(accumulatedColor), float(accumulatedOpacity));
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
    texture3d<float> normalTexture [[texture(7)]]) {

  VolumeFragmentOut output;

  // Reconstruct the pixel ray in global [0,1] space — same convention as the
  // composite shader (fragment_layer_composite_main).
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float2 uv  = in.position.xy / volumeUniforms.viewportSize;
  float2 ndc = uv * 2.0 - 1.0;
  float4 wn = volumeUniforms.inverseViewProjection * float4(ndc.x, -ndc.y, 0.0, 1.0); wn.xyz /= wn.w;
  float4 wf = volumeUniforms.inverseViewProjection * float4(ndc.x, -ndc.y, 1.0, 1.0); wf.xyz /= wf.w;
  float3 bszF = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 localN = ((volumeUniforms.worldToVolume * float4(wn.xyz, 1.0)).xyz - volumeUniforms.volumeBoundsMin.xyz) / bszF;
  float3 localF = ((volumeUniforms.worldToVolume * float4(wf.xyz, 1.0)).xyz - volumeUniforms.volumeBoundsMin.xyz) / bszF;
  float3 rayDir = normalize(localF - localN);

  float3 blockMinGlobal = (b.volumeBoundsMin.xyz - volumeUniforms.volumeBoundsMin.xyz) / max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 blockMaxGlobal = (b.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz) / max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);

  float3 texMinGlobal = (b.textureBoundsMin.xyz - volumeUniforms.volumeBoundsMin.xyz) / max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 texMaxGlobal = (b.textureBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz) / max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);

  float stepSize = volumeUniforms.sampleDistance;

  float2 t = intersectBox(cameraPos, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) discard_fragment();

  float3 entryPoint = cameraPos + rayDir * tStart;
  float3 exitPoint = cameraPos + rayDir * t.y;
  float totalDist = length(exitPoint - entryPoint);

  if (volumeUniforms.useClipping > 0.5) {
    int numClipPlanes = int(volumeUniforms.numClippingPlanes);
    float4 planeOrigins[8] = { volumeUniforms.clippingPlane0Origin, volumeUniforms.clippingPlane1Origin, volumeUniforms.clippingPlane2Origin, volumeUniforms.clippingPlane3Origin, volumeUniforms.clippingPlane4Origin, volumeUniforms.clippingPlane5Origin, volumeUniforms.clippingPlane6Origin, volumeUniforms.clippingPlane7Origin };
    float4 planeNormals[8] = { volumeUniforms.clippingPlane0Normal, volumeUniforms.clippingPlane1Normal, volumeUniforms.clippingPlane2Normal, volumeUniforms.clippingPlane3Normal, volumeUniforms.clippingPlane4Normal, volumeUniforms.clippingPlane5Normal, volumeUniforms.clippingPlane6Normal, volumeUniforms.clippingPlane7Normal };

    for (int cp = 0; cp < numClipPlanes; cp++) {
      float3 planeOrigin = planeOrigins[cp].xyz;
      float3 planeNormal = normalize(planeNormals[cp].xyz);
      float startDistance = dot(planeNormal, planeOrigin - entryPoint);
      float stopDistance = dot(planeNormal, planeOrigin - exitPoint);

      if (startDistance > 0.0 && stopDistance > 0.0) discard_fragment();
      float rayDotNormal = dot(rayDir, planeNormal);

      if (rayDotNormal > 0.0 && startDistance > 0.0) entryPoint += (startDistance / rayDotNormal) * rayDir;
      if (rayDotNormal <= 0.0 && stopDistance > 0.0) exitPoint += (stopDistance / rayDotNormal) * rayDir;
    }
    totalDist = length(exitPoint - entryPoint);
    if (totalDist < 1e-6) discard_fragment();
  }

  float tTerminateMax = 1e30;
  if (volumeUniforms.useDepthTexture > 0.5) {
    float depthSample = depthTexture.sample(sNearest, in.position.xy / volumeUniforms.viewportSize).r;
    if (depthSample < 1.0) {
      float2 ndcXY = (in.position.xy / volumeUniforms.viewportSize) * 2.0 - 1.0;
      float4 worldTermination = volumeUniforms.inverseViewProjection * float4(ndcXY.x, -ndcXY.y, depthSample, 1.0);
      worldTermination.xyz /= worldTermination.w;
      float3 terminationLocal = (worldTermination.xyz - volumeUniforms.volumeBoundsMin.xyz) / max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
      tTerminateMax = dot(terminationLocal - entryPoint, rayDir);
      if (tTerminateMax < 0.0) tTerminateMax = 1e30;
    }
  }

  // --- LOCAL CACHE WARM-UP ---
  const bool doShading = fc_shading && (volumeUniforms.useGradientShading > 0.5);
  const bool doGradOp = fc_gradientOpacity && (volumeUniforms.useGradientOpacity > 0.5);
  const bool doCropping = volumeUniforms.useCropping > 0.5;
  const bool doMask = fc_mask && (volumeUniforms.useMask > 0.5);

  half scalarScale = half(1.0 / (volumeUniforms.scalarMax - volumeUniforms.scalarMin));
  half scalarBias  = half(-volumeUniforms.scalarMin) * scalarScale;

  half gradNormFactor = half(max(1e-8f, volumeUniforms.gradientOpacityRange.y));

  float3 texSizeGlobal = max(texMaxGlobal - texMinGlobal, 1e-6);
  float3 dt = max(volumeUniforms.gradientStep, 1e-8);
  half3 gradScale = half3(1.0 / (dt * texSizeGlobal));

  half3 viewDirHalf  = half3(normalize(entryPoint - cameraPos));
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

  uint cropBitmask = 0;
  if (doCropping) {
    float4 cropF[8] = { volumeUniforms.croppingFlagsRow0, volumeUniforms.croppingFlagsRow1, volumeUniforms.croppingFlagsRow2, volumeUniforms.croppingFlagsRow3, volumeUniforms.croppingFlagsRow4, volumeUniforms.croppingFlagsRow5, volumeUniforms.croppingFlagsRow6, volumeUniforms.croppingFlagsRow7 };
    for (int j = 0; j < 32; j++) {
      if (cropF[j / 4][j % 4] > 0.0) cropBitmask |= (1u << j);
    }
  }

  // --- SOFTWARE PIPELINING INITIALIZATION ---
  float jitter = volumeUniforms.useJittering > 0.5
    ? interleavedGradientNoise(in.position.xy + volumeUniforms.frameIndex) * stepSize
    : 0.0;
  float3 stepVec = rayDir * stepSize;
  float3 currentPoint = entryPoint + (rayDir * jitter);
  float currentT = jitter;

  int maxSteps = min(max(1, int(ceil(totalDist / stepSize))), MAX_RAY_STEPS);

  half3 accumulatedColor = half3(0.0);
  half accumulatedOpacity = 0.0;

  // PREFETCH the very first samples before the loop starts
  float3 texLocalPos0 = (currentPoint - texMinGlobal) / max(texMaxGlobal - texMinGlobal, 1e-6);
  float3 evalPoint0 = texLocalPos0;
  float prefetchScalar =   volumeTexture.sample(sVolume, evalPoint0, level(0)).r;
  float prefetchMask = doMask ? maskTexture.sample(sNearest, evalPoint0, level(0)).r : 0.0;
  bool prefetchValid = true;
  int3  curCell     = int3(-1);
  int3  curCoarseCell = int3(-1);
  bool  curCoarseEmpty = false;
  bool  curCellEmpty = false;
  float3 mmDimF     = b.minMaxInfo.yzw;

  // --- THE RAYMARCHING LOOP ---
  for (int i = 0; i < maxSteps; i++) {
    if (any(currentPoint < blockMinGlobal - 1e-4) || any(currentPoint > blockMaxGlobal + 1e-4)) break;

    // 0. MIN-MAX ACCELERATION (hierarchical, two-level)
    if (fc_minmax &&
        b.minMaxInfo.x > 0.5 &&
        b.minMaxInfo.y > 0.5 &&
        b.minMaxInfo.z > 0.5 &&
        b.minMaxInfo.w > 0.5) {
      float3 texLocalPos = (currentPoint - texMinGlobal) / max(texMaxGlobal - texMinGlobal, 1e-6);
      float3 mmPos = clamp(texLocalPos, float3(0.0), float3(1.0));
      float3 rayDirTexLocal = rayDir / max(texMaxGlobal - texMinGlobal, 1e-6);

      // Coarse level (mip 1, 2x)
      float3 coarseDimF = max(mmDimF * 0.5, 1.0);
      bool hasCoarse = b.minMaxInfo.x > 1.5;
      if (hasCoarse) {
        int3 newCoarse = min(int3(mmPos * coarseDimF), int3(coarseDimF) - 1);
        if (any(newCoarse != curCoarseCell)) {
          curCoarseCell = newCoarse;
          curCoarseEmpty = minMaxTexture.sample(sNearest, mmPos, level(1)).r > 0.5;
        }
        if (curCoarseEmpty) {
          float skipDist = computeMinMaxSkip(mmPos, coarseDimF, rayDir, rayDirTexLocal, stepSize);
          currentPoint += rayDir * skipDist;
          currentT += skipDist;
          if (any(currentPoint < blockMinGlobal - 1e-4) || any(currentPoint > blockMaxGlobal + 1e-4) || currentT >= t.y - tStart) break;
          prefetchValid = false;
          curCell = int3(-1);
          curCoarseCell = int3(-1);
          continue;
        }
      }

      // Fine level (mip 0)
      int3 newCell = min(int3(mmPos * mmDimF), int3(mmDimF) - 1);
      if (any(newCell != curCell)) {
        curCell      = newCell;
        curCellEmpty = minMaxTexture.sample(sNearest, mmPos, level(0)).r > 0.5;
      }
      if (curCellEmpty) {
        float skipDist = computeMinMaxSkip(mmPos, mmDimF, rayDir, rayDirTexLocal, stepSize);
        currentPoint += rayDir * skipDist;
        currentT += skipDist;
        if (any(currentPoint < blockMinGlobal - 1e-4) || any(currentPoint > blockMaxGlobal + 1e-4) || currentT >= t.y - tStart) break;
        prefetchValid = false;
        curCell = int3(-1);
        curCoarseCell = int3(-1);
        continue;
      }
    }

    // 1. Claim prefetched data
    float3 texLocalPos = (currentPoint - texMinGlobal) / max(texMaxGlobal - texMinGlobal, 1e-6);
    float3 evalPoint = texLocalPos;
    bool needsFetch = !prefetchValid;
    float rawScalar = needsFetch
      ? volumeTexture.sample(sVolume, evalPoint, level(0)).r
      : prefetchScalar;
    float rawMask = (doMask && needsFetch)
      ? maskTexture.sample(sNearest, evalPoint, level(0)).r
      : prefetchMask;

    // 2. Advance ray trackers
    float3 lastPoint = currentPoint;
    currentPoint += stepVec;
    currentT += stepSize;

    // 3. LAUNCH PREFETCH FOR NEXT ITERATION
    if (i + 1 < maxSteps) {
      float3 nextTexLocalPos = (currentPoint - texMinGlobal) / max(texMaxGlobal - texMinGlobal, 1e-6);
      float3 nextEvalPoint = nextTexLocalPos;
      prefetchScalar = volumeTexture.sample(sVolume, nextEvalPoint, level(0)).r;
      if (doMask) {
        prefetchMask = maskTexture.sample(sNearest, nextEvalPoint, level(0)).r;
      }
      prefetchValid = true;
    }

    // 4. MATH & EVALUATION
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

        sampleColor = computePhongLightingVolumeFast(sampleColor, normal, lightDirHalf, viewDirHalf, ambientMat, diffuseMat, specularMat, shininessMat);

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

    // Early Ray Termination (ERT)
    if (accumulatedOpacity >= 0.99h) {
      accumulatedColor /= max(accumulatedOpacity, 1e-4h);
      accumulatedOpacity = 1.0h;
      break;
    }
    if (currentT >= tTerminateMax) {
      break;
    }
  }

  output.color = float4(float3(accumulatedColor), float(accumulatedOpacity));
  return output;
}

// Inter-block accumulation shader: reads previous blocks' accumulated color/opacity
// via Metal framebuffer fetch ([[color(0)]]), enabling global early ray termination
// across block boundaries. Used when rendering partitioned volumes front-to-back.
fragment VolumeFragmentOut fragment_volume_accum_main(
    VolumeVertexOut in [[stage_in]],
    bool isFrontFace [[front_facing]],
    float4 prevAccum [[color(0)]], // Metal Framebuffer Fetch
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> transferFunctionTexture [[texture(1)]],
    texture2d<float> depthTexture [[texture(2)]],
    texture2d<float> gradientOpacityTexture [[texture(3)]],
    texture3d<float> maskTexture [[texture(4)]],
    texture2d<float> labelMapTransferTexture [[texture(5)]],
    texture3d<float> minMaxTexture [[texture(6)]],
    texture3d<float> normalTexture [[texture(7)]]) {

  if (!isFrontFace) discard_fragment();

  VolumeFragmentOut output;

  // Global ERT: Skip if previous blocks already made this pixel opaque
  if (prevAccum.a >= 0.99h) {
      discard_fragment();
  }

  // Initialize accumulators with previous block's results
  half3 accumulatedColor = half3(prevAccum.rgb);
  half accumulatedOpacity = half(prevAccum.a);

  float3 blockMinGlobal = (b.volumeBoundsMin.xyz - volumeUniforms.volumeBoundsMin.xyz) / max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 blockMaxGlobal = (b.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz) / max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);

  float3 texMinGlobal = (b.textureBoundsMin.xyz - volumeUniforms.volumeBoundsMin.xyz) / max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 texMaxGlobal = (b.textureBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz) / max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);

  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float stepSize = volumeUniforms.sampleDistance;

  float3 rayDir = in.localPos - cameraPos;
  float dirLength = length(rayDir);
  if (dirLength < 0.0001) discard_fragment();

  rayDir /= dirLength;
  float2 t = intersectBox(cameraPos, rayDir, blockMinGlobal, blockMaxGlobal);
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) discard_fragment();

  float3 entryPoint = cameraPos + rayDir * tStart;
  float3 exitPoint = cameraPos + rayDir * t.y;
  float totalDist = length(exitPoint - entryPoint);

  if (volumeUniforms.useClipping > 0.5) {
    int numClipPlanes = int(volumeUniforms.numClippingPlanes);
    float4 planeOrigins[8] = { volumeUniforms.clippingPlane0Origin, volumeUniforms.clippingPlane1Origin, volumeUniforms.clippingPlane2Origin, volumeUniforms.clippingPlane3Origin, volumeUniforms.clippingPlane4Origin, volumeUniforms.clippingPlane5Origin, volumeUniforms.clippingPlane6Origin, volumeUniforms.clippingPlane7Origin };
    float4 planeNormals[8] = { volumeUniforms.clippingPlane0Normal, volumeUniforms.clippingPlane1Normal, volumeUniforms.clippingPlane2Normal, volumeUniforms.clippingPlane3Normal, volumeUniforms.clippingPlane4Normal, volumeUniforms.clippingPlane5Normal, volumeUniforms.clippingPlane6Normal, volumeUniforms.clippingPlane7Normal };

    for (int cp = 0; cp < numClipPlanes; cp++) {
      float3 planeOrigin = planeOrigins[cp].xyz;
      float3 planeNormal = normalize(planeNormals[cp].xyz);
      float startDistance = dot(planeNormal, planeOrigin - entryPoint);
      float stopDistance = dot(planeNormal, planeOrigin - exitPoint);

      if (startDistance > 0.0 && stopDistance > 0.0) discard_fragment();
      float rayDotNormal = dot(rayDir, planeNormal);

      if (rayDotNormal > 0.0 && startDistance > 0.0) entryPoint += (startDistance / rayDotNormal) * rayDir;
      if (rayDotNormal <= 0.0 && stopDistance > 0.0) exitPoint += (stopDistance / rayDotNormal) * rayDir;
    }
    totalDist = length(exitPoint - entryPoint);
    if (totalDist < 1e-6) discard_fragment();
  }

  float tTerminateMax = 1e30;
  if (volumeUniforms.useDepthTexture > 0.5) {
    float depthSample = depthTexture.sample(sNearest, in.position.xy / volumeUniforms.viewportSize).r;
    if (depthSample < 1.0) {
      float2 ndcXY = (in.position.xy / volumeUniforms.viewportSize) * 2.0 - 1.0;
      float4 worldTermination = volumeUniforms.inverseViewProjection * float4(ndcXY.x, -ndcXY.y, depthSample, 1.0);
      worldTermination.xyz /= worldTermination.w;
      float3 terminationLocal = (worldTermination.xyz - volumeUniforms.volumeBoundsMin.xyz) / max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
      tTerminateMax = dot(terminationLocal - entryPoint, rayDir);
      if (tTerminateMax < 0.0) tTerminateMax = 1e30;
    }
  }

  // --- LOCAL CACHE WARM-UP ---
  const bool doShading = fc_shading && (volumeUniforms.useGradientShading > 0.5);
  const bool doGradOp = fc_gradientOpacity && (volumeUniforms.useGradientOpacity > 0.5);
  const bool doCropping = volumeUniforms.useCropping > 0.5;
  const bool doMask = fc_mask && (volumeUniforms.useMask > 0.5);

  half scalarScale = half(1.0 / (volumeUniforms.scalarMax - volumeUniforms.scalarMin));
  half scalarBias  = half(-volumeUniforms.scalarMin) * scalarScale;

  half gradNormFactor = half(max(1e-8f, volumeUniforms.gradientOpacityRange.y));

  float3 texSizeGlobalFrag2 = max(texMaxGlobal - texMinGlobal, 1e-6);
  float3 dtFrag2 = max(volumeUniforms.gradientStep, 1e-8);
  half3 gradScaleFrag2 = half3(1.0 / (dtFrag2 * texSizeGlobalFrag2));

  half3 viewDirHalf  = half3(normalize(entryPoint - cameraPos));
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

  uint cropBitmask = 0;
  if (doCropping) {
    float4 cropF[8] = { volumeUniforms.croppingFlagsRow0, volumeUniforms.croppingFlagsRow1, volumeUniforms.croppingFlagsRow2, volumeUniforms.croppingFlagsRow3, volumeUniforms.croppingFlagsRow4, volumeUniforms.croppingFlagsRow5, volumeUniforms.croppingFlagsRow6, volumeUniforms.croppingFlagsRow7 };
    for (int j = 0; j < 32; j++) {
      if (cropF[j / 4][j % 4] > 0.0) cropBitmask |= (1u << j);
    }
  }

  // --- SOFTWARE PIPELINING INITIALIZATION ---
  float jitter = volumeUniforms.useJittering > 0.5
    ? interleavedGradientNoise(in.position.xy + volumeUniforms.frameIndex) * stepSize
    : 0.0;
  float3 stepVec = rayDir * stepSize;
  float3 currentPoint = entryPoint + (rayDir * jitter);
  float currentT = jitter;

  int maxSteps = min(max(1, int(ceil(totalDist / stepSize))), MAX_RAY_STEPS);

  // PREFETCH the very first samples before the loop starts
  float3 texLocalPos0 = (currentPoint - texMinGlobal) / max(texMaxGlobal - texMinGlobal, 1e-6);
  float3 evalPoint0 = texLocalPos0;
  float prefetchScalar =   volumeTexture.sample(sVolume, evalPoint0, level(0)).r;
  float prefetchMask = doMask ? maskTexture.sample(sNearest, evalPoint0, level(0)).r : 0.0;
  bool prefetchValid = true;

  // MIN-MAX CELL CACHE
  int3  curCell     = int3(-1);
  int3  curCoarseCell = int3(-1);
  bool  curCoarseEmpty = false;
  bool  curCellEmpty = false;
  float3 mmDimF     = b.minMaxInfo.yzw;

  // --- THE RAYMARCHING LOOP ---
  for (int i = 0; i < maxSteps; i++) {
    if (any(currentPoint < blockMinGlobal - 1e-4) || any(currentPoint > blockMaxGlobal + 1e-4)) break;

    if (fc_minmax &&
        b.minMaxInfo.x > 0.5 &&
        b.minMaxInfo.y > 0.5 &&
        b.minMaxInfo.z > 0.5 &&
        b.minMaxInfo.w > 0.5) {
      float3 texLocalPos = (currentPoint - texMinGlobal) / max(texMaxGlobal - texMinGlobal, 1e-6);
      float3 mmPos = clamp(texLocalPos, float3(0.0), float3(1.0));
      float3 rayDirTexLocal = rayDir / max(texMaxGlobal - texMinGlobal, 1e-6);

      // Coarse level (mip 1, 2x)
      float3 coarseDimF = max(mmDimF * 0.5, 1.0);
      bool hasCoarse = b.minMaxInfo.x > 1.5;
      if (hasCoarse) {
        int3 newCoarse = min(int3(mmPos * coarseDimF), int3(coarseDimF) - 1);
        if (any(newCoarse != curCoarseCell)) {
          curCoarseCell = newCoarse;
          curCoarseEmpty = minMaxTexture.sample(sNearest, mmPos, level(1)).r > 0.5;
        }
        if (curCoarseEmpty) {
          float skipDist = computeMinMaxSkip(mmPos, coarseDimF, rayDir, rayDirTexLocal, stepSize);
          currentPoint += rayDir * skipDist;
          currentT += skipDist;
          if (any(currentPoint < blockMinGlobal - 1e-4) || any(currentPoint > blockMaxGlobal + 1e-4) || currentT >= t.y - tStart) break;
          prefetchValid = false;
          curCell = int3(-1);
          curCoarseCell = int3(-1);
          continue;
        }
      }

      // Fine level (mip 0)
      int3 newCell = min(int3(mmPos * mmDimF), int3(mmDimF) - 1);
      if (any(newCell != curCell)) {
        curCell      = newCell;
        curCellEmpty = minMaxTexture.sample(sNearest, mmPos, level(0)).r > 0.5;
      }
      if (curCellEmpty) {
        float skipDist = computeMinMaxSkip(mmPos, mmDimF, rayDir, rayDirTexLocal, stepSize);
        currentPoint += rayDir * skipDist;
        currentT += skipDist;
        if (any(currentPoint < blockMinGlobal - 1e-4) || any(currentPoint > blockMaxGlobal + 1e-4) || currentT >= t.y - tStart) break;
        prefetchValid = false;
        curCell = int3(-1);
        curCoarseCell = int3(-1);
        continue;
      }
    }

    // 1. Claim prefetched data
    float3 texLocalPos = (currentPoint - texMinGlobal) / max(texMaxGlobal - texMinGlobal, 1e-6);
    float3 evalPoint = texLocalPos;
    bool needsFetch = !prefetchValid;
    float rawScalar = needsFetch
      ? volumeTexture.sample(sVolume, evalPoint, level(0)).r
      : prefetchScalar;
    float rawMask = (doMask && needsFetch)
      ? maskTexture.sample(sNearest, evalPoint, level(0)).r
      : prefetchMask;

    // 2. Advance ray trackers
    float3 lastPoint = currentPoint;
    currentPoint += stepVec;
    currentT += stepSize;

    // 3. LAUNCH PREFETCH FOR NEXT ITERATION
    if (i + 1 < maxSteps) {
      float3 nextTexLocalPos = (currentPoint - texMinGlobal) / max(texMaxGlobal - texMinGlobal, 1e-6);
      float3 nextEvalPoint = nextTexLocalPos;
      prefetchScalar = volumeTexture.sample(sVolume, nextEvalPoint, level(0)).r;
      if (doMask) {
        prefetchMask = maskTexture.sample(sNearest, nextEvalPoint, level(0)).r;
      }
      prefetchValid = true;
    }

    // 4. MATH & EVALUATION
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

    if (sampleOpacity > 0.001h) {
      half3 sampleColor = colorOpacity.rgb;
      half weight = 1.0h - accumulatedOpacity;

      // Ghost lighting bypass: skip expensive gradient/lighting for near-transparent voxels
      if (sampleOpacity < 0.01h) {
        accumulatedColor += weight * sampleColor * sampleOpacity;
        accumulatedOpacity += weight * sampleOpacity;
      } else {

      // Visual Significance Threshold
      if (doShading && maskLabel == 0.0h && (sampleOpacity * weight > 0.002h)) {

        half3 normal;
        half gradMag;

        if (fc_normalTexture && volumeUniforms.useNormalTexture > 0.5) {
          half4 nrmSample = half4(normalTexture.sample(sVolume, evalPoint, level(0)));
          normal = normalize(nrmSample.xyz * 2.0h - 1.0h);
          gradMag = nrmSample.w;
        } else {
          half4 grad = computeGradientFast(volumeTexture, evalPoint, b.gradientStep.xyz, gradScaleFrag2, gradNormFactor);
          normal = grad.xyz;
          gradMag = grad.w;
        }

        sampleColor = computePhongLightingVolumeFast(sampleColor, normal, lightDirHalf, viewDirHalf, ambientMat, diffuseMat, specularMat, shininessMat);

        if (doGradOp) {
          sampleOpacity *= half(gradientOpacityTexture.sample(sVolume, float2(float(gradMag), 0.5), level(0)).r);
        }
      } else if (doShading) {
        // Fallback for "invisible" fuzz/noise to maintain baseline brightness
        sampleColor = ambientMat * sampleColor;
      }

      accumulatedColor += weight * sampleColor * sampleOpacity;
      accumulatedOpacity += weight * sampleOpacity;
      } // end ghost lighting bypass
    }

    // Early Ray Termination (ERT)
    if (accumulatedOpacity >= 0.99h) {
      accumulatedColor /= max(accumulatedOpacity, 1e-4h);
      accumulatedOpacity = 1.0h;
      break;
    }
    if (currentT >= tTerminateMax) {
      break;
    }
  }

  output.color = float4(float3(accumulatedColor), float(accumulatedOpacity));
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
    float tf = (cellMin - u.scalarMin) * u.scalarScale;
    uint idxMin = clamp(uint(tf), 0u, 255u);
    tf = (cellMax - u.scalarMin) * u.scalarScale;
    uint idxMax = clamp(uint(tf), 0u, 255u);
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

// Conservative mip: coarse cell is solid if ANY child is solid.
// Empty only if ALL 2x2x2 children are empty.  This guarantees no
// false-negatives at the coarse level (empty coarse == empty children).
kernel void volume_minmax_mip(
    texture3d<float, access::read> fineLevel [[texture(0)]],
    texture3d<float, access::write> coarseLevel [[texture(1)]],
    uint3 gid [[thread_position_in_grid]])
{
  uint3 fineDims = uint3(fineLevel.get_width(), fineLevel.get_height(), fineLevel.get_depth());
  uint3 coarseDims = uint3(coarseLevel.get_width(), coarseLevel.get_height(), coarseLevel.get_depth());
  if (any(gid >= coarseDims)) return;

  bool solid = false;
  uint3 childBase = gid * 2;
  for (uint dz = 0; dz < 2 && !solid; dz++) {
    for (uint dy = 0; dy < 2 && !solid; dy++) {
      for (uint dx = 0; dx < 2 && !solid; dx++) {
        uint3 child = childBase + uint3(dx, dy, dz);
        if (all(child < fineDims)) {
          if (fineLevel.read(child).r < 0.5) {
            solid = true;
          }
        }
      }
    }
  }

  coarseLevel.write(solid ? 0.0 : 1.0, gid);
}

fragment float4 fragment_image_sample_blit(
    FullscreenVertexOut in [[stage_in]],
    texture2d<float> offscreenColor [[texture(0)]]) {
  return offscreenColor.sample(sVolume, in.texCoord);
}

// --- ORDER-INDEPENDENT COMPOSITING: per-brick layer composite ---
// Reads up to 8 layer textures (one per brick), reconstructs the pixel ray,
// intersects each brick box to get true ray-entry depth, sorts per-pixel,
// and folds the premultiplied layers front-to-back. Exact == unpartitioned march.
struct LayerCompositeUniforms {
    float4 blockMin[8];  // brick bounding boxes in global [0,1] space
    float4 blockMax[8];
    float4 params;       // x = brickCount, yzw unused
};

fragment float4 fragment_layer_composite_main(
    FullscreenVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& u  [[buffer(1)]],
    constant LayerCompositeUniforms& lc [[buffer(2)]],
    texture2d_array<float, access::read> layers [[texture(0)]])
{
    uint2 pixel = uint2(in.position.xy);
    int count = int(lc.params.x);
    if (count <= 0) { return float4(0.0, 0.0, 0.0, 0.0); }

    // Read only up to `count` active slices from the texture array.
    float4 L[8];
    for (int i = 0; i < min(count, 8); ++i) {
        L[i] = layers.read(pixel, i);
    }

    // Reconstruct the pixel ray in global [0,1] space, identical convention to brick shaders.
    float3 cameraPos = u.cameraVolumePos.xyz;
    float2 uv  = in.position.xy / u.viewportSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 wn = u.inverseViewProjection * float4(ndc.x, -ndc.y, 0.0, 1.0); wn.xyz /= wn.w;
    float4 wf = u.inverseViewProjection * float4(ndc.x, -ndc.y, 1.0, 1.0); wf.xyz /= wf.w;
    float3 bsz = max(u.volumeBoundsMax.xyz - u.volumeBoundsMin.xyz, 1e-6);
    float3 localN = ((u.worldToVolume * float4(wn.xyz, 1.0)).xyz - u.volumeBoundsMin.xyz) / bsz;
    float3 localF = ((u.worldToVolume * float4(wf.xyz, 1.0)).xyz - u.volumeBoundsMin.xyz) / bsz;
    float3 dir = normalize(localF - localN);

    // Gather active layers (alpha>0) with their true ray-entry param.
    int   idx[8]; float tE[8]; int m = 0;
    for (int i = 0; i < count; ++i) {
        if (L[i].a < 1e-4) continue;
        float2 tt = intersectBox(cameraPos, dir, lc.blockMin[i].xyz, lc.blockMax[i].xyz);
        if (tt.x <= tt.y) { idx[m] = i; tE[m] = tt.x; ++m; }
    }
    // Per-pixel depth sort (the fix): always sort by ray-entry depth.
    for (int i = 1; i < m; ++i) {
        float kt = tE[i]; int ki = idx[i]; int j = i - 1;
        while (j >= 0 && tE[j] > kt) { tE[j+1] = tE[j]; idx[j+1] = idx[j]; --j; }
        tE[j+1] = kt; idx[j+1] = ki;
    }
    // Front-to-back fold of premultiplied layers: R = R + L*(1 - R.a). Exact == unpartitioned march.
    float3 R = 0.0; float Ra = 0.0;
    for (int p = 0; p < m; ++p) {
        float4 c = L[idx[p]];
        float  w = 1.0 - Ra;
        R  += c.rgb * w;
        Ra += c.a   * w;
    }
    return float4(R, saturate(Ra));
}