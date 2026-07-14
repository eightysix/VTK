// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
//
// Metal shaders for VTK Metal rendering backend.
//

#include <metal_stdlib>
using namespace metal;

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
  float _padMask[3];
};

struct VolumeVertexOut {
  float4 position [[position]];
  float3 localPos;
};

struct VolumeVertexIn {
  float3 position [[attribute(0)]];
};

vertex VolumeVertexOut vertex_volume_main(
    VolumeVertexIn in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]]) {
  VolumeVertexOut out;

  float3 modelPos = in.position;
  out.position = volumeUniforms.viewProjection * volumeUniforms.volumeToWorld * float4(modelPos, 1.0);
  out.localPos = (modelPos - volumeUniforms.volumeBoundsMin.xyz) / (volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz);
  return out;
}

struct VolumeFragmentOut { float4 color [[color(0)]]; };

constant int MAX_RAY_STEPS = 2000;

inline float volume_random(float2 st) {
  return fract(sin(dot(st.xy, float2(12.9898, 78.233))) * 43758.5453123);
}

inline float2 intersectBox(float3 orig, float3 dir, float3 boxMin, float3 boxMax) {
  float3 invDir = 1.0 / (dir + float3(1e-8));
  float3 tbot = invDir * (boxMin - orig);
  float3 ttop = invDir * (boxMax - orig);
  float3 tmin = min(ttop, tbot);
  float3 tmax = max(ttop, tbot);
  return float2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

inline float4 computeGradient(texture3d<float> volTex, sampler volSamp, float3 pos, float3 gradStep, float gradNormFactor) {
  float sPX = volTex.sample(volSamp, pos + float3(gradStep.x, 0, 0), level(0)).r;
  float sNX = volTex.sample(volSamp, pos - float3(gradStep.x, 0, 0), level(0)).r;
  float sPY = volTex.sample(volSamp, pos + float3(0, gradStep.y, 0), level(0)).r;
  float sNY = volTex.sample(volSamp, pos - float3(0, gradStep.y, 0), level(0)).r;
  float sPZ = volTex.sample(volSamp, pos + float3(0, 0, gradStep.z), level(0)).r;
  float sNZ = volTex.sample(volSamp, pos - float3(0, 0, gradStep.z), level(0)).r;

  float3 grad = float3(sPX - sNX, sPY - sNY, sPZ - sNZ);
  float mag = length(grad);
  return float4(mag > 0.0 ? grad / mag : float3(0.0), clamp(mag / max(1e-8, gradNormFactor), 0.0, 1.0));
}

inline half3 computePhongLightingVolume(half3 sampleColor, float3 gradDir, float3 lightDir, float3 viewDir, half3 ambientMat, half3 diffuseMat, half3 specularMat, float shininess) {
  float nDotL = dot(gradDir, -lightDir);
  half3 diffuse = half3(0.0);
  half3 specular = half3(0.0);

  if (nDotL > 0.0) {
    diffuse = half3(nDotL) * diffuseMat * sampleColor;
    float3 r = normalize(2.0 * nDotL * gradDir + lightDir);
    float vDotR = max(dot(r, -viewDir), 0.0);
    specular = half3(pow(vDotR, shininess)) * specularMat;
  }
  return ambientMat * sampleColor + diffuse + specular;
}

// Branchless, fast crop region evaluator
inline int computeCropRegion(float3 cropMin, float3 cropMax, float3 pos) {
  int3 r = 1 + int3(step(cropMin, pos)) + int3(step(cropMax, pos));
  return r.x + (r.y - 1) * 3 + (r.z - 1) * 9;
}

fragment VolumeFragmentOut fragment_volume_main(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> transferFunctionTexture [[texture(1)]],
    texture2d<float> depthTexture [[texture(2)]],
    texture2d<float> gradientOpacityTexture [[texture(3)]],
    texture3d<float> maskTexture [[texture(4)]],
    texture2d<float> labelMapTransferTexture [[texture(5)]],
    texture2d<float> labelMapGradientOpacityTexture [[texture(6)]],
    sampler transferFunctionSampler [[sampler(0)]],
    sampler volumeSampler [[sampler(1)]],
    sampler depthSampler [[sampler(2)]],
    sampler gradientOpacitySampler [[sampler(3)]],
    sampler maskSampler [[sampler(4)]],
    sampler labelMapSampler [[sampler(5)]],
    sampler labelMapGradOpSampler [[sampler(6)]]) {
  
  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float stepSize = volumeUniforms.sampleDistance;

  float3 rayDir = in.localPos - cameraPos;
  float dirLength = length(rayDir);
  if (dirLength < 0.0001) discard_fragment();
  
  rayDir /= dirLength;
  float2 t = intersectBox(cameraPos, rayDir, float3(0.0), float3(1.0));
  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) discard_fragment();

  float3 entryPoint = cameraPos + rayDir * tStart;
  float3 exitPoint = cameraPos + rayDir * t.y;
  float totalDist = length(exitPoint - entryPoint);

  float tTerminateMax = 1e30;
  float depthSample = depthTexture.sample(depthSampler, in.position.xy / volumeUniforms.viewportSize).r;

  if (depthSample < 1.0) {
    float2 ndcXY = (in.position.xy / volumeUniforms.viewportSize) * 2.0 - 1.0;
    float4 worldTermination = volumeUniforms.inverseViewProjection * float4(ndcXY.x, -ndcXY.y, depthSample, 1.0);
    float3 terminationLocal = ((worldTermination.xyz / worldTermination.w) - volumeUniforms.volumeBoundsMin.xyz) / (volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz);
    tTerminateMax = length(terminationLocal - entryPoint);
  }

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

  float jitter = volumeUniforms.useJittering > 0.5 ? volume_random(in.position.xy) * stepSize : 0.0;
  float3 currentPoint = entryPoint + (rayDir * jitter);
  float3 stepVec = rayDir * stepSize;

  half3 accumulatedColor = half3(0.0);
  half accumulatedOpacity = 0.0;
  half scalarRangeRcp = half(1.0 / (volumeUniforms.scalarMax - volumeUniforms.scalarMin));

  const bool doShading = volumeUniforms.useGradientShading > 0.5;
  const bool doGradOp = volumeUniforms.useGradientOpacity > 0.5;
  const bool doCropping = volumeUniforms.useCropping > 0.5;
  const bool doMask = volumeUniforms.useMask > 0.5;

  float3 cropMin = volumeUniforms.croppingPlanes.xyz;
  float3 cropMax = float3(volumeUniforms.croppingPlanes.w, volumeUniforms.croppingPlanes2.xy);
  
  // Collapse clipping map array lookups to a dynamic branching-free bitmask (huge performance boost on fast inner loops)
  uint cropBitmask = 0;
  if (doCropping) {
    float4 cropF[8] = { volumeUniforms.croppingFlagsRow0, volumeUniforms.croppingFlagsRow1, volumeUniforms.croppingFlagsRow2, volumeUniforms.croppingFlagsRow3, volumeUniforms.croppingFlagsRow4, volumeUniforms.croppingFlagsRow5, volumeUniforms.croppingFlagsRow6, volumeUniforms.croppingFlagsRow7 };
    for (int j = 0; j < 32; j++) {
      if (cropF[j / 4][j % 4] > 0.0) {
        cropBitmask |= (1u << j);
      }
    }
  }

  int maxSteps = min(max(1, int(ceil(totalDist / stepSize))), MAX_RAY_STEPS);
  float currentT = jitter;

  int i = 0;
  int maxStepsEven = maxSteps & ~1;
  
  for (; i < maxStepsEven; i += 2) {
    // --- Sample 1 ---
    bool cropped1 = false;
    if (doCropping) {
      if ((cropBitmask & (1u << computeCropRegion(cropMin, cropMax, currentPoint))) == 0u) cropped1 = true;
    }

    if (!cropped1) {
      float rawScalar = volumeTexture.sample(volumeSampler, currentPoint, level(0)).r;
      half scalarNorm = clamp(half(rawScalar - volumeUniforms.scalarMin) * scalarRangeRcp, 0.0h, 1.0h);

      half4 colorOpacity;
      half maskLabel = 0.0h;
      if (doMask) {
        float maskVal = maskTexture.sample(maskSampler, currentPoint, level(0)).r * volumeUniforms.maskScale + volumeUniforms.maskBias;
        if (volumeUniforms.labelMapNumLabels > 0.0) {
          maskLabel = half(floor(maskVal * volumeUniforms.labelMapNumLabels) / volumeUniforms.labelMapNumLabels);
        }
        if (maskLabel > 0.0h) {
          colorOpacity = half4(labelMapTransferTexture.sample(labelMapSampler, float2(float(scalarNorm), float(maskLabel)), level(0)));
        } else {
          colorOpacity = half4(transferFunctionTexture.sample(transferFunctionSampler, float2(float(scalarNorm), 0.5), level(0)));
        }
      } else {
        colorOpacity = half4(transferFunctionTexture.sample(transferFunctionSampler, float2(float(scalarNorm), 0.5), level(0)));
      }

      half sampleOpacity = colorOpacity.a;
      if (sampleOpacity > 0.001h) {
        half3 sampleColor = colorOpacity.rgb;

        if (doShading && maskLabel == 0.0h) {
          float4 grad = computeGradient(volumeTexture, volumeSampler, currentPoint, volumeUniforms.gradientStep, volumeUniforms.gradientOpacityRange.y);
          sampleColor = computePhongLightingVolume(sampleColor, grad.xyz, normalize(volumeUniforms.lightDirection), normalize(entryPoint - cameraPos), half3(volumeUniforms.ambientColor.rgb), half3(volumeUniforms.diffuseColor.rgb), half3(volumeUniforms.specularColor.rgb), volumeUniforms.shininess);

          if (doGradOp) sampleOpacity *= half(gradientOpacityTexture.sample(gradientOpacitySampler, float2(float(grad.w), 0.5), level(0)).r);
        }

        half w = 1.0h - accumulatedOpacity;
        accumulatedColor += w * sampleColor * sampleOpacity;
        accumulatedOpacity += w * sampleOpacity;
      }
    }

    currentPoint += stepVec;
    currentT += stepSize;
    if (accumulatedOpacity >= 0.95h || currentT >= tTerminateMax) {
      accumulatedOpacity = 1.0h;
      break;
    }

    // --- Sample 2 ---
    bool cropped2 = false;
    if (doCropping) {
      if ((cropBitmask & (1u << computeCropRegion(cropMin, cropMax, currentPoint))) == 0u) cropped2 = true;
    }

    if (!cropped2) {
      float rawScalar = volumeTexture.sample(volumeSampler, currentPoint, level(0)).r;
      half scalarNorm = clamp(half(rawScalar - volumeUniforms.scalarMin) * scalarRangeRcp, 0.0h, 1.0h);

      half4 colorOpacity;
      half maskLabel = 0.0h;
      if (doMask) {
        float maskVal = maskTexture.sample(maskSampler, currentPoint, level(0)).r * volumeUniforms.maskScale + volumeUniforms.maskBias;
        if (volumeUniforms.labelMapNumLabels > 0.0) {
          maskLabel = half(floor(maskVal * volumeUniforms.labelMapNumLabels) / volumeUniforms.labelMapNumLabels);
        }
        if (maskLabel > 0.0h) {
          colorOpacity = half4(labelMapTransferTexture.sample(labelMapSampler, float2(float(scalarNorm), float(maskLabel)), level(0)));
        } else {
          colorOpacity = half4(transferFunctionTexture.sample(transferFunctionSampler, float2(float(scalarNorm), 0.5), level(0)));
        }
      } else {
        colorOpacity = half4(transferFunctionTexture.sample(transferFunctionSampler, float2(float(scalarNorm), 0.5), level(0)));
      }

      half sampleOpacity = colorOpacity.a;
      if (sampleOpacity > 0.001h) {
        half3 sampleColor = colorOpacity.rgb;

        if (doShading && maskLabel == 0.0h) {
          float4 grad = computeGradient(volumeTexture, volumeSampler, currentPoint, volumeUniforms.gradientStep, volumeUniforms.gradientOpacityRange.y);
          sampleColor = computePhongLightingVolume(sampleColor, grad.xyz, normalize(volumeUniforms.lightDirection), normalize(entryPoint - cameraPos), half3(volumeUniforms.ambientColor.rgb), half3(volumeUniforms.diffuseColor.rgb), half3(volumeUniforms.specularColor.rgb), volumeUniforms.shininess);

          if (doGradOp) sampleOpacity *= half(gradientOpacityTexture.sample(gradientOpacitySampler, float2(float(grad.w), 0.5), level(0)).r);
        }

        half w = 1.0h - accumulatedOpacity;
        accumulatedColor += w * sampleColor * sampleOpacity;
        accumulatedOpacity += w * sampleOpacity;
      }
    }

    currentPoint += stepVec;
    currentT += stepSize;
    if (accumulatedOpacity >= 0.95h || currentT >= tTerminateMax) {
      accumulatedOpacity = 1.0h;
      break;
    }
  }

  // Handle remaining odd iterations if bound requires
  for (; i < maxSteps; i++) {
    bool croppedOdd = false;
    if (doCropping) {
      int regionNo = computeCropRegion(cropMin, cropMax, currentPoint);
      if ((cropBitmask & (1u << regionNo)) == 0u) croppedOdd = true;
    }

    if (!croppedOdd) {
      float rawScalar = volumeTexture.sample(volumeSampler, currentPoint, level(0)).r;
      half scalarNorm = clamp(half(rawScalar - volumeUniforms.scalarMin) * scalarRangeRcp, 0.0h, 1.0h);

      half4 colorOpacity;
      half maskLabel = 0.0h;
      if (doMask) {
        float maskVal = maskTexture.sample(maskSampler, currentPoint, level(0)).r * volumeUniforms.maskScale + volumeUniforms.maskBias;
        if (volumeUniforms.labelMapNumLabels > 0.0) {
          maskLabel = half(floor(maskVal * volumeUniforms.labelMapNumLabels) / volumeUniforms.labelMapNumLabels);
        }
        if (maskLabel > 0.0h) {
          colorOpacity = half4(labelMapTransferTexture.sample(labelMapSampler, float2(float(scalarNorm), float(maskLabel)), level(0)));
        } else {
          colorOpacity = half4(transferFunctionTexture.sample(transferFunctionSampler, float2(float(scalarNorm), 0.5), level(0)));
        }
      } else {
        colorOpacity = half4(transferFunctionTexture.sample(transferFunctionSampler, float2(float(scalarNorm), 0.5), level(0)));
      }

      half sampleOpacity = colorOpacity.a;
      if (sampleOpacity > 0.001h) {
        half3 sampleColor = colorOpacity.rgb;

        if (doShading && maskLabel == 0.0h) {
          float4 grad = computeGradient(volumeTexture, volumeSampler, currentPoint, volumeUniforms.gradientStep, volumeUniforms.gradientOpacityRange.y);
          sampleColor = computePhongLightingVolume(sampleColor, grad.xyz, normalize(volumeUniforms.lightDirection), normalize(entryPoint - cameraPos), half3(volumeUniforms.ambientColor.rgb), half3(volumeUniforms.diffuseColor.rgb), half3(volumeUniforms.specularColor.rgb), volumeUniforms.shininess);

          if (doGradOp) {
            sampleOpacity *= half(gradientOpacityTexture.sample(gradientOpacitySampler, float2(float(grad.w), 0.5), level(0)).r);
          }
        }

        half w = 1.0h - accumulatedOpacity;
        accumulatedColor += w * sampleColor * sampleOpacity;
        accumulatedOpacity += w * sampleOpacity;
      }
    }
    
    currentT += stepSize;
    if (accumulatedOpacity >= 0.95h || currentT >= tTerminateMax) {
      accumulatedOpacity = 1.0h;
      break;
    }
    currentPoint += stepVec;
  }

  output.color = float4(float3(accumulatedColor), float(accumulatedOpacity));
  return output;
}

fragment float4 fragment_image_sample_blit(
    FullscreenVertexOut in [[stage_in]],
    texture2d<float> offscreenColor [[texture(0)]],
    sampler offscreenSampler [[sampler(0)]]) {
  return offscreenColor.sample(offscreenSampler, in.texCoord);
}