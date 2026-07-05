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
};

// ---------------------------------------------------------------------------
// Vertex shader
// ---------------------------------------------------------------------------
vertex VertexOut vertex_main(VertexIn in [[stage_in]],
                             constant SceneUniforms& scene [[buffer(2)]]) {
  VertexOut out;

  float4 worldPos = scene.modelMatrix * float4(in.position, 1.0);
  float4 viewPos = scene.viewMatrix * worldPos;
  out.viewPos = viewPos.xyz;

  out.position = scene.projectionMatrix * viewPos;

  out.viewNormal = scene.normalMatrix * in.normal;

  return out;
}

// ---------------------------------------------------------------------------
// Fragment shader — Phong lighting matching WebGPU backend
// ---------------------------------------------------------------------------
fragment float4 fragment_main(VertexOut in [[stage_in]],
                              constant MaterialUniforms& material [[buffer(0)]],
                              constant LightUniforms& lights [[buffer(1)]]) {
  float3 N = normalize(in.viewNormal);

  float3 ambientColor = material.ambientColor.rgb;
  float ambientIntensity = material.ambientColor.w;
  float3 diffuseColor = material.diffuseColor.rgb;
  float diffuseIntensity = material.diffuseColor.w;
  float3 specularColor = material.specularColor.rgb;
  float specularIntensity = material.specularColor.w;

  float3 totalAmbient = ambientIntensity * ambientColor;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);

  float3 viewDir = normalize(-in.viewPos);

  for (int i = 0; i < lights.lightCount && i < MAX_LIGHTS; ++i) {
    Light L = lights.lights[i];
    int lightType = int(L.position.w);
    float3 lightColor = L.color.rgb * L.color.w;
    float attenuation = 1.0;
    float df = 0.0;
    float3 reflDir = float3(0.0);
    float3 toLight = float3(0.0);

    if (lightType == 0) {
      // Headlight — matches WebGPU: df = max(0.000001, normal_VC.z)
      toLight = float3(0.0, 0.0, 1.0);
      df = max(N.z, 0.000001);
      reflDir = reflect(float3(0.0, 0.0, -1.0), N);
    } else if (lightType == 1) {
      // Directional
      toLight = normalize(-L.direction.xyz);
      df = max(dot(N, toLight), 0.0);
      reflDir = reflect(L.direction.xyz, N);
    } else {
      // Point or spot
      toLight = L.position.xyz - in.viewPos;
      float dist = length(toLight);
      toLight /= dist;
      attenuation = 1.0 / (L.attenuation.x + L.attenuation.y * dist + L.attenuation.z * dist * dist);
      df = max(dot(N, toLight), 0.0);
      reflDir = reflect(-toLight, N);

      if (lightType == 3) {
        float3 spotDir = normalize(L.direction.xyz);
        float spotCos = dot(-toLight, spotDir);
        float spotCutoff = cos(L.direction.w * M_PI_F / 180.0);
        if (spotCos > spotCutoff) {
          attenuation *= pow(spotCos, L.attenuation.w);
        } else {
          attenuation = 0.0;
        }
      }
    }

    totalDiffuse += df * diffuseColor * lightColor * attenuation;

    float NdotL = max(dot(N, toLight), 0.0);
    if (NdotL > 0.0) {
      float sf = pow(max(dot(viewDir, reflDir), 0.0), material.specularPower);
      totalSpecular += sf * specularIntensity * specularColor * lightColor * attenuation;
    }
  }

  return float4(totalAmbient + diffuseIntensity * totalDiffuse + totalSpecular, material.opacity);
}

// ---------------------------------------------------------------------------
// Point rendering shaders (basic 1px and shaped)
// ---------------------------------------------------------------------------

struct PointVertexOut {
  float4 position [[position]];
  float point_size [[point_size]];
  float3 viewPos;
  float3 viewNormal;
  float4 pointColor;
};

// Basic 1px point vertex shader — positions stored as packed float3 array.
// Buffer(0) = positions (3 floats per point), buffer(1) = SceneUniforms,
// buffer(2) = point_normals (3 floats per point), buffer(3) = point_colors (4 floats per point).
vertex PointVertexOut vertex_point_main(
    uint vertex_id [[vertex_id]],
    constant float3* point_positions [[buffer(0)]],
    constant SceneUniforms& scene [[buffer(1)]],
    constant float3* point_normals [[buffer(2)]],
    constant float4* point_colors [[buffer(3)]]) {
  PointVertexOut out;
  float3 pos = point_positions[vertex_id];

  float4 worldPos = scene.modelMatrix * float4(pos, 1.0);
  float4 viewPos = scene.viewMatrix * worldPos;
  out.viewPos = viewPos.xyz;
  out.position = scene.projectionMatrix * viewPos;
  out.viewNormal = scene.normalMatrix * point_normals[vertex_id];
  out.point_size = 1.0;
  out.pointColor = point_colors[vertex_id];
  return out;
}

// Fragment shader for 1px points — identical lighting to triangle/line fragment.
fragment float4 fragment_point_main(PointVertexOut in [[stage_in]],
                                    constant MaterialUniforms& material [[buffer(0)]],
                                    constant LightUniforms& lights [[buffer(1)]]) {
  float3 N = normalize(in.viewNormal);
  // Use per-point color for ambient/diffuse when available (matches WebGPU)
  float3 ambientColor = in.pointColor.rgb;
  float ambientIntensity = material.ambientColor.w;
  float3 diffuseColor = in.pointColor.rgb;
  float diffuseIntensity = material.diffuseColor.w;
  float3 specularColor = material.specularColor.rgb;
  float specularIntensity = material.specularColor.w;
  float3 totalAmbient = ambientIntensity * ambientColor;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);
  float3 viewDir = normalize(-in.viewPos);
  for (int i = 0; i < lights.lightCount && i < MAX_LIGHTS; ++i) {
    Light L = lights.lights[i];
    int lightType = int(L.position.w);
    float3 lightColor = L.color.rgb * L.color.w;
    float attenuation = 1.0;
    float df = 0.0;
    float3 reflDir = float3(0.0);
    float3 toLight = float3(0.0);
    if (lightType == 0) {
      toLight = float3(0.0, 0.0, 1.0);
      df = max(N.z, 0.000001);
      reflDir = reflect(float3(0.0, 0.0, -1.0), N);
    } else if (lightType == 1) {
      toLight = normalize(-L.direction.xyz);
      df = max(dot(N, toLight), 0.0);
      reflDir = reflect(L.direction.xyz, N);
    } else {
      toLight = L.position.xyz - in.viewPos;
      float dist = length(toLight);
      toLight /= dist;
      attenuation = 1.0 / (L.attenuation.x + L.attenuation.y * dist + L.attenuation.z * dist * dist);
      df = max(dot(N, toLight), 0.0);
      reflDir = reflect(-toLight, N);
      if (lightType == 3) {
        float3 spotDir = normalize(L.direction.xyz);
        float spotCos = dot(-toLight, spotDir);
        float spotCutoff = cos(L.direction.w * M_PI_F / 180.0);
        if (spotCos > spotCutoff) {
          attenuation *= pow(spotCos, L.attenuation.w);
        } else {
          attenuation = 0.0;
        }
      }
    }
    totalDiffuse += df * diffuseColor * lightColor * attenuation;
    float NdotL = max(dot(N, toLight), 0.0);
    if (NdotL > 0.0) {
      float sf = pow(max(dot(viewDir, reflDir), 0.0), material.specularPower);
      totalSpecular += sf * specularIntensity * specularColor * lightColor * attenuation;
    }
  }
  return float4(totalAmbient + diffuseIntensity * totalDiffuse + totalSpecular, in.pointColor.a * material.opacity);
}

// -----------------------------------------------------------------------
// Shaped point vertex shader — quad-per-point instancing
// Buffer(0) = point positions (float3[]), buffer(1) = connectivity (uint[]),
// buffer(2) = SceneUniforms, buffer(3) = point normals (float3[]),
// buffer(4) = point colors (float4[]).
// Drawn as MTLPrimitiveTypeTriangleStrip, 4 vertices, N instances.
// -----------------------------------------------------------------------

struct PointShapedVertexOut {
  float4 position [[position]];
  float3 viewPos;
  float3 viewNormal;
  float2 p_coord;
  float4 pointColor;
};

vertex PointShapedVertexOut vertex_point_shaped_main(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    constant float3* point_positions [[buffer(0)]],
    constant uint* connectivity [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant float3* point_normals [[buffer(3)]],
    constant float4* point_colors [[buffer(4)]]) {
  // Quad corners matching WebGPU TRIANGLE_VERTS
  const float2 tri_verts[4] = {
    float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1)
  };

  uint point_id = connectivity[instance_id];
  float3 pos = point_positions[point_id];

  float4 worldPos = scene.modelMatrix * float4(pos, 1.0);
  float4 viewPos = scene.viewMatrix * worldPos;
  float4 clipPos = scene.projectionMatrix * viewPos;

  // Convert to screen space, expand quad by point_size
  float2 resolution = scene.viewport.zw;
  float2 screenPos = resolution * (0.5 * clipPos.xy / clipPos.w + 0.5);
  float ptSize = scene.pointSize;
  float2 corner = tri_verts[vertex_id];
  float2 expanded = screenPos + 0.5 * ptSize * corner;

  PointShapedVertexOut out;
  out.position = float4(clipPos.w * ((2.0 * expanded) / resolution - 1.0),
                        clipPos.z, clipPos.w);
  out.viewPos = viewPos.xyz;
  out.viewNormal = scene.normalMatrix * point_normals[point_id];
  out.p_coord = corner;
  out.pointColor = point_colors[point_id];
  return out;
}

// Fragment output struct with explicit depth — needed for sphere depth correction.
struct PointFragmentOutput {
  float4 color [[color(0)]];
  float depth [[depth(any)]];
};

// Fragment for shaped points — sphere shading with depth correction.
// Matches WebGPU's ReplaceFragmentShaderNormals for POINTS_SHAPED.
// Flags: bit 0 = parallel projection, bit 5 = render as spheres, bit 7 = point 2D shape (0=round,1=square)
fragment PointFragmentOutput fragment_point_shaped_main(
    PointShapedVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]]) {
  PointFragmentOutput out;

  float d = length(in.p_coord);

  // Bit 5: render points as spheres, bit 7: point 2D shape (0=round, 1=square)
  bool drawSpheres = (scene.flags & (1u << 5)) != 0u;
  bool isRound = ((scene.flags >> 7) & 1u) == 0u; // 0=Round, 1=Square

  // Discard fragments outside the point shape
  if ((isRound || drawSpheres) && d > 1.0) {
    discard_fragment();
  }

  float3 N;
  if (drawSpheres && d <= 1.0) {
    // Compute fake sphere normal from p_coord — matches WebGPU
    N = normalize(float3(in.p_coord, 1.0));
    N.z = sqrt(1.0 - d * d);

    // Fake sphere depth correction.
    // See Rendering/OpenGL2/PixelsToZBufferConversion.txt for the math.
    // WebGPU depth is [0,1].
    float pointSize = clamp(scene.pointSize, 1.0, 100000.0);
    float r = pointSize / (scene.viewport.z * scene.projectionMatrix[0][0]);
    bool parallel = (scene.flags & 1u) != 0u;
    if (parallel) {
      float s = scene.projectionMatrix[2][2];
      out.depth = in.position.z + N.z * r * s;
    } else {
      float s = -scene.projectionMatrix[2][2];
      out.depth = (s - in.position.z) / (N.z * r - 1.0) + s;
    }
  } else {
    N = normalize(in.viewNormal);
    out.depth = in.position.z;
  }

  // Use per-point color for ambient/diffuse when available (matches WebGPU)
  float3 ambientColor = in.pointColor.rgb;
  float ambientIntensity = material.ambientColor.w;
  float3 diffuseColor = in.pointColor.rgb;
  float diffuseIntensity = material.diffuseColor.w;
  float3 specularColor = material.specularColor.rgb;
  float specularIntensity = material.specularColor.w;
  float3 totalAmbient = ambientIntensity * ambientColor;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);
  float3 viewDir = normalize(-in.viewPos);
  for (int i = 0; i < lights.lightCount && i < MAX_LIGHTS; ++i) {
    Light L = lights.lights[i];
    int lightType = int(L.position.w);
    float3 lightColor = L.color.rgb * L.color.w;
    float attenuation = 1.0;
    float df = 0.0;
    float3 reflDir = float3(0.0);
    float3 toLight = float3(0.0);
    if (lightType == 0) {
      toLight = float3(0.0, 0.0, 1.0);
      df = max(N.z, 0.000001);
      reflDir = reflect(float3(0.0, 0.0, -1.0), N);
    } else if (lightType == 1) {
      toLight = normalize(-L.direction.xyz);
      df = max(dot(N, toLight), 0.0);
      reflDir = reflect(L.direction.xyz, N);
    } else {
      toLight = L.position.xyz - in.viewPos;
      float dist = length(toLight);
      toLight /= dist;
      attenuation = 1.0 / (L.attenuation.x + L.attenuation.y * dist + L.attenuation.z * dist * dist);
      df = max(dot(N, toLight), 0.0);
      reflDir = reflect(-toLight, N);
      if (lightType == 3) {
        float spotDir = normalize(L.direction.xyz);
        float spotCos = dot(-toLight, spotDir);
        float spotCutoff = cos(L.direction.w * M_PI_F / 180.0);
        if (spotCos > spotCutoff) {
          attenuation *= pow(spotCos, L.attenuation.w);
        } else {
          attenuation = 0.0;
        }
      }
    }
    totalDiffuse += df * diffuseColor * lightColor * attenuation;
    float NdotL = max(dot(N, toLight), 0.0);
    if (NdotL > 0.0) {
      float sf = pow(max(dot(viewDir, reflDir), 0.0), material.specularPower);
      totalSpecular += sf * specularIntensity * specularColor * lightColor * attenuation;
    }
  }
  out.color = float4(totalAmbient + diffuseIntensity * totalDiffuse + totalSpecular, in.pointColor.a * material.opacity);
  return out;
}
