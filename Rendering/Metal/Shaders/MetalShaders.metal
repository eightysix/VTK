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
};

// Basic 1px point vertex shader — positions stored as packed float3 array.
// Buffer(0) = positions (3 floats per point), buffer(1) = SceneUniforms.
vertex PointVertexOut vertex_point_main(
    uint vertex_id [[vertex_id]],
    constant float3* point_positions [[buffer(0)]],
    constant SceneUniforms& scene [[buffer(1)]]) {
  PointVertexOut out;
  float3 pos = point_positions[vertex_id];

  float4 worldPos = scene.modelMatrix * float4(pos, 1.0);
  float4 viewPos = scene.viewMatrix * worldPos;
  out.viewPos = viewPos.xyz;
  out.position = scene.projectionMatrix * viewPos;
  out.viewNormal = float3(0.0, 0.0, 1.0);
  out.point_size = 1.0;
  return out;
}

// Fragment shader for 1px points — identical lighting to triangle/line fragment.
fragment float4 fragment_point_main(PointVertexOut in [[stage_in]],
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
  return float4(totalAmbient + diffuseIntensity * totalDiffuse + totalSpecular, material.opacity);
}

// -----------------------------------------------------------------------
// Shaped point vertex shader — quad-per-point instancing
// Buffer(0) = point positions (float3[]), buffer(1) = connectivity (uint[]),
// buffer(2) = SceneUniforms.
// Drawn as MTLPrimitiveTypeTriangleStrip, 4 vertices, N instances.
// -----------------------------------------------------------------------

struct PointShapedVertexOut {
  float4 position [[position]];
  float3 viewPos;
  float3 viewNormal;
  float2 p_coord;
};

vertex PointShapedVertexOut vertex_point_shaped_main(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    constant float3* point_positions [[buffer(0)]],
    constant uint* connectivity [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]]) {
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
  out.viewNormal = float3(0.0, 0.0, 1.0);
  out.p_coord = corner;
  return out;
}

// Fragment for shaped points — same lighting, with optional round discard.
fragment float4 fragment_point_shaped_main(
    PointShapedVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]]) {
  // Discard fragments outside unit circle for round points
  if (dot(in.p_coord, in.p_coord) > 1.0) {
    discard_fragment();
  }

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
  return float4(totalAmbient + diffuseIntensity * totalDiffuse + totalSpecular, material.opacity);
}
