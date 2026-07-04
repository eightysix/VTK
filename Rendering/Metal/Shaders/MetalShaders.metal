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
  float3x3 normalMatrix;       // inverse-transpose of view * model
  float4x4 modelMatrix;
  float4 viewport;             // x, y, width, height
  uint flags;
};

// Per-material uniforms
struct MaterialUniforms {
  float4 color;                // rgba
  float4 ambient;              // rgb + intensity
  float4 diffuse;              // rgb + intensity
  float4 specular;             // rgb + intensity
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
// Fragment shader — Phong lighting
// ---------------------------------------------------------------------------
fragment float4 fragment_main(VertexOut in [[stage_in]],
                              constant MaterialUniforms& material [[buffer(0)]],
                              constant LightUniforms& lights [[buffer(1)]]) {
  float3 N = normalize(in.viewNormal);

  // Accumulate lighting
  float3 ambientAccum = float3(0.0);
  float3 diffuseAccum = float3(0.0);
  float3 specularAccum = float3(0.0);

  float3 matAmbient = material.color.rgb * material.ambient.w;
  float3 matDiffuse = material.color.rgb * material.diffuse.w;
  float3 matSpecular = material.specular.rgb * material.specular.w;

  // In view space, camera is at origin
  float3 viewDir = normalize(-in.viewPos);

  for (int i = 0; i < lights.lightCount && i < MAX_LIGHTS; ++i) {
    Light L = lights.lights[i];
    int lightType = int(L.position.w);

    float attenuation = 1.0;

    if (lightType == 0) {
      // Headlight: use the normal's z component directly (positive = facing camera = lit)
      diffuseAccum += matDiffuse * L.color.rgb * L.color.w * max(N.z, 0.000001) * attenuation;

      if (N.z > 0.0) {
        float3 halfDir = normalize(float3(0.0, 0.0, -1.0) + viewDir);
        float NdotH = max(dot(N, halfDir), 0.0);
        specularAccum += matSpecular * L.color.rgb * L.color.w *
                         pow(NdotH, material.specularPower) * attenuation;
      }

      ambientAccum += matAmbient * L.color.rgb * L.color.w;
      continue;
    }

    float3 toLight;
    if (lightType == 1) {
      // Directional — direction points FROM light TO scene, negate for surface-to-light
      toLight = normalize(-L.direction.xyz);
    } else {
      // Point or spot — positions already in view space
      toLight = L.position.xyz - in.viewPos;
      float dist = length(toLight);
      toLight = toLight / dist;

      float attenConst = L.attenuation.x;
      float attenLinear = L.attenuation.y;
      float attenQuad = L.attenuation.z;
      attenuation = 1.0 / (attenConst + attenLinear * dist + attenQuad * dist * dist);

      // Spot light
      if (lightType == 3) {
        float3 spotDir = normalize(L.direction.xyz);
        float spotCos = dot(-toLight, spotDir);
        float spotAngle = L.direction.w;
        float spotExponent = L.attenuation.w;
        float spotCosCutoff = cos(spotAngle * M_PI_F / 180.0);
        if (spotCos < spotCosCutoff) {
          attenuation = 0.0;
        } else {
          attenuation *= pow(spotCos, spotExponent);
        }
      }
    }

    float NdotL = max(dot(N, toLight), 0.0);

    // Diffuse
    diffuseAccum += matDiffuse * L.color.rgb * L.color.w * NdotL * attenuation;

    // Specular (Blinn-Phong)
    if (NdotL > 0.0) {
      float3 halfDir = normalize(toLight + viewDir);
      float NdotH = max(dot(N, halfDir), 0.0);
      specularAccum += matSpecular * L.color.rgb * L.color.w *
                       pow(NdotH, material.specularPower) * attenuation;
    }

    // Ambient — no attenuation
    ambientAccum += matAmbient * L.color.rgb * L.color.w;
  }

  float3 color = ambientAccum + diffuseAccum + specularAccum;

  return float4(saturate(color), material.opacity);
}
