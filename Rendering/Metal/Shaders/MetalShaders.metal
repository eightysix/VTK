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

  float3 matColor = material.color.rgb;
  float3 matSpecular = material.specular.rgb * material.specular.w;

  float3 totalAmbient = matColor * 0.3;
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

    totalDiffuse += matColor * df * lightColor * attenuation;

    float NdotL = max(dot(N, toLight), 0.0);
    if (NdotL > 0.0) {
      float sf = pow(max(dot(viewDir, reflDir), 0.0), material.specularPower);
      totalSpecular += matSpecular * sf * lightColor * attenuation;
    }
  }

  return float4(saturate(totalAmbient + totalDiffuse + totalSpecular), material.opacity);
}
