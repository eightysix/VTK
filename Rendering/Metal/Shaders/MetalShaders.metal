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

// Coincident topology offset (P1-5) — separate uniform buffer
struct CoincidentOffsetUniforms {
  float polygonFactor;         // slope-scale factor for polygons
  float polygonOffset;         // constant offset for polygons
  float lineFactor;            // slope-scale factor for lines
  float lineOffset;            // constant offset for lines
  float pointOffset;           // constant offset for points
};

// Vertex color override (P1-4) — used when vertex visibility is on
struct VertexColorUniforms {
  float4 color;                // vertex color (rgb + alpha)
};

// Clipping planes (P1-6)
struct ClipPlaneUniforms {
  float4 planes[6];            // up to 6 clip planes (ax+by+cz+d)
  int numClipPlanes;
};

// Cell ID offset (P2-7) — for homogeneous cell size variants
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
  float4 clipDistances;  // P1-6: clip plane distances (x,y,z,w for planes 0-3)
  uint cellId;           // P2-8: flat-interpolated cell ID (1-based, 0=background)
  uint propId;           // P2-8: flat-interpolated prop ID (1-based, 0=background)
};

// Fragment output with explicit depth — needed for coincident topology offset
struct FragmentOutput {
  float4 color [[color(0)]];
  uint4 ids [[color(1)]];   // P2-8: {cell, prop, composite, process} IDs
  float depth [[depth(any)]];
};

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

  // P1-1A: per-vertex color from scalar mapping
  out.vertexColor = vertexColors[vertex_id];

  // P5-5A: texture coordinates
  out.uv = triangleUVs[vertex_id];

  // P1-6: compute clip distances for up to 4 planes
  out.clipDistances = float4(
    dot(float4(in.position, 1.0), clipPlanes.planes[0]),
    dot(float4(in.position, 1.0), clipPlanes.planes[1]),
    dot(float4(in.position, 1.0), clipPlanes.planes[2]),
    dot(float4(in.position, 1.0), clipPlanes.planes[3]));

  // P2-8: picking IDs (already 1-based from compute kernel)
  out.cellId = cellIds[vertex_id];
  out.propId = propId + 1u;

  return out;
}

// ---------------------------------------------------------------------------
// Fragment shader — Phong lighting matching WebGPU backend
// Includes coincident topology offset (P1-5).
// ---------------------------------------------------------------------------
fragment FragmentOutput fragment_main(VertexOut in [[stage_in]],
                              constant MaterialUniforms& material [[buffer(0)]],
                              constant LightUniforms& lights [[buffer(1)]],
                              constant SceneUniforms& scene [[buffer(2)]],
                              constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
                              constant ClipPlaneUniforms& clipPlanes [[buffer(5)]],
                              texture2d<float> actorTexture [[texture(0)]],
                              sampler actorSampler [[sampler(0)]]) {
  // P1-6: discard fragments outside clip planes
  if (clipPlanes.numClipPlanes > 0 && in.clipDistances.x < 0.0) discard_fragment();
  if (clipPlanes.numClipPlanes > 1 && in.clipDistances.y < 0.0) discard_fragment();
  if (clipPlanes.numClipPlanes > 2 && in.clipDistances.z < 0.0) discard_fragment();
  if (clipPlanes.numClipPlanes > 3 && in.clipDistances.w < 0.0) discard_fragment();

  float3 N = normalize(in.viewNormal);

  // P1-1A: per-vertex color overrides material colors when active (bit 8)
  bool hasVertexColors = (scene.flags & (1u << 8)) != 0u;
  float3 ambientColor = hasVertexColors ? in.vertexColor.rgb : material.ambientColor.rgb;
  float ambientIntensity = material.ambientColor.w;
  float3 diffuseColor = hasVertexColors ? in.vertexColor.rgb : material.diffuseColor.rgb;
  float diffuseIntensity = material.diffuseColor.w;
  float3 specularColor = material.specularColor.rgb;
  float specularIntensity = material.specularColor.w;
  float baseOpacity = hasVertexColors ? in.vertexColor.a : material.opacity;

  // P5-5A: texture sampling — when bit 9 is set, multiply colors by texture sample
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

  FragmentOutput out;
  out.color = float4(totalAmbient + diffuseIntensity * totalDiffuse + totalSpecular, baseOpacity);
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);  // P2-8: {cell, prop, composite=1, process=0}
  // Coincident topology offset for polygons — matches WebGPU
  float c_factor = coinOffset.polygonFactor;
  float c_offset = coinOffset.polygonOffset;
  float cscale = length(float2(dfdx(in.position.z), dfdy(in.position.z)));
  out.depth = in.position.z + c_factor * cscale + c_offset / 65000.0;
  return out;
}

// ---------------------------------------------------------------------------
// P2-2B: Edge fragment shader — outputs flat edge color from uniform buffer
// Used for wireframe overlay when edge visibility is on.
// ---------------------------------------------------------------------------
fragment FragmentOutput fragment_edge_main(VertexOut in [[stage_in]],
                                   constant MaterialUniforms& material [[buffer(0)]],
                                   constant LightUniforms& lights [[buffer(1)]],
                                   constant SceneUniforms& scene [[buffer(2)]],
                                   constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
                                   constant float4& edgeColor [[buffer(4)]]) {
  FragmentOutput out;

  // Output flat edge color with full opacity
  out.color = float4(edgeColor.rgb, edgeColor.a * material.opacity);
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);

  // Coincident topology offset for lines — push edges forward to avoid z-fighting
  float c_factor = coinOffset.lineFactor;
  float c_offset = coinOffset.lineOffset;
  float cscale = length(float2(dfdx(in.position.z), dfdy(in.position.z)));
  out.depth = in.position.z + c_factor * cscale + c_offset / 65000.0;
  return out;
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
  float3 tangent;    // P2-9: tangent in view space
  float2 uv;         // P2-10: texture coordinates
  float2 lut_uv;     // P2-10: color texture coordinates
  uint cellId;       // P2-8: flat-interpolated cell ID
  uint propId;       // P2-8: flat-interpolated prop ID
};

// Basic 1px point vertex shader — positions stored as packed float3 array.
// Buffer(0) = positions, buffer(1) = SceneUniforms, buffer(2) = normals,
// buffer(3) = colors, buffer(6) = tangents, buffer(7) = uvs, buffer(8) = color_uvs.
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

// Fragment shader for 1px points — coincident offset + vertex visibility.
// Flags: bit 3 = vertex visibility
fragment FragmentOutput fragment_point_main(PointVertexOut in [[stage_in]],
                                    constant MaterialUniforms& material [[buffer(0)]],
                                    constant LightUniforms& lights [[buffer(1)]],
                                    constant SceneUniforms& scene [[buffer(2)]],
                                    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
                                    constant VertexColorUniforms& vertexColorUniform [[buffer(4)]]) {
  float3 N = normalize(in.viewNormal);

  // Determine color: vertex visibility overrides per-point color (matches WebGPU)
  bool showVertices = (scene.flags & (1u << 3)) != 0u;
  float3 baseColor = showVertices ? vertexColorUniform.color.rgb : in.pointColor.rgb;
  float baseAlpha = showVertices ? vertexColorUniform.color.a : in.pointColor.a;

  float3 ambientColor = baseColor;
  float ambientIntensity = material.ambientColor.w;
  float3 diffuseColor = baseColor;
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
  FragmentOutput out;
  out.color = float4(totalAmbient + diffuseIntensity * totalDiffuse + totalSpecular, baseAlpha * material.opacity);
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);  // P2-8
  // Coincident topology offset for points
  out.depth = in.position.z + coinOffset.pointOffset / 65000.0;
  return out;
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
  float3 tangent;    // P2-9: tangent in view space
  float2 uv;         // P2-10: texture coordinates
  float2 lut_uv;     // P2-10: color texture coordinates
  uint cellId;       // P2-8: flat-interpolated cell ID
  uint propId;       // P2-8: flat-interpolated prop ID
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
  out.tangent = scene.normalMatrix * point_tangents[point_id];
  out.uv = point_uvs[point_id];
  out.lut_uv = point_color_uvs[point_id];
  out.cellId = shapedCellIds[point_id];
  out.propId = shapedPropId + 1u;
  return out;
}

// Fragment output struct with explicit depth — needed for sphere depth correction.
struct PointFragmentOutput {
  float4 color [[color(0)]];
  uint4 ids [[color(1)]];   // P2-8: {cell, prop, composite, process} IDs
  float depth [[depth(any)]];
};

// Fragment for shaped points — sphere shading, depth correction, vertex visibility.
// Matches WebGPU's ReplaceFragmentShaderNormals for POINTS_SHAPED.
// Flags: bit 0 = parallel projection, bit 3 = vertex visibility,
//        bit 5 = render as spheres, bit 7 = point 2D shape (0=round,1=square)
fragment PointFragmentOutput fragment_point_shaped_main(
    PointShapedVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant VertexColorUniforms& vertexColorUniform [[buffer(4)]]) {
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

  // Determine color: vertex visibility overrides per-point color (matches WebGPU)
  bool showVertices = (scene.flags & (1u << 3)) != 0u;
  float3 baseColor = showVertices ? vertexColorUniform.color.rgb : in.pointColor.rgb;
  float baseAlpha = showVertices ? vertexColorUniform.color.a : in.pointColor.a;

  float3 ambientColor = baseColor;
  float ambientIntensity = material.ambientColor.w;
  float3 diffuseColor = baseColor;
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
  out.color = float4(totalAmbient + diffuseIntensity * totalDiffuse + totalSpecular, baseAlpha * material.opacity);
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);  // P2-8
  // Apply point coincident offset to depth (additive, after sphere depth correction)
  out.depth += coinOffset.pointOffset / 65000.0;
  return out;
}

// ---------------------------------------------------------------------------
// P3-3A: Thick line shaders — screen-space quad expansion (NoJoin)
// Each line segment is expanded into a screen-space quad (4 vertices per instance).
// Vertex shader reads both endpoints from the index buffer, transforms to screen
// space, and expands perpendicular to the line direction by lineWidth/2.
// ---------------------------------------------------------------------------

struct ThickLineVertexOut {
  float4 position [[position]];
  float3 viewPos;
  float3 viewNormal;
  float4 vertexColor;
  float dist_to_centerline;  // perpendicular distance from center (-0.5..0.5)
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
  // Quad corners: (-1,-1), (1,-1), (-1,1), (1,1)
  const float2 tri_verts[4] = {
    float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1)
  };

  float2 p_coord = tri_verts[vertex_id];

  // Read the two endpoint indices for this line segment
  uint p0_idx = lineIndices[instance_id * 2];
  uint p1_idx = lineIndices[instance_id * 2 + 1];

  float3 p0_MC = positions[p0_idx];
  float3 p1_MC = positions[p1_idx];

  // Transform both endpoints to clip space
  float4 p0_DC = scene.projectionMatrix * scene.viewMatrix * scene.modelMatrix * float4(p0_MC, 1.0);
  float4 p1_DC = scene.projectionMatrix * scene.viewMatrix * scene.modelMatrix * float4(p1_MC, 1.0);

  // Transform to screen space
  float2 resolution = scene.viewport.zw;
  float2 p0_screen = resolution * (0.5 * p0_DC.xy / p0_DC.w + 0.5);
  float2 p1_screen = resolution * (0.5 * p1_DC.xy / p1_DC.w + 0.5);

  // Compute line direction and perpendicular in screen space
  float2 x_basis = normalize(p1_screen - p0_screen);
  // Handle degenerate case (zero-length segment)
  float segLen = length(p1_screen - p0_screen);
  x_basis = select(x_basis, float2(1.0, 0.0), segLen < 0.001);
  float2 y_basis = float2(-x_basis.y, x_basis.x);

  // Expand into a quad by lineWidth
  float w = max(lineWidth, 1.0);
  float2 adjusted_p0 = p0_screen + p_coord.x * x_basis + p_coord.y * y_basis * w;
  float2 adjusted_p1 = p1_screen + p_coord.x * x_basis + p_coord.y * y_basis * w;
  float2 p = mix(adjusted_p0, adjusted_p1, p_coord.x);

  // Select z/w from the appropriate endpoint
  float4 p_DC = mix(p0_DC, p1_DC, p_coord.x);

  ThickLineVertexOut out;
  out.position = float4(p_DC.w * ((2.0 * p) / resolution - 1.0), p_DC.z, p_DC.w);

  // Interpolate view-space position for lighting
  float3 mid_MC = mix(p0_MC, p1_MC, p_coord.x);
  out.viewPos = (scene.viewMatrix * scene.modelMatrix * float4(mid_MC, 1.0)).xyz;
  out.viewNormal = scene.normalMatrix * float3(0.0, 0.0, 1.0);

  // Use the color from the starting endpoint of the segment
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

  // Tube-like shading: modify normal based on distance from centerline
  float3 N = normalize(in.viewNormal);
  float d = abs(in.dist_to_centerline);
  N.z = 1.0 - 2.0 * d;
  N = normalize(N);

  float3 totalAmbient = material.ambientColor.w * baseColor;
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

    totalDiffuse += df * baseColor * lightColor * attenuation;
    float NdotL = max(dot(N, toLight), 0.0);
    if (NdotL > 0.0) {
      float sf = pow(max(dot(viewDir, reflDir), 0.0), material.specularPower);
      totalSpecular += sf * material.specularColor.w * material.specularColor.rgb * lightColor * attenuation;
    }
  }

  out.color = float4(totalAmbient + material.diffuseColor.w * totalDiffuse + totalSpecular, baseAlpha);
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);

  // Coincident topology offset for lines
  float c_factor = coinOffset.lineFactor;
  float c_offset = coinOffset.lineOffset;
  float cscale = length(float2(dfdx(in.position.z), dfdy(in.position.z)));
  out.depth = in.position.z + c_factor * cscale + c_offset / 65000.0;
  return out;
}

// ---------------------------------------------------------------------------
// P3-3B: Round Cap + Round Join line shaders — screen-space quad expansion with semicircle caps
// Each line segment is expanded into a screen-space shape (36 vertices per instance):
//   - 6 vertices for the body quad (2 triangles)
//   - 15 vertices for the left semicircle cap (triangle fan, 5 segments × 3 verts)
//   - 15 vertices for the right semicircle cap (triangle fan, 5 segments × 3 verts)
// The third coordinate (p_coord.z) is used for interpolation along the line:
//   z=0 → p0 endpoint (left cap), z=1 → p1 endpoint (right cap)
// Fragment shader computes distance from line center for anti-aliasing.
// ---------------------------------------------------------------------------

struct RoundCapLineVertexOut {
  float4 position [[position]];
  float3 viewPos;
  float3 viewNormal;
  float4 vertexColor;
  float dist_to_centerline;  // perpendicular distance from center (-0.5..0.5)
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
  // 36-vertex template: 6 body + 15 left cap + 15 right cap (triangle strip)
  // Body quad: (-0.5,0,0), (-0.5,0,1), (0.5,0,0), (0.5,0,1), (-0.5,1,0), (-0.5,1,1)
  // The x coordinate maps to the semicircle offset (0=center, ±0.5=edges)
  // The y coordinate maps to the parametric position along the line (0=start, 1=end)
  // The z coordinate maps to interpolation factor (0=p0, 1=p1)

  // Generate template vertex for this vertex_id
  float3 p_coord;
  const int CAP_SEGMENTS = 5;
  const float PI = 3.14159265358979;

  if (vertex_id < 6) {
    // Body quad (triangle strip): alternating bottom/top vertices
    // v0=(-0.5,0,0), v1=(-0.5,0,1), v2=(0.5,0,0), v3=(0.5,0,1), v4=(-0.5,1,0), v5=(-0.5,1,1)
    const float3 body_verts[6] = {
      float3(-0.5, 0.0, 0.0),
      float3(-0.5, 0.0, 1.0),
      float3( 0.5, 0.0, 0.0),
      float3( 0.5, 0.0, 1.0),
      float3(-0.5, 1.0, 0.0),
      float3(-0.5, 1.0, 1.0)
    };
    p_coord = body_verts[vertex_id];
  } else if (vertex_id < 21) {
    // Left semicircle cap (triangle fan, 5 segments × 3 verts)
    // Fan center at (0, 0, 0), arcs from angle PI/2 to 3PI/2
    int local = vertex_id - 6;
    int seg = local / 3;
    int triVert = local % 3;
    float theta0 = PI * 0.5 + (float(seg) * PI) / float(CAP_SEGMENTS);
    float theta1 = PI * 0.5 + (float(seg + 1) * PI) / float(CAP_SEGMENTS);
    if (triVert == 0) {
      p_coord = float3(0.0, 0.0, 0.0);
    } else if (triVert == 1) {
      p_coord = float3(0.5 * cos(theta0), 0.5 * sin(theta0), 0.0);
    } else {
      p_coord = float3(0.5 * cos(theta1), 0.5 * sin(theta1), 0.0);
    }
  } else {
    // Right semicircle cap (triangle fan, 5 segments × 3 verts)
    // Fan center at (0, 1, 1), arcs from angle 3PI/2 to 5PI/2
    int local = vertex_id - 21;
    int seg = local / 3;
    int triVert = local % 3;
    float theta0 = 1.5 * PI + (float(seg) * PI) / float(CAP_SEGMENTS);
    float theta1 = 1.5 * PI + (float(seg + 1) * PI) / float(CAP_SEGMENTS);
    if (triVert == 0) {
      p_coord = float3(0.0, 1.0, 1.0);
    } else if (triVert == 1) {
      p_coord = float3(0.5 * cos(theta0), 0.5 * sin(theta0) + 1.0, 1.0);
    } else {
      p_coord = float3(0.5 * cos(theta1), 0.5 * sin(theta1) + 1.0, 1.0);
    }
  }

  // Read the two endpoint indices for this line segment
  uint p0_idx = lineIndices[instance_id * 2];
  uint p1_idx = lineIndices[instance_id * 2 + 1];

  float3 p0_MC = positions[p0_idx];
  float3 p1_MC = positions[p1_idx];

  // Transform both endpoints to clip space
  float4 p0_DC = scene.projectionMatrix * scene.viewMatrix * scene.modelMatrix * float4(p0_MC, 1.0);
  float4 p1_DC = scene.projectionMatrix * scene.viewMatrix * scene.modelMatrix * float4(p1_MC, 1.0);

  // Transform to screen space
  float2 resolution = scene.viewport.zw;
  float2 p0_screen = resolution * (0.5 * p0_DC.xy / p0_DC.w + 0.5);
  float2 p1_screen = resolution * (0.5 * p1_DC.xy / p1_DC.w + 0.5);

  // Compute line direction and perpendicular in screen space
  float2 x_basis = normalize(p1_screen - p0_screen);
  float segLen = length(p1_screen - p0_screen);
  x_basis = select(x_basis, float2(1.0, 0.0), segLen < 0.001);
  float2 y_basis = float2(-x_basis.y, x_basis.x);

  // Expand into shape by lineWidth
  float w = max(lineWidth, 1.0);
  float2 adjusted_p0 = p0_screen + (p_coord.x * x_basis + p_coord.y * y_basis) * w;
  float2 adjusted_p1 = p1_screen + (p_coord.x * x_basis + p_coord.y * y_basis) * w;
  float2 p = mix(adjusted_p0, adjusted_p1, p_coord.z);

  // Select z/w from the appropriate endpoint based on p_coord.z
  float4 p_DC = mix(p0_DC, p1_DC, p_coord.z);

  RoundCapLineVertexOut out;
  out.position = float4(p_DC.w * ((2.0 * p) / resolution - 1.0), p_DC.z, p_DC.w);

  // Interpolate view-space position for lighting
  float3 mid_MC = mix(p0_MC, p1_MC, p_coord.z);
  out.viewPos = (scene.viewMatrix * scene.modelMatrix * float4(mid_MC, 1.0)).xyz;
  out.viewNormal = scene.normalMatrix * float3(0.0, 0.0, 1.0);

  // Use the color from the appropriate endpoint
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

  // Tube-like shading: modify normal based on distance from centerline
  float3 N = normalize(in.viewNormal);
  float d = abs(in.dist_to_centerline);
  N.z = 1.0 - 2.0 * d;
  N = normalize(N);

  float3 totalAmbient = material.ambientColor.w * baseColor;
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

    totalDiffuse += df * baseColor * lightColor * attenuation;
    float NdotL = max(dot(N, toLight), 0.0);
    if (NdotL > 0.0) {
      float sf = pow(max(dot(viewDir, reflDir), 0.0), material.specularPower);
      totalSpecular += sf * material.specularColor.w * material.specularColor.rgb * lightColor * attenuation;
    }
  }

  out.color = float4(totalAmbient + material.diffuseColor.w * totalDiffuse + totalSpecular, baseAlpha);
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);

  // Coincident topology offset for lines
  float c_factor = coinOffset.lineFactor;
  float c_offset = coinOffset.lineOffset;
  float cscale = length(float2(dfdx(in.position.z), dfdy(in.position.z)));
  out.depth = in.position.z + c_factor * cscale + c_offset / 65000.0;
  return out;
}

// ---------------------------------------------------------------------------
// P3-3C: Miter Join line shaders — screen-space quad expansion with miter offsets
// Same 4-vertex triangle strip as thick lines, but the vertex shader looks at
// adjacent segments to compute miter join offsets at shared endpoints.
// Falls back to bevel when the miter angle is too acute.
// ---------------------------------------------------------------------------

struct MiterJoinLineVertexOut {
  float4 position [[position]];
  float3 viewPos;
  float3 viewNormal;
  float4 vertexColor;
  float dist_to_centerline;  // perpendicular distance from center (-0.5..0.5)
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
  // Quad corners: (-1,-1), (1,-1), (-1,1), (1,1)
  const float2 tri_verts[4] = {
    float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1)
  };

  float2 p_coord = tri_verts[vertex_id];

  // Read the two endpoint indices for this line segment
  uint p0_idx = lineIndices[instance_id * 2];
  uint p1_idx = lineIndices[instance_id * 2 + 1];

  float3 p0_MC = positions[p0_idx];
  float3 p1_MC = positions[p1_idx];

  // Transform both endpoints to clip space
  float4 p0_DC = scene.projectionMatrix * scene.viewMatrix * scene.modelMatrix * float4(p0_MC, 1.0);
  float4 p1_DC = scene.projectionMatrix * scene.viewMatrix * scene.modelMatrix * float4(p1_MC, 1.0);

  // Transform to screen space
  float2 resolution = scene.viewport.zw;
  float2 p0_screen = resolution * (0.5 * p0_DC.xy / p0_DC.w + 0.5);
  float2 p1_screen = resolution * (0.5 * p1_DC.xy / p1_DC.w + 0.5);

  // Compute line direction and perpendicular in screen space
  float2 x_basis = normalize(p1_screen - p0_screen);
  float segLen = length(p1_screen - p0_screen);
  x_basis = select(x_basis, float2(1.0, 0.0), segLen < 0.001);
  float2 y_basis = float2(-x_basis.y, x_basis.x);

  // Miter join: look at adjacent segments to compute miter offset at shared endpoints
  float w = max(lineWidth, 1.0);
  float2 offset = p_coord.x * x_basis + p_coord.y * y_basis * w;

  // Check for miter at p0 (start of segment, p_coord.x == -1)
  if (p_coord.x == -1.0 && instance_id > 0) {
    // Check if previous segment shares the same cell (i.e., same polyline)
    if (cellIds[instance_id - 1] == cellIds[instance_id]) {
      uint prev_p0_idx = lineIndices[(instance_id - 1) * 2];
      uint prev_p1_idx = lineIndices[(instance_id - 1) * 2 + 1];
      float3 prev_p0_MC = positions[prev_p0_idx];
      float3 prev_p1_MC = positions[prev_p1_idx];

      float4 prev_p0_DC = scene.projectionMatrix * scene.viewMatrix * scene.modelMatrix * float4(prev_p0_MC, 1.0);
      float4 prev_p1_DC = scene.projectionMatrix * scene.viewMatrix * scene.modelMatrix * float4(prev_p1_MC, 1.0);

      float2 prev_p0_screen = resolution * (0.5 * prev_p0_DC.xy / prev_p0_DC.w + 0.5);
      float2 prev_p1_screen = resolution * (0.5 * prev_p1_DC.xy / prev_p1_DC.w + 0.5);

      // Direction of previous segment in screen space
      float2 prev_dir = normalize(prev_p1_screen - prev_p0_screen);
      // Current segment direction
      float2 curr_dir = x_basis;

      // Miter direction = normalized sum of the two edge normals
      float2 prev_normal = float2(-prev_dir.y, prev_dir.x);
      float2 curr_normal = float2(-curr_dir.y, curr_dir.x);
      float2 miter = prev_normal + curr_normal;
      float miter_len = length(miter);

      // Miter limit: if miter is too long (angle too acute), fall back to bevel
      float MITER_LIMIT = 2.0;
      if (miter_len > 0.001 && miter_len < MITER_LIMIT * 2.0) {
        miter = miter / miter_len;
        // Offset along miter direction
        float miterOffset = w * 0.5 / dot(miter, curr_normal);
        if (sign(dot(p_coord.y * y_basis, miter)) == sign(dot(float2(0.0, 1.0), miter))) {
          offset = p_coord.x * x_basis + miter * miterOffset;
        }
      }
    }
  }

  // Check for miter at p1 (end of segment, p_coord.x == 1)
  if (p_coord.x == 1.0 && instance_id < segmentCount - 1) {
    // Check if next segment shares the same cell
    if (cellIds[instance_id + 1] == cellIds[instance_id]) {
      uint next_p0_idx = lineIndices[(instance_id + 1) * 2];
      uint next_p1_idx = lineIndices[(instance_id + 1) * 2 + 1];
      float3 next_p0_MC = positions[next_p0_idx];
      float3 next_p1_MC = positions[next_p1_idx];

      float4 next_p0_DC = scene.projectionMatrix * scene.viewMatrix * scene.modelMatrix * float4(next_p0_MC, 1.0);
      float4 next_p1_DC = scene.projectionMatrix * scene.viewMatrix * scene.modelMatrix * float4(next_p1_MC, 1.0);

      float2 next_p0_screen = resolution * (0.5 * next_p0_DC.xy / next_p0_DC.w + 0.5);
      float2 next_p1_screen = resolution * (0.5 * next_p1_DC.xy / next_p1_DC.w + 0.5);

      // Direction of next segment in screen space
      float2 next_dir = normalize(next_p1_screen - next_p0_screen);
      // Current segment direction
      float2 curr_dir = x_basis;

      // Miter direction = normalized sum of the two edge normals
      float2 curr_normal = float2(-curr_dir.y, curr_dir.x);
      float2 next_normal = float2(-next_dir.y, next_dir.x);
      float2 miter = curr_normal + next_normal;
      float miter_len = length(miter);

      // Miter limit: if miter is too long (angle too acute), fall back to bevel
      float MITER_LIMIT = 2.0;
      if (miter_len > 0.001 && miter_len < MITER_LIMIT * 2.0) {
        miter = miter / miter_len;
        float miterOffset = w * 0.5 / dot(miter, curr_normal);
        if (sign(dot(p_coord.y * y_basis, miter)) == sign(dot(float2(0.0, 1.0), miter))) {
          offset = p_coord.x * x_basis + miter * miterOffset;
        }
      }
    }
  }

  float2 p = p0_screen + offset + (p1_screen - p0_screen) * 0.5 * (p_coord.x + 1.0);

  // Select z/w from the appropriate endpoint
  float4 p_DC = mix(p0_DC, p1_DC, p_coord.x);

  MiterJoinLineVertexOut out;
  out.position = float4(p_DC.w * ((2.0 * p) / resolution - 1.0), p_DC.z, p_DC.w);

  // Interpolate view-space position for lighting
  float3 mid_MC = mix(p0_MC, p1_MC, p_coord.x);
  out.viewPos = (scene.viewMatrix * scene.modelMatrix * float4(mid_MC, 1.0)).xyz;
  out.viewNormal = scene.normalMatrix * float3(0.0, 0.0, 1.0);

  // Use the color from the appropriate endpoint
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

  // Tube-like shading: modify normal based on distance from centerline
  float3 N = normalize(in.viewNormal);
  float d = abs(in.dist_to_centerline);
  N.z = 1.0 - 2.0 * d;
  N = normalize(N);

  float3 totalAmbient = material.ambientColor.w * baseColor;
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

    totalDiffuse += df * baseColor * lightColor * attenuation;
    float NdotL = max(dot(N, toLight), 0.0);
    if (NdotL > 0.0) {
      float sf = pow(max(dot(viewDir, reflDir), 0.0), material.specularPower);
      totalSpecular += sf * material.specularColor.w * material.specularColor.rgb * lightColor * attenuation;
    }
  }

  out.color = float4(totalAmbient + material.diffuseColor.w * totalDiffuse + totalSpecular, baseAlpha);
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);

  // Coincident topology offset for lines
  float c_factor = coinOffset.lineFactor;
  float c_offset = coinOffset.lineOffset;
  float cscale = length(float2(dfdx(in.position.z), dfdy(in.position.z)));
  out.depth = in.position.z + c_factor * cscale + c_offset / 65000.0;
  return out;
}

// ---------------------------------------------------------------------------
// P2-8: Compute kernel for cell-to-primitive mapping
// Maps each primitive (triangle/line segment) to its owning cell ID.
// Mirrors WebGPU's VTKCellToGraphicsPrimitive.wgsl.
// ---------------------------------------------------------------------------
kernel void cellToPrimitive(
    device uint* cellIds [[buffer(0)]],
    constant uint* primitiveToCell [[buffer(1)]],
    constant uint& cellIdOffset [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  cellIds[gid] = primitiveToCell[gid] + cellIdOffset + 1u;  // 1-based
}
