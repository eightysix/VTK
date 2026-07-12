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

// ---------------------------------------------------------------------------
// P6-6A: GPU Tessellation Compute Kernels
// Replace CPU fan-triangulation with compute-based polygon → triangle conversion.
// Mirrors WebGPU's polygon_to_triangle, poly_line_to_line, polygon_edges_to_lines.
// ---------------------------------------------------------------------------

struct TessParams {
  uint numCells;
  uint cellIdOffset;
};

// Polygon → Triangle fan tessellation.
// Each thread processes one polygon cell and emits (npts-2) triangles via fan from vertex 0.
// Also produces edge array for hiding internal fan edges when edge visibility is on.
//
// Bindings:
//   0: outConnectivity (output) — tessellated triangle indices, 3 per triangle
//   1: edgeArray (output) — per-triangle edge visibility flag (-1, 0, 1, or 2)
//   2: cellIds (output) — per-triangle cell ID
//   3: connectivity (input) — flat array of point IDs for all polygon cells
//   4: offsets (input) — per-cell start offset into connectivity (length = numCells+1)
//   5: primitiveCounts (input) — prefix-sum of triangle counts (length = numCells+1)
//   6: params (input) — numCells and cellIdOffset
kernel void polygonToTriangle(
    device uint* outConnectivity [[buffer(0)]],
    device float* edgeArray [[buffer(1)]],
    device uint* cellIds [[buffer(2)]],
    constant uint* connectivity [[buffer(3)]],
    constant uint* offsets [[buffer(4)]],
    constant uint* primitiveCounts [[buffer(5)]],
    constant TessParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= params.numCells) return;

  uint numTriangles = primitiveCounts[gid + 1u] - primitiveCounts[gid];
  uint outputOffset = primitiveCounts[gid] * 3u;
  uint inputOffset = offsets[gid];

  for (uint i = 0u; i < numTriangles; i++)
  {
    uint p0 = connectivity[inputOffset];
    uint p1 = connectivity[inputOffset + i + 1u];
    uint p2 = connectivity[inputOffset + i + 2u];

    uint triangleId = primitiveCounts[gid] + i;

    // Edge array: hides internal fan edges of a polygon when edge visibility is on.
    // -1 = single triangle (all edges are boundary), 2 = first triangle,
    // 0 = last triangle, 1 = interior triangle.
    // Matches WebGPU VTKCellToGraphicsPrimitive.wgsl line 49.
    if (numTriangles == 1u)
      edgeArray[triangleId] = -1.0;
    else if (i == 0u)
      edgeArray[triangleId] = 2.0;
    else if (i == numTriangles - 1u)
      edgeArray[triangleId] = 0.0;
    else
      edgeArray[triangleId] = 1.0;

    cellIds[triangleId] = gid + params.cellIdOffset;

    outConnectivity[outputOffset] = p0;
    outConnectivity[outputOffset + 1u] = p1;
    outConnectivity[outputOffset + 2u] = p2;
    outputOffset += 3u;
  }
}

// Polyline → Line segment tessellation.
// Each thread processes one polyline cell and emits (npts-1) line segments.
//
// Bindings:
//   0: outConnectivity (output) — line segment index pairs, 2 per segment
//   1: cellIds (output) — per-segment cell ID
//   2: connectivity (input) — flat point IDs for all line cells
//   3: offsets (input) — per-cell start offset into connectivity
//   4: primitiveCounts (input) — prefix-sum of segment counts
//   5: params (input) — numCells and cellIdOffset
kernel void polyLineToLine(
    device uint* outConnectivity [[buffer(0)]],
    device uint* cellIds [[buffer(1)]],
    constant uint* connectivity [[buffer(2)]],
    constant uint* offsets [[buffer(3)]],
    constant uint* primitiveCounts [[buffer(4)]],
    constant TessParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= params.numCells) return;

  uint numLines = primitiveCounts[gid + 1u] - primitiveCounts[gid];
  uint outputOffset = primitiveCounts[gid] * 2u;
  uint inputOffset = offsets[gid];

  for (uint i = 0u; i < numLines; i++)
  {
    uint p0 = connectivity[inputOffset + i];
    uint p1 = connectivity[inputOffset + i + 1u];

    uint lineId = primitiveCounts[gid] + i;
    cellIds[lineId] = gid + params.cellIdOffset;

    outConnectivity[outputOffset] = p0;
    outConnectivity[outputOffset + 1u] = p1;
    outputOffset += 2u;
  }
}

// Polygon boundary edges → Line segments (wireframe / edge visibility).
// Each thread processes one polygon cell and emits npts line segments
// (including the closing edge from last vertex back to first).
//
// Bindings:
//   0: outConnectivity (output) — edge index pairs, 2 per edge
//   1: cellIds (output) — per-edge cell ID
//   2: connectivity (input) — flat point IDs for all polygon cells
//   3: offsets (input) — per-cell start offset into connectivity
//   4: primitiveCounts (input) — prefix-sum of edge counts
//   5: params (input) — numCells and cellIdOffset
kernel void polygonEdgesToLines(
    device uint* outConnectivity [[buffer(0)]],
    device uint* cellIds [[buffer(1)]],
    constant uint* connectivity [[buffer(2)]],
    constant uint* offsets [[buffer(3)]],
    constant uint* primitiveCounts [[buffer(4)]],
    constant TessParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= params.numCells) return;

  uint numEdges = primitiveCounts[gid + 1u] - primitiveCounts[gid];
  uint outputOffset = primitiveCounts[gid] * 2u;
  uint inputOffset = offsets[gid];

  for (uint i = 0u; i < numEdges; i++)
  {
    uint p0 = connectivity[inputOffset + i];
    uint p1 = connectivity[inputOffset + (i + 1u) % numEdges];

    uint edgeId = primitiveCounts[gid] + i;
    cellIds[edgeId] = gid + params.cellIdOffset;

    outConnectivity[outputOffset] = p0;
    outConnectivity[outputOffset + 1u] = p1;
    outputOffset += 2u;
  }
}

// ============================================================================
// 2D Mapper shaders (P7-7A)
// ============================================================================

// 2D mapper uniforms — orthographic projection + per-draw state
struct Mapper2DState {
  float4x4 wcvcMatrix;       // world-to-viewport-clip matrix (orthographic)
  float4 color;              // base color (RGBA)
  float pointSize;           // point size in pixels
  float lineWidth;           // line width in pixels
  uint flags;                // bit 0: use point color, bit 1: use cell color
};

// 2D vertex input — position (float2 or float3) + optional color
struct Vertex2DIn {
  float2 position [[attribute(0)]];
};

// 2D vertex output
struct Vertex2DOut {
  float4 position [[position]];
  float4 color;
};

// 2D vertex shader — transforms 2D positions to clip space using WCVC matrix
vertex Vertex2DOut vertex_2d_main(
    Vertex2DIn in [[stage_in]],
    constant Mapper2DState& state [[buffer(1)]])
{
  Vertex2DOut out;
  out.position = state.wcvcMatrix * float4(in.position, 0.0, 1.0);
  out.color = state.color;
  return out;
}

// 2D fragment shader — outputs flat color
fragment float4 fragment_2d_main(
    Vertex2DOut in [[stage_in]])
{
  return in.color;
}

// ============================================================================
// 8B: Depth Peeling / Correct Translucency (OIT)
// ============================================================================

// Peeling uniforms — passed per render pass
struct PeelUniforms {
  uint mode;          // 0=init, 1=peel, 2=alphaBlend
  uint peelPass;      // current peel iteration
  float2 viewportSize;
};

// Fullscreen vertex output (for composite/init passes)
struct FullscreenVertexOut {
  float4 position [[position]];
  float2 texCoord;
};

// Fullscreen vertex shader — generates a single triangle covering the viewport
vertex FullscreenVertexOut vertex_fullscreen_main(uint vertex_id [[vertex_id]]) {
  // Oversized triangle: covers entire viewport with 1 triangle (2 fewer verts than a quad)
  float2 positions[3] = {
    float2(-1, -1),
    float2( 3, -1),
    float2(-1,  3)
  };
  float2 texCoords[3] = {
    float2(0, 1),
    float2(2, 1),
    float2(0, -1)
  };
  FullscreenVertexOut out;
  out.position = float4(positions[vertex_id], 0, 1);
  out.texCoord = texCoords[vertex_id];
  return out;
}

// ---------------------------------------------------------------------------
// Depth peeling — fragment output structs
// ---------------------------------------------------------------------------

// Init pass output: just min/max depth (RG32Float)
struct PeelInitOutput {
  float2 depthRange [[color(0)]];
};

// Peel pass output: back temp, front accumulation, depth range
struct PeelPassOutput {
  float4 backTemp  [[color(0)]];   // premultiplied back fragment
  float4 frontDest [[color(1)]];   // front accumulation (alpha stored as 1-alpha)
  float2 depthDest [[color(2)]];   // min/max depth for next iteration
};

// ---------------------------------------------------------------------------
// Init pass fragment shader — establishes initial min/max depth range
// Renders translucent geometry with MAX blending to find visible depth bounds.
// Depth test=Less ensures fragments behind opaque are discarded by hardware.
// MAX blending on RG32Float target picks the nearest min and farthest max.
// Depth is negated for min so that MAX(blue) picks the smallest depth.
// ---------------------------------------------------------------------------
fragment PeelInitOutput fragment_peel_init(
    VertexOut in [[stage_in]],
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

  PeelInitOutput out;
  float depth = in.position.z;
  // Negate min depth so that MAX blending picks the nearest (smallest) depth
  out.depthRange = float2(-depth, depth);
  return out;
}

// ---------------------------------------------------------------------------
// Peel pass fragment shader — main depth peeling logic
// Reads previous depth range and front accumulation, outputs to three targets.
// Uses the four-zone algorithm from vtkDualDepthPeelingPass:
//   Zone 1 (outside):   fragment is irrelevant, pass through previous data
//   Zone 2 (inside):    fragment will be peeled later, mark its depth
//   Zone 3 (on front):  nearest unpeeled fragment, under-blend into front
//   Zone 4 (on back):   farthest unpeeled fragment, premultiply and output
// ---------------------------------------------------------------------------
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
  // P1-6: discard fragments outside clip planes
  if (clipPlanes.numClipPlanes > 0 && in.clipDistances.x < 0.0) discard_fragment();
  if (clipPlanes.numClipPlanes > 1 && in.clipDistances.y < 0.0) discard_fragment();
  if (clipPlanes.numClipPlanes > 2 && in.clipDistances.z < 0.0) discard_fragment();
  if (clipPlanes.numClipPlanes > 3 && in.clipDistances.w < 0.0) discard_fragment();

  // Compute pixel coordinates for texture reads
  uint2 pixel = uint2(in.position.xy) - uint2(scene.viewport.xy);

  // Read previous peel state
  float4 prevFront = prevFrontTex.read(pixel);
  float2 prevDepth = prevDepthTex.read(pixel).rg;
  float minDepth = -prevDepth.x;  // negate back to positive
  float maxDepth = prevDepth.y;
  float fragDepth = in.position.z;
  float epsilon = 0.0000001;

  // Default outputs: pass through previous state, no depth change
  PeelPassOutput out;
  out.backTemp = float4(0.0);
  out.frontDest = prevFront;
  out.depthDest = float2(-1.0, -1.0);

  // Compute fragment color via Phong lighting (same as fragment_main)
  float3 N = normalize(in.viewNormal);
  bool hasVertexColors = (scene.flags & (1u << 8)) != 0u;
  float3 ambientColor = hasVertexColors ? in.vertexColor.rgb : material.ambientColor.rgb;
  float ambientIntensity = material.ambientColor.w;
  float3 diffuseColor = hasVertexColors ? in.vertexColor.rgb : material.diffuseColor.rgb;
  float diffuseIntensity = material.diffuseColor.w;
  float3 specularColor = material.specularColor.rgb;
  float specularIntensity = material.specularColor.w;
  float baseOpacity = hasVertexColors ? in.vertexColor.a : material.opacity;

  // P5-5A: texture sampling
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

  float3 fragRGB = totalAmbient + diffuseIntensity * totalDiffuse + totalSpecular;
  float fragAlpha = baseOpacity;

  // Four-zone depth comparison
  // Zone 1: Outside current peels — fragment is irrelevant
  if (fragDepth < minDepth - epsilon || fragDepth > maxDepth + epsilon) {
    return out;
  }

  // Zone 2: Strictly inside current peels — will be peeled in a future pass
  if (fragDepth > minDepth + epsilon && fragDepth < maxDepth - epsilon) {
    out.depthDest = float2(-fragDepth, fragDepth);
    return out;
  }

  // Zone 3: On the front peel (nearest unpeeled)
  if (fragDepth >= minDepth - epsilon && fragDepth <= minDepth + epsilon) {
    float prevAlpha = 1.0 - prevFront.a;  // stored as (1-alpha), convert back
    // Under-blend: accumulate front-to-back
    out.frontDest.rgb = prevAlpha * fragAlpha * fragRGB + prevFront.rgb;
    out.frontDest.a = 1.0 - (prevAlpha * (1.0 - fragAlpha));  // store as (1-newAlpha)
    return out;
  }

  // Zone 4: On the back peel (farthest unpeeled)
  if (fragDepth >= maxDepth - epsilon && fragDepth <= maxDepth + epsilon) {
    out.backTemp = float4(fragRGB * fragAlpha, fragAlpha);  // premultiplied alpha
    return out;
  }

  return out;
}

// ---------------------------------------------------------------------------
// Alpha blend pass fragment shader — blends remaining unpeeled fragments
// Used when the peel loop terminates early (occlusion threshold exceeded).
// Discards fragments outside the last peel depth range.
// Outputs premultiplied fragment color for over-blending into back accumulation.
// ---------------------------------------------------------------------------
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
  // P1-6: discard fragments outside clip planes
  if (clipPlanes.numClipPlanes > 0 && in.clipDistances.x < 0.0) discard_fragment();
  if (clipPlanes.numClipPlanes > 1 && in.clipDistances.y < 0.0) discard_fragment();
  if (clipPlanes.numClipPlanes > 2 && in.clipDistances.z < 0.0) discard_fragment();
  if (clipPlanes.numClipPlanes > 3 && in.clipDistances.w < 0.0) discard_fragment();

  uint2 pixel = uint2(in.position.xy) - uint2(scene.viewport.xy);
  float2 prevDepth = prevDepthTex.read(pixel).rg;
  float minDepth = -prevDepth.x;
  float maxDepth = prevDepth.y;
  float fragDepth = in.position.z;
  float epsilon = 0.0000001;

  // Discard fragments outside the last peel range
  if (fragDepth < minDepth - epsilon || fragDepth > maxDepth + epsilon) {
    discard_fragment();
  }

  // Compute fragment color (same Phong lighting)
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

  float3 fragRGB = totalAmbient + diffuseIntensity * totalDiffuse + totalSpecular;
  float fragAlpha = baseOpacity;
  return float4(fragRGB * fragAlpha, fragAlpha);  // premultiplied alpha
}

// ---------------------------------------------------------------------------
// Composite pass fragment shader — composites front and back accumulation
// onto the framebuffer. Front accumulation has alpha stored as (1-alpha).
// Uses premultiplied-alpha over-blending: src=One, dst=OneMinusSrcAlpha.
// ---------------------------------------------------------------------------
fragment float4 fragment_peel_composite(
    FullscreenVertexOut in [[stage_in]],
    texture2d<float, access::read> frontTex [[texture(0)]],
    texture2d<float, access::read> backTex [[texture(1)]]) {
  uint2 pixel = uint2(in.position.xy);
  float4 front = frontTex.read(pixel);
  float4 back = backTex.read(pixel);

  float frontAlpha = 1.0 - front.a;  // stored as (1-alpha), convert back
  // Under-blend: back underneath front
  float3 color = front.rgb + back.rgb * frontAlpha;
  // Convert under-blend alpha to over-blend alpha for final compositing
  float alpha = 1.0 - frontAlpha * (1.0 - back.a);

  return float4(color, alpha);
}

// ---------------------------------------------------------------------------
// Back blend pass fragment shader — blends BackTemp into Back accumulation
// Fullscreen quad that reads BackTemp and over-blends into the back buffer.
// Discards fragments with zero alpha (no back fragment this peel).
// Uses premultiplied-alpha over-blending.
// ---------------------------------------------------------------------------
fragment float4 fragment_peel_back_blend(
    FullscreenVertexOut in [[stage_in]],
    texture2d<float, access::read> backTempTex [[texture(0)]]) {
  uint2 pixel = uint2(in.position.xy);
  float4 backTemp = backTempTex.read(pixel);

  // Discard if no back fragment was written
  if (backTemp.a < 0.001) discard_fragment();

  // BackTemp is already premultiplied, pass through for over-blending
  return backTemp;
}

// ---------------------------------------------------------------------------
// P7-7D: Glyph3D instanced rendering
//
// Vertex function that renders source geometry (triangles) with per-instance
// glyph transforms, colors, and pick IDs. Each instance is a glyph placed at
// an input point with optional scaling, rotation, and color.
//
// Source geometry (per-vertex):
//   buffer(0) = positions (float3)
//   buffer(1) = normals (float3)
//
// Instance data (per-instance, stepFunctionPerInstance):
//   buffer(2)  = glyph transforms (float4x4 per instance, column-major)
//   buffer(3)  = glyph normal transforms (float3x3 per instance, column-major)
//   buffer(4)  = glyph colors (float4 per instance, RGBA)
//   buffer(5)  = glyph pick IDs (uint per instance)
//
// Uniforms:
//   buffer(8)  = SceneUniforms (vertex + fragment)
//   buffer(9)  = ClipPlaneUniforms (vertex + fragment)
//
// Fragment uniforms:
//   buffer(0)  = MaterialUniforms
//   buffer(1)  = LightUniforms
//   buffer(2)  = SceneUniforms (shared with vertex)
//   buffer(3)  = CoincidentOffsetUniforms
//   buffer(9)  = ClipPlaneUniforms (shared with vertex)
// ---------------------------------------------------------------------------

struct GlyphVertexOut {
  float4 position [[position]];
  float3 viewPos;
  float3 viewNormal;
  float4 glyphColor;
  float4 clipDistances;
  uint cellId;
  uint propId;
};

vertex GlyphVertexOut vertex_glyph_main(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    constant float3* positions [[buffer(0)]],
    constant float3* normals [[buffer(1)]],
    constant float4x4* glyphTransforms [[buffer(2)]],
    constant float3x3* glyphNormalTransforms [[buffer(3)]],
    constant float4* glyphColors [[buffer(4)]],
    constant uint* glyphPickIds [[buffer(5)]],
    constant SceneUniforms& scene [[buffer(8)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(9)]],
    constant uint& propId [[buffer(10)]]) {
  GlyphVertexOut out;

  float3 pos = positions[vertex_id];
  float3 norm = normals[vertex_id];

  float4x4 glyphTransform = glyphTransforms[instance_id];
  float3x3 normalTransform = glyphNormalTransforms[instance_id];

  float4 worldPos = scene.modelMatrix * glyphTransform * float4(pos, 1.0);
  float4 viewPos = scene.viewMatrix * worldPos;
  out.viewPos = viewPos.xyz;
  out.position = scene.projectionMatrix * viewPos;
  out.viewNormal = scene.normalMatrix * normalTransform * norm;
  out.glyphColor = glyphColors[instance_id];
  out.cellId = glyphPickIds[instance_id] + 1u;
  out.propId = propId + 1u;

  out.clipDistances = float4(
    dot(float4(pos, 1.0), clipPlanes.planes[0]),
    dot(float4(pos, 1.0), clipPlanes.planes[1]),
    dot(float4(pos, 1.0), clipPlanes.planes[2]),
    dot(float4(pos, 1.0), clipPlanes.planes[3]));

  return out;
}

fragment FragmentOutput fragment_glyph_main(
    GlyphVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(9)]]) {
  if (clipPlanes.numClipPlanes > 0 && in.clipDistances.x < 0.0) discard_fragment();
  if (clipPlanes.numClipPlanes > 1 && in.clipDistances.y < 0.0) discard_fragment();
  if (clipPlanes.numClipPlanes > 2 && in.clipDistances.z < 0.0) discard_fragment();
  if (clipPlanes.numClipPlanes > 3 && in.clipDistances.w < 0.0) discard_fragment();

  float3 N = normalize(in.viewNormal);
  float3 ambientColor = in.glyphColor.rgb;
  float ambientIntensity = material.ambientColor.w;
  float3 diffuseColor = in.glyphColor.rgb;
  float diffuseIntensity = material.diffuseColor.w;
  float3 specularColor = material.specularColor.rgb;
  float specularIntensity = material.specularColor.w;
  float baseOpacity = in.glyphColor.a;

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
  out.color = float4(totalAmbient + diffuseIntensity * totalDiffuse + totalSpecular,
                     baseOpacity * material.opacity);
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);
  out.depth = in.position.z;
  return out;
}

// ---------------------------------------------------------------------------
// P7-7D: Glyph3D line source geometry vertex shader
//
// Same as glyph triangle shader but draws as line segments (MTLPrimitiveTypeLine).
// Buffer layout identical to vertex_glyph_main.
// ---------------------------------------------------------------------------

struct GlyphLineVertexOut {
  float4 position [[position]];
  float3 viewPos;
  float3 viewNormal;
  float4 glyphColor;
  float4 clipDistances;
  uint cellId;
  uint propId;
};

vertex GlyphLineVertexOut vertex_glyph_line_main(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    constant float3* positions [[buffer(0)]],
    constant float3* normals [[buffer(1)]],
    constant float4x4* glyphTransforms [[buffer(2)]],
    constant float3x3* glyphNormalTransforms [[buffer(3)]],
    constant float4* glyphColors [[buffer(4)]],
    constant uint* glyphPickIds [[buffer(5)]],
    constant SceneUniforms& scene [[buffer(8)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(9)]],
    constant uint& propId [[buffer(10)]]) {
  GlyphLineVertexOut out;

  float3 pos = positions[vertex_id];
  float3 norm = normals[vertex_id];

  float4x4 glyphTransform = glyphTransforms[instance_id];
  float3x3 normalTransform = glyphNormalTransforms[instance_id];

  float4 worldPos = scene.modelMatrix * glyphTransform * float4(pos, 1.0);
  float4 viewPos = scene.viewMatrix * worldPos;
  out.viewPos = viewPos.xyz;
  out.position = scene.projectionMatrix * viewPos;
  out.viewNormal = scene.normalMatrix * normalTransform * norm;
  out.glyphColor = glyphColors[instance_id];
  out.cellId = glyphPickIds[instance_id] + 1u;
  out.propId = propId + 1u;

  out.clipDistances = float4(
    dot(float4(pos, 1.0), clipPlanes.planes[0]),
    dot(float4(pos, 1.0), clipPlanes.planes[1]),
    dot(float4(pos, 1.0), clipPlanes.planes[2]),
    dot(float4(pos, 1.0), clipPlanes.planes[3]));

  return out;
}

fragment FragmentOutput fragment_glyph_line_main(
    GlyphLineVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(9)]]) {
  if (clipPlanes.numClipPlanes > 0 && in.clipDistances.x < 0.0) discard_fragment();
  if (clipPlanes.numClipPlanes > 1 && in.clipDistances.y < 0.0) discard_fragment();
  if (clipPlanes.numClipPlanes > 2 && in.clipDistances.z < 0.0) discard_fragment();
  if (clipPlanes.numClipPlanes > 3 && in.clipDistances.w < 0.0) discard_fragment();

  float3 N = normalize(in.viewNormal);
  float3 ambientColor = in.glyphColor.rgb;
  float ambientIntensity = material.ambientColor.w;
  float3 diffuseColor = in.glyphColor.rgb;
  float diffuseIntensity = material.diffuseColor.w;
  float3 specularColor = material.specularColor.rgb;
  float specularIntensity = material.specularColor.w;
  float baseOpacity = in.glyphColor.a;

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
  out.color = float4(totalAmbient + diffuseIntensity * totalDiffuse + totalSpecular,
                     baseOpacity * material.opacity);
  out.ids = uint4(in.cellId, in.propId, 1u, 0u);
  out.depth = in.position.z;
  return out;
}

// ---------------------------------------------------------------------------
// P7-7D: Glyph3D point source geometry vertex shader
//
// Renders glyph source points. Same instance data as triangle/line glyphs.
// Uses MTLPrimitiveTypePoint with point size from SceneUniforms.
// ---------------------------------------------------------------------------

struct GlyphPointVertexOut {
  float4 position [[position]];
  float3 viewPos;
  float3 viewNormal;
  float4 glyphColor;
  float4 clipDistances;
  uint cellId;
  uint propId;
  float point_size;
};

vertex GlyphPointVertexOut vertex_glyph_point_main(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    constant float3* positions [[buffer(0)]],
    constant float3* normals [[buffer(1)]],
    constant float4x4* glyphTransforms [[buffer(2)]],
    constant float3x3* glyphNormalTransforms [[buffer(3)]],
    constant float4* glyphColors [[buffer(4)]],
    constant uint* glyphPickIds [[buffer(5)]],
    constant SceneUniforms& scene [[buffer(8)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(9)]],
    constant uint& propId [[buffer(10)]]) {
  GlyphPointVertexOut out;

  float3 pos = positions[vertex_id];
  float3 norm = normals[vertex_id];

  float4x4 glyphTransform = glyphTransforms[instance_id];
  float3x3 normalTransform = glyphNormalTransforms[instance_id];

  float4 worldPos = scene.modelMatrix * glyphTransform * float4(pos, 1.0);
  float4 viewPos = scene.viewMatrix * worldPos;
  out.viewPos = viewPos.xyz;
  out.position = scene.projectionMatrix * viewPos;
  out.viewNormal = scene.normalMatrix * normalTransform * norm;
  out.glyphColor = glyphColors[instance_id];
  out.cellId = glyphPickIds[instance_id] + 1u;
  out.propId = propId + 1u;
  out.point_size = scene.pointSize;

  out.clipDistances = float4(
    dot(float4(pos, 1.0), clipPlanes.planes[0]),
    dot(float4(pos, 1.0), clipPlanes.planes[1]),
    dot(float4(pos, 1.0), clipPlanes.planes[2]),
    dot(float4(pos, 1.0), clipPlanes.planes[3]));

  return out;
}

fragment FragmentOutput fragment_glyph_point_main(
    GlyphPointVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(9)]]) {
  if (clipPlanes.numClipPlanes > 0 && in.clipDistances.x < 0.0) discard_fragment();
  if (clipPlanes.numClipPlanes > 1 && in.clipDistances.y < 0.0) discard_fragment();
  if (clipPlanes.numClipPlanes > 2 && in.clipDistances.z < 0.0) discard_fragment();
  if (clipPlanes.numClipPlanes > 3 && in.clipDistances.w < 0.0) discard_fragment();

  float3 N = normalize(in.viewNormal);
  float3 ambientColor = in.glyphColor.rgb;
  float ambientIntensity = material.ambientColor.w;
  float3 diffuseColor = in.glyphColor.rgb;
  float diffuseIntensity = material.diffuseColor.w;
  float3 specularColor = material.specularColor.rgb;
  float specularIntensity = material.specularColor.w;
  float baseOpacity = in.glyphColor.a;

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
  out.color = float4(totalAmbient + diffuseIntensity * totalDiffuse + totalSpecular,
                     baseOpacity * material.opacity);
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
  float4 coarseMapInvDims; // 1/coarseDims.x, y, z; unused .w
};

struct VolumeVertexOut {
  float4 position [[position]];
  float3 localPos;
};

// Volume vertex shader — transforms bounding box vertices and computes
// local-space position for ray entry in the fragment shader.
struct VolumeVertexIn {
  float3 position [[attribute(0)]];
};

vertex VolumeVertexOut vertex_volume_main(
    VolumeVertexIn in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]]) {
  VolumeVertexOut out;

  float3 modelPos = in.position;
  float4 worldPos = volumeUniforms.volumeToWorld * float4(modelPos, 1.0);
  out.position = volumeUniforms.viewProjection * worldPos;

  float3 boundsSize = volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz;
  out.localPos = (modelPos - volumeUniforms.volumeBoundsMin.xyz) / boundsSize;

  return out;
}

struct VolumeFragmentOut {
  float4 color [[color(0)]];
};

constant int MAX_RAY_STEPS = 2000;

// Pseudo-random for jittering
inline float volume_random(float2 st) {
  return fract(sin(dot(st.xy, float2(12.9898, 78.233))) * 43758.5453123);
}

// Ray-box intersection
inline float2 intersectBox(float3 orig, float3 dir, float3 boxMin, float3 boxMax) {
  float3 invDir = 1.0 / (dir + float3(1e-8));
  float3 tbot = invDir * (boxMin - orig);
  float3 ttop = invDir * (boxMax - orig);
  float3 tmin = min(ttop, tbot);
  float3 tmax = max(ttop, tbot);
  float t0 = max(max(tmin.x, tmin.y), tmin.z);
  float t1 = min(min(tmax.x, tmax.y), tmax.z);
  return float2(t0, t1);
}

fragment VolumeFragmentOut fragment_volume_main(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> transferFunctionTexture [[texture(1)]],
    texture3d<float> coarseOpacityTexture [[texture(2)]],
    sampler transferFunctionSampler [[sampler(0)]],
    sampler volumeSampler [[sampler(1)]],
    sampler coarseSampler [[sampler(2)]]) {
  VolumeFragmentOut output;

  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float stepSize = volumeUniforms.sampleDistance;

  float3 startPoint = in.localPos;
  float3 rayDir = startPoint - cameraPos;
  float dirLength = length(rayDir);

  if (dirLength < 0.0001) {
    discard_fragment();
  }
  rayDir = rayDir / dirLength;

  float2 t = intersectBox(cameraPos, rayDir, float3(0.0), float3(1.0));

  float tStart = max(t.x, 0.0);
  if (tStart >= t.y) {
    discard_fragment();
  }

  float3 entryPoint = cameraPos + rayDir * tStart;
  float3 exitPoint = cameraPos + rayDir * t.y;
  float totalDist = length(exitPoint - entryPoint);
  int maxSteps = min(max(1, int(ceil(totalDist / stepSize))), MAX_RAY_STEPS);

  float jitter = 0.0;
  if (volumeUniforms.useJittering > 0.5) {
    jitter = volume_random(in.position.xy) * stepSize;
  }

  float3 currentPoint = entryPoint + (rayDir * jitter);
  float3 stepVec = rayDir * stepSize;

  // Use half precision for accumulators — Apple GPUs run half ALU at 2x throughput.
  // Texture coordinates stay float for addressing accuracy.
  half3 accumulatedColor = half3(0.0);
  half accumulatedOpacity = 0.0;
  half scalarRangeRcp = half(1.0 / (volumeUniforms.scalarMax - volumeUniforms.scalarMin));

  // Empty-space skipping via coarse 3D opacity map (32x32x32).
  // When a coarse cell has max opacity ≈ 0, the ray jumps to the next cell
  // boundary, avoiding all expensive volume texture fetches in that region.
  float3 coarseCellSize = volumeUniforms.coarseMapInvDims.xyz; // e.g. 1/32
  bool hasCoarseMap = volumeUniforms.coarseMapInvDims.x > 0.0;

  int i = 0;
  int maxIter = maxSteps * 3; // safety: allow extra iterations for skips
  int iter = 0;
  while (i < maxSteps && iter < maxIter) {
    iter++;

    // Bounds check — ensure we're still inside [0,1]^3
    if (any(currentPoint < 0.0) || any(currentPoint > 1.0)) {
      break;
    }

    // --- Coarse map empty-space skip ---
    if (hasCoarseMap) {
      float coarseMaxOp = coarseOpacityTexture.sample(
        coarseSampler, currentPoint, level(0)).r;
      if (coarseMaxOp < 0.002) {
        // Compute which coarse cell we're in
        float3 cellf = currentPoint / coarseCellSize;
        int3 cell = clamp(int3(cellf), 0, 31);
        float3 cellMin = float3(cell) * coarseCellSize;
        float3 cellMax = cellMin + coarseCellSize;

        // Distance to exit this coarse cell along the ray
        float3 invDir = 1.0 / (rayDir + float3(1e-8));
        float3 t1 = invDir * (cellMin - currentPoint);
        float3 t2 = invDir * (cellMax - currentPoint);
        float3 tmax = max(t1, t2);
        float tSkip = min(min(tmax.x, tmax.y), tmax.z);
        tSkip = max(tSkip, stepSize);

        currentPoint += rayDir * tSkip;
        continue; // don't increment i — skips are free
      }
    }

    // --- Normal sample ---
    float rawScalar = volumeTexture.sample(volumeSampler, currentPoint, level(0)).r;
    half scalarNorm = clamp(
      half(rawScalar - volumeUniforms.scalarMin) * scalarRangeRcp,
      0.0h, 1.0h);
    half4 colorOpacity = half4(transferFunctionTexture.sample(
      transferFunctionSampler, float2(float(scalarNorm), 0.5), level(0)));
    half sampleOpacity = colorOpacity.a;
    if (sampleOpacity > 0.001h) {
      half3 sampleColor = colorOpacity.rgb;
      half w = 1.0h - accumulatedOpacity;
      accumulatedColor += w * sampleColor * sampleOpacity;
      accumulatedOpacity += w * sampleOpacity;
    }

    if (accumulatedOpacity >= 0.95h) {
      accumulatedOpacity = 1.0h;
      break;
    }

    currentPoint += stepVec;
    i++;
  }

  output.color = float4(float3(accumulatedColor), float(accumulatedOpacity));
  return output;
}
