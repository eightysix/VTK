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
// sVolumeClampZero — linear min/mag, clamp-to-zero; experiment variant (fc_marchVariant
//   == 2) samples the volume with this and relies on the in-shader clamp so the
//   edge behavior is unchanged. See PERFORMANCE_INVESTIGATION.md section 4.2 step 4.
constexpr sampler sVolume(filter::linear, address::clamp_to_edge);
constexpr sampler sNearest(filter::nearest, address::clamp_to_edge);
constexpr sampler sVolumeClampZero(filter::linear, address::clamp_to_zero);

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

// Scene flag bits (must match VTK_METAL_SCENE_FLAG_* in vtkMetalPolyDataMapper.mm).
constant uint kSceneFlagParallelProjection  = 1u << 0;
constant uint kSceneFlagVertexVisibility    = 1u << 3;
[[maybe_unused]] constant uint kSceneFlagSpherePoints        = 1u << 5;
[[maybe_unused]] constant uint kSceneFlagPointShape          = 1u << 7;
constant uint kSceneFlagHasSurfaceColors    = 1u << 8;
constant uint kSceneFlagHasActorTexture     = 1u << 9;
[[maybe_unused]] constant uint kSceneFlagHasSurfaceAlpha     = 1u << 10;
constant uint kSceneFlagHasCellTexture      = 1u << 11;
constant uint kSceneFlagUsePrimitiveCellIds = 1u << 12;
constant uint kSceneFlagHasScalarLUT        = 1u << 13;
constant uint kSceneFlagLinesUnlit           = 1u << 14;
constant uint kSceneFlagGlyphHasNormals      = 1u << 15;
// The actor's input actually has point normals. When clear the vertex-normal
// buffer holds the mapper's dummy (0,1,0) fill, so lit line
// fragments synthesize a camera-aligned derivative normal like GL.
constant uint kSceneFlagLinesHaveNormals     = 1u << 20;
// Property lighting disabled (vtkProperty::SetLighting(false)): every draw of
// the actor skips the Phong loop and emits the flat ambient+diffuse material
// color, matching vtkGLSLModLight's complexity-0 path in GL.
constant uint kSceneFlagLightingDisabled     = 1u << 16;
// Per-point scalar colors were baked into the point color buffer (GL's
// scalarColor VBO for the point draw). When clear, point fragments light the
// material ambient/diffuse colors like GL's no-scalar-color material path.
constant uint kSceneFlagHasPointColors       = 1u << 17;
// The actor has a backface property: the glyph fragment swaps the backface
// material in (and colors backfaces from it) instead of using the mirrored
// front-material slots, matching vtkOpenGLGlyph3DHelper::ReplaceShaderColor.
constant uint kSceneFlagGlyphHasBackface     = 1u << 18;
// RenderLinesAsTubes: wide lines build a fake cylinder normal across the width
// (lit tube look); when clear, wide lines are flat across the width like GL's
// native glLineWidth rendering.
constant uint kSceneFlagLinesTubeShading     = 1u << 19;

// Compile-time feature specialization for the surface shader (the "GL way"):
// one shader source, specialized per feature set at pipeline creation via
// MTLFunctionConstantValues. Every pipeline that uses vertex_main/fragment_main
// must specify a value for each constant (indices 6-11 avoid colliding with the
// volume shaders' function constants 0-5); the mapper sets them all-true for the
// full-behavior pipelines (peel/OIT/edge/line/base) and per-feature-mask for
// the surface pipelines, so a plain opaque surface compiles to a lean program
// with no vertexColor/uv/edge/ID work — matching what GL's shader-template
// substitution produces.
constant bool kHasSurfaceColors [[function_constant(6)]];
constant bool kHasActorTexture  [[function_constant(7)]];
constant bool kHasSurfaceAlpha  [[function_constant(8)]];
constant bool kHasBackface      [[function_constant(9)]];
constant bool kHasEdgeFlags     [[function_constant(10)]];
constant bool kEmitIds          [[function_constant(11)]];
constant bool kHasCellTexture   [[function_constant(12)]];
// Number of enabled lights, baked per pipeline like GL's shader-template
// unrolling: the surface pipelines compile with the exact count so the lighting
// loop unrolls to exactly that many iterations with no runtime guards; the
// shared full-behavior pipelines (base/peel/OIT/glyph/point/line) are
// specialized with the maximum and keep the runtime guard. Metal function
// constants cannot carry an initializer, so every pipeline whose fragment
// function reaches computePhongLighting must supply index 13.
constant int kLightCount [[function_constant(13)]];
// Light type of the first light (the only one when kLightCount == 1), baked the
// same way: 0 = headlight, 1 = directional/camera, 2 = point, 3 = spot. With
// the single-light surface pipelines the compiler folds this into the type
// dispatch, pruning the dead paths — matching GL's per-complexity shaders.
constant int kLightType [[function_constant(14)]];
// Scalar-texture coloring (InterpolateScalarsBeforeMapping): when set, the
// fragment stage interpolates a per-vertex scalar texture coordinate
// (out.scalarCoord) and looks the color up in a LUT texture
// (kScalarLUTTex, texture(9)) — GL's texture(colortexture, colorTCoord) path —
// instead of using per-vertex LUT colors. Must be supplied by every pipeline
// whose vertex/fragment functions reach the surface entries (indices 6-14 are
// the existing constants; 15 is this one).
constant bool kHasScalarLUT [[function_constant(15)]];
// Property lighting disabled (vtkProperty::SetLighting(false)): the compile-time
// analog of vtkGLSLModLight's complexity-0 shader. The mapper bakes it into the
// per-actor specialized pipelines (surface + thick-line tube variants) via
// MTLFunctionConstantValues, so the unlit variant drops the Phong loop (and the
// fake-tube normal construction) entirely — the same "no lighting code compiled
// in" outcome GL gets from selecting a NoLighting shader. Every fragment
// function that reaches shadeLineFragment/evaluateSurfaceFragment must supply
// this constant; the shared full-behavior pipelines (base/1px-line) bake NO and
// fall back to the runtime kSceneFlagLightingDisabled scene flag, since a single
// shared pipeline serves actors of mixed lighting state.
constant bool kLightingDisabled [[function_constant(16)]];

// The per-cell color port ("cell texture"): when per-cell colors are present
// and the vertex stream is deduplicated, the cell RGBA cannot live in the
// vertices (a shared corner belongs to multiple cells), so it is resolved
// per-primitive in the fragment stage via [[primitive_id]] — the direct analog
// of GL's gl_PrimitiveID + textureC. The cell colors live in a 2D RGBA8Unorm
// texture (kCellColorTex, texture(8)) laid out row-major with width
// kCellTextureWidth so the div/mod in the shader compile to a shift and mask;
// this mirrors GL's RGBA8 buffer texture, which fetches through the texture
// unit rather than a raw device-buffer load. cellPrimitiveIds is the matching
// 1-based cell id per triangle (0 = background) so picking reports the exact
// owning cell instead of the provoking vertex's first-wins value.
// kSceneFlagHasCellTexture (set only when a cell-color texture was built)
// switches the resolved color/ID source at runtime so the all-true
// full-feature pipelines stay correct for plain per-vertex actors.
// Texture layout constant, must match kCellTextureWidth in
// vtkMetalPolyDataMapper.mm. 2^13 width gives up to 8192*16384 = 134M
// triangles, the same cap as a desktop GL buffer texture.
constant uint kCellTextureWidth = 8192u;

// Coincident topology offset (P1-5)
struct CoincidentOffsetUniforms {
  float polygonFactor;
  float polygonOffset;
  float lineFactor;
  float lineOffset;
  float pointOffset;
};

// Single-pass surface edges (per-frame uniform, fragment buffer(4))
struct EdgeUniforms {
  float4 edgeColor;     // rgb + edgeOpacity
  float  edgeWidth;     // pixels = UseLineWidthForEdgeThickness ? LineWidth : EdgeWidth
  uint   flags;         // bit0 = edgeVisibility, bit1 = renderLinesAsTubes
  float  _pad;
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
  // ShowTexturesOnBackface: when 0 the actor texture is not applied to back
  // faces (matches vtkOpenGLPolyDataMapper's showTexturesOnBackface uniform).
  float showTexturesOnBackface;
  float _padding;
  // Backface material (mirrors the front layout). Used by fragment_main and the
  // peel shaders when the actor has a backface property; identical to the front
  // material otherwise.
  float4 backfaceAmbientColor;
  float4 backfaceDiffuseColor;
  float4 backfaceSpecularColor;
  float4 backfaceColor;
  float backfaceOpacity;
  float backfaceSpecularPower;
  float2 _padding2;
  // Actor-texture ClampToBorder support: Metal's sampler border-color presets
  // cannot represent arbitrary border colors, so when a ClampToBorder texture
  // is active the sampler clamps to edge and this field supplies the border
  // color; .w is 1.0 when the fallback is in use, 0.0 otherwise.
  float4 borderColor;
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
  //VTK::Normal::Dec  // user shader (Metal): insert extra vertex-out/fragment-in varyings here
  float4 vertexColor;    // P1-1A: per-vertex color from scalar mapping
  float2 uv;             // P5-5A: texture coordinates
  float2 scalarCoord;    // interpolated LUT texture coordinate (scalar-texture coloring)
  float3 modelPos;       // Optimized 6-plane clip validation
  uint cellId;           // P2-8: flat-interpolated cell ID (1-based, 0=background)
  uint propId;           // P2-8: flat-interpolated prop ID (1-based, 0=background)
  uint compositeIndex;   // P2-8: flat-interpolated composite index (0 = no composite)
  uint   edgeFlags;      // boundary-edge mask (indexed entry only)
  float2 ePos0;          // window-space triangle corner positions (indexed entry only)
  float2 ePos1;
  float2 ePos2;
};

// Per-draw picking identity. propId is the renderer's PropArray index assigned
// per-render during selection passes; compositeIndex carries the composite
// dataset block index (flat index) for batched blocks.
struct PickIds {
  uint propId;
  uint compositeIndex;
};

// Fragment output with explicit depth. Only used when a coincident polygon
// offset is active; writing depth from the fragment stage disables early-Z, so
// the plain-surface pipelines use fragment_main_nodepth below and let the
// rasterizer handle depth like GL (which only emits gl_FragDepth when the
// Coincident shader replacement is applied).
struct FragmentOutput {
  float4 color [[color(0)]];
  uint4 ids [[color(1)]];
  float depth [[depth(any)]];
};

// Early-Z-friendly surface output: no [[depth(any)]] member, so the hardware
// performs the depth test before the fragment shader and writes depth itself.
struct FragmentOutputNoDepth {
  float4 color [[color(0)]];
  uint4 ids [[color(1)]];
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
// lightLoopBound is the loop upper bound. The surface fragments pass the baked
// function constant kLightCount (exact enabled-light count, so the loop unrolls
// to exactly that many constant-index iterations — matching GL's shader-template
// unrolling of lightColor0, lightDirectionVC1, ...). Every other fragment entry
// passes the plain constant MAX_LIGHTS so it keeps the runtime lightCount guard
// and never references a function constant (those pipelines need no constant
// specialization).
// bakedLightType mirrors GL's per-complexity shaders: for the single-light
// surface pipelines it is the compile-time kLightType (so the type dispatch
// folds to one path); everywhere else it is -1 and the type is read from the
// light uniform.
inline void computePhongLighting(
    float3 N, float3 viewPos, float3 diffuseColor, float3 specularColor, 
    float specularIntensity, float specularPower,
    int lightLoopBound, int bakedLightType,
    constant LightUniforms& lights,
    thread float3& totalDiffuse, thread float3& totalSpecular) {
  
  float3 viewDir = normalize(-viewPos);
  
  #pragma unroll
  for (int i = 0; i < lightLoopBound; ++i)
  {
    if (i >= lights.lightCount) continue;
    Light L = lights.lights[i];
    int lightType = (lightLoopBound == 1 && bakedLightType >= 0) ? bakedLightType : int(L.position.w);
    // L.color is already diffuse*intensity on the CPU (color.w is always 1.0),
    // and L.direction is already unit length (normalized then rotated by the
    // rigid view transform in UpdateLightUniforms), so no per-fragment multiply
    // or normalize — matching GL's pre-baked lightColor0 / lightDirectionVC.
    float3 lightColor = L.color.rgb;
    float attenuation = 1.0;
    float3 toLight;
    
    if (lightType == 0) { // Headlight
      toLight = float3(0.0, 0.0, 1.0);
    } else if (lightType == 1) { // Directional
      toLight = -L.direction.xyz;
    } else { // Point / Spot
      toLight = L.position.xyz - viewPos;
      float dist = length(toLight);
      toLight = dist > 0.00001 ? toLight / dist : float3(0,0,1);
      attenuation = 1.0 / (L.attenuation.x + L.attenuation.y * dist + L.attenuation.z * dist * dist);
      
      if (lightType == 3) { // Spot specifics
        float spotCos = dot(-toLight, L.direction.xyz);
        float spotCutoff = cos(L.direction.w * (M_PI_F / 180.0));
        attenuation *= select(0.0f, pow(max(spotCos, 0.0f), L.attenuation.w), spotCos > spotCutoff);
      }
    }
    float NdotL = max(dot(N, toLight), 0.0);
    // Headlight follows the VTK/OpenGL convention: both diffuse and specular
    // are driven by N.z directly (L == V == H for a headlight).
    float df = (lightType == 0) ? max(N.z, 0.000001f) : NdotL;
    
    totalDiffuse += df * diffuseColor * lightColor * attenuation;
    
    // Only calculate reflect direction / specular if illuminated and valid specular intensity
    if (df > 0.0 && specularIntensity > 0.0)
    {
      float sf;
      if (lightType == 0)
      {
        sf = pow(df, specularPower);
      }
      else
      {
        float3 reflDir = reflect(-toLight, N);
        sf = pow(max(dot(viewDir, reflDir), 0.0), specularPower);
      }
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

// useVertexColor / useTexture are the effective "is this actor colored / is the
// actor texture active" decisions. The specialized surface pipelines pass the
// compile-time function-constant values so the selects fold away exactly like
// GL's shader-template substitution (which bakes the scalar-color path in at
// compile time); the shared OIT/peel/edge pipelines pass the runtime scene
// flags since a single pipeline serves arbitrary actors.
inline ResolvedMaterial resolveMaterial(
    MaterialUniforms material,
    float4 vertexColor, float2 uv,
    texture2d<float> actorTexture, sampler actorSampler,
    bool useVertexColor, bool useTexture,
    bool frontFacing, bool showTexturesOnBackface)
{
    ResolvedMaterial r;
    r.ambient = useVertexColor ? vertexColor.rgb : material.ambientColor.rgb;
    r.diffuse = useVertexColor ? vertexColor.rgb : material.diffuseColor.rgb;
    r.opacity = useVertexColor ? vertexColor.a : material.opacity;
    if (useTexture && (frontFacing || showTexturesOnBackface)) {
        float4 tex;
        if (material.borderColor.w > 0.0f) {
            // ClampToBorder with an arbitrary border color (Metal sampler
            // presets only cover black/white): clamp the sample coordinates to
            // [0,1] and substitute the border color where uv escapes the range.
            float2 clampedUV = clamp(uv, 0.0f, 1.0f);
            tex = actorTexture.sample(actorSampler, clampedUV);
            float outside = max(step(1.0f, abs(uv.x - 0.5f) * 2.0f),
                                step(1.0f, abs(uv.y - 0.5f) * 2.0f));
            tex = mix(tex, material.borderColor, outside);
        } else {
            tex = actorTexture.sample(actorSampler, uv);
        }
        r.ambient *= tex.rgb;
        r.diffuse *= tex.rgb;
        r.opacity *= tex.a;
    }
    return r;
}

// Per-cell color port / scalar-texture coloring: the effective per-fragment
// color. When the mapper built a per-primitive cell-color texture
// (kSceneFlagHasCellTexture) the cell RGBA indexed by primitive id wins over
// the flat per-vertex color; when scalar-texture coloring is active
// (kSceneFlagHasScalarLUT) the LUT lookup replaces both —
// GL's texture(colortexture, colorTCoord) path, driven by the mapper's
// ColorCoordinates + ColorTextureMap. The kHasCellTexture/kHasScalarLUT
// compile-time gates let the lean opaque-surface pipelines drop the texture
// accesses entirely.
inline float4 resolveCellColor(VertexOut in, uint primId,
                               constant SceneUniforms& scene,
                               texture2d<float, access::read> cellColorTex,
                               texture2d<float> lutTexture,
                               sampler lutSampler) {
  if (kHasScalarLUT && (scene.flags & kSceneFlagHasScalarLUT) != 0u)
  {
    return lutTexture.sample(lutSampler, in.scalarCoord);
  }
  if (kHasCellTexture && (scene.flags & kSceneFlagHasCellTexture) != 0u)
  {
    return cellColorTex.read(
      uint2(primId % kCellTextureWidth, primId / kCellTextureWidth));
  }
  return in.vertexColor;
}

// Single-pass surface edges (port of vtkOpenGLPolyDataMapper::ReplaceShaderEdges).
// The edge band is drawn on the surface fragment itself, so no separate depth
// term is added. Per-triangle window-space edge equations are built from the
// three corner positions (passed as constant varyings from the indexed vertex
// entry) and evaluated at the fragment's window position. This mirrors the GL
// geometry shader exactly, avoiding the barycentric/gradient reconstruction.
// Sign conventions match GL: edist > 0 means inside the triangle.
inline void applySurfaceEdges(thread float3& N,
                              thread ResolvedMaterial& r,
                              VertexOut in,
                              constant LightUniforms& lights,
                              constant EdgeUniforms& edge,
                              constant SceneUniforms& scene,
                              float ambientIntensity) {
  if ((edge.flags & 1u) == 0u) return;
  if (in.edgeFlags == 0u) return;

  // Window-space corner positions (y-up, matching GL window layout).
  float2 p[3] = { in.ePos0, in.ePos1, in.ePos2 };
  // Fragment window position flipped to the same y-up convention.
  float2 fp = float2(in.position.x - scene.viewport.x,
                     scene.viewport.w - (in.position.y - scene.viewport.y));

  // GL GS: ccw = sign of the 2D cross of the two edges at corner 0.
  float ccw = sign((p[1].x - p[0].x) * (p[2].y - p[0].y) -
                   (p[1].y - p[0].y) * (p[2].x - p[0].x));

  // GL GS: edgeEqn[i] = (n.x, n.y, 0, -dot(p[i], n)), n = ccw * (-dy, dx)/len.
  float4 eq[3];
  for (int i = 0; i < 3; ++i) {
    float2 e = normalize(p[(i + 1) % 3] - p[i]);
    float2 n = ccw * float2(-e.y, e.x);
    eq[i] = float4(n.x, n.y, 0.0, -dot(p[i], n));
  }

  float lw = max(edge.edgeWidth, 0.001);
  // GL: inactive edges get edgeEqn[i].z = lineWidth so they never win the min.
  // CPU bit layout (c1c2, c2c0, c0c1): bit0 = edge(1,2), bit1 = edge(2,0),
  // bit2 = edge(0,1).
  eq[0].z = (in.edgeFlags & 4u) != 0u ? 0.0 : lw;   // edge 0->1
  eq[1].z = (in.edgeFlags & 1u) != 0u ? 0.0 : lw;   // edge 1->2
  eq[2].z = (in.edgeFlags & 2u) != 0u ? 0.0 : lw;   // edge 2->0

  float ed[3];
  ed[0] = dot(eq[0].xy, fp) + eq[0].w + eq[0].z;
  ed[1] = dot(eq[1].xy, fp) + eq[1].w + eq[1].z;
  ed[2] = dot(eq[2].xy, fp) + eq[2].w + eq[2].z;

  float emix = clamp(0.5 + 0.5 * lw - min(ed[0], min(ed[1], ed[2])), 0.0, 1.0);

  // nearest-edge screen normal (GL cedge.xy)
  float2 n2 = (ed[0] <= ed[1] && ed[0] <= ed[2]) ? eq[0].xy
             : ((ed[1] <= ed[2]) ? eq[1].xy : eq[2].xy);
  float3 tnorm = normalize(cross(N, cross(float3(n2, 0.0), N)));

  float cdist = min(ed[0], ed[1]);
  float rdist = 2.0 * min(cdist, ed[2]) / lw;

  float3 ec = edge.edgeColor.rgb;
  float  eo = edge.edgeColor.a;
  // GL only fakes tubes when lights are present; otherwise the flat branch
  bool tube = ((edge.flags & 2u) != 0u) && lights.lightCount > 0;

  if (tube) {
    // tube branch: mix BOTH diffuse and ambient toward edgeColor
    r.diffuse = mix(r.diffuse, ec, emix * eo);
    // GL mixes toward ambientIntensity*edgeColor (vtkPolyDataFS :706) because
    // the intensity is pre-applied to ambientColor; mirror that here.
    r.ambient = mix(r.ambient, ambientIntensity * ec, emix * eo);
    // self-occlusion A-term (GL :736-737)
    float A = tnorm.z;
    rdist = 0.5 * rdist + 0.5 * (rdist + A) / (1.0 + abs(A));
    float lenZ = clamp(sqrt(max(1.0 - rdist * rdist, 0.0)), 0.0, 1.0);
    float3 tubeN = normalize(rdist * tnorm + N * lenZ);
    N = mix(N, tubeN, emix);
  } else {
    // flat branch (RenderLinesAsTubes off): GL mixes toward the FULL edge
    // color (vtkPolyDataFS :713), so the pre-applied intensity must be
    // dropped. This is what makes edges render at their SetEdgeColor even when
    // the property ambient coefficient is 0.
    r.diffuse = mix(r.diffuse, float3(0.0), emix * eo);
    r.ambient = mix(r.ambient, ec,            emix * eo);
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
                             constant PickIds& pickIds [[buffer(7)]],
                             constant float2* triangleUVs [[buffer(8)]],
                             constant float2* scalarCoords [[buffer(12)]]) {
  VertexOut out;

  float4 worldPos = scene.modelMatrix * float4(in.position, 1.0);
  float4 viewPos = scene.viewMatrix * worldPos;
  out.viewPos = viewPos.xyz;
  out.position = scene.projectionMatrix * viewPos;
  out.viewNormal = scene.normalMatrix * in.normal;
  //VTK::Normal::Impl  // user shader (Metal): vertex-body insertion point (model-space normal is in.normal)
  // Feature-conditional per-vertex loads (compile-time via function constants):
  // the lean surface variant skips the color/UV/ID streams the fragment shader
  // does not consume, so the loads are not pure per-vertex bandwidth. The
  // scalar-coord load is also gated on the runtime scene flag: the shared
  // full-behavior peel/OIT pipelines compile kHasScalarLUT on for every
  // translucent actor, but only actors actually using a scalar LUT set the flag.
  out.vertexColor = kHasSurfaceColors ? vertexColors[vertex_id] : float4(0.0);
  out.uv = kHasActorTexture ? triangleUVs[vertex_id] : float2(0.0);
  out.scalarCoord =
    (kHasScalarLUT && (scene.flags & kSceneFlagHasScalarLUT) != 0u)
    ? scalarCoords[vertex_id]
    : float2(0.0);
  out.modelPos = in.position; // Direct pass for unbounded planes evaluation
  if (kEmitIds)
  {
    out.cellId = cellIds[vertex_id];
    out.propId = mapPropId(pickIds.propId);
    out.compositeIndex = pickIds.compositeIndex;
  }
  else
  {
    out.cellId = 0u;
    out.propId = 0u;
    out.compositeIndex = 0u;
  }
  out.edgeFlags = 0u;
  out.ePos0 = float2(0.0);
  out.ePos1 = float2(0.0);
  out.ePos2 = float2(0.0);

  return out;
}

// Indirection vertex entry for single-pass surface edges. The deduplicated
// triangle vertex arrays are read through the triangle index buffer, while
// edge flags are read by vertex_id (per triangle corner). Drawn non-indexed.
vertex VertexOut vertex_main_indexed(uint vertex_id [[vertex_id]],
                                     constant packed_float3* positions [[buffer(0)]],
                                     constant packed_float3* normals   [[buffer(1)]],
                                     constant SceneUniforms& scene     [[buffer(2)]],
                                     constant float4*        colors    [[buffer(3)]],
                                     constant ClipPlaneUniforms& clip  [[buffer(5)]],
                                     constant uint*          cellIds   [[buffer(6)]],
                                     constant PickIds&       pickIds   [[buffer(7)]],
                                     constant float2*        uvs       [[buffer(8)]],
                                     constant uint*          triIdx    [[buffer(9)]],
                                     constant uint*          eflags    [[buffer(10)]],
                                     constant packed_float3* triPos    [[buffer(11)]],
                                     constant float2*        scalarCoords [[buffer(12)]]) {
  VertexOut out;

  uint idx = triIdx[vertex_id];
  float3 inPos = float3(positions[idx]);
  float3 inNrm = float3(normals[idx]);

  float4 worldPos = scene.modelMatrix * float4(inPos, 1.0);
  float4 viewPos = scene.viewMatrix * worldPos;
  out.viewPos = viewPos.xyz;
  out.position = scene.projectionMatrix * viewPos;
  out.viewNormal = scene.normalMatrix * inNrm;
  out.vertexColor = kHasSurfaceColors ? colors[idx] : float4(0.0);
  out.uv = kHasActorTexture ? uvs[idx] : float2(0.0);
  out.scalarCoord =
    (kHasScalarLUT && (scene.flags & kSceneFlagHasScalarLUT) != 0u)
    ? scalarCoords[idx]
    : float2(0.0);
  out.modelPos = inPos;
  if (kEmitIds)
  {
    out.cellId = cellIds[idx];
    out.propId = mapPropId(pickIds.propId);
    out.compositeIndex = pickIds.compositeIndex;
  }
  else
  {
    out.cellId = 0u;
    out.propId = 0u;
    out.compositeIndex = 0u;
  }
  if (kHasEdgeFlags)
  {
    out.edgeFlags = eflags[vertex_id];

    // Window-space positions of the triangle's 3 corners, for the surface-edge
    // distance field. Record layout: 3 consecutive float3 per corner record
    // (corner 0/1/2 object positions), replicated for each of the 3 corners.
    // All 3 corners emit identical values, so interpolation is exact.
    float2 eW[3];
    for (int j = 0; j < 3; ++j) {
      float4 clip = scene.projectionMatrix * scene.viewMatrix * scene.modelMatrix *
                    float4(float3(triPos[vertex_id * 3 + j]), 1.0);
      float2 ndc = clip.xy / clip.w;
      eW[j] = scene.viewport.zw * (0.5 * ndc + 0.5);   // y-up window coords
    }
    out.ePos0 = eW[0];
    out.ePos1 = eW[1];
    out.ePos2 = eW[2];
  }
  else
  {
    out.edgeFlags = 0u;
    out.ePos0 = float2(0.0);
    out.ePos1 = float2(0.0);
    out.ePos2 = float2(0.0);
  }

  return out;
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------
struct FragmentColorAndIds {
  float4 color;
  uint4 ids;
  bool emitIds;
};

// Shared surface fragment evaluation (used by both the early-Z and the
// coincident-offset entry points so the two never drift).
// VTK-METAL-SCOPE: fragment
inline FragmentColorAndIds evaluateSurfaceFragment(VertexOut in,
                             constant MaterialUniforms& material,
                             constant LightUniforms& lights,
                             constant SceneUniforms& scene,
                             constant EdgeUniforms& edge,
                             constant ClipPlaneUniforms& clipPlanes,
                             texture2d<float, access::read> cellColorTex,
                             constant uint* cellPrimitiveIds,
                             texture2d<float> actorTexture,
                             sampler actorSampler,
                             texture2d<float> lutTexture,
                             sampler lutSampler,
                             uint prim_id,
                             bool frontFacing,
                             bool unlit) {
  if (isClipped(in.modelPos, clipPlanes)) discard_fragment();

  // Match vtkOpenGLPolyDataMapper: backfaces flip the geometric normal (so
  // lighting sees the outward normal) and, when a backface property is set,
  // swap in the backface material. The material swap is compile-time gated on
  // kHasBackface; without a backface property the backface fields mirror the
  // front ones, so skipping the swap is equivalent (and matches GL).
  float3 N = normalize(in.viewNormal);
  MaterialUniforms m = material;
  if (!frontFacing)
  {
    N = -N;
    if (kHasBackface)
    {
      m.ambientColor = material.backfaceAmbientColor;
      m.diffuseColor = material.backfaceDiffuseColor;
      m.specularColor = material.backfaceSpecularColor;
      m.color = material.backfaceColor;
      m.opacity = material.backfaceOpacity;
      m.specularPower = material.backfaceSpecularPower;
    }
  }

  // Feature-conditional material resolution (compile-time via function
  // constants): the lean surface variant skips resolveMaterial/applySurfaceEdges
  // entirely, matching what GL's shader-template substitution produces for a
  // plain opaque surface. The per-cell color port replaces the flat vertex
  // color with the per-primitive cell RGBA when the mapper built a cell-color
  // buffer; scalar-texture coloring (kSceneFlagHasScalarLUT) replaces it with
  // the LUT lookup. GL computes
  //   opacity = opacityUniform * texColor.a
  // for the texture-mapped path, with the per-actor/per-block opacity baked
  // into in.vertexColor.a by the mapper (the vertex stream is otherwise unused
  // in this mode), so the baked alpha is multiplied in after resolveMaterial.
  const bool scalarLUTActive = kHasScalarLUT && (scene.flags & kSceneFlagHasScalarLUT) != 0u;
  float4 effColor = resolveCellColor(in, prim_id, scene, cellColorTex, lutTexture, lutSampler);
  ResolvedMaterial r;
  if (kHasSurfaceColors || kHasActorTexture || kHasSurfaceAlpha || kHasCellTexture || kHasScalarLUT)
  {
    // Compile-time flags: the specialized surface pipelines bake the color and
    // texture decisions in (GL compiles the scalar-color path in directly).
    r = resolveMaterial(m, effColor, in.uv, actorTexture, actorSampler,
      kHasSurfaceColors || kHasCellTexture || scalarLUTActive, kHasActorTexture,
      frontFacing, material.showTexturesOnBackface);
    if (scalarLUTActive)
    {
      r.opacity = in.vertexColor.a * r.opacity;
    }
  }
  else
  {
    r.ambient = m.ambientColor.rgb;
    r.diffuse = m.diffuseColor.rgb;
    r.opacity = m.opacity;
  }

  // GL pre-applies the ambient intensity to ambientColor (ambientIntensity *
  // ambientColorUniform, vtkOpenGLPolyDataMapper Color::Impl) and the edge mix
  // then replaces that whole value with the full edge color. Pre-apply here so
  // the flat edge branch can drop the intensity exactly like GL; surfaces with
  // no edges see the same totalAmbient either way.
  r.ambient = m.ambientColor.w * r.ambient;

  //VTK::Normal::Impl  // user shader (Metal): fragment-body insertion point (resolved material is in r)

  if (kHasEdgeFlags)
  {
    applySurfaceEdges(N, r, in, lights, edge, scene, m.ambientColor.w);
  }

  float3 totalAmbient = r.ambient;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);

  // Unlit draws render with the flat vertex/material color: GL's NoLighting
  // path emits gl_FragData[0] = vec4(ambientColor + diffuseColor, opacity),
  // where ambientColor = ambientIntensity*ambientColorUniform and diffuseColor
  // = diffuseIntensity*diffuseColorUniform. Skip the Phong loop and output the
  // ambient/diffuse material terms (which carry the vertex color) — exactly the
  // pre-applied-intensity form used by the lit path. A light-less renderer is
  // also NoLighting in GL (complexity 0), so lights.lightCount == 0 takes the
  // same flat path.
  const bool flat = unlit || lights.lightCount == 0;
  if (!flat)
  {
    computePhongLighting(N, in.viewPos, r.diffuse, m.specularColor.rgb, m.specularColor.w, m.specularPower, kLightCount, kLightType, lights, totalDiffuse, totalSpecular);
  }

  FragmentColorAndIds out;
  float3 finalColor = flat
    ? (totalAmbient + m.diffuseColor.w * r.diffuse)
    : (totalAmbient + m.diffuseColor.w * totalDiffuse + totalSpecular);
  out.color = float4(finalColor, r.opacity);
  out.emitIds = kEmitIds;
  if (kEmitIds)
  {
    // The flat-interpolated in.cellId is ambiguous for deduplicated (shared
    // vertex) geometry: a shared vertex carries the first triangle's id, so the
    // provoking vertex's "first-wins" value does not always name the owning
    // cell. When the mapper built the per-primitive cell-id buffer
    // (kSceneFlagUsePrimitiveCellIds), read the exact id by primitive id
    // instead — matching GL, which uses gl_PrimitiveID. With no cell ids at all
    // (kEmitIds is a compile-time constant), the whole block is dead.
    uint cellId = ((scene.flags & kSceneFlagUsePrimitiveCellIds) != 0u)
      ? cellPrimitiveIds[prim_id] : in.cellId;
    out.ids = uint4(cellId, in.propId, in.compositeIndex, 0u);
  }
  return out;
}

// Assemble the depth-writing fragment output (color + optional IDs + the
// coincident-offset depth term, mirroring GL's ReplaceShaderCoincidentOffset).
// Shared by fragment_main and fragment_main_line so the two never drift.
// The caller passes the primitive-specific factor/offset: surfaces use the
// polygon parameters, lines use the line parameters (GL selects by primitive
// type in vtkGLSLModCoincidentTopology::GetCoincidentParameters).
inline FragmentOutput makeFragmentOutput(FragmentColorAndIds v, thread const VertexOut& in,
                                         float factor, float offset) {
  FragmentOutput out;
  out.color = v.color;
  if (v.emitIds)
  {
    out.ids = v.ids;
  }
  float cscale = length(float2(dfdx(in.position.z), dfdy(in.position.z)));
  out.depth = in.position.z + factor * cscale + offset / 65000.0;
  return out;
}

// Early-Z variant of the output assembly (no depth output; the rasterizer
// writes depth, keeping early-Z for opaque surface draws).
inline FragmentOutputNoDepth makeFragmentOutputNoDepth(FragmentColorAndIds v) {
  FragmentOutputNoDepth out;
  out.color = v.color;
  if (v.emitIds)
  {
    out.ids = v.ids;
  }
  return out;
}

// Plain-surface variant: no depth output, so the renderer keeps early-Z and
// writes depth through the rasterizer (what GL does when no coincident offset
// is active). This is the lean fast path for opaque surface draws.
fragment FragmentOutputNoDepth fragment_main_nodepth(VertexOut in [[stage_in]],
                              constant MaterialUniforms& material [[buffer(0)]],
                              constant LightUniforms& lights [[buffer(1)]],
                              constant SceneUniforms& scene [[buffer(2)]],
                              constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
                              constant EdgeUniforms& edge [[buffer(4)]],
                              constant ClipPlaneUniforms& clipPlanes [[buffer(5)]],
                              texture2d<float, access::read> cellColorTex [[texture(8)]],
                              constant uint* cellPrimitiveIds [[buffer(7)]],
                              texture2d<float> actorTexture [[texture(0)]],
                              sampler actorSampler [[sampler(0)]],
                              texture2d<float> lutTexture [[texture(9)]],
                              sampler lutSampler [[sampler(1)]],
                              uint prim_id [[primitive_id]],
                              bool frontFacing [[front_facing]]) {
  return makeFragmentOutputNoDepth(
    evaluateSurfaceFragment(in, material, lights, scene, edge,
      clipPlanes, cellColorTex, cellPrimitiveIds, actorTexture, actorSampler,
      lutTexture, lutSampler, prim_id, frontFacing,
      kLightingDisabled || (scene.flags & kSceneFlagLightingDisabled) != 0u));
}

// Coincident-offset variant: writes depth from the fragment stage exactly like
// GL's ReplaceShaderCoincidentOffset. Only selected by the mapper when a
// polygon factor/offset is actually active.
fragment FragmentOutput fragment_main(VertexOut in [[stage_in]],
                              constant MaterialUniforms& material [[buffer(0)]],
                              constant LightUniforms& lights [[buffer(1)]],
                              constant SceneUniforms& scene [[buffer(2)]],
                              constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
                              constant EdgeUniforms& edge [[buffer(4)]],
                              constant ClipPlaneUniforms& clipPlanes [[buffer(5)]],
                              texture2d<float, access::read> cellColorTex [[texture(8)]],
                              constant uint* cellPrimitiveIds [[buffer(7)]],
                              texture2d<float> actorTexture [[texture(0)]],
                              sampler actorSampler [[sampler(0)]],
                              texture2d<float> lutTexture [[texture(9)]],
                              sampler lutSampler [[sampler(1)]],
                              uint prim_id [[primitive_id]],
                              bool frontFacing [[front_facing]]) {
  return makeFragmentOutput(
    evaluateSurfaceFragment(in, material, lights, scene, edge,
      clipPlanes, cellColorTex, cellPrimitiveIds, actorTexture, actorSampler,
      lutTexture, lutSampler, prim_id, frontFacing,
      kLightingDisabled || (scene.flags & kSceneFlagLightingDisabled) != 0u),
    in, coinOffset.polygonFactor, coinOffset.polygonOffset);
}

// Synthesize a camera-aligned normal for line fragments whose input has no
// point normals, matching GL's ReplaceShaderNormal no-point-normals path
// (vtkOpenGLPolyDataMapper). The scale is derived from the raw fdx/fdy
// (abs(fdx)+abs(fdy) is exactly fwidth) so the derivatives are evaluated once.
inline float3 lineDerivativeNormal(float3 viewPos)
{
  float3 fdx = dfdx(viewPos);
  float3 fdy = dfdy(viewPos);
  float scale = 1.0 / length(abs(fdx) + abs(fdy));
  fdx *= scale;
  fdy *= scale;
  float addOrSubtract = (dot(fdx, fdy) >= 0.0) ? 1.0 : -1.0;
  float3 lineVec = addOrSubtract * fdy + fdx;
  return normalize(cross(float3(lineVec.y, -lineVec.x, 0.0), lineVec));
}

// 1px line variant of fragment_main: honors the kSceneFlagLinesUnlit scene flag
// so lines drawn from inputs without point normals (or flat interpolation)
// render with the flat vertex color, matching vtkOpenGLPolyDataMapper's
// NoLighting path for line primitives. Surface draws respect the property
// lighting flag instead (kSceneFlagLightingDisabled) in fragment_main /
// fragment_main_nodepth, so a shared actor flag never affects triangle shading.
fragment FragmentOutput fragment_main_line(VertexOut in [[stage_in]],
                              constant MaterialUniforms& material [[buffer(0)]],
                              constant LightUniforms& lights [[buffer(1)]],
                              constant SceneUniforms& scene [[buffer(2)]],
                              constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
                              constant EdgeUniforms& edge [[buffer(4)]],
                              constant ClipPlaneUniforms& clipPlanes [[buffer(5)]],
                              texture2d<float, access::read> cellColorTex [[texture(8)]],
                              constant uint* cellPrimitiveIds [[buffer(7)]],
                              texture2d<float> actorTexture [[texture(0)]],
                              sampler actorSampler [[sampler(0)]],
                              texture2d<float> lutTexture [[texture(9)]],
                              sampler lutSampler [[sampler(1)]],
                              uint prim_id [[primitive_id]],
                              bool frontFacing [[front_facing]]) {
  VertexOut v = in;
  // Wireframe polygon edges and line cells from inputs without point normals
  // carry the mapper's dummy (0,1,0) vertex normal (the Metal vertex descriptor
  // needs a slot-1 buffer), which would light wrongly. Match GL's
  // ReplaceShaderNormal no-point-normals path (vtkOpenGLPolyDataMapper):
  // generate a camera-aligned normal from screen-space derivatives of the view
  // position so lit wireframes (isTrisOrStrips) shade like the GL PrimitiveTris
  // draw. The mapper clears kSceneFlagLinesHaveNormals for such inputs. The
  // synthesis runs only when the normal is actually consumed: line vertices
  // always carry edgeFlags == 0 (vertex_main), so applySurfaceEdges never reads
  // it, and the flat path (kSceneFlagLinesUnlit or no lights) skips Phong.
  const bool noPointNormals = (scene.flags & kSceneFlagLinesHaveNormals) == 0u;
  const bool litLines =
    (scene.flags & kSceneFlagLinesUnlit) == 0u && lights.lightCount > 0;
  if (noPointNormals && litLines)
  {
    v.viewNormal = lineDerivativeNormal(v.viewPos);
  }
  // Lines are always front-facing in GL (gl_FrontFacing is true for line
  // primitives); the rasterizer's front_facing is only defined for triangles.
  return makeFragmentOutput(
    evaluateSurfaceFragment(v, material, lights, scene, edge,
      clipPlanes, cellColorTex, cellPrimitiveIds, actorTexture, actorSampler,
      lutTexture, lutSampler, prim_id, frontFacing || noPointNormals,
      (scene.flags & kSceneFlagLinesUnlit) != 0u),
    in, coinOffset.lineFactor, coinOffset.lineOffset);
}

// OIT accumulate output: matches the shader-side premultiplication done by
// vtkOrderIndependentTranslucentPass::PostReplaceShaderValues in GL:
//   color0 = (lit.rgb * opacity, opacity)
//   color1 = opacity  (revealage)
// The pipeline blends color0 with (ONE, ONE) for RGB and (ZERO,
// ONE_MINUS_SRC_ALPHA) for alpha, and color1 with (ONE, ONE), mirroring
// glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ZERO, GL_ONE_MINUS_SRC_ALPHA).
struct OITAccumulateOutput {
  float4 color [[color(0)]];
  float reveal [[color(1)]];
  float depth [[depth(any)]];
};

// OIT accumulate variant of fragment_main_line: mirrors fragment_main_oit so
// 1px translucent lines accumulate into the order-independent translucent pass
// (premultiplied color to color(0) RGBA16F, revealage to color(1) R16F). The
// unlit decision uses the kSceneFlagLinesUnlit scene flag like fragment_main_line.
fragment OITAccumulateOutput fragment_main_line_oit(VertexOut in [[stage_in]],
                              constant MaterialUniforms& material [[buffer(0)]],
                              constant LightUniforms& lights [[buffer(1)]],
                              constant SceneUniforms& scene [[buffer(2)]],
                              constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
                              constant EdgeUniforms& edge [[buffer(4)]],
                              constant ClipPlaneUniforms& clipPlanes [[buffer(5)]],
                              texture2d<float, access::read> cellColorTex [[texture(8)]],
                              texture2d<float> actorTexture [[texture(0)]],
                              sampler actorSampler [[sampler(0)]],
                              texture2d<float> lutTexture [[texture(9)]],
                              sampler lutSampler [[sampler(1)]],
                              uint prim_id [[primitive_id]],
                              bool frontFacing [[front_facing]]) {
  if (isClipped(in.modelPos, clipPlanes)) discard_fragment();

  // Same no-point-normals fallback as fragment_main_line: wireframe edges and
  // line cells from inputs without point normals get a camera-aligned
  // derivative normal (GL's ReplaceShaderNormal path) so lit OIT wireframes
  // shade like GL. The synthesis runs only when the normal is actually
  // consumed — line vertices always carry edgeFlags == 0 (vertex_main), so
  // applySurfaceEdges never reads it, and the flat path (kSceneFlagLinesUnlit
  // or no lights) skips Phong. See fragment_main_line for the
  // kSceneFlagLinesHaveNormals rationale.
  VertexOut v = in;
  const bool noPointNormals = (scene.flags & kSceneFlagLinesHaveNormals) == 0u;
  const bool litLines =
    (scene.flags & kSceneFlagLinesUnlit) == 0u && lights.lightCount > 0;
  if (noPointNormals && litLines)
  {
    v.viewNormal = lineDerivativeNormal(v.viewPos);
  }
  // Lines are always front-facing in GL (gl_FrontFacing is true for line
  // primitives); the rasterizer's front_facing is only defined for triangles.
  const bool isFrontFacing = frontFacing || noPointNormals;

  float3 N = normalize(v.viewNormal);
  MaterialUniforms m = material;
  if (!isFrontFacing)
  {
    N = -N;
    m.ambientColor = material.backfaceAmbientColor;
    m.diffuseColor = material.backfaceDiffuseColor;
    m.specularColor = material.backfaceSpecularColor;
    m.color = material.backfaceColor;
    m.opacity = material.backfaceOpacity;
    m.specularPower = material.backfaceSpecularPower;
  }

  const bool scalarLUTActive = kHasScalarLUT && (scene.flags & kSceneFlagHasScalarLUT) != 0u;
  ResolvedMaterial r = resolveMaterial(m, resolveCellColor(in, prim_id, scene, cellColorTex, lutTexture, lutSampler), in.uv, actorTexture, actorSampler,
    (scene.flags & (kSceneFlagHasSurfaceColors | kSceneFlagHasScalarLUT)) != 0u,
    (scene.flags & kSceneFlagHasActorTexture) != 0u,
    isFrontFacing, material.showTexturesOnBackface);
  if (scalarLUTActive)
  {
    r.opacity = in.vertexColor.a * r.opacity;
  }
  r.ambient = m.ambientColor.w * r.ambient;
  applySurfaceEdges(N, r, in, lights, edge, scene, m.ambientColor.w);

  float3 totalAmbient = r.ambient;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);

  const bool unlit = (scene.flags & kSceneFlagLinesUnlit) != 0u;
  const bool flat = unlit || lights.lightCount == 0;
  if (!flat)
  {
    computePhongLighting(N, in.viewPos, r.diffuse, m.specularColor.rgb, m.specularColor.w, m.specularPower, MAX_LIGHTS, -1, lights, totalDiffuse, totalSpecular);
  }

  float3 litRGB = flat
    ? (totalAmbient + m.diffuseColor.w * r.diffuse)
    : (totalAmbient + m.diffuseColor.w * totalDiffuse + totalSpecular);
  float opacity = r.opacity;

  OITAccumulateOutput out;
  out.color = float4(litRGB * opacity, opacity);
  out.reveal = opacity;
  float cscale = length(float2(dfdx(in.position.z), dfdy(in.position.z)));
  out.depth = in.position.z + coinOffset.lineFactor * cscale + coinOffset.lineOffset / 65000.0;
  return out;
}

fragment OITAccumulateOutput fragment_main_oit(VertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant EdgeUniforms& edge [[buffer(4)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(5)]],
    texture2d<float, access::read> cellColorTex [[texture(8)]],
    texture2d<float> actorTexture [[texture(0)]],
    sampler actorSampler [[sampler(0)]],
    texture2d<float> lutTexture [[texture(9)]],
    sampler lutSampler [[sampler(1)]],
    uint prim_id [[primitive_id]],
    bool frontFacing [[front_facing]]) {
  if (isClipped(in.modelPos, clipPlanes)) discard_fragment();

  // Same backface handling as fragment_main.
  float3 N = normalize(in.viewNormal);
  MaterialUniforms m = material;
  if (!frontFacing)
  {
    N = -N;
    m.ambientColor = material.backfaceAmbientColor;
    m.diffuseColor = material.backfaceDiffuseColor;
    m.specularColor = material.backfaceSpecularColor;
    m.color = material.backfaceColor;
    m.opacity = material.backfaceOpacity;
    m.specularPower = material.backfaceSpecularPower;
  }

  const bool scalarLUTActive = kHasScalarLUT && (scene.flags & kSceneFlagHasScalarLUT) != 0u;
  ResolvedMaterial r = resolveMaterial(m, resolveCellColor(in, prim_id, scene, cellColorTex, lutTexture, lutSampler), in.uv, actorTexture, actorSampler,
    (scene.flags & (kSceneFlagHasSurfaceColors | kSceneFlagHasScalarLUT)) != 0u,
    (scene.flags & kSceneFlagHasActorTexture) != 0u,
    frontFacing, material.showTexturesOnBackface);
  if (scalarLUTActive)
  {
    r.opacity = in.vertexColor.a * r.opacity;
  }
  // Pre-apply the ambient intensity so applySurfaceEdges can replace it with
  // the full edge color (see evaluateSurfaceFragment).
  r.ambient = m.ambientColor.w * r.ambient;
  applySurfaceEdges(N, r, in, lights, edge, scene, m.ambientColor.w);

  float3 totalAmbient = r.ambient;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);

  // Property lighting disabled (vtkProperty::SetLighting(false)): emit the flat
  // ambient/diffuse material color like GL's NoLighting path. A light-less
  // renderer is also NoLighting in GL (complexity 0).
  const bool unlit = (scene.flags & kSceneFlagLightingDisabled) != 0u;
  const bool flat = unlit || lights.lightCount == 0;
  if (!flat)
  {
    computePhongLighting(N, in.viewPos, r.diffuse, m.specularColor.rgb, m.specularColor.w, m.specularPower, MAX_LIGHTS, -1, lights, totalDiffuse, totalSpecular);
  }

  float3 litRGB = flat
    ? (totalAmbient + m.diffuseColor.w * r.diffuse)
    : (totalAmbient + m.diffuseColor.w * totalDiffuse + totalSpecular);
  float opacity = r.opacity;

  OITAccumulateOutput out;
  out.color = float4(litRGB * opacity, opacity);
  out.reveal = opacity;

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
  float3 modelPos;
  uint cellId;
  uint propId;
  uint compositeIndex;
};

vertex PointVertexOut vertex_point_main(
    uint vertex_id [[vertex_id]],
    constant packed_float3* point_positions [[buffer(0)]],
    constant SceneUniforms& scene [[buffer(1)]],
    constant packed_float3* point_normals [[buffer(2)]],
    constant float4* point_colors [[buffer(3)]],
    constant packed_float3* point_tangents [[buffer(6)]],
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
  out.viewNormal = scene.normalMatrix * float3(point_normals[vertex_id]);
  out.point_size = 1.0;
  out.pointColor = point_colors[vertex_id];
  out.tangent = scene.normalMatrix * float3(point_tangents[vertex_id]);
  out.uv = point_uvs[vertex_id];
  out.lut_uv = point_color_uvs[vertex_id];
  out.modelPos = pos;
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
                                    constant VertexColorUniforms& vertexColorUniform [[buffer(4)]],
                                    constant ClipPlaneUniforms& clipPlanes [[buffer(5)]]) {
  if (isClipped(in.modelPos, clipPlanes)) discard_fragment();
  float3 N = normalize(in.viewNormal);

  bool showVertices = (scene.flags & kSceneFlagVertexVisibility) != 0u;
  // GL resolves the point base color by whether the scalarColor VBO exists: the
  // per-point scalar color when present, otherwise the material ambient/diffuse
  // colors (ambientColorUniform/diffuseColorUniform in the material path of
  // vtkOpenGLPolyDataMapper::ReplaceShaderColor). Vertex-visibility dots always
  // use the property vertex color (GL's DrawingVertices material path).
  const bool hasPointColors = (scene.flags & kSceneFlagHasPointColors) != 0u;
  float3 ambientBase;
  float3 diffuseBase;
  float baseAlpha;
  if (showVertices)
  {
    ambientBase = diffuseBase = vertexColorUniform.color.rgb;
    baseAlpha = vertexColorUniform.color.a;
  }
  else if (hasPointColors)
  {
    ambientBase = diffuseBase = in.pointColor.rgb;
    baseAlpha = in.pointColor.a;
  }
  else
  {
    ambientBase = material.ambientColor.rgb;
    diffuseBase = material.diffuseColor.rgb;
    baseAlpha = 1.0f;
  }

  // Property lighting disabled (vtkProperty::SetLighting(false)): emit the flat
  // vertex color like GL's NoLighting path. A light-less renderer is also
  // NoLighting in GL (complexity 0).
  const bool unlit = (scene.flags & kSceneFlagLightingDisabled) != 0u;
  const bool flat = unlit || lights.lightCount == 0;
  float3 totalAmbient = material.ambientColor.w * ambientBase;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);

  if (!flat)
  {
    computePhongLighting(N, in.viewPos, diffuseBase, material.specularColor.rgb, material.specularColor.w, material.specularPower, MAX_LIGHTS, -1, lights, totalDiffuse, totalSpecular);
  }

  FragmentOutput out;
  out.color = float4(flat
    ? (totalAmbient + material.diffuseColor.w * diffuseBase)
    : (totalAmbient + material.diffuseColor.w * totalDiffuse + totalSpecular), baseAlpha * material.opacity);
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
  float3 modelPos;
  uint cellId;
  uint propId;
  uint compositeIndex;
};

vertex PointShapedVertexOut vertex_point_shaped_main(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    constant packed_float3* point_positions [[buffer(0)]],
    constant uint* connectivity [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant packed_float3* point_normals [[buffer(3)]],
    constant float4* point_colors [[buffer(4)]],
    constant packed_float3* point_tangents [[buffer(6)]],
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
  out.viewNormal = scene.normalMatrix * float3(point_normals[point_id]);
  out.p_coord = corner;
  out.pointColor = point_colors[point_id];
  out.tangent = scene.normalMatrix * float3(point_tangents[point_id]);
  out.uv = point_uvs[point_id];
  out.lut_uv = point_color_uvs[point_id];
  out.modelPos = pos;
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
    constant VertexColorUniforms& vertexColorUniform [[buffer(4)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(5)]]) {
  FragmentOutput out;

  if (isClipped(in.modelPos, clipPlanes)) discard_fragment();

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

  bool showVertices = (scene.flags & kSceneFlagVertexVisibility) != 0u;
  // Same base-color resolution as fragment_point_main: per-point scalar colors
  // when present (GL's scalarColor VBO), otherwise the material colors.
  const bool hasPointColors = (scene.flags & kSceneFlagHasPointColors) != 0u;
  float3 ambientBase;
  float3 diffuseBase;
  float baseAlpha;
  if (showVertices)
  {
    ambientBase = diffuseBase = vertexColorUniform.color.rgb;
    baseAlpha = vertexColorUniform.color.a;
  }
  else if (hasPointColors)
  {
    ambientBase = diffuseBase = in.pointColor.rgb;
    baseAlpha = in.pointColor.a;
  }
  else
  {
    ambientBase = material.ambientColor.rgb;
    diffuseBase = material.diffuseColor.rgb;
    baseAlpha = 1.0f;
  }

  // Property lighting disabled (vtkProperty::SetLighting(false)): emit the flat
  // vertex color like GL's NoLighting path. A light-less renderer is also
  // NoLighting in GL (complexity 0).
  const bool unlit = (scene.flags & kSceneFlagLightingDisabled) != 0u;
  const bool flat = unlit || lights.lightCount == 0;
  float3 totalAmbient = material.ambientColor.w * ambientBase;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);

  if (!flat)
  {
    computePhongLighting(N, in.viewPos, diffuseBase, material.specularColor.rgb, material.specularColor.w, material.specularPower, MAX_LIGHTS, -1, lights, totalDiffuse, totalSpecular);
  }

  out.color = float4(flat
    ? (totalAmbient + material.diffuseColor.w * diffuseBase)
    : (totalAmbient + material.diffuseColor.w * totalDiffuse + totalSpecular), baseAlpha * material.opacity);
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
  float lineHalfW;
  float3 modelPos;
  uint cellId;
  uint propId;
  uint compositeIndex;
};

// Lateral (across-the-tube) direction in view space for fake-tube shading.
inline float3 lateralDir(float3 segViewDir) {
  float3 s = normalize(segViewDir);
  float3 lat = cross(s, float3(0.0, 0.0, 1.0));
  if (length(lat) < 1e-4) { lat = float3(1.0, 0.0, 0.0); }
  return normalize(lat);
}

inline FragmentOutput shadeLineFragment(LineVertexOut in,
    constant MaterialUniforms& material,
    constant LightUniforms& lights,
    constant CoincidentOffsetUniforms& coinOffset,
    constant SceneUniforms& scene,
    constant ClipPlaneUniforms& clipPlanes)
{
  FragmentOutput out;
  if (isClipped(in.modelPos, clipPlanes)) discard_fragment();
  float3 baseColor = in.vertexColor.rgb;
  float baseAlpha = in.vertexColor.a * material.opacity;

  float3 totalDiffuse = float3(0.0), totalSpecular = float3(0.0);
  // Property lighting disabled (vtkProperty::SetLighting(false)): emit the flat
  // ambient+diffuse material color, matching vtkGLSLModLight's complexity-0
  // path. This is a compile-time branch via kLightingDisabled (GL bakes the
  // complexity into a NoLighting shader), so the unlit pipeline variant drops
  // the Phong loop entirely. A light-less renderer (lights.lightCount == 0) is
  // also complexity-0 in GL and must emit the same flat color.
  const bool flat = kLightingDisabled || lights.lightCount == 0;
  if (!flat)
  {
    // Wide lines emulate GL's native glLineWidth rendering with a flat
    // view-facing normal (uniform lighting across the width). With
    // RenderLinesAsTubes, GL builds real tube geometry whose fragment normals
    // rotate around the tube axis, so the fake-tube normal replicates that
    // cylinder shading instead.
    float3 N;
    if ((scene.flags & kSceneFlagLinesTubeShading) != 0u)
    {
      float r = clamp(in.dist_to_centerline, -1.0, 1.0);
      float lenZ = clamp(sqrt(max(1.0 - r * r, 0.0)), 0.0, 1.0);
      float3 lateral = normalize(in.viewNormal);
      float3 cylinderN = normalize(r * lateral + lenZ * float3(0.0, 0.0, 1.0));
      float emix = clamp(0.5 + in.lineHalfW * (1.0 - abs(r)), 0.0, 1.0);
      N = normalize(mix(float3(0.0, 0.0, 1.0), cylinderN, emix));
    }
    else
    {
      N = float3(0.0, 0.0, 1.0);
    }

    computePhongLighting(N, in.viewPos, baseColor, material.specularColor.rgb,
        material.specularColor.w, material.specularPower, MAX_LIGHTS, -1, lights, totalDiffuse, totalSpecular);
  }

  out.color = float4(material.ambientColor.w * baseColor
                   + material.diffuseColor.w * (flat ? baseColor : totalDiffuse) + totalSpecular, baseAlpha);
  out.ids = uint4(in.cellId, in.propId, in.compositeIndex, 0u);
  float cscale = length(float2(dfdx(in.position.z), dfdy(in.position.z)));
  out.depth = in.position.z + coinOffset.lineFactor * cscale + coinOffset.lineOffset / 65000.0;
  return out;
}

vertex LineVertexOut vertex_thick_line_main(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    constant packed_float3* positions [[buffer(0)]],
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
  float3 segView = (scene.viewMatrix * scene.modelMatrix * float4(p1_MC, 1.0)).xyz
                 - (scene.viewMatrix * scene.modelMatrix * float4(p0_MC, 1.0)).xyz;
  out.viewNormal = lateralDir(segView);
  out.vertexColor = mix(vertexColors[p0_idx], vertexColors[p1_idx], t);
  out.dist_to_centerline = side;
  out.lineHalfW = halfW;
  out.modelPos = mix(p0_MC, p1_MC, t);
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
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(5)]]) {
  return shadeLineFragment(in, material, lights, coinOffset, scene, clipPlanes);
}

// ---------------------------------------------------------------------------
// Round Cap Line Shaders 
// ---------------------------------------------------------------------------
vertex LineVertexOut vertex_round_cap_line_main(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    constant packed_float3* positions [[buffer(0)]],
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
  float3 segView = (scene.viewMatrix * scene.modelMatrix * float4(p1_MC, 1.0)).xyz
                 - (scene.viewMatrix * scene.modelMatrix * float4(p0_MC, 1.0)).xyz;
  out.viewNormal = lateralDir(segView);
  out.vertexColor = mix(vertexColors[p0_idx], vertexColors[p1_idx], p_coord.z);
  out.dist_to_centerline = 2.0 * p_coord.y - 1.0;
  out.lineHalfW = 0.5 * w;
  out.modelPos = mix(p0_MC, p1_MC, p_coord.z);
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
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(5)]]) {
  return shadeLineFragment(in, material, lights, coinOffset, scene, clipPlanes);
}

// ---------------------------------------------------------------------------
// Miter Join Line Shaders
// ---------------------------------------------------------------------------
vertex LineVertexOut vertex_miter_join_line_main(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    constant packed_float3* positions [[buffer(0)]],
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
  out.viewPos = (scene.viewMatrix * scene.modelMatrix * float4(mix(float3(positions[p0_idx]), float3(positions[p1_idx]), t), 1.0)).xyz;
  float3 segView = (scene.viewMatrix * scene.modelMatrix * float4(float3(positions[p1_idx]), 1.0)).xyz
                 - (scene.viewMatrix * scene.modelMatrix * float4(float3(positions[p0_idx]), 1.0)).xyz;
  out.viewNormal = lateralDir(segView);
  out.vertexColor = mix(vertexColors[p0_idx], vertexColors[p1_idx], t);
  out.dist_to_centerline = side;
  out.lineHalfW = halfW;
  out.modelPos = mix(float3(positions[p0_idx]), float3(positions[p1_idx]), t);
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
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(5)]]) {
  return shadeLineFragment(in, material, lights, coinOffset, scene, clipPlanes);
}

// OIT accumulate variant of shadeLineFragment: emits the same lit color as the
// tube line fragments but writes premultiplied color to color(0) (RGBA16F) and
// revealage to color(1) (R16F), mirroring fragment_main_oit so translucent lines
// participate in the order-independent translucent pass like triangles.
inline OITAccumulateOutput shadeLineFragmentOIT(LineVertexOut in,
    constant MaterialUniforms& material,
    constant LightUniforms& lights,
    constant CoincidentOffsetUniforms& coinOffset,
    constant SceneUniforms& scene,
    constant ClipPlaneUniforms& clipPlanes)
{
  if (isClipped(in.modelPos, clipPlanes)) discard_fragment();
  float3 baseColor = in.vertexColor.rgb;
  float opacity = in.vertexColor.a * material.opacity;

  float3 totalDiffuse = float3(0.0), totalSpecular = float3(0.0);
  // Match shadeLineFragment: property lighting disabled or a light-less renderer
  // emits the flat ambient+diffuse material color (GL's complexity-0 path).
  const bool flat = kLightingDisabled || lights.lightCount == 0;
  if (!flat)
  {
    float3 N;
    if ((scene.flags & kSceneFlagLinesTubeShading) != 0u)
    {
      float r = clamp(in.dist_to_centerline, -1.0, 1.0);
      float lenZ = clamp(sqrt(max(1.0 - r * r, 0.0)), 0.0, 1.0);
      float3 lateral = normalize(in.viewNormal);
      float3 cylinderN = normalize(r * lateral + lenZ * float3(0.0, 0.0, 1.0));
      float emix = clamp(0.5 + in.lineHalfW * (1.0 - abs(r)), 0.0, 1.0);
      N = normalize(mix(float3(0.0, 0.0, 1.0), cylinderN, emix));
    }
    else
    {
      N = float3(0.0, 0.0, 1.0);
    }

    computePhongLighting(N, in.viewPos, baseColor, material.specularColor.rgb,
        material.specularColor.w, material.specularPower, MAX_LIGHTS, -1, lights, totalDiffuse, totalSpecular);
  }

  float3 litRGB = material.ambientColor.w * baseColor
                + material.diffuseColor.w * (flat ? baseColor : totalDiffuse) + totalSpecular;

  OITAccumulateOutput out;
  out.color = float4(litRGB * opacity, opacity);
  out.reveal = opacity;
  float cscale = length(float2(dfdx(in.position.z), dfdy(in.position.z)));
  out.depth = in.position.z + coinOffset.lineFactor * cscale + coinOffset.lineOffset / 65000.0;
  return out;
}

fragment OITAccumulateOutput fragment_thick_line_main_oit(
    LineVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(5)]]) {
  return shadeLineFragmentOIT(in, material, lights, coinOffset, scene, clipPlanes);
}

fragment OITAccumulateOutput fragment_round_cap_line_main_oit(
    LineVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(5)]]) {
  return shadeLineFragmentOIT(in, material, lights, coinOffset, scene, clipPlanes);
}

fragment OITAccumulateOutput fragment_miter_join_line_main_oit(
    LineVertexOut in [[stage_in]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(5)]]) {
  return shadeLineFragmentOIT(in, material, lights, coinOffset, scene, clipPlanes);
}

// ---------------------------------------------------------------------------
// Compute Kernels (Tessellation mapping)
// ---------------------------------------------------------------------------
struct TessParams { uint numCells; uint cellIdOffset; uint writeEdgeFlags; uint hasUserEdgeFlags; };

inline bool tessIsBoundary(uint a, uint b, constant uint* connectivity, uint inputOffset, uint npts) {
  for (uint k = 0u; k < npts; ++k) {
    uint x = connectivity[inputOffset + k];
    uint y = connectivity[inputOffset + (k + 1u) % npts];
    if ((x == a && y == b) || (x == b && y == a)) return true;
  }
  return false;
}

// A polygon edge (a -> b) is visible when it is a polygon boundary edge and,
// when a user edge-flag attribute is present (hasUserEdgeFlags), the flag of
// the edge's starting point is non-zero (GL AppendEdgeFlagIndexBuffer). When
// hasUserEdgeFlags is 0 the flag term is skipped, so the boundary-only
// behavior (all polygon edges) is preserved.
inline bool tessEdgeVisible(uint a, uint b, constant uint* connectivity, uint inputOffset,
                            uint npts, constant uint* edgeFlags, uint hasUserEdgeFlags) {
  if (!tessIsBoundary(a, b, connectivity, inputOffset, npts)) return false;
  if (hasUserEdgeFlags != 0u && edgeFlags[a] == 0u) return false;
  return true;
}

kernel void polygonToTriangle(
    device uint* outConnectivity [[buffer(0)]],
    device float* edgeArray [[buffer(1)]],
    device uint* cellIds [[buffer(2)]],
    device uint* outEdgeFlags [[buffer(7)]],
    constant uint* connectivity [[buffer(3)]],
    constant uint* offsets [[buffer(4)]],
    constant uint* primitiveCounts [[buffer(5)]],
    constant TessParams& params [[buffer(6)]],
    constant uint* edgeFlags [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.numCells) return;

  uint npts = offsets[gid + 1u] - offsets[gid];
  uint numTriangles = primitiveCounts[gid + 1u] - primitiveCounts[gid];
  uint outputOffset = primitiveCounts[gid] * 3u;
  uint inputOffset = offsets[gid];

  for (uint i = 0u; i < numTriangles; i++) {
    uint triangleId = primitiveCounts[gid] + i;
    edgeArray[triangleId] = (numTriangles == 1u) ? -1.0 : (i == 0u ? 2.0 : (i == numTriangles - 1u ? 0.0 : 1.0));
    cellIds[triangleId] = gid + params.cellIdOffset + 1u;

    uint c0 = connectivity[inputOffset];
    uint c1 = connectivity[inputOffset + i + 1u];
    uint c2 = connectivity[inputOffset + i + 2u];
    outConnectivity[outputOffset] = c0;
    outConnectivity[outputOffset + 1u] = c1;
    outConnectivity[outputOffset + 2u] = c2;

    // Single-pass surface edges: boundary mask is only needed when the
    // indexed-entry pipeline is active, so skip the O(npts) boundary test
    // entirely otherwise. When a user edge-flag attribute is present, an edge
    // is additionally gated on the flag of its starting corner (GL
    // val & mask semantics).
    if (params.writeEdgeFlags != 0u) {
      uint f12 = tessEdgeVisible(c1, c2, connectivity, inputOffset, npts, edgeFlags, params.hasUserEdgeFlags) ? 1u : 0u;
      uint f20 = tessEdgeVisible(c2, c0, connectivity, inputOffset, npts, edgeFlags, params.hasUserEdgeFlags) ? 1u : 0u;
      uint f01 = tessEdgeVisible(c0, c1, connectivity, inputOffset, npts, edgeFlags, params.hasUserEdgeFlags) ? 1u : 0u;
      uint packed = (f12) | (f20 << 1u) | (f01 << 2u);
      outEdgeFlags[outputOffset] = packed;
      outEdgeFlags[outputOffset + 1u] = packed;
      outEdgeFlags[outputOffset + 2u] = packed;
    }
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
    constant uint* edgeFlags [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.numCells) return;

  // primitiveCounts holds cumulative VISIBLE edge counts; when a user
  // edge-flag attribute is present (writeEdgeFlags != 0) hidden edges are
  // skipped entirely, otherwise every polygon edge is emitted.
  uint outputOffset = primitiveCounts[gid] * 2u;
  uint inputOffset = offsets[gid];
  uint npts = offsets[gid + 1u] - offsets[gid];

  uint out = 0u;
  for (uint i = 0u; i < npts; i++) {
    bool visible =
      params.writeEdgeFlags == 0u || edgeFlags[connectivity[inputOffset + i]] != 0u;
    if (!visible) continue;
    cellIds[primitiveCounts[gid] + out] = gid + params.cellIdOffset + 1u;
    outConnectivity[outputOffset] = connectivity[inputOffset + i];
    outConnectivity[outputOffset + 1u] = connectivity[inputOffset + (i + 1u) % npts];
    outputOffset += 2u;
    out++;
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
    int volTransposed;
};
static_assert(sizeof(NormalComputeUniforms) == 52, "NormalComputeUniforms must be 52 bytes");

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

    // Central differences (6 texel fetches — same as computeGradientFast).
    // volTransposed carries the orientation code: 0=identity, 1=X-depth
    // (tex holds z,y,x), 2=Y-depth (tex holds x,z,y). The mapping is linear,
    // so every neighbor position (base ± axis offset) goes through the SAME
    // swizzle — the data-space gradient components stay in original axes.
    bool vtr = u.volTransposed != 0;
    float3 pPX = pos + float3(gs.x, 0, 0);
    float3 pNX = pos - float3(gs.x, 0, 0);
    float3 pPY = pos + float3(0, gs.y, 0);
    float3 pNY = pos - float3(0, gs.y, 0);
    float3 pPZ = pos + float3(0, 0, gs.z);
    float3 pNZ = pos - float3(0, 0, gs.z);
    float sPX = volume.sample(sVolume, (u.volTransposed == 2) ? pPX.xzy : (vtr ? pPX.zyx : pPX), level(0)).r;
    float sNX = volume.sample(sVolume, (u.volTransposed == 2) ? pNX.xzy : (vtr ? pNX.zyx : pNX), level(0)).r;
    float sPY = volume.sample(sVolume, (u.volTransposed == 2) ? pPY.xzy : (vtr ? pPY.zyx : pPY), level(0)).r;
    float sNY = volume.sample(sVolume, (u.volTransposed == 2) ? pNY.xzy : (vtr ? pNY.zyx : pNY), level(0)).r;
    float sPZ = volume.sample(sVolume, (u.volTransposed == 2) ? pPZ.xzy : (vtr ? pPZ.zyx : pPZ), level(0)).r;
    float sNZ = volume.sample(sVolume, (u.volTransposed == 2) ? pNZ.xzy : (vtr ? pNZ.zyx : pNZ), level(0)).r;

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
 // Mapper2DState.flags bits. Must match the kFlagUse* constants in
 // vtkMetalPolyDataMapper2D.mm.
 constant uint kFlagUseVertexColors = 1u;
 constant uint kFlagUseCellColors = 2u;
 constant uint kFlagPicking = 4u;

 struct Mapper2DState { float4x4 wcvcMatrix; float4 color; float pointSize; float lineWidth; uint flags; uint pickingId; };
 struct Vertex2DIn {
   float2 position [[attribute(0)]];
   float4 color [[attribute(2)]];
 };
 struct Vertex2DOut { float4 position [[position]]; float4 color; };

 // 2D fragment output. Mirrors the 3D surface fragment layout: the RGBA32Uint
 // picking attachment (color(1)) carries {attributeId, propId, compositeIndex,
 // processId} with the prop id stored 1-based (0 = background) so the selector
 // readback can tell an actual prop from an unrendered pixel. The ids output is
 // only written while the picking flag is set; otherwise the pipeline's color(1)
 // attachment is absent (or the output is discarded by Metal).
 struct Fragment2DOut {
   float4 color [[color(0)]];
   uint4 ids [[color(1)]];
 };

// [[point_size]] is only valid for point topology; Metal rejects a pipeline
// whose vertex shader outputs it for triangle/line topology, so the 2D point
// path uses a dedicated struct (same pattern as the 3D glyph point shaders).
struct Vertex2DPointOut {
  float4 position [[position]];
  float4 color;
  float point_size [[point_size]];
};

vertex Vertex2DOut vertex_2d_main(Vertex2DIn in [[stage_in]], constant Mapper2DState& state [[buffer(1)]]) {
  Vertex2DOut out;
  out.position = state.wcvcMatrix * float4(in.position, 0.0, 1.0);
  out.color = (state.flags & kFlagUseVertexColors) != 0u ? in.color : state.color;
  return out;
}

vertex Vertex2DPointOut vertex_2d_point_main(Vertex2DIn in [[stage_in]], constant Mapper2DState& state [[buffer(1)]]) {
  Vertex2DPointOut out;
  out.position = state.wcvcMatrix * float4(in.position, 0.0, 1.0);
  out.color = (state.flags & kFlagUseVertexColors) != 0u ? in.color : state.color;
  out.point_size = max(state.pointSize, 1.0);
  return out;
}

fragment Fragment2DOut fragment_2d_main(Vertex2DOut in [[stage_in]],
                                        constant Mapper2DState& state [[buffer(0)]],
                                        constant float4* cellColors [[buffer(1)]],
                                        uint prim_id [[primitive_id]]) {
  Fragment2DOut out;
  out.ids = uint4(0u);
  if ((state.flags & kFlagUseCellColors) != 0u) {
    out.color = cellColors[prim_id];
  } else {
    out.color = in.color;
  }
  if ((state.flags & kFlagPicking) != 0u) {
    out.ids = uint4(0u, mapPropId(state.pickingId), 0u, 0u);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Thick 2D lines. Metal has no line-width support, so lineWidth > 1 is drawn
// as a screen-space quad per segment (triangle strip of 4 vertices per
// instance), mirroring the 3D mapper's vertex_thick_line_main. The 2D mapper's
// positions are already in viewport pixels, so the expansion happens directly
// in pixel space and the wcvc matrix maps the expanded quad to NDC.
//
// Buffer layout (vertex):
//   buffer(0): packed float2 positions (viewport pixels)
//   buffer(1): line index buffer (uint pairs, one pair per segment/instance)
//   buffer(2): Mapper2DState
//   buffer(3): per-vertex float4 colors
//   buffer(4): per-segment float4 cell colors (when cell scalars are active)
// ---------------------------------------------------------------------------
vertex Vertex2DOut vertex_thick_line_2d_main(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    constant float2* positions [[buffer(0)]],
    constant uint* lineIndices [[buffer(1)]],
    constant Mapper2DState& state [[buffer(2)]],
    constant float4* vertexColors [[buffer(3)]],
    constant float4* cellColors [[buffer(4)]]) {
  const float2 tri_verts[4] = { float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1) };
  float2 p_coord = tri_verts[vertex_id];

  uint p0_idx = lineIndices[instance_id * 2];
  uint p1_idx = lineIndices[instance_id * 2 + 1];
  float2 p0_pix = positions[p0_idx];
  float2 p1_pix = positions[p1_idx];

  float2 delta = p1_pix - p0_pix;
  float segLen = length(delta);
  float2 x_basis = segLen < 0.001 ? float2(1.0, 0.0) : (delta / segLen);
  float2 y_basis = float2(-x_basis.y, x_basis.x);

  float t = (p_coord.x + 1.0) * 0.5;
  float side = p_coord.y;
  float halfW = max(state.lineWidth, 1.0) * 0.5;

  float2 center = mix(p0_pix, p1_pix, t);
  float2 p = center + side * y_basis * halfW;

  Vertex2DOut out;
  out.position = state.wcvcMatrix * float4(p, 0.0, 1.0);
  if ((state.flags & kFlagUseCellColors) != 0u) {
    out.color = cellColors[instance_id];
  } else if ((state.flags & kFlagUseVertexColors) != 0u) {
    out.color = mix(vertexColors[p0_idx], vertexColors[p1_idx], t);
  } else {
    out.color = state.color;
  }
  return out;
}

fragment Fragment2DOut fragment_thick_line_2d_main(Vertex2DOut in [[stage_in]],
                                                   constant Mapper2DState& state [[buffer(0)]]) {
  Fragment2DOut out;
  out.ids = uint4(0u);
  out.color = in.color;
  if ((state.flags & kFlagPicking) != 0u) {
    out.ids = uint4(0u, mapPropId(state.pickingId), 0u, 0u);
  }
  return out;
}

// ---------------------------------------------------------------------------
// 2D Image Mapper shaders (textured quad, used by vtkMetalImageMapper).
// Reuses the Mapper2DState wcvc matrix so image coordinates (viewport pixels,
// VTK bottom-left origin) map to NDC exactly like the plain 2D mapper.
// ---------------------------------------------------------------------------
struct Image2DVertexIn {
  float2 position [[attribute(0)]];
  float2 texCoord [[attribute(1)]];
};

struct Image2DVertexOut {
  float4 position [[position]];
  float2 texCoord;
};

vertex Image2DVertexOut vertex_2d_image_main(Image2DVertexIn in [[stage_in]],
                                             constant Mapper2DState& state [[buffer(1)]]) {
  Image2DVertexOut out;
  out.position = state.wcvcMatrix * float4(in.position, 0.0, 1.0);
  out.texCoord = in.texCoord;
  return out;
}

fragment float4 fragment_2d_image_main(Image2DVertexOut in [[stage_in]],
                                       texture2d<float> imageTexture [[texture(0)]]) {
  return imageTexture.sample(sNearest, in.texCoord);
}

// 2D textured text fragment shader (used by vtkMetalPolyDataMapper2D for
// vtkTextActor / vtkTextMapper). Multiplies the sampled texture color by the
// actor's color/opacity, matching vtkPolyData2DFS.glsl's
// "gl_FragData[0] = gl_FragData[0] * texture2D(texture1, ...)". Fragments with
// fully transparent alpha are discarded (as the GL shader does) so that the
// overlay depth pass only writes depth at glyph pixels; otherwise a
// foreground text quad's full bounding box would occlude background props.
fragment Fragment2DOut fragment_2d_text_main(Image2DVertexOut in [[stage_in]],
                                             constant Mapper2DState& state [[buffer(0)]],
                                             texture2d<float> imageTexture [[texture(0)]]) {
  Fragment2DOut out;
  out.ids = uint4(0u);
  if ((state.flags & kFlagPicking) != 0u) {
    // Picking renders the whole quad as the prop (matching the GL 2D shader,
    // which substitutes gl_FragData[0] = vec4(mapperIndex,1.0) before its alpha
    // discard, so glyph and transparent quad pixels both carry the prop id).
    out.color = float4(1.0);
    out.ids = uint4(0u, mapPropId(state.pickingId), 0u, 0u);
    return out;
  }
  float4 texColor = imageTexture.sample(sNearest, in.texCoord);
  float4 fragColor = texColor * state.color;
  if (fragColor.a <= 0.0) {
    discard_fragment();
  }
  out.color = fragColor;
  return out;
}

// ---------------------------------------------------------------------------
// Image Slice Mapper selection shaders (cell/point-ID picking).
// Encodes the pixel index (computed from the texture coordinates and the image
// dimensions) into the RGBA32Uint picking attachment using the same Ids layout
// as the 3D surface shaders: {attributeId, propId, compositeIndex, processId},
// with attributeId and propId stored 1-based (0 = background) to match
// vtkMetalHardwareSelector::Convert / GetPixelInformation.
// ---------------------------------------------------------------------------
struct SliceSelectionVertexIn {
  float3 position  [[attribute(0)]];
  float2 texCoord  [[attribute(1)]];
};

struct SliceSelectionUniforms {
  float2 imgDims;      // stride dimensions (cells: width-1/height-1, points: width/height)
  uint   propId;       // 0-based selector PropArray index
  uint   compositeIndex;
};

vertex Image2DVertexOut vertex_slice_selection_main(SliceSelectionVertexIn in [[stage_in]],
                                                    constant SceneUniforms& scene [[buffer(1)]]) {
  Image2DVertexOut out;
  float4 clip = scene.projectionMatrix * scene.viewMatrix * scene.modelMatrix *
                float4(in.position, 1.0);
  out.position = clip;
  out.texCoord = in.texCoord;
  return out;
}

struct SliceSelectionFragmentOut {
  uint4 ids [[color(1)]];   // matches the 3D surface shaders' picking attachment
};

fragment SliceSelectionFragmentOut fragment_slice_selection_main(Image2DVertexOut in [[stage_in]],
                                                                 constant SliceSelectionUniforms& u [[buffer(2)]]) {
  int i = int(floor(in.texCoord.x * u.imgDims.x));
  int j = int(floor(in.texCoord.y * u.imgDims.y));
  i = clamp(i, 0, int(u.imgDims.x) - 1);
  j = clamp(j, 0, int(u.imgDims.y) - 1);
  uint pixelId = uint(j) * uint(u.imgDims.x) + uint(i);
  SliceSelectionFragmentOut out;
  out.ids = uint4(pixelId + 1u, u.propId + 1u, u.compositeIndex, 0u);
  return out;
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

// ---------------------------------------------------------------------------
// Background gradient (matches vtkOpenGLRenderer's gradient background)
// ---------------------------------------------------------------------------
struct BackgroundGradientUniforms {
  float4 stopColors[2];    // [0]=Background (bottom), [1]=Background2 (top)
  int mode;                // VTK_GRADIENT_VERTICAL=0, HORIZONTAL=1, RADIAL_SIDE=2, RADIAL_CORNER=3
  int dither;              // add +/-0.5/255 noise, like GL DitherGradient
  float2 _pad;
  // The gradient quad in GL spans the renderer's full (virtual) viewport and is
  // clipped to the current tile, so the fragment's tcoord equals its position
  // normalized across the RENDERER's viewport (not the tile). The tile rect in
  // drawable pixels gives the tile-local position; rendererViewport is the
  // renderer's normalized viewport; tileViewport is that viewport clipped to
  // the window's tile viewport. Mapping tile-local -> tileViewport -> renderer
  // reproduces the GL tcoord, which keeps the gradient coherent across the
  // assembled image for vtkWindowToImageFilter tiling.
  float4 viewportRect;     // (originX, originY, width, height) of the renderer's
                           // tile-cropped viewport, in drawable pixels (Metal
                           // top-left origin).
  float4 rendererViewport; // renderer's normalized viewport (x0, y0, x1, y1)
  float4 tileViewport;     // rendererViewport clipped to the window's tile
                           // viewport, normalized (x0, y0, x1, y1)
};

struct GradientFragmentOutput {
  float4 color [[color(0)]];
  uint4 ids [[color(1)]];
};

fragment GradientFragmentOutput fragment_gradient_background(
    FullscreenVertexOut in [[stage_in]],
    constant BackgroundGradientUniforms& u [[buffer(0)]]) {
  float2 tileLocal = float2((in.position.x - u.viewportRect.x) / u.viewportRect.z,
                            1.0 - (in.position.y - u.viewportRect.y) / u.viewportRect.w);
  float2 viewportSize = u.rendererViewport.zw - u.rendererViewport.xy;
  float2 uv = (tileLocal * (u.tileViewport.zw - u.tileViewport.xy) +
               u.tileViewport.xy - u.rendererViewport.xy) / viewportSize;
  float value = 0.0;
  if (u.mode == 1) {
    value = uv.x;
  } else if (u.mode == 2) {
    value = clamp(length(uv - 0.5) * 2.0, 0.0, 1.0);
  } else if (u.mode == 3) {
    value = length(uv - 0.5) * 1.41421356;
  } else {
    value = uv.y;
  }
  float3 color = mix(u.stopColors[0].rgb, u.stopColors[1].rgb, value);
  if (u.dither != 0) {
    const float granularity = 0.001960784313725;  // 0.5 / 255.0
    float noise = fract(sin(dot(uv, float2(12.9898, 78.233))) * 43758.5453123);
    color += mix(-granularity, granularity, noise);
  }
  GradientFragmentOutput out;
  // GL: the flat background is the clear color (alpha = BackgroundAlpha, no
  // overlay is drawn); the gradient overlay writes alpha 1.0. The alpha is
  // carried in stopColors[0].a to match both.
  out.color = float4(color, u.stopColors[0].a);
  out.ids = uint4(0u, 0u, 0u, 0u);
  return out;
}

// ---------------------------------------------------------------------------
// Textured background (matches vtkOpenGLRenderer's textured background). The v
// coordinate is flipped because vertex_fullscreen_main emits v=1 at the window
// bottom while the texture is uploaded with VTK row 0 (min-y) first, so the
// image appears upright (image top at the window top) exactly like OpenGL.
// ---------------------------------------------------------------------------
struct TexturedBackgroundOutput {
  float4 color [[color(0)]];
  uint4 ids [[color(1)]];
};

fragment TexturedBackgroundOutput fragment_textured_background(
    FullscreenVertexOut in [[stage_in]],
    texture2d<float> backgroundTexture [[texture(0)]]) {
  float2 uv = float2(in.texCoord.x, 1.0 - in.texCoord.y);
  TexturedBackgroundOutput out;
  out.color = backgroundTexture.sample(sVolume, uv);
  out.ids = uint4(0u, 0u, 0u, 0u);
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
    bool frontFacing [[front_facing]],
    constant MaterialUniforms& material [[buffer(0)]],
    constant LightUniforms& lights [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    constant CoincidentOffsetUniforms& coinOffset [[buffer(3)]],
    constant EdgeUniforms& edge [[buffer(4)]],
    constant ClipPlaneUniforms& clipPlanes [[buffer(5)]],
    texture2d<float, access::read> cellColorTex [[texture(8)]],
    texture2d<float> actorTexture [[texture(0)]],
    sampler actorSampler [[sampler(0)]],
    texture2d<float> lutTexture [[texture(9)]],
    sampler lutSampler [[sampler(1)]],
    texture2d<float, access::read> prevFrontTex [[texture(1)]],
    texture2d<float, access::read> prevDepthTex [[texture(2)]],
    uint prim_id [[primitive_id]]) {
  
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

  // Backfaces flip the geometric normal and swap in the backface material.
  float3 N = normalize(in.viewNormal);
  MaterialUniforms m = material;
  if (!frontFacing)
  {
    N = -N;
    m.ambientColor = material.backfaceAmbientColor;
    m.diffuseColor = material.backfaceDiffuseColor;
    m.specularColor = material.backfaceSpecularColor;
    m.color = material.backfaceColor;
    m.opacity = material.backfaceOpacity;
    m.specularPower = material.backfaceSpecularPower;
  }
  const bool scalarLUTActive = kHasScalarLUT && (scene.flags & kSceneFlagHasScalarLUT) != 0u;
  ResolvedMaterial r = resolveMaterial(m, resolveCellColor(in, prim_id, scene, cellColorTex, lutTexture, lutSampler), in.uv, actorTexture, actorSampler,
    (scene.flags & (kSceneFlagHasSurfaceColors | kSceneFlagHasScalarLUT)) != 0u,
    (scene.flags & kSceneFlagHasActorTexture) != 0u,
    frontFacing, material.showTexturesOnBackface);
  if (scalarLUTActive)
  {
    r.opacity = in.vertexColor.a * r.opacity;
  }
  // Pre-apply the ambient intensity so applySurfaceEdges can replace it with
  // the full edge color (see evaluateSurfaceFragment).
  r.ambient = m.ambientColor.w * r.ambient;
  applySurfaceEdges(N, r, in, lights, edge, scene, m.ambientColor.w);
  float3 totalAmbient = r.ambient;
  float3 totalDiffuse = float3(0.0);
  float3 totalSpecular = float3(0.0);

  // Property lighting disabled (vtkProperty::SetLighting(false)): emit the flat
  // ambient/diffuse material color like GL's NoLighting path. A light-less
  // renderer is also NoLighting in GL (complexity 0).
  const bool unlit = (scene.flags & kSceneFlagLightingDisabled) != 0u;
  const bool flat = unlit || lights.lightCount == 0;
  if (!flat)
  {
    computePhongLighting(N, in.viewPos, r.diffuse, m.specularColor.rgb, m.specularColor.w, m.specularPower, MAX_LIGHTS, -1, lights, totalDiffuse, totalSpecular);
  }

  float3 fragRGB = flat
    ? (totalAmbient + m.diffuseColor.w * r.diffuse)
    : (totalAmbient + m.diffuseColor.w * totalDiffuse + totalSpecular);

  if (fragDepth >= minDepth - epsilon && fragDepth <= minDepth + epsilon) {
    float prevAlpha = 1.0 - prevFront.a;
    out.frontDest.rgb = prevAlpha * r.opacity * fragRGB + prevFront.rgb;
    out.frontDest.a = 1.0 - (prevAlpha * (1.0 - r.opacity));
  } else if (fragDepth >= maxDepth - epsilon && fragDepth <= maxDepth + epsilon) {
    out.backTemp = float4(fragRGB * r.opacity, r.opacity);
  }

  return out;
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

// OIT resolve: mirrors vtkOrderIndependentTranslucentPassFinalFS.glsl.
// AccumTex.a holds the accumulated transmittance product (1 - sum of opacities),
// so total opacity = 1 - accum.a. RGB holds the weighted color sum, divided by
// the revealage (sum of opacities) to recover the weighted average color.
// The pipeline blends this over the destination with the standard over blend
// (SRC_ALPHA, ONE_MINUS_SRC_ALPHA).
fragment float4 fragment_oit_resolve(
    FullscreenVertexOut in [[stage_in]],
    texture2d<float, access::read> accumTex [[texture(0)]],
    texture2d<float, access::read> revealTex [[texture(1)]]) {
  
  uint2 pixel = uint2(in.position.xy);
  float4 accum = accumTex.read(pixel);
  float reveal = revealTex.read(pixel).r;
  return float4(accum.rgb / max(reveal, 0.01), 1.0 - accum.a);
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
    constant packed_float3* positions, constant packed_float3* normals,
    constant float4x4* glyphTransforms, constant float3x3* glyphNormalTransforms,
    constant float4* glyphColors, constant uint* glyphPickIds,
    constant SceneUniforms& scene, constant PickIds& pickIds)
{
  T out;
  float3 pos = positions[vertex_id];
  float4 worldPos = scene.modelMatrix * glyphTransforms[instance_id] * float4(pos, 1.0);
  out.viewPos = (scene.viewMatrix * worldPos).xyz;
  out.position = scene.projectionMatrix * float4(out.viewPos, 1.0);
  out.viewNormal = scene.normalMatrix * glyphNormalTransforms[instance_id] * float3(normals[vertex_id]);
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
    constant ClipPlaneUniforms& clipPlanes, uint sceneFlags, float depthBias,
    bool frontFacing)
{
  if (isClipped(in.modelPos, clipPlanes)) discard_fragment();
  // Backface handling mirrors evaluateSurfaceFragment and GL's light mod: flip
  // the geometric normal so lighting sees the outward normal and, when the actor
  // has a backface property, swap in the backface material. Lines and points are
  // always front-facing (GL's rule), so their entry points pass frontFacing true.
  const bool backface = !frontFacing;
  const bool hasBackface = (sceneFlags & kSceneFlagGlyphHasBackface) != 0u;
  // Sources without point normals (e.g. vtkArrowSource) get a geometric normal
  // computed from screen-space derivatives, oriented to face the camera, matching
  // vtkOpenGLPolyDataMapper's no-point-normals fragment path.
  float3 N;
  if ((sceneFlags & kSceneFlagGlyphHasNormals) != 0u)
  {
    N = normalize(in.viewNormal);
    if (backface) N = -N;
  }
  else
  {
    float3 fdx = dfdx(in.viewPos);
    float3 fdy = dfdy(in.viewPos);
    N = normalize(cross(fdx, fdy));
    if ((sceneFlags & kSceneFlagParallelProjection) != 0u)
    {
      if (N.z < 0.0) N = -N;
    }
    else if (dot(N, in.viewPos) > 0.0)
    {
      N = -N;
    }
    if (backface) N = -N;
  }
  // GL colors front faces with the per-glyph color and backfaces with the
  // (backface) material color, and the backface opacity is the property's value
  // without the glyph-color alpha (vtkOpenGLGlyph3DHelper::ReplaceShaderColor).
  MaterialUniforms m = material;
  if (backface && hasBackface)
  {
    m.ambientColor = material.backfaceAmbientColor;
    m.diffuseColor = material.backfaceDiffuseColor;
    m.specularColor = material.backfaceSpecularColor;
    m.color = material.backfaceColor;
    m.opacity = material.backfaceOpacity;
    m.specularPower = material.backfaceSpecularPower;
  }
  const float3 baseAmbient = (backface && hasBackface) ? m.ambientColor.rgb : in.glyphColor.rgb;
  const float3 baseDiffuse = (backface && hasBackface) ? m.diffuseColor.rgb : in.glyphColor.rgb;
  // Property lighting disabled (vtkProperty::SetLighting(false)): emit the flat
  // glyph color like GL's NoLighting path. A light-less renderer is also
  // NoLighting in GL (complexity 0).
  const bool unlit = (sceneFlags & kSceneFlagLightingDisabled) != 0u;
  const bool flat = unlit || lights.lightCount == 0;
  float3 totalDiffuse = float3(0.0), totalSpecular = float3(0.0);
  if (!flat)
  {
    computePhongLighting(N, in.viewPos, baseDiffuse, m.specularColor.rgb,
        m.specularColor.w, m.specularPower, MAX_LIGHTS, -1, lights, totalDiffuse, totalSpecular);
  }
  float3 lit = flat
    ? m.diffuseColor.w * baseDiffuse
    : m.diffuseColor.w * totalDiffuse + totalSpecular;
  FragmentOutput out;
  out.color = float4(m.ambientColor.w * baseAmbient + lit,
                   (backface && hasBackface) ? m.opacity : in.glyphColor.a * m.opacity);
  out.ids = uint4(in.cellId, in.propId, in.compositeIndex, 0u);
  out.depth = in.position.z + depthBias;
  return out;
}

vertex GlyphVertexOut vertex_glyph_main(
    uint vertex_id [[vertex_id]], uint instance_id [[instance_id]],
    constant packed_float3* positions [[buffer(0)]], constant packed_float3* normals [[buffer(1)]],
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
    constant ClipPlaneUniforms& clipPlanes [[buffer(9)]],
    bool frontFacing [[front_facing]]) {
  return shadeGlyphFragment(in, material, lights, clipPlanes, scene.flags, 0.0, frontFacing);
}

vertex GlyphVertexOut vertex_glyph_line_main(
    uint vertex_id [[vertex_id]], uint instance_id [[instance_id]],
    constant packed_float3* positions [[buffer(0)]], constant packed_float3* normals [[buffer(1)]],
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
  return shadeGlyphFragment(in, material, lights, clipPlanes, scene.flags, 0.0, true);
}

vertex GlyphPointVertexOut vertex_glyph_point_main(
    uint vertex_id [[vertex_id]], uint instance_id [[instance_id]],
    constant packed_float3* positions [[buffer(0)]], constant packed_float3* normals [[buffer(1)]],
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
  return shadeGlyphFragment(in, material, lights, clipPlanes, scene.flags, coinOffset.pointOffset / 65000.0, true);
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
// Bakes vtkVolumeProperty::GetInterpolationType() into the pipeline, so the
// unused volume/TF/gradient-opacity sampler path is dead-code eliminated.
constant bool fc_linearInterpolation [[function_constant(5)]];
// Bakes vtkVolumeMapper::GetComputeNormalFromOpacity() into the pipeline: the
// shading normal becomes the gradient of the opacity field (OpenGL
// computeDensityGradient parity) instead of the scalar field.
constant bool fc_computeNormalFromOpacity [[function_constant(18)]];
// Bakes vtkVolumeMapper::GetBlendMode() into the pipeline.
// 0=composite (default), 1=maximum intensity (MIP), 2=minimum intensity (MinIP),
// 3=average intensity (AverageIP), 4=additive.
constant int fc_blendMode [[function_constant(17)]];
// Bakes the independent multi-component path into the pipeline (OpenGL
// independent-components parity). Only volumes with more than one independent
// component enable this; single-component pipelines compile the path out
// entirely, keeping the hot march loop free of its per-sample arrays and
// branches.
constant bool fc_independentComponents [[function_constant(19)]];
// Bakes whether the 2D transfer-function (TF_2D) path is active into the
// pipeline: non-TF_2D pipelines compile the 2D lookup-table sampling (and its
// second-axis fetch) out of the hot loop entirely.
constant bool fc_transfer2D [[function_constant(20)]];
// Bakes the rectilinear-grid path into the pipeline: non-rectilinear
// pipelines compile the per-axis coordinate-curve remapping out of the hot
// loop entirely.
constant bool fc_rectilinear [[function_constant(21)]];
// Bakes the renderer's default (single headlight) lighting state into the
// pipeline: headlight pipelines compile the multi-light accumulation loop out
// entirely, keeping only the fast Phong headlight path.
constant bool fc_defaultLighting [[function_constant(22)]];
// Bakes the active light count into the pipeline so the multi-light loop trip
// count is compile-time. Only reached when fc_defaultLighting is false.
constant int fc_lightCount [[function_constant(23)]];
// Bakes the dependent multi-component RGBA path into the pipeline (OpenGL
// 4-component dependent RGBA parity): color is the raw RGB channels and
// opacity is the last channel mapped through the opacity LUT.
constant bool fc_dependentRGBA [[function_constant(24)]];
// Bakes the dependent multi-component LA path into the pipeline (OpenGL
// 2-component dependent LA parity): color comes from the color LUT at the
// first component and opacity from the opacity LUT at the last component.
constant bool fc_dependentLA [[function_constant(25)]];
// Bakes the RenderToImage (depth-image export) path into the pipeline: the
// first-opaque-sample tracking is compiled out of non-RTT pipelines.
constant bool fc_renderToTexture [[function_constant(26)]];
// Bakes the cropping-region path into the pipeline: non-cropping pipelines
// compile the per-sample crop-region bitmask test out of the hot loop entirely.
constant bool fc_cropping [[function_constant(27)]];
// Bakes the uniform-grid blanking path into the pipeline: non-blanked pipelines
// compile the seven per-sample blanking texture fetches out of the hot loop.
constant bool fc_blanking [[function_constant(28)]];
// March-experiment selector (PERFORMANCE_INVESTIGATION.md section 9/10), driven
// by VTK_METAL_TEST_MARCH_VARIANT. 0 = current behavior.
//   1 = manual 8-tap trilinear (co-compiles 8 same-texture volume samples so the
//       MSL backend overlaps the fetches; output-equivalent to sVolume sampling).
//   2 = clamp_to_zero volume sampler (in-shader clamp preserves edge behavior).
constant int fc_marchVariant [[function_constant(29)]];
// Composite slab tiling (VTK_METAL_TEST_NUM_SLABS > 1): each pass composites
// only a ray-length-fraction index range [ceil(idx*maxSteps/K), ceil((idx+1)*
// maxSteps/K)) from zero, and the mapper combines the partials with
// (ONE, ONE_MINUS_SRC_ALPHA) hardware blending. Front-to-back premultiplied
// `over` is associative, so the combined result equals a single-pass composite
// up to fp rounding (PERFORMANCE_INVESTIGATION.md / minimal_gap phase-2). The
// block is dead-code-eliminated when the flag is clear, so non-slab pipelines
// are bit-identical to the previous code.
constant int fc_slabMode [[function_constant(30)]];
// V31 back-edge exit experiment (VTK_METAL_TEST_DOEXIT=1): reshapes the
// baseline divergent march into a do-while with all exit conditions folded
// into the loop back-edge (divergent_tail_repro V31 root-cause fix). Dead-code
// eliminated when clear, so default pipelines are bit-identical.
constant bool fc_doExit [[function_constant(31)]];
// RG8 pair-packed slices experiment (VTK_METAL_TEST_RG8=1): the volume
// texture stores R=slice 2z / G=slice 2z+1 over a halved-depth RG8 grid; the
// march's trilinear z-blend is reconstructed in-shader (divergent_tail
// V24/V32). Dead-code eliminated when clear.
constant bool fc_volRg8 [[function_constant(32)]];
// Transposed volume representation experiment (VTK_METAL_TEST_TRANSPOSE=1):
// the texture uploads x<->z transposed — the slice axis occupies the
// texture's x extent. Every scalar fetch maps original-orientation coords
// through .zyx and texelCount un-swizzles below; ctpScale/ctpOffset/evalStep
// stay expressed over ORIGINAL dims so ray setup is untouched.
// Root cause (2026-08-22, jitter_gap_repro): Metal's private 3D tiling is
// axis-biased — with the slice axis as texture depth, trilinear z-pair
// fetches under per-pixel jitter phase scatter pay a huge DRAM tax
// (+23 ms jitter delta vs GL's +12 @2048 oblique); transposing collapses it
// to +5 ms with byte-identical renders. Dead-code eliminated when clear.
// Mutually exclusive with fc_volRg8 (mapper refuses the combination).
constant bool fc_volTransposed [[function_constant(33)]];

// §29 orientation companion to fc_volTransposed: when set, the upload moved the
// original Y axis into texture depth instead of X (texture holds (W,D,H),
// fetch coords map through .xzy). Compile-time specialized per pipeline so the
// hot fetch path keeps its hard-coded swizzle.
constant bool fc_volTransposedY [[function_constant(34)]];

// Two-level occupancy summary (VTK_METAL_TEST_MM_BLOCKS): compile-time gate
// for the block-summary fast path in the minmax lattice walk. Specialized per
// pipeline so non-block pipelines keep byte-identical codegen to HEAD and
// block pipelines fold every gate away.
constant bool fc_mmBlocks [[function_constant(35)]];
// §35.5 headroom A/B (VTK_METAL_TEST_MM_SUPER): third occupancy level — an
// R8 texture marking whole 8³-block groups of the block summary that are ALL
// empty, letting the walk leap 64 fine cells per fetch. Output-equivalent by
// construction (super-empty => every covered block empty => every cell empty);
// landings stay on the step lattice like the block leaps.
constant bool fc_mmSuper [[function_constant(36)]];
// §35.14 async segment pre-pass (VTK_METAL_TEST_MM_SEG=1, featureMaskExtra
// bit 16): the march consumes per-ray skip segments precomputed by the
// volume_segment_build compute kernel from the rasterized ray atlas
// (fragment_volume_ray_atlas). When set, the mv9 preamble walk compiles out
// entirely and the loop hops precomputed gaps with integer tests. The legacy
// preamble is retained under fc_segHop=false so pipelines stay comparable.
constant bool fc_segHop [[function_constant(37)]];

// §38 TF-adaptive opacity-saturation exit (VTK_METAL_TEST_EXIT_THETA ->
// VolumeMapperUniforms::exitAlpha): compile-time switch between the legacy
// 8-bit latch threshold (1 - 1/255) and a uniform-supplied accumulated-
// opacity exit. fc=false folds to the exact legacy literal; fc=true reads
// the uniform once per fragment. Motivated by low-max-opacity CLUTs
// (Airways II tops at 0.25/sample, doc §35.4) whose rays never latch.
constant bool fc_exitTheta [[function_constant(38)]];

// §38.10 compute-march unroll-batch specialization (VTK_METAL_TEST_CM_BATCH):
// compile-time ladder width for marchRayFromAtlasCore. Unlike the runtime
// batchOverride probe, a function constant lets the compiler eliminate dead
// rungs of the 48-wide fetch/composite ladder, shrinking register allocation
// (occupancy) — the FS path keeps its own runtime MaxBatchWidth.
// 0 = unset -> fall back to the uniform-driven runtime value.
constant int fc_cmBatch [[function_constant(39)]];

// §38.12 stride-parity split: when true, the MAIN 32-wide rung body is
// executed as two independent fetch/composite halves with NO exit test
// between them. Mathematically transparent (same per-sample ops in the
// same sequential order; ADVANCE(32) still tests once), but peak register
// liveness halves IF the scheduler keeps the halves ordered — restoring
// legacy-batch output semantics at reduced register pressure.
constant bool fc_cmSplit [[function_constant(40)]];

// Fragment compile-time batch specialization (VTK_METAL_TEST_FRAG_BATCH):
// mirrors fc_cmBatch but for the fragment march ladder. 0 = unset -> runtime
// MaxBatchWidth. Lets the compiler shed dead ladder rungs to cut registers.
constant int fc_fragBatch [[function_constant(41)]];

// Grad 4-fetch central via gather (VTK_METAL_TEST_GRAD4): 2 fetches vs 6, 33% save
constant bool fc_grad4 [[function_constant(42)]];
// Float vs half for sPX (VTK_METAL_TEST_GRAD_FLOAT): float reduces thr headroom
constant bool fc_gradFloat [[function_constant(43)]];
// SD-aware batch cap / grad (fine SD <0.75 world units): shade 2 vs 4 and 4-fetch
constant bool fc_fineSD [[function_constant(44)]];
// Grad via sNearest (VTK_METAL_TEST_GRAD_NEAREST): 6*1 texel vs 6*8, -10% but thr 5.21
constant bool fc_gradNearest [[function_constant(45)]];
// SetupVolumeRay specializations for fixed per-fragment overhead (§17 SD4 30% vs 8%): depth/cameraInside dead-strip
constant bool fc_useDepthTexture [[function_constant(46)]];
constant bool fc_useCameraInside [[function_constant(47)]];
// Dense coarse bypass for per-batch minMax preamble (§17 20*2+1): dense volumes skip R8 fetches
constant bool fc_dense [[function_constant(48)]];
// Volume scalar nearest for coarse SD4 4x stride (§17 world stride): 1 vs 8 texels, -10% thr 2.02
constant bool fc_volumeNearestCoarse [[function_constant(49)]];
// Quad-coop grad §13.5: 4 sC share 24->4 fetches via quad_shuffle, ~30% SD0.5 thr0 if pos+dx coherence
constant bool fc_quadGrad [[function_constant(51)]];
// Cinematic — shaded DVR (wax AO/SSS, single 8x8, no Woodcock/HG/NEE)
// fc_cinematic/fc_denoise removed — reads u.cinematicEnabled, denoise via separate kernel

  // §38.17 segment consume for the COMPUTE marcher (VTK_METAL_TEST_MM_SEG):
// mirrors the fragment engine's fc_segHop — per-ray skip gaps precomputed by
// volume_segment_build replace the march-time preamble walk entirely, moving
// the summary probes off the serialized dependency chain into the
// throughput-bound builder (§38.16.2: the probes themselves are the tax at
// low frame parallelism). Gaps are fine-lattice granular; decisions are the
// builder's own walk, not the preamble's.
constant bool fc_cmSegHop [[function_constant(50)]];

// Map an original-orientation sample position into texture space for the live
// transposed representation (no-op when clear).
inline float3 volumeFetchSwizzle(float3 pos) {
  if (fc_volTransposedY) return pos.xzy;
  if (fc_volTransposed)  return pos.zyx;
  return pos;
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
  float useLinearVolumeInterpolation; // 1.0 = trilinear (VTK_LINEAR_INTERPOLATION), 0.0 = nearest
  // Min-max acceleration texture
  float useMinMaxAccel;
  float minMaxDimX;
  float minMaxDimY;
  float minMaxDimZ;
  float useRenderToImage;
  float clampDepthToBackface;
  // 2D transfer function mode (TF_2D): sample the primary scalar against the
  // Y-axis scalar array in a 2D color/opacity lookup image. When
  // transfer2DUseGradient == 1.0 the second axis is the gradient magnitude
  // instead of a Y-axis array (OpenGL Transfer2DUseGradient parity).
  float useTransfer2D;
  float transfer2DYAxisScale;  // yNorm = yRaw * scale + bias
  float transfer2DYAxisBias;
  float transfer2DUseGradient; // 1.0 = y-axis is gradient magnitude
  // AverageIP scalar range (native scalar units, pre-divided by the volume
  // normalization factor so it compares against scalarMin + (scalarMax-scalarMin)*norm)
  float averageIPRangeMin;
  float averageIPRangeMax;
  // Mask type: 0 = label map (colored via labelMapTransferTexture), 1 = binary
  // (samples with mask <= 0 are skipped, matching the OpenGL backend).
  float maskType;
  // Final color window/level (matches OpenGL in_scale/in_bias in finalizeRayCast)
  float finalColorScale;
  float finalColorBias;
  // Uniform-grid blanking (ghost arrays): blankingMode 1=cell, 2=point, 3=both.
  // Mirrors the OpenGL backend's g_skip blanking logic.
  float useBlanking;
  float blankingMode;
  // Image-data direction support (OpenGL TextureToDataset parity): maps [0,1]
  // texture coords to model-space (rotated dataset) coords, and its inverse.
  float4x4 textureToVolume;
  float4x4 volumeToTexture;
  // Independent multi-component support (OpenGL in_scalarsRange parity).
  // scalarMinComp/scalarMaxComp are per-component scalar ranges divided by the
  // volume normalization factor, in the same [0,1] space as the raw sample.
  half scalarMinComp[4];
  half _padSMComp[4];
  half scalarMaxComp[4];
  half _padSMaxComp[4];
  float componentWeight[4];
  uint numComponents;
  float useIndependentComponents;
  // 1.0 = dependent 2-component (LA): color is looked up in the color LUT at
  // the first component's normalized value (RGB channels) and opacity in the
  // opacity LUT at the LAST component's normalized value (A channel) — OpenGL
  // computeColor/computeOpacity LA parity. The single RGBA transfer-function
  // texture stores RGB over component 0's scalar range and A over the last
  // component's scalar range, so it is sampled twice.
  float useDependentLA;
  float _padIndependent[2];
  float useDependentRGBA;
  // 1.0 = shade with the opacity-field gradient (OpenGL
  // vtkVolumeMapper::GetComputeNormalFromOpacity parity). Occupies what would
  // otherwise be the float4-alignment pad before ambientColorComp.
  float useComputeNormalFromOpacity;
  // Per-component material (OpenGL in_ambient[i]/in_diffuse[i]/in_specular[i]/
  // in_shininess[i] parity). Only consulted by the independent path, which
  // shades each component against its own material and its own gradient.
  float4 ambientColorComp[4];
  float4 diffuseColorComp[4];
  float4 specularColorComp[4];
  float shininessComp[4];
  // Hardware-selection (vtkHardwareSelector) support — OpenGL PickingActorPassExit
  // / PickingIdLow24PassExit parity. The fragment_volume_selection_main variants
  // write {voxelId, propId, compositeIndex} into the RGBA32Uint color(1)
  // attachment wherever the ray accumulated opacity > 3/255; selectionMode == 0
  // keeps the normal path.
  float selectionMode;
  float _padSel[3];
  uint selectionPropId;          // 0-based selector PropArray index (shader adds 1)
  uint selectionCompositeIndex;  // 0 for a single vtkVolume
  uint selectionVolumeDimX;      // volume dimensions for the voxel index
  uint selectionVolumeDimY;
  uint selectionVolumeDimZ;
  uint _padSelEnd[3];
  // Parallel-projection support (OpenGL in_projectionDirection parity): when
  // useParallelProjection is set every ray is built from the interpolated
  // proxy-box position along this constant direction (in [0,1] normalized
  // volume space) instead of converging rays from the camera position. w unused.
  float useParallelProjection;
  float projectionDirection[4];
  float _padParallelEnd[3];
  // Precomputed NDC -> [0,1] normalized volume-space matrix (folds
  // inverseViewProjection * worldToVolume * the volume-bounds normalize into a
  // single transform) so the fullscreen/grid ray setup and depth-termination
  // paths do one matrix-vector multiply instead of two matrix chains plus a
  // bounds re-normalize per fragment.
  float4x4 ndcToVolume;
  // Rectilinear-grid support (OpenGL in_coordTexs / in_coordsScale /
  // in_coordsBias parity): when useRectilinear is set, the fragment shader walks
  // the per-axis coordinate curves (rectCoords buffer at fragment buffer(5),
  // float3 per index padded to the longest axis) to map each sample's data-space
  // position back to the index-space texture coordinate instead of sampling the
  // uniform-spacing proxy directly. rectCoordsSizes xyz = point count per axis.
  float useRectilinear;
  float _padRect[3];
  float4 rectCoordsSizes;   // xyz = number of coordinates per axis
  float4 rectCoordsScale;   // per-axis GetScaleAndBias scale
  float4 rectCoordsBias;    // per-axis GetScaleAndBias bias
  // Camera-inside near-plane clip (OpenGL near-plane proxy-clip parity): when
  // the camera is inside the volume (near frustum plane crosses the bounding
  // box), OpenGL clips the proxy box against the near plane — pushed into the
  // volume by a precision offset — and starts sampling there. Metal's
  // fullscreen/proxy ray setup reconstructs the ray from the eye and only
  // intersects the box, so setupVolumeRay clamps the entry to this plane.
  // Origin and normal are in [0,1] normalized volume space.
  float useCameraInsideNearClip;
  float _padNearClip[3];
  float4 cameraInsideNearPlaneOrigin;
  float4 cameraInsideNearPlaneNormal;
  // OpenGL proxy-box parity: the camera-OUTSIDE proxy box is uploaded in
  // dataset (model) space (like the camera-inside cap), so the vertex shader
  // forwards in.position unchanged instead of scaling the unit cube.
  float useDataSpaceBoxVertices;
  // 1.0 = use Interleaved Gradient Noise (Jimenez 2014) for sample jittering
  // instead of the GL-parity blue-noise tile (kBlueNoise64).
  float useIGNJitter;
  // Pixels per IGN-jitter coherence block (default 1 = legacy per-pixel).
  // Jittering every fragment independently makes adjacent lanes take divergent
  // min-max skip paths; a small block keeps the stochastic offset while
  // restoring lockstep marching (measured -20% at sample distance 4, bit-identical
  // at 0.5 where sub-voxel offsets round to the same lattice). Opt-in: the
  // legacy per-pixel output stays the default for bit-identical renders.
  float jitterBlockSize;
  // Non-divergent march (PERFORMANCE_INVESTIGATION.md section 4): when > 0, the
  // fragment march runs a uniform iteration count (frame-max ray-box chord /
  // sample distance, computed per frame on the CPU) with all data-dependent
  // exits predicated instead of breaking, so SIMT lanes stay locked. 0 keeps
  // the legacy per-fragment loop bound. Occupies the former _padDSBV slot.
  float maxStepsFrame;
  // Adaptive-width march cap (fc_marchVariant 9): the largest batch width the
  // shader may dispatch. SINGLE-TIER 32 at all sample distances (HARNESS_VS_APP_GAP
  // §37.11): with the block-summary leaps default-on, wide batches no longer
  // waste slots, and 32 measured fastest in every cell probed (mm and raw arms,
  // SD0.5-4, 400^2-4096^2). The unrolled ladder tops out at 48; caps >= 48
  // dispatch identically. VTK_METAL_TEST_MARCH_CAP overrides.
  float maxBatchWidth;
  // §37.15 block-or-nothing (VTK_METAL_TEST_MM_BLOCKSONLY): > 0.5 disables
  // the per-cell tier of the march preamble — super/block leaps stay active,
  // mixed blocks dispatch their batch un-walked. Byte-identical output to the
  // default walk (dropped skips cover provably-zero samples at unchanged
  // positions).
  float mmBlocksOnly;
  // §37.17 leap-granularity selector (VTK_METAL_TEST_MM_LEAPLEVEL): 2 =
  // super+block leaps (default), 1 = super only, <=0 = none. Coherence
  // probe for the axis-chord deficit (SolidFlat proved the deficit is
  // 100% leap dynamics; fewer/larger leaps cut lane-scatter events).
  float mmLeapLevel;
  // §37.18 block-summary size in fine cells (VTK_METAL_TEST_MM_BLOCKSIZE,
  // {4,8,16,32}, default 8). Must match the CPU VolumeMinMaxBlockSize() that
  // sizes/builds minMaxBlockTexture; supers stay fixed at 64-cell tiles, so
  // blocks-per-super = 64 / mmBlockSizeCells.
  float mmBlockSizeCells;
  // §37.19 warp-coherent skipping (VTK_METAL_TEST_MM_WARPMIN): > 0.5 makes
  // each outer iteration probe the current block and advance the whole
  // SIMD-group by the simd-min leap; zero (any dissent) falls back to the
  // legacy per-lane walk. Skipped samples are provably-zero at unchanged
  // positions, so output stays byte-identical to the default walk.
  float mmWarpMin;
  // §38 TF-adaptive exit threshold (fc_exitTheta): accumulated-opacity value
  // that terminates the march, replacing the legacy 8-bit latch (1 - 1/255).
  // Read once per fragment; only meaningful when fc_exitTheta is true.
  float exitAlpha;
  // Cinematic rendering (optimal compute) — must match C++ VolumeMapperUniforms tail
  uint cinematicSamples;
  uint cinematicMaxBounces;
  float cinematicScatteringAnisotropy;
  float cinematicReach;
  float cinematicBlend;
  float cinematicDenoise;
  float subsurfaceColorR;
  float subsurfaceColorG;
  float subsurfaceColorB;
  float subsurfaceStrength;
  uint cinematicFrameSeed;
  uint cinematicAccumCount;
  float cinematicMajorantSigma;
  uint cinematicQuality; // 0 Preview, 1 PathTraced
  float cinematicEnabled; // 0 off, 1 preview, 2 PT
  float cinematicEnv; // env radiance for PT (0 black, 0.2 silhouette)
  float _padCinematicEnd[2];
};

inline float3 projectionDir(constant VolumeMapperUniforms& u) {
    return float3(u.projectionDirection[0], u.projectionDirection[1], u.projectionDirection[2]);
}

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
  float4 slabInfo;      // slabIndex, slabCount, slabAxis, spatialMode (fc_slabMode)
};

vertex VolumeVertexOut vertex_volume_main(
    VolumeVertexIn in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]]) {
  VolumeVertexOut out;

  // Camera-inside (useCameraInsideNearClip set) and the data-space camera-
  // outside box (useDataSpaceBoxVertices set): the vertex buffer holds
  // data-space proxy positions (OpenGL parity: GL uploads the clipped/densified
  // geometry in dataset space and interpolates in_vertexPos directly), so the
  // rasterizer interpolates in data space and the interpolated anchor matches
  // GL's ip_vertexPos to float32. (The unit-cube convention is retired; the
  // camera-outside box is densified and uploaded in model space too.)
  float3 modelPos;
  if (volumeUniforms.useCameraInsideNearClip > 0.5 ||
      volumeUniforms.useDataSpaceBoxVertices > 0.5)
  {
    modelPos = in.position;
  }
  else
  {
    modelPos = b.volumeBoundsMin.xyz + in.position * (b.volumeBoundsMax.xyz - b.volumeBoundsMin.xyz);
  }
  out.position = volumeUniforms.viewProjection * volumeUniforms.volumeToWorld * float4(modelPos, 1.0);
  if (volumeUniforms.useCameraInsideNearClip > 0.5)
  {
    out.localPos = modelPos;  // data-space (interpolated in data space like GL)
  }
  else
  {
    out.localPos = (modelPos - volumeUniforms.volumeBoundsMin.xyz) / max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  }
  out.instanceID = 0;
  return out;
}

struct VolumeFragmentOut { float4 color [[color(0)]]; };
struct VolumeFragmentOutRTT { float4 color [[color(0)]]; float depth [[color(1)]]; };
struct VolumeSelectionOut { float4 color [[color(0)]]; uint4 ids [[color(1)]]; };

// Shared picking-id encoding for the hardware-selection fragment variants
// (OpenGL PickingActorPassExit / PickingIdLow24PassExit parity): returns the
// RGBA32Uint attachment value {voxelIdx + 1, propId + 1, compositeIndex, 0}
// wherever the ray accumulated opacity > 3/255, otherwise 0. The ids are
// encoded as value + 1 because vtkMetalHardwareSelector decodes - 1 and index 0
// means "empty space".
inline uint4 volumeSelectionIds(float3 entryPoint, float accumulatedOpacity,
    constant VolumeMapperUniforms& volumeUniforms) {
  if (accumulatedOpacity <= 3.0 / 255.0) { return uint4(0u); }
  float3 dimsF = float3(float(volumeUniforms.selectionVolumeDimX),
      float(volumeUniforms.selectionVolumeDimY), float(volumeUniforms.selectionVolumeDimZ));
  uint3 voxelCoords = uint3(entryPoint * dimsF);
  uint dimX = max(volumeUniforms.selectionVolumeDimX, 1u);
  uint dimY = max(volumeUniforms.selectionVolumeDimY, 1u);
  uint voxelIdx = dimX * dimY * voxelCoords.z + dimX * voxelCoords.y + voxelCoords.x + 1u;
  return uint4(voxelIdx, volumeUniforms.selectionPropId + 1u,
      volumeUniforms.selectionCompositeIndex, 0u);
}

// The OpenGL raycaster marches the ray with an unbounded loop, terminating only
// when the sample position passes the exit face (g_currentT >= g_terminatePointMax)
// or the opacity threshold is reached. Metal's maxSteps is therefore computed from
// the ray length / step size alone (always finite for a bounded volume); a fixed
// sample-count clamp would truncate long camera-inside rays at fine steps and
// render the far end of the volume dimmer than GL.

// OpenGL's jitter noise, bit-identical (vtkOpenGLRenderWindow::GetNoiseTextureUnit):
// a 64x64 blue-noise tile (BlueNoiseTexture64x64.jpg) decoded by vtkJPEGReader,
// component 0 (luminance) / 255.0f, NEAREST filter + REPEAT wrap, sampled by the
// volume shaders at gl_FragCoord.xy / 64 (vtkVolumeShaderComposer.h). Both GL and
// Metal evaluate at the pixel center (pixel + 0.5), but the two coordinate systems
// have opposite y origins: GL gl_FragCoord.y counts from the bottom of the viewport,
// Metal in.position.y from the top. For the same physical pixel (px, py_mt) in
// Metal top-down coords, GL sees py_gl = H-1-py_mt, and NEAREST + REPEAT picks
// texture texel (px % 64, py_gl % 64). kBlueNoise64 holds the same 64x64 luminance
// tile in the JPEG's top-down orientation (PIL/libjpeg byte order, verified against
// vtkJPEGReader's own decode of BlueNoiseTexture64x64.jpg). vtkJPEGReader writes its
// output rows bottom-up (vtkJPEGReader.cxx: destLine = height - output_scanline), so
// GL's texture row py_gl holds JPEG row (63 - py_gl); with py_gl = H-1-py_mt the
// JPEG row GL samples is (py_mt - H) mod 64. That is the row this function indexes,
// so every pass samples the exact same noise value GL samples for each pixel. This
// replaces the former Interleaved Gradient Noise (Jimenez 2014), whose per-pixel
// values differed from GL's blue noise and shifted the sample-lattice phase on
// jittered renders.
constant uchar kBlueNoise64[4096] = {
  64,245,203,175,58,145,97,137,122,60,109,209,22,218,150,104,
  247,82,119,224,6,82,228,165,45,240,111,41,122,95,200,37,
  227,192,252,206,148,14,84,136,246,194,178,225,57,89,133,168,
  49,180,142,6,53,164,29,204,17,75,196,7,157,183,90,199,
  172,143,27,129,110,255,35,210,81,198,173,145,164,10,73,129,
  183,23,155,47,191,172,70,8,188,73,0,201,160,65,142,176,
  109,128,62,99,176,232,170,108,30,76,2,238,162,193,2,244,
  222,26,68,195,246,89,143,128,246,176,229,96,235,22,216,113,
  234,49,80,156,224,71,19,167,233,23,45,86,126,243,187,93,
  7,217,162,104,20,255,122,148,210,130,88,234,21,53,250,13,
  78,151,25,5,131,42,59,192,222,92,130,20,211,46,108,146,
  80,106,234,115,41,104,218,70,52,146,32,127,48,143,62,8,
  189,108,215,4,48,199,179,102,146,1,226,251,55,28,167,43,
  246,62,210,138,85,204,33,57,242,31,169,153,113,192,204,129,
  178,240,217,193,248,103,28,239,167,54,156,107,79,247,33,193,
  62,176,147,23,211,171,18,186,113,211,94,75,174,200,83,160,
  31,72,177,239,158,83,120,60,216,126,156,100,181,140,213,107,
  141,116,31,227,73,150,176,106,95,225,49,66,228,101,86,8,
  43,114,85,140,71,163,197,83,138,12,230,184,137,64,163,212,
  8,132,253,76,47,168,233,80,9,239,191,14,255,112,231,129,
  247,150,108,66,131,206,25,245,91,67,27,201,13,80,223,17,
  205,87,176,51,248,19,222,8,140,159,201,20,134,26,169,215,
  199,67,35,166,53,221,101,19,116,214,58,194,41,224,117,94,
  236,34,204,98,144,120,36,95,136,168,57,137,157,8,90,40,
  194,14,222,95,22,223,140,45,187,174,239,112,53,188,131,64,
  171,152,14,100,135,115,206,62,191,80,117,236,189,59,146,248,
  102,152,229,123,3,178,35,150,187,250,91,4,94,176,22,184,
  56,114,166,0,216,248,62,210,230,39,88,215,33,206,64,175,
  83,48,185,40,168,75,114,159,11,216,80,142,161,241,27,93,
  251,44,211,193,172,46,85,125,250,19,43,150,105,208,74,23,
  132,12,183,250,92,213,240,128,67,44,166,130,243,47,149,84,
  142,222,72,27,181,82,191,146,23,179,120,107,185,140,239,119,
  225,137,84,255,123,188,245,60,97,126,37,9,213,106,149,50,
  222,135,121,75,246,32,230,156,52,176,213,94,0,170,46,189,
  89,214,74,31,137,61,83,7,227,107,29,146,212,72,204,250,
  21,198,153,240,136,44,3,118,74,253,11,231,76,50,22,153,
  105,167,206,4,149,58,20,208,233,150,189,246,60,34,203,185,
  19,85,58,9,155,92,183,4,143,232,105,60,255,222,117,230,
  179,47,149,202,157,111,200,174,152,201,239,79,183,118,1,132,
  103,39,95,118,54,227,171,92,213,126,65,196,162,91,213,11,
  240,22,67,229,101,85,175,136,72,15,196,96,132,172,81,112,
  159,240,203,221,128,69,207,109,69,37,152,197,132,29,159,68,
  110,39,237,107,18,47,218,33,95,52,5,181,61,28,235,169,
  67,179,237,207,22,107,197,234,41,165,196,3,128,252,178,70,
  194,122,42,146,201,9,220,121,34,116,46,161,22,236,221,0,
  145,97,171,41,163,244,16,174,227,123,21,182,50,84,12,245,
  139,7,215,81,181,254,140,73,104,249,127,134,100,156,236,89,
  205,5,136,30,171,140,71,181,29,85,110,65,150,24,115,56,
  146,175,218,75,160,242,56,157,246,170,227,213,97,125,57,47,
  245,67,22,111,46,146,86,199,55,245,91,213,145,101,181,207,
  95,189,132,62,170,13,118,232,22,196,159,215,15,202,46,120,
  28,158,74,222,84,255,13,51,128,243,180,228,46,223,201,87,
  246,20,95,132,33,116,95,186,4,84,146,36,9,145,178,190,
  133,207,80,190,221,23,124,39,140,164,12,72,234,222,124,56,
  162,41,225,152,101,51,209,158,180,64,43,86,255,69,142,103,
  227,249,108,174,44,124,223,104,153,212,23,82,182,94,141,6,
  107,215,51,230,176,14,215,47,130,235,61,188,204,85,226,96,
  29,165,116,255,149,71,234,189,80,212,123,40,165,11,28,80,
  254,14,114,30,236,201,36,83,133,4,234,111,23,162,218,3,
  182,57,101,149,0,196,163,76,187,8,115,61,162,18,232,71,
  183,158,8,190,72,245,141,71,206,25,122,105,254,154,54,12,
  221,144,37,6,90,207,177,0,31,102,248,144,190,92,237,196,
  135,181,87,209,140,14,116,247,96,213,145,201,185,63,125,82,
  154,14,207,239,62,94,235,31,204,99,146,246,201,128,170,38,
  251,105,133,120,88,166,111,161,95,182,165,19,67,32,116,174,
  109,61,231,185,53,131,106,159,229,197,58,25,128,63,111,153,
  23,220,47,168,74,174,190,63,32,168,53,123,89,238,40,244,
  171,70,34,133,181,24,117,48,165,69,220,34,90,49,213,79,
  27,56,210,232,46,20,219,4,243,57,228,148,200,135,217,245,
  76,205,167,102,22,246,46,66,129,86,182,225,159,209,176,45,
  100,66,132,247,108,47,231,155,136,245,77,9,156,20,103,198,
  117,225,140,100,231,151,219,137,252,11,127,181,222,9,118,148,
  130,174,3,72,149,202,58,119,41,133,79,49,232,89,18,185,
  147,15,130,81,223,192,140,210,11,147,51,14,75,33,252,0,
  232,207,154,24,7,143,86,12,105,224,32,188,228,203,140,49,
  21,191,91,170,13,55,79,106,181,46,115,76,156,103,239,192,
  88,225,116,242,180,128,237,191,206,101,218,0,115,168,53,98,
  31,49,251,148,114,35,176,81,247,173,111,236,107,193,125,140,
  88,166,49,189,214,131,255,200,29,188,100,128,57,168,79,235,
  99,56,210,42,244,197,175,0,232,150,207,191,23,64,182,17,
  74,29,144,52,97,27,87,68,156,175,120,64,184,136,204,232,
  122,214,191,25,69,232,4,100,202,39,212,183,139,54,73,190,
  16,117,73,238,100,65,176,87,55,172,145,255,29,113,5,176,
  147,255,13,132,67,121,31,125,67,89,9,244,144,217,41,205,
  244,164,189,214,16,138,165,51,9,19,195,236,28,219,17,66,
  160,138,59,88,167,155,121,52,132,17,68,126,9,233,157,210,
  242,60,137,181,52,156,38,129,236,119,0,68,219,90,192,221,
  39,117,155,187,95,216,167,250,197,172,50,100,120,88,168,135,
  120,4,76,65,201,254,175,228,110,243,93,149,49,103,125,90,
  245,4,106,233,200,38,217,190,236,159,84,207,175,43,26,98,
  35,147,228,3,118,212,18,220,95,174,50,199,133,245,44,133,
  64,112,206,23,147,194,45,107,25,137,215,34,177,223,28,89,
  236,171,227,126,99,42,82,210,128,64,198,73,163,253,208,27,
  174,189,36,131,253,32,81,108,165,28,247,77,146,106,252,132,
  198,114,21,248,92,189,144,110,40,229,211,25,108,150,21,164,
  239,51,81,242,65,6,86,179,76,238,111,187,67,11,191,58,
  107,44,22,167,148,7,118,153,28,165,42,117,178,7,61,153,
  94,227,55,75,156,176,1,71,228,49,127,192,0,220,82,167,
  67,181,76,168,45,230,71,186,10,156,78,122,183,60,84,192,
  3,213,174,35,219,132,236,149,43,192,6,129,249,149,206,144,
  215,68,203,241,85,233,181,64,96,210,236,18,217,116,190,46,
  135,118,203,12,105,124,239,141,192,18,104,227,61,113,194,49,
  12,125,204,98,137,27,127,247,55,146,99,243,36,231,204,99,
  112,139,156,100,113,165,204,15,220,62,157,92,50,106,80,21,
  255,157,131,54,36,204,42,253,3,188,135,73,91,140,234,76,
  249,22,167,242,177,62,190,34,121,165,153,43,183,25,158,235,
  225,142,244,41,222,61,174,82,111,201,196,1,169,50,140,250,
  35,61,234,5,191,46,72,121,109,137,205,233,29,224,130,175,
  89,11,124,99,186,106,144,158,79,104,228,44,161,30,186,9,
  199,64,147,84,221,41,160,96,248,70,205,84,242,136,103,27,
  84,57,5,156,107,203,7,213,165,21,66,223,128,111,18,159,
  177,47,192,77,255,89,26,177,241,79,25,116,167,189,58,40,
  168,44,219,175,11,214,25,233,125,148,10,203,254,57,99,131,
  174,52,104,37,129,20,236,77,7,133,226,22,129,73,200,173,
  119,190,79,185,91,253,146,42,236,47,88,154,81,231,203,67,
  91,217,133,14,157,138,233,180,33,217,49,68,216,7,152,226,
  74,193,242,78,145,64,85,190,37,57,179,109,130,213,159,221,
  84,240,198,229,94,206,123,175,216,49,95,161,193,7,255,44,
  134,241,211,17,137,52,103,75,119,183,255,18,187,44,100,26,
  245,139,104,174,204,60,113,12,167,123,153,250,137,96,206,116,
  136,20,30,205,124,252,170,75,213,245,92,22,90,6,71,40,
  22,150,177,0,169,75,138,27,156,188,36,146,58,105,222,63,
  16,150,46,69,235,185,18,228,140,190,104,167,217,58,172,132,
  188,58,35,85,225,37,147,72,193,92,1,176,46,84,23,243,
  128,85,107,161,44,99,29,119,9,148,207,236,143,185,245,208,
  114,80,60,136,46,242,197,55,107,243,117,233,178,82,159,185,
  99,203,119,174,146,58,169,202,25,56,6,81,122,139,230,208,
  3,116,250,199,10,96,247,212,50,237,219,111,193,161,56,176,
  37,215,236,176,4,202,224,185,107,129,69,44,167,106,56,176,
  143,252,26,219,90,112,33,232,82,9,60,201,23,30,124,235,
  9,83,255,104,11,214,116,92,152,73,224,242,10,37,75,48,
  84,161,172,72,133,191,173,111,21,134,36,60,244,13,221,189,
  5,145,63,80,139,243,48,152,80,226,16,95,230,31,127,5,
  191,94,161,204,181,17,154,208,127,167,226,140,85,240,142,49,
  216,164,42,199,32,78,240,39,242,141,100,178,205,184,151,98,
  244,212,15,107,151,23,85,65,180,153,77,169,148,114,98,78,
  163,223,192,20,114,67,166,10,39,250,174,201,70,155,235,88,
  44,233,28,134,80,239,159,95,51,23,155,59,42,210,187,72,
  119,26,143,90,223,179,137,10,182,197,129,45,64,250,109,196,
  21,125,60,234,46,222,233,7,206,255,90,191,25,213,136,247,
  106,117,48,248,184,89,221,128,196,54,117,139,19,208,64,168,
  215,69,110,51,127,4,65,197,226,184,97,251,119,160,6,100,
  245,188,57,242,153,52,160,93,67,23,30,170,89,130,9,168,
  73,144,181,205,75,118,171,140,106,35,118,14,236,52,66,34,
  141,13,88,160,38,206,24,113,228,95,166,0,244,110,86,129,
  17,175,158,241,174,218,141,29,73,134,12,210,27,152,84,224,
  37,174,129,0,114,18,211,124,219,254,112,211,14,232,52,220,
  39,251,97,0,172,98,52,66,191,219,69,199,104,161,179,199,
  240,211,225,63,142,242,152,70,175,28,210,52,178,222,37,255,
  143,209,10,88,210,47,103,254,118,233,86,191,73,228,197,133,
  21,152,86,218,74,242,199,37,83,165,143,192,81,148,118,199,
  80,160,119,34,213,245,12,239,27,139,162,147,75,251,3,87,
  50,26,174,121,2,92,48,205,5,245,128,85,67,149,23,187,
  76,119,34,160,95,14,156,178,55,165,43,158,117,54,94,59,
  251,203,47,183,167,37,133,157,10,61,228,100,35,246,182,102,
  15,186,233,58,134,186,155,106,177,80,5,232,41,92,155,128,
  181,71,97,198,115,225,185,122,63,144,108,191,227,121,169,54,
  104,222,63,242,131,202,71,12,206,24,138,221,1,178,166,25,
  113,103,142,232,62,103,93,225,180,70,134,0,211,63,17,139,
  239,47,107,148,77,38,90,210,62,253,100,119,212,21,207,114,
  37,219,146,255,43,19,166,97,221,24,201,43,10,98,209,12,
  235,172,137,190,52,230,157,216,92,105,242,75,99,249,214,139,
  233,70,10,28,129,205,14,250,109,200,32,160,120,194,97,152,
  70,209,12,200,236,18,225,129,48,196,33,186,135,170,63,249,
  160,193,14,53,137,215,152,34,248,82,177,153,255,58,197,131,
  44,85,35,2,104,76,25,127,250,61,172,188,133,21,61,41,
  190,162,219,156,243,76,177,22,56,150,221,240,83,174,218,37,
  128,166,86,176,121,101,169,0,140,159,237,71,49,242,99,0,
  89,120,77,176,199,86,67,121,191,127,58,116,33,142,81,242,
  152,189,115,205,250,178,146,41,180,151,11,44,90,158,207,78,
  125,50,88,115,58,194,142,123,209,90,100,43,143,16,46,255,
  114,185,29,248,42,64,195,243,79,112,8,152,89,23,201,141,
  46,215,231,25,111,247,19,45,232,12,95,197,221,181,1,111,
  221,70,232,151,41,83,102,232,11,77,215,234,45,179,112,238,
  16,198,252,176,0,102,43,240,59,168,6,191,64,211,107,196,
  2,77,220,59,137,160,208,35,56,176,207,218,122,165,226,65,
  21,243,156,131,29,203,189,150,212,166,73,235,53,125,161,53,
  35,12,94,64,166,192,25,198,127,168,103,119,147,241,5,152,
  101,140,67,27,232,201,71,183,145,22,248,108,229,130,150,174,
  59,150,116,197,10,115,186,89,124,227,24,100,8,252,109,185,
  144,94,55,74,173,140,58,101,127,29,114,150,12,90,235,190,
  145,180,203,129,7,218,115,240,58,70,251,30,189,79,40,216,
  56,226,161,116,132,157,111,13,213,123,201,157,35,12,90,229,
  44,241,159,97,236,70,22,252,141,49,135,192,75,60,35,203,
  170,11,191,223,110,1,228,83,43,250,180,63,175,203,70,136,
  249,74,108,245,47,73,158,84,34,198,173,11,221,96,161,135,
  190,7,85,241,18,54,252,173,97,47,77,180,55,248,116,26,
  194,99,30,19,181,173,135,52,206,162,64,241,179,158,133,81,
  31,116,254,83,182,21,243,159,200,9,221,136,240,8,103,41,
  25,222,1,183,135,251,173,17,125,150,101,46,138,59,215,19,
  102,203,42,174,211,87,35,150,230,24,136,92,218,185,71,214,
  133,167,222,123,66,215,91,16,225,80,108,32,120,8,239,217,
  232,133,61,34,142,206,97,118,67,147,85,95,37,154,202,116,
  170,57,159,92,33,61,107,215,187,238,85,231,156,120,251,85,
  233,148,119,95,153,197,109,69,206,163,239,4,139,163,48,109,
  9,77,52,252,147,36,242,163,118,1,213,232,140,194,90,69,
  37,187,158,218,76,51,194,32,231,176,49,129,188,78,255,140,
  88,236,209,125,195,157,225,22,56,3,205,110,31,197,5,153,
  31,72,218,7,25,240,146,8,188,40,107,196,32,91,254,149,
  182,226,99,191,6,106,77,182,58,148,170,72,17,53,171,105,
  151,198,9,96,251,124,148,11,109,168,25,214,230,13,53,181,
  18,190,52,16,229,40,83,119,142,68,174,35,186,79,48,178,
  117,190,247,59,183,80,221,97,124,79,250,54,118,220,20,66,
  199,35,117,160,177,220,143,29,195,252,41,93,190,247,202,13,
  120,77,235,115,20,172,65,223,252,87,194,70,109,178,131,223,
  35,111,73,137,175,101,195,159,252,101,214,124,251,132,214,236,
  90,17,166,135,105,152,51,18,219,178,149,71,175,210,129,98,
  242,134,64,23,80,43,231,96,122,24,214,131,150,112,81,226,
  28,86,153,53,220,195,137,68,40,131,158,0,149,44,92,76,
  243,147,164,251,59,9,230,38,200,47,156,11,59,102,17,162,
  131,48,83,212,32,239,178,202,60,20,227,7,139,39,74,173,
  0,45,188,248,212,131,9,204,69,161,106,231,6,62,43,180,
  254,61,207,164,88,4,100,215,179,58,232,101,249,210,196,157,
  0,207,91,24,215,155,75,129,18,103,190,90,226,196,105,32,
  203,223,148,192,10,87,114,251,145,100,193,110,184,243,53,219,
  197,145,226,105,164,55,158,171,248,46,84,32,185,239,134,159,
  8,194,127,35,68,246,122,13,152,24,205,86,29,171,18,120,
  64,220,42,132,194,112,89,226,177,71,231,29,171,152,49,255,
  69,26,102,243,68,136,29,85,33,134,42,241,15,92,153,114,
  68,129,88,31,21,114,225,37,110,143,210,151,127,87,210,109,
  225,112,177,233,139,167,192,233,87,176,116,188,135,47,95,247,
  140,172,84,242,7,56,170,36,247,139,120,206,38,124,86,163,
  140,179,121,43,165,201,154,233,210,161,74,210,147,62,230,24,
  14,209,57,179,203,76,191,87,5,183,223,59,9,202,43,74,
  145,84,21,100,48,27,113,56,42,242,72,9,222,148,110,196,
  15,32,106,184,226,148,205,124,0,57,158,79,13,232,192,218,
  4,57,203,84,226,125,49,6,185,116,21,87,168,127,201,136,
  254,165,235,132,144,249,49,135,237,21,126,95,249,172,128,27,
  208,50,210,186,239,73,186,224,161,141,94,241,60,177,223,78,
  232,159,58,130,74,22,98,188,86,208,98,245,134,64,28,98,
  111,241,156,32,14,249,94,62,99,225,193,251,35,101,49,184,
  62,100,45,4,87,27,160,98,193,75,168,33,110,67,187,235,
  163,117,4,130,152,208,12,86,124,1,196,161,17,129,41,25,
  185,118,209,252,27,166,241,65,230,21,194,180,114,216,152,204,
  39,173,132,67,186,108,220,169,25,135,62,155,1,225,78,162,
  34,153,195,118,184,219,58,216,122,27,247,153,217,15,99,35,
  141,254,66,168,43,103,138,253,32,208,53,115,203,93,143,164,
  97,47,8,191,96,107,47,137,153,40,131,49,0,146,80,124,
  49,99,227,214,163,46,141,200,227,40,87,121,184,203,241,119,
  85,218,248,69,104,240,39,174,154,53,200,72,191,162,239,86,
  215,20,190,98,219,14,206,164,75,185,234,85,24,255,215,65,
  246,214,139,81,168,226,196,11,106,171,227,92,241,34,227,180,
  249,14,107,6,60,83,131,31,79,186,239,112,67,50,142,21,
  174,12,130,55,169,12,129,91,1,230,103,134,11,43,132,59,
  179,110,51,143,245,65,113,56,138,101,128,148,66,170,0,86,
  126,164,108,17,59,240,128,76,248,208,62,118,162,190,103,14,
  66,166,196,142,248,182,14,251,118,155,13,217,166,15,95,236,
  63,187,38,209,152,197,111,255,68,185,27,90,172,229,194,15,
  235,155,213,120,31,160,87,195,223,23,10,226,184,140,119,192,
  19,36,233,180,157,36,31,181,61,14,157,26,57,83,133,212,
  154,222,123,35,110,230,103,170,66,205,85,22,255,189,153,203,
  120,85,249,97,19,74,51,215,141,163,240,207,39,115,147,79,
  37,102,8,74,224,184,5,240,40,162,246,76,95,39,229,53,
  170,200,68,96,212,134,111,207,98,135,221,178,217,248,10,114,
  49,82,30,74,172,51,212,24,148,49,103,143,116,37,74,229,
  17,111,159,215,125,236,180,101,38,117,60,136,73,248,29,134,
};

// st is the framebuffer pixel-center coordinate: Metal in.position.xy is
// top-left origin (pixel + 0.5), GL gl_FragCoord.xy is bottom-left origin, so
// the same physical pixel has GL row (H-1 - py_mt). GL samples the noise at
// gl_FragCoord.xy (row py_gl from the bottom) of the texture vtkOpenGLRenderWindow
// uploads, and that texture's rows are themselves flipped by vtkJPEGReader (its
// decode is bottom-up: texture row y == JPEG row 63-y). Composing both flips,
// the effective row (in JPEG top-down orientation) for a Metal top-down pixel
// py_mt is (py_mt - H) mod 64: GL samples texture row py_gl mod 64 which holds
// JPEG row (63 - py_gl) mod 64 = (py_mt - H) mod 64. (For H a multiple of 64,
// e.g. the 512x512 reference, this collapses to py_mt mod 64.) Verified against
// vtkJPEGReader's actual decode of the embedded BlueNoiseTexture64x64 bytes.
inline float sampleJitterNoise(float2 st, float viewportH, float blockSize) {
  // GL-parity block sampling (2026-08-19): GL samples the 64x64 tile at
  // gl_FragCoord.xy/64 with NEAREST, i.e. the jitter is constant over
  // texel-sized blocks (viewportH/64 px). The per-pixel form scatters SIMT
  // lanes (~+60% harness at 2048/SD4) while the GL-parity block form is
  // ~free (+4-5%) and reproduces GL's field exactly. Selected by the CPU
  // with nSize = 0 (VTK_METAL_TEST_JITTER_PARITY).
  if (blockSize < 0.5f)
  {
    // GL-parity block sampling (2026-08-19): GL samples the 64x64 tile at
    // gl_FragCoord.xy/64 with NEAREST, i.e. the jitter is constant over
    // texel-sized blocks (viewportH/64 px). The per-pixel form scatters SIMT
    // lanes (~+60% harness at 2048/SD4) while the GL-parity block form is
    // ~free (+4-5%) and reproduces GL's field exactly. Selected by the CPU
    // with nSize = 0 (VTK_METAL_TEST_JITTER_PARITY).
    float texel = viewportH / 64.0f;
    int2 t = int2(floor(st.x / texel), floor((st.y - viewportH) / texel)) & 63;
    return float(kBlueNoise64[t.y * 64 + t.x]) / 255.0f;
  }
  // Block-coherent sampling (2026-08-18): quantize the pixel coordinate to
  // the jitter block cell so adjacent pixels march in lockstep (per-pixel
  // jitter scatters warps and costs ~+57% on the raw path; block4 brings it
  // to GL's +27% level). blockSize 1 = legacy per-pixel (bit-identical).
  // NOTE: never divide by the raw blockSize — nSize 0 (GL-parity above)
  // would make this div-by-zero NaN and trap the shader under fast-math
  // speculation (the black-render bug, 2026-08-19).
  float bs = max(blockSize, 1.0f);
  st = floor(st / bs) * bs + 0.5f * bs;
  int2 t = int2(floor(st.x), floor(st.y - viewportH)) & 63;
  return float(kBlueNoise64[t.y * 64 + t.x]) / 255.0f;
}

// Interleaved Gradient Noise (Jimenez 2014), the jitter used before the GL
// blue-noise tile landed. Smooth, low-discrepancy, deterministic per pixel,
// no sin-hash streaks, no texture required. Selected per-render via
// volumeUniforms.useIGNJitter; the default stays on the GL-parity blue noise.
inline float sampleIGNJitter(float2 st, float blockSize) {
  // Never divide by the raw blockSize: nSize 0 (GL-parity) would make this
  // div-by-zero NaN (fast-math) and kill the march (2026-08-19).
  blockSize = max(blockSize, 1.0f);
  float2 blockCenter = floor(st / blockSize) * blockSize + 0.5f * blockSize;
  return fract(52.9829189 * fract(dot(blockCenter, float2(0.06711056, 0.00583715))));
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

// Honors vtkVolumeProperty::GetInterpolationType(): the OpenGL backend applies
// the property's interpolation to the volume data, transfer-function and
// gradient-opacity textures (vtkVolumeInputHelper), defaulting to nearest.
// The Metal backend previously hardcoded trilinear sampling (sVolume), which
// made a default-property volume render smoother than the GL reference.
// Interpolation is specialized at pipeline-creation time via the
// fc_linearInterpolation function constant (driven by the property), so the
// unused sampler path is eliminated — matching GL, which bakes interpolation
// into texture filter state with no per-sample branch.
//
// fc_marchVariant == 1 replaces the single hardware-linear fetch with a manual
// 8-tap trilinear (PERFORMANCE_INVESTIGATION.md section 9/10): co-compiling 8
// independent same-texture vol.sample calls makes the MSL backend overlap the
// fetches (the bare single-fetch loop serializes at ~0.42 ns/sample; the
// co-compiled form runs ~0.06 ns/sample). The taps use sNearest at the
// surrounding 8 texel corners with cell-to-point weights, which reproduces
// sVolume's hardware trilinear result (the repro verified identical readbacks).
inline float sampleVolumeLinear8Tap(texture3d<float> volTex, float3 pos) {
  const float3 vdims = float3(volTex.get_width(), volTex.get_height(), volTex.get_depth());
  const float3 vdims1 = vdims - 1.0;
  float3 tc = pos * vdims - 0.5;
  float3 f = floor(tc);
  float3 w = tc - f;
  float3 f0 = clamp(f, 0.0, vdims1) / vdims;
  float3 f1 = clamp(f + 1.0, 0.0, vdims1) / vdims;
  float v000 = volTex.sample(sNearest, float3(f0.x, f0.y, f0.z)).r;
  float v100 = volTex.sample(sNearest, float3(f1.x, f0.y, f0.z)).r;
  float v010 = volTex.sample(sNearest, float3(f0.x, f1.y, f0.z)).r;
  float v110 = volTex.sample(sNearest, float3(f1.x, f1.y, f0.z)).r;
  float v001 = volTex.sample(sNearest, float3(f0.x, f0.y, f1.z)).r;
  float v101 = volTex.sample(sNearest, float3(f1.x, f0.y, f1.z)).r;
  float v011 = volTex.sample(sNearest, float3(f0.x, f1.y, f1.z)).r;
  float v111 = volTex.sample(sNearest, float3(f1.x, f1.y, f1.z)).r;
  float v0 = mix(mix(v000, v100, w.x), mix(v010, v110, w.x), w.y);
  float v1 = mix(mix(v001, v101, w.x), mix(v011, v111, w.x), w.y);
  return mix(v0, v1, w.z);
}

// RG8 pair-packed slice fetch (fc_volRg8): texel z of the halved-depth RG8
// volume holds R=slice 2z, G=slice 2z+1. The trilinear z-blend over the
// unpacked layout becomes one XY-bilinear tap when floor(z) is even (both
// slices live in one texel) and two taps when odd; the z mix runs in
// registers. Algebraically identical to hardware trilinear, ±fp ulp.
inline float sampleVolumeScalarRG8Pair(texture3d<float> volTex, float3 pos) {
  float pairs = float(volTex.get_depth());
  float slices = 2.0f * pairs;
  float zf = saturate(pos.z) * slices - 0.5f;
  float k = floor(zf);
  float fz = zf - k;
  if (zf < 0.0f) { k = 0.0f; fz = 0.0f; } // hardware clamps the first slice
  int kp = int(clamp(k, 0.0f, slices - 1.0f));
  // Pair-center z: add the texel-unit nudge AFTER the division. At magnitude
  // ~pairs an fp32 ulp is ~6e-5 TEXEL units (pairs=897), so a nudge baked
  // into the numerator rounds away; a post-divide +1e-3/pairs shift is 18x
  // above ulp and guarantees floor() selects pair p. Cost: the tap blends
  // 0.1% of the NEXT pair's channels (<=0.25 LSB on 8-bit data).
  if ((kp & 1) == 0) {
    float zp = (float(kp >> 1) + 0.5f) / pairs + (1e-3f / pairs);
    float2 t = volTex.sample(sVolume, float3(pos.x, pos.y, zp), level(0)).rg;
    return mix(t.x, t.y, fz);
  }
  int pa = (kp - 1) >> 1;
  float za = (float(pa) + 0.5f) / pairs + (1e-3f / pairs);
  float2 ta = volTex.sample(sVolume, float3(pos.x, pos.y, za), level(0)).rg;
  if (kp + 1 <= int(slices) - 1) {
    float zb = (float(pa + 1) + 0.5f) / pairs + (1e-3f / pairs);
    float2 tb = volTex.sample(sVolume, float3(pos.x, pos.y, zb), level(0)).rg;
    return mix(ta.y, tb.x, fz);
  }
  // Last slice is odd: hardware clamp-to-edge repeats it for z+1.
  return ta.y;
}

inline float sampleVolumeScalar(texture3d<float> volTex, float3 pos) {
  if (fc_volTransposed) {
    // TRANSPOSE upload: the slice axis lives out of texture depth; original-
    // orientation (x,y,z) maps to texture (z,y,x) for X-depth or (x,z,y) for
    // Y-depth. The mv1 8-tap helper below is self-consistent in whatever
    // space it receives (its dims come from the live texture), so a single
    // entry-point swizzle covers every interpolation variant.
    pos = volumeFetchSwizzle(pos);
  }
  // Specialization §17: coarse SD4 4x stride with sNearest (1 texel vs 8) saves ~30% bandwidth for dense 41-step rays
  // Env-gated via fc_volumeNearestCoarse (VTK_METAL_TEST_VOLUME_NEAREST) to keep thr 0.000 by default; thr 2.02 <5 when enabled, -9% at 1024 SD4 6.42 vs 7.09
  if (fc_volumeNearestCoarse && fc_linearInterpolation && !fc_volRg8 && fc_marchVariant != 1 && fc_marchVariant != 2) {
    return volTex.sample(sNearest, pos, level(0)).r;
  }
  if (fc_linearInterpolation) {
    if (fc_marchVariant == 1) {
      return sampleVolumeLinear8Tap(volTex, pos);
    }
    if (fc_marchVariant == 2) {
      return volTex.sample(sVolumeClampZero, pos, level(0)).r;
    }
    if (fc_volRg8) {
      return sampleVolumeScalarRG8Pair(volTex, pos);
    }
    return volTex.sample(sVolume, pos, level(0)).r;
  }
  return volTex.sample(sNearest, pos, level(0)).r;
}

// Full-texel fetch honoring the pipeline's interpolation specialization. Used
// by the independent multi-component path, which needs every channel (one per
// component) at the same sample position.
inline float4 sampleVolumeTexel(texture3d<float> volTex, float3 pos) {
  if (fc_linearInterpolation) {
    if (fc_marchVariant == 1) {
      // Manual 8-tap texel: co-compiles the independent fetches and reproduces
      // the hardware trilinear result per channel (see sampleVolumeLinear8Tap).
      const float3 vdims = float3(volTex.get_width(), volTex.get_height(), volTex.get_depth());
      const float3 vdims1 = vdims - 1.0;
      float3 tc = pos * vdims - 0.5;
      float3 f = floor(tc);
      float3 w = tc - f;
      float3 f0 = clamp(f, 0.0, vdims1) / vdims;
      float3 f1 = clamp(f + 1.0, 0.0, vdims1) / vdims;
      float4 v000 = volTex.sample(sNearest, float3(f0.x, f0.y, f0.z));
      float4 v100 = volTex.sample(sNearest, float3(f1.x, f0.y, f0.z));
      float4 v010 = volTex.sample(sNearest, float3(f0.x, f1.y, f0.z));
      float4 v110 = volTex.sample(sNearest, float3(f1.x, f1.y, f0.z));
      float4 v001 = volTex.sample(sNearest, float3(f0.x, f0.y, f1.z));
      float4 v101 = volTex.sample(sNearest, float3(f1.x, f0.y, f1.z));
      float4 v011 = volTex.sample(sNearest, float3(f0.x, f1.y, f1.z));
      float4 v111 = volTex.sample(sNearest, float3(f1.x, f1.y, f1.z));
      float4 v0 = mix(mix(v000, v100, w.x), mix(v010, v110, w.x), w.y);
      float4 v1 = mix(mix(v001, v101, w.x), mix(v011, v111, w.x), w.y);
      return mix(v0, v1, w.z);
    }
    if (fc_marchVariant == 2) {
      return volTex.sample(sVolumeClampZero, pos, level(0));
    }
    return volTex.sample(sVolume, pos, level(0));
  }
  return volTex.sample(sNearest, pos, level(0));
}

inline half4 sampleTransferFunction(texture2d<float> tfTex, float2 uv) {
  if (fc_linearInterpolation) {
    return half4(tfTex.sample(sVolume, uv, level(0)));
  }
  return half4(tfTex.sample(sNearest, uv, level(0)));
}

// Per-component transfer-function lookup for the independent multi-component
// path (OpenGL computeColor/computeOpacity parity): selects the table of the
// requested component. All four tables are always uploaded for this path.
inline half4 sampleComponentTransferFunction(
    texture2d<float> tf0, texture2d<float> tf1,
    texture2d<float> tf2, texture2d<float> tf3,
    float2 uv, int c) {
  if (c == 0) return sampleTransferFunction(tf0, uv);
  if (c == 1) return sampleTransferFunction(tf1, uv);
  if (c == 2) return sampleTransferFunction(tf2, uv);
  return sampleTransferFunction(tf3, uv);
}

// 2D transfer function lookup at (primaryScalarNorm, secondScalarNorm).
inline half4 sampleTransferFunction2D(texture2d<float> tf2DTex, float2 uv) {
  if (fc_linearInterpolation) {
    return half4(tf2DTex.sample(sVolume, uv, level(0)));
  }
  return half4(tf2DTex.sample(sNearest, uv, level(0)));
}

// Fetch the Y-axis scalar array (e.g. "Temp") at the same normalized volume
// coordinate as the primary volume texture.
inline float sampleSecondScalar(texture3d<float> yAxisTex, float3 pos) {
  if (fc_linearInterpolation) {
    return yAxisTex.sample(sVolume, pos, level(0)).r;
  }
  return yAxisTex.sample(sNearest, pos, level(0)).r;
}

inline half sampleGradientOpacity(texture2d<float> gradTex, float value) {
  if (fc_linearInterpolation) {
    return half(gradTex.sample(sVolume, float2(value, 0.5), level(0)).r);
  }
  return half(gradTex.sample(sNearest, float2(value, 0.5), level(0)).r);
}

// Mirrors vtkVolumeTexture::ComputeCellToPointMatrix for point data: shifts
// [0,1] texture coordinates so samples land on texel centers, matching the
// OpenGL backend's ip_textureCoords convention.
// scale/offset are precomputed once per march (see marchVolumeUnified) so the
// per-sample conversion is a single fused multiply-add instead of 3 texture
// queries (get_width/get_height/get_depth) plus 3 divisions.
inline float3 cellToPointTextureCoord(float3 texCoord, float3 scale, float3 offset) {
  return texCoord * scale + offset;
}

// OpenGL ShadingSingleInput rectilinear parity (vtkVolumeShaderComposer.h): the
// proxy renders with uniform spacing over the bounds, but the actual per-axis
// coordinate curves are non-uniform. For each sample, find the containing cell
// (ijk, pCoords) in the axis's coordinate array and convert the recovered
// index-space position (ijk + pCoords) to a [0,1] texture coordinate by dividing
// by the per-axis point count — the same remap GL applies at every evaluate
// point, including the sign(in_cellSpacing) flip for descending grids.
// rectCoords holds float3 texels (x/y/z per axis, padded to the longest axis),
// each axis pre-scaled by its own GetScaleAndBias so the lookup happens in
// [0,1]-normalized coordinate space (OpenGL in_coordTexs parity).
//
// Ascending axes use a monotonic binary search (O(log n) vs the GL linear
// scan); the recovered cell is identical for strictly-increasing coordinate
// curves, including the GL edge cases: samples below coord[0] or above
// coord[n-1] map to 0 (the linear scan completes without a match, leaving
// ijk = pCoords = 0), and a sample exactly on coord[n-1] maps to (n-1)/n (the
// scan hits the s == xNext branch at the last index). Descending axes keep the
// exact GL linear-scan swap semantics (pathological, not exercised by tests).
inline float3 rectilinearSampleCoord(float3 dataPosWorld,
                                     constant packed_float3* rectCoords,
                                     constant VolumeMapperUniforms& u)
{
  int3 sizes = max(int3(u.rectCoordsSizes.xyz), int3(1));
  float3 scaled = dataPosWorld * u.rectCoordsScale.xyz + u.rectCoordsBias.xyz;
  float3 out = float3(0.0);
  for (int j = 0; j < 3; ++j) {
    int n = sizes[j];
    if (n <= 1) { continue; }
    float s = scaled[j];
    float xPrev = rectCoords[0][j];
    float xNext = rectCoords[n - 1][j];
    if (xNext < xPrev) {
      // Descending curve: exact OpenGL linear-scan parity (swap + sign flip).
      float tmp = xNext;
      xNext = xPrev;
      xPrev = tmp;
      int ijk = 0;
      float pCoords = 0.0;
      for (int i = 0; i < n; ++i) {
        xNext = rectCoords[i][j];
        if (s >= xPrev && s < xNext) {
          ijk = i - 1;
          pCoords = (s - xPrev) / max(xNext - xPrev, 1e-8);
          break;
        }
        if (s == xNext) {
          ijk = i - 1;
          pCoords = 1.0;
          break;
        }
        xPrev = xNext;
      }
      out[j] = -(float(ijk) + pCoords) / float(n);
      continue;
    }
    // Ascending: largest index with coord[lo] <= s.
    int lo = 0;
    int hi = n - 1;
    while (lo < hi) {
      int mid = (lo + hi + 1) >> 1;
      if (rectCoords[mid][j] <= s) { lo = mid; }
      else { hi = mid - 1; }
    }
    if (s < xPrev) { continue; }          // below range -> 0
    if (lo == n - 1) {                    // at/above the top coordinate
      if (s == xNext) { out[j] = float(n - 1) / float(n); }
      continue;                           // above range -> 0
    }
    float denom = max(rectCoords[lo + 1][j] - rectCoords[lo][j], 1e-8);
    float pCoords = (s - rectCoords[lo][j]) / denom;
    out[j] = (float(lo) + pCoords) / float(n);
  }
  return out;
}

// Converts a cellToPoint-shifted proxy [0,1] texture coordinate (Metal evalPoint
// == OpenGL g_dataPos) to the index-space texture coordinate used for the scalar
// fetch in rectilinear mode. The ray marches along the uniform-spacing proxy and
// only the scalar lookup is remapped through the real coordinate curves; the
// crop/mask/min-max/gradient tests keep using the proxy position, exactly like
// the OpenGL backend. Returns evalPoint unchanged when not in rectilinear mode.
inline float3 rectilinearSamplePosition(float3 evalPoint, bool doRectilinear,
                                        constant packed_float3* rectCoords,
                                        constant VolumeMapperUniforms& u)
{
  if (!doRectilinear) return evalPoint;
  float3 boundsSize = max(u.volumeBoundsMax.xyz - u.volumeBoundsMin.xyz, 1e-6);
  float3 dataPosWorld = u.volumeBoundsMin.xyz + evalPoint * boundsSize;
  return rectilinearSampleCoord(dataPosWorld, rectCoords, u);
}

// Gradient fetch with direction correction for anisotropic spacing and image-data
// direction. The raw central-difference gradient is in texel units along the
// texture (i/j/k) axes; dividing by gradStep gives a per-texture-coordinate-unit
// gradient, which is then transformed to model (data) space via the transpose of
// the volumeToTexture linear part. This handles both per-axis spacing and the
// direction matrix (matches the OpenGL backend's textureToEye * computeGradient()
// normal transform).
// Shared tail of computeGradientFast / densityGradientFast: transforms a
// texture-space gradient into model space, normalizes it, and packs the
// gradient-opacity magnitude. Computed in float to avoid half-precision
// overflow (mag > 65504) for full 3D gradients.
inline half4 normalizedGradient(float3 gradTex, float4x4 volumeToTexture, half gradNormFactor) {
  float3x3 texToModelLin =
    float3x3(volumeToTexture[0].xyz, volumeToTexture[1].xyz, volumeToTexture[2].xyz);
  float3 correctedGrad = transpose(texToModelLin) * gradTex;
  float mag = length(correctedGrad);
  half3 normal = mag > 0.0f ? half3(correctedGrad / mag) : half3(0.0h);
  return half4(normal, saturate(half(mag) / gradNormFactor));
}

// OpenGL computeDensityGradient parity (vtkVolumeShaderComposer.h
// ComputeDensityGradientDeclaration): when vtkVolumeMapper's
// ComputeNormalFromOpacity is set the shading normal is the gradient of the
// *opacity* field rather than the scalar field — six neighbor scalars are
// normalized (scalarScale/scalarBias) and mapped through the component's
// opacity transfer function, then the central difference of the opacities
// forms the gradient. The return shape matches computeGradientFast (xyz =
// normalized model-space normal, w = normalized magnitude); only the direction
// is consumed for lighting — gradient opacity keeps the scalar-gradient
// magnitude (OpenGL uses computeGradient for gradient opacity even when
// ComputeNormalFromOpacity is set).
inline half4 densityGradientFromNeighbors(
    half sPX, half sNX, half sPY, half sNY, half sPZ, half sNZ,
    texture2d<float> tf0, texture2d<float> tf1,
    texture2d<float> tf2, texture2d<float> tf3,
    float3 gradStep, float4x4 volumeToTexture,
    half gradNormFactor, int c, half scalarScale, half scalarBias) {
  half opPX = sampleComponentTransferFunction(tf0, tf1, tf2, tf3,
      float2(saturate(sPX * scalarScale + scalarBias), 0.5), c).a;
  half opNX = sampleComponentTransferFunction(tf0, tf1, tf2, tf3,
      float2(saturate(sNX * scalarScale + scalarBias), 0.5), c).a;
  half opPY = sampleComponentTransferFunction(tf0, tf1, tf2, tf3,
      float2(saturate(sPY * scalarScale + scalarBias), 0.5), c).a;
  half opNY = sampleComponentTransferFunction(tf0, tf1, tf2, tf3,
      float2(saturate(sNY * scalarScale + scalarBias), 0.5), c).a;
  half opPZ = sampleComponentTransferFunction(tf0, tf1, tf2, tf3,
      float2(saturate(sPZ * scalarScale + scalarBias), 0.5), c).a;
  half opNZ = sampleComponentTransferFunction(tf0, tf1, tf2, tf3,
      float2(saturate(sNZ * scalarScale + scalarBias), 0.5), c).a;

  half3 opGrad = half3(opPX - opNX, opPY - opNY, opPZ - opNZ);
  float3 gradTex = float3(opGrad) / max(gradStep, 1e-8);
  return normalizedGradient(gradTex, volumeToTexture, gradNormFactor);
}

inline half4 computeGradientFast(texture3d<float> volTex, float3 pos,
                                 float3 gradStep, float4x4 volumeToTexture, half gradNormFactor) {
  if (fc_gradNearest) {
    // 6*1 texel via sNearest vs 6*8 via sVolume: -10% win, thr 5.21 vs 2.93
    half sPX = half(volTex.sample(sNearest, volumeFetchSwizzle(pos + float3(gradStep.x, 0, 0)), level(0)).r);
    half sNX = half(volTex.sample(sNearest, volumeFetchSwizzle(pos - float3(gradStep.x, 0, 0)), level(0)).r);
    half sPY = half(volTex.sample(sNearest, volumeFetchSwizzle(pos + float3(0, gradStep.y, 0)), level(0)).r);
    half sNY = half(volTex.sample(sNearest, volumeFetchSwizzle(pos - float3(0, gradStep.y, 0)), level(0)).r);
    half sPZ = half(volTex.sample(sNearest, volumeFetchSwizzle(pos + float3(0, 0, gradStep.z)), level(0)).r);
    half sNZ = half(volTex.sample(sNearest, volumeFetchSwizzle(pos - float3(0, 0, gradStep.z)), level(0)).r);
    half3 rawGrad = half3(sPX - sNX, sPY - sNY, sPZ - sNZ);
    float3 gradTex = float3(rawGrad) / max(gradStep, 1e-8);
    return normalizedGradient(gradTex, volumeToTexture, gradNormFactor);
  }
  if (fc_grad4) {
    // 4-fetch forward sC+3 for fine SD: 33% save, -18% at SD0.5 thr 2.29 vs 0.689 PASS
    // Env VTK_METAL_TEST_GRAD4 forces it; fineSD-gated default disabled for 0-degradation bench
    float sC = sampleVolumeScalar(volTex, pos);
    float sPX = sampleVolumeScalar(volTex, pos + float3(gradStep.x, 0, 0));
    float sPY = sampleVolumeScalar(volTex, pos + float3(0, gradStep.y, 0));
    float sPZ = sampleVolumeScalar(volTex, pos + float3(0, 0, gradStep.z));
    half3 rawGrad = half3(half(sPX - sC), half(sPY - sC), half(sPZ - sC)) * 2.0h;
    float3 gradTex = float3(rawGrad) / max(gradStep, 1e-8);
    return normalizedGradient(gradTex, volumeToTexture, gradNormFactor);
  }
  if (fc_gradFloat) {
    float sPX = sampleVolumeScalar(volTex, pos + float3(gradStep.x, 0, 0));
    float sNX = sampleVolumeScalar(volTex, pos - float3(gradStep.x, 0, 0));
    float sPY = sampleVolumeScalar(volTex, pos + float3(0, gradStep.y, 0));
    float sNY = sampleVolumeScalar(volTex, pos - float3(0, gradStep.y, 0));
    float sPZ = sampleVolumeScalar(volTex, pos + float3(0, 0, gradStep.z));
    float sNZ = sampleVolumeScalar(volTex, pos - float3(0, 0, gradStep.z));
    half3 rawGrad = half3(half(sPX - sNX), half(sPY - sNY), half(sPZ - sNZ));
    float3 gradTex = float3(rawGrad) / max(gradStep, 1e-8);
    return normalizedGradient(gradTex, volumeToTexture, gradNormFactor);
  }
  if (fc_quadGrad) {
    // Quad-coop §13.5 prototype disabled: dfdx/quad_shuffle gave thr 69/26829 >>5 for NIFTI SD4/SD0.5, not thr0
    // Keep fallback to 6-fetch to keep thr 0 and compile clean; use VTK_METAL_TEST_QUAD_GRAD=1 for A/B but no win yet
    half sPX = half(sampleVolumeScalar(volTex, pos + float3(gradStep.x, 0, 0)));
    half sNX = half(sampleVolumeScalar(volTex, pos - float3(gradStep.x, 0, 0)));
    half sPY = half(sampleVolumeScalar(volTex, pos + float3(0, gradStep.y, 0)));
    half sNY = half(sampleVolumeScalar(volTex, pos - float3(0, gradStep.y, 0)));
    half sPZ = half(sampleVolumeScalar(volTex, pos + float3(0, 0, gradStep.z)));
    half sNZ = half(sampleVolumeScalar(volTex, pos - float3(0, 0, gradStep.z)));
    half3 rawGrad = half3(sPX - sNX, sPY - sNY, sPZ - sNZ);
    float3 gradTex = float3(rawGrad) / max(gradStep, 1e-8);
    return normalizedGradient(gradTex, volumeToTexture, gradNormFactor);
  }
  half sPX = half(sampleVolumeScalar(volTex, pos + float3(gradStep.x, 0, 0)));
  half sNX = half(sampleVolumeScalar(volTex, pos - float3(gradStep.x, 0, 0)));
  half sPY = half(sampleVolumeScalar(volTex, pos + float3(0, gradStep.y, 0)));
  half sNY = half(sampleVolumeScalar(volTex, pos - float3(0, gradStep.y, 0)));
  half sPZ = half(sampleVolumeScalar(volTex, pos + float3(0, 0, gradStep.z)));
  half sNZ = half(sampleVolumeScalar(volTex, pos - float3(0, 0, gradStep.z)));

  half3 rawGrad = half3(sPX - sNX, sPY - sNY, sPZ - sNZ);
  float3 gradTex = float3(rawGrad) / max(gradStep, 1e-8);
  return normalizedGradient(gradTex, volumeToTexture, gradNormFactor);
}

inline half4 computeDensityGradientFast(
    texture3d<float> volTex,
    texture2d<float> tf0, texture2d<float> tf1,
    texture2d<float> tf2, texture2d<float> tf3,
    float3 pos, float3 gradStep,
    float4x4 volumeToTexture,
    half gradNormFactor,
    int c, half scalarScale, half scalarBias) {
  half sPX = half(sampleVolumeTexel(volTex, pos + float3(gradStep.x, 0, 0))[c]);
  half sNX = half(sampleVolumeTexel(volTex, pos - float3(gradStep.x, 0, 0))[c]);
  half sPY = half(sampleVolumeTexel(volTex, pos + float3(0, gradStep.y, 0))[c]);
  half sNY = half(sampleVolumeTexel(volTex, pos - float3(0, gradStep.y, 0))[c]);
  half sPZ = half(sampleVolumeTexel(volTex, pos + float3(0, 0, gradStep.z))[c]);
  half sNZ = half(sampleVolumeTexel(volTex, pos - float3(0, 0, gradStep.z))[c]);
  return densityGradientFromNeighbors(sPX, sNX, sPY, sNY, sPZ, sNZ,
      tf0, tf1, tf2, tf3, gradStep, volumeToTexture, gradNormFactor, c, scalarScale, scalarBias);
}

// Scalar + opacity-field gradient from a single shared fetch of the six
// neighbors (dependent path only): gradient opacity needs the scalar-gradient
// magnitude while ComputeNormalFromOpacity needs the opacity-gradient
// direction, so computing both from the same six texels avoids a duplicated
// 6-fetch when both features are active. Returns the scalar gradient (for
// sharedGrad / gradient opacity) and writes the density gradient via
// densityGradOut (for the shading normal).
inline half4 computeScalarAndDensityGradient(
    texture3d<float> volTex,
    texture2d<float> tf0, texture2d<float> tf1,
    texture2d<float> tf2, texture2d<float> tf3,
    float3 pos, float3 gradStep,
    float4x4 volumeToTexture,
    half gradNormFactor,
    half scalarScale, half scalarBias,
    thread half4& densityGradOut) {
  half sPX = half(sampleVolumeScalar(volTex, pos + float3(gradStep.x, 0, 0)));
  half sNX = half(sampleVolumeScalar(volTex, pos - float3(gradStep.x, 0, 0)));
  half sPY = half(sampleVolumeScalar(volTex, pos + float3(0, gradStep.y, 0)));
  half sNY = half(sampleVolumeScalar(volTex, pos - float3(0, gradStep.y, 0)));
  half sPZ = half(sampleVolumeScalar(volTex, pos + float3(0, 0, gradStep.z)));
  half sNZ = half(sampleVolumeScalar(volTex, pos - float3(0, 0, gradStep.z)));

  densityGradOut = densityGradientFromNeighbors(sPX, sNX, sPY, sNY, sPZ, sNZ,
      tf0, tf1, tf2, tf3, gradStep, volumeToTexture, gradNormFactor, 0, scalarScale, scalarBias);

  half3 rawGrad = half3(sPX - sNX, sPY - sNY, sPZ - sNZ);
  float3 gradTex = float3(rawGrad) / max(gradStep, 1e-8);
  return normalizedGradient(gradTex, volumeToTexture, gradNormFactor);
}

// Central-difference gradient for every component at once (independent
// multi-component path). The volume texture stores one channel per component,
// so six RGBA texel fetches yield all components' gradients (OpenGL
// computeGradient(g_dataPos, c, ...) parity — each gradient is computed from
// its own component's channel). Only direction matters for the shading model;
// the magnitude is stored in .w for the gradient-opacity table.
inline void computeGradientsAllComponents(
    texture3d<float> volTex, float3 pos, float3 gradStep,
    float4x4 volumeToTexture, half gradNormFactor, thread half4 gradOut[4]) {
  float4 pX = sampleVolumeTexel(volTex, pos + float3(gradStep.x, 0, 0));
  float4 nX = sampleVolumeTexel(volTex, pos - float3(gradStep.x, 0, 0));
  float4 pY = sampleVolumeTexel(volTex, pos + float3(0, gradStep.y, 0));
  float4 nY = sampleVolumeTexel(volTex, pos - float3(0, gradStep.y, 0));
  float4 pZ = sampleVolumeTexel(volTex, pos + float3(0, 0, gradStep.z));
  float4 nZ = sampleVolumeTexel(volTex, pos - float3(0, 0, gradStep.z));

  float3x3 texToModelLin =
    float3x3(volumeToTexture[0].xyz, volumeToTexture[1].xyz, volumeToTexture[2].xyz);
  float3x3 texToModelT = transpose(texToModelLin);

  for (int c = 0; c < 4; ++c) {
    float3 gradTex = float3(pX[c] - nX[c], pY[c] - nY[c], pZ[c] - nZ[c]) / max(gradStep, 1e-8);
    float3 corrected = texToModelT * gradTex;
    float mag = length(corrected);
    half3 normal = mag > 0.0f ? half3(corrected / mag) : half3(0.0h);
    gradOut[c] = half4(normal, saturate(half(mag) / gradNormFactor));
  }
}

// Mirrors OpenGL's ComputeLightingDeclaration default-light path (headlight):
//   nDotL = dot(normal, -g_ldir)     with g_ldir = normalize(cameraPos - vertexPos)
//   r     = normalize(2*nDotL*normal + g_ldir)
//   vDotR = dot(r, -g_vdir)
//   specular = pow(vDotR, shininess) * in_specular * in_lightSpecularColor
inline half3 computePhongLightingVolumeFast(half3 sampleColor, half3 normal, half3 lightDir, half3 viewDir,
                                            half3 ambientMat, half3 diffuseMat, half3 specularMat, half shininess,
                                            bool twoSided = false) {
  half nDotL = dot(normal, -lightDir);
  // Reflection vector uses the un-negated nDotL (matches OpenGL's
  // ComputeLightingDeclaration, which computes r before the two-sided flip).
  // Guard on (nDotL > 0 || twoSided): for back-facing non-twoSided samples the
  // result is discarded, so skip the normalize/dot entirely.
  if (nDotL > 0.0h || twoSided) {
    half3 r = normalize(normal * (2.0h * nDotL) + lightDir);
    half vDotR = max(dot(r, -viewDir), 0.0h);
    if (nDotL < 0.0h && twoSided) {
      nDotL = -nDotL;
    }
    if (nDotL > 0.0h) {
      half3 diffuse = nDotL * diffuseMat * sampleColor;
      // Fast specular skip: pow(vDotR,20) < 1e-6 for vDotR<0.5 (0.5^20=9e-7),
      // negligible vs diffuse. Skip pow ALU for ~50% of shaded samples.
      half3 specular = half3(0.0h);
      if (vDotR > 0.5h) {
        specular = fast::pow(vDotR, shininess) * specularMat;
      }
      return ambientMat * sampleColor + diffuse + specular;
    }
  }
  return ambientMat * sampleColor;
}

// Full multi-light volume shading. Loops over all active lights, accumulating
// ambient + diffuse + specular contributions. Handles directional, positional,
// and spot lights with attenuation. Matches OpenGL's ComputeLightingDeclaration:
// the light direction is the direction the light *travels* (from the light
// toward the scene / fragment), consistent with in_lightDirection =
// normalize(focalPoint - position), and vDotR uses -viewDir (OpenGL's
// dot(-viewDirection, r)).
inline half3 computeVolumeLighting(
    half3 sampleColor,
    half3 normal,
    half3 viewDir,           // normalized, pointing toward camera (data space)
    half3 ambientMat,
    half3 diffuseMat,
    half3 specularMat,
    half shininess,
    constant VolumeLightUniforms& lightUniforms,
    float3 fragPosVolume)    // current sample position in data space
{
    half3 totalAmbient  = half3(0.0h);
    half3 totalDiffuse  = half3(0.0h);
    half3 totalSpecular = half3(0.0h);

    int numLights = fc_lightCount;
    bool twoSided = lightUniforms.twoSidedLighting != 0;

    for (int i = 0; i < numLights && i < MAX_LIGHTS; ++i) {
        constant VolumeLight& L = lightUniforms.lights[i];

        half3 lightAmbient  = half3(L.ambientColor.rgb);
        half3 lightDiffuse  = half3(L.diffuseColor.rgb);
        half3 lightSpecular = half3(L.specularColor.rgb);

        half3 toLight;
        half attenuation = 1.0h;

        if (L.position.w < 0.5) {
            // Directional light: direction is pre-normalized in data space and
            // points along the light's travel direction (OpenGL
            // in_lightDirection parity).
            toLight = half3(L.direction.xyz);
        } else {
            // Positional light: compute the direction the light travels
            // (light -> fragment, matching OpenGL's
            // normalize(fragWorldPos - lightPosition)).
            half3 lightPos = half3(L.position.xyz);
            half3 delta = half3(fragPosVolume) - lightPos;
            half dist = length(delta);
            toLight = dist > 0.0001h ? delta / dist : half3(0.0h, 0.0h, 1.0h);

            // Attenuation: 1 / (constant + linear*d + quadratic*d^2)
            half attenDenom = half(L.attenuation.x)
                            + half(L.attenuation.y) * dist
                            + half(L.attenuation.z) * dist * dist;
            attenuation = attenDenom > 0.0h ? 1.0h / attenDenom : 0.0h;

            // Spot light cone check: the cone axis is L.direction (the light's
            // travel direction); a fragment is inside the cone when the
            // light->fragment direction is within the cone angle of the axis
            // (OpenGL coneDot = dot(vertLightDirection, lightDir)).
            if (L.direction.w <= 90.0) {
                half spotCos = dot(toLight, half3(normalize(L.direction.xyz)));
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
            half vDotR = dot(-viewDir, r);
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
    float4 wn = u.ndcToVolume * float4(ndc.x, -ndc.y, 0.0, 1.0); wn.xyz /= wn.w;
    float4 wf = u.ndcToVolume * float4(ndc.x, -ndc.y, 1.0, 1.0); wf.xyz /= wf.w;
    return normalize(wf.xyz - wn.xyz);
}

// The pixel's position on the near (view) plane in [0,1] normalized volume
// space. Used as the ray origin for parallel projection on the fullscreen
// (camera-inside / grid-traversal) paths, matching OpenGL's g_rayOrigin for
// parallel cameras.
inline float3 parallelRayOrigin(float2 screenPos, float2 viewportSize,
    constant VolumeMapperUniforms& u)
{
    float2 ndc = (screenPos / viewportSize) * 2.0 - 1.0;
    float4 wn = u.ndcToVolume * float4(ndc.x, -ndc.y, 0.0, 1.0);
    return wn.xyz / wn.w;
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

    // Camera-inside near-plane clip (OpenGL parity): OpenGL clips the proxy box
    // against the camera near plane (pushed in by a precision offset) and starts
    // the march there, so the eye->near-plane slab is never sampled. The fullscreen
    // and proxy paths reconstruct the ray from the eye, so clamp the entry to the
    // same plane to reproduce the OpenGL sample comb. The plane intersection
    // distance varies per ray (off-axis rays meet the plane further out), so the
    // plane is passed as origin+normal rather than a single scalar.
    // Specialization §17: fc_useCameraInside dead-strips this block for outside views (NIFTI SD4 30% fixed overhead)
    if (fc_useCameraInside && volumeUniforms.useCameraInsideNearClip > 0.5) {
        float denom = dot(rayDir, volumeUniforms.cameraInsideNearPlaneNormal.xyz);
        if (abs(denom) > 1e-6) {
            float tNear = dot(volumeUniforms.cameraInsideNearPlaneOrigin.xyz - cameraPos,
                volumeUniforms.cameraInsideNearPlaneNormal.xyz) / denom;
            tStart = max(tStart, tNear);
            if (tStart >= t.y) return s;
        }
    }

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
        // totalBoxT must reflect the clipped entry/exit: it is the march length
        // passed to marchVolume as p.tEnd. Ray dir is unit length, so the chord
        // length equals the traversal time along rayDir.
        s.totalBoxT = s.totalDist;
    }

    s.tTerminateMax = 1e30;
    // Specialization §17: fc_useDepthTexture dead-strips depth sample for NIFTI w/o depth (saves matrix*vec + sample per fragment)
    if (fc_useDepthTexture && volumeUniforms.useDepthTexture > 0.5) {
        float depthSample = depthTexture.sample(sNearest, screenPos / viewportSize).r;
        if (depthSample < 1.0) {
            float2 ndcXY = (screenPos / viewportSize) * 2.0 - 1.0;
            float4 worldTermination = volumeUniforms.ndcToVolume * float4(ndcXY.x, -ndcXY.y, depthSample, 1.0);
            float3 terminationLocal = worldTermination.xyz / worldTermination.w;
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
    half3 initialColor, half initialOpacity, half4 slabFar,
    constant VolumeMapperUniforms& volumeUniforms,
    constant PerBlockData& b,
    texture3d<float> volumeTexture,
    texture2d<float> transferFunctionTexture,
    texture2d<float> transferFunctionTexture1,
    texture2d<float> transferFunctionTexture2,
    texture2d<float> transferFunctionTexture3,
    texture2d<float> transferFunction2DTexture,
    texture3d<float> transfer2DYAxisTexture,
    texture2d<float> gradientOpacityTexture,
    texture3d<float> maskTexture,
    texture2d<float> labelMapTransferTexture,
    texture3d<float> minMaxTexture,
    texture3d<float> minMaxBlockTexture,
    texture3d<float> minMaxSuperTexture,
    texture3d<float> normalTexture,
    texture3d<float> blankingTexture,
    constant packed_float3* rectCoords,
    constant VolumeLightUniforms* lightUniforms,
    thread float3* firstOpaquePos,
    thread bool*  haveOpaquePos,
    device uint* segIndexMap,
    device uint* segPool)
{
  // Feature flags are baked into the pipeline via function constants (see
  // VolumeShaderFeatureFlags in the mapper): each flag below is set iff the
  // corresponding runtime uniform was on at pipeline-build time, so the
  // redundant `uniform > 0.5` re-checks are dropped and the compiler sees pure
  // compile-time booleans in the hot loop.
  const bool doShading = fc_shading;
  const bool doGradOp = fc_gradientOpacity;
  const bool doCropping = fc_cropping;
  const bool doMask = fc_mask;
  const bool doBlanking = fc_blanking;
  const bool doTransfer2D = fc_transfer2D;
  const bool doRectilinear = fc_rectilinear;

  half scalarScale = half(1.0 / max((volumeUniforms.scalarMax - volumeUniforms.scalarMin), 1e-4h));
  half scalarBias  = half(-volumeUniforms.scalarMin) * scalarScale;

  half secondScale = half(volumeUniforms.transfer2DYAxisScale);
  half secondBias  = half(volumeUniforms.transfer2DYAxisBias);

  half gradNormFactor = half(max(1e-8f, volumeUniforms.gradientOpacityRange.y));

  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz
                        - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  // Advance of the sample position in texture space per ray step (constant),
  // including the image-data direction matrix: the ray direction/step are in
  // normalized volume space and are converted to [0,1] texture coords via
  // volumeToTexture (OpenGL TextureToDataset parity).
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(p.rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * p.stepSize;
  // Cell-to-point conversion factors, computed once (texel centers at (i+0.5)/dims).
  // RG8 pair-pack (fc_volRg8): the texture's z extent is the PAIR count
  // (original slices / 2), so restore the original slice count here — every
  // CTP-adjusted z phase and step below is expressed over it.
  // TRANSPOSE (fc_volTransposed): texture extents hold (D,H,W) for X-depth or
  // (W,D,H) for Y-depth; un-swizzle to the ORIGINAL dims so every CTP factor
  // below keeps its data-space meaning.
  float texelCountZ = volumeTexture.get_depth() *
    ((fc_volRg8 && !fc_volTransposed) ? 2.0f : 1.0f);
  float3 texelCount = fc_volTransposedY
    ? float3(volumeTexture.get_width(), volumeTexture.get_depth(),
             volumeTexture.get_height())
    : fc_volTransposed
      ? float3(volumeTexture.get_depth(), volumeTexture.get_height(),
               volumeTexture.get_width())
      : float3(volumeTexture.get_width(), volumeTexture.get_height(), texelCountZ);
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  // Lighting directions must live in the same (physical/data) space as the
  // gradient normal: the normal is expressed per world-unit (the gradient is
  // scaled by the direction/spacing transform), so the headlight direction must
  // be converted from the normalized volume frame back to data space (offset *
  // boundsSize) before computing nDotL/vDotR. OpenGL computes g_ldir/g_vdir
  // directly in object space (normalize(eyePosObj - vertexPosObj)); using the
  // distorted volume frame here would bias nDotL for anisotropic bounds and fire
  // specular on surfaces where OpenGL's nDotL is <= 0.
  float3 entryVolPos = p.rayOrigin + p.rayDir * p.tStart;
  half3 viewDirHalf  = half3(normalize((entryVolPos - volumeUniforms.cameraVolumePos.xyz) * boundsSize));
  half3 ambientMat   = half3(volumeUniforms.ambientColor.rgb);
  half3 diffuseMat   = half3(volumeUniforms.diffuseColor.rgb);
  half3 specularMat  = half3(volumeUniforms.specularColor.rgb);
  half shininessMat  = half(volumeUniforms.shininess);

  float maskScale = volumeUniforms.maskScale;
  float maskBias  = volumeUniforms.maskBias;
  float numLabels = volumeUniforms.labelMapNumLabels;

  // Independent multi-component path (OpenGL independent components parity):
  // each component is normalized against its own scalar range and looked up in
  // its own color/opacity table, then results are combined via component
  // weights. Baked into the pipeline via fc_independentComponents (set by the
  // mapper only when the volume actually uses independent components and the
  // 2D transfer-function / label-map fallbacks are inactive), so single-
  // component pipelines eliminate every branch below at compile time.
  const bool useIndependentPath = fc_independentComponents;

  // Blanking: the blanking texture has the same (point) dimensions as the
  // global volume, so the half-cell-step offset in normalized texel space is
  // computed from its own dimensions (which equal the volume dims in the
  // single-block path; grid-traversal bricks map the full [0,1] range so the
  // global blanking dims remain correct). Sampling at +-half step catches the
  // neighboring point/cell flags, mirroring the OpenGL backend's
  // in_cellStep/2.0 offsets.
  float3 blankHalfStep = 0.5f / float3(blankingTexture.get_width(),
                                        blankingTexture.get_height(),
                                        blankingTexture.get_depth());

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

  // Non-divergent march loop bounds:
  // - variant 4: uniform frame-max bound (all exits latched, no divergence).
  // - variant 5: hybrid — uniform main loop of mainSteps (frame-average chord)
  //   with all exits latched, then a divergent tail loop up to the per-fragment
  //   bound. The per-fragment bound is floored at mainSteps so short rays keep
  //   SIMT lanes locked through the uniform phase.
  int maxSteps = max(1, int(ceil((p.tEnd - firstT) / p.stepSize)));
  int mainSteps = 0;
  if (fc_marchVariant == 4 && volumeUniforms.maxStepsFrame > 0.5)
  {
    maxSteps = int(volumeUniforms.maxStepsFrame);
  }
  else if (fc_marchVariant == 5 && volumeUniforms.maxStepsFrame > 0.5)
  {
    mainSteps = int(volumeUniforms.maxStepsFrame);
    maxSteps = max(maxSteps, mainSteps);
  }
  else if (volumeUniforms.maxStepsFrame > 0.5)
  {
    // Fixed-steps probe (VTK_METAL_TEST_MARCH_STEPS, any march variant): cap
    // the baseline divergent march at a uniform iteration count so j0/j1 cost
    // decomposes into envelope vs per-sample. Only nonzero when the probe env
    // var is set (variants 0/6-9 never set maxStepsFrame in production).
    maxSteps = min(maxSteps, int(volumeUniforms.maxStepsFrame));
  }

  // Composite slab tiling (fc_slabMode, VTK_METAL_TEST_NUM_SLABS > 1): the
  // ray's sample indices [0, maxSteps) are partitioned into slabCount equal
  // index ranges via shared ceilings, so consecutive passes start on the same
  // ceil() of the same lattice and the slab sample sets tile the full ray
  // exactly (minimal_gap phase-2 kEndT). Each pass composites only its interval;
  // the mapper renders the passes in RAY order (front-to-back) into two
  // alternating private textures, each pass sampling the other as its NEAR-side
  // composite (slabFar, premultiplied RGBA). Starting the opacity accumulator
  // at the near-side alpha makes the per-sample weights and the saturation
  // latch track the single-pass march's global accumulation exactly, and the
  // over-chain's associativity makes the combined result equal to a single-pass
  // composite up to fp rounding. currentT is an index-offset accumulator, so
  // re-anchoring to the slab's first sample keeps comparisons and the tEnd
  // break consistent. Dead-code-eliminated when the flag is clear. Variants 4/5
  // (uniform frame-max bound) are excluded: their bound is frame-uniform, not
  // per-fragment.
  //
  // Spatial mode (slabInfo.w == 1.0, VTK_METAL_TEST_SLAB_SPATIAL=1,
  // SLAB_BENCHMARKS.md §5.2): the passes instead split the ray by uniform
  // planes perpendicular to the dominant view axis (slabInfo.z = 0/1/2 for
  // x/y/z, set per frame on the CPU), at normalized-volume positions idx/K
  // and (idx+1)/K. All rays in a pass then fetch from the same thin flat band
  // of the volume, which keeps the working set cache-resident at any RT size
  // (the ray-fraction split's per-pass fetch set is a ray-space wedge that
  // still spans the whole volume at small RTs). The index lattice stays
  // contiguous across passes: adjacent passes share the exact same plane value
  // and ceil arithmetic (kEnd of pass p == kStart of pass p+1), and the
  // firstT/tEnd clamps only cut empty prefixes/suffixes, so the union still
  // tiles [0, maxSteps) exactly and the composite stays bit-identical to
  // single-pass up to fp rounding.
  if (fc_slabMode && fc_marchVariant != 4 && fc_marchVariant != 5)
  {
    const float K = max(b.slabInfo.y, 1.0f);
    const float idx = clamp(b.slabInfo.x, 0.0f, K - 1.0f);
    // The first pass drawn this frame has no near side: its feedback texture
    // was not written by any earlier pass, so ignore the stale content.
    // Ray-fraction passes (spatial off) are drawn front-to-back starting at
    // idx 0; spatial passes start at idx 0 when the rays travel the +dominant
    // axis and at idx K-1 when they travel the -axis (the mapper's draw order
    // follows the camera side).
    const int slabAxis = (int)(b.slabInfo.z + 0.5f);
    const float slabAd = (slabAxis == 0) ? p.rayDir.x
                       : (slabAxis == 1) ? p.rayDir.y
                                         : p.rayDir.z;
    slabFar = ((b.slabInfo.w <= 0.5f && idx <= 0.0f) ||
               (b.slabInfo.w > 0.5f && slabAd > 0.0f && idx <= 0.0f) ||
               (b.slabInfo.w > 0.5f && slabAd < 0.0f && idx >= K - 1.0f))
        ? half4(0.0h) : slabFar;
    int kStart = 0;
    int kEnd = 0;
    float z0 = 0.0f;
    float z1 = 0.0f;
    if (b.slabInfo.w > 0.5f)
    {
      const int axis = slabAxis;
      const float a0 = (axis == 0) ? p.rayOrigin.x
                    : (axis == 1) ? p.rayOrigin.y
                                  : p.rayOrigin.z;
      const float ad = slabAd;
      // The march's t is relative to the ray entry (checkBounds: firstT =
      // jitter, tEnd = totalBoxT), so the absolute plane intersections are
      // shifted by the camera-to-entry distance p.tStart. The crossing of a
      // plane q that the ray never reaches must not use the analytic
      // intersection: grazing rays that enter/exit through the band's own face
      // keep their clamped surface samples (which sit exactly on the plane) in
      // the band, so a plane at or beyond the entry side maps to the entry
      // (t=0) and a plane at or beyond the exit side maps to tEnd.
      const float z0Local = a0 + ad * p.tStart;
      const float z1Local = a0 + ad * (p.tStart + p.tEnd);
      z0 = z0Local;
      z1 = z1Local;
      const float qA = idx / K;
      const float qB = (idx + 1.0f) / K;
      float cA, cB;
      if (ad >= 0.0f)
      {
        cA = (qA <= z0) ? 0.0f : ((qA >= z1) ? p.tEnd : (qA - z0) / ad);
        cB = (qB <= z0) ? 0.0f : ((qB >= z1) ? p.tEnd : (qB - z0) / ad);
      }
      else
      {
        cA = (qA >= z0) ? 0.0f : ((qA <= z1) ? p.tEnd : (qA - z0) / ad);
        cB = (qB >= z0) ? 0.0f : ((qB <= z1) ? p.tEnd : (qB - z0) / ad);
      }
      const float tlo = max(firstT, min(cA, cB));
      const float thi = min(p.tEnd, max(cA, cB));
      kStart = max(int(ceil((tlo - firstT) / p.stepSize)), 0);
      kEnd = max(int(ceil((thi - firstT) / p.stepSize)), 0);
      // The march guarantees at least one sample (the entry-clamped sample,
      // max(1, ceil(...))). For grazing rays (chord shorter than the first
      // lattice sample, p.tEnd <= firstT) every band's window ends before the
      // first lattice sample, so the entry band — the only band whose plane
      // range contains the entry point — must keep that sample. For longer
      // rays the next band's window starts at the shared plane value (kEnd of
      // pass p == kStart of pass p+1), so it already covers index 0; forcing
      // it here too would double-composite the sample.
      const float zEntryClamped = clamp(z0, 0.0f, 1.0f);
      if (p.tEnd <= firstT && kEnd == 0 &&
          zEntryClamped >= qA &&
          (zEntryClamped < qB || qB == 1.0f))
      {
        kEnd = 1;
      }
    }
    else
    {
      kStart = int(ceil(idx * float(maxSteps) / K));
      kEnd = int(ceil((idx + 1.0f) * float(maxSteps) / K));
    }
    firstT += float(kStart) * p.stepSize;
    currentT = firstT;
    currentPoint = p.rayOrigin + p.rayDir * (p.checkBounds ? p.tStart : 0.0)
                 + p.rayDir * firstT;
    // The full-ray sample count before this pass's band range is applied.
    const int maxStepsFull = maxSteps;
    maxSteps = max(kEnd - kStart, 0);

    // Spatial slab debug (VTK_METAL_TEST_SLAB_SPATIAL=2): encode the pass'
    // kStart/kEnd as fractions of the full ray (0-255) and the pass maxSteps
    // scaled by 0.1 (raw values beyond the 8-bit range stay readable).
    if (b.slabInfo.w > 1.5f && b.slabInfo.w < 3.0f)
    {
      const float invFull = (maxStepsFull > 0) ? 255.0f / float(maxStepsFull) : 0.0f;
      return half4(float(kStart) * invFull, float(kEnd) * invFull,
        float(maxSteps) * 0.1f, 1.0h);
    }
    // Spatial slab debug (VTK_METAL_TEST_SLAB_SPATIAL=0.6): ray geometry:
    // tEnd*255/4096 (traversal time), stepSize*255/1.0, firstT/step scaled
    // by 255/8192 (the band's first sample index).
    if (b.slabInfo.w > 0.5f && b.slabInfo.w < 0.75f)
    {
      return half4(p.tEnd * (255.0f / 4096.0f), p.stepSize * 255.0f,
        (firstT / p.stepSize) * (255.0f / 8192.0f), 1.0h);
    }
    // Spatial slab debug (VTK_METAL_TEST_SLAB_SPATIAL=0.8): volume bounds and
    // camera: (bsz.z*255/2000, cameraPos.z*255/8, volumeBoundsMin.z*255/1000).
    if (b.slabInfo.w > 0.75f && b.slabInfo.w < 0.9f)
    {
      const float bsz = max(volumeUniforms.volumeBoundsMax.z - volumeUniforms.volumeBoundsMin.z, 1e-6);
      return half4(bsz * (255.0f / 5000.0f),
        volumeUniforms.cameraVolumePos.z * (255.0f / 4000.0f),
        volumeUniforms.volumeBoundsMin.z * (255.0f / 5000.0f), 1.0h);
    }
    // Spatial slab debug (any value in (0.75, 1.5) except 1.0): raw
    // kStart/kEnd/maxStepsFull scaled by 255/8192 (values up to 8192 stay
    // readable). w == 1.0 is the clean spatial mode and falls through to the
    // march below.
    if (b.slabInfo.w > 0.75f && b.slabInfo.w < 1.5f && b.slabInfo.w != 1.0f)
    {
      return half4(float(maxStepsFull) * (255.0f / 8192.0f),
        float(kEnd) * (255.0f / 8192.0f), float(kStart) * (255.0f / 8192.0f), 1.0h);
    }
    if (b.slabInfo.w < 0.0f)
    {
      return half4(float(kStart), float(kEnd), float(maxSteps), 1.0h);
    }
    if (b.slabInfo.w > 3.5f)
    {
      return half4(z0 * 255.0f, z1 * 255.0f,
        (firstT / p.stepSize) * 255.0f, 1.0h);
    }
    if (b.slabInfo.w > 3.0f && b.slabInfo.w < 3.5f)
    {
      return half4(slabFar.a * 255.0f, slabFar.g * 255.0f, slabFar.b * 255.0f, 1.0h);
    }
  }

  half3 accumulatedColor = initialColor;
  // §38 opacity-saturation exit threshold: the legacy 8-bit latch value, or
  // the uniform-supplied TF-adaptive value when fc_exitTheta specializes in.
  // Hoisted once; every exit site below compares against this single const.
  half kExitAcc =
      fc_exitTheta ? half(volumeUniforms.exitAlpha) : (1.0h - 1.0h / 255.0h);
  // Slab passes (fc_slabMode) inherit the NEAR-side composite (slabFar, the
  // premultiplied RGBA of the ray-order-earlier passes, sampled by
  // fragment_volume_main from the ping-pong feedback texture) so the
  // opacity-saturation latch tracks the GLOBAL accumulation exactly like the
  // single-pass march: a single pass latches after the ray-earlier samples have
  // accumulated past the threshold, which for a slab pass is exactly the
  // inherited near-side alpha plus the pass's own accumulation. The per-sample
  // color weights then match the true over-chain (each sample attenuated by the
  // ray-earlier samples: near-side alpha first, then the pass's own earlier
  // samples). The first pass drawn has no near side; its feedback texture
  // content is stale (it is never read as feedback again within the frame).
  half accumulatedOpacity =
      initialOpacity + (fc_slabMode ? slabFar.a * (1.0h - initialOpacity) : 0.0h);
  // Note: accumulatedOpacity is the GLOBAL over-chain alpha (the near-side
  // slabFar combined with the pass's own samples), so the final composite
  // alpha below is exactly accumulatedOpacity — no separate "own alpha"
  // accumulator is needed (the over-chain alpha is symmetric).

  // Non-composite blend-mode accumulators. Only the active mode's accumulator
  // is ever touched; dead branches are eliminated via the fc_blendMode function
  // constant so composite pipelines carry no extra cost.
  half mipMaxScalar = 0.0h;    // MIP: max normalized scalar along the ray
  half minipMinScalar = 1.0h;  // MinIP: min normalized scalar along the ray
  half avgBlendSum = 0.0h;     // AverageIP: sum(opacity * scalar) over in-range samples
  int avgBlendCount = 0;       // AverageIP: number of in-range samples
  half additiveSum = 0.0h;     // Additive: sum(opacity * scalar)
  bool firstBlendSample = true;

  // Debug iter counter (enabled by _padCropFlags[0] > 0.5 uniform flag).
  int marchIter = 0;

  // TEMP-DIAG minmax walk probe (MM_PROBE env -> _padCropFlags[4]): counts
  // loop iterations walked, lattice cell crossings (= R8 fetches) and sample
  // steps consumed by empty-cell jumps in the baseline march. With METAL_ITER
  // set, the debug exit returns them instead of marchIter
  // (R = visits/64, G = crossings/16, B = skippedSteps/64).
  int mmVisits = 0;
  int mmCross  = 0;
  int mmSkipped = 0;

  // Per-component accumulators for the independent multi-component path.
  // Only the active blend mode's arrays are touched.
  half mipMaxScalarComp[4] = {0.0h, 0.0h, 0.0h, 0.0h};
  half minipMinScalarComp[4] = {1.0h, 1.0h, 1.0h, 1.0h};
  half avgBlendSumComp[4] = {0.0h, 0.0h, 0.0h, 0.0h};
  int avgBlendCountComp[4] = {0, 0, 0, 0};
  half additiveSumComp[4] = {0.0h, 0.0h, 0.0h, 0.0h};

  // Per-component gradients (independent path only): computed lazily at most
  // once per sample from a single six-texel batch and reused by the
  // gradient-opacity step and the per-component shading in the composite loop.
  half4 compGrad[4] = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)};
  bool compGradReady = false;

  // Sample position carried incrementally through the march: advance one ray
  // step per iteration instead of recomputing. texLocalPos lives in [0,1]
  // texture space; currentPoint is in normalized volume space (the AABB).
  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize, 1.0)).xyz;
  float3 evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
  // NOPREFETCH probe (_padCropFlags[2], env VTK_METAL_TEST_NOPREFETCH): the
  // OpenGL loop issues ONE volume fetch per iteration and consumes it
  // immediately; this march additionally keeps sample i+1 in flight. Under a
  // per-pixel-jittered field that doubles the distinct cache lines each warp
  // holds, which measurably amplifies the phase-scatter tax. When set, drop
  // the prefetch-ahead pipeline and fetch directly per iteration (GL fetch
  // discipline). Default 0 keeps legacy behavior.
  float prefetchScalar = 0.0;
  bool prefetchValid = false;
  if (volumeUniforms._padCropFlags[2] < 0.5f)
  {
    prefetchScalar = sampleVolumeScalar(volumeTexture,
        rectilinearSamplePosition(evalPoint, doRectilinear, rectCoords, volumeUniforms));
    prefetchValid = true;
  }
  float prefetchMask = doMask ? maskTexture.sample(sNearest, evalPoint, level(0)).r : 0.0;
  int3  curCell     = int3(-1);
  bool  curCellEmpty = false;
  int3  curBlock    = int3(-1);
  bool  curBlockEmpty = false;
  // Three-state block summary (empty / mixed / all-solid): while the current
  // block is all-solid, per-cell lattice work is suspended for solidRun more
  // iterations (skips cannot occur in solid terrain, so one step is consumed
  // per iteration); exhaustion forces a summary refetch at the boundary.
  bool  curBlockSolid = false;
  int   solidRun     = 0;
  float3 mmDimF     = b.minMaxInfo.yzw;
  // Specialization §17: fc_dense bypasses per-batch R8 preamble for dense coarse volumes (SD4 30% fixed overhead, tail 41=20*2+1, 4x stride)
  // Dense && !fineSD => coarse dense bypass; fine dense keeps minMax (SD0.5 still benefits from leaps)
  const bool useMinMax = fc_minmax && !(fc_dense && !fc_fineSD) &&
    !useIndependentPath &&
    b.minMaxInfo.x > 0.5 &&
    b.minMaxInfo.y > 0.5 &&
    b.minMaxInfo.z > 0.5 &&
    b.minMaxInfo.w > 0.5;
  // Two-level occupancy summary (VTK_METAL_TEST_MM_BLOCKS -> fc_mmBlocks):
  // a coarser R8 texture marks whole-block regions of the DILATED fine lattice
  // whose cells are ALL empty. When on, the walks leap to block edges with one
  // summary fetch instead of grinding per cell. Block emptiness is derived
  // from the exact per-cell semantics, so the set of composited samples (and
  // therefore the rendered bytes) is unchanged — only jump granularity differs.
  // The bound texture is a 1x1x1 dummy when the feature is off; mmBlkDimF.x > 1
  // gates it off in that case.
  float3 mmBlkDimF = float3(minMaxBlockTexture.get_width(),
                            minMaxBlockTexture.get_height(),
                            minMaxBlockTexture.get_depth());
  const bool useMinMaxBlocks = useMinMax && fc_mmBlocks && mmBlkDimF.x > 1.0f;
  // True once any sample inside [0,1]^3 texture space has been reached. The
  // texture cube is axis-aligned and the ray is a straight line in texture
  // space, so a ray's in-bounds samples form a single contiguous interval:
  // after it has been inside and gone out, it can never re-enter.
  bool seenInBounds = false;
  // Non-divergent march (fc_marchVariant == 3, PERFORMANCE_INVESTIGATION.md
  // section 4.2): latch "opacity threshold reached" instead of breaking, so the
  // data-dependent loop exit no longer desynchronizes SIMT lanes. The texture
  // fetch stays unconditional (the pipeline stays full); only the accumulation
  // is gated (by select, not branch) so once a fragment reaches the threshold
  // it contributes nothing further — matching the baseline's break semantics.
  bool marchOpaque = false;
  // Variant 4 extends the same latch to the geometric/data-dependent exits
  // (CTP directional bounds, minmax-skip overshoot, tTerminateMax) so the loop
  // can run a CPU-computed uniform frame-max iteration count with all lanes
  // locked. Accumulation is gated on both flags; the fetch stays unconditional.
  bool marchDone = false;

  // ------------------------------------------------------------------------
  // V31 back-edge exit port of the baseline divergent march
  // (VTK_METAL_TEST_DOEXIT=1 -> VolumeFeature_MarchDoExit -> fc_doExit).
  //
  // divergent_tail_repro root cause: the MSL->Air compiler loses 4-12% to
  // GLSL->Air on DRAM-resident volumes when the march carries data-dependent
  // trip counts in a mid-body-exit CFG (for(...) { fetch; if (a>thr) break; }
  // — the sampler sandwiched between a header branch and a mid-body exit
  // branch). Folding all exit conditions into the loop BACK-EDGE (one branch
  // per iteration, do-while) restores Air quality to GL parity
  // (divergent_tail V31/V32; full RT x SD matrix 0.93-1.04). The
  // jitter_gap_repro DOEXIT port measured neutral because that harness march
  // differs from the production one; this applies the identical transformation
  // to the production march.
  //
  // Semantics vs the baseline loop further below:
  // - the for-header's first test becomes an explicit maxSteps > 0 entry
  //   guard (an unguarded do-while composites one bogus sample on zero-step
  //   corner-grazer rays — jitter_gap_repro lesson);
  // - every mid-body break becomes a marchStop latch; latched iterations
  //   still perform exactly the state updates the baseline performs around
  //   each break site (clamp fixup, advance, prefetch) and exit at the
  //   back-edge without compositing further;
  // - the slab-inheritance saturation check stays at the top of the body like
  //   the baseline (the continue-skip paths bypass the bottom latches, so the
  //   top check is what terminates post-skip iterations);
  // - marchIter keeps the baseline's update quirk (the final, stopping
  //   iteration does not touch it) so METAL_ITER PPMs stay comparable;
  // - latchExit/suppressAccum machinery is omitted: it is dead at variant 0
  //   (fc_marchVariant 0 never sets marchOpaque/marchDone).
  if (fc_doExit && fc_blendMode == 0 && !doCropping && !doMask && !doBlanking &&
      !doRectilinear && !doTransfer2D && !useIndependentPath &&
      !fc_dependentRGBA && !fc_dependentLA)
  {
    int i = 0;
    bool marchStop = maxSteps <= 0;
    do {
      // A slab pass whose inherited near-side alpha already exceeds the
      // saturation threshold contributes nothing; camera-inside rays
      // (checkBounds == false) additionally stop at tEnd (the bounded rays'
      // stop-at-tEnd comes from maxSteps).
      if ((fc_slabMode && accumulatedOpacity > kExitAcc) ||
          (!p.checkBounds && currentT >= p.tEnd - 1e-6))
      {
        marchStop = true;
      }

      const float3 adjTexMin = ctpOffset;
      const float3 adjTexMax = ctpOffset + ctpScale;
      if (!marchStop)
      {
        // The proxy box spans the axis-aligned bounds of the rotated volume,
        // so rays through its corner regions fall outside the [0,1]^3 texture
        // cube. Clamp-and-sample the boundary slab until the ray has been
        // inside once and left (seenInBounds); that is the baseline's
        // directional per-axis CTP bounds break (OpenGL
        // TerminationImplementation parity).
        if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
            any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f)))
        {
          if (seenInBounds)
          {
            marchStop = true; // baseline breaks here
          }
          texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
          evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
          prefetchValid = false;
        }
        else
        {
          seenInBounds = true;
        }
      }

      if (!marchStop)
      {
        if (useMinMax) {
          float3 mmPos = clamp(evalPoint, float3(0.0), float3(1.0));
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
            tToEdge.x = abs(p.rayDir.x) > 1e-5 ? distToEdge.x / (abs(p.rayDir.x) * mmDimF.x) : 1e30;
            tToEdge.y = abs(p.rayDir.y) > 1e-5 ? distToEdge.y / (abs(p.rayDir.y) * mmDimF.y) : 1e30;
            tToEdge.z = abs(p.rayDir.z) > 1e-5 ? distToEdge.z / (abs(p.rayDir.z) * mmDimF.z) : 1e30;

            float exactSkip = min(min(tToEdge.x, tToEdge.y), tToEdge.z);
            exactSkip += 1e-4;
            float skipDist = ceil(exactSkip / p.stepSize) * p.stepSize;
            skipDist = max(p.stepSize, skipDist);

            currentPoint += p.rayDir * skipDist;
            currentT += skipDist;

            if (p.checkBounds && (any(currentPoint < p.blockMinGlobal - 1e-4) || any(currentPoint > p.blockMaxGlobal + 1e-4) || currentT >= p.tEnd)) {
              marchStop = true; // baseline breaks here
            }

            // Re-sync the incremental sample position after the empty-cell jump.
            texLocalPos = (volumeUniforms.volumeToTexture *
                float4(volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize, 1.0)).xyz;
            evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
            prefetchValid = false;
            curCell = int3(-1);
            continue;
          }
        }

        bool needsFetch = !prefetchValid;
        float3 rectEvalPoint = evalPoint;
        if (doRectilinear &&
            (needsFetch || useIndependentPath || fc_dependentRGBA || fc_dependentLA)) {
          rectEvalPoint = rectilinearSamplePosition(evalPoint, true, rectCoords, volumeUniforms);
        }
        float rawScalar = needsFetch
          ? sampleVolumeScalar(volumeTexture, rectEvalPoint)
          : prefetchScalar;
        float4 rawScalar4 = float4(rawScalar, 0.0, 0.0, 0.0);
        if (useIndependentPath) {
          if (fc_linearInterpolation) {
            rawScalar4 = volumeTexture.sample(sVolume,
          volumeFetchSwizzle(rectEvalPoint), level(0));
          } else {
            rawScalar4 = volumeTexture.sample(sNearest,
          volumeFetchSwizzle(rectEvalPoint), level(0));
          }
        } else if (fc_dependentRGBA || fc_dependentLA) {
          rawScalar4 = sampleVolumeTexel(volumeTexture, rectEvalPoint);
        }
        float rawMask = (doMask && needsFetch)
          ? maskTexture.sample(sNearest, evalPoint, level(0)).r
          : prefetchMask;

        if (doCropping && ((cropBitmask & (1u << computeCropRegion(cropMin, cropMax, evalPoint))) == 0u)) {
          currentPoint += stepVec;
          currentT += p.stepSize;
          texLocalPos += texStep;
          evalPoint += evalStep;
          prefetchValid = false;
          continue;
        }

        if (doMask && volumeUniforms.maskType > 0.5) {
          float binMask = rawMask * maskScale + maskBias;
          if (binMask <= 0.0) {
            currentPoint += stepVec;
            currentT += p.stepSize;
            texLocalPos += texStep;
            evalPoint += evalStep;
            prefetchValid = false;
            continue;
          }
        }

        if (doBlanking) {
          float4 bCur = blankingTexture.sample(sNearest, evalPoint, level(0));
          float4 bXP = blankingTexture.sample(sNearest, evalPoint + float3(blankHalfStep.x, 0.0, 0.0), level(0));
          float4 bXN = blankingTexture.sample(sNearest, evalPoint - float3(blankHalfStep.x, 0.0, 0.0), level(0));
          float4 bYP = blankingTexture.sample(sNearest, evalPoint + float3(0.0, blankHalfStep.y, 0.0), level(0));
          float4 bYN = blankingTexture.sample(sNearest, evalPoint - float3(0.0, blankHalfStep.y, 0.0), level(0));
          float4 bZP = blankingTexture.sample(sNearest, evalPoint + float3(0.0, 0.0, blankHalfStep.z), level(0));
          float4 bZN = blankingTexture.sample(sNearest, evalPoint - float3(0.0, 0.0, blankHalfStep.z), level(0));

          const bool anyPoint = (bCur.x > 0.0 || bXP.x > 0.0 || bXN.x > 0.0 ||
                                 bYP.x > 0.0 || bYN.x > 0.0 || bZP.x > 0.0 ||
                                 bZN.x > 0.0);
          const bool anyCell  = (bCur.y > 0.0 || bXP.y > 0.0 || bXN.y > 0.0 ||
                                 bYP.y > 0.0 || bYN.y > 0.0 || bZP.y > 0.0 ||
                                 bZN.y > 0.0);

          bool blanked = false;
          if (volumeUniforms.blankingMode == 1.0) {
            blanked = anyCell;
          } else if (volumeUniforms.blankingMode == 2.0) {
            blanked = anyPoint;
          } else {
            blanked = (anyCell || anyPoint);
          }
          if (blanked) {
            currentPoint += stepVec;
            currentT += p.stepSize;
            texLocalPos += texStep;
            evalPoint += evalStep;
            prefetchValid = false;
            continue;
          }
        }

        half scalarNorm = saturate(half(rawScalar) * scalarScale + scalarBias);

        half scalarNormComp[4] = {scalarNorm, scalarNorm, scalarNorm, scalarNorm};
        half compScale[4] = {0.0h, 0.0h, 0.0h, 0.0h};
        half compBias[4] = {0.0h, 0.0h, 0.0h, 0.0h};
        if (useIndependentPath) {
          int nComp = min(4, int(volumeUniforms.numComponents));
          for (int c = 0; c < nComp; ++c) {
            half cMin = half(volumeUniforms.scalarMinComp[c]);
            half cRange = max(half(volumeUniforms.scalarMaxComp[c]) - cMin, 1e-4h);
            compScale[c] = 1.0h / cRange;
            compBias[c] = -cMin / cRange;
            float rawComp;
            if (c == 0) rawComp = rawScalar;
            else if (c == 1) rawComp = rawScalar4.g;
            else if (c == 2) rawComp = rawScalar4.b;
            else rawComp = rawScalar4.a;
            scalarNormComp[c] = saturate((half(rawComp) - cMin) / cRange);
          }
        }

        half4 colorOpacity;
        half maskLabel = 0.0h;
        half4 sharedGrad = half4(0.0h);
        bool sharedGradReady = false;
        half4 cachedDensityGrad = half4(0.0h);
        bool densityGradReady = false;

        half4 compColor[4] = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)};

        const bool fc_needsPerSampleOpacity =
          (fc_blendMode == 0 || fc_blendMode == 3 || fc_blendMode == 4);
        if (useIndependentPath) {
          if (fc_needsPerSampleOpacity) {
            int nComp = min(4, int(volumeUniforms.numComponents));
            for (int c = 0; c < nComp; ++c) {
              compColor[c] = sampleComponentTransferFunction(
                  transferFunctionTexture, transferFunctionTexture1,
                  transferFunctionTexture2, transferFunctionTexture3,
                  float2(float(scalarNormComp[c]), 0.5), c);
            }
          }
        } else if (fc_needsPerSampleOpacity && doTransfer2D) {
          half secondNorm;
          if (volumeUniforms.transfer2DUseGradient > 0.5) {
            sharedGrad = computeGradientFast(volumeTexture, evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor);
            sharedGradReady = true;
            secondNorm = sharedGrad.w;
          } else {
            secondNorm = saturate(
                half(sampleSecondScalar(transfer2DYAxisTexture, evalPoint)) * secondScale + secondBias);
          }
          colorOpacity = sampleTransferFunction2D(
              transferFunction2DTexture, float2(float(scalarNorm), float(secondNorm)));
        } else if (fc_needsPerSampleOpacity && doMask) {
          float maskVal = rawMask * maskScale + maskBias;
          if (numLabels > 0.0) {
            float label = floor(maskVal + 0.5);
            if (label > 0.0) {
              label = clamp(label, 1.0, numLabels - 1.0);
              maskLabel = half(label);
              float labelY = (label + 0.5) / numLabels;
              colorOpacity = half4(labelMapTransferTexture.sample(sNearest, float2(float(scalarNorm), labelY), level(0)));
            } else {
              colorOpacity = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
            }
          } else {
            colorOpacity = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
          }
        } else if (fc_needsPerSampleOpacity) {
          if (fc_dependentRGBA) {
            half rgbaOpacity =
              sampleTransferFunction(transferFunctionTexture, float2(rawScalar4.a, 0.5)).a;
            colorOpacity = half4(half3(rawScalar4.rgb), rgbaOpacity);
          } else if (fc_dependentLA) {
            half4 laColor = sampleTransferFunction(
                transferFunctionTexture, float2(float(scalarNorm), 0.5));
            half lastMin = half(volumeUniforms.scalarMinComp[1]);
            half lastMax = half(volumeUniforms.scalarMaxComp[1]);
            half lastNorm = saturate(
                (half(rawScalar4.g) - lastMin) / max(lastMax - lastMin, 1e-4h));
            half laOpacity = sampleTransferFunction(
                transferFunctionTexture, float2(float(lastNorm), 0.5)).a;
            colorOpacity = half4(laColor.rgb, laOpacity);
          } else {
            colorOpacity = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
          }
        } else {
          colorOpacity = half4(0.0h);
        }

        half sampleOpacity = colorOpacity.a;

        if (useIndependentPath) {
          if (fc_needsPerSampleOpacity && doGradOp) {
            if (!compGradReady) {
              computeGradientsAllComponents(volumeTexture, evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor, compGrad);
              compGradReady = true;
            }
            int nComp = min(4, int(volumeUniforms.numComponents));
            for (int c = 0; c < nComp; ++c) {
              compColor[c].a *= sampleGradientOpacity(gradientOpacityTexture, float(compGrad[c].w));
            }
          }
          half totalAlpha = 0.0h;
          int nComp = min(4, int(volumeUniforms.numComponents));
          for (int c = 0; c < nComp; ++c) {
            if (volumeUniforms.componentWeight[c] <= 0.0) continue;
            totalAlpha += compColor[c].a * half(volumeUniforms.componentWeight[c]);
          }
          sampleOpacity = totalAlpha;
        }

        if (useIndependentPath) {
          if (fc_blendMode == 1) {           // MAXIMUM_INTENSITY_BLEND
            int nComp = min(4, int(volumeUniforms.numComponents));
            for (int c = 0; c < nComp; ++c) {
              if (firstBlendSample || mipMaxScalarComp[c] < scalarNormComp[c]) {
                mipMaxScalarComp[c] = scalarNormComp[c];
              }
            }
            firstBlendSample = false;
          } else if (fc_blendMode == 2) {    // MINIMUM_INTENSITY_BLEND
            int nComp = min(4, int(volumeUniforms.numComponents));
            for (int c = 0; c < nComp; ++c) {
              if (firstBlendSample || minipMinScalarComp[c] > scalarNormComp[c]) {
                minipMinScalarComp[c] = scalarNormComp[c];
              }
            }
            firstBlendSample = false;
          } else if (fc_blendMode == 3) {    // AVERAGE_INTENSITY_BLEND
            int nComp = min(4, int(volumeUniforms.numComponents));
            for (int c = 0; c < nComp; ++c) {
              half intensityNorm =
                half(volumeUniforms.scalarMinComp[c]) +
                (half(volumeUniforms.scalarMaxComp[c]) - half(volumeUniforms.scalarMinComp[c])) * scalarNormComp[c];
              if (intensityNorm >= half(volumeUniforms.averageIPRangeMin) &&
                  intensityNorm <= half(volumeUniforms.averageIPRangeMax)) {
                avgBlendSumComp[c] += compColor[c].a * scalarNormComp[c];
                avgBlendCountComp[c]++;
              }
            }
          } else if (fc_blendMode == 4) {    // ADDITIVE_BLEND
            int nComp = min(4, int(volumeUniforms.numComponents));
            for (int c = 0; c < nComp; ++c) {
              additiveSumComp[c] += compColor[c].a * scalarNormComp[c];
            }
          }
        } else if (fc_blendMode == 1) {           // MAXIMUM_INTENSITY_BLEND
          if (firstBlendSample || mipMaxScalar < scalarNorm) {
            mipMaxScalar = scalarNorm;
          }
          firstBlendSample = false;
        } else if (fc_blendMode == 2) {    // MINIMUM_INTENSITY_BLEND
          if (firstBlendSample || minipMinScalar > scalarNorm) {
            minipMinScalar = scalarNorm;
          }
          firstBlendSample = false;
        } else if (fc_blendMode == 3) {    // AVERAGE_INTENSITY_BLEND
          half intensityNorm =
            volumeUniforms.scalarMin + (volumeUniforms.scalarMax - volumeUniforms.scalarMin) * scalarNorm;
          if (intensityNorm >= half(volumeUniforms.averageIPRangeMin) &&
              intensityNorm <= half(volumeUniforms.averageIPRangeMax)) {
            avgBlendSum += sampleOpacity * scalarNorm;
            avgBlendCount++;
          }
        } else if (fc_blendMode == 4) {    // ADDITIVE_BLEND
          additiveSum += sampleOpacity * scalarNorm;
        }
        if (fc_renderToTexture && haveOpaquePos != nullptr && *haveOpaquePos && sampleOpacity > 0.0h) {
          *firstOpaquePos = currentPoint;
          *haveOpaquePos = false;
        }

        if (useIndependentPath) {
          if (fc_blendMode == 0 && sampleOpacity > 0.0h) {
            half3 tmpRGB = half3(0.0h);
            half tmpA = 0.0h;
            int nComp = min(4, int(volumeUniforms.numComponents));
            for (int c = 0; c < nComp; ++c) {
              half w = half(volumeUniforms.componentWeight[c]);
              if (w <= 0.0h) continue;
              half4 cc = compColor[c];
              half3 ccRGB = cc.rgb;
              if (sampleOpacity >= 0.01h && doShading) {
                half3 normal;
                if (fc_computeNormalFromOpacity) {
                  normal = computeDensityGradientFast(volumeTexture,
                      transferFunctionTexture, transferFunctionTexture1,
                      transferFunctionTexture2, transferFunctionTexture3,
                      evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture,
                      gradNormFactor, c, compScale[c], compBias[c]).xyz;
                } else if (fc_normalTexture) {
                  half4 nrmSample = half4(normalTexture.sample(sVolume, evalPoint, level(0)));
                  normal = normalize(nrmSample.xyz * 2.0h - 1.0h);
                } else {
                  if (!compGradReady) {
                    computeGradientsAllComponents(volumeTexture, evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor, compGrad);
                    compGradReady = true;
                  }
                  normal = compGrad[c].xyz;
                }
                half3 ambC = half3(volumeUniforms.ambientColorComp[c].rgb);
                half3 difC = half3(volumeUniforms.diffuseColorComp[c].rgb);
                half3 speC = half3(volumeUniforms.specularColorComp[c].rgb);
                half  shiC = half(volumeUniforms.shininessComp[c]);
                if (lightUniforms != nullptr && !fc_defaultLighting) {
                  ccRGB = computeVolumeLighting(ccRGB, normal, -viewDirHalf,
                      ambC, difC, speC, shiC,
                      *lightUniforms,
                      volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize);
                } else {
                  bool twoSided = (lightUniforms != nullptr && lightUniforms->twoSidedLighting != 0);
                  ccRGB = computePhongLightingVolumeFast(ccRGB, normal, -viewDirHalf, -viewDirHalf,
                      ambC, difC, speC, shiC, twoSided);
                }
              }
              tmpRGB += ccRGB * cc.a * w;
              tmpA += (cc.a * cc.a) / sampleOpacity;
            }
            half weight = 1.0h - accumulatedOpacity;
            accumulatedColor += weight * tmpRGB;
            accumulatedOpacity += weight * tmpA;
          }
        } else if (fc_blendMode == 0 && sampleOpacity > 0.0h) {
          half3 sampleColor = colorOpacity.rgb;
          half weight = 1.0h - accumulatedOpacity;

          if (doGradOp && maskLabel == 0.0h) {
            if (!sharedGradReady) {
              if (fc_normalTexture) {
                half4 nrmSample = half4(normalTexture.sample(sVolume, evalPoint, level(0)));
                sharedGrad = half4(normalize(nrmSample.xyz * 2.0h - 1.0h), nrmSample.w);
              } else if (fc_computeNormalFromOpacity) {
                sharedGrad = computeScalarAndDensityGradient(volumeTexture,
                    transferFunctionTexture, transferFunctionTexture1,
                    transferFunctionTexture2, transferFunctionTexture3,
                    evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture,
                    gradNormFactor, scalarScale, scalarBias, cachedDensityGrad);
                densityGradReady = true;
              } else {
                sharedGrad = computeGradientFast(volumeTexture, evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor);
              }
              sharedGradReady = true;
            }
            sampleOpacity *= sampleGradientOpacity(gradientOpacityTexture, float(sharedGrad.w));
          }

          if (doShading && maskLabel == 0.0h && sampleOpacity > 0.02h) {

            half3 normal;
            if (fc_computeNormalFromOpacity) {
              if (densityGradReady) {
                normal = cachedDensityGrad.xyz;
              } else {
                normal = computeDensityGradientFast(volumeTexture,
                    transferFunctionTexture, transferFunctionTexture1,
                    transferFunctionTexture2, transferFunctionTexture3,
                    evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture,
                    gradNormFactor, 0, scalarScale, scalarBias).xyz;
              }
            } else {
              if (!sharedGradReady) {
                if (fc_normalTexture) {
                  half4 nrmSample = half4(normalTexture.sample(sVolume, evalPoint, level(0)));
                  sharedGrad = half4(normalize(nrmSample.xyz * 2.0h - 1.0h), nrmSample.w);
                } else {
                  sharedGrad = computeGradientFast(volumeTexture, evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor);
                }
                sharedGradReady = true;
              }
              normal = sharedGrad.xyz;
            }

            if (lightUniforms != nullptr && !fc_defaultLighting) {
              sampleColor = computeVolumeLighting(sampleColor, normal, -viewDirHalf,
                  ambientMat, diffuseMat, specularMat, shininessMat,
                  *lightUniforms,
                  volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize);
            } else {
              bool twoSided = (lightUniforms != nullptr && lightUniforms->twoSidedLighting != 0);
              sampleColor = computePhongLightingVolumeFast(sampleColor, normal, -viewDirHalf, -viewDirHalf,
                  ambientMat, diffuseMat, specularMat, shininessMat, twoSided);
            }
          } else if (doShading && sampleOpacity > 0.0h) {
            sampleColor = ambientMat * sampleColor;
          }

          accumulatedColor += weight * (sampleColor * sampleOpacity);
          accumulatedOpacity += weight * sampleOpacity;
        }

        currentPoint += stepVec;
        currentT += p.stepSize;
        texLocalPos += texStep;
        evalPoint += evalStep;

        if (volumeUniforms._padCropFlags[2] < 0.5f && i + 1 < maxSteps) {
          prefetchScalar = sampleVolumeScalar(volumeTexture,
              rectilinearSamplePosition(evalPoint, doRectilinear, rectCoords, volumeUniforms));
          if (doMask) {
            prefetchMask = maskTexture.sample(sNearest, evalPoint, level(0)).r;
          }
          prefetchValid = true;
        }

        // Bottom latches (the baseline breaks here): OpenGL-parity threshold
        // g_opacityThreshold = 1 - 1/255 evaluated WITHOUT clamping the
        // accumulated opacity, then the terminate-plane distance.
        if (accumulatedOpacity > kExitAcc) {
          marchStop = true;
        }
        if (currentT >= p.tTerminateMax) {
          marchStop = true;
        }
      }

      // Baseline sets marchIter only on completed (non-breaking) iterations.
      if (!marchStop) {
        marchIter = i + 1;
      }
    } while (++i < maxSteps && !marchStop);
  }
  // Env-gated N-way unrolled march (fc_marchVariant 6 = 8x, 7 = 4x, selected
  // via VTK_METAL_TEST_MARCH_VARIANT). PERFORMANCE_INVESTIGATION.md section 14:
  // the harness showed the bare-fetch march is texture-latency bound and that N
  // independent scalar fetches issued back-to-back (before any consume) hide
  // the issue latency, while prefetch-ahead with a loop-carried dependency does
  // not. This branch restructures the fetch scheduling the same way: each batch
  // issues N independent scalar fetches on the step lattice, then consumes them
  // sequentially. All exits are latched (marchOpaque/marchDone) and accumulation
  // is gated by suppressAccum, so batches may over-march the per-fragment bound
  // without changing the result (variant-4 semantics); a mid-batch minmax
  // empty-cell skip or out-of-cube clamp invalidates the remaining pre-fetched
  // scalars and forces a refill from the moved position. The guard is fully
  // compile-time, so any other feature combination keeps the baseline loop.
  else if (fc_marchVariant >= 6)
  {
    const int unrollN = (fc_marchVariant == 7) ? 4 : 8;
    const float3 adjTexMin = ctpOffset;
    const float3 adjTexMax = ctpOffset + ctpScale;
    if (fc_marchVariant == 9)
    {
      // fc_marchVariant 9: adaptive-width harness scheduling with inline sample
      // addresses (probe v39 fragment_march_phase_batch_w48). Same scheduling
      // as variant 8 but each batch issues N independent scalar fetches
      // back-to-back and each address dies at issue (no p0..pN float3 live
      // registers), so far more volume fetches are in flight per warp and the
      // issue latency is hidden even harder. The probe measured w48 35.7-38.7ms
      // vs w8 (v34) 54.4-57.9ms and w16 47.7-50.1ms on the 512x512x1794 R8
      // workload.
      //
      // Batch width is adaptive: each iteration picks the largest width from
      // {48, 32, 16, 8, 4, 2, 1} that fits the remaining steps. 48-wide
      // batches keep the fine-sample-distance win, and the small-width ladder
      // replaces the old scalar tail so short rays (coarse sample distances,
      // shallow chords) never fall back to a per-sample loop (see
      // PERFORMANCE_INVESTIGATION.md section 19).
      //
      // Minmax (fc_minmax) is handled by a preamble lattice walk that runs
      // BEFORE the width dispatch, replacing the batch-8 consume minmax path
      // (which prefetched a full batch before the lattice check, so every
      // empty cell discarded up to 8 volume fetches and desynced i from the
      // advanced position -- measured 150-580ms and unstable on the DICOM
      // study). The walk issues only tiny R8 lattice fetches while the leading
      // run is empty, so empty space is traversed without volume fetches at
      // all; its extent is capped at the remaining steps so a batch is never
      // dispatched past the ray end (tEnd), which kept the coarse-SD output
      // error-free (see section 19). Empty cells contribute zero (their
      // scalars map to opacity-zero TF indices, verified by the occupancy
      // prefix table), so the unconditional composite chain stays unchanged
      // and correct for both the lean and minmax cases. fc_minmax is a
      // function constant, so the lean specialization is identical to the
      // pre-minmax code and the walk is dead-code-eliminated. Any other
      // feature combination keeps the batch-8 consume.
#define MV9_FETCH(_j) \
  float3 pos##_j = evalPoint + evalStep * (float)_j; \
  float3 rPos##_j = fc_rectilinear ? rectilinearSamplePosition(pos##_j, true, rectCoords, volumeUniforms) : pos##_j; \
  float s##_j = sampleVolumeScalar(volumeTexture, rPos##_j);
#define MV9_COMPOSITE(_j) \
  float3 posC##_j = evalPoint + evalStep * (float)_j; \
  float3 rPosC##_j = fc_rectilinear ? rectilinearSamplePosition(posC##_j, true, rectCoords, volumeUniforms) : posC##_j; \
  bool skip##_j = false; \
  if (fc_cropping) { \
    if ((cropBitmask & (1u << computeCropRegion(cropMin, cropMax, rPosC##_j))) == 0u) skip##_j = true; \
  } \
  if (!skip##_j && fc_blanking) { \
    float3 bPos##_j = rPosC##_j + blankHalfStep * (volumeUniforms.blankingMode > 1.5 ? 1.0f : 0.0f); \
    float bVal##_j = blankingTexture.sample(sNearest, rPosC##_j, level(0)).r; \
    if (bVal##_j < 0.5f) skip##_j = true; \
    if (!skip##_j && volumeUniforms.blankingMode > 0.5f) { \
      float bVal2##_j = blankingTexture.sample(sNearest, bPos##_j, level(0)).r; \
      if (bVal2##_j < 0.5f) skip##_j = true; \
    } \
  } \
  if (fc_independentComponents) { \
    if (!skip##_j) { \
      float4 s4_##_j = sampleVolumeTexel(volumeTexture, rPosC##_j); \
      int nComp##_j = min(4, int(volumeUniforms.numComponents)); \
      half scalarNormComp##_j[4] = {half(0.0h), half(0.0h), half(0.0h), half(0.0h)}; \
      half compScale##_j[4] = {0.0h, 0.0h, 0.0h, 0.0h}; \
      half compBias##_j[4] = {0.0h, 0.0h, 0.0h, 0.0h}; \
      for (int c = 0; c < nComp##_j; ++c) { \
        half cMin = half(volumeUniforms.scalarMinComp[c]); \
        half cRange = max(half(volumeUniforms.scalarMaxComp[c]) - cMin, 1e-4h); \
        compScale##_j[c] = 1.0h / cRange; \
        compBias##_j[c] = -cMin / cRange; \
        float rawComp = (c == 0 ? s4_##_j.r : (c == 1 ? s4_##_j.g : (c == 2 ? s4_##_j.b : s4_##_j.a))); \
        scalarNormComp##_j[c] = saturate((half(rawComp) - cMin) / cRange); \
      } \
      if (fc_blendMode == 1) { \
        for (int c = 0; c < nComp##_j; ++c) { if (firstBlendSample || mipMaxScalarComp[c] < scalarNormComp##_j[c]) mipMaxScalarComp[c] = scalarNormComp##_j[c]; } \
        firstBlendSample = false; \
      } else if (fc_blendMode == 2) { \
        for (int c = 0; c < nComp##_j; ++c) { if (firstBlendSample || minipMinScalarComp[c] > scalarNormComp##_j[c]) minipMinScalarComp[c] = scalarNormComp##_j[c]; } \
        firstBlendSample = false; \
      } else if (fc_blendMode == 3) { \
        half4 compColor##_j[4] = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)}; \
        for (int c = 0; c < nComp##_j; ++c) compColor##_j[c] = sampleComponentTransferFunction(transferFunctionTexture, transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3, float2(float(scalarNormComp##_j[c]), 0.5), c); \
        if (fc_gradientOpacity) { \
          half4 compGrad##_j[4] = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)}; \
          computeGradientsAllComponents(volumeTexture, rPosC##_j, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor, compGrad##_j); \
          for (int c = 0; c < nComp##_j; ++c) compColor##_j[c].a *= sampleGradientOpacity(gradientOpacityTexture, float(compGrad##_j[c].w)); \
        } \
        for (int c = 0; c < nComp##_j; ++c) { \
          half intensityNorm = half(volumeUniforms.scalarMinComp[c]) + (half(volumeUniforms.scalarMaxComp[c]) - half(volumeUniforms.scalarMinComp[c])) * scalarNormComp##_j[c]; \
          if (intensityNorm >= half(volumeUniforms.averageIPRangeMin) && intensityNorm <= half(volumeUniforms.averageIPRangeMax)) { avgBlendSumComp[c] += compColor##_j[c].a * scalarNormComp##_j[c]; avgBlendCountComp[c]++; } \
        } \
      } else if (fc_blendMode == 4) { \
        half4 compColor##_j[4] = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)}; \
        for (int c = 0; c < nComp##_j; ++c) compColor##_j[c] = sampleComponentTransferFunction(transferFunctionTexture, transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3, float2(float(scalarNormComp##_j[c]), 0.5), c); \
        if (fc_gradientOpacity) { \
          half4 compGrad##_j[4] = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)}; \
          computeGradientsAllComponents(volumeTexture, rPosC##_j, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor, compGrad##_j); \
          for (int c = 0; c < nComp##_j; ++c) compColor##_j[c].a *= sampleGradientOpacity(gradientOpacityTexture, float(compGrad##_j[c].w)); \
        } \
        for (int c = 0; c < nComp##_j; ++c) additiveSumComp[c] += compColor##_j[c].a * scalarNormComp##_j[c]; \
      } else { \
        half4 compColor##_j[4] = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)}; \
        for (int c = 0; c < nComp##_j; ++c) compColor##_j[c] = sampleComponentTransferFunction(transferFunctionTexture, transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3, float2(float(scalarNormComp##_j[c]), 0.5), c); \
        half totalAlpha##_j = 0.0h; \
        half4 compGrad##_j[4] = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)}; \
        bool compGradReady##_j = false; \
        if (fc_gradientOpacity) { \
          computeGradientsAllComponents(volumeTexture, rPosC##_j, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor, compGrad##_j); \
          compGradReady##_j = true; \
          for (int c = 0; c < nComp##_j; ++c) compColor##_j[c].a *= sampleGradientOpacity(gradientOpacityTexture, float(compGrad##_j[c].w)); \
        } \
        for (int c = 0; c < nComp##_j; ++c) { if (volumeUniforms.componentWeight[c] <= 0.0) continue; totalAlpha##_j += compColor##_j[c].a * half(volumeUniforms.componentWeight[c]); } \
        if (fc_renderToTexture && haveOpaquePos != nullptr && *haveOpaquePos && totalAlpha##_j > 0.0h) { *firstOpaquePos = currentPoint + stepVec * (float)_j; *haveOpaquePos = false; } \
        if (totalAlpha##_j > 0.0h) { \
          half3 tmpRGB##_j = half3(0.0h); \
          half tmpA##_j = 0.0h; \
          for (int c = 0; c < nComp##_j; ++c) { \
            half wC = half(volumeUniforms.componentWeight[c]); \
            if (wC <= 0.0h) continue; \
            half4 cc = compColor##_j[c]; \
            half3 ccRGB = cc.rgb; \
            if (fc_shading && totalAlpha##_j >= 0.01h) { \
              half3 nTmp##_j; \
              if (fc_computeNormalFromOpacity) { \
                nTmp##_j = computeDensityGradientFast(volumeTexture, transferFunctionTexture, transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3, rPosC##_j, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor, c, compScale##_j[c], compBias##_j[c]).xyz; \
              } else if (fc_normalTexture) { \
                half4 nsTmp##_j = half4(normalTexture.sample(sVolume, rPosC##_j, level(0))); \
                nTmp##_j = normalize(nsTmp##_j.xyz * 2.0h - 1.0h); \
              } else { \
                if (!compGradReady##_j) { computeGradientsAllComponents(volumeTexture, rPosC##_j, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor, compGrad##_j); compGradReady##_j = true; } \
                nTmp##_j = compGrad##_j[c].xyz; \
              } \
              half3 ambC = half3(volumeUniforms.ambientColorComp[c].rgb); \
              half3 difC = half3(volumeUniforms.diffuseColorComp[c].rgb); \
              half3 speC = half3(volumeUniforms.specularColorComp[c].rgb); \
              half shiC = half(volumeUniforms.shininessComp[c]); \
              if (lightUniforms != nullptr && !fc_defaultLighting) { \
                ccRGB = computeVolumeLighting(ccRGB, nTmp##_j, -viewDirHalf, ambC, difC, speC, shiC, *lightUniforms, volumeUniforms.volumeBoundsMin.xyz + (currentPoint + stepVec * (float)_j) * boundsSize); \
              } else { \
                bool twoSidedTmp##_j = (lightUniforms != nullptr && lightUniforms->twoSidedLighting != 0); \
                ccRGB = computePhongLightingVolumeFast(ccRGB, nTmp##_j, -viewDirHalf, -viewDirHalf, ambC, difC, speC, shiC, twoSidedTmp##_j); \
              } \
            } \
            tmpRGB##_j += ccRGB * cc.a * wC; \
            tmpA##_j += (cc.a * cc.a) / totalAlpha##_j; \
          } \
          half w##_j = 1.0h - accumulatedOpacity; \
          accumulatedColor += w##_j * tmpRGB##_j; \
          accumulatedOpacity += w##_j * tmpA##_j; \
        } \
      } \
    } \
  } else { \
    half scalarNorm##_j = saturate(half(s##_j) * scalarScale + scalarBias); \
    if (fc_blendMode == 1) { \
      if (!skip##_j) { if (firstBlendSample || mipMaxScalar < scalarNorm##_j) mipMaxScalar = scalarNorm##_j; firstBlendSample = false; } \
    } else if (fc_blendMode == 2) { \
      if (!skip##_j) { if (firstBlendSample || minipMinScalar > scalarNorm##_j) minipMinScalar = scalarNorm##_j; firstBlendSample = false; } \
    } else if (fc_blendMode == 3) { \
      if (!skip##_j) { \
        half4 c##_j = half4(0.0h); \
        if (fc_transfer2D) { \
          half secondNorm##_j; \
          if (volumeUniforms.transfer2DUseGradient > 0.5) { \
            half4 g2##_j = computeGradientFast(volumeTexture, rPosC##_j, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor); \
            secondNorm##_j = g2##_j.w; \
          } else { \
            secondNorm##_j = half(sampleSecondScalar(transfer2DYAxisTexture, rPosC##_j) * secondScale + secondBias); \
          } \
          c##_j = sampleTransferFunction2D(transferFunction2DTexture, float2(float(scalarNorm##_j), float(secondNorm##_j))); \
        } else if (fc_mask) { \
          float rawMask##_j = maskTexture.sample(sNearest, rPosC##_j, level(0)).r; \
          float maskVal##_j = rawMask##_j * maskScale + maskBias; \
          if (volumeUniforms.maskType > 0.5) { \
            if (maskVal##_j > 0.0) c##_j = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm##_j), 0.5)); \
            else skip##_j = true; \
          } else { \
            if (numLabels > 0.0) { \
              float label##_j = floor(maskVal##_j + 0.5); \
              if (label##_j > 0.0) { label##_j = clamp(label##_j, 1.0, numLabels - 1.0); float labelY##_j = (label##_j + 0.5) / numLabels; c##_j = half4(labelMapTransferTexture.sample(sNearest, float2(float(scalarNorm##_j), labelY##_j), level(0))); } \
              else c##_j = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm##_j), 0.5)); \
            } else c##_j = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm##_j), 0.5)); \
          } \
        } else if (fc_dependentRGBA) { \
          float4 s4_dep_##_j = sampleVolumeTexel(volumeTexture, rPosC##_j); \
          half rgbaOpacity##_j = sampleTransferFunction(transferFunctionTexture, float2(s4_dep_##_j.a, 0.5)).a; \
          c##_j = half4(half3(s4_dep_##_j.rgb), rgbaOpacity##_j); \
        } else if (fc_dependentLA) { \
          float4 s4_dep_##_j = sampleVolumeTexel(volumeTexture, rPosC##_j); \
          half4 laColor##_j = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm##_j), 0.5)); \
          half lastMin##_j = half(volumeUniforms.scalarMinComp[1]); \
          half lastMax##_j = half(volumeUniforms.scalarMaxComp[1]); \
          half lastNorm##_j = saturate((half(s4_dep_##_j.g) - lastMin##_j) / max(lastMax##_j - lastMin##_j, 1e-4h)); \
          half laOpacity##_j = sampleTransferFunction(transferFunctionTexture, float2(float(lastNorm##_j), 0.5)).a; \
          c##_j = half4(laColor##_j.rgb, laOpacity##_j); \
        } else { \
          c##_j = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm##_j), 0.5)); \
        } \
        if (!skip##_j) { \
          half intensityNorm##_j = volumeUniforms.scalarMin + (volumeUniforms.scalarMax - volumeUniforms.scalarMin) * scalarNorm##_j; \
          if (intensityNorm##_j >= half(volumeUniforms.averageIPRangeMin) && intensityNorm##_j <= half(volumeUniforms.averageIPRangeMax)) { avgBlendSum += c##_j.a * scalarNorm##_j; avgBlendCount++; } \
        } \
      } \
    } else if (fc_blendMode == 4) { \
      if (!skip##_j) { \
        half4 c##_j = half4(0.0h); \
        if (fc_transfer2D) { \
          half secondNorm##_j; \
          if (volumeUniforms.transfer2DUseGradient > 0.5) { \
            half4 g2##_j = computeGradientFast(volumeTexture, rPosC##_j, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor); \
            secondNorm##_j = g2##_j.w; \
          } else { \
            secondNorm##_j = half(sampleSecondScalar(transfer2DYAxisTexture, rPosC##_j) * secondScale + secondBias); \
          } \
          c##_j = sampleTransferFunction2D(transferFunction2DTexture, float2(float(scalarNorm##_j), float(secondNorm##_j))); \
        } else if (fc_mask) { \
          float rawMask##_j = maskTexture.sample(sNearest, rPosC##_j, level(0)).r; \
          float maskVal##_j = rawMask##_j * maskScale + maskBias; \
          if (volumeUniforms.maskType > 0.5) { \
            if (maskVal##_j > 0.0) c##_j = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm##_j), 0.5)); \
            else skip##_j = true; \
          } else { \
            if (numLabels > 0.0) { \
              float label##_j = floor(maskVal##_j + 0.5); \
              if (label##_j > 0.0) { label##_j = clamp(label##_j, 1.0, numLabels - 1.0); float labelY##_j = (label##_j + 0.5) / numLabels; c##_j = half4(labelMapTransferTexture.sample(sNearest, float2(float(scalarNorm##_j), labelY##_j), level(0))); } \
              else c##_j = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm##_j), 0.5)); \
            } else c##_j = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm##_j), 0.5)); \
          } \
        } else if (fc_dependentRGBA) { \
          float4 s4_dep_##_j = sampleVolumeTexel(volumeTexture, rPosC##_j); \
          half rgbaOpacity##_j = sampleTransferFunction(transferFunctionTexture, float2(s4_dep_##_j.a, 0.5)).a; \
          c##_j = half4(half3(s4_dep_##_j.rgb), rgbaOpacity##_j); \
        } else if (fc_dependentLA) { \
          float4 s4_dep_##_j = sampleVolumeTexel(volumeTexture, rPosC##_j); \
          half4 laColor##_j = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm##_j), 0.5)); \
          half lastMin##_j = half(volumeUniforms.scalarMinComp[1]); \
          half lastMax##_j = half(volumeUniforms.scalarMaxComp[1]); \
          half lastNorm##_j = saturate((half(s4_dep_##_j.g) - lastMin##_j) / max(lastMax##_j - lastMin##_j, 1e-4h)); \
          half laOpacity##_j = sampleTransferFunction(transferFunctionTexture, float2(float(lastNorm##_j), 0.5)).a; \
          c##_j = half4(laColor##_j.rgb, laOpacity##_j); \
        } else { \
          c##_j = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm##_j), 0.5)); \
        } \
        if (!skip##_j) additiveSum += c##_j.a * scalarNorm##_j; \
      } \
    } else { \
      half4 c##_j = half4(0.0h); \
      half3 col##_j = half3(0.0h); \
      half opa##_j = 0.0h; \
      half maskLabel##_j = 0.0h; \
      if (!skip##_j) { \
        if (fc_transfer2D) { \
          half secondNorm##_j; \
          if (volumeUniforms.transfer2DUseGradient > 0.5) { \
            half4 g2##_j = computeGradientFast(volumeTexture, rPosC##_j, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor); \
            secondNorm##_j = g2##_j.w; \
          } else { \
            secondNorm##_j = half(sampleSecondScalar(transfer2DYAxisTexture, rPosC##_j) * secondScale + secondBias); \
          } \
          c##_j = sampleTransferFunction2D(transferFunction2DTexture, float2(float(scalarNorm##_j), float(secondNorm##_j))); \
        } else if (fc_mask) { \
          float rawMask##_j = maskTexture.sample(sNearest, rPosC##_j, level(0)).r; \
          float maskVal##_j = rawMask##_j * maskScale + maskBias; \
          if (volumeUniforms.maskType > 0.5) { \
            if (maskVal##_j <= 0.0) { skip##_j = true; } \
            else { c##_j = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm##_j), 0.5)); } \
          } else { \
            if (numLabels > 0.0) { \
              float label##_j = floor(maskVal##_j + 0.5); \
              if (label##_j > 0.0) { \
                label##_j = clamp(label##_j, 1.0, numLabels - 1.0); \
                float labelY##_j = (label##_j + 0.5) / numLabels; \
                c##_j = half4(labelMapTransferTexture.sample(sNearest, float2(float(scalarNorm##_j), labelY##_j), level(0))); \
              } else { \
                c##_j = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm##_j), 0.5)); \
              } \
            } else { \
              c##_j = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm##_j), 0.5)); \
            } \
          } \
        } else if (fc_dependentRGBA) { \
          float4 s4_dep_##_j = sampleVolumeTexel(volumeTexture, rPosC##_j); \
          half rgbaOpacity##_j = sampleTransferFunction(transferFunctionTexture, float2(s4_dep_##_j.a, 0.5)).a; \
          c##_j = half4(half3(s4_dep_##_j.rgb), rgbaOpacity##_j); \
        } else if (fc_dependentLA) { \
          float4 s4_dep_##_j = sampleVolumeTexel(volumeTexture, rPosC##_j); \
          half4 laColor##_j = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm##_j), 0.5)); \
          half lastMin##_j = half(volumeUniforms.scalarMinComp[1]); \
          half lastMax##_j = half(volumeUniforms.scalarMaxComp[1]); \
          half lastNorm##_j = saturate((half(s4_dep_##_j.g) - lastMin##_j) / max(lastMax##_j - lastMin##_j, 1e-4h)); \
          half laOpacity##_j = sampleTransferFunction(transferFunctionTexture, float2(float(lastNorm##_j), 0.5)).a; \
          c##_j = half4(laColor##_j.rgb, laOpacity##_j); \
        } else { \
          c##_j = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm##_j), 0.5)); \
        } \
        if (!skip##_j) { col##_j = c##_j.rgb; opa##_j = c##_j.a; maskLabel##_j = 0.0h; } \
      } \
      if (!skip##_j && fc_gradientOpacity) { \
        if (opa##_j > 0.0h && maskLabel##_j == 0.0h) { \
          half4 gTmp##_j = computeGradientFast(volumeTexture, rPosC##_j, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor); \
          if (fc_computeNormalFromOpacity) { \
            half4 cached##_j = half4(0.0h); \
            half4 gTmp2##_j = computeScalarAndDensityGradient(volumeTexture, transferFunctionTexture, transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3, rPosC##_j, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor, scalarScale, scalarBias, cached##_j); \
            opa##_j *= sampleGradientOpacity(gradientOpacityTexture, float(gTmp2##_j.w)); \
          } else { \
            opa##_j *= sampleGradientOpacity(gradientOpacityTexture, float(gTmp##_j.w)); \
          } \
        } \
      } \
      if (!skip##_j && fc_shading) { \
        if (opa##_j > 0.02h && maskLabel##_j == 0.0h) { \
          half3 n##_j; \
          if (fc_normalTexture) { \
            half4 ns##_j = half4(normalTexture.sample(sVolume, rPosC##_j, level(0))); \
            n##_j = normalize(ns##_j.xyz * 2.0h - 1.0h); \
          } else if (fc_computeNormalFromOpacity) { \
            n##_j = computeDensityGradientFast(volumeTexture, transferFunctionTexture, transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3, rPosC##_j, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor, 0, scalarScale, scalarBias).xyz; \
          } else { \
            n##_j = computeGradientFast(volumeTexture, rPosC##_j, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor).xyz; \
          } \
          if (lightUniforms != nullptr && !fc_defaultLighting) { \
            col##_j = computeVolumeLighting(col##_j, n##_j, -viewDirHalf, ambientMat, diffuseMat, specularMat, shininessMat, *lightUniforms, volumeUniforms.volumeBoundsMin.xyz + (currentPoint + stepVec * (float)_j) * boundsSize); \
          } else { \
            bool twoSided##_j = (lightUniforms != nullptr && lightUniforms->twoSidedLighting != 0); \
            col##_j = computePhongLightingVolumeFast(col##_j, n##_j, -viewDirHalf, -viewDirHalf, ambientMat, diffuseMat, specularMat, shininessMat, twoSided##_j); \
          } \
        } else if (opa##_j > 0.0h && maskLabel##_j == 0.0h) { \
          col##_j = ambientMat * col##_j; \
        } else if (maskLabel##_j != 0.0h) { \
          col##_j = ambientMat * col##_j; \
        } \
      } \
      if (fc_renderToTexture && haveOpaquePos != nullptr && *haveOpaquePos && opa##_j > 0.0h) { *firstOpaquePos = currentPoint + stepVec * (float)_j; *haveOpaquePos = false; } \
      half w##_j = 1.0h - accumulatedOpacity; \
      accumulatedColor += w##_j * (col##_j * opa##_j); \
      accumulatedOpacity += w##_j * opa##_j; \
    } \
  }

#define MV9_ADVANCE(_W) \
  currentPoint += stepVec * (float)_W; \
  currentT += p.stepSize * (float)_W; \
  texLocalPos += texStep * (float)_W; \
  evalPoint += evalStep * (float)_W; \
  i += _W; \
  if (accumulatedOpacity > kExitAcc) { break; } \
  if (currentT >= p.tTerminateMax) { break; }
      int i = 0;
      const int steps = maxSteps;
      // Largest dispatchable batch width: compile-time fc_fragBatch (register
      // diet) overrides the runtime MaxBatchWidth when set. Dials in §38.10
      // compute trick for the fragment ladder — narrow compile-time widths
      // should shed registers / raise occupancy.
      // Structural fix for M/GL>1 on dense+shaded short rays (NIFTI): wide
      // batches (32) with 6-fetch gradient + pow spill registers and hurt
      // short chords (41 steps SD4, 200 SD0.5) while long sparse DICOM
      // benefits. fc_shading/fc_gradientOpacity pipelines get a narrower
      // compile-time cap (shade 8 for fine SD<1.5, 2 for coarse) so the
      // compiler sheds 8/16/32/48 rungs and occupancy rises (37% -> 56%+
      // class). Static per-PSO, not per-ray adaptive. After TF cull 0.02h
      // 2026-08-29 re-sweep: NIFTI SD0.5 best f8 (9.22 vs 9.91 f4, 3.21 vs 3.76 y),
      // SD4 best f2/def (6.90 vs 7.08 f1) — cull saves 30% shade work so fine
      // long rays now prefer even wider 8 vs pre-cull 4. Lean keeps 16.
      const int shadeCap = fc_fineSD ? 8 : 2;
      const int batchCap = (fc_fragBatch > 0) ? fc_fragBatch
                       : ((fc_shading || fc_gradientOpacity) ? min(shadeCap, max(1, int(volumeUniforms.maxBatchWidth)))
                                     : min(16, max(1, int(volumeUniforms.maxBatchWidth))));
      // Block-summary cache (fc_mmBlocks): persists across batches — position
      // advances only along the ray, so a block-index compare detects every
      // change. State: 0 mixed (per-cell work), 1 all-empty (leap), 2
      // all-solid (batches composite normally; preamble just stops early).
      const float3 invMMDimF9 = 1.0f / mmDimF;
      // §37.18 block size in fine cells (default 8); supers remain fixed
      // 64-cell tiles, so blocks-per-super-line derives from it.
      // §37.20: the per-iteration integer DIVIDES this used to cost real time
      // on walk-heavy chords (+13% axis-z); hoisted here as EXACT fp
      // reciprocals — every legal block size is a power of two, so x*invBs is
      // bit-exact against x/bsI and truncation still matches division.
      const int bsI = max(int(volumeUniforms.mmBlockSizeCells), 1);
      const int bpsI = 64 / bsI;
      const float invBs = 1.0f / float(bsI);
      const float invBps = 1.0f / float(bpsI);
      int3 mv9Blk = int3(-1);
      int mv9BlkState = -1;
      // §35.5 (VTK_METAL_TEST_MM_SUPER -> fc_mmSuper): third occupancy level.
      // Super indices derive from BLOCK indices via integer divide (same
      // lesson as the block level: never from normalized-coordinate
      // products); the texel is sampled at its center; supers tile fine
      // cells [64k, 64k+64) = blocks [8k, 8k+8).
      const float3 mmSbDimF = float3(minMaxSuperTexture.get_width(),
                                     minMaxSuperTexture.get_height(),
                                     minMaxSuperTexture.get_depth());
      const bool useMinMaxSuper = useMinMaxBlocks && fc_mmSuper &&
                                  mmSbDimF.x > 1.0f;
      int3 mv9Sb = int3(-1);
      bool mv9SbEmpty = false;
      // §35.14 streaming consume state (fc_segHop): the per-ray skip segments
      // precomputed by volume_segment_build are pulled from the compacted pool
      // one gap at a time — only (recOff, cnt, idx, gStart, gEnd) stay live, so
      // the register budget that refuted the in-fragment pre-walk (§35.13)
      // stays untouched. segIndexMap arrives PRE-OFFSET to this pixel's entry
      // (the caller adds its pixel id). Overflowed rays carry UINT_MAX and
      // composite every empty sample exactly like a raw march (safe: empties
      // contribute zero).
      uint segRecOff = 0xFFFFFFFFu;
      int segCnt = 0;
      int segIdx = 0;
      int gS = 0;
      int gE = 0;
      if (useMinMax && fc_segHop && !fc_slabMode)
      {
        segRecOff = segIndexMap[0];
        if (segRecOff != 0xFFFFFFFFu)
        {
          segCnt = (int)segPool[segRecOff];
          if (segCnt > 0)
          {
            const uint pr = segPool[segRecOff + 1u];
            gS = (int)(pr >> 16u);
            gE = (int)(pr & 0xFFFFu);
            segIdx = 1;
          }
        }
      }
      while (i < steps)
      {
        // A slab pass whose inherited near-side alpha already exceeds the
        // saturation threshold must contribute nothing (the single-pass march
        // would have latched before its first sample). Checked before the
        // batch dispatch so the first batch is gated too.
        if (fc_slabMode && accumulatedOpacity > kExitAcc) { break; }
        if (i > 0 && currentT >= p.tEnd - 1e-6f) break;
        if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
            any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
          if (seenInBounds) { break; }
          texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
          evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
        } else {
          seenInBounds = true;
        }
        if (useMinMax && fc_segHop && !fc_slabMode)
        {
          // §35.14 consume: integer tests against the precomputed gap list
          // replace the per-batch preamble walk entirely. At most one hop per
          // outer iteration — the loop top re-checks tEnd/bounds/latches, so a
          // hop landing past the ray end exits exactly like the legacy skip.
          if (segIdx < segCnt)
          {
            if (i >= gE)
            {
              ++segIdx;
              if (segIdx < segCnt)
              {
                const uint pr = segPool[segRecOff + 1u + uint(segIdx)];
                gS = (int)(pr >> 16u);
                gE = (int)(pr & 0xFFFFu);
              }
            }
            else if (i >= gS)
            {
              // Entire gap [gS,gE) is empty-lattice: samples there contribute
              // zero (opacity-zero TF), so jumping them is output-safe up to
              // the usual +-1-step fp-landing class.
              const int hopW = gE - i;
              currentPoint += stepVec * (float)hopW;
              currentT += p.stepSize * (float)hopW;
              texLocalPos += texStep * (float)hopW;
              evalPoint += evalStep * (float)hopW;
              i += hopW;
              ++segIdx;
              if (segIdx < segCnt)
              {
                const uint pr = segPool[segRecOff + 1u + uint(segIdx)];
                gS = (int)(pr >> 16u);
                gE = (int)(pr & 0xFFFFu);
              }
              continue;
            }
            // else i < gS: solid terrain until the next gap — dispatch batch.
          }
        }
        else if (useMinMax)
        {
          // §37.19 warp-coherent skip: probe the CURRENT block once per lane;
          // advance all lanes by the warp-minimum leap so straight chords keep
          // their cross-lane slice-lockstep (raw's cache-coalescing advantage).
          // A single dissenting lane zeroes the leap -> legacy walk below.
          // simd_min is called from potentially non-converged active sets
          // (latch/tEnd breaks); inactive-lane contributions only risk
          // disabling a skip (garbage <= 0 falls through), never corrupting
          // output — verified by byte-compare when the feature is on.
          if (volumeUniforms.mmWarpMin > 0.5f && bsI > 0)
          {
            const float3 wp = clamp(evalPoint, float3(0.0), float3(1.0));
            const int3 wb = min(int3(wp * mmDimF) / bsI, int3(mmBlkDimF) - 1);
            const float wv = minMaxBlockTexture.sample(sNearest,
                (float3(wb) + 0.5f) / mmBlkDimF, level(0)).r;
            int myLeap = 0;
            if (wv > 0.75f)
            {
              // All-empty block ahead: distance to its far face along the ray.
              const int3 wLo = wb * bsI;
              const float3 wLoN = float3(wLo) * invMMDimF9;
              const float3 wHiN = float3(min(wLo + bsI, int3(mmDimF))) * invMMDimF9;
              float3 wRem;
              wRem.x = p.rayDir.x > 0.0 ? (wHiN.x - wp.x) : (wp.x - wLoN.x);
              wRem.y = p.rayDir.y > 0.0 ? (wHiN.y - wp.y) : (wp.y - wLoN.y);
              wRem.z = p.rayDir.z > 0.0 ? (wHiN.z - wp.z) : (wp.z - wLoN.z);
              wRem = max(wRem, float3(0.0f));
              float3 wT;
              wT.x = abs(p.rayDir.x) > 1e-5 ? wRem.x / abs(p.rayDir.x) : 1e30;
              wT.y = abs(p.rayDir.y) > 1e-5 ? wRem.y / abs(p.rayDir.y) : 1e30;
              wT.z = abs(p.rayDir.z) > 1e-5 ? wRem.z / abs(p.rayDir.z) : 1e30;
              float wSkip = min(min(wT.x, wT.y), wT.z) + 1e-4;
              myLeap = max(1, (int)ceil(wSkip / p.stepSize));
            }
            const int warpLeap = simd_min(myLeap);
            // §37.19 refinement: acting on tiny mins makes the warp crawl
            // one sample at a time (probe cost per sample >> saved work);
            // require a substantial warp-wide leap, else legacy walk.
            if (warpLeap >= int(volumeUniforms.mmWarpMin))
            {
              const int adv = min(warpLeap, steps - i);
              currentPoint += stepVec * (float)adv;
              currentT += p.stepSize * (float)adv;
              texLocalPos += texStep * (float)adv;
              evalPoint += evalStep * (float)adv;
              i += adv;
              continue;
            }
          }
          int w = 0;
          const int extent = min(48, steps - i);
          // Block-summary state is cached across batches (mv9Blk above).
          while (w < extent)
          {
            float3 mmPos = clamp(evalPoint + evalStep * (float)w, float3(0.0), float3(1.0));
            float3 cellCoord = mmPos * mmDimF;
            if (useMinMaxBlocks)
            {
              // Three-state block summary; indices derive from CELL indices
              // (integer divide — an mmPos*blockDim product disagrees with
              // the kernel tiling wherever fineDim/8 is not integer), and the
              // texel is sampled at its center.
              int3 newBlk = min(int3(cellCoord * invBs), int3(mmBlkDimF) - 1);
              if (any(newBlk != mv9Blk))
              {
                mv9Blk = newBlk;
                float bsv = minMaxBlockTexture.sample(sNearest,
                    (float3(mv9Blk) + 0.5f) / mmBlkDimF, level(0)).r;
                mv9BlkState = bsv > 0.75 ? 1 : (bsv < 0.25 ? 2 : 0);
              }
              if (useMinMaxSuper)
              {
                int3 newSb = min(int3(float3(mv9Blk) * invBps), int3(mmSbDimF) - 1);
                if (any(newSb != mv9Sb))
                {
                  mv9Sb = newSb;
                  float ssv = minMaxSuperTexture.sample(sNearest,
                      (float3(mv9Sb) + 0.5f) / mmSbDimF, level(0)).r;
                  mv9SbEmpty = ssv > 0.5f;
                }
                if (mv9SbEmpty && volumeUniforms.mmLeapLevel > 0.5f)
                {
                  // All-empty super-block: every covered cell is empty, so
                  // leap to its far boundary along the ray in fine-cell units
                  // (supers tile cells [64k, 64k+64)).
                  int3 sbLo = mv9Sb * 64;
                  float3 loN = float3(sbLo) * invMMDimF9;
                  float3 hiN = float3(min(sbLo + 64, int3(mmDimF))) * invMMDimF9;
                  float3 rem;
                  rem.x = p.rayDir.x > 0.0 ? (hiN.x - mmPos.x) : (mmPos.x - loN.x);
                  rem.y = p.rayDir.y > 0.0 ? (hiN.y - mmPos.y) : (mmPos.y - loN.y);
                  rem.z = p.rayDir.z > 0.0 ? (hiN.z - mmPos.z) : (mmPos.z - loN.z);
                  rem = max(rem, float3(0.0f));
                  float3 tToFace;
                  tToFace.x = abs(p.rayDir.x) > 1e-5 ? rem.x / abs(p.rayDir.x) : 1e30;
                  tToFace.y = abs(p.rayDir.y) > 1e-5 ? rem.y / abs(p.rayDir.y) : 1e30;
                  tToFace.z = abs(p.rayDir.z) > 1e-5 ? rem.z / abs(p.rayDir.z) : 1e30;
                  float exactSkip = min(min(tToFace.x, tToFace.y), tToFace.z) + 1e-4;
                  int leapSteps = (int)ceil(exactSkip / p.stepSize);
                  if (leapSteps < 1) leapSteps = 1;
                  w += leapSteps;
                  continue;
                }
              }
              if (mv9BlkState == 1 && volumeUniforms.mmLeapLevel > 1.5f)
              {
                // All-empty block: leap to its far boundary along the ray in
                // fine-cell units (blocks tile cells [8k,8k+8)).
                int3 blkLo = mv9Blk * bsI;
                float3 loN = float3(blkLo) * invMMDimF9;
                float3 hiN = float3(min(blkLo + bsI, int3(mmDimF))) * invMMDimF9;
                float3 rem;
                rem.x = p.rayDir.x > 0.0 ? (hiN.x - mmPos.x) : (mmPos.x - loN.x);
                rem.y = p.rayDir.y > 0.0 ? (hiN.y - mmPos.y) : (mmPos.y - loN.y);
                rem.z = p.rayDir.z > 0.0 ? (hiN.z - mmPos.z) : (mmPos.z - loN.z);
                rem = max(rem, float3(0.0f));
                float3 tToFace;
                tToFace.x = abs(p.rayDir.x) > 1e-5 ? rem.x / abs(p.rayDir.x) : 1e30;
                tToFace.y = abs(p.rayDir.y) > 1e-5 ? rem.y / abs(p.rayDir.y) : 1e30;
                tToFace.z = abs(p.rayDir.z) > 1e-5 ? rem.z / abs(p.rayDir.z) : 1e30;
                float exactSkip = min(min(tToFace.x, tToFace.y), tToFace.z) + 1e-4;
                int leapSteps = (int)ceil(exactSkip / p.stepSize);
                if (leapSteps < 1) leapSteps = 1;
                w += leapSteps;
                continue;
              }
              if (mv9BlkState == 2)
              {
                // All-solid block: every fine fetch inside would end the skip
                // loop anyway — stop preamble skipping and let the batches
                // composite these steps normally. The cached state makes the
                // re-check after each batch nearly free.
                break;
              }
            }
            // §37.15 block-or-nothing (mmBlocksOnly): a mixed block dispatches
            // its batch un-walked. On fragmented axis chords the per-cell walk
            // below pays serialized lattice-tap + boundary-solve work per step
            // for near-zero yield (blocks rarely certify empty there), while
            // raw composites empties branch-free. Dropped skips cover
            // provably-zero samples at unchanged positions, so output is
            // byte-identical to running this walk.
            if (volumeUniforms.mmBlocksOnly > 0.5f) { break; }
            if (minMaxTexture.sample(sNearest, mmPos, level(0)).r <= 0.5) break;
            float3 fractCoord = fract(cellCoord);
            float3 distToEdge;
            distToEdge.x = p.rayDir.x > 0.0 ? (1.0 - fractCoord.x) : fractCoord.x;
            distToEdge.y = p.rayDir.y > 0.0 ? (1.0 - fractCoord.y) : fractCoord.y;
            distToEdge.z = p.rayDir.z > 0.0 ? (1.0 - fractCoord.z) : fractCoord.z;
            distToEdge = mix(distToEdge, float3(1.0), float3(distToEdge <= 1e-5));
            float3 tToEdge;
            tToEdge.x = abs(p.rayDir.x) > 1e-5 ? distToEdge.x / (abs(p.rayDir.x) * mmDimF.x) : 1e30;
            tToEdge.y = abs(p.rayDir.y) > 1e-5 ? distToEdge.y / (abs(p.rayDir.y) * mmDimF.y) : 1e30;
            tToEdge.z = abs(p.rayDir.z) > 1e-5 ? distToEdge.z / (abs(p.rayDir.z) * mmDimF.z) : 1e30;
            float exactSkip = min(min(tToEdge.x, tToEdge.y), tToEdge.z) + 1e-4;
            int cellSteps = (int)ceil(exactSkip / p.stepSize);
            if (cellSteps < 1) cellSteps = 1;
            w += cellSteps;
          }
          if (w >= extent)
          {
            currentPoint += stepVec * (float)extent;
            currentT += p.stepSize * (float)extent;
            texLocalPos += texStep * (float)extent;
            evalPoint += evalStep * (float)extent;
            i += extent;
            continue;
          }
          if (w > 0)
          {
            currentPoint += stepVec * (float)w;
            currentT += p.stepSize * (float)w;
            texLocalPos += texStep * (float)w;
            evalPoint += evalStep * (float)w;
            i += w;
          }
        }
        if (fc_fineSD || (!fc_shading && !fc_gradientOpacity)) {
          // fine SD0.5 or coarse lean (DICOM 16) keep 48..2/1, coarse shade (NIFTI 2) only 2/1 dead-strip 48/32/16/8/4 thr0
          if (batchCap >= 48 && i + 48 <= steps)
          {
            MV9_FETCH(0)
            MV9_FETCH(1)
            MV9_FETCH(2)
            MV9_FETCH(3)
            MV9_FETCH(4)
            MV9_FETCH(5)
            MV9_FETCH(6)
            MV9_FETCH(7)
            MV9_FETCH(8)
            MV9_FETCH(9)
            MV9_FETCH(10)
            MV9_FETCH(11)
            MV9_FETCH(12)
            MV9_FETCH(13)
            MV9_FETCH(14)
            MV9_FETCH(15)
            MV9_FETCH(16)
            MV9_FETCH(17)
            MV9_FETCH(18)
            MV9_FETCH(19)
            MV9_FETCH(20)
            MV9_FETCH(21)
            MV9_FETCH(22)
            MV9_FETCH(23)
            MV9_FETCH(24)
            MV9_FETCH(25)
            MV9_FETCH(26)
            MV9_FETCH(27)
            MV9_FETCH(28)
            MV9_FETCH(29)
            MV9_FETCH(30)
            MV9_FETCH(31)
            MV9_FETCH(32)
            MV9_FETCH(33)
            MV9_FETCH(34)
            MV9_FETCH(35)
            MV9_FETCH(36)
            MV9_FETCH(37)
            MV9_FETCH(38)
            MV9_FETCH(39)
            MV9_FETCH(40)
            MV9_FETCH(41)
            MV9_FETCH(42)
            MV9_FETCH(43)
            MV9_FETCH(44)
            MV9_FETCH(45)
            MV9_FETCH(46)
            MV9_FETCH(47)
            MV9_COMPOSITE(0)
            MV9_COMPOSITE(1)
            MV9_COMPOSITE(2)
            MV9_COMPOSITE(3)
            MV9_COMPOSITE(4)
            MV9_COMPOSITE(5)
            MV9_COMPOSITE(6)
            MV9_COMPOSITE(7)
            MV9_COMPOSITE(8)
            MV9_COMPOSITE(9)
            MV9_COMPOSITE(10)
            MV9_COMPOSITE(11)
            MV9_COMPOSITE(12)
            MV9_COMPOSITE(13)
            MV9_COMPOSITE(14)
            MV9_COMPOSITE(15)
            MV9_COMPOSITE(16)
            MV9_COMPOSITE(17)
            MV9_COMPOSITE(18)
            MV9_COMPOSITE(19)
            MV9_COMPOSITE(20)
            MV9_COMPOSITE(21)
            MV9_COMPOSITE(22)
            MV9_COMPOSITE(23)
            MV9_COMPOSITE(24)
            MV9_COMPOSITE(25)
            MV9_COMPOSITE(26)
            MV9_COMPOSITE(27)
            MV9_COMPOSITE(28)
            MV9_COMPOSITE(29)
            MV9_COMPOSITE(30)
            MV9_COMPOSITE(31)
            MV9_COMPOSITE(32)
            MV9_COMPOSITE(33)
            MV9_COMPOSITE(34)
            MV9_COMPOSITE(35)
            MV9_COMPOSITE(36)
            MV9_COMPOSITE(37)
            MV9_COMPOSITE(38)
            MV9_COMPOSITE(39)
            MV9_COMPOSITE(40)
            MV9_COMPOSITE(41)
            MV9_COMPOSITE(42)
            MV9_COMPOSITE(43)
            MV9_COMPOSITE(44)
            MV9_COMPOSITE(45)
            MV9_COMPOSITE(46)
            MV9_COMPOSITE(47)
            MV9_ADVANCE(48)
          }
          else if (batchCap >= 32 && i + 32 <= steps)
          {
            MV9_FETCH(0)
            MV9_FETCH(1)
            MV9_FETCH(2)
            MV9_FETCH(3)
            MV9_FETCH(4)
            MV9_FETCH(5)
            MV9_FETCH(6)
            MV9_FETCH(7)
            MV9_FETCH(8)
            MV9_FETCH(9)
            MV9_FETCH(10)
            MV9_FETCH(11)
            MV9_FETCH(12)
            MV9_FETCH(13)
            MV9_FETCH(14)
            MV9_FETCH(15)
            MV9_FETCH(16)
            MV9_FETCH(17)
            MV9_FETCH(18)
            MV9_FETCH(19)
            MV9_FETCH(20)
            MV9_FETCH(21)
            MV9_FETCH(22)
            MV9_FETCH(23)
            MV9_FETCH(24)
            MV9_FETCH(25)
            MV9_FETCH(26)
            MV9_FETCH(27)
            MV9_FETCH(28)
            MV9_FETCH(29)
            MV9_FETCH(30)
            MV9_FETCH(31)
            MV9_COMPOSITE(0)
            MV9_COMPOSITE(1)
            MV9_COMPOSITE(2)
            MV9_COMPOSITE(3)
            MV9_COMPOSITE(4)
            MV9_COMPOSITE(5)
            MV9_COMPOSITE(6)
            MV9_COMPOSITE(7)
            MV9_COMPOSITE(8)
            MV9_COMPOSITE(9)
            MV9_COMPOSITE(10)
            MV9_COMPOSITE(11)
            MV9_COMPOSITE(12)
            MV9_COMPOSITE(13)
            MV9_COMPOSITE(14)
            MV9_COMPOSITE(15)
            MV9_COMPOSITE(16)
            MV9_COMPOSITE(17)
            MV9_COMPOSITE(18)
            MV9_COMPOSITE(19)
            MV9_COMPOSITE(20)
            MV9_COMPOSITE(21)
            MV9_COMPOSITE(22)
            MV9_COMPOSITE(23)
            MV9_COMPOSITE(24)
            MV9_COMPOSITE(25)
            MV9_COMPOSITE(26)
            MV9_COMPOSITE(27)
            MV9_COMPOSITE(28)
            MV9_COMPOSITE(29)
            MV9_COMPOSITE(30)
            MV9_COMPOSITE(31)
            MV9_ADVANCE(32)
          }
          else if (batchCap >= 16 && i + 16 <= steps)
          {
            MV9_FETCH(0)
            MV9_FETCH(1)
            MV9_FETCH(2)
            MV9_FETCH(3)
            MV9_FETCH(4)
            MV9_FETCH(5)
            MV9_FETCH(6)
            MV9_FETCH(7)
            MV9_FETCH(8)
            MV9_FETCH(9)
            MV9_FETCH(10)
            MV9_FETCH(11)
            MV9_FETCH(12)
            MV9_FETCH(13)
            MV9_FETCH(14)
            MV9_FETCH(15)
            MV9_COMPOSITE(0)
            MV9_COMPOSITE(1)
            MV9_COMPOSITE(2)
            MV9_COMPOSITE(3)
            MV9_COMPOSITE(4)
            MV9_COMPOSITE(5)
            MV9_COMPOSITE(6)
            MV9_COMPOSITE(7)
            MV9_COMPOSITE(8)
            MV9_COMPOSITE(9)
            MV9_COMPOSITE(10)
            MV9_COMPOSITE(11)
            MV9_COMPOSITE(12)
            MV9_COMPOSITE(13)
            MV9_COMPOSITE(14)
            MV9_COMPOSITE(15)
            MV9_ADVANCE(16)
          }
          else if (batchCap >= 8 && i + 8 <= steps)
          {
            MV9_FETCH(0)
            MV9_FETCH(1)
            MV9_FETCH(2)
            MV9_FETCH(3)
            MV9_FETCH(4)
            MV9_FETCH(5)
            MV9_FETCH(6)
            MV9_FETCH(7)
            MV9_COMPOSITE(0)
            MV9_COMPOSITE(1)
            MV9_COMPOSITE(2)
            MV9_COMPOSITE(3)
            MV9_COMPOSITE(4)
            MV9_COMPOSITE(5)
            MV9_COMPOSITE(6)
            MV9_COMPOSITE(7)
            MV9_ADVANCE(8)
          }
          else if (batchCap >= 4 && i + 4 <= steps)
          {
            MV9_FETCH(0)
            MV9_FETCH(1)
            MV9_FETCH(2)
            MV9_FETCH(3)
            MV9_COMPOSITE(0)
            MV9_COMPOSITE(1)
            MV9_COMPOSITE(2)
            MV9_COMPOSITE(3)
            MV9_ADVANCE(4)
          }
          else if (batchCap >= 2 && i + 2 <= steps)
          {
            MV9_FETCH(0)
            MV9_FETCH(1)
            MV9_COMPOSITE(0)
            MV9_COMPOSITE(1)
            MV9_ADVANCE(2)
          }
          else if (i + 1 <= steps)
          {
            MV9_FETCH(0)
            MV9_COMPOSITE(0)
            MV9_ADVANCE(1)
          }
        } else {
          // coarse SD4: only 2 and 1, dead-strip 48/32/16/8/4 rungs for I$ and registers, thr0
          if (batchCap >= 2 && i + 2 <= steps)
          {
            MV9_FETCH(0)
            MV9_FETCH(1)
            MV9_COMPOSITE(0)
            MV9_COMPOSITE(1)
            MV9_ADVANCE(2)
          }
          else if (i + 1 <= steps)
          {
            MV9_FETCH(0)
            MV9_COMPOSITE(0)
            MV9_ADVANCE(1)
          }
        }
      }
#undef MV9_FETCH
#undef MV9_COMPOSITE
#undef MV9_ADVANCE
    }
    else if (fc_marchVariant == 8 && !fc_minmax && !fc_shading && !fc_gradientOpacity &&
        !fc_renderToTexture)
    {
      // fc_marchVariant 8: harness-style scheduling (minimal_gap/metal_gap.m
      // BuildUnrollBody, PERFORMANCE_INVESTIGATION.md sections 16.3/17). The probe
      // v34 (fragment_march_phase_batch_sched) measured 54ms vs the batch-8
      // array consume's 74ms: computing all positions first, issuing all N
      // volume fetches and all N TF fetches back-to-back (independent, in
      // flight), then the serial composite chain with ONE advance per batch
      // (evalPoint += evalStep * 8, a 1-op loop-carried chain instead of 8
      // serial adds) and ONE break check per batch, plus a scalar tail loop.
      // Gated to the lean compile-time combo (no minmax/shading/gradop/
      // renderToTexture); any other feature combination keeps the batch-8
      // consume below, which is correct for all flags.
      int i = 0;
      const int steps = maxSteps;
      for (; i + unrollN <= steps; i += unrollN)
      {
        if (fc_slabMode && accumulatedOpacity > kExitAcc) { break; }
        if (currentT >= p.tEnd - 1e-6f) break;
        if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
            any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
          if (seenInBounds) { break; }
          texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
          evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
        } else {
          seenInBounds = true;
        }
        float3 p0 = evalPoint;
        float3 p1 = evalPoint + evalStep * 1.0f;
        float3 p2 = evalPoint + evalStep * 2.0f;
        float3 p3 = evalPoint + evalStep * 3.0f;
        float3 p4 = evalPoint + evalStep * 4.0f;
        float3 p5 = evalPoint + evalStep * 5.0f;
        float3 p6 = evalPoint + evalStep * 6.0f;
        float3 p7 = evalPoint + evalStep * 7.0f;
        float s0 = sampleVolumeScalar(volumeTexture, p0);
        float s1 = sampleVolumeScalar(volumeTexture, p1);
        float s2 = sampleVolumeScalar(volumeTexture, p2);
        float s3 = sampleVolumeScalar(volumeTexture, p3);
        float s4 = sampleVolumeScalar(volumeTexture, p4);
        float s5 = sampleVolumeScalar(volumeTexture, p5);
        float s6 = sampleVolumeScalar(volumeTexture, p6);
        float s7 = sampleVolumeScalar(volumeTexture, p7);
        half4 c0 = sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s0) * scalarScale + scalarBias)), 0.5));
        half4 c1 = sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s1) * scalarScale + scalarBias)), 0.5));
        half4 c2 = sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s2) * scalarScale + scalarBias)), 0.5));
        half4 c3 = sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s3) * scalarScale + scalarBias)), 0.5));
        half4 c4 = sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s4) * scalarScale + scalarBias)), 0.5));
        half4 c5 = sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s5) * scalarScale + scalarBias)), 0.5));
        half4 c6 = sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s6) * scalarScale + scalarBias)), 0.5));
        half4 c7 = sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s7) * scalarScale + scalarBias)), 0.5));
        half w0 = 1.0h - accumulatedOpacity;
        accumulatedColor += w0 * (c0.rgb * c0.a);
        accumulatedOpacity += w0 * c0.a;
        half w1 = 1.0h - accumulatedOpacity;
        accumulatedColor += w1 * (c1.rgb * c1.a);
        accumulatedOpacity += w1 * c1.a;
        half w2 = 1.0h - accumulatedOpacity;
        accumulatedColor += w2 * (c2.rgb * c2.a);
        accumulatedOpacity += w2 * c2.a;
        half w3 = 1.0h - accumulatedOpacity;
        accumulatedColor += w3 * (c3.rgb * c3.a);
        accumulatedOpacity += w3 * c3.a;
        half w4 = 1.0h - accumulatedOpacity;
        accumulatedColor += w4 * (c4.rgb * c4.a);
        accumulatedOpacity += w4 * c4.a;
        half w5 = 1.0h - accumulatedOpacity;
        accumulatedColor += w5 * (c5.rgb * c5.a);
        accumulatedOpacity += w5 * c5.a;
        half w6 = 1.0h - accumulatedOpacity;
        accumulatedColor += w6 * (c6.rgb * c6.a);
        accumulatedOpacity += w6 * c6.a;
        half w7 = 1.0h - accumulatedOpacity;
        accumulatedColor += w7 * (c7.rgb * c7.a);
        accumulatedOpacity += w7 * c7.a;
        currentPoint += stepVec * 8.0f;
        currentT += p.stepSize * 8.0f;
        texLocalPos += texStep * 8.0f;
        evalPoint += evalStep * 8.0f;
        if (accumulatedOpacity > kExitAcc) { break; }
        if (currentT >= p.tTerminateMax) { break; }
      }
      for (; i < steps; i++)
      {
        if (fc_slabMode && accumulatedOpacity > kExitAcc) { break; }
        if (currentT >= p.tEnd - 1e-6f) break;
        if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
            any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
          if (seenInBounds) { break; }
          texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
          evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
        } else {
          seenInBounds = true;
        }
        float s = sampleVolumeScalar(volumeTexture, evalPoint);
        half4 c = sampleTransferFunction(transferFunctionTexture, float2(float(saturate(half(s) * scalarScale + scalarBias)), 0.5));
        half w = 1.0h - accumulatedOpacity;
        accumulatedColor += w * (c.rgb * c.a);
        accumulatedOpacity += w * c.a;
        currentPoint += stepVec;
        currentT += p.stepSize;
        texLocalPos += texStep;
        evalPoint += evalStep;
        if (accumulatedOpacity > kExitAcc) { break; }
        if (currentT >= p.tTerminateMax) { break; }
      }
    }
    else
    {
    int i = 0;
    while (i < maxSteps)
    {
      // A slab pass whose inherited near-side alpha already exceeds the
      // saturation threshold contributes nothing; latch instead of breaking so
      // the unrolled samples stay gated by suppressAccum (the first sample of
      // the first batch included).
      if (fc_slabMode && accumulatedOpacity > kExitAcc)
      {
        marchOpaque = true;
      }
      const int nBatch = (maxSteps - i >= unrollN) ? unrollN : (maxSteps - i);
      float bs[8];
      if (nBatch > 0) { bs[0] = sampleVolumeScalar(volumeTexture, evalPoint); }
      if (nBatch > 1) { bs[1] = sampleVolumeScalar(volumeTexture, evalPoint + evalStep); }
      if (nBatch > 2) { bs[2] = sampleVolumeScalar(volumeTexture, evalPoint + 2.0 * evalStep); }
      if (nBatch > 3) { bs[3] = sampleVolumeScalar(volumeTexture, evalPoint + 3.0 * evalStep); }
      if (nBatch > 4) { bs[4] = sampleVolumeScalar(volumeTexture, evalPoint + 4.0 * evalStep); }
      if (nBatch > 5) { bs[5] = sampleVolumeScalar(volumeTexture, evalPoint + 5.0 * evalStep); }
      if (nBatch > 6) { bs[6] = sampleVolumeScalar(volumeTexture, evalPoint + 6.0 * evalStep); }
      if (nBatch > 7) { bs[7] = sampleVolumeScalar(volumeTexture, evalPoint + 7.0 * evalStep); }
      bool refillNeeded = false;
#define PROC_UNROLL_SAMPLE(_j) \
      { \
        prefetchScalar = bs[_j]; \
        prefetchValid = true; \
        bool batchAbort = false; \
        if (currentT >= p.tEnd - 1e-6) { marchDone = true; } \
        if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) || \
            any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) { \
          if (seenInBounds) { marchDone = true; } \
          texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0)); \
          evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset); \
          prefetchValid = false; \
          currentPoint += stepVec; \
          currentT += p.stepSize; \
          texLocalPos += texStep; \
          evalPoint += evalStep; \
          batchAbort = true; \
        } else { \
          seenInBounds = true; \
        } \
        if (useMinMax) { \
          float3 mmPos = clamp(evalPoint, float3(0.0), float3(1.0)); \
          int3 newCell = min(int3(mmPos * mmDimF), int3(mmDimF) - 1); \
          if (any(newCell != curCell)) { \
            curCell = newCell; \
            curCellEmpty = minMaxTexture.sample(sNearest, mmPos, level(0)).r > 0.5; \
          } \
          if (curCellEmpty) { \
            float3 cellCoord = mmPos * mmDimF; \
            float3 fractCoord = fract(cellCoord); \
            float3 distToEdge; \
            distToEdge.x = p.rayDir.x > 0.0 ? (1.0 - fractCoord.x) : fractCoord.x; \
            distToEdge.y = p.rayDir.y > 0.0 ? (1.0 - fractCoord.y) : fractCoord.y; \
            distToEdge.z = p.rayDir.z > 0.0 ? (1.0 - fractCoord.z) : fractCoord.z; \
            distToEdge = mix(distToEdge, float3(1.0), float3(distToEdge <= 1e-5)); \
            float3 tToEdge; \
            tToEdge.x = abs(p.rayDir.x) > 1e-5 ? distToEdge.x / (abs(p.rayDir.x) * mmDimF.x) : 1e30; \
            tToEdge.y = abs(p.rayDir.y) > 1e-5 ? distToEdge.y / (abs(p.rayDir.y) * mmDimF.y) : 1e30; \
            tToEdge.z = abs(p.rayDir.z) > 1e-5 ? distToEdge.z / (abs(p.rayDir.z) * mmDimF.z) : 1e30; \
            float exactSkip = min(min(tToEdge.x, tToEdge.y), tToEdge.z); \
            exactSkip += 1e-4; \
            float skipDist = ceil(exactSkip / p.stepSize) * p.stepSize; \
            skipDist = max(p.stepSize, skipDist); \
            currentPoint += p.rayDir * skipDist; \
            currentT += skipDist; \
            if (p.checkBounds && (any(currentPoint < p.blockMinGlobal - 1e-4) || any(currentPoint > p.blockMaxGlobal + 1e-4) || currentT >= p.tEnd)) { \
              marchDone = true; \
            } \
            texLocalPos = (volumeUniforms.volumeToTexture * \
                float4(volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize, 1.0)).xyz; \
            evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset); \
            prefetchValid = false; \
            curCell = int3(-1); \
            batchAbort = true; \
          } \
        } \
        if (batchAbort) { \
          refillNeeded = true; \
        } else { \
          bool needsFetch = !prefetchValid; \
          float3 rectEvalPoint = evalPoint; \
          if (doRectilinear && (needsFetch || useIndependentPath || fc_dependentRGBA || fc_dependentLA)) { \
            rectEvalPoint = rectilinearSamplePosition(evalPoint, true, rectCoords, volumeUniforms); \
          } \
          float rawScalar = needsFetch ? sampleVolumeScalar(volumeTexture, rectEvalPoint) : prefetchScalar; \
          half scalarNorm = saturate(half(rawScalar) * scalarScale + scalarBias); \
          half4 colorOpacity; \
          half maskLabel = 0.0h; \
          half4 sharedGrad = half4(0.0h); \
          bool sharedGradReady = false; \
          half4 cachedDensityGrad = half4(0.0h); \
          bool densityGradReady = false; \
          colorOpacity = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5)); \
          half sampleOpacity = colorOpacity.a; \
          if (fc_renderToTexture && haveOpaquePos != nullptr && *haveOpaquePos && sampleOpacity > 0.0h) { \
            *firstOpaquePos = currentPoint; \
            *haveOpaquePos = false; \
          } \
          half3 sampleColor = colorOpacity.rgb; \
          half weight = 1.0h - accumulatedOpacity; \
          if (doGradOp && maskLabel == 0.0h) { \
            if (!sharedGradReady) { \
              if (fc_normalTexture) { \
                half4 nrmSample = half4(normalTexture.sample(sVolume, evalPoint, level(0))); \
                sharedGrad = half4(normalize(nrmSample.xyz * 2.0h - 1.0h), nrmSample.w); \
              } else if (fc_computeNormalFromOpacity) { \
                sharedGrad = computeScalarAndDensityGradient(volumeTexture, \
                    transferFunctionTexture, transferFunctionTexture1, \
                    transferFunctionTexture2, transferFunctionTexture3, \
                    evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture, \
                    gradNormFactor, scalarScale, scalarBias, cachedDensityGrad); \
                densityGradReady = true; \
              } else { \
                sharedGrad = computeGradientFast(volumeTexture, evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor); \
              } \
              sharedGradReady = true; \
            } \
            sampleOpacity *= sampleGradientOpacity(gradientOpacityTexture, float(sharedGrad.w)); \
          } \
          if (doShading && maskLabel == 0.0h && sampleOpacity > 0.02h) { \
            half3 normal; \
            if (fc_computeNormalFromOpacity) { \
              if (densityGradReady) { \
                normal = cachedDensityGrad.xyz; \
              } else { \
                normal = computeDensityGradientFast(volumeTexture, \
                    transferFunctionTexture, transferFunctionTexture1, \
                    transferFunctionTexture2, transferFunctionTexture3, \
                    evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture, \
                    gradNormFactor, 0, scalarScale, scalarBias).xyz; \
              } \
            } else { \
              if (!sharedGradReady) { \
                if (fc_normalTexture) { \
                  half4 nrmSample = half4(normalTexture.sample(sVolume, evalPoint, level(0))); \
                  sharedGrad = half4(normalize(nrmSample.xyz * 2.0h - 1.0h), nrmSample.w); \
                } else { \
                  sharedGrad = computeGradientFast(volumeTexture, evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor); \
                } \
                sharedGradReady = true; \
              } \
              normal = sharedGrad.xyz; \
            } \
            if (lightUniforms != nullptr && !fc_defaultLighting) { \
              sampleColor = computeVolumeLighting(sampleColor, normal, -viewDirHalf, \
                  ambientMat, diffuseMat, specularMat, shininessMat, \
                  *lightUniforms, \
                  volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize); \
            } else { \
              bool twoSided = (lightUniforms != nullptr && lightUniforms->twoSidedLighting != 0); \
              sampleColor = computePhongLightingVolumeFast(sampleColor, normal, -viewDirHalf, -viewDirHalf, \
                  ambientMat, diffuseMat, specularMat, shininessMat, twoSided); \
            } \
          } else if (doShading && sampleOpacity > 0.0h) { \
            sampleColor = ambientMat * sampleColor; \
          } \
          const bool suppressAccum = \
            ((fc_marchVariant >= 3 && marchOpaque) || (fc_marchVariant >= 4 && marchDone)); \
          accumulatedColor += suppressAccum ? 0.0h : weight * (sampleColor * sampleOpacity); \
          accumulatedOpacity += suppressAccum ? 0.0h : weight * sampleOpacity; \
          currentPoint += stepVec; \
          currentT += p.stepSize; \
          texLocalPos += texStep; \
          evalPoint += evalStep; \
          if (accumulatedOpacity > kExitAcc) { \
            marchOpaque = true; \
          } \
          if (currentT >= p.tTerminateMax) { \
            marchDone = true; \
          } \
        } \
        i++; \
      }
      if (nBatch > 0) { PROC_UNROLL_SAMPLE(0) }
      if (nBatch > 1 && !refillNeeded) { PROC_UNROLL_SAMPLE(1) }
      if (nBatch > 2 && !refillNeeded) { PROC_UNROLL_SAMPLE(2) }
      if (nBatch > 3 && !refillNeeded) { PROC_UNROLL_SAMPLE(3) }
      if (nBatch > 4 && !refillNeeded) { PROC_UNROLL_SAMPLE(4) }
      if (nBatch > 5 && !refillNeeded) { PROC_UNROLL_SAMPLE(5) }
      if (nBatch > 6 && !refillNeeded) { PROC_UNROLL_SAMPLE(6) }
      if (nBatch > 7 && !refillNeeded) { PROC_UNROLL_SAMPLE(7) }
#undef PROC_UNROLL_SAMPLE
    }
    } // closes else (fc_marchVariant == 8 lean fast path vs batch-8 consume)
  }
  else
  {
  for (int i = 0; i < maxSteps; i++) {
    mmVisits++;
    // A slab pass whose inherited near-side alpha already exceeds the
    // saturation threshold must contribute nothing (the single-pass march
    // would have latched before its first sample). Latch for the non-divergent
    // variants, break for the baseline's divergent march.
    if (fc_slabMode && accumulatedOpacity > kExitAcc)
    {
      if (fc_marchVariant >= 3)
      {
        marchOpaque = true;
      }
      else
      {
        break;
      }
    }
    // Variant 5 hybrid: exits are latched (not broken) during the uniform main
    // phase i < mainSteps, then the loop reverts to baseline per-fragment
    // breaks in the divergent tail so short rays exit as soon as they finish.
    const bool latchExit = (fc_marchVariant == 5) ? (i < mainSteps)
                                                  : (fc_marchVariant >= 4);
    if (latchExit) {
      // Baseline stops accumulating at tEnd via the loop bound
      // maxSteps = ceil((tEnd - firstT)/stepSize); the uniform frame-max loop
      // overshoots it, and the CTP-bounds exit latches marchDone slightly later
      // than tEnd (half-texel-adjusted bounds), so a few extra near-boundary
      // samples would otherwise be composited. Latch here to reproduce the
      // baseline's stop-at-tEnd semantics for both the bounded and unbounded
      // (checkBounds == false, camera-inside) ray cases.
      if (currentT >= p.tEnd - 1e-6) {
        marchDone = true;
      }
    } else if (!p.checkBounds && currentT >= p.tEnd - 1e-6) {
      break;
    }
    // Variant 5 tail phase: rays already finished during the uniform main phase
    // exit immediately instead of grinding through the (possibly long) tail.
    if (!latchExit && (marchDone || marchOpaque)) {
      break;
    }

    // The proxy box spans the axis-aligned bounds of the rotated volume, so
    // rays through its corner regions fall outside the [0,1]^3 texture cube.
    // The OpenGL backend never skips these samples: its clamp-to-edge sampler
    // clamps the coordinate back into the cube and the boundary voxel is
    // composited (grazing rays, rotated-volume corner regions). Replicate that
    // by clamping texLocalPos and sampling the boundary slab. seenInBounds
    // stays false while entry-side samples are outside so the ray keeps
    // marching through rotated-volume corner regions; axis-aligned grazing
    // rays still terminate via the bounds exit below, and once the ray
    // has been inside the cube and left it, stop entirely.
    // OpenGL TerminationImplementation parity (vtkVolumeShaderComposer.h): the
    // loop breaks when the ray lattice position leaves the cell-to-point
    // adjusted bounds along any axis of travel. g_dataPos lives on the
    // g_dirStep lattice (== evalPoint here) in adjusted texture space, so the
    // test is a directional per-axis comparison against [ctpOffset, ctpOffset
    // + ctpScale].
    const float3 adjTexMin = ctpOffset;
    const float3 adjTexMax = ctpOffset + ctpScale;
    if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
        any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
      if (seenInBounds) {
        // Ray has left the cube. Baseline breaks; the non-divergent march
        // latches and keeps the (clamped) fetch unconditional so SIMT lanes
        // stay locked for the uniform frame-max loop. The accumulation below
        // is gated by marchDone, reproducing the baseline's post-break state.
        if (latchExit) {
          marchDone = true;
        } else {
          break;
        }
      }
      texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
      evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
      prefetchValid = false;
      // Position discontinuity: any cached solid-run countdown is invalid.
      solidRun = 0;
    } else {
      seenInBounds = true;
    }

    if (useMinMax) {
      float3 mmPos = clamp(evalPoint, float3(0.0), float3(1.0));
      int3 newCell = min(int3(mmPos * mmDimF), int3(mmDimF) - 1);
      if (any(newCell != curCell)) {
        curCell      = newCell;
        if (useMinMaxBlocks) {
          // Two-level lookup: one coarse summary fetch first; the fine
          // per-cell fetch runs only inside non-empty blocks. Block emptiness
          // is derived from the DILATED fine lattice, so block-empty implies
          // cell-empty and the composited sample set is unchanged.
          // IMPORTANT: derive the block index from the CELL index (integer
          // divide, matching the reduce kernel's tiling) — never from an
          // independent mmPos*blockDims product: when fineDim/8 is not an
          // exact integer (e.g. 897/8) the two mappings disagree almost
          // everywhere along that axis.
          int3 newBlock = min(newCell / 8, int3(mmBlkDimF) - 1);
          if (any(newBlock != curBlock)) {
            curBlock      = newBlock;
            // Sample the block texel at its CENTER: fetching at raw mmPos can
            // let the sampler's normalized->texel rounding pick the neighbor
            // block right at boundaries, wrongly clearing cells whose own
            // block is solid.
            float bsv = minMaxBlockTexture.sample(sNearest,
                (float3(curBlock) + 0.5) / mmBlkDimF, level(0)).r;
            curBlockEmpty = bsv > 0.75;
            curBlockSolid = bsv < 0.25;
          }
          // Block-empty ⇒ every covered fine cell is empty (reduce kernel
          // tiles by CELL indices; see newBlock above): skip the fine fetch.
          curCellEmpty = curBlockEmpty ||
            (minMaxTexture.sample(sNearest, mmPos, level(0)).r > 0.5);
        } else {
          curCellEmpty = minMaxTexture.sample(sNearest, mmPos, level(0)).r > 0.5;
        }
        mmCross++;
      }

      if (curCellEmpty) {
        // Leap to the edge of the current empty CELL — or of the whole empty
        // BLOCK when the summary says every cell in it is empty. Both landings
        // stay on the step lattice and stop at the same first-solid sample.
        float3 skipDims = mmDimF;
        float3 cellCoord = mmPos * skipDims;
        float3 fractCoord = fract(cellCoord);

        float3 distToEdge;
        distToEdge.x = p.rayDir.x > 0.0 ? (1.0 - fractCoord.x) : fractCoord.x;
        distToEdge.y = p.rayDir.y > 0.0 ? (1.0 - fractCoord.y) : fractCoord.y;
        distToEdge.z = p.rayDir.z > 0.0 ? (1.0 - fractCoord.z) : fractCoord.z;
        distToEdge = mix(distToEdge, float3(1.0), float3(distToEdge <= 1e-5));

        float3 tToEdge;
        tToEdge.x = abs(p.rayDir.x) > 1e-5 ? distToEdge.x / (abs(p.rayDir.x) * skipDims.x) : 1e30;
        tToEdge.y = abs(p.rayDir.y) > 1e-5 ? distToEdge.y / (abs(p.rayDir.y) * skipDims.y) : 1e30;
        tToEdge.z = abs(p.rayDir.z) > 1e-5 ? distToEdge.z / (abs(p.rayDir.z) * skipDims.z) : 1e30;

        if (useMinMaxBlocks && curBlockEmpty) {
          // Block leap: target the block boundary at its TRUE normalized
          // position in fine-cell units — blocks tile cells [8k, 8k+8), so
          // their edges sit at 8k/fineDim, NOT at k/blockDim. The fraction
          // space drifts whenever fineDim/blockSize is not an integer
          // (897 cells -> 113 blocks), skipping across thin solid cells.
          const int bsU = max(int(volumeUniforms.mmBlockSizeCells), 1);
          int3 blkLo = curBlock * bsU;
          float3 loN = float3(blkLo) / mmDimF;
          float3 hiN = float3(min(blkLo + bsU, int3(mmDimF))) / mmDimF;
          float3 rem;
          rem.x = p.rayDir.x > 0.0 ? (hiN.x - mmPos.x) : (mmPos.x - loN.x);
          rem.y = p.rayDir.y > 0.0 ? (hiN.y - mmPos.y) : (mmPos.y - loN.y);
          rem.z = p.rayDir.z > 0.0 ? (hiN.z - mmPos.z) : (mmPos.z - loN.z);
          rem = max(rem, float3(0.0));
          tToEdge.x = abs(p.rayDir.x) > 1e-5 ? rem.x / abs(p.rayDir.x) : 1e30;
          tToEdge.y = abs(p.rayDir.y) > 1e-5 ? rem.y / abs(p.rayDir.y) : 1e30;
          tToEdge.z = abs(p.rayDir.z) > 1e-5 ? rem.z / abs(p.rayDir.z) : 1e30;
        }

        float exactSkip = min(min(tToEdge.x, tToEdge.y), tToEdge.z);
        exactSkip += 1e-4;
        float skipDist = ceil(exactSkip / p.stepSize) * p.stepSize;
        skipDist = max(p.stepSize, skipDist);
        mmSkipped += max(1, (int)(skipDist / p.stepSize + 0.5f));

        currentPoint += p.rayDir * skipDist;
        currentT += skipDist;

        if (p.checkBounds && (any(currentPoint < p.blockMinGlobal - 1e-4) || any(currentPoint > p.blockMaxGlobal + 1e-4) || currentT >= p.tEnd)) {
          if (latchExit) {
            marchDone = true;
          } else {
            break;
          }
        }

        // Re-sync the incremental sample position after the empty-cell jump.
        texLocalPos = (volumeUniforms.volumeToTexture *
            float4(volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize, 1.0)).xyz;
        evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
        prefetchValid = false;
        curCell = int3(-1);
        continue;
      }
    }

    bool needsFetch = !prefetchValid;
    // Rectilinear mode remaps every scalar/texel fetch through the per-axis
    // coordinate curves (OpenGL ShadingSingleInput remaps for all component
    // counts). The remapped coordinate is computed at most once per sample and
    // only when a fetch actually needs it: the needsFetch scalar, the
    // independent multi-component texel, and the dependent-RGBA texel all read
    // the same evalPoint, so one walk serves all three. Non-rectilinear inputs
    // keep rectEvalPoint == evalPoint and the branch is a no-op.
    float3 rectEvalPoint = evalPoint;
    if (doRectilinear &&
        (needsFetch || useIndependentPath || fc_dependentRGBA || fc_dependentLA)) {
      rectEvalPoint = rectilinearSamplePosition(evalPoint, true, rectCoords, volumeUniforms);
    }
    float rawScalar = needsFetch
      ? sampleVolumeScalar(volumeTexture, rectEvalPoint)
      : prefetchScalar;
    // Independent multi-component path: the volume texture stores one channel
    // per component (2-comp -> RG, 3-comp -> RGBA with filler alpha), so the
    // per-component raw values must come from the texel's channels. The
    // prefetch cache covers component 0 only, so fetch the full texel here.
    float4 rawScalar4 = float4(rawScalar, 0.0, 0.0, 0.0);
    if (useIndependentPath) {
      if (fc_linearInterpolation) {
        rawScalar4 = volumeTexture.sample(sVolume,
          volumeFetchSwizzle(rectEvalPoint), level(0));
      } else {
        rawScalar4 = volumeTexture.sample(sNearest,
          volumeFetchSwizzle(rectEvalPoint), level(0));
      }
    } else if (fc_dependentRGBA || fc_dependentLA) {
      // 4-component dependent RGBA / 2-component dependent LA: color and
      // opacity come from the raw channels (OpenGL computeColor/computeOpacity
      // RGBA/LA parity), so the full texel is needed regardless of the
      // component-0 prefetch cache.
      rawScalar4 = sampleVolumeTexel(volumeTexture, rectEvalPoint);
    }
    float rawMask = (doMask && needsFetch)
      ? maskTexture.sample(sNearest, evalPoint, level(0)).r
      : prefetchMask;

    // Crop region is tested against the cellToPoint-shifted sample position
    // (evalPoint) to match the OpenGL baseline: GL crops against g_dataPos,
    // which lives in cellToPoint-adjusted texture space, while the crop planes
    // are in plain [0,1] texture space. Testing the unshifted currentPoint here
    // shifts the fence edges by ~half a texel vs the baseline.
    if (doCropping && ((cropBitmask & (1u << computeCropRegion(cropMin, cropMax, evalPoint))) == 0u)) {
      currentPoint += stepVec;
      currentT += p.stepSize;
      texLocalPos += texStep;
      evalPoint += evalStep;
      prefetchValid = false;
      continue;
    }

    // Binary mask: exclude samples whose (scaled/biased) mask value is <= 0,
    // matching the OpenGL backend's g_skip in BinaryMaskImplementation.
    // Label maps (maskType == 0) instead colorize samples in the doMask branch below.
    if (doMask && volumeUniforms.maskType > 0.5) {
      float binMask = rawMask * maskScale + maskBias;
      if (binMask <= 0.0) {
        currentPoint += stepVec;
        currentT += p.stepSize;
        texLocalPos += texStep;
        evalPoint += evalStep;
        prefetchValid = false;
        continue;
      }
    }

    // Uniform-grid blanking: skip the texel if the current or neighboring
    // points/cells (that share this texel) are blanked, matching the OpenGL
    // backend's g_skip logic. Component 0 = point blanking, 1 = cell blanking.
    // Each neighbor is sampled once; both flag components are read from the
    // same RG texel. blankingMode: 1 = cell, 2 = point, 3 = both.
    if (doBlanking) {
      float4 bCur = blankingTexture.sample(sNearest, evalPoint, level(0));
      float4 bXP = blankingTexture.sample(sNearest, evalPoint + float3(blankHalfStep.x, 0.0, 0.0), level(0));
      float4 bXN = blankingTexture.sample(sNearest, evalPoint - float3(blankHalfStep.x, 0.0, 0.0), level(0));
      float4 bYP = blankingTexture.sample(sNearest, evalPoint + float3(0.0, blankHalfStep.y, 0.0), level(0));
      float4 bYN = blankingTexture.sample(sNearest, evalPoint - float3(0.0, blankHalfStep.y, 0.0), level(0));
      float4 bZP = blankingTexture.sample(sNearest, evalPoint + float3(0.0, 0.0, blankHalfStep.z), level(0));
      float4 bZN = blankingTexture.sample(sNearest, evalPoint - float3(0.0, 0.0, blankHalfStep.z), level(0));

      const bool anyPoint = (bCur.x > 0.0 || bXP.x > 0.0 || bXN.x > 0.0 ||
                             bYP.x > 0.0 || bYN.x > 0.0 || bZP.x > 0.0 || bZN.x > 0.0);
      const bool anyCell  = (bCur.y > 0.0 || bXP.y > 0.0 || bXN.y > 0.0 ||
                             bYP.y > 0.0 || bYN.y > 0.0 || bZP.y > 0.0 || bZN.y > 0.0);

      bool blanked = false;
      if (volumeUniforms.blankingMode == 1.0) {
        blanked = anyCell;
      } else if (volumeUniforms.blankingMode == 2.0) {
        blanked = anyPoint;
      } else {
        blanked = (anyCell || anyPoint);
      }
      if (blanked) {
        currentPoint += stepVec;
        currentT += p.stepSize;
        texLocalPos += texStep;
        evalPoint += evalStep;
        prefetchValid = false;
        continue;
      }
    }

    half scalarNorm = saturate(half(rawScalar) * scalarScale + scalarBias);

    // Per-component normalization against each component's own scalar range
    // (OpenGL in_scalarsRange parity); defaults to the single-path norm when
    // the independent path is inactive.
    half scalarNormComp[4] = {scalarNorm, scalarNorm, scalarNorm, scalarNorm};
    // Per-component scalar->normalized scale/bias (OpenGL in_scalarsRange
    // parity), computed here once per sample so the density-gradient shading
    // path reuses them instead of recomputing the reciprocal per lit sample.
    half compScale[4] = {0.0h, 0.0h, 0.0h, 0.0h};
    half compBias[4] = {0.0h, 0.0h, 0.0h, 0.0h};
    if (useIndependentPath) {
      int nComp = min(4, int(volumeUniforms.numComponents));
      for (int c = 0; c < nComp; ++c) {
        half cMin = half(volumeUniforms.scalarMinComp[c]);
        half cRange = max(half(volumeUniforms.scalarMaxComp[c]) - cMin, 1e-4h);
        compScale[c] = 1.0h / cRange;
        compBias[c] = -cMin / cRange;
        float rawComp;
        if (c == 0) rawComp = rawScalar;
        else if (c == 1) rawComp = rawScalar4.g;
        else if (c == 2) rawComp = rawScalar4.b;
        else rawComp = rawScalar4.a;
        scalarNormComp[c] = saturate((half(rawComp) - cMin) / cRange);
      }
    }

    half4 colorOpacity;
    half maskLabel = 0.0h;
    // Gradient shared between the TF_2D gradient y-axis and shading/gradient
    // opacity so it is computed at most once per sample (computeGradientFast
    // is 6 texture fetches).
    half4 sharedGrad = half4(0.0h);
    bool sharedGradReady = false;
    // Opacity-field gradient cached by the gradient-opacity block when
    // ComputeNormalFromOpacity is combined with gradient opacity, so the
    // shading block reuses the shared six-neighbor fetch instead of refetching.
    half4 cachedDensityGrad = half4(0.0h);
    bool densityGradReady = false;

    // Per-component transfer-function results (independent path only).
    half4 compColor[4] = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)};

    // MIP/MinIP only track the scalar extremum; the transfer function is
    // re-sampled once at the end (matching OpenGL, whose MIP/MinIP path never
    // computes g_srcColor inside the march loop). All other modes need the
    // per-sample opacity. The fc_needsPerSampleOpacity test folds away at
    // compile time via the fc_blendMode function constant, so MIP/MinIP
    // pipelines carry no transfer-function fetch in the loop.
    const bool fc_needsPerSampleOpacity =
      (fc_blendMode == 0 || fc_blendMode == 3 || fc_blendMode == 4);
    if (useIndependentPath) {
      if (fc_needsPerSampleOpacity) {
        int nComp = min(4, int(volumeUniforms.numComponents));
        for (int c = 0; c < nComp; ++c) {
          compColor[c] = sampleComponentTransferFunction(
              transferFunctionTexture, transferFunctionTexture1,
              transferFunctionTexture2, transferFunctionTexture3,
              float2(float(scalarNormComp[c]), 0.5), c);
        }
      }
    } else if (fc_needsPerSampleOpacity && doTransfer2D) {
      half secondNorm;
      if (volumeUniforms.transfer2DUseGradient > 0.5) {
        // Legacy TF_2D (no Y-axis array): the second axis is the gradient
        // magnitude, using the same normalization as gradient opacity — this is
        // the OpenGL backend's grad.w, which it feeds to both the gradient
        // opacity table and the 2D transfer function y-coordinate.
        sharedGrad = computeGradientFast(volumeTexture, evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor);
        sharedGradReady = true;
        secondNorm = sharedGrad.w;
      } else {
        secondNorm = saturate(
            half(sampleSecondScalar(transfer2DYAxisTexture, evalPoint)) * secondScale + secondBias);
      }
      colorOpacity = sampleTransferFunction2D(
          transferFunction2DTexture, float2(float(scalarNorm), float(secondNorm)));
    } else if (fc_needsPerSampleOpacity && doMask) {
      float maskVal = rawMask * maskScale + maskBias;
      if (numLabels > 0.0) {
        float label = floor(maskVal + 0.5);
        if (label > 0.0) {
          label = clamp(label, 1.0, numLabels - 1.0);
          maskLabel = half(label);
          float labelY = (label + 0.5) / numLabels;
          colorOpacity = half4(labelMapTransferTexture.sample(sNearest, float2(float(scalarNorm), labelY), level(0)));
        } else {
          colorOpacity = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
        }
      } else {
        colorOpacity = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
      }
    } else if (fc_needsPerSampleOpacity) {
      if (fc_dependentRGBA) {
        // 4-component dependent RGBA: color is the raw RGB channels and opacity
        // comes from the 4th component mapped through the opacity LUT (OpenGL
        // computeColor/computeOpacity RGBA parity: computeColor returns
        // vec4(scalar.xyz, opacity), computeOpacity reads scalar.w). The LUT is
        // built over the last component's scalar range, so the raw normalized
        // fetch (rawScalar4.a) is the table coordinate.
        half rgbaOpacity =
          sampleTransferFunction(transferFunctionTexture, float2(rawScalar4.a, 0.5)).a;
        colorOpacity = half4(half3(rawScalar4.rgb), rgbaOpacity);
      } else if (fc_dependentLA) {
        // 2-component dependent LA: color is the color LUT at the first
        // component's normalized value (scalarNorm, RGB channels) and opacity
        // is the opacity LUT at the LAST component's normalized value (A
        // channel) — OpenGL computeColor/computeOpacity LA parity (color at
        // scalar.x, opacity at scalar.y). The two LUTs share the single RGBA
        // table — RGB over component 0's range, A over the last component's
        // range — so it is sampled at the two different coordinates.
        half4 laColor = sampleTransferFunction(
            transferFunctionTexture, float2(float(scalarNorm), 0.5));
        half lastMin = half(volumeUniforms.scalarMinComp[1]);
        half lastMax = half(volumeUniforms.scalarMaxComp[1]);
        half lastNorm = saturate(
            (half(rawScalar4.g) - lastMin) / max(lastMax - lastMin, 1e-4h));
        half laOpacity = sampleTransferFunction(
            transferFunctionTexture, float2(float(lastNorm), 0.5)).a;
        colorOpacity = half4(laColor.rgb, laOpacity);
      } else {
        colorOpacity = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
      }
    } else {
      colorOpacity = half4(0.0h);
    }

    half sampleOpacity = colorOpacity.a;

    if (useIndependentPath) {
      // Gradient opacity applied per component using that component's own
      // gradient magnitude (OpenGL computeGradientOpacity(gradient, i) parity).
      if (fc_needsPerSampleOpacity && doGradOp) {
        if (!compGradReady) {
          computeGradientsAllComponents(volumeTexture, evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor, compGrad);
          compGradReady = true;
        }
        int nComp = min(4, int(volumeUniforms.numComponents));
        for (int c = 0; c < nComp; ++c) {
          compColor[c].a *= sampleGradientOpacity(gradientOpacityTexture, float(compGrad[c].w));
        }
      }
      // Combined per-sample alpha (OpenGL totalAlpha): weighted sum of the
      // component opacities. Drives the RTT depth and the composite gate.
      half totalAlpha = 0.0h;
      int nComp = min(4, int(volumeUniforms.numComponents));
      for (int c = 0; c < nComp; ++c) {
        if (volumeUniforms.componentWeight[c] <= 0.0) continue;
        totalAlpha += compColor[c].a * half(volumeUniforms.componentWeight[c]);
      }
      sampleOpacity = totalAlpha;
    }

    // Non-composite blend modes: accumulate over every non-skipped sample.
    // MIP/MinIP track the raw scalar (independent of opacity, matching the
    // OpenGL backend). AverageIP/Additive accumulate opacity-weighted scalar.
    // Dead branches are eliminated at compile time via fc_blendMode.
    if (useIndependentPath) {
      if (fc_blendMode == 1) {           // MAXIMUM_INTENSITY_BLEND
        int nComp = min(4, int(volumeUniforms.numComponents));
        for (int c = 0; c < nComp; ++c) {
          if (firstBlendSample || mipMaxScalarComp[c] < scalarNormComp[c]) {
            mipMaxScalarComp[c] = scalarNormComp[c];
          }
        }
        firstBlendSample = false;
      } else if (fc_blendMode == 2) {    // MINIMUM_INTENSITY_BLEND
        int nComp = min(4, int(volumeUniforms.numComponents));
        for (int c = 0; c < nComp; ++c) {
          if (firstBlendSample || minipMinScalarComp[c] > scalarNormComp[c]) {
            minipMinScalarComp[c] = scalarNormComp[c];
          }
        }
        firstBlendSample = false;
      } else if (fc_blendMode == 3) {    // AVERAGE_INTENSITY_BLEND
        int nComp = min(4, int(volumeUniforms.numComponents));
        for (int c = 0; c < nComp; ++c) {
          half intensityNorm =
            half(volumeUniforms.scalarMinComp[c]) +
            (half(volumeUniforms.scalarMaxComp[c]) - half(volumeUniforms.scalarMinComp[c])) * scalarNormComp[c];
          if (intensityNorm >= half(volumeUniforms.averageIPRangeMin) &&
              intensityNorm <= half(volumeUniforms.averageIPRangeMax)) {
            avgBlendSumComp[c] += compColor[c].a * scalarNormComp[c];
            avgBlendCountComp[c]++;
          }
        }
      } else if (fc_blendMode == 4) {    // ADDITIVE_BLEND
        int nComp = min(4, int(volumeUniforms.numComponents));
        for (int c = 0; c < nComp; ++c) {
          additiveSumComp[c] += compColor[c].a * scalarNormComp[c];
        }
      }
    } else if (fc_blendMode == 1) {           // MAXIMUM_INTENSITY_BLEND
      if (firstBlendSample || mipMaxScalar < scalarNorm) {
        mipMaxScalar = scalarNorm;
      }
      firstBlendSample = false;
    } else if (fc_blendMode == 2) {    // MINIMUM_INTENSITY_BLEND
      if (firstBlendSample || minipMinScalar > scalarNorm) {
        minipMinScalar = scalarNorm;
      }
      firstBlendSample = false;
    } else if (fc_blendMode == 3) {    // AVERAGE_INTENSITY_BLEND
      // Intensity in the volume scalar range (native units pre-divided by the
      // normalization factor, matching the averageIPRangeMin/Max uniforms).
      half intensityNorm =
        volumeUniforms.scalarMin + (volumeUniforms.scalarMax - volumeUniforms.scalarMin) * scalarNorm;
      if (intensityNorm >= half(volumeUniforms.averageIPRangeMin) &&
          intensityNorm <= half(volumeUniforms.averageIPRangeMax)) {
        avgBlendSum += sampleOpacity * scalarNorm;
        avgBlendCount++;
      }
    } else if (fc_blendMode == 4) {    // ADDITIVE_BLEND
      additiveSum += sampleOpacity * scalarNorm;
    }
    // RenderToImage depth: record the world position of the first non-skipped
    // sample whose transfer-function opacity is positive (matches the OpenGL
    // backend's l_opaqueFragPos update).
    if (fc_renderToTexture && haveOpaquePos != nullptr && *haveOpaquePos && sampleOpacity > 0.0h) {
      *firstOpaquePos = currentPoint;
      *haveOpaquePos = false;
    }
    // Opacity pre-integration is baked into the transfer function texture
    // on the CPU at TF-build time (matches OpenGL backend).

    if (useIndependentPath) {
      // OpenGL composites every sample with positive opacity (g_srcColor.a > 0.0);
      // the 0.001h gate here would drop low-opacity leading samples (border rays).
      if (fc_blendMode == 0 && sampleOpacity > 0.0h) {
        // OpenGL composite accumulation: per-component colors are combined via
        // g_srcColor = sum(color[i] * weight[i]) and the weighted opacity sum
        // is used for the alpha accumulation (srcBlend = dstAlpha factor),
        // matching vtkVolumeShaderComposer's independent-component loop.
        half3 tmpRGB = half3(0.0h);
        half tmpA = 0.0h;
        int nComp = min(4, int(volumeUniforms.numComponents));
        for (int c = 0; c < nComp; ++c) {
          half w = half(volumeUniforms.componentWeight[c]);
          if (w <= 0.0h) continue;
          half4 cc = compColor[c];
          half3 ccRGB = cc.rgb;
          if (sampleOpacity >= 0.01h && doShading) {
            half3 normal;
            if (fc_computeNormalFromOpacity) {
              normal = computeDensityGradientFast(volumeTexture,
                  transferFunctionTexture, transferFunctionTexture1,
                  transferFunctionTexture2, transferFunctionTexture3,
                  evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture,
                  gradNormFactor, c, compScale[c], compBias[c]).xyz;
            } else if (fc_normalTexture) {
              half4 nrmSample = half4(normalTexture.sample(sVolume, evalPoint, level(0)));
              normal = normalize(nrmSample.xyz * 2.0h - 1.0h);
            } else {
              if (!compGradReady) {
                computeGradientsAllComponents(volumeTexture, evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor, compGrad);
                compGradReady = true;
              }
              normal = compGrad[c].xyz;
            }
            // Per-component material and shininess (OpenGL lightingComponent
            // index parity).
            half3 ambC = half3(volumeUniforms.ambientColorComp[c].rgb);
            half3 difC = half3(volumeUniforms.diffuseColorComp[c].rgb);
            half3 speC = half3(volumeUniforms.specularColorComp[c].rgb);
            half  shiC = half(volumeUniforms.shininessComp[c]);
            if (lightUniforms != nullptr && !fc_defaultLighting) {
              ccRGB = computeVolumeLighting(ccRGB, normal, -viewDirHalf,
                  ambC, difC, speC, shiC,
                  *lightUniforms,
                  volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize);
            } else {
              bool twoSided = (lightUniforms != nullptr && lightUniforms->twoSidedLighting != 0);
              // OpenGL headlight convention: light and view directions are the per-pixel
              // ray direction toward the camera (g_ldir == g_vdir == normalize(cameraPos - vertexPos)).
              ccRGB = computePhongLightingVolumeFast(ccRGB, normal, -viewDirHalf, -viewDirHalf,
                  ambC, difC, speC, shiC, twoSided);
            }
          }
          tmpRGB += ccRGB * cc.a * w;
          tmpA += (cc.a * cc.a) / sampleOpacity;
        }
        half weight = 1.0h - accumulatedOpacity;
      const bool suppressAccum =
        ((fc_marchVariant >= 3 && marchOpaque) || (fc_marchVariant >= 4 && marchDone));
        accumulatedColor += suppressAccum ? 0.0h : weight * tmpRGB;
        accumulatedOpacity += suppressAccum ? 0.0h : weight * tmpA;
      }
    } else if (fc_blendMode == 0 && sampleOpacity > 0.0h) {
      half3 sampleColor = colorOpacity.rgb;
      half weight = 1.0h - accumulatedOpacity;

      // Gradient magnitude for gradient opacity and/or shading, computed at
      // most once per sample (sharedGrad/sharedGradReady). Gradient opacity is
      // applied to the per-sample alpha whenever the property declares it,
      // independent of shading — OpenGL ComputeLightingSingleInput parity
      // (color.a *= computeGradientOpacity(gradient)), which fixes the
      // dependent-component path rendering flat solid color when shading is off.
      if (doGradOp && maskLabel == 0.0h) {
        if (!sharedGradReady) {
          if (fc_normalTexture) {
            half4 nrmSample = half4(normalTexture.sample(sVolume, evalPoint, level(0)));
            sharedGrad = half4(normalize(nrmSample.xyz * 2.0h - 1.0h), nrmSample.w);
          } else if (fc_computeNormalFromOpacity) {
            sharedGrad = computeScalarAndDensityGradient(volumeTexture,
                transferFunctionTexture, transferFunctionTexture1,
                transferFunctionTexture2, transferFunctionTexture3,
                evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture,
                gradNormFactor, scalarScale, scalarBias, cachedDensityGrad);
            densityGradReady = true;
          } else {
            sharedGrad = computeGradientFast(volumeTexture, evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor);
          }
          sharedGradReady = true;
        }
        sampleOpacity *= sampleGradientOpacity(gradientOpacityTexture, float(sharedGrad.w));
      }

      // TF-aware cull: shade only when a>0.02h (FLASH25 foot 0.015@10), rest ambient
      // saves 6 fetches + pow for ~30% dense samples with thr 0.02 keep vs 0.0h
      if (doShading && maskLabel == 0.0h && sampleOpacity > 0.02h) {

        half3 normal;
        if (fc_computeNormalFromOpacity) {
          if (densityGradReady) {
            normal = cachedDensityGrad.xyz;
          } else {
            normal = computeDensityGradientFast(volumeTexture,
                transferFunctionTexture, transferFunctionTexture1,
                transferFunctionTexture2, transferFunctionTexture3,
                evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture,
                gradNormFactor, 0, scalarScale, scalarBias).xyz;
          }
        } else {
          if (!sharedGradReady) {
            if (fc_normalTexture) {
              half4 nrmSample = half4(normalTexture.sample(sVolume, evalPoint, level(0)));
              sharedGrad = half4(normalize(nrmSample.xyz * 2.0h - 1.0h), nrmSample.w);
            } else {
              sharedGrad = computeGradientFast(volumeTexture, evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor);
            }
            sharedGradReady = true;
          }
          normal = sharedGrad.xyz;
        }

        if (lightUniforms != nullptr && !fc_defaultLighting) {
          sampleColor = computeVolumeLighting(sampleColor, normal, -viewDirHalf,
              ambientMat, diffuseMat, specularMat, shininessMat,
              *lightUniforms,
              volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize);
        } else {
          bool twoSided = (lightUniforms != nullptr && lightUniforms->twoSidedLighting != 0);
          // OpenGL headlight convention: light and view directions are the per-pixel
          // ray direction toward the camera (g_ldir == g_vdir == normalize(cameraPos - vertexPos)).
          sampleColor = computePhongLightingVolumeFast(sampleColor, normal, -viewDirHalf, -viewDirHalf,
              ambientMat, diffuseMat, specularMat, shininessMat, twoSided);
        }
      } else if (doShading) {
        sampleColor = ambientMat * sampleColor;
      }

      const bool suppressAccum =
        ((fc_marchVariant >= 3 && marchOpaque) || (fc_marchVariant >= 4 && marchDone));
      accumulatedColor += suppressAccum ? 0.0h : weight * (sampleColor * sampleOpacity);
      accumulatedOpacity += suppressAccum ? 0.0h : weight * sampleOpacity;
    }

    currentPoint += stepVec;
    currentT += p.stepSize;
    texLocalPos += texStep;
    evalPoint += evalStep;

    if (volumeUniforms._padCropFlags[2] < 0.5f && i + 1 < maxSteps) {
      prefetchScalar = sampleVolumeScalar(volumeTexture,
          rectilinearSamplePosition(evalPoint, doRectilinear, rectCoords, volumeUniforms));
      if (doMask) {
        prefetchMask = maskTexture.sample(sNearest, evalPoint, level(0)).r;
      }
      prefetchValid = true;
    }

    if (fc_marchVariant >= 3) {
      // Non-divergent march: latch the opacity threshold instead of breaking so
      // SIMT lanes stay locked (accumulation is gated above by select). The
      // next sample's fetch still happens (pipeline stays full); its
      // accumulation is skipped, which matches the baseline's post-break state.
      if (accumulatedOpacity > kExitAcc) {
        marchOpaque = true;
      }
    } else {
      // OpenGL parity: g_opacityThreshold = 1.0 - 1.0/255.0 (vtkVolumeShaderComposer.h).
      // OpenGL breaks WITHOUT clamping the accumulated opacity (TerminationImplementation
      // in vtkVolumeShaderComposer.h: `g_fragColor.a > g_opacityThreshold`). Clamping here
      // made 1-src.a = 0 at blend time, dropping the background blend term that GL keeps
      // (dst*(1-a), a ~ 0.9969). Keep the raw accumulated opacity for blend parity.
      if (accumulatedOpacity > kExitAcc) {
        break;
      }
    }
    if (currentT >= p.tTerminateMax) {
      if (latchExit) {
        marchDone = true;
      } else {
        break;
      }
    }
    // OpenGL has no block-bounds exit: TerminationImplementation breaks only on
    // the CTP bounds test (g_dataPos vs in_texMin/in_texMax), the opacity
    // threshold, and g_currentT >= g_terminatePointMax. The block-bounds check
    // (currentPoint vs blockMinGlobal/blockMaxGlobal on the separate ray-box
    // lattice) fired one sample BEFORE the CTP test on the evalPoint lattice,
    // making the camera-inside proxy composite one fewer positive-opacity term
    // than GL at exit-boundary pixels. Rely on the CTP test alone for parity.
    // (Loop bound i < maxSteps still caps runaway rays.)
    marchIter = i + 1;
  }
  }

  half4 finalColor;
  if (useIndependentPath) {
    if (fc_blendMode == 1) {   // MAXIMUM_INTENSITY_BLEND
      // Per-component extremum re-sampled through each component's own table
      // and combined by weight (OpenGL ShadingExit parity).
      half3 c = half3(0.0h);
      half a = 0.0h;
      int nComp = min(4, int(volumeUniforms.numComponents));
      for (int i = 0; i < nComp; ++i) {
        half4 t = sampleComponentTransferFunction(
            transferFunctionTexture, transferFunctionTexture1,
            transferFunctionTexture2, transferFunctionTexture3,
            float2(float(mipMaxScalarComp[i]), 0.5), i);
        half w = half(volumeUniforms.componentWeight[i]);
        c += t.rgb * t.a * w;
        a += t.a * w;
      }
      finalColor = half4(c, a);
    } else if (fc_blendMode == 2) {  // MINIMUM_INTENSITY_BLEND
      half3 c = half3(0.0h);
      half a = 0.0h;
      int nComp = min(4, int(volumeUniforms.numComponents));
      for (int i = 0; i < nComp; ++i) {
        half4 t = sampleComponentTransferFunction(
            transferFunctionTexture, transferFunctionTexture1,
            transferFunctionTexture2, transferFunctionTexture3,
            float2(float(minipMinScalarComp[i]), 0.5), i);
        half w = half(volumeUniforms.componentWeight[i]);
        c += t.rgb * t.a * w;
        a += t.a * w;
      }
      finalColor = half4(c, a);
    } else if (fc_blendMode == 3) {  // AVERAGE_INTENSITY_BLEND
      // Per-component in-range average combined by weight. OpenGL discards the
      // fragment when no in-range sample was found; return a fully transparent
      // fragment so the background shows through.
      half avg = 0.0h;
      bool anySample = false;
      int nComp = min(4, int(volumeUniforms.numComponents));
      for (int i = 0; i < nComp; ++i) {
        if (avgBlendCountComp[i] > 0) {
          anySample = true;
          avg += saturate(avgBlendSumComp[i] / half(avgBlendCountComp[i])) *
                 half(volumeUniforms.componentWeight[i]);
        }
      }
      if (!anySample) {
        return half4(0.0h);
      }
      finalColor = half4(avg, avg, avg, 1.0h);
    } else if (fc_blendMode == 4) {  // ADDITIVE_BLEND
      half sum = 0.0h;
      int nComp = min(4, int(volumeUniforms.numComponents));
      for (int i = 0; i < nComp; ++i) {
        sum += additiveSumComp[i] * half(volumeUniforms.componentWeight[i]);
      }
      sum = saturate(sum);
      finalColor = half4(sum, sum, sum, 1.0h);
    } else {
      finalColor = half4(accumulatedColor, accumulatedOpacity);
      if (fc_slabMode)
      {
        // The near-side composite (slabFar) lies behind this pass's samples:
        // it is added unattenuated; the pass's samples already carry the
        // (1 - nearAlpha) weight via the accumulatedOpacity init above, and
        // accumulatedOpacity is the global over-chain alpha of (slabFar over
        // this pass), so it is exactly the composite's alpha.
        finalColor.rgb += slabFar.rgb;
      }
    }
  } else if (fc_blendMode == 1) {   // MAXIMUM_INTENSITY_BLEND
    half4 c = sampleTransferFunction(transferFunctionTexture, float2(float(mipMaxScalar), 0.5));
    finalColor = half4(c.rgb * c.a, c.a);
  } else if (fc_blendMode == 2) {  // MINIMUM_INTENSITY_BLEND
    half4 c = sampleTransferFunction(transferFunctionTexture, float2(float(minipMinScalar), 0.5));
    finalColor = half4(c.rgb * c.a, c.a);
  } else if (fc_blendMode == 3) {  // AVERAGE_INTENSITY_BLEND
    // OpenGL discards the fragment when no in-range sample was found; return a
    // fully transparent fragment so the background shows through.
    if (avgBlendCount == 0) {
      return half4(0.0h);
    }
    half avg = saturate(avgBlendSum / half(avgBlendCount));
    finalColor = half4(avg, avg, avg, 1.0h);
  } else if (fc_blendMode == 4) {  // ADDITIVE_BLEND
    half sum = saturate(additiveSum);
    finalColor = half4(sum, sum, sum, 1.0h);
  } else {
    finalColor = half4(accumulatedColor, accumulatedOpacity);
    if (fc_slabMode)
    {
      finalColor.rgb += slabFar.rgb;
    }
  }

  // Final color window/level (matches OpenGL raycasterfs.glsl finalizeRayCast):
  //   rgb = rgb * in_scale + in_bias * alpha
  half wlScale = half(volumeUniforms.finalColorScale);
  half wlBias = half(volumeUniforms.finalColorBias);
  finalColor.rgb = finalColor.rgb * wlScale + wlBias * finalColor.a;
  if (volumeUniforms._padCropFlags[0] > 0.5f) {
    // TEMP-DIAG minmax walk probe (MM_PROBE -> _padCropFlags[4]): return the
    // baseline-march counters instead of marchIter. Encodes store byte =
    // count/scale (value = count/(scale*255)): visits = R*32, crossings =
    // G*8, skippedSteps = B*32. Caps: 8160 / 2040 / 8160.
    if (volumeUniforms._padCropFlags[4] > 0.5f) {
      return half4(
          half(min(float(mmVisits) * (1.0f / (32.0f * 255.0f)), 1.0f)),
          half(min(float(mmCross) * (1.0f / (8.0f * 255.0f)), 1.0f)),
          half(min(float(mmSkipped) * (1.0f / (32.0f * 255.0f)), 1.0f)), 1.0h);
    }
    // Encode iter count directly: red_channel/255 = marchIter/256, so
    // uint8 red ≈ marchIter. Max representable = 255 iterations.
    // TEMP RG8 debug: green = fract(zf) i.e. fz weight within the slice
    // interval, blue = floor(z) parity * 255 (pair-class tag).
    float zfDbg = saturate(evalPoint.z) * float(volumeTexture.get_depth() * 2) - 0.5f;
    float fzDbg = max(zfDbg - floor(zfDbg), 0.0f);
    float parDbg = (int(clamp(floor(zfDbg), 0.0f, float(volumeTexture.get_depth() * 2 - 1))) & 1) != 0 ? 1.0f : 0.0f;
    return half4(half(float(marchIter) / 256.0f),
                 half(fzDbg),
                 half(parDbg), 1.0h);
  }
  return finalColor;
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
    float3 localPos,
    half3 initialColor,
    half initialOpacity,
    half4 slabFar,
    constant VolumeMapperUniforms& volumeUniforms,
    constant PerBlockData& b,
    texture3d<float> volumeTexture,
    texture2d<float> transferFunctionTexture,
    texture2d<float> transferFunctionTexture1,
    texture2d<float> transferFunctionTexture2,
    texture2d<float> transferFunctionTexture3,
    texture2d<float> transferFunction2DTexture,
    texture3d<float> transfer2DYAxisTexture,
    texture2d<float> depthTexture,
    texture2d<float> gradientOpacityTexture,
    texture3d<float> maskTexture,
    texture2d<float> labelMapTransferTexture,
    texture3d<float> minMaxTexture,
    texture3d<float> minMaxBlockTexture,
    texture3d<float> minMaxSuperTexture,
    texture3d<float> normalTexture,
    texture3d<float> blankingTexture,
    constant packed_float3* rectCoords,
    constant VolumeLightUniforms* lightUniforms,
    device uint* segIndexMap,
    device uint* segPool)
{
  (void)segPool;
  float jSel = (volumeUniforms.useJittering > 0.5
      ? (volumeUniforms.useIGNJitter > 0.5
            ? sampleIGNJitter(screenPos, volumeUniforms.jitterBlockSize)
            : sampleJitterNoise(screenPos, volumeUniforms.viewportSize.y, volumeUniforms.jitterBlockSize))
      : 1.0f);
  // JSCALE probe (_padCropFlags[1], env VTK_METAL_TEST_JSCALE): shrink the
  // per-pixel phase spread toward the coherent j0 lattice via
  // 1 + s*(noise-1), keeping the noise tap/ALU and trip counts identical.
  // s=1 -> native j1; s=0 -> coherent j0-equivalent start phase. The CPU fill
  // always writes this slot (default 1.0); out-of-range values read as 1.0.
  float jScale = volumeUniforms._padCropFlags[1];
  jScale = (jScale >= 0.0f && jScale <= 1.0f) ? jScale : 1.0f;
  float jitter = mix(1.0f, jSel, jScale) * stepSize;
  float tStart = dot(entryPoint - cameraPos, rayDir);
  // OpenGL camera-inside parity (update 22): GL ignores the box/near-plane
  // entry for camera-inside proxy fragments and marches from the interpolated
  // anchor (g_rayOrigin = ip_textureCoords + one step), so far-face fragments
  // start at z>1 and only composite the clamped far slab instead of re-marching
  // the volume from the near plane. setupVolumeRay recomputes the near-plane
  // entry for those same fragments; clamp tStart to the anchor distance whenever
  // the anchor lies beyond the computed entry (the cap anchor coincides with the
  // entry, so cap fragments are unchanged; the fullscreen path passes
  // localPos == entryPoint, so this is a no-op there).
  if (volumeUniforms.useCameraInsideNearClip > 0.5 &&
      volumeUniforms.useParallelProjection < 0.5)
  {
    float tStartAnchor = dot(localPos - cameraPos, rayDir);
    if (tStartAnchor > tStart)
    {
      tStart = tStartAnchor;
    }
  }
  MarchParams p = {cameraPos, rayDir, tStart, totalBoxT, stepSize, jitter, tTerminateMax,
      blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal, true};
  // §35.14: the fragment pre-offsets segIndexMap to this pixel's entry.
  return marchVolumeUnified(p, initialColor, initialOpacity, slabFar,
      volumeUniforms, b, volumeTexture, transferFunctionTexture,
      transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3,
      transferFunction2DTexture, transfer2DYAxisTexture,
      gradientOpacityTexture, maskTexture, labelMapTransferTexture,
      minMaxTexture, minMaxBlockTexture, minMaxSuperTexture, normalTexture, blankingTexture, rectCoords, lightUniforms,
      nullptr, nullptr, segIndexMap, segPool);
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
    texture2d<float> transferFunctionTexture1,
    texture2d<float> transferFunctionTexture2,
    texture2d<float> transferFunctionTexture3,
    texture2d<float> transferFunction2DTexture,
    texture3d<float> transfer2DYAxisTexture,
    texture2d<float> gradientOpacityTexture,
    texture3d<float> maskTexture,
    texture2d<float> labelMapTransferTexture,
    texture3d<float> minMaxTexture,
    texture3d<float> minMaxBlockTexture,
    texture3d<float> minMaxSuperTexture,
    texture3d<float> normalTexture,
    texture3d<float> blankingTexture,
    constant packed_float3* rectCoords,
    constant VolumeLightUniforms* lightUniforms)
{
  float3 zero = float3(0.0);
  float3 one = float3(1.0);
  MarchParams p = {rayOrigin, rayDir, t0, t1, stepSize, jitter, tTerminateMax,
      zero, one, zero, one, false};
  half4 result = marchVolumeUnified(p, accumulatedColor, accumulatedOpacity, half4(0.0h),
      volumeUniforms, b, volumeTexture, transferFunctionTexture,
      transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3,
      transferFunction2DTexture, transfer2DYAxisTexture,
      gradientOpacityTexture, maskTexture, labelMapTransferTexture,
      minMaxTexture, minMaxBlockTexture, minMaxSuperTexture, normalTexture, blankingTexture, rectCoords, lightUniforms,
      nullptr, nullptr, nullptr, nullptr);
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
    texture3d<float> minMaxBlockTexture [[texture(16)]],
    texture3d<float> minMaxSuperTexture [[texture(17)]],
    texture3d<float> normalTexture [[texture(7)]],
    texture2d<float> transferFunction2DTexture [[texture(9)]],
    texture3d<float> transfer2DYAxisTexture [[texture(10)]],
    texture3d<float> blankingTexture [[texture(11)]],
    texture2d<float> transferFunctionTexture1 [[texture(12)]],
    texture2d<float> transferFunctionTexture2 [[texture(13)]],
    texture2d<float> transferFunctionTexture3 [[texture(14)]],
    constant packed_float3* rectCoords [[buffer(5)]],
    constant VolumeLightUniforms& volumeLights [[buffer(4)]],
    device uint* segIndexMap [[buffer(6)]],
    device uint* segPool [[buffer(7)]],
    texture2d<float> slabFeedbackTexture [[texture(15)]]) {

  VolumeFragmentOut output;
  // TEMP PROBE (VTK_METAL_TEST_NOPREFETCH doubles as probe toggle when
  // combined with VTK_METAL_TEST_RG8): render sampleVolumeScalar at fixed
  // z=0.5 over screen UV to validate the volume path decoupled from the
  // march. Investigation-only.
  if (volumeUniforms._padCropFlags[2] > 0.5f)
  {
    float2 uv = (in.position.xy + 0.5) / volumeUniforms.viewportSize;
    float pz = volumeUniforms._padCropFlags[3];
    // TEMP-DIAG (VTK_METAL_TEST_PROBE_RAW): flags[3] < 0 renders the RAW
    // texture plane i=|pz| directly — shows what the GPU texture holds,
    // independent of any swizzle logic.
    if (pz < -0.25f)
    {
      float v = volumeTexture.sample(
        sVolume, float3(0.5f, uv.y, uv.x), level(0)).r;
      output.color = float4(float3(v), 1.0);
      return output;
    }
    if (pz <= 0.0f) pz = 0.5f;
    float v = fc_volRg8 ? sampleVolumeScalarRG8Pair(volumeTexture, float3(uv, pz))
                        : sampleVolumeScalar(volumeTexture, float3(uv, pz));
    output.color = float4(float3(v), 1.0);
    return output;
  }
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  // Camera-inside proxy (useCameraInsideNearClip set): the vertex buffer holds
  // data-space positions (GL parity) so the interpolated in.localPos is a
  // dataset-space anchor. Everything downstream (ray direction, near-clip clamp,
  // marching) uses volume-space [0,1] coordinates, so convert here.
  bool cameraInsideProxy = volumeUniforms.useCameraInsideNearClip > 0.5;
  float3 localPos = in.localPos;
  if (cameraInsideProxy)
  {
    float3 bsz = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
    localPos = (in.localPos - volumeUniforms.volumeBoundsMin.xyz) / bsz;
  }

  // Parallel projection (OpenGL in_projectionDirection parity): cast parallel
  // rays starting on the proxy box along the constant projection direction.
  // Perspective keeps the converging-ray formulation (vertexPos - cameraPos).
  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel)
  {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); return output; }
    rayDir /= dirLength;
  }

  RaySetup s = setupVolumeRay(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal,
      in.position.xy, volumeUniforms.viewportSize, volumeUniforms, depthTexture);
  if (!s.valid) { output.color = float4(0.0); return output; }

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  // Slab passes sample the ping-pong feedback texture (the composite of all
  // previous passes, premultiplied RGBA) at this fragment's screen position.
  // Non-slab pipelines (fc_slabMode == 0) never sample it.
  half4 slabFar = half4(0.0h);
  if (fc_slabMode)
  {
    slabFar = half4(slabFeedbackTexture.sample(sNearest,
        in.position.xy / volumeUniforms.viewportSize));
  }
  half4 _marchResult = marchVolume(s.entryPoint, s.exitPoint, s.totalDist, s.tTerminateMax, rayDir,
      blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal, rayOrigin,
      stepSize, s.totalBoxT, in.position.xy, localPos,
      half3(0.0), 0.0h, slabFar, volumeUniforms, b,
      volumeTexture, transferFunctionTexture, transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3,
      transferFunction2DTexture, transfer2DYAxisTexture,
      depthTexture, gradientOpacityTexture,
      maskTexture, labelMapTransferTexture, minMaxTexture, minMaxBlockTexture, minMaxSuperTexture, normalTexture,
      blankingTexture, rectCoords, &volumeLights,
      // §35.14: pre-offset the segment index map to this pixel's entry.
      segIndexMap + (fc_segHop
          ? uint(in.position.x) + uint(in.position.y) * uint(volumeUniforms.viewportSize.x)
          : 0u),
      segPool);
  output.color = float4(float3(_marchResult.xyz), float(_marchResult.w));
  return output;
}

// Hardware-selection variant of fragment_volume_main (vtkHardwareSelector,
// cell field association): runs the identical ray march but additionally writes
// the picking IDs — {voxel index, prop id, composite index} — into the RGBA32Uint
// color(1) attachment wherever the ray accumulated opacity > 3/255. This is the
// Metal equivalent of the OpenGL backend's PickingActorPassExit shader exit.
// The ids are encoded as value + 1 (vtkMetalHardwareSelector decodes - 1, and
// index 0 means "empty space"), so 1 is added to the voxel and prop indices.
// Fragments below the opacity threshold write 0 and leave the underlying
// polygonal prop's ids untouched.
fragment VolumeSelectionOut fragment_volume_selection_main(
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
    texture3d<float> minMaxBlockTexture [[texture(16)]],
    texture3d<float> minMaxSuperTexture [[texture(17)]],
    texture3d<float> normalTexture [[texture(7)]],
    texture2d<float> transferFunction2DTexture [[texture(9)]],
    texture3d<float> transfer2DYAxisTexture [[texture(10)]],
    texture3d<float> blankingTexture [[texture(11)]],
    texture2d<float> transferFunctionTexture1 [[texture(12)]],
    texture2d<float> transferFunctionTexture2 [[texture(13)]],
    texture2d<float> transferFunctionTexture3 [[texture(14)]],
    constant packed_float3* rectCoords [[buffer(5)]],
    constant VolumeLightUniforms& volumeLights [[buffer(4)]]) {

  VolumeSelectionOut output;
  output.color = float4(0.0);
  output.ids = uint4(0u);
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  // Camera-inside proxy (useCameraInsideNearClip set): the vertex buffer holds
  // data-space positions (GL parity) so the interpolated in.localPos is a
  // dataset-space anchor. Everything downstream (ray direction, near-clip clamp,
  // marching) uses volume-space [0,1] coordinates, so convert here.
  bool cameraInsideProxy = volumeUniforms.useCameraInsideNearClip > 0.5;
  float3 localPos = in.localPos;
  if (cameraInsideProxy)
  {
    float3 bsz = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
    localPos = (in.localPos - volumeUniforms.volumeBoundsMin.xyz) / bsz;
  }

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel)
  {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { return output; }
    rayDir /= dirLength;
  }

  RaySetup s = setupVolumeRay(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal,
      in.position.xy, volumeUniforms.viewportSize, volumeUniforms, depthTexture);
  if (!s.valid) { return output; }

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  half4 _marchResult = marchVolume(s.entryPoint, s.exitPoint, s.totalDist, s.tTerminateMax, rayDir,
      blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal, rayOrigin,
      stepSize, s.totalBoxT, in.position.xy, localPos,
      half3(0.0), 0.0h, half4(0.0h), volumeUniforms, b,
      volumeTexture, transferFunctionTexture, transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3,
      transferFunction2DTexture, transfer2DYAxisTexture,
      depthTexture, gradientOpacityTexture,
      maskTexture, labelMapTransferTexture, minMaxTexture, minMaxBlockTexture, minMaxSuperTexture, normalTexture,
      blankingTexture, rectCoords, &volumeLights, nullptr, nullptr);

  // PickingActorPassExit parity: only fragments that accumulated a certain
  // level of opacity receive a picking id (index 0 is reserved for empty space).
  // PickingActorPassExit parity: only fragments that accumulated a certain
  // level of opacity receive a picking id (index 0 is reserved for empty space).
  output.ids = volumeSelectionIds(s.entryPoint, float(_marchResult.w), volumeUniforms);
  output.color = float4(float3(_marchResult.xyz), float(_marchResult.w));
  return output;
}
// Reconstructs the ray from screen UV (via the precomputed ndcToVolume matrix)
// instead of relying on proxy-geometry vertices.
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
    texture3d<float> minMaxBlockTexture [[texture(16)]],
    texture3d<float> minMaxSuperTexture [[texture(17)]],
    texture3d<float> normalTexture [[texture(7)]],
    texture2d<float> transferFunction2DTexture [[texture(9)]],
    texture3d<float> transfer2DYAxisTexture [[texture(10)]],
    texture3d<float> blankingTexture [[texture(11)]],
    texture2d<float> transferFunctionTexture1 [[texture(12)]],
    texture2d<float> transferFunctionTexture2 [[texture(13)]],
    texture2d<float> transferFunctionTexture3 [[texture(14)]],
    constant packed_float3* rectCoords [[buffer(5)]],
    constant VolumeLightUniforms& volumeLights [[buffer(4)]]) {

  VolumeFragmentOut output;
  // TEMP PROBE (VTK_METAL_TEST_NOPREFETCH doubles as probe toggle when
  // combined with VTK_METAL_TEST_RG8): render sampleVolumeScalar at fixed
  // z=0.5 over screen UV to validate the volume path decoupled from the
  // march. Investigation-only.
  if (volumeUniforms._padCropFlags[2] > 0.5f)
  {
    float2 uv = (in.position.xy + 0.5) / volumeUniforms.viewportSize;
    float pz = volumeUniforms._padCropFlags[3];
    // TEMP-DIAG (VTK_METAL_TEST_PROBE_RAW): flags[3] < 0 renders the RAW
    // texture plane i=|pz| directly — shows what the GPU texture holds,
    // independent of any swizzle logic.
    if (pz < -0.25f)
    {
      float v = volumeTexture.sample(
        sVolume, float3(0.5f, uv.y, uv.x), level(0)).r;
      output.color = float4(float3(v), 1.0);
      return output;
    }
    if (pz <= 0.0f) pz = 0.5f;
    float v = fc_volRg8 ? sampleVolumeScalarRG8Pair(volumeTexture, float3(uv, pz))
                        : sampleVolumeScalar(volumeTexture, float3(uv, pz));
    output.color = float4(float3(v), 1.0);
    return output;
  }
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 rayOrigin = parallel
    ? parallelRayOrigin(in.position.xy, volumeUniforms.viewportSize, volumeUniforms)
    : cameraPos;
  float3 rayDir = parallel
    ? projectionDir(volumeUniforms)
    : reconstructRayDir(in.position.xy, volumeUniforms.viewportSize, volumeUniforms);

  RaySetup s = setupVolumeRay(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal,
      in.position.xy, volumeUniforms.viewportSize, volumeUniforms, depthTexture);
  if (!s.valid) { output.color = float4(0.0); return output; }

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  half4 _marchResult = marchVolume(s.entryPoint, s.exitPoint, s.totalDist, s.tTerminateMax, rayDir,
      blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal, rayOrigin,
      stepSize, s.totalBoxT, in.position.xy, s.entryPoint,
      half3(0.0), 0.0h, half4(0.0h), volumeUniforms, b,
      volumeTexture, transferFunctionTexture, transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3,
      transferFunction2DTexture, transfer2DYAxisTexture,
      depthTexture, gradientOpacityTexture,
      maskTexture, labelMapTransferTexture, minMaxTexture, minMaxBlockTexture, minMaxSuperTexture, normalTexture,
      blankingTexture, rectCoords, &volumeLights, nullptr, nullptr);
  output.color = float4(float3(_marchResult.xyz), float(_marchResult.w));
  return output;
}

// Hardware-selection variant of fragment_volume_fullscreen_main: same ray march
// (camera-inside path) plus picking-id output, mirroring
// fragment_volume_selection_main.
fragment VolumeSelectionOut fragment_volume_fullscreen_selection_main(
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
    texture3d<float> minMaxBlockTexture [[texture(16)]],
    texture3d<float> minMaxSuperTexture [[texture(17)]],
    texture3d<float> normalTexture [[texture(7)]],
    texture2d<float> transferFunction2DTexture [[texture(9)]],
    texture3d<float> transfer2DYAxisTexture [[texture(10)]],
    texture3d<float> blankingTexture [[texture(11)]],
    texture2d<float> transferFunctionTexture1 [[texture(12)]],
    texture2d<float> transferFunctionTexture2 [[texture(13)]],
    texture2d<float> transferFunctionTexture3 [[texture(14)]],
    constant packed_float3* rectCoords [[buffer(5)]],
    constant VolumeLightUniforms& volumeLights [[buffer(4)]]) {

  VolumeSelectionOut output;
  output.color = float4(0.0);
  output.ids = uint4(0u);
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 rayOrigin = parallel
    ? parallelRayOrigin(in.position.xy, volumeUniforms.viewportSize, volumeUniforms)
    : cameraPos;
  float3 rayDir = parallel
    ? projectionDir(volumeUniforms)
    : reconstructRayDir(in.position.xy, volumeUniforms.viewportSize, volumeUniforms);

  RaySetup s = setupVolumeRay(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal,
      in.position.xy, volumeUniforms.viewportSize, volumeUniforms, depthTexture);
  if (!s.valid) { return output; }

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  half4 _marchResult = marchVolume(s.entryPoint, s.exitPoint, s.totalDist, s.tTerminateMax, rayDir,
      blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal, rayOrigin,
      stepSize, s.totalBoxT, in.position.xy, s.entryPoint,
      half3(0.0), 0.0h, half4(0.0h), volumeUniforms, b,
      volumeTexture, transferFunctionTexture, transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3,
      transferFunction2DTexture, transfer2DYAxisTexture,
      depthTexture, gradientOpacityTexture,
      maskTexture, labelMapTransferTexture, minMaxTexture, minMaxBlockTexture, minMaxSuperTexture, normalTexture,
      blankingTexture, rectCoords, &volumeLights, nullptr, nullptr);

  output.ids = volumeSelectionIds(s.entryPoint, float(_marchResult.w), volumeUniforms);
  output.color = float4(float3(_marchResult.xyz), float(_marchResult.w));
  return output;
}

// RenderToImage fragment shader: proxy-geometry ray-cast that additionally
// exports a depth image. color(0) holds the composited color, color(1) holds
// the NDC depth (mapped to [0,1]) of the first sample whose transfer-function
// opacity is positive (1.0 when the ray passes the volume without accumulating
// opacity). Mirrors fragment_volume_main so the render matches the on-screen
// (proxy-geometry) path.
fragment VolumeFragmentOutRTT fragment_volume_rtt_main(
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
    texture3d<float> minMaxBlockTexture [[texture(16)]],
    texture3d<float> minMaxSuperTexture [[texture(17)]],
    texture3d<float> normalTexture [[texture(7)]],
    texture2d<float> transferFunction2DTexture [[texture(9)]],
    texture3d<float> transfer2DYAxisTexture [[texture(10)]],
    texture3d<float> blankingTexture [[texture(11)]],
    texture2d<float> transferFunctionTexture1 [[texture(12)]],
    texture2d<float> transferFunctionTexture2 [[texture(13)]],
    texture2d<float> transferFunctionTexture3 [[texture(14)]],
    constant packed_float3* rectCoords [[buffer(5)]],
    constant VolumeLightUniforms& volumeLights [[buffer(4)]]) {

  VolumeFragmentOutRTT output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  // Camera-inside proxy (useCameraInsideNearClip set): the vertex buffer holds
  // data-space positions (GL parity) so the interpolated in.localPos is a
  // dataset-space anchor. Everything downstream (ray direction, near-clip clamp,
  // marching) uses volume-space [0,1] coordinates, so convert here.
  bool cameraInsideProxy = volumeUniforms.useCameraInsideNearClip > 0.5;
  float3 localPos = in.localPos;
  if (cameraInsideProxy)
  {
    float3 bsz = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
    localPos = (in.localPos - volumeUniforms.volumeBoundsMin.xyz) / bsz;
  }

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel)
  {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { output.color = float4(0.0); output.depth = 1.0; return output; }
    rayDir /= dirLength;
  }

  RaySetup s = setupVolumeRay(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal,
      in.position.xy, volumeUniforms.viewportSize, volumeUniforms, depthTexture);
  if (!s.valid) { output.color = float4(0.0); output.depth = 1.0; return output; }

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float jitter = (volumeUniforms.useJittering > 0.5
      ? (volumeUniforms.useIGNJitter > 0.5
            ? sampleIGNJitter(in.position.xy, volumeUniforms.jitterBlockSize)
            : sampleJitterNoise(in.position.xy, volumeUniforms.viewportSize.y, volumeUniforms.jitterBlockSize))
      : 1.0) * stepSize;
  float tStart = parallel ? 0.0 : dot(s.entryPoint - cameraPos, rayDir);
  MarchParams p = {rayOrigin, rayDir, tStart, s.totalBoxT, stepSize, jitter, s.tTerminateMax,
      blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal, true};

  float3 firstOpaquePos = float3(-1.0);
  bool searching = true;
  if (volumeUniforms.clampDepthToBackface > 0.5) {
    firstOpaquePos = s.entryPoint;
  }

  half4 _marchResult = marchVolumeUnified(p, half3(0.0), 0.0h, half4(0.0h),
      volumeUniforms, b, volumeTexture, transferFunctionTexture,
      transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3,
      transferFunction2DTexture, transfer2DYAxisTexture,
      gradientOpacityTexture, maskTexture, labelMapTransferTexture,
      minMaxTexture, minMaxBlockTexture, minMaxSuperTexture, normalTexture, blankingTexture, rectCoords, &volumeLights,
      &firstOpaquePos, &searching, nullptr, nullptr);

  output.color = float4(float3(_marchResult.xyz), float(_marchResult.w));

  // searching == false means the march recorded the first opaque sample.
  // If no sample accumulated opacity, fall back to the entry point when
  // clamping to the backface is requested, otherwise emit depth 1.0.
  // The march operates in volume-normalized space ([0,1] box); transform the
  // recorded position to world space before projecting (mirrors the OpenGL
  // backend's in_volumeMatrix[0] * in_textureDatasetMatrix[0] chain). The Metal
  // projection maps clip Z to [0,1] (nearz=0, farz=1), which is already the
  // [0,1] depth range expected by GetDepthImage consumers.
  float3 firstOpaqueWorld = volumeUniforms.volumeBoundsMin.xyz +
      firstOpaquePos * (volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz);
  float4 clipPos = volumeUniforms.viewProjection * float4(firstOpaqueWorld, 1.0);
  float ndcZ = clamp(clipPos.z / clipPos.w, 0.0, 1.0);
  if (searching) {
    if (volumeUniforms.clampDepthToBackface > 0.5) {
      output.depth = ndcZ;
    } else {
      output.depth = 1.0;
    }
  } else {
    output.depth = ndcZ;
  }
  return output;
}


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
    texture3d<float> minMaxBlockTexture [[texture(16)]],
    texture3d<float> minMaxSuperTexture [[texture(17)]],
    texture3d<float> normalTexture [[texture(7)]],
    texture3d<float> brickOccupancy [[texture(8)]],
    texture2d<float> transferFunction2DTexture [[texture(9)]],
    texture3d<float> transfer2DYAxisTexture [[texture(10)]],
    texture3d<float> blankingTexture [[texture(11)]],
    texture2d<float> transferFunctionTexture1 [[texture(12)]],
    texture2d<float> transferFunctionTexture2 [[texture(13)]],
    texture2d<float> transferFunctionTexture3 [[texture(14)]],
    constant packed_float3* rectCoords [[buffer(5)]],
    constant VolumeLightUniforms& volumeLights [[buffer(4)]])
{
    VolumeFragmentOut output;

    float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;

    // Parallel projection: origin on the view plane, constant direction.
    // Perspective: converging rays from the camera position.
    bool parallel = volumeUniforms.useParallelProjection > 0.5;
    float3 rayOrigin = parallel
      ? parallelRayOrigin(in.position.xy, volumeUniforms.viewportSize, volumeUniforms)
      : cameraPos;

    // Reconstruct ray in normalized [0,1] volume space
    float3 rayDir = parallel
      ? projectionDir(volumeUniforms)
      : reconstructRayDir(in.position.xy, volumeUniforms.viewportSize, volumeUniforms);

    // Intersect with full volume bounds [0,1]
    float3 volMin = float3(0.0);
    float3 volMax = float3(1.0);
    float2 tVol = intersectBox(rayOrigin, rayDir, volMin, volMax);
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

        float3 entryPoint = rayOrigin + rayDir * tStart;
        float3 exitPoint = rayOrigin + rayDir * tEnd;

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
        tStart = dot(entryPoint - rayOrigin, rayDir);
        tEnd = tStart + length(d);
    }

    // Depth occlusion termination
    float tTerminateMax = 1e30;
    if (volumeUniforms.useDepthTexture > 0.5) {
        float2 uv = in.position.xy / volumeUniforms.viewportSize;
        float depthSample = depthTexture.sample(sNearest, uv).r;
        if (depthSample < 1.0) {
            float2 ndc = uv * 2.0 - 1.0;
            float4 worldTermination = volumeUniforms.ndcToVolume * float4(ndc.x, -ndc.y, depthSample, 1.0);
            float3 terminationLocal = worldTermination.xyz / worldTermination.w;
            float tDepth = dot(terminationLocal - (rayOrigin + rayDir * tStart), rayDir);
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

    // Global sample schedule. Jitter must use the same per-pixel coordinate as
    // the fullscreen/RTT passes (in.position.xy == GL gl_FragCoord.xy) so the
    // sample lattice has the same phase in grid traversal and the composite
    // passes; the previous +0.5 half-pixel shift desynchronized the grid pass
    // from the rest and from GL's in_noiseSampler(gl_FragCoord.xy) sampling.
    float stepSize = physicalSampleStep(rayDir, volumeUniforms);
    float jitter = volumeUniforms.useJittering > 0.5
        ? (volumeUniforms.useIGNJitter > 0.5
              ? sampleIGNJitter(in.position.xy + float2(0.5, 0.5), volumeUniforms.jitterBlockSize) * stepSize
              : sampleJitterNoise(in.position.xy, volumeUniforms.viewportSize.y, volumeUniforms.jitterBlockSize) * stepSize)
        : 1.0 * stepSize;

    // Grid traversal loop
    half3 color = 0.0h;
    half opacity = 0.0h;

    int3 gridDims = int3(grid.gridDimsX, grid.gridDimsY, grid.gridDimsZ);
    float tEndRel = tEnd - tStart;
    GridWalker walker = initGridWalker(rayOrigin, rayDir, tStart, tEndRel, gridDims);

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
                    rayOrigin, rayDir,
                    segmentT0, segmentT1,
                    stepSize, jitter, tTerminateMax,
                    color, opacity,
                    volumeUniforms, b,
                    volumeTexture, transferFunctionTexture,
                    transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3,
                    transferFunction2DTexture, transfer2DYAxisTexture,
                    gradientOpacityTexture, maskTexture, labelMapTransferTexture,
                    minMaxTexture, minMaxBlockTexture, minMaxSuperTexture, normalTexture, blankingTexture, rectCoords,
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
  int volTransposed;              // VTK_METAL_TEST_TRANSPOSE: sample coords map via .zyx
  float opacityLut[256];          // §33.2 item 2 (VTK_METAL_TEST_MM_EPS): the TF
                                  // opacity table so emptiness can use a max-
                                  // achievable-opacity threshold instead of exact zero
  float mmEps;                    // emptiness threshold (0 = exact prefix semantics)
  uint  _pad2[3];
};

// §35.14 async segment pre-pass, stage 1 of 3 (VTK_METAL_TEST_MM_SEG=1):
// rasterized ray atlas. Runs the fragment_volume_main prologue + the
// marchVolume jitter/tStart block + the marchVolumeUnified setup head VERBATIM
// on the same interpolated varyings (same geometry, same vertex function), so
// every value below is bit-identical to what the main pass would compute, and
// writes the march setup per pixel: A = (evalPoint.xyz, steps),
// B = (evalStep.xyz, stepSize), C = (rayDir.xyz). The volume_segment_build
// kernel turns these into per-ray skip-segment lists; the mv9 march consumes
// them with integer tests.
// steps = 0 marks pixels the builder should skip (invalid ray, minmax off, or
// unsupported slab mode).
struct VolumeAtlasOut
{
  float4 setupA [[color(0)]];   // (evalPoint.xyz, steps)
  float4 setupB [[color(1)]];   // (evalStep.xyz, stepSize)
  float4 setupC [[color(2)]];   // (rayDir.xyz, unused)
};

fragment VolumeAtlasOut fragment_volume_ray_atlas(
    VolumeVertexOut in [[stage_in]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(0)]],
    texture2d<float> depthTexture [[texture(2)]]) {

  VolumeAtlasOut output;
  output.setupA = float4(0.0);
  output.setupB = float4(0.0);
  output.setupC = float4(0.0);

  // ---- fragment_volume_main prologue (verbatim) ----
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  bool cameraInsideProxy = volumeUniforms.useCameraInsideNearClip > 0.5;
  float3 localPos = in.localPos;
  if (cameraInsideProxy)
  {
    float3 bsz = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
    localPos = (in.localPos - volumeUniforms.volumeBoundsMin.xyz) / bsz;
  }

  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  float3 rayOrigin = parallel ? localPos : cameraPos;
  float3 rayDir = parallel ? projectionDir(volumeUniforms) : (localPos - cameraPos);
  if (!parallel)
  {
    float dirLength = length(rayDir);
    if (dirLength < 0.0001) { return output; }
    rayDir /= dirLength;
  }

  RaySetup s = setupVolumeRay(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal,
      in.position.xy, volumeUniforms.viewportSize, volumeUniforms, depthTexture);
  if (!s.valid) { return output; }

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);

  // ---- marchVolume wrapper body up to MarchParams (verbatim) ----
  float jSel = (volumeUniforms.useJittering > 0.5
      ? (volumeUniforms.useIGNJitter > 0.5
            ? sampleIGNJitter(in.position.xy, volumeUniforms.jitterBlockSize)
            : sampleJitterNoise(in.position.xy, volumeUniforms.viewportSize.y, volumeUniforms.jitterBlockSize))
      : 1.0f);
  float jScale = volumeUniforms._padCropFlags[1];
  jScale = (jScale >= 0.0f && jScale <= 1.0f) ? jScale : 1.0f;
  float jitter = mix(1.0f, jSel, jScale) * stepSize;
  float tStart = dot(s.entryPoint - cameraPos, rayDir);
  if (volumeUniforms.useCameraInsideNearClip > 0.5 &&
      volumeUniforms.useParallelProjection < 0.5)
  {
    float tStartAnchor = dot(localPos - cameraPos, rayDir);
    if (tStartAnchor > tStart)
    {
      tStart = tStartAnchor;
    }
  }
  MarchParams p = {cameraPos, rayDir, tStart, s.totalBoxT, stepSize, jitter, s.tTerminateMax,
      blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal, true};

  // ---- marchVolumeUnified setup head (verbatim) ----
  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz
                        - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(p.rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * p.stepSize;
  float texelCountZ = volumeTexture.get_depth() *
    ((fc_volRg8 && !fc_volTransposed) ? 2.0f : 1.0f);
  float3 texelCount = fc_volTransposedY
    ? float3(volumeTexture.get_width(), volumeTexture.get_depth(),
             volumeTexture.get_height())
    : fc_volTransposed
      ? float3(volumeTexture.get_depth(), volumeTexture.get_height(),
               volumeTexture.get_width())
      : float3(volumeTexture.get_width(), volumeTexture.get_height(), texelCountZ);
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float firstT = p.checkBounds
      ? p.jitter
      : p.jitter + ceil((p.tStart - p.jitter) / p.stepSize) * p.stepSize;
  float3 currentPoint = p.rayOrigin + p.rayDir * (p.checkBounds ? p.tStart : 0.0)
                      + p.rayDir * firstT;
  float currentT = firstT;

  int maxSteps = max(1, int(ceil((p.tEnd - firstT) / p.stepSize)));
  int mainSteps = 0;
  if (fc_marchVariant == 4 && volumeUniforms.maxStepsFrame > 0.5)
  {
    maxSteps = int(volumeUniforms.maxStepsFrame);
  }
  else if (fc_marchVariant == 5 && volumeUniforms.maxStepsFrame > 0.5)
  {
    mainSteps = int(volumeUniforms.maxStepsFrame);
    maxSteps = max(maxSteps, mainSteps);
  }
  else if (volumeUniforms.maxStepsFrame > 0.5)
  {
    maxSteps = min(maxSteps, int(volumeUniforms.maxStepsFrame));
  }

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize, 1.0)).xyz;
  float3 evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);

  const bool useMinMax = fc_minmax &&
    b.minMaxInfo.x > 0.5 &&
    b.minMaxInfo.y > 0.5 &&
    b.minMaxInfo.z > 0.5 &&
    b.minMaxInfo.w > 0.5;
  // Slab passes re-partition [0,maxSteps) per pass — unsupported by the v1
  // segment design; the sentinel makes the builder emit a no-gap record.
  const float stepsOut = (fc_slabMode) ? 0.0f : (float)maxSteps;
  output.setupA = float4(evalPoint, stepsOut);
  output.setupB = float4(evalStep, p.stepSize);
  output.setupC = float4(p.rayDir, s.tTerminateMax);
  return output;
}

struct SynthRay {
  float3 evalPoint;
  float3 evalStep;
  float  stepSize;
  float3 rayDir;
  float  tTerminateMax;
  int    steps;
};

inline bool synthesizeAtlasRay(
    uint2 gid,
    constant VolumeMapperUniforms& volumeUniforms,
    constant PerBlockData& b,
    texture3d<float> volumeTexture,
    texture2d<float> depthTexture,
    thread SynthRay& o)
{
  const float2 screenPos = float2(gid) + 0.5;

  // ---- fragment_volume_main prologue (verbatim, minus interpolated
  // localPos — reconstructed analytically below; box faces are planar so the
  // difference vs rasterizer interpolation is ulp-class) ----
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  bool parallel = volumeUniforms.useParallelProjection > 0.5;
  // Camera-inside normalization relies on the rasterized surface point;
  // unsupported by synthesis v1 (bench cameras are outside-proxy).
  if (volumeUniforms.useCameraInsideNearClip > 0.5) return false;

  float3 rayOrigin, rayDir;
  if (parallel)
  {
    rayDir = projectionDir(volumeUniforms);
    rayOrigin = parallelRayOrigin(screenPos, volumeUniforms.viewportSize,
                                  volumeUniforms);
  }
  else
  {
    rayOrigin = cameraPos;
    rayDir = reconstructRayDir(screenPos, volumeUniforms.viewportSize,
                               volumeUniforms);
  }

  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal,
                      blockMaxGlobal, texMinGlobal, texMaxGlobal);

  RaySetup s = setupVolumeRay(rayOrigin, rayDir, blockMinGlobal,
      blockMaxGlobal, screenPos, volumeUniforms.viewportSize, volumeUniforms,
      depthTexture);
  if (!s.valid) return false;

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);

  // ---- marchVolume wrapper body up to MarchParams (verbatim) ----
  float jSel = (volumeUniforms.useJittering > 0.5
      ? (volumeUniforms.useIGNJitter > 0.5
            ? sampleIGNJitter(screenPos, volumeUniforms.jitterBlockSize)
            : sampleJitterNoise(screenPos, volumeUniforms.viewportSize.y,
                                volumeUniforms.jitterBlockSize))
      : 1.0f);
  float jScale = volumeUniforms._padCropFlags[1];
  jScale = (jScale >= 0.0f && jScale <= 1.0f) ? jScale : 1.0f;
  float jitter = mix(1.0f, jSel, jScale) * stepSize;
  float tStart = dot(s.entryPoint - cameraPos, rayDir);

  // ---- marchVolumeUnified setup head (verbatim) ----
  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz
                        - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal =
    (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;
  float texelCountZ = volumeTexture.get_depth() *
    ((fc_volRg8 && !fc_volTransposed) ? 2.0f : 1.0f);
  float3 texelCount = fc_volTransposedY
    ? float3(volumeTexture.get_width(), volumeTexture.get_depth(),
             volumeTexture.get_height())
    : fc_volTransposed
      ? float3(volumeTexture.get_depth(), volumeTexture.get_height(),
               volumeTexture.get_width())
      : float3(volumeTexture.get_width(), volumeTexture.get_height(),
               texelCountZ);
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;

  float firstT = jitter;   // checkBounds=true branch
  float3 currentPoint = rayOrigin + rayDir * tStart + rayDir * firstT;
  int maxSteps = max(1, int(ceil((s.totalBoxT - firstT) / stepSize)));
  if (volumeUniforms.maxStepsFrame > 0.5)
  {
    maxSteps = min(maxSteps, int(volumeUniforms.maxStepsFrame));
  }

  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize,
             1.0)).xyz;
  o.evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
  o.evalStep = evalStep;
  o.stepSize = stepSize;
  o.rayDir = rayDir;
  o.tTerminateMax = s.tTerminateMax;
  o.steps = maxSteps;
  return true;
}

kernel void volume_segment_build(
    texture2d<float, access::read> segAtlasA [[texture(0)]],
    texture2d<float, access::read> segAtlasB [[texture(1)]],
    texture2d<float, access::read> segAtlasC [[texture(3)]],
    texture3d<float> minMaxTexture [[texture(2)]],
    device uint* segIndexMap [[buffer(0)]],
    device atomic_uint* poolCounter [[buffer(1)]],
    device uint* pool [[buffer(2)]],
    constant uint4& segMeta [[buffer(3)]],   // x=width y=height z=poolCapWords w=maxGaps
    constant PerBlockData& b [[buffer(4)]],
    // §38.17 compute-port inputs: synth mode rebuilds the per-ray setup
    // analytically instead of reading the atlas (CM_SYNTH has no raster).
    texture3d<float> volumeTexture [[texture(4)]],
    texture2d<float> depthSynthTexture [[texture(5)]],
    texture3d<float> minMaxBlockTexture [[texture(6)]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(5)]],
    constant uint& buildFlags [[buffer(6)]],  // bit0: synth input, bit1: blocks
    device uint* dbgOut [[buffer(7)]],        // MM_SEG_DEBUG mirror
    uint2 gid [[thread_position_in_grid]])
{
  if ((int)gid.x >= (int)segMeta.x || (int)gid.y >= (int)segMeta.y) return;
  const uint fpid = gid.x + gid.y * segMeta.x;
  const bool dbgMirror = (buildFlags & 2u) != 0u;
  // §38.17 MM_SEG_DEBUG mirror scratch (written inside the active walk,
  // emitted after the pool claim).
  uint dbgGuard = 0u, dbgEpZ = 0u, dbgEsz = 0u, dbgSs = 0u;
  uint dbgGaps[8] = { 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u };
  uint dbgUse = 0u;

  const bool synthIn = (buildFlags & 1u) != 0u;
  float4 A, B, C;
  int stepsPre;
  if (synthIn)
  {
    SynthRay so;
    if (!synthesizeAtlasRay(gid, volumeUniforms, b, volumeTexture,
                            depthSynthTexture, so))
    {
      segIndexMap[fpid] = 0xFFFFFFFFu;
      return;
    }
    A = float4(so.evalPoint, (float)so.steps);
    B = float4(so.evalStep, so.stepSize);
    C = float4(so.rayDir, so.tTerminateMax);
    stepsPre = so.steps;
  }
  else
  {
    A = segAtlasA.read(gid);
    B = segAtlasB.read(gid);
    C = segAtlasC.read(gid);
    stepsPre = (int)A.w;
  }
  const int steps = stepsPre;
  const float4 mmInfo = b.minMaxInfo;
  const bool active = steps > 0 &&
      mmInfo.x > 0.5f && mmInfo.y > 0.5f && mmInfo.z > 0.5f && mmInfo.w > 0.5f;

  // Gap storage: four u32 registers packing two u16 pairs each (start<<16|end),
  // appended via static select chains — a dynamically indexed local array here
  // spills to device memory per append and dominates the kernel cost.
  uint4 segR0 = 0, segR1 = 0, segR2 = 0, segR3 = 0;
  uint cnt = 0;
  if (active)
  {
    const float3 ep = A.xyz;   // evalPoint at step 0 ([0,1] cube space)
    const float3 es = B.xyz;   // evalStep (cube-space advance per sample)
    const float ss = B.w;      // p.stepSize
    const float3 rd = C.xyz;   // p.rayDir (normalized volume space)
    const float3 mmDimF = mmInfo.yzw;
    // §38.17 stage 2: block-level leaps (buildFlags bit1) — same decisions as
    // the marcher preamble's block walk: on entering a block, probe it once;
    // all-empty ⇒ leap to the far block face (recording the skipped span as a
    // gap); mixed/solid ⇒ fall through to the fine cell walk.
    const bool useBlocks = (buildFlags & 2u) != 0u;
    const int bsI = max(int(volumeUniforms.mmBlockSizeCells), 1);
    const float invBs = 1.0f / float(bsI);
    const float3 mmBlkDimF = float3(minMaxBlockTexture.get_width(),
                                    minMaxBlockTexture.get_height(),
                                    minMaxBlockTexture.get_depth());
    int3 curBlk = int3(-1);
    int w = 0;
    // Safety valve: the leap chain always advances >=1 step, so this bound is
    // unreachable in practice; it guarantees kernel termination (and thus no
    // GPU-watchdog involvement) even under pathological fp conditions.
    int guard = 8192;
    while (w < steps && --guard >= 0)
    {
      float3 mp = clamp(ep + es * (float)w, float3(0.0), float3(1.0));
      float3 cellCoord = mp * mmDimF;
      if (useBlocks)
      {
        int3 newBlk = min(int3(cellCoord * invBs), int3(mmBlkDimF) - 1);
        if (any(newBlk != curBlk))
        {
          curBlk = newBlk;
          const float bsv = minMaxBlockTexture.sample(sNearest,
              (float3(curBlk) + 0.5f) / mmBlkDimF, level(0)).r;
          if (bsv > 0.75f)
          {
            // All-empty block: leap to its far face along the ray (preamble
            // state==1 math), recording the skipped span.
            const float3 loN = float3(curBlk) * invBs;
            const float3 hiN = min(float3(curBlk + 1) * invBs, float3(1.0));
            float3 rem;
            rem.x = rd.x > 0.0 ? (hiN.x - mp.x) : (mp.x - loN.x);
            rem.y = rd.y > 0.0 ? (hiN.y - mp.y) : (mp.y - loN.y);
            rem.z = rd.z > 0.0 ? (hiN.z - mp.z) : (mp.z - loN.z);
            rem = max(rem, float3(0.0f));
            float3 tToFace;
            tToFace.x = abs(rd.x) > 1e-5 ? rem.x / abs(rd.x) : 1e30;
            tToFace.y = abs(rd.y) > 1e-5 ? rem.y / abs(rd.y) : 1e30;
            tToFace.z = abs(rd.z) > 1e-5 ? rem.z / abs(rd.z) : 1e30;
            float exactSkip = min(min(tToFace.x, tToFace.y), tToFace.z) + 1e-4;
            int leapSteps = (int)ceil(exactSkip / ss);
            if (leapSteps < 1) leapSteps = 1;
            if (cnt < segMeta.w)
            {
              const uint pr = ((uint)w << 16u) |
                              (uint)min(w + leapSteps, steps);
              if (cnt < 1u)       segR0.x = pr;
              else if (cnt < 2u)  segR0.y = pr;
              else if (cnt < 3u)  segR0.z = pr;
              else if (cnt < 4u)  segR0.w = pr;
              else if (cnt < 5u)  segR1.x = pr;
              else if (cnt < 6u)  segR1.y = pr;
              else if (cnt < 7u)  segR1.z = pr;
              else if (cnt < 8u)  segR1.w = pr;
              else if (cnt < 9u)  segR2.x = pr;
              else if (cnt < 10u) segR2.y = pr;
              else if (cnt < 11u) segR2.z = pr;
              else if (cnt < 12u) segR2.w = pr;
              else if (cnt < 13u) segR3.x = pr;
              else if (cnt < 14u) segR3.y = pr;
              else if (cnt < 15u) segR3.z = pr;
              else                segR3.w = pr;
              ++cnt;
            }
            w += leapSteps;
            continue;
          }
        }
      }
      if (minMaxTexture.sample(sNearest, mp, level(0)).r > 0.5f)
      {
        // Empty run [gs, w): leap exactly like the preamble until a solid cell.
        const int gs = w;
        while (w < steps)
        {
          float3 m2 = clamp(ep + es * (float)w, float3(0.0), float3(1.0));
          if (minMaxTexture.sample(sNearest, m2, level(0)).r <= 0.5f) break;
          float3 cc = m2 * mmDimF;
          float3 frc = fract(cc);
          float3 distToEdge;
          distToEdge.x = rd.x > 0.0 ? (1.0 - frc.x) : frc.x;
          distToEdge.y = rd.y > 0.0 ? (1.0 - frc.y) : frc.y;
          distToEdge.z = rd.z > 0.0 ? (1.0 - frc.z) : frc.z;
          distToEdge = mix(distToEdge, float3(1.0), float3(distToEdge <= 1e-5));
          float3 tToEdge;
          tToEdge.x = abs(rd.x) > 1e-5 ? distToEdge.x / (abs(rd.x) * mmDimF.x) : 1e30;
          tToEdge.y = abs(rd.y) > 1e-5 ? distToEdge.y / (abs(rd.y) * mmDimF.y) : 1e30;
          tToEdge.z = abs(rd.z) > 1e-5 ? distToEdge.z / (abs(rd.z) * mmDimF.z) : 1e30;
          float exactSkip = min(min(tToEdge.x, tToEdge.y), tToEdge.z) + 1e-4;
          int cellSteps = (int)ceil(exactSkip / ss);
          if (cellSteps < 1) cellSteps = 1;
          w += cellSteps;
        }
        if (cnt >= segMeta.w) break;   // remainder composites without skips
        {
          // §38.17 fix: ONE gap per u32 word (start<<16|end), direct
          // assignment — the previous OR-accumulation shifted 32-bit gap
          // words by 16 lanes, truncating odd gaps' starts and corrupting
          // even gaps' starts via OR. Output stayed correct only because
          // every corrupted endpoint still landed inside empty terrain
          // (skipped samples contribute zero); exact gaps make the hops
          // land precisely at the builder's run boundaries.
          const uint pr = ((uint)gs << 16u) | (uint)min(w, steps);
          if (cnt < 1u)       segR0.x = pr;
          else if (cnt < 2u)  segR0.y = pr;
          else if (cnt < 3u)  segR0.z = pr;
          else if (cnt < 4u)  segR0.w = pr;
          else if (cnt < 5u)  segR1.x = pr;
          else if (cnt < 6u)  segR1.y = pr;
          else if (cnt < 7u)  segR1.z = pr;
          else if (cnt < 8u)  segR1.w = pr;
          else if (cnt < 9u)  segR2.x = pr;
          else if (cnt < 10u) segR2.y = pr;
          else if (cnt < 11u) segR2.z = pr;
          else if (cnt < 12u) segR2.w = pr;
          else if (cnt < 13u) segR3.x = pr;
          else if (cnt < 14u) segR3.y = pr;
          else if (cnt < 15u) segR3.z = pr;
          else                segR3.w = pr;
          ++cnt;
        }
      }
      else
      {
        // Solid terrain: one probe per crossed cell (lattice is constant per
        // cell), jumping to the next cell face with the same boundary solve.
        float3 cc = mp * mmDimF;
        float3 frc = fract(cc);
        float3 distToEdge;
        distToEdge.x = rd.x > 0.0 ? (1.0 - frc.x) : frc.x;
        distToEdge.y = rd.y > 0.0 ? (1.0 - frc.y) : frc.y;
        distToEdge.z = rd.z > 0.0 ? (1.0 - frc.z) : frc.z;
        distToEdge = mix(distToEdge, float3(1.0), float3(distToEdge <= 1e-5));
        float3 tToEdge;
        tToEdge.x = abs(rd.x) > 1e-5 ? distToEdge.x / (abs(rd.x) * mmDimF.x) : 1e30;
        tToEdge.y = abs(rd.y) > 1e-5 ? distToEdge.y / (abs(rd.y) * mmDimF.y) : 1e30;
        tToEdge.z = abs(rd.z) > 1e-5 ? distToEdge.z / (abs(rd.z) * mmDimF.z) : 1e30;
        float exactSkip = min(min(tToEdge.x, tToEdge.y), tToEdge.z) + 1e-4;
        int adv = (int)ceil(exactSkip / ss);
        w += max(1, adv);
      }
    }
    if (dbgMirror)
    {
      dbgUse = useBlocks ? ((uint(bsI) << 1u) | 1u) : 0u;
      dbgGuard = (uint)max(guard, 0);
      dbgEpZ = (uint)max(0, (int)round(ep.z * 4096.0f));
      dbgEsz = (uint)max(0, (int)round(es.z * 65536.0f));
      dbgSs = (uint)max(0, (int)round(ss * 65536.0f));
      dbgGaps[0] = segR0.x;  dbgGaps[1] = segR0.y;
      dbgGaps[2] = segR0.z;  dbgGaps[3] = segR0.w;
      dbgGaps[4] = segR1.x;  dbgGaps[5] = segR1.y;
      dbgGaps[6] = segR1.z;  dbgGaps[7] = segR1.w;
      // (segR2/segR3 gaps 8-15 not mirrored; cnt is the ground truth.)
    }
  }

  const uint need = 1u + cnt;
  const uint base = atomic_fetch_add_explicit(poolCounter, need, memory_order_relaxed);
  // §38.17 MM_SEG_DEBUG mirror payload — captured inside the active walk
  // above, emitted after the pool claim.
  const uint dbgRec = base;
  const uint dbgSteps = (uint)max(steps, 0);
  if (base + need > segMeta.z)
  {
    segIndexMap[fpid] = 0xFFFFFFFFu;   // pool exhausted: composite-all fallback
    if (dbgMirror)
    {
      dbgOut[fpid * 16u + 0u] = 0xFFFFFFFFu;
    }
    return;
  }
  pool[base] = cnt;
  if (dbgMirror)
  {
    // 16 words/ray: recOff, cnt, steps, guard, g0..g7, epZ, esZ, ss, 0.
    dbgOut[fpid * 16u + 0u] = dbgRec;
    dbgOut[fpid * 16u + 1u] = cnt;
    dbgOut[fpid * 16u + 2u] = dbgSteps;
    dbgOut[fpid * 16u + 3u] = dbgGuard;
    dbgOut[fpid * 16u + 4u] = dbgGaps[0];
    dbgOut[fpid * 16u + 5u] = dbgGaps[1];
    dbgOut[fpid * 16u + 6u] = dbgGaps[2];
    dbgOut[fpid * 16u + 7u] = dbgGaps[3];
    dbgOut[fpid * 16u + 8u] = dbgGaps[4];
    dbgOut[fpid * 16u + 9u] = dbgGaps[5];
    dbgOut[fpid * 16u + 10u] = dbgGaps[6];
    dbgOut[fpid * 16u + 11u] = dbgGaps[7];
    dbgOut[fpid * 16u + 12u] = dbgEpZ;
    dbgOut[fpid * 16u + 13u] = dbgEsz;
    dbgOut[fpid * 16u + 14u] = dbgSs;
    dbgOut[fpid * 16u + 15u] = dbgUse;
  }
  if (cnt > 0u)  pool[base + 1u] = segR0.x;
  if (cnt > 1u)  pool[base + 2u] = segR0.y;
  if (cnt > 2u)  pool[base + 3u] = segR0.z;
  if (cnt > 3u)  pool[base + 4u] = segR0.w;
  if (cnt > 4u)  pool[base + 5u] = segR1.x;
  if (cnt > 5u)  pool[base + 6u] = segR1.y;
  if (cnt > 6u)  pool[base + 7u] = segR1.z;
  if (cnt > 7u)  pool[base + 8u] = segR1.w;
  if (cnt > 8u)  pool[base + 9u] = segR2.x;
  if (cnt > 9u)  pool[base + 10u] = segR2.y;
  if (cnt > 10u) pool[base + 11u] = segR2.z;
  if (cnt > 11u) pool[base + 12u] = segR2.w;
  if (cnt > 12u) pool[base + 13u] = segR3.x;
  if (cnt > 13u) pool[base + 14u] = segR3.y;
  if (cnt > 14u) pool[base + 15u] = segR3.z;
  if (cnt > 15u) pool[base + 16u] = segR3.w;
  segIndexMap[fpid] = base;
}

// =============================================================================
// §38.6 / §36.4 Design B — Compute Marcher & Ray-Binned Marching
// =============================================================================

inline half4 marchRayFromAtlasCore(
    float3 evalPointIn,
    int maxSteps,
    float3 evalStepIn,
    float stepSize,
    float3 rayDir,
    float tTerminateMax,
    constant VolumeMapperUniforms& volumeUniforms,
    constant PerBlockData& b,
    texture3d<float> volumeTexture,
    texture2d<float> transferFunctionTexture,
    texture2d<float> transferFunctionTexture1,
    texture2d<float> transferFunctionTexture2,
    texture2d<float> transferFunctionTexture3,
    texture2d<float> transferFunction2DTexture,
    texture3d<float> transfer2DYAxisTexture,
    texture2d<float> gradientOpacityTexture,
    texture3d<float> maskTexture,
    texture2d<float> labelMapTransferTexture,
    texture3d<float> minMaxTexture,
    texture3d<float> minMaxBlockTexture,
    texture3d<float> minMaxSuperTexture,
    texture3d<float> normalTexture,
    texture3d<float> blankingTexture,
    constant packed_float3* rectCoords,
    constant VolumeLightUniforms* lightUniforms,
    int batchCapIn,
    device uint* segIndexMapPre,
    device uint* segPool,
    device uint* segDbg,
    bool dbgRay)
{
  const bool doShading = fc_shading;
  const bool doGradOp = fc_gradientOpacity;
  const bool doCropping = fc_cropping;
  const bool doMask = fc_mask;
  const bool doBlanking = fc_blanking;
  const bool doTransfer2D = fc_transfer2D;
  const bool doRectilinear = fc_rectilinear;

  half scalarScale = half(1.0 / max((volumeUniforms.scalarMax - volumeUniforms.scalarMin), 1e-4h));
  half scalarBias  = half(-volumeUniforms.scalarMin) * scalarScale;

  half gradNormFactor = half(max(1e-8f, volumeUniforms.gradientOpacityRange.y));

  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz
                        - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * stepSize;

  float texelCountZ = volumeTexture.get_depth() *
    ((fc_volRg8 && !fc_volTransposed) ? 2.0f : 1.0f);
  float3 texelCount = fc_volTransposedY
    ? float3(volumeTexture.get_width(), volumeTexture.get_depth(),
             volumeTexture.get_height())
    : fc_volTransposed
      ? float3(volumeTexture.get_depth(), volumeTexture.get_height(),
               volumeTexture.get_width())
      : float3(volumeTexture.get_width(), volumeTexture.get_height(), texelCountZ);
  float3 ctpScale   = max(texelCount - 1.0, 1e-4) / texelCount;
  float3 ctpOffset  = 0.5 / texelCount;
  float3 evalStep = texStep * ctpScale;
  float3 evalPoint = evalPointIn;

  half3 viewDirHalf  = half3(normalize(-rayDir * boundsSize));
  half3 ambientMat   = half3(volumeUniforms.ambientColor.rgb);
  half3 diffuseMat   = half3(volumeUniforms.diffuseColor.rgb);
  half3 specularMat  = half3(volumeUniforms.specularColor.rgb);
  half shininessMat  = half(volumeUniforms.shininess);

  float3 stepVec = rayDir * stepSize;
  float3 currentPoint = float3(0.0);
  float currentT = 0.0f;
  float3 texLocalPos = evalPoint;

  const float kExitAcc = fc_exitTheta ? volumeUniforms.exitAlpha : 0.99607843137f;

  const bool useMinMax = fc_minmax &&
    b.minMaxInfo.x > 0.5 &&
    b.minMaxInfo.y > 0.5 &&
    b.minMaxInfo.z > 0.5 &&
    b.minMaxInfo.w > 0.5;
  const float3 mmDimF = b.minMaxInfo.yzw;
  const float3 mmBlkDimF = float3(minMaxBlockTexture.get_width(),
                                 minMaxBlockTexture.get_height(),
                                 minMaxBlockTexture.get_depth());
  const bool useMinMaxBlocks = useMinMax && fc_mmBlocks && mmBlkDimF.x > 1.0f;
  const float3 invMMDimF9 = 1.0f / mmDimF;
  const int bsI = max(int(volumeUniforms.mmBlockSizeCells), 1);
  const int bpsI = 64 / bsI;
  const float invBs = 1.0f / float(bsI);
  const float invBps = 1.0f / float(bpsI);
  int3 mv9Blk = int3(-1);
  int mv9BlkState = -1;
  const float3 mmSbDimF = float3(minMaxSuperTexture.get_width(),
                                 minMaxSuperTexture.get_height(),
                                 minMaxSuperTexture.get_depth());
  const bool useMinMaxSuper = useMinMaxBlocks && fc_mmSuper &&
                              mmSbDimF.x > 1.0f;
  int3 mv9Sb = int3(-1);
  bool mv9SbEmpty = false;

  half3 accumulatedColor = half3(0.0h);
  half accumulatedOpacity = 0.0h;

  const float3 adjTexMin = ctpOffset;
  const float3 adjTexMax = ctpOffset + ctpScale;
  bool seenInBounds = false;
  int i = 0;
  const int steps = maxSteps;
  const int batchCap = batchCapIn;

  // §38.17 streaming consume state (fc_cmSegHop) — fragment fc_segHop
  // pattern verbatim (see marchVolumeUnified): only (recOff, cnt, idx,
  // gStart, gEnd) stay live; segIndexMap arrives PRE-OFFSET to this ray.
  uint segRecOff = 0xFFFFFFFFu;
  int segCnt = 0;
  int segIdx = 0;
  int gS = 0;
  int gE = 0;
  if (useMinMax && fc_cmSegHop)
  {
    segRecOff = segIndexMapPre[0];
    if (segRecOff != 0xFFFFFFFFu)
    {
      segCnt = (int)segPool[segRecOff];
      if (segCnt > 0)
      {
        const uint pr = segPool[segRecOff + 1u];
        gS = (int)(pr >> 16u);
        gE = (int)(pr & 0xFFFFu);
        segIdx = 1;
        // §38.17 defensive validation: a malformed gap (start>=end or past
        // the ray) would hop the ray to oblivion; treat it as list end.
        if (gS >= gE || gE > steps)
        {
          segCnt = 0;
          segIdx = 0;
        }
      }
    }
  }

#define MV9_C_FETCH(_j) \
  float s##_j = sampleVolumeScalar(volumeTexture, evalPoint + evalStep * (float)_j);
#define MV9_C_COMPOSITE(_j) \
  half4 c##_j = sampleTransferFunction(transferFunctionTexture, \
      float2(float(saturate(half(s##_j) * scalarScale + scalarBias)), 0.5)); \
  half w##_j = 1.0h - accumulatedOpacity; \
  accumulatedColor += w##_j * (c##_j.rgb * c##_j.a); \
  accumulatedOpacity += w##_j * c##_j.a;
#define MV9_C_ADVANCE(_W) \
  currentPoint += stepVec * (float)_W; \
  currentT += stepSize * (float)_W; \
  texLocalPos += texStep * (float)_W; \
  evalPoint += evalStep * (float)_W; \
  i += _W; \
  if (accumulatedOpacity > kExitAcc) { break; } \
  if (currentT >= tTerminateMax) { break; }

  while (i < steps)
  {
    if (accumulatedOpacity > kExitAcc) break;
    if (currentT >= tTerminateMax) break;
    if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
        any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
      if (seenInBounds) { break; }
      texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
      evalPoint = cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
    } else {
      seenInBounds = true;
    }

    if (useMinMax && fc_cmSegHop)
    {
      // §38.17 consume: at most one hop per outer iteration — the loop top
      // re-checks tEnd/latch/bounds, so a hop landing past the ray end exits
      // exactly like the legacy skip (fragment fc_segHop semantics).
      if (segIdx < segCnt)
      {
        if (i >= gE)
        {
          ++segIdx;
          if (segIdx < segCnt)
          {
            const uint pr = segPool[segRecOff + 1u + uint(segIdx)];
            gS = (int)(pr >> 16u);
            gE = (int)(pr & 0xFFFFu);
            if (gS >= gE || gE > steps) { segCnt = segIdx; }
          }
        }
        else if (i >= gS)
        {
          const int hopW = gE - i;
          currentPoint += stepVec * (float)hopW;
          currentT += stepSize * (float)hopW;
          texLocalPos += texStep * (float)hopW;
          evalPoint += evalStep * (float)hopW;
          i += hopW;
          ++segIdx;
          if (segIdx < segCnt)
          {
            const uint pr = segPool[segRecOff + 1u + uint(segIdx)];
            gS = (int)(pr >> 16u);
            gE = (int)(pr & 0xFFFFu);
            if (gS >= gE || gE > steps) { segCnt = segIdx; }
          }
          continue;
        }
        // else i < gS: solid terrain until the next gap — dispatch batch.
      }
    }
    else if (useMinMax)
    {
      int w = 0;
      const int extent = min(48, steps - i);
      while (w < extent)
      {
        float3 mmPos = clamp(evalPoint + evalStep * (float)w, float3(0.0), float3(1.0));
        float3 cellCoord = mmPos * mmDimF;
        if (useMinMaxBlocks)
        {
          int3 newBlk = min(int3(cellCoord * invBs), int3(mmBlkDimF) - 1);
          if (any(newBlk != mv9Blk))
          {
            mv9Blk = newBlk;
            float bsv = minMaxBlockTexture.sample(sNearest,
                (float3(mv9Blk) + 0.5f) / mmBlkDimF, level(0)).r;
            mv9BlkState = bsv > 0.75 ? 1 : (bsv < 0.25 ? 2 : 0);
          }
          if (useMinMaxSuper)
          {
            int3 newSb = min(int3(float3(mv9Blk) * invBps), int3(mmSbDimF) - 1);
            if (any(newSb != mv9Sb))
            {
              mv9Sb = newSb;
              float ssv = minMaxSuperTexture.sample(sNearest,
                  (float3(mv9Sb) + 0.5f) / mmSbDimF, level(0)).r;
              mv9SbEmpty = ssv > 0.5f;
            }
            if (mv9SbEmpty && volumeUniforms.mmLeapLevel > 0.5f)
            {
              int3 sbLo = mv9Sb * 64;
              float3 loN = float3(sbLo) * invMMDimF9;
              float3 hiN = float3(min(sbLo + 64, int3(mmDimF))) * invMMDimF9;
              float3 rem;
              rem.x = rayDir.x > 0.0 ? (hiN.x - mmPos.x) : (mmPos.x - loN.x);
              rem.y = rayDir.y > 0.0 ? (hiN.y - mmPos.y) : (mmPos.y - loN.y);
              rem.z = rayDir.z > 0.0 ? (hiN.z - mmPos.z) : (mmPos.z - loN.z);
              rem = max(rem, float3(0.0f));
              float3 tToFace;
              tToFace.x = abs(rayDir.x) > 1e-5 ? rem.x / abs(rayDir.x) : 1e30;
              tToFace.y = abs(rayDir.y) > 1e-5 ? rem.y / abs(rayDir.y) : 1e30;
              tToFace.z = abs(rayDir.z) > 1e-5 ? rem.z / abs(rayDir.z) : 1e30;
              float exactSkip = min(min(tToFace.x, tToFace.y), tToFace.z) + 1e-4;
              int leapSteps = (int)ceil(exactSkip / stepSize);
              if (leapSteps < 1) leapSteps = 1;
              w += leapSteps;
              continue;
            }
          }
          if (mv9BlkState == 1 && volumeUniforms.mmLeapLevel > 1.5f)
          {
            int3 blkLo = mv9Blk * bsI;
            float3 loN = float3(blkLo) * invMMDimF9;
            float3 hiN = float3(min(blkLo + bsI, int3(mmDimF))) * invMMDimF9;
            float3 rem;
            rem.x = rayDir.x > 0.0 ? (hiN.x - mmPos.x) : (mmPos.x - loN.x);
            rem.y = rayDir.y > 0.0 ? (hiN.y - mmPos.y) : (mmPos.y - loN.y);
            rem.z = rayDir.z > 0.0 ? (hiN.z - mmPos.z) : (mmPos.z - loN.z);
            rem = max(rem, float3(0.0f));
            float3 tToFace;
            tToFace.x = abs(rayDir.x) > 1e-5 ? rem.x / abs(rayDir.x) : 1e30;
            tToFace.y = abs(rayDir.y) > 1e-5 ? rem.y / abs(rayDir.y) : 1e30;
            tToFace.z = abs(rayDir.z) > 1e-5 ? rem.z / abs(rayDir.z) : 1e30;
            float exactSkip = min(min(tToFace.x, tToFace.y), tToFace.z) + 1e-4;
            int leapSteps = (int)ceil(exactSkip / stepSize);
            if (leapSteps < 1) leapSteps = 1;
            w += leapSteps;
            continue;
          }
          if (mv9BlkState == 2)
          {
            break;
          }
        }
        if (volumeUniforms.mmBlocksOnly > 0.5f) { break; }
        if (minMaxTexture.sample(sNearest, mmPos, level(0)).r <= 0.5) break;
        float3 fractCoord = fract(cellCoord);
        float3 distToEdge;
        distToEdge.x = rayDir.x > 0.0 ? (1.0 - fractCoord.x) : fractCoord.x;
        distToEdge.y = rayDir.y > 0.0 ? (1.0 - fractCoord.y) : fractCoord.y;
        distToEdge.z = rayDir.z > 0.0 ? (1.0 - fractCoord.z) : fractCoord.z;
        distToEdge = mix(distToEdge, float3(1.0), float3(distToEdge <= 1e-5));
        float3 tToEdge;
        tToEdge.x = abs(rayDir.x) > 1e-5 ? distToEdge.x / (abs(rayDir.x) * mmDimF.x) : 1e30;
        tToEdge.y = abs(rayDir.y) > 1e-5 ? distToEdge.y / (abs(rayDir.y) * mmDimF.y) : 1e30;
        tToEdge.z = abs(rayDir.z) > 1e-5 ? distToEdge.z / (abs(rayDir.z) * mmDimF.z) : 1e30;
        float exactSkip = min(min(tToEdge.x, tToEdge.y), tToEdge.z) + 1e-4;
        int cellSteps = (int)ceil(exactSkip / stepSize);
        if (cellSteps < 1) cellSteps = 1;
        w += cellSteps;
      }
      if (w >= extent)
      {
        currentPoint += stepVec * (float)extent;
        currentT += stepSize * (float)extent;
        texLocalPos += texStep * (float)extent;
        evalPoint += evalStep * (float)extent;
        i += extent;
        continue;
      }
      if (w > 0)
      {
        currentPoint += stepVec * (float)w;
        currentT += stepSize * (float)w;
        texLocalPos += texStep * (float)w;
        evalPoint += evalStep * (float)w;
        i += w;
      }
    }

    // §38.16 PRE-v2: issue the next batch's first block-summary read here so
    // its miss latency overlaps the composite ladder below. Prediction uses
    // the exact batch extent (batchCap), unlike v1's fixed min(48,·) which
    if (batchCap >= 48 && i + 48 <= steps)
    {
      MV9_C_FETCH(0) MV9_C_FETCH(1) MV9_C_FETCH(2) MV9_C_FETCH(3)
      MV9_C_FETCH(4) MV9_C_FETCH(5) MV9_C_FETCH(6) MV9_C_FETCH(7)
      MV9_C_FETCH(8) MV9_C_FETCH(9) MV9_C_FETCH(10) MV9_C_FETCH(11)
      MV9_C_FETCH(12) MV9_C_FETCH(13) MV9_C_FETCH(14) MV9_C_FETCH(15)
      MV9_C_FETCH(16) MV9_C_FETCH(17) MV9_C_FETCH(18) MV9_C_FETCH(19)
      MV9_C_FETCH(20) MV9_C_FETCH(21) MV9_C_FETCH(22) MV9_C_FETCH(23)
      MV9_C_FETCH(24) MV9_C_FETCH(25) MV9_C_FETCH(26) MV9_C_FETCH(27)
      MV9_C_FETCH(28) MV9_C_FETCH(29) MV9_C_FETCH(30) MV9_C_FETCH(31)
      MV9_C_FETCH(32) MV9_C_FETCH(33) MV9_C_FETCH(34) MV9_C_FETCH(35)
      MV9_C_FETCH(36) MV9_C_FETCH(37) MV9_C_FETCH(38) MV9_C_FETCH(39)
      MV9_C_FETCH(40) MV9_C_FETCH(41) MV9_C_FETCH(42) MV9_C_FETCH(43)
      MV9_C_FETCH(44) MV9_C_FETCH(45) MV9_C_FETCH(46) MV9_C_FETCH(47)
      MV9_C_COMPOSITE(0) MV9_C_COMPOSITE(1) MV9_C_COMPOSITE(2) MV9_C_COMPOSITE(3)
      MV9_C_COMPOSITE(4) MV9_C_COMPOSITE(5) MV9_C_COMPOSITE(6) MV9_C_COMPOSITE(7)
      MV9_C_COMPOSITE(8) MV9_C_COMPOSITE(9) MV9_C_COMPOSITE(10) MV9_C_COMPOSITE(11)
      MV9_C_COMPOSITE(12) MV9_C_COMPOSITE(13) MV9_C_COMPOSITE(14) MV9_C_COMPOSITE(15)
      MV9_C_COMPOSITE(16) MV9_C_COMPOSITE(17) MV9_C_COMPOSITE(18) MV9_C_COMPOSITE(19)
      MV9_C_COMPOSITE(20) MV9_C_COMPOSITE(21) MV9_C_COMPOSITE(22) MV9_C_COMPOSITE(23)
      MV9_C_COMPOSITE(24) MV9_C_COMPOSITE(25) MV9_C_COMPOSITE(26) MV9_C_COMPOSITE(27)
      MV9_C_COMPOSITE(28) MV9_C_COMPOSITE(29) MV9_C_COMPOSITE(30) MV9_C_COMPOSITE(31)
      MV9_C_COMPOSITE(32) MV9_C_COMPOSITE(33) MV9_C_COMPOSITE(34) MV9_C_COMPOSITE(35)
      MV9_C_COMPOSITE(36) MV9_C_COMPOSITE(37) MV9_C_COMPOSITE(38) MV9_C_COMPOSITE(39)
      MV9_C_COMPOSITE(40) MV9_C_COMPOSITE(41) MV9_C_COMPOSITE(42) MV9_C_COMPOSITE(43)
      MV9_C_COMPOSITE(44) MV9_C_COMPOSITE(45) MV9_C_COMPOSITE(46) MV9_C_COMPOSITE(47)
      MV9_C_ADVANCE(48)
    }
    // §38.12 stride-parity split: legacy 32-rung cadence (tests/preamble
    // every 32 samples exactly as batchCap=32) with the rung body split into
    // two independent 16-halves — identical fp op sequence, lower peak
    // register liveness when fc_cmBatch <= 16.
    else if (fc_cmSplit && batchCap >= 32 && i + 32 <= steps)
    {
      {
        MV9_C_FETCH(0) MV9_C_FETCH(1) MV9_C_FETCH(2) MV9_C_FETCH(3)
        MV9_C_FETCH(4) MV9_C_FETCH(5) MV9_C_FETCH(6) MV9_C_FETCH(7)
        MV9_C_FETCH(8) MV9_C_FETCH(9) MV9_C_FETCH(10) MV9_C_FETCH(11)
        MV9_C_FETCH(12) MV9_C_FETCH(13) MV9_C_FETCH(14) MV9_C_FETCH(15)
        MV9_C_COMPOSITE(0) MV9_C_COMPOSITE(1) MV9_C_COMPOSITE(2) MV9_C_COMPOSITE(3)
        MV9_C_COMPOSITE(4) MV9_C_COMPOSITE(5) MV9_C_COMPOSITE(6) MV9_C_COMPOSITE(7)
        MV9_C_COMPOSITE(8) MV9_C_COMPOSITE(9) MV9_C_COMPOSITE(10) MV9_C_COMPOSITE(11)
        MV9_C_COMPOSITE(12) MV9_C_COMPOSITE(13) MV9_C_COMPOSITE(14) MV9_C_COMPOSITE(15)
      }
      {
        MV9_C_FETCH(16) MV9_C_FETCH(17) MV9_C_FETCH(18) MV9_C_FETCH(19)
        MV9_C_FETCH(20) MV9_C_FETCH(21) MV9_C_FETCH(22) MV9_C_FETCH(23)
        MV9_C_FETCH(24) MV9_C_FETCH(25) MV9_C_FETCH(26) MV9_C_FETCH(27)
        MV9_C_FETCH(28) MV9_C_FETCH(29) MV9_C_FETCH(30) MV9_C_FETCH(31)
        MV9_C_COMPOSITE(16) MV9_C_COMPOSITE(17) MV9_C_COMPOSITE(18) MV9_C_COMPOSITE(19)
        MV9_C_COMPOSITE(20) MV9_C_COMPOSITE(21) MV9_C_COMPOSITE(22) MV9_C_COMPOSITE(23)
        MV9_C_COMPOSITE(24) MV9_C_COMPOSITE(25) MV9_C_COMPOSITE(26) MV9_C_COMPOSITE(27)
        MV9_C_COMPOSITE(28) MV9_C_COMPOSITE(29) MV9_C_COMPOSITE(30) MV9_C_COMPOSITE(31)
      }
      MV9_C_ADVANCE(32)
    }
    else if (batchCap >= 32 && i + 32 <= steps)
    {
      MV9_C_FETCH(0) MV9_C_FETCH(1) MV9_C_FETCH(2) MV9_C_FETCH(3)
      MV9_C_FETCH(4) MV9_C_FETCH(5) MV9_C_FETCH(6) MV9_C_FETCH(7)
      MV9_C_FETCH(8) MV9_C_FETCH(9) MV9_C_FETCH(10) MV9_C_FETCH(11)
      MV9_C_FETCH(12) MV9_C_FETCH(13) MV9_C_FETCH(14) MV9_C_FETCH(15)
      MV9_C_FETCH(16) MV9_C_FETCH(17) MV9_C_FETCH(18) MV9_C_FETCH(19)
      MV9_C_FETCH(20) MV9_C_FETCH(21) MV9_C_FETCH(22) MV9_C_FETCH(23)
      MV9_C_FETCH(24) MV9_C_FETCH(25) MV9_C_FETCH(26) MV9_C_FETCH(27)
      MV9_C_FETCH(28) MV9_C_FETCH(29) MV9_C_FETCH(30) MV9_C_FETCH(31)
      MV9_C_COMPOSITE(0) MV9_C_COMPOSITE(1) MV9_C_COMPOSITE(2) MV9_C_COMPOSITE(3)
      MV9_C_COMPOSITE(4) MV9_C_COMPOSITE(5) MV9_C_COMPOSITE(6) MV9_C_COMPOSITE(7)
      MV9_C_COMPOSITE(8) MV9_C_COMPOSITE(9) MV9_C_COMPOSITE(10) MV9_C_COMPOSITE(11)
      MV9_C_COMPOSITE(12) MV9_C_COMPOSITE(13) MV9_C_COMPOSITE(14) MV9_C_COMPOSITE(15)
      MV9_C_COMPOSITE(16) MV9_C_COMPOSITE(17) MV9_C_COMPOSITE(18) MV9_C_COMPOSITE(19)
      MV9_C_COMPOSITE(20) MV9_C_COMPOSITE(21) MV9_C_COMPOSITE(22) MV9_C_COMPOSITE(23)
      MV9_C_COMPOSITE(24) MV9_C_COMPOSITE(25) MV9_C_COMPOSITE(26) MV9_C_COMPOSITE(27)
      MV9_C_COMPOSITE(28) MV9_C_COMPOSITE(29) MV9_C_COMPOSITE(30) MV9_C_COMPOSITE(31)
      MV9_C_ADVANCE(32)
    }
    else if (batchCap >= 16 && i + 16 <= steps)
    {
      MV9_C_FETCH(0) MV9_C_FETCH(1) MV9_C_FETCH(2) MV9_C_FETCH(3)
      MV9_C_FETCH(4) MV9_C_FETCH(5) MV9_C_FETCH(6) MV9_C_FETCH(7)
      MV9_C_FETCH(8) MV9_C_FETCH(9) MV9_C_FETCH(10) MV9_C_FETCH(11)
      MV9_C_FETCH(12) MV9_C_FETCH(13) MV9_C_FETCH(14) MV9_C_FETCH(15)
      MV9_C_COMPOSITE(0) MV9_C_COMPOSITE(1) MV9_C_COMPOSITE(2) MV9_C_COMPOSITE(3)
      MV9_C_COMPOSITE(4) MV9_C_COMPOSITE(5) MV9_C_COMPOSITE(6) MV9_C_COMPOSITE(7)
      MV9_C_COMPOSITE(8) MV9_C_COMPOSITE(9) MV9_C_COMPOSITE(10) MV9_C_COMPOSITE(11)
      MV9_C_COMPOSITE(12) MV9_C_COMPOSITE(13) MV9_C_COMPOSITE(14) MV9_C_COMPOSITE(15)
      MV9_C_ADVANCE(16)
    }
    else if (batchCap >= 8 && i + 8 <= steps)
    {
      MV9_C_FETCH(0) MV9_C_FETCH(1) MV9_C_FETCH(2) MV9_C_FETCH(3)
      MV9_C_FETCH(4) MV9_C_FETCH(5) MV9_C_FETCH(6) MV9_C_FETCH(7)
      MV9_C_COMPOSITE(0) MV9_C_COMPOSITE(1) MV9_C_COMPOSITE(2) MV9_C_COMPOSITE(3)
      MV9_C_COMPOSITE(4) MV9_C_COMPOSITE(5) MV9_C_COMPOSITE(6) MV9_C_COMPOSITE(7)
      MV9_C_ADVANCE(8)
    }
    else if (batchCap >= 4 && i + 4 <= steps)
    {
      MV9_C_FETCH(0) MV9_C_FETCH(1) MV9_C_FETCH(2) MV9_C_FETCH(3)
      MV9_C_COMPOSITE(0) MV9_C_COMPOSITE(1) MV9_C_COMPOSITE(2) MV9_C_COMPOSITE(3)
      MV9_C_ADVANCE(4)
    }
    else if (batchCap >= 2 && i + 2 <= steps)
    {
      MV9_C_FETCH(0) MV9_C_FETCH(1)
      MV9_C_COMPOSITE(0) MV9_C_COMPOSITE(1)
      MV9_C_ADVANCE(2)
    }
    else
    {
      MV9_C_FETCH(0)
      MV9_C_COMPOSITE(0)
      MV9_C_ADVANCE(1)
    }
  }

#undef MV9_C_FETCH
#undef MV9_C_COMPOSITE
#undef MV9_C_ADVANCE

  accumulatedColor = clamp(accumulatedColor, half3(0.0h), half3(1.0h));
  accumulatedOpacity = clamp(accumulatedOpacity, 0.0h, 1.0h);
  if (dbgRay)
  {
    segDbg[0] = (uint)i;
    segDbg[1] = (uint)segIdx;
    segDbg[2] = (uint)segCnt;
    segDbg[3] = as_type<uint>(float(accumulatedOpacity));
  }
  return half4(accumulatedColor, accumulatedOpacity);
}

// §38.6 compute-march probes (TEMP-DIAG, env-gated from the mapper):
// floorMode skips the march body to expose the dispatch+atlas+blit floor;
// stepCap forces a maximum march length (per-length slope decomposition).
// synthMode rebuilds the atlas planes in-kernel from uniforms + analytic
// ray-box entry (CM_SYNTH) — removes the render->compute atlas dependency,
// which measured as the dominant cost (~7.4 ms @1024² of ~8.6 total).
struct ComputeMarchControl {
  uint floorMode; uint stepCap; uint synthMode;
  // TEMP-DIAG: if nonzero, overrides volumeUniforms.maxBatchWidth for THIS
  // dispatch only (register-pressure/occupancy probe; fragment unaffected).
  // Accumulation order is unchanged (sequential over j); only checkpoint
  // frequency of the exit tests changes => <=1LSB-class output drift.
  uint batchOverride;
};

kernel void volume_compute_march(
    texture2d<float, access::read> segAtlasA [[texture(0)]],
    texture2d<float, access::read> segAtlasB [[texture(1)]],
    texture2d<float, access::read> segAtlasC [[texture(3)]],
    texture2d<half, access::write> outColorTexture [[texture(4)]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(5)]],
    texture2d<float> transferFunctionTexture [[texture(6)]],
    texture2d<float> transferFunctionTexture1 [[texture(7)]],
    texture2d<float> transferFunctionTexture2 [[texture(8)]],
    texture2d<float> transferFunctionTexture3 [[texture(9)]],
    texture2d<float> transferFunction2DTexture [[texture(10)]],
    texture3d<float> transfer2DYAxisTexture [[texture(11)]],
    texture2d<float> gradientOpacityTexture [[texture(12)]],
    texture3d<float> maskTexture [[texture(13)]],
    texture2d<float> labelMapTransferTexture [[texture(14)]],
    texture3d<float> minMaxTexture [[texture(15)]],
    texture3d<float> minMaxBlockTexture [[texture(16)]],
    texture3d<float> minMaxSuperTexture [[texture(17)]],
    texture3d<float> normalTexture [[texture(18)]],
    texture3d<float> blankingTexture [[texture(19)]],
    constant packed_float3* rectCoords [[buffer(3)]],
    constant VolumeLightUniforms& volumeLights [[buffer(4)]],
    constant ComputeMarchControl& cmc [[buffer(7)]],
    // §38.17: per-ray skip segments (pre-offset per ray inside the march).
    device uint* segIndexMap [[buffer(8)]],
    device uint* segPool [[buffer(9)]],
    device uint* segDbgOut [[buffer(10)]],
    texture2d<float> depthSynthTexture [[texture(20)]],
    uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= (uint)volumeUniforms.viewportSize.x ||
      gid.y >= (uint)volumeUniforms.viewportSize.y) return;

  if (cmc.floorMode)
  {
    outColorTexture.write(half4(0.0h), gid);
    return;
  }

  float4 A, B, C;
  int steps;
  if (cmc.synthMode)
  {
    SynthRay so;
    if (!synthesizeAtlasRay(gid, volumeUniforms, b, volumeTexture,
                            depthSynthTexture, so))
    {
      outColorTexture.write(half4(0.0h), gid);
      return;
    }
    steps = min(max(so.steps, 0), cmc.stepCap ? (int)cmc.stepCap : (1 << 20));
    const int effBatch =
      (fc_cmBatch > 0) ? fc_cmBatch
        : (cmc.batchOverride >= 1 && cmc.batchOverride <= 48)
          ? int(cmc.batchOverride)
          : max(1, int(volumeUniforms.maxBatchWidth));
    half4 color = marchRayFromAtlasCore(so.evalPoint, steps, so.evalStep,
        so.stepSize, so.rayDir, so.tTerminateMax,
        volumeUniforms, b, volumeTexture, transferFunctionTexture,
        transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3,
        transferFunction2DTexture, transfer2DYAxisTexture,
        gradientOpacityTexture, maskTexture, labelMapTransferTexture,
        minMaxTexture, minMaxBlockTexture, minMaxSuperTexture, normalTexture, blankingTexture,
        rectCoords, &volumeLights, effBatch,
        // §38.17: pre-offset to this ray's segment record.
        segIndexMap + (uint(gid.x) + uint(gid.y) * uint(volumeUniforms.viewportSize.x)),
        segPool,
        segDbgOut,
        fc_cmSegHop && gid.x == (uint(volumeUniforms.viewportSize.x) >> 1u) &&
          gid.y == (uint(volumeUniforms.viewportSize.y) >> 1u));
    outColorTexture.write(color, gid);
    return;
  }

  A = segAtlasA.read(gid);
  B = segAtlasB.read(gid);
  C = segAtlasC.read(gid);
  // Garbage-guard (mirrors volume_segment_build's guard valve): the atlas is
  // cleared outside the proxy footprint, but clamp anyway so no pathological
  // A.w can ever wedge the command buffer inside waitUntilCompleted.
  steps = min(max(int(A.w), 0), cmc.stepCap ? (int)cmc.stepCap : (1 << 20));
  if (steps <= 0)
  {
    outColorTexture.write(half4(0.0h), gid);
    return;
  }
  const int effBatch =
    (fc_cmBatch > 0) ? fc_cmBatch
      : (cmc.batchOverride >= 1 && cmc.batchOverride <= 48)
        ? int(cmc.batchOverride)
        : max(1, int(volumeUniforms.maxBatchWidth));

  half4 color = marchRayFromAtlasCore(A.xyz, steps, B.xyz, B.w, C.xyz, C.w,
      volumeUniforms, b, volumeTexture, transferFunctionTexture,
      transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3,
      transferFunction2DTexture, transfer2DYAxisTexture,
      gradientOpacityTexture, maskTexture, labelMapTransferTexture,
      minMaxTexture, minMaxBlockTexture, minMaxSuperTexture, normalTexture, blankingTexture,
      rectCoords, &volumeLights, effBatch,
      // §38.17: pre-offset to this ray's segment record.
      segIndexMap + (uint(gid.x) + uint(gid.y) * uint(volumeUniforms.viewportSize.x)),
      segPool,
      segDbgOut,
      fc_cmSegHop && gid.x == (uint(volumeUniforms.viewportSize.x) >> 1u) &&
        gid.y == (uint(volumeUniforms.viewportSize.y) >> 1u));

  outColorTexture.write(color, gid);
}

kernel void volume_ray_bin_classify(
    texture2d<float, access::read> segAtlasA [[texture(0)]],
    device atomic_uint* binCounters [[buffer(0)]],
    device uint* binRayIndices [[buffer(1)]],
    constant uint4& binMeta [[buffer(2)]], // x=width, y=height, z=binCapacity, w=numBins
    uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= binMeta.x || gid.y >= binMeta.y) return;
  float4 A = segAtlasA.read(gid);
  int steps = int(A.w);
  if (steps <= 0) return;

  uint numBins = binMeta.w;
  uint binIdx = 0;
  if (numBins > 1)
  {
    if (steps <= 32) binIdx = 0;
    else if (steps <= 128) binIdx = 1;
    else if (steps <= 512) binIdx = 2;
    else binIdx = 3;
    if (binIdx >= numBins) binIdx = numBins - 1;
  }

  uint slot = atomic_fetch_add_explicit(&binCounters[binIdx], 1, memory_order_relaxed);
  uint binCap = binMeta.z;
  if (slot < binCap)
  {
    uint packedUv = (gid.y << 16u) | (gid.x & 0xFFFFu);
    binRayIndices[binIdx * binCap + slot] = packedUv;
  }
}

kernel void volume_compute_march_binned(
    texture2d<float, access::read> segAtlasA [[texture(0)]],
    texture2d<float, access::read> segAtlasB [[texture(1)]],
    texture2d<float, access::read> segAtlasC [[texture(3)]],
    texture2d<half, access::write> outColorTexture [[texture(4)]],
    device const uint* binRayIndices [[buffer(0)]],
    constant uint2& binnedMeta [[buffer(6)]], // x=binCapacity, y=binOffset
    device atomic_uint* binCounters [[buffer(5)]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    constant ComputeMarchControl& cmc [[buffer(7)]],
    // §38.17: per-ray skip segments.
    device uint* segIndexMap [[buffer(8)]],
    device uint* segPool [[buffer(9)]],
    device uint* segDbgOut [[buffer(10)]],
    texture3d<float> volumeTexture [[texture(5)]],
    texture2d<float> transferFunctionTexture [[texture(6)]],
    texture2d<float> transferFunctionTexture1 [[texture(7)]],
    texture2d<float> transferFunctionTexture2 [[texture(8)]],
    texture2d<float> transferFunctionTexture3 [[texture(9)]],
    texture2d<float> transferFunction2DTexture [[texture(10)]],
    texture3d<float> transfer2DYAxisTexture [[texture(11)]],
    texture2d<float> gradientOpacityTexture [[texture(12)]],
    texture3d<float> maskTexture [[texture(13)]],
    texture2d<float> labelMapTransferTexture [[texture(14)]],
    texture3d<float> minMaxTexture [[texture(15)]],
    texture3d<float> minMaxBlockTexture [[texture(16)]],
    texture3d<float> minMaxSuperTexture [[texture(17)]],
    texture3d<float> normalTexture [[texture(18)]],
    texture3d<float> blankingTexture [[texture(19)]],
    constant packed_float3* rectCoords [[buffer(3)]],
    constant VolumeLightUniforms& volumeLights [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  // Threads beyond this frame's live count for this bin exit without
  // marching: unwritten index slots hold stale UVs from previous frames.
  const uint binIdx = binnedMeta.y / binnedMeta.x;
  const uint liveCount = atomic_load_explicit(&binCounters[binIdx],
                                              memory_order_relaxed);
  if (gid >= min(liveCount, binnedMeta.x)) return;
  uint packedUv = binRayIndices[binnedMeta.y + gid];
  uint2 uv = uint2(packedUv & 0xFFFFu, packedUv >> 16u);

  float4 A = segAtlasA.read(uv);
  float4 B = segAtlasB.read(uv);
  float4 C = segAtlasC.read(uv);
  if (cmc.floorMode) return;
  int steps = min(max(int(A.w), 0), cmc.stepCap ? (int)cmc.stepCap : (1 << 20));
  if (steps <= 0) return;
  const int effBatch =
    (fc_cmBatch > 0) ? fc_cmBatch
      : (cmc.batchOverride >= 1 && cmc.batchOverride <= 48)
        ? int(cmc.batchOverride)
        : max(1, int(volumeUniforms.maxBatchWidth));

  half4 color = marchRayFromAtlasCore(A.xyz, steps, B.xyz, B.w, C.xyz, C.w,
      volumeUniforms, b, volumeTexture, transferFunctionTexture,
      transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3,
      transferFunction2DTexture, transfer2DYAxisTexture,
      gradientOpacityTexture, maskTexture, labelMapTransferTexture,
      minMaxTexture, minMaxBlockTexture, minMaxSuperTexture, normalTexture, blankingTexture,
      rectCoords, &volumeLights, effBatch,
      // §38.17: pre-offset to this ray's segment record (binned: uv is the
      // original image-space pixel).
      segIndexMap + (uv.x + uv.y * uint(volumeUniforms.viewportSize.x)),
      segPool,
      segDbgOut,
      false);

  outColorTexture.write(color, uv);
}

// =============================================================================
// Cinematic rendering — shaded DVR cinematic (wax AO+SSS, front-to-back over)
// First-surface wrap/AO/SSS + premul over black, HG phase on headlight.
// Reserved: cinematicSamples / cinematicMaxBounces / cinematicMajorantSigma
// (majorant/bounces unused at 1 spp). No delta-tracking at 1 spp.
// =============================================================================
// Hash-based RNG (xorshift) — cheap, no texture fetch, coherent in 8x8 tiles.
inline uint cinematic_hash(uint x) {
  x ^= x >> 16; x *= 0x7feb352d;
  x ^= x >> 15; x *= 0x846ca68b;
  x ^= x >> 16;
  return x;
}
inline float cinematic_rand(thread uint &s) {
  s = cinematic_hash(s);
  return float(s & 0x00FFFFFF) / 16777216.0;
}
inline float sample_opacity(texture3d<float> vol, texture2d<float> tf, float3 p, half scale, half bias) {
  if (any(p < 0.0) || any(p > 1.0)) return 0.0;
  float s = sampleVolumeScalar(vol, p);
  half n = saturate(half(s) * scale + bias);
  return float(sampleTransferFunction(tf, float2(float(n), 0.5)).a);
}
inline float optical_depth(texture3d<float> vol, texture2d<float> tf, float3 origin, float3 dir, float maxT, int nSteps, half scale, half bias, float sigmaScale) {
  float dt = maxT / float(nSteps);
  float tau = 0.0;
  float3 p = origin + dir * (0.5 * dt);
  for (int i=0;i<nSteps;++i) {
    float a = sample_opacity(vol, tf, p, scale, bias);
    tau += a * dt * sigmaScale;
    if (tau > 6.0) break;
    p += dir * dt;
  }
  return tau;
}
inline float sample_blur(texture3d<float> vol, float3 p, float3 gs) {
  float s = 0.0;
  s += sampleVolumeScalar(vol, p);
  s += sampleVolumeScalar(vol, p + float3(gs.x, 0, 0));
  s += sampleVolumeScalar(vol, p - float3(gs.x, 0, 0));
  s += sampleVolumeScalar(vol, p + float3(0, gs.y, 0));
  s += sampleVolumeScalar(vol, p - float3(0, gs.y, 0));
  s += sampleVolumeScalar(vol, p + float3(0, 0, gs.z));
  s += sampleVolumeScalar(vol, p - float3(0, 0, gs.z));
  return s / 7.0;
}
inline float3 cinematic_onb_n(float3 n, float phi, float cosT)
{
  float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
  float3 up = fabs(n.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
  float3 t = normalize(cross(up, n));
  float3 b = cross(n, t);
  return normalize(t * cos(phi) * sinT + b * sin(phi) * sinT + n * cosT);
}
inline half4 cine_accum(half4 curr, half4 prev, uint n) { // fade, not spp: mixes 1 spp frames
  if (n <= 1) return curr;
  float a = 1.0 / float(min(n, 16u));
  if (curr.a < 0.01h && prev.a > 0.01h) return prev;
  return mix(prev, curr, half(a));
}
inline half4 cinematic_march_core(
    float3 evalPoint, int maxSteps, float3 evalStep, float stepSize, float3 rayDir,
    float tTerminateMax,
    constant VolumeMapperUniforms& u, constant PerBlockData& b,
    texture3d<float> volumeTexture, texture2d<float> transferFunctionTexture,
    texture2d<float> transferFunctionTexture1, texture2d<float> transferFunctionTexture2,
    texture2d<float> transferFunctionTexture3, texture2d<float> transferFunction2DTexture,
    texture3d<float> transfer2DYAxisTexture, texture2d<float> gradientOpacityTexture,
    texture3d<float> maskTexture, texture2d<float> labelMapTransferTexture,
    texture3d<float> minMaxTexture, texture3d<float> minMaxBlockTexture,
    texture3d<float> minMaxSuperTexture, texture3d<float> normalTexture,
    texture3d<float> blankingTexture, constant packed_float3* rectCoords,
    constant VolumeLightUniforms* lightUniforms, uint frameSeed, uint2 gid)
{
  if (u.cinematicEnabled < 0.5) {
    return marchRayFromAtlasCore(evalPoint, maxSteps, evalStep, stepSize, rayDir, tTerminateMax,
      u, b, volumeTexture, transferFunctionTexture, transferFunctionTexture1,
      transferFunctionTexture2, transferFunctionTexture3, transferFunction2DTexture,
      transfer2DYAxisTexture, gradientOpacityTexture, maskTexture, labelMapTransferTexture,
      minMaxTexture, minMaxBlockTexture, minMaxSuperTexture, normalTexture, blankingTexture,
      rectCoords, lightUniforms, 16, nullptr, nullptr, nullptr, false);
  }

  half scalarScale = half(1.0 / max((u.scalarMax - u.scalarMin), 1e-4h));
  half scalarBias  = half(-u.scalarMin) * scalarScale;
  float ss = clamp(u.subsurfaceStrength, 0.0, 1.0);
  float3 ssColor = saturate(float3(u.subsurfaceColorR, u.subsurfaceColorG, u.subsurfaceColorB));
  float reach = clamp(u.cinematicReach, 0.0, 1.0);

  // Studio: 57° key + dim fill + rim (specimen, not flashlight)
  float3 V     = -normalize(rayDir);
  float3 L     = cinematic_onb_n(V, 0.55, 0.55);
  float3 Lfill = cinematic_onb_n(V, 3.70, 0.45);
  float3 Lrim  = cinematic_onb_n(V, 3.0, -0.25); // behind, toward camera

  float voxel  = max(length(b.gradientStep.xyz), 1e-4);
  float aoDist = mix(4.0, 14.0, reach) * voxel;  // 12-step cavity, not 6-step kink
  float sigma  = max(u.cinematicBlend, 1.0);
  float k      = 1.0 / voxel; // UV → voxel: one voxel ≈ a*sigma

  uint seed = cinematic_hash(gid.x * 1973u + gid.y * 9277u + frameSeed);
  float j = (u.cinematicAccumCount <= 1) ? 0.0 : 1.0; // stills: 0 (no sparkle), temporal: 1.0
  float3 cur = evalPoint + evalStep * (cinematic_rand(seed) * j);

  int steps = min(maxSteps, 1024);

  for (int i = 0; i < steps; ++i) {
    if (any(cur < 0.0 || cur > 1.0)) break;

    float s = sampleVolumeScalar(volumeTexture, cur);
    half nrm = saturate(half(s) * scalarScale + scalarBias);
    half4 tf = sampleTransferFunction(transferFunctionTexture, float2(float(nrm), 0.5));
    float a = float(tf.a);
    if (a < 0.04) { cur += evalStep; continue; }

    float3 N;
    float gmag;
    if (u.useNormalTexture > 0.5) {
      float4 nt = normalTexture.sample(sVolume, cur);
      N = normalize(nt.xyz * 2.0 - 1.0);
      if (dot(N, V) < 0.0) N = -N;
      // magnitude: 2-tap, not nt.w (often unused/1)
      float3 gs1 = b.gradientStep.xyz;
      float3 g2 = float3(
        sampleVolumeScalar(volumeTexture, cur + float3(gs1.x,0,0)) -
        sampleVolumeScalar(volumeTexture, cur - float3(gs1.x,0,0)),
        sampleVolumeScalar(volumeTexture, cur + float3(0,gs1.y,0)) -
        sampleVolumeScalar(volumeTexture, cur - float3(0,gs1.y,0)),
        sampleVolumeScalar(volumeTexture, cur + float3(0,0,gs1.z)) -
        sampleVolumeScalar(volumeTexture, cur - float3(0,0,gs1.z)));
      gmag = saturate((length(g2) - 0.018) * 5.5);
    } else {
      // Two magnitudes: filtered N (smooth), raw gmag (edge)
      // Blurred N at 2.2x gives folia; raw gmag at 1.5x keeps rim/SSS/shade alive
      float3 gsN    = b.gradientStep.xyz * 2.2;
      float3 gsBlur = b.gradientStep.xyz;
      float3 gsMag  = b.gradientStep.xyz * 1.5;
      float3 gradN = float3(
        sample_blur(volumeTexture, cur + float3(gsN.x, 0, 0), gsBlur) -
        sample_blur(volumeTexture, cur - float3(gsN.x, 0, 0), gsBlur),
        sample_blur(volumeTexture, cur + float3(0, gsN.y, 0), gsBlur) -
        sample_blur(volumeTexture, cur - float3(0, gsN.y, 0), gsBlur),
        sample_blur(volumeTexture, cur + float3(0, 0, gsN.z), gsBlur) -
        sample_blur(volumeTexture, cur - float3(0, 0, gsN.z), gsBlur));
      float3 gradM = float3(
        sampleVolumeScalar(volumeTexture, cur + float3(gsMag.x,0,0)) -
        sampleVolumeScalar(volumeTexture, cur - float3(gsMag.x,0,0)),
        sampleVolumeScalar(volumeTexture, cur + float3(0,gsMag.y,0)) -
        sampleVolumeScalar(volumeTexture, cur - float3(0,gsMag.y,0)),
        sampleVolumeScalar(volumeTexture, cur + float3(0,0,gsMag.z)) -
        sampleVolumeScalar(volumeTexture, cur - float3(0,0,gsMag.z)));
      float glenN = length(gradN);
      gmag = saturate((length(gradM) - 0.018) * 5.5);
      N = (glenN > 1e-6) ? normalize(-gradN) : V;
      if (dot(N, V) < 0.0) N = -N;
    }
    if (a < 0.08) { cur += evalStep; continue; }

    float tauAO = optical_depth(volumeTexture, transferFunctionTexture,
                                cur + N * voxel,  N, aoDist,      12, scalarScale, scalarBias, sigma);
    float tauSS = optical_depth(volumeTexture, transferFunctionTexture,
                                cur - N * voxel, -N, aoDist * 1.8, 12, scalarScale, scalarBias, sigma);
    float ao    = saturate(exp(-tauAO * k * 0.55));
    float thick = saturate(exp(-tauSS * k * 0.35));
    ao = max(ao, 0.40); // fissure cannot go to [2,1,1]
    ao = mix(1.0, ao, saturate(gmag * 1.8)); // cuts: weak gradient -> no cave-AO

    float edge = smoothstep(0.06, 0.22, gmag); // binary-ish, not salt

    float ndl  = dot(N, L);
    float ndlF = dot(N, Lfill);
    float wrap = pow(saturate((ndl  + 0.22) / 1.22), 1.20);
    float fill = pow(saturate((ndlF + 0.50) / 1.50), 1.10) * 0.22;
    float rim  = pow(saturate(dot(N, Lrim)), 2.0) * 0.18 * edge; // silver edge only
    float shade = saturate(wrap + fill); // no mix(..., gmag)
    shade *= mix(0.55, 1.0, ao);

    float3 albedo = float3(tf.rgb) * 0.90;
    albedo *= mix(float3(1.0), ssColor, 0.08 * ss);
    // no albedo *= mix(0.90, 1.0, gmag)

    float side    = saturate(1.0 - abs(ndl));
    float wrapSSS = pow(saturate((-ndl + 0.45) / 1.45), 1.3);
    float3 sss = ssColor * ss * edge
               * (0.12 * side + 0.38 * wrapSSS)
               * mix(0.12, 1.0, thick);

    float3 H   = normalize(L + V);
    float spec = pow(saturate(dot(N, H)), 72.0) * edge * 0.025;
    spec *= mix(0.10, 1.0, ao);

    float3 sCol = saturate(albedo * shade + sss + spec + albedo * rim);

    // Thin lit skin: latch at 0.08 keeps dim edge, composite 8 lit samples
    // with same shade/N. Left a~0.11 ×8 -> opaque wax; right a~0.82 ×1 -> one hit.
    // No cov(a), no unlit tail, no second gradient.
    // sCol locked at hit 0: same shade/N, only a changes. Stops TF grain.
    float aAccum = 0.0;
    float3 colAccum = float3(0.0);
    float3 p = cur;
    for (int k = 0; k < 8 && aAccum < 0.95; ++k) {
      if (k > 0) {
        p += evalStep;
        if (any(p < 0.0 || p > 1.0)) break;
        float s2 = sampleVolumeScalar(volumeTexture, p);
        half n2 = saturate(half(s2) * scalarScale + scalarBias);
        half4 tf2 = sampleTransferFunction(transferFunctionTexture, float2(float(n2), 0.5));
        float a2 = float(tf2.a);
        if (a2 < 0.04) continue;
        if (a2 < 0.08) break; // left interface: air/CSF, don't composite noise
        a = a2;
      }
      float w = (1.0 - aAccum) * a;
      colAccum += sCol * w;
      aAccum   += w;
    }
    return half4(half3(clamp(colAccum, 0.0, 1.0)), half(saturate(aAccum)));
  }

  return half4(0.0h);
}

kernel void volume_compute_march_cinematic(
    texture2d<half, access::write> outColorTexture [[texture(4)]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(5)]],
    texture2d<float> transferFunctionTexture [[texture(6)]],
    texture2d<float> transferFunctionTexture1 [[texture(7)]],
    texture2d<float> transferFunctionTexture2 [[texture(8)]],
    texture2d<float> transferFunctionTexture3 [[texture(9)]],
    texture2d<float> transferFunction2DTexture [[texture(10)]],
    texture3d<float> transfer2DYAxisTexture [[texture(11)]],
    texture2d<float> gradientOpacityTexture [[texture(12)]],
    texture3d<float> maskTexture [[texture(13)]],
    texture2d<float> labelMapTransferTexture [[texture(14)]],
    texture3d<float> minMaxTexture [[texture(15)]],
    texture3d<float> minMaxBlockTexture [[texture(16)]],
    texture3d<float> minMaxSuperTexture [[texture(17)]],
    texture3d<float> normalTexture [[texture(18)]],
    texture3d<float> blankingTexture [[texture(19)]],
    constant packed_float3* rectCoords [[buffer(3)]],
    constant VolumeLightUniforms& volumeLights [[buffer(4)]],
    texture2d<float> depthSynthTexture [[texture(20)]],
    texture2d<half, access::read> accumPrevTexture [[texture(21)]],
    uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(volumeUniforms.viewportSize.x) || gid.y >= uint(volumeUniforms.viewportSize.y)) return;
  if (volumeUniforms.cinematicEnabled < 0.5) { outColorTexture.write(half4(0.0h), gid); return; }
  SynthRay so;
  if (!synthesizeAtlasRay(gid, volumeUniforms, b, volumeTexture, depthSynthTexture, so)) {
    half4 prev = half4(0.0h);
    if (volumeUniforms.cinematicAccumCount > 1) prev = accumPrevTexture.read(gid);
    half4 curr = half4(0.0h,0.0h,0.0h,0.0h);
    outColorTexture.write(cine_accum(curr, prev, volumeUniforms.cinematicAccumCount), gid);
    return;
  }
  half4 curr = cinematic_march_core(so.evalPoint, so.steps, so.evalStep, so.stepSize, so.rayDir, so.tTerminateMax,
    volumeUniforms, b, volumeTexture, transferFunctionTexture, transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3, transferFunction2DTexture, transfer2DYAxisTexture, gradientOpacityTexture, maskTexture, labelMapTransferTexture, minMaxTexture, minMaxBlockTexture, minMaxSuperTexture, normalTexture, blankingTexture, rectCoords, &volumeLights, volumeUniforms.cinematicFrameSeed, gid);
  half4 prev = half4(0.0h);
  if (volumeUniforms.cinematicAccumCount > 1) prev = accumPrevTexture.read(gid);
  outColorTexture.write(cine_accum(curr, prev, volumeUniforms.cinematicAccumCount), gid);
}

// Edge-aware 5x5 bilateral denoise for cinematic (removes Image1 grain while keeping sulci edges).
kernel void volume_cinematic_denoise(
    texture2d<half, access::read> inTexture [[texture(0)]],
    texture2d<half, access::write> outTexture [[texture(1)]],
    constant float& denoiseWeight [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) return;
  half4 c0 = inTexture.read(gid);
  // Transparent background stays transparent
  if (c0.a < 0.01h) { outTexture.write(c0, gid); return; }
  half4 sum = c0;
  float sumW = 1.0;
  // 5x5 window, skip center
  for (int dy=-2; dy<=2; ++dy) {
    for (int dx=-2; dx<=2; ++dx) {
      if (dx==0 && dy==0) continue;
      int2 p = int2(gid) + int2(dx,dy);
      if (p.x <0 || p.y<0 || p.x >= int(inTexture.get_width()) || p.y >= int(inTexture.get_height())) continue;
      half4 n = inTexture.read(uint2(p));
      float colorDist = length(float3(n.rgb - c0.rgb));
      // Bilateral: color sigma ~0.25 (exp -dist*4), spatial weight by radius
      float spatialW = (abs(dx)+abs(dy) <=1) ? 1.0 : (abs(dx)<=1 && abs(dy)<=1 ? 0.55 : 0.30);
      float w = exp(-colorDist * 4.0) * spatialW;
      // Don't blur across strong opacity edges (keep sulci crisp)
      float alphaDist = fabs(float(n.a - c0.a));
      w *= exp(-alphaDist * 6.0);
      sum += n * half(w);
      sumW += w;
    }
  }
  half4 blurred = sum / half(sumW);
  half t = half(clamp(denoiseWeight, 0.0, 1.0));
  half4 out = mix(c0, blurred, t);
  // Preserve solid alpha
  out.a = c0.a;
  outTexture.write(out, gid);
}

// ============================================================================
// PathTraced helpers — unbiased Woodcock volume path tracer (steps 1-6)
// Preview DVR uses cinematic_march_core; PT stills use volume_path_trace.
// ============================================================================
inline float halton_pt(uint idx, int base) {
  float f = 1.0; float r = 0.0;
  float b = float(base);
  while (idx > 0) { f /= b; r += f * float(idx % uint(base)); idx /= uint(base); }
  return r;
}
inline uint pt_pcg_hash(uint x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16; return x; }
inline float pt_rand(thread uint &s) { s = pt_pcg_hash(s); return float(s & 0x00FFFFFF) / 16777216.0; }
inline float hg_phase_pt(float cosTheta, float g) {
  float g2 = g*g;
  float denom = 1.0 + g2 - 2.0 * g * cosTheta;
  return (1.0 - g2) / (4.0 * M_PI_F * denom * sqrt(max(denom, 1e-6)));
}
inline float3 hg_sample_dir_pt(float3 wo, float g, thread uint &rng) {
  float xi1 = pt_rand(rng); float xi2 = pt_rand(rng);
  float cosTheta;
  if (abs(g) < 1e-3) cosTheta = 1.0 - 2.0 * xi1;
  else {
    float sq = (1.0 - g*g) / (1.0 - g + 2.0 * g * xi1);
    cosTheta = (1.0 + g*g - sq*sq) / (2.0 * g);
    cosTheta = clamp(cosTheta, -1.0, 1.0);
  }
  float sinTheta = sqrt(max(0.0, 1.0 - cosTheta*cosTheta));
  float phi = 2.0 * M_PI_F * xi2;
  float3 up = fabs(wo.z) < 0.999 ? float3(0,0,1) : float3(1,0,0);
  float3 t = normalize(cross(up, wo));
  float3 b = cross(wo, t);
  return normalize(t * cos(phi) * sinTheta + b * sin(phi) * sinTheta + wo * cosTheta);
}
inline float3 sampleMediumAlbedo(device float4 *table, float s, float sMin, float sMax) {
  float u = saturate((s - sMin) / max(sMax - sMin, 1e-6));
  uint idx = min(uint(u * 1023.0 + 0.5), 1023u);
  return table[idx].yzw; // yzw = albedo
}
inline float sampleMediumSigma(device float4 *table, float s, float sMin, float sMax) {
  float u = saturate((s - sMin) / max(sMax - sMin, 1e-6));
  uint idx = min(uint(u * 1023.0 + 0.5), 1023u);
  return table[idx].x; // sigma_t
}

// Step 0: grey running-mean test — proves PT accum/blit/reset without volume logic.
// Writes constant grey into HDR sum (RGBA32F) via ping-pong running mean.
// spp = cinematicAccumCount (1..cap). sum.rgb = prev.rgb + L, sum.a = spp
kernel void volume_path_trace(
    texture2d<float, access::write> accumCurr [[texture(4)]],
    texture2d<float, access::read> accumPrev [[texture(21)]],
    texture2d<float> depthSynth [[texture(20)]],
    constant VolumeMapperUniforms& u [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]],
    texture3d<float> volumeTexture [[texture(5)]],
    texture2d<float> transferFunctionTexture [[texture(6)]],
    texture2d<float> transferFunctionTexture1 [[texture(7)]],
    texture2d<float> transferFunctionTexture2 [[texture(8)]],
    texture2d<float> transferFunctionTexture3 [[texture(9)]],
    texture2d<float> transferFunction2DTexture [[texture(10)]],
    texture3d<float> transfer2DYAxisTexture [[texture(11)]],
    texture2d<float> gradientOpacityTexture [[texture(12)]],
    texture3d<float> maskTexture [[texture(13)]],
    texture2d<float> labelMapTransferTexture [[texture(14)]],
    texture3d<float> minMaxTexture [[texture(15)]],
    texture3d<float> minMaxBlockTexture [[texture(16)]],
    texture3d<float> minMaxSuperTexture [[texture(17)]],
    texture3d<float> normalTexture [[texture(18)]],
    texture3d<float> blankingTexture [[texture(19)]],
    constant packed_float3* rectCoords [[buffer(3)]],
    constant VolumeLightUniforms& volumeLights [[buffer(4)]],
    device float4 *mediumTable [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(u.viewportSize.x) || gid.y >= uint(u.viewportSize.y)) return;
  if (u.cinematicEnabled < 0.5 || u.cinematicQuality != 1) { accumCurr.write(float4(0.0), gid); return; }
  uint spp = max(u.cinematicAccumCount, 1u);
  float4 prev = float4(0.0);
  if (spp > 1) prev = accumPrev.read(gid);
  // Woodcock delta-tracking volume path tracer (steps 1-3)
  // For step 3 black test: env=0 => black; env=0.2 => silhouette; with bounces/g
  SynthRay so;
  float3 L = float3(0.0);
  if (synthesizeAtlasRay(gid, u, b, volumeTexture, depthSynth, so)) {
    uint rng = pt_pcg_hash(gid.x * 1973u + gid.y * 9277u + u.cinematicFrameSeed * 104729u + 17u);
    float2 hv = float2(halton_pt(spp, 2), halton_pt(spp, 3));
    // Jitter inside pixel is already handled by atlas? Keep Halton for future.
    float sigma_maj = max(u.cinematicMajorantSigma, 1e-4);
    float sMin = float(u.scalarMin);
    float sMax = float(u.scalarMax);
    float g = clamp(u.cinematicScatteringAnisotropy, -0.9, 0.9);
    float env = u.cinematicEnv;
    float3 o = so.evalPoint;
    float3 d = normalize(so.rayDir);
    float3 beta = float3(1.0);
    uint maxBounces = max(u.cinematicMaxBounces, 1u);
    float3 curO = o;
    float3 curD = d;
    bool scattered = false; bool addedEnv = false;
    for (uint bounce = 0; bounce < maxBounces; ++bounce) {
      float3 o2 = curO + curD * 1e-4;
      float2 tBox = intersectBox(o2, curD, float3(0.0), float3(1.0));
      float tExit = tBox.y;
      if (tExit <= 1e-6) { if (scattered) { L += beta * env; addedEnv = true; } break; }
      float t = 0.0;
      float3 hitPos = float3(0.0);
      bool realHit = false;
      int iter = 0;
      bool truncated = false;
      while (t < tExit && iter++ < 1024) {
        float xi = pt_rand(rng);
        float dt = -log(max(1.0 - xi, 1e-6)) / sigma_maj;
        t += dt;
        if (t >= tExit) break;
        float3 x = o2 + curD * t;
        if (any(x < 0.0) || any(x > 1.0)) break;
        float s = sampleVolumeScalar(volumeTexture, x);
        float sigma = sampleMediumSigma(mediumTable, s, sMin, sMax);
        float xi2 = pt_rand(rng);
        if (xi2 * sigma_maj < sigma) { hitPos = x; realHit = true; break; }
      }
      if (iter >= 1024 && t < tExit && !realHit) truncated = true;
      if (truncated) break;
      if (!realHit) { if (scattered) { L += beta * env; addedEnv = true; } break; }
      float sHit = sampleVolumeScalar(volumeTexture, hitPos);
      float3 albedo = clamp(sampleMediumAlbedo(mediumTable, sHit, sMin, sMax), 0.0, 0.99);
      float avgA = clamp((albedo.r + albedo.g + albedo.b) / 3.0, 0.0, 0.99);
      // Surface mix: filtered N at 2.2x + raw gmag at 1.5x (volume kill inside branch)
      // Surface mix: filtered N at 2.2x + raw gmag at 1.5x
      float3 Nsurf = float3(0,0,1);
      float gmagSurf = 0.0;
      {
        float3 gsN = b.gradientStep.xyz * 2.2;
        float3 gsBlur = b.gradientStep.xyz;
        float3 gsMag = b.gradientStep.xyz * 1.5;
        float3 gradN = float3(
          sample_blur(volumeTexture, hitPos + float3(gsN.x,0,0), gsBlur) - sample_blur(volumeTexture, hitPos - float3(gsN.x,0,0), gsBlur),
          sample_blur(volumeTexture, hitPos + float3(0,gsN.y,0), gsBlur) - sample_blur(volumeTexture, hitPos - float3(0,gsN.y,0), gsBlur),
          sample_blur(volumeTexture, hitPos + float3(0,0,gsN.z), gsBlur) - sample_blur(volumeTexture, hitPos - float3(0,0,gsN.z), gsBlur));
        float3 gradM = float3(
          sampleVolumeScalar(volumeTexture, hitPos + float3(gsMag.x,0,0)) - sampleVolumeScalar(volumeTexture, hitPos - float3(gsMag.x,0,0)),
          sampleVolumeScalar(volumeTexture, hitPos + float3(0,gsMag.y,0)) - sampleVolumeScalar(volumeTexture, hitPos - float3(0,gsMag.y,0)),
          sampleVolumeScalar(volumeTexture, hitPos + float3(0,0,gsMag.z)) - sampleVolumeScalar(volumeTexture, hitPos - float3(0,0,gsMag.z)));
        float glenN = length(gradN);
        gmagSurf = saturate((length(gradM) - 0.018) * 5.5);
        Nsurf = (glenN > 1e-6) ? normalize(-gradN) : -curD;
        if (dot(Nsurf, -curD) < 0.0) Nsurf = -Nsurf;
      }
      float w_surf = smoothstep(0.06, 0.22, gmagSurf) * saturate(u.cinematicBlend);
      float xiSurf = pt_rand(rng);
      if (xiSurf < w_surf) {
        // Surface event: Lambert diffuse + NEE with cosine, then cosine hemisphere bounce
        float3 N = Nsurf;
        // NEE for surface: Lambert BRDF = albedo/pi
        {
          float3 lightCenter = volumeLights.lights[0].position.xyz;
          float lightRadius = volumeLights.lights[0].attenuation.x;
          float3 lightRadiance = volumeLights.lights[0].diffuseColor.xyz;
          float3 Nlight = volumeLights.lights[0].direction.xyz;
          float nlen = length(Nlight);
          if (nlen < 1e-3) Nlight = float3(0,0,-1); else Nlight /= nlen;
          if (lightRadius > 1e-4 && dot(lightRadiance, lightRadiance) > 1e-6) {
            float r = sqrt(pt_rand(rng));
            float theta = 2.0 * M_PI_F * pt_rand(rng);
            float3 up = fabs(Nlight.z) < 0.999 ? float3(0,0,1) : float3(1,0,0);
            float3 tangent = normalize(cross(up, Nlight));
            float3 bitangent = cross(Nlight, tangent);
            float3 y = lightCenter + (tangent * cos(theta) + bitangent * sin(theta)) * r * lightRadius;
            float3 wi = y - hitPos;
            float dist = length(wi);
            if (dist > 1e-4) {
              wi /= dist;
              float cosLight = saturate(dot(Nlight, -wi));
              float cosSurf = saturate(dot(N, wi));
              if (cosLight > 1e-4 && cosSurf > 1e-4) {
                float pdfA = 1.0 / (M_PI_F * lightRadius * lightRadius);
                float G = cosLight / (dist*dist);
                float T = 1.0;
                float3 oS = hitPos + N * 1e-4;
                float tS = 0.0;
                float tExitS = dist - 2e-4;
                bool blocked = false;
                int sIter=0; while (tS < tExitS && sIter++ < 1024) {
                  float xiS = pt_rand(rng);
                  float dtS = -log(max(1.0 - xiS, 1e-6)) / sigma_maj;
                  tS += dtS;
                  if (tS >= tExitS) break;
                  float3 xs = oS + wi * tS;
                  if (any(xs < 0.0) || any(xs > 1.0)) break;
                  float sS = sampleVolumeScalar(volumeTexture, xs);
                  float sigmaS = sampleMediumSigma(mediumTable, sS, sMin, sMax);
                  float xiS2 = pt_rand(rng);
                  if (xiS2 * sigma_maj < sigmaS) { blocked = true; break; }
                }
                if (sIter >= 1024 && tS < tExitS && !blocked) blocked = true;
                if (blocked) T = 0.0;
                float3 brdf = albedo / M_PI_F;
                L += beta * lightRadiance * brdf * cosSurf * T * G / pdfA;
              }
            }
          }
        }
        // Cosine hemisphere bounce (diffuse)
        float r1 = pt_rand(rng); float r2 = pt_rand(rng);
        float phi = 2.0 * M_PI_F * r1;
        float cosTheta = sqrt(max(1.0 - r2, 0.0));
        float sinTheta = sqrt(r2);
        float3 up2 = fabs(N.z) < 0.999 ? float3(0,0,1) : float3(1,0,0);
        float3 tangent2 = normalize(cross(up2, N));
        float3 bitangent2 = cross(N, tangent2);
        curD = normalize(tangent2 * cos(phi) * sinTheta + bitangent2 * sin(phi) * sinTheta + N * cosTheta);
        curO = hitPos + N * 1e-4;
        beta *= albedo; // brdf/pdf = albedo
        scattered = true;
        // If bounce goes into medium (dot <0) continue Woodcock, else reflect (rare)
      } else {
        float xiAbsV = pt_rand(rng);
        if (xiAbsV > avgA) break;
        beta *= albedo / max(avgA, 1e-4);
        scattered = true;
        // Volume NEE (HG)
        {
          float3 lightCenter = volumeLights.lights[0].position.xyz;
          float lightRadius = volumeLights.lights[0].attenuation.x;
          float3 lightRadiance = volumeLights.lights[0].diffuseColor.xyz;
          float3 Nlight = volumeLights.lights[0].direction.xyz;
          float nlen = length(Nlight);
          if (nlen < 1e-3) Nlight = float3(0,0,-1); else Nlight /= nlen;
          if (lightRadius > 1e-4 && dot(lightRadiance, lightRadiance) > 1e-6) {
            float r = sqrt(pt_rand(rng));
            float theta = 2.0 * M_PI_F * pt_rand(rng);
            float3 up = fabs(Nlight.z) < 0.999 ? float3(0,0,1) : float3(1,0,0);
            float3 tangent = normalize(cross(up, Nlight));
            float3 bitangent = cross(Nlight, tangent);
            float3 y = lightCenter + (tangent * cos(theta) + bitangent * sin(theta)) * r * lightRadius;
            float3 wi = y - hitPos;
            float dist = length(wi);
            if (dist > 1e-4) {
              wi /= dist;
              float cosLight = saturate(dot(Nlight, -wi));
              if (cosLight > 1e-4) {
                float pdfA = 1.0 / (M_PI_F * lightRadius * lightRadius);
                float G = cosLight / (dist*dist);
                float T = 1.0;
                float3 oS = hitPos + wi * 1e-4;
                float tS = 0.0;
                float tExitS = dist - 2e-4;
                bool blocked = false;
                int sIter=0; while (tS < tExitS && sIter++ < 1024) {
                  float xiS = pt_rand(rng);
                  float dtS = -log(max(1.0 - xiS, 1e-6)) / sigma_maj;
                  tS += dtS;
                  if (tS >= tExitS) break;
                  float3 xs = oS + wi * tS;
                  if (any(xs < 0.0) || any(xs > 1.0)) break;
                  float sS = sampleVolumeScalar(volumeTexture, xs);
                  float sigmaS = sampleMediumSigma(mediumTable, sS, sMin, sMax);
                  float xiS2 = pt_rand(rng);
                  if (xiS2 * sigma_maj < sigmaS) { blocked = true; break; }
                }
                if (sIter >= 1024 && tS < tExitS && !blocked) blocked = true;
                if (blocked) T = 0.0;
                float p_hg = hg_phase_pt(dot(wi, -curD), g);
                L += beta * lightRadiance * p_hg * T * G / pdfA;
              }
            }
          }
        }
        curO = hitPos;
        curD = hg_sample_dir_pt(-curD, g, rng);
      }
      if (bounce >= 2) {
        float xiRR = pt_rand(rng);
        if (xiRR > 0.5) break;
        beta /= 0.5;
      }
    }
    // If we scattered but ran out of bounces, still need to connect to env (faint brain silhouette)
    if (scattered && !addedEnv && env > 1e-6) {
      float3 oF = curO + curD * 1e-4;
      float2 tBoxF = intersectBox(oF, curD, float3(0.0), float3(1.0));
      float tExitF = tBoxF.y;
      bool hitF = false;
      float tF = 0.0;
      int fIter=0; while (tF < tExitF && fIter++ < 1024) {
        float xiF = pt_rand(rng);
        float dtF = -log(max(1.0 - xiF, 1e-6)) / sigma_maj;
        tF += dtF;
        if (tF >= tExitF) break;
        float3 xF = oF + curD * tF;
        if (any(xF < 0.0) || any(xF > 1.0)) break;
        float sF = sampleVolumeScalar(volumeTexture, xF);
        float sigmaF = sampleMediumSigma(mediumTable, sF, sMin, sMax);
        float xiF2 = pt_rand(rng);
        if (xiF2 * sigma_maj < sigmaF) { hitF = true; break; }
      }
      if (fIter >= 1024 && tF < tExitF && !hitF) hitF = true;
      if (!hitF) L += beta * env;
    }
  } else {
    L = float3(0.0);
  }
  float4 sum;
  sum.rgb = prev.rgb + L;
  sum.a = float(spp);
  accumCurr.write(sum, gid);
}

inline float3 aces_tonemap(float3 x) {
  // ACES fitted (Narkowicz)
  float3 a = x * (2.51 * x + 0.03);
  float3 b = x * (2.43 * x + 0.59) + 0.14;
  return saturate(a / b);
}
inline float3 linear_to_srgb_pt(float3 c) {
  return pow(saturate(c), float3(1.0/2.2));
}
kernel void volume_cinematic_tonemap(
    texture2d<float, access::read> accum [[texture(0)]],
    texture2d<half, access::write> outDisplay [[texture(1)]],
    constant VolumeMapperUniforms& u [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= uint(u.viewportSize.x) || gid.y >= uint(u.viewportSize.y)) return;
  float4 sum = accum.read(gid);
  float spp = max(sum.a, 1.0);
  if (sum.a < 0.5) spp = float(max(u.cinematicAccumCount, 1u));
  float3 hdr = sum.rgb / spp;
  float3 mapped = aces_tonemap(hdr * 1.5); // exposure 1.5 for wax
  float3 srgb = linear_to_srgb_pt(mapped);
  float alpha = length(hdr) > 1e-4 ? 1.0 : 0.0; // opaque where we scattered, transparent background
  if (u.cinematicQuality != 1) alpha = 1.0; // preview keep opaque
  outDisplay.write(half4(half3(srgb), half(alpha)), gid);
}

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
        // volTransposed carries the orientation code: 0=identity, 1=X-depth
        // (tex holds z,y,x), 2=Y-depth (tex holds x,z,y).
        float3 tpos = pos;
        if (u.volTransposed == 1)      tpos = pos.zyx;
        else if (u.volTransposed == 2) tpos = pos.xzy;
        float v = volume.sample(sNearest, tpos, level(0)).r;
        cellMin = min(cellMin, v);
        cellMax = max(cellMax, v);
      }
    }
  }

  // Check emptiness via opacity prefix table. VTK_METAL_TEST_MM_EPS (§33.2
  // item 2): when mmEps > 0 the predicate becomes "max achievable opacity in
  // [idxMin,idxMax] <= eps" — an approximation that marks barely-visible
  // cells empty so the walk can skip them. eps == 0 reduces to the exact
  // zero-opacity semantics of the prefix sum.
  if (cellMin <= cellMax) {
    int iMin = int(floor((cellMin - u.scalarMin) * u.scalarScale));
    int iMax = int(floor((cellMax - u.scalarMin) * u.scalarScale));
    uint idxMin = uint(clamp(iMin, 0, 255));
    uint idxMax = uint(clamp(iMax, 0, 255));
    bool empty = false;
    if (u.mmEps > 0.0) {
      float maxOp = 0.0;
      for (uint i = idxMin; i <= idxMax; ++i) {
        maxOp = max(maxOp, u.opacityLut[i]);
      }
      empty = (maxOp <= u.mmEps);
    } else {
      empty = (u.opacityPrefix[idxMax + 1] == u.opacityPrefix[idxMin]);
    }
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

// Two-level occupancy summary (VTK_METAL_TEST_MM_BLOCKS): one thread per
// block texel of the DILATED fine lattice; writes a three-state mark per
// block — 1.0 all-empty (leap through it), 0.0 mixed (per-cell walk),
// 0.5 all-solid (no cell can be skipped: suspend per-cell checking until the
// block boundary). blockSize arrives as a single uint. emptyCount accumulates
// the number of all-empty blocks so the CPU can gate the walk off for
// transfer functions whose lattices are too solid for skipping to pay (a
// static per-TF decision, resolved once per rebuild).
kernel void volume_reduce_minmax_blocks(
    texture3d<float, access::read> fine [[texture(0)]],
    texture3d<float, access::write> blocks [[texture(1)]],
    constant uint& blockSize [[buffer(0)]],
    device atomic_uint* emptyCount [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]])
{
  uint3 bdims = uint3(blocks.get_width(), blocks.get_height(), blocks.get_depth());
  if (any(gid >= bdims)) return;

  uint3 fdims = uint3(fine.get_width(), fine.get_height(), fine.get_depth());
  uint3 start = gid * blockSize;
  uint3 end = min(start + blockSize, fdims);

  bool allEmpty = true;
  bool allSolid = true;
  for (uint z = start.z; z < end.z && (allEmpty || allSolid); z++) {
    for (uint y = start.y; y < end.y && (allEmpty || allSolid); y++) {
      for (uint x = start.x; x < end.x && (allEmpty || allSolid); x++) {
        if (fine.read(uint3(x, y, z)).r < 0.5) {
          allEmpty = false;
        } else {
          allSolid = false;
        }
      }
    }
  }

  blocks.write(allEmpty ? 1.0 : (allSolid ? 0.5 : 0.0), gid);
  if (allEmpty) {
    atomic_fetch_add_explicit(emptyCount, 1u, memory_order_relaxed);
  }
}

// §38.16 (VTK_METAL_TEST_MM_MIP -> fc_mmMip): write the block summary into
// mip level dstLod of the FINE lattice instead of the standalone block
// texture, so the marcher's block taps read the same resource as the cell
// taps (§38.15/38.16: the standalone summary resource carries a catastrophic
// small-viewport access tax that survives format/path/slot/padding changes;
// reads from the fine-lattice resource measured free). The reduce logic is
// byte-identical to volume_reduce_minmax_blocks (same clamped partial-edge
// loops, same three-state encoding), so values match the standalone texture
// exactly; a blit mirrors level dstLod back into the standalone texture for
// its other consumers.
kernel void volume_reduce_minmax_mipblocks(
    texture3d<float, access::read> fine [[texture(0)]],
    texture3d<float, access::write> pyr [[texture(1)]],
    constant uint& blockSize [[buffer(0)]],
    device atomic_uint* emptyCount [[buffer(1)]],
    constant uint& dstLod [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
  uint3 pdims = uint3(pyr.get_width(dstLod), pyr.get_height(dstLod),
                      pyr.get_depth(dstLod));
  if (any(gid >= pdims)) return;

  uint3 fdims = uint3(fine.get_width(), fine.get_height(), fine.get_depth());
  uint3 start = gid * blockSize;
  uint3 end = min(start + blockSize, fdims);

  bool allEmpty = true;
  bool allSolid = true;
  for (uint z = start.z; z < end.z && (allEmpty || allSolid); z++) {
    for (uint y = start.y; y < end.y && (allEmpty || allSolid); y++) {
      for (uint x = start.x; x < end.x && (allEmpty || allSolid); x++) {
        if (fine.read(uint3(x, y, z)).r < 0.5) {
          allEmpty = false;
        } else {
          allSolid = false;
        }
      }
    }
  }

  pyr.write(allEmpty ? 1.0 : (allSolid ? 0.5 : 0.0), gid, dstLod);
  if (allEmpty) {
    atomic_fetch_add_explicit(emptyCount, 1u, memory_order_relaxed);
  }
}

// §35.5 headroom A/B (VTK_METAL_TEST_MM_SUPER): third occupancy level — one
// thread per super-block texel of the BLOCK summary; writes 1.0 when every
// covered block is all-empty (leap through the whole group), else 0.0.
// Derived from the block semantics, so the composited sample set is unchanged;
// only jump granularity differs (same argument as the blocks level).
kernel void volume_reduce_minmax_superblocks(
    texture3d<float, access::read> blocks [[texture(0)]],
    texture3d<float, access::write> supers [[texture(1)]],
    constant uint& blocksPerSuper [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]])
{
  uint3 sdims = uint3(supers.get_width(), supers.get_height(), supers.get_depth());
  if (any(gid >= sdims)) return;

  uint3 bdims = uint3(blocks.get_width(), blocks.get_height(), blocks.get_depth());
  const uint bps = max(blocksPerSuper, 1u);
  uint3 start = gid * bps;
  uint3 end = min(start + bps, bdims);

  bool allEmpty = true;
  for (uint z = start.z; z < end.z && allEmpty; z++) {
    for (uint y = start.y; y < end.y && allEmpty; y++) {
      for (uint x = start.x; x < end.x && allEmpty; x++) {
        if (blocks.read(uint3(x, y, z)).r < 0.75) {
          allEmpty = false;
        }
      }
    }
  }

  supers.write(allEmpty ? 1.0 : 0.0, gid);
}

// ---------------------------------------------------------------------------
// §28/§29 GPU volume transpose (VTK_METAL_TEST_GPU_TRANSPOSE): replaces the
// CPU blocked repack for the transposed-volume upload. One thread per SOURCE
// texel; reads coalesce along source-x from the staging buffer and writes land
// at the transposed coordinate in the swapped-dims destination texture.
// trMode selects the orientation: 1 = X-depth (dst(z,y,x), texture (D,H,W)),
// 2 = Y-depth (dst(x,z,y), texture (W,D,H)). R8Unorm round-trip is byte-exact
// (b/255*255 rounds back to b).
kernel void volume_transpose_xz(
    device const unsigned char* src [[buffer(0)]],
    texture3d<float, access::write> dst [[texture(1)]],
    constant uint4& srcDimsPad [[buffer(2)]],   // (W,H,D,0) as uint4
    constant uint& trMode [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
  uint3 srcDims = srcDimsPad.xyz;
  if (any(gid >= srcDims)) return;
  size_t idx = (static_cast<size_t>(gid.z) * srcDims.y + gid.y) * srcDims.x + gid.x;
  float v = static_cast<float>(src[idx]) / 255.0f;
  if (trMode == 2) {
    // Y-depth: T(u=x, v=z, w=y) = V(x,y,z)
    dst.write(float4(v, 0.0f, 0.0f, 1.0f), uint3(gid.x, gid.z, gid.y));
  } else {
    // X-depth: T(x'=z, y'=y, z'=x) = V(x,y,z); the destination's width extent
    // holds the original slice axis.
    dst.write(float4(v, 0.0f, 0.0f, 1.0f), uint3(gid.z, gid.y, gid.x));
  }
}

fragment float4 fragment_image_sample_blit(
    FullscreenVertexOut in [[stage_in]],
    texture2d<float> offscreenColor [[texture(0)]]) {
  return offscreenColor.sample(sVolume, in.texCoord);
}

// ---------------------------------------------------------------------------
// RayCastImageDisplayHelper: draw a CPU-produced ray-cast image (vtkFixedPoint
// VolumeRayCastMapper) as a textured quad at a viewport region, matching
// vtkOpenGLRayCastImageDisplayHelper.
// ---------------------------------------------------------------------------
struct RayCastDisplayVertex {
  float x, y, z, w;  // NDC xyz (w unused) - plain floats, 16 bytes
  float u, v;        // 8 bytes -> 24-byte stride matching the C++ side
};

struct RayCastDisplayVertexOut {
  float4 position [[position]];
  float2 texCoord;
};

vertex RayCastDisplayVertexOut vertex_raycast_display(
    const device RayCastDisplayVertex* verts [[buffer(0)]],
    uint vertex_id [[vertex_id]]) {
  RayCastDisplayVertexOut out;
  out.position = float4(verts[vertex_id].x, verts[vertex_id].y, verts[vertex_id].z, 1.0);
  out.texCoord = float2(verts[vertex_id].u, verts[vertex_id].v);
  return out;
}

fragment float4 fragment_raycast_display(
    RayCastDisplayVertexOut in [[stage_in]],
    texture2d<float> source [[texture(0)]],
    sampler sourceSampler [[sampler(0)]],
    constant float& scale [[buffer(0)]]) {
  return source.sample(sourceSampler, in.texCoord) * scale;
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

// ---------------------------------------------------------------------------
// Projected tetrahedra (Shirley & Tuchman / Wylie 2002). The CPU-side mapper
// (vtkMetalProjectedTetrahedraMapper) packs interleaved vertices with a 24-byte
// stride: clip-space xyz (float3), packed RGBA color (uchar4), attenuation
// (float), depth (float). The clip-space coordinates are already divided by w
// (NDC), matching vtkglProjectedTetrahedraVS which emits gl_Position = vertexDC.
// ---------------------------------------------------------------------------
struct ProjectedTetrahedraVertexOut {
  float4 position [[position]];
  float3 fcolor;
  float fdepth;
  float fattenuation;
};

vertex ProjectedTetrahedraVertexOut vertex_projected_tetrahedra_main(
    const device float* positions [[buffer(0)]],
    const device uchar4* colors [[buffer(1)]],
    const device float2* attenuationDepth [[buffer(2)]],
    uint vertex_id [[vertex_id]]) {
  ProjectedTetrahedraVertexOut out;
  uint base = 3 * vertex_id;
  out.position = float4(positions[base], positions[base + 1], positions[base + 2], 1.0);
  // GL's scalarColor attribute is VTK_UNSIGNED_CHAR with normalized=true
  // (3 components), so convert [0,255] -> [0,1] here.
  out.fcolor = float3(colors[vertex_id].rgb) / 255.0;
  out.fattenuation = attenuationDepth[vertex_id].x;
  out.fdepth = attenuationDepth[vertex_id].y;
  return out;
}

fragment float4 fragment_projected_tetrahedra_main(
    ProjectedTetrahedraVertexOut in [[stage_in]]) {
  float a = 1.0 - exp(-in.fattenuation * in.fdepth);
  if (a <= 0.0) {
    discard_fragment();
  }
  return float4(in.fcolor, a);
}