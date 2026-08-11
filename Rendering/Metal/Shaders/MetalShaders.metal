// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
//
// Metal shaders for VTK Metal rendering backend.
//

#include <metal_stdlib>
using namespace metal;

// Metal shader logging (os_log, Metal 3.2 / macOS 15+): the volume ray-cast
// mapper compiles this source with the VTK_METAL_ENABLE_LOGGING preprocessor
// macro (and -fmetal-enable-logging via MTLCompileOptions.enableLogging) in
// test builds, so the os_log calls below are only compiled into the volume
// shader library there. Every other library compiled from this source passes
// no such macro, so the #if blocks vanish and the shaders behave exactly as
// before. Messages are held in the Metal log buffer until the process is
// launched with MTL_LOG_LEVEL set (they are dropped otherwise), e.g.:
//
//   MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_TO_STDERR=1 ctest -R TestMetalVolumeShaderLog
//
// which forwards them to the test's stderr.
#if defined(VTK_METAL_ENABLE_LOGGING)
#include <metal_logging>
#endif

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

// Scene flag bits (must match VTK_METAL_SCENE_FLAG_* in vtkMetalPolyDataMapper.mm).
constant uint kSceneFlagParallelProjection  = 1u << 0;
constant uint kSceneFlagVertexVisibility    = 1u << 3;
constant uint kSceneFlagSpherePoints        = 1u << 5;
constant uint kSceneFlagPointShape          = 1u << 7;
constant uint kSceneFlagHasSurfaceColors    = 1u << 8;
constant uint kSceneFlagHasActorTexture     = 1u << 9;
constant uint kSceneFlagHasSurfaceAlpha     = 1u << 10;
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
  uint numEdges = primitiveCounts[gid + 1u] - primitiveCounts[gid];
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
struct FullscreenVertexOut {
  float4 position [[position]];
  float2 texCoord;
  float4 clipPos;  // debug: clip-space position (P*V*M*v), interpolated in the fragment
};

vertex FullscreenVertexOut vertex_fullscreen_main(uint vertex_id [[vertex_id]]) {
  const float2 positions[3] = { float2(-1, -1), float2( 3, -1), float2(-1,  3) };
  const float2 texCoords[3] = { float2(0, 1), float2(2, 1), float2(0, -1) };
  FullscreenVertexOut out;
  out.position = float4(positions[vertex_id], 0, 1);
  out.clipPos = out.position;
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
  float sampleDistance;             // full float32 (OpenGL in_sampleDistance parity; was half)
  float scalarMin;                  // full float32 (OpenGL in_volume_scale/in_volume_bias parity; was half)
  float scalarMax;                  // full float32 (OpenGL in_volume_scale/in_volume_bias parity; was half)
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
  // Stored as float32 (OpenGL in_scalarsRange parity; was half + pad pairs).
  float scalarMinComp[4];
  float scalarMaxComp[4];
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
  // World-unit sample distance (OpenGL in_sampleDistance value, NOT divided by
  // maxBoundsSize). marchVolumeUnified builds evalStep with GL's g_dirStep
  // float32 chain (object-space normalize * adjustedLin * this).
  float sampleDistanceWorld;
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
  // OpenGL ComputeClipPosition parity: the GL vertex shader computes
  // gl_Position = in_projectionMatrix * in_modelViewMatrix * in_volumeMatrix[0] * v
  // with three separate float32 uniforms multiplied in the shader. Feeding the
  // vertex shader the same three matrices (instead of a CPU-precomputed
  // viewProjection) reproduces GL's float32 intermediate rounding, so the
  // window-space barycentric weights — and therefore the interpolated
  // data-space cap anchor — are bit-identical. Rows 0,1,3 of projectionMatrix
  // match GL's (nearz=0 vs -1 only changes the Z row, irrelevant to XY/w).
  float4x4 projectionMatrix;
  float4x4 modelViewMatrix;
  // OpenGL in_eyePosObjs[0] parity: object-space eye cast to float32 (see CPU
  // EyePosData). Used directly for normalize(anchorData - eyePosData).
  float4 eyePosData;
  // TEMP DEBUG (anchor A/B): additive data-space anchor perturbation from
  // VTK_METAL_ANCHOR_PERTURB, applied before the dirObj normalize.
  float3 anchorPerturbData;
  // OpenGL ip_textureCoords parity: per-vertex cell-to-point texel adjustment
  // (in_cellToPoint scale + offset), applied in the vertex shader so the
  // interpolated ray anchor matches GL's per-vertex float-rounded texcoord.
  float3 cellToPointScale;
  float3 cellToPointOffset;
  // OpenGL in_inversePVM parity: CPU-composed
  // inverseVolumeMatrix * inverseModelViewMatrix * inverseProjectionMatrix
  // (double-precision vtk product, cast to float32). The camera-inside proxy
  // path unprojects the fragment through the near/far planes with this matrix
  // to build the ray direction analytically (GL computeRayDirection parity),
  // removing the interpolated-anchor dependence that drifted evalStep from
  // g_dirStep by ~3e-8/step.
  float4x4 inversePVM;
  // TEMP DEBUG (analytic-anchor experiment): when > 0.5 the camera-inside
  // proxy fragment shader bypasses the interpolated in.texcoord anchor and
  // reconstructs the per-fragment texcoord from pixel-center barycentrics +
  // per-vertex clip/texcoord (triangle anchor buffer at fragment buffer(3)).
  // 1 = float32 weights, 2 = float64 weights (update 76 sect 4 A/B).
  float analyticAnchorMode;
  float _padAnchor[3];
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
  float4 clipPos;  // debug: exact clip-space position (P*V*M*v), interpolated in the fragment
  uint instanceID [[flat]];
  uint flatVid [[flat]];  // debug: provoking-vertex index (covering triangle) for GL parity
  float3 texcoord;  // OpenGL ip_textureCoords parity: cellToPoint-adjusted per-vertex texture coord
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

// The Metal compiler is invoked with fast-math enabled (MTLCompileOptions
// defaults to fastMathEnabled=YES), which reassociates floating-point
// expressions and contracts mul+add into FMA. The clip chain below must use
// strict [0,1,2,3] mul+add in both the matrix product and the vector multiply
// to match the GL driver's compiled arithmetic bit-for-bit; disable
// reassociation and contraction for the following functions so the exact
// ordering and rounding survive compilation.
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)

// Strict [0,1,2,3] 4x4 multiply matching the GLSL mat4 operator codegen
// (column-major float4x4, so (A*B)[c][r] = sum_k A[k][r] * B[c][k]), with the
// same contraction pattern as GLSL dot products: first term plain multiply,
// remaining terms fused (fma). Verified bit-exact against GL's clip on the
// camera-inside cap mesh.
inline float4x4 matrixMulStrict(const float4x4 A, const float4x4 B)
{
  float4x4 R;
  for (int c = 0; c < 4; ++c)
  {
    for (int r = 0; r < 4; ++r)
    {
      float t = A[0][r] * B[c][0];
      t = fma(A[1][r], B[c][1], t);
      t = fma(A[2][r], B[c][2], t);
      t = fma(A[3][r], B[c][3], t);
      R[c][r] = t;
    }
  }
  return R;
}

// Strict [0,1,2,3] mat4*vec4 matching the GLSL mat4*vec4 operator codegen
// (first term plain multiply, middle terms fused, last term mul+add), the same
// contraction the GL driver applies to the analytic-pixel-ray unprojection.
inline float4 vecMulStrict(const float4x4 m, const float4 v)
{
  float4 r;
  r.x = m[0][0] * v.x;
  r.x = fma(m[1][0], v.y, r.x);
  r.x = fma(m[2][0], v.z, r.x);
  r.x += m[3][0] * v.w;
  r.y = m[0][1] * v.x;
  r.y = fma(m[1][1], v.y, r.y);
  r.y = fma(m[2][1], v.z, r.y);
  r.y += m[3][1] * v.w;
  r.z = m[0][2] * v.x;
  r.z = fma(m[1][2], v.y, r.z);
  r.z = fma(m[2][2], v.z, r.z);
  r.z += m[3][2] * v.w;
  r.w = m[0][3] * v.x;
  r.w = fma(m[1][3], v.y, r.w);
  r.w = fma(m[2][3], v.z, r.w);
  r.w += m[3][3] * v.w;
  return r;
}

vertex VolumeVertexOut vertex_volume_main(
    VolumeVertexIn in [[stage_in]],
    uint vertexId [[vertex_id]],
    constant VolumeMapperUniforms& volumeUniforms [[buffer(1)]],
    constant PerBlockData& b [[buffer(2)]]) {
  VolumeVertexOut out;

  // Camera-inside (useCameraInsideNearClip set): the vertex buffer holds
  // data-space proxy positions (OpenGL parity: GL uploads the clipped/densified
  // geometry in dataset space and interpolates in_vertexPos directly), so the
  // rasterizer interpolates in data space and the interpolated anchor matches
  // GL's ip_vertexPos to float32. Camera-outside keeps the unit-cube [0,1]
  // convention, scaled to the block's model-space bounds.
  float3 modelPos;
  if (volumeUniforms.useCameraInsideNearClip > 0.5)
  {
    modelPos = in.position;
  }
  else
  {
    modelPos = b.volumeBoundsMin.xyz + in.position * (b.volumeBoundsMax.xyz - b.volumeBoundsMin.xyz);
  }
  // OpenGL ComputeClipPosition parity: GL computes
  // gl_Position = in_projectionMatrix * in_modelViewMatrix * in_volumeMatrix[0] * v
  // with three separate float32 uniforms in the shader. Mirroring that exactly
  // (rather than consuming a CPU-precomputed viewProjection) keeps the
  // window-space barycentric weights — hence the interpolated data-space anchor
  // — bit-identical with GL.
  //
  // The built-in float4x4 multiply and matrix-vector multiply do not match the
  // GL driver's instruction order (≤1 ULP clip shifts that perturb the
  // interpolated anchor). Hand-write both the matrix product ((P*V)*W) and the
  // vector multiply in strict [0,1,2,3] mul+add order so the clip matches GL
  // bit-for-bit.
  float4x4 mvp = matrixMulStrict(
      matrixMulStrict(volumeUniforms.projectionMatrix, volumeUniforms.modelViewMatrix),
      volumeUniforms.volumeToWorld);
  float4 v = float4(modelPos, 1.0);
  // GLSL dot-product codegen (Apple GL on Metal): the first term is a plain
  // multiply, the second and third terms are fused (fma), and the last term is
  // a plain mul+add. Mirror that exact contraction pattern so the clip —
  // hence the interpolated anchor — matches GL bit-for-bit.
  float4 clip;
  float acc;
  acc = mvp[0][0] * v.x; acc = fma(mvp[1][0], v.y, acc); acc = fma(mvp[2][0], v.z, acc); acc = acc + mvp[3][0] * v.w; clip.x = acc;
  acc = mvp[0][1] * v.x; acc = fma(mvp[1][1], v.y, acc); acc = fma(mvp[2][1], v.z, acc); acc = acc + mvp[3][1] * v.w; clip.y = acc;
  acc = mvp[0][2] * v.x; acc = fma(mvp[1][2], v.y, acc); acc = fma(mvp[2][2], v.z, acc); acc = acc + mvp[3][2] * v.w; clip.z = acc;
  acc = mvp[0][3] * v.x; acc = fma(mvp[1][3], v.y, acc); acc = fma(mvp[2][3], v.z, acc); acc = acc + mvp[3][3] * v.w; clip.w = acc;
  out.position = clip;
  out.clipPos = out.position;
  if (volumeUniforms.useCameraInsideNearClip > 0.5)
  {
    out.localPos = modelPos;  // data-space (interpolated in data space like GL)
  }
  else
  {
    out.localPos = (modelPos - volumeUniforms.volumeBoundsMin.xyz) / max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  }
  out.instanceID = 0;
  out.flatVid = vertexId;
  // OpenGL ip_textureCoords parity: GL computes
  //   uvx = in_inverseTextureDatasetMatrix[0] * vec4(in_vertexPos, 1.0)
  //   ip_textureCoords = (in_cellToPoint[0] * vec4(uvx, 1.0)).xyz
  // and the rasterizer interpolates that per-vertex float result. Replicate the
  // exact GLSL mat4*vec4 contraction (mul, fma, fma, mul+add) so the per-vertex
  // texcoord — hence the interpolated ray anchor — is bit-identical.
  float3 uvx;
  float au;
  au = volumeUniforms.volumeToTexture[0][0] * v.x;
  au = fma(volumeUniforms.volumeToTexture[1][0], v.y, au);
  au = fma(volumeUniforms.volumeToTexture[2][0], v.z, au);
  au = au + volumeUniforms.volumeToTexture[3][0] * v.w;
  uvx.x = au;
  au = volumeUniforms.volumeToTexture[0][1] * v.x;
  au = fma(volumeUniforms.volumeToTexture[1][1], v.y, au);
  au = fma(volumeUniforms.volumeToTexture[2][1], v.z, au);
  au = au + volumeUniforms.volumeToTexture[3][1] * v.w;
  uvx.y = au;
  au = volumeUniforms.volumeToTexture[0][2] * v.x;
  au = fma(volumeUniforms.volumeToTexture[1][2], v.y, au);
  au = fma(volumeUniforms.volumeToTexture[2][2], v.z, au);
  au = au + volumeUniforms.volumeToTexture[3][2] * v.w;
  uvx.z = au;
  out.texcoord = volumeUniforms.cellToPointScale * uvx + volumeUniforms.cellToPointOffset;
  // Debug only (test builds): the proxy geometry has only a handful of
  // vertices, so this stays bounded to a few messages per frame. Verified by
  // TestMetalVolumeShaderLog.
#if defined(VTK_METAL_ENABLE_LOGGING)
  os_log_default.log_info("VTK_METAL_VOLUME_LOG vertex_volume_main vid=%u modelPos=(%.9g, %.9g, %.9g) clip=(%.9g, %.9g, %.9g, %.9g) uvx=(%.9g, %.9g, %.9g) texcoord=(%.9g, %.9g, %.9g)",
    vertexId, modelPos.x, modelPos.y, modelPos.z, out.position.x, out.position.y, out.position.z, out.position.w, uvx.x, uvx.y, uvx.z, out.texcoord.x, out.texcoord.y, out.texcoord.z);
#endif
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
  return u.sampleDistance * maxBound / max(physPerNorm, 1e-6);
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
inline float sampleVolumeScalar(texture3d<float> volTex, float3 pos) {
  if (fc_linearInterpolation) {
    return volTex.sample(sVolume, pos, level(0)).r;
  }
  return volTex.sample(sNearest, pos, level(0)).r;
}

// Full-texel fetch honoring the pipeline's interpolation specialization. Used
// by the independent multi-component path, which needs every channel (one per
// component) at the same sample position.
inline float4 sampleVolumeTexel(texture3d<float> volTex, float3 pos) {
  if (fc_linearInterpolation) {
    return volTex.sample(sVolume, pos, level(0));
  }
  return volTex.sample(sNearest, pos, level(0));
}

inline float4 sampleTransferFunction(texture2d<float> tfTex, float2 uv) {
  if (fc_linearInterpolation) {
    return tfTex.sample(sVolume, uv, level(0));
  }
  return tfTex.sample(sNearest, uv, level(0));
}

// Per-component transfer-function lookup for the independent multi-component
// path (OpenGL computeColor/computeOpacity parity): selects the table of the
// requested component. All four tables are always uploaded for this path.
inline float4 sampleComponentTransferFunction(
    texture2d<float> tf0, texture2d<float> tf1,
    texture2d<float> tf2, texture2d<float> tf3,
    float2 uv, int c) {
  if (c == 0) return sampleTransferFunction(tf0, uv);
  if (c == 1) return sampleTransferFunction(tf1, uv);
  if (c == 2) return sampleTransferFunction(tf2, uv);
  return sampleTransferFunction(tf3, uv);
}

// 2D transfer function lookup at (primaryScalarNorm, secondScalarNorm).
inline float4 sampleTransferFunction2D(texture2d<float> tf2DTex, float2 uv) {
  if (fc_linearInterpolation) {
    return tf2DTex.sample(sVolume, uv, level(0));
  }
  return tf2DTex.sample(sNearest, uv, level(0));
}

// Fetch the Y-axis scalar array (e.g. "Temp") at the same normalized volume
// coordinate as the primary volume texture.
inline float sampleSecondScalar(texture3d<float> yAxisTex, float3 pos) {
  if (fc_linearInterpolation) {
    return yAxisTex.sample(sVolume, pos, level(0)).r;
  }
  return yAxisTex.sample(sNearest, pos, level(0)).r;
}

inline float sampleGradientOpacity(texture2d<float> gradTex, float value) {
  if (fc_linearInterpolation) {
    return gradTex.sample(sVolume, float2(value, 0.5), level(0)).r;
  }
  return gradTex.sample(sNearest, float2(value, 0.5), level(0)).r;
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
// overflow (mag > 65504) for full 3D gradients, and to match the OpenGL
// backend's all-float computeGradient: half rounding of the six neighbor
// samples and of grad.w was amplified by steep gradient-opacity ramps.
inline float4 normalizedGradient(float3 gradTex, float4x4 volumeToTexture, float gradNormFactor) {
  float3x3 texToModelLin =
    float3x3(volumeToTexture[0].xyz, volumeToTexture[1].xyz, volumeToTexture[2].xyz);
  float3 correctedGrad = transpose(texToModelLin) * gradTex;
  float mag = length(correctedGrad);
  float3 normal = mag > 0.0f ? correctedGrad / mag : float3(0.0f);
  return float4(normal, saturate(mag / gradNormFactor));
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
inline float4 densityGradientFromNeighbors(
    float sPX, float sNX, float sPY, float sNY, float sPZ, float sNZ,
    texture2d<float> tf0, texture2d<float> tf1,
    texture2d<float> tf2, texture2d<float> tf3,
    float3 gradStep, float4x4 volumeToTexture,
    float gradNormFactor, int c, float scalarScale, float scalarBias) {
  float opPX = float(sampleComponentTransferFunction(tf0, tf1, tf2, tf3,
      float2(saturate(sPX * scalarScale + scalarBias), 0.5), c).a);
  float opNX = float(sampleComponentTransferFunction(tf0, tf1, tf2, tf3,
      float2(saturate(sNX * scalarScale + scalarBias), 0.5), c).a);
  float opPY = float(sampleComponentTransferFunction(tf0, tf1, tf2, tf3,
      float2(saturate(sPY * scalarScale + scalarBias), 0.5), c).a);
  float opNY = float(sampleComponentTransferFunction(tf0, tf1, tf2, tf3,
      float2(saturate(sNY * scalarScale + scalarBias), 0.5), c).a);
  float opPZ = float(sampleComponentTransferFunction(tf0, tf1, tf2, tf3,
      float2(saturate(sPZ * scalarScale + scalarBias), 0.5), c).a);
  float opNZ = float(sampleComponentTransferFunction(tf0, tf1, tf2, tf3,
      float2(saturate(sNZ * scalarScale + scalarBias), 0.5), c).a);

  float3 opGrad = float3(opPX - opNX, opPY - opNY, opPZ - opNZ);
  float3 gradTex = opGrad / max(gradStep, 1e-8);
  return normalizedGradient(gradTex, volumeToTexture, gradNormFactor);
}

inline float4 computeGradientFast(texture3d<float> volTex, float3 pos,
                                 float3 gradStep, float4x4 volumeToTexture, float gradNormFactor) {
  float sPX = sampleVolumeScalar(volTex, pos + float3(gradStep.x, 0, 0));
  float sNX = sampleVolumeScalar(volTex, pos - float3(gradStep.x, 0, 0));
  float sPY = sampleVolumeScalar(volTex, pos + float3(0, gradStep.y, 0));
  float sNY = sampleVolumeScalar(volTex, pos - float3(0, gradStep.y, 0));
  float sPZ = sampleVolumeScalar(volTex, pos + float3(0, 0, gradStep.z));
  float sNZ = sampleVolumeScalar(volTex, pos - float3(0, 0, gradStep.z));

  float3 rawGrad = float3(sPX - sNX, sPY - sNY, sPZ - sNZ);
  float3 gradTex = rawGrad / max(gradStep, 1e-8);
  return normalizedGradient(gradTex, volumeToTexture, gradNormFactor);
}

inline float4 computeDensityGradientFast(
    texture3d<float> volTex,
    texture2d<float> tf0, texture2d<float> tf1,
    texture2d<float> tf2, texture2d<float> tf3,
    float3 pos, float3 gradStep,
    float4x4 volumeToTexture,
    float gradNormFactor,
    int c, float scalarScale, float scalarBias) {
  float sPX = sampleVolumeScalar(volTex, pos + float3(gradStep.x, 0, 0));
  float sNX = sampleVolumeScalar(volTex, pos - float3(gradStep.x, 0, 0));
  float sPY = sampleVolumeScalar(volTex, pos + float3(0, gradStep.y, 0));
  float sNY = sampleVolumeScalar(volTex, pos - float3(0, gradStep.y, 0));
  float sPZ = sampleVolumeScalar(volTex, pos + float3(0, 0, gradStep.z));
  float sNZ = sampleVolumeScalar(volTex, pos - float3(0, 0, gradStep.z));
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
inline float4 computeScalarAndDensityGradient(
    texture3d<float> volTex,
    texture2d<float> tf0, texture2d<float> tf1,
    texture2d<float> tf2, texture2d<float> tf3,
    float3 pos, float3 gradStep,
    float4x4 volumeToTexture,
    float gradNormFactor,
    float scalarScale, float scalarBias,
    thread float4& densityGradOut) {
  float sPX = sampleVolumeScalar(volTex, pos + float3(gradStep.x, 0, 0));
  float sNX = sampleVolumeScalar(volTex, pos - float3(gradStep.x, 0, 0));
  float sPY = sampleVolumeScalar(volTex, pos + float3(0, gradStep.y, 0));
  float sNY = sampleVolumeScalar(volTex, pos - float3(0, gradStep.y, 0));
  float sPZ = sampleVolumeScalar(volTex, pos + float3(0, 0, gradStep.z));
  float sNZ = sampleVolumeScalar(volTex, pos - float3(0, 0, gradStep.z));

  densityGradOut = densityGradientFromNeighbors(sPX, sNX, sPY, sNY, sPZ, sNZ,
      tf0, tf1, tf2, tf3, gradStep, volumeToTexture, gradNormFactor, 0, scalarScale, scalarBias);

  float3 rawGrad = float3(sPX - sNX, sPY - sNY, sPZ - sNZ);
  float3 gradTex = rawGrad / max(gradStep, 1e-8);
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
    float4x4 volumeToTexture, float gradNormFactor, thread float4 gradOut[4]) {
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
    float3 normal = mag > 0.0f ? corrected / mag : float3(0.0f);
    gradOut[c] = float4(normal, saturate(mag / gradNormFactor));
  }
}

// Mirrors OpenGL's ComputeLightingDeclaration default-light path (headlight):
//   nDotL = dot(normal, -g_ldir)     with g_ldir = normalize(cameraPos - vertexPos)
//   r     = normalize(2*nDotL*normal + g_ldir)
//   vDotR = dot(r, -g_vdir)
//   specular = pow(vDotR, shininess) * in_specular * in_lightSpecularColor
inline float3 computePhongLightingVolumeFast(float3 sampleColor, float3 normal, float3 lightDir, float3 viewDir,
                                             float3 ambientMat, float3 diffuseMat, float3 specularMat, float shininess,
                                             bool twoSided = false) {
  float nDotL = dot(normal, -lightDir);
  // Reflection vector uses the un-negated nDotL (matches OpenGL's
  // ComputeLightingDeclaration, which computes r before the two-sided flip).
  // Guard on (nDotL > 0 || twoSided): for back-facing non-twoSided samples the
  // result is discarded, so skip the normalize/dot entirely.
  if (nDotL > 0.0f || twoSided) {
    float3 r = normalize(normal * (2.0f * nDotL) + lightDir);
    float vDotR = max(dot(r, -viewDir), 0.0f);
    if (nDotL < 0.0f && twoSided) {
      nDotL = -nDotL;
    }
    if (nDotL > 0.0f) {
      float3 diffuse = nDotL * diffuseMat * sampleColor;
      float3 specular = fast::pow(vDotR, shininess) * specularMat;
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
inline float3 computeVolumeLighting(
    float3 sampleColor,
    float3 normal,
    float3 viewDir,           // normalized, pointing toward camera (data space)
    float3 ambientMat,
    float3 diffuseMat,
    float3 specularMat,
    float shininess,
    constant VolumeLightUniforms& lightUniforms,
    float3 fragPosVolume)    // current sample position in data space
{
    float3 totalAmbient  = float3(0.0f);
    float3 totalDiffuse  = float3(0.0f);
    float3 totalSpecular = float3(0.0f);

    int numLights = lightUniforms.lightCount;
    bool twoSided = lightUniforms.twoSidedLighting != 0;

    for (int i = 0; i < numLights && i < MAX_LIGHTS; ++i) {
        constant VolumeLight& L = lightUniforms.lights[i];

        float3 lightAmbient  = float3(L.ambientColor.rgb);
        float3 lightDiffuse  = float3(L.diffuseColor.rgb);
        float3 lightSpecular = float3(L.specularColor.rgb);

        float3 toLight;
        float attenuation = 1.0f;

        if (L.position.w < 0.5) {
            // Directional light: direction is pre-normalized in data space and
            // points along the light's travel direction (OpenGL
            // in_lightDirection parity).
            toLight = float3(L.direction.xyz);
        } else {
            // Positional light: compute the direction the light travels
            // (light -> fragment, matching OpenGL's
            // normalize(fragWorldPos - lightPosition)).
            float3 lightPos = float3(L.position.xyz);
            float3 delta = float3(fragPosVolume) - lightPos;
            float dist = length(delta);
            toLight = dist > 0.0001f ? delta / dist : float3(0.0f, 0.0f, 1.0f);

            // Attenuation: 1 / (constant + linear*d + quadratic*d^2)
            float attenDenom = L.attenuation.x
                             + L.attenuation.y * dist
                             + L.attenuation.z * dist * dist;
            attenuation = attenDenom > 0.0f ? 1.0f / attenDenom : 0.0f;

            // Spot light cone check: the cone axis is L.direction (the light's
            // travel direction); a fragment is inside the cone when the
            // light->fragment direction is within the cone angle of the axis
            // (OpenGL coneDot = dot(vertLightDirection, lightDir)).
            if (L.direction.w <= 90.0) {
                float spotCos = dot(toLight, float3(normalize(L.direction.xyz)));
                float spotCutoff = cos(L.direction.w * (M_PI_F / 180.0));
                if (spotCos < spotCutoff) {
                    attenuation = 0.0f;
                } else {
                    attenuation *= fast::pow(spotCos, L.attenuation.w);
                }
            }
        }

        // Diffuse
        float nDotL = dot(normal, toLight);
        if (nDotL < 0.0f && twoSided) {
            nDotL = -nDotL;
        }
        if (nDotL > 0.0f) {
            totalDiffuse += nDotL * lightDiffuse * attenuation;

            // Phong reflection vector (matches OpenGL's ComputeLightingDeclaration
            // and computePhongLightingVolumeFast)
            float3 r = normalize(normal * (2.0f * nDotL) - toLight);
            float vDotR = dot(-viewDir, r);
            if (vDotR < 0.0f && twoSided) {
                vDotR = -vDotR;
            }
            if (vDotR > 0.0f) {
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
    // OpenGL divides a vec by its w as rcp+mul (raw * (1.0/w), GLSL codegen),
    // NOT IEEE division; replicate the rcp+mul rounding so the unprojected
    // points are bit-identical (findings update 71 sect2.4 / update 74).
    float4 wn = u.ndcToVolume * float4(ndc.x, -ndc.y, 0.0, 1.0); wn.xyz *= (1.0f / wn.w);
    float4 wf = u.ndcToVolume * float4(ndc.x, -ndc.y, 1.0, 1.0); wf.xyz *= (1.0f / wf.w);
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
    if (volumeUniforms.useCameraInsideNearClip > 0.5) {
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
    if (volumeUniforms.useDepthTexture > 0.5) {
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
    float2 screenPos;  // in-viewport pixel coords; (-1,-1) when unknown
    float3 localPos;   // interpolated fragment position in [0,1] volume space
    float4 clipPos;    // debug: interpolated clip-space position (perspective-correct)
    bool   checkBounds;
    float3 anchorData; // interpolated anchor in dataset space (GL ip_vertexPos parity)
    bool   anchorIsData; // anchorData holds a data-space position (camera-inside proxy path)
    uint   flatVid;     // debug: provoking-vertex index of the covering triangle
    uint   primId;      // debug: primitive (triangle) index of the covering triangle
};

// Debug helper (test builds): per-sample march dumps, gated to a handful of
// pixels on the final CameraInside frame. Used to verify the ray geometry and
// scalar normalization against ground truth offline.
inline bool debugMarchGate(float3 camera, float2 screenPos) {
  float3 dc = camera;
  bool camOk = all(abs(dc - float3(0.678174, 0.486826, 0.964175)) < 1e-3);
  // screenPos is the fragment's pixel-center coordinate, i.e. (px + 0.5, py + 0.5).
  bool pxOk =
      all(abs(screenPos - float2(46.5, 1.5)) < 0.5) ||
      all(abs(screenPos - float2(17.5, 1.5)) < 0.5) ||
      all(abs(screenPos - float2(50.5, 15.5)) < 0.5) ||
      all(abs(screenPos - float2(150.5, 150.5)) < 0.5);
  // TEMP DEBUG: ClippingUserTransform proxy path (parallel, camera outside).
  bool camOkClip = all(abs(dc - float3(0.5, 3.141854, 0.5)) < 1e-3);
  bool pxOkClip = all(abs(screenPos - float2(250.5, 250.5)) < 0.5);
  // TEMP DEBUG: CameraInsideTransformation (camera inside rotated volume).
  bool pxOkAny = all(abs(screenPos - float2(256.5, 256.5)) < 0.5);
  // TEMP DEBUG: NoShadeNoGradOpNoTransform camera-inside divergent pixels.
  bool pxOkContained =
      all(abs(screenPos - float2(372.5, 131.5)) < 0.5) ||
      all(abs(screenPos - float2(422.5, 92.5)) < 0.5);
  // TEMP DEBUG: always dump the GL-matched pixel (422, 419) regardless of camera.
  //   Metal screenPos (top-left) == GL glReadPixels (422, 92). This is the worst
  //   |Metal-GL| pixel in the NoJitter render (d=199: GL dark, Metal bright).
  bool pxOkAlways = all(abs(screenPos - float2(422.5, 419.5)) < 0.5);
  // TEMP DEBUG: u60 knife-edge pixels (largest Metal-vs-GL gf deltas).
  bool pxOkKnife =
      all(abs(screenPos - float2(397.5, 110.5)) < 0.5) ||
      all(abs(screenPos - float2(360.5, 229.5)) < 0.5) ||
      all(abs(screenPos - float2(349.5, 255.5)) < 0.5) ||
      all(abs(screenPos - float2(405.5, 171.5)) < 0.5) ||
      all(abs(screenPos - float2(9.5, 18.5)) < 0.5) ||
      all(abs(screenPos - float2(293.5, 298.5)) < 0.5) ||
      all(abs(screenPos - float2(338.5, 432.5)) < 0.5) ||
      all(abs(screenPos - float2(350.5, 5.5)) < 0.5) ||
      all(abs(screenPos - float2(153.5, 32.5)) < 0.5) ||
      all(abs(screenPos - float2(482.5, 33.5)) < 0.5) ||
      all(abs(screenPos - float2(120.5, 167.5)) < 0.5) ||
      all(abs(screenPos - float2(470.5, 269.5)) < 0.5) ||
      all(abs(screenPos - float2(439.5, 281.5)) < 0.5) ||
      all(abs(screenPos - float2(469.5, 463.5)) < 0.5);
  // TEMP DEBUG: NoShade left-half comparison pixels (left side matches GL).
  bool pxOkLeft =
      all(abs(screenPos - float2(80.5, 400.5)) < 0.5) ||
      all(abs(screenPos - float2(150.5, 250.5)) < 0.5);
  // TEMP DEBUG: CamOutsideFixedStep ring + border pixels.
  bool pxOkCamOut =
      all(abs(screenPos - float2(45.5, 113.5)) < 0.5) ||   // ring, d=46
      all(abs(screenPos - float2(32.5, 346.5)) < 0.5) ||   // ring, max|d|=34
      all(abs(screenPos - float2(0.5, 256.5)) < 0.5) ||    // left border
      all(abs(screenPos - float2(511.5, 256.5)) < 0.5) ||  // right border
      all(abs(screenPos - float2(256.5, 2.5)) < 0.5) ||    // top border
      all(abs(screenPos - float2(256.5, 510.5)) < 0.5) ||  // bottom
      all(abs(screenPos - float2(480.5, 508.5)) < 0.5) ||  // high-delta skin surface
      all(abs(screenPos - float2(201.5, 13.5)) < 0.5);     // high-delta skin surface
  // TEMP DEBUG: post-option-A residual pixels (|d|>=5 at VTK_FIXED_SAMPLE_DISTANCE=0.5).
  bool pxOkResid =
      all(abs(screenPos - float2(182.5, 43.5)) < 0.5) ||
      all(abs(screenPos - float2(183.5, 43.5)) < 0.5) ||
      all(abs(screenPos - float2(219.5, 3.5)) < 0.5) ||
      all(abs(screenPos - float2(283.5, 7.5)) < 0.5) ||
      all(abs(screenPos - float2(279.5, 11.5)) < 0.5) ||
      all(abs(screenPos - float2(311.5, 16.5)) < 0.5) ||
      all(abs(screenPos - float2(311.5, 17.5)) < 0.5) ||
      all(abs(screenPos - float2(319.5, 25.5)) < 0.5) ||
      all(abs(screenPos - float2(319.5, 26.5)) < 0.5) ||
      all(abs(screenPos - float2(208.5, 83.5)) < 0.5) ||
      all(abs(screenPos - float2(327.5, 112.5)) < 0.5) ||
      all(abs(screenPos - float2(195.5, 133.5)) < 0.5) ||
      all(abs(screenPos - float2(357.5, 154.5)) < 0.5) ||
      all(abs(screenPos - float2(322.5, 172.5)) < 0.5) ||
      all(abs(screenPos - float2(372.5, 175.5)) < 0.5) ||
      all(abs(screenPos - float2(104.5, 245.5)) < 0.5) ||
      all(abs(screenPos - float2(265.5, 246.5)) < 0.5) ||
      all(abs(screenPos - float2(382.5, 207.5)) < 0.5) ||
      all(abs(screenPos - float2(188.5, 307.5)) < 0.5) ||
      all(abs(screenPos - float2(197.5, 280.5)) < 0.5) ||
      all(abs(screenPos - float2(249.5, 314.5)) < 0.5) ||
      all(abs(screenPos - float2(250.5, 314.5)) < 0.5) ||
      all(abs(screenPos - float2(8.5, 324.5)) < 0.5) ||
      all(abs(screenPos - float2(222.5, 326.5)) < 0.5) ||
      all(abs(screenPos - float2(210.5, 375.5)) < 0.5) ||
      all(abs(screenPos - float2(262.5, 356.5)) < 0.5) ||
      all(abs(screenPos - float2(501.5, 332.5)) < 0.5) ||
      all(abs(screenPos - float2(190.5, 437.5)) < 0.5) ||
      all(abs(screenPos - float2(296.5, 438.5)) < 0.5) ||
      all(abs(screenPos - float2(482.5, 398.5)) < 0.5) ||
      all(abs(screenPos - float2(38.5, 448.5)) < 0.5) ||
      all(abs(screenPos - float2(6.5, 451.5)) < 0.5) ||
      all(abs(screenPos - float2(34.5, 485.5)) < 0.5) ||
      all(abs(screenPos - float2(14.5, 493.5)) < 0.5) ||
      all(abs(screenPos - float2(14.5, 494.5)) < 0.5) ||
      all(abs(screenPos - float2(373.5, 466.5)) < 0.5) ||
      all(abs(screenPos - float2(466.5, 451.5)) < 0.5) ||
      all(abs(screenPos - float2(482.5, 469.5)) < 0.5) ||
      all(abs(screenPos - float2(482.5, 470.5)) < 0.5) ||
      all(abs(screenPos - float2(482.5, 471.5)) < 0.5) ||
      all(abs(screenPos - float2(482.5, 472.5)) < 0.5) ||
      all(abs(screenPos - float2(482.5, 473.5)) < 0.5) ||
      all(abs(screenPos - float2(484.5, 479.5)) < 0.5) ||
      all(abs(screenPos - float2(490.5, 484.5)) < 0.5) ||
      all(abs(screenPos - float2(491.5, 485.5)) < 0.5) ||
      all(abs(screenPos - float2(495.5, 497.5)) < 0.5) ||
      all(abs(screenPos - float2(480.5, 511.5)) < 0.5);
  // TEMP DEBUG: CamOutsideNoJitter max-delta residual pixels.
  bool pxOkNoJitter =
      all(abs(screenPos - float2(307.5, 8.5)) < 0.5) ||
      all(abs(screenPos - float2(307.5, 7.5)) < 0.5) ||
      all(abs(screenPos - float2(307.5, 9.5)) < 0.5) ||
      all(abs(screenPos - float2(480.5, 400.5)) < 0.5) ||
      all(abs(screenPos - float2(496.5, 488.5)) < 0.5) ||
      all(abs(screenPos - float2(93.5, 201.5)) < 0.5) ||
      all(abs(screenPos - float2(242.5, 330.5)) < 0.5);
  // TEMP DEBUG: update-69 B (constant-scalar) remaining 18 residual pixels.
  bool pxOkResid69 =
      all(abs(screenPos - float2(140.5, 6.5)) < 0.5) ||
      all(abs(screenPos - float2(170.5, 42.5)) < 0.5) ||
      all(abs(screenPos - float2(181.5, 96.5)) < 0.5) ||
      all(abs(screenPos - float2(18.5, 163.5)) < 0.5) ||
      all(abs(screenPos - float2(312.5, 183.5)) < 0.5) ||
      all(abs(screenPos - float2(366.5, 262.5)) < 0.5) ||
      all(abs(screenPos - float2(249.5, 317.5)) < 0.5) ||
      all(abs(screenPos - float2(305.5, 335.5)) < 0.5) ||
      all(abs(screenPos - float2(268.5, 364.5)) < 0.5) ||
      all(abs(screenPos - float2(0.5, 375.5)) < 0.5) ||
      all(abs(screenPos - float2(197.5, 401.5)) < 0.5) ||
      all(abs(screenPos - float2(11.5, 419.5)) < 0.5) ||
      all(abs(screenPos - float2(70.5, 424.5)) < 0.5) ||
      all(abs(screenPos - float2(71.5, 424.5)) < 0.5) ||
      all(abs(screenPos - float2(74.5, 424.5)) < 0.5) ||
      all(abs(screenPos - float2(75.5, 424.5)) < 0.5) ||
      all(abs(screenPos - float2(229.5, 425.5)) < 0.5) ||
      all(abs(screenPos - float2(174.5, 445.5)) < 0.5) ||
      all(abs(screenPos - float2(435.5, 480.5)) < 0.5);
  return (camOk && pxOk) || (camOkClip && pxOkClip) || pxOkAny || pxOkContained || pxOkLeft ||
         pxOkCamOut || pxOkResid || pxOkNoJitter || pxOkAlways || pxOkKnife || pxOkResid69;
}

// TEMP DEBUG: per-fragment-only gate for the MARCH/STEP dumps (NOT the
// per-sample SAMPLE/GRADOP/LIGHT dumps). Adds a sparse full-frame grid (every
// 8th pixel, 64x64 = 4096 px) plus a dense 64x64 block around the
// CameraInsideNoTransform knife edge (349,255) to debugMarchGate's pixels, so
// the dense displacement-field experiment can back out the effective sample
// NDC per pixel without exploding the per-sample log volume. The grid/dense
// region applies only to the CameraInsideTransformation camera; other cameras
// keep debugMarchGate's pixels only.
inline bool analyticPixelGate(float2 screenPos);  // defined with the analytic helpers

inline bool debugStepGate(float3 camera, float2 screenPos) {
  float3 dc = camera;
  bool camOk = all(abs(dc - float3(0.506559, 0.506559, 0.446101)) < 1e-3);
  if (camOk)
  {
    int px = int(floor(screenPos.x - 0.5));
    int py = int(floor(screenPos.y - 0.5));
    bool pxOkGrid = (px % 8 == 0) && (py % 8 == 0);
    bool pxOkDense = px >= 317 && px <= 380 && py >= 223 && py <= 286;
    // TEMP DEBUG: GL VTK_GL_RAY_DUMP gate pixels (matched pair experiment).
    bool pxOkGlRay = false;
    int glRayPx[][2] = {
      {140,505},{170,469},{181,415},{18,348},{312,328},{366,249},{249,194},
      {305,176},{268,147},{0,136},{197,110},{11,92},{70,87},{71,87},{74,87},
      {75,87},{229,86},{174,66},{435,31}
    };
    for (int g = 0; g < 19; ++g)
    {
      if (px == glRayPx[g][0] && py == glRayPx[g][1]) { pxOkGlRay = true; break; }
    }
    if (pxOkGrid || pxOkDense || pxOkGlRay)
    {
      return true;
    }
  }
  return debugMarchGate(camera, screenPos);
}

inline float4 marchVolumeUnified(
    MarchParams p,
    float3 initialColor, float initialOpacity,
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
    texture3d<float> normalTexture,
    texture3d<float> blankingTexture,
    constant packed_float3* rectCoords,
    constant VolumeLightUniforms* lightUniforms,
    thread float3* firstOpaquePos,
    thread bool*  haveOpaquePos)
{
  const bool doShading = fc_shading && (volumeUniforms.useGradientShading > 0.5);
  const bool doGradOp = fc_gradientOpacity && (volumeUniforms.useGradientOpacity > 0.5);
  const bool doCropping = volumeUniforms.useCropping > 0.5;
  const bool doMask = fc_mask && (volumeUniforms.useMask > 0.5);
  const bool doTransfer2D = volumeUniforms.useTransfer2D > 0.5;
  const bool doBlanking = volumeUniforms.useBlanking > 0.5;
  const bool doRectilinear = volumeUniforms.useRectilinear > 0.5;

  // Scalar window/level normalization in full float32 (OpenGL parity: the GL
  // shader applies scalar = raw * in_volume_scale + in_volume_bias in float
  // after sampling; the half-precision version here quantized the norm to half
  // ulp (~1e-4) which shifted the opacity/color ramp lookups).
  float scalarScale = 1.0f / max((volumeUniforms.scalarMax - volumeUniforms.scalarMin), 1e-4f);
  float scalarBias  = -volumeUniforms.scalarMin * scalarScale;

  float secondScale = volumeUniforms.transfer2DYAxisScale;
  float secondBias  = volumeUniforms.transfer2DYAxisBias;

  float gradNormFactor = max(1e-8f, volumeUniforms.gradientOpacityRange.y);

  float3 boundsSize = max(volumeUniforms.volumeBoundsMax.xyz
                        - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
  // Advance of the sample position in texture space per ray step (constant),
  // including the image-data direction matrix: the ray direction/step are in
  // normalized volume space and are converted to [0,1] texture coords via
  // volumeToTexture (OpenGL TextureToDataset parity).
  float3 rayDirTexLocal = (volumeUniforms.volumeToTexture * float4(p.rayDir * boundsSize, 0.0)).xyz;
  float3 texStep = rayDirTexLocal * p.stepSize;
  // Cell-to-point conversion factors, computed once (texel centers at (i+0.5)/dims).
  // OpenGL CellToPointMatrix parity (vtkVolumeTexture::ComputeCellToPointMatrix):
  // scale = (d-0.5)/d - 0.5/d, offset = 0.5/d, both float32 divisions -- NOT
  // (d-1)/d, which can differ from GL's two-stage expression in the last ulp.
  float3 texelCount = float3(volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  float3 ctpScale   = max((texelCount - 0.5) / texelCount - 0.5 / texelCount, 1e-4);
  float3 ctpOffset  = 0.5 / texelCount;
  // Advance evalPoint in GL's cell-to-point-adjusted texture space with GL's
  // g_dirStep arithmetic: fold the cell-to-point scale into the dataset->texture
  // matrix (GL's ip_inverseTextureDataAdjusted = in_cellToPoint *
  // in_inverseTextureDatasetMatrix; since cellToPoint is diagonal, the linear 3x3
  // of that product is the rows of volumeToTexture scaled by ctpScale), then one
  // mat-vec times the dataset-space direction and one scalar multiply by the step.
  // This removes the per-sample ctpScale re-multiply that accumulated rounding.
  float3x3 adjustedLin = float3x3(
      volumeUniforms.volumeToTexture[0].xyz * ctpScale,
      volumeUniforms.volumeToTexture[1].xyz * ctpScale,
      volumeUniforms.volumeToTexture[2].xyz * ctpScale);
  // OpenGL g_dirStep parity (bit-exact step): GL computes
  //   g_dirStep = (ip_inverseTextureDataAdjusted * normalize(vertexPos - eyePos)).xyz
  //               * in_sampleDistance
  // with the normalize done in DATASET/OBJECT space (computeRayDirection =
  // normalize(ip_vertexPos.xyz - in_eyePosObjs[0].xyz)). The previous form
  // normalized in volume space and folded the sample distance across the
  // SampleDistance/maxBoundsSize uniform and physicalSampleStep, which drifted
  // ~1 ulp/step against GL. For the camera-inside proxy path the interpolated
  // anchor is now carried in dataset space (anchorData, GL ip_vertexPos parity),
  // so replicate GL's float32 chain exactly: normalize (anchorData - eyeData)
  // once, then one mat-vec and one scalar multiply by the world-unit sample
  // distance. The legacy paths (camera-outside box, fullscreen, grid traversal)
  // keep the volume-space direction converted through boundsSize.
  float3 dirObj;
  float4 dbgNearP = float4(0.0, 0.0, 0.0, 1.0);
  float4 dbgFarP = float4(0.0, 0.0, 0.0, 1.0);
  float3 d = float3(0.0, 0.0, 0.0);
  if (p.anchorIsData && volumeUniforms.useParallelProjection < 0.5)
  {
    // OpenGL computeRayDirection parity (analytic pixel ray): unproject the
    // fragment through the near/far planes with the CPU-composed inversePVM
    // (inverseVolume * inverseModelView * inverseProjection, GL in_inversePVM
    // bytes) and normalize the difference in dataset space. This removes the
    // interpolated-anchor dependence (normalize(anchorData - eyePosData)) that
    // differed between the backends by ~2.2e-5 on the anchor and accumulated
    // ~3e-8/step in evalStep. GLSL mat4*vec4 contracts [0,1,2,3]
    // mul,fma,fma,mul+add, so use vecMulStrict. p.screenPos is Metal's top-left
    // window pixel center; negating ndc.y converts it to GL's bottom-up
    // convention (gl_FragCoord.y - in_windowLowerLeftCorner). z = -1/+1 are
    // GL's ndc near/far, matching the GL-nearz-convention inversePVM.
    // (viewportSize == window size and windowLowerLeftCorner == (0,0) here, so
    // (screenPos/viewportSize)*2-1 == (fragCoord - ll)*2*inverseWindowSize - 1.)
    float2 ndc = (p.screenPos / volumeUniforms.viewportSize) * 2.0 - 1.0;
    float4 nearP = vecMulStrict(volumeUniforms.inversePVM, float4(ndc.x, -ndc.y, -1.0, 1.0));
    // OpenGL divides by w as rcp+mul (raw * (1.0/w), GLSL codegen), not IEEE
    // division; GL's logged nearP/farP == raw*rcp bit-for-bit, and division
    // differs by 1 ulp in the last digits (findings update 74). Replicate.
    nearP *= (1.0f / nearP.w);
    float4 farP = vecMulStrict(volumeUniforms.inversePVM, float4(ndc.x, -ndc.y, 1.0, 1.0));
    farP *= (1.0f / farP.w);
    dbgNearP = nearP;
    dbgFarP = farP;
    d = farP.xyz - nearP.xyz;
    // GLSL normalize compiles to dir * inversesqrt(dot(dir,dir)) where
    // inversesqrt is the GPU's approximate reciprocal-sqrt instruction, 1 ulp
    // above the correctly-rounded metal::rsqrt (findings update 74: GL inv=
    // 3ba6b788 vs IEEE 3ba6b787 at d2=4716e769). Metal's normalize() uses the
    // correctly-rounded rsqrt, so replicate GL's approximate instruction.
    dirObj = d * fast::rsqrt(dot(d, d));
  }
  else
  {
    dirObj = normalize(p.rayDir * boundsSize);
  }
  float3 evalStep = (adjustedLin * dirObj) * volumeUniforms.sampleDistanceWorld;

  // Lighting directions must live in the same (physical/data) space as the
  // gradient normal: the normal is expressed per world-unit (the gradient is
  // scaled by the direction/spacing transform), so the headlight direction must
  // be converted from the normalized volume frame back to data space (offset *
  // boundsSize) before computing nDotL/vDotR. OpenGL computes g_ldir/g_vdir
  // directly in object space (normalize(eyePosObj - vertexPosObj)); using the
  // distorted volume frame here would bias nDotL for anisotropic bounds and fire
  // specular on surfaces where OpenGL's nDotL is <= 0.
  float3 entryVolPos = p.rayOrigin + p.rayDir * p.tStart;
  float3 viewDirHalf  = normalize((entryVolPos - volumeUniforms.cameraVolumePos.xyz) * boundsSize);
  float3 lightDirHalf = normalize(volumeUniforms.lightDirection * boundsSize);
  float3 ambientMat   = float3(volumeUniforms.ambientColor.rgb);
  float3 diffuseMat   = float3(volumeUniforms.diffuseColor.rgb);
  float3 specularMat  = float3(volumeUniforms.specularColor.rgb);
  float shininessMat  = float(volumeUniforms.shininess);

  float maskScale = volumeUniforms.maskScale;
  float maskBias  = volumeUniforms.maskBias;
  float numLabels = volumeUniforms.labelMapNumLabels;

  // Independent multi-component path (OpenGL independent components parity):
  // each component is normalized against its own scalar range and looked up in
  // its own color/opacity table, then results are combined via component
  // weights. The 2D transfer-function and label-map modes always use the
  // single-component path.
  const bool useIndependentPath = (volumeUniforms.useIndependentComponents > 0.5) &&
    !doTransfer2D && !(doMask && numLabels > 0.0);

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
  // Integer step counter (OpenGL g_currentT parity: ++g_currentT per sample).
  // The ray distance is recovered as firstT + currentT * p.stepSize.
  int currentT = 0;

  // Step capacity uses the OpenGL g_dirStep length (evalStep), not the texStep
  // length: GL's g_terminatePointMax = length(g_terminatePos - g_rayOrigin) /
  // length(g_dirStep), and evalStep replicates g_dirStep bit-exactly while
  // texStep is ~0.2% longer (constant-physical-distance step). Basing the cap
  // on the texStep length lets the loop run out of headroom one sample earlier
  // than GL on rays whose bounds exit fires last (findings update 68).
  int maxSteps = max(1, int(ceil((p.tEnd - firstT) / length(evalStep))));

  // OpenGL composites in full float (g_fragColor is a vec4; the front-to-back
  // accumulator and weights are float). Accumulating in half caps the opacity
  // once per-sample increments fall below the half mantissa ulp (~0.00049 at
  // accA~0.8), so the interior of a volume renders systematically dimmer than
  // the GL reference. Promote the composite accumulators (and the weight
  // arithmetic) to float; the per-sample TF values stay half.
  float3 accumulatedColor = float3(initialColor);
  float accumulatedOpacity = float(initialOpacity);

  // Non-composite blend-mode accumulators. Only the active mode's accumulator
  // is ever touched; dead branches are eliminated via the fc_blendMode function
  // constant so composite pipelines carry no extra cost.
  float mipMaxScalar = 0.0f;    // MIP: max normalized scalar along the ray
  float minipMinScalar = 1.0f;  // MinIP: min normalized scalar along the ray
  float avgBlendSum = 0.0f;     // AverageIP: sum(opacity * scalar) over in-range samples
  int avgBlendCount = 0;       // AverageIP: number of in-range samples
  float additiveSum = 0.0f;    // Additive: sum(opacity * scalar)
  bool firstBlendSample = true;

  // Per-component accumulators for the independent multi-component path.
  // Only the active blend mode's arrays are touched.
  float mipMaxScalarComp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float minipMinScalarComp[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  float avgBlendSumComp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  int avgBlendCountComp[4] = {0, 0, 0, 0};
  float additiveSumComp[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  // Per-component gradients (independent path only): computed lazily at most
  // once per sample from a single six-texel batch and reused by the
  // gradient-opacity step and the per-component shading in the composite loop.
  float4 compGrad[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
  bool compGradReady = false;

  // Sample position carried incrementally through the march: advance one ray
  // step per iteration instead of recomputing. texLocalPos lives in [0,1]
  // texture space; currentPoint is in normalized volume space (the AABB).
  float3 texLocalPos = (volumeUniforms.volumeToTexture *
      float4(volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize, 1.0)).xyz;
  // OpenGL g_rayOrigin parity (camera-inside proxy): GL anchors the march at the
  // interpolated, cell-to-point-adjusted texture coordinate plus one (jitter-
  // scaled) step: g_rayOrigin = ip_textureCoords + g_rayJitter with g_rayJitter =
  // g_dirStep * jitterValue. Anchoring on the ray-box entry (texLocalPos) here
  // reproduces a ~9.7e-5 texel offset in z (the near-plane entry differs from the
  // interpolated anchor), which puts the fetch positions on the wrong side of
  // tissue boundaries (the systematic knife-edge mismatch). For those paths
  // p.localPos carries the interpolated per-vertex texcoord (in.texcoord) and
  // evalStep already replicates g_dirStep, so the start is bit-parity with GL.
  // Legacy paths (camera outside, grid traversal) keep the entry-anchored
  // construction.
  float jitterFrac = p.stepSize > 0.0 ? (p.jitter / p.stepSize) : 1.0;
  float3 evalPoint = (p.anchorIsData && volumeUniforms.useParallelProjection < 0.5)
      ? (p.localPos + evalStep * jitterFrac)
      : cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
  float prefetchScalar = sampleVolumeScalar(volumeTexture,
      rectilinearSamplePosition(evalPoint, doRectilinear, rectCoords, volumeUniforms));
  float prefetchMask = doMask ? maskTexture.sample(sNearest, evalPoint, level(0)).r : 0.0;
  bool prefetchValid = true;
  int3  curCell     = int3(-1);
  bool  curCellEmpty = false;
  float3 mmDimF     = b.minMaxInfo.yzw;
  const bool useMinMax = fc_minmax &&
    !useIndependentPath &&
    b.minMaxInfo.x > 0.5 &&
    b.minMaxInfo.y > 0.5 &&
    b.minMaxInfo.z > 0.5 &&
    b.minMaxInfo.w > 0.5;
  // True once any sample inside [0,1]^3 texture space has been reached. The
  // texture cube is axis-aligned and the ray is a straight line in texture
  // space, so a ray's in-bounds samples form a single contiguous interval:
  // after it has been inside and gone out, it can never re-enter.
  bool seenInBounds = false;

  // DEBUG: one header per gated fragment (test builds only).
#if defined(VTK_METAL_ENABLE_LOGGING)
    if (p.screenPos.x > 0.0 && (debugStepGate(volumeUniforms.cameraVolumePos.xyz, p.screenPos) || analyticPixelGate(p.screenPos))) {
    os_log_default.log_info("VTK_METAL_VOLUME_LOG DEBUG MARCH px=(%d, %d) camera=(%f, %f, %f) rayDir=(%f, %f, %f) tStart=%f tEnd=%f useClip=%f nClip=%f p0o=(%f, %f, %f) p0n=(%f, %f, %f) p1o=(%f, %f, %f) p1n=(%f, %f, %f) stepSize=%f firstT=%f jitter=%f entry=(%f, %f, %f) scalarMin=%f scalarMax=%f texelCount=(%f, %f, %f) texelDims=(%u, %u, %u)",
        int(p.screenPos.x), int(p.screenPos.y),
        volumeUniforms.cameraVolumePos.x, volumeUniforms.cameraVolumePos.y, volumeUniforms.cameraVolumePos.z,
        p.rayDir.x, p.rayDir.y, p.rayDir.z,
        p.tStart, p.tEnd,
        float(volumeUniforms.useClipping), float(volumeUniforms.numClippingPlanes),
        volumeUniforms.clippingPlane0Origin.x, volumeUniforms.clippingPlane0Origin.y, volumeUniforms.clippingPlane0Origin.z,
        volumeUniforms.clippingPlane0Normal.x, volumeUniforms.clippingPlane0Normal.y, volumeUniforms.clippingPlane0Normal.z,
        volumeUniforms.clippingPlane1Origin.x, volumeUniforms.clippingPlane1Origin.y, volumeUniforms.clippingPlane1Origin.z,
        volumeUniforms.clippingPlane1Normal.x, volumeUniforms.clippingPlane1Normal.y, volumeUniforms.clippingPlane1Normal.z,
        p.stepSize, firstT, p.jitter,
        p.rayOrigin.x + p.rayDir.x * p.tStart,
        p.rayOrigin.y + p.rayDir.y * p.tStart,
        p.rayOrigin.z + p.rayDir.z * p.tStart,
        float(volumeUniforms.scalarMin), float(volumeUniforms.scalarMax),
        float(texelCount.x), float(texelCount.y), float(texelCount.z),
        volumeTexture.get_width(), volumeTexture.get_height(), volumeTexture.get_depth());
  }
#endif

  // DEBUG: full-precision step geometry (test builds only). The per-sample
  // position fit (findings update 16) showed the Metal evalStep and the GL
  // g_dirStep differ per-axis (x +0.03%, y +0.41%, z -0.003%), i.e. NOT a
  // uniform scale. Dump the full float32 chain so it can be diffed against
  // GL's GL_RAY step= / GL_UNIFORMS values at the same pixel and frame:
  //   p.rayDir          : normalized object-space direction (in.localPos - cam)
  //   dirObj            : normalize(p.rayDir * boundsSize) -- GL normalize is in
  //                       object space only; the *boundsSize re-normalize can
  //                       rotate the direction if boundsSize is non-uniform
  //   evalStep          : (adjustedLin * dirObj) * sampleDistanceWorld
  //   texStep           : rayDirTexLocal * p.stepSize (ray-loop tex advance)
  //   adjustedLin       : volumeToTexture rows scaled by ctpScale
  //   boundsSize        : volumeBoundsMax - volumeBoundsMin
  //   sampleDistanceWorld: GL in_sampleDistance (world units)
#if defined(VTK_METAL_ENABLE_LOGGING)
    if (p.screenPos.x > 0.0 && (debugStepGate(volumeUniforms.cameraVolumePos.xyz, p.screenPos) || analyticPixelGate(p.screenPos))) {
    os_log_default.log_info("VTK_METAL_VOLUME_LOG DEBUG STEP px=(%d, %d) screenPos=(%0.9e, %0.9e) flatVid=%u primId=%u cameraVol=(%0.9e, %0.9e, %0.9e) localPos=(%0.9e, %0.9e, %0.9e) clip=(%0.9e, %0.9e, %0.9e, %0.9e) anchorData=(%0.9e, %0.9e, %0.9e) rayDir=(%0.9e, %0.9e, %0.9e) dirObj=(%0.9e, %0.9e, %0.9e) evalStep=(%0.9e, %0.9e, %0.9e) texStep=(%0.9e, %0.9e, %0.9e) boundsSize=(%0.9e, %0.9e, %0.9e) sampleDistanceWorld=%0.9e ctpScale=(%0.9e, %0.9e, %0.9e) ctpOffset=(%0.9e, %0.9e, %0.9e) dirBits=%08x%08x%08x evalStepBits=%08x%08x%08x nearBits=%08x%08x%08x farBits=%08x%08x%08x dBits=%08x%08x%08x invPVMBits=%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x%08x",
        int(p.screenPos.x), int(p.screenPos.y),
        p.screenPos.x, p.screenPos.y,
        p.flatVid, p.primId,
        volumeUniforms.cameraVolumePos.x, volumeUniforms.cameraVolumePos.y, volumeUniforms.cameraVolumePos.z,
        p.localPos.x, p.localPos.y, p.localPos.z,
        p.clipPos.x, p.clipPos.y, p.clipPos.z, p.clipPos.w,
        p.anchorData.x, p.anchorData.y, p.anchorData.z,
        p.rayDir.x, p.rayDir.y, p.rayDir.z,
        dirObj.x, dirObj.y, dirObj.z,
        evalStep.x, evalStep.y, evalStep.z,
        texStep.x, texStep.y, texStep.z,
        boundsSize.x, boundsSize.y, boundsSize.z,
        volumeUniforms.sampleDistanceWorld,
        ctpScale.x, ctpScale.y, ctpScale.z,
        ctpOffset.x, ctpOffset.y, ctpOffset.z,
        as_type<uint>(dirObj.x), as_type<uint>(dirObj.y), as_type<uint>(dirObj.z),
        as_type<uint>(evalStep.x), as_type<uint>(evalStep.y), as_type<uint>(evalStep.z),
        as_type<uint>(dbgNearP.x), as_type<uint>(dbgNearP.y), as_type<uint>(dbgNearP.z),
        as_type<uint>(dbgFarP.x), as_type<uint>(dbgFarP.y), as_type<uint>(dbgFarP.z),
        as_type<uint>(d.x), as_type<uint>(d.y), as_type<uint>(d.z),
        as_type<uint>(volumeUniforms.inversePVM[0][0]), as_type<uint>(volumeUniforms.inversePVM[0][1]),
        as_type<uint>(volumeUniforms.inversePVM[0][2]), as_type<uint>(volumeUniforms.inversePVM[0][3]),
        as_type<uint>(volumeUniforms.inversePVM[1][0]), as_type<uint>(volumeUniforms.inversePVM[1][1]),
        as_type<uint>(volumeUniforms.inversePVM[1][2]), as_type<uint>(volumeUniforms.inversePVM[1][3]),
        as_type<uint>(volumeUniforms.inversePVM[2][0]), as_type<uint>(volumeUniforms.inversePVM[2][1]),
        as_type<uint>(volumeUniforms.inversePVM[2][2]), as_type<uint>(volumeUniforms.inversePVM[2][3]),
        as_type<uint>(volumeUniforms.inversePVM[3][0]), as_type<uint>(volumeUniforms.inversePVM[3][1]),
        as_type<uint>(volumeUniforms.inversePVM[3][2]), as_type<uint>(volumeUniforms.inversePVM[3][3]));
  }
#endif

  int lastIter = -1;
  int breakWhy = 0;
  float3 breakEval = float3(-2.0f);
  for (int i = 0; i < maxSteps; i++) {
    if (!p.checkBounds && currentT >= maxSteps) { breakWhy = 1; breakEval = evalPoint; break; }

    // The proxy box spans the axis-aligned bounds of the rotated volume, so
    // rays through its corner regions fall outside the [0,1]^3 texture cube.
    // The OpenGL backend never skips these samples: its clamp-to-edge sampler
    // clamps the coordinate back into the cube and the boundary voxel is
    // composited (grazing rays, rotated-volume corner regions). Replicate that
    // by clamping texLocalPos and sampling the boundary slab. seenInBounds
    // stays false while entry-side samples are outside so the ray keeps
    // marching through rotated-volume corner regions; axis-aligned grazing
    // rays still terminate via the block-bounds exit below, and once the ray
    // has been inside the cube and left it, stop entirely.
    // OpenGL TerminationImplementation parity (vtkVolumeShaderComposer.h): the
    // loop breaks when the g_dirStep lattice position leaves the cell-to-point
    // adjusted bounds along any axis of travel. g_dataPos lives on the
    // g_dirStep lattice (== evalPoint here) in adjusted texture space, so the
    // test is a directional per-axis comparison against [ctpOffset, ctpOffset
    // + ctpScale]. Testing the raw texStep lattice (texLocalPos vs [0,1])
    // instead crossed the SAME exit planes but on a lattice whose step is
    // ~0.2% longer than g_dirStep, so ~2% of rays exited one sample early and
    // composited one fewer positive-opacity term (findings update 68).
    const float3 adjTexMin = ctpOffset;
    const float3 adjTexMax = ctpOffset + ctpScale;
    if (any(max(evalStep, float3(0.0f)) * (evalPoint - adjTexMax) > float3(0.0f)) ||
        any(min(evalStep, float3(0.0f)) * (evalPoint - adjTexMin) > float3(0.0f))) {
      if (seenInBounds) { breakWhy = 2; breakEval = evalPoint; break; }
      texLocalPos = clamp(texLocalPos, float3(0.0), float3(1.0));
      // Keep the camera-inside proxy anchored on the interpolated texcoord (GL
      // g_rayOrigin parity) after the out-of-bounds clamp; the counter-based
      // rebuild is exact for the current sample index.
      evalPoint = (p.anchorIsData && volumeUniforms.useParallelProjection < 0.5)
          ? (p.localPos + evalStep * (jitterFrac + float(currentT)))
          : cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
      prefetchValid = false;
    } else {
      seenInBounds = true;
    }

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
        tToEdge.x = abs(rayDirTexLocal.x) > 1e-5 ? distToEdge.x / abs(rayDirTexLocal.x * mmDimF.x) : 1e30;
        tToEdge.y = abs(rayDirTexLocal.y) > 1e-5 ? distToEdge.y / abs(rayDirTexLocal.y * mmDimF.y) : 1e30;
        tToEdge.z = abs(rayDirTexLocal.z) > 1e-5 ? distToEdge.z / abs(rayDirTexLocal.z * mmDimF.z) : 1e30;

        float exactSkip = min(min(tToEdge.x, tToEdge.y), tToEdge.z);
        exactSkip += 1e-4;
        // Skip an integer number of steps; keep the float distance consistent
        // with the integer counter (skipDist == skipSteps * p.stepSize).
        int skipSteps = max(1, int(ceil(exactSkip / p.stepSize)));
        float skipDist = float(skipSteps) * p.stepSize;

        currentPoint += p.rayDir * skipDist;
        currentT += skipSteps;

        if (p.checkBounds && (any(currentPoint < p.blockMinGlobal - 1e-4) || any(currentPoint > p.blockMaxGlobal + 1e-4) || currentT >= maxSteps)) {
          break;
        }

        // Re-sync the incremental sample position after the empty-cell jump.
        texLocalPos = (volumeUniforms.volumeToTexture *
            float4(volumeUniforms.volumeBoundsMin.xyz + currentPoint * boundsSize, 1.0)).xyz;
        // Camera-inside proxy: rebuild from the integer counter so the skip
        // stays on GL's g_rayOrigin + g_dirStep * g_currentT lattice.
        evalPoint = (p.anchorIsData && volumeUniforms.useParallelProjection < 0.5)
            ? (p.localPos + evalStep * (jitterFrac + float(currentT)))
            : cellToPointTextureCoord(texLocalPos, ctpScale, ctpOffset);
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
        (needsFetch || useIndependentPath || volumeUniforms.useDependentRGBA > 0.5 ||
         volumeUniforms.useDependentLA > 0.5)) {
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
        rawScalar4 = volumeTexture.sample(sVolume, rectEvalPoint, level(0));
      } else {
        rawScalar4 = volumeTexture.sample(sNearest, rectEvalPoint, level(0));
      }
    } else if (volumeUniforms.useDependentRGBA > 0.5 || volumeUniforms.useDependentLA > 0.5) {
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
      currentT += 1;
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
        currentT += 1;
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
        currentT += 1;
        texLocalPos += texStep;
        evalPoint += evalStep;
        prefetchValid = false;
        continue;
      }
    }

    float scalarNorm = saturate(rawScalar * scalarScale + scalarBias);

    // Per-component normalization against each component's own scalar range
    // (OpenGL in_scalarsRange parity); defaults to the single-path norm when
    // the independent path is inactive.
    float scalarNormComp[4] = {scalarNorm, scalarNorm, scalarNorm, scalarNorm};
    // Per-component scalar->normalized scale/bias (OpenGL in_scalarsRange
    // parity), computed here once per sample so the density-gradient shading
    // path reuses them instead of recomputing the reciprocal per lit sample.
    float compScale[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float compBias[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (useIndependentPath) {
      int nComp = min(4, int(volumeUniforms.numComponents));
      for (int c = 0; c < nComp; ++c) {
        float cMin = volumeUniforms.scalarMinComp[c];
        float cRange = max(volumeUniforms.scalarMaxComp[c] - cMin, 1e-4f);
        compScale[c] = 1.0f / cRange;
        compBias[c] = -cMin / cRange;
        float rawComp;
        if (c == 0) rawComp = rawScalar;
        else if (c == 1) rawComp = rawScalar4.g;
        else if (c == 2) rawComp = rawScalar4.b;
        else rawComp = rawScalar4.a;
        scalarNormComp[c] = saturate((rawComp - cMin) / cRange);
      }
    }

    float4 colorOpacity;
    float maskLabel = 0.0f;
    // Gradient shared between the TF_2D gradient y-axis and shading/gradient
    // opacity so it is computed at most once per sample (computeGradientFast
    // is 6 texture fetches).
    float4 sharedGrad = float4(0.0f);
    bool sharedGradReady = false;
    // Opacity-field gradient cached by the gradient-opacity block when
    // ComputeNormalFromOpacity is combined with gradient opacity, so the
    // shading block reuses the shared six-neighbor fetch instead of refetching.
    float4 cachedDensityGrad = float4(0.0f);
    bool densityGradReady = false;

    // Per-component transfer-function results (independent path only).
    float4 compColor[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};

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
          compColor[c] = float4(sampleComponentTransferFunction(
              transferFunctionTexture, transferFunctionTexture1,
              transferFunctionTexture2, transferFunctionTexture3,
              float2(scalarNormComp[c], 0.5), c));
        }
      }
    } else if (fc_needsPerSampleOpacity && doTransfer2D) {
      float secondNorm;
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
            sampleSecondScalar(transfer2DYAxisTexture, evalPoint) * secondScale + secondBias);
      }
      colorOpacity = float4(sampleTransferFunction2D(
          transferFunction2DTexture, float2(scalarNorm, secondNorm)));
    } else if (fc_needsPerSampleOpacity && doMask) {
      float maskVal = rawMask * maskScale + maskBias;
      if (numLabels > 0.0) {
        float label = floor(maskVal + 0.5);
        if (label > 0.0) {
          label = clamp(label, 1.0, numLabels - 1.0);
          maskLabel = label;
          float labelY = (label + 0.5) / numLabels;
          colorOpacity = float4(labelMapTransferTexture.sample(sNearest, float2(float(scalarNorm), labelY), level(0)));
        } else {
          colorOpacity = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
        }
      } else {
        colorOpacity = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
      }
    } else if (fc_needsPerSampleOpacity) {
      if (volumeUniforms.useDependentRGBA > 0.5) {
        // 4-component dependent RGBA: color is the raw RGB channels and opacity
        // comes from the 4th component mapped through the opacity LUT (OpenGL
        // computeColor/computeOpacity RGBA parity: computeColor returns
        // vec4(scalar.xyz, opacity), computeOpacity reads scalar.w). The LUT is
        // built over the last component's scalar range, so the raw normalized
        // fetch (rawScalar4.a) is the table coordinate.
        float rgbaOpacity =
          sampleTransferFunction(transferFunctionTexture, float2(rawScalar4.a, 0.5)).a;
        colorOpacity = float4(rawScalar4.rgb, rgbaOpacity);
      } else if (volumeUniforms.useDependentLA > 0.5) {
        // 2-component dependent LA: color is the color LUT at the first
        // component's normalized value (scalarNorm, RGB channels) and opacity
        // is the opacity LUT at the LAST component's normalized value (A
        // channel) — OpenGL computeColor/computeOpacity LA parity (color at
        // scalar.x, opacity at scalar.y). The two LUTs share the single RGBA
        // table — RGB over component 0's range, A over the last component's
        // range — so it is sampled at the two different coordinates.
        float4 laColor = sampleTransferFunction(
            transferFunctionTexture, float2(float(scalarNorm), 0.5));
        float lastMin = volumeUniforms.scalarMinComp[1];
        float lastMax = volumeUniforms.scalarMaxComp[1];
        float lastNorm = saturate(
            (rawScalar4.g - lastMin) / max(lastMax - lastMin, 1e-4f));
        float laOpacity = sampleTransferFunction(
            transferFunctionTexture, float2(lastNorm, 0.5)).a;
        colorOpacity = float4(laColor.rgb, laOpacity);
      } else {
        colorOpacity = sampleTransferFunction(transferFunctionTexture, float2(float(scalarNorm), 0.5));
      }
    } else {
      colorOpacity = float4(0.0f);
    }

    float sampleOpacity = colorOpacity.a;

    // DEBUG: per-sample march data (test builds only).
#if defined(VTK_METAL_ENABLE_LOGGING)
    if (p.screenPos.x > 0.0 && debugMarchGate(volumeUniforms.cameraVolumePos.xyz, p.screenPos)) {
      os_log_default.log_info("VTK_METAL_VOLUME_LOG DEBUG SAMPLE px=(%d, %d) i=%d t=%f tex=(%f, %f, %f) eval=(%f, %f, %f) raw=%f norm=%f op=%f mip=%f rgb=(%f, %f, %f) w=%f accA=%f accC=(%f, %f, %f) maxSteps=%d termMax=%f",
          int(p.screenPos.x), int(p.screenPos.y), i, firstT + float(currentT) * p.stepSize,
          texLocalPos.x, texLocalPos.y, texLocalPos.z,
          evalPoint.x, evalPoint.y, evalPoint.z,
          rawScalar, float(scalarNorm), float(sampleOpacity), float(mipMaxScalar),
          float(colorOpacity.r), float(colorOpacity.g), float(colorOpacity.b),
          float(1.0h - accumulatedOpacity),
          float(accumulatedOpacity),
          float(accumulatedColor.r), float(accumulatedColor.g), float(accumulatedColor.b),
          maxSteps, p.tTerminateMax);
    }
#endif

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
      float totalAlpha = 0.0f;
      int nComp = min(4, int(volumeUniforms.numComponents));
      for (int c = 0; c < nComp; ++c) {
        if (volumeUniforms.componentWeight[c] <= 0.0) continue;
        totalAlpha += compColor[c].a * volumeUniforms.componentWeight[c];
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
          float intensityNorm =
            volumeUniforms.scalarMinComp[c] +
            (volumeUniforms.scalarMaxComp[c] - volumeUniforms.scalarMinComp[c]) * scalarNormComp[c];
          if (intensityNorm >= volumeUniforms.averageIPRangeMin &&
              intensityNorm <= volumeUniforms.averageIPRangeMax) {
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
      float intensityNorm =
        volumeUniforms.scalarMin + (volumeUniforms.scalarMax - volumeUniforms.scalarMin) * scalarNorm;
      if (intensityNorm >= volumeUniforms.averageIPRangeMin &&
          intensityNorm <= volumeUniforms.averageIPRangeMax) {
        avgBlendSum += sampleOpacity * scalarNorm;
        avgBlendCount++;
      }
    } else if (fc_blendMode == 4) {    // ADDITIVE_BLEND
      additiveSum += sampleOpacity * scalarNorm;
    }
    // RenderToImage depth: record the world position of the first non-skipped
    // sample whose transfer-function opacity is positive (matches the OpenGL
    // backend's l_opaqueFragPos update).
    if (haveOpaquePos != nullptr && *haveOpaquePos && sampleOpacity > 0.0f) {
      *firstOpaquePos = currentPoint;
      *haveOpaquePos = false;
    }
    // Opacity pre-integration is baked into the transfer function texture
    // on the CPU at TF-build time (matches OpenGL backend).

    if (useIndependentPath) {
      // OpenGL composites every sample with positive opacity (g_srcColor.a > 0.0);
      // the 0.001h gate here would drop low-opacity leading samples (border rays).
      if (fc_blendMode == 0 && sampleOpacity > 0.0f) {
        // OpenGL composite accumulation: per-component colors are combined via
        // g_srcColor = sum(color[i] * weight[i]) and the weighted opacity sum
        // is used for the alpha accumulation (srcBlend = dstAlpha factor),
        // matching vtkVolumeShaderComposer's independent-component loop.
        float3 tmpRGB = float3(0.0f);
        float tmpA = 0.0f;
        int nComp = min(4, int(volumeUniforms.numComponents));
        for (int c = 0; c < nComp; ++c) {
          float w = float(volumeUniforms.componentWeight[c]);
          if (w <= 0.0f) continue;
          float4 cc = compColor[c];
          float3 ccRGB = cc.rgb;
          if (sampleOpacity >= 0.01f && doShading) {
            float3 normal;
            if (fc_computeNormalFromOpacity && volumeUniforms.useComputeNormalFromOpacity > 0.5) {
              normal = float3(computeDensityGradientFast(volumeTexture,
                  transferFunctionTexture, transferFunctionTexture1,
                  transferFunctionTexture2, transferFunctionTexture3,
                  evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture,
                  gradNormFactor, c, compScale[c], compBias[c]).xyz);
            } else if (fc_normalTexture && volumeUniforms.useNormalTexture > 0.5) {
              float4 nrmSample = float4(normalTexture.sample(sVolume, evalPoint, level(0)));
              normal = normalize(nrmSample.xyz * 2.0f - 1.0f);
            } else {
              if (!compGradReady) {
                computeGradientsAllComponents(volumeTexture, evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor, compGrad);
                compGradReady = true;
              }
              normal = float3(compGrad[c].xyz);
            }
            // Per-component material and shininess (OpenGL lightingComponent
            // index parity).
            float3 ambC = float3(volumeUniforms.ambientColorComp[c].rgb);
            float3 difC = float3(volumeUniforms.diffuseColorComp[c].rgb);
            float3 speC = float3(volumeUniforms.specularColorComp[c].rgb);
            float  shiC = float(volumeUniforms.shininessComp[c]);
            if (lightUniforms != nullptr && lightUniforms->defaultLighting == 0) {
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
        float weight = 1.0f - accumulatedOpacity;
        accumulatedColor += weight * tmpRGB;
        accumulatedOpacity += weight * tmpA;
      }
    } else if (fc_blendMode == 0 && sampleOpacity > 0.0f) {
      float3 sampleColor = colorOpacity.rgb;
      float weight = 1.0f - accumulatedOpacity;

      // Gradient magnitude for gradient opacity and/or shading, computed at
      // most once per sample (sharedGrad/sharedGradReady). Gradient opacity is
      // applied to the per-sample alpha whenever the property declares it,
      // independent of shading — OpenGL ComputeLightingSingleInput parity
      // (color.a *= computeGradientOpacity(gradient)), which fixes the
      // dependent-component path rendering flat solid color when shading is off.
      if (doGradOp && maskLabel == 0.0f) {
        if (!sharedGradReady) {
          if (fc_normalTexture && volumeUniforms.useNormalTexture > 0.5) {
            float4 nrmSample = float4(normalTexture.sample(sVolume, evalPoint, level(0)));
            sharedGrad = float4(float3(normalize(nrmSample.xyz * 2.0f - 1.0f)), nrmSample.w);
          } else if (fc_computeNormalFromOpacity && volumeUniforms.useComputeNormalFromOpacity > 0.5) {
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
        float opBeforeGf = sampleOpacity;
        sampleOpacity *= float(sampleGradientOpacity(gradientOpacityTexture, float(sharedGrad.w)));
#if defined(VTK_METAL_ENABLE_LOGGING)
        if (p.screenPos.x > 0.0 && debugMarchGate(volumeUniforms.cameraVolumePos.xyz, p.screenPos)) {
          half dSPX = half(sampleVolumeScalar(volumeTexture, evalPoint + float3(b.gradientStep.x, 0, 0)));
          half dSNX = half(sampleVolumeScalar(volumeTexture, evalPoint - float3(b.gradientStep.x, 0, 0)));
          half dSPY = half(sampleVolumeScalar(volumeTexture, evalPoint + float3(0, b.gradientStep.y, 0)));
          half dSNY = half(sampleVolumeScalar(volumeTexture, evalPoint - float3(0, b.gradientStep.y, 0)));
          half dSPZ = half(sampleVolumeScalar(volumeTexture, evalPoint + float3(0, 0, b.gradientStep.z)));
          half dSNZ = half(sampleVolumeScalar(volumeTexture, evalPoint - float3(0, 0, b.gradientStep.z)));
          os_log_default.log_info("VTK_METAL_VOLUME_LOG DEBUG GRADOP px=(%d, %d) i=%d pos=(%f, %f, %f) gradW=%f gradOp=%f opBefore=%f opAfter=%f norm=%f raw=%f gstep=(%f, %f, %f) nb=(%f, %f, %f, %f, %f, %f)",
              int(p.screenPos.x), int(p.screenPos.y), i,
              evalPoint.x, evalPoint.y, evalPoint.z,
              float(sharedGrad.w), float(sampleGradientOpacity(gradientOpacityTexture, float(sharedGrad.w))),
              float(opBeforeGf), float(sampleOpacity), float(scalarNorm), rawScalar,
              b.gradientStep.x, b.gradientStep.y, b.gradientStep.z,
              float(dSPX), float(dSNX), float(dSPY), float(dSNY), float(dSPZ), float(dSNZ));
        }
#endif
      }

      // Apply shading to every alpha>0 sample, matching OpenGL's composite
      // (g_srcColor = computeColor() whenever g_srcColor.a > 0.0). The previous
      // 0.01 early-exit left low-gradient samples unlit (flat LUT color), which
      // diverged from OpenGL's lit color and rendered the shaded result
      // systematically brighter.
      if (doShading && maskLabel == 0.0f && sampleOpacity > 0.0f) {

        float3 normal;
        if (fc_computeNormalFromOpacity && volumeUniforms.useComputeNormalFromOpacity > 0.5) {
          if (densityGradReady) {
            normal = float3(cachedDensityGrad.xyz);
          } else {
            normal = float3(computeDensityGradientFast(volumeTexture,
                transferFunctionTexture, transferFunctionTexture1,
                transferFunctionTexture2, transferFunctionTexture3,
                evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture,
                gradNormFactor, 0, scalarScale, scalarBias).xyz);
          }
        } else {
          if (!sharedGradReady) {
            if (fc_normalTexture && volumeUniforms.useNormalTexture > 0.5) {
              float4 nrmSample = float4(normalTexture.sample(sVolume, evalPoint, level(0)));
              sharedGrad = float4(float3(normalize(nrmSample.xyz * 2.0f - 1.0f)), nrmSample.w);
            } else {
              sharedGrad = computeGradientFast(volumeTexture, evalPoint, b.gradientStep.xyz, volumeUniforms.volumeToTexture, gradNormFactor);
            }
            sharedGradReady = true;
          }
          normal = float3(sharedGrad.xyz);
        }

        // TEMP DEBUG: lighting per-sample values (test builds only).
#if defined(VTK_METAL_ENABLE_LOGGING)
        if (p.screenPos.x > 0.0 && debugMarchGate(volumeUniforms.cameraVolumePos.xyz, p.screenPos)) {
          float nDotLDbg = dot(normal, viewDirHalf);
          os_log_default.log_info("VTK_METAL_VOLUME_LOG DEBUG LIGHT px=(%d, %d) i=%d raw=%f norm=%f opIn=%f gradW=%f gradOp=%f nDotL=%f amb=(%f, %f, %f) dif=(%f, %f, %f) spe=(%f, %f, %f) colBefore=(%f, %f, %f) colAfter=(%f, %f, %f) vd=(%f, %f, %f) pos=(%f, %f, %f) nrm=(%f, %f, %f)",
              int(p.screenPos.x), int(p.screenPos.y), i,
              rawScalar, float(scalarNorm), float(sampleOpacity),
              float(sharedGrad.w), float(sampleGradientOpacity(gradientOpacityTexture, float(sharedGrad.w))),
              float(nDotLDbg),
              float(ambientMat.x), float(ambientMat.y), float(ambientMat.z),
              float(diffuseMat.x), float(diffuseMat.y), float(diffuseMat.z),
              float(specularMat.x), float(specularMat.y), float(specularMat.z),
              float(colorOpacity.r), float(colorOpacity.g), float(colorOpacity.b),
              float(sampleColor.r), float(sampleColor.g), float(sampleColor.b),
              float(viewDirHalf.x), float(viewDirHalf.y), float(viewDirHalf.z),
              float(evalPoint.x), float(evalPoint.y), float(evalPoint.z),
              float(normal.x), float(normal.y), float(normal.z));
        }
#endif

        if (lightUniforms != nullptr && lightUniforms->defaultLighting == 0) {
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

        // TEMP DEBUG (after lighting applied).
#if defined(VTK_METAL_ENABLE_LOGGING)
        if (p.screenPos.x > 0.0 && debugMarchGate(volumeUniforms.cameraVolumePos.xyz, p.screenPos)) {
          os_log_default.log_info("VTK_METAL_VOLUME_LOG DEBUG LIGHT2 px=(%d, %d) i=%d nDotL=%f litBefore=(%f, %f, %f) litAfter=(%f, %f, %f)",
              int(p.screenPos.x), int(p.screenPos.y), i,
              float(dot(normal, viewDirHalf)),
              float(colorOpacity.r), float(colorOpacity.g), float(colorOpacity.b),
              float(sampleColor.r), float(sampleColor.g), float(sampleColor.b));
        }
#endif
      } else if (doShading) {
        sampleColor = float3(ambientMat) * sampleColor;
      }

      accumulatedColor = fma(weight, sampleColor * sampleOpacity, accumulatedColor);
      accumulatedOpacity = fma(weight, sampleOpacity, accumulatedOpacity);
    }

    currentPoint += stepVec;
    currentT += 1;
    texLocalPos += texStep;
    evalPoint += evalStep;
    lastIter = i;

    if (i + 1 < maxSteps) {
      prefetchScalar = sampleVolumeScalar(volumeTexture,
          rectilinearSamplePosition(evalPoint, doRectilinear, rectCoords, volumeUniforms));
      if (doMask) {
        prefetchMask = maskTexture.sample(sNearest, evalPoint, level(0)).r;
      }
      prefetchValid = true;
    }

    // OpenGL parity: g_opacityThreshold = 1.0 - 1.0/255.0 (vtkVolumeShaderComposer.h).
    // OpenGL breaks WITHOUT clamping the accumulated opacity (TerminationImplementation
    // in vtkVolumeShaderComposer.h: `g_fragColor.a > g_opacityThreshold`). Clamping here
    // made 1-src.a = 0 at blend time, dropping the background blend term that GL keeps
    // (dst*(1-a), a ~ 0.9969). Keep the raw accumulated opacity for blend parity.
    if (accumulatedOpacity > 1.0f - 1.0f / 255.0f) {
      breakWhy = 3; breakEval = evalPoint;
      break;
    }
    // OpenGL g_terminatePointMax parity: the counter is compared in units of
    // |g_dirStep| (== |evalStep|), not |texStep| (findings update 68).
    if (firstT + float(currentT) * length(evalStep) >= p.tTerminateMax) {
      breakWhy = 4; breakEval = evalPoint;
      break;
    }
    // OpenGL has no block-bounds exit: TerminationImplementation breaks only on
    // the CTP bounds test (g_dataPos vs in_texMin/in_texMax), the opacity
    // threshold, and g_currentT >= g_terminatePointMax. The block-bounds check
    // (currentPoint vs blockMinGlobal/blockMaxGlobal on the separate ray-box
    // lattice) fired one sample BEFORE the CTP test on the evalPoint lattice,
    // making the camera-inside proxy composite one fewer positive-opacity term
    // than GL at exit-boundary pixels. Rely on the CTP test alone for parity.
    // (Loop bound i < maxSteps still caps runaway rays.)
  }

  float4 finalColor;
  if (useIndependentPath) {
    if (fc_blendMode == 1) {   // MAXIMUM_INTENSITY_BLEND
      // Per-component extremum re-sampled through each component's own table
      // and combined by weight (OpenGL ShadingExit parity).
      float3 c = float3(0.0f);
      float a = 0.0f;
      int nComp = min(4, int(volumeUniforms.numComponents));
      for (int i = 0; i < nComp; ++i) {
        float4 t = float4(sampleComponentTransferFunction(
            transferFunctionTexture, transferFunctionTexture1,
            transferFunctionTexture2, transferFunctionTexture3,
            float2(mipMaxScalarComp[i], 0.5), i));
        float w = volumeUniforms.componentWeight[i];
        c += t.rgb * t.a * w;
        a += t.a * w;
      }
      finalColor = float4(c, a);
    } else if (fc_blendMode == 2) {  // MINIMUM_INTENSITY_BLEND
      float3 c = float3(0.0f);
      float a = 0.0f;
      int nComp = min(4, int(volumeUniforms.numComponents));
      for (int i = 0; i < nComp; ++i) {
        float4 t = float4(sampleComponentTransferFunction(
            transferFunctionTexture, transferFunctionTexture1,
            transferFunctionTexture2, transferFunctionTexture3,
            float2(minipMinScalarComp[i], 0.5), i));
        float w = volumeUniforms.componentWeight[i];
        c += t.rgb * t.a * w;
        a += t.a * w;
      }
      finalColor = float4(c, a);
    } else if (fc_blendMode == 3) {  // AVERAGE_INTENSITY_BLEND
      // Per-component in-range average combined by weight. OpenGL discards the
      // fragment when no in-range sample was found; return a fully transparent
      // fragment so the background shows through.
      float avg = 0.0f;
      bool anySample = false;
      int nComp = min(4, int(volumeUniforms.numComponents));
      for (int i = 0; i < nComp; ++i) {
        if (avgBlendCountComp[i] > 0) {
          anySample = true;
          avg += saturate(avgBlendSumComp[i] / float(avgBlendCountComp[i])) *
                 volumeUniforms.componentWeight[i];
        }
      }
      if (!anySample) {
        return float4(0.0f);
      }
      finalColor = float4(avg, avg, avg, 1.0f);
    } else if (fc_blendMode == 4) {  // ADDITIVE_BLEND
      float sum = 0.0f;
      int nComp = min(4, int(volumeUniforms.numComponents));
      for (int i = 0; i < nComp; ++i) {
        sum += additiveSumComp[i] * volumeUniforms.componentWeight[i];
      }
      sum = saturate(sum);
      finalColor = float4(sum, sum, sum, 1.0f);
    } else {
      finalColor = float4(accumulatedColor, accumulatedOpacity);
    }
  } else if (fc_blendMode == 1) {   // MAXIMUM_INTENSITY_BLEND
    float4 c = float4(sampleTransferFunction(transferFunctionTexture, float2(mipMaxScalar, 0.5)));
    finalColor = float4(c.rgb * c.a, c.a);
#if defined(VTK_METAL_ENABLE_LOGGING)
    if (p.screenPos.x > 0.0 && debugMarchGate(volumeUniforms.cameraVolumePos.xyz, p.screenPos)) {
      os_log_default.log_info("VTK_METAL_VOLUME_LOG DEBUG MIPFINAL px=(%d, %d) mip=%f c=(%f, %f, %f, %f) out=(%f, %f, %f)",
          int(p.screenPos.x), int(p.screenPos.y),
          float(mipMaxScalar),
          float(c.r), float(c.g), float(c.b), float(c.a),
          float(finalColor.r), float(finalColor.g), float(finalColor.b));
    }
#endif
  } else if (fc_blendMode == 2) {  // MINIMUM_INTENSITY_BLEND
    float4 c = float4(sampleTransferFunction(transferFunctionTexture, float2(minipMinScalar, 0.5)));
    finalColor = float4(c.rgb * c.a, c.a);
  } else if (fc_blendMode == 3) {  // AVERAGE_INTENSITY_BLEND
    // OpenGL discards the fragment when no in-range sample was found; return a
    // fully transparent fragment so the background shows through.
    if (avgBlendCount == 0) {
      return float4(0.0f);
    }
    float avg = saturate(avgBlendSum / float(avgBlendCount));
    finalColor = float4(avg, avg, avg, 1.0f);
  } else if (fc_blendMode == 4) {  // ADDITIVE_BLEND
    float sum = saturate(additiveSum);
    finalColor = float4(sum, sum, sum, 1.0f);
  } else {
    finalColor = float4(accumulatedColor, accumulatedOpacity);
  }

  // Final color window/level (matches OpenGL raycasterfs.glsl finalizeRayCast):
  //   rgb = rgb * in_scale + in_bias * alpha
  float wlScale = volumeUniforms.finalColorScale;
  float wlBias = volumeUniforms.finalColorBias;
  finalColor.rgb = finalColor.rgb * wlScale + wlBias * finalColor.a;

#if defined(VTK_METAL_ENABLE_LOGGING)
  bool gridGate = (((int(p.screenPos.x) & 31) == 16) && ((int(p.screenPos.y) & 31) == 16));
  bool dumpAll = 0;
#if defined(VTK_METAL_FLOAT_DUMP)
  dumpAll = 1;
#endif
  if (p.screenPos.x > 0.0 && (dumpAll || gridGate || debugMarchGate(volumeUniforms.cameraVolumePos.xyz, p.screenPos))) {
    os_log_default.log_info("VTK_METAL_VOLUME_LOG DEBUG FINAL px=(%d, %d) vp=(%f, %f) lastIter=%d accOp=%f accCol=(%f, %f, %f) final=(%f, %f, %f) brkWhy=%d brkEval=(%0.9e, %0.9e, %0.9e) brkT=%d maxS=%d chkB=%d",
        int(p.screenPos.x), int(p.screenPos.y), volumeUniforms.viewportSize.x, volumeUniforms.viewportSize.y, lastIter,
        float(accumulatedOpacity),
        float(accumulatedColor.r), float(accumulatedColor.g), float(accumulatedColor.b),
        float(finalColor.r), float(finalColor.g), float(finalColor.b),
        breakWhy,
        float(breakEval.x), float(breakEval.y), float(breakEval.z),
        currentT, maxSteps, p.checkBounds ? 1 : 0);
  }
#endif
  return finalColor;
}

inline float4 marchVolume(
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
    float4 clipPos,
    float3 anchorData,
    bool anchorIsData,
    uint flatVid,
    uint primId,
    float3 initialColor,
    float initialOpacity,
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
    texture3d<float> normalTexture,
    texture3d<float> blankingTexture,
    constant packed_float3* rectCoords,
    constant VolumeLightUniforms* lightUniforms)
{
  (void)exitPoint;
  (void)totalDist;
  float jitter = (volumeUniforms.useJittering > 0.5 ? volume_random(screenPos) : 1.0) * stepSize;
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
      blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal, screenPos, localPos, clipPos,
      true, anchorData, anchorIsData, flatVid, primId};
  return marchVolumeUnified(p, initialColor, initialOpacity,
      volumeUniforms, b, volumeTexture, transferFunctionTexture,
      transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3,
      transferFunction2DTexture, transfer2DYAxisTexture,
      gradientOpacityTexture, maskTexture, labelMapTransferTexture,
      minMaxTexture, normalTexture, blankingTexture, rectCoords, lightUniforms,
      nullptr, nullptr);
}

inline void marchSegment(
    float3 rayOrigin,
    float3 rayDir,
    float t0,
    float t1,
    float stepSize,
    float jitter,
    float tTerminateMax,
    float2 screenPos,
    thread float3& accumulatedColor,
    thread float& accumulatedOpacity,
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
    texture3d<float> normalTexture,
    texture3d<float> blankingTexture,
    constant packed_float3* rectCoords,
    constant VolumeLightUniforms* lightUniforms)
{
  float3 zero = float3(0.0);
  float3 one = float3(1.0);
  MarchParams p = {rayOrigin, rayDir, t0, t1, stepSize, jitter, tTerminateMax,
      zero, one, zero, one, screenPos, rayOrigin + rayDir * t0, float4(0.0, 0.0, 0.0, 1.0), false,
      rayOrigin + rayDir * t0, false};
  float4 result = marchVolumeUnified(p, accumulatedColor, accumulatedOpacity,
      volumeUniforms, b, volumeTexture, transferFunctionTexture,
      transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3,
      transferFunction2DTexture, transfer2DYAxisTexture,
      gradientOpacityTexture, maskTexture, labelMapTransferTexture,
      minMaxTexture, normalTexture, blankingTexture, rectCoords, lightUniforms,
      nullptr, nullptr);
  accumulatedColor = result.xyz;
  accumulatedOpacity = result.w;
}

// TEMP DEBUG (analytic-anchor experiment): bypass the rasterizer's interpolated
// in.texcoord for the camera-inside proxy anchor and reconstruct the per-fragment
// texcoord from pixel-center barycentrics + per-vertex clip/texcoord (triangle
// anchor buffer, fragment buffer(3)). The buffer is a float4 count header (the
// triangle count in .x) followed by per-triangle records of 3 * 7 floats
// (clip.xyzw, texcoord.xyz per vertex), indexed by primitive_id. Weights are
// computed in window space (y-down, Metal top-left) from the vertex NDC
// (clip.xy / clip.w) and the pixel-center in.position.xy, so they are
// perspective-correct; the attribute is reconstructed as
//   anchorTex = sum_i(l_i * tex_i / clipW_i) / sum_i(l_i / clipW_i)
// exactly the interpolator's (attr/w, 1/w) perspective-correct weighted
// average then divide (update 76 sect 4: the (attr*w, w) form is NOT the
// interpolator's result and does not reproduce GL).
// analyticAnchorMode: 1 = float32 weights, 2 = float64 weights (update 76
// sect 4 A/B). Returns false (caller falls back to in.texcoord) when the
// buffer is absent or primId is out of range.
// MSL has no double type, so the mode-2 path emulates float64 with double-float
// (two-float hi+lo, ~53-bit significand) arithmetic; these helpers compile under
// the file-scope #pragma clang fp contract(off) so twoProd's fma is exact.
struct DoubleFloat { float hi; float lo; };

inline DoubleFloat dfFromFloat(float x) { return DoubleFloat{ x, 0.0f }; }

inline DoubleFloat twoSum(float a, float b)
{
  float s = a + b;
  float v = s - a;
  float e = (a - (s - v)) + (b - v);
  return DoubleFloat{ s, e };
}

inline DoubleFloat twoProd(float a, float b)
{
  float p = a * b;
  float e = fma(a, b, -p);
  return DoubleFloat{ p, e };
}

inline DoubleFloat dfAdd(DoubleFloat a, DoubleFloat b)
{
  DoubleFloat s = twoSum(a.hi, b.hi);
  float lo = s.lo + a.lo + b.lo;
  float hi = s.hi + lo;
  float lo2 = lo - (hi - s.hi);
  return DoubleFloat{ hi, lo2 };
}

inline DoubleFloat dfSub(DoubleFloat a, DoubleFloat b)
{
  return dfAdd(a, DoubleFloat{ -b.hi, -b.lo });
}

inline DoubleFloat dfMul(DoubleFloat a, DoubleFloat b)
{
  DoubleFloat p = twoProd(a.hi, b.hi);
  float lo = fma(a.hi, b.lo, fma(a.lo, b.hi, p.lo));
  float hi = p.hi + lo;
  float lo2 = lo - (hi - p.hi);
  return DoubleFloat{ hi, lo2 };
}

inline DoubleFloat dfDiv(DoubleFloat a, DoubleFloat b)
{
  float q = a.hi / b.hi;
  DoubleFloat r = dfSub(a, dfMul(b, dfFromFloat(q)));
  float q2 = q + r.hi / b.hi;
  return dfFromFloat(q2);
}

inline float dfToFloat(DoubleFloat a) { return a.hi + a.lo; }

// TEMP DEBUG: GL_RAY dump pixels (Metal coords, y = 511 - GL y) so the
// interpolated anchor can be diffed directly against VTK_GL_RAY_DUMP.
inline bool analyticPixelGate(float2 screenPos)
{
  int px = int(floor(screenPos.x - 0.5));
  int py = int(floor(screenPos.y - 0.5));
  return (px == 397 && py == 110) || (px == 256 && py == 256) ||
    (px == 422 && py == 92) || (px == 372 && py == 131) ||
    (px == 360 && py == 229) || (px == 349 && py == 255) ||
    (px == 405 && py == 171) || (px == 9 && py == 18) ||
    (px == 293 && py == 298) || (px == 338 && py == 432) ||
    (px % 32 == 0 && py % 32 == 0) ||
    (px == 307 && (py == 8 || py == 7 || py == 9)) ||
    (px == 480 && py == 400) || (px == 496 && py == 488) ||
    (px == 93 && py == 201) || (px == 242 && py == 330) ||
    (px == 322 && py == 172) || (px == 382 && py == 207) ||
    (px == 357 && py == 154) || (px == 104 && py == 245) ||
    (px == 188 && py == 307) || (px == 350 && py == 5) ||
    (px == 153 && py == 32) || (px == 482 && py == 33) ||
    (px == 120 && py == 167) || (px == 470 && py == 269) ||
    (px == 439 && py == 281) || (px == 469 && py == 463) ||
    (px == 140 && py == 6) || (px == 170 && py == 42) ||
    (px == 181 && py == 96) || (px == 18 && py == 163) ||
    (px == 312 && py == 183) || (px == 366 && py == 262) ||
    (px == 249 && py == 317) || (px == 305 && py == 335) ||
    (px == 268 && py == 364) || (px == 0 && py == 375) ||
    (px == 197 && py == 401) || (px == 11 && py == 419) ||
    (px == 70 && py == 424) || (px == 71 && py == 424) ||
    (px == 74 && py == 424) || (px == 75 && py == 424) ||
    (px == 229 && py == 425) || (px == 174 && py == 445) ||
    (px == 435 && py == 480) || (px == 397 && py == 110);
}

inline bool analyticAnchorTexcoord(float2 screenPos, float3 interpTex,
    constant VolumeMapperUniforms& volumeUniforms,
    constant float* triAnchor, uint primId, thread float3& anchorTex)
{
  uint triCount = uint(triAnchor[0]);
  if (primId >= triCount) { return false; }
  constant float* rec = triAnchor + 4 + (uint)primId * 21u;
  float wWin = volumeUniforms.viewportSize.x;
  float hWin = volumeUniforms.viewportSize.y;
#if defined(VTK_METAL_ENABLE_LOGGING)
  {
    int px = int(floor(screenPos.x - 0.5));
    int py = int(floor(screenPos.y - 0.5));
    bool gate = analyticPixelGate(screenPos);
    if (gate)
    {
      os_log_default.log_info("VTK_METAL_VOLUME_LOG DEBUG ANALYTIC_IN px=(%d, %d) primId=%u interp=(%.9e, %.9e, %.9e) rec=[%.9e, %.9e, %.9e, %.9e, %.9e, %.9e, %.9e, %.9e, %.9e, %.9e, %.9e, %.9e, %.9e, %.9e, %.9e, %.9e, %.9e, %.9e, %.9e, %.9e, %.9e]",
        px, py, primId,
        interpTex.x, interpTex.y, interpTex.z,
        rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], rec[6],
        rec[7], rec[8], rec[9], rec[10], rec[11], rec[12], rec[13],
        rec[14], rec[15], rec[16], rec[17], rec[18], rec[19], rec[20]);
    }
  }
#endif
  if (volumeUniforms.analyticAnchorMode < 1.5)
  {
    float v0x = (rec[0] / rec[3] + 1.0f) * 0.5f * wWin;
    float v0y = (1.0f - rec[1] / rec[3]) * 0.5f * hWin;
    float v1x = (rec[7] / rec[10] + 1.0f) * 0.5f * wWin;
    float v1y = (1.0f - rec[8] / rec[10]) * 0.5f * hWin;
    float v2x = (rec[14] / rec[17] + 1.0f) * 0.5f * wWin;
    float v2y = (1.0f - rec[15] / rec[17]) * 0.5f * hWin;
    float px = screenPos.x, py = screenPos.y;
    float det = (v1x - v0x) * (v2y - v0y) - (v1y - v0y) * (v2x - v0x);
    float l0 = ((v1x - px) * (v2y - py) - (v1y - py) * (v2x - px)) / det;
    float l1 = ((v2x - px) * (v0y - py) - (v2y - py) * (v0x - px)) / det;
    float l2 = 1.0f - l0 - l1;
    float den = l0 / rec[3] + l1 / rec[10] + l2 / rec[17];
    anchorTex.x = (l0 * (rec[4] / rec[3]) + l1 * (rec[11] / rec[10]) + l2 * (rec[18] / rec[17])) / den;
    anchorTex.y = (l0 * (rec[5] / rec[3]) + l1 * (rec[12] / rec[10]) + l2 * (rec[19] / rec[17])) / den;
    anchorTex.z = (l0 * (rec[6] / rec[3]) + l1 * (rec[13] / rec[10]) + l2 * (rec[20] / rec[17])) / den;
#if defined(VTK_METAL_ENABLE_LOGGING)
    {
      int opx = int(floor(screenPos.x - 0.5));
      int opy = int(floor(screenPos.y - 0.5));
      bool ogate = (opx == 397 && opy == 110) || (opx == 256 && opy == 256) ||
        (opx == 422 && opy == 92) || (opx == 372 && opy == 131) ||
        (opx == 360 && opy == 229) || (opx == 349 && opy == 255) ||
        (opx == 405 && opy == 171) || (opx == 9 && opy == 18) ||
        (opx == 293 && opy == 298) || (opx == 338 && opy == 432) ||
        (opx % 32 == 0 && opy % 32 == 0);
      if (ogate)
      {
        os_log_default.log_info("VTK_METAL_VOLUME_LOG DEBUG ANALYTIC_OUT32 px=(%d, %d) anchor=(%.9e, %.9e, %.9e)",
          opx, opy, anchorTex.x, anchorTex.y, anchorTex.z);
      }
    }
#endif
    return true;
  }
  else
  {
    // float64-equivalent weights (update 76 sect 4): same recipe as the float32
    // branch but every intermediate is a double-float, so the vertex NDC divide
    // and the barycentric weights carry no float32 rounding.
    DoubleFloat px = dfFromFloat(screenPos.x);
    DoubleFloat py = dfFromFloat(screenPos.y);
    DoubleFloat hscale = dfFromFloat(0.5f);
    DoubleFloat wWinD = dfFromFloat(wWin);
    DoubleFloat hWinD = dfFromFloat(hWin);
    DoubleFloat one = dfFromFloat(1.0f);
    DoubleFloat v0x = dfMul(dfMul(dfAdd(dfDiv(dfFromFloat(rec[0]), dfFromFloat(rec[3])), one), hscale), wWinD);
    DoubleFloat v0y = dfMul(dfMul(dfSub(one, dfDiv(dfFromFloat(rec[1]), dfFromFloat(rec[3]))), hscale), hWinD);
    DoubleFloat v1x = dfMul(dfMul(dfAdd(dfDiv(dfFromFloat(rec[7]), dfFromFloat(rec[10])), one), hscale), wWinD);
    DoubleFloat v1y = dfMul(dfMul(dfSub(one, dfDiv(dfFromFloat(rec[8]), dfFromFloat(rec[10]))), hscale), hWinD);
    DoubleFloat v2x = dfMul(dfMul(dfAdd(dfDiv(dfFromFloat(rec[14]), dfFromFloat(rec[17])), one), hscale), wWinD);
    DoubleFloat v2y = dfMul(dfMul(dfSub(one, dfDiv(dfFromFloat(rec[15]), dfFromFloat(rec[17]))), hscale), hWinD);
    DoubleFloat det = dfSub(dfMul(dfSub(v1x, v0x), dfSub(v2y, v0y)), dfMul(dfSub(v1y, v0y), dfSub(v2x, v0x)));
    DoubleFloat l0 = dfDiv(dfSub(dfMul(dfSub(v1x, px), dfSub(v2y, py)), dfMul(dfSub(v1y, py), dfSub(v2x, px))), det);
    DoubleFloat l1 = dfDiv(dfSub(dfMul(dfSub(v2x, px), dfSub(v0y, py)), dfMul(dfSub(v2y, py), dfSub(v0x, px))), det);
    DoubleFloat l2 = dfSub(dfSub(one, l0), l1);
    DoubleFloat w0 = dfFromFloat(rec[3]);
    DoubleFloat w1 = dfFromFloat(rec[10]);
    DoubleFloat w2 = dfFromFloat(rec[17]);
    DoubleFloat invW0 = dfDiv(one, w0);
    DoubleFloat invW1 = dfDiv(one, w1);
    DoubleFloat invW2 = dfDiv(one, w2);
    DoubleFloat t0x = dfFromFloat(rec[4]);
    DoubleFloat t0y = dfFromFloat(rec[5]);
    DoubleFloat t0z = dfFromFloat(rec[6]);
    DoubleFloat t1x = dfFromFloat(rec[11]);
    DoubleFloat t1y = dfFromFloat(rec[12]);
    DoubleFloat t1z = dfFromFloat(rec[13]);
    DoubleFloat t2x = dfFromFloat(rec[18]);
    DoubleFloat t2y = dfFromFloat(rec[19]);
    DoubleFloat t2z = dfFromFloat(rec[20]);
    DoubleFloat den = dfAdd(dfAdd(dfMul(l0, invW0), dfMul(l1, invW1)), dfMul(l2, invW2));
    DoubleFloat numX = dfAdd(dfAdd(dfMul(l0, dfDiv(t0x, w0)), dfMul(l1, dfDiv(t1x, w1))), dfMul(l2, dfDiv(t2x, w2)));
    DoubleFloat numY = dfAdd(dfAdd(dfMul(l0, dfDiv(t0y, w0)), dfMul(l1, dfDiv(t1y, w1))), dfMul(l2, dfDiv(t2y, w2)));
    DoubleFloat numZ = dfAdd(dfAdd(dfMul(l0, dfDiv(t0z, w0)), dfMul(l1, dfDiv(t1z, w1))), dfMul(l2, dfDiv(t2z, w2)));
    anchorTex.x = dfToFloat(dfDiv(numX, den));
    anchorTex.y = dfToFloat(dfDiv(numY, den));
    anchorTex.z = dfToFloat(dfDiv(numZ, den));
#if defined(VTK_METAL_ENABLE_LOGGING)
    {
      int opx = int(floor(screenPos.x - 0.5));
      int opy = int(floor(screenPos.y - 0.5));
      bool ogate = (opx == 397 && opy == 110) || (opx == 256 && opy == 256) ||
        (opx == 422 && opy == 92) || (opx == 372 && opy == 131) ||
        (opx == 360 && opy == 229) || (opx == 349 && opy == 255) ||
        (opx == 405 && opy == 171) || (opx == 9 && opy == 18) ||
        (opx == 293 && opy == 298) || (opx == 338 && opy == 432) ||
        (opx % 32 == 0 && opy % 32 == 0);
      if (ogate)
      {
        os_log_default.log_info("VTK_METAL_VOLUME_LOG DEBUG ANALYTIC_OUT64 px=(%d, %d) anchor=(%.9e, %.9e, %.9e)",
          opx, opy, anchorTex.x, anchorTex.y, anchorTex.z);
      }
    }
#endif
    return true;
  }
}

fragment VolumeFragmentOut fragment_volume_main(
    VolumeVertexOut in [[stage_in]],
    uint primId [[primitive_id]],
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
    texture2d<float> transferFunction2DTexture [[texture(9)]],
    texture3d<float> transfer2DYAxisTexture [[texture(10)]],
    texture3d<float> blankingTexture [[texture(11)]],
    texture2d<float> transferFunctionTexture1 [[texture(12)]],
    texture2d<float> transferFunctionTexture2 [[texture(13)]],
    texture2d<float> transferFunctionTexture3 [[texture(14)]],
    constant packed_float3* rectCoords [[buffer(5)]],
    constant VolumeLightUniforms& volumeLights [[buffer(4)]],
    constant float* triAnchor [[buffer(3)]]) {

  VolumeFragmentOut output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  // Camera-inside proxy (useCameraInsideNearClip set): the vertex buffer holds
  // data-space positions (GL parity) so the interpolated in.localPos is a
  // dataset-space anchor. Everything downstream (ray direction, near-clip clamp,
  // marching) uses volume-space [0,1] coordinates, so convert here; the raw
  // data-space anchor is forwarded to the march for the OpenGL g_dirStep parity.
  bool cameraInsideProxy = volumeUniforms.useCameraInsideNearClip > 0.5;
  float3 anchorData = in.localPos;
  float3 localPos = in.localPos;
  float3 anchorTex = in.localPos;
  if (cameraInsideProxy)
  {
    float3 bsz = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
    localPos = (in.localPos - volumeUniforms.volumeBoundsMin.xyz) / bsz;
    // OpenGL ip_textureCoords parity: GL's vertex shader maps each proxy vertex
    // through in_inverseTextureDatasetMatrix * pos and then the cell-to-point
    // texel adjustment (scale + offset), and the rasterizer interpolates that
    // per-vertex float result. Use the interpolated per-vertex texcoord so the
    // ray anchor is bit-identical to GL (a fragment-time affine of the
    // interpolated data position only reproduced the systematic shift).
    // TEMP DEBUG (analytic-anchor experiment): when analyticAnchorMode is set,
    // bypass the interpolator and reconstruct the per-fragment texcoord from
    // pixel-center barycentrics + per-vertex clip/texcoord (buffer 3).
    if (volumeUniforms.analyticAnchorMode > 0.5)
    {
      if (!analyticAnchorTexcoord(in.position.xy, in.texcoord, volumeUniforms, triAnchor, primId, anchorTex))
      {
        anchorTex = in.texcoord;
      }
    }
    else
    {
      anchorTex = in.texcoord;
    }
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
  // Debug only (test builds): one message per frame, gated to the center
  // pixel. Verified by TestMetalVolumeShaderLog.
#if defined(VTK_METAL_ENABLE_LOGGING)
  if (all(abs(in.position.xy - 0.5f * volumeUniforms.viewportSize) < 1.0f))
  {
    os_log_default.log_info("VTK_METAL_VOLUME_LOG fragment_volume_main center camera=(%f, %f, %f)",
      cameraPos.x, cameraPos.y, cameraPos.z);
  }
#endif

  RaySetup s = setupVolumeRay(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal,
      in.position.xy, volumeUniforms.viewportSize, volumeUniforms, depthTexture);
  if (!s.valid) { output.color = float4(0.0); return output; }

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float4 _marchResult = marchVolume(s.entryPoint, s.exitPoint, s.totalDist, s.tTerminateMax, rayDir,
      blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal, rayOrigin,
      stepSize, s.totalBoxT, in.position.xy, anchorTex, in.clipPos, anchorData, cameraInsideProxy,
      in.flatVid, primId,
      float3(0.0), 0.0f, volumeUniforms, b,
      volumeTexture, transferFunctionTexture, transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3,
      transferFunction2DTexture, transfer2DYAxisTexture,
      depthTexture, gradientOpacityTexture,
      maskTexture, labelMapTransferTexture, minMaxTexture, normalTexture,
      blankingTexture, rectCoords, &volumeLights);
  output.color = _marchResult;
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
    uint primId [[primitive_id]],
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
    texture2d<float> transferFunction2DTexture [[texture(9)]],
    texture3d<float> transfer2DYAxisTexture [[texture(10)]],
    texture3d<float> blankingTexture [[texture(11)]],
    texture2d<float> transferFunctionTexture1 [[texture(12)]],
    texture2d<float> transferFunctionTexture2 [[texture(13)]],
    texture2d<float> transferFunctionTexture3 [[texture(14)]],
    constant packed_float3* rectCoords [[buffer(5)]],
    constant VolumeLightUniforms& volumeLights [[buffer(4)]],
    constant float* triAnchor [[buffer(3)]]) {

  VolumeSelectionOut output;
  output.color = float4(0.0);
  output.ids = uint4(0u);
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  // Camera-inside proxy (useCameraInsideNearClip set): the vertex buffer holds
  // data-space positions (GL parity) so the interpolated in.localPos is a
  // dataset-space anchor. Everything downstream (ray direction, near-clip clamp,
  // marching) uses volume-space [0,1] coordinates, so convert here; the raw
  // data-space anchor is forwarded to the march for the OpenGL g_dirStep parity.
  bool cameraInsideProxy = volumeUniforms.useCameraInsideNearClip > 0.5;
  float3 anchorData = in.localPos;
  float3 localPos = in.localPos;
  float3 anchorTex = in.localPos;
  if (cameraInsideProxy)
  {
    float3 bsz = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
    localPos = (in.localPos - volumeUniforms.volumeBoundsMin.xyz) / bsz;
    // OpenGL ip_textureCoords parity: GL's vertex shader maps each proxy vertex
    // through in_inverseTextureDatasetMatrix * pos and then the cell-to-point
    // texel adjustment (scale + offset), and the rasterizer interpolates that
    // per-vertex float result. Use the interpolated per-vertex texcoord so the
    // ray anchor is bit-identical to GL (a fragment-time affine of the
    // interpolated data position only reproduced the systematic shift).
    // TEMP DEBUG (analytic-anchor experiment): when analyticAnchorMode is set,
    // bypass the interpolator and reconstruct the per-fragment texcoord from
    // pixel-center barycentrics + per-vertex clip/texcoord (buffer 3).
    if (volumeUniforms.analyticAnchorMode > 0.5)
    {
      if (!analyticAnchorTexcoord(in.position.xy, in.texcoord, volumeUniforms, triAnchor, primId, anchorTex))
      {
        anchorTex = in.texcoord;
      }
    }
    else
    {
      anchorTex = in.texcoord;
    }
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
  if (!s.valid) { output.color = float4(0.0); return output; }

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float4 _marchResult = marchVolume(s.entryPoint, s.exitPoint, s.totalDist, s.tTerminateMax, rayDir,
      blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal, rayOrigin,
      stepSize, s.totalBoxT, in.position.xy, anchorTex, in.clipPos, anchorData, cameraInsideProxy,
      in.flatVid, primId,
      float3(0.0), 0.0f, volumeUniforms, b,
      volumeTexture, transferFunctionTexture, transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3,
      transferFunction2DTexture, transfer2DYAxisTexture,
      depthTexture, gradientOpacityTexture,
      maskTexture, labelMapTransferTexture, minMaxTexture, normalTexture,
      blankingTexture, rectCoords, &volumeLights);

  // PickingActorPassExit parity: only fragments that accumulated a certain
  // level of opacity receive a picking id (index 0 is reserved for empty space).
  // PickingActorPassExit parity: only fragments that accumulated a certain
  // level of opacity receive a picking id (index 0 is reserved for empty space).
  output.ids = volumeSelectionIds(s.entryPoint, float(_marchResult.w), volumeUniforms);
  output.color = _marchResult;
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
  // Debug only (test builds): one message per frame, gated to the center
  // pixel. Verified by TestMetalVolumeShaderLog.
#if defined(VTK_METAL_ENABLE_LOGGING)
  if (all(abs(in.position.xy - 0.5f * volumeUniforms.viewportSize) < 1.0f))
  {
    os_log_default.log_info("VTK_METAL_VOLUME_LOG fragment_volume_fullscreen_main center camera=(%f, %f, %f)",
      cameraPos.x, cameraPos.y, cameraPos.z);
  }
#endif

  RaySetup s = setupVolumeRay(rayOrigin, rayDir, blockMinGlobal, blockMaxGlobal,
      in.position.xy, volumeUniforms.viewportSize, volumeUniforms, depthTexture);
  if (!s.valid) { output.color = float4(0.0); return output; }

  float stepSize = physicalSampleStep(rayDir, volumeUniforms);
  float4 _marchResult = marchVolume(s.entryPoint, s.exitPoint, s.totalDist, s.tTerminateMax, rayDir,
      blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal, rayOrigin,
      stepSize, s.totalBoxT, in.position.xy, s.entryPoint, in.clipPos, s.entryPoint, false,
      0, 0,
      float3(0.0), 0.0f, volumeUniforms, b,
      volumeTexture, transferFunctionTexture, transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3,
      transferFunction2DTexture, transfer2DYAxisTexture,
      depthTexture, gradientOpacityTexture,
      maskTexture, labelMapTransferTexture, minMaxTexture, normalTexture,
      blankingTexture, rectCoords, &volumeLights);
  output.color = _marchResult;
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
  float4 _marchResult = marchVolume(s.entryPoint, s.exitPoint, s.totalDist, s.tTerminateMax, rayDir,
      blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal, rayOrigin,
      stepSize, s.totalBoxT, in.position.xy, s.entryPoint, in.clipPos, s.entryPoint, false,
      0, 0,
      float3(0.0), 0.0f, volumeUniforms, b,
      volumeTexture, transferFunctionTexture, transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3,
      transferFunction2DTexture, transfer2DYAxisTexture,
      depthTexture, gradientOpacityTexture,
      maskTexture, labelMapTransferTexture, minMaxTexture, normalTexture,
      blankingTexture, rectCoords, &volumeLights);

  output.ids = volumeSelectionIds(s.entryPoint, float(_marchResult.w), volumeUniforms);
  output.color = _marchResult;
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
    uint primId [[primitive_id]],
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
    texture2d<float> transferFunction2DTexture [[texture(9)]],
    texture3d<float> transfer2DYAxisTexture [[texture(10)]],
    texture3d<float> blankingTexture [[texture(11)]],
    texture2d<float> transferFunctionTexture1 [[texture(12)]],
    texture2d<float> transferFunctionTexture2 [[texture(13)]],
    texture2d<float> transferFunctionTexture3 [[texture(14)]],
    constant packed_float3* rectCoords [[buffer(5)]],
    constant VolumeLightUniforms& volumeLights [[buffer(4)]],
    constant float* triAnchor [[buffer(3)]]) {

  VolumeFragmentOutRTT output;
  float3 cameraPos = volumeUniforms.cameraVolumePos.xyz;
  float3 blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal;
  computeVolumeBounds(b, volumeUniforms, blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal);

  // Camera-inside proxy (useCameraInsideNearClip set): the vertex buffer holds
  // data-space positions (GL parity) so the interpolated in.localPos is a
  // dataset-space anchor. Everything downstream (ray direction, near-clip clamp,
  // marching) uses volume-space [0,1] coordinates, so convert here; the raw
  // data-space anchor is forwarded to the march for the OpenGL g_dirStep parity.
  bool cameraInsideProxy = volumeUniforms.useCameraInsideNearClip > 0.5;
  float3 anchorData = in.localPos;
  float3 localPos = in.localPos;
  float3 anchorTex = in.localPos;
  if (cameraInsideProxy)
  {
    float3 bsz = max(volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz, 1e-6);
    localPos = (in.localPos - volumeUniforms.volumeBoundsMin.xyz) / bsz;
    // OpenGL ip_textureCoords parity: GL's vertex shader maps each proxy vertex
    // through in_inverseTextureDatasetMatrix * pos and then the cell-to-point
    // texel adjustment (scale + offset), and the rasterizer interpolates that
    // per-vertex float result. Use the interpolated per-vertex texcoord so the
    // ray anchor is bit-identical to GL (a fragment-time affine of the
    // interpolated data position only reproduced the systematic shift).
    // TEMP DEBUG (analytic-anchor experiment): when analyticAnchorMode is set,
    // bypass the interpolator and reconstruct the per-fragment texcoord from
    // pixel-center barycentrics + per-vertex clip/texcoord (buffer 3).
    if (volumeUniforms.analyticAnchorMode > 0.5)
    {
      if (!analyticAnchorTexcoord(in.position.xy, in.texcoord, volumeUniforms, triAnchor, primId, anchorTex))
      {
        anchorTex = in.texcoord;
      }
    }
    else
    {
      anchorTex = in.texcoord;
    }
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
  float jitter = (volumeUniforms.useJittering > 0.5 ? volume_random(in.position.xy) : 1.0) * stepSize;
  float tStart = parallel ? 0.0 : dot(s.entryPoint - cameraPos, rayDir);
  MarchParams p = {rayOrigin, rayDir, tStart, s.totalBoxT, stepSize, jitter, s.tTerminateMax,
      blockMinGlobal, blockMaxGlobal, texMinGlobal, texMaxGlobal, in.position.xy, anchorTex,
      in.clipPos, true, anchorData, cameraInsideProxy};

  float3 firstOpaquePos = float3(-1.0);
  bool searching = true;
  if (volumeUniforms.clampDepthToBackface > 0.5) {
    firstOpaquePos = s.entryPoint;
  }

  float4 _marchResult = marchVolumeUnified(p, float3(0.0), 0.0f,
      volumeUniforms, b, volumeTexture, transferFunctionTexture,
      transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3,
      transferFunction2DTexture, transfer2DYAxisTexture,
      gradientOpacityTexture, maskTexture, labelMapTransferTexture,
      minMaxTexture, normalTexture, blankingTexture, rectCoords, &volumeLights,
      &firstOpaquePos, &searching);

  output.color = _marchResult;

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

    // Global sample schedule
    float stepSize = physicalSampleStep(rayDir, volumeUniforms);
    float jitter = volumeUniforms.useJittering > 0.5
        ? volume_random(in.position.xy + float2(0.5, 0.5)) * stepSize
        : 1.0 * stepSize;

    // Grid traversal loop
    float3 color = 0.0f;
    float opacity = 0.0f;

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
                    in.position.xy,
                    color, opacity,
                    volumeUniforms, b,
                    volumeTexture, transferFunctionTexture,
                    transferFunctionTexture1, transferFunctionTexture2, transferFunctionTexture3,
                    transferFunction2DTexture, transfer2DYAxisTexture,
                    gradientOpacityTexture, maskTexture, labelMapTransferTexture,
                    minMaxTexture, normalTexture, blankingTexture, rectCoords,
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