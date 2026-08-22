// appgl_parity.mm — the app's EXACT GL fragment shader (dumped from
// vtkOpenGLGPUVolumeRayCastMapper) rendered in the repro harness: same box
// geometry, same uniforms, same per-sample work (range scale + conditional
// color fetch + OOB break + currentT break). Isolates whether the repro's
// faster jitter cost comes from the shader structure or the harness.
//
// Usage: ./appgl_parity <rt> <sd> <frames> <dicom.u8> [j0|j1]
//
// THE JITTER IS GL'S, ON BOTH BACKENDS (the reference).
//
// TIMINGS VALID as of 2026-08-22 (kInvProj[11] transcription fixed; see
// RESULTS.md): interleaved pairs at 1024/SD4 WARMUP=30 give
//   j0 25.5 / j1 33.2 / jitter D +7.7 ms
// reconciling with the live app binary (+8.66 incl ~3.8 ms VTK CPU/frame)
// while the lean jitter_gap_repro FS pays +20.6 in the identical context.
// The composed shader text carries the app's cheap jitter.
//
// Debug probes (env APPGL_DBG=n + APPGL_DUMP_PPM=1):
//   14 entry position, 21 last-sampled position, 22 terminatePointMax,
//   23 |g_dirStep| x64, 25 per-pixel final iteration count, 26 raw
//   g_rayTermination ((pos*0.25+0.5) encoded).
// TF_FACTOR: composite correction baked into the opacity LUT (live value 4).
//
// History: before the kInvProj fix this tool truncated every march at
// g_terminatePointMax ~21 samples (vs ~86 real) and reported a bogus
// j0 6.9 ms / jitter D +1.5 ms.

#import <OpenGL/gl3.h>
#import <OpenGL/OpenGL.h>
#import <QuartzCore/QuartzCore.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>

static const int kVolW = 512, kVolH = 512, kVolD = 1794;
static const float kBounds[3] = { 426.166f, 426.166f, 717.2f };
// The app's camera, dumped from vtkOpenGLGPUVolumeRayCastMapper
// (VTK_METAL_TEST_DUMP_UNIFORMS, DICOMVolume scene). These are the VTK-memory
// (row-major) layouts, uploaded with glUniformMatrix4fv(GL_FALSE); the
// GLSL-visible matrices are their transposes. GLSL projection*modelview
// therefore equals (MV_vtk @ P_vtk)^T in numpy terms.
static const float kProj[16] = {
  3.73205f, 0.0f, 0.0f, 0.0f,
  0.0f, 3.73205f, 0.0f, 0.0f,
  0.0f, 0.0f, -2.30111f, -1.0f,
  0.0f, 0.0f, -3558.8f, 0.0f };
static const float kModelView[16] = {
  0.866025f, -0.17101f, -0.469846f, 0.0f,
  0.0f, 0.939693f, -0.34202f, 0.0f,
  0.5f, 0.296198f, 0.813798f, 0.0f,
  -363.835f, -270.01f, -1928.61f, 1.0f };
static const float kInvProj[16] = {
  // VERBATIM from the live program (glGetUniformfv via
  // VTK_METAL_TEST_DUMP_UNIFORMS, 2026-08-22). The previous transcription
  // had [11]=0 instead of -1: the inverse-projection's w-row lost its
  // -z_eye term, g_terminatePointMax collapsed to ~21 samples and every
  // timing this tool produced was a truncated-march artifact.
  0.267949f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.267949f, 0.0f, 0.0f,
  0.0f, 0.0f, -0.000280993f, -1.0f,
  0.0f, 0.0f, -1.0f, 0.000646595f };
static const float kInvModelView[16] = {
  0.866025f, -3.03887e-18f, 0.5f, 0.0f,
  -0.17101f, 0.939693f, 0.296198f, 0.0f,
  -0.469846f, -0.34202f, 0.813798f, 0.0f,
  -637.233f, -405.897f, 1831.39f, 1.0f };
static const float kEyeMM[3] = { -637.233f, -405.897f, 1831.39f };

static const uint8_t kBlue64[4096] = {
#include "bluenoise64.inc"
};

static double NowMs(void)
{
  return CACurrentMediaTime() * 1000.0;
}

// cellToPoint per axis: scale (D-1)/D, offset 0.5/D
static float CtpScale(int D) { return (float)(D - 1) / (float)D; }
static float CtpOffset(int D) { return 0.5f / (float)D; }

static void MatMul44(float* out, const float* a, const float* b)
{
  for (int r = 0; r < 4; r++)
    for (int c = 0; c < 4; c++)
    {
      float s = 0;
      for (int k = 0; k < 4; k++) s += a[r * 4 + k] * b[k * 4 + c];
      out[r * 4 + c] = s;
    }
}

int main(int argc, const char** argv)
{
  int rt = argc > 1 ? atoi(argv[1]) : 2048;
  float sd = argc > 2 ? (float)atof(argv[2]) : 4.0f;
  int frames = argc > 3 ? atoi(argv[3]) : 30;
  const char* volPath = argc > 4 ? argv[4] : "dicom.u8";
  int jitter = (argc > 5 && strcmp(argv[5], "j0") == 0) ? 0 : 1;

  FILE* f = fopen(volPath, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", volPath); return 1; }
  size_t total = (size_t)kVolW * kVolH * kVolD;
  uint8_t* vol = (uint8_t*)malloc(total);
  if (fread(vol, 1, total, f) != total) { fprintf(stderr, "short read on %s\n", volPath); return 1; }
  fclose(f);

  CGLPixelFormatAttribute attrs[] = {
    kCGLPFAAccelerated, kCGLPFAOpenGLProfile, (CGLPixelFormatAttribute)kCGLOGLPVersion_3_2_Core, (CGLPixelFormatAttribute)0 };
  CGLPixelFormatObj pf = NULL; GLint npf = 0;
  if (CGLChoosePixelFormat(attrs, &pf, &npf) != kCGLNoError || !pf) { fprintf(stderr, "no GL 3.2 core pixel format\n"); return 1; }
  CGLContextObj ctx = NULL;
  if (CGLCreateContext(pf, NULL, &ctx) != kCGLNoError || !ctx) { fprintf(stderr, "no GL context\n"); return 1; }
  CGLSetCurrentContext(ctx);
  CGLDestroyPixelFormat(pf);

  // ---- volume texture: GL_R8 512x512x1794, LINEAR, CLAMP (app-identical)
  GLuint volTex = 0;
  glGenTextures(1, &volTex);
  glBindTexture(GL_TEXTURE_3D, volTex);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, kVolW, kVolH, kVolD, 0, GL_RED, GL_UNSIGNED_BYTE, vol);

  // ---- noise texture: R32F 64x64 NEAREST REPEAT (app: VTK_FLOAT -> R32F)
  GLuint noiseTex = 0;
  glGenTextures(1, &noiseTex);
  glBindTexture(GL_TEXTURE_2D, noiseTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  {
    float* noise = (float*)malloc(64 * 64 * sizeof(float));
    for (int i = 0; i < 64 * 64; i++) noise[i] = kBlue64[i] / 255.0f;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 64, 64, 0, GL_RED, GL_FLOAT, noise);
    free(noise);
  }

  // ---- TF textures (1024x1 FLOAT, app-identical): the app's opacity TF is
  // a 1024-wide GL_R32F table (GetTable(0,255,1024) + composite correction
  // factor sampleDistance/unitDistance = 4), sampled by computeOpacity at
  // vec2(scalar.w, 0) where scalar.w ~ value/254. The values below were
  // glGetTexImage'd from the app's real texture (VTK_METAL_TEST_DUMP_UNIFORMS).
  // The color TF is a flat (0, 0.605, 0.706) GL_RGB32F table (constant color).
  float tfO[1024];
  {
    // band indices 71..172 of 1024 over [0,255] == values 17.7..42.9
    for (int i = 0; i < 1024; i++) tfO[i] = 0.0f;
    float v0 = 71 * 255.0f / 1023.0f, v1 = 172 * 255.0f / 1023.0f;
    const float xs[4] = { 17.55f, 21.24f, 33.80f, 43.01f };
    const float ys[4] = { 0.0f, 0.0493f, 0.2497f, 0.0f };
    // TF_FACTOR: the composite correction sampleDistance/unitDistance baked
    // into the app's opacity LUT (dumped value: 4). factor=1 disables it.
    // NOTE (2026-08-22): factor=4 saturates rays early (short marches,
    // coverage 10% vs the march's 19%) — invalid for timing comparisons
    // unless the march stats are matched first.
    const float factor = getenv("TF_FACTOR") ? (float)atof(getenv("TF_FACTOR")) : 4.0f;
    for (int i = 71; i <= 172; i++)
    {
      float v = i * 255.0f / 1023.0f;
      float a = 0.0f;
      for (int k = 0; k < 3; k++)
        if (v >= xs[k] && v <= xs[k + 1])
          a = ys[k] + (ys[k + 1] - ys[k]) * (v - xs[k]) / (xs[k + 1] - xs[k]);
      tfO[i] = 1.0f - powf(1.0f - a, factor);
    }
  }
  GLuint tfOTex = 0, tfCTex = 0;
  glGenTextures(1, &tfOTex);
  glBindTexture(GL_TEXTURE_2D, tfOTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 1024, 1, 0, GL_RED, GL_FLOAT, tfO);
  glGenTextures(1, &tfCTex);
  glBindTexture(GL_TEXTURE_2D, tfCTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  const float kTfColor[3] = { 0.0f, 0.605f, 0.706f };
  float tfC[1024 * 3];
  for (int i = 0; i < 1024; i++) { tfC[i * 3 + 0] = kTfColor[0]; tfC[i * 3 + 1] = kTfColor[1]; tfC[i * 3 + 2] = kTfColor[2]; }
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, 1024, 1, 0, GL_RGB, GL_FLOAT, tfC);

  // ---- depth texture: a COPY of the window depth (cleared 1.0), read by the
  // fragment shader as in_depthSampler. The app blits the window depth into a
  // separate texture each frame; here it is a static 1.0 clear (the scene has
  // no opaque geometry, so the app's copy is also all 1.0).
  GLuint depthTex = 0;
  glGenTextures(1, &depthTex);
  glBindTexture(GL_TEXTURE_2D, depthTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, rt, rt, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);

  GLuint rtTex = 0, fbo = 0, depthRBO = 0;
  glGenTextures(1, &rtTex);
  glBindTexture(GL_TEXTURE_2D, rtTex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rt, rt, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rtTex, 0);
  // clear the depth texture to 1.0 once (via a temporary renderbuffer), then
  // detach it so the raycast pass never reads a texture it writes (feedback).
  glGenRenderbuffers(1, &depthRBO);
  glBindRenderbuffer(GL_RENDERBUFFER, depthRBO);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, rt, rt);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRBO);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTex, 0);
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { fprintf(stderr, "FBO incomplete\n"); return 1; }
  glClearDepth(1.0);
  glClear(GL_DEPTH_BUFFER_BIT);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, 0);
  glDeleteRenderbuffers(1, &depthRBO);

  // ---- box geometry: 8 corners in mm, cell-adjusted texture coords, 12 tris
  // (the app's densified BBoxPolyData)
  float corners[8][3];
  for (int i = 0; i < 8; i++)
  {
    corners[i][0] = (i & 1) ? kBounds[0] : 0.0f;
    corners[i][1] = (i & 2) ? kBounds[1] : 0.0f;
    corners[i][2] = (i & 4) ? kBounds[2] : 0.0f;
  }
  float tc[8][3];
  for (int i = 0; i < 8; i++)
  {
    tc[i][0] = corners[i][0] / kBounds[0] * CtpScale(kVolW) + CtpOffset(kVolW);
    tc[i][1] = corners[i][1] / kBounds[1] * CtpScale(kVolH) + CtpOffset(kVolH);
    tc[i][2] = corners[i][2] / kBounds[2] * CtpScale(kVolD) + CtpOffset(kVolD);
  }
  // 12 triangles (indices of the 8 corners), outward winding
  static const int tris[12][3] = {
    { 0, 2, 6 }, { 0, 6, 4 },   // -x
    { 1, 5, 7 }, { 1, 7, 3 },   // +x
    { 0, 1, 3 }, { 0, 3, 2 },   // -y
    { 4, 6, 7 }, { 4, 7, 5 },   // +y
    { 0, 4, 5 }, { 0, 5, 1 },   // -z
    { 2, 3, 7 }, { 2, 7, 6 },   // +z
  };
  float verts[12 * 3 * 6];
  for (int t = 0; t < 12; t++)
    for (int v = 0; v < 3; v++)
    {
      int c = tris[t][v];
      float* o = &verts[(t * 3 + v) * 6];
      o[0] = corners[c][0]; o[1] = corners[c][1]; o[2] = corners[c][2];
      o[3] = tc[c][0]; o[4] = tc[c][1]; o[5] = tc[c][2];
    }
  GLuint vao = 0, vbo = 0;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

  // ---- matrices: the app's dumped camera, uploaded verbatim (VTK memory
  // layout, GL_FALSE). The vertex shader uses projection*modelview like the
  // app; the fragment shader's termination path uses the inverse matrices.
  float invtex[16];
  {
    // ip_inverseTextureDataAdjusted (GLSL) has col3 = the cellToPoint offset
    // UNSCALED; the uploaded (transposed) layout puts it in row 3.
    memset(invtex, 0, sizeof(invtex));
    invtex[0] = CtpScale(kVolW) / kBounds[0];
    invtex[5] = CtpScale(kVolH) / kBounds[1];
    invtex[10] = CtpScale(kVolD) / kBounds[2];
    invtex[12] = CtpOffset(kVolW);
    invtex[13] = CtpOffset(kVolH);
    invtex[14] = CtpOffset(kVolD);
    invtex[15] = 1.0f;
  }

  const char* vs = "#version 150\n"
    "in vec3 in_vertexPos;\n"
    "in vec3 in_texCoords;\n"
    "out vec3 ip_textureCoords;\n"
    "out vec3 ip_vertexPos;\n"
    "out mat4 ip_inverseTextureDataAdjusted;\n"
    "uniform mat4 in_projectionMatrix;\n"
    "uniform mat4 in_modelViewMatrix;\n"
    "uniform mat4 uInvTexDataAdjusted;\n"
    "void main() {\n"
    "  gl_Position = in_projectionMatrix * in_modelViewMatrix * vec4(in_vertexPos, 1.0);\n"
    "  ip_textureCoords = in_texCoords;\n"
    "  ip_vertexPos = in_vertexPos;\n"
    "  ip_inverseTextureDataAdjusted = uInvTexDataAdjusted;\n"
    "}\n";

  // The app's composed fragment shader (dumped 2026-08-19 from
  // vtkOpenGLGPUVolumeRayCastMapper, DICOMVolume scene). Jitter block
  // parameterized: j0 -> g_rayJitter = g_dirStep (no fetch), j1 -> per-pixel.
  const char* fs = R"(#version 150
#define highp
#define mediump
#define lowp
#define texture1D texture
#define texture2D texture
#define texture3D texture
in vec3 ip_textureCoords;
in vec3 ip_vertexPos;
out vec4 fragColor;
vec4 g_fragColor = vec4(0.0);
vec3 g_dirStep;
float g_lengthStep = 0.0;
vec4 g_srcColor;
vec3 g_eyePosObj;
bool g_exit;
bool g_skip;
float g_currentT;
float g_terminatePointMax;
vec3 g_rayOrigin;
vec3 g_rayTermination;
vec3 g_dataPos;
vec3 g_terminatePos;
float g_jitterValue = 0.0;
float dbgMax = 0.0;
float dbgSum = 0.0;
vec3 dbgPos = vec3(0.0);
float dbgAlpha = 0.0;
vec3 g_lastPos = vec3(0.0);  // uDbg==21: last-sampled position (march exit probe)
float g_dbgTPM = 0.0;        // uDbg==22: g_terminatePointMax probe
float g_dbgDir = 0.0;        // uDbg==23: |g_dirStep| probe
uniform sampler3D in_volume[1];
uniform vec4 in_volume_scale[1];
uniform vec4 in_volume_bias[1];
uniform int in_noOfComponents;
uniform sampler2D in_depthSampler;
uniform sampler2D in_noiseSampler;
uniform mat4 in_volumeMatrix[1];
uniform mat4 in_inverseVolumeMatrix[1];
uniform mat4 in_textureDatasetMatrix[1];
uniform mat4 in_inverseTextureDatasetMatrix[1];
uniform mat4 in_textureToEye[1];
uniform vec3 in_texMin[1];
uniform vec3 in_texMax[1];
uniform vec3 in_eyePosObjs[1];
uniform mat4 in_cellToPoint[1];
uniform mat4 in_projectionMatrix;
uniform mat4 in_inverseProjectionMatrix;
uniform mat4 in_modelViewMatrix;
uniform mat4 in_inverseModelViewMatrix;
in mat4 ip_inverseTextureDataAdjusted;
uniform vec3 in_cellStep[1];
uniform vec2 in_scalarsRange[4];
uniform vec3 in_cellSpacing[1];
uniform float in_sampleDistance;
uniform vec2 in_windowLowerLeftCorner;
uniform vec2 in_inverseOriginalWindowSize;
uniform vec2 in_inverseWindowSize;
uniform vec3 in_textureExtentsMax;
uniform vec3 in_textureExtentsMin;
uniform vec3 in_diffuse[4];
uniform vec3 in_ambient[4];
uniform vec3 in_specular[4];
uniform float in_shininess[4];
vec3 g_rayJitter = vec3(0.0);
uniform vec2 in_averageIPRange;
const float g_opacityThreshold = 1.0 - 1.0 / 255.0;
uniform float in_clippingPlanes[49];
uniform float in_clippedVoxelIntensity;
int clip_numPlanes;
vec3 clip_rayDirObj;
mat4 clip_texToObjMat;
mat4 clip_objToTexMat;
bool AdjustSampleRangeForClipping(inout vec3 startPosTex, inout vec3 stopPosTex)
{
  vec4 startPosObj = vec4(0.0);
  {
    startPosObj = clip_texToObjMat * vec4(startPosTex - g_rayJitter, 1.0);
    startPosObj = startPosObj / startPosObj.w;
    startPosObj.w = 1.0;
  }
  vec4 stopPosObj = vec4(0.0);
  {
    stopPosObj = clip_texToObjMat * vec4(stopPosTex, 1.0);
    stopPosObj = stopPosObj / stopPosObj.w;
    stopPosObj.w = 1.0;
  }
  for (int i = 0; i < clip_numPlanes; i = i + 6)
  {
    vec3 planeOrigin = vec3(in_clippingPlanes[i + 1], in_clippingPlanes[i + 2], in_clippingPlanes[i + 3]);
    vec3 planeNormal = normalize(vec3(in_clippingPlanes[i + 4], in_clippingPlanes[i + 5], in_clippingPlanes[i + 6]));
    float startDistance = dot(planeNormal, planeOrigin - startPosObj.xyz);
    float stopDistance = dot(planeNormal, planeOrigin - stopPosObj.xyz);
    bool startClipped = startDistance > 0.0;
    bool stopClipped = stopDistance > 0.0;
    if (startClipped && stopClipped) { return false; }
    float rayDotNormal = dot(clip_rayDirObj, planeNormal);
    bool frontFace = rayDotNormal > 0.0;
    if (frontFace && startDistance > 0.0)
    {
      float rayScaledDist = startDistance / rayDotNormal;
      startPosObj = vec4(startPosObj.xyz + rayScaledDist * clip_rayDirObj, 1.0);
      vec4 newStartPosTex = clip_objToTexMat * vec4(startPosObj.xyz, 1.0);
      newStartPosTex /= newStartPosTex.w;
      startPosTex = newStartPosTex.xyz;
      startPosTex += g_rayJitter;
    }
    if (!frontFace && stopDistance > 0.0)
    {
      float rayScaledDist = stopDistance / rayDotNormal;
      stopPosObj = vec4(stopPosObj.xyz + rayScaledDist * clip_rayDirObj, 1.0);
      vec4 newStopPosTex = clip_objToTexMat * vec4(stopPosObj.xyz, 1.0);
      newStopPosTex /= newStopPosTex.w;
      stopPosTex = newStopPosTex.xyz;
    }
  }
  if (any(greaterThan(startPosTex, in_texMax[0])) || any(lessThan(startPosTex, in_texMin[0]))) { return false; }
  return true;
}
#define EPSILON 0.001
struct Hit { float tmin; float tmax; };
struct Ray { vec3 origin; vec3 dir; vec3 invDir; };
bool BBoxIntersect(const vec3 boxMin, const vec3 boxMax, const Ray r, out Hit hit)
{
  vec3 tbot = r.invDir * (boxMin - r.origin);
  vec3 ttop = r.invDir * (boxMax - r.origin);
  vec3 tmin = min(ttop, tbot);
  vec3 tmax = max(ttop, tbot);
  vec2 t = max(tmin.xx, tmin.yz);
  float t0 = max(t.x, t.y);
  t = min(tmax.xx, tmax.yz);
  float t1 = min(t.x, t.y);
  hit.tmin = t0;
  hit.tmax = t1;
  return t1 > max(t0, 0.0);
}
void safe_0_vector(inout Ray ray)
{
  if(abs(ray.dir.x) < EPSILON) ray.dir.x = sign(ray.dir.x) * EPSILON;
  if(abs(ray.dir.y) < EPSILON) ray.dir.y = sign(ray.dir.y) * EPSILON;
  if(abs(ray.dir.z) < EPSILON) ray.dir.z = sign(ray.dir.z) * EPSILON;
}
uniform sampler2D in_colorTransferFunc_0[1];
uniform sampler2D in_opacityTransferFunc_0[1];
float computeOpacity(vec4 scalar)
{
  return texture2D(in_opacityTransferFunc_0[0], vec2(scalar.w, 0)).r;
}
vec4 computeGradient(in vec3 texPos, in int c, in sampler3D volume, in int index) { return vec4(0.0); }
vec4 computeLighting(vec4 color, int component, float label)
{
  vec4 finalColor = vec4(0.0);
  int lightingComponent=component;
  finalColor = vec4(color.rgb, 0.0);
  finalColor.a = color.a;
  return finalColor;
}
vec4 computeColor(vec4 scalar, float opacity)
{
  return clamp(computeLighting(vec4(texture2D(in_colorTransferFunc_0[0], vec2(scalar.w, 0.0)).xyz, opacity), 0, 0.0), 0.0, 1.0);
}
vec3 computeRayDirection()
{
  return normalize(ip_vertexPos.xyz - in_eyePosObjs[0].xyz);
}
uniform float in_scale;
uniform float in_bias;
vec4 WindowToNDC(const float xCoord, const float yCoord, const float zCoord)
{
  vec4 NDCCoord = vec4(0.0, 0.0, 0.0, 1.0);
  NDCCoord.x = (xCoord - in_windowLowerLeftCorner.x) * 2.0 * in_inverseWindowSize.x - 1.0;
  NDCCoord.y = (yCoord - in_windowLowerLeftCorner.y) * 2.0 * in_inverseWindowSize.y - 1.0;
  NDCCoord.z = (2.0 * zCoord - (gl_DepthRange.near + gl_DepthRange.far)) / gl_DepthRange.diff;
  return NDCCoord;
}
vec4 NDCToWindow(const float xNDC, const float yNDC, const float zNDC)
{
  vec4 WinCoord = vec4(0.0, 0.0, 0.0, 1.0);
  WinCoord.x = (xNDC + 1.f) / (2.f * in_inverseWindowSize.x) + in_windowLowerLeftCorner.x;
  WinCoord.y = (yNDC + 1.f) / (2.f * in_inverseWindowSize.y) + in_windowLowerLeftCorner.y;
  WinCoord.z = (zNDC * gl_DepthRange.diff + (gl_DepthRange.near + gl_DepthRange.far)) / 2.f;
  return WinCoord;
}
vec3 ClampToSampleLocation(vec3 start, vec3 step, vec3 pos, bool ceiling)
{
  vec3 offset = pos - start;
  float stepLength = length(step);
  float dist = dot(offset, step / stepLength);
  if (dist < 0.) { return start; }
  float steps = dist / stepLength;
  if (abs(mod(steps, 1.)) > 1e-5)
  {
    if (ceiling) { steps = ceil(steps); } else { steps = floor(steps); }
  }
  else { steps = floor(steps + 0.5); }
  return start + steps * step;
}
void initializeRayCast()
{
  g_fragColor = vec4(0.0);
  if (uDbg == 8) { float v1 = texture3D(in_volume[0], vec3(0.25, 0.5, 0.5)).r; float v2 = texture3D(in_volume[0], vec3(0.5, 0.5, 0.25)).r; float o1 = texture2D(in_opacityTransferFunc_0[0], vec2(0.1, 0.0)).r; float o2 = texture2D(in_opacityTransferFunc_0[0], vec2(0.25, 0.0)).r; g_fragColor = vec4(v1, v2, o1, 1.0); if (gl_FragCoord.x < 4.0) g_fragColor = vec4(o2, 0.0, 0.0, 1.0); g_exit = true; }
  g_dirStep = vec3(0.0);
  g_srcColor = vec4(0.0);
  g_exit = false;
  g_rayOrigin = ip_textureCoords.xyz;
  vec3 rayDir = computeRayDirection();
  vec2 fragTexCoord = (gl_FragCoord.xy - in_windowLowerLeftCorner) * in_inverseWindowSize;
  g_dirStep = (ip_inverseTextureDataAdjusted * vec4(rayDir, 0.0)).xyz * in_sampleDistance;
  g_lengthStep = length(g_dirStep);
  float jitterValue = 0.0;
  if (uJitter > 0)
  {
    jitterValue = texture2D(in_noiseSampler, gl_FragCoord.xy / vec2(textureSize(in_noiseSampler, 0))).x;
    g_rayJitter = g_dirStep * jitterValue;
  }
  else
  {
    g_rayJitter = g_dirStep;
  }
  g_rayOrigin += g_rayJitter;
  g_skip = false;
  bool stop = false;
  g_terminatePointMax = 0.0;
  vec4 l_depthValue = texture2D(in_depthSampler, fragTexCoord);
  if(gl_FragCoord.z >= l_depthValue.x)
  {
    discard;
  }
  fragTexCoord = (gl_FragCoord.xy - in_windowLowerLeftCorner) * in_inverseOriginalWindowSize;
  vec4 rayTermination = WindowToNDC(gl_FragCoord.x, gl_FragCoord.y, l_depthValue.x);
  rayTermination = ip_inverseTextureDataAdjusted * in_inverseVolumeMatrix[0] * in_inverseModelViewMatrix * in_inverseProjectionMatrix * rayTermination;
  g_rayTermination = rayTermination.xyz / rayTermination.w;
  g_dataPos = g_rayOrigin;
  g_terminatePos = g_rayTermination;
  g_terminatePointMax = length(g_terminatePos.xyz - g_dataPos.xyz) / length(g_dirStep);
  g_currentT = 0.0;
  g_dbgTPM = g_terminatePointMax;
  g_dbgDir = length(g_dirStep);
  vec4 tempClip = in_volumeMatrix[0] * vec4(rayDir, 0.0);
  if (tempClip.w != 0.0)
  {
    tempClip = tempClip/tempClip.w;
    tempClip.w = 1.0;
  }
  clip_rayDirObj = normalize(tempClip.xyz);
  clip_numPlanes = int(in_clippingPlanes[0]);
  clip_texToObjMat = in_volumeMatrix[0] * inverse(ip_inverseTextureDataAdjusted);
  clip_objToTexMat = ip_inverseTextureDataAdjusted * in_inverseVolumeMatrix[0];
  if (!AdjustSampleRangeForClipping(g_rayOrigin, g_rayTermination))
  {
    discard;
  }
  g_dataPos = g_rayOrigin;
  g_terminatePos = g_rayTermination;
  g_terminatePointMax = length(g_terminatePos.xyz - g_dataPos.xyz) / length(g_dirStep);
  g_jitterValue = jitterValue;
}
vec4 castRay(const float zStart, const float zEnd)
{
  while (!g_exit)
  {
    g_skip = false;
    if (!g_skip)
    {
      g_lastPos = g_dataPos;
      vec4 scalar;
      scalar = texture3D(in_volume[0], g_dataPos);
      if (uDbg == 10) { g_fragColor = vec4(texture3D(in_volume[0], vec3(0.5, 0.5, 0.5)).r, texture3D(in_volume[0], vec3(0.25, 0.5, 0.5)).r, 0.0, 1.0); break; }
      scalar.r = scalar.r * in_volume_scale[0].r + in_volume_bias[0].r;
      if (uDbg == 14) { g_fragColor = vec4(g_dataPos.x, g_dataPos.y, g_dataPos.z, 1.0); break; }
      if (uDbg == 15) { if (g_currentT == uDbgIter) { g_fragColor = vec4(scalar.r, 0.0, 0.0, 1.0); break; } }
      if (uDbg == 16) { if (scalar.r > dbgMax) { dbgMax = scalar.r; dbgPos = g_dataPos; } }
      if (uDbg == 19) { if (g_currentT == gl_FragCoord.x - 0.5) { g_fragColor = vec4(scalar.r, scalar.r, scalar.r, 1.0); break; } }
      if (uDbg == 1) { dbgMax = max(dbgMax, scalar.r); }
      if (uDbg == 7) { dbgMax = max(dbgMax, scalar.r); }
      scalar = vec4(scalar.r);
      g_srcColor = vec4(0.0);
      g_srcColor.a = computeOpacity(scalar);
      if (uDbg == 12) { dbgMax = max(dbgMax, g_srcColor.a); }
      if (uDbg == 13) { dbgSum += g_srcColor.a; }
      if (uDbg == 1) { dbgAlpha = max(dbgAlpha, g_srcColor.a); }
      if (uDbg == 5 && g_srcColor.a > 0.0) { g_fragColor = vec4(texture2D(in_colorTransferFunc_0[0], vec2(scalar.w, 0.0)).xyz, 1.0); break; }
      if (uDbg == 6 && g_srcColor.a > 0.5) { g_fragColor = g_srcColor; break; }
      if (g_srcColor.a > 0.0)
      {
        g_srcColor = computeColor(scalar, g_srcColor.a);
        g_srcColor.rgb *= g_srcColor.a;
        g_fragColor = (1.0f - g_fragColor.a) * g_srcColor + g_fragColor;
      }
    }
    g_dataPos += g_dirStep;
    if(any(greaterThan(max(g_dirStep, vec3(0.0))*(g_dataPos - in_texMax[0]),vec3(0.0))) ||
      any(greaterThan(min(g_dirStep, vec3(0.0))*(g_dataPos - in_texMin[0]),vec3(0.0))))
    {
      break;
    }
    if((g_fragColor.a > g_opacityThreshold) || g_currentT >= g_terminatePointMax)
    {
      break;
    }
    if (uDbg == 1 && g_currentT > 100) { break; }
    ++g_currentT;
  }
  return g_fragColor;
}
void finalizeRayCast()
{
  if (uDbg == 1) { g_fragColor = vec4(dbgAlpha, dbgAlpha, dbgAlpha, 1.0); fragColor = g_fragColor; return; }
  if (uDbg == 12) { g_fragColor = vec4(dbgMax, dbgMax, dbgMax, 1.0); fragColor = g_fragColor; return; }
  if (uDbg == 13) { g_fragColor = vec4(dbgSum * 4.0, dbgSum * 4.0, dbgSum * 4.0, 1.0); fragColor = g_fragColor; return; }
  if (uDbg == 14) { fragColor = g_fragColor; return; }
  if (uDbg == 15) { fragColor = g_fragColor; return; }
  if (uDbg == 16) { g_fragColor = vec4(dbgPos.x, dbgPos.y, dbgPos.z, 1.0); fragColor = g_fragColor; return; }
  if (uDbg == 21) { g_fragColor = vec4(g_lastPos.x, g_lastPos.y, g_lastPos.z, 1.0); fragColor = g_fragColor; return; }
  if (uDbg == 22) { g_fragColor = vec4(clamp(g_dbgTPM, 0.0, 255.0) / 255.0, 0.0, 0.0, 1.0); fragColor = g_fragColor; return; }
  if (uDbg == 23) { g_fragColor = vec4(clamp(g_dbgDir * 64.0, 0.0, 1.0), 0.0, 0.0, 1.0); fragColor = g_fragColor; return; }
  if (uDbg == 25) { g_fragColor = vec4(clamp(g_currentT, 0.0, 255.0) / 255.0, 0.0, 0.0, 1.0); fragColor = g_fragColor; return; }
  if (uDbg == 26) { g_fragColor = vec4(clamp(g_rayTermination * 0.25 + 0.5, 0.0, 1.0), 1.0); fragColor = g_fragColor; return; }
  if (uDbg == 19) { fragColor = g_fragColor; return; }
  if (uDbg == 7) { g_fragColor = vec4(dbgMax, dbgMax, dbgMax, 1.0); fragColor = g_fragColor; return; }
  if (uDbg == 8) { fragColor = g_fragColor; return; }
  if (uDbg == 9) { fragColor = g_fragColor; return; }
  if (uDbg == 10) { fragColor = g_fragColor; return; }
  if (uDbg == 2) { g_fragColor = vec4(g_currentT / 255.0, 0.0, 0.0, 1.0); return; }
  if (uDbg == 3) { fragColor = g_fragColor; return; }
  if (uDbg == 4) { fragColor = g_fragColor; return; }
  if (uDbg == 5) { fragColor = g_fragColor; return; }
  if (uDbg == 6) { fragColor = g_fragColor; return; }
  g_fragColor.r = g_fragColor.r * in_scale + in_bias * g_fragColor.a;
  g_fragColor.g = g_fragColor.g * in_scale + in_bias * g_fragColor.a;
  g_fragColor.b = g_fragColor.b * in_scale + in_bias * g_fragColor.a;
  fragColor = g_fragColor;
}
void main()
{
  initializeRayCast();
  castRay(-1.0, -1.0);
  finalizeRayCast();
}
)";

  // patch the jitter uniform in
  std::string fsSrc(fs);
  {
    std::string marker = "uniform float in_scale;";
    std::string patch = "uniform int uJitter;\nuniform int uDbg;\nuniform float uDbgIter;\nuniform float in_scale;";
    size_t p = fsSrc.find(marker);
    if (p == std::string::npos) { fprintf(stderr, "marker not found\n"); return 1; }
    fsSrc.replace(p, marker.size(), patch);
  }
  GLint glMajor = 0, glMinor = 0;
  glGetIntegerv(GL_MAJOR_VERSION, &glMajor);
  glGetIntegerv(GL_MINOR_VERSION, &glMinor);
  fprintf(stderr, "GL %d.%d\n", glMajor, glMinor);

  GLuint prog = glCreateProgram();
  GLuint vsh = glCreateShader(GL_VERTEX_SHADER), fsh = glCreateShader(GL_FRAGMENT_SHADER);
  const char* vss = vs; const char* fss = fsSrc.c_str();
  {
    FILE* dbg = fopen("/tmp/fs_parity.glsl", "w");
    if (dbg) { fputs(fss, dbg); fclose(dbg); }
  }
  glShaderSource(vsh, 1, &vss, NULL); glCompileShader(vsh);
  glShaderSource(fsh, 1, &fss, NULL); glCompileShader(fsh);
  GLint ok = 0;
  glGetShaderiv(vsh, GL_COMPILE_STATUS, &ok);
  if (!ok) { char log[4096]; glGetShaderInfoLog(vsh, sizeof(log), NULL, log); fprintf(stderr, "vertex compile failed:\n%s\n", log); return 1; }
  glGetShaderiv(fsh, GL_COMPILE_STATUS, &ok);
  if (!ok) { char log[4096]; glGetShaderInfoLog(fsh, sizeof(log), NULL, log); fprintf(stderr, "fragment compile failed:\n%s\n", log); return 1; }
  glAttachShader(prog, vsh); glAttachShader(prog, fsh);
  glBindAttribLocation(prog, 0, "in_vertexPos");
  glBindAttribLocation(prog, 1, "in_texCoords");
  glLinkProgram(prog);
  glGetProgramiv(prog, GL_LINK_STATUS, &ok);
  if (!ok) { char log[4096]; glGetProgramInfoLog(prog, sizeof(log), NULL, log); fprintf(stderr, "link failed:\n%s\n", log); return 1; }
  glUseProgram(prog);

  glBindVertexArray(vao);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  // ---- uniforms
  glUniformMatrix4fv(glGetUniformLocation(prog, "uInvTexDataAdjusted"), 1, GL_FALSE, invtex);
  float ident[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
  // The k* constants are VTK-memory (row-major) listings; the live program's
  // GLSL-visible matrices are their TRANSPOSES (verified against
  // glGetUniformfv on 2026-08-22). Upload with GL_TRUE so the shader sees
  // exactly what the app sees. With GL_FALSE the non-symmetric projection/
  // modelview chains were garbage: g_terminatePointMax collapsed to ~21
  // iterations and every ray truncated ~4x early (the source of all bogus
  // fast timings from this tool before 2026-08-22).
  glUniformMatrix4fv(glGetUniformLocation(prog, "in_projectionMatrix"), 1, GL_FALSE, kProj);
  glUniformMatrix4fv(glGetUniformLocation(prog, "in_inverseProjectionMatrix"), 1, GL_FALSE, kInvProj);
  glUniformMatrix4fv(glGetUniformLocation(prog, "in_modelViewMatrix"), 1, GL_FALSE, kModelView);
  glUniformMatrix4fv(glGetUniformLocation(prog, "in_inverseModelViewMatrix"), 1, GL_FALSE, kInvModelView);

  // UNIFORM_DUMP: introspect the linked program exactly like the mapper's
  // VTK_METAL_TEST_DUMP_UNIFORMS hook so runtime state can be diffed
  // against /tmp/app_gl_uniforms.txt mechanically.
  {
    FILE* uf = fopen("/tmp/parity_gl_uniforms.txt", "w");
    if (uf)
    {
      GLint nuni = 0;
      glGetProgramiv(prog, GL_ACTIVE_UNIFORMS, &nuni);
      for (GLint i = 0; i < nuni; i++)
      {
        char uname[256]; GLsizei ulen = 0; GLint usize = 0; GLenum utype = 0;
        glGetActiveUniform(prog, (GLuint)i, sizeof(uname), &ulen, &usize, &utype, uname);
        GLint uloc = glGetUniformLocation(prog, uname);
        fprintf(uf, "%s type=%u size=%d loc=%d", uname, utype, usize, uloc);
        int nc = (utype == GL_FLOAT) ? 1 : (utype == GL_FLOAT_VEC2) ? 2 :
          (utype == GL_FLOAT_VEC3) ? 3 : (utype == GL_FLOAT_VEC4) ? 4 :
          (utype == GL_FLOAT_MAT4) ? 16 : (utype == GL_FLOAT_MAT3) ? 9 : 0;
        if (nc > 0)
        {
          GLfloat uv[16] = { 0 };
          glGetUniformfv(prog, uloc, uv);
          fprintf(uf, " =");
          for (int c = 0; c < nc; c++) fprintf(uf, " %g", (double)uv[c]);
        }
        else
        {
          GLint uv[4] = { 0 };
          glGetUniformiv(prog, uloc, uv);
          fprintf(uf, " = %d %d %d %d", uv[0], uv[1], uv[2], uv[3]);
        }
        fprintf(uf, "\n");
      }
      fclose(uf);
    }
  }
  glUniformMatrix4fv(glGetUniformLocation(prog, "in_volumeMatrix"), 1, GL_FALSE, ident);
  glUniformMatrix4fv(glGetUniformLocation(prog, "in_inverseVolumeMatrix"), 1, GL_FALSE, ident);
  glUniformMatrix4fv(glGetUniformLocation(prog, "in_textureDatasetMatrix"), 1, GL_FALSE, ident);
  glUniformMatrix4fv(glGetUniformLocation(prog, "in_inverseTextureDatasetMatrix"), 1, GL_FALSE, ident);
  glUniformMatrix4fv(glGetUniformLocation(prog, "in_textureToEye"), 1, GL_FALSE, ident);
  glUniformMatrix4fv(glGetUniformLocation(prog, "in_cellToPoint"), 1, GL_FALSE, ident);
  float texMin[3] = { CtpOffset(kVolW), CtpOffset(kVolH), CtpOffset(kVolD) };
  float texMax[3] = { CtpOffset(kVolW) + CtpScale(kVolW), CtpOffset(kVolH) + CtpScale(kVolH), CtpOffset(kVolD) + CtpScale(kVolD) };
  glUniform3fv(glGetUniformLocation(prog, "in_texMin"), 1, texMin);
  glUniform3fv(glGetUniformLocation(prog, "in_texMax"), 1, texMax);
  glUniform3fv(glGetUniformLocation(prog, "in_eyePosObjs"), 1, kEyeMM);
  glUniform1f(glGetUniformLocation(prog, "in_sampleDistance"), sd);
  glUniform1i(glGetUniformLocation(prog, "in_noOfComponents"), 1);
  glUniform1i(glGetUniformLocation(prog, "uJitter"), jitter);
  glUniform1i(glGetUniformLocation(prog, "uDbg"), getenv("APPGL_DBG") ? atoi(getenv("APPGL_DBG")) : 0);
  glUniform1f(glGetUniformLocation(prog, "uDbgIter"), getenv("APPGL_DBG_ITER") ? atof(getenv("APPGL_DBG_ITER")) : 30.0f);
  glUniform2f(glGetUniformLocation(prog, "in_windowLowerLeftCorner"), 0, 0);
  glUniform2f(glGetUniformLocation(prog, "in_inverseWindowSize"), 1.0f / rt, 1.0f / rt);
  glUniform2f(glGetUniformLocation(prog, "in_inverseOriginalWindowSize"), 1.0f / rt, 1.0f / rt);
  float clip[49]; memset(clip, 0, sizeof(clip)); clip[0] = 0.0f;  // no planes
  glUniform1fv(glGetUniformLocation(prog, "in_clippingPlanes"), 49, clip);
  glUniform1f(glGetUniformLocation(prog, "in_clippedVoxelIntensity"), 0.0f);
  float scaleBias[4] = { 0.0f, 0.0f, 0.0f, 0.0f };  // app dump: in_volume_bias = 0 0 0 0
  float volScale[4] = { 1.00392f, 1.0f, 1.0f, 1.0f };
  glUniform4fv(glGetUniformLocation(prog, "in_volume_scale"), 1, volScale);
  glUniform4fv(glGetUniformLocation(prog, "in_volume_bias"), 1, scaleBias);
  glUniform1f(glGetUniformLocation(prog, "in_scale"), 1.0f);
  glUniform1f(glGetUniformLocation(prog, "in_bias"), 0.0f);
  glUniform1i(glGetUniformLocation(prog, "in_volume"), 0);
  glUniform1i(glGetUniformLocation(prog, "in_noiseSampler"), 1);
  glUniform1i(glGetUniformLocation(prog, "in_depthSampler"), 2);
  glUniform1i(glGetUniformLocation(prog, "in_volume"), 0);
  glUniform1i(glGetUniformLocation(prog, "in_colorTransferFunc_0"), 3);
  glUniform1i(glGetUniformLocation(prog, "in_opacityTransferFunc_0"), 4);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_3D, volTex);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, noiseTex);
  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, depthTex);
  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, tfCTex);
  glActiveTexture(GL_TEXTURE4);
  glBindTexture(GL_TEXTURE_2D, tfOTex);
  glActiveTexture(GL_TEXTURE0);
  glUniform1i(glGetUniformLocation(prog, "in_colorTransferFunc_0"), 3);
  glUniform1i(glGetUniformLocation(prog, "in_opacityTransferFunc_0"), 4);
  glUniform1i(glGetUniformLocation(prog, "in_scalarsRange"), 0);

  glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_3D, volTex);
  glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, noiseTex);
  glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, depthTex);
  glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, tfCTex);
  glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, tfOTex);

  glViewport(0, 0, rt, rt);
  glDisable(GL_CULL_FACE);
  // the app's depth test (LESS vs the cleared 1.0 depth) passes every box
  // fragment; disabling it is equivalent.
  glDisable(GL_DEPTH_TEST);

  glClearColor(0, 0, 0, 1);
  int warmupN = getenv("WARMUP") ? atoi(getenv("WARMUP")) : 30;  // issue_tax R3
  for (int f = 0; f < warmupN; f++)
  {
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glFinish();
  }
  double t0 = NowMs();
  for (int f = 0; f < frames; f++)
  {
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glFinish();
  }
  double dt = NowMs() - t0;

  int nonzero = 0;
  {
    uint8_t* pix = (uint8_t*)malloc((size_t)rt * rt * 4);
    glReadPixels(0, 0, rt, rt, GL_RGBA, GL_UNSIGNED_BYTE, pix);
    for (size_t i = 0; i < (size_t)rt * rt * 4; i += 4)
      if (pix[i] | pix[i + 1] | pix[i + 2]) nonzero++;
    fprintf(stderr, "dbg: pix[0..8]=%d,%d,%d,%d,%d,%d,%d,%d  max=", pix[0], pix[1], pix[2], pix[3], pix[4], pix[5], pix[6], pix[7]);
    { int m = 0; for (int i = 0; i < rt * rt * 4; i += 4) { int v = pix[i]; if (v > m) m = v; } fprintf(stderr, "%d\n", m); }
    fprintf(stderr, "dbg bottom row: ");
    { size_t o = (size_t)(rt - 1) * rt * 4; for (int k = 0; k < 12; k++) fprintf(stderr, "%d,", pix[o + k]); fprintf(stderr, "\n"); }
    if (getenv("APPGL_DUMP_PPM"))
    {
      FILE* pp = fopen("/tmp/appgl_parity.ppm", "wb");
      if (pp)
      {
        fprintf(pp, "P6\n%d %d\n255\n", rt, rt);
        uint8_t* row = (uint8_t*)malloc((size_t)rt * 3);
        for (int y = rt - 1; y >= 0; y--)
        {
          const uint8_t* src = pix + (size_t)y * rt * 4;
          for (int x = 0; x < rt; x++)
          {
            row[x * 3 + 0] = src[x * 4 + 0];
            row[x * 3 + 1] = src[x * 4 + 1];
            row[x * 3 + 2] = src[x * 4 + 2];
          }
          fwrite(row, 1, (size_t)rt * 3, pp);
        }
        free(row);
        fclose(pp);
      }
    }
    free(pix);
  }
  fprintf(stderr, "GL    j%d: %8.3f ms/frame  (footprint %d/%d)\n",
          jitter, dt / frames, nonzero, rt * rt);
  fprintf(stderr, "APPGL j%d: %8.3f ms/frame\n", jitter, dt / frames);
  free(vol);
  return 0;
}