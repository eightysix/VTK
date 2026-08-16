// Minimal, self-contained Metal microbenchmark: the real DICOM scene's exact
// divergent perspective rays marching through a 512x512x1794 R8 3D texture
// with hardware trilinear filtering. Matches the app's VolumeMapperUniforms
// (CameraVolumePos, inverse view-projection, per-ray sample distance) so the
// raw-fetch march cost is identical to the production Metal renderer. The
// paired GL version (gl_gap) runs the same rays so the intrinsic Metal-vs-GL
// 3D sampler throughput gap can be measured in isolation. Pass rtSize
// (argv[10]) and sampleDistMM (argv[11]) to reproduce the coarse-SD high-res
// "lag" case (e.g. 2048 4.0).
//
// Build: clang -fobjc-arc -framework Metal -framework Foundation metal_gap.m -o metal_gap
// Run:   ./metal_gap [frames] [maxIter] [halfSampler] [depthMode] [compute] [lod0]
//   halfSampler=1: texture3d<half> + half accumulator (FP16 sampler return path)
//   depthMode: 0 = depth attached, MTLStoreActionStore (legacy), 1 = depth
//              attached, MTLStoreActionDontCare, 2 = no depth attachment
//              (matches gl_gap, which attaches no depth buffer by default)
//   compute=1: replace the fragment render pass with a compute kernel (same
//              march, one thread per pixel, no rasterizer/TBDR tile machinery)
//   lod0=1: sample(..., level(0.0f)) — explicit LOD 0, skipping the implicit
//              screen-space gradient computation MSL fragment .sample() does
//   rtSize (argv[10]): render-target size in px (default 400). Scale to
//              1024/2048 to reproduce the coarse-SD high-res "lag" case.
//   sampleDistMM (argv[11]): physical sample distance (default 0.5). Coarse
//              values (2.0/4.0) reproduce the high-SD lag the app shows.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <simd/simd.h>

struct Uniforms {
  simd_float4 eye;         // camera pos, normalized volume space
  simd_float4 boundsSize;  // physical volume size (mm)
  simd_float4x4 invVP;     // NDC -> physical volume coords
  float sampleDistMM;      // physical sample distance (mm)
  float _pad0;
  float _pad1;
  int   maxIter;           // safety loop bound
};

static int kW = 512;  // volume dims (slice-stacked, Z is the long axis)
static int kH = 512;
static int kD = 1794;
static int kRT = 400;       // render target size (fragments); argv[10]

// Real app uniforms for the DICOMVolume scene (dump via
// VTK_METAL_TEST_DUMP_UNIFORMS): CameraVolumePos (normalized),
// VolumeBoundsMax (physical mm), InverseViewProjection (NDC->physical).
// Builds the march MSL. `halfSampler` switches the volume texture and the
// accumulator to FP16 so the TMU/ALU sample-return path stays in half instead
// of forcing FP32 registers (experiment: does the GL stack's implicit 16-bit
// fast path explain part of the gap?).
// Builds an N-way unrolled march body: fetches samples s0..s(N-1) back-to-back
// (independent, all in flight) before consuming any, then advances by N steps.
// A scalar break-aware remainder handles the tail so the step count stays ~the
// baseline (avgIter parity preserved to within the coarse break granularity).
// n must be even. body must be >= 8192 bytes.
static void BuildUnrollBody(char* body, size_t cap, int n, const char* lt, const char* lod)
{
  int off = 0;
  off += snprintf(body + off, cap - off,
    "\n  int i = 0;\n"
    "  for (; i + %d <= steps; i += %d) {\n"
    "    if (currentT >= tExit - 1e-6f) break;\n", n, n);
  for (int k = 0; k < n; k++)
  {
    if (k == 0)
      off += snprintf(body + off, cap - off, "    float3 p%d = evalPoint;\n", k);
    else
      off += snprintf(body + off, cap - off, "    float3 p%d = evalPoint + evalStep * %d.0f;\n", k, k);
  }
  for (int k = 0; k < n; k++)
    off += snprintf(body + off, cap - off, "    %s s%d = volTex.sample(volSampler, p%d%s).r;\n", lt, k, k, lod);
  char expr[512];
  int eo = 0;
  for (int k = 0; k < n; k += 2)
  {
    if (eo == 0)
      eo = snprintf(expr, sizeof(expr), "max(s%d, s%d)", k, k + 1);
    else
    {
      char tmp[256];
      snprintf(tmp, sizeof(tmp), "max(%s, max(s%d, s%d))", expr, k, k + 1);
      eo = snprintf(expr, sizeof(expr), "%s", tmp);
    }
  }
  off += snprintf(body + off, cap - off, "    acc = max(acc, %s);\n", expr);
  off += snprintf(body + off, cap - off,
    "    currentT += stepSize * %d.0f; texLocal += texStep * %d.0f;\n"
    "    evalPoint += evalStep * %d.0f; n += %d.0f;\n"
    "  }\n", n, n, n, n);
  snprintf(body + off, cap - off,
    "  for (; i < steps; i++) {\n"
    "    if (currentT >= tExit - 1e-6f) break;\n"
    "    acc = max(acc, volTex.sample(volSampler, evalPoint%s).r);\n"
    "    currentT += stepSize; texLocal += texStep; evalPoint += evalStep; n += 1.0f;\n"
    "  }\n", lod);
}

// Same N-way unrolled march but sampling a texture2d_array (one layer per Z
// slice) with manual Z interpolation (2 fetches + mix per sample). Tests the
// 3D-Morton-vs-2D-slice-stack texture layout hypothesis: if Metal's default
// MTLTextureType3D swizzle is cache-hostile for coarse-stepped trilinear
// access on anisotropic NPOT volumes, slice-oriented storage should recover
// most of the noise-data gap. n = unroll factor (0 = scalar loop).
static void BuildSlicedBody(char* body, size_t cap, int n, const char* lt)
{
  int off = 0;
  if (n <= 1)
  {
    off += snprintf(body + off, cap - off, R"(
  for (int i = 0; i < steps; i++) {
    if (currentT >= tExit - 1e-6f) break;
    float zc = evalPoint.z * zScale;
    uint z0 = uint(zc);
    %s f = %s(zc - float(z0));
    uint z1 = min(z0 + 1u, zMax);
    %s m = mix(volTex.sample(s2d, evalPoint.xy, z0).r, volTex.sample(s2d, evalPoint.xy, z1).r, f);
    acc = max(acc, m);
    currentT += stepSize; texLocal += texStep; evalPoint += evalStep; n += 1.0f;
  }
)", lt, lt, lt);
    return;
  }
  off += snprintf(body + off, cap - off,
    "\n  int i = 0;\n"
    "  for (; i + %d <= steps; i += %d) {\n"
    "    if (currentT >= tExit - 1e-6f) break;\n", n, n);
  for (int k = 0; k < n; k++)
  {
    if (k == 0)
      off += snprintf(body + off, cap - off, "    float3 p%d = evalPoint;\n", k);
    else
      off += snprintf(body + off, cap - off, "    float3 p%d = evalPoint + evalStep * %d.0f;\n", k, k);
  }
  for (int k = 0; k < n; k++)
    off += snprintf(body + off, cap - off,
      "    float zc%d = p%d.z * zScale;\n"
      "    uint z0%d = uint(zc%d);\n"
      "    %s f%d = %s(zc%d - float(z0%d));\n"
      "    uint z1%d = min(z0%d + 1u, zMax);\n", k, k, k, k, lt, k, lt, k, k, k, k);
  for (int k = 0; k < n; k++)
    off += snprintf(body + off, cap - off,
      "    %s s%da = volTex.sample(s2d, p%d.xy, z0%d).r;\n"
      "    %s s%db = volTex.sample(s2d, p%d.xy, z1%d).r;\n", lt, k, k, k, lt, k, k, k);
  for (int k = 0; k < n; k++)
    off += snprintf(body + off, cap - off, "    %s m%d = mix(s%da, s%db, f%d);\n", lt, k, k, k, k);
  char expr[512];
  int eo = 0;
  for (int k = 0; k < n; k += 2)
  {
    if (eo == 0)
      eo = snprintf(expr, sizeof(expr), "max(m%d, m%d)", k, k + 1);
    else
    {
      char tmp[256];
      snprintf(tmp, sizeof(tmp), "max(%s, max(m%d, m%d))", expr, k, k + 1);
      eo = snprintf(expr, sizeof(expr), "%s", tmp);
    }
  }
  off += snprintf(body + off, cap - off, "    acc = max(acc, %s);\n", expr);
  off += snprintf(body + off, cap - off,
    "    currentT += stepSize * %d.0f; texLocal += texStep * %d.0f;\n"
    "    evalPoint += evalStep * %d.0f; n += %d.0f;\n"
    "  }\n", n, n, n, n);
  snprintf(body + off, cap - off,
    "  for (; i < steps; i++) {\n"
    "    if (currentT >= tExit - 1e-6f) break;\n"
    "    float zc = evalPoint.z * zScale;\n"
    "    uint z0 = uint(zc);\n"
    "    %s f = %s(zc - float(z0));\n"
    "    uint z1 = min(z0 + 1u, zMax);\n"
    "    %s m = mix(volTex.sample(s2d, evalPoint.xy, z0).r, volTex.sample(s2d, evalPoint.xy, z1).r, f);\n"
    "    acc = max(acc, m);\n"
    "    currentT += stepSize; texLocal += texStep; evalPoint += evalStep; n += 1.0f;\n"
    "  }\n", lt, lt, lt);
}

// Replace "filter::linear" with "filter::nearest" in an emitted MSL string.
// Used to isolate the trilinear 8-texel-span cost from the bare fetch cost.
static void ApplyFilter(char* buf, const char* filt)
{
  if (strcmp(filt, "nearest") != 0) return;
  const char needle[] = "filter::linear";
  const char repl[] = "filter::nearest";
  char* p;
  while ((p = strstr(buf, needle)) != NULL)
  {
    memmove(p + strlen(repl), p + strlen(needle), strlen(p + strlen(needle)) + 1);
    memcpy(p, repl, strlen(repl));
  }
}

static void ApplySlab(char* buf, int slabStart, int slabEnd, int numSlabs)
{
  char clamp[256];
  snprintf(clamp, sizeof(clamp),
    "float3 evalPoint = texLocal * ctpScale + ctpOffset;\n"
    "    float zfrac = (evalPoint.z - ctpOffset.z) / ctpScale.z;\n"
    "    evalPoint.z = clamp(zfrac, %d.0f/%d.0f, %d.0f/%d.0f) * ctpScale.z + ctpOffset.z;\n",
    slabStart, numSlabs, slabEnd, numSlabs);
  const char needle[] = "float3 evalPoint = texLocal * ctpScale + ctpOffset;\n";
  char* p = buf;
  size_t clen = strlen(clamp), nlen = strlen(needle);
  while ((p = strstr(p, needle)) != NULL)
  {
    memmove(p + clen, p + nlen, strlen(p + nlen) + 1);
    memcpy(p, clamp, clen);
    p += clen;
  }
}

// texture2d_array variant of BuildMSL: identical rays, one 2D layer per Z
// slice, manual Z trilinear. `layoutMode` selects it (see main()).
static char* BuildSlicedMSL(bool halfSampler, int pipeline, const char* filt)
{
  const char* lt = halfSampler ? "half" : "float";
  const char* ls = halfSampler ? "h" : "f";
  char body[16384];
  BuildSlicedBody(body, sizeof(body), pipeline >= 2 ? pipeline : 0, lt);

  const char* fmt = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct VOut {
  float4 position [[position]];
  float2 uv;
};

vertex VOut vertex_main(uint vid [[vertex_id]]) {
  VOut o;
  float2 p = float2(float(vid & 1u) * 4.0f - 1.0f, float((vid >> 1u) & 1u) * 2.0f - 1.0f);
  if (vid == 2u) p = float2(-1.0f, 3.0f);
  o.position = float4(p, 0.0f, 1.0f);
  o.uv = p * 0.5f + 0.5f;
  return o;
}

struct Uniforms {
  float4 eye;         // camera position, normalized volume space
  float4 boundsSize;  // physical volume size (mm)
  float4x4 invVP;     // NDC -> physical volume coords
  float  sampleDistMM; // physical sample distance (mm)
  float  pad1;
  float  pad2;
  int    maxIter;     // safety loop bound
};

fragment float4 fragment_main(VOut in [[stage_in]],
                              texture2d_array<%s> volTex [[texture(0)]],
                              constant Uniforms& u [[buffer(0)]]) {
  constexpr sampler s2d(filter::linear, address::clamp_to_edge);
  float2 ndc = in.uv * 2.0f - 1.0f;

  float4 w4 = u.invVP * float4(ndc, 0.0f, 1.0f);
  float3 ptPhys = w4.xyz / w4.w;
  float3 eye = u.eye.xyz;
  float3 rayDir = normalize(ptPhys / u.boundsSize.xyz - eye);
  float3 inv = 1.0f / rayDir;

  float3 t0 = (float3(0.0f) - eye) * inv;
  float3 t1 = (float3(1.0f) - eye) * inv;
  float3 tmin3 = min(t0, t1);
  float3 tmax3 = max(t0, t1);
  float tEnter = max(max(tmin3.x, tmin3.y), tmin3.z);
  float tExit  = min(min(tmax3.x, tmax3.y), tmax3.z);
  if (tExit <= 0.0f || tEnter >= tExit) {
    return float4(0.0f, 0.0f, 0.0f, 1.0f);
  }
  float tStart = max(tEnter, 0.0f);

  float physPerNorm = length(rayDir * u.boundsSize.xyz);
  float stepSize = u.sampleDistMM / max(physPerNorm, 1e-6f);
  int maxSteps = max(1, int(ceil((tExit - tStart) / stepSize)));

  float3 texelCount = float3(float(volTex.get_width()), float(volTex.get_height()), float(volTex.get_array_size()));
  float3 ctpScale = max(texelCount - 1.0f, 1e-4f) / texelCount;
  float3 ctpOffset = 0.5f / texelCount;
  float3 texStep = rayDir * stepSize;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;
  float3 texLocal = eye + rayDir * currentT;
  float3 evalPoint = texLocal * ctpScale + ctpOffset;

  float zScale = float(volTex.get_array_size()) - 1.0f;
  uint zMax = uint(volTex.get_array_size()) - 1u;

  %s acc = 0.0%s;
  float n = 0.0f;
  int steps = min(u.maxIter, maxSteps);
  %s
  uint nc = uint(n);
  return float4(float(nc & 255u) / 255.0f, float((nc >> 8u) & 255u) / 255.0f, float(acc), 1.0f);
}
)MSL";
  char* buf = malloc(strlen(fmt) + strlen(body) + 64);
  snprintf(buf, strlen(fmt) + strlen(body) + 64, fmt, lt, lt, ls, body);
  ApplyFilter(buf, filt);
  return buf;
}

// BuildMSL: builds the march MSL. `halfSampler` switches the volume texture
// and the accumulator to FP16. `lod0` appends `level(0.0f)` to the sample
// call, forcing an explicit LOD-0 fetch and (hypothesis) skipping the implicit
// screen-space gradient/LOD computation that MSL fragment `.sample()` does per
// iteration; GL with GL_LINEAR (no mipmaps) needs no such gradients.
// `pipeline` selects the inner-loop structure (ILP / latency-hiding test):
//   0 = baseline (break, serial fetch->max chain)
//   1 = 2-way software-pipelined: fetch step i+1 before consuming step i,
//       break kept per-step (semantics identical to baseline, avgIter parity)
//   >=2 = N-way unroll (N even): N independent back-to-back fetches per
//       iteration, scalar break-aware tail. N = 2/4/8 tested.
//   3 = no break / fixed count (isolates the data-dependent break's cost)
static char* BuildMSL(bool halfSampler, bool lod0, int pipeline, const char* filt)
{
  const char* lt = halfSampler ? "half" : "float";
  const char* ls = halfSampler ? "h" : "f";
  const char* lod = lod0 ? ", level(0.0f)" : "";
  char body[8192];

  if (pipeline == 1)
  {
    snprintf(body, sizeof(body), R"MSL(
  %s cur = volTex.sample(volSampler, evalPoint%s).r;
  currentT += stepSize; texLocal += texStep; evalPoint += evalStep; n += 1.0f;
  for (int i = 1; i < steps; i++) {
    if (currentT >= tExit - 1e-6f) { acc = max(acc, cur); break; }
    %s nxt = volTex.sample(volSampler, evalPoint%s).r;
    acc = max(acc, cur); cur = nxt;
    currentT += stepSize; texLocal += texStep; evalPoint += evalStep; n += 1.0f;
  }
)MSL", lt, lod, lt, lod);
  }
  else if (pipeline == 3)
  {
    snprintf(body, sizeof(body), R"MSL(
  for (int i = 0; i < steps; i++) {
    acc = max(acc, volTex.sample(volSampler, evalPoint%s).r);
    currentT += stepSize; texLocal += texStep; evalPoint += evalStep; n += 1.0f;
  }
)MSL", lod);
  }
  else if (pipeline >= 2)
  {
    BuildUnrollBody(body, sizeof(body), pipeline, lt, lod);
  }
  else
  {
    snprintf(body, sizeof(body), R"MSL(
  for (int i = 0; i < steps; i++) {
    if (currentT >= tExit - 1e-6f) break;
    acc = max(acc, volTex.sample(volSampler, evalPoint%s).r);
    currentT += stepSize; texLocal += texStep; evalPoint += evalStep; n += 1.0f;
  }
)MSL", lod);
  }

  const char* fmt = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct VOut {
  float4 position [[position]];
  float2 uv;
};

vertex VOut vertex_main(uint vid [[vertex_id]]) {
  VOut o;
  float2 p = float2(float(vid & 1u) * 4.0f - 1.0f, float((vid >> 1u) & 1u) * 2.0f - 1.0f);
  if (vid == 2u) p = float2(-1.0f, 3.0f);
  o.position = float4(p, 0.0f, 1.0f);
  o.uv = p * 0.5f + 0.5f;
  return o;
}

struct Uniforms {
  float4 eye;         // camera position, normalized volume space
  float4 boundsSize;  // physical volume size (mm)
  float4x4 invVP;     // NDC -> physical volume coords
  float  sampleDistMM; // physical sample distance (mm)
  float  pad1;
  float  pad2;
  int    maxIter;     // safety loop bound
};

fragment float4 fragment_main(VOut in [[stage_in]],
                              texture3d<%s> volTex [[texture(0)]],
                              constant Uniforms& u [[buffer(0)]]) {
  constexpr sampler volSampler(filter::linear, address::clamp_to_edge);
  float2 ndc = in.uv * 2.0f - 1.0f;

  float4 w4 = u.invVP * float4(ndc, 0.0f, 1.0f);
  float3 ptPhys = w4.xyz / w4.w;
  float3 eye = u.eye.xyz;
  float3 rayDir = normalize(ptPhys / u.boundsSize.xyz - eye);
  float3 inv = 1.0f / rayDir;

  float3 t0 = (float3(0.0f) - eye) * inv;
  float3 t1 = (float3(1.0f) - eye) * inv;
  float3 tmin3 = min(t0, t1);
  float3 tmax3 = max(t0, t1);
  float tEnter = max(max(tmin3.x, tmin3.y), tmin3.z);
  float tExit  = min(min(tmax3.x, tmax3.y), tmax3.z);
  if (tExit <= 0.0f || tEnter >= tExit) {
    return float4(0.0f, 0.0f, 0.0f, 1.0f);
  }
  float tStart = max(tEnter, 0.0f);

  float physPerNorm = length(rayDir * u.boundsSize.xyz);
  float stepSize = u.sampleDistMM / max(physPerNorm, 1e-6f);
  int maxSteps = max(1, int(ceil((tExit - tStart) / stepSize)));

  float3 texelCount = float3(volTex.get_width(), volTex.get_height(), volTex.get_depth());
  float3 ctpScale = max(texelCount - 1.0f, 1e-4f) / texelCount;
  float3 ctpOffset = 0.5f / texelCount;
  float3 texStep = rayDir * stepSize;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;
  float3 texLocal = eye + rayDir * currentT;
  float3 evalPoint = texLocal * ctpScale + ctpOffset;

  %s acc = 0.0%s;
  float n = 0.0f;
  int steps = min(u.maxIter, maxSteps);
  %s
  uint nc = uint(n);
  return float4(float(nc & 255u) / 255.0f, float((nc >> 8u) & 255u) / 255.0f, float(acc), 1.0f);
}
)MSL";
  char* buf = malloc(strlen(fmt) + strlen(body) + 64);
  snprintf(buf, strlen(fmt) + strlen(body) + 64, fmt, lt, lt, ls, body);
  ApplyFilter(buf, filt);
  return buf;
}

// Diagnostic variant: writes rayDir*0.5+0.5 to RGB so the actual ray field can
// be compared against the fp64 reference (used to explain the avgIter gap).
static void ApplySlabT(char* buf, int slabStart, int slabEnd, int numSlabs)
{
  char clamp[512];
  snprintf(clamp, sizeof(clamp),
    "float stepSize = u.sampleDistMM / max(physPerNorm, 1e-6f);\n"
    "    float t_s = (%d.0f/%d.0f - u.eye.z) / rayDir.z;\n"
    "    float t_e = (%d.0f/%d.0f - u.eye.z) / rayDir.z;\n"
    "    float tlo = max(tStart, min(t_s, t_e));\n"
    "    float thi = min(tExit, max(t_s, t_e));\n"
    "    float kk = ceil(max((tlo - tStart) / stepSize, 0.0f));\n"
    "    tStart = tStart + kk * stepSize;\n"
    "    tExit = thi;\n",
    slabStart, numSlabs, slabEnd, numSlabs);
  const char needle[] = "float stepSize = u.sampleDistMM / max(physPerNorm, 1e-6f);\n";
  char* p = buf;
  size_t clen = strlen(clamp), nlen = strlen(needle);
  while ((p = strstr(p, needle)) != NULL)
  {
    memmove(p + clen, p + nlen, strlen(p + nlen) + 1);
    memcpy(p, clamp, clen);
    p += clen;
  }
}

static void ApplyAccOut(char* buf)
{
  const char needle[] = "float(nc & 255u) / 255.0f, float((nc >> 8u) & 255u) / 255.0f, float(acc), 1.0f);";
  const char repl[] = "acc, acc, acc, 1.0f);";
  char* p;
  while ((p = strstr(buf, needle)) != NULL)
  {
    memmove(p + strlen(repl), p + strlen(needle), strlen(p + strlen(needle)) + 1);
    memcpy(p, repl, strlen(repl));
  }
}

static char* BuildDiagMSL(void)
{
  const char* fmt = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct VOut { float4 position [[position]]; float2 uv; };

vertex VOut vertex_main(uint vid [[vertex_id]]) {
  VOut o;
  float2 p = float2(float(vid & 1u) * 4.0f - 1.0f, float((vid >> 1u) & 1u) * 2.0f - 1.0f);
  if (vid == 2u) p = float2(-1.0f, 3.0f);
  o.position = float4(p, 0.0f, 1.0f);
  o.uv = p * 0.5f + 0.5f;
  return o;
}

struct Uniforms {
  float4 eye;         // camera position, normalized volume space
  float4 boundsSize;  // physical volume size (mm)
  float4x4 invVP;     // NDC -> physical volume coords
  float  sampleDistMM; // physical sample distance (mm)
  float  pad1;
  float  pad2;
  int    maxIter;     // safety loop bound
};

fragment float4 fragment_main(VOut in [[stage_in]],
                              texture3d<float> volTex [[texture(0)]],
                              constant Uniforms& u [[buffer(0)]]) {
  float2 ndc = in.uv * 2.0f - 1.0f;
  float4 w4 = u.invVP * float4(ndc, 0.0f, 1.0f);
  float3 ptPhys = w4.xyz / w4.w;
  float3 eye = u.eye.xyz;
  float3 rayDir = normalize(ptPhys / u.boundsSize.xyz - eye);
  return float4(rayDir * 0.5f + 0.5f, 1.0f);
}
)MSL";
  char* buf = malloc(strlen(fmt) + 8);
  strcpy(buf, fmt);
  return buf;
}

// depthMode: 0 = depth attached + MTLStoreActionStore, 1 = depth attached +
// MTLStoreActionDontCare (skip the tile->memory depth writeback), 2 = no depth
// attachment at all (matches gl_gap's default no-depth setup).
static MTLRenderPassDescriptor* MakeRPDAction(id<MTLTexture> rt, id<MTLTexture> depth, int depthMode, MTLLoadAction load)
{
  MTLRenderPassDescriptor* rpd = [[MTLRenderPassDescriptor alloc] init];
  rpd.colorAttachments[0].texture = rt;
  rpd.colorAttachments[0].loadAction = load;
  rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
  if (depthMode != 2) {
    rpd.depthAttachment.texture = depth;
    rpd.depthAttachment.loadAction = MTLLoadActionClear;
    rpd.depthAttachment.storeAction = (depthMode == 1) ? MTLStoreActionDontCare : MTLStoreActionStore;
  }
  return rpd;
}

static MTLRenderPassDescriptor* MakeRPD(id<MTLTexture> rt, id<MTLTexture> depth, int depthMode)
{
  return MakeRPDAction(rt, depth, depthMode, MTLLoadActionClear);
}

// Compute-kernel variant of the same march: one thread per pixel, no render
// pass / rasterizer / depth attachment. Tests whether the fragment pipeline's
// per-tile machinery (vs a raw compute dispatch) contributes to the Metal gap.
static char* BuildComputeMSL(bool halfSampler, const char* filt)
{
  const char* fmt = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
  float4 eye;         // camera position, normalized volume space
  float4 boundsSize;  // physical volume size (mm)
  float4x4 invVP;     // NDC -> physical volume coords
  float  sampleDistMM; // physical sample distance (mm)
  float  pad1;
  float  pad2;
  int    maxIter;     // safety loop bound
};

kernel void compute_main(uint2 gid [[thread_position_in_grid]],
                         texture3d<%s> volTex [[texture(0)]],
                         texture2d<float, access::write> outTex [[texture(1)]],
                         constant Uniforms& u [[buffer(0)]],
                         constant int& rtSize [[buffer(1)]]) {
  constexpr sampler volSampler(filter::linear, address::clamp_to_edge);
  float2 uv = (float2(gid) + 0.5f) / float2(rtSize, rtSize);
  float2 ndc = uv * 2.0f - 1.0f;

  float4 w4 = u.invVP * float4(ndc, 0.0f, 1.0f);
  float3 ptPhys = w4.xyz / w4.w;
  float3 eye = u.eye.xyz;
  float3 rayDir = normalize(ptPhys / u.boundsSize.xyz - eye);
  float3 inv = 1.0f / rayDir;

  float3 t0 = (float3(0.0f) - eye) * inv;
  float3 t1 = (float3(1.0f) - eye) * inv;
  float3 tmin3 = min(t0, t1);
  float3 tmax3 = max(t0, t1);
  float tEnter = max(max(tmin3.x, tmin3.y), tmin3.z);
  float tExit  = min(min(tmax3.x, tmax3.y), tmax3.z);
  if (tExit <= 0.0f || tEnter >= tExit) {
    outTex.write(float4(0.0f, 0.0f, 0.0f, 1.0f), gid);
    return;
  }
  float tStart = max(tEnter, 0.0f);

  float physPerNorm = length(rayDir * u.boundsSize.xyz);
  float stepSize = u.sampleDistMM / max(physPerNorm, 1e-6f);
  int maxSteps = max(1, int(ceil((tExit - tStart) / stepSize)));

  float3 texelCount = float3(volTex.get_width(), volTex.get_height(), volTex.get_depth());
  float3 ctpScale = max(texelCount - 1.0f, 1e-4f) / texelCount;
  float3 ctpOffset = 0.5f / texelCount;
  float3 texStep = rayDir * stepSize;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;
  float3 texLocal = eye + rayDir * currentT;
  float3 evalPoint = texLocal * ctpScale + ctpOffset;

  %s acc = 0.0%s;
  float n = 0.0f;
  for (int i = 0; i < min(u.maxIter, maxSteps); i++) {
    if (currentT >= tExit - 1e-6f) break;
    acc = max(acc, volTex.sample(volSampler, evalPoint).r);
    currentT += stepSize;
    texLocal += texStep;
    evalPoint += evalStep;
    n += 1.0f;
  }
  uint nc = uint(n);
  outTex.write(float4(float(nc & 255u) / 255.0f, float((nc >> 8u) & 255u) / 255.0f, float(acc), 1.0f), gid);
}
)MSL";
  char* buf = malloc(strlen(fmt) + 32);
  snprintf(buf, strlen(fmt) + 32, fmt,
    halfSampler ? "half" : "float",
    halfSampler ? "half" : "float",
    halfSampler ? "h" : "f");
  ApplyFilter(buf, filt);
  return buf;
}

int main(int argc, const char** argv)
{
  @autoreleasepool {
    int frames = 100;
    int maxIter = 4096;
    int halfSampler = 0;
    int depthMode = 0;
    int compute = 0;
    int lod0 = 0;
    int fastMath = 1;
    int diag = 0;
    int pipeline = 0;
    float sampleDistMM = 0.5f;
    int dataMode = 0;
    int layoutMode = 0;
    int filterMode = 0;
    int volDiv = 1;
    int numSlabs = 0;
    int slabIndex = 0;
    int slabT = 0;
    int maccum = 0;
    if (argc > 1) frames = atoi(argv[1]);
    if (argc > 2) maxIter = atoi(argv[2]);
    if (argc > 3) halfSampler = atoi(argv[3]);
    if (argc > 4) depthMode = atoi(argv[4]);
    if (argc > 5) compute = atoi(argv[5]);
    if (argc > 6) lod0 = atoi(argv[6]);
    if (argc > 7) fastMath = atoi(argv[7]);
    if (argc > 8) diag = atoi(argv[8]);
    if (argc > 9) pipeline = atoi(argv[9]);
    if (argc > 10) kRT = atoi(argv[10]);
    if (argc > 11) sampleDistMM = (float)atof(argv[11]);
    if (argc > 12) dataMode = atoi(argv[12]);
    if (argc > 13) layoutMode = atoi(argv[13]);
    if (argc > 14) filterMode = atoi(argv[14]);
    if (argc > 15) { volDiv = atoi(argv[15]); if (volDiv < 1) volDiv = 1; kW = 512 / volDiv; kH = 512 / volDiv; kD = 1794 / volDiv; }
    if (argc > 16) numSlabs = atoi(argv[16]);
    if (argc > 17) slabIndex = atoi(argv[17]);
    if (argc > 18) slabT = atoi(argv[18]);
    if (argc > 19) maccum = atoi(argv[19]);

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    fprintf(stderr, "Metal device: %s\n", device.name.UTF8String);
    fprintf(stderr, "volume %dx%dx%d R8, rt %dx%d, frames %d, maxIter %d, halfSampler=%d, depthMode=%d, compute=%d, lod0=%d, fastMath=%d, diag=%d, pipeline=%d, sampleDistMM=%.1f, dataMode=%d, layoutMode=%d, filterMode=%d, volDiv=%d, slab=%d/%d, slabT=%d, maccum=%d\n",
      kW, kH, kD, kRT, kRT, frames, maxIter, halfSampler, depthMode, compute, lod0, fastMath, diag, pipeline, sampleDistMM, dataMode, layoutMode, filterMode, volDiv, slabIndex, numSlabs, slabT, maccum);

    NSError* err = nil;
    const char* filt = filterMode ? "nearest" : "linear";
    const int ns = (maccum && numSlabs > 0 && !compute) ? numSlabs : 1;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.fastMathEnabled = (fastMath != 0) ? YES : NO;
    id<MTLFunction> vertF = nil;
    id<MTLFunction> fragF = nil;
    id<MTLFunction> kernF = nil;
    id<MTLRenderPipelineState> pso = nil;
    id<MTLComputePipelineState> cps = nil;
    NSMutableArray* psoArr = [NSMutableArray arrayWithCapacity:(NSUInteger)ns];
    for (int si = 0; si < ns; si++) {
      char* msl = diag ? BuildDiagMSL()
                       : (compute ? BuildComputeMSL(halfSampler != 0, filt)
                                  : (layoutMode ? BuildSlicedMSL(halfSampler != 0, pipeline, filt)
                                                : BuildMSL(halfSampler != 0, lod0 != 0, pipeline, filt)));
      if (numSlabs > 0) { if (slabT) ApplySlabT(msl, si, si + 1, numSlabs); else ApplySlab(msl, slabIndex, slabIndex + 1, numSlabs); }
      if (maccum) ApplyAccOut(msl);
      if (numSlabs > 0 && !slabT && si == 0) { FILE* f = fopen("/tmp/slab.msl", "w"); fputs(msl, f); fclose(f); }
      id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithUTF8String:msl]
                                                 options:opts error:&err];
      free(msl);
      if (!lib) { fprintf(stderr, "library compile failed: %s\n", err.description.UTF8String); return 1; }
      if (compute) {
        kernF = [lib newFunctionWithName:@"compute_main"];
        if (!kernF) { fprintf(stderr, "kernel lookup failed\n"); return 1; }
        cps = [device newComputePipelineStateWithFunction:kernF error:&err];
        if (!cps) { fprintf(stderr, "compute pso failed: %s\n", err.description.UTF8String); return 1; }
      } else {
        vertF = [lib newFunctionWithName:@"vertex_main"];
        fragF = [lib newFunctionWithName:@"fragment_main"];
        if (!vertF || !fragF) { fprintf(stderr, "function lookup failed\n"); return 1; }
        MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
        pd.vertexFunction = vertF;
        pd.fragmentFunction = fragF;
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        if (depthMode != 2) {
          pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
        }
        if (maccum) {
          pd.colorAttachments[0].blendingEnabled = YES;
          pd.colorAttachments[0].rgbBlendOperation = MTLBlendOperationMax;
          pd.colorAttachments[0].alphaBlendOperation = MTLBlendOperationMax;
          pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
          pd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
        }
        pso = [device newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!pso) { fprintf(stderr, "pso failed: %s\n", err.description.UTF8String); return 1; }
        [psoArr addObject:pso];
      }
    }

    // Synthetic 512^3 volume. dataMode (argv[12]): 0 = z-slice gradient (highly
    // cache-local), 1 = xorshift noise (worst-case texture-cache locality,
    // closer to real CT data's per-texel variation).
    size_t total = (size_t)kW * kH * kD;
    uint8_t* host = malloc(total);
    if (dataMode) {
      uint32_t x = 0x12345678u;
      for (size_t i = 0; i < total; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        host[i] = (uint8_t)(x >> 24);
      }
    } else {
      for (size_t i = 0; i < total; i++) host[i] = (uint8_t)((i >> 10) & 0xff);
    }

    MTLTextureDescriptor* vd = [[MTLTextureDescriptor alloc] init];
    if (layoutMode) {
      vd.textureType = MTLTextureType2DArray;
      vd.arrayLength = kD;
      vd.depth = 1;
    } else {
      vd.textureType = MTLTextureType3D;
      vd.depth = kD;
    }
    vd.pixelFormat = MTLPixelFormatR8Unorm;
    vd.width = kW; vd.height = kH;
    vd.mipmapLevelCount = 1;
    vd.usage = MTLTextureUsageShaderRead;
    vd.storageMode = MTLStorageModePrivate;
    id<MTLTexture> volTex = [device newTextureWithDescriptor:vd];

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> cb0 = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb0 blitCommandEncoder];
    id<MTLBuffer> stageBuf = [device newBufferWithBytes:host length:total options:MTLResourceStorageModeShared];
    if (layoutMode) {
      for (int s = 0; s < kD; s++)
        [blit copyFromBuffer:stageBuf sourceOffset:(size_t)s * kW * kH
                sourceBytesPerRow:kW sourceBytesPerImage:kW*kH
                sourceSize:MTLSizeMake(kW, kH, 1) toTexture:volTex
                destinationSlice:s destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
    } else {
      [blit copyFromBuffer:stageBuf sourceOffset:0 sourceBytesPerRow:kW sourceBytesPerImage:kW*kH
              sourceSize:MTLSizeMake(kW, kH, kD) toTexture:volTex
              destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
    }
    [blit endEncoding];
    [cb0 commit];
    [cb0 waitUntilCompleted];
    free(host);

    // Render target.
    MTLTextureDescriptor* rtd = [[MTLTextureDescriptor alloc] init];
    rtd.textureType = MTLTextureType2D;
    rtd.pixelFormat = MTLPixelFormatBGRA8Unorm;
    rtd.width = kRT; rtd.height = kRT;
    rtd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderWrite;
    rtd.storageMode = MTLStorageModeShared;
    id<MTLTexture> rt = [device newTextureWithDescriptor:rtd];

    MTLTextureDescriptor* dd = [[MTLTextureDescriptor alloc] init];
    dd.textureType = MTLTextureType2D;
    dd.pixelFormat = MTLPixelFormatDepth32Float;
    dd.width = kRT; dd.height = kRT;
    dd.usage = MTLTextureUsageRenderTarget;
    dd.storageMode = MTLStorageModePrivate;
    id<MTLTexture> depth = [device newTextureWithDescriptor:dd];

    struct Uniforms u;
    // Real app camera (DICOMVolume scene, VTK_METAL_TEST_DUMP_UNIFORMS dump):
    // eye normalized in volume space, bounds physical mm, inverseVP NDC->phys.
    u.eye = (simd_float4){ -1.49527037f, -0.95243806f, 2.55352926f, 0.0f };
    u.boundsSize = (simd_float4){ 426.166f, 426.166f, 717.2f, 0.0f };
    const float invVP[16] = {
      0.23205078f, -6.974323e-09f, 0.13397458f, 1.2084004e-11f,
      -0.045822006f, 0.25178984f, 0.079366043f, -3.2157374e-12f,
      0.35811684f, 0.22810864f, -1.0292176f, -0.00056198676f,
      -0.12124427f, -0.03448515f, 0.8849802f, 0.00092758867f };
    memcpy(&u.invVP, invVP, sizeof(invVP));
    u.sampleDistMM = sampleDistMM;
    u._pad0 = 0.0f;
    u._pad1 = 0.0f;
    u.maxIter = maxIter;
    id<MTLBuffer> ubuf = [device newBufferWithBytes:&u length:sizeof(u) options:MTLResourceStorageModeShared];

    if (diag) {
      // Print what Metal actually receives for invVP (columns) so we can tell
      // whether the simd_float4x4 layout or the MSL multiply is the problem.
      for (int c = 0; c < 4; c++) {
        simd_float4 col = u.invVP.columns[c];
        fprintf(stderr, "invVP col%d = %.7f %.7f %.7f %.7f\n", c,
          col.x, col.y, col.z, col.w);
      }
      fprintf(stderr, "invVP raw = ");
      for (int i = 0; i < 16; i++) fprintf(stderr, "%.7f ", ((float*)&u.invVP)[i]);
      fprintf(stderr, "\n");
    }

    // Warmup.
    for (int f = 0; f < 10; f++) {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      if (compute) {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:cps];
        [enc setBuffer:ubuf offset:0 atIndex:0];
        [enc setTexture:volTex atIndex:0];
        [enc setTexture:rt atIndex:1];
        [enc setBytes:&kRT length:sizeof(int) atIndex:1];
        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake(kRT, kRT, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
      } else if (maccum) {
        for (int p = 0; p < ns; p++) {
          MTLRenderPassDescriptor* rpd = MakeRPDAction(rt, depth, depthMode, p == 0 ? MTLLoadActionClear : MTLLoadActionLoad);
          id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
          [enc setRenderPipelineState:[psoArr objectAtIndex:p]];
          [enc setFragmentBuffer:ubuf offset:0 atIndex:0];
          [enc setFragmentTexture:volTex atIndex:0];
          [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
          [enc endEncoding];
        }
      } else {
        MTLRenderPassDescriptor* rpd = MakeRPD(rt, depth, depthMode);
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
        [enc setRenderPipelineState:pso];
        [enc setFragmentBuffer:ubuf offset:0 atIndex:0];
        [enc setFragmentTexture:volTex atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
      }
      [cb commit];
      [cb waitUntilCompleted];
    }

    // Timed.
    double acc = 0.0;
    for (int f = 0; f < frames; f++) {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      if (compute) {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:cps];
        [enc setBuffer:ubuf offset:0 atIndex:0];
        [enc setTexture:volTex atIndex:0];
        [enc setTexture:rt atIndex:1];
        [enc setBytes:&kRT length:sizeof(int) atIndex:1];
        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake(kRT, kRT, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
      } else if (maccum) {
        for (int p = 0; p < ns; p++) {
          MTLRenderPassDescriptor* rpd = MakeRPDAction(rt, depth, depthMode, p == 0 ? MTLLoadActionClear : MTLLoadActionLoad);
          id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
          [enc setRenderPipelineState:[psoArr objectAtIndex:p]];
          [enc setFragmentBuffer:ubuf offset:0 atIndex:0];
          [enc setFragmentTexture:volTex atIndex:0];
          [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
          [enc endEncoding];
        }
      } else {
        MTLRenderPassDescriptor* rpd = MakeRPD(rt, depth, depthMode);
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
        [enc setRenderPipelineState:pso];
        [enc setFragmentBuffer:ubuf offset:0 atIndex:0];
        [enc setFragmentTexture:volTex atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
      }
      [cb commit];
      [cb waitUntilCompleted];
      double t = cb.GPUEndTime - cb.GPUStartTime;
      if (t > 0) acc += t;
    }
    fprintf(stderr, "METAL avg frame: %.3f ms\n", acc / frames * 1e3);

    // Sanity readback. The RT is BGRA8Unorm, so memory byte order is B,G,R,A
    // while the shader writes float4(R=iterLow, G=iterHigh, B=acc, A=1).
    // Memory byte0=B=acc, byte1=G=iterHigh, byte2=R=iterLow. The "true"
    // iteration count per pixel is byte2 + 256*byte1; meanB is byte0/255.
    // (A naive byte0+256*byte1 read mixes acc into the low byte and inflates
    // avgIter; GL's RGBA8 readback does not have this trap.)
    // Nonzero pixels are marched fragments. In diag mode the shader wrote
    // rayDir*0.5+0.5 instead (writes metal_gap_diag.ppm).
    uint8_t* pix = malloc((size_t)kRT * kRT * 4);
    [rt getBytes:pix bytesPerRow:kRT * 4 fromRegion:MTLRegionMake2D(0, 0, kRT, kRT) mipmapLevel:0];
    double sumB = 0.0;
    double lo = 0.0;
    double hi = 0.0;
    int nz = 0;
    for (size_t i = 0; i < (size_t)kRT * kRT * 4; i += 4) {
      sumB += pix[i + 2];
      lo += pix[i];
      hi += pix[i + 1];
      if (pix[i] + pix[i + 1] + pix[i + 2] > 0) nz++;
    }
    double npix = (double)kRT * kRT;
    double avgN = (hi / npix) * 256.0 + lo / npix;
    // The RT is BGRA8Unorm, so memory byte0=B(shader acc), byte1=G(shader
    // iterHigh), byte2=R(shader iterLow). True iteration count per pixel:
    // iter = byte2 + 256*byte1.
    double sumIter = 0.0;
    for (size_t i = 0; i < (size_t)kRT * kRT * 4; i += 4) {
      sumIter += pix[i + 2] + 256.0 * (double)pix[i + 1];
    }
    fprintf(stderr, "METAL readback: meanB=%.3f nonzero=%d/%d avgIter=%.1f (true=%.1f)\n",
      sumB / npix / 255.0, nz, kRT * kRT, avgN, sumIter / npix);
    const char* ppmName = diag ? "metal_gap_diag.ppm" : "metal_gap.ppm";
    FILE* fp = fopen(ppmName, "wb");
    if (fp) {
      fprintf(fp, "P6\n%d %d\n255\n", kRT, kRT);
      for (int i = 0; i < kRT * kRT; i++)
        fwrite(pix + i * 4, 1, 3, fp);
      fclose(fp);
    }
    free(pix);
  }
  return 0;
}
