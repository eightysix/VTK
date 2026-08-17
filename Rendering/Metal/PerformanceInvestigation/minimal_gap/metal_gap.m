// Minimal, self-contained Metal microbenchmark: the real DICOM scene's exact
// divergent perspective rays marching through a 512x512x1794 R8 3D texture
// with hardware trilinear filtering. Matches the app's VolumeMapperUniforms
// (CameraVolumePos, inverse view-projection, per-ray sample distance) so the
// raw-fetch march cost is identical to the production Metal renderer. The
// paired GL version (gl_gap) runs the same rays so the intrinsic Metal-vs-GL
// 3D sampler throughput gap can be measured in isolation.
//
// Build: clang -fobjc-arc -framework Metal -framework Foundation metal_gap.m -o metal_gap
// Run:   ./metal_gap [flags]
//
// Named flags (positional args are also accepted for backward compat):
//   --frames N        frames to time (default 10)
//   --maxiter N       safety loop bound; 0 yields zero iterations (default 99999)
//   --half            texture3d<half> + half accumulator (FP16 sampler return)
//   --depth MODE      0 = depth attached MTLStoreActionStore, 1 = DontCare,
//                     2 = no depth attachment (matches gl_gap)
//   --compute         replace the fragment pass with a compute kernel
//   --lod0            explicit level(0.0f) sample, no implicit LOD gradient
//   --fastmath 0/1    MTLCompileOptions.fastMathEnabled
//   --diag            ray-field diagnostic shader
//   --pipeline N      0 = serial, 2 = serial unroll, 3+ = latch unroll group N
//   --rt N            render-target size in px (default 400)
//   --sd F            physical sample distance mm (default 0.5)
//   --data N          volume data mode (default 0 = DICOM gradient)
//   --layout N        1 = texture2d_array sliced layout
//   --filter N        0 = linear, 1 = nearest
//   --div N           volume div (512/N per axis)
//   --slabs N         slab count (0 = single pass)
//   --slabindex N     single-slab render test
//   --slabt           z-tiling slab clamp (app ResolveNumSlabs model)
//   --maccum          accumulate slabs into the RT via blending
//   --muladd          mul-add loop variant
//   --kendt           mulAdd pass end-aligned at thi (vs integer next index)
//   --uniformslab     slab bounds via u.slabStart/u.slabEnd uniforms
//   --camera N        camera preset: 0 oblique, 1 axial(z), 2 coronal(y),
//                     3 sagittal(x), 4 oblique45
//   --composite 0/1   TF lookup + front-to-back over-composite (Airways II
//                     ramp + constant color (0, 0.605, 0.706), break at
//                     acc > 1 - 1/255, app TerminationImplementation parity)
//   --jitter 0/1      IGN jitter on the sample lattice (app parity)
//   --jitterblock N   jitter block size in px (default 1)
//   --clip 0/1        (0,0,1) near-plane ray clip (DICOM scene parity)
//   --preint F        opacity preintegration correction 1-(1-a)^F (default 1)
//   --blend over|max  maccum RT blend (default: over when composite, else max)
//
// Readback decodes (BGRA8Unorm RT): byte0=B(shader .b), byte1=G, byte2=R.
// MIP: B = max scalar, avgIter = byte2 + 256*byte1 = sample count.
// Composite: B = alpha (single pass) or accColor.b (maccum premultiplied),
// avgIter is garbage in maccum mode (color channels, not iterations).

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
  float slabStart;         // slab z-bounds in normalized volume units
  float slabEnd;
  int deadPath;
  float useComposite;      // TF lookup + front-to-back over-composite
  float useJittering;      // IGN jitter on the sample lattice (app parity)
  float useClipping;       // (0,0,1) near-plane ray clip (DICOM scene parity)
  int   jitterBlock;       // IGN jitter block size in px
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
  char expr[2048];
  int eo = 0;
  for (int k = 0; k < n; k += 2)
  {
    if (eo == 0)
      eo = snprintf(expr, sizeof(expr), "max(s%d, s%d)", k, k + 1);
    else
    {
      char tmp[2048];
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

// Bake the DICOM reference scene's "Airways II" transfer function into a
// 256-entry RGBA8 row (TestMetalScenes.h BuildDICOMVolumeScene): constant
// color (0, 0.605, 0.706); opacity points (-742.1,0)(-683,0.0493)(-481,0.2497)
// (-333.5,0) with x rescaled to the U8 volume domain: (hu+1024)*255/4095.
// `preint` applies the OpenGL composite-blend pre-integration correction
// 1-(1-a)^factor (vtkOpenGLVolumeOpacityTable::InternalUpdate; the Metal
// backend bakes the same into the TF texture on CPU). factor = sampleDist /
// unitDist; pass --preint to match the app's per-sample opacity (e.g. 1.25
// for 0.5mm SD over 0.4mm z spacing).
static void BakeTFTable(uint8_t* row, double preint)
{
  const double xs[4] = {
    (-742.1 + 1024.0) * 255.0 / 4095.0,
    (-683.0 + 1024.0) * 255.0 / 4095.0,
    (-481.0 + 1024.0) * 255.0 / 4095.0,
    (-333.5 + 1024.0) * 255.0 / 4095.0
  };
  const double ys[4] = { 0.0, 0.0493, 0.2497, 0.0 };
  for (int i = 0; i < 256; i++) {
    double x = (double)i;
    double a = 0.0;
    for (int k = 0; k < 3; k++) {
      if (x >= xs[k] && x <= xs[k + 1]) {
        double f = (xs[k + 1] > xs[k]) ? (x - xs[k]) / (xs[k + 1] - xs[k]) : 0.0;
        a = ys[k] + f * (ys[k + 1] - ys[k]);
        break;
      }
    }
    if (a > 0.0001 && preint > 0.0) a = 1.0 - pow(1.0 - a, preint);
    row[i * 4 + 0] = (uint8_t)(0.0 * 255.0 + 0.5);
    row[i * 4 + 1] = (uint8_t)(0.605 * 255.0 + 0.5);
    row[i * 4 + 2] = (uint8_t)(0.706 * 255.0 + 0.5);
    row[i * 4 + 3] = (uint8_t)(a * 255.0 + 0.5);
  }
}

// Composite-mode march bodies (TF lookup + front-to-back over-composite with
// the app's opacity threshold 1-1/255), mirroring MetalShaders.metal:
//   scalar = clamp(s) (scalarScale=1/scalarBias=0 for R8 normalized data)
//   c = tfTex.sample(tfSampler, float2(scalarNorm, 0.5))
//   w = 1 - acc; accColor += w*(c.rgb*c.a); acc = acc + w*c.a
//   break when acc > 1 - 1/255 (GL TerminationImplementation parity, break
//   WITHOUT clamping the accumulated opacity).
// The scalar variant breaks per sample (app marchVariant 0-2 divergence); the
// N-way unroll latches instead (marchOpaque/marchDone select, app marchVariant
// >= 3): the group's fetches stay unconditional so the pipeline stays full and
// only the accumulation is gated. `acc` is the outer alpha accumulator (keeps
// the existing readback expression), accColor is the premultiplied color.
static void BuildCompositeBody(char* body, size_t cap, int n, const char* lt, const char* lod)
{
  if (n <= 1)
  {
    snprintf(body, cap, R"MSL(
  float3 accColor = float3(0.0f);
  for (int i = 0; i < steps; i++) {
    if (currentT >= tExit - 1e-6f) break;
    %s s = volTex.sample(volSampler, evalPoint%s).r;
    n += 1.0f;
    float4 c = tfTex.sample(tfSampler, float2(clamp(%s(s), 0.0f, 1.0f), 0.5f));
    float w = 1.0f - acc;
    accColor += w * c.rgb * c.a;
    acc = acc + w * c.a;
    if (acc > 1.0f - 1.0f/255.0f) break;
    currentT += stepSize; texLocal += texStep; evalPoint += evalStep;
  }
)MSL", lt, lod, lt);
    return;
  }
  int off = 0;
  off += snprintf(body + off, cap - off,
    "\n  float3 accColor = float3(0.0f);\n"
    "  bool marchDone = false;\n"
    "  int i = 0;\n"
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
  off += snprintf(body + off, cap - off,
    "    {\n");
  for (int k = 0; k < n; k++)
    off += snprintf(body + off, cap - off,
      "      if (!marchDone) {\n"
      "        float4 c = tfTex.sample(tfSampler, float2(clamp(%s(s%d), 0.0f, 1.0f), 0.5f));\n"
      "        float w = 1.0f - acc;\n"
      "        accColor += w * c.rgb * c.a;\n"
      "        acc = acc + w * c.a;\n"
      "        if (acc > 1.0f - 1.0f/255.0f) marchDone = true;\n"
      "      }\n", lt, k);
  off += snprintf(body + off, cap - off, "    }\n");
  off += snprintf(body + off, cap - off,
    "    currentT += stepSize * %d.0f; texLocal += texStep * %d.0f;\n"
    "    evalPoint += evalStep * %d.0f; n += %d.0f;\n"
    "  }\n", n, n, n, n);
  snprintf(body + off, cap - off,
    "  for (; i < steps; i++) {\n"
    "    if (currentT >= tExit - 1e-6f) break;\n"
    "    %s s = volTex.sample(volSampler, evalPoint%s).r;\n"
    "    n += 1.0f;\n"
    "    if (!marchDone) {\n"
    "      float4 c = tfTex.sample(tfSampler, float2(clamp(%s(s), 0.0f, 1.0f), 0.5f));\n"
    "      float w = 1.0f - acc;\n"
    "      accColor += w * c.rgb * c.a;\n"
    "      acc = acc + w * c.a;\n"
    "      if (acc > 1.0f - 1.0f/255.0f) marchDone = true;\n"
    "    }\n"
    "    currentT += stepSize; texLocal += texStep; evalPoint += evalStep;\n"
    "  }\n", lt, lod, lt);
}

// texture2d_array variant of BuildMSL: identical rays, one 2D layer per Z
// slice, manual Z trilinear. `layoutMode` selects it (see main()).
static char* BuildSlicedMSL(bool halfSampler, int pipeline, const char* filt, bool composite)
{
  const char* lt = halfSampler ? "half" : "float";
  const char* ls = halfSampler ? "h" : "f";
  char body[16384];
  if (composite)
    BuildCompositeBody(body, sizeof(body), 1, lt, "");
  else
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
  float  slabStart;   // slab z-bounds (normalized volume units)
  float  slabEnd;
  int    deadPath;
  float  useComposite; // TF lookup + front-to-back over-composite
  float  useJittering; // IGN jitter on the sample lattice (app parity)
  float  useClipping;  // (0,0,1) near-plane ray clip (DICOM scene parity)
  int    jitterBlock;  // IGN jitter block size in px
};

fragment float4 fragment_main(VOut in [[stage_in]],
                              texture2d_array<%s> volTex [[texture(0)]],
                              texture2d<float> tfTex [[texture(1)]],
                              constant Uniforms& u [[buffer(0)]]) {
  constexpr sampler s2d(filter::linear, address::clamp_to_edge);
  constexpr sampler tfSampler(filter::nearest, address::clamp_to_edge);
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
  char* buf = malloc(strlen(fmt) + strlen(body) + 8192);
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
// `composite` replaces the MIP max-accumulate with the DICOM scene's TF
// lookup + front-to-back over-composite (BuildCompositeBody): scalar
// pipelines break per sample (app marchVariant 0-2), N-way unroll latches
// (app marchVariant >= 3 non-divergent). `acc` then holds the accumulated
// alpha and accColor the premultiplied color; single-pass output keeps the
// iteration count in R/G and the final alpha in B.
static char* BuildMSL(bool halfSampler, bool lod0, int pipeline, const char* filt, bool mulAdd, bool composite)
{
  const char* lt = halfSampler ? "half" : "float";
  const char* ls = halfSampler ? "h" : "f";
  const char* lod = lod0 ? ", level(0.0f)" : "";
  char body[16384];

  if (mulAdd)
  {
    if (composite)
      snprintf(body, sizeof(body), R"MSL(
  float3 evalBaseE = ctpOffset + (eye + rayDir * tStartRaw) * ctpScale;
  float3 evalStepE = rayDir * ctpScale * stepSize;
  float3 accColor = float3(0.0f);
  int j = kPass;
  for (int i = 0; i < steps; i++, j++) {
    float currentT = tStartRaw + float(j) * stepSize;
    if (currentT >= min(tExit, tExitRaw) - 1e-6f) break;
    float3 evalPoint = evalBaseE + float(j) * evalStepE;
    %s s = volTex.sample(volSampler, evalPoint%s).r;
    n += 1.0f;
    float4 c = tfTex.sample(tfSampler, float2(clamp(%s(s), 0.0f, 1.0f), 0.5f));
    float w = 1.0f - acc;
    accColor += w * c.rgb * c.a;
    acc = acc + w * c.a;
    if (acc > 1.0f - 1.0f/255.0f) break;
  }
)MSL", lt, lod, lt);
    else
      snprintf(body, sizeof(body), R"MSL(
  float3 evalBaseE = ctpOffset + (eye + rayDir * tStartRaw) * ctpScale;
  float3 evalStepE = rayDir * ctpScale * stepSize;
  int j = kPass;
  for (int i = 0; i < steps; i++, j++) {
    float currentT = tStartRaw + float(j) * stepSize;
    if (currentT >= min(tExit, tExitRaw) - 1e-6f) break;
    float3 evalPoint = evalBaseE + float(j) * evalStepE;
    acc = max(acc, volTex.sample(volSampler, evalPoint%s).r);
    n += 1.0f;
  }
)MSL", lod);
  }
  else if (composite)
  {
    if (pipeline == 3)
    {
      snprintf(body, sizeof(body), R"MSL(
  float3 accColor = float3(0.0f);
  for (int i = 0; i < steps; i++) {
    %s s = volTex.sample(volSampler, evalPoint%s).r;
    n += 1.0f;
    float4 c = tfTex.sample(tfSampler, float2(clamp(%s(s), 0.0f, 1.0f), 0.5f));
    float w = 1.0f - acc;
    accColor += w * c.rgb * c.a;
    acc = acc + w * c.a;
    if (acc > 1.0f - 1.0f/255.0f) break;
    currentT += stepSize; texLocal += texStep; evalPoint += evalStep;
  }
)MSL", lt, lod, lt);
    }
    else if (pipeline >= 2 && pipeline < 100)
    {
      BuildCompositeBody(body, sizeof(body), pipeline, lt, lod);
    }
    else
    {
      BuildCompositeBody(body, sizeof(body), 1, lt, lod);
    }
  }
  else if (pipeline == 1)
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
  else if (pipeline >= 2 && pipeline < 100)
  {
    BuildUnrollBody(body, sizeof(body), pipeline, lt, lod);
  }
  else if (pipeline == 100)
  {
    // Dead-path scheduling trick: 4 dummy samples in a uniform-gated branch
    // that never executes. The compiler must co-compile them (can't prove
    // deadPath >= 0 at compile time), which should trigger the same fast
    // fetch scheduling as the unroll variants — without any loop restructuring.
    snprintf(body, sizeof(body), R"MSL(
  for (int i = 0; i < steps; i++) {
    if (currentT >= tExit - 1e-6f) break;
    float s = volTex.sample(volSampler, evalPoint%s).r;
    acc = max(acc, (half)s);
    if (u.deadPath < 0) {
      float d0 = volTex.sample(volSampler, evalPoint * 0.5f%s).r;
      float d1 = volTex.sample(volSampler, evalPoint * 0.3f%s).r;
      float d2 = volTex.sample(volSampler, evalPoint * 0.7f%s).r;
      float d3 = volTex.sample(volSampler, evalPoint * 1.1f%s).r;
      acc = max(acc, (half)max(max(d0, d1), max(d2, d3)));
    }
    currentT += stepSize; texLocal += texStep; evalPoint += evalStep; n += 1.0f;
  }
)MSL", lod, lod, lod, lod, lod);
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
  float  slabStart;   // slab z-bounds (normalized volume units)
  float  slabEnd;
  int    deadPath;
  float  useComposite; // TF lookup + front-to-back over-composite
  float  useJittering; // IGN jitter on the sample lattice (app parity)
  float  useClipping;  // (0,0,1) near-plane ray clip (DICOM scene parity)
  int    jitterBlock;  // IGN jitter block size in px
};

fragment float4 fragment_main(VOut in [[stage_in]],
                              texture3d<%s> volTex [[texture(0)]],
                              texture2d<float> tfTex [[texture(1)]],
                              constant Uniforms& u [[buffer(0)]]) {
  constexpr sampler volSampler(filter::linear, address::clamp_to_edge);
  constexpr sampler tfSampler(filter::nearest, address::clamp_to_edge);
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
  char* buf = malloc(strlen(fmt) + strlen(body) + 8192);
  snprintf(buf, strlen(fmt) + strlen(body) + 64, fmt, lt, lt, ls, body);
  ApplyFilter(buf, filt);
  return buf;
}

// Diagnostic variant: writes rayDir*0.5+0.5 to RGB so the actual ray field can
// be compared against the fp64 reference (used to explain the avgIter gap).
// mulAdd mode (BuildMSL mulAdd): declare tStartRaw/tExitRaw/kPass right after
// tStart so the slab clamp's tStart realignment cannot contaminate them. Must
// run BEFORE ApplySlabT (the clamp is inserted later in the buffer).
static void ApplyMulAddDecls(char* buf)
{
  const char needle[] = "  float tStart = max(tEnter, 0.0f);\n";
  const char decls[] = "  float tStart = max(tEnter, 0.0f);\n"
    "  float tStartRaw = tStart;\n"
    "  float tExitRaw = tExit;\n"
    "  int kPass = 0;\n";
  char* p = buf;
  while ((p = strstr(p, needle)) != NULL)
  {
    memmove(p + strlen(decls), p + strlen(needle), strlen(p + strlen(needle)) + 1);
    memcpy(p, decls, strlen(decls));
    p += strlen(decls);
  }
}

// Slab clamp with jitter. The jitter lattice (jitterF/jitterT) is declared at
// the top of the clamp text (right after stepSize, before the slab math uses
// jitterT as the pass origin), mirroring the app's
// firstT = jitter + ceil((tStart - jitter)/stepSize)*stepSize with the lattice
// anchored per-pixel so every slab pass shares the same phase.
static void ApplySlabT(char* buf, int slabStart, int slabEnd, int numSlabs, bool mulAdd, bool kEndT, bool uniformSlab, bool jitter)
{
  char clamp[2048];
  const bool muladdPath = mulAdd;
  const char* base = jitter ? "jitterT" : "tStart";
  char jitterDecl[1024];
  if (jitter)
    snprintf(jitterDecl, sizeof(jitterDecl),
      "  float jitterF = u.useJittering > 0.5f ? fract(52.9829189f * fract(dot(floor((float2(in.position.xy) + 0.5f) / float(u.jitterBlock)) * float(u.jitterBlock) + 0.5f * float(u.jitterBlock), float2(0.06711056f, 0.00583715f)))) * stepSize : 0.0f;\n"
      "  float jitterT = jitterF + ceil((tStart - jitterF) / stepSize) * stepSize;\n");
  else
    jitterDecl[0] = '\0';
  char kpass[256];
  if (muladdPath)
    snprintf(kpass, sizeof(kpass), "    kPass = (int)kStartF;\n");
  else if (jitter)
    snprintf(kpass, sizeof(kpass),
      "    int kPass = (int)kStartF;\n"
      "    float tStartRaw = jitterT;\n"
      "    float tExitRaw = tExit;\n");
  else
    snprintf(kpass, sizeof(kpass),
      "    int kPass = (int)kStartF;\n"
      "    float tStartRaw = tStart;\n"
      "    float tExitRaw = tExit;\n");
  // mulAdd interior slabs break on the NEXT slab's integer start index
  // (kPass + i >= kNext) instead of an fp value, so consecutive ceil() calls
  // can never disagree by an index and the index-set union tiles single-pass
  // exactly. kNext and the (generous) tExit bound are computed from the raw
  // base so the union is contiguous by construction.
  char align[1024];
  if (muladdPath && !kEndT)
    snprintf(align, sizeof(align),
      "    float tNext = max(tStartRaw, max(t_s, t_e));\n"
      "    int kNext = (int)ceil(max((tNext - tStartRaw) / stepSize, 0.0f));\n"
      "    tExit = tStartRaw + float(kNext + 1) * stepSize;\n");
  else if (muladdPath)
    snprintf(align, sizeof(align),
      "    float kEnd = ceil(max((thi - tStartRaw) / stepSize, 0.0f));\n"
      "    tExit = tStartRaw + kEnd * stepSize;\n");
  else if (slabEnd >= numSlabs)
    snprintf(align, sizeof(align), "    tExit = thi;\n");
  else
    snprintf(align, sizeof(align),
      "    float kEnd = ceil(max((thi - tStart) / stepSize, 0.0f));\n"
      "    tExit = tStart + kEnd * stepSize;\n");
  if (uniformSlab)
    snprintf(clamp, sizeof(clamp),
      "float stepSize = u.sampleDistMM / max(physPerNorm, 1e-6f);\n"
      "%s"
      "    float t_s = (u.slabStart - u.eye.z) * inv.z;\n"
      "    float t_e = (u.slabEnd - u.eye.z) * inv.z;\n"
      "    float tlo = max(%s, min(t_s, t_e));\n"
      "    float thi = min(tExit, max(t_s, t_e));\n"
      "    float kStartF = ceil(max((tlo - %s) / stepSize, 0.0f));\n"
      "%s"
      "    tStart = %s + kStartF * stepSize;\n"
      "%s",
      jitterDecl, base, base, kpass, base, align);
  else
    snprintf(clamp, sizeof(clamp),
      "float stepSize = u.sampleDistMM / max(physPerNorm, 1e-6f);\n"
      "%s"
      "    float t_s = (%d.0f/%d.0f - u.eye.z) * inv.z;\n"
      "    float t_e = (%d.0f/%d.0f - u.eye.z) * inv.z;\n"
      "    float tlo = max(%s, min(t_s, t_e));\n"
      "    float thi = min(tExit, max(t_s, t_e));\n"
      "    float kStartF = ceil(max((tlo - %s) / stepSize, 0.0f));\n"
      "%s"
      "    tStart = %s + kStartF * stepSize;\n"
      "%s",
      jitterDecl, slabStart, numSlabs, slabEnd, numSlabs, base, base, kpass, base, align);
  const char needle[] = "float stepSize = u.sampleDistMM / max(physPerNorm, 1e-6f);\n";
  char* p = buf;
  size_t clen = strlen(clamp), nlen = strlen(needle);
  while ((p = strstr(p, needle)) != NULL)
  {
    memmove(p + clen, p + nlen, strlen(p + nlen) + 1);
    memcpy(p, clamp, clen);
    p += clen;
  }
  if (muladdPath && !kEndT)
  {
    const char ibneedle[] = "if (currentT >= min(tExit, tExitRaw) - 1e-6f) break;\n";
    const char ibrepl[] = "if (kPass + i >= kNext || currentT >= min(tExit, tExitRaw) - 1e-6f) break;\n";
    while ((p = strstr(buf, ibneedle)) != NULL)
    {
      memmove(p + strlen(ibrepl), p + strlen(ibneedle), strlen(p + strlen(ibneedle)) + 1);
      memcpy(p, ibrepl, strlen(ibrepl));
    }
  }
  if (!muladdPath)
  {
    const char wneedle[] = "  float currentT = tStart;\n"
      "  float3 texLocal = eye + rayDir * currentT;\n"
      "  float3 evalPoint = texLocal * ctpScale + ctpOffset;\n";
    const char warmup[] = "  float currentT = tStartRaw;\n"
      "  float3 texLocal = eye + rayDir * currentT;\n"
      "  float3 evalPoint = texLocal * ctpScale + ctpOffset;\n"
      "  for (int w = 0; w < kPass; w++) { currentT += stepSize; texLocal += texStep; evalPoint += evalStep; }\n";
    p = buf;
    while ((p = strstr(p, wneedle)) != NULL)
    {
      memmove(p + strlen(warmup), p + strlen(wneedle), strlen(p + strlen(wneedle)) + 1);
      memcpy(p, warmup, strlen(warmup));
      p += strlen(warmup);
    }
  }
  const char mneedle[] = "int maxSteps = max(1, int(ceil((tExit - tStart) / stepSize)));";
  const char mrepl[] = "int maxSteps = max(0, int(ceil((tExit - tStart) / stepSize)));";
  while ((p = strstr(buf, mneedle)) != NULL)
  {
    memmove(p + strlen(mrepl), p + strlen(mneedle), strlen(p + strlen(mneedle)) + 1);
    memcpy(p, mrepl, strlen(mrepl));
  }
  const char bneedle[] = "if (currentT >= tExit - 1e-6f) break;\n";
  const char brepl[] = "if (currentT >= min(tExit, tExitRaw) - 1e-6f) break;\n";
  while ((p = strstr(buf, bneedle)) != NULL)
  {
    memmove(p + strlen(brepl), p + strlen(bneedle), strlen(p + strlen(bneedle)) + 1);
    memcpy(p, brepl, strlen(brepl));
  }
  // mulAdd: the ApplyMulAddDecls tStartRaw capture must anchor at the jittered
  // pass origin (jitterT), not the raw box entry, so the evalBaseE lattice and
  // the kNext/kEnd align blocks use the same origin as tStart.
  if (muladdPath && jitter)
  {
    const char jn[] = "  float tStartRaw = tStart;\n";
    const char jr[] = "  float tStartRaw = jitterT;\n";
    while ((p = strstr(buf, jn)) != NULL)
    {
      memmove(p + strlen(jr), p + strlen(jn), strlen(p + strlen(jn)) + 1);
      memcpy(p, jr, strlen(jr));
    }
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

// Composite-mode maccum output: premultiplied accumulated color + alpha, the
// input the over-blend (ONE, ONE_MINUS_SRC_ALPHA) combine expects from each
// slab pass (app composite slab tiling parity). avgIter readback is garbage in
// this mode (same as ApplyAccOut for MIP).
static void ApplyCompositeOut(char* buf)
{
  const char needle[] = "float4(float(nc & 255u) / 255.0f, float((nc >> 8u) & 255u) / 255.0f, float(acc), 1.0f)";
  const char repl[] = "float4(accColor, acc)";
  char* p;
  while ((p = strstr(buf, needle)) != NULL)
  {
    memmove(p + strlen(repl), p + strlen(needle), strlen(p + strlen(needle)) + 1);
    memcpy(p, repl, strlen(repl));
  }
}

// DICOM scene clip plane (0,0,1) at the near z bound (TestMetalScenes.h:
// origin at bounds[4] = min z, normal (0,0,1)). Mirrors the app's setupVolumeRay
// clip (MetalShaders.metal 3965-3989): the plane sits on the box front so
// nothing is clipped, but the per-ray plane math cost is modeled. Runs BEFORE
// ApplyMulAddDecls so tStartRaw/tExitRaw capture the clipped entry/exit.
// `discard` is the source text that rejects the fragment when the ray misses
// the plane side entirely (fragment return vs compute outTex.write).
static void ApplyClip(char* buf, const char* discard)
{
  char clip[2048];
  snprintf(clip, sizeof(clip),
    "  float tStart = max(tEnter, 0.0f);\n"
    "  if (u.useClipping > 0.5f) {\n"
    "    float3 planeOrigin = float3(0.0f, 0.0f, 0.0f);\n"
    "    float3 planeNormal = float3(0.0f, 0.0f, 1.0f);\n"
    "    float3 entryPoint = eye + rayDir * tStart;\n"
    "    float3 exitPoint = eye + rayDir * tExit;\n"
    "    float startDistance = dot(planeNormal, planeOrigin - entryPoint);\n"
    "    float stopDistance = dot(planeNormal, planeOrigin - exitPoint);\n"
    "    if (startDistance > 0.0f && stopDistance > 0.0f) { %s }\n"
    "    float rayDotNormal = dot(rayDir, planeNormal);\n"
    "    if (rayDotNormal > 0.0f && startDistance > 0.0f) entryPoint += (startDistance / rayDotNormal) * rayDir;\n"
    "    if (rayDotNormal <= 0.0f && stopDistance > 0.0f) exitPoint += (stopDistance / rayDotNormal) * rayDir;\n"
    "    tStart = dot(entryPoint - eye, rayDir);\n"
    "    tExit = tStart + length(exitPoint - entryPoint);\n"
    "    if (tExit <= tStart) { %s }\n"
    "  }\n", discard, discard);
  const char needle[] = "  float tStart = max(tEnter, 0.0f);\n";
  char* p = buf;
  size_t clen = strlen(clip), nlen = strlen(needle);
  while ((p = strstr(p, needle)) != NULL)
  {
    memmove(p + clen, p + nlen, strlen(p + nlen) + 1);
    memcpy(p, clip, clen);
    p += clen;
  }
}

// IGN jitter (Jimenez 2014, app sampleIGNJitter): per-pixel deterministic
// value in [0,1) scaled by stepSize, lattice-aligned against tStart exactly
// like the app's firstT = jitter + ceil((tStart - jitter)/stepSize)*stepSize.
// Defines jitterF/jitterT right after stepSize. SINGLE-PASS ONLY: the slabT
// path has no lattice alignment here because ApplySlabT's clamp (which owns
// the jitterT decls, anchored to the slab-clamped tStart) must run first;
// main() gates this on !slabT. For mulAdd the tStartRaw decl (ApplyMulAddDecls)
// is re-pointed at jitterT so the absolute lattice index j counts from the
// jittered origin. `coord` is the per-pixel screen coordinate expression
// ("float2(in.position.xy) + 0.5f" for fragment, "float2(gid) + 0.5f" for
// compute).
static void ApplyJitterDecl(char* buf, const char* coord, bool mulAdd)
{
  char jit[1536];
  snprintf(jit, sizeof(jit),
    "float stepSize = u.sampleDistMM / max(physPerNorm, 1e-6f);\n"
    "  float jitterF = u.useJittering > 0.5f ? fract(52.9829189f * fract(dot(floor((%s) / float(u.jitterBlock)) * float(u.jitterBlock) + 0.5f * float(u.jitterBlock), float2(0.06711056f, 0.00583715f)))) * stepSize : 0.0f;\n"
    "  float jitterT = jitterF + ceil((tStart - jitterF) / stepSize) * stepSize;\n", coord);
  const char needle[] = "float stepSize = u.sampleDistMM / max(physPerNorm, 1e-6f);\n";
  char* p = buf;
  size_t clen = strlen(jit), nlen = strlen(needle);
  while ((p = strstr(p, needle)) != NULL)
  {
    memmove(p + clen, p + nlen, strlen(p + nlen) + 1);
    memcpy(p, jit, clen);
    p += clen;
  }
  if (mulAdd)
  {
    const char rneedle[] = "  float tStartRaw = tStart;\n";
    const char rrepl[] = "  float tStartRaw = jitterT;\n";
    while ((p = strstr(buf, rneedle)) != NULL)
    {
      memmove(p + strlen(rrepl), p + strlen(rneedle), strlen(p + strlen(rneedle)) + 1);
      memcpy(p, rrepl, strlen(rrepl));
    }
  }
}

// Single-pass (non-slabT) jitter: start the march at jitterT. In slabT mode
// ApplySlabT's warmup already starts from tStartRaw (== jitterT when jitter
// is on), so this is a no-op there (the needle no longer exists).
static void ApplyJitterStart(char* buf)
{
  const char needle[] = "  float currentT = tStart;\n";
  const char repl[] = "  float currentT = jitterT;\n";
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
  float  slabStart;   // slab z-bounds (normalized volume units)
  float  slabEnd;
  int    deadPath;
  float  useComposite; // TF lookup + front-to-back over-composite
  float  useJittering; // IGN jitter on the sample lattice (app parity)
  float  useClipping;  // (0,0,1) near-plane ray clip (DICOM scene parity)
  int    jitterBlock;  // IGN jitter block size in px
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
  char* buf = malloc(strlen(fmt) + 8192);
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
static char* BuildComputeMSL(bool halfSampler, const char* filt, bool composite)
{
  const char* lt = halfSampler ? "half" : "float";
  const char* ls = halfSampler ? "h" : "f";
  char body[8192];
  if (composite)
    snprintf(body, sizeof(body),
      "  float3 accColor = float3(0.0f);\n"
      "  for (int i = 0; i < min(u.maxIter, maxSteps); i++) {\n"
      "    if (currentT >= tExit - 1e-6f) break;\n"
      "    %s s = volTex.sample(volSampler, evalPoint).r;\n"
      "    n += 1.0f;\n"
      "    float4 c = tfTex.sample(tfSampler, float2(clamp(%s(s), 0.0f, 1.0f), 0.5f));\n"
      "    float w = 1.0f - acc;\n"
      "    accColor += w * c.rgb * c.a;\n"
      "    acc = acc + w * c.a;\n"
      "    if (acc > 1.0f - 1.0f/255.0f) break;\n"
      "    currentT += stepSize; texLocal += texStep; evalPoint += evalStep;\n"
      "  }\n", lt, lt);
  else
    snprintf(body, sizeof(body),
      "  for (int i = 0; i < min(u.maxIter, maxSteps); i++) {\n"
      "    if (currentT >= tExit - 1e-6f) break;\n"
      "    acc = max(acc, volTex.sample(volSampler, evalPoint).r);\n"
      "    currentT += stepSize; texLocal += texStep; evalPoint += evalStep;\n"
      "    n += 1.0f;\n"
      "  }\n");
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
  float  slabStart;   // slab z-bounds (normalized volume units)
  float  slabEnd;
  int    deadPath;
  float  useComposite; // TF lookup + front-to-back over-composite
  float  useJittering; // IGN jitter on the sample lattice (app parity)
  float  useClipping;  // (0,0,1) near-plane ray clip (DICOM scene parity)
  int    jitterBlock;  // IGN jitter block size in px
};

kernel void compute_main(uint2 gid [[thread_position_in_grid]],
                         texture3d<%s> volTex [[texture(0)]],
                         texture2d<float> tfTex [[texture(1)]],
                         texture2d<float, access::write> outTex [[texture(2)]],
                         constant Uniforms& u [[buffer(0)]],
                         constant int& rtSize [[buffer(1)]]) {
  constexpr sampler volSampler(filter::linear, address::clamp_to_edge);
  constexpr sampler tfSampler(filter::nearest, address::clamp_to_edge);
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
  %s
  uint nc = uint(n);
  outTex.write(float4(float(nc & 255u) / 255.0f, float((nc >> 8u) & 255u) / 255.0f, float(acc), 1.0f), gid);
}
)MSL";
  char* buf = malloc(strlen(fmt) + strlen(body) + 8192);
  snprintf(buf, strlen(fmt) + strlen(body) + 64, fmt, lt, lt, ls, body);
  ApplyFilter(buf, filt);
  return buf;
}

static int IntArg(int argc, const char** argv, const char* name, int def)
{
  for (int i = 1; i < argc - 1; i++)
    if (strcmp(argv[i], name) == 0) return atoi(argv[i + 1]);
  return def;
}

static float FloatArg(int argc, const char** argv, const char* name, float def)
{
  for (int i = 1; i < argc - 1; i++)
    if (strcmp(argv[i], name) == 0) return (float)atof(argv[i + 1]);
  return def;
}

static const char* StrArg(int argc, const char** argv, const char* name, const char* def)
{
  for (int i = 1; i < argc - 1; i++)
    if (strcmp(argv[i], name) == 0) return argv[i + 1];
  return def;
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
    int mulAdd = 0;
    int kEndT = 0;
    int uniformSlab = 0;
    int camera = 0;
    int composite = 0;
    int jitter = 0;
    int jitterBlock = 1;
    int clip = 0;
    float preint = 1.0f;
    int blendOver = -1;   // -1 = auto: over when composite else max
    if (argc > 1 && argv[1][0] == '-' && argv[1][1] == '-')
    {
      frames = IntArg(argc, argv, "--frames", frames);
      maxIter = IntArg(argc, argv, "--maxiter", maxIter);
      halfSampler = IntArg(argc, argv, "--half", halfSampler);
      depthMode = IntArg(argc, argv, "--depth", depthMode);
      compute = IntArg(argc, argv, "--compute", compute);
      lod0 = IntArg(argc, argv, "--lod0", lod0);
      fastMath = IntArg(argc, argv, "--fastmath", fastMath);
      diag = IntArg(argc, argv, "--diag", diag);
      pipeline = IntArg(argc, argv, "--pipeline", pipeline);
      kRT = IntArg(argc, argv, "--rt", kRT);
      sampleDistMM = FloatArg(argc, argv, "--sd", sampleDistMM);
      dataMode = IntArg(argc, argv, "--data", dataMode);
      layoutMode = IntArg(argc, argv, "--layout", layoutMode);
      const char* filt = StrArg(argc, argv, "--filter", NULL);
      if (filt) filterMode = strcmp(filt, "nearest") == 0 ? 1 : 0;
      volDiv = IntArg(argc, argv, "--div", volDiv);
      numSlabs = IntArg(argc, argv, "--slabs", numSlabs);
      slabIndex = IntArg(argc, argv, "--slabindex", slabIndex);
      slabT = IntArg(argc, argv, "--slabt", slabT);
      maccum = IntArg(argc, argv, "--maccum", maccum);
      mulAdd = IntArg(argc, argv, "--muladd", mulAdd);
      kEndT = IntArg(argc, argv, "--kendt", kEndT);
      uniformSlab = IntArg(argc, argv, "--uniformslab", uniformSlab);
      camera = IntArg(argc, argv, "--camera", camera);
      composite = IntArg(argc, argv, "--composite", composite);
      jitter = IntArg(argc, argv, "--jitter", jitter);
      jitterBlock = IntArg(argc, argv, "--jitterblock", jitterBlock);
      clip = IntArg(argc, argv, "--clip", clip);
      preint = FloatArg(argc, argv, "--preint", preint);
      const char* blend = StrArg(argc, argv, "--blend", NULL);
      if (blend) blendOver = strcmp(blend, "over") == 0 ? 1 : 0;
    }
    else
    {
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
      if (argc > 20) mulAdd = atoi(argv[20]);
      if (argc > 21) kEndT = atoi(argv[21]);
      if (argc > 22) uniformSlab = atoi(argv[22]);
      if (argc > 23) camera = atoi(argv[23]);
    }
    if (volDiv > 1) { kW = 512 / volDiv; kH = 512 / volDiv; kD = 1794 / volDiv; }
    if (blendOver < 0) blendOver = composite ? 1 : 0;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    fprintf(stderr, "Metal device: %s\n", device.name.UTF8String);
    fprintf(stderr, "volume %dx%dx%d R8, rt %dx%d, frames %d, maxIter %d, halfSampler=%d, depthMode=%d, compute=%d, lod0=%d, fastMath=%d, diag=%d, pipeline=%d, sampleDistMM=%.1f, dataMode=%d, layoutMode=%d, filterMode=%d, volDiv=%d, slab=%d/%d, slabT=%d, maccum=%d, mulAdd=%d, kEndT=%d, uniformSlab=%d, camera=%d, composite=%d, jitter=%d(jb=%d), clip=%d, preint=%.2f, blend=%s\n",
      kW, kH, kD, kRT, kRT, frames, maxIter, halfSampler, depthMode, compute, lod0, fastMath, diag, pipeline, sampleDistMM, dataMode, layoutMode, filterMode, volDiv, slabIndex, numSlabs, slabT, maccum, mulAdd, kEndT, uniformSlab, camera, composite, jitter, jitterBlock, clip, preint, blendOver ? "over" : "max");

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
    const char* discardFrag = "return float4(0.0f, 0.0f, 0.0f, 1.0f);";
    const char* discardKern = "outTex.write(float4(0.0f, 0.0f, 0.0f, 1.0f), gid); return;";
    for (int si = 0; si < ns; si++) {
      char* msl = diag ? BuildDiagMSL()
                       : (compute ? BuildComputeMSL(halfSampler != 0, filt, composite != 0)
                                  : (layoutMode ? BuildSlicedMSL(halfSampler != 0, pipeline, filt, composite != 0)
                                                : BuildMSL(halfSampler != 0, lod0 != 0, pipeline, filt, mulAdd != 0, composite != 0)));
      if (clip) ApplyClip(msl, compute ? discardKern : discardFrag);
      if (mulAdd) ApplyMulAddDecls(msl);
      if (jitter && !slabT) ApplyJitterDecl(msl, compute ? "float2(gid) + 0.5f" : "float2(in.position.xy) + 0.5f", mulAdd != 0);
      if (numSlabs > 0) { if (slabT) { if (uniformSlab) ApplySlabT(msl, 0, 1, 1, mulAdd != 0, kEndT != 0, 1, jitter != 0); else ApplySlabT(msl, maccum ? si : slabIndex, maccum ? si + 1 : slabIndex + 1, numSlabs, mulAdd != 0, kEndT != 0, 0, jitter != 0); } else ApplySlab(msl, slabIndex, slabIndex + 1, numSlabs); }
      if (jitter && !slabT) ApplyJitterStart(msl);
      if (maccum) { if (composite) ApplyCompositeOut(msl); else ApplyAccOut(msl); }
      if (si == 0) { FILE* f = fopen(numSlabs > 0 ? "/tmp/slab.msl" : "/tmp/single.msl", "w"); fputs(msl, f); fclose(f); }
      if (maccum && si == numSlabs - 1) { FILE* f = fopen("/tmp/lastslab.msl", "w"); fputs(msl, f); fclose(f); }
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
          if (blendOver) {
            // Composite slab tiling parity: each pass outputs premultiplied
            // (color, alpha), combined front-to-back with (ONE, ONE_MINUS_SRC_ALPHA)
            // on both RGB and alpha (app composite slab combine).
            pd.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
            pd.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
            pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
            pd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            pd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
            pd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
          } else {
            pd.colorAttachments[0].rgbBlendOperation = MTLBlendOperationMax;
            pd.colorAttachments[0].alphaBlendOperation = MTLBlendOperationMax;
            pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
            pd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
          }
        }
        pso = [device newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!pso) { fprintf(stderr, "pso failed: %s\n", err.description.UTF8String); return 1; }
        [psoArr addObject:pso];
        if (uniformSlab && si == 0) break;
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

    // Transfer-function texture: DICOM "Airways II" 256x1 RGBA8 (BakeTFTable).
    // Uploaded unconditionally so texture(1) is always bound; the shader only
    // samples it in composite mode.
    uint8_t tfRow[256 * 4];
    BakeTFTable(tfRow, preint);
    MTLTextureDescriptor* tfd = [[MTLTextureDescriptor alloc] init];
    tfd.textureType = MTLTextureType2D;
    tfd.pixelFormat = MTLPixelFormatRGBA8Unorm;
    tfd.width = 256; tfd.height = 1;
    tfd.mipmapLevelCount = 1;
    tfd.usage = MTLTextureUsageShaderRead;
    tfd.storageMode = MTLStorageModePrivate;
    id<MTLTexture> tfTex = [device newTextureWithDescriptor:tfd];
    id<MTLCommandBuffer> cb1 = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit1 = [cb1 blitCommandEncoder];
    id<MTLBuffer> tfBuf = [device newBufferWithBytes:tfRow length:sizeof(tfRow) options:MTLResourceStorageModeShared];
    [blit1 copyFromBuffer:tfBuf sourceOffset:0 sourceBytesPerRow:sizeof(tfRow) sourceBytesPerImage:0
            sourceSize:MTLSizeMake(256, 1, 1) toTexture:tfTex
            destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit1 endEncoding];
    [cb1 commit];
    [cb1 waitUntilCompleted];

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
    u.slabStart = 0.0f;
    u.slabEnd = 1.0f;
  u.deadPath = 0;
    u.useComposite = composite ? 1.0f : 0.0f;
    u.useJittering = jitter ? 1.0f : 0.0f;
    u.useClipping = clip ? 1.0f : 0.0f;
    u.jitterBlock = jitterBlock;

    // Camera presets: 0=original oblique, 1=axial(z), 2=coronal(y), 3=sagittal(x), 4=oblique45
    if (camera == 1) {
        u.eye = (simd_float4){0.5f, 0.5f, 2.5f, 0.0f};
        const float iv[16] = {
            4.14213562e-01f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f,
            0.00000000e+00f, 4.14213562e-01f, 0.00000000e+00f, 0.00000000e+00f,
            -1.06540435e+04f, -1.06540435e+04f, -8.96491035e+04f, -4.99995000e+01f,
            1.06542565e+04f, 1.06542565e+04f, 8.96498965e+04f, 5.00005000e+01f};
        memcpy(&u.invVP, iv, sizeof(iv));
    } else if (camera == 2) {
        u.eye = (simd_float4){0.5f, 2.5f, 0.5f, 0.0f};
        const float iv[16] = {
            -4.14213562e-01f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f,
            0.00000000e+00f, 0.00000000e+00f, 4.14213562e-01f, 0.00000000e+00f,
            -1.06540435e+04f, -5.32702173e+04f, -1.79298207e+04f, -4.99995000e+01f,
            1.06542565e+04f, 5.32702827e+04f, 1.79301793e+04f, 5.00005000e+01f};
        memcpy(&u.invVP, iv, sizeof(iv));
    } else if (camera == 3) {
        u.eye = (simd_float4){2.5f, 0.5f, 0.5f, 0.0f};
        const float iv[16] = {
            0.00000000e+00f, 4.14213562e-01f, 0.00000000e+00f, 0.00000000e+00f,
            0.00000000e+00f, 0.00000000e+00f, 4.14213562e-01f, 0.00000000e+00f,
            -5.32702173e+04f, -1.06540435e+04f, -1.79298207e+04f, -4.99995000e+01f,
            5.32702827e+04f, 1.06542565e+04f, 1.79301793e+04f, 5.00005000e+01f};
        memcpy(&u.invVP, iv, sizeof(iv));
    } else if (camera == 4) {
        u.eye = (simd_float4){1.5f, 1.5f, 1.5f, 0.0f};
        const float iv[16] = {
            3.56091876e-01f, -1.36776692e-12f, -2.11592653e-01f, -2.13967351e-15f,
            -9.62561335e-02f, 3.68872156e-01f, -1.61990630e-01f, -2.42710924e-15f,
            -3.19621304e+04f, -3.19621304e+04f, -5.37894621e+04f, -4.99995000e+01f,
            3.19623147e+04f, 3.19623147e+04f, 5.37897723e+04f, 5.00005000e+01f};
        memcpy(&u.invVP, iv, sizeof(iv));
    }

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
        [enc setTexture:tfTex atIndex:1];
        [enc setTexture:rt atIndex:2];
        [enc setBytes:&kRT length:sizeof(int) atIndex:1];
        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake(kRT, kRT, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
      } else if (maccum) {
        for (int p = 0; p < ns; p++) {
          if (uniformSlab) {
            u.slabStart = (float)p / numSlabs;
            u.slabEnd = (float)(p + 1) / numSlabs;
          }
          MTLRenderPassDescriptor* rpd = MakeRPDAction(rt, depth, depthMode, p == 0 ? MTLLoadActionClear : MTLLoadActionLoad);
          id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
          [enc setRenderPipelineState:[psoArr objectAtIndex:uniformSlab ? 0 : p]];
          if (uniformSlab) [enc setFragmentBytes:&u length:sizeof(u) atIndex:0];
          else [enc setFragmentBuffer:ubuf offset:0 atIndex:0];
          [enc setFragmentTexture:volTex atIndex:0];
          [enc setFragmentTexture:tfTex atIndex:1];
          [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
          [enc endEncoding];
        }
      } else {
        MTLRenderPassDescriptor* rpd = MakeRPD(rt, depth, depthMode);
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
        [enc setRenderPipelineState:pso];
        [enc setFragmentBuffer:ubuf offset:0 atIndex:0];
        [enc setFragmentTexture:volTex atIndex:0];
          [enc setFragmentTexture:tfTex atIndex:1];
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
        [enc setTexture:tfTex atIndex:1];
        [enc setTexture:rt atIndex:2];
        [enc setBytes:&kRT length:sizeof(int) atIndex:1];
        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake(kRT, kRT, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
      } else if (maccum) {
        for (int p = 0; p < ns; p++) {
          if (uniformSlab) {
            u.slabStart = (float)p / numSlabs;
            u.slabEnd = (float)(p + 1) / numSlabs;
          }
          MTLRenderPassDescriptor* rpd = MakeRPDAction(rt, depth, depthMode, p == 0 ? MTLLoadActionClear : MTLLoadActionLoad);
          id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
          [enc setRenderPipelineState:[psoArr objectAtIndex:uniformSlab ? 0 : p]];
          if (uniformSlab) [enc setFragmentBytes:&u length:sizeof(u) atIndex:0];
          else [enc setFragmentBuffer:ubuf offset:0 atIndex:0];
          [enc setFragmentTexture:volTex atIndex:0];
          [enc setFragmentTexture:tfTex atIndex:1];
          [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
          [enc endEncoding];
        }
      } else {
        MTLRenderPassDescriptor* rpd = MakeRPD(rt, depth, depthMode);
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
        [enc setRenderPipelineState:pso];
        [enc setFragmentBuffer:ubuf offset:0 atIndex:0];
        [enc setFragmentTexture:volTex atIndex:0];
          [enc setFragmentTexture:tfTex atIndex:1];
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
    fprintf(stderr, "METAL readback: meanB=%.3f nonzero=%d/%d avgIter=%.1f (true=%.1f) sumIter=%.0f\n",
      sumB / npix / 255.0, nz, kRT * kRT, avgN, sumIter / npix, sumIter);
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
