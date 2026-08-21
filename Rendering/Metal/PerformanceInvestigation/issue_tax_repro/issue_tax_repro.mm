// issue_tax_repro.mm — minimal repro of the RESIDUAL Metal-vs-GL gap that
// remains AFTER the do-while back-edge fix: at 1024x1024, SD 0.5 (step
// 0.0005), the plain 3D do-while march runs ~2-3% slower on Metal than GL.
//
// Ablation pinning (see ../divergent_tail/README.md):
//   - ALU-only loop (no fetch):        Metal FASTER (0.84)
//   - L1-resident fetches, no break:   Metal +3%   <- this file's second pair
//   - full streaming march:            Metal +2-3% <- this file's first pair
// => the residual is a per-tap sampler ISSUE tax, visible even when every
// tap hits one L1-resident texel, partially hidden under DRAM latency in
// the full march. RG8 pair-packing halves tap count and already mitigates
// it where it matters (2048-class), but the 3D path cannot go below one
// tap per sample.
//
// Fair timing: wall clock with per-frame drain on BOTH sides (glFinish /
// waitUntilCompleted). GPU-timestamp modes are biased here — GL's
// GL_TIME_ELAPSED ends before its tile-store drains.
//
// Follow-up bisection (see README): pairs l1_ce / l1_near / l1_ce_near /
// l1_read / l1_x2 / l1_rep / l1_pix ablate sampler specialization, filter
// mode, address mode, coordinate space and the raw-load path. Result: the
// tax is the sampler front-end's fixed per-tap rate (both APIs agree within
// 3%); Metal's read() escape hatch runs the same warp-uniform L1-resident
// traffic ~4x faster but cannot help a real march.

#include <Metal/Metal.h>
#include <OpenGL/gl3.h>
#include <OpenGL/OpenGL.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <string>
#include <vector>
#include <cmath>

static int kRT = 1024;
static int kFrames = 15;
static float kStep = 0.0005f; // SD 0.5 — the pathological regime
static int kWarmup = 30; // 10 was insufficient: GL's driver keeps re-JITting
                         // fresh programs into the first timed blocks (see README)
static int useKnobs = 1;             // argv[4]: 1 = 3.2+fastMath (main-harness options)
static const int kVolX = 512, kVolY = 512, kVolZ = 1794;

static std::vector<uint8_t> makeVolume()
{
  std::vector<uint8_t> v((size_t)kVolX * kVolY * kVolZ);
  const float cx = (kVolX - 1) * 0.5f, cy = (kVolY - 1) * 0.5f, cz = (kVolZ - 1) * 0.5f;
  const float r = std::min(std::min(kVolX, kVolY), kVolZ) * 0.22f;
  uint32_t s = 12345u;
  for (size_t i = 0; i < v.size(); ++i)
  {
    s = s * 1664525u + 1013904223u;
    const size_t z = i / ((size_t)kVolX * kVolY);
    const size_t r0 = i % ((size_t)kVolX * kVolY);
    const size_t y = r0 / kVolX;
    const size_t x = r0 % kVolX;
    const float dx = (float)x - cx, dy = (float)y - cy, dz = (float)z - cz;
    const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
    v[i] = (d < r) ? (uint8_t)(60 + (s >> 24) % 150) : 0;
  }
  return v;
}

// ---------------------------- GLSL ----------------------------
static const char* kGLVertSrc = R"(#version 410 core
layout(location=0) in vec2 aPos;
out vec2 vUV;
void main() {
  vUV = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

static const char* kGLFragSrc = R"(#version 410 core
in vec2 vUV;
out vec4 outColor;
uniform sampler3D volumeTex;
uniform sampler2DArray volumeArr;
uniform float uStep;
uniform vec3 uDims;
void main() {
  vec2 ndc = vUV * 2.0 - 1.0;
  vec3 eye = vec3(0.5, 0.5, -0.35);
  vec3 dir = normalize(vec3(ndc * 2.5, 1.0));
  float ca = cos(0.35), sa = sin(0.35);
  dir = vec3(ca * dir.x + sa * dir.z, dir.y, -sa * dir.x + ca * dir.z);
  vec3 inv = 1.0 / dir;
  vec3 t0 = (vec3(0.0) - eye) * inv;
  vec3 t1 = (vec3(1.0) - eye) * inv;
  float tEnter = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
  float tExit  = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
  if (tExit <= 0.0 || tEnter >= tExit) { outColor = vec4(0.0); return; }
  int steps = max(1, int(ceil((tExit - tEnter) / uStep)));
  vec3 base = eye + dir * (tEnter + 0.5 * uStep);
  vec3 d = dir * uStep;
  float acc = 0.0;
  float alpha = 0.0;
  int done = 0;
#ifdef L1ARR
  // Frozen coordinate on the slice-array twin (single bilinear tap).
  for (int i = 0; i < steps; ++i) {
    acc += textureLod(volumeArr, vec3(base.xy, float(int(uDims.z) / 2)), 0.0).r;
    done = i + 1;
  }
#elif defined(MARCH2TA)
  // V23-shape: two-tap slice-array march, do-while back-edge exit —
  // identical math to the Metal variant.
  int i = 0;
  do {
    vec3 c = base + float(i) * d;
    float g = clamp(c.z, 0.0, 1.0) * uDims.z - 0.5;
    float flr = floor(g);
    float bl = max(flr, 0.0);
    int l0 = int(min(bl, uDims.z - 1.0));
    int l1 = min(l0 + 1, int(uDims.z) - 1);
    float fz = clamp(g - bl, 0.0, 1.0);
    float s0 = textureLod(volumeArr, vec3(c.xy, float(l0)), 0.0).r;
    float s1 = textureLod(volumeArr, vec3(c.xy, float(l1)), 0.0).r;
    float s = mix(s0, s1, fz);
    float o = s * 1.0;
    float w = 1.0 - alpha;
    acc += w * o;
    alpha += w * o;
    ++i;
    done = i;
  } while (i < steps && alpha <= 0.9);
#elif defined(MARCHT)
  // transmittance form — identical arithmetic to the Metal variant 13/14
  int i = 0;
  float T = 1.0;
  do {
    float s = textureLod(volumeTex, base + float(i) * d, 0.0).r;
    float o = s * 1.0;
    acc += T * o;
    T -= T * o;
    ++i;
    done = i;
  } while (i < steps && T > 0.1);
#elif defined(L1FETCH)
  // Frozen coordinate: every tap hits the same L1-resident texel — pure
  // sampler issue rate, zero DRAM streaming.
  for (int i = 0; i < steps; ++i) {
    acc += textureLod(volumeTex, base, 0.0).r;
    done = i + 1;
  }
#elif defined(L1READ)
  // Frozen coordinate via texelFetch — load path with no sampler unit.
  for (int i = 0; i < steps; ++i) {
    acc += texelFetch(volumeTex, ivec3(base * uDims), 0).r;
    done = i + 1;
  }
#elif defined(L1X2)
  // Two independent taps per iteration (quarter-texel offset).
  float acc0 = 0.0, acc1 = 0.0;
  float ox = 0.25 / uDims.x;
  for (int i = 0; i < steps; ++i) {
    acc0 += textureLod(volumeTex, base, 0.0).r;
    acc1 += textureLod(volumeTex, base + vec3(ox, 0.0, 0.0), 0.0).r;
    done = i + 1;
  }
  acc = acc0 + acc1;
#else
  // The shipped fix: back-edge exit (do-while). This is the shape that
  // reaches GL parity everywhere else; the residual lives HERE.
  int i = 0;
  do {
    float s = textureLod(volumeTex, base + float(i) * d, 0.0).r;
    float o = s * 1.0;
    float w = 1.0 - alpha;
    acc += w * o;
    alpha += w * o;
    ++i;
    done = i;
  } while (i < steps && alpha <= 0.9);
#endif
  outColor = vec4(acc / float(steps), float(done) / 255.0, 0.0, 1.0);
}
)";

struct GLState
{
  CGLContextObj ctx = nullptr;
  GLuint fbo = 0, colorTex = 0, vao = 0, vbo = 0, volTex = 0, volArrTex = 0;
  GLuint progMarch = 0, progL1 = 0, progL1Read = 0, progL1X2 = 0;
  GLuint progL1Arr = 0, progM2TA = 0, progMARCHT = 0;
};

// variant ids shared with the pair table in main():
// 0 march31, 1 l1fetch, 2 l1_ce (GL twin == 1), 3 l1_near (twin == 1, NEAREST
// state), 4 l1_ce_near (twin == 3), 5 l1read (texelFetch), 6 l1x2.
static bool compileGL(GLState& s, GLuint* prog, int variant)
{
  const char* versionEnd = strstr(kGLFragSrc, "\n");
  std::string src;
  src.append(kGLFragSrc, versionEnd - kGLFragSrc + 1);
  if (variant == 1 || variant == 2 || variant == 3 || variant == 4) src += "#define L1FETCH 1\n";
  if (variant == 5) src += "#define L1READ 1\n";
  if (variant == 6) src += "#define L1X2 1\n";
  if (variant == 11) src += "#define L1ARR 1\n";
  if (variant == 12) src += "#define MARCH2TA 1\n";
  if (variant == 13 || variant == 14) src += "#define MARCHT 1\n";
  src += versionEnd + 1;
  GLuint vs = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vs, 1, &kGLVertSrc, nullptr);
  glCompileShader(vs);
  GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
  const char* srcC = src.c_str();
  glShaderSource(fs, 1, &srcC, nullptr);
  glCompileShader(fs);
  char log[1024];
  for (GLuint sh : { vs, fs })
  {
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
      glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
      std::fprintf(stderr, "GL compile error:\n%s\n", log);
      return false;
    }
  }
  *prog = glCreateProgram();
  glAttachShader(*prog, vs);
  glAttachShader(*prog, fs);
  glLinkProgram(*prog);
  return true;
}

static bool setupGL(GLState& s, const std::vector<uint8_t>& vol)
{
  CGLPixelFormatAttribute attrs[] = {
    kCGLPFAOpenGLProfile, (CGLPixelFormatAttribute)kCGLOGLPVersion_GL4_Core,
    kCGLPFAAccelerated,
    (CGLPixelFormatAttribute)0,
  };
  CGLPixelFormatObj pf = nullptr;
  GLint npix = 0;
  if (CGLChoosePixelFormat(attrs, &pf, &npix) != kCGLNoError || !pf) return false;
  if (CGLCreateContext(pf, nullptr, &s.ctx) != kCGLNoError || !s.ctx) return false;
  CGLSetCurrentContext(s.ctx);
  glGenFramebuffers(1, &s.fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, s.fbo);
  glGenTextures(1, &s.colorTex);
  glBindTexture(GL_TEXTURE_2D, s.colorTex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, kRT, kRT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, s.colorTex, 0);
  if (!compileGL(s, &s.progMarch, 0) || !compileGL(s, &s.progL1, 1) ||
      !compileGL(s, &s.progL1Read, 5) || !compileGL(s, &s.progL1X2, 6) ||
      !compileGL(s, &s.progL1Arr, 11) || !compileGL(s, &s.progM2TA, 12) ||
      !compileGL(s, &s.progMARCHT, 13)) return false;
  glGenVertexArrays(1, &s.vao);
  glBindVertexArray(s.vao);
  glGenBuffers(1, &s.vbo);
  glBindBuffer(GL_ARRAY_BUFFER, s.vbo);
  const float tri[] = { -1.f, -1.f, 3.f, -1.f, -1.f, 3.f };
  glBufferData(GL_ARRAY_BUFFER, sizeof(tri), tri, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
  glGenTextures(1, &s.volTex);
  glBindTexture(GL_TEXTURE_3D, s.volTex);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, kVolX, kVolY, kVolZ, 0, GL_RED,
    GL_UNSIGNED_BYTE, vol.data());
  glGenTextures(1, &s.volArrTex);
  glBindTexture(GL_TEXTURE_2D_ARRAY, s.volArrTex);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R8, kVolX, kVolY, kVolZ, 0, GL_RED,
    GL_UNSIGNED_BYTE, vol.data());
  return true;
}

static GLuint glProgFor(GLState& s, int gv)
{
  switch (gv)
  {
    case 0: return s.progMarch;
    case 5: return s.progL1Read;
    case 6: return s.progL1X2;
    case 11: return s.progL1Arr;
    case 12: return s.progM2TA;
    case 13:
    case 14: return s.progMARCHT;
    default: return s.progL1; // 1/2/3/4/9
  }
}

static double timeGL(GLState& s, int variant, bool nearestFilter)
{
  GLuint prog = glProgFor(s, variant);
  glBindFramebuffer(GL_FRAMEBUFFER, s.fbo);
  glViewport(0, 0, kRT, kRT);
  glUseProgram(prog);
  glActiveTexture(GL_TEXTURE0);
  if (variant == 11 || variant == 12)
  {
    glBindTexture(GL_TEXTURE_2D_ARRAY, s.volArrTex);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER,
      nearestFilter ? GL_NEAREST : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER,
      nearestFilter ? GL_NEAREST : GL_LINEAR);
    GLint uArr = glGetUniformLocation(prog, "volumeArr");
    if (uArr >= 0) glUniform1i(uArr, 0);
  }
  else
  {
    glBindTexture(GL_TEXTURE_3D, s.volTex);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER,
      nearestFilter ? GL_NEAREST : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER,
      nearestFilter ? GL_NEAREST : GL_LINEAR);
  }
  glUniform1f(glGetUniformLocation(prog, "uStep"), kStep);
  GLint uDims = glGetUniformLocation(prog, "uDims");
  if (uDims >= 0) glUniform3f(uDims, (float)kVolX, (float)kVolY, (float)kVolZ);
  glBindVertexArray(s.vao);
  for (int i = 0; i < kWarmup; ++i) { glDrawArrays(GL_TRIANGLES, 0, 3); }
  glFinish();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kFrames; ++i)
  {
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
  }
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / kFrames;
}

static void readbackGL(GLState& s, long long* cov, double* mean)
{
  std::vector<unsigned char> px((size_t)kRT * kRT * 4);
  glReadPixels(0, 0, kRT, kRT, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
  long long c = 0, sum = 0;
  for (size_t i = 0; i < px.size(); i += 4)
  {
    if (px[i + 3] != 0) { ++c; sum += px[i + 1]; }
  }
  *cov = c;
  *mean = c ? (double)sum / c : 0.0;
}

// ---------------------------- Metal ----------------------------
struct MetalState
{
  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> q = nil;
  id<MTLTexture> volTex = nil, colorTex = nil, arrTex = nil;
  id<MTLBuffer> vbuf = nil;
  id<MTLSamplerState> smp = nil, smpNear = nil;
  id<MTLRenderPipelineState> ps[18] = { nil };
};

static NSString* kMetalSrc = @R"(
#include <metal_stdlib>
using namespace metal;
struct FragIn { float4 pos [[position]]; float2 uv; };
struct Params { float step; float4 dims; };
constexpr sampler ce_lin(filter::linear, address::clamp_to_edge);
constexpr sampler ce_near(filter::nearest, address::clamp_to_edge);
constexpr sampler ce_rep(filter::linear, address::repeat);
constexpr sampler ce_pix(filter::linear, address::clamp_to_edge, coord::pixel);
constexpr sampler ce_nolod(filter::linear, address::clamp_to_edge, mip_filter::none);
vertex FragIn vsfull(const device float2* pos [[buffer(0)]], uint vid [[vertex_id]]) {
  FragIn o;
  float2 p = pos[vid];
  o.pos = float4(p, 0.0, 1.0);
  o.uv = p * 0.5 + 0.5;
  return o;
}
fragment float4 march(texture3d<float, access::sample> vol [[texture(0)]],
                      texture2d_array<float, access::sample> arr [[texture(1)]],
                      sampler rsmp [[sampler(0)]],
                      constant Params& p [[buffer(0)]],
                      FragIn in [[stage_in]]) {
  float2 ndc = in.uv * 2.0 - 1.0;
  float3 eye = float3(0.5, 0.5, -0.35);
  float3 dir = normalize(float3(ndc * 2.5, 1.0));
  float ca = cos(0.35), sa = sin(0.35);
  dir = float3(ca * dir.x + sa * dir.z, dir.y, -sa * dir.x + ca * dir.z);
  float3 inv = 1.0 / dir;
  float3 t0 = (float3(0.0) - eye) * inv;
  float3 t1 = (float3(1.0) - eye) * inv;
  float tEnter = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
  float tExit  = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
  if (tExit <= 0.0 || tEnter >= tExit) return float4(0.0);
  int steps = max(1, int(ceil((tExit - tEnter) / p.step)));
  float3 base = eye + dir * (tEnter + 0.5 * p.step);
  float3 d = dir * p.step;
  float acc = 0.0;
  float alpha = 0.0;
  int done = 0;
#if MARCH_VARIANT == 0
  // shipped do-while march, RUNTIME sampler
  int i = 0;
  do {
    float s = vol.sample(rsmp, base + float(i) * d).r;
    float o = s * 1.0;
    float w = 1.0 - alpha;
    acc += w * o;
    alpha += w * o;
    ++i;
    done = i;
  } while (i < steps && alpha <= 0.9);
#elif MARCH_VARIANT == 1 || MARCH_VARIANT == 3
  // frozen coordinate, runtime sampler (lin vs near chosen at bind time)
  for (int i = 0; i < steps; ++i) {
    acc += vol.sample(rsmp, base, level(0)).r;
    done = i + 1;
  }
#elif MARCH_VARIANT == 2
  // frozen coordinate, CONSTEXPR linear sampler — the shipped sVolume shape
  for (int i = 0; i < steps; ++i) {
    acc += vol.sample(ce_lin, base, level(0)).r;
    done = i + 1;
  }
#elif MARCH_VARIANT == 4
  // frozen coordinate, CONSTEXPR nearest sampler
  for (int i = 0; i < steps; ++i) {
    acc += vol.sample(ce_near, base, level(0)).r;
    done = i + 1;
  }
#elif MARCH_VARIANT == 5
  // frozen single texel via read() — no sampler unit at all
  uint3 tc = uint3(base * p.dims.xyz);
  for (int i = 0; i < steps; ++i) {
    acc += vol.read(tc).r;
    done = i + 1;
  }
#elif MARCH_VARIANT == 6
  // two independent taps per iteration (quarter-texel offset), runtime sampler:
  // marginal per-tap issue cost under ILP
  float acc0 = 0.0, acc1 = 0.0;
  float ox = 0.25 / p.dims.x;
  for (int i = 0; i < steps; ++i) {
    acc0 += vol.sample(rsmp, base, level(0)).r;
    acc1 += vol.sample(rsmp, base + float3(ox, 0.0, 0.0), level(0)).r;
    done = i + 1;
  }
  acc = acc0 + acc1;
#elif MARCH_VARIANT == 7
  // frozen coordinate, CONSTEXPR linear, REPEAT addressing: probes whether
  // clamp-to-edge handling inside the sampler front-end is the issue cost
  for (int i = 0; i < steps; ++i) {
    acc += vol.sample(ce_rep, base, level(0)).r;
    done = i + 1;
  }
#elif MARCH_VARIANT == 8
  // frozen coordinate, CONSTEXPR linear, PIXEL coordinates: moves the
  // normalized->texel transform out of the sampler into shader ALU
  float3 pc = base * p.dims.xyz;
  for (int i = 0; i < steps; ++i) {
    acc += vol.sample(ce_pix, pc, level(0)).r;
    done = i + 1;
  }
#elif MARCH_VARIANT == 9
  // frozen coordinate, CONSTEXPR sampler with mip_filter::none and NO level
  // operand — leanest possible tap spelling
  for (int i = 0; i < steps; ++i) {
    acc += vol.sample(ce_nolod, base).r;
    done = i + 1;
  }
#elif MARCH_VARIANT == 10
  // shipped do-while march through the no-level-op spelling
  int i = 0;
  do {
    float s = vol.sample(ce_nolod, base + float(i) * d).r;
    float o = s * 1.0;
    float w = 1.0 - alpha;
    acc += w * o;
    alpha += w * o;
    ++i;
    done = i;
  } while (i < steps && alpha <= 0.9);
#elif MARCH_VARIANT == 11
  // frozen coordinate on the SLICE-ARRAY twin (single bilinear tap):
  // measures Metal's 2D-array sampler rate against its own 3D rate
  uint layerC = uint(p.dims.z) / 2u;
  for (int i = 0; i < steps; ++i) {
    acc += arr.sample(rsmp, base.xy, layerC, level(0)).r;
    done = i + 1;
  }
#elif MARCH_VARIANT == 12
  // V23-shape: two-tap slice-array march (bilinear x2 layers, z lerp in ALU),
  // do-while back-edge exit
  int i = 0;
  do {
    float3 c = base + float(i) * d;
    float g = clamp(c.z, 0.0, 1.0) * p.dims.z - 0.5;
    float flr = floor(g);
    float bl = max(flr, 0.0);
    int l0 = int(min(bl, p.dims.z - 1.0));
    int l1 = min(l0 + 1, int(p.dims.z) - 1);
    float fz = clamp(g - bl, 0.0, 1.0);
    float s0 = arr.sample(rsmp, c.xy, l0, level(0)).r;
    float s1 = arr.sample(rsmp, c.xy, l1, level(0)).r;
    float s = mix(s0, s1, fz);
    float o = s * 1.0;
    float w = 1.0 - alpha;
    acc += w * o;
    alpha += w * o;
    ++i;
    done = i;
  } while (i < steps && alpha <= 0.9);
#elif MARCH_VARIANT == 13
  // transmittance form of the shipped do-while march: T = 1 - alpha,
  // acc += T*o; T -= T*o — IDENTICAL arithmetic sequence, exit on T > 0.1.
  // Gives the backend a single-register latch candidate.
  int i = 0;
  float T = 1.0;
  do {
    float s = vol.sample(rsmp, base + float(i) * d).r;
    float o = s * 1.0;
    acc += T * o;
    T -= T * o;
    ++i;
    done = i;
  } while (i < steps && T > 0.1);
#elif MARCH_VARIANT == 14
  // transmittance + no-level-op constexpr sampler + condition reordered
  // (data-dependent term first): the leanest exit spelling available
  int i = 0;
  float T = 1.0;
  do {
    float s = vol.sample(ce_nolod, base + float(i) * d).r;
    float o = s * 1.0;
    acc += T * o;
    T -= T * o;
    ++i;
    done = i;
  } while (T > 0.1 && i < steps);
#elif MARCH_VARIANT == 15
  // ISOLATION: T-form + RUNTIME sampler + swapped order
  int i = 0;
  float T = 1.0;
  do {
    float s = vol.sample(rsmp, base + float(i) * d).r;
    float o = s * 1.0;
    acc += T * o;
    T -= T * o;
    ++i;
    done = i;
  } while (T > 0.1 && i < steps);
#elif MARCH_VARIANT == 16
  // ISOLATION: T-form + nolod sampler + ORIGINAL order
  int i = 0;
  float T = 1.0;
  do {
    float s = vol.sample(ce_nolod, base + float(i) * d).r;
    float o = s * 1.0;
    acc += T * o;
    T -= T * o;
    ++i;
    done = i;
  } while (i < steps && T > 0.1);
#elif MARCH_VARIANT == 17
  // ISOLATION: ALPHA-form + nolod sampler + swapped order
  int i = 0;
  do {
    float s = vol.sample(ce_nolod, base + float(i) * d).r;
    float o = s * 1.0;
    float w = 1.0 - alpha;
    acc += w * o;
    alpha += w * o;
    ++i;
    done = i;
  } while (alpha <= 0.9 && i < steps);
#endif
  return float4(acc / float(steps), float(done) / 255.0, 0.0, 1.0);
}
)";

static bool setupMetal(MetalState& s, const std::vector<uint8_t>& vol)
{
  s.dev = MTLCreateSystemDefaultDevice();
  s.q = [s.dev newCommandQueue];
  NSError* err = nil;
  for (int v = 0; v < 18; ++v)
  {
    MTLCompileOptions* copts = [[MTLCompileOptions alloc] init];
    copts.preprocessorMacros = @{ @"MARCH_VARIANT" : [NSString stringWithFormat:@"%d", v] };
    if (useKnobs) {
      copts.languageVersion = MTLLanguageVersion3_2;
      copts.mathMode = MTLMathModeFast;
    }
    id<MTLLibrary> lib = [s.dev newLibraryWithSource:kMetalSrc options:copts error:&err];
    if (!lib)
    {
      std::fprintf(stderr, "MSL variant %d error: %s\n", v,
        err.localizedDescription.UTF8String);
      return false;
    }
    id<MTLFunction> fs = [lib newFunctionWithName:@"march"];
    id<MTLFunction> vs = [lib newFunctionWithName:@"vsfull"];
    MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = vs;
    pd.fragmentFunction = fs;
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> p = [s.dev newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!p) return false;
    s.ps[v] = p;
  }
  MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
  td.textureType = MTLTextureType3D;
  td.pixelFormat = MTLPixelFormatR8Unorm;
  td.width = kVolX; td.height = kVolY; td.depth = kVolZ;
  td.mipmapLevelCount = 1;
  td.usage = MTLTextureUsageShaderRead;
  td.storageMode = MTLStorageModePrivate;
  td.allowGPUOptimizedContents = NO; // app constraint
  s.volTex = [s.dev newTextureWithDescriptor:td];
  MTLTextureDescriptor* ad = [[MTLTextureDescriptor alloc] init];
  ad.textureType = MTLTextureType2DArray;
  ad.pixelFormat = MTLPixelFormatR8Unorm;
  ad.width = kVolX; ad.height = kVolY; ad.arrayLength = kVolZ;
  ad.mipmapLevelCount = 1;
  ad.usage = MTLTextureUsageShaderRead;
  ad.storageMode = MTLStorageModePrivate;
  ad.allowGPUOptimizedContents = NO; // app constraint
  s.arrTex = [s.dev newTextureWithDescriptor:ad];
  MTLTextureDescriptor* rt =
    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
      width:kRT height:kRT mipmapped:NO];
  rt.storageMode = MTLStorageModePrivate;
  rt.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderWrite;
  s.colorTex = [s.dev newTextureWithDescriptor:rt];
  id<MTLBuffer> upload = [s.dev newBufferWithBytes:vol.data()
    length:vol.size() options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb = [s.q commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  [blit copyFromBuffer:upload sourceOffset:0 sourceBytesPerRow:kVolX
    sourceBytesPerImage:(NSUInteger)kVolX * kVolY
    sourceSize:MTLSizeMake(kVolX, kVolY, kVolZ)
    toTexture:s.volTex destinationSlice:0 destinationLevel:0
    destinationOrigin:MTLOriginMake(0, 0, 0)];
  // slice-array twin: same bytes, one layer per z slice (depth maps to slices)
  [blit copyFromBuffer:upload sourceOffset:0 sourceBytesPerRow:kVolX
    sourceBytesPerImage:(NSUInteger)kVolX * kVolY
    sourceSize:MTLSizeMake(kVolX, kVolY, kVolZ)
    toTexture:s.arrTex destinationSlice:0 destinationLevel:0
    destinationOrigin:MTLOriginMake(0, 0, 0)];
  [blit endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  s.vbuf = [s.dev newBufferWithBytes:(const float[]){ -1,-1, 3,-1, -1,3 }
    length:6 * sizeof(float) options:MTLResourceStorageModeShared];
  MTLSamplerDescriptor* sd = [[MTLSamplerDescriptor alloc] init];
  sd.minFilter = MTLSamplerMinMagFilterLinear;
  sd.magFilter = MTLSamplerMinMagFilterLinear;
  sd.sAddressMode = sd.tAddressMode = sd.rAddressMode = MTLSamplerAddressModeClampToEdge;
  s.smp = [s.dev newSamplerStateWithDescriptor:sd];
  sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterNearest;
  s.smpNear = [s.dev newSamplerStateWithDescriptor:sd];
  return true;
}

struct alignas(16) MParams { float step; uint32_t pad[3]; float dims[4]; };
static_assert(sizeof(MParams) == 32, "MSL constant-struct layout");

// Metal variant -> sampler binding (constexpr variants ignore the bound sampler)
static id<MTLSamplerState> mSmpFor(MetalState& s, int mv)
{
  return (mv == 3) ? s.smpNear : s.smp;
}

static double timeMetal(MetalState& s, int variant)
{
  MParams params = { kStep, { 0, 0, 0 }, { (float)kVolX, (float)kVolY, (float)kVolZ, 0.0f } };
  MTLRenderPassDescriptor* rpd = [[MTLRenderPassDescriptor alloc] init];
  rpd.colorAttachments[0].texture = s.colorTex;
  rpd.colorAttachments[0].loadAction = MTLLoadActionDontCare;
  rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
  id<MTLRenderPipelineState> ps = s.ps[variant];
  id<MTLSamplerState> msmp = mSmpFor(s, variant);
  auto run = [&]() {
    id<MTLCommandBuffer> cb = [s.q commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
    [enc setRenderPipelineState:ps];
    [enc setVertexBuffer:s.vbuf offset:0 atIndex:0];
    [enc setFragmentTexture:s.volTex atIndex:0];
    [enc setFragmentTexture:s.arrTex atIndex:1];
    [enc setFragmentSamplerState:msmp atIndex:0];
    [enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
  };
  for (int i = 0; i < kWarmup; ++i) { run(); }
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kFrames; ++i) { run(); }
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / kFrames;
}

static void readbackMetal(MetalState& s, long long* cov, double* mean)
{
  id<MTLBuffer> staging =
    [s.dev newBufferWithLength:(NSUInteger)kRT * kRT * 4 options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb = [s.q commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  [blit copyFromTexture:s.colorTex sourceSlice:0 sourceLevel:0
    sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(kRT, kRT, 1)
    toBuffer:staging destinationOffset:0 destinationBytesPerRow:(NSUInteger)kRT * 4
    destinationBytesPerImage:(NSUInteger)kRT * kRT * 4];
  [blit endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  const unsigned char* px = (const unsigned char*)staging.contents;
  long long c = 0, sum = 0;
  for (size_t i = 0; i < (size_t)kRT * kRT * 4; i += 4)
  {
    if (px[i + 3] != 0) { ++c; sum += px[i + 1]; }
  }
  *cov = c;
  *mean = c ? (double)sum / c : 0.0;
}

int main(int argc, char** argv)
{
  if (argc > 1) kRT = std::atoi(argv[1]);
  if (argc > 2) kFrames = std::atoi(argv[2]);
  if (argc > 3) kStep = std::atof(argv[3]);
  if (argc > 4) useKnobs = std::atoi(argv[4]); // 0 = default MSL compile options
  std::vector<uint8_t> vol = makeVolume();
  GLState gl;
  if (!setupGL(gl, vol)) { std::fprintf(stderr, "GL setup failed\n"); return 1; }
  MetalState m;
  if (!setupMetal(m, vol)) { std::fprintf(stderr, "Metal setup failed\n"); return 1; }
  vol.clear();
  vol.shrink_to_fit();
  std::printf("rt=%d frames=%d step=%.5f knobs=%d\n", kRT, kFrames, kStep, useKnobs);
  std::printf("%-24s %10s %11s %7s\n", "pair", "GL ms/f", "Metal ms/f", "M/GL");
  struct Pair { const char* name; int gv; bool glnear; int mv; };
  static const Pair pairs[] = {
    { "march31 (residual)",   0, false, 0 },
    { "l1fetch (issue tax)",  1, false, 1 },
    { "l1_ce (constexpr)",    1, false, 2 },
    { "l1_near (nearest)",    1, true,  3 },
    { "l1_ce_near",           1, true,  4 },
    { "l1_read (no sampler)", 5, false, 5 },
    { "l1_x2 (2 taps/iter)",  6, false, 6 },
    { "l1_rep (addr repeat)", 1, false, 7 },
    { "l1_pix (coord pixel)", 1, false, 8 },
    { "l1_nolod (no level)",  1, false, 9 },
    { "march_nolod",          0, false, 10 },
    { "l1_arr (2D-array)",    11, false, 11 },
    { "march_2ta (arr march)",12, false, 12 },
    { "march_T (transmit)",   13, false, 13 },
    { "march_T_nl (leanest)", 13, false, 14 },
    { "march_T_rsmp_sw",      13, false, 15 },
    { "march_T_nolod_orig",   13, false, 16 },
    { "march_a_nolod_sw",     0, false, 17 },
  };
  static const int nPairs = (int)(sizeof(pairs) / sizeof(pairs[0]));
  double gms[nPairs] = { 0 }, mms[nPairs] = { 0 };
  long long gCov[nPairs] = { 0 }, mCov[nPairs] = { 0 };
  double gMean[nPairs] = { 0 }, mMean[nPairs] = { 0 };
  // Interleaved rounds cancel machine drift (same protocol as the main harness).
  for (int r = 0; r < 3; ++r)
  {
    for (int pi = 0; pi < nPairs; ++pi)
    {
      gms[pi] += timeGL(gl, pairs[pi].gv, pairs[pi].glnear);
      mms[pi] += timeMetal(m, pairs[pi].mv);
      if (r == 0)
      {
        readbackGL(gl, &gCov[pi], &gMean[pi]);
        readbackMetal(m, &mCov[pi], &mMean[pi]);
      }
    }
  }
  for (int pi = 0; pi < nPairs; ++pi)
  {
    std::printf("%-24s %10.2f %11.2f %7.2f   parity %lld/%.1f vs %lld/%.1f\n",
      pairs[pi].name, gms[pi] / 3.0, mms[pi] / 3.0, mms[pi] / gms[pi],
      gCov[pi], gMean[pi], mCov[pi], mMean[pi]);
  }
  return 0;
}
