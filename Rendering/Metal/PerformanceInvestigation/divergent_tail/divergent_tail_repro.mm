// divergent_tail_repro.mm — minimal self-contained repro of the Metal vs
// OpenGL "divergent-tail" deficit (no VTK).
//
// The app finding (J0_FIXED_STEPS_DECOMP.md): at 2048 the j0 Metal renderer is
// SLOWER than GL uncapped (M/GL ~1.05-1.11, Metal ~43.5 ms vs GL ~40.5 ms)
// even though it marches FEWER mean samples (81 vs 86.5), while at any FIXED
// step count Metal is faster (M/GL 0.38-0.91). No march variant (latch, unroll,
// harness w48 schedule) closes the uncapped gap. The cost tracks the
// DISTRIBUTION of per-fragment loop bounds (mean 81, max ~222) and resolution
// (1024 ties, 2048 loses) — a SIMT divergent-loop cost, not envelope or
// per-sample.
//
// This file isolates exactly that: a fullscreen ray-box march, both backends,
// same camera/volume/rays, same TOTAL sample count, with the loop bound either
// per-fragment (divergent, from the ray-box exit) or uniform (fixed = the
// frame mean, computed on the CPU by simulating the same geometry). If the
// hypothesis is right: fixed mode is at parity or Metal wins, divergent mode
// at 2048 Metal loses, at 1024 it ties.
//
// Build:
//   clang++ -std=c++17 -fobjc-arc -O2 -framework Metal -framework OpenGL \
//           -framework Foundation -DGL_SILENCE_DEPRECATION \
//           divergent_tail_repro.mm -o divergent_tail
//
// Run: ./divergent_tail [rtSize=2048] [frames=30] [step=0.01]
//      [volXY=512] [volZ=1794] [alphaMul=1.0] [fixedOverride=0]
//      [optContents=0] [kMax=288] [harness=0]
//
//   alphaMul: opacity-curve steepness; tunes where the data-dependent break
//     fires. MUST stay high enough that some rays break early and others run
//     the full geometric length. With uniform noise (all values random) every
//     ray breaks at ~the same count and the march is only weakly divergent
//     (Metal wins both modes). With the structured volume (dense sphere in
//     air) the break-point distribution is bimodal (~10 vs ~283) and the
//     divergent regime reproduces the app: M/GL ~1.10-1.15 at 2048 while the
//     fixed mode drops to ~0.75-0.98.
//   fixedOverride: cap for the fixed mode (0 = frame geometric mean, ~87).
//     At 128/87 fixed hovers ~0.9-1.08; at 32/16 Metal wins ~0.91-0.92.
//   optContents: allowGPUOptimizedContents on the Metal volume texture (0/1).
//     The ONLY knob that measurably moves the divergent gap (~10%): with YES
//     Metal divergent drops ~31.2 -> 28.7 ms and M/GL collapses to ~1.05.
//     The app forces NO for image-accuracy reasons, so the app's real gap is
//     the NO regime. Still does not close the gap to parity.
//   kMax: function-constant compile-time ceiling for V4 (default 288, above
//     the CPU max of 283). 0 disables it (compiler default constant).
//   harness: 0 = GPU timestamps + DontCare load (fair), 1 = wall clock +
//     Clear (old). Both give the same ratio — the harness is not the gap.
//
// Structure of the volume is the key control: makeVolume() puts a dense
// sphere at the center (CT tissue) and air (0) everywhere else, mirroring a
// DICOM slab where rays through the middle terminate early and rays that
// miss it march the full box. That spatial correlation is what the app's
// deficit tracks — plain random data cannot reproduce it.
//
// EVALUATION (V0-V5 sweep, optContents=NO): no Metal loop-shape variant
// moves the divergent gap. V0 baseline through V5 (chunked reconverge) all
// land at Metal ~31.2-31.5 ms vs GL ~27.2-27.8 ms, M/GL 1.12-1.15. The
// hypothesis that "MSL loses to the GLSL frontend on a high-max per-lane
// trip count" is FALSIFIED: uniform simd_max headers, predicated bodies,
// whole-group exits, compile-time ceilings, and chunked reconvergence are
// all bit-for-bit the same time. The deficit is a property of GPU execution
// of the long-tailed divergent loop (the same device runs the bounded fixed
// mode faster than GL, M/GL 0.75-0.98), plus a ~10% texture-layout effect
// that the app cannot use.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <OpenGL/gl3.h>
#include <OpenGL/OpenGL.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <string>
#include <vector>

static int kVolX = 512;           // volume dims (app: 512x512x1794 = 470 MB)
static int kVolY = 512;
static int kVolZ = 1794;
static int kRT = 2048;
static int kFrames = 30;
static int kWarmup = 10;
static float kStep = 0.004f;          // normalized-volume step size (app SD4-ish)
static float kAlphaMul = 0.35f;       // opacity curve steepness (tunes mean steps)
static int kFixedOverride = 0;        // 0 = use geometric mean as cap, else override
static BOOL kOptContents = NO;        // allowGPUOptimizedContents on the volume texture
static int kMaxConstant = 288;        // compile-time ceiling for variant 4 (must be > max geomSteps)
static int kHarness = 0;              // 0 = GPU timestamps + DontCare (fair), 1 = wall clock + Clear (old)
static int kChunk = 256;              // V6 persistent kernel chunk size (multiple of tgSize)
static int kComputeTG = 256;          // V6 threadgroup size
static int kTile = 32;                 // V10 tile size (square)
static int kComputeGroups = 0;        // V6 threadgroup count (0 = maxThreadsPerThreadgroup.width*8/tg)

#define DBG(...) std::fprintf(stderr, "[repro] " __VA_ARGS__), std::fflush(stderr)

// ---------------------------------------------------------------------------
// Structured volume like the app's DICOM: a dense sphere in the middle (CT
// tissue), air outside. Rays through the dense core accumulate opacity fast
// and break early; rays that miss it run the full geometric length. This
// bimodal distribution (some rays ~10 steps, some ~283) is what makes the
// march strongly divergent — uniform noise would make every ray break at
// roughly the same count (tight distribution, weak divergence).
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
    if (d < r)
    {
      v[i] = 60 + (uint8_t)(s >> 24) % 150; // dense tissue
    }
    else
    {
      v[i] = 0; // air: no opacity, rays march on
    }
  }
  return v;
}

// Same ray-box math as the shaders: eye at (0.5, 0.5, -0.35), box [0,1]^3.
// Returns step count for pixel (px,py) of an rtSize^2 grid, or 0 if the ray
// misses the box. Used to (a) compute the frame-mean step count for the fixed
// mode and (b) print the distribution.
static int cpuSteps(int px, int py, int rt)
{
  const float u = (px + 0.5f) / rt * 2.0f - 1.0f;
  const float v = (py + 0.5f) / rt * 2.0f - 1.0f;
  const float fov = 2.5f;
  const float ex = 0.5f, ey = 0.5f, ez = -0.35f;
  const float dx = u * fov, dy = v * fov, dz = 1.0f;
  const float ca = std::cos(0.35f), sa = std::sin(0.35f);
  const float rx = ca * dx + sa * dz;
  const float rz = -sa * dx + ca * dz;
  const float len = std::sqrt(rx * rx + dy * dy + rz * rz);
  const float nx = rx / len, ny = dy / len, nz = rz / len;
  const float invx = 1.0f / nx, invy = 1.0f / ny, invz = 1.0f / nz;
  const float t0x = (0.0f - ex) * invx, t1x = (1.0f - ex) * invx;
  const float t0y = (0.0f - ey) * invy, t1y = (1.0f - ey) * invy;
  const float t0z = (0.0f - ez) * invz, t1z = (1.0f - ez) * invz;
  const float tEnter =
    std::max(std::max(std::min(t0x, t1x), std::min(t0y, t1y)), std::min(t0z, t1z));
  const float tExit =
    std::min(std::min(std::max(t0x, t1x), std::max(t0y, t1y)), std::max(t0z, t1z));
  if (tExit <= 0.0f || tEnter >= tExit)
  {
    return 0;
  }
  return std::max(1, (int)std::ceil((tExit - tEnter) / kStep));
}

// ---------------------------------------------------------------------------
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
uniform int uMode;      // 0 = divergent (per-fragment ray-box bound), 1 = fixed
uniform int uFixedSteps;
uniform float uStep;
uniform float uAlphaMul;
#ifdef USE_LOD
#define FETCH(coord) textureLod(volumeTex, (coord), 0.0).r
#else
#define FETCH(coord) texture(volumeTex, (coord)).r
#endif
void main() {
  vec2 ndc = vUV * 2.0 - 1.0;
  vec3 eye = vec3(0.5, 0.5, -0.35);
  vec3 dir = normalize(vec3(ndc * 2.5, 1.0));
  // Oblique camera: rotate the rays so they cross the volume at an angle like
  // the app's DICOM view. Straight-on rays march along the +z axis with all
  // SIMT lanes reading the same slices in lockstep (coherent, cache-friendly);
  // oblique rays scatter accesses across the whole volume (the app's regime).
  float ca = cos(0.35), sa = sin(0.35);
  dir = vec3(ca * dir.x + sa * dir.z, dir.y, -sa * dir.x + ca * dir.z);
  vec3 inv = 1.0 / dir;
  vec3 t0 = (vec3(0.0) - eye) * inv;
  vec3 t1 = (vec3(1.0) - eye) * inv;
  float tEnter = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
  float tExit  = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
  if (tExit <= 0.0 || tEnter >= tExit) {
    outColor = vec4(0.0, 0.0, 0.0, 0.0);
    return;
  }
  int geomSteps = max(1, int(ceil((tExit - tEnter) / uStep)));
  int steps = (uMode == 1) ? min(geomSteps, uFixedSteps) : geomSteps;
  vec3 base = eye + dir * (tEnter + 0.5 * uStep);
  vec3 d = dir * uStep;
  float acc = 0.0;
  float alpha = 0.0;
  int done = 0;
  for (int i = 0; i < steps; ++i) {
    // Explicit LOD (level 0) kills the implicit screen-space gradient
    // computation. Volume marching does not want derivatives — every sample
    // should be a full-res texel. USE_LOD switches texture() -> textureLod().
    // Each is compiled as a separate program (USE_LOD baked in) so the
    // comparison is not polluted by a runtime uniform branch.
    float s = FETCH(base + float(i) * d);
    // Opacity early-exit: alpha accumulates from the data, so rays terminate
    // at DATA-DEPENDENT points (some at step 10, some at 283). This is what
    // makes the app's march divergent in the first place. In divergent mode
    // the geometric bound is uncapped (max 283) so long rays pin SIMT lanes;
    // in fixed mode the bound is capped at the frame mean so no ray runs
    // longer than the cap.
    float o = s * uAlphaMul;
    float w = 1.0 - alpha;
    acc += w * o;
    alpha += w * o;
    done = i + 1;
    if (alpha > 0.9) break;
  }
  outColor = vec4(acc / float(steps), float(done) / 255.0, 0.0, 1.0);
}
)";

struct GLState
{
  CGLContextObj ctx = nullptr;
  GLuint fbo = 0, colorTex = 0, vao = 0, vbo = 0, volTex = 0;
  GLuint progImplicit = 0, progLod = 0; // 0 = texture(), 1 = textureLod()
  GLint uMode = -1, uFixedSteps = -1, uStep = -1, uAlphaMul = -1;
};

static bool compileGLProgram(GLState& s, GLuint* prog, bool useLod)
{
  // USE_LOD must come AFTER the #version line, so splice it in rather than
  // prepending.
  const char* versionEnd = strstr(kGLFragSrc, "\n");
  std::string src;
  src.reserve(strlen(kGLFragSrc) + 32);
  src.append(kGLFragSrc, versionEnd - kGLFragSrc + 1);
  src += useLod ? "#define USE_LOD 1\n" : "";
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
      std::fprintf(stderr, "GL shader compile error (%s):\n%s\n",
        useLod ? "lod" : "implicit", log);
      return false;
    }
  }
  *prog = glCreateProgram();
  glAttachShader(*prog, vs);
  glAttachShader(*prog, fs);
  glLinkProgram(*prog);
  GLint ok = 0;
  glGetProgramiv(*prog, GL_LINK_STATUS, &ok);
  if (!ok)
  {
    glGetProgramInfoLog(*prog, sizeof(log), nullptr, log);
    std::fprintf(stderr, "GL link error (%s):\n%s\n", useLod ? "lod" : "implicit", log);
    return false;
  }
  return true;
}

static bool setupGL(GLState& s)
{
  CGLPixelFormatAttribute attrs[] = {
    kCGLPFAOpenGLProfile, (CGLPixelFormatAttribute)kCGLOGLPVersion_GL4_Core,
    kCGLPFAAccelerated,
    (CGLPixelFormatAttribute)0,
  };
  CGLPixelFormatObj pf = nullptr;
  GLint npix = 0;
  if (CGLChoosePixelFormat(attrs, &pf, &npix) != kCGLNoError || pf == nullptr)
  {
    std::fprintf(stderr, "GL: no accelerated pixel format\n");
    return false;
  }
  if (CGLCreateContext(pf, nullptr, &s.ctx) != kCGLNoError || s.ctx == nullptr)
  {
    std::fprintf(stderr, "GL: could not create context\n");
    return false;
  }
  CGLSetCurrentContext(s.ctx);

  glGenFramebuffers(1, &s.fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, s.fbo);
  glGenTextures(1, &s.colorTex);
  glBindTexture(GL_TEXTURE_2D, s.colorTex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, kRT, kRT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, s.colorTex, 0);
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
  {
    std::fprintf(stderr, "GL: FBO incomplete\n");
    return false;
  }

  // Two fragment programs: implicit texture() (variant 0) and textureLod()
  // (variants 1-4). The fetch is baked in via USE_LOD so the comparison is
  // not polluted by a runtime uniform branch. Each compiles its own vertex
  // shader.
  if (!compileGLProgram(s, &s.progImplicit, false) ||
      !compileGLProgram(s, &s.progLod, true))
  {
    return false;
  }

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
  glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, kVolX, kVolY, kVolZ, 0, GL_RED, GL_UNSIGNED_BYTE,
    nullptr);
  return true;
}

static void uploadGLVolume(GLState& s, const std::vector<uint8_t>& v)
{
  glBindTexture(GL_TEXTURE_3D, s.volTex);
  glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, kVolX, kVolY, kVolZ, GL_RED, GL_UNSIGNED_BYTE,
    v.data());
}

static double timeGL(GLState& s, int mode, int fixedSteps, int useLod)
{
  glBindFramebuffer(GL_FRAMEBUFFER, s.fbo);
  glViewport(0, 0, kRT, kRT);
  glUseProgram(useLod ? s.progLod : s.progImplicit);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_3D, s.volTex);
  GLint uMode = glGetUniformLocation(useLod ? s.progLod : s.progImplicit, "uMode");
  GLint uFixedSteps = glGetUniformLocation(useLod ? s.progLod : s.progImplicit, "uFixedSteps");
  GLint uStep = glGetUniformLocation(useLod ? s.progLod : s.progImplicit, "uStep");
  GLint uAlphaMul = glGetUniformLocation(useLod ? s.progLod : s.progImplicit, "uAlphaMul");
  glUniform1i(uMode, mode);
  glUniform1i(uFixedSteps, fixedSteps);
  glUniform1f(uStep, kStep);
  glUniform1f(uAlphaMul, kAlphaMul);
  glBindVertexArray(s.vao);
  for (int i = 0; i < kWarmup; ++i)
  {
    glDrawArrays(GL_TRIANGLES, 0, 3);
  }
  glFinish();
  if (kHarness == 1)
  {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kFrames; ++i)
    {
      glDrawArrays(GL_TRIANGLES, 0, 3);
      glFinish();
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / kFrames;
  }
  // GPU time via GL_TIME_ELAPSED (ns), one query per frame, comparable to
  // Metal's GPUStartTime/GPUEndTime. Host wall time would include driver
  // stalls that the Metal measurement does not. One query object per frame:
  // reusing a single object makes GL_QUERY_RESULT return only the last frame.
  std::vector<GLuint> qids(kFrames);
  glGenQueries(kFrames, qids.data());
  const auto w0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kFrames; ++i)
  {
    glBeginQuery(GL_TIME_ELAPSED, qids[i]);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndQuery(GL_TIME_ELAPSED);
  }
  glFinish();
  const auto w1 = std::chrono::steady_clock::now();
  const double wallMs = std::chrono::duration<double, std::milli>(w1 - w0).count() / kFrames;
  GLuint64 totalNs = 0;
  for (int i = 0; i < kFrames; ++i)
  {
    GLuint64 t = 0;
    glGetQueryObjectui64v(qids[i], GL_QUERY_RESULT, &t);
    totalNs += t;
  }
  glDeleteQueries(kFrames, qids.data());
  DBG("timeGL wall=%.2f gpu=%.2f mode=%d\n", wallMs, (double)totalNs / kFrames / 1e6, mode);
  return (double)totalNs / kFrames / 1e6; // ns -> ms
}

// Returns (coveredPixels, meanSteps) from the G channel.
static void readbackGL(GLState& s, long long* covered, double* meanSteps)
{
  std::vector<unsigned char> px((size_t)kRT * kRT * 4);
  glReadPixels(0, 0, kRT, kRT, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
  long long cov = 0, sum = 0;
  for (size_t i = 0; i < px.size(); i += 4)
  {
    if (px[i + 3] != 0)
    {
      ++cov;
      sum += px[i + 1];
    }
  }
  *covered = cov;
  *meanSteps = cov ? (double)sum / cov : 0.0;
}

// ---------------------------------------------------------------------------
static NSString* kMetalSrc = @R"(
#include <metal_stdlib>
using namespace metal;
struct FragIn { float4 pos [[position]]; float2 uv; };
struct Params { int mode; int fixedSteps; float step; float alphaMul;
                int width; int height; int nPixels; int tileW; int tileH;
                float deadFlag; int bucketLo; int bucketHi; int bucketCap; };
constant int kMax [[function_constant(0)]]; // compile-time ceiling (variant 4)
vertex FragIn vsfull(const device float2* pos [[buffer(0)]], uint vid [[vertex_id]]) {
  FragIn o;
  float2 p = pos[vid];
  o.pos = float4(p, 0.0, 1.0);
  o.uv = p * 0.5 + 0.5;
  return o;
}
#if MARCH_VARIANT == 12
struct MarchOut { float4 c [[color(0)]]; float h [[color(1)]]; };
fragment MarchOut march(texture3d<float, access::sample> vol [[texture(0)]],
                        texture2d<float, access::read> histTex [[texture(1)]],
                        sampler smp [[sampler(0)]],
                        constant Params& p [[buffer(0)]],
                        FragIn in [[stage_in]]) {
#else
fragment float4 march(texture3d<float, access::sample> vol [[texture(0)]],
                      sampler smp [[sampler(0)]],
                      constant Params& p [[buffer(0)]],
                      FragIn in [[stage_in]]) {
#endif
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
  if (tExit <= 0.0 || tEnter >= tExit) {
#if MARCH_VARIANT == 12
    return MarchOut{ float4(0.0, 0.0, 0.0, 0.0), 0.0 };
#else
    return float4(0.0, 0.0, 0.0, 0.0);
#endif
  }
  int geomSteps = max(1, int(ceil((tExit - tEnter) / p.step)));
  int steps = (p.mode == 1) ? min(geomSteps, p.fixedSteps) : geomSteps;
  float3 base = eye + dir * (tEnter + 0.5 * p.step);
  float3 d = dir * p.step;
  float acc = 0.0;
  float alpha = 0.0;
  int done = 0;
#if MARCH_VARIANT == 0
  // V0: baseline — implicit sample gradients, divergent per-lane for bound.
  // This is the mode that loses at 2048 (M/GL ~1.13).
  for (int i = 0; i < steps; ++i) {
    float s = vol.sample(smp, base + float(i) * d).r;
    float o = s * p.alphaMul;
    float w = 1.0 - alpha;
    acc += w * o;
    alpha += w * o;
    done = i + 1;
    if (alpha > 0.9) break;
  }
#elif MARCH_VARIANT == 1 || MARCH_VARIANT == 6
  // V1: step A — explicit level(0.0). Same loop shape, same work, but no
  // implicit gradient/quad machinery. Should be strictly cheaper than V0.
  // (V6 compiles this same fragment for library-shape parity; the compute
  // kernel below is what actually runs for variant 6.)
  for (int i = 0; i < steps; ++i) {
    float s = vol.sample(smp, base + float(i) * d, level(0.0)).r;
    float o = s * p.alphaMul;
    float w = 1.0 - alpha;
    acc += w * o;
    alpha += w * o;
    done = i + 1;
    if (alpha > 0.9) break;
  }
#elif MARCH_VARIANT == 2
  // V2: step B — uniform header. Bound is the SIMD-group max of the per-lane
  // step counts, body is predicated on (i < steps && alpha <= 0.9). Per-pixel
  // work identical to V1; only the loop shape the compiler sees changes.
  int limit = simd_max(steps);
  for (int i = 0; i < limit; ++i) {
    bool alive = (i < steps) && (alpha <= 0.9);
    if (alive) {
      float s = vol.sample(smp, base + float(i) * d, level(0.0)).r;
      float o = s * p.alphaMul;
      float w = 1.0 - alpha;
      acc += w * o;
      alpha += w * o;
      done = i + 1;
    }
  }
#elif MARCH_VARIANT == 3
  // V3: step C — whole-group early exit. Same as V2 plus simd_any(alive) so
  // the whole SIMD group leaves the loop once no lane is alive. This gives the
  // compiler a uniform loop terminator it can sink to.
  int limit = simd_max(steps);
  for (int i = 0; i < limit; ++i) {
    bool alive = (i < steps) && (alpha <= 0.9);
    if (alive) {
      float s = vol.sample(smp, base + float(i) * d, level(0.0)).r;
      float o = s * p.alphaMul;
      float w = 1.0 - alpha;
      acc += w * o;
      alpha += w * o;
      done = i + 1;
    }
    if (!simd_any(alive)) break;
  }
#elif MARCH_VARIANT == 4
  // V4: step D — compile-time ceiling. kMax is a function constant (set to
  // 288 when creating the PSO, above the CPU max of 283) so the loop bound is
  // a compile-time constant while the per-lane trip count stays uncapped.
  // unroll(1) keeps it a tight hardware loop instead of trying to
  // unroll 283 iterations.
  int stepsC = min(steps, kMax);
  [[unroll(1)]]
  for (int i = 0; i < kMax; ++i) {
    if (i >= stepsC) break;
    float s = vol.sample(smp, base + float(i) * d, level(0.0)).r;
    float o = s * p.alphaMul;
    float w = 1.0 - alpha;
    acc += w * o;
    alpha += w * o;
    done = i + 1;
    if (alpha > 0.9) break;
  }
#elif MARCH_VARIANT == 5
  // V5: chunked reconverge — the SIMD-width chunk version. The outer loop
  // over 32-step chunks has a data-dependent bound, but each inner loop's
  // bound is simd_max'd so it is SIMT-uniform; the inner body is predicated
  // on j < n && alpha <= 0.9, and the whole group exits a chunk early once
  // no lane is alive. Same per-pixel work.
  const int CHUNK = 32;
  for (int b = 0; b < steps; b += CHUNK) {
    int n = min(CHUNK, steps - b);
    int nU = simd_max(n);
    int lastDone = done;
    for (int j = 0; j < nU; ++j) {
      bool alive = (j < n) && (alpha <= 0.9);
      if (alive) {
        float s = vol.sample(smp, base + float(b + j) * d, level(0.0)).r;
        float o = s * p.alphaMul;
        float w = 1.0 - alpha;
        acc += w * o;
        alpha += w * o;
        done = b + j + 1;
      }
    }
    if (b + n >= steps) break;
    bool anyAlive = simd_any(alpha <= 0.9);
    if (!anyAlive) break;
    (void)lastDone;
  }
#elif MARCH_VARIANT == 8 || MARCH_VARIANT == 9 || MARCH_VARIANT == 10 || MARCH_VARIANT == 11
  // V13-V16: v34 exact shape with batch width BATCH_N (8/16/32/48).
  // ONE break check per batch (MV9_ADVANCE), all positions computed first,
  // all fetches back-to-back, BRANCH-FREE serial composite chain with a
  // select-gated opacity latch (app MV9_COMPOSITE semantics, exact parity:
  // acc/alpha/done freeze the moment alpha crosses 0.9).
  // NOTE: unused lanes MUST be compiled out per variant (#if); the MSL
  // compiler does not DCE texture samples referenced by runtime guards.
#if MARCH_VARIANT == 8
  const int BATCH_N = 8;
#elif MARCH_VARIANT == 9
  const int BATCH_N = 16;
#elif MARCH_VARIANT == 10
  const int BATCH_N = 32;
#else
  const int BATCH_N = 48;
#endif
  int i = 0;
  const int stepsB = steps;
  bool opaque = false;
  for (; i + BATCH_N <= stepsB; i += BATCH_N)
  {
    float3 p0 = base + float(i + 0) * d;
    float3 p1 = base + float(i + 1) * d;
    float3 p2 = base + float(i + 2) * d;
    float3 p3 = base + float(i + 3) * d;
    float3 p4 = base + float(i + 4) * d;
    float3 p5 = base + float(i + 5) * d;
    float3 p6 = base + float(i + 6) * d;
    float3 p7 = base + float(i + 7) * d;
#if MARCH_VARIANT >= 9
    float3 p8 = base + float(i + 8) * d;
    float3 p9 = base + float(i + 9) * d;
    float3 p10 = base + float(i + 10) * d;
    float3 p11 = base + float(i + 11) * d;
    float3 p12 = base + float(i + 12) * d;
    float3 p13 = base + float(i + 13) * d;
    float3 p14 = base + float(i + 14) * d;
    float3 p15 = base + float(i + 15) * d;
#endif
#if MARCH_VARIANT >= 10
    float3 p16 = base + float(i + 16) * d;
    float3 p17 = base + float(i + 17) * d;
    float3 p18 = base + float(i + 18) * d;
    float3 p19 = base + float(i + 19) * d;
    float3 p20 = base + float(i + 20) * d;
    float3 p21 = base + float(i + 21) * d;
    float3 p22 = base + float(i + 22) * d;
    float3 p23 = base + float(i + 23) * d;
    float3 p24 = base + float(i + 24) * d;
    float3 p25 = base + float(i + 25) * d;
    float3 p26 = base + float(i + 26) * d;
    float3 p27 = base + float(i + 27) * d;
    float3 p28 = base + float(i + 28) * d;
    float3 p29 = base + float(i + 29) * d;
    float3 p30 = base + float(i + 30) * d;
    float3 p31 = base + float(i + 31) * d;
#endif
#if MARCH_VARIANT >= 11
    float3 p32 = base + float(i + 32) * d;
    float3 p33 = base + float(i + 33) * d;
    float3 p34 = base + float(i + 34) * d;
    float3 p35 = base + float(i + 35) * d;
    float3 p36 = base + float(i + 36) * d;
    float3 p37 = base + float(i + 37) * d;
    float3 p38 = base + float(i + 38) * d;
    float3 p39 = base + float(i + 39) * d;
    float3 p40 = base + float(i + 40) * d;
    float3 p41 = base + float(i + 41) * d;
    float3 p42 = base + float(i + 42) * d;
    float3 p43 = base + float(i + 43) * d;
    float3 p44 = base + float(i + 44) * d;
    float3 p45 = base + float(i + 45) * d;
    float3 p46 = base + float(i + 46) * d;
    float3 p47 = base + float(i + 47) * d;
#endif
    float s0 = vol.sample(smp, p0, level(0.0)).r;
    float s1 = vol.sample(smp, p1, level(0.0)).r;
    float s2 = vol.sample(smp, p2, level(0.0)).r;
    float s3 = vol.sample(smp, p3, level(0.0)).r;
    float s4 = vol.sample(smp, p4, level(0.0)).r;
    float s5 = vol.sample(smp, p5, level(0.0)).r;
    float s6 = vol.sample(smp, p6, level(0.0)).r;
    float s7 = vol.sample(smp, p7, level(0.0)).r;
#if MARCH_VARIANT >= 9
    float s8 = vol.sample(smp, p8, level(0.0)).r;
    float s9 = vol.sample(smp, p9, level(0.0)).r;
    float s10 = vol.sample(smp, p10, level(0.0)).r;
    float s11 = vol.sample(smp, p11, level(0.0)).r;
    float s12 = vol.sample(smp, p12, level(0.0)).r;
    float s13 = vol.sample(smp, p13, level(0.0)).r;
    float s14 = vol.sample(smp, p14, level(0.0)).r;
    float s15 = vol.sample(smp, p15, level(0.0)).r;
#endif
#if MARCH_VARIANT >= 10
    float s16 = vol.sample(smp, p16, level(0.0)).r;
    float s17 = vol.sample(smp, p17, level(0.0)).r;
    float s18 = vol.sample(smp, p18, level(0.0)).r;
    float s19 = vol.sample(smp, p19, level(0.0)).r;
    float s20 = vol.sample(smp, p20, level(0.0)).r;
    float s21 = vol.sample(smp, p21, level(0.0)).r;
    float s22 = vol.sample(smp, p22, level(0.0)).r;
    float s23 = vol.sample(smp, p23, level(0.0)).r;
    float s24 = vol.sample(smp, p24, level(0.0)).r;
    float s25 = vol.sample(smp, p25, level(0.0)).r;
    float s26 = vol.sample(smp, p26, level(0.0)).r;
    float s27 = vol.sample(smp, p27, level(0.0)).r;
    float s28 = vol.sample(smp, p28, level(0.0)).r;
    float s29 = vol.sample(smp, p29, level(0.0)).r;
    float s30 = vol.sample(smp, p30, level(0.0)).r;
    float s31 = vol.sample(smp, p31, level(0.0)).r;
#endif
#if MARCH_VARIANT >= 11
    float s32 = vol.sample(smp, p32, level(0.0)).r;
    float s33 = vol.sample(smp, p33, level(0.0)).r;
    float s34 = vol.sample(smp, p34, level(0.0)).r;
    float s35 = vol.sample(smp, p35, level(0.0)).r;
    float s36 = vol.sample(smp, p36, level(0.0)).r;
    float s37 = vol.sample(smp, p37, level(0.0)).r;
    float s38 = vol.sample(smp, p38, level(0.0)).r;
    float s39 = vol.sample(smp, p39, level(0.0)).r;
    float s40 = vol.sample(smp, p40, level(0.0)).r;
    float s41 = vol.sample(smp, p41, level(0.0)).r;
    float s42 = vol.sample(smp, p42, level(0.0)).r;
    float s43 = vol.sample(smp, p43, level(0.0)).r;
    float s44 = vol.sample(smp, p44, level(0.0)).r;
    float s45 = vol.sample(smp, p45, level(0.0)).r;
    float s46 = vol.sample(smp, p46, level(0.0)).r;
    float s47 = vol.sample(smp, p47, level(0.0)).r;
#endif
    float o0 = s0 * p.alphaMul;
    float g0 = opaque ? 0.0 : 1.0;
    float w0 = 1.0 - alpha;
    acc += g0 * w0 * o0;
    alpha += g0 * w0 * o0;
    done = opaque ? done : (i + 1);
    opaque = (alpha > 0.9);
    float o1 = s1 * p.alphaMul;
    float g1 = opaque ? 0.0 : 1.0;
    float w1 = 1.0 - alpha;
    acc += g1 * w1 * o1;
    alpha += g1 * w1 * o1;
    done = opaque ? done : (i + 2);
    opaque = (alpha > 0.9);
    float o2 = s2 * p.alphaMul;
    float g2 = opaque ? 0.0 : 1.0;
    float w2 = 1.0 - alpha;
    acc += g2 * w2 * o2;
    alpha += g2 * w2 * o2;
    done = opaque ? done : (i + 3);
    opaque = (alpha > 0.9);
    float o3 = s3 * p.alphaMul;
    float g3 = opaque ? 0.0 : 1.0;
    float w3 = 1.0 - alpha;
    acc += g3 * w3 * o3;
    alpha += g3 * w3 * o3;
    done = opaque ? done : (i + 4);
    opaque = (alpha > 0.9);
    float o4 = s4 * p.alphaMul;
    float g4 = opaque ? 0.0 : 1.0;
    float w4 = 1.0 - alpha;
    acc += g4 * w4 * o4;
    alpha += g4 * w4 * o4;
    done = opaque ? done : (i + 5);
    opaque = (alpha > 0.9);
    float o5 = s5 * p.alphaMul;
    float g5 = opaque ? 0.0 : 1.0;
    float w5 = 1.0 - alpha;
    acc += g5 * w5 * o5;
    alpha += g5 * w5 * o5;
    done = opaque ? done : (i + 6);
    opaque = (alpha > 0.9);
    float o6 = s6 * p.alphaMul;
    float g6 = opaque ? 0.0 : 1.0;
    float w6 = 1.0 - alpha;
    acc += g6 * w6 * o6;
    alpha += g6 * w6 * o6;
    done = opaque ? done : (i + 7);
    opaque = (alpha > 0.9);
    float o7 = s7 * p.alphaMul;
    float g7 = opaque ? 0.0 : 1.0;
    float w7 = 1.0 - alpha;
    acc += g7 * w7 * o7;
    alpha += g7 * w7 * o7;
    done = opaque ? done : (i + 8);
    opaque = (alpha > 0.9);
#if MARCH_VARIANT >= 9
    float o8 = s8 * p.alphaMul;
    float g8 = opaque ? 0.0 : 1.0;
    float w8 = 1.0 - alpha;
    acc += g8 * w8 * o8;
    alpha += g8 * w8 * o8;
    done = opaque ? done : (i + 9);
    opaque = (alpha > 0.9);
    float o9 = s9 * p.alphaMul;
    float g9 = opaque ? 0.0 : 1.0;
    float w9 = 1.0 - alpha;
    acc += g9 * w9 * o9;
    alpha += g9 * w9 * o9;
    done = opaque ? done : (i + 10);
    opaque = (alpha > 0.9);
    float o10 = s10 * p.alphaMul;
    float g10 = opaque ? 0.0 : 1.0;
    float w10 = 1.0 - alpha;
    acc += g10 * w10 * o10;
    alpha += g10 * w10 * o10;
    done = opaque ? done : (i + 11);
    opaque = (alpha > 0.9);
    float o11 = s11 * p.alphaMul;
    float g11 = opaque ? 0.0 : 1.0;
    float w11 = 1.0 - alpha;
    acc += g11 * w11 * o11;
    alpha += g11 * w11 * o11;
    done = opaque ? done : (i + 12);
    opaque = (alpha > 0.9);
    float o12 = s12 * p.alphaMul;
    float g12 = opaque ? 0.0 : 1.0;
    float w12 = 1.0 - alpha;
    acc += g12 * w12 * o12;
    alpha += g12 * w12 * o12;
    done = opaque ? done : (i + 13);
    opaque = (alpha > 0.9);
    float o13 = s13 * p.alphaMul;
    float g13 = opaque ? 0.0 : 1.0;
    float w13 = 1.0 - alpha;
    acc += g13 * w13 * o13;
    alpha += g13 * w13 * o13;
    done = opaque ? done : (i + 14);
    opaque = (alpha > 0.9);
    float o14 = s14 * p.alphaMul;
    float g14 = opaque ? 0.0 : 1.0;
    float w14 = 1.0 - alpha;
    acc += g14 * w14 * o14;
    alpha += g14 * w14 * o14;
    done = opaque ? done : (i + 15);
    opaque = (alpha > 0.9);
    float o15 = s15 * p.alphaMul;
    float g15 = opaque ? 0.0 : 1.0;
    float w15 = 1.0 - alpha;
    acc += g15 * w15 * o15;
    alpha += g15 * w15 * o15;
    done = opaque ? done : (i + 16);
    opaque = (alpha > 0.9);
#endif
#if MARCH_VARIANT >= 10
    float o16 = s16 * p.alphaMul;
    float g16 = opaque ? 0.0 : 1.0;
    float w16 = 1.0 - alpha;
    acc += g16 * w16 * o16;
    alpha += g16 * w16 * o16;
    done = opaque ? done : (i + 17);
    opaque = (alpha > 0.9);
    float o17 = s17 * p.alphaMul;
    float g17 = opaque ? 0.0 : 1.0;
    float w17 = 1.0 - alpha;
    acc += g17 * w17 * o17;
    alpha += g17 * w17 * o17;
    done = opaque ? done : (i + 18);
    opaque = (alpha > 0.9);
    float o18 = s18 * p.alphaMul;
    float g18 = opaque ? 0.0 : 1.0;
    float w18 = 1.0 - alpha;
    acc += g18 * w18 * o18;
    alpha += g18 * w18 * o18;
    done = opaque ? done : (i + 19);
    opaque = (alpha > 0.9);
    float o19 = s19 * p.alphaMul;
    float g19 = opaque ? 0.0 : 1.0;
    float w19 = 1.0 - alpha;
    acc += g19 * w19 * o19;
    alpha += g19 * w19 * o19;
    done = opaque ? done : (i + 20);
    opaque = (alpha > 0.9);
    float o20 = s20 * p.alphaMul;
    float g20 = opaque ? 0.0 : 1.0;
    float w20 = 1.0 - alpha;
    acc += g20 * w20 * o20;
    alpha += g20 * w20 * o20;
    done = opaque ? done : (i + 21);
    opaque = (alpha > 0.9);
    float o21 = s21 * p.alphaMul;
    float g21 = opaque ? 0.0 : 1.0;
    float w21 = 1.0 - alpha;
    acc += g21 * w21 * o21;
    alpha += g21 * w21 * o21;
    done = opaque ? done : (i + 22);
    opaque = (alpha > 0.9);
    float o22 = s22 * p.alphaMul;
    float g22 = opaque ? 0.0 : 1.0;
    float w22 = 1.0 - alpha;
    acc += g22 * w22 * o22;
    alpha += g22 * w22 * o22;
    done = opaque ? done : (i + 23);
    opaque = (alpha > 0.9);
    float o23 = s23 * p.alphaMul;
    float g23 = opaque ? 0.0 : 1.0;
    float w23 = 1.0 - alpha;
    acc += g23 * w23 * o23;
    alpha += g23 * w23 * o23;
    done = opaque ? done : (i + 24);
    opaque = (alpha > 0.9);
    float o24 = s24 * p.alphaMul;
    float g24 = opaque ? 0.0 : 1.0;
    float w24 = 1.0 - alpha;
    acc += g24 * w24 * o24;
    alpha += g24 * w24 * o24;
    done = opaque ? done : (i + 25);
    opaque = (alpha > 0.9);
    float o25 = s25 * p.alphaMul;
    float g25 = opaque ? 0.0 : 1.0;
    float w25 = 1.0 - alpha;
    acc += g25 * w25 * o25;
    alpha += g25 * w25 * o25;
    done = opaque ? done : (i + 26);
    opaque = (alpha > 0.9);
    float o26 = s26 * p.alphaMul;
    float g26 = opaque ? 0.0 : 1.0;
    float w26 = 1.0 - alpha;
    acc += g26 * w26 * o26;
    alpha += g26 * w26 * o26;
    done = opaque ? done : (i + 27);
    opaque = (alpha > 0.9);
    float o27 = s27 * p.alphaMul;
    float g27 = opaque ? 0.0 : 1.0;
    float w27 = 1.0 - alpha;
    acc += g27 * w27 * o27;
    alpha += g27 * w27 * o27;
    done = opaque ? done : (i + 28);
    opaque = (alpha > 0.9);
    float o28 = s28 * p.alphaMul;
    float g28 = opaque ? 0.0 : 1.0;
    float w28 = 1.0 - alpha;
    acc += g28 * w28 * o28;
    alpha += g28 * w28 * o28;
    done = opaque ? done : (i + 29);
    opaque = (alpha > 0.9);
    float o29 = s29 * p.alphaMul;
    float g29 = opaque ? 0.0 : 1.0;
    float w29 = 1.0 - alpha;
    acc += g29 * w29 * o29;
    alpha += g29 * w29 * o29;
    done = opaque ? done : (i + 30);
    opaque = (alpha > 0.9);
    float o30 = s30 * p.alphaMul;
    float g30 = opaque ? 0.0 : 1.0;
    float w30 = 1.0 - alpha;
    acc += g30 * w30 * o30;
    alpha += g30 * w30 * o30;
    done = opaque ? done : (i + 31);
    opaque = (alpha > 0.9);
    float o31 = s31 * p.alphaMul;
    float g31 = opaque ? 0.0 : 1.0;
    float w31 = 1.0 - alpha;
    acc += g31 * w31 * o31;
    alpha += g31 * w31 * o31;
    done = opaque ? done : (i + 32);
    opaque = (alpha > 0.9);
#endif
#if MARCH_VARIANT >= 11
    float o32 = s32 * p.alphaMul;
    float g32 = opaque ? 0.0 : 1.0;
    float w32 = 1.0 - alpha;
    acc += g32 * w32 * o32;
    alpha += g32 * w32 * o32;
    done = opaque ? done : (i + 33);
    opaque = (alpha > 0.9);
    float o33 = s33 * p.alphaMul;
    float g33 = opaque ? 0.0 : 1.0;
    float w33 = 1.0 - alpha;
    acc += g33 * w33 * o33;
    alpha += g33 * w33 * o33;
    done = opaque ? done : (i + 34);
    opaque = (alpha > 0.9);
    float o34 = s34 * p.alphaMul;
    float g34 = opaque ? 0.0 : 1.0;
    float w34 = 1.0 - alpha;
    acc += g34 * w34 * o34;
    alpha += g34 * w34 * o34;
    done = opaque ? done : (i + 35);
    opaque = (alpha > 0.9);
    float o35 = s35 * p.alphaMul;
    float g35 = opaque ? 0.0 : 1.0;
    float w35 = 1.0 - alpha;
    acc += g35 * w35 * o35;
    alpha += g35 * w35 * o35;
    done = opaque ? done : (i + 36);
    opaque = (alpha > 0.9);
    float o36 = s36 * p.alphaMul;
    float g36 = opaque ? 0.0 : 1.0;
    float w36 = 1.0 - alpha;
    acc += g36 * w36 * o36;
    alpha += g36 * w36 * o36;
    done = opaque ? done : (i + 37);
    opaque = (alpha > 0.9);
    float o37 = s37 * p.alphaMul;
    float g37 = opaque ? 0.0 : 1.0;
    float w37 = 1.0 - alpha;
    acc += g37 * w37 * o37;
    alpha += g37 * w37 * o37;
    done = opaque ? done : (i + 38);
    opaque = (alpha > 0.9);
    float o38 = s38 * p.alphaMul;
    float g38 = opaque ? 0.0 : 1.0;
    float w38 = 1.0 - alpha;
    acc += g38 * w38 * o38;
    alpha += g38 * w38 * o38;
    done = opaque ? done : (i + 39);
    opaque = (alpha > 0.9);
    float o39 = s39 * p.alphaMul;
    float g39 = opaque ? 0.0 : 1.0;
    float w39 = 1.0 - alpha;
    acc += g39 * w39 * o39;
    alpha += g39 * w39 * o39;
    done = opaque ? done : (i + 40);
    opaque = (alpha > 0.9);
    float o40 = s40 * p.alphaMul;
    float g40 = opaque ? 0.0 : 1.0;
    float w40 = 1.0 - alpha;
    acc += g40 * w40 * o40;
    alpha += g40 * w40 * o40;
    done = opaque ? done : (i + 41);
    opaque = (alpha > 0.9);
    float o41 = s41 * p.alphaMul;
    float g41 = opaque ? 0.0 : 1.0;
    float w41 = 1.0 - alpha;
    acc += g41 * w41 * o41;
    alpha += g41 * w41 * o41;
    done = opaque ? done : (i + 42);
    opaque = (alpha > 0.9);
    float o42 = s42 * p.alphaMul;
    float g42 = opaque ? 0.0 : 1.0;
    float w42 = 1.0 - alpha;
    acc += g42 * w42 * o42;
    alpha += g42 * w42 * o42;
    done = opaque ? done : (i + 43);
    opaque = (alpha > 0.9);
    float o43 = s43 * p.alphaMul;
    float g43 = opaque ? 0.0 : 1.0;
    float w43 = 1.0 - alpha;
    acc += g43 * w43 * o43;
    alpha += g43 * w43 * o43;
    done = opaque ? done : (i + 44);
    opaque = (alpha > 0.9);
    float o44 = s44 * p.alphaMul;
    float g44 = opaque ? 0.0 : 1.0;
    float w44 = 1.0 - alpha;
    acc += g44 * w44 * o44;
    alpha += g44 * w44 * o44;
    done = opaque ? done : (i + 45);
    opaque = (alpha > 0.9);
    float o45 = s45 * p.alphaMul;
    float g45 = opaque ? 0.0 : 1.0;
    float w45 = 1.0 - alpha;
    acc += g45 * w45 * o45;
    alpha += g45 * w45 * o45;
    done = opaque ? done : (i + 46);
    opaque = (alpha > 0.9);
    float o46 = s46 * p.alphaMul;
    float g46 = opaque ? 0.0 : 1.0;
    float w46 = 1.0 - alpha;
    acc += g46 * w46 * o46;
    alpha += g46 * w46 * o46;
    done = opaque ? done : (i + 47);
    opaque = (alpha > 0.9);
    float o47 = s47 * p.alphaMul;
    float g47 = opaque ? 0.0 : 1.0;
    float w47 = 1.0 - alpha;
    acc += g47 * w47 * o47;
    alpha += g47 * w47 * o47;
    done = opaque ? done : (i + 48);
    opaque = (alpha > 0.9);
#endif
    if (opaque) break;
  }
  // Scalar break-aware tail (harness style: per-sample breaks).
  for (; i < stepsB; ++i) {
    float s = vol.sample(smp, base + float(i) * d, level(0.0)).r;
    float o = s * p.alphaMul;
    float w = 1.0 - alpha;
    acc += w * o;
    alpha += w * o;
    done = i + 1;
    if (alpha > 0.9) break;
  }
#elif MARCH_VARIANT == 7
  // V12: S29-style scheduling fix (PERFORMANCE_INVESTIGATION.md section 9).
  // The bare single-loop march compiles to a schedule that serializes each
  // 3-D linear fetch (0.42 ns/sample). Co-compiling >= 4 extra vol.sample
  // calls in the same shader flips the backend to an overlapped schedule
  // (0.06 ns/sample) without changing executed work. The dead path never runs
  // (deadFlag is always 0), but the samples co-compile with the hot loop.
  // S30_4 style: extra taps in an else path INSIDE the loop (the doc's
  // "4 extra taps in an else path" structure that measured 5.86 ms).
  for (int i = 0; i < steps; ++i) {
    float s = vol.sample(smp, base + float(i) * d, level(0.0)).r;
    float o = s * p.alphaMul;
    float w = 1.0 - alpha;
    acc += w * o;
    alpha += w * o;
    done = i + 1;
    if (p.deadFlag != 0.0) {
      float4 t0 = vol.sample(smp, base + float(i + 0) * d, level(0.0));
      float4 t1 = vol.sample(smp, base + float(i + 1) * d, level(0.0));
      float4 t2 = vol.sample(smp, base + float(i + 2) * d, level(0.0));
      float4 t3 = vol.sample(smp, base + float(i + 3) * d, level(0.0));
      acc += (t0.r + t1.r + t2.r + t3.r) * p.deadFlag;
    }
    if (alpha > 0.9) break;
  }
#elif MARCH_VARIANT == 12
  // V17: binned-pass march (Experiment C). Four per-frame passes, each
  // covering pixels whose previous-frame done is in (bucketLo, bucketHi]; all
  // other pixels discard instantly. In-bucket pixels march exactly like V0 but
  // capped at bucketCap (>= their true done in a static scene, so results are
  // identical). Each pass also writes done to histTex (color(1), R16Unorm) so
  // the bucket assignment refreshes every frame.
  {
    uint2 px = uint2(clamp(uint(in.uv.x * float(p.width)), 0u, uint(p.width) - 1u),
                     clamp(uint(in.uv.y * float(p.height)), 0u, uint(p.height) - 1u));
    float prevF = histTex.read(px).r;
    int prevDone = int(prevF * 65535.0 + 0.5);
    if (prevDone <= p.bucketLo || prevDone > p.bucketHi)
    {
      discard_fragment();
      return MarchOut{ float4(0.0, 0.0, 0.0, 0.0), 0.0 };
    }
    const int capSteps = min(steps, p.bucketCap);
    for (int i = 0; i < capSteps; ++i) {
      float s = vol.sample(smp, base + float(i) * d).r;
      float o = s * p.alphaMul;
      float w = 1.0 - alpha;
      acc += w * o;
      alpha += w * o;
      done = i + 1;
      if (alpha > 0.9) break;
    }
    return MarchOut{ float4(acc / float(steps), float(done) / 255.0, 0.0, 1.0),
                     float(done) / 65535.0 };
  }
#endif
#if MARCH_VARIANT == 12
  return MarchOut{ float4(acc / float(steps), float(done) / 255.0, 0.0, 1.0),
                   float(done) / 65535.0 };
#else
  return float4(acc / float(steps), float(done) / 255.0, 0.0, 1.0);
#endif
}

// Shared march body used by the persistent compute kernel (V6). Identical
// per-pixel work to V1: explicit level(0.0), divergent per-lane bound, opacity
// early-exit.
static float4 marchRay(texture3d<float, access::sample> vol,
                       sampler smp,
                       uint2 uv,
                       constant Params& p) {
  float2 fuv = float2(uv) / float2(p.width, p.height);
  float2 ndc = fuv * 2.0 - 1.0;
  float3 eye = float3(0.5, 0.5, -0.35);
  float3 dir = normalize(float3(ndc * 2.5, 1.0));
  float ca = cos(0.35), sa = sin(0.35);
  dir = float3(ca * dir.x + sa * dir.z, dir.y, -sa * dir.x + ca * dir.z);
  float3 inv = 1.0 / dir;
  float3 t0 = (float3(0.0) - eye) * inv;
  float3 t1 = (float3(1.0) - eye) * inv;
  float tEnter = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
  float tExit  = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
  if (tExit <= 0.0 || tEnter >= tExit) {
    return float4(0.0, 0.0, 0.0, 0.0);
  }
  int geomSteps = max(1, int(ceil((tExit - tEnter) / p.step)));
  int steps = (p.mode == 1) ? min(geomSteps, p.fixedSteps) : geomSteps;
  float3 base = eye + dir * (tEnter + 0.5 * p.step);
  float3 d = dir * p.step;
  float acc = 0.0;
  float alpha = 0.0;
  int done = 0;
  for (int i = 0; i < steps; ++i) {
    float s = vol.sample(smp, base + float(i) * d, level(0.0)).r;
    float o = s * p.alphaMul;
    float w = 1.0 - alpha;
    acc += w * o;
    alpha += w * o;
    done = i + 1;
    if (alpha > 0.9) break;
  }
  return float4(acc / float(steps), float(done) / 255.0, 0.0, 1.0);
}

// V6: persistent-threads kernel (Aila et al.). A fixed pool of threads pulls
// CHUNK-sized work items off an atomic counter instead of one thread per
// pixel. A ray that dies at step 10 immediately takes the next pixel instead
// of padding to 283 — the SIMD divergence tax becomes idle-free work.
// CHUNK is a function constant (default 256) so the harness can sweep it; it
// MUST be a multiple of the threadgroup size so no thread idles in a chunk.
constant uint kChunk [[function_constant(1)]];
kernel void dvrPersistent(texture3d<float, access::sample> vol [[texture(0)]],
                          texture2d<float, access::write> color [[texture(1)]],
                          sampler smp [[sampler(0)]],
                          device atomic_uint& nextWork [[buffer(0)]],
                          constant Params& p [[buffer(1)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint tgSize [[threads_per_threadgroup]]) {
  const uint CHUNK = kChunk;
  threadgroup uint base;
  while (true) {
    if (tid == 0) {
      base = atomic_fetch_add_explicit(&nextWork, CHUNK, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint b = base;
    if (b >= (uint)p.nPixels) return;
    for (uint i = tid; i < CHUNK; i += tgSize) {
      uint pix = b + i;
      if (pix >= (uint)p.nPixels) continue;
      uint2 uv = uint2(pix % (uint)p.width, pix / (uint)p.width);
      color.write(marchRay(vol, smp, uv, p), uv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// V7 diagnostic: plain one-pixel-per-thread compute march, no atomics, no
// persistent loop. Isolates whether compute texture marching itself is slow
// or whether the persistent/atomic structure is the cost.
kernel void dvrOneShot(texture3d<float, access::sample> vol [[texture(0)]],
                       texture2d<float, access::write> color [[texture(1)]],
                       sampler smp [[sampler(0)]],
                       constant Params& p [[buffer(1)]],
                       uint gid [[thread_position_in_grid]]) {
  if (gid >= (uint)p.nPixels) return;
  uint2 uv = uint2(gid % (uint)p.width, gid / (uint)p.width);
  color.write(marchRay(vol, smp, uv, p), uv);
}

// V8 diagnostic: write-only one-shot, no texture reads at all. Floor cost of
// compute dispatch + color.write at this resolution.
kernel void dvrWriteOnly(texture2d<float, access::write> color [[texture(1)]],
                         constant Params& p [[buffer(1)]],
                         uint gid [[thread_position_in_grid]]) {
  if (gid >= (uint)p.nPixels) return;
  uint2 uv = uint2(gid % (uint)p.width, gid / (uint)p.width);
  color.write(float4((float)gid / (float)p.nPixels, 0.5, 0.5, 1.0), uv);
}

// V10 diagnostic: tiled one-shot. One threadgroup per WxH screen tile (like
// the tile-based fragment renderer), thread (i,j) covers the tile pixel. Tests
// whether compute's 1D row mapping destroys volume L2 locality.
kernel void dvrTiled(texture3d<float, access::sample> vol [[texture(0)]],
                     texture2d<float, access::write> color [[texture(1)]],
                     sampler smp [[sampler(0)]],
                     constant Params& p [[buffer(1)]],
                     uint2 tgid [[threadgroup_position_in_grid]],
                     uint2 gtid [[thread_position_in_threadgroup]]) {
  uint2 uv = uint2(tgid.x * (uint)p.tileW + gtid.x,
                   tgid.y * (uint)p.tileH + gtid.y);
  if (uv.x >= (uint)p.width || uv.y >= (uint)p.height) return;
  color.write(marchRay(vol, smp, uv, p), uv);
}

// V11: tiled persistent-threads. One threadgroup per kTile x kTile screen
// tile (volume L2 locality preserved), but the threadgroup has only
// kComputeTG threads (1D). Threads steal work from a per-tile atomic counter,
// so a thread whose ray dies at step 10 immediately grabs the next pixel in
// the tile instead of padding to 283 -- the divergence waste (max/mean within
// a SIMD group) is recovered while keeping texture access local.
kernel void dvrTiledPersist(texture3d<float, access::sample> vol [[texture(0)]],
                            texture2d<float, access::write> color [[texture(1)]],
                            sampler smp [[sampler(0)]],
                            constant Params& p [[buffer(1)]],
                            uint2 tgid [[threadgroup_position_in_grid]],
                            uint2 gtid [[thread_position_in_threadgroup]]) {
  threadgroup atomic_uint base;
  if (gtid.x == 0 && gtid.y == 0)
    atomic_store_explicit(&base, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint tileW = (uint)p.tileW, tileH = (uint)p.tileH;
  const uint tileSize = tileW * tileH;
  const uint2 origin = uint2(tgid.x * tileW, tgid.y * tileH);
  while (true) {
    const uint b = atomic_fetch_add_explicit(&base, 1, memory_order_relaxed);
    if (b >= tileSize) return;
    const uint2 uv = uint2(origin.x + b % tileW, origin.y + b / tileW);
    if (uv.x >= (uint)p.width || uv.y >= (uint)p.height) continue;
    color.write(marchRay(vol, smp, uv, p), uv);
  }
}

// V9 diagnostic: march with the sample replaced by a constant (no texture
// traffic). Isolates the texture-read cost from the loop/ALU cost.
kernel void dvrNoTex(texture2d<float, access::write> color [[texture(1)]],
                     constant Params& p [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
  if (gid >= (uint)p.nPixels) return;
  uint2 uv = uint2(gid % (uint)p.width, gid / (uint)p.width);
  float2 fuv = float2(uv) / float2(p.width, p.height);
  float2 ndc = fuv * 2.0 - 1.0;
  float3 eye = float3(0.5, 0.5, -0.35);
  float3 dir = normalize(float3(ndc * 2.5, 1.0));
  float ca = cos(0.35), sa = sin(0.35);
  dir = float3(ca * dir.x + sa * dir.z, dir.y, -sa * dir.x + ca * dir.z);
  float3 inv = 1.0 / dir;
  float3 t0 = (float3(0.0) - eye) * inv;
  float3 t1 = (float3(1.0) - eye) * inv;
  float tEnter = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
  float tExit  = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
  int geomSteps = (tExit > 0.0 && tEnter < tExit)
    ? max(1, int(ceil((tExit - tEnter) / p.step))) : 1;
  int steps = (p.mode == 1) ? min(geomSteps, p.fixedSteps) : geomSteps;
  float3 base = eye + dir * (tEnter + 0.5 * p.step);
  float3 d = dir * p.step;
  float acc = 0.0;
  float alpha = 0.0;
  int done = 0;
  for (int i = 0; i < steps; ++i) {
    float s = 0.5; // constant instead of vol.sample
    float o = s * p.alphaMul;
    float w = 1.0 - alpha;
    acc += w * o;
    alpha += w * o;
    done = i + 1;
    if (alpha > 0.9) break;
  }
  color.write(float4(acc / float(steps), float(done) / 255.0, 0.0, 1.0), uv);
}
)";

struct MetalState
{
  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> q = nil;
  std::vector<id<MTLRenderPipelineState>> ps; // one per march variant (0..4)
  id<MTLComputePipelineState> cps = nil;      // persistent kernel (variant 5)
  id<MTLComputePipelineState> cpsOneShot = nil; // diagnostic one-shot kernel
  id<MTLComputePipelineState> cpsWriteOnly = nil; // no-read floor diagnostic
  id<MTLComputePipelineState> cpsNoTex = nil; // march-without-texture diagnostic
  id<MTLComputePipelineState> cpsTiled = nil; // tiled one-shot diagnostic
  id<MTLComputePipelineState> cpsTiledPersist = nil; // tiled persistent-threads
  id<MTLSamplerState> smp = nil;
  id<MTLTexture> volTex = nil, colorTex = nil, histTex[2] = {nil, nil};
  id<MTLBuffer> vbuf = nil, workBuf = nil;    // atomic work counter (compute)
};

static bool setupMetal(MetalState& s, const std::vector<uint8_t>& vol, int kMax)
{
  s.dev = MTLCreateSystemDefaultDevice();
  if (!s.dev)
  {
    std::fprintf(stderr, "Metal: no device\n");
    return false;
  }
  s.q = [s.dev newCommandQueue];

  // One library per variant: MARCH_VARIANT is baked in as a preprocessor macro
  // so each PSO gets the exact code shape we want to measure.
  const int nVariants = 13;
  for (int v = 0; v < nVariants; ++v)
  {
    NSError* err = nil;
    MTLCompileOptions* copts = [[MTLCompileOptions alloc] init];
    copts.preprocessorMacros = @{ @"MARCH_VARIANT" : @(v) };
    id<MTLLibrary> lib = [s.dev newLibraryWithSource:kMetalSrc options:copts error:&err];
    if (!lib)
    {
      std::fprintf(stderr, "Metal: library compile failed for variant %d: %s\n", v,
        err.localizedDescription.UTF8String);
      return false;
    }
    id<MTLFunction> vs = [lib newFunctionWithName:@"vsfull"];
    id<MTLFunction> fs = nil;
    if (v == 4)
    {
      MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
      [cv setConstantValue:&kMax type:MTLDataTypeInt atIndex:0];
      fs = [lib newFunctionWithName:@"march" constantValues:cv error:&err];
    }
    else
    {
      fs = [lib newFunctionWithName:@"march"];
    }
    if (!fs)
    {
      std::fprintf(stderr, "Metal: no march function for variant %d: %s\n", v,
        err.localizedDescription.UTF8String);
      return false;
    }
    MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = vs;
    pd.fragmentFunction = fs;
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    if (v == 12)
    {
      pd.colorAttachments[1].pixelFormat = MTLPixelFormatR16Unorm;
    }
    id<MTLRenderPipelineState> p = [s.dev newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!p)
    {
      std::fprintf(stderr, "Metal: pipeline failed for variant %d: %s\n", v,
        err.localizedDescription.UTF8String);
      return false;
    }
    s.ps.push_back(p);
  }

  // Compute pipeline for the persistent-threads variant (built from the
  // variant-5 library, which still contains the kernel).
  {
    NSError* err = nil;
    MTLCompileOptions* copts = [[MTLCompileOptions alloc] init];
    copts.preprocessorMacros = @{ @"MARCH_VARIANT" : @(5) };
    id<MTLLibrary> lib = [s.dev newLibraryWithSource:kMetalSrc options:copts error:&err];
    if (!lib)
    {
      std::fprintf(stderr, "Metal: library compile failed for compute: %s\n",
        err.localizedDescription.UTF8String);
      return false;
    }
    id<MTLFunction> kf = nil;
    {
      MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
      int c = kChunk;
      [cv setConstantValue:&c type:MTLDataTypeUInt atIndex:1];
      kf = [lib newFunctionWithName:@"dvrPersistent" constantValues:cv error:&err];
    }
    if (!kf)
    {
      std::fprintf(stderr, "Metal: no dvrPersistent kernel: %s\n",
        err.localizedDescription.UTF8String);
      return false;
    }
    s.cps = [s.dev newComputePipelineStateWithFunction:kf error:&err];
    if (!s.cps)
    {
      std::fprintf(stderr, "Metal: compute pipeline failed: %s\n",
        err.localizedDescription.UTF8String);
      return false;
    }
    id<MTLFunction> kf2 = [lib newFunctionWithName:@"dvrOneShot"];
    if (!kf2)
    {
      std::fprintf(stderr, "Metal: no dvrOneShot kernel\n");
      return false;
    }
    s.cpsOneShot = [s.dev newComputePipelineStateWithFunction:kf2 error:&err];
    if (!s.cpsOneShot)
    {
      std::fprintf(stderr, "Metal: one-shot pipeline failed: %s\n",
        err.localizedDescription.UTF8String);
      return false;
    }
    for (const char* name : { "dvrWriteOnly", "dvrNoTex", "dvrTiled", "dvrTiledPersist" })
    {
      NSString* nsName = [NSString stringWithUTF8String:name];
      id<MTLFunction> kf3 = [lib newFunctionWithName:nsName];
      id<MTLComputePipelineState> p3 =
        [s.dev newComputePipelineStateWithFunction:kf3 error:&err];
      if (!p3)
      {
        std::fprintf(stderr, "Metal: %s pipeline failed: %s\n", name,
          err.localizedDescription.UTF8String);
        return false;
      }
      if (strcmp(name, "dvrWriteOnly") == 0) s.cpsWriteOnly = p3;
      else if (strcmp(name, "dvrNoTex") == 0) s.cpsNoTex = p3;
      else if (strcmp(name, "dvrTiled") == 0) s.cpsTiled = p3;
      else s.cpsTiledPersist = p3;
    }
  }

  MTLSamplerDescriptor* sd = [[MTLSamplerDescriptor alloc] init];
  sd.minFilter = MTLSamplerMinMagFilterLinear;
  sd.magFilter = MTLSamplerMinMagFilterLinear;
  sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
  sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
  sd.rAddressMode = MTLSamplerAddressModeClampToEdge;
  s.smp = [s.dev newSamplerStateWithDescriptor:sd];

  MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
  td.textureType = MTLTextureType3D;
  td.pixelFormat = MTLPixelFormatR8Unorm;
  td.width = kVolX;
  td.height = kVolY;
  td.depth = kVolZ;
  td.mipmapLevelCount = 1;
  td.usage = MTLTextureUsageShaderRead;
  td.storageMode = MTLStorageModePrivate;
  td.allowGPUOptimizedContents = kOptContents; // NO matches the app's volume layout
  s.volTex = [s.dev newTextureWithDescriptor:td];
  MTLTextureDescriptor* rt =
    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
      width:kRT height:kRT mipmapped:NO];
  rt.storageMode = MTLStorageModePrivate;
  rt.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderWrite;
  s.colorTex = [s.dev newTextureWithDescriptor:rt];
  // Per-frame done history for the binned-pass variant (V17): R16Unorm, exact
  // done values (0..283) read via texture.read in the binned fragment.
  MTLTextureDescriptor* ht =
    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Unorm
      width:kRT height:kRT mipmapped:NO];
  ht.storageMode = MTLStorageModePrivate;
  ht.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  s.histTex[0] = [s.dev newTextureWithDescriptor:ht];
  s.histTex[1] = [s.dev newTextureWithDescriptor:ht];

  const float tri[] = { -1.f, -1.f, 3.f, -1.f, -1.f, 3.f };
  s.vbuf = [s.dev newBufferWithBytes:tri length:sizeof(tri) options:MTLResourceStorageModeShared];

  // Atomic work counter for the persistent kernel. Initialized to 0.
  s.workBuf = [s.dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
  memset(s.workBuf.contents, 0, 4);

  id<MTLBuffer> upload = [s.dev newBufferWithBytes:vol.data()
    length:(NSUInteger)(vol.size()) options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb = [s.q commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  [blit copyFromBuffer:upload sourceOffset:0 sourceBytesPerRow:kVolX
    sourceBytesPerImage:(NSUInteger)kVolX * kVolY
    sourceSize:MTLSizeMake(kVolX, kVolY, kVolZ)
    toTexture:s.volTex destinationSlice:0 destinationLevel:0
    destinationOrigin:MTLOriginMake(0, 0, 0)];
  [blit endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  return true;
}

static double timeMetal(MetalState& s, int mode, int fixedSteps, int variant)
{
  struct { int mode, fixedSteps; float step, alphaMul; int width, height, nPixels, tileW, tileH; float deadFlag; int bucketLo, bucketHi, bucketCap; } params =
    { mode, fixedSteps, kStep, kAlphaMul, kRT, kRT, kRT * kRT, kTile, kTile, 0.0f, -1, -1, 0 };
  if (variant == 17)
  {
    // V17: binned-pass march (Experiment C). Per-frame 4 passes, each covering
    // pixels whose previous-frame done fell in (bucketLo, bucketHi]; the march
    // is capped at bucketCap (>= the true done, so results are identical to the
    // single-pass baseline). Bucket caps are quartiles of the done histogram,
    // computed once per mode from a full-cap bucket-build frame readback.
    static int bCaps[2][3] = {{0, 0, 0}, {0, 0, 0}};
    static bool bReady[2] = {false, false};
    const int m = (mode == 1) ? 1 : 0;
    if (!bReady[m])
    {
      // Bucket-build frame: one full-cap binned pass (cap 1e6, covers every
      // pixel, identical output to the baseline) that also fills histTex[0].
      struct { int mode, fixedSteps; float step, alphaMul; int width, height, nPixels, tileW, tileH; float deadFlag; int bucketLo, bucketHi, bucketCap; } bp =
        { mode, fixedSteps, kStep, kAlphaMul, kRT, kRT, kRT * kRT, kTile, kTile, 0.0f, -1, 100000, 100000 };
      MTLRenderPassDescriptor* rb = [[MTLRenderPassDescriptor alloc] init];
      rb.colorAttachments[0].texture = s.colorTex;
      rb.colorAttachments[0].loadAction = MTLLoadActionClear;
      rb.colorAttachments[0].storeAction = MTLStoreActionStore;
      rb.colorAttachments[1].texture = s.histTex[0];
      rb.colorAttachments[1].loadAction = MTLLoadActionClear;
      rb.colorAttachments[1].storeAction = MTLStoreActionStore;
      id<MTLCommandBuffer> cbb = [s.q commandBuffer];
      id<MTLRenderCommandEncoder> eb = [cbb renderCommandEncoderWithDescriptor:rb];
      [eb setRenderPipelineState:s.ps[12]];
      [eb setVertexBuffer:s.vbuf offset:0 atIndex:0];
      [eb setFragmentTexture:s.volTex atIndex:0];
      [eb setFragmentTexture:s.histTex[0] atIndex:1];
      [eb setFragmentSamplerState:s.smp atIndex:0];
      [eb setFragmentBytes:&bp length:sizeof(bp) atIndex:0];
      [eb drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [eb endEncoding];
      [cbb commit];
      [cbb waitUntilCompleted];
      // Read back histTex[0] (R16Unorm): exact done per covered pixel.
      id<MTLBuffer> staging =
        [s.dev newBufferWithLength:(NSUInteger)kRT * kRT * 2 options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> cbh = [s.q commandBuffer];
      id<MTLBlitCommandEncoder> blh = [cbh blitCommandEncoder];
      [blh copyFromTexture:s.histTex[0] sourceSlice:0 sourceLevel:0
        sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(kRT, kRT, 1)
        toBuffer:staging destinationOffset:0 destinationBytesPerRow:(NSUInteger)kRT * 2
        destinationBytesPerImage:(NSUInteger)kRT * kRT * 2];
      [blh endEncoding];
      [cbh commit];
      [cbh waitUntilCompleted];
      const unsigned char* px = (const unsigned char*)staging.contents;
      uint64_t hist[512] = {0};
      long long cov = 0;
      for (size_t i = 0; i < (size_t)kRT * kRT * 2; i += 2)
      {
        uint16_t v = (uint16_t)(px[i] | (px[i + 1] << 8));
        int done = (int)((double)v * 65535.0 / 65535.0 + 0.5);
        if (v > 0) { ++cov; if (done < 512) ++hist[done]; }
      }
      int caps[3];
      for (int q = 0; q < 3; ++q)
      {
        const uint64_t target = (uint64_t)((double)cov * (double)(q + 1) / 4.0);
        uint64_t acc = 0;
        int d = 1;
        for (; d < 512; ++d)
        {
          acc += hist[d];
          if (acc >= target) break;
        }
        caps[q] = (d < 512) ? d : 512;
      }
      bCaps[m][0] = caps[0];
      bCaps[m][1] = caps[1];
      bCaps[m][2] = caps[2];
      bReady[m] = true;
      fprintf(stderr, "V17 mode=%d bucket caps %d %d %d (cov %lld)\n",
        mode, caps[0], caps[1], caps[2], cov);
    }
    // Binning only pays when the done histogram is strongly bimodal: the 25th
    // percentile cap must be <= half the 75th percentile (so the short buckets
    // actually truncate warps), and the uncapped long bucket must not absorb
    // most of the pixels. Otherwise (fine SD, uniform-ish marches) fall back to
    // a single baseline pass — binning would only add pass overhead.
    const bool useBinned = (mode == 0) &&
      (bCaps[m][0] <= bCaps[m][2] / 2) && (bCaps[m][2] - bCaps[m][0] >= 16);
    if (!useBinned)
    {
      struct { int mode, fixedSteps; float step, alphaMul; int width, height, nPixels, tileW, tileH; float deadFlag; int bucketLo, bucketHi, bucketCap; } bp =
        { mode, fixedSteps, kStep, kAlphaMul, kRT, kRT, kRT * kRT, kTile, kTile, 0.0f, -1, -1, 0 };
      MTLRenderPassDescriptor* rps = [[MTLRenderPassDescriptor alloc] init];
      rps.colorAttachments[0].texture = s.colorTex;
      rps.colorAttachments[0].loadAction = (kHarness == 1) ? MTLLoadActionClear : MTLLoadActionDontCare;
      rps.colorAttachments[0].storeAction = MTLStoreActionStore;
      auto runSingle = [&]() {
        id<MTLCommandBuffer> cb = [s.q commandBuffer];
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rps];
        [enc setRenderPipelineState:s.ps[0]];
        [enc setVertexBuffer:s.vbuf offset:0 atIndex:0];
        [enc setFragmentTexture:s.volTex atIndex:0];
        [enc setFragmentSamplerState:s.smp atIndex:0];
        [enc setFragmentBytes:&bp length:sizeof(bp) atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
      };
      for (int i = 0; i < kWarmup; ++i)
      {
        runSingle();
      }
      if (kHarness == 1)
      {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kFrames; ++i)
        {
          runSingle();
        }
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / kFrames;
      }
      double total = 0;
      for (int i = 0; i < kFrames; ++i)
      {
        id<MTLCommandBuffer> cb = [s.q commandBuffer];
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rps];
        [enc setRenderPipelineState:s.ps[0]];
        [enc setVertexBuffer:s.vbuf offset:0 atIndex:0];
        [enc setFragmentTexture:s.volTex atIndex:0];
        [enc setFragmentSamplerState:s.smp atIndex:0];
        [enc setFragmentBytes:&bp length:sizeof(bp) atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        total += cb.GPUEndTime - cb.GPUStartTime;
      }
      return total / kFrames * 1000.0;
    }
    // Four passes per frame. Pass p covers prevDone in (lo_p, hi_p], cap_p.
    struct { int mode, fixedSteps; float step, alphaMul; int width, height, nPixels, tileW, tileH; float deadFlag; int bucketLo, bucketHi, bucketCap; }
      passP[4] = {
        { mode, fixedSteps, kStep, kAlphaMul, kRT, kRT, kRT * kRT, kTile, kTile, 0.0f, -1, bCaps[m][0], bCaps[m][0] },
        { mode, fixedSteps, kStep, kAlphaMul, kRT, kRT, kRT * kRT, kTile, kTile, 0.0f, bCaps[m][0], bCaps[m][1], bCaps[m][1] },
        { mode, fixedSteps, kStep, kAlphaMul, kRT, kRT, kRT * kRT, kTile, kTile, 0.0f, bCaps[m][1], bCaps[m][2], bCaps[m][2] },
        { mode, fixedSteps, kStep, kAlphaMul, kRT, kRT, kRT * kRT, kTile, kTile, 0.0f, bCaps[m][2], 100000, 100000 },
      };
    // histTex is double-buffered: each frame reads the previous frame's done
    // (readTex) while writing its own (writeTex); no read/write hazard.
    id<MTLTexture> readTex = s.histTex[0];
    id<MTLTexture> writeTex = s.histTex[1];
    auto runBinned = [&]() {
      for (int p = 0; p < 4; ++p)
      {
        MTLRenderPassDescriptor* rpb = [[MTLRenderPassDescriptor alloc] init];
        rpb.colorAttachments[0].texture = s.colorTex;
        rpb.colorAttachments[0].loadAction = (p == 0) ? MTLLoadActionClear : MTLLoadActionLoad;
        rpb.colorAttachments[0].storeAction = MTLStoreActionStore;
        rpb.colorAttachments[1].texture = writeTex;
        rpb.colorAttachments[1].loadAction = (p == 0) ? MTLLoadActionClear : MTLLoadActionLoad;
        rpb.colorAttachments[1].storeAction = MTLStoreActionStore;
        id<MTLCommandBuffer> cb = [s.q commandBuffer];
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpb];
        [enc setRenderPipelineState:s.ps[12]];
        [enc setVertexBuffer:s.vbuf offset:0 atIndex:0];
        [enc setFragmentTexture:s.volTex atIndex:0];
        [enc setFragmentTexture:readTex atIndex:1];
        [enc setFragmentSamplerState:s.smp atIndex:0];
        [enc setFragmentBytes:&passP[p] length:sizeof(passP[p]) atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
      }
      id<MTLTexture> t = readTex;
      readTex = writeTex;
      writeTex = t;
    };
    for (int i = 0; i < kWarmup; ++i)
    {
      runBinned();
    }
    if (kHarness == 1)
    {
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < kFrames; ++i)
      {
        runBinned();
      }
      const auto t1 = std::chrono::steady_clock::now();
      return std::chrono::duration<double, std::milli>(t1 - t0).count() / kFrames;
    }
    // Per-frame GPU timestamps: time the 4-pass command buffer chain.
    double gtotal = 0;
    for (int i = 0; i < kFrames; ++i)
    {
      id<MTLCommandBuffer> cb = [s.q commandBuffer];
      for (int p = 0; p < 4; ++p)
      {
        MTLRenderPassDescriptor* rpb = [[MTLRenderPassDescriptor alloc] init];
        rpb.colorAttachments[0].texture = s.colorTex;
        rpb.colorAttachments[0].loadAction = (p == 0) ? MTLLoadActionClear : MTLLoadActionLoad;
        rpb.colorAttachments[0].storeAction = MTLStoreActionStore;
        rpb.colorAttachments[1].texture = writeTex;
        rpb.colorAttachments[1].loadAction = (p == 0) ? MTLLoadActionClear : MTLLoadActionLoad;
        rpb.colorAttachments[1].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpb];
        [enc setRenderPipelineState:s.ps[12]];
        [enc setVertexBuffer:s.vbuf offset:0 atIndex:0];
        [enc setFragmentTexture:s.volTex atIndex:0];
        [enc setFragmentTexture:readTex atIndex:1];
        [enc setFragmentSamplerState:s.smp atIndex:0];
        [enc setFragmentBytes:&passP[p] length:sizeof(passP[p]) atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
      }
      [cb commit];
      [cb waitUntilCompleted];
      gtotal += cb.GPUEndTime - cb.GPUStartTime;
      id<MTLTexture> t = readTex;
      readTex = writeTex;
      writeTex = t;
    }
    return gtotal / kFrames * 1000.0;
  }
  if (variant >= 6 && variant <= 11)
  {
    // Compute dispatches:
    //  V6 persistent-threads over global atomic chunks
    //  V7 one-shot: one thread per pixel, full march (texture)
    //  V8 write-only: floor cost of dispatch + color.write
    //  V9 no-tex march: ALU/loop cost without texture traffic
    //  V10 tiled one-shot: 32x32 threadgroup tiles, tests volume L2 locality
    //  V11 tiled persistent: tile-locality + per-tile work stealing
    id<MTLComputePipelineState> cps = nil;
    switch (variant)
    {
      case 6: cps = s.cps; break;
      case 7: cps = s.cpsOneShot; break;
      case 8: cps = s.cpsWriteOnly; break;
      case 9: cps = s.cpsNoTex; break;
      case 10: cps = s.cpsTiled; break;
      case 11: cps = s.cpsTiledPersist; break;
    }
    const bool persistent = (variant == 6);
    const bool tiled = (variant == 10 || variant == 11);
    const bool tiledPersist = (variant == 11);
    const NSUInteger tileW = (NSUInteger)kTile, tileH = (NSUInteger)kTile;
    auto run = [&]() {
      memset(s.workBuf.contents, 0, 4);
      id<MTLCommandBuffer> cb = [s.q commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:cps];
      [enc setTexture:s.volTex atIndex:0];
      [enc setTexture:s.colorTex atIndex:1];
      [enc setSamplerState:s.smp atIndex:0];
      [enc setBytes:&params length:sizeof(params) atIndex:1];
      [enc setBuffer:s.workBuf offset:0 atIndex:0];
      if (persistent)
      {
        // Persistent: fill the GPU with a fixed pool, each looping on chunks.
        const NSUInteger tgSize = (NSUInteger)kComputeTG;
        const NSUInteger tgs = kComputeGroups > 0 ? (NSUInteger)kComputeGroups : s.dev.maxThreadsPerThreadgroup.width * 8 / tgSize;
        [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
      }
      else if (tiled)
      {
        // One threadgroup per screen tile. V10: thread (i,j) covers tile pixel
        // (threadgroup = tile x tile). V11: kComputeTG threads (1D) steal work
        // from a per-tile atomic counter.
        const NSUInteger gx = ((NSUInteger)kRT + tileW - 1) / tileW;
        const NSUInteger gy = ((NSUInteger)kRT + tileH - 1) / tileH;
        if (tiledPersist)
          [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)kComputeTG, 1, 1)];
        else
          [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
            threadsPerThreadgroup:MTLSizeMake(tileW, tileH, 1)];
      }
      else
      {
        // Full grid: one thread per pixel.
        const NSUInteger tgSize = (NSUInteger)kComputeTG;
        const NSUInteger threads = (NSUInteger)kRT * kRT;
        const NSUInteger tgs = (threads + tgSize - 1) / tgSize;
        [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
      }
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
    };
    for (int i = 0; i < kWarmup; ++i)
    {
      run();
    }
    if (kHarness == 1)
    {
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < kFrames; ++i)
      {
        run();
      }
      const auto t1 = std::chrono::steady_clock::now();
      return std::chrono::duration<double, std::milli>(t1 - t0).count() / kFrames;
    }
    double total = 0;
    for (int i = 0; i < kFrames; ++i)
    {
      id<MTLCommandBuffer> cb = [s.q commandBuffer];
      memset(s.workBuf.contents, 0, 4);
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:cps];
      [enc setTexture:s.volTex atIndex:0];
      [enc setTexture:s.colorTex atIndex:1];
      [enc setSamplerState:s.smp atIndex:0];
      [enc setBytes:&params length:sizeof(params) atIndex:1];
      [enc setBuffer:s.workBuf offset:0 atIndex:0];
      if (persistent)
      {
        const NSUInteger tgSize = (NSUInteger)kComputeTG;
        const NSUInteger tgs = kComputeGroups > 0 ? (NSUInteger)kComputeGroups : s.dev.maxThreadsPerThreadgroup.width * 8 / tgSize;
        [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
      }
      else if (tiled)
      {
        const NSUInteger gx = ((NSUInteger)kRT + tileW - 1) / tileW;
        const NSUInteger gy = ((NSUInteger)kRT + tileH - 1) / tileH;
        if (tiledPersist)
          [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)kComputeTG, 1, 1)];
        else
          [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
            threadsPerThreadgroup:MTLSizeMake(tileW, tileH, 1)];
      }
      else
      {
        const NSUInteger tgSize = (NSUInteger)kComputeTG;
        const NSUInteger threads = (NSUInteger)kRT * kRT;
        const NSUInteger tgs = (threads + tgSize - 1) / tgSize;
        [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
      }
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      total += cb.GPUEndTime - cb.GPUStartTime;
    }
    return total / kFrames * 1000.0;
  }
  MTLRenderPassDescriptor* rpd = [[MTLRenderPassDescriptor alloc] init];
  rpd.colorAttachments[0].texture = s.colorTex;
  // DontCare, not Clear, in the fair harness: the shader writes every pixel
  // (misses write alpha=0), so clearing is wasted work GL does not do. The
  // old harness used Clear; kHarness=1 restores that.
  rpd.colorAttachments[0].loadAction = (kHarness == 1) ? MTLLoadActionClear : MTLLoadActionDontCare;
  rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

  auto run = [&]() {
    id<MTLCommandBuffer> cb = [s.q commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
    [enc setRenderPipelineState:s.ps[(variant == 12) ? 7 : (variant == 13) ? 8 : (variant == 14) ? 9 : (variant == 15) ? 10 : (variant == 16) ? 11 : (variant == 17) ? 12 : variant]];
    [enc setVertexBuffer:s.vbuf offset:0 atIndex:0];
    [enc setFragmentTexture:s.volTex atIndex:0];
    [enc setFragmentSamplerState:s.smp atIndex:0];
    [enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
  };
  for (int i = 0; i < kWarmup; ++i)
  {
    run();
  }
  if (kHarness == 1)
  {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kFrames; ++i)
    {
      run();
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / kFrames;
  }
  // GPU timestamps, not host wall time: GPUStartTime/GPUEndTime measure the
  // actual device execution of the render pass, comparable to GL_TIME_ELAPSED.
  double total = 0;
  for (int i = 0; i < kFrames; ++i)
  {
    id<MTLCommandBuffer> cb = [s.q commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
    [enc setRenderPipelineState:s.ps[(variant == 12) ? 7 : (variant == 13) ? 8 : (variant == 14) ? 9 : (variant == 15) ? 10 : (variant == 16) ? 11 : (variant == 17) ? 12 : variant]];
    [enc setVertexBuffer:s.vbuf offset:0 atIndex:0];
    [enc setFragmentTexture:s.volTex atIndex:0];
    [enc setFragmentSamplerState:s.smp atIndex:0];
    [enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    total += cb.GPUEndTime - cb.GPUStartTime;
  }
  return total / kFrames * 1000.0; // seconds -> ms
}

static void readbackMetal(MetalState& s, long long* covered, double* meanSteps)
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
  long long cov = 0, sum = 0;
  for (size_t i = 0; i < (size_t)kRT * kRT * 4; i += 4)
  {
    if (px[i + 3] != 0)
    {
      ++cov;
      sum += px[i + 1];
    }
  }
  *covered = cov;
  *meanSteps = cov ? (double)sum / cov : 0.0;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
  if (argc > 1) kRT = std::atoi(argv[1]);
  if (argc > 2) kFrames = std::atoi(argv[2]);
  if (argc > 3) kStep = std::atof(argv[3]);
  if (argc > 4) kVolX = kVolY = std::atoi(argv[4]);
  if (argc > 5) kVolZ = std::atoi(argv[5]);
  if (argc > 6) kAlphaMul = std::atof(argv[6]);
  if (argc > 7) kFixedOverride = std::atoi(argv[7]);
  if (argc > 8) kOptContents = (BOOL)std::atoi(argv[8]);
  if (argc > 9) kMaxConstant = std::atoi(argv[9]);
  if (argc > 10) kHarness = std::atoi(argv[10]);
  if (argc > 11) kChunk = std::atoi(argv[11]);
  if (argc > 12) kComputeTG = std::atoi(argv[12]);
  if (argc > 13) kComputeGroups = std::atoi(argv[13]);
  if (argc > 14) kTile = std::atoi(argv[14]);

  // CPU-simulate the same geometry to get the frame-mean step count (the
  // "fixed" mode runs every hit pixel at this count, so total work matches the
  // divergent mode exactly) and the distribution summary.
  long long hitPx = 0, sumSteps = 0, maxSteps = 0;
  for (int py = 0; py < kRT; ++py)
  {
    for (int px = 0; px < kRT; ++px)
    {
      const int s = cpuSteps(px, py, kRT);
      if (s > 0)
      {
        ++hitPx;
        sumSteps += s;
        maxSteps = std::max(maxSteps, (long long)s);
      }
    }
  }
  const double meanSteps = hitPx ? (double)sumSteps / hitPx : 0.0;
  const int fixedSteps = kFixedOverride > 0 ? kFixedOverride : std::max(1, (int)std::lround(meanSteps));
  DBG("rt=%d frames=%d step=%.4f hit=%lld meanSteps=%.1f maxSteps=%lld fixedSteps=%d\n",
    kRT, kFrames, kStep, hitPx, meanSteps, maxSteps, fixedSteps);
  std::printf("rt=%d  hitPx=%lld  mean=%.1f  max=%lld  fixedSteps=%d\n",
    kRT, hitPx, meanSteps, maxSteps, fixedSteps);
  std::printf("config                GL ms/f   Metal ms/f   M/GL\n");

  const std::vector<uint8_t> vol = makeVolume();
  GLState gl;
  if (!setupGL(gl))
  {
    return 1;
  }
  uploadGLVolume(gl, vol);
  MetalState m;
  if (!setupMetal(m, vol, kMaxConstant))
  {
    return 1;
  }

  // Variant sweep. V0 = baseline (the losing 1.13 config), V1 = explicit LOD
  // only, V2 = uniform header (simd_max), V3 = +whole-group exit, V4 =
  // compile-time ceiling (kMax constant, no unroll). GL mirrors V0/V1 via
  // uUseLod (implicit texture() vs textureLod); V2-V4 are Metal loop-shape
  // changes with no GL equivalent (GL already generates a uniform SIMT loop).
  const char* variantNames[] = {
    "V0 baseline       ",
    "V1 explicit lod   ",
    "V2 simd_max header",
    "V3 +group exit    ",
    "V4 kMax=288 const ",
    "V5 chunked reconv  ",
    "V6 persistent comp ",
    "V7 one-shot diag  ",
    "V8 write-only    ",
    "V9 no-tex march  ",
    "V10 tiled one-shot",
    "V11 tiled persist ",
    "V12 S29 sched fix ",
    "V13 mv9 batch-8  ",
    "V14 batch-16     ",
    "V15 batch-32     ",
    "V16 batch-48     ",
    "V17 binned-4pass ",
  };
  for (int v = 0; v < 18; ++v)
  {
    const int useLod = (v >= 1) ? 1 : 0; // GL: implicit until V1, explicit after
    // Interleave the two modes within each backend to cancel drift, and the two
    // backends across modes: divergent GL, divergent Metal, fixed GL, fixed
    // Metal, repeated. Report per-mode averages.
    double glDiv = 0, mDiv = 0, glFix = 0, mFix = 0;
    long long glDivCov = 0, mDivCov = 0, glFixCov = 0, mFixCov = 0;
    double glDivMean = 0, mDivMean = 0, glFixMean = 0, mFixMean = 0;
    const int rounds = 3;
    for (int r = 0; r < rounds; ++r)
    {
      glDiv += timeGL(gl, 0, fixedSteps, useLod);
      if (r == 0) readbackGL(gl, &glDivCov, &glDivMean);
      mDiv += timeMetal(m, 0, fixedSteps, v);
      if (r == 0) readbackMetal(m, &mDivCov, &mDivMean);
      glFix += timeGL(gl, 1, fixedSteps, useLod);
      if (r == 0) readbackGL(gl, &glFixCov, &glFixMean);
      mFix += timeMetal(m, 1, fixedSteps, v);
      if (r == 0) readbackMetal(m, &mFixCov, &mFixMean);
    }
    glDiv /= rounds; mDiv /= rounds; glFix /= rounds; mFix /= rounds;
    std::printf("%s divergent %6.2f %6.2f %5.2f | fixed %6.2f %6.2f %5.2f\n",
      variantNames[v], glDiv, mDiv, mDiv / glDiv, glFix, mFix, mFix / glFix);
    std::printf("%s parity div %lld/%.1f %lld/%.1f | fix %lld/%.1f %lld/%.1f\n",
      variantNames[v], glDivCov, glDivMean, mDivCov, mDivMean,
      glFixCov, glFixMean, mFixCov, mFixMean);
  }
  return 0;
}