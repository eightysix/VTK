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
struct Params { int mode; int fixedSteps; float step; float alphaMul; };
constant int kMax [[function_constant(0)]]; // compile-time ceiling (variant 4)
vertex FragIn vsfull(const device float2* pos [[buffer(0)]], uint vid [[vertex_id]]) {
  FragIn o;
  float2 p = pos[vid];
  o.pos = float4(p, 0.0, 1.0);
  o.uv = p * 0.5 + 0.5;
  return o;
}
fragment float4 march(texture3d<float, access::sample> vol [[texture(0)]],
                      sampler smp [[sampler(0)]],
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
#elif MARCH_VARIANT == 1
  // V1: step A — explicit level(0.0). Same loop shape, same work, but no
  // implicit gradient/quad machinery. Should be strictly cheaper than V0.
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
#endif
  return float4(acc / float(steps), float(done) / 255.0, 0.0, 1.0);
}
)";

struct MetalState
{
  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> q = nil;
  std::vector<id<MTLRenderPipelineState>> ps; // one per march variant (0..4)
  id<MTLSamplerState> smp = nil;
  id<MTLTexture> volTex = nil, colorTex = nil;
  id<MTLBuffer> vbuf = nil;
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
  const int nVariants = 6;
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
    id<MTLRenderPipelineState> p = [s.dev newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!p)
    {
      std::fprintf(stderr, "Metal: pipeline failed for variant %d: %s\n", v,
        err.localizedDescription.UTF8String);
      return false;
    }
    s.ps.push_back(p);
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
  s.colorTex = [s.dev newTextureWithDescriptor:rt];

  const float tri[] = { -1.f, -1.f, 3.f, -1.f, -1.f, 3.f };
  s.vbuf = [s.dev newBufferWithBytes:tri length:sizeof(tri) options:MTLResourceStorageModeShared];

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
  MTLRenderPassDescriptor* rpd = [[MTLRenderPassDescriptor alloc] init];
  rpd.colorAttachments[0].texture = s.colorTex;
  // DontCare, not Clear, in the fair harness: the shader writes every pixel
  // (misses write alpha=0), so clearing is wasted work GL does not do. The
  // old harness used Clear; kHarness=1 restores that.
  rpd.colorAttachments[0].loadAction = (kHarness == 1) ? MTLLoadActionClear : MTLLoadActionDontCare;
  rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

  struct { int mode, fixedSteps; float step, alphaMul; } params = { mode, fixedSteps, kStep, kAlphaMul };
  auto run = [&]() {
    id<MTLCommandBuffer> cb = [s.q commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
    [enc setRenderPipelineState:s.ps[variant]];
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
    [enc setRenderPipelineState:s.ps[variant]];
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
  };
  for (int v = 0; v < 6; ++v)
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