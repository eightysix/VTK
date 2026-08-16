// Minimal self-contained repro of the Metal vs OpenGL 3D-volume sampling
// throughput gap (no VTK). Loads the SAME R8 512x512x1794 synthetic volume into
// a GL_TEXTURE_3D and an MTLTexture3D, then times a minimal raymarch fragment
// shader with nearest/linear filtering and marching along Z or X, using the
// same host-clock + per-frame sync timing as the VTK harness.
//
// Build:
//   clang++ -std=c++17 -fobjc-arc -O2 -framework Metal -framework OpenGL \
//           -framework Foundation minimal_repro.mm -o minimal_repro
//
// Run: ./minimal_repro [frames]

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

static const int kVolX = 512;
static const int kVolY = 512;
static const int kVolZ = 1794; // long axis
static const int kWin = 275;   // 275x275 = 75,625 fragments
static const int kSteps = 604; // matches the VTK scene (~45.6 M samples/frame)
static int kWarmup = 30;
static int kFrames = 120;

// ---------------------------------------------------------------------------
#define DBG(...) std::fprintf(stderr, "[repro] " __VA_ARGS__), std::fflush(stderr)
// Data mode probes whether Metal's hardware lossless texture compression is
// what penalizes the linear filter: "random" is incompressible, "zero" and
// "gradient" are very compressible.
static std::vector<uint8_t> makeVolume(const char* mode)
{
  std::vector<uint8_t> v((size_t)kVolX * kVolY * kVolZ);
  if (std::strcmp(mode, "zero") == 0)
  {
    return v;
  }
  if (std::strcmp(mode, "gradient") == 0)
  {
    for (size_t i = 0; i < v.size(); ++i)
    {
      const uint32_t z = (uint32_t)(i / ((size_t)kVolX * kVolY));
      const uint32_t r = (uint32_t)(i % ((size_t)kVolX * kVolY));
      const uint32_t y = r / kVolX;
      const uint32_t x = r % kVolX;
      v[i] = (uint8_t)((z * 37u + y * 11u + x * 7u) & 0xffu);
    }
    return v;
  }
  uint32_t s = 12345u;
  for (size_t i = 0; i < v.size(); ++i)
  {
    s = s * 1664525u + 1013904223u;
    v[i] = (uint8_t)(s >> 24);
  }
  return v;
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
uniform int uSteps;
uniform int uAxis; // 2 = march +Z, 0 = march +X
void main() {
  float step = 1.0 / float(uSteps);
  vec3 p = (uAxis == 2) ? vec3(vUV, 0.0) : vec3(0.0, vUV.x, vUV.y);
  vec3 d = (uAxis == 2) ? vec3(0.0, 0.0, step) : vec3(step, 0.0, 0.0);
  float acc = 0.0;
  for (int i = 0; i < uSteps; ++i) {
    acc += texture(volumeTex, p).r;
    p += d;
  }
  outColor = vec4(acc / float(uSteps));
}
)";

struct GLState
{
  CGLContextObj ctx = nullptr;
  GLuint fbo = 0, colorTex = 0, vao = 0, vbo = 0, volTex = 0, prog = 0;
  GLint uSteps = -1, uAxis = -1;
};

static bool setupGL(GLState& s)
{
  CGLPixelFormatAttribute attrs[] = {
    kCGLPFAOpenGLProfile, (CGLPixelFormatAttribute)kCGLOGLPVersion_GL4_Core,
    kCGLPFAColorSize, (CGLPixelFormatAttribute)24,
    kCGLPFAAlphaSize, (CGLPixelFormatAttribute)8,
    kCGLPFAAccelerated,
    (CGLPixelFormatAttribute)0,
  };
  CGLPixelFormatObj pf = nullptr;
  GLint npix = 0;
  if (CGLChoosePixelFormat(attrs, &pf, &npix) != kCGLNoError || pf == nullptr)
  {
    std::fprintf(stderr, "GL: no accelerated GL 4.x pixel format (falling back)\n");
    CGLPixelFormatAttribute alt[] = {
      kCGLPFAOpenGLProfile, (CGLPixelFormatAttribute)kCGLOGLPVersion_GL4_Core,
      (CGLPixelFormatAttribute)0,
    };
    if (CGLChoosePixelFormat(alt, &pf, &npix) != kCGLNoError || pf == nullptr)
    {
      std::fprintf(stderr, "GL: could not create pixel format\n");
      return false;
    }
  }
  if (CGLCreateContext(pf, nullptr, &s.ctx) != kCGLNoError || s.ctx == nullptr)
  {
    std::fprintf(stderr, "GL: could not create context\n");
    return false;
  }
  CGLSetCurrentContext(s.ctx);

DBG("GL: framebuffer\n");
  glGenFramebuffers(1, &s.fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, s.fbo);
  glGenTextures(1, &s.colorTex);
  glBindTexture(GL_TEXTURE_2D, s.colorTex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, kWin, kWin, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, s.colorTex, 0);
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
  {
    std::fprintf(stderr, "GL: FBO incomplete\n");
    return false;
  }

  GLuint vs = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vs, 1, &kGLVertSrc, nullptr);
  glCompileShader(vs);
  GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fs, 1, &kGLFragSrc, nullptr);
  glCompileShader(fs);
  char log[1024];
  for (GLuint sh : { vs, fs })
  {
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
      glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
      std::fprintf(stderr, "GL shader compile error:\n%s\n", log);
      return false;
    }
  }
  s.prog = glCreateProgram();
  glAttachShader(s.prog, vs);
  glAttachShader(s.prog, fs);
  glLinkProgram(s.prog);
  GLint ok = 0;
  glGetProgramiv(s.prog, GL_LINK_STATUS, &ok);
  if (!ok)
  {
    glGetProgramInfoLog(s.prog, sizeof(log), nullptr, log);
    std::fprintf(stderr, "GL link error:\n%s\n", log);
    return false;
  }
  s.uSteps = glGetUniformLocation(s.prog, "uSteps");
  s.uAxis = glGetUniformLocation(s.prog, "uAxis");

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
  return true;
}

static void uploadGLVolume(GLState& s, const std::vector<uint8_t>& v)
{
  glBindTexture(GL_TEXTURE_3D, s.volTex);
  DBG("GL: upload volume\n");
  glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, kVolX, kVolY, kVolZ, 0, GL_RED, GL_UNSIGNED_BYTE,
    v.data());
  DBG("GL: upload done\n");
}

static double timeGL(GLState& s, GLenum filter, int axis)
{
  glBindFramebuffer(GL_FRAMEBUFFER, s.fbo);
  glViewport(0, 0, kWin, kWin);
  glUseProgram(s.prog);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_3D, s.volTex);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, filter);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, filter);
  glUniform1i(s.uSteps, kSteps);
  glUniform1i(s.uAxis, axis);
  glBindVertexArray(s.vao);
  for (int i = 0; i < kWarmup; ++i)
  {
    glDrawArrays(GL_TRIANGLES, 0, 3);
  }
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

static float readbackGL(GLState& s)
{
  std::vector<unsigned char> px((size_t)kWin * kWin * 4);
  glReadPixels(0, 0, kWin, kWin, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
  float sum = 0.f;
  for (size_t i = 0; i < px.size(); i += 4)
  {
    sum += px[i];
  }
  return sum;
}

// ---------------------------------------------------------------------------
static NSString* kMetalSrc = @R"(
#include <metal_stdlib>
using namespace metal;
struct FragIn { float4 pos [[position]]; float2 uv; };
struct Params { int axis; int steps; int strategy; };
vertex FragIn vsfull(const device float2* pos [[buffer(0)]], uint vid [[vertex_id]]) {
  FragIn o;
  float2 p = pos[vid];
  o.pos = float4(p, 0.0, 1.0);
  o.uv = p * 0.5 + 0.5;
  return o;
}
fragment float4 march(texture3d<float, access::sample> vol [[texture(0)]],
                      texture2d_array<float, access::sample> arr [[texture(1)]],
                      sampler smp [[sampler(0)]],
                      sampler smpN [[sampler(1)]],
                      constant Params& p [[buffer(0)]],
                      FragIn in [[stage_in]]) {
  float step = 1.0 / float(p.steps);
  float3 base = (p.axis == 2) ? float3(in.uv, 0.0) : float3(0.0, in.uv.x, in.uv.y);
  float3 d = (p.axis == 2) ? float3(0.0, 0.0, step) : float3(step, 0.0, 0.0);
  float acc = 0.0;
  // Six branches, co-compiled in one function: nearest/linear (implicit LOD),
  // the same with explicit level(0), and the manual 8-tap path. Co-compiling
  // these independent sample sites is what makes the MSL backend overlap the
  // fetches (see report section 9): a lone bare sampling loop runs ~0.42
  // ns/sample regardless of explicit LOD (S33/S34), this co-compiled form
  // runs ~0.06 ns/sample (faster than GL).
  // Strategy values:
  //   -1 nearest (implicit LOD) | 0 linear S29 (implicit LOD) | 1 manual 8-tap
  //    2 linear S29 + explicit level(0) | 3 nearest + explicit level(0)
  if (p.strategy == -1) {
    for (int i = 0; i < p.steps; ++i) {
      acc += vol.sample(smpN, base + float(i) * d).r;
    }
  } else if (p.strategy == 3) {
    for (int i = 0; i < p.steps; ++i) {
      acc += vol.sample(smpN, base + float(i) * d, level(0.0)).r;
    }
  } else if (p.strategy == 0) {
    for (int i = 0; i < p.steps; ++i) {
      acc += vol.sample(smp, base + float(i) * d).r;
    }
  } else if (p.strategy == 2) {
    for (int i = 0; i < p.steps; ++i) {
      acc += vol.sample(smp, base + float(i) * d, level(0.0)).r;
    }
  } else {
    const float3 vdims = float3(512.0, 512.0, 1794.0);
    const float3 vdims1 = vdims - 1.0;
    for (int i = 0; i < p.steps; ++i) {
      float3 tc = (base + float(i) * d) * vdims - 0.5;
      float3 f = floor(tc);
      float3 w = tc - f;
      float3 f0 = clamp(f, 0.0, vdims1) / vdims;
      float3 f1 = clamp(f + 1.0, 0.0, vdims1) / vdims;
      float v000 = vol.sample(smpN, float3(f0.x, f0.y, f0.z)).r;
      float v100 = vol.sample(smpN, float3(f1.x, f0.y, f0.z)).r;
      float v010 = vol.sample(smpN, float3(f0.x, f1.y, f0.z)).r;
      float v110 = vol.sample(smpN, float3(f1.x, f1.y, f0.z)).r;
      float v001 = vol.sample(smpN, float3(f0.x, f0.y, f1.z)).r;
      float v101 = vol.sample(smpN, float3(f1.x, f0.y, f1.z)).r;
      float v011 = vol.sample(smpN, float3(f0.x, f1.y, f1.z)).r;
      float v111 = vol.sample(smpN, float3(f1.x, f1.y, f1.z)).r;
      float v0 = mix(mix(v000, v100, w.x), mix(v010, v110, w.x), w.y);
      float v1 = mix(mix(v001, v101, w.x), mix(v011, v111, w.x), w.y);
      acc += mix(v0, v1, w.z);
    }
  }
  return float4(acc / float(p.steps));
}
// Isolated single-loop baselines: ONLY this loop is in the compiled fragment
// function (no co-compiled siblings), so they isolate whether explicit
// level(0) alone defeats the fetch-serialization (vs needing co-compiled
// samples). Implicit-LOD bare loop was the ~0.42 ns/sample "slow" baseline.
fragment float4 march_bare(texture3d<float, access::sample> vol [[texture(0)]],
                           sampler smp [[sampler(0)]],
                           constant Params& p [[buffer(0)]],
                           FragIn in [[stage_in]]) {
  float step = 1.0 / float(p.steps);
  float3 base = (p.axis == 2) ? float3(in.uv, 0.0) : float3(0.0, in.uv.x, in.uv.y);
  float3 d = (p.axis == 2) ? float3(0.0, 0.0, step) : float3(step, 0.0, 0.0);
  float acc = 0.0;
  for (int i = 0; i < p.steps; ++i) {
    acc += vol.sample(smp, base + float(i) * d).r;
  }
  return float4(acc / float(p.steps));
}
fragment float4 march_bare_lod(texture3d<float, access::sample> vol [[texture(0)]],
                               sampler smp [[sampler(0)]],
                               constant Params& p [[buffer(0)]],
                               FragIn in [[stage_in]]) {
  float step = 1.0 / float(p.steps);
  float3 base = (p.axis == 2) ? float3(in.uv, 0.0) : float3(0.0, in.uv.x, in.uv.y);
  float3 d = (p.axis == 2) ? float3(0.0, 0.0, step) : float3(step, 0.0, 0.0);
  float acc = 0.0;
  for (int i = 0; i < p.steps; ++i) {
    acc += vol.sample(smp, base + float(i) * d, level(0.0)).r;
  }
  return float4(acc / float(p.steps));
}
)";

struct MetalState
{
  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> q = nil;
  id<MTLRenderPipelineState> ps = nil, psBare = nil, psBareLod = nil;
  id<MTLSamplerState> smpLinear = nil, smpNearest = nil;
  id<MTLTexture> volTex = nil, vol2d = nil, colorTex = nil;
  id<MTLBuffer> vbuf = nil;
};

static bool setupMetal(MetalState& s)
{
  s.dev = MTLCreateSystemDefaultDevice();
  if (!s.dev)
  {
    std::fprintf(stderr, "Metal: no device\n");
    return false;
  }
  s.q = [s.dev newCommandQueue];

  NSError* err = nil;
  MTLCompileOptions* copts = [[MTLCompileOptions alloc] init];
  const char* fmEnv = std::getenv("REPRO_FASTMATH");
  if (fmEnv && std::strcmp(fmEnv, "0") == 0)
  {
    copts.fastMathEnabled = NO;
    DBG("Metal: fast math DISABLED\n");
  }
  id<MTLLibrary> lib = [s.dev newLibraryWithSource:kMetalSrc options:copts error:&err];
  if (!lib)
  {
    std::fprintf(stderr, "Metal: library compile failed: %s\n", err.localizedDescription.UTF8String);
    return false;
  }
  id<MTLFunction> vs = [lib newFunctionWithName:@"vsfull"];
  id<MTLFunction> fs = [lib newFunctionWithName:@"march"];
  MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
  pd.vertexFunction = vs;
  pd.fragmentFunction = fs;
  pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
  s.ps = [s.dev newRenderPipelineStateWithDescriptor:pd error:&err];
  if (!s.ps)
  {
    std::fprintf(stderr, "Metal: pipeline failed: %s\n", err.localizedDescription.UTF8String);
    return false;
  }
  for (int fi = 0; fi < 2; ++fi)
  {
    id<MTLFunction> bfs = [lib newFunctionWithName:(fi == 0) ? @"march_bare" : @"march_bare_lod"];
    MTLRenderPipelineDescriptor* bpd = [[MTLRenderPipelineDescriptor alloc] init];
    bpd.vertexFunction = vs;
    bpd.fragmentFunction = bfs;
    bpd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> st = [s.dev newRenderPipelineStateWithDescriptor:bpd error:&err];
    if (!st)
    {
      std::fprintf(stderr, "Metal: bare pipeline failed: %s\n", err.localizedDescription.UTF8String);
      return false;
    }
    if (fi == 0)
    {
      s.psBare = st;
    }
    else
    {
      s.psBareLod = st;
    }
  }

  for (int i = 0; i < 2; ++i)
  {
    MTLSamplerDescriptor* sd = [[MTLSamplerDescriptor alloc] init];
    sd.minFilter = sd.magFilter = (i == 0) ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    sd.sAddressMode = sd.tAddressMode = sd.rAddressMode = MTLSamplerAddressModeClampToEdge;
    id<MTLSamplerState> st = [s.dev newSamplerStateWithDescriptor:sd];
    if (i == 0)
    {
      s.smpLinear = st;
    }
    else
    {
      s.smpNearest = st;
    }
  }

  MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
  td.textureType = MTLTextureType3D;
  td.pixelFormat = MTLPixelFormatR8Unorm;
  td.width = kVolX;
  td.height = kVolY;
  td.depth = kVolZ;
  td.mipmapLevelCount = 1;
  td.storageMode = MTLStorageModeShared;
  td.usage = MTLTextureUsageShaderRead;
  s.volTex = [s.dev newTextureWithDescriptor:td];

  MTLTextureDescriptor* ad = [[MTLTextureDescriptor alloc] init];
  ad.textureType = MTLTextureType2DArray;
  ad.pixelFormat = MTLPixelFormatR8Unorm;
  ad.width = kVolX;
  ad.height = kVolY;
  ad.arrayLength = kVolZ;
  ad.mipmapLevelCount = 1;
  ad.storageMode = MTLStorageModeShared;
  ad.usage = MTLTextureUsageShaderRead;
  s.vol2d = [s.dev newTextureWithDescriptor:ad];

  MTLTextureDescriptor* cd = [[MTLTextureDescriptor alloc] init];
  cd.textureType = MTLTextureType2D;
  cd.pixelFormat = MTLPixelFormatRGBA8Unorm;
  cd.width = kWin;
  cd.height = kWin;
  cd.mipmapLevelCount = 1;
  cd.usage = MTLTextureUsageRenderTarget;
  s.colorTex = [s.dev newTextureWithDescriptor:cd];

  const float tri[] = { -1.f, -1.f, 3.f, -1.f, -1.f, 3.f };
  s.vbuf = [s.dev newBufferWithBytes:tri length:sizeof(tri) options:MTLResourceStorageModeShared];
  return true;
}

static void uploadMetalVolume(MetalState& s, const std::vector<uint8_t>& v)
{
  MTLRegion reg = MTLRegionMake3D(0, 0, 0, kVolX, kVolY, kVolZ);
  [s.volTex replaceRegion:reg mipmapLevel:0 slice:0 withBytes:v.data()
      bytesPerRow:kVolX bytesPerImage:(NSUInteger)kVolX * kVolY];
  MTLRegion r2 = MTLRegionMake2D(0, 0, kVolX, kVolY);
  const size_t sliceBytes = (size_t)kVolX * kVolY;
  for (NSUInteger i = 0; i < (NSUInteger)kVolZ; ++i)
  {
    [s.vol2d replaceRegion:r2 mipmapLevel:0 slice:i withBytes:v.data() + i * sliceBytes
        bytesPerRow:kVolX bytesPerImage:sliceBytes];
  }
}

static double timeMetal(MetalState& s, id<MTLRenderPipelineState> ps, int strategy, int axis)
{
  struct Params
  {
    int axis;
    int steps;
    int strategy;
  } params = { axis, kSteps, strategy };

  for (int i = 0; i < kWarmup; ++i)
  {
    @autoreleasepool
    {
      MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
      rpd.colorAttachments[0].texture = s.colorTex;
      rpd.colorAttachments[0].loadAction = MTLLoadActionDontCare;
      rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
      id<MTLCommandBuffer> cb = [s.q commandBuffer];
      id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rpd];
      [e setRenderPipelineState:ps];
      [e setVertexBuffer:s.vbuf offset:0 atIndex:0];
      [e setFragmentTexture:s.volTex atIndex:0];
      [e setFragmentTexture:s.vol2d atIndex:1];
      [e setFragmentSamplerState:s.smpLinear atIndex:0];
      [e setFragmentSamplerState:s.smpNearest atIndex:1];
      [e setFragmentBytes:&params length:sizeof(params) atIndex:0];
      [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [e endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
    }
  }
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kFrames; ++i)
  {
    @autoreleasepool
    {
      MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
      rpd.colorAttachments[0].texture = s.colorTex;
      rpd.colorAttachments[0].loadAction = MTLLoadActionDontCare;
      rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
      id<MTLCommandBuffer> cb = [s.q commandBuffer];
      id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rpd];
      [e setRenderPipelineState:ps];
      [e setVertexBuffer:s.vbuf offset:0 atIndex:0];
      [e setFragmentTexture:s.volTex atIndex:0];
      [e setFragmentTexture:s.vol2d atIndex:1];
      [e setFragmentSamplerState:s.smpLinear atIndex:0];
      [e setFragmentSamplerState:s.smpNearest atIndex:1];
      [e setFragmentBytes:&params length:sizeof(params) atIndex:0];
      [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [e endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / kFrames;
}

static float readbackMetal(MetalState& s)
{
  std::vector<unsigned char> px((size_t)kWin * kWin * 4);
  MTLRegion reg = MTLRegionMake2D(0, 0, kWin, kWin);
  [s.colorTex getBytes:px.data() bytesPerRow:(NSUInteger)kWin * 4 fromRegion:reg mipmapLevel:0];
  float sum = 0.f;
  for (size_t i = 0; i < px.size(); i += 4)
  {
    sum += px[i];
  }
  return sum;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
  if (argc > 1)
  {
    kFrames = std::atoi(argv[1]);
  }
  const double samplesPerFrame = (double)kWin * kWin * kSteps;

  const char* dataMode = std::getenv("REPRO_DATA");
  if (!dataMode)
  {
    dataMode = "random";
  }
  std::vector<uint8_t> vol = makeVolume(dataMode);
  std::fprintf(stderr, "volume %dx%dx%d = %.1f MB, %d frags x %d steps = %.2f M samples/frame\n",
    kVolX, kVolY, kVolZ, vol.size() / 1e6, kWin * kWin, kSteps, samplesPerFrame / 1e6);

  DBG("setup GL...\n");
  GLState gl;
  if (!setupGL(gl))
  {
    return 1;
  }
  DBG("upload GL volume...\n");
  uploadGLVolume(gl, vol);
  DBG("upload GL done\n");

  DBG("setup Metal...\n");
  MetalState mtl;
  if (!setupMetal(mtl))
  {
    return 1;
  }
  DBG("upload Metal volume...\n");
  uploadMetalVolume(mtl, vol);
  DBG("upload Metal done\n");

  std::printf("%-8s %-7s %-6s %10s %12s %12s\n", "backend", "filter", "march", "ms/frame",
    "ns/sample", "readback");
  std::printf("--------------------------------------------------------------\n");

  const char* stratEnv = std::getenv("REPRO_STRATEGY");
  const int linStrategy = (stratEnv && std::strcmp(stratEnv, "manual") == 0) ? 1 : 0;

  const struct
  {
    const char* name;
    GLenum glFilter;
    id<MTLRenderPipelineState> pipe;
    int mtlStrategy;
  } cfgs[] = {
    { "nearest", GL_NEAREST, mtl.ps, -1 },
    { "nearest+lod0", GL_NEAREST, mtl.ps, 3 },
    { "linear", GL_LINEAR, mtl.ps, linStrategy },
    { "lin+lod0", GL_LINEAR, mtl.ps, 2 },
    { "bare", GL_LINEAR, mtl.psBare, 0 },
    { "bare+lod0", GL_LINEAR, mtl.psBareLod, 0 },
  };

  for (int round = 0; round < 2; ++round)
  {
    std::printf("========== round %d ==========\n", round);
    for (const char* axisName : { "Z", "X" })
    {
      const int axis = (axisName[0] == 'Z') ? 2 : 0;
      for (const auto& c : cfgs)
      {
        const double glMs = timeGL(gl, c.glFilter, axis);
        const double mtMs = timeMetal(mtl, c.pipe, c.mtlStrategy, axis);
        const float glRb = readbackGL(gl);
        const float mtRb = readbackMetal(mtl);
        const char* mtlName = c.mtlStrategy == -1 ? "nearest" : c.mtlStrategy == 0 ? "lin-nat" :
                               c.mtlStrategy == 1 ? "lin-man" : c.mtlStrategy == 2 ? "lin+lod0" :
                               c.mtlStrategy == 3 ? "nearest+lod0" : "?";
        if (c.pipe == mtl.psBare || c.pipe == mtl.psBareLod)
        {
          mtlName = c.name;
        }
        std::printf("%-8s %-10s %-6s %10.2f %12.2f %12.0f\n", "gl", c.name, axisName, glMs,
          glMs * 1e6 / samplesPerFrame, glRb);
        std::printf("%-8s %-10s %-6s %10.2f %12.2f %12.0f\n", "metal", mtlName, axisName, mtMs,
          mtMs * 1e6 / samplesPerFrame, mtRb);
      }
    }
  }
  return 0;
}
