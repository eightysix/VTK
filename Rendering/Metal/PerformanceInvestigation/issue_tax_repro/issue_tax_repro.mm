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
static int kWarmup = 10;
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
uniform float uStep;
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
#ifdef L1FETCH
  // Frozen coordinate: every tap hits the same L1-resident texel — pure
  // sampler issue rate, zero DRAM streaming.
  for (int i = 0; i < steps; ++i) {
    acc += textureLod(volumeTex, base, 0.0).r;
    done = i + 1;
  }
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
  GLuint fbo = 0, colorTex = 0, vao = 0, vbo = 0, volTex = 0;
  GLuint progMarch = 0, progL1 = 0;
};

static bool compileGL(GLState& s, GLuint* prog, bool l1fetch)
{
  const char* versionEnd = strstr(kGLFragSrc, "\n");
  std::string src;
  src.append(kGLFragSrc, versionEnd - kGLFragSrc + 1);
  src += l1fetch ? "#define L1FETCH 1\n" : "";
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
  if (!compileGL(s, &s.progMarch, false) || !compileGL(s, &s.progL1, true)) return false;
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
  return true;
}

static double timeGL(GLState& s, bool l1fetch)
{
  glBindFramebuffer(GL_FRAMEBUFFER, s.fbo);
  glViewport(0, 0, kRT, kRT);
  glUseProgram(l1fetch ? s.progL1 : s.progMarch);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_3D, s.volTex);
  glUniform1f(glGetUniformLocation(l1fetch ? s.progL1 : s.progMarch, "uStep"), kStep);
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
  id<MTLTexture> volTex = nil, colorTex = nil;
  id<MTLBuffer> vbuf = nil;
  id<MTLSamplerState> smp = nil;
  id<MTLRenderPipelineState> psMarch = nil, psL1 = nil;
};

static NSString* kMetalSrc = @R"(
#include <metal_stdlib>
using namespace metal;
struct FragIn { float4 pos [[position]]; float2 uv; };
struct Params { float step; };
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
  if (tExit <= 0.0 || tEnter >= tExit) return float4(0.0);
  int steps = max(1, int(ceil((tExit - tEnter) / p.step)));
  float3 base = eye + dir * (tEnter + 0.5 * p.step);
  float3 d = dir * p.step;
  float acc = 0.0;
  float alpha = 0.0;
  int done = 0;
#if L1FETCH
  for (int i = 0; i < steps; ++i) {
    acc += vol.sample(smp, base, level(0.0)).r;
    done = i + 1;
  }
#else
  int i = 0;
  do {
    float s = vol.sample(smp, base + float(i) * d).r;
    float o = s * 1.0;
    float w = 1.0 - alpha;
    acc += w * o;
    alpha += w * o;
    ++i;
    done = i;
  } while (i < steps && alpha <= 0.9);
#endif
  return float4(acc / float(steps), float(done) / 255.0, 0.0, 1.0);
}
)";

static bool setupMetal(MetalState& s, const std::vector<uint8_t>& vol)
{
  s.dev = MTLCreateSystemDefaultDevice();
  s.q = [s.dev newCommandQueue];
  NSError* err = nil;
  for (int v = 0; v < 2; ++v)
  {
    MTLCompileOptions* copts = [[MTLCompileOptions alloc] init];
    copts.preprocessorMacros = @{ @"L1FETCH" : (v == 1 ? @"1" : @"0") };
    copts.languageVersion = MTLLanguageVersion3_2;
    copts.mathMode = MTLMathModeFast;
    id<MTLLibrary> lib = [s.dev newLibraryWithSource:kMetalSrc options:copts error:&err];
    if (!lib) return false;
    id<MTLFunction> fs = [lib newFunctionWithName:@"march"];
    id<MTLFunction> vs = [lib newFunctionWithName:@"vsfull"];
    MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = vs;
    pd.fragmentFunction = fs;
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> p = [s.dev newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!p) return false;
    if (v == 0) s.psMarch = p; else s.psL1 = p;
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
  return true;
}

static double timeMetal(MetalState& s, bool l1fetch)
{
  struct { float step; } params = { kStep };
  MTLRenderPassDescriptor* rpd = [[MTLRenderPassDescriptor alloc] init];
  rpd.colorAttachments[0].texture = s.colorTex;
  rpd.colorAttachments[0].loadAction = MTLLoadActionDontCare;
  rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
  id<MTLRenderPipelineState> ps = l1fetch ? s.psL1 : s.psMarch;
  auto run = [&]() {
    id<MTLCommandBuffer> cb = [s.q commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
    [enc setRenderPipelineState:ps];
    [enc setVertexBuffer:s.vbuf offset:0 atIndex:0];
    [enc setFragmentTexture:s.volTex atIndex:0];
    [enc setFragmentSamplerState:s.smp atIndex:0];
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
  const std::vector<uint8_t> vol = makeVolume();
  GLState gl;
  if (!setupGL(gl, vol)) { std::fprintf(stderr, "GL setup failed\n"); return 1; }
  MetalState m;
  if (!setupMetal(m, vol)) { std::fprintf(stderr, "Metal setup failed\n"); return 1; }
  std::printf("rt=%d frames=%d step=%.5f\n", kRT, kFrames, kStep);
  std::printf("%-22s %10s %11s %7s\n", "pair", "GL ms/f", "Metal ms/f", "M/GL");
  // Interleaved rounds cancel machine drift (same protocol as the main harness).
  double gMarch = 0, mMarch = 0,gL1 = 0, mL1 = 0;
  long long gcM = 0, mcM = 0;
  double gmM = 0, mmM = 0;
  for (int r = 0; r < 3; ++r)
  {
    gMarch += timeGL(gl, false);
    mMarch += timeMetal(m, false);
    if (r == 0) readbackGL(gl, &gcM, &gmM);
    if (r == 0) readbackMetal(m, &mcM, &mmM);
    gL1 += timeGL(gl, true);
    mL1 += timeMetal(m, true);
  }
  gMarch /= 3; mMarch /= 3; gL1 /= 3; mL1 /= 3;
  std::printf("%-22s %10.2f %11.2f %7.2f   parity %lld/%.1f vs %lld/%.1f\n",
    "march31 (residual)", gMarch, mMarch, mMarch / gMarch, gcM, gmM, mcM, mmM);
  std::printf("%-22s %10.2f %11.2f %7.2f\n",
    "l1fetch (issue tax)", gL1, mL1, mL1 / gL1);
  return 0;
}
