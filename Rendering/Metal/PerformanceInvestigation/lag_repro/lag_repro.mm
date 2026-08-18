// lag_repro.mm — minimal GL-vs-Metal 3D-texture sampling lag repro (Apple silicon)
//
// Self-contained (no VTK, no DICOM, no external deps). Renders the SAME oblique
// ray march through the SAME 512x512x1794 R8 volume with the SAME camera, once
// through OpenGL (CGL 3.2 core) and once through Metal, MIP max-accumulate into
// an offscreen RT, N timed frames, and prints each backend's avg frame time and
// the M/GL ratio.
//
// The reproduced phenomenon: on smooth (gradient) data Metal >= GL, but on
// per-texel-varying (noise) data Metal loses ~1.7x at a 2048x2048 RT / 4 mm
// sample distance — a cache/DRAM working-set behavior of the texture path that
// survives identical rays, identical shader math, and the same GPU (macOS GL is
// itself a Metal-based layer; both go through the same silicon).
//
// Build: clang -fobjc-arc -framework Metal -framework Foundation -framework OpenGL lag_repro.mm -o lag_repro
// Run:   ./lag_repro [gl|metal|both] [rt 2048] [sd 4.0] [frames 30] [smooth 0]
//        e.g. ./lag_repro both 2048 4 30       -> noise (lag: M/GL ~1.67 on M2 MBA)
//             ./lag_repro both 2048 4 30 1     -> smooth ramp (no lag: M/GL ~0.91)
//
// Output: "GL:   xx.xxx ms/frame", "METAL: xx.xxx ms/frame", "M/GL: x.xx".
// Verified against the app/harness: the same cell measures GL ~47-48 vs Metal
// ~77-80 ms in minimal_gap's gl_gap/metal_gap (dataMode 1), and the app's DICOM
// single-pass oblique is GL 40.3 vs Metal 61.8 at the same RT/SD.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <OpenGL/gl3.h>
#import <OpenGL/OpenGL.h>
#import <QuartzCore/QuartzCore.h>
#import <simd/simd.h>
#include <stdlib.h>
#include <string.h>

static const int kVolW = 512, kVolH = 512, kVolD = 1794;  // DICOM-like NPOT volume
// App DICOMVolume camera dump (normalized volume space / physical mm):
static const float kEye[3] = { -1.49527037f, -0.95243806f, 2.55352926f };
static const float kBounds[3] = { 426.166f, 426.166f, 717.2f };
static const float kInvVP[16] = {
  0.23205078f, -6.974323e-09f, 0.13397458f, 1.2084004e-11f,
  -0.045822006f, 0.25178984f, 0.079366043f, -3.2157374e-12f,
  0.35811684f, 0.22810864f, -1.0292176f, -0.00056198676f,
  -0.12124427f, -0.03448515f, 0.8849802f, 0.00092758867f };

static uint8_t* MakeVolume(int smooth)
{
  size_t total = (size_t)kVolW * kVolH * kVolD;
  uint8_t* host = (uint8_t*)malloc(total);
  if (smooth) {
    for (size_t i = 0; i < total; i++) host[i] = (uint8_t)((i >> 10) & 0xff);  // gradient
  } else {
    uint32_t x = 0x12345678u;  // xorshift32 noise — per-texel variation
    for (size_t i = 0; i < total; i++) {
      x ^= x << 13; x ^= x >> 17; x ^= x << 5;
      host[i] = (uint8_t)(x >> 24);
    }
  }
  return host;
}

static double NowMs(void) { return CACurrentMediaTime() * 1000.0; }

// ---------------------------------------------------------------- GL backend
static double RunGL(int rt, float sdMM, int frames, const uint8_t* vol, int smooth)
{
  CGLPixelFormatAttribute attrs[] = {
    kCGLPFAAccelerated, kCGLPFAOpenGLProfile, (CGLPixelFormatAttribute)kCGLOGLPVersion_3_2_Core, (CGLPixelFormatAttribute)0 };
  CGLPixelFormatObj pf = NULL; GLint npf = 0;
  if (CGLChoosePixelFormat(attrs, &pf, &npf) != kCGLNoError || !pf) { fprintf(stderr, "no GL 3.2 core pixel format\n"); exit(1); }
  CGLContextObj ctx = NULL;
  if (CGLCreateContext(pf, NULL, &ctx) != kCGLNoError || !ctx) { fprintf(stderr, "no GL context\n"); exit(1); }
  CGLSetCurrentContext(ctx);
  CGLDestroyPixelFormat(pf);

  GLuint volTex = 0;
  glGenTextures(1, &volTex);
  glBindTexture(GL_TEXTURE_3D, volTex);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, kVolW, kVolH, kVolD, 0, GL_RED, GL_UNSIGNED_BYTE, vol);

  GLuint rtTex = 0, fbo = 0;
  glGenTextures(1, &rtTex);
  glBindTexture(GL_TEXTURE_2D, rtTex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rt, rt, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rtTex, 0);
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { fprintf(stderr, "FBO incomplete\n"); exit(1); }

  const char* vs = "#version 150\n"
    "out vec2 vUV;\n"
    "void main() {\n"
    "  vec2 p = vec2(float(gl_VertexID & 1) * 4.0 - 1.0, float((gl_VertexID >> 1) & 1) * 2.0 - 1.0);\n"
    "  if (gl_VertexID == 2) p = vec2(-1.0, 3.0);\n"
    "  gl_Position = vec4(p, 0.0, 1.0); vUV = p * 0.5 + 0.5;\n"
    "}\n";
  const char* fs = "#version 150\n"
    "in vec2 vUV;\n"
    "out vec4 fragColor;\n"
    "uniform sampler3D uVol;\n"
    "uniform vec3 uTexelCount;\n"
    "uniform vec3 uEye;\n"
    "uniform vec3 uBoundsSize;\n"
    "uniform mat4 uInvVP;\n"
    "uniform float uSampleDistMM;\n"
    "uniform int uMaxIter;\n"
    "void main() {\n"
    "  vec2 ndc = vUV * 2.0 - 1.0;\n"
    "  vec4 w4 = uInvVP * vec4(ndc, 0.0, 1.0);\n"
    "  vec3 ptPhys = w4.xyz / w4.w;\n"
    "  vec3 rayDir = normalize(ptPhys / uBoundsSize - uEye);\n"
    "  vec3 inv = 1.0 / rayDir;\n"
    "  vec3 t0 = (vec3(0.0) - uEye) * inv;\n"
    "  vec3 t1 = (vec3(1.0) - uEye) * inv;\n"
    "  vec3 tmin3 = min(t0, t1);\n"
    "  vec3 tmax3 = max(t0, t1);\n"
    "  float tEnter = max(max(tmin3.x, tmin3.y), tmin3.z);\n"
    "  float tExit = min(min(tmax3.x, tmax3.y), tmax3.z);\n"
    "  if (tExit <= 0.0 || tEnter >= tExit) { fragColor = vec4(0.0,0.0,0.0,1.0); return; }\n"
    "  float tStart = max(tEnter, 0.0);\n"
    "  float physPerNorm = length(rayDir * uBoundsSize);\n"
    "  float stepSize = uSampleDistMM / max(physPerNorm, 1e-6);\n"
    "  int maxSteps = max(0, int(ceil((tExit - tStart) / stepSize)));\n"
    "  vec3 ctpScale = max(uTexelCount - 1.0, 1e-4) / uTexelCount;\n"
    "  vec3 ctpOffset = 0.5 / uTexelCount;\n"
    "  vec3 evalBase = ctpOffset + (uEye + rayDir * tStart) * ctpScale;\n"
    "  vec3 evalStep = rayDir * ctpScale * stepSize;\n"
    "  float acc = 0.0;\n"
    "  float n = 0.0;\n"
    "  for (int i = 0; i < min(uMaxIter, maxSteps); i++) {\n"
    "    float currentT = tStart + float(i) * stepSize;\n"
    "    if (currentT >= tExit - 1e-6) break;\n"
    "    vec3 evalPoint = evalBase + float(i) * evalStep;\n"
    "    acc = max(acc, texture(uVol, evalPoint).r);\n"
    "    n += 1.0;\n"
    "  }\n"
    "  int nc = int(n);\n"
    "  fragColor = vec4(float(nc & 255) / 255.0, float((nc >> 8) & 255) / 255.0, acc, 1.0);\n"
    "}\n";

  GLuint prog = glCreateProgram();
  GLuint vsh = glCreateShader(GL_VERTEX_SHADER), fsh = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(vsh, 1, &vs, NULL); glCompileShader(vsh);
  glShaderSource(fsh, 1, &fs, NULL); glCompileShader(fsh);
  GLint ok = 0; glGetShaderiv(fsh, GL_COMPILE_STATUS, &ok);
  if (!ok) { char log[4096]; glGetShaderInfoLog(fsh, sizeof(log), NULL, log); fprintf(stderr, "fragment link failed:\n%s\n", log); exit(1); }
  glAttachShader(prog, vsh); glAttachShader(prog, fsh); glLinkProgram(prog);
  glUseProgram(prog);
  GLint uVol = glGetUniformLocation(prog, "uVol");
  glUniform1i(uVol, 0);
  GLint uTexelCount = glGetUniformLocation(prog, "uTexelCount");
  GLint uEye = glGetUniformLocation(prog, "uEye");
  GLint uBoundsSize = glGetUniformLocation(prog, "uBoundsSize");
  GLint uInvVP = glGetUniformLocation(prog, "uInvVP");
  GLint uSampleDistMM = glGetUniformLocation(prog, "uSampleDistMM");
  GLint uMaxIter = glGetUniformLocation(prog, "uMaxIter");
  glUniform3f(uTexelCount, kVolW, kVolH, kVolD);
  glUniform3f(uEye, kEye[0], kEye[1], kEye[2]);
  glUniform3f(uBoundsSize, kBounds[0], kBounds[1], kBounds[2]);
  glUniformMatrix4fv(uInvVP, 1, GL_FALSE, kInvVP);
  glUniform1f(uSampleDistMM, sdMM);
  glUniform1i(uMaxIter, 8192);

  glViewport(0, 0, rt, rt);
  GLuint vao = 0;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glEnable(GL_BLEND);
  glBlendEquation(GL_MAX);            // MIP accumulate
  glBlendFunc(GL_ONE, GL_ONE);

  glClearColor(0, 0, 0, 1);
  double t0 = NowMs();
  for (int f = 0; f < frames; f++) {
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
  }
  double dt = NowMs() - t0;

  int nonzero = 0;
  {
    uint8_t* pix = (uint8_t*)malloc((size_t)rt * rt * 4);
    glReadPixels(0, 0, rt, rt, GL_RGBA, GL_UNSIGNED_BYTE, pix);
    for (size_t i = 0; i < (size_t)rt * rt * 4; i += 4)
      if (pix[i] | pix[i + 1] | pix[i + 2]) nonzero++;
    free(pix);
  }
  fprintf(stderr, "GL:     %8.3f ms/frame  (footprint %d/%d, data %s)\n",
          dt / frames, nonzero, rt * rt, smooth ? "smooth" : "noise");
  CGLSetCurrentContext(NULL);
  CGLDestroyContext(ctx);
  return dt / frames;
}

// -------------------------------------------------------------- Metal backend
static double RunMetal(int rt, float sdMM, int frames, const uint8_t* vol, int smooth)
{
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) { fprintf(stderr, "no Metal device\n"); exit(1); }

  MTLTextureDescriptor* vd = [[MTLTextureDescriptor alloc] init];
  vd.textureType = MTLTextureType3D;
  vd.pixelFormat = MTLPixelFormatR8Unorm;
  vd.width = kVolW; vd.height = kVolH; vd.depth = kVolD;
  vd.mipmapLevelCount = 1;
  vd.usage = MTLTextureUsageShaderRead;
  vd.storageMode = MTLStorageModePrivate;
  id<MTLTexture> volTex = [device newTextureWithDescriptor:vd];
  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLCommandBuffer> cb0 = [queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cb0 blitCommandEncoder];
  size_t total = (size_t)kVolW * kVolH * kVolD;
  id<MTLBuffer> stage = [device newBufferWithBytes:vol length:total options:MTLResourceStorageModeShared];
  [blit copyFromBuffer:stage sourceOffset:0 sourceBytesPerRow:kVolW sourceBytesPerImage:kVolW * kVolH
          sourceSize:MTLSizeMake(kVolW, kVolH, kVolD) toTexture:volTex
          destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
  [blit endEncoding];
  [cb0 commit];
  [cb0 waitUntilCompleted];

  MTLTextureDescriptor* rtd = [[MTLTextureDescriptor alloc] init];
  rtd.textureType = MTLTextureType2D;
  rtd.pixelFormat = MTLPixelFormatBGRA8Unorm;
  rtd.width = rt; rtd.height = rt;
  rtd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  rtd.storageMode = MTLStorageModeShared;
  id<MTLTexture> rtTex = [device newTextureWithDescriptor:rtd];

  const char* msl =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "struct VOut { float4 position [[position]]; float2 uv; };\n"
    "vertex VOut vertex_main(uint vid [[vertex_id]]) {\n"
    "  float2 p = float2(float(vid & 1u) * 4.0f - 1.0f, float((vid >> 1u) & 1u) * 2.0f - 1.0f);\n"
    "  if (vid == 2u) p = float2(-1.0f, 3.0f);\n"
    "  VOut o; o.position = float4(p, 0.0f, 1.0f); o.uv = p * 0.5f + 0.5f; return o;\n"
    "}\n"
    "struct Uniforms { float4 eye; float4 boundsSize; float4x4 invVP; float sampleDistMM; float pad0; float pad1; int maxIter; };\n"
    "fragment float4 fragment_main(VOut in [[stage_in]],\n"
    "                              texture3d<float> volTex [[texture(0)]],\n"
    "                              constant Uniforms& u [[buffer(0)]]) {\n"
    "  float2 ndc = in.uv * 2.0f - 1.0f;\n"
    "  float4 w4 = u.invVP * float4(ndc, 0.0f, 1.0f);\n"
    "  float3 ptPhys = w4.xyz / w4.w;\n"
    "  float3 eye = u.eye.xyz;\n"
    "  float3 rayDir = normalize(ptPhys / u.boundsSize.xyz - eye);\n"
    "  float3 inv = 1.0f / rayDir;\n"
    "  float3 t0 = (float3(0.0f) - eye) * inv;\n"
    "  float3 t1 = (float3(1.0f) - eye) * inv;\n"
    "  float3 tmin3 = min(t0, t1);\n"
    "  float3 tmax3 = max(t0, t1);\n"
    "  float tEnter = max(max(tmin3.x, tmin3.y), tmin3.z);\n"
    "  float tExit = min(min(tmax3.x, tmax3.y), tmax3.z);\n"
    "  if (tExit <= 0.0f || tEnter >= tExit) return float4(0.0f, 0.0f, 0.0f, 1.0f);\n"
    "  float tStart = max(tEnter, 0.0f);\n"
    "  float physPerNorm = length(rayDir * u.boundsSize.xyz);\n"
    "  float stepSize = u.sampleDistMM / max(physPerNorm, 1e-6f);\n"
    "  int maxSteps = max(0, int(ceil((tExit - tStart) / stepSize)));\n"
    "  float3 texelCount = float3(volTex.get_width(), volTex.get_height(), volTex.get_depth());\n"
    "  float3 ctpScale = max(texelCount - 1.0f, 1e-4f) / texelCount;\n"
    "  float3 ctpOffset = 0.5f / texelCount;\n"
    "  float3 evalBase = ctpOffset + (eye + rayDir * tStart) * ctpScale;\n"
    "  float3 evalStep = rayDir * ctpScale * stepSize;\n"
    "  constexpr sampler volSampler(filter::linear, address::clamp_to_edge);\n"
    "  float acc = 0.0f;\n"
    "  float n = 0.0f;\n"
    "  for (int i = 0; i < min(u.maxIter, maxSteps); i++) {\n"
    "    float currentT = tStart + float(i) * stepSize;\n"
    "    if (currentT >= tExit - 1e-6f) break;\n"
    "    float3 evalPoint = evalBase + float(i) * evalStep;\n"
    "    acc = max(acc, volTex.sample(volSampler, evalPoint).r);\n"
    "    n += 1.0f;\n"
    "  }\n"
    "  int nc = int(n);\n"
    "  return float4(float(nc & 255) / 255.0f, float((nc >> 8) & 255) / 255.0f, acc, 1.0f);\n"
    "}\n";
  NSError* err = nil;
  id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithUTF8String:msl] options:nil error:&err];
  if (!lib) { fprintf(stderr, "MSL compile failed: %s\n", err.description.UTF8String); exit(1); }
  id<MTLFunction> vf = [lib newFunctionWithName:@"vertex_main"];
  id<MTLFunction> ff = [lib newFunctionWithName:@"fragment_main"];
  MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
  pd.vertexFunction = vf; pd.fragmentFunction = ff;
  pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
  pd.colorAttachments[0].blendingEnabled = YES;
  pd.colorAttachments[0].rgbBlendOperation = MTLBlendOperationMax;  // MIP accumulate
  pd.colorAttachments[0].alphaBlendOperation = MTLBlendOperationMax;
  pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
  pd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
  pd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
  pd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
  id<MTLRenderPipelineState> pso = [device newRenderPipelineStateWithDescriptor:pd error:&err];
  if (!pso) { fprintf(stderr, "PSO failed: %s\n", err.description.UTF8String); exit(1); }

  struct Uniforms {
    simd_float4 eye, boundsSize;
    simd_float4x4 invVP;
    float sampleDistMM, pad0, pad1;
    int maxIter;
  } u;
  u.eye = (simd_float4){ kEye[0], kEye[1], kEye[2], 0.0f };
  u.boundsSize = (simd_float4){ kBounds[0], kBounds[1], kBounds[2], 0.0f };
  memcpy(&u.invVP, kInvVP, 16 * sizeof(float));
  u.sampleDistMM = sdMM; u.pad0 = 0; u.pad1 = 0; u.maxIter = 8192;
  id<MTLBuffer> ubuf = [device newBufferWithBytes:&u length:sizeof(u) options:MTLResourceStorageModeShared];

  double t0 = NowMs();
  id<MTLCommandBuffer> lastCB = nil;
  for (int f = 0; f < frames; f++) {
    MTLRenderPassDescriptor* rpd = [[MTLRenderPassDescriptor alloc] init];
    rpd.colorAttachments[0].texture = rtTex;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    lastCB = [queue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [lastCB renderCommandEncoderWithDescriptor:rpd];
    [enc setRenderPipelineState:pso];
    [enc setFragmentTexture:volTex atIndex:0];
    [enc setFragmentBuffer:ubuf offset:0 atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    [lastCB commit];
  }
  [lastCB waitUntilCompleted];
  double dt = NowMs() - t0;

  int nonzero = 0;
  {
    uint8_t pix[4];
    for (int y = 0; y < rt; y += 37) for (int x = 0; x < rt; x += 37) {
      [rtTex getBytes:pix bytesPerRow:4 bytesPerImage:4 fromRegion:MTLRegionMake2D(x, y, 1, 1) mipmapLevel:0 slice:0];
      if (pix[0] | pix[1] | pix[2]) nonzero++;
    }
  }
  fprintf(stderr, "METAL:  %8.3f ms/frame  (footprint probe %d/%d, data %s)\n",
          dt / frames, nonzero, (rt / 37) * (rt / 37), smooth ? "smooth" : "noise");
  return dt / frames;
}

int main(int argc, const char** argv)
{
  @autoreleasepool {
    const char* which = argc > 1 ? argv[1] : "both";
    int rt = argc > 2 ? atoi(argv[2]) : 2048;
    float sd = argc > 3 ? (float)atof(argv[3]) : 4.0f;
    int frames = argc > 4 ? atoi(argv[4]) : 30;
    int smooth = argc > 5 ? atoi(argv[5]) : 0;
    fprintf(stderr, "volume %dx%dx%d R8 (%d MB), rt %dx%d, sd %.1f mm, frames %d, data %s\n",
            kVolW, kVolH, kVolD, (int)((size_t)kVolW * kVolH * kVolD >> 20), rt, rt, sd, frames,
            smooth ? "smooth" : "noise");
    uint8_t* vol = MakeVolume(smooth);
    double g = 0, m = 0;
    if (!strcmp(which, "gl") || !strcmp(which, "both")) g = RunGL(rt, sd, frames, vol, smooth);
    if (!strcmp(which, "metal") || !strcmp(which, "both")) m = RunMetal(rt, sd, frames, vol, smooth);
    if (!strcmp(which, "both")) {
      fprintf(stderr, "M/GL:   %.2f  <- %s\n", m / g,
              (m / g > 1.15f) ? "LAG REPRODUCED (Metal slower on per-texel-varying data)"
                              : "no lag (Metal at or above GL parity)");
    }
    free(vol);
  }
  return 0;
}