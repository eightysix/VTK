// jitter_lag_repro.mm — minimal GL-vs-Metal jitter-cost gap repro (Apple silicon)
//
// Self-contained (no VTK, no DICOM, no external deps). Renders the SAME oblique
// ray march through the SAME 512x512x1794 R8 gradient volume with the SAME
// camera, once through OpenGL (CGL 3.2 core) and once through Metal, MIP
// max-accumulate into an offscreen RT, with per-fragment ray-start jitter off
// (j0) and on (j1), N timed frames each, and prints the four cells plus the
// jitter deltas and the M/GL ratios.
//
// The reproduced phenomenon: the app's Metal and GL backends jitter
// DIFFERENTLY. GL samples a bilinear-filtered noise TEXTURE (smooth correlated
// field — adjacent fragments get nearly the same ray-start phase, the warp
// march stays coherent) and the jitter costs ~nothing (+9-22%). Metal uses a
// per-pixel SHARP noise (independent phase per fragment; lane divergence +
// lattice-shift fetch-set change on gradient data) and the jitter costs
// +55-65%. Same GPU, same rays, same march math — the backend's jitter noise
// path is the entire gap (M/GL j0 ~1.0x, M/GL j1 ~1.34-1.41x, matching the
// app's DICOM single-pass 1.36-1.42x).
//
// Build: clang -fobjc-arc -framework Metal -framework Foundation -framework OpenGL jitter_lag_repro.mm -o jitter_lag_repro
// Run:   ./jitter_lag_repro [rt 2048] [sd 4.0] [frames 30] [mode all|sharp|point|texture]
//        e.g. ./jitter_lag_repro 2048 4 30        -> all six cells (app cell, M2 MBA)
//        ./jitter_lag_repro 2048 4 30 point       -> only the point mode pair
//
// Output:
//   GL    j0 41.3 ms   j1 48.5 ms   +17.4%
//   METAL j0 41.2 ms   j1(sharp) 66.4 ms +61.2% | j1(point) 45.3 ms +10% | j1(texture) 45.1 ms +9.5%
//   M/GL  j0 1.00      sharp 1.37 | point 0.98 | texture 0.97
//   <- GAP REPRODUCED for sharp; CLOSED for point/texture (correlated field)
//
// The jitter semantics are the app's: jitterF in [0, stepSize) shifts the ray
// start to the lattice phase jitterF (tStart' = jitterF + ceil((tStart -
// jitterF)/stepSize)*stepSize). GL noise: the app's exact field — the 64x64
// blue-noise tile sampled at gl_FragCoord.xy/64 with NEAREST + REPEAT
// (vtkVolumeShaderComposer.h g_noiseSampler), block-constant over 32px at
// 2048. Metal noise source is selectable:
//   sharp   per-pixel IGN hash (the app's blue-noise tile class: independent
//           per fragment — harness A/B verified +57-63% either way)
//   point   Fix 1: point-sampled sawtooth analog (128-wide tile, 13-step,
//           no texture)
//   texture Fix 2: the GL sawtooth tile via texture2d, LINEAR + REPEAT, with
//           the GL y-flip — bit-identical jitter field to old GL
//   blue    production proposal: the app's 64x64 blue-noise tile via texture2d
//           LINEAR + REPEAT at position.xy/64 — bilinear-correlated version
//           of the GL field (GL uses NEAREST)
//   parity  same field as GL, no texture: kBlue64 indexed at
//           int2(floor(x/texel), floor((y-rt)/texel)) & 63 (the app's parity
//           branch) — isolates the read path from the field shape
// Verified against minimal_gap: sharp measures GL j1 ~48-52 vs Metal j1
// ~65-69 on this volume; point/texture should collapse M/GL j1 to ~1.0.
// Representative check (2026-08-19): GL j1 must now cost the app's +27%
// (block-constant blue field); if Metal parity (same field) stays ~1.3x vs
// GL, the correlated-field fix is dead on this march — the read-path gap is
// not the field shape.

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

static const int kNoiseTile = 128;  // GL jitter noise field size (px of RT)

// App Metal blue-noise tile (the kBlueNoise64 array from MetalShaders.metal):
// 64x64 luminance, top-down JPEG orientation (tile row y = JPEG row y).
static const uint8_t kBlue64[4096] = {
#include "bluenoise64.inc"
};

static uint8_t* MakeVolume(void)
{
  // z-slice gradient (minimal_gap --data 0 / lag_repro smooth): highly
  // cache-local data — the DICOM-like cell where the jitter gap appears.
  size_t total = (size_t)kVolW * kVolH * kVolD;
  uint8_t* host = (uint8_t*)malloc(total);
  for (size_t i = 0; i < total; i++) host[i] = (uint8_t)((i >> 10) & 0xff);
  return host;
}

static double NowMs(void) { return CACurrentMediaTime() * 1000.0; }

// ---------------------------------------------------------------- GL backend
static double RunGL(int rt, float sdMM, int frames, const uint8_t* vol, int jitter)
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

  // Jitter noise field: the app's exact GL field — the 64x64 blue-noise tile
  // at gl_FragCoord.xy/64, NEAREST + REPEAT (vtkVolumeShaderComposer.h
  // g_noiseSampler). Block-constant over 32px blocks at 2048.
  GLuint noiseTex = 0;
  glGenTextures(1, &noiseTex);
  glBindTexture(GL_TEXTURE_2D, noiseTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, 64, 64, 0, GL_RED, GL_UNSIGNED_BYTE, kBlue64);

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
    "uniform sampler2D uNoise;\n"
    "uniform vec3 uTexelCount;\n"
    "uniform vec3 uEye;\n"
    "uniform vec3 uBoundsSize;\n"
    "uniform mat4 uInvVP;\n"
    "uniform float uSampleDistMM;\n"
    "uniform int uMaxIter;\n"
    "uniform int uJitter;\n"
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
    "  if (uJitter > 0) {\n"
    "    float jitterF = texture(uNoise, gl_FragCoord.xy / " "64.0" ").x * stepSize;\n"
    "    tStart = jitterF + ceil((tStart - jitterF) / stepSize) * stepSize;\n"
    "  }\n"
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
  glUniform1i(glGetUniformLocation(prog, "uVol"), 0);
  glUniform1i(glGetUniformLocation(prog, "uNoise"), 1);
  glUniform3f(glGetUniformLocation(prog, "uTexelCount"), kVolW, kVolH, kVolD);
  glUniform3f(glGetUniformLocation(prog, "uEye"), kEye[0], kEye[1], kEye[2]);
  glUniform3f(glGetUniformLocation(prog, "uBoundsSize"), kBounds[0], kBounds[1], kBounds[2]);
  glUniformMatrix4fv(glGetUniformLocation(prog, "uInvVP"), 1, GL_FALSE, kInvVP);
  glUniform1f(glGetUniformLocation(prog, "uSampleDistMM"), sdMM);
  glUniform1i(glGetUniformLocation(prog, "uMaxIter"), 8192);
  glUniform1i(glGetUniformLocation(prog, "uJitter"), jitter);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_3D, volTex);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, noiseTex);

  glViewport(0, 0, rt, rt);
  GLuint vao = 0;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glEnable(GL_BLEND);
  glBlendEquation(GL_MAX);            // MIP accumulate
  glBlendFunc(GL_ONE, GL_ONE);

  glClearColor(0, 0, 0, 1);
  for (int f = 0; f < 5; f++) {          // warm-up (cold-start skews GL badly)
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
  }
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
  fprintf(stderr, "GL    j%d: %8.3f ms/frame  (footprint %d/%d)\n",
          jitter, dt / frames, nonzero, rt * rt);
  CGLSetCurrentContext(NULL);
  CGLDestroyContext(ctx);
  return dt / frames;
}

// -------------------------------------------------------------- Metal backend
// mode: 0 = sharp per-pixel IGN (app blue-noise class), 1 = point-sampled GL
// sawtooth analog (Fix 1), 2 = the GL tile via texture2d LINEAR+REPEAT (Fix 2),
// 3 = app blue tile via texture2d LINEAR (production proposal), 4 = parity:
// the app's exact GL field (kBlue64 indexed, NEAREST semantics, no texture).
static double RunMetal(int rt, float sdMM, int frames, const uint8_t* vol, int jitter, int mode)
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
  vd.allowGPUOptimizedContents = NO;  // app parity: lag_repro root cause (2026-08-18)
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

  // Jitter noise tile for the texture modes (2 = GL sawtooth, 3 = app blue
  // noise), shared storage, LINEAR + REPEAT on the sampler side.
  id<MTLTexture> noiseTex = nil;
  if (mode == 2 || mode == 3) {
    int tile = mode == 3 ? 64 : kNoiseTile;
    MTLTextureDescriptor* nd =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                         width:tile
                                                        height:tile
                                                     mipmapped:NO];
    nd.usage = MTLTextureUsageShaderRead;
    nd.storageMode = MTLStorageModeShared;
    noiseTex = [device newTextureWithDescriptor:nd];
    uint8_t* noise = (uint8_t*)malloc((size_t)tile * tile);
    if (mode == 3) {
      memcpy(noise, kBlue64, sizeof(kBlue64));
    } else {
      for (int i = 0; i < tile * tile; i++) noise[i] = (uint8_t)(i * 13 % 256);
    }
    [noiseTex replaceRegion:MTLRegionMake2D(0, 0, tile, tile)
                mipmapLevel:0
                  withBytes:noise
                bytesPerRow:tile];
    free(noise);
  }

  MTLTextureDescriptor* rtd = [[MTLTextureDescriptor alloc] init];
  rtd.textureType = MTLTextureType2D;
  rtd.pixelFormat = MTLPixelFormatBGRA8Unorm;
  rtd.width = rt; rtd.height = rt;
  rtd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  rtd.storageMode = MTLStorageModeShared;
  id<MTLTexture> rtTex = [device newTextureWithDescriptor:rtd];

  const char* msl =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "constant unsigned char kBlue64[4096] = { ";
  {
    char buf[64];
    char* p = (char*)malloc(4096 * 8 + 64);
    size_t n = 0;
    for (int i = 0; i < 4096; i++) { int l = snprintf(buf, sizeof(buf), "%u,", kBlue64[i]); memcpy(p + n, buf, l); n += l; }
    memcpy(p + n, " };\n", 5); n += 5;
    size_t mslLen = strlen(msl) + n + 1;
    char* full = (char*)malloc(mslLen);
    strcpy(full, msl);
    memcpy(full + strlen(msl), p, n);
    full[strlen(msl) + n] = 0;
    free(p);
    msl = full;
  }
  const char* msl2 =
    "struct VOut { float4 position [[position]]; float2 uv; };\n"
    "vertex VOut vertex_main(uint vid [[vertex_id]]) {\n"
    "  float2 p = float2(float(vid & 1u) * 4.0f - 1.0f, float((vid >> 1u) & 1u) * 2.0f - 1.0f);\n"
    "  if (vid == 2u) p = float2(-1.0f, 3.0f);\n"
    "  VOut o; o.position = float4(p, 0.0f, 1.0f); o.uv = p * 0.5f + 0.5f; return o;\n"
    "}\n"
    "struct Uniforms { float4 eye; float4 boundsSize; float4x4 invVP; float sampleDistMM; int jitter; int jitterMode; int maxIter; float rtSize; };\n"
    "fragment float4 fragment_main(VOut in [[stage_in]],\n"
    "                              texture3d<float> volTex [[texture(0)]],\n"
    "                              texture2d<float> noiseTex [[texture(1)]],\n"
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
    "  if (u.jitter > 0) {\n"
    "    float2 fid = floor(in.position.xy);\n"
    "    float jitterF;\n"
    "    if (u.jitterMode == 2 || u.jitterMode == 3) {\n"
    "      constexpr sampler noiseSampler(filter::linear, address::repeat);\n"
    "      float2 nuv = float2(fid.x, u.rtSize - fid.y) / float(u.jitterMode == 3 ? 64 : 128);\n"
    "      jitterF = noiseTex.sample(noiseSampler, nuv).r * stepSize;\n"
    "    } else if (u.jitterMode == 4) {\n"
    "      int2 t = int2(int(fid.x) & 63, int(u.rtSize - fid.y) & 63);\n"
    "      jitterF = (kBlue64[t.y * 64 + t.x] / 255.0f) * stepSize;\n"
    "    } else if (u.jitterMode == 1) {\n"
    "      float n = fmod(fid.x * 13.0f + fid.y * 128.0f, 256.0f) * (1.0f / 255.0f);\n"
    "      jitterF = n * stepSize;\n"
    "    } else {\n"
    "      jitterF = fract(52.9829189f * fract(dot(fid, float2(0.06711056f, 0.00583715f)))) * stepSize;\n"
    "    }\n"
    "    tStart = jitterF + ceil((tStart - jitterF) / stepSize) * stepSize;\n"
    "  }\n"
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
  id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithFormat:@"%s%s", msl, msl2] options:nil error:&err];
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
    float sampleDistMM;
    int jitter, jitterMode, maxIter;
    float rtSize;
  } u;
  u.eye = (simd_float4){ kEye[0], kEye[1], kEye[2], 0.0f };
  u.boundsSize = (simd_float4){ kBounds[0], kBounds[1], kBounds[2], 0.0f };
  memcpy(&u.invVP, kInvVP, 16 * sizeof(float));
  u.sampleDistMM = sdMM; u.jitter = jitter; u.jitterMode = mode; u.maxIter = 8192;
  u.rtSize = (float)rt;
  id<MTLBuffer> ubuf = [device newBufferWithBytes:&u length:sizeof(u) options:MTLResourceStorageModeShared];

  for (int f = 0; f < 5; f++) {          // warm-up
    MTLRenderPassDescriptor* rpd = [[MTLRenderPassDescriptor alloc] init];
    rpd.colorAttachments[0].texture = rtTex;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    id<MTLCommandBuffer> wcb = [queue commandBuffer];
    id<MTLRenderCommandEncoder> wenc = [wcb renderCommandEncoderWithDescriptor:rpd];
    [wenc setRenderPipelineState:pso];
    [wenc setFragmentTexture:volTex atIndex:0];
    if (noiseTex) [wenc setFragmentTexture:noiseTex atIndex:1];
    [wenc setFragmentBuffer:ubuf offset:0 atIndex:0];
    [wenc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [wenc endEncoding];
    [wcb commit];
    [wcb waitUntilCompleted];
  }

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
    if (noiseTex) [enc setFragmentTexture:noiseTex atIndex:1];
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
  fprintf(stderr, "METAL j%d: %8.3f ms/frame  (footprint probe %d/%d)\n",
          jitter, dt / frames, nonzero, (rt / 37) * (rt / 37));
  return dt / frames;
}

int main(int argc, const char** argv)
{
  @autoreleasepool {
    int rt = argc > 1 ? atoi(argv[1]) : 2048;
    float sd = argc > 2 ? (float)atof(argv[2]) : 4.0f;
    int frames = argc > 3 ? atoi(argv[3]) : 30;
    const char* modeArg = argc > 4 ? argv[4] : "all";
    int all = !strcmp(modeArg, "all");
    int wantSharp = all || !strcmp(modeArg, "sharp");
    int wantPoint = all || !strcmp(modeArg, "point");
    int wantTexture = all || !strcmp(modeArg, "texture");
    int wantBlue = all || !strcmp(modeArg, "blue");
    int wantParity = all || !strcmp(modeArg, "parity");
    fprintf(stderr, "volume %dx%dx%d R8 (%d MB), rt %dx%d, sd %.1f mm, frames %d, gradient data, metal jitter mode %s\n",
            kVolW, kVolH, kVolD, (int)((size_t)kVolW * kVolH * kVolD >> 20), rt, rt, sd, frames, modeArg);
    uint8_t* vol = MakeVolume();

    // Interleaved A/B: j0 pair then j1 pairs (battery drift is between pairs,
    // the deltas are measured back-to-back in the same thermal window).
    double g0 = RunGL(rt, sd, frames, vol, 0);
    double g1 = RunGL(rt, sd, frames, vol, 1);
    double m0 = RunMetal(rt, sd, frames, vol, 0, 0);
    double m1s = wantSharp ? RunMetal(rt, sd, frames, vol, 1, 0) : 0.0;
    double m1p = wantPoint ? RunMetal(rt, sd, frames, vol, 1, 1) : 0.0;
    double m1t = wantTexture ? RunMetal(rt, sd, frames, vol, 1, 2) : 0.0;
    double m1b = wantBlue ? RunMetal(rt, sd, frames, vol, 1, 3) : 0.0;
    double m1r = wantParity ? RunMetal(rt, sd, frames, vol, 1, 4) : 0.0;

    fprintf(stderr, "\nGL    j0 %8.3f   j1 %8.3f   jitter %+6.1f%%\n", g0, g1, (g1 / g0 - 1.0) * 100.0);
    if (wantSharp)
      fprintf(stderr, "METAL j0 %8.3f   j1(sharp) %8.3f   jitter %+6.1f%%\n", m0, m1s, (m1s / m0 - 1.0) * 100.0);
    if (wantPoint)
      fprintf(stderr, "METAL j0 %8.3f   j1(point) %8.3f   jitter %+6.1f%%\n", m0, m1p, (m1p / m0 - 1.0) * 100.0);
    if (wantTexture)
      fprintf(stderr, "METAL j0 %8.3f   j1(texture) %8.3f   jitter %+6.1f%%\n", m0, m1t, (m1t / m0 - 1.0) * 100.0);
    if (wantBlue)
      fprintf(stderr, "METAL j0 %8.3f   j1(blue) %8.3f   jitter %+6.1f%%\n", m0, m1b, (m1b / m0 - 1.0) * 100.0);
    if (wantParity)
      fprintf(stderr, "METAL j0 %8.3f   j1(parity) %8.3f   jitter %+6.1f%%\n", m0, m1r, (m1r / m0 - 1.0) * 100.0);
    fprintf(stderr, "M/GL  j0 %8.2f", m0 / g0);
    if (wantSharp) fprintf(stderr, "   sharp %5.2f", m1s / g1);
    if (wantPoint) fprintf(stderr, "   point %5.2f", m1p / g1);
    if (wantTexture) fprintf(stderr, "   texture %5.2f", m1t / g1);
    if (wantBlue) fprintf(stderr, "   blue %5.2f", m1b / g1);
    if (wantParity) fprintf(stderr, "   parity %5.2f", m1r / g1);
    fprintf(stderr, "\n");
    if (all) {
      // Representative check: GL j1 carries the app's block-constant blue
      // field (~+27%); parity is the same field in Metal. If parity stays
      // ~1.3x (field-independent read-path gap), the correlated-field fixes
      // are dead on this march and the gap must be attacked elsewhere.
      if (m1s / g1 > 1.25f && (m1r / g1 > 1.2f || m1b / g1 > 1.2f))
        fprintf(stderr, "RESULT: sharp reproduces the gap (+%.0f%%); same-field parity still +%.0f%% -> correlated-field fix DEAD on this march\n",
                (m1s / g1 - 1.0) * 100.0, (m1r / g1 - 1.0) * 100.0);
      else if (m1s / g1 > 1.25f && m1r / g1 < 1.1f)
        fprintf(stderr, "RESULT: sharp reproduces the gap (+%.0f%%), parity (same field) CLOSES it -> field shape is the gap\n",
                (m1s / g1 - 1.0) * 100.0);
      else
        fprintf(stderr, "RESULT: unexpected shape (check thermal state; rerun interleaved)\n");
    }
    free(vol);
  }
  return 0;
}
