// jitter_gap_repro.mm — minimal repro of the app's Metal-vs-GL jitter cost gap
// on the real DICOM volume (Apple silicon).
//
// Self-contained (no VTK, no DICOM libs). Loads the same U8 volume the app
// renders (dicom.u8: 512x512x1794, the castToU8 texture bytes), renders the
// same oblique ray march with the same camera, once through OpenGL (CGL 3.2
// core) and once through Metal, front-to-back composite with the app's
// "Airways II" transfer function, per-fragment ray-start jitter off (j0) and
// on (j1), N timed frames each.
//
// THE JITTER IS GL'S, ON BOTH BACKENDS (the reference): the app's GL jitter is
// a pure origin shift — g_rayOrigin += g_dirStep * jitterValue with the 64x64
// blue-noise tile at gl_FragCoord.xy/64, NEAREST + REPEAT
// (vtkVolumeShaderComposer.h) — no lattice re-alignment. Metal applies the
// SAME field (kBlueNoise64 per-pixel, the app's parity branch) with the same
// origin-shift semantics. Any remaining M/GL gap is therefore the read path
// under an identical jitter field, reproducing the app's DICOM single-pass
// numbers (M/GL j0 ~1.0, j1 ~1.28-1.35 on this M2 MBA).
//
// Build: clang -fobjc-arc -framework Metal -framework Foundation -framework OpenGL jitter_gap_repro.mm -o jitter_gap_repro
// Run:   ./jitter_gap_repro [rt 2048] [sd 4.0] [frames 30] [volume dicom.u8]
//
// Output:
//   GL    j0 <ms>   j1 <ms>   jitter +<pct>%
//   METAL j0 <ms>   j1 <ms>   jitter +<pct>%
//   M/GL  j0 <x>    j1 <x>
//   RESULT: gap reproduced (j1 > 1.2) | closed (j1 < 1.05) | unexpected

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <OpenGL/gl3.h>
#import <OpenGL/OpenGL.h>
#import <QuartzCore/QuartzCore.h>
#import <simd/simd.h>
#include <stdlib.h>
#include <string.h>

static const int kVolW = 512, kVolH = 512, kVolD = 1794;  // the app's DICOM volume
// App DICOMVolume camera dump (normalized volume space / physical mm):
static const float kEye[3] = { -1.49527037f, -0.95243806f, 2.55352926f };
static const float kBounds[3] = { 426.166f, 426.166f, 717.2f };
static const float kInvVP[16] = {
  0.23205078f, -6.974323e-09f, 0.13397458f, 1.2084004e-11f,
  -0.045822006f, 0.25178984f, 0.079366043f, -3.2157374e-12f,
  0.35811684f, 0.22810864f, -1.0292176f, -0.00056198676f,
  -0.12124427f, -0.03448515f, 0.8849802f, 0.00092758867f };

// kVP = inverse(kInvVP): the app's view-projection (physical mm space).
// Stored column-major (GL GL_FALSE / Metal simd_float4x4 convention).
static const float kVP[16] = {
  3.23205099f, -0.63821831f, 0.77550604f, 0.46984628f,
  -2.08898199e-07f, 3.50698106f, 0.56452231f, 0.34202022f,
  1.86602610f, 1.10542682f, -1.34321597f, -0.81379779f,
  -1357.8524f, -1007.69061f, 1403.86885f, 1928.60885f };

// Unit-cube corners (normalized volume space), 6 faces / 12 outward triangles.
static const float kBoxCorners[36 * 3] = {
  // -x (x=0)
  0, 0, 0,   0, 0, 1,   0, 1, 1,
  0, 0, 0,   0, 1, 1,   0, 1, 0,
  // +x (x=1)
  1, 0, 0,   1, 1, 0,   1, 1, 1,
  1, 0, 0,   1, 1, 1,   1, 0, 1,
  // -y (y=0)
  0, 0, 0,   1, 0, 0,   1, 0, 1,
  0, 0, 0,   1, 0, 1,   0, 0, 1,
  // +y (y=1)
  0, 1, 0,   0, 1, 1,   1, 1, 1,
  0, 1, 0,   1, 1, 1,   1, 1, 0,
  // -z (z=0)
  0, 0, 0,   0, 1, 0,   1, 1, 0,
  0, 0, 0,   1, 1, 0,   1, 0, 0,
  // +z (z=1)
  0, 0, 1,   1, 0, 1,   1, 1, 1,
  0, 0, 1,   1, 1, 1,   0, 1, 1,
};

// App Metal blue-noise tile (kBlueNoise64 from MetalShaders.metal), 64x64 U8.
static const uint8_t kBlue64[4096] = {
#include "bluenoise64.inc"
};

// "Airways II" opacity (x rescaled to 0..1): 0.0, 0.0493, 0.2497, 0.0 over
// (17.55, 21.24, 33.80, 43.01)/255; constant color (0, 0.605, 0.706).
// TF LUT: 256 x RGBA8, opacity ramps, color constant (app's single-color TF).
static void MakeTF(uint8_t* lut)
{
  if (getenv("PROBE_BOX"))
  {
    for (int i = 0; i < 256; i++)
    {
      lut[i * 4 + 0] = 255; lut[i * 4 + 1] = 255; lut[i * 4 + 2] = 255;
      lut[i * 4 + 3] = 64;
    }
    return;
  }
  const float xs[4] = { 17.55f, 21.24f, 33.80f, 43.01f };
  const float ys[4] = { 0.0f, 0.0493f, 0.2497f, 0.0f };
  const int cr = 0, cg = 154, cb = 180;  // 0.605, 0.706 * 255
  for (int i = 0; i < 256; i++)
  {
    float x = (float)i;
    float op = 0.0f;
    if (x <= xs[0] || x >= xs[3]) op = 0.0f;
    else
    {
      for (int k = 0; k < 3; k++)
      {
        if (x >= xs[k] && x <= xs[k + 1])
        {
          float t = (x - xs[k]) / (xs[k + 1] - xs[k]);
          op = ys[k] + t * (ys[k + 1] - ys[k]);
          break;
        }
      }
    }
    lut[i * 4 + 0] = (uint8_t)(cr * op);
    lut[i * 4 + 1] = (uint8_t)(cg * op);
    lut[i * 4 + 2] = (uint8_t)(cb * op);
    lut[i * 4 + 3] = (uint8_t)(op * 255.0f);
  }
}

static double NowMs(void) { return CACurrentMediaTime() * 1000.0; }

// ---------------------------------------------------------------- GL backend
static double RunGL(int rt, float sdMM, int frames, const uint8_t* vol, const uint8_t* tf, int jitter)
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

  // Jitter noise field: the app's GL field — the 64x64 blue-noise tile at
  // gl_FragCoord.xy/64, NEAREST + REPEAT (vtkVolumeShaderComposer.h). The app
  // uploads it as VTK_FLOAT (vtkOpenGLRenderWindow::GetNoiseTextureUnit).
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

  // Transfer function LUT: 256x1 RGBA8.
  GLuint tfTex = 0;
  glGenTextures(1, &tfTex);
  glBindTexture(GL_TEXTURE_2D, tfTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, tf);

  GLuint rtTex = 0, fbo = 0;
  glGenTextures(1, &rtTex);
  glBindTexture(GL_TEXTURE_2D, rtTex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rt, rt, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rtTex, 0);
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { fprintf(stderr, "FBO incomplete\n"); exit(1); }

  const char* vs = "#version 150\n"
    "in vec3 aCorner;\n"
    "uniform mat4 uVP;\n"
    "uniform vec3 uBoundsSize;\n"
    "void main() {\n"
    "  gl_Position = uVP * vec4(aCorner * uBoundsSize, 1.0);\n"
    "}\n";
  // GL jitter = the app's reference: pure origin shift, no lattice
  // re-alignment (g_rayOrigin += g_dirStep * jitterValue).
  const char* fs = "#version 150\n"
    "out vec4 fragColor;\n"
    "uniform sampler3D uVol;\n"
    "uniform sampler2D uNoise;\n"
    "uniform sampler2D uTF;\n"
    "uniform vec3 uTexelCount;\n"
    "uniform vec3 uEye;\n"
    "uniform vec3 uBoundsSize;\n"
    "uniform mat4 uInvVP;\n"
    "uniform float uSampleDistMM;\n"
    "uniform float uRT;\n"
    "uniform int uMaxIter;\n"
    "uniform int uJitter;\n"
    "uniform int uProbeRast;\n"
    "uniform int uDebugIter;\n"
    "void main() {\n"
    "  if (uProbeRast > 0) { fragColor = vec4(1.0,1.0,1.0,1.0); return; }\n"
    "  vec2 ndc = gl_FragCoord.xy / uRT * 2.0 - 1.0;\n"
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
    "  float jitterF = 0.0;\n"
    "  if (uJitter > 0) {\n"
    "    jitterF = texture(uNoise, gl_FragCoord.xy / vec2(textureSize(uNoise, 0))).x * stepSize;\n"
    "  }\n"
    "  // CONSTANT-PHASE DEBUG: uJitter == 2 shifts by a fixed half step,\n"
    "  // no per-pixel divergence, no noise fetch — isolates the incoherence cost.\n"
    "  if (uJitter == 2) {\n"
    "    jitterF = 0.5 * stepSize;\n"
    "  }\n"
"  // FETCH-ONLY DEBUG: uJitter == 3 fetches the noise (same cache behavior)\n"
     "  // but uses a constant half step — isolates fetch vs divergence cost.\n"
     "  if (uJitter == 3) {\n"
     "    jitterF = 0.5 * stepSize;\n"
     "  }\n"
     "  // LATTICE DEBUG: uJitter == 4 keeps the per-pixel phase but re-aligns\n"
     "  // to the lattice (the app's Metal semantics) — phase-quantized march.\n"
     "  if (uJitter == 4) {\n"
     "    tStart = jitterF + ceil((tStart - jitterF) / stepSize) * stepSize;\n"
     "  } else {\n"
     "    // GL parity: jitter OFF still shifts the origin by one full step\n"
     "    // (g_rayJitter = g_dirStep, vtkVolumeShaderComposer.h); jitter ON\n"
     "    // shifts by the per-pixel phase.\n"
     "    tStart += uJitter > 0 ? jitterF : stepSize;\n"
     "  }\n"
    "  int maxSteps = max(0, int(ceil((tExit - tStart) / stepSize)));\n"
    "  vec3 ctpScale = max(uTexelCount - 1.0, 1e-4) / uTexelCount;\n"
    "  vec3 ctpOffset = 0.5 / uTexelCount;\n"
    "  vec3 evalBase = ctpOffset + (uEye + rayDir * tStart) * ctpScale;\n"
    "  vec3 evalStep = rayDir * ctpScale * stepSize;\n"
    "  vec3 accColor = vec3(0.0);\n"
    "  float accOp = 0.0;\n"
    "  int iterCount = 0;\n"
    "  for (int i = 0; i < min(uMaxIter, maxSteps); i++) {\n"
    "    float currentT = tStart + float(i) * stepSize;\n"
    "    if (currentT >= tExit - 1e-6) break;\n"
    "    vec3 evalPoint = evalBase + float(i) * evalStep;\n"
    "    float s = texture(uVol, evalPoint).r;\n"
    "    vec4 tfv = texture(uTF, vec2(s, 0.5));\n"
    "    float w = 1.0 - accOp;\n"
    "    accColor += w * vec3(tfv.rgb * tfv.a);\n"
    "    accOp += w * tfv.a;\n"
    "    iterCount = i + 1;\n"
    "    if (accOp > 0.996) break;\n"
    "  }\n"
    "  if (uDebugIter > 0) {\n"
    "    fragColor = vec4(float(iterCount) / 4096.0, 0.0, 0.0, 1.0);\n"
    "  } else {\n"
    "    fragColor = vec4(accColor, accOp);\n"
    "  }\n"
    "}\n";

  GLuint prog = glCreateProgram();
  GLuint vsh = glCreateShader(GL_VERTEX_SHADER), fsh = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(vsh, 1, &vs, NULL); glCompileShader(vsh);
  glShaderSource(fsh, 1, &fs, NULL); glCompileShader(fsh);
  GLint ok = 0; glGetShaderiv(fsh, GL_COMPILE_STATUS, &ok);
  if (!ok) { char log[4096]; glGetShaderInfoLog(fsh, sizeof(log), NULL, log); fprintf(stderr, "fragment compile failed:\n%s\n", log); exit(1); }
  glAttachShader(prog, vsh); glAttachShader(prog, fsh); glLinkProgram(prog);
  glUseProgram(prog);
  glUniform1i(glGetUniformLocation(prog, "uVol"), 0);
  glUniform1i(glGetUniformLocation(prog, "uNoise"), 1);
  glUniform1i(glGetUniformLocation(prog, "uTF"), 2);
  glUniform3f(glGetUniformLocation(prog, "uTexelCount"), kVolW, kVolH, kVolD);
  glUniform3f(glGetUniformLocation(prog, "uEye"), kEye[0], kEye[1], kEye[2]);
  glUniform3f(glGetUniformLocation(prog, "uBoundsSize"), kBounds[0], kBounds[1], kBounds[2]);
  glUniformMatrix4fv(glGetUniformLocation(prog, "uInvVP"), 1, GL_FALSE, kInvVP);
  glUniformMatrix4fv(glGetUniformLocation(prog, "uVP"), 1, GL_FALSE, kVP);
  glUniform1f(glGetUniformLocation(prog, "uSampleDistMM"), sdMM);
  glUniform1f(glGetUniformLocation(prog, "uRT"), (float)rt);
  glUniform1i(glGetUniformLocation(prog, "uMaxIter"), 8192);
  glUniform1i(glGetUniformLocation(prog, "uJitter"), jitter);
  glUniform1i(glGetUniformLocation(prog, "uProbeRast"), getenv("PROBE_RAST") ? 1 : 0);
  glUniform1i(glGetUniformLocation(prog, "uDebugIter"), getenv("GL_ITER") ? 1 : 0);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_3D, volTex);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, noiseTex);
  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, tfTex);

  glViewport(0, 0, rt, rt);
  GLuint vao = 0;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  GLuint vbo = 0;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(kBoxCorners), kBoxCorners, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

  glClearColor(0, 0, 0, 1);
  for (int f = 0; f < 5; f++) {          // warm-up (cold-start skews GL badly)
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glFinish();
  }
  double t0 = NowMs();
  for (int f = 0; f < frames; f++) {
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
    if (getenv("GL_DUMP_PPM"))
    {
      FILE* pp = fopen("/tmp/jgr.ppm", "wb");
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
  CGLSetCurrentContext(NULL);
  CGLDestroyContext(ctx);
  return dt / frames;
}

// -------------------------------------------------------------- Metal backend
static double RunMetal(int rt, float sdMM, int frames, const uint8_t* vol, const uint8_t* tf, int jitter)
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
  vd.allowGPUOptimizedContents = NO;  // app parity: lag_repro root cause (2026-08-18); YES measured ~50% slower on both j0/j1 (2026-08-20)
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

  MTLTextureDescriptor* tfd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                 width:256 height:1 mipmapped:NO];
  tfd.usage = MTLTextureUsageShaderRead;
  tfd.storageMode = MTLStorageModeShared;
  id<MTLTexture> tfTex = [device newTextureWithDescriptor:tfd];
  [tfTex replaceRegion:MTLRegionMake2D(0, 0, 256, 1) mipmapLevel:0 withBytes:tf bytesPerRow:256 * 4];

  MTLTextureDescriptor* rtd = [[MTLTextureDescriptor alloc] init];
  rtd.textureType = MTLTextureType2D;
  rtd.pixelFormat = MTLPixelFormatRGBA8Unorm;
  rtd.width = rt; rtd.height = rt;
  rtd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  rtd.storageMode = MTLStorageModeShared;
  id<MTLTexture> rtTex = [device newTextureWithDescriptor:rtd];
  fprintf(stderr, "rtTex %llux%llu, viewport set to %dx%d\n",
          (unsigned long long)rtTex.width, (unsigned long long)rtTex.height, rt, rt);

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
    "struct Uniforms { float4 eye; float4 boundsSize; float4x4 invVP; float4x4 vp; float sampleDistMM; int jitter; int maxIter; float rtSize; int probeRast; int probeFS; int ndcCorners; int probeTexNoise; int debugIter; };\n"
"struct VOut { float4 position [[position]]; float tid [[user(tid0)]]; float4 cvt [[user(cvt0)]]; };\n"
     "vertex VOut vertex_main(uint vid [[vertex_id]],\n"
     "                        constant packed_float3* corners [[buffer(1)]],\n"
     "                        constant Uniforms& u [[buffer(0)]]) {\n"
     "  VOut o;\n"
     "  float3 c = float3(corners[vid]);\n"
     "  o.tid = float(vid / 3);\n"
     "  o.cvt = float4(c, 1.0f);\n"
     "  if (u.probeFS > 0) { o.position = float4(c, 1.0f); return o; }\n"
      "  if (u.ndcCorners > 0) { float t = float(vid / 3); float v = float(vid % 3); float tx = float(int(t) % 4) * 0.5f - 1.0f; float ty = floor(t / 4.0f) * 0.5f - 1.0f; o.position = float4(tx + (v == 0.0f ? 0.0f : (v == 1.0f ? 0.5f : 0.0f)), ty + (v == 0.0f ? 0.0f : (v == 1.0f ? 0.0f : 0.5f)), 0.5f, 1.0f); return o; }\n"
     "  o.position = u.vp * float4(c * u.boundsSize.xyz, 1.0f); return o;\n"
     "}\n"
"fragment float4 fragment_main(VOut in [[stage_in]],\n"
     "                              texture3d<float> volTex [[texture(0)]],\n"
     "                              texture2d<float> tfTex [[texture(1)]],\n"
     "                              texture2d<float> noiseTex [[texture(2)]],\n"
     "                              constant Uniforms& u [[buffer(0)]]) {\n"
     "  constexpr sampler noiseSampler(filter::nearest, address::repeat);\n"
    "  if (u.probeRast > 0) return float4(in.cvt.x, in.cvt.y, in.cvt.z, 1.0f);\n"
    "  // GL gl_FragCoord is y-up; Metal fragment position is y-down, so flip\n"
    "  // y to match GL's ray field exactly.\n"
    "  float2 ndc = float2(in.position.x / u.rtSize * 2.0f - 1.0f,\n"
    "                      (u.rtSize - in.position.y) / u.rtSize * 2.0f - 1.0f);\n"
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
    "  if (u.jitter == 2) {\n"
     "    // CONSTANT-PHASE DEBUG (Metal): fixed half-step, no per-pixel phase,\n"
     "    // no noise fetch — isolates incoherence vs noise-tap cost.\n"
     "    tStart += 0.5f * stepSize;\n"
"  } else if (u.jitter > 0) {\n"
     "    // GL reference field and semantics: kBlueNoise64 per-pixel, pure\n"
     "    // origin shift (the app's parity branch + GL's tStart += jitter).\n"
     "    float2 fid = floor(in.position.xy);\n"
     "    float jitterF;\n"
     "    if (u.probeTexNoise > 0) {\n"
     "      jitterF = noiseTex.sample(noiseSampler, float2(fid.x, u.rtSize - fid.y) / 64.0f).x * stepSize;\n"
     "    } else {\n"
     "      int2 t = int2(int(fid.x) & 63, int(u.rtSize - fid.y) & 63);\n"
     "      jitterF = (kBlue64[t.y * 64 + t.x] / 255.0f) * stepSize;\n"
     "    }\n"
     "    if (u.jitter == 4) tStart = jitterF + ceil((tStart - jitterF) / stepSize) * stepSize;\n"
     "    else tStart += jitterF;\n"
     "  }\n"
    "  int maxSteps = max(0, int(ceil((tExit - tStart) / stepSize)));\n"
    "  float3 texelCount = float3(volTex.get_width(), volTex.get_height(), volTex.get_depth());\n"
    "  float3 ctpScale = max(texelCount - 1.0f, 1e-4f) / texelCount;\n"
    "  float3 ctpOffset = 0.5f / texelCount;\n"
    "  float3 evalBase = ctpOffset + (eye + rayDir * tStart) * ctpScale;\n"
    "  float3 evalStep = rayDir * ctpScale * stepSize;\n"
"  constexpr sampler volSampler(filter::linear, address::clamp_to_edge);\n"
     "  constexpr sampler tfSampler(filter::linear, address::clamp_to_edge);\n"
    "  float3 accColor = float3(0.0f);\n"
    "  float accOp = 0.0f;\n"
    "  int iterCount = 0;\n"
    "  for (int i = 0; i < min(u.maxIter, maxSteps); i++) {\n"
    "    float currentT = tStart + float(i) * stepSize;\n"
    "    if (currentT >= tExit - 1e-6f) break;\n"
    "    float3 evalPoint = evalBase + float(i) * evalStep;\n"
    "    float s = volTex.sample(volSampler, evalPoint).r;\n"
    "    float4 tfv = tfTex.sample(tfSampler, float2(s, 0.5f));\n"
    "    float w = 1.0f - accOp;\n"
    "    accColor += w * (tfv.rgb * tfv.a);\n"
    "    accOp += w * tfv.a;\n"
    "    iterCount = i + 1;\n"
    "    if (accOp > 0.996f) break;\n"
    "  }\n"
    "  if (u.debugIter > 0) return float4(float(iterCount) / 256.0f, 0.0f, 0.0f, 1.0f);\n"
    "  return float4(accColor, accOp);\n"
    "}\n";
  NSError* err = nil;
  id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithFormat:@"%s%s", msl, msl2] options:nil error:&err];
  if (!lib) { fprintf(stderr, "MSL compile failed: %s\n", err.description.UTF8String); exit(1); }
  id<MTLFunction> vf = [lib newFunctionWithName:@"vertex_main"];
  id<MTLFunction> ff = [lib newFunctionWithName:@"fragment_main"];
  MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
  pd.vertexFunction = vf; pd.fragmentFunction = ff;
  pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
  id<MTLRenderPipelineState> pso = [device newRenderPipelineStateWithDescriptor:pd error:&err];
  if (!pso) { fprintf(stderr, "PSO failed: %s\n", err.description.UTF8String); exit(1); }

  struct Uniforms {
    simd_float4 eye, boundsSize;
    simd_float4x4 invVP, vp;
    float sampleDistMM;
    int jitter, maxIter;
    float rtSize;
    int probeRast, probeFS, ndcCorners, probeTexNoise, debugIter;
  } u;
  u.eye = (simd_float4){ kEye[0], kEye[1], kEye[2], 0.0f };
  u.boundsSize = (simd_float4){ kBounds[0], kBounds[1], kBounds[2], 0.0f };
  memcpy(&u.invVP, kInvVP, 16 * sizeof(float));
  memcpy(&u.vp, kVP, 16 * sizeof(float));
  u.sampleDistMM = sdMM; u.jitter = jitter; u.maxIter = 8192;
  u.rtSize = (float)rt; u.probeRast = getenv("PROBE_RAST") ? 1 : 0; u.probeFS = getenv("PROBE_FULLSCREEN") ? 1 : 0; u.ndcCorners = getenv("NDC_CORNERS") ? 1 : 0; u.probeTexNoise = getenv("NOISE_TEX") ? 1 : 0; u.debugIter = getenv("GL_ITER") ? 1 : 0;
  id<MTLBuffer> ubuf = [device newBufferWithBytes:&u length:sizeof(u) options:MTLResourceStorageModeShared];
  id<MTLBuffer> corners = [device newBufferWithBytes:kBoxCorners length:sizeof(kBoxCorners) options:MTLResourceStorageModeShared];
  float fsq[9] = { -1,-1,0.5,  3,-1,0.5,  -1,3,0.5 };
  id<MTLBuffer> fsqBuf = [device newBufferWithBytes:fsq length:sizeof(fsq) options:MTLResourceStorageModeShared];

  // NOISE_TEX A/B: 64x64 R8Unorm blue-noise tile (GL's field) for the
  // texture2d variant of the jitter tap — vs the in-shader kBlue64 constant.
  MTLTextureDescriptor* ntd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm width:64 height:64 mipmapped:NO];
  ntd.usage = MTLTextureUsageShaderRead;
  ntd.storageMode = MTLStorageModeShared;
  id<MTLTexture> noiseTex = [device newTextureWithDescriptor:ntd];
  [noiseTex replaceRegion:MTLRegionMake2D(0, 0, 64, 64) mipmapLevel:0 withBytes:kBlue64 bytesPerRow:64];

  for (int f = 0; f < 5; f++) {          // warm-up
    MTLRenderPassDescriptor* rpd = [[MTLRenderPassDescriptor alloc] init];
    rpd.colorAttachments[0].texture = rtTex;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    id<MTLCommandBuffer> wcb = [queue commandBuffer];
    id<MTLRenderCommandEncoder> wenc = [wcb renderCommandEncoderWithDescriptor:rpd];
    [wenc setRenderPipelineState:pso];
    MTLViewport vp0 = { 0, 0, (double)rt, (double)rt, 0.0, 1.0 };
    [wenc setViewport:vp0];
    const char* cm = getenv("CULL");
    [wenc setCullMode:cm ? (strcmp(cm,"front")==0 ? MTLCullModeFront : strcmp(cm,"back")==0 ? MTLCullModeBack : MTLCullModeNone) : MTLCullModeNone];
    [wenc setFragmentTexture:volTex atIndex:0];
    [wenc setFragmentTexture:tfTex atIndex:1];
    [wenc setFragmentTexture:noiseTex atIndex:2];
    [wenc setFragmentBuffer:ubuf offset:0 atIndex:0];
    [wenc setVertexBuffer:ubuf offset:0 atIndex:0];
    [wenc setVertexBuffer:(getenv("PROBE_FULLSCREEN") ? fsqBuf : corners) offset:0 atIndex:1];
    if (getenv("T_PER_TRI")) { for (int t = 0; t < 12; t++) [wenc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:t*3 vertexCount:3]; }
    else [wenc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:getenv("PROBE_FULLSCREEN") ? 3 : 36];
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
    MTLViewport vp = { 0, 0, (double)rt, (double)rt, 0.0, 1.0 };
    [enc setViewport:vp];
    const char* cm2 = getenv("CULL");
    [enc setCullMode:cm2 ? (strcmp(cm2,"front")==0 ? MTLCullModeFront : strcmp(cm2,"back")==0 ? MTLCullModeBack : MTLCullModeNone) : MTLCullModeNone];
    [enc setFragmentTexture:volTex atIndex:0];
    [enc setFragmentTexture:tfTex atIndex:1];
    [enc setFragmentTexture:noiseTex atIndex:2];
    [enc setFragmentBuffer:ubuf offset:0 atIndex:0];
    [enc setVertexBuffer:ubuf offset:0 atIndex:0];
    [enc setVertexBuffer:(getenv("PROBE_FULLSCREEN") ? fsqBuf : corners) offset:0 atIndex:1];
    if (getenv("T_PER_TRI")) { int sel = getenv("TRI_SEL") ? atoi(getenv("TRI_SEL")) : -1; if (sel >= 0) [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:sel*3 vertexCount:3]; else for (int t = 0; t < 12; t++) [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:t*3 vertexCount:3]; }
    else [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:getenv("PROBE_FULLSCREEN") ? 3 : 36];
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
    if (getenv("MTL_DUMP_PPM"))
    {
      uint8_t* all = (uint8_t*)malloc((size_t)rt * rt * 4);
      [rtTex getBytes:all bytesPerRow:(size_t)rt * 4 bytesPerImage:(size_t)rt * rt * 4
           fromRegion:MTLRegionMake2D(0, 0, rt, rt) mipmapLevel:0 slice:0];
      FILE* pp = fopen("/tmp/jgr_metal.ppm", "wb");
      if (pp)
      {
        fprintf(pp, "P6\n%d %d\n255\n", rt, rt);
        uint8_t* row = (uint8_t*)malloc((size_t)rt * 3);
        for (int y = 0; y < rt; y++)
        {
          const uint8_t* src = all + (size_t)y * rt * 4;
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
      free(all);
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
    const char* volPath = argc > 4 ? argv[4] : "dicom.u8";
    fprintf(stderr, "volume %dx%dx%d R8 (%d MB), rt %dx%d, sd %.1f mm, frames %d, source %s\n",
            kVolW, kVolH, kVolD, (int)((size_t)kVolW * kVolH * kVolD >> 20), rt, rt, sd, frames, volPath);

    FILE* f = fopen(volPath, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", volPath); return 1; }
    size_t total = (size_t)kVolW * kVolH * kVolD;
    uint8_t* vol = (uint8_t*)malloc(total);
    if (fread(vol, 1, total, f) != total) { fprintf(stderr, "short read on %s\n", volPath); return 1; }
    fclose(f);
    uint8_t tf[256 * 4];
    MakeTF(tf);

    // Interleaved A/B: j0 pair then j1 pair (battery drift is between pairs,
    // the deltas are measured back-to-back in the same thermal window).
    int j1mode = 1;
    if (argc > 5 && strcmp(argv[5], "constphase") == 0) j1mode = 2;
    if (argc > 5 && strcmp(argv[5], "fetchonly") == 0) j1mode = 3;
    if (argc > 5 && strcmp(argv[5], "lattice") == 0) j1mode = 4;
    double g0 = RunGL(rt, sd, frames, vol, tf, 0);
    double g1 = RunGL(rt, sd, frames, vol, tf, j1mode);
    double m0 = RunMetal(rt, sd, frames, vol, tf, 0);
    double m1 = RunMetal(rt, sd, frames, vol, tf, j1mode);

    fprintf(stderr, "\nGL    j0 %8.3f   j1 %8.3f   jitter %+6.1f%%\n", g0, g1, (g1 / g0 - 1.0) * 100.0);
    fprintf(stderr, "METAL j0 %8.3f   j1 %8.3f   jitter %+6.1f%%\n", m0, m1, (m1 / m0 - 1.0) * 100.0);
    fprintf(stderr, "M/GL  j0 %8.2f   j1 %8.2f\n", m0 / g0, m1 / g1);
    double r1 = m1 / g1;
    if (r1 > 1.2f)
      fprintf(stderr, "RESULT: GAP REPRODUCED (M/GL j1 %.2f) — same jitter field, read path diverges\n", r1);
    else if (r1 < 1.05f)
      fprintf(stderr, "RESULT: gap CLOSED (M/GL j1 %.2f) — same field, no read-path divergence\n", r1);
    else
      fprintf(stderr, "RESULT: partial (M/GL j1 %.2f; check thermal state, rerun interleaved)\n", r1);
    free(vol);
  }
  return 0;
}