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

#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <OpenGL/gl3.h>
#import <OpenGL/OpenGL.h>
#import <QuartzCore/QuartzCore.h>
#import <simd/simd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <string>
#include <vector>

// Knob summary (HARNESS_VS_APP_GAP.md §22 HANDOFF + issue_tax_repro R3):
//   WARMUP=n   warm-up draws per timed block (default 30; the historical 5 is
//              proven contaminated — Apple's driver re-JITs fresh complex
//              programs far past 10 draws; first block ran +32% hot).
//   ROUNDS=n   order-alternated interleaved j0/j1 rounds (default 1), mean±sd
//              reported; round 1 shows any residual warm-in transient.
//   SURFACE=1  window-backed NSOpenGLView context (drawable attached) instead
//              of headless CGL+FBO — HANDOFF candidate 1.
//   BLEND=1    GL_BLEND (ONE, ONE_MINUS_SRC_ALPHA) during the timed draw (the
//              app composites through it) — HANDOFF candidate 2.
//   UBO=1      FS uniforms via std140 UBO instead of per-draw glUniform* —
//              HANDOFF candidate 4.
//   PAD=1      dead-but-unremovable prologue work (texture-seeded ALU chains)
//              to emulate app register pressure — HANDOFF candidate 5.

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
extern "C" void glTexStorage3D(unsigned target, int levels, unsigned internalformat, int w, int h, int d);

// ---------------------------------------------------------------- GL backend
static const int kWarmupDefault = 30;  // issue_tax_repro R3: 10 insufficient
// Process-lifetime surface objects: a scoped NSWindow would deallocate while
// its NSOpenGLContext survives, crashing the next drawable-backed run.
static NSWindow* sWin = nil;
static NSOpenGLView* sView = nil;

static double RunGL(int rt, float sdMM, int frames, const uint8_t* vol, const uint8_t* tf, int jitter)
{
  const int warmupN = getenv("WARMUP") ? atoi(getenv("WARMUP")) : kWarmupDefault;
  const bool surfMode = getenv("SURFACE") != NULL;
  const bool blendMode = getenv("BLEND") != NULL;
  const bool uboMode = getenv("UBO") != NULL;
  const bool padMode = getenv("PAD") != NULL;
  // AZSTEP=deg: per-frame camera orbit about world-Y through the volume
  // center — replicates the app bench, which calls camera->Azimuth(0.1)
  // INSIDE its timed loop so rays change every frame (TestMetalGLVisualComparison.cxx).
  const float azStep = getenv("AZSTEP") ? (float)atof(getenv("AZSTEP")) : 0.0f;
  // GL41=1: request the 4.1 core profile (the app's VTK context class)
  // instead of 3.2 core — different driver entry may pick different texture
  // tiling/sampling paths under phase-scattered trilinear.
  const char* profEnv = getenv("GL41");
  int glProfile = (int)(profEnv ? kCGLOGLPVersion_GL4_Core : kCGLOGLPVersion_3_2_Core);
  NSOpenGLContext* viewCtx = nil;
  CGLContextObj ctx = NULL;
  if (surfMode)
  {
    // SURFACE=1 (HANDOFF candidate 1): window-backed NSOpenGLView — the app
    // renders through an NSOpenGLContext with a real drawable surface; the
    // old harness was pure offscreen CGL+FBO.
    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
    NSRect frame = NSMakeRect(40, 40, rt, rt);
    NSOpenGLPixelFormatAttribute vattrs[] = {
      NSOpenGLPFADoubleBuffer,
      NSOpenGLPFANoRecovery,
      NSOpenGLPFAAccelerated,
      NSOpenGLPFAOpenGLProfile,
      (NSOpenGLPixelFormatAttribute)(profEnv ? NSOpenGLProfileVersion4_1Core : NSOpenGLProfileVersion3_2Core),
      (NSOpenGLPixelFormatAttribute)0 };
    NSOpenGLPixelFormat* vpf = [[NSOpenGLPixelFormat alloc] initWithAttributes:vattrs];
    if (!vpf) { fprintf(stderr, "no NSOpenGL pixel format\n"); exit(1); }
    NSOpenGLView* view = [[NSOpenGLView alloc] initWithFrame:frame pixelFormat:vpf];
    view.wantsBestResolutionOpenGLSurface = NO;  // backing store == points == rt
    // defer:NO — a deferred borderless window never creates its native
    // surface until moved onscreen, leaving FB0 incomplete.
    NSWindow* win = [[NSWindow alloc] initWithContentRect:frame
        styleMask:NSWindowStyleMaskBorderless backing:NSBackingStoreBuffered defer:NO];
    win.contentView = view;
    win.title = @"jgr";
    win.releasedWhenClosed = NO;
    [win orderFrontRegardless];  // attach the surface without stealing focus
    sWin = win;
    sView = view;
    viewCtx = view.openGLContext;
    [viewCtx setView:view];
    [viewCtx update];
    [viewCtx makeCurrentContext];
    ctx = viewCtx.CGLContextObj;
    fprintf(stderr, "[glnob] profile=%s surface=drawable %ux%u\n",
            profEnv ? "4.1-core" : "3.2-core",
            (unsigned)view.bounds.size.width, (unsigned)view.bounds.size.height);
  }
  else
  {
    CGLPixelFormatAttribute attrs[] = {
      kCGLPFAAccelerated, kCGLPFAOpenGLProfile, (CGLPixelFormatAttribute)glProfile, (CGLPixelFormatAttribute)0 };
    CGLPixelFormatObj pf = NULL; GLint npf = 0;
    if (CGLChoosePixelFormat(attrs, &pf, &npf) != kCGLNoError || !pf) { fprintf(stderr, "no GL core pixel format (profile %d)\n", (int)glProfile); exit(1); }
    fprintf(stderr, "[glnob] profile=%s surface=headless-FBO\n", profEnv ? "4.1-core" : "3.2-core");
    if (CGLCreateContext(pf, NULL, &ctx) != kCGLNoError || !ctx) { fprintf(stderr, "no GL context\n"); exit(1); }
    CGLSetCurrentContext(ctx);
    CGLDestroyPixelFormat(pf);
  }
  fprintf(stderr, "[glnob] warmup=%d%s%s%s%s%s%s\n", warmupN,
          surfMode ? " SURFACE" : "", blendMode ? " BLEND" : "",
          uboMode ? " UBO" : "", padMode ? " PAD" : "",
          azStep != 0.0f ? " AZSTEP" : "", getenv("CLIP") ? " CLIP" : "");

  GLuint volTex = 0;
  glGenTextures(1, &volTex);
  glBindTexture(GL_TEXTURE_3D, volTex);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  // GLSTORAGE=1: immutable storage (glTexStorage3D, the app/VTK upload
  // class) instead of mutable glTexImage3D — lets the driver commit to its
  // optimal tiling up front. A/B for the phase-scattered trilinear tax.
  if (getenv("GLSTORAGE"))
  {
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_R8, kVolW, kVolH, kVolD);
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, kVolW, kVolH, kVolD, GL_RED, GL_UNSIGNED_BYTE, vol);
    fprintf(stderr, "[glnob] storage=immutable\n");
  }
  else
  {
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, kVolW, kVolH, kVolD, 0, GL_RED, GL_UNSIGNED_BYTE, vol);
    fprintf(stderr, "[glnob] storage=mutable\n");
  }

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

  // Split-TF knob (SPLIT_TF=1, H4 A/B): app-shaped color/opacity LUTs built
  // from the same tf data (color = rgb + a=255, opacity = a + 0 r/g/b).
  GLuint colorTex = 0, opTex = 0;
  if (getenv("SPLIT_TF"))
  {
    uint8_t* colorLut = (uint8_t*)malloc(256 * 4);
    uint8_t* opLut = (uint8_t*)malloc(256 * 4);
    for (int i = 0; i < 256; i++)
    {
      colorLut[i * 4 + 0] = tf[i * 4 + 0];
      colorLut[i * 4 + 1] = tf[i * 4 + 1];
      colorLut[i * 4 + 2] = tf[i * 4 + 2];
      colorLut[i * 4 + 3] = 255;
      opLut[i * 4 + 0] = opLut[i * 4 + 1] = opLut[i * 4 + 2] = tf[i * 4 + 3];
      opLut[i * 4 + 3] = tf[i * 4 + 3];
    }
    glGenTextures(1, &colorTex);
    glBindTexture(GL_TEXTURE_2D, colorTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, colorLut);
    glGenTextures(1, &opTex);
    glBindTexture(GL_TEXTURE_2D, opTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, opLut);
    free(colorLut);
    free(opLut);
  }

  GLuint rtTex = 0, fbo = 0, depthRbo = 0;
  if (!surfMode)
  {
    glGenTextures(1, &rtTex);
    glBindTexture(GL_TEXTURE_2D, rtTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rt, rt, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rtTex, 0);
  }
  else
  {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);  // the drawable's back buffer
  }
  // DEPTH knob (DEPTH=1, H4 A/B §3): attach a depth RBO so the pass carries
  // the same depth attachment as the app, and bind a dummy depth texture
  // fetched at FS start (pass split / TBDR structure without changing march).
  GLuint depthTex = 0;
  if (getenv("DEPTH"))
  {
    glGenTextures(1, &depthTex);
    glBindTexture(GL_TEXTURE_2D, depthTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, rt, rt, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTex, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
  }
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
  // VOL_FETCH knob (LOD=1): textureLod kills the implicit dFdx/dFdy gradient
  // on the per-pixel-jittered evalPoint (H8 A/B, HARNESS_VS_APP_GAP.md §11).
  // STEP_BODY knob (SPLIT_TF=1): app-shaped step — separate color/opacity LUTs
  // with an a>0 gate (H4 A/B). The app's texture3D fetch stays texture() for
  // this knob unless LOD=1 is also set.
  // LOOP_SHAPE knob (WHILE=1, H4 A/B): the app's while loop with accumulated
  // evalPoint/currentT (loop-carried dependent chain) and a 6-way box-exit
  // test on the eval lattice — compiler/occupancy shape, not algorithm.
  // DOEXIT knob (V31 port): all exit conditions moved into the loop
  // BACK-EDGE (do-while) — one branch per iteration instead of two around
  // the fetch. Bit-exact per-iteration values; the shape that closed the
  // MSL codegen deficit in divergent_tail.
  const char* loopShape = getenv("DOEXIT")
    ? "  int i = 0;\n"
      "  if (i < min(uMaxIter, maxSteps)) {\n"
      "  do {\n"
      "    float currentT = tStart + float(i) * stepSize;\n"
      "    vec3 evalPoint = evalBase + float(i) * evalStep;\n"
      "    float s = VOL_FETCH;\n"
      "    %s"
      "    iterCount = i + 1;\n"
      "    ++i;\n"
      "  } while (i < min(uMaxIter, maxSteps)\n"
      "        && tStart + float(i) * stepSize < tExit - 1e-6\n"
      "        && accOp <= 0.996);\n"
      "  }\n"
    : getenv("INCR")
    ? // INCR=1: the composer's incremental position accumulation
      // (g_dataPos += g_dirStep) instead of analytic index-based positions.
      "  vec3 evalPoint = evalBase;\n"
      "  for (int i = 0; i < min(uMaxIter, maxSteps); i++) {\n"
      "    float s = VOL_FETCH;\n"
      "    %s"
      "    iterCount = i + 1;\n"
      "    if (accOp > 0.996) break;\n"
      "    evalPoint += evalStep;\n"
      "  }\n"
    : getenv("WHILE")
    ? "  float currentT = tStart;\n"
      "  vec3 evalPoint = evalBase;\n"
      "  int i = 0;\n"
      "  while (i < min(uMaxIter, maxSteps)) {\n"
      "    if (currentT >= tExit - 1e-6) break;\n"
      "    if (evalPoint.x < 0.0 || evalPoint.x > 1.0 ||\n"
      "        evalPoint.y < 0.0 || evalPoint.y > 1.0 ||\n"
      "        evalPoint.z < 0.0 || evalPoint.z > 1.0) break;\n"
      "    float s = VOL_FETCH;\n"
      "    %s"
      "    iterCount = i + 1;\n"
      "    if (accOp > 0.996) break;\n"
      "    evalPoint += evalStep;\n"
      "    currentT += stepSize;\n"
      "    i++;\n"
      "  }\n"
    : "  for (int i = 0; i < min(uMaxIter, maxSteps); i++) {\n"
      "    float currentT = tStart + float(i) * stepSize;\n"
      "    if (currentT >= tExit - 1e-6) break;\n"
      "    vec3 evalPoint = evalBase + float(i) * evalStep;\n"
      "    float s = VOL_FETCH;\n"
      "    %s"
      "    iterCount = i + 1;\n"
      "    if (accOp > 0.996) break;\n"
      "  }\n";
  const char* fetchDef = getenv("LOD")
    ? "#define VOL_FETCH textureLod(uVol, evalPoint, 0.0).r\n"
    : "#define VOL_FETCH texture(uVol, evalPoint).r\n";
  const char* stepBody = getenv("SPLIT_TF")
    ? "    float a = texture(uOpacityTF, vec2(s, 0.5)).r;\n"
      "    if (a > 0.0) {\n"
      "      vec3 c = texture(uColorTF, vec2(s, 0.5)).rgb;\n"
      "      float w = 1.0 - accOp;\n"
      "      accColor += w * (c * a);\n"
      "      accOp += w * a;\n"
      "    }\n"
    : "    vec4 tfv = texture(uTF, vec2(s, 0.5));\n"
      "    float w = 1.0 - accOp;\n"
      "    accColor += w * vec3(tfv.rgb * tfv.a);\n"
      "    accOp += w * tfv.a;\n";
  // DEPTH knob (DEPTH=1, H4 A/B §3): dummy depth-texture fetch at FS start
  // (NEAREST, full-res) fed into the composite with a tiny weight — tests the
  // pass split / TBDR structure without changing the march (the value is
  // ~0 on the volume interior since the FBO depth is cleared to 1.0).
  const char* depthFetch = getenv("DEPTH")
    ? "  float dummyDepth = texture(uDepth, gl_FragCoord.xy / uRT).r;\n"
    : "  float dummyDepth = 0.0;\n";
  // UBO=1 (HANDOFF candidate 4): move the FS value uniforms into a std140
  // uniform block fed by a buffer object instead of per-draw glUniform*.
  std::string uniDecl;
  if (!uboMode)
  {
    uniDecl =
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
      "uniform int uRayDump;\n"
      "uniform mat4 uAz;\n"
      "uniform vec3 uEyeAz;\n"
      "uniform int uClip;\n"
      "uniform mat4 uClipT2O;\n"
      "uniform mat4 uClipO2T;\n"
      "uniform vec3 uClipOrigin;\n"
      "uniform vec3 uClipNormal;\n";
  }
  else
  {
    uniDecl =
      "layout(std140) uniform UBlock {\n"
      "  vec3 uTexelCount; vec3 uEye; vec3 uBoundsSize; mat4 uInvVP;\n"
      "  float uSampleDistMM; float uRT; int uMaxIter; int uJitter;\n"
      "  int uProbeRast; int uDebugIter; int uRayDump;\n"
      "};\n"
      "uniform mat4 uAz;\n"
      "uniform vec3 uEyeAz;\n"
      "uniform int uClip;\n"
      "uniform mat4 uClipT2O;\n"
      "uniform mat4 uClipO2T;\n"
      "uniform vec3 uClipOrigin;\n"
      "uniform vec3 uClipNormal;\n";  }
  // PAD=1 (HANDOFF candidate 5): dead-but-unremovable prologue work — texture-
  // seeded ALU chains kept live ACROSS the march loop by a 1e-9-weighted
  // consume in the output — emulates the app FS's register pressure.
  const char* padCode = padMode ?
    "  vec4 padA = texture(uNoise, gl_FragCoord.xy / vec2(textureSize(uNoise, 0)) + vec2(0.25)).xyzw;\n"
    "  vec4 padB = padA * 91.17 + vec4(uRT * 0.173);\n"
    "  vec4 padC = padA.yzxw * 47451.31 - vec4(uSampleDistMM);\n"
    "  vec4 padD = padB * padC + padA.wzyx;\n"
    "  padD = padD * 1.13 + padC * 0.97;\n"
    : "";
  const char* padConsume = padMode ? " + dot(padD, vec4(1e-9))" : "";
  // CLIP=1: the app's AdjustSampleRangeForClipping prologue, ported with the
  // same instruction shape (tex->obj mat4, uniform-fed plane loop, obj->tex
  // mat4, texMin/texMax validity test). The scene's plane (normal +Z at the
  // min-Z face) clips NOTHING geometrically — like the app bench config, the
  // work runs every fragment while the march stays bit-identical.
  const char* clipCode = getenv("CLIP") ?
    "  if (uClip > 0) {\n"
    "    vec3 ctpS = max(uTexelCount - 1.0, 1e-4) / uTexelCount;\n"
    "    vec3 ctpO = 0.5 / uTexelCount;\n"
    "    vec3 startTex = ctpO + (eyeN + rayDir * tStart) * ctpS;\n"
    "    vec3 stopTex = ctpO + (eyeN + rayDir * tExit) * ctpS;\n"
    "    vec3 latOff = rayDir * (ctpS * jitterF);\n"
    "    vec4 startPosObj = uClipT2O * vec4(startTex - latOff, 1.0);\n"
    "    startPosObj = startPosObj / startPosObj.w;\n"
    "    vec4 stopPosObj = uClipT2O * vec4(stopTex, 1.0);\n"
    "    stopPosObj = stopPosObj / stopPosObj.w;\n"
    "    vec3 dirObj = normalize(rayDir * uBoundsSize);\n"
    "    float startDistance = dot(uClipNormal, uClipOrigin - startPosObj.xyz);\n"
    "    float stopDistance = dot(uClipNormal, uClipOrigin - stopPosObj.xyz);\n"
    "    bool startClipped = startDistance > 0.0;\n"
    "    bool stopClipped = stopDistance > 0.0;\n"
    "    if (!(startClipped && stopClipped)) {\n"
    "      float rayDotNormal = dot(dirObj, uClipNormal);\n"
    "      bool frontFace = rayDotNormal > 0.0;\n"
    "      if (frontFace && startClipped) {\n"
    "        float rayScaledDist = startDistance / rayDotNormal;\n"
    "        startPosObj = vec4(startPosObj.xyz + rayScaledDist * dirObj, 1.0);\n"
    "        vec4 nsp = uClipO2T * startPosObj;\n"
    "        nsp /= nsp.w;\n"
    "        vec3 q = (nsp.xyz + latOff - ctpO) / ctpS - eyeN;\n"
    "        tStart = dot(q, rayDir) / max(dot(rayDir, rayDir), 1e-12);\n"
    "      }\n"
    "      if (!frontFace && stopClipped) {\n"
    "        float rayScaledDist = stopDistance / rayDotNormal;\n"
    "        stopPosObj = vec4(stopPosObj.xyz + rayScaledDist * dirObj, 1.0);\n"
    "        vec4 nsp = uClipO2T * stopPosObj;\n"
    "        nsp /= nsp.w;\n"
    "        vec3 q2 = (nsp.xyz - ctpO) / ctpS - eyeN;\n"
    "        tExit = dot(q2, rayDir) / max(dot(rayDir, rayDir), 1e-12);\n"
    "      }\n"
    "      if (any(greaterThan(startTex, uTexelCount)) ||\n"
    "          any(lessThan(startTex, vec3(0.0)))) {\n"
    "        fragColor = vec4(0.0);\n"
    "        return;\n"
    "      }\n"
    "      int maxStepsC = max(0, int(ceil((tExit - tStart) / stepSize)));\n"
    "      maxSteps = min(maxSteps, maxStepsC);\n"
    "    }\n"
    "  }\n"
    : "";
  char* fsBuf = (char*)malloc(24576);
  char* loopBuf = (char*)malloc(4096);
  snprintf(loopBuf, 4096, loopShape, stepBody);
  snprintf(fsBuf, 24576,
    "#version 150\n"
    "%s"
    "out vec4 fragColor;\n"
    "uniform sampler3D uVol;\n"
    "uniform sampler2D uNoise;\n"
    "uniform sampler2D uTF;\n"
    "uniform sampler2D uColorTF;\n"
    "uniform sampler2D uOpacityTF;\n"
    "uniform sampler2D uDepth;\n"
    "%s"
    "void main() {\n"
    "  if (uProbeRast > 0) { fragColor = vec4(1.0,1.0,1.0,1.0); return; }\n"
    "  %s"
    "  %s"
    "  vec2 ndc = gl_FragCoord.xy / uRT * 2.0 - 1.0;\n"
    "  vec4 w4 = uInvVP * vec4(ndc, 0.0, 1.0);\n"
    "  // AZSTEP: orbit the camera about world-Y through the volume center\n"
    "  // (the app bench's per-frame Azimuth(0.1)); identity when disabled.\n"
    "  vec3 ptPhys = (uAz * vec4(w4.xyz / w4.w, 1.0)).xyz;\n"
    "  vec3 eyeN = uEyeAz;\n"
    "  vec3 rayDir = normalize(ptPhys / uBoundsSize - eyeN);\n"
    "  vec3 inv = 1.0 / rayDir;\n"
    "  vec3 t0 = (vec3(0.0) - eyeN) * inv;\n"
    "  vec3 t1 = (vec3(1.0) - eyeN) * inv;\n"
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
    "  %s"
    "  vec3 ctpScale = max(uTexelCount - 1.0, 1e-4) / uTexelCount;\n"
    "  vec3 ctpOffset = 0.5 / uTexelCount;\n"
    "  vec3 evalBase = ctpOffset + (eyeN + rayDir * tStart) * ctpScale;\n"
    "  vec3 evalStep = rayDir * ctpScale * stepSize;\n"
    "  vec3 accColor = vec3(0.0);\n"
    "  float accOp = 0.0;\n"
    "  int iterCount = 0;\n"
    "  %s"
    "  if (uDebugIter > 0) {\n"
    "    fragColor = vec4(float(iterCount) / 4096.0, 0.0, 0.0, 1.0);\n"
    "  } else if (uRayDump == 1) {\n"
    "    fragColor = vec4(evalBase, 1.0);\n"
    "  } else if (uRayDump == 2) {\n"
    "    fragColor = vec4(evalStep * uTexelCount / 16.0, 1.0);\n"
    "  } else {\n"
    "    fragColor = vec4(accColor + dummyDepth * 1e-4%s, accOp);\n"
    "  }\n"
    "}\n", fetchDef, uniDecl.c_str(), depthFetch, padCode, clipCode, loopBuf, padConsume);
  const char* fs = fsBuf;

  GLuint prog = glCreateProgram();
  GLuint vsh = glCreateShader(GL_VERTEX_SHADER), fsh = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(vsh, 1, &vs, NULL); glCompileShader(vsh);
  glShaderSource(fsh, 1, &fs, NULL); glCompileShader(fsh);
  GLint ok = 0; glGetShaderiv(fsh, GL_COMPILE_STATUS, &ok);
  if (!ok) { char log[4096]; glGetShaderInfoLog(fsh, sizeof(log), NULL, log); fprintf(stderr, "fragment compile failed:\n%s\n", log); exit(1); }
  free(fsBuf);
  free(loopBuf);
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
  glUniform1i(glGetUniformLocation(prog, "uRayDump"), getenv("RAYS") ? atoi(getenv("RAYS")) : 0);
  glUniform1i(glGetUniformLocation(prog, "uColorTF"), 3);
  glUniform1i(glGetUniformLocation(prog, "uOpacityTF"), 4);
  glUniform1i(glGetUniformLocation(prog, "uDepth"), 5);

  if (uboMode)
  {
    // Mirrors layout(std140) UBlock (offsets 0/16/32/48, scalars from 112).
    struct UBOData {
      float texelCount[4], eye[4], boundsSize[4], invVP[16];
      float sampleDistMM, rt; int maxIter, jitter;
      int probeRast, debugIter, rayDump, endPad;
    } ub;
    memset(&ub, 0, sizeof(ub));
    ub.texelCount[0] = kVolW; ub.texelCount[1] = kVolH; ub.texelCount[2] = kVolD;
    ub.eye[0] = kEye[0]; ub.eye[1] = kEye[1]; ub.eye[2] = kEye[2];
    ub.boundsSize[0] = kBounds[0]; ub.boundsSize[1] = kBounds[1]; ub.boundsSize[2] = kBounds[2];
    memcpy(ub.invVP, kInvVP, sizeof(ub.invVP));
    ub.sampleDistMM = sdMM; ub.rt = (float)rt; ub.maxIter = 8192; ub.jitter = jitter;
    ub.probeRast = getenv("PROBE_RAST") ? 1 : 0;
    ub.debugIter = getenv("GL_ITER") ? 1 : 0;
    ub.rayDump = getenv("RAYS") ? atoi(getenv("RAYS")) : 0;
    GLuint uboBuf = 0;
    glGenBuffers(1, &uboBuf);
    glBindBuffer(GL_UNIFORM_BUFFER, uboBuf);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(ub), &ub, GL_STATIC_DRAW);
    GLuint bi = glGetUniformBlockIndex(prog, "UBlock");
    glUniformBlockBinding(prog, bi, 7);
    glBindBufferBase(GL_UNIFORM_BUFFER, 7, uboBuf);
    fprintf(stderr, "[glnob] UBO bound (block idx %u, %zu bytes)\n", bi, sizeof(ub));
  }

  if (blendMode)
  {
    // BLEND=1 (HANDOFF candidate 2): the app composites the volume pass
    // through premultiplied-over blending; read-modify-write keeps the RT in
    // tile memory. Output image on the cleared black target is ~unchanged.
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    fprintf(stderr, "[glnob] blend ONE/ONE_MINUS_SRC_ALPHA\n");
  }

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_3D, volTex);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, noiseTex);
  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, tfTex);
  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, colorTex);
  glActiveTexture(GL_TEXTURE4);
  glBindTexture(GL_TEXTURE_2D, opTex);
  glActiveTexture(GL_TEXTURE5);
  glBindTexture(GL_TEXTURE_2D, depthTex);

  glUniform1i(glGetUniformLocation(prog, "uClip"), getenv("CLIP") ? 1 : 0);
  {
    // tex->obj / obj->tex: obj_mm = (tex - offset)/scale * bounds (the
    // composer's clip_texToObjMat/clip_objToTexMat pair).
    float t2o[16] = {0}, o2t[16] = {0};
    const float sc[3] = { (kVolW - 1.0f) / kVolW, (kVolH - 1.0f) / kVolH, (kVolD - 1.0f) / kVolD };
    const float of[3] = { 0.5f / kVolW, 0.5f / kVolH, 0.5f / kVolD };
    for (int a = 0; a < 3; a++)
    {
      t2o[a * 4 + a] = kBounds[a] / sc[a];       // column a, row a
      t2o[12 + a] = -of[a] * kBounds[a] / sc[a]; // column 3, row a
      o2t[a * 4 + a] = sc[a] / kBounds[a];
      o2t[12 + a] = of[a];
    }
    t2o[15] = o2t[15] = 1.0f;
    glUniformMatrix4fv(glGetUniformLocation(prog, "uClipT2O"), 1, GL_FALSE, t2o);
    glUniformMatrix4fv(glGetUniformLocation(prog, "uClipO2T"), 1, GL_FALSE, o2t);
    // the DICOMVolume scene's plane: normal +Z at the min-Z face — clips nothing
    glUniform3f(glGetUniformLocation(prog, "uClipOrigin"), 0.0f, 0.0f, 0.0f);
    glUniform3f(glGetUniformLocation(prog, "uClipNormal"), 0.0f, 0.0f, 1.0f);
  }

  GLint azLoc = glGetUniformLocation(prog, "uAz");
  GLint eyeAzLoc = glGetUniformLocation(prog, "uEyeAz");
  {
    float ident[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
    glUniformMatrix4fv(azLoc, 1, GL_FALSE, ident);
    glUniform3f(eyeAzLoc, kEye[0], kEye[1], kEye[2]);
  }
  double azCur = 0.0;
  auto AdvanceAzimuth = [&]()
  {
    if (azStep == 0.0f) return;
    azCur += azStep;
    const double th = azCur * M_PI / 180.0;
    const double cth = cos(th), sth = sin(th);
    // p' = c + R*(p - c) in physical mm; column-major mat4 for glUniform.
    const float cx = kBounds[0] * 0.5f, cyv = kBounds[1] * 0.5f, cz = kBounds[2] * 0.5f;
    float m[16] = { 0 };
    m[0] = (float)cth;  m[2] = (float)sth;
    m[5] = 1.0f;
    m[8] = (float)-sth; m[10] = (float)cth;
    m[12] = (float)(cx - cth * cx - sth * cz);
    m[13] = 0.0f;
    m[14] = (float)(cz + sth * cx - cth * cz);
    m[15] = 1.0f;
    glUniformMatrix4fv(azLoc, 1, GL_FALSE, m);
    const double ex = kEye[0] * kBounds[0], ez = kEye[2] * kBounds[2];
    const double rx = cth * (ex - cx) + sth * (ez - cz) + cx;
    const double rz = -sth * (ex - cx) + cth * (ez - cz) + cz;
    glUniform3f(eyeAzLoc, (float)(rx / kBounds[0]), kEye[1], (float)(rz / kBounds[2]));
  };

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
  // WARMUP knob (issue_tax_repro R3): fresh complex programs keep re-JITting
  // far past a handful of draws — first timed block ran +32% hot with <10
  // warm-up frames. Default raised 5 -> 30; legacy runs: WARMUP=5.
  for (int f = 0; f < warmupN; f++) {
    AdvanceAzimuth();
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glFinish();
    if (surfMode) [viewCtx flushBuffer];
  }
  double t0 = NowMs();
  for (int f = 0; f < frames; f++) {
    AdvanceAzimuth();
    if (getenv("DEPTH")) glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    else glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glFinish();
    if (surfMode) [viewCtx flushBuffer];
  }
  double dt = NowMs() - t0;

  int nonzero = 0;
  {
    // Untimed complete draw so the readable buffer holds a full image
    // (SURFACE back buffers are swapped by flushBuffer above).
    glClear(getenv("DEPTH") ? (GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT) : GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glFinish();
    uint8_t* pix = (uint8_t*)malloc((size_t)rt * rt * 4);
    glReadPixels(0, 0, rt, rt, GL_RGBA, GL_UNSIGNED_BYTE, pix);
    for (size_t i = 0; i < (size_t)rt * rt * 4; i += 4)
      if (pix[i] | pix[i + 1] | pix[i + 2]) nonzero++;
    if (getenv("GL_DUMP_PPM"))
    {
      const char* outPath = getenv("RAYS")
        ? (atoi(getenv("RAYS")) == 2 ? "/tmp/harness_step.ppm" : "/tmp/harness_firsthit.ppm")
        : "/tmp/jgr.ppm";
      FILE* pp = fopen(outPath, "wb");
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
  viewCtx = nil;  // releases the view/window drawable
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
  // DOEXIT knob (V31 port): back-edge exit for the Metal march too.
  std::string msl2s = msl2;
  if (getenv("DOEXIT")) {
    const char* oldLoop =
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
      "  }\n";
    const char* newLoop =
      "  int i = 0;\n"
      "  if (i < min(u.maxIter, maxSteps)) {\n"
      "  do {\n"
      "    float currentT = tStart + float(i) * stepSize;\n"
      "    float3 evalPoint = evalBase + float(i) * evalStep;\n"
      "    float s = volTex.sample(volSampler, evalPoint).r;\n"
      "    float4 tfv = tfTex.sample(tfSampler, float2(s, 0.5f));\n"
      "    float w = 1.0f - accOp;\n"
      "    accColor += w * (tfv.rgb * tfv.a);\n"
      "    accOp += w * tfv.a;\n"
      "    iterCount = i + 1;\n"
      "    ++i;\n"
      "  } while (i < min(u.maxIter, maxSteps)\n"
      "        && tStart + float(i) * stepSize < tExit - 1e-6f\n"
      "        && accOp <= 0.996f);\n"
      "  }\n";
    size_t pos = msl2s.find(oldLoop);
    if (pos == std::string::npos) { fprintf(stderr, "DOEXIT: Metal loop pattern not found\n"); exit(1); }
    msl2s.replace(pos, strlen(oldLoop), newLoop);
  }
  NSError* err = nil;
  id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithFormat:@"%s%s", msl, msl2s.c_str()] options:nil error:&err];
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

    // Interleaved order-alternated A/B (the doc's standing protocol): each
    // round runs the full matrix; odd rounds run the exact reverse order so
    // warm-in/thermal bias cancels across rounds instead of loading the first
    // cell. ROUNDS=1 reproduces the historical single pass.
    int j1mode = 1;
    if (argc > 5 && strcmp(argv[5], "constphase") == 0) j1mode = 2;
    if (argc > 5 && strcmp(argv[5], "fetchonly") == 0) j1mode = 3;
    if (argc > 5 && strcmp(argv[5], "lattice") == 0) j1mode = 4;
    int rounds = getenv("ROUNDS") ? atoi(getenv("ROUNDS")) : 1;
    if (rounds < 1) rounds = 1;
    if (rounds > 8) rounds = 8;
    // GAPMS=n: idle before EVERY timed block. The app protocol measures
    // cool-start bursts (fresh process per jitter value; seconds of startup
    // idle between them) while the old harness marched blocks back-to-back,
    // soaking the GPU — a duty-cycle difference, not a shader/state one.
    const int gapMs = getenv("GAPMS") ? atoi(getenv("GAPMS")) : 0;
    // SELECT=g0,g1,m0,m1: run only the listed cells (default all). ONLYGL=1
    // drops the Metal cells so each invocation mirrors one app bench run.
    bool sel[4] = { true, true, true, true };
    if (const char* s = getenv("SELECT"))
    {
      sel[0] = sel[1] = sel[2] = sel[3] = false;
      if (strstr(s, "g0")) sel[0] = true;
      if (strstr(s, "g1")) sel[1] = true;
      if (strstr(s, "m0")) sel[2] = true;
      if (strstr(s, "m1")) sel[3] = true;
    }
    if (getenv("ONLYGL")) { sel[2] = sel[3] = false; }
    fprintf(stderr, "[proto] rounds=%d frames/run=%d gapMs=%d\n", rounds, frames, gapMs);
    double G0[8] = {0}, G1[8] = {0}, M0[8] = {0}, M1[8] = {0};
    for (int r = 0; r < rounds; r++)
    {
      const bool flip = (r & 1) != 0;
      if (!flip)
      {
        if (sel[0]) { if (gapMs) usleep(gapMs * 1000); G0[r] = RunGL(rt, sd, frames, vol, tf, 0); }
        if (sel[1]) { if (gapMs) usleep(gapMs * 1000); G1[r] = RunGL(rt, sd, frames, vol, tf, j1mode); }
        if (sel[2]) { if (gapMs) usleep(gapMs * 1000); M0[r] = RunMetal(rt, sd, frames, vol, tf, 0); }
        if (sel[3]) { if (gapMs) usleep(gapMs * 1000); M1[r] = RunMetal(rt, sd, frames, vol, tf, j1mode); }
      }
      else
      {
        if (sel[3]) { if (gapMs) usleep(gapMs * 1000); M1[r] = RunMetal(rt, sd, frames, vol, tf, j1mode); }
        if (sel[2]) { if (gapMs) usleep(gapMs * 1000); M0[r] = RunMetal(rt, sd, frames, vol, tf, 0); }
        if (sel[1]) { if (gapMs) usleep(gapMs * 1000); G1[r] = RunGL(rt, sd, frames, vol, tf, j1mode); }
        if (sel[0]) { if (gapMs) usleep(gapMs * 1000); G0[r] = RunGL(rt, sd, frames, vol, tf, 0); }
      }
      fprintf(stderr, "[round %d] GL dJit %+.3f ms | METAL dJit %+.3f ms\n",
              r, G1[r] - G0[r], M1[r] - M0[r]);
    }
    auto meanOf = [](const double* v, int n) { double s = 0; for (int i = 0; i < n; i++) s += v[i]; return s / n; };
    auto sdOf = [](const double* v, int n) {
      if (n < 2) return 0.0;
      double m = 0; for (int i = 0; i < n; i++) m += v[i]; m /= n;
      double s = 0; for (int i = 0; i < n; i++) s += (v[i] - m) * (v[i] - m);
      return sqrt(s / (n - 1));
    };
    double g0 = meanOf(G0, rounds), g1 = meanOf(G1, rounds);
    double m0 = meanOf(M0, rounds), m1 = meanOf(M1, rounds);
    if (!sel[0]) g0 = 0;
    if (!sel[1]) g1 = 0;
    if (!sel[2]) m0 = 0;
    if (!sel[3]) m1 = 0;
    double gd[8], md[8];
    for (int r = 0; r < rounds; r++) { gd[r] = G1[r] - G0[r]; md[r] = M1[r] - M0[r]; }

    if (sel[0] && sel[1])
      fprintf(stderr, "\nGL    j0 %8.3f+-%.3f   j1 %8.3f+-%.3f   jitter %+6.1f%%\n",
              g0, sdOf(G0, rounds), g1, sdOf(G1, rounds), (g1 / g0 - 1.0) * 100.0);
    if (sel[2] && sel[3])
      fprintf(stderr, "METAL j0 %8.3f+-%.3f   j1 %8.3f+-%.3f   jitter %+6.1f%%\n",
              m0, sdOf(M0, rounds), m1, sdOf(M1, rounds), (m1 / m0 - 1.0) * 100.0);
    if (g0 > 0 && g1 > 0 && m0 > 0 && m1 > 0)
      fprintf(stderr, "M/GL  j0 %8.2f   j1 %8.2f\n", m0 / g0, m1 / g1);
    if (sel[0] && sel[1])
      fprintf(stderr, "dJit  GL %+.2f+-%.2f ms%s",
              g1 - g0, sdOf(gd, rounds), sel[2] && sel[3] ? "   " : "\n");
    if (sel[2] && sel[3])
      fprintf(stderr, "METAL %+.2f+-%.2f ms\n", m1 - m0, sdOf(md, rounds));
    if (!(g0 > 0 && g1 > 0 && m0 > 0 && m1 > 0)) { free(vol); return 0; }
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