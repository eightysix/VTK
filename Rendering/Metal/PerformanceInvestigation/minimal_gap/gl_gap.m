// Minimal, self-contained OpenGL microbenchmark — the exact counterpart of
// metal_gap.m. Same synthetic 512x512x1794 R8 3D texture, same real-scene
// divergent-ray trilinear march (identical rays via inverseVP), same 400x400
// fragments (rtSize overridable), measured with glFinish + wall clock. Exposes
// the intrinsic Metal-vs-GL 3D sampler gap. Pass rtSize=1024/2048 +
// sampleDistMM=2.0/4.0 to reproduce the coarse-SD high-res "lag" case.
//
// Build: clang -framework AppKit -framework OpenGL gl_gap.m -o gl_gap
// Run:   ./gl_gap [frames] [iterations]

#import <AppKit/AppKit.h>
#import <OpenGL/OpenGL.h>
#import <OpenGL/gl3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static int kW = 512;
static int kH = 512;
static int kD = 1794;

static const char* kVertSrc =
  "#version 150\n"
  "in vec2 aPos;\n"
  "out vec2 vUV;\n"
  "void main() {\n"
  "  gl_Position = vec4(aPos, 0.0, 1.0);\n"
  "  vUV = aPos * 0.5 + 0.5;\n"
  "}\n";

static const char* kFragSrc =
  "#version 150\n"
  "in vec2 vUV;\n"
  "out vec4 fragColor;\n"
  "uniform sampler3D uVol;\n"
  "uniform vec3 uTexelCount;\n"
  "uniform float uSlabStart;\n"
  "uniform float uSlabEnd;\n"
  "uniform bool uSlabT;\n"
  "uniform vec3 uEye;\n"          // camera pos, normalized volume space
  "uniform vec3 uBoundsSize;\n"   // physical volume size (mm)
  "uniform mat4 uInvVP;\n"        // NDC -> physical volume coords
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
  "  float tlo = tStart;\n"
  "  float thi = tExit;\n"
  "  if (uSlabT) {\n"
  "    float t_s = (uSlabStart - uEye.z) / rayDir.z;\n"
  "    float t_e = (uSlabEnd - uEye.z) / rayDir.z;\n"
  "    tlo = max(tStart, min(t_s, t_e));\n"
  "    thi = min(tExit, max(t_s, t_e));\n"
  "  }\n"
  "  float physPerNorm = length(rayDir * uBoundsSize);\n"
  "  float stepSize = uSampleDistMM / max(physPerNorm, 1e-6);\n"
  "  if (uSlabT) {\n"
  "    tStart = tStart + ceil(max((tlo - tStart) / stepSize, 0.0)) * stepSize;\n"
  "    if (uSlabEnd < 1.0) tExit = tStart + ceil(max((thi - tStart) / stepSize, 0.0)) * stepSize;\n"
  "    else tExit = thi;\n"
  "  }\n"
  "  int maxSteps = max(0, int(ceil((tExit - tStart) / stepSize)));\n"
  "  vec3 texelCount = uTexelCount;\n"
  "  vec3 ctpScale = max(texelCount - 1.0, 1e-4) / texelCount;\n"
  "  vec3 ctpOffset = 0.5 / texelCount;\n"
  "  vec3 texStep = rayDir * stepSize;\n"
  "  vec3 evalStep = texStep * ctpScale;\n"
  "  float currentT = tStart;\n"
  "  vec3 texLocal = uEye + rayDir * currentT;\n"
  "  vec3 evalPoint = texLocal * ctpScale + ctpOffset;\n"
  "  evalPoint.z = clamp(evalPoint.z, uSlabStart * ctpScale.z + ctpOffset.z, uSlabEnd * ctpScale.z + ctpOffset.z);\n"
  "  float acc = 0.0;\n"
  "  float n = 0.0;\n"
  "  for (int i = 0; i < min(uMaxIter, maxSteps); i++) {\n"
  "    if (currentT >= tExit - 1e-6) break;\n"
  "    acc = max(acc, texture(uVol, evalPoint).r);\n"
  "    currentT += stepSize;\n"
  "    texLocal += texStep;\n"
  "    evalPoint += evalStep;\n"
  "    n += 1.0;\n"
  "  }\n"
  "  int nc = int(n);\n"
  "  fragColor = vec4(float(nc & 255) / 255.0, float((nc >> 8) & 255) / 255.0, acc, 1.0);\n"
  "}\n";

static const char* kFragMulAddSrc =
  "#version 150\n"
  "in vec2 vUV;\n"
  "out vec4 fragColor;\n"
  "uniform sampler3D uVol;\n"
  "uniform vec3 uTexelCount;\n"
  "uniform float uSlabStart;\n"
  "uniform float uSlabEnd;\n"
  "uniform bool uSlabT;\n"
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
  "  float tExitRaw = min(min(tmax3.x, tmax3.y), tmax3.z);\n"
  "  if (tExitRaw <= 0.0 || tEnter >= tExitRaw) { fragColor = vec4(0.0,0.0,0.0,1.0); return; }\n"
  "  float tStartRaw = max(tEnter, 0.0);\n"
  "  float tlo = tStartRaw;\n"
  "  float thi = tExitRaw;\n"
  "  if (uSlabT) {\n"
  "    float t_s = (uSlabStart - uEye.z) * inv.z;\n"
  "    float t_e = (uSlabEnd - uEye.z) * inv.z;\n"
  "    tlo = max(tStartRaw, min(t_s, t_e));\n"
  "    thi = min(tExitRaw, max(t_s, t_e));\n"
  "  }\n"
  "  float physPerNorm = length(rayDir * uBoundsSize);\n"
  "  float stepSize = uSampleDistMM / max(physPerNorm, 1e-6);\n"
  "  int kPass = int(ceil(max((tlo - tStartRaw) / stepSize, 0.0)));\n"
  "  float tStart = tStartRaw + float(kPass) * stepSize;\n"
  "  float tExit = tExitRaw;\n"
  "  if (uSlabT) tExit = tStartRaw + float(int(ceil(max((thi - tStartRaw) / stepSize, 0.0)))) * stepSize;\n"
  "  int maxSteps = max(0, int(ceil((tExit - tStart) / stepSize)));\n"
  "  vec3 texelCount = uTexelCount;\n"
  "  vec3 ctpScale = max(texelCount - 1.0, 1e-4) / texelCount;\n"
  "  vec3 ctpOffset = 0.5 / texelCount;\n"
  "  vec3 evalBase = ctpOffset + (uEye + rayDir * tStartRaw) * ctpScale;\n"
  "  vec3 evalStep = rayDir * ctpScale * stepSize;\n"
  "  float acc = 0.0;\n"
  "  float n = 0.0;\n"
  "  for (int i = 0; i < min(uMaxIter, maxSteps); i++) {\n"
  "    float currentT = tStartRaw + float(kPass + i) * stepSize;\n"
  "    if (currentT >= min(tExit, tExitRaw) - 1e-6) break;\n"
  "    vec3 evalPoint = evalBase + float(kPass + i) * evalStep;\n"
  "    acc = max(acc, texture(uVol, evalPoint).r);\n"
  "    n += 1.0;\n"
  "  }\n"
  "  int nc = int(n);\n"
  "  fragColor = vec4(float(nc & 255) / 255.0, float((nc >> 8) & 255) / 255.0, acc, 1.0);\n"
  "}\n";

static const char* kFragNoFetchSrc =
  "#version 150\n"
  "out vec4 fragColor;\n"
  "in vec2 vUV;\n"
  "uniform float uSlabStart;\n"
  "uniform float uSlabEnd;\n"
  "uniform bool uSlabT;\n"
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
  "  float tlo = tStart;\n"
  "  float thi = tExit;\n"
  "  if (uSlabT) {\n"
  "    float t_s = (uSlabStart - uEye.z) / rayDir.z;\n"
  "    float t_e = (uSlabEnd - uEye.z) / rayDir.z;\n"
  "    tlo = max(tStart, min(t_s, t_e));\n"
  "    thi = min(tExit, max(t_s, t_e));\n"
  "  }\n"
  "  float physPerNorm = length(rayDir * uBoundsSize);\n"
  "  float stepSize = uSampleDistMM / max(physPerNorm, 1e-6);\n"
  "  if (uSlabT) {\n"
  "    tStart = tStart + ceil(max((tlo - tStart) / stepSize, 0.0)) * stepSize;\n"
  "    if (uSlabEnd < 1.0) tExit = tStart + ceil(max((thi - tStart) / stepSize, 0.0)) * stepSize;\n"
  "    else tExit = thi;\n"
  "  }\n"
  "  int maxSteps = max(0, int(ceil((tExit - tStart) / stepSize)));\n"
  "  vec3 texelCount = uTexelCount;\n"
  "  vec3 ctpScale = max(texelCount - 1.0, 1e-4) / texelCount;\n"
  "  vec3 ctpOffset = 0.5 / texelCount;\n"
  "  vec3 texStep = rayDir * stepSize;\n"
  "  vec3 evalStep = texStep * ctpScale;\n"
  "  float currentT = tStart;\n"
  "  vec3 texLocal = uEye + rayDir * currentT;\n"
  "  vec3 evalPoint = texLocal * ctpScale + ctpOffset;\n"
  "  evalPoint.z = clamp(evalPoint.z, uSlabStart * ctpScale.z + ctpOffset.z, uSlabEnd * ctpScale.z + ctpOffset.z);\n"
  "  float acc = 0.0;\n"
  "  float n = 0.0;\n"
  "  for (int i = 0; i < min(uMaxIter, maxSteps); i++) {\n"
  "    if (currentT >= tExit - 1e-6) break;\n"
  "    acc = max(acc, evalPoint.x * 0.001 + 0.5);\n"
  "    currentT += stepSize;\n"
  "    texLocal += texStep;\n"
  "    evalPoint += evalStep;\n"
  "    n += 1.0;\n"
  "  }\n"
  "  int nc = int(n);\n"
  "  fragColor = vec4(float(nc & 255) / 255.0, float((nc >> 8) & 255) / 255.0, acc, 1.0);\n"
  "}\n";

static double now_sec(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static GLuint CompileShader(GLenum type, const char* src, const char* name)
{
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, NULL);
  glCompileShader(s);
  GLint ok = 0;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[4096];
    glGetShaderInfoLog(s, sizeof(log), NULL, log);
    fprintf(stderr, "%s shader compile failed:\n%s\n", name, log);
    exit(1);
  }
  return s;
}

// Build: clang -framework AppKit -framework OpenGL gl_gap.m -o gl_gap
// Run:   ./gl_gap [frames] [maxIter] [noFetch] [useDepth] [fmt16] [flipY] [rtSize] [sampleDistMM]
//   flipY=1 negates NDC y. GL rasterizes with a bottom-left window origin and
//   Metal with a top-left origin, so the two gap harnesses otherwise trace
//   y-mirrored ray fields for the same readback row; since the DICOM scene is
//   not y-symmetric this changes the per-pixel step counts (avgIter). Set
//   flipY=1 on the GL side to trace the same rays row-for-row as metal_gap.
//   rtSize: render-target size in px (default 400). Coarse-SD high-res "lag":
//   pass 1024/2048 with sampleDistMM 2.0/4.0 to match the app's lag cases.
// Note: the harness reads back / processes rows in framebuffer order
// (GL bottom-first, Metal top-first).

int main(int argc, const char** argv)
{
  int frames = 100;
  int maxIter = 8192;
  int noFetch = 0;
  int useDepth = 0;
  int fmt16 = 0;
  int flipY = 0;
  int rtSize = 400;
  float sampleDistMM = 0.5f;
  int dataMode = 0;
  int filterMode = 0;
  int volDiv = 1;
  int numSlabs = 0;
  int slabIndex = 0;
  int slabT = 0;
  int maccum = 0;
  int mulAdd = 0;
  if (argc > 1) frames = atoi(argv[1]);
  if (argc > 2) maxIter = atoi(argv[2]);
  if (argc > 3) noFetch = atoi(argv[3]);
  if (argc > 4) useDepth = atoi(argv[4]);
  if (argc > 5) fmt16 = atoi(argv[5]);
  if (argc > 6) flipY = atoi(argv[6]);
  if (argc > 7) rtSize = atoi(argv[7]);
  if (argc > 8) sampleDistMM = (float)atof(argv[8]);
  if (argc > 9) dataMode = atoi(argv[9]);
  if (argc > 10) filterMode = atoi(argv[10]);
  if (argc > 11) { volDiv = atoi(argv[11]); if (volDiv < 1) volDiv = 1; kW = 512 / volDiv; kH = 512 / volDiv; kD = 1794 / volDiv; }
  if (argc > 12) numSlabs = atoi(argv[12]);
  if (argc > 13) slabIndex = atoi(argv[13]);
  if (argc > 14) slabT = atoi(argv[14]);
  if (argc > 15) maccum = atoi(argv[15]);
  if (argc > 16) mulAdd = atoi(argv[16]);

  NSOpenGLPixelFormatAttribute attrs[] = {
    NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion3_2Core,
    NSOpenGLPFAAccelerated,
    0
  };
  NSOpenGLPixelFormat* pf = [[NSOpenGLPixelFormat alloc] initWithAttributes:attrs];
  if (!pf) { fprintf(stderr, "no GL pixel format (3.2 core)\n"); return 1; }
  NSOpenGLContext* ctx = [[NSOpenGLContext alloc] initWithFormat:pf shareContext:nil];
  if (!ctx) { fprintf(stderr, "no GL context\n"); return 1; }
  [ctx makeCurrentContext];

  GLint max3d = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max3d);
  printf("GL_VERSION: %s\n", glGetString(GL_VERSION));
  printf("GL_MAX_3D_TEXTURE_SIZE: %d\n", (int)max3d);
  printf("volume %dx%dx%d R8, rt %dx%d, frames %d, maxIter %d, noFetch %d, flipY %d, sampleDistMM=%.1f, dataMode=%d, filterMode=%d, volDiv=%d, slab=%d/%d, slabT=%d, maccum=%d, mulAdd=%d\n",
    kW, kH, kD, rtSize, rtSize, frames, maxIter, noFetch, flipY, sampleDistMM, dataMode, filterMode, volDiv, slabIndex, numSlabs, slabT, maccum, mulAdd);

  // Program. flipY=1 negates NDC y so the same readback row traces the same
  // ray as metal_gap (which uses Metal's top-left window convention).
  const char* fragSrc0 = mulAdd ? kFragMulAddSrc : (noFetch ? kFragNoFetchSrc : kFragSrc);
  const char* fragSrc = fragSrc0;
  char* flipped = NULL;
  if (flipY) {
    const char* needle = "vec2 ndc = vUV * 2.0 - 1.0;";
    char* p = strstr((char*)fragSrc0, needle);
    if (!p) { fprintf(stderr, "flipY: ndc line not found\n"); return 1; }
    size_t off = (size_t)(p - fragSrc0);
    flipped = malloc(strlen(fragSrc0) + 32);
    memcpy(flipped, fragSrc0, off);
    strcpy(flipped + off, needle);
    strcat(flipped + off + strlen(needle), " ndc.y = -ndc.y;");
    strcat(flipped + off + strlen(needle), (fragSrc0 + off + strlen(needle)));
    fragSrc = flipped;
  }
  GLuint vs = CompileShader(GL_VERTEX_SHADER, kVertSrc, "vertex");
  GLuint fs = CompileShader(GL_FRAGMENT_SHADER, fragSrc, "fragment");
  if (flipped) free(flipped);
  GLuint prog = glCreateProgram();
  glAttachShader(prog, vs);
  glAttachShader(prog, fs);
  glBindAttribLocation(prog, 0, "aPos");
  glLinkProgram(prog);
  GLint ok = 0;
  glGetProgramiv(prog, GL_LINK_STATUS, &ok);
  if (!ok) { char log[4096]; glGetProgramInfoLog(prog, sizeof(log), NULL, log); fprintf(stderr, "link failed:\n%s\n", log); return 1; }
  glUseProgram(prog);
  GLint uVol = glGetUniformLocation(prog, "uVol");
  GLint uEye = glGetUniformLocation(prog, "uEye");
  GLint uBoundsSize = glGetUniformLocation(prog, "uBoundsSize");
  GLint uInvVP = glGetUniformLocation(prog, "uInvVP");
  GLint uSampleDistMM = glGetUniformLocation(prog, "uSampleDistMM");
  GLint uMaxIter = glGetUniformLocation(prog, "uMaxIter");
  GLint uTexelCount = glGetUniformLocation(prog, "uTexelCount");
  GLint uSlabStart = glGetUniformLocation(prog, "uSlabStart");
  GLint uSlabEnd = glGetUniformLocation(prog, "uSlabEnd");
  GLint uSlabT = glGetUniformLocation(prog, "uSlabT");

  // Fullscreen triangle. NOTE: 2 floats per vertex; a stray 3rd component
  // shifts the stride and breaks the triangle into a thin sliver.
  GLfloat verts[6] = { -1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f };
  GLuint vao, vbo;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);

  // Volume texture.
  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_3D, tex);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, filterMode ? GL_NEAREST : GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, filterMode ? GL_NEAREST : GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  size_t total = (size_t)kW * kH * kD;
  if (fmt16) {
    short* h16 = malloc(total * 2);
    for (size_t i = 0; i < total; i++) h16[i] = (short)(((i >> 10) & 0x1fff) - 0x1000);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage3D(GL_TEXTURE_3D, 0, 0x8F98, kW, kH, kD, 0, GL_RED, GL_SHORT, h16);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    free(h16);
  } else {
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
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, kW, kH, kD, 0, GL_RED, GL_UNSIGNED_BYTE, host);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    free(host);
  }
  GLint ifmt0 = 0;
  glGetTexLevelParameteriv(GL_TEXTURE_3D, 0, GL_TEXTURE_INTERNAL_FORMAT, &ifmt0);
  fprintf(stderr, "volume internal format = 0x%x\n", ifmt0);

  // Uniforms (must match metal_gap: real app camera, physical bounds, invVP).
  glUniform1i(uVol, 0);
  glUniform3f(uEye, -1.49527037f, -0.95243806f, 2.55352926f);
  glUniform3f(uBoundsSize, 426.166f, 426.166f, 717.2f);
  const GLfloat invVP[16] = {
    0.23205078f, -6.974323e-09f, 0.13397458f, 1.2084004e-11f,
    -0.045822006f, 0.25178984f, 0.079366043f, -3.2157374e-12f,
    0.35811684f, 0.22810864f, -1.0292176f, -0.00056198676f,
    -0.12124427f, -0.03448515f, 0.8849802f, 0.00092758867f };
  glUniformMatrix4fv(uInvVP, 1, GL_FALSE, invVP);
  glUniform1f(uSampleDistMM, sampleDistMM);
  glUniform3f(uTexelCount, (float)kW, (float)kH, (float)kD);
  glUniform1f(uSlabStart, numSlabs > 0 ? (float)slabIndex / numSlabs : 0.0f);
  glUniform1f(uSlabEnd, numSlabs > 0 ? (float)(slabIndex + 1) / numSlabs : 1.0f);
  glUniform1i(uSlabT, slabT ? 1 : 0);
  glUniform1i(uMaxIter, maxIter);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_3D, tex);

  glViewport(0, 0, rtSize, rtSize);
  glClearColor(0, 0, 0, 1);

  // Offscreen render target (windowless context has no default drawable).
  GLuint fbo, rbo;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glGenRenderbuffers(1, &rbo);
  glBindRenderbuffer(GL_RENDERBUFFER, rbo);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, rtSize, rtSize);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);
  GLuint depthRbo = 0;
  if (useDepth) {
    glGenRenderbuffers(1, &depthRbo);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, rtSize, rtSize);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRbo);
  }
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    fprintf(stderr, "FBO incomplete\n");
    return 1;
  }

  // Warmup.
  if (maccum) {
    glEnable(GL_BLEND);
    glBlendEquation(GL_MAX);
    glBlendFunc(GL_ONE, GL_ONE);
  }
  for (int f = 0; f < 10; f++) {
    if (maccum) {
      glClear(GL_COLOR_BUFFER_BIT);
      for (int p = 0; p < numSlabs; p++) {
        glUniform1f(uSlabStart, (float)p / numSlabs);
        glUniform1f(uSlabEnd, (float)(p + 1) / numSlabs);
        glDrawArrays(GL_TRIANGLES, 0, 3);
      }
    } else {
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    glFinish();
  }

  // Timed.
  double acc = 0.0;
  for (int f = 0; f < frames; f++) {
    double t0 = now_sec();
    if (maccum) {
      glClear(GL_COLOR_BUFFER_BIT);
      for (int p = 0; p < numSlabs; p++) {
        glUniform1f(uSlabStart, (float)p / numSlabs);
        glUniform1f(uSlabEnd, (float)(p + 1) / numSlabs);
        glDrawArrays(GL_TRIANGLES, 0, 3);
      }
    } else {
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    glFinish();
    double t1 = now_sec();
    acc += t1 - t0;
  }
  fprintf(stderr, "GL avg frame: %.3f ms\n", acc / frames * 1e3);

  // Sanity readback: R/G carry the iteration count (low/high byte), B the
  // sample value; nonzero pixels are marched fragments.
  uint8_t* pix = malloc((size_t)rtSize * rtSize * 4);
  glReadPixels(0, 0, rtSize, rtSize, GL_RGBA, GL_UNSIGNED_BYTE, pix);
  double sumB = 0.0;
  double lo = 0.0;
  double hi = 0.0;
  int nz = 0;
  for (size_t i = 0; i < (size_t)rtSize * rtSize * 4; i += 4) {
    sumB += pix[i + 2];
    lo += pix[i];
    hi += pix[i + 1];
    if (pix[i] + pix[i + 1] + pix[i + 2] > 0) nz++;
  }
  double npix = (double)rtSize * rtSize;
  double avgN = (hi / npix) * 256.0 + lo / npix;
  fprintf(stderr, "GL readback: meanB=%.3f nonzero=%d/%d avgIter=%.1f\n",
    sumB / npix / 255.0, nz, rtSize * rtSize, avgN);
  FILE* fp = fopen("gl_gap.ppm", "wb");
  if (fp) {
    fprintf(fp, "P6\n%d %d\n255\n", rtSize, rtSize);
    for (int i = 0; i < rtSize * rtSize; i++)
      fwrite(pix + i * 4, 1, 3, fp);
    fclose(fp);
  }
  free(pix);

  [ctx clearDrawable];
  return 0;
}
