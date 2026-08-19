// Reproduces the app's composed GL volume raycast shaders in a standalone
// harness, with the exact uniforms dumped by the app and an R8 volume texture
// matching the app's actual 8-bit texture (verified: requested/realized
// GL_R8, Rbits=8). Used to explain why the full app GL render (~49 ms) is
// faster per sample than the bare-fetch microbenchmark (~62 ms) despite doing
// strictly more work per sample (TF lookups, compositing).
// Build: clang -framework AppKit -framework OpenGL gl_app_shader.m -o gl_app_shader
// Usage: gl_app_shader [frames] [iterMode]

#include <AppKit/AppKit.h>
#include <OpenGL/gl3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define kRT 400
#define KW 512
#define KH 512
#define KD 1794
#define R16_SNORM 0x8F98

static char* ReadFile(const char* path, size_t* outLen)
{
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char* buf = malloc(sz + 1);
  fread(buf, 1, sz, f);
  buf[sz] = 0;
  fclose(f);
  if (outLen) *outLen = (size_t)sz;
  return buf;
}

static const char* kVertSrc = NULL;
static char* kFragSrc = NULL;

static GLuint CompileShader(GLenum type, const char* src, const char* name)
{
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, NULL);
  glCompileShader(s);
  GLint ok = 0;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[8192];
    glGetShaderInfoLog(s, sizeof(log), NULL, log);
    fprintf(stderr, "%s shader compile failed:\n%s\n", name, log);
    exit(1);
  }
  return s;
}

static double now_sec(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, const char** argv)
{
  int frames = 60;
  int iterMode = 0;
  int rt = kRT;
  float sd = 0.5f;
  int dataMode = 0;
  int noiseMode = 0;
  if (argc > 1) frames = atoi(argv[1]);
  if (argc > 2) iterMode = atoi(argv[2]);
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i - 1], "--rt") == 0) rt = atoi(argv[i]);
    if (strcmp(argv[i - 1], "--sd") == 0) sd = (float)atof(argv[i]);
    if (strcmp(argv[i - 1], "--data") == 0) dataMode = atoi(argv[i]);
    if (strcmp(argv[i - 1], "--noise") == 0) noiseMode = atoi(argv[i]);
  }

  // Load the app's exact composed shaders.
  size_t vlen = 0, flen = 0;
  char* vbase = ReadFile("/tmp/app_gl_vert.glsl", &vlen);
  char* fbase = ReadFile("/tmp/app_gl_frag.glsl", &flen);
  if (!vbase || !fbase) { fprintf(stderr, "missing /tmp/app_gl_*.glsl\n"); return 1; }

  // Vertex: prepend GLSL 150.
  size_t voutlen = strlen("#version 150\n") + vlen + 1;
  char* vsrc = malloc(voutlen);
  snprintf(vsrc, voutlen, "#version 150\n%s", vbase);
  free(vbase);

  // Fragment: prepend GLSL 150 + output declaration + VTK's texture2D/3D
  // shims, replace gl_FragData[0].
  size_t foutlen = strlen(
    "#version 150\n"
    "#define texture2D texture\n"
    "#define texture3D texture\n"
    "out vec4 fragColorOut0;\n") + flen + 1;
  char* fsrc = malloc(foutlen);
  snprintf(fsrc, foutlen,
    "#version 150\n"
    "#define texture2D texture\n"
    "#define texture3D texture\n"
    "out vec4 fragColorOut0;\n%s", fbase);
  free(fbase);
  char* hit = strstr(fsrc, "gl_FragData[0]");
  if (hit) {
    memcpy(hit, "fragColorOut0 ", 14);
  }
  if (iterMode) {
    const char* old = "  fragColorOut0  = g_fragColor;";
    const char* rep = "  fragColorOut0 = vec4(g_currentT/4096.0, g_terminatePointMax/2048.0, 0.0, 1.0);";
    char* p = strstr(fsrc, old);
    if (p) {
      size_t off = (size_t)(p - fsrc);
      char* nsrc = malloc(strlen(fsrc) - strlen(old) + strlen(rep) + 1);
      memcpy(nsrc, fsrc, off);
      memcpy(nsrc + off, rep, strlen(rep));
      strcpy(nsrc + off + strlen(rep), p + strlen(old));
      free(fsrc);
      fsrc = nsrc;
    } else {
      fprintf(stderr, "iterMode: finalize pattern not found\n");
    }
  }
  FILE* dbg = fopen("/tmp/debug_frag.glsl", "w");
  if (dbg) { fwrite(fsrc, 1, strlen(fsrc), dbg); fclose(dbg); }

  NSOpenGLPixelFormatAttribute attrs[] = {
    NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion3_2Core,
    NSOpenGLPFAAccelerated,
    0
  };
  NSOpenGLPixelFormat* pf = [[NSOpenGLPixelFormat alloc] initWithAttributes:attrs];
  if (!pf) { fprintf(stderr, "no GL pixel format\n"); return 1; }
  NSOpenGLContext* ctx = [[NSOpenGLContext alloc] initWithFormat:pf shareContext:nil];
  if (!ctx) { fprintf(stderr, "no GL context\n"); return 1; }
  [ctx makeCurrentContext];

  printf("GL_VERSION: %s\n", glGetString(GL_VERSION));

  // Program.
  GLuint vs = CompileShader(GL_VERTEX_SHADER, vsrc, "vertex");
  GLuint fs = CompileShader(GL_FRAGMENT_SHADER, fsrc, "fragment");
  GLuint prog = glCreateProgram();
  glAttachShader(prog, vs);
  glAttachShader(prog, fs);
  glBindAttribLocation(prog, 0, "in_vertexPos");
  glLinkProgram(prog);
  GLint ok = 0;
  glGetProgramiv(prog, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[8192];
    glGetProgramInfoLog(prog, sizeof(log), NULL, log);
    fprintf(stderr, "link failed:\n%s\n", log);
    return 1;
  }
  glUseProgram(prog);

  // Box geometry in physical coords (matches the app volume bounds).
  const float X = 426.166f, Y = 426.166f, Z = 717.2f;
  const float box[24] = {
    0,0,0,  X,0,0,  X,Y,0,  0,Y,0,
    0,0,Z,  X,0,Z,  X,Y,Z,  0,Y,Z
  };
  const GLushort idx[36] = {
    0,1,2, 0,2,3, 4,6,5, 4,7,6,
    0,4,5, 0,5,1, 1,5,6, 1,6,2,
    3,2,6, 3,6,7, 0,3,7, 0,7,4
  };
  GLuint vao, vbo, ibo;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(box), box, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
  glGenBuffers(1, &ibo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

  // ---- Textures (units as in the app) ----
  // Unit 0: in_depthSampler = 1x1 white (so gl_FragCoord.z >= 1 is never true).
  unsigned char white[4] = { 255, 255, 255, 255 };
  GLuint depthTex;
  glGenTextures(1, &depthTex);
  glBindTexture(GL_TEXTURE_2D, depthTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, white);

  // Unit 1: in_noiseSampler. 0 = 128x128 R8 ramp (LINEAR), 1 = app's exact
  // 64x64 R32F blue-noise tile (NEAREST), 2 = same tile (LINEAR).
  GLuint noiseTex;
  glGenTextures(1, &noiseTex);
  glBindTexture(GL_TEXTURE_2D, noiseTex);
  if (noiseMode) {
    FILE* nf = fopen("/tmp/bluenoise64.bin", "rb");
    if (!nf) { fprintf(stderr, "missing /tmp/bluenoise64.bin\n"); return 1; }
    float* nf32 = malloc(64 * 64 * sizeof(float));
    size_t nrd = fread(nf32, sizeof(float), 64 * 64, nf);
    fclose(nf);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, noiseMode == 1 ? GL_NEAREST : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, noiseMode == 1 ? GL_NEAREST : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 64, 64, 0, GL_RED, GL_FLOAT, nf32);
    free(nf32);
  } else {
    unsigned char* noise = malloc(128 * 128);
    for (int i = 0; i < 128 * 128; i++) noise[i] = (unsigned char)(i * 13 % 256);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, 128, 128, 0, GL_RED, GL_UNSIGNED_BYTE, noise);
    free(noise);
  }

  // Unit 2: volume R8 512x512x1794 (matches the app's 8-bit texture).
  size_t vtotal = (size_t)KW * KH * KD;
  unsigned char* vol = malloc(vtotal);
  if (dataMode) {
    uint32_t x = 0x12345678u;
    for (size_t i = 0; i < vtotal; i++) {
      x ^= x << 13; x ^= x >> 17; x ^= x << 5;
      vol[i] = (uint8_t)(x >> 24);
    }
  } else {
    for (size_t i = 0; i < vtotal; i++) {
      vol[i] = (unsigned char)((i >> 10) & 0xff);
    }
  }
  GLuint volTex;
  glGenTextures(1, &volTex);
  glBindTexture(GL_TEXTURE_3D, volTex);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, KW, KH, KD, 0, GL_RED, GL_UNSIGNED_BYTE, vol);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
  free(vol);

  // Units 3/4: opacity/color transfer functions 256x2 RGBA8.
  // Red channel (read by computeOpacity) is kept tiny so the accumulated
  // opacity never saturates and the loop runs all ~659 iterations, like the
  // app (which uses a low-opacity TF over mostly-air data).
  unsigned char tf[256 * 2 * 4];
  for (int i = 0; i < 256 * 2; i++) {
    tf[i * 4 + 0] = 2;
    tf[i * 4 + 1] = (unsigned char)(255 - i);
    tf[i * 4 + 2] = 128;
    tf[i * 4 + 3] = 255;
  }
  GLuint opacTex, colorTex;
  glGenTextures(1, &opacTex);
  glBindTexture(GL_TEXTURE_2D, opacTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, tf);
  glGenTextures(1, &colorTex);
  glBindTexture(GL_TEXTURE_2D, colorTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, tf);

  // ---- Uniforms (exact values from the app dump, column-major buffers) ----
  glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, depthTex);
  glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, noiseTex);
  glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_3D, volTex);
  glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, opacTex);
  glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, colorTex);

  glUniform1i(glGetUniformLocation(prog, "in_depthSampler"), 0);
  glUniform1i(glGetUniformLocation(prog, "in_noiseSampler"), 1);
  glUniform1i(glGetUniformLocation(prog, "in_volume[0]"), 2);
  glUniform1i(glGetUniformLocation(prog, "in_opacityTransferFunc_0[0]"), 3);
  glUniform1i(glGetUniformLocation(prog, "in_colorTransferFunc_0[0]"), 4);
  glUniform1i(glGetUniformLocation(prog, "in_noOfComponents"), 1);

  const float ident[16] = {
    1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1
  };
  const float invTexDataset[16] = {
    0.0023465f,0,0,0, 0,0.0023465f,0,0, 0,0,0.00139431f,0, 0,0,0,1
  };
  const float cellToPoint[16] = {
    0.998047f,0,0,0, 0,0.998047f,0,0, 0,0,0.999443f,0, 0.000976562f,0.000976562f,0.000278707f,1
  };
  const float modelView[16] = {
    0.866025f,-0.17101f,-0.469846f,0, 0,0.939693f,-0.34202f,0, 0.5f,0.296198f,0.813798f,0, -363.835f,-270.01f,-1928.61f,1
  };
  const float invModelView[16] = {
    0.866025f,-3.03887e-18f,0.5f,-0, -0.17101f,0.939693f,0.296198f,0, -0.469846f,-0.34202f,0.813798f,0, -637.233f,-405.897f,1831.39f,1
  };
  const float proj[16] = {
    3.73205f,0,0,0, 0,3.73205f,0,0, 0,0,-2.30111f,-1, 0,0,-3558.8f,0
  };
  const float invProj[16] = {
    0.267949f,0,0,0, 0,0.267949f,0,0, 0,0,0,-0.000280993f, 0,0,-1,0.000646595f
  };

  glUniformMatrix4fv(glGetUniformLocation(prog, "in_volumeMatrix[0]"), 1, GL_FALSE, ident);
  glUniformMatrix4fv(glGetUniformLocation(prog, "in_inverseVolumeMatrix[0]"), 1, GL_FALSE, ident);
  glUniformMatrix4fv(glGetUniformLocation(prog, "in_inverseTextureDatasetMatrix[0]"), 1, GL_FALSE, invTexDataset);
  glUniformMatrix4fv(glGetUniformLocation(prog, "in_cellToPoint[0]"), 1, GL_FALSE, cellToPoint);
  glUniformMatrix4fv(glGetUniformLocation(prog, "in_modelViewMatrix"), 1, GL_FALSE, modelView);
  glUniformMatrix4fv(glGetUniformLocation(prog, "in_inverseModelViewMatrix"), 1, GL_FALSE, invModelView);
  glUniformMatrix4fv(glGetUniformLocation(prog, "in_projectionMatrix"), 1, GL_FALSE, proj);
  glUniformMatrix4fv(glGetUniformLocation(prog, "in_inverseProjectionMatrix"), 1, GL_FALSE, invProj);

  glUniform3f(glGetUniformLocation(prog, "in_texMin[0]"), 0.000976562f, 0.000976562f, 0.000278707f);
  glUniform3f(glGetUniformLocation(prog, "in_texMax[0]"), 0.999023f, 0.999023f, 0.999721f);
  glUniform3f(glGetUniformLocation(prog, "in_eyePosObjs[0]"), -637.233f, -405.897f, 1831.39f);
  glUniform3f(glGetUniformLocation(prog, "in_cellSpacing[0]"), 0.833984f, 0.833984f, 0.4f);
  glUniform4f(glGetUniformLocation(prog, "in_volume_scale[0]"), 1.00392f, 1.0f, 1.0f, 1.0f);
  glUniform4f(glGetUniformLocation(prog, "in_volume_bias[0]"), 0.0f, 0.0f, 0.0f, 0.0f);
  glUniform2f(glGetUniformLocation(prog, "in_windowLowerLeftCorner"), 0.0f, 0.0f);
  glUniform2f(glGetUniformLocation(prog, "in_inverseWindowSize"), 1.0f / rt, 1.0f / rt);
  glUniform2f(glGetUniformLocation(prog, "in_inverseOriginalWindowSize"), 1.0f / rt, 1.0f / rt);
  glUniform1f(glGetUniformLocation(prog, "in_sampleDistance"), sd);
  glUniform1f(glGetUniformLocation(prog, "in_scale"), 1.0f);
  glUniform1f(glGetUniformLocation(prog, "in_bias"), 0.0f);
  float clip[49] = { 0 };
  clip[0] = 6.0f;
  glUniform1fv(glGetUniformLocation(prog, "in_clippingPlanes[0]"), 49, clip);

  // Offscreen render target.
  glViewport(0, 0, rt, rt);
  glClearColor(0, 0, 0, 1);
  GLuint fbo, rbo;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glGenRenderbuffers(1, &rbo);
  glBindRenderbuffer(GL_RENDERBUFFER, rbo);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, rt, rt);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    fprintf(stderr, "FBO incomplete\n");
    return 1;
  }

  // Warmup.
  for (int f = 0; f < 10; f++) {
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, 0);
    glFinish();
  }

  // Timed.
  double acc = 0.0;
  for (int f = 0; f < frames; f++) {
    double t0 = now_sec();
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, 0);
    glFinish();
    double t1 = now_sec();
    acc += t1 - t0;
  }
  fprintf(stderr, "GL app-shader avg frame: %.3f ms (frames=%d)\n", acc / frames * 1e3, frames);

  // Sanity readback: nonzero fraction + average sample value (or iteration
  // count when iterMode).
  unsigned char* pix = malloc((size_t)rt * rt * 4);
  glReadPixels(0, 0, rt, rt, GL_RGBA, GL_UNSIGNED_BYTE, pix);
  int nz = 0;
  double sum = 0;
  double lo = 0, hi = 0;
  for (size_t i = 0; i < (size_t)rt * rt * 4; i += 4) {
    if (pix[i] + pix[i + 1] + pix[i + 2] > 0) nz++;
    sum += pix[i + 2];
    lo += pix[i];
    hi += pix[i + 1];
  }
  if (iterMode) {
    double npix = (double)rt * rt;
    double avgIter = (lo / npix) * 4096.0 / 255.0;
    double avgTerm = (hi / npix) * 2048.0 / 255.0;
    fprintf(stderr, "readback: nonzero=%d/%d avgIter=%.1f avgTerm=%.1f\n", nz, rt * rt, avgIter, avgTerm);
    FILE* fp = fopen("gl_app_shader_iter.ppm", "wb");
    if (fp) {
      fprintf(fp, "P6\n%d %d\n255\n", rt, rt);
      for (int i = 0; i < rt * rt; i++)
        fwrite(pix + i * 4, 1, 3, fp);
      fclose(fp);
    }
  } else {
    fprintf(stderr, "readback: nonzero=%d/%d meanB=%.3f\n", nz, rt * rt, sum / ((double)rt * rt * 255.0));
  }
  free(pix);

  [ctx clearDrawable];
  return 0;
}
