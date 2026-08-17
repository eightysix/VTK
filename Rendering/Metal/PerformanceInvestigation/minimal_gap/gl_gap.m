// Minimal, self-contained OpenGL microbenchmark — the exact counterpart of
// metal_gap.m. Same synthetic 512x512x1794 R8 3D texture, same real-scene
// divergent-ray trilinear march (identical rays via inverseVP), same 400x400
// fragments (rtSize overridable), measured with glFinish + wall clock. Exposes
// the intrinsic Metal-vs-GL 3D sampler gap.
//
// Build: clang -framework AppKit -framework OpenGL gl_gap.m -o gl_gap
// Run:   ./gl_gap [flags]
//
// Named flags (positional args are also accepted for backward compat):
//   --frames N      frames to time (default 100)
//   --maxiter N     safety loop bound (default 8192)
//   --nofetch       diagnostic: no volume texture fetch
//   --usedepth      attach a depth renderbuffer
//   --fmt16         GL_R16 volume upload path
//   --flipy         negate NDC y (traces the same rays row-for-row as
//                   metal_gap; GL rasterizes bottom-left, Metal top-left)
//   --rt N          render-target size in px (default 400)
//   --sd F          physical sample distance mm (default 0.5)
//   --data N        1 = pseudo-random volume data
//   --filter N      0 = linear, 1 = nearest
//   --div N         volume div (512/N per axis)
//   --slabs N       slab count (0 = single pass)
//   --slabindex N   single-slab render test
//   --slabt         z-tiling slab clamp
//   --maccum        accumulate slabs into the RT via blending
//   --muladd        mul-add loop variant
//   --camera N      camera preset: 0 oblique, 1 axial(z), 2 coronal(y),
//                   3 sagittal(x), 4 oblique45
//   --composite 0/1 TF lookup + front-to-back over-composite (Airways II
//                   ramp + constant color (0, 0.605, 0.706), break at
//                   acc > 1 - 1/255; --maccum switches the output to the
//                   premultiplied float4(accColor, acc) form)
//   --jitter 0/1    IGN jitter on the sample lattice (app parity)
//   --jitterblock N jitter block size in px (default 1)
//   --clip 0/1      (0,0,1) near-plane ray clip (DICOM scene parity)
//   --preint F      opacity preintegration correction 1-(1-a)^F (default 1)
//   --blend over|max maccum RT blend (default: over when composite, else max)
//
// Readback (glReadPixels RGBA): R/G carry the iteration count (low/high byte),
// B the sample value (MIP) or alpha (composite single pass). With --maccum
// the composite output is premultiplied color: R/G/B = accColor, A = alpha
// (avgIter/meanB decode no longer apply).

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

// Composite variant of kFragSrc: TF lookup + front-to-back over-composite
// (scalar body, app marchVariant 0-2 parity; mirror of metal_gap's
// BuildCompositeBody n<=1). uClip and the IGN jitter are uniform-gated inline
// (same math as metal_gap's ApplyClip/ApplyJitterDecl+ApplyJitterStart; note
// gl_FragCoord.xy needs no +0.5 and the readback row order is bottom-first).
// uMaccum switches the output to the premultiplied float4(accColor, acc) form
// for the over-blend accumulation path (metal_gap ApplyCompositeOut).
static const char* kFragCompositeSrc =
  "#version 150\n"
  "in vec2 vUV;\n"
  "out vec4 fragColor;\n"
  "uniform sampler3D uVol;\n"
  "uniform sampler2D uTF;\n"
  "uniform vec3 uTexelCount;\n"
  "uniform float uSlabStart;\n"
  "uniform float uSlabEnd;\n"
  "uniform bool uSlabT;\n"
  "uniform vec3 uEye;\n"
  "uniform vec3 uBoundsSize;\n"
  "uniform mat4 uInvVP;\n"
  "uniform float uSampleDistMM;\n"
  "uniform int uMaxIter;\n"
  "uniform int uUseJittering;\n"
  "uniform int uJitterBlock;\n"
  "uniform int uClip;\n"
  "uniform int uMaccum;\n"
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
  "  if (uClip > 0) {\n"
  "    vec3 planeOrigin = vec3(0.0, 0.0, 0.0);\n"
  "    vec3 planeNormal = vec3(0.0, 0.0, 1.0);\n"
  "    vec3 entryPoint = uEye + rayDir * tStart;\n"
  "    vec3 exitPoint = uEye + rayDir * tExit;\n"
  "    float startDistance = dot(planeNormal, planeOrigin - entryPoint);\n"
  "    float stopDistance = dot(planeNormal, planeOrigin - exitPoint);\n"
  "    if (startDistance > 0.0 && stopDistance > 0.0) { fragColor = vec4(0.0,0.0,0.0,1.0); return; }\n"
  "    float rayDotNormal = dot(rayDir, planeNormal);\n"
  "    if (rayDotNormal > 0.0 && startDistance > 0.0) entryPoint += (startDistance / rayDotNormal) * rayDir;\n"
  "    if (rayDotNormal <= 0.0 && stopDistance > 0.0) exitPoint += (stopDistance / rayDotNormal) * rayDir;\n"
  "    tStart = dot(entryPoint - uEye, rayDir);\n"
  "    tExit = tStart + length(exitPoint - entryPoint);\n"
  "    if (tExit <= tStart) { fragColor = vec4(0.0,0.0,0.0,1.0); return; }\n"
  "  }\n"
  "  float physPerNorm = length(rayDir * uBoundsSize);\n"
  "  float stepSize = uSampleDistMM / max(physPerNorm, 1e-6);\n"
  "  float jitterF = uUseJittering > 0 ? fract(52.9829189 * fract(dot(floor(gl_FragCoord.xy / float(max(uJitterBlock, 1))) * float(max(uJitterBlock, 1)) + 0.5 * float(max(uJitterBlock, 1)), vec2(0.06711056, 0.00583715)))) * stepSize : 0.0;\n"
  "  float jitterT = jitterF + ceil((tStart - jitterF) / stepSize) * stepSize;\n"
  "  float startT = jitterT;\n"
  "  if (uSlabT) {\n"
  "    float t_s = (uSlabStart - uEye.z) / rayDir.z;\n"
  "    float t_e = (uSlabEnd - uEye.z) / rayDir.z;\n"
  "    float tlo = max(startT, min(t_s, t_e));\n"
  "    float thi = min(tExit, max(t_s, t_e));\n"
  "    startT = startT + ceil(max((tlo - startT) / stepSize, 0.0)) * stepSize;\n"
  "    if (uSlabEnd < 1.0) tExit = startT + ceil(max((thi - startT) / stepSize, 0.0)) * stepSize;\n"
  "    else tExit = thi;\n"
  "  }\n"
  "  int maxSteps = max(0, int(ceil((tExit - startT) / stepSize)));\n"
  "  vec3 texelCount = uTexelCount;\n"
  "  vec3 ctpScale = max(texelCount - 1.0, 1e-4) / texelCount;\n"
  "  vec3 ctpOffset = 0.5 / texelCount;\n"
  "  vec3 texStep = rayDir * stepSize;\n"
  "  vec3 evalStep = texStep * ctpScale;\n"
  "  float currentT = startT;\n"
  "  vec3 texLocal = uEye + rayDir * currentT;\n"
  "  vec3 evalPoint = texLocal * ctpScale + ctpOffset;\n"
  "  evalPoint.z = clamp(evalPoint.z, uSlabStart * ctpScale.z + ctpOffset.z, uSlabEnd * ctpScale.z + ctpOffset.z);\n"
  "  float acc = 0.0;\n"
  "  float n = 0.0;\n"
  "  vec3 accColor = vec3(0.0);\n"
  "  for (int i = 0; i < min(uMaxIter, maxSteps); i++) {\n"
  "    if (currentT >= tExit - 1e-6) break;\n"
  "    float s = texture(uVol, evalPoint).r;\n"
  "    n += 1.0;\n"
  "    vec4 c = texture(uTF, vec2(clamp(s, 0.0, 1.0), 0.5)).rgba;\n"
  "    float w = 1.0 - acc;\n"
  "    accColor += w * c.rgb * c.a;\n"
  "    acc = acc + w * c.a;\n"
  "    if (acc > 1.0 - 1.0/255.0) break;\n"
  "    currentT += stepSize;\n"
  "    texLocal += texStep;\n"
  "    evalPoint += evalStep;\n"
  "  }\n"
  "  int nc = int(n);\n"
  "  if (uMaccum > 0) fragColor = vec4(accColor, acc);\n"
  "  else fragColor = vec4(float(nc & 255) / 255.0, float((nc >> 8) & 255) / 255.0, acc, 1.0);\n"
  "}\n";

// Composite variant of kFragMulAddSrc: mul-add loop with the TF lookup +
// over-composite body (metal_gap ApplyMulAddDecls/ApplySlabT parity: raw
// tStartRaw/tExitRaw captured AFTER the clip, jitter re-points tStartRaw at
// the lattice origin, kPass realigns the slab start, tExit ceil-bound on the
// raw base).
static const char* kFragMulAddCompositeSrc =
  "#version 150\n"
  "in vec2 vUV;\n"
  "out vec4 fragColor;\n"
  "uniform sampler3D uVol;\n"
  "uniform sampler2D uTF;\n"
  "uniform vec3 uTexelCount;\n"
  "uniform float uSlabStart;\n"
  "uniform float uSlabEnd;\n"
  "uniform bool uSlabT;\n"
  "uniform vec3 uEye;\n"
  "uniform vec3 uBoundsSize;\n"
  "uniform mat4 uInvVP;\n"
  "uniform float uSampleDistMM;\n"
  "uniform int uMaxIter;\n"
  "uniform int uUseJittering;\n"
  "uniform int uJitterBlock;\n"
  "uniform int uClip;\n"
  "uniform int uMaccum;\n"
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
  "  if (uClip > 0) {\n"
  "    vec3 planeOrigin = vec3(0.0, 0.0, 0.0);\n"
  "    vec3 planeNormal = vec3(0.0, 0.0, 1.0);\n"
  "    vec3 entryPoint = uEye + rayDir * tStart;\n"
  "    vec3 exitPoint = uEye + rayDir * tExit;\n"
  "    float startDistance = dot(planeNormal, planeOrigin - entryPoint);\n"
  "    float stopDistance = dot(planeNormal, planeOrigin - exitPoint);\n"
  "    if (startDistance > 0.0 && stopDistance > 0.0) { fragColor = vec4(0.0,0.0,0.0,1.0); return; }\n"
  "    float rayDotNormal = dot(rayDir, planeNormal);\n"
  "    if (rayDotNormal > 0.0 && startDistance > 0.0) entryPoint += (startDistance / rayDotNormal) * rayDir;\n"
  "    if (rayDotNormal <= 0.0 && stopDistance > 0.0) exitPoint += (stopDistance / rayDotNormal) * rayDir;\n"
  "    tStart = dot(entryPoint - uEye, rayDir);\n"
  "    tExit = tStart + length(exitPoint - entryPoint);\n"
  "    if (tExit <= tStart) { fragColor = vec4(0.0,0.0,0.0,1.0); return; }\n"
  "  }\n"
  "  float tStartRaw = tStart;\n"
  "  float tExitRaw = tExit;\n"
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
  "  float jitterF = uUseJittering > 0 ? fract(52.9829189 * fract(dot(floor(gl_FragCoord.xy / float(max(uJitterBlock, 1))) * float(max(uJitterBlock, 1)) + 0.5 * float(max(uJitterBlock, 1)), vec2(0.06711056, 0.00583715)))) * stepSize : 0.0;\n"
  "  float jitterT = jitterF + ceil((tStartRaw - jitterF) / stepSize) * stepSize;\n"
  "  if (uUseJittering > 0) tStartRaw = jitterT;\n"
  "  int kPass = int(ceil(max((tlo - tStartRaw) / stepSize, 0.0)));\n"
  "  tStart = tStartRaw + float(kPass) * stepSize;\n"
  "  if (uSlabT) tExit = tStartRaw + float(int(ceil(max((thi - tStartRaw) / stepSize, 0.0)))) * stepSize;\n"
  "  int maxSteps = max(0, int(ceil((tExit - tStart) / stepSize)));\n"
  "  vec3 texelCount = uTexelCount;\n"
  "  vec3 ctpScale = max(texelCount - 1.0, 1e-4) / texelCount;\n"
  "  vec3 ctpOffset = 0.5 / texelCount;\n"
  "  vec3 evalBase = ctpOffset + (uEye + rayDir * tStartRaw) * ctpScale;\n"
  "  vec3 evalStep = rayDir * ctpScale * stepSize;\n"
  "  float acc = 0.0;\n"
  "  float n = 0.0;\n"
  "  vec3 accColor = vec3(0.0);\n"
  "  for (int i = 0; i < min(uMaxIter, maxSteps); i++) {\n"
  "    float currentT = tStartRaw + float(kPass + i) * stepSize;\n"
  "    if (currentT >= min(tExit, tExitRaw) - 1e-6) break;\n"
  "    vec3 evalPoint = evalBase + float(kPass + i) * evalStep;\n"
  "    float s = texture(uVol, evalPoint).r;\n"
  "    n += 1.0;\n"
  "    vec4 c = texture(uTF, vec2(clamp(s, 0.0, 1.0), 0.5)).rgba;\n"
  "    float w = 1.0 - acc;\n"
  "    accColor += w * c.rgb * c.a;\n"
  "    acc = acc + w * c.a;\n"
  "    if (acc > 1.0 - 1.0/255.0) break;\n"
  "  }\n"
  "  int nc = int(n);\n"
  "  if (uMaccum > 0) fragColor = vec4(accColor, acc);\n"
  "  else fragColor = vec4(float(nc & 255) / 255.0, float((nc >> 8) & 255) / 255.0, acc, 1.0);\n"
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

// Bake the DICOM reference scene's "Airways II" transfer function into a
// 256-entry RGBA8 row (TestMetalScenes.h BuildDICOMVolumeScene): constant
// color (0, 0.605, 0.706); opacity points (-742.1,0)(-683,0.0493)(-481,0.2497)
// (-333.5,0) with x rescaled to the U8 volume domain: (hu+1024)*255/4095.
// `preint` applies the OpenGL composite-blend pre-integration correction
// 1-(1-a)^factor (vtkOpenGLVolumeOpacityTable::InternalUpdate). Must match
// metal_gap's BakeTFTable exactly (same ramp, same quantization).
static void BakeTFTable(uint8_t* row, double preint)
{
  const double xs[4] = {
    (-742.1 + 1024.0) * 255.0 / 4095.0,
    (-683.0 + 1024.0) * 255.0 / 4095.0,
    (-481.0 + 1024.0) * 255.0 / 4095.0,
    (-333.5 + 1024.0) * 255.0 / 4095.0
  };
  const double ys[4] = { 0.0, 0.0493, 0.2497, 0.0 };
  for (int i = 0; i < 256; i++) {
    double x = (double)i;
    double a = 0.0;
    for (int k = 0; k < 3; k++) {
      if (x >= xs[k] && x <= xs[k + 1]) {
        double f = (xs[k + 1] > xs[k]) ? (x - xs[k]) / (xs[k + 1] - xs[k]) : 0.0;
        a = ys[k] + f * (ys[k + 1] - ys[k]);
        break;
      }
    }
    if (a > 0.0001 && preint > 0.0) a = 1.0 - pow(1.0 - a, preint);
    row[i * 4 + 0] = (uint8_t)(0.0 * 255.0 + 0.5);
    row[i * 4 + 1] = (uint8_t)(0.605 * 255.0 + 0.5);
    row[i * 4 + 2] = (uint8_t)(0.706 * 255.0 + 0.5);
    row[i * 4 + 3] = (uint8_t)(a * 255.0 + 0.5);
  }
}

static int IntArg(int argc, const char** argv, const char* name, int def)
{
  for (int i = 2; i < argc; i++)
    if (strcmp(argv[i - 1], name) == 0) return atoi(argv[i]);
  return def;
}

static float FloatArg(int argc, const char** argv, const char* name, float def)
{
  for (int i = 2; i < argc; i++)
    if (strcmp(argv[i - 1], name) == 0) return (float)atof(argv[i]);
  return def;
}

static const char* StrArg(int argc, const char** argv, const char* name, const char* def)
{
  for (int i = 2; i < argc; i++)
    if (strcmp(argv[i - 1], name) == 0) return argv[i];
  return def;
}

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
  int composite = 0;
  int jitter = 0;
  int jitterBlock = 1;
  int clip = 0;
  float preint = 1.0f;
  int blendOver = -1;
  int camera = 0;
  const bool named = (argc > 1 && argv[1][0] == '-' && argv[1][1] == '-');
  if (named) {
    frames = IntArg(argc, argv, "--frames", frames);
    maxIter = IntArg(argc, argv, "--maxiter", maxIter);
    noFetch = IntArg(argc, argv, "--nofetch", noFetch);
    useDepth = IntArg(argc, argv, "--usedepth", useDepth);
    fmt16 = IntArg(argc, argv, "--fmt16", fmt16);
    flipY = IntArg(argc, argv, "--flipy", flipY);
    rtSize = IntArg(argc, argv, "--rt", rtSize);
    sampleDistMM = FloatArg(argc, argv, "--sd", sampleDistMM);
    dataMode = IntArg(argc, argv, "--data", dataMode);
    filterMode = IntArg(argc, argv, "--filter", filterMode);
    volDiv = IntArg(argc, argv, "--div", volDiv);
    numSlabs = IntArg(argc, argv, "--slabs", numSlabs);
    slabIndex = IntArg(argc, argv, "--slabindex", slabIndex);
    slabT = IntArg(argc, argv, "--slabt", slabT);
    maccum = IntArg(argc, argv, "--maccum", maccum);
    mulAdd = IntArg(argc, argv, "--muladd", mulAdd);
    camera = IntArg(argc, argv, "--camera", camera);
    composite = IntArg(argc, argv, "--composite", composite);
    jitter = IntArg(argc, argv, "--jitter", jitter);
    jitterBlock = IntArg(argc, argv, "--jitterblock", jitterBlock);
    clip = IntArg(argc, argv, "--clip", clip);
    preint = FloatArg(argc, argv, "--preint", preint);
    const char* blend = StrArg(argc, argv, "--blend", NULL);
    if (blend) blendOver = strcmp(blend, "over") == 0 ? 1 : 0;
  } else {
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
  if (argc > 12) numSlabs = atoi(argv[12]);
    if (argc > 13) slabIndex = atoi(argv[13]);
    if (argc > 14) slabT = atoi(argv[14]);
    if (argc > 15) maccum = atoi(argv[15]);
    if (argc > 16) mulAdd = atoi(argv[16]);
    if (argc > 17) camera = atoi(argv[17]);
  }
  if (volDiv > 1) { kW = 512 / volDiv; kH = 512 / volDiv; kD = 1794 / volDiv; }
  if (blendOver < 0) blendOver = composite ? 1 : 0;

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
  printf("volume %dx%dx%d R8, rt %dx%d, frames %d, maxIter %d, noFetch %d, flipY %d, sampleDistMM=%.1f, dataMode=%d, filterMode=%d, volDiv=%d, slab=%d/%d, slabT=%d, maccum=%d, mulAdd=%d, camera=%d, composite=%d, jitter=%d(jb=%d), clip=%d, preint=%.2f, blend=%s\n",
    kW, kH, kD, rtSize, rtSize, frames, maxIter, noFetch, flipY, sampleDistMM, dataMode, filterMode, volDiv, slabIndex, numSlabs, slabT, maccum, mulAdd, camera, composite, jitter, jitterBlock, clip, preint, blendOver ? "over" : "max");

  // Program. flipY=1 negates NDC y so the same readback row traces the same
  // ray as metal_gap (which uses Metal's top-left window convention).
  const char* fragSrc0;
  if (composite) fragSrc0 = mulAdd ? kFragMulAddCompositeSrc : kFragCompositeSrc;
  else fragSrc0 = mulAdd ? kFragMulAddSrc : (noFetch ? kFragNoFetchSrc : kFragSrc);
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
  GLint uTF = glGetUniformLocation(prog, "uTF");
  GLint uUseJittering = glGetUniformLocation(prog, "uUseJittering");
  GLint uJitterBlock = glGetUniformLocation(prog, "uJitterBlock");
  GLint uClip = glGetUniformLocation(prog, "uClip");
  GLint uMaccum = glGetUniformLocation(prog, "uMaccum");

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

  // Transfer function texture (Airways II, 256x1 RGBA8, nearest): unit 1.
  if (composite) {
    uint8_t tf[256 * 4];
    BakeTFTable(tf, preint);
    GLuint tfTex;
    glGenTextures(1, &tfTex);
    glBindTexture(GL_TEXTURE_2D, tfTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, tf);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tfTex);
    glUniform1i(uTF, 1);
    glActiveTexture(GL_TEXTURE0);
  }

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

  // Camera presets: 0=original oblique, 1=axial(z), 2=coronal(y), 3=sagittal(x), 4=oblique45
  GLfloat camEye[3];
  GLfloat camInvVP[16];
  memcpy(camEye, (GLfloat[]){-1.49527037f, -0.95243806f, 2.55352926f}, sizeof(camEye));
  memcpy(camInvVP, invVP, sizeof(camInvVP));
  if (camera == 1) {
    camEye[0]=0.5f; camEye[1]=0.5f; camEye[2]=2.5f;
    const GLfloat iv[16] = {
        4.14213562e-01f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f,
        0.00000000e+00f, 4.14213562e-01f, 0.00000000e+00f, 0.00000000e+00f,
        -1.06540435e+04f, -1.06540435e+04f, -8.96491035e+04f, -4.99995000e+01f,
        1.06542565e+04f, 1.06542565e+04f, 8.96498965e+04f, 5.00005000e+01f};
    memcpy(camInvVP, iv, sizeof(iv));
  } else if (camera == 2) {
    camEye[0]=0.5f; camEye[1]=2.5f; camEye[2]=0.5f;
    const GLfloat iv[16] = {
        -4.14213562e-01f, 0.00000000e+00f, 0.00000000e+00f, 0.00000000e+00f,
        0.00000000e+00f, 0.00000000e+00f, 4.14213562e-01f, 0.00000000e+00f,
        -1.06540435e+04f, -5.32702173e+04f, -1.79298207e+04f, -4.99995000e+01f,
        1.06542565e+04f, 5.32702827e+04f, 1.79301793e+04f, 5.00005000e+01f};
    memcpy(camInvVP, iv, sizeof(iv));
  } else if (camera == 3) {
    camEye[0]=2.5f; camEye[1]=0.5f; camEye[2]=0.5f;
    const GLfloat iv[16] = {
        0.00000000e+00f, 4.14213562e-01f, 0.00000000e+00f, 0.00000000e+00f,
        0.00000000e+00f, 0.00000000e+00f, 4.14213562e-01f, 0.00000000e+00f,
        -5.32702173e+04f, -1.06540435e+04f, -1.79298207e+04f, -4.99995000e+01f,
        5.32702827e+04f, 1.06542565e+04f, 1.79301793e+04f, 5.00005000e+01f};
    memcpy(camInvVP, iv, sizeof(iv));
  } else if (camera == 4) {
    camEye[0]=1.5f; camEye[1]=1.5f; camEye[2]=1.5f;
    const GLfloat iv[16] = {
        3.56091876e-01f, -1.36776692e-12f, -2.11592653e-01f, -2.13967351e-15f,
        -9.62561335e-02f, 3.68872156e-01f, -1.61990630e-01f, -2.42710924e-15f,
        -3.19621304e+04f, -3.19621304e+04f, -5.37894621e+04f, -4.99995000e+01f,
        3.19623147e+04f, 3.19623147e+04f, 5.37897723e+04f, 5.00005000e+01f};
    memcpy(camInvVP, iv, sizeof(iv));
  }
  glUniform3f(uEye, camEye[0], camEye[1], camEye[2]);
  glUniformMatrix4fv(uInvVP, 1, GL_FALSE, camInvVP);
  glUniform1f(uSampleDistMM, sampleDistMM);
  glUniform3f(uTexelCount, (float)kW, (float)kH, (float)kD);
  glUniform1f(uSlabStart, numSlabs > 0 ? (float)slabIndex / numSlabs : 0.0f);
  glUniform1f(uSlabEnd, numSlabs > 0 ? (float)(slabIndex + 1) / numSlabs : 1.0f);
  glUniform1i(uSlabT, slabT ? 1 : 0);
  glUniform1i(uMaxIter, maxIter);
  glUniform1i(uUseJittering, jitter ? 1 : 0);
  glUniform1i(uJitterBlock, jitterBlock > 0 ? jitterBlock : 1);
  glUniform1i(uClip, clip ? 1 : 0);
  glUniform1i(uMaccum, maccum ? 1 : 0);
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
    if (blendOver) {
      glBlendEquation(GL_FUNC_ADD);
      glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    } else {
      glBlendEquation(GL_MAX);
      glBlendFunc(GL_ONE, GL_ONE);
    }
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
