// Minimal, self-contained Metal microbenchmark: the real DICOM scene's exact
// divergent perspective rays marching through a 512x512x1794 R8 3D texture
// with hardware trilinear filtering. Matches the app's VolumeMapperUniforms
// (CameraVolumePos, inverse view-projection, per-ray sample distance) so the
// raw-fetch march cost is identical to the production Metal renderer. The
// paired GL version (gl_gap) runs the same rays so the intrinsic Metal-vs-GL
// 3D sampler throughput gap can be measured in isolation.
//
// Build: clang -fobjc-arc -framework Metal -framework Foundation metal_gap.m -o metal_gap
// Run:   ./metal_gap [frames] [maxIter] [halfSampler] [depthMode] [compute] [lod0]
//   halfSampler=1: texture3d<half> + half accumulator (FP16 sampler return path)
//   depthMode: 0 = depth attached, MTLStoreActionStore (legacy), 1 = depth
//              attached, MTLStoreActionDontCare, 2 = no depth attachment
//              (matches gl_gap, which attaches no depth buffer by default)
//   compute=1: replace the fragment render pass with a compute kernel (same
//              march, one thread per pixel, no rasterizer/TBDR tile machinery)
//   lod0=1: sample(..., level(0.0f)) — explicit LOD 0, skipping the implicit
//              screen-space gradient computation MSL fragment .sample() does

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <simd/simd.h>

struct Uniforms {
  simd_float4 eye;         // camera pos, normalized volume space
  simd_float4 boundsSize;  // physical volume size (mm)
  simd_float4x4 invVP;     // NDC -> physical volume coords
  float sampleDistMM;      // physical sample distance (mm)
  float _pad0;
  float _pad1;
  int   maxIter;           // safety loop bound
};

static const int kW = 512;  // volume dims (slice-stacked, Z is the long axis)
static const int kH = 512;
static const int kD = 1794;
static const int kRT = 400; // render target size (fragments), matches real scene

// Real app uniforms for the DICOMVolume scene (dump via
// VTK_METAL_TEST_DUMP_UNIFORMS): CameraVolumePos (normalized),
// VolumeBoundsMax (physical mm), InverseViewProjection (NDC->physical).
// Builds the march MSL. `halfSampler` switches the volume texture and the
// accumulator to FP16 so the TMU/ALU sample-return path stays in half instead
// of forcing FP32 registers (experiment: does the GL stack's implicit 16-bit
// fast path explain part of the gap?).
// BuildMSL: builds the march MSL. `halfSampler` switches the volume texture
// and the accumulator to FP16. `lod0` appends `level(0.0f)` to the sample
// call, forcing an explicit LOD-0 fetch and (hypothesis) skipping the implicit
// screen-space gradient/LOD computation that MSL fragment `.sample()` does per
// iteration; GL with GL_LINEAR (no mipmaps) needs no such gradients.
static char* BuildMSL(bool halfSampler, bool lod0)
{
  const char* fmt = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct VOut {
  float4 position [[position]];
  float2 uv;
};

vertex VOut vertex_main(uint vid [[vertex_id]]) {
  VOut o;
  float2 p = float2(float(vid & 1u) * 4.0f - 1.0f, float((vid >> 1u) & 1u) * 2.0f - 1.0f);
  if (vid == 2u) p = float2(-1.0f, 3.0f);
  o.position = float4(p, 0.0f, 1.0f);
  o.uv = p * 0.5f + 0.5f;
  return o;
}

struct Uniforms {
  float4 eye;         // camera position, normalized volume space
  float4 boundsSize;  // physical volume size (mm)
  float4x4 invVP;     // NDC -> physical volume coords
  float  sampleDistMM; // physical sample distance (mm)
  float  pad1;
  float  pad2;
  int    maxIter;     // safety loop bound
};

fragment float4 fragment_main(VOut in [[stage_in]],
                              texture3d<%s> volTex [[texture(0)]],
                              constant Uniforms& u [[buffer(0)]]) {
  constexpr sampler volSampler(filter::linear, address::clamp_to_edge);
  float2 ndc = in.uv * 2.0f - 1.0f;

  // Reconstruct the pixel ray exactly like the app: unproject the fragment to
  // a physical volume point via inverseVP, then normalize the volume-space
  // direction from the camera through that point (OpenGL/Metal parity: the
  // interpolated proxy-box anchor lies on the same camera ray).
  float4 w4 = u.invVP * float4(ndc, 0.0f, 1.0f);
  float3 ptPhys = w4.xyz / w4.w;
  float3 eye = u.eye.xyz;
  float3 rayDir = normalize(ptPhys / u.boundsSize.xyz - eye);
  float3 inv = 1.0f / rayDir;

  // Slab intersect with the normalized [0,1]^3 box (app: intersectBox).
  float3 t0 = (float3(0.0f) - eye) * inv;
  float3 t1 = (float3(1.0f) - eye) * inv;
  float3 tmin3 = min(t0, t1);
  float3 tmax3 = max(t0, t1);
  float tEnter = max(max(tmin3.x, tmin3.y), tmin3.z);
  float tExit  = min(min(tmax3.x, tmax3.y), tmax3.z);
  if (tExit <= 0.0f || tEnter >= tExit) {
    return float4(0.0f, 0.0f, 0.0f, 1.0f);
  }
  float tStart = max(tEnter, 0.0f);

  // App's physicalSampleStep: normalized step for a constant physical sample
  // distance along the ray (u.sampleDistance was stored as 0.5mm/maxBound).
  float physPerNorm = length(rayDir * u.boundsSize.xyz);
  float stepSize = u.sampleDistMM / max(physPerNorm, 1e-6f);
  int maxSteps = max(1, int(ceil((tExit - tStart) / stepSize)));

  // Clamp-to-edge texel correction (app: ctpScale/ctpOffset).
  float3 texelCount = float3(volTex.get_width(), volTex.get_height(), volTex.get_depth());
  float3 ctpScale = max(texelCount - 1.0f, 1e-4f) / texelCount;
  float3 ctpOffset = 0.5f / texelCount;
  float3 texStep = rayDir * stepSize;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;
  float3 texLocal = eye + rayDir * currentT;
  float3 evalPoint = texLocal * ctpScale + ctpOffset;

  %s acc = 0.0%s;
  float n = 0.0f;
  for (int i = 0; i < min(u.maxIter, maxSteps); i++) {
    if (currentT >= tExit - 1e-6f) break;
    acc = max(acc, volTex.sample(volSampler, evalPoint%s).r);
    currentT += stepSize;
    texLocal += texStep;
    evalPoint += evalStep;
    n += 1.0f;
  }
  // Pack iteration count into R/G (low/high byte), sample value into B.
  uint nc = uint(n);
  return float4(float(nc & 255u) / 255.0f, float((nc >> 8u) & 255u) / 255.0f, float(acc), 1.0f);
}
)MSL";
  char* buf = malloc(strlen(fmt) + 32);
  snprintf(buf, strlen(fmt) + 32, fmt,
    halfSampler ? "half" : "float",
    halfSampler ? "half" : "float",
    halfSampler ? "h" : "f",
    lod0 ? ", level(0.0f)" : "");
  return buf;
}

// Diagnostic variant: writes rayDir*0.5+0.5 to RGB so the actual ray field can
// be compared against the fp64 reference (used to explain the avgIter gap).
static char* BuildDiagMSL(void)
{
  const char* fmt = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct VOut { float4 position [[position]]; float2 uv; };

vertex VOut vertex_main(uint vid [[vertex_id]]) {
  VOut o;
  float2 p = float2(float(vid & 1u) * 4.0f - 1.0f, float((vid >> 1u) & 1u) * 2.0f - 1.0f);
  if (vid == 2u) p = float2(-1.0f, 3.0f);
  o.position = float4(p, 0.0f, 1.0f);
  o.uv = p * 0.5f + 0.5f;
  return o;
}

struct Uniforms {
  float4 eye;         // camera position, normalized volume space
  float4 boundsSize;  // physical volume size (mm)
  float4x4 invVP;     // NDC -> physical volume coords
  float  sampleDistMM; // physical sample distance (mm)
  float  pad1;
  float  pad2;
  int    maxIter;     // safety loop bound
};

fragment float4 fragment_main(VOut in [[stage_in]],
                              texture3d<float> volTex [[texture(0)]],
                              constant Uniforms& u [[buffer(0)]]) {
  float2 ndc = in.uv * 2.0f - 1.0f;
  float4 w4 = u.invVP * float4(ndc, 0.0f, 1.0f);
  float3 ptPhys = w4.xyz / w4.w;
  float3 eye = u.eye.xyz;
  float3 rayDir = normalize(ptPhys / u.boundsSize.xyz - eye);
  return float4(rayDir * 0.5f + 0.5f, 1.0f);
}
)MSL";
  char* buf = malloc(strlen(fmt) + 8);
  strcpy(buf, fmt);
  return buf;
}

// depthMode: 0 = depth attached + MTLStoreActionStore, 1 = depth attached +
// MTLStoreActionDontCare (skip the tile->memory depth writeback), 2 = no depth
// attachment at all (matches gl_gap's default no-depth setup).
static MTLRenderPassDescriptor* MakeRPD(id<MTLTexture> rt, id<MTLTexture> depth, int depthMode)
{
  MTLRenderPassDescriptor* rpd = [[MTLRenderPassDescriptor alloc] init];
  rpd.colorAttachments[0].texture = rt;
  rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
  rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
  if (depthMode != 2) {
    rpd.depthAttachment.texture = depth;
    rpd.depthAttachment.loadAction = MTLLoadActionClear;
    rpd.depthAttachment.storeAction = (depthMode == 1) ? MTLStoreActionDontCare : MTLStoreActionStore;
  }
  return rpd;
}

// Compute-kernel variant of the same march: one thread per pixel, no render
// pass / rasterizer / depth attachment. Tests whether the fragment pipeline's
// per-tile machinery (vs a raw compute dispatch) contributes to the Metal gap.
static char* BuildComputeMSL(bool halfSampler)
{
  const char* fmt = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
  float4 eye;         // camera position, normalized volume space
  float4 boundsSize;  // physical volume size (mm)
  float4x4 invVP;     // NDC -> physical volume coords
  float  sampleDistMM; // physical sample distance (mm)
  float  pad1;
  float  pad2;
  int    maxIter;     // safety loop bound
};

kernel void compute_main(uint2 gid [[thread_position_in_grid]],
                         texture3d<%s> volTex [[texture(0)]],
                         texture2d<float, access::write> outTex [[texture(1)]],
                         constant Uniforms& u [[buffer(0)]]) {
  const uint kRT = 400;
  constexpr sampler volSampler(filter::linear, address::clamp_to_edge);
  float2 uv = (float2(gid) + 0.5f) / float2(kRT, kRT);
  float2 ndc = uv * 2.0f - 1.0f;

  float4 w4 = u.invVP * float4(ndc, 0.0f, 1.0f);
  float3 ptPhys = w4.xyz / w4.w;
  float3 eye = u.eye.xyz;
  float3 rayDir = normalize(ptPhys / u.boundsSize.xyz - eye);
  float3 inv = 1.0f / rayDir;

  float3 t0 = (float3(0.0f) - eye) * inv;
  float3 t1 = (float3(1.0f) - eye) * inv;
  float3 tmin3 = min(t0, t1);
  float3 tmax3 = max(t0, t1);
  float tEnter = max(max(tmin3.x, tmin3.y), tmin3.z);
  float tExit  = min(min(tmax3.x, tmax3.y), tmax3.z);
  if (tExit <= 0.0f || tEnter >= tExit) {
    outTex.write(float4(0.0f, 0.0f, 0.0f, 1.0f), gid);
    return;
  }
  float tStart = max(tEnter, 0.0f);

  float physPerNorm = length(rayDir * u.boundsSize.xyz);
  float stepSize = u.sampleDistMM / max(physPerNorm, 1e-6f);
  int maxSteps = max(1, int(ceil((tExit - tStart) / stepSize)));

  float3 texelCount = float3(volTex.get_width(), volTex.get_height(), volTex.get_depth());
  float3 ctpScale = max(texelCount - 1.0f, 1e-4f) / texelCount;
  float3 ctpOffset = 0.5f / texelCount;
  float3 texStep = rayDir * stepSize;
  float3 evalStep = texStep * ctpScale;

  float currentT = tStart;
  float3 texLocal = eye + rayDir * currentT;
  float3 evalPoint = texLocal * ctpScale + ctpOffset;

  %s acc = 0.0%s;
  float n = 0.0f;
  for (int i = 0; i < min(u.maxIter, maxSteps); i++) {
    if (currentT >= tExit - 1e-6f) break;
    acc = max(acc, volTex.sample(volSampler, evalPoint).r);
    currentT += stepSize;
    texLocal += texStep;
    evalPoint += evalStep;
    n += 1.0f;
  }
  uint nc = uint(n);
  outTex.write(float4(float(nc & 255u) / 255.0f, float((nc >> 8u) & 255u) / 255.0f, float(acc), 1.0f), gid);
}
)MSL";
  char* buf = malloc(strlen(fmt) + 32);
  snprintf(buf, strlen(fmt) + 32, fmt,
    halfSampler ? "half" : "float",
    halfSampler ? "half" : "float",
    halfSampler ? "h" : "f");
  return buf;
}

int main(int argc, const char** argv)
{
  @autoreleasepool {
    int frames = 100;
    int maxIter = 4096;
    int halfSampler = 0;
    int depthMode = 0;
    int compute = 0;
    int lod0 = 0;
    int fastMath = 1;
    int diag = 0;
    if (argc > 1) frames = atoi(argv[1]);
    if (argc > 2) maxIter = atoi(argv[2]);
    if (argc > 3) halfSampler = atoi(argv[3]);
    if (argc > 4) depthMode = atoi(argv[4]);
    if (argc > 5) compute = atoi(argv[5]);
    if (argc > 6) lod0 = atoi(argv[6]);
    if (argc > 7) fastMath = atoi(argv[7]);
    if (argc > 8) diag = atoi(argv[8]);

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    fprintf(stderr, "Metal device: %s\n", device.name.UTF8String);
    fprintf(stderr, "volume %dx%dx%d R8, rt %dx%d, frames %d, maxIter %d, halfSampler=%d, depthMode=%d, compute=%d, lod0=%d, fastMath=%d, diag=%d\n",
      kW, kH, kD, kRT, kRT, frames, maxIter, halfSampler, depthMode, compute, lod0, fastMath, diag);

    NSError* err = nil;
    char* msl = diag ? BuildDiagMSL()
                     : (compute ? BuildComputeMSL(halfSampler != 0) : BuildMSL(halfSampler != 0, lod0 != 0));
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.fastMathEnabled = (fastMath != 0) ? YES : NO;
    id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithUTF8String:msl]
                                              options:opts error:&err];
    free(msl);
    if (!lib) { fprintf(stderr, "library compile failed: %s\n", err.description.UTF8String); return 1; }
    id<MTLFunction> vertF = nil;
    id<MTLFunction> fragF = nil;
    id<MTLFunction> kernF = nil;
    id<MTLRenderPipelineState> pso = nil;
    id<MTLComputePipelineState> cps = nil;
    if (compute) {
      kernF = [lib newFunctionWithName:@"compute_main"];
      if (!kernF) { fprintf(stderr, "kernel lookup failed\n"); return 1; }
      cps = [device newComputePipelineStateWithFunction:kernF error:&err];
      if (!cps) { fprintf(stderr, "compute pso failed: %s\n", err.description.UTF8String); return 1; }
    } else {
      vertF = [lib newFunctionWithName:@"vertex_main"];
      fragF = [lib newFunctionWithName:@"fragment_main"];
      if (!vertF || !fragF) { fprintf(stderr, "function lookup failed\n"); return 1; }
      MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
      pd.vertexFunction = vertF;
      pd.fragmentFunction = fragF;
      pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
      if (depthMode != 2) {
        pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
      }
      pso = [device newRenderPipelineStateWithDescriptor:pd error:&err];
      if (!pso) { fprintf(stderr, "pso failed: %s\n", err.description.UTF8String); return 1; }
    }

    // Synthetic 512^3 volume.
    size_t total = (size_t)kW * kH * kD;
    uint8_t* host = malloc(total);
    for (size_t i = 0; i < total; i++) host[i] = (uint8_t)((i >> 10) & 0xff);

    MTLTextureDescriptor* vd = [[MTLTextureDescriptor alloc] init];
    vd.textureType = MTLTextureType3D;
    vd.pixelFormat = MTLPixelFormatR8Unorm;
    vd.width = kW; vd.height = kH; vd.depth = kD;
    vd.mipmapLevelCount = 1;
    vd.usage = MTLTextureUsageShaderRead;
    vd.storageMode = MTLStorageModePrivate;
    id<MTLTexture> volTex = [device newTextureWithDescriptor:vd];

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> cb0 = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb0 blitCommandEncoder];
    [blit copyFromBuffer:[device newBufferWithBytes:host length:total options:MTLResourceStorageModeShared]
            sourceOffset:0 sourceBytesPerRow:kW sourceBytesPerImage:kW*kH
            sourceSize:MTLSizeMake(kW, kH, kD) toTexture:volTex
            destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
    [blit endEncoding];
    [cb0 commit];
    [cb0 waitUntilCompleted];
    free(host);

    // Render target.
    MTLTextureDescriptor* rtd = [[MTLTextureDescriptor alloc] init];
    rtd.textureType = MTLTextureType2D;
    rtd.pixelFormat = MTLPixelFormatBGRA8Unorm;
    rtd.width = kRT; rtd.height = kRT;
    rtd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderWrite;
    rtd.storageMode = MTLStorageModeShared;
    id<MTLTexture> rt = [device newTextureWithDescriptor:rtd];

    MTLTextureDescriptor* dd = [[MTLTextureDescriptor alloc] init];
    dd.textureType = MTLTextureType2D;
    dd.pixelFormat = MTLPixelFormatDepth32Float;
    dd.width = kRT; dd.height = kRT;
    dd.usage = MTLTextureUsageRenderTarget;
    dd.storageMode = MTLStorageModePrivate;
    id<MTLTexture> depth = [device newTextureWithDescriptor:dd];

    struct Uniforms u;
    // Real app camera (DICOMVolume scene, VTK_METAL_TEST_DUMP_UNIFORMS dump):
    // eye normalized in volume space, bounds physical mm, inverseVP NDC->phys.
    u.eye = (simd_float4){ -1.49527037f, -0.95243806f, 2.55352926f, 0.0f };
    u.boundsSize = (simd_float4){ 426.166f, 426.166f, 717.2f, 0.0f };
    const float invVP[16] = {
      0.23205078f, -6.974323e-09f, 0.13397458f, 1.2084004e-11f,
      -0.045822006f, 0.25178984f, 0.079366043f, -3.2157374e-12f,
      0.35811684f, 0.22810864f, -1.0292176f, -0.00056198676f,
      -0.12124427f, -0.03448515f, 0.8849802f, 0.00092758867f };
    memcpy(&u.invVP, invVP, sizeof(invVP));
    u.sampleDistMM = 0.5f;
    u._pad0 = 0.0f;
    u._pad1 = 0.0f;
    u.maxIter = maxIter;
    id<MTLBuffer> ubuf = [device newBufferWithBytes:&u length:sizeof(u) options:MTLResourceStorageModeShared];

    if (diag) {
      // Print what Metal actually receives for invVP (columns) so we can tell
      // whether the simd_float4x4 layout or the MSL multiply is the problem.
      for (int c = 0; c < 4; c++) {
        simd_float4 col = u.invVP.columns[c];
        fprintf(stderr, "invVP col%d = %.7f %.7f %.7f %.7f\n", c,
          col.x, col.y, col.z, col.w);
      }
      fprintf(stderr, "invVP raw = ");
      for (int i = 0; i < 16; i++) fprintf(stderr, "%.7f ", ((float*)&u.invVP)[i]);
      fprintf(stderr, "\n");
    }

    // Warmup.
    for (int f = 0; f < 10; f++) {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      if (compute) {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:cps];
        [enc setBuffer:ubuf offset:0 atIndex:0];
        [enc setTexture:volTex atIndex:0];
        [enc setTexture:rt atIndex:1];
        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake(kRT, kRT, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
      } else {
        MTLRenderPassDescriptor* rpd = MakeRPD(rt, depth, depthMode);
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
        [enc setRenderPipelineState:pso];
        [enc setFragmentBuffer:ubuf offset:0 atIndex:0];
        [enc setFragmentTexture:volTex atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
      }
      [cb commit];
      [cb waitUntilCompleted];
    }

    // Timed.
    double acc = 0.0;
    for (int f = 0; f < frames; f++) {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      if (compute) {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:cps];
        [enc setBuffer:ubuf offset:0 atIndex:0];
        [enc setTexture:volTex atIndex:0];
        [enc setTexture:rt atIndex:1];
        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake(kRT, kRT, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
      } else {
        MTLRenderPassDescriptor* rpd = MakeRPD(rt, depth, depthMode);
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
        [enc setRenderPipelineState:pso];
        [enc setFragmentBuffer:ubuf offset:0 atIndex:0];
        [enc setFragmentTexture:volTex atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
      }
      [cb commit];
      [cb waitUntilCompleted];
      double t = cb.GPUEndTime - cb.GPUStartTime;
      if (t > 0) acc += t;
    }
    fprintf(stderr, "METAL avg frame: %.3f ms\n", acc / frames * 1e3);

    // Sanity readback. The RT is BGRA8Unorm, so memory byte order is B,G,R,A
    // while the shader writes float4(R=iterLow, G=iterHigh, B=acc, A=1).
    // Memory byte0=B=acc, byte1=G=iterHigh, byte2=R=iterLow. The "true"
    // iteration count per pixel is byte2 + 256*byte1; meanB is byte0/255.
    // (A naive byte0+256*byte1 read mixes acc into the low byte and inflates
    // avgIter; GL's RGBA8 readback does not have this trap.)
    // Nonzero pixels are marched fragments. In diag mode the shader wrote
    // rayDir*0.5+0.5 instead (writes metal_gap_diag.ppm).
    uint8_t* pix = malloc((size_t)kRT * kRT * 4);
    [rt getBytes:pix bytesPerRow:kRT * 4 fromRegion:MTLRegionMake2D(0, 0, kRT, kRT) mipmapLevel:0];
    double sumB = 0.0;
    double lo = 0.0;
    double hi = 0.0;
    int nz = 0;
    for (size_t i = 0; i < (size_t)kRT * kRT * 4; i += 4) {
      sumB += pix[i + 2];
      lo += pix[i];
      hi += pix[i + 1];
      if (pix[i] + pix[i + 1] + pix[i + 2] > 0) nz++;
    }
    double npix = (double)kRT * kRT;
    double avgN = (hi / npix) * 256.0 + lo / npix;
    // The RT is BGRA8Unorm, so memory byte0=B(shader acc), byte1=G(shader
    // iterHigh), byte2=R(shader iterLow). True iteration count per pixel:
    // iter = byte2 + 256*byte1.
    double sumIter = 0.0;
    for (size_t i = 0; i < (size_t)kRT * kRT * 4; i += 4) {
      sumIter += pix[i + 2] + 256.0 * (double)pix[i + 1];
    }
    fprintf(stderr, "METAL readback: meanB=%.3f nonzero=%d/%d avgIter=%.1f (true=%.1f)\n",
      sumB / npix / 255.0, nz, kRT * kRT, avgN, sumIter / npix);
    const char* ppmName = diag ? "metal_gap_diag.ppm" : "metal_gap.ppm";
    FILE* fp = fopen(ppmName, "wb");
    if (fp) {
      fprintf(fp, "P6\n%d %d\n255\n", kRT, kRT);
      for (int i = 0; i < kRT * kRT; i++)
        fwrite(pix + i * 4, 1, 3, fp);
      fclose(fp);
    }
    free(pix);
  }
  return 0;
}
