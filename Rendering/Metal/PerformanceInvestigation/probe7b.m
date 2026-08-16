#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t fp32_to_fp16(float f) {
  uint32_t b;
  memcpy(&b, &f, 4);
  uint32_t sign = (b >> 16) & 0x8000u;
  int32_t exp = (int32_t)((b >> 23) & 0xff) - 127 + 15;
  uint32_t man = b & 0x7fffffu;
  if (((b >> 23) & 0xff) == 0xff) return sign | 0x7c00u | (man >> 13);
  if (((b >> 23) & 0xff) == 0) return sign;
  if (exp >= 31) return sign | 0x7c00u;
  if (exp <= 0) {
    if (exp < -10) return sign;
    man |= 0x800000u;
    uint32_t shift = (uint32_t)(14 - exp);
    return sign | (man >> shift);
  }
  return sign | ((uint32_t)exp << 10) | (man >> 13);
}

// Same U/Block/Lights mirrors as probe7.m (offsets verified).
typedef struct {
  float WorldToVolumeMatrix[16];
  float VolumeToWorldMatrix[16];
  float VolumeBoundsMin[4];
  float VolumeBoundsMax[4];
  float CameraVolumePos[4];
  float ViewProjectionMatrix[16];
  uint16_t SampleDistanceHalf;
  uint16_t OpacityPreIntegrationFactorHalf;
  uint16_t ScalarMinHalf;
  uint16_t _padSM;
  uint16_t ScalarMaxHalf;
  uint16_t _padSMax;
  float UseJittering;
  float InverseViewProjection[16];
  float ViewportSize[2];
  float _padViewport[2];
  float GradientStep[3];
  float _padGradStep;
  float UseGradientShading;
  float _padGradOpRange;
  float GradientOpacityMin;
  float GradientOpacityMax;
  float UseGradientOpacity;
  float _padAmbient[3];
  float AmbientColor[3];
  float _padAmb;
  float DiffuseColor[3];
  float _padDiff;
  float SpecularColor[3];
  float _padSpec;
  float Shininess;
  float _padLightDir[3];
  float LightDirection[3];
  float _padLight;
  float _padEnd[4];
  float CroppingPlanes[4];
  float CroppingPlanes2[4];
  uint32_t CroppingBitmask;
  float _padCropFlags[31];
  float UseCropping;
  float UseClipping;
  float NumClippingPlanes;
  float _padClipping[2];
  float _padClipAlign[3];
  float ClippingPlane0Origin[4];
  float ClippingPlane0Normal[4];
  float ClippingPlane1Origin[4];
  float ClippingPlane1Normal[4];
  float ClippingPlane2Origin[4];
  float ClippingPlane2Normal[4];
  float ClippingPlane3Origin[4];
  float ClippingPlane3Normal[4];
  float ClippingPlane4Origin[4];
  float ClippingPlane4Normal[4];
  float ClippingPlane5Origin[4];
  float ClippingPlane5Normal[4];
  float ClippingPlane6Origin[4];
  float ClippingPlane6Normal[4];
  float ClippingPlane7Origin[4];
  float ClippingPlane7Normal[4];
  float UseMask;
  float MaskBlendFactor;
  float MaskScale;
  float MaskBias;
  float LabelMapNumLabels;
  float UseDepthTexture;
  float UseNormalTexture;
  float UseLinearVolumeInterpolation;
  float UseMinMaxAccel;
  float MinMaxDimX;
  float MinMaxDimY;
  float MinMaxDimZ;
  float UseRenderToImage;
  float ClampDepthToBackface;
  float UseTransfer2D;
  float Transfer2DYAxisScale;
  float Transfer2DYAxisBias;
  float Transfer2DUseGradient;
  float AverageIPRangeMin;
  float AverageIPRangeMax;
  float MaskType;
  float FinalColorScale;
  float FinalColorBias;
  float UseBlanking;
  float BlankingMode;
  float _padDirAlign[3];
  float TextureToVolumeMatrix[16];
  float VolumeToTextureMatrix[16];
  uint16_t ScalarMinCompHalf[4];
  uint16_t _padSMComp[4];
  uint16_t ScalarMaxCompHalf[4];
  uint16_t _padSMaxComp[4];
  float ComponentWeight[4];
  uint32_t NumComponents;
  float UseIndependentComponents;
  float UseDependentLA;
  float _padIndependent[2];
  float UseDependentRGBA;
  float UseComputeNormalFromOpacity;
  float _padEnd2[1];
  float AmbientColorComp[4][4];
  float DiffuseColorComp[4][4];
  float SpecularColorComp[4][4];
  float ShininessComp[4];
  float SelectionMode;
  float _padSel[3];
  uint32_t SelectionPropId;
  uint32_t SelectionCompositeIndex;
  uint32_t SelectionVolumeDimX;
  uint32_t SelectionVolumeDimY;
  uint32_t SelectionVolumeDimZ;
  uint32_t _padSelEnd[3];
  float UseParallelProjection;
  float ProjectionDirection[4];
  float _padParallelEnd[3];
  float NDCToVolumeMatrix[16];
  float UseRectilinear;
  float _padRect[3];
  float RectCoordsSizes[4];
  float RectCoordsScale[4];
  float RectCoordsBias[4];
  float UseCameraInsideNearClip;
  float _padNearClip[3];
  float CameraInsideNearPlaneOrigin[4];
  float CameraInsideNearPlaneNormal[4];
  float UseDataSpaceBoxVertices;
  float UseIGNJitter;
  float JitterBlockSize;
  float MaxStepsFrame;
} U;
_Static_assert(sizeof(U) == 1728, "U must be 1728");

typedef struct {
  float VolumeBoundsMin[4];
  float VolumeBoundsMax[4];
  float TextureBoundsMin[4];
  float TextureBoundsMax[4];
  float GradientStep[4];
  float MinMaxInfo[4];
} Block;
_Static_assert(sizeof(Block) == 96, "Block must be 96");

typedef struct {
  float lights[8][24];
  int32_t lightCount;
  int32_t numPositionalLights;
  int32_t twoSidedLighting;
  int32_t defaultLighting;
  int32_t _pad[4];
} Lights;
_Static_assert(sizeof(Lights) == 800, "Lights must be 800");

static void* load_file(const char* path, size_t* outLen) {
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  void* buf = malloc((size_t)len);
  if (fread(buf, 1, (size_t)len, f) != (size_t)len) { fprintf(stderr, "short read %s\n", path); exit(1); }
  fclose(f);
  if (outLen) *outLen = (size_t)len;
  return buf;
}

static double benchGPU(id<MTLDevice> device, id<MTLCommandQueue> queue,
    id<MTLRenderPipelineState> pso, id<MTLTexture> rt, id<MTLTexture> depthTex,
    id<MTLTexture> volTex, id<MTLTexture> tfTex, id<MTLTexture> dummyVol,
    id<MTLTexture> dummyMask, id<MTLTexture> dummyMinMax,
    id<MTLBuffer> ubuf, id<MTLBuffer> bbuf, id<MTLBuffer> lbuf, id<MTLBuffer> rbuf,
    id<MTLBuffer> vbuf, id<MTLBuffer> ibuf, id<MTLDepthStencilState> ds,
    uint32_t nidx, int frames, id<MTLBuffer> cbuf, id<MTLBuffer> fixedNBuf,
    id<MTLBuffer> decompBuf, id<MTLBuffer> v25Buf, id<MTLTexture> volArrTex) {
  double acc = 0;
  for (int f = 0; f < frames; ++f) {
    if (cbuf) {
      uint32_t* c = (uint32_t*)cbuf.contents;
      c[0] = 0; c[1] = 0; c[2] = 0; c[3] = 0;
    }
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    MTLRenderPassDescriptor* rpd = [[MTLRenderPassDescriptor alloc] init];
    rpd.colorAttachments[0].texture = rt;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.1, 0.1, 0.1, 1.0);
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.depthAttachment.texture = depthTex;
    rpd.depthAttachment.loadAction = MTLLoadActionClear;
    rpd.depthAttachment.clearDepth = 1.0;
    rpd.depthAttachment.storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
    [enc setRenderPipelineState:pso];
    [enc setFrontFacingWinding:MTLWindingCounterClockwise];
    [enc setCullMode:MTLCullModeBack];
    if (ds) [enc setDepthStencilState:ds];
    [enc setVertexBuffer:vbuf offset:0 atIndex:0];
    [enc setVertexBuffer:ubuf offset:0 atIndex:1];
    [enc setVertexBuffer:bbuf offset:0 atIndex:2];
    [enc setFragmentBuffer:ubuf offset:0 atIndex:1];
    [enc setFragmentBuffer:bbuf offset:0 atIndex:2];
    [enc setFragmentBuffer:lbuf offset:0 atIndex:4];
    [enc setFragmentBuffer:rbuf offset:0 atIndex:5];
    [enc setFragmentBuffer:cbuf offset:0 atIndex:6];
    if (fixedNBuf) [enc setFragmentBuffer:fixedNBuf offset:0 atIndex:3];
    if (decompBuf) [enc setFragmentBuffer:decompBuf offset:0 atIndex:7];
    if (v25Buf) [enc setFragmentBuffer:v25Buf offset:0 atIndex:8];
    [enc setFragmentTexture:volTex atIndex:0];
    [enc setFragmentTexture:tfTex atIndex:1];
    [enc setFragmentTexture:depthTex atIndex:2];
    [enc setFragmentTexture:tfTex atIndex:3];
    [enc setFragmentTexture:dummyMask atIndex:4];
    [enc setFragmentTexture:tfTex atIndex:5];
    [enc setFragmentTexture:dummyMinMax atIndex:6];
    [enc setFragmentTexture:dummyVol atIndex:7];
    [enc setFragmentTexture:tfTex atIndex:9];
    [enc setFragmentTexture:dummyVol atIndex:10];
    [enc setFragmentTexture:dummyVol atIndex:11];
    [enc setFragmentTexture:tfTex atIndex:12];
    [enc setFragmentTexture:tfTex atIndex:13];
    [enc setFragmentTexture:tfTex atIndex:14];
    if (volArrTex) [enc setFragmentTexture:volArrTex atIndex:15];
    [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:nidx
        indexType:MTLIndexTypeUInt32 indexBuffer:ibuf indexBufferOffset:0];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    double t = cb.GPUEndTime - cb.GPUStartTime;
    if (t > 0) acc += t;
  }
  return acc / frames;
}

int main(int argc, const char** argv) {
  // variant: 0 = full march (probe7 parity), 1 = march-only (no TF/composite/bounds)
  int variant = argc > 1 ? atoi(argv[1]) : 0;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    fprintf(stderr, "device: %s\n", device.name.UTF8String);
    NSError* err = nil;

    // Base directory for the .metal sources and dicom.u8. Defaults to the
    // directory containing the executable; override with PROBE_BASE.
    NSString* base;
    const char* baseEnv = getenv("PROBE_BASE");
    if (baseEnv && baseEnv[0]) {
      base = [NSString stringWithUTF8String:baseEnv];
    } else {
      base = [NSString stringWithUTF8String:argv[0]];
      base = [base stringByDeletingLastPathComponent];
    }
    NSString* libSrc = [NSString stringWithContentsOfFile:[base stringByAppendingPathComponent:@"probe6lib.metal"]
        encoding:NSUTF8StringEncoding error:&err];
    NSString* extra = [NSString stringWithContentsOfFile:[base stringByAppendingPathComponent:@"probe7_extra.metal"]
        encoding:NSUTF8StringEncoding error:&err];
    NSString* src = [libSrc stringByAppendingString:extra];
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
    if (!lib) { fprintf(stderr, "lib compile err: %s\n", err.description.UTF8String); return 1; }
    fprintf(stderr, "lib compiled\n");

    id<MTLFunction> frag = nil;
    MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
    BOOL bNo = NO, bYes = YES;
    [cv setConstantValue:&bNo type:MTLDataTypeBool withName:@"fc_shading"];
    [cv setConstantValue:&bNo type:MTLDataTypeBool withName:@"fc_gradientOpacity"];
    [cv setConstantValue:&bNo type:MTLDataTypeBool withName:@"fc_mask"];
    [cv setConstantValue:&bNo type:MTLDataTypeBool withName:@"fc_minmax"];
    [cv setConstantValue:&bNo type:MTLDataTypeBool withName:@"fc_normalTexture"];
    [cv setConstantValue:&bYes type:MTLDataTypeBool withName:@"fc_linearInterpolation"];
    [cv setConstantValue:&bNo type:MTLDataTypeBool withName:@"fc_computeNormalFromOpacity"];
    [cv setConstantValue:&bNo type:MTLDataTypeBool withName:@"fc_independentComponents"];
    [cv setConstantValue:&bNo type:MTLDataTypeBool withName:@"fc_transfer2D"];
    [cv setConstantValue:&bNo type:MTLDataTypeBool withName:@"fc_rectilinear"];
    [cv setConstantValue:&bNo type:MTLDataTypeBool withName:@"fc_defaultLighting"];
    int lightCount = 0;
    [cv setConstantValue:&lightCount type:MTLDataTypeInt withName:@"fc_lightCount"];
    [cv setConstantValue:&bNo type:MTLDataTypeBool withName:@"fc_dependentRGBA"];
    [cv setConstantValue:&bNo type:MTLDataTypeBool withName:@"fc_dependentLA"];
    [cv setConstantValue:&bNo type:MTLDataTypeBool withName:@"fc_renderToTexture"];
    [cv setConstantValue:&bNo type:MTLDataTypeBool withName:@"fc_cropping"];
    [cv setConstantValue:&bNo type:MTLDataTypeBool withName:@"fc_blanking"];
    int blendMode = 0;
    [cv setConstantValue:&blendMode type:MTLDataTypeInt withName:@"fc_blendMode"];
    int marchVariant = 0;
    const char* mvenv = getenv("PROBE_MARCH_VARIANT");
    if (mvenv) marchVariant = atoi(mvenv);
    [cv setConstantValue:&marchVariant type:MTLDataTypeInt withName:@"fc_marchVariant"];
    fprintf(stderr, "fc_marchVariant=%d\n", marchVariant);
    bool probeMinimal = false;
    const char* pmenv = getenv("PROBE_MINIMAL");
    if (pmenv) probeMinimal = atoi(pmenv) != 0;
    [cv setConstantValue:&probeMinimal type:MTLDataTypeBool withName:@"fc_probeMinimal"];
    if (probeMinimal) fprintf(stderr, "fc_probeMinimal=1\n");
    bool probePrefetch = false;
    const char* ppenv = getenv("PROBE_PREFETCH");
    if (ppenv) probePrefetch = atoi(ppenv) != 0;
    [cv setConstantValue:&probePrefetch type:MTLDataTypeBool withName:@"fc_probePrefetch"];
    if (probePrefetch) fprintf(stderr, "fc_probePrefetch=1\n");
    bool probeScalars = false;
    const char* scenv = getenv("PROBE_SCALARS");
    if (scenv) probeScalars = atoi(scenv) != 0;
    [cv setConstantValue:&probeScalars type:MTLDataTypeBool withName:@"fc_probeScalars"];
    if (probeScalars) fprintf(stderr, "fc_probeScalars=1\n");
    bool probeSlim = false;
    const char* slenv = getenv("PROBE_SLIM");
    if (slenv) probeSlim = atoi(slenv) != 0;
    [cv setConstantValue:&probeSlim type:MTLDataTypeBool withName:@"fc_probeSlim"];
    if (probeSlim) fprintf(stderr, "fc_probeSlim=1\n");
    const char* fragNames[] = {"fragment_volume_main", "fragment_march_only",
        "fragment_march_no_fetch", "fragment_march_only_nearest", "fragment_count_steps",
        "fragment_debug_steps", "fragment_march_linear_implicit",
        "fragment_march_linear_fixedN", "fragment_march_nearest_fixedN",
        "fragment_march_linear_pipe2", "fragment_march_linear_pipe3",
        "fragment_march_manual_trilinear", "fragment_march_linear_clampZero",
        "fragment_march_linear_2Darray", "fragment_march_linear_repeat", "fragment_fixedpoint_linear", "fragment_fixedpoint_nearest",         "fragment_march_linear_select", "fragment_march_xdir_linear", "fragment_march_xdir_linear_counted",         "fragment_march_real_decomp", "fragment_march_linear_clamp", "fragment_march_nearest_clamp", "fragment_march_xybilinear_znearest_clone", "fragment_march_nearest_clamp_double", "fragment_march_decomp_unrolled", "fragment_march_decomp_array", "fragment_march_decomp_deadarr", "fragment_march_decomp_deadarr2",         "fragment_march_phase_batch", "fragment_march_phase_batch_scalar", "fragment_march_phase_batch_sched", "fragment_march_phase_batch_lean", "fragment_march_phase_batch_w16", "fragment_march_phase_batch_w32", "fragment_march_phase_batch_w48", "fragment_march_phase_batch_w64"};
    if (variant < 0 || variant > 36) variant = 0;
    NSString* fragName = [NSString stringWithUTF8String:fragNames[variant]];
    frag = [lib newFunctionWithName:fragName constantValues:cv error:&err];
    if (!frag) { fprintf(stderr, "frag func err: %s\n", err.description.UTF8String); return 1; }
    id<MTLFunction> vert = [lib newFunctionWithName:@"vertex_volume_main"];
    if (!vert) { fprintf(stderr, "vert func err\n"); return 1; }
    fprintf(stderr, "functions created\n");

    MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = vert;
    pd.fragmentFunction = frag;
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pd.colorAttachments[0].blendingEnabled = YES;
    pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
    pd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pd.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    pd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    pd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pd.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    MTLVertexDescriptor* vd = [[MTLVertexDescriptor alloc] init];
    vd.attributes[0].format = MTLVertexFormatFloat3;
    vd.attributes[0].offset = 0;
    vd.attributes[0].bufferIndex = 0;
    vd.layouts[0].stride = 12;
    vd.layouts[0].stepRate = 1;
    vd.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    pd.vertexDescriptor = vd;
    NSError* pe = nil;
    id<MTLRenderPipelineState> pso = [device newRenderPipelineStateWithDescriptor:pd error:&pe];
    if (!pso) { fprintf(stderr, "pso err: %s\n", pe.description.UTF8String); return 1; }
    fprintf(stderr, "pso created\n");

    MTLDepthStencilDescriptor* dsd = [[MTLDepthStencilDescriptor alloc] init];
    dsd.depthCompareFunction = MTLCompareFunctionLessEqual;
    dsd.depthWriteEnabled = NO;
    id<MTLDepthStencilState> ds = [device newDepthStencilStateWithDescriptor:dsd];

    U u;
    size_t ulen = 0;
    void* ub = load_file("/tmp/app_uniforms.bin", &ulen);
    memcpy(&u, ub, sizeof(u));

    const char* sdenv = getenv("PROBE_SAMPLE_DISTANCE_MM");
    if (sdenv) {
      float sdmm = (float)atof(sdenv);
      float maxB = fmaxf(u.VolumeBoundsMax[1], fmaxf(u.VolumeBoundsMax[0], u.VolumeBoundsMax[2]));
      float half = (sdmm / 2.0f) / maxB;
      uint32_t halfBits = fp32_to_fp16(half);
      u.SampleDistanceHalf = (uint16_t)halfBits;
      fprintf(stderr, "override sampleDistance=%.3fmm -> half 0x%04x\n", sdmm, u.SampleDistanceHalf);
    }

    Block blk;
    size_t blen = 0;
    void* bb = load_file("/tmp/app_pbd.bin", &blen);
    memcpy(&blk, bb, sizeof(blk));

    Lights l;
    FILE* lf = fopen("/tmp/app_light.bin", "rb");
    fread(&l, 1, sizeof(l), lf); fclose(lf);

    size_t nverts = 0, nidx = 0;
    void* vbufData = load_file("/tmp/app_verts.bin", &nverts);
    void* ibufData = load_file("/tmp/app_idxs.bin", &nidx);
    uint32_t ni = (uint32_t)(nidx / 4);
    id<MTLBuffer> vbuf = [device newBufferWithBytes:vbufData length:nverts options:MTLResourceStorageModeShared];
    id<MTLBuffer> ibuf = [device newBufferWithBytes:ibufData length:nidx options:MTLResourceStorageModeShared];

    id<MTLCommandQueue> queue = [device newCommandQueue];
    uint32_t TW = 512, TH = 512, TD = 1794;
    size_t total = (size_t)TW * TH * TD;
    FILE* f = fopen([[base stringByAppendingPathComponent:@"dicom.u8"] UTF8String], "rb");
    uint8_t* host = malloc(total);
    size_t got = fread(host, 1, total, f); fclose(f);
    fprintf(stderr, "loaded %zu bytes\n", got);
    // Optional transpose (PROBE_TRANSPOSE=1): store the volume so the long
    // (slice) axis becomes X, reproducing the app's VTK_METAL_TEST_TRANSPOSE
    // layout. Dump the app bins with the same transpose set so the uniforms
    // match. Data reorder: new(x',y,z') = old(x=z', y, z=x').
    if (getenv("PROBE_TRANSPOSE")) {
      uint32_t TW2 = TD, TH2 = TH, TD2 = TW;
      uint8_t* thost = malloc(total);
      for (uint32_t xp = 0; xp < TW2; ++xp)
        for (uint32_t yp = 0; yp < TH2; ++yp)
          for (uint32_t zp = 0; zp < TD2; ++zp)
            thost[((zp * TH2) + yp) * TW2 + xp] = host[((xp * TH2) + yp) * TW + zp];
      free(host);
      host = thost;
      TW = TW2; TH = TH2; TD = TD2;
      fprintf(stderr, "transposed upload dims %ux%ux%u\n", TW, TH, TD);
    }
    id<MTLBuffer> staging = [device newBufferWithBytes:host length:total options:MTLResourceStorageModeShared];
    MTLTextureDescriptor* d = [[MTLTextureDescriptor alloc] init];
    d.textureType = MTLTextureType3D; d.pixelFormat = MTLPixelFormatR8Unorm;
    d.width = TW; d.height = TH; d.depth = TD; d.mipmapLevelCount = 1;
    d.usage = MTLTextureUsageShaderRead; d.storageMode = MTLStorageModePrivate;
    id<MTLTexture> volTex = [device newTextureWithDescriptor:d];
    id<MTLCommandBuffer> cb0 = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb0 blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 sourceBytesPerRow:TW sourceBytesPerImage:TW*TH
            sourceSize:MTLSizeMake(TW, TH, TD) toTexture:volTex
            destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
    [blit endEncoding]; [cb0 commit]; [cb0 waitUntilCompleted];
    fprintf(stderr, "volume uploaded\n");

    // 2D array texture (Apple-recommended volume layout): slices of 512x512.
    MTLTextureDescriptor* ad = [[MTLTextureDescriptor alloc] init];
    ad.textureType = MTLTextureType2DArray; ad.pixelFormat = MTLPixelFormatR8Unorm;
    ad.width = TW; ad.height = TH; ad.arrayLength = TD; ad.mipmapLevelCount = 1;
    ad.usage = MTLTextureUsageShaderRead; ad.storageMode = MTLStorageModePrivate;
    id<MTLTexture> volArrTex = [device newTextureWithDescriptor:ad];
    id<MTLCommandBuffer> cbA = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blitA = [cbA blitCommandEncoder];
    [blitA copyFromBuffer:staging sourceOffset:0 sourceBytesPerRow:TW sourceBytesPerImage:TW*TH
            sourceSize:MTLSizeMake(TW, TH, 1) toTexture:volArrTex
            destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
    [blitA endEncoding]; [cbA commit]; [cbA waitUntilCompleted];
    // Copy remaining slices.
    for (int z = 1; z < TD; ++z) {
      id<MTLCommandBuffer> cbz = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blitz = [cbz blitCommandEncoder];
      [blitz copyFromBuffer:staging sourceOffset:(size_t)z*TW*TH sourceBytesPerRow:TW sourceBytesPerImage:TW*TH
              sourceSize:MTLSizeMake(TW, TH, 1) toTexture:volArrTex
              destinationSlice:z destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
      [blitz endEncoding]; [cbz commit]; [cbz waitUntilCompleted];
    }
    fprintf(stderr, "2darray uploaded (%d slices)\n", TD);

    MTLTextureDescriptor* tfD = [[MTLTextureDescriptor alloc] init];
    tfD.textureType = MTLTextureType2D; tfD.pixelFormat = MTLPixelFormatRGBA8Unorm;
    tfD.width = 256; tfD.height = 2; tfD.mipmapLevelCount = 1;
    tfD.usage = MTLTextureUsageShaderRead; tfD.storageMode = MTLStorageModeShared;
    id<MTLTexture> tfTex = [device newTextureWithDescriptor:tfD];
    uint8_t tfBuf[256*2*4];
    for (int x = 0; x < 256; ++x) {
      float v = (float)x / 255.0f;
      for (int y = 0; y < 2; ++y) {
        int i = (y*256 + x) * 4;
        tfBuf[i+0] = (uint8_t)(v*255); tfBuf[i+1] = (uint8_t)((1-v)*255);
        tfBuf[i+2] = (uint8_t)128;     tfBuf[i+3] = 0;
      }
    }
    [tfTex replaceRegion:MTLRegionMake2D(0,0,256,2) mipmapLevel:0 withBytes:tfBuf bytesPerRow:256*4];

    MTLTextureDescriptor* d1 = [[MTLTextureDescriptor alloc] init];
    d1.textureType = MTLTextureType3D; d1.pixelFormat = MTLPixelFormatR8Unorm;
    d1.width = 1; d1.height = 1; d1.depth = 1; d1.mipmapLevelCount = 1;
    d1.usage = MTLTextureUsageShaderRead; d1.storageMode = MTLStorageModeShared;
    id<MTLTexture> dummyVol = [device newTextureWithDescriptor:d1];
    uint8_t one = 255;
    [dummyVol replaceRegion:MTLRegionMake3D(0,0,0,1,1,1) mipmapLevel:0 slice:0 withBytes:&one bytesPerRow:1 bytesPerImage:1];
    id<MTLTexture> dummyMask = [device newTextureWithDescriptor:d1];
    id<MTLTexture> dummyMinMax = [device newTextureWithDescriptor:d1];

    uint32_t RTW = (uint32_t)u.ViewportSize[0], RTH = (uint32_t)u.ViewportSize[1];
    MTLTextureDescriptor* rtD = [[MTLTextureDescriptor alloc] init];
    rtD.textureType = MTLTextureType2D; rtD.pixelFormat = MTLPixelFormatBGRA8Unorm;
    rtD.width = RTW; rtD.height = RTH; rtD.mipmapLevelCount = 1;
    rtD.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    rtD.storageMode = MTLStorageModePrivate;
    id<MTLTexture> rt = [device newTextureWithDescriptor:rtD];
    MTLTextureDescriptor* depD = [[MTLTextureDescriptor alloc] init];
    depD.textureType = MTLTextureType2D; depD.pixelFormat = MTLPixelFormatDepth32Float;
    depD.width = RTW; depD.height = RTH; depD.mipmapLevelCount = 1;
    depD.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    depD.storageMode = MTLStorageModePrivate;
    id<MTLTexture> depthTex = [device newTextureWithDescriptor:depD];

    id<MTLBuffer> ubuf = [device newBufferWithBytes:&u length:sizeof(u) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bbuf = [device newBufferWithBytes:&blk length:sizeof(blk) options:MTLResourceStorageModeShared];
    id<MTLBuffer> lbuf = [device newBufferWithBytes:&l length:sizeof(l) options:MTLResourceStorageModeShared];
    float rectCoords[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    id<MTLBuffer> rbuf = [device newBufferWithBytes:rectCoords length:sizeof(rectCoords) options:MTLResourceStorageModeShared];
    id<MTLBuffer> cbuf = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];

    int fixedN = 659;
    const char* nenv = getenv("PROBE_FIXED_N");
    if (nenv) fixedN = atoi(nenv);
    id<MTLBuffer> fixedNBuf = [device newBufferWithBytes:&fixedN length:4 options:MTLResourceStorageModeShared];

    int decomp = 0;
    const char* denv = getenv("PROBE_DECOMP");
    if (denv) decomp = atoi(denv);
    id<MTLBuffer> decompBuf = [device newBufferWithBytes:&decomp length:4 options:MTLResourceStorageModeShared];

    int v25mode = 0;
    const char* v25env = getenv("PROBE_V25");
    if (v25env) v25mode = atoi(v25env);
    id<MTLBuffer> v25Buf = [device newBufferWithBytes:&v25mode length:4 options:MTLResourceStorageModeShared];
    if (variant == 25 || variant == 29 || variant == 30 || variant == 31 || variant == 32 || variant == 33 || variant == 34 || variant == 35 || variant == 36) fprintf(stderr, "fc_v25mode=%d\n", v25mode);

    for (int i = 0; i < 10; ++i) {
      benchGPU(device, queue, pso, rt, depthTex, volTex, tfTex, dummyVol, dummyMask, dummyMinMax,
          ubuf, bbuf, lbuf, rbuf, vbuf, ibuf, ds, ni, 1, cbuf, fixedNBuf, decompBuf, v25Buf, volArrTex);
    }
    double t = benchGPU(device, queue, pso, rt, depthTex, volTex, tfTex, dummyVol, dummyMask, dummyMinMax,
        ubuf, bbuf, lbuf, rbuf, vbuf, ibuf, ds, ni, 100, cbuf, fixedNBuf, decompBuf, v25Buf, volArrTex);
    fprintf(stderr, "variant %d: %.2f ms\n", variant, t*1e3);

    if (variant == 4 || variant == 19) {
      benchGPU(device, queue, pso, rt, depthTex, volTex, tfTex, dummyVol, dummyMask, dummyMinMax,
          ubuf, bbuf, lbuf, rbuf, vbuf, ibuf, ds, ni, 1, cbuf, fixedNBuf, decompBuf, v25Buf, volArrTex);
      uint32_t* c = (uint32_t*)cbuf.contents;
      double total = c[0]; double totalPx = c[1]; double marched = c[2];
      fprintf(stderr, "count: fragments=%g marched_pixels=%.0f avg_steps=%.2f total_iters=%.0f\n",
          totalPx, marched, total/marched, total);
    } else if (variant == 5) {
      benchGPU(device, queue, pso, rt, depthTex, volTex, tfTex, dummyVol, dummyMask, dummyMinMax,
          ubuf, bbuf, lbuf, rbuf, vbuf, ibuf, ds, ni, 1, cbuf, fixedNBuf, decompBuf, v25Buf, volArrTex);
      size_t rtsz = (size_t)RTW * RTH * 4;
      id<MTLBuffer> rtb = [device newBufferWithLength:rtsz options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLBlitCommandEncoder> bl2 = [cb blitCommandEncoder];
      [bl2 copyFromTexture:rt sourceSlice:0 sourceLevel:0
              sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(RTW,RTH,1)
              toBuffer:rtb destinationOffset:0 destinationBytesPerRow:RTW*4 destinationBytesPerImage:rtsz];
      [bl2 endEncoding]; [cb commit]; [cb waitUntilCompleted];
      uint8_t* px = (uint8_t*)rtb.contents;
      double nHit = 0, stepSum = 0, countSum = 0;
      uint64_t maxN = 0;
      double minStep = 1e9, maxStep = 0;
      for (uint32_t i = 0; i < (uint32_t)(RTW*RTH); ++i) {
        int n = (int)px[i*4] + ((int)px[i*4+1] << 8);
        float step = (float)px[i*4+2] / 255.0f * 0.002f;
        if (!(px[i*4] == 0 && px[i*4+1] == 0 && px[i*4+2] == 255)) { // not the 'miss' blue
          nHit++; countSum += n; stepSum += step;
          if ((uint64_t)n > maxN) maxN = (uint64_t)n;
          if (step < minStep) minStep = step;
          if (step > maxStep) maxStep = step;
        }
      }
      fprintf(stderr, "debug: hit=%.0f avgN=%.2f maxN=%llu step[min=%.6f max=%.6f avg=%.6f]\n",
          nHit, countSum/nHit, (unsigned long long)maxN, minStep, maxStep, stepSum/nHit);
    }

    free(host);
  }
  return 0;
}
