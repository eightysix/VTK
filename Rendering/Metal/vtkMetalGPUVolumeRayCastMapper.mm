// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalGPUVolumeRayCastMapper.h"

#include "vtkMetalRenderWindow.h"
#include "vtkMetalRenderer.h"
#include "vtkMetalCamera.h"
#include "vtkMetalShaders.h"
#include "vtkColorTransferFunction.h"
#include "vtkImageData.h"
#include "vtkObjectFactory.h"
#include "vtkPiecewiseFunction.h"
#include "vtkPointData.h"
#include "vtkRenderer.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkCamera.h"
#include "vtkMatrix4x4.h"
#include "vtkSMPTools.h"
#include "vtkMath.h"
#include "vtkClipConvexPolyData.h"
#include "vtkDensifyPolyData.h"
#include "vtkTriangleFilter.h"
#include "vtkPlaneCollection.h"
#include "vtkPlane.h"
#include "vtkPolyData.h"
#include "vtkPoints.h"
#include "vtkCellArray.h"
#include <limits>

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <Accelerate/Accelerate.h>

#include <algorithm>
#include <cstring>
#include <cstddef>
#include <set>
#include <vector>
#include <dispatch/dispatch.h>

// Metal constant-address-space structs align float3 to 16 bytes (size 16),
// float4/float4x4 to 16 bytes, and float2 to 8 bytes.  This creates
// padding that plain C++ float[] arrays do not.  The layout below exactly
// mirrors the Metal compiler's computation (480 bytes total).
struct VolumeMapperUniforms
{
  // --- fields matching Metal layout 1:1, offsets verified ---
  float WorldToVolumeMatrix[16];     // 0..63
  float VolumeToWorldMatrix[16];     // 64..127
  float VolumeBoundsMin[4];          // 128..143
  float VolumeBoundsMax[4];          // 144..159
  float CameraVolumePos[4];          // 160..175
  float ViewProjectionMatrix[16];    // 176..239
  float SampleDistance;              // 240
  float ScalarMin;                   // 244
  float ScalarMax;                   // 248
  float UseJittering;                // 252
  float InverseViewProjection[16];   // 256..319
  float ViewportSize[2];            // 320..327
  float _padViewport[2];            // 328..335  (pad to 16-byte for float3)
  float GradientStep[3];            // 336..347
  float _padGradStep;               // 348..351  (Metal: float3 = 16 bytes)
  float UseGradientShading;         // 352
  float _padGradOpRange;            // 356..359  (pad to 8-byte for float2)
  float GradientOpacityMin;         // 360
  float GradientOpacityMax;         // 364
  float UseGradientOpacity;         // 368
  float _padAmbient[3];             // 372..383  (pad to 16-byte for float4)
  float AmbientColor[3];            // 384..395
  float _padAmb;                    // 396..399  (Metal: float4 = 16 bytes)
  float DiffuseColor[3];            // 400..411
  float _padDiff;                   // 412..415
  float SpecularColor[3];           // 416..427
  float _padSpec;                   // 428..431
  float Shininess;                  // 432
  float _padLightDir[3];            // 436..447  (pad to 16-byte for float3)
  float LightDirection[3];          // 448..459
  float _padLight;                  // 460..463  (Metal: float3 = 16 bytes)
  float _padEnd[4];                 // 464..479  (trailing pad to 480)
  // Cropping regions (new)
  float CroppingPlanes[4];          // 480..495  (minX, maxX, minY, maxY)
  float CroppingPlanes2[4];         // 496..511  (minZ, maxZ, 0, 0)
  uint32_t CroppingBitmask;         // 512..515  (packed bitmask from GetCroppingRegionFlags)
  float _padCropFlags[31];          // 516..639  (maintain total struct size)
  float UseCropping;                // 640
  float UseClipping;                // 644
  float NumClippingPlanes;          // 648
  float _padClipping[2];            // 652..659
  float _padClipAlign[3];           // 660..671
  // Clipping planes (up to 8 arbitrary planes)
  float ClippingPlane0Origin[4];    // 672..687 (origin.xyz, 1.0)
  float ClippingPlane0Normal[4];    // 688..703 (normal.xyz, 0.0)
  float ClippingPlane1Origin[4];    // 704..719
  float ClippingPlane1Normal[4];    // 720..735
  float ClippingPlane2Origin[4];    // 736..751
  float ClippingPlane2Normal[4];    // 752..767
  float ClippingPlane3Origin[4];    // 768..783
  float ClippingPlane3Normal[4];    // 784..799
  float ClippingPlane4Origin[4];    // 800..815
  float ClippingPlane4Normal[4];    // 816..831
  float ClippingPlane5Origin[4];    // 832..847
  float ClippingPlane5Normal[4];    // 848..863
  float ClippingPlane6Origin[4];    // 864..879
  float ClippingPlane6Normal[4];    // 880..895
  float ClippingPlane7Origin[4];    // 896..911
  float ClippingPlane7Normal[4];    // 912..927
  // Mask / label map support
  float UseMask;                  // 928
  float MaskBlendFactor;          // 932
  float MaskScale;                // 936
  float MaskBias;                 // 940
  float LabelMapNumLabels;        // 944
  float UseDepthTexture;          // 948
  float UseNormalTexture;         // 952
  float _padMask;                 // 956
  // Min-max acceleration texture
  float UseMinMaxAccel;           // 960
  float MinMaxDimX;               // 964
  float MinMaxDimY;               // 968
  float MinMaxDimZ;               // 972
};

static_assert(sizeof(VolumeMapperUniforms) == 976,
  "VolumeMapperUniforms must be 976 bytes to match Metal shader struct");

static_assert(offsetof(VolumeMapperUniforms, UseCropping) == 640, "");
static_assert(offsetof(VolumeMapperUniforms, UseClipping) == 644, "");
static_assert(offsetof(VolumeMapperUniforms, NumClippingPlanes) == 648, "");
static_assert(offsetof(VolumeMapperUniforms, ClippingPlane0Origin) == 672, "");
static_assert(offsetof(VolumeMapperUniforms, UseMask) == 928, "");
static_assert(offsetof(VolumeMapperUniforms, UseDepthTexture) == 948, "");
static_assert(offsetof(VolumeMapperUniforms, UseNormalTexture) == 952, "");
static_assert(offsetof(VolumeMapperUniforms, UseMinMaxAccel) == 960, "");

// Per-block data for volume rendering — must match Metal PerBlockData struct
struct PerBlockData {
  float VolumeBoundsMin[4]; // 0..15
  float VolumeBoundsMax[4]; // 16..31
  float TextureBoundsMin[4]; // 32..47
  float TextureBoundsMax[4]; // 48..63
  float GradientStep[4];    // 64..79  (xyz + pad)
  float MinMaxInfo[4];      // 80..95  (useMinMax, dimX, dimY, dimZ)
};

static_assert(sizeof(PerBlockData) == 96,
  "PerBlockData must be 96 bytes to match Metal shader struct");

// Maximum number of bricks covered by the order-independent layer composite.
// Capped at 8 to stay within Metal's texture bind limit on older hardware.
static const size_t MAX_LAYER_BRICKS = 8;

// Must match Metal LayerCompositeUniforms struct
struct LayerCompositeUniforms {
  float BlockMin[8][4];
  float BlockMax[8][4];
  float Params[4]; // x = brickCount, y = usePerPixelOrder (1=fix, 0=cpu order), zw unused
};

// Must match Metal VolumeConvertUniforms struct (Phase 7: GPU data type conversion)
struct VolumeConvertUniforms {
  uint32_t dimX, dimY, dimZ;
  uint32_t numComponents;
  uint32_t outputComponents;
  uint32_t _pad;
};
static_assert(sizeof(VolumeConvertUniforms) == 6 * 4, "VolumeConvertUniforms size must match Metal struct");

// Must match Metal MinMaxComputeUniforms struct (Phase 5: GPU min-max)
struct MinMaxComputeUniforms {
  uint32_t mmDimX, mmDimY, mmDimZ;
  uint32_t volDimX, volDimY, volDimZ;
  float ds;
  float scalarMin;
  float scalarScale;
  float _pad;
  uint32_t opacityPrefix[257];
};

static_assert(sizeof(MinMaxComputeUniforms) == 4*6 + 4 + 4 + 4 + 4 + 257*4,
  "MinMaxComputeUniforms size must match Metal struct");

namespace
{
inline uint16_t FloatToHalf(float f)
{
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  uint32_t sign = (bits >> 16) & 0x8000;
  int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
  uint32_t mantissa = bits & 0x7FFFFF;

  if ((bits & 0x7F800000) == 0x7F800000)
  {
    uint16_t halfMantissa = mantissa >> 13;
    return static_cast<uint16_t>(sign | 0x7C00 | halfMantissa);
  }

  if (exponent <= 0)
  {
    if (exponent < -10)
      return static_cast<uint16_t>(sign);
    mantissa = (mantissa | 0x800000) >> (1 - exponent);
    return static_cast<uint16_t>(sign | (mantissa >> 13));
  }
  if (exponent > 30)
    return static_cast<uint16_t>(sign | 0x7C00);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

// Returns true when half-float can safely represent the full scalar range.
inline bool HalfRangeIsSafe(double r0, double r1)
{
  const double halfMax = 65504.0;
  return std::isfinite(r0) && std::isfinite(r1) &&
    r0 >= -halfMax && r1 <= halfMax;
}

// Returns true when the data is a single-component float array that can be
// converted to half-precision in-place using Accelerate (vImage).
inline bool IsContiguousScalarFloat(int dataType, int numComponents, vtkIdType numTuples,
  const void* ptr, size_t bytesPerVoxel)
{
  return dataType == VTK_FLOAT && numComponents == 1 &&
    bytesPerVoxel == sizeof(float) &&
    ptr == static_cast<const float*>(ptr); // alignment check
}

// Release a Metal object held as a void* member (MRC helper).
// Uses -release rather than CFRelease for proper Objective-C semantics.
inline void ReleaseMetalObject(void*& obj)
{
  if (obj)
  {
    [(__bridge id)obj release];
    obj = nullptr;
  }
}

// Takes ownership of a +1 Metal object into a void* member slot.
// Releases the previous occupant if any.
inline void AssignMetalObject(void*& slot, id obj)
{
  if (slot == (__bridge void*)obj)
  {
    return;
  }
  if (slot)
  {
    [(__bridge id)slot release];
  }
  slot = (__bridge void*)obj;
}
}

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalGPUVolumeRayCastMapper);

//------------------------------------------------------------------------------
vtkMetalGPUVolumeRayCastMapper::vtkMetalGPUVolumeRayCastMapper()
{
  this->SampleDistance = 1.0f;
  this->FrameSemaphore = dispatch_semaphore_create(3);
}

//------------------------------------------------------------------------------
vtkMetalGPUVolumeRayCastMapper::~vtkMetalGPUVolumeRayCastMapper()
{
  this->WaitForInFlightFrames();
  this->ReleaseGraphicsResources(nullptr);
#if !__has_feature(objc_arc)
  if (this->FrameSemaphore)
  {
    dispatch_release((dispatch_semaphore_t)this->FrameSemaphore);
    this->FrameSemaphore = nullptr;
  }
#endif
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::WaitForInFlightFrames()
{
  if (!this->FrameSemaphore)
  {
    return;
  }
  dispatch_semaphore_t sem = (__bridge dispatch_semaphore_t)this->FrameSemaphore;
  for (int i = 0; i < 3; ++i)
  {
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
  }
  // Restore the semaphore to valid state so a future GPURender (e.g. after
  // ReleaseGraphicsResources for a window resize) does not deadlock.
  for (int i = 0; i < 3; ++i)
  {
    dispatch_semaphore_signal(sem);
  }
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "PreferHalfPrecision: " << this->PreferHalfPrecision << "\n";
  os << indent << "UseGPUMinMax: " << this->UseGPUMinMax << "\n";
  os << indent << "UseFullscreenCameraInside: " << this->UseFullscreenCameraInside << "\n";
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::EnsureShaderLibrary(void* mtlDeviceVoid)
{
  if (this->CachedShaderLibrary)
  {
    return true;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;
    NSError* error = nil;
    NSString* shaderSource = [NSString stringWithUTF8String:vtkMetalShaders];
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];

    if (!library)
    {
      vtkErrorMacro(<< "Failed to compile Metal shader library: "
                    << [[error localizedDescription] UTF8String]);
      return false;
    }

    AssignMetalObject(this->CachedShaderLibrary, library);
  }

  return true;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::SetPartitions(
  unsigned short x, unsigned short y, unsigned short z)
{
  if (x > 0 && y > 0 && z > 0)
  {
    this->Partitions[0] = x;
    this->Partitions[1] = y;
    this->Partitions[2] = z;
  }
  else
  {
    this->Partitions[0] = this->Partitions[1] = this->Partitions[2] = 1;
  }
  this->Modified();
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ComputeReductionFactor(double allocatedTime)
{
  if (!this->AutoAdjustSampleDistances)
  {
    this->ReductionFactor = 1.0 / this->ImageSampleDistance;
    return;
  }

  if (this->TimeToDraw)
  {
    double oldFactor = this->ReductionFactor;

    double timeToDraw;
    if (allocatedTime < 1.0)
    {
      timeToDraw = this->SmallTimeToDraw;
      if (timeToDraw == 0.0)
      {
        timeToDraw = this->BigTimeToDraw / 3.0;
      }
    }
    else
    {
      timeToDraw = this->BigTimeToDraw;
    }

    if (timeToDraw == 0.0)
    {
      timeToDraw = 10.0;
    }

    double fullTime = timeToDraw / this->ReductionFactor;
    double newFactor = allocatedTime / fullTime;

    this->ReductionFactor = (newFactor + oldFactor) / 2.0;

    // Discretize to avoid visual oscillation
    this->ReductionFactor = (this->ReductionFactor > 1.0) ? 1.0 : (this->ReductionFactor);

    if (this->ReductionFactor < 0.20)
    {
      this->ReductionFactor = 0.10;
    }
    else if (this->ReductionFactor < 0.50)
    {
      this->ReductionFactor = 0.20;
    }
    else if (this->ReductionFactor < 1.0)
    {
      this->ReductionFactor = 0.50;
    }

    // Clamp to user-specified bounds
    if (1.0 / this->ReductionFactor > this->MaximumImageSampleDistance)
    {
      this->ReductionFactor = 1.0 / this->MaximumImageSampleDistance;
    }
    if (1.0 / this->ReductionFactor < this->MinimumImageSampleDistance)
    {
      this->ReductionFactor = 1.0 / this->MinimumImageSampleDistance;
    }
  }
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::EnsureImageSampleResources(
  void* deviceVoid, int width, int height)
{
  // Force recreation if pixel format changed (e.g., BGRA8Unorm -> RGBA16Float)
  const int desiredFormat = MTLPixelFormatRGBA16Float;
  if (this->ImageSampleColorTexture && this->ImageSampleFBOWidth == width &&
    this->ImageSampleFBOHeight == height && this->ImageSamplePixelFormat == desiredFormat)
  {
    return true;
  }

  this->ReleaseImageSampleResources();

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)deviceVoid;

    // Create offscreen color texture (RGBA16Float for inter-block accumulation)
    MTLTextureDescriptor* colorDesc = [[MTLTextureDescriptor alloc] init];
    colorDesc.textureType = MTLTextureType2D;
    colorDesc.pixelFormat = MTLPixelFormatRGBA16Float;
    colorDesc.width = width;
    colorDesc.height = height;
    colorDesc.mipmapLevelCount = 1;
    colorDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    colorDesc.storageMode = MTLStorageModePrivate;

    id<MTLTexture> colorTex = [device newTextureWithDescriptor:colorDesc];
    [colorDesc release];
    if (!colorTex)
    {
      vtkErrorMacro("Failed to create image-sample color texture");
      return false;
    }
    AssignMetalObject(this->ImageSampleColorTexture, colorTex);

    // Create blit pipeline (fullscreen quad that samples the offscreen texture)
    if (!this->EnsureShaderLibrary(deviceVoid))
    {
      this->ReleaseImageSampleResources();
      return false;
    }
    id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary;

    NSError* error = nil;
    id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_fullscreen_main"];
    id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragment_image_sample_blit"];
    if (!vertexFunc || !fragmentFunc)
    {
      vtkErrorMacro("Failed to find image-sample blit shader functions");
      [vertexFunc release];
      [fragmentFunc release];
      this->ReleaseImageSampleResources();
      return false;
    }

    MTLRenderPipelineDescriptor* pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunc;
    pipelineDesc.fragmentFunction = fragmentFunc;
    pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
    // Enable blending so the blit composites the accumulated volume over the background
    pipelineDesc.colorAttachments[0].blendingEnabled = YES;
    pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
    pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    pipelineDesc.rasterSampleCount = 1; // Always 1 for blit; MSAA is on the main pass

    id<MTLRenderPipelineState> pso =
      [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    [pipelineDesc release];
    if (!pso)
    {
      vtkErrorMacro(<< "Image-sample blit pipeline: " << [[error localizedDescription] UTF8String]);
      [vertexFunc release];
      [fragmentFunc release];
      this->ReleaseImageSampleResources();
      return false;
    }
    AssignMetalObject(this->ImageSamplePipeline, pso);
    {
      VolumePipelineKey k = { static_cast<uint32_t>(VolumePipelineType::ImageSampleBlit),
        MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1, 0 };
      [(__bridge id)pso retain];
      this->PipelineCache[k] = (__bridge void*)pso;
    }

    [vertexFunc release];
    [fragmentFunc release];

    this->ImageSampleFBOWidth = width;
    this->ImageSampleFBOHeight = height;
    this->ImageSamplePixelFormat = desiredFormat;
  }

  return true;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ReleaseImageSampleResources()
{
  ReleaseMetalObject(this->ImageSampleColorTexture);
  ReleaseMetalObject(this->ImageSampleDepthTexture);
  ReleaseMetalObject(this->ImageSamplePipeline);
  this->ImageSampleFBOWidth = 0;
  this->ImageSampleFBOHeight = 0;
  this->ImageSamplePixelFormat = 0;

  // Release order-independent compositing layer texture array
  ReleaseMetalObject(this->LayerTextureArray);
  this->LayerTextureCapacity = 0;
  this->LayerFBOWidth = 0;
  this->LayerFBOHeight = 0;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ReleaseGradientNormalTexture()
{
  ReleaseMetalObject(this->GradientNormalTexture);
  ReleaseMetalObject(this->NormalComputePipeline);
  this->NormalTextureDims[0] = 0;
  this->NormalTextureDims[1] = 0;
  this->NormalTextureDims[2] = 0;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::EnsureGradientNormalTexture(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkVolume* vol)
{
  if (!this->VolumeTexture)
  {
    return false;
  }

  if (!this->UsePrecomputedNormals)
  {
    this->ReleaseGradientNormalTexture();
    return false;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mtlQueueVoid;

  id<MTLTexture> volTex = (__bridge id<MTLTexture>)this->VolumeTexture;
  int dims[3] = { static_cast<int>(volTex.width), static_cast<int>(volTex.height), static_cast<int>(volTex.depth) };

  // Reuse if still valid — data, scalar range, and params haven't changed
  id<MTLTexture> oldTex = (__bridge id<MTLTexture>)this->GradientNormalTexture;
  bool stale = !oldTex ||
    static_cast<int>(oldTex.width) != dims[0] ||
    static_cast<int>(oldTex.height) != dims[1] ||
    static_cast<int>(oldTex.depth) != dims[2] ||
    this->VolumeUploadTime.GetMTime() > this->NormalTextureTime.GetMTime() ||
    this->GetMTime() > this->NormalTextureTime.GetMTime();

  if (!stale)
  {
    return true;
  }

  this->ReleaseGradientNormalTexture();

  if (!this->EnsureShaderLibrary(mtlDeviceVoid))
  {
    return false;
  }
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary;

  @autoreleasepool
  {
    // Create 3D normal texture (RGBA8Unorm: normal.xyz*0.5+0.5 in RGB, gradMag in A)
    MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType = MTLTextureType3D;
    desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
    desc.width = dims[0];
    desc.height = dims[1];
    desc.depth = dims[2];
    desc.mipmapLevelCount = 1;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModePrivate;

    id<MTLTexture> normalTex = [device newTextureWithDescriptor:desc];
    [desc release];
    if (!normalTex)
    {
      vtkErrorMacro("Failed to create gradient normal texture");
      return false;
    }
    AssignMetalObject(this->GradientNormalTexture, normalTex);
    this->NormalTextureDims[0] = dims[0];
    this->NormalTextureDims[1] = dims[1];
    this->NormalTextureDims[2] = dims[2];

    // Create compute pipeline state (cache it)
    if (!this->NormalComputePipeline)
    {
      id<MTLFunction> kernelFunc = [library newFunctionWithName:@"volume_compute_normals"];
      if (!kernelFunc)
      {
        vtkErrorMacro("Failed to find volume_compute_normals kernel");
        this->ReleaseGradientNormalTexture();
        return false;
      }

      NSError* error = nil;
      id<MTLComputePipelineState> cps =
        [device newComputePipelineStateWithFunction:kernelFunc error:&error];
      [kernelFunc release];
      if (!cps)
      {
        vtkErrorMacro(<< "Failed to create normal compute pipeline: "
                      << [[error localizedDescription] UTF8String]);
        this->ReleaseGradientNormalTexture();
        return false;
      }
      AssignMetalObject(this->NormalComputePipeline, cps);
    }

    // Build NormalComputeUniforms
    double scalarRange = this->ScalarRange[1] - this->ScalarRange[0];
    if (scalarRange <= 0.0) scalarRange = 1.0;

    struct NormalComputeUniforms {
      uint32_t dimX, dimY, dimZ;
      float gsX, gsY, gsZ;
      float scalarScale;
      float scalarBias;
      float gradNormFactor;
    };

    NormalComputeUniforms u;
    u.dimX = static_cast<uint32_t>(dims[0]);
    u.dimY = static_cast<uint32_t>(dims[1]);
    u.dimZ = static_cast<uint32_t>(dims[2]);
    u.gsX = 1.0f / std::max(dims[0], 1);
    u.gsY = 1.0f / std::max(dims[1], 1);
    u.gsZ = 1.0f / std::max(dims[2], 1);
    float normFactor = this->ScalarNormalizationFactor;
    u.scalarScale = 1.0f / std::max(static_cast<float>((this->ScalarRange[1] - this->ScalarRange[0]) / normFactor), 1e-6f);
    u.scalarBias = -(static_cast<float>(this->ScalarRange[0] / normFactor)) * u.scalarScale;
    u.gradNormFactor = static_cast<float>(scalarRange * 0.25 / normFactor);

    // Dispatch compute
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    cmdBuf.label = @"VTK Volume Normal Compute";

    id<MTLComputeCommandEncoder> compEnc = [cmdBuf computeCommandEncoder];
    [compEnc setComputePipelineState:(__bridge id<MTLComputePipelineState>)this->NormalComputePipeline];
    [compEnc setTexture:volTex atIndex:0];
    [compEnc setTexture:normalTex atIndex:1];
    [compEnc setBytes:&u length:sizeof(u) atIndex:0];

    MTLSize gridSize = MTLSizeMake(dims[0], dims[1], dims[2]);
    NSUInteger tgw_max = 16;
    NSUInteger tgw_x = std::min(tgw_max, static_cast<NSUInteger>(dims[0]));
    NSUInteger tgw_y = std::min(tgw_max, static_cast<NSUInteger>(dims[1]));
    NSUInteger tgw_z = std::min(static_cast<NSUInteger>(1024) / (tgw_x * tgw_y), static_cast<NSUInteger>(dims[2]));
    MTLSize threadGroupSize = MTLSizeMake(tgw_x, tgw_y, tgw_z);
    [compEnc dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [compEnc endEncoding];
    [cmdBuf commit];

    this->NormalTextureTime.Modified();
  }

  return this->GradientNormalTexture != nullptr;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::EnsureLayerResources(void* deviceVoid, int w, int h, int neededSlices)
{
  int capacity = std::max(neededSlices, 1);
  if (this->LayerTextureArray && this->LayerFBOWidth == w && this->LayerFBOHeight == h &&
        this->LayerTextureCapacity >= capacity)
    return true;

  // Release old texture array if size or capacity changed
  ReleaseMetalObject(this->LayerTextureArray);
  this->LayerTextureCapacity = 0;

  id<MTLDevice> device = (__bridge id<MTLDevice>)deviceVoid;
  MTLTextureDescriptor* d = [[MTLTextureDescriptor alloc] init];
  d.textureType = MTLTextureType2DArray;
  d.pixelFormat = MTLPixelFormatRGBA16Float;
  d.width = w;
  d.height = h;
  d.depth = 1;
  d.arrayLength = capacity;
  d.mipmapLevelCount = 1;
  d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  d.storageMode = MTLStorageModePrivate;

  id<MTLTexture> texArray = [device newTextureWithDescriptor:d];
  [d release];
  if (!texArray)
  {
    vtkErrorMacro("Failed to create layer texture array");
    return false;
  }
  AssignMetalObject(this->LayerTextureArray, texArray);
  this->LayerTextureCapacity = capacity;
  this->LayerFBOWidth = w;
  this->LayerFBOHeight = h;
  return true;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ReleaseGraphicsResources(vtkWindow* vtkNotUsed(window))
{
  this->ReleaseImageSampleResources();
  this->ClearBlocks();

  ReleaseMetalObject(this->PipelineState);
  ReleaseMetalObject(this->AccumulationPipelineState);
  ReleaseMetalObject(this->LayerPipelineState);
  ReleaseMetalObject(this->CompositePipelineState);
  ReleaseMetalObject(this->VolumeTexture);
  ReleaseMetalObject(this->ColorOpacityTexture);
  ReleaseMetalObject(this->GradientOpacityTexture);
  ReleaseMetalObject(this->MinMaxTexture);
  ReleaseMetalObject(this->MinMaxScratchTexture);
  this->ReleaseGradientNormalTexture();

  this->ReleaseMaskResources();

  // Phase 5: Release GPU min-max compute pipelines
  ReleaseMetalObject(this->MinMaxComputePipeline);
  ReleaseMetalObject(this->DilateComputePipeline);

  // Phase 7: Release GPU data-type conversion compute pipelines
  ReleaseMetalObject(this->ConvertShortToHalfPipeline);
  ReleaseMetalObject(this->ConvertShortToFloatPipeline);
  ReleaseMetalObject(this->ConvertIntToHalfPipeline);
  ReleaseMetalObject(this->ConvertIntToFloatPipeline);
  ReleaseMetalObject(this->ConvertUIntToHalfPipeline);
  ReleaseMetalObject(this->ConvertUIntToFloatPipeline);
  ReleaseMetalObject(this->DummyDepthTexture);
  ReleaseMetalObject(this->DummyVolumeTexture);
  ReleaseMetalObject(this->DummyMaskTexture);
  ReleaseMetalObject(this->DummyMinMaxTexture);
  ReleaseMetalObject(this->DepthStencilState);

  // Phase 1A: Release cached shader library
  ReleaseMetalObject(this->CachedShaderLibrary);

  // Phase 1C: Reset pipeline pre-warm guard so it re-warms after device loss / resize
  this->PipelinesPreWarmed = false;

  // Phase 1B: Clear pipeline cache
  for (auto& entry : this->PipelineCache)
  {
    [(__bridge id)entry.second release];
  }
  this->PipelineCache.clear();

  for (int i = 0; i < 3; ++i)
  {
    ReleaseMetalObject(this->UniformBuffers[i]);
  }

  ReleaseMetalObject(this->VertexBuffer);
  ReleaseMetalObject(this->IndexBuffer);
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::GetReductionRatio(double ratio[3])
{
  // Image-space downsampling reduces the rendering resolution
  double imageRatio = 1.0 / this->ImageSampleDistance;
  ratio[0] = imageRatio;
  ratio[1] = imageRatio;
  ratio[2] = 1.0;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::PreRender(vtkRenderer* vtkNotUsed(ren),
  vtkVolume* vtkNotUsed(vol), double vtkNotUsed(datasetBounds)[6],
  double vtkNotUsed(scalarRange)[2], int vtkNotUsed(noOfComponents),
  unsigned int vtkNotUsed(numberOfLevels))
{
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::RenderBlock(vtkRenderer* vtkNotUsed(ren),
  vtkVolume* vtkNotUsed(vol), unsigned int vtkNotUsed(level))
{
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::PostRender(vtkRenderer* vtkNotUsed(ren),
  int vtkNotUsed(numberOfScalarComponents))
{
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::UpdateVolumeTexture(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkVolume* vol)
{
  vtkImageData* input = vtkImageData::SafeDownCast(this->GetInput());
  if (!input)
  {
    return false;
  }

  vtkDataArray* scalars = input->GetPointData()->GetScalars();
  if (!scalars)
  {
    return false;
  }

  bool doReload = (this->VolumeTexture == nullptr);
  doReload |= (input->GetMTime() > this->VolumeUploadTime.GetMTime());

  // Check if partitioning is active — route to block-based texture creation
  bool usePartitions = (this->Partitions[0] > 1 || this->Partitions[1] > 1 || this->Partitions[2] > 1);

  // Clear stale blocks when partitions are disabled
  if (!usePartitions && !this->Blocks.empty())
  {
    this->ClearBlocks();
  }

  if (usePartitions)
  {
    // Only reload if blocks don't exist yet or data has changed
    bool blockNeedsReload = this->Blocks.empty();
    blockNeedsReload |= (input->GetMTime() > this->VolumeUploadTime.GetMTime());
    blockNeedsReload |= (this->GetMTime() > this->VolumeUploadTime.GetMTime());

    vtkVolumeProperty* property = vol ? vol->GetProperty() : nullptr;
    vtkPiecewiseFunction* opacityFunc =
      property ? property->GetScalarOpacity() : nullptr;
    if (opacityFunc)
    {
      blockNeedsReload |= (opacityFunc->GetMTime() > this->VolumeUploadTime.GetMTime());
    }
    if (blockNeedsReload)
    {
      // Split the volume into blocks and create per-block textures
      int fullExt[6];
      input->GetExtent(fullExt);

      // Clear old blocks and create new ones
      this->ClearBlocks();

      int nx = this->Partitions[0];
      int ny = this->Partitions[1];
      int nz = this->Partitions[2];
      int deltaX = (fullExt[1] - fullExt[0] + 1) / nx;
      int deltaY = (fullExt[3] - fullExt[2] + 1) / ny;
      int deltaZ = (fullExt[5] - fullExt[4] + 1) / nz;

      for (int k = 0; k < nz; ++k)
      {
        for (int j = 0; j < ny; ++j)
        {
          for (int i = 0; i < nx; ++i)
          {
            VolumeBlock block;
            block.Extents[0] = fullExt[0] + i * deltaX;
            block.Extents[1] = (i == nx - 1) ? fullExt[1] : fullExt[0] + (i + 1) * deltaX - 1;
            block.Extents[2] = fullExt[2] + j * deltaY;
            block.Extents[3] = (j == ny - 1) ? fullExt[3] : fullExt[2] + (j + 1) * deltaY - 1;
            block.Extents[4] = fullExt[4] + k * deltaZ;
            block.Extents[5] = (k == nz - 1) ? fullExt[5] : fullExt[4] + (k + 1) * deltaZ - 1;
            this->Blocks.push_back(block);
          }
        }
      }

      // Store full volume bounds for vertex buffer (covers entire volume) using extent
      double origin[3], spacing[3];
      input->GetOrigin(origin);
      input->GetSpacing(spacing);
      double x0 = origin[0] + spacing[0] * fullExt[0];
      double x1 = origin[0] + spacing[0] * fullExt[1];
      double y0 = origin[1] + spacing[1] * fullExt[2];
      double y1 = origin[1] + spacing[1] * fullExt[3];
      double z0 = origin[2] + spacing[2] * fullExt[4];
      double z1 = origin[2] + spacing[2] * fullExt[5];
      this->ModelBounds[0] = std::min(x0, x1);
      this->ModelBounds[1] = std::max(x0, x1);
      this->ModelBounds[2] = std::min(y0, y1);
      this->ModelBounds[3] = std::max(y0, y1);
      this->ModelBounds[4] = std::min(z0, z1);
      this->ModelBounds[5] = std::max(z0, z1);

      this->VolumeNumComponents = scalars->GetNumberOfComponents();

      if (!this->UpdateBlockTextures(
            mtlDeviceVoid, mtlQueueVoid, vol, input, scalars, this->VolumeNumComponents))
      {
        return false;
      }

      this->VolumeUploadTime.Modified();
    }
    return true;
  }

  if (doReload)
  {
    @autoreleasepool
    {
      id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;
      id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mtlQueueVoid;

      int dims[3];
      input->GetDimensions(dims);

      int dataType = scalars->GetDataType();
      int numComponents = scalars->GetNumberOfComponents();
      vtkIdType numTuples = scalars->GetNumberOfTuples();

      if (numComponents < 1 || numComponents > 4)
      {
        vtkErrorMacro("Unsupported number of scalar components: " << numComponents);
        return false;
      }

      this->VolumeNumComponents = numComponents;

      // Store model-space bounds using image extent (handles non-zero extents and negative spacing)
      int ext[6];
      input->GetExtent(ext);
      double origin[3], spacing[3];
      input->GetOrigin(origin);
      input->GetSpacing(spacing);
      double x0 = origin[0] + spacing[0] * ext[0];
      double x1 = origin[0] + spacing[0] * ext[1];
      double y0 = origin[1] + spacing[1] * ext[2];
      double y1 = origin[1] + spacing[1] * ext[3];
      double z0 = origin[2] + spacing[2] * ext[4];
      double z1 = origin[2] + spacing[2] * ext[5];
      this->ModelBounds[0] = std::min(x0, x1);
      this->ModelBounds[1] = std::max(x0, x1);
      this->ModelBounds[2] = std::min(y0, y1);
      this->ModelBounds[3] = std::max(y0, y1);
      this->ModelBounds[4] = std::min(z0, z1);
      this->ModelBounds[5] = std::max(z0, z1);

      // Select optimal texture format for this data type
      int componentsForFormat = (numComponents == 3) ? 4 : numComponents;

      struct FormatInfo
      {
        MTLPixelFormat format;
        int bytesPerComponent;
        float normalizationFactor;
        bool needsConversion;
      };

      FormatInfo fmtInfo = {};
      fmtInfo.needsConversion = true;
      fmtInfo.normalizationFactor = 1.0f;

      bool useHalf = this->PreferHalfPrecision &&
        HalfRangeIsSafe(this->ScalarRange[0], this->ScalarRange[1]);

      switch (dataType)
      {
        case VTK_FLOAT:
        {
          if (useHalf)
          {
            fmtInfo.bytesPerComponent = 2;
            fmtInfo.needsConversion = true;
            fmtInfo.normalizationFactor = 1.0f;
            switch (componentsForFormat)
            {
              case 1:
                fmtInfo.format = MTLPixelFormatR16Float;
                break;
              case 2:
                fmtInfo.format = MTLPixelFormatRG16Float;
                break;
              default:
                fmtInfo.format = MTLPixelFormatRGBA16Float;
                break;
            }
          }
          else
          {
            fmtInfo.bytesPerComponent = 4;
            fmtInfo.needsConversion = false;
            fmtInfo.normalizationFactor = 1.0f;
            switch (componentsForFormat)
            {
              case 1:
                fmtInfo.format = MTLPixelFormatR32Float;
                break;
              case 2:
                fmtInfo.format = MTLPixelFormatRG32Float;
                break;
              default:
                fmtInfo.format = MTLPixelFormatRGBA32Float;
                break;
            }
          }
          break;
        }
        case VTK_UNSIGNED_CHAR:
        {
          fmtInfo.bytesPerComponent = 1;
          fmtInfo.needsConversion = false;
          fmtInfo.normalizationFactor = 255.0f;
          switch (componentsForFormat)
          {
            case 1:
              fmtInfo.format = MTLPixelFormatR8Unorm;
              break;
            case 2:
              fmtInfo.format = MTLPixelFormatRG8Unorm;
              break;
            default:
              fmtInfo.format = MTLPixelFormatRGBA8Unorm;
              break;
          }
          break;
        }
        case VTK_UNSIGNED_SHORT:
        {
          fmtInfo.bytesPerComponent = 2;
          fmtInfo.needsConversion = false;
          fmtInfo.normalizationFactor = 65535.0f;
          switch (componentsForFormat)
          {
            case 1:
              fmtInfo.format = MTLPixelFormatR16Unorm;
              break;
            case 2:
              fmtInfo.format = MTLPixelFormatRG16Unorm;
              break;
            default:
              fmtInfo.format = MTLPixelFormatRGBA16Unorm;
              break;
          }
          break;
        }
        default:
        {

          if (useHalf)
          {
            fmtInfo.bytesPerComponent = 2;
            fmtInfo.needsConversion = true;
            fmtInfo.normalizationFactor = 1.0f;
            switch (componentsForFormat)
            {
              case 1:
                fmtInfo.format = MTLPixelFormatR16Float;
                break;
              case 2:
                fmtInfo.format = MTLPixelFormatRG16Float;
                break;
              default:
                fmtInfo.format = MTLPixelFormatRGBA16Float;
                break;
            }
          }
          else
          {
            fmtInfo.bytesPerComponent = 4;
            fmtInfo.needsConversion = true;
            fmtInfo.normalizationFactor = 1.0f;
            switch (componentsForFormat)
            {
              case 1:
                fmtInfo.format = MTLPixelFormatR32Float;
                break;
              case 2:
                fmtInfo.format = MTLPixelFormatRG32Float;
                break;
              default:
                fmtInfo.format = MTLPixelFormatRGBA32Float;
                break;
            }
          }
          break;
        }
      }

      this->ScalarNormalizationFactor = fmtInfo.normalizationFactor;

      const void* uploadPointer = nullptr;
      std::vector<uint16_t> halfData;
      std::vector<float> floatData;
      std::vector<uint8_t> conversionBuffer;
      bool gpuConversionUsed = false;

      // Phase 7: GPU data type conversion (replaces CPU vtkSMPTools loop)
      // For short/int/uint/double data types, dispatch a Metal compute kernel
      // that reads from a shared buffer and writes directly to the 3D texture.
      if (fmtInfo.needsConversion && this->UseGPUConversion)
      {
        // Determine kernel name based on (dataType, useHalf) pair
        const char* kernelName = nullptr;
             if (dataType == VTK_SHORT)        kernelName = useHalf ? "volume_convert_short_to_half"  : "volume_convert_short_to_float";
        else if (dataType == VTK_INT)          kernelName = useHalf ? "volume_convert_int_to_half"    : "volume_convert_int_to_float";
        else if (dataType == VTK_UNSIGNED_INT) kernelName = useHalf ? "volume_convert_uint_to_half"   : "volume_convert_uint_to_float";
        // Note: VTK_DOUBLE is not supported in Metal device address space; falls through to CPU.

        if (!kernelName)
        {
          // Metal cannot read double in device address space.
          // Fall through to CPU conversion.
        }
        else
        {
          if (!this->EnsureConversionPipelines(mtlDeviceVoid))
          {
            return false;
          }

          id<MTLComputePipelineState> pipeline = nullptr;
               if (dataType == VTK_SHORT)        pipeline = (__bridge id<MTLComputePipelineState>)(useHalf ? this->ConvertShortToHalfPipeline : this->ConvertShortToFloatPipeline);
          else if (dataType == VTK_INT)          pipeline = (__bridge id<MTLComputePipelineState>)(useHalf ? this->ConvertIntToHalfPipeline : this->ConvertIntToFloatPipeline);
          else if (dataType == VTK_UNSIGNED_INT) pipeline = (__bridge id<MTLComputePipelineState>)(useHalf ? this->ConvertUIntToHalfPipeline : this->ConvertUIntToFloatPipeline);
          if (!pipeline)
          {
            vtkErrorMacro("GPU conversion pipeline not available for data type " << dataType);
            return false;
          }

          int outputComponents = (numComponents == 3) ? 4 : numComponents;
          size_t srcBytesPerVoxel = static_cast<size_t>(vtkDataArray::GetDataTypeSize(dataType)) * numComponents;
          size_t totalSrcBytes = static_cast<size_t>(numTuples) * srcBytesPerVoxel;

          id<MTLBuffer> srcBuf = [device newBufferWithBytes:scalars->GetVoidPointer(0)
                                                     length:totalSrcBytes
                                                    options:MTLResourceStorageModeShared];
          if (!srcBuf)
          {
            vtkErrorMacro("Failed to create source buffer for GPU conversion");
            return false;
          }

          // Ensure texture has ShaderWrite usage for compute kernel output
          ReleaseMetalObject(this->VolumeTexture);

          MTLTextureDescriptor* texDesc = [[MTLTextureDescriptor alloc] init];
          texDesc.textureType = MTLTextureType3D;
          texDesc.pixelFormat = fmtInfo.format;
          texDesc.width = dims[0];
          texDesc.height = dims[1];
          texDesc.depth = dims[2];
          texDesc.mipmapLevelCount = 1;
          texDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
          texDesc.storageMode = MTLStorageModePrivate;

          id<MTLTexture> tex = [device newTextureWithDescriptor:texDesc];
          [texDesc release];
          if (!tex)
          {
            vtkErrorMacro("Failed to create 3D volume texture for GPU conversion");
            [srcBuf release];
            return false;
          }
          AssignMetalObject(this->VolumeTexture, tex);

          // Dispatch compute kernel
          VolumeConvertUniforms vu;
          vu.dimX = static_cast<uint32_t>(dims[0]);
          vu.dimY = static_cast<uint32_t>(dims[1]);
          vu.dimZ = static_cast<uint32_t>(dims[2]);
          vu.numComponents = static_cast<uint32_t>(numComponents);
          vu.outputComponents = static_cast<uint32_t>(outputComponents);
          vu._pad = 0;

          id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
          cmdBuf.label = @"VTK Volume GPU Convert";
          id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
          enc.label = @"Volume Convert";
          [enc setBuffer:srcBuf offset:0 atIndex:0];
          [enc setTexture:tex atIndex:0];
          [enc setBytes:&vu length:sizeof(vu) atIndex:1];

          MTLSize grid = MTLSizeMake(dims[0], dims[1], dims[2]);
          MTLSize tg = MTLSizeMake(8, 8, 8);
          [enc dispatchThreads:grid threadsPerThreadgroup:tg];
          [enc endEncoding];
          [cmdBuf commit];

          [srcBuf release];
          gpuConversionUsed = true;
        }
      }
      if (fmtInfo.needsConversion && !gpuConversionUsed)
      {
        int outputComponents = (numComponents == 3) ? 4 : numComponents;

        if (useHalf)
        {
          halfData.resize(static_cast<size_t>(numTuples) * outputComponents);

          switch (dataType)
          {
            case VTK_SHORT:
            {
              const short* src = static_cast<const short*>(scalars->GetVoidPointer(0));
              vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
                for (vtkIdType i = begin; i < end; ++i)
                {
                  for (int c = 0; c < numComponents; ++c)
                    halfData[i * outputComponents + c] =
                      FloatToHalf(static_cast<float>(src[i * numComponents + c]));
                  for (int c = numComponents; c < outputComponents; ++c)
                    halfData[i * outputComponents + c] = FloatToHalf(0.0f);
                }
              });
              break;
            }
            case VTK_INT:
            {
              const int* src = static_cast<const int*>(scalars->GetVoidPointer(0));
              vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
                for (vtkIdType i = begin; i < end; ++i)
                {
                  for (int c = 0; c < numComponents; ++c)
                    halfData[i * outputComponents + c] =
                      FloatToHalf(static_cast<float>(src[i * numComponents + c]));
                  for (int c = numComponents; c < outputComponents; ++c)
                    halfData[i * outputComponents + c] = FloatToHalf(0.0f);
                }
              });
              break;
            }
            case VTK_UNSIGNED_INT:
            {
              const unsigned int* src =
                static_cast<const unsigned int*>(scalars->GetVoidPointer(0));
              vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
                for (vtkIdType i = begin; i < end; ++i)
                {
                  for (int c = 0; c < numComponents; ++c)
                    halfData[i * outputComponents + c] =
                      FloatToHalf(static_cast<float>(src[i * numComponents + c]));
                  for (int c = numComponents; c < outputComponents; ++c)
                    halfData[i * outputComponents + c] = FloatToHalf(0.0f);
                }
              });
              break;
            }
            case VTK_DOUBLE:
            {
              const double* src = static_cast<const double*>(scalars->GetVoidPointer(0));
              vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
                for (vtkIdType i = begin; i < end; ++i)
                {
                  for (int c = 0; c < numComponents; ++c)
                    halfData[i * outputComponents + c] =
                      FloatToHalf(static_cast<float>(src[i * numComponents + c]));
                  for (int c = numComponents; c < outputComponents; ++c)
                    halfData[i * outputComponents + c] = FloatToHalf(0.0f);
                }
              });
              break;
            }
            case VTK_FLOAT:
            {
              const float* src = static_cast<const float*>(scalars->GetVoidPointer(0));
              vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
                for (vtkIdType i = begin; i < end; ++i)
                {
                  for (int c = 0; c < numComponents; ++c)
                    halfData[i * outputComponents + c] =
                      FloatToHalf(src[i * numComponents + c]);
                  for (int c = numComponents; c < outputComponents; ++c)
                    halfData[i * outputComponents + c] = FloatToHalf(0.0f);
                }
              });
              break;
            }
            default:
            {
              vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
                for (vtkIdType i = begin; i < end; ++i)
                {
                  for (int c = 0; c < numComponents; ++c)
                    halfData[i * outputComponents + c] =
                      FloatToHalf(static_cast<float>(scalars->GetComponent(i, c)));
                  for (int c = numComponents; c < outputComponents; ++c)
                    halfData[i * outputComponents + c] = FloatToHalf(0.0f);
                }
              });
              break;
            }
          }
          uploadPointer = halfData.data();
        }
        else
        {
          floatData.resize(static_cast<size_t>(numTuples) * outputComponents);

          switch (dataType)
          {
            case VTK_SHORT:
            {
              const short* src = static_cast<const short*>(scalars->GetVoidPointer(0));
              vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
                for (vtkIdType i = begin; i < end; ++i)
                {
                  for (int c = 0; c < numComponents; ++c)
                    floatData[i * outputComponents + c] =
                      static_cast<float>(src[i * numComponents + c]);
                  for (int c = numComponents; c < outputComponents; ++c)
                    floatData[i * outputComponents + c] = 0.0f;
                }
              });
              break;
            }
            case VTK_INT:
            {
              const int* src = static_cast<const int*>(scalars->GetVoidPointer(0));
              vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
                for (vtkIdType i = begin; i < end; ++i)
                {
                  for (int c = 0; c < numComponents; ++c)
                    floatData[i * outputComponents + c] =
                      static_cast<float>(src[i * numComponents + c]);
                  for (int c = numComponents; c < outputComponents; ++c)
                    floatData[i * outputComponents + c] = 0.0f;
                }
              });
              break;
            }
            case VTK_UNSIGNED_INT:
            {
              const unsigned int* src =
                static_cast<const unsigned int*>(scalars->GetVoidPointer(0));
              vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
                for (vtkIdType i = begin; i < end; ++i)
                {
                  for (int c = 0; c < numComponents; ++c)
                    floatData[i * outputComponents + c] =
                      static_cast<float>(src[i * numComponents + c]);
                  for (int c = numComponents; c < outputComponents; ++c)
                    floatData[i * outputComponents + c] = 0.0f;
                }
              });
              break;
            }
            case VTK_DOUBLE:
            {
              const double* src = static_cast<const double*>(scalars->GetVoidPointer(0));
              vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
                for (vtkIdType i = begin; i < end; ++i)
                {
                  for (int c = 0; c < numComponents; ++c)
                    floatData[i * outputComponents + c] =
                      static_cast<float>(src[i * numComponents + c]);
                  for (int c = numComponents; c < outputComponents; ++c)
                    floatData[i * outputComponents + c] = 0.0f;
                }
              });
              break;
            }
            default:
            {
              vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
                for (vtkIdType i = begin; i < end; ++i)
                {
                  for (int c = 0; c < numComponents; ++c)
                    floatData[i * outputComponents + c] =
                      static_cast<float>(scalars->GetComponent(i, c));
                  for (int c = numComponents; c < outputComponents; ++c)
                    floatData[i * outputComponents + c] = 0.0f;
                }
              });
              break;
            }
          }
          uploadPointer = floatData.data();
        }
      }
      else if (dataType == VTK_FLOAT)
      {
        if (numComponents == 3)
        {
          const float* src = static_cast<const float*>(scalars->GetVoidPointer(0));
          conversionBuffer.resize(static_cast<size_t>(numTuples) * 4 * sizeof(float));
          float* dst = reinterpret_cast<float*>(conversionBuffer.data());
          vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
            for (vtkIdType i = begin; i < end; ++i)
            {
              dst[i * 4 + 0] = src[i * 3 + 0];
              dst[i * 4 + 1] = src[i * 3 + 1];
              dst[i * 4 + 2] = src[i * 3 + 2];
              dst[i * 4 + 3] = 0.0f;
            }
          });
          uploadPointer = conversionBuffer.data();
        }
        else
        {
          uploadPointer = scalars->GetVoidPointer(0);
        }
      }
      else if (dataType == VTK_UNSIGNED_CHAR)
      {
        if (numComponents == 3)
        {
          const unsigned char* src =
            static_cast<const unsigned char*>(scalars->GetVoidPointer(0));
          conversionBuffer.resize(static_cast<size_t>(numTuples) * 4);
          vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
            for (vtkIdType i = begin; i < end; ++i)
            {
              conversionBuffer[i * 4 + 0] = src[i * 3 + 0];
              conversionBuffer[i * 4 + 1] = src[i * 3 + 1];
              conversionBuffer[i * 4 + 2] = src[i * 3 + 2];
              conversionBuffer[i * 4 + 3] = 255;
            }
          });
          uploadPointer = conversionBuffer.data();
        }
        else
        {
          uploadPointer = scalars->GetVoidPointer(0);
        }
      }
      else if (dataType == VTK_UNSIGNED_SHORT)
      {
        if (numComponents == 3)
        {
          const unsigned short* src =
            static_cast<const unsigned short*>(scalars->GetVoidPointer(0));
          conversionBuffer.resize(static_cast<size_t>(numTuples) * 4 * 2);
          unsigned short* dst = reinterpret_cast<unsigned short*>(conversionBuffer.data());
          vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
            for (vtkIdType i = begin; i < end; ++i)
            {
              dst[i * 4 + 0] = src[i * 3 + 0];
              dst[i * 4 + 1] = src[i * 3 + 1];
              dst[i * 4 + 2] = src[i * 3 + 2];
              dst[i * 4 + 3] = 65535;
            }
          });
          uploadPointer = conversionBuffer.data();
        }
        else
        {
          uploadPointer = scalars->GetVoidPointer(0);
        }
      }

      if (!gpuConversionUsed)
      {
      id<MTLTexture> oldTex = (__bridge id<MTLTexture>)this->VolumeTexture;
      id<MTLTexture> tex = nil;

      if (oldTex &&
          oldTex.width == dims[0] &&
          oldTex.height == dims[1] &&
          oldTex.depth == dims[2] &&
          oldTex.pixelFormat == fmtInfo.format)
      {
        tex = oldTex;
      }
      else
      {
        ReleaseMetalObject(this->VolumeTexture);

        MTLTextureDescriptor* texDesc = [[MTLTextureDescriptor alloc] init];
        texDesc.textureType = MTLTextureType3D;
        texDesc.pixelFormat = fmtInfo.format;
        texDesc.width = dims[0];
        texDesc.height = dims[1];
        texDesc.depth = dims[2];
        texDesc.mipmapLevelCount = 1;
        texDesc.usage = MTLTextureUsageShaderRead;
        texDesc.storageMode = MTLStorageModePrivate;

        tex = [device newTextureWithDescriptor:texDesc];
        [texDesc release];
        if (!tex)
        {
          vtkErrorMacro("Failed to create 3D volume texture");
          return false;
        }
        AssignMetalObject(this->VolumeTexture, tex);
      }

      int actualComponents = (numComponents == 3) ? 4 : numComponents;
      NSUInteger bytesPerRow = static_cast<NSUInteger>(dims[0]) * fmtInfo.bytesPerComponent *
        actualComponents;
      NSUInteger bytesPerImage = bytesPerRow * dims[1];

      // Upload via staging buffer + blit encoder (works on all platforms)
      NSUInteger totalBytes = bytesPerImage * dims[2];

      id<MTLBuffer> stagingBuf = [device newBufferWithBytes:uploadPointer
                                                     length:totalBytes
                                                    options:MTLResourceStorageModeShared];
      if (!stagingBuf)
      {
        vtkErrorMacro("Failed to create volume staging buffer");
        return false;
      }

      id<MTLCommandBuffer> uploadCmdBuf = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [uploadCmdBuf blitCommandEncoder];
      [blit copyFromBuffer:stagingBuf
              sourceOffset:0
       sourceBytesPerRow:bytesPerRow
     sourceBytesPerImage:bytesPerImage
              sourceSize:MTLSizeMake(dims[0], dims[1], dims[2])
               toTexture:tex
        destinationSlice:0
        destinationLevel:0
       destinationOrigin:MTLOriginMake(0, 0, 0)];
      [blit endEncoding];
      [uploadCmdBuf commit];
      // No waitUntilCompleted — Metal retains the staging buffer for the
      // lifetime of the committed command buffer. Command buffers on the same
      // queue execute in order, so the render pass will not read the texture
      // until after this blit completes.

      } // end if (!gpuConversionUsed)

      this->VolumeUploadTime.Modified();
    }
  }

  // Phase 1C: Pre-warm common pipeline permutations to avoid first-use hitches.
  // PSO compilation takes 50-200ms on Apple Silicon; warming them up here on the
  // render thread (synchronous) ensures subsequent toggles are frame-rate smooth.
  if (!this->PipelinesPreWarmed)
  {
    static const uint32_t kPreWarmMasks[] = {
      0,
      VolumeFeature_Shading,
      VolumeFeature_Shading | VolumeFeature_GradientOpacity,
      VolumeFeature_Shading | VolumeFeature_MinMax,
      VolumeFeature_Shading | VolumeFeature_GradientOpacity | VolumeFeature_MinMax,
      VolumeFeature_Shading | VolumeFeature_Mask,
      VolumeFeature_Shading | VolumeFeature_Mask | VolumeFeature_MinMax,
      VolumeFeature_Shading | VolumeFeature_NormalTexture,
      VolumeFeature_Shading | VolumeFeature_NormalTexture | VolumeFeature_MinMax,
    };

    struct PreWarmSpec {
      uint32_t type;
      uint32_t colorFormat;
      uint32_t depthFormat;
      uint32_t samples;
    };

    int sampleCount = this->CurrentSampleCount > 0 ? this->CurrentSampleCount : 1;
    uint32_t sc = static_cast<uint32_t>(sampleCount);
    PreWarmSpec specs[] = {
      { static_cast<uint32_t>(VolumePipelineType::DirectScreen),
        MTLPixelFormatBGRA8Unorm, MTLPixelFormatDepth32Float, sc },
      { static_cast<uint32_t>(VolumePipelineType::OffscreenLayer),
        MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1 },
      { static_cast<uint32_t>(VolumePipelineType::OffscreenAccumulation),
        MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1 },
      { static_cast<uint32_t>(VolumePipelineType::FullscreenDirect),
        MTLPixelFormatBGRA8Unorm, MTLPixelFormatDepth32Float, sc },
      { static_cast<uint32_t>(VolumePipelineType::FullscreenOffscreen),
        MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1 },
    };

    for (auto& spec : specs)
    {
      for (uint32_t mask : kPreWarmMasks)
      {
        this->GetOrCreateVolumePipeline(mtlDeviceVoid,
          spec.type, spec.colorFormat, spec.depthFormat, spec.samples, mask);
      }
    }
    this->PipelinesPreWarmed = true;
  }

  return true;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::UpdateTransferFunctionTexture(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkVolume* vol, double actualSampleDistance)
{
  vtkVolumeProperty* property = vol->GetProperty();
  if (!property)
  {
    return false;
  }

  vtkColorTransferFunction* colorFunc = property->GetRGBTransferFunction();
  vtkPiecewiseFunction* opacityFunc = property->GetScalarOpacity();
  if (!colorFunc || !opacityFunc)
  {
    return false;
  }

  bool scalarRangeChanged =
    (this->ScalarRange[0] != this->LastTransferFunctionScalarRange[0]) ||
    (this->ScalarRange[1] != this->LastTransferFunctionScalarRange[1]);

  bool doReload = (this->ColorOpacityTexture == nullptr);
  doReload |= (colorFunc->GetMTime() > this->TransferFunctionUploadTime.GetMTime());
  doReload |= (opacityFunc->GetMTime() > this->TransferFunctionUploadTime.GetMTime());
  doReload |= scalarRangeChanged;
  // Re-upload when sample distance changes (affects opacity pre-integration)
  doReload |= (actualSampleDistance != this->LastTransferFunctionSampleDistance);

  if (doReload)
  {
    @autoreleasepool
    {
      id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

      // Pre-integration factor: matches OpenGL mapper's opacity table correction.
      // The OpenGL mapper applies: adjusted = 1 - pow(1 - raw, sampleDistance / unitDistance)
      // This bakes the step distance into the opacity texture so the shader
      // can use simple front-to-back compositing without multiplying by step distance.
      const double unitDistance = property->GetScalarOpacityUnitDistance(0);
      const double factor = actualSampleDistance / unitDistance;

      unsigned char tfData[256 * 4];
      for (int i = 0; i < 256; ++i)
      {
        double val = this->ScalarRange[0] + (this->ScalarRange[1] - this->ScalarRange[0]) * (i / 255.0);
        double rgb[3];
        colorFunc->GetColor(val, rgb);
        double opacity = opacityFunc->GetValue(val);

        // Pre-integrate opacity (matches vtkOpenGLVolumeOpacityTable::InternalUpdate)
        if (opacity > 0.0001)
        {
          opacity = 1.0 - pow(1.0 - opacity, factor);
        }

        rgb[0] = std::clamp(rgb[0], 0.0, 1.0);
        rgb[1] = std::clamp(rgb[1], 0.0, 1.0);
        rgb[2] = std::clamp(rgb[2], 0.0, 1.0);
        opacity = std::clamp(opacity, 0.0, 1.0);

        tfData[i * 4 + 0] = static_cast<unsigned char>(rgb[0] * 255.0);
        tfData[i * 4 + 1] = static_cast<unsigned char>(rgb[1] * 255.0);
        tfData[i * 4 + 2] = static_cast<unsigned char>(rgb[2] * 255.0);
        tfData[i * 4 + 3] = static_cast<unsigned char>(opacity * 255.0);
      }

      id<MTLTexture> oldTfTex = (__bridge id<MTLTexture>)this->ColorOpacityTexture;
      id<MTLTexture> tex = nil;

      if (oldTfTex)
      {
        tex = oldTfTex;
      }
      else
      {
        ReleaseMetalObject(this->ColorOpacityTexture);

        MTLTextureDescriptor* tfDesc = [[MTLTextureDescriptor alloc] init];
        tfDesc.textureType = MTLTextureType2D;
        tfDesc.pixelFormat = MTLPixelFormatRGBA8Unorm;
        tfDesc.width = 256;
        tfDesc.height = 1;
        tfDesc.mipmapLevelCount = 1;
        tfDesc.usage = MTLTextureUsageShaderRead;
        tfDesc.storageMode = MTLStorageModeShared;

        tex = [device newTextureWithDescriptor:tfDesc];
        [tfDesc release];
        if (!tex)
        {
          vtkErrorMacro("Failed to create transfer function texture");
          return false;
        }
        AssignMetalObject(this->ColorOpacityTexture, tex);
      }

      MTLRegion region = MTLRegionMake2D(0, 0, 256, 1);
      [tex replaceRegion:region
            mipmapLevel:0
              withBytes:tfData
            bytesPerRow:256 * 4];

      this->LastTransferFunctionScalarRange[0] = this->ScalarRange[0];
      this->LastTransferFunctionScalarRange[1] = this->ScalarRange[1];
      this->TransferFunctionUploadTime.Modified();
      this->LastTransferFunctionSampleDistance = actualSampleDistance;
    }
  }

  return this->ColorOpacityTexture != nullptr;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::UpdateGradientOpacityTexture(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkVolume* vol)
{
  vtkVolumeProperty* property = vol->GetProperty();
  if (!property || !property->HasGradientOpacity())
  {
    return false;
  }

  vtkPiecewiseFunction* gradOpacityFunc = property->GetGradientOpacity();
  if (!gradOpacityFunc)
  {
    return false;
  }

  bool doReload = (this->GradientOpacityTexture == nullptr);
  doReload |= (gradOpacityFunc->GetMTime() > this->GradientOpacityUploadTime.GetMTime());

  if (doReload)
  {
    @autoreleasepool
    {
      id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

      // Build 256-entry gradient opacity lookup table.
      // Range: [0, 0.25 * scalarRange] — matches the normalization in the shader
      // where gradient magnitude is normalized to [0, 0.25 * dataRange].
      double scalarRange = this->ScalarRange[1] - this->ScalarRange[0];
      if (scalarRange <= 0.0)
      {
        scalarRange = 1.0;
      }
      double gradMax = scalarRange * 0.25;

      unsigned char gradData[256 * 4]; // RGBA8Unorm (R channel used)
      double table[256];
      gradOpacityFunc->GetTable(0.0, gradMax, 256, table);

      for (int i = 0; i < 256; ++i)
      {
        unsigned char val =
          static_cast<unsigned char>(std::max(0.0, std::min(1.0, table[i])) * 255.0);
        gradData[i * 4 + 0] = val;
        gradData[i * 4 + 1] = val;
        gradData[i * 4 + 2] = val;
        gradData[i * 4 + 3] = 255;
      }

      id<MTLTexture> oldTex = (__bridge id<MTLTexture>)this->GradientOpacityTexture;
      id<MTLTexture> tex = nil;

      if (oldTex)
      {
        tex = oldTex;
      }
      else
      {
        ReleaseMetalObject(this->GradientOpacityTexture);

        MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
        desc.textureType = MTLTextureType2D;
        desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
        desc.width = 256;
        desc.height = 1;
        desc.mipmapLevelCount = 1;
        desc.usage = MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared;

        tex = [device newTextureWithDescriptor:desc];
        [desc release];
        if (!tex)
        {
          vtkErrorMacro("Failed to create gradient opacity texture");
          return false;
        }
        AssignMetalObject(this->GradientOpacityTexture, tex);
      }

      MTLRegion region = MTLRegionMake2D(0, 0, 256, 1);
      [tex replaceRegion:region
            mipmapLevel:0
              withBytes:gradData
            bytesPerRow:256 * 4];

      this->GradientOpacityUploadTime.Modified();
    }
  }

  return this->GradientOpacityTexture != nullptr;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::UpdateMaskTexture(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkVolume* vol)
{
  vtkImageData* maskInput = this->MaskInput;
  if (!maskInput)
  {
    return false;
  }

  vtkVolumeProperty* property = vol->GetProperty();
  if (!property)
  {
    return false;
  }

  // Get the scalar array from the mask input
  int isCellData = 0;
  vtkDataArray* arr = this->GetScalars(
    maskInput, this->ScalarMode, this->ArrayAccessMode,
    this->ArrayId, this->ArrayName, isCellData);
  if (!arr)
  {
    return false;
  }

  bool doReload = (this->MaskTexture == nullptr);
  doReload |= (maskInput->GetMTime() > this->MaskUpdateTime.GetMTime());
  doReload |= (arr->GetMTime() > this->MaskUpdateTime.GetMTime());

  if (doReload)
  {
    @autoreleasepool
    {
      id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

      // Get mask dimensions
      int dims[3];
      maskInput->GetDimensions(dims);

      // Get the data pointer
      int numComponents = arr->GetNumberOfComponents();

      // Choose mask texture format based on data type to minimize memory
      vtkIdType numTuples = arr->GetNumberOfTuples();
      int dataType = arr->GetDataType();
      MTLPixelFormat chosenFormat;
      NSUInteger bytesPerPixel = 0;
      const void* uploadSrc = nullptr;
      std::vector<uint8_t> byteData;
      std::vector<uint16_t> shortData;
      std::vector<float> floatData;

      if (dataType == VTK_UNSIGNED_CHAR)
      {
        chosenFormat = MTLPixelFormatR8Unorm;
        bytesPerPixel = 1;
        byteData.resize(numTuples);
        const unsigned char* src = static_cast<const unsigned char*>(arr->GetVoidPointer(0));
        if (numComponents == 1)
        {
          std::memcpy(byteData.data(), src, numTuples);
        }
        else
        {
          vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
            for (vtkIdType i = begin; i < end; ++i)
              byteData[i] = src[i * numComponents];
          });
        }
        uploadSrc = byteData.data();
      }
      else if (dataType == VTK_UNSIGNED_SHORT)
      {
        chosenFormat = MTLPixelFormatR16Unorm;
        bytesPerPixel = 2;
        shortData.resize(numTuples);
        const unsigned short* src = static_cast<const unsigned short*>(arr->GetVoidPointer(0));
        if (numComponents == 1)
        {
          std::memcpy(shortData.data(), src, numTuples * sizeof(uint16_t));
        }
        else
        {
          vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
            for (vtkIdType i = begin; i < end; ++i)
              shortData[i] = src[i * numComponents];
          });
        }
        uploadSrc = shortData.data();
      }
      else
      {
        chosenFormat = MTLPixelFormatR32Float;
        bytesPerPixel = 4;
        floatData.resize(numTuples);

        if (dataType == VTK_FLOAT)
        {
          const float* src = static_cast<const float*>(arr->GetVoidPointer(0));
          if (numComponents == 1)
          {
            std::memcpy(floatData.data(), src, numTuples * sizeof(float));
          }
          else
          {
            vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
              for (vtkIdType i = begin; i < end; ++i)
                floatData[i] = src[i * numComponents];
            });
          }
        }
        else
        {
          for (vtkIdType i = 0; i < numTuples; ++i)
          {
            floatData[i] = static_cast<float>(arr->GetComponent(i, 0));
          }
        }
        uploadSrc = floatData.data();
      }

      // Create or update the 3D mask texture
      id<MTLTexture> oldTex = (__bridge id<MTLTexture>)this->MaskTexture;
      id<MTLTexture> tex = nil;

      if (oldTex &&
          oldTex.width == static_cast<NSUInteger>(dims[0]) &&
          oldTex.height == static_cast<NSUInteger>(dims[1]) &&
          oldTex.depth == static_cast<NSUInteger>(dims[2]) &&
          oldTex.pixelFormat == chosenFormat)
      {
        tex = oldTex;
      }
      else
      {
        ReleaseMetalObject(this->MaskTexture);

        MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
        desc.textureType = MTLTextureType3D;
        desc.pixelFormat = chosenFormat;
        desc.width = dims[0];
        desc.height = dims[1];
        desc.depth = dims[2];
        desc.mipmapLevelCount = 1;
        desc.usage = MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared;

        tex = [device newTextureWithDescriptor:desc];
        [desc release];
        if (!tex)
        {
          vtkErrorMacro("Failed to create mask texture");
          return false;
        }
        AssignMetalObject(this->MaskTexture, tex);
      }

      // Upload mask data to texture
      MTLRegion region = MTLRegionMake3D(0, 0, 0, dims[0], dims[1], dims[2]);
      NSUInteger maskBytesPerRow = static_cast<NSUInteger>(dims[0]) * bytesPerPixel;
      NSUInteger maskBytesPerImage = maskBytesPerRow * dims[1];
      [tex replaceRegion:region
            mipmapLevel:0
                  slice:0
              withBytes:uploadSrc
            bytesPerRow:maskBytesPerRow
          bytesPerImage:maskBytesPerImage];

      this->MaskUpdateTime.Modified();
    }
  }

  return this->MaskTexture != nullptr;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::UpdateLabelMapTransferTexture(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkVolume* vol)
{
  vtkVolumeProperty* property = vol->GetProperty();
  if (!property)
  {
    return false;
  }

  // Get label map labels
  std::set<int> labels = property->GetLabelMapLabels();
  if (labels.empty())
  {
    return false;
  }

  // Get the maximum label value
  int maxLabel = *(labels.rbegin());
  int numLabels = maxLabel + 1; // +1 because label 0 is included

  // Check if we need to reload
  vtkMTimeType latestMTime = 0;
  for (int label : labels)
  {
    vtkColorTransferFunction* colorFunc = property->GetLabelColor(label);
    vtkPiecewiseFunction* opacityFunc = property->GetLabelScalarOpacity(label);
    if (colorFunc && colorFunc->GetMTime() > latestMTime)
    {
      latestMTime = colorFunc->GetMTime();
    }
    if (opacityFunc && opacityFunc->GetMTime() > latestMTime)
    {
      latestMTime = opacityFunc->GetMTime();
    }
  }

  bool scalarRangeChanged =
    (this->ScalarRange[0] != this->LastLabelMapScalarRange[0]) ||
    (this->ScalarRange[1] != this->LastLabelMapScalarRange[1]);

  bool labelSetChanged =
    (maxLabel != this->LastLabelMapMaxLabel) ||
    (labels.size() != this->LastLabelMapLabelCount);

  bool doReload = (this->LabelMapTransferTexture == nullptr);
  doReload |= (latestMTime > this->MaskUpdateTime.GetMTime());
  doReload |= scalarRangeChanged;
  doReload |= labelSetChanged;

  if (doReload)
  {
    @autoreleasepool
    {
      id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

      // Create 2D label map transfer function texture
      // Width = 1024 (scalar range samples), Height = numLabels (one row per label)
      const int tfWidth = 1024;
      const int tfHeight = numLabels;

      // Get the scalar range from the volume texture
      double scalarRange[2] = { this->ScalarRange[0], this->ScalarRange[1] };

      // Build the 2D transfer function texture data (RGBA8Unorm)
      std::vector<uint8_t> tfData(tfWidth * tfHeight * 4);

      // Row 0: label 0 (default) - zeros (will be filled by default TF)
      std::fill(tfData.begin(), tfData.begin() + tfWidth * 4, static_cast<uint8_t>(0));

      // Rows 1..maxLabel: per-label transfer functions
      for (int label = 1; label < numLabels; ++label)
      {
        uint8_t* rowPtr = tfData.data() + label * tfWidth * 4;

        // Get color transfer function for this label
        vtkColorTransferFunction* colorFunc = property->GetLabelColor(label);
        if (!colorFunc)
        {
          colorFunc = property->GetRGBTransferFunction(); // fallback to default
        }

        // Get opacity function for this label
        vtkPiecewiseFunction* opacityFunc = property->GetLabelScalarOpacity(label);
        if (!opacityFunc)
        {
          opacityFunc = property->GetScalarOpacity(); // fallback to default
        }

        if (colorFunc)
        {
          std::vector<double> colorTable(tfWidth * 3);
          colorFunc->GetTable(scalarRange[0], scalarRange[1], tfWidth, colorTable.data());
          for (int i = 0; i < tfWidth; ++i)
          {
            rowPtr[i * 4 + 0] = static_cast<uint8_t>(std::clamp(colorTable[i * 3 + 0], 0.0, 1.0) * 255.0);
            rowPtr[i * 4 + 1] = static_cast<uint8_t>(std::clamp(colorTable[i * 3 + 1], 0.0, 1.0) * 255.0);
            rowPtr[i * 4 + 2] = static_cast<uint8_t>(std::clamp(colorTable[i * 3 + 2], 0.0, 1.0) * 255.0);
          }
        }

        if (opacityFunc)
        {
          std::vector<double> opacityTable(tfWidth);
          opacityFunc->GetTable(scalarRange[0], scalarRange[1], tfWidth, opacityTable.data());
          for (int i = 0; i < tfWidth; ++i)
          {
            rowPtr[i * 4 + 3] = static_cast<uint8_t>(std::clamp(opacityTable[i], 0.0, 1.0) * 255.0);
          }
        }
      }

      // Create or update the 2D texture
      id<MTLTexture> oldTex = (__bridge id<MTLTexture>)this->LabelMapTransferTexture;
      id<MTLTexture> tex = nil;

      // Check if existing texture has the right dimensions (numLabels may have changed)
      if (oldTex && static_cast<int>(oldTex.width) == tfWidth &&
          static_cast<int>(oldTex.height) == tfHeight &&
          oldTex.pixelFormat == MTLPixelFormatRGBA8Unorm)
      {
        tex = oldTex;
      }
      else
      {
        if (this->LabelMapTransferTexture)
        {
          ReleaseMetalObject(this->LabelMapTransferTexture);
        }

        MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
        desc.textureType = MTLTextureType2D;
        desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
        desc.width = tfWidth;
        desc.height = tfHeight;
        desc.mipmapLevelCount = 1;
        desc.usage = MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared;

        tex = [device newTextureWithDescriptor:desc];
        [desc release];
        if (!tex)
        {
          vtkErrorMacro("Failed to create label map transfer texture");
          return false;
        }
        AssignMetalObject(this->LabelMapTransferTexture, tex);
      }

      // Upload data to texture
      MTLRegion region = MTLRegionMake2D(0, 0, tfWidth, tfHeight);
      [tex replaceRegion:region
            mipmapLevel:0
              withBytes:tfData.data()
            bytesPerRow:tfWidth * 4];

      this->LastLabelMapScalarRange[0] = this->ScalarRange[0];
      this->LastLabelMapScalarRange[1] = this->ScalarRange[1];
      this->LastLabelMapMaxLabel = maxLabel;
      this->LastLabelMapLabelCount = labels.size();
      this->MaskUpdateTime.Modified();
    }
  }

  return this->LabelMapTransferTexture != nullptr;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::SetMaskUniforms(void* uniforms, vtkVolume* vol)
{
  VolumeMapperUniforms* u = static_cast<VolumeMapperUniforms*>(uniforms);

  vtkImageData* maskInput = this->MaskInput;
  vtkVolumeProperty* property = vol->GetProperty();

  if (maskInput && property &&
      this->MaskType == vtkGPUVolumeRayCastMapper::LabelMapMaskType)
  {
    u->UseMask = 1.0f;
    u->MaskBlendFactor = this->MaskBlendFactor;
    u->MaskScale = 1.0f;  // Default scale for unsigned char mask
    u->MaskBias = 0.0f;   // Default bias for unsigned char mask

    // Compute mask scale/bias based on the mask data type
    int cellFlag = 0;
    vtkDataArray* arr = this->GetScalars(
      maskInput, this->ScalarMode, this->ArrayAccessMode,
      this->ArrayId, this->ArrayName, cellFlag);
    if (arr)
    {
      int dataType = arr->GetDataType();
      if (dataType == VTK_UNSIGNED_CHAR)
      {
        u->MaskScale = 1.0f / 255.0f;
        u->MaskBias = 0.0f;
      }
      else if (dataType == VTK_CHAR)
      {
        u->MaskScale = 2.0f / 255.0f;
        u->MaskBias = -1.0f;
      }
      else if (dataType == VTK_UNSIGNED_SHORT)
      {
        u->MaskScale = 1.0f / 65535.0f;
        u->MaskBias = 0.0f;
      }
      else if (dataType == VTK_SHORT)
      {
        u->MaskScale = 2.0f / 65535.0f;
        u->MaskBias = -1.0f;
      }
      else
      {
        // For float or other types, compute from range
        double range[2];
        arr->GetRange(range);
        double dataRange = range[1] - range[0];
        if (dataRange > 0.0)
        {
          u->MaskScale = static_cast<float>(1.0 / dataRange);
          u->MaskBias = static_cast<float>(-range[0] / dataRange);
        }
      }
    }

    // Get the number of labels for quantization
    // numLabels = maxLabel + 1, so the shader can use label indices directly.
    std::set<int> labels = property->GetLabelMapLabels();
    int maxLabel = labels.empty() ? 0 : *(labels.rbegin());
    int numLabels = labels.empty() ? 0 : maxLabel + 1;
    u->LabelMapNumLabels = static_cast<float>(numLabels);
    // Determine scale to convert sampled mask value back to label index.
    // Unorm formats normalize to [0,1] at sample time, so we scale back.
    id<MTLTexture> maskTex = (__bridge id<MTLTexture>)this->MaskTexture;
    if (maskTex)
    {
      switch (maskTex.pixelFormat)
      {
        case MTLPixelFormatR8Unorm:
          u->MaskScale = 255.0f;
          break;
        case MTLPixelFormatR16Unorm:
          u->MaskScale = 65535.0f;
          break;
        default:
          u->MaskScale = 1.0f;
          break;
      }
    }
    else
    {
      u->MaskScale = 1.0f;
    }
    u->MaskBias = 0.0f;
  }
  else
  {
    u->UseMask = 0.0f;
    u->MaskBlendFactor = 0.0f;
    u->MaskScale = 1.0f;
    u->MaskBias = 0.0f;
    u->LabelMapNumLabels = 0.0f;
  }
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ReleaseMaskResources()
{
  ReleaseMetalObject(this->MaskTexture);
  ReleaseMetalObject(this->LabelMapTransferTexture);
  ReleaseMetalObject(this->LabelMapGradientOpacityTexture);
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ClearBlocks()
{
  for (auto& block : this->Blocks)
  {
    ReleaseMetalObject(block.Texture);
    ReleaseMetalObject(block.MinMaxTexture);
    ReleaseMetalObject(block.NormalTexture);
  }
  this->Blocks.clear();
  this->BlockScalarRanges.clear();
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::IsBlockEmpty(
  double blockMin, double blockMax, vtkPiecewiseFunction* opacityFunc)
{
  if (blockMin > blockMax)
  {
    return true;
  }
  if (!opacityFunc)
  {
    return false;
  }

  double range = this->ScalarRange[1] - this->ScalarRange[0];
  if (range <= 0.0)
  {
    range = 1.0;
  }

  // Build the 256-entry opacity lookup table matching the shader's quantization
  double opacityTable[256];
  opacityFunc->GetTable(this->ScalarRange[0], this->ScalarRange[1], 256, opacityTable);

  // Map block scalar range to table indices
  int idxMin = static_cast<int>((blockMin - this->ScalarRange[0]) / range * 255.0);
  int idxMax = static_cast<int>((blockMax - this->ScalarRange[0]) / range * 255.0);
  idxMin = std::max(0, std::min(255, idxMin));
  idxMax = std::max(0, std::min(255, idxMax));

  // Check if any opacity entry in this range is non-zero
  for (int i = idxMin; i <= idxMax; ++i)
  {
    if (opacityTable[i] > 0.0)
    {
      return false;
    }
  }

  return true;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::EnsureMinMaxComputePipelines(void* mtlDeviceVoid)
{
  if (this->MinMaxComputePipeline && this->DilateComputePipeline)
  {
    return true;
  }

  if (!this->EnsureShaderLibrary(mtlDeviceVoid))
  {
    return false;
  }

  id<MTLDevice> dev = (__bridge id<MTLDevice>)mtlDeviceVoid;
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary;

  @autoreleasepool
  {
    if (!this->MinMaxComputePipeline)
    {
      id<MTLFunction> func = [library newFunctionWithName:@"volume_compute_minmax"];
      if (!func)
      {
        vtkErrorMacro("Failed to find volume_compute_minmax kernel");
        return false;
      }
      NSError* error = nil;
      id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:func error:&error];
      [func release];
      if (!pso)
      {
        vtkErrorMacro(<< "Failed to create minmax compute pipeline: "
                      << [[error localizedDescription] UTF8String]);
        return false;
      }
      AssignMetalObject(this->MinMaxComputePipeline, pso);
    }

    if (!this->DilateComputePipeline)
    {
      id<MTLFunction> func = [library newFunctionWithName:@"volume_dilate_minmax"];
      if (!func)
      {
        vtkErrorMacro("Failed to find volume_dilate_minmax kernel");
        return false;
      }
      NSError* error = nil;
      id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:func error:&error];
      [func release];
      if (!pso)
      {
        vtkErrorMacro(<< "Failed to create dilate compute pipeline: "
                      << [[error localizedDescription] UTF8String]);
        return false;
      }
      AssignMetalObject(this->DilateComputePipeline, pso);
    }
  }

  return true;
}

//------------------------------------------------------------------------------
// Phase 7: Ensure compute pipelines for GPU data type conversion exist.
// Creates all 8 conversion pipelines (4 input types × half/float output).
// Returns true on success, false on failure (caller falls back to CPU).
bool vtkMetalGPUVolumeRayCastMapper::EnsureConversionPipelines(void* mtlDeviceVoid)
{
  if (!this->EnsureShaderLibrary(mtlDeviceVoid))
  {
    return false;
  }

  id<MTLDevice> dev = (__bridge id<MTLDevice>)mtlDeviceVoid;
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary;

  @autoreleasepool
  {
#define VTK_CREATE_CONVERT_PIPELINE(kernelName, member) \
    if (!member) { \
      id<MTLFunction> func = [library newFunctionWithName:@kernelName]; \
      if (!func) { \
        vtkErrorMacro("Failed to find " << kernelName << " kernel"); \
        return false; \
      } \
      NSError* error = nil; \
      id<MTLComputePipelineState> pso = \
        [dev newComputePipelineStateWithFunction:func error:&error]; \
      [func release]; \
      if (!pso) { \
        vtkErrorMacro(<< "Failed to create " << kernelName << " pipeline: " \
                      << [[error localizedDescription] UTF8String]); \
        return false; \
      } \
      AssignMetalObject(member, pso); \
    }

    VTK_CREATE_CONVERT_PIPELINE("volume_convert_short_to_half", this->ConvertShortToHalfPipeline);
    VTK_CREATE_CONVERT_PIPELINE("volume_convert_short_to_float", this->ConvertShortToFloatPipeline);
    VTK_CREATE_CONVERT_PIPELINE("volume_convert_int_to_half", this->ConvertIntToHalfPipeline);
    VTK_CREATE_CONVERT_PIPELINE("volume_convert_int_to_float", this->ConvertIntToFloatPipeline);
    VTK_CREATE_CONVERT_PIPELINE("volume_convert_uint_to_half", this->ConvertUIntToHalfPipeline);
    VTK_CREATE_CONVERT_PIPELINE("volume_convert_uint_to_float", this->ConvertUIntToFloatPipeline);
    // Note: VTK_DOUBLE kernels are not provided (Metal does not support 'double' in device address space).

#undef VTK_CREATE_CONVERT_PIPELINE
  }

  return true;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::ComputeMinMaxGPU(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkVolume* vol,
  vtkImageData* input, vtkDataArray* scalars)
{
  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mtlQueueVoid;

  if (!this->VolumeTexture)
  {
    return false;
  }

  // Timestamp-based caching: skip recompute when nothing changed.
  if (input && this->MinMaxTexture)
  {
    vtkVolumeProperty* property = vol ? vol->GetProperty() : nullptr;
    vtkPiecewiseFunction* opFunc = property ? property->GetScalarOpacity() : nullptr;
    if (opFunc &&
        input->GetMTime() <= this->MinMaxUploadTime.GetMTime() &&
        opFunc->GetMTime() <= this->MinMaxUploadTime.GetMTime())
    {
      return true;
    }
  }

  id<MTLTexture> volTex = (__bridge id<MTLTexture>)this->VolumeTexture;
  int dims[3] = { static_cast<int>(volTex.width),
                  static_cast<int>(volTex.height),
                  static_cast<int>(volTex.depth) };

  const int DS = 4;
  int mmDims[3] = {
    std::max(1, (dims[0] + DS - 1) / DS),
    std::max(1, (dims[1] + DS - 1) / DS),
    std::max(1, (dims[2] + DS - 1) / DS)
  };
  this->MinMaxDims[0] = mmDims[0];
  this->MinMaxDims[1] = mmDims[1];
  this->MinMaxDims[2] = mmDims[2];

  if (!this->EnsureMinMaxComputePipelines(mtlDeviceVoid))
  {
    return false;
  }

  @autoreleasepool
  {
    // --- Reuse or create temporary occupancy texture ---
    id<MTLTexture> rawOcc = (__bridge id<MTLTexture>)this->MinMaxScratchTexture;
    if (!rawOcc ||
        rawOcc.width != static_cast<NSUInteger>(mmDims[0]) ||
        rawOcc.height != static_cast<NSUInteger>(mmDims[1]) ||
        rawOcc.depth != static_cast<NSUInteger>(mmDims[2]))
    {
      MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
      desc.textureType = MTLTextureType3D;
      desc.pixelFormat = MTLPixelFormatR8Unorm;
      desc.width = mmDims[0];
      desc.height = mmDims[1];
      desc.depth = mmDims[2];
      desc.mipmapLevelCount = 1;
      desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
      desc.storageMode = MTLStorageModePrivate;

      rawOcc = [device newTextureWithDescriptor:desc];
      [desc release];

      if (!rawOcc)
      {
        return false;
      }
      AssignMetalObject(this->MinMaxScratchTexture, rawOcc);
    }

    // --- Build opacity prefix table from transfer function ---
    vtkVolumeProperty* property = vol ? vol->GetProperty() : nullptr;
    vtkPiecewiseFunction* opFunc = property ? property->GetScalarOpacity() : nullptr;
    if (!opFunc)
    {
      return false;
    }

    double opacityTable[256];
    opFunc->GetTable(this->ScalarRange[0], this->ScalarRange[1], 256, opacityTable);

    uint32_t opacityPrefix[257];
    opacityPrefix[0] = 0;
    for (int i = 0; i < 256; ++i)
    {
      opacityPrefix[i + 1] = opacityPrefix[i] + (opacityTable[i] > 0.0 ? 1u : 0u);
    }

    double scalarRange = this->ScalarRange[1] - this->ScalarRange[0];
    if (scalarRange <= 0.0) scalarRange = 1.0;

    // --- Build uniforms ---
    // IMPORTANT: texture sampler returns normalized values for R8Unorm/R16Unorm
    // formats but raw values for R32Float.  We must adjust scalarMin/scalarScale
    // to work in the sampler's output space (raw/normalized) so the TF index
    // computation matches the CPU path.
    float normFactor = this->ScalarNormalizationFactor;

    MinMaxComputeUniforms u;
    u.mmDimX = static_cast<uint32_t>(mmDims[0]);
    u.mmDimY = static_cast<uint32_t>(mmDims[1]);
    u.mmDimZ = static_cast<uint32_t>(mmDims[2]);
    u.volDimX = static_cast<uint32_t>(dims[0]);
    u.volDimY = static_cast<uint32_t>(dims[1]);
    u.volDimZ = static_cast<uint32_t>(dims[2]);
    u.ds = static_cast<float>(DS);
    u.scalarMin = static_cast<float>(this->ScalarRange[0] / normFactor);
    u.scalarScale = static_cast<float>(255.0 * normFactor / scalarRange);
    u._pad = 0.0f;
    memcpy(u.opacityPrefix, opacityPrefix, sizeof(opacityPrefix));

    // --- Command buffer ---
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    cmdBuf.label = @"VTK Volume MinMax Compute";

    // --- Dispatch kernel 1: macrocell occupancy ---
    id<MTLComputeCommandEncoder> enc1 = [cmdBuf computeCommandEncoder];
    enc1.label = @"Volume Compute MinMax";
    [enc1 setComputePipelineState:(__bridge id<MTLComputePipelineState>)this->MinMaxComputePipeline];
    [enc1 setTexture:volTex atIndex:0];
    [enc1 setTexture:rawOcc atIndex:1];
    [enc1 setBytes:&u length:sizeof(u) atIndex:0];

    MTLSize gridSize = MTLSizeMake(mmDims[0], mmDims[1], mmDims[2]);
    NSUInteger tgw = 8;
    MTLSize tgSize = MTLSizeMake(tgw, tgw, tgw);
    [enc1 dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [enc1 endEncoding];

    // --- Reuse or create persistent MinMax texture (dilation writes here directly) ---
    id<MTLTexture> permTex = (__bridge id<MTLTexture>)this->MinMaxTexture;
    if (!permTex ||
        permTex.width != static_cast<NSUInteger>(mmDims[0]) ||
        permTex.height != static_cast<NSUInteger>(mmDims[1]) ||
        permTex.depth != static_cast<NSUInteger>(mmDims[2]))
    {
      MTLTextureDescriptor* permDesc = [[MTLTextureDescriptor alloc] init];
      permDesc.textureType = MTLTextureType3D;
      permDesc.pixelFormat = MTLPixelFormatR8Unorm;
      permDesc.width = mmDims[0];
      permDesc.height = mmDims[1];
      permDesc.depth = mmDims[2];
      permDesc.mipmapLevelCount = 1;
      permDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
      permDesc.storageMode = MTLStorageModePrivate;

      permTex = [device newTextureWithDescriptor:permDesc];
      [permDesc release];
      if (!permTex)
      {
        vtkErrorMacro("Failed to create persistent min-max texture");
        return false;
      }
      AssignMetalObject(this->MinMaxTexture, permTex);
    }

    // --- Dispatch kernel 2: dilation (writes directly to permTex) ---
    id<MTLComputeCommandEncoder> enc2 = [cmdBuf computeCommandEncoder];
    enc2.label = @"Volume Dilate MinMax";
    [enc2 setComputePipelineState:(__bridge id<MTLComputePipelineState>)this->DilateComputePipeline];
    [enc2 setTexture:rawOcc atIndex:0];
    [enc2 setTexture:permTex atIndex:1];
    [enc2 dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [enc2 endEncoding];

    [cmdBuf commit];

    this->MinMaxUploadTime.Modified();
  }

  return this->MinMaxTexture != nullptr;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::UpdateMinMaxTexture(
  void* mtlDeviceVoid, vtkVolume* vol, vtkImageData* input, vtkDataArray* scalars,
  bool skipGlobalTexture)
{
  if (!input || !scalars || !vol)
  {
    return false;
  }

  vtkVolumeProperty* property = vol->GetProperty();
  vtkPiecewiseFunction* opFunc = property ? property->GetScalarOpacity() : nullptr;

  // Without an opacity function we cannot determine empty space — disable acceleration
  if (!opFunc)
  {
    return false;
  }

  // When skipGlobalTexture is true (partitioned mode), MinMaxTexture is
  // intentionally never created. Use MacrocellScalarMin as the validity
  // sentinel instead, so we don't re-run the full voxel scan every frame.
  bool doReload;
  if (skipGlobalTexture)
  {
    doReload = this->MacrocellScalarMin.empty();
  }
  else
  {
    doReload = (this->MinMaxTexture == nullptr);
  }
  doReload |= (input->GetMTime() > this->MinMaxUploadTime.GetMTime());
  doReload |= (opFunc->GetMTime() > this->MinMaxUploadTime.GetMTime());

  if (!doReload)
  {
    if (skipGlobalTexture)
    {
      return this->MacrocellScalarMin.empty() == false;
    }
    return this->MinMaxTexture != nullptr;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

    int dims[3];
    input->GetDimensions(dims);

    // Downsample factor: 4x in each dimension
    const int DS = 4;
    int mmDims[3] = {
      std::max(1, (dims[0] + DS - 1) / DS),
      std::max(1, (dims[1] + DS - 1) / DS),
      std::max(1, (dims[2] + DS - 1) / DS)
    };
    this->MinMaxDims[0] = mmDims[0];
    this->MinMaxDims[1] = mmDims[1];
    this->MinMaxDims[2] = mmDims[2];

    int dataType = scalars->GetDataType();
    int extents[6];
    input->GetExtent(extents);
    vtkIdType inc[3];
    input->GetIncrements(inc);

    const void* dataPtr = scalars->GetVoidPointer(0);

    // Precompute the 256-entry opacity lookup table ONCE.
    // IsBlockEmpty rebuilds this per macrocell (~1.4M times for a CT volume);
    // doing it here eliminates that redundancy entirely.
    double opacityTable[256];
    opFunc->GetTable(this->ScalarRange[0], this->ScalarRange[1], 256, opacityTable);
    const double scalarRange = this->ScalarRange[1] - this->ScalarRange[0];
    const double rangeRecip = (scalarRange > 0.0) ? (255.0 / scalarRange) : 1.0;
    const double rangeOffset = this->ScalarRange[0];

    // Snapshot loop-invariant values for the parallel lambda
    const int mmDims0 = mmDims[0];
    const int mmDims1 = mmDims[1];
    const int mmDims2 = mmDims[2];
    const int extXSize = extents[1] - extents[0] + 1;
    const int extYSize = extents[3] - extents[2] + 1;
    const int extZSize = extents[5] - extents[4] + 1;
    const vtkIdType inc0 = inc[0];
    const vtkIdType inc1 = inc[1];
    const vtkIdType inc2 = inc[2];

    // Store 1 byte per macrocell: 255 = empty, 0 = solid
    vtkIdType numCells = static_cast<vtkIdType>(mmDims0) * mmDims1 * mmDims2;

    // 1. Create a RAW buffer for the initial pass
    std::vector<uint8_t> rawMinMax(numCells, 255);

    // Per-macrocell scalar min/max — consumed later by UpdateBlockTextures
    // to compute per-block ranges without re-walking every voxel.
    this->MacrocellScalarMin.resize(numCells, 1e30f);
    this->MacrocellScalarMax.resize(numCells, -1e30f);
    float* mcMin = this->MacrocellScalarMin.data();
    float* mcMax = this->MacrocellScalarMax.data();

    // Parallel scan: each macrocell is independent — perfect for vtkSMPTools
    vtkSMPTools::For(0, numCells, [&](vtkIdType begin, vtkIdType end) {
      for (vtkIdType cellIdx = begin; cellIdx < end; ++cellIdx)
      {
        const int gx = static_cast<int>(cellIdx % mmDims0);
        const int gy = static_cast<int>((cellIdx / mmDims0) % mmDims1);
        const int gz = static_cast<int>(cellIdx / (mmDims0 * mmDims1));

        const int zStart = gz * DS;
        const int zEnd = std::min(zStart + DS, extZSize);
        const int yStart = gy * DS;
        const int yEnd = std::min(yStart + DS, extYSize);
        const int xStart = gx * DS;
        const int xEnd = std::min(xStart + DS, extXSize);

        float cellMin = 1e30f;
        float cellMax = -1e30f;

        for (int z = zStart; z < zEnd; ++z)
        {
          for (int y = yStart; y < yEnd; ++y)
          {
            for (int x = xStart; x < xEnd; ++x)
            {
              float v = 0.0f;
              switch (dataType)
              {
                case VTK_FLOAT:
                  v = static_cast<float>(
                    static_cast<const float*>(dataPtr)[z * inc2 + y * inc1 + x * inc0]);
                  break;
                case VTK_UNSIGNED_CHAR:
                  v = static_cast<float>(
                    static_cast<const unsigned char*>(dataPtr)[z * inc2 + y * inc1 + x * inc0]);
                  break;
                case VTK_UNSIGNED_SHORT:
                  v = static_cast<float>(
                    static_cast<const unsigned short*>(dataPtr)[z * inc2 + y * inc1 + x * inc0]);
                  break;
                case VTK_SHORT:
                  v = static_cast<float>(
                    static_cast<const short*>(dataPtr)[z * inc2 + y * inc1 + x * inc0]);
                  break;
                default:
                {
                  vtkIdType tupleIdx = z * (inc2 / inc0) + y * (inc1 / inc0) + x;
                  v = static_cast<float>(scalars->GetComponent(tupleIdx, 0));
                  break;
                }
              }
              if (v < cellMin) cellMin = v;
              if (v > cellMax) cellMax = v;
            }
          }
        }

        // Store macrocell scalar range for later block-range reduction
        mcMin[cellIdx] = cellMin;
        mcMax[cellIdx] = cellMax;

        // Inline empty check using the precomputed opacity table
        bool empty = true;
        if (cellMin <= cellMax)
        {
          int idxMin = std::max(0,
            std::min(255, static_cast<int>((cellMin - rangeOffset) * rangeRecip)));
          int idxMax = std::max(0,
            std::min(255, static_cast<int>((cellMax - rangeOffset) * rangeRecip)));
          for (int i = idxMin; i <= idxMax; ++i)
          {
            if (opacityTable[i] > 0.0)
            {
              empty = false;
              break;
            }
          }
        }

        rawMinMax[cellIdx] = empty ? 255 : 0;
      }
    });

    // 2. DILATION PASS: Pad solid blocks by 1 macrocell in all directions.
    // This guarantees the ray resumes normal stepping *before* hitting the surface,
    // preserving perfect trilinear interpolation and lighting gradients.
    // Gather-style stencil: read from rawMinMax, write to minMaxData —
    // embarrassingly parallel since each output cell is independent.
    std::vector<uint8_t> minMaxData(numCells, 255);
    vtkSMPTools::For(0, numCells, [&](vtkIdType begin, vtkIdType end) {
      for (vtkIdType cellIdx = begin; cellIdx < end; ++cellIdx)
      {
        const int gx = static_cast<int>(cellIdx % mmDims0);
        const int gy = static_cast<int>((cellIdx / mmDims0) % mmDims1);
        const int gz = static_cast<int>(cellIdx / (mmDims0 * mmDims1));

        const int x0 = std::max(0, gx - 1), x1 = std::min(mmDims0 - 1, gx + 1);
        const int y0 = std::max(0, gy - 1), y1 = std::min(mmDims1 - 1, gy + 1);
        const int z0 = std::max(0, gz - 1), z1 = std::min(mmDims2 - 1, gz + 1);

        bool solid = false;
        for (int nz = z0; nz <= z1 && !solid; ++nz)
        {
          for (int ny = y0; ny <= y1 && !solid; ++ny)
          {
            for (int nx = x0; nx <= x1 && !solid; ++nx)
            {
              if (rawMinMax[(nz * mmDims1 + ny) * mmDims0 + nx] == 0)
              {
                solid = true;
              }
            }
          }
        }

        minMaxData[cellIdx] = solid ? 0 : 255;
      }
    });

    // 3. Create or reuse the 3D occupancy texture (R8Unorm).
    // For partitioned volumes, skip this — blocks build their own min-max textures.
    if (!skipGlobalTexture)
    {
      id<MTLTexture> oldTex = (__bridge id<MTLTexture>)this->MinMaxTexture;
      id<MTLTexture> tex = nil;

      if (oldTex &&
          oldTex.width == mmDims0 &&
          oldTex.height == mmDims1 &&
          oldTex.depth == mmDims2)
      {
        tex = oldTex;
      }
      else
      {
        ReleaseMetalObject(this->MinMaxTexture);

        MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
        desc.textureType = MTLTextureType3D;
        desc.pixelFormat = MTLPixelFormatR8Unorm;
        desc.width = mmDims0;
        desc.height = mmDims1;
        desc.depth = mmDims2;
        desc.mipmapLevelCount = 1;
        desc.usage = MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared;

        tex = [device newTextureWithDescriptor:desc];
        [desc release];
        if (!tex)
        {
          vtkErrorMacro("Failed to create min-max acceleration texture");
          return false;
        }
        AssignMetalObject(this->MinMaxTexture, tex);
      }

      // Upload data
      MTLRegion region = MTLRegionMake3D(0, 0, 0, mmDims0, mmDims1, mmDims2);
      NSUInteger bytesPerRow = mmDims0 * sizeof(uint8_t);
      NSUInteger bytesPerImage = bytesPerRow * mmDims1;
      [tex replaceRegion:region
             mipmapLevel:0
                   slice:0
               withBytes:minMaxData.data()
             bytesPerRow:bytesPerRow
           bytesPerImage:bytesPerImage];
    }

    this->MinMaxUploadTime.Modified();
  }

  return this->MinMaxTexture != nullptr;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::SortBlocksBackToFront(
  vtkRenderer* ren, vtkVolume* vol)
{
  if (this->Blocks.size() <= 1)
  {
    return;
  }

  // Transform camera position into volume-local (grid) space.
  vtkNew<vtkMatrix4x4> modelToWorld;
  vol->GetModelToWorldMatrix(modelToWorld);
  vtkNew<vtkMatrix4x4> worldToModel;
  vtkMatrix4x4::Invert(modelToWorld, worldToModel);

  double* camPosWorld = ren->GetActiveCamera()->GetPosition();
  double camLocal[4] = { camPosWorld[0], camPosWorld[1], camPosWorld[2], 1.0 };
  worldToModel->MultiplyPoint(camLocal, camLocal);

  // For each grid axis, determine whether an INCREASING block index moves
  // toward or away from the camera. This uses camera POSITION relative to
  // the volume's split planes -- which is what actually determines ray
  // entry order for axis-aligned, non-overlapping blocks -- rather than
  // projecting onto the view direction (which is only valid for
  // orthographic projection and fails at grazing angles near a shared
  // partition boundary under perspective).
  double gridCenter[3] = {
    (this->ModelBounds[0] + this->ModelBounds[1]) * 0.5,
    (this->ModelBounds[2] + this->ModelBounds[3]) * 0.5,
    (this->ModelBounds[4] + this->ModelBounds[5]) * 0.5
  };
  bool ascendingIsCloser[3] = {
    camLocal[0] < gridCenter[0],
    camLocal[1] < gridCenter[1],
    camLocal[2] < gridCenter[2]
  };

  const int nx = this->Partitions[0];
  const int ny = this->Partitions[1];
  const int NZ = this->Partitions[2];

  // Initialize sorted order -- skip blocks with no texture (empty-space skipped)
  this->SortedBlockOrder.clear();
  this->SortedBlockOrder.reserve(this->Blocks.size());
  for (size_t idx = 0; idx < this->Blocks.size(); ++idx)
  {
    if (this->Blocks[idx].Texture)
    {
      this->SortedBlockOrder.push_back(static_cast<int>(idx));
    }
  }

  // Blocks were built as: for(k) for(j) for(i) push_back(...)
  // so idx = (k*ny + j)*nx + i. Recover (i,j,k) from the flat index.
  std::sort(this->SortedBlockOrder.begin(), this->SortedBlockOrder.end(),
    [&](int a, int b) {
      int ai = a % nx;
      int aj = (a / nx) % ny;
      int ak = a / (nx * ny);
      int bi = b % nx;
      int bj = (b / nx) % ny;
      int bk = b / (nx * ny);

      // Remap each axis so "closer to camera" is always the smaller index,
      // then combine as a single nested key. Any fixed, consistent priority
      // is valid here because the blocks are a non-overlapping axis-aligned
      // grid and the camera sits outside the grid's bounding box in normal use.
      auto key = [&](int i, int j, int k) -> long long {
        int ii = ascendingIsCloser[0] ? i : (nx - 1 - i);
        int jj = ascendingIsCloser[1] ? j : (ny - 1 - j);
        int kk = ascendingIsCloser[2] ? k : (NZ - 1 - k);
        return (static_cast<long long>(kk) * ny + jj) * nx + ii;
      };

      return key(ai, aj, ak) < key(bi, bj, bk); // ascending = closest first
    });
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::UpdateBlockTextures(void* mtlDeviceVoid,
  void* mtlQueueVoid, vtkVolume* vol, vtkImageData* input, vtkDataArray* scalars,
  int numComponents)
{
  if (this->Blocks.empty())
  {
    return false;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mtlQueueVoid;

    int fullDims[3];
    input->GetDimensions(fullDims);

    int fullExt[6];
    input->GetExtent(fullExt);

    int dataType = scalars->GetDataType();
    int componentsForFormat = (numComponents == 3) ? 4 : numComponents;

    // Determine pixel format (same logic as single-texture path)
    MTLPixelFormat pixelFormat;
    int bytesPerComponent = 2;
    float normalizationFactor = 1.0f;
    bool blockUseHalf = this->PreferHalfPrecision &&
      HalfRangeIsSafe(this->ScalarRange[0], this->ScalarRange[1]);

    switch (dataType)
    {
      case VTK_FLOAT:
        if (blockUseHalf)
        {
          bytesPerComponent = 2;
          normalizationFactor = 1.0f;
          switch (componentsForFormat)
          {
            case 1:
              pixelFormat = MTLPixelFormatR16Float;
              break;
            case 2:
              pixelFormat = MTLPixelFormatRG16Float;
              break;
            default:
              pixelFormat = MTLPixelFormatRGBA16Float;
              break;
          }
        }
        else
        {
          bytesPerComponent = 4;
          normalizationFactor = 1.0f;
          switch (componentsForFormat)
          {
            case 1:
              pixelFormat = MTLPixelFormatR32Float;
              break;
            case 2:
              pixelFormat = MTLPixelFormatRG32Float;
              break;
            default:
              pixelFormat = MTLPixelFormatRGBA32Float;
              break;
          }
        }
        break;
      case VTK_UNSIGNED_CHAR:
        bytesPerComponent = 1;
        normalizationFactor = 255.0f;
        switch (componentsForFormat)
        {
          case 1:
            pixelFormat = MTLPixelFormatR8Unorm;
            break;
          case 2:
            pixelFormat = MTLPixelFormatRG8Unorm;
            break;
          default:
            pixelFormat = MTLPixelFormatRGBA8Unorm;
            break;
        }
        break;
      case VTK_UNSIGNED_SHORT:
        bytesPerComponent = 2;
        normalizationFactor = 65535.0f;
        switch (componentsForFormat)
        {
          case 1:
            pixelFormat = MTLPixelFormatR16Unorm;
            break;
          case 2:
            pixelFormat = MTLPixelFormatRG16Unorm;
            break;
          default:
            pixelFormat = MTLPixelFormatRGBA16Unorm;
            break;
        }
        break;
      default:
      {
        if (blockUseHalf)
        {
          bytesPerComponent = 2;
          normalizationFactor = 1.0f;
          switch (componentsForFormat)
          {
            case 1:
              pixelFormat = MTLPixelFormatR16Float;
              break;
            case 2:
              pixelFormat = MTLPixelFormatRG16Float;
              break;
            default:
              pixelFormat = MTLPixelFormatRGBA16Float;
              break;
          }
        }
        else
        {
          bytesPerComponent = 4;
          normalizationFactor = 1.0f;
          switch (componentsForFormat)
          {
            case 1:
              pixelFormat = MTLPixelFormatR32Float;
              break;
            case 2:
              pixelFormat = MTLPixelFormatRG32Float;
              break;
            default:
              pixelFormat = MTLPixelFormatRGBA32Float;
              break;
          }
        }
        break;
      }
    }

    this->ScalarNormalizationFactor = normalizationFactor;

    // Precompute the 256-entry opacity lookup table for per-block min-max generation.
    vtkVolumeProperty* property = vol ? vol->GetProperty() : nullptr;
    vtkPiecewiseFunction* opFunc = property ? property->GetScalarOpacity() : nullptr;
    double opacityTable[256] = {0};
    bool hasOpacityFunc = (opFunc != nullptr);
    if (hasOpacityFunc)
    {
      opFunc->GetTable(this->ScalarRange[0], this->ScalarRange[1], 256, opacityTable);
    }

    double origin[3], spacing[3];
    input->GetOrigin(origin);
    input->GetSpacing(spacing);

    vtkIdType totalTuples = scalars->GetNumberOfTuples();
    const void* fullDataPtr = scalars->GetVoidPointer(0);

    vtkIdType inc[3];
    input->GetIncrements(inc);

    int actualComponents = (numComponents == 3) ? 4 : numComponents;
    size_t bytesPerVoxel = static_cast<size_t>(bytesPerComponent) * actualComponents;

    // Compute full-volume strides for strided blit
    NSUInteger srcBytesPerRow = static_cast<NSUInteger>(fullDims[0]) * bytesPerVoxel;
    NSUInteger srcBytesPerImage = srcBytesPerRow * fullDims[1];
    NSUInteger totalVolumeBytes = srcBytesPerImage * fullDims[2];

    // Convert data types that don't match the chosen pixel format.
    // For single-component float data, use Accelerate (NEON-vectorized) for
    // the float-to-half conversion instead of the scalar FloatToHalf loop.
    bool needsConversion = (dataType != VTK_UNSIGNED_CHAR &&
      dataType != VTK_UNSIGNED_SHORT);
    if (dataType == VTK_FLOAT && !blockUseHalf)
    {
      needsConversion = false;
    }

    // --- Phase 7: GPU compute conversion for non-native data types ---
    bool gpuConversionUsed = false;
    id<MTLTexture> gpuFullTex = nullptr;
    id<MTLBuffer> stagingBuf = nil;

    if (needsConversion && this->UseGPUConversion)
    {
      const char* kernelName = nullptr;
           if (dataType == VTK_SHORT)        kernelName = blockUseHalf ? "volume_convert_short_to_half"  : "volume_convert_short_to_float";
      else if (dataType == VTK_INT)          kernelName = blockUseHalf ? "volume_convert_int_to_half"    : "volume_convert_int_to_float";
      else if (dataType == VTK_UNSIGNED_INT) kernelName = blockUseHalf ? "volume_convert_uint_to_half"   : "volume_convert_uint_to_float";
      // VTK_DOUBLE: kernelName stays nullptr (Metal does not support double in device address space)

      if (kernelName && this->EnsureConversionPipelines(mtlDeviceVoid))
      {
        id<MTLComputePipelineState> pipeline = nullptr;
             if (dataType == VTK_SHORT)        pipeline = (__bridge id<MTLComputePipelineState>)(blockUseHalf ? this->ConvertShortToHalfPipeline : this->ConvertShortToFloatPipeline);
        else if (dataType == VTK_INT)          pipeline = (__bridge id<MTLComputePipelineState>)(blockUseHalf ? this->ConvertIntToHalfPipeline : this->ConvertIntToFloatPipeline);
        else if (dataType == VTK_UNSIGNED_INT) pipeline = (__bridge id<MTLComputePipelineState>)(blockUseHalf ? this->ConvertUIntToHalfPipeline : this->ConvertUIntToFloatPipeline);

        if (pipeline)
        {
          int outputComponents = (numComponents == 3) ? 4 : numComponents;
          size_t srcBytesPerVoxelGPU = static_cast<size_t>(vtkDataArray::GetDataTypeSize(dataType)) * numComponents;
          size_t totalSrcBytes = static_cast<size_t>(totalTuples) * srcBytesPerVoxelGPU;

          id<MTLBuffer> srcBuf = [device newBufferWithBytes:scalars->GetVoidPointer(0)
                                                     length:totalSrcBytes
                                                    options:MTLResourceStorageModeShared];
          if (srcBuf)
          {
            MTLTextureDescriptor* texDesc = [[MTLTextureDescriptor alloc] init];
            texDesc.textureType = MTLTextureType3D;
            texDesc.pixelFormat = pixelFormat;
            texDesc.width = static_cast<NSUInteger>(fullDims[0]);
            texDesc.height = static_cast<NSUInteger>(fullDims[1]);
            texDesc.depth = static_cast<NSUInteger>(fullDims[2]);
            texDesc.mipmapLevelCount = 1;
            texDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            texDesc.storageMode = MTLStorageModePrivate;

            gpuFullTex = [device newTextureWithDescriptor:texDesc];
            [texDesc release];

            if (gpuFullTex)
            {
              id<MTLCommandBuffer> convCmdBuf = [queue commandBuffer];
              convCmdBuf.label = @"VTK Block Volume Convert";
              id<MTLComputeCommandEncoder> enc = [convCmdBuf computeCommandEncoder];
              [enc setComputePipelineState:pipeline];
              [enc setBuffer:srcBuf offset:0 atIndex:0];
              [enc setTexture:gpuFullTex atIndex:0];

              VolumeConvertUniforms vu;
              vu.dimX = static_cast<uint32_t>(fullDims[0]);
              vu.dimY = static_cast<uint32_t>(fullDims[1]);
              vu.dimZ = static_cast<uint32_t>(fullDims[2]);
              vu.numComponents = static_cast<uint32_t>(numComponents);
              vu.outputComponents = static_cast<uint32_t>(outputComponents);
              [enc setBytes:&vu length:sizeof(vu) atIndex:1];

              MTLSize tgSize = MTLSizeMake(8, 8, 8);
              MTLSize tgCount = MTLSizeMake(
                (static_cast<NSUInteger>(fullDims[0]) + 7) / 8,
                (static_cast<NSUInteger>(fullDims[1]) + 7) / 8,
                (static_cast<NSUInteger>(fullDims[2]) + 7) / 8);
              [enc dispatchThreadgroups:tgCount threadsPerThreadgroup:tgSize];
              [enc endEncoding];

              [convCmdBuf commit];
              [srcBuf release];

              gpuConversionUsed = true;
            }
          }
        }
      }
    }

    if (!gpuConversionUsed)
    {
      // Create a single staging buffer for the full volume. Conversion writes
      // directly into it, eliminating intermediate std::vector copies.
      stagingBuf = [device newBufferWithLength:totalVolumeBytes
                                       options:MTLResourceStorageModeShared];
      if (!stagingBuf)
      {
        vtkErrorMacro("Failed to create full-volume staging buffer");
        return false;
      }
      void* uploadPointer = [stagingBuf contents];

      int outputComponents = (numComponents == 3) ? 4 : numComponents;

      if (needsConversion)
      {
        if (blockUseHalf)
        {
          // Fast path: single-component float → half via Accelerate (NEON/SED)
          size_t srcBytesPerVoxel =
            static_cast<size_t>(vtkDataArray::GetDataTypeSize(dataType)) * numComponents;
          if (IsContiguousScalarFloat(dataType, numComponents, totalTuples, fullDataPtr, srcBytesPerVoxel))
          {
            std::memcpy(uploadPointer, fullDataPtr, static_cast<size_t>(totalVolumeBytes));
            vImage_Buffer srcBuf = { uploadPointer, 1, static_cast<vImagePixelCount>(totalTuples),
              static_cast<vImagePixelCount>(totalTuples * sizeof(float)) };
            vImage_Buffer dstBuf = { uploadPointer, 1, static_cast<vImagePixelCount>(totalTuples),
              static_cast<vImagePixelCount>(totalTuples * sizeof(uint16_t)) };
            vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0);
          }
          else
          {
            switch (dataType)
            {
              case VTK_SHORT:
              {
                const short* src = static_cast<const short*>(fullDataPtr);
                vtkSMPTools::For(0, totalTuples, [&](vtkIdType begin, vtkIdType end) {
                  for (vtkIdType i = begin; i < end; ++i)
                  {
                    for (int c = 0; c < numComponents; ++c)
                      static_cast<uint16_t*>(uploadPointer)[i * outputComponents + c] =
                        FloatToHalf(static_cast<float>(src[i * numComponents + c]));
                    for (int c = numComponents; c < outputComponents; ++c)
                      static_cast<uint16_t*>(uploadPointer)[i * outputComponents + c] =
                        FloatToHalf(0.0f);
                  }
                });
                break;
              }
              case VTK_INT:
              {
                const int* src = static_cast<const int*>(fullDataPtr);
                vtkSMPTools::For(0, totalTuples, [&](vtkIdType begin, vtkIdType end) {
                  for (vtkIdType i = begin; i < end; ++i)
                  {
                    for (int c = 0; c < numComponents; ++c)
                      static_cast<uint16_t*>(uploadPointer)[i * outputComponents + c] =
                        FloatToHalf(static_cast<float>(src[i * numComponents + c]));
                    for (int c = numComponents; c < outputComponents; ++c)
                      static_cast<uint16_t*>(uploadPointer)[i * outputComponents + c] =
                        FloatToHalf(0.0f);
                  }
                });
                break;
              }
              case VTK_DOUBLE:
              {
                const double* src = static_cast<const double*>(fullDataPtr);
                vtkSMPTools::For(0, totalTuples, [&](vtkIdType begin, vtkIdType end) {
                  for (vtkIdType i = begin; i < end; ++i)
                  {
                    for (int c = 0; c < numComponents; ++c)
                      static_cast<uint16_t*>(uploadPointer)[i * outputComponents + c] =
                        FloatToHalf(static_cast<float>(src[i * numComponents + c]));
                    for (int c = numComponents; c < outputComponents; ++c)
                      static_cast<uint16_t*>(uploadPointer)[i * outputComponents + c] =
                        FloatToHalf(0.0f);
                  }
                });
                break;
              }
              case VTK_FLOAT:
              {
                const float* src = static_cast<const float*>(fullDataPtr);
                vtkSMPTools::For(0, totalTuples, [&](vtkIdType begin, vtkIdType end) {
                  for (vtkIdType i = begin; i < end; ++i)
                  {
                    for (int c = 0; c < numComponents; ++c)
                      static_cast<uint16_t*>(uploadPointer)[i * outputComponents + c] =
                        FloatToHalf(src[i * numComponents + c]);
                    for (int c = numComponents; c < outputComponents; ++c)
                      static_cast<uint16_t*>(uploadPointer)[i * outputComponents + c] =
                        FloatToHalf(0.0f);
                  }
                });
                break;
              }
              default:
              {
                vtkSMPTools::For(0, totalTuples, [&](vtkIdType begin, vtkIdType end) {
                  for (vtkIdType i = begin; i < end; ++i)
                  {
                    for (int c = 0; c < numComponents; ++c)
                      static_cast<uint16_t*>(uploadPointer)[i * outputComponents + c] =
                        FloatToHalf(static_cast<float>(scalars->GetComponent(i, c)));
                    for (int c = numComponents; c < outputComponents; ++c)
                      static_cast<uint16_t*>(uploadPointer)[i * outputComponents + c] =
                        FloatToHalf(0.0f);
                  }
                });
                break;
              }
            }
          }
        }
        else
        {
          switch (dataType)
          {
            case VTK_SHORT:
            {
              const short* src = static_cast<const short*>(fullDataPtr);
              vtkSMPTools::For(0, totalTuples, [&](vtkIdType begin, vtkIdType end) {
                for (vtkIdType i = begin; i < end; ++i)
                {
                  for (int c = 0; c < numComponents; ++c)
                    static_cast<float*>(uploadPointer)[i * outputComponents + c] =
                      static_cast<float>(src[i * numComponents + c]);
                  for (int c = numComponents; c < outputComponents; ++c)
                    static_cast<float*>(uploadPointer)[i * outputComponents + c] = 0.0f;
                }
              });
              break;
            }
            case VTK_INT:
            {
              const int* src = static_cast<const int*>(fullDataPtr);
              vtkSMPTools::For(0, totalTuples, [&](vtkIdType begin, vtkIdType end) {
                for (vtkIdType i = begin; i < end; ++i)
                {
                  for (int c = 0; c < numComponents; ++c)
                    static_cast<float*>(uploadPointer)[i * outputComponents + c] =
                      static_cast<float>(src[i * numComponents + c]);
                  for (int c = numComponents; c < outputComponents; ++c)
                    static_cast<float*>(uploadPointer)[i * outputComponents + c] = 0.0f;
                }
              });
              break;
            }
            case VTK_DOUBLE:
            {
              const double* src = static_cast<const double*>(fullDataPtr);
              vtkSMPTools::For(0, totalTuples, [&](vtkIdType begin, vtkIdType end) {
                for (vtkIdType i = begin; i < end; ++i)
                {
                  for (int c = 0; c < numComponents; ++c)
                    static_cast<float*>(uploadPointer)[i * outputComponents + c] =
                      static_cast<float>(src[i * numComponents + c]);
                  for (int c = numComponents; c < outputComponents; ++c)
                    static_cast<float*>(uploadPointer)[i * outputComponents + c] = 0.0f;
                }
              });
              break;
            }
            default:
            {
              vtkSMPTools::For(0, totalTuples, [&](vtkIdType begin, vtkIdType end) {
                for (vtkIdType i = begin; i < end; ++i)
                {
                  for (int c = 0; c < numComponents; ++c)
                    static_cast<float*>(uploadPointer)[i * outputComponents + c] =
                      static_cast<float>(scalars->GetComponent(i, c));
                  for (int c = numComponents; c < outputComponents; ++c)
                    static_cast<float*>(uploadPointer)[i * outputComponents + c] = 0.0f;
                }
              });
              break;
            }
          }
        }
      }
      else if (dataType == VTK_FLOAT && numComponents == 3)
      {
        // Expand 3-component float to RGBA (pad alpha with 0)
        const float* src = static_cast<const float*>(fullDataPtr);
        float* dst = static_cast<float*>(uploadPointer);
        vtkSMPTools::For(0, totalTuples, [&](vtkIdType begin, vtkIdType end) {
          for (vtkIdType i = begin; i < end; ++i)
          {
            dst[i * 4 + 0] = src[i * 3 + 0];
            dst[i * 4 + 1] = src[i * 3 + 1];
            dst[i * 4 + 2] = src[i * 3 + 2];
            dst[i * 4 + 3] = 0.0f;
          }
        });
      }
      else if (dataType == VTK_UNSIGNED_CHAR && numComponents == 3)
      {
        // Expand 3-component uchar to RGBA (pad alpha with 255)
        const unsigned char* src = static_cast<const unsigned char*>(fullDataPtr);
        unsigned char* dst = static_cast<unsigned char*>(uploadPointer);
        vtkSMPTools::For(0, totalTuples, [&](vtkIdType begin, vtkIdType end) {
          for (vtkIdType i = begin; i < end; ++i)
          {
            dst[i * 4 + 0] = src[i * 3 + 0];
            dst[i * 4 + 1] = src[i * 3 + 1];
            dst[i * 4 + 2] = src[i * 3 + 2];
            dst[i * 4 + 3] = 255;
          }
        });
      }
      else if (dataType == VTK_UNSIGNED_SHORT && numComponents == 3)
      {
        // Expand 3-component ushort to RGBA (pad alpha with 65535)
        const unsigned short* src = static_cast<const unsigned short*>(fullDataPtr);
        unsigned short* dst = static_cast<unsigned short*>(uploadPointer);
        vtkSMPTools::For(0, totalTuples, [&](vtkIdType begin, vtkIdType end) {
          for (vtkIdType i = begin; i < end; ++i)
          {
            dst[i * 4 + 0] = src[i * 3 + 0];
            dst[i * 4 + 1] = src[i * 3 + 1];
            dst[i * 4 + 2] = src[i * 3 + 2];
            dst[i * 4 + 3] = 65535;
          }
        });
      }
      else
      {
        // Native type, no conversion needed — copy directly into staging buffer
        std::memcpy(uploadPointer, fullDataPtr, static_cast<size_t>(totalVolumeBytes));
      }
    }

    // Use one command buffer for all block uploads
    id<MTLCommandBuffer> uploadCmdBuf = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [uploadCmdBuf blitCommandEncoder];

    // Resize BlockScalarRanges to match the number of blocks.
    // ClearBlocks() empties this vector, so we must re-allocate before the loop.
    this->BlockScalarRanges.resize(this->Blocks.size());

    // Create a 3D texture for each block, using strided blit from the
    // single staging buffer. This eliminates per-block memcpy and allocation.
    for (size_t idx = 0; idx < this->Blocks.size(); ++idx)
    {
      auto& block = this->Blocks[idx];

      // Ghost Voxels: Pad texture bounds by 1 voxel for correct boundary gradients
      int texExt[6] = {
        std::max(fullExt[0], block.Extents[0] - 1),
        std::min(fullExt[1], block.Extents[1] + 1),
        std::max(fullExt[2], block.Extents[2] - 1),
        std::min(fullExt[3], block.Extents[3] + 1),
        std::max(fullExt[4], block.Extents[4] - 1),
        std::min(fullExt[5], block.Extents[5] + 1)
      };

      int bDims[3] = {
        texExt[1] - texExt[0] + 1,
        texExt[3] - texExt[2] + 1,
        texExt[5] - texExt[4] + 1
      };
      block.Dims[0] = bDims[0];
      block.Dims[1] = bDims[1];
      block.Dims[2] = bDims[2];

      // Compute model-space bounds for this block
      block.BoundsMin[0] = origin[0] + (block.Extents[0] - 0.5) * spacing[0];
      block.BoundsMax[0] = origin[0] + (block.Extents[1] + 0.5) * spacing[0];
      block.BoundsMin[1] = origin[1] + (block.Extents[2] - 0.5) * spacing[1];
      block.BoundsMax[1] = origin[1] + (block.Extents[3] + 0.5) * spacing[1];
      block.BoundsMin[2] = origin[2] + (block.Extents[4] - 0.5) * spacing[2];
      block.BoundsMax[2] = origin[2] + (block.Extents[5] + 0.5) * spacing[2];

      // Compute scalar min/max for this block (used for empty-space skipping).
      // Prefer reducing over pre-computed macrocell min/max from UpdateMinMaxTexture
      // (typically ~16³ = 4096 reductions vs. 250³ voxel reads for a 1 GB CT).
      double blockMin = VTK_DOUBLE_MAX;
      double blockMax = -VTK_DOUBLE_MAX;
      if (!this->MacrocellScalarMin.empty() && !this->MacrocellScalarMax.empty())
      {
        // Reduce over macrocells that overlap this block using constant DS = 4
        // and block extents relative to the full volume extent.
        const int DS = 4;
        const int mcDims0 = this->MinMaxDims[0];
        const int mcDims1 = this->MinMaxDims[1];
        const int mcDims2 = this->MinMaxDims[2];

        int relX0 = block.Extents[0] - fullExt[0];
        int relX1 = block.Extents[1] - fullExt[0];
        int relY0 = block.Extents[2] - fullExt[2];
        int relY1 = block.Extents[3] - fullExt[2];
        int relZ0 = block.Extents[4] - fullExt[4];
        int relZ1 = block.Extents[5] - fullExt[4];

        int mcX0 = relX0 / DS;
        int mcX1 = std::min(relX1 / DS, mcDims0 - 1);
        int mcY0 = relY0 / DS;
        int mcY1 = std::min(relY1 / DS, mcDims1 - 1);
        int mcZ0 = relZ0 / DS;
        int mcZ1 = std::min(relZ1 / DS, mcDims2 - 1);

        const float* mcMin = this->MacrocellScalarMin.data();
        const float* mcMax = this->MacrocellScalarMax.data();

        for (int mz = mcZ0; mz <= mcZ1; ++mz)
        {
          for (int my = mcY0; my <= mcY1; ++my)
          {
            for (int mx = mcX0; mx <= mcX1; ++mx)
            {
              vtkIdType ci = (static_cast<vtkIdType>(mz) * mcDims1 + my) * mcDims0 + mx;
              if (mcMin[ci] < blockMin) blockMin = mcMin[ci];
              if (mcMax[ci] > blockMax) blockMax = mcMax[ci];
            }
          }
        }
      }
      else
      {
        // Fallback: walk every voxel in this block (original path)
        int ext[6] = {
          block.Extents[0], block.Extents[1],
          block.Extents[2], block.Extents[3],
          block.Extents[4], block.Extents[5]
        };
        vtkIdType inc[3];
        input->GetIncrements(inc);

        switch (dataType)
        {
          case VTK_FLOAT:
          {
            const float* ptr = static_cast<const float*>(fullDataPtr);
            for (int z = ext[4]; z <= ext[5]; ++z)
            {
              for (int y = ext[2]; y <= ext[3]; ++y)
              {
                const float* row = ptr + z * inc[2] + y * inc[1] + ext[0] * inc[0];
                for (int x = ext[0]; x <= ext[1]; ++x, row += inc[0])
                {
                  double v = static_cast<double>(*row);
                  if (v < blockMin) blockMin = v;
                  if (v > blockMax) blockMax = v;
                }
              }
            }
            break;
          }
          case VTK_UNSIGNED_CHAR:
          {
            const unsigned char* ptr = static_cast<const unsigned char*>(fullDataPtr);
            for (int z = ext[4]; z <= ext[5]; ++z)
            {
              for (int y = ext[2]; y <= ext[3]; ++y)
              {
                const unsigned char* row = ptr + z * inc[2] + y * inc[1] + ext[0] * inc[0];
                for (int x = ext[0]; x <= ext[1]; ++x, row += inc[0])
                {
                  double v = static_cast<double>(*row);
                  if (v < blockMin) blockMin = v;
                  if (v > blockMax) blockMax = v;
                }
              }
            }
            break;
          }
          case VTK_UNSIGNED_SHORT:
          {
            const unsigned short* ptr = static_cast<const unsigned short*>(fullDataPtr);
            for (int z = ext[4]; z <= ext[5]; ++z)
            {
              for (int y = ext[2]; y <= ext[3]; ++y)
              {
                const unsigned short* row = ptr + z * inc[2] + y * inc[1] + ext[0] * inc[0];
                for (int x = ext[0]; x <= ext[1]; ++x, row += inc[0])
                {
                  double v = static_cast<double>(*row);
                  if (v < blockMin) blockMin = v;
                  if (v > blockMax) blockMax = v;
                }
              }
            }
            break;
          }
          default:
          {
            for (int z = ext[4]; z <= ext[5]; ++z)
            {
              for (int y = ext[2]; y <= ext[3]; ++y)
              {
                for (int x = ext[0]; x <= ext[1]; ++x)
                {
                  vtkIdType tupleIdx = z * inc[2] / inc[0] + y * inc[1] / inc[0] + x;
                  double v = scalars->GetComponent(tupleIdx, 0);
                  if (v < blockMin) blockMin = v;
                  if (v > blockMax) blockMax = v;
                }
              }
            }
            break;
          }
        }
      }

      this->BlockScalarRanges[idx][0] = blockMin;
      this->BlockScalarRanges[idx][1] = blockMax;

      // Check if this block is entirely transparent (empty-space skipping)
      vtkVolumeProperty* property = vol ? vol->GetProperty() : nullptr;
      vtkPiecewiseFunction* opacityFunc = property ? property->GetScalarOpacity() : nullptr;
      if (this->IsBlockEmpty(blockMin, blockMax, opacityFunc))
      {
        block.Texture = nullptr;
        continue;
      }

      // Create the 3D texture for this block
      MTLTextureDescriptor* texDesc = [[MTLTextureDescriptor alloc] init];
      texDesc.textureType = MTLTextureType3D;
      texDesc.pixelFormat = pixelFormat;
      texDesc.width = bDims[0];
      texDesc.height = bDims[1];
      texDesc.depth = bDims[2];
      texDesc.mipmapLevelCount = 1;
      texDesc.usage = MTLTextureUsageShaderRead;
      texDesc.storageMode = MTLStorageModePrivate;

      id<MTLTexture> tex = [device newTextureWithDescriptor:texDesc];
      [texDesc release];
      if (!tex)
      {
        vtkErrorMacro(<< "Failed to create block " << idx << " 3D texture ("
                      << bDims[0] << "x" << bDims[1] << "x" << bDims[2] << ")");
        return false;
      }
      AssignMetalObject(block.Texture, tex);

      // --- Per-block min-max texture generation ---
      if (hasOpacityFunc)
      {
        const int DS = 4; // Downsample factor
        int mmDims[3] = {
          std::max(1, (bDims[0] + DS - 1) / DS),
          std::max(1, (bDims[1] + DS - 1) / DS),
          std::max(1, (bDims[2] + DS - 1) / DS)
        };
        block.MinMaxDims[0] = mmDims[0];
        block.MinMaxDims[1] = mmDims[1];
        block.MinMaxDims[2] = mmDims[2];

        vtkIdType numCells = static_cast<vtkIdType>(mmDims[0]) * mmDims[1] * mmDims[2];
        const double blockRange = this->ScalarRange[1] - this->ScalarRange[0];
        const double blockRangeRecip = (blockRange > 0.0) ? (255.0 / blockRange) : 1.0;
        const double blockRangeOffset = this->ScalarRange[0];

        // 1. Slice from global macrocell min/max instead of walking every voxel.
        // Mapped from global MacrocellScalarMin/Max (computed in UpdateMinMaxTexture)
        // which covers the entire volume at DS=4 resolution.
        std::vector<uint8_t> rawMinMax(numCells, 255);
        const int mmDims0 = mmDims[0];
        const int mmDims1 = mmDims[1];
        const int mmDims2 = mmDims[2];

        const int mcDims0 = this->MinMaxDims[0];
        const int mcDims1 = this->MinMaxDims[1];
        const int mcDims2 = this->MinMaxDims[2];
        const bool haveGlobalMinMax = !this->MacrocellScalarMin.empty() && !this->MacrocellScalarMax.empty();

        if (haveGlobalMinMax)
        {
          // Fast path: use precomputed global macrocell data
          const float* mcMin = this->MacrocellScalarMin.data();
          const float* mcMax = this->MacrocellScalarMax.data();
          const int relX0 = texExt[0] - fullExt[0];
          const int relY0 = texExt[2] - fullExt[2];
          const int relZ0 = texExt[4] - fullExt[4];

          vtkSMPTools::For(0, numCells, [&](vtkIdType begin, vtkIdType end) {
            for (vtkIdType cellIdx = begin; cellIdx < end; ++cellIdx)
            {
              const int gx = static_cast<int>(cellIdx % mmDims0);
              const int gy = static_cast<int>((cellIdx / mmDims0) % mmDims1);
              const int gz = static_cast<int>(cellIdx / (mmDims0 * mmDims1));

              // Map block macrocell range to overlapping global macrocells
              const int xVoxelStart = relX0 + gx * DS;
              const int xVoxelEnd = relX0 + std::min((gx + 1) * DS, bDims[0]);
              const int yVoxelStart = relY0 + gy * DS;
              const int yVoxelEnd = relY0 + std::min((gy + 1) * DS, bDims[1]);
              const int zVoxelStart = relZ0 + gz * DS;
              const int zVoxelEnd = relZ0 + std::min((gz + 1) * DS, bDims[2]);

              const int mcX0 = std::min(xVoxelStart / DS, mcDims0 - 1);
              const int mcX1 = std::min(std::max(xVoxelEnd - 1, 0) / DS, mcDims0 - 1);
              const int mcY0 = std::min(yVoxelStart / DS, mcDims1 - 1);
              const int mcY1 = std::min(std::max(yVoxelEnd - 1, 0) / DS, mcDims1 - 1);
              const int mcZ0 = std::min(zVoxelStart / DS, mcDims2 - 1);
              const int mcZ1 = std::min(std::max(zVoxelEnd - 1, 0) / DS, mcDims2 - 1);

              float cellMin = 1e30f;
              float cellMax = -1e30f;
              for (int mz = mcZ0; mz <= mcZ1; ++mz)
              {
                for (int my = mcY0; my <= mcY1; ++my)
                {
                  for (int mx = mcX0; mx <= mcX1; ++mx)
                  {
                    vtkIdType idx = (static_cast<vtkIdType>(mz) * mcDims1 + my) * mcDims0 + mx;
                    if (mcMin[idx] < cellMin) cellMin = mcMin[idx];
                    if (mcMax[idx] > cellMax) cellMax = mcMax[idx];
                  }
                }
              }

              bool empty = true;
              if (cellMin <= cellMax)
              {
                int idxMin = std::max(0,
                  std::min(255, static_cast<int>((cellMin - blockRangeOffset) * blockRangeRecip)));
                int idxMax = std::max(0,
                  std::min(255, static_cast<int>((cellMax - blockRangeOffset) * blockRangeRecip)));
                for (int i = idxMin; i <= idxMax; ++i)
                {
                  if (opacityTable[i] > 0.0)
                  {
                    empty = false;
                    break;
                  }
                }
              }
              rawMinMax[cellIdx] = empty ? 255 : 0;
            }
          });
        }
        else
        {
          // Fallback: walk every voxel (original path)
          vtkSMPTools::For(0, numCells, [&](vtkIdType begin, vtkIdType end) {
            for (vtkIdType cellIdx = begin; cellIdx < end; ++cellIdx)
            {
              const int gx = static_cast<int>(cellIdx % mmDims0);
              const int gy = static_cast<int>((cellIdx / mmDims0) % mmDims1);
              const int gz = static_cast<int>(cellIdx / (mmDims0 * mmDims1));

              const int zStart = texExt[4] + gz * DS;
              const int zEnd = std::min(zStart + DS, texExt[5] + 1);
              const int yStart = texExt[2] + gy * DS;
              const int yEnd = std::min(yStart + DS, texExt[3] + 1);
              const int xStart = texExt[0] + gx * DS;
              const int xEnd = std::min(xStart + DS, texExt[1] + 1);

              float cellMin = 1e30f;
              float cellMax = -1e30f;

              for (int z = zStart; z < zEnd; ++z)
              {
                int zOffset = z - fullExt[4];
                for (int y = yStart; y < yEnd; ++y)
                {
                  int yOffset = y - fullExt[2];
                  for (int x = xStart; x < xEnd; ++x)
                  {
                    int xOffset = x - fullExt[0];
                    float v = 0.0f;
                    switch (dataType)
                    {
                      case VTK_FLOAT:
                        v = static_cast<float>(
                          static_cast<const float*>(fullDataPtr)[zOffset * inc[2] + yOffset * inc[1] + xOffset * inc[0]]);
                        break;
                      case VTK_UNSIGNED_CHAR:
                        v = static_cast<float>(
                          static_cast<const unsigned char*>(fullDataPtr)[zOffset * inc[2] + yOffset * inc[1] + xOffset * inc[0]]);
                        break;
                      case VTK_UNSIGNED_SHORT:
                        v = static_cast<float>(
                          static_cast<const unsigned short*>(fullDataPtr)[zOffset * inc[2] + yOffset * inc[1] + xOffset * inc[0]]);
                        break;
                      case VTK_SHORT:
                        v = static_cast<float>(
                          static_cast<const short*>(fullDataPtr)[zOffset * inc[2] + yOffset * inc[1] + xOffset * inc[0]]);
                        break;
                      default:
                      {
                        vtkIdType tupleIdx = zOffset * (inc[2] / inc[0]) + yOffset * (inc[1] / inc[0]) + xOffset;
                        v = static_cast<float>(scalars->GetComponent(tupleIdx, 0));
                        break;
                      }
                    }
                    if (v < cellMin) cellMin = v;
                    if (v > cellMax) cellMax = v;
                  }
                }
              }

              bool empty = true;
              if (cellMin <= cellMax)
              {
                int idxMin = std::max(0,
                  std::min(255, static_cast<int>((cellMin - blockRangeOffset) * blockRangeRecip)));
                int idxMax = std::max(0,
                  std::min(255, static_cast<int>((cellMax - blockRangeOffset) * blockRangeRecip)));
                for (int i = idxMin; i <= idxMax; ++i)
                {
                  if (opacityTable[i] > 0.0)
                  {
                    empty = false;
                    break;
                  }
                }
              }
              rawMinMax[cellIdx] = empty ? 255 : 0;
            }
          });
        }

        // 2. Dilation pass (avoid holes at macrocell boundaries)
        std::vector<uint8_t> minMaxData(numCells, 255);
        vtkSMPTools::For(0, numCells, [&](vtkIdType begin, vtkIdType end) {
          for (vtkIdType cellIdx = begin; cellIdx < end; ++cellIdx)
          {
            const int gx = static_cast<int>(cellIdx % mmDims0);
            const int gy = static_cast<int>((cellIdx / mmDims0) % mmDims1);
            const int gz = static_cast<int>(cellIdx / (mmDims0 * mmDims1));

            int z0 = std::max(0, gz - 1), z1 = std::min(mmDims2 - 1, gz + 1);
            int y0 = std::max(0, gy - 1), y1 = std::min(mmDims1 - 1, gy + 1);
            int x0 = std::max(0, gx - 1), x1 = std::min(mmDims0 - 1, gx + 1);

            bool solid = false;
            for (int nz = z0; nz <= z1 && !solid; ++nz)
              for (int ny = y0; ny <= y1 && !solid; ++ny)
                for (int nx = x0; nx <= x1 && !solid; ++nx)
                  if (rawMinMax[(nz * mmDims1 + ny) * mmDims0 + nx] == 0) solid = true;

            minMaxData[cellIdx] = solid ? 0 : 255;
          }
        });

        // 3. Create and upload the Metal 3D texture
        MTLTextureDescriptor* mmDesc = [[MTLTextureDescriptor alloc] init];
        mmDesc.textureType = MTLTextureType3D;
        mmDesc.pixelFormat = MTLPixelFormatR8Unorm;
        mmDesc.width = mmDims0;
        mmDesc.height = mmDims1;
        mmDesc.depth = mmDims2;
        mmDesc.mipmapLevelCount = 1;
        mmDesc.usage = MTLTextureUsageShaderRead;
        mmDesc.storageMode = MTLStorageModeShared;

        id<MTLTexture> mmTex = [device newTextureWithDescriptor:mmDesc];
        [mmDesc release];
        if (mmTex)
        {
          MTLRegion region = MTLRegionMake3D(0, 0, 0, mmDims0, mmDims1, mmDims2);
          NSUInteger mmBytesPerRow = mmDims0 * sizeof(uint8_t);
          NSUInteger mmBytesPerImage = mmBytesPerRow * mmDims1;
          [mmTex replaceRegion:region
                  mipmapLevel:0
                        slice:0
                    withBytes:minMaxData.data()
                  bytesPerRow:mmBytesPerRow
                bytesPerImage:mmBytesPerImage];

          AssignMetalObject(block.MinMaxTexture, mmTex);
        }
        else
        {
          block.MinMaxTexture = nullptr;
        }
      }
      else
      {
        block.MinMaxTexture = nullptr;
      }
      // --- End per-block min-max texture generation ---

      if (gpuConversionUsed && gpuFullTex)
      {
        // GPU-converted full volume: extract per-block sub-region via texture-to-texture copy
        [blit copyFromTexture:gpuFullTex
                  sourceSlice:0
                  sourceLevel:0
                 sourceOrigin:MTLOriginMake(
                     static_cast<NSUInteger>(texExt[0] - fullExt[0]),
                     static_cast<NSUInteger>(texExt[2] - fullExt[2]),
                     static_cast<NSUInteger>(texExt[4] - fullExt[4]))
                   sourceSize:MTLSizeMake(
                     static_cast<NSUInteger>(bDims[0]),
                     static_cast<NSUInteger>(bDims[1]),
                     static_cast<NSUInteger>(bDims[2]))
                    toTexture:tex
             destinationSlice:0
             destinationLevel:0
            destinationOrigin:MTLOriginMake(0, 0, 0)];
      }
      else
      {
        // CPU-converted staging buffer: extract per-block sub-region via buffer-to-texture copy
        NSUInteger sourceOffset =
          static_cast<NSUInteger>(texExt[4] - fullExt[4]) * srcBytesPerImage +
          static_cast<NSUInteger>(texExt[2] - fullExt[2]) * srcBytesPerRow +
          static_cast<NSUInteger>(texExt[0] - fullExt[0]) * bytesPerVoxel;

        [blit copyFromBuffer:stagingBuf
                sourceOffset:sourceOffset
         sourceBytesPerRow:srcBytesPerRow
       sourceBytesPerImage:srcBytesPerImage
                sourceSize:MTLSizeMake(bDims[0], bDims[1], bDims[2])
                 toTexture:tex
          destinationSlice:0
          destinationLevel:0
         destinationOrigin:MTLOriginMake(0, 0, 0)];
      }
    }

    [blit endEncoding];

    // Phase 4: Precompute per-block gradient/normal textures for partitioned volumes.
    if (this->UsePrecomputedNormals && !this->Blocks.empty())
    {
      if (!this->EnsureShaderLibrary(mtlDeviceVoid))
      {
        return false;
      }
      id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary;

      if (!this->NormalComputePipeline)
      {
        id<MTLFunction> kernelFunc = [library newFunctionWithName:@"volume_compute_normals"];
        if (kernelFunc)
        {
          NSError* err = nil;
          id<MTLComputePipelineState> cps =
            [device newComputePipelineStateWithFunction:kernelFunc error:&err];
          [kernelFunc release];
          if (cps)
          {
            AssignMetalObject(this->NormalComputePipeline, cps);
          }
        }
      }

      if (this->NormalComputePipeline)
      {
        double scalarRange = this->ScalarRange[1] - this->ScalarRange[0];
        if (scalarRange <= 0.0) scalarRange = 1.0;
        float normFactorLocal = this->ScalarNormalizationFactor;
        float localGradNormFactor = static_cast<float>(scalarRange * 0.25 / normFactorLocal);

        struct NormalComputeUniforms {
          uint32_t dimX, dimY, dimZ;
          float gsX, gsY, gsZ;
          float scalarScale;
          float scalarBias;
          float gradNormFactor;
        };

        id<MTLComputeCommandEncoder> compEnc = [uploadCmdBuf computeCommandEncoder];
        compEnc.label = @"VTK Block Normal Compute";

        for (auto& block : this->Blocks)
        {
          if (!block.Texture) continue;

          id<MTLTexture> blockTex = (__bridge id<MTLTexture>)block.Texture;
          int bdims[3] = { static_cast<int>(blockTex.width),
                           static_cast<int>(blockTex.height),
                           static_cast<int>(blockTex.depth) };

          // Create per-block normal texture
          MTLTextureDescriptor* nd = [[MTLTextureDescriptor alloc] init];
          nd.textureType = MTLTextureType3D;
          nd.pixelFormat = MTLPixelFormatRGBA8Unorm;
          nd.width = bdims[0];
          nd.height = bdims[1];
          nd.depth = bdims[2];
          nd.mipmapLevelCount = 1;
          nd.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
          nd.storageMode = MTLStorageModePrivate;

          id<MTLTexture> blockNrm = [device newTextureWithDescriptor:nd];
          [nd release];
          if (!blockNrm) continue;
          AssignMetalObject(block.NormalTexture, blockNrm);

          NormalComputeUniforms u;
          u.dimX = static_cast<uint32_t>(bdims[0]);
          u.dimY = static_cast<uint32_t>(bdims[1]);
          u.dimZ = static_cast<uint32_t>(bdims[2]);
          u.gsX = 1.0f / std::max(bdims[0], 1);
          u.gsY = 1.0f / std::max(bdims[1], 1);
          u.gsZ = 1.0f / std::max(bdims[2], 1);
          u.scalarScale = 1.0f / std::max(static_cast<float>((this->ScalarRange[1] - this->ScalarRange[0]) / normFactorLocal), 1e-6f);
          u.scalarBias = -(static_cast<float>(this->ScalarRange[0] / normFactorLocal)) * u.scalarScale;
          u.gradNormFactor = localGradNormFactor;

          [compEnc setComputePipelineState:(__bridge id<MTLComputePipelineState>)this->NormalComputePipeline];
          [compEnc setTexture:blockTex atIndex:0];
          [compEnc setTexture:blockNrm atIndex:1];
          [compEnc setBytes:&u length:sizeof(u) atIndex:0];

          MTLSize gridSize = MTLSizeMake(bdims[0], bdims[1], bdims[2]);
          NSUInteger tgw_max = 16;
          NSUInteger tgw_x = std::min(tgw_max, static_cast<NSUInteger>(bdims[0]));
          NSUInteger tgw_y = std::min(tgw_max, static_cast<NSUInteger>(bdims[1]));
          NSUInteger tgw_z = std::min(static_cast<NSUInteger>(1024) / (tgw_x * tgw_y), static_cast<NSUInteger>(bdims[2]));
          MTLSize threadGroupSize = MTLSizeMake(tgw_x, tgw_y, tgw_z);
          [compEnc dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        }

        [compEnc endEncoding];
      }
    }

    [uploadCmdBuf commit];
    if (gpuFullTex)
    {
      [gpuFullTex release];
    }
    [stagingBuf release];
  }

  return true;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::IsCameraInside(
  vtkRenderer* ren, vtkVolume* vol)
{
  vtkNew<vtkMatrix4x4> dataToWorld;
  vol->GetModelToWorldMatrix(dataToWorld);

  vtkCamera* cam = ren->GetActiveCamera();

  double planes[24];
  cam->GetFrustumPlanes(ren->GetTiledAspectRatio(), planes);

  // Transform 8 bounding-box corners from data-space to world-space
  double geometry[24] = {
    this->ModelBounds[0], this->ModelBounds[2], this->ModelBounds[4],
    this->ModelBounds[1], this->ModelBounds[2], this->ModelBounds[4],
    this->ModelBounds[1], this->ModelBounds[3], this->ModelBounds[4],
    this->ModelBounds[0], this->ModelBounds[3], this->ModelBounds[4],
    this->ModelBounds[0], this->ModelBounds[2], this->ModelBounds[5],
    this->ModelBounds[1], this->ModelBounds[2], this->ModelBounds[5],
    this->ModelBounds[1], this->ModelBounds[3], this->ModelBounds[5],
    this->ModelBounds[0], this->ModelBounds[3], this->ModelBounds[5],
  };

  double in[4];
  in[3] = 1.0;
  double out[4];
  double worldGeometry[24];

  for (int i = 0; i < 8; ++i)
  {
    in[0] = geometry[i * 3];
    in[1] = geometry[i * 3 + 1];
    in[2] = geometry[i * 3 + 2];
    dataToWorld->MultiplyPoint(in, out);
    worldGeometry[i * 3] = out[0] / out[3];
    worldGeometry[i * 3 + 1] = out[1] / out[3];
    worldGeometry[i * 3 + 2] = out[2] / out[3];
  }

  // Test if near frustum plane (index 4*4=16) intersects the bounding box
  // Returns true if some corners are on positive side and some on negative side
  bool hasPositive = false;
  bool hasNegative = false;
  bool hasZero = false;

  for (int i = 0; i < 8; ++i)
  {
    double val = planes[4 * 4] * worldGeometry[i * 3] +
      planes[4 * 4 + 1] * worldGeometry[i * 3 + 1] +
      planes[4 * 4 + 2] * worldGeometry[i * 3 + 2] +
      planes[4 * 4 + 3];

    if (val < 0)
    {
      hasNegative = true;
    }
    else if (val > 0)
    {
      hasPositive = true;
    }
    else
    {
      hasZero = true;
    }
  }

  return hasZero || (hasNegative && hasPositive);
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::SetClippingPlaneUniforms(
  void* uniformsVoid, vtkRenderer* ren, vtkVolume* vol,
  vtkMatrix4x4* modelMatrix, vtkMatrix4x4* invModelMatrix)
{
  VolumeMapperUniforms* uniforms = static_cast<VolumeMapperUniforms*>(uniformsVoid);

  if (!this->GetClippingPlanes())
  {
    uniforms->UseClipping = 0.0f;
    uniforms->NumClippingPlanes = 0.0f;
    return;
  }

  uniforms->UseClipping = 1.0f;

  double* modelBounds = this->ModelBounds;
  double boundsSize[3] = {
    modelBounds[1] - modelBounds[0],
    modelBounds[3] - modelBounds[2],
    modelBounds[5] - modelBounds[4]
  };
  for (int k = 0; k < 3; ++k)
  {
    if (boundsSize[k] < 1e-10)
      boundsSize[k] = 1.0;
  }

  // Store plane pointers for indexed access
  float* planeOrigins[8] = {
    uniforms->ClippingPlane0Origin, uniforms->ClippingPlane1Origin,
    uniforms->ClippingPlane2Origin, uniforms->ClippingPlane3Origin,
    uniforms->ClippingPlane4Origin, uniforms->ClippingPlane5Origin,
    uniforms->ClippingPlane6Origin, uniforms->ClippingPlane7Origin
  };
  float* planeNormals[8] = {
    uniforms->ClippingPlane0Normal, uniforms->ClippingPlane1Normal,
    uniforms->ClippingPlane2Normal, uniforms->ClippingPlane3Normal,
    uniforms->ClippingPlane4Normal, uniforms->ClippingPlane5Normal,
    uniforms->ClippingPlane6Normal, uniforms->ClippingPlane7Normal
  };

  int numPlanes = 0;

  this->ClippingPlanes->InitTraversal();
  vtkPlane* plane;
  while ((plane = this->ClippingPlanes->GetNextItem()) && numPlanes < 8)
  {
    double planeOrigin[3], planeNormal[3];
    plane->GetOrigin(planeOrigin);
    plane->GetNormal(planeNormal);

    // Origin: transform as a point from world to model/data space.
    double originLocal[4] = { planeOrigin[0], planeOrigin[1], planeOrigin[2], 1.0 };
    invModelMatrix->MultiplyPoint(originLocal, originLocal);

    if (fabs(originLocal[3]) > 1e-12)
    {
      originLocal[0] /= originLocal[3];
      originLocal[1] /= originLocal[3];
      originLocal[2] /= originLocal[3];
    }

    double originVol[3] = {
      (originLocal[0] - modelBounds[0]) / boundsSize[0],
      (originLocal[1] - modelBounds[2]) / boundsSize[1],
      (originLocal[2] - modelBounds[4]) / boundsSize[2]
    };

    // Normal: transform as a normal/covector.
    // We want transpose(modelMatrix) * normalWorld.
    // VTK matrices are accessed as GetElement(row, col).
    // For column c of transpose(M):
    //   (M^T n)[c] = sum_r M[r][c] * n[r]
    double normalModel[3];
    normalModel[0] =
      modelMatrix->GetElement(0, 0) * planeNormal[0] +
      modelMatrix->GetElement(1, 0) * planeNormal[1] +
      modelMatrix->GetElement(2, 0) * planeNormal[2];

    normalModel[1] =
      modelMatrix->GetElement(0, 1) * planeNormal[0] +
      modelMatrix->GetElement(1, 1) * planeNormal[1] +
      modelMatrix->GetElement(2, 1) * planeNormal[2];

    normalModel[2] =
      modelMatrix->GetElement(0, 2) * planeNormal[0] +
      modelMatrix->GetElement(1, 2) * planeNormal[1] +
      modelMatrix->GetElement(2, 2) * planeNormal[2];

    // Scale into normalized volume space.
    double normalVol[3] = {
      normalModel[0] * boundsSize[0],
      normalModel[1] * boundsSize[1],
      normalModel[2] * boundsSize[2]
    };

    double normalLen = sqrt(
      normalVol[0] * normalVol[0] +
      normalVol[1] * normalVol[1] +
      normalVol[2] * normalVol[2]);

    if (normalLen > 1e-10)
    {
      normalVol[0] /= normalLen;
      normalVol[1] /= normalLen;
      normalVol[2] /= normalLen;
    }

    // Store as float4 (origin.xyz, 1.0) and (normal.xyz, 0.0)
    planeOrigins[numPlanes][0] = static_cast<float>(originVol[0]);
    planeOrigins[numPlanes][1] = static_cast<float>(originVol[1]);
    planeOrigins[numPlanes][2] = static_cast<float>(originVol[2]);
    planeOrigins[numPlanes][3] = 1.0f;

    planeNormals[numPlanes][0] = static_cast<float>(normalVol[0]);
    planeNormals[numPlanes][1] = static_cast<float>(normalVol[1]);
    planeNormals[numPlanes][2] = static_cast<float>(normalVol[2]);
    planeNormals[numPlanes][3] = 0.0f;

    numPlanes++;
  }

  uniforms->NumClippingPlanes = static_cast<float>(numPlanes);
}
//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::SetupBuffers(
  void* mtlDeviceVoid, vtkRenderer* ren, vtkVolume* vol, vtkImageData* input)
{
  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

    if (!this->UniformBuffers[0])
    {
      for (int i = 0; i < 3; ++i)
      {
        id<MTLBuffer> buf = [device newBufferWithLength:sizeof(VolumeMapperUniforms)
                                                options:MTLResourceStorageModeShared];
        if (!buf)
        {
          vtkErrorMacro("Failed to create uniform buffer");
          return false;
        }
        AssignMetalObject(this->UniformBuffers[i], buf);
      }
    }

    // Use model-space bounds for vertex positions (using extent for correctness)
    if (input)
    {
      int ext[6];
      double origin[3], spacing[3];
      input->GetExtent(ext);
      input->GetOrigin(origin);
      input->GetSpacing(spacing);
      double x0 = origin[0] + spacing[0] * ext[0];
      double x1 = origin[0] + spacing[0] * ext[1];
      double y0 = origin[1] + spacing[1] * ext[2];
      double y1 = origin[1] + spacing[1] * ext[3];
      double z0 = origin[2] + spacing[2] * ext[4];
      double z1 = origin[2] + spacing[2] * ext[5];
      this->ModelBounds[0] = std::min(x0, x1);
      this->ModelBounds[1] = std::max(x0, x1);
      this->ModelBounds[2] = std::min(y0, y1);
      this->ModelBounds[3] = std::max(y0, y1);
      this->ModelBounds[4] = std::min(z0, z1);
      this->ModelBounds[5] = std::max(z0, z1);
    }

    // Check if camera is inside the volume
    bool cameraInside = this->IsCameraInside(ren, vol);

    // Phase 6: Fullscreen camera-inside path skips vertex buffer creation entirely.
    // The fullscreen triangle is generated by vertex_fullscreen_main with no buffers.
    if (cameraInside && this->UseFullscreenCameraInside)
    {
      this->CameraWasInsideInLastUpdate = true;
      return true;
    }

    // Check if geometry needs rebuild
    bool needsVertexRebuild = !this->VertexBuffer;
    needsVertexRebuild |= (this->VolumeUploadTime.GetMTime() > this->VertexBufferUploadTime.GetMTime());
    needsVertexRebuild |= (this->GetMTime() > this->VertexBufferUploadTime.GetMTime());
    needsVertexRebuild |= (cameraInside != this->CameraWasInsideInLastUpdate);

    if (cameraInside)
    {
      vtkCamera* cam = ren->GetActiveCamera();
      if (cam)
      {
        needsVertexRebuild |= (cam->GetMTime() > this->VertexBufferUploadTime.GetMTime());
      }
      needsVertexRebuild |= (ren->GetMTime() > this->VertexBufferUploadTime.GetMTime());
    }

    if (needsVertexRebuild)
    {
      // Camera outside: simple 8-vertex box (original fast path)
      // Camera inside: clip against near plane, densify, triangulate
      if (!cameraInside)
      {
        float vertices[24];
        unsigned int indices[36];

        if (!this->Blocks.empty())
        {
          // Unit cube [0,1] — the vertex shader scales to each block's model-space bounds.
          float unitVerts[] = {
            0,0,0, 1,0,0, 1,1,0, 0,1,0,
            0,0,1, 1,0,1, 1,1,1, 0,1,1
          };
          memcpy(vertices, unitVerts, sizeof(unitVerts));
        }
        else
        {
          // Unit cube [0,1] for all volume rendering. The vertex shader
          // scales this to the block's model-space bounds.
          float unitVerts[] = {
            0,0,0, 1,0,0, 1,1,0, 0,1,0,
            0,0,1, 1,0,1, 1,1,1, 0,1,1
          };
          memcpy(vertices, unitVerts, sizeof(unitVerts));
        }

        indices[0] = 0;  indices[1] = 2;  indices[2] = 1;  indices[3] = 0;  indices[4] = 3;  indices[5] = 2;
        indices[6] = 4;  indices[7] = 5;  indices[8] = 6;  indices[9] = 4;  indices[10] = 6; indices[11] = 7;
        indices[12] = 0; indices[13] = 7; indices[14] = 3; indices[15] = 0; indices[16] = 4; indices[17] = 7;
        indices[18] = 1; indices[19] = 2; indices[20] = 6; indices[21] = 1; indices[22] = 6; indices[23] = 5;
        indices[24] = 3; indices[25] = 6; indices[26] = 2; indices[27] = 3; indices[28] = 7; indices[29] = 6;
        indices[30] = 0; indices[31] = 1; indices[32] = 5; indices[33] = 0; indices[34] = 5; indices[35] = 4;

        this->IndexCount = sizeof(indices) / sizeof(unsigned int);

        // Release old buffers
        if (this->VertexBuffer)
        {
          ReleaseMetalObject(this->VertexBuffer);
        }
        if (this->IndexBuffer)
        {
          ReleaseMetalObject(this->IndexBuffer);
        }

        {
          id<MTLBuffer> vbuf = [device newBufferWithBytes:vertices
                                                  length:sizeof(vertices)
                                                 options:MTLResourceStorageModeShared];
          if (!vbuf)
          {
            vtkErrorMacro("Failed to create vertex buffer");
            return false;
          }
          AssignMetalObject(this->VertexBuffer, vbuf);
        }

        {
          id<MTLBuffer> ibuf = [device newBufferWithBytes:indices
                                                  length:sizeof(indices)
                                                 options:MTLResourceStorageModeShared];
          if (!ibuf)
          {
            vtkErrorMacro("Failed to create index buffer");
            return false;
          }
          AssignMetalObject(this->IndexBuffer, ibuf);
        }

        this->CameraWasInsideInLastUpdate = false;
      }
      else
      {
        // Camera inside: clip bounding box against near plane
        vtkNew<vtkPolyData> boxSource;

        {
          vtkNew<vtkCellArray> cells;
          vtkNew<vtkPoints> points;
          points->SetDataTypeToDouble();

          double geometry[24] = {
            this->ModelBounds[0], this->ModelBounds[2], this->ModelBounds[4],
            this->ModelBounds[1], this->ModelBounds[2], this->ModelBounds[4],
            this->ModelBounds[1], this->ModelBounds[3], this->ModelBounds[4],
            this->ModelBounds[0], this->ModelBounds[3], this->ModelBounds[4],
            this->ModelBounds[0], this->ModelBounds[2], this->ModelBounds[5],
            this->ModelBounds[1], this->ModelBounds[2], this->ModelBounds[5],
            this->ModelBounds[1], this->ModelBounds[3], this->ModelBounds[5],
            this->ModelBounds[0], this->ModelBounds[3], this->ModelBounds[5],
          };

          for (int i = 0; i < 8; ++i)
          {
            points->InsertNextPoint(geometry + i * 3);
          }

          // 6 faces 12 triangles (clockwise winding for vtkClipConvexPolyData)
          int tris[36] = {
            0, 1, 2,
            1, 3, 2,
            1, 5, 3,
            5, 7, 3,
            5, 4, 7,
            4, 6, 7,
            4, 0, 6,
            0, 2, 6,
            2, 3, 6,
            3, 7, 6,
            0, 4, 1,
            1, 4, 5
          };

          for (int i = 0; i < 12; ++i)
          {
            cells->InsertNextCell(3);
            // Clockwise convention for vtkClipConvexPolyData
            cells->InsertCellPoint(tris[i * 3]);
            cells->InsertCellPoint(tris[i * 3 + 2]);
            cells->InsertCellPoint(tris[i * 3 + 1]);
          }

          boxSource->SetPoints(points);
          boxSource->SetPolys(cells);
        }

        // Clip bounding box against near plane
        vtkNew<vtkMatrix4x4> dataToWorld;
        vol->GetModelToWorldMatrix(dataToWorld);

        vtkCamera* cam = ren->GetActiveCamera();

        double fplanes[24];
        cam->GetFrustumPlanes(ren->GetTiledAspectRatio(), fplanes);

        // Extract near frustum plane (index 4*4=16)
        double pOrigin[4];
        pOrigin[3] = 1.0;
        double pNormal[3];

        for (int i = 0; i < 3; ++i)
        {
          pNormal[i] = fplanes[16 + i];
          pOrigin[i] = -fplanes[16 + 3] * fplanes[16 + i];
        }

        // Transform normal to volume coordinates using inverse transpose
        // For transforming normals from world space to object space, we need
        // the inverse transpose of the model matrix
        vtkNew<vtkMatrix4x4> worldToData;
        vtkMatrix4x4::Invert(dataToWorld, worldToData);

        // Transform origin point to volume coordinates using inverse matrix
        worldToData->MultiplyPoint(pOrigin, pOrigin);

        // Transform normal using transpose of inverse (i.e., inverse transpose)
        double* invMat = worldToData->GetData();
        double pNormalV[3];
        pNormalV[0] = pNormal[0] * invMat[0] + pNormal[1] * invMat[1] + pNormal[2] * invMat[2];
        pNormalV[1] = pNormal[0] * invMat[4] + pNormal[1] * invMat[5] + pNormal[2] * invMat[6];
        pNormalV[2] = pNormal[0] * invMat[8] + pNormal[1] * invMat[9] + pNormal[2] * invMat[10];
        vtkMath::Normalize(pNormalV);

        // Apply offset to prevent hardware near-plane clipping
        double offset = (cam->GetClippingRange()[1] - cam->GetClippingRange()[0]) * 0.001;
        double minOffset = static_cast<double>(std::numeric_limits<float>::epsilon()) * 1000.0;
        offset = offset < minOffset ? minOffset : offset;

        for (int i = 0; i < 3; ++i)
        {
          pOrigin[i] += (pNormalV[i] * offset);
        }

        vtkNew<vtkPlane> nearPlane;
        nearPlane->SetOrigin(pOrigin);
        nearPlane->SetNormal(pNormalV);

        vtkNew<vtkPlaneCollection> planes;
        planes->RemoveAllItems();
        planes->AddItem(nearPlane);

        vtkNew<vtkClipConvexPolyData> clip;
        clip->SetInputData(boxSource);
        clip->SetPlanes(planes);

        // Clip, densify, then triangulate to guarantee triangle output
        vtkNew<vtkDensifyPolyData> densifyPolyData;
        densifyPolyData->SetInputConnection(clip->GetOutputPort());
        densifyPolyData->SetNumberOfSubdivisions(2);

        vtkNew<vtkTriangleFilter> triFilter;
        triFilter->SetInputConnection(densifyPolyData->GetOutputPort());
        triFilter->Update();

        vtkPolyData* finalPolyData = triFilter->GetOutput();
        vtkPoints* points = finalPolyData->GetPoints();
        vtkCellArray* polys = finalPolyData->GetPolys();

        // Normalize clipped points from model-space to [0,1] for the vertex shader
        double bmin[3] = { this->ModelBounds[0], this->ModelBounds[2], this->ModelBounds[4] };
        double bsize[3] = {
          this->ModelBounds[1] - this->ModelBounds[0],
          this->ModelBounds[3] - this->ModelBounds[2],
          this->ModelBounds[5] - this->ModelBounds[4]
        };
        for (int k = 0; k < 3; ++k)
        {
          if (std::fabs(bsize[k]) < 1e-10) bsize[k] = 1.0;
        }

        std::vector<float> vertices;
        vertices.reserve(points->GetNumberOfPoints() * 3);

        for (vtkIdType i = 0; i < points->GetNumberOfPoints(); ++i)
        {
          double pt[3];
          points->GetPoint(i, pt);
          vertices.push_back(static_cast<float>((pt[0] - bmin[0]) / bsize[0]));
          vertices.push_back(static_cast<float>((pt[1] - bmin[1]) / bsize[1]));
          vertices.push_back(static_cast<float>((pt[2] - bmin[2]) / bsize[2]));
        }

        std::vector<unsigned int> indices;
        vtkIdType npts;
        const vtkIdType* pts;

        polys->InitTraversal();
        while (polys->GetNextCell(npts, pts))
        {
          if (npts != 3) continue;
          indices.push_back(static_cast<unsigned int>(pts[0]));
          indices.push_back(static_cast<unsigned int>(pts[1]));
          indices.push_back(static_cast<unsigned int>(pts[2]));
        }

        if (indices.empty())
        {
          vtkErrorMacro("Camera-inside clipped proxy produced no triangles");
          return false;
        }

        this->IndexCount = static_cast<int>(indices.size());

        // Release old buffers
        if (this->VertexBuffer)
        {
          ReleaseMetalObject(this->VertexBuffer);
        }
        if (this->IndexBuffer)
        {
          ReleaseMetalObject(this->IndexBuffer);
        }

        // Create new vertex buffer
        {
          id<MTLBuffer> vbuf = [device newBufferWithBytes:vertices.data()
                                                  length:vertices.size() * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
          if (!vbuf)
          {
            vtkErrorMacro("Failed to create vertex buffer");
            return false;
          }
          AssignMetalObject(this->VertexBuffer, vbuf);
        }

        // Create new index buffer
        {
          id<MTLBuffer> ibuf = [device newBufferWithBytes:indices.data()
                                                  length:indices.size() * sizeof(unsigned int)
                                                 options:MTLResourceStorageModeShared];
          if (!ibuf)
          {
            vtkErrorMacro("Failed to create index buffer");
            return false;
          }
          AssignMetalObject(this->IndexBuffer, ibuf);
        }

        this->CameraWasInsideInLastUpdate = true;
      }

      this->VertexBufferUploadTime.Modified();
    }
  }

  return this->VertexBuffer && this->IndexBuffer && this->UniformBuffers[0];
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::SetupPipeline(void* mtlDeviceVoid, vtkRenderer* ren)
{
  // Get sample count and invalidate PSO if it changed (e.g., MSAA toggled)
  auto* metalRenderWindow = vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());
  int sampleCount = metalRenderWindow ? metalRenderWindow->GetEffectiveSampleCount() : 1;

  if (this->PipelineState && sampleCount != this->CurrentSampleCount)
  {
    ReleaseMetalObject(this->PipelineState);
  }

  // Accumulation pipeline is always rasterSampleCount=1
  // (used only for offscreen rendering), so it doesn't need invalidation
  // when the window's MSAA sample count changes.

  if (this->PipelineState)
  {
    return true;
  }

  if (!this->EnsureShaderLibrary(mtlDeviceVoid))
  {
    return false;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;
    id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary;

    // Create dummy depth texture (1x1 R32Float with value 1.0) for when no real depth texture is bound
    if (!this->DummyDepthTexture)
    {
      MTLTextureDescriptor* dummyDesc = [[MTLTextureDescriptor alloc] init];
      dummyDesc.textureType = MTLTextureType2D;
      dummyDesc.pixelFormat = MTLPixelFormatR32Float;
      dummyDesc.width = 1;
      dummyDesc.height = 1;
      dummyDesc.mipmapLevelCount = 1;
      dummyDesc.usage = MTLTextureUsageShaderRead;
      dummyDesc.storageMode = MTLStorageModeShared;

      id<MTLTexture> dummyTex = [device newTextureWithDescriptor:dummyDesc];
      [dummyDesc release];
      if (dummyTex)
      {
        float one = 1.0f;
        MTLRegion region = MTLRegionMake2D(0, 0, 1, 1);
        [dummyTex replaceRegion:region mipmapLevel:0 withBytes:&one bytesPerRow:sizeof(float)];
        AssignMetalObject(this->DummyDepthTexture, dummyTex);
      }
    }

    // Note: constexpr samplers in MetalShaders.metal replace all runtime
    // sampler state objects. No sampler creation needed here.

    // Create dummy 3D textures for fallback bindings (prevent nil texture binds).
    if (!this->DummyVolumeTexture)
    {
      MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
      desc.textureType = MTLTextureType3D;
      desc.pixelFormat = MTLPixelFormatR32Float;
      desc.width = 1;
      desc.height = 1;
      desc.depth = 1;
      desc.mipmapLevelCount = 1;
      desc.usage = MTLTextureUsageShaderRead;
      desc.storageMode = MTLStorageModeShared;
      id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
      if (tex)
      {
        float zero = 0.0f;
        MTLRegion region = MTLRegionMake3D(0, 0, 0, 1, 1, 1);
        [tex replaceRegion:region mipmapLevel:0 slice:0 withBytes:&zero
               bytesPerRow:sizeof(float) bytesPerImage:sizeof(float)];
      }
      AssignMetalObject(this->DummyVolumeTexture, tex);
      [desc release];
    }

    if (!this->DummyMaskTexture)
    {
      MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
      desc.textureType = MTLTextureType3D;
      desc.pixelFormat = MTLPixelFormatR32Float;
      desc.width = 1;
      desc.height = 1;
      desc.depth = 1;
      desc.mipmapLevelCount = 1;
      desc.usage = MTLTextureUsageShaderRead;
      desc.storageMode = MTLStorageModeShared;
      id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
      if (tex)
      {
        float zero = 0.0f;
        MTLRegion region = MTLRegionMake3D(0, 0, 0, 1, 1, 1);
        [tex replaceRegion:region mipmapLevel:0 slice:0 withBytes:&zero
               bytesPerRow:sizeof(float) bytesPerImage:sizeof(float)];
      }
      AssignMetalObject(this->DummyMaskTexture, tex);
      [desc release];
    }

    if (!this->DummyMinMaxTexture)
    {
      MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
      desc.textureType = MTLTextureType3D;
      desc.pixelFormat = MTLPixelFormatR8Unorm;
      desc.width = 1;
      desc.height = 1;
      desc.depth = 1;
      desc.mipmapLevelCount = 1;
      desc.usage = MTLTextureUsageShaderRead;
      desc.storageMode = MTLStorageModeShared;
      id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
      if (tex)
      {
        uint8_t zero = 0;
        MTLRegion region = MTLRegionMake3D(0, 0, 0, 1, 1, 1);
        [tex replaceRegion:region mipmapLevel:0 slice:0 withBytes:&zero
               bytesPerRow:sizeof(uint8_t) bytesPerImage:sizeof(uint8_t)];
      }
      AssignMetalObject(this->DummyMinMaxTexture, tex);
      [desc release];
    }

    // Create and cache a depth stencil state.
    if (!this->DepthStencilState)
    {
      MTLDepthStencilDescriptor* dsDesc = [[MTLDepthStencilDescriptor alloc] init];
      dsDesc.depthCompareFunction = MTLCompareFunctionLessEqual;
      dsDesc.depthWriteEnabled = NO;
      id<MTLDepthStencilState> ds = [device newDepthStencilStateWithDescriptor:dsDesc];
      [dsDesc release];
      AssignMetalObject(this->DepthStencilState, ds);
    }

    // Create volume rendering pipelines via the caching helper.
    // Base pipelines (featureMask=0) are created here for backward compat;
    // specialized pipelines with non-zero feature masks are created on demand
    // in GPURender() via GetOrCreateVolumePipeline.
    void* pso = this->GetOrCreateVolumePipeline(mtlDeviceVoid,
      static_cast<uint32_t>(VolumePipelineType::DirectScreen),
      MTLPixelFormatBGRA8Unorm, MTLPixelFormatDepth32Float,
      static_cast<uint32_t>(sampleCount), 0);
    if (!pso)
    {
      return false;
    }
    AssignMetalObject(this->PipelineState, (__bridge id)pso);
    this->CurrentSampleCount = sampleCount;

    void* accumPso = this->GetOrCreateVolumePipeline(mtlDeviceVoid,
      static_cast<uint32_t>(VolumePipelineType::OffscreenAccumulation),
      MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1, 0);
    if (!accumPso)
    {
      return false;
    }
    AssignMetalObject(this->AccumulationPipelineState, (__bridge id)accumPso);

    void* layerPso = this->GetOrCreateVolumePipeline(mtlDeviceVoid,
      static_cast<uint32_t>(VolumePipelineType::OffscreenLayer),
      MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1, 0);
    if (!layerPso)
    {
      return false;
    }
    AssignMetalObject(this->LayerPipelineState, (__bridge id)layerPso);

    // Pre-create fullscreen camera-inside pipelines (cached in PipelineCache).
    // FullscreenDirect: BGRA + depth + blending, matching DirectScreen.
    void* fsDirectPso = this->GetOrCreateVolumePipeline(mtlDeviceVoid,
      static_cast<uint32_t>(VolumePipelineType::FullscreenDirect),
      MTLPixelFormatBGRA8Unorm, MTLPixelFormatDepth32Float,
      static_cast<uint32_t>(sampleCount), 0);
    if (!fsDirectPso)
    {
      return false;
    }
    // FullscreenOffscreen: RGBA16Float, no depth, no blending, matching OffscreenLayer.
    void* fsOffscreenPso = this->GetOrCreateVolumePipeline(mtlDeviceVoid,
      static_cast<uint32_t>(VolumePipelineType::FullscreenOffscreen),
      MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1, 0);
    if (!fsOffscreenPso)
    {
      return false;
    }

    // Composite pipeline: fullscreen resolve of the 8 layers.
    if (!this->CompositePipelineState)
    {
      MTLRenderPipelineDescriptor* cd = [[MTLRenderPipelineDescriptor alloc] init];
      id<MTLFunction> cdVert = [library newFunctionWithName:@"vertex_fullscreen_main"];
      id<MTLFunction> cdFrag = [library newFunctionWithName:@"fragment_layer_composite_main"];
      cd.vertexFunction = cdVert;
      cd.fragmentFunction = cdFrag;
      cd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
      cd.colorAttachments[0].blendingEnabled = NO;
      cd.depthAttachmentPixelFormat = MTLPixelFormatInvalid;
      cd.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
      cd.rasterSampleCount = 1;

      NSError* compError = nil;
      id<MTLRenderPipelineState> p = [device newRenderPipelineStateWithDescriptor:cd error:&compError];
      [cd release];
      [cdVert release];
      [cdFrag release];
      if (!p)
      {
        vtkErrorMacro(<< "Composite pipeline: " << [[compError localizedDescription] UTF8String]);
        return false;
      }
      AssignMetalObject(this->CompositePipelineState, p);
      {
        VolumePipelineKey k = { static_cast<uint32_t>(VolumePipelineType::LayerComposite),
          MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1, 0 };
        [(__bridge id)p retain];
        this->PipelineCache[k] = (__bridge void*)p;
      }
    }
  }

  return true;
}

//------------------------------------------------------------------------------
void* vtkMetalGPUVolumeRayCastMapper::GetOrCreateVolumePipeline(
  void* mtlDeviceVoid, uint32_t type, uint32_t colorFormat,
  uint32_t depthFormat, uint32_t sampleCount, uint32_t featureMask)
{
  VolumePipelineKey key = { type, colorFormat, depthFormat, sampleCount, featureMask };
  auto it = this->PipelineCache.find(key);
  if (it != this->PipelineCache.end())
  {
    return it->second;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary;
  if (!library)
  {
    return nullptr;
  }

  NSError* error = nil;

  // Determine the fragment function name and whether to use vertex_volume_main
  // or vertex_fullscreen_main based on pipeline type.
  NSString* fragName = @"fragment_volume_main";
  bool useVolumeVertex = true;

  switch (static_cast<VolumePipelineType>(type))
  {
    case VolumePipelineType::OffscreenAccumulation:
      fragName = @"fragment_volume_accum_main";
      useVolumeVertex = true;
      break;
    case VolumePipelineType::DirectScreen:
    case VolumePipelineType::OffscreenLayer:
      fragName = @"fragment_volume_main";
      useVolumeVertex = true;
      break;
    case VolumePipelineType::FullscreenDirect:
    case VolumePipelineType::FullscreenOffscreen:
      fragName = @"fragment_volume_fullscreen_main";
      useVolumeVertex = false;
      break;
    case VolumePipelineType::LayerComposite:
      fragName = @"fragment_layer_composite_main";
      useVolumeVertex = false;
      break;
    case VolumePipelineType::ImageSampleBlit:
      fragName = @"fragment_image_sample_blit";
      useVolumeVertex = false;
      break;
  }

  // Create function constants for shader specialization.
  // Only volume fragment shaders (volume_main/volume_accum_main) use
  // function constants; composite and blit passes do not.
  id<MTLFunction> fragFunc = nil;
  VolumePipelineType pt = static_cast<VolumePipelineType>(type);
  BOOL hasFeatureConstants = (pt == VolumePipelineType::DirectScreen ||
    pt == VolumePipelineType::OffscreenLayer ||
    pt == VolumePipelineType::OffscreenAccumulation ||
    pt == VolumePipelineType::FullscreenDirect ||
    pt == VolumePipelineType::FullscreenOffscreen);

  if (hasFeatureConstants)
  {
    MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];

    BOOL shading = (featureMask & VolumeFeature_Shading) ? YES : NO;
    BOOL gradOp  = (featureMask & VolumeFeature_GradientOpacity) ? YES : NO;
    BOOL mask    = (featureMask & VolumeFeature_Mask) ? YES : NO;
    BOOL minmax  = (featureMask & VolumeFeature_MinMax) ? YES : NO;
    BOOL normalTex = (featureMask & VolumeFeature_NormalTexture) ? YES : NO;

    [constants setConstantValue:&shading type:MTLDataTypeBool withName:@"fc_shading"];
    [constants setConstantValue:&gradOp  type:MTLDataTypeBool withName:@"fc_gradientOpacity"];
    [constants setConstantValue:&mask    type:MTLDataTypeBool withName:@"fc_mask"];
    [constants setConstantValue:&minmax  type:MTLDataTypeBool withName:@"fc_minmax"];
    [constants setConstantValue:&normalTex type:MTLDataTypeBool withName:@"fc_normalTexture"];

    fragFunc = [library newFunctionWithName:fragName
                             constantValues:constants
                                      error:&error];
    [constants release];
  }
  else
  {
    // Non-volume pipelines (composite, blit) have no function constants.
    fragFunc = [library newFunctionWithName:fragName];
  }

  if (!fragFunc)
  {
    vtkErrorMacro(<< "Failed to find fragment function " << [fragName UTF8String]);
    return nullptr;
  }

  id<MTLFunction> vertexFunc = nil;
  if (useVolumeVertex)
  {
    vertexFunc = [library newFunctionWithName:@"vertex_volume_main"];
  }
  else
  {
    vertexFunc = [library newFunctionWithName:@"vertex_fullscreen_main"];
  }

  if (!vertexFunc)
  {
    vtkErrorMacro("Failed to find vertex function for pipeline");
    [fragFunc release];
    return nullptr;
  }

  MTLRenderPipelineDescriptor* pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
  pipelineDesc.vertexFunction = vertexFunc;
  pipelineDesc.fragmentFunction = fragFunc;

  if (useVolumeVertex)
  {
    MTLVertexDescriptor* vertexDesc = [[MTLVertexDescriptor alloc] init];
    vertexDesc.attributes[0].format = MTLVertexFormatFloat3;
    vertexDesc.attributes[0].offset = 0;
    vertexDesc.attributes[0].bufferIndex = 0;
    vertexDesc.layouts[0].stride = sizeof(float) * 3;
    vertexDesc.layouts[0].stepRate = 1;
    vertexDesc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    pipelineDesc.vertexDescriptor = vertexDesc;
    [vertexDesc release];
  }

  pipelineDesc.colorAttachments[0].pixelFormat = (MTLPixelFormat)colorFormat;

  // DirectScreen and FullscreenDirect use blending; offscreen pipelines do not.
  if (pt == VolumePipelineType::DirectScreen || pt == VolumePipelineType::FullscreenDirect)
  {
    pipelineDesc.colorAttachments[0].blendingEnabled = YES;
    pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
    pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
  }
  else
  {
    pipelineDesc.colorAttachments[0].blendingEnabled = NO;
  }

  if (depthFormat != MTLPixelFormatInvalid)
  {
    pipelineDesc.depthAttachmentPixelFormat = (MTLPixelFormat)depthFormat;
  }

  pipelineDesc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
  pipelineDesc.rasterSampleCount = sampleCount;

  id<MTLRenderPipelineState> pso =
    [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
  [pipelineDesc release];
  [vertexFunc release];
  [fragFunc release];

  if (!pso)
  {
    vtkErrorMacro(<< "Pipeline creation failed for type " << type << ": "
                  << [[error localizedDescription] UTF8String]);
    return nullptr;
  }

  // Cache and return
  this->PipelineCache[key] = (__bridge void*)pso;
  return (__bridge void*)pso;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::BindEncoderResources(
  void* encoderVoid, void* uniformBufVoid, void* pipelineStateVoid, bool hasDepth)
{
  id<MTLRenderCommandEncoder> encoder =
    (__bridge id<MTLRenderCommandEncoder>)encoderVoid;
  id<MTLBuffer> uniformBuf = (__bridge id<MTLBuffer>)uniformBufVoid;

  // Set pipeline and render state (use provided pipeline, or fall back to default)
  id<MTLRenderPipelineState> pipeline;
  if (pipelineStateVoid)
  {
    pipeline = (__bridge id<MTLRenderPipelineState>)pipelineStateVoid;
  }
  else
  {
    pipeline = (__bridge id<MTLRenderPipelineState>)this->PipelineState;
  }
  [encoder setRenderPipelineState:pipeline];
  [encoder setCullMode:MTLCullModeBack];

  // Only bind depth state if the pipeline uses depth testing.
  if (this->DepthStencilState && hasDepth)
  {
    id<MTLDepthStencilState> ds =
      (__bridge id<MTLDepthStencilState>)this->DepthStencilState;
    [encoder setDepthStencilState:ds];
  }

  // Bind buffers
  id<MTLBuffer> vertexBuf = (__bridge id<MTLBuffer>)this->VertexBuffer;
  [encoder setVertexBuffer:vertexBuf offset:0 atIndex:0];
  [encoder setVertexBuffer:uniformBuf offset:0 atIndex:1];
  [encoder setFragmentBuffer:uniformBuf offset:0 atIndex:1];

  id<MTLTexture> volTex = this->VolumeTexture
    ? (__bridge id<MTLTexture>)this->VolumeTexture
    : (__bridge id<MTLTexture>)this->DummyVolumeTexture;
  id<MTLTexture> tfTex = (__bridge id<MTLTexture>)this->ColorOpacityTexture;
  [encoder setFragmentTexture:volTex atIndex:0];
  [encoder setFragmentTexture:tfTex atIndex:1];

  // Bind scene depth texture for early ray termination (fragment index 2).
  // Use the dummy depth texture (value 1.0) when no real depth texture is available.
  id<MTLTexture> depthTex = this->DepthTextureOcclusion
    ? (__bridge id<MTLTexture>)this->DepthTextureOcclusion
    : (__bridge id<MTLTexture>)this->DummyDepthTexture;
  [encoder setFragmentTexture:depthTex atIndex:2];

  // Bind gradient opacity texture for gradient-based shading (fragment index 3).
  // The shader uses constexpr samplers, so no sampler bindings are needed.
  if (this->GradientOpacityTexture)
  {
    id<MTLTexture> goTex = (__bridge id<MTLTexture>)this->GradientOpacityTexture;
    [encoder setFragmentTexture:goTex atIndex:3];
  }
  else
  {
    [encoder setFragmentTexture:tfTex atIndex:3];
  }

  // Bind mask / label map textures (fragment index 4).
  id<MTLTexture> maskFallbackTex =
    (__bridge id<MTLTexture>)this->DummyMaskTexture;
  if (this->MaskTexture)
  {
    id<MTLTexture> maskTex = (__bridge id<MTLTexture>)this->MaskTexture;
    [encoder setFragmentTexture:maskTex atIndex:4];
  }
  else
  {
    [encoder setFragmentTexture:maskFallbackTex atIndex:4];
  }

  // Bind label map transfer texture (fragment index 5)
  if (this->LabelMapTransferTexture)
  {
    id<MTLTexture> lmTex = (__bridge id<MTLTexture>)this->LabelMapTransferTexture;
    [encoder setFragmentTexture:lmTex atIndex:5];
  }
  else
  {
    [encoder setFragmentTexture:tfTex atIndex:5];
  }

  // Bind min-max acceleration texture (fragment index 6).
  id<MTLTexture> minMaxFallbackTex =
    (__bridge id<MTLTexture>)this->DummyMinMaxTexture;
  if (this->MinMaxTexture)
  {
    id<MTLTexture> mmTex = (__bridge id<MTLTexture>)this->MinMaxTexture;
    [encoder setFragmentTexture:mmTex atIndex:6];
  }
  else
  {
    [encoder setFragmentTexture:minMaxFallbackTex atIndex:6];
  }

  // Bind precomputed gradient normal texture (fragment index 7).
  if (this->GradientNormalTexture)
  {
    [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->GradientNormalTexture atIndex:7];
  }
  else
  {
    [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->DummyVolumeTexture atIndex:7];
  }
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::DrawBlocks(
  void* encoderVoid, void* uniformBufVoid, vtkRenderer* ren, vtkVolume* vol,
  void* uniformsVoid, vtkMatrix4x4* invModelMatrix)
{
  id<MTLRenderCommandEncoder> encoder =
    (__bridge id<MTLRenderCommandEncoder>)encoderVoid;
  id<MTLBuffer> uniformBuf = (__bridge id<MTLBuffer>)uniformBufVoid;
  VolumeMapperUniforms* uniforms = static_cast<VolumeMapperUniforms*>(uniformsVoid);
  id<MTLBuffer> indexBuf = (__bridge id<MTLBuffer>)this->IndexBuffer;

  vtkImageData* input = vtkImageData::SafeDownCast(this->GetInput());
  int fullExt[6];
  input->GetExtent(fullExt);
  double origin[3], spacing[3];
  input->GetOrigin(origin);
  input->GetSpacing(spacing);

  if (!this->Blocks.empty())
  {
    this->SortBlocksBackToFront(ren, vol);

    // NOTE: the old single-draw INSTANCED path was removed. Partitioned volumes
    // with <= MAX_LAYER_BRICKS bricks are now composited order-independently in
    // GPURender (per-brick layer textures + per-pixel-sorted composite), which
    // makes draw order irrelevant. This function is therefore only reached for
    // the > MAX_LAYER_BRICKS fallback (order-dependent; see "Known limitations"
    // in GPURender) and never instances.

    // --- FALLBACK PATH (> MAX_LAYER_BRICKS bricks): one draw per brick,
    //     composited front-to-back via framebuffer fetch. Order-dependent. ---
    for (size_t bi = 0; bi < this->SortedBlockOrder.size(); ++bi)
    {
      int si = this->SortedBlockOrder[bi];
      auto& block = this->Blocks[si];

      PerBlockData pbd;
      pbd.VolumeBoundsMin[0] = static_cast<float>(block.BoundsMin[0]);
      pbd.VolumeBoundsMin[1] = static_cast<float>(block.BoundsMin[1]);
      pbd.VolumeBoundsMin[2] = static_cast<float>(block.BoundsMin[2]);
      pbd.VolumeBoundsMin[3] = 1.0f;

      pbd.VolumeBoundsMax[0] = static_cast<float>(block.BoundsMax[0]);
      pbd.VolumeBoundsMax[1] = static_cast<float>(block.BoundsMax[1]);
      pbd.VolumeBoundsMax[2] = static_cast<float>(block.BoundsMax[2]);
      pbd.VolumeBoundsMax[3] = 1.0f;

      int texExt[6] = {
        std::max(fullExt[0], block.Extents[0] - 1),
        std::min(fullExt[1], block.Extents[1] + 1),
        std::max(fullExt[2], block.Extents[2] - 1),
        std::min(fullExt[3], block.Extents[3] + 1),
        std::max(fullExt[4], block.Extents[4] - 1),
        std::min(fullExt[5], block.Extents[5] + 1)
      };

      pbd.TextureBoundsMin[0] = static_cast<float>(origin[0] + (texExt[0] - 0.5) * spacing[0]);
      pbd.TextureBoundsMin[1] = static_cast<float>(origin[1] + (texExt[2] - 0.5) * spacing[1]);
      pbd.TextureBoundsMin[2] = static_cast<float>(origin[2] + (texExt[4] - 0.5) * spacing[2]);
      pbd.TextureBoundsMin[3] = 1.0f;

      pbd.TextureBoundsMax[0] = static_cast<float>(origin[0] + (texExt[1] + 0.5) * spacing[0]);
      pbd.TextureBoundsMax[1] = static_cast<float>(origin[1] + (texExt[3] + 0.5) * spacing[1]);
      pbd.TextureBoundsMax[2] = static_cast<float>(origin[2] + (texExt[5] + 0.5) * spacing[2]);
      pbd.TextureBoundsMax[3] = 1.0f;

      for (int k = 0; k < 3; ++k)
      {
        pbd.GradientStep[k] = (block.Dims[k] > 0) ? 1.0f / block.Dims[k] : 1.0f;
      }
      pbd.GradientStep[3] = 0.0f;

      if (block.MinMaxTexture)
      {
        pbd.MinMaxInfo[0] = 1.0f;
        pbd.MinMaxInfo[1] = static_cast<float>(block.MinMaxDims[0]);
        pbd.MinMaxInfo[2] = static_cast<float>(block.MinMaxDims[1]);
        pbd.MinMaxInfo[3] = static_cast<float>(block.MinMaxDims[2]);

        id<MTLTexture> blockMmTex = (__bridge id<MTLTexture>)block.MinMaxTexture;
        [encoder setFragmentTexture:blockMmTex atIndex:6];
      }
      else
      {
        pbd.MinMaxInfo[0] = 0.0f;
        pbd.MinMaxInfo[1] = 0.0f;
        pbd.MinMaxInfo[2] = 0.0f;
        pbd.MinMaxInfo[3] = 0.0f;
      }

      [encoder setVertexBytes:&pbd length:sizeof(PerBlockData) atIndex:2];
      [encoder setFragmentBytes:&pbd length:sizeof(PerBlockData) atIndex:2];

      // Bind this block's 3D texture (scalar data)
      id<MTLTexture> blockTex = (__bridge id<MTLTexture>)block.Texture;
      [encoder setFragmentTexture:blockTex atIndex:0];
      // Bind per-block normal texture at index 7 if available
      if (block.NormalTexture)
      {
        [encoder setFragmentTexture:(__bridge id<MTLTexture>)block.NormalTexture atIndex:7];
      }
      else
      {
        [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->DummyVolumeTexture atIndex:7];
      }

      // Draw
      [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                          indexCount:this->IndexCount
                           indexType:MTLIndexTypeUInt32
                         indexBuffer:indexBuf
                   indexBufferOffset:0];
    }
  }
  else
  {
    // Single-block path (no partitioning)
    PerBlockData pbd = {};
    pbd.VolumeBoundsMin[0] = uniforms->VolumeBoundsMin[0];
    pbd.VolumeBoundsMin[1] = uniforms->VolumeBoundsMin[1];
    pbd.VolumeBoundsMin[2] = uniforms->VolumeBoundsMin[2];
    pbd.VolumeBoundsMin[3] = 1.0f;

    pbd.VolumeBoundsMax[0] = uniforms->VolumeBoundsMax[0];
    pbd.VolumeBoundsMax[1] = uniforms->VolumeBoundsMax[1];
    pbd.VolumeBoundsMax[2] = uniforms->VolumeBoundsMax[2];
    pbd.VolumeBoundsMax[3] = 1.0f;

    pbd.TextureBoundsMin[0] = uniforms->VolumeBoundsMin[0];
    pbd.TextureBoundsMin[1] = uniforms->VolumeBoundsMin[1];
    pbd.TextureBoundsMin[2] = uniforms->VolumeBoundsMin[2];
    pbd.TextureBoundsMin[3] = 1.0f;

    pbd.TextureBoundsMax[0] = uniforms->VolumeBoundsMax[0];
    pbd.TextureBoundsMax[1] = uniforms->VolumeBoundsMax[1];
    pbd.TextureBoundsMax[2] = uniforms->VolumeBoundsMax[2];
    pbd.TextureBoundsMax[3] = 1.0f;

    pbd.GradientStep[0] = uniforms->GradientStep[0];
    pbd.GradientStep[1] = uniforms->GradientStep[1];
    pbd.GradientStep[2] = uniforms->GradientStep[2];
    pbd.GradientStep[3] = 0.0f;

    pbd.MinMaxInfo[0] = uniforms->UseMinMaxAccel;
    pbd.MinMaxInfo[1] = uniforms->MinMaxDimX;
    pbd.MinMaxInfo[2] = uniforms->MinMaxDimY;
    pbd.MinMaxInfo[3] = uniforms->MinMaxDimZ;

    [encoder setVertexBytes:&pbd length:sizeof(PerBlockData) atIndex:2];
    [encoder setFragmentBytes:&pbd length:sizeof(PerBlockData) atIndex:2];

    [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                        indexCount:this->IndexCount
                         indexType:MTLIndexTypeUInt32
                       indexBuffer:indexBuf
                 indexBufferOffset:0];
  }
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::DrawBlocksFullscreen(
  void* encoderVoid, void* uniformBufVoid, vtkRenderer* ren, vtkVolume* vol,
  void* uniformsVoid, vtkMatrix4x4* invModelMatrix, bool useDirectPipeline)
{
  id<MTLRenderCommandEncoder> encoder =
    (__bridge id<MTLRenderCommandEncoder>)encoderVoid;
  id<MTLBuffer> uniformBuf = (__bridge id<MTLBuffer>)uniformBufVoid;
  VolumeMapperUniforms* uniforms = static_cast<VolumeMapperUniforms*>(uniformsVoid);

  vtkImageData* input = vtkImageData::SafeDownCast(this->GetInput());
  if (!input) return;
  int fullExt[6];
  input->GetExtent(fullExt);
  double origin[3], spacing[3];
  input->GetOrigin(origin);
  input->GetSpacing(spacing);

  // Determine pipeline type based on whether this is direct-screen or offscreen.
  uint32_t pipelineType = useDirectPipeline
    ? static_cast<uint32_t>(VolumePipelineType::FullscreenDirect)
    : static_cast<uint32_t>(VolumePipelineType::FullscreenOffscreen);

  // Build feature mask matching the current rendering state.
  int featureMask = 0;
  if (uniforms->UseGradientShading > 0.5f)
    featureMask |= VolumeFeature_Shading;
  if (uniforms->UseGradientOpacity > 0.5f)
    featureMask |= VolumeFeature_GradientOpacity;
  if (uniforms->UseMask > 0.5f)
    featureMask |= VolumeFeature_Mask;
  if (uniforms->UseMinMaxAccel > 0.5f || !this->Blocks.empty())
    featureMask |= VolumeFeature_MinMax;

  id<MTLDevice> device = (__bridge id<MTLDevice>)
    (static_cast<vtkMetalRenderWindow*>(ren->GetRenderWindow()))->GetMetalDevice();

  // Determine color/depth format for pipeline key.
  uint32_t colorFormat = useDirectPipeline ? MTLPixelFormatBGRA8Unorm : MTLPixelFormatRGBA16Float;
  uint32_t depthFormat = useDirectPipeline ? MTLPixelFormatDepth32Float : MTLPixelFormatInvalid;
  auto* metalRenderWindow = vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());
  uint32_t sampleCount = useDirectPipeline
    ? static_cast<uint32_t>(metalRenderWindow ? metalRenderWindow->GetEffectiveSampleCount() : 1)
    : 1;

  // Handle partitioned volumes: sort blocks, render each brick.
  if (!this->Blocks.empty())
  {
    this->SortBlocksBackToFront(ren, vol);

    for (size_t bi = 0; bi < this->SortedBlockOrder.size(); ++bi)
    {
      int si = this->SortedBlockOrder[bi];
      auto& block = this->Blocks[si];

      VolumePipelineType pt = useDirectPipeline ? VolumePipelineType::FullscreenDirect
                                                : VolumePipelineType::FullscreenOffscreen;
      uint32_t fsType = static_cast<uint32_t>(pt);

      void* pso = this->GetOrCreateVolumePipeline(
        (__bridge void*)device, fsType, colorFormat, depthFormat, sampleCount, featureMask);
      if (!pso) continue;
      [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)pso];

      // Build PerBlockData for this brick.
      PerBlockData pbd;
      pbd.VolumeBoundsMin[0] = static_cast<float>(block.BoundsMin[0]);
      pbd.VolumeBoundsMin[1] = static_cast<float>(block.BoundsMin[1]);
      pbd.VolumeBoundsMin[2] = static_cast<float>(block.BoundsMin[2]);
      pbd.VolumeBoundsMin[3] = 1.0f;
      pbd.VolumeBoundsMax[0] = static_cast<float>(block.BoundsMax[0]);
      pbd.VolumeBoundsMax[1] = static_cast<float>(block.BoundsMax[1]);
      pbd.VolumeBoundsMax[2] = static_cast<float>(block.BoundsMax[2]);
      pbd.VolumeBoundsMax[3] = 1.0f;

      int texExt[6] = {
        std::max(fullExt[0], block.Extents[0] - 1),
        std::min(fullExt[1], block.Extents[1] + 1),
        std::max(fullExt[2], block.Extents[2] - 1),
        std::min(fullExt[3], block.Extents[3] + 1),
        std::max(fullExt[4], block.Extents[4] - 1),
        std::min(fullExt[5], block.Extents[5] + 1)
      };
      pbd.TextureBoundsMin[0] = static_cast<float>(origin[0] + (texExt[0] - 0.5) * spacing[0]);
      pbd.TextureBoundsMin[1] = static_cast<float>(origin[1] + (texExt[2] - 0.5) * spacing[1]);
      pbd.TextureBoundsMin[2] = static_cast<float>(origin[2] + (texExt[4] - 0.5) * spacing[2]);
      pbd.TextureBoundsMin[3] = 1.0f;
      pbd.TextureBoundsMax[0] = static_cast<float>(origin[0] + (texExt[1] + 0.5) * spacing[0]);
      pbd.TextureBoundsMax[1] = static_cast<float>(origin[1] + (texExt[3] + 0.5) * spacing[1]);
      pbd.TextureBoundsMax[2] = static_cast<float>(origin[2] + (texExt[5] + 0.5) * spacing[2]);
      pbd.TextureBoundsMax[3] = 1.0f;

      for (int k = 0; k < 3; ++k)
        pbd.GradientStep[k] = (block.Dims[k] > 0) ? 1.0f / block.Dims[k] : 1.0f;
      pbd.GradientStep[3] = 0.0f;

      if (block.MinMaxTexture)
      {
        pbd.MinMaxInfo[0] = 1.0f;
        pbd.MinMaxInfo[1] = static_cast<float>(block.MinMaxDims[0]);
        pbd.MinMaxInfo[2] = static_cast<float>(block.MinMaxDims[1]);
        pbd.MinMaxInfo[3] = static_cast<float>(block.MinMaxDims[2]);
      }
      else
      {
        pbd.MinMaxInfo[0] = pbd.MinMaxInfo[1] = pbd.MinMaxInfo[2] = pbd.MinMaxInfo[3] = 0.0f;
      }

      // Bind resources manually (avoid BindEncoderResources which sets a potentially
      // nil VertexBuffer — the fullscreen path skips vertex buffer creation).
      [encoder setCullMode:MTLCullModeBack];
      if (this->DepthStencilState && useDirectPipeline)
      {
        [encoder setDepthStencilState:(__bridge id<MTLDepthStencilState>)this->DepthStencilState];
      }
      [encoder setFragmentBuffer:uniformBuf offset:0 atIndex:1];
      [encoder setFragmentTexture:(__bridge id<MTLTexture>)block.Texture atIndex:0];
      [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->ColorOpacityTexture atIndex:1];
      id<MTLTexture> depthTex = this->DepthTextureOcclusion
        ? (__bridge id<MTLTexture>)this->DepthTextureOcclusion
        : (__bridge id<MTLTexture>)this->DummyDepthTexture;
      [encoder setFragmentTexture:depthTex atIndex:2];
      if (this->GradientOpacityTexture)
        [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->GradientOpacityTexture atIndex:3];
      else
        [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->ColorOpacityTexture atIndex:3];
      if (this->MaskTexture)
        [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->MaskTexture atIndex:4];
      else
        [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->DummyMaskTexture atIndex:4];
      if (this->LabelMapTransferTexture)
        [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->LabelMapTransferTexture atIndex:5];
      else
        [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->ColorOpacityTexture atIndex:5];
      if (block.MinMaxTexture)
        [encoder setFragmentTexture:(__bridge id<MTLTexture>)block.MinMaxTexture atIndex:6];
      else
        [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->DummyMinMaxTexture atIndex:6];
      if (block.NormalTexture)
        [encoder setFragmentTexture:(__bridge id<MTLTexture>)block.NormalTexture atIndex:7];
      else
        [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->DummyVolumeTexture atIndex:7];

      // Bind PerBlockData as bytes.
      [encoder setVertexBytes:&pbd length:sizeof(PerBlockData) atIndex:2];
      [encoder setFragmentBytes:&pbd length:sizeof(PerBlockData) atIndex:2];

      // Draw fullscreen triangle (no vertex/index buffers needed).
      [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
  }
  else
  {
    // Single-block volume: render fullscreen with global volume bounds.
    void* pso = this->GetOrCreateVolumePipeline(
      (__bridge void*)device, pipelineType, colorFormat, depthFormat, sampleCount, featureMask);
    if (!pso) return;
    [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)pso];

    PerBlockData pbd = {};
    pbd.VolumeBoundsMin[0] = uniforms->VolumeBoundsMin[0];
    pbd.VolumeBoundsMin[1] = uniforms->VolumeBoundsMin[1];
    pbd.VolumeBoundsMin[2] = uniforms->VolumeBoundsMin[2];
    pbd.VolumeBoundsMin[3] = 1.0f;
    pbd.VolumeBoundsMax[0] = uniforms->VolumeBoundsMax[0];
    pbd.VolumeBoundsMax[1] = uniforms->VolumeBoundsMax[1];
    pbd.VolumeBoundsMax[2] = uniforms->VolumeBoundsMax[2];
    pbd.VolumeBoundsMax[3] = 1.0f;
    pbd.TextureBoundsMin[0] = uniforms->VolumeBoundsMin[0];
    pbd.TextureBoundsMin[1] = uniforms->VolumeBoundsMin[1];
    pbd.TextureBoundsMin[2] = uniforms->VolumeBoundsMin[2];
    pbd.TextureBoundsMin[3] = 1.0f;
    pbd.TextureBoundsMax[0] = uniforms->VolumeBoundsMax[0];
    pbd.TextureBoundsMax[1] = uniforms->VolumeBoundsMax[1];
    pbd.TextureBoundsMax[2] = uniforms->VolumeBoundsMax[2];
    pbd.TextureBoundsMax[3] = 1.0f;
    pbd.GradientStep[0] = uniforms->GradientStep[0];
    pbd.GradientStep[1] = uniforms->GradientStep[1];
    pbd.GradientStep[2] = uniforms->GradientStep[2];
    pbd.GradientStep[3] = 0.0f;
    pbd.MinMaxInfo[0] = uniforms->UseMinMaxAccel;
    pbd.MinMaxInfo[1] = uniforms->MinMaxDimX;
    pbd.MinMaxInfo[2] = uniforms->MinMaxDimY;
    pbd.MinMaxInfo[3] = uniforms->MinMaxDimZ;

    // Bind resources manually (avoid BindEncoderResources which sets VertexBuffer).
    [encoder setCullMode:MTLCullModeBack];
    if (this->DepthStencilState && useDirectPipeline)
    {
      [encoder setDepthStencilState:(__bridge id<MTLDepthStencilState>)this->DepthStencilState];
    }
    [encoder setVertexBytes:&pbd length:sizeof(PerBlockData) atIndex:2];
    [encoder setFragmentBytes:&pbd length:sizeof(PerBlockData) atIndex:2];
    [encoder setFragmentBuffer:uniformBuf offset:0 atIndex:1];
    [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->VolumeTexture atIndex:0];
    [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->ColorOpacityTexture atIndex:1];
    id<MTLTexture> depthTex = this->DepthTextureOcclusion
      ? (__bridge id<MTLTexture>)this->DepthTextureOcclusion
      : (__bridge id<MTLTexture>)this->DummyDepthTexture;
    [encoder setFragmentTexture:depthTex atIndex:2];
    if (this->GradientOpacityTexture)
      [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->GradientOpacityTexture atIndex:3];
    else
      [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->ColorOpacityTexture atIndex:3];
    if (this->MaskTexture)
      [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->MaskTexture atIndex:4];
    else
      [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->DummyMaskTexture atIndex:4];
    if (this->LabelMapTransferTexture)
      [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->LabelMapTransferTexture atIndex:5];
    else
      [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->ColorOpacityTexture atIndex:5];
    if (this->MinMaxTexture)
      [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->MinMaxTexture atIndex:6];
    else
      [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->DummyMinMaxTexture atIndex:6];
    if (this->GradientNormalTexture)
      [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->GradientNormalTexture atIndex:7];
    else
      [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->DummyVolumeTexture atIndex:7];

    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  }
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::GPURender(vtkRenderer* ren, vtkVolume* vol)
{
  @autoreleasepool
  {
  auto* metalRenderer = vtkMetalRenderer::SafeDownCast(ren);
  auto* metalRenderWindow = vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());
  if (!metalRenderer || !metalRenderWindow)
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)metalRenderWindow->GetMetalDevice();
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)metalRenderWindow->GetMetalQueue();
  id<MTLCommandBuffer> commandBuffer =
    (__bridge id<MTLCommandBuffer>)metalRenderWindow->GetCurrentCommandBuffer();

  if (!device || !commandBuffer)
  {
    return;
  }

  void* mtlDevice = metalRenderWindow->GetMetalDevice();
  void* mtlQueue = metalRenderWindow->GetMetalQueue();

  vtkImageData* input = vtkImageData::SafeDownCast(this->GetInput());
  if (!input)
  {
    return;
  }

  // Cache scalar range once (used by both TF texture and uniforms)
  vtkDataArray* scalars = input->GetPointData()->GetScalars();
  if (scalars)
  {
    scalars->GetRange(this->ScalarRange, 0);
  }
  else
  {
    this->ScalarRange[0] = 0.0;
    this->ScalarRange[1] = 1.0;
  }

  // Phase 5: GPU-accelerated min-max generation.
  // For single-block volumes with UseGPUMinMax, we must upload the volume
  // texture first, then dispatch compute kernels. For partitioned volumes
  // (or when GPU min-max is disabled), the CPU path runs before volume
  // upload so that UpdateBlockTextures can reuse the per-macrocell data.
  bool usePartitions = (this->Partitions[0] > 1 || this->Partitions[1] > 1 || this->Partitions[2] > 1);

  if (this->UseGPUMinMax && !usePartitions)
  {
    // GPU min-max path: upload volume texture first, then dispatch compute.
    if (!this->UpdateVolumeTexture(mtlDevice, mtlQueue, vol))
    {
      return;
    }

    if (!this->ComputeMinMaxGPU(mtlDevice, mtlQueue, vol, input, scalars))
    {
      // GPU path failed — release any private texture so the CPU
      // fallback's replaceRegion (which needs StorageModeShared) works.
      ReleaseMetalObject(this->MinMaxTexture);
      this->UpdateMinMaxTexture(mtlDevice, vol, input, scalars, false);
    }
  }
  else
  {
    // CPU min-max path: compute from raw scalar data before volume upload.
    // Partitioned volumes still need CPU macrocell data for per-block ranges.
    this->UpdateMinMaxTexture(mtlDevice, vol, input, scalars, usePartitions);

    if (!this->UpdateVolumeTexture(mtlDevice, mtlQueue, vol))
    {
      return;
    }
  }

  // Precompute gradient/normal texture for non-partitioned volumes (Phase 4).
  // Partitioned volumes generate per-block normals in UpdateBlockTextures.
  if (!usePartitions)
  {
    this->EnsureGradientNormalTexture(mtlDevice, mtlQueue, vol);
  }

  // Compute actual sample distance early so it can be used for opacity pre-integration
  // (matches the OpenGL mapper's vtkInternal::ComputeSampleDistance).
  this->ComputeReductionFactor(vol->GetAllocatedRenderTime());

  double actualSampleDistance;
  if (this->AutoAdjustSampleDistances)
  {
    vtkNew<vtkMatrix4x4> modelToWorld;
    vol->GetModelToWorldMatrix(modelToWorld);
    double cellSpacing[3];
    input->GetSpacing(cellSpacing);
    double minWorldSpacing = VTK_DOUBLE_MAX;
    for (int i = 0; i < 3; ++i)
    {
      double tmp = modelToWorld->GetElement(0, i);
      double tmp2 = tmp * tmp;
      tmp = modelToWorld->GetElement(1, i);
      tmp2 += tmp * tmp;
      tmp = modelToWorld->GetElement(2, i);
      tmp2 += tmp * tmp;
      double worldSpacing = fabs(cellSpacing[i] * sqrt(tmp2));
      minWorldSpacing = std::min(worldSpacing, minWorldSpacing);
    }
    actualSampleDistance = minWorldSpacing;
    if (this->ReductionFactor < 1.0 && this->ReductionFactor != 0.0)
    {
      actualSampleDistance /= this->ReductionFactor;
    }
  }
  else if (this->LockSampleDistanceToInputSpacing)
  {
    double cellSpacing[3];
    input->GetSpacing(cellSpacing);
    int extents[6];
    input->GetExtent(extents);
    double spacingDist = (cellSpacing[0] + cellSpacing[1] + cellSpacing[2]) / 6.0;
    double avgNumVoxels = pow(
      static_cast<double>((extents[1] - extents[0]) * (extents[3] - extents[2]) *
        (extents[5] - extents[4])),
      0.333);
    if (avgNumVoxels < 100)
    {
      spacingDist *= 0.01 + (1 - 0.01) * avgNumVoxels / 100;
    }
    float d = static_cast<float>(spacingDist);
    float sample = static_cast<float>(this->SampleDistance);
    actualSampleDistance =
      (sample / d < 0.999f || sample / d > 1.001f) ? d : this->SampleDistance;
  }
  else
  {
    actualSampleDistance = this->GetSampleDistance();
  }

  if (!this->UpdateTransferFunctionTexture(mtlDevice, mtlQueue, vol, actualSampleDistance))
  {
    return;
  }
  this->UpdateGradientOpacityTexture(mtlDevice, mtlQueue, vol);

  // Update mask / label map textures
  if (this->MaskInput && this->MaskType == vtkGPUVolumeRayCastMapper::LabelMapMaskType)
  {
    this->UpdateMaskTexture(mtlDevice, mtlQueue, vol);
    this->UpdateLabelMapTransferTexture(mtlDevice, mtlQueue, vol);
  }

  if (!this->SetupBuffers(mtlDevice, ren, vol, input))
  {
    return;
  }
  if (!this->SetupPipeline(mtlDevice, ren))
  {
    return;
  }

  // Update uniforms
  VolumeMapperUniforms uniforms = {};

  vtkNew<vtkMatrix4x4> modelMatrix;
  vol->GetModelToWorldMatrix(modelMatrix);

  vtkNew<vtkMatrix4x4> invModelMatrix;
  vtkMatrix4x4::Invert(modelMatrix, invModelMatrix);

  for (int r = 0; r < 4; ++r)
  {
    for (int c = 0; c < 4; ++c)
    {
      uniforms.VolumeToWorldMatrix[c * 4 + r] = modelMatrix->GetElement(r, c);
      uniforms.WorldToVolumeMatrix[c * 4 + r] = invModelMatrix->GetElement(r, c);
    }
  }

  double* modelBounds = this->ModelBounds;
  uniforms.VolumeBoundsMin[0] = static_cast<float>(modelBounds[0]);
  uniforms.VolumeBoundsMin[1] = static_cast<float>(modelBounds[2]);
  uniforms.VolumeBoundsMin[2] = static_cast<float>(modelBounds[4]);
  uniforms.VolumeBoundsMin[3] = 1.0f;

  uniforms.VolumeBoundsMax[0] = static_cast<float>(modelBounds[1]);
  uniforms.VolumeBoundsMax[1] = static_cast<float>(modelBounds[3]);
  uniforms.VolumeBoundsMax[2] = static_cast<float>(modelBounds[5]);
  uniforms.VolumeBoundsMax[3] = 1.0f;

  double boundsSize[3] = {
    modelBounds[1] - modelBounds[0],
    modelBounds[3] - modelBounds[2],
    modelBounds[5] - modelBounds[4]
  };
  for (int k = 0; k < 3; ++k)
  {
    if (boundsSize[k] < 1e-10)
      boundsSize[k] = 1.0;
  }

  double* camPosWorld = ren->GetActiveCamera()->GetPosition();
  double camPosVolume[4] = { camPosWorld[0], camPosWorld[1], camPosWorld[2], 1.0 };
  invModelMatrix->MultiplyPoint(camPosVolume, camPosVolume);

  uniforms.CameraVolumePos[0] =
    static_cast<float>((camPosVolume[0] - modelBounds[0]) / boundsSize[0]);
  uniforms.CameraVolumePos[1] =
    static_cast<float>((camPosVolume[1] - modelBounds[2]) / boundsSize[1]);
  uniforms.CameraVolumePos[2] =
    static_cast<float>((camPosVolume[2] - modelBounds[4]) / boundsSize[2]);
  uniforms.CameraVolumePos[3] = 1.0f;

  double maxBoundsSize = std::max({ boundsSize[0], boundsSize[1], boundsSize[2] });

  uniforms.SampleDistance =
    static_cast<float>(actualSampleDistance / maxBoundsSize);

  {
    float normFactor = this->ScalarNormalizationFactor;
    uniforms.ScalarMin = static_cast<float>(this->ScalarRange[0] / normFactor);
    uniforms.ScalarMax = static_cast<float>(
      (this->ScalarRange[1] > this->ScalarRange[0]
         ? this->ScalarRange[1]
         : this->ScalarRange[0] + 1.0) /
      normFactor);
  }

  uniforms.UseJittering = this->GetUseJittering() ? 1.0f : 0.0f;

  // Gradient-based shading uniforms
  {
    vtkVolumeProperty* property = vol->GetProperty();
    bool shadeOn = property && property->GetShade();
    bool hasGradOp = property && property->HasGradientOpacity();

    uniforms.UseGradientShading = shadeOn ? 1.0f : 0.0f;
    uniforms.UseGradientOpacity = (shadeOn && hasGradOp) ? 1.0f : 0.0f;

    // Gradient step: 1/dims per axis for central differences in [0,1] space
    int dims[3];
    input->GetDimensions(dims);
    for (int k = 0; k < 3; ++k)
    {
      uniforms.GradientStep[k] = (dims[k] > 0) ? 1.0f / dims[k] : 1.0f;
    }

    // Gradient opacity normalization range
    double scalarRange = this->ScalarRange[1] - this->ScalarRange[0];
    if (scalarRange <= 0.0)
      scalarRange = 1.0;
    uniforms.GradientOpacityMin = 0.0f;
    uniforms.GradientOpacityMax = static_cast<float>(
      (scalarRange * 0.25) / this->ScalarNormalizationFactor);

    // Material properties from volume property
    if (property)
    {
      double amb = property->GetAmbient();
      double dif = property->GetDiffuse();
      double spec = property->GetSpecular();
      double power = property->GetSpecularPower();
      uniforms.AmbientColor[0] = uniforms.AmbientColor[1] = uniforms.AmbientColor[2] =
        static_cast<float>(amb);
      uniforms.DiffuseColor[0] = uniforms.DiffuseColor[1] = uniforms.DiffuseColor[2] =
        static_cast<float>(dif);
      uniforms.SpecularColor[0] = uniforms.SpecularColor[1] = uniforms.SpecularColor[2] =
        static_cast<float>(spec);
      uniforms.Shininess = static_cast<float>(power);
    }

    // Light direction: headlight (camera-to-volume direction in volume [0,1] space)
    // The gradient normal points inward (toward increasing scalar), matching the
    // OpenGL convention where normals are negated in the lighting calculation.
    double camDirWorld[3];
    ren->GetActiveCamera()->GetDirectionOfProjection(camDirWorld);
    // Transform to volume-local [0,1] space using inverse model matrix
    double camDirLocal[4] = { camDirWorld[0], camDirWorld[1], camDirWorld[2], 0.0 };
    invModelMatrix->MultiplyPoint(camDirLocal, camDirLocal);
    // Account for anisotropic bounds scaling in texture space:
    // direction in [0,1] space must be divided by bounds size to preserve anisotropy
    camDirLocal[0] /= boundsSize[0];
    camDirLocal[1] /= boundsSize[1];
    camDirLocal[2] /= boundsSize[2];
    // Normalize in volume [0,1] space
    double dirLen = sqrt(camDirLocal[0] * camDirLocal[0] + camDirLocal[1] * camDirLocal[1] +
      camDirLocal[2] * camDirLocal[2]);
    if (dirLen > 1e-10)
    {
      camDirLocal[0] /= dirLen;
      camDirLocal[1] /= dirLen;
      camDirLocal[2] /= dirLen;
    }
    uniforms.LightDirection[0] = static_cast<float>(camDirLocal[0]);
    uniforms.LightDirection[1] = static_cast<float>(camDirLocal[1]);
    uniforms.LightDirection[2] = static_cast<float>(camDirLocal[2]);
  }

  // Cropping regions
  if (this->GetCropping())
  {
    uniforms.UseCropping = 1.0f;

    double croppingRegionPlanes[6];
    this->GetCroppingRegionPlanes(croppingRegionPlanes);

    // Clamp to loaded bounds (same as OpenGL mapper)
    for (int i = 0; i < 3; ++i)
    {
      int minIdx = i * 2;
      int maxIdx = i * 2 + 1;
      croppingRegionPlanes[minIdx] =
        std::max(croppingRegionPlanes[minIdx], modelBounds[minIdx]);
      croppingRegionPlanes[minIdx] =
        std::min(croppingRegionPlanes[minIdx], modelBounds[maxIdx]);
      croppingRegionPlanes[maxIdx] =
        std::max(croppingRegionPlanes[maxIdx], modelBounds[minIdx]);
      croppingRegionPlanes[maxIdx] =
        std::min(croppingRegionPlanes[maxIdx], modelBounds[maxIdx]);
    }

    // Convert from model/data coordinates to volume-local [0,1] space.
    // VTK's GetCroppingRegionPlanes() returns planes in model/data coordinates,
    // so the direct normalization against modelBounds is correct.
    uniforms.CroppingPlanes[0] =
      static_cast<float>((croppingRegionPlanes[0] - modelBounds[0]) / boundsSize[0]);
    uniforms.CroppingPlanes[1] =
      static_cast<float>((croppingRegionPlanes[1] - modelBounds[0]) / boundsSize[0]);
    uniforms.CroppingPlanes[2] =
      static_cast<float>((croppingRegionPlanes[2] - modelBounds[2]) / boundsSize[1]);
    uniforms.CroppingPlanes[3] =
      static_cast<float>((croppingRegionPlanes[3] - modelBounds[2]) / boundsSize[1]);
    uniforms.CroppingPlanes2[0] =
      static_cast<float>((croppingRegionPlanes[4] - modelBounds[4]) / boundsSize[2]);
    uniforms.CroppingPlanes2[1] =
      static_cast<float>((croppingRegionPlanes[5] - modelBounds[4]) / boundsSize[2]);
    uniforms.CroppingPlanes2[2] = 0.0f;
    uniforms.CroppingPlanes2[3] = 0.0f;

    uniforms.CroppingBitmask = static_cast<uint32_t>(this->GetCroppingRegionFlags());
  }
  else
  {
    uniforms.UseCropping = 0.0f;
  }

  // Clipping planes
  this->SetClippingPlaneUniforms(&uniforms, ren, vol, modelMatrix, invModelMatrix);

  // Mask / label map
  this->SetMaskUniforms(&uniforms, vol);

  // Depth texture flag — set to 1 when we have a real scene depth texture
  uniforms.UseDepthTexture = this->DepthTextureOcclusion ? 1.0f : 0.0f;

  // Min-max acceleration texture
  uniforms.UseMinMaxAccel = this->MinMaxTexture ? 1.0f : 0.0f;
  uniforms.MinMaxDimX = static_cast<float>(this->MinMaxDims[0]);
  uniforms.MinMaxDimY = static_cast<float>(this->MinMaxDims[1]);
  uniforms.MinMaxDimZ = static_cast<float>(this->MinMaxDims[2]);

  // Precomputed gradient normal texture (Phase 4)
  bool hasNormalTexture = (this->GradientNormalTexture != nullptr);
  if (!hasNormalTexture && !this->Blocks.empty())
  {
    for (auto& block : this->Blocks)
    {
      if (block.NormalTexture)
      {
        hasNormalTexture = true;
        break;
      }
    }
  }
  uniforms.UseNormalTexture = hasNormalTexture ? 1.0f : 0.0f;

  // Build feature mask for shader function constant specialization.
  // When any feature is actively used at runtime, the corresponding bit is set
  // so GetOrCreateVolumePipeline selects a PSO compiled with that constant = 1.
  int featureMask = 0;
  if (uniforms.UseGradientShading > 0.5f)
    featureMask |= VolumeFeature_Shading;
  if (uniforms.UseGradientOpacity > 0.5f)
    featureMask |= VolumeFeature_GradientOpacity;
  if (uniforms.UseMask > 0.5f)
    featureMask |= VolumeFeature_Mask;
  if (uniforms.UseMinMaxAccel > 0.5f || !this->Blocks.empty())
    featureMask |= VolumeFeature_MinMax;
  if (hasNormalTexture)
    featureMask |= VolumeFeature_NormalTexture;

  // Determine if image-space downsampling is active.
  // Force offscreen rendering when blocks are present to enable inter-block
  // opacity propagation via Metal framebuffer fetch ([[color(0)]]).
  const float imageSampleDist = this->ImageSampleDistance;
  const bool useImageSampling = (imageSampleDist != 1.0f) || !this->Blocks.empty();

  // Viewport size for depth texture UV computation in the shader
  int* winSize = ren->GetSize();
  int renderWidth = winSize[0];
  int renderHeight = winSize[1];
  if (useImageSampling)
  {
    renderWidth = std::max(1, static_cast<int>(winSize[0] / imageSampleDist));
    renderHeight = std::max(1, static_cast<int>(winSize[1] / imageSampleDist));
  }
  uniforms.ViewportSize[0] = static_cast<float>(renderWidth);
  uniforms.ViewportSize[1] = static_cast<float>(renderHeight);

  // Compute view-projection matrix via generic vtkCamera API.
  // Try the Metal camera first (has a precomputed cached layout), fall back to
  // computing it from vtkCamera::GetViewTransformMatrix /
  // GetProjectionTransformMatrix so the mapper stays functional even if the
  // camera override is not in place.
  vtkMetalCamera* metalCamera = vtkMetalCamera::SafeDownCast(ren->GetActiveCamera());
  if (metalCamera)
  {
    const float* sceneData = static_cast<const float*>(metalCamera->GetCachedSceneTransforms());
    const float* V = sceneData;         // ViewMatrix at offset 0
    const float* P = sceneData + 16;    // ProjectionMatrix at offset 64 (16 floats)
    for (int c = 0; c < 4; ++c)
    {
      for (int r = 0; r < 4; ++r)
      {
        uniforms.ViewProjectionMatrix[c * 4 + r] = P[0 * 4 + r] * V[c * 4 + 0] +
          P[1 * 4 + r] * V[c * 4 + 1] + P[2 * 4 + r] * V[c * 4 + 2] +
          P[3 * 4 + r] * V[c * 4 + 3];
      }
    }
  }
  else
  {
    // Generic fallback: compute VP from vtkCamera matrices.
    // Metal clip-space uses Z in [0,1], so nearz=0, farz=1.
    vtkCamera* cam = ren->GetActiveCamera();
    int* size = ren->GetSize();
    double aspect = (size[1] > 0) ? static_cast<double>(size[0]) / size[1] : 1.0;
    vtkMatrix4x4* V4 = cam->GetViewTransformMatrix();
    vtkMatrix4x4* P4 = cam->GetProjectionTransformMatrix(aspect, 0.0, 1.0);
    // Compute P*V column-major (Metal convention: column vectors)
    for (int c = 0; c < 4; ++c)
    {
      for (int r = 0; r < 4; ++r)
      {
        float sum = 0.0f;
        for (int k = 0; k < 4; ++k)
          sum += static_cast<float>(P4->GetElement(r, k)) *
                 static_cast<float>(V4->GetElement(k, c));
        uniforms.ViewProjectionMatrix[c * 4 + r] = sum;
      }
    }
  }

  // Compute inverse view-projection matrix for depth buffer occlusion.
  // Used in the fragment shader to unproject depth values to world space.
  {
    float VP[16];
    memcpy(VP, uniforms.ViewProjectionMatrix, sizeof(VP));
    float invDet = 0.0f;
    float invVP[16];

    // 4x4 inverse via cofactors (inline to avoid vtkMatrix4x4 dependency in hot path)
    invVP[0] = VP[5] * (VP[10] * VP[15] - VP[11] * VP[14]) -
               VP[9] * (VP[6] * VP[15] - VP[7] * VP[14]) +
               VP[13] * (VP[6] * VP[11] - VP[7] * VP[10]);
    invVP[4] = -VP[4] * (VP[10] * VP[15] - VP[11] * VP[14]) +
               VP[8] * (VP[6] * VP[15] - VP[7] * VP[14]) -
               VP[12] * (VP[6] * VP[11] - VP[7] * VP[10]);
    invVP[8] = VP[4] * (VP[9] * VP[15] - VP[11] * VP[13]) -
               VP[8] * (VP[5] * VP[15] - VP[7] * VP[13]) +
               VP[12] * (VP[5] * VP[11] - VP[7] * VP[9]);
    invVP[12] = -VP[4] * (VP[9] * VP[14] - VP[10] * VP[13]) +
                VP[8] * (VP[5] * VP[14] - VP[6] * VP[13]) -
                VP[12] * (VP[5] * VP[10] - VP[6] * VP[9]);
    invVP[1] = -VP[1] * (VP[10] * VP[15] - VP[11] * VP[14]) +
               VP[9] * (VP[2] * VP[15] - VP[3] * VP[14]) -
               VP[13] * (VP[2] * VP[11] - VP[3] * VP[10]);
    invVP[5] = VP[0] * (VP[10] * VP[15] - VP[11] * VP[14]) -
               VP[8] * (VP[2] * VP[15] - VP[3] * VP[14]) +
               VP[12] * (VP[2] * VP[11] - VP[3] * VP[10]);
    invVP[9] = -VP[0] * (VP[9] * VP[15] - VP[11] * VP[13]) +
               VP[8] * (VP[1] * VP[15] - VP[3] * VP[13]) -
               VP[12] * (VP[1] * VP[11] - VP[3] * VP[9]);
    invVP[13] = VP[0] * (VP[9] * VP[14] - VP[10] * VP[13]) -
                VP[8] * (VP[1] * VP[14] - VP[2] * VP[13]) +
                VP[12] * (VP[1] * VP[10] - VP[2] * VP[9]);
    invVP[2] = VP[1] * (VP[6] * VP[15] - VP[7] * VP[14]) -
               VP[5] * (VP[2] * VP[15] - VP[3] * VP[14]) +
               VP[13] * (VP[2] * VP[7] - VP[3] * VP[6]);
    invVP[6] = -VP[0] * (VP[6] * VP[15] - VP[7] * VP[14]) +
               VP[4] * (VP[2] * VP[15] - VP[3] * VP[14]) -
               VP[12] * (VP[2] * VP[7] - VP[3] * VP[6]);
    invVP[10] = VP[0] * (VP[5] * VP[15] - VP[7] * VP[13]) -
                VP[4] * (VP[1] * VP[15] - VP[3] * VP[13]) +
                VP[12] * (VP[1] * VP[7] - VP[3] * VP[5]);
    invVP[14] = -VP[0] * (VP[5] * VP[14] - VP[6] * VP[13]) +
                VP[4] * (VP[1] * VP[14] - VP[2] * VP[13]) -
                VP[12] * (VP[1] * VP[6] - VP[2] * VP[5]);
    invVP[3] = -VP[1] * (VP[6] * VP[11] - VP[7] * VP[10]) +
               VP[5] * (VP[2] * VP[11] - VP[3] * VP[10]) -
               VP[9] * (VP[2] * VP[7] - VP[3] * VP[6]);
    invVP[7] = VP[0] * (VP[6] * VP[11] - VP[7] * VP[10]) -
               VP[4] * (VP[2] * VP[11] - VP[3] * VP[10]) +
               VP[8] * (VP[2] * VP[7] - VP[3] * VP[6]);
    invVP[11] = -VP[0] * (VP[5] * VP[11] - VP[7] * VP[9]) +
                VP[4] * (VP[1] * VP[11] - VP[3] * VP[9]) -
                VP[8] * (VP[1] * VP[7] - VP[3] * VP[5]);
    invVP[15] = VP[0] * (VP[5] * VP[10] - VP[6] * VP[9]) -
                VP[4] * (VP[1] * VP[10] - VP[2] * VP[9]) +
                VP[8] * (VP[1] * VP[6] - VP[2] * VP[5]);

    invDet = VP[0] * invVP[0] + VP[1] * invVP[4] + VP[2] * invVP[8] + VP[3] * invVP[12];
    if (fabs(invDet) > 1e-10f)
    {
      float invDetRcp = 1.0f / invDet;
      for (int i = 0; i < 16; ++i)
        uniforms.InverseViewProjection[i] = invVP[i] * invDetRcp;
    }
    else
    {
      memset(uniforms.InverseViewProjection, 0, sizeof(uniforms.InverseViewProjection));
    }
  }

  // Capture the scene depth texture for early ray termination.
  // The depth buffer is written by opaque geometry in the earlier render pass.
  // When MSAA is active, the depth texture is multisampled and cannot be sampled
  // directly by a shader — disable depth occlusion in that case.
  int sampleCount = metalRenderWindow ? metalRenderWindow->GetEffectiveSampleCount() : 1;
  this->DepthTextureOcclusion = (sampleCount > 1) ? nullptr : metalRenderWindow->GetDepthTexture();

  // Wait for the uniform buffer slot for this frame to be free
  dispatch_semaphore_wait((dispatch_semaphore_t)this->FrameSemaphore, DISPATCH_TIME_FOREVER);

  // RAII guard: signals the semaphore on scope exit (early return, exception, etc.).
  // Dismiss after the completion handler is installed below.
  struct SemaphoreSignalGuard {
    dispatch_semaphore_t sem;
    bool active = true;
    ~SemaphoreSignalGuard() { if (active && sem) dispatch_semaphore_signal(sem); }
    void dismiss() { active = false; }
  } semGuard{ (__bridge dispatch_semaphore_t)this->FrameSemaphore };

  int bufIdx = this->UniformFrameIndex % 3;
  this->UniformFrameIndex++;

  // Update uniform buffer (now includes viewProjection + inverseViewProjection)
  id<MTLBuffer> uniformBuf = (__bridge id<MTLBuffer>)this->UniformBuffers[bufIdx];
  memcpy([uniformBuf contents], &uniforms, sizeof(uniforms));

  // Phase 6: Determine if camera is inside the volume for the fullscreen ray-cast path.
  bool cameraInside = this->UseFullscreenCameraInside && this->IsCameraInside(ren, vol);

  if (useImageSampling)
  {
    // Image-space downsampling: render to offscreen texture at reduced resolution,
    // then blit to screen. This cuts fragment count by up to 4x at 0.5x scale.
    int* winSize = ren->GetSize();
    int fboWidth = std::max(1, static_cast<int>(winSize[0] / imageSampleDist));
    int fboHeight = std::max(1, static_cast<int>(winSize[1] / imageSampleDist));

    if (!this->EnsureImageSampleResources(mtlDevice, fboWidth, fboHeight))
    {
      return;
    }

    // Must end the renderer's active encoder before creating a new one
    id<MTLRenderCommandEncoder> currentEncoder =
      (__bridge id<MTLRenderCommandEncoder>)metalRenderWindow->GetCurrentRenderCommandEncoder();
    if (currentEncoder)
    {
      [currentEncoder endEncoding];
      metalRenderWindow->SetCurrentRenderCommandEncoder(nullptr);
    }

    // Render volume to offscreen texture
    id<MTLTexture> offscreenColor =
      (__bridge id<MTLTexture>)this->ImageSampleColorTexture;

    // ============================================================================
    // ORDER-INDEPENDENT BRICK COMPOSITING (partitioned volumes, <= MAX_LAYER_BRICKS)
    // ----------------------------------------------------------------------------
    // Why this exists: the previous compositor (fragment_volume_accum_main / the
    // old instanced path) folded bricks through a Metal framebuffer fetch in a
    // single global draw order produced by SortBlocksBackToFront(). That is correct
    // ONLY when one total front-to-back order is valid for every pixel at once. For
    // an axis-aligned brick grid that holds while the camera is outside the brick
    // span on every split axis, but the moment the camera sits between two split
    // planes on a split axis (typical side/orbit views), pixels in the top and
    // bottom of the frame traverse the same two bricks in OPPOSITE orders, and no
    // global sort can satisfy both. The mis-ordered brick then composites a
    // high-opacity voxel at the cut (the body surface crossing the partition plane)
    // at full weight instead of attenuated by the true front brick -> a bright ring
    // that "peeks through", visible only in translucent renders (opaque renders hide
    // it via early ray termination) and only at the affected angles. No sampling,
    // slack, or sort-comparator change can fix it, because the requirement is
    // per-pixel order, which a single draw sequence cannot provide.
    //
    // The fix makes compositing order-independent:
    //   1. Each brick is ray-marched INDEPENDENTLY (fragment_volume_main, no fetch,
    //      accumulators start at 0) into its own RGBA16Float layer texture
    //      (texture array slice i). Because each pass writes a different slice,
    //      draw order is irrelevant.
    //   2. A final fullscreen pass (fragment_layer_composite_main) reconstructs the
    //      pixel ray, intersects the <= MAX_LAYER_BRICKS brick boxes to get the TRUE
    //      per-pixel entry order, and folds the non-empty layers front-to-back with
    //      premultiplied Porter-Duff over (R += L*(1-R.a)). Because over is
    //      associative, grouping each brick's samples into one layer and compositing
    //      the layers in true order is bit-equivalent to a single unpartitioned
    //      march -- which is why toggling SetPartitions(1,1,1) now matches.
    // Cost: N+1 passes instead of 1, but the bricks' projected boxes overlap across
    // the whole silhouette, so the bricks had to serialize at every covered pixel
    // anyway; the layer path also drops the framebuffer fetch. Net GPU cost is
    // comparable for <= MAX_LAYER_BRICKS, and we trade a few CPU draw calls for
    // correctness at all camera angles.
    // ============================================================================
    if (!this->Blocks.empty() && this->Blocks.size() <= MAX_LAYER_BRICKS &&
        this->LayerPipelineState && this->CompositePipelineState)
    {
      this->SortBlocksBackToFront(ren, vol);
      int neededSlices = static_cast<int>(this->SortedBlockOrder.size());
      if (!this->EnsureLayerResources(mtlDevice, fboWidth, fboHeight, neededSlices))
      {
        return;
      }

      id<MTLBuffer> indexBuf = (__bridge id<MTLBuffer>)this->IndexBuffer;

      // Get origin and spacing for PerBlockData texture bounds
      vtkImageData* input = vtkImageData::SafeDownCast(this->GetInput());
      int fullExt[6];
      input->GetExtent(fullExt);
      double origin[3], spacing[3];
      input->GetOrigin(origin);
      input->GetSpacing(spacing);

      double* modelBounds = this->ModelBounds;
      double bsz[3] = {
        modelBounds[1] - modelBounds[0],
        modelBounds[3] - modelBounds[2],
        modelBounds[5] - modelBounds[4]
      };
      for (int k = 0; k < 3; ++k)
        if (bsz[k] < 1e-10) bsz[k] = 1.0;

      // One render pass per brick -> its own layer texture. The brick is marched with
      // the SAME shader as the on-screen single-block path (fragment_volume_main), so
      // the per-brick result is exactly that brick's ray segment as one premultiplied
      // layer (C, a). Bindings mirror BindEncoderResources but slot 0 = this brick's
      // volume texture and slot 7 = this brick's min-max texture + the NEAREST
      // min-max sampler (the global min-max is nil in partitioned mode, so do not rely
      // on BindEncoderResources' slot-7 sampler here).
      for (size_t bi = 0; bi < this->SortedBlockOrder.size(); ++bi)
      {
        int si = this->SortedBlockOrder[bi];
        auto& block = this->Blocks[si];

        MTLRenderPassDescriptor* lrpd = [MTLRenderPassDescriptor renderPassDescriptor];
        lrpd.colorAttachments[0].texture = (__bridge id<MTLTexture>)this->LayerTextureArray;
        lrpd.colorAttachments[0].slice = static_cast<NSUInteger>(bi);
        lrpd.colorAttachments[0].loadAction = MTLLoadActionClear;
        lrpd.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
        lrpd.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> layerEnc = [commandBuffer renderCommandEncoderWithDescriptor:lrpd];
        MTLViewport vp = {0, 0, (double)fboWidth, (double)fboHeight, 0.0, 1.0};
        [layerEnc setViewport:vp];

        PerBlockData pbd = {};
        pbd.VolumeBoundsMin[0] = static_cast<float>(block.BoundsMin[0]);
        pbd.VolumeBoundsMin[1] = static_cast<float>(block.BoundsMin[1]);
        pbd.VolumeBoundsMin[2] = static_cast<float>(block.BoundsMin[2]);
        pbd.VolumeBoundsMin[3] = 1.0f;
        pbd.VolumeBoundsMax[0] = static_cast<float>(block.BoundsMax[0]);
        pbd.VolumeBoundsMax[1] = static_cast<float>(block.BoundsMax[1]);
        pbd.VolumeBoundsMax[2] = static_cast<float>(block.BoundsMax[2]);
        pbd.VolumeBoundsMax[3] = 1.0f;

        int texExt[6] = {
          std::max(fullExt[0], block.Extents[0] - 1),
          std::min(fullExt[1], block.Extents[1] + 1),
          std::max(fullExt[2], block.Extents[2] - 1),
          std::min(fullExt[3], block.Extents[3] + 1),
          std::max(fullExt[4], block.Extents[4] - 1),
          std::min(fullExt[5], block.Extents[5] + 1)
        };
        pbd.TextureBoundsMin[0] = static_cast<float>(origin[0] + (texExt[0] - 0.5) * spacing[0]);
        pbd.TextureBoundsMin[1] = static_cast<float>(origin[1] + (texExt[2] - 0.5) * spacing[1]);
        pbd.TextureBoundsMin[2] = static_cast<float>(origin[2] + (texExt[4] - 0.5) * spacing[2]);
        pbd.TextureBoundsMin[3] = 1.0f;
        pbd.TextureBoundsMax[0] = static_cast<float>(origin[0] + (texExt[1] + 0.5) * spacing[0]);
        pbd.TextureBoundsMax[1] = static_cast<float>(origin[1] + (texExt[3] + 0.5) * spacing[1]);
        pbd.TextureBoundsMax[2] = static_cast<float>(origin[2] + (texExt[5] + 0.5) * spacing[2]);
        pbd.TextureBoundsMax[3] = 1.0f;

        for (int k = 0; k < 3; ++k)
          pbd.GradientStep[k] = (block.Dims[k] > 0) ? 1.0f / block.Dims[k] : 1.0f;
        pbd.GradientStep[3] = 0.0f;

        if (block.MinMaxTexture)
        {
          pbd.MinMaxInfo[0] = 1.0f;
          pbd.MinMaxInfo[1] = static_cast<float>(block.MinMaxDims[0]);
          pbd.MinMaxInfo[2] = static_cast<float>(block.MinMaxDims[1]);
          pbd.MinMaxInfo[3] = static_cast<float>(block.MinMaxDims[2]);
          [layerEnc setFragmentTexture:(__bridge id<MTLTexture>)block.MinMaxTexture atIndex:6];
        }
        else
        {
          pbd.MinMaxInfo[0] = pbd.MinMaxInfo[1] = pbd.MinMaxInfo[2] = pbd.MinMaxInfo[3] = 0.0f;
        }

        [layerEnc setVertexBytes:&pbd length:sizeof(PerBlockData) atIndex:2];
        [layerEnc setFragmentBytes:&pbd length:sizeof(PerBlockData) atIndex:2];

        if (cameraInside)
        {
          // Fullscreen camera-inside path: use vertex_fullscreen_main +
          // fragment_volume_fullscreen_main. No vertex/index buffers needed.
          void* fsLayerPso = this->GetOrCreateVolumePipeline(mtlDevice,
            static_cast<uint32_t>(VolumePipelineType::FullscreenOffscreen),
            MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1, featureMask);
          [layerEnc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)fsLayerPso];
          [layerEnc setCullMode:MTLCullModeNone];
          // Bind uniform buffer and textures manually (avoid BindEncoderResources
          // which sets VertexBuffer — null in fullscreen path).
          [layerEnc setFragmentBuffer:uniformBuf offset:0 atIndex:1];
          [layerEnc setFragmentTexture:(__bridge id<MTLTexture>)block.Texture atIndex:0];
          [layerEnc setFragmentTexture:(__bridge id<MTLTexture>)this->ColorOpacityTexture atIndex:1];
          id<MTLTexture> depthTex = this->DepthTextureOcclusion
            ? (__bridge id<MTLTexture>)this->DepthTextureOcclusion
            : (__bridge id<MTLTexture>)this->DummyDepthTexture;
          [layerEnc setFragmentTexture:depthTex atIndex:2];
          if (this->GradientOpacityTexture)
            [layerEnc setFragmentTexture:(__bridge id<MTLTexture>)this->GradientOpacityTexture atIndex:3];
          else
            [layerEnc setFragmentTexture:(__bridge id<MTLTexture>)this->ColorOpacityTexture atIndex:3];
          if (this->MaskTexture)
            [layerEnc setFragmentTexture:(__bridge id<MTLTexture>)this->MaskTexture atIndex:4];
          else
            [layerEnc setFragmentTexture:(__bridge id<MTLTexture>)this->DummyMaskTexture atIndex:4];
          if (this->LabelMapTransferTexture)
            [layerEnc setFragmentTexture:(__bridge id<MTLTexture>)this->LabelMapTransferTexture atIndex:5];
          else
            [layerEnc setFragmentTexture:(__bridge id<MTLTexture>)this->ColorOpacityTexture atIndex:5];
          if (block.MinMaxTexture)
            [layerEnc setFragmentTexture:(__bridge id<MTLTexture>)block.MinMaxTexture atIndex:6];
          else
            [layerEnc setFragmentTexture:(__bridge id<MTLTexture>)this->DummyMinMaxTexture atIndex:6];
          if (block.NormalTexture)
            [layerEnc setFragmentTexture:(__bridge id<MTLTexture>)block.NormalTexture atIndex:7];
          else
            [layerEnc setFragmentTexture:(__bridge id<MTLTexture>)this->DummyVolumeTexture atIndex:7];
          [layerEnc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }
        else
        {
          // Shared bindings (layer pipeline, vertex buf, uniforms, TF/depth/grad/mask/minmax)
          void* layerPso = this->GetOrCreateVolumePipeline(mtlDevice,
            static_cast<uint32_t>(VolumePipelineType::OffscreenLayer),
            MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1, featureMask);
          this->BindEncoderResources(layerEnc, uniformBuf, layerPso, false);
          // Override the nil global texture BindEncoderResources set at index 0
          [layerEnc setFragmentTexture:(__bridge id<MTLTexture>)block.Texture atIndex:0];
          // Override index 7 with per-block normal texture if available
          if (block.NormalTexture)
          {
            [layerEnc setFragmentTexture:(__bridge id<MTLTexture>)block.NormalTexture atIndex:7];
          }

          [layerEnc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                               indexCount:this->IndexCount
                                indexType:MTLIndexTypeUInt32
                              indexBuffer:indexBuf
                        indexBufferOffset:0];
        }
        [layerEnc endEncoding];
      }

      // Composite pass: read the <= MAX_LAYER_BRICKS layers and fold them in TRUE
      // per-pixel depth order (reconstructed ray vs. each brick box). The insertion
      // sort is over <= MAX_LAYER_BRICKS items, so it is free. lc.BlockMin/Max are the
      // bricks' bounds in global [0,1] space -- the SAME convention the brick shaders
      // use for blockMinGlobal/blockMaxGlobal, so the entry params line up.
      //
      // LANDMINE: the ray reconstructed below (the inverseViewProjection unproject at
      // z=0 and z=1, the -ndc.y flip, and the (worldToVolume - boundsMin)/boundsSize
      // affine) MUST stay in lockstep with the brick shaders' ray setup
      // (rayDir = normalize(localPos - cameraVolumePos), localPos in [0,1]). If anyone
      // changes the brick vertex/fragment ray convention, update this reconstruction
      // too, or the per-pixel order will silently disagree with the march and the
      // seam will return.
      LayerCompositeUniforms lc = {};
      for (size_t bi = 0; bi < this->SortedBlockOrder.size(); ++bi)
      {
        auto& block = this->Blocks[this->SortedBlockOrder[bi]];
        for (int k = 0; k < 3; ++k)
        {
          lc.BlockMin[bi][k] = static_cast<float>((block.BoundsMin[k] - modelBounds[2 * k]) / bsz[k]);
          lc.BlockMax[bi][k] = static_cast<float>((block.BoundsMax[k] - modelBounds[2 * k]) / bsz[k]);
        }
        lc.BlockMin[bi][3] = 0.0f;
        lc.BlockMax[bi][3] = 0.0f;
      }
      lc.Params[0] = static_cast<float>(this->SortedBlockOrder.size());

      MTLRenderPassDescriptor* crpd = [MTLRenderPassDescriptor renderPassDescriptor];
      crpd.colorAttachments[0].texture = offscreenColor;
      crpd.colorAttachments[0].loadAction = MTLLoadActionClear;
      crpd.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
      crpd.colorAttachments[0].storeAction = MTLStoreActionStore;
      id<MTLRenderCommandEncoder> compEnc = [commandBuffer renderCommandEncoderWithDescriptor:crpd];
      [compEnc setViewport:(MTLViewport){0, 0, (double)fboWidth, (double)fboHeight, 0.0, 1.0}];
      [compEnc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)this->CompositePipelineState];
      [compEnc setCullMode:MTLCullModeNone];
      [compEnc setFragmentBuffer:uniformBuf offset:0 atIndex:1];
      [compEnc setFragmentBytes:&lc length:sizeof(lc) atIndex:2];

      [compEnc setFragmentTexture:(__bridge id<MTLTexture>)this->LayerTextureArray atIndex:0];

      [compEnc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [compEnc endEncoding];
    }
    else
    {
      // --- ORDER-DEPENDENT PATH: single-block volumes, or partitioned volumes
      //     with > MAX_LAYER_BRICKS bricks (the layer composite is capped at
      //     MAX_LAYER_BRICKS; see "Known limitations"). ---
      MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
      rpd.colorAttachments[0].texture = offscreenColor;
      rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
      rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
      rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

      id<MTLRenderCommandEncoder> offscreenEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:rpd];
      offscreenEncoder.label = @"VTK Volume ImageSample Offscreen";

      MTLViewport metalViewport;
      metalViewport.originX = 0;
      metalViewport.originY = 0;
      metalViewport.width = fboWidth;
      metalViewport.height = fboHeight;
      metalViewport.znear = 0.0;
      metalViewport.zfar = 1.0;
      [offscreenEncoder setViewport:metalViewport];

      // --- ORDER-DEPENDENT PATH: single-block volumes, or partitioned volumes
      //     with > MAX_LAYER_BRICKS bricks (the layer composite is capped at
      //     MAX_LAYER_BRICKS; see "Known limitations"). ---
      bool useAccumulation = !this->Blocks.empty(); // true only for the >8-brick case here
      if (cameraInside)
      {
        // Use fullscreen ray-cast path — no proxy geometry needed.
        this->DrawBlocksFullscreen(offscreenEncoder, uniformBuf, ren, vol,
          &uniforms, invModelMatrix, false);
      }
      else
      {
        void* activePipeline = this->GetOrCreateVolumePipeline(mtlDevice,
          static_cast<uint32_t>(useAccumulation ? VolumePipelineType::OffscreenAccumulation
                                                : VolumePipelineType::OffscreenLayer),
          MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1, featureMask);
        this->BindEncoderResources(offscreenEncoder, uniformBuf, activePipeline, false);

        this->DrawBlocks(offscreenEncoder, uniformBuf, ren, vol, &uniforms, invModelMatrix);
      }

      [offscreenEncoder endEncoding];
    }
    // Note: The renderer's blit phase (Phase 3b) will blit the offscreen
    // texture to the screen after all volumes are rendered.
  }
  else
  {
    // Standard path: render directly to the current encoder
    id<MTLRenderCommandEncoder> encoder =
      (__bridge id<MTLRenderCommandEncoder>)metalRenderWindow->GetCurrentRenderCommandEncoder();

    if (!encoder)
    {
      return;
    }

    // Bind all encoder resources (pipeline, textures, samplers, buffers)
    if (cameraInside)
    {
      // Fullscreen ray-cast path — no proxy geometry, no vertex/index buffers.
      this->DrawBlocksFullscreen(encoder, uniformBuf, ren, vol,
        &uniforms, invModelMatrix, true);
    }
    else
    {
      void* directPso = this->GetOrCreateVolumePipeline(mtlDevice,
        static_cast<uint32_t>(VolumePipelineType::DirectScreen),
        MTLPixelFormatBGRA8Unorm, MTLPixelFormatDepth32Float,
        static_cast<uint32_t>(sampleCount), featureMask);
      this->BindEncoderResources(encoder, uniformBuf, directPso, true);

      // Draw volume — handle partitioned (multi-block) and single-block cases
      this->DrawBlocks(encoder, uniformBuf, ren, vol, &uniforms, invModelMatrix);
    }
  }

  // Signal the semaphore when the GPU finishes this frame's command buffer.
  // Retain the semaphore under MRC to prevent use-after-free if the mapper
  // is destroyed before the handler fires.
  dispatch_semaphore_t sem = (__bridge dispatch_semaphore_t)this->FrameSemaphore;
#if !__has_feature(objc_arc)
  dispatch_retain(sem);
#endif
  [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
    dispatch_semaphore_signal(sem);
#if !__has_feature(objc_arc)
    dispatch_release(sem);
#endif
  }];

  // Completion handler is installed — the guard is no longer needed.
  semGuard.dismiss();

  } // @autoreleasepool
}

// ============================================================================
// KNOWN LIMITATIONS & FUTURE WORK (partitioned volume rendering)
// ----------------------------------------------------------------------------
// 1. BRICK CAP. The order-independent composite reads from a texture array
//    (layers[0..7]) and LayerCompositeUniforms holds [8] bounds, so the fix
//    covers <= MAX_LAYER_BRICKS bricks only. Volumes with more partitions fall
//    through to the order-DEPENDENT accumulation path (fragment_volume_accum_main)
//    and WILL show the boundary seam at some camera angles. To lift the cap,
//    either (a) bind the layers as a texture array and loop the composite over a
//    uniform count (Metal needs the array size at compile time, so pick a new
//    higher cap), or (b) composite in log2(N) pairwise fullscreen passes, or
//    (c) switch to a k-buffer / weighted-blended OIT (see #4). Do NOT delete
//    fragment_volume_accum_main / AccumulationPipelineState until this is done --
//    they are the only correct-ish path for > MAX_LAYER_BRICKS today.
//
// 2. SILHOUETTE "TABS" (cosmetic, flat/label-map views only). Each brick's proxy
//    box is its own AABB, so a curved body outline jaggles at brick boundaries in
//    opaque renders. It is invisible in translucent renders and unrelated to the
//    seam. Cheap fix: drive every brick's VERTEX box from the FULL volume bounds
//    (one smooth silhouette) and let the fragment window the ray to its brick via
//    tStart/t.y (the march already does this). Requires the vertex stage to read
//    full bounds while the fragment reads per-brick bounds, so do not reuse the
//    same PerBlockData field for both.
//
// 3. FAINT 1-VOXEL LINE AT INTERNAL CUTS (optional polish). fragment_volume_main
//    still carries the +/-1e-4 boundary slacks. In the old fetch chain these were
//    largely masked; with separate layers they can double-deposit the plane voxel
//    once (front brick's +1e-4 and back brick's -1e-4 entry both touch it),
//    showing as a thin, angle-INDEPENDENT line -- much fainter than the old ring.
//    Only fix if you actually see it: make the boundary half-open in the layer
//    path (break on currentT >= t.y at the brick exit, drop the -1e-4 entry bias),
//    applied to INTERNAL faces only (pass a per-brick "which faces are internal"
//    flag so the outer silhouette keeps its slack). Leave the on-screen single-
//    block path's slacks untouched.
//
// 4. MEMORY ALTERNATIVE. 8x RGBA16Float fullscreen layers at high resolution is
//    the main cost of this fix. If that matters, weighted-blended OIT (two
//    additive accumulators + a final normalize) is order-independent with O(1)
//    storage and no sort, at the price of the usual WBOIT banding on thin/over-
//    lapping high-alpha features. The current per-layer approach is exact, which
//    is why it was chosen; WBOIT is the fallback if memory wins.
//
// 5. SORT IS NOW DECORATIVE. SortBlocksBackToFront no longer affects correctness
//    on this path (the composite re-sorts per pixel); it only fixes which brick
//    lands in which layer slot. Safe to replace with a plain collect-non-empty
//    loop if you want one less camera-dependent branch.
// ============================================================================

VTK_ABI_NAMESPACE_END
