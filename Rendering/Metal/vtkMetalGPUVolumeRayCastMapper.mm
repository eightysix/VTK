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
#import <simd/simd.h>

#include <algorithm>
#include <cstring>
#include <cstddef>
#include <set>
#include <vector>
#include <dispatch/dispatch.h>

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

// ---------------------------------------------------------------------------
// vtkMetalResource — RAII for Metal Obj-C resources stored as void*.
// Implementation in .mm where Obj-C runtime is available.
// ---------------------------------------------------------------------------
vtkMetalResource::~vtkMetalResource() { this->reset(); }

vtkMetalResource::vtkMetalResource(vtkMetalResource&& o) noexcept
  : Obj(o.Obj)
{
  o.Obj = nullptr;
}

vtkMetalResource& vtkMetalResource::operator=(vtkMetalResource&& o) noexcept
{
  if (this != &o)
  {
    this->reset();
    this->Obj = o.Obj;
    o.Obj = nullptr;
  }
  return *this;
}

vtkMetalResource& vtkMetalResource::operator=(void* o)
{
  this->take(o);
  return *this;
}

void vtkMetalResource::reset()
{
  if (this->Obj)
  {
    [(__bridge id)this->Obj release];
    this->Obj = nullptr;
  }
}

void vtkMetalResource::take(void* o)
{
  if (this->Obj != o)
  {
    this->reset();
    this->Obj = o;
  }
}

void vtkMetalResource::retain(void* o)
{
  if (this->Obj != o)
  {
    this->reset();
    this->Obj = o ? (__bridge_retained void*)(__bridge id)o : nullptr;
  }
}

namespace
{
inline uint16_t FloatToHalf(float f)
{
#if defined(__aarch64__) || defined(__ARM_ARCH_7A__)
  // Hardware conversion via __fp16 — handles all IEEE 754 cases correctly
  // (round-to-nearest-even, NaN payload, signed zero, denormals).
  __fp16 h = static_cast<__fp16>(f);
  uint16_t bits;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
#else
  // Software fallback with proper IEEE 754 round-to-nearest-even
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  uint32_t sign = (bits >> 16) & 0x8000;
  int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
  uint32_t mantissa = bits & 0x7FFFFF;

  // NaN / Infinity
  if ((bits & 0x7F800000) == 0x7F800000)
  {
    if ((bits & 0x7FFFFF) == 0)
      return static_cast<uint16_t>(sign | 0x7C00);
    // NaN — preserve payload; ensure at least one mantissa bit is set
    uint16_t halfMantissa = (mantissa >> 13) & 0x03FF;
    return static_cast<uint16_t>(sign | 0x7C00 | halfMantissa | (halfMantissa == 0 ? 1 : 0));
  }

  // Denormals / underflow
  if (exponent <= 0)
  {
    if (exponent < -10)
      return static_cast<uint16_t>(sign);
    mantissa = (mantissa | 0x800000) >> (1 - exponent);
    uint32_t roundBit = (mantissa >> 12) & 1;
    uint32_t sticky = mantissa & 0xFFF;
    mantissa >>= 13;
    if (roundBit && (sticky > 0 || (mantissa & 1)))
      ++mantissa;
    return static_cast<uint16_t>(sign | (mantissa > 0x3FF ? 0x7C00 : mantissa));
  }

  // Overflow to infinity
  if (exponent > 30)
    return static_cast<uint16_t>(sign | 0x7C00);

  // Normal number with round-to-nearest-even
  uint32_t halfMantissa = mantissa >> 13;
  uint32_t roundBit = (mantissa >> 12) & 1;
  uint32_t sticky = mantissa & 0xFFF;

  if (roundBit && (sticky > 0 || (halfMantissa & 1)))
  {
    ++halfMantissa;
    if (halfMantissa > 0x3FF)
    {
      halfMantissa = 0;
      ++exponent;
      if (exponent > 30)
        return static_cast<uint16_t>(sign | 0x7C00);
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | halfMantissa);
#endif
}

// Returns true when half-float can safely represent the full scalar range.
inline bool HalfRangeIsSafe(double r0, double r1)
{
  const double halfMax = 65504.0;
  return std::isfinite(r0) && std::isfinite(r1) &&
    r0 >= -halfMax && r1 <= halfMax;
}

//------------------------------------------------------------------------------
// Templated data conversion: source type -> half
template <typename SrcType>
void ConvertToHalf(const SrcType* src, uint16_t* dst,
                   vtkIdType numTuples, int numComp, int outComp) {
    vtkSMPTools::For(0, numTuples, [&](vtkIdType b, vtkIdType e) {
        for (vtkIdType i = b; i < e; ++i)
            for (int c = 0; c < outComp; ++c)
                dst[i * outComp + c] = (c < numComp)
                    ? FloatToHalf(static_cast<float>(src[i * numComp + c]))
                    : FloatToHalf(0.0f);
    });
}

// Templated data conversion: source type -> float
template <typename SrcType>
void ConvertToFloat(const SrcType* src, float* dst,
                    vtkIdType numTuples, int numComp, int outComp) {
    vtkSMPTools::For(0, numTuples, [&](vtkIdType b, vtkIdType e) {
        for (vtkIdType i = b; i < e; ++i)
            for (int c = 0; c < outComp; ++c)
                dst[i * outComp + c] = (c < numComp)
                    ? static_cast<float>(src[i * numComp + c])
                    : 0.0f;
    });
}

//------------------------------------------------------------------------------
// Shared helper: convert volume data from various data types to half/float.
// Writes outputComponents interleaved components per tuple (3-component sources
// are expanded to 4 with a zero pad component).
void ConvertVolumeData(const void* src, int dataType, int numComponents,
  vtkIdType numTuples, void* dst, bool useHalf,
  int outputComponents, vtkDataArray* scalars)
{
  if (useHalf)
  {
      // Accelerate fast path for single-component float -> half.
      // Use the original src as the read-only source buffer to avoid
      // overlapping source/destination in vImageConvert_PlanarFtoPlanar16F.
      if (dataType == VTK_FLOAT && numComponents == 1)
      {
        vImage_Buffer vSrc = { const_cast<void*>(src), 1,
          static_cast<vImagePixelCount>(numTuples),
          static_cast<vImagePixelCount>(numTuples * sizeof(float)) };
        vImage_Buffer vDst = { dst, 1,
          static_cast<vImagePixelCount>(numTuples),
          static_cast<vImagePixelCount>(numTuples * sizeof(uint16_t)) };
        vImageConvert_PlanarFtoPlanar16F(&vSrc, &vDst, 0);
        return;
      }
    uint16_t* h = static_cast<uint16_t*>(dst);
    switch (dataType)
    {
      case VTK_SHORT:        ConvertToHalf(static_cast<const short*>(src), h, numTuples, numComponents, outputComponents); break;
      case VTK_INT:          ConvertToHalf(static_cast<const int*>(src), h, numTuples, numComponents, outputComponents); break;
      case VTK_UNSIGNED_INT: ConvertToHalf(static_cast<const unsigned int*>(src), h, numTuples, numComponents, outputComponents); break;
      case VTK_DOUBLE:       ConvertToHalf(static_cast<const double*>(src), h, numTuples, numComponents, outputComponents); break;
      case VTK_UNSIGNED_CHAR: ConvertToHalf(static_cast<const unsigned char*>(src), h, numTuples, numComponents, outputComponents); break;
      case VTK_UNSIGNED_SHORT: ConvertToHalf(static_cast<const unsigned short*>(src), h, numTuples, numComponents, outputComponents); break;
      case VTK_FLOAT:        ConvertToHalf(static_cast<const float*>(src), h, numTuples, numComponents, outputComponents); break;
      default:
      {
        vtkSMPTools::For(0, numTuples, [&](vtkIdType b, vtkIdType e) {
          for (vtkIdType i = b; i < e; ++i)
            for (int c = 0; c < outputComponents; ++c)
              h[i * outputComponents + c] = (c < numComponents)
                ? FloatToHalf(static_cast<float>(scalars->GetComponent(i, c)))
                : FloatToHalf(0.0f);
        });
        break;
      }
    }
  }
  else
  {
    float* f = static_cast<float*>(dst);
    switch (dataType)
    {
      case VTK_SHORT:        ConvertToFloat(static_cast<const short*>(src), f, numTuples, numComponents, outputComponents); break;
      case VTK_INT:          ConvertToFloat(static_cast<const int*>(src), f, numTuples, numComponents, outputComponents); break;
      case VTK_UNSIGNED_INT: ConvertToFloat(static_cast<const unsigned int*>(src), f, numTuples, numComponents, outputComponents); break;
      case VTK_FLOAT:        ConvertToFloat(static_cast<const float*>(src), f, numTuples, numComponents, outputComponents); break;
      case VTK_UNSIGNED_CHAR: ConvertToFloat(static_cast<const unsigned char*>(src), f, numTuples, numComponents, outputComponents); break;
      case VTK_UNSIGNED_SHORT: ConvertToFloat(static_cast<const unsigned short*>(src), f, numTuples, numComponents, outputComponents); break;
      case VTK_DOUBLE:       ConvertToFloat(static_cast<const double*>(src), f, numTuples, numComponents, outputComponents); break;
      default:
      {
        vtkSMPTools::For(0, numTuples, [&](vtkIdType b, vtkIdType e) {
          for (vtkIdType i = b; i < e; ++i)
            for (int c = 0; c < outputComponents; ++c)
              f[i * outputComponents + c] = (c < numComponents)
                ? static_cast<float>(scalars->GetComponent(i, c))
                : 0.0f;
        });
        break;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// EnsureTexture3D — create or reuse a 3D texture matching the given format/size.
// Returns the existing texture if sizes and format match, or creates a new one.
// ---------------------------------------------------------------------------
static id<MTLTexture> EnsureTexture3D(id<MTLDevice> device, vtkMetalResource& slot,
  MTLPixelFormat format, NSUInteger w, NSUInteger h, NSUInteger d,
  MTLTextureUsage usage, MTLStorageMode storage)
{
  id<MTLTexture> existing = (__bridge id<MTLTexture>)slot.get();
  if (existing && existing.width == w && existing.height == h &&
      existing.depth == d && existing.pixelFormat == format &&
      existing.storageMode == storage)
  {
    return existing;
  }

  MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                   width:w
                                                                                  height:h
                                                                               mipmapped:NO];
  desc.textureType = MTLTextureType3D;
  desc.depth = d;
  desc.usage = usage;
  desc.storageMode = storage;

  id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
  slot.take((__bridge void*)tex);
  return tex;
}

// ---------------------------------------------------------------------------
// EnsureTexture2D — create or reuse a 2D texture matching the given format/size.
// ---------------------------------------------------------------------------
static id<MTLTexture> EnsureTexture2D(id<MTLDevice> device, vtkMetalResource& slot,
  MTLPixelFormat format, NSUInteger w, NSUInteger h,
  MTLTextureUsage usage, MTLStorageMode storage, NSUInteger slices = 1)
{
  id<MTLTexture> existing = (__bridge id<MTLTexture>)slot.get();
  if (existing && existing.width == w && existing.height == h &&
      existing.pixelFormat == format && existing.storageMode == storage &&
      existing.textureType == (slices > 1 ? MTLTextureType2DArray : MTLTextureType2D))
  {
    return existing;
  }

  MTLTextureDescriptor* desc;
  if (slices > 1)
  {
    desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
    desc.textureType = MTLTextureType2DArray;
    desc.arrayLength = slices;
  }
  else
  {
    desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
  }
  desc.usage = usage;
  desc.storageMode = storage;

  id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
  slot.take((__bridge void*)tex);
  return tex;
}

// ---------------------------------------------------------------------------
// GetVoxelScalar — read a single voxel value from any VTK data type.
// ---------------------------------------------------------------------------
template <typename Functor>
static void GetVoxelScalar(int dataType, const void* dataPtr, vtkIdType idx, Functor&& func,
  vtkDataArray* scalars = nullptr)
{
  switch (dataType)
  {
    vtkTemplateMacro(
      using T = VTK_TT;
      func(static_cast<float>(static_cast<const T*>(dataPtr)[idx]));
    );
    default:
    {
      if (scalars)
      {
        func(static_cast<float>(scalars->GetComponent(idx, 0)));
      }
      break;
    }
  }
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
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::WaitForInFlightFrames()
{
  if (!this->FrameSemaphore)
  {
    return;
  }
  dispatch_semaphore_t sem = (__bridge dispatch_semaphore_t)this->FrameSemaphore.get();
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

    CachedShaderLibrary.take((__bridge void*)library);
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
    id<MTLTexture> colorTex = EnsureTexture2D(device, this->ImageSampleColorTexture,
      MTLPixelFormatRGBA16Float, width, height,
      MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead, MTLStorageModePrivate);
    if (!colorTex)
    {
      vtkErrorMacro("Failed to create image-sample color texture");
      return false;
    }

    // Create blit pipeline (fullscreen quad that samples the offscreen texture)
    if (!this->EnsureShaderLibrary(deviceVoid))
    {
      this->ReleaseImageSampleResources();
      return false;
    }
    id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary.get();

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
    // The +1 from new() goes to the cache.  AssignRetainedMetalObject
    // adds a separate +1 for the member slot.
    {
      VolumePipelineKey k = { static_cast<uint32_t>(VolumePipelineType::ImageSampleBlit),
        MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1, 0 };
      this->PipelineCache[k] = (__bridge void*)pso;
    }
    ImageSamplePipeline.retain((__bridge void*)pso);

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
  ImageSampleColorTexture.reset();
  ImageSampleDepthTexture.reset();
  ImageSamplePipeline.reset();
  this->ImageSampleFBOWidth = 0;
  this->ImageSampleFBOHeight = 0;
  this->ImageSamplePixelFormat = 0;

  // Release order-independent compositing layer texture array
  LayerTextureArray.reset();
  this->LayerTextureCapacity = 0;
  this->LayerFBOWidth = 0;
  this->LayerFBOHeight = 0;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ReleaseGradientNormalTexture()
{
  GradientNormalTexture.reset();
  NormalComputePipeline.reset();
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

  id<MTLTexture> volTex = (__bridge id<MTLTexture>)this->VolumeTexture.get();
  int dims[3] = { static_cast<int>(volTex.width), static_cast<int>(volTex.height), static_cast<int>(volTex.depth) };

  // Reuse if still valid — data, scalar range, and params haven't changed
  id<MTLTexture> oldTex = (__bridge id<MTLTexture>)this->GradientNormalTexture.get();
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
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary.get();

  @autoreleasepool
  {
    id<MTLTexture> normalTex = EnsureTexture3D(device, this->GradientNormalTexture,
      MTLPixelFormatRGBA8Unorm, dims[0], dims[1], dims[2],
      MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, MTLStorageModePrivate);
    if (!normalTex)
    {
      vtkErrorMacro("Failed to create gradient normal texture");
      return false;
    }
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
      NormalComputePipeline.take((__bridge void*)cps);
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
    [compEnc setComputePipelineState:(__bridge id<MTLComputePipelineState>)this->NormalComputePipeline.get()];
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
  LayerTextureArray.reset();
  this->LayerTextureCapacity = 0;

  id<MTLDevice> device = (__bridge id<MTLDevice>)deviceVoid;
  id<MTLTexture> texArray = EnsureTexture2D(device, this->LayerTextureArray,
    MTLPixelFormatRGBA16Float, w, h,
    MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead, MTLStorageModePrivate, capacity);
  if (!texArray)
  {
    vtkErrorMacro("Failed to create layer texture array");
    return false;
  }
  this->LayerTextureCapacity = capacity;
  this->LayerFBOWidth = w;
  this->LayerFBOHeight = h;
  return true;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ReleaseGraphicsResources(vtkWindow*)
{
  this->ReleaseImageSampleResources();
  this->ClearBlocks();
  this->ReleaseGradientNormalTexture();
  this->ReleaseMaskResources();

  for (auto& entry : this->PipelineCache)
  {
    [(__bridge id)entry.second release];
  }
  this->PipelineCache.clear();
  this->PipelinesPreWarmed = false;
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
    bool needsReblockify = this->Blocks.empty();
    bool needsDataReload = (input->GetMTime() > this->VolumeUploadTime.GetMTime());
    bool needsMapperReload = (this->GetMTime() > this->VolumeUploadTime.GetMTime());

    vtkVolumeProperty* property = vol ? vol->GetProperty() : nullptr;
    vtkPiecewiseFunction* opacityFunc =
      property ? property->GetScalarOpacity() : nullptr;
    bool needsOpacityReload =
      opacityFunc && (opacityFunc->GetMTime() > this->VolumeUploadTime.GetMTime());

    if (needsReblockify || needsDataReload || needsMapperReload)
    {
      // Full rebuild: re-blockify and re-upload everything
      int fullExt[6];
      input->GetExtent(fullExt);

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
    else if (needsOpacityReload && !this->Blocks.empty())
    {
      // Only opacity/TF changed: update empty-block classification and
      // minmax textures without re-uploading scalar voxel data.
      if (this->MacrocellScalarMin.empty() || this->MacrocellScalarMax.empty())
      {
        this->UpdateMinMaxTexture(mtlDeviceVoid, vol, input, scalars, true);
      }
      if (!this->UpdateBlockMinMaxTextures(
            mtlDeviceVoid, mtlQueueVoid, vol, input, scalars, this->VolumeNumComponents))
      {
        // Fallback: if the lightweight update cannot handle the transition
        // (e.g. empty→non-empty block), do a full rebuild.
        this->ClearBlocks();
        return this->UpdateVolumeTexture(mtlDeviceVoid, mtlQueueVoid, vol);
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

      if (dims[0] < 1 || dims[1] < 1 || dims[2] < 1)
      {
        vtkErrorMacro("Volume has zero dimensions");
        return false;
      }

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
          if (this->ScalarRange[0] >= 0.0 && this->ScalarRange[1] <= 255.0)
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
          }
          else
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

      bool gpuConversionUsed = false;

      int actualComponents = (numComponents == 3) ? 4 : numComponents;
      NSUInteger bytesPerRow = static_cast<NSUInteger>(dims[0]) * fmtInfo.bytesPerComponent *
        actualComponents;
      NSUInteger bytesPerImage = bytesPerRow * dims[1];
      NSUInteger totalBytes = bytesPerImage * dims[2];

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
        else if (dataType == VTK_FLOAT && useHalf) kernelName = "volume_convert_float_to_half";
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
               if (dataType == VTK_SHORT)        pipeline = (__bridge id<MTLComputePipelineState>)(useHalf ? (id<MTLComputePipelineState>)(void*)this->ConvertShortToHalfPipeline.get() : (id<MTLComputePipelineState>)(void*)this->ConvertShortToFloatPipeline.get());
          else if (dataType == VTK_INT)          pipeline = (__bridge id<MTLComputePipelineState>)(useHalf ? (id<MTLComputePipelineState>)(void*)this->ConvertIntToHalfPipeline.get() : (id<MTLComputePipelineState>)(void*)this->ConvertIntToFloatPipeline.get());
          else if (dataType == VTK_UNSIGNED_INT) pipeline = (__bridge id<MTLComputePipelineState>)(useHalf ? (id<MTLComputePipelineState>)(void*)this->ConvertUIntToHalfPipeline.get() : (id<MTLComputePipelineState>)(void*)this->ConvertUIntToFloatPipeline.get());
          else if (dataType == VTK_FLOAT && useHalf) pipeline = (__bridge id<MTLComputePipelineState>)this->ConvertFloatToHalfPipeline.get();
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
          id<MTLTexture> tex = EnsureTexture3D(device, this->VolumeTexture,
            fmtInfo.format, dims[0], dims[1], dims[2],
            MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead, MTLStorageModePrivate);
          if (!tex)
          {
            vtkErrorMacro("Failed to create 3D volume texture for GPU conversion");
            [srcBuf release];
            return false;
          }

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
          [enc setComputePipelineState:pipeline];
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
      if (!gpuConversionUsed)
      {
        id<MTLBuffer> stagingBuf = [device newBufferWithLength:totalBytes
                                                       options:MTLResourceStorageModeShared];
        if (!stagingBuf)
        {
          vtkErrorMacro("Failed to create volume staging buffer");
          return false;
        }
        void* uploadPointer = [stagingBuf contents];

        if (fmtInfo.needsConversion)
        {
          ConvertVolumeData(scalars->GetVoidPointer(0), dataType, numComponents,
            numTuples, uploadPointer, useHalf, actualComponents, scalars);
        }
        else if (dataType == VTK_FLOAT && numComponents == 3)
        {
          const float* src = static_cast<const float*>(scalars->GetVoidPointer(0));
          float* dst = static_cast<float*>(uploadPointer);
          vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
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
          const unsigned char* src =
            static_cast<const unsigned char*>(scalars->GetVoidPointer(0));
          unsigned char* dst = static_cast<unsigned char*>(uploadPointer);
          vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
            for (vtkIdType i = begin; i < end; ++i)
            {
              dst[i * 4 + 0] = src[i * 3 + 0];
              dst[i * 4 + 1] = src[i * 3 + 1];
              dst[i * 4 + 2] = src[i * 3 + 2];
              dst[i * 4 + 3] = 255;
            }
          });
        }
        else if (dataType == VTK_UNSIGNED_SHORT && this->ScalarNormalizationFactor == 255.0f)
        {
          const unsigned short* src =
            static_cast<const unsigned short*>(scalars->GetVoidPointer(0));
          unsigned char* dst = static_cast<unsigned char*>(uploadPointer);
          vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
            for (vtkIdType i = begin; i < end; ++i)
            {
              for (int c = 0; c < numComponents; ++c)
                dst[i * actualComponents + c] = static_cast<unsigned char>(std::min<unsigned short>(src[i * numComponents + c], 255));
              if (numComponents == 3)
                dst[i * 4 + 3] = 255;
            }
          });
        }
        else if (dataType == VTK_UNSIGNED_SHORT && numComponents == 3)
        {
          const unsigned short* src =
            static_cast<const unsigned short*>(scalars->GetVoidPointer(0));
          unsigned short* dst = static_cast<unsigned short*>(uploadPointer);
          vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
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
          std::memcpy(uploadPointer, scalars->GetVoidPointer(0), totalBytes);
        }

      id<MTLTexture> tex = EnsureTexture3D(device, this->VolumeTexture,
        fmtInfo.format, dims[0], dims[1], dims[2],
        MTLTextureUsageShaderRead, MTLStorageModePrivate);
      if (!tex)
      {
        vtkErrorMacro("Failed to create 3D volume texture");
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
      // Release our reference to the staging buffer. Metal keeps the buffer
      // alive internally until the command buffer completes.
      [stagingBuf release];

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
      // Pre-integrated TF permutations
      VolumeFeature_PreIntegratedTF,
      VolumeFeature_PreIntegratedTF | VolumeFeature_MinMax,
      VolumeFeature_Shading | VolumeFeature_PreIntegratedTF,
      VolumeFeature_Shading | VolumeFeature_PreIntegratedTF | VolumeFeature_MinMax,
      VolumeFeature_Shading | VolumeFeature_PreIntegratedTF | VolumeFeature_GradientOpacity | VolumeFeature_MinMax,
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
      { static_cast<uint32_t>(VolumePipelineType::FullscreenAccumulation),
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
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkVolume* vol)
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

  if (doReload)
  {
    @autoreleasepool
    {
      id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

      unsigned char tfData[256 * 4];
      for (int i = 0; i < 256; ++i)
      {
        double val = this->ScalarRange[0] + (this->ScalarRange[1] - this->ScalarRange[0]) * (i / 255.0);
        double rgb[3];
        colorFunc->GetColor(val, rgb);
        double opacity = opacityFunc->GetValue(val);

        rgb[0] = std::clamp(rgb[0], 0.0, 1.0);
        rgb[1] = std::clamp(rgb[1], 0.0, 1.0);
        rgb[2] = std::clamp(rgb[2], 0.0, 1.0);
        opacity = std::clamp(opacity, 0.0, 1.0);

        tfData[i * 4 + 0] = static_cast<unsigned char>(rgb[0] * 255.0);
        tfData[i * 4 + 1] = static_cast<unsigned char>(rgb[1] * 255.0);
        tfData[i * 4 + 2] = static_cast<unsigned char>(rgb[2] * 255.0);
        tfData[i * 4 + 3] = static_cast<unsigned char>(opacity * 255.0);
      }

      id<MTLTexture> tex = EnsureTexture2D(device, this->ColorOpacityTexture,
        MTLPixelFormatRGBA8Unorm, 256, 1,
        MTLTextureUsageShaderRead, MTLStorageModeShared);
      if (!tex)
      {
        vtkErrorMacro("Failed to create transfer function texture");
        return false;
      }

      MTLRegion region = MTLRegionMake2D(0, 0, 256, 1);
      [tex replaceRegion:region
            mipmapLevel:0
              withBytes:tfData
            bytesPerRow:256 * 4];

      this->LastTransferFunctionScalarRange[0] = this->ScalarRange[0];
      this->LastTransferFunctionScalarRange[1] = this->ScalarRange[1];
      this->TransferFunctionUploadTime.Modified();
    }
  }

  return this->ColorOpacityTexture != nullptr;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::UpdatePreIntegratedTFTexture(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkVolume* vol)
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

  double unitDist = property->GetScalarOpacityUnitDistance(0);
  if (unitDist <= 0.0) unitDist = 1.0;

  bool scalarRangeChanged =
    (this->ScalarRange[0] != this->LastPreIntegScalarRange[0]) ||
    (this->ScalarRange[1] != this->LastPreIntegScalarRange[1]);

  bool doReload = (this->PreIntegratedTFTexture == nullptr);
  doReload |= (colorFunc->GetMTime() > this->PreIntegratedTFUploadTime.GetMTime());
  doReload |= (opacityFunc->GetMTime() > this->PreIntegratedTFUploadTime.GetMTime());
  doReload |= scalarRangeChanged;
  doReload |= (unitDist != this->LastPreIntegUnitDistance);

  if (doReload)
  {
    @autoreleasepool
    {
      id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

      const int TF_SIZE = 256;
      const int N_SUBSTEPS = 32;
      std::vector<uint16_t> texData(TF_SIZE * TF_SIZE * 4);

      vtkSMPTools::For(0, TF_SIZE, [&](vtkIdType begin, vtkIdType end) {
        for (vtkIdType sb = begin; sb < end; ++sb)
        {
          for (int sf = 0; sf < TF_SIZE; ++sf)
          {
            double sfVal = this->ScalarRange[0] +
              (this->ScalarRange[1] - this->ScalarRange[0]) * sf / (TF_SIZE - 1.0);
            double sbVal = this->ScalarRange[0] +
              (this->ScalarRange[1] - this->ScalarRange[0]) * sb / (TF_SIZE - 1.0);

            double integralTau = 0.0;
            double integralColorTau[3] = { 0.0, 0.0, 0.0 };
            double accumulatedTau = 0.0;

            for (int k = 0; k < N_SUBSTEPS; ++k)
            {
              double t0 = static_cast<double>(k) / N_SUBSTEPS;
              double t1 = static_cast<double>(k + 1) / N_SUBSTEPS;
              double sMid = sfVal + (sbVal - sfVal) * (t0 + t1) / 2.0;

              double rgb[3];
              colorFunc->GetColor(sMid, rgb);
              double tau = opacityFunc->GetValue(sMid);

              double dt = 1.0 / N_SUBSTEPS;
              integralTau += tau * dt;

              double extinctionSoFar = accumulatedTau;
              double weight = exp(-extinctionSoFar);
              for (int c = 0; c < 3; ++c)
              {
                integralColorTau[c] += rgb[c] * tau * weight * dt;
              }
              accumulatedTau += tau * dt;
            }

            double piOpacity = 1.0 - exp(-integralTau);
            double piColor[3];
            if (piOpacity > 1e-6)
            {
              for (int c = 0; c < 3; ++c)
                piColor[c] = integralColorTau[c] / piOpacity;
            }
            else
            {
              piColor[0] = piColor[1] = piColor[2] = 0.0;
            }

            int idx = static_cast<int>(sb) * TF_SIZE + sf;
            texData[idx * 4 + 0] = FloatToHalf(
              static_cast<float>(std::clamp(piColor[0], 0.0, 1.0)));
            texData[idx * 4 + 1] = FloatToHalf(
              static_cast<float>(std::clamp(piColor[1], 0.0, 1.0)));
            texData[idx * 4 + 2] = FloatToHalf(
              static_cast<float>(std::clamp(piColor[2], 0.0, 1.0)));
            texData[idx * 4 + 3] = FloatToHalf(
              static_cast<float>(std::clamp(integralTau, 0.0, 64.0)));
          }
        }
      });

      id<MTLTexture> tex = EnsureTexture2D(device, this->PreIntegratedTFTexture,
        MTLPixelFormatRGBA16Float, TF_SIZE, TF_SIZE,
        MTLTextureUsageShaderRead, MTLStorageModeShared);
      if (!tex)
      {
        vtkErrorMacro("Failed to create pre-integrated transfer function texture");
        return false;
      }

      MTLRegion region = MTLRegionMake2D(0, 0, TF_SIZE, TF_SIZE);
      [tex replaceRegion:region
            mipmapLevel:0
              withBytes:texData.data()
            bytesPerRow:TF_SIZE * 4 * sizeof(uint16_t)];

      this->LastPreIntegScalarRange[0] = this->ScalarRange[0];
      this->LastPreIntegScalarRange[1] = this->ScalarRange[1];
      this->LastPreIntegUnitDistance = unitDist;
      this->PreIntegratedTFUploadTime.Modified();
    }
  }

  return this->PreIntegratedTFTexture != nullptr;
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

      id<MTLTexture> tex = EnsureTexture2D(device, this->GradientOpacityTexture,
        MTLPixelFormatRGBA8Unorm, 256, 1,
        MTLTextureUsageShaderRead, MTLStorageModeShared);
      if (!tex)
      {
        vtkErrorMacro("Failed to create gradient opacity texture");
        return false;
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
      id<MTLTexture> tex = EnsureTexture3D(device, this->MaskTexture,
        chosenFormat, dims[0], dims[1], dims[2],
        MTLTextureUsageShaderRead, MTLStorageModeShared);
      if (!tex)
      {
        vtkErrorMacro("Failed to create mask texture");
        return false;
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
      id<MTLTexture> tex = EnsureTexture2D(device, this->LabelMapTransferTexture,
        MTLPixelFormatRGBA8Unorm, tfWidth, tfHeight,
        MTLTextureUsageShaderRead, MTLStorageModeShared);
      if (!tex)
      {
        vtkErrorMacro("Failed to create label map transfer texture");
        return false;
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
    id<MTLTexture> maskTex = (__bridge id<MTLTexture>)this->MaskTexture.get();
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
  MaskTexture.reset();
  LabelMapTransferTexture.reset();
  LabelMapGradientOpacityTexture.reset();
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ClearBlocks()
{
  for (auto& block : this->Blocks)
  {
    block.Texture.reset();
    block.MinMaxTexture.reset();
    block.NormalTexture.reset();
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
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary.get();

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
      MinMaxComputePipeline.take((__bridge void*)pso);
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
      DilateComputePipeline.take((__bridge void*)pso);
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
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary.get();

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
      member.take((__bridge void*)pso); \
    }

    VTK_CREATE_CONVERT_PIPELINE("volume_convert_short_to_half", this->ConvertShortToHalfPipeline);
    VTK_CREATE_CONVERT_PIPELINE("volume_convert_short_to_float", this->ConvertShortToFloatPipeline);
    VTK_CREATE_CONVERT_PIPELINE("volume_convert_int_to_half", this->ConvertIntToHalfPipeline);
    VTK_CREATE_CONVERT_PIPELINE("volume_convert_int_to_float", this->ConvertIntToFloatPipeline);
    VTK_CREATE_CONVERT_PIPELINE("volume_convert_uint_to_half", this->ConvertUIntToHalfPipeline);
    VTK_CREATE_CONVERT_PIPELINE("volume_convert_uint_to_float", this->ConvertUIntToFloatPipeline);
    VTK_CREATE_CONVERT_PIPELINE("volume_convert_float_to_half", this->ConvertFloatToHalfPipeline);
    VTK_CREATE_CONVERT_PIPELINE("volume_convert_ushort_to_uchar", this->ConvertUShortToUCharPipeline);
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

  id<MTLTexture> volTex = (__bridge id<MTLTexture>)this->VolumeTexture.get();
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
    id<MTLTexture> rawOcc = EnsureTexture3D(device, this->MinMaxScratchTexture,
      MTLPixelFormatR8Unorm, mmDims[0], mmDims[1], mmDims[2],
      MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, MTLStorageModePrivate);
    if (!rawOcc) return false;

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
    [enc1 setComputePipelineState:(__bridge id<MTLComputePipelineState>)this->MinMaxComputePipeline.get()];
    [enc1 setTexture:volTex atIndex:0];
    [enc1 setTexture:rawOcc atIndex:1];
    [enc1 setBytes:&u length:sizeof(u) atIndex:0];

    MTLSize gridSize = MTLSizeMake(mmDims[0], mmDims[1], mmDims[2]);
    NSUInteger tgw = 8;
    MTLSize tgSize = MTLSizeMake(tgw, tgw, tgw);
    [enc1 dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [enc1 endEncoding];

    id<MTLTexture> permTex = EnsureTexture3D(device, this->MinMaxTexture,
      MTLPixelFormatR8Unorm, mmDims[0], mmDims[1], mmDims[2],
      MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, MTLStorageModePrivate);
    if (!permTex)
    {
      vtkErrorMacro("Failed to create persistent min-max texture");
      return false;
    }

    // --- Dispatch kernel 2: dilation (writes directly to permTex) ---
    id<MTLComputeCommandEncoder> enc2 = [cmdBuf computeCommandEncoder];
    enc2.label = @"Volume Dilate MinMax";
    [enc2 setComputePipelineState:(__bridge id<MTLComputePipelineState>)this->DilateComputePipeline.get()];
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
      return !this->MacrocellScalarMin.empty();
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

    vtkIdType numCells = static_cast<vtkIdType>(mmDims0) * mmDims1 * mmDims2;

    // Per-macrocell scalar min/max — consumed later by UpdateBlockTextures
    // to compute per-block ranges without re-walking every voxel.
    this->MacrocellScalarMin.resize(numCells, 1e30f);
    this->MacrocellScalarMax.resize(numCells, -1e30f);
    float* mcMin = this->MacrocellScalarMin.data();
    float* mcMax = this->MacrocellScalarMax.data();

    // RAW buffer for occupancy (only needed for global texture)
    std::vector<uint8_t> rawMinMax;
    if (!skipGlobalTexture)
    {
      rawMinMax.resize(numCells, 255);
    }

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
              GetVoxelScalar(dataType, dataPtr, z * inc2 + y * inc1 + x * inc0, [&](float val) { v = val; }, scalars);
              if (v < cellMin) cellMin = v;
              if (v > cellMax) cellMax = v;
            }
          }
        }

        // Store macrocell scalar range for later block-range reduction
        mcMin[cellIdx] = cellMin;
        mcMax[cellIdx] = cellMax;

        if (!skipGlobalTexture)
        {
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
      }
    });

    if (!skipGlobalTexture)
    {
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
      id<MTLTexture> tex = EnsureTexture3D(device, this->MinMaxTexture,
        MTLPixelFormatR8Unorm, mmDims0, mmDims1, mmDims2,
        MTLTextureUsageShaderRead, MTLStorageModeShared);
      if (!tex)
      {
        vtkErrorMacro("Failed to create min-max acceleration texture");
        return false;
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

  if (skipGlobalTexture)
  {
    return !this->MacrocellScalarMin.empty();
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
        if (this->ScalarRange[0] >= 0.0 && this->ScalarRange[1] <= 255.0)
        {
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
        }
        else
        {
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

    // Use one command buffer for all work (conversion, upload, minmax, normals)
    id<MTLCommandBuffer> uploadCmdBuf = [queue commandBuffer];

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
      else if (dataType == VTK_FLOAT && blockUseHalf) kernelName = "volume_convert_float_to_half";
      // VTK_DOUBLE: kernelName stays nullptr (Metal does not support double in device address space)

      if (kernelName && this->EnsureConversionPipelines(mtlDeviceVoid))
      {
        id<MTLComputePipelineState> pipeline = nullptr;
             if (dataType == VTK_SHORT)        pipeline = (__bridge id<MTLComputePipelineState>)(blockUseHalf ? (id<MTLComputePipelineState>)(void*)this->ConvertShortToHalfPipeline.get() : (id<MTLComputePipelineState>)(void*)this->ConvertShortToFloatPipeline.get());
        else if (dataType == VTK_INT)          pipeline = (__bridge id<MTLComputePipelineState>)(blockUseHalf ? (id<MTLComputePipelineState>)(void*)this->ConvertIntToHalfPipeline.get() : (id<MTLComputePipelineState>)(void*)this->ConvertIntToFloatPipeline.get());
        else if (dataType == VTK_UNSIGNED_INT) pipeline = (__bridge id<MTLComputePipelineState>)(blockUseHalf ? (id<MTLComputePipelineState>)(void*)this->ConvertUIntToHalfPipeline.get() : (id<MTLComputePipelineState>)(void*)this->ConvertUIntToFloatPipeline.get());
        else if (dataType == VTK_FLOAT && blockUseHalf) pipeline = (__bridge id<MTLComputePipelineState>)this->ConvertFloatToHalfPipeline.get();

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
            id<MTLTexture> gpuFullTex = EnsureTexture3D(device, this->DummyVolumeTexture,
              pixelFormat, fullDims[0], fullDims[1], fullDims[2],
              MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, MTLStorageModePrivate);
            if (!gpuFullTex) { return false; }
            id<MTLComputeCommandEncoder> enc = [uploadCmdBuf computeCommandEncoder];
            enc.label = @"VTK Block Volume Convert";
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
            [srcBuf release];

            gpuConversionUsed = true;
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
        ConvertVolumeData(fullDataPtr, dataType, numComponents,
          totalTuples, uploadPointer, blockUseHalf, outputComponents, scalars);
      }
      else if (dataType == VTK_UNSIGNED_SHORT && bytesPerComponent == 1)
      {
        int outComp = (numComponents == 3) ? 4 : numComponents;
        const unsigned short* src = static_cast<const unsigned short*>(fullDataPtr);
        unsigned char* dst = static_cast<unsigned char*>(uploadPointer);
        vtkSMPTools::For(0, totalTuples, [&](vtkIdType begin, vtkIdType end) {
          for (vtkIdType i = begin; i < end; ++i)
          {
            for (int c = 0; c < numComponents; ++c)
              dst[i * outComp + c] = static_cast<unsigned char>(std::min<unsigned short>(src[i * numComponents + c], 255));
            if (numComponents == 3)
              dst[i * 4 + 3] = 255;
          }
        });
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
          case VTK_INT:
          {
            const int* ptr = static_cast<const int*>(fullDataPtr);
            for (int z = ext[4]; z <= ext[5]; ++z)
            {
              for (int y = ext[2]; y <= ext[3]; ++y)
              {
                const int* row = ptr + z * inc[2] + y * inc[1] + ext[0] * inc[0];
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
          case VTK_UNSIGNED_INT:
          {
            const unsigned int* ptr = static_cast<const unsigned int*>(fullDataPtr);
            for (int z = ext[4]; z <= ext[5]; ++z)
            {
              for (int y = ext[2]; y <= ext[3]; ++y)
              {
                const unsigned int* row = ptr + z * inc[2] + y * inc[1] + ext[0] * inc[0];
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
      id<MTLTexture> tex = EnsureTexture3D(device, block.Texture,
        pixelFormat, bDims[0], bDims[1], bDims[2],
        MTLTextureUsageShaderRead, MTLStorageModePrivate);
      if (!tex)
      {
        vtkErrorMacro(<< "Failed to create block " << idx << " 3D texture ("
                      << bDims[0] << "x" << bDims[1] << "x" << bDims[2] << ")");
        return false;
      }

      // --- Per-block min-max texture generation ---
      // When UseGPUMinMax is true, skip CPU generation and let the GPU compute
      // phase after the blit encoder handle it.  Just store the dims so the
      // GPU phase can reuse them.
      if (this->UseGPUMinMax)
      {
        const int DS = 4;
        block.MinMaxDims[0] = std::max(1, (bDims[0] + DS - 1) / DS);
        block.MinMaxDims[1] = std::max(1, (bDims[1] + DS - 1) / DS);
        block.MinMaxDims[2] = std::max(1, (bDims[2] + DS - 1) / DS);
        block.MinMaxTexture = nullptr;
      }
      else if (hasOpacityFunc)
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
                      case VTK_INT:
                        v = static_cast<float>(
                          static_cast<const int*>(fullDataPtr)[zOffset * inc[2] + yOffset * inc[1] + xOffset * inc[0]]);
                        break;
                      case VTK_UNSIGNED_INT:
                        v = static_cast<float>(
                          static_cast<const unsigned int*>(fullDataPtr)[zOffset * inc[2] + yOffset * inc[1] + xOffset * inc[0]]);
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
        id<MTLTexture> mmTex = EnsureTexture3D(device, block.MinMaxTexture,
          MTLPixelFormatR8Unorm, mmDims0, mmDims1, mmDims2,
          MTLTextureUsageShaderRead, MTLStorageModeShared);
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

    // Phase 5b: GPU per-block min-max texture generation for partitioned volumes.
    // After all block textures are uploaded, dispatch volume_compute_minmax and
    // volume_dilate_minmax compute kernels per block, reusing the same shaders
    // as the single-block GPU min-max path but with per-block texture bindings.
    if (this->UseGPUMinMax && hasOpacityFunc &&
        this->EnsureMinMaxComputePipelines(mtlDeviceVoid))
    {
      // Build opacity prefix table from the transfer function
      double scalarRange = this->ScalarRange[1] - this->ScalarRange[0];
      if (scalarRange <= 0.0) scalarRange = 1.0;
      float normFactor = this->ScalarNormalizationFactor;

      double opacityTable[256];
      opFunc->GetTable(this->ScalarRange[0], this->ScalarRange[1], 256, opacityTable);

      MinMaxComputeUniforms mmu;
      mmu.ds = 4.0f;
      mmu.scalarMin = static_cast<float>(this->ScalarRange[0] / normFactor);
      mmu.scalarScale = static_cast<float>(255.0 * normFactor / scalarRange);
      mmu._pad = 0.0f;
      mmu.opacityPrefix[0] = 0;
      for (int i = 0; i < 256; ++i)
        mmu.opacityPrefix[i + 1] = mmu.opacityPrefix[i] + (opacityTable[i] > 0.0 ? 1u : 0u);

      id<MTLComputeCommandEncoder> mmEnc = [uploadCmdBuf computeCommandEncoder];
      mmEnc.label = @"VTK Block MinMax Compute";

      for (auto& block : this->Blocks)
      {
        if (!block.Texture) continue;

        id<MTLTexture> blockTex = (__bridge id<MTLTexture>)block.Texture.get();
        int bdims[3] = { block.Dims[0], block.Dims[1], block.Dims[2] };

        this->DispatchBlockMinMaxGPU(mtlDeviceVoid, (__bridge void*)mmEnc, (__bridge void*)blockTex,
          block, bdims, normFactor, scalarRange, opacityTable);
      }

      [mmEnc endEncoding];
    }

    // Phase 4: Precompute per-block gradient/normal textures for partitioned volumes.
    if (this->UsePrecomputedNormals && !this->Blocks.empty())
    {
      if (!this->EnsureShaderLibrary(mtlDeviceVoid))
      {
        return false;
      }
      id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary.get();

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
            NormalComputePipeline.take((__bridge void*)cps);
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

          id<MTLTexture> blockTex = (__bridge id<MTLTexture>)block.Texture.get();
          int bdims[3] = { static_cast<int>(blockTex.width),
                           static_cast<int>(blockTex.height),
                           static_cast<int>(blockTex.depth) };

          // Create per-block normal texture
          id<MTLTexture> blockNrm = EnsureTexture3D(device, block.NormalTexture,
            MTLPixelFormatRGBA8Unorm, bdims[0], bdims[1], bdims[2],
            MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, MTLStorageModePrivate);
          if (!blockNrm) continue;

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

          [compEnc setComputePipelineState:(__bridge id<MTLComputePipelineState>)this->NormalComputePipeline.get()];
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
// Updates per-block minmax textures and empty-block classification when only
// the opacity function changed (scalar voxel data is unchanged). Skips the
// expensive scalar texture re-upload that a full UpdateBlockTextures would do.
// Returns false if a full rebuild is required (e.g. a block transitioned from
// empty to non-empty and needs a new scalar texture).
bool vtkMetalGPUVolumeRayCastMapper::UpdateBlockMinMaxTextures(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkVolume* vol,
  vtkImageData* input, vtkDataArray* scalars, int numComponents)
{
  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mtlQueueVoid;

    int fullExt[6];
    input->GetExtent(fullExt);

    vtkVolumeProperty* property = vol ? vol->GetProperty() : nullptr;
    vtkPiecewiseFunction* opFunc = property ? property->GetScalarOpacity() : nullptr;
    const bool hasOpacityFunc = opFunc != nullptr;

    // Step 1: Re-evaluate empty-block classification.
    // Blocks that become empty release their textures; blocks that become
    // non-empty require a full rebuild (uncommon).
    for (size_t idx = 0; idx < this->Blocks.size(); ++idx)
    {
      auto& block = this->Blocks[idx];
      double blockMin = this->BlockScalarRanges[idx][0];
      double blockMax = this->BlockScalarRanges[idx][1];
      bool isEmpty = !hasOpacityFunc || this->IsBlockEmpty(blockMin, blockMax, opFunc);

      if (isEmpty)
      {
        if (block.Texture)
        {
          block.Texture.reset();
          block.Texture = nullptr;
        }
        block.MinMaxTexture.reset();
        block.MinMaxTexture = nullptr;
        block.NormalTexture.reset();
        block.NormalTexture = nullptr;
      }
      else if (!block.Texture)
      {
        return false; // empty→non-empty transition: needs full rebuild
      }
    }

    // Step 2: Regenerate per-block minmax textures for non-empty blocks.
    // Opacity table shared across all blocks.
    const double scalarRange = this->ScalarRange[1] - this->ScalarRange[0];
    const double rangeRecip = (scalarRange > 0.0) ? (255.0 / scalarRange) : 1.0;
    const double rangeOffset = this->ScalarRange[0];
    double opacityTable[256];
    if (hasOpacityFunc)
    {
      opFunc->GetTable(this->ScalarRange[0], this->ScalarRange[1], 256, opacityTable);
    }

    // GPU compute encoder and opacity prefix table (used only in GPU path)
    id<MTLComputeCommandEncoder> mmEnc = nil;
    id<MTLCommandBuffer> mmCmdBuf = nil;
    MinMaxComputeUniforms mmu = {};
    bool gpuMinMaxReady = false;
    float normFactor = this->ScalarNormalizationFactor;

    if (this->UseGPUMinMax && hasOpacityFunc &&
        this->EnsureMinMaxComputePipelines(mtlDeviceVoid))
    {
      mmCmdBuf = [queue commandBuffer];
      mmCmdBuf.label = @"VTK Block Opacity MinMax Compute";
      mmEnc = [mmCmdBuf computeCommandEncoder];
      mmEnc.label = @"VTK Block MinMax Compute (opacity update)";

      mmu.ds = 4.0f;
      mmu.scalarMin = static_cast<float>(this->ScalarRange[0] / normFactor);
      mmu.scalarScale = static_cast<float>(255.0 * normFactor / (scalarRange > 0.0 ? scalarRange : 1.0));
      mmu._pad = 0.0f;
      mmu.opacityPrefix[0] = 0;
      for (int i = 0; i < 256; ++i)
        mmu.opacityPrefix[i + 1] = mmu.opacityPrefix[i] + (opacityTable[i] > 0.0 ? 1u : 0u);
      gpuMinMaxReady = true;
    }

    const bool haveGlobalMinMax = !this->MacrocellScalarMin.empty() && !this->MacrocellScalarMax.empty();
    const int mcDims0 = this->MinMaxDims[0];
    const int mcDims1 = this->MinMaxDims[1];
    const int mcDims2 = this->MinMaxDims[2];

    for (auto& block : this->Blocks)
    {
      if (!block.Texture) continue;

      const int bDims[3] = { block.Dims[0], block.Dims[1], block.Dims[2] };
      const int texExt[6] = {
        block.Extents[0], block.Extents[1],
        block.Extents[2], block.Extents[3],
        block.Extents[4], block.Extents[5]
      };
      const int DS = 4;
      int mmDims[3] = {
        std::max(1, (bDims[0] + DS - 1) / DS),
        std::max(1, (bDims[1] + DS - 1) / DS),
        std::max(1, (bDims[2] + DS - 1) / DS)
      };
      block.MinMaxDims[0] = mmDims[0];
      block.MinMaxDims[1] = mmDims[1];
      block.MinMaxDims[2] = mmDims[2];

      if (this->UseGPUMinMax)
      {
        // GPU path: release old minmax texture; compute kernel will create new one
        block.MinMaxTexture.reset();
        block.MinMaxTexture = nullptr;
        continue; // dispatched below in the compute encoder
      }

      if (!hasOpacityFunc)
      {
        block.MinMaxTexture = nullptr;
        continue;
      }

      // --- CPU minmax regeneration ---
      vtkIdType numCells = static_cast<vtkIdType>(mmDims[0]) * mmDims[1] * mmDims[2];

      // Step 2a: Build raw occupancy from global MacrocellScalarMin/Max
      std::vector<uint8_t> rawMinMax(numCells, 255);

      if (haveGlobalMinMax)
      {
        const float* mcMin = this->MacrocellScalarMin.data();
        const float* mcMax = this->MacrocellScalarMax.data();
        const int relX0 = texExt[0] - fullExt[0];
        const int relY0 = texExt[2] - fullExt[2];
        const int relZ0 = texExt[4] - fullExt[4];

        vtkSMPTools::For(0, numCells, [&](vtkIdType begin, vtkIdType end) {
          for (vtkIdType cellIdx = begin; cellIdx < end; ++cellIdx)
          {
            const int gx = static_cast<int>(cellIdx % mmDims[0]);
            const int gy = static_cast<int>((cellIdx / mmDims[0]) % mmDims[1]);
            const int gz = static_cast<int>(cellIdx / (mmDims[0] * mmDims[1]));

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
                  vtkIdType mcIdx = (static_cast<vtkIdType>(mz) * mcDims1 + my) * mcDims0 + mx;
                  if (mcMin[mcIdx] < cellMin) cellMin = mcMin[mcIdx];
                  if (mcMax[mcIdx] > cellMax) cellMax = mcMax[mcIdx];
                }
              }
            }

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
      }
      else
      {
        // Fallback: walk every voxel in this block
        int dataType = scalars->GetDataType();
        const void* fullDataPtr = scalars->GetVoidPointer(0);
        vtkIdType inc[3];
        input->GetIncrements(inc);

        vtkSMPTools::For(0, numCells, [&](vtkIdType begin, vtkIdType end) {
          for (vtkIdType cellIdx = begin; cellIdx < end; ++cellIdx)
          {
            const int gx = static_cast<int>(cellIdx % mmDims[0]);
            const int gy = static_cast<int>((cellIdx / mmDims[0]) % mmDims[1]);
            const int gz = static_cast<int>(cellIdx / (mmDims[0] * mmDims[1]));

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
                    case VTK_INT:
                      v = static_cast<float>(
                        static_cast<const int*>(fullDataPtr)[zOffset * inc[2] + yOffset * inc[1] + xOffset * inc[0]]);
                      break;
                    case VTK_UNSIGNED_INT:
                      v = static_cast<float>(
                        static_cast<const unsigned int*>(fullDataPtr)[zOffset * inc[2] + yOffset * inc[1] + xOffset * inc[0]]);
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
      }

      // Step 2b: Dilation
      std::vector<uint8_t> minMaxData(numCells, 255);
      vtkSMPTools::For(0, numCells, [&](vtkIdType begin, vtkIdType end) {
        for (vtkIdType cellIdx = begin; cellIdx < end; ++cellIdx)
        {
          const int gx = static_cast<int>(cellIdx % mmDims[0]);
          const int gy = static_cast<int>((cellIdx / mmDims[0]) % mmDims[1]);
          const int gz = static_cast<int>(cellIdx / (mmDims[0] * mmDims[1]));

          int z0 = std::max(0, gz - 1), z1 = std::min(mmDims[2] - 1, gz + 1);
          int y0 = std::max(0, gy - 1), y1 = std::min(mmDims[1] - 1, gy + 1);
          int x0 = std::max(0, gx - 1), x1 = std::min(mmDims[0] - 1, gx + 1);

          bool solid = false;
          for (int nz = z0; nz <= z1 && !solid; ++nz)
            for (int ny = y0; ny <= y1 && !solid; ++ny)
              for (int nx = x0; nx <= x1 && !solid; ++nx)
                if (rawMinMax[(nz * mmDims[1] + ny) * mmDims[0] + nx] == 0) solid = true;

          minMaxData[cellIdx] = solid ? 0 : 255;
        }
      });

      // Step 2c: Create and upload R8Unorm texture
      id<MTLTexture> mmTex = EnsureTexture3D(device, block.MinMaxTexture,
        MTLPixelFormatR8Unorm, mmDims[0], mmDims[1], mmDims[2],
        MTLTextureUsageShaderRead, MTLStorageModeShared);
      if (mmTex)
      {
        MTLRegion region = MTLRegionMake3D(0, 0, 0,
          static_cast<NSUInteger>(mmDims[0]),
          static_cast<NSUInteger>(mmDims[1]),
          static_cast<NSUInteger>(mmDims[2]));
        NSUInteger mmBytesPerRow = static_cast<NSUInteger>(mmDims[0]) * sizeof(uint8_t);
        NSUInteger mmBytesPerImage = mmBytesPerRow * static_cast<NSUInteger>(mmDims[1]);
        [mmTex replaceRegion:region
                mipmapLevel:0
                      slice:0
                  withBytes:minMaxData.data()
                bytesPerRow:mmBytesPerRow
              bytesPerImage:mmBytesPerImage];
      }
    }

    // Step 3: GPU per-block minmax compute (if UseGPUMinMax)
    if (gpuMinMaxReady && mmEnc)
    {
      for (auto& block : this->Blocks)
      {
        if (!block.Texture) continue;

        id<MTLTexture> blockTex = (__bridge id<MTLTexture>)block.Texture.get();
        int bdims[3] = { block.Dims[0], block.Dims[1], block.Dims[2] };
        this->DispatchBlockMinMaxGPU(mtlDeviceVoid, (__bridge void*)mmEnc, (__bridge void*)blockTex,
          block, bdims, normFactor, scalarRange, opacityTable);
      }
      [mmEnc endEncoding];
      [mmCmdBuf commit];
    }
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

  if (this->ClippingPlanes->GetNumberOfItems() >= 8)
  {
    vtkWarningMacro("More than 8 clipping planes provided; extras ignored.");
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
        UniformBuffers[i].take((__bridge void*)buf);
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
          VertexBuffer.reset();
        }
        if (this->IndexBuffer)
        {
          IndexBuffer.reset();
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
          VertexBuffer.take((__bridge void*)vbuf);
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
          IndexBuffer.take((__bridge void*)ibuf);
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
          VertexBuffer.reset();
        }
        if (this->IndexBuffer)
        {
          IndexBuffer.reset();
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
          VertexBuffer.take((__bridge void*)vbuf);
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
          IndexBuffer.take((__bridge void*)ibuf);
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
    PipelineState.reset();
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
    id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary.get();

    auto ensureDummy3D = [&](vtkMetalResource& slot, MTLPixelFormat fmt, const void* data) {
      if (slot) return;
      id<MTLTexture> tex = EnsureTexture3D(device, slot, fmt, 1, 1, 1,
        MTLTextureUsageShaderRead, MTLStorageModeShared);
      if (tex)
      {
        MTLRegion region = MTLRegionMake3D(0, 0, 0, 1, 1, 1);
        [tex replaceRegion:region mipmapLevel:0 slice:0 withBytes:data
               bytesPerRow:sizeof(float) bytesPerImage:sizeof(float)];
      }
    };

    // Create dummy depth texture (1x1 R32Float with value 1.0) for when no real depth texture is bound
    if (!this->DummyDepthTexture)
    {
      id<MTLTexture> dummyTex = EnsureTexture2D(device, this->DummyDepthTexture,
        MTLPixelFormatR32Float, 1, 1,
        MTLTextureUsageShaderRead, MTLStorageModeShared);
      if (dummyTex)
      {
        float one = 1.0f;
        MTLRegion region = MTLRegionMake2D(0, 0, 1, 1);
        [dummyTex replaceRegion:region mipmapLevel:0 withBytes:&one bytesPerRow:sizeof(float)];
      }
    }

    // Note: constexpr samplers in MetalShaders.metal replace all runtime
    // sampler state objects. No sampler creation needed here.

    // Create dummy 3D textures for fallback bindings (prevent nil texture binds).
    {
      float zero = 0.0f;
      ensureDummy3D(this->DummyVolumeTexture, MTLPixelFormatR32Float, &zero);
      ensureDummy3D(this->DummyMaskTexture, MTLPixelFormatR32Float, &zero);
    }
    {
      uint8_t zero = 0;
      ensureDummy3D(this->DummyMinMaxTexture, MTLPixelFormatR8Unorm, &zero);
    }

    // Create and cache a depth stencil state.
    if (!this->DepthStencilState)
    {
      MTLDepthStencilDescriptor* dsDesc = [[MTLDepthStencilDescriptor alloc] init];
      dsDesc.depthCompareFunction = MTLCompareFunctionLessEqual;
      dsDesc.depthWriteEnabled = NO;
      id<MTLDepthStencilState> ds = [device newDepthStencilStateWithDescriptor:dsDesc];
      [dsDesc release];
      DepthStencilState.take((__bridge void*)ds);
    }

    // Create volume rendering pipelines via the caching helper.
    // Base pipelines (featureMask=0) are created here for backward compat;
    // specialized pipelines with non-zero feature masks are created on demand
    // in GPURender() via GetOrCreateVolumePipeline.
    // GetOrCreateVolumePipeline returns a non-owning pointer (cache owns +1);
    // use AssignRetainedMetalObject to give each member its own +1.
    void* pso = this->GetOrCreateVolumePipeline(mtlDeviceVoid,
      static_cast<uint32_t>(VolumePipelineType::DirectScreen),
      MTLPixelFormatBGRA8Unorm, MTLPixelFormatDepth32Float,
      static_cast<uint32_t>(sampleCount), 0);
    if (!pso)
    {
      return false;
    }
    PipelineState.retain((__bridge void*)(__bridge id)pso);
    this->CurrentSampleCount = sampleCount;

    void* accumPso = this->GetOrCreateVolumePipeline(mtlDeviceVoid,
      static_cast<uint32_t>(VolumePipelineType::OffscreenAccumulation),
      MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1, 0);
    if (!accumPso)
    {
      return false;
    }
    AccumulationPipelineState.retain((__bridge void*)(__bridge id)accumPso);

    void* layerPso = this->GetOrCreateVolumePipeline(mtlDeviceVoid,
      static_cast<uint32_t>(VolumePipelineType::OffscreenLayer),
      MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1, 0);
    if (!layerPso)
    {
      return false;
    }
    LayerPipelineState.retain((__bridge void*)(__bridge id)layerPso);

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
      // The +1 from new() goes to the cache.  AssignRetainedMetalObject
      // adds a separate +1 for the member slot.
      {
        VolumePipelineKey k = { static_cast<uint32_t>(VolumePipelineType::LayerComposite),
          MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1, 0 };
        this->PipelineCache[k] = (__bridge void*)p;
      }
      CompositePipelineState.retain((__bridge void*)p);
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
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary.get();
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
    case VolumePipelineType::FullscreenAccumulation:
      fragName = @"fragment_volume_fullscreen_accum_main";
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
    pt == VolumePipelineType::FullscreenOffscreen ||
    pt == VolumePipelineType::FullscreenAccumulation);

  if (hasFeatureConstants)
  {
    MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];

    BOOL shading = (featureMask & VolumeFeature_Shading) ? YES : NO;
    BOOL gradOp  = (featureMask & VolumeFeature_GradientOpacity) ? YES : NO;
    BOOL mask    = (featureMask & VolumeFeature_Mask) ? YES : NO;
    BOOL minmax  = (featureMask & VolumeFeature_MinMax) ? YES : NO;
    BOOL normalTex = (featureMask & VolumeFeature_NormalTexture) ? YES : NO;
    BOOL preInteg = (featureMask & VolumeFeature_PreIntegratedTF) ? YES : NO;

    [constants setConstantValue:&shading type:MTLDataTypeBool withName:@"fc_shading"];
    [constants setConstantValue:&gradOp  type:MTLDataTypeBool withName:@"fc_gradientOpacity"];
    [constants setConstantValue:&mask    type:MTLDataTypeBool withName:@"fc_mask"];
    [constants setConstantValue:&minmax  type:MTLDataTypeBool withName:@"fc_minmax"];
    [constants setConstantValue:&normalTex type:MTLDataTypeBool withName:@"fc_normalTexture"];
    [constants setConstantValue:&preInteg type:MTLDataTypeBool withName:@"fc_preIntegratedTF"];

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

  // Cache and return.
  // The +1 from new() is owned by the cache.
  // The caller receives a non-owning handle; use AssignRetainedMetalObject
  // when storing into a member slot.
  this->PipelineCache[key] = (__bridge void*)pso;
  return (__bridge void*)pso;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::DispatchBlockMinMaxGPU(void* deviceVoid, void* mmEncVoid,
  void* blockTexVoid, VolumeBlock& block,
  const int bdims[3], const float normFactor, const double scalarRange,
  const double opacityTable[256])
{
  id<MTLDevice> device = (__bridge id<MTLDevice>)deviceVoid;
  id<MTLComputeCommandEncoder> mmEnc = (__bridge id<MTLComputeCommandEncoder>)mmEncVoid;
  id<MTLTexture> blockTex = (__bridge id<MTLTexture>)blockTexVoid;

  int mmDims[3] = { block.MinMaxDims[0], block.MinMaxDims[1], block.MinMaxDims[2] };
  if (mmDims[0] <= 0) mmDims[0] = 1;
  if (mmDims[1] <= 0) mmDims[1] = 1;
  if (mmDims[2] <= 0) mmDims[2] = 1;
  block.MinMaxDims[0] = mmDims[0]; block.MinMaxDims[1] = mmDims[1]; block.MinMaxDims[2] = mmDims[2];

  // Create scratch R8Unorm texture for raw occupancy (temporary)
  MTLTextureDescriptor* scratchDesc = [[MTLTextureDescriptor alloc] init];
  scratchDesc.textureType = MTLTextureType3D;
  scratchDesc.pixelFormat = MTLPixelFormatR8Unorm;
  scratchDesc.width = static_cast<NSUInteger>(mmDims[0]);
  scratchDesc.height = static_cast<NSUInteger>(mmDims[1]);
  scratchDesc.depth = static_cast<NSUInteger>(mmDims[2]);
  scratchDesc.mipmapLevelCount = 1;
  scratchDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  scratchDesc.storageMode = MTLStorageModePrivate;

  id<MTLTexture> scratchTex = [device newTextureWithDescriptor:scratchDesc];
  [scratchDesc release];
  if (!scratchTex) return;

  // Create persistent per-block MinMax texture (dilated result)
  id<MTLTexture> mmTex = EnsureTexture3D(device, block.MinMaxTexture,
    MTLPixelFormatR8Unorm, mmDims[0], mmDims[1], mmDims[2],
    MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, MTLStorageModePrivate);
  if (!mmTex) { [scratchTex release]; return; }

  // Setup uniforms
  MinMaxComputeUniforms mmu;
  mmu.ds = 4.0f;
  mmu.scalarMin = static_cast<float>(this->ScalarRange[0] / normFactor);
  mmu.scalarScale = static_cast<float>(255.0 * normFactor / (scalarRange > 0.0 ? scalarRange : 1.0));
  mmu._pad = 0.0f;
  mmu.opacityPrefix[0] = 0;
  for (int i = 0; i < 256; ++i)
    mmu.opacityPrefix[i + 1] = mmu.opacityPrefix[i] + (opacityTable[i] > 0.0 ? 1u : 0u);
  mmu.mmDimX = static_cast<uint32_t>(mmDims[0]);
  mmu.mmDimY = static_cast<uint32_t>(mmDims[1]);
  mmu.mmDimZ = static_cast<uint32_t>(mmDims[2]);
  mmu.volDimX = static_cast<uint32_t>(bdims[0]);
  mmu.volDimY = static_cast<uint32_t>(bdims[1]);
  mmu.volDimZ = static_cast<uint32_t>(bdims[2]);

  // Dispatch volume_compute_minmax: blockTex -> scratchTex
  [mmEnc setComputePipelineState:(__bridge id<MTLComputePipelineState>)this->MinMaxComputePipeline.get()];
  [mmEnc setTexture:blockTex atIndex:0];
  [mmEnc setTexture:scratchTex atIndex:1];
  [mmEnc setBytes:&mmu length:sizeof(mmu) atIndex:0];

  MTLSize gridSize = MTLSizeMake(
    static_cast<NSUInteger>(mmDims[0]),
    static_cast<NSUInteger>(mmDims[1]),
    static_cast<NSUInteger>(mmDims[2]));
  NSUInteger tgw = 8;
  MTLSize tgSize = MTLSizeMake(
    std::min(tgw, static_cast<NSUInteger>(mmDims[0])),
    std::min(tgw, static_cast<NSUInteger>(mmDims[1])),
    std::min(tgw, static_cast<NSUInteger>(mmDims[2])));
  [mmEnc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

  // Dispatch volume_dilate_minmax: scratchTex -> mmTex
  [mmEnc setComputePipelineState:(__bridge id<MTLComputePipelineState>)this->DilateComputePipeline.get()];
  [mmEnc setTexture:scratchTex atIndex:0];
  [mmEnc setTexture:mmTex atIndex:1];
  [mmEnc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

  [scratchTex release];
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::BindFragmentTextures(
  void* encoderVoid, void* volTex, void* minMaxTex, void* normalTex)
{
  id<MTLRenderCommandEncoder> enc = (__bridge id<MTLRenderCommandEncoder>)encoderVoid;
  id<MTLTexture> tfTex = (__bridge id<MTLTexture>)this->ColorOpacityTexture.get();
  id<MTLTexture> depthTex = this->DepthTextureOcclusion
    ? (__bridge id<MTLTexture>)this->DepthTextureOcclusion.get()
    : (__bridge id<MTLTexture>)this->DummyDepthTexture.get();
  id<MTLTexture> gradOpTex = this->GradientOpacityTexture
    ? (__bridge id<MTLTexture>)this->GradientOpacityTexture.get() : tfTex;
  id<MTLTexture> maskTex = this->MaskTexture
    ? (__bridge id<MTLTexture>)this->MaskTexture.get()
    : (__bridge id<MTLTexture>)this->DummyMaskTexture.get();
  id<MTLTexture> labelTfTex = this->LabelMapTransferTexture
    ? (__bridge id<MTLTexture>)this->LabelMapTransferTexture.get() : tfTex;
  id<MTLTexture> mmTex = minMaxTex
    ? (__bridge id<MTLTexture>)minMaxTex
    : (this->MinMaxTexture
      ? (__bridge id<MTLTexture>)this->MinMaxTexture.get()
      : (__bridge id<MTLTexture>)this->DummyMinMaxTexture.get());
  id<MTLTexture> nrmTex = normalTex
    ? (__bridge id<MTLTexture>)normalTex
    : (this->GradientNormalTexture
      ? (__bridge id<MTLTexture>)this->GradientNormalTexture.get()
      : (volTex ? (__bridge id<MTLTexture>)volTex : (__bridge id<MTLTexture>)this->DummyVolumeTexture.get()));

  id<MTLTexture> vol = volTex
    ? (__bridge id<MTLTexture>)volTex
    : (this->VolumeTexture
      ? (__bridge id<MTLTexture>)this->VolumeTexture.get()
      : (__bridge id<MTLTexture>)this->DummyVolumeTexture.get());

  id<MTLTexture> piTex = this->PreIntegratedTFTexture
    ? (__bridge id<MTLTexture>)this->PreIntegratedTFTexture.get()
    : tfTex; // fallback: 1D TF (will be sampled as 2D with linear, gated by usePreIntegratedTF)

  id<MTLTexture> textures[9] = { vol, tfTex, depthTex, gradOpTex, maskTex, labelTfTex, mmTex, nrmTex, piTex };
  for (NSUInteger i = 0; i < 9; ++i)
    [enc setFragmentTexture:textures[i] atIndex:i];
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
    pipeline = (__bridge id<MTLRenderPipelineState>)this->PipelineState.get();
  }
  [encoder setRenderPipelineState:pipeline];
  [encoder setCullMode:MTLCullModeBack];

  // Only bind depth state if the pipeline uses depth testing.
  if (this->DepthStencilState && hasDepth)
  {
    id<MTLDepthStencilState> ds =
      (__bridge id<MTLDepthStencilState>)this->DepthStencilState.get();
    [encoder setDepthStencilState:ds];
  }

  // Bind buffers
  id<MTLBuffer> vertexBuf = (__bridge id<MTLBuffer>)this->VertexBuffer.get();
  [encoder setVertexBuffer:vertexBuf offset:0 atIndex:0];
  [encoder setVertexBuffer:uniformBuf offset:0 atIndex:1];
  [encoder setFragmentBuffer:uniformBuf offset:0 atIndex:1];

  this->BindFragmentTextures(encoder, this->VolumeTexture.get(), this->MinMaxTexture.get(), this->GradientNormalTexture.get());
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::BindFullscreenTextures(
  void* encoderVoid, void* uniformBufVoid,
  void* volTexVoid, void* minMaxTexVoid, void* normalTexVoid,
  bool useDepth, const void* pbd, uint32_t cullMode)
{
  id<MTLRenderCommandEncoder> encoder =
    (__bridge id<MTLRenderCommandEncoder>)encoderVoid;
  id<MTLBuffer> uniformBuf = (__bridge id<MTLBuffer>)uniformBufVoid;

  [encoder setCullMode:(MTLCullMode)cullMode];
  if (this->DepthStencilState && useDepth)
  {
    [encoder setDepthStencilState:(__bridge id<MTLDepthStencilState>)this->DepthStencilState.get()];
  }

  [encoder setVertexBytes:pbd length:sizeof(PerBlockData) atIndex:2];
  [encoder setFragmentBytes:pbd length:sizeof(PerBlockData) atIndex:2];
  [encoder setFragmentBuffer:uniformBuf offset:0 atIndex:1];

  this->BindFragmentTextures(encoder, volTexVoid, minMaxTexVoid, normalTexVoid);
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::BuildPerBlockData(PerBlockData& pbd,
  const VolumeBlock& block,
  const int fullExt[6], const double origin[3], const double spacing[3])
{
  pbd.VolumeBoundsMin[0] = static_cast<float>(block.BoundsMin[0]);
  pbd.VolumeBoundsMin[1] = static_cast<float>(block.BoundsMin[1]);
  pbd.VolumeBoundsMin[2] = static_cast<float>(block.BoundsMin[2]);
  pbd.VolumeBoundsMin[3] = 1.0f;

  pbd.VolumeBoundsMax[0] = static_cast<float>(block.BoundsMax[0]);
  pbd.VolumeBoundsMax[1] = static_cast<float>(block.BoundsMax[1]);
  pbd.VolumeBoundsMax[2] = static_cast<float>(block.BoundsMax[2]);
  pbd.VolumeBoundsMax[3] = 1.0f;

  int texExt[6] = {
    std::max(fullExt[0], static_cast<int>(block.Extents[0] - 1)),
    std::min(fullExt[1], static_cast<int>(block.Extents[1] + 1)),
    std::max(fullExt[2], static_cast<int>(block.Extents[2] - 1)),
    std::min(fullExt[3], static_cast<int>(block.Extents[3] + 1)),
    std::max(fullExt[4], static_cast<int>(block.Extents[4] - 1)),
    std::min(fullExt[5], static_cast<int>(block.Extents[5] + 1))
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
    pbd.MinMaxInfo[0] = 0.0f;
    pbd.MinMaxInfo[1] = 0.0f;
    pbd.MinMaxInfo[2] = 0.0f;
    pbd.MinMaxInfo[3] = 0.0f;
  }
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::BuildPerBlockData(PerBlockData& pbd,
  const VolumeMapperUniforms* uniforms)
{
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
  id<MTLBuffer> indexBuf = (__bridge id<MTLBuffer>)this->IndexBuffer.get();

  vtkImageData* input = vtkImageData::SafeDownCast(this->GetInput());
  int fullExt[6];
  input->GetExtent(fullExt);
  double origin[3], spacing[3];
  input->GetOrigin(origin);
  input->GetSpacing(spacing);

  if (!this->Blocks.empty())
  {
    // NOTE: the old single-draw INSTANCED path was removed. Partitioned volumes
    // with <= MAX_LAYER_BRICKS bricks are now composited order-independently in
    // GPURender (per-brick layer textures + per-pixel-sorted composite), which
    // makes draw order irrelevant. This function is therefore only reached for
    // the > MAX_LAYER_BRICKS fallback (order-dependent; see "Known limitations"
    // in GPURender) — SortBlocksBackToFront is called by the caller in that path.
    // Single-block volumes (Blocks.empty()) do not enter this branch.

    // --- FALLBACK PATH (> MAX_LAYER_BRICKS bricks): one draw per brick,
    //     composited front-to-back via framebuffer fetch. Order-dependent. ---
    for (size_t bi = 0; bi < this->SortedBlockOrder.size(); ++bi)
    {
      int si = this->SortedBlockOrder[bi];
      auto& block = this->Blocks[si];

      PerBlockData pbd;
      BuildPerBlockData(pbd, block, fullExt, origin, spacing);

      // Override per-block textures on top of the common textures set by BindEncoderResources
      id<MTLTexture> blockTex = (__bridge id<MTLTexture>)block.Texture.get();
      [encoder setFragmentTexture:blockTex atIndex:0];
      if (block.MinMaxTexture)
        [encoder setFragmentTexture:(__bridge id<MTLTexture>)block.MinMaxTexture.get() atIndex:6];
      else
        [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->DummyMinMaxTexture.get() atIndex:6];
      if (block.NormalTexture)
        [encoder setFragmentTexture:(__bridge id<MTLTexture>)block.NormalTexture.get() atIndex:7];
      else
        [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->DummyVolumeTexture.get() atIndex:7];

      [encoder setVertexBytes:&pbd length:sizeof(PerBlockData) atIndex:2];
      [encoder setFragmentBytes:&pbd length:sizeof(PerBlockData) atIndex:2];

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
    PerBlockData pbd = {};
    BuildPerBlockData(pbd, uniforms);

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

  // Handle partitioned volumes: render blocks in caller-established order.
  // SortBlocksBackToFront is called by the caller before this function when the
  // order-dependent (> MAX_LAYER_BRICKS) fallback is needed; the order-independent
  // layer composite path (<= MAX_LAYER_BRICKS) bypasses this function entirely.
  if (!this->Blocks.empty())
  {
    for (size_t bi = 0; bi < this->SortedBlockOrder.size(); ++bi)
    {
      int si = this->SortedBlockOrder[bi];
      auto& block = this->Blocks[si];

      VolumePipelineType pt = useDirectPipeline ? VolumePipelineType::FullscreenDirect
                                                : VolumePipelineType::FullscreenAccumulation;
      uint32_t fsType = static_cast<uint32_t>(pt);

      void* pso = this->GetOrCreateVolumePipeline(
        (__bridge void*)device, fsType, colorFormat, depthFormat, sampleCount, featureMask);
      if (!pso) continue;
      [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)pso];

      PerBlockData pbd;
      BuildPerBlockData(pbd, block, fullExt, origin, spacing);

      this->BindFullscreenTextures(encoder, uniformBuf,
        block.Texture, block.MinMaxTexture, block.NormalTexture,
        useDirectPipeline, &pbd, MTLCullModeBack);

      [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
  }
  else
  {
    void* pso = this->GetOrCreateVolumePipeline(
      (__bridge void*)device, pipelineType, colorFormat, depthFormat, sampleCount, featureMask);
    if (!pso) return;
    [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)pso];

    PerBlockData pbd = {};
    BuildPerBlockData(pbd, uniforms);

    this->BindFullscreenTextures(encoder, uniformBuf,
      this->VolumeTexture,
      this->MinMaxTexture,
      this->GradientNormalTexture,
      useDirectPipeline, &pbd, MTLCullModeBack);

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
      MinMaxTexture.reset();
      this->UpdateMinMaxTexture(mtlDevice, vol, input, scalars, false);
    }
  }
  else if (this->UseGPUMinMax && usePartitions)
  {
    // GPU min-max path for partitioned volumes: skip the expensive CPU macrocell
    // voxel scan. UpdateBlockTextures generates per-block minmax textures on GPU
    // directly from each block's uploaded texture, reusing volume_compute_minmax
    // and volume_dilate_minmax compute kernels with block-specific uniforms.
    //
    // Still run a lightweight CPU macrocell scan to populate MacrocellScalarMin/Max
    // so that UpdateBlockTextures (called inside UpdateVolumeTexture) can compute
    // per-block scalar ranges via macrocell reduction instead of an expensive
    // per-voxel walk that defeats the purpose of GPU acceleration.
    if (this->MacrocellScalarMin.empty() || this->MacrocellScalarMax.empty())
    {
      this->UpdateMinMaxTexture(mtlDevice, vol, input, scalars, true);
    }
    if (!this->UpdateVolumeTexture(mtlDevice, mtlQueue, vol))
    {
      return;
    }
  }
  else
  {
    // CPU min-max path: compute from raw scalar data before volume upload.
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

  if (!this->UpdateTransferFunctionTexture(mtlDevice, mtlQueue, vol))
  {
    return;
  }
  this->UpdateGradientOpacityTexture(mtlDevice, mtlQueue, vol);

  // Pre-integrated transfer function (Phase 1A)
  this->UpdatePreIntegratedTFTexture(mtlDevice, mtlQueue, vol);

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

  // --- Uniform caching ---
  // Start from cached values; only rebuild fields whose dependencies changed.
  VolumeMapperUniforms uniforms = this->CachedUniforms;

  vtkMTimeType volMTime = vol->GetMTime();

  // Lazy model matrix + inverse (cached to avoid redundant vtkMatrix4x4::Invert)
  vtkNew<vtkMatrix4x4> modelMatrix;
  vol->GetModelToWorldMatrix(modelMatrix);
  vtkNew<vtkMatrix4x4> invModelMatrix;

  if (this->CachedMatrixBuildTime < volMTime)
  {
    vtkMatrix4x4::Invert(modelMatrix, invModelMatrix);
    for (int c = 0; c < 4; ++c)
      for (int r = 0; r < 4; ++r)
      {
        int idx = c * 4 + r;
        this->CachedModelMatrixData[idx] = modelMatrix->GetElement(r, c);
        this->CachedInvModelMatrixData[idx] = invModelMatrix->GetElement(r, c);
      }
    this->CachedMatrixBuildTime.Modified();
  }
  else
  {
    for (int c = 0; c < 4; ++c)
      for (int r = 0; r < 4; ++r)
        invModelMatrix->SetElement(r, c, this->CachedInvModelMatrixData[c * 4 + r]);
  }

  vtkMTimeType propMTime = vol->GetProperty() ? vol->GetProperty()->GetMTime() : 0;
  vtkMTimeType inputMTime = input->GetMTime();
  vtkMTimeType selfMTime = this->GetMTime();

  bool rebuildUniforms =
    (this->UniformsBuildTime < volMTime) ||
    (this->UniformsBuildTime < propMTime) ||
    (this->UniformsBuildTime < inputMTime) ||
    (this->UniformsBuildTime < selfMTime) ||
    (this->UniformsBuildTime < this->VolumeUploadTime) ||
    (this->UniformsBuildTime < this->TransferFunctionUploadTime) ||
    (this->UniformsBuildTime < this->GradientOpacityUploadTime) ||
    (this->UniformsBuildTime < this->MaskUpdateTime);

  if (rebuildUniforms)
  {
    // ======= Model matrix uniforms =======
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
      {
        uniforms.VolumeToWorldMatrix[c * 4 + r] = modelMatrix->GetElement(r, c);
        uniforms.WorldToVolumeMatrix[c * 4 + r] = invModelMatrix->GetElement(r, c);
      }

    // ======= Volume bounds =======
    {
      double* mb = this->ModelBounds;
      uniforms.VolumeBoundsMin[0] = static_cast<float>(mb[0]);
      uniforms.VolumeBoundsMin[1] = static_cast<float>(mb[2]);
      uniforms.VolumeBoundsMin[2] = static_cast<float>(mb[4]);
      uniforms.VolumeBoundsMin[3] = 1.0f;
      uniforms.VolumeBoundsMax[0] = static_cast<float>(mb[1]);
      uniforms.VolumeBoundsMax[1] = static_cast<float>(mb[3]);
      uniforms.VolumeBoundsMax[2] = static_cast<float>(mb[5]);
      uniforms.VolumeBoundsMax[3] = 1.0f;
    }

    // ======= Scalar range (half-precision) =======
    {
      float normFactor = this->ScalarNormalizationFactor;
      uniforms.ScalarMinHalf = FloatToHalf(static_cast<float>(this->ScalarRange[0] / normFactor));
      uniforms.ScalarMaxHalf = FloatToHalf(static_cast<float>(
        (this->ScalarRange[1] > this->ScalarRange[0]
           ? this->ScalarRange[1]
           : this->ScalarRange[0] + 1.0) /
        normFactor));
    }

    // ======= Gradient-based shading (property-dependent) =======
    {
      vtkVolumeProperty* property = vol->GetProperty();
      bool shadeOn = property && property->GetShade();
      bool hasGradOp = property && property->HasGradientOpacity();

      uniforms.UseGradientShading = shadeOn ? 1.0f : 0.0f;
      uniforms.UseGradientOpacity = (shadeOn && hasGradOp) ? 1.0f : 0.0f;

      int dims[3];
      input->GetDimensions(dims);
      for (int k = 0; k < 3; ++k)
        uniforms.GradientStep[k] = (dims[k] > 0) ? 1.0f / dims[k] : 1.0f;

      double scalarRange = this->ScalarRange[1] - this->ScalarRange[0];
      if (scalarRange <= 0.0)
        scalarRange = 1.0;
      uniforms.GradientOpacityMin = 0.0f;
      uniforms.GradientOpacityMax = static_cast<float>(
        (scalarRange * 0.25) / this->ScalarNormalizationFactor);

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
    }

    // ======= UseJittering =======
    uniforms.UseJittering = this->GetUseJittering() ? 1.0f : 0.0f;

    // ======= Cropping =======
    if (this->GetCropping())
    {
      uniforms.UseCropping = 1.0f;
      double* mb = this->ModelBounds;
      double bs[3] = { mb[1] - mb[0], mb[3] - mb[2], mb[5] - mb[4] };
      for (int k = 0; k < 3; ++k)
        if (bs[k] < 1e-10) bs[k] = 1.0;

      double croppingRegionPlanes[6];
      this->GetCroppingRegionPlanes(croppingRegionPlanes);

      for (int i = 0; i < 3; ++i)
      {
        int minIdx = i * 2;
        int maxIdx = i * 2 + 1;
        croppingRegionPlanes[minIdx] =
          std::max(croppingRegionPlanes[minIdx], mb[minIdx]);
        croppingRegionPlanes[minIdx] =
          std::min(croppingRegionPlanes[minIdx], mb[maxIdx]);
        croppingRegionPlanes[maxIdx] =
          std::max(croppingRegionPlanes[maxIdx], mb[minIdx]);
        croppingRegionPlanes[maxIdx] =
          std::min(croppingRegionPlanes[maxIdx], mb[maxIdx]);
      }

      uniforms.CroppingPlanes[0] =
        static_cast<float>((croppingRegionPlanes[0] - mb[0]) / bs[0]);
      uniforms.CroppingPlanes[1] =
        static_cast<float>((croppingRegionPlanes[1] - mb[0]) / bs[0]);
      uniforms.CroppingPlanes[2] =
        static_cast<float>((croppingRegionPlanes[2] - mb[2]) / bs[1]);
      uniforms.CroppingPlanes[3] =
        static_cast<float>((croppingRegionPlanes[3] - mb[2]) / bs[1]);
      uniforms.CroppingPlanes2[0] =
        static_cast<float>((croppingRegionPlanes[4] - mb[4]) / bs[2]);
      uniforms.CroppingPlanes2[1] =
        static_cast<float>((croppingRegionPlanes[5] - mb[4]) / bs[2]);
      uniforms.CroppingPlanes2[2] = 0.0f;
      uniforms.CroppingPlanes2[3] = 0.0f;

      uniforms.CroppingBitmask = static_cast<uint32_t>(this->GetCroppingRegionFlags());
    }
    else
    {
      uniforms.UseCropping = 0.0f;
    }

    // ======= Clipping planes =======
    this->SetClippingPlaneUniforms(&uniforms, ren, vol, modelMatrix, invModelMatrix);

    // ======= Mask / label map =======
    this->SetMaskUniforms(&uniforms, vol);

    this->CachedUniforms = uniforms;
    this->UniformsBuildTime.Modified();
  }

  // ======= Always-updated fields (camera-dependent or per-frame) =======

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

  // Camera volume position (camera-dependent)
  {
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
  }

  double maxBoundsSize = std::max({ boundsSize[0], boundsSize[1], boundsSize[2] });

  // Sample distance (computed per-frame from ReductionFactor)
  uniforms.SampleDistanceHalf =
    FloatToHalf(static_cast<float>(actualSampleDistance / maxBoundsSize));

  // Opacity pre-integration factor (legacy 1D path)
  {
    vtkVolumeProperty* volProp = vol->GetProperty();
    double unitDist = volProp ? volProp->GetScalarOpacityUnitDistance(0) : 1.0;
    if (unitDist <= 0.0) unitDist = 1.0;
    uniforms.OpacityPreIntegrationFactorHalf =
      FloatToHalf(static_cast<float>(actualSampleDistance / unitDist));
  }

  // Pre-integrated TF (2D path)
  {
    uniforms.UsePreIntegratedTF = this->PreIntegratedTFTexture ? 1.0f : 0.0f;
    vtkVolumeProperty* volProp = vol->GetProperty();
    double unitDist = volProp ? volProp->GetScalarOpacityUnitDistance(0) : 1.0;
    if (unitDist <= 0.0) unitDist = 1.0;
    uniforms.PreIntegStepFactor =
      static_cast<float>(actualSampleDistance / unitDist);
  }

  // Light direction (camera-dependent)
  {
    double camDirWorld[3];
    ren->GetActiveCamera()->GetDirectionOfProjection(camDirWorld);
    double camDirLocal[4] = { camDirWorld[0], camDirWorld[1], camDirWorld[2], 0.0 };
    invModelMatrix->MultiplyPoint(camDirLocal, camDirLocal);
    camDirLocal[0] /= boundsSize[0];
    camDirLocal[1] /= boundsSize[1];
    camDirLocal[2] /= boundsSize[2];
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

  // Depth / MinMax / Normal texture flags (resource availability may change per frame)
  int sampleCount = metalRenderWindow ? metalRenderWindow->GetEffectiveSampleCount() : 1;
  this->DepthTextureOcclusion = (sampleCount > 1) ? nullptr : metalRenderWindow->GetDepthTexture();
  uniforms.UseDepthTexture = this->DepthTextureOcclusion ? 1.0f : 0.0f;

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
  if (this->PreIntegratedTFTexture && !(uniforms.UseMask > 0.5f))
    featureMask |= VolumeFeature_PreIntegratedTF;

  // Determine if image-space downsampling is active.
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
  vtkMetalCamera* metalCamera = vtkMetalCamera::SafeDownCast(ren->GetActiveCamera());
  if (metalCamera)
  {
    const float* sceneData = static_cast<const float*>(metalCamera->GetCachedSceneTransforms());
    const float* V = sceneData;
    const float* P = sceneData + 16;
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
    vtkCamera* cam = ren->GetActiveCamera();
    int* size = ren->GetSize();
    double aspect = (size[1] > 0) ? static_cast<double>(size[0]) / size[1] : 1.0;
    vtkMatrix4x4* V4 = cam->GetViewTransformMatrix();
    vtkMatrix4x4* P4 = cam->GetProjectionTransformMatrix(aspect, 0.0, 1.0);
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
  {
    simd_float4x4 vpMat;
    memcpy(&vpMat, uniforms.ViewProjectionMatrix, sizeof(vpMat));
    float det = simd::determinant(vpMat);
    if (fabs(det) > 1e-10f)
    {
      simd_float4x4 invVP = simd::inverse(vpMat);
      memcpy(uniforms.InverseViewProjection, &invVP, sizeof(invVP));
    }
    else
    {
      memset(uniforms.InverseViewProjection, 0, sizeof(uniforms.InverseViewProjection));
    }
  }

  // Wait for the uniform buffer slot for this frame to be free
  dispatch_semaphore_wait((dispatch_semaphore_t)this->FrameSemaphore.get(), DISPATCH_TIME_FOREVER);

  // RAII guard: signals the semaphore on scope exit (early return, exception, etc.).
  // Dismiss after the completion handler is installed below.
  struct SemaphoreSignalGuard {
    dispatch_semaphore_t sem;
    bool active = true;
    ~SemaphoreSignalGuard() { if (active && sem) dispatch_semaphore_signal(sem); }
    void dismiss() { active = false; }
  } semGuard{ (__bridge dispatch_semaphore_t)this->FrameSemaphore.get() };

  int bufIdx = this->UniformFrameIndex % 3;
  this->UniformFrameIndex++;

  // Update uniform buffer (now includes viewProjection + inverseViewProjection)
  id<MTLBuffer> uniformBuf = (__bridge id<MTLBuffer>)this->UniformBuffers[bufIdx].get();
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
      (__bridge id<MTLTexture>)this->ImageSampleColorTexture.get();

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
    // Count only non-empty blocks (those with a texture) — many blocks may be
    // skipped as empty space, so total block count is a poor gate for the
    // order-independent path.
    int nonEmptyCount = 0;
    for (auto& block : this->Blocks)
      if (block.Texture) ++nonEmptyCount;

    if (!this->Blocks.empty() && nonEmptyCount > 0 &&
        static_cast<size_t>(nonEmptyCount) <= MAX_LAYER_BRICKS &&
        this->LayerPipelineState && this->CompositePipelineState)
    {
      // Sort is decorative for the order-independent path (composite re-sorts per
      // pixel in the shader). Only collect non-empty blocks without sorting.
      this->SortedBlockOrder.clear();
      this->SortedBlockOrder.reserve(this->Blocks.size());
      for (size_t idx = 0; idx < this->Blocks.size(); ++idx)
      {
        if (this->Blocks[idx].Texture)
        {
          this->SortedBlockOrder.push_back(static_cast<int>(idx));
        }
      }
      int neededSlices = static_cast<int>(this->SortedBlockOrder.size());
      if (!this->EnsureLayerResources(mtlDevice, fboWidth, fboHeight, neededSlices))
      {
        return;
      }

      id<MTLBuffer> indexBuf = (__bridge id<MTLBuffer>)this->IndexBuffer.get();

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
        lrpd.colorAttachments[0].texture = (__bridge id<MTLTexture>)this->LayerTextureArray.get();
        lrpd.colorAttachments[0].slice = static_cast<NSUInteger>(bi);
        lrpd.colorAttachments[0].loadAction = MTLLoadActionClear;
        lrpd.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
        lrpd.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> layerEnc = [commandBuffer renderCommandEncoderWithDescriptor:lrpd];
        MTLViewport vp = {0, 0, (double)fboWidth, (double)fboHeight, 0.0, 1.0};
        [layerEnc setViewport:vp];

        PerBlockData pbd;
        BuildPerBlockData(pbd, block, fullExt, origin, spacing);

        if (cameraInside)
        {
          void* fsLayerPso = this->GetOrCreateVolumePipeline(mtlDevice,
            static_cast<uint32_t>(VolumePipelineType::FullscreenOffscreen),
            MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1, featureMask);
          [layerEnc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)fsLayerPso];
          this->BindFullscreenTextures(layerEnc, uniformBuf,
            block.Texture, block.MinMaxTexture, block.NormalTexture,
            false, &pbd, MTLCullModeNone);
          [layerEnc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }
        else
        {
          // Shared bindings (layer pipeline, vertex buf, uniforms, TF/depth/grad/mask/minmax)
          void* layerPso = this->GetOrCreateVolumePipeline(mtlDevice,
            static_cast<uint32_t>(VolumePipelineType::OffscreenLayer),
            MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1, featureMask);
          this->BindEncoderResources(layerEnc, uniformBuf, layerPso, false);
          // Override the global textures BindEncoderResources set with per-block textures
          [layerEnc setFragmentTexture:(__bridge id<MTLTexture>)block.Texture.get() atIndex:0];
          if (block.MinMaxTexture)
            [layerEnc setFragmentTexture:(__bridge id<MTLTexture>)block.MinMaxTexture.get() atIndex:6];
          else
            [layerEnc setFragmentTexture:(__bridge id<MTLTexture>)this->DummyMinMaxTexture.get() atIndex:6];
          // Override index 7 with per-block normal texture if available
          if (block.NormalTexture)
          {
            [layerEnc setFragmentTexture:(__bridge id<MTLTexture>)block.NormalTexture.get() atIndex:7];
          }

          [layerEnc setVertexBytes:&pbd length:sizeof(PerBlockData) atIndex:2];
          [layerEnc setFragmentBytes:&pbd length:sizeof(PerBlockData) atIndex:2];

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
      [compEnc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)this->CompositePipelineState.get()];
      [compEnc setCullMode:MTLCullModeNone];
      [compEnc setFragmentBuffer:uniformBuf offset:0 atIndex:1];
      [compEnc setFragmentBytes:&lc length:sizeof(lc) atIndex:2];

      [compEnc setFragmentTexture:(__bridge id<MTLTexture>)this->LayerTextureArray.get() atIndex:0];

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
      if (useAccumulation)
      {
        this->SortBlocksBackToFront(ren, vol);
      }
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
  dispatch_semaphore_t sem = (__bridge dispatch_semaphore_t)this->FrameSemaphore.get();
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
