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

#include <algorithm>
#include <cstring>
#include <set>
#include <vector>

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
  float CroppingFlagsRow0[4];       // 512..527
  float CroppingFlagsRow1[4];       // 528..543
  float CroppingFlagsRow2[4];       // 544..559
  float CroppingFlagsRow3[4];       // 560..575
  float CroppingFlagsRow4[4];       // 576..591
  float CroppingFlagsRow5[4];       // 592..607
  float CroppingFlagsRow6[4];       // 608..623
  float CroppingFlagsRow7[4];       // 624..639
  float UseCropping;                // 640
  float _padCropping[3];            // 644..655
  // Clipping planes (up to 8 arbitrary planes)
  float UseClipping;                // 656
  float NumClippingPlanes;          // 660
  float _padClipping[2];            // 664..671 (pad to 16-byte for float4)
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
  float _padMask[3];              // 948..959 (pad to 16-byte alignment)
};

static_assert(sizeof(VolumeMapperUniforms) == 960,
  "VolumeMapperUniforms must be 960 bytes to match Metal shader struct");

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
}

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalGPUVolumeRayCastMapper);

//------------------------------------------------------------------------------
vtkMetalGPUVolumeRayCastMapper::vtkMetalGPUVolumeRayCastMapper()
{
  this->SampleDistance = 1.0f;
}

//------------------------------------------------------------------------------
vtkMetalGPUVolumeRayCastMapper::~vtkMetalGPUVolumeRayCastMapper()
{
  this->ReleaseGraphicsResources(nullptr);
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
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
  if (this->ImageSampleColorTexture && this->ImageSampleFBOWidth == width &&
    this->ImageSampleFBOHeight == height)
  {
    return true;
  }

  this->ReleaseImageSampleResources();

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)deviceVoid;

    // Create offscreen color texture (BGRA8Unorm, matches layer pixel format)
    MTLTextureDescriptor* colorDesc = [[MTLTextureDescriptor alloc] init];
    colorDesc.textureType = MTLTextureType2D;
    colorDesc.pixelFormat = MTLPixelFormatBGRA8Unorm;
    colorDesc.width = width;
    colorDesc.height = height;
    colorDesc.mipmapLevelCount = 1;
    colorDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    colorDesc.storageMode = MTLStorageModePrivate;

    id<MTLTexture> colorTex = [device newTextureWithDescriptor:colorDesc];
    if (!colorTex)
    {
      vtkErrorMacro("Failed to create image-sample color texture");
      return false;
    }
    this->ImageSampleColorTexture = (__bridge void*)colorTex;
    CFRetain((__bridge CFTypeRef)colorTex);

    // Create offscreen depth texture
    MTLTextureDescriptor* depthDesc = [[MTLTextureDescriptor alloc] init];
    depthDesc.textureType = MTLTextureType2D;
    depthDesc.pixelFormat = MTLPixelFormatDepth32Float;
    depthDesc.width = width;
    depthDesc.height = height;
    depthDesc.mipmapLevelCount = 1;
    depthDesc.usage = MTLTextureUsageRenderTarget;
    depthDesc.storageMode = MTLStorageModePrivate;

    id<MTLTexture> depthTex = [device newTextureWithDescriptor:depthDesc];
    if (!depthTex)
    {
      vtkErrorMacro("Failed to create image-sample depth texture");
      this->ReleaseImageSampleResources();
      return false;
    }
    this->ImageSampleDepthTexture = (__bridge void*)depthTex;
    CFRetain((__bridge CFTypeRef)depthTex);

    // Create blit pipeline (fullscreen quad that samples the offscreen texture)
    NSError* error = nil;
    NSString* shaderSource = [NSString stringWithUTF8String:vtkMetalShaders];
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
    if (!library)
    {
      vtkErrorMacro(<< "Failed to compile Metal shader for image-sample blit: "
                    << [[error localizedDescription] UTF8String]);
      this->ReleaseImageSampleResources();
      return false;
    }

    id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_fullscreen_main"];
    id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragment_image_sample_blit"];
    if (!vertexFunc || !fragmentFunc)
    {
      vtkErrorMacro("Failed to find image-sample blit shader functions");
      this->ReleaseImageSampleResources();
      return false;
    }

    MTLRenderPipelineDescriptor* pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunc;
    pipelineDesc.fragmentFunction = fragmentFunc;
    pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pipelineDesc.colorAttachments[0].blendingEnabled = NO;
    pipelineDesc.rasterSampleCount = 1; // Always 1 for blit; MSAA is on the main pass

    id<MTLRenderPipelineState> pso =
      [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    if (!pso)
    {
      vtkErrorMacro(<< "Image-sample blit pipeline: " << [[error localizedDescription] UTF8String]);
      this->ReleaseImageSampleResources();
      return false;
    }
    this->ImageSamplePipeline = (__bridge void*)pso;
    CFRetain((__bridge CFTypeRef)pso);

    // Create linear sampler for blit
    MTLSamplerDescriptor* sDesc = [[MTLSamplerDescriptor alloc] init];
    sDesc.minFilter = MTLSamplerMinMagFilterLinear;
    sDesc.magFilter = MTLSamplerMinMagFilterLinear;
    sDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
    id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:sDesc];
    this->ImageSampleSampler = (__bridge void*)sampler;
    CFRetain((__bridge CFTypeRef)sampler);

    this->ImageSampleFBOWidth = width;
    this->ImageSampleFBOHeight = height;
  }

  return true;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ReleaseImageSampleResources()
{
  if (this->ImageSampleColorTexture)
  {
    CFRelease(this->ImageSampleColorTexture);
    this->ImageSampleColorTexture = nullptr;
  }
  if (this->ImageSampleDepthTexture)
  {
    CFRelease(this->ImageSampleDepthTexture);
    this->ImageSampleDepthTexture = nullptr;
  }
  if (this->ImageSamplePipeline)
  {
    CFRelease(this->ImageSamplePipeline);
    this->ImageSamplePipeline = nullptr;
  }
  if (this->ImageSampleSampler)
  {
    CFRelease(this->ImageSampleSampler);
    this->ImageSampleSampler = nullptr;
  }
  this->ImageSampleFBOWidth = 0;
  this->ImageSampleFBOHeight = 0;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ReleaseGraphicsResources(vtkWindow* vtkNotUsed(window))
{
  this->ReleaseImageSampleResources();
  this->ClearBlocks();

  if (this->PipelineState)
  {
    CFRelease(this->PipelineState);
    this->PipelineState = nullptr;
  }

  if (this->StagingBuffer)
  {
    CFRelease(this->StagingBuffer);
    this->StagingBuffer = nullptr;
  }

  if (this->VolumeTexture)
  {
    CFRelease(this->VolumeTexture);
    this->VolumeTexture = nullptr;
  }
  this->VolumeTextureView = nullptr;

  if (this->VolumeSampler)
  {
    CFRelease(this->VolumeSampler);
    this->VolumeSampler = nullptr;
  }

  if (this->ColorOpacityTexture)
  {
    CFRelease(this->ColorOpacityTexture);
    this->ColorOpacityTexture = nullptr;
  }
  this->ColorOpacityTextureView = nullptr;

  if (this->ColorOpacitySampler)
  {
    CFRelease(this->ColorOpacitySampler);
    this->ColorOpacitySampler = nullptr;
  }

  if (this->GradientOpacityTexture)
  {
    CFRelease(this->GradientOpacityTexture);
    this->GradientOpacityTexture = nullptr;
  }

  if (this->GradientOpacitySampler)
  {
    CFRelease(this->GradientOpacitySampler);
    this->GradientOpacitySampler = nullptr;
  }

  this->ReleaseMaskResources();

  if (this->DepthSampler)
  {
    CFRelease(this->DepthSampler);
    this->DepthSampler = nullptr;
  }

  if (this->DepthStencilState)
  {
    CFRelease(this->DepthStencilState);
    this->DepthStencilState = nullptr;
  }

  if (this->UniformBuffer)
  {
    CFRelease(this->UniformBuffer);
    this->UniformBuffer = nullptr;
  }

  if (this->VertexBuffer)
  {
    CFRelease(this->VertexBuffer);
    this->VertexBuffer = nullptr;
  }

  if (this->IndexBuffer)
  {
    CFRelease(this->IndexBuffer);
    this->IndexBuffer = nullptr;
  }
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
  if (usePartitions)
  {
    // Only reload if blocks don't exist yet or data has changed
    bool blockNeedsReload = this->Blocks.empty();
    blockNeedsReload |= (input->GetMTime() > this->VolumeUploadTime.GetMTime());
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
      int deltaX = (fullExt[1] - fullExt[0]) / nx;
      int deltaY = (fullExt[3] - fullExt[2]) / ny;
      int deltaZ = (fullExt[5] - fullExt[4]) / nz;

      for (int k = 0; k < nz; ++k)
      {
        for (int j = 0; j < ny; ++j)
        {
          for (int i = 0; i < nx; ++i)
          {
            VolumeBlock block;
            block.Extents[0] = fullExt[0] + i * deltaX;
            block.Extents[1] = fullExt[0] + (i + 1) * deltaX;
            block.Extents[2] = fullExt[2] + j * deltaY;
            block.Extents[3] = fullExt[2] + (j + 1) * deltaY;
            block.Extents[4] = fullExt[4] + k * deltaZ;
            block.Extents[5] = fullExt[4] + (k + 1) * deltaZ;
            this->Blocks.push_back(block);
          }
        }
      }

      // Store full volume bounds for vertex buffer (covers entire volume)
      double origin[3], spacing[3];
      input->GetOrigin(origin);
      input->GetSpacing(spacing);
      this->ModelBounds[0] = origin[0];
      this->ModelBounds[1] = origin[0] + spacing[0] * (input->GetDimensions()[0] - 1);
      this->ModelBounds[2] = origin[1];
      this->ModelBounds[3] = origin[1] + spacing[1] * (input->GetDimensions()[1] - 1);
      this->ModelBounds[4] = origin[2];
      this->ModelBounds[5] = origin[2] + spacing[2] * (input->GetDimensions()[2] - 1);

      this->VolumeNumComponents = scalars->GetNumberOfComponents();

      if (!this->UpdateBlockTextures(
            mtlDeviceVoid, mtlQueueVoid, input, scalars, this->VolumeNumComponents))
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

      this->VolumeNumComponents = numComponents;

      // Store model-space bounds
      double origin[3], spacing[3];
      input->GetOrigin(origin);
      input->GetSpacing(spacing);
      this->ModelBounds[0] = origin[0];
      this->ModelBounds[1] = origin[0] + spacing[0] * (dims[0] - 1);
      this->ModelBounds[2] = origin[1];
      this->ModelBounds[3] = origin[1] + spacing[1] * (dims[1] - 1);
      this->ModelBounds[4] = origin[2];
      this->ModelBounds[5] = origin[2] + spacing[2] * (dims[2] - 1);

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

      switch (dataType)
      {
        case VTK_FLOAT:
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
          break;
        }
      }

      this->ScalarNormalizationFactor = fmtInfo.normalizationFactor;

      const void* uploadPointer = nullptr;
      std::vector<uint16_t> halfData;
      std::vector<uint8_t> conversionBuffer;

      if (fmtInfo.needsConversion)
      {
        int outputComponents = (numComponents == 3) ? 4 : numComponents;
        halfData.resize(static_cast<size_t>(numTuples) * outputComponents);

        switch (dataType)
        {
          case VTK_FLOAT:
          {
            const float* src = static_cast<const float*>(scalars->GetVoidPointer(0));
            vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
              for (vtkIdType i = begin; i < end; ++i)
              {
                for (int c = 0; c < numComponents; ++c)
                  halfData[i * outputComponents + c] = FloatToHalf(src[i * numComponents + c]);
                for (int c = numComponents; c < outputComponents; ++c)
                  halfData[i * outputComponents + c] = FloatToHalf(0.0f);
              }
            });
            break;
          }
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
        if (this->VolumeTexture)
        {
          CFRelease(this->VolumeTexture);
          this->VolumeTexture = nullptr;
          this->VolumeTextureView = nullptr;
        }

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
        if (!tex)
        {
          vtkErrorMacro("Failed to create 3D volume texture");
          return false;
        }
        this->VolumeTexture = (__bridge void*)tex;
        CFRetain((__bridge CFTypeRef)tex);
        this->VolumeTextureView = this->VolumeTexture;
      }

      int actualComponents = (numComponents == 3) ? 4 : numComponents;
      NSUInteger bytesPerRow = static_cast<NSUInteger>(dims[0]) * fmtInfo.bytesPerComponent *
        actualComponents;
      NSUInteger bytesPerImage = bytesPerRow * dims[1];

      // Upload via staging buffer + blit encoder (works on all platforms)
      NSUInteger totalBytes = bytesPerImage * dims[2];

      // Release old staging buffer before creating a new one
      if (this->StagingBuffer)
      {
        CFRelease(this->StagingBuffer);
        this->StagingBuffer = nullptr;
      }

      id<MTLBuffer> stagingBuf = [device newBufferWithBytes:uploadPointer
                                                     length:totalBytes
                                                    options:MTLResourceStorageModeShared];
      if (!stagingBuf)
      {
        vtkErrorMacro("Failed to create volume staging buffer");
        return false;
      }
      this->StagingBuffer = (__bridge void*)stagingBuf;
      CFRetain((__bridge CFTypeRef)stagingBuf);

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
      // No waitUntilCompleted — the staging buffer is retained as a member
      // so it stays alive until the blit finishes on the GPU. Command buffers
      // on the same queue execute in order, so the render pass will not read
      // the texture until after this blit completes.

      this->VolumeUploadTime.Modified();
    }
  }

  return this->VolumeTexture != nullptr;
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

  bool doReload = (this->ColorOpacityTexture == nullptr);
  doReload |= (colorFunc->GetMTime() > this->TransferFunctionUploadTime.GetMTime());
  doReload |= (opacityFunc->GetMTime() > this->TransferFunctionUploadTime.GetMTime());

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
        if (this->ColorOpacityTexture)
        {
          CFRelease(this->ColorOpacityTexture);
          this->ColorOpacityTexture = nullptr;
          this->ColorOpacityTextureView = nullptr;
        }

        MTLTextureDescriptor* tfDesc = [[MTLTextureDescriptor alloc] init];
        tfDesc.textureType = MTLTextureType2D;
        tfDesc.pixelFormat = MTLPixelFormatRGBA8Unorm;
        tfDesc.width = 256;
        tfDesc.height = 1;
        tfDesc.mipmapLevelCount = 1;
        tfDesc.usage = MTLTextureUsageShaderRead;
        tfDesc.storageMode = MTLStorageModeShared;

        tex = [device newTextureWithDescriptor:tfDesc];
        if (!tex)
        {
          vtkErrorMacro("Failed to create transfer function texture");
          return false;
        }
        this->ColorOpacityTexture = (__bridge void*)tex;
        CFRetain((__bridge CFTypeRef)tex);
        this->ColorOpacityTextureView = this->ColorOpacityTexture;
      }

      MTLRegion region = MTLRegionMake2D(0, 0, 256, 1);
      [tex replaceRegion:region
            mipmapLevel:0
              withBytes:tfData
            bytesPerRow:256 * 4];

      this->TransferFunctionUploadTime.Modified();
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
        if (this->GradientOpacityTexture)
        {
          CFRelease(this->GradientOpacityTexture);
          this->GradientOpacityTexture = nullptr;
        }

        MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
        desc.textureType = MTLTextureType2D;
        desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
        desc.width = 256;
        desc.height = 1;
        desc.mipmapLevelCount = 1;
        desc.usage = MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared;

        tex = [device newTextureWithDescriptor:desc];
        if (!tex)
        {
          vtkErrorMacro("Failed to create gradient opacity texture");
          return false;
        }
        this->GradientOpacityTexture = (__bridge void*)tex;
        CFRetain((__bridge CFTypeRef)tex);
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

      // Convert mask data to float for the 3D texture
      // Mask is typically unsigned char (0-255) for label maps
      vtkIdType numTuples = arr->GetNumberOfTuples();
      std::vector<float> maskData(numTuples * numComponents);

      for (vtkIdType i = 0; i < numTuples; ++i)
      {
        for (int c = 0; c < numComponents; ++c)
        {
          double val = arr->GetComponent(i, c);
          maskData[i * numComponents + c] = static_cast<float>(val);
        }
      }

      // Create or update the 3D mask texture
      id<MTLTexture> oldTex = (__bridge id<MTLTexture>)this->MaskTexture;
      id<MTLTexture> tex = nil;

      if (oldTex)
      {
        tex = oldTex;
      }
      else
      {
        if (this->MaskTexture)
        {
          CFRelease(this->MaskTexture);
          this->MaskTexture = nullptr;
        }

        MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
        desc.textureType = MTLTextureType3D;
        desc.pixelFormat = MTLPixelFormatR32Float;
        desc.width = dims[0];
        desc.height = dims[1];
        desc.depth = dims[2];
        desc.mipmapLevelCount = 1;
        desc.usage = MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared;

        tex = [device newTextureWithDescriptor:desc];
        if (!tex)
        {
          vtkErrorMacro("Failed to create mask texture");
          return false;
        }
        this->MaskTexture = (__bridge void*)tex;
        CFRetain((__bridge CFTypeRef)tex);
      }

      // Upload mask data to texture
      MTLRegion region = MTLRegionMake3D(0, 0, 0, dims[0], dims[1], dims[2]);
      NSUInteger maskBytesPerRow = static_cast<NSUInteger>(dims[0]) * numComponents * sizeof(float);
      NSUInteger maskBytesPerImage = maskBytesPerRow * dims[1];
      [tex replaceRegion:region
            mipmapLevel:0
                  slice:0
              withBytes:maskData.data()
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

  bool doReload = (this->LabelMapTransferTexture == nullptr);
  doReload |= (latestMTime > this->MaskUpdateTime.GetMTime());

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

      // Build the 2D transfer function texture data (RGBA float)
      std::vector<float> tfData(tfWidth * tfHeight * 4);

      // Row 0: label 0 (default) - zeros (will be filled by default TF)
      std::fill(tfData.begin(), tfData.begin() + tfWidth * 4, 0.0f);

      // Rows 1..maxLabel: per-label transfer functions
      for (int label = 1; label < numLabels; ++label)
      {
        float* rowPtr = tfData.data() + label * tfWidth * 4;

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
            rowPtr[i * 4 + 0] = static_cast<float>(colorTable[i * 3 + 0]);
            rowPtr[i * 4 + 1] = static_cast<float>(colorTable[i * 3 + 1]);
            rowPtr[i * 4 + 2] = static_cast<float>(colorTable[i * 3 + 2]);
          }
        }

        if (opacityFunc)
        {
          std::vector<double> opacityTable(tfWidth);
          opacityFunc->GetTable(scalarRange[0], scalarRange[1], tfWidth, opacityTable.data());
          for (int i = 0; i < tfWidth; ++i)
          {
            rowPtr[i * 4 + 3] = static_cast<float>(opacityTable[i]);
          }
        }
      }

      // Create or update the 2D texture
      id<MTLTexture> oldTex = (__bridge id<MTLTexture>)this->LabelMapTransferTexture;
      id<MTLTexture> tex = nil;

      if (oldTex)
      {
        tex = oldTex;
      }
      else
      {
        if (this->LabelMapTransferTexture)
        {
          CFRelease(this->LabelMapTransferTexture);
          this->LabelMapTransferTexture = nullptr;
        }

        MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
        desc.textureType = MTLTextureType2D;
        desc.pixelFormat = MTLPixelFormatRGBA32Float;
        desc.width = tfWidth;
        desc.height = tfHeight;
        desc.mipmapLevelCount = 1;
        desc.usage = MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared;

        tex = [device newTextureWithDescriptor:desc];
        if (!tex)
        {
          vtkErrorMacro("Failed to create label map transfer texture");
          return false;
        }
        this->LabelMapTransferTexture = (__bridge void*)tex;
        CFRetain((__bridge CFTypeRef)tex);
      }

      // Upload data to texture
      MTLRegion region = MTLRegionMake2D(0, 0, tfWidth, tfHeight);
      [tex replaceRegion:region
            mipmapLevel:0
              withBytes:tfData.data()
            bytesPerRow:tfWidth * 4 * sizeof(float)];

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
    std::set<int> labels = property->GetLabelMapLabels();
    int maxLabel = labels.empty() ? 0 : *(labels.rbegin());
    u->LabelMapNumLabels = static_cast<float>(maxLabel);
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
  if (this->MaskTexture)
  {
    CFRelease(this->MaskTexture);
    this->MaskTexture = nullptr;
  }
  if (this->MaskSampler)
  {
    CFRelease(this->MaskSampler);
    this->MaskSampler = nullptr;
  }
  if (this->LabelMapTransferTexture)
  {
    CFRelease(this->LabelMapTransferTexture);
    this->LabelMapTransferTexture = nullptr;
  }
  if (this->LabelMapTransferSampler)
  {
    CFRelease(this->LabelMapTransferSampler);
    this->LabelMapTransferSampler = nullptr;
  }
  if (this->LabelMapGradientOpacityTexture)
  {
    CFRelease(this->LabelMapGradientOpacityTexture);
    this->LabelMapGradientOpacityTexture = nullptr;
  }
  if (this->LabelMapGradientOpacitySampler)
  {
    CFRelease(this->LabelMapGradientOpacitySampler);
    this->LabelMapGradientOpacitySampler = nullptr;
  }
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ClearBlocks()
{
  for (auto& block : this->Blocks)
  {
    if (block.Texture)
    {
      CFRelease(block.Texture);
      block.Texture = nullptr;
    }
  }
  this->Blocks.clear();
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::SortBlocksBackToFront(
  vtkRenderer* ren, vtkVolume* vol)
{
  if (this->Blocks.size() <= 1)
  {
    return;
  }

  vtkNew<vtkMatrix4x4> modelToWorld;
  vol->GetModelToWorldMatrix(modelToWorld);

  double* camPos = ren->GetActiveCamera()->GetPosition();
  double* camDir = ren->GetActiveCamera()->GetDirectionOfProjection();

  // Compute world-space center and distance for each block
  for (size_t i = 0; i < this->Blocks.size(); ++i)
  {
    auto& block = this->Blocks[i];
    double centerModel[4] = {
      (block.BoundsMin[0] + block.BoundsMax[0]) * 0.5,
      (block.BoundsMin[1] + block.BoundsMax[1]) * 0.5,
      (block.BoundsMin[2] + block.BoundsMax[2]) * 0.5,
      1.0
    };
    double centerWorld[4];
    modelToWorld->MultiplyPoint(centerModel, centerWorld);
    block.Center[0] = centerWorld[0];
    block.Center[1] = centerWorld[1];
    block.Center[2] = centerWorld[2];
  }

  // Initialize sorted order
  for (size_t i = 0; i < this->Blocks.size(); ++i)
  {
    this->SortedBlockOrder[i] = static_cast<int>(i);
  }

  // Sort by distance to camera (farthest first = back-to-front)
  std::sort(this->SortedBlockOrder, this->SortedBlockOrder + this->Blocks.size(),
    [&](int a, int b) {
      double da = (this->Blocks[a].Center[0] - camPos[0]) * camDir[0] +
        (this->Blocks[a].Center[1] - camPos[1]) * camDir[1] +
        (this->Blocks[a].Center[2] - camPos[2]) * camDir[2];
      double db = (this->Blocks[b].Center[0] - camPos[0]) * camDir[0] +
        (this->Blocks[b].Center[1] - camPos[1]) * camDir[1] +
        (this->Blocks[b].Center[2] - camPos[2]) * camDir[2];
      return da > db; // farthest first
    });
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::UpdateBlockTextures(void* mtlDeviceVoid,
  void* mtlQueueVoid, vtkImageData* input, vtkDataArray* scalars, int numComponents)
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

    int dataType = scalars->GetDataType();
    int componentsForFormat = (numComponents == 3) ? 4 : numComponents;

    // Determine pixel format (same logic as single-texture path)
    MTLPixelFormat pixelFormat;
    int bytesPerComponent = 2;
    float normalizationFactor = 1.0f;

    switch (dataType)
    {
      case VTK_FLOAT:
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
        break;
    }

    this->ScalarNormalizationFactor = normalizationFactor;

    double origin[3], spacing[3];
    input->GetOrigin(origin);
    input->GetSpacing(spacing);

    vtkIdType totalTuples = scalars->GetNumberOfTuples();
    const void* fullDataPtr = scalars->GetVoidPointer(0);

    // For non-native types, we need to convert first. Build a conversion buffer.
    std::vector<uint16_t> halfData;
    std::vector<uint8_t> conversionBuffer;
    const void* nativeDataPtr = fullDataPtr;

    bool needsConversion = (dataType != VTK_FLOAT && dataType != VTK_UNSIGNED_CHAR &&
      dataType != VTK_UNSIGNED_SHORT);

    if (needsConversion)
    {
      int outputComponents = (numComponents == 3) ? 4 : numComponents;
      halfData.resize(static_cast<size_t>(totalTuples) * outputComponents);
      switch (dataType)
      {
        case VTK_SHORT:
        {
          const short* src = static_cast<const short*>(fullDataPtr);
          vtkSMPTools::For(0, totalTuples, [&](vtkIdType begin, vtkIdType end) {
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
          const int* src = static_cast<const int*>(fullDataPtr);
          vtkSMPTools::For(0, totalTuples, [&](vtkIdType begin, vtkIdType end) {
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
          const double* src = static_cast<const double*>(fullDataPtr);
          vtkSMPTools::For(0, totalTuples, [&](vtkIdType begin, vtkIdType end) {
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
        default:
        {
          vtkSMPTools::For(0, totalTuples, [&](vtkIdType begin, vtkIdType end) {
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
      nativeDataPtr = halfData.data();
    }
    else if (dataType == VTK_FLOAT && numComponents == 3)
    {
      const float* src = static_cast<const float*>(fullDataPtr);
      conversionBuffer.resize(static_cast<size_t>(totalTuples) * 4 * sizeof(float));
      float* dst = reinterpret_cast<float*>(conversionBuffer.data());
      vtkSMPTools::For(0, totalTuples, [&](vtkIdType begin, vtkIdType end) {
        for (vtkIdType i = begin; i < end; ++i)
        {
          dst[i * 4 + 0] = src[i * 3 + 0];
          dst[i * 4 + 1] = src[i * 3 + 1];
          dst[i * 4 + 2] = src[i * 3 + 2];
          dst[i * 4 + 3] = 0.0f;
        }
      });
      nativeDataPtr = conversionBuffer.data();
    }
    else if (dataType == VTK_UNSIGNED_CHAR && numComponents == 3)
    {
      const unsigned char* src = static_cast<const unsigned char*>(fullDataPtr);
      conversionBuffer.resize(static_cast<size_t>(totalTuples) * 4);
      vtkSMPTools::For(0, totalTuples, [&](vtkIdType begin, vtkIdType end) {
        for (vtkIdType i = begin; i < end; ++i)
        {
          conversionBuffer[i * 4 + 0] = src[i * 3 + 0];
          conversionBuffer[i * 4 + 1] = src[i * 3 + 1];
          conversionBuffer[i * 4 + 2] = src[i * 3 + 2];
          conversionBuffer[i * 4 + 3] = 255;
        }
      });
      nativeDataPtr = conversionBuffer.data();
    }
    else if (dataType == VTK_UNSIGNED_SHORT && numComponents == 3)
    {
      const unsigned short* src = static_cast<const unsigned short*>(fullDataPtr);
      conversionBuffer.resize(static_cast<size_t>(totalTuples) * 4 * 2);
      unsigned short* dst = reinterpret_cast<unsigned short*>(conversionBuffer.data());
      vtkSMPTools::For(0, totalTuples, [&](vtkIdType begin, vtkIdType end) {
        for (vtkIdType i = begin; i < end; ++i)
        {
          dst[i * 4 + 0] = src[i * 3 + 0];
          dst[i * 4 + 1] = src[i * 3 + 1];
          dst[i * 4 + 2] = src[i * 3 + 2];
          dst[i * 4 + 3] = 65535;
        }
      });
      nativeDataPtr = conversionBuffer.data();
    }

    int actualComponents = (numComponents == 3) ? 4 : numComponents;
    size_t bytesPerVoxel = static_cast<size_t>(bytesPerComponent) * actualComponents;

    // Release old per-block textures
    this->ClearBlocks();

    // Create a 3D texture for each block
    for (size_t idx = 0; idx < this->Blocks.size(); ++idx)
    {
      auto& block = this->Blocks[idx];
      int bDims[3] = {
        block.Extents[1] - block.Extents[0] + 1,
        block.Extents[3] - block.Extents[2] + 1,
        block.Extents[5] - block.Extents[4] + 1
      };
      block.Dims[0] = bDims[0];
      block.Dims[1] = bDims[1];
      block.Dims[2] = bDims[2];

      // Compute model-space bounds for this block
      block.BoundsMin[0] = origin[0] + block.Extents[0] * spacing[0];
      block.BoundsMax[0] = origin[0] + block.Extents[1] * spacing[0];
      block.BoundsMin[1] = origin[1] + block.Extents[2] * spacing[1];
      block.BoundsMax[1] = origin[1] + block.Extents[3] * spacing[1];
      block.BoundsMin[2] = origin[2] + block.Extents[4] * spacing[2];
      block.BoundsMax[2] = origin[2] + block.Extents[5] * spacing[2];

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
      if (!tex)
      {
        vtkErrorMacro(<< "Failed to create block " << idx << " 3D texture ("
                      << bDims[0] << "x" << bDims[1] << "x" << bDims[2] << ")");
        return false;
      }
      block.Texture = (__bridge void*)tex;
      CFRetain((__bridge CFTypeRef)tex);

      // Copy block data from the full volume array
      NSUInteger blockBytesPerRow = static_cast<NSUInteger>(bDims[0]) * bytesPerVoxel;
      NSUInteger blockBytesPerImage = blockBytesPerRow * bDims[1];
      NSUInteger blockTotalBytes = blockBytesPerImage * bDims[2];

      // Build a staging buffer with the block's sub-region data
      // We need to extract the sub-volume from the full array with strided access
      std::vector<uint8_t> blockData(static_cast<size_t>(blockTotalBytes));

      for (int k = 0; k < bDims[2]; ++k)
      {
        for (int j = 0; j < bDims[1]; ++j)
        {
          // Source index in the full volume: (ext0+k) * fullDims[1] * fullDims[0] + (ext2+j) * fullDims[0] + ext0
          vtkIdType srcTuple =
            static_cast<vtkIdType>(block.Extents[4] + k) * fullDims[1] * fullDims[0] +
            static_cast<vtkIdType>(block.Extents[2] + j) * fullDims[0] +
            block.Extents[0];
          size_t dstOffset =
            static_cast<size_t>(k) * blockBytesPerImage + static_cast<size_t>(j) * blockBytesPerRow;

          const uint8_t* srcPtr =
            static_cast<const uint8_t*>(nativeDataPtr) + srcTuple * bytesPerVoxel;
          std::memcpy(blockData.data() + dstOffset, srcPtr, bDims[0] * bytesPerVoxel);
        }
      }

      id<MTLBuffer> blockStaging = [device newBufferWithBytes:blockData.data()
                                                       length:blockTotalBytes
                                                      options:MTLResourceStorageModeShared];
      if (!blockStaging)
      {
        vtkErrorMacro(<< "Failed to create staging buffer for block " << idx);
        return false;
      }

      id<MTLCommandBuffer> uploadCmdBuf = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [uploadCmdBuf blitCommandEncoder];
      [blit copyFromBuffer:blockStaging
              sourceOffset:0
       sourceBytesPerRow:blockBytesPerRow
     sourceBytesPerImage:blockBytesPerImage
              sourceSize:MTLSizeMake(bDims[0], bDims[1], bDims[2])
               toTexture:tex
        destinationSlice:0
        destinationLevel:0
       destinationOrigin:MTLOriginMake(0, 0, 0)];
      [blit endEncoding];
      [uploadCmdBuf commit];
      // Staging buffer is stack-local; block.Texture is retained so the GPU can read it
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
  void* uniformsVoid, vtkRenderer* ren, vtkVolume* vol)
{
  VolumeMapperUniforms* uniforms = static_cast<VolumeMapperUniforms*>(uniformsVoid);

  if (!this->GetClippingPlanes())
  {
    uniforms->UseClipping = 0.0f;
    uniforms->NumClippingPlanes = 0.0f;
    return;
  }

  uniforms->UseClipping = 1.0f;

  // Get volume model matrix for transforming planes to volume-local space
  vtkNew<vtkMatrix4x4> modelMatrix;
  vol->GetModelToWorldMatrix(modelMatrix);

  vtkNew<vtkMatrix4x4> invModelMatrix;
  vtkMatrix4x4::Invert(modelMatrix, invModelMatrix);

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
    // Get plane origin and normal in world coordinates
    double planeOrigin[3], planeNormal[3];
    plane->GetOrigin(planeOrigin);
    plane->GetNormal(planeNormal);

    // Transform origin to volume-local [0,1] space
    double originLocal[4] = { planeOrigin[0], planeOrigin[1], planeOrigin[2], 1.0 };
    invModelMatrix->MultiplyPoint(originLocal, originLocal);

    double originVol[3] = {
      (originLocal[0] - modelBounds[0]) / boundsSize[0],
      (originLocal[1] - modelBounds[2]) / boundsSize[1],
      (originLocal[2] - modelBounds[4]) / boundsSize[2]
    };

    // Transform normal to volume-local [0,1] space
    // Use the inverse transpose of the model matrix for normal transformation
    double normalLocal[4] = { planeNormal[0], planeNormal[1], planeNormal[2], 0.0 };
    invModelMatrix->MultiplyPoint(normalLocal, normalLocal);

    // Scale normal by bounds to account for non-uniform scaling in [0,1] space
    double normalVol[3] = {
      normalLocal[0] / boundsSize[0],
      normalLocal[1] / boundsSize[1],
      normalLocal[2] / boundsSize[2]
    };

    // Normalize
    double normalLen = sqrt(normalVol[0] * normalVol[0] + normalVol[1] * normalVol[1] +
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

    if (!this->UniformBuffer)
    {
      id<MTLBuffer> buf = [device newBufferWithLength:sizeof(VolumeMapperUniforms)
                                              options:MTLResourceStorageModeShared];
      if (!buf)
      {
        vtkErrorMacro("Failed to create uniform buffer");
        return false;
      }
      this->UniformBuffer = (__bridge void*)buf;
      CFRetain((__bridge CFTypeRef)buf);
    }

    // Use model-space bounds for vertex positions
    if (input)
    {
      int dims[3];
      double origin[3], spacing[3];
      input->GetDimensions(dims);
      input->GetOrigin(origin);
      input->GetSpacing(spacing);
      this->ModelBounds[0] = origin[0];
      this->ModelBounds[1] = origin[0] + spacing[0] * (dims[0] - 1);
      this->ModelBounds[2] = origin[1];
      this->ModelBounds[3] = origin[1] + spacing[1] * (dims[1] - 1);
      this->ModelBounds[4] = origin[2];
      this->ModelBounds[5] = origin[2] + spacing[2] * (dims[2] - 1);
    }

    // Check if camera is inside the volume
    bool cameraInside = this->IsCameraInside(ren, vol);

    // Check if geometry needs rebuild
    bool needsVertexRebuild = !this->VertexBuffer;
    needsVertexRebuild |= (this->VolumeUploadTime.GetMTime() > this->VertexBufferUploadTime.GetMTime());
    needsVertexRebuild |= (this->GetMTime() > this->VertexBufferUploadTime.GetMTime());
    needsVertexRebuild |= (cameraInside != this->CameraWasInsideInLastUpdate);

    if (needsVertexRebuild)
    {
      // Camera outside: simple 8-vertex box (original fast path)
      // Camera inside: clip against near plane, densify, triangulate
      if (!cameraInside)
      {
        float boundsMin[3] = {
          static_cast<float>(this->ModelBounds[0]),
          static_cast<float>(this->ModelBounds[2]),
          static_cast<float>(this->ModelBounds[4])
        };
        float boundsMax[3] = {
          static_cast<float>(this->ModelBounds[1]),
          static_cast<float>(this->ModelBounds[3]),
          static_cast<float>(this->ModelBounds[5])
        };

        float vertices[] = {
          boundsMin[0], boundsMin[1], boundsMin[2],
          boundsMax[0], boundsMin[1], boundsMin[2],
          boundsMax[0], boundsMax[1], boundsMin[2],
          boundsMin[0], boundsMax[1], boundsMin[2],
          boundsMin[0], boundsMin[1], boundsMax[2],
          boundsMax[0], boundsMin[1], boundsMax[2],
          boundsMax[0], boundsMax[1], boundsMax[2],
          boundsMin[0], boundsMax[1], boundsMax[2],
        };

        unsigned int indices[] = {
          0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 7, 3, 0, 4, 7, 1, 2, 6, 1, 6, 5,
          3, 6, 2, 3, 7, 6, 0, 1, 5, 0, 5, 4,
        };

        this->IndexCount = sizeof(indices) / sizeof(unsigned int);

        // Release old buffers
        if (this->VertexBuffer)
        {
          CFRelease(this->VertexBuffer);
          this->VertexBuffer = nullptr;
        }
        if (this->IndexBuffer)
        {
          CFRelease(this->IndexBuffer);
          this->IndexBuffer = nullptr;
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
          this->VertexBuffer = (__bridge void*)vbuf;
          CFRetain((__bridge CFTypeRef)vbuf);
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
          this->IndexBuffer = (__bridge void*)ibuf;
          CFRetain((__bridge CFTypeRef)ibuf);
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

        // Transform normal to volume coordinates (transpose-inverse)
        double* dmat = dataToWorld->GetData();
        dataToWorld->Transpose();
        double pNormalV[3];
        pNormalV[0] = pNormal[0] * dmat[0] + pNormal[1] * dmat[1] + pNormal[2] * dmat[2];
        pNormalV[1] = pNormal[0] * dmat[4] + pNormal[1] * dmat[5] + pNormal[2] * dmat[6];
        pNormalV[2] = pNormal[0] * dmat[8] + pNormal[1] * dmat[9] + pNormal[2] * dmat[10];
        vtkMath::Normalize(pNormalV);

        // Transform origin point to volume coordinates
        dataToWorld->Transpose();
        dataToWorld->Invert();
        dataToWorld->MultiplyPoint(pOrigin, pOrigin);

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

        // Convert to float array for Metal buffer
        std::vector<float> vertices;
        vertices.reserve(points->GetNumberOfPoints() * 3);

        for (vtkIdType i = 0; i < points->GetNumberOfPoints(); ++i)
        {
          double pt[3];
          points->GetPoint(i, pt);
          vertices.push_back(static_cast<float>(pt[0]));
          vertices.push_back(static_cast<float>(pt[1]));
          vertices.push_back(static_cast<float>(pt[2]));
        }

        std::vector<unsigned int> indices;
        vtkIdType npts;
        const vtkIdType* pts;

        polys->InitTraversal();
        while (polys->GetNextCell(npts, pts))
        {
          for (vtkIdType i = 0; i < npts; ++i)
          {
            indices.push_back(static_cast<unsigned int>(pts[i]));
          }
        }

        this->IndexCount = static_cast<int>(indices.size());

        // Release old buffers
        if (this->VertexBuffer)
        {
          CFRelease(this->VertexBuffer);
          this->VertexBuffer = nullptr;
        }
        if (this->IndexBuffer)
        {
          CFRelease(this->IndexBuffer);
          this->IndexBuffer = nullptr;
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
          this->VertexBuffer = (__bridge void*)vbuf;
          CFRetain((__bridge CFTypeRef)vbuf);
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
          this->IndexBuffer = (__bridge void*)ibuf;
          CFRetain((__bridge CFTypeRef)ibuf);
        }

        this->CameraWasInsideInLastUpdate = true;
      }

      this->VertexBufferUploadTime.Modified();
    }
  }

  return this->VertexBuffer && this->IndexBuffer && this->UniformBuffer;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::SetupPipeline(void* mtlDeviceVoid, vtkRenderer* ren)
{
  // Get sample count and invalidate PSO if it changed (e.g., MSAA toggled)
  auto* metalRenderWindow = vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());
  int sampleCount = metalRenderWindow ? metalRenderWindow->GetEffectiveSampleCount() : 1;

  if (this->PipelineState && sampleCount != this->CurrentSampleCount)
  {
    CFRelease(this->PipelineState);
    this->PipelineState = nullptr;
  }

  if (this->PipelineState)
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
      vtkErrorMacro(<< "Failed to compile Metal volume shader: "
                    << [[error localizedDescription] UTF8String]);
      return false;
    }

    id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_volume_main"];
    id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragment_volume_main"];
    if (!vertexFunc || !fragmentFunc)
    {
      vtkErrorMacro("Failed to find volume shader functions");
      return false;
    }

    // Create samplers
    if (!this->ColorOpacitySampler)
    {
      MTLSamplerDescriptor* samplerDesc = [[MTLSamplerDescriptor alloc] init];
      samplerDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
      samplerDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
      samplerDesc.magFilter = MTLSamplerMinMagFilterLinear;
      samplerDesc.minFilter = MTLSamplerMinMagFilterLinear;
      id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:samplerDesc];
      this->ColorOpacitySampler = (__bridge void*)sampler;
      CFRetain((__bridge CFTypeRef)sampler);
    }

    if (!this->VolumeSampler)
    {
      MTLSamplerDescriptor* samplerDesc = [[MTLSamplerDescriptor alloc] init];
      samplerDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
      samplerDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
      samplerDesc.rAddressMode = MTLSamplerAddressModeClampToEdge;
      samplerDesc.magFilter = MTLSamplerMinMagFilterLinear;
      samplerDesc.minFilter = MTLSamplerMinMagFilterLinear;
      samplerDesc.mipFilter = MTLSamplerMipFilterLinear;
      id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:samplerDesc];
      this->VolumeSampler = (__bridge void*)sampler;
      CFRetain((__bridge CFTypeRef)sampler);
    }

    // Nearest sampler for depth texture (depth buffer occlusion)
    if (!this->DepthSampler)
    {
      MTLSamplerDescriptor* depthSampDesc = [[MTLSamplerDescriptor alloc] init];
      depthSampDesc.minFilter = MTLSamplerMinMagFilterNearest;
      depthSampDesc.magFilter = MTLSamplerMinMagFilterNearest;
      depthSampDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
      depthSampDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
      id<MTLSamplerState> depthSamp = [device newSamplerStateWithDescriptor:depthSampDesc];
      this->DepthSampler = (__bridge void*)depthSamp;
      CFRetain((__bridge CFTypeRef)depthSamp);
    }

    // Linear sampler for gradient opacity texture (1D lookup)
    if (!this->GradientOpacitySampler)
    {
      MTLSamplerDescriptor* goDesc = [[MTLSamplerDescriptor alloc] init];
      goDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
      goDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
      goDesc.magFilter = MTLSamplerMinMagFilterLinear;
      goDesc.minFilter = MTLSamplerMinMagFilterLinear;
      id<MTLSamplerState> goSamp = [device newSamplerStateWithDescriptor:goDesc];
      this->GradientOpacitySampler = (__bridge void*)goSamp;
      CFRetain((__bridge CFTypeRef)goSamp);
    }

    // Nearest sampler for mask texture (3D, nearest interpolation for label maps)
    if (!this->MaskSampler)
    {
      MTLSamplerDescriptor* maskDesc = [[MTLSamplerDescriptor alloc] init];
      maskDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
      maskDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
      maskDesc.rAddressMode = MTLSamplerAddressModeClampToEdge;
      maskDesc.magFilter = MTLSamplerMinMagFilterNearest;
      maskDesc.minFilter = MTLSamplerMinMagFilterNearest;
      id<MTLSamplerState> maskSamp = [device newSamplerStateWithDescriptor:maskDesc];
      this->MaskSampler = (__bridge void*)maskSamp;
      CFRetain((__bridge CFTypeRef)maskSamp);
    }

    // Nearest sampler for label map transfer function texture (2D, nearest for label lookup)
    if (!this->LabelMapTransferSampler)
    {
      MTLSamplerDescriptor* lmDesc = [[MTLSamplerDescriptor alloc] init];
      lmDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
      lmDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
      lmDesc.magFilter = MTLSamplerMinMagFilterNearest;
      lmDesc.minFilter = MTLSamplerMinMagFilterNearest;
      id<MTLSamplerState> lmSamp = [device newSamplerStateWithDescriptor:lmDesc];
      this->LabelMapTransferSampler = (__bridge void*)lmSamp;
      CFRetain((__bridge CFTypeRef)lmSamp);
    }

    // Nearest sampler for label map gradient opacity texture
    if (!this->LabelMapGradientOpacitySampler)
    {
      MTLSamplerDescriptor* lgoDesc = [[MTLSamplerDescriptor alloc] init];
      lgoDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
      lgoDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
      lgoDesc.magFilter = MTLSamplerMinMagFilterNearest;
      lgoDesc.minFilter = MTLSamplerMinMagFilterNearest;
      id<MTLSamplerState> lgoSamp = [device newSamplerStateWithDescriptor:lgoDesc];
      this->LabelMapGradientOpacitySampler = (__bridge void*)lgoSamp;
      CFRetain((__bridge CFTypeRef)lgoSamp);
    }

    // Create and cache a depth stencil state.
    // Volume rendering reads but does not write depth — this prevents the
    // bounding box from z-fighting with itself and allows correct occlusion
    // by opaque geometry that wrote depth earlier in the render pass.
    if (!this->DepthStencilState)
    {
      MTLDepthStencilDescriptor* dsDesc = [[MTLDepthStencilDescriptor alloc] init];
      dsDesc.depthCompareFunction = MTLCompareFunctionLessEqual;
      dsDesc.depthWriteEnabled = NO;
      id<MTLDepthStencilState> ds = [device newDepthStencilStateWithDescriptor:dsDesc];
      this->DepthStencilState = (__bridge void*)ds;
      CFRetain((__bridge CFTypeRef)ds);
    }

    // Pipeline: vertex buffer layout — float3 position at buffer index 0
    MTLVertexDescriptor* vertexDesc = [[MTLVertexDescriptor alloc] init];
    vertexDesc.attributes[0].format = MTLVertexFormatFloat3;
    vertexDesc.attributes[0].offset = 0;
    vertexDesc.attributes[0].bufferIndex = 0;
    vertexDesc.layouts[0].stride = sizeof(float) * 3;
    vertexDesc.layouts[0].stepRate = 1;
    vertexDesc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;

    MTLRenderPipelineDescriptor* pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunc;
    pipelineDesc.fragmentFunction = fragmentFunc;
    pipelineDesc.vertexDescriptor = vertexDesc;
    pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

    // The raymarching shader accumulates premultiplied color (color * alpha)
    // into accumulatedColor. Using MTLBlendFactorOne as source avoids
    // double-multiplying by alpha again at the blend stage. This matches
    // the WebGPU volume mapper blend mode.
    pipelineDesc.colorAttachments[0].blendingEnabled = YES;
    pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
    pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor =
      MTLBlendFactorOneMinusSourceAlpha;
    pipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;

    pipelineDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    pipelineDesc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    pipelineDesc.rasterSampleCount = sampleCount;

    id<MTLRenderPipelineState> pso =
      [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    if (!pso)
    {
      vtkErrorMacro(<< "Volume pipeline: " << [[error localizedDescription] UTF8String]);
      return false;
    }
    this->PipelineState = (__bridge void*)pso;
    CFRetain((__bridge CFTypeRef)pso);
    this->CurrentSampleCount = sampleCount;
  }

  return true;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::GPURender(vtkRenderer* ren, vtkVolume* vol)
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
    scalars->GetRange(this->ScalarRange);
  }
  else
  {
    this->ScalarRange[0] = 0.0;
    this->ScalarRange[1] = 1.0;
  }

  if (!this->UpdateVolumeTexture(mtlDevice, mtlQueue, vol))
  {
    return;
  }
  if (!this->UpdateTransferFunctionTexture(mtlDevice, mtlQueue, vol))
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
  VolumeMapperUniforms uniforms;

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

  // Compute adaptive sample distance (matches OpenGL mapper logic)
  this->ComputeReductionFactor(vol->GetAllocatedRenderTime());

  double actualSampleDistance;
  if (this->AutoAdjustSampleDistances)
  {
    // Compute minimum world-space voxel spacing across all 3 axes
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
    // Lock sample distance to input spacing: adapts step size to voxel density
    // for optimal quality/performance balance. Computes 1/2 average spacing
    // and scales down for small volumes (< 100 voxels).
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

    // Use spacing-adjusted distance unless user explicitly set a custom value
    // (within 0.1% tolerance). This matches the OpenGL mapper logic.
    actualSampleDistance =
      (sample / d < 0.999f || sample / d > 1.001f) ? d : this->SampleDistance;
  }
  else
  {
    actualSampleDistance = this->GetSampleDistance();
  }

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

    // Gradient step: 1/(dims-1) per axis for central differences in [0,1] space
    int dims[3];
    input->GetDimensions(dims);
    for (int k = 0; k < 3; ++k)
    {
      uniforms.GradientStep[k] = (dims[k] > 1) ? 1.0f / (dims[k] - 1) : 1.0f;
    }

    // Gradient opacity normalization range
    double scalarRange = this->ScalarRange[1] - this->ScalarRange[0];
    if (scalarRange <= 0.0)
      scalarRange = 1.0;
    uniforms.GradientOpacityMin = 0.0f;
    uniforms.GradientOpacityMax = static_cast<float>(scalarRange * 0.25);

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

    // Convert from world coordinates to volume-local [0,1] space
    // using the same normalization as CameraVolumePos
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

    // Decode CroppingRegionFlags bitmask into 32-element array
    int cropFlags = this->GetCroppingRegionFlags();
    float flagsArray[32] = {};
    flagsArray[0] = 0; // region 0 is always 0
    for (int fi = 1; fi < 32; ++fi)
    {
      flagsArray[fi] = static_cast<float>(cropFlags & 1);
      cropFlags >>= 1;
    }

    // Pack into float4 rows
    for (int r = 0; r < 8; ++r)
    {
      float row[4] = { flagsArray[r * 4], flagsArray[r * 4 + 1], flagsArray[r * 4 + 2],
        flagsArray[r * 4 + 3] };
      memcpy(&uniforms.CroppingFlagsRow0 + r, row, sizeof(row));
    }
  }
  else
  {
    uniforms.UseCropping = 0.0f;
  }

  // Clipping planes
  this->SetClippingPlaneUniforms(&uniforms, ren, vol);

  // Mask / label map
  this->SetMaskUniforms(&uniforms, vol);

  // Viewport size for depth texture UV computation in the shader
  int* winSize = ren->GetSize();
  uniforms.ViewportSize[0] = static_cast<float>(winSize[0]);
  uniforms.ViewportSize[1] = static_cast<float>(winSize[1]);

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
  this->DepthTextureOcclusion = metalRenderWindow->GetDepthTexture();

  // Update uniform buffer (now includes viewProjection + inverseViewProjection)
  id<MTLBuffer> uniformBuf = (__bridge id<MTLBuffer>)this->UniformBuffer;
  memcpy([uniformBuf contents], &uniforms, sizeof(uniforms));

  // Determine if image-space downsampling is active
  const float imageSampleDist = this->ImageSampleDistance;
  const bool useImageSampling = (imageSampleDist != 1.0f);

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
    id<MTLTexture> offscreenDepth =
      (__bridge id<MTLTexture>)this->ImageSampleDepthTexture;

    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = offscreenColor;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.depthAttachment.texture = offscreenDepth;
    rpd.depthAttachment.loadAction = MTLLoadActionClear;
    rpd.depthAttachment.clearDepth = 1.0;
    rpd.depthAttachment.storeAction = MTLStoreActionDontCare;

    id<MTLRenderCommandEncoder> offscreenEncoder =
      [commandBuffer renderCommandEncoderWithDescriptor:rpd];
    offscreenEncoder.label = @"VTK Volume ImageSample Offscreen";

    // Set viewport to offscreen dimensions
    MTLViewport metalViewport;
    metalViewport.originX = 0;
    metalViewport.originY = 0;
    metalViewport.width = fboWidth;
    metalViewport.height = fboHeight;
    metalViewport.znear = 0.0;
    metalViewport.zfar = 1.0;
    [offscreenEncoder setViewport:metalViewport];

    // Set pipeline and render state
    id<MTLRenderPipelineState> pipeline =
      (__bridge id<MTLRenderPipelineState>)this->PipelineState;
    [offscreenEncoder setRenderPipelineState:pipeline];
    [offscreenEncoder setCullMode:MTLCullModeNone];

    if (this->DepthStencilState)
    {
      id<MTLDepthStencilState> ds =
        (__bridge id<MTLDepthStencilState>)this->DepthStencilState;
      [offscreenEncoder setDepthStencilState:ds];
    }

    // Bind buffers and textures
    id<MTLBuffer> vertexBuf = (__bridge id<MTLBuffer>)this->VertexBuffer;
    [offscreenEncoder setVertexBuffer:vertexBuf offset:0 atIndex:0];
    [offscreenEncoder setVertexBuffer:uniformBuf offset:0 atIndex:1];
    [offscreenEncoder setFragmentBuffer:uniformBuf offset:0 atIndex:1];

    id<MTLTexture> volTex = (__bridge id<MTLTexture>)this->VolumeTexture;
    id<MTLSamplerState> volSamp = (__bridge id<MTLSamplerState>)this->VolumeSampler;
    [offscreenEncoder setFragmentTexture:volTex atIndex:0];
    [offscreenEncoder setFragmentSamplerState:volSamp atIndex:0];

    id<MTLTexture> tfTex = (__bridge id<MTLTexture>)this->ColorOpacityTexture;
    id<MTLSamplerState> tfSamp = (__bridge id<MTLSamplerState>)this->ColorOpacitySampler;
    [offscreenEncoder setFragmentTexture:tfTex atIndex:1];
    [offscreenEncoder setFragmentSamplerState:tfSamp atIndex:1];

    // Bind scene depth texture for early ray termination (depth buffer occlusion)
    if (this->DepthTextureOcclusion)
    {
      id<MTLTexture> depthTex = (__bridge id<MTLTexture>)this->DepthTextureOcclusion;
      id<MTLSamplerState> depthSamp = (__bridge id<MTLSamplerState>)this->DepthSampler;
      [offscreenEncoder setFragmentTexture:depthTex atIndex:2];
      [offscreenEncoder setFragmentSamplerState:depthSamp atIndex:2];
    }

    // Bind gradient opacity texture for gradient-based shading
    if (this->GradientOpacityTexture)
    {
      id<MTLTexture> goTex = (__bridge id<MTLTexture>)this->GradientOpacityTexture;
      id<MTLSamplerState> goSamp = (__bridge id<MTLSamplerState>)this->GradientOpacitySampler;
      [offscreenEncoder setFragmentTexture:goTex atIndex:3];
      [offscreenEncoder setFragmentSamplerState:goSamp atIndex:3];
    }

    // Bind mask / label map textures.
    // Metal requires all declared fragment texture/sampler arguments to be bound,
    // so bind the volume texture as fallback when mask is not used.
    if (this->MaskTexture)
    {
      id<MTLTexture> maskTex = (__bridge id<MTLTexture>)this->MaskTexture;
      id<MTLSamplerState> maskSamp = (__bridge id<MTLSamplerState>)this->MaskSampler;
      [offscreenEncoder setFragmentTexture:maskTex atIndex:4];
      [offscreenEncoder setFragmentSamplerState:maskSamp atIndex:4];
    }
    else
    {
      [offscreenEncoder setFragmentTexture:volTex atIndex:4];
      [offscreenEncoder setFragmentSamplerState:volSamp atIndex:4];
    }

    if (this->LabelMapTransferTexture)
    {
      id<MTLTexture> lmTex = (__bridge id<MTLTexture>)this->LabelMapTransferTexture;
      id<MTLSamplerState> lmSamp = (__bridge id<MTLSamplerState>)this->LabelMapTransferSampler;
      [offscreenEncoder setFragmentTexture:lmTex atIndex:5];
      [offscreenEncoder setFragmentSamplerState:lmSamp atIndex:5];
    }
    else
    {
      [offscreenEncoder setFragmentTexture:tfTex atIndex:5];
      [offscreenEncoder setFragmentSamplerState:tfSamp atIndex:5];
    }

    if (this->LabelMapGradientOpacityTexture)
    {
      id<MTLTexture> lgoTex = (__bridge id<MTLTexture>)this->LabelMapGradientOpacityTexture;
      id<MTLSamplerState> lgoSamp = (__bridge id<MTLSamplerState>)this->LabelMapGradientOpacitySampler;
      [offscreenEncoder setFragmentTexture:lgoTex atIndex:6];
      [offscreenEncoder setFragmentSamplerState:lgoSamp atIndex:6];
    }
    else
    {
      [offscreenEncoder setFragmentTexture:tfTex atIndex:6];
      [offscreenEncoder setFragmentSamplerState:tfSamp atIndex:6];
    }

    // Draw volume — handle partitioned (multi-block) and single-block cases
    id<MTLBuffer> indexBuf = (__bridge id<MTLBuffer>)this->IndexBuffer;
    if (!this->Blocks.empty())
    {
      this->SortBlocksBackToFront(ren, vol);

      // Per-block rendering: update uniform bounds and bind block's texture
      VolumeMapperUniforms blockUniforms;
      memcpy(&blockUniforms, &uniforms, sizeof(blockUniforms));

      double* camPosWorld = ren->GetActiveCamera()->GetPosition();
      vtkNew<vtkMatrix4x4> invModelMatrix;
      vtkNew<vtkMatrix4x4> modelMatrix;
      vol->GetModelToWorldMatrix(modelMatrix);
      vtkMatrix4x4::Invert(modelMatrix, invModelMatrix);

      for (size_t bi = 0; bi < this->Blocks.size(); ++bi)
      {
        int si = this->SortedBlockOrder[bi];
        auto& block = this->Blocks[si];

        // Update bounds for this block
        blockUniforms.VolumeBoundsMin[0] = static_cast<float>(block.BoundsMin[0]);
        blockUniforms.VolumeBoundsMin[1] = static_cast<float>(block.BoundsMin[1]);
        blockUniforms.VolumeBoundsMin[2] = static_cast<float>(block.BoundsMin[2]);
        blockUniforms.VolumeBoundsMin[3] = 1.0f;

        blockUniforms.VolumeBoundsMax[0] = static_cast<float>(block.BoundsMax[0]);
        blockUniforms.VolumeBoundsMax[1] = static_cast<float>(block.BoundsMax[1]);
        blockUniforms.VolumeBoundsMax[2] = static_cast<float>(block.BoundsMax[2]);
        blockUniforms.VolumeBoundsMax[3] = 1.0f;

        // Recompute camera position in block-local [0,1] space
        double camPosVolume[4] = { camPosWorld[0], camPosWorld[1], camPosWorld[2], 1.0 };
        invModelMatrix->MultiplyPoint(camPosVolume, camPosVolume);
        double blockBoundsSize[3] = {
          block.BoundsMax[0] - block.BoundsMin[0],
          block.BoundsMax[1] - block.BoundsMin[1],
          block.BoundsMax[2] - block.BoundsMin[2]
        };
        for (int k = 0; k < 3; ++k)
        {
          if (blockBoundsSize[k] < 1e-10)
            blockBoundsSize[k] = 1.0;
        }
        blockUniforms.CameraVolumePos[0] =
          static_cast<float>((camPosVolume[0] - block.BoundsMin[0]) / blockBoundsSize[0]);
        blockUniforms.CameraVolumePos[1] =
          static_cast<float>((camPosVolume[1] - block.BoundsMin[1]) / blockBoundsSize[1]);
        blockUniforms.CameraVolumePos[2] =
          static_cast<float>((camPosVolume[2] - block.BoundsMin[2]) / blockBoundsSize[2]);
        blockUniforms.CameraVolumePos[3] = 1.0f;

        // Update gradient step for this block's dimensions
        for (int k = 0; k < 3; ++k)
        {
          blockUniforms.GradientStep[k] =
            (block.Dims[k] > 1) ? 1.0f / (block.Dims[k] - 1) : 1.0f;
        }

        // Update uniform buffer with block-specific bounds
        memcpy([uniformBuf contents], &blockUniforms, sizeof(blockUniforms));

        // Bind this block's 3D texture
        id<MTLTexture> blockTex = (__bridge id<MTLTexture>)block.Texture;
        [offscreenEncoder setFragmentTexture:blockTex atIndex:0];

        // Draw
        [offscreenEncoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                    indexCount:this->IndexCount
                                     indexType:MTLIndexTypeUInt32
                                   indexBuffer:indexBuf
                             indexBufferOffset:0];
      }

      // Restore original uniforms
      memcpy([uniformBuf contents], &uniforms, sizeof(uniforms));
    }
    else
    {
      // Single-block path (no partitioning)
      [offscreenEncoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                  indexCount:this->IndexCount
                                   indexType:MTLIndexTypeUInt32
                                 indexBuffer:indexBuf
                           indexBufferOffset:0];
    }

    [offscreenEncoder endEncoding];
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

    // Set pipeline and arguments
    id<MTLRenderPipelineState> pipeline =
      (__bridge id<MTLRenderPipelineState>)this->PipelineState;
    [encoder setRenderPipelineState:pipeline];

    // Disable face culling: volume bounding box is viewed from both inside and
    // outside depending on camera position, so both front and back faces must be
    // rendered as ray entry/exit points.
    [encoder setCullMode:MTLCullModeNone];

    // Apply cached depth-stencil state: read depth (LessEqual) but do not write
    // it. This lets opaque geometry occlude the volume correctly while preventing
    // the bounding-box triangles from z-fighting with each other.
    if (this->DepthStencilState)
    {
      id<MTLDepthStencilState> ds =
        (__bridge id<MTLDepthStencilState>)this->DepthStencilState;
      [encoder setDepthStencilState:ds];
    }

    // Bind vertex buffer (positions)
    id<MTLBuffer> vertexBuf = (__bridge id<MTLBuffer>)this->VertexBuffer;
    [encoder setVertexBuffer:vertexBuf offset:0 atIndex:0];

    // Bind uniform buffer (vertex + fragment) — now contains viewProjection
    [encoder setVertexBuffer:uniformBuf offset:0 atIndex:1];
    [encoder setFragmentBuffer:uniformBuf offset:0 atIndex:1];

    // Bind volume texture and sampler (fragment)
    id<MTLTexture> volTex = (__bridge id<MTLTexture>)this->VolumeTexture;
    id<MTLSamplerState> volSamp = (__bridge id<MTLSamplerState>)this->VolumeSampler;
    [encoder setFragmentTexture:volTex atIndex:0];
    [encoder setFragmentSamplerState:volSamp atIndex:0];

    // Bind transfer function texture and sampler (fragment)
    id<MTLTexture> tfTex = (__bridge id<MTLTexture>)this->ColorOpacityTexture;
    id<MTLSamplerState> tfSamp = (__bridge id<MTLSamplerState>)this->ColorOpacitySampler;
    [encoder setFragmentTexture:tfTex atIndex:1];
    [encoder setFragmentSamplerState:tfSamp atIndex:1];

    // Bind scene depth texture for early ray termination (depth buffer occlusion)
    if (this->DepthTextureOcclusion)
    {
      id<MTLTexture> depthTex = (__bridge id<MTLTexture>)this->DepthTextureOcclusion;
      id<MTLSamplerState> depthSamp = (__bridge id<MTLSamplerState>)this->DepthSampler;
      [encoder setFragmentTexture:depthTex atIndex:2];
      [encoder setFragmentSamplerState:depthSamp atIndex:2];
    }

    // Bind gradient opacity texture for gradient-based shading.
    // Metal requires all declared fragment texture/sampler arguments to be bound,
    // so bind the transfer function texture as fallback when gradient opacity is disabled.
    if (this->GradientOpacityTexture)
    {
      id<MTLTexture> goTex = (__bridge id<MTLTexture>)this->GradientOpacityTexture;
      id<MTLSamplerState> goSamp = (__bridge id<MTLSamplerState>)this->GradientOpacitySampler;
      [encoder setFragmentTexture:goTex atIndex:3];
      [encoder setFragmentSamplerState:goSamp atIndex:3];
    }
    else
    {
      [encoder setFragmentTexture:tfTex atIndex:3];
      [encoder setFragmentSamplerState:tfSamp atIndex:3];
    }

    // Bind mask / label map textures.
    // Metal requires all declared fragment texture/sampler arguments to be bound,
    // so bind the volume/TF textures as fallback when mask is not used.
    if (this->MaskTexture)
    {
      id<MTLTexture> maskTex = (__bridge id<MTLTexture>)this->MaskTexture;
      id<MTLSamplerState> maskSamp = (__bridge id<MTLSamplerState>)this->MaskSampler;
      [encoder setFragmentTexture:maskTex atIndex:4];
      [encoder setFragmentSamplerState:maskSamp atIndex:4];
    }
    else
    {
      [encoder setFragmentTexture:volTex atIndex:4];
      [encoder setFragmentSamplerState:volSamp atIndex:4];
    }

    if (this->LabelMapTransferTexture)
    {
      id<MTLTexture> lmTex = (__bridge id<MTLTexture>)this->LabelMapTransferTexture;
      id<MTLSamplerState> lmSamp = (__bridge id<MTLSamplerState>)this->LabelMapTransferSampler;
      [encoder setFragmentTexture:lmTex atIndex:5];
      [encoder setFragmentSamplerState:lmSamp atIndex:5];
    }
    else
    {
      [encoder setFragmentTexture:tfTex atIndex:5];
      [encoder setFragmentSamplerState:tfSamp atIndex:5];
    }

    if (this->LabelMapGradientOpacityTexture)
    {
      id<MTLTexture> lgoTex = (__bridge id<MTLTexture>)this->LabelMapGradientOpacityTexture;
      id<MTLSamplerState> lgoSamp = (__bridge id<MTLSamplerState>)this->LabelMapGradientOpacitySampler;
      [encoder setFragmentTexture:lgoTex atIndex:6];
      [encoder setFragmentSamplerState:lgoSamp atIndex:6];
    }
    else
    {
      [encoder setFragmentTexture:tfTex atIndex:6];
      [encoder setFragmentSamplerState:tfSamp atIndex:6];
    }

    // Draw — handle partitioned (multi-block) and single-block cases
    id<MTLBuffer> indexBuf = (__bridge id<MTLBuffer>)this->IndexBuffer;
    if (!this->Blocks.empty())
    {
      this->SortBlocksBackToFront(ren, vol);

      // Per-block rendering: update uniform bounds and bind block's texture
      VolumeMapperUniforms blockUniforms;
      memcpy(&blockUniforms, &uniforms, sizeof(blockUniforms));

      double* camPosWorld = ren->GetActiveCamera()->GetPosition();
      vtkNew<vtkMatrix4x4> invModelMatrix;
      vtkNew<vtkMatrix4x4> modelMatrix;
      vol->GetModelToWorldMatrix(modelMatrix);
      vtkMatrix4x4::Invert(modelMatrix, invModelMatrix);

      for (size_t bi = 0; bi < this->Blocks.size(); ++bi)
      {
        int si = this->SortedBlockOrder[bi];
        auto& block = this->Blocks[si];

        // Update bounds for this block
        blockUniforms.VolumeBoundsMin[0] = static_cast<float>(block.BoundsMin[0]);
        blockUniforms.VolumeBoundsMin[1] = static_cast<float>(block.BoundsMin[1]);
        blockUniforms.VolumeBoundsMin[2] = static_cast<float>(block.BoundsMin[2]);
        blockUniforms.VolumeBoundsMin[3] = 1.0f;

        blockUniforms.VolumeBoundsMax[0] = static_cast<float>(block.BoundsMax[0]);
        blockUniforms.VolumeBoundsMax[1] = static_cast<float>(block.BoundsMax[1]);
        blockUniforms.VolumeBoundsMax[2] = static_cast<float>(block.BoundsMax[2]);
        blockUniforms.VolumeBoundsMax[3] = 1.0f;

        // Recompute camera position in block-local [0,1] space
        double camPosVolume[4] = { camPosWorld[0], camPosWorld[1], camPosWorld[2], 1.0 };
        invModelMatrix->MultiplyPoint(camPosVolume, camPosVolume);
        double blockBoundsSize[3] = {
          block.BoundsMax[0] - block.BoundsMin[0],
          block.BoundsMax[1] - block.BoundsMin[1],
          block.BoundsMax[2] - block.BoundsMin[2]
        };
        for (int k = 0; k < 3; ++k)
        {
          if (blockBoundsSize[k] < 1e-10)
            blockBoundsSize[k] = 1.0;
        }
        blockUniforms.CameraVolumePos[0] =
          static_cast<float>((camPosVolume[0] - block.BoundsMin[0]) / blockBoundsSize[0]);
        blockUniforms.CameraVolumePos[1] =
          static_cast<float>((camPosVolume[1] - block.BoundsMin[1]) / blockBoundsSize[1]);
        blockUniforms.CameraVolumePos[2] =
          static_cast<float>((camPosVolume[2] - block.BoundsMin[2]) / blockBoundsSize[2]);
        blockUniforms.CameraVolumePos[3] = 1.0f;

        // Update gradient step for this block's dimensions
        for (int k = 0; k < 3; ++k)
        {
          blockUniforms.GradientStep[k] =
            (block.Dims[k] > 1) ? 1.0f / (block.Dims[k] - 1) : 1.0f;
        }

        // Update uniform buffer with block-specific bounds
        memcpy([uniformBuf contents], &blockUniforms, sizeof(blockUniforms));

        // Bind this block's 3D texture
        id<MTLTexture> blockTex = (__bridge id<MTLTexture>)block.Texture;
        [encoder setFragmentTexture:blockTex atIndex:0];

        // Draw
        [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                            indexCount:this->IndexCount
                             indexType:MTLIndexTypeUInt32
                           indexBuffer:indexBuf
                     indexBufferOffset:0];
      }

      // Restore original uniforms
      memcpy([uniformBuf contents], &uniforms, sizeof(uniforms));
    }
    else
    {
      // Single-block path (no partitioning)
      [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                          indexCount:this->IndexCount
                           indexType:MTLIndexTypeUInt32
                         indexBuffer:indexBuf
                   indexBufferOffset:0];
    }
  }
}

VTK_ABI_NAMESPACE_END
