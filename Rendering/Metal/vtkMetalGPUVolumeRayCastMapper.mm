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

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <algorithm>
#include <cstring>
#include <vector>

struct VolumeMapperUniforms
{
  float WorldToVolumeMatrix[16];
  float VolumeToWorldMatrix[16];
  float VolumeBoundsMin[4];
  float VolumeBoundsMax[4];
  float CameraVolumePos[4];
  float SampleDistance;
  float ScalarMin;
  float ScalarMax;
  float UseJittering;
  float Padding[4]; // ensure 16-byte alignment (Metal rounds struct to 208)
};

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
void vtkMetalGPUVolumeRayCastMapper::ReleaseGraphicsResources(vtkWindow* vtkNotUsed(window))
{
  if (this->PipelineState)
  {
    CFRelease(this->PipelineState);
    this->PipelineState = nullptr;
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
  ratio[0] = 1.0;
  ratio[1] = 1.0;
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

      // Release old texture
      if (this->VolumeTexture)
      {
        CFRelease(this->VolumeTexture);
        this->VolumeTexture = nullptr;
        this->VolumeTextureView = nullptr;
      }

      int actualComponents = (numComponents == 3) ? 4 : numComponents;
      NSUInteger bytesPerRow = static_cast<NSUInteger>(dims[0]) * fmtInfo.bytesPerComponent *
        actualComponents;
      NSUInteger bytesPerImage = bytesPerRow * dims[1];

      MTLTextureDescriptor* texDesc = [[MTLTextureDescriptor alloc] init];
      texDesc.textureType = MTLTextureType3D;
      texDesc.pixelFormat = fmtInfo.format;
      texDesc.width = dims[0];
      texDesc.height = dims[1];
      texDesc.depth = dims[2];
      texDesc.mipmapLevelCount = 1;
      texDesc.usage = MTLTextureUsageShaderRead;
      texDesc.storageMode = MTLStorageModePrivate;

      id<MTLTexture> tex = [device newTextureWithDescriptor:texDesc];
      if (!tex)
      {
        vtkErrorMacro("Failed to create 3D volume texture");
        return false;
      }
      this->VolumeTexture = (__bridge void*)tex;
      CFRetain((__bridge CFTypeRef)tex);
      this->VolumeTextureView = this->VolumeTexture;

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
      [uploadCmdBuf waitUntilCompleted];

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

      double range[2];
      vtkImageData* input = vtkImageData::SafeDownCast(this->GetInput());
      if (input && input->GetPointData()->GetScalars())
      {
        input->GetPointData()->GetScalars()->GetRange(range);
      }
      else
      {
        range[0] = 0.0;
        range[1] = 255.0;
      }

      unsigned char tfData[256 * 4];
      for (int i = 0; i < 256; ++i)
      {
        double val = range[0] + (range[1] - range[0]) * (i / 255.0);
        double rgb[3];
        colorFunc->GetColor(val, rgb);
        double opacity = opacityFunc->GetValue(val);
        tfData[i * 4 + 0] = static_cast<unsigned char>(rgb[0] * 255.0);
        tfData[i * 4 + 1] = static_cast<unsigned char>(rgb[1] * 255.0);
        tfData[i * 4 + 2] = static_cast<unsigned char>(rgb[2] * 255.0);
        tfData[i * 4 + 3] = static_cast<unsigned char>(opacity * 255.0);
      }

      // Release old texture
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

      id<MTLTexture> tex = [device newTextureWithDescriptor:tfDesc];
      if (!tex)
      {
        vtkErrorMacro("Failed to create transfer function texture");
        return false;
      }
      this->ColorOpacityTexture = (__bridge void*)tex;
      CFRetain((__bridge CFTypeRef)tex);
      this->ColorOpacityTextureView = this->ColorOpacityTexture;

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
bool vtkMetalGPUVolumeRayCastMapper::SetupBuffers(
  void* mtlDeviceVoid, vtkVolume* vol, vtkImageData* input)
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

    if (!this->VertexBuffer || this->GetMTime() > this->VolumeUploadTime)
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
    }
  }

  return this->VertexBuffer && this->IndexBuffer && this->UniformBuffer;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::SetupPipeline(void* mtlDeviceVoid, vtkRenderer* ren)
{
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

    // Pipeline: vertex buffer layout — float3 position at buffer index 0
    MTLVertexDescriptor* vertexDesc = [[MTLVertexDescriptor alloc] init];
    vertexDesc.attributes[0].format = MTLVertexFormatFloat3;
    vertexDesc.attributes[0].offset = 0;
    vertexDesc.attributes[0].bufferIndex = 0;
    vertexDesc.layouts[0].stride = sizeof(float) * 3;
    vertexDesc.layouts[0].stepRate = 1;
    vertexDesc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;

    // Get sample count from renderer
    auto* metalRenderWindow = vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());
    int sampleCount = metalRenderWindow ? metalRenderWindow->GetEffectiveSampleCount() : 1;

    MTLRenderPipelineDescriptor* pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunc;
    pipelineDesc.fragmentFunction = fragmentFunc;
    pipelineDesc.vertexDescriptor = vertexDesc;
    pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

    // Blending: SrcAlpha, OneMinusSrcAlpha (matching WebGPU)
    pipelineDesc.colorAttachments[0].blendingEnabled = YES;
    pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
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
  id<MTLRenderCommandEncoder> encoder =
    (__bridge id<MTLRenderCommandEncoder>)metalRenderWindow->GetCurrentRenderCommandEncoder();

  if (!device || !encoder)
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

  if (!this->UpdateVolumeTexture(mtlDevice, mtlQueue, vol))
  {
    return;
  }
  if (!this->UpdateTransferFunctionTexture(mtlDevice, mtlQueue, vol))
  {
    return;
  }
  if (!this->SetupBuffers(mtlDevice, vol, input))
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
  uniforms.SampleDistance =
    static_cast<float>(this->GetSampleDistance() / maxBoundsSize);

  {
    vtkImageData* inputImg = vtkImageData::SafeDownCast(this->GetInput());
    double scalarRange[2] = { 0.0, 1.0 };
    if (inputImg && inputImg->GetPointData()->GetScalars())
    {
      inputImg->GetPointData()->GetScalars()->GetRange(scalarRange);
    }
    float normFactor = this->ScalarNormalizationFactor;
    uniforms.ScalarMin = static_cast<float>(scalarRange[0] / normFactor);
    uniforms.ScalarMax = static_cast<float>(
      (scalarRange[1] > scalarRange[0] ? scalarRange[1] : scalarRange[0] + 1.0) / normFactor);
  }

  uniforms.UseJittering = this->GetUseJittering() ? 1.0f : 0.0f;

  // Update uniform buffer
  id<MTLBuffer> uniformBuf = (__bridge id<MTLBuffer>)this->UniformBuffer;
  memcpy([uniformBuf contents], &uniforms, sizeof(uniforms));

  // Compute viewProjection matrix = projection * view
  // SceneTransforms layout (from vtkMetalCamera.h):
  //   offset 0:   float ViewMatrix[4][4]     (64 bytes)
  //   offset 64:  float ProjectionMatrix[4][4] (64 bytes)
  vtkMetalCamera* metalCamera = vtkMetalCamera::SafeDownCast(ren->GetActiveCamera());
  float viewProjection[16];
  if (metalCamera)
  {
    const float* sceneData = static_cast<const float*>(metalCamera->GetCachedSceneTransforms());
    const float* V = sceneData;         // ViewMatrix at offset 0
    const float* P = sceneData + 16;    // ProjectionMatrix at offset 64 (16 floats)
    for (int c = 0; c < 4; ++c)
    {
      for (int r = 0; r < 4; ++r)
      {
        viewProjection[c * 4 + r] = P[0 * 4 + r] * V[c * 4 + 0] +
          P[1 * 4 + r] * V[c * 4 + 1] + P[2 * 4 + r] * V[c * 4 + 2] +
          P[3 * 4 + r] * V[c * 4 + 3];
      }
    }
  }
  else
  {
    memset(viewProjection, 0, sizeof(viewProjection));
    viewProjection[0] = viewProjection[5] = viewProjection[10] = viewProjection[15] = 1.0f;
  }

  // Create or reuse viewProjection buffer
  id<MTLBuffer> vpBuf = [device newBufferWithBytes:viewProjection
                                            length:sizeof(viewProjection)
                                           options:MTLResourceStorageModeShared];

  // Set pipeline and arguments
  id<MTLRenderPipelineState> pipeline = (__bridge id<MTLRenderPipelineState>)this->PipelineState;
  [encoder setRenderPipelineState:pipeline];

  // Bind vertex buffer (positions)
  id<MTLBuffer> vertexBuf = (__bridge id<MTLBuffer>)this->VertexBuffer;
  [encoder setVertexBuffer:vertexBuf offset:0 atIndex:0];

  // Bind uniform buffer (vertex + fragment)
  [encoder setVertexBuffer:uniformBuf offset:0 atIndex:1];
  [encoder setFragmentBuffer:uniformBuf offset:0 atIndex:1];

  // Bind viewProjection matrix (vertex only)
  [encoder setVertexBuffer:vpBuf offset:0 atIndex:2];

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

  // Draw
  id<MTLBuffer> indexBuf = (__bridge id<MTLBuffer>)this->IndexBuffer;
  [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                      indexCount:this->IndexCount
                       indexType:MTLIndexTypeUInt32
                     indexBuffer:indexBuf
               indexBufferOffset:0];
}

VTK_ABI_NAMESPACE_END
