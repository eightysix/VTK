// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalRayCastImageDisplayHelper.h"

#include "vtkFixedPointRayCastImage.h"
#include "vtkMetalRenderWindow.h"
#include "vtkObjectFactory.h"
#include "vtkOverrideAttribute.h"
#include "vtkRenderer.h"
#include "vtkVolume.h"

#import <Metal/Metal.h>

#include <cstring>
#include <vector>

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalRayCastImageDisplayHelper);

//------------------------------------------------------------------------------
vtkOverrideAttribute* vtkMetalRayCastImageDisplayHelper::CreateOverrideAttributes()
{
  return vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "Metal", nullptr);
}

//------------------------------------------------------------------------------
vtkMetalRayCastImageDisplayHelper::vtkMetalRayCastImageDisplayHelper() = default;

//------------------------------------------------------------------------------
vtkMetalRayCastImageDisplayHelper::~vtkMetalRayCastImageDisplayHelper()
{
  this->ReleaseGraphicsResources(nullptr);
}

//------------------------------------------------------------------------------
void vtkMetalRayCastImageDisplayHelper::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//------------------------------------------------------------------------------
void vtkMetalRayCastImageDisplayHelper::ReleaseGraphicsResources(vtkWindow*)
{
  @autoreleasepool
  {
    if (this->RenderPipeline)
    {
      [(__bridge id)this->RenderPipeline release];
      this->RenderPipeline = nullptr;
    }
    if (this->Sampler)
    {
      [(__bridge id)this->Sampler release];
      this->Sampler = nullptr;
    }
    if (this->ShaderLibrary)
    {
      [(__bridge id)this->ShaderLibrary release];
      this->ShaderLibrary = nullptr;
    }
  }
}

//------------------------------------------------------------------------------
void vtkMetalRayCastImageDisplayHelper::RenderTexture(
  vtkVolume* vol, vtkRenderer* ren, vtkFixedPointRayCastImage* image, float requestedDepth)
{
  this->RenderTextureInternal(vol, ren, image->GetImageMemorySize(), image->GetImageViewportSize(),
    image->GetImageInUseSize(), image->GetImageOrigin(), requestedDepth, VTK_UNSIGNED_SHORT,
    image->GetImage());
}

//------------------------------------------------------------------------------
void vtkMetalRayCastImageDisplayHelper::RenderTexture(vtkVolume* vol, vtkRenderer* ren,
  int imageMemorySize[2], int imageViewportSize[2], int imageInUseSize[2], int imageOrigin[2],
  float requestedDepth, unsigned char* image)
{
  this->RenderTextureInternal(vol, ren, imageMemorySize, imageViewportSize, imageInUseSize,
    imageOrigin, requestedDepth, VTK_UNSIGNED_CHAR, static_cast<void*>(image));
}

//------------------------------------------------------------------------------
void vtkMetalRayCastImageDisplayHelper::RenderTexture(vtkVolume* vol, vtkRenderer* ren,
  int imageMemorySize[2], int imageViewportSize[2], int imageInUseSize[2], int imageOrigin[2],
  float requestedDepth, unsigned short* image)
{
  this->RenderTextureInternal(vol, ren, imageMemorySize, imageViewportSize, imageInUseSize,
    imageOrigin, requestedDepth, VTK_UNSIGNED_SHORT, static_cast<void*>(image));
}

//------------------------------------------------------------------------------
void vtkMetalRayCastImageDisplayHelper::RenderTextureInternal(vtkVolume* vol, vtkRenderer* ren,
  int imageMemorySize[2], int imageViewportSize[2], int imageInUseSize[2], int imageOrigin[2],
  float requestedDepth, int imageScalarType, void* image)
{
  if (!image || imageMemorySize[0] <= 0 || imageMemorySize[1] <= 0)
  {
    return;
  }

  vtkMetalRenderWindow* renWin = vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());
  if (!renWin)
  {
    return;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)renWin->GetMetalDevice();
    id<MTLRenderCommandEncoder> encoder =
      (__bridge id<MTLRenderCommandEncoder>)renWin->GetCurrentRenderCommandEncoder();
    if (!device || !encoder)
    {
      return;
    }

    // Window-space depth at which to draw the image. The Metal camera projects
    // to clip-space z in [0,1] (nearz=0, farz=1) and the viewport maps that
    // identity to the depth buffer, so both branches below yield a value that
    // can be used directly as the quad's NDC z.
    float depth;
    if (requestedDepth > 0.0f && requestedDepth <= 1.0f)
    {
      depth = requestedDepth;
    }
    else
    {
      // Pass the center of the volume through the world to display function of
      // the renderer to get the depth of the center of the volume.
      ren->SetWorldPoint(vol->GetCenter()[0], vol->GetCenter()[1], vol->GetCenter()[2], 1.0);
      ren->WorldToDisplay();
      depth = static_cast<float>(ren->GetDisplayPoint()[2]);
    }

    // Upload the CPU ray-cast image into a texture. vtkFixedPointVolumeRayCastMapper
    // uses RGBA unsigned-short (15-bit) values; the char overload handles 8-bit.
    const bool isShort = (imageScalarType == VTK_UNSIGNED_SHORT);
    const size_t bytesPerComponent = isShort ? 2 : 1;
    MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType = MTLTextureType2D;
    desc.pixelFormat =
      isShort ? MTLPixelFormatRGBA16Unorm : MTLPixelFormatRGBA8Unorm;
    desc.width = static_cast<NSUInteger>(imageMemorySize[0]);
    desc.height = static_cast<NSUInteger>(imageMemorySize[1]);
    desc.mipmapLevelCount = 1;
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared; // CPU writes via replaceRegion
    id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
    [desc release];
    if (!tex)
    {
      return;
    }
    MTLRegion region = MTLRegionMake2D(
      0, 0, static_cast<NSUInteger>(imageMemorySize[0]), static_cast<NSUInteger>(imageMemorySize[1]));
    const size_t srcStride =
      static_cast<size_t>(imageMemorySize[0]) * 4 * bytesPerComponent;
    // replaceRegion requires bytesPerRow to be a multiple of 64 on macOS.
    const size_t alignedStride = ((srcStride + 63) / 64) * 64;
    if (alignedStride == srcStride)
    {
      [tex replaceRegion:region mipmapLevel:0 withBytes:image bytesPerRow:srcStride];
    }
    else
    {
      std::vector<uint8_t> padded(alignedStride * static_cast<size_t>(imageMemorySize[1]));
      for (int row = 0; row < imageMemorySize[1]; ++row)
      {
        std::memcpy(padded.data() + row * alignedStride,
          static_cast<const uint8_t*>(image) + row * srcStride, srcStride);
      }
      [tex replaceRegion:region mipmapLevel:0 withBytes:padded.data() bytesPerRow:alignedStride];
    }

    if (!this->Sampler)
    {
      MTLSamplerDescriptor* sdesc = [[MTLSamplerDescriptor alloc] init];
      sdesc.minFilter = MTLSamplerMinMagFilterLinear;
      sdesc.magFilter = MTLSamplerMinMagFilterLinear;
      sdesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
      sdesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
      this->Sampler = (__bridge void*)[device newSamplerStateWithDescriptor:sdesc];
      [sdesc release];
    }

    const NSUInteger sampleCount =
      static_cast<NSUInteger>(renWin->GetEffectiveSampleCount());
    const bool premultiplied = (this->PreMultipliedColors != 0);

    // (Re)create the display pipeline when the sample count or blend mode
    // changes; the color/depth attachment formats are fixed for the volume pass.
    if (!this->RenderPipeline || this->LastSampleCount != sampleCount ||
      this->LastPreMultiplied != premultiplied)
    {
      if (!this->ShaderLibrary)
      {
        this->ShaderLibrary = renWin->GetSharedShaderLibrary();
        if (this->ShaderLibrary)
        {
          [(__bridge id)this->ShaderLibrary retain];
        }
      }
      id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->ShaderLibrary;
      if (!library)
      {
        return;
      }

      if (this->RenderPipeline)
      {
        [(__bridge id)this->RenderPipeline release];
        this->RenderPipeline = nullptr;
      }

      id<MTLFunction> vert = [library newFunctionWithName:@"vertex_raycast_display"];
      id<MTLFunction> frag = [library newFunctionWithName:@"fragment_raycast_display"];
      if (vert && frag)
      {
        MTLRenderPipelineDescriptor* pdesc = [[MTLRenderPipelineDescriptor alloc] init];
        pdesc.vertexFunction = vert;
        pdesc.fragmentFunction = frag;
        pdesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        pdesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
        pdesc.colorAttachments[0].blendingEnabled = YES;
        if (premultiplied)
        {
          pdesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
          pdesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
          pdesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
          pdesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        }
        else
        {
          pdesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
          pdesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
          pdesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
          pdesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        }
        pdesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
        pdesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        pdesc.rasterSampleCount = sampleCount;
        NSError* error = nil;
        id<MTLRenderPipelineState> pso =
          [device newRenderPipelineStateWithDescriptor:pdesc error:&error];
        [pdesc release];
        if (!pso)
        {
          vtkErrorMacro(<< "RayCast image display pipeline: "
                        << [[error localizedDescription] UTF8String]);
        }
        this->RenderPipeline = (__bridge void*)pso;
        this->LastSampleCount = static_cast<uint32_t>(sampleCount);
        this->LastPreMultiplied = premultiplied;
      }
      [vert release];
      [frag release];
    }

    if (!this->RenderPipeline)
    {
      return;
    }

    const float memW = static_cast<float>(imageMemorySize[0]);
    const float memH = static_cast<float>(imageMemorySize[1]);
    const float vpW = static_cast<float>(imageViewportSize[0]);
    const float vpH = static_cast<float>(imageViewportSize[1]);
    const float inW = static_cast<float>(imageInUseSize[0]);
    const float inH = static_cast<float>(imageInUseSize[1]);
    const float oX = static_cast<float>(imageOrigin[0]);
    const float oY = static_cast<float>(imageOrigin[1]);

    const float offX = 0.5f / memW;
    const float offY = 0.5f / memH;

    // Texture coordinates, matching vtkOpenGLRayCastImageDisplayHelper.
    const float u0 = 0.0f + offX;
    const float v0 = 0.0f + offY;
    const float u1 = inW / memW - offX;
    const float v1 = offY;
    const float u2 = inW / memW - offX;
    const float v2 = inH / memH - offY;
    const float u3 = offX;
    const float v3 = inH / memH - offY;

    // NDC corners of the image rect in the viewport (lower-left origin).
    const float nx0 = 2.0f * oX / vpW - 1.0f;
    const float ny0 = 2.0f * oY / vpH - 1.0f;
    const float nx1 = 2.0f * (oX + inW) / vpW - 1.0f;
    const float ny1 = 2.0f * (oY + inH) / vpH - 1.0f;

    // Two triangles: (LL, LR, UR) and (LL, UR, UL).
    struct RayCastDisplayVertexData
    {
      float x, y, z, w;
      float u, v;
    };
    RayCastDisplayVertexData verts[6] = {
      { nx0, ny0, depth, 1.0f, u0, v0 },
      { nx1, ny0, depth, 1.0f, u1, v1 },
      { nx1, ny1, depth, 1.0f, u2, v2 },
      { nx0, ny0, depth, 1.0f, u0, v0 },
      { nx1, ny1, depth, 1.0f, u2, v2 },
      { nx0, ny1, depth, 1.0f, u3, v3 },
    };

    [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)this->RenderPipeline];
    [encoder setVertexBytes:verts length:sizeof(verts) atIndex:0];
    [encoder setFragmentTexture:tex atIndex:0];
    [encoder setFragmentSamplerState:(__bridge id<MTLSamplerState>)this->Sampler atIndex:0];
    [encoder setFragmentBytes:&this->PixelScale length:sizeof(float) atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];

    [tex release];
  }
}
VTK_ABI_NAMESPACE_END
