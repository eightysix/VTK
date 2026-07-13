// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalRenderWindow.h"

#include "vtkObjectFactory.h"
#include "vtkRenderer.h"
#include "vtkRendererCollection.h"
#include "vtkCommand.h"
#include "vtkUnsignedIntArray.h"

#include <algorithm>
#include <vector>

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#if TARGET_OS_IOS
#import <UIKit/UIKit.h>
#endif

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalRenderWindow);

//------------------------------------------------------------------------------
vtkMetalRenderWindow::vtkMetalRenderWindow() = default;

//------------------------------------------------------------------------------
vtkMetalRenderWindow::~vtkMetalRenderWindow()
{
  this->Finalize();
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//------------------------------------------------------------------------------
bool vtkMetalRenderWindow::InitializeMetal()
{
  if (this->Initialized)
  {
    return true;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device)
    {
      vtkErrorMacro(<< "Failed to create Metal device");
      return false;
    }
    this->MetalDevice = (__bridge void*)device;
    CFRetain((__bridge CFTypeRef)device);

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue)
    {
      vtkErrorMacro(<< "Failed to create Metal command queue");
      return false;
    }
    this->MetalQueue = (__bridge void*)queue;
    CFRetain((__bridge CFTypeRef)queue);
  }

  this->Initialized = true;
  return true;
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::CreateMetalLayer()
{
  if (this->MetalLayer)
  {
    return;
  }

  @autoreleasepool
  {
    CAMetalLayer* layer = [CAMetalLayer layer];
    layer.device = (__bridge id<MTLDevice>)this->MetalDevice;
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = YES;
    layer.opaque = NO;
#if TARGET_OS_IOS
    layer.contentsScale = [UIScreen mainScreen].nativeScale;
#else
    layer.contentsScale = 1.0;
#endif

    CGFloat w = (this->Size[0] > 0) ? this->Size[0] : 300;
    CGFloat h = (this->Size[1] > 0) ? this->Size[1] : 300;
    layer.drawableSize = CGSizeMake(w, h);

    this->MetalLayer = (__bridge void*)layer;
    CFRetain((__bridge CFTypeRef)layer);
  }
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::Initialize()
{
  if (this->Initialized)
  {
    return;
  }

  this->InitializeMetal();
  this->CreateMetalLayer();

  this->Superclass::Initialize();
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::Finalize()
{
  this->ReleaseDrawable();

  // Encoder and CommandBuffer are transient; managed by ARC in vtkMetalRenderer.
  this->Encoder = nullptr;
  this->CommandBuffer = nullptr;

  if (this->DepthTexture)
  {
    CFRelease(this->DepthTexture);
    this->DepthTexture = nullptr;
  }

  if (this->IdsTexture)
  {
    CFRelease(this->IdsTexture);
    this->IdsTexture = nullptr;
  }

  this->DestroyMultisampleAttachments();

  if (this->MetalLayer)
  {
    CFRelease(this->MetalLayer);
    this->MetalLayer = nullptr;
  }

  if (this->MetalQueue)
  {
    CFRelease(this->MetalQueue);
    this->MetalQueue = nullptr;
  }

  if (this->MetalDevice)
  {
    CFRelease(this->MetalDevice);
    this->MetalDevice = nullptr;
  }

  this->Initialized = false;
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::Start()
{
  if (!this->Initialized)
  {
    this->Initialize();
  }
}

//------------------------------------------------------------------------------
bool vtkMetalRenderWindow::AcquireDrawable()
{
  if (this->CurrentDrawable)
  {
    return true;
  }

  @autoreleasepool
  {
    CAMetalLayer* layer = (__bridge CAMetalLayer*)this->MetalLayer;
    if (!layer)
    {
      return false;
    }

    id<CAMetalDrawable> drawable = [layer nextDrawable];
    if (!drawable)
    {
      return false;
    }
    this->CurrentDrawable = (__bridge void*)drawable;
    CFRetain((__bridge CFTypeRef)drawable);
  }
  return true;
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::ReleaseDrawable()
{
  if (this->CurrentDrawable)
  {
    CFRelease(this->CurrentDrawable);
    this->CurrentDrawable = nullptr;
  }
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::RecreateDepthTexture()
{
  if (this->DepthTexture)
  {
    CFRelease(this->DepthTexture);
    this->DepthTexture = nullptr;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)this->MetalDevice;
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                   width:this->Size[0]
                                                                                  height:this->Size[1]
                                                                               mipmapped:NO];
    desc.usage = MTLTextureUsageRenderTarget;
    desc.storageMode = MTLStorageModePrivate;

    id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
    this->DepthTexture = (__bridge void*)tex;
    CFRetain((__bridge CFTypeRef)tex);
  }
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::RecreateIdsTexture()
{
  if (this->IdsTexture)
  {
    CFRelease(this->IdsTexture);
    this->IdsTexture = nullptr;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)this->MetalDevice;
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Uint
                                                                                    width:this->Size[0]
                                                                                   height:this->Size[1]
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageRenderTarget;
    desc.storageMode = MTLStorageModeShared;

    id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
    this->IdsTexture = (__bridge void*)tex;
    CFRetain((__bridge CFTypeRef)tex);
  }
}

//------------------------------------------------------------------------------
int vtkMetalRenderWindow::GetEffectiveSampleCount()
{
  return (this->MultiSamples > 1) ? this->MultiSamples : 1;
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::CreateMultisampleAttachments()
{
  int sampleCount = this->GetEffectiveSampleCount();
  if (sampleCount <= 1)
  {
    return;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)this->MetalDevice;
    if (!device)
    {
      return;
    }

    // --- Multisampled color attachment (BGRA8Unorm, matching the drawable) ---
    {
      MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
      desc.textureType = MTLTextureType2DMultisample;
      desc.pixelFormat = MTLPixelFormatBGRA8Unorm;
      desc.width = this->Size[0];
      desc.height = this->Size[1];
      desc.mipmapLevelCount = 1;
      desc.sampleCount = sampleCount;
      desc.usage = MTLTextureUsageRenderTarget;
      desc.storageMode = MTLStorageModePrivate;

      id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
      if (tex)
      {
        this->MultisampleColorTexture = (__bridge void*)tex;
        CFRetain((__bridge CFTypeRef)tex);
      }
    }

    // --- Multisampled depth attachment (Depth32Float) ---
    {
      MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
      desc.textureType = MTLTextureType2DMultisample;
      desc.pixelFormat = MTLPixelFormatDepth32Float;
      desc.width = this->Size[0];
      desc.height = this->Size[1];
      desc.mipmapLevelCount = 1;
      desc.sampleCount = sampleCount;
      desc.usage = MTLTextureUsageRenderTarget;
      desc.storageMode = MTLStorageModePrivate;

      id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
      if (tex)
      {
        this->MultisampleDepthTexture = (__bridge void*)tex;
        CFRetain((__bridge CFTypeRef)tex);
      }
    }
  }
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::DestroyMultisampleAttachments()
{
  if (this->MultisampleColorTexture)
  {
    CFRelease(this->MultisampleColorTexture);
    this->MultisampleColorTexture = nullptr;
  }
  if (this->MultisampleDepthTexture)
  {
    CFRelease(this->MultisampleDepthTexture);
    this->MultisampleDepthTexture = nullptr;
  }
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::Render()
{
  if (!this->Initialized)
  {
    this->Initialize();
  }

  if (this->Size[0] > 0 && this->Size[1] > 0)
  {
    @autoreleasepool
    {
      id<MTLTexture> depthTex = (__bridge id<MTLTexture>)this->DepthTexture;
      if (!depthTex || depthTex.width != (NSUInteger)this->Size[0] ||
          depthTex.height != (NSUInteger)this->Size[1])
      {
        this->RecreateDepthTexture();
      }

      id<MTLTexture> idsTex = (__bridge id<MTLTexture>)this->IdsTexture;
      if (!idsTex || idsTex.width != (NSUInteger)this->Size[0] ||
          idsTex.height != (NSUInteger)this->Size[1])
      {
        this->RecreateIdsTexture();
      }

      // Recreate multisample attachments when size or sample count changes
      int effectiveSamples = this->GetEffectiveSampleCount();
      id<MTLTexture> msaaColorTex = (__bridge id<MTLTexture>)this->MultisampleColorTexture;
      bool needsMSAACreation = (effectiveSamples > 1) &&
        (!msaaColorTex || msaaColorTex.width != (NSUInteger)this->Size[0] ||
         msaaColorTex.height != (NSUInteger)this->Size[1] ||
         (int)msaaColorTex.sampleCount != effectiveSamples);
      if (needsMSAACreation)
      {
        this->DestroyMultisampleAttachments();
        this->CreateMultisampleAttachments();
      }
    }
  }

  this->Superclass::Render();
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::Frame()
{
  this->Superclass::Frame();
  this->ReleaseDrawable();
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::End()
{
  this->Superclass::End();
}

//------------------------------------------------------------------------------
const char* vtkMetalRenderWindow::GetRenderingBackend()
{
  return "Metal";
}

//------------------------------------------------------------------------------
void* vtkMetalRenderWindow::GetGenericContext()
{
  return this->MetalDevice;
}

//------------------------------------------------------------------------------
void* vtkMetalRenderWindow::GetGenericDisplayId()
{
  return this->MetalDevice;
}

//------------------------------------------------------------------------------
void* vtkMetalRenderWindow::GetMetalLayer()
{
  return this->MetalLayer;
}

//------------------------------------------------------------------------------
void* vtkMetalRenderWindow::GetMetalDevice()
{
  return this->MetalDevice;
}

//------------------------------------------------------------------------------
void* vtkMetalRenderWindow::GetMetalQueue()
{
  return this->MetalQueue;
}

//------------------------------------------------------------------------------
void* vtkMetalRenderWindow::GetCurrentRenderCommandEncoder()
{
  return this->Encoder;
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::SetCurrentRenderCommandEncoder(void* encoder)
{
  this->Encoder = encoder;
}

//------------------------------------------------------------------------------
void* vtkMetalRenderWindow::GetCurrentCommandBuffer()
{
  return this->CommandBuffer;
}

//------------------------------------------------------------------------------
void* vtkMetalRenderWindow::GetCurrentDrawableTexture()
{
  if (this->CurrentDrawable)
  {
    id<CAMetalDrawable> drawable = (__bridge id<CAMetalDrawable>)this->CurrentDrawable;
    return (__bridge void*)drawable.texture;
  }
  return nullptr;
}

//------------------------------------------------------------------------------
void* vtkMetalRenderWindow::GetDepthTexture()
{
  return this->DepthTexture;
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::SetSize(int width, int height)
{
  if (this->Size[0] != width || this->Size[1] != height)
  {
    this->Superclass::SetSize(width, height);

    @autoreleasepool
    {
      CAMetalLayer* layer = (__bridge CAMetalLayer*)this->MetalLayer;
      if (layer)
      {
        layer.drawableSize = CGSizeMake(width, height);
      }
    }

    this->Modified();
  }
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::SetPosition(int x, int y)
{
  this->Superclass::SetPosition(x, y);
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::WaitForCompletion()
{
  @autoreleasepool
  {
    id<MTLCommandBuffer> buf = (__bridge id<MTLCommandBuffer>)this->CommandBuffer;
    if (buf)
    {
      [buf waitUntilCompleted];
    }
  }
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::GetIdsData(int x1, int y1, int x2, int y2,
                                       vtkUnsignedIntArray* data)
{
  if (!this->IdsTexture || !data)
  {
    return;
  }

  @autoreleasepool
  {
    id<MTLTexture> idsTex = (__bridge id<MTLTexture>)this->IdsTexture;

    // Clamp to texture bounds
    int texW = static_cast<int>(idsTex.width);
    int texH = static_cast<int>(idsTex.height);
    int xMin = std::max(0, std::min(x1, x2));
    int yMin = std::max(0, std::min(y1, y2));
    int xMax = std::min(texW - 1, std::max(x1, x2));
    int yMax = std::min(texH - 1, std::max(y1, y2));

    int width = xMax - xMin + 1;
    int height = yMax - yMin + 1;
    if (width <= 0 || height <= 0)
    {
      return;
    }

    // Read the texture region. MTLStorageModeShared allows direct CPU access.
    // Region in texture coordinates: origin is top-left for Metal.
    MTLRegion region = MTLRegionMake2D(xMin, yMin, width, height);
    std::vector<uint32_t> texData(width * height * 4);
    [idsTex getBytes:texData.data()
         fromBytesPerRow:width * 4 * sizeof(uint32_t)
        fromRegion:region
       mipmapLevel:0];

    // Copy into the output array with Y-flip (Metal top-left → VTK bottom-left).
    // Output format: 4 components per pixel: {CellId, PropId, CompositeId, ProcessId}.
    data->SetNumberOfComponents(4);
    data->SetNumberOfTuples(width * height);
    uint32_t* dst = data->GetPointer(0);

    for (int y = 0; y < height; ++y)
    {
      int srcY = height - 1 - y; // flip Y
      for (int x = 0; x < width; ++x)
      {
        int srcIdx = (srcY * width + x) * 4;
        int dstIdx = (y * width + x) * 4;
        dst[dstIdx + 0] = texData[srcIdx + 0]; // cell_id
        dst[dstIdx + 1] = texData[srcIdx + 1]; // prop_id
        dst[dstIdx + 2] = texData[srcIdx + 2]; // composite_id
        dst[dstIdx + 3] = texData[srcIdx + 3]; // process_id
      }
    }
  }
}

VTK_ABI_NAMESPACE_END
