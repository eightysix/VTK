// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalRenderWindow.h"

#include "vtkObjectFactory.h"
#include "vtkOverrideAttribute.h"
#include "vtkRenderer.h"
#include "vtkRendererCollection.h"
#include "vtkCommand.h"
#include "vtkUnsignedIntArray.h"
#include "vtkUnsignedCharArray.h"
#include "vtkFloatArray.h"
#include "vtkNew.h"
#include "vtkMetalShaders.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <vector>

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#if TARGET_OS_IOS
#import <UIKit/UIKit.h>
#endif

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalRenderWindow);

//------------------------------------------------------------------------------
vtkOverrideAttribute* vtkMetalRenderWindow::CreateOverrideAttributes()
{
  return vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "Metal", nullptr);
}

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
    // MTLCreateSystemDefaultDevice returns +1 (Create rule); member owns it directly.
    this->MetalDevice = (void*)device;

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue)
    {
      vtkErrorMacro(<< "Failed to create Metal command queue");
      [device release];
      this->MetalDevice = nullptr;
      return false;
    }
    this->MetalQueue = (void*)queue;
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
    layer.device = (id<MTLDevice>)this->MetalDevice;
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
#ifdef VTK_METAL_ENABLE_COLOR_READBACK
    // NO so the drawable texture can be blitted into the shared color-copy
    // texture used for CPU read-back (GetPixelData / image regression tests).
    layer.framebufferOnly = NO;
#else
    // Production default: lets the system optimize the drawable (it is only
    // ever presented, never read back).
    layer.framebufferOnly = YES;
#endif
    layer.opaque = NO;
#if TARGET_OS_IOS
    layer.contentsScale = [UIScreen mainScreen].nativeScale;
#else
    layer.contentsScale = 1.0;
#endif

    CGFloat w = (this->Size[0] > 0) ? this->Size[0] : 300;
    CGFloat h = (this->Size[1] > 0) ? this->Size[1] : 300;
    layer.drawableSize = CGSizeMake(w, h);

    this->MetalLayer = (void*)layer;
    CFRetain((CFTypeRef)layer);
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
void* vtkMetalRenderWindow::GetSharedShaderLibrary()
{
  std::call_once(this->LibraryInitFlag, [this]() {
    @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)this->MetalDevice;
      if (!device)
      {
        vtkErrorMacro(<< "Cannot compile shader library: Metal device is null");
        return;
      }
      NSString* source = [NSString stringWithUTF8String:vtkMetalShaders];
      NSError* error = nil;
      id<MTLLibrary> lib = [device newLibraryWithSource:source options:nil error:&error];
      if (!lib)
      {
        vtkErrorMacro(<< "Failed to compile shared shader library: "
                      << [[error localizedDescription] UTF8String]);
        return;
      }
      this->SharedShaderLibrary = (void*)lib;
    }
  });
  return this->SharedShaderLibrary;
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::Finalize()
{
  this->ReleaseDrawable();

  // Encoder is transient (managed by ARC in vtkMetalRenderer); the command
  // buffer is retained by SetCurrentCommandBuffer, so release it here.
  this->Encoder = nullptr;
  this->SetCurrentCommandBuffer(nullptr);

  if (this->RenderCompletionCallback)
  {
    [(id)this->RenderCompletionCallback release];
    this->RenderCompletionCallback = nullptr;
  }

  if (this->DepthTexture)
  {
    [(id)this->DepthTexture release];
    this->DepthTexture = nullptr;
  }

  if (this->IdsTexture)
  {
    [(id)this->IdsTexture release];
    this->IdsTexture = nullptr;
  }

  if (this->ColorCopyTexture)
  {
    [(id)this->ColorCopyTexture release];
    this->ColorCopyTexture = nullptr;
  }

  if (this->DepthCopyTexture)
  {
    [(id)this->DepthCopyTexture release];
    this->DepthCopyTexture = nullptr;
  }

#ifdef VTK_METAL_ENABLE_OFFSCREEN_TARGET
  if (this->OffscreenColorTexture)
  {
    [(id)this->OffscreenColorTexture release];
    this->OffscreenColorTexture = nullptr;
  }
#endif

  this->DestroyMultisampleAttachments();

  if (this->SharedShaderLibrary)
  {
    [(id)this->SharedShaderLibrary release];
    this->SharedShaderLibrary = nullptr;
  }

  if (this->MetalLayer)
  {
    [(id)this->MetalLayer release];
    this->MetalLayer = nullptr;
  }

  if (this->MetalQueue)
  {
    [(id)this->MetalQueue release];
    this->MetalQueue = nullptr;
  }

  if (this->MetalDevice)
  {
    [(id)this->MetalDevice release];
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
    CAMetalLayer* layer = (CAMetalLayer*)this->MetalLayer;
    if (!layer)
    {
      return false;
    }

    id<CAMetalDrawable> drawable = [layer nextDrawable];
    if (!drawable)
    {
      return false;
    }
    this->CurrentDrawable = (void*)drawable;
    CFRetain((CFTypeRef)drawable);
  }
  return true;
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::ReleaseDrawable()
{
  if (this->CurrentDrawable)
  {
    [(id)this->CurrentDrawable release];
    this->CurrentDrawable = nullptr;
  }
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::RecreateDepthTexture()
{
  if (this->DepthTexture)
  {
    [(id)this->DepthTexture release];
    this->DepthTexture = nullptr;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (id<MTLDevice>)this->MetalDevice;
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                   width:this->Size[0]
                                                                                  height:this->Size[1]
                                                                               mipmapped:NO];
    // ShaderRead allows the volume mapper to sample scene depth for early
    // ray termination (depth buffer occlusion).
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModePrivate;

    // newTextureWithDescriptor returns +1 (new rule); member owns it directly.
    id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
    this->DepthTexture = (void*)tex;
  }
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::RecreateIdsTexture()
{
  if (this->IdsTexture)
  {
    [(id)this->IdsTexture release];
    this->IdsTexture = nullptr;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (id<MTLDevice>)this->MetalDevice;
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Uint
                                                                                    width:this->Size[0]
                                                                                   height:this->Size[1]
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageRenderTarget;
    desc.storageMode = MTLStorageModeShared;

    // newTextureWithDescriptor returns +1 (new rule); member owns it directly.
    id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
    this->IdsTexture = (void*)tex;
  }
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::RecreateColorCopyTexture()
{
#ifdef VTK_METAL_ENABLE_COLOR_READBACK
  if (this->ColorCopyTexture)
  {
    [(id)this->ColorCopyTexture release];
    this->ColorCopyTexture = nullptr;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (id<MTLDevice>)this->MetalDevice;
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                   width:this->Size[0]
                                                                                  height:this->Size[1]
                                                                               mipmapped:NO];
    // Shared storage allows synchronous CPU reads via getBytes after the GPU
    // frame completes. RenderTarget/ShaderRead usage keeps it a valid blit
    // destination.
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;

    // newTextureWithDescriptor returns +1 (new rule); member owns it directly.
    id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
    this->ColorCopyTexture = (void*)tex;
  }
#endif
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::RecreateDepthCopyTexture()
{
#ifdef VTK_METAL_ENABLE_COLOR_READBACK
  if (this->DepthCopyTexture)
  {
    [(id)this->DepthCopyTexture release];
    this->DepthCopyTexture = nullptr;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (id<MTLDevice>)this->MetalDevice;
    if (!device)
    {
      return;
    }
    MTLTextureDescriptor* desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                         width:this->Size[0]
                                                        height:this->Size[1]
                                                     mipmapped:NO];
    // Shared storage allows synchronous CPU reads via getBytes after the GPU
    // frame completes. RenderTarget/ShaderRead/ShaderWrite usage makes it a
    // valid blit (and MSAA resolve) destination.
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModeShared;

    // newTextureWithDescriptor returns +1 (new rule); member owns it directly.
    id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
    this->DepthCopyTexture = (void*)tex;
  }
#endif
}

//------------------------------------------------------------------------------
#ifdef VTK_METAL_ENABLE_OFFSCREEN_TARGET
void vtkMetalRenderWindow::RecreateOffscreenColorTexture()
{
  if (this->OffscreenColorTexture)
  {
    [(id)this->OffscreenColorTexture release];
    this->OffscreenColorTexture = nullptr;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (id<MTLDevice>)this->MetalDevice;
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                   width:this->Size[0]
                                                                                  height:this->Size[1]
                                                                               mipmapped:NO];
    // Private storage: only the GPU renders to this buffer; it is never read
    // back by the CPU (benchmark timing only). RenderTarget/ShaderRead usage
    // matches the drawable texture so all passes (opaque, translucent, OIT,
    // depth-peel and overlay resolves) accept it as a color attachment.
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModePrivate;

    // newTextureWithDescriptor returns +1 (new rule); member owns it directly.
    id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
    this->OffscreenColorTexture = (void*)tex;
  }
}
#endif

//------------------------------------------------------------------------------
int vtkMetalRenderWindow::GetEffectiveSampleCount()
{
  int requested = (this->MultiSamples > 1) ? this->MultiSamples : 1;
  if (requested <= 1)
  {
    return 1;
  }

  id<MTLDevice> device = (id<MTLDevice>)this->MetalDevice;
  if (!device)
  {
    // Device not created yet; be conservative so texture creation never
    // requests an unsupported sample count.
    return 4;
  }

  // Metal validates the sample count when the multisample texture is created
  // and aborts on an unsupported value, so clamp to the device's maximum.
  // The Apple GPU family supports only 1/2/4 samples; discrete Mac GPUs
  // typically support up to 8.
  bool appleGpu = false;
  if (@available(macOS 10.15, iOS 13.0, *))
  {
    appleGpu = [device supportsFamily:MTLGPUFamilyApple1];
  }
  else
  {
    appleGpu = true;
  }
  int maxSamples = appleGpu ? 4 : 8;

  return (requested < maxSamples) ? requested : maxSamples;
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
    id<MTLDevice> device = (id<MTLDevice>)this->MetalDevice;
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
      [desc release];
      if (tex)
      {
        this->MultisampleColorTexture = (void*)tex;
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
      [desc release];
      if (tex)
      {
        this->MultisampleDepthTexture = (void*)tex;
      }
    }
  }
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::DestroyMultisampleAttachments()
{
  if (this->MultisampleColorTexture)
  {
    [(id)this->MultisampleColorTexture release];
    this->MultisampleColorTexture = nullptr;
  }
  if (this->MultisampleDepthTexture)
  {
    [(id)this->MultisampleDepthTexture release];
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

  this->FrameRendererIndex = 0;

  if (this->Size[0] > 0 && this->Size[1] > 0)
  {
    @autoreleasepool
    {
      id<MTLTexture> depthTex = (id<MTLTexture>)this->DepthTexture;
      if (!depthTex || depthTex.width != (NSUInteger)this->Size[0] ||
          depthTex.height != (NSUInteger)this->Size[1])
      {
        this->RecreateDepthTexture();
      }

      id<MTLTexture> idsTex = (id<MTLTexture>)this->IdsTexture;
      if (!idsTex || idsTex.width != (NSUInteger)this->Size[0] ||
          idsTex.height != (NSUInteger)this->Size[1])
      {
        this->RecreateIdsTexture();
      }

#ifdef VTK_METAL_ENABLE_COLOR_READBACK
      id<MTLTexture> colorCopyTex = (id<MTLTexture>)this->ColorCopyTexture;
      if (!colorCopyTex || colorCopyTex.width != (NSUInteger)this->Size[0] ||
          colorCopyTex.height != (NSUInteger)this->Size[1])
      {
        this->RecreateColorCopyTexture();
      }

      id<MTLTexture> depthCopyTex = (id<MTLTexture>)this->DepthCopyTexture;
      if (!depthCopyTex || depthCopyTex.width != (NSUInteger)this->Size[0] ||
          depthCopyTex.height != (NSUInteger)this->Size[1])
      {
        this->RecreateDepthCopyTexture();
      }
#endif

#ifdef VTK_METAL_ENABLE_OFFSCREEN_TARGET
      if (this->GetOffScreenRendering())
      {
        id<MTLTexture> offscreenTex = (id<MTLTexture>)this->OffscreenColorTexture;
        if (!offscreenTex || offscreenTex.width != (NSUInteger)this->Size[0] ||
            offscreenTex.height != (NSUInteger)this->Size[1])
        {
          this->RecreateOffscreenColorTexture();
        }
      }
#endif

      // Recreate multisample attachments when size or sample count changes
      int effectiveSamples = this->GetEffectiveSampleCount();
      id<MTLTexture> msaaColorTex = (id<MTLTexture>)this->MultisampleColorTexture;
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
void vtkMetalRenderWindow::SetCurrentCommandBuffer(void* commandBuffer)
{
  if (commandBuffer == this->CommandBuffer)
  {
    return;
  }

  @autoreleasepool
  {
    if (this->CommandBuffer)
    {
      CFRelease((CFTypeRef)this->CommandBuffer);
    }
    this->CommandBuffer = commandBuffer;
    if (commandBuffer)
    {
      // The renderer's ARC local is released when DeviceRender returns, so
      // retain the buffer here to keep it alive until the next frame (or
      // Finalize) replaces it. This makes WaitForCompletion() safe to call
      // after the frame's autorelease scope has drained.
      CFRetain((CFTypeRef)commandBuffer);
    }
  }
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::SetRenderCompletionCallback(VTKRenderCompletionBlock block)
{
  if (this->RenderCompletionCallback)
  {
    [(id)this->RenderCompletionCallback release];
    this->RenderCompletionCallback = nullptr;
  }
  if (block)
  {
    this->RenderCompletionCallback = (void*)[block copy];
  }
}

//------------------------------------------------------------------------------
void* vtkMetalRenderWindow::GetCurrentDrawableTexture()
{
  if (this->CurrentDrawable)
  {
    id<CAMetalDrawable> drawable = (id<CAMetalDrawable>)this->CurrentDrawable;
    return (void*)drawable.texture;
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
      CAMetalLayer* layer = (CAMetalLayer*)this->MetalLayer;
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
    id<MTLCommandBuffer> buf = (id<MTLCommandBuffer>)this->CommandBuffer;
    if (buf)
    {
      [buf waitUntilCompleted];
    }
  }
}

//------------------------------------------------------------------------------
uint64_t vtkMetalRenderWindow::GetAllocatedSize()
{
  @autoreleasepool
  {
    id<MTLDevice> device = (id<MTLDevice>)this->MetalDevice;
    if (!device)
    {
      return 0;
    }
    return (uint64_t)device.currentAllocatedSize;
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
    id<MTLTexture> idsTex = (id<MTLTexture>)this->IdsTexture;

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
    // Region in texture coordinates: origin is top-left for Metal, while VTK
    // y grows upward. Map the VTK y range [yMin, yMax] to texture rows
    // [texH - 1 - yMax, texH - 1 - yMin] (top to bottom) so the crop samples
    // the same vertical band the color image shows; the loop below then flips
    // rows within the crop so row 0 is VTK yMin.
    int texYMin = texH - 1 - yMax;
    MTLRegion region = MTLRegionMake2D(xMin, texYMin, width, height);
    std::vector<uint32_t> texData(width * height * 4);
    [idsTex getBytes:texData.data()
         bytesPerRow:width * 4 * sizeof(uint32_t)
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

//------------------------------------------------------------------------------
int vtkMetalRenderWindow::ReadColorCopyData(
  int x, int y, int width, int height, int ncomp, void* dest)
{
#ifndef VTK_METAL_ENABLE_COLOR_READBACK
  (void)x;
  (void)y;
  (void)width;
  (void)height;
  (void)ncomp;
  (void)dest;
  return 0;
#else
  if (!dest)
  {
    return 0;
  }

  @autoreleasepool
  {
    id<MTLTexture> colorTex = (__bridge id<MTLTexture>)this->ColorCopyTexture;
    if (!colorTex)
    {
      return 0;
    }

    // Clamp to texture bounds
    int texW = static_cast<int>(colorTex.width);
    int texH = static_cast<int>(colorTex.height);
    int xMin = std::max(0, x);
    int yMin = std::max(0, y);
    int xMax = std::min(texW - 1, x + width - 1);
    int yMax = std::min(texH - 1, y + height - 1);

    int w = xMax - xMin + 1;
    int h = yMax - yMin + 1;
    if (w <= 0 || h <= 0)
    {
      return 0;
    }

    // Wait for the frame whose blit wrote ColorCopyTexture to finish before
    // reading it back (getBytes would otherwise race the GPU).
    this->WaitForCompletion();

    // The requested rect uses VTK window coordinates (origin at the lower
    // left, y up), but the texture is top-origin (texture row 0 is the top of
    // the framebuffer). A VTK row vtkY maps to texture row (texH-1-vtkY), so
    // the contiguous texture region holding the rect's rows runs from
    // (texH-1)-(yMin+h-1) to (texH-1)-yMin, in reverse row order. Read that
    // region, then emit rows bottom-up so output row 0 is the rect's bottom
    // (VTK convention), matching vtkOpenGLRenderWindow::GetPixelData.
    const int firstTexRow = (texH - 1) - (yMin + h - 1);
    MTLRegion region = MTLRegionMake2D(xMin, firstTexRow, w, h);
    std::vector<uint8_t> texData(w * h * 4);
    [colorTex getBytes:texData.data() bytesPerRow:w * 4 fromRegion:region mipmapLevel:0];

    // Copy into the output array (bottom-up) with BGRA → RGB(A) channel
    // conversion.
    unsigned char* out = static_cast<unsigned char*>(dest);
    for (int row = 0; row < h; ++row)
    {
      const int vtkRow = yMin + row;
      const int srcRow = (yMin + h - 1) - vtkRow; // rows are reversed in the texture
      for (int col = 0; col < w; ++col)
      {
        const uint8_t* src = &texData[(srcRow * w + col) * 4];
        unsigned char* dst = &out[(row * w + col) * ncomp];
        dst[0] = src[2]; // B
        dst[1] = src[1]; // G
        dst[2] = src[0]; // R
        if (ncomp == 4)
        {
          dst[3] = src[3]; // A
        }
      }
    }
    return 1;
  }
#endif
}

//------------------------------------------------------------------------------
unsigned char* vtkMetalRenderWindow::GetPixelData(
  int x1, int y1, int x2, int y2, int front, int right)
{
  (void)front;
  (void)right;

  int x_low = std::min(x1, x2);
  int x_hi = std::max(x1, x2);
  int y_low = std::min(y1, y2);
  int y_hi = std::max(y1, y2);

  int width = (x_hi - x_low) + 1;
  int height = (y_hi - y_low) + 1;

  unsigned char* data = new unsigned char[width * height * 3];
  if (!this->ReadColorCopyData(x_low, y_low, width, height, 3, data))
  {
    // No color-copy texture available (e.g. no frame rendered yet): return a
    // zeroed image rather than null so callers can still consume the buffer.
    std::fill(data, data + width * height * 3, 0);
  }
  return data;
}

//------------------------------------------------------------------------------
int vtkMetalRenderWindow::GetPixelData(
  int x1, int y1, int x2, int y2, int front, vtkUnsignedCharArray* data, int right)
{
  (void)front;
  (void)right;

  if (!data)
  {
    return 0;
  }

  int x_low = std::min(x1, x2);
  int x_hi = std::max(x1, x2);
  int y_low = std::min(y1, y2);
  int y_hi = std::max(y1, y2);

  int width = (x_hi - x_low) + 1;
  int height = (y_hi - y_low) + 1;
  int size = 3 * width * height;

  data->SetNumberOfComponents(3);
  data->SetNumberOfValues(size);

  if (!this->ReadColorCopyData(x_low, y_low, width, height, 3, data->GetPointer(0)))
  {
    std::fill(data->GetPointer(0), data->GetPointer(0) + size, 0);
    return 0;
  }
  return 1;
}

//------------------------------------------------------------------------------
unsigned char* vtkMetalRenderWindow::GetRGBACharPixelData(
  int x1, int y1, int x2, int y2, int front, int right)
{
  (void)front;
  (void)right;

  int x_low = std::min(x1, x2);
  int x_hi = std::max(x1, x2);
  int y_low = std::min(y1, y2);
  int y_hi = std::max(y1, y2);

  int width = (x_hi - x_low) + 1;
  int height = (y_hi - y_low) + 1;

  unsigned char* data = new unsigned char[width * height * 4];
  if (!this->ReadColorCopyData(x_low, y_low, width, height, 4, data))
  {
    std::fill(data, data + width * height * 4, 0);
  }
  return data;
}

//------------------------------------------------------------------------------
int vtkMetalRenderWindow::GetRGBACharPixelData(
  int x1, int y1, int x2, int y2, int front, vtkUnsignedCharArray* data, int right)
{
  (void)front;
  (void)right;

  if (!data)
  {
    return 0;
  }

  int x_low = std::min(x1, x2);
  int x_hi = std::max(x1, x2);
  int y_low = std::min(y1, y2);
  int y_hi = std::max(y1, y2);

  int width = (x_hi - x_low) + 1;
  int height = (y_hi - y_low) + 1;
  int size = 4 * width * height;

  data->SetNumberOfComponents(4);
  data->SetNumberOfValues(size);

  if (!this->ReadColorCopyData(x_low, y_low, width, height, 4, data->GetPointer(0)))
  {
    std::fill(data->GetPointer(0), data->GetPointer(0) + size, 0);
    return 0;
  }
  return 1;
}

//------------------------------------------------------------------------------
float* vtkMetalRenderWindow::GetRGBAPixelData(int x1, int y1, int x2, int y2, int front, int right)
{
  (void)front;
  (void)right;

  vtkNew<vtkUnsignedCharArray> rgba;
  if (!this->GetRGBACharPixelData(x1, y1, x2, y2, front, rgba, right))
  {
    return nullptr;
  }

  const vtkIdType n = rgba->GetNumberOfValues();
  float* data = new float[n];
  const unsigned char* src = rgba->GetPointer(0);
  const float inv = 1.0f / 255.0f;
  for (vtkIdType i = 0; i < n; ++i)
  {
    data[i] = src[i] * inv;
  }
  return data;
}

//------------------------------------------------------------------------------
int vtkMetalRenderWindow::GetRGBAPixelData(
  int x1, int y1, int x2, int y2, int front, vtkFloatArray* data, int right)
{
  (void)front;
  (void)right;

  if (!data)
  {
    return 0;
  }

  vtkNew<vtkUnsignedCharArray> rgba;
  if (!this->GetRGBACharPixelData(x1, y1, x2, y2, front, rgba, right))
  {
    return 0;
  }

  const vtkIdType n = rgba->GetNumberOfValues();
  data->SetNumberOfComponents(4);
  data->SetNumberOfValues(n);
  const unsigned char* src = rgba->GetPointer(0);
  float* dst = data->GetPointer(0);
  const float inv = 1.0f / 255.0f;
  for (vtkIdType i = 0; i < n; ++i)
  {
    dst[i] = src[i] * inv;
  }
  return 1;
}

//------------------------------------------------------------------------------
void vtkMetalRenderWindow::ReleaseRGBAPixelData(float* data)
{
  delete[] data;
}

//------------------------------------------------------------------------------
int vtkMetalRenderWindow::GetColorBufferSizes(int* rgba)
{
  if (rgba == nullptr)
  {
    return 0;
  }

  // The drawable, the multisample color attachment, and the color-copy
  // texture are all BGRA8Unorm (8 bits per channel).
  rgba[0] = 8;
  rgba[1] = 8;
  rgba[2] = 8;
  rgba[3] = 8;
  return 1;
}

//------------------------------------------------------------------------------
int vtkMetalRenderWindow::ReadDepthCopyData(int x, int y, int width, int height, float* dest)
{
#ifndef VTK_METAL_ENABLE_COLOR_READBACK
  (void)x;
  (void)y;
  (void)width;
  (void)height;
  (void)dest;
  return 0;
#else
  if (!dest)
  {
    return 0;
  }

  @autoreleasepool
  {
    id<MTLTexture> depthTex = (__bridge id<MTLTexture>)this->DepthCopyTexture;
    if (!depthTex)
    {
      return 0;
    }

    int texW = static_cast<int>(depthTex.width);
    int texH = static_cast<int>(depthTex.height);
    int xMin = std::max(0, x);
    int yMin = std::max(0, y);
    int xMax = std::min(texW - 1, x + width - 1);
    int yMax = std::min(texH - 1, y + height - 1);

    int w = xMax - xMin + 1;
    int h = yMax - yMin + 1;
    if (w <= 0 || h <= 0)
    {
      return 0;
    }

    // Wait for the frame whose copy wrote DepthCopyTexture to finish before
    // reading it back (getBytes would otherwise race the GPU).
    this->WaitForCompletion();

    // The requested rect uses VTK window coordinates (origin at the lower
    // left, y up), but the texture is top-origin (texture row 0 is the top of
    // the framebuffer), so read the flipped region and emit rows bottom-up.
    const int firstTexRow = (texH - 1) - (yMin + h - 1);
    MTLRegion region = MTLRegionMake2D(xMin, firstTexRow, w, h);
    std::vector<float> texData(w * h);
    [depthTex getBytes:texData.data() bytesPerRow:w * sizeof(float) fromRegion:region mipmapLevel:0];

    for (int row = 0; row < h; ++row)
    {
      const int vtkRow = yMin + row;
      const int srcRow = (yMin + h - 1) - vtkRow; // rows are reversed in the texture
      std::copy(&texData[srcRow * w], &texData[srcRow * w] + w, dest + row * w);
    }
    return 1;
  }
#endif
}

//------------------------------------------------------------------------------
float* vtkMetalRenderWindow::GetZbufferData(int x1, int y1, int x2, int y2)
{
  int x_low = std::min(x1, x2);
  int x_hi = std::max(x1, x2);
  int y_low = std::min(y1, y2);
  int y_hi = std::max(y1, y2);

  int width = (x_hi - x_low) + 1;
  int height = (y_hi - y_low) + 1;

  float* data = new float[width * height];
  if (!this->ReadDepthCopyData(x_low, y_low, width, height, data))
  {
    std::fill(data, data + width * height, 0.0f);
  }
  return data;
}

//------------------------------------------------------------------------------
int vtkMetalRenderWindow::GetZbufferData(int x1, int y1, int x2, int y2, float* z)
{
  if (!z)
  {
    return 0;
  }

  int x_low = std::min(x1, x2);
  int x_hi = std::max(x1, x2);
  int y_low = std::min(y1, y2);
  int y_hi = std::max(y1, y2);

  int width = (x_hi - x_low) + 1;
  int height = (y_hi - y_low) + 1;

  if (!this->ReadDepthCopyData(x_low, y_low, width, height, z))
  {
    std::fill(z, z + width * height, 0.0f);
    return 0;
  }
  return 1;
}

//------------------------------------------------------------------------------
int vtkMetalRenderWindow::GetZbufferData(int x1, int y1, int x2, int y2, vtkFloatArray* z)
{
  if (!z)
  {
    return 0;
  }

  int x_low = std::min(x1, x2);
  int x_hi = std::max(x1, x2);
  int y_low = std::min(y1, y2);
  int y_hi = std::max(y1, y2);

  int width = (x_hi - x_low) + 1;
  int height = (y_hi - y_low) + 1;
  int size = width * height;

  z->SetNumberOfComponents(1);
  z->SetNumberOfValues(size);

  if (!this->ReadDepthCopyData(x_low, y_low, width, height, z->GetPointer(0)))
  {
    std::fill(z->GetPointer(0), z->GetPointer(0) + size, 0.0f);
    return 0;
  }
  return 1;
}

VTK_ABI_NAMESPACE_END
