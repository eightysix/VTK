// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalRenderWindow.h"

#include "vtkObjectFactory.h"
#include "vtkRenderer.h"
#include "vtkRendererCollection.h"
#include "vtkCommand.h"

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

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
    layer.contentsScale = 1.0;

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

  if (this->DepthTexture)
  {
    CFRelease(this->DepthTexture);
    this->DepthTexture = nullptr;
  }

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
void* vtkMetalRenderWindow::GetCurrentRenderCommandEncoder()
{
  return nullptr;
}

//------------------------------------------------------------------------------
void* vtkMetalRenderWindow::GetCurrentCommandBuffer()
{
  return this->CommandBuffer;
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

VTK_ABI_NAMESPACE_END
