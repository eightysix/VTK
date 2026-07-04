// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalRenderer.h"

#include "vtkMetalRenderWindow.h"
#include "vtkMetalCamera.h"
#include "vtkObjectFactory.h"
#include "vtkRenderer.h"
#include "vtkRendererCollection.h"
#include "vtkViewport.h"

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalRenderer);

//------------------------------------------------------------------------------
vtkMetalRenderer::vtkMetalRenderer() = default;

//------------------------------------------------------------------------------
vtkMetalRenderer::~vtkMetalRenderer() = default;

//------------------------------------------------------------------------------
void vtkMetalRenderer::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//------------------------------------------------------------------------------
void vtkMetalRenderer::Clear()
{
  // Background clearing is done via the render pass load action
  this->Superclass::Clear();
}

//------------------------------------------------------------------------------
void vtkMetalRenderer::DeviceRender()
{
  vtkMetalRenderWindow* renWin =
    vtkMetalRenderWindow::SafeDownCast(this->GetRenderWindow());
  if (!renWin)
  {
    return;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)renWin->GetMetalDevice();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)[device newCommandQueue];
    if (!queue)
    {
      return;
    }

    // Acquire drawable
    CAMetalLayer* layer = (__bridge CAMetalLayer*)renWin->GetMetalLayer();
    if (!layer)
    {
      return;
    }

    id<CAMetalDrawable> drawable = [layer nextDrawable];
    if (!drawable)
    {
      return;
    }

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    commandBuffer.label = @"VTK Metal Renderer";

    // Create render pass descriptor
    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = drawable.texture;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;

    // Set background color
    double bgColor[3];
    this->GetBackground(bgColor);
    rpd.colorAttachments[0].clearColor = MTLClearColorMake(bgColor[0], bgColor[1], bgColor[2], 1.0);
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

    // Depth attachment
    id<MTLTexture> depthTex = (__bridge id<MTLTexture>)[renWin GetGenericContext];
    // Get depth texture from render window
    // For now, skip depth if not available

    id<MTLRenderCommandEncoder> encoder =
      [commandBuffer renderCommandEncoderWithDescriptor:rpd];
    encoder.label = @"VTK Render Encoder";

    // Store encoder and command buffer for mappers to use
    renWin->CommandBuffer = (__bridge_retained void*)commandBuffer;

    // Update camera and set viewport
    if (this->ActiveCamera)
    {
      this->ActiveCamera->Render(this);
      this->ActiveCamera->UpdateViewport(this);
    }

    // Render opaque geometry
    this->UpdateOpaquePolygonalGeometry();

    // Render translucent geometry (sorted back-to-front)
    this->UpdateTranslucentPolygonalGeometry();

    [encoder endEncoding];

    // Commit and present
    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];

    // Clean up
    if (renWin->CommandBuffer)
    {
      CFRelease(renWin->CommandBuffer);
      renWin->CommandBuffer = nullptr;
    }
  }
}

//------------------------------------------------------------------------------
int vtkMetalRenderer::UpdateGeometry(vtkFrameBufferObjectBase*)
{
  return this->UpdateOpaquePolygonalGeometry() +
         this->UpdateTranslucentPolygonalGeometry();
}

//------------------------------------------------------------------------------
void vtkMetalRenderer::ReleaseGraphicsResources(vtkWindow* w)
{
  this->Superclass::ReleaseGraphicsResources(w);
}

VTK_ABI_NAMESPACE_END
