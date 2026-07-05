// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalRenderer.h"

#include "vtkMetalRenderWindow.h"
#include "vtkMetalCamera.h"
#include "vtkObjectFactory.h"
#include "vtkRenderer.h"
#include "vtkRendererCollection.h"
#include "vtkLightCollection.h"
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
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)renWin->GetMetalQueue();
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

    // Attach IDs texture for GPU-based picking (color attachment 1)
    id<MTLTexture> idsTex = (__bridge id<MTLTexture>)renWin->IdsTexture;
    if (idsTex)
    {
      rpd.colorAttachments[1].texture = idsTex;
      rpd.colorAttachments[1].loadAction = MTLLoadActionClear;
      rpd.colorAttachments[1].clearColor = MTLClearColorMake(0, 0, 0, 0);
      rpd.colorAttachments[1].storeAction = MTLStoreActionStore;
    }

    // Attach depth texture for depth testing
    id<MTLTexture> depthTex = (__bridge id<MTLTexture>)renWin->DepthTexture;
    if (depthTex)
    {
      rpd.depthAttachment.texture = depthTex;
      rpd.depthAttachment.loadAction = MTLLoadActionClear;
      rpd.depthAttachment.clearDepth = 1.0;
      rpd.depthAttachment.storeAction = MTLStoreActionDontCare;
    }

    id<MTLRenderCommandEncoder> encoder =
      [commandBuffer renderCommandEncoderWithDescriptor:rpd];
    encoder.label = @"VTK Render Encoder";

    // Set depth stencil state for depth testing (Less comparison, write enabled)
    if (depthTex)
    {
      MTLDepthStencilDescriptor* dsDesc = [[MTLDepthStencilDescriptor alloc] init];
      dsDesc.depthCompareFunction = MTLCompareFunctionLess;
      dsDesc.depthWriteEnabled = YES;
      id<MTLDepthStencilState> depthState = [device newDepthStencilStateWithDescriptor:dsDesc];
      [encoder setDepthStencilState:depthState];
    }

    // Store encoder and command buffer for mappers to use.
    // No CFRetain needed: the local ARC strong references keep these alive
    // for the duration of this @autoreleasepool block.
    renWin->CommandBuffer = (__bridge void*)commandBuffer;
    renWin->Encoder = (__bridge void*)encoder;

    // Set viewport
    int* size = this->GetSize();
    double* viewport = this->GetViewport();
    MTLViewport metalViewport;
    metalViewport.originX = viewport[0] * size[0];
    metalViewport.originY = viewport[1] * size[1];
    metalViewport.width = viewport[2] * size[0];
    metalViewport.height = viewport[3] * size[1];
    metalViewport.znear = 0.0;
    metalViewport.zfar = 1.0;
    [encoder setViewport:metalViewport];

    // Update camera and set viewport
    if (this->ActiveCamera)
    {
      this->ActiveCamera->Render(this);
      this->ActiveCamera->UpdateViewport(this);
    }

    // Create default headlight if none exist (matching WebGPU renderer behavior)
    if (this->GetLights()->GetNumberOfItems() == 0 && this->AutomaticLightCreation)
    {
      this->CreateLight();
    }

    // Render opaque geometry
    this->UpdateOpaquePolygonalGeometry();

    // Render translucent geometry (sorted back-to-front)
    this->UpdateTranslucentPolygonalGeometry();

    [encoder endEncoding];

    // Commit and present
    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];

    // Clear pointers (objects are released by ARC when block exits)
    renWin->Encoder = nullptr;
    renWin->CommandBuffer = nullptr;
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
