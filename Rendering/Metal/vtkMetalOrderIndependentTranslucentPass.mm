// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalOrderIndependentTranslucentPass.h"
#include "vtkMetalRenderer.h"
#include "vtkMetalRenderWindow.h"
#include "vtkMetalShaders.h"
#include "vtkObjectFactory.h"
#include "vtkRenderer.h"

#import <Metal/Metal.h>

VTK_ABI_NAMESPACE_BEGIN

//------------------------------------------------------------------------------
vtkMetalOrderIndependentTranslucentPass::vtkMetalOrderIndependentTranslucentPass() = default;

//------------------------------------------------------------------------------
vtkMetalOrderIndependentTranslucentPass::~vtkMetalOrderIndependentTranslucentPass()
{
  this->Release();
}

//------------------------------------------------------------------------------
bool vtkMetalOrderIndependentTranslucentPass::NeedsInitialization(int width, int height) const
{
  return !this->AccumTexture ||
         this->CurrentWidth != width ||
         this->CurrentHeight != height;
}

//------------------------------------------------------------------------------
void vtkMetalOrderIndependentTranslucentPass::Release()
{
  [this->AccumTexture release];
  this->AccumTexture = nil;
  [this->RevealTexture release];
  this->RevealTexture = nil;
  [this->ResolvePipeline release];
  this->ResolvePipeline = nil;
  [this->ReadOnlyDepthState release];
  this->ReadOnlyDepthState = nil;
  this->PipelinesCreated = false;
  this->CurrentWidth = 0;
  this->CurrentHeight = 0;
}

//------------------------------------------------------------------------------
void vtkMetalOrderIndependentTranslucentPass::CreateTextures(id<MTLDevice> device, int width, int height)
{
  [this->AccumTexture release];
  this->AccumTexture = nil;
  [this->RevealTexture release];
  this->RevealTexture = nil;

  this->CurrentWidth = width;
  this->CurrentHeight = height;

  // RGBA16F accumulation texture: RGB = weighted color sum, A = transmittance product
  MTLTextureDescriptor* accumDesc =
    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                       width:width
                                                      height:height
                                                   mipmapped:NO];
  accumDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  accumDesc.storageMode = MTLStorageModePrivate;
  this->AccumTexture = [device newTextureWithDescriptor:accumDesc];

  // R16F revealage texture: accumulated opacity sum
  MTLTextureDescriptor* revealDesc =
    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                       width:width
                                                      height:height
                                                   mipmapped:NO];
  revealDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  revealDesc.storageMode = MTLStorageModePrivate;
  this->RevealTexture = [device newTextureWithDescriptor:revealDesc];
}

//------------------------------------------------------------------------------
void vtkMetalOrderIndependentTranslucentPass::CreatePipelines(vtkMetalRenderWindow* renWin)
{
  if (this->PipelinesCreated)
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)renWin->GetMetalDevice();
  NSError* error = nil;
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)renWin->GetSharedShaderLibrary();
  if (!library)
  {
    vtkGenericWarningMacro(<< "No shared shader library available for OIT");
    return;
  }

  // --- Resolve pipeline ---
  // Fullscreen pass: reads AccumTexture + RevealTexture, outputs BGRA8 with
  // the standard over blend (SRC_ALPHA, ONE_MINUS_SRC_ALPHA).
  {
    id<MTLFunction> vFunc = [library newFunctionWithName:@"vertex_fullscreen_main"];
    id<MTLFunction> fFunc = [library newFunctionWithName:@"fragment_oit_resolve"];
    if (vFunc && fFunc)
    {
      MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
      desc.vertexFunction = vFunc;
      desc.fragmentFunction = fFunc;
      desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
      desc.colorAttachments[0].blendingEnabled = YES;
      desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
      desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
      desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
      desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
      desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      desc.depthAttachmentPixelFormat = MTLPixelFormatInvalid;

      this->ResolvePipeline = [device newRenderPipelineStateWithDescriptor:desc error:&error];
      if (!this->ResolvePipeline)
      {
        vtkGenericWarningMacro(<< "OIT resolve pipeline: " << [[error localizedDescription] UTF8String]);
      }
      [desc release];
    }
    [vFunc release];
    [fFunc release];
  }

  // --- Cached depth-stencil state (immutable, created once) ---
  if (!this->ReadOnlyDepthState)
  {
    MTLDepthStencilDescriptor* dsDesc = [[MTLDepthStencilDescriptor alloc] init];
    dsDesc.depthCompareFunction = MTLCompareFunctionLess;
    dsDesc.depthWriteEnabled = NO;
    this->ReadOnlyDepthState = [device newDepthStencilStateWithDescriptor:dsDesc];
    [dsDesc release];
  }

  this->PipelinesCreated = (this->ResolvePipeline != nil);
}

//------------------------------------------------------------------------------
int vtkMetalOrderIndependentTranslucentPass::RenderTranslucentGeometry(
  vtkMetalRenderer* renderer,
  id<MTLCommandBuffer> commandBuffer,
  id<MTLTexture> drawableTexture,
  id<MTLTexture> depthTexture)
{
  if (!renderer || !commandBuffer || !drawableTexture)
  {
    return 0;
  }

  vtkMetalRenderWindow* renWin =
    vtkMetalRenderWindow::SafeDownCast(renderer->GetRenderWindow());
  if (!renWin || !renWin->GetMetalDevice())
  {
    return 0;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)renWin->GetMetalDevice();

  // Get viewport dimensions
  int* size = renderer->GetSize();
  int width = size[0];
  int height = size[1];

  // Renderer sub-rect inside the full-size textures (matches vtkMetalRenderer)
  double* viewport = renderer->GetViewport();

  if (width <= 0 || height <= 0)
  {
    return 0;
  }

  // OIT does not support MSAA or nil depth textures. The intermediate
  // accumulate textures are non-MSAA, so an MSAA depth attachment would
  // mismatch the color attachment sample count.
  if (!depthTexture || depthTexture.sampleCount > 1)
  {
    vtkGenericWarningMacro(<< "vtkMetalOrderIndependentTranslucentPass: invalid or MSAA depth "
                           << "texture; falling back to standard transparency.");
    return 0;
  }

  // Initialize textures and pipelines if needed
  if (this->NeedsInitialization(width, height))
  {
    this->CreateTextures(device, width, height);
  }
  this->CreatePipelines(renWin);

  if (!this->PipelinesCreated || !this->AccumTexture || !this->RevealTexture)
  {
    return 0;
  }

  // --- Pass 1: Clear the accumulation textures ---
  // AccumTexture alpha is cleared to 1.0 (transmittance starts at 1); the
  // (ZERO, ONE_MINUS_SRC_ALPHA) alpha blend multiplies it down per fragment.
  {
    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = this->AccumTexture;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1.0);
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.colorAttachments[1].texture = this->RevealTexture;
    rpd.colorAttachments[1].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[1].clearColor = MTLClearColorMake(0, 0, 0, 0);
    rpd.colorAttachments[1].storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> enc =
      [commandBuffer renderCommandEncoderWithDescriptor:rpd];
    enc.label = @"VTK Order Independent - Clear";
    [enc endEncoding];
  }

  // --- Pass 2: Accumulate translucent geometry ---
  {
    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = this->AccumTexture;
    rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.colorAttachments[1].texture = this->RevealTexture;
    rpd.colorAttachments[1].loadAction = MTLLoadActionLoad;
    rpd.colorAttachments[1].storeAction = MTLStoreActionStore;
    rpd.depthAttachment.texture = depthTexture;
    rpd.depthAttachment.loadAction = MTLLoadActionLoad;
    rpd.depthAttachment.storeAction = MTLStoreActionDontCare;

    id<MTLRenderCommandEncoder> encoder =
      [commandBuffer renderCommandEncoderWithDescriptor:rpd];
    encoder.label = @"VTK Order Independent - Accumulate";

    renWin->SetCurrentCommandBuffer((__bridge void*)commandBuffer);
    renWin->Encoder = (__bridge void*)encoder;

    MTLViewport metalViewport;
    metalViewport.originX = viewport[0] * width;
    metalViewport.originY = viewport[1] * height;
    metalViewport.width = viewport[2] * width;
    metalViewport.height = viewport[3] * height;
    metalViewport.znear = 0.0;
    metalViewport.zfar = 1.0;
    [encoder setViewport:metalViewport];

    // Depth test: Less (match opaque), depth write: No
    [encoder setDepthStencilState:this->ReadOnlyDepthState];
    [encoder setFrontFacingWinding:MTLWindingCounterClockwise];

    renWin->OITActive = true;
    renderer->RenderTranslucentGeometry();
    renWin->OITActive = false;

    [encoder endEncoding];
    renWin->Encoder = nullptr;
  }

  // --- Pass 3: Resolve onto the drawable ---
  {
    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = drawableTexture;
    rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> encoder =
      [commandBuffer renderCommandEncoderWithDescriptor:rpd];
    encoder.label = @"VTK Order Independent - Resolve";

    MTLViewport vp;
    vp.originX = viewport[0] * width;
    vp.originY = viewport[1] * height;
    vp.width = viewport[2] * width;
    vp.height = viewport[3] * height;
    vp.znear = 0.0;
    vp.zfar = 1.0;
    [encoder setViewport:vp];

    [encoder setRenderPipelineState:this->ResolvePipeline];
    [encoder setFragmentTexture:this->AccumTexture atIndex:0];
    [encoder setFragmentTexture:this->RevealTexture atIndex:1];

    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];

    [encoder endEncoding];
  }

  return 1;
}

VTK_ABI_NAMESPACE_END
