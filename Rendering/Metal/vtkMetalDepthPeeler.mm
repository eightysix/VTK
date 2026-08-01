// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalDepthPeeler.h"
#include "vtkMetalRenderer.h"
#include "vtkMetalRenderWindow.h"
#include "vtkMetalShaders.h"
#include "vtkObjectFactory.h"
#include "vtkRenderer.h"

#import <Metal/Metal.h>

#include <vector>
#include <functional>

VTK_ABI_NAMESPACE_BEGIN

//------------------------------------------------------------------------------
vtkMetalDepthPeeler::vtkMetalDepthPeeler() = default;

//------------------------------------------------------------------------------
vtkMetalDepthPeeler::~vtkMetalDepthPeeler()
{
  this->Release();
}

//------------------------------------------------------------------------------
bool vtkMetalDepthPeeler::NeedsInitialization(int width, int height) const
{
  return !this->FrontPeelA ||
         this->CurrentWidth != width ||
         this->CurrentHeight != height;
}

//------------------------------------------------------------------------------
void vtkMetalDepthPeeler::Release()
{
  [this->FrontPeelA release];
  this->FrontPeelA = nil;
  [this->FrontPeelB release];
  this->FrontPeelB = nil;
  [this->BackPeelTemp release];
  this->BackPeelTemp = nil;
  [this->BackAccum release];
  this->BackAccum = nil;
  [this->DepthPeelA release];
  this->DepthPeelA = nil;
  [this->DepthPeelB release];
  this->DepthPeelB = nil;
  [this->CompositePipeline release];
  this->CompositePipeline = nil;
  [this->BackBlendPipeline release];
  this->BackBlendPipeline = nil;
  [this->ReadOnlyDepthState release];
  this->ReadOnlyDepthState = nil;
  [this->AlwaysDepthState release];
  this->AlwaysDepthState = nil;
  this->PipelinesCreated = false;
  this->CurrentWidth = 0;
  this->CurrentHeight = 0;
}

//------------------------------------------------------------------------------
void vtkMetalDepthPeeler::CreateTextures(id<MTLDevice> device, int width, int height)
{
  // Release old textures before creating new ones
  [this->FrontPeelA release];   this->FrontPeelA = nil;
  [this->FrontPeelB release];   this->FrontPeelB = nil;
  [this->BackPeelTemp release]; this->BackPeelTemp = nil;
  [this->BackAccum release];    this->BackAccum = nil;
  [this->DepthPeelA release];   this->DepthPeelA = nil;
  [this->DepthPeelB release];   this->DepthPeelB = nil;

  this->CurrentWidth = width;
  this->CurrentHeight = height;

  // Front accumulation textures (RGBA8, ping-pong)
  auto createRGBA8 = [&](int w, int h) -> id<MTLTexture> {
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                    width:w
                                                                                   height:h
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModePrivate;
    return [device newTextureWithDescriptor:desc];
  };

  // Depth range textures (RG32Float, ping-pong)
  auto createRG32F = [&](int w, int h) -> id<MTLTexture> {
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG32Float
                                                                                    width:w
                                                                                   height:h
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModePrivate;
    return [device newTextureWithDescriptor:desc];
  };

  this->FrontPeelA = createRGBA8(width, height);
  this->FrontPeelB = createRGBA8(width, height);
  this->BackPeelTemp = createRGBA8(width, height);
  this->BackAccum = createRGBA8(width, height);
  this->DepthPeelA = createRG32F(width, height);
  this->DepthPeelB = createRG32F(width, height);
}

//------------------------------------------------------------------------------
void vtkMetalDepthPeeler::CreatePipelines(vtkMetalRenderWindow* renWin)
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
    vtkGenericWarningMacro(<< "No shared shader library available for depth peeling");
    return;
  }

  // --- Composite pipeline ---
  // Fullscreen pass: reads front + back textures, outputs BGRA8 with over-blend
  {
    id<MTLFunction> vFunc = [library newFunctionWithName:@"vertex_fullscreen_main"];
    id<MTLFunction> fFunc = [library newFunctionWithName:@"fragment_peel_composite"];
    if (vFunc && fFunc)
    {
      MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
      desc.vertexFunction = vFunc;
      desc.fragmentFunction = fFunc;
      desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
      desc.colorAttachments[0].blendingEnabled = YES;
      desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
      desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
      desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
      desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
      desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      desc.depthAttachmentPixelFormat = MTLPixelFormatInvalid;

      this->CompositePipeline = [device newRenderPipelineStateWithDescriptor:desc error:&error];
      if (!this->CompositePipeline)
      {
        vtkGenericWarningMacro(<< "Composite pipeline: " << [[error localizedDescription] UTF8String]);
      }
      [desc release];
    }
    [vFunc release];
    [fFunc release];
  }

  // --- Back blend pipeline ---
  // Fullscreen pass: reads backTemp, outputs RGBA8 with over-blend
  {
    id<MTLFunction> vFunc = [library newFunctionWithName:@"vertex_fullscreen_main"];
    id<MTLFunction> fFunc = [library newFunctionWithName:@"fragment_peel_back_blend"];
    if (vFunc && fFunc)
    {
      MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
      desc.vertexFunction = vFunc;
      desc.fragmentFunction = fFunc;
      desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
      desc.colorAttachments[0].blendingEnabled = YES;
      desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
      desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
      desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
      desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
      desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      desc.depthAttachmentPixelFormat = MTLPixelFormatInvalid;

      this->BackBlendPipeline = [device newRenderPipelineStateWithDescriptor:desc error:&error];
      if (!this->BackBlendPipeline)
      {
        vtkGenericWarningMacro(<< "Back blend pipeline: " << [[error localizedDescription] UTF8String]);
      }
      [desc release];
    }
    [vFunc release];
    [fFunc release];
  }

  // --- Cached depth-stencil states (immutable, created once) ---
  if (!this->ReadOnlyDepthState)
  {
    MTLDepthStencilDescriptor* dsDesc = [[MTLDepthStencilDescriptor alloc] init];
    dsDesc.depthCompareFunction = MTLCompareFunctionLess;
    dsDesc.depthWriteEnabled = NO;
    this->ReadOnlyDepthState = [device newDepthStencilStateWithDescriptor:dsDesc];
    [dsDesc release];
  }

  if (!this->AlwaysDepthState)
  {
    MTLDepthStencilDescriptor* dsDesc = [[MTLDepthStencilDescriptor alloc] init];
    dsDesc.depthCompareFunction = MTLCompareFunctionAlways;
    dsDesc.depthWriteEnabled = NO;
    this->AlwaysDepthState = [device newDepthStencilStateWithDescriptor:dsDesc];
    [dsDesc release];
  }

  this->PipelinesCreated = (this->CompositePipeline && this->BackBlendPipeline);
}

//------------------------------------------------------------------------------
int vtkMetalDepthPeeler::RenderTranslucentGeometry(
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

  if (width <= 0 || height <= 0)
  {
    return 0;
  }

  // Depth peeling does not support MSAA or nil depth textures.
  // The intermediate peel textures are non-MSAA, so an MSAA depth attachment
  // would mismatch the color attachment sample count.
  if (!depthTexture || depthTexture.sampleCount > 1)
  {
    vtkGenericWarningMacro(<< "vtkMetalDepthPeeler: invalid or MSAA depth texture; "
                           << "falling back to standard transparency.");
    return 0;
  }

  // Initialize textures and pipelines if needed
  if (this->NeedsInitialization(width, height))
  {
    this->CreateTextures(device, width, height);
  }
  this->CreatePipelines(renWin);

  if (!this->PipelinesCreated)
  {
    return 0;
  }

  // Helper to clear a single-attachment render target
  auto clearTexture = [&](id<MTLTexture> tex, MTLClearColor clear) {
    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = tex;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].clearColor = clear;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> enc = [commandBuffer renderCommandEncoderWithDescriptor:rpd];
    [enc endEncoding];
  };

  // Clear the two textures that are read (never cleared in-loop) before the
  // loop starts: FrontPeelA (front source of peel 0) and BackAccum (blended
  // into by every back-blend). The remaining targets are cleared by folding
  // loadAction=Clear into the pass that first writes them (the init pass for
  // DepthPeelA, the peel pass for DepthPeelB/BackPeelTemp/frontDst), avoiding
  // a dedicated render pass per clear.
  clearTexture(this->FrontPeelA, MTLClearColorMake(0, 0, 0, 0));
  clearTexture(this->BackAccum, MTLClearColorMake(0, 0, 0, 0));

  // --- Pass 1: Initialize depth range ---
  // Render translucent geometry with MAX blending on DepthPeelA.
  // Depth test=Less against the opaque depth buffer ensures occluded fragments are discarded.
  {
    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = this->DepthPeelA;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].clearColor = MTLClearColorMake(-1.0, -1.0, 0, 0);
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.depthAttachment.texture = depthTexture;
    rpd.depthAttachment.loadAction = MTLLoadActionLoad;
    rpd.depthAttachment.storeAction = MTLStoreActionDontCare;

    id<MTLRenderCommandEncoder> encoder =
      [commandBuffer renderCommandEncoderWithDescriptor:rpd];
    encoder.label = @"VTK Depth Peeling - Init";

    renWin->SetCurrentCommandBuffer((__bridge void*)commandBuffer);
    renWin->Encoder = (__bridge void*)encoder;

    MTLViewport metalViewport;
    metalViewport.originX = 0;
    metalViewport.originY = 0;
    metalViewport.width = width;
    metalViewport.height = height;
    metalViewport.znear = 0.0;
    metalViewport.zfar = 1.0;
    [encoder setViewport:metalViewport];

    [encoder setDepthStencilState:this->ReadOnlyDepthState];

    renWin->DepthPeelingMode = 1;
    renWin->PeelFrontTexture = nullptr;
    renWin->PeelDepthTexture = (__bridge void*)this->DepthPeelB;
    renWin->PeelIndex = 0;

    renderer->RenderTranslucentGeometry();

    renWin->DepthPeelingMode = 0;

    [encoder endEncoding];
    renWin->Encoder = nullptr;
  }

  // --- Pass 2: Peel loop ---
  id<MTLTexture> frontSrc = this->FrontPeelA;
  id<MTLTexture> frontDst = this->FrontPeelB;
  id<MTLTexture> depthSrc = this->DepthPeelA;
  id<MTLTexture> depthDst = this->DepthPeelB;

  int numPeels = 0;
  for (int peel = 0; peel < this->MaximumNumberOfPeels; ++peel)
  {
    // Peel pass: render translucent geometry to 3 targets. The targets are
    // cleared here (loadAction=Clear) rather than in separate render passes
    // (each clear is a full render-pass encoder in Metal).
    {
      MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
      rpd.colorAttachments[0].texture = this->BackPeelTemp;
      rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
      rpd.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
      rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
      rpd.colorAttachments[1].texture = frontDst;
      rpd.colorAttachments[1].loadAction = MTLLoadActionClear;
      rpd.colorAttachments[1].clearColor = MTLClearColorMake(0, 0, 0, 0);
      rpd.colorAttachments[1].storeAction = MTLStoreActionStore;
      rpd.colorAttachments[2].texture = depthDst;
      rpd.colorAttachments[2].loadAction = MTLLoadActionClear;
      rpd.colorAttachments[2].clearColor = MTLClearColorMake(-1.0, -1.0, 0, 0);
      rpd.colorAttachments[2].storeAction = MTLStoreActionStore;

      id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:rpd];
      encoder.label = [NSString stringWithFormat:@"VTK Depth Peeling - Peel %d", peel];

      renWin->SetCurrentCommandBuffer((__bridge void*)commandBuffer);
      renWin->Encoder = (__bridge void*)encoder;

      MTLViewport metalViewport;
      metalViewport.originX = 0;
      metalViewport.originY = 0;
      metalViewport.width = width;
      metalViewport.height = height;
      metalViewport.znear = 0.0;
      metalViewport.zfar = 1.0;
      [encoder setViewport:metalViewport];

      [encoder setDepthStencilState:this->AlwaysDepthState];

      renWin->DepthPeelingMode = 2;
      renWin->PeelFrontTexture = (__bridge void*)frontSrc;
      renWin->PeelDepthTexture = (__bridge void*)depthSrc;
      renWin->PeelIndex = peel;

      renderer->RenderTranslucentGeometry();

      renWin->DepthPeelingMode = 0;

      [encoder endEncoding];
      renWin->Encoder = nullptr;
    }

    // Blend backTemp into back accumulation
    {
      MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
      rpd.colorAttachments[0].texture = this->BackAccum;
      rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;
      rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

      id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:rpd];
      encoder.label = @"VTK Depth Peeling - Back Blend";

      MTLViewport vp;
      vp.originX = 0; vp.originY = 0;
      vp.width = width; vp.height = height;
      vp.znear = 0.0; vp.zfar = 1.0;
      [encoder setViewport:vp];

      [encoder setRenderPipelineState:this->BackBlendPipeline];
      [encoder setFragmentTexture:this->BackPeelTemp atIndex:0];

      [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];

      [encoder endEncoding];
    }

    // Swap ping-pong buffers
    std::swap(frontSrc, frontDst);
    std::swap(depthSrc, depthDst);
    numPeels++;
  }

  // --- Pass 3: Composite ---
  {
    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = drawableTexture;
    rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> encoder =
      [commandBuffer renderCommandEncoderWithDescriptor:rpd];
    encoder.label = @"VTK Depth Peeling - Composite";

    MTLViewport vp;
    vp.originX = 0; vp.originY = 0;
    vp.width = width; vp.height = height;
    vp.znear = 0.0; vp.zfar = 1.0;
    [encoder setViewport:vp];

    [encoder setRenderPipelineState:this->CompositePipeline];
    [encoder setFragmentTexture:frontSrc atIndex:0];
    [encoder setFragmentTexture:this->BackAccum atIndex:1];

    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];

    [encoder endEncoding];
  }

  return numPeels;
}

VTK_ABI_NAMESPACE_END
