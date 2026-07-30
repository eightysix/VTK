// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalRenderer.h"

#include "vtkMetalRenderWindow.h"
#include "vtkMetalDepthPeeler.h"
#include "vtkMetalCamera.h"
#include "vtkMetalGPUVolumeRayCastMapper.h"
#include "vtkMetalShaders.h"
#include "vtkObjectFactory.h"
#include "vtkRenderer.h"
#include "vtkRendererCollection.h"
#include "vtkLightCollection.h"
#include "vtkViewport.h"
#include "vtkActor.h"
#include "vtkActorCollection.h"
#include "vtkVolume.h"
#include "vtkVolumeCollection.h"

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

VTK_ABI_NAMESPACE_BEGIN

// Cached depth-stencil states (immutable, size-independent).
// Created lazily on first use; live for the process lifetime.
static id<MTLDepthStencilState> sOpaqueDepthState = nil;      // Less, write=YES
static id<MTLDepthStencilState> sReadOnlyDepthState = nil;    // Less, write=NO

static void EnsureDepthStencilStates(id<MTLDevice> device)
{
  if (sOpaqueDepthState && sReadOnlyDepthState)
  {
    return;
  }

  @autoreleasepool
  {
    if (!sOpaqueDepthState)
    {
      MTLDepthStencilDescriptor* desc = [[MTLDepthStencilDescriptor alloc] init];
      desc.depthCompareFunction = MTLCompareFunctionLess;
      desc.depthWriteEnabled = YES;
      sOpaqueDepthState = [device newDepthStencilStateWithDescriptor:desc];
      [desc release];
    }

    if (!sReadOnlyDepthState)
    {
      MTLDepthStencilDescriptor* desc = [[MTLDepthStencilDescriptor alloc] init];
      desc.depthCompareFunction = MTLCompareFunctionLess;
      desc.depthWriteEnabled = NO;
      sReadOnlyDepthState = [device newDepthStencilStateWithDescriptor:desc];
      [desc release];
    }
  }
}

vtkStandardNewMacro(vtkMetalRenderer);

//------------------------------------------------------------------------------
vtkMetalRenderer::vtkMetalRenderer()
  : DepthPeeler(new vtkMetalDepthPeeler)
{
}

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
bool vtkMetalRenderer::HasTranslucentPolygonalGeometry()
{
  // Check if any visible prop has translucent geometry
  for (int i = 0; i < this->PropArrayCount; i++)
  {
    if (this->PropArray[i]->GetVisibility() &&
        this->PropArray[i]->HasTranslucentPolygonalGeometry())
    {
      return true;
    }
  }
  return false;
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

    EnsureDepthStencilStates(device);

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

    // Get viewport dimensions
    int* size = this->GetSize();
    double* viewport = this->GetViewport();

    // Determine if MSAA is active
    const bool msaa = (renWin->GetEffectiveSampleCount() > 1);
    id<MTLTexture> msaaColorTex = (__bridge id<MTLTexture>)renWin->MultisampleColorTexture;
    id<MTLTexture> msaaDepthTex = (__bridge id<MTLTexture>)renWin->MultisampleDepthTexture;

    // === Phase 1: Render opaque geometry ===
    {
      MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];

      double bgColor[3];
      this->GetBackground(bgColor);

      if (msaa && msaaColorTex)
      {
        rpd.colorAttachments[0].texture = msaaColorTex;
        rpd.colorAttachments[0].resolveTexture = drawable.texture;
        rpd.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
      }
      else
      {
        rpd.colorAttachments[0].texture = drawable.texture;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
      }
      rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
      rpd.colorAttachments[0].clearColor = MTLClearColorMake(bgColor[0], bgColor[1], bgColor[2], 1.0);

      // IDs texture for GPU picking (skip when MSAA active)
      if (!msaa)
      {
        id<MTLTexture> idsTex = (__bridge id<MTLTexture>)renWin->IdsTexture;
        if (idsTex)
        {
          rpd.colorAttachments[1].texture = idsTex;
          rpd.colorAttachments[1].loadAction = MTLLoadActionClear;
          rpd.colorAttachments[1].clearColor = MTLClearColorMake(0, 0, 0, 0);
          rpd.colorAttachments[1].storeAction = MTLStoreActionStore;
        }
      }

      // Depth attachment — always store for depth peeling and volume rendering
      // (volume mapper samples depth for early ray termination / depth occlusion)
      if (msaa && msaaDepthTex)
      {
        rpd.depthAttachment.texture = msaaDepthTex;
        rpd.depthAttachment.loadAction = MTLLoadActionClear;
        rpd.depthAttachment.clearDepth = 1.0;
        rpd.depthAttachment.storeAction = MTLStoreActionStore;
      }
      else
      {
        id<MTLTexture> depthTex = (__bridge id<MTLTexture>)renWin->DepthTexture;
        if (depthTex)
        {
          rpd.depthAttachment.texture = depthTex;
          rpd.depthAttachment.loadAction = MTLLoadActionClear;
          rpd.depthAttachment.clearDepth = 1.0;
          rpd.depthAttachment.storeAction = MTLStoreActionStore;
        }
      }

      id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:rpd];
      encoder.label = @"VTK Opaque Encoder";

      // Set depth stencil state
      id<MTLTexture> activeDepthTex = msaa ? msaaDepthTex :
        (__bridge id<MTLTexture>)renWin->DepthTexture;
      if (activeDepthTex)
      {
        [encoder setDepthStencilState:sOpaqueDepthState];
      }
      [encoder setFrontFacingWinding:MTLWindingCounterClockwise];

      renWin->CommandBuffer = (__bridge void*)commandBuffer;
      renWin->Encoder = (__bridge void*)encoder;

      // Set viewport
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

      // Create default headlight if none exist
      if (this->GetLights()->GetNumberOfItems() == 0 && this->AutomaticLightCreation)
      {
        this->CreateLight();
      }

      // Render opaque geometry
      this->UpdateOpaquePolygonalGeometry();

      [encoder endEncoding];
      renWin->Encoder = nullptr;
    }

    // === Phase 2: Render translucent geometry ===
    bool hasTranslucent = this->HasTranslucentPolygonalGeometry();

    // Resolve MSAA depth to regular depth texture for volume mapper sampling.
    // When translucent geometry exists, the translucent pass also resolves MSAA
    // depth (via StoreAndMultisampleResolve), so the standalone resolve is
    // only needed when there is no translucent geometry.
    if (msaa && msaaDepthTex && !hasTranslucent)
    {
      id<MTLTexture> depthTex = (__bridge id<MTLTexture>)renWin->DepthTexture;
      if (depthTex)
      {
        MTLRenderPassDescriptor* resolveRpd =
          [MTLRenderPassDescriptor renderPassDescriptor];
        resolveRpd.depthAttachment.texture = msaaDepthTex;
        resolveRpd.depthAttachment.resolveTexture = depthTex;
        resolveRpd.depthAttachment.loadAction = MTLLoadActionLoad;
        resolveRpd.depthAttachment.storeAction = MTLStoreActionStoreAndMultisampleResolve;

        id<MTLRenderCommandEncoder> depthResolve =
          [commandBuffer renderCommandEncoderWithDescriptor:resolveRpd];
        depthResolve.label = @"VTK MSAA Depth Resolve";
        [depthResolve endEncoding];
      }
    }

    // Depth peeling is incompatible with MSAA — the intermediate peel textures
    // are non-MSAA and cannot match an MSAA depth attachment.
    bool useDepthPeeling = this->GetUseDepthPeeling() && !msaa;

    auto renderStandardTranslucentPass = [&]()
    {
      if (!hasTranslucent)
      {
        return;
      }

      MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];

      if (msaa && msaaColorTex)
      {
        rpd.colorAttachments[0].texture = msaaColorTex;
        rpd.colorAttachments[0].resolveTexture = drawable.texture;
        rpd.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
      }
      else
      {
        rpd.colorAttachments[0].texture = drawable.texture;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
      }
      rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;  // preserve opaque rendering

      if (msaa && msaaDepthTex)
      {
        id<MTLTexture> depthTex = (__bridge id<MTLTexture>)renWin->DepthTexture;
        rpd.depthAttachment.texture = msaaDepthTex;
        rpd.depthAttachment.loadAction = MTLLoadActionLoad;
        if (depthTex)
        {
          rpd.depthAttachment.resolveTexture = depthTex;
          rpd.depthAttachment.storeAction = MTLStoreActionStoreAndMultisampleResolve;
        }
        else
        {
          rpd.depthAttachment.storeAction = MTLStoreActionDontCare;
        }
      }
      else
      {
        id<MTLTexture> depthTex = (__bridge id<MTLTexture>)renWin->DepthTexture;
        if (depthTex)
        {
          rpd.depthAttachment.texture = depthTex;
          rpd.depthAttachment.loadAction = MTLLoadActionLoad;
          rpd.depthAttachment.storeAction = MTLStoreActionStore;
        }
      }

      id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:rpd];
      encoder.label = @"VTK Translucent Encoder";

      // Depth test: Less (match opaque), depth write: No
      id<MTLTexture> activeDepthTex = msaa ? msaaDepthTex :
        (__bridge id<MTLTexture>)renWin->DepthTexture;
      if (activeDepthTex)
      {
        [encoder setDepthStencilState:sReadOnlyDepthState];
      }
      [encoder setFrontFacingWinding:MTLWindingCounterClockwise];

      renWin->CommandBuffer = (__bridge void*)commandBuffer;
      renWin->Encoder = (__bridge void*)encoder;

      MTLViewport metalViewport;
      metalViewport.originX = viewport[0] * size[0];
      metalViewport.originY = viewport[1] * size[1];
      metalViewport.width = viewport[2] * size[0];
      metalViewport.height = viewport[3] * size[1];
      metalViewport.znear = 0.0;
      metalViewport.zfar = 1.0;
      [encoder setViewport:metalViewport];

      this->UpdateTranslucentPolygonalGeometry();

      [encoder endEncoding];
      renWin->Encoder = nullptr;
    };

    if (hasTranslucent && useDepthPeeling)
    {
      if (this->MaximumNumberOfPeels <= 0)
      {
        renderStandardTranslucentPass();
      }
      else
      {
        // Use depth peeling for correct order-independent transparency
        // Get the resolved depth texture from the opaque pass
        // (msaa is always false here because depth peeling is disabled under MSAA)
        id<MTLTexture> depthTex = (__bridge id<MTLTexture>)renWin->DepthTexture;

        this->DepthPeeler->SetMaximumNumberOfPeels(this->MaximumNumberOfPeels);
        int peels = this->DepthPeeler->RenderTranslucentGeometry(
          this, commandBuffer, drawable.texture, depthTex);

        // If depth peeling produced no results (e.g. nil depth texture), fall
        // back to standard alpha-blended translucent rendering (with encoder).
        // Peel textures are re-created on the next RenderTranslucentGeometry call.
        if (peels == 0)
        {
          renderStandardTranslucentPass();
        }
      }
    }
    else if (hasTranslucent)
    {
      renderStandardTranslucentPass();
    }

    // === Phase 3: Render volumetric geometry ===
    if (!this->UseDepthPeelingForVolumes)
    {
      MTLRenderPassDescriptor* rpd =
        [MTLRenderPassDescriptor renderPassDescriptor];

      if (msaa && msaaColorTex)
      {
        rpd.colorAttachments[0].texture = msaaColorTex;
        rpd.colorAttachments[0].resolveTexture = drawable.texture;
        rpd.colorAttachments[0].storeAction =
          MTLStoreActionMultisampleResolve;
      }
      else
      {
        rpd.colorAttachments[0].texture = drawable.texture;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
      }
      rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;

      if (msaa && msaaDepthTex)
      {
        rpd.depthAttachment.texture = msaaDepthTex;
        rpd.depthAttachment.loadAction = MTLLoadActionLoad;
        rpd.depthAttachment.storeAction = MTLStoreActionDontCare;
      }
      else
      {
        id<MTLTexture> depthTex =
          (__bridge id<MTLTexture>)renWin->DepthTexture;
        if (depthTex)
        {
          rpd.depthAttachment.texture = depthTex;
          rpd.depthAttachment.loadAction = MTLLoadActionLoad;
          rpd.depthAttachment.storeAction = MTLStoreActionDontCare;
        }
      }

      id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:rpd];
      encoder.label = @"VTK Volume Encoder";

      // Depth test: Less, depth write: Off (volumes are translucent)
      id<MTLTexture> activeDepthTex = msaa ? msaaDepthTex
        : (__bridge id<MTLTexture>)renWin->DepthTexture;
      if (activeDepthTex)
      {
        [encoder setDepthStencilState:sReadOnlyDepthState];
      }
      [encoder setFrontFacingWinding:MTLWindingCounterClockwise];

      renWin->CommandBuffer = (__bridge void*)commandBuffer;
      renWin->Encoder = (__bridge void*)encoder;

      MTLViewport metalViewport;
      metalViewport.originX = viewport[0] * size[0];
      metalViewport.originY = viewport[1] * size[1];
      metalViewport.width = viewport[2] * size[0];
      metalViewport.height = viewport[3] * size[1];
      metalViewport.znear = 0.0;
      metalViewport.zfar = 1.0;
      [encoder setViewport:metalViewport];

      for (int i = 0; i < this->PropArrayCount; i++)
      {
        this->NumberOfPropsRendered +=
          this->PropArray[i]->RenderVolumetricGeometry(this);
      }

      // A volume mapper (image-sample path) may have ended this encoder
      // and replaced it. Only end if it's still the same encoder.
      if (renWin->GetCurrentRenderCommandEncoder() == (__bridge void*)encoder)
      {
        [encoder endEncoding];
      }
      renWin->SetCurrentRenderCommandEncoder(nullptr);
    }

    // === Phase 3b: Blit image-sampled volumes to screen ===
    // After all volumes are rendered, check if any volume mapper used
    // image-space downsampling (ImageSampleDistance != 1.0). If so, blit
    // the offscreen textures to the drawable.
    {
      vtkVolumeCollection* coll = this->GetVolumes();
      coll->InitTraversal();
      bool needsBlit = false;

      while (vtkVolume* vol = coll->GetNextVolume())
      {
        vtkGPUVolumeRayCastMapper* volMapper =
          vtkGPUVolumeRayCastMapper::SafeDownCast(vol->GetMapper());
        vtkMetalGPUVolumeRayCastMapper* metalMapper =
          vtkMetalGPUVolumeRayCastMapper::SafeDownCast(volMapper);
        if (metalMapper && metalMapper->GetImageSampleColorTexture() &&
            metalMapper->GetImageSampleWidth() > 0)
        {
          needsBlit = true;
          break;
        }
      }

      if (needsBlit)
      {
        // Create blit pipeline if needed
        static id<MTLRenderPipelineState> blitPipeline = nil;
        if (!blitPipeline)
        {
          @autoreleasepool
          {
            NSError* error = nil;
            NSString* shaderSource = [NSString stringWithUTF8String:vtkMetalShaders];
            id<MTLLibrary> library = [device newLibraryWithSource:shaderSource
                                                         options:nil
                                                           error:&error];
            if (library)
            {
              id<MTLFunction> vFunc = [library newFunctionWithName:@"vertex_fullscreen_main"];
              id<MTLFunction> fFunc = [library newFunctionWithName:@"fragment_image_sample_blit"];
              if (vFunc && fFunc)
              {
                MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
                desc.vertexFunction = vFunc;
                desc.fragmentFunction = fFunc;
                desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
                desc.colorAttachments[0].blendingEnabled = YES;
                desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
                desc.colorAttachments[0].destinationRGBBlendFactor =
                  MTLBlendFactorOneMinusSourceAlpha;
                desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
                desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
                desc.colorAttachments[0].destinationAlphaBlendFactor =
                  MTLBlendFactorOneMinusSourceAlpha;
                desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
                desc.depthAttachmentPixelFormat = MTLPixelFormatInvalid;

                blitPipeline = [device newRenderPipelineStateWithDescriptor:desc
                                                                     error:&error];
                [desc release];
              }
              [vFunc release];
              [fFunc release];
            }
            [library release];
          }
        }

        if (blitPipeline)
        {
          // Create sampler for blit
          static id<MTLSamplerState> blitSampler = nil;
          if (!blitSampler)
          {
            MTLSamplerDescriptor* sDesc = [[MTLSamplerDescriptor alloc] init];
            sDesc.minFilter = MTLSamplerMinMagFilterLinear;
            sDesc.magFilter = MTLSamplerMinMagFilterLinear;
            sDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
            sDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
            blitSampler = [device newSamplerStateWithDescriptor:sDesc];
            [sDesc release];
          }

          // Create blit encoder
          MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
          if (msaa && msaaColorTex)
          {
            rpd.colorAttachments[0].texture = msaaColorTex;
            rpd.colorAttachments[0].resolveTexture = drawable.texture;
            rpd.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
          }
          else
          {
            rpd.colorAttachments[0].texture = drawable.texture;
            rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
          }
          rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;

          id<MTLRenderCommandEncoder> blitEncoder =
            [commandBuffer renderCommandEncoderWithDescriptor:rpd];
          blitEncoder.label = @"VTK Volume ImageSample Blit";

          [blitEncoder setRenderPipelineState:blitPipeline];
          [blitEncoder setCullMode:MTLCullModeNone];

          // Blit each volume's offscreen texture
          coll->InitTraversal();
          while (vtkVolume* vol = coll->GetNextVolume())
          {
            vtkGPUVolumeRayCastMapper* volMapper =
              vtkGPUVolumeRayCastMapper::SafeDownCast(vol->GetMapper());
            vtkMetalGPUVolumeRayCastMapper* metalMapper =
              vtkMetalGPUVolumeRayCastMapper::SafeDownCast(volMapper);
            if (metalMapper && metalMapper->GetImageSampleColorTexture() &&
                metalMapper->GetImageSampleWidth() > 0)
            {
              id<MTLTexture> offscreenTex =
                (__bridge id<MTLTexture>)metalMapper->GetImageSampleColorTexture();

              // Set viewport to full window
              MTLViewport metalViewport;
              metalViewport.originX = viewport[0] * size[0];
              metalViewport.originY = viewport[1] * size[1];
              metalViewport.width = viewport[2] * size[0];
              metalViewport.height = viewport[3] * size[1];
              metalViewport.znear = 0.0;
              metalViewport.zfar = 1.0;
              [blitEncoder setViewport:metalViewport];

              [blitEncoder setFragmentTexture:offscreenTex atIndex:0];
              [blitEncoder setFragmentSamplerState:blitSampler atIndex:0];

              [blitEncoder drawPrimitives:MTLPrimitiveTypeTriangle
                             vertexStart:0
                             vertexCount:3];
            }
          }

          [blitEncoder endEncoding];
        }
      }
    }

    // Commit and present
    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];

    // Notify application of GPU completion (benchmarking etc.)
    if (renWin->RenderCompletionCallback)
    {
      [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        double gpuMs = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        dispatch_async(dispatch_get_main_queue(), ^{
          VTKRenderCompletionBlock block =
            (__bridge VTKRenderCompletionBlock)renWin->RenderCompletionCallback;
          if (block)
          {
            block(gpuMs);
          }
        });
      }];
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
  if (this->DepthPeeler)
  {
    this->DepthPeeler->Release();
  }
  this->Superclass::ReleaseGraphicsResources(w);
}

//------------------------------------------------------------------------------
void vtkMetalRenderer::RenderTranslucentGeometry()
{
  this->UpdateTranslucentPolygonalGeometry();
}

VTK_ABI_NAMESPACE_END
