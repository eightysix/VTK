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
        MTLDepthStencilDescriptor* dsDesc = [[MTLDepthStencilDescriptor alloc] init];
        dsDesc.depthCompareFunction = MTLCompareFunctionLess;
        dsDesc.depthWriteEnabled = YES;
        id<MTLDepthStencilState> depthState = [device newDepthStencilStateWithDescriptor:dsDesc];
        [encoder setDepthStencilState:depthState];
      }

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

    if (hasTranslucent && this->GetUseDepthPeeling())
    {
      // Use depth peeling for correct order-independent transparency
      // Get the depth texture used in the opaque pass
      id<MTLTexture> depthTex = msaa ? msaaDepthTex :
        (__bridge id<MTLTexture>)renWin->DepthTexture;

      this->DepthPeeler->SetMaximumNumberOfPeels(this->MaximumNumberOfPeels);
      this->DepthPeeler->RenderTranslucentGeometry(
        this, commandBuffer, drawable.texture, depthTex);
    }
    else if (hasTranslucent)
    {
      // Fallback: simple alpha blending (no depth peeling)
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
        rpd.depthAttachment.texture = msaaDepthTex;
        rpd.depthAttachment.loadAction = MTLLoadActionLoad;
        rpd.depthAttachment.storeAction = MTLStoreActionDontCare;
      }
      else
      {
        id<MTLTexture> depthTex = (__bridge id<MTLTexture>)renWin->DepthTexture;
        if (depthTex)
        {
          rpd.depthAttachment.texture = depthTex;
          rpd.depthAttachment.loadAction = MTLLoadActionLoad;
          rpd.depthAttachment.storeAction = MTLStoreActionDontCare;
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
        MTLDepthStencilDescriptor* dsDesc = [[MTLDepthStencilDescriptor alloc] init];
        dsDesc.depthCompareFunction = MTLCompareFunctionLess;
        dsDesc.depthWriteEnabled = NO;
        id<MTLDepthStencilState> depthState = [device newDepthStencilStateWithDescriptor:dsDesc];
        [encoder setDepthStencilState:depthState];
      }

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
    }

    // Resolve MSAA depth to regular depth texture for volume mapper sampling.
    // The volume shader samples the depth buffer for early ray termination,
    // but multisampled textures cannot be sampled in fragment shaders.
    if (msaa && msaaDepthTex)
    {
      id<MTLTexture> depthTex = (__bridge id<MTLTexture>)renWin->DepthTexture;
      if (depthTex)
      {
        MTLRenderPassDescriptor* resolveRpd =
          [MTLRenderPassDescriptor renderPassDescriptor];
        resolveRpd.depthAttachment.texture = msaaDepthTex;
        resolveRpd.depthAttachment.resolveTexture = depthTex;
        resolveRpd.depthAttachment.loadAction = MTLLoadActionLoad;
        resolveRpd.depthAttachment.storeAction = MTLStoreActionMultisampleResolve;
        resolveRpd.depthAttachment.clearDepth = 1.0;

        id<MTLRenderCommandEncoder> depthResolve =
          [commandBuffer renderCommandEncoderWithDescriptor:resolveRpd];
        depthResolve.label = @"VTK MSAA Depth Resolve for Volume";
        [depthResolve endEncoding];
      }
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
        MTLDepthStencilDescriptor* dsDesc =
          [[MTLDepthStencilDescriptor alloc] init];
        dsDesc.depthCompareFunction = MTLCompareFunctionLess;
        dsDesc.depthWriteEnabled = NO;
        id<MTLDepthStencilState> depthState =
          [device newDepthStencilStateWithDescriptor:dsDesc];
        [encoder setDepthStencilState:depthState];
      }

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
              }
            }
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
