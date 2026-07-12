// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalRenderer.h"

#include "vtkMetalRenderWindow.h"
#include "vtkMetalDepthPeeler.h"
#include "vtkMetalTemporalUpscaler.h"
#include "vtkMetalCamera.h"
#include "vtkMetalShaders.h"
#include "vtkObjectFactory.h"
#include "vtkRenderer.h"
#include "vtkRendererCollection.h"
#include "vtkLightCollection.h"
#include "vtkViewport.h"
#include "vtkActor.h"
#include "vtkActorCollection.h"
#include "vtkMatrix4x4.h"

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalRenderer);

//------------------------------------------------------------------------------
vtkMetalRenderer::vtkMetalRenderer()
  : DepthPeeler(new vtkMetalDepthPeeler),
    TemporalUpscaler(new vtkMetalTemporalUpscaler)
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
// Compute 4x4 matrix inverse using the adjugate method.
// Input/output are column-major float[16] arrays matching Metal's matrix layout.
static bool ComputeMatrixInverse4x4(const float* m, float* inv)
{
  // Compute 2x2 sub-determinants for cofactor expansion
  float s0 = m[0] * m[5] - m[4] * m[1];
  float s1 = m[0] * m[6] - m[4] * m[2];
  float s2 = m[0] * m[7] - m[4] * m[3];
  float s3 = m[1] * m[6] - m[5] * m[2];
  float s4 = m[1] * m[7] - m[5] * m[3];
  float s5 = m[2] * m[7] - m[6] * m[3];

  float c5 = m[10] * m[15] - m[14] * m[11];
  float c4 = m[9] * m[15] - m[13] * m[11];
  float c3 = m[9] * m[14] - m[13] * m[10];
  float c2 = m[8] * m[15] - m[12] * m[11];
  float c1 = m[8] * m[14] - m[12] * m[10];
  float c0 = m[8] * m[13] - m[12] * m[9];

  float det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
  if (fabsf(det) < 1e-10f)
  {
    return false;
  }

  float invDet = 1.0f / det;

  inv[0] = (m[5] * c5 - m[6] * c4 + m[7] * c3) * invDet;
  inv[1] = (-m[1] * c5 + m[2] * c4 - m[3] * c3) * invDet;
  inv[2] = (m[13] * s5 - m[14] * s4 + m[15] * s3) * invDet;
  inv[3] = (-m[9] * s5 + m[10] * s4 - m[11] * s3) * invDet;

  inv[4] = (-m[4] * c5 + m[6] * c2 - m[7] * c1) * invDet;
  inv[5] = (m[0] * c5 - m[2] * c2 + m[3] * c1) * invDet;
  inv[6] = (-m[12] * s5 + m[14] * s2 - m[15] * s1) * invDet;
  inv[7] = (m[8] * s5 - m[10] * s2 + m[11] * s1) * invDet;

  inv[8] = (m[4] * c4 - m[5] * c2 + m[7] * c0) * invDet;
  inv[9] = (-m[0] * c4 + m[1] * c2 - m[3] * c0) * invDet;
  inv[10] = (m[12] * s4 - m[13] * s2 + m[15] * s0) * invDet;
  inv[11] = (-m[8] * s4 + m[9] * s2 - m[11] * s0) * invDet;

  inv[12] = (-m[4] * c3 + m[5] * c1 - m[6] * c0) * invDet;
  inv[13] = (m[0] * c3 - m[1] * c1 + m[2] * c0) * invDet;
  inv[14] = (-m[12] * s3 + m[13] * s1 - m[14] * s0) * invDet;
  inv[15] = (m[8] * s3 - m[9] * s1 + m[10] * s0) * invDet;

  return true;
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

    // Determine if temporal upscaling is active
    const bool temporalUpscale = renWin->IsTemporalUpscalingEnabled();

    // Determine effective render dimensions
    int renderW = temporalUpscale ? renWin->GetRenderResolutionWidth() : size[0];
    int renderH = temporalUpscale ? renWin->GetRenderResolutionHeight() : size[1];

    // Determine if MSAA is active (only for non-upscaling path)
    const bool msaa = !temporalUpscale && (renWin->GetEffectiveSampleCount() > 1);
    id<MTLTexture> msaaColorTex = (__bridge id<MTLTexture>)renWin->MultisampleColorTexture;
    id<MTLTexture> msaaDepthTex = (__bridge id<MTLTexture>)renWin->MultisampleDepthTexture;

    // Select render targets based on upscaling mode
    id<MTLTexture> colorTarget;
    id<MTLTexture> depthTarget;
    if (temporalUpscale)
    {
      colorTarget = (__bridge id<MTLTexture>)renWin->InternalColorTexture;
      depthTarget = (__bridge id<MTLTexture>)renWin->InternalDepthTexture;
    }
    else if (msaa && msaaColorTex)
    {
      colorTarget = msaaColorTex;
      depthTarget = msaaDepthTex;
    }
    else
    {
      colorTarget = drawable.texture;
      depthTarget = (__bridge id<MTLTexture>)renWin->DepthTexture;
    }

    if (!colorTarget || !depthTarget)
    {
      return;
    }

    // Initialize temporal upscaler if needed
    if (temporalUpscale && !this->TemporalUpscaler->IsInitialized())
    {
      this->TemporalUpscaler->Initialize(
        device, renderW, renderH, size[0], size[1]);
      float scale = (float)renderW / (float)size[0];
      this->TemporalUpscaler->SetMotionVectorScale(scale, scale);
    }

    // Get jitter offset for temporal upscaling
    float jitterX = 0.0f, jitterY = 0.0f;
    if (temporalUpscale)
    {
      this->TemporalUpscaler->GetJitterOffset(
        this->TemporalFrameIndex, jitterX, jitterY);
    }

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
        rpd.colorAttachments[0].texture = colorTarget;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
      }
      rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
      rpd.colorAttachments[0].clearColor = MTLClearColorMake(bgColor[0], bgColor[1], bgColor[2], 1.0);

      // IDs texture for GPU picking (skip when MSAA or temporal upscale active)
      if (!msaa && !temporalUpscale)
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

      // Depth attachment — store for potential depth peeling
      rpd.depthAttachment.texture = depthTarget;
      rpd.depthAttachment.loadAction = MTLLoadActionClear;
      rpd.depthAttachment.clearDepth = 1.0;
      bool needDepth = this->HasTranslucentPolygonalGeometry();
      rpd.depthAttachment.storeAction = needDepth ? MTLStoreActionStore : MTLStoreActionDontCare;

      id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:rpd];
      encoder.label = @"VTK Opaque Encoder";

      // Set depth stencil state
      MTLDepthStencilDescriptor* dsDesc = [[MTLDepthStencilDescriptor alloc] init];
      dsDesc.depthCompareFunction = MTLCompareFunctionLess;
      dsDesc.depthWriteEnabled = YES;
      id<MTLDepthStencilState> depthState = [device newDepthStencilStateWithDescriptor:dsDesc];
      [encoder setDepthStencilState:depthState];

      renWin->CommandBuffer = (__bridge void*)commandBuffer;
      renWin->Encoder = (__bridge void*)encoder;

      // Set viewport to render resolution
      MTLViewport metalViewport;
      metalViewport.originX = viewport[0] * renderW;
      metalViewport.originY = viewport[1] * renderH;
      metalViewport.width = viewport[2] * renderW;
      metalViewport.height = viewport[3] * renderH;
      metalViewport.znear = 0.0;
      metalViewport.zfar = 1.0;
      [encoder setViewport:metalViewport];

      // Update camera and set viewport
      if (this->ActiveCamera)
      {
        this->ActiveCamera->Render(this);

        // Apply jitter to projection matrix for temporal upscaling
        if (temporalUpscale)
        {
          vtkMetalCamera* metalCamera =
            vtkMetalCamera::SafeDownCast(this->ActiveCamera);
          if (metalCamera)
          {
            // GetCachedSceneTransforms() returns void* to SceneTransforms struct
            // Layout: ViewMatrix(64) + ProjectionMatrix(64) + ...
            // ProjectionMatrix starts at offset 64 (after ViewMatrix)
            float* transforms = static_cast<float*>(metalCamera->GetCachedSceneTransforms());
            float* projMatrix = transforms + 16; // Skip ViewMatrix (4x4 = 16 floats)

            double jitterOffsetX = jitterX / renderW;
            double jitterOffsetY = jitterY / renderH;

            // Apply jitter as a translation in clip space
            // projMatrix is column-major: [col][row], so column 3 is at indices 12-15
            projMatrix[12] += static_cast<float>(jitterOffsetX);
            projMatrix[13] += static_cast<float>(jitterOffsetY);
          }
        }

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
      this->DepthPeeler->SetMaximumNumberOfPeels(this->MaximumNumberOfPeels);
      this->DepthPeeler->RenderTranslucentGeometry(
        this, commandBuffer, colorTarget, depthTarget);
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
        rpd.colorAttachments[0].texture = colorTarget;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
      }
      rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;

      rpd.depthAttachment.texture = depthTarget;
      rpd.depthAttachment.loadAction = MTLLoadActionLoad;
      rpd.depthAttachment.storeAction = MTLStoreActionDontCare;

      id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:rpd];
      encoder.label = @"VTK Translucent Encoder";

      // Depth test: Less (match opaque), depth write: No
      MTLDepthStencilDescriptor* dsDesc = [[MTLDepthStencilDescriptor alloc] init];
      dsDesc.depthCompareFunction = MTLCompareFunctionLess;
      dsDesc.depthWriteEnabled = NO;
      id<MTLDepthStencilState> depthState = [device newDepthStencilStateWithDescriptor:dsDesc];
      [encoder setDepthStencilState:depthState];

      renWin->CommandBuffer = (__bridge void*)commandBuffer;
      renWin->Encoder = (__bridge void*)encoder;

      MTLViewport metalViewport;
      metalViewport.originX = viewport[0] * renderW;
      metalViewport.originY = viewport[1] * renderH;
      metalViewport.width = viewport[2] * renderW;
      metalViewport.height = viewport[3] * renderH;
      metalViewport.znear = 0.0;
      metalViewport.zfar = 1.0;
      [encoder setViewport:metalViewport];

      this->UpdateTranslucentPolygonalGeometry();

      [encoder endEncoding];
      renWin->Encoder = nullptr;
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
        rpd.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
      }
      else
      {
        rpd.colorAttachments[0].texture = colorTarget;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
      }
      rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;

      rpd.depthAttachment.texture = depthTarget;
      rpd.depthAttachment.loadAction = MTLLoadActionLoad;
      rpd.depthAttachment.storeAction = MTLStoreActionDontCare;

      id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:rpd];
      encoder.label = @"VTK Volume Encoder";

      // Depth test: Less, depth write: Off (volumes are translucent)
      MTLDepthStencilDescriptor* dsDesc =
        [[MTLDepthStencilDescriptor alloc] init];
      dsDesc.depthCompareFunction = MTLCompareFunctionLess;
      dsDesc.depthWriteEnabled = NO;
      id<MTLDepthStencilState> depthState =
        [device newDepthStencilStateWithDescriptor:dsDesc];
      [encoder setDepthStencilState:depthState];

      renWin->CommandBuffer = (__bridge void*)commandBuffer;
      renWin->Encoder = (__bridge void*)encoder;

      MTLViewport metalViewport;
      metalViewport.originX = viewport[0] * renderW;
      metalViewport.originY = viewport[1] * renderH;
      metalViewport.width = viewport[2] * renderW;
      metalViewport.height = viewport[3] * renderH;
      metalViewport.znear = 0.0;
      metalViewport.zfar = 1.0;
      [encoder setViewport:metalViewport];

      for (int i = 0; i < this->PropArrayCount; i++)
      {
        this->NumberOfPropsRendered +=
          this->PropArray[i]->RenderVolumetricGeometry(this);
      }

      [encoder endEncoding];
      renWin->Encoder = nullptr;
    }

    // === Phase 4: MetalFX Temporal Upscaling ===
    if (temporalUpscale)
    {
      id<MTLTexture> motionTex = (__bridge id<MTLTexture>)renWin->MotionVectorTexture;
      id<MTLTexture> outputTex = (__bridge id<MTLTexture>)renWin->UpscaleOutputTexture;

      if (motionTex && outputTex)
      {
        // Generate motion vectors using depth buffer and MVP matrices
        {
          MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
          rpd.colorAttachments[0].texture = motionTex;
          rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
          rpd.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
          rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

          id<MTLRenderCommandEncoder> encoder =
            [commandBuffer renderCommandEncoderWithDescriptor:rpd];
          encoder.label = @"VTK Motion Vector Encoder";

          // Set viewport to render resolution
          MTLViewport metalViewport;
          metalViewport.originX = 0;
          metalViewport.originY = 0;
          metalViewport.width = renderW;
          metalViewport.height = renderH;
          metalViewport.znear = 0.0;
          metalViewport.zfar = 1.0;
          [encoder setViewport:metalViewport];

          // Create pipeline for motion vector shader if needed
          if (!this->MotionVectorPipeline)
          {
            NSString* shaderSource = [NSString stringWithUTF8String:vtkMetalShaders];
            id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:nil];
            if (library)
            {
              id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_fullscreen_main"];
              id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragment_motion_vector"];
              if (vertexFunc && fragmentFunc)
              {
                MTLRenderPipelineDescriptor* pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
                pipelineDesc.vertexFunction = vertexFunc;
                pipelineDesc.fragmentFunction = fragmentFunc;
                pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRG16Float;
                this->MotionVectorPipeline = (__bridge void*)
                  [device newRenderPipelineStateWithDescriptor:pipelineDesc error:nil];
              }
            }
          }

          if (this->MotionVectorPipeline)
          {
            id<MTLRenderPipelineState> pipeline =
              (__bridge id<MTLRenderPipelineState>)this->MotionVectorPipeline;
            [encoder setRenderPipelineState:pipeline];

            // Get current MVP from camera (column-major)
            vtkMetalCamera* metalCamera =
              vtkMetalCamera::SafeDownCast(this->ActiveCamera);
            float currentMVP[16];
            float inverseMVP[16];
            if (metalCamera)
            {
              auto* transforms = static_cast<float*>(metalCamera->GetCachedSceneTransforms());
              // SceneTransforms layout: ViewMatrix(16) + ProjectionMatrix(16) + ...
              // MVP = Projection * View (column-major)
              float viewMatrix[16];
              float projMatrix[16];
              memcpy(viewMatrix, transforms, 16 * sizeof(float));
              memcpy(projMatrix, transforms + 16, 16 * sizeof(float));

              // Compute MVP = Projection * View (column-major multiplication)
              for (int col = 0; col < 4; col++)
              {
                for (int row = 0; row < 4; row++)
                {
                  currentMVP[col * 4 + row] =
                    projMatrix[0 * 4 + row] * viewMatrix[col * 4 + 0] +
                    projMatrix[1 * 4 + row] * viewMatrix[col * 4 + 1] +
                    projMatrix[2 * 4 + row] * viewMatrix[col * 4 + 2] +
                    projMatrix[3 * 4 + row] * viewMatrix[col * 4 + 3];
                }
              }

              // Compute inverse of current MVP
              ComputeMatrixInverse4x4(currentMVP, inverseMVP);
            }

            // Build uniform buffer
            struct {
              float inverseCurrentMVP[16];
              float previousMVP[16];
              float outputSize[2];
              float renderSize[2];
            } uniforms;
            memcpy(uniforms.inverseCurrentMVP, inverseMVP, 16 * sizeof(float));
            memcpy(uniforms.previousMVP, renWin->PreviousMVP, 16 * sizeof(float));
            uniforms.outputSize[0] = static_cast<float>(size[0]);
            uniforms.outputSize[1] = static_cast<float>(size[1]);
            uniforms.renderSize[0] = static_cast<float>(renderW);
            uniforms.renderSize[1] = static_cast<float>(renderH);

            [encoder setFragmentBytes:&uniforms length:sizeof(uniforms) atIndex:0];

            // Bind depth texture
            id<MTLTexture> depthTex = (__bridge id<MTLTexture>)renWin->InternalDepthTexture;
            [encoder setFragmentTexture:depthTex atIndex:0];

            // Create a nearest-neighbor sampler for depth
            MTLSamplerDescriptor* samplerDesc = [[MTLSamplerDescriptor alloc] init];
            samplerDesc.minFilter = MTLSamplerMinMagFilterNearest;
            samplerDesc.magFilter = MTLSamplerMinMagFilterNearest;
            samplerDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
            samplerDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
            id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:samplerDesc];
            [encoder setFragmentSamplerState:sampler atIndex:0];

            // Draw fullscreen triangle
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];

            // Store current MVP for next frame
            memcpy(renWin->PreviousMVP, currentMVP, 16 * sizeof(float));
            renWin->PreviousMVPValid = true;
          }

          [encoder endEncoding];
        }

        // Encode temporal upscaling
        bool resetHistory = !renWin->PreviousMVPValid;
        this->TemporalUpscaler->Encode(
          commandBuffer,
          colorTarget,         // jittered color at render resolution
          depthTarget,         // depth at render resolution
          motionTex,           // motion vectors at render resolution
          outputTex,           // upscaled output at output resolution
          jitterX, jitterY,
          resetHistory);

        // Blit upscaled result to drawable using a fullscreen render pass
        // (can't use blit encoder due to pixel format mismatch and framebufferOnly)
        {
          MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
          rpd.colorAttachments[0].texture = drawable.texture;
          rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;
          rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

          id<MTLRenderCommandEncoder> encoder =
            [commandBuffer renderCommandEncoderWithDescriptor:rpd];
          encoder.label = @"VTK Upscale Blit Encoder";

          // Create pipeline state for blit shader if needed
          if (!this->BlitPipeline)
          {
            NSString* shaderSource = [NSString stringWithUTF8String:vtkMetalShaders];
            id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:nil];
            if (library)
            {
              id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_fullscreen_main"];
              id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragment_blit"];
              if (vertexFunc && fragmentFunc)
              {
                MTLRenderPipelineDescriptor* pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
                pipelineDesc.vertexFunction = vertexFunc;
                pipelineDesc.fragmentFunction = fragmentFunc;
                pipelineDesc.colorAttachments[0].pixelFormat = drawable.texture.pixelFormat;
                this->BlitPipeline = (__bridge void*)
                  [device newRenderPipelineStateWithDescriptor:pipelineDesc error:nil];
              }
            }
          }

          if (this->BlitPipeline)
          {
            id<MTLRenderPipelineState> pipeline =
              (__bridge id<MTLRenderPipelineState>)this->BlitPipeline;
            [encoder setRenderPipelineState:pipeline];

            // Bind the upscaled texture
            [encoder setFragmentTexture:outputTex atIndex:0];

            // Create a linear sampler for the texture
            MTLSamplerDescriptor* samplerDesc = [[MTLSamplerDescriptor alloc] init];
            samplerDesc.minFilter = MTLSamplerMinMagFilterLinear;
            samplerDesc.magFilter = MTLSamplerMinMagFilterLinear;
            samplerDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
            samplerDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
            id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:samplerDesc];
            [encoder setFragmentSamplerState:sampler atIndex:0];

            // Draw fullscreen triangle (3 vertices, no geometry needed)
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
          }

          [encoder endEncoding];
        }
      }
    }

    // Commit and present
    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];

    // Update temporal upscaling state
    if (temporalUpscale)
    {
      this->TemporalFrameIndex++;
    }
    else
    {
      this->TemporalFrameIndex = 0;
      renWin->PreviousMVPValid = false;
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
  if (this->TemporalUpscaler)
  {
    this->TemporalUpscaler->Release();
  }
  this->TemporalFrameIndex = 0;

  // Reset render window temporal upscaling state
  vtkMetalRenderWindow* renWin = vtkMetalRenderWindow::SafeDownCast(w);
  if (renWin)
  {
    renWin->PreviousMVPValid = false;
  }

  this->Superclass::ReleaseGraphicsResources(w);
}

//------------------------------------------------------------------------------
void vtkMetalRenderer::RenderTranslucentGeometry()
{
  this->UpdateTranslucentPolygonalGeometry();
}

VTK_ABI_NAMESPACE_END
