// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalTemporalUpscaler.h"

#import <MetalFX/MetalFX.h>
#include <cmath>

VTK_ABI_NAMESPACE_BEGIN

//------------------------------------------------------------------------------
vtkMetalTemporalUpscaler::vtkMetalTemporalUpscaler() = default;

//------------------------------------------------------------------------------
vtkMetalTemporalUpscaler::~vtkMetalTemporalUpscaler()
{
  this->Release();
}

//------------------------------------------------------------------------------
bool vtkMetalTemporalUpscaler::SupportsDevice(id<MTLDevice> device)
{
  if (!device)
  {
    return false;
  }
  return [MTLFXTemporalScalerDescriptor supportsDevice:device];
}

//------------------------------------------------------------------------------
void vtkMetalTemporalUpscaler::Initialize(id<MTLDevice> device, int inputW, int inputH,
                                           int outputW, int outputH)
{
  if (!device || inputW <= 0 || inputH <= 0 || outputW <= 0 || outputH <= 0)
  {
    return;
  }

  // Skip if already initialized at the same resolution
  if (this->TemporalScaler && this->CurrentInputWidth == inputW &&
    this->CurrentInputHeight == inputH && this->CurrentOutputWidth == outputW &&
    this->CurrentOutputHeight == outputH)
  {
    return;
  }

  this->Release();

  @autoreleasepool
  {
    MTLFXTemporalScalerDescriptor* desc = [[MTLFXTemporalScalerDescriptor alloc] init];

    // Input dimensions (render resolution)
    desc.inputWidth = inputW;
    desc.inputHeight = inputH;

    // Output dimensions (display resolution)
    desc.outputWidth = outputW;
    desc.outputHeight = outputH;

    // Texture formats — must match what we create in the render window
    desc.colorTextureFormat = MTLPixelFormatRGBA16Float;
    desc.depthTextureFormat = MTLPixelFormatDepth32Float;
    desc.motionTextureFormat = MTLPixelFormatRG16Float;
    desc.outputTextureFormat = MTLPixelFormatRGBA16Float;

    // Auto exposure — MetalFX handles exposure calculation
    desc.autoExposureEnabled = YES;

    // Synchronous initialization — avoid hitch on first frame
    desc.requiresSynchronousInitialization = YES;

    // Create the temporal scaler
    this->TemporalScaler = [desc newTemporalScalerWithDevice:device];
    if (!this->TemporalScaler)
    {
      return;
    }

    // Set motion vector scale: MetalFX expects motion in render-resolution pixel space.
    this->TemporalScaler.motionVectorScaleX = this->MotionVectorScaleX;
    this->TemporalScaler.motionVectorScaleY = this->MotionVectorScaleY;

    this->CurrentInputWidth = inputW;
    this->CurrentInputHeight = inputH;
    this->CurrentOutputWidth = outputW;
    this->CurrentOutputHeight = outputH;
  }
}

//------------------------------------------------------------------------------
void vtkMetalTemporalUpscaler::Release()
{
  this->TemporalScaler = nil;
  this->CurrentInputWidth = 0;
  this->CurrentInputHeight = 0;
  this->CurrentOutputWidth = 0;
  this->CurrentOutputHeight = 0;
}

//------------------------------------------------------------------------------
void vtkMetalTemporalUpscaler::Encode(id<MTLCommandBuffer> commandBuffer,
                                       id<MTLTexture> colorTexture,
                                       id<MTLTexture> depthTexture,
                                       id<MTLTexture> motionTexture,
                                       id<MTLTexture> outputTexture,
                                       float jitterX, float jitterY,
                                       bool resetHistory)
{
  if (!this->TemporalScaler || !commandBuffer)
  {
    return;
  }

  // Set per-frame properties
  this->TemporalScaler.colorTexture = colorTexture;
  this->TemporalScaler.depthTexture = depthTexture;
  this->TemporalScaler.motionTexture = motionTexture;
  this->TemporalScaler.outputTexture = outputTexture;

  this->TemporalScaler.inputContentWidth = this->CurrentInputWidth;
  this->TemporalScaler.inputContentHeight = this->CurrentInputHeight;

  this->TemporalScaler.jitterOffsetX = jitterX;
  this->TemporalScaler.jitterOffsetY = jitterY;
  this->TemporalScaler.reset = resetHistory;

  // Encode the upscaling effect
  [this->TemporalScaler encodeToCommandBuffer:commandBuffer];
}

//------------------------------------------------------------------------------
void vtkMetalTemporalUpscaler::GetJitterOffset(int frameIndex, float& outX, float& outY) const
{
  // Halton(2,3) sequence — produces well-distributed sample points.
  // 32 samples gives ~8 samples per output pixel for 2× upscaling.
  const int sequenceLength = 32;
  int idx = frameIndex % sequenceLength;

  // Halton base-2 component
  float halton2 = 0.0f;
  float f = 0.5f;
  int n = idx + 1; // 1-indexed for Halton sequence
  while (n > 0)
  {
    halton2 += f * (float)(n % 2);
    f *= 0.5f;
    n /= 2;
  }

  // Halton base-3 component
  float halton3 = 0.0f;
  f = 1.0f / 3.0f;
  n = idx + 1;
  while (n > 0)
  {
    halton3 += f * (float)(n % 3);
    f *= (1.0f / 3.0f);
    n /= 3;
  }

  // Map from [0,1) to [-0.5, 0.5)
  outX = halton2 - 0.5f;
  outY = halton3 - 0.5f;
}

//------------------------------------------------------------------------------
void vtkMetalTemporalUpscaler::SetMotionVectorScale(float sx, float sy)
{
  this->MotionVectorScaleX = sx;
  this->MotionVectorScaleY = sy;

  if (this->TemporalScaler)
  {
    this->TemporalScaler.motionVectorScaleX = sx;
    this->TemporalScaler.motionVectorScaleY = sy;
  }
}

//------------------------------------------------------------------------------
bool vtkMetalTemporalUpscaler::IsInitialized() const
{
  return this->TemporalScaler != nil;
}

VTK_ABI_NAMESPACE_END
