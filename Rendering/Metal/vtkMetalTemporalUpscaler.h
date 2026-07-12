// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkMetalTemporalUpscaler
 * @brief   MetalFX temporal upscaling for the Metal rendering backend
 *
 * vtkMetalTemporalUpscaler manages Apple's MetalFX temporal anti-aliasing
 * and upscaling effect. It renders geometry at a lower resolution and
 * upscales to full resolution with high quality, significantly improving
 * GPU performance for complex scenes.
 *
 * The temporal scaler requires:
 *   - Color input: RGBA16Float at render resolution
 *   - Depth input: Depth32Float at render resolution
 *   - Motion vectors: RG16Float at render resolution
 *   - Output: RGBA16Float at output resolution
 *
 * MetalFX maintains an internal history buffer, so only the current frame's
 * textures need to be provided. The scaler blends current and previous frame
 * samples to produce anti-aliased upscaled output.
 *
 * Available: macOS 13.0+, iOS 16.0+
 */

#ifndef vtkMetalTemporalUpscaler_h
#define vtkMetalTemporalUpscaler_h

#include "vtkRenderingMetalModule.h"

#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

VTK_ABI_NAMESPACE_BEGIN

class VTKRENDERINGMETAL_EXPORT vtkMetalTemporalUpscaler
{
public:
  vtkMetalTemporalUpscaler();
  ~vtkMetalTemporalUpscaler();

  /**
   * Check if the device supports MetalFX temporal upscaling.
   * Returns false on macOS < 13 or iOS < 16, or if the GPU is unsupported.
   */
  static bool SupportsDevice(id<MTLDevice> device);

  /**
   * Initialize or recreate the temporal scaler for the given resolutions.
   * Call when the window size or scale factor changes.
   * This is expensive — only call on resize or first setup.
   */
  void Initialize(id<MTLDevice> device, int inputW, int inputH, int outputW, int outputH);

  /**
   * Release all MetalFX resources.
   */
  void Release();

  /**
   * Encode the temporal upscaling effect into the command buffer.
   *
   * @param commandBuffer  The command buffer to encode into
   * @param colorTexture   Current frame's jittered color (RGBA16Float, input res)
   * @param depthTexture   Current frame's depth (Depth32Float, input res)
   * @param motionTexture  Motion vectors (RG16Float, input res)
   * @param outputTexture  Destination for upscaled output (RGBA16Float, output res)
   * @param jitterX        Horizontal jitter offset [-0.5, 0.5]
   * @param jitterY        Vertical jitter offset [-0.5, 0.5]
   * @param resetHistory   True on first frame or scene cut (clears temporal history)
   */
  void Encode(id<MTLCommandBuffer> commandBuffer,
              id<MTLTexture> colorTexture,
              id<MTLTexture> depthTexture,
              id<MTLTexture> motionTexture,
              id<MTLTexture> outputTexture,
              float jitterX, float jitterY,
              bool resetHistory);

  /**
   * Get the jitter offset for a given frame index.
   * Uses Halton(2,3) sequence with 32 samples (recommended for 2× upscaling).
   * Offsets are in the range [-0.5, 0.5].
   */
  void GetJitterOffset(int frameIndex, float& outX, float& outY) const;

  /**
   * Set the motion vector scale factors.
   * These scale the motion data to MetalFX's expected pixel-space units.
   * Default: (1.0, 1.0) — set to (inputW/outputW, inputH/outputH) for proper scaling.
   */
  void SetMotionVectorScale(float sx, float sy);

  /**
   * Whether the scaler has been successfully initialized.
   */
  bool IsInitialized() const;

private:
  vtkMetalTemporalUpscaler(const vtkMetalTemporalUpscaler&) = delete;
  void operator=(const vtkMetalTemporalUpscaler&) = delete;

  id<MTLFXTemporalScaler> TemporalScaler = nil;
  int CurrentInputWidth = 0;
  int CurrentInputHeight = 0;
  int CurrentOutputWidth = 0;
  int CurrentOutputHeight = 0;
  float MotionVectorScaleX = 1.0f;
  float MotionVectorScaleY = 1.0f;
};

VTK_ABI_NAMESPACE_END
#endif // vtkMetalTemporalUpscaler_h
