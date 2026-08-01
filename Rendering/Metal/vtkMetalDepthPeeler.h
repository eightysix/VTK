// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkMetalDepthPeeler
 * @brief   Depth peeling for order-independent transparency on Metal
 *
 * vtkMetalDepthPeeler implements dual depth peeling for correct translucent
 * rendering on Apple's Metal API. It manages the peeling textures, pipeline
 * states for fullscreen passes, and orchestrates the multi-pass rendering
 * algorithm.
 *
 * The algorithm follows vtkDualDepthPeelingPass (OpenGL2) but adapted for
 * Metal's render pass and blend state model:
 *   1. Render opaque geometry (handled by the renderer)
 *   2. Initialize depth range from translucent geometry (MAX blending)
 *   3. Peel loop: for each iteration, peel front and back fragments
 *   4. Composite front and back accumulation buffers
 *
 * Key difference from OpenGL2: instead of stalling the GPU with per-peel
 * occlusion queries, the back-blend pass counts the fragments it writes via
 * MTLRenderPassDescriptor.visibilityResultMode (MTLVisibilityResultModeCounting).
 * The count is read back once the frame's command buffer completes, and the
 * next frame runs only as many peels as were observed to write fragments
 * (plus a safety margin), bounded by the maximum peel count. This converges
 * to the scene's actual depth complexity while never stalling the pipeline.
 */

#ifndef vtkMetalDepthPeeler_h
#define vtkMetalDepthPeeler_h

#include "vtkRenderingMetalModule.h"

#import <Metal/Metal.h>

VTK_ABI_NAMESPACE_BEGIN

class vtkMetalRenderer;
class vtkMetalRenderWindow;

class VTKRENDERINGMETAL_EXPORT vtkMetalDepthPeeler
{
public:
  vtkMetalDepthPeeler();
  ~vtkMetalDepthPeeler();

  /**
   * Check if peeling textures need to be (re)created for the given size.
   */
  bool NeedsInitialization(int width, int height) const;

  /**
   * Create or recreate peeling textures for the given viewport size.
   */
  void Initialize(id<MTLDevice> device, int width, int height);

  /**
   * Release all GPU resources.
   */
  void Release();

  /**
   * Run depth peeling for all translucent actors.
   * Returns the number of peel iterations performed.
   *
   * The opaque geometry must already be rendered and the depth buffer
   * must contain opaque depths. This method:
   *   1. Initializes the depth range via MAX blending
   *   2. Runs the peel loop
   *   3. Composites the result onto the drawable
   */
  int RenderTranslucentGeometry(
    vtkMetalRenderer* renderer,
    id<MTLCommandBuffer> commandBuffer,
    id<MTLTexture> drawableTexture,
    id<MTLTexture> depthTexture);

  /**
   * Maximum number of peel iterations (default 8).
   */
  void SetMaximumNumberOfPeels(int count) { this->MaximumNumberOfPeels = count; }
  int GetMaximumNumberOfPeels() const { return this->MaximumNumberOfPeels; }

private:
  vtkMetalDepthPeeler(const vtkMetalDepthPeeler&) = delete;
  void operator=(const vtkMetalDepthPeeler&) = delete;

  void CreateTextures(id<MTLDevice> device, int width, int height);
  void CreatePipelines(vtkMetalRenderWindow* renWin);

  // Fullscreen pass pipeline states
  id<MTLRenderPipelineState> CompositePipeline = nil;
  id<MTLRenderPipelineState> BackBlendPipeline = nil;
  // Peeling textures
  id<MTLTexture> FrontPeelA = nil;     // RGBA8 — front accumulation ping-pong A
  id<MTLTexture> FrontPeelB = nil;     // RGBA8 — front accumulation ping-pong B
  id<MTLTexture> BackPeelTemp = nil;   // RGBA8 — back fragment per peel iteration
  id<MTLTexture> BackAccum = nil;      // RGBA8 — back-to-front accumulation
  id<MTLTexture> DepthPeelA = nil;     // RG32Float — min/max depth ping-pong A
  id<MTLTexture> DepthPeelB = nil;     // RG32Float — min/max depth ping-pong B

  // Cached depth-stencil states (immutable, created once)
  id<MTLDepthStencilState> ReadOnlyDepthState = nil;   // Less, write=NO
  id<MTLDepthStencilState> AlwaysDepthState = nil;     // Always, write=NO

  // State
  int CurrentWidth = 0;
  int CurrentHeight = 0;
  int MaximumNumberOfPeels = 8;

  // Visibility buffer: each peel's back-blend pass counts the fragments it
  // wrote (MTLVisibilityResultModeCounting) into this shared buffer. The
  // count is read back after the command buffer completes and feeds
  // NextPeelCount, so the next frame stops early once no more fragments are
  // being written. Shared storage lets the CPU read it without a blit.
  id<MTLBuffer> VisibilityBuffer = nil;
  int NextPeelCount = 0; // 0 means "unknown, run all MaximumNumberOfPeels"

  // Previous frame's command buffer; its status tells whether VisibilityBuffer
  // holds a completed frame's counts before we read them.
  id<MTLCommandBuffer> LastCommandBuffer = nil;

  bool PipelinesCreated = false;
};

VTK_ABI_NAMESPACE_END
#endif // vtkMetalDepthPeeler_h
