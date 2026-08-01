// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkMetalOrderIndependentTranslucentPass
 * @brief   Order-independent transparency (OIT) for translucent rendering on Metal
 *
 * vtkMetalOrderIndependentTranslucentPass implements the OIT accumulation
 * algorithm used by vtkOrderIndependentTranslucentPass on OpenGL2, adapted for
 * Metal's render pass and blend state model:
 *   1. Render translucent geometry into an RGBA16F accumulation texture and an
 *      R16F revealage texture, testing against the resolved opaque depth buffer.
 *      The blend state mirrors
 *      glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ZERO, GL_ONE_MINUS_SRC_ALPHA)
 *      combined with the shader-side premultiplication performed by
 *      vtkOrderIndependentTranslucentPass::PostReplaceShaderValues.
 *   2. Resolve a fullscreen pass over the drawable: recover the weighted
 *      average color (accum.rgb / reveal) and total opacity (1 - accum.a),
 *      blended with the standard over blend.
 *
 * Requires a non-MSAA depth buffer; the intermediate textures are non-MSAA.
 */

#ifndef vtkMetalOrderIndependentTranslucentPass_h
#define vtkMetalOrderIndependentTranslucentPass_h

#include "vtkRenderingMetalModule.h"

#import <Metal/Metal.h>

VTK_ABI_NAMESPACE_BEGIN

class vtkMetalRenderer;
class vtkMetalRenderWindow;

class VTKRENDERINGMETAL_EXPORT vtkMetalOrderIndependentTranslucentPass
{
public:
  vtkMetalOrderIndependentTranslucentPass();
  ~vtkMetalOrderIndependentTranslucentPass();

  /**
   * Check if the OIT textures need to be (re)created for the given size.
   */
  bool NeedsInitialization(int width, int height) const;

  /**
   * Release all GPU resources.
   */
  void Release();

  /**
   * Run the OIT accumulate + resolve passes for all translucent actors.
   * Returns non-zero on success.
   *
   * The opaque geometry must already be rendered and the depth buffer must
   * contain opaque depths. This method:
   *   1. Clears the accumulation textures
   *   2. Renders translucent geometry into them (accumulate pass)
   *   3. Resolves the result onto the drawable (fullscreen pass)
   */
  int RenderTranslucentGeometry(
    vtkMetalRenderer* renderer,
    id<MTLCommandBuffer> commandBuffer,
    id<MTLTexture> drawableTexture,
    id<MTLTexture> depthTexture);

private:
  vtkMetalOrderIndependentTranslucentPass(const vtkMetalOrderIndependentTranslucentPass&) = delete;
  void operator=(const vtkMetalOrderIndependentTranslucentPass&) = delete;

  void CreateTextures(id<MTLDevice> device, int width, int height);
  void CreatePipelines(vtkMetalRenderWindow* renWin);

  // Accumulation textures
  id<MTLTexture> AccumTexture = nil;   // RGBA16F — weighted color + transmittance
  id<MTLTexture> RevealTexture = nil;  // R16F — accumulated opacity (revealage)

  // Fullscreen resolve pipeline state
  id<MTLRenderPipelineState> ResolvePipeline = nil;

  // Cached depth-stencil states (immutable, created once)
  id<MTLDepthStencilState> ReadOnlyDepthState = nil;   // Less, write=NO

  // State
  int CurrentWidth = 0;
  int CurrentHeight = 0;

  bool PipelinesCreated = false;
};

VTK_ABI_NAMESPACE_END
#endif // vtkMetalOrderIndependentTranslucentPass_h
