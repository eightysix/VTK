// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkMetalRenderer
 * @brief   Metal rendering renderer
 *
 * vtkMetalRenderer is a concrete implementation of vtkRenderer that
 * uses Apple's Metal API for rendering. It drives the per-frame rendering
 * pipeline, managing the render pass encoder and coordinating mappers.
 */

#ifndef vtkMetalRenderer_h
#define vtkMetalRenderer_h

#include "vtkRenderer.h"
#include "vtkRenderingMetalModule.h" // for export macro
#include "vtkWrappingHints.h"        // for VTK_MARSHALAUTO

#include <memory>

VTK_ABI_NAMESPACE_BEGIN

class vtkOverrideAttribute;
class vtkMetalDepthPeeler;

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkMetalRenderer : public vtkRenderer
{
public:
  static vtkMetalRenderer* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalRenderer, vtkRenderer);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /**
   * Clear the image to the background color.
   */
  void Clear() override;

  /**
   * Create an image. This is the main rendering entry point.
   */
  void DeviceRender() override;

  /**
   * Update geometry for rendering.
   */
  int UpdateGeometry(vtkFrameBufferObjectBase* fbo = nullptr) override;

  /**
   * Release graphics resources.
   */
  void ReleaseGraphicsResources(vtkWindow* w) override;

  /**
   * Render translucent polygonal geometry (public wrapper for depth peeler).
   */
  void RenderTranslucentGeometry();

protected:
  vtkMetalRenderer();
  ~vtkMetalRenderer() override;

  /**
   * Check if any visible props have translucent geometry.
   */
  bool HasTranslucentPolygonalGeometry();

private:
  std::unique_ptr<vtkMetalDepthPeeler> DepthPeeler;

  // Cached depth/stencil states to avoid per-frame allocation.
  // void* to avoid Obj-C types in the header.
  void* DepthWriteOnState = nullptr;  // id<MTLDepthStencilState> — Less + write ON
  void* DepthWriteOffState = nullptr; // id<MTLDepthStencilState> — Less + write OFF

  vtkMetalRenderer(const vtkMetalRenderer&) = delete;
  void operator=(const vtkMetalRenderer&) = delete;
};

VTK_ABI_NAMESPACE_END
#endif // vtkMetalRenderer_h
