// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkMetalImageMapper
 * @brief   2D image display support for Metal
 *
 * vtkMetalImageMapper is a concrete subclass of vtkImageMapper that
 * renders images under Metal.
 *
 * @warning
 * vtkMetalImageMapper does not support vtkBitArray, you have to convert the array first
 * to vtkUnsignedCharArray (for example)
 *
 * @sa
 * vtkImageMapper
 */

#ifndef vtkMetalImageMapper_h
#define vtkMetalImageMapper_h

#include "vtkImageMapper.h"
#include "vtkRenderingMetalModule.h" // For export macro
#include "vtkWrappingHints.h"        // For VTK_MARSHALAUTO

#include <memory>

VTK_ABI_NAMESPACE_BEGIN
class vtkActor2D;
class vtkOverrideAttribute;

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkMetalImageMapper : public vtkImageMapper
{
public:
  static vtkMetalImageMapper* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalImageMapper, vtkImageMapper);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /**
   * Handle the render method.
   */
  void RenderOverlay(vtkViewport* viewport, vtkActor2D* actor) override
  {
    this->RenderStart(viewport, actor);
  }

  /**
   * Called by the Render function in vtkImageMapper.  Actually draws
   * the image to the screen.
   */
  void RenderData(vtkViewport* viewport, vtkImageData* data, vtkActor2D* actor) override;

  /**
   * draw the data once it has been converted to uchar, windowed leveled
   * used internally by the templated functions
   */
  void DrawPixels(vtkViewport* viewport, int width, int height, int numComponents, void* data);

  /**
   * Release any graphics resources that are being consumed by this
   * mapper, the image texture in particular.
   */
  void ReleaseGraphicsResources(vtkWindow*) override;

protected:
  vtkMetalImageMapper();
  ~vtkMetalImageMapper() override;

private:
  vtkMetalImageMapper(const vtkMetalImageMapper&) = delete;
  void operator=(const vtkMetalImageMapper&) = delete;

  struct vtkMetalImageMapperInternals;
  std::unique_ptr<vtkMetalImageMapperInternals> Internals;
};

#define vtkMetalImageMapper_OVERRIDE_ATTRIBUTES vtkMetalImageMapper::CreateOverrideAttributes()
VTK_ABI_NAMESPACE_END
#endif
