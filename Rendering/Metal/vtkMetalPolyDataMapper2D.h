// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkMetalPolyDataMapper2D
 * @brief   2D PolyData support for Metal
 *
 * vtkMetalPolyDataMapper2D provides 2D PolyData annotation support for
 * VTK under Metal. Normally the user should use vtkPolyDataMapper2D
 * which in turn will use this class.
 *
 * @sa
 * vtkPolyDataMapper2D
 */

#ifndef vtkMetalPolyDataMapper2D_h
#define vtkMetalPolyDataMapper2D_h

#include "vtkPolyDataMapper2D.h"
#include "vtkRenderingMetalModule.h" // for export macro
#include "vtkWrappingHints.h"        // for VTK_MARSHALAUTO

#include <memory>

VTK_ABI_NAMESPACE_BEGIN
class vtkActor2D;
class vtkOverrideAttribute;

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkMetalPolyDataMapper2D
  : public vtkPolyDataMapper2D
{
public:
  vtkTypeMacro(vtkMetalPolyDataMapper2D, vtkPolyDataMapper2D);
  static vtkMetalPolyDataMapper2D* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /**
   * Actually draw the poly data.
   */
  void RenderOverlay(vtkViewport* viewport, vtkActor2D* actor) override;

  /**
   * Release any graphics resources that are being consumed by this mapper.
   */
  void ReleaseGraphicsResources(vtkWindow* w) override;

protected:
  vtkMetalPolyDataMapper2D();
  ~vtkMetalPolyDataMapper2D() override;

private:
  vtkMetalPolyDataMapper2D(const vtkMetalPolyDataMapper2D&) = delete;
  void operator=(const vtkMetalPolyDataMapper2D&) = delete;

  struct vtkMetalPolyDataMapper2DInternals;
  std::unique_ptr<vtkMetalPolyDataMapper2DInternals> Internals;
};

#define vtkMetalPolyDataMapper2D_OVERRIDE_ATTRIBUTES \
  vtkMetalPolyDataMapper2D::CreateOverrideAttributes()

VTK_ABI_NAMESPACE_END
#endif
