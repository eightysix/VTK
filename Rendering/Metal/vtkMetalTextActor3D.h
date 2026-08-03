// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @class   vtkMetalTextActor3D
 * @brief   vtkTextActor3D override for the Metal rendering backend.
 *
 * vtkTextActor3D renders through a vtkImageActor backed by a vtkImageSliceMapper.
 * With the "RenderingBackend=Metal" factory override active, the mapper resolves
 * to vtkMetalImageSliceMapper, so no extra work is required here. This class
 * exists so the object factory selects a Metal-specific vtkTextActor3D, mirroring
 * the OpenGL2 backend.
 */

#ifndef vtkMetalTextActor3D_h
#define vtkMetalTextActor3D_h

#include "vtkRenderingMetalModule.h" // for export macro
#include "vtkTextActor3D.h"
#include "vtkWrappingHints.h" // for VTK_MARSHALAUTO

VTK_ABI_NAMESPACE_BEGIN

class vtkOverrideAttribute;

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkMetalTextActor3D : public vtkTextActor3D
{
public:
  static vtkMetalTextActor3D* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalTextActor3D, vtkTextActor3D);
  void PrintSelf(ostream& os, vtkIndent indent) override;

protected:
  vtkMetalTextActor3D();
  ~vtkMetalTextActor3D() override;

private:
  vtkMetalTextActor3D(const vtkMetalTextActor3D&) = delete;
  void operator=(const vtkMetalTextActor3D&) = delete;
};

#define vtkMetalTextActor3D_OVERRIDE_ATTRIBUTES vtkMetalTextActor3D::CreateOverrideAttributes()

VTK_ABI_NAMESPACE_END
#endif // vtkMetalTextActor3D_h
