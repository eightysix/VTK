// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @class   vtkMetalBillboardTextActor3D
 * @brief   vtkBillboardTextActor3D override for the Metal rendering backend.
 *
 * vtkBillboardTextActor3D renders by texturing a vtkActor quad with a vtkTexture.
 * With the "RenderingBackend=Metal" factory override active, the actor/mapper and
 * texture resolve to their Metal implementations (vtkMetalActor, vtkMetalPolyDataMapper,
 * vtkMetalTexture), which already support actor textures. This class exists so the
 * object factory selects a Metal-specific vtkBillboardTextActor3D, mirroring the
 * OpenGL2 backend (which only overrides for GL2PS export, a feature Metal does not have).
 */

#ifndef vtkMetalBillboardTextActor3D_h
#define vtkMetalBillboardTextActor3D_h

#include "vtkBillboardTextActor3D.h"
#include "vtkRenderingMetalModule.h" // for export macro

VTK_ABI_NAMESPACE_BEGIN

class vtkOverrideAttribute;

class VTKRENDERINGMETAL_EXPORT vtkMetalBillboardTextActor3D : public vtkBillboardTextActor3D
{
public:
  static vtkMetalBillboardTextActor3D* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalBillboardTextActor3D, vtkBillboardTextActor3D);
  void PrintSelf(ostream& os, vtkIndent indent) override;

protected:
  vtkMetalBillboardTextActor3D();
  ~vtkMetalBillboardTextActor3D() override;

private:
  vtkMetalBillboardTextActor3D(const vtkMetalBillboardTextActor3D&) = delete;
  void operator=(const vtkMetalBillboardTextActor3D&) = delete;
};

#define vtkMetalBillboardTextActor3D_OVERRIDE_ATTRIBUTES                                        \
  vtkMetalBillboardTextActor3D::CreateOverrideAttributes()

VTK_ABI_NAMESPACE_END
#endif // vtkMetalBillboardTextActor3D_h
