// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @class   vtkMetalTextActor
 * @brief   vtkTextActor override for the Metal rendering backend.
 *
 * vtkTextActor renders via a vtkTexturedActor2D backed by a vtkPolyDataMapper2D
 * and a vtkTexture. With the "RenderingBackend=Metal" factory override active,
 * those components resolve to their Metal implementations (vtkMetalPolyDataMapper2D,
 * vtkMetalTexture), so no extra work is required here. This class exists so the
 * object factory selects a Metal-specific vtkTextActor, mirroring the OpenGL2
 * backend.
 */

#ifndef vtkMetalTextActor_h
#define vtkMetalTextActor_h

#include "vtkRenderingMetalModule.h" // for export macro
#include "vtkTextActor.h"
#include "vtkWrappingHints.h" // for VTK_MARSHALAUTO

VTK_ABI_NAMESPACE_BEGIN

class vtkOverrideAttribute;

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkMetalTextActor : public vtkTextActor
{
public:
  static vtkMetalTextActor* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalTextActor, vtkTextActor);
  void PrintSelf(ostream& os, vtkIndent indent) override;

protected:
  vtkMetalTextActor();
  ~vtkMetalTextActor() override;

private:
  vtkMetalTextActor(const vtkMetalTextActor&) = delete;
  void operator=(const vtkMetalTextActor&) = delete;
};

#define vtkMetalTextActor_OVERRIDE_ATTRIBUTES vtkMetalTextActor::CreateOverrideAttributes()

VTK_ABI_NAMESPACE_END
#endif // vtkMetalTextActor_h
