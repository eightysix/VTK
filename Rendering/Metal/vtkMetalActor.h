// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkMetalActor
 * @brief   Metal actor
 *
 * vtkMetalActor is a concrete implementation of vtkActor for Metal rendering.
 */

#ifndef vtkMetalActor_h
#define vtkMetalActor_h

#include "vtkActor.h"
#include "vtkRenderingMetalModule.h" // for export macro
#include "vtkWrappingHints.h"        // for VTK_MARSHALAUTO

VTK_ABI_NAMESPACE_BEGIN

class vtkOverrideAttribute;

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkMetalActor : public vtkActor
{
public:
  static vtkMetalActor* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalActor, vtkActor);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  void Render(vtkRenderer* renderer, vtkMapper* mapper) override;
  void ReleaseGraphicsResources(vtkWindow* window) override;

protected:
  vtkMetalActor();
  ~vtkMetalActor() override;

private:
  vtkMetalActor(const vtkMetalActor&) = delete;
  void operator=(const vtkMetalActor&) = delete;
};

VTK_ABI_NAMESPACE_END
#endif // vtkMetalActor_h
