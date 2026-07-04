// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
#ifndef vtkMetalProperty_h
#define vtkMetalProperty_h

#include "vtkProperty.h"
#include "vtkRenderingMetalModule.h"
#include "vtkWrappingHints.h"

VTK_ABI_NAMESPACE_BEGIN
class vtkOverrideAttribute;

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkMetalProperty : public vtkProperty
{
public:
  static vtkMetalProperty* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalProperty, vtkProperty);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  void Render(vtkActor* actor, vtkRenderer* ren) override;
  void BackfaceRender(vtkActor* actor, vtkRenderer* ren) override;
  void PostRender(vtkActor* actor, vtkRenderer* ren) override;

protected:
  vtkMetalProperty();
  ~vtkMetalProperty() override;

private:
  vtkMetalProperty(const vtkMetalProperty&) = delete;
  void operator=(const vtkMetalProperty&) = delete;
};

VTK_ABI_NAMESPACE_END
#endif
