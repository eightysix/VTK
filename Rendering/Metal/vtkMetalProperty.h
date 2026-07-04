// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkMetalProperty
 * @brief   Metal property
 *
 * vtkMetalProperty is a concrete implementation of vtkProperty for Metal rendering.
 */

#ifndef vtkMetalProperty_h
#define vtkMetalProperty_h

#include "vtkProperty.h"
#include "vtkRenderingMetalModule.h" // for export macro
#include "vtkWrappingHints.h"        // for VTK_MARSHALAUTO

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

  void Render(vtkActor* actor) override;
  void BackfaceRender(vtkActor* actor) override;
  void PostRender(vtkActor* actor) override;

protected:
  vtkMetalProperty();
  ~vtkMetalProperty() override;

private:
  vtkMetalProperty(const vtkMetalProperty&) = delete;
  void operator=(const vtkMetalProperty&) = delete;
};

VTK_ABI_NAMESPACE_END
#endif // vtkMetalProperty_h
