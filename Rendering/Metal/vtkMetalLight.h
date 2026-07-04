// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkMetalLight
 * @brief   Metal light
 *
 * vtkMetalLight is a concrete implementation of vtkLight for Metal rendering.
 */

#ifndef vtkMetalLight_h
#define vtkMetalLight_h

#include "vtkLight.h"
#include "vtkRenderingMetalModule.h" // for export macro
#include "vtkWrappingHints.h"        // for VTK_MARSHALAUTO

VTK_ABI_NAMESPACE_BEGIN

class vtkOverrideAttribute;

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkMetalLight : public vtkLight
{
public:
  static vtkMetalLight* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalLight, vtkLight);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  void Render(vtkRenderer* ren, int light_index) override;

protected:
  vtkMetalLight();
  ~vtkMetalLight() override;

private:
  vtkMetalLight(const vtkMetalLight&) = delete;
  void operator=(const vtkMetalLight&) = delete;
};

VTK_ABI_NAMESPACE_END
#endif // vtkMetalLight_h
