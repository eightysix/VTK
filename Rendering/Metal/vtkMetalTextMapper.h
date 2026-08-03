// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @class   vtkMetalTextMapper
 * @brief   vtkTextMapper override for the Metal rendering backend.
 *
 * vtkTextMapper renders via a vtkPolyDataMapper2D and a vtkTexture. With the
 * "RenderingBackend=Metal" factory override active, those components resolve to
 * their Metal implementations (vtkMetalPolyDataMapper2D, vtkMetalTexture), so no
 * extra work is required here. This class exists so the object factory selects a
 * Metal-specific vtkTextMapper, mirroring the OpenGL2 backend.
 */

#ifndef vtkMetalTextMapper_h
#define vtkMetalTextMapper_h

#include "vtkRenderingMetalModule.h" // for export macro
#include "vtkTextMapper.h"

VTK_ABI_NAMESPACE_BEGIN

class vtkOverrideAttribute;

class VTKRENDERINGMETAL_EXPORT vtkMetalTextMapper : public vtkTextMapper
{
public:
  static vtkMetalTextMapper* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalTextMapper, vtkTextMapper);
  void PrintSelf(ostream& os, vtkIndent indent) override;

protected:
  vtkMetalTextMapper();
  ~vtkMetalTextMapper() override;

private:
  vtkMetalTextMapper(const vtkMetalTextMapper&) = delete;
  void operator=(const vtkMetalTextMapper&) = delete;
};

#define vtkMetalTextMapper_OVERRIDE_ATTRIBUTES vtkMetalTextMapper::CreateOverrideAttributes()

VTK_ABI_NAMESPACE_END
#endif // vtkMetalTextMapper_h
