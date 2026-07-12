// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class vtkMetalGlyph3DMapper
 * @brief Generate 3D glyphs at points in input dataset using Metal instanced rendering.
 *
 * Renders source geometry (polydata) at each input point with per-instance
 * transforms (translation, rotation, scaling) and per-instance colors.
 * Uses Metal instanced drawing with per-instance vertex attribute buffers
 * for transforms, normal transforms, colors, and pick IDs.
 *
 * @sa vtkGlyph3DMapper vtkMetalPolyDataMapper vtkWebGPUGlyph3DMapper
 */

#ifndef vtkMetalGlyph3DMapper_h
#define vtkMetalGlyph3DMapper_h

#include "vtkGlyph3DMapper.h"
#include "vtkRenderingMetalModule.h"
#include "vtkWrappingHints.h"

#include <memory>
#include <vector>

VTK_ABI_NAMESPACE_BEGIN
class vtkOverrideAttribute;

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkMetalGlyph3DMapper
  : public vtkGlyph3DMapper
{
public:
  static vtkMetalGlyph3DMapper* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalGlyph3DMapper, vtkGlyph3DMapper);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  void Render(vtkRenderer* renderer, vtkActor* actor) override;
  void ReleaseGraphicsResources(vtkWindow* window) override;

  struct vtkMetalGlyph3DMapperInternals;
  std::unique_ptr<vtkMetalGlyph3DMapperInternals> Internals;

protected:
  vtkMetalGlyph3DMapper();
  ~vtkMetalGlyph3DMapper() override;

private:
  vtkMetalGlyph3DMapper(const vtkMetalGlyph3DMapper&) = delete;
  void operator=(const vtkMetalGlyph3DMapper&) = delete;
};

#define vtkMetalGlyph3DMapper_OVERRIDE_ATTRIBUTES \
  vtkMetalGlyph3DMapper::CreateOverrideAttributes()

VTK_ABI_NAMESPACE_END
#endif
