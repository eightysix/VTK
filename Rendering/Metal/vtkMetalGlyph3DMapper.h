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
class vtkDataObject;
class vtkDataSet;
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

  // Build (and upload) the instanced glyph buffers for one dataset. The color
  // and opacity already carry any per-block display-attribute overrides. Cached
  // per dataset; rebuilt when the dataset, the source configuration, or
  // BlockAttributes change. Returns true when at least one source has drawable
  // geometry and instances.
  bool BuildAndUploadInstances(
    void* mtlDevice, vtkActor* actor, vtkDataSet* ds, const double color[3], double opacity, vtkIdType numSources);

  // Issue one instanced draw per glyph source for a dataset's cached instance
  // buffers, writing the PickIds {propId, compositeIndex} for the block.
  void DrawInstances(vtkRenderer* ren, vtkActor* actor, void* mtlDevice, void* encoder,
    vtkDataSet* ds, unsigned int compositeIndex, vtkIdType numSources);

  // Render the glyphs for one (leaf) dataset, building its cached instance
  // buffers if needed and drawing them.
  void RenderDataSet(vtkRenderer* ren, vtkActor* actor, vtkDataSet* ds, unsigned int compositeIndex,
    const double color[3], double opacity, bool pickable, void* mtlDevice, void* encoder,
    vtkIdType numSources);

  // Recursively walk a composite dataset, applying the per-block display
  // attributes (visibility, pickability, color, opacity) inherited from parent
  // blocks, and render each visible leaf dataset.
  void RenderChildren(vtkRenderer* ren, vtkActor* actor, vtkDataObject* dobj,
    unsigned int& flatIndex, const double color[3], double opacity, bool pickable, void* mtlDevice,
    void* encoder, vtkIdType numSources);

private:
  vtkMetalGlyph3DMapper(const vtkMetalGlyph3DMapper&) = delete;
  void operator=(const vtkMetalGlyph3DMapper&) = delete;
};

#define vtkMetalGlyph3DMapper_OVERRIDE_ATTRIBUTES \
  vtkMetalGlyph3DMapper::CreateOverrideAttributes()

VTK_ABI_NAMESPACE_END
#endif
