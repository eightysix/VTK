// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkMetalPolyDataMapper
 * @brief   Metal polydata mapper
 *
 * vtkMetalPolyDataMapper is a concrete implementation of vtkPolyDataMapper
 * for Metal rendering. It converts VTK cell arrays into Metal vertex/index
 * buffers and issues draw calls for triangles and lines.
 */

#ifndef vtkMetalPolyDataMapper_h
#define vtkMetalPolyDataMapper_h

#include "vtkPolyDataMapper.h"
#include "vtkRenderingMetalModule.h" // for export macro
#include "vtkWrappingHints.h"        // for VTK_MARSHALAUTO

#include <memory>

VTK_ABI_NAMESPACE_BEGIN

class vtkOverrideAttribute;

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkMetalPolyDataMapper
  : public vtkPolyDataMapper
{
public:
  static vtkMetalPolyDataMapper* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalPolyDataMapper, vtkPolyDataMapper);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /**
   * Main render entry point called by the actor.
   */
  void RenderPiece(vtkRenderer* renderer, vtkActor* actor) override;

  /**
   * Release graphics resources.
   */
  void ReleaseGraphicsResources(vtkWindow*) override;

  /**
   * Generate a hash for caching pipeline state.
   */
  MapperHashType GenerateHash(vtkPolyData* polydata) override;

protected:
  vtkMetalPolyDataMapper();
  ~vtkMetalPolyDataMapper() override;

  /**
   * Build Metal vertex and index buffers from VTK polydata.
   */
  void BuildGeometryBuffers(void* mtlDevice, vtkPolyData* polydata);

  /**
   * Ensure Metal render pipeline states exist.
   */
  void EnsurePipelineStates(void* mtlDevice);

  /**
   * Update the material uniform buffer from the actor's property.
   */
  void UpdateMaterialUniforms(void* mtlDevice, vtkActor* actor);

  /**
   * Update the light uniform buffer from the renderer's lights.
   */
  void UpdateLightUniforms(void* mtlDevice, vtkRenderer* ren);

private:
  vtkMetalPolyDataMapper(const vtkMetalPolyDataMapper&) = delete;
  void operator=(const vtkMetalPolyDataMapper&) = delete;

  struct vtkMetalPolyDataMapperInternals;
  std::unique_ptr<vtkMetalPolyDataMapperInternals> Internals;
};

VTK_ABI_NAMESPACE_END
#endif // vtkMetalPolyDataMapper_h
