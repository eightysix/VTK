// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
#ifndef vtkMetalPolyDataMapper_h
#define vtkMetalPolyDataMapper_h

#include "vtkPolyDataMapper.h"
#include "vtkRenderingMetalModule.h"
#include "vtkWrappingHints.h"

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

  void RenderPiece(vtkRenderer* renderer, vtkActor* actor) override;
  void ReleaseGraphicsResources(vtkWindow*) override;
  MapperHashType GenerateHash(vtkPolyData* polydata) override;

protected:
  vtkMetalPolyDataMapper();
  ~vtkMetalPolyDataMapper() override;

  void BuildGeometryBuffers(void* mtlDevice, vtkPolyData* polydata, vtkActor* actor);
  void EnsurePipelineStates(void* mtlDevice);
  void EnsurePointPipelineStates(void* mtlDevice);
  void EnsureEdgePipelineState(void* mtlDevice);
  void EnsureThickLinePipelineState(void* mtlDevice);
  void UpdateMaterialUniforms(void* mtlDevice, vtkActor* actor);
  void UpdateLightUniforms(void* mtlDevice, vtkRenderer* ren);
  void UpdateCoincidentOffsetUniforms(void* mtlDevice, vtkActor* actor);
  void UpdateVertexColorUniforms(void* mtlDevice, vtkActor* actor);
  void UpdateEdgeColorUniform(void* mtlDevice, vtkActor* actor);
  void UpdateClipPlaneUniforms(void* mtlDevice, vtkActor* actor);
  void UpdateActorTexture(void* mtlDevice, vtkActor* actor);

private:
  vtkMetalPolyDataMapper(const vtkMetalPolyDataMapper&) = delete;
  void operator=(const vtkMetalPolyDataMapper&) = delete;

  struct vtkMetalPolyDataMapperInternals;
  std::unique_ptr<vtkMetalPolyDataMapperInternals> Internals;
};

VTK_ABI_NAMESPACE_END
#endif
