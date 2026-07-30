// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
#ifndef vtkMetalPolyDataMapper_h
#define vtkMetalPolyDataMapper_h

#include "vtkPolyDataMapper.h"
#include "vtkRenderingMetalModule.h"
#include "vtkWrappingHints.h"

#include <map>
#include <memory>
#include <string>

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

  // 8D: Vertex attribute mapping — map VTK data arrays to generic vertex attributes
  void MapDataArrayToVertexAttribute(const char* vertexAttributeName,
    const char* dataArrayName,
    int fieldAssociation,
    int componentno = -1) override;
  void RemoveVertexAttributeMapping(const char* vertexAttributeName) override;
  void RemoveAllVertexAttributeMappings() override;

  // P2-8C: Set per-frame prop ID for picking
  void SetPropId(uint32_t propId);
  uint32_t GetOrCreatePropId(vtkActor* act);

  struct ExtraAttributeValue
  {
    std::string DataArrayName;
    int FieldAssociation; // vtkDataObject::FIELD_ASSOCIATION_POINTS or _CELLS
    int ComponentNumber;  // -1 = all components
  };

protected:
  vtkMetalPolyDataMapper();
  ~vtkMetalPolyDataMapper() override;

  void BuildGeometryBuffers(void* mtlDevice, vtkPolyData* polydata, vtkActor* actor);
  void EnsurePipelineStates(void* mtlDevice);
  void EnsurePointPipelineStates(void* mtlDevice);
  void EnsureEdgePipelineState(void* mtlDevice);
  void EnsureThickLinePipelineState(void* mtlDevice);
  void EnsureRoundCapLinePipelineState(void* mtlDevice);
  void EnsureMiterJoinLinePipelineState(void* mtlDevice);
  void UpdateMaterialUniforms(void* mtlDevice, vtkActor* actor);
  void UpdateLightUniforms(void* mtlDevice, vtkRenderer* ren);
  void UpdateCoincidentOffsetUniforms(void* mtlDevice, vtkActor* actor);
  void UpdateVertexColorUniforms(void* mtlDevice, vtkActor* actor);
  void UpdateEdgeColorUniform(void* mtlDevice, vtkActor* actor);
  void UpdateClipPlaneUniforms(void* mtlDevice, vtkActor* actor);
  void UpdateActorTexture(void* mtlDevice, vtkActor* actor);
  void EnsurePeelPipelineStates(void* mtlDevice);

  // 8D: Vertex attribute mappings (attribute name → data source)
  std::map<std::string, ExtraAttributeValue> ExtraAttributes;
  vtkTimeStamp ExtraAttributesMTime;

  // 8C: Render bundle caching — pre-recorded encoder commands for static geometry
  void ReplayRenderBundle(void* mtlRenderCommandEncoder);
  void RebuildRenderBundle(void* mtlRenderCommandEncoder, vtkRenderer* ren, vtkActor* act);

private:
  vtkMetalPolyDataMapper(const vtkMetalPolyDataMapper&) = delete;
  void operator=(const vtkMetalPolyDataMapper&) = delete;

  struct vtkMetalPolyDataMapperInternals;
  std::unique_ptr<vtkMetalPolyDataMapperInternals> Internals;
};

VTK_ABI_NAMESPACE_END
#endif
