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
#include <unordered_map>
#include <vector>

VTK_ABI_NAMESPACE_BEGIN
class vtkOverrideAttribute;
class vtkPointData;
class vtkTexture;
class vtkUnsignedCharArray;

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

  // Override prop ID (set by vtkMetalBatchedPolyDataMapper for per-block picking)
  void SetOverridePropId(uint32_t zeroBasedPropId);
  void SetOverridePropIdToNone();
  void ClearOverridePropId();

  // Override the composite index written to the picking texture's CompositeId
  // channel (set by vtkMetalBatchedPolyDataMapper for per-block picking).
  void SetOverrideCompositeIndex(uint32_t compositeIndex);
  void ClearOverrideCompositeIndex();

  // Batch visual overrides (set by vtkMetalBatchedPolyDataMapper)
  void SetBatchVisualOverride(
    bool overrideColor,
    const double color[3],
    bool overrideOpacity,
    double opacity);
  void ClearBatchVisualOverride();

  // Per-block texture override (set by vtkMetalBatchedPolyDataMapper for blocks
  // with a block texture image). Takes precedence over the actor's texture.
  // Passing nullptr falls back to the actor's texture.
  void SetBlockTexture(vtkTexture* texture);

  // A/B switch for the single-pass surface edges (new path) vs. the legacy
  // chord-depth edge overlay draw. Default off (new path).
  vtkSetMacro(UseLegacyEdgeOverlay, bool);
  vtkGetMacro(UseLegacyEdgeOverlay, bool);
  vtkBooleanMacro(UseLegacyEdgeOverlay, bool);

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
  void EnsurePipelineStates(void* mtlDevice, vtkActor* actor);
  void EnsurePointPipelineStates(void* mtlDevice);
  void EnsureEdgePipelineState(void* mtlDevice);
  void EnsureThickLinePipelineState(void* mtlDevice, bool lightingDisabled);
  void EnsureRoundCapLinePipelineState(void* mtlDevice, bool lightingDisabled);
  void EnsureMiterJoinLinePipelineState(void* mtlDevice, bool lightingDisabled);
  void EnsureThickLineOITPipelineState(void* mtlDevice, bool lightingDisabled);
  void EnsureRoundCapLineOITPipelineState(void* mtlDevice, bool lightingDisabled);
  void EnsureMiterJoinLineOITPipelineState(void* mtlDevice, bool lightingDisabled);
  void EnsureLineOITPipelineState(void* mtlDevice);
  void UpdateMaterialUniforms(void* mtlDevice, vtkActor* actor);
  void UpdateLightUniforms(void* mtlDevice, vtkRenderer* ren);
  void UpdateCoincidentOffsetUniforms(void* mtlDevice, vtkActor* actor);
  void UpdateVertexColorUniforms(void* mtlDevice, vtkActor* actor);
  void UpdateEdgeColorUniform(void* mtlDevice, vtkActor* actor);
  void UpdateEdgeUniforms(void* mtlDevice, vtkActor* actor);
  void UpdateClipPlaneUniforms(void* mtlDevice, vtkActor* actor);
  void UpdateActorTexture(void* mtlDevice, vtkActor* actor);
  void EnsurePeelPipelineStates(void* mtlDevice);
  void EnsureOITPipelineStates(void* mtlDevice);

  // Picking: write {propId, compositeIndex} into the PropIdBuffer (PickIds).
  // During a selection pass propId is the prop's per-render PropArray index
  // (queried from the hardware selector); otherwise it falls back to
  // overrides/0.
  void UpdatePickUniforms(vtkRenderer* ren, vtkActor* act);

  // 8D: Vertex attribute mappings (attribute name → data source)
  std::map<std::string, ExtraAttributeValue> ExtraAttributes;
  vtkTimeStamp ExtraAttributesMTime;

  // A/B switch for the single-pass surface edges vs. the legacy chord-depth
  // edge overlay draw. Default off (new path).
  bool UseLegacyEdgeOverlay = false;

  // Upload vertex data from CPU vectors to GPU Metal buffers
  void UploadVertexDataToMTLBuffers(void* mtlDevice, vtkPolyData* polydata,
    vtkPointData* pd, vtkUnsignedCharArray* mappedColors, int cellFlag,
    int representation, bool gpuTessUsed, const float defaultRGBA[4],
    std::vector<float>& positions, std::vector<float>& normals,
    const std::vector<float>& surfaceColors, const std::vector<float>& triangleUVs,
    const std::vector<float>& triangleScalarCoords, bool useScalarLUT,
    const std::vector<uint32_t>& lineIndices,
    const std::vector<uint32_t>& triangleIndices,
    const std::vector<uint32_t>& triangleEdgeFlags,
    const std::vector<float>& trianglePos,
    const std::vector<float>& edgePositions,
    std::vector<float>& edgeNormals, const std::vector<float>& edgeColors,
    const std::vector<float>& edgeUVs, const std::vector<uint32_t>& edgeIndices,
    const std::vector<uint32_t>& triangleVertexCellIds,
    const std::vector<uint32_t>& lineVertexCellIds,
    const std::vector<uint32_t>& lineSegmentCellIds,
    const std::vector<uint32_t>& edgeVertexCellIds,
    const std::vector<uint32_t>& edgeTubeIndices,
    const std::vector<uint32_t>& edgeTubeCellIds,
    const std::vector<float>& cellColors,
    const std::vector<uint32_t>& cellPrimitiveIds,
    std::unordered_map<std::string, std::vector<float>>& extraAttrArrays);

  // P1-3: Ensure fallback buffers exist for all shader-required bindings
  void EnsureRequiredBindingFallbacks(void* mtlDevice);

  // 8C: Render bundle caching — pre-recorded encoder commands for static geometry
  void ReplayRenderBundle(void* mtlRenderCommandEncoder, vtkRenderer* ren, vtkActor* act);
  void RebuildRenderBundle(void* mtlRenderCommandEncoder, vtkRenderer* ren, vtkActor* act);

private:
  vtkMetalPolyDataMapper(const vtkMetalPolyDataMapper&) = delete;
  void operator=(const vtkMetalPolyDataMapper&) = delete;

  struct vtkMetalPolyDataMapperInternals;
  std::unique_ptr<vtkMetalPolyDataMapperInternals> Internals;
};

#define vtkMetalPolyDataMapper_OVERRIDE_ATTRIBUTES vtkMetalPolyDataMapper::CreateOverrideAttributes()

VTK_ABI_NAMESPACE_END
#endif
