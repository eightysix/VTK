// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @class vtkMetalBatchedPolyDataMapper
 * @brief A Metal mapper for batched rendering of multiple vtkPolyData.
 *
 * Tactical batched renderer that caches one child vtkMetalPolyDataMapper per
 * vtkPolyData and renders visible elements in flat-index order. This avoids
 * changing the parent mapper input every frame.
 *
 * Future work: true combined vertex/index buffers, per-mesh property binding,
 * and per-batch-element visual overrides.
 *
 * @sa vtkMetalPolyDataMapper vtkCompositePolyDataMapperDelegator
 */

#ifndef vtkMetalBatchedPolyDataMapper_h
#define vtkMetalBatchedPolyDataMapper_h

#include "vtkMetalPolyDataMapper.h"
#include "vtkRenderingMetalModule.h"
#include "vtkCompositePolyDataMapperDelegator.h"
#include "vtkSmartPointer.h"

#include <memory>
#include <vector>
#include <map>

VTK_ABI_NAMESPACE_BEGIN
class vtkCompositePolyDataMapper;
class vtkPolyData;
class vtkMetalPolyDataMapper;

class VTKRENDERINGMETAL_EXPORT vtkMetalBatchedPolyDataMapper : public vtkMetalPolyDataMapper
{
public:
  static vtkMetalBatchedPolyDataMapper* New();
  vtkTypeMacro(vtkMetalBatchedPolyDataMapper, vtkMetalPolyDataMapper);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  using BatchElement = vtkCompositePolyDataMapperDelegator::BatchElement;

  void AddBatchElement(unsigned int flatIndex, BatchElement&& batchElement);
  BatchElement* GetBatchElement(vtkPolyData* polydata);
  void ClearBatchElements();
  std::vector<vtkPolyData*> GetRenderedList() const;
  void SetParent(vtkCompositePolyDataMapper* parent);
  void UnmarkBatchElements();
  void ClearUnmarkedBatchElements();

  void RenderPiece(vtkRenderer* renderer, vtkActor* actor) override;
  void ReleaseGraphicsResources(vtkWindow*) override;
  vtkMTimeType GetMTime() override;

protected:
  vtkMetalBatchedPolyDataMapper();
  ~vtkMetalBatchedPolyDataMapper() override;

private:
  vtkMetalBatchedPolyDataMapper(const vtkMetalBatchedPolyDataMapper&) = delete;
  void operator=(const vtkMetalBatchedPolyDataMapper&) = delete;

  /**
   * Per-mesh properties struct, matching WebGPU's CompositeDataProperties.
   * Stored in a single MTLBuffer with 256-byte alignment per entry.
   */
  struct CompositeDataProperties
  {
    uint32_t ApplyOverrideColors = 0;
    float Opacity = 1.0f;
    uint32_t CompositeId = 0;
    uint32_t Pickable = 1;
    float Ambient[3] = { 1.0f, 1.0f, 1.0f };
    uint32_t CellIdOffsetForVerts = 0;
    float Diffuse[3] = { 1.0f, 1.0f, 1.0f };
    uint32_t CellIdOffsetForLines = 0;
    uint32_t CellIdOffsetForPolys = 0;
    uint32_t CellIdOffsetForSelector = 0;
  };

  // Align to 256 bytes for Metal uniform buffer alignment
  static constexpr size_t PropertiesAlignment = 256;
  static constexpr size_t AlignedPropertiesSize =
    (sizeof(CompositeDataProperties) + PropertiesAlignment - 1) & ~(PropertiesAlignment - 1);

  void UpdateBatchPropertiesBuffer(void* mtlDevice);

  vtkCompositePolyDataMapper* Parent = nullptr;
  std::map<std::uintptr_t, std::unique_ptr<BatchElement>> VTKPolyDataToBatchElement;
  std::map<unsigned int, std::uintptr_t> FlatIndexToPolyData;

  // Per-actor property buffer (all meshes' properties packed with alignment)
  // id<MTLBuffer> stored as void* to avoid ObjC in C++ header
  void* BatchPropertiesBuffer = nullptr;
  size_t BatchPropertiesBufferSize = 0;

  vtkTimeStamp ResourcesSyncTimeStamp;
  bool GeometryDirty = true;
  vtkMTimeType CachedChildConfigurationMTime = 0;

  std::map<std::uintptr_t, vtkSmartPointer<vtkMetalPolyDataMapper>> ChildMappers;

  vtkSmartPointer<vtkMetalPolyDataMapper> GetChildMapper(vtkPolyData* polydata);
  void ReleaseChildMappers(vtkWindow* w);
  void ReleaseBatchPropertiesBuffer();
  void ConfigureChildMapper(vtkMetalPolyDataMapper* child);
  void SetBatchPropertiesBufferConsumed(void* buffer);
};

VTK_ABI_NAMESPACE_END
#endif
