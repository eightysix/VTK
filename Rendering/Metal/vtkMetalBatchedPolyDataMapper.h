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

  vtkCompositePolyDataMapper* Parent = nullptr;
  std::map<std::uintptr_t, std::unique_ptr<BatchElement>> VTKPolyDataToBatchElement;
  std::map<unsigned int, std::uintptr_t> FlatIndexToPolyData;

  bool GeometryDirty = true;
  vtkMTimeType CachedChildConfigurationMTime = 0;

  std::map<std::uintptr_t, vtkSmartPointer<vtkMetalPolyDataMapper>> ChildMappers;

  vtkSmartPointer<vtkMetalPolyDataMapper> GetChildMapper(vtkPolyData* polydata);
  void ReleaseChildMappers(vtkWindow* w);
  void ConfigureChildMapper(vtkMetalPolyDataMapper* child);
};

VTK_ABI_NAMESPACE_END
#endif
