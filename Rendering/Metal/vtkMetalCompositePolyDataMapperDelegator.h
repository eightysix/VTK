// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @class vtkMetalCompositePolyDataMapperDelegator
 * @brief A Metal delegator for batched rendering of multiple polydata with similar structure.
 *
 * This class delegates work to vtkMetalBatchedPolyDataMapper which can do batched rendering
 * of many polydata.
 *
 * @sa vtkMetalBatchedPolyDataMapper
 */

#ifndef vtkMetalCompositePolyDataMapperDelegator_h
#define vtkMetalCompositePolyDataMapperDelegator_h

#include "vtkCompositePolyDataMapperDelegator.h"
#include "vtkRenderingMetalModule.h"

VTK_ABI_NAMESPACE_BEGIN

class vtkOverrideAttribute;
class vtkMetalBatchedPolyDataMapper;

class VTKRENDERINGMETAL_EXPORT vtkMetalCompositePolyDataMapperDelegator
  : public vtkCompositePolyDataMapperDelegator
{
public:
  static vtkMetalCompositePolyDataMapperDelegator* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalCompositePolyDataMapperDelegator, vtkCompositePolyDataMapperDelegator);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  using BatchElement = vtkCompositePolyDataMapperDelegator::BatchElement;

  ///@{
  /**
   * Implement parent class API.
   */
  void ShallowCopy(vtkCompositePolyDataMapper* mapper) override;
  void ClearUnmarkedBatchElements() override;
  void UnmarkBatchElements() override;
  ///@}

protected:
  vtkMetalCompositePolyDataMapperDelegator();
  ~vtkMetalCompositePolyDataMapperDelegator() override;

  ///@{
  /**
   * Implement parent class API.
   */
  std::vector<vtkPolyData*> GetRenderedList() const override;
  void SetParent(vtkCompositePolyDataMapper* mapper) override;
  void Insert(BatchElement&& item) override;
  BatchElement* Get(vtkPolyData* polydata) override;
  void Clear() override;
  ///@}

  // The actual mapper which renders multiple vtkPolyData.
  vtkMetalBatchedPolyDataMapper* MetalDelegate = nullptr;

private:
  vtkMetalCompositePolyDataMapperDelegator(const vtkMetalCompositePolyDataMapperDelegator&) = delete;
  void operator=(const vtkMetalCompositePolyDataMapperDelegator&) = delete;
};

#define vtkMetalCompositePolyDataMapperDelegator_OVERRIDE_ATTRIBUTES \
  vtkMetalCompositePolyDataMapperDelegator::CreateOverrideAttributes()

VTK_ABI_NAMESPACE_END
#endif
