// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkMetalShaderProperty
 * @brief   represent GPU shader properties for Metal rendering
 *
 * vtkMetalShaderProperty is the concrete, backend-selected implementation of
 * the abstract vtkShaderProperty for the Metal rendering backend.  It stores
 * user-defined shader replacements (the same replacement map the OpenGL
 * backend uses) that the Metal poly data mapper applies when building
 * per-actor shader libraries.
 *
 * @sa
 * vtkShaderProperty vtkUniforms
 */

#ifndef vtkMetalShaderProperty_h
#define vtkMetalShaderProperty_h

#include "vtkRenderingMetalModule.h" // For export macro
#include "vtkShaderProperty.h"
#include "vtkWrappingHints.h" // For VTK_MARSHALAUTO

VTK_ABI_NAMESPACE_BEGIN
class vtkOverrideAttribute;

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkMetalShaderProperty : public vtkShaderProperty
{
public:
  static vtkMetalShaderProperty* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalShaderProperty, vtkShaderProperty);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  void AddVertexShaderReplacement(const std::string& originalValue,
    bool replaceFirst, // do this replacement before the default
    const std::string& replacementValue, bool replaceAll) override;
  void AddFragmentShaderReplacement(const std::string& originalValue,
    bool replaceFirst, // do this replacement before the default
    const std::string& replacementValue, bool replaceAll) override;
  void AddGeometryShaderReplacement(const std::string& originalValue,
    bool replaceFirst, // do this replacement before the default
    const std::string& replacementValue, bool replaceAll) override;
  void AddTessControlShaderReplacement(const std::string& originalValue,
    bool replaceFirst, // do this replacement before the default
    const std::string& replacementValue, bool replaceAll) override;
  void AddTessEvaluationShaderReplacement(const std::string& originalValue,
    bool replaceFirst, // do this replacement before the default
    const std::string& replacementValue, bool replaceAll) override;

  int GetNumberOfShaderReplacements() override;
  std::string GetNthShaderReplacementTypeAsString(vtkIdType index) override;
  void GetNthShaderReplacement(vtkIdType index, std::string& name, bool& replaceFirst,
    std::string& replacementValue, bool& replaceAll) override;

  void ClearVertexShaderReplacement(const std::string& originalValue, bool replaceFirst) override;
  void ClearFragmentShaderReplacement(const std::string& originalValue, bool replaceFirst) override;
  void ClearGeometryShaderReplacement(const std::string& originalValue, bool replaceFirst) override;
  void ClearTessControlShaderReplacement(
    const std::string& originalValue, bool replaceFirst) override;
  void ClearTessEvaluationShaderReplacement(
    const std::string& originalValue, bool replaceFirst) override;
  void ClearAllVertexShaderReplacements() override;
  void ClearAllFragmentShaderReplacements() override;
  void ClearAllGeometryShaderReplacements() override;
  void ClearAllTessControlShaderReplacements() override;
  void ClearAllTessEvalShaderReplacements() override;
  void ClearAllShaderReplacements() override;

protected:
  vtkMetalShaderProperty();
  ~vtkMetalShaderProperty() override;

private:
  vtkMetalShaderProperty(const vtkMetalShaderProperty&) = delete;
  void operator=(const vtkMetalShaderProperty&) = delete;

  class vtkInternals;
  vtkInternals* Internals;

  void ClearAllShaderReplacementsForStage(int stage);
};

#define vtkMetalShaderProperty_OVERRIDE_ATTRIBUTES vtkMetalShaderProperty::CreateOverrideAttributes()

VTK_ABI_NAMESPACE_END
#endif // vtkMetalShaderProperty_h
