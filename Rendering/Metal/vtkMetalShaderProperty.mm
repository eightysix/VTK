// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
#include "vtkMetalShaderProperty.h"

#include "vtkObjectFactory.h"
#include "vtkOverrideAttribute.h"
#include <map>
#include <string>
#include <tuple>

VTK_ABI_NAMESPACE_BEGIN

// Shader-stage identifiers matching the strings returned by
// GetNthShaderReplacementTypeAsString (the abstract vtkShaderProperty API
// exposes the replacements only as strings, so no vtkShader dependency is
// needed here; the OpenGL implementation reaches the same ordering through
// its vtkShader::Type enum).
namespace
{
enum class MetalShaderStage
{
  Vertex = 0,
  Fragment = 1,
  Geometry = 2,
  TessControl = 3,
  TessEvaluation = 4
};

const char* MetalShaderStageString(MetalShaderStage stage)
{
  switch (stage)
  {
    case MetalShaderStage::Vertex:
      return "Vertex";
    case MetalShaderStage::Fragment:
      return "Fragment";
    case MetalShaderStage::Geometry:
      return "Geometry";
    case MetalShaderStage::TessControl:
      return "TessControl";
    case MetalShaderStage::TessEvaluation:
      return "TessEvaluation";
  }
  return "Unknown";
}
} // namespace

class vtkMetalShaderProperty::vtkInternals
{
public:
  using Key = std::tuple<int, std::string, bool>; // stage, original value, replace-first
  using Value = std::tuple<std::string, bool>;    // replacement text, replace-all
  std::map<Key, Value> UserShaderReplacements;
};

vtkStandardNewMacro(vtkMetalShaderProperty);

vtkMetalShaderProperty::vtkMetalShaderProperty()
  : Internals(new vtkMetalShaderProperty::vtkInternals())
{
}

vtkMetalShaderProperty::~vtkMetalShaderProperty()
{
  delete this->Internals;
}

vtkOverrideAttribute* vtkMetalShaderProperty::CreateOverrideAttributes()
{
  auto* renderingBackendAttribute =
    vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "Metal", nullptr);
  return renderingBackendAttribute;
}

void vtkMetalShaderProperty::AddVertexShaderReplacement(const std::string& originalValue,
  bool replaceFirst, // do this replacement before the default
  const std::string& replacementValue, bool replaceAll)
{
  this->Internals->UserShaderReplacements[vtkInternals::Key(
    static_cast<int>(MetalShaderStage::Vertex), originalValue, replaceFirst)] =
    vtkInternals::Value(replacementValue, replaceAll);
  this->Modified();
}

void vtkMetalShaderProperty::AddFragmentShaderReplacement(const std::string& originalValue,
  bool replaceFirst, // do this replacement before the default
  const std::string& replacementValue, bool replaceAll)
{
  this->Internals->UserShaderReplacements[vtkInternals::Key(
    static_cast<int>(MetalShaderStage::Fragment), originalValue, replaceFirst)] =
    vtkInternals::Value(replacementValue, replaceAll);
  this->Modified();
}

void vtkMetalShaderProperty::AddGeometryShaderReplacement(const std::string& originalValue,
  bool replaceFirst, // do this replacement before the default
  const std::string& replacementValue, bool replaceAll)
{
  this->Internals->UserShaderReplacements[vtkInternals::Key(
    static_cast<int>(MetalShaderStage::Geometry), originalValue, replaceFirst)] =
    vtkInternals::Value(replacementValue, replaceAll);
  this->Modified();
}

void vtkMetalShaderProperty::AddTessControlShaderReplacement(const std::string& originalValue,
  bool replaceFirst, // do this replacement before the default
  const std::string& replacementValue, bool replaceAll)
{
  this->Internals->UserShaderReplacements[vtkInternals::Key(
    static_cast<int>(MetalShaderStage::TessControl), originalValue, replaceFirst)] =
    vtkInternals::Value(replacementValue, replaceAll);
  this->Modified();
}

void vtkMetalShaderProperty::AddTessEvaluationShaderReplacement(const std::string& originalValue,
  bool replaceFirst, // do this replacement before the default
  const std::string& replacementValue, bool replaceAll)
{
  this->Internals->UserShaderReplacements[vtkInternals::Key(
    static_cast<int>(MetalShaderStage::TessEvaluation), originalValue, replaceFirst)] =
    vtkInternals::Value(replacementValue, replaceAll);
  this->Modified();
}

int vtkMetalShaderProperty::GetNumberOfShaderReplacements()
{
  return static_cast<int>(this->Internals->UserShaderReplacements.size());
}

std::string vtkMetalShaderProperty::GetNthShaderReplacementTypeAsString(vtkIdType index)
{
  if (index >= static_cast<vtkIdType>(this->Internals->UserShaderReplacements.size()))
  {
    vtkErrorMacro(<< "Trying to access out of bound shader replacement.");
    return std::string("");
  }
  auto it = this->Internals->UserShaderReplacements.begin();
  std::advance(it, index);
  return MetalShaderStageString(static_cast<MetalShaderStage>(std::get<0>(it->first)));
}

void vtkMetalShaderProperty::GetNthShaderReplacement(vtkIdType index, std::string& name,
  bool& replaceFirst, std::string& replacementValue, bool& replaceAll)
{
  if (index >= static_cast<vtkIdType>(this->Internals->UserShaderReplacements.size()))
  {
    vtkErrorMacro(<< "Trying to access out of bound shader replacement.");
    return;
  }
  auto it = this->Internals->UserShaderReplacements.begin();
  std::advance(it, index);
  name = std::get<1>(it->first);
  replaceFirst = std::get<2>(it->first);
  replacementValue = std::get<0>(it->second);
  replaceAll = std::get<1>(it->second);
}

void vtkMetalShaderProperty::ClearVertexShaderReplacement(
  const std::string& originalValue, bool replaceFirst)
{
  if (this->Internals->UserShaderReplacements.erase(
        vtkInternals::Key(static_cast<int>(MetalShaderStage::Vertex), originalValue, replaceFirst)) >
    0)
  {
    this->Modified();
  }
}

void vtkMetalShaderProperty::ClearFragmentShaderReplacement(
  const std::string& originalValue, bool replaceFirst)
{
  if (this->Internals->UserShaderReplacements.erase(
        vtkInternals::Key(static_cast<int>(MetalShaderStage::Fragment), originalValue, replaceFirst)) >
    0)
  {
    this->Modified();
  }
}

void vtkMetalShaderProperty::ClearGeometryShaderReplacement(
  const std::string& originalValue, bool replaceFirst)
{
  if (this->Internals->UserShaderReplacements.erase(
        vtkInternals::Key(static_cast<int>(MetalShaderStage::Geometry), originalValue, replaceFirst)) >
    0)
  {
    this->Modified();
  }
}

void vtkMetalShaderProperty::ClearTessControlShaderReplacement(
  const std::string& originalValue, bool replaceFirst)
{
  if (this->Internals->UserShaderReplacements.erase(
        vtkInternals::Key(static_cast<int>(MetalShaderStage::TessControl), originalValue,
          replaceFirst)) > 0)
  {
    this->Modified();
  }
}

void vtkMetalShaderProperty::ClearTessEvaluationShaderReplacement(
  const std::string& originalValue, bool replaceFirst)
{
  if (this->Internals->UserShaderReplacements.erase(
        vtkInternals::Key(static_cast<int>(MetalShaderStage::TessEvaluation), originalValue,
          replaceFirst)) > 0)
  {
    this->Modified();
  }
}

void vtkMetalShaderProperty::ClearAllVertexShaderReplacements()
{
  this->ClearAllShaderReplacementsForStage(static_cast<int>(MetalShaderStage::Vertex));
}

void vtkMetalShaderProperty::ClearAllFragmentShaderReplacements()
{
  this->ClearAllShaderReplacementsForStage(static_cast<int>(MetalShaderStage::Fragment));
}

void vtkMetalShaderProperty::ClearAllGeometryShaderReplacements()
{
  this->ClearAllShaderReplacementsForStage(static_cast<int>(MetalShaderStage::Geometry));
}

void vtkMetalShaderProperty::ClearAllTessControlShaderReplacements()
{
  this->ClearAllShaderReplacementsForStage(static_cast<int>(MetalShaderStage::TessControl));
}

void vtkMetalShaderProperty::ClearAllTessEvalShaderReplacements()
{
  this->ClearAllShaderReplacementsForStage(static_cast<int>(MetalShaderStage::TessEvaluation));
}

//------------------------------------------------------------------------------
void vtkMetalShaderProperty::ClearAllShaderReplacements()
{
  this->SetVertexShaderCode(nullptr);
  this->SetFragmentShaderCode(nullptr);
  this->SetGeometryShaderCode(nullptr);
  this->SetTessControlShaderCode(nullptr);
  this->SetTessEvaluationShaderCode(nullptr);
  this->Internals->UserShaderReplacements.clear();
  this->Modified();
}

//------------------------------------------------------------------------------
void vtkMetalShaderProperty::ClearAllShaderReplacementsForStage(int stage)
{
  bool modified = false;
  // First clear all shader code
  if ((stage == static_cast<int>(MetalShaderStage::Vertex)) && this->VertexShaderCode)
  {
    this->SetVertexShaderCode(nullptr);
    modified = true;
  }
  else if ((stage == static_cast<int>(MetalShaderStage::Fragment)) && this->FragmentShaderCode)
  {
    this->SetFragmentShaderCode(nullptr);
    modified = true;
  }
  else if ((stage == static_cast<int>(MetalShaderStage::Geometry)) && this->GeometryShaderCode)
  {
    this->SetGeometryShaderCode(nullptr);
    modified = true;
  }
  else if ((stage == static_cast<int>(MetalShaderStage::TessControl)) && this->TessControlShaderCode)
  {
    this->SetTessControlShaderCode(nullptr);
    modified = true;
  }
  else if ((stage == static_cast<int>(MetalShaderStage::TessEvaluation)) &&
    this->TessEvaluationShaderCode)
  {
    this->SetTessEvaluationShaderCode(nullptr);
    modified = true;
  }

  // Now clear custom tag replacements
  for (auto it = this->Internals->UserShaderReplacements.begin();
       it != this->Internals->UserShaderReplacements.end();)
  {
    if (std::get<0>(it->first) == stage)
    {
      this->Internals->UserShaderReplacements.erase(it++);
      modified = true;
    }
    else
    {
      ++it;
    }
  }
  if (modified)
  {
    this->Modified();
  }
}

//------------------------------------------------------------------------------
void vtkMetalShaderProperty::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
VTK_ABI_NAMESPACE_END
