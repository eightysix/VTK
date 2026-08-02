// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkMetalTexture
 * @brief   Metal texture map
 *
 * vtkMetalTexture is a concrete implementation of the abstract class
 * vtkTexture for the Metal rendering backend. The Metal poly data mapper
 * uploads and binds actor textures itself (see
 * vtkMetalPolyDataMapper::UpdateActorTexture), so the Load/PostRender hooks
 * here are no-ops; the base-class Render keeps the input pipeline up to date.
 */

#ifndef vtkMetalTexture_h
#define vtkMetalTexture_h

#include "vtkRenderingMetalModule.h" // For export macro
#include "vtkTexture.h"
#include "vtkWrappingHints.h" // For VTK_MARSHALAUTO

VTK_ABI_NAMESPACE_BEGIN
class vtkOverrideAttribute;

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkMetalTexture : public vtkTexture
{
public:
  static vtkMetalTexture* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalTexture, vtkTexture);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /**
   * Implement base class method. The Metal mapper uploads the texture itself,
   * so this does nothing.
   */
  void Load(vtkRenderer*) override;

  /**
   * Clean up after the rendering is complete.
   */
  void PostRender(vtkRenderer*) override;

  /**
   * Release any graphics resources that are being consumed by this texture.
   */
  void ReleaseGraphicsResources(vtkWindow*) override;

protected:
  vtkMetalTexture();
  ~vtkMetalTexture() override;

private:
  vtkMetalTexture(const vtkMetalTexture&) = delete;
  void operator=(const vtkMetalTexture&) = delete;
};

#define vtkMetalTexture_OVERRIDE_ATTRIBUTES vtkMetalTexture::CreateOverrideAttributes()

VTK_ABI_NAMESPACE_END
#endif
