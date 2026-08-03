// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkMetalTexture
 * @brief   Metal texture map
 *
 * vtkMetalTexture is a concrete implementation of the abstract class
 * vtkTexture for the Metal rendering backend. It mirrors the OpenGL backend's
 * Load/Render model: the texture input image is uploaded to an id<MTLTexture>
 * and registered with the vtkMetalRenderWindow's per-unit texture registry
 * (vtkMetalRenderWindow::SetBoundTexture) keyed by GetTextureUnit(). 2D
 * mappers (vtkMetalPolyDataMapper2D) later resolve the same unit from the
 * actor's GENERAL_TEXTURE_UNIT property key and sample the registered texture,
 * so vtkTextMapper/vtkTexturedActor2D text and images render exactly like the
 * OpenGL path. The 3D poly-data mapper still uploads actor textures itself
 * (vtkMetalPolyDataMapper::UpdateActorTexture) and does not use this registry.
 */

#ifndef vtkMetalTexture_h
#define vtkMetalTexture_h

#include "vtkRenderingMetalModule.h" // For export macro
#include "vtkTexture.h"
#include "vtkWrappingHints.h" // For VTK_MARSHALAUTO

#include <memory>

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
   * Implement base class method. Uploads the input image into an
   * id<MTLTexture> and registers it with the render window's texture-unit
   * registry so 2D mappers can bind it during overlay rendering.
   */
  void Load(vtkRenderer*) override;

  /**
   * Clean up after the rendering is complete.
   */
  void PostRender(vtkRenderer*) override;

  /**
   * Each texture is registered in the render window's texture-unit registry
   * under its own unique unit (the base class always returns 0, which would
   * make distinct textures overwrite each other in the registry). The unit is
   * allocated once at construction and stays stable so Load() cache hits still
   * resolve the correct texture.
   */
  int GetTextureUnit() override;

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

  struct vtkMetalTextureInternals;
  std::unique_ptr<vtkMetalTextureInternals> Internals;
};

#define vtkMetalTexture_OVERRIDE_ATTRIBUTES vtkMetalTexture::CreateOverrideAttributes()

VTK_ABI_NAMESPACE_END
#endif
