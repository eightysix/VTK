// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkMetalCamera
 * @brief   Metal camera
 *
 * vtkMetalCamera is a concrete implementation of vtkCamera for Metal rendering.
 * It computes and caches the view/projection matrices for Metal shaders.
 */

#ifndef vtkMetalCamera_h
#define vtkMetalCamera_h

#include "vtkCamera.h"
#include "vtkMatrix4x4.h"
#include "vtkNew.h"
#include "vtkRenderingMetalModule.h" // for export macro
#include "vtkWrappingHints.h"        // for VTK_MARSHALAUTO

VTK_ABI_NAMESPACE_BEGIN

class vtkOverrideAttribute;

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkMetalCamera : public vtkCamera
{
public:
  static vtkMetalCamera* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalCamera, vtkCamera);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  void Render(vtkRenderer* renderer) override;
  void UpdateViewport(vtkRenderer* renderer) override;

  /**
   * Get the cached scene transform data (view, projection, normal matrices).
   */
  void* GetCachedSceneTransforms() { return &this->CachedSceneTransforms; }

protected:
  vtkMetalCamera();
  ~vtkMetalCamera() override;

private:
  vtkMetalCamera(const vtkMetalCamera&) = delete;
  void operator=(const vtkMetalCamera&) = delete;

  vtkTimeStamp KeyMatrixTime;
  vtkRenderer* LastRenderer = nullptr;
  vtkNew<vtkMatrix3x3> NormalMatrix;

  // Uniform struct matching the Metal shader layout
  struct SceneTransforms
  {
    vtkTypeFloat32 ViewMatrix[4][4] = {};
    vtkTypeFloat32 ProjectionMatrix[4][4] = {};
    vtkTypeFloat32 NormalMatrix[3][4] = {}; // 3x4 padded
    vtkTypeFloat32 ModelMatrix[4][4] = {};
    vtkTypeFloat32 Viewport[4] = {};
    vtkTypeUInt32 Flags = 0;
  };
  SceneTransforms CachedSceneTransforms;
};

VTK_ABI_NAMESPACE_END
#endif // vtkMetalCamera_h
