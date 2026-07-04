// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalCamera.h"
#include "vtkMatrix4x4.h"
#include "vtkObjectFactory.h"
#include "vtkRenderer.h"
#include "vtkViewport.h"

#import <Metal/Metal.h>

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalCamera);

vtkMetalCamera::vtkMetalCamera() = default;
vtkMetalCamera::~vtkMetalCamera() = default;

void vtkMetalCamera::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

void vtkMetalCamera::Render(vtkRenderer* ren)
{
  this->ComputeViewTransform();

  int* size = ren->GetSize();
  double aspect = (size[1] > 0) ? static_cast<double>(size[0]) / size[1] : 1.0;

  vtkMatrix4x4* viewMatrix = this->GetViewTransformMatrix();
  // Metal maps clip-space Z to [0, 1] (not [-1, 1] like OpenGL).
  // GetProjectionTransformMatrix with nearz=0, farz=1 produces a projection
  // compatible with Metal's depth convention, matching the WebGPU camera.
  // GetCompositeProjectionTransformMatrix would produce Z in [-1, 1], causing
  // Metal to cull all visible geometry (Z < 0 is clipped).
  vtkMatrix4x4* projMatrix =
    this->GetProjectionTransformMatrix(aspect, /*nearz=*/0, /*farz=*/1);
  vtkMatrix4x4* modelMatrix = this->GetModelTransformMatrix();

  for (int col = 0; col < 4; ++col)
  {
    for (int row = 0; row < 4; ++row)
    {
      this->CachedSceneTransforms.ViewMatrix[col][row] =
        static_cast<float>(viewMatrix->GetElement(row, col));
      this->CachedSceneTransforms.ProjectionMatrix[col][row] =
        static_cast<float>(projMatrix->GetElement(row, col));
      this->CachedSceneTransforms.ModelMatrix[col][row] =
        static_cast<float>(modelMatrix->GetElement(row, col));
    }
  }

  // Compute normal matrix (inverse-transpose of view * model)
  vtkNew<vtkMatrix4x4> vm;
  vtkMatrix4x4::Multiply4x4(viewMatrix, modelMatrix, vm);

  double m[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      m[i][j] = vm->GetElement(i, j);

  double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
               m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

  if (fabs(det) > 1e-10)
  {
    double invDet = 1.0 / det;
    double inv[3][3];
    inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * invDet;
    inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invDet;
    inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invDet;
    inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invDet;
    inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invDet;
    inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * invDet;
    inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * invDet;
    inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * invDet;
    inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * invDet;

    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        this->CachedSceneTransforms.NormalMatrix[i][j] = static_cast<float>(inv[j][i]);
  }

  int* sz = ren->GetSize();
  this->CachedSceneTransforms.Viewport[0] = 0;
  this->CachedSceneTransforms.Viewport[1] = 0;
  this->CachedSceneTransforms.Viewport[2] = static_cast<float>(sz[0]);
  this->CachedSceneTransforms.Viewport[3] = static_cast<float>(sz[1]);

  this->KeyMatrixTime.Modified();
}

void vtkMetalCamera::UpdateViewport(vtkRenderer* ren)
{
  // Viewport is set by the renderer when creating the render pass encoder
}

VTK_ABI_NAMESPACE_END
