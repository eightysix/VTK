// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalCamera.h"

#include "vtkMatrix4x4.h"
#include "vtkMatrix3x3.h"
#include "vtkObjectFactory.h"
#include "vtkRenderer.h"
#include "vtkViewport.h"

#import <Metal/Metal.h>

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalCamera);

//------------------------------------------------------------------------------
vtkMetalCamera::vtkMetalCamera() = default;

//------------------------------------------------------------------------------
vtkMetalCamera::~vtkMetalCamera() = default;

//------------------------------------------------------------------------------
void vtkMetalCamera::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//------------------------------------------------------------------------------
void vtkMetalCamera::Render(vtkRenderer* ren)
{
  // Compute key matrices
  this->ComputeViewTransform();
  this->ComputeProjectionTransform(ren, ren->GetViewport()[2] / ren->GetViewport()[3]);

  vtkMatrix4x4* viewMatrix = this->GetViewTransformMatrix();
  vtkMatrix4x4* projMatrix = this->GetProjectionTransformMatrix(
    ren->GetViewport()[2] / ren->GetViewport()[3]);
  vtkMatrix4x4* modelMatrix = this->GetModelTransformMatrix();

  // Copy to our cached struct (column-major for Metal)
  for (int col = 0; col < 4; ++col)
  {
    for (int row = 0; row < 4; ++row)
    {
      this->CachedSceneTransforms.ViewMatrix[col][row] =
        static_cast<vtkTypeFloat32>(viewMatrix->GetElement(row, col));
      this->CachedSceneTransforms.ProjectionMatrix[col][row] =
        static_cast<vtkTypeFloat32>(projMatrix->GetElement(row, col));
      this->CachedSceneTransforms.ModelMatrix[col][row] =
        static_cast<vtkTypeFloat32>(modelMatrix->GetElement(row, col));
    }
  }

  // Compute normal matrix (inverse-transpose of view * model)
  vtkNew<vtkMatrix4x4> vm;
  vtkMatrix4x4::Multiply4x4(viewMatrix, modelMatrix, vm);

  // Extract 3x3 and compute inverse-transpose
  double m[3][3];
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      m[i][j] = vm->GetElement(i, j);
    }
  }

  // Compute cofactors for 3x3 inverse
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

    // Transpose (inverse-transpose = transpose of inverse)
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        this->CachedSceneTransforms.NormalMatrix[i][j] =
          static_cast<vtkTypeFloat32>(inv[j][i]);
      }
    }
  }

  // Store viewport
  int* size = ren->GetSize();
  this->CachedSceneTransforms.Viewport[0] = 0;
  this->CachedSceneTransforms.Viewport[1] = 0;
  this->CachedSceneTransforms.Viewport[2] = static_cast<vtkTypeFloat32>(size[0]);
  this->CachedSceneTransforms.Viewport[3] = static_cast<vtkTypeFloat32>(size[1]);

  this->KeyMatrixTime.Modified();
}

//------------------------------------------------------------------------------
void vtkMetalCamera::UpdateViewport(vtkRenderer* ren)
{
  int* size = ren->GetSize();
  double* viewport = ren->GetViewport();

  int x = static_cast<int>(viewport[0] * size[0]);
  int y = static_cast<int>(viewport[1] * size[1]);
  int w = static_cast<int>(viewport[2] * size[0]);
  int h = static_cast<int>(viewport[3] * size[1]);

  // Metal viewport has Y flipped compared to OpenGL
  MTLViewport metalViewport;
  metalViewport.originX = x;
  metalViewport.originY = y;
  metalViewport.width = w;
  metalViewport.height = h;
  metalViewport.znear = 0.0;
  metalViewport.zfar = 1.0;

  // The encoder is set by the renderer - we just store the viewport info
  // The actual setViewport call happens in the renderer's DeviceRender
}

VTK_ABI_NAMESPACE_END
