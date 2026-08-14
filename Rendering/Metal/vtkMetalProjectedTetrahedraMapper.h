// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-FileCopyrightText: Copyright 2003 Sandia Corporation
// SPDX-License-Identifier: LicenseRef-BSD-3-Clause-Sandia-USGov

/**
 * @class   vtkMetalProjectedTetrahedraMapper
 * @brief   Metal implementation of the projected tetrahedra (PT) volume mapper.
 *
 * Port of vtkOpenGLProjectedTetrahedraMapper that renders the Shirley & Tuchman
 * / Wylie 2002 projection into the Metal renderer's volume pass. The CPU side
 * (point transform, color/depth packing, per-tet segment intersection) is
 * shared logic, while the triangle draw uses a small vertex/fragment shader
 * pair in MetalShaders.metal that mirrors vtkglProjectedTetrahedraVS/FS.
 */

#ifndef vtkMetalProjectedTetrahedraMapper_h
#define vtkMetalProjectedTetrahedraMapper_h

#include "vtkNew.h" // For ivars
#include "vtkOverrideAttribute.h"
#include "vtkProjectedTetrahedraMapper.h"
#include "vtkRenderingMetalModule.h" // For export macro
#include "vtkTimeStamp.h"            // For time stamps
#include "vtkWrappingHints.h"        // For VTK_MARSHALAUTO

VTK_ABI_NAMESPACE_BEGIN
class vtkFloatArray;
class vtkMatrix4x4;
class vtkRenderWindow;
class vtkUnsignedCharArray;
class vtkVolumeProperty;

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkMetalProjectedTetrahedraMapper
  : public vtkProjectedTetrahedraMapper
{
public:
  vtkTypeMacro(vtkMetalProjectedTetrahedraMapper, vtkProjectedTetrahedraMapper);
  static vtkMetalProjectedTetrahedraMapper* New();
  void PrintSelf(ostream& os, vtkIndent indent) override;

  static vtkOverrideAttribute* CreateOverrideAttributes();

  void ReleaseGraphicsResources(vtkWindow* window) override;

  void Render(vtkRenderer* renderer, vtkVolume* volume) override;

  /**
   * Return true if the rendering context provides the necessary
   * functionality to use this class.
   */
  bool IsSupported(vtkRenderWindow* context) override;

protected:
  vtkMetalProjectedTetrahedraMapper();
  ~vtkMetalProjectedTetrahedraMapper() override;

  // CPU-side PT state (mirrors vtkOpenGLProjectedTetrahedraMapper).
  vtkUnsignedCharArray* Colors;
  int UsingCellColors;

  vtkFloatArray* TransformedPoints;

  float MaxCellSize;
  vtkTimeStamp InputAnalyzedTime;
  vtkTimeStamp ColorsMappedTime;

  int GaveError;

  vtkVolumeProperty* LastProperty;

  float* SqrtTable;
  float SqrtTableBias;

  vtkNew<vtkMatrix4x4> tmpMat;
  vtkNew<vtkMatrix4x4> tmpMat2;

  // Metal resources held as void* (MRC; see the .mm file for the casts).
  void* CachedLibrary = nullptr;   // id<MTLLibrary>
  void* PipelineState = nullptr;   // id<MTLRenderPipelineState>
  void* DepthStencilState = nullptr; // id<MTLDepthStencilState>
  void* CachedDevice = nullptr;    // id<MTLDevice> this state was built for
  int CachedSampleCount = 0;

  void* PositionsBuffer = nullptr; // id<MTLBuffer>, device float3*
  uint64_t PositionsCapacity = 0;
  void* ColorsBuffer = nullptr;    // id<MTLBuffer>, device uchar4*
  uint64_t ColorsCapacity = 0;
  void* AttenDepthBuffer = nullptr; // id<MTLBuffer>, device float2*
  uint64_t AttenDepthCapacity = 0;
  void* IndexBuffer = nullptr;     // id<MTLBuffer>, device uint*
  uint64_t IndexCapacity = 0;

  virtual void ProjectTetrahedra(vtkRenderer* renderer, vtkVolume* volume);

  float GetCorrectedDepth(float x, float y, float z1, float z2,
    const float inverse_projection_mat[16], int use_linear_depth_correction,
    float linear_depth_correction);

private:
  vtkMetalProjectedTetrahedraMapper(const vtkMetalProjectedTetrahedraMapper&) = delete;
  void operator=(const vtkMetalProjectedTetrahedraMapper&) = delete;
};

#define vtkMetalProjectedTetrahedraMapper_OVERRIDE_ATTRIBUTES \
  vtkMetalProjectedTetrahedraMapper::CreateOverrideAttributes()

VTK_ABI_NAMESPACE_END
#endif
