// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#ifndef vtkMetalRayCastImageDisplayHelper_h
#define vtkMetalRayCastImageDisplayHelper_h

#include "vtkRenderingMetalModule.h" // For export macro
#include "vtkRayCastImageDisplayHelper.h"
#include "vtkWrappingHints.h"

VTK_ABI_NAMESPACE_BEGIN
class vtkFixedPointRayCastImage;
class vtkOverrideAttribute;

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkMetalRayCastImageDisplayHelper
  : public vtkRayCastImageDisplayHelper
{
public:
  static vtkMetalRayCastImageDisplayHelper* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalRayCastImageDisplayHelper, vtkRayCastImageDisplayHelper);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  void RenderTexture(vtkVolume* vol, vtkRenderer* ren, int imageMemorySize[2],
    int imageViewportSize[2], int imageInUseSize[2], int imageOrigin[2], float requestedDepth,
    unsigned char* image) override;

  void RenderTexture(vtkVolume* vol, vtkRenderer* ren, int imageMemorySize[2],
    int imageViewportSize[2], int imageInUseSize[2], int imageOrigin[2], float requestedDepth,
    unsigned short* image) override;

  void RenderTexture(
    vtkVolume* vol, vtkRenderer* ren, vtkFixedPointRayCastImage* image, float requestedDepth) override;

  void ReleaseGraphicsResources(vtkWindow*) override;

protected:
  vtkMetalRayCastImageDisplayHelper();
  ~vtkMetalRayCastImageDisplayHelper() override;

  void RenderTextureInternal(vtkVolume* vol, vtkRenderer* ren, int imageMemorySize[2],
    int imageViewportSize[2], int imageInUseSize[2], int imageOrigin[2], float requestedDepth,
    int imageScalarType, void* image);

  // Metal resources (owned, MRC)
  void* ShaderLibrary = nullptr;  // id<MTLLibrary>
  void* RenderPipeline = nullptr; // id<MTLRenderPipelineState>
  void* Sampler = nullptr;        // id<MTLSamplerState>

  // Pipeline cache key: the pipeline must be rebuilt when the MSAA sample
  // count or the premultiplied-alpha blend mode changes.
  uint32_t LastSampleCount = 0;
  bool LastPreMultiplied = false;

private:
  vtkMetalRayCastImageDisplayHelper(const vtkMetalRayCastImageDisplayHelper&) = delete;
  void operator=(const vtkMetalRayCastImageDisplayHelper&) = delete;
};

#define vtkMetalRayCastImageDisplayHelper_OVERRIDE_ATTRIBUTES \
  vtkMetalRayCastImageDisplayHelper::CreateOverrideAttributes()

VTK_ABI_NAMESPACE_END
#endif
