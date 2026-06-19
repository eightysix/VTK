// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @class   vtkWebGPUGPUInfoList
 * @brief   Minimal vtkGPUInfoList implementation for the WebGPU backend.
 *
 * The WebGPU rendering module does not link against vtkRenderingOpenGL2, which
 * is the only module that otherwise registers a concrete vtkGPUInfoList factory
 * override (vtkDummyGPUInfoList). Without a registered override,
 * vtkGPUInfoList::New() returns nullptr and the vtkGPUVolumeRayCastMapper
 * constructor crashes with EXC_BAD_ACCESS when it calls l->Probe().
 *
 * This class registers itself as the "WebGPU" override so that
 * vtkGPUInfoList::New() returns a valid, zero-GPU instance on any
 * platform where only the WebGPU backend is present.
 *
 * @sa vtkGPUInfo vtkGPUInfoList
 */

#ifndef vtkWebGPUGPUInfoList_h
#define vtkWebGPUGPUInfoList_h

#include "vtkGPUInfoList.h"
#include "vtkRenderingWebGPUModule.h" // For export macro

VTK_ABI_NAMESPACE_BEGIN

class vtkOverrideAttribute;

class VTKRENDERINGWEBGPU_EXPORT vtkWebGPUGPUInfoList : public vtkGPUInfoList
{
public:
  static vtkWebGPUGPUInfoList* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkWebGPUGPUInfoList, vtkGPUInfoList);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /**
   * Satisfies the postcondition IsProbed() == true with an empty GPU list.
   * GPU memory information is not available through this stub; the
   * vtkGPUVolumeRayCastMapper will fall back to its built-in 128 MB default.
   * \post probed: IsProbed()
   */
  void Probe() override;

protected:
  vtkWebGPUGPUInfoList();
  ~vtkWebGPUGPUInfoList() override;

private:
  vtkWebGPUGPUInfoList(const vtkWebGPUGPUInfoList&) = delete;
  void operator=(const vtkWebGPUGPUInfoList&) = delete;
};

#define vtkWebGPUGPUInfoList_OVERRIDE_ATTRIBUTES vtkWebGPUGPUInfoList::CreateOverrideAttributes()

VTK_ABI_NAMESPACE_END
#endif
