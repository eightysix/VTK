// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkMetalHardwareSelector
 * @brief   implements device-specific picking for Metal rendering
 *
 * Leverages Metal's single-pass picking (IDs in color attachment 1)
 * to avoid the expensive multi-pass approach of the base class.
 *
 * @sa vtkHardwareSelector
 */

#ifndef vtkMetalHardwareSelector_h
#define vtkMetalHardwareSelector_h

#include "vtkHardwareSelector.h"
#include "vtkRenderingMetalModule.h" // for export macro
#include "vtkWrappingHints.h"        // for VTK_MARSHALAUTO
#include "vtkNew.h"

class vtkUnsignedIntArray;

VTK_ABI_NAMESPACE_BEGIN

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkMetalHardwareSelector
  : public vtkHardwareSelector
{
public:
  static vtkMetalHardwareSelector* New();
  VTK_NEWINSTANCE
  vtkTypeMacro(vtkMetalHardwareSelector, vtkHardwareSelector);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /**
   * Single-pass capture: re-render the scene (IDs written to color attachment 1),
   * then read back the IDs texture.
   */
  bool CaptureBuffers() override;

  void BeginRenderProp() override {}
  void EndRenderProp() override {}
  void RenderCompositeIndex(unsigned int) override {}
  void RenderProcessId(unsigned int) override {}

  void BeginSelection() override;
  void EndSelection() override;

  vtkProp* GetPropFromID(int id) override;

  PixelInformation GetPixelInformation(
    const unsigned int inDisplayPosition[2], int maxDist,
    unsigned int outSelectedPosition[2]) override;

protected:
  vtkMetalHardwareSelector();
  ~vtkMetalHardwareSelector() override;

  void PreCapturePass(int) override {}
  void PostCapturePass(int) override {}
  void BeginRenderProp(vtkRenderWindow*) override {}
  void EndRenderProp(vtkRenderWindow*) override {}
  void SavePixelBuffer(int) override {}
  void ReleasePixBuffers() override;

  int Convert(int xx, int yy, unsigned char* pb) override;

private:
  struct Ids
  {
    unsigned int AttributeId;
    unsigned int PropId;
    unsigned int CompositeId;
    unsigned int ProcessId;
  };

  vtkNew<vtkUnsignedIntArray> IdBuffer;
  vtkProp** PropArray = nullptr;

  vtkMetalHardwareSelector(const vtkMetalHardwareSelector&) = delete;
  void operator=(const vtkMetalHardwareSelector&) = delete;
};

VTK_ABI_NAMESPACE_END
#endif
