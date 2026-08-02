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
#include "vtkOverrideAttribute.h"
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
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalHardwareSelector, vtkHardwareSelector);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /**
   * Single-pass capture: re-render the scene (IDs written to color attachment 1),
   * then read back the IDs texture.
   */
  bool CaptureBuffers() override;

  /**
   * Return the per-render prop ID (index into the visible PropArray) for the
   * given prop, or -1 if the prop is not part of the selection render.
   *
   * The Metal renderer renders props through its normal opaque/translucent
   * path during a selection render (the base class's per-pass Render() loop is
   * not used), so the selector cannot rely on vtkHardwareSelector::Render to
   * assign PropID per prop. Instead, mappers look up their own index here while
   * they render; the index matches what GetPropFromID() expects on readback.
   */
  int GetPropID(vtkProp* prop) const;

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
  int PropCount = 0;

  vtkMetalHardwareSelector(const vtkMetalHardwareSelector&) = delete;
  void operator=(const vtkMetalHardwareSelector&) = delete;
};

#define vtkMetalHardwareSelector_OVERRIDE_ATTRIBUTES \
  vtkMetalHardwareSelector::CreateOverrideAttributes()

VTK_ABI_NAMESPACE_END
#endif
