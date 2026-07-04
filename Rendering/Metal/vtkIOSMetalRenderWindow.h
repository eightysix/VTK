// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkIOSMetalRenderWindow
 * @brief   iOS Metal rendering window
 *
 * vtkIOSMetalRenderWindow creates a UIView backed by a CAMetalLayer for
 * rendering on iOS. It manages the UIView lifecycle and provides the
 * native view handle for integration with UIKit view controllers.
 */

#ifndef vtkIOSMetalRenderWindow_h
#define vtkIOSMetalRenderWindow_h

#include "vtkMetalRenderWindow.h"
#include "vtkRenderingMetalModule.h" // for export macro
#include "vtkWrappingHints.h"        // for VTK_MARSHALAUTO

// Forward declare Objective-C classes
#ifdef __OBJC__
@class UIView;
#else
class UIView;
#endif

VTK_ABI_NAMESPACE_BEGIN

class vtkOverrideAttribute;

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkIOSMetalRenderWindow
  : public vtkMetalRenderWindow
{
public:
  static vtkIOSMetalRenderWindow* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkIOSMetalRenderWindow, vtkMetalRenderWindow);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /**
   * Get the native iOS view.
   */
  UIView* GetViewId();

  /**
   * Set the window size in pixels.
   */
  void SetSize(int width, int height) override;

protected:
  vtkIOSMetalRenderWindow();
  ~vtkIOSMetalRenderWindow() override;

  void CreateAWindow() override;

private:
  vtkIOSMetalRenderWindow(const vtkIOSMetalRenderWindow&) = delete;
  void operator=(const vtkIOSMetalRenderWindow&) = delete;

  UIView* ViewId = nullptr;
};

VTK_ABI_NAMESPACE_END
#endif // vtkIOSMetalRenderWindow_h
