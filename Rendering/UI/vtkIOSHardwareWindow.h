// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkIOSHardwareWindow
 * @brief   represents a window in an iOS UI
 *
 * vtkIOSHardwareWindow is a class for managing a native iOS window.
 * It is backed by a UIWindow and its view is configured with a CAMetalLayer,
 * making it suitable for Metal-based rendering.
 */

#ifndef vtkIOSHardwareWindow_h
#define vtkIOSHardwareWindow_h

#include "vtkHardwareWindow.h"
#include "vtkRenderingUIModule.h" // For export macro

// Forward declare Objective-C classes
#ifdef __OBJC__
@class UIWindow;
@class UIView;
#else
class UIWindow;
class UIView;
#endif

VTK_ABI_NAMESPACE_BEGIN

class VTKRENDERINGUI_EXPORT vtkIOSHardwareWindow : public vtkHardwareWindow
{
public:
  static vtkIOSHardwareWindow* New();
  vtkTypeMacro(vtkIOSHardwareWindow, vtkHardwareWindow);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /**
   * Get the native iOS window object.
   */
  UIWindow* GetWindowId();

  /**
   * Get the native iOS view object.
   */
  UIView* GetViewId();

  /**
   * Get the metal layer object.
   */
  void* GetMetalLayer();

  // vtkHardwareWindow overrides
  void Create() override;
  void Destroy() override;

  void* GetGenericWindowId() override;
  void* GetGenericParentId() override;

  ///@{
  /**
   * Set the size of the window in screen coordinates.
   */
  void SetSize(int, int) override;
  using vtkHardwareWindow::SetSize;
  ///@}

  ///@{
  /**
   * Set the position of the window in screen coordinates.
   */
  void SetPosition(int, int) override;
  using vtkHardwareWindow::SetPosition;
  ///@}

  /**
   * Set the name (title) of the window.
   */
  void SetWindowName(const char*) override;

protected:
  vtkIOSHardwareWindow();
  ~vtkIOSHardwareWindow() override;

  UIWindow* WindowId = nullptr;
  UIView* ViewId = nullptr;

  bool OwnsWindow = false;

private:
  vtkIOSHardwareWindow(const vtkIOSHardwareWindow&) = delete;
  void operator=(const vtkIOSHardwareWindow&) = delete;
};

VTK_ABI_NAMESPACE_END
#endif // vtkIOSHardwareWindow_h
