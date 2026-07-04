// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
#ifndef vtkIOSMetalRenderWindow_h
#define vtkIOSMetalRenderWindow_h

#include "vtkMetalRenderWindow.h"
#include "vtkRenderingMetalModule.h"
#include "vtkWrappingHints.h"

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

  UIView* GetViewId();

  void SetSize(int width, int height) override;

protected:
  vtkIOSMetalRenderWindow();
  ~vtkIOSMetalRenderWindow() override;

private:
  vtkIOSMetalRenderWindow(const vtkIOSMetalRenderWindow&) = delete;
  void operator=(const vtkIOSMetalRenderWindow&) = delete;

  UIView* ViewId = nullptr;
};

VTK_ABI_NAMESPACE_END
#endif
