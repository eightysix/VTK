// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#import "vtkIOSHardwareWindow.h"
#import "vtkCommand.h"
#import "vtkObjectFactory.h"
#import <UIKit/UIKit.h>

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkIOSHardwareWindow);

//------------------------------------------------------------------------------
vtkIOSHardwareWindow::vtkIOSHardwareWindow()
{
  this->WindowId = nullptr;
  this->ViewId = nullptr;
  this->OwnsWindow = false;
  this->Mapped = false;
  this->Platform = "iOS";
}

//------------------------------------------------------------------------------
vtkIOSHardwareWindow::~vtkIOSHardwareWindow()
{
  this->Destroy();
}

//------------------------------------------------------------------------------
void vtkIOSHardwareWindow::Create()
{
  if (this->WindowId)
  {
    return;
  }

  @autoreleasepool
  {
    CGFloat initialWidth = (this->Size[0] > 0) ? this->Size[0] : 300;
    CGFloat initialHeight = (this->Size[1] > 0) ? this->Size[1] : 300;

    CGRect screenRect = [[UIScreen mainScreen] bounds];
    CGRect frame = CGRectMake(this->Position[0],
      screenRect.size.height - this->Position[1] - initialHeight,
      initialWidth, initialHeight);

    this->WindowId = [[UIWindow alloc] initWithFrame:frame];
    if (!this->WindowId)
    {
      vtkErrorMacro("Could not create UIWindow.");
      return;
    }
    this->OwnsWindow = true;

    UIView* view = [[UIView alloc] initWithFrame:frame];
    this->ViewId = view;
    [view setOpaque:YES];

    // Configure a CAMetalLayer as the view's backing layer
    [view setWantsLayer:YES];

    [this->WindowId addSubview:view];

    if (this->WindowName)
    {
      this->SetWindowName(this->WindowName);
    }

    this->Mapped = true;
  }
}

//------------------------------------------------------------------------------
void vtkIOSHardwareWindow::Destroy()
{
  if (this->OwnsWindow && this->WindowId)
  {
    this->WindowId = nullptr;
    this->ViewId = nullptr;
  }
  this->Mapped = false;
}

//------------------------------------------------------------------------------
void vtkIOSHardwareWindow::SetSize(int width, int height)
{
  if (this->Size[0] != width || this->Size[1] != height)
  {
    this->Superclass::SetSize(width, height);
    if (this->WindowId && this->ViewId)
    {
      CGRect frame = CGRectMake(0, 0, (CGFloat)width, (CGFloat)height);
      [this->ViewId setFrame:frame];
    }
    this->Modified();
  }
}

//------------------------------------------------------------------------------
void vtkIOSHardwareWindow::SetPosition(int x, int y)
{
  if (this->Position[0] != x || this->Position[1] != y)
  {
    this->Superclass::SetPosition(x, y);
    if (this->WindowId)
    {
      CGRect frame = [this->WindowId frame];
      frame.origin.x = x;
      frame.origin.y = y;
      [this->WindowId setFrame:frame];
    }
    this->Modified();
  }
}

//------------------------------------------------------------------------------
void vtkIOSHardwareWindow::SetWindowName(const char* name)
{
  this->Superclass::SetWindowName(name);
}

//------------------------------------------------------------------------------
UIWindow* vtkIOSHardwareWindow::GetWindowId()
{
  return this->WindowId;
}

//------------------------------------------------------------------------------
UIView* vtkIOSHardwareWindow::GetViewId()
{
  return this->ViewId;
}

//------------------------------------------------------------------------------
void* vtkIOSHardwareWindow::GetMetalLayer()
{
  return (__bridge void*)[this->ViewId layer];
}

//------------------------------------------------------------------------------
void* vtkIOSHardwareWindow::GetGenericWindowId()
{
  return (__bridge void*)this->WindowId;
}

//------------------------------------------------------------------------------
void* vtkIOSHardwareWindow::GetGenericParentId()
{
  return nullptr;
}

//------------------------------------------------------------------------------
void vtkIOSHardwareWindow::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "WindowId: " << this->WindowId << "\n";
  os << indent << "ViewId: " << this->ViewId << "\n";
  os << indent << "OwnsWindow: " << (this->OwnsWindow ? "Yes" : "No") << "\n";
}

VTK_ABI_NAMESPACE_END
