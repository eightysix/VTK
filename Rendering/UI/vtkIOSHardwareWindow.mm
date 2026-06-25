// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#import <UIKit/UIKit.h>
#import <QuartzCore/CAMetalLayer.h>

#import "vtkIOSHardwareWindow.h"
#import "vtkCommand.h"
#import "vtkObjectFactory.h"

// Custom UIView that uses a CAMetalLayer backing.
// On iOS, override +layerClass to return CAMetalLayer; this is the
// correct way to provide a Metal-compatible layer.  setWantsLayer:
// is a macOS- (AppKit) only method.
@interface vtkIOSMetalLayerView : UIView
@end

@implementation vtkIOSMetalLayerView
+ (Class)layerClass
{
  return [CAMetalLayer class];
}
@end

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkIOSHardwareWindow);

//------------------------------------------------------------------------------
vtkIOSHardwareWindow::vtkIOSHardwareWindow()
{
  this->ViewId = nullptr;
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
  if (this->ViewId)
  {
    return;
  }

  @autoreleasepool
  {
    CGFloat initialWidth = (this->Size[0] > 0) ? this->Size[0] : 300;
    CGFloat initialHeight = (this->Size[1] > 0) ? this->Size[1] : 300;

    CGRect frame = CGRectMake(0, 0, initialWidth, initialHeight);

    vtkIOSMetalLayerView* view = [[vtkIOSMetalLayerView alloc] initWithFrame:frame];
    this->ViewId = view;
    [view setOpaque:YES];
    CAMetalLayer* metalLayer = (CAMetalLayer*)[view layer];
    metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    metalLayer.contentsScale = [UIScreen mainScreen].nativeScale;

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
  if (this->ViewId)
  {
    [this->ViewId removeFromSuperview];
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
    // Do not set the UIView frame here; the size is in pixels (for the Metal drawable),
    // while the view frame is in points. The view is positioned via autoresizing or by
    // the containing view controller (e.g. vtkView.frame = self.view.bounds).
    this->Modified();
  }
}

//------------------------------------------------------------------------------
void vtkIOSHardwareWindow::SetPosition(int x, int y)
{
  if (this->Position[0] != x || this->Position[1] != y)
  {
    this->Superclass::SetPosition(x, y);
    if (this->ViewId)
    {
      CGRect frame = [this->ViewId frame];
      frame.origin.x = x;
      frame.origin.y = y;
      [this->ViewId setFrame:frame];
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
UIView* vtkIOSHardwareWindow::GetViewId()
{
  return this->ViewId;
}

//------------------------------------------------------------------------------
void* vtkIOSHardwareWindow::GetMetalLayer()
{
  if (!this->ViewId)
  {
    return nullptr;
  }
  return (__bridge void*)[this->ViewId layer];
}

//------------------------------------------------------------------------------
void* vtkIOSHardwareWindow::GetGenericWindowId()
{
  return (__bridge void*)this->ViewId;
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
  os << indent << "ViewId: " << this->ViewId << "\n";
}

VTK_ABI_NAMESPACE_END
