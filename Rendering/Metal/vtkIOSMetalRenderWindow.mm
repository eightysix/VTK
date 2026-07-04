// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkIOSMetalRenderWindow.h"

#include "vtkObjectFactory.h"

#import <UIKit/UIKit.h>
#import <QuartzCore/CAMetalLayer.h>

@interface vtkIOSMetalView : UIView
@end

@implementation vtkIOSMetalView
+ (Class)layerClass
{
  return [CAMetalLayer class];
}
@end

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkIOSMetalRenderWindow);

//------------------------------------------------------------------------------
vtkIOSMetalRenderWindow::vtkIOSMetalRenderWindow() = default;

//------------------------------------------------------------------------------
vtkIOSMetalRenderWindow::~vtkIOSMetalRenderWindow()
{
  if (this->ViewId)
  {
    [this->ViewId removeFromSuperview];
    this->ViewId = nullptr;
  }
}

//------------------------------------------------------------------------------
void vtkIOSMetalRenderWindow::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "ViewId: " << this->ViewId << "\n";
}

//------------------------------------------------------------------------------
UIView* vtkIOSMetalRenderWindow::GetViewId()
{
  // Create the view on first access
  if (!this->ViewId)
  {
    @autoreleasepool
    {
      CGFloat scale = [UIScreen mainScreen].nativeScale;
      CGFloat w = (this->Size[0] > 0) ? this->Size[0] / scale : 300;
      CGFloat h = (this->Size[1] > 0) ? this->Size[1] / scale : 300;
      CGRect frame = CGRectMake(0, 0, w, h);
      vtkIOSMetalView* view = [[vtkIOSMetalView alloc] initWithFrame:frame];
      [view setOpaque:YES];
      this->ViewId = view;

      CAMetalLayer* metalLayer = (CAMetalLayer*)[view layer];
      metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
      metalLayer.contentsScale = scale;
      metalLayer.framebufferOnly = YES;
      this->MetalLayer = (__bridge void*)metalLayer;
      CFRetain((__bridge CFTypeRef)metalLayer);
    }
  }
  return this->ViewId;
}

//------------------------------------------------------------------------------
void vtkIOSMetalRenderWindow::SetSize(int width, int height)
{
  this->Superclass::SetSize(width, height);

  if (this->ViewId)
  {
    @autoreleasepool
    {
      CGFloat scale = [UIScreen mainScreen].nativeScale;
      CGRect frame = CGRectMake(0, 0, width / scale, height / scale);
      [this->ViewId setFrame:frame];

      CAMetalLayer* layer = (CAMetalLayer*)[this->ViewId layer];
      layer.drawableSize = CGSizeMake(width, height);
    }
  }
}

VTK_ABI_NAMESPACE_END
