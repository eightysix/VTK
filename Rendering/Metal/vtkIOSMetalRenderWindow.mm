// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkIOSMetalRenderWindow.h"

#include "vtkObjectFactory.h"

#import <UIKit/UIKit.h>
#import <QuartzCore/CAMetalLayer.h>

// Custom UIView that uses a CAMetalLayer backing.
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
void vtkIOSMetalRenderWindow::CreateAWindow()
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
    vtkIOSMetalView* view = [[vtkIOSMetalView alloc] initWithFrame:frame];
    [view setOpaque:YES];

    this->ViewId = view;

    // Set the Metal layer properties
    CAMetalLayer* metalLayer = (CAMetalLayer*)[view layer];
    metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    metalLayer.contentsScale = [UIScreen mainScreen].nativeScale;
    metalLayer.framebufferOnly = YES;

    // Store the layer reference for the parent class
    this->MetalLayer = (__bridge_retained void*)metalLayer;
  }
}

//------------------------------------------------------------------------------
UIView* vtkIOSMetalRenderWindow::GetViewId()
{
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
