// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkCocoaMetalRenderWindow.h"

#include "vtkObjectFactory.h"

#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>
#import <objc/runtime.h>

static const void* const kVtkCocoaMetalViewEventDelegateKey =
  &kVtkCocoaMetalViewEventDelegateKey;

@interface vtkCocoaMetalView : NSView
@end

@implementation vtkCocoaMetalView

- (instancetype)initWithFrame:(NSRect)frameRect
{
  self = [super initWithFrame:frameRect];
  if (self)
  {
    [self setWantsLayer:YES];
    CAMetalLayer* layer = [CAMetalLayer layer];
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = YES;
    layer.opaque = NO;
    [self setLayer:layer];
    self.allowedTouchTypes = NSTouchTypeMaskDirect;
  }
  return self;
}

- (void)forwardEvent:(NSEvent*)event toSelector:(SEL)selector
{
  id<VTKCocoaMetalViewDelegate> delegate =
    objc_getAssociatedObject(self, kVtkCocoaMetalViewEventDelegateKey);
  if (delegate && [delegate respondsToSelector:selector])
  {
    [delegate performSelector:selector withObject:self withObject:event];
  }
}

- (void)mouseDown:(NSEvent*)event
{
  [self forwardEvent:event toSelector:@selector(metalView:mouseDown:)];
}

- (void)mouseDragged:(NSEvent*)event
{
  [self forwardEvent:event toSelector:@selector(metalView:mouseDragged:)];
}

- (void)mouseUp:(NSEvent*)event
{
  [self forwardEvent:event toSelector:@selector(metalView:mouseUp:)];
}

- (void)scrollWheel:(NSEvent*)event
{
  [self forwardEvent:event toSelector:@selector(metalView:scrollWheel:)];
}

- (void)magnifyWithEvent:(NSEvent*)event
{
  [self forwardEvent:event toSelector:@selector(metalView:magnifyWithEvent:)];
}

- (void)rotateWithEvent:(NSEvent*)event
{
  [self forwardEvent:event toSelector:@selector(metalView:rotateWithEvent:)];
}

- (BOOL)acceptsFirstResponder
{
  return YES;
}

@end

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkCocoaMetalRenderWindow);

//------------------------------------------------------------------------------
vtkCocoaMetalRenderWindow::vtkCocoaMetalRenderWindow() = default;

//------------------------------------------------------------------------------
vtkCocoaMetalRenderWindow::~vtkCocoaMetalRenderWindow()
{
  if (this->ViewId)
  {
    [this->ViewId removeFromSuperview];
    [this->ViewId release];
    this->ViewId = nullptr;
    this->MetalLayer = nullptr;
  }
}

//------------------------------------------------------------------------------
void vtkCocoaMetalRenderWindow::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "ViewId: " << this->ViewId << "\n";
}

//------------------------------------------------------------------------------
NSView* vtkCocoaMetalRenderWindow::GetViewId()
{
  // Create the view on first access
  if (!this->ViewId)
  {
    CGFloat scale = [[NSScreen mainScreen] backingScaleFactor];
    if (scale <= 0.0)
    {
      scale = 1.0;
    }
    CGFloat w = (this->Size[0] > 0) ? this->Size[0] / scale : 300;
    CGFloat h = (this->Size[1] > 0) ? this->Size[1] / scale : 300;
    NSRect frame = NSMakeRect(0, 0, w, h);
    vtkCocoaMetalView* view = [[vtkCocoaMetalView alloc] initWithFrame:frame];
    this->ViewId = view;

    CAMetalLayer* metalLayer = (CAMetalLayer*)[view layer];
    metalLayer.device = (id<MTLDevice>)this->MetalDevice;
    metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    metalLayer.contentsScale = scale;
    metalLayer.framebufferOnly = YES;
    metalLayer.drawableSize = CGSizeMake(this->Size[0], this->Size[1]);

    // Drop the standalone layer created by vtkMetalRenderWindow::CreateMetalLayer
    // in favor of the view's backing layer (owned by the view).
    if (this->MetalLayer && this->MetalLayer != (void*)metalLayer)
    {
      [(id)this->MetalLayer release];
    }
    this->MetalLayer = (void*)metalLayer;
  }
  return this->ViewId;
}

//------------------------------------------------------------------------------
void vtkCocoaMetalRenderWindow::SetSize(int width, int height)
{
  this->Superclass::SetSize(width, height);

  if (this->ViewId)
  {
    CGFloat scale = [[NSScreen mainScreen] backingScaleFactor];
    if (scale <= 0.0)
    {
      scale = 1.0;
    }
    NSRect frame = NSMakeRect(0, 0, width / scale, height / scale);
    [this->ViewId setFrame:frame];

    CAMetalLayer* layer = (CAMetalLayer*)[this->ViewId layer];
    layer.drawableSize = CGSizeMake(width, height);
  }
}

//------------------------------------------------------------------------------
void vtkCocoaMetalRenderWindow::SetViewEventDelegate(id<VTKCocoaMetalViewDelegate> delegate)
{
  if (this->ViewId)
  {
    objc_setAssociatedObject(this->ViewId, kVtkCocoaMetalViewEventDelegateKey, delegate,
      OBJC_ASSOCIATION_ASSIGN);
  }
}

VTK_ABI_NAMESPACE_END
