// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
#ifndef vtkCocoaMetalRenderWindow_h
#define vtkCocoaMetalRenderWindow_h

#include "vtkMetalRenderWindow.h"
#include "vtkRenderingMetalModule.h"
#include "vtkWrappingHints.h"

#ifdef __OBJC__
#import <Foundation/Foundation.h>

@class NSView;
@class NSEvent;

/** Optional delegate that receives mouse/scroll/trackpad events from the
 *  NSView created by vtkCocoaMetalRenderWindow.  The view itself does not
 *  interpret the events; it simply forwards them to the delegate so the
 *  owning controller can drive the VTK interactor.
 */
@protocol VTKCocoaMetalViewDelegate <NSObject>
@optional
- (void)metalView:(NSView*)view mouseDown:(NSEvent*)event;
- (void)metalView:(NSView*)view mouseDragged:(NSEvent*)event;
- (void)metalView:(NSView*)view mouseUp:(NSEvent*)event;
- (void)metalView:(NSView*)view scrollWheel:(NSEvent*)event;
- (void)metalView:(NSView*)view magnifyWithEvent:(NSEvent*)event;
- (void)metalView:(NSView*)view rotateWithEvent:(NSEvent*)event;
@end
#else
class NSView;
#endif

VTK_ABI_NAMESPACE_BEGIN
class vtkOverrideAttribute;

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkCocoaMetalRenderWindow
  : public vtkMetalRenderWindow
{
public:
  static vtkCocoaMetalRenderWindow* New();
  VTK_NEWINSTANCE
  vtkTypeMacro(vtkCocoaMetalRenderWindow, vtkMetalRenderWindow);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  NSView* GetViewId();

  void SetSize(int width, int height) override;

#ifdef __OBJC__
  /**
   * Set the delegate that receives mouse/scroll/trackpad events from the
   * backing NSView.  The delegate reference is held weakly.
   */
  void SetViewEventDelegate(id<VTKCocoaMetalViewDelegate> delegate);
#endif

protected:
  vtkCocoaMetalRenderWindow();
  ~vtkCocoaMetalRenderWindow() override;

private:
  vtkCocoaMetalRenderWindow(const vtkCocoaMetalRenderWindow&) = delete;
  void operator=(const vtkCocoaMetalRenderWindow&) = delete;

  NSView* ViewId = nullptr;
};

#define vtkCocoaMetalRenderWindow_OVERRIDE_ATTRIBUTES \
  vtkCocoaMetalRenderWindow::CreateOverrideAttributes()

VTK_ABI_NAMESPACE_END
#endif
