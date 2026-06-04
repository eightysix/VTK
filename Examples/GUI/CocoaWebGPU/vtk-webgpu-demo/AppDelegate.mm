#import "AppDelegate.h"

#include "vtkCocoaHardwareWindow.h"
#include "vtkNew.h"
#include "vtkProperty.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkSphereSource.h"
#include "vtkWebGPUActor.h"
#include "vtkWebGPUCamera.h"
#include "vtkWebGPUPolyDataMapper.h"
#include "vtkWebGPURenderer.h"
#include "vtkWebGPURenderWindow.h"

@interface AppDelegate ()
@end

@implementation AppDelegate
{
  vtkNew<vtkCocoaHardwareWindow> _hw;
  vtkNew<vtkWebGPURenderWindow> _renWin;
  vtkNew<vtkWebGPURenderer> _renderer;
  vtkNew<vtkRenderWindowInteractor> _iren;
}

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    NSMenu *menubar = [NSMenu new];
    NSMenuItem *appMenuItem = [[NSMenuItem alloc] initWithTitle:@"vtk-webgpu-demo" action:nil keyEquivalent:@""];
    [menubar addItem:appMenuItem];
    [NSApp setMainMenu:menubar];

    NSMenu *appMenu = [[NSMenu alloc] initWithTitle:@"vtk-webgpu-demo"];
    [appMenu addItemWithTitle:@"Quit vtk-webgpu-demo" action:@selector(terminate:) keyEquivalent:@"q"];
    [appMenuItem setSubmenu:appMenu];

    // Setup VTK scene
    vtkNew<vtkSphereSource> sphere;
    vtkNew<vtkWebGPUPolyDataMapper> mapper;
    mapper->SetInputConnection(sphere->GetOutputPort());

    vtkNew<vtkWebGPUActor> actor;
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(0.2, 0.6, 1.0);
    _renderer->AddActor(actor);

    vtkNew<vtkWebGPUCamera> camera;
    _renderer->SetActiveCamera(camera);

    _renderer->SetBackground(0.1, 0.1, 0.2);
    _renWin->AddRenderer(_renderer);
    _renWin->SetSize(800, 600);
    _renWin->SetWindowName("vtk-webgpu-demo");
    _iren->SetRenderWindow(_renWin);

    // Wire up the hardware window before Render()
    _renWin->SetHardwareWindow(_hw);
    _hw->SetInteractor(_iren);

    // Render triggers hardware window Create() -> NSWindow + vtkCocoaHardwareView
    _renderer->ResetCamera();
    _renWin->Render();

    // Adopt the NSWindow created by vtkCocoaHardwareWindow
    self.window = _hw->GetWindowId();

    [NSApp activateIgnoringOtherApps:YES];
}

- (void)applicationWillTerminate:(NSNotification *)aNotification {
}

- (BOOL)applicationSupportsSecureRestorableState:(NSApplication *)app {
    return YES;
}

@end
