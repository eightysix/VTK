#import "ViewController.h"

#include "vtkCallbackCommand.h"
#include "vtkIOSHardwareWindow.h"
#include "vtkNew.h"
#include "vtkProperty.h"
#include "vtkInteractorStyleMultiTouchCamera.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkConeSource.h"
#include "vtkWebGPUActor.h"
#include "vtkWebGPUCamera.h"
#include "vtkWebGPUPolyDataMapper.h"
#include "vtkWebGPURenderer.h"
#include "vtkWebGPURenderWindow.h"

@interface ViewController ()
{
  vtkNew<vtkIOSHardwareWindow> _hw;
  vtkNew<vtkWebGPURenderWindow> _renWin;
  vtkNew<vtkWebGPURenderer> _renderer;
  vtkNew<vtkRenderWindowInteractor> _iren;
}
@end

@implementation ViewController

- (void)viewDidLoad {
  [super viewDidLoad];

  // Setup VTK scene
  vtkNew<vtkConeSource> cone;
  cone->SetResolution(128);
  vtkNew<vtkWebGPUPolyDataMapper> mapper;
  mapper->SetInputConnection(cone->GetOutputPort());

  vtkNew<vtkWebGPUActor> actor;
  actor->SetMapper(mapper);
  actor->GetProperty()->SetColor(0.2, 0.6, 1.0);
  _renderer->AddActor(actor);

  vtkNew<vtkWebGPUCamera> camera;
  _renderer->SetActiveCamera(camera);

  _renderer->SetBackground(0.1, 0.1, 0.2);
  _renWin->AddRenderer(_renderer);
  _renWin->SetSize(self.view.bounds.size.width, self.view.bounds.size.height);
  _renWin->SetWindowName("test-vtk-webgpu");
  _iren->SetRenderWindow(_renWin);

  // Wire up the hardware window before Render()
  _renWin->SetHardwareWindow(_hw);
  _hw->SetInteractor(_iren);

  // Set multitouch interactor style
  vtkNew<vtkInteractorStyleMultiTouchCamera> style;
  _iren->SetInteractorStyle(style);

  // Initialize the interactor
  _iren->Initialize();

  // Render triggers hardware window Create()
  _renderer->ResetCamera();
  _renWin->Render();

  // Embed the VTK view into our view hierarchy
  UIView* vtkView = _hw->GetViewId();
  if (vtkView) {
    vtkView.frame = self.view.bounds;
    vtkView.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    [self.view addSubview:vtkView];
  }
}

- (void)viewDidLayoutSubviews {
  [super viewDidLayoutSubviews];
  _renWin->SetSize(self.view.bounds.size.width, self.view.bounds.size.height);
  _iren->UpdateSize(self.view.bounds.size.width, self.view.bounds.size.height);
}

@end
