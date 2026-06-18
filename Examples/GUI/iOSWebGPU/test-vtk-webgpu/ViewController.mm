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

@interface ViewController () <UIGestureRecognizerDelegate>
{
  vtkNew<vtkIOSHardwareWindow> _hw;
  vtkNew<vtkWebGPURenderWindow> _renWin;
  vtkNew<vtkWebGPURenderer> _renderer;
  vtkNew<vtkRenderWindowInteractor> _iren;
}
@property (nonatomic, strong) UIPinchGestureRecognizer *pinchRecognizer;
@property (nonatomic, strong) UIPanGestureRecognizer *panRecognizer;
@property (nonatomic, strong) UIRotationGestureRecognizer *rotationRecognizer;
@end

@implementation ViewController

- (void)viewDidLoad {
  [super viewDidLoad];

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

  _renWin->SetHardwareWindow(_hw);
  _hw->SetInteractor(_iren);

  vtkNew<vtkInteractorStyleMultiTouchCamera> style;
  _iren->SetInteractorStyle(style);

  _iren->Initialize();

  _renderer->ResetCamera();
  _renWin->Render();

  UIView *vtkView = _hw->GetViewId();
  if (vtkView) {
    vtkView.frame = self.view.bounds;
    vtkView.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    [self.view addSubview:vtkView];
    [self setupGestureRecognizersOnView:vtkView];
  }
}

- (void)setupGestureRecognizersOnView:(UIView *)view {
  self.pinchRecognizer = [[UIPinchGestureRecognizer alloc] initWithTarget:self
                                                                    action:@selector(handlePinch:)];
  self.pinchRecognizer.delegate = self;
  [view addGestureRecognizer:self.pinchRecognizer];

  self.panRecognizer = [[UIPanGestureRecognizer alloc] initWithTarget:self
                                                                action:@selector(handlePan:)];
  self.panRecognizer.delegate = self;
  self.panRecognizer.maximumNumberOfTouches = 1;
  [view addGestureRecognizer:self.panRecognizer];

  self.rotationRecognizer = [[UIRotationGestureRecognizer alloc] initWithTarget:self
                                                                          action:@selector(handleRotation:)];
  self.rotationRecognizer.delegate = self;
  [view addGestureRecognizer:self.rotationRecognizer];
}

- (void)forwardTouchPosition:(UIGestureRecognizer *)recognizer {
  CGPoint p = [recognizer locationInView:recognizer.view];
  CGFloat height = recognizer.view.bounds.size.height;
  _iren->SetEventInformation((int)p.x, (int)(height - p.y), 0, 0, 0, 0, 0);
}

- (void)handlePinch:(UIPinchGestureRecognizer *)recognizer {
  [self forwardTouchPosition:recognizer];

  CGFloat scale = recognizer.scale;
  scale = MAX(-3.0, MIN(3.0, scale));
  _iren->SetScale(scale);

  switch (recognizer.state) {
    case UIGestureRecognizerStateBegan:
      _iren->StartPinchEvent();
      break;
    case UIGestureRecognizerStateChanged:
      _iren->PinchEvent();
      break;
    case UIGestureRecognizerStateEnded:
    case UIGestureRecognizerStateCancelled:
      _iren->EndPinchEvent();
      break;
    default:
      break;
  }

  _renWin->Render();
}

- (void)handlePan:(UIPanGestureRecognizer *)recognizer {
  [self forwardTouchPosition:recognizer];

  CGPoint translation = [recognizer translationInView:recognizer.view];
  CGFloat scale = self.view.contentScaleFactor;
  double t[2] = {scale * translation.x, -scale * translation.y};
  _iren->SetTranslation(t);

  switch (recognizer.state) {
    case UIGestureRecognizerStateBegan:
      _iren->StartPanEvent();
      break;
    case UIGestureRecognizerStateChanged:
      _iren->PanEvent();
      break;
    case UIGestureRecognizerStateEnded:
    case UIGestureRecognizerStateCancelled:
      _iren->EndPanEvent();
      break;
    default:
      break;
  }

  _renWin->Render();
}

- (void)handleRotation:(UIRotationGestureRecognizer *)recognizer {
  [self forwardTouchPosition:recognizer];

  double angle = -[recognizer rotation] * 180.0 / M_PI;
  _iren->SetRotation(angle);

  switch (recognizer.state) {
    case UIGestureRecognizerStateBegan:
      _iren->StartRotateEvent();
      break;
    case UIGestureRecognizerStateChanged:
      _iren->RotateEvent();
      break;
    case UIGestureRecognizerStateEnded:
    case UIGestureRecognizerStateCancelled:
      _iren->EndRotateEvent();
      break;
    default:
      break;
  }

  _renWin->Render();
}

- (BOOL)gestureRecognizer:(UIGestureRecognizer *)a
    shouldRecognizeSimultaneouslyWithGestureRecognizer:(UIGestureRecognizer *)b {
  BOOL pinchOrRotation =
      ([a isKindOfClass:[UIPinchGestureRecognizer class]] ||
       [a isKindOfClass:[UIRotationGestureRecognizer class]]);
  BOOL otherIsSame =
      ([b isKindOfClass:[UIPinchGestureRecognizer class]] ||
       [b isKindOfClass:[UIRotationGestureRecognizer class]]);
  return pinchOrRotation && otherIsSame;
}

- (void)viewDidLayoutSubviews {
  [super viewDidLayoutSubviews];
  _renWin->SetSize(self.view.bounds.size.width, self.view.bounds.size.height);
  _iren->UpdateSize(self.view.bounds.size.width, self.view.bounds.size.height);
}

@end
