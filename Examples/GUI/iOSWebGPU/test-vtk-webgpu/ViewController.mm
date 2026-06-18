#import "ViewController.h"

#include "vtkCallbackCommand.h"
#include "vtkIOSHardwareWindow.h"
#include "vtkNew.h"
#include "vtkProperty.h"
#include "vtkInteractorStyleMultiTouchCamera.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkCommand.h"
#include "vtkConeSource.h"
#include "vtkWebGPUActor.h"
#include "vtkWebGPUCamera.h"
#include "vtkWebGPUPolyDataMapper.h"
#include "vtkWebGPURenderer.h"
#include "vtkWebGPURenderWindow.h"
#include <string>

@interface ViewController () <UIGestureRecognizerDelegate>
{
  vtkNew<vtkIOSHardwareWindow> _hw;
  vtkNew<vtkWebGPURenderWindow> _renWin;
  vtkNew<vtkWebGPURenderer> _renderer;
  vtkNew<vtkRenderWindowInteractor> _iren;
  BOOL _autoRotate;
  NSTimer *_rotationTimer;
  vtkNew<vtkCallbackCommand> _keyPressCallback;
  vtkNew<vtkWebGPUActor> _coneActor;
}
@property (nonatomic, strong) UIPinchGestureRecognizer *pinchRecognizer;
@property (nonatomic, strong) UIPanGestureRecognizer *panRecognizer;
@property (nonatomic, strong) UIRotationGestureRecognizer *rotationRecognizer;
@property (nonatomic, strong) UIButton *toggleButton;
@end

@implementation ViewController

static void OnKeyPress(vtkObject *object, unsigned long, void *clientdata, void *) {
  vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::SafeDownCast(object);
  if (!iren) return;
  ViewController *vc = (__bridge ViewController *)clientdata;
  if (!vc) return;

  std::string key = iren->GetKeySym();
  if (key == "space") {
    dispatch_async(dispatch_get_main_queue(), ^{
      [vc toggleAutoRotation];
    });
  }
}

- (void)viewDidLoad {
  [super viewDidLoad];

  vtkNew<vtkConeSource> cone;
  cone->SetResolution(128);
  vtkNew<vtkWebGPUPolyDataMapper> mapper;
  mapper->SetInputConnection(cone->GetOutputPort());

  _coneActor->SetMapper(mapper);
  _coneActor->GetProperty()->SetColor(0.2, 0.6, 1.0);
  _renderer->AddActor(_coneActor);

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

  _keyPressCallback->SetCallback(OnKeyPress);
  _keyPressCallback->SetClientData((__bridge void *)self);
  _iren->AddObserver(vtkCommand::KeyPressEvent, _keyPressCallback);

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

  _toggleButton = [UIButton buttonWithType:UIButtonTypeSystem];
  [_toggleButton setTitle:@"Play" forState:UIControlStateNormal];
  _toggleButton.titleLabel.font = [UIFont boldSystemFontOfSize:17];
  _toggleButton.backgroundColor = [UIColor whiteColor];
  _toggleButton.layer.cornerRadius = 22;
  [_toggleButton setTitleColor:[UIColor blackColor] forState:UIControlStateNormal];
  [_toggleButton addTarget:self action:@selector(toggleAutoRotation) forControlEvents:UIControlEventTouchUpInside];
  _toggleButton.translatesAutoresizingMaskIntoConstraints = NO;
  [self.view addSubview:_toggleButton];

  [NSLayoutConstraint activateConstraints:@[
    [_toggleButton.trailingAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.trailingAnchor constant:-20],
    [_toggleButton.bottomAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.bottomAnchor constant:-20],
    [_toggleButton.widthAnchor constraintEqualToConstant:80],
    [_toggleButton.heightAnchor constraintEqualToConstant:44]
  ]];
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
  BOOL trackball = recognizer.modifierFlags & UIKeyModifierShift;

  [self forwardTouchPosition:recognizer];

  if (trackball) {
    switch (recognizer.state) {
      case UIGestureRecognizerStateBegan:
        _iren->InvokeEvent(vtkCommand::LeftButtonPressEvent, nullptr);
        break;
      case UIGestureRecognizerStateChanged:
        _iren->InvokeEvent(vtkCommand::MouseMoveEvent, nullptr);
        break;
      case UIGestureRecognizerStateEnded:
      case UIGestureRecognizerStateCancelled:
        _iren->InvokeEvent(vtkCommand::LeftButtonReleaseEvent, nullptr);
        break;
      default:
        break;
    }
  } else {
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

- (void)toggleAutoRotation {
  _autoRotate = !_autoRotate;
  [_toggleButton setTitle:(_autoRotate ? @"Pause" : @"Play") forState:UIControlStateNormal];

  if (_autoRotate) {
    _rotationTimer = [NSTimer scheduledTimerWithTimeInterval:1.0 / 60.0
                                                     target:self
                                                   selector:@selector(rotateStep)
                                                   userInfo:nil
                                                    repeats:YES];
  } else {
    [_rotationTimer invalidate];
    _rotationTimer = nil;
  }
}

- (void)rotateStep {
  if (_autoRotate) {
    _coneActor->RotateY(1.0);
    _renWin->Render();
  }
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
