#import "VTKMetalBaseViewController.h"

#include "vtkNew.h"
#include "vtkProperty.h"
#include "vtkInteractorStyleMultiTouchCamera.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkCommand.h"
#include "vtkMetalActor.h"
#include "vtkMetalCamera.h"
#include "vtkMetalPolyDataMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkMetalRenderWindow.h"
#include "vtkIOSMetalRenderWindow.h"

@interface VTKMetalBaseViewController ()
{
  vtkNew<vtkIOSMetalRenderWindow> _renWin;
  vtkNew<vtkMetalRenderer> _renderer;
  vtkNew<vtkRenderWindowInteractor> _iren;
  BOOL _trackballMode;
}
@property (nonatomic, strong) UIPinchGestureRecognizer* pinchRecognizer;
@property (nonatomic, strong) UIPanGestureRecognizer* panRecognizer;
@property (nonatomic, strong) UIRotationGestureRecognizer* rotationRecognizer;
@end

@implementation VTKMetalBaseViewController

- (void*)renderer
{
  return _renderer;
}

- (void*)renderWindow
{
  return _renWin;
}

- (void)viewDidLoad
{
  [super viewDidLoad];

  // Initialize Metal render window
  _renWin->Initialize();

  // Let subclass set up the VTK pipeline
  [self setupVTKPipeline];

  // Set up camera
  vtkNew<vtkMetalCamera> camera;
  _renderer->SetActiveCamera(camera);

  _renderer->SetBackground(0.1, 0.1, 0.2);
  _renWin->AddRenderer(_renderer);

  // Set size (Retina-aware)
  CGFloat scale = [UIScreen mainScreen].nativeScale;
  _renWin->SetSize((int)lround(scale * self.view.bounds.size.width),
                   (int)lround(scale * self.view.bounds.size.height));

  // Set up interactor
  _iren->SetRenderWindow(_renWin);
  _iren->SetEnableRender(false);

  vtkNew<vtkInteractorStyleMultiTouchCamera> style;
  _iren->SetInteractorStyle(style);
  _iren->Initialize();

  _renderer->ResetCamera();
  _renWin->Render();

  // Add the Metal view as subview
  UIView* vtkView = _renWin->GetViewId();
  if (vtkView)
  {
    vtkView.frame = self.view.bounds;
    vtkView.autoresizingMask =
      UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    [self.view addSubview:vtkView];
    [self setupGestureRecognizersOnView:vtkView];
  }
}

- (void)setupVTKPipeline
{
  // Subclasses must override this method
}

- (void)setupGestureRecognizersOnView:(UIView*)view
{
  self.pinchRecognizer =
    [[UIPinchGestureRecognizer alloc] initWithTarget:self action:@selector(handlePinch:)];
  self.pinchRecognizer.delegate = self;
  [view addGestureRecognizer:self.pinchRecognizer];

  self.panRecognizer =
    [[UIPanGestureRecognizer alloc] initWithTarget:self action:@selector(handlePan:)];
  self.panRecognizer.delegate = self;
  self.panRecognizer.maximumNumberOfTouches = 1;
  [view addGestureRecognizer:self.panRecognizer];

  self.rotationRecognizer =
    [[UIRotationGestureRecognizer alloc] initWithTarget:self
                                                  action:@selector(handleRotation:)];
  self.rotationRecognizer.delegate = self;
  [view addGestureRecognizer:self.rotationRecognizer];
}

- (void)forwardTouchPosition:(UIGestureRecognizer*)recognizer
{
  CGPoint p = [recognizer locationInView:recognizer.view];
  CGFloat height = recognizer.view.bounds.size.height;
  _iren->SetEventInformation((int)p.x, (int)(height - p.y), 0, 0, 0, 0, 0);
}

- (void)handlePinch:(UIPinchGestureRecognizer*)recognizer
{
  [self forwardTouchPosition:recognizer];

  CGFloat scale = recognizer.scale;
  scale = MAX(-3.0, MIN(3.0, scale));
  _iren->SetScale(scale);

  switch (recognizer.state)
  {
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

- (void)handlePan:(UIPanGestureRecognizer*)recognizer
{
  BOOL trackball = (recognizer.modifierFlags & UIKeyModifierShift) == 0;

  [self forwardTouchPosition:recognizer];

  switch (recognizer.state)
  {
    case UIGestureRecognizerStateBegan:
      _trackballMode = trackball;
      if (trackball)
      {
        _iren->InvokeEvent(vtkCommand::LeftButtonPressEvent, nullptr);
      }
      else
      {
        CGPoint translation = [recognizer translationInView:recognizer.view];
        CGFloat scale = self.view.contentScaleFactor;
        double t[2] = { scale * translation.x, -scale * translation.y };
        _iren->SetTranslation(t);
        _iren->StartPanEvent();
      }
      break;

    case UIGestureRecognizerStateChanged:
      if (trackball != _trackballMode)
      {
        if (_trackballMode)
        {
          _iren->InvokeEvent(vtkCommand::LeftButtonReleaseEvent, nullptr);
          CGPoint translation = [recognizer translationInView:recognizer.view];
          CGFloat scale = self.view.contentScaleFactor;
          double t[2] = { scale * translation.x, -scale * translation.y };
          _iren->SetTranslation(t);
          _iren->StartPanEvent();
        }
        else
        {
          _iren->EndPanEvent();
          _iren->InvokeEvent(vtkCommand::LeftButtonPressEvent, nullptr);
        }
        _trackballMode = trackball;
      }
      else if (trackball)
      {
        _iren->InvokeEvent(vtkCommand::MouseMoveEvent, nullptr);
      }
      else
      {
        CGPoint translation = [recognizer translationInView:recognizer.view];
        CGFloat scale = self.view.contentScaleFactor;
        double t[2] = { scale * translation.x, -scale * translation.y };
        _iren->SetTranslation(t);
        _iren->PanEvent();
      }
      break;

    case UIGestureRecognizerStateEnded:
    case UIGestureRecognizerStateCancelled:
      if (_trackballMode)
      {
        _iren->InvokeEvent(vtkCommand::LeftButtonReleaseEvent, nullptr);
      }
      else
      {
        _iren->EndPanEvent();
      }
      _trackballMode = NO;
      break;

    default:
      break;
  }

  _renWin->Render();
}

- (void)handleRotation:(UIRotationGestureRecognizer*)recognizer
{
  [self forwardTouchPosition:recognizer];

  double angle = -[recognizer rotation] * 180.0 / M_PI;
  _iren->SetRotation(angle);

  switch (recognizer.state)
  {
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

- (BOOL)gestureRecognizer:(UIGestureRecognizer*)a
    shouldRecognizeSimultaneouslyWithGestureRecognizer:(UIGestureRecognizer*)b
{
  BOOL pinchOrRotation =
    ([a isKindOfClass:[UIPinchGestureRecognizer class]] ||
     [a isKindOfClass:[UIRotationGestureRecognizer class]]);
  BOOL otherIsSame =
    ([b isKindOfClass:[UIPinchGestureRecognizer class]] ||
     [b isKindOfClass:[UIRotationGestureRecognizer class]]);
  return pinchOrRotation && otherIsSame;
}

- (void)viewDidLayoutSubviews
{
  [super viewDidLayoutSubviews];
  CGFloat scale = [UIScreen mainScreen].nativeScale;
  int w = (int)lround(scale * self.view.bounds.size.width);
  int h = (int)lround(scale * self.view.bounds.size.height);
  _renWin->SetSize(w, h);
  _iren->UpdateSize(w, h);
  _renWin->Render();
}

@end
