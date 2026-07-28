#import "VTKMetalBaseViewController.h"

#include "vtkCamera.h"
#include "vtkMath.h"
#include "vtkNew.h"
#include "vtkProperty.h"
#include "vtkInteractorStyleMultiTouchCamera.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRendererCollection.h"
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

  // Zoom anchor state
  int _zoomAnchorDisplay[2];
  double _zoomAnchorWorld[3];
  BOOL _zoomAnchorValid;

  // Benchmark state
  BOOL _benchmarkRunning;
  NSInteger _benchmarkFrameCount;
  NSInteger _benchmarkTotalFrames;
  CFTimeInterval _benchmarkStartTime;
  CFTimeInterval _benchmarkTotalStartTime;
  double _benchmarkAccumulatedGPUTime;
  double _benchmarkAccumulatedAngle;
}
@property (nonatomic, strong) UIPinchGestureRecognizer* pinchRecognizer;
@property (nonatomic, strong) UIPanGestureRecognizer* panRecognizer;
@property (nonatomic, strong) UIRotationGestureRecognizer* rotationRecognizer;
@end

@implementation VTKMetalBaseViewController

- (instancetype)init
{
  self = [super init];
  if (self)
  {
    _interactionMode = VTKInteractionModePan;
  }
  return self;
}

- (instancetype)initWithCoder:(NSCoder*)coder
{
  self = [super initWithCoder:coder];
  if (self)
  {
    _interactionMode = VTKInteractionModePan;
  }
  return self;
}

- (NSString*)interactionModeTitle
{
  switch (self.interactionMode)
  {
    case VTKInteractionModePan: return @"Pan";
    case VTKInteractionModeZoom: return @"Zoom";
    case VTKInteractionModeTrackball: return @"Trackball";
    case VTKInteractionModeScrollSlices: return @"Scroll Slices";
  }
}

- (NSString*)interactionModeImageName
{
  switch (self.interactionMode)
  {
    case VTKInteractionModePan: return @"hand.point.up";
    case VTKInteractionModeZoom: return @"magnifyingglass";
    case VTKInteractionModeTrackball: return @"cube.transparent";
    case VTKInteractionModeScrollSlices: return @"arrow.up.and.down";
  }
}

- (void)setInteractionMode:(VTKInteractionMode)mode
{
  _interactionMode = mode;
  _zoomAnchorValid = NO;
}

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
  CGFloat scale = recognizer.view.contentScaleFactor;
  _iren->SetEventInformationFlipY((int)(p.x * scale), (int)(p.y * scale), 0, 0, 0, 0, 0);
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
  [self forwardTouchPosition:recognizer];

  switch (self.interactionMode)
  {
    case VTKInteractionModeZoom:
      [self handleZoomPan:recognizer];
      break;
    case VTKInteractionModeTrackball:
      [self handleTrackballPan:recognizer];
      break;
    default:
      [self handlePanPan:recognizer];
      break;
  }

  _renWin->Render();
}

- (void)handlePanPan:(UIPanGestureRecognizer*)recognizer
{
  CGPoint translation = [recognizer translationInView:recognizer.view];
  CGFloat scale = recognizer.view.contentScaleFactor;
  double t[2] = { scale * translation.x, -scale * translation.y };
  _iren->SetTranslation(t);

  switch (recognizer.state)
  {
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

- (void)handleZoomPan:(UIPanGestureRecognizer*)recognizer
{
  vtkRenderer* ren = _renderer;
  if (!ren) return;
  vtkCamera* cam = ren->GetActiveCamera();
  if (!cam) return;

  if (recognizer.state == UIGestureRecognizerStateBegan || !_zoomAnchorValid)
  {
    CGPoint p = [recognizer locationInView:recognizer.view];
    CGFloat scale = recognizer.view.contentScaleFactor;
    _zoomAnchorDisplay[0] = (int)(p.x * scale);
    _zoomAnchorDisplay[1] = (int)(p.y * scale);

    ren->SetDisplayPoint(_zoomAnchorDisplay[0], _zoomAnchorDisplay[1], 0);
    ren->DisplayToWorld();

    double hom[4];
    ren->GetWorldPoint(hom);
    if (hom[3] == 0.0) return;
    for (int i = 0; i < 3; ++i) _zoomAnchorWorld[i] = hom[i] / hom[3];

    _zoomAnchorValid = YES;

    if (recognizer.state == UIGestureRecognizerStateBegan)
    {
      _iren->InvokeEvent(vtkCommand::StartInteractionEvent, nullptr);
    }
    return;
  }

  CGPoint deltaPt = [recognizer translationInView:recognizer.view];
  double factor = 1.0 - deltaPt.y / 200.0;
  factor = vtkMath::ClampValue(factor, 0.01, 100.0);
  [recognizer setTranslation:CGPointZero inView:recognizer.view];

  const BOOL didZoom = (fabs(factor - 1.0) > 1e-6);

  if (didZoom)
  {
    cam->Zoom(factor);

    ren->SetDisplayPoint(_zoomAnchorDisplay[0], _zoomAnchorDisplay[1], 0);
    ren->DisplayToWorld();
    double newHom[4];
    ren->GetWorldPoint(newHom);
    if (newHom[3] == 0.0) return;

    double newWorld[3] = {
      newHom[0] / newHom[3],
      newHom[1] / newHom[3],
      newHom[2] / newHom[3]};

    double d[3] = {
      _zoomAnchorWorld[0] - newWorld[0],
      _zoomAnchorWorld[1] - newWorld[1],
      _zoomAnchorWorld[2] - newWorld[2]};

    double pos[3], fp[3];
    cam->GetPosition(pos);
    cam->GetFocalPoint(fp);
    for (int i = 0; i < 3; ++i)
    {
      pos[i] += d[i];
      fp[i] += d[i];
    }
    cam->SetPosition(pos);
    cam->SetFocalPoint(fp);

    ren->ResetCameraClippingRange();

    ren->SetDisplayPoint(_zoomAnchorDisplay[0], _zoomAnchorDisplay[1], 0);
    ren->DisplayToWorld();
    ren->GetWorldPoint(newHom);
    if (newHom[3] != 0.0)
      for (int i = 0; i < 3; ++i) _zoomAnchorWorld[i] = newHom[i] / newHom[3];
  }

  switch (recognizer.state)
  {
    case UIGestureRecognizerStateChanged:
      if (didZoom)
        _iren->InvokeEvent(vtkCommand::InteractionEvent, nullptr);
      break;
    case UIGestureRecognizerStateEnded:
    case UIGestureRecognizerStateCancelled:
      _iren->InvokeEvent(vtkCommand::EndInteractionEvent, nullptr);
      _zoomAnchorValid = NO;
      break;
    default:
      break;
  }
}

- (void)handleTrackballPan:(UIPanGestureRecognizer*)recognizer
{
  switch (recognizer.state)
  {
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

#pragma mark - Camera

- (void)resetCamera
{
  vtkNew<vtkMetalCamera> camera;
  _renderer->SetActiveCamera(camera);
  _renderer->ResetCamera();
  _renderer->ResetCameraClippingRange();
  _zoomAnchorValid = NO;
  _renWin->Render();
}

#pragma mark - Benchmark

- (BOOL)isBenchmarkRunning
{
  return _benchmarkRunning;
}

- (void)startBenchmark
{
  if (_benchmarkRunning) { return; }

  _benchmarkRunning = YES;
  _benchmarkFrameCount = 0;
  _benchmarkTotalFrames = 0;
  _benchmarkStartTime = CACurrentMediaTime();
  _benchmarkTotalStartTime = _benchmarkStartTime;
  _benchmarkAccumulatedGPUTime = 0.0;
  _benchmarkAccumulatedAngle = 0.0;

  vtkMetalRenderWindow* renWin =
    static_cast<vtkMetalRenderWindow*>(_renWin.GetPointer());

  __weak __typeof(self) weakSelf = self;
  renWin->SetRenderCompletionCallback(^(double gpuTimeMs) {
    __typeof(self) strongSelf = weakSelf;
    if (strongSelf)
    {
      [strongSelf benchmarkFrameWithGPUTime:gpuTimeMs];
    }
  });

  // Kick off the first frame
  [self rotateCameraForNextFrame];
  _renWin->Render();
}

- (void)stopBenchmark
{
  if (!_benchmarkRunning) { return; }

  _benchmarkRunning = NO;
  vtkMetalRenderWindow* renWin =
    static_cast<vtkMetalRenderWindow*>(_renWin.GetPointer());

  renWin->SetRenderCompletionCallback(nil);

  // Log final summary
  CFTimeInterval totalTime = CACurrentMediaTime() - _benchmarkTotalStartTime;
  if (_benchmarkTotalFrames > 0 && totalTime > 0.0)
  {
    double avgFPS = _benchmarkTotalFrames / totalTime;
    NSLog(@"[Benchmark] Result: %.1f FPS | Frames: %ld | Duration: %.2fs | Angle: %.1f deg",
          avgFPS, (long)_benchmarkTotalFrames, totalTime, _benchmarkAccumulatedAngle);
  }
}

- (void)benchmarkFrameWithGPUTime:(double)gpuTimeMs
{
  if (!_benchmarkRunning) { return; }

  _benchmarkFrameCount++;
  _benchmarkTotalFrames++;
  _benchmarkAccumulatedGPUTime += gpuTimeMs;

  CFTimeInterval now = CACurrentMediaTime();
  CFTimeInterval elapsed = now - _benchmarkStartTime;

  if (elapsed >= 1.0)
  {
    double avgFPS = (double)_benchmarkFrameCount / elapsed;
    double avgGPU = _benchmarkAccumulatedGPUTime / (double)_benchmarkFrameCount;
    NSLog(@"[Benchmark] FPS: %.1f | GPU: %.2f ms | Angle: %.1f deg",
          avgFPS, avgGPU, _benchmarkAccumulatedAngle);

    _benchmarkFrameCount = 0;
    _benchmarkStartTime = now;
    _benchmarkAccumulatedGPUTime = 0.0;
  }

  [self rotateCameraForNextFrame];

  // Stop after 1.5 full rotations, with a safety cap at 3000 frames
  BOOL shouldStop = (_benchmarkAccumulatedAngle >= 540.0) || (_benchmarkTotalFrames >= 3000);
  if (shouldStop)
  {
    [self stopBenchmark];
    return;
  }
  _renWin->Render();
}

- (void)rotateCameraForNextFrame
{
  // Rotate the camera ~0.5 degrees per frame around the scene
  vtkCamera* camera = _renderer->GetActiveCamera();
  camera->Azimuth(0.5);
  camera->OrthogonalizeViewUp();
  _renderer->ResetCameraClippingRange();
  _benchmarkAccumulatedAngle += 0.5;
}

@end
