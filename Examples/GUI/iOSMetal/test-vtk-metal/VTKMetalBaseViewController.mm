#import "VTKMetalBaseViewController.h"

#include "vtkAutoInit.h"
VTK_MODULE_INIT(vtkRenderingMetal);

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
#if TARGET_OS_OSX
#include "vtkCocoaMetalRenderWindow.h"
#else
#include "vtkIOSMetalRenderWindow.h"
#endif

#import <QuartzCore/QuartzCore.h>

@interface VTKMetalBaseViewController ()
{
#if TARGET_OS_OSX
  vtkNew<vtkCocoaMetalRenderWindow> _renWin;
#else
  vtkNew<vtkIOSMetalRenderWindow> _renWin;
#endif
  vtkNew<vtkMetalRenderer> _renderer;
  vtkNew<vtkRenderWindowInteractor> _iren;

  // Zoom anchor state
  int _zoomAnchorDisplay[2];
  double _zoomAnchorWorld[3];
  BOOL _zoomAnchorValid;
  double _zoomLastTranslationY;

  // Benchmark state
  BOOL _benchmarkRunning;
  NSInteger _benchmarkFrameCount;
  NSInteger _benchmarkTotalFrames;
  CFTimeInterval _benchmarkStartTime;
  CFTimeInterval _benchmarkTotalStartTime;
  double _benchmarkAccumulatedGPUTime;
  double _benchmarkAccumulatedAngle;

#if TARGET_OS_OSX
  // Mouse drag state
  BOOL _dragActive;
  CGPoint _dragStartPoint;
  double _macPinchScale;
  BOOL _magnifyActive;
  double _macRotationAngle;
  double _scrollAccumulator;
#endif

  // Gesture interaction state shared by both platforms.  macOS magnify/rotate
  // events and iOS pinch/rotation recognizers drive the same handlers.
  BOOL _pinchActive;
  BOOL _rotateActive;
  CGFloat _lastPinchScale;
  CGFloat _lastRotationAngle;
  NSInteger _lastPanTouchCount;
}
#if !TARGET_OS_OSX
@property (nonatomic, strong) UIPinchGestureRecognizer* pinchRecognizer;
@property (nonatomic, strong) UIPanGestureRecognizer* panRecognizer;
@property (nonatomic, strong) UIRotationGestureRecognizer* rotationRecognizer;
#endif
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
    case VTKInteractionModeWindowLevel: return @"Window/Level";
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
    case VTKInteractionModeWindowLevel: return @"sun.max";
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
  _renWin->SetSize((int)lround([self contentScaleFactor] * self.view.bounds.size.width),
                   (int)lround([self contentScaleFactor] * self.view.bounds.size.height));

  // Set up interactor
  _iren->SetRenderWindow(_renWin);
  _iren->SetEnableRender(false);

  vtkNew<vtkInteractorStyleMultiTouchCamera> style;
  _iren->SetInteractorStyle(style);
  _iren->Initialize();

  _renderer->ResetCamera();
  _renWin->Render();

  // Add the Metal view as subview
#if TARGET_OS_OSX
  NSView* vtkView = _renWin->GetViewId();
  if (vtkView)
  {
    vtkView.frame = self.view.bounds;
    vtkView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    [self.view addSubview:vtkView];
    _renWin->SetViewEventDelegate(self);
  }
#else
  UIView* vtkView = _renWin->GetViewId();
  if (vtkView)
  {
    vtkView.frame = self.view.bounds;
    vtkView.autoresizingMask =
      UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    [self.view addSubview:vtkView];
    [self setupGestureRecognizersOnView:vtkView];
  }
#endif
}

- (void)setupVTKPipeline
{
  // Subclasses must override this method
}

#pragma mark - Scale factor

- (CGFloat)contentScaleFactor
{
#if TARGET_OS_OSX
  return self.view.window ? self.view.window.backingScaleFactor
                          : [NSScreen mainScreen].backingScaleFactor;
#else
  // vtkIOSMetalRenderWindow uses the screen's native scale to convert between
  // point-based frames and pixel drawable sizes, so we must stay in sync with it.
  // (On Mac Catalyst / "Designed for iPad" the view's contentScaleFactor may
  // report 1.0 while the backing store is 2x, which would shrink the view.)
  return [UIScreen mainScreen].nativeScale;
#endif
}

#pragma mark - iOS Gesture Recognizers

#if !TARGET_OS_OSX

- (void)setupGestureRecognizersOnView:(UIView*)view
{
  self.pinchRecognizer =
    [[UIPinchGestureRecognizer alloc] initWithTarget:self action:@selector(handlePinch:)];
  self.pinchRecognizer.delegate = self;
  [view addGestureRecognizer:self.pinchRecognizer];

  self.panRecognizer =
    [[UIPanGestureRecognizer alloc] initWithTarget:self action:@selector(handlePan:)];
  self.panRecognizer.delegate = self;
  self.panRecognizer.minimumNumberOfTouches = 1;
  self.panRecognizer.maximumNumberOfTouches = 2;
  [view addGestureRecognizer:self.panRecognizer];

  self.rotationRecognizer =
    [[UIRotationGestureRecognizer alloc] initWithTarget:self
                                                  action:@selector(handleRotation:)];
  self.rotationRecognizer.delegate = self;
  [view addGestureRecognizer:self.rotationRecognizer];

  _lastPanTouchCount = 0;
  _pinchActive = NO;
  _rotateActive = NO;
}

- (void)forwardTouchPosition:(UIGestureRecognizer*)recognizer
{
  CGPoint p = [recognizer locationInView:recognizer.view];
  CGFloat scale = recognizer.view.contentScaleFactor;
  _iren->SetEventInformationFlipY((int)(p.x * scale), (int)(p.y * scale), 0, 0, 0, 0, 0);
}

- (VTKGestureState)gestureStateFromRecognizer:(UIGestureRecognizer*)recognizer
{
  switch (recognizer.state)
  {
    case UIGestureRecognizerStateBegan: return VTKGestureStateBegan;
    case UIGestureRecognizerStateChanged: return VTKGestureStateChanged;
    case UIGestureRecognizerStateEnded: return VTKGestureStateEnded;
    case UIGestureRecognizerStateCancelled: return VTKGestureStateCancelled;
    case UIGestureRecognizerStateFailed:
    case UIGestureRecognizerStatePossible:
    default: return VTKGestureStateNone;
  }
}

- (void)handlePinch:(UIPinchGestureRecognizer*)recognizer
{
  [self forwardTouchPosition:recognizer];
  [self handlePinchScale:recognizer.scale
                   state:[self gestureStateFromRecognizer:recognizer]];
}

- (void)handlePan:(UIPanGestureRecognizer*)recognizer
{
  [self forwardTouchPosition:recognizer];

  CGPoint p = [recognizer locationInView:recognizer.view];
  CGPoint t = [recognizer translationInView:recognizer.view];
  NSInteger touches = recognizer.numberOfTouches;
  VTKGestureState state = [self gestureStateFromRecognizer:recognizer];
  if (state == VTKGestureStateNone)
  {
    return;
  }

  // The pan recognizer accepts 1-2 fingers.  When the finger count changes we
  // bridge between the one-finger interaction (trackball/zoom/pan, driven via
  // handleDragAtViewPoint:) and the two-finger gestures (pinch/rotate, driven
  // by their own recognizers).  Without this the pan would fail when a second
  // finger lands and the one-finger interaction would stay dead until every
  // finger was lifted.
  if (state == VTKGestureStateEnded || state == VTKGestureStateCancelled)
  {
    if (_lastPanTouchCount == 2)
    {
      // Two fingers lifted: close out any in-flight pinch/rotate interaction.
      if (_rotateActive)
      {
        [self handleRotationAngle:_lastRotationAngle state:VTKGestureStateEnded];
      }
      if (_pinchActive)
      {
        [self handlePinchScale:_lastPinchScale state:VTKGestureStateEnded];
      }
    }
    else if (_lastPanTouchCount == 1)
    {
      [self handleDragAtViewPoint:p translation:t state:state];
    }
    _lastPanTouchCount = 0;
    return;
  }

  if (touches == _lastPanTouchCount)
  {
    // Finger count unchanged: only a single finger drives the one-finger
    // interaction.  Two fingers are handled by the pinch/rotation recognizers.
    if (touches == 1)
    {
      [self handleDragAtViewPoint:p translation:t state:state];
    }
    return;
  }

  NSInteger oldCount = _lastPanTouchCount;
  _lastPanTouchCount = touches;

  if (touches == 1 && oldCount == 2)
  {
    // Two fingers -> one finger: end the two-finger gestures (if they are
    // still active) and restart the one-finger interaction so it keeps
    // responding without requiring a full lift-and-retouch.
    if (_rotateActive)
    {
      [self handleRotationAngle:_lastRotationAngle state:VTKGestureStateEnded];
    }
    if (_pinchActive)
    {
      [self handlePinchScale:_lastPinchScale state:VTKGestureStateEnded];
    }
    [self handleDragAtViewPoint:p translation:t state:VTKGestureStateBegan];
  }
  else if (touches == 2 && oldCount == 1)
  {
    // One finger -> two fingers: end the one-finger interaction; the pinch and
    // rotation recognizers take over.  They may already have fired their Start
    // events while the one-finger interaction was still active (in which case
    // the style ignored StartGesture because its state was busy), so re-arm the
    // gesture state if needed.
    [self handleDragAtViewPoint:p translation:t state:VTKGestureStateEnded];
    if (_rotateActive)
    {
      _iren->StartRotateEvent();
    }
    if (_pinchActive)
    {
      _iren->StartPinchEvent();
    }
  }
  else if (touches == 1)
  {
    // Gesture began with a single finger.
    [self handleDragAtViewPoint:p translation:t state:state];
  }
}

- (void)handleRotation:(UIRotationGestureRecognizer*)recognizer
{
  [self forwardTouchPosition:recognizer];
  double angle = -[recognizer rotation] * 180.0 / M_PI;
  [self handleRotationAngle:angle state:[self gestureStateFromRecognizer:recognizer]];
}

- (BOOL)gestureRecognizer:(UIGestureRecognizer*)a
    shouldRecognizeSimultaneouslyWithGestureRecognizer:(UIGestureRecognizer*)b
{
  // Pinch, rotation, and the (1-2 finger) pan all track the same touches.  Let
  // them recognize together so the interaction can transition between
  // one-finger and two-finger gestures without a recognizer failing midway and
  // leaving the interaction dead until all fingers are lifted.
  BOOL isOurs =
    (a == self.pinchRecognizer || a == self.rotationRecognizer || a == self.panRecognizer);
  BOOL otherIsOurs =
    (b == self.pinchRecognizer || b == self.rotationRecognizer || b == self.panRecognizer);
  return isOurs && otherIsOurs;
}

#endif

#pragma mark - macOS Event Forwarding

#if TARGET_OS_OSX

- (void)updateEventPositionForView:(NSView*)view event:(NSEvent*)event
{
  CGPoint p = [view convertPoint:event.locationInWindow fromView:nil];
  CGFloat scale = [self contentScaleFactor];
  // NSView coordinates are already bottom-left based (unlike UIKit), so no flip is needed.
  _iren->SetEventInformation((int)(p.x * scale), (int)(p.y * scale),
    (event.modifierFlags & NSEventModifierFlagControl) ? 1 : 0,
    (event.modifierFlags & NSEventModifierFlagShift) ? 1 : 0, 0, 0, 0);
}

- (void)metalView:(NSView*)view mouseDown:(NSEvent*)event
{
  _dragActive = YES;
  _dragStartPoint = [view convertPoint:event.locationInWindow fromView:nil];
  [self updateEventPositionForView:view event:event];
  [self handleDragAtViewPoint:_dragStartPoint
                  translation:CGPointZero
                        state:VTKGestureStateBegan];
}

- (void)metalView:(NSView*)view mouseDragged:(NSEvent*)event
{
  if (!_dragActive) { return; }
  CGPoint p = [view convertPoint:event.locationInWindow fromView:nil];
  [self updateEventPositionForView:view event:event];
  // Normalize translation to "down-positive" (UIKit convention) so the shared
  // drag handlers behave identically on both platforms.
  CGPoint t = CGPointMake(p.x - _dragStartPoint.x, _dragStartPoint.y - p.y);
  [self handleDragAtViewPoint:p translation:t state:VTKGestureStateChanged];
}

- (void)metalView:(NSView*)view mouseUp:(NSEvent*)event
{
  if (!_dragActive) { return; }
  _dragActive = NO;
  CGPoint p = [view convertPoint:event.locationInWindow fromView:nil];
  [self updateEventPositionForView:view event:event];
  CGPoint t = CGPointMake(p.x - _dragStartPoint.x, _dragStartPoint.y - p.y);
  [self handleDragAtViewPoint:p translation:t state:VTKGestureStateEnded];
}

- (void)metalView:(NSView*)view scrollWheel:(NSEvent*)event
{
  [self updateEventPositionForView:view event:event];

  if (event.hasPreciseScrollingDeltas)
  {
    // Trackpad: accumulate precise deltas and emit discrete wheel steps.
    _scrollAccumulator += event.scrollingDeltaY;
    const double kStep = 12.0;
    while (_scrollAccumulator >= kStep)
    {
      [self handleScrollDeltaY:kStep];
      _scrollAccumulator -= kStep;
    }
    while (_scrollAccumulator <= -kStep)
    {
      [self handleScrollDeltaY:-kStep];
      _scrollAccumulator += kStep;
    }
  }
  else
  {
    [self handleScrollDeltaY:event.scrollingDeltaY];
  }
}

- (void)metalView:(NSView*)view magnifyWithEvent:(NSEvent*)event
{
  [self updateEventPositionForView:view event:event];

  NSEventPhase phase = event.phase;
  BOOL began = (phase == NSEventPhaseBegan) ||
    (!_magnifyActive && phase != NSEventPhaseEnded && phase != NSEventPhaseCancelled);
  BOOL ended = (phase == NSEventPhaseEnded) || (phase == NSEventPhaseCancelled);

  if (began)
  {
    _magnifyActive = YES;
    _macPinchScale = 1.0;
  }
  if (!ended)
  {
    _macPinchScale *= (1.0 + event.magnification);
  }

  [self handlePinchScale:_macPinchScale
                   state:began ? VTKGestureStateBegan
                                : (ended ? VTKGestureStateEnded : VTKGestureStateChanged)];
  if (ended)
  {
    _magnifyActive = NO;
  }
}

- (void)metalView:(NSView*)view rotateWithEvent:(NSEvent*)event
{
  [self updateEventPositionForView:view event:event];

  NSEventPhase phase = event.phase;
  BOOL began = (phase == NSEventPhaseBegan) ||
    (!_rotateActive && phase != NSEventPhaseEnded && phase != NSEventPhaseCancelled);
  BOOL ended = (phase == NSEventPhaseEnded) || (phase == NSEventPhaseCancelled);

  if (began)
  {
    _rotateActive = YES;
    _macRotationAngle = 0.0;
  }
  if (!ended)
  {
    _macRotationAngle += -event.rotation * 180.0 / M_PI;
  }

  [self handleRotationAngle:_macRotationAngle
                      state:began ? VTKGestureStateBegan
                                  : (ended ? VTKGestureStateEnded : VTKGestureStateChanged)];
  if (ended)
  {
    _rotateActive = NO;
  }
}

#endif

#pragma mark - Shared Interaction Hooks

- (void)handleDragAtViewPoint:(CGPoint)point
                  translation:(CGPoint)translation
                        state:(VTKGestureState)state
{
  if (state == VTKGestureStateBegan)
  {
    _zoomLastTranslationY = 0.0;
    [self interactionDidStart];
  }

  switch (self.interactionMode)
  {
    case VTKInteractionModeZoom:
      [self handleZoomDragAtViewPoint:point translation:translation state:state];
      break;
    case VTKInteractionModeTrackball:
      [self handleTrackballDragAtViewPoint:point state:state];
      break;
    default:
      [self handlePanDragAtViewPoint:point translation:translation state:state];
      break;
  }

  if (state == VTKGestureStateEnded || state == VTKGestureStateCancelled)
  {
    [self interactionDidEnd];
  }

  _renWin->Render();
}

- (void)handlePanDragAtViewPoint:(CGPoint)point
                     translation:(CGPoint)translation
                           state:(VTKGestureState)state
{
  double t[2] = { [self contentScaleFactor] * translation.x,
                  [self contentScaleFactor] * -translation.y };
  _iren->SetTranslation(t);

  switch (state)
  {
    case VTKGestureStateBegan:
      _iren->StartPanEvent();
      break;
    case VTKGestureStateChanged:
      _iren->PanEvent();
      break;
    case VTKGestureStateEnded:
    case VTKGestureStateCancelled:
      _iren->EndPanEvent();
      break;
    default:
      break;
  }
}

- (void)handleZoomDragAtViewPoint:(CGPoint)point
                      translation:(CGPoint)translation
                            state:(VTKGestureState)state
{
  vtkRenderer* ren = _renderer;
  if (!ren) return;
  vtkCamera* cam = ren->GetActiveCamera();
  if (!cam) return;

  if (state == VTKGestureStateBegan || !_zoomAnchorValid)
  {
    CGFloat scale = [self contentScaleFactor];
    int* vpSize = ren->GetSize();
    _zoomAnchorDisplay[0] = (int)(point.x * scale);
    _zoomAnchorDisplay[1] = vpSize[1] - (int)(point.y * scale); // Flip Y (top-left → bottom-left)

    // Compute anchor world point at the focal-plane depth
    ren->SetDisplayPoint(_zoomAnchorDisplay[0], _zoomAnchorDisplay[1], 0);
    ren->DisplayToWorld();
    double hom[4];
    ren->GetWorldPoint(hom);
    if (hom[3] == 0.0) return;

    double pos[3], fp[3];
    cam->GetPosition(pos);
    cam->GetFocalPoint(fp);
    double dist = cam->GetDistance();
    double dir[3] = {fp[0]-pos[0], fp[1]-pos[1], fp[2]-pos[2]};
    vtkMath::Normalize(dir);

    double ray[3] = {hom[0]/hom[3] - pos[0], hom[1]/hom[3] - pos[1], hom[2]/hom[3] - pos[2]};
    vtkMath::Normalize(ray);

    double cosAngle = vtkMath::Dot(ray, dir);
    if (cosAngle <= 0.0) return;
    double t = dist / cosAngle;
    for (int i = 0; i < 3; ++i) _zoomAnchorWorld[i] = pos[i] + ray[i] * t;

    _zoomAnchorValid = YES;

    if (state == VTKGestureStateBegan)
    {
      _iren->InvokeEvent(vtkCommand::StartInteractionEvent, nullptr);
    }
    return;
  }

  double deltaY = translation.y - _zoomLastTranslationY;
  _zoomLastTranslationY = translation.y;
  double factor = 1.0 - deltaY / 200.0;
  factor = vtkMath::ClampValue(factor, 0.01, 100.0);

  const BOOL didZoom = (fabs(factor - 1.0) > 1e-6);

  if (didZoom)
  {
    cam->Dolly(factor);

    // Compute new world point at focal plane for the same screen anchor
    double pos[3], fp[3];
    cam->GetPosition(pos);
    cam->GetFocalPoint(fp);
    double newDist = cam->GetDistance();
    double dir[3] = {fp[0]-pos[0], fp[1]-pos[1], fp[2]-pos[2]};
    vtkMath::Normalize(dir);

    ren->SetDisplayPoint(_zoomAnchorDisplay[0], _zoomAnchorDisplay[1], 0);
    ren->DisplayToWorld();
    double hom[4];
    ren->GetWorldPoint(hom);
    if (hom[3] == 0.0) return;

    double ray[3] = {hom[0]/hom[3] - pos[0], hom[1]/hom[3] - pos[1], hom[2]/hom[3] - pos[2]};
    vtkMath::Normalize(ray);

    double cosAngle = vtkMath::Dot(ray, dir);
    if (cosAngle <= 0.0) return;
    double t = newDist / cosAngle;
    double newAnchorWorld[3] = {pos[0] + ray[0]*t, pos[1] + ray[1]*t, pos[2] + ray[2]*t};

    // Translate camera to keep the anchor fixed in world space
    double d[3] = {
      _zoomAnchorWorld[0] - newAnchorWorld[0],
      _zoomAnchorWorld[1] - newAnchorWorld[1],
      _zoomAnchorWorld[2] - newAnchorWorld[2]};

    for (int i = 0; i < 3; ++i) { pos[i] += d[i]; fp[i] += d[i]; }
    cam->SetPosition(pos);
    cam->SetFocalPoint(fp);

    ren->ResetCameraClippingRange();
  }

  switch (state)
  {
    case VTKGestureStateChanged:
      if (didZoom)
        _iren->InvokeEvent(vtkCommand::InteractionEvent, nullptr);
      break;
    case VTKGestureStateEnded:
    case VTKGestureStateCancelled:
      _iren->InvokeEvent(vtkCommand::EndInteractionEvent, nullptr);
      _zoomAnchorValid = NO;
      break;
    default:
      break;
  }
}

- (void)handleTrackballDragAtViewPoint:(CGPoint)point state:(VTKGestureState)state
{
  switch (state)
  {
    case VTKGestureStateBegan:
      _iren->InvokeEvent(vtkCommand::LeftButtonPressEvent, nullptr);
      break;
    case VTKGestureStateChanged:
      _iren->InvokeEvent(vtkCommand::MouseMoveEvent, nullptr);
      break;
    case VTKGestureStateEnded:
    case VTKGestureStateCancelled:
      _iren->InvokeEvent(vtkCommand::LeftButtonReleaseEvent, nullptr);
      break;
    default:
      break;
  }
}

- (void)handlePinchScale:(CGFloat)scale state:(VTKGestureState)state
{
  if (state == VTKGestureStateNone)
  {
    return;
  }

  if (state == VTKGestureStateBegan)
  {
    _pinchActive = YES;
    [self interactionDidStart];
  }
  else if (!_pinchActive)
  {
    // Ignore Changed/Ended/Cancelled for an interaction that never started (a
    // recognizer that failed or was force-ended during a finger-count
    // transition).  Ending it again would tear down a different interaction
    // that is still in flight.
    return;
  }

  scale = MAX(-3.0, MIN(3.0, scale));
  _lastPinchScale = scale;
  _iren->SetScale(scale);

  switch (state)
  {
    case VTKGestureStateBegan:
      _iren->StartPinchEvent();
      break;
    case VTKGestureStateChanged:
      _iren->PinchEvent();
      break;
    case VTKGestureStateEnded:
    case VTKGestureStateCancelled:
      _iren->EndPinchEvent();
      _pinchActive = NO;
      [self interactionDidEnd];
      break;
    default:
      break;
  }

  _renWin->Render();
}

- (void)handleRotationAngle:(CGFloat)angle state:(VTKGestureState)state
{
  if (state == VTKGestureStateNone)
  {
    return;
  }

  if (state == VTKGestureStateBegan)
  {
    _rotateActive = YES;
    [self interactionDidStart];
  }
  else if (!_rotateActive)
  {
    return;
  }

  _lastRotationAngle = angle;
  _iren->SetRotation(angle);

  switch (state)
  {
    case VTKGestureStateBegan:
      _iren->StartRotateEvent();
      break;
    case VTKGestureStateChanged:
      _iren->RotateEvent();
      break;
    case VTKGestureStateEnded:
    case VTKGestureStateCancelled:
      _iren->EndRotateEvent();
      _rotateActive = NO;
      [self interactionDidEnd];
      break;
    default:
      break;
  }

  _renWin->Render();
}

- (void)handleScrollDeltaY:(CGFloat)deltaY
{
  if (deltaY > 0)
  {
    _iren->InvokeEvent(vtkCommand::MouseWheelForwardEvent, nullptr);
  }
  else if (deltaY < 0)
  {
    _iren->InvokeEvent(vtkCommand::MouseWheelBackwardEvent, nullptr);
  }
  _renWin->Render();
}

- (void)interactionDidStart
{
}

- (void)interactionDidEnd
{
}

#pragma mark - Layout

#if TARGET_OS_OSX
- (void)viewDidLayout
{
  [super viewDidLayout];
  [self updateRenderSize];
}
#else
- (void)viewDidLayoutSubviews
{
  [super viewDidLayoutSubviews];
  [self updateRenderSize];
}
#endif

- (void)updateRenderSize
{
  int w = (int)lround([self contentScaleFactor] * self.view.bounds.size.width);
  int h = (int)lround([self contentScaleFactor] * self.view.bounds.size.height);
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
