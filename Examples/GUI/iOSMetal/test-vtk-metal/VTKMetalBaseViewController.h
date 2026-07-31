#import <TargetConditionals.h>
#import "VTKInteractionMode.h"

#if TARGET_OS_OSX
#import <Cocoa/Cocoa.h>
#import "vtkCocoaMetalRenderWindow.h"
#else
#import <UIKit/UIKit.h>
#endif

/** Platform-neutral gesture state used by the interaction hooks below so that
 *  subclasses can handle mouse/trackpad events on macOS and touch gestures on
 *  iOS without depending on UIKit types.
 */
typedef NS_ENUM(NSInteger, VTKGestureState)
{
  VTKGestureStateNone = -1,
  VTKGestureStateBegan = 0,
  VTKGestureStateChanged = 1,
  VTKGestureStateEnded = 2,
  VTKGestureStateCancelled = 3,
};

#if TARGET_OS_OSX
@interface VTKMetalBaseViewController : NSViewController <VTKCocoaMetalViewDelegate>
#else
@interface VTKMetalBaseViewController : UIViewController <UIGestureRecognizerDelegate>
#endif
@property (nonatomic, readonly) void* renderer;
@property (nonatomic, readonly) void* renderWindow;

@property (nonatomic) VTKInteractionMode interactionMode;
- (NSString*)interactionModeTitle;
- (NSString*)interactionModeImageName;

- (void)resetCamera;
- (void)startBenchmark;
- (void)stopBenchmark;
@property (nonatomic, readonly, getter=isBenchmarkRunning) BOOL benchmarkRunning;

// Interaction hooks.  Subclasses override these (instead of gesture recognizer
// methods) to customize behavior on both platforms.
- (void)handleDragAtViewPoint:(CGPoint)point
                  translation:(CGPoint)translation
                        state:(VTKGestureState)state;
- (void)handlePinchScale:(CGFloat)scale state:(VTKGestureState)state;
- (void)handleRotationAngle:(CGFloat)angle state:(VTKGestureState)state;
- (void)handleScrollDeltaY:(CGFloat)deltaY;

// Overridable interaction lifecycle callbacks (e.g. for dynamic sample rate).
- (void)interactionDidStart;
- (void)interactionDidEnd;

@end
