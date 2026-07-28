#import <UIKit/UIKit.h>
#import "VTKInteractionMode.h"

@interface VTKMetalBaseViewController : UIViewController <UIGestureRecognizerDelegate>
@property (nonatomic, readonly) void* renderer;
@property (nonatomic, readonly) void* renderWindow;

@property (nonatomic) VTKInteractionMode interactionMode;
- (NSString*)interactionModeTitle;
- (NSString*)interactionModeImageName;

- (void)resetCamera;
- (void)startBenchmark;
- (void)stopBenchmark;
@property (nonatomic, readonly, getter=isBenchmarkRunning) BOOL benchmarkRunning;
@end
