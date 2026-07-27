#import <UIKit/UIKit.h>

@interface VTKMetalBaseViewController : UIViewController <UIGestureRecognizerDelegate>
@property (nonatomic, readonly) void* renderer;
@property (nonatomic, readonly) void* renderWindow;

- (void)startBenchmark;
- (void)stopBenchmark;
@property (nonatomic, readonly, getter=isBenchmarkRunning) BOOL benchmarkRunning;
@end
