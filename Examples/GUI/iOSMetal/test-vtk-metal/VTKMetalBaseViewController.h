#import <UIKit/UIKit.h>

@interface VTKMetalBaseViewController : UIViewController <UIGestureRecognizerDelegate>
@property (nonatomic, readonly) void* renderer;
@property (nonatomic, readonly) void* renderWindow;
@end
