#import <UIKit/UIKit.h>

@interface VTKBaseViewController : UIViewController <UIGestureRecognizerDelegate>
@property (nonatomic, readonly) void *renderer;
@property (nonatomic, readonly) void *renderWindow;
@end
