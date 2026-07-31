#import <TargetConditionals.h>

#if TARGET_OS_OSX
#import <Cocoa/Cocoa.h>
#else
#import <UIKit/UIKit.h>
#endif

#if TARGET_OS_OSX
@interface ViewController : NSViewController
#else
@interface ViewController : UIViewController
#endif
@property (nonatomic) NSInteger selectedIndex;
#if TARGET_OS_OSX
@property (nonatomic, readonly) NSViewController* currentViewController;
#else
@property (nonatomic, readonly) UIViewController* currentViewController;
#endif
@end
