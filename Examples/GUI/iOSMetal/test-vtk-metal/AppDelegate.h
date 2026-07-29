#import <UIKit/UIKit.h>
#import "VTKMetalBaseViewController.h"

@interface AppDelegate : UIResponder <UIApplicationDelegate>
@property (nonatomic, strong) UIWindow* window;

- (VTKMetalBaseViewController*)findMetalViewController;

- (void)activatePanMode:(id)sender;
- (void)activateZoomMode:(id)sender;
- (void)activateTrackballMode:(id)sender;
- (void)activateScrollSlicesMode:(id)sender;
- (void)activateWindowLevelMode:(id)sender;
- (void)resetCamera:(id)sender;
- (void)toggleDynamicSampleRate:(id)sender;
@end
