#import "VTKMetalBaseViewController.h"
#if TARGET_OS_OSX
#import <Cocoa/Cocoa.h>
#else
#import <UIKit/UIKit.h>
#endif

#if TARGET_OS_OSX
@interface AppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@property (nonatomic, strong) NSWindow* window;
#else
@interface AppDelegate : UIResponder <UIApplicationDelegate>
@property (nonatomic, strong) UIWindow* window;
#endif

- (VTKMetalBaseViewController*)findMetalViewController;

- (void)activatePanMode:(id)sender;
- (void)activateZoomMode:(id)sender;
- (void)activateTrackballMode:(id)sender;
- (void)activateScrollSlicesMode:(id)sender;
- (void)activateWindowLevelMode:(id)sender;
- (void)resetCamera:(id)sender;
- (void)toggleDynamicSampleRate:(id)sender;

#if TARGET_OS_OSX
- (void)nextPreset:(id)sender;
- (void)previousPreset:(id)sender;
- (void)toggleBenchmark:(id)sender;
- (void)increaseSampleDistance:(id)sender;
- (void)decreaseSampleDistance:(id)sender;
- (void)loadFile:(id)sender;
- (void)toggleVolumeTranspose:(id)sender;
- (void)toggleMinMaxAcceleration:(id)sender;
- (void)toggleBlueNoiseJitter:(id)sender;
#endif
@end
