#import "AppDelegate.h"
#if TARGET_OS_OSX
#import <Cocoa/Cocoa.h>

int main(int argc, char* argv[])
{
  @autoreleasepool
  {
    NSApplication* app = [NSApplication sharedApplication];
    AppDelegate* delegate = [[AppDelegate alloc] init];
    app.delegate = delegate;
    [app setActivationPolicy:NSApplicationActivationPolicyRegular];
    [app run];
  }
  return 0;
}
#else
#import <UIKit/UIKit.h>

int main(int argc, char* argv[])
{
  @autoreleasepool
  {
    return UIApplicationMain(argc, argv, nil, NSStringFromClass([AppDelegate class]));
  }
}
#endif
