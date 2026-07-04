#import "AppDelegate.h"
#import "ViewController.h"

#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingMetal);

@implementation AppDelegate

- (BOOL)application:(UIApplication*)application
    didFinishLaunchingWithOptions:(NSDictionary*)launchOptions
{
  self.window = [[UIWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];
  ViewController* vc = [[ViewController alloc] init];
  self.window.rootViewController = vc;
  [self.window makeKeyAndVisible];
  return YES;
}

@end
