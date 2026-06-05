#import "AppDelegate.h"

#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingWebGPU);

@interface AppDelegate ()
@end

@implementation AppDelegate

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    return YES;
}

@end
