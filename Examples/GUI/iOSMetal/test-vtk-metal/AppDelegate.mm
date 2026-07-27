#import "AppDelegate.h"
#import "ViewController.h"
#import "FileVolumeViewController.h"
#import "VTKMetalBaseViewController.h"
#import <UIKit/UIKit.h>

#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingMetal);

@interface AppDelegate ()
@property (nonatomic, readwrite) BOOL benchmarkAutoStarted;
@end

@implementation AppDelegate

- (BOOL)application:(UIApplication*)application
    didFinishLaunchingWithOptions:(NSDictionary*)launchOptions
{
  self.window = [[UIWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];
  ViewController* vc = [[ViewController alloc] init];
  self.window.rootViewController = vc;
  [self.window makeKeyAndVisible];

  // Auto-start benchmark if launched with -benchmark flag
  if ([[NSProcessInfo processInfo].arguments containsObject:@"-benchmark"])
  {
    dispatch_async(dispatch_get_main_queue(), ^{
      [self startBenchmarkOnVolumeTab];
    });
  }

  return YES;
}

- (void)buildMenuWithBuilder:(id<UIMenuBuilder>)builder
{
  [super buildMenuWithBuilder:builder];

  if (builder.system != [UIMenuSystem mainSystem]) return;

  // VR Preset commands — actions go through the responder chain
  UIKeyCommand* nextPresetCmd = [UIKeyCommand
      keyCommandWithInput:@"k"
            modifierFlags:UIKeyModifierControl | UIKeyModifierAlternate
                   action:@selector(nextPreset:)];
  nextPresetCmd.title = @"Next VR Preset";
  nextPresetCmd.discoverabilityTitle = @"Next VR Preset";

  UIKeyCommand* prevPresetCmd = [UIKeyCommand
      keyCommandWithInput:@"j"
            modifierFlags:UIKeyModifierControl | UIKeyModifierAlternate
                   action:@selector(previousPreset:)];
  prevPresetCmd.title = @"Previous VR Preset";
  prevPresetCmd.discoverabilityTitle = @"Previous VR Preset";

  UIKeyCommand* benchmarkCmd = [UIKeyCommand
      keyCommandWithInput:@"b"
            modifierFlags:UIKeyModifierCommand
                   action:@selector(toggleBenchmark:)];
  benchmarkCmd.title = @"Toggle Benchmark";
  benchmarkCmd.discoverabilityTitle = @"Toggle GPU Benchmark";

  // Rendering menu with inline preset commands
  UIMenu* renderingMenu = [UIMenu
      menuWithTitle:@"Rendering"
          children:@[ nextPresetCmd, prevPresetCmd, benchmarkCmd ]];

  [builder insertSiblingMenu:renderingMenu
      afterMenuForIdentifier:UIMenuView];
}

#pragma mark - Benchmark Actions

- (void)toggleBenchmark:(id)sender
{
  VTKMetalBaseViewController* vc = [self findMetalViewController];
  if (vc)
  {
    if (vc.isBenchmarkRunning)
    {
      [vc stopBenchmark];
    }
    else
    {
      [vc startBenchmark];
    }
  }
}

- (void)startBenchmarkOnVolumeTab
{
  UITabBarController* tabBar = [self findTabBarController];
  if (!tabBar) { return; }

  // Volume tab is at index 2, select it
  tabBar.selectedIndex = 2;

  VTKMetalBaseViewController* vc = [self findMetalViewController];
  if (vc)
  {
    [vc startBenchmark];
    self.benchmarkAutoStarted = YES;
  }
}

- (VTKMetalBaseViewController*)findMetalViewController
{
  UITabBarController* tabBar = [self findTabBarController];
  if (!tabBar) { return nil; }

  UIViewController* selected = tabBar.selectedViewController;
  if ([selected isKindOfClass:[UINavigationController class]])
  {
    selected = [(UINavigationController*)selected topViewController];
  }

  if ([selected isKindOfClass:[VTKMetalBaseViewController class]])
  {
    return (VTKMetalBaseViewController*)selected;
  }
  return nil;
}

- (UITabBarController*)findTabBarController
{
  UIViewController* root = self.window.rootViewController;
  if ([root isKindOfClass:[UITabBarController class]])
  {
    return (UITabBarController*)root;
  }
  // If root is ViewController which has a tab bar as child
  for (UIViewController* child in root.children)
  {
    if ([child isKindOfClass:[UITabBarController class]])
    {
      return (UITabBarController*)child;
    }
  }
  return nil;
}

@end
