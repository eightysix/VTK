#import "AppDelegate.h"
#import "ViewController.h"
#import "FileVolumeViewController.h"
#import <UIKit/UIKit.h>

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

  // Rendering menu with inline preset commands
  UIMenu* renderingMenu = [UIMenu
      menuWithTitle:@"Rendering"
          children:@[ nextPresetCmd, prevPresetCmd ]];

  [builder insertSiblingMenu:renderingMenu
      afterMenuForIdentifier:UIMenuView];
}

@end
