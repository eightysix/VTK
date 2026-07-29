#import "AppDelegate.h"
#import "BaseVolumeViewController.h"
#import "ViewController.h"
#import "DICOMVolumeViewController.h"
#import "FileVolumeViewController.h"
#import "NIFTIVolumeViewController.h"
#import "VTKMetalBaseViewController.h"
#import "VTKInteractionMode.h"
#import <UIKit/UIKit.h>

@interface AppDelegate ()
@property (nonatomic, readwrite) BOOL benchmarkAutoStarted;
@end

static NSArray<NSDictionary*>* ViewCommandDefs(void)
{
  static NSArray* defs;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    defs = @[
      @{ @"title": @"Cube",   @"action": @"selectCube:" },
      @{ @"title": @"Cone",   @"action": @"selectCone:" },
      @{ @"title": @"Volume", @"action": @"selectVolume:" },
      @{ @"title": @"DICOM",  @"action": @"selectDICOM:" },
      @{ @"title": @"NIfTI",  @"action": @"selectNIfTI:" },
      @{ @"title": @"Base",   @"action": @"selectBase:" },
    ];
  });
  return defs;
}

@implementation AppDelegate

- (BOOL)validateMenuItem:(UIKeyCommand*)menuItem
{
  SEL action = menuItem.action;
  if (action == @selector(activateScrollSlicesMode:) ||
      action == @selector(activateWindowLevelMode:))
  {
    VTKMetalBaseViewController* vc = [self findMetalViewController];
    return [vc isKindOfClass:[DICOMVolumeViewController class]] ||
           [vc isKindOfClass:[NIFTIVolumeViewController class]];
  }
  return YES;
}

- (void)validateCommand:(UICommand*)command
{
  NSUInteger index = [ViewCommandDefs() indexOfObjectPassingTest:^BOOL(NSDictionary* d, NSUInteger idx, BOOL* stop) {
    return NSSelectorFromString(d[@"action"]) == command.action;
  }];

  if (index != NSNotFound)
  {
    ViewController* rootVC = (ViewController*)self.window.rootViewController;
    if ([rootVC isKindOfClass:[ViewController class]])
    {
      command.state = (rootVC.selectedIndex == (NSInteger)index) ? UIMenuElementStateOn : UIMenuElementStateOff;
    }
    return;
  }

  // Interaction mode commands
  SEL cmdAction = command.action;
  VTKInteractionMode cmdMode = (VTKInteractionMode)[command.propertyList integerValue];
  if (cmdAction == @selector(activatePanMode:) ||
      cmdAction == @selector(activateZoomMode:) ||
      cmdAction == @selector(activateTrackballMode:) ||
      cmdAction == @selector(activateScrollSlicesMode:) ||
      cmdAction == @selector(activateWindowLevelMode:))
  {
    VTKMetalBaseViewController* vc = [self findMetalViewController];
    command.state = (vc && vc.interactionMode == cmdMode) ? UIMenuElementStateOn : UIMenuElementStateOff;

    if (cmdAction == @selector(activateScrollSlicesMode:) ||
        cmdAction == @selector(activateWindowLevelMode:))
    {
      BOOL isDICOMorNIFTI = [vc isKindOfClass:[DICOMVolumeViewController class]] ||
                             [vc isKindOfClass:[NIFTIVolumeViewController class]];
      if (!isDICOMorNIFTI)
        command.attributes = UIMenuElementAttributesDisabled;
    }
    return;
  }

  if (cmdAction == @selector(toggleDynamicSampleRate:))
  {
    VTKMetalBaseViewController* vc = [self findMetalViewController];
    if ([vc isKindOfClass:[BaseVolumeViewController class]])
    {
      BaseVolumeViewController* bvc = (BaseVolumeViewController*)vc;
      command.state = bvc.isDynamicSampleRateAdjustmentEnabled
        ? UIMenuElementStateOn : UIMenuElementStateOff;
    }
    else
    {
      command.attributes = UIMenuElementAttributesDisabled;
      command.state = UIMenuElementStateOn;
    }
  }
}

- (BOOL)application:(UIApplication*)application
didFinishLaunchingWithOptions:(NSDictionary*)launchOptions
{
  self.window = [[UIWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];
  ViewController* vc = [[ViewController alloc] init];
  self.window.rootViewController = vc;
  [self.window makeKeyAndVisible];
  
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
  
  NSMutableArray* viewCommands = [NSMutableArray array];
  [ViewCommandDefs() enumerateObjectsUsingBlock:^(NSDictionary* d, NSUInteger i, BOOL* stop) {
    SEL action = NSSelectorFromString(d[@"action"]);
    UIKeyCommand* cmd = [UIKeyCommand commandWithTitle:d[@"title"]
                                                 image:nil
                                                action:action
                                                 input:[@(i + 1) stringValue]
                                         modifierFlags:UIKeyModifierCommand
                                          propertyList:@(i)];
    cmd.discoverabilityTitle = [NSString stringWithFormat:@"Switch to %@", d[@"title"]];
    [viewCommands addObject:cmd];
  }];
  
  // Build submenus for the VTK top-level menu
  
  // Views submenu
  UIMenu* viewsMenu = [UIMenu menuWithTitle:@"Views" children:viewCommands];

  // Interaction Mode submenu
  UIKeyCommand* panCmd = [UIKeyCommand commandWithTitle:@"Pan"
                                                  image:[UIImage systemImageNamed:@"hand.point.up"]
                                                 action:@selector(activatePanMode:)
                                                  input:@"s"
                                          modifierFlags:0
                                           propertyList:@(VTKInteractionModePan)];
  panCmd.discoverabilityTitle = @"Pan";

  UIKeyCommand* zoomCmd = [UIKeyCommand commandWithTitle:@"Zoom"
                                                    image:[UIImage systemImageNamed:@"magnifyingglass"]
                                                   action:@selector(activateZoomMode:)
                                                    input:@"a"
                                            modifierFlags:0
                                             propertyList:@(VTKInteractionModeZoom)];
  zoomCmd.discoverabilityTitle = @"Zoom";

  UIKeyCommand* trackballCmd = [UIKeyCommand commandWithTitle:@"Trackball"
                                                          image:[UIImage systemImageNamed:@"cube.transparent"]
                                                         action:@selector(activateTrackballMode:)
                                                          input:@"x"
                                                  modifierFlags:0
                                                   propertyList:@(VTKInteractionModeTrackball)];
  trackballCmd.discoverabilityTitle = @"Trackball";

  UIKeyCommand* scrollSlicesCmd = [UIKeyCommand commandWithTitle:@"Scroll Slices"
                                                             image:[UIImage systemImageNamed:@"arrow.up.and.down"]
                                                            action:@selector(activateScrollSlicesMode:)
                                                             input:@"d"
                                                     modifierFlags:0
                                                      propertyList:@(VTKInteractionModeScrollSlices)];
  scrollSlicesCmd.discoverabilityTitle = @"Clip through volume";

  UIKeyCommand* windowLevelCmd = [UIKeyCommand commandWithTitle:@"Window/Level"
                                                          image:[UIImage systemImageNamed:@"sun.max"]
                                                         action:@selector(activateWindowLevelMode:)
                                                          input:@"w"
                                                  modifierFlags:0
                                                   propertyList:@(VTKInteractionModeWindowLevel)];
  windowLevelCmd.discoverabilityTitle = @"Adjust window/level";

  UIMenu* interactionMenu = [UIMenu menuWithTitle:@"Interaction Mode"
                                         children:@[ panCmd, zoomCmd, trackballCmd, scrollSlicesCmd, windowLevelCmd ]];

  // Camera submenu
  UIKeyCommand* resetCameraCmd = [UIKeyCommand
                                   keyCommandWithInput:@"r"
                                   modifierFlags:UIKeyModifierCommand
                                   action:@selector(resetCamera:)];
  resetCameraCmd.title = @"Reset Camera";
  resetCameraCmd.discoverabilityTitle = @"Reset camera to default view";

  UIMenu* cameraMenu = [UIMenu menuWithTitle:@"Camera" children:@[ resetCameraCmd ]];
  
  // Rendering submenu
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
                                modifierFlags:UIKeyModifierCommand | UIKeyModifierAlternate
                                action:@selector(toggleBenchmark:)];
  benchmarkCmd.title = @"Toggle Benchmark";
  benchmarkCmd.discoverabilityTitle = @"Toggle GPU Benchmark";
  
  UIKeyCommand* incSampleCmd = [UIKeyCommand
                                keyCommandWithInput:@"+"
                                modifierFlags:UIKeyModifierControl | UIKeyModifierAlternate
                                action:@selector(increaseSampleDistance:)];
  incSampleCmd.title = @"Increase Sample Distance";
  incSampleCmd.discoverabilityTitle = @"Increase sample distance by 0.5";

  UIKeyCommand* decSampleCmd = [UIKeyCommand
                                keyCommandWithInput:@"-"
                                modifierFlags:UIKeyModifierControl | UIKeyModifierAlternate
                                action:@selector(decreaseSampleDistance:)];
  decSampleCmd.title = @"Decrease Sample Distance";
  decSampleCmd.discoverabilityTitle = @"Decrease sample distance by 0.5";

  UIKeyCommand* dynSampleCmd = [UIKeyCommand
                                keyCommandWithInput:@"y"
                                modifierFlags:UIKeyModifierCommand | UIKeyModifierAlternate
                                action:@selector(toggleDynamicSampleRate:)];
  dynSampleCmd.title = @"Toggle Dynamic Sample Rate";
  dynSampleCmd.discoverabilityTitle = @"Toggle automatic sample rate adjustment during interaction";

  UIMenu* renderingMenu = [UIMenu
                           menuWithTitle:@"Rendering"
                           children:@[ nextPresetCmd, prevPresetCmd, benchmarkCmd, dynSampleCmd, incSampleCmd, decSampleCmd ]];

  // File submenu
  UIKeyCommand* loadFileCmd = [UIKeyCommand
                               keyCommandWithInput:@"o"
                               modifierFlags:UIKeyModifierCommand
                               action:@selector(loadFile:)];
  loadFileCmd.title = @"Load File…";
  loadFileCmd.discoverabilityTitle = @"Open a file via document picker";
  
  UIMenu* fileMenu = [UIMenu menuWithTitle:@"File" children:@[ loadFileCmd ]];

  // Top-level VTK menu containing all submenus
  UIMenu* vtkMenu = [UIMenu menuWithTitle:@"VTK"
                                   children:@[ viewsMenu, interactionMenu, cameraMenu, renderingMenu, fileMenu ]];
  [builder insertSiblingMenu:vtkMenu afterMenuForIdentifier:UIMenuApplication];
}

#pragma mark - View Switching

- (ViewController*)viewControllerForSwitch
{
  ViewController* rootVC = (ViewController*)self.window.rootViewController;
  NSAssert([rootVC isKindOfClass:[ViewController class]],
           @"Root view controller must be a ViewController");
  return rootVC;
}

- (void)selectViewAtIndex:(NSInteger)index
{
  [self viewControllerForSwitch].selectedIndex = index;
}

- (void)selectCube:(id)sender   { [self selectViewAtIndex:0]; }
- (void)selectCone:(id)sender   { [self selectViewAtIndex:1]; }
- (void)selectVolume:(id)sender { [self selectViewAtIndex:2]; }
- (void)selectDICOM:(id)sender  { [self selectViewAtIndex:3]; }
- (void)selectNIfTI:(id)sender  { [self selectViewAtIndex:4]; }
- (void)selectBase:(id)sender   { [self selectViewAtIndex:5]; }

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
  [self viewControllerForSwitch].selectedIndex = 2;
  
  VTKMetalBaseViewController* vc = [self findMetalViewController];
  if (vc)
  {
    [vc startBenchmark];
    self.benchmarkAutoStarted = YES;
  }
}

- (VTKMetalBaseViewController*)findMetalViewController
{
  UIViewController* current = [self viewControllerForSwitch].currentViewController;
  
  if ([current isKindOfClass:[VTKMetalBaseViewController class]])
  {
    return (VTKMetalBaseViewController*)current;
  }
  return nil;
}

#pragma mark - Interaction Mode Actions

- (void)activatePanMode:(id)sender
{
  [self findMetalViewController].interactionMode = VTKInteractionModePan;
  [self updateInteractionModeMenu];
}

- (void)activateZoomMode:(id)sender
{
  [self findMetalViewController].interactionMode = VTKInteractionModeZoom;
  [self updateInteractionModeMenu];
}

- (void)activateTrackballMode:(id)sender
{
  [self findMetalViewController].interactionMode = VTKInteractionModeTrackball;
  [self updateInteractionModeMenu];
}

- (void)activateScrollSlicesMode:(id)sender
{
  [self findMetalViewController].interactionMode = VTKInteractionModeScrollSlices;
  [self updateInteractionModeMenu];
}

- (void)activateWindowLevelMode:(id)sender
{
  [self findMetalViewController].interactionMode = VTKInteractionModeWindowLevel;
  [self updateInteractionModeMenu];
}

- (void)resetCamera:(id)sender
{
  [[self findMetalViewController] resetCamera];
}

- (void)toggleDynamicSampleRate:(id)sender
{
  VTKMetalBaseViewController* vc = [self findMetalViewController];
  if ([vc respondsToSelector:@selector(isDynamicSampleRateAdjustmentEnabled)])
  {
    BaseVolumeViewController* bvc = (BaseVolumeViewController*)vc;
    bvc.dynamicSampleRateAdjustmentEnabled = !bvc.isDynamicSampleRateAdjustmentEnabled;
    [UIMenuSystem.mainSystem setNeedsRebuild];
  }
}

- (void)updateInteractionModeMenu
{
  [UIMenuSystem.mainSystem setNeedsRebuild];
}

@end
