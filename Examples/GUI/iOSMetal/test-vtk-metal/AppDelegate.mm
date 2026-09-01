#import "AppDelegate.h"
#import "BaseVolumeViewController.h"
#import "ViewController.h"
#import "DICOMVolumeViewController.h"
#import "FileVolumeViewController.h"
#import "NIFTIVolumeViewController.h"
#import "VTKMetalBaseViewController.h"
#import "VTKInteractionMode.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "vtkMetalGPUVolumeRayCastMapper.h"
#include "vtkRenderWindow.h"
#if TARGET_OS_OSX
#import <Cocoa/Cocoa.h>
#else
#import <UIKit/UIKit.h>
#endif

@interface AppDelegate ()
@property (nonatomic, readwrite) BOOL benchmarkAutoStarted;
// Runtime render-config toggles (Rendering menu). Defaults mirror the
// mapper/env defaults: transpose on, minmax accel on, IGN (not blue-noise)
// jitter.
@property (nonatomic) BOOL volumeTransposeEnabled;
@property (nonatomic) BOOL minMaxAccelerationEnabled;
@property (nonatomic) BOOL blueNoiseJitterEnabled;
// March-variant toggle (mv9 = 48-wide inline scheduled march vs mv0 baseline).
// marchVariantEnvAtLaunch preserves a custom VTK_METAL_TEST_MARCH_VARIANT set
// at launch so toggling off restores it, not just the 0 default.
@property (nonatomic) BOOL marchVariant9Enabled;
@property (nonatomic, copy) NSString* marchVariantEnvAtLaunch;
// Two-level occupancy summary (HARNESS_VS_APP_GAP §34): 8³-cell all-empty
// block summary of the dilated macrocell lattice; the minmax walk leaps whole
// empty blocks. Mapper-gated to the fine-SD tier (sampleDistance < 1.5).
@property (nonatomic) BOOL minMaxBlocksEnabled;
// Fragment compile-time batch specialization (HARNESS_VS_APP_GAP §39):
// fc_fragBatch [[function_constant(41)]] - 0=shipped runtime 32 heavy,
// 8/16/32=light (16 orbit-best). Baked into featureMaskExtra bits [10:15].
@property (nonatomic) int fragBatchValue;
// Transposed depth-axis choice (HARNESS_VS_APP_GAP §35.5/§35.8): which short
// in-plane extent plays texture depth under VOLTRANSPOSE. Y measures -8..-14%
// vs the policy's X tie-break under mm+blocks at fine SD. OFF restores x (the
// argmin-dims tie-break); ON forces VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y.
// Inert while Volume Transpose is off.
@property (nonatomic) BOOL volumeTransposeAxisYEnabled;
// Tie-break toggle for 512x512 (LL vs AP): when ON, X==Y ties prefer Y
// (VTK_METAL_TEST_VOLTRANSPOSE_Y_TIE=1) e.g. DICOM 512x512x1794. Default X.
@property (nonatomic) BOOL volumeTransposeYTieEnabled;
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

#if TARGET_OS_OSX

#pragma mark - Application Lifecycle (macOS)

- (void)applicationDidFinishLaunching:(NSNotification*)notification
{
  self.volumeTransposeEnabled = YES;
  if (const char* v = std::getenv("VTK_METAL_TEST_VOLTRANSPOSE"))
    self.volumeTransposeEnabled = std::atoi(v) != 0;
  self.minMaxAccelerationEnabled = YES;
  self.blueNoiseJitterEnabled = NO;

  self.marchVariantEnvAtLaunch = @"0";
  if (const char* v = std::getenv("VTK_METAL_TEST_MARCH_VARIANT"))
  {
    self.marchVariantEnvAtLaunch = [NSString stringWithUTF8String:v];
    self.marchVariant9Enabled = std::atoi(v) == 9;
  }

  // Mirrors the mapper env default: block summary off unless opted in.
  self.minMaxBlocksEnabled = NO;
  if (const char* v = std::getenv("VTK_METAL_TEST_MM_BLOCKS"))
    self.minMaxBlocksEnabled = std::atoi(v) != 0;

  // Depth-axis override state; unset env means the argmin-dims policy (which
  // prefers X on ties), so OFF matches the effective default on Z-long studies.
  self.volumeTransposeAxisYEnabled = NO;
  if (const char* v = std::getenv("VTK_METAL_TEST_VOLTRANSPOSE_AXIS"))
    self.volumeTransposeAxisYEnabled = (v[0] == 'y' || v[0] == 'Y');
  // Tie-break toggle for 512x512 X==Y (LL vs AP): OFF X tie (mm:496), ON Y tie
  // (VTK_METAL_TEST_VOLTRANSPOSE_Y_TIE=1) e.g. DICOM 512x512x1794.
  self.volumeTransposeYTieEnabled = NO;
  if (const char* v = std::getenv("VTK_METAL_TEST_VOLTRANSPOSE_Y_TIE"))
    self.volumeTransposeYTieEnabled = std::atoi(v) != 0;

  // Fragment batch specialization (§39): 0=shipped runtime 32, 8/16/32=light.
  self.fragBatchValue = 0;
  if (const char* v = std::getenv("VTK_METAL_TEST_FRAG_BATCH"))
    self.fragBatchValue = std::max(0, std::min(48, std::atoi(v)));

  NSRect contentRect = NSMakeRect(0, 0, 1100, 800);
  NSWindowStyleMask styleMask = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
    NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;
  self.window = [[NSWindow alloc] initWithContentRect:contentRect
                                            styleMask:styleMask
                                              backing:NSBackingStoreBuffered
                                                defer:NO];
  [self.window setTitle:@"VTK Metal"];
  [self.window setDelegate:self];

  ViewController* vc = [[ViewController alloc] init];
  [vc.view setFrameSize:self.window.contentView.bounds.size];
  [self.window setContentViewController:vc];

  [self buildMenu];

  [self.window center];
  [self.window makeKeyAndOrderFront:nil];

  if ([[NSProcessInfo processInfo].arguments containsObject:@"-benchmark"])
  {
    dispatch_async(dispatch_get_main_queue(), ^{
      [self startBenchmarkOnVolumeTab];
    });
  }

  [NSApp activateIgnoringOtherApps:YES];
}

- (void)applicationWillTerminate:(NSNotification*)notification
{
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)sender
{
  return YES;
}

- (BOOL)applicationSupportsSecureRestorableState:(NSApplication*)app
{
  return YES;
}

#pragma mark - Menu Building (macOS)

- (void)buildMenu
{
  NSMenu* menubar = [[NSMenu alloc] init];

  // Application menu
  NSMenuItem* appMenuItem = [[NSMenuItem alloc] initWithTitle:@"VTK Metal"
                                                       action:nil
                                                keyEquivalent:@""];
  [menubar addItem:appMenuItem];
  NSMenu* appMenu = [[NSMenu alloc] initWithTitle:@"VTK Metal"];
  [appMenu addItemWithTitle:@"About VTK Metal"
                     action:@selector(orderFrontStandardAboutPanel:)
              keyEquivalent:@""];
  [appMenu addItem:[NSMenuItem separatorItem]];
  [appMenu addItemWithTitle:@"Hide VTK Metal" action:@selector(hide:) keyEquivalent:@"h"];
  [appMenu addItem:[NSMenuItem separatorItem]];
  [appMenu addItemWithTitle:@"Quit VTK Metal" action:@selector(terminate:) keyEquivalent:@"q"];
  [appMenuItem setSubmenu:appMenu];

  // VTK top-level menu
  NSMenuItem* vtkMenuItem = [[NSMenuItem alloc] initWithTitle:@"VTK"
                                                       action:nil
                                                keyEquivalent:@""];
  [menubar addItem:vtkMenuItem];
  NSMenu* vtkMenu = [[NSMenu alloc] initWithTitle:@"VTK"];
  [vtkMenuItem setSubmenu:vtkMenu];

  // Views submenu
  NSMenu* viewsMenu = [[NSMenu alloc] initWithTitle:@"Views"];
  [ViewCommandDefs() enumerateObjectsUsingBlock:^(NSDictionary* d, NSUInteger i, BOOL* stop) {
    SEL action = NSSelectorFromString(d[@"action"]);
    NSMenuItem* item = [[NSMenuItem alloc] initWithTitle:d[@"title"]
                                                  action:action
                                           keyEquivalent:[@(i + 1) stringValue]];
    item.target = self;
    item.keyEquivalentModifierMask = NSEventModifierFlagCommand;
    [viewsMenu addItem:item];
  }];
  [self addSubmenu:viewsMenu titled:@"Views" toMenu:vtkMenu];

  // Interaction Mode submenu
  NSMenu* interactionMenu = [[NSMenu alloc] initWithTitle:@"Interaction Mode"];
  [self addModeItem:@"Pan" key:@"s" mode:VTKInteractionModePan action:@selector(activatePanMode:) toMenu:interactionMenu];
  [self addModeItem:@"Zoom" key:@"a" mode:VTKInteractionModeZoom action:@selector(activateZoomMode:) toMenu:interactionMenu];
  [self addModeItem:@"Trackball" key:@"x" mode:VTKInteractionModeTrackball action:@selector(activateTrackballMode:) toMenu:interactionMenu];
  [self addModeItem:@"Scroll Slices" key:@"d" mode:VTKInteractionModeScrollSlices action:@selector(activateScrollSlicesMode:) toMenu:interactionMenu];
  [self addModeItem:@"Window/Level" key:@"w" mode:VTKInteractionModeWindowLevel action:@selector(activateWindowLevelMode:) toMenu:interactionMenu];
  [self addSubmenu:interactionMenu titled:@"Interaction Mode" toMenu:vtkMenu];

  // Camera submenu
  NSMenu* cameraMenu = [[NSMenu alloc] initWithTitle:@"Camera"];
  [self addItem:@"Reset Camera"
         action:@selector(resetCamera:)
           key:@"r"
     modifiers:NSEventModifierFlagCommand
         target:self
         toMenu:cameraMenu];
  [self addSubmenu:cameraMenu titled:@"Camera" toMenu:vtkMenu];

  // Rendering submenu
  NSMenu* renderingMenu = [[NSMenu alloc] initWithTitle:@"Rendering"];
  [self addItem:@"Next VR Preset"
         action:@selector(nextPreset:)
           key:@"k"
     modifiers:NSEventModifierFlagControl | NSEventModifierFlagOption
         target:self
         toMenu:renderingMenu];
  [self addItem:@"Previous VR Preset"
         action:@selector(previousPreset:)
           key:@"j"
     modifiers:NSEventModifierFlagControl | NSEventModifierFlagOption
         target:self
         toMenu:renderingMenu];
  [self addItem:@"Toggle Benchmark"
         action:@selector(toggleBenchmark:)
           key:@"b"
     modifiers:NSEventModifierFlagCommand | NSEventModifierFlagOption
         target:self
         toMenu:renderingMenu];
  [self addItem:@"Toggle Dynamic Sample Rate"
         action:@selector(toggleDynamicSampleRate:)
           key:@"y"
     modifiers:NSEventModifierFlagCommand | NSEventModifierFlagOption
         target:self
         toMenu:renderingMenu];
  [self addItem:@"Increase Sample Distance"
         action:@selector(increaseSampleDistance:)
           key:@"+"
     modifiers:NSEventModifierFlagControl | NSEventModifierFlagOption
         target:self
         toMenu:renderingMenu];
  [self addItem:@"Decrease Sample Distance"
         action:@selector(decreaseSampleDistance:)
           key:@"-"
     modifiers:NSEventModifierFlagControl | NSEventModifierFlagOption
         target:self
         toMenu:renderingMenu];
  [renderingMenu addItem:[NSMenuItem separatorItem]];
  [self addItem:@"Volume Transpose (x<->z)"
         action:@selector(toggleVolumeTranspose:)
           key:@"t"
     modifiers:NSEventModifierFlagCommand | NSEventModifierFlagOption
         target:self
         toMenu:renderingMenu];
  [self addItem:@"Transposed Depth Axis = Y (vs X)"
         action:@selector(toggleVolumeTransposeAxisY:)
           key:@"z"
     modifiers:NSEventModifierFlagCommand | NSEventModifierFlagOption
         target:self
         toMenu:renderingMenu];
  [self addItem:@"Transpose Y-Tie (512x512 X==Y -> Y)"
         action:@selector(toggleVolumeTransposeYTie:)
           key:@"y"
     modifiers:NSEventModifierFlagCommand | NSEventModifierFlagShift
         target:self
         toMenu:renderingMenu];
  [self addItem:@"MinMax Acceleration"
         action:@selector(toggleMinMaxAcceleration:)
           key:@"m"
     modifiers:NSEventModifierFlagCommand | NSEventModifierFlagOption
         target:self
         toMenu:renderingMenu];
  [self addItem:@"Blue-Noise Jitter (non-IGN)"
         action:@selector(toggleBlueNoiseJitter:)
           key:@"n"
     modifiers:NSEventModifierFlagCommand | NSEventModifierFlagOption
         target:self
         toMenu:renderingMenu];
  [self addItem:@"March Variant 9 (48-wide Scheduled)"
         action:@selector(toggleMarchVariant9:)
           key:@"v"
      modifiers:NSEventModifierFlagCommand | NSEventModifierFlagOption
          target:self
          toMenu:renderingMenu];
  [self addItem:@"MinMax Block Summary (fine SD)"
         action:@selector(toggleMinMaxBlocks:)
           key:@"l"
      modifiers:NSEventModifierFlagCommand | NSEventModifierFlagOption
          target:self
          toMenu:renderingMenu];
  [renderingMenu addItem:[NSMenuItem separatorItem]];
  // Fragment Batch (§39) - compile-time register diet. 0=shipped runtime 32
  // heavy, 8/16/32=light. 16 is orbit-best on IMRToraceAddome SD4.
  NSMenu* fragBatchMenu = [[NSMenu alloc] initWithTitle:@"Fragment Batch"];
  [self addFragBatchItem:@"Shipped (32 heavy) [0]" batch:0 key:@"0" toMenu:fragBatchMenu];
  [self addFragBatchItem:@"8  (light, scatter-best)" batch:8 key:@"8" toMenu:fragBatchMenu];
  [self addFragBatchItem:@"16 (light, orbit-best)" batch:16 key:@"9" toMenu:fragBatchMenu];
  [self addFragBatchItem:@"32 (light)" batch:32 key:@"7" toMenu:fragBatchMenu];
  [self addSubmenu:fragBatchMenu titled:@"Fragment Batch" toMenu:renderingMenu];
  [self addSubmenu:renderingMenu titled:@"Rendering" toMenu:vtkMenu];

  // File submenu
  NSMenu* fileMenu = [[NSMenu alloc] initWithTitle:@"File"];
  [self addItem:@"Load File…"
         action:@selector(loadFile:)
           key:@"o"
     modifiers:NSEventModifierFlagCommand
         target:self
         toMenu:fileMenu];
  [self addSubmenu:fileMenu titled:@"File" toMenu:vtkMenu];

  [NSApp setMainMenu:menubar];
}

- (void)addItem:(NSString*)title
         action:(SEL)action
           key:(NSString*)key
     modifiers:(NSEventModifierFlags)modifiers
         target:(id)target
         toMenu:(NSMenu*)menu
{
  NSMenuItem* item = [[NSMenuItem alloc] initWithTitle:title action:action keyEquivalent:key];
  item.target = target;
  item.keyEquivalentModifierMask = modifiers;
  [menu addItem:item];
}

- (void)addModeItem:(NSString*)title
                key:(NSString*)key
               mode:(VTKInteractionMode)mode
             action:(SEL)action
             toMenu:(NSMenu*)menu
{
  NSMenuItem* item = [[NSMenuItem alloc] initWithTitle:title action:action keyEquivalent:key];
  item.target = self;
  item.keyEquivalentModifierMask = 0;
  item.tag = mode;
  [menu addItem:item];
}

- (void)addSubmenu:(NSMenu*)submenu titled:(NSString*)title toMenu:(NSMenu*)menu
{
  NSMenuItem* item = [[NSMenuItem alloc] initWithTitle:title action:nil keyEquivalent:@""];
  item.submenu = submenu;
  [menu addItem:item];
}

- (void)addFragBatchItem:(NSString*)title batch:(int)batch key:(NSString*)key toMenu:(NSMenu*)menu
{
  NSMenuItem* item = [[NSMenuItem alloc] initWithTitle:title action:@selector(setFragBatch:) keyEquivalent:key];
  item.target = self;
  item.keyEquivalentModifierMask = NSEventModifierFlagCommand | NSEventModifierFlagShift;
  item.tag = batch;
  [menu addItem:item];
}

#pragma mark - Menu Validation (macOS)

- (BOOL)validateMenuItem:(NSMenuItem*)menuItem
{
  SEL action = menuItem.action;

  NSUInteger index = [ViewCommandDefs() indexOfObjectPassingTest:^BOOL(NSDictionary* d, NSUInteger idx, BOOL* stop) {
    return NSSelectorFromString(d[@"action"]) == action;
  }];
  if (index != NSNotFound)
  {
    ViewController* rootVC = (ViewController*)self.window.contentViewController;
    if ([rootVC isKindOfClass:[ViewController class]])
    {
      menuItem.state = (rootVC.selectedIndex == (NSInteger)index)
        ? NSControlStateValueOn
        : NSControlStateValueOff;
    }
    return YES;
  }

  VTKInteractionMode cmdMode = (VTKInteractionMode)menuItem.tag;
  if (action == @selector(activatePanMode:) ||
      action == @selector(activateZoomMode:) ||
      action == @selector(activateTrackballMode:) ||
      action == @selector(activateScrollSlicesMode:) ||
      action == @selector(activateWindowLevelMode:))
  {
    VTKMetalBaseViewController* vc = [self findMetalViewController];
    menuItem.state = (vc && vc.interactionMode == cmdMode)
      ? NSControlStateValueOn
      : NSControlStateValueOff;

    if (action == @selector(activateScrollSlicesMode:) ||
        action == @selector(activateWindowLevelMode:))
    {
      BOOL isDICOMorNIFTI = [vc isKindOfClass:[DICOMVolumeViewController class]] ||
                            [vc isKindOfClass:[NIFTIVolumeViewController class]];
      if (!isDICOMorNIFTI)
      {
        menuItem.enabled = NO;
        return NO;
      }
    }
    return YES;
  }

  if (action == @selector(toggleDynamicSampleRate:))
  {
    VTKMetalBaseViewController* vc = [self findMetalViewController];
    if ([vc isKindOfClass:[BaseVolumeViewController class]])
    {
      BaseVolumeViewController* bvc = (BaseVolumeViewController*)vc;
      menuItem.state = bvc.isDynamicSampleRateAdjustmentEnabled
        ? NSControlStateValueOn
        : NSControlStateValueOff;
      return YES;
    }
    menuItem.enabled = NO;
    return NO;
  }

  if (action == @selector(toggleBenchmark:))
  {
    VTKMetalBaseViewController* vc = [self findMetalViewController];
    menuItem.state = (vc && vc.isBenchmarkRunning) ? NSControlStateValueOn : NSControlStateValueOff;
    return (vc != nil);
  }

  if (action == @selector(nextPreset:) || action == @selector(previousPreset:))
  {
    return [[self findMetalViewController] isKindOfClass:[FileVolumeViewController class]];
  }

  if (action == @selector(increaseSampleDistance:) ||
      action == @selector(decreaseSampleDistance:))
  {
    return [[self findMetalViewController] isKindOfClass:[BaseVolumeViewController class]];
  }

  if (action == @selector(loadFile:))
  {
    return [[self findMetalViewController] isKindOfClass:[FileVolumeViewController class]];
  }

  if (action == @selector(toggleVolumeTranspose:))
  {
    menuItem.state = self.volumeTransposeEnabled ? NSControlStateValueOn : NSControlStateValueOff;
    return YES;
  }

  if (action == @selector(toggleVolumeTransposeAxisY:))
  {
    menuItem.state = self.volumeTransposeAxisYEnabled ? NSControlStateValueOn : NSControlStateValueOff;
    return YES;
  }

  if (action == @selector(toggleVolumeTransposeYTie:))
  {
    menuItem.state = self.volumeTransposeYTieEnabled ? NSControlStateValueOn : NSControlStateValueOff;
    return YES;
  }

  if (action == @selector(toggleMinMaxAcceleration:) ||
      action == @selector(toggleBlueNoiseJitter:))
  {
    BOOL isVolumeVC = [[self findMetalViewController] isKindOfClass:[BaseVolumeViewController class]];
    menuItem.enabled = isVolumeVC;
    if (!isVolumeVC)
      return NO;
    if (action == @selector(toggleMinMaxAcceleration:))
      menuItem.state = self.minMaxAccelerationEnabled ? NSControlStateValueOn : NSControlStateValueOff;
    else
      menuItem.state = self.blueNoiseJitterEnabled ? NSControlStateValueOn : NSControlStateValueOff;
    return YES;
  }

  if (action == @selector(toggleMarchVariant9:))
  {
    BOOL isVolumeVC = [[self findMetalViewController] isKindOfClass:[BaseVolumeViewController class]];
    menuItem.enabled = isVolumeVC;
    if (!isVolumeVC)
      return NO;
    menuItem.state = self.marchVariant9Enabled ? NSControlStateValueOn : NSControlStateValueOff;
    return YES;
  }

  if (action == @selector(toggleMinMaxBlocks:))
  {
    BOOL isVolumeVC = [[self findMetalViewController] isKindOfClass:[BaseVolumeViewController class]];
    menuItem.enabled = isVolumeVC;
    if (!isVolumeVC)
      return NO;
    menuItem.state = self.minMaxBlocksEnabled ? NSControlStateValueOn : NSControlStateValueOff;
    return YES;
  }

  if (action == @selector(setFragBatch:))
  {
    BOOL isVolumeVC = [[self findMetalViewController] isKindOfClass:[BaseVolumeViewController class]];
    menuItem.enabled = isVolumeVC;
    if (!isVolumeVC)
      return NO;
    menuItem.state = (menuItem.tag == self.fragBatchValue) ? NSControlStateValueOn : NSControlStateValueOff;
    return YES;
  }

  return YES;
}

- (void)nextPreset:(id)sender
{
  FileVolumeViewController* vc = (FileVolumeViewController*)[self findMetalViewController];
  if ([vc isKindOfClass:[FileVolumeViewController class]])
  {
    [vc nextPreset:sender];
  }
}

- (void)previousPreset:(id)sender
{
  FileVolumeViewController* vc = (FileVolumeViewController*)[self findMetalViewController];
  if ([vc isKindOfClass:[FileVolumeViewController class]])
  {
    [vc previousPreset:sender];
  }
}

- (void)increaseSampleDistance:(id)sender
{
  BaseVolumeViewController* vc = (BaseVolumeViewController*)[self findMetalViewController];
  if ([vc isKindOfClass:[BaseVolumeViewController class]])
  {
    [vc increaseSampleDistance:sender];
  }
}

- (void)decreaseSampleDistance:(id)sender
{
  BaseVolumeViewController* vc = (BaseVolumeViewController*)[self findMetalViewController];
  if ([vc isKindOfClass:[BaseVolumeViewController class]])
  {
    [vc decreaseSampleDistance:sender];
  }
}

- (void)loadFile:(id)sender
{
  FileVolumeViewController* vc = (FileVolumeViewController*)[self findMetalViewController];
  if ([vc isKindOfClass:[FileVolumeViewController class]])
  {
    [vc loadFile:sender];
  }
}

#pragma mark - Render Config Toggles

- (vtkMetalGPUVolumeRayCastMapper*)currentVolumeMapper
{
  VTKMetalBaseViewController* vc = [self findMetalViewController];
  if ([vc isKindOfClass:[BaseVolumeViewController class]])
  {
    return ((BaseVolumeViewController*)vc).mapper;
  }
  return nil;
}

- (void)renderCurrentWindow
{
  VTKMetalBaseViewController* vc = [self findMetalViewController];
  if (vc.renderWindow)
  {
    static_cast<vtkRenderWindow*>(vc.renderWindow)->Render();
  }
}

// Transpose changes the uploaded texture layout and the shader fetch path, so
// the volume must re-upload; ForceResourceReupload does that in place, keeping
// the loaded dataset (unlike recreating the view controller, which would drop
// it). Both env knobs flip together: GPU transpose is just the fast repack for
// the transposed representation.
- (void)toggleVolumeTranspose:(id)sender
{
  self.volumeTransposeEnabled = !self.volumeTransposeEnabled;
  setenv("VTK_METAL_TEST_VOLTRANSPOSE", self.volumeTransposeEnabled ? "1" : "0", 1);
  setenv("VTK_METAL_TEST_GPU_TRANSPOSE", self.volumeTransposeEnabled ? "1" : "0", 1);
  if (vtkMetalGPUVolumeRayCastMapper* mapper = [self currentVolumeMapper])
  {
    mapper->ForceResourceReupload();
    [self renderCurrentWindow];
  }
}

// Depth-axis flip changes which physical 3D texture layout gets uploaded
// (same data, swapped extents), so like Volume Transpose it needs a resource
// re-upload. The mapper reads VTK_METAL_TEST_VOLTRANSPOSE_AXIS live inside
// VolumeTransposedAxisDepth at each upload; OFF restores x — matching the
// argmin-dims policy's tie-break on Z-long studies. Inert while transpose is
// off (the axis code is only consulted for the transposed representation).
- (void)toggleVolumeTransposeAxisY:(id)sender
{
  self.volumeTransposeAxisYEnabled = !self.volumeTransposeAxisYEnabled;
  setenv("VTK_METAL_TEST_VOLTRANSPOSE_AXIS",
         self.volumeTransposeAxisYEnabled ? "y" : "x", 1);
  if (vtkMetalGPUVolumeRayCastMapper* mapper = [self currentVolumeMapper])
  {
    mapper->ForceResourceReupload();
    [self renderCurrentWindow];
  }
}

- (void)toggleVolumeTransposeYTie:(id)sender
{
  self.volumeTransposeYTieEnabled = !self.volumeTransposeYTieEnabled;
  setenv("VTK_METAL_TEST_VOLTRANSPOSE_Y_TIE", self.volumeTransposeYTieEnabled ? "1" : "0", 1);
  if (vtkMetalGPUVolumeRayCastMapper* mapper = [self currentVolumeMapper])
  {
    mapper->ForceResourceReupload();
    [self renderCurrentWindow];
  }
}

- (void)toggleMinMaxAcceleration:(id)sender
{
  self.minMaxAccelerationEnabled = !self.minMaxAccelerationEnabled;
  if (vtkMetalGPUVolumeRayCastMapper* mapper = [self currentVolumeMapper])
  {
    // Both switches together: UseGPUMinMax selects the compute-kernel lattice
    // builder, UseMinMaxAcceleration gates empty-space skipping in the march.
    // Re-upload too — the lattice cell size differs between the GPU and CPU
    // builders, and a stale/nil min-max texture would desync the shader.
    mapper->SetUseGPUMinMax(self.minMaxAccelerationEnabled);
    mapper->SetUseMinMaxAcceleration(self.minMaxAccelerationEnabled);
    mapper->ForceResourceReupload();
    [self renderCurrentWindow];
  }
}

// Pure shader-constant flip (IGN vs blue-noise tile); the pipeline rebuilds
// from the changed feature mask on the next frame.
- (void)toggleBlueNoiseJitter:(id)sender
{
  self.blueNoiseJitterEnabled = !self.blueNoiseJitterEnabled;
  if (vtkMetalGPUVolumeRayCastMapper* mapper = [self currentVolumeMapper])
  {
    // NO => blue-noise tile; YES => Interleaved Gradient Noise.
    mapper->SetUseIGNJitter(!self.blueNoiseJitterEnabled);
    [self renderCurrentWindow];
  }
}

// The march variant rides the feature mask (bits 24-27), so flipping the env
// rebuilds the specialized pipeline on the next frame — no resource re-upload
// needed. The mapper reads VTK_METAL_TEST_MARCH_VARIANT live (per feature-mask
// build), matching the transpose toggle's setenv pattern; toggling off
// restores whatever the process was launched with.
- (void)toggleMarchVariant9:(id)sender
{
  self.marchVariant9Enabled = !self.marchVariant9Enabled;
  const char* value = self.marchVariant9Enabled
    ? "9"
    : self.marchVariantEnvAtLaunch.UTF8String;
  setenv("VTK_METAL_TEST_MARCH_VARIANT", value, 1);
  [self renderCurrentWindow];
}

// Two-level occupancy summary (HARNESS_VS_APP_GAP §34): the mapper re-reads
// VTK_METAL_TEST_MM_BLOCKS per lattice build and pipeline specialization, and
// the lattice cache key carries the block wish — so flipping the env rebuilds
// (or drops) the summary texture on the next frame and respecializes the
// march pipelines via their featureMaskExtra bit. No resource re-upload is
// needed. Since 2026-08-23 VolumeMinMaxBlocksWanted() no longer gates to
// fine-SD (sampleDistance <1.5) — it returns true whenever gpuMinMax is on
// (sampleDistance kept for call-site stability) mm:1403; re-add
// if(sampleDistance>=1.5) return false; to restore fine-only and avoid the
// 1024 SD4 dense +10% 6.52->7.23 §22.
- (void)toggleMinMaxBlocks:(id)sender
{
  self.minMaxBlocksEnabled = !self.minMaxBlocksEnabled;
  setenv("VTK_METAL_TEST_MM_BLOCKS", self.minMaxBlocksEnabled ? "1" : "0", 1);
  [self renderCurrentWindow];
}

// Fragment Batch (§39): compile-time register diet. Baked into
// featureMaskExtra bits [10:15] (fragBatch<<10) -> own PSO per width.
// Mirrors fc_cmBatch trick for compute (HARNESS_VS_APP_GAP §38.10).
- (void)setFragBatch:(id)sender
{
  NSInteger batch = 0;
  if ([sender respondsToSelector:@selector(tag)])
    batch = [sender tag];
  else if ([sender isKindOfClass:[NSNumber class]])
    batch = [sender integerValue];
  self.fragBatchValue = (int)batch;
  char buf[16];
  snprintf(buf, sizeof(buf), "%d", self.fragBatchValue);
  setenv("VTK_METAL_TEST_FRAG_BATCH", buf, 1);
  [self renderCurrentWindow];
}

#else

#pragma mark - Application Lifecycle (iOS)

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
  if (action == @selector(setFragBatch:))
  {
    // Enable only on volume view controllers; state handled in validateCommand for UIMenu.
    VTKMetalBaseViewController* vc = [self findMetalViewController];
    return [vc isKindOfClass:[BaseVolumeViewController class]];
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
    return;
  }

  if (cmdAction == @selector(setFragBatch:))
  {
    VTKMetalBaseViewController* vc = [self findMetalViewController];
    BOOL isVolumeVC = [vc isKindOfClass:[BaseVolumeViewController class]];
    if (!isVolumeVC)
    {
      command.attributes = UIMenuElementAttributesDisabled;
      return;
    }
    NSInteger batch = [command.propertyList integerValue];
    command.state = (batch == self.fragBatchValue) ? UIMenuElementStateOn : UIMenuElementStateOff;
  }
}

- (BOOL)application:(UIApplication*)application
didFinishLaunchingWithOptions:(NSDictionary*)launchOptions
{
  self.fragBatchValue = 0;
  if (const char* v = std::getenv("VTK_METAL_TEST_FRAG_BATCH"))
    self.fragBatchValue = std::max(0, std::min(48, std::atoi(v)));
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

  // Fragment Batch (§39) - compile-time register diet submenu
  UICommand* frag0Cmd = [UICommand commandWithTitle:@"Fragment Batch: Shipped (32 heavy) [0]"
                                              image:nil
                                             action:@selector(setFragBatch:)
                                       propertyList:@(0)];
  frag0Cmd.discoverabilityTitle = @"Shipped runtime 32";
  UICommand* frag8Cmd = [UICommand commandWithTitle:@"Fragment Batch: 8 (light, scatter-best)"
                                              image:nil
                                             action:@selector(setFragBatch:)
                                       propertyList:@(8)];
  frag8Cmd.discoverabilityTitle = @"Fragment batch 8";
  UICommand* frag16Cmd = [UICommand commandWithTitle:@"Fragment Batch: 16 (light, orbit-best)"
                                               image:nil
                                              action:@selector(setFragBatch:)
                                        propertyList:@(16)];
  frag16Cmd.discoverabilityTitle = @"Fragment batch 16";
  UICommand* frag32Cmd = [UICommand commandWithTitle:@"Fragment Batch: 32 (light)"
                                               image:nil
                                              action:@selector(setFragBatch:)
                                        propertyList:@(32)];
  frag32Cmd.discoverabilityTitle = @"Fragment batch 32";
  UIMenu* fragBatchMenu = [UIMenu menuWithTitle:@"Fragment Batch"
                                           children:@[ frag0Cmd, frag8Cmd, frag16Cmd, frag32Cmd ]];

  UIMenu* renderingMenu = [UIMenu
                           menuWithTitle:@"Rendering"
                           children:@[ nextPresetCmd, prevPresetCmd, benchmarkCmd, dynSampleCmd, incSampleCmd, decSampleCmd, fragBatchMenu ]];

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

- (void)setFragBatch:(id)sender
{
  NSInteger batch = 0;
  if ([sender respondsToSelector:@selector(tag)])
    batch = [sender tag];
  else if ([sender isKindOfClass:[NSNumber class]])
    batch = [sender integerValue];
  // UIMenu case: propertyList holds the batch value
  if ([sender isKindOfClass:[UICommand class]])
  {
    UICommand* cmd = (UICommand*)sender;
    if (cmd.propertyList)
      batch = [cmd.propertyList integerValue];
  }
  self.fragBatchValue = (int)batch;
  char buf[16];
  snprintf(buf, sizeof(buf), "%d", self.fragBatchValue);
  setenv("VTK_METAL_TEST_FRAG_BATCH", buf, 1);
  [self renderCurrentWindow];
}

#endif

#pragma mark - View Switching

- (ViewController*)viewControllerForSwitch
{
#if TARGET_OS_OSX
  ViewController* rootVC = (ViewController*)self.window.contentViewController;
#else
  ViewController* rootVC = (ViewController*)self.window.rootViewController;
#endif
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
  ViewController* root = [self viewControllerForSwitch];
  if (!root)
  {
    return nil;
  }
  id current = root.currentViewController;
  
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
    [self updateInteractionModeMenu];
  }
}

- (void)updateInteractionModeMenu
{
#if TARGET_OS_OSX
  [[NSApp mainMenu] update];
#else
  [UIMenuSystem.mainSystem setNeedsRebuild];
#endif
}

@end
