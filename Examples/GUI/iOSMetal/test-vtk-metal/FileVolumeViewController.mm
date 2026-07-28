#import "FileVolumeViewController.h"
#import "VolumeRenderingPreset.h"
#import "VolumeRenderingPresetsManager.h"

#include "vtkVolume.h"
#include "vtkMetalRenderer.h"
#include "vtkIOSMetalRenderWindow.h"

@interface FileVolumeViewController ()
@property (nonatomic, strong, readwrite) VolumeRenderingPreset *currentPreset;
@property (nonatomic, assign, readwrite) NSInteger currentPresetIndex;
@end

@implementation FileVolumeViewController

- (NSArray<NSString *> *)documentTypes
{
  return @[ @"public.data" ];
}

- (void)loadFromURL:(NSURL *)url
{
  // Subclasses must override
}

- (void)viewDidLoad
{
  [super viewDidLoad];

  // Start with the first preset
  self.currentPresetIndex = 0;
  NSArray<VolumeRenderingPreset *> *presets = VolumeRenderingPresetsManager.presets;
  if (presets.count > 0) {
    self.currentPreset = presets[0];
  }
}

- (void)loadFile:(id)sender
{
  UIDocumentPickerViewController *picker =
      [[UIDocumentPickerViewController alloc] initWithDocumentTypes:[self documentTypes]
                                                              inMode:UIDocumentPickerModeOpen];
  picker.delegate = self;
  picker.allowsMultipleSelection = NO;
  [self presentViewController:picker animated:YES completion:nil];
}

- (void)documentPicker:(UIDocumentPickerViewController *)controller
    didPickDocumentsAtURLs:(NSArray<NSURL *> *)urls
{
  if (urls.count == 0) {
    return;
  }

  NSURL *url = urls.firstObject;
  BOOL coordinated = [url startAccessingSecurityScopedResource];

  [self loadFromURL:url];

  if (coordinated) {
    [url stopAccessingSecurityScopedResource];
  }
}

- (void)documentPickerWasCancelled:(UIDocumentPickerViewController *)controller
{
}

- (void)setupVTKPipeline
{
}

#pragma mark - Preset Cycling

- (IBAction)nextPreset:(id)sender
{
  NSArray<VolumeRenderingPreset *> *presets = VolumeRenderingPresetsManager.presets;
  if (presets.count == 0) return;

  self.currentPresetIndex = (self.currentPresetIndex + 1) % presets.count;
  self.currentPreset = presets[self.currentPresetIndex];

  NSLog(@"Preset: %@", self.currentPreset.name);
}

- (IBAction)previousPreset:(id)sender
{
  NSArray<VolumeRenderingPreset *> *presets = VolumeRenderingPresetsManager.presets;
  if (presets.count == 0) return;

  self.currentPresetIndex =
      (self.currentPresetIndex - 1 + presets.count) % presets.count;
  self.currentPreset = presets[self.currentPresetIndex];

  NSLog(@"Preset: %@", self.currentPreset.name);
}

@end
