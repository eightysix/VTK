#import "FileVolumeViewController.h"
#import "VolumeRenderingPreset.h"
#import "VolumeRenderingPresetsManager.h"

#include "vtkNew.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkColorTransferFunction.h"
#include "vtkPiecewiseFunction.h"
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

#pragma mark - Preset Application

- (void)applyCurrentPreset
{
  if (!self.volume) return;

  vtkVolumeProperty *property = self.volume->GetProperty();
  if (!property) return;

  vtkNew<vtkColorTransferFunction> colorFunc;
  vtkNew<vtkPiecewiseFunction> opacityFunc;

  VolumeRenderingPreset *preset = self.currentPreset;
  if (preset &&
      preset.colorTransferFunctions.count == preset.opacityTransferFunctions.count) {
    for (NSUInteger i = 0; i < preset.opacityTransferFunctions.count; i++) {
      NSArray<OpacityTransferFunctionMember *> *otf = preset.opacityTransferFunctions[i];
      NSArray<ColorTransferFunctionMember *> *ctf = preset.colorTransferFunctions[i];
      if (otf.count == ctf.count) {
        for (NSUInteger j = 0; j < otf.count; j++) {
          opacityFunc->AddPoint([self rescale:otf[j].x], otf[j].y);
          colorFunc->AddRGBPoint([self rescale:otf[j].x], ctf[j].red, ctf[j].green, ctf[j].blue);
        }
      }
    }
  } else {
    colorFunc->AddRGBPoint(0.0, 0.0, 0.0, 0.0);
    colorFunc->AddRGBPoint(255.0, 1.0, 1.0, 1.0);
    opacityFunc->AddPoint(0.0, 0.0);
    opacityFunc->AddPoint(255.0, 1.0);
  }

  property->SetColor(colorFunc);
  property->SetScalarOpacity(opacityFunc);
  property->SetInterpolationTypeToLinear();

  static_cast<vtkIOSMetalRenderWindow *>([self renderWindow])->Render();
}

#pragma mark - Preset Cycling

- (IBAction)nextPreset:(id)sender
{
  NSArray<VolumeRenderingPreset *> *presets = VolumeRenderingPresetsManager.presets;
  if (presets.count == 0) return;

  self.currentPresetIndex = (self.currentPresetIndex + 1) % presets.count;
  self.currentPreset = presets[self.currentPresetIndex];

  NSLog(@"Preset: %@", self.currentPreset.name);
  [self applyCurrentPreset];
}

- (IBAction)previousPreset:(id)sender
{
  NSArray<VolumeRenderingPreset *> *presets = VolumeRenderingPresetsManager.presets;
  if (presets.count == 0) return;

  self.currentPresetIndex =
      (self.currentPresetIndex - 1 + presets.count) % presets.count;
  self.currentPreset = presets[self.currentPresetIndex];

  NSLog(@"Preset: %@", self.currentPreset.name);
  [self applyCurrentPreset];
}

@end
