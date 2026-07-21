#import "DICOMVolumeViewController.h"
#import "VolumeRenderingPreset.h"
#import "VolumeRenderingPresetsManager.h"

#include "vtkNew.h"
#include "vtkColorTransferFunction.h"
#include "vtkPiecewiseFunction.h"
#include "vtkDICOMImageReader.h"
#include "vtkImageData.h"
#include "vtkImageShiftScale.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkMetalGPUVolumeRayCastMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkIOSMetalRenderWindow.h"

#include <vtkSmartPointer.h>

@interface DICOMVolumeViewController ()
@property (nonatomic, assign) double dataMin;
@property (nonatomic, assign) double dataRange;
@property (nonatomic, assign) BOOL dataLoaded;
@end

@implementation DICOMVolumeViewController

- (NSString *)loadButtonTitle
{
  return @"Load DICOM";
}

- (NSArray<NSString *> *)documentTypes
{
  return @[ @"public.folder" ];
}

- (void)loadFromURL:(NSURL *)url
{
  NSString *path = url.path;
  vtkMetalRenderer *renderer = static_cast<vtkMetalRenderer *>([self renderer]);

  vtkNew<vtkDICOMImageReader> reader;
  reader->SetDirectoryName([path UTF8String]);
  reader->Update();

  // Cast to unsigned char for better performance on iOS.
  // Rescale data to [0, 255] using the scalar range from the reader.
  double scalarRange[2];
  reader->GetOutput()->GetScalarRange(scalarRange);
  self.dataMin = scalarRange[0];
  double dataMax = scalarRange[1];
  self.dataRange = dataMax - self.dataMin;
  if (self.dataRange == 0.0) {
    self.dataRange = 1.0;
  }

  vtkNew<vtkImageShiftScale> castToU8;
  castToU8->SetInputConnection(reader->GetOutputPort());
  castToU8->SetShift(-self.dataMin);
  castToU8->SetScale(255.0 / self.dataRange);
  castToU8->SetOutputScalarTypeToUnsignedChar();
  castToU8->ClampOverflowOn();
  castToU8->Update();
  // Original 16-bit data is no longer needed — free it.
  reader->GetOutput()->ReleaseData();

  vtkNew<vtkMetalGPUVolumeRayCastMapper> mapper;
  mapper->SetInputData(castToU8->GetOutput());
  mapper->UseJitteringOn();
  mapper->AutoAdjustSampleDistancesOff();
  mapper->SetSampleDistance(0.5);

  vtkNew<vtkVolumeProperty> property;

  vtkNew<vtkVolume> volume;
  volume->SetMapper(mapper);
  volume->SetProperty(property);

  renderer->AddVolume(volume);

  self.dataLoaded = YES;

  [self applyCurrentPreset];

  renderer->ResetCamera();
  static_cast<vtkIOSMetalRenderWindow *>([self renderWindow])->Render();
}

- (void)applyCurrentPreset
{
  if (!self.dataLoaded) return;

  vtkMetalRenderer *renderer = static_cast<vtkMetalRenderer *>([self renderer]);
  vtkVolume *volume = renderer->GetVolumes()->GetNextVolume();
  if (!volume) return;

  vtkVolumeProperty *property = volume->GetProperty();

  auto rescale = [&](double hu) -> double {
    return (hu - self.dataMin) / self.dataRange * 255.0;
  };

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
          opacityFunc->AddPoint(rescale(otf[j].x), otf[j].y);
          colorFunc->AddRGBPoint(rescale(otf[j].x), ctf[j].red, ctf[j].green, ctf[j].blue);
        }
      }
    }
  } else {
    // Fallback: simple linear preset
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

- (IBAction)nextPreset:(id)sender
{
  [super nextPreset:sender];
  [self applyCurrentPreset];
}

- (IBAction)previousPreset:(id)sender
{
  [super previousPreset:sender];
  [self applyCurrentPreset];
}

@end
