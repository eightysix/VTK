#import "DICOMVolumeViewController.h"
#import "VolumeRenderingPreset.h"
#import "VolumeRenderingPresetsManager.h"

#include "vtkNew.h"
#include "vtkColorTransferFunction.h"
#include "vtkPiecewiseFunction.h"
#include "vtkDICOMDirectory.h"
#include "vtkDICOMReader.h"
#include "vtkStringArray.h"
#include "vtkImageData.h"
#include "vtkImageShiftScale.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkMetalGPUVolumeRayCastMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkIOSMetalRenderWindow.h"

#include <vtkSmartPointer.h>

// Fixed CT HU→u8 mapping matching Eyesight-iOS VTK_HU2U8.
// Maps the standard CT range [-1024, 3071] to [0, 255].
static const double kHUshift = 1024.0;
static const double kHUscale = 255.0 / 4095.0;

static inline double HU2U8(double hu) {
  return (hu + kHUshift) * kHUscale;
}

@interface DICOMVolumeViewController ()
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

  // Scan the directory for DICOM series.
  vtkNew<vtkDICOMDirectory> dicomDir;
  dicomDir->SetDirectoryName([path UTF8String]);
  dicomDir->Update();

  int numSeries = dicomDir->GetNumberOfSeries();
  if (numSeries == 0) {
    NSLog(@"No DICOM series found in %s", [path UTF8String]);
    return;
  }

  // Read the first series.
  vtkNew<vtkDICOMReader> reader;
  reader->SetFileNames(dicomDir->GetFileNamesForSeries(0));
  reader->Update();

  // Cast to float using fixed CT HU range mapping, preserving full
  // dynamic range in a 0–255 space suitable for the preset transfer
  // functions.  The mapper will convert these floats to half-float
  // (R16Float / RGBA16Float) when PreferHalfPrecision is on, cutting
  // texture bandwidth by ~2× vs. R32Float.
  vtkNew<vtkImageShiftScale> castToFloat;
  castToFloat->SetInputConnection(reader->GetOutputPort());
  castToFloat->SetShift(kHUshift);
  castToFloat->SetScale(kHUscale);
  castToFloat->SetOutputScalarTypeToFloat();
  castToFloat->ClampOverflowOn();
  castToFloat->Update();
  // Original 16-bit data is no longer needed — free it.
  reader->GetOutput()->ReleaseData();

  vtkNew<vtkMetalGPUVolumeRayCastMapper> mapper;
  mapper->SetInputData(castToFloat->GetOutput());
  mapper->UseJitteringOn();
  mapper->AutoAdjustSampleDistancesOff();
  mapper->SetSampleDistance(0.5);
  mapper->SetUseGPUMinMax(true);
  mapper->SetPreferHalfPrecision(true);

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
          opacityFunc->AddPoint(HU2U8(otf[j].x), otf[j].y);
          colorFunc->AddRGBPoint(HU2U8(otf[j].x), ctf[j].red, ctf[j].green, ctf[j].blue);
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
