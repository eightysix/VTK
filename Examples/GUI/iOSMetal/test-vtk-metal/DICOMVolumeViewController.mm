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

@implementation DICOMVolumeViewController

- (NSArray<NSString *> *)documentTypes
{
  return @[ @"public.folder" ];
}

- (void)loadFromURL:(NSURL *)url
{
  NSString *path = url.path;
  vtkMetalRenderer *renderer = static_cast<vtkMetalRenderer *>([self renderer]);

  vtkNew<vtkDICOMDirectory> dicomDir;
  dicomDir->SetDirectoryName([path UTF8String]);
  dicomDir->Update();

  int numSeries = dicomDir->GetNumberOfSeries();
  if (numSeries == 0) {
    NSLog(@"No DICOM series found in %s", [path UTF8String]);
    return;
  }

  vtkNew<vtkDICOMReader> reader;
  reader->SetFileNames(dicomDir->GetFileNamesForSeries(0));
  reader->Update();

  vtkNew<vtkImageShiftScale> castToU8;
  castToU8->SetInputConnection(reader->GetOutputPort());
  castToU8->SetShift(1024.0);
  castToU8->SetScale(255.0 / 4095.0);
  castToU8->SetOutputScalarTypeToUnsignedChar();
  castToU8->ClampOverflowOn();
  castToU8->Update();
  reader->GetOutput()->ReleaseData();

  vtkNew<vtkMetalGPUVolumeRayCastMapper> mapper;
  mapper->SetInputData(castToU8->GetOutput());
  mapper->UseJitteringOn();
  mapper->AutoAdjustSampleDistancesOff();
  mapper->SetSampleDistance(0.5);
  mapper->SetUseGPUMinMax(true);

  vtkNew<vtkVolumeProperty> property;

  vtkNew<vtkVolume> volume;
  volume->SetMapper(mapper);
  volume->SetProperty(property);

  self.mapper = mapper;
  self.property = property;
  self.volume = volume;

  renderer->AddVolume(volume);

  [self applyCurrentPreset];

  renderer->ResetCamera();
  static_cast<vtkIOSMetalRenderWindow *>([self renderWindow])->Render();
}

- (double)rescale:(double)hu
{
  return (hu + 1024.0) * (255.0 / 4095.0);
}

@end
