#import "NIFTIVolumeViewController.h"
#import "VolumeRenderingPreset.h"
#import "VolumeRenderingPresetsManager.h"

#include "vtkNew.h"
#include "vtkColorTransferFunction.h"
#include "vtkPiecewiseFunction.h"
#include "vtkNIFTIImageReader.h"
#include "vtkImageData.h"
#include "vtkImageShiftScale.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkMetalGPUVolumeRayCastMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkIOSMetalRenderWindow.h"

@interface NIFTIVolumeViewController ()
@property (nonatomic, assign) double dataMin;
@property (nonatomic, assign) double dataRange;
@end

@implementation NIFTIVolumeViewController

- (void)loadFromURL:(NSURL *)url
{
  NSString *path = url.path;
  vtkMetalRenderer *renderer = static_cast<vtkMetalRenderer *>([self renderer]);

  vtkNew<vtkNIFTIImageReader> reader;
  reader->SetFileName([path UTF8String]);
  reader->Update();

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
  reader->GetOutput()->ReleaseData();

  vtkNew<vtkMetalGPUVolumeRayCastMapper> mapper;
  mapper->SetInputData(castToU8->GetOutput());
  mapper->UseJitteringOn();
  mapper->AutoAdjustSampleDistancesOff();
  mapper->SetSampleDistance(0.5);
  mapper->SetPartitions(1, 1, 4);
  mapper->SetDisableInstanceRendering(true);

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
  return (hu - self.dataMin) / self.dataRange * 255.0;
}

@end
