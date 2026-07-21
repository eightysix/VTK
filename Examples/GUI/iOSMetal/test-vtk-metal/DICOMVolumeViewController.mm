#import "DICOMVolumeViewController.h"

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
  double dataMin = scalarRange[0];
  double dataMax = scalarRange[1];
  double dataRange = dataMax - dataMin;
  if (dataRange == 0.0) {
    dataRange = 1.0;
  }

  vtkNew<vtkImageShiftScale> castToU8;
  castToU8->SetInputConnection(reader->GetOutputPort());
  castToU8->SetShift(-dataMin);
  castToU8->SetScale(255.0 / dataRange);
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

  // Bone + Skin II preset — remapped to u8 [0, 255] range.
  // Original CLUT was authored against HU range; scale x-values by
  //   u8 = (hu - dataMin) / dataRange * 255
  auto rescale = [&](double hu) -> double {
    return (hu - dataMin) / dataRange * 255.0;
  };

  vtkNew<vtkColorTransferFunction> colorFunc;
  // Curve 1: cyan-ish tones for soft tissue
  colorFunc->AddRGBPoint(rescale(-713.84), 0.072, 0.994, 1.0);
  colorFunc->AddRGBPoint(rescale(-653.98), 0.072, 0.994, 1.0);
  colorFunc->AddRGBPoint(rescale(-640.25), 0.072, 0.994, 1.0);
  colorFunc->AddRGBPoint(rescale(-590.33), 0.072, 0.994, 1.0);
  colorFunc->AddRGBPoint(rescale(-544.65), 0.072, 0.994, 1.0);
  // Curve 2: red -> pink -> white for bone
  colorFunc->AddRGBPoint(rescale(66.73), 0.0, 0.0, 0.0);
  colorFunc->AddRGBPoint(rescale(84.34), 1.0, 0.0, 0.0);
  colorFunc->AddRGBPoint(rescale(366.83), 1.0, 0.999, 1.0);
  colorFunc->AddRGBPoint(rescale(1585.43), 1.0, 1.0, 1.0);

  vtkNew<vtkPiecewiseFunction> opacityFunc;
  // Curve 1
  opacityFunc->AddPoint(rescale(-713.84), 0.0);
  opacityFunc->AddPoint(rescale(-653.98), 0.209);
  opacityFunc->AddPoint(rescale(-640.25), 0.29);
  opacityFunc->AddPoint(rescale(-590.33), 0.209);
  opacityFunc->AddPoint(rescale(-544.65), 0.209);
  // Curve 2
  opacityFunc->AddPoint(rescale(66.73), 0.0);
  opacityFunc->AddPoint(rescale(84.34), 0.189);
  opacityFunc->AddPoint(rescale(366.83), 0.645);
  opacityFunc->AddPoint(rescale(1585.43), 0.789);

  vtkNew<vtkVolumeProperty> property;
  property->SetColor(colorFunc);
  property->SetScalarOpacity(opacityFunc);
  property->SetInterpolationTypeToLinear();

  vtkNew<vtkVolume> volume;
  volume->SetMapper(mapper);
  volume->SetProperty(property);

  renderer->AddVolume(volume);
  renderer->ResetCamera();
  static_cast<vtkIOSMetalRenderWindow *>([self renderWindow])->Render();
}

@end
