#import "VolumeViewController.h"

#include "vtkNew.h"
#include "vtkColorTransferFunction.h"
#include "vtkPiecewiseFunction.h"
#include "vtkRTAnalyticSource.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkWebGPUGPUVolumeRayCastMapper.h"
#include "vtkWebGPURenderer.h"

@implementation VolumeViewController

- (void)setupVTKPipeline {
  vtkWebGPURenderer *renderer = static_cast<vtkWebGPURenderer *>([self renderer]);

  vtkNew<vtkRTAnalyticSource> source;
  source->SetWholeExtent(0, 49, 0, 49, 0, 49);
  source->SetCenter(25.0, 25.0, 25.0);
  source->SetXFreq(6.0);
  source->SetYFreq(3.0);
  source->SetZFreq(4.0);
  source->Update();

  vtkNew<vtkWebGPUGPUVolumeRayCastMapper> mapper;
  mapper->SetInputConnection(source->GetOutputPort());

  vtkNew<vtkColorTransferFunction> colorFunc;
  colorFunc->AddRGBPoint(0.0, 0.0, 0.0, 0.0);
  colorFunc->AddRGBPoint(64.0, 0.0, 0.0, 0.5);
  colorFunc->AddRGBPoint(128.0, 0.0, 0.6, 0.2);
  colorFunc->AddRGBPoint(192.0, 0.9, 0.3, 0.0);
  colorFunc->AddRGBPoint(255.0, 1.0, 1.0, 1.0);

  vtkNew<vtkPiecewiseFunction> opacityFunc;
  opacityFunc->AddPoint(0.0, 0.0);
  opacityFunc->AddPoint(50.0, 0.0);
  opacityFunc->AddPoint(70.0, 0.2);
  opacityFunc->AddPoint(120.0, 0.3);
  opacityFunc->AddPoint(200.0, 0.5);
  opacityFunc->AddPoint(255.0, 0.8);

  vtkNew<vtkVolumeProperty> property;
  property->SetColor(colorFunc);
  property->SetScalarOpacity(opacityFunc);
  property->SetInterpolationTypeToLinear();

  vtkNew<vtkVolume> volume;
  volume->SetMapper(mapper);
  volume->SetProperty(property);

  renderer->AddVolume(volume);
}

@end
