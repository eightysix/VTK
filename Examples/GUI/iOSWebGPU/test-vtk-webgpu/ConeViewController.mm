#import "ConeViewController.h"

#include "vtkNew.h"
#include "vtkProperty.h"
#include "vtkConeSource.h"
#include "vtkWebGPUActor.h"
#include "vtkWebGPUPolyDataMapper.h"
#include "vtkWebGPURenderer.h"

@implementation ConeViewController

- (void)setupVTKPipeline {
  vtkWebGPURenderer *renderer = static_cast<vtkWebGPURenderer *>([self renderer]);

  vtkNew<vtkConeSource> cone;
  cone->SetResolution(128);
  vtkNew<vtkWebGPUPolyDataMapper> mapper;
  mapper->SetInputConnection(cone->GetOutputPort());

  vtkNew<vtkWebGPUActor> actor;
  actor->SetMapper(mapper);
  actor->GetProperty()->SetColor(0.2, 0.6, 1.0);
  renderer->AddActor(actor);
}

@end
