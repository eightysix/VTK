#import "CubeViewController.h"

#include "vtkNew.h"
#include "vtkProperty.h"
#include "vtkCubeSource.h"
#include "vtkWebGPUActor.h"
#include "vtkWebGPUPolyDataMapper.h"
#include "vtkWebGPURenderer.h"

@implementation CubeViewController

- (void)setupVTKPipeline {
  vtkWebGPURenderer *renderer = static_cast<vtkWebGPURenderer *>([self renderer]);

  vtkNew<vtkCubeSource> cube;
  cube->SetXLength(1.5);
  cube->SetYLength(1.5);
  cube->SetZLength(1.5);
  vtkNew<vtkWebGPUPolyDataMapper> mapper;
  mapper->SetInputConnection(cube->GetOutputPort());

  vtkNew<vtkWebGPUActor> actor;
  actor->SetMapper(mapper);
  actor->GetProperty()->SetColor(1.0, 0.4, 0.2);
  renderer->AddActor(actor);
}

@end
