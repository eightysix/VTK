#import "CubeViewController.h"

#include "vtkNew.h"
#include "vtkProperty.h"
#include "vtkCubeSource.h"
#include "vtkMetalActor.h"
#include "vtkMetalPolyDataMapper.h"
#include "vtkMetalRenderer.h"

@implementation CubeViewController

- (void)setupVTKPipeline
{
  vtkMetalRenderer* renderer = static_cast<vtkMetalRenderer*>([self renderer]);

  vtkNew<vtkCubeSource> cube;
  cube->SetXLength(1.5);
  cube->SetYLength(1.5);
  cube->SetZLength(1.5);
  vtkNew<vtkMetalPolyDataMapper> mapper;
  mapper->SetInputConnection(cube->GetOutputPort());

  vtkNew<vtkMetalActor> actor;
  actor->SetMapper(mapper);
  actor->GetProperty()->SetColor(1.0, 0.4, 0.2);
  renderer->AddActor(actor);
}

@end
