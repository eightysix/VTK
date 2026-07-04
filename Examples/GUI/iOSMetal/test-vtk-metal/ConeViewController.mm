#import "ConeViewController.h"

#include "vtkNew.h"
#include "vtkProperty.h"
#include "vtkConeSource.h"
#include "vtkMetalActor.h"
#include "vtkMetalPolyDataMapper.h"
#include "vtkMetalRenderer.h"

@implementation ConeViewController

- (void)setupVTKPipeline
{
  vtkMetalRenderer* renderer = static_cast<vtkMetalRenderer*>([self renderer]);

  vtkNew<vtkConeSource> cone;
  cone->SetResolution(128);
  vtkNew<vtkMetalPolyDataMapper> mapper;
  mapper->SetInputConnection(cone->GetOutputPort());

  vtkNew<vtkMetalActor> actor;
  actor->SetMapper(mapper);
  actor->GetProperty()->SetColor(0.2, 0.6, 1.0);
  renderer->AddActor(actor);
}

@end
