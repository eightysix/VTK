#import "VTKMetalBaseViewController.h"

class vtkVolume;
class vtkMetalGPUVolumeRayCastMapper;
class vtkVolumeProperty;

@interface BaseVolumeViewController : VTKMetalBaseViewController

@property (nonatomic, assign) vtkVolume *volume;
@property (nonatomic, assign) vtkMetalGPUVolumeRayCastMapper *mapper;
@property (nonatomic, assign) vtkVolumeProperty *property;

- (double)rescale:(double)hu;

- (IBAction)increaseSampleDistance:(nullable id)sender;
- (IBAction)decreaseSampleDistance:(nullable id)sender;

@end
