#import "BaseVolumeViewController.h"

#include "vtkSmartPointer.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkMetalGPUVolumeRayCastMapper.h"

@interface BaseVolumeViewController ()
{
    vtkSmartPointer<vtkVolume> _volume;
    vtkSmartPointer<vtkMetalGPUVolumeRayCastMapper> _mapper;
    vtkSmartPointer<vtkVolumeProperty> _property;
}
@end

@implementation BaseVolumeViewController

- (vtkVolume *)volume { return _volume; }
- (vtkMetalGPUVolumeRayCastMapper *)mapper { return _mapper; }
- (vtkVolumeProperty *)property { return _property; }

- (void)setVolume:(vtkVolume *)vol { _volume = vol; }
- (void)setMapper:(vtkMetalGPUVolumeRayCastMapper *)map { _mapper = map; }
- (void)setProperty:(vtkVolumeProperty *)prop { _property = prop; }

- (double)rescale:(double)hu { return hu; }

@end
