#import "BaseVolumeViewController.h"

#include "vtkSmartPointer.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkMetalGPUVolumeRayCastMapper.h"
#include "vtkRenderWindow.h"

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

- (IBAction)increaseSampleDistance:(id)sender
{
    _mapper->AutoAdjustSampleDistancesOff();
    float sd = _mapper->GetSampleDistance();
    float newSd;
    if (sd < 0.5f) {
        newSd = 0.5f;
    } else {
        newSd = (floorf(sd / 0.5f) + 1.0f) * 0.5f;
    }
    _mapper->SetSampleDistance(newSd);
    NSLog(@"Sample distance increased to %.1f", newSd);
    static_cast<vtkRenderWindow*>([self renderWindow])->Render();
}

- (IBAction)decreaseSampleDistance:(id)sender
{
    _mapper->AutoAdjustSampleDistancesOff();
    float sd = _mapper->GetSampleDistance();
    float newSd;
    if (sd <= 0.5f) {
        newSd = 0.1f;
    } else {
        newSd = (ceilf(sd / 0.5f) - 1.0f) * 0.5f;
    }
    _mapper->SetSampleDistance(newSd);
    NSLog(@"Sample distance decreased to %.1f", newSd);
    static_cast<vtkRenderWindow*>([self renderWindow])->Render();
}

@end
