#import "VolumeViewController.h"

#include "vtkNew.h"
#include "vtkColorTransferFunction.h"
#include "vtkPiecewiseFunction.h"
#include "vtkRTAnalyticSource.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkMetalGPUVolumeRayCastMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkIOSMetalRenderWindow.h"

@interface VolumeViewController ()
@property (nonatomic, strong) UILabel* sampleDistanceLabel;
@property (nonatomic, strong) UIStepper* sampleDistanceStepper;
@property (nonatomic, assign) float currentSampleDistance;
@end

@implementation VolumeViewController

- (void)viewDidLoad {
  [super viewDidLoad];
  [self setupUI];
}

- (void)setupUI {
  // Create label for sample distance
  self.sampleDistanceLabel = [[UILabel alloc] init];
  self.sampleDistanceLabel.textColor = [UIColor whiteColor];
  self.sampleDistanceLabel.backgroundColor = [[UIColor blackColor] colorWithAlphaComponent:0.7];
  self.sampleDistanceLabel.textAlignment = NSTextAlignmentCenter;
  self.sampleDistanceLabel.layer.cornerRadius = 8;
  self.sampleDistanceLabel.clipsToBounds = YES;
  self.sampleDistanceLabel.translatesAutoresizingMaskIntoConstraints = NO;
  [self.view addSubview:self.sampleDistanceLabel];

  // Create stepper for sample distance
  self.sampleDistanceStepper = [[UIStepper alloc] init];
  self.sampleDistanceStepper.minimumValue = 0.1;
  self.sampleDistanceStepper.maximumValue = 5.0;
  self.sampleDistanceStepper.stepValue = 0.1;
  self.sampleDistanceStepper.value = 1.0;  // Will be updated after VTK pipeline setup
  self.sampleDistanceStepper.tintColor = [UIColor whiteColor];
  [self.sampleDistanceStepper addTarget:self
                                action:@selector(sampleDistanceChanged:)
                      forControlEvents:UIControlEventValueChanged];
  self.sampleDistanceStepper.translatesAutoresizingMaskIntoConstraints = NO;
  [self.view addSubview:self.sampleDistanceStepper];

  // Create label for title
  UILabel* titleLabel = [[UILabel alloc] init];
  titleLabel.text = @"Sample Distance:";
  titleLabel.textColor = [UIColor whiteColor];
  titleLabel.backgroundColor = [[UIColor blackColor] colorWithAlphaComponent:0.7];
  titleLabel.textAlignment = NSTextAlignmentCenter;
  titleLabel.layer.cornerRadius = 8;
  titleLabel.clipsToBounds = YES;
  titleLabel.translatesAutoresizingMaskIntoConstraints = NO;
  [self.view addSubview:titleLabel];

  // Layout constraints
  [NSLayoutConstraint activateConstraints:@[
    [titleLabel.bottomAnchor
        constraintEqualToAnchor:self.sampleDistanceStepper.topAnchor
                       constant:-10],
    [titleLabel.centerXAnchor constraintEqualToAnchor:self.view.centerXAnchor],
    [titleLabel.widthAnchor constraintEqualToConstant:150],
    [titleLabel.heightAnchor constraintEqualToConstant:30],

    [self.sampleDistanceStepper.centerXAnchor
        constraintEqualToAnchor:self.view.centerXAnchor],
    [self.sampleDistanceStepper.bottomAnchor
        constraintEqualToAnchor:self.view.safeAreaLayoutGuide.bottomAnchor
                       constant:-20],

    [self.sampleDistanceLabel.topAnchor
        constraintEqualToAnchor:self.sampleDistanceStepper.bottomAnchor
                       constant:10],
    [self.sampleDistanceLabel.centerXAnchor constraintEqualToAnchor:self.view.centerXAnchor],
    [self.sampleDistanceLabel.widthAnchor constraintEqualToConstant:100],
    [self.sampleDistanceLabel.heightAnchor constraintEqualToConstant:30]
  ]];

  [self updateSampleDistanceLabel];
}

- (void)setupVTKPipeline {
  vtkMetalRenderer* renderer = static_cast<vtkMetalRenderer*>([self renderer]);

  vtkNew<vtkRTAnalyticSource> source;
  source->SetWholeExtent(0, 49, 0, 49, 0, 49);
  source->SetCenter(25.0, 25.0, 25.0);
  source->SetXFreq(6.0);
  source->SetYFreq(3.0);
  source->SetZFreq(4.0);
  source->Update();

  vtkNew<vtkMetalGPUVolumeRayCastMapper> mapper;
  mapper->SetInputConnection(source->GetOutputPort());
  mapper->SetAutoAdjustSampleDistances(0);  // Disable auto-adjust
  mapper->SetSampleDistance(1.0);  // Set initial sample distance

  // Store mapper for later use
  self.currentSampleDistance = 1.0;
  self.sampleDistanceStepper.value = self.currentSampleDistance;
  [self updateSampleDistanceLabel];

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

- (void)sampleDistanceChanged:(UIStepper*)sender {
  self.currentSampleDistance = sender.value;
  [self updateSampleDistanceLabel];
  [self updateMapperSampleDistance];
}

- (void)updateSampleDistanceLabel {
  self.sampleDistanceLabel.text =
      [NSString stringWithFormat:@"%.1f", self.currentSampleDistance];
}

- (void)updateMapperSampleDistance {
  vtkMetalRenderer* renderer = static_cast<vtkMetalRenderer*>([self renderer]);

  // Get the volume from the renderer
  vtkVolumeCollection* volumeCollection = renderer->GetVolumes();
  volumeCollection->InitTraversal();
  vtkVolume* volume = volumeCollection->GetNextVolume();

  if (volume) {
    vtkMetalGPUVolumeRayCastMapper* mapper =
        static_cast<vtkMetalGPUVolumeRayCastMapper*>(volume->GetMapper());
    if (mapper) {
      mapper->SetSampleDistance(self.currentSampleDistance);
      static_cast<vtkIOSMetalRenderWindow*>([self renderWindow])->Render();
    }
  }
}

@end
