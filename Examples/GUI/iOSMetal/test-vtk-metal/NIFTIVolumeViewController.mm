#import "NIFTIVolumeViewController.h"

#include "vtkNew.h"
#include "vtkColorTransferFunction.h"
#include "vtkPiecewiseFunction.h"
#include "vtkNIFTIImageReader.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkMetalGPUVolumeRayCastMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkIOSMetalRenderWindow.h"

@interface NIFTIVolumeViewController ()
@property (nonatomic, strong) UIButton* loadButton;
@property (nonatomic, strong) UILabel* sampleDistanceLabel;
@property (nonatomic, strong) UIStepper* sampleDistanceStepper;
@property (nonatomic, strong) UILabel* titleLabel;
@property (nonatomic, assign) float currentSampleDistance;
@property (nonatomic, assign) BOOL hasVolume;
@end

@implementation NIFTIVolumeViewController

- (void)viewDidLoad {
  [super viewDidLoad];
  [self setupUI];
}

- (void)setupUI {
  // Create load button
  self.loadButton = [UIButton buttonWithType:UIButtonTypeSystem];
  [self.loadButton setTitle:@"Load NIfTI" forState:UIControlStateNormal];
  self.loadButton.titleLabel.font = [UIFont boldSystemFontOfSize:18];
  self.loadButton.backgroundColor = [[UIColor blackColor] colorWithAlphaComponent:0.5];
  self.loadButton.layer.cornerRadius = 8;
  self.loadButton.contentEdgeInsets = UIEdgeInsetsMake(12, 24, 12, 24);
  [self.loadButton setTitleColor:[UIColor whiteColor] forState:UIControlStateNormal];
  [self.loadButton addTarget:self
                      action:@selector(loadNIFTIFile)
            forControlEvents:UIControlEventTouchUpInside];
  self.loadButton.translatesAutoresizingMaskIntoConstraints = NO;
  [self.view addSubview:self.loadButton];

  // Create label for sample distance
  self.sampleDistanceLabel = [[UILabel alloc] init];
  self.sampleDistanceLabel.textColor = [UIColor whiteColor];
  self.sampleDistanceLabel.backgroundColor = [[UIColor blackColor] colorWithAlphaComponent:0.7];
  self.sampleDistanceLabel.textAlignment = NSTextAlignmentCenter;
  self.sampleDistanceLabel.layer.cornerRadius = 8;
  self.sampleDistanceLabel.clipsToBounds = YES;
  self.sampleDistanceLabel.translatesAutoresizingMaskIntoConstraints = NO;
  self.sampleDistanceLabel.hidden = YES;
  [self.view addSubview:self.sampleDistanceLabel];

  // Create stepper for sample distance
  self.sampleDistanceStepper = [[UIStepper alloc] init];
  self.sampleDistanceStepper.minimumValue = 0.1;
  self.sampleDistanceStepper.maximumValue = 5.0;
  self.sampleDistanceStepper.stepValue = 0.1;
  self.sampleDistanceStepper.value = 0.5;
  self.sampleDistanceStepper.tintColor = [UIColor whiteColor];
  [self.sampleDistanceStepper addTarget:self
                                action:@selector(sampleDistanceChanged:)
                      forControlEvents:UIControlEventValueChanged];
  self.sampleDistanceStepper.translatesAutoresizingMaskIntoConstraints = NO;
  self.sampleDistanceStepper.hidden = YES;
  [self.view addSubview:self.sampleDistanceStepper];

  // Create label for title
  self.titleLabel = [[UILabel alloc] init];
  self.titleLabel.text = @"Sample Distance:";
  self.titleLabel.textColor = [UIColor whiteColor];
  self.titleLabel.backgroundColor = [[UIColor blackColor] colorWithAlphaComponent:0.7];
  self.titleLabel.textAlignment = NSTextAlignmentCenter;
  self.titleLabel.layer.cornerRadius = 8;
  self.titleLabel.clipsToBounds = YES;
  self.titleLabel.translatesAutoresizingMaskIntoConstraints = NO;
  self.titleLabel.hidden = YES;
  [self.view addSubview:self.titleLabel];

  // Layout constraints
  [NSLayoutConstraint activateConstraints:@[
    [self.loadButton.centerXAnchor constraintEqualToAnchor:self.view.centerXAnchor],
    [self.loadButton.bottomAnchor
        constraintEqualToAnchor:self.view.safeAreaLayoutGuide.bottomAnchor
                       constant:-20],

    [self.titleLabel.bottomAnchor
        constraintEqualToAnchor:self.sampleDistanceStepper.topAnchor
                       constant:-10],
    [self.titleLabel.centerXAnchor constraintEqualToAnchor:self.view.centerXAnchor],
    [self.titleLabel.widthAnchor constraintEqualToConstant:150],
    [self.titleLabel.heightAnchor constraintEqualToConstant:30],

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

  self.currentSampleDistance = 0.5;
  [self updateSampleDistanceLabel];
}

- (void)loadNIFTIFile {
  UIDocumentPickerViewController *picker =
      [[UIDocumentPickerViewController alloc] initWithDocumentTypes:@[@"public.data"]
                                                              inMode:UIDocumentPickerModeOpen];
  picker.delegate = self;
  picker.allowsMultipleSelection = NO;
  [self presentViewController:picker animated:YES completion:nil];
}

- (void)documentPicker:(UIDocumentPickerViewController *)controller
    didPickDocumentsAtURLs:(NSArray<NSURL *> *)urls {
  if (urls.count == 0) {
    return;
  }

  NSURL *url = urls.firstObject;
  BOOL coordinated = [url startAccessingSecurityScopedResource];
  NSString *path = url.path;

  vtkMetalRenderer *renderer = static_cast<vtkMetalRenderer *>([self renderer]);

  vtkNew<vtkNIFTIImageReader> reader;
  reader->SetFileName([path UTF8String]);
  reader->Update();

  vtkNew<vtkMetalGPUVolumeRayCastMapper> mapper;
  mapper->SetInputConnection(reader->GetOutputPort());
  mapper->UseJitteringOn();
  mapper->SetAutoAdjustSampleDistances(0);  // Disable auto-adjust
  mapper->SetSampleDistance(self.currentSampleDistance);

  // Simple linear preset to verify volume renders at all
  vtkNew<vtkColorTransferFunction> colorFunc;
  colorFunc->AddRGBPoint(0, 0.0, 0.0, 0.0);
  colorFunc->AddRGBPoint(255, 1.0, 1.0, 1.0);

  vtkNew<vtkPiecewiseFunction> opacityFunc;
  opacityFunc->AddPoint(0, 0.0);
  opacityFunc->AddPoint(255, 1.0);

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

  // Show stepper controls after loading volume
  self.hasVolume = YES;
  self.loadButton.hidden = YES;
  self.titleLabel.hidden = NO;
  self.sampleDistanceStepper.hidden = NO;
  self.sampleDistanceLabel.hidden = NO;

  if (coordinated) {
    [url stopAccessingSecurityScopedResource];
  }
}

- (void)documentPickerWasCancelled:(UIDocumentPickerViewController *)controller {
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
  vtkMetalRenderer *renderer = static_cast<vtkMetalRenderer *>([self renderer]);

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

- (void)setupVTKPipeline {
}

@end
