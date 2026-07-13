#import "DICOMVolumeViewController.h"

#include "vtkNew.h"
#include "vtkColorTransferFunction.h"
#include "vtkPiecewiseFunction.h"
#include "vtkDICOMImageReader.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkMetalGPUVolumeRayCastMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkIOSMetalRenderWindow.h"

@interface DICOMVolumeViewController ()
@property (nonatomic, strong) UIButton* loadButton;
@property (nonatomic, strong) UILabel* sampleDistanceLabel;
@property (nonatomic, strong) UIStepper* sampleDistanceStepper;
@property (nonatomic, strong) UILabel* titleLabel;
@property (nonatomic, assign) float currentSampleDistance;
@property (nonatomic, assign) BOOL hasVolume;
@end

@implementation DICOMVolumeViewController

- (void)viewDidLoad {
  [super viewDidLoad];
  [self setupUI];
}

- (void)setupUI {
  // Create load button
  self.loadButton = [UIButton buttonWithType:UIButtonTypeSystem];
  [self.loadButton setTitle:@"Load DICOM" forState:UIControlStateNormal];
  self.loadButton.titleLabel.font = [UIFont boldSystemFontOfSize:18];
  self.loadButton.backgroundColor = [[UIColor blackColor] colorWithAlphaComponent:0.5];
  self.loadButton.layer.cornerRadius = 8;
  self.loadButton.contentEdgeInsets = UIEdgeInsetsMake(12, 24, 12, 24);
  [self.loadButton setTitleColor:[UIColor whiteColor] forState:UIControlStateNormal];
  [self.loadButton addTarget:self
                      action:@selector(loadDICOMFolder)
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

- (void)loadDICOMFolder {
  UIDocumentPickerViewController *picker =
      [[UIDocumentPickerViewController alloc] initWithDocumentTypes:@[@"public.folder"]
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

  vtkNew<vtkDICOMImageReader> reader;
  reader->SetDirectoryName([path UTF8String]);
  reader->Update();

  vtkNew<vtkMetalGPUVolumeRayCastMapper> mapper;
  mapper->SetInputConnection(reader->GetOutputPort());
  mapper->UseJitteringOn();
  mapper->SetAutoAdjustSampleDistances(0);  // Disable auto-adjust
  mapper->SetSampleDistance(self.currentSampleDistance);

  // Bone + Skin II preset (16-bit CLUT)
  vtkNew<vtkColorTransferFunction> colorFunc;
  // Curve 1: cyan-ish tones for soft tissue
  colorFunc->AddRGBPoint(-713.84, 0.072, 0.994, 1.0);
  colorFunc->AddRGBPoint(-653.98, 0.072, 0.994, 1.0);
  colorFunc->AddRGBPoint(-640.25, 0.072, 0.994, 1.0);
  colorFunc->AddRGBPoint(-590.33, 0.072, 0.994, 1.0);
  colorFunc->AddRGBPoint(-544.65, 0.072, 0.994, 1.0);
  // Curve 2: red -> pink -> white for bone
  colorFunc->AddRGBPoint(66.73, 0.0, 0.0, 0.0);
  colorFunc->AddRGBPoint(84.34, 1.0, 0.0, 0.0);
  colorFunc->AddRGBPoint(366.83, 1.0, 0.999, 1.0);
  colorFunc->AddRGBPoint(1585.43, 1.0, 1.0, 1.0);

  vtkNew<vtkPiecewiseFunction> opacityFunc;
  // Curve 1
  opacityFunc->AddPoint(-713.84, 0.0);
  opacityFunc->AddPoint(-653.98, 0.209);
  opacityFunc->AddPoint(-640.25, 0.29);
  opacityFunc->AddPoint(-590.33, 0.209);
  opacityFunc->AddPoint(-544.65, 0.209);
  // Curve 2
  opacityFunc->AddPoint(66.73, 0.0);
  opacityFunc->AddPoint(84.34, 0.189);
  opacityFunc->AddPoint(366.83, 0.645);
  opacityFunc->AddPoint(1585.43, 0.789);

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
