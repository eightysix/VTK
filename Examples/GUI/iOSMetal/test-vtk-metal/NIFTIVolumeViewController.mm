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

@implementation NIFTIVolumeViewController

- (void)viewDidLoad {
  [super viewDidLoad];

  // Enable Metal volume partition debugging (block tinting + sort-order log).
  // The mapper reads VTK_METAL_VOLUME_DEBUG via std::getenv, so set it in-process.
  setenv("VTK_METAL_VOLUME_DEBUG", "1", 1);

  UIButton *button = [UIButton buttonWithType:UIButtonTypeSystem];
  [button setTitle:@"Load NIfTI" forState:UIControlStateNormal];
  button.titleLabel.font = [UIFont boldSystemFontOfSize:18];
  button.backgroundColor = [[UIColor blackColor] colorWithAlphaComponent:0.5];
  button.layer.cornerRadius = 8;
  button.contentEdgeInsets = UIEdgeInsetsMake(12, 24, 12, 24);
  [button setTitleColor:[UIColor whiteColor] forState:UIControlStateNormal];
  [button addTarget:self action:@selector(loadNIFTIFile) forControlEvents:UIControlEventTouchUpInside];
  button.translatesAutoresizingMaskIntoConstraints = NO;
  [self.view addSubview:button];

  [NSLayoutConstraint activateConstraints:@[
    [button.centerXAnchor constraintEqualToAnchor:self.view.centerXAnchor],
    [button.bottomAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.bottomAnchor constant:-20]
  ]];
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
  mapper->AutoAdjustSampleDistancesOff();
  mapper->SetSampleDistance(0.5);
  mapper->SetPartitions(1, 1, 4);

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

  if (coordinated) {
    [url stopAccessingSecurityScopedResource];
  }
}

- (void)documentPickerWasCancelled:(UIDocumentPickerViewController *)controller {
}

- (void)setupVTKPipeline {
}

@end
