#import "NIFTIVolumeViewController.h"

#include "vtkNew.h"
#include "vtkColorTransferFunction.h"
#include "vtkPiecewiseFunction.h"
#include "vtkNIFTIImageReader.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkWebGPUGPUVolumeRayCastMapper.h"
#include "vtkWebGPURenderer.h"
#include "vtkWebGPURenderWindow.h"

@implementation NIFTIVolumeViewController

- (void)viewDidLoad {
  [super viewDidLoad];

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

  vtkWebGPURenderer *renderer = static_cast<vtkWebGPURenderer *>([self renderer]);

  vtkNew<vtkNIFTIImageReader> reader;
  reader->SetFileName([path UTF8String]);
  reader->Update();

  vtkNew<vtkWebGPUGPUVolumeRayCastMapper> mapper;
  mapper->SetInputConnection(reader->GetOutputPort());
  mapper->UseJitteringOn();
  mapper->SetSampleDistance(0.5);

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
  static_cast<vtkWebGPURenderWindow *>([self renderWindow])->Render();

  if (coordinated) {
    [url stopAccessingSecurityScopedResource];
  }
}

- (void)documentPickerWasCancelled:(UIDocumentPickerViewController *)controller {
}

- (void)setupVTKPipeline {
}

@end
