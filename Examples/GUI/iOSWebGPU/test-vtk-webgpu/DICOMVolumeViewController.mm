#import "DICOMVolumeViewController.h"

#include "vtkNew.h"
#include "vtkColorTransferFunction.h"
#include "vtkPiecewiseFunction.h"
#include "vtkDICOMImageReader.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkWebGPUGPUVolumeRayCastMapper.h"
#include "vtkWebGPURenderer.h"
#include "vtkWebGPURenderWindow.h"

@implementation DICOMVolumeViewController

- (void)viewDidLoad {
  [super viewDidLoad];

  UIButton *button = [UIButton buttonWithType:UIButtonTypeSystem];
  [button setTitle:@"Load DICOM" forState:UIControlStateNormal];
  button.titleLabel.font = [UIFont boldSystemFontOfSize:18];
  button.backgroundColor = [[UIColor blackColor] colorWithAlphaComponent:0.5];
  button.layer.cornerRadius = 8;
  button.contentEdgeInsets = UIEdgeInsetsMake(12, 24, 12, 24);
  [button setTitleColor:[UIColor whiteColor] forState:UIControlStateNormal];
  [button addTarget:self action:@selector(loadDICOMFolder) forControlEvents:UIControlEventTouchUpInside];
  button.translatesAutoresizingMaskIntoConstraints = NO;
  [self.view addSubview:button];

  [NSLayoutConstraint activateConstraints:@[
    [button.centerXAnchor constraintEqualToAnchor:self.view.centerXAnchor],
    [button.bottomAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.bottomAnchor constant:-20]
  ]];
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

  vtkWebGPURenderer *renderer = static_cast<vtkWebGPURenderer *>([self renderer]);

  vtkNew<vtkDICOMImageReader> reader;
  reader->SetDirectoryName([path UTF8String]);
  reader->Update();

  vtkNew<vtkWebGPUGPUVolumeRayCastMapper> mapper;
  mapper->SetInputConnection(reader->GetOutputPort());

  vtkNew<vtkColorTransferFunction> colorFunc;
  colorFunc->AddRGBPoint(-1024.0, 0.0, 0.0, 0.0);
  colorFunc->AddRGBPoint(150.0, 0.0, 0.0, 0.0);
  colorFunc->AddRGBPoint(200.0, 0.8, 0.6, 0.5);
  colorFunc->AddRGBPoint(600.0, 0.95, 0.85, 0.75);
  colorFunc->AddRGBPoint(3000.0, 1.0, 1.0, 1.0);

  vtkNew<vtkPiecewiseFunction> opacityFunc;
  opacityFunc->AddPoint(-1024.0, 0.0);
  opacityFunc->AddPoint(150.0, 0.0);
  opacityFunc->AddPoint(200.0, 0.15);
  opacityFunc->AddPoint(400.0, 0.5);
  opacityFunc->AddPoint(3000.0, 0.85);

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
