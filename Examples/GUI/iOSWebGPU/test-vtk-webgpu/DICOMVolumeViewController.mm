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
  mapper->UseJitteringOff();
  mapper->SetSampleDistance(0.5);

  // Bone + Skin II preset (16-bit CLUT)
  vtkNew<vtkColorTransferFunction> colorFunc;
  // Curve 1: cyan-ish tones for soft tissue
  colorFunc->AddRGBPoint(-713.84, 0.072, 0.994, 1.0);
  colorFunc->AddRGBPoint(-653.98, 0.072, 0.994, 1.0);
  colorFunc->AddRGBPoint(-640.25, 0.072, 0.994, 1.0);
  colorFunc->AddRGBPoint(-590.33, 0.072, 0.994, 1.0);
  colorFunc->AddRGBPoint(-544.65, 0.072, 0.994, 1.0);
  // Curve 2: red → pink → white for bone
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
