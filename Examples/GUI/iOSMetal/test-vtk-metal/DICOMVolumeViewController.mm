#import "DICOMVolumeViewController.h"
#import "VolumeRenderingPreset.h"
#import "VolumeRenderingPresetsManager.h"

#include "vtkCamera.h"
#include "vtkColorTransferFunction.h"
#include "vtkDICOMDirectory.h"
#include "vtkDICOMReader.h"
#include "vtkImageData.h"
#include "vtkImageShiftScale.h"
#include "vtkIOSMetalRenderWindow.h"
#include "vtkMath.h"
#include "vtkMetalGPUVolumeRayCastMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkNew.h"
#include "vtkPiecewiseFunction.h"
#include "vtkPlane.h"
#include "vtkRenderer.h"
#include "vtkStringArray.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"

@interface VTKMetalBaseViewController (ClipPlaneSupp)
- (void)handlePan:(UIPanGestureRecognizer*)recognizer;
@end

@interface DICOMVolumeViewController ()
{
  vtkSmartPointer<vtkPlane> _clipPlane;
  vtkSmartPointer<vtkImageData> _volumeData;
  int _clipAxis;
  BOOL _clipPlaneHasUserPos;

  vtkSmartPointer<vtkColorTransferFunction> _savedCTF;
  vtkSmartPointer<vtkPiecewiseFunction> _savedOTF;
  double _windowWidth;
  double _windowLevel;
}
@end

@implementation DICOMVolumeViewController

- (NSArray<NSString *> *)documentTypes
{
  return @[ @"public.folder" ];
}

- (void)loadFromURL:(NSURL *)url
{
  NSString *path = url.path;
  vtkMetalRenderer *renderer = static_cast<vtkMetalRenderer *>([self renderer]);

  vtkNew<vtkDICOMDirectory> dicomDir;
  dicomDir->SetDirectoryName([path UTF8String]);
  dicomDir->Update();

  int numSeries = dicomDir->GetNumberOfSeries();
  if (numSeries == 0) {
    NSLog(@"No DICOM series found in %s", [path UTF8String]);
    return;
  }

  vtkNew<vtkDICOMReader> reader;
  reader->SetFileNames(dicomDir->GetFileNamesForSeries(0));
  reader->Update();

  vtkNew<vtkImageShiftScale> castToU8;
  castToU8->SetInputConnection(reader->GetOutputPort());
  castToU8->SetShift(1024.0);
  castToU8->SetScale(255.0 / 4095.0);
  castToU8->SetOutputScalarTypeToUnsignedChar();
  castToU8->ClampOverflowOn();
  castToU8->Update();
  reader->GetOutput()->ReleaseData();

  vtkNew<vtkMetalGPUVolumeRayCastMapper> mapper;
  mapper->SetInputData(castToU8->GetOutput());
  mapper->UseJitteringOn();
  mapper->AutoAdjustSampleDistancesOff();
  mapper->SetSampleDistance(0.5);
  mapper->SetUseGPUMinMax(true);

  _volumeData = castToU8->GetOutput();
  _clipPlane = vtkSmartPointer<vtkPlane>::New();
  _clipPlane->SetNormal(0, 0, 1);
  mapper->AddClippingPlane(_clipPlane);
  _clipPlaneHasUserPos = NO;
  _clipAxis = 2;

  vtkNew<vtkVolumeProperty> property;

  vtkNew<vtkVolume> volume;
  volume->SetMapper(mapper);
  volume->SetProperty(property);

  self.mapper = mapper;
  self.property = property;
  self.volume = volume;

  renderer->AddVolume(volume);

  [self applyCurrentPreset];

  renderer->ResetCamera();
  static_cast<vtkIOSMetalRenderWindow *>([self renderWindow])->Render();
}

- (double)rescale:(double)hu
{
  return (hu + 1024.0) * (255.0 / 4095.0);
}

#pragma mark - Preset Override

- (void)applyCurrentPreset
{
  [super applyCurrentPreset];
  [self saveOriginalTransferFunctions];
}

#pragma mark - Window / Level

- (void)saveOriginalTransferFunctions
{
  vtkVolumeProperty* prop = self.volume->GetProperty();
  vtkColorTransferFunction* ctf = prop->GetRGBTransferFunction();
  vtkPiecewiseFunction* otf = prop->GetScalarOpacity();
  _savedCTF = vtkSmartPointer<vtkColorTransferFunction>::New();
  _savedCTF->DeepCopy(ctf);
  _savedOTF = vtkSmartPointer<vtkPiecewiseFunction>::New();
  _savedOTF->DeepCopy(otf);
  _windowWidth = 255.0;
  _windowLevel = 127.5;
}

- (void)applyWindowLevelWithWidth:(double)width level:(double)level
{
  _windowWidth = width;
  _windowLevel = level;
  vtkVolumeProperty* prop = self.volume->GetProperty();

  vtkNew<vtkColorTransferFunction> newCTF;
  for (int i = 0; i < _savedCTF->GetSize(); i++)
  {
    double node[6];
    _savedCTF->GetNodeValue(i, node);
    double newX = _windowLevel + (node[0] - 127.5) * _windowWidth / 255.0;
    newCTF->AddRGBPoint(newX, node[1], node[2], node[3], node[4], node[5]);
  }

  vtkNew<vtkPiecewiseFunction> newOTF;
  for (int i = 0; i < _savedOTF->GetSize(); i++)
  {
    double node[4];
    _savedOTF->GetNodeValue(i, node);
    double newX = _windowLevel + (node[0] - 127.5) * _windowWidth / 255.0;
    newOTF->AddPoint(newX, node[1], node[2], node[3]);
  }

  prop->SetColor(newCTF);
  prop->SetScalarOpacity(newOTF);

  static_cast<vtkIOSMetalRenderWindow*>([self renderWindow])->Render();
}

- (void)handleWindowLevelPan:(UIPanGestureRecognizer*)recognizer
{
  if (recognizer.state != UIGestureRecognizerStateChanged) return;

  const double kScale = 0.5;
  CGPoint t = [recognizer translationInView:recognizer.view];
  double dW = t.x * kScale;
  double dL = -t.y * kScale;
  [self applyWindowLevelWithWidth:_windowWidth + dW level:_windowLevel + dL];
  [recognizer setTranslation:CGPointZero inView:recognizer.view];
}

#pragma mark - Clipping Plane (Scroll Slices)

- (void)handlePan:(UIPanGestureRecognizer*)recognizer
{
  if (self.interactionMode == VTKInteractionModeScrollSlices)
  {
    [self handleScrollSlicesPan:recognizer];
  }
  else if (self.interactionMode == VTKInteractionModeWindowLevel)
  {
    [self handleWindowLevelPan:recognizer];
  }
  else
  {
    [super handlePan:recognizer];
  }
}

- (void)handleScrollSlicesPan:(UIPanGestureRecognizer*)recognizer
{
  if (!_clipPlane || !_volumeData) return;

  vtkRenderer* ren = static_cast<vtkMetalRenderer*>([self renderer]);
  if (!ren) return;
  vtkCamera* cam = ren->GetActiveCamera();
  if (!cam) return;

  switch (recognizer.state)
  {
    case UIGestureRecognizerStateBegan:
      [self prepareVRClippingPlaneWithCamera:cam];
      break;
    case UIGestureRecognizerStateChanged:
    {
      CGPoint translation = [recognizer translationInView:recognizer.view];
      [self translateVRClippingPlane:translation.y];
      [recognizer setTranslation:CGPointZero inView:recognizer.view];
      break;
    }
    default:
      break;
  }

  static_cast<vtkIOSMetalRenderWindow*>([self renderWindow])->Render();
}

- (void)prepareVRClippingPlaneWithCamera:(vtkCamera*)cam
{
  double fp[3], pos[3], dop[3];
  cam->GetFocalPoint(fp);
  cam->GetPosition(pos);
  dop[0] = fp[0] - pos[0];
  dop[1] = fp[1] - pos[1];
  dop[2] = fp[2] - pos[2];
  vtkMath::Normalize(dop);

  int newAxis = 0;
  double maxAbs = fabs(dop[0]);
  if (fabs(dop[1]) > maxAbs) { newAxis = 1; maxAbs = fabs(dop[1]); }
  if (fabs(dop[2]) > maxAbs) { newAxis = 2; }

  if (_clipPlaneHasUserPos && newAxis == _clipAxis)
  {
    double curN[3];
    _clipPlane->GetNormal(curN);
    if (curN[newAxis] * dop[newAxis] > 0.0) return;
  }

  _clipAxis = newAxis;

  double n[3] = {0, 0, 0};
  n[_clipAxis] = (dop[_clipAxis] >= 0.0) ? 1.0 : -1.0;
  _clipPlane->SetNormal(n);

  double b[6];
  _volumeData->GetBounds(b);
  double ori[3] = {(b[0] + b[1]) * 0.5, (b[2] + b[3]) * 0.5, (b[4] + b[5]) * 0.5};
  ori[_clipAxis] = (n[_clipAxis] > 0.0) ? b[2 * _clipAxis] : b[2 * _clipAxis + 1];

  _clipPlane->SetOrigin(ori);
  _clipPlaneHasUserPos = NO;
}

- (void)translateVRClippingPlane:(double)pixelDy
{
  if (!_clipPlane || !_volumeData) return;

  double ori[3];
  _clipPlane->GetOrigin(ori);

  double spc[3];
  _volumeData->GetSpacing(spc);

  const double kSpeed = 1.0;
  ori[_clipAxis] += kSpeed * pixelDy * spc[_clipAxis];

  double b[6];
  _volumeData->GetBounds(b);
  int ax = _clipAxis;
  if (ori[ax] < b[2 * ax]) ori[ax] = b[2 * ax];
  if (ori[ax] > b[2 * ax + 1]) ori[ax] = b[2 * ax + 1];

  _clipPlane->SetOrigin(ori);
  _clipPlaneHasUserPos = YES;
}

@end
