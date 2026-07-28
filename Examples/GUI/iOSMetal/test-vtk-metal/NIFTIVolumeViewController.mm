#import "NIFTIVolumeViewController.h"
#import "VolumeRenderingPreset.h"
#import "VolumeRenderingPresetsManager.h"

#include "vtkCamera.h"
#include "vtkColorTransferFunction.h"
#include "vtkIOSMetalRenderWindow.h"
#include "vtkImageData.h"
#include "vtkImageShiftScale.h"
#include "vtkMath.h"
#include "vtkMetalGPUVolumeRayCastMapper.h"
#include "vtkMetalRenderer.h"
#include "vtkNew.h"
#include "vtkNIFTIImageReader.h"
#include "vtkPiecewiseFunction.h"
#include "vtkPlane.h"
#include "vtkRenderer.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"

@interface VTKMetalBaseViewController (ClipPlaneSupp)
- (void)handlePan:(UIPanGestureRecognizer*)recognizer;
@end

@interface NIFTIVolumeViewController ()
{
  vtkSmartPointer<vtkPlane> _clipPlane;
  vtkSmartPointer<vtkImageData> _volumeData;
  int _clipAxis;
  BOOL _clipPlaneHasUserPos;
}
@property (nonatomic, assign) double dataMin;
@property (nonatomic, assign) double dataRange;
@end

@implementation NIFTIVolumeViewController

- (void)loadFromURL:(NSURL *)url
{
  NSString *path = url.path;
  vtkMetalRenderer *renderer = static_cast<vtkMetalRenderer *>([self renderer]);

  vtkNew<vtkNIFTIImageReader> reader;
  reader->SetFileName([path UTF8String]);
  reader->Update();

  double scalarRange[2];
  reader->GetOutput()->GetScalarRange(scalarRange);
  self.dataMin = scalarRange[0];
  double dataMax = scalarRange[1];
  self.dataRange = dataMax - self.dataMin;
  if (self.dataRange == 0.0) {
    self.dataRange = 1.0;
  }

  vtkNew<vtkImageShiftScale> castToU8;
  castToU8->SetInputConnection(reader->GetOutputPort());
  castToU8->SetShift(-self.dataMin);
  castToU8->SetScale(255.0 / self.dataRange);
  castToU8->SetOutputScalarTypeToUnsignedChar();
  castToU8->ClampOverflowOn();
  castToU8->Update();
  reader->GetOutput()->ReleaseData();

  vtkNew<vtkMetalGPUVolumeRayCastMapper> mapper;
  mapper->SetInputData(castToU8->GetOutput());
  mapper->UseJitteringOn();
  mapper->AutoAdjustSampleDistancesOff();
  mapper->SetSampleDistance(0.5);
  mapper->SetPartitions(1, 1, 4);
  mapper->SetDisableInstanceRendering(true);

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
  return (hu - self.dataMin) / self.dataRange * 255.0;
}

#pragma mark - Clipping Plane (Scroll Slices)

- (void)handlePan:(UIPanGestureRecognizer*)recognizer
{
  if (self.interactionMode == VTKInteractionModeScrollSlices)
  {
    [self handleScrollSlicesPan:recognizer];
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
