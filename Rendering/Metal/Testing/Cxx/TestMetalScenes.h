// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
//
// Backend-agnostic scene builders used by TestMetalGLVisualComparison.cxx.
// Each builder reproduces the final rendered configuration of the
// corresponding Rendering/Metal regression test, but instantiates the
// backend-specific classes (window, renderer, camera, actor, mapper) through
// the BackendKind helpers below, so the same scene can be rendered with both
// the Metal and the OpenGL backends for side-by-side visual comparison.
//
// The Metal renderer drives vtkActor::RenderOpaqueGeometry (via
// vtkRenderer::UpdateOpaquePolygonalGeometry) just like OpenGL, so it invokes
// Property::Render. This harness auto-initializes vtkRenderingOpenGL2 (the
// OpenGL backend needs the vtkShaderProgram object-factory override), which
// would make vtkProperty::New() return a vtkOpenGLProperty even for Metal
// actors -- and vtkOpenGLProperty::Render static-casts the renderer to
// vtkOpenGLRenderer and crashes on a vtkMetalRenderer. To be robust to the
// object-factory autoinit ordering, both backends set an explicit property on
// every actor: vtkMetalProperty for Metal, vtkOpenGLProperty for OpenGL.

#ifndef TestMetalScenes_h
#define TestMetalScenes_h

#include <iostream>

#include "vtkActor.h"
#include "vtkActor2D.h"
#include "vtkAppendPolyData.h"
#include "vtkCamera.h"
#include "vtkCellArray.h"
#include "vtkCellData.h"
#include "vtkCocoaMetalRenderWindow.h"
#include "vtkCocoaRenderWindow.h"
#include "vtkColorTransferFunction.h"
#include "vtkCompositePolyDataMapper.h"
#include "vtkCompositePolyDataMapperDelegator.h"
#include "vtkConeSource.h"
#include "vtkCubeSource.h"
#include "vtkDICOMDirectory.h"
#include "vtkDICOMReader.h"
#include "vtkElevationFilter.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "vtkGlyph3DMapper.h"
#include "vtkGPUVolumeRayCastMapper.h"
#include "vtkImageData.h"
#include "vtkImageMapper.h"
#include "vtkImageProperty.h"
#include "vtkImageShiftScale.h"
#include "vtkImageSlice.h"
#include "vtkImageSliceMapper.h"
#include "vtkLight.h"
#include "vtkLightCollection.h"
#include "vtkMetalActor.h"
#include "vtkMetalCamera.h"
#include "vtkMetalCompositePolyDataMapperDelegator.h"
#include "vtkMetalGlyph3DMapper.h"
#include "vtkMetalGPUVolumeRayCastMapper.h"
#include "vtkMetalImageMapper.h"
#include "vtkMetalPolyDataMapper.h"
#include "vtkMetalPolyDataMapper2D.h"
#include "vtkMetalProperty.h"
#include "vtkMetalRenderer.h"
#include "vtkNew.h"
#include "vtkObjectFactory.h"
#include "vtkOpenGLActor.h"
#include "vtkOpenGLCamera.h"
#include "vtkOpenGLGlyph3DMapper.h"
#include "vtkOpenGLGPUVolumeRayCastMapper.h"
#include "vtkOpenGLImageMapper.h"
#include "vtkOpenGLPolyDataMapper.h"
#include "vtkOpenGLPolyDataMapper2D.h"
#include "vtkOpenGLProperty.h"
#include "vtkOpenGLRenderer.h"
#include "vtkOpenGLTexture.h"
#include "vtkPartitionedDataSetCollection.h"
#include "vtkPiecewiseFunction.h"
#include "vtkPlane.h"
#include "vtkPlaneSource.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkProperty.h"
#include "vtkProperty2D.h"
#include "vtkRTAnalyticSource.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkSmartPointer.h"
#include "vtkSphereSource.h"
#include "vtkTexture.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"

namespace vtkMetalScenes
{

enum class BackendKind
{
  Metal,
  OpenGL
};

// DICOM study directory for the DICOM CT scene, set by the harness --dicom
// argument (defined in TestMetalGLVisualComparison.cxx). The scene builder
// falls back to the analytic volume when no directory is provided.
extern const char* gDicomDir;

// ---- Backend class factories ---------------------------------------------

inline vtkSmartPointer<vtkRenderWindow> NewRenderWindow(BackendKind b)
{
  if (b == BackendKind::OpenGL)
  {
    return vtkSmartPointer<vtkCocoaRenderWindow>::New();
  }
  return vtkSmartPointer<vtkCocoaMetalRenderWindow>::New();
}

inline vtkSmartPointer<vtkRenderer> NewRenderer(BackendKind b)
{
  if (b == BackendKind::OpenGL)
  {
    return vtkSmartPointer<vtkOpenGLRenderer>::New();
  }
  return vtkSmartPointer<vtkMetalRenderer>::New();
}

inline vtkSmartPointer<vtkCamera> NewCamera(BackendKind b)
{
  if (b == BackendKind::OpenGL)
  {
    return vtkSmartPointer<vtkOpenGLCamera>::New();
  }
  return vtkSmartPointer<vtkMetalCamera>::New();
}

inline vtkSmartPointer<vtkActor> NewActor(BackendKind b)
{
  if (b == BackendKind::OpenGL)
  {
    return vtkSmartPointer<vtkOpenGLActor>::New();
  }
  return vtkSmartPointer<vtkMetalActor>::New();
}

inline vtkSmartPointer<vtkPolyDataMapper> NewPolyDataMapper(BackendKind b)
{
  if (b == BackendKind::OpenGL)
  {
    return vtkSmartPointer<vtkOpenGLPolyDataMapper>::New();
  }
  return vtkSmartPointer<vtkMetalPolyDataMapper>::New();
}

inline vtkSmartPointer<vtkGlyph3DMapper> NewGlyph3DMapper(BackendKind b)
{
  if (b == BackendKind::OpenGL)
  {
    return vtkSmartPointer<vtkOpenGLGlyph3DMapper>::New();
  }
  return vtkSmartPointer<vtkMetalGlyph3DMapper>::New();
}

inline vtkSmartPointer<vtkPolyDataMapper2D> NewPolyDataMapper2D(BackendKind b)
{
  if (b == BackendKind::OpenGL)
  {
    return vtkSmartPointer<vtkOpenGLPolyDataMapper2D>::New();
  }
  return vtkSmartPointer<vtkMetalPolyDataMapper2D>::New();
}

inline vtkSmartPointer<vtkImageMapper> NewImageMapper(BackendKind b)
{
  if (b == BackendKind::OpenGL)
  {
    return vtkSmartPointer<vtkOpenGLImageMapper>::New();
  }
  return vtkSmartPointer<vtkMetalImageMapper>::New();
}

inline vtkSmartPointer<vtkGPUVolumeRayCastMapper> NewVolumeMapper(BackendKind b)
{
  if (b == BackendKind::OpenGL)
  {
    return vtkSmartPointer<vtkOpenGLGPUVolumeRayCastMapper>::New();
  }
  return vtkSmartPointer<vtkMetalGPUVolumeRayCastMapper>::New();
}

// vtkTexture::New() is hijacked by the vtkRenderingOpenGL2 object factory
// (returning vtkOpenGLTexture), whose Load() static-casts the renderer to
// vtkOpenGLRenderer. The Metal texture scene needs a texture with base
// vtkTexture behavior, so give it this harness-local subclass (its New() is
// not overridden by any factory).
namespace
{
class MetalTexture : public vtkTexture
{
public:
  static MetalTexture* New();
  vtkTypeMacro(MetalTexture, vtkTexture);

protected:
  MetalTexture() = default;
  ~MetalTexture() override = default;

private:
  MetalTexture(const MetalTexture&) = delete;
  void operator=(const MetalTexture&) = delete;
};
vtkStandardNewMacro(MetalTexture);
}

inline vtkSmartPointer<vtkTexture> NewTexture(BackendKind b)
{
  if (b == BackendKind::OpenGL)
  {
    return vtkSmartPointer<vtkOpenGLTexture>::New();
  }
  return vtkSmartPointer<MetalTexture>::New();
}

// Give both backends an explicit property so the object-factory autoinit of
// vtkRenderingOpenGL2 (required by the GL backend) cannot leak a
// vtkOpenGLProperty into Metal actors. See the header comment for details.
inline vtkSmartPointer<vtkProperty> NewProperty(BackendKind b)
{
  if (b == BackendKind::OpenGL)
  {
    return vtkSmartPointer<vtkOpenGLProperty>::New();
  }
  return vtkSmartPointer<vtkMetalProperty>::New();
}

inline void ConfigureActor(vtkActor* actor, BackendKind b)
{
  actor->SetProperty(NewProperty(b));
}

// ---- Scene builders -------------------------------------------------------

// TestMetalRenderWindow: a single cone.
inline void BuildRenderWindowScene(vtkRenderer* renderer, BackendKind b)
{
  vtkNew<vtkConeSource> cone;
  vtkSmartPointer<vtkPolyDataMapper> mapper = NewPolyDataMapper(b);
  mapper->SetInputConnection(cone->GetOutputPort());
  vtkSmartPointer<vtkActor> actor = NewActor(b);
  ConfigureActor(actor, b);
  actor->SetMapper(mapper);
  renderer->AddActor(actor);

  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
}

// TestMetalCamera: cone with the full camera-operation sequence, ending in
// parallel projection.
inline void BuildCameraScene(vtkRenderer* renderer, BackendKind b)
{
  vtkNew<vtkConeSource> cone;
  vtkSmartPointer<vtkPolyDataMapper> mapper = NewPolyDataMapper(b);
  mapper->SetInputConnection(cone->GetOutputPort());
  vtkSmartPointer<vtkActor> actor = NewActor(b);
  ConfigureActor(actor, b);
  actor->SetMapper(mapper);
  renderer->AddActor(actor);

  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);
  camera->SetPosition(0.0, 0.0, 1.0);
  camera->SetFocalPoint(0.0, 0.0, 0.0);
  camera->SetViewUp(0.0, 1.0, 0.0);
  renderer->ResetCamera();
  camera->Azimuth(45.0);
  renderer->ResetCameraClippingRange();
  camera->Elevation(30.0);
  renderer->ResetCameraClippingRange();
  camera->Roll(25.0);
  camera->OrthogonalizeViewUp();
  camera->Dolly(1.5);
  renderer->ResetCameraClippingRange();
  camera->Zoom(1.5);
  renderer->ResetCameraClippingRange();
  camera->ParallelProjectionOn();
  camera->SetParallelScale(1.5);
}

// TestMetalLight: cone lit by headlight, directional, point and spot lights;
// ends with only the spot light (the test removes the others).
inline void BuildLightScene(vtkRenderer* renderer, BackendKind b)
{
  vtkNew<vtkConeSource> cone;
  vtkSmartPointer<vtkPolyDataMapper> mapper = NewPolyDataMapper(b);
  mapper->SetInputConnection(cone->GetOutputPort());
  vtkSmartPointer<vtkActor> actor = NewActor(b);
  ConfigureActor(actor, b);
  actor->SetMapper(mapper);
  actor->GetProperty()->SetAmbient(0.1);
  actor->GetProperty()->SetDiffuse(0.8);
  actor->GetProperty()->SetSpecular(0.4);
  actor->GetProperty()->SetSpecularPower(40.0);
  renderer->AddActor(actor);

  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();

  vtkNew<vtkLight> headlight;
  headlight->SetLightTypeToHeadlight();
  renderer->AddLight(headlight);
  vtkNew<vtkLight> directional;
  directional->SetLightTypeToSceneLight();
  directional->SetPositional(0);
  directional->SetFocalPoint(0.0, 0.0, 0.0);
  directional->SetPosition(1.0, 1.0, 1.0);
  renderer->AddLight(directional);
  vtkNew<vtkLight> point;
  point->SetLightTypeToSceneLight();
  point->SetPositional(1);
  point->SetPosition(2.0, 3.0, 4.0);
  point->SetConeAngle(180.0);
  renderer->AddLight(point);
  vtkNew<vtkLight> spot;
  spot->SetLightTypeToSceneLight();
  spot->SetPositional(1);
  spot->SetFocalPoint(0.0, 0.0, 0.0);
  spot->SetPosition(0.0, 0.0, 5.0);
  spot->SetConeAngle(25.0);
  spot->SetExponent(5.0);
  renderer->AddLight(spot);

  // Match the final state of the test: only the spot light remains.
  renderer->GetLights()->RemoveItem(0);
  renderer->GetLights()->RemoveItem(0);
  renderer->GetLights()->RemoveItem(0);
  renderer->ResetCameraClippingRange();
}

// TestMetalActorProperty: translucent sphere with a backface property.
inline void BuildActorPropertyScene(vtkRenderer* renderer, BackendKind b)
{
  vtkNew<vtkSphereSource> sphere;
  vtkSmartPointer<vtkPolyDataMapper> mapper = NewPolyDataMapper(b);
  mapper->SetInputConnection(sphere->GetOutputPort());
  vtkSmartPointer<vtkActor> actor = NewActor(b);
  ConfigureActor(actor, b);
  actor->SetMapper(mapper);
  actor->GetProperty()->SetAmbient(0.1);
  actor->GetProperty()->SetDiffuse(0.8);
  actor->GetProperty()->SetSpecular(0.5);
  actor->GetProperty()->SetSpecularPower(30.0);
  actor->GetProperty()->SetDiffuseColor(0.4, 1.0, 1.0);
  actor->GetProperty()->SetAmbientColor(0.1, 0.2, 0.3);
  actor->GetProperty()->SetSpecularColor(1.0, 1.0, 1.0);
  vtkSmartPointer<vtkProperty> backProp = NewProperty(b);
  backProp->SetDiffuseColor(0.4, 0.65, 0.8);
  actor->SetBackfaceProperty(backProp);
  renderer->AddActor(actor);

  // Final state of the test: translucent surface, edges off.
  actor->GetProperty()->SetRepresentationToSurface();
  actor->GetProperty()->EdgeVisibilityOff();
  actor->GetProperty()->SetOpacity(0.5);

  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
}

// --- Order-independent transparency (OIT) regression scenes ---------------
// A single translucent sphere (with an optional backface property) rendered
// through the order-independent translucent pass. These exercise the OIT
// accumulate + resolve math with a discriminating set of front/back opacities
// and front/back colors: front opacity in {0.25, 0.5, 0.75}, back opacity in
// {0.25, 0.5, 1.0}, and green/red, green/blue, red/green color pairs. Each
// front/back (a_f, a_b) combination has a closed-form OIT result, so an exact
// GL match here pins down the OIT compositing and backface-material handling.
inline void BuildAP_DiagSceneEx(vtkRenderer* renderer, BackendKind b, double opacity,
  bool useBackProp, double bfOpacity, const double frontColor[3], const double backColor[3])
{
  vtkNew<vtkSphereSource> sphere;
  vtkSmartPointer<vtkPolyDataMapper> mapper = NewPolyDataMapper(b);
  mapper->SetInputConnection(sphere->GetOutputPort());
  vtkSmartPointer<vtkActor> actor = NewActor(b);
  ConfigureActor(actor, b);
  actor->SetMapper(mapper);
  actor->GetProperty()->SetAmbient(0.0);
  actor->GetProperty()->SetDiffuse(1.0);
  actor->GetProperty()->SetSpecular(0.0);
  actor->GetProperty()->SetDiffuseColor(frontColor[0], frontColor[1], frontColor[2]);
  if (useBackProp)
  {
    vtkSmartPointer<vtkProperty> backProp = NewProperty(b);
    backProp->SetDiffuseColor(backColor[0], backColor[1], backColor[2]);
    backProp->SetOpacity(bfOpacity);
    actor->SetBackfaceProperty(backProp);
  }
  renderer->AddActor(actor);
  actor->GetProperty()->SetRepresentationToSurface();
  actor->GetProperty()->EdgeVisibilityOff();
  actor->GetProperty()->SetOpacity(opacity);
  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
}

inline void BuildAP_DiagScene(vtkRenderer* renderer, BackendKind b, double opacity,
  bool useBackProp, double bfOpacity)
{
  const double frontColor[3] = { 0.0, 1.0, 0.0 };
  const double backColor[3] = { 1.0, 0.0, 0.0 };
  BuildAP_DiagSceneEx(renderer, b, opacity, useBackProp, bfOpacity, frontColor, backColor);
}

inline void BuildAP_OpaqueNoBF(vtkRenderer* r, BackendKind b) { BuildAP_DiagScene(r, b, 1.0, false, 1.0); }
inline void BuildAP_OpaqueBF(vtkRenderer* r, BackendKind b) { BuildAP_DiagScene(r, b, 1.0, true, 1.0); }
inline void BuildAP_TransNoBF(vtkRenderer* r, BackendKind b) { BuildAP_DiagScene(r, b, 0.5, false, 1.0); }
inline void BuildAP_TransBFbf05(vtkRenderer* r, BackendKind b) { BuildAP_DiagScene(r, b, 0.5, true, 0.5); }
inline void BuildAP_TransBFbf10(vtkRenderer* r, BackendKind b) { BuildAP_DiagScene(r, b, 0.5, true, 1.0); }
inline void BuildAP_OpFrTransBF(vtkRenderer* r, BackendKind b) { BuildAP_DiagScene(r, b, 1.0, true, 0.5); }
inline void BuildAP_Trans25BF05(vtkRenderer* r, BackendKind b) { BuildAP_DiagScene(r, b, 0.25, true, 0.5); }
inline void BuildAP_Trans75BF05(vtkRenderer* r, BackendKind b) { BuildAP_DiagScene(r, b, 0.75, true, 0.5); }

inline void BuildAP_CullScene(vtkRenderer* renderer, BackendKind b, int frontCull, int backCull)
{
  BuildAP_DiagScene(renderer, b, 0.5, true, 0.5);
  vtkActor* actor = renderer->GetActors()->GetLastActor();
  vtkProperty* prop = actor->GetProperty();
  prop->SetFrontfaceCulling(frontCull);
  prop->SetBackfaceCulling(backCull);
}
inline void BuildAP_FrontCull(vtkRenderer* r, BackendKind b) { BuildAP_CullScene(r, b, 1, 0); }
inline void BuildAP_BackCull(vtkRenderer* r, BackendKind b) { BuildAP_CullScene(r, b, 0, 1); }

inline void BuildAP_GB05(vtkRenderer* r, BackendKind b)
{
  const double f[3] = { 0.0, 1.0, 0.0 };
  const double bk[3] = { 0.0, 0.0, 1.0 };
  BuildAP_DiagSceneEx(r, b, 0.5, true, 0.5, f, bk);
}
inline void BuildAP_GB10(vtkRenderer* r, BackendKind b)
{
  const double f[3] = { 0.0, 1.0, 0.0 };
  const double bk[3] = { 0.0, 0.0, 1.0 };
  BuildAP_DiagSceneEx(r, b, 0.5, true, 1.0, f, bk);
}
inline void BuildAP_RG05(vtkRenderer* r, BackendKind b)
{
  const double f[3] = { 1.0, 0.0, 0.0 };
  const double bk[3] = { 0.0, 1.0, 0.0 };
  BuildAP_DiagSceneEx(r, b, 0.5, true, 0.5, f, bk);
}
inline void BuildAP_GR25(vtkRenderer* r, BackendKind b)
{
  const double f[3] = { 0.0, 1.0, 0.0 };
  const double bk[3] = { 1.0, 0.0, 0.0 };
  BuildAP_DiagSceneEx(r, b, 0.5, true, 0.25, f, bk);
}
inline void BuildAP_GR7510(vtkRenderer* r, BackendKind b)
{
  const double f[3] = { 0.0, 1.0, 0.0 };
  const double bk[3] = { 1.0, 0.0, 0.0 };
  BuildAP_DiagSceneEx(r, b, 0.75, true, 1.0, f, bk);
}
inline void BuildAP_GR7525(vtkRenderer* r, BackendKind b)
{
  const double f[3] = { 0.0, 1.0, 0.0 };
  const double bk[3] = { 1.0, 0.0, 0.0 };
  BuildAP_DiagSceneEx(r, b, 0.75, true, 0.25, f, bk);
}
inline void BuildAP_GR2510(vtkRenderer* r, BackendKind b)
{
  const double f[3] = { 0.0, 1.0, 0.0 };
  const double bk[3] = { 1.0, 0.0, 0.0 };
  BuildAP_DiagSceneEx(r, b, 0.25, true, 1.0, f, bk);
}

// TestMetalPointRender: sphere with thick tube edges (final state of the
// test's points-as-spheres / lines-as-tubes sequence).
inline void BuildPointRenderScene(vtkRenderer* renderer, BackendKind b)
{
  vtkNew<vtkSphereSource> sphere;
  vtkSmartPointer<vtkPolyDataMapper> mapper = NewPolyDataMapper(b);
  mapper->SetInputConnection(sphere->GetOutputPort());
  vtkSmartPointer<vtkActor> actor = NewActor(b);
  ConfigureActor(actor, b);
  actor->SetMapper(mapper);
  actor->GetProperty()->SetDiffuseColor(1.0, 0.65, 0.7);
  actor->GetProperty()->SetSpecular(0.5);
  actor->GetProperty()->SetDiffuse(0.7);
  actor->GetProperty()->SetSpecularPower(20.0);
  actor->GetProperty()->SetRepresentationToSurface();
  actor->GetProperty()->EdgeVisibilityOn();
  actor->GetProperty()->SetLineWidth(7.0);
  actor->GetProperty()->RenderLinesAsTubesOn();
  actor->GetProperty()->SetEdgeColor(1.0, 1.0, 1.0);
  renderer->AddActor(actor);

  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
  camera->Elevation(-45.0);
  camera->OrthogonalizeViewUp();
  camera->Zoom(1.5);
  renderer->ResetCameraClippingRange();
}

// TestMetalDepthPeeling: three overlapping translucent spheres rendered with
// depth peeling enabled.
inline vtkSmartPointer<vtkActor> MakeTranslucentSphere(BackendKind b, vtkSphereSource* sphere,
  double x, double y, double z, double r, double g, double colorB, double opacity)
{
  vtkSmartPointer<vtkPolyDataMapper> mapper = NewPolyDataMapper(b);
  mapper->SetInputConnection(sphere->GetOutputPort());
  vtkSmartPointer<vtkActor> actor = NewActor(b);
  ConfigureActor(actor, b);
  actor->SetMapper(mapper);
  actor->GetProperty()->SetAmbientColor(1.0, 0.0, 0.0);
  actor->GetProperty()->SetDiffuseColor(r, g, colorB);
  actor->GetProperty()->SetSpecular(0.0);
  actor->GetProperty()->SetDiffuse(0.5);
  actor->GetProperty()->SetAmbient(0.3);
  actor->GetProperty()->SetOpacity(opacity);
  actor->SetPosition(x, y, z);
  return actor;
}

inline void BuildDepthPeelingScene(vtkRenderer* renderer, BackendKind b)
{
  renderer->SetBackground(1.0, 1.0, 1.0);
  renderer->SetBackground2(0.3, 0.1, 0.2);
  renderer->GradientBackgroundOn();

  vtkNew<vtkSphereSource> sphere;
  sphere->SetThetaResolution(24);
  sphere->SetPhiResolution(24);
  renderer->AddActor(MakeTranslucentSphere(b, sphere, -0.5, 0.0, 0.0, 1.0, 0.8, 0.3, 0.35));
  renderer->AddActor(MakeTranslucentSphere(b, sphere, 0.0, 0.0, 0.2, 0.2, 1.0, 0.8, 0.2));
  renderer->AddActor(MakeTranslucentSphere(b, sphere, 0.5, 0.0, -0.2, 0.5, 0.65, 1.0, 0.35));

  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
  camera->Azimuth(15.0);
  camera->Zoom(1.5);
  renderer->ResetCameraClippingRange();

  renderer->SetUseDepthPeeling(true);
  renderer->SetMaximumNumberOfPeels(20);
  renderer->SetOcclusionRatio(0.0);

  // Second camera move from the test (exercises peel-texture reuse).
  camera->Azimuth(30.0);
  renderer->ResetCameraClippingRange();
}

// TestMetalCompositePolyDataMapper: cone, sphere and cube as a partitioned
// dataset collection rendered by the composite poly data mapper.
namespace
{
class MetalCompositePolyDataMapper : public vtkCompositePolyDataMapper
{
public:
  static MetalCompositePolyDataMapper* New();
  vtkTypeMacro(MetalCompositePolyDataMapper, vtkCompositePolyDataMapper);

  vtkCompositePolyDataMapperDelegator* CreateADelegator() override
  {
    return vtkMetalCompositePolyDataMapperDelegator::New();
  }

protected:
  MetalCompositePolyDataMapper() = default;
  ~MetalCompositePolyDataMapper() override = default;

private:
  MetalCompositePolyDataMapper(const MetalCompositePolyDataMapper&) = delete;
  void operator=(const MetalCompositePolyDataMapper&) = delete;
};
vtkStandardNewMacro(MetalCompositePolyDataMapper);
}

inline void BuildCompositeScene(vtkRenderer* renderer, BackendKind b)
{
  vtkNew<vtkPartitionedDataSetCollection> pdc;
  vtkNew<vtkConeSource> cone;
  cone->SetCenter(-2.0, 0.0, 0.0);
  cone->SetResolution(24);
  cone->Update();
  vtkNew<vtkSphereSource> sphere;
  sphere->SetCenter(0.0, 0.0, 0.0);
  sphere->SetThetaResolution(24);
  sphere->SetPhiResolution(24);
  sphere->Update();
  vtkNew<vtkCubeSource> cube;
  cube->SetCenter(2.0, 0.0, 0.0);
  cube->Update();
  pdc->SetPartition(0, 0, cone->GetOutput());
  pdc->SetPartition(1, 0, sphere->GetOutput());
  pdc->SetPartition(2, 0, cube->GetOutput());

  vtkSmartPointer<vtkCompositePolyDataMapper> mapper;
  if (b == BackendKind::OpenGL)
  {
    mapper = vtkSmartPointer<vtkCompositePolyDataMapper>::New();
  }
  else
  {
    mapper = vtkSmartPointer<MetalCompositePolyDataMapper>::New();
  }
  mapper->SetInputDataObject(pdc);

  vtkSmartPointer<vtkActor> actor = NewActor(b);
  ConfigureActor(actor, b);
  actor->SetMapper(mapper);
  renderer->AddActor(actor);

  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
  renderer->GetActiveCamera()->Zoom(1.4);
  renderer->ResetCameraClippingRange();
  renderer->GetActiveCamera()->Azimuth(30.0);
  renderer->ResetCameraClippingRange();
}

// TestMetalGlyph3DMapper: a 4x4 point plane instancing spheres with
// per-instance colors.
inline void BuildGlyphScene(vtkRenderer* renderer, BackendKind b)
{
  vtkNew<vtkPlaneSource> plane;
  plane->SetResolution(3, 3);
  vtkNew<vtkElevationFilter> colors;
  colors->SetInputConnection(plane->GetOutputPort());
  colors->SetLowPoint(-1, -1, -1);
  colors->SetHighPoint(1, 1, 1);

  vtkNew<vtkSphereSource> squad;
  squad->SetPhiResolution(6);
  squad->SetThetaResolution(12);

  vtkSmartPointer<vtkGlyph3DMapper> glypher = NewGlyph3DMapper(b);
  glypher->SetInputConnection(colors->GetOutputPort());
  glypher->SetSourceConnection(squad->GetOutputPort());
  glypher->SetScaleFactor(0.25);
  glypher->ScalarVisibilityOn();
  glypher->SetColorModeToMapScalars();
  glypher->SetScalarRange(0.0, 1.0);

  vtkSmartPointer<vtkActor> glyphActor = NewActor(b);
  ConfigureActor(glyphActor, b);
  glyphActor->SetMapper(glypher);
  renderer->AddActor(glyphActor);

  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
  renderer->GetActiveCamera()->Zoom(1.2);
}

// TestMetalHardwareSelector: cone and sphere side by side.
inline void BuildHardwareSelectorScene(vtkRenderer* renderer, BackendKind b)
{
  vtkNew<vtkConeSource> cone;
  cone->SetResolution(32);
  vtkSmartPointer<vtkPolyDataMapper> coneMapper = NewPolyDataMapper(b);
  coneMapper->SetInputConnection(cone->GetOutputPort());
  vtkSmartPointer<vtkActor> coneActor = NewActor(b);
  ConfigureActor(coneActor, b);
  coneActor->SetMapper(coneMapper);
  coneActor->SetPosition(-1.5, 0, 0);
  renderer->AddActor(coneActor);

  vtkNew<vtkSphereSource> sphere;
  sphere->SetPhiResolution(24);
  sphere->SetThetaResolution(24);
  vtkSmartPointer<vtkPolyDataMapper> sphereMapper = NewPolyDataMapper(b);
  sphereMapper->SetInputConnection(sphere->GetOutputPort());
  vtkSmartPointer<vtkActor> sphereActor = NewActor(b);
  ConfigureActor(sphereActor, b);
  sphereActor->SetMapper(sphereMapper);
  sphereActor->SetPosition(1.5, 0, 0);
  renderer->AddActor(sphereActor);

  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
  renderer->GetActiveCamera()->Zoom(1.4);
}

// TestMetalPolyDataMapper2D: a 3D cone plus a 2D overlay quad in display
// coordinates. Note: the Metal backend does not yet drive RenderOverlay, so
// the quad only appears in the OpenGL render.
inline void BuildPolyDataMapper2DScene(vtkRenderer* renderer, BackendKind b)
{
  vtkNew<vtkConeSource> cone;
  cone->SetResolution(32);
  vtkSmartPointer<vtkPolyDataMapper> coneMapper = NewPolyDataMapper(b);
  coneMapper->SetInputConnection(cone->GetOutputPort());
  vtkSmartPointer<vtkActor> coneActor = NewActor(b);
  ConfigureActor(coneActor, b);
  coneActor->SetMapper(coneMapper);
  coneActor->SetPosition(-1.5, 0, 0);
  renderer->AddActor(coneActor);

  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
  renderer->GetActiveCamera()->Zoom(1.4);

  vtkNew<vtkPoints> points;
  points->InsertNextPoint(450, 40, 0);
  points->InsertNextPoint(580, 40, 0);
  points->InsertNextPoint(580, 260, 0);
  points->InsertNextPoint(450, 260, 0);
  vtkNew<vtkCellArray> cellArray;
  vtkIdType quad[4] = { 0, 1, 2, 3 };
  cellArray->InsertNextCell(4, quad);
  vtkNew<vtkPolyData> quadData;
  quadData->SetPoints(points);
  quadData->SetPolys(cellArray);

  vtkSmartPointer<vtkPolyDataMapper2D> quadMapper = NewPolyDataMapper2D(b);
  quadMapper->SetInputData(quadData);
  vtkNew<vtkActor2D> quadActor;
  quadActor->SetMapper(quadMapper);
  quadActor->GetProperty()->SetColor(1.0, 0.5, 0.0);
  renderer->AddActor(quadActor);
}

// TestMetalImageMapper: two 2D images rendered in the overlay pass -- a
// 64x64 RGB uchar image with a distinct color per quadrant (identity
// window/level, char path) at (0,0), and a 64x64 single-component
// unsigned-short gradient 0..1000 along x+y (real window/level, short path)
// at (300,0). The window must be at least 600x300 for both to fit.
inline vtkSmartPointer<vtkImageData> CreateQuadrantImage()
{
  constexpr int dim = 64;
  vtkNew<vtkImageData> image;
  image->SetDimensions(dim, dim, 1);
  image->AllocateScalars(VTK_UNSIGNED_CHAR, 3);

  unsigned char* ptr = static_cast<unsigned char*>(image->GetScalarPointer());
  for (int y = 0; y < dim; ++y)
  {
    for (int x = 0; x < dim; ++x)
    {
      unsigned char r = 0, g = 0, b = 0;
      const bool left = x < dim / 2;
      const bool bottom = y < dim / 2;
      if (left && bottom)
      {
        r = 255;
      }
      else if (!left && bottom)
      {
        g = 255;
      }
      else if (left && !bottom)
      {
        b = 255;
      }
      else
      {
        r = g = b = 255;
      }
      ptr[(y * dim + x) * 3 + 0] = r;
      ptr[(y * dim + x) * 3 + 1] = g;
      ptr[(y * dim + x) * 3 + 2] = b;
    }
  }
  return image;
}

inline vtkSmartPointer<vtkImageData> CreateGradientImage()
{
  constexpr int dim = 64;
  vtkNew<vtkImageData> image;
  image->SetDimensions(dim, dim, 1);
  image->AllocateScalars(VTK_UNSIGNED_SHORT, 1);

  unsigned short* ptr = static_cast<unsigned short*>(image->GetScalarPointer());
  for (int y = 0; y < dim; ++y)
  {
    for (int x = 0; x < dim; ++x)
    {
      ptr[y * dim + x] = static_cast<unsigned short>(1000 * (x + y) / (2 * (dim - 1)));
    }
  }
  return image;
}

inline void BuildImageMapperScene(vtkRenderer* renderer, BackendKind b)
{
  renderer->SetBackground(0.2, 0.2, 0.2);

  // Give the renderer an explicit backend camera. Without one vtkRenderer
  // auto-creates vtkCamera::New(), which the vtkRenderingOpenGL2 object
  // factory hijacks into vtkOpenGLCamera even on a Metal renderer.
  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);

  vtkSmartPointer<vtkImageMapper> mapper1 = NewImageMapper(b);
  mapper1->SetInputData(CreateQuadrantImage());
  mapper1->SetColorLevel(127.5);
  mapper1->SetColorWindow(255.0);
  vtkNew<vtkActor2D> actor1;
  actor1->SetMapper(mapper1);
  actor1->SetPosition(0, 0);
  renderer->AddActor(actor1);

  vtkSmartPointer<vtkImageMapper> mapper2 = NewImageMapper(b);
  mapper2->SetInputData(CreateGradientImage());
  mapper2->SetColorLevel(500.0);
  mapper2->SetColorWindow(1000.0);
  vtkNew<vtkActor2D> actor2;
  actor2->SetMapper(mapper2);
  actor2->SetPosition(300, 0);
  renderer->AddActor(actor2);
}

// TestMetalImageSliceMapper: a 3D image slice (vtkImageSlice +
// vtkImageSliceMapper) displaying a 64x64 quadrant image, rendered through the
// opaque geometry pass. The mapper is created through the object factory, so
// it resolves to vtkMetalImageSliceMapper / vtkOpenGLImageSliceMapper for the
// respective backends (the vtkRenderingMetal / vtkRenderingOpenGL2 factory
// overrides carry the RenderingBackend override attribute).
inline void BuildImageSliceMapperScene(vtkRenderer* renderer, BackendKind b)
{
  renderer->SetBackground(0.2, 0.2, 0.2);

  // Give the renderer an explicit backend camera (see BuildImageMapperScene).
  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);

  vtkNew<vtkImageSliceMapper> mapper;
  mapper->SetInputData(CreateQuadrantImage());

  vtkNew<vtkImageSlice> slice;
  slice->SetMapper(mapper);
  renderer->AddViewProp(slice);

  renderer->ResetCamera();
  renderer->GetActiveCamera()->Zoom(1.4);
  renderer->ResetCameraClippingRange();
}

// Large-image fill-rate scenes: a single RGB image sized to fill the window,
// exercising the image-mapper texture/fragment path at 1:1 (CpxImg*).
inline vtkSmartPointer<vtkImageData> CreateSizedImage(int dim)
{
  vtkNew<vtkImageData> image;
  image->SetDimensions(dim, dim, 1);
  image->AllocateScalars(VTK_UNSIGNED_CHAR, 3);

  unsigned char* ptr = static_cast<unsigned char*>(image->GetScalarPointer());
  for (int y = 0; y < dim; ++y)
  {
    for (int x = 0; x < dim; ++x)
    {
      const int idx = y * dim + x;
      ptr[idx * 3 + 0] = static_cast<unsigned char>(x * 255 / (dim - 1));
      ptr[idx * 3 + 1] = static_cast<unsigned char>(y * 255 / (dim - 1));
      ptr[idx * 3 + 2] = static_cast<unsigned char>((x ^ y) & 0xff);
    }
  }
  return image;
}

inline void BuildImageSizeScene(vtkRenderer* renderer, BackendKind b, int dim)
{
  renderer->SetBackground(0.2, 0.2, 0.2);

  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);

  vtkSmartPointer<vtkImageMapper> mapper = NewImageMapper(b);
  mapper->SetInputData(CreateSizedImage(dim));
  mapper->SetColorLevel(127.5);
  mapper->SetColorWindow(255.0);
  vtkNew<vtkActor2D> actor;
  actor->SetMapper(mapper);
  actor->SetPosition(0, 0);
  renderer->AddActor(actor);
}

// TestMetalTexture: a plane textured with a 64x64 checkerboard image.
inline vtkSmartPointer<vtkImageData> CreateCheckerboardImage()
{
  constexpr int dim = 64;
  vtkNew<vtkImageData> image;
  image->SetDimensions(dim, dim, 1);
  image->SetSpacing(1, 1, 1);
  image->SetOrigin(0, 0, 0);
  image->AllocateScalars(VTK_UNSIGNED_CHAR, 3);

  unsigned char* ptr = static_cast<unsigned char*>(image->GetScalarPointer());
  for (int y = 0; y < dim; ++y)
  {
    for (int x = 0; x < dim; ++x)
    {
      const bool white = ((x / 8) + (y / 8)) % 2 == 0;
      const unsigned char v = white ? 255 : 32;
      ptr[(y * dim + x) * 3 + 0] = v;
      ptr[(y * dim + x) * 3 + 1] = white ? 200 : 32;
      ptr[(y * dim + x) * 3 + 2] = white ? 100 : 32;
    }
  }
  return image;
}

inline void BuildTextureScene(vtkRenderer* renderer, BackendKind b)
{
  vtkNew<vtkPlaneSource> plane;
  plane->SetResolution(8, 8);
  vtkSmartPointer<vtkPolyDataMapper> mapper = NewPolyDataMapper(b);
  mapper->SetInputConnection(plane->GetOutputPort());

  vtkSmartPointer<vtkTexture> texture = NewTexture(b);
  texture->InterpolateOn();
  texture->RepeatOff();
  texture->EdgeClampOn();
  texture->SetInputData(CreateCheckerboardImage());

  vtkSmartPointer<vtkActor> actor = NewActor(b);
  ConfigureActor(actor, b);
  actor->SetMapper(mapper);
  actor->SetTexture(texture);
  renderer->AddActor(actor);

  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
  renderer->GetActiveCamera()->Azimuth(-20);
  renderer->GetActiveCamera()->Elevation(20);
}

// TestMetalVolumeRayCast: an analytic volume rendered with the GPU volume
// ray-cast mapper and a shaded transfer function.
inline void BuildVolumeScene(vtkRenderer* renderer, BackendKind b)
{
  vtkNew<vtkRTAnalyticSource> source;
  source->SetWholeExtent(0, 32, 0, 32, 0, 32);
  source->SetCenter(16, 16, 16);

  vtkNew<vtkColorTransferFunction> color;
  color->AddRGBPoint(0.0, 0.0, 0.0, 0.0);
  color->AddRGBPoint(0.25, 1.0, 0.0, 0.0);
  color->AddRGBPoint(0.75, 0.0, 1.0, 0.0);
  color->AddRGBPoint(1.0, 0.0, 0.0, 1.0);

  vtkNew<vtkPiecewiseFunction> opacity;
  opacity->AddPoint(0.0, 0.0);
  opacity->AddPoint(0.5, 0.15);
  opacity->AddPoint(1.0, 0.9);

  vtkNew<vtkVolumeProperty> property;
  property->SetColor(color);
  property->SetScalarOpacity(opacity);
  property->ShadeOn();
  property->SetAmbient(0.2);
  property->SetDiffuse(0.8);
  property->SetSpecular(0.3);

  vtkSmartPointer<vtkGPUVolumeRayCastMapper> mapper = NewVolumeMapper(b);
  mapper->SetInputConnection(source->GetOutputPort());

  vtkNew<vtkVolume> volume;
  volume->SetMapper(mapper);
  volume->SetProperty(property);
  renderer->AddVolume(volume);

  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
  renderer->GetActiveCamera()->Elevation(20);
  renderer->GetActiveCamera()->Azimuth(30);
  renderer->GetActiveCamera()->Elevation(-40);
  renderer->GetActiveCamera()->Azimuth(-60);
}

// TestMetalVolumeRayCast with a parameterized volume extent.
inline void BuildVolumeSceneSized(vtkRenderer* renderer, BackendKind b, int dim)
{
  vtkNew<vtkRTAnalyticSource> source;
  source->SetWholeExtent(0, dim, 0, dim, 0, dim);
  source->SetCenter(dim / 2, dim / 2, dim / 2);

  vtkNew<vtkColorTransferFunction> color;
  color->AddRGBPoint(0.0, 0.0, 0.0, 0.0);
  color->AddRGBPoint(0.25, 1.0, 0.0, 0.0);
  color->AddRGBPoint(0.75, 0.0, 1.0, 0.0);
  color->AddRGBPoint(1.0, 0.0, 0.0, 1.0);

  vtkNew<vtkPiecewiseFunction> opacity;
  opacity->AddPoint(0.0, 0.0);
  opacity->AddPoint(0.5, 0.15);
  opacity->AddPoint(1.0, 0.9);

  vtkNew<vtkVolumeProperty> property;
  property->SetColor(color);
  property->SetScalarOpacity(opacity);
  property->ShadeOn();
  property->SetAmbient(0.2);
  property->SetDiffuse(0.8);
  property->SetSpecular(0.3);

  vtkSmartPointer<vtkGPUVolumeRayCastMapper> mapper = NewVolumeMapper(b);
  mapper->SetInputConnection(source->GetOutputPort());

  vtkNew<vtkVolume> volume;
  volume->SetMapper(mapper);
  volume->SetProperty(property);
  renderer->AddVolume(volume);

  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
  renderer->GetActiveCamera()->Elevation(20);
  renderer->GetActiveCamera()->Azimuth(30);
  renderer->GetActiveCamera()->Elevation(-40);
  renderer->GetActiveCamera()->Azimuth(-60);
}

// ---- DICOM CT scene ---------------------------------------------------------
// Replicates the DICOMVolumeViewController pipeline
// (Examples/GUI/iOSMetal/test-vtk-metal/DICOMVolumeViewController.mm): a
// vtkDICOMDirectory scan of the study -> series 0 -> vtkDICOMReader ->
// vtkImageShiftScale cast to U8 (shift 1024, scale 255/4095, clamped, exactly
// as the app does) -> the volume mapper configured with jittering, fixed 0.5
// sample distance, IGN jitter and GPU minmax (Metal), a (0,0,1) clipping plane
// and the "Airways II" preset's transfer function (the first preset the app
// applies, its 16-bit CLUT x-values rescaled to the U8 range).
//
// The reader output for CT is signed short (Hounsfield units) so the U8 cast
// is the same 1-byte-per-voxel upload the app performs. The study directory
// comes from gDicomDir (the harness --dicom argument). If no directory is
// available, this scene falls back to the analytic volume so the harness still
// renders a meaningful comparison everywhere.
inline void BuildDICOMVolumeScene(vtkRenderer* renderer, BackendKind b)
{
  // The U8 volume is computed once and cached so both backends (and every
  // bench rep) render the same data without re-reading the study.
  static vtkSmartPointer<vtkImageData> cachedU8Volume;
  static bool tried = false;
  if (!tried)
  {
    tried = true;
    if (gDicomDir)
    {
      vtkNew<vtkDICOMDirectory> dicomDir;
      dicomDir->SetDirectoryName(gDicomDir);
      dicomDir->Update();
      if (dicomDir->GetNumberOfSeries() > 0)
      {
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
        cachedU8Volume = castToU8->GetOutput();
      }
    }
    if (!cachedU8Volume)
    {
      std::cerr << "BuildDICOMVolumeScene: no DICOM series (--dicom not set or "
                   "no series found); falling back to the analytic volume.\n";
    }
  }

  if (!cachedU8Volume)
  {
    BuildVolumeSceneSized(renderer, b, 128);
    return;
  }

  // "Airways II" preset transfer function (VRPresets/Airways II.plist),
  // x-values rescaled by the app's rescale: (hu + 1024) * (255/4095).
  // Opacity: (-742.1,0) (-683,0.0493) (-481,0.2497) (-333.5,0); single color
  // (0,0.605,0.706).
  vtkNew<vtkColorTransferFunction> color;
  vtkNew<vtkPiecewiseFunction> opacity;
  const double xs[4] = { -742.1, -683.0, -481.0, -333.5 };
  const double ys[4] = { 0.0, 0.0493, 0.2497, 0.0 };
  for (int i = 0; i < 4; ++i)
  {
    const double x = (xs[i] + 1024.0) * (255.0 / 4095.0);
    opacity->AddPoint(x, ys[i]);
    color->AddRGBPoint(x, 0.0, 0.605, 0.706);
  }

  vtkNew<vtkVolumeProperty> property;
  property->SetColor(color);
  property->SetScalarOpacity(opacity);
  property->SetInterpolationTypeToLinear();

  vtkSmartPointer<vtkGPUVolumeRayCastMapper> mapper = NewVolumeMapper(b);
  mapper->SetInputData(cachedU8Volume);
  mapper->UseJitteringOn();
  mapper->AutoAdjustSampleDistancesOff();
  mapper->SetSampleDistance(0.5);
  if (b == BackendKind::Metal)
  {
    if (auto* metal = vtkMetalGPUVolumeRayCastMapper::SafeDownCast(mapper))
    {
      metal->SetUseIGNJitter(true);
      metal->SetUseGPUMinMax(true);
    }
  }

  // Clipping plane as the app starts with: normal (0,0,1), origin at the near
  // z bound so nothing is clipped yet (the app scrolls the plane into the
  // volume; here it starts at the front face, matching its initial state).
  vtkNew<vtkPlane> clipPlane;
  clipPlane->SetNormal(0, 0, 1);
  double bounds[6];
  cachedU8Volume->GetBounds(bounds);
  clipPlane->SetOrigin(0, 0, bounds[4]);
  mapper->AddClippingPlane(clipPlane);

  vtkNew<vtkVolume> volume;
  volume->SetMapper(mapper);
  volume->SetProperty(property);
  renderer->AddVolume(volume);

  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
  renderer->GetActiveCamera()->Elevation(20);
  renderer->GetActiveCamera()->Azimuth(30);
  renderer->GetActiveCamera()->Elevation(-40);
  renderer->GetActiveCamera()->Azimuth(-60);
}

// ---- Complexity-scaling benchmark scenes ----------------------------------
// These are registered in the harness's bench-only list (--complexity) to
// probe how the Metal/GL timing ratio behaves as a single workload axis
// grows. Four axes: GPU-bound geometry (merged into one draw call), CPU-bound
// draw-call count (many actors), depth-peel count (overlapping translucent
// spheres along the view axis), and volume size.

// GPU-bound: a grid of spheres merged into a single polydata rendered by one
// mapper/draw call, so the frame time is dominated by rasterization/fill
// rather than per-draw overhead. n*n spheres, each res*res*2 triangles.
inline void BuildGeometryGridScene(vtkRenderer* renderer, BackendKind b, int n, int res)
{
  vtkNew<vtkAppendPolyData> append;
  const double spacing = 1.8;
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      vtkNew<vtkSphereSource> sphere;
      sphere->SetThetaResolution(res);
      sphere->SetPhiResolution(res);
      sphere->SetRadius(0.8);
      sphere->SetCenter(i * spacing, j * spacing, 0.0);
      sphere->Update();
      append->AddInputData(sphere->GetOutput());
    }
  }

  vtkSmartPointer<vtkPolyDataMapper> mapper = NewPolyDataMapper(b);
  mapper->SetInputConnection(append->GetOutputPort());
  vtkSmartPointer<vtkActor> actor = NewActor(b);
  ConfigureActor(actor, b);
  actor->SetMapper(mapper);
  renderer->AddActor(actor);

  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
}

// Same geometry grid as BuildGeometryGridScene, but with per-cell scalar colors
// (ScalarModeToUseCellData) so the mapper exercises the "cell-texture" port:
// the vertex stream stays deduplicated (point count, not 3x triangles) while
// each triangle resolves the RGBA of its owning cell per-primitive.
inline void BuildCellColorGridScene(vtkRenderer* renderer, BackendKind b, int n, int res)
{
  vtkNew<vtkAppendPolyData> append;
  const double spacing = 1.8;
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      vtkNew<vtkSphereSource> sphere;
      sphere->SetThetaResolution(res);
      sphere->SetPhiResolution(res);
      sphere->SetRadius(0.8);
      sphere->SetCenter(i * spacing, j * spacing, 0.0);
      sphere->Update();
      append->AddInputData(sphere->GetOutput());
    }
  }
  append->Update();
  vtkSmartPointer<vtkPolyData> poly = vtkSmartPointer<vtkPolyData>::New();
  poly->ShallowCopy(append->GetOutput());

  const vtkIdType ncells = poly->GetNumberOfCells();
  vtkNew<vtkFloatArray> cellScalars;
  cellScalars->SetNumberOfComponents(1);
  cellScalars->SetNumberOfTuples(ncells);
  for (vtkIdType c = 0; c < ncells; ++c)
  {
    cellScalars->SetValue(c, (ncells > 1) ? static_cast<double>(c) / (ncells - 1) : 0.0);
  }
  cellScalars->SetName("cellScalars");
  poly->GetCellData()->SetScalars(cellScalars);

  vtkSmartPointer<vtkPolyDataMapper> mapper = NewPolyDataMapper(b);
  mapper->SetInputData(poly);
  mapper->ScalarVisibilityOn();
  mapper->SetScalarModeToUseCellData();
  mapper->SetColorModeToMapScalars();
  mapper->SetScalarRange(0.0, 1.0);
  vtkSmartPointer<vtkActor> actor = NewActor(b);
  ConfigureActor(actor, b);
  actor->SetMapper(mapper);
  renderer->AddActor(actor);

  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
}

// Diagnostic: the same grid as BuildGeometryGridScene but with per-point
// scalar colors, so the bench can compare the lean (no scalars), per-point,
// and per-cell colored fragment paths on identical geometry.
inline void BuildPointColorGridScene(vtkRenderer* renderer, BackendKind b, int n, int res)
{
  vtkNew<vtkAppendPolyData> append;
  const double spacing = 1.8;
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      vtkNew<vtkSphereSource> sphere;
      sphere->SetThetaResolution(res);
      sphere->SetPhiResolution(res);
      sphere->SetRadius(0.8);
      sphere->SetCenter(i * spacing, j * spacing, 0.0);
      sphere->Update();
      append->AddInputData(sphere->GetOutput());
    }
  }
  append->Update();
  vtkSmartPointer<vtkPolyData> poly = vtkSmartPointer<vtkPolyData>::New();
  poly->ShallowCopy(append->GetOutput());

  const vtkIdType npoints = poly->GetNumberOfPoints();
  vtkNew<vtkFloatArray> pointScalars;
  pointScalars->SetNumberOfComponents(1);
  pointScalars->SetNumberOfTuples(npoints);
  for (vtkIdType p = 0; p < npoints; ++p)
  {
    pointScalars->SetValue(p, (npoints > 1) ? static_cast<double>(p) / (npoints - 1) : 0.0);
  }
  pointScalars->SetName("pointScalars");
  poly->GetPointData()->SetScalars(pointScalars);

  vtkSmartPointer<vtkPolyDataMapper> mapper = NewPolyDataMapper(b);
  mapper->SetInputData(poly);
  mapper->ScalarVisibilityOn();
  mapper->SetScalarModeToUsePointData();
  mapper->SetColorModeToMapScalars();
  mapper->SetScalarRange(0.0, 1.0);
  vtkSmartPointer<vtkActor> actor = NewActor(b);
  ConfigureActor(actor, b);
  actor->SetMapper(mapper);
  renderer->AddActor(actor);

  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
}

// CPU-bound: an n*n grid of spheres, each its own mapper and actor, so the
// frame time is dominated by per-actor draw-call/state-change overhead rather
// than fill rate (small, cheap spheres).
inline void BuildActorGridScene(vtkRenderer* renderer, BackendKind b, int n)
{
  const double spacing = 1.5;
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      vtkNew<vtkSphereSource> sphere;
      sphere->SetThetaResolution(8);
      sphere->SetPhiResolution(8);
      sphere->SetRadius(0.55);
      vtkSmartPointer<vtkPolyDataMapper> mapper = NewPolyDataMapper(b);
      mapper->SetInputConnection(sphere->GetOutputPort());
      vtkSmartPointer<vtkActor> actor = NewActor(b);
      ConfigureActor(actor, b);
      actor->SetMapper(mapper);
      actor->SetPosition(i * spacing, j * spacing, 0.0);
      renderer->AddActor(actor);
    }
  }

  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
}

// Depth-peel count: n translucent spheres stacked along the view axis, so the
// screen-space depth complexity at the center grows with n. GL pays one
// blocking occlusion-query stall per peel; Metal's adaptive early exit does
// not, so the ratio should widen as n grows.
inline void BuildPeelChainScene(vtkRenderer* renderer, BackendKind b, int n)
{
  renderer->SetBackground(0.2, 0.2, 0.3);
  renderer->SetBackground2(0.4, 0.2, 0.3);
  renderer->GradientBackgroundOn();

  vtkNew<vtkSphereSource> sphere;
  sphere->SetThetaResolution(20);
  sphere->SetPhiResolution(20);
  for (int i = 0; i < n; ++i)
  {
    const double z = (n == 1) ? 0.0 : (2.0 * i / (n - 1) - 1.0);
    const double r = 0.4 + 0.6 * i / n;
    renderer->AddActor(MakeTranslucentSphere(b, sphere, 0.0, 0.0, z * 2.0, r, 0.6, 0.8, 0.35));
  }

  vtkSmartPointer<vtkCamera> camera = NewCamera(b);
  renderer->SetActiveCamera(camera);
  renderer->ResetCamera();
  camera->Zoom(1.2);
  renderer->ResetCameraClippingRange();

  renderer->SetUseDepthPeeling(true);
  renderer->SetMaximumNumberOfPeels(32);
  renderer->SetOcclusionRatio(0.0);
}

} // namespace vtkMetalScenes

#endif // TestMetalScenes_h
