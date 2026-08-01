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

#include "vtkActor.h"
#include "vtkActor2D.h"
#include "vtkCamera.h"
#include "vtkCellArray.h"
#include "vtkCocoaMetalRenderWindow.h"
#include "vtkCocoaRenderWindow.h"
#include "vtkColorTransferFunction.h"
#include "vtkCompositePolyDataMapper.h"
#include "vtkCompositePolyDataMapperDelegator.h"
#include "vtkConeSource.h"
#include "vtkCubeSource.h"
#include "vtkElevationFilter.h"
#include "vtkGlyph3DMapper.h"
#include "vtkGPUVolumeRayCastMapper.h"
#include "vtkImageData.h"
#include "vtkLight.h"
#include "vtkLightCollection.h"
#include "vtkMetalActor.h"
#include "vtkMetalCamera.h"
#include "vtkMetalCompositePolyDataMapperDelegator.h"
#include "vtkMetalGlyph3DMapper.h"
#include "vtkMetalGPUVolumeRayCastMapper.h"
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
#include "vtkOpenGLPolyDataMapper.h"
#include "vtkOpenGLPolyDataMapper2D.h"
#include "vtkOpenGLProperty.h"
#include "vtkOpenGLRenderer.h"
#include "vtkOpenGLTexture.h"
#include "vtkPartitionedDataSetCollection.h"
#include "vtkPiecewiseFunction.h"
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

} // namespace vtkMetalScenes

#endif // TestMetalScenes_h
