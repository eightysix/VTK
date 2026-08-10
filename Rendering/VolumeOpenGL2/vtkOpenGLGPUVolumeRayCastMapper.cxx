// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkOpenGLGPUVolumeRayCastMapper.h"

#include <vtk_glad.h>

#include "vtkVolumeShaderComposer.h"
#include "vtkVolumeStateRAII.h"

// Include compiled shader code
#include <raycasterfs.h>
#include <raycastervs.h>

// VTK includes
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkOpenGLActor.h"
#include "vtkOpenGLResourceFreeCallback.h"
#include "vtkOpenGLState.h"
#include "vtkOpenGLUniforms.h"
#include <vtkBoundingBox.h>
#include <vtkCamera.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkClipConvexPolyData.h>
#include <vtkColorTransferFunction.h>
#include <vtkCommand.h>
#include <vtkContourFilter.h>
#include <vtkDataArray.h>
#include <vtkDensifyPolyData.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkLight.h>
#include <vtkLightCollection.h>
#include <vtkMath.h>
#include <vtkMatrix4x4.h>
#include <vtkMultiVolume.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkOpenGLBufferObject.h>
#include <vtkOpenGLCamera.h>
#include <vtkOpenGLError.h>
#include <vtkOpenGLFramebufferObject.h>
#include <vtkOpenGLRenderPass.h>
#include <vtkOpenGLRenderUtilities.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkOpenGLShaderCache.h>
#include <vtkOpenGLShaderProperty.h>
#include <vtkOpenGLVertexArrayObject.h>
#include <vtkOverrideAttribute.h>
#include <vtkPixelBufferObject.h>
#include <vtkPixelExtent.h>
#include <vtkPixelTransfer.h>
#include <vtkPlaneCollection.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkRectilinearGrid.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkShader.h>
#include <vtkShaderProgram.h>
#include <vtkSmartPointer.h>
#include <vtkTessellatedBoxSource.h>
#include <vtkTextureObject.h>
#include <vtkTimerLog.h>
#include <vtkTransform.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnsignedIntArray.h>

#include <vtkVolumeInputHelper.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>

#include "vtkOpenGLVolumeGradientOpacityTable.h"
#include "vtkOpenGLVolumeMaskGradientOpacityTransferFunction2D.h"
#include "vtkOpenGLVolumeMaskTransferFunction2D.h"
#include "vtkOpenGLVolumeOpacityTable.h"
#include "vtkOpenGLVolumeRGBTable.h"
#include "vtkOpenGLVolumeTransferFunction2D.h"

#include <vtkHardwareSelector.h>
#include <vtkVolumeMask.h>
#include <vtkVolumeProperty.h>
#include <vtkVolumeTexture.h>

// C/C++ includes
#include <cassert>
#include <limits>
#include <map>
#include <sstream>
#include <string>

#include <iostream>

VTK_ABI_NAMESPACE_BEGIN
vtkStandardNewMacro(vtkOpenGLGPUVolumeRayCastMapper);

//------------------------------------------------------------------------------
vtkOverrideAttribute* vtkOpenGLGPUVolumeRayCastMapper::CreateOverrideAttributes()
{
  return vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "OpenGL", nullptr);
}

//------------------------------------------------------------------------------
class vtkOpenGLGPUVolumeRayCastMapper::vtkInternal
{
public:
  // Constructor
  //--------------------------------------------------------------------------
  vtkInternal(vtkOpenGLGPUVolumeRayCastMapper* parent)
  {
    this->Parent = parent;
    this->ValidTransferFunction = false;
    this->LoadDepthTextureExtensionsSucceeded = false;
    this->CameraWasInsideInLastUpdate = false;
    this->CubeVBOId = 0;
    this->CubeVAOId = 0;
    this->CubeIndicesId = 0;
    this->DepthTextureObject = nullptr;
    this->DepthCopyColorTextureObject = nullptr;
    this->DepthCopyFBO = nullptr;
    this->SharedDepthTextureObject = false;
    this->TextureWidth = 1024;
    this->ActualSampleDistance = 1.0;
    this->CurrentMask = nullptr;
    this->TextureSize[0] = this->TextureSize[1] = this->TextureSize[2] = -1;
    this->WindowLowerLeft[0] = this->WindowLowerLeft[1] = 0;
    this->WindowSize[0] = this->WindowSize[1] = 0;
    this->LastDepthPassWindowSize[0] = this->LastDepthPassWindowSize[1] = 0;
    this->LastRenderToImageWindowSize[0] = 0;
    this->LastRenderToImageWindowSize[1] = 0;
    this->CurrentSelectionPass = vtkHardwareSelector::MIN_KNOWN_PASS - 1;

    this->TotalNumberOfLights = 0;
    this->NumberPositionalLights = 0;
    this->DefaultLighting = true;

    this->NeedToInitializeResources = false;
    this->ShaderCache = nullptr;

    this->FBO = nullptr;
    this->RTTDepthBufferTextureObject = nullptr;
    this->RTTDepthTextureObject = nullptr;
    this->RTTColorTextureObject = nullptr;
    this->RTTDepthTextureType = -1;

    this->DPFBO = nullptr;
    this->DPDepthBufferTextureObject = nullptr;
    this->DPColorTextureObject = nullptr;
    this->PreserveViewport = false;
    this->PreserveGLState = false;
    this->DepthMaskOverride = false;

    this->Partitions[0] = this->Partitions[1] = this->Partitions[2] = 1;

    // TEMP DEBUG: dump interpolated ray geometry when VTK_GL_RAY_DUMP is set.
    this->DebugRayDump = getenv("VTK_GL_RAY_DUMP") != nullptr;
  }

  // Destructor
  //--------------------------------------------------------------------------
  ~vtkInternal()
  {
    if (this->DepthTextureObject)
    {
      this->DepthTextureObject->Delete();
      this->DepthTextureObject = nullptr;
    }

    if (this->FBO)
    {
      this->FBO->Delete();
      this->FBO = nullptr;
    }

    if (this->RTTDepthBufferTextureObject)
    {
      this->RTTDepthBufferTextureObject->Delete();
      this->RTTDepthBufferTextureObject = nullptr;
    }

    if (this->RTTDepthTextureObject)
    {
      this->RTTDepthTextureObject->Delete();
      this->RTTDepthTextureObject = nullptr;
    }

    if (this->RTTColorTextureObject)
    {
      this->RTTColorTextureObject->Delete();
      this->RTTColorTextureObject = nullptr;
    }

    if (this->ImageSampleFBO)
    {
      this->ImageSampleFBO->Delete();
      this->ImageSampleFBO = nullptr;
    }

    for (auto& tex : this->ImageSampleTexture)
    {
      tex = nullptr;
    }
    this->ImageSampleTexture.clear();
    this->ImageSampleTexNames.clear();

    if (this->ImageSampleVAO)
    {
      this->ImageSampleVAO->Delete();
      this->ImageSampleVAO = nullptr;
    }
    this->DeleteMaskTransfer();

    // Do not delete the shader programs - Let the cache clean them up.
    this->ImageSampleProg = nullptr;
  }

  // Helper methods
  //--------------------------------------------------------------------------
  template <typename T>
  static void ToFloat(const T& in1, const T& in2, float (&out)[2]);
  template <typename T>
  static void ToFloat(const T& in1, const T& in2, const T& in3, float (&out)[3]);
  template <typename T>
  static void ToFloat(T* in, float* out, int noOfComponents);
  template <typename T>
  static void ToFloat(T (&in)[3], float (&out)[3]);
  template <unsigned int N, typename T>
  static std::array<float, N> ToFloat(T* in);
  template <typename T>
  static void ToFloat(T (&in)[2], float (&out)[2]);
  template <typename T>
  static void ToFloat(T& in, float& out);
  template <typename T>
  static void ToFloat(T (&in)[4][2], float (&out)[4][2]);
  template <typename T, int SizeX, int SizeY>
  static void CopyMatrixToVector(T* matrix, float* matrixVec, int offset);
  template <typename T, int SizeSrc>
  static void CopyVector(T* srcVec, T* dstVec, int offset);

  ///@{
  /**
   * \brief Setup and clean-up transfer functions for each vtkVolumeInputHelper
   * and masks.
   */
  void UpdateTransferFunctions(vtkRenderer* ren);

  void RefreshMaskTransfer(vtkRenderer* ren, VolumeInput& input);
  int UpdateMaskTransfer(vtkRenderer* ren, vtkVolume* vol, unsigned int component);
  void SetupMaskTransfer(vtkRenderer* ren);
  void ReleaseGraphicsMaskTransfer(vtkWindow* window);
  void DeleteMaskTransfer();
  ///@}

  void UpdateTransfer2DYAxisArray(vtkRenderer* ren, vtkVolume* vol);

  bool LoadMask(vtkRenderer* ren, vtkVolume* vol);

  // Update the depth sampler with the current state of the z-buffer. The
  // sampler is used for z-buffer compositing with opaque geometry during
  // ray-casting (rays are early-terminated if hidden begin opaque geometry).
  void CaptureDepthTexture(vtkRenderer* ren);

  // Test if camera is inside the volume geometry
  bool IsCameraInside(vtkRenderer* ren, vtkVolume* vol, double geometry[24]);

  ///@{
  /**
   * Update volume's proxy-geometry and draw it
   */
  bool IsGeometryUpdateRequired(vtkRenderer* ren, vtkVolume* vol, double geometry[24]);
  void RenderVolumeGeometry(
    vtkRenderer* ren, vtkShaderProgram* prog, vtkVolume* vol, double geometry[24]);
  // TEMP DEBUG: re-draw the proxy indices as 1px points (per-vertex clip dump).
  void RenderVolumeGeometryPoints(
    vtkRenderer* ren, vtkShaderProgram* prog, vtkVolume* vol, double geometry[24]);
  ///@}

  // Update cropping params to shader
  void SetCroppingRegions(vtkShaderProgram* prog, double loadedBounds[6]);

  // Update clipping params to shader
  void SetClippingPlanes(vtkRenderer* ren, vtkShaderProgram* prog, vtkVolume* vol);

  // Update the ray sampling distance. Sampling distance should be updated
  // before updating opacity transfer functions.
  void UpdateSamplingDistance(vtkRenderer* ren);

  // Check if the mapper should enter picking mode.
  void CheckPickingState(vtkRenderer* ren);

  // Look for property keys used to control the mapper's state.
  // This is necessary for some render passes which need to ensure
  // a specific OpenGL state when rendering through this mapper.
  void CheckPropertyKeys(vtkVolume* vol);

  // Configure the vtkHardwareSelector to begin a picking pass. This call
  // changes GL_BLEND, so it needs to be called before constructing
  // vtkVolumeStateRAII.
  void BeginPicking(vtkRenderer* ren);

  // Update the prop Id if hardware selection is enabled.
  void SetPickingId(vtkRenderer* ren);

  // Configure the vtkHardwareSelector to end a picking pass.
  void EndPicking(vtkRenderer* ren);

  // Load OpenGL extensiosn required to grab depth sampler buffer
  void LoadRequireDepthTextureExtensions(vtkRenderWindow* renWin);

  // Create GL buffers
  void CreateBufferObjects();

  // Dispose / free GL buffers
  void DeleteBufferObjects();

  // Convert vtkTextureObject to vtkImageData
  void ConvertTextureToImageData(vtkTextureObject* texture, vtkImageData* output);

  // Render to texture for final rendering
  void SetupRenderToTexture(vtkRenderer* ren);
  void ExitRenderToTexture(vtkRenderer* ren);

  // Render to texture for depth pass
  void SetupDepthPass(vtkRenderer* ren);
  void RenderContourPass(vtkRenderer* ren);
  void ExitDepthPass(vtkRenderer* ren);
  void RenderWithDepthPass(vtkRenderer* ren, vtkOpenGLCamera* cam, vtkMTimeType renderPassTime);

  void RenderSingleInput(vtkRenderer* ren, vtkOpenGLCamera* cam, vtkShaderProgram* prog);

  void RenderMultipleInputs(vtkRenderer* ren, vtkOpenGLCamera* cam, vtkShaderProgram* prog);

  ///@{
  /**
   * Update shader parameters.
   */
  void SetLightingShaderParameters(
    vtkRenderer* ren, vtkShaderProgram* prog, vtkVolume* vol, int numberOfSamplers);

  /**
   * Global parameters.
   */
  void SetMapperShaderParameters(
    vtkShaderProgram* prog, vtkRenderer* ren, int independent, int numComponents);

  /**
   * Per input data/ per component parameters.
   */
  void SetVolumeShaderParameters(
    vtkShaderProgram* prog, int independent, int noOfComponents, vtkMatrix4x4* modelViewMat);
  void BindTransformations(vtkShaderProgram* prog, vtkMatrix4x4* modelViewMat);

  /**
   * Transformation parameters.
   */
  void SetCameraShaderParameters(vtkShaderProgram* prog, vtkRenderer* ren, vtkOpenGLCamera* cam);

  /**
   * Feature specific.
   */
  void SetMaskShaderParameters(vtkShaderProgram* prog, vtkVolumeProperty* prop, int noOfComponents);
  void SetRenderToImageParameters(vtkShaderProgram* prog);
  void SetAdvancedShaderParameters(vtkRenderer* ren, vtkShaderProgram* prog, vtkVolume* vol,
    vtkVolumeTexture::VolumeBlock* block, int numComp);
  ///@}

  void FinishRendering(int numComponents);

  /**
   * TEMP DEBUG: dump the interpolated ray geometry (g_rayOrigin / g_dirStep)
   * for a handful of pixels so the Metal backend can reproduce the OpenGL
   * reference exactly. Enabled via the VTK_GL_RAY_DUMP environment variable
   * (the shader-side hook is injected in BuildShader).
   */
  void DumpDebugRays(vtkRenderer* ren, vtkShaderProgram* prog, vtkVolume* vol, double geometry[24]);

  /**
   * TEMP DEBUG: render the debug-injected shader into an RGBA32F framebuffer
   * and write the raw per-pixel interpolated attributes for the whole frame.
   * 3 passes: attr 0 = (ip_textureCoords.xyz, flatVid), 1 = ip_debugClip.xyzw,
   * 2 = (ip_vertexPos.xyz, gl_PrimitiveID). Each pass appends w*h*4 float32
   * RGBA values to the raw file (row 0 = gl_FragCoord y 0), one group per
   * frame. Enabled via VTK_GL_ATTR_DUMP (path override VTK_GL_ATTR_DUMP_OUT).
   */
  void DumpDebugAttrField(
    vtkRenderer* ren, vtkShaderProgram* prog, vtkVolume* vol, double geometry[24]);

  /**
   * TEMP DEBUG: re-render the clean volume shader into an RGBA32F framebuffer
   * and write the true final fragment floats to a raw binary file (RGBA,
   * row 0 = gl_FragCoord y 0). Enabled via VTK_GL_FLOAT_DUMP (path override
   * VTK_GL_FLOAT_DUMP_OUT). Deliberately independent of DebugRayDump so the
   * shader is NOT debug-injected and the floats are clean GL's own.
   */
  void DumpCleanGLFloats(
    vtkRenderer* ren, vtkShaderProgram* prog, vtkVolume* vol, double geometry[24]);

  vtkMTimeType LastModifiedLightTime(vtkLightCollection* lights);

  inline bool ShaderRebuildNeeded(
    vtkCamera* cam, vtkVolume* vol, vtkMTimeType renderPassTime, vtkRenderer* ren);
  bool VolumePropertyChanged = true;

  ///@{
  /**
   * Image XY-Sampling
   * Render to an internal framebuffer with lower resolution than the currently
   * bound one (hence casting less rays and improving performance). The rendered
   * image is subsequently rendered as a texture-mapped quad (linearly
   * interpolated) to the default (or previously attached) framebuffer. If a
   * vtkOpenGLRenderPass is attached, a variable number of render targets are
   * supported (as specified by the RenderPass). The render targets are assumed
   * to be ordered from GL_COLOR_ATTACHMENT0 to GL_COLOR_ATTACHMENT$N$, where
   * $N$ is the number of targets specified (targets of the previously bound
   * framebuffer as activated through ActivateDrawBuffers(int)). Without a
   * RenderPass attached, it relies on FramebufferObject to re-activate the
   * appropriate previous DrawBuffer.
   *
   * \sa vtkOpenGLRenderPass vtkOpenGLFramebufferObject
   */
  void BeginImageSample(vtkRenderer* ren);
  bool InitializeImageSampleFBO(vtkRenderer* ren);
  void EndImageSample(vtkRenderer* ren);
  size_t GetNumImageSampleDrawBuffers(vtkVolume* vol);
  ///@}

  ///@{
  /**
   * Allocate and update input data. A list of active ports is maintained
   * by the parent class. This list is traversed to update internal structures
   * used during rendering.
   */
  bool UpdateInputs(vtkRenderer* ren, vtkVolume* vol);

  /**
   * Cleanup resources of inputs that have been removed.
   */
  void ClearRemovedInputs(vtkWindow* win);

  /**
   * Forces transfer functions in all of the active vtkVolumeInputHelpers to
   * re-initialize in the next update. This is essential if the order in
   * AssembledInputs changes (inputs are added or removed), given that variable
   * names cached in vtkVolumeInputHelper instances are indexed.
   */
  void ForceTransferInit();
  ///@}

  vtkVolume* GetActiveVolume()
  {
    return this->MultiVolume ? this->MultiVolume : this->Parent->AssembledInputs[0].Volume;
  }
  int GetComponentMode(vtkVolumeProperty* prop, vtkDataArray* array) const;

  void ReleaseRenderToTextureGraphicsResources(vtkWindow* win);
  void ReleaseImageSampleGraphicsResources(vtkWindow* win);
  void ReleaseDepthPassGraphicsResources(vtkWindow* win);

  // Private member variables
  //--------------------------------------------------------------------------
  vtkOpenGLGPUVolumeRayCastMapper* Parent;

  bool ValidTransferFunction;
  bool LoadDepthTextureExtensionsSucceeded;
  bool CameraWasInsideInLastUpdate;

  GLuint CubeVBOId;
  GLuint CubeVAOId;
  GLuint CubeIndicesId;

  vtkTextureObject* DepthTextureObject;
  vtkTextureObject* DepthCopyColorTextureObject;
  vtkOpenGLFramebufferObject* DepthCopyFBO;
  bool SharedDepthTextureObject;

  int TextureWidth;

  float ActualSampleDistance;

  int LastProjectionParallel;
  int TextureSize[3];
  int WindowLowerLeft[2];
  int WindowSize[2];
  int LastDepthPassWindowSize[2];
  int LastRenderToImageWindowSize[2];

  int TotalNumberOfLights;
  bool DefaultLighting;
  int NumberPositionalLights;

  std::ostringstream ExtensionsStringStream;

  vtkSmartPointer<vtkOpenGLVolumeMaskTransferFunction2D> LabelMapTransfer2D;
  vtkSmartPointer<vtkOpenGLVolumeMaskGradientOpacityTransferFunction2D> LabelMapGradientOpacity;

  vtkTimeStamp ShaderBuildTime;

  vtkNew<vtkMatrix4x4> InverseProjectionMat;
  vtkNew<vtkMatrix4x4> InverseModelViewMat;
  vtkNew<vtkMatrix4x4> InverseVolumeMat;
  vtkNew<vtkMatrix4x4> InverseViewProjectionToDataMat;
  vtkNew<vtkMatrix4x4> InversePVMMat;

  vtkNew<vtkMatrix4x4> TempMatrix4x4;

  vtkSmartPointer<vtkPolyData> BBoxPolyData;
  vtkSmartPointer<vtkVolumeTexture> CurrentMask;

  vtkTimeStamp InitializationTime;
  vtkTimeStamp MaskUpdateTime;
  vtkTimeStamp ReleaseResourcesTime;
  vtkTimeStamp DepthPassTime;
  vtkTimeStamp DepthPassSetupTime;
  vtkTimeStamp SelectionStateTime;
  int CurrentSelectionPass;
  bool IsPicking;

  bool NeedToInitializeResources;
  bool PreserveViewport;
  bool PreserveGLState;
  bool DepthMaskOverride;

  vtkShaderProgram* ShaderProgram;
  vtkOpenGLShaderCache* ShaderCache;

  vtkOpenGLFramebufferObject* FBO;
  vtkTextureObject* RTTDepthBufferTextureObject;
  vtkTextureObject* RTTDepthTextureObject;
  vtkTextureObject* RTTColorTextureObject;
  int RTTDepthTextureType;

  vtkOpenGLFramebufferObject* DPFBO;
  vtkTextureObject* DPDepthBufferTextureObject;
  vtkTextureObject* DPColorTextureObject;

  vtkOpenGLFramebufferObject* ImageSampleFBO = nullptr;
  std::vector<vtkSmartPointer<vtkTextureObject>> ImageSampleTexture;
  std::vector<std::string> ImageSampleTexNames;
  vtkShaderProgram* ImageSampleProg = nullptr;
  vtkOpenGLVertexArrayObject* ImageSampleVAO = nullptr;
  size_t NumImageSampleDrawBuffers = 0;
  bool RebuildImageSampleProg = false;
  bool RenderPassAttached = false;

  bool Transfer2DUseGradient = true;
  vtkSmartPointer<vtkVolumeTexture> Transfer2DYAxisScalars;
  vtkTimeStamp Transfer2DYAxisScalarsUpdateTime;

  // TEMP DEBUG: dump interpolated ray geometry (VTK_GL_RAY_DUMP).
  bool DebugRayDump = false;

  vtkNew<vtkContourFilter> ContourFilter;
  vtkNew<vtkPolyDataMapper> ContourMapper;
  vtkNew<vtkActor> ContourActor;

  unsigned short Partitions[3];
  vtkMultiVolume* MultiVolume = nullptr;

  std::vector<float> VolMatVec, InvMatVec, TexMatVec, InvTexMatVec, TexEyeMatVec, CellToPointVec,
    TexMinVec, TexMaxVec, EyePosVec, ScaleVec, BiasVec, StepVec, SpacingVec, RangeVec;
};

//------------------------------------------------------------------------------
template <typename T>
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::ToFloat(
  const T& in1, const T& in2, float (&out)[2])
{
  out[0] = static_cast<float>(in1);
  out[1] = static_cast<float>(in2);
}

template <typename T>
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::ToFloat(
  const T& in1, const T& in2, const T& in3, float (&out)[3])
{
  out[0] = static_cast<float>(in1);
  out[1] = static_cast<float>(in2);
  out[2] = static_cast<float>(in3);
}

//------------------------------------------------------------------------------
template <typename T>
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::ToFloat(T* in, float* out, int noOfComponents)
{
  for (int i = 0; i < noOfComponents; ++i)
  {
    out[i] = static_cast<float>(in[i]);
  }
}

//------------------------------------------------------------------------------
template <typename T>
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::ToFloat(T (&in)[3], float (&out)[3])
{
  out[0] = static_cast<float>(in[0]);
  out[1] = static_cast<float>(in[1]);
  out[2] = static_cast<float>(in[2]);
}

//------------------------------------------------------------------------------
template <unsigned N, typename T>
std::array<float, N> vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::ToFloat(T* in)
{
  std::array<float, N> out;
  for (size_t i = 0; i < N; i++)
  {
    out[i] = static_cast<float>(in[i]);
  }
  return out;
}

//------------------------------------------------------------------------------
template <typename T>
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::ToFloat(T (&in)[2], float (&out)[2])
{
  out[0] = static_cast<float>(in[0]);
  out[1] = static_cast<float>(in[1]);
}

//------------------------------------------------------------------------------
template <typename T>
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::ToFloat(T& in, float& out)
{
  out = static_cast<float>(in);
}

//------------------------------------------------------------------------------
template <typename T>
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::ToFloat(T (&in)[4][2], float (&out)[4][2])
{
  out[0][0] = static_cast<float>(in[0][0]);
  out[0][1] = static_cast<float>(in[0][1]);
  out[1][0] = static_cast<float>(in[1][0]);
  out[1][1] = static_cast<float>(in[1][1]);
  out[2][0] = static_cast<float>(in[2][0]);
  out[2][1] = static_cast<float>(in[2][1]);
  out[3][0] = static_cast<float>(in[3][0]);
  out[3][1] = static_cast<float>(in[3][1]);
}

//------------------------------------------------------------------------------
template <typename T, int SizeX, int SizeY>
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::CopyMatrixToVector(
  T* matrix, float* matrixVec, int offset)
{
  constexpr int MatSize = SizeX * SizeY;
  for (int j = 0; j < MatSize; j++)
  {
    matrixVec[offset + j] = matrix->Element[j / SizeX][j % SizeY];
  }
}

//------------------------------------------------------------------------------
template <typename T, int SizeSrc>
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::CopyVector(T* srcVec, T* dstVec, int offset)
{
  for (int j = 0; j < SizeSrc; j++)
  {
    dstVec[offset + j] = srcVec[j];
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::SetupMaskTransfer(vtkRenderer* ren)
{
  this->ReleaseGraphicsMaskTransfer(ren->GetRenderWindow());
  this->DeleteMaskTransfer();

  if (this->Parent->MaskInput != nullptr && this->Parent->MaskType == LabelMapMaskType &&
    !this->LabelMapTransfer2D)
  {
    this->LabelMapTransfer2D = vtkSmartPointer<vtkOpenGLVolumeMaskTransferFunction2D>::New();
    this->LabelMapGradientOpacity =
      vtkSmartPointer<vtkOpenGLVolumeMaskGradientOpacityTransferFunction2D>::New();
  }

  this->InitializationTime.Modified();
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::RefreshMaskTransfer(
  vtkRenderer* ren, VolumeInput& input)
{
  auto vol = input.Volume;
  if (this->NeedToInitializeResources ||
    input.Volume->GetProperty()->GetMTime() > this->InitializationTime.GetMTime())
  {
    this->SetupMaskTransfer(ren);
  }
  this->UpdateMaskTransfer(ren, vol, 0);
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::UpdateTransferFunctions(vtkRenderer* ren)
{
  int uniformIndex = 0;
  for (const auto& port : this->Parent->Ports)
  {
    auto& input = this->Parent->AssembledInputs[port];
    input.ColorRangeType = this->Parent->GetColorRangeType();
    input.ScalarOpacityRangeType = this->Parent->GetScalarOpacityRangeType();
    input.GradientOpacityRangeType = this->Parent->GetGradientOpacityRangeType();
    input.RefreshTransferFunction(
      ren, uniformIndex, this->Parent->BlendMode, this->ActualSampleDistance);

    uniformIndex++;
  }

  if (!this->MultiVolume)
  {
    this->RefreshMaskTransfer(ren, this->Parent->AssembledInputs[0]);
  }
}

//------------------------------------------------------------------------------
bool vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::LoadMask(vtkRenderer* ren, vtkVolume* vol)
{
  bool result = true;
  auto maskInput = this->Parent->MaskInput;
  if (maskInput)
  {
    if (!this->CurrentMask)
    {
      this->CurrentMask = vtkSmartPointer<vtkVolumeTexture>::New();

      const auto part = this->Partitions;
      this->CurrentMask->SetPartitions(part[0], part[1], part[2]);
    }

    int isCellData;
    vtkDataArray* arr =
      vtkOpenGLGPUVolumeRayCastMapper::GetScalars(maskInput, this->Parent->ScalarMode,
        this->Parent->ArrayAccessMode, this->Parent->ArrayId, this->Parent->ArrayName, isCellData);
    if (maskInput->GetMTime() > this->MaskUpdateTime ||
      this->CurrentMask->GetLoadedScalars() != arr ||
      (arr && arr->GetMTime() > this->MaskUpdateTime))
    {
      // Setup the scalar range of the mask volume based on the number of transfer functions in the
      // property. This is done so that the value for texture lookup in the shader is scaled and
      // biased based on the range of the texture created by the label transfer functions.
      vtkVolumeProperty* volumeProperty = vol->GetProperty();
      auto const numLabels = volumeProperty->GetLabelMapLabels().size();
      double maskRange[2] = { 0.0, (numLabels > 0.0 ? numLabels : 1.0) };
      vtkNew<vtkInformationVector> infoVec;
      infoVec->SetNumberOfInformationObjects(1);
      infoVec->GetInformationObject(0)->Set(vtkDataArray::COMPONENT_RANGE(), maskRange, 2);
      arr->GetInformation()->Set(vtkDataArray::PER_FINITE_COMPONENT(), infoVec);
      result =
        this->CurrentMask->LoadVolume(ren, maskInput, arr, isCellData, VTK_NEAREST_INTERPOLATION);

      this->MaskUpdateTime.Modified();
    }
  }

  return result;
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::ReleaseGraphicsMaskTransfer(vtkWindow* window)
{
  if (this->LabelMapTransfer2D)
  {
    this->LabelMapTransfer2D->ReleaseGraphicsResources(window);
  }
  if (this->LabelMapGradientOpacity)
  {
    this->LabelMapGradientOpacity->ReleaseGraphicsResources(window);
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::DeleteMaskTransfer()
{
  this->LabelMapTransfer2D = nullptr;
  this->LabelMapGradientOpacity = nullptr;
}

//------------------------------------------------------------------------------
int vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::UpdateMaskTransfer(
  vtkRenderer* ren, vtkVolume* vol, unsigned int component)
{
  vtkVolumeProperty* volumeProperty = vol->GetProperty();

  auto volumeTex = this->Parent->AssembledInputs[0].Texture.GetPointer();
  double componentRange[2];
  for (int i = 0; i < 2; ++i)
  {
    componentRange[i] = volumeTex->ScalarRange[component][i];
  }

  if (this->Parent->MaskInput != nullptr && this->Parent->MaskType == LabelMapMaskType)
  {
    this->LabelMapTransfer2D->Update(volumeProperty, componentRange, 0, 0, 0,
      vtkTextureObject::Nearest, vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow()));

    if (volumeProperty->HasLabelGradientOpacity())
    {
      this->LabelMapGradientOpacity->Update(volumeProperty, componentRange, 0, 0, 0,
        vtkTextureObject::Nearest, vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow()));
    }
  }

  return 0;
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::CaptureDepthTexture(vtkRenderer* ren)
{
  // Make sure our render window is the current OpenGL context
  ren->GetRenderWindow()->MakeCurrent();

  // Load required extensions for grabbing depth sampler buffer
  if (!this->LoadDepthTextureExtensionsSucceeded)
  {
    this->LoadRequireDepthTextureExtensions(ren->GetRenderWindow());
  }

  // If we can't load the necessary extensions, provide
  // feedback on why it failed.
  if (!this->LoadDepthTextureExtensionsSucceeded)
  {
    std::cerr << this->ExtensionsStringStream.str() << std::endl;
    return;
  }

  if (!this->DepthTextureObject)
  {
    this->DepthTextureObject = vtkTextureObject::New();
    this->DepthCopyColorTextureObject = vtkTextureObject::New();
  }

  vtkOpenGLRenderWindow* orenWin = vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow());
  this->DepthTextureObject->SetContext(orenWin);
  this->DepthCopyColorTextureObject->SetContext(orenWin);

  //  this->DepthTextureObject->Activate();
  if (!this->DepthTextureObject->GetHandle())
  {
    // First set the parameters
    this->DepthTextureObject->SetWrapS(vtkTextureObject::Repeat);
    this->DepthTextureObject->SetWrapT(vtkTextureObject::Repeat);
    // GLES3 specificity: can't use Linear filtering when the texture array's internal format is not
    // texture-filterable (In the specification: 3.8.13, Texture Completeness)
#ifdef GL_ES_VERSION_3_0
    this->DepthTextureObject->SetMagnificationFilter(vtkTextureObject::Nearest);
    this->DepthTextureObject->SetMinificationFilter(vtkTextureObject::Nearest);
#else
    this->DepthTextureObject->SetMagnificationFilter(vtkTextureObject::Linear);
    this->DepthTextureObject->SetMinificationFilter(vtkTextureObject::Linear);
#endif

    if (orenWin->GetStencilCapable())
    {
      this->DepthTextureObject->AllocateDepthStencil(this->WindowSize[0], this->WindowSize[1]);
    }
    else
    {
      // For now, the format is set by default to GL_DEPTH_COMPONENT24.
      // This should be configurable in the future
      // See https://gitlab.kitware.com/vtk/vtk/-/issues/19823
#ifdef GL_ES_VERSION_3_0
      this->DepthTextureObject->AllocateDepth(this->WindowSize[0], this->WindowSize[1], 3);
#else
      this->DepthTextureObject->AllocateDepth(this->WindowSize[0], this->WindowSize[1], 4);
#endif
    }
  }

  if (!this->DepthCopyColorTextureObject->GetHandle())
  {
    // First set the parameters
    this->DepthCopyColorTextureObject->SetWrapS(vtkTextureObject::Repeat);
    this->DepthCopyColorTextureObject->SetWrapT(vtkTextureObject::Repeat);
    this->DepthCopyColorTextureObject->SetMagnificationFilter(vtkTextureObject::Linear);
    this->DepthCopyColorTextureObject->SetMinificationFilter(vtkTextureObject::Linear);
    this->DepthCopyColorTextureObject->Allocate2D(
      this->WindowSize[0], this->WindowSize[1], 4, VTK_UNSIGNED_CHAR);
  }
  this->DepthTextureObject->Resize(this->WindowSize[0], this->WindowSize[1]);
  this->DepthCopyColorTextureObject->Resize(this->WindowSize[0], this->WindowSize[1]);

  // copy depth with a blit
  if (!this->DepthCopyFBO)
  {
    this->DepthCopyFBO = vtkOpenGLFramebufferObject::New();
    this->DepthCopyFBO->SetContext(orenWin);
    orenWin->GetState()->PushDrawFramebufferBinding();
    this->DepthCopyFBO->Bind(GL_DRAW_FRAMEBUFFER);
    this->DepthCopyFBO->AddDepthAttachment(this->DepthTextureObject);
    this->DepthCopyFBO->AddColorAttachment(0, this->DepthCopyColorTextureObject);
  }
  else
  {
    orenWin->GetState()->PushDrawFramebufferBinding();
  }

  this->DepthCopyFBO->Bind(GL_DRAW_FRAMEBUFFER);
  orenWin->GetState()->vtkglBlitFramebuffer(this->WindowLowerLeft[0], this->WindowLowerLeft[1],
    this->WindowLowerLeft[0] + this->WindowSize[0], this->WindowLowerLeft[1] + this->WindowSize[1],
    0, 0, this->WindowSize[0], this->WindowSize[1], GL_DEPTH_BUFFER_BIT, GL_NEAREST);

  orenWin->GetState()->PopDrawFramebufferBinding();
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::SetLightingShaderParameters(
  vtkRenderer* ren, vtkShaderProgram* prog, vtkVolume* vol, int numberOfSamplers)
{
  // Set basic lighting parameters (per component)
  if (!ren || !prog || !vol)
  {
    return;
  }

  auto volumeProperty = vol->GetProperty();
  float ambient[4][3];
  float diffuse[4][3];
  float specular[4][3];
  float specularPower[4];

  for (int i = 0; i < numberOfSamplers; ++i)
  {
    ambient[i][0] = ambient[i][1] = ambient[i][2] = volumeProperty->GetAmbient(i);
    diffuse[i][0] = diffuse[i][1] = diffuse[i][2] = volumeProperty->GetDiffuse(i);
    specular[i][0] = specular[i][1] = specular[i][2] = volumeProperty->GetSpecular(i);
    specularPower[i] = volumeProperty->GetSpecularPower(i);
  }

  prog->SetUniform3fv("in_ambient", numberOfSamplers, ambient);
  prog->SetUniform3fv("in_diffuse", numberOfSamplers, diffuse);
  prog->SetUniform3fv("in_specular", numberOfSamplers, specular);
  prog->SetUniform1fv("in_shininess", numberOfSamplers, specularPower);

  // Set advanced lighting features
  if ((vol && !vol->GetProperty()->GetShade()) || this->TotalNumberOfLights == 0)
  {
    return;
  }

  prog->SetUniformi("in_twoSidedLighting", ren->GetTwoSidedLighting());

  // Shadows extent parameter
  if (this->Parent->GetVolumetricScatteringBlending() > 0.0)
  {
    float shadowExtent = this->Parent->GetGlobalIlluminationReach();
    // we map the shadow extent from [0, 1] to [sampleDistance, sqrt(3)]
    // 0.1 corresponds to the minimum length of a shadow ray (the texture unit cube has size 1)
    // sqrt(3) corresponds to the maximum (the diagonal of the cube)
    float* invTexMat = this->InvTexMatVec.data();
    float minCoef = VTK_FLOAT_MAX;
    // only take 3x3 sub-matrix because it will be multiplied by a vec4(..., 0.0) normalized vec
    for (int i = 0; i < 3; i++)
    {
      // diagonal coefficient
      minCoef = std::min(minCoef, std::abs(invTexMat[5 * i]));
    }
    float minExtent = minCoef * this->ActualSampleDistance;
    constexpr float maxExtent = 1.73205;
    shadowExtent = (minExtent - maxExtent) * std::pow(1.0 - shadowExtent, 0.33) + maxExtent;
    prog->SetUniformf("in_giReach", shadowExtent);
  }

  // for lightkit case there are some parameters to set
  vtkCamera* cam = ren->GetActiveCamera();
  vtkTransform* viewTF = cam->GetModelViewTransformObject();

  // Bind some light settings
  vtkLightCollection* lc = ren->GetLights();
  vtkLight* light;

  vtkCollectionSimpleIterator sit;
  // those light parameters are used by both positional and directional lights
  std::vector<std::array<float, 3>> lightAmbientColor(this->TotalNumberOfLights);
  std::vector<std::array<float, 3>> lightDiffuseColor(this->TotalNumberOfLights);
  std::vector<std::array<float, 3>> lightSpecularColor(this->TotalNumberOfLights);
  std::vector<std::array<float, 3>> lightDirection(this->TotalNumberOfLights);
  // so we have TotalNumberOfLights 3-vectors by parameter, and we organize them :
  // [positionalLight1,..., positionalLight_M, directionalLight1, ...]
  int idxPositional = 0;
  int idxDirectional = this->NumberPositionalLights;
  int idxLight = 0;
  for (lc->InitTraversal(sit); (light = lc->GetNextLight(sit));)
  {
    float status = light->GetSwitch();
    if (status > 0.0)
    {
      idxLight = light->GetPositional() ? idxPositional : idxDirectional;
      double* aColor = light->GetAmbientColor();
      double* dColor = light->GetDiffuseColor();
      double* sColor = light->GetSpecularColor();
      double intensity = light->GetIntensity();
      for (int i = 0; i < 3; i++)
      {
        lightAmbientColor[idxLight][i] = aColor[i] * intensity;
        lightDiffuseColor[idxLight][i] = dColor[i] * intensity;
        lightSpecularColor[idxLight][i] = sColor[i] * intensity;
      }
      // Get required info from light
      double* lfp = light->GetTransformedFocalPoint();
      double* lp = light->GetTransformedPosition();
      double lightDir[3];
      vtkMath::Subtract(lfp, lp, lightDir);
      vtkMath::Normalize(lightDir);
      double* tDir = viewTF->TransformNormal(lightDir);
      lightDirection[idxLight] = ToFloat<3>(tDir);
      light->GetPositional() ? idxPositional++ : idxDirectional++;
    }
  }

  prog->SetUniform3fv(
    "in_lightAmbientColor", this->TotalNumberOfLights, lightAmbientColor.data()->data());
  prog->SetUniform3fv(
    "in_lightDiffuseColor", this->TotalNumberOfLights, lightDiffuseColor.data()->data());
  prog->SetUniform3fv(
    "in_lightSpecularColor", this->TotalNumberOfLights, lightSpecularColor.data()->data());
  prog->SetUniform3fv(
    "in_lightDirection", this->TotalNumberOfLights, lightDirection.data()->data());

  // we are done if we only had default lighting
  if (this->DefaultLighting)
  {
    return;
  }

  // if positional lights pass down more parameters
  // these parameters are only used by positional lights,
  // so there are only NumberPositionalLights of them
  if (this->NumberPositionalLights > 0)
  {
    std::vector<std::array<float, 3>> lightAttenuation(this->NumberPositionalLights);
    std::vector<std::array<float, 3>> lightPosition(this->NumberPositionalLights);
    std::vector<float> lightConeAngle(this->NumberPositionalLights);
    std::vector<float> lightExponent(this->NumberPositionalLights);
    idxPositional = 0;
    for (lc->InitTraversal(sit); (light = lc->GetNextLight(sit));)
    {
      float status = light->GetSwitch();
      if (status > 0.0 && light->GetPositional())
      {
        double* attn = light->GetAttenuationValues();
        lightAttenuation[idxPositional] = ToFloat<3>(attn);
        lightExponent[idxPositional] = light->GetExponent();
        lightConeAngle[idxPositional] = light->GetConeAngle();
        double* lp = light->GetTransformedPosition();
        double* tlp = viewTF->TransformPoint(lp);
        lightPosition[idxPositional] = ToFloat<3>(tlp);
        idxPositional++;
      }
    }
    prog->SetUniform3fv(
      "in_lightAttenuation", this->NumberPositionalLights, lightAttenuation.data()->data());
    prog->SetUniform3fv(
      "in_lightPosition", this->NumberPositionalLights, lightPosition.data()->data());
    prog->SetUniform1fv("in_lightExponent", this->NumberPositionalLights, lightExponent.data());
    prog->SetUniform1fv("in_lightConeAngle", this->NumberPositionalLights, lightConeAngle.data());
  }
}

//------------------------------------------------------------------------------
bool vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::IsCameraInside(
  vtkRenderer* ren, vtkVolume* vol, double geometry[24])
{
  vtkNew<vtkMatrix4x4> dataToWorld;
  vol->GetModelToWorldMatrix(dataToWorld);

  vtkCamera* cam = ren->GetActiveCamera();

  double planes[24];
  cam->GetFrustumPlanes(ren->GetTiledAspectRatio(), planes);

  // convert geometry to world then compare to frustum planes
  double in[4];
  in[3] = 1.0;
  double out[4];
  double worldGeometry[24];
  for (int i = 0; i < 8; ++i)
  {
    in[0] = geometry[i * 3];
    in[1] = geometry[i * 3 + 1];
    in[2] = geometry[i * 3 + 2];
    dataToWorld->MultiplyPoint(in, out);
    worldGeometry[i * 3] = out[0] / out[3];
    worldGeometry[i * 3 + 1] = out[1] / out[3];
    worldGeometry[i * 3 + 2] = out[2] / out[3];
  }

  // does the front clipping plane intersect the volume?
  // true if points are on both sides of the plane
  bool hasPositive = false;
  bool hasNegative = false;
  bool hasZero = false;
  for (int i = 0; i < 8; ++i)
  {
    double val = planes[4 * 4] * worldGeometry[i * 3] +
      planes[4 * 4 + 1] * worldGeometry[i * 3 + 1] + planes[4 * 4 + 2] * worldGeometry[i * 3 + 2] +
      planes[4 * 4 + 3];
    if (val < 0)
    {
      hasNegative = true;
    }
    else if (val > 0)
    {
      hasPositive = true;
    }
    else
    {
      hasZero = true;
    }
  }

  return hasZero || (hasNegative && hasPositive);
}

//------------------------------------------------------------------------------
bool vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::IsGeometryUpdateRequired(
  vtkRenderer* ren, vtkVolume* vol, double geometry[24])
{
  if (!this->BBoxPolyData)
  {
    return true;
  }

  const auto GeomTime = this->BBoxPolyData->GetMTime();
  const bool uploadTimeChanged =
    any_of(this->Parent->AssembledInputs.begin(), this->Parent->AssembledInputs.end(),
      [&GeomTime](const std::pair<int, vtkVolumeInputHelper>& item)
      { return item.second.Texture->UploadTime > GeomTime; });

  return (this->NeedToInitializeResources || uploadTimeChanged ||
    this->IsCameraInside(ren, vol, geometry) || this->CameraWasInsideInLastUpdate ||
    (this->MultiVolume && this->MultiVolume->GetBoundsTime() > this->BBoxPolyData->GetMTime()));
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::RenderVolumeGeometry(
  vtkRenderer* ren, vtkShaderProgram* prog, vtkVolume* vol, double geometry[24])
{
  if (this->IsGeometryUpdateRequired(ren, vol, geometry))
  {
    vtkNew<vtkPolyData> boxSource;

    {
      vtkNew<vtkCellArray> cells;
      vtkNew<vtkPoints> points;
      points->SetDataTypeToDouble();
      for (int i = 0; i < 8; ++i)
      {
        points->InsertNextPoint(geometry + i * 3);
      }
      // 6 faces 12 triangles
      int tris[36] = {
        0, 1, 2, //
        1, 3, 2, //
        1, 5, 3, //
        5, 7, 3, //
        5, 4, 7, //
        4, 6, 7, //
        4, 0, 6, //
        0, 2, 6, //
        2, 3, 6, //
        3, 7, 6, //
        0, 4, 1, //
        1, 4, 5  //
      };
      for (int i = 0; i < 12; ++i)
      {
        cells->InsertNextCell(3);
        // this code uses a clockwise convention for some reason
        // no clue why but the ClipConvexPolyData assumes the same
        // so we add verts as 0 2 1 instead of 0 1 2
        cells->InsertCellPoint(tris[i * 3]);
        cells->InsertCellPoint(tris[i * 3 + 2]);
        cells->InsertCellPoint(tris[i * 3 + 1]);
      }
      boxSource->SetPoints(points);
      boxSource->SetPolys(cells);
    }

    vtkNew<vtkDensifyPolyData> densifyPolyData;
    if (this->IsCameraInside(ren, vol, geometry))
    {
      vtkNew<vtkMatrix4x4> dataToWorld;
      vol->GetModelToWorldMatrix(dataToWorld);

      vtkCamera* cam = ren->GetActiveCamera();
      double pos[3], fp[3], up[3];
      cam->GetPosition(pos);
      cam->GetFocalPoint(fp);
      cam->GetViewUp(up);
      std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_CAM position=(" << pos[0] << ", " << pos[1]
                << ", " << pos[2] << ") focal=(" << fp[0] << ", " << fp[1] << ", " << fp[2]
                << ") up=(" << up[0] << ", " << up[1] << ", " << up[2] << ") viewAngle="
                << cam->GetViewAngle() << " clipRange=(" << cam->GetClippingRange()[0] << ", "
                << cam->GetClippingRange()[1] << ")" << std::endl;

      double fplanes[24];
      cam->GetFrustumPlanes(ren->GetTiledAspectRatio(), fplanes);

      // have to convert the 5th plane to volume coordinates
      double pOrigin[4];
      pOrigin[3] = 1.0;
      double pNormal[3];
      for (int i = 0; i < 3; ++i)
      {
        pNormal[i] = fplanes[16 + i];
        pOrigin[i] = -fplanes[16 + 3] * fplanes[16 + i];
      }

      // convert the normal
      double* dmat = dataToWorld->GetData();
      dataToWorld->Transpose();
      double pNormalV[3];
      pNormalV[0] = pNormal[0] * dmat[0] + pNormal[1] * dmat[1] + pNormal[2] * dmat[2];
      pNormalV[1] = pNormal[0] * dmat[4] + pNormal[1] * dmat[5] + pNormal[2] * dmat[6];
      pNormalV[2] = pNormal[0] * dmat[8] + pNormal[1] * dmat[9] + pNormal[2] * dmat[10];
      vtkMath::Normalize(pNormalV);

      // convert the point
      dataToWorld->Transpose();
      dataToWorld->Invert();
      dataToWorld->MultiplyPoint(pOrigin, pOrigin);

      vtkNew<vtkPlane> nearPlane;

      // We add an offset to the near plane to avoid hardware clipping of the
      // near plane due to floating-point precision.
      // camPlaneNormal is a unit vector, if the offset is larger than the
      // distance between near and far point, it will not work. Hence, we choose
      // a fraction of the near-far distance. However, care should be taken
      // to avoid hardware clipping in volumes with very small spacing where the
      // distance between near and far plane is also very small. In that case,
      // a minimum offset is chosen. This is chosen based on the typical
      // epsilon values on x86 systems.
      double offset = (cam->GetClippingRange()[1] - cam->GetClippingRange()[0]) * 0.001;
      // Minimum offset to avoid floating point precision issues for volumes
      // with very small spacing
      double minOffset = static_cast<double>(std::numeric_limits<float>::epsilon()) * 1000.0;
      offset = offset < minOffset ? minOffset : offset;

      for (int i = 0; i < 3; ++i)
      {
        pOrigin[i] += (pNormalV[i] * offset);
      }

      nearPlane->SetOrigin(pOrigin);
      nearPlane->SetNormal(pNormalV);
      std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_NEARPLANE origin=(" << pOrigin[0] << ", "
                << pOrigin[1] << ", " << pOrigin[2] << ") normal=(" << pNormalV[0] << ", "
                << pNormalV[1] << ", " << pNormalV[2] << ")" << std::endl;

      vtkNew<vtkPlaneCollection> planes;
      planes->RemoveAllItems();
      planes->AddItem(nearPlane);

      vtkNew<vtkClipConvexPolyData> clip;
      clip->SetInputData(boxSource);
      clip->SetPlanes(planes);

      densifyPolyData->SetInputConnection(clip->GetOutputPort());

      this->CameraWasInsideInLastUpdate = true;
    }
    else
    {
      densifyPolyData->SetInputData(boxSource);
      this->CameraWasInsideInLastUpdate = false;
    }

    densifyPolyData->SetNumberOfSubdivisions(2);
    densifyPolyData->Update();

    this->BBoxPolyData = vtkSmartPointer<vtkPolyData>::New();
    this->BBoxPolyData->ShallowCopy(densifyPolyData->GetOutput());
    vtkPoints* points = this->BBoxPolyData->GetPoints();
    vtkCellArray* cells = this->BBoxPolyData->GetPolys();

    vtkNew<vtkUnsignedIntArray> polys;
    polys->SetNumberOfComponents(3);
    vtkIdType npts;
    const vtkIdType* pts;

    // See if the volume transform is orientation-preserving
    // and orient polygons accordingly
    vol->GetModelToWorldMatrix(this->TempMatrix4x4);
    vtkMatrix4x4* volMat = this->TempMatrix4x4;
    double det = vtkMath::Determinant3x3(volMat->GetElement(0, 0), volMat->GetElement(0, 1),
      volMat->GetElement(0, 2), volMat->GetElement(1, 0), volMat->GetElement(1, 1),
      volMat->GetElement(1, 2), volMat->GetElement(2, 0), volMat->GetElement(2, 1),
      volMat->GetElement(2, 2));
    bool preservesOrientation = det > 0.0;

    const vtkIdType indexMap[3] = { preservesOrientation ? 0 : 2, 1, preservesOrientation ? 2 : 0 };

    while (cells->GetNextCell(npts, pts))
    {
      polys->InsertNextTuple3(pts[indexMap[0]], pts[indexMap[1]], pts[indexMap[2]]);
    }

    // TEMP DEBUG (probe #1): dump the uploaded cap-triangle vertex attributes and
    // indices as float32 bits for Metal byte-compare.
    {
      vtkCamera* cam = ren->GetActiveCamera();
      double pos[3], fp[3], up[3];
      cam->GetPosition(pos);
      cam->GetFocalPoint(fp);
      cam->GetViewUp(up);
      std::cerr << std::setprecision(9)
                << "VTK_METAL_VOLUME_LOG DEBUG GL_CAPMESH cam=(" << pos[0] << ", " << pos[1]
                << ", " << pos[2] << ") focal=(" << fp[0] << ", " << fp[1] << ", " << fp[2]
                << ") viewAngle=" << cam->GetViewAngle() << " clipRange=("
                << cam->GetClippingRange()[0] << ", " << cam->GetClippingRange()[1] << ")"
                << std::endl;
      auto* fpts = vtkAOSDataArrayTemplate<float>::FastDownCast(points->GetData());
      std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_CAPVERTS n=" << points->GetNumberOfPoints()
                << " dt=" << points->GetData()->GetDataType() << " comp="
                << points->GetData()->GetNumberOfComponents() << " floats=" << (fpts ? 1 : 0)
                << std::endl;
      for (vtkIdType i = 0; fpts && i < points->GetNumberOfPoints(); ++i)
      {
        const float* p = fpts->GetPointer(0) + i * 3;
        uint32_t b0, b1, b2;
        std::memcpy(&b0, p + 0, 4);
        std::memcpy(&b1, p + 1, 4);
        std::memcpy(&b2, p + 2, 4);
        std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_CAPVERT " << i << " 0x" << std::hex
                  << std::setw(8) << std::setfill('0') << b0 << " 0x" << std::setw(8) << b1
                  << " 0x" << std::setw(8) << b2 << std::dec << std::setfill(' ') << std::endl;
      }
      std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_CAPINDICES n=" << polys->GetNumberOfTuples()
                << std::endl;
      for (vtkIdType i = 0; i < polys->GetNumberOfTuples(); ++i)
      {
        const unsigned int* t = polys->GetPointer(0) + i * 3;
        std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_CAPINDEX " << i << " " << t[0] << " " << t[1]
                  << " " << t[2] << std::endl;
      }
    }

    // Dispose any previously created buffers
    this->DeleteBufferObjects();

    // Now create new ones
    this->CreateBufferObjects();

    // TODO: should really use the built in VAO class
    glBindVertexArray(this->CubeVAOId);

    // Pass cube vertices to buffer object memory
    glBindBuffer(GL_ARRAY_BUFFER, this->CubeVBOId);
    glBufferData(GL_ARRAY_BUFFER,
      points->GetData()->GetDataSize() * points->GetData()->GetDataTypeSize(),
      vtkAOSDataArrayTemplate<float>::FastDownCast(points->GetData())->GetPointer(0),
      GL_STATIC_DRAW);

    prog->EnableAttributeArray("in_vertexPos");
    prog->UseAttributeArray("in_vertexPos", 0, 0, VTK_FLOAT, 3, vtkShaderProgram::NoNormalize);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->CubeIndicesId);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, polys->GetDataSize() * polys->GetDataTypeSize(),
      polys->GetPointer(0), GL_STATIC_DRAW);
  }
  else
  {
    glBindVertexArray(this->CubeVAOId);
  }

  glDrawElements(
    GL_TRIANGLES, this->BBoxPolyData->GetNumberOfCells() * 3, GL_UNSIGNED_INT, nullptr);

  vtkOpenGLStaticCheckErrorMacro("Error after glDrawElements in"
                                 " RenderVolumeGeometry!");
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::RenderVolumeGeometryPoints(
  vtkRenderer* ren, vtkShaderProgram* prog, vtkVolume* vol, double geometry[24])
{
  (void)ren;
  (void)vol;
  (void)geometry;
  // TEMP DEBUG: GL_POINTS does not rasterize on this macOS GL implementation, so
  // draw each cap vertex as a tiny triangle. Every vertex is duplicated 3x; the
  // vertex shader (in_debugVertexMode=1) maps the 3 copies to a fixed 2px
  // triangle whose position is a linear function of k=gl_VertexID/3, and passes
  // the real clip value through ip_debugClip (constant over the triangle). The
  // clip values are then read back from the strip of tiny triangles.
  vtkPoints* pts = this->BBoxPolyData->GetPoints();
  const vtkIdType n = pts->GetNumberOfPoints();
  std::vector<float> tri(static_cast<size_t>(n) * 3 * 3);
  for (vtkIdType k = 0; k < n; ++k)
  {
    double* p = pts->GetPoint(k);
    for (int c = 0; c < 3; ++c)
    {
      tri[(static_cast<size_t>(k) * 3 + c) * 3 + 0] = static_cast<float>(p[0]);
      tri[(static_cast<size_t>(k) * 3 + c) * 3 + 1] = static_cast<float>(p[1]);
      tri[(static_cast<size_t>(k) * 3 + c) * 3 + 2] = static_cast<float>(p[2]);
    }
  }
  glBindVertexArray(this->CubeVAOId);
  glBindBuffer(GL_ARRAY_BUFFER, this->CubeVBOId);
  glBufferData(GL_ARRAY_BUFFER,
    static_cast<GLsizeiptr>(tri.size() * sizeof(float)), tri.data(), GL_STREAM_DRAW);
  prog->EnableAttributeArray("in_vertexPos");
  prog->UseAttributeArray("in_vertexPos", 0, 0, VTK_FLOAT, 3, vtkShaderProgram::NoNormalize);
  prog->SetUniformi("in_debugVertexMode", 1);
  prog->SetUniformi("in_debugVertexCount", static_cast<int>(n));
  glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(n * 3));
  prog->SetUniformi("in_debugVertexMode", 0);
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::SetCroppingRegions(
  vtkShaderProgram* prog, double loadedBounds[6])
{
  if (this->Parent->GetCropping())
  {
    int cropFlags = this->Parent->GetCroppingRegionFlags();
    double croppingRegionPlanes[6];
    this->Parent->GetCroppingRegionPlanes(croppingRegionPlanes);

    // Clamp it
    croppingRegionPlanes[0] =
      croppingRegionPlanes[0] < loadedBounds[0] ? loadedBounds[0] : croppingRegionPlanes[0];
    croppingRegionPlanes[0] =
      croppingRegionPlanes[0] > loadedBounds[1] ? loadedBounds[1] : croppingRegionPlanes[0];
    croppingRegionPlanes[1] =
      croppingRegionPlanes[1] < loadedBounds[0] ? loadedBounds[0] : croppingRegionPlanes[1];
    croppingRegionPlanes[1] =
      croppingRegionPlanes[1] > loadedBounds[1] ? loadedBounds[1] : croppingRegionPlanes[1];

    croppingRegionPlanes[2] =
      croppingRegionPlanes[2] < loadedBounds[2] ? loadedBounds[2] : croppingRegionPlanes[2];
    croppingRegionPlanes[2] =
      croppingRegionPlanes[2] > loadedBounds[3] ? loadedBounds[3] : croppingRegionPlanes[2];
    croppingRegionPlanes[3] =
      croppingRegionPlanes[3] < loadedBounds[2] ? loadedBounds[2] : croppingRegionPlanes[3];
    croppingRegionPlanes[3] =
      croppingRegionPlanes[3] > loadedBounds[3] ? loadedBounds[3] : croppingRegionPlanes[3];

    croppingRegionPlanes[4] =
      croppingRegionPlanes[4] < loadedBounds[4] ? loadedBounds[4] : croppingRegionPlanes[4];
    croppingRegionPlanes[4] =
      croppingRegionPlanes[4] > loadedBounds[5] ? loadedBounds[5] : croppingRegionPlanes[4];
    croppingRegionPlanes[5] =
      croppingRegionPlanes[5] < loadedBounds[4] ? loadedBounds[4] : croppingRegionPlanes[5];
    croppingRegionPlanes[5] =
      croppingRegionPlanes[5] > loadedBounds[5] ? loadedBounds[5] : croppingRegionPlanes[5];

    float cropPlanes[6] = { static_cast<float>(croppingRegionPlanes[0]),
      static_cast<float>(croppingRegionPlanes[1]), static_cast<float>(croppingRegionPlanes[2]),
      static_cast<float>(croppingRegionPlanes[3]), static_cast<float>(croppingRegionPlanes[4]),
      static_cast<float>(croppingRegionPlanes[5]) };

    prog->SetUniform1fv("in_croppingPlanes", 6, cropPlanes);
    constexpr int numberOfRegions = 32;
    int cropFlagsArray[numberOfRegions];
    cropFlagsArray[0] = 0;
    int i = 1;
    while (cropFlags && i < 32)
    {
      cropFlagsArray[i] = cropFlags & 1;
      cropFlags = cropFlags >> 1;
      ++i;
    }
    for (; i < 32; ++i)
    {
      cropFlagsArray[i] = 0;
    }

    prog->SetUniform1iv("in_croppingFlags", numberOfRegions, cropFlagsArray);
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::SetClippingPlanes(
  vtkRenderer* vtkNotUsed(ren), vtkShaderProgram* prog, vtkVolume* vol)
{
  if (this->Parent->GetClippingPlanes())
  {
    std::vector<float> clippingPlanes;
    // Currently we don't have any clipping plane
    clippingPlanes.push_back(0);

    this->Parent->ClippingPlanes->InitTraversal();
    vtkPlane* plane;
    while ((plane = this->Parent->ClippingPlanes->GetNextItem()))
    {
      // Planes are in world coordinates
      double planeOrigin[3], planeNormal[3];
      plane->GetOrigin(planeOrigin);
      plane->GetNormal(planeNormal);

      clippingPlanes.push_back(planeOrigin[0]);
      clippingPlanes.push_back(planeOrigin[1]);
      clippingPlanes.push_back(planeOrigin[2]);
      clippingPlanes.push_back(planeNormal[0]);
      clippingPlanes.push_back(planeNormal[1]);
      clippingPlanes.push_back(planeNormal[2]);
    }

    clippingPlanes[0] = clippingPlanes.size() > 1 ? static_cast<int>(clippingPlanes.size() - 1) : 0;

    prog->SetUniform1fv(
      "in_clippingPlanes", static_cast<int>(clippingPlanes.size()), clippingPlanes.data());
    float clippedVoxelIntensity =
      static_cast<float>(vol->GetProperty()->GetClippedVoxelIntensity());
    prog->SetUniformf("in_clippedVoxelIntensity", clippedVoxelIntensity);
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::CheckPropertyKeys(vtkVolume* vol)
{
  // Check the property keys to see if we should modify the blend/etc state:
  // Otherwise this breaks volume/translucent geo depth peeling.
  vtkInformation* volumeKeys = vol->GetPropertyKeys();
  this->PreserveGLState = false;
  this->DepthMaskOverride = false;
  if (volumeKeys && volumeKeys->Has(vtkOpenGLActor::GLDepthMaskOverride()))
  {
    // Give a chance to volumes to write to the depth buffer.
    // A value of 0 keeps the depth mask disabled as set by vtkVolumeStateRAII.
    // A value of 1 enables the depth mask and allows writing to the depth buffer.
    // Any other value will prevent vtkVolumeStateRAII from changing the state.
    int override = volumeKeys->Get(vtkOpenGLActor::GLDepthMaskOverride());
    switch (override)
    {
      case 0: // glDepthMask(GL_TRUE)
        this->DepthMaskOverride = false;
        break;
      case 1: // glDepthMask(GL_FALSE)
        this->DepthMaskOverride = true;
        break;
      default: // no-op
        this->PreserveGLState = true;
        break;
    }
  }

  // Some render passes (e.g. DualDepthPeeling) adjust the viewport for
  // intermediate passes so it is necessary to preserve it. This is a
  // temporary fix for vtkDualDepthPeelingPass to work when various viewports
  // are defined.  The correct way of fixing this would be to avoid setting the
  // viewport within the mapper.  It is enough for now to check for the
  // RenderPasses() vtkInfo given that vtkDualDepthPeelingPass is the only pass
  // currently supported by this mapper, the viewport will have to be adjusted
  // externally before adding support for other passes.
  vtkInformation* info = vol->GetPropertyKeys();
  this->PreserveViewport = info && info->Has(vtkOpenGLRenderPass::RenderPasses());
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::CheckPickingState(vtkRenderer* ren)
{
  vtkHardwareSelector* selector = ren->GetSelector();
  bool selectorPicking = selector != nullptr;
  if (selector)
  {
    // this mapper currently only supports cell picking
    selectorPicking &= selector->GetFieldAssociation() == vtkDataObject::FIELD_ASSOCIATION_CELLS;
  }

  this->IsPicking = selectorPicking;
  if (this->IsPicking)
  {
    // rebuild the shader on every pass
    this->SelectionStateTime.Modified();
    this->CurrentSelectionPass =
      selector ? selector->GetCurrentPass() : vtkHardwareSelector::ACTOR_PASS;
  }
  else if (this->CurrentSelectionPass != vtkHardwareSelector::MIN_KNOWN_PASS - 1)
  {
    // return to the regular rendering state
    this->SelectionStateTime.Modified();
    this->CurrentSelectionPass = vtkHardwareSelector::MIN_KNOWN_PASS - 1;
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::BeginPicking(vtkRenderer* ren)
{
  vtkHardwareSelector* selector = ren->GetSelector();
  if (selector && this->IsPicking)
  {
    selector->BeginRenderProp();
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::SetPickingId(vtkRenderer* ren)
{
  float propIdColor[3] = { 0.0, 0.0, 0.0 };
  vtkHardwareSelector* selector = ren->GetSelector();

  if (selector && this->IsPicking)
  {
    // query the selector for the appropriate id
    selector->GetPropColorValue(propIdColor);
  }

  this->ShaderProgram->SetUniform3f("in_propId", propIdColor);
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::EndPicking(vtkRenderer* ren)
{
  vtkHardwareSelector* selector = ren->GetSelector();
  if (selector && this->IsPicking)
  {
    if (this->CurrentSelectionPass >= vtkHardwareSelector::POINT_ID_LOW24)
    {
      // Only supported on single-input
      int extents[6];
      auto dataSet = this->Parent->GetTransformedInput(0);
      if (auto imData = vtkImageData::SafeDownCast(dataSet))
      {
        imData->GetExtent(extents);
      }
      else if (auto rectGrid = vtkRectilinearGrid::SafeDownCast(dataSet))
      {
        rectGrid->GetExtent(extents);
      }

      // Tell the selector the maximum number of cells that the mapper could render
      unsigned int const numVoxels = (extents[1] - extents[0] + 1) * (extents[3] - extents[2] + 1) *
        (extents[5] - extents[4] + 1);
      selector->UpdateMaximumPointId(numVoxels);
      selector->UpdateMaximumCellId(numVoxels);
    }
    selector->EndRenderProp();
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::UpdateSamplingDistance(
  vtkRenderer* vtkNotUsed(ren))
{
  auto input = this->Parent->GetTransformedInput(0);
  auto imData = vtkImageData::SafeDownCast(input);
  auto rectGrid = vtkRectilinearGrid::SafeDownCast(input);
  auto vol = this->Parent->AssembledInputs[0].Volume;
  double cellSpacing[3];
  if (imData)
  {
    imData->GetSpacing(cellSpacing);
  }
  else if (rectGrid)
  {
    double bounds[6];
    rectGrid->GetBounds(bounds);
    int dims[3];
    rectGrid->GetDimensions(dims);
    for (int cc = 0; cc < 3; ++cc)
    {
      cellSpacing[cc] = (bounds[2 * cc + 1] - bounds[2 * cc]) / dims[cc];
    }
  }

  std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_SAMPLING autoAdjust="
            << this->Parent->AutoAdjustSampleDistances
            << " lock=" << this->Parent->LockSampleDistanceToInputSpacing
            << " sampleDistance=" << this->Parent->SampleDistance << std::endl;
  if (!this->Parent->AutoAdjustSampleDistances)
  {
    if (this->Parent->LockSampleDistanceToInputSpacing)
    {
      int extents[6];
      if (imData)
      {
        imData->GetExtent(extents);
      }
      else if (rectGrid)
      {
        rectGrid->GetExtent(extents);
      }

      float const d =
        static_cast<float>(this->Parent->SpacingAdjustedSampleDistance(cellSpacing, extents));
      float const sample = this->Parent->SampleDistance;

      // ActualSampleDistance will grow proportionally to numVoxels^(1/3) (see
      // vtkVolumeMapper.cxx). Until it reaches 1/2 average voxel size when number of
      // voxels is 1E6.
      this->ActualSampleDistance =
        (sample / d < 0.999f || sample / d > 1.001f) ? d : this->Parent->SampleDistance;

      return;
    }

    this->ActualSampleDistance = this->Parent->SampleDistance;
    std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_SAMPLING_RESULT !autoAdjust actual="
              << this->ActualSampleDistance << std::endl;
  }
  else
  {
    vol->GetModelToWorldMatrix(this->TempMatrix4x4);
    vtkMatrix4x4* worldToDataset = this->TempMatrix4x4;
    double minWorldSpacing = VTK_DOUBLE_MAX;
    int i = 0;
    while (i < 3)
    {
      double tmp = worldToDataset->GetElement(0, i);
      double tmp2 = tmp * tmp;
      tmp = worldToDataset->GetElement(1, i);
      tmp2 += tmp * tmp;
      tmp = worldToDataset->GetElement(2, i);
      tmp2 += tmp * tmp;

      // We use fabs() in case the spacing is negative.
      double worldSpacing = fabs(cellSpacing[i] * sqrt(tmp2));
      minWorldSpacing = std::min(worldSpacing, minWorldSpacing);
      ++i;
    }

    // minWorldSpacing is the optimal sample distance in world space.
    // To go faster (reduceFactor<1.0), we multiply this distance
    // by 1/reduceFactor.
    this->ActualSampleDistance = static_cast<float>(minWorldSpacing);
    std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_SAMPLING_RESULT autoAdjust actual="
              << this->ActualSampleDistance
              << " minWorldSpacing=" << minWorldSpacing
              << " reduction=" << this->Parent->ReductionFactor << std::endl;

    if (this->Parent->ReductionFactor < 1.0 && this->Parent->ReductionFactor != 0.0)
    {
      this->ActualSampleDistance /= static_cast<GLfloat>(this->Parent->ReductionFactor);
    }
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::LoadRequireDepthTextureExtensions(
  vtkRenderWindow* vtkNotUsed(renWin))
{
  // Reset the message stream for extensions
  this->LoadDepthTextureExtensionsSucceeded = true;
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::CreateBufferObjects()
{
  glGenVertexArrays(1, &this->CubeVAOId);
  glGenBuffers(1, &this->CubeVBOId);
  glGenBuffers(1, &this->CubeIndicesId);
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::DeleteBufferObjects()
{
  if (this->CubeVBOId)
  {
    glBindBuffer(GL_ARRAY_BUFFER, this->CubeVBOId);
    glDeleteBuffers(1, &this->CubeVBOId);
    this->CubeVBOId = 0;
  }

  if (this->CubeIndicesId)
  {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->CubeIndicesId);
    glDeleteBuffers(1, &this->CubeIndicesId);
    this->CubeIndicesId = 0;
  }

  if (this->CubeVAOId)
  {
    glDeleteVertexArrays(1, &this->CubeVAOId);
    this->CubeVAOId = 0;
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::ConvertTextureToImageData(
  vtkTextureObject* texture, vtkImageData* output)
{
  if (!texture)
  {
    return;
  }
  unsigned int tw = texture->GetWidth();
  unsigned int th = texture->GetHeight();
  unsigned int tnc = texture->GetComponents();
  int tt = texture->GetVTKDataType();

  vtkPixelExtent texExt(0U, tw - 1U, 0U, th - 1U);

  int dataExt[6] = { 0, 0, 0, 0, 0, 0 };
  texExt.GetData(dataExt);

  double dataOrigin[6] = { 0, 0, 0, 0, 0, 0 };

  vtkImageData* id = vtkImageData::New();
  id->SetOrigin(dataOrigin);
  id->SetDimensions(tw, th, 1);
  id->SetExtent(dataExt);
  id->AllocateScalars(tt, tnc);

  vtkPixelBufferObject* pbo = texture->Download();

  vtkPixelTransfer::Blit(texExt, texExt, texExt, texExt, tnc, tt, pbo->MapPackedBuffer(), tnc, tt,
    id->GetScalarPointer(0, 0, 0));

  pbo->UnmapPackedBuffer();
  pbo->Delete();

  if (!output)
  {
    output = vtkImageData::New();
  }
  output->DeepCopy(id);
  id->Delete();
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::BeginImageSample(vtkRenderer* ren)
{
  auto vol = this->GetActiveVolume();
  const auto numBuffers = this->GetNumImageSampleDrawBuffers(vol);
  if (numBuffers != this->NumImageSampleDrawBuffers)
  {
    if (numBuffers > this->NumImageSampleDrawBuffers)
    {
      this->ReleaseImageSampleGraphicsResources(ren->GetRenderWindow());
    }

    this->NumImageSampleDrawBuffers = numBuffers;
    this->RebuildImageSampleProg = true;
  }

  float const xySampleDist = this->Parent->ImageSampleDistance;
  if (xySampleDist != 1.f && this->InitializeImageSampleFBO(ren))
  {
    this->ImageSampleFBO->GetContext()->GetState()->PushDrawFramebufferBinding();
    this->ImageSampleFBO->Bind(GL_DRAW_FRAMEBUFFER);
    this->ImageSampleFBO->ActivateDrawBuffers(
      static_cast<unsigned int>(this->NumImageSampleDrawBuffers));

    this->ImageSampleFBO->GetContext()->GetState()->vtkglClearColor(0.0, 0.0, 0.0, 0.0);
    this->ImageSampleFBO->GetContext()->GetState()->vtkglClear(GL_COLOR_BUFFER_BIT);
  }
}

//------------------------------------------------------------------------------
bool vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::InitializeImageSampleFBO(vtkRenderer* ren)
{
  // Set the FBO viewport size. These are used in the shader to normalize the
  // fragment coordinate, the normalized coordinate is used to fetch the depth
  // buffer.
  this->WindowSize[0] /= this->Parent->ImageSampleDistance;
  this->WindowSize[1] /= this->Parent->ImageSampleDistance;
  this->WindowLowerLeft[0] = 0;
  this->WindowLowerLeft[1] = 0;

  vtkOpenGLRenderWindow* win = vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow());

  // Set FBO viewport
  win->GetState()->vtkglViewport(
    this->WindowLowerLeft[0], this->WindowLowerLeft[1], this->WindowSize[0], this->WindowSize[1]);

  if (!this->ImageSampleFBO)
  {
    this->ImageSampleTexture.reserve(this->NumImageSampleDrawBuffers);
    this->ImageSampleTexNames.reserve(this->NumImageSampleDrawBuffers);
    for (size_t i = 0; i < this->NumImageSampleDrawBuffers; i++)
    {
      auto tex = vtkSmartPointer<vtkTextureObject>::New();
      tex->SetContext(win);
      tex->Create2D(this->WindowSize[0], this->WindowSize[1], 4, VTK_UNSIGNED_CHAR, false);
      tex->Activate();
      tex->SetMinificationFilter(vtkTextureObject::Linear);
      tex->SetMagnificationFilter(vtkTextureObject::Linear);
      tex->SetWrapS(vtkTextureObject::ClampToEdge);
      tex->SetWrapT(vtkTextureObject::ClampToEdge);
      this->ImageSampleTexture.push_back(tex);

      std::stringstream ss;
      ss << i;
      const std::string name = "renderedTex_" + ss.str();
      this->ImageSampleTexNames.push_back(name);
    }

    this->ImageSampleFBO = vtkOpenGLFramebufferObject::New();
    this->ImageSampleFBO->SetContext(win);
    win->GetState()->PushFramebufferBindings();
    this->ImageSampleFBO->Bind();
    this->ImageSampleFBO->InitializeViewport(this->WindowSize[0], this->WindowSize[1]);

    auto num = static_cast<unsigned int>(this->NumImageSampleDrawBuffers);
    for (unsigned int i = 0; i < num; i++)
    {
      this->ImageSampleFBO->AddColorAttachment(i, this->ImageSampleTexture[i]);
    }

    // Verify completeness
    const int complete = this->ImageSampleFBO->CheckFrameBufferStatus(GL_FRAMEBUFFER);
    for (auto& tex : this->ImageSampleTexture)
    {
      tex->Deactivate();
    }
    win->GetState()->PopFramebufferBindings();

    if (!complete)
    {
      vtkGenericWarningMacro(<< "Failed to attach ImageSampleFBO!");
      this->ReleaseImageSampleGraphicsResources(win);
      return false;
    }

    this->RebuildImageSampleProg = true;
    return true;
  }

  // Resize if necessary
  int lastSize[2];
  this->ImageSampleFBO->GetLastSize(lastSize);
  if (lastSize[0] != this->WindowSize[0] || lastSize[1] != this->WindowSize[1])
  {
    this->ImageSampleFBO->Resize(this->WindowSize[0], this->WindowSize[1]);
  }

  return true;
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::EndImageSample(vtkRenderer* ren)
{
  if (this->Parent->ImageSampleDistance != 1.f)
  {
    this->ImageSampleFBO->DeactivateDrawBuffers();
    if (this->RenderPassAttached)
    {
      this->ImageSampleFBO->ActivateDrawBuffers(
        static_cast<unsigned int>(this->NumImageSampleDrawBuffers));
    }
    vtkOpenGLRenderWindow* win = static_cast<vtkOpenGLRenderWindow*>(ren->GetRenderWindow());
    win->GetState()->PopDrawFramebufferBinding();

    // Render the contents of ImageSampleFBO as a quad to intermix with the
    // rest of the scene.
    typedef vtkOpenGLRenderUtilities GLUtil;

    if (this->RebuildImageSampleProg)
    {
      std::string frag = GLUtil::GetFullScreenQuadFragmentShaderTemplate();

      vtkShaderProgram::Substitute(frag, "//VTK::FSQ::Decl",
        vtkvolume::ImageSampleDeclarationFrag(
          this->ImageSampleTexNames, this->NumImageSampleDrawBuffers));
      vtkShaderProgram::Substitute(frag, "//VTK::FSQ::Impl",
        vtkvolume::ImageSampleImplementationFrag(
          this->ImageSampleTexNames, this->NumImageSampleDrawBuffers));

      this->ImageSampleProg =
        win->GetShaderCache()->ReadyShaderProgram(GLUtil::GetFullScreenQuadVertexShader().c_str(),
          frag.c_str(), GLUtil::GetFullScreenQuadGeometryShader().c_str());
    }
    else
    {
      win->GetShaderCache()->ReadyShaderProgram(this->ImageSampleProg);
    }

    if (!this->ImageSampleProg)
    {
      vtkGenericWarningMacro(<< "Failed to initialize ImageSampleProgram!");
      return;
    }

    if (!this->ImageSampleVAO)
    {
      this->ImageSampleVAO = vtkOpenGLVertexArrayObject::New();
      GLUtil::PrepFullScreenVAO(win, this->ImageSampleVAO, this->ImageSampleProg);
    }

    vtkOpenGLState* ostate = win->GetState();

    // Adjust the GL viewport to VTK's defined viewport
    ren->GetTiledSizeAndOrigin(
      this->WindowSize, this->WindowSize + 1, this->WindowLowerLeft, this->WindowLowerLeft + 1);
    ostate->vtkglViewport(
      this->WindowLowerLeft[0], this->WindowLowerLeft[1], this->WindowSize[0], this->WindowSize[1]);

    // Bind objects and draw
    ostate->vtkglEnable(GL_BLEND);
    ostate->vtkglBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    ostate->vtkglDisable(GL_DEPTH_TEST);

    for (size_t i = 0; i < this->NumImageSampleDrawBuffers; i++)
    {
      this->ImageSampleTexture[i]->Activate();
      this->ImageSampleProg->SetUniformi(
        this->ImageSampleTexNames[i].c_str(), this->ImageSampleTexture[i]->GetTextureUnit());
    }

    this->ImageSampleVAO->Bind();
    GLUtil::DrawFullScreenQuad();
    this->ImageSampleVAO->Release();
    vtkOpenGLStaticCheckErrorMacro("Error after DrawFullScreenQuad()!");

    for (auto& tex : this->ImageSampleTexture)
    {
      tex->Deactivate();
    }
  }
}

//------------------------------------------------------------------------------
size_t vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::GetNumImageSampleDrawBuffers(vtkVolume* vol)
{
  if (this->RenderPassAttached)
  {
    vtkInformation* info = vol->GetPropertyKeys();
    const int num = info->Length(vtkOpenGLRenderPass::RenderPasses());
    vtkObjectBase* rpBase = info->Get(vtkOpenGLRenderPass::RenderPasses(), num - 1);
    vtkOpenGLRenderPass* rp = static_cast<vtkOpenGLRenderPass*>(rpBase);
    return static_cast<size_t>(rp->GetActiveDrawBuffers());
  }

  return 1;
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::SetupRenderToTexture(vtkRenderer* ren)
{
  if (this->Parent->RenderToImage && this->Parent->CurrentPass == RenderPass)
  {
    if (this->Parent->ImageSampleDistance != 1.f)
    {
      this->WindowSize[0] /= this->Parent->ImageSampleDistance;
      this->WindowSize[1] /= this->Parent->ImageSampleDistance;
    }

    if ((this->LastRenderToImageWindowSize[0] != this->WindowSize[0]) ||
      (this->LastRenderToImageWindowSize[1] != this->WindowSize[1]))
    {
      this->LastRenderToImageWindowSize[0] = this->WindowSize[0];
      this->LastRenderToImageWindowSize[1] = this->WindowSize[1];
      this->ReleaseRenderToTextureGraphicsResources(ren->GetRenderWindow());
    }

    if (!this->FBO)
    {
      this->FBO = vtkOpenGLFramebufferObject::New();
    }

    vtkOpenGLRenderWindow* renWin = vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow());
    this->FBO->SetContext(renWin);

    renWin->GetState()->PushFramebufferBindings();
    this->FBO->Bind();
    this->FBO->InitializeViewport(this->WindowSize[0], this->WindowSize[1]);

    int depthImageScalarType = this->Parent->GetDepthImageScalarType();
    bool initDepthTexture = true;
    // Re-instantiate the depth texture object if the scalar type requested has
    // changed from the last frame
    if (this->RTTDepthTextureObject && this->RTTDepthTextureType == depthImageScalarType)
    {
      initDepthTexture = false;
    }

    if (initDepthTexture)
    {
      if (this->RTTDepthTextureObject)
      {
        this->RTTDepthTextureObject->Delete();
        this->RTTDepthTextureObject = nullptr;
      }
      this->RTTDepthTextureObject = vtkTextureObject::New();
      this->RTTDepthTextureObject->SetContext(renWin);
      this->RTTDepthTextureObject->Create2D(
        this->WindowSize[0], this->WindowSize[1], 1, depthImageScalarType, false);
      this->RTTDepthTextureObject->Activate();
      this->RTTDepthTextureObject->SetMinificationFilter(vtkTextureObject::Nearest);
      this->RTTDepthTextureObject->SetMagnificationFilter(vtkTextureObject::Nearest);
      this->RTTDepthTextureObject->SetAutoParameters(0);

      // Cache the value of the scalar type
      this->RTTDepthTextureType = depthImageScalarType;
    }

    if (!this->RTTColorTextureObject)
    {
      this->RTTColorTextureObject = vtkTextureObject::New();

      this->RTTColorTextureObject->SetContext(
        vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow()));
      this->RTTColorTextureObject->Create2D(
        this->WindowSize[0], this->WindowSize[1], 4, VTK_UNSIGNED_CHAR, false);
      this->RTTColorTextureObject->Activate();
      this->RTTColorTextureObject->SetMinificationFilter(vtkTextureObject::Nearest);
      this->RTTColorTextureObject->SetMagnificationFilter(vtkTextureObject::Nearest);
      this->RTTColorTextureObject->SetAutoParameters(0);
    }

    if (!this->RTTDepthBufferTextureObject)
    {
      this->RTTDepthBufferTextureObject = vtkTextureObject::New();
      this->RTTDepthBufferTextureObject->SetContext(renWin);
      this->RTTDepthBufferTextureObject->AllocateDepth(
        this->WindowSize[0], this->WindowSize[1], vtkTextureObject::Float32);
      this->RTTDepthBufferTextureObject->Activate();
      this->RTTDepthBufferTextureObject->SetMinificationFilter(vtkTextureObject::Nearest);
      this->RTTDepthBufferTextureObject->SetMagnificationFilter(vtkTextureObject::Nearest);
      this->RTTDepthBufferTextureObject->SetAutoParameters(0);
    }

    this->FBO->Bind(GL_FRAMEBUFFER);
    this->FBO->AddDepthAttachment(this->RTTDepthBufferTextureObject);
    this->FBO->AddColorAttachment(0U, this->RTTColorTextureObject);
    this->FBO->AddColorAttachment(1U, this->RTTDepthTextureObject);
    this->FBO->ActivateDrawBuffers(2);

    this->FBO->CheckFrameBufferStatus(GL_FRAMEBUFFER);

    this->FBO->GetContext()->GetState()->vtkglClearColor(1.0, 1.0, 1.0, 0.0);
    this->FBO->GetContext()->GetState()->vtkglClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::ExitRenderToTexture(vtkRenderer* vtkNotUsed(ren))
{
  if (this->Parent->RenderToImage && this->Parent->CurrentPass == RenderPass)
  {
    this->FBO->RemoveDepthAttachment();
    this->FBO->RemoveColorAttachment(0U);
    this->FBO->RemoveColorAttachment(1U);
    this->FBO->DeactivateDrawBuffers();
    this->FBO->GetContext()->GetState()->PopFramebufferBindings();

    this->RTTDepthBufferTextureObject->Deactivate();
    this->RTTColorTextureObject->Deactivate();
    this->RTTDepthTextureObject->Deactivate();
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::SetupDepthPass(vtkRenderer* ren)
{
  if (this->Parent->ImageSampleDistance != 1.f)
  {
    this->WindowSize[0] /= this->Parent->ImageSampleDistance;
    this->WindowSize[1] /= this->Parent->ImageSampleDistance;
  }

  if ((this->LastDepthPassWindowSize[0] != this->WindowSize[0]) ||
    (this->LastDepthPassWindowSize[1] != this->WindowSize[1]))
  {
    this->LastDepthPassWindowSize[0] = this->WindowSize[0];
    this->LastDepthPassWindowSize[1] = this->WindowSize[1];
    this->ReleaseDepthPassGraphicsResources(ren->GetRenderWindow());
  }

  if (!this->DPFBO)
  {
    this->DPFBO = vtkOpenGLFramebufferObject::New();
  }

  vtkOpenGLRenderWindow* renWin = vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow());
  this->DPFBO->SetContext(renWin);

  renWin->GetState()->PushFramebufferBindings();
  this->DPFBO->Bind();
  this->DPFBO->InitializeViewport(this->WindowSize[0], this->WindowSize[1]);

  if (!this->DPDepthBufferTextureObject || !this->DPColorTextureObject)
  {
    this->DPDepthBufferTextureObject = vtkTextureObject::New();
    this->DPDepthBufferTextureObject->SetContext(renWin);
    this->DPDepthBufferTextureObject->AllocateDepth(
      this->WindowSize[0], this->WindowSize[1], vtkTextureObject::Native);
    this->DPDepthBufferTextureObject->Activate();
    this->DPDepthBufferTextureObject->SetMinificationFilter(vtkTextureObject::Nearest);
    this->DPDepthBufferTextureObject->SetMagnificationFilter(vtkTextureObject::Nearest);
    this->DPDepthBufferTextureObject->SetAutoParameters(0);
    this->DPDepthBufferTextureObject->Bind();

    this->DPColorTextureObject = vtkTextureObject::New();

    this->DPColorTextureObject->SetContext(renWin);
    this->DPColorTextureObject->Create2D(
      this->WindowSize[0], this->WindowSize[1], 4, VTK_UNSIGNED_CHAR, false);
    this->DPColorTextureObject->Activate();
    this->DPColorTextureObject->SetMinificationFilter(vtkTextureObject::Nearest);
    this->DPColorTextureObject->SetMagnificationFilter(vtkTextureObject::Nearest);
    this->DPColorTextureObject->SetAutoParameters(0);

    this->DPFBO->AddDepthAttachment(this->DPDepthBufferTextureObject);

    this->DPFBO->AddColorAttachment(0U, this->DPColorTextureObject);
  }

  this->DPFBO->ActivateDrawBuffers(1);
  this->DPFBO->CheckFrameBufferStatus(GL_FRAMEBUFFER);

  // Setup the contour polydata mapper to render to DPFBO
  this->ContourMapper->SetInputConnection(this->ContourFilter->GetOutputPort());

  vtkOpenGLState* ostate = this->DPFBO->GetContext()->GetState();
  ostate->vtkglClearColor(0.0, 0.0, 0.0, 0.0);
  ostate->vtkglClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  ostate->vtkglEnable(GL_DEPTH_TEST);
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::RenderContourPass(vtkRenderer* ren)
{
  this->SetupDepthPass(ren);
  this->ContourActor->Render(ren, this->ContourMapper.GetPointer());
  this->ExitDepthPass(ren);
  this->DepthPassTime.Modified();
  this->Parent->CurrentPass = this->Parent->RenderPass;
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::ExitDepthPass(vtkRenderer* vtkNotUsed(ren))
{
  this->DPFBO->DeactivateDrawBuffers();
  vtkOpenGLState* ostate = this->DPFBO->GetContext()->GetState();
  ostate->PopFramebufferBindings();

  this->DPDepthBufferTextureObject->Deactivate();
  this->DPColorTextureObject->Deactivate();
  ostate->vtkglDisable(GL_DEPTH_TEST);
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal ::ReleaseRenderToTextureGraphicsResources(
  vtkWindow* win)
{
  vtkOpenGLRenderWindow* rwin = vtkOpenGLRenderWindow::SafeDownCast(win);

  if (rwin)
  {
    if (this->FBO)
    {
      this->FBO->Delete();
      this->FBO = nullptr;
    }

    if (this->RTTDepthBufferTextureObject)
    {
      this->RTTDepthBufferTextureObject->ReleaseGraphicsResources(win);
      this->RTTDepthBufferTextureObject->Delete();
      this->RTTDepthBufferTextureObject = nullptr;
    }

    if (this->RTTDepthTextureObject)
    {
      this->RTTDepthTextureObject->ReleaseGraphicsResources(win);
      this->RTTDepthTextureObject->Delete();
      this->RTTDepthTextureObject = nullptr;
    }

    if (this->RTTColorTextureObject)
    {
      this->RTTColorTextureObject->ReleaseGraphicsResources(win);
      this->RTTColorTextureObject->Delete();
      this->RTTColorTextureObject = nullptr;
    }
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal ::ReleaseDepthPassGraphicsResources(
  vtkWindow* win)
{
  vtkOpenGLRenderWindow* rwin = vtkOpenGLRenderWindow::SafeDownCast(win);

  if (rwin)
  {
    if (this->DPFBO)
    {
      this->DPFBO->Delete();
      this->DPFBO = nullptr;
    }

    if (this->DPDepthBufferTextureObject)
    {
      this->DPDepthBufferTextureObject->ReleaseGraphicsResources(win);
      this->DPDepthBufferTextureObject->Delete();
      this->DPDepthBufferTextureObject = nullptr;
    }

    if (this->DPColorTextureObject)
    {
      this->DPColorTextureObject->ReleaseGraphicsResources(win);
      this->DPColorTextureObject->Delete();
      this->DPColorTextureObject = nullptr;
    }

    this->ContourMapper->ReleaseGraphicsResources(win);
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal ::ReleaseImageSampleGraphicsResources(
  vtkWindow* win)
{
  vtkOpenGLRenderWindow* rwin = vtkOpenGLRenderWindow::SafeDownCast(win);

  if (rwin)
  {
    if (this->ImageSampleFBO)
    {
      this->ImageSampleFBO->Delete();
      this->ImageSampleFBO = nullptr;
    }

    for (auto& tex : this->ImageSampleTexture)
    {
      tex->ReleaseGraphicsResources(win);
      tex = nullptr;
    }
    this->ImageSampleTexture.clear();
    this->ImageSampleTexNames.clear();

    if (this->ImageSampleVAO)
    {
      this->ImageSampleVAO->Delete();
      this->ImageSampleVAO = nullptr;
    }

    // Do not delete the shader program - Let the cache clean it up.
    this->ImageSampleProg = nullptr;
  }
}

//------------------------------------------------------------------------------
vtkOpenGLGPUVolumeRayCastMapper::vtkOpenGLGPUVolumeRayCastMapper()
{
  this->Impl = new vtkInternal(this);
  this->ReductionFactor = 1.0;
  this->CurrentPass = RenderPass;

  this->ResourceCallback = new vtkOpenGLResourceFreeCallback<vtkOpenGLGPUVolumeRayCastMapper>(
    this, &vtkOpenGLGPUVolumeRayCastMapper::ReleaseGraphicsResources);
}

//------------------------------------------------------------------------------
vtkOpenGLGPUVolumeRayCastMapper::~vtkOpenGLGPUVolumeRayCastMapper()
{
  if (this->ResourceCallback)
  {
    this->ResourceCallback->Release();
    delete this->ResourceCallback;
    this->ResourceCallback = nullptr;
  }

  delete this->Impl;
  this->Impl = nullptr;

  this->AssembledInputs.clear();
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "ReductionFactor: " << this->ReductionFactor << "\n";
  os << indent << "CurrentPass: " << this->CurrentPass << "\n";
}

void vtkOpenGLGPUVolumeRayCastMapper::SetSharedDepthTexture(vtkTextureObject* nt)
{
  if (this->Impl->DepthTextureObject == nt)
  {
    return;
  }
  if (this->Impl->DepthTextureObject)
  {
    this->Impl->DepthTextureObject->Delete();
  }
  this->Impl->DepthTextureObject = nt;

  if (nt)
  {
    nt->Register(this); // as it will get deleted later on
    this->Impl->SharedDepthTextureObject = true;
  }
  else
  {
    this->Impl->SharedDepthTextureObject = false;
  }
}

//------------------------------------------------------------------------------
vtkTextureObject* vtkOpenGLGPUVolumeRayCastMapper::GetDepthTexture()
{
  return this->Impl->RTTDepthTextureObject;
}

//------------------------------------------------------------------------------
vtkTextureObject* vtkOpenGLGPUVolumeRayCastMapper::GetColorTexture()
{
  return this->Impl->RTTColorTextureObject;
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::GetDepthImage(vtkImageData* output)
{
  this->Impl->ConvertTextureToImageData(this->Impl->RTTDepthTextureObject, output);
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::GetColorImage(vtkImageData* output)
{
  this->Impl->ConvertTextureToImageData(this->Impl->RTTColorTextureObject, output);
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::ReleaseGraphicsResources(vtkWindow* window)
{
  if (!this->ResourceCallback->IsReleasing())
  {
    this->ResourceCallback->Release();
    return;
  }

  this->Impl->DeleteBufferObjects();

  for (auto& input : this->AssembledInputs)
  {
    input.second.ReleaseGraphicsResources(window);
  }

  if (this->Impl->DepthTextureObject && !this->Impl->SharedDepthTextureObject)
  {
    this->Impl->DepthTextureObject->ReleaseGraphicsResources(window);
    this->Impl->DepthTextureObject->Delete();
    this->Impl->DepthTextureObject = nullptr;
    this->Impl->DepthCopyColorTextureObject->ReleaseGraphicsResources(window);
    this->Impl->DepthCopyColorTextureObject->Delete();
    this->Impl->DepthCopyColorTextureObject = nullptr;
    this->Impl->DepthCopyFBO->ReleaseGraphicsResources(window);
    this->Impl->DepthCopyFBO->Delete();
    this->Impl->DepthCopyFBO = nullptr;
  }

  this->Impl->ReleaseRenderToTextureGraphicsResources(window);
  this->Impl->ReleaseDepthPassGraphicsResources(window);
  this->Impl->ReleaseImageSampleGraphicsResources(window);

  if (this->Impl->CurrentMask)
  {
    this->Impl->CurrentMask->ReleaseGraphicsResources(window);
    this->Impl->CurrentMask = nullptr;
  }

  this->Impl->ReleaseGraphicsMaskTransfer(window);
  this->Impl->DeleteMaskTransfer();

  this->Impl->ReleaseResourcesTime.Modified();
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::GetShaderTemplate(
  std::map<vtkShader::Type, vtkShader*>& shaders, vtkOpenGLShaderProperty* p)
{
  if (shaders[vtkShader::Vertex])
  {
    if (p->HasVertexShaderCode())
    {
      shaders[vtkShader::Vertex]->SetSource(p->GetVertexShaderCode());
    }
    else
    {
      shaders[vtkShader::Vertex]->SetSource(raycastervs);
    }
  }

  if (shaders[vtkShader::Fragment])
  {
    if (p->HasFragmentShaderCode())
    {
      shaders[vtkShader::Fragment]->SetSource(p->GetFragmentShaderCode());
    }
    else
    {
      shaders[vtkShader::Fragment]->SetSource(raycasterfs);
    }
  }

  if (shaders[vtkShader::Geometry])
  {
    shaders[vtkShader::Geometry]->SetSource("");
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::ReplaceShaderCustomUniforms(
  std::map<vtkShader::Type, vtkShader*>& shaders, vtkOpenGLShaderProperty* p)
{
  vtkShader* vertexShader = shaders[vtkShader::Vertex];
  vtkOpenGLUniforms* vu = static_cast<vtkOpenGLUniforms*>(p->GetVertexCustomUniforms());
  vtkShaderProgram::Substitute(vertexShader, "//VTK::CustomUniforms::Dec", vu->GetDeclarations());

  vtkShader* fragmentShader = shaders[vtkShader::Fragment];
  vtkOpenGLUniforms* fu = static_cast<vtkOpenGLUniforms*>(p->GetFragmentCustomUniforms());
  vtkShaderProgram::Substitute(fragmentShader, "//VTK::CustomUniforms::Dec", fu->GetDeclarations());

  vtkShader* geometryShader = shaders[vtkShader::Geometry];
  vtkOpenGLUniforms* gu = static_cast<vtkOpenGLUniforms*>(p->GetGeometryCustomUniforms());
  vtkShaderProgram::Substitute(geometryShader, "//VTK::CustomUniforms::Dec", gu->GetDeclarations());
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::ReplaceShaderBase(
  std::map<vtkShader::Type, vtkShader*>& shaders, vtkRenderer* ren, vtkVolume* vol, int numComps)
{
  vtkShader* vertexShader = shaders[vtkShader::Vertex];
  vtkShader* fragmentShader = shaders[vtkShader::Fragment];

  // Every volume should have a property (cannot be nullptr);
  vtkVolumeProperty* volumeProperty = vol->GetProperty();
  int independentComponents = volumeProperty->GetIndependentComponents();

  vtkShaderProgram::Substitute(vertexShader, "//VTK::ComputeClipPos::Impl",
    vtkvolume::ComputeClipPositionImplementation(ren, this, vol));

  vtkShaderProgram::Substitute(vertexShader, "//VTK::ComputeTextureCoords::Impl",
    vtkvolume::ComputeTextureCoordinates(ren, this, vol));

  vtkShaderProgram::Substitute(vertexShader, "//VTK::Base::Dec",
    vtkvolume::BaseDeclarationVertex(ren, this, vol, this->Impl->MultiVolume != nullptr));

  vtkShaderProgram::Substitute(
    fragmentShader, "//VTK::CallWorker::Impl", vtkvolume::WorkerImplementation(ren, this, vol));

  vtkShaderProgram::Substitute(fragmentShader, "//VTK::Base::Dec",
    vtkvolume::BaseDeclarationFragment(ren, this, this->AssembledInputs,
      this->Impl->TotalNumberOfLights, this->Impl->NumberPositionalLights,
      this->Impl->DefaultLighting, numComps, independentComponents));

  vtkShaderProgram::Substitute(fragmentShader, "//VTK::Base::Init",
    vtkvolume::BaseInit(ren, this, this->AssembledInputs, this->Impl->DefaultLighting));

  vtkShaderProgram::Substitute(
    fragmentShader, "//VTK::Base::Impl", vtkvolume::BaseImplementation(ren, this, vol));

  vtkShaderProgram::Substitute(
    fragmentShader, "//VTK::Base::Exit", vtkvolume::BaseExit(ren, this, vol));
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::ReplaceShaderTermination(
  std::map<vtkShader::Type, vtkShader*>& shaders, vtkRenderer* ren, vtkVolume* vol,
  int vtkNotUsed(numComps))
{
  vtkShader* vertexShader = shaders[vtkShader::Vertex];
  vtkShader* fragmentShader = shaders[vtkShader::Fragment];

  vtkShaderProgram::Substitute(vertexShader, "//VTK::Termination::Dec",
    vtkvolume::TerminationDeclarationVertex(ren, this, vol));

  vtkShaderProgram::Substitute(fragmentShader, "//VTK::Termination::Dec",
    vtkvolume::TerminationDeclarationFragment(ren, this, vol));

  vtkShaderProgram::Substitute(
    fragmentShader, "//VTK::Terminate::Init", vtkvolume::TerminationInit(ren, this, vol));

  vtkShaderProgram::Substitute(
    fragmentShader, "//VTK::Terminate::Impl", vtkvolume::TerminationImplementation(ren, this, vol));

  vtkShaderProgram::Substitute(
    fragmentShader, "//VTK::Terminate::Exit", vtkvolume::TerminationExit(ren, this, vol));
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::ReplaceShaderShading(
  std::map<vtkShader::Type, vtkShader*>& shaders, vtkRenderer* ren, vtkVolume* vol, int numComps)
{
  vtkShader* vertexShader = shaders[vtkShader::Vertex];
  vtkShader* fragmentShader = shaders[vtkShader::Fragment];

  // Every volume should have a property (cannot be nullptr);
  vtkVolumeProperty* volumeProperty = vol->GetProperty();
  int independentComponents = volumeProperty->GetIndependentComponents();

  vtkShaderProgram::Substitute(
    vertexShader, "//VTK::Shading::Dec", vtkvolume::ShadingDeclarationVertex(ren, this, vol));

  vtkShaderProgram::Substitute(
    fragmentShader, "//VTK::Shading::Dec", vtkvolume::ShadingDeclarationFragment(ren, this, vol));

  vtkShaderProgram::Substitute(
    fragmentShader, "//VTK::Shading::Init", vtkvolume::ShadingInit(ren, this, vol));

  if (this->Impl->MultiVolume)
  {
    vtkShaderProgram::Substitute(fragmentShader, "//VTK::Shading::Impl",
      vtkvolume::ShadingMultipleInputs(this, this->AssembledInputs));
  }
  else
  {
    vtkShaderProgram::Substitute(fragmentShader, "//VTK::Shading::Impl",
      vtkvolume::ShadingSingleInput(ren, this, vol, this->MaskInput, this->Impl->CurrentMask,
        this->MaskType, numComps, independentComponents));
  }

  vtkShaderProgram::Substitute(fragmentShader, "//VTK::Shading::Exit",
    vtkvolume::ShadingExit(ren, this, vol, numComps, independentComponents));
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::ReplaceShaderCompute(
  std::map<vtkShader::Type, vtkShader*>& shaders, vtkRenderer* ren, vtkVolume* vol, int numComps)
{
  vtkShader* fragmentShader = shaders[vtkShader::Fragment];

  // Every volume should have a property (cannot be nullptr);
  vtkVolumeProperty* volumeProperty = vol->GetProperty();
  int independentComponents = volumeProperty->GetIndependentComponents();

  vtkShaderProgram::Substitute(fragmentShader, "//VTK::ComputeGradient::Dec",
    vtkvolume::ComputeGradientDeclaration(this, this->AssembledInputs));

  if (this->ComputeNormalFromOpacity)
  {
    vtkShaderProgram::Substitute(fragmentShader, "//VTK::ComputeDensityGradient::Dec",
      vtkvolume::ComputeDensityGradientDeclaration(this, this->AssembledInputs, numComps,
        independentComponents, this->Impl->Transfer2DUseGradient));
  }

  if (this->GetVolumetricScatteringBlending() > 0.0 && this->Impl->TotalNumberOfLights > 0.0)
  {
    vtkShaderProgram::Substitute(fragmentShader, "//VTK::PhaseFunction::Dec",
      vtkvolume::PhaseFunctionDeclaration(ren, this, vol));

    vtkShaderProgram::Substitute(fragmentShader, "//VTK::ComputeVolumetricShadow::Dec",
      vtkvolume::ComputeVolumetricShadowDec(this, vol, numComps, independentComponents,
        this->AssembledInputs, this->Impl->Transfer2DUseGradient));

    if (!this->Impl->DefaultLighting)
    {
      vtkShaderProgram::Substitute(fragmentShader, "//VTK::Matrices::Init",
        vtkvolume::ComputeMatricesInit(this, this->Impl->NumberPositionalLights));
    }
  }

  vtkShaderProgram::Substitute(fragmentShader, "//VTK::ComputeColor::Unif",
    vtkvolume::ComputeColorUniforms(this->AssembledInputs, numComps, volumeProperty));

  if (this->Impl->MultiVolume)
  {
    vtkShaderProgram::Substitute(fragmentShader, "//VTK::GradientCache::Dec",
      vtkvolume::GradientCacheDec(ren, vol, this->AssembledInputs, independentComponents));

    vtkShaderProgram::Substitute(fragmentShader, "//VTK::Transfer2D::Dec",
      vtkvolume::Transfer2DDeclaration(this->AssembledInputs));

    vtkShaderProgram::Substitute(fragmentShader, "//VTK::ComputeOpacity::Dec",
      vtkvolume::ComputeOpacityMultiDeclaration(this->AssembledInputs));

    vtkShaderProgram::Substitute(fragmentShader, "//VTK::ComputeGradientOpacity1D::Dec",
      vtkvolume::ComputeGradientOpacityMulti1DDecl(this->AssembledInputs));

    vtkShaderProgram::Substitute(fragmentShader, "//VTK::ComputeColor::Dec",
      vtkvolume::ComputeColorMultiDeclaration(
        this->AssembledInputs, vol->GetProperty()->HasGradientOpacity()));

    vtkShaderProgram::Substitute(fragmentShader, "//VTK::ComputeLighting::Dec",
      vtkvolume::ComputeLightingMultiDeclaration(ren, this, vol, numComps, independentComponents,
        this->Impl->TotalNumberOfLights, this->Impl->DefaultLighting));
  }
  else
  {
    // Single input
    switch (volumeProperty->GetTransferFunctionMode())
    {
      case vtkVolumeProperty::TF_1D:
      {
        auto& input = this->AssembledInputs[0];

        vtkShaderProgram::Substitute(fragmentShader, "//VTK::ComputeOpacity::Dec",
          vtkvolume::ComputeOpacityDeclaration(
            ren, this, vol, numComps, independentComponents, input.OpacityTablesMap));

        vtkShaderProgram::Substitute(fragmentShader, "//VTK::ComputeGradientOpacity1D::Dec",
          vtkvolume::ComputeGradientOpacity1DDecl(
            vol, numComps, independentComponents, input.GradientOpacityTablesMap));

        vtkShaderProgram::Substitute(fragmentShader, "//VTK::ComputeColor::Dec",
          vtkvolume::ComputeColorDeclaration(
            ren, this, vol, numComps, independentComponents, input.RGBTablesMap));
      }
      break;
      case vtkVolumeProperty::TF_2D:
        vtkShaderProgram::Substitute(fragmentShader, "//VTK::ComputeOpacity::Dec",
          vtkvolume::ComputeOpacity2DDeclaration(ren, this, vol, numComps, independentComponents,
            this->AssembledInputs[0].TransferFunctions2DMap, this->Impl->Transfer2DUseGradient));

        vtkShaderProgram::Substitute(fragmentShader, "//VTK::ComputeRGBA2DWithGradient::Dec",
          vtkvolume::ComputeRGBA2DWithGradientDeclaration(ren, this, vol, numComps,
            independentComponents, this->AssembledInputs[0].TransferFunctions2DMap,
            this->Impl->Transfer2DUseGradient));

        vtkShaderProgram::Substitute(fragmentShader, "//VTK::ComputeColor::Dec",
          vtkvolume::ComputeColor2DDeclaration(ren, this, vol, numComps, independentComponents,
            this->AssembledInputs[0].TransferFunctions2DMap, this->Impl->Transfer2DUseGradient));

        vtkShaderProgram::Substitute(fragmentShader, "//VTK::GradientCache::Dec",
          vtkvolume::GradientCacheDec(ren, vol, this->AssembledInputs, independentComponents));

        vtkShaderProgram::Substitute(fragmentShader, "//VTK::PreComputeGradients::Impl",
          vtkvolume::PreComputeGradientsImpl(ren, vol, numComps, independentComponents));

        vtkShaderProgram::Substitute(fragmentShader, "//VTK::Transfer2D::Dec",
          vtkvolume::Transfer2DDeclaration(this->AssembledInputs));
        break;
    }

    vtkShaderProgram::Substitute(fragmentShader, "//VTK::ComputeLighting::Dec",
      vtkvolume::ComputeLightingDeclaration(ren, this, vol, numComps, independentComponents,
        this->Impl->TotalNumberOfLights, this->Impl->NumberPositionalLights,
        this->Impl->DefaultLighting));
  }

  vtkShaderProgram::Substitute(fragmentShader, "//VTK::ComputeRayDirection::Dec",
    vtkvolume::ComputeRayDirectionDeclaration(ren, this, vol, numComps));
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::ReplaceShaderCropping(
  std::map<vtkShader::Type, vtkShader*>& shaders, vtkRenderer* ren, vtkVolume* vol,
  int vtkNotUsed(numComps))
{
  vtkShader* vertexShader = shaders[vtkShader::Vertex];
  vtkShader* fragmentShader = shaders[vtkShader::Fragment];

  vtkShaderProgram::Substitute(
    vertexShader, "//VTK::Cropping::Dec", vtkvolume::CroppingDeclarationVertex(ren, this, vol));

  vtkShaderProgram::Substitute(
    fragmentShader, "//VTK::Cropping::Dec", vtkvolume::CroppingDeclarationFragment(ren, this, vol));

  vtkShaderProgram::Substitute(
    fragmentShader, "//VTK::Cropping::Init", vtkvolume::CroppingInit(ren, this, vol));

  vtkShaderProgram::Substitute(
    fragmentShader, "//VTK::Cropping::Impl", vtkvolume::CroppingImplementation(ren, this, vol));
  // true);

  vtkShaderProgram::Substitute(
    fragmentShader, "//VTK::Cropping::Exit", vtkvolume::CroppingExit(ren, this, vol));
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::ReplaceShaderClipping(
  std::map<vtkShader::Type, vtkShader*>& shaders, vtkRenderer* ren, vtkVolume* vol,
  int vtkNotUsed(numComps))
{
  vtkShader* vertexShader = shaders[vtkShader::Vertex];
  vtkShader* fragmentShader = shaders[vtkShader::Fragment];

  vtkShaderProgram::Substitute(
    vertexShader, "//VTK::Clipping::Dec", vtkvolume::ClippingDeclarationVertex(ren, this, vol));

  vtkShaderProgram::Substitute(
    fragmentShader, "//VTK::Clipping::Dec", vtkvolume::ClippingDeclarationFragment(ren, this, vol));

  vtkShaderProgram::Substitute(
    fragmentShader, "//VTK::Clipping::Init", vtkvolume::ClippingInit(ren, this, vol));

  vtkShaderProgram::Substitute(
    fragmentShader, "//VTK::Clipping::Impl", vtkvolume::ClippingImplementation(ren, this, vol));

  vtkShaderProgram::Substitute(
    fragmentShader, "//VTK::Clipping::Exit", vtkvolume::ClippingExit(ren, this, vol));
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::ReplaceShaderMasking(
  std::map<vtkShader::Type, vtkShader*>& shaders, vtkRenderer* ren, vtkVolume* vol, int numComps)
{
  vtkShader* fragmentShader = shaders[vtkShader::Fragment];

  vtkShaderProgram::Substitute(fragmentShader, "//VTK::BinaryMask::Dec",
    vtkvolume::BinaryMaskDeclaration(
      ren, this, vol, this->MaskInput, this->Impl->CurrentMask, this->MaskType));

  vtkShaderProgram::Substitute(fragmentShader, "//VTK::BinaryMask::Impl",
    vtkvolume::BinaryMaskImplementation(
      ren, this, vol, this->MaskInput, this->Impl->CurrentMask, this->MaskType));

  vtkShaderProgram::Substitute(fragmentShader, "//VTK::CompositeMask::Dec",
    vtkvolume::CompositeMaskDeclarationFragment(
      ren, this, vol, this->MaskInput, this->Impl->CurrentMask, this->MaskType));

  vtkShaderProgram::Substitute(fragmentShader, "//VTK::CompositeMask::Impl",
    vtkvolume::CompositeMaskImplementation(
      ren, this, vol, this->MaskInput, this->Impl->CurrentMask, this->MaskType, numComps));
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::ReplaceShaderPicking(
  std::map<vtkShader::Type, vtkShader*>& shaders, vtkRenderer* ren, vtkVolume* vol,
  int vtkNotUsed(numComps))
{
  vtkShader* fragmentShader = shaders[vtkShader::Fragment];

  if (this->Impl->CurrentSelectionPass != (vtkHardwareSelector::MIN_KNOWN_PASS - 1))
  {
    switch (this->Impl->CurrentSelectionPass)
    {
      case vtkHardwareSelector::CELL_ID_LOW24:
        vtkShaderProgram::Substitute(fragmentShader, "//VTK::Picking::Exit",
          vtkvolume::PickingIdLow24PassExit(ren, this, vol));
        break;
      case vtkHardwareSelector::CELL_ID_HIGH24:
        vtkShaderProgram::Substitute(fragmentShader, "//VTK::Picking::Exit",
          vtkvolume::PickingIdHigh24PassExit(ren, this, vol));
        break;
      default: // ACTOR_PASS, PROCESS_PASS
        vtkShaderProgram::Substitute(fragmentShader, "//VTK::Picking::Dec",
          vtkvolume::PickingActorPassDeclaration(ren, this, vol));

        vtkShaderProgram::Substitute(
          fragmentShader, "//VTK::Picking::Exit", vtkvolume::PickingActorPassExit(ren, this, vol));
        break;
    }
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::ReplaceShaderRTT(
  std::map<vtkShader::Type, vtkShader*>& shaders, vtkRenderer* ren, vtkVolume* vol,
  int vtkNotUsed(numComps))
{
  vtkShader* fragmentShader = shaders[vtkShader::Fragment];

  if (this->RenderToImage)
  {
    vtkShaderProgram::Substitute(fragmentShader, "//VTK::RenderToImage::Dec",
      vtkvolume::RenderToImageDeclarationFragment(ren, this, vol));

    vtkShaderProgram::Substitute(
      fragmentShader, "//VTK::RenderToImage::Init", vtkvolume::RenderToImageInit(ren, this, vol));

    vtkShaderProgram::Substitute(fragmentShader, "//VTK::RenderToImage::Impl",
      vtkvolume::RenderToImageImplementation(ren, this, vol));

    vtkShaderProgram::Substitute(
      fragmentShader, "//VTK::RenderToImage::Exit", vtkvolume::RenderToImageExit(ren, this, vol));
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::ReplaceShaderValues(
  std::map<vtkShader::Type, vtkShader*>& shaders, vtkRenderer* ren, vtkVolume* vol,
  int noOfComponents)
{
  // Every volume should have a property (cannot be nullptr);
  vtkVolumeProperty* volumeProperty = vol->GetProperty();
  auto shaderProperty = vtkOpenGLShaderProperty::SafeDownCast(vol->GetShaderProperty());

  this->Impl->TotalNumberOfLights = 0;
  this->Impl->NumberPositionalLights = 0;
  this->Impl->DefaultLighting = false;

  if (volumeProperty->GetShade())
  {
    vtkLightCollection* lc = ren->GetLights();
    vtkLight* light;

    // Compute light information.
    vtkCollectionSimpleIterator sit;
    for (lc->InitTraversal(sit); (light = lc->GetNextLight(sit));)
    {
      float status = light->GetSwitch();
      if (status > 0.0)
      {
        if (this->Impl->TotalNumberOfLights == 0)
        {
          // we set default lighting to true for the first light
          this->Impl->DefaultLighting = true;
        }
        this->Impl->TotalNumberOfLights++;
        if (light->GetPositional())
        {
          this->Impl->NumberPositionalLights++;
        }
      }

      if (this->Impl->DefaultLighting &&
        (this->Impl->TotalNumberOfLights > 1 || light->GetIntensity() != 1.0 ||
          light->GetLightType() != VTK_LIGHT_TYPE_HEADLIGHT))
      {
        this->Impl->DefaultLighting = false;
      }
    }
  }

  // Render pass pre replacements
  //---------------------------------------------------------------------------
  this->ReplaceShaderRenderPass(shaders, vol, true);

  // Custom uniform variables replacements
  //---------------------------------------------------------------------------
  this->ReplaceShaderCustomUniforms(shaders, shaderProperty);

  // Base methods replacements
  //---------------------------------------------------------------------------
  this->ReplaceShaderBase(shaders, ren, vol, noOfComponents);

  // Termination methods replacements
  //---------------------------------------------------------------------------
  this->ReplaceShaderTermination(shaders, ren, vol, noOfComponents);

  // Shading methods replacements
  //---------------------------------------------------------------------------
  this->ReplaceShaderShading(shaders, ren, vol, noOfComponents);

  // Compute methods replacements
  //---------------------------------------------------------------------------
  this->ReplaceShaderCompute(shaders, ren, vol, noOfComponents);

  // Cropping methods replacements
  //---------------------------------------------------------------------------
  this->ReplaceShaderCropping(shaders, ren, vol, noOfComponents);

  // Clipping methods replacements
  //---------------------------------------------------------------------------
  this->ReplaceShaderClipping(shaders, ren, vol, noOfComponents);

  // Masking methods replacements
  //---------------------------------------------------------------------------
  this->ReplaceShaderMasking(shaders, ren, vol, noOfComponents);

  // Picking replacements
  //---------------------------------------------------------------------------
  this->ReplaceShaderPicking(shaders, ren, vol, noOfComponents);

  // Render to texture
  //---------------------------------------------------------------------------
  this->ReplaceShaderRTT(shaders, ren, vol, noOfComponents);

  // Set number of isosurfaces
  if (this->GetBlendMode() == vtkVolumeMapper::ISOSURFACE_BLEND)
  {
    std::ostringstream ss;
    ss << volumeProperty->GetIsoSurfaceValues()->GetNumberOfContours();
    vtkShaderProgram::Substitute(shaders[vtkShader::Fragment], "NUMBER_OF_CONTOURS", ss.str());
  }

  // Render pass post replacements
  //---------------------------------------------------------------------------
  this->ReplaceShaderRenderPass(shaders, vol, false);
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::BuildShader(vtkRenderer* ren)
{
  std::map<vtkShader::Type, vtkShader*> shaders;
  vtkShader* vertexShader = vtkShader::New();
  vertexShader->SetType(vtkShader::Vertex);
  shaders[vtkShader::Vertex] = vertexShader;
  vtkShader* fragmentShader = vtkShader::New();
  fragmentShader->SetType(vtkShader::Fragment);
  shaders[vtkShader::Fragment] = fragmentShader;
  vtkShader* geometryShader = vtkShader::New();
  geometryShader->SetType(vtkShader::Geometry);
  shaders[vtkShader::Geometry] = geometryShader;

  auto vol = this->Impl->GetActiveVolume();

  vtkOpenGLShaderProperty* sp = vtkOpenGLShaderProperty::SafeDownCast(vol->GetShaderProperty());
  this->GetShaderTemplate(shaders, sp);

  // user specified pre replacements
  vtkOpenGLShaderProperty::ReplacementMap repMap = sp->GetAllShaderReplacements();
  for (const auto& i : repMap)
  {
    if (i.first.ReplaceFirst)
    {
      std::string ssrc = shaders[i.first.ShaderType]->GetSource();
      vtkShaderProgram::Substitute(
        ssrc, i.first.OriginalValue, i.second.Replacement, i.second.ReplaceAll);
      shaders[i.first.ShaderType]->SetSource(ssrc);
    }
  }

  auto numComp = this->AssembledInputs[0].Texture->GetLoadedScalars()->GetNumberOfComponents();

  // TEMP DEBUG: hook the fragment shader to dump the interpolated ray geometry
  // (g_rayOrigin / g_dirStep) for the gated pixels. Enabled by VTK_GL_RAY_DUMP.
  if (this->Impl->DebugRayDump)
  {
    vtkShader* fragmentShader = shaders[vtkShader::Fragment];
    std::string fragSrc = fragmentShader->GetSource();
    vtkShaderProgram::Substitute(
      fragSrc, "in vec3 ip_vertexPos;",
      "in vec3 ip_vertexPos;\n"
      "in vec4 ip_debugClip;\n"
      "flat in vec4 ip_debugClipFlat;\n"
      "flat in int ip_vid;\n"
      "uniform vec2 in_debugPixel;\n"
      "uniform int in_debugChannel;\n"
      "uniform int in_debugSample;\n"
      "uniform vec3 in_debugTexel;\n"
      "uniform vec3 in_debugTexelDims;\n"
      "int g_dbgIter = 0;\n"
      "bool g_dbgDone = false;");
    vtkShaderProgram::Substitute(fragSrc, "//VTK::CallWorker::Impl",
      "if (in_debugTexel.x >= 0.0)\n"
      "  {\n"
      "    if (all(lessThan(abs(gl_FragCoord.xy - in_debugPixel), vec2(0.5))))\n"
      "    {\n"
      "      vec3 tc = (in_debugTexel + 0.5) / in_debugTexelDims;\n"
      "      float tv = texture3D(in_volume[0], tc).r;\n"
      "      float tva = abs(tv);\n"
      "      float b0 = 0.0;\n"
      "      float b1 = 0.0;\n"
      "      float b2 = 0.0;\n"
      "      float b3 = 0.0;\n"
      "      if (tva != 0.0)\n"
      "      {\n"
      "        float e = floor(log2(tva));\n"
      "        float f = tva * exp2(-e);\n"
      "        if (f >= 2.0)\n"
      "        {\n"
      "          f *= 0.5;\n"
      "          e += 1.0;\n"
      "        }\n"
      "        if (f < 1.0)\n"
      "        {\n"
      "          f *= 2.0;\n"
      "          e -= 1.0;\n"
      "        }\n"
      "        float m23 = floor((f - 1.0) * 8388608.0);\n"
      "        b0 = mod(m23, 256.0);\n"
      "        b1 = mod(floor(m23 / 256.0), 256.0);\n"
      "        b2 = floor(m23 / 65536.0);\n"
      "        b3 = clamp(e + 64.0, 0.0, 127.0);\n"
      "      }\n"
      "      fragOutput0 = vec4(b0, b1, b2, b3) / 255.0;\n"
      "      return;\n"
      "    }\n"
      "  }\n"
       "  if (in_debugChannel >= 100 && in_debugChannel < 110 && in_debugSample < 0)\n"
      "  {\n"
      "    int cf = in_debugChannel - 100;\n"
      "    float v = (cf == 0) ? float(ip_vid) : ((cf == 1) ? ip_debugClipFlat.x :\n"
      "      ((cf == 2) ? ip_debugClipFlat.y : ((cf == 3) ? ip_debugClipFlat.z :\n"
      "      ((cf == 4) ? ip_debugClipFlat.w : ((cf == 5) ? ip_vertexPos.x :\n"
      "      ((cf == 6) ? ip_vertexPos.y : ((cf == 7) ? float(gl_PrimitiveID) :\n"
      "      ((cf == 8) ? ip_textureCoords.x : ((cf == 9) ? ip_textureCoords.y :\n"
      "      ((cf == 10) ? ip_textureCoords.z :\n"
      "      ip_vertexPos.z))))))))));\n"
      "    float av = abs(v);\n"
      "    float b0 = 0.0;\n"
      "    float b1 = 0.0;\n"
      "    float b2 = 0.0;\n"
      "    float b3 = 0.0;\n"
      "    if (av != 0.0)\n"
      "    {\n"
      "      float e = 0.0;\n"
      "      while (av >= 2.0) { av *= 0.5; e += 1.0; }\n"
      "      while (av < 1.0) { av *= 2.0; e -= 1.0; }\n"
      "      float m23 = floor((av - 1.0) * 8388608.0);\n"
      "      b0 = mod(m23, 256.0);\n"
      "      b1 = mod(floor(m23 / 256.0), 256.0);\n"
      "      b2 = floor(m23 / 65536.0);\n"
      "      b3 = clamp(e + 64.0, 0.0, 127.0);\n"
      "      if (v < 0.0) b3 += 128.0;\n"
      "    }\n"
      "    fragOutput0 = vec4(b0, b1, b2, b3) / 255.0;\n"
      "    return;\n"
      "  }\n"
  "  initializeRayCast();\n"
      "  if (in_debugChannel >= 200 && in_debugChannel < 203 && in_debugSample < 0)\n"
      "  {\n"
      "    int af = in_debugChannel - 200;\n"
      "    if (af == 0)\n"
      "    {\n"
      "      fragOutput0 = vec4(ip_textureCoords.xyz, float(ip_vid));\n"
      "    }\n"
      "    else if (af == 1)\n"
      "    {\n"
      "      fragOutput0 = ip_debugClip;\n"
      "    }\n"
      "    else\n"
      "    {\n"
      "      fragOutput0 = vec4(ip_vertexPos.xyz, float(gl_PrimitiveID));\n"
      "    }\n"
      "    return;\n"
      "  }\n"
  "  if (in_debugChannel >= 0 && in_debugChannel < 60 && in_debugSample < 0)\n"
  "  {\n"
      "    if (all(lessThan(abs(gl_FragCoord.xy - in_debugPixel), vec2(0.5))))\n"
      "    {\n"
      "      int field = (in_debugChannel < 16) ? (in_debugChannel % 3) : ((in_debugChannel - 16) % 3);\n"
      "      vec3 base = (in_debugChannel < 3) ? g_rayOrigin : ((in_debugChannel < 6) ? g_dirStep :\n"
      "        ((in_debugChannel < 9) ? ip_vertexPos : ((in_debugChannel < 12) ? ip_textureCoords :\n"
      "        ((in_debugChannel < 16) ? vec3(0.0) : ((in_debugChannel < 19) ? g_dbgRayDir :\n"
      "        ((in_debugChannel < 22) ? g_dbgNearP : ((in_debugChannel < 25) ? g_dbgFarP :\n"
      "        ((in_debugChannel < 33) ? vec3(0.0) : ((in_debugChannel < 36) ? g_dbgDir :\n"
      "        ((in_debugChannel < 43) ? g_dbgRayDir2 : g_dbgNearP))))))))));\n"
      "      float v = (field == 0) ? base.x : ((field == 1) ? base.y : base.z);\n"
      "      if (in_debugChannel >= 12 && in_debugChannel < 16)\n"
      "      {\n"
      "        int cf = in_debugChannel - 12;\n"
      "        v = (cf == 0) ? ip_debugClip.x : ((cf == 1) ? ip_debugClip.y :\n"
      "          ((cf == 2) ? ip_debugClip.z : ip_debugClip.w));\n"
      "      }\n"
      "      else if (in_debugChannel >= 25 && in_debugChannel < 29)\n"
      "      {\n"
      "        int cf = in_debugChannel - 25;\n"
      "        v = (cf == 0) ? g_dbgNearPRaw.x : ((cf == 1) ? g_dbgNearPRaw.y :\n"
      "          ((cf == 2) ? g_dbgNearPRaw.z : g_dbgNearPRaw.w));\n"
      "      }\n"
      "      else if (in_debugChannel >= 29 && in_debugChannel < 33)\n"
      "      {\n"
      "        int cf = in_debugChannel - 29;\n"
      "        v = (cf == 0) ? g_dbgFarPRaw.x : ((cf == 1) ? g_dbgFarPRaw.y :\n"
      "          ((cf == 2) ? g_dbgFarPRaw.z : g_dbgFarPRaw.w));\n"
      "      }\n"
      "      else if (in_debugChannel == 36)\n"
      "      {\n"
      "        v = g_dbgD2;\n"
      "      }\n"
      "      else if (in_debugChannel == 37)\n"
      "      {\n"
      "        v = g_dbgInv;\n"
      "      }\n"
      "      else if (in_debugChannel == 38)\n"
      "      {\n"
      "        v = g_dbgNearPRaw.w;\n"
      "      }\n"
      "      else if (in_debugChannel == 39)\n"
      "      {\n"
      "        v = g_dbgFarPRaw.w;\n"
      "      }\n"
      "      float av = abs(v);\n"
      "      float b0 = 0.0;\n"
      "      float b1 = 0.0;\n"
      "      float b2 = 0.0;\n"
      "      float b3 = 0.0;\n"
      "      if (av != 0.0)\n"
      "      {\n"
      "        float e = floor(log2(av));\n"
      "        float f = av * exp2(-e);\n"
      "        if (f >= 2.0)\n"
      "        {\n"
      "          f *= 0.5;\n"
      "          e += 1.0;\n"
      "        }\n"
      "        if (f < 1.0)\n"
      "        {\n"
      "          f *= 2.0;\n"
      "          e -= 1.0;\n"
      "        }\n"
      "        float m23 = floor((f - 1.0) * 8388608.0);\n"
      "        b0 = mod(m23, 256.0);\n"
      "        b1 = mod(floor(m23 / 256.0), 256.0);\n"
      "        b2 = floor(m23 / 65536.0);\n"
      "        b3 = clamp(e + 64.0, 0.0, 127.0);\n"
      "        if (v < 0.0)\n"
      "        {\n"
      "          b3 += 128.0;\n"
      "        }\n"
      "      }\n"
      "      fragOutput0 = vec4(b0, b1, b2, b3) / 255.0;\n"
      "      return;\n"
      "    }\n"
      "  }\n"
      "  g_dbgIter = 0;\n"
      "  g_dbgDone = false;\n"
      "  castRay(-1.0, -1.0);\n"
      "  if (g_dbgDone)\n"
      "  {\n"
      "    return;\n"
      "  }\n"
      "  vec4 g_dbgPreFinal = g_fragColor;\n"
      "  finalizeRayCast();\n"
      "  if (in_debugChannel >= 60 && in_debugSample < 0)\n"
      "  {\n"
      "    if (all(lessThan(abs(gl_FragCoord.xy - in_debugPixel), vec2(0.5))))\n"
      "    {\n"
      "      int cf = in_debugChannel - 60;\n"
      "      float val = (cf < 4) ? ((cf == 0) ? g_dbgPreFinal.r : ((cf == 1) ? g_dbgPreFinal.g :\n"
      "        ((cf == 2) ? g_dbgPreFinal.b : g_dbgPreFinal.a))) :\n"
      "        ((cf == 4) ? g_fragColor.r : ((cf == 5) ? g_fragColor.g :\n"
      "        ((cf == 6) ? g_fragColor.b : g_fragColor.a)));\n"
      "      float av = abs(val);\n"
      "      float b0 = 0.0;\n"
      "      float b1 = 0.0;\n"
      "      float b2 = 0.0;\n"
      "      float b3 = 0.0;\n"
      "      if (av != 0.0)\n"
      "      {\n"
      "        float e = floor(log2(av));\n"
      "        float f = av * exp2(-e);\n"
      "        if (f >= 2.0)\n"
      "        {\n"
      "          f *= 0.5;\n"
      "          e += 1.0;\n"
      "        }\n"
      "        if (f < 1.0)\n"
      "        {\n"
      "          f *= 2.0;\n"
      "          e -= 1.0;\n"
      "        }\n"
      "        float m23 = floor((f - 1.0) * 8388608.0);\n"
      "        b0 = mod(m23, 256.0);\n"
      "        b1 = mod(floor(m23 / 256.0), 256.0);\n"
      "        b2 = floor(m23 / 65536.0);\n"
      "        b3 = clamp(e + 64.0, 0.0, 127.0);\n"
      "        if (val < 0.0)\n"
      "        {\n"
      "          b3 += 128.0;\n"
      "        }\n"
      "      }\n"
      "      fragOutput0 = vec4(b0, b1, b2, b3) / 255.0;\n"
      "      return;\n"
      "    }\n"
      "  }\n");
    vtkShaderProgram::Substitute(fragSrc, "//VTK::RenderToImage::Impl",
      "//VTK::RenderToImage::Impl\n"
      "  if (in_debugSample >= 0 && !g_dbgDone)\n"
      "  {\n"
      "    if (all(lessThan(abs(gl_FragCoord.xy - in_debugPixel), vec2(0.5))))\n"
      "    {\n"
      "      if (g_dbgIter == in_debugSample)\n"
      "      {\n"
      "        float dbgRaw = texture3D(in_volume[0], g_dataPos).r;\n"
      "        float val = (in_debugChannel == 1) ? g_dataPos.x :\n"
      "                    (in_debugChannel == 2) ? g_dataPos.y :\n"
      "                    (in_debugChannel == 3) ? g_dataPos.z :\n"
      "                    (in_debugChannel == 4) ? g_srcColor.r :\n"
      "                    (in_debugChannel == 5) ? g_srcColor.g :\n"
      "                    (in_debugChannel == 6) ? g_srcColor.b :\n"
      "                    (in_debugChannel == 7) ? g_srcColor.a : dbgRaw;\n"
      "        float av = abs(val);\n"
      "        float b0 = 0.0;\n"
      "        float b1 = 0.0;\n"
      "        float b2 = 0.0;\n"
      "        float b3 = 0.0;\n"
      "        if (av != 0.0)\n"
      "        {\n"
      "          float e = floor(log2(av));\n"
      "          float f = av * exp2(-e);\n"
      "          if (f >= 2.0)\n"
      "          {\n"
      "            f *= 0.5;\n"
      "            e += 1.0;\n"
      "          }\n"
      "          if (f < 1.0)\n"
      "          {\n"
      "            f *= 2.0;\n"
      "            e -= 1.0;\n"
      "          }\n"
      "          float m23 = floor((f - 1.0) * 8388608.0);\n"
      "          b0 = mod(m23, 256.0);\n"
      "          b1 = mod(floor(m23 / 256.0), 256.0);\n"
      "          b2 = floor(m23 / 65536.0);\n"
      "          b3 = clamp(e + 64.0, 0.0, 127.0);\n"
      "          if (val < 0.0)\n"
      "          {\n"
      "            b3 += 128.0;\n"
      "          }\n"
      "        }\n"
      "        fragOutput0 = vec4(b0, b1, b2, b3) / 255.0;\n"
      "        g_dbgDone = true;\n"
      "      }\n"
      "    }\n"
      "    g_dbgIter++;\n"
      "  }\n");
    fragmentShader->SetSource(fragSrc);

    std::cerr << "VTK_METAL_VOLUME_LOG === FRAG SHADER ===\n" << fragSrc
              << "\n=== FRAG SHADER END ===\n";
    std::cerr << "VTK_METAL_VOLUME_LOG === VERT SHADER ===\n"
              << shaders[vtkShader::Vertex]->GetSource() << "\n=== VERT SHADER END ===\n";
  }

  this->ReplaceShaderValues(shaders, ren, vol, numComp);
  if (this->Impl->DebugRayDump)
  {
    std::cerr << "VTK_METAL_VOLUME_LOG === COMPILED FRAG ===\n"
              << shaders[vtkShader::Fragment]->GetSource() << "\n=== COMPILED FRAG END ===\n";
  }

  // user specified post replacements
  for (const auto& i : repMap)
  {
    if (!i.first.ReplaceFirst)
    {
      std::string ssrc = shaders[i.first.ShaderType]->GetSource();
      vtkShaderProgram::Substitute(
        ssrc, i.first.OriginalValue, i.second.Replacement, i.second.ReplaceAll);
      shaders[i.first.ShaderType]->SetSource(ssrc);
    }
  }

  // Now compile the shader
  //--------------------------------------------------------------------------
  this->Impl->ShaderProgram = this->Impl->ShaderCache->ReadyShaderProgram(shaders);
  if (!this->Impl->ShaderProgram || !this->Impl->ShaderProgram->GetCompiled())
  {
    vtkErrorMacro("Shader failed to compile");
  }

  vertexShader->Delete();
  fragmentShader->Delete();
  geometryShader->Delete();

  this->Impl->ShaderBuildTime.Modified();
}

//------------------------------------------------------------------------------
// Update the reduction factor of the render viewport (this->ReductionFactor)
// according to the time spent in seconds to render the previous frame
// (this->TimeToDraw) and a time in seconds allocated to render the next
// frame (allocatedTime).
// \pre valid_current_reduction_range: this->ReductionFactor>0.0 &&
// this->ReductionFactor<=1.0 \pre positive_TimeToDraw: this->TimeToDraw>=0.0
// \pre positive_time: allocatedTime>0.0
// \post valid_new_reduction_range: this->ReductionFactor>0.0 &&
// this->ReductionFactor<=1.0
//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::ComputeReductionFactor(double allocatedTime)
{
  if (!this->AutoAdjustSampleDistances)
  {
    this->ReductionFactor = 1.0 / this->ImageSampleDistance;
    return;
  }

  if (this->TimeToDraw)
  {
    double oldFactor = this->ReductionFactor;

    double timeToDraw;
    if (allocatedTime < 1.0)
    {
      timeToDraw = this->SmallTimeToDraw;
      if (timeToDraw == 0.0)
      {
        timeToDraw = this->BigTimeToDraw / 3.0;
      }
    }
    else
    {
      timeToDraw = this->BigTimeToDraw;
    }

    // This should be the case when rendering the volume very first time
    // 10.0 is an arbitrary value chosen which happen to a large number
    // in this context
    if (timeToDraw == 0.0)
    {
      timeToDraw = 10.0;
    }

    double fullTime = timeToDraw / this->ReductionFactor;
    double newFactor = allocatedTime / fullTime;

    // Compute average factor
    this->ReductionFactor = (newFactor + oldFactor) / 2.0;

    // Discretize reduction factor so that it doesn't cause
    // visual artifacts when used to reduce the sample distance
    this->ReductionFactor = (this->ReductionFactor > 1.0) ? 1.0 : (this->ReductionFactor);

    if (this->ReductionFactor < 0.20)
    {
      this->ReductionFactor = 0.10;
    }
    else if (this->ReductionFactor < 0.50)
    {
      this->ReductionFactor = 0.20;
    }
    else if (this->ReductionFactor < 1.0)
    {
      this->ReductionFactor = 0.50;
    }

    // Clamp it
    if (1.0 / this->ReductionFactor > this->MaximumImageSampleDistance)
    {
      this->ReductionFactor = 1.0 / this->MaximumImageSampleDistance;
    }
    if (1.0 / this->ReductionFactor < this->MinimumImageSampleDistance)
    {
      this->ReductionFactor = 1.0 / this->MinimumImageSampleDistance;
    }
  }
}

//------------------------------------------------------------------------------
bool vtkOpenGLGPUVolumeRayCastMapper::PreLoadData(vtkRenderer* ren, vtkVolume* vol)
{
  if (!this->ValidateRender(ren, vol))
  {
    return false;
  }

  // have to register if we preload
  this->ResourceCallback->RegisterGraphicsResources(
    static_cast<vtkOpenGLRenderWindow*>(ren->GetVTKWindow()));

  this->Impl->ClearRemovedInputs(ren->GetRenderWindow());
  return this->Impl->UpdateInputs(ren, vol);
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::ForceTransferInit()
{
  auto& inputs = this->Parent->AssembledInputs;
  auto fu = [](std::pair<const int, vtkVolumeInputHelper>& p) { p.second.ForceTransferInit(); };
  std::for_each(inputs.begin(), inputs.end(), fu);
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::ClearRemovedInputs(vtkWindow* win)
{
  bool orderChanged = false;
  for (const int& port : this->Parent->RemovedPorts)
  {
    auto it = this->Parent->AssembledInputs.find(port);
    if (it == this->Parent->AssembledInputs.cend())
    {
      continue;
    }

    auto input = (*it).second;
    if (input.Texture)
    {
      input.Texture->ReleaseGraphicsResources(win);
    }
    if (input.GradientOpacityTables)
    {
      input.GradientOpacityTables->ReleaseGraphicsResources(win);
    }
    if (input.OpacityTables)
    {
      input.OpacityTables->ReleaseGraphicsResources(win);
    }
    if (input.RGBTables)
    {
      input.RGBTables->ReleaseGraphicsResources(win);
    }
    this->Parent->AssembledInputs.erase(it);
    orderChanged = true;
  }
  this->Parent->RemovedPorts.clear();

  if (orderChanged)
  {
    this->ForceTransferInit();
  }
}

//------------------------------------------------------------------------------
bool vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::UpdateInputs(vtkRenderer* ren, vtkVolume* vol)
{
  this->VolumePropertyChanged = false;
  bool orderChanged = false;
  bool success = true;
  for (const auto& port : this->Parent->Ports)
  {
    if (this->MultiVolume)
    {
      vol = this->MultiVolume->GetVolume(port);
    }
    auto property = vol->GetProperty();
    auto input = this->Parent->GetTransformedInput(port);

    // Check for property changes
    this->VolumePropertyChanged |= property->GetMTime() > this->ShaderBuildTime.GetMTime();

    auto it = this->Parent->AssembledInputs.find(port);
    if (it == this->Parent->AssembledInputs.cend() || it->second.Volume != vol)
    {
      // Create new input structure
      auto texture = vtkSmartPointer<vtkVolumeTexture>::New();

      VolumeInput currentInput(texture, vol);
      this->Parent->AssembledInputs[port] = std::move(currentInput);
      orderChanged = true;

      it = this->Parent->AssembledInputs.find(port);
    }
    assert(it != this->Parent->AssembledInputs.cend());

    /// TODO Currently, only input arrays with the same name/id/mode can be
    // (across input objects) can be rendered. This could be addressed by
    // overriding the mapper's settings with array settings defined in the
    // vtkMultiVolume instance.
    vtkDataArray* scalars = vtkOpenGLGPUVolumeRayCastMapper::GetScalars(input,
      this->Parent->ScalarMode, this->Parent->ArrayAccessMode, this->Parent->ArrayId,
      this->Parent->ArrayName, this->Parent->CellFlag);

    if (this->NeedToInitializeResources || (input->GetMTime() > it->second.Texture->UploadTime) ||
      (scalars != it->second.Texture->GetLoadedScalars()) ||
      (scalars != nullptr && scalars->GetMTime() > it->second.Texture->UploadTime))
    {
      auto& volInput = this->Parent->AssembledInputs[port];
      auto volumeTex = volInput.Texture.GetPointer();
      volumeTex->SetPartitions(this->Partitions[0], this->Partitions[1], this->Partitions[2]);
      success &= volumeTex->LoadVolume(
        ren, input, scalars, this->Parent->CellFlag, property->GetInterpolationType());
      volInput.ComponentMode = this->GetComponentMode(property, scalars);
    }
    else
    {
      // Update vtkVolumeTexture
      it->second.Texture->UpdateVolume(property);
    }

    // Volume may have changed, so make sure the helper updates its reference to it.
    it->second.Volume = vol;
  }

  if (orderChanged)
  {
    this->ForceTransferInit();
  }

  if (this->MultiVolume)
  {
    bool hasGradient = this->Parent->AssembledInputs[0].Volume->GetProperty()->HasGradientOpacity();
    for (auto& item : this->Parent->AssembledInputs)
    {
      if (item.second.Volume->GetProperty()->HasGradientOpacity() != hasGradient)
      {
        vtkGenericWarningMacro(
          "Current implementation of vtkOpenGLGPUVolumeRayCastMapper does not support MultiVolume "
          "where some volumes have a gradient opacity function and some others don't. "
          "Rendering of the MultiVolume is disabled.");
        success = false;
        break;
      }
    }
  }

  return success;
}

//----------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::UpdateTransfer2DYAxisArray(
  vtkRenderer* ren, vtkVolume* vol)
{
  vtkVolumeProperty* prop = vol->GetProperty();
  vtkImageData* input = vtkImageData::SafeDownCast(this->Parent->GetTransformedInput(0));
  if (prop->GetTransferFunctionMode() != vtkVolumeProperty::TF_2D)
  {
    this->Transfer2DUseGradient = true;
    return;
  }

  bool hasTransfer2DYAxisArray = this->Parent->GetTransfer2DYAxisArray() != nullptr;
  bool isCellData = hasTransfer2DYAxisArray &&
    input->GetCellData()->HasArray(this->Parent->GetTransfer2DYAxisArray());
  bool isPointData = hasTransfer2DYAxisArray &&
    input->GetPointData()->HasArray(this->Parent->GetTransfer2DYAxisArray());
  hasTransfer2DYAxisArray = hasTransfer2DYAxisArray && (isCellData || isPointData);
  if (!hasTransfer2DYAxisArray)
  {
    this->Transfer2DUseGradient = true;
    return;
  }
  else
  {
    this->Transfer2DUseGradient = false;
  }

  // Now load the array
  if (!this->Transfer2DYAxisScalars)
  {
    this->Transfer2DYAxisScalars = vtkSmartPointer<vtkVolumeTexture>::New();

    const auto part = this->Partitions;
    this->Transfer2DYAxisScalars->SetPartitions(part[0], part[1], part[2]);
  }
  vtkDataArray* arr = nullptr;
  if (isPointData)
  {
    arr = input->GetPointData()->GetArray(this->Parent->GetTransfer2DYAxisArray());
  }
  else
  {
    arr = input->GetCellData()->GetArray(this->Parent->GetTransfer2DYAxisArray());
  }

  if (input->GetMTime() > this->Transfer2DYAxisScalarsUpdateTime ||
    this->Transfer2DYAxisScalars->GetLoadedScalars() != arr ||
    (arr && arr->GetMTime() > this->Transfer2DYAxisScalarsUpdateTime))
  {
    this->Transfer2DYAxisScalars->LoadVolume(
      ren, input, arr, isCellData, prop->GetInterpolationType());
    this->Transfer2DYAxisScalarsUpdateTime.Modified();
  }
}

//----------------------------------------------------------------------------
int vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::GetComponentMode(
  vtkVolumeProperty* prop, vtkDataArray* array) const
{
  if (prop->GetIndependentComponents())
  {
    return VolumeInput::INDEPENDENT;
  }
  else
  {
    const auto numComp = array->GetNumberOfComponents();
    if (numComp == 1 || numComp == 2)
      return VolumeInput::LA;
    else if (numComp == 4)
      return VolumeInput::RGBA;
    else if (numComp == 3)
    {
      vtkGenericWarningMacro(<< "3 dependent components (e.g. RGB) are not supported."
                                "Only 2 (LA) and 4 (RGBA) supported.");
      return VolumeInput::INVALID;
    }
    else
      return VolumeInput::INVALID;
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::GPURender(vtkRenderer* ren, vtkVolume* vol)
{
  vtkOpenGLClearErrorMacro();

  const bool isInteractive = vol->GetAllocatedRenderTime() < 1.0;
  const auto previousVolumetricScatteringBlending = this->VolumetricScatteringBlending;
  if (isInteractive)
  {
    // Turn off volumetric multi-scattering for interactive renders
    this->VolumetricScatteringBlending = 0;
  }

  vtkOpenGLCamera* cam = vtkOpenGLCamera::SafeDownCast(ren->GetActiveCamera());

  if (this->GetBlendMode() == vtkVolumeMapper::ISOSURFACE_BLEND &&
    vol->GetProperty()->GetIsoSurfaceValues()->GetNumberOfContours() == 0)
  {
    // Early exit: nothing to render.
    return;
  }

  vtkOpenGLRenderWindow* renWin = vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow());
  this->ResourceCallback->RegisterGraphicsResources(renWin);
  // Make sure the context is current
  renWin->MakeCurrent();

  // Get window size and corners
  this->Impl->CheckPropertyKeys(vol);
  if (!this->Impl->PreserveViewport)
  {
    ren->GetTiledSizeAndOrigin(this->Impl->WindowSize, this->Impl->WindowSize + 1,
      this->Impl->WindowLowerLeft, this->Impl->WindowLowerLeft + 1);
  }
  else
  {
    int vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    this->Impl->WindowLowerLeft[0] = vp[0];
    this->Impl->WindowLowerLeft[1] = vp[1];
    this->Impl->WindowSize[0] = vp[2];
    this->Impl->WindowSize[1] = vp[3];
  }

  this->Impl->NeedToInitializeResources =
    (this->Impl->ReleaseResourcesTime.GetMTime() > this->Impl->InitializationTime.GetMTime());

  this->ComputeReductionFactor(vol->GetAllocatedRenderTime());
  if (!this->Impl->SharedDepthTextureObject)
  {
    this->Impl->CaptureDepthTexture(ren);
  }

  vtkMTimeType renderPassTime = this->GetRenderPassStageMTime(vol);

  const auto multiVol = vtkMultiVolume::SafeDownCast(vol);
  this->Impl->MultiVolume = multiVol && this->GetInputCount() > 1 ? multiVol : nullptr;

  this->Impl->ClearRemovedInputs(renWin);
  if (!this->Impl->UpdateInputs(ren, vol))
  {
    return;
  }
  this->Impl->UpdateSamplingDistance(ren);
  this->Impl->UpdateTransfer2DYAxisArray(ren, vol);
  this->Impl->UpdateTransferFunctions(ren);

  // Masks are only supported on single-input rendering.
  if (!this->Impl->MultiVolume)
  {
    this->Impl->LoadMask(ren, vol);
  }

  // Get the shader cache. This is important to make sure that shader cache
  // knows the state of various shader programs in use.
  this->Impl->ShaderCache =
    vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow())->GetShaderCache();

  this->Impl->CheckPickingState(ren);

  if (this->UseDepthPass && this->GetBlendMode() == vtkVolumeMapper::COMPOSITE_BLEND)
  {
    this->Impl->RenderWithDepthPass(ren, cam, renderPassTime);
  }
  else
  {
    if (this->Impl->IsPicking && !this->Impl->MultiVolume)
    {
      this->Impl->BeginPicking(ren);
    }
    vtkVolumeStateRAII glState(renWin->GetState(), this->Impl->PreserveGLState);

    // Override the default depth mask value if the corresponding property key was specified.
    if (this->Impl->DepthMaskOverride)
    {
      renWin->GetState()->vtkglDepthMask(GL_TRUE);
    }

    if (this->Impl->ShaderRebuildNeeded(cam, vol, renderPassTime, ren))
    {
      this->Impl->LastProjectionParallel = cam->GetParallelProjection();
      this->BuildShader(ren);
    }
    else
    {
      // Bind the shader
      this->Impl->ShaderCache->ReadyShaderProgram(this->Impl->ShaderProgram);
      this->InvokeEvent(vtkCommand::UpdateShaderEvent, this->Impl->ShaderProgram);
    }

    vtkOpenGLShaderProperty* shaderProperty =
      vtkOpenGLShaderProperty::SafeDownCast(vol->GetShaderProperty());
    if (this->RenderToImage)
    {
      this->Impl->SetupRenderToTexture(ren);
      this->Impl->SetRenderToImageParameters(this->Impl->ShaderProgram);
      this->DoGPURender(ren, cam, this->Impl->ShaderProgram, shaderProperty);
      this->Impl->ExitRenderToTexture(ren);
    }
    else
    {
      this->Impl->BeginImageSample(ren);
      this->DoGPURender(ren, cam, this->Impl->ShaderProgram, shaderProperty);
      this->Impl->EndImageSample(ren);
    }

    if (this->Impl->IsPicking && !this->Impl->MultiVolume)
    {
      this->Impl->EndPicking(ren);
    }
  }

  glFinish();

  // Restore the previous scattering blending settings
  this->VolumetricScatteringBlending = previousVolumetricScatteringBlending;
}

//------------------------------------------------------------------------------
vtkMTimeType vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::LastModifiedLightTime(
  vtkLightCollection* lights)
{
  vtkMTimeType lastModified = lights->GetMTime();
  vtkLight* light;
  vtkCollectionSimpleIterator sit;
  for (lights->InitTraversal(sit); (light = lights->GetNextLight(sit));)
  {
    lastModified = std::max(lastModified, light->GetMTime());
  }
  return lastModified;
}

//------------------------------------------------------------------------------
bool vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::ShaderRebuildNeeded(
  vtkCamera* cam, vtkVolume* vol, vtkMTimeType renderPassTime, vtkRenderer* ren)
{
  return (this->NeedToInitializeResources || this->VolumePropertyChanged ||
    vol->GetShaderProperty()->GetShaderMTime() > this->ShaderBuildTime.GetMTime() ||
    this->Parent->GetMTime() > this->ShaderBuildTime.GetMTime() ||
    cam->GetParallelProjection() != this->LastProjectionParallel ||
    this->SelectionStateTime.GetMTime() > this->ShaderBuildTime.GetMTime() ||
    renderPassTime > this->ShaderBuildTime ||
    ren->GetLights()->GetMTime() > this->ShaderBuildTime.GetMTime() ||
    this->LastModifiedLightTime(ren->GetLights()) > this->ShaderBuildTime.GetMTime());
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::RenderWithDepthPass(
  vtkRenderer* ren, vtkOpenGLCamera* cam, vtkMTimeType renderPassTime)
{
  this->Parent->CurrentPass = DepthPass;
  auto& input = this->Parent->AssembledInputs[0];
  auto vol = input.Volume;
  auto volumeProperty = vol->GetProperty();
  auto shaderProperty = vtkOpenGLShaderProperty::SafeDownCast(vol->GetShaderProperty());

  if (this->NeedToInitializeResources ||
    volumeProperty->GetMTime() > this->DepthPassSetupTime.GetMTime() ||
    this->Parent->GetMTime() > this->DepthPassSetupTime.GetMTime() ||
    cam->GetParallelProjection() != this->LastProjectionParallel ||
    this->SelectionStateTime.GetMTime() > this->ShaderBuildTime.GetMTime() ||
    renderPassTime > this->ShaderBuildTime ||
    shaderProperty->GetShaderMTime() > this->ShaderBuildTime ||
    ren->GetLights()->GetMTime() > this->ShaderBuildTime.GetMTime() ||
    this->LastModifiedLightTime(ren->GetLights()) > this->ShaderBuildTime.GetMTime())
  {
    this->LastProjectionParallel = cam->GetParallelProjection();

    this->ContourFilter->SetInputData(this->Parent->GetTransformedInput(0));
    for (vtkIdType i = 0; i < this->Parent->GetDepthPassContourValues()->GetNumberOfContours(); ++i)
    {
      this->ContourFilter->SetValue(i, this->Parent->DepthPassContourValues->GetValue(i));
    }

    this->RenderContourPass(ren);
    this->DepthPassSetupTime.Modified();
    this->Parent->BuildShader(ren);
  }
  else if (cam->GetMTime() > this->DepthPassTime.GetMTime())
  {
    this->RenderContourPass(ren);
  }

  if (this->IsPicking)
  {
    this->BeginPicking(ren);
  }

  // Set OpenGL states
  vtkOpenGLRenderWindow* renWin = vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow());
  vtkVolumeStateRAII glState(renWin->GetState(), this->PreserveGLState);

  // Override the default depth mask value if the corresponding property key was specified.
  if (this->DepthMaskOverride)
  {
    renWin->GetState()->vtkglDepthMask(GL_TRUE);
  }

  if (this->Parent->RenderToImage)
  {
    this->SetupRenderToTexture(ren);
  }

  if (!this->PreserveViewport)
  {
    // NOTE: This is a must call or else, multiple viewport rendering would
    // not work. The glViewport could have been changed by any of the internal
    // FBOs (RenderToTexture, etc.).  The viewport should (ideally) not be set
    // within the mapper, because it could cause issues when vtkOpenGLRenderPass
    // instances modify it too (this is a workaround for that).
    renWin->GetState()->vtkglViewport(
      this->WindowLowerLeft[0], this->WindowLowerLeft[1], this->WindowSize[0], this->WindowSize[1]);
  }

  renWin->GetShaderCache()->ReadyShaderProgram(this->ShaderProgram);
  this->Parent->InvokeEvent(vtkCommand::UpdateShaderEvent, this->ShaderProgram);

  this->DPDepthBufferTextureObject->Activate();
  this->ShaderProgram->SetUniformi(
    "in_depthPassSampler", this->DPDepthBufferTextureObject->GetTextureUnit());
  this->Parent->DoGPURender(ren, cam, this->ShaderProgram, shaderProperty);
  this->DPDepthBufferTextureObject->Deactivate();

  if (this->IsPicking)
  {
    this->EndPicking(ren);
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::BindTransformations(
  vtkShaderProgram* prog, vtkMatrix4x4* modelViewMat)
{
  // Bind transformations. Because the bounding box has its own transformations,
  // it is considered here as an actual volume (numInputs + 1).
  const int numInputs = static_cast<int>(this->Parent->AssembledInputs.size());
  const int numVolumes = this->MultiVolume ? numInputs + 1 : numInputs;

  this->VolMatVec.resize(numVolumes * 16, 0);
  this->InvMatVec.resize(numVolumes * 16, 0);
  this->TexMatVec.resize(numVolumes * 16, 0);
  this->InvTexMatVec.resize(numVolumes * 16, 0);
  this->TexEyeMatVec.resize(numVolumes * 16, 0);
  this->CellToPointVec.resize(numVolumes * 16, 0);
  this->TexMinVec.resize(numVolumes * 3, 0);
  this->TexMaxVec.resize(numVolumes * 3, 0);
  this->EyePosVec.resize(numVolumes * 3, 0);

  vtkNew<vtkMatrix4x4> dataToWorld, dataToView, texToDataMat, texToViewMat, cellToPointMat;
  float defaultTexMin[3] = { 0.f, 0.f, 0.f };
  float defaultTexMax[3] = { 1.f, 1.f, 1.f };
  float eyePos[3] = { 0.f, 0.f, 0.f };

  auto it = this->Parent->AssembledInputs.begin();
  for (int i = 0; i < numVolumes; i++)
  {
    const int vecOffset = i * 16;
    float *texMin, *texMax;

    if (this->MultiVolume && i == 0)
    {
      // Bounding box
      auto bBoxToWorld = this->MultiVolume->GetMatrix();
      dataToWorld->DeepCopy(bBoxToWorld);

      auto texToBBox = this->MultiVolume->GetTextureMatrix();
      texToDataMat->DeepCopy(texToBBox);

      cellToPointMat->Identity();
      texMin = defaultTexMin;
      texMax = defaultTexMax;
    }
    else
    {
      // Volume inputs
      auto& inputData = (*it).second;
      it++;
      auto volTex = inputData.Texture;
      inputData.Volume->GetModelToWorldMatrix(this->TempMatrix4x4);
      vtkMatrix4x4* volMatrix = this->TempMatrix4x4;
      dataToWorld->DeepCopy(volMatrix);
      texToDataMat->DeepCopy(volTex->GetCurrentBlock()->TextureToDataset.GetPointer());

      {
        // TEMP DEBUG: probe the actual CellToPointMatrix source for parity.
        std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_CTP "
                  << volTex->CellToPointMatrix->Element[0][0] << ","
                  << volTex->CellToPointMatrix->Element[1][1] << ","
                  << volTex->CellToPointMatrix->Element[2][2] << ","
                  << volTex->CellToPointMatrix->Element[0][3] << ","
                  << volTex->CellToPointMatrix->Element[1][3] << ","
                  << volTex->CellToPointMatrix->Element[2][3] << "\n";
      }

      // Texture matrices (texture to view)
      vtkMatrix4x4::Multiply4x4(volMatrix, texToDataMat.GetPointer(), texToViewMat.GetPointer());
      vtkMatrix4x4::Multiply4x4(modelViewMat, texToViewMat.GetPointer(), texToViewMat.GetPointer());

      // texToViewMat->Transpose();
      vtkInternal::CopyMatrixToVector<vtkMatrix4x4, 4, 4>(
        texToViewMat.GetPointer(), this->TexEyeMatVec.data(), vecOffset);

      // Cell to Point (texture-cells to texture-points)
      cellToPointMat->DeepCopy(volTex->CellToPointMatrix.GetPointer());
      texMin = volTex->AdjustedTexMin;
      texMax = volTex->AdjustedTexMax;
      if (volTex->CoordsTex)
      {
        volTex->CoordsTex->Activate();
        prog->SetUniformi("in_coordTexs", volTex->CoordsTex->GetTextureUnit());
        float fvalue3[3];
        vtkInternal::ToFloat(volTex->CoordsTexSizes, fvalue3, 3);
        prog->SetUniform3fv("in_coordTexSizes", 1, &fvalue3);
        prog->SetUniform3fv("in_coordsScale", 1, volTex->CoordsScale);
        prog->SetUniform3fv("in_coordsBias", 1, volTex->CoordsBias);
      }

      if (volTex->BlankingTex)
      {
        volTex->BlankingTex->Activate();
        prog->SetUniformi("in_blanking", volTex->BlankingTex->GetTextureUnit());
      }
    }

    // Volume matrices (dataset to world)
    dataToWorld->Transpose();

    // Get the effective position of the eye in world coordinates for this
    // volume (or the bbox).
    // This multiply may look backwards, but dataToWorld and modelViewMat are
    // both already transposed to send to OpenGL.
    vtkMatrix4x4::Multiply4x4(dataToWorld.GetPointer(), modelViewMat, dataToView.GetPointer());
    dataToView->Invert();
    eyePos[0] = dataToView->GetElement(3, 0);
    eyePos[1] = dataToView->GetElement(3, 1);
    eyePos[2] = dataToView->GetElement(3, 2);
    vtkInternal::CopyVector<float, 3>(eyePos, this->EyePosVec.data(), i * 3);
    std::cerr << std::setprecision(9) << "VTK_METAL_VOLUME_LOG DEBUG GL_EYE double=("
              << dataToView->GetElement(3, 0) << ", " << dataToView->GetElement(3, 1) << ", "
              << dataToView->GetElement(3, 2) << ") float=(" << eyePos[0] << ", " << eyePos[1]
              << ", " << eyePos[2] << ")" << std::endl;

    vtkInternal::CopyMatrixToVector<vtkMatrix4x4, 4, 4>(
      dataToWorld.GetPointer(), this->VolMatVec.data(), vecOffset);

    this->InverseVolumeMat->DeepCopy(dataToWorld.GetPointer());
    this->InverseVolumeMat->Invert();
    vtkInternal::CopyMatrixToVector<vtkMatrix4x4, 4, 4>(
      this->InverseVolumeMat.GetPointer(), this->InvMatVec.data(), vecOffset);

    // Texture matrices (texture to dataset)
    texToDataMat->Transpose();
    vtkInternal::CopyMatrixToVector<vtkMatrix4x4, 4, 4>(
      texToDataMat.GetPointer(), this->TexMatVec.data(), vecOffset);

    texToDataMat->Invert();
    vtkInternal::CopyMatrixToVector<vtkMatrix4x4, 4, 4>(
      texToDataMat.GetPointer(), this->InvTexMatVec.data(), vecOffset);

    // Cell to Point (texture adjustment)
    cellToPointMat->Transpose();
    vtkInternal::CopyMatrixToVector<vtkMatrix4x4, 4, 4>(
      cellToPointMat.GetPointer(), this->CellToPointVec.data(), vecOffset);
    vtkInternal::CopyVector<float, 3>(texMin, this->TexMinVec.data(), i * 3);
    vtkInternal::CopyVector<float, 3>(texMax, this->TexMaxVec.data(), i * 3);
  }

  // the matrix from data to world
  prog->SetUniformMatrix4x4v("in_volumeMatrix", numVolumes, this->VolMatVec.data());
  prog->SetUniformMatrix4x4v("in_inverseVolumeMatrix", numVolumes, this->InvMatVec.data());

  // the matrix from tcoords to data
  prog->SetUniformMatrix4x4v("in_textureDatasetMatrix", numVolumes, this->TexMatVec.data());
  prog->SetUniformMatrix4x4v(
    "in_inverseTextureDatasetMatrix", numVolumes, this->InvTexMatVec.data());

  // matrix from texture to view coordinates
  prog->SetUniformMatrix4x4v("in_textureToEye", numVolumes, this->TexEyeMatVec.data());

  // handle cell/point differences in tcoords
  prog->SetUniformMatrix4x4v("in_cellToPoint", numVolumes, this->CellToPointVec.data());

  {
    // TEMP DEBUG: dump the matrices that drive g_dirStep so the Metal backend
    // can reproduce GL's step arithmetic exactly.
    const float* it = this->InvTexMatVec.data();
    const float* ct = this->CellToPointVec.data();
    std::cerr << std::setprecision(9) << "VTK_METAL_VOLUME_LOG DEBUG GL_UNIFORMS nVol=" << numVolumes
              << " sampleDist=" << this->ActualSampleDistance << "\n"
              << "  invTexDataset=[" << it[0] << "," << it[1] << "," << it[2] << "," << it[3]
              << "," << it[4] << "," << it[5] << "," << it[6] << "," << it[7] << "," << it[8]
              << "," << it[9] << "," << it[10] << "," << it[11] << "," << it[12] << "," << it[13]
              << "," << it[14] << "," << it[15] << "]\n"
              << "  cellToPoint=[" << ct[0] << "," << ct[1] << "," << ct[2] << "," << ct[3] << ","
              << ct[4] << "," << ct[5] << "," << ct[6] << "," << ct[7] << "," << ct[8] << ","
              << ct[9] << "," << ct[10] << "," << ct[11] << "," << ct[12] << "," << ct[13] << ","
              << ct[14] << "," << ct[15] << "]\n"
              << "  eyePosObjs=(" << this->EyePosVec[0] << "," << this->EyePosVec[1] << ","
              << this->EyePosVec[2] << ")\n";
  }

  prog->SetUniform3fv(
    "in_texMin", numVolumes, reinterpret_cast<const float(*)[3]>(this->TexMinVec.data()));
  prog->SetUniform3fv(
    "in_texMax", numVolumes, reinterpret_cast<const float(*)[3]>(this->TexMaxVec.data()));
  prog->SetUniform3fv(
    "in_eyePosObjs", numVolumes, reinterpret_cast<const float(*)[3]>(this->EyePosVec.data()));
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::SetVolumeShaderParameters(
  vtkShaderProgram* prog, int independentComponents, int noOfComponents, vtkMatrix4x4* modelViewMat)
{
  this->BindTransformations(prog, modelViewMat);

  // Bind other properties (per-input)
  const int numInputs = static_cast<int>(this->Parent->AssembledInputs.size());
  this->ScaleVec.resize(numInputs * 4, 0);
  this->BiasVec.resize(numInputs * 4, 0);
  this->StepVec.resize(numInputs * 3, 0);
  this->SpacingVec.resize(numInputs * 3, 0);
  this->RangeVec.resize(numInputs * 8, 0);

  int index = 0;
  for (auto& input : this->Parent->AssembledInputs)
  {
    // Bind volume textures
    auto block = input.second.Texture->GetCurrentBlock();
    std::stringstream ss;
    ss << "in_volume[" << index << "]";
    block->TextureObject->Activate();
    prog->SetUniformi(ss.str().c_str(), block->TextureObject->GetTextureUnit());

    // LargeDataTypes have been already biased and scaled so in those cases 0s
    // and 1s are passed respectively.
    float tscale[4] = { 1.0, 1.0, 1.0, 1.0 };
    float tbias[4] = { 0.0, 0.0, 0.0, 0.0 };
    float(*scalePtr)[4] = &tscale;
    float(*biasPtr)[4] = &tbias;
    auto volTex = input.second.Texture.GetPointer();
    if (!volTex->HandleLargeDataTypes &&
      (noOfComponents == 1 || noOfComponents == 2 || independentComponents))
    {
      scalePtr = &volTex->Scale;
      biasPtr = &volTex->Bias;
    }
    std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_TEX scale=(" << (*scalePtr)[0] << ","
              << (*scalePtr)[1] << "," << (*scalePtr)[2] << "," << (*scalePtr)[3]
              << ") bias=(" << (*biasPtr)[0] << "," << (*biasPtr)[1] << "," << (*biasPtr)[2]
              << "," << (*biasPtr)[3] << ") scalarRange=(" << volTex->ScalarRange[0][0] << ","
              << volTex->ScalarRange[0][1] << ") handleLarge=" << volTex->HandleLargeDataTypes
              << std::endl;
    vtkInternal::CopyVector<float, 4>(*scalePtr, this->ScaleVec.data(), index * 4);
    vtkInternal::CopyVector<float, 4>(*biasPtr, this->BiasVec.data(), index * 4);
    vtkInternal::CopyVector<float, 3>(block->CellStep, this->StepVec.data(), index * 3);
    vtkInternal::CopyVector<float, 3>(volTex->CellSpacing, this->SpacingVec.data(), index * 3);

    // 8 elements stands for [min, max] per 4-components
    vtkInternal::CopyVector<float, 8>(
      reinterpret_cast<float*>(volTex->ScalarRange), this->RangeVec.data(), index * 8);

    input.second.ActivateTransferFunction(prog, this->Parent->BlendMode);
    index++;
  }
  prog->SetUniform4fv(
    "in_volume_scale", numInputs, reinterpret_cast<const float(*)[4]>(this->ScaleVec.data()));
  prog->SetUniform4fv(
    "in_volume_bias", numInputs, reinterpret_cast<const float(*)[4]>(this->BiasVec.data()));
  prog->SetUniform2fv(
    "in_scalarsRange", 4 * numInputs, reinterpret_cast<const float(*)[2]>(this->RangeVec.data()));
  prog->SetUniform3fv(
    "in_cellStep", numInputs, reinterpret_cast<const float(*)[3]>(this->StepVec.data()));
  prog->SetUniform3fv(
    "in_cellSpacing", numInputs, reinterpret_cast<const float(*)[3]>(this->SpacingVec.data()));

  if (this->Parent->GetVolumetricScatteringBlending() > 0.0)
  {
    // set anisotropy of the first volume
    prog->SetUniformf("in_anisotropy",
      this->Parent->AssembledInputs[0].Volume->GetProperty()->GetScatteringAnisotropy());
    prog->SetUniformf(
      "in_volumetricScatteringBlending", this->Parent->GetVolumetricScatteringBlending() / 2.0);
  }
}

////----------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::SetMapperShaderParameters(
  vtkShaderProgram* prog, vtkRenderer* ren, int independent, int numComp)
{
  if (!this->SharedDepthTextureObject)
  {
    this->DepthTextureObject->Activate();
  }
  prog->SetUniformi("in_depthSampler", this->DepthTextureObject->GetTextureUnit());

  if (this->Parent->GetUseJittering())
  {
    vtkOpenGLRenderWindow* win = static_cast<vtkOpenGLRenderWindow*>(ren->GetRenderWindow());
    prog->SetUniformi("in_noiseSampler", win->GetNoiseTextureUnit());
  }

  prog->SetUniformi("in_noOfComponents", numComp);
  prog->SetUniformf("in_sampleDistance", this->ActualSampleDistance);

  // Set the scale and bias for color correction
  prog->SetUniformf("in_scale", 1.0 / this->Parent->FinalColorWindow);
  prog->SetUniformf(
    "in_bias", (0.5 - (this->Parent->FinalColorLevel / this->Parent->FinalColorWindow)));
  // prog->SetUniformi("in_useTransfer2DGradient", this->Transfer2DUseGradient);
  if (!this->Transfer2DUseGradient && this->Transfer2DYAxisScalars)
  {
    vtkTextureObject* transfer2DYAxisTex =
      this->Transfer2DYAxisScalars->GetCurrentBlock()->TextureObject;
    transfer2DYAxisTex->Activate();
    prog->SetUniformi("in_transfer2DYAxis", transfer2DYAxisTex->GetTextureUnit());
    // LargeDataTypes have been already biased and scaled so in those cases 0s
    // and 1s are passed respectively.
    float tscale[4] = { 1.0, 1.0, 1.0, 1.0 };
    float tbias[4] = { 0.0, 0.0, 0.0, 0.0 };
    auto volTex = this->Transfer2DYAxisScalars;
    auto noOfComponents = volTex->GetLoadedScalars()->GetNumberOfComponents();
    if (!volTex->HandleLargeDataTypes &&
      (noOfComponents == 1 || noOfComponents == 2 || independent))
    {
      prog->SetUniform4f("in_transfer2DYAxis_scale", volTex->Scale);
      prog->SetUniform4f("in_transfer2DYAxis_bias", volTex->Bias);
    }
    else
    {
      prog->SetUniform4f("in_transfer2DYAxis_scale", tscale);
      prog->SetUniform4f("in_transfer2DYAxis_bias", tbias);
    }
  }
  else
  {
    prog->SetUniformi("in_transfer2DYAxis", 0);
  }
}

////----------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::SetCameraShaderParameters(
  vtkShaderProgram* prog, vtkRenderer* ren, vtkOpenGLCamera* cam)
{
  vtkMatrix4x4* glTransformMatrix;
  vtkMatrix4x4* modelViewMatrix;
  vtkMatrix3x3* normalMatrix;
  vtkMatrix4x4* projectionMatrix;
  cam->GetKeyMatrices(ren, modelViewMatrix, normalMatrix, projectionMatrix, glTransformMatrix);

  this->InverseProjectionMat->DeepCopy(projectionMatrix);
  this->InverseProjectionMat->Invert();
  prog->SetUniformMatrix("in_projectionMatrix", projectionMatrix);
  prog->SetUniformMatrix("in_inverseProjectionMatrix", this->InverseProjectionMat.GetPointer());

  this->InverseModelViewMat->DeepCopy(modelViewMatrix);
  this->InverseModelViewMat->Invert();
  prog->SetUniformMatrix("in_modelViewMatrix", modelViewMatrix);
  prog->SetUniformMatrix("in_inverseModelViewMatrix", this->InverseModelViewMat.GetPointer());

  // Composed inverse view-projection-to-dataset matrix for the analytic pixel
  // ray: inverseProjectionMatrix * inverseModelViewMatrix * inverseVolumeMatrix
  // (double-precision vtk product, uploaded once as float32). Both the OpenGL
  // computeRayDirection and the Metal backend unproject the fragment with these
  // exact bytes, so the ray direction no longer depends on the rasterizer's
  // barycentric interpolation. Note: vtkMatrix4x4 stores row-major but GLSL/Metal
  // read the uniform as column-major, so the row-major product is
  // inv(P) * inv(V) * inv(M) (projection inverse applied first).
  vtkMatrix4x4::Multiply4x4(this->InverseProjectionMat.GetPointer(),
    this->InverseModelViewMat.GetPointer(), this->InverseViewProjectionToDataMat.GetPointer());
  vtkMatrix4x4::Multiply4x4(this->InverseViewProjectionToDataMat.GetPointer(),
    this->InverseVolumeMat.GetPointer(), this->InversePVMMat.GetPointer());
  prog->SetUniformMatrix("in_inversePVM", this->InversePVMMat.GetPointer());

  // TEMP DEBUG: dump the three clip-chain matrices as float32 raw bytes, in the
  // same column-major layout SetUniformMatrix uploads (data[i] =
  // Element[i/4][i%4]), for byte-compare against Metal's MTL_CLIPMAT.
  if (getenv("VTK_GL_VERTEX_DUMP") != nullptr)
  {
    auto dumpMat = [](const char* tag, vtkMatrix4x4* m)
    {
      std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_CLIPMAT " << tag << "=";
      for (int i = 0; i < 16; ++i)
      {
        float f = static_cast<float>(m->Element[i / 4][i % 4]);
        std::cerr << std::hex << std::setfill('0') << std::setw(2)
                  << (int)reinterpret_cast<const unsigned char*>(&f)[0]
                  << (int)reinterpret_cast<const unsigned char*>(&f)[1]
                  << (int)reinterpret_cast<const unsigned char*>(&f)[2]
                  << (int)reinterpret_cast<const unsigned char*>(&f)[3];
      }
    };
    dumpMat("P", projectionMatrix);
    dumpMat("V", modelViewMatrix);
    std::cerr << " M=";
    for (int i = 0; i < 16; ++i)
    {
      float f = this->VolMatVec[0 * 16 + i];
      std::cerr << std::hex << std::setfill('0') << std::setw(2)
                << (int)reinterpret_cast<const unsigned char*>(&f)[0]
                << (int)reinterpret_cast<const unsigned char*>(&f)[1]
                << (int)reinterpret_cast<const unsigned char*>(&f)[2]
                << (int)reinterpret_cast<const unsigned char*>(&f)[3];
    }
    dumpMat(" I", this->InversePVMMat.GetPointer());
    std::cerr << std::dec << std::endl;
  }

  // TEMP DEBUG: dump float32 bits of the three clip-chain matrices.
  {
    unsigned char pbits[64], vbits[64], mbits[64], ibits[64];
    for (int i = 0; i < 16; ++i)
    {
      float pf = static_cast<float>(projectionMatrix->GetElement(i / 4, i % 4));
      float vf = static_cast<float>(modelViewMatrix->GetElement(i / 4, i % 4));
      float mf = static_cast<float>(this->InversePVMMat->GetElement(i / 4, i % 4));
      memcpy(pbits + i * 4, &pf, 4);
      memcpy(vbits + i * 4, &vf, 4);
      memcpy(ibits + i * 4, &mf, 4);
    }
    for (int i = 0; i < 16; ++i)
    {
      float mf = static_cast<float>(this->VolMatVec[i]);
      memcpy(mbits + i * 4, &mf, 4);
    }
    std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_CLIPMAT P=";
    for (int i = 0; i < 64; ++i)
      std::cerr << std::hex << std::setfill('0') << std::setw(2) << (int)pbits[i];
    std::cerr << " V=";
    for (int i = 0; i < 64; ++i)
      std::cerr << std::hex << std::setfill('0') << std::setw(2) << (int)vbits[i];
    std::cerr << " M=";
    for (int i = 0; i < 64; ++i)
      std::cerr << std::hex << std::setfill('0') << std::setw(2) << (int)mbits[i];
    std::cerr << " I=";
    for (int i = 0; i < 64; ++i)
      std::cerr << std::hex << std::setfill('0') << std::setw(2) << (int)ibits[i];
    std::cerr << std::dec << std::endl;
  }

  if (cam->GetParallelProjection())
  {
    float fvalue3[3];
    double dir[4];
    cam->GetDirectionOfProjection(dir);
    vtkInternal::ToFloat(dir[0], dir[1], dir[2], fvalue3);
    prog->SetUniform3fv("in_projectionDirection", 1, &fvalue3);
  }

  // TODO Take consideration of reduction factor
  float fvalue2[2];
  vtkInternal::ToFloat(this->WindowLowerLeft, fvalue2);
  prog->SetUniform2fv("in_windowLowerLeftCorner", 1, &fvalue2);

  vtkInternal::ToFloat(1.0 / this->WindowSize[0], 1.0 / this->WindowSize[1], fvalue2);
  prog->SetUniform2fv("in_inverseOriginalWindowSize", 1, &fvalue2);

  vtkInternal::ToFloat(1.0 / this->WindowSize[0], 1.0 / this->WindowSize[1], fvalue2);
  prog->SetUniform2fv("in_inverseWindowSize", 1, &fvalue2);
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::SetMaskShaderParameters(
  vtkShaderProgram* prog, vtkVolumeProperty* prop, int noOfComponents)
{
  if (this->CurrentMask)
  {
    auto maskTex = this->CurrentMask->GetCurrentBlock()->TextureObject;
    maskTex->Activate();
    prog->SetUniformi("in_mask", maskTex->GetTextureUnit());
  }

  if (noOfComponents == 1 && this->Parent->BlendMode != vtkGPUVolumeRayCastMapper::ADDITIVE_BLEND)
  {
    if (this->Parent->MaskInput != nullptr && this->Parent->MaskType == LabelMapMaskType)
    {
      this->LabelMapTransfer2D->Activate();
      prog->SetUniformi("in_labelMapTransfer", this->LabelMapTransfer2D->GetTextureUnit());
      if (prop->HasLabelGradientOpacity())
      {
        this->LabelMapGradientOpacity->Activate();
        prog->SetUniformi(
          "in_labelMapGradientOpacity", this->LabelMapGradientOpacity->GetTextureUnit());
      }
      prog->SetUniformf("in_maskBlendFactor", this->Parent->MaskBlendFactor);
      prog->SetUniformf("in_mask_scale", this->CurrentMask->Scale[0]);
      prog->SetUniformf("in_mask_bias", this->CurrentMask->Bias[0]);
      prog->SetUniformi("in_labelMapNumLabels", this->LabelMapTransfer2D->GetTextureHeight() - 1);
    }
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::SetRenderToImageParameters(
  vtkShaderProgram* prog)
{
  prog->SetUniformi("in_clampDepthToBackface", this->Parent->GetClampDepthToBackface());
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::SetAdvancedShaderParameters(vtkRenderer* ren,
  vtkShaderProgram* prog, vtkVolume* vol, vtkVolumeTexture::VolumeBlock* block, int numComp)
{
  // Cropping and clipping
  auto bounds = block->LoadedBoundsAA;
  this->SetCroppingRegions(prog, bounds);
  this->SetClippingPlanes(ren, prog, vol);

  // Picking
  if (this->CurrentSelectionPass < vtkHardwareSelector::POINT_ID_LOW24)
  {
    this->SetPickingId(ren);
  }

  auto blockExt = block->Extents;
  float fvalue3[3];
  vtkInternal::ToFloat(blockExt[0], blockExt[2], blockExt[4], fvalue3);
  prog->SetUniform3fv("in_textureExtentsMin", 1, &fvalue3);

  vtkInternal::ToFloat(blockExt[1], blockExt[3], blockExt[5], fvalue3);
  prog->SetUniform3fv("in_textureExtentsMax", 1, &fvalue3);

  // Component weights (independent components)
  auto volProperty = vol->GetProperty();
  float fvalue4[4];
  if (numComp > 1 && volProperty->GetIndependentComponents())
  {
    for (int i = 0; i < numComp; ++i)
    {
      fvalue4[i] = static_cast<float>(volProperty->GetComponentWeight(i));
    }
    prog->SetUniform4fv("in_componentWeight", 1, &fvalue4);
  }

  // Set the scalar range to be considered for average ip blend
  double avgRange[2];
  float fvalue2[2];
  this->Parent->GetAverageIPScalarRange(avgRange);
  if (avgRange[1] < avgRange[0])
  {
    double tmp = avgRange[1];
    avgRange[1] = avgRange[0];
    avgRange[0] = tmp;
  }
  vtkInternal::ToFloat(avgRange[0], avgRange[1], fvalue2);
  prog->SetUniform2fv("in_averageIPRange", 1, &fvalue2);

  // Set contour values for isosurface blend mode
  //--------------------------------------------------------------------------
  if (this->Parent->BlendMode == vtkVolumeMapper::ISOSURFACE_BLEND)
  {
    vtkIdType nbContours = volProperty->GetIsoSurfaceValues()->GetNumberOfContours();

    std::vector<float> values(nbContours);
    for (int i = 0; i < nbContours; i++)
    {
      values[i] = static_cast<float>(volProperty->GetIsoSurfaceValues()->GetValue(i));
    }

    // The shader expect (for efficiency purposes) the isovalues to be sorted.
    std::sort(values.begin(), values.end());

    prog->SetUniform1fv("in_isosurfacesValues", nbContours, values.data());
  }

  // Set function attributes for slice blend mode
  //--------------------------------------------------------------------------
  if (this->Parent->BlendMode == vtkVolumeMapper::SLICE_BLEND)
  {
    vtkPlane* plane = vtkPlane::SafeDownCast(volProperty->GetSliceFunction());

    if (plane)
    {
      double planeOrigin[3];
      double planeNormal[3];

      plane->GetOrigin(planeOrigin);
      plane->GetNormal(planeNormal);

      prog->SetUniform3f("in_slicePlaneOrigin", planeOrigin);
      prog->SetUniform3f("in_slicePlaneNormal", planeNormal);
    }
  }
}

void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::FinishRendering(const int numComp)
{
  for (auto& item : this->Parent->AssembledInputs)
  {
    auto& input = item.second;
    input.Texture->GetCurrentBlock()->TextureObject->Deactivate();
    input.DeactivateTransferFunction(this->Parent->BlendMode);
  }

  if (this->DepthTextureObject && !this->SharedDepthTextureObject)
  {
    this->DepthTextureObject->Deactivate();
  }

  if (this->CurrentMask)
  {
    this->CurrentMask->GetCurrentBlock()->TextureObject->Deactivate();
  }

  if (numComp == 1 && this->Parent->BlendMode != vtkGPUVolumeRayCastMapper::ADDITIVE_BLEND)
  {
    if (this->Parent->MaskInput != nullptr && this->Parent->MaskType == LabelMapMaskType)
    {
      this->LabelMapTransfer2D->Deactivate();
      this->LabelMapGradientOpacity->Deactivate();
    }
  }

  vtkOpenGLStaticCheckErrorMacro("Failed after FinishRendering!");
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::DoGPURender(vtkRenderer* ren, vtkOpenGLCamera* cam,
  vtkShaderProgram* prog, vtkOpenGLShaderProperty* shaderProperty)
{
  if (!prog)
  {
    return;
  }

  // Upload the value of user-defined uniforms in the program
  auto vu = static_cast<vtkOpenGLUniforms*>(shaderProperty->GetVertexCustomUniforms());
  vu->SetUniforms(prog);
  auto fu = static_cast<vtkOpenGLUniforms*>(shaderProperty->GetFragmentCustomUniforms());
  fu->SetUniforms(prog);
  auto gu = static_cast<vtkOpenGLUniforms*>(shaderProperty->GetGeometryCustomUniforms());
  gu->SetUniforms(prog);

  this->SetShaderParametersRenderPass();
  if (!this->Impl->MultiVolume)
  {
    this->Impl->RenderSingleInput(ren, cam, prog);
  }
  else
  {
    this->Impl->RenderMultipleInputs(ren, cam, prog);
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::RenderMultipleInputs(
  vtkRenderer* ren, vtkOpenGLCamera* cam, vtkShaderProgram* prog)
{
  auto& input = this->Parent->AssembledInputs[0];
  auto vol = input.Volume;
  auto volumeTex = input.Texture.GetPointer();
  const int independent = vol->GetProperty()->GetIndependentComponents();
  const int numComp = volumeTex->GetLoadedScalars()->GetNumberOfComponents();
  int const numSamplers = (independent ? numComp : 1);
  auto geometry = this->MultiVolume->GetDataGeometry();

  vtkMatrix4x4 *wcvc, *vcdc, *wcdc;
  vtkMatrix3x3* norm;
  cam->GetKeyMatrices(ren, wcvc, norm, vcdc, wcdc);

  this->SetMapperShaderParameters(prog, ren, independent, numComp);
  this->SetVolumeShaderParameters(prog, independent, numComp, wcvc);
  this->SetLightingShaderParameters(ren, prog, this->MultiVolume, numSamplers);
  this->SetCameraShaderParameters(prog, ren, cam);

  this->SetClippingPlanes(ren, prog, this->MultiVolume);

  this->RenderVolumeGeometry(ren, prog, this->MultiVolume, geometry);
  this->FinishRendering(numComp);
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::RenderSingleInput(
  vtkRenderer* ren, vtkOpenGLCamera* cam, vtkShaderProgram* prog)
{
  auto& input = this->Parent->AssembledInputs[0];
  auto vol = input.Volume;
  auto volumeTex = input.Texture.GetPointer();

  // Sort blocks in case the viewpoint changed, it immediately returns if there
  // is a single block.
  vol->GetModelToWorldMatrix(this->TempMatrix4x4);
  volumeTex->SortBlocksBackToFront(ren, this->TempMatrix4x4);
  vtkVolumeTexture::VolumeBlock* block = volumeTex->GetCurrentBlock();

  if (this->CurrentMask)
  {
    this->CurrentMask->SortBlocksBackToFront(ren, this->TempMatrix4x4);
  }

  const int independent = vol->GetProperty()->GetIndependentComponents();
  const int numComp = volumeTex->GetLoadedScalars()->GetNumberOfComponents();
  while (block != nullptr)
  {
    const int numSamplers = (independent ? numComp : 1);
    this->SetMapperShaderParameters(prog, ren, independent, numComp);

    vtkMatrix4x4 *wcvc, *vcdc, *wcdc;
    vtkMatrix3x3* norm;
    cam->GetKeyMatrices(ren, wcvc, norm, vcdc, wcdc);
    this->SetVolumeShaderParameters(prog, independent, numComp, wcvc);

    this->SetMaskShaderParameters(prog, vol->GetProperty(), numComp);
    this->SetLightingShaderParameters(ren, prog, vol, numSamplers);
    this->SetCameraShaderParameters(prog, ren, cam);
    this->SetAdvancedShaderParameters(ren, prog, vol, block, numComp);

    this->RenderVolumeGeometry(ren, prog, vol, block->VolumeGeometry);

    // TEMP DEBUG: dump interpolated ray geometry for Metal-parity analysis.
    this->DumpDebugRays(ren, prog, vol, block->VolumeGeometry);

    // TEMP DEBUG: full-framebuffer interpolated-attribute field dump.
    this->DumpDebugAttrField(ren, prog, vol, block->VolumeGeometry);

    // TEMP DEBUG: re-render the clean shader into an RGBA32F FBO and write the
    // true final fragment floats (VTK_GL_FLOAT_DUMP). Independent of the
    // DebugRayDump injection above.
    this->DumpCleanGLFloats(ren, prog, vol, block->VolumeGeometry);

    this->FinishRendering(numComp);
    block = volumeTex->GetNextBlock();
    if (this->CurrentMask)
    {
      this->CurrentMask->GetNextBlock();
    }
  }
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::DumpDebugRays(
  vtkRenderer* ren, vtkShaderProgram* prog, vtkVolume* vol, double geometry[24])
{
  // TEMP DEBUG: dump the interpolated ray geometry (g_rayOrigin / g_dirStep) for
  // a handful of pixels so the Metal backend can match the OpenGL reference
  // exactly. Each float is encoded as a 23-bit mantissa (bytes 0..2) plus a
  // sign bit and 5-bit exponent (byte 3) across a single render, since
  // floatBitsToUint is unavailable in the GLSL 150 shader. Channel layout:
  // origin.xyz = 0..2, step.xyz = 3..5.
  // Enabled via the VTK_GL_RAY_DUMP environment variable.
  if (!this->DebugRayDump)
  {
    return;
  }

  // The volume pass composites with GL_ONE/GL_ONE_MINUS_SRC_ALPHA, which would
  // corrupt the encoded bytes, so blend must be disabled for the debug renders.
  glDisable(GL_BLEND);
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

  // Initialize the debug capture selectors that gate the ray/sample/texel
  // shader blocks (the GLSL default for an unset int uniform is 0).
  prog->SetUniformi("in_debugSample", -1);
  float texelInit[3] = { -1.0f, -1.0f, -1.0f };
  prog->SetUniform3f("in_debugTexel", texelInit);

  // Pixel centers in gl_FragCoord space (y flipped relative to Metal's screenPos).
  // Pixels are paired with the Metal debugMarchGate pixels so the two backends'
  // rays can be compared directly. Metal screenPos (x, y) == GL (x, 511 - y).
  const float pixels[29][2] = { { 307.5f, 503.5f }, { 307.5f, 504.5f }, { 307.5f, 502.5f },
    { 480.5f, 111.5f }, { 496.5f, 23.5f }, { 93.5f, 310.5f }, { 242.5f, 181.5f },
    { 322.5f, 339.5f }, { 382.5f, 304.5f }, { 357.5f, 357.5f }, { 372.5f, 380.5f },
    { 256.5f, 256.5f }, { 104.5f, 266.5f }, { 188.5f, 204.5f }, { 422.5f, 419.5f },
    { 397.5f, 401.5f }, { 360.5f, 282.5f }, { 349.5f, 256.5f }, { 405.5f, 340.5f },
    { 9.5f, 493.5f }, { 293.5f, 213.5f }, { 338.5f, 79.5f }, { 350.5f, 506.5f },
    { 153.5f, 479.5f }, { 482.5f, 478.5f }, { 120.5f, 344.5f }, { 470.5f, 242.5f },
    { 439.5f, 230.5f }, { 469.5f, 48.5f } };

  // TEMP DEBUG: update-69 B residual pixels (Metal PNG coords (px, py), GL
  // window coords (px, 511 - py)).
  const float residPixels[19][2] = { { 140.5f, 505.5f }, { 170.5f, 469.5f }, { 181.5f, 415.5f },
    { 18.5f, 348.5f }, { 312.5f, 328.5f }, { 366.5f, 249.5f }, { 249.5f, 194.5f },
    { 305.5f, 176.5f }, { 268.5f, 147.5f }, { 0.5f, 136.5f }, { 197.5f, 110.5f },
    { 11.5f, 92.5f }, { 70.5f, 87.5f }, { 71.5f, 87.5f }, { 74.5f, 87.5f }, { 75.5f, 87.5f },
    { 229.5f, 86.5f }, { 174.5f, 66.5f }, { 435.5f, 31.5f } };
  const float(*rayPixels)[2] = pixels;
  size_t rayPixelCount = 29;
  if (getenv("VTK_GL_RESID_DUMP") != nullptr)
  {
    rayPixels = residPixels;
    rayPixelCount = 19;
  }

  for (size_t p = 0; p < rayPixelCount; ++p)
  {
    const float px = rayPixels[p][0];
    const float py = rayPixels[p][1];
    const GLint gx = static_cast<GLint>(px);
    const GLint gy = static_cast<GLint>(py);

    float origin[3] = { 0.0f, 0.0f, 0.0f };
    float step[3] = { 0.0f, 0.0f, 0.0f };
    float vpos[3] = { 0.0f, 0.0f, 0.0f };
    float tcoord[3] = { 0.0f, 0.0f, 0.0f };
    float clip[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float rayDir[3] = { 0.0f, 0.0f, 0.0f };
    float nearP[3] = { 0.0f, 0.0f, 0.0f };
    float farP[3] = { 0.0f, 0.0f, 0.0f };
    float nearPRaw[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float farPRaw[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float dir[3] = { 0.0f, 0.0f, 0.0f };
    float d2 = 0.0f;
    float inv = 0.0f;
    float rayDir2[3] = { 0.0f, 0.0f, 0.0f };
    float debugPixel[2] = { px, py };

    for (int f = 0; f < 43; ++f)
    {
      unsigned char bytes[4] = { 0, 0, 0, 0 };

      prog->SetUniform2f("in_debugPixel", debugPixel);
      prog->SetUniformi("in_debugChannel", f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      this->RenderVolumeGeometry(ren, prog, vol, geometry);
      glReadPixels(gx, gy, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, bytes);

      float v = 0.0f;
      if (bytes[0] != 0 || bytes[1] != 0 || bytes[2] != 0 || bytes[3] != 0)
      {
        double m23 = static_cast<double>(bytes[0]) + 256.0 * bytes[1] + 65536.0 * bytes[2];
        double mant = 1.0 + m23 / 8388608.0;
        int e = static_cast<int>(bytes[3] & 0x7F) - 64;
        double sign = (bytes[3] & 0x80) ? -1.0 : 1.0;
        v = static_cast<float>(sign * mant * std::pow(2.0, e));
      }
      if (f < 12)
      {
        float* dest = (f < 3) ? origin : (f < 6) ? step : (f < 9) ? vpos : tcoord;
        dest[f % 3] = v;
      }
      else if (f < 16)
      {
        clip[f - 12] = v;
      }
      else if (f < 25)
      {
        float* dest = (f < 19) ? rayDir : (f < 22) ? nearP : farP;
        dest[(f - 16) % 3] = v;
      }
      else if (f < 29)
      {
        nearPRaw[f - 25] = v;
      }
      else if (f < 33)
      {
        farPRaw[f - 29] = v;
      }
      else if (f < 36)
      {
        // shader field order for 33/34/35 is dir.z/x/y (field=(f-16)%3)
        dir[(f == 34) ? 0 : (f == 35) ? 1 : 2] = v;
      }
      else if (f == 36)
      {
        d2 = v;
      }
      else if (f == 37)
      {
        inv = v;
      }
      else if (f == 38)
      {
        nearPRaw[3] = v;
      }
      else if (f == 39)
      {
        farPRaw[3] = v;
      }
      else
      {
        // 40/41/42 = rayDir2.xyz via (f-16)%3 = 0/1/2
        rayDir2[f - 40] = v;
      }
    }

    // Channel 100: flat covering-triangle provoking-vertex index (ip_vid).
    unsigned char vidBytes[4] = { 0, 0, 0, 0 };
    prog->SetUniform2f("in_debugPixel", debugPixel);
    prog->SetUniformi("in_debugChannel", 100);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    this->RenderVolumeGeometry(ren, prog, vol, geometry);
    glReadPixels(gx, gy, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, vidBytes);
    int flatVid = -1;
    if (vidBytes[0] != 0 || vidBytes[1] != 0 || vidBytes[2] != 0 || vidBytes[3] != 0)
    {
      double m23 =
        static_cast<double>(vidBytes[0]) + 256.0 * vidBytes[1] + 65536.0 * vidBytes[2];
      double mant = 1.0 + m23 / 8388608.0;
      int e = static_cast<int>(vidBytes[3] & 0x7F) - 64;
      double sign = (vidBytes[3] & 0x80) ? -1.0 : 1.0;
      flatVid = static_cast<int>(sign * mant * std::pow(2.0, e));
    }

    // Channel 107: covering-triangle primitive index (gl_PrimitiveID).
    unsigned char triBytes[4] = { 0, 0, 0, 0 };
    prog->SetUniform2f("in_debugPixel", debugPixel);
    prog->SetUniformi("in_debugChannel", 107);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    this->RenderVolumeGeometry(ren, prog, vol, geometry);
    glReadPixels(gx, gy, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, triBytes);
    int glPrim = -1;
    if (triBytes[0] != 0 || triBytes[1] != 0 || triBytes[2] != 0 || triBytes[3] != 0)
    {
      double m23 =
        static_cast<double>(triBytes[0]) + 256.0 * triBytes[1] + 65536.0 * triBytes[2];
      double mant = 1.0 + m23 / 8388608.0;
      int e = static_cast<int>(triBytes[3] & 0x7F) - 64;
      double sign = (triBytes[3] & 0x80) ? -1.0 : 1.0;
      glPrim = static_cast<int>(sign * mant * std::pow(2.0, e));
    }

    // Skip pixels that were never covered by the volume (framebuffer held the
    // clear color instead of an encoded value).
    if (origin[0] == 0.0f && origin[1] == 0.0f && origin[2] == 0.0f)
    {
      continue;
    }

    double* cp = ren->GetActiveCamera()->GetPosition();
    std::cerr << std::setprecision(9) << "VTK_METAL_VOLUME_LOG DEBUG GL_RAY px=("
              << static_cast<int>(px) << ", "
              << static_cast<int>(py) << ") cam=(" << cp[0] << ", " << cp[1] << ", " << cp[2]
              << ") origin=(" << origin[0] << ", " << origin[1] << ", " << origin[2] << ") step=("
              << step[0] << ", " << step[1] << ", " << step[2] << ") vpos=(" << vpos[0] << ", "
              << vpos[1] << ", " << vpos[2] << ") tex=(" << tcoord[0] << ", " << tcoord[1] << ", "
              << tcoord[2] << ") clip=(" << clip[0] << ", " << clip[1] << ", " << clip[2] << ", "
              << clip[3] << ") rd=(" << rayDir[0] << ", " << rayDir[1] << ", " << rayDir[2]
              << ") nearP=(" << nearP[0] << ", " << nearP[1] << ", " << nearP[2] << ") farP=("
              << farP[0] << ", " << farP[1] << ", " << farP[2] << ") flatVid=" << flatVid
              << " primId=" << glPrim << "\n";
    std::cerr << std::hex;
    for (int i = 0; i < 3; ++i)
    {
      std::cerr << "  step" << i << "=" << std::setw(8) << std::setfill('0')
                << *reinterpret_cast<const uint32_t*>(&step[i]);
    }
    for (int i = 0; i < 3; ++i)
    {
      std::cerr << "  rd" << i << "=" << std::setw(8) << std::setfill('0')
                << *reinterpret_cast<const uint32_t*>(&rayDir[i]);
    }
    for (int i = 0; i < 3; ++i)
    {
      std::cerr << "  nearP" << i << "=" << std::setw(8) << std::setfill('0')
                << *reinterpret_cast<const uint32_t*>(&nearP[i]);
    }
    for (int i = 0; i < 3; ++i)
    {
      std::cerr << "  farP" << i << "=" << std::setw(8) << std::setfill('0')
                << *reinterpret_cast<const uint32_t*>(&farP[i]);
    }
    for (int i = 0; i < 4; ++i)
    {
      std::cerr << "  nearPRaw" << i << "=" << std::setw(8) << std::setfill('0')
                << *reinterpret_cast<const uint32_t*>(&nearPRaw[i]);
    }
    for (int i = 0; i < 4; ++i)
    {
      std::cerr << "  farPRaw" << i << "=" << std::setw(8) << std::setfill('0')
                << *reinterpret_cast<const uint32_t*>(&farPRaw[i]);
    }
    for (int i = 0; i < 3; ++i)
    {
      std::cerr << "  dir" << i << "=" << std::setw(8) << std::setfill('0')
                << *reinterpret_cast<const uint32_t*>(&dir[i]);
    }
    std::cerr << "  d2=" << std::setw(8) << std::setfill('0')
              << *reinterpret_cast<const uint32_t*>(&d2)
              << "  inv=" << std::setw(8) << std::setfill('0')
              << *reinterpret_cast<const uint32_t*>(&inv)
              << "  rd2=(" << rayDir2[0] << ", " << rayDir2[1] << ", " << rayDir2[2] << ")";
    std::cerr << std::dec << std::setfill(' ') << std::endl;
    std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_BOX " << std::setprecision(6);
    for (int i = 0; i < 8; ++i)
    {
      std::cerr << " c" << i << "=(" << geometry[i * 3] << ", " << geometry[i * 3 + 1] << ", "
                << geometry[i * 3 + 2] << ")";
    }
    std::cerr << std::endl;
  }

  // Restore the real composite image for the gated pixels.
  prog->SetUniformi("in_debugChannel", -1);
  prog->SetUniformi("in_debugSample", -1);

  // Per-sample raw dump for the (422.5, 419.5) pixel (Metal parity analysis):
  // re-render once per sample index, capturing the volume raw value sampled at
  // g_dataPos at that index, in the same units as the Metal backend (value/65535).
  // Channel 0 = raw, 1/2/3 = g_dataPos.x/y/z (same float encoding as the ray dump).
  if (getenv("VTK_GL_SAMPLE_DUMP") != nullptr)
  {
    float px = 422.5f;
    float py = 419.5f;
    GLint gx = 422;
    GLint gy = 419;
    const char* dbgPx = getenv("VTK_GL_SAMPLE_DUMP_PX");
    if (dbgPx != nullptr)
    {
      gx = atoi(dbgPx);
      const char* comma = strchr(dbgPx, ',');
      gy = (comma != nullptr) ? atoi(comma + 1) : gy;
      px = static_cast<float>(gx) + 0.5f;
      py = static_cast<float>(gy) + 0.5f;
    }
    float debugPixel[2] = { px, py };
    prog->SetUniform2f("in_debugPixel", debugPixel);
    const int maxSample = getenv("VTK_GL_SAMPLE_DUMP_MAX") != nullptr
      ? atoi(getenv("VTK_GL_SAMPLE_DUMP_MAX"))
      : 175;
    for (int s = 0; s < maxSample; ++s)
    {
      float chan[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
      prog->SetUniformi("in_debugSample", s);
      for (int c = 0; c < 8; ++c)
      {
        unsigned char bytes[4] = { 0, 0, 0, 0 };
        prog->SetUniformi("in_debugChannel", c);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        this->RenderVolumeGeometry(ren, prog, vol, geometry);
        glReadPixels(gx, gy, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, bytes);
        if (s < 2)
        {
          std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_SAMPLE_DBG s=" << s << " c=" << c
                    << " bytes=" << static_cast<int>(bytes[0]) << "," << static_cast<int>(bytes[1])
                    << "," << static_cast<int>(bytes[2]) << "," << static_cast<int>(bytes[3])
                    << std::endl;
        }

        float v = 0.0f;
        if (bytes[0] != 0 || bytes[1] != 0 || bytes[2] != 0 || bytes[3] != 0)
        {
          double m23 = static_cast<double>(bytes[0]) + 256.0 * bytes[1] + 65536.0 * bytes[2];
          double mant = 1.0 + m23 / 8388608.0;
          int e = static_cast<int>(bytes[3] & 0x7F) - 64;
          double sign = (bytes[3] & 0x80) ? -1.0 : 1.0;
          v = static_cast<float>(sign * mant * std::pow(2.0, e));
        }
        chan[c] = v;
      }
      if (chan[0] != 0.0f || chan[1] != 0.0f || chan[2] != 0.0f || chan[3] != 0.0f ||
        chan[4] != 0.0f || chan[5] != 0.0f || chan[6] != 0.0f || chan[7] != 0.0f)
      {
        std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_SAMPLE px=(" << gx << ", " << gy
                  << ") i=" << s << " raw=" << chan[0]
                  << " pos=(" << chan[1] << ", " << chan[2] << ", " << chan[3] << ")"
                  << " color=(" << chan[4] << ", " << chan[5] << ", " << chan[6] << ")"
                  << " op=" << chan[7] << std::endl;
      }
    }
  }

  // Restore the real composite image for the gated pixels.
  prog->SetUniformi("in_debugSample", -1);

  // Final-composite float dump: captures g_fragColor immediately after castRay
  // (channels 60-63 = pre-finalize rgba) and after finalizeRayCast (channels
  // 64-67 = the exact value the framebuffer stores, post scale/bias), byte-encoded
  // as true float32. Enabled via VTK_GL_FINAL_DUMP; pixel via VTK_GL_SAMPLE_DUMP_PX.
  if (getenv("VTK_GL_FINAL_DUMP") != nullptr)
  {
    float px = 422.5f;
    float py = 419.5f;
    GLint gx = 422;
    GLint gy = 419;
    const char* dbgPx = getenv("VTK_GL_SAMPLE_DUMP_PX");
    if (dbgPx != nullptr)
    {
      gx = atoi(dbgPx);
      const char* comma = strchr(dbgPx, ',');
      gy = (comma != nullptr) ? atoi(comma + 1) : gy;
      px = static_cast<float>(gx) + 0.5f;
      py = static_cast<float>(gy) + 0.5f;
    }
    float debugPixel[2] = { px, py };
    prog->SetUniform2f("in_debugPixel", debugPixel);
    prog->SetUniformi("in_debugSample", -1);
    for (int c = 60; c <= 67; ++c)
    {
      unsigned char bytes[4] = { 0, 0, 0, 0 };
      prog->SetUniformi("in_debugChannel", c);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      this->RenderVolumeGeometry(ren, prog, vol, geometry);
      glReadPixels(gx, gy, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, bytes);
      float v = 0.0f;
      if (bytes[0] != 0 || bytes[1] != 0 || bytes[2] != 0 || bytes[3] != 0)
      {
        double m23 = static_cast<double>(bytes[0]) + 256.0 * bytes[1] + 65536.0 * bytes[2];
        double mant = 1.0 + m23 / 8388608.0;
        int e = static_cast<int>(bytes[3] & 0x7F) - 64;
        double sign = (bytes[3] & 0x80) ? -1.0 : 1.0;
        v = static_cast<float>(sign * mant * std::pow(2.0, e));
      }
      std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_FINAL px=(" << gx << ", " << gy << ") c=" << c
                << " v=" << std::setprecision(9) << v << std::endl;
    }
    prog->SetUniformi("in_debugChannel", -1);
  }

  // Per-texel dump of the volume texture in the bone-plateau neighborhood
  // (Metal parity analysis): samples the texel centers through the current
  // (linear) filter, which returns exactly the texel value at (t+0.5)/dims.
  // Enabled via VTK_GL_TEX_DUMP; range via VTK_GL_TEX_DUMP_MIN/MAX.
  if (getenv("VTK_GL_TEX_DUMP") != nullptr)
  {
    int tmin[3] = { 0, 0, 0 };
    int tmax[3] = { 1, 1, 1 };
    vtkImageData* input = vtkImageData::SafeDownCast(this->Parent->GetInput());
    int tdims[3] = { 512, 512, 512 };
    if (input)
    {
      input->GetDimensions(tdims);
    }
    if (getenv("VTK_GL_TEX_DUMP_MIN") != nullptr &&
      sscanf(getenv("VTK_GL_TEX_DUMP_MIN"), "%d,%d,%d", &tmin[0], &tmin[1], &tmin[2]) != 3)
    {
      tmin[0] = tmin[1] = tmin[2] = 0;
    }
    if (getenv("VTK_GL_TEX_DUMP_MAX") != nullptr &&
      sscanf(getenv("VTK_GL_TEX_DUMP_MAX"), "%d,%d,%d", &tmax[0], &tmax[1], &tmax[2]) != 3)
    {
      tmax[0] = tmax[1] = tmax[2] = 1;
    }
    float tdimsF[3] = { static_cast<float>(tdims[0]), static_cast<float>(tdims[1]),
      static_cast<float>(tdims[2]) };
    prog->SetUniform3f("in_debugTexelDims", tdimsF);
    float debugPixel[2] = { 422.5f, 419.5f };
    prog->SetUniform2f("in_debugPixel", debugPixel);
    glDisable(GL_BLEND);
    for (int tz = tmin[2]; tz < tmax[2]; ++tz)
    {
      for (int ty = tmin[1]; ty < tmax[1]; ++ty)
      {
        for (int tx = tmin[0]; tx < tmax[0]; ++tx)
        {
          unsigned char bytes[4] = { 0, 0, 0, 0 };
          float texelF[3] = { static_cast<float>(tx), static_cast<float>(ty),
            static_cast<float>(tz) };
          prog->SetUniform3f("in_debugTexel", texelF);
          glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
          this->RenderVolumeGeometry(ren, prog, vol, geometry);
          glReadPixels(422, 419, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, bytes);

          float v = 0.0f;
          if (bytes[0] != 0 || bytes[1] != 0 || bytes[2] != 0 || bytes[3] != 0)
          {
            double m23 = static_cast<double>(bytes[0]) + 256.0 * bytes[1] + 65536.0 * bytes[2];
            double mant = 1.0 + m23 / 8388608.0;
            int e = static_cast<int>(bytes[3] & 0x7F) - 64;
            v = static_cast<float>(mant * std::pow(2.0, e));
          }
          std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_TEX (" << tx << ", " << ty << ", " << tz
                    << ") raw=" << v << std::endl;
        }
      }
    }
    float texelNone[3] = { -1.0f, -1.0f, -1.0f };
    prog->SetUniform3f("in_debugTexel", texelNone);
  }

  // TEMP DEBUG: per-vertex clip-space position dump (GL vs Metal byte-compare).
  // Renders the proxy indices as 1px points; the fragment encodes ip_vid and
  // ip_debugClip for every point (channels 100..104). The full framebuffer is
  // read back per channel so points at arbitrary window positions are captured.
  // Enabled via VTK_GL_VERTEX_DUMP.
  if (getenv("VTK_GL_VERTEX_DUMP") != nullptr)
  {
    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    const GLsizei w = vp[2];
    const GLsizei h = vp[3];
    std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_VERTCTX vp=(" << vp[0] << ", " << vp[1] << ", "
              << vp[2] << ", " << vp[3] << ") err=" << glGetError() << std::endl;
    const int nChannels = 11;
    std::vector<std::vector<float>> vals(nChannels, std::vector<float>(w * h, 0.0f));
    std::vector<unsigned char> buf(static_cast<size_t>(w) * h * 4);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_BLEND);
    for (int ch = 0; ch < nChannels; ++ch)
    {
      prog->SetUniformi("in_debugChannel", 100 + ch);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      this->RenderVolumeGeometryPoints(ren, prog, vol, geometry);
      GLenum e = glGetError();
      glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buf.data());
      size_t nonzero = 0;
      for (size_t i = 0; i < static_cast<size_t>(w) * h * 4; ++i)
      {
        nonzero += (buf[i] != 0);
      }
      std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_VERTCHAN ch=" << ch << " nonzero=" << nonzero
                << " err=" << e << std::endl;
      for (GLsizei y = 0; y < h; ++y)
      {
        for (GLsizei x = 0; x < w; ++x)
        {
          const unsigned char* b = &buf[(static_cast<size_t>(y) * w + x) * 4];
          if (b[0] != 0 || b[1] != 0 || b[2] != 0 || b[3] != 0)
          {
            double m23 = static_cast<double>(b[0]) + 256.0 * b[1] + 65536.0 * b[2];
            double mant = 1.0 + m23 / 8388608.0;
            int e2 = static_cast<int>(b[3] & 0x7F) - 64;
            double sign = (b[3] & 0x80) ? -1.0 : 1.0;
            vals[ch][static_cast<size_t>(y) * w + x] =
              static_cast<float>(sign * mant * std::pow(2.0, e2));
          }
        }
      }
    }
    for (GLsizei y = 0; y < h; ++y)
    {
      for (GLsizei x = 0; x < w; ++x)
      {
        const size_t idx = static_cast<size_t>(y) * w + x;
        if (vals[0][idx] != 0.0f)
        {
          std::cerr << std::setprecision(9) << "VTK_METAL_VOLUME_LOG DEBUG GL_VERT "
                    << static_cast<int>(vals[0][idx]) << " px=(" << x << ", " << y << ") clip=("
                    << vals[1][idx] << ", " << vals[2][idx] << ", " << vals[3][idx] << ", "
                    << vals[4][idx] << ") pos=(" << vals[5][idx] << ", " << vals[6][idx] << ", "
                    << vals[7][idx] << ") tex=(" << vals[8][idx] << ", " << vals[9][idx] << ", "
                    << vals[10][idx] << ")" << std::endl;
        }
      }
    }
    glEnable(GL_SCISSOR_TEST);
    glEnable(GL_DEPTH_TEST);
    prog->SetUniformi("in_debugChannel", -1);
  }

  // Restore the real composite image for the gated pixels.
  prog->SetUniformi("in_debugSample", -1);
  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  this->RenderVolumeGeometry(ren, prog, vol, geometry);

  // TEMP DEBUG: read back the real composite framebuffer at the Metal-parity
  // pixels (glReadPixels coords, bottom-left) to see what GL actually writes.
  if (getenv("VTK_GL_COMPOSITE_DUMP") != nullptr)
  {
    const int dbgPx[4][2] = { { 422, 92 }, { 422, 419 }, { 256, 256 }, { 372, 131 } };
    for (const auto& px : dbgPx)
    {
      unsigned char bytes[4] = { 0, 0, 0, 0 };
      glReadPixels(px[0], px[1], 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, bytes);
      std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_COMPOSITE px=(" << px[0] << ", " << px[1]
                << ") rgba=" << static_cast<int>(bytes[0]) << "," << static_cast<int>(bytes[1])
                << "," << static_cast<int>(bytes[2]) << "," << static_cast<int>(bytes[3])
                << std::endl;
    }
    if (getenv("VTK_GL_COMPOSITE_ROW") != nullptr)
    {
      int rowY = atoi(getenv("VTK_GL_COMPOSITE_ROW"));
      for (int gx2 = 400; gx2 <= 440; ++gx2)
      {
        unsigned char bytes[4] = { 0, 0, 0, 0 };
        glReadPixels(gx2, rowY, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, bytes);
        std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_COMPOSITE_ROW y=" << rowY << " x=" << gx2
                  << " rgba=" << static_cast<int>(bytes[0]) << "," << static_cast<int>(bytes[1])
                  << "," << static_cast<int>(bytes[2]) << std::endl;
      }
    }
  }

}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::DumpDebugAttrField(
  vtkRenderer* ren, vtkShaderProgram* prog, vtkVolume* vol, double geometry[24])
{
  // TEMP DEBUG: full-framebuffer interpolated-attribute field dump for the
  // Metal-parity displacement analysis. Renders the debug-injected shader into
  // an RGBA32F FBO with in_debugChannel = 200/201/202 (writes raw float4 for
  // EVERY pixel, no pixel gate, no byte encoding; see BuildShader) and appends
  // w*h*4 float32 RGBA values to the raw file per frame/pass. Layout of each
  // 3-pass group (each pass = w*h*4 floats):
  //   pass 0: (ip_textureCoords.xyz, float(flatVid))
  //   pass 1: ip_debugClip.xyzw
  //   pass 2: (ip_vertexPos.xyz, float(gl_PrimitiveID))
  // Row 0 = gl_FragCoord y 0 (bottom-left, matches glReadPixels).
  const char* env = getenv("VTK_GL_ATTR_DUMP");
  if (env == nullptr)
  {
    return;
  }
  const char* outPath = getenv("VTK_GL_ATTR_DUMP_OUT");
  const std::string out = (outPath != nullptr) ? outPath : "/tmp/bc/gl_attr_dump.raw";

  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);
  const GLsizei w = vp[2];
  const GLsizei h = vp[3];

  GLuint fbo = 0;
  GLuint tex = 0;
  GLuint rbo = 0;
  glGenFramebuffers(1, &fbo);
  glGenTextures(1, &tex);
  GLint prevTex2D = 0;
  glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex2D);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex2D));
  glGenRenderbuffers(1, &rbo);
  glBindRenderbuffer(GL_RENDERBUFFER, rbo);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);

  GLint prevFbo = 0;
  GLint prevDrawBuf = 0;
  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
  glGetIntegerv(GL_DRAW_BUFFER, &prevDrawBuf);
  GLboolean blendWasOn = glIsEnabled(GL_BLEND);
  GLboolean depthWasOn = glIsEnabled(GL_DEPTH_TEST);

  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);
  glDrawBuffer(GL_COLOR_ATTACHMENT0);
  glViewport(0, 0, w, h);
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClearDepth(1.0);
  glDisable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);

  std::vector<float> buf(static_cast<size_t>(w) * h * 4);
  FILE* f = fopen(out.c_str(), "ab");
  if (!f)
  {
    std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_ATTR_DUMP FAILED to open " << out << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(prevFbo));
    glDrawBuffer(static_cast<GLenum>(prevDrawBuf));
    glViewport(vp[0], vp[1], vp[2], vp[3]);
    if (!blendWasOn)
    {
      glDisable(GL_BLEND);
    }
    if (!depthWasOn)
    {
      glDisable(GL_DEPTH_TEST);
    }
    glDeleteRenderbuffers(1, &rbo);
    glDeleteTextures(1, &tex);
    glDeleteFramebuffers(1, &fbo);
    return;
  }

  prog->SetUniformi("in_debugChannel", -1);
  prog->SetUniformi("in_debugSample", -1);
  const float texelInit[3] = { -1.0f, -1.0f, -1.0f };
  prog->SetUniform3f("in_debugTexel", texelInit);
  for (int a = 0; a < 3; ++a)
  {
    prog->SetUniformi("in_debugChannel", 200 + a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    this->RenderVolumeGeometry(ren, prog, vol, geometry);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, buf.data());
    fwrite(buf.data(), sizeof(float), buf.size(), f);
  }
  prog->SetUniformi("in_debugChannel", -1);
  prog->SetUniformi("in_debugSample", -1);
  fclose(f);
  std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_ATTR_DUMP w=" << w << " h=" << h
            << " passes=3 plane0=texcoords+flatVid plane1=clip plane2=vertexPos+primId"
            << " appended to " << out << std::endl;

  // Restore state.
  glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(prevFbo));
  glDrawBuffer(static_cast<GLenum>(prevDrawBuf));
  glViewport(vp[0], vp[1], vp[2], vp[3]);
  if (!blendWasOn)
  {
    glDisable(GL_BLEND);
  }
  if (!depthWasOn)
  {
    glDisable(GL_DEPTH_TEST);
  }
  glDeleteRenderbuffers(1, &rbo);
  glDeleteTextures(1, &tex);
  glDeleteFramebuffers(1, &fbo);
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::DumpCleanGLFloats(
  vtkRenderer* ren, vtkShaderProgram* prog, vtkVolume* vol, double geometry[24])
{
  // Re-render the CLEAN volume shader into an RGBA32F FBO (dst = black, so the
  // blend output is exactly g_fragColor) and write the true final floats to a
  // raw RGBA float32 file (row 0 = gl_FragCoord y 0). Env gate is independent
  // of VTK_GL_RAY_DUMP so the shader is not debug-injected.
  const char* env = getenv("VTK_GL_FLOAT_DUMP");
  if (env == nullptr)
  {
    return;
  }
  // TEMP DEBUG: dump on EVERY frame (overwriting) so the final dump file aligns
  // with the final stored image frame (the camera animates between frames, and
  // knife-edge rays are hypersensitive to the camera pose).
  const char* outPath = getenv("VTK_GL_FLOAT_DUMP_OUT");
  const std::string out = (outPath != nullptr) ? outPath : "/tmp/bc/gl_float_dump.raw";

  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);
  const GLsizei w = vp[2];
  const GLsizei h = vp[3];

  GLuint fbo = 0;
  GLuint tex = 0;
  GLuint rbo = 0;
  glGenFramebuffers(1, &fbo);
  glGenTextures(1, &tex);

  GLint prevActiveUnit = 0;
  GLint prevTex2D = 0;
  glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveUnit);
  glActiveTexture(static_cast<GLenum>(prevActiveUnit));
  glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex2D);

  // Dump the texture state the volume shader samples from (all units, 2D and
  // 3D) so we can tell whether the dump below disturbs the volume/depth/TF
  // textures (env-gated by VTK_GL_FLOAT_DUMP_TEXTURES).
  if (getenv("VTK_GL_FLOAT_DUMP_TEXTURES") != nullptr)
  {
    for (GLint u = 0; u < 16; ++u)
    {
      GLint b2d = 0;
      GLint b3d = 0;
      glActiveTexture(static_cast<GLenum>(GL_TEXTURE0 + u));
      glGetIntegerv(GL_TEXTURE_BINDING_2D, &b2d);
      glGetIntegerv(GL_TEXTURE_BINDING_3D, &b3d);
      std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_FLOAT_DUMP textures unit=" << u << " tex2d=" << b2d
                << " tex3d=" << b3d << std::endl;
    }
    glActiveTexture(static_cast<GLenum>(prevActiveUnit));
  }

  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex2D));
  glGenRenderbuffers(1, &rbo);
  glBindRenderbuffer(GL_RENDERBUFFER, rbo);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);

  GLint prevFbo = 0;
  GLint prevDrawBuf = 0;
  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
  glGetIntegerv(GL_DRAW_BUFFER, &prevDrawBuf);
  GLboolean blendWasOn = glIsEnabled(GL_BLEND);
  GLboolean depthWasOn = glIsEnabled(GL_DEPTH_TEST);
  GLint prevBlendSrc = GL_ONE;
  GLint prevBlendDst = GL_ZERO;
  glGetIntegerv(GL_BLEND_SRC_ALPHA, &prevBlendSrc);
  glGetIntegerv(GL_BLEND_DST_ALPHA, &prevBlendDst);

  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);
  glDrawBuffer(GL_COLOR_ATTACHMENT0);
  GLenum fbStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_FLOAT_DUMP fbo_status=" << std::hex << fbStatus
            << " err=" << std::hex << glGetError() << std::dec << std::endl;

  GLint cm[4] = { 1, 1, 1, 1 };
  glGetIntegerv(GL_COLOR_WRITEMASK, cm);
  std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_FLOAT_DUMP colorwritemask_before=(" << cm[0] << ","
            << cm[1] << "," << cm[2] << "," << cm[3] << ")" << std::endl;

  std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_FLOAT_DUMP prev_active_unit=" << prevActiveUnit
            << " prev_tex2d=" << prevTex2D << " our_tex=" << tex << std::endl;

  glViewport(0, 0, w, h);
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClearDepth(1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDisable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_DEPTH_TEST);

  // Control test: does the FBO write/read RGB at all via a colored clear?
  {
    std::vector<float> ctl(static_cast<size_t>(w) * h * 4, 0.0f);
    glClearColor(0.25f, 0.5f, 0.75f, 0.25f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, tex);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, ctl.data());
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex2D));
    std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_FLOAT_DUMP ctl_clear_px0=(" << ctl[0] << ","
              << ctl[1] << "," << ctl[2] << ") a=" << ctl[3] << " err=" << std::hex
              << glGetError() << std::dec << std::endl;
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
  }

  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

  this->RenderVolumeGeometry(ren, prog, vol, geometry);

  std::vector<float> buf(static_cast<size_t>(w) * h * 4);
  glBindTexture(GL_TEXTURE_2D, tex);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, buf.data());
  glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex2D));
  std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_FLOAT_DUMP texImage err=" << std::hex << glGetError()
            << std::dec << " sample_px422_419=(" << buf[(size_t)419 * w * 4 + 422 * 4 + 0] << ", "
            << buf[(size_t)419 * w * 4 + 422 * 4 + 1] << ", " << buf[(size_t)419 * w * 4 + 422 * 4 + 2]
            << ") a=" << buf[(size_t)419 * w * 4 + 422 * 4 + 3] << std::endl;
  std::vector<float> rp(static_cast<size_t>(w) * h * 4);
  glReadBuffer(GL_COLOR_ATTACHMENT0);
  glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, rp.data());
  std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_FLOAT_DUMP readpixels px422_419=("
            << rp[(size_t)419 * w * 4 + 422 * 4 + 0] << ", "
            << rp[(size_t)419 * w * 4 + 422 * 4 + 1] << ", " << rp[(size_t)419 * w * 4 + 422 * 4 + 2]
            << ") a=" << rp[(size_t)419 * w * 4 + 422 * 4 + 3] << std::endl;

  FILE* f = fopen(out.c_str(), "wb");
  if (f)
  {
    fwrite(buf.data(), sizeof(float), buf.size(), f);
    fclose(f);
    std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_FLOAT_DUMP wrote " << w << "x" << h
              << " rgba32f to " << out << std::endl;
  }
  else
  {
    std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_FLOAT_DUMP FAILED to open " << out << std::endl;
  }

  // Restore state.
  glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(prevFbo));
  glDrawBuffer(static_cast<GLenum>(prevDrawBuf));
  glViewport(vp[0], vp[1], vp[2], vp[3]);
  if (!blendWasOn)
  {
    glDisable(GL_BLEND);
  }
  if (!depthWasOn)
  {
    glDisable(GL_DEPTH_TEST);
  }
  glBlendFunc(static_cast<GLenum>(prevBlendSrc), static_cast<GLenum>(prevBlendDst));
  glDeleteRenderbuffers(1, &rbo);
  glDeleteTextures(1, &tex);
  glDeleteFramebuffers(1, &fbo);
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::SetPartitions(
  unsigned short x, unsigned short y, unsigned short z)
{
  this->Impl->Partitions[0] = x;
  this->Impl->Partitions[1] = y;
  this->Impl->Partitions[2] = z;
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::GetReductionRatio(double* ratio)
{
  ratio[0] = ratio[1] = ratio[2] = 1.0; // default

  vtkImageData* input = vtkImageData::SafeDownCast(this->GetInput());
  if (!input)
  {
    return;
  }

  const int* dims = input->GetDimensions();
  const int dimCount = input->GetDataDimension();
  const size_t scalarSize = static_cast<size_t>(input->GetScalarSize());
  const size_t currentSize = static_cast<size_t>(dims[0]) * dims[1] * dims[2] * scalarSize;
  const size_t maxSize = static_cast<size_t>(
    this->GetMaxMemoryInBytes() * static_cast<double>(this->GetMaxMemoryFraction()));

  if (currentSize > maxSize)
  {
    const double totalRatio = static_cast<double>(maxSize) / currentSize;

    ratio[0] = 1.0 - ((1.0 - totalRatio) / dimCount);

    if (dims[1] != 1)
    {
      ratio[1] = ratio[0];
    }

    if (dims[2] != 1)
    {
      ratio[2] = ratio[0];
    }
  }
}

//------------------------------------------------------------------------------
vtkMTimeType vtkOpenGLGPUVolumeRayCastMapper::GetRenderPassStageMTime(vtkVolume* vol)
{
  vtkInformation* info = vol->GetPropertyKeys();
  vtkMTimeType renderPassMTime = 0;

  int curRenderPasses = 0;
  this->Impl->RenderPassAttached = false;
  if (info && info->Has(vtkOpenGLRenderPass::RenderPasses()))
  {
    curRenderPasses = info->Length(vtkOpenGLRenderPass::RenderPasses());
    this->Impl->RenderPassAttached = true;
  }

  int lastRenderPasses = 0;
  if (this->LastRenderPassInfo->Has(vtkOpenGLRenderPass::RenderPasses()))
  {
    lastRenderPasses = this->LastRenderPassInfo->Length(vtkOpenGLRenderPass::RenderPasses());
  }

  // Determine the last time a render pass changed stages:
  if (curRenderPasses != lastRenderPasses)
  {
    // Number of passes changed, definitely need to update.
    // Fake the time to force an update:
    renderPassMTime = VTK_MTIME_MAX;
  }
  else
  {
    // Compare the current to the previous render passes:
    for (int i = 0; i < curRenderPasses; ++i)
    {
      vtkObjectBase* curRP = info->Get(vtkOpenGLRenderPass::RenderPasses(), i);
      vtkObjectBase* lastRP = this->LastRenderPassInfo->Get(vtkOpenGLRenderPass::RenderPasses(), i);

      if (curRP != lastRP)
      {
        // Render passes have changed. Force update:
        renderPassMTime = VTK_MTIME_MAX;
        break;
      }
      else
      {
        // Render passes have not changed -- check MTime.
        vtkOpenGLRenderPass* rp = static_cast<vtkOpenGLRenderPass*>(curRP);
        renderPassMTime = std::max(renderPassMTime, rp->GetShaderStageMTime());
      }
    }
  }

  // Cache the current set of render passes for next time:
  if (info)
  {
    this->LastRenderPassInfo->CopyEntry(info, vtkOpenGLRenderPass::RenderPasses());
  }
  else
  {
    this->LastRenderPassInfo->Clear();
  }

  return renderPassMTime;
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::ReplaceShaderRenderPass(
  std::map<vtkShader::Type, vtkShader*>& shaders, vtkVolume* vol, bool prePass)
{
  std::string vertShader = shaders[vtkShader::Vertex]->GetSource();
  std::string geomShader = shaders[vtkShader::Geometry]->GetSource();
  std::string fragShader = shaders[vtkShader::Fragment]->GetSource();
  vtkInformation* info = vol->GetPropertyKeys();
  if (info && info->Has(vtkOpenGLRenderPass::RenderPasses()))
  {
    int numRenderPasses = info->Length(vtkOpenGLRenderPass::RenderPasses());
    for (int i = 0; i < numRenderPasses; ++i)
    {
      vtkObjectBase* rpBase = info->Get(vtkOpenGLRenderPass::RenderPasses(), i);
      vtkOpenGLRenderPass* rp = static_cast<vtkOpenGLRenderPass*>(rpBase);
      if (prePass)
      {
        if (!rp->PreReplaceShaderValues(vertShader, geomShader, fragShader, this, vol))
        {
          vtkErrorMacro(
            "vtkOpenGLRenderPass::PreReplaceShaderValues failed for " << rp->GetClassName());
        }
      }
      else
      {
        if (!rp->PostReplaceShaderValues(vertShader, geomShader, fragShader, this, vol))
        {
          vtkErrorMacro(
            "vtkOpenGLRenderPass::PostReplaceShaderValues failed for " << rp->GetClassName());
        }
      }
    }
  }
  shaders[vtkShader::Vertex]->SetSource(vertShader);
  shaders[vtkShader::Geometry]->SetSource(geomShader);
  shaders[vtkShader::Fragment]->SetSource(fragShader);
}

//------------------------------------------------------------------------------
void vtkOpenGLGPUVolumeRayCastMapper::SetShaderParametersRenderPass()
{
  auto vol = this->Impl->GetActiveVolume();
  vtkInformation* info = vol->GetPropertyKeys();
  if (info && info->Has(vtkOpenGLRenderPass::RenderPasses()))
  {
    int numRenderPasses = info->Length(vtkOpenGLRenderPass::RenderPasses());
    for (int i = 0; i < numRenderPasses; ++i)
    {
      vtkObjectBase* rpBase = info->Get(vtkOpenGLRenderPass::RenderPasses(), i);
      vtkOpenGLRenderPass* rp = static_cast<vtkOpenGLRenderPass*>(rpBase);
      if (!rp->SetShaderParameters(this->Impl->ShaderProgram, this, vol))
      {
        vtkErrorMacro(
          "RenderPass::SetShaderParameters failed for renderpass: " << rp->GetClassName());
      }
    }
  }
}
VTK_ABI_NAMESPACE_END
