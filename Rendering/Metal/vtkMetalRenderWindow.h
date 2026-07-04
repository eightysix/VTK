// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkMetalRenderWindow
 * @brief   Metal rendering window
 *
 * vtkMetalRenderWindow is a concrete implementation of vtkRenderWindow
 * that uses Apple's Metal API for rendering. It manages the MTLDevice,
 * MTLCommandQueue, CAMetalLayer, and drawable lifecycle.
 */

#ifndef vtkMetalRenderWindow_h
#define vtkMetalRenderWindow_h

#include "vtkRenderWindow.h"
#include "vtkRenderingMetalModule.h" // for export macro
#include "vtkWrappingHints.h"        // for VTK_MARSHALAUTO

#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLCommandQueue;
@protocol MTLCommandBuffer;
@protocol MTLRenderPipelineState;
@class CAMetalLayer;
@class CAMetalDrawable;
#else
class id;
#endif

VTK_ABI_NAMESPACE_BEGIN

class vtkOverrideAttribute;

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkMetalRenderWindow
  : public vtkRenderWindow
{
public:
  static vtkMetalRenderWindow* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalRenderWindow, vtkRenderWindow);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  void Initialize() override;
  void Finalize() override;

  void Start() override;
  void Frame() override;
  void End() override;

  void Render() override;

  const char* GetRenderingBackend() override;

  void* GetGenericContext() override;
  void* GetGenericDisplayId() override;

  void SetSize(int width, int height) override;
  void SetPosition(int x, int y) override;

  void MakeCurrent() override {}
  void ReleaseCurrent() override {}
  void ReleaseGraphicsResources(vtkWindow*) override {}
  void WaitForCompletion() override;

  /**
   * Get the Metal layer used for rendering.
   */
  void* GetMetalLayer();

  /**
   * Get the underlying Metal device.
   */
  void* GetMetalDevice();

  /**
   * Get the current render command encoder.
   * Only valid during a render pass.
   */
  void* GetCurrentRenderCommandEncoder();

  /**
   * Get the current command buffer.
   * Only valid between Start() and Frame().
   */
  void* GetCurrentCommandBuffer();

protected:
  vtkMetalRenderWindow();
  ~vtkMetalRenderWindow() override;

  /**
   * Initialize the Metal device and command queue.
   */
  virtual bool InitializeMetal();

  /**
   * Create the CAMetalLayer.
   */
  virtual void CreateMetalLayer();

  /**
   * Recreate the depth texture when the window size changes.
   */
  void RecreateDepthTexture();

  /**
   * Acquire the next drawable from the CAMetalLayer.
   */
  bool AcquireDrawable();

  /**
   * Release the current drawable.
   */
  void ReleaseDrawable();

  // Metal objects (stored as void* to avoid Obj-C in header)
  void* MetalDevice = nullptr;    // id<MTLDevice>
  void* MetalQueue = nullptr;     // id<MTLCommandQueue>
  void* MetalLayer = nullptr;     // CAMetalLayer*
  void* CurrentDrawable = nullptr; // CAMetalDrawable*
  void* CommandBuffer = nullptr;   // id<MTLCommandBuffer>
  void* Encoder = nullptr;         // id<MTLRenderCommandEncoder>
  void* DepthTexture = nullptr;    // id<MTLTexture>
  void* ColorCopyPipeline = nullptr; // id<MTLRenderPipelineState>

  bool Initialized = false;

private:
  friend class vtkMetalRenderer;
  friend class vtkMetalPolyDataMapper;

  vtkMetalRenderWindow(const vtkMetalRenderWindow&) = delete;
  void operator=(const vtkMetalRenderWindow&) = delete;
};

VTK_ABI_NAMESPACE_END
#endif // vtkMetalRenderWindow_h
