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
@protocol MTLTexture;
@class CAMetalLayer;
@class CAMetalDrawable;

/** Block invoked when the GPU finishes rendering a frame.
 *  @param gpuTimeMs GPU frame time in milliseconds.
 *  Only valid on iOS 10.3+ / macOS 10.13+. */
typedef void (^VTKRenderCompletionBlock)(double gpuTimeMs);
#else
class id;
#endif

class vtkUnsignedIntArray;

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
   * Get the Metal command queue.
   */
  void* GetMetalQueue();

  /**
   * Get the current render command encoder.
   * Only valid during a render pass.
   */
  void* GetCurrentRenderCommandEncoder();

  /**
   * Set the current render command encoder.
   * Used by mappers that need to replace the encoder (e.g. offscreen rendering).
   */
  void SetCurrentRenderCommandEncoder(void* encoder);

  /**
   * Get the current command buffer.
   * Only valid between Start() and Frame().
   */
  void* GetCurrentCommandBuffer();

  /**
   * Get the current drawable texture.
   * Only valid during a render pass when a drawable is acquired.
   */
  void* GetCurrentDrawableTexture();

  /**
   * Get the depth texture used for the current render pass.
   */
  void* GetDepthTexture();

#ifdef __OBJC__
  /**
   * Set a block to be called when the GPU finishes rendering each frame.
   * The block is called on an internal Metal queue. Pass nil to clear.
   * Only meaningful when set from Objective-C context.
   */
  void SetRenderCompletionCallback(VTKRenderCompletionBlock block);
#endif

  /**
   * Get the effective sample count for multisampling.
   * Returns MultiSamples if > 1, otherwise 1.
   */
  int GetEffectiveSampleCount();

  /**
   * Read back the IDs texture (RGBA32Uint) into a vtkTypeUInt32Array.
   * The array is populated with 4 components per pixel: {CellId, PropId, CompositeId, ProcessId}.
   * Y-axis is flipped to match VTK's bottom-left origin.
   */
  void GetIdsData(int x1, int y1, int x2, int y2, vtkUnsignedIntArray* data);

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
   * Recreate the IDs texture for GPU-based picking.
   */
  void RecreateIdsTexture();

  /**
   * Create/destroy multisampled color and depth textures for MSAA rendering.
   */
  void CreateMultisampleAttachments();
  void DestroyMultisampleAttachments();

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
  void* RenderCompletionCallback = nullptr; // VTKRenderCompletionBlock, retained
  void* Encoder = nullptr;         // id<MTLRenderCommandEncoder>
  void* DepthTexture = nullptr;    // id<MTLTexture>
  void* IdsTexture = nullptr;      // id<MTLTexture> — RGBA32Uint for picking IDs
  void* MultisampleColorTexture = nullptr; // id<MTLTexture> — MSAA color (MTLTextureType2DMultisample)
  void* MultisampleDepthTexture = nullptr; // id<MTLTexture> — MSAA depth (MTLTextureType2DMultisample)
  void* ColorCopyPipeline = nullptr; // id<MTLRenderPipelineState>

  // 8B: Depth peeling state — set by vtkMetalDepthPeeler before each pass,
  // read by vtkMetalPolyDataMapper during RenderPiece().
  int DepthPeelingMode = 0;       // 0=normal, 1=init, 2=peel
  void* PeelFrontTexture = nullptr; // id<MTLTexture> — previous front accumulation
  void* PeelDepthTexture = nullptr; // id<MTLTexture> — previous depth (RG32Float)
  int PeelIndex = 0;              // current peel iteration

  bool Initialized = false;

private:
  friend class vtkMetalRenderer;
  friend class vtkMetalDepthPeeler;
  friend class vtkMetalPolyDataMapper;
  friend class vtkMetalPolyDataMapper2D;

  vtkMetalRenderWindow(const vtkMetalRenderWindow&) = delete;
  void operator=(const vtkMetalRenderWindow&) = delete;
};

VTK_ABI_NAMESPACE_END
#endif // vtkMetalRenderWindow_h
