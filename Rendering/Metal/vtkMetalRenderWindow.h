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

#include <cstdint>
#include <mutex>

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
class vtkUnsignedCharArray;

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
   * Get the Metal device's current allocated GPU memory (bytes).
   */
  uint64_t GetAllocatedSize();

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
   * Store the command buffer of the current frame. Takes ownership (retains)
   * so that WaitForCompletion() can safely block on it after the renderer's
   * autorelease scope has drained; the previous buffer is released.
   */
  void SetCurrentCommandBuffer(void* commandBuffer);

  /**
   * Get the current drawable texture.
   * Only valid during a render pass when a drawable is acquired.
   */
  void* GetCurrentDrawableTexture();

  /**
   * Get the depth texture used for the current render pass.
   */
  void* GetDepthTexture();

  /**
   * Toggle the per-frame color read-back blit (drawable -> shared texture).
   * Enabled by default so GetPixelData/GetRGBACharPixelData can read back the
   * rendered frame. Disable to remove the blit's per-frame overhead when no
   * image is being captured (e.g. while benchmarking).
   */
  vtkSetMacro(ColorReadbackEnabled, bool);
  vtkGetMacro(ColorReadbackEnabled, bool);

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
   * Returns MultiSamples if > 1, otherwise 1, clamped to the device's maximum
   * supported MSAA sample count (4 on Apple GPU family, 8 elsewhere). All
   * MSAA resource creation and pipeline-state sample counts must go through
   * this value so the requested count is never validated and rejected by
   * Metal.
   */
  int GetEffectiveSampleCount();

  /**
   * Get the shared Metal shader library, compiling it on first access.
   * Returns nil on compilation failure (check error log).
   * Thread-safe: uses a std::call_once guard.
   */
  void* GetSharedShaderLibrary();

  /**
   * Read back the IDs texture (RGBA32Uint) into a vtkTypeUInt32Array.
   * The array is populated with 4 components per pixel: {CellId, PropId, CompositeId, ProcessId}.
   * Y-axis is flipped to match VTK's bottom-left origin.
   */
  void GetIdsData(int x1, int y1, int x2, int y2, vtkUnsignedIntArray* data);

  /**
   * Read the color framebuffer back as RGBRGBRGB (3 components per pixel).
   * The caller owns the returned array. The front/right arguments are
   * accepted for API compatibility; Metal has a single (drawable) buffer.
   * Y-axis is flipped to match VTK's bottom-left origin.
   */
  unsigned char* GetPixelData(
    int x1, int y1, int x2, int y2, int front, int right = 0) override;

  /**
   * Read the color framebuffer back as RGBRGBRGB into the given array.
   */
  int GetPixelData(
    int x1, int y1, int x2, int y2, int front, vtkUnsignedCharArray* data, int right = 0) override;

  /**
   * Read the color framebuffer back as RGBARGBA (4 components per pixel).
   * The caller owns the returned array. The front/right arguments are
   * accepted for API compatibility; Metal has a single (drawable) buffer.
   * Y-axis is flipped to match VTK's bottom-left origin.
   */
  unsigned char* GetRGBACharPixelData(
    int x1, int y1, int x2, int y2, int front, int right = 0) override;

  /**
   * Read the color framebuffer back as RGBARGBA into the given array.
   */
  int GetRGBACharPixelData(int x1, int y1, int x2, int y2, int front,
    vtkUnsignedCharArray* data, int right = 0) override;

  /**
   * Read the color framebuffer back as normalized float RGBARGBA (4
   * components per pixel, each in [0,1]). The caller owns the returned
   * array. The front/right arguments are accepted for API compatibility;
   * Metal has a single (drawable) buffer. Y-axis is flipped to match VTK's
   * bottom-left origin.
   */
  float* GetRGBAPixelData(
    int x1, int y1, int x2, int y2, int front, int right = 0) override;

  /**
   * Read the color framebuffer back as normalized float RGBARGBA into the
   * given array.
   */
  int GetRGBAPixelData(
    int x1, int y1, int x2, int y2, int front, vtkFloatArray* data, int right = 0) override;

  /**
   * Free memory returned by the float GetRGBAPixelData overload.
   */
  void ReleaseRGBAPixelData(float* data) override;

  /**
   * Return the number of bits per channel of the color buffers. The Metal
   * color attachments are BGRA8Unorm (8 bits per channel).
   */
  int GetColorBufferSizes(int* rgba) override;

  /**
   * Read the depth framebuffer back as normalized floats in [0,1] (near to
   * far). The caller owns the returned array. Y-axis is flipped to match
   * VTK's bottom-left origin.
   */
  float* GetZbufferData(int x, int y, int x2, int y2) override;

  /**
   * Read the depth framebuffer back into a caller-provided float array.
   */
  int GetZbufferData(int x, int y, int x2, int y2, float* z) override;

  /**
   * Read the depth framebuffer back into the given array.
   */
  int GetZbufferData(int x, int y, int x2, int y2, vtkFloatArray* z) override;

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
   * Recreate the shared color-copy texture used for CPU read-back of the
   * rendered frame (GetPixelData / GetRGBACharPixelData).
   */
  void RecreateColorCopyTexture();

  /**
   * Recreate the shared depth-copy texture used for CPU read-back of the
   * rendered depth buffer (GetZbufferData).
   */
  void RecreateDepthCopyTexture();

#ifdef VTK_METAL_ENABLE_OFFSCREEN_TARGET
  /**
   * Recreate the private offscreen color texture used when OffScreenRendering
   * is enabled (benchmark timing). Same format/usage as the drawable texture
   * so every pass accepts it as the color attachment.
   */
  void RecreateOffscreenColorTexture();
#endif

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

  /**
   * Per-frame index of the renderer currently rendering (0-based). Reset to 0
   * at the start of Render(); incremented by vtkMetalRenderer::DeviceRender.
   * Used to share one drawable across renderers (first clears, last presents).
   */
  int GetFrameRendererIndex() const { return FrameRendererIndex; }
  void BumpFrameRendererIndex() { ++FrameRendererIndex; }

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
  void* ColorCopyTexture = nullptr; // id<MTLTexture> — BGRA8Unorm, MTLStorageModeShared, color read-back
  void* DepthCopyTexture = nullptr; // id<MTLTexture> — Depth32Float, MTLStorageModeShared, depth read-back
#ifdef VTK_METAL_ENABLE_OFFSCREEN_TARGET
  void* OffscreenColorTexture = nullptr; // id<MTLTexture> — BGRA8Unorm, MTLStorageModePrivate, offscreen target
#endif
  void* MultisampleColorTexture = nullptr; // id<MTLTexture> — MSAA color (MTLTextureType2DMultisample)
  void* MultisampleDepthTexture = nullptr; // id<MTLTexture> — MSAA depth (MTLTextureType2DMultisample)
  void* ColorCopyPipeline = nullptr; // id<MTLRenderPipelineState>

  // 8B: Depth peeling state — set by vtkMetalDepthPeeler before each pass,
  // read by vtkMetalPolyDataMapper during RenderPiece().
  int DepthPeelingMode = 0;       // 0=normal, 1=init, 2=peel
  void* PeelFrontTexture = nullptr; // id<MTLTexture> — previous front accumulation
  void* PeelDepthTexture = nullptr; // id<MTLTexture> — previous depth (RG32Float)
  int PeelIndex = 0;              // current peel iteration

  // 8C: Order-independent transparency (OIT) state — set by
  // vtkMetalOrderIndependentTranslucentPass before rendering the translucent
  // accumulate pass, read by vtkMetalPolyDataMapper during RenderPiece().
  bool OITActive = false;

  // Per-frame renderer index (see GetFrameRendererIndex).
  int FrameRendererIndex = 0;

  bool Initialized = false;

  // When true, the resolved color buffer is blitted into the shared color-copy
  // texture at the end of each frame so GetPixelData/GetRGBACharPixelData can
  // read it back. Defaults to true; benchmark drivers disable it.
  bool ColorReadbackEnabled = true;

  // Shared shader library — compiled once and reused across all pipelines.
  void* SharedShaderLibrary = nullptr;  // id<MTLLibrary>
  std::once_flag LibraryInitFlag;

private:
  friend class vtkMetalRenderer;
  friend class vtkMetalDepthPeeler;
  friend class vtkMetalOrderIndependentTranslucentPass;
  friend class vtkMetalPolyDataMapper;
  friend class vtkMetalPolyDataMapper2D;

  /**
   * Copy a region of the shared color-copy texture into a byte buffer.
   * ncomp is 3 (RGB) or 4 (RGBA). Output rows are bottom-up (VTK convention)
   * and BGRA is converted to RGB(A). Returns 1 on success, 0 on failure.
   */
  int ReadColorCopyData(int x, int y, int width, int height, int ncomp, void* dest);

  /**
   * Copy a region of the shared depth-copy texture into a float buffer.
   * Output rows are bottom-up (VTK convention). Returns 1 on success, 0 on
   * failure.
   */
  int ReadDepthCopyData(int x, int y, int width, int height, float* dest);

  vtkMetalRenderWindow(const vtkMetalRenderWindow&) = delete;
  void operator=(const vtkMetalRenderWindow&) = delete;
};

VTK_ABI_NAMESPACE_END
#endif // vtkMetalRenderWindow_h
