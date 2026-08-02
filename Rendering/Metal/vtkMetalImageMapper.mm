// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalImageMapper.h"

#include "vtkMetalRenderWindow.h"
#include "vtkMetalMRC.h"
#include "vtkMetalShaders.h"
#include "vtkObjectFactory.h"
#include "vtkOverrideAttribute.h"

#include "vtkActor2D.h"
#include "vtkCoordinate.h"
#include "vtkDataArray.h"
#include "vtkImageData.h"
#include "vtkPointData.h"
#include "vtkViewport.h"
#include "vtkWindow.h"

#include "vtkNew.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstring>

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalImageMapper);

//------------------------------------------------------------------------------
struct vtkMetalImageMapper::vtkMetalImageMapperInternals
{
  // Image texture holding the window/level-converted RGBA pixels.
  id<MTLTexture> Texture = nil;
  int TextureWidth = 0;
  int TextureHeight = 0;

  // Pipeline + fixed-size geometry/state buffers for the textured quad.
  id<MTLRenderPipelineState> Pipeline = nil;
  id<MTLDepthStencilState> OverlayDepthState = nil;
  id<MTLBuffer> VertexBuffer = nil;
  id<MTLBuffer> IndexBuffer = nil;
  id<MTLBuffer> StateBuffer = nil;

  int CachedSampleCount = 0;
  MTLPixelFormat CachedDepthFormat = MTLPixelFormatInvalid;

  // Actor position (in viewport pixels, VTK bottom-left origin) computed by
  // RenderData. Position2 is used for the RenderToRectangle path.
  int Position[2] = { 0, 0 };
  int Position2[2] = { 0, 0 };

  void ReleasePipeline()
  {
    vtkMetalMRC::ReleaseAndNil(Pipeline);
  }

  void ReleaseBuffers()
  {
    vtkMetalMRC::ReleaseAndNil(Texture);
    vtkMetalMRC::ReleaseAndNil(VertexBuffer);
    vtkMetalMRC::ReleaseAndNil(IndexBuffer);
    vtkMetalMRC::ReleaseAndNil(StateBuffer);
    TextureWidth = 0;
    TextureHeight = 0;
    CachedSampleCount = 0;
    CachedDepthFormat = MTLPixelFormatInvalid;
  }

  void ReleaseState()
  {
    vtkMetalMRC::ReleaseAndNil(OverlayDepthState);
  }

  ~vtkMetalImageMapperInternals()
  {
    ReleaseBuffers();
    ReleaseState();
    ReleasePipeline();
  }
};

//------------------------------------------------------------------------------
vtkMetalImageMapper::vtkMetalImageMapper()
  : Internals(new vtkMetalImageMapperInternals())
{
}

//------------------------------------------------------------------------------
vtkMetalImageMapper::~vtkMetalImageMapper() = default;

//------------------------------------------------------------------------------
vtkOverrideAttribute* vtkMetalImageMapper::CreateOverrideAttributes()
{
  auto* renderingBackendAttribute =
    vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "Metal", nullptr);
  return renderingBackendAttribute;
}

//------------------------------------------------------------------------------
void vtkMetalImageMapper::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//------------------------------------------------------------------------------
// Release the graphics resources used by this texture.
void vtkMetalImageMapper::ReleaseGraphicsResources(vtkWindow* renWin)
{
  this->Internals->ReleaseBuffers();
  this->Internals->ReleaseState();
  this->Internals->ReleasePipeline();
  this->Superclass::ReleaseGraphicsResources(renWin);
}

//------------------------------------------------------------------------------
// I know #define can be evil, but this macro absolutely ensures
// that the code will be inlined.  The macro expects 'val' to
// be predefined to the same type as y

#define vtkClampToUnsignedChar(x, y)                                                               \
  do                                                                                               \
  {                                                                                                \
    val = (y);                                                                                     \
    if (val < 0)                                                                                   \
    {                                                                                              \
      val = 0;                                                                                     \
    }                                                                                              \
    if (val > 255)                                                                                 \
    {                                                                                              \
      val = 255;                                                                                   \
    }                                                                                              \
    (x) = static_cast<unsigned char>(val);                                                         \
  } while (false)

// the bit-shift must be done after the comparison to zero
// because bit-shift is undefined behaviour for negative numbers
#define vtkClampIntToUnsignedChar(x, y, shift)                                                     \
  do                                                                                               \
  {                                                                                                \
    val = (y);                                                                                     \
    if (val < 0)                                                                                   \
    {                                                                                              \
      val = 0;                                                                                     \
    }                                                                                              \
    val >>= (shift);                                                                               \
    if (val > 255)                                                                                 \
    {                                                                                              \
      val = 255;                                                                                   \
    }                                                                                              \
    (x) = static_cast<unsigned char>(val);                                                         \
  } while (false)

//---------------------------------------------------------------
// render the image by doing the following:
// 1) apply shift and scale to pixel values
// 2) clamp to [0,255] and convert to unsigned char RGBA
// 3) draw using DrawPixels

template <class T>
void vtkMetalImageMapperRenderDouble(vtkMetalImageMapper* self, vtkImageData* data, T* dataPtr,
  double shift, double scale, vtkViewport* viewport)
{
  int inMin0 = self->DisplayExtent[0];
  int inMax0 = self->DisplayExtent[1];
  int inMin1 = self->DisplayExtent[2];
  int inMax1 = self->DisplayExtent[3];

  int width = inMax0 - inMin0 + 1;
  int height = inMax1 - inMin1 + 1;

  vtkIdType tempIncs[3];
  data->GetIncrements(tempIncs);
  vtkIdType inInc1 = tempIncs[1];

  int bpp = data->GetNumberOfScalarComponents();

  T* inPtr = dataPtr;
  T* inPtr1 = inPtr;

  int i;
  int j = height;

  unsigned char* newPtr = new unsigned char[4 * width * height];
  unsigned char* ptr = newPtr;
  double val;
  unsigned char tmp;

  while (--j >= 0)
  {
    inPtr = inPtr1;
    i = width;
    switch (bpp)
    {
      case 1:
        while (--i >= 0)
        {
          vtkClampToUnsignedChar(tmp, ((*inPtr++ + shift) * scale));
          *ptr++ = tmp;
          *ptr++ = tmp;
          *ptr++ = tmp;
          *ptr++ = 255;
        }
        break;

      case 2:
        while (--i >= 0)
        {
          vtkClampToUnsignedChar(tmp, ((*inPtr++ + shift) * scale));
          *ptr++ = tmp;
          vtkClampToUnsignedChar(*ptr++, ((*inPtr++ + shift) * scale));
          *ptr++ = tmp;
          *ptr++ = 255;
        }
        break;

      case 3:
        while (--i >= 0)
        {
          vtkClampToUnsignedChar(*ptr++, ((*inPtr++ + shift) * scale));
          vtkClampToUnsignedChar(*ptr++, ((*inPtr++ + shift) * scale));
          vtkClampToUnsignedChar(*ptr++, ((*inPtr++ + shift) * scale));
          *ptr++ = 255;
        }
        break;

      default:
        while (--i >= 0)
        {
          vtkClampToUnsignedChar(*ptr++, ((*inPtr++ + shift) * scale));
          vtkClampToUnsignedChar(*ptr++, ((*inPtr++ + shift) * scale));
          vtkClampToUnsignedChar(*ptr++, ((*inPtr++ + shift) * scale));
          vtkClampToUnsignedChar(*ptr++, ((*inPtr++ + shift) * scale));
          inPtr += bpp - 4;
        }
        break;
    }
    inPtr1 += inInc1;
  }

  self->DrawPixels(viewport, width, height, 4, static_cast<void*>(newPtr));

  delete[] newPtr;
}

//---------------------------------------------------------------
// Same as above, but uses fixed-point math for shift and scale.
// The number of bits used for the fraction is determined from the
// scale.  Enough bits are always left over for the integer that
// overflow cannot occur.

template <class T>
void vtkMetalImageMapperRenderShort(vtkMetalImageMapper* self, vtkImageData* data, T* dataPtr,
  double shift, double scale, vtkViewport* viewport)
{
  int inMin0 = self->DisplayExtent[0];
  int inMax0 = self->DisplayExtent[1];
  int inMin1 = self->DisplayExtent[2];
  int inMax1 = self->DisplayExtent[3];

  int width = inMax0 - inMin0 + 1;
  int height = inMax1 - inMin1 + 1;

  vtkIdType tempIncs[3];
  data->GetIncrements(tempIncs);
  vtkIdType inInc1 = tempIncs[1];

  int bpp = data->GetNumberOfScalarComponents();

  // Find the number of bits to use for the fraction:
  // continue increasing the bits until there is an overflow
  // in the worst case, then decrease by 1.
  // The "*2.0" and "*1.0" ensure that the comparison is done
  // with double-precision math.
  int bitShift = 0;
  double absScale = std::abs(scale);

  while ((static_cast<long>(1 << bitShift) * absScale) * 2.0 * USHRT_MAX < INT_MAX * 1.0)
  {
    bitShift++;
  }
  bitShift--;
  bitShift = std::max(bitShift, 0);

  long sscale = static_cast<long>(scale * (1 << bitShift));
  long sshift = static_cast<long>(sscale * shift);
  long val;
  unsigned char tmp;

  T* inPtr = dataPtr;
  T* inPtr1 = inPtr;

  int i;
  int j = height;

  unsigned char* newPtr = new unsigned char[4 * width * height];
  unsigned char* ptr = newPtr;

  while (--j >= 0)
  {
    inPtr = inPtr1;
    i = width;

    switch (bpp)
    {
      case 1:
        while (--i >= 0)
        {
          vtkClampIntToUnsignedChar(tmp, (*inPtr++ * sscale + sshift), bitShift);
          *ptr++ = tmp;
          *ptr++ = tmp;
          *ptr++ = tmp;
          *ptr++ = 255;
        }
        break;

      case 2:
        while (--i >= 0)
        {
          vtkClampIntToUnsignedChar(tmp, (*inPtr++ * sscale + sshift), bitShift);
          *ptr++ = tmp;
          vtkClampIntToUnsignedChar(*ptr++, (*inPtr++ * sscale + sshift), bitShift);
          *ptr++ = tmp;
          *ptr++ = 255;
        }
        break;

      case 3:
        while (--i >= 0)
        {
          vtkClampIntToUnsignedChar(*ptr++, (*inPtr++ * sscale + sshift), bitShift);
          vtkClampIntToUnsignedChar(*ptr++, (*inPtr++ * sscale + sshift), bitShift);
          vtkClampIntToUnsignedChar(*ptr++, (*inPtr++ * sscale + sshift), bitShift);
          *ptr++ = 255;
        }
        break;

      default:
        while (--i >= 0)
        {
          vtkClampIntToUnsignedChar(*ptr++, (*inPtr++ * sscale + sshift), bitShift);
          vtkClampIntToUnsignedChar(*ptr++, (*inPtr++ * sscale + sshift), bitShift);
          vtkClampIntToUnsignedChar(*ptr++, (*inPtr++ * sscale + sshift), bitShift);
          vtkClampIntToUnsignedChar(*ptr++, (*inPtr++ * sscale + sshift), bitShift);
          inPtr += bpp - 4;
        }
        break;
    }
    inPtr1 += inInc1;
  }

  self->DrawPixels(viewport, width, height, 4, static_cast<void*>(newPtr));

  delete[] newPtr;
}

//---------------------------------------------------------------
// render unsigned char data without any shift/scale

template <class T>
void vtkMetalImageMapperRenderChar(
  vtkMetalImageMapper* self, vtkImageData* data, T* dataPtr, vtkViewport* viewport)
{
  int inMin0 = self->DisplayExtent[0];
  int inMax0 = self->DisplayExtent[1];
  int inMin1 = self->DisplayExtent[2];
  int inMax1 = self->DisplayExtent[3];

  int width = inMax0 - inMin0 + 1;
  int height = inMax1 - inMin1 + 1;

  vtkIdType tempIncs[3];
  data->GetIncrements(tempIncs);
  vtkIdType inInc1 = tempIncs[1];

  int bpp = data->GetPointData()->GetScalars()->GetNumberOfComponents();

  T* inPtr = dataPtr;
  T* inPtr1 = inPtr;
  unsigned char tmp;

  int i;
  int j = height;

  unsigned char* newPtr = new unsigned char[4 * width * height];
  unsigned char* ptr = newPtr;

  while (--j >= 0)
  {
    inPtr = inPtr1;
    i = width;

    switch (bpp)
    {
      case 1:
        while (--i >= 0)
        {
          *ptr++ = tmp = *inPtr++;
          *ptr++ = tmp;
          *ptr++ = tmp;
          *ptr++ = 255;
        }
        break;

      case 2:
        while (--i >= 0)
        {
          *ptr++ = tmp = *inPtr++;
          *ptr++ = tmp;
          *ptr++ = tmp;
          *ptr++ = *inPtr++;
        }
        break;

      case 3:
        while (--i >= 0)
        {
          *ptr++ = *inPtr++;
          *ptr++ = *inPtr++;
          *ptr++ = *inPtr++;
          *ptr++ = 255;
        }
        break;

      default:
        while (--i >= 0)
        {
          *ptr++ = *inPtr++;
          *ptr++ = *inPtr++;
          *ptr++ = *inPtr++;
          *ptr++ = *inPtr++;
          inPtr += bpp - 4;
        }
        break;
    }
    inPtr1 += inInc1;
  }

  self->DrawPixels(viewport, width, height, 4, static_cast<void*>(newPtr));

  delete[] newPtr;
}

//------------------------------------------------------------------------------
// Define overloads to help the template macro below dispatch to a
// suitable implementation for each type.  The last argument is of
// type "long" for the template and of type "int" for the
// non-templates.  The template macro's call to this function always
// passes a literal "1" as the last argument, which requires a
// conversion to produce a long.  This helps broken compilers select
// the non-template even when the template is otherwise an equal
// match.
template <class T>
void vtkMetalImageMapperRender(vtkMetalImageMapper* self, vtkImageData* data, T* dataPtr,
  double shift, double scale, vtkViewport* viewport)
{
  vtkMetalImageMapperRenderDouble(self, data, dataPtr, shift, scale, viewport);
}

static void vtkMetalImageMapperRender(vtkMetalImageMapper* self, vtkImageData* data, char* dataPtr,
  double shift, double scale, vtkViewport* viewport)
{
  if (shift == 0.0 && scale == 1.0)
  {
    vtkMetalImageMapperRenderChar(self, data, dataPtr, viewport);
  }
  else
  {
    vtkMetalImageMapperRenderShort(self, data, dataPtr, shift, scale, viewport);
  }
}

static void vtkMetalImageMapperRender(vtkMetalImageMapper* self, vtkImageData* data,
  unsigned char* dataPtr, double shift, double scale, vtkViewport* viewport)
{
  if (shift == 0.0 && scale == 1.0)
  {
    vtkMetalImageMapperRenderChar(self, data, dataPtr, viewport);
  }
  else
  {
    vtkMetalImageMapperRenderShort(self, data, dataPtr, shift, scale, viewport);
  }
}

static void vtkMetalImageMapperRender(vtkMetalImageMapper* self, vtkImageData* data,
  signed char* dataPtr, double shift, double scale, vtkViewport* viewport)
{
  if (shift == 0.0 && scale == 1.0)
  {
    vtkMetalImageMapperRenderChar(self, data, dataPtr, viewport);
  }
  else
  {
    vtkMetalImageMapperRenderShort(self, data, dataPtr, shift, scale, viewport);
  }
}

static void vtkMetalImageMapperRender(vtkMetalImageMapper* self, vtkImageData* data,
  short* dataPtr, double shift, double scale, vtkViewport* viewport)
{
  vtkMetalImageMapperRenderShort(self, data, dataPtr, shift, scale, viewport);
}

static void vtkMetalImageMapperRender(vtkMetalImageMapper* self, vtkImageData* data,
  unsigned short* dataPtr, double shift, double scale, vtkViewport* viewport)
{
  vtkMetalImageMapperRenderShort(self, data, dataPtr, shift, scale, viewport);
}

//------------------------------------------------------------------------------
// Expects data to be X, Y, components

void vtkMetalImageMapper::RenderData(vtkViewport* viewport, vtkImageData* data, vtkActor2D* actor)
{
  void* ptr0;
  double shift, scale;

  vtkWindow* window = viewport->GetVTKWindow();
  if (!window)
  {
    vtkErrorMacro(<< "vtkMetalImageMapper::RenderData - no window set for viewport");
    return;
  }

  if (!data->GetPointData()->GetScalars())
  {
    return;
  }

  shift = this->GetColorShift();
  scale = this->GetColorScale();

  ptr0 =
    data->GetScalarPointer(this->DisplayExtent[0], this->DisplayExtent[2], this->DisplayExtent[4]);

  // Get the position of the image actor
  int* actorPos = actor->GetActualPositionCoordinate()->GetComputedViewportValue(viewport);
  this->Internals->Position[0] = actorPos[0] + this->PositionAdjustment[0];
  this->Internals->Position[1] = actorPos[1] + this->PositionAdjustment[1];
  int* actorPos2 = actor->GetActualPosition2Coordinate()->GetComputedViewportValue(viewport);
  this->Internals->Position2[0] = actorPos2[0];
  this->Internals->Position2[1] = actorPos2[1];

  switch (data->GetPointData()->GetScalars()->GetDataType())
  {
    vtkTemplateMacro(
      vtkMetalImageMapperRender(this, data, static_cast<VTK_TT*>(ptr0), shift, scale, viewport));
    default:
      vtkErrorMacro(<< "Unsupported image type: " << data->GetScalarType());
  }
}

//------------------------------------------------------------------------------
void vtkMetalImageMapper::DrawPixels(
  vtkViewport* viewport, int width, int height, int numComponents, void* data)
{
  if (width <= 0 || height <= 0 || numComponents <= 0)
  {
    return;
  }

  vtkMetalRenderWindow* renWin =
    vtkMetalRenderWindow::SafeDownCast(viewport->GetVTKWindow());
  if (!renWin || !renWin->GetMetalDevice())
  {
    return;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (id<MTLDevice>)renWin->GetMetalDevice();
    id<MTLRenderCommandEncoder> encoder =
      (id<MTLRenderCommandEncoder>)renWin->GetCurrentRenderCommandEncoder();
    if (!encoder)
    {
      return;
    }

    float xscale = 1.0f;
    float yscale = 1.0f;
    if (this->GetRenderToRectangle())
    {
      int rectwidth = (this->Internals->Position2[0] - this->Internals->Position[0]) + 1;
      int rectheight = (this->Internals->Position2[1] - this->Internals->Position[1]) + 1;
      xscale = static_cast<float>(rectwidth) / width;
      yscale = static_cast<float>(rectheight) / height;
    }

    const float ox = static_cast<float>(this->Internals->Position[0]);
    const float oy = static_cast<float>(this->Internals->Position[1]);
    const float qw = width * xscale;
    const float qh = height * yscale;

    // Interleaved quad: position (viewport pixels) + texture coordinates.
    struct ImageVertex
    {
      float position[2];
      float texCoord[2];
    };
    ImageVertex verts[4] = {
      { { ox, oy }, { 0.0f, 0.0f } },
      { { ox + qw, oy }, { 1.0f, 0.0f } },
      { { ox + qw, oy + qh }, { 1.0f, 1.0f } },
      { { ox, oy + qh }, { 0.0f, 1.0f } },
    };
    const uint32_t indices[6] = { 0, 1, 2, 0, 2, 3 };

    // Compute the wcvc matrix (identical construction to
    // vtkMetalPolyDataMapper2D): maps viewport pixels (VTK bottom-left origin)
    // to Metal NDC. Metal's framebuffer y grows downward, so the Y row of the
    // orthographic matrix is negated.
    int* size = viewport->GetSize();
    double* vp = viewport->GetViewport();
    if (size[0] <= 0 || size[1] <= 0)
    {
      return;
    }
    float vpX = static_cast<float>(vp[0] * size[0]);
    float vpY = static_cast<float>(vp[1] * size[1]);
    float vpW = static_cast<float>((vp[2] - vp[0]) * size[0]);
    float vpH = static_cast<float>((vp[3] - vp[1]) * size[1]);
    if (vpW <= 0.0f || vpH <= 0.0f)
    {
      return;
    }

    // Standard orthographic matrix mapping the viewport pixel rect (VTK
    // bottom-left origin) to NDC. Metal's NDC y matches OpenGL's (+1 top, -1
    // bottom), so no Y negation is applied here: the quad's bottom vertex
    // (texCoord v=0, which samples the texture's first row = the image's
    // bottom row after upload) lands at NDC -1, the bottom of the framebuffer.
    float wcvc[16] = { 0 };
    wcvc[0] = 2.0f / vpW;
    wcvc[5] = 2.0f / vpH;
    wcvc[10] = 1.0f;
    wcvc[12] = -(2.0f * vpX + vpW) / vpW;
    wcvc[13] = -(2.0f * vpY + vpH) / vpH;
    wcvc[15] = 1.0f;

    // State buffer (only the wcvc matrix is consumed by the vertex shader).
    if (!this->Internals->StateBuffer)
    {
      this->Internals->StateBuffer = [device newBufferWithLength:sizeof(float) * 16
                                                        options:MTLResourceStorageModeShared];
    }
    memcpy([this->Internals->StateBuffer contents], wcvc, sizeof(float) * 16);

    // Vertex + index buffers.
    if (!this->Internals->VertexBuffer)
    {
      this->Internals->VertexBuffer =
        [device newBufferWithLength:sizeof(ImageVertex) * 4
                            options:MTLResourceStorageModeShared];
    }
    if (!this->Internals->IndexBuffer)
    {
      this->Internals->IndexBuffer =
        [device newBufferWithBytes:indices
                           length:sizeof(indices)
                          options:MTLResourceStorageModeShared];
    }
    memcpy([this->Internals->VertexBuffer contents], verts, sizeof(verts));

    // Image texture: recreate when the dimensions change, then upload.
    if (!this->Internals->Texture || this->Internals->TextureWidth != width ||
      this->Internals->TextureHeight != height)
    {
      MTLTextureDescriptor* texDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
      texDesc.usage = MTLTextureUsageShaderRead;
      id<MTLTexture> tex = [device newTextureWithDescriptor:texDesc];
      vtkMetalMRC::AssignConsumed(this->Internals->Texture, tex);
      this->Internals->TextureWidth = width;
      this->Internals->TextureHeight = height;
    }
    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
    [this->Internals->Texture replaceRegion:region
                                mipmapLevel:0
                                  withBytes:data
                                bytesPerRow:width * numComponents];

    // Pipeline: recreate on sample-count or depth-format change.
    int sampleCount = renWin->GetEffectiveSampleCount();
    MTLPixelFormat depthFormat = MTLPixelFormatInvalid;
    id<MTLTexture> depthTex = (id<MTLTexture>)renWin->GetDepthTexture();
    if (depthTex)
    {
      depthFormat = [depthTex pixelFormat];
    }
    if (!this->Internals->Pipeline || this->Internals->CachedSampleCount != sampleCount ||
      this->Internals->CachedDepthFormat != depthFormat)
    {
      this->Internals->ReleasePipeline();
      this->Internals->CachedSampleCount = sampleCount;
      this->Internals->CachedDepthFormat = depthFormat;

      NSError* error = nil;
      id<MTLLibrary> library = (__bridge id<MTLLibrary>)renWin->GetSharedShaderLibrary();
      if (!library)
      {
        vtkErrorMacro(<< "No shared shader library available for image mapper");
        return;
      }

      id<MTLFunction> vFunc = [library newFunctionWithName:@"vertex_2d_image_main"];
      id<MTLFunction> fFunc = [library newFunctionWithName:@"fragment_2d_image_main"];
      if (!vFunc || !fFunc)
      {
        vtkErrorMacro(<< "Failed to find 2D image shader functions");
        [vFunc release];
        [fFunc release];
        return;
      }

      MTLVertexDescriptor* vertexDesc = [[MTLVertexDescriptor alloc] init];
      vertexDesc.attributes[0].format = MTLVertexFormatFloat2;
      vertexDesc.attributes[0].offset = 0;
      vertexDesc.attributes[0].bufferIndex = 0;
      vertexDesc.attributes[1].format = MTLVertexFormatFloat2;
      vertexDesc.attributes[1].offset = sizeof(float) * 2;
      vertexDesc.attributes[1].bufferIndex = 0;
      vertexDesc.layouts[0].stride = sizeof(float) * 4;
      vertexDesc.layouts[0].stepRate = 1;
      vertexDesc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;

      MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
      desc.vertexFunction = vFunc;
      desc.fragmentFunction = fFunc;
      desc.vertexDescriptor = vertexDesc;
      desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
      desc.depthAttachmentPixelFormat = depthFormat;
      desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
      desc.rasterSampleCount = sampleCount;
      desc.colorAttachments[0].blendingEnabled = YES;
      desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
      desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
      desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

      this->Internals->Pipeline =
        [device newRenderPipelineStateWithDescriptor:desc error:&error];
      if (!this->Internals->Pipeline)
      {
        vtkErrorMacro(<< "2D image pipeline: " << [[error localizedDescription] UTF8String]);
      }
      [desc release];
      [vertexDesc release];
      [vFunc release];
      [fFunc release];
    }

    if (!this->Internals->Pipeline)
    {
      return;
    }

    // Overlay depth-stencil state (always pass, no write).
    if (!this->Internals->OverlayDepthState)
    {
      MTLDepthStencilDescriptor* dsDesc = [[MTLDepthStencilDescriptor alloc] init];
      dsDesc.depthCompareFunction = MTLCompareFunctionAlways;
      dsDesc.depthWriteEnabled = NO;
      this->Internals->OverlayDepthState = [device newDepthStencilStateWithDescriptor:dsDesc];
      [dsDesc release];
    }

    [encoder setRenderPipelineState:this->Internals->Pipeline];
    [encoder setDepthStencilState:this->Internals->OverlayDepthState];
    [encoder setVertexBuffer:this->Internals->VertexBuffer offset:0 atIndex:0];
    [encoder setVertexBuffer:this->Internals->StateBuffer offset:0 atIndex:1];
    [encoder setFragmentTexture:this->Internals->Texture atIndex:0];
    [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                        indexCount:6
                         indexType:MTLIndexTypeUInt32
                       indexBuffer:this->Internals->IndexBuffer
                 indexBufferOffset:0];
  }
}

VTK_ABI_NAMESPACE_END
