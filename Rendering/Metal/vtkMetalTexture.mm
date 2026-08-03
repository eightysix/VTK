// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalTexture.h"

#include "vtkMetalRenderWindow.h"
#include "vtkMetalMRC.h"
#include "vtkObjectFactory.h"
#include "vtkOverrideAttribute.h"
#include "vtkDataArray.h"
#include "vtkImageData.h"
#include "vtkPointData.h"
#include "vtkRenderer.h"
#include "vtkWindow.h"

#include <atomic>

#import <Metal/Metal.h>

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalTexture);

//------------------------------------------------------------------------------
struct vtkMetalTexture::vtkMetalTextureInternals
{
  // The uploaded texture (RGBA8Unorm). Owned here; also registered in the
  // render window's texture-unit registry while the input is unchanged.
  id<MTLTexture> Texture = nil;
  int TextureWidth = 0;
  int TextureHeight = 0;

  vtkMTimeType CachedMTime = 0;
  bool Registered = false;
  int TextureUnit = 0;

  void ReleaseTexture()
  {
    vtkMetalMRC::ReleaseAndNil(Texture);
    TextureWidth = 0;
    TextureHeight = 0;
    CachedMTime = 0;
    Registered = false;
  }
};

//------------------------------------------------------------------------------
vtkOverrideAttribute* vtkMetalTexture::CreateOverrideAttributes()
{
  return vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "Metal", nullptr);
}

vtkMetalTexture::vtkMetalTexture()
  : Internals(new vtkMetalTextureInternals())
{
  // Allocate a unique, stable texture unit for this instance so distinct
  // textures never collide in the render window's BoundTextures registry
  // (mirrors the GL backend's per-texture texture-object units).
  static std::atomic<int> nextUnit{ 0 };
  this->Internals->TextureUnit = ++nextUnit;
}

int vtkMetalTexture::GetTextureUnit()
{
  return this->Internals->TextureUnit;
}

vtkMetalTexture::~vtkMetalTexture()
{
  this->Internals->ReleaseTexture();
}

//------------------------------------------------------------------------------
void vtkMetalTexture::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//------------------------------------------------------------------------------
void vtkMetalTexture::Load(vtkRenderer* ren)
{
  vtkMetalRenderWindow* renWin =
    vtkMetalRenderWindow::SafeDownCast(ren ? ren->GetVTKWindow() : nullptr);
  if (!renWin || !renWin->GetMetalDevice())
  {
    return;
  }

  id<MTLDevice> device = (id<MTLDevice>)renWin->GetMetalDevice();

  vtkImageData* image = this->GetInput();
  vtkDataArray* scalars = image ? image->GetPointData()->GetScalars() : nullptr;
  if (!image || !scalars)
  {
    renWin->SetBoundTexture(this->GetTextureUnit(), nullptr);
    this->Internals->ReleaseTexture();
    return;
  }

  // Guard: only unsigned char textures are supported (same as the 3D mapper).
  if (scalars->GetDataType() != VTK_UNSIGNED_CHAR)
  {
    vtkErrorMacro(<< "vtkMetalTexture: only unsigned char textures are currently supported");
    renWin->SetBoundTexture(this->GetTextureUnit(), nullptr);
    this->Internals->ReleaseTexture();
    return;
  }

  vtkMTimeType imageMTime = image->GetMTime();
  if (this->Internals->Registered && this->Internals->CachedMTime == imageMTime)
  {
    return;
  }

  int extent[6];
  image->GetExtent(extent);
  int width = extent[1] - extent[0] + 1;
  int height = extent[3] - extent[2] + 1;
  int numComponents = image->GetNumberOfScalarComponents();

  // Match vtkOpenGLTexture::Load: when the scalar tuple count equals the cell
  // count (a one-tuple-per-cell image), shrink the texture dimensions by 1
  // along each axis so the extent-sized grid does not read out of bounds.
  if (image->GetNumberOfCells() == scalars->GetNumberOfTuples())
  {
    if (width > 1)
    {
      --width;
    }
    if (height > 1)
    {
      --height;
    }
  }

  if (width <= 0 || height <= 0)
  {
    renWin->SetBoundTexture(this->GetTextureUnit(), nullptr);
    this->Internals->ReleaseTexture();
    return;
  }

  // Create the texture when the dimensions change.
  if (!this->Internals->Texture || this->Internals->TextureWidth != width ||
    this->Internals->TextureHeight != height)
  {
    MTLTextureDescriptor* texDesc = [[MTLTextureDescriptor alloc] init];
    texDesc.textureType = MTLTextureType2D;
    texDesc.pixelFormat = MTLPixelFormatRGBA8Unorm;
    texDesc.width = width;
    texDesc.height = height;
    texDesc.mipmapLevelCount = 1;
    texDesc.usage = MTLTextureUsageShaderRead;
    texDesc.storageMode = MTLStorageModeShared;

    id<MTLTexture> newTexture = [device newTextureWithDescriptor:texDesc];
    vtkMetalMRC::AssignConsumed(this->Internals->Texture, newTexture);
    [texDesc release];
    if (!this->Internals->Texture)
    {
      vtkErrorMacro(<< "Failed to create Metal texture");
      return;
    }
    this->Internals->TextureWidth = width;
    this->Internals->TextureHeight = height;
  }

  // Convert the image to RGBA8 and upload. VTK image row 0 (min y) is uploaded
  // first, so texcoord (0,0) samples the bottom row exactly as OpenGL renders
  // it (no vertical flip), matching vtkMetalPolyDataMapper::UpdateActorTexture.
  unsigned char* rgbaData = new unsigned char[width * height * 4];
  int xMin = extent[0];
  int yMin = extent[2];
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      unsigned char* srcPtr =
        static_cast<unsigned char*>(image->GetScalarPointer(xMin + x, yMin + y, 0));
      unsigned char* dst = rgbaData + (y * width + x) * 4;
      switch (numComponents)
      {
        case 1:
          dst[0] = dst[1] = dst[2] = srcPtr[0];
          dst[3] = 255;
          break;
        case 2:
          dst[0] = dst[1] = dst[2] = srcPtr[0];
          dst[3] = srcPtr[1];
          break;
        case 3:
          dst[0] = srcPtr[0];
          dst[1] = srcPtr[1];
          dst[2] = srcPtr[2];
          dst[3] = 255;
          break;
        case 4:
          dst[0] = srcPtr[0];
          dst[1] = srcPtr[1];
          dst[2] = srcPtr[2];
          dst[3] = srcPtr[3];
          break;
        default:
          dst[0] = dst[1] = dst[2] = dst[3] = 255;
          break;
      }
    }
  }

  MTLRegion region = MTLRegionMake2D(0, 0, width, height);
  [this->Internals->Texture replaceRegion:region
                              mipmapLevel:0
                                withBytes:rgbaData
                              bytesPerRow:width * 4];
  delete[] rgbaData;

  // Register the texture so 2D mappers can resolve it by texture unit.
  renWin->SetBoundTexture(this->GetTextureUnit(), (void*)this->Internals->Texture);
  this->Internals->CachedMTime = imageMTime;
  this->Internals->Registered = true;
}

//------------------------------------------------------------------------------
void vtkMetalTexture::PostRender(vtkRenderer*)
{
  // The texture stays registered in the window; the next render's Load() will
  // reuse it while the input image is unchanged.
}

//------------------------------------------------------------------------------
void vtkMetalTexture::ReleaseGraphicsResources(vtkWindow* win)
{
  vtkMetalRenderWindow* renWin = vtkMetalRenderWindow::SafeDownCast(win);
  if (renWin)
  {
    renWin->SetBoundTexture(this->GetTextureUnit(), nullptr);
  }
  this->Internals->ReleaseTexture();
}

VTK_ABI_NAMESPACE_END
