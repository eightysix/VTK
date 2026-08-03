// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @class   vtkMetalImageSliceMapper
 * @brief   vtkImageSliceMapper override for the Metal rendering backend.
 *
 * Mirrors vtkOpenGLImageSliceMapper: the image slice is rendered as one or more
 * textured polygons (subdivided recursively when larger than the maximum texture
 * size) using an internal vtkMetalPolyDataMapper-backed actor that carries the
 * slice's vtkTexture. Backing and background polygons are rendered the same way.
 * Hardware selection (cell-ID picking) is supported via a dedicated Metal
 * pipeline that encodes the pixel index into the RGBA32Uint picking attachment
 * consumed by vtkMetalHardwareSelector.
 */

#ifndef vtkMetalImageSliceMapper_h
#define vtkMetalImageSliceMapper_h

#include "vtkImageSliceMapper.h"
#include "vtkRenderingMetalModule.h" // for export macro

VTK_ABI_NAMESPACE_BEGIN

class vtkActor;
class vtkHardwareSelector;
class vtkImageProperty;
class vtkOverrideAttribute;
class vtkRenderer;

class VTKRENDERINGMETAL_EXPORT vtkMetalImageSliceMapper : public vtkImageSliceMapper
{
public:
  static vtkMetalImageSliceMapper* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalImageSliceMapper, vtkImageSliceMapper);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /**
   * Render the slice.
   */
  void Render(vtkRenderer* ren, vtkImageSlice* prop) override;

  /**
   * Release any graphics resources that are being consumed by this mapper.
   */
  void ReleaseGraphicsResources(vtkWindow*) override;

protected:
  vtkMetalImageSliceMapper();
  ~vtkMetalImageSliceMapper() override;

  // Render a large image that has been divided into tiles
  void RecursiveRenderTexturedPolygon(vtkRenderer* ren, vtkImageProperty* property,
    vtkImageData* input, int extent[6], bool recursive);

  // Render a textured polygon
  void RenderTexturedPolygon(
    vtkRenderer* ren, vtkImageProperty* property, vtkImageData* input, int extent[6], bool recursive);

  // Render a polygon
  void RenderPolygon(vtkActor* actor, vtkPoints* points, const int extent[6], vtkRenderer* ren);

  // Render a wide black border around the polygon, wide enough to fill
  // the entire viewport
  void RenderBackground(vtkActor* actor, vtkPoints* points, const int extent[6], vtkRenderer* ren);

  // Compute the texture size
  void ComputeTextureSize(
    const int extent[6], int& xdim, int& ydim, int imageSize[2], int textureSize[2]) override;

  // Determine if a given texture size is supported by the device
  bool TextureSizeOK(const int size[2], vtkRenderer* ren);

  // Render the slice for hardware selection (cell-ID picking)
  void RenderForSelection(vtkRenderer* ren, vtkImageSlice* prop, vtkHardwareSelector* selector);

private:
  vtkMetalImageSliceMapper(const vtkMetalImageSliceMapper&) = delete;
  void operator=(const vtkMetalImageSliceMapper&) = delete;

  vtkActor* PolyDataActor;
  vtkActor* BackingPolyDataActor;
  vtkActor* BackgroundPolyDataActor;

  vtkTimeStamp LoadTime;
  int TextureSize[2];
  int TextureBytesPerPixel;

  int LastOrientation;
  int LastSliceNumber;

  class vtkInternals;
  vtkInternals* Internals;
};

#define vtkMetalImageSliceMapper_OVERRIDE_ATTRIBUTES                                             \
  vtkMetalImageSliceMapper::CreateOverrideAttributes()

VTK_ABI_NAMESPACE_END
#endif // vtkMetalImageSliceMapper_h
