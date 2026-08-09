// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkOpenGLVolumeRGBTable.h"

#include "vtkColorTransferFunction.h"
#include "vtkObjectFactory.h"
#include "vtkOpenGLRenderWindow.h"
#include "vtkTextureObject.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>

VTK_ABI_NAMESPACE_BEGIN
vtkStandardNewMacro(vtkOpenGLVolumeRGBTable);

//------------------------------------------------------------------------------
vtkOpenGLVolumeRGBTable::vtkOpenGLVolumeRGBTable()
{
  this->NumberOfColorComponents = 3;
}

//------------------------------------------------------------------------------
void vtkOpenGLVolumeRGBTable::InternalUpdate(vtkObject* func, int vtkNotUsed(blendMode),
  double vtkNotUsed(sampleDistance), double vtkNotUsed(unitDistance), int filterValue)
{
  vtkColorTransferFunction* scalarRGB = vtkColorTransferFunction::SafeDownCast(func);
  if (!scalarRGB)
  {
    return;
  }
  scalarRGB->GetTable(this->LastRange[0], this->LastRange[1], this->TextureWidth, this->Table);
  if (getenv("VTK_GL_OPTABLE_DUMP"))
  {
    unsigned char* bytes = reinterpret_cast<unsigned char*>(this->Table);
    for (int i = 0; i < this->TextureWidth; ++i)
    {
      char hex[25];
      for (int c = 0; c < 3; ++c)
      {
        for (int b = 0; b < 4; ++b)
        {
          std::snprintf(hex + c * 8 + b * 2, 3, "%02x", bytes[(i * 3 + c) * 4 + b]);
        }
      }
      hex[24] = '\0';
      std::cerr << "VTK_METAL_VOLUME_LOG DEBUG GL_RGBTABLE_DUMP idx=" << i << " rgb=" << hex
                << std::endl;
    }
  }
  this->TextureObject->SetWrapS(vtkTextureObject::ClampToEdge);
  this->TextureObject->SetWrapT(vtkTextureObject::ClampToEdge);
  this->TextureObject->SetMagnificationFilter(filterValue);
  this->TextureObject->SetMinificationFilter(filterValue);
  this->TextureObject->Create2DFromRaw(
    this->TextureWidth, 1, this->NumberOfColorComponents, VTK_FLOAT, this->Table);
}

//------------------------------------------------------------------------------
void vtkOpenGLVolumeRGBTable::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
VTK_ABI_NAMESPACE_END
