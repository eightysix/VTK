// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkMetalPickTypes
 * @brief   shared GPU/CPU picking ID layout for the Metal backend
 *
 * Mirrors the `PickIds` struct in Shaders/MetalShaders.metal, which vertex
 * shaders consume as `constant PickIds&` (triangle/edge/point buffer 7,
 * thick/round/miter line buffer 6, point buffer 12, glyph buffer 10).
 *
 * `PropId` is the prop's per-render index into the active
 * vtkMetalHardwareSelector's visible PropArray (0 when no selection render is
 * active). `CompositeIndex` carries the flat composite-dataset block index for
 * batched blocks (0 = not a composite block).
 */
#ifndef vtkMetalPickTypes_h
#define vtkMetalPickTypes_h

#include "vtkABINamespace.h"

#include <cstdint>

VTK_ABI_NAMESPACE_BEGIN

struct vtkMetalPickIds
{
  uint32_t PropId;
  uint32_t CompositeIndex;
};

#ifdef __cplusplus
static_assert(sizeof(vtkMetalPickIds) == 8,
  "vtkMetalPickIds must match the Metal PickIds struct (2 x uint32)");
#endif

VTK_ABI_NAMESPACE_END

#endif
