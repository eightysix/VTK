// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @class   vtkMetalLabeledContourMapper
 * @brief   vtkLabeledContourMapper override for the Metal rendering backend.
 *
 * vtkLabeledContourMapper draws labeled isolines using stenciling to keep the
 * isolines from drawing through the label quads. The Metal backend does not yet
 * implement a stencil buffer, so the stencil passes are no-ops: the isolines
 * are drawn continuously and the labels are composited on top of them, which
 * reproduces the OpenGL result for the common case where the label quads have
 * opaque or near-opaque backgrounds (the translucent-background labels differ
 * only where the thin isoline passes behind the label).
 *
 * The OpenGL override additionally pushes the actor's matrix onto each label's
 * user matrix so the labels follow a transformed actor; the base class does not
 * do that, so a transformed actor (e.g. TestLabeledContourMapperWithActorMatrix)
 * would draw its labels in the wrong place without this override.
 */

#ifndef vtkMetalLabeledContourMapper_h
#define vtkMetalLabeledContourMapper_h

#include "vtkLabeledContourMapper.h"
#include "vtkRenderingMetalModule.h" // for export macro
#include "vtkWrappingHints.h"        // for VTK_MARSHALAUTO

VTK_ABI_NAMESPACE_BEGIN

class vtkOverrideAttribute;

class VTKRENDERINGMETAL_EXPORT VTK_MARSHALAUTO vtkMetalLabeledContourMapper
  : public vtkLabeledContourMapper
{
public:
  static vtkMetalLabeledContourMapper* New();
  VTK_NEWINSTANCE
  static vtkOverrideAttribute* CreateOverrideAttributes();
  vtkTypeMacro(vtkMetalLabeledContourMapper, vtkLabeledContourMapper);
  void PrintSelf(ostream& os, vtkIndent indent) override;

protected:
  vtkMetalLabeledContourMapper();
  ~vtkMetalLabeledContourMapper() override;

  // The Metal backend passes each actor's matrix to the shader individually, so
  // this mapper's actor matrix does not affect the label rendering; mirror the
  // OpenGL override by folding the actor's matrix into each label's user matrix.
  bool CreateLabels(vtkActor* actor) override;

private:
  vtkMetalLabeledContourMapper(const vtkMetalLabeledContourMapper&) = delete;
  void operator=(const vtkMetalLabeledContourMapper&) = delete;
};

#define vtkMetalLabeledContourMapper_OVERRIDE_ATTRIBUTES                                           \
  vtkMetalLabeledContourMapper::CreateOverrideAttributes()

VTK_ABI_NAMESPACE_END
#endif // vtkMetalLabeledContourMapper_h
