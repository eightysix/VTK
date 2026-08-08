// Standalone reproduction of vtkMetalGPUVolumeRayCastMapper::UpdateGeometry
// camera-inside proxy mesh (boxSource + near-plane clip + densify(2) +
// first-3-verts index list) and dumps vertices + indices + camera matrices for
// winding analysis. See
// VolumeRayCastBackendComparisonFindingsUpdate21.md for the build/run recipe.
#include "vtkCamera.h"
#include "vtkCellArray.h"
#include "vtkClipConvexPolyData.h"
#include "vtkDensifyPolyData.h"
#include "vtkMatrix4x4.h"
#include "vtkNew.h"
#include "vtkPlane.h"
#include "vtkPlaneCollection.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char* argv[])
{
  // Output directory (defaults to /tmp/bc/meshdump).
  std::string outDir = argc > 1 ? argv[1] : "/tmp/bc/meshdump";

  // Volume model bounds (headsq quarter: 64x64x93 @ 3.2x3.2x1.5).
  double mb[6] = { 0, 201.6, 0, 201.6, 0, 138 };
  // Camera from the capture logs (NoJitter variant, W2IF-perturbed angle is
  // applied by the harness only at comparison frames; the base 30.0 is used).
  double camPos[3] = { 102.122314, 102.122314, 61.5619835 };
  double camFocal[3] = { 100.8, 100.8, 69 };
  double clipRange[2] = { 0.192447679, 192.447679 };
  double viewAngle = 30.0;
  double aspect = 1.0;

  vtkNew<vtkPolyData> boxSource;
  {
    vtkNew<vtkCellArray> cells;
    vtkNew<vtkPoints> points;
    points->SetDataTypeToDouble();
    double geometry[24] = {
      mb[0], mb[2], mb[4], mb[1], mb[2], mb[4], mb[1], mb[3], mb[4], mb[0], mb[3], mb[4],
      mb[0], mb[2], mb[5], mb[1], mb[2], mb[5], mb[1], mb[3], mb[5], mb[0], mb[3], mb[5],
    };
    for (int i = 0; i < 8; ++i) points->InsertNextPoint(geometry + i * 3);
    // vtkBoxSource triangle list (12 tris).
    int tris[36] = {
      0, 1, 2, 1, 3, 2, 1, 5, 3, 5, 7, 3, 5, 4, 7, 4, 6, 7, 4, 0, 6, 0, 2, 6, 2, 3, 6, 3, 7, 6,
      0, 4, 1, 1, 4, 5
    };
    for (int i = 0; i < 12; ++i)
    {
      // Inserted as "0 2 1" (the OpenGL/Metal UpdateGeometry wedge convention)
      // to reproduce the exact mesh both backends build.
      cells->InsertNextCell(3);
      cells->InsertCellPoint(tris[i * 3]);
      cells->InsertCellPoint(tris[i * 3 + 2]);
      cells->InsertCellPoint(tris[i * 3 + 1]);
    }
    boxSource->SetPoints(points);
    boxSource->SetPolys(cells);
  }

  vtkNew<vtkMatrix4x4> dataToWorld;
  dataToWorld->Identity();
  vtkNew<vtkMatrix4x4> worldToData;
  vtkMatrix4x4::Invert(dataToWorld, worldToData);

  vtkNew<vtkCamera> cam;
  cam->SetPosition(camPos);
  cam->SetFocalPoint(camFocal);
  cam->SetViewUp(0, 1, 0);
  cam->SetViewAngle(viewAngle);
  cam->SetClippingRange(clipRange);

  FILE* fm = fopen((outDir + "/matrices.txt").c_str(), "w");
  {
    vtkMatrix4x4* V = cam->GetViewTransformMatrix();
    vtkMatrix4x4* P = cam->GetProjectionTransformMatrix(aspect, 0.0, 1.0);
    fprintf(fm, "V\n");
    for (int r = 0; r < 4; ++r)
    {
      fprintf(fm, "%.9g %.9g %.9g %.9g\n", V->GetElement(r, 0), V->GetElement(r, 1),
        V->GetElement(r, 2), V->GetElement(r, 3));
    }
    fprintf(fm, "P\n");
    for (int r = 0; r < 4; ++r)
    {
      fprintf(fm, "%.9g %.9g %.9g %.9g\n", P->GetElement(r, 0), P->GetElement(r, 1),
        P->GetElement(r, 2), P->GetElement(r, 3));
    }
  }
  fclose(fm);

  // Near-plane clip plane: camera frustum near plane pushed into the volume by
  // the precision offset (far-near)*0.001, expressed in data space.
  double fplanes[24];
  cam->GetFrustumPlanes(aspect, fplanes);
  double pOrigin[4];
  pOrigin[3] = 1.0;
  double pNormal[3];
  for (int i = 0; i < 3; ++i)
  {
    pNormal[i] = fplanes[16 + i];
    pOrigin[i] = -fplanes[16 + 3] * fplanes[16 + i];
  }
  double* invMat = worldToData->GetData();
  double pNormalV[3];
  pNormalV[0] = pNormal[0] * invMat[0] + pNormal[1] * invMat[1] + pNormal[2] * invMat[2];
  pNormalV[1] = pNormal[0] * invMat[4] + pNormal[1] * invMat[5] + pNormal[2] * invMat[6];
  pNormalV[2] = pNormal[0] * invMat[8] + pNormal[1] * invMat[9] + pNormal[2] * invMat[10];
  vtkMath::Normalize(pNormalV);

  double offset = (clipRange[1] - clipRange[0]) * 0.001;
  double minOffset = 1.1920928955078125e-7 * 1000.0;
  offset = offset < minOffset ? minOffset : offset;
  for (int i = 0; i < 3; ++i) pOrigin[i] += pNormalV[i] * offset;

  fprintf(stderr, "NEARPLANE origin=(%.9f, %.9f, %.9f) normal=(%.9f, %.9f, %.9f)\n", pOrigin[0],
    pOrigin[1], pOrigin[2], pNormalV[0], pNormalV[1], pNormalV[2]);

  vtkNew<vtkPlane> nearPlane;
  nearPlane->SetOrigin(pOrigin);
  nearPlane->SetNormal(pNormalV);
  vtkNew<vtkPlaneCollection> planes;
  planes->AddItem(nearPlane);

  vtkNew<vtkClipConvexPolyData> clip;
  clip->SetInputData(boxSource);
  clip->SetPlanes(planes);

  vtkNew<vtkDensifyPolyData> densify;
  densify->SetInputConnection(clip->GetOutputPort());
  densify->SetNumberOfSubdivisions(2);
  densify->Update();

  vtkPolyData* out = densify->GetOutput();
  vtkPoints* points = out->GetPoints();
  vtkCellArray* polys = out->GetPolys();

  double bmin[3] = { mb[0], mb[2], mb[4] };
  double bsize[3] = { mb[1] - mb[0], mb[3] - mb[2], mb[5] - mb[4] };

  FILE* fv = fopen((outDir + "/verts.txt").c_str(), "w");
  for (vtkIdType i = 0; i < points->GetNumberOfPoints(); ++i)
  {
    double pt[3];
    points->GetPoint(i, pt);
    fprintf(fv, "(%.9f, %.9f, %.9f)\n", (pt[0] - bmin[0]) / bsize[0],
      (pt[1] - bmin[1]) / bsize[1], (pt[2] - bmin[2]) / bsize[2]);
  }
  fclose(fv);

  FILE* fi = fopen((outDir + "/indices.txt").c_str(), "w");
  vtkIdType npts;
  const vtkIdType* pts;
  int cellIdx = 0;
  polys->InitTraversal();
  while (polys->GetNextCell(npts, pts))
  {
    if (npts < 3) continue;
    fprintf(fi, "%d %lld %lld %lld %lld\n", cellIdx, npts, (long long)pts[0], (long long)pts[1],
      (long long)pts[2]);
    ++cellIdx;
  }
  fclose(fi);
  fprintf(stderr, "verts=%lld cells=%d\n", (long long)points->GetNumberOfPoints(), cellIdx);
  return 0;
}
