// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

#include "vtkMetalGPUVolumeRayCastMapper.h"

#include "vtkMetalRenderWindow.h"
#include "vtkMetalRenderer.h"
#include "vtkMetalCamera.h"
#include "vtkMetalShaders.h"
#include "vtkColorTransferFunction.h"
#include "vtkDataSet.h"
#include "vtkImageData.h"
#include "vtkNew.h"
#include "vtkObjectFactory.h"
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "vtkOverrideAttribute.h"
#include "vtkPiecewiseFunction.h"
#include "vtkPointData.h"
#include "vtkRectilinearGrid.h"
#include "vtkRenderer.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkCamera.h"
#include "vtkCellData.h"
#include "vtkMatrix4x4.h"
#include "vtkUnsignedCharArray.h"
#include "vtkSMPTools.h"
#include "vtkMath.h"
#include "vtkClipConvexPolyData.h"
#include "vtkDensifyPolyData.h"
#include "vtkPlaneCollection.h"
#include "vtkPlane.h"
#include "vtkLightCollection.h"
#include "vtkLight.h"
#include "vtkPolyData.h"
#include "vtkPoints.h"
#include "vtkCellArray.h"
#include "vtkDataObject.h"
#include "vtkHardwareSelector.h"
#include "vtkMetalHardwareSelector.h"
#include <limits>

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <Accelerate/Accelerate.h>
#import <simd/simd.h>

#include <algorithm>
#include <cstring>
#include <cstddef>
#include <set>
#include <vector>
#include <chrono>
#include <dispatch/dispatch.h>

// Metal constant-address-space structs align float3 to 16 bytes (size 16),
// float4/float4x4 to 16 bytes, and float2 to 8 bytes.  This creates
// padding that plain C++ float[] arrays do not.  The layout below exactly
// mirrors the Metal compiler's computation (480 bytes total).
struct VolumeMapperUniforms
{
  // --- fields matching Metal layout 1:1, offsets verified ---
  float WorldToVolumeMatrix[16];     // 0..63
  float VolumeToWorldMatrix[16];     // 64..127
  float VolumeBoundsMin[4];          // 128..143
  float VolumeBoundsMax[4];          // 144..159
  float CameraVolumePos[4];          // 160..175
  float ViewProjectionMatrix[16];    // 176..239
  uint16_t SampleDistanceHalf;      // 240  (half precision: sufficient for [0,1] space)
  uint16_t OpacityPreIntegrationFactorHalf; // 242  unused; pre-integration baked into TF on CPU. Kept for struct layout.
  uint16_t ScalarMinHalf;           // 244
  uint16_t _padSM;                  // 246
  uint16_t ScalarMaxHalf;           // 248
  uint16_t _padSMax;                // 250
  float UseJittering;                // 252
  float InverseViewProjection[16];   // 256..319
  float ViewportSize[2];            // 320..327
  float _padViewport[2];            // 328..335  (pad to 16-byte for float3)
  float GradientStep[3];            // 336..347
  float _padGradStep;               // 348..351  (Metal: float3 = 16 bytes)
  float UseGradientShading;         // 352
  float _padGradOpRange;            // 356..359  (pad to 8-byte for float2)
  float GradientOpacityMin;         // 360
  float GradientOpacityMax;         // 364
  float UseGradientOpacity;         // 368
  float _padAmbient[3];             // 372..383  (pad to 16-byte for float4)
  float AmbientColor[3];            // 384..395
  float _padAmb;                    // 396..399  (Metal: float4 = 16 bytes)
  float DiffuseColor[3];            // 400..411
  float _padDiff;                   // 412..415
  float SpecularColor[3];           // 416..427
  float _padSpec;                   // 428..431
  float Shininess;                  // 432
  float _padLightDir[3];            // 436..447  (pad to 16-byte for float3)
  float LightDirection[3];          // 448..459
  float _padLight;                  // 460..463  (Metal: float3 = 16 bytes)
  float _padEnd[4];                 // 464..479  (trailing pad to 480)
  // Cropping regions (new)
  float CroppingPlanes[4];          // 480..495  (minX, maxX, minY, maxY)
  float CroppingPlanes2[4];         // 496..511  (minZ, maxZ, 0, 0)
  uint32_t CroppingBitmask;         // 512..515  (packed bitmask from GetCroppingRegionFlags)
  float _padCropFlags[31];          // 516..639  (maintain total struct size)
  float UseCropping;                // 640
  float UseClipping;                // 644
  float NumClippingPlanes;          // 648
  float _padClipping[2];            // 652..659
  float _padClipAlign[3];           // 660..671
  // Clipping planes (up to 8 arbitrary planes)
  float ClippingPlane0Origin[4];    // 672..687 (origin.xyz, 1.0)
  float ClippingPlane0Normal[4];    // 688..703 (normal.xyz, 0.0)
  float ClippingPlane1Origin[4];    // 704..719
  float ClippingPlane1Normal[4];    // 720..735
  float ClippingPlane2Origin[4];    // 736..751
  float ClippingPlane2Normal[4];    // 752..767
  float ClippingPlane3Origin[4];    // 768..783
  float ClippingPlane3Normal[4];    // 784..799
  float ClippingPlane4Origin[4];    // 800..815
  float ClippingPlane4Normal[4];    // 816..831
  float ClippingPlane5Origin[4];    // 832..847
  float ClippingPlane5Normal[4];    // 848..863
  float ClippingPlane6Origin[4];    // 864..879
  float ClippingPlane6Normal[4];    // 880..895
  float ClippingPlane7Origin[4];    // 896..911
  float ClippingPlane7Normal[4];    // 912..927
  // Mask / label map support
  float UseMask;                  // 928
  float MaskBlendFactor;          // 932
  float MaskScale;                // 936
  float MaskBias;                 // 940
  float LabelMapNumLabels;        // 944
  float UseDepthTexture;          // 948
  float UseNormalTexture;         // 952
  float UseLinearVolumeInterpolation; // 956  1.0 = trilinear (VTK_LINEAR_INTERPOLATION), 0.0 = nearest
  // Min-max acceleration texture
  float UseMinMaxAccel;           // 960
  float MinMaxDimX;               // 964
  float MinMaxDimY;               // 968
  float MinMaxDimZ;               // 972
  float UseRenderToImage;         // 976
  float ClampDepthToBackface;     // 980
  // 2D transfer function mode (TF_2D)
  float UseTransfer2D;            // 984  yNorm = yRaw * scale + bias
  float Transfer2DYAxisScale;     // 988
  float Transfer2DYAxisBias;      // 992
  float Transfer2DUseGradient;    // 996  (1.0 = y-axis is gradient magnitude, no Y-axis array)
  float AverageIPRangeMin;        // 1000  (native scalar units / ScalarNormalizationFactor)
  float AverageIPRangeMax;        // 1004
  float MaskType;                 // 1008  (0=label map, 1=binary mask)
  // Final color window/level (matches OpenGL in_scale/in_bias in finalizeRayCast)
  float FinalColorScale;          // 1012  (1.0 / FinalColorWindow)
  float FinalColorBias;           // 1016  (0.5 - FinalColorLevel / FinalColorWindow)
  // Uniform-grid blanking (ghost arrays)
  float UseBlanking;              // 1020  (1.0 = blanking texture bound and in use)
  float BlankingMode;             // 1024  (1 = cell, 2 = point, 3 = both)
  float _padDirAlign[3];          // 1028..1039  (pad to 16-byte for float4x4)
  // Image-data direction support (OpenGL TextureToDataset parity):
  // TextureToVolume maps [0,1] texture coords to model-space (rotated dataset)
  // coords including the image-data direction matrix; VolumeToTexture is its
  // inverse.
  float TextureToVolumeMatrix[16]; // 1040..1103
  float VolumeToTextureMatrix[16]; // 1104..1167
  // Independent multi-component support (OpenGL in_scalarsRange parity).
  uint16_t ScalarMinCompHalf[4];   // 1168..1175  per-component scalar range / ScalarNormalizationFactor
  uint16_t _padSMComp[4];          // 1176..1183
  uint16_t ScalarMaxCompHalf[4];   // 1184..1191
  uint16_t _padSMaxComp[4];        // 1192..1199
  float ComponentWeight[4];        // 1200..1215
  uint32_t NumComponents;          // 1216..1219
  float UseIndependentComponents;  // 1220..1223
  float UseDependentLA;            // 1224..1227  (dependent 2-comp LA: color LUT on comp0, opacity LUT on last comp)
  float _padIndependent[2];        // 1228..1235
  float UseDependentRGBA;          // 1236..1239  (4-comp dependent RGBA: raw RGB color, opacity LUT on scalar.w)
  float UseComputeNormalFromOpacity; // 1240..1243  (1.0 = shade with the opacity-field gradient; OpenGL ComputeNormalFromOpacity parity)
  float _padEnd2[1];               // 1244..1247  (trailing pad to 1248)
  // Per-component material (OpenGL in_ambient[4]/in_diffuse[4]/in_specular[4]/
  // in_shininess[4] parity). Indexed by component in the independent path.
  float AmbientColorComp[4][4];    // 1248..1311
  float DiffuseColorComp[4][4];    // 1312..1375
  float SpecularColorComp[4][4];   // 1376..1439
  float ShininessComp[4];          // 1440..1455  (total 1456, 16-byte aligned)
  // Hardware-selection (vtkHardwareSelector) support — OpenGL CheckPickingState
  // / PickingActorPassExit parity. selectionMode == 0 keeps the normal path.
  float SelectionMode;             // 1456
  float _padSel[3];                // 1460..1471
  uint32_t SelectionPropId;        // 1472  0-based selector PropArray index
  uint32_t SelectionCompositeIndex; // 1476 (0 for a single vtkVolume)
  uint32_t SelectionVolumeDimX;    // 1480  volume dimensions for the voxel index
  uint32_t SelectionVolumeDimY;    // 1484
  uint32_t SelectionVolumeDimZ;    // 1488
  uint32_t _padSelEnd[3];          // 1492..1503
  // Parallel-projection support (OpenGL in_projectionDirection parity): when
  // useParallelProjection is set the fragment shader builds every ray from the
  // interpolated proxy-box position along this constant direction (in [0,1]
  // normalized volume space) instead of converging rays from the camera
  // position. w is unused.
  float UseParallelProjection;     // 1504
  float ProjectionDirection[4];    // 1508..1523
  float _padParallelEnd[3];        // 1524..1535
  // Precomputed NDC -> [0,1] normalized volume-space matrix (folds
  // InverseViewProjection * WorldToVolumeMatrix * the volume-bounds normalize
  // into a single transform) so the fullscreen/grid ray setup and the depth
  // termination paths do one matrix-vector multiply instead of two matrix
  // chains plus a bounds re-normalize per fragment.
  float NDCToVolumeMatrix[16];     // 1536..1599 (total 1600, 16-byte aligned)
  // Rectilinear-grid support (OpenGL in_coordTexs / in_coordsScale /
  // in_coordsBias parity): when UseRectilinear is set the fragment shader walks
  // the per-axis coordinate curves (RectCoordsBuffer at fragment buffer 5,
  // float3 per index padded to the longest axis) to remap each sample's
  // data-space position to the index-space texture coordinate instead of
  // sampling the uniform-spacing proxy directly.
  float UseRectilinear;            // 1600
  float _padRect[3];               // 1604..1615
  float RectCoordsSizes[4];        // 1616..1631  xyz = number of coords per axis
  float RectCoordsScale[4];        // 1632..1647  per-axis GetScaleAndBias scale
  float RectCoordsBias[4];         // 1648..1663  per-axis GetScaleAndBias bias
  // Camera-inside near-plane clip (OpenGL near-plane proxy-clip parity): when
  // the near frustum plane crosses the bounding box, OpenGL clips the proxy box
  // against the near plane (pushed in by a precision offset) and starts the
  // march there. setupVolumeRay clamps the ray entry to this plane (origin and
  // normal in [0,1] normalized volume space).
  float UseCameraInsideNearClip;   // 1664
  float _padNearClip[3];           // 1668..1679
  float CameraInsideNearPlaneOrigin[4]; // 1680..1695
  float CameraInsideNearPlaneNormal[4]; // 1696..1711
  // OpenGL proxy-box parity: the camera-OUTSIDE proxy box is uploaded in
  // dataset (model) space like GL's densified BBoxPolyData (unit-cube scaling
  // in the vertex shader rounds the centroid vertices ~1 ulp off GL's double
  // centroids, which kept the interpolated anchor ~1 ulp off). When set, the
  // vertex shader forwards in.position unchanged (modelPos = in.position).
  float UseDataSpaceBoxVertices;   // 1712
  float UseIGNJitter;              // 1716  (1.0 = Interleaved Gradient Noise jitter instead of the GL blue-noise tile)
  float JitterBlockSize;           // 1720  (pixels per IGN-jitter coherence block; 1 = legacy per-pixel)
  // Non-divergent march (PERFORMANCE_INVESTIGATION.md section 4): when > 0, the
  // fragment march runs a uniform iteration count (frame-max ray-box chord /
  // sample distance, computed per frame on the CPU) with all data-dependent
  // exits predicated instead of breaking, so SIMT lanes stay locked. 0 keeps
  // the legacy per-fragment loop bound. Occupies the former _padDSBV slot.
  float MaxStepsFrame;             // 1724..1727
  float MaxBatchWidth;             // 1728..1731 (adaptive-width march cap for
                                   // fc_marchVariant 9, set from sample distance;
                                   // total 1744, 16-byte aligned)
  // §37.15 block-or-nothing (VTK_METAL_TEST_MM_BLOCKSONLY): when > 0.5 the
  // fragment-march preamble consults ONLY the super/block occupancy levels;
  // a mixed block dispatches its batch un-walked instead of running the
  // per-cell lattice walk. Targets the axis-chord pathology (§37.13): on
  // fragmented terrain the cell walk pays serialized per-step work for
  // near-zero skip yield. Output-safe: dropped skips cover provably-zero
  // samples only (identical sample positions, zero-opacity contributions).
  float MmBlocksOnly;              // 1732..1735
  // §37.17 leap-granularity selector (VTK_METAL_TEST_MM_LEAPLEVEL): 2 =
  // super+block leaps (default/landed behavior), 1 = super leaps only
  // (coherence probe: 8x fewer leap events), <=0 = no occupancy leaps.
  float MmLeapLevel;               // 1736..1739
  // §37.18 block-summary edge in fine cells ({4,8,16,32}, default 8); must
  // match VolumeMinMaxBlockSize() used to build minMaxBlockTexture. Supers
  // remain fixed 64-cell tiles (blocks-per-super = 64 / this).
  float MmBlockSizeCells;          // 1740..1743
  // §37.19 warp-coherent skipping (VTK_METAL_TEST_MM_WARPMIN): when > 0.5,
  // each outer march iteration probes the block summary once per lane and
  // advances the whole SIMD-group by the warp-minimum leap (all-empty-block
  // distance), preserving cross-lane slice-lockstep on straight chords.
  // Any dissenting lane (mixed/solid block) zeroes the warp leap and the
  // march falls back to the legacy per-lane walk, leaving oblique-path
  // skipping fully intact.
  float MmWarpMin;                 // 1744..1747
  // §38 TF-adaptive opacity-saturation exit (VTK_METAL_TEST_EXIT_THETA):
  // accumulated-opacity value that terminates the march when fc_exitTheta
  // specializes the pipeline in; legacy latch value (1 - 1/255) otherwise.
  float ExitAlpha;                 // 1748..1751
  // Cinematic rendering — shaded DVR cinematic (wax AO+SSS, front-to-back
  // over, first-surface AO/SSS, HG phase on headlight). Added at struct tail
  // to preserve all prior offsets (bit-identical for non-cinematic renders).
  // Reserved: CinematicSamples / MaxBounces / MajorantSigma are staging for
  // future delta-tracking but unused at 1 spp: the current march is DVR, not
  // Woodcock.
  uint32_t CinematicSamples;       // 1752..1755 (64 brain default)
  uint32_t CinematicMaxBounces;    // 1756..1759 (4)
  float CinematicScatteringAnisotropy; // 1760..1763 g Hengyey-Greenstein
  float CinematicReach;            // 1764..1767 GlobalIlluminationReach
  float CinematicBlend;            // 1768..1771 VolumetricScatteringBlending
  float CinematicDenoise;          // 1772..1775 0..1 guided filter weight
  float SubsurfaceColor[3];        // 1776..1787 warm pink 0.89/0.73/0.68
  float SubsurfaceStrength;        // 1788..1791 0..1
  uint32_t CinematicFrameSeed;     // 1792..1795 temporal jitter seed
  uint32_t CinematicAccumCount;    // 1796..1799 progressive count
  float CinematicMajorantSigma;    // 1800..1803 reserved (Woodcock majorant, unused at 1 spp DVR)
  float _padCinematic;             // 1804..1807 pad
  float CinematicEnabled;          // 1808..1811 1.0 when cinematic active
  float _padCinematicEnd[3];       // 1812..1823 tail pad to 1824
};

static_assert(sizeof(VolumeMapperUniforms) == 1824,
  "VolumeMapperUniforms must be 1824 bytes to match Metal shader struct");

// MSL rounds the shader-side struct up to its 16-byte alignment (float4/float3
// members), so the pipeline expects round_up(1732,16)=1744 and Metal's
// validation layer asserts when the bound buffer is smaller (Xcode/Debug
// launches). All FIELD offsets are identical on both sides — only trailing
// padding differs — so allocating the rounded size is purely a validation fix.
static constexpr NSUInteger VolumeUniformBufferSize =
  (static_cast<NSUInteger>(sizeof(VolumeMapperUniforms)) + 15) & ~NSUInteger(15);
static_assert(VolumeUniformBufferSize == 1824, "rounded uniform size");

static_assert(offsetof(VolumeMapperUniforms, UseCropping) == 640, "");
static_assert(offsetof(VolumeMapperUniforms, UseClipping) == 644, "");
static_assert(offsetof(VolumeMapperUniforms, NumClippingPlanes) == 648, "");
static_assert(offsetof(VolumeMapperUniforms, ClippingPlane0Origin) == 672, "");
static_assert(offsetof(VolumeMapperUniforms, UseMask) == 928, "");
static_assert(offsetof(VolumeMapperUniforms, UseDepthTexture) == 948, "");
static_assert(offsetof(VolumeMapperUniforms, UseNormalTexture) == 952, "");
static_assert(offsetof(VolumeMapperUniforms, UseMinMaxAccel) == 960, "");
static_assert(offsetof(VolumeMapperUniforms, AverageIPRangeMin) == 1000, "");
static_assert(offsetof(VolumeMapperUniforms, AverageIPRangeMax) == 1004, "");
static_assert(offsetof(VolumeMapperUniforms, MaskType) == 1008, "");
static_assert(offsetof(VolumeMapperUniforms, TextureToVolumeMatrix) == 1040, "");
static_assert(offsetof(VolumeMapperUniforms, VolumeToTextureMatrix) == 1104, "");
static_assert(offsetof(VolumeMapperUniforms, ScalarMinCompHalf) == 1168, "");
static_assert(offsetof(VolumeMapperUniforms, ComponentWeight) == 1200, "");
static_assert(offsetof(VolumeMapperUniforms, NumComponents) == 1216, "");
static_assert(offsetof(VolumeMapperUniforms, UseComputeNormalFromOpacity) == 1240, "");
static_assert(offsetof(VolumeMapperUniforms, AmbientColorComp) == 1248, "");
static_assert(offsetof(VolumeMapperUniforms, DiffuseColorComp) == 1312, "");
static_assert(offsetof(VolumeMapperUniforms, SpecularColorComp) == 1376, "");
static_assert(offsetof(VolumeMapperUniforms, ShininessComp) == 1440, "");
static_assert(offsetof(VolumeMapperUniforms, SelectionMode) == 1456, "");
static_assert(offsetof(VolumeMapperUniforms, SelectionPropId) == 1472, "");
static_assert(offsetof(VolumeMapperUniforms, UseRectilinear) == 1600, "");
static_assert(offsetof(VolumeMapperUniforms, RectCoordsBias) == 1648, "");
static_assert(offsetof(VolumeMapperUniforms, UseCameraInsideNearClip) == 1664, "");
static_assert(offsetof(VolumeMapperUniforms, CameraInsideNearPlaneOrigin) == 1680, "");
static_assert(offsetof(VolumeMapperUniforms, CameraInsideNearPlaneNormal) == 1696, "");
static_assert(offsetof(VolumeMapperUniforms, UseDataSpaceBoxVertices) == 1712, "");

// Per-light data for volume shading — must match Metal VolumeLight struct
// Must match Metal VolumeLight (6 x float4 = 96 bytes per light)
struct VolumeLightData {
  float position[4];      // xyz + type (0=directional, 1=positional)
  float direction[4];     // xyz + cone angle
  float ambientColor[4];  // rgb * intensity
  float diffuseColor[4];  // rgb * intensity
  float specularColor[4]; // rgb * intensity
  float attenuation[4];   // constant, linear, quadratic, spot exponent
};

// Must match Metal VolumeLightUniforms exactly (800 bytes)
struct VolumeLightUniforms {
  VolumeLightData lights[8];  // 8 * 96 = 768 bytes
  int32_t lightCount;         // 4 bytes
  int32_t numPositionalLights;// 4 bytes (informational, not used in shader)
  int32_t twoSidedLighting;   // 4 bytes
  int32_t defaultLighting;    // 4 bytes
  int32_t _pad[4];            // 16 bytes; total = 800
};

static_assert(sizeof(VolumeLightUniforms) == 800,
  "VolumeLightUniforms must be 800 bytes to match Metal shader struct");

// Map vtkVolumeMapper::BlendMode to the VolumeShaderFeatureFlags bit used to
// specialize the fc_blendMode function constant (composite → no flag).
static uint32_t BlendModeToFeatureFlag(int blendMode)
{
  switch (blendMode)
  {
    case vtkVolumeMapper::MAXIMUM_INTENSITY_BLEND:
      return VolumeFeature_BlendMaximumIntensity;
    case vtkVolumeMapper::MINIMUM_INTENSITY_BLEND:
      return VolumeFeature_BlendMinimumIntensity;
    case vtkVolumeMapper::AVERAGE_INTENSITY_BLEND:
      return VolumeFeature_BlendAverageIntensity;
    case vtkVolumeMapper::ADDITIVE_BLEND:
      return VolumeFeature_BlendAdditive;
    default:
      return 0;
  }
}

// March-experiment selector from VTK_METAL_TEST_MARCH_VARIANT (0=none,
// 1=manual 8-tap trilinear, 2=clamp_to_zero sampler, 3=predicated opacity exit,
// 4=uniform frame-max loop with all exits predicated, 6=8x unrolled march,
// 7=4x unrolled march, 8=harness-style scheduled march, 9=48-wide inline
// scheduled march with minmax). Encoded into the feature mask so each
// experiment gets its own specialized pipeline.
//
// Read LIVE (not cached) so GUI toggles can flip it between pipelines: every
// consumer runs during per-frame feature-mask/uniform setup, so an env change
// rebuilds the specialized PSO on the next render — same mechanism as any
// function-constant flip. Default is 9: the 48-wide inline-address scheduled
// march (probe w48), which beats the 8-wide harness-style scheduled march
// (variant 8) on the DICOM app benchmark: mv=9 50-53 ms vs mv=8 60-66 ms (GL
// 48-51 ms) with byte-identical output (see PERFORMANCE_INVESTIGATION.md
// sections 17-18). Variant 9 also carries the minmax lattice walk (fc_minmax),
// replacing the batch-8 consume minmax path that was unstable and up to 12x
// slower than GL (150-580ms vs 47ms on the DICOM study). With minmax enabled
// the same 48-wide batches are issued only over non-empty macrocells: mv=9 +
// minmax measures ~25ms vs GL 47-49ms (0.51x), thresholded error 0.000.
// Variants 6/7/8 fall back to the batch-8 consume when a non-lean feature is
// active. Setting the env var overrides the default for A/B testing.
static int VolumeMarchVariant()
{
  if (const char* v = getenv("VTK_METAL_TEST_MARCH_VARIANT"))
    return std::atoi(v);
  return 0; // TEMP-REPRO: 0 = baseline march, no experiment (revert to 9)
}

// V31 back-edge exit experiment (VTK_METAL_TEST_DOEXIT=1): reshapes the
// baseline divergent march into a do-while with all exit conditions folded
// into the back-edge (divergent_tail_repro V31 root-cause fix: MSL codegen for
// the mid-body-exit CFG loses to GLSL->Air on data-dependent trip counts).
// Encoded as its own feature bit so it gets a specialized pipeline
// independent of fc_marchVariant. Reads the env var once per process.
static bool VolumeMarchDoExit()
{
  static const bool doExit = getenv("VTK_METAL_TEST_DOEXIT") != nullptr;
  return doExit;
}

// Volume texture upload layout (lag_repro root cause, 2026-08-18): the GPU's
// lossless optimized swizzle (MTLTextureDescriptor.allowGPUOptimizedContents,
// default YES) taxes incompressible per-texel-varying payloads (DICOM/CT-like
// noise) ~1.3-1.8x on the 3D read path; with the flag NO Metal matches or beats
// GL (lag_repro: noise 79.5 -> 44.2 ms; metal_gap --noopt: sd0.5 noise 310 ->
// 92 ms, 3.4x; GL parity 11/13 harness cells). Default NO for volume data per
// the lag_repro rule (keep YES only for compressible data); set
// VTK_METAL_TEST_GPU_OPTIMIZED_CONTENTS=1 to restore the legacy YES for A/B.
static bool VolumeGPUOptimizedContents()
{
  static const bool opt = [] {
    if (const char* v = getenv("VTK_METAL_TEST_GPU_OPTIMIZED_CONTENTS"))
      return std::atoi(v) != 0;
    return false; // NO: uncompressed layout (the root-cause fix)
  }();
  return opt;
}

// RG8 pair-packed slice representation (VTK_METAL_TEST_RG8=1,
// divergent_tail_repro V24/V32): store R=slice 2z / G=slice 2z+1 in each
// texel of a halved-depth RG8 volume so a trilinear z-blend needs ONE
// XY-bilinear tap when floor(z) is even and two when odd (~1.25 average vs
// the 3D trilinear's two z-slice accesses). Image-exact reconstruction of
// the hardware trilinear result; targets the phase-scatter sampler tax that
// scales with z-slice-pair traffic. Single-component 8-bit direct-upload
// volumes only (no conversion pipeline, no shading/gradient paths — those
// read the volume raw and would need their own port).
static bool VolumeRg8PairActive()
{
  static const bool rg8 = [] {
    if (const char* v = getenv("VTK_METAL_TEST_RG8"))
      return std::atoi(v) != 0;
    return false;
  }();
  return rg8;
}

// Transposed volume representation from VTK_METAL_TEST_VOLTRANSPOSE (see the
// VolumeFeature_VolTransposed docs). ON by default; set to 0 to opt out.
// Single-component 8-bit direct-upload volumes only — the same constraint
// class as the RG8 repack (no conversion pipeline, no multi-component
// expansion may run before the transpose).
static bool VolumeTransposedActive()
{
  // NOTE: distinct from TestMetalScenes' older scene-level
  // VTK_METAL_TEST_TRANSPOSE diagnostic (vtkImagePermute at data load).
  // This one transposes INSIDE the upload and swizzles fetches in-shader,
  // so world-space rendering stays identical (byte parity).
  // Re-read per call (called only at volume upload / pipeline build, never
  // per frame) so test-app runtime toggles take effect on the next reload.
  if (const char* v = getenv("VTK_METAL_TEST_VOLTRANSPOSE"))
    return std::atoi(v) != 0;
  return true;
}

// §29 orientation policy for the transposed representation. Returns which
// ORIGINAL axis the texture's DEPTH extent should hold: 0 = identity (no
// transpose), 1 = X-depth, 2 = Y-depth.
//
// Rationale (HARNESS_VS_APP_GAP §26/§29): Metal's private 3D tiling taxes
// trilinear depth-pair fetches under jitter phase scatter, and the tax
// scales with the extent marched inside texture depth (long-axis-in-depth
// measured +60.9 ms jitter delta @2048 vs +5.2 ms short-axis-in-depth on
// identical data). So the shortest array dimension goes to depth; when the
// original Z is already (tied-)shortest, keep the identity layout and skip
// the repack entirely; ties between X and Y prefer X (matches every cell
// validated in §26.5/§27).
//
// Y-on-ties was A/B'd across all 12 coarse-tier view classes (doc §38,
// 2026-08-24, cap32/blocks-default code): Y won 10/12 (raw az45 -45%, mm
// az45 -33%, axx ~-21%) but LOST raw axis-y +27% (mm +2.7%) — not a uniform
// winner, so the X tie-break stays (a per-view gate is an anti-pattern,
// §25.5). Y-depth remains available as an opt-in for axis-dominant static
// workloads via VTK_METAL_TEST_VOLTRANSPOSE_AXIS=y; in the fine-SD tier
// (sd < 1.5) it won EVERY view measured (-8..-11%, §35.8 + doc §38).
//
// VTK_METAL_TEST_VOLTRANSPOSE_AXIS=x|y|z forces the orientation for A/B and
// diagnostics regardless of dims.
static int VolumeTransposedAxisDepth(const int dims[3])
{
  if (!VolumeTransposedActive())
    return 0;
  if (const char* a = getenv("VTK_METAL_TEST_VOLTRANSPOSE_AXIS"))
  {
    if (getenv("VTK_METAL_TEST_TR_DUMP") || getenv("VTK_METAL_TEST_TR_BENCH"))
      fprintf(stderr, "[TRPOLICY] axis env '%s' dims %dx%dx%d -> %d\n", a,
        dims[0], dims[1], dims[2],
        a[0] == 'x' || a[0] == 'X' ? 1 : (a[0] == 'y' || a[0] == 'Y' ? 2 : 0));
    switch (a[0])
    {
      case 'x': case 'X': return 1;
      case 'y': case 'Y': return 2;
      default:            return 0;
    }
  }
  if (dims[2] <= dims[0] && dims[2] <= dims[1])
    return 0;                          // z already (tied-)shortest: no-op
  return (dims[0] <= dims[1]) ? 1 : 2; // else the shorter in-plane axis
}

// §28 GPU transpose pass (VTK_METAL_TEST_GPU_TRANSPOSE): when the transposed
// representation is active, skip the CPU blocked x<->z repack (seconds at load
// for ~450 MB volumes) and instead upload the ORIGINAL-layout staging and run
// a one-pass compute kernel that writes the swapped-dims volume texture
// directly from the staging bytes. Same command-queue ordering as the blit it
// replaces, so downstream renders stay correct; the final texture contents are
// byte-identical to the CPU repack path. ON by default; set to 0 to force the
// CPU repack. Re-read per call like VolumeTransposedActive().
static bool VolumeTransposeGPU()
{
  if (const char* v = getenv("VTK_METAL_TEST_GPU_TRANSPOSE"))
    return std::atoi(v) != 0;
  return true;
}

// Composite slab count from VTK_METAL_TEST_NUM_SLABS. Slab tiling is DISABLED
// by default (count 1 = the bit-identical single-pass path). N > 1 splits
// each ray into N ray-length-fraction index ranges (ceil(j*maxSteps/N) ..
// ceil((j+1)*maxSteps/N) for j in [0,N)) and renders N front-to-back passes
// whose partial composites are combined by (ONE, ONE_MINUS_SRC_ALPHA) hardware
// blending. Front-to-back premultiplied `over` is associative, so the result
// equals a single-pass composite up to fp rounding. Depth-sliced marching keeps
// the volume working set cache-resident, which is decisive on the raw (minmax
// off) coarse-sample-distance path where a single full-ray pass exceeds the
// M2/Apple SoC cache and Metal loses to GL by 1.3-1.8x (PERFORMANCE_INVESTIGATION
// "raw-path coarse-SD lag"); with 8 slabs those regimes flip to Metal winning
// (2048x2048/SD4: 88-93ms -> 44ms vs GL 50-53ms, M/GL 0.83-0.89). Reads the env
// var once per process; clamped to [1, 32]. 0 = adaptive, the count is chosen
// per frame (see ResolveNumSlabs). Unset = 1 (single pass).
static int VolumeSlabCount()
{
  static const int count = [] {
    if (const char* v = getenv("VTK_METAL_TEST_NUM_SLABS"))
    {
      const int n = std::atoi(v);
      return (n <= 0) ? 0 : std::min(n, 32);
    }
    return 1;
  }();
  return count;
}

// View-aligned adaptive slab count. Slab tiling only pays off when the volume
// working set per pass exceeds the on-chip cache, which happens when rays sweep
// the texture obliquely. For a near-axis view the single-pass working set is
// already cache-friendly and tiling is pure pass-count overhead (measured
// ~1.9-2x slower at 2048x2048 on coronal/sagittal DICOM views, neutral on
// axial; oblique views are conversely 2.2-2.6x faster WITH 8 slabs — see
// PERFORMANCE_INVESTIGATION.md section on orientation). Pick the count from the
// max |dot| between the volume-space view direction and the volume axes:
// aligned (>= VTK_METAL_TEST_SLAB_ALIGN, default 0.95) -> 1 slab, otherwise
// maxSlabs. Every count composites bit-identically, so this is purely a
// performance trade-off.
static int AdaptiveVolumeSlabCount(const float camVolPos[3], int maxSlabs)
{
  double dx = 0.5 - camVolPos[0];
  double dy = 0.5 - camVolPos[1];
  double dz = 0.5 - camVolPos[2];
  const double len = std::sqrt(dx * dx + dy * dy + dz * dz);
  double align = 0.0;
  if (len > 1e-12)
  {
    align = std::max({ std::abs(dx), std::abs(dy), std::abs(dz) }) / len;
  }
  double threshold = 0.95;
  if (const char* v = getenv("VTK_METAL_TEST_SLAB_ALIGN"))
  {
    threshold = std::atof(v);
  }
  return (align >= threshold) ? 1 : maxSlabs;
}

// Resolve the per-frame slab count: an explicit VTK_METAL_TEST_NUM_SLABS wins
// (0 re-enables the view-aligned adaptive choice); unset defaults to 1, i.e.
// slab tiling disabled (single pass).
static int ResolveNumSlabs(const float camVolPos[3])
{
  const int configured = VolumeSlabCount();
  return (configured > 0) ? configured : AdaptiveVolumeSlabCount(camVolPos, 8);
}

// Dominant view axis (0/1/2 = x/y/z) in normalized volume space: the volume
// axis with the largest |dot| against the view direction (the same
// approximation AdaptiveVolumeSlabCount uses for its alignment). The spatial
// slab planes (SLAB_BENCHMARKS.md §5.2) are perpendicular to this axis, which
// keeps every pass' fetch set a thin flat band of the volume for any view.
// |dot| >= 1/sqrt(3) by construction, so the shader's plane intersections are
// never degenerate.
static int VolumeSlabAxis(const float camVolPos[3])
{
  double d[3] = { 0.5 - camVolPos[0], 0.5 - camVolPos[1], 0.5 - camVolPos[2] };
  int axis = 0;
  if (std::abs(d[1]) > std::abs(d[axis])) axis = 1;
  if (std::abs(d[2]) > std::abs(d[axis])) axis = 2;
  return axis;
}

// Spatial (uniform world-plane) slab tiling selector
// (VTK_METAL_TEST_SLAB_SPATIAL, SLAB_BENCHMARKS.md §5.2): when set, the slab
// passes split the ray by uniform planes perpendicular to the dominant view
// axis instead of by per-fragment sample-index fractions. Reads the env var
// once per process; unset = 0 (ray-fraction split). Non-integer values (e.g.
// 0.6) are debug modes that also enable the spatial split.
static float VolumeSlabSpatial()
{
  static const float spatial = [] {
    if (const char* v = getenv("VTK_METAL_TEST_SLAB_SPATIAL"))
      return (float)std::atof(v);
    return 0.0f;
  }();
  return spatial;
}

// Fixed uniform iteration count for the variant-4 non-divergent march
// (VTK_METAL_TEST_MARCH_STEPS). 0 = compute a frame-max bound from the camera
// instead. Reads the env var once per process.
static int VolumeMarchSteps()
{
  static const int steps = [] {
    if (const char* v = getenv("VTK_METAL_TEST_MARCH_STEPS"))
      return std::atoi(v);
    return 0;
  }();
  return steps;
}

// Frame-max / frame-average ray-box chord in physical mm for the current camera
// (PERFORMANCE_INVESTIGATION.md section 4.2). Replicates the shader's
// intersectBox in [0,1] volume space: the maximum chord through the volume box
// from the camera is bounded by the chords of the rays through the 8 box
// corners, the 6 face centers and the box center (an over-bound that also
// covers in-between directions), clamped to the box diagonal. For parallel
// projection every ray shares the same direction, so the chord is exactly the
// box width along the projection direction. Returns the max chord (used as the
// variant-4 uniform bound) and, via meanChordMM, the mean of the sampled chords
// (used as the variant-5 uniform main-loop bound).
static double ComputeMaxChordMM(double s0, double s1, double s2, const float cam[3],
  bool parallel, const float projDir[3], double sampleDistanceMM,
  double* meanChordMM = nullptr)
{
  const double diagMM = std::sqrt(s0 * s0 + s1 * s1 + s2 * s2);
  if (meanChordMM)
    *meanChordMM = diagMM;

  if (parallel)
  {
    // Box width along the (unit) projection direction: sum_i Size_i |d_i|.
    const double d0 = projDir[0], d1 = projDir[1], d2 = projDir[2];
    double widthMM = s0 * std::abs(d0) + s1 * std::abs(d1) + s2 * std::abs(d2);
    return std::min(widthMM, diagMM);
  }

  // Rays through the 8 corners, 6 face centers and box center (volume space).
  float pts[15][3] = {
    {0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f},
    {1.f, 1.f, 0.f}, {1.f, 0.f, 1.f}, {0.f, 1.f, 1.f}, {1.f, 1.f, 1.f},
    {0.5f, 0.5f, 0.f}, {0.5f, 0.5f, 1.f}, {0.5f, 0.f, 0.5f}, {0.5f, 1.f, 0.5f},
    {0.f, 0.5f, 0.5f}, {1.f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f},
  };
  double maxChordMM = 0.0;
  double sumChordMM = 0.0;
  int nChord = 0;
  for (int p = 0; p < 15; ++p)
  {
    double dx = pts[p][0] - cam[0];
    double dy = pts[p][1] - cam[1];
    double dz = pts[p][2] - cam[2];
    double len = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (len < 1e-9)
      continue;
    dx /= len; dy /= len; dz /= len;

    // intersectBox(cam, dir, 0, 1) with safeRecip (1e-20 epsilon).
    const double eps = 1e-20;
    double invx = (std::abs(dx) < eps) ? (dx < 0 ? -1.0 / eps : 1.0 / eps) : (1.0 / dx);
    double invy = (std::abs(dy) < eps) ? (dy < 0 ? -1.0 / eps : 1.0 / eps) : (1.0 / dy);
    double invz = (std::abs(dz) < eps) ? (dz < 0 ? -1.0 / eps : 1.0 / eps) : (1.0 / dz);
    double tbotx = invx * (0.0 - cam[0]), ttopx = invx * (1.0 - cam[0]);
    double tboty = invy * (0.0 - cam[1]), ttopy = invy * (1.0 - cam[1]);
    double tbotz = invz * (0.0 - cam[2]), ttopz = invz * (1.0 - cam[2]);
    double tminx = std::min(ttopx, tbotx), tmaxx = std::max(ttopx, tbotx);
    double tminy = std::min(ttopy, tboty), tmaxy = std::max(ttopy, tboty);
    double tminz = std::min(ttopz, tbotz), tmaxz = std::max(ttopz, tbotz);
    double tN = std::max(std::max(tminx, tminy), tminz);
    double tF = std::min(std::min(tmaxx, tmaxy), tmaxz);
    if (tN > tF || tF < 0.0)
      continue;
    double tStart = std::max(tN, 0.0);
    if (tStart >= tF)
      continue;
    double chordVol = tF - tStart;
    // Physical chord = chordVol * physPerNorm = chordVol * |dir * boundsSize|.
    double physPerNorm = std::sqrt((dx * s0) * (dx * s0) +
      (dy * s1) * (dy * s1) + (dz * s2) * (dz * s2));
    double chordMM = chordVol * physPerNorm;
    if (chordMM > maxChordMM)
      maxChordMM = chordMM;
    sumChordMM += chordMM;
    nChord++;
    if (getenv("VTK_METAL_TEST_MARCH_DEBUG"))
      fprintf(stderr, "  [march] pt=%d chordMM=%.2f\n", p, chordMM);
  }
  // Clamp to the box diagonal (absolute upper bound for any chord) and to the
  // sample distance ratio so a degenerate camera never yields a huge frame-max.
  maxChordMM = std::min(maxChordMM, diagMM);
  if (sampleDistanceMM > 0.0)
    maxChordMM = std::min(maxChordMM, diagMM);
  if (meanChordMM)
    *meanChordMM = (nChord > 0) ? std::min(sumChordMM / nChord, diagMM) : diagMM;
  return maxChordMM;
}

// Whether the independent multi-component shader path is active for this
// render. Mirrors the shader's (former) runtime useIndependentPath condition:
// independent components, without the 2D transfer-function or label-map
// fallbacks that always take the single-component path. Baking this into the
// pipeline lets single-component renders compile the per-sample arrays and
// branches out entirely.
static bool VolumeFeatureIndependentPath(
  const VolumeMapperUniforms& uniforms, int featureMask)
{
  const bool maskActive = (featureMask & VolumeFeature_Mask) != 0;
  return uniforms.UseIndependentComponents > 0.5f &&
    uniforms.UseTransfer2D <= 0.5f &&
    !(maskActive && uniforms.UseMask > 0.5f && uniforms.LabelMapNumLabels > 0.0f);
}

// Per-block data for volume rendering — must match Metal PerBlockData struct
struct PerBlockData {
  float VolumeBoundsMin[4]; // 0..15
  float VolumeBoundsMax[4]; // 16..31
  float TextureBoundsMin[4]; // 32..47
  float TextureBoundsMax[4]; // 48..63
  float GradientStep[4];    // 64..79  (xyz + pad)
  float MinMaxInfo[4];      // 80..95  (useMinMax, dimX, dimY, dimZ)
  float SlabInfo[4];        // 96..111 (slabIndex, slabCount, slabAxis, spatialMode)
};

static_assert(sizeof(PerBlockData) == 112,
  "PerBlockData must be 112 bytes to match Metal shader struct");

// Must match Metal GridTraversalUniforms struct
struct GridTraversalUniforms {
  int32_t GridDimsX;
  int32_t GridDimsY;
  int32_t GridDimsZ;
  int32_t _pad;
};
static_assert(sizeof(GridTraversalUniforms) == 16,
  "GridTraversalUniforms must be 16 bytes to match Metal shader struct");

// Must match Metal VolumeConvertUniforms struct (Phase 7: GPU data type conversion)
struct VolumeConvertUniforms {
  uint32_t dimX, dimY, dimZ;
  uint32_t numComponents;
  uint32_t outputComponents;
  uint32_t _pad;
};
static_assert(sizeof(VolumeConvertUniforms) == 6 * 4, "VolumeConvertUniforms size must match Metal struct");

// Must match Metal MinMaxComputeUniforms struct (Phase 5: GPU min-max)
struct MinMaxComputeUniforms {
  uint32_t mmDimX, mmDimY, mmDimZ;
  uint32_t volDimX, volDimY, volDimZ;
  float ds;
  float scalarMin;
  float scalarScale;
  float _pad;
  uint32_t opacityPrefix[257];
  int32_t volTransposed;
  float opacityLut[256];   // §33.2 item 2 (VTK_METAL_TEST_MM_EPS)
  float mmEps;             // emptiness threshold (0 = exact semantics)
  uint32_t _pad2[3];
};

static_assert(sizeof(MinMaxComputeUniforms) ==
    4*6 + 4 + 4 + 4 + 4 + 257*4 + 4 + 256*4 + 4 + 3*4,
  "MinMaxComputeUniforms size must match Metal struct");

namespace
{
inline uint16_t FloatToHalf(float f)
{
#if defined(__aarch64__) || defined(__ARM_ARCH_7A__)
  // Hardware conversion via __fp16 — handles all IEEE 754 cases correctly
  // (round-to-nearest-even, NaN payload, signed zero, denormals).
  __fp16 h = static_cast<__fp16>(f);
  uint16_t bits;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
#else
  // Software fallback with proper IEEE 754 round-to-nearest-even
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  uint32_t sign = (bits >> 16) & 0x8000;
  int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
  uint32_t mantissa = bits & 0x7FFFFF;

  // NaN / Infinity
  if ((bits & 0x7F800000) == 0x7F800000)
  {
    if ((bits & 0x7FFFFF) == 0)
      return static_cast<uint16_t>(sign | 0x7C00);
    // NaN — preserve payload; ensure at least one mantissa bit is set
    uint16_t halfMantissa = (mantissa >> 13) & 0x03FF;
    return static_cast<uint16_t>(sign | 0x7C00 | halfMantissa | (halfMantissa == 0 ? 1 : 0));
  }

  // Denormals / underflow
  if (exponent <= 0)
  {
    if (exponent < -10)
      return static_cast<uint16_t>(sign);
    mantissa = (mantissa | 0x800000) >> (1 - exponent);
    uint32_t roundBit = (mantissa >> 12) & 1;
    uint32_t sticky = mantissa & 0xFFF;
    mantissa >>= 13;
    if (roundBit && (sticky > 0 || (mantissa & 1)))
      ++mantissa;
    return static_cast<uint16_t>(sign | (mantissa > 0x3FF ? 0x7C00 : mantissa));
  }

  // Overflow to infinity
  if (exponent > 30)
    return static_cast<uint16_t>(sign | 0x7C00);

  // Normal number with round-to-nearest-even
  uint32_t halfMantissa = mantissa >> 13;
  uint32_t roundBit = (mantissa >> 12) & 1;
  uint32_t sticky = mantissa & 0xFFF;

  if (roundBit && (sticky > 0 || (halfMantissa & 1)))
  {
    ++halfMantissa;
    if (halfMantissa > 0x3FF)
    {
      halfMantissa = 0;
      ++exponent;
      if (exponent > 30)
        return static_cast<uint16_t>(sign | 0x7C00);
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | halfMantissa);
#endif
}

// Returns true when half-float can safely represent the full scalar range.
inline bool HalfRangeIsSafe(double r0, double r1)
{
  const double halfMax = 65504.0;
  return std::isfinite(r0) && std::isfinite(r1) &&
    r0 >= -halfMax && r1 <= halfMax;
}

//------------------------------------------------------------------------------
// Generic scalar conversion loop: source type -> writer callback
template <typename SrcType, typename Writer>
void ConvertScalarsGeneric(
  const SrcType* src,
  vtkIdType numTuples,
  int numComp,
  int outComp,
  Writer&& write)
{
  vtkSMPTools::For(0, numTuples, [&](vtkIdType b, vtkIdType e) {
    for (vtkIdType i = b; i < e; ++i)
    {
      for (int c = 0; c < outComp; ++c)
      {
        float value = (c < numComp)
          ? static_cast<float>(src[i * numComp + c])
          : 0.0f;

        write(i, c, value);
      }
    }
  });
}

// Templated data conversion: source type -> half
template <typename SrcType>
void ConvertToHalf(
  const SrcType* src,
  uint16_t* dst,
  vtkIdType numTuples,
  int numComp,
  int outComp)
{
  ConvertScalarsGeneric(src, numTuples, numComp, outComp,
    [&](vtkIdType i, int c, float value) {
      dst[i * outComp + c] = FloatToHalf(value);
    });
}

// Templated data conversion: source type -> float
template <typename SrcType>
void ConvertToFloat(
  const SrcType* src,
  float* dst,
  vtkIdType numTuples,
  int numComp,
  int outComp)
{
  ConvertScalarsGeneric(src, numTuples, numComp, outComp,
    [&](vtkIdType i, int c, float value) {
      dst[i * outComp + c] = value;
    });
}

//------------------------------------------------------------------------------
// Templated 3-to-4 component expander with a constant alpha pad.
template <typename T>
void Expand3To4(
  const T* src,
  T* dst,
  vtkIdType numTuples,
  T alpha)
{
  vtkSMPTools::For(0, numTuples, [&](vtkIdType b, vtkIdType e) {
    for (vtkIdType i = b; i < e; ++i)
    {
      dst[i * 4 + 0] = src[i * 3 + 0];
      dst[i * 4 + 1] = src[i * 3 + 1];
      dst[i * 4 + 2] = src[i * 3 + 2];
      dst[i * 4 + 3] = alpha;
    }
  });
}

//------------------------------------------------------------------------------
// Shared helper: convert volume data from various data types to half/float.
// Writes outputComponents interleaved components per tuple (3-component sources
// are expanded to 4 with a zero pad component).
void ConvertVolumeData(const void* src, int dataType, int numComponents,
  vtkIdType numTuples, void* dst, bool useHalf,
  int outputComponents, vtkDataArray* scalars)
{
  if (useHalf)
  {
      // Accelerate fast path for single-component float -> half.
      // Use the original src as the read-only source buffer to avoid
      // overlapping source/destination in vImageConvert_PlanarFtoPlanar16F.
      if (dataType == VTK_FLOAT && numComponents == 1)
      {
        vImage_Buffer vSrc = { const_cast<void*>(src), 1,
          static_cast<vImagePixelCount>(numTuples),
          static_cast<vImagePixelCount>(numTuples * sizeof(float)) };
        vImage_Buffer vDst = { dst, 1,
          static_cast<vImagePixelCount>(numTuples),
          static_cast<vImagePixelCount>(numTuples * sizeof(uint16_t)) };
        vImageConvert_PlanarFtoPlanar16F(&vSrc, &vDst, 0);
        return;
      }
    uint16_t* h = static_cast<uint16_t*>(dst);
    switch (dataType)
    {
      case VTK_SHORT:        ConvertToHalf(static_cast<const short*>(src), h, numTuples, numComponents, outputComponents); break;
      case VTK_INT:          ConvertToHalf(static_cast<const int*>(src), h, numTuples, numComponents, outputComponents); break;
      case VTK_UNSIGNED_INT: ConvertToHalf(static_cast<const unsigned int*>(src), h, numTuples, numComponents, outputComponents); break;
      case VTK_DOUBLE:       ConvertToHalf(static_cast<const double*>(src), h, numTuples, numComponents, outputComponents); break;
      case VTK_UNSIGNED_CHAR: ConvertToHalf(static_cast<const unsigned char*>(src), h, numTuples, numComponents, outputComponents); break;
      case VTK_UNSIGNED_SHORT: ConvertToHalf(static_cast<const unsigned short*>(src), h, numTuples, numComponents, outputComponents); break;
      case VTK_FLOAT:        ConvertToHalf(static_cast<const float*>(src), h, numTuples, numComponents, outputComponents); break;
      default:
      {
        vtkSMPTools::For(0, numTuples, [&](vtkIdType b, vtkIdType e) {
          for (vtkIdType i = b; i < e; ++i)
            for (int c = 0; c < outputComponents; ++c)
              h[i * outputComponents + c] = (c < numComponents)
                ? FloatToHalf(static_cast<float>(scalars->GetComponent(i, c)))
                : FloatToHalf(0.0f);
        });
        break;
      }
    }
  }
  else
  {
    float* f = static_cast<float*>(dst);
    switch (dataType)
    {
      case VTK_SHORT:        ConvertToFloat(static_cast<const short*>(src), f, numTuples, numComponents, outputComponents); break;
      case VTK_INT:          ConvertToFloat(static_cast<const int*>(src), f, numTuples, numComponents, outputComponents); break;
      case VTK_UNSIGNED_INT: ConvertToFloat(static_cast<const unsigned int*>(src), f, numTuples, numComponents, outputComponents); break;
      case VTK_FLOAT:        ConvertToFloat(static_cast<const float*>(src), f, numTuples, numComponents, outputComponents); break;
      case VTK_UNSIGNED_CHAR: ConvertToFloat(static_cast<const unsigned char*>(src), f, numTuples, numComponents, outputComponents); break;
      case VTK_UNSIGNED_SHORT: ConvertToFloat(static_cast<const unsigned short*>(src), f, numTuples, numComponents, outputComponents); break;
      case VTK_DOUBLE:       ConvertToFloat(static_cast<const double*>(src), f, numTuples, numComponents, outputComponents); break;
      default:
      {
        vtkSMPTools::For(0, numTuples, [&](vtkIdType b, vtkIdType e) {
          for (vtkIdType i = b; i < e; ++i)
            for (int c = 0; c < outputComponents; ++c)
              f[i * outputComponents + c] = (c < numComponents)
                ? static_cast<float>(scalars->GetComponent(i, c))
                : 0.0f;
        });
        break;
      }
    }
  }
}

//------------------------------------------------------------------------------
struct VolumeFormat
{
  MTLPixelFormat Format = MTLPixelFormatInvalid;
  int BytesPerComponent = 0;
  float NormalizationFactor = 1.0f;
  bool NeedsConversion = false;
};

static VolumeFormat ChooseVolumeFormat(
  int dataType,
  int numComponents,
  const double scalarRange[2],
  bool preferHalf)
{
  VolumeFormat fmt;
  const int componentsForFormat = (numComponents == 3) ? 4 : numComponents;
  const bool useHalf =
    preferHalf && HalfRangeIsSafe(scalarRange[0], scalarRange[1]);

  switch (dataType)
  {
    case VTK_FLOAT:
      if (useHalf)
      {
        fmt.BytesPerComponent = 2;
        fmt.NeedsConversion = true;
        fmt.NormalizationFactor = 1.0f;
        switch (componentsForFormat)
        {
          case 1: fmt.Format = MTLPixelFormatR16Float; break;
          case 2: fmt.Format = MTLPixelFormatRG16Float; break;
          default: fmt.Format = MTLPixelFormatRGBA16Float; break;
        }
      }
      else
      {
        fmt.BytesPerComponent = 4;
        fmt.NeedsConversion = false;
        fmt.NormalizationFactor = 1.0f;
        switch (componentsForFormat)
        {
          case 1: fmt.Format = MTLPixelFormatR32Float; break;
          case 2: fmt.Format = MTLPixelFormatRG32Float; break;
          default: fmt.Format = MTLPixelFormatRGBA32Float; break;
        }
      }
      break;

    case VTK_UNSIGNED_CHAR:
      fmt.BytesPerComponent = 1;
      fmt.NeedsConversion = false;
      fmt.NormalizationFactor = 255.0f;
      switch (componentsForFormat)
      {
        case 1: fmt.Format = MTLPixelFormatR8Unorm; break;
        case 2: fmt.Format = MTLPixelFormatRG8Unorm; break;
        default: fmt.Format = MTLPixelFormatRGBA8Unorm; break;
      }
      break;

    case VTK_UNSIGNED_SHORT:
      if (scalarRange[0] >= 0.0 && scalarRange[1] <= 255.0)
      {
        fmt.BytesPerComponent = 1;
        fmt.NeedsConversion = false;
        fmt.NormalizationFactor = 255.0f;
        switch (componentsForFormat)
        {
          case 1: fmt.Format = MTLPixelFormatR8Unorm; break;
          case 2: fmt.Format = MTLPixelFormatRG8Unorm; break;
          default: fmt.Format = MTLPixelFormatRGBA8Unorm; break;
        }
      }
      else
      {
        fmt.BytesPerComponent = 2;
        fmt.NeedsConversion = false;
        fmt.NormalizationFactor = 65535.0f;
        switch (componentsForFormat)
        {
          case 1: fmt.Format = MTLPixelFormatR16Unorm; break;
          case 2: fmt.Format = MTLPixelFormatRG16Unorm; break;
          default: fmt.Format = MTLPixelFormatRGBA16Unorm; break;
        }
      }
      break;

    case VTK_SHORT:
      // Signed 16-bit data stored as signed-normalized (OpenGL R16_SNORM
      // parity): the raw short bits upload directly, the GPU normalizes each
      // value v to v/32767, and the shader un-normalizes back to the [0,1]
      // scalar range via scalarMin/scalarMax. Unlike half-float storage this
      // keeps every short value exact -- half loses the low bit above 2048
      // (ulp 2), which rounds odd label values down one entry and shifts the
      // transfer-function sample (e.g. label 4001 -> 4000 -> transparent).
      fmt.BytesPerComponent = 2;
      fmt.NeedsConversion = false;
      fmt.NormalizationFactor = 32768.0f;
      switch (componentsForFormat)
      {
        case 1: fmt.Format = MTLPixelFormatR16Snorm; break;
        case 2: fmt.Format = MTLPixelFormatRG16Snorm; break;
        default: fmt.Format = MTLPixelFormatRGBA16Snorm; break;
      }
      break;

    default:
      if (useHalf)
      {
        fmt.BytesPerComponent = 2;
        fmt.NeedsConversion = true;
        fmt.NormalizationFactor = 1.0f;
        switch (componentsForFormat)
        {
          case 1: fmt.Format = MTLPixelFormatR16Float; break;
          case 2: fmt.Format = MTLPixelFormatRG16Float; break;
          default: fmt.Format = MTLPixelFormatRGBA16Float; break;
        }
      }
      else
      {
        fmt.BytesPerComponent = 4;
        fmt.NeedsConversion = true;
        fmt.NormalizationFactor = 1.0f;
        switch (componentsForFormat)
        {
          case 1: fmt.Format = MTLPixelFormatR32Float; break;
          case 2: fmt.Format = MTLPixelFormatRG32Float; break;
          default: fmt.Format = MTLPixelFormatRGBA32Float; break;
        }
      }
      break;
  }

  return fmt;
}

// Release a Metal object held as a void* member (MRC helper).
// Uses -release rather than CFRelease for proper Objective-C semantics.
inline void ReleaseMetalObject(void*& obj)
{
  if (obj)
  {
    [(__bridge id)obj release];
    obj = nullptr;
  }
}

// Takes ownership of a +1 Metal object into a void* member slot.
// Releases the previous occupant if any.
inline void AssignMetalObject(void*& slot, id obj)
{
  if (slot == (__bridge void*)obj)
  {
    return;
  }
  if (slot)
  {
    [(__bridge id)slot release];
  }
  slot = (__bridge void*)obj;
}

// Retaining assignment: releases the previous occupant, then retains and
// stores the new object. Used when the +1 is owned elsewhere (e.g. cache).
inline void AssignRetainedMetalObject(void*& slot, id obj)
{
  if (slot == (__bridge void*)obj)
  {
    return;
  }
  if (slot)
  {
    [(__bridge id)slot release];
  }
  slot = (__bridge void*)[obj retain];
}

//------------------------------------------------------------------------------
// Helper: create a MTLTextureType3D texture with mipmapLevelCount = 1.
// Returns nil on failure; caller owns the +1 retain count.
static id<MTLTexture> NewTexture3D(
  id<MTLDevice> device,
  MTLPixelFormat format,
  NSUInteger width,
  NSUInteger height,
  NSUInteger depth,
  MTLTextureUsage usage,
  MTLStorageMode storage,
  bool allowGPUOptimizedContents = true,
  NSUInteger mipmapLevels = 1)
{
  MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
  desc.textureType = MTLTextureType3D;
  desc.pixelFormat = format;
  desc.width = width;
  desc.height = height;
  desc.depth = depth;
  desc.mipmapLevelCount = mipmapLevels;
  desc.usage = usage;
  desc.storageMode = storage;
  desc.allowGPUOptimizedContents = allowGPUOptimizedContents;
  id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
  if (getenv("VTK_METAL_TEST_TR_DUMP"))
    fprintf(stderr, "[TEXCRE] %dx%dx%d fmt=%u usage=%u\n",
      (int)width, (int)height, (int)depth, (unsigned)format, (unsigned)usage);
  [desc release];
  return tex;
}

//------------------------------------------------------------------------------
// Helper: create a MTLTextureType2D texture with mipmapLevelCount = 1.
static id<MTLTexture> NewTexture2D(
  id<MTLDevice> device,
  MTLPixelFormat format,
  NSUInteger width,
  NSUInteger height,
  MTLTextureUsage usage,
  MTLStorageMode storage)
{
  MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
  desc.textureType = MTLTextureType2D;
  desc.pixelFormat = format;
  desc.width = width;
  desc.height = height;
  desc.mipmapLevelCount = 1;
  desc.usage = usage;
  desc.storageMode = storage;
  id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
  [desc release];
  return tex;
}

//------------------------------------------------------------------------------
// Helper: upload R8 data (one byte per voxel) to a 3D texture region [0..x) x [0..y) x [0..z).
static void UploadR8Volume3D(
  id<MTLTexture> tex,
  const uint8_t* data,
  int x, int y, int z)
{
  MTLRegion region = MTLRegionMake3D(0, 0, 0, x, y, z);
  NSUInteger bytesPerRow = static_cast<NSUInteger>(x) * sizeof(uint8_t);
  NSUInteger bytesPerImage = bytesPerRow * static_cast<NSUInteger>(y);
    [tex replaceRegion:region
            mipmapLevel:0
                  slice:0
              withBytes:data
            bytesPerRow:bytesPerRow
          bytesPerImage:bytesPerImage];
}

//------------------------------------------------------------------------------
struct OccupancyGrid
{
  std::vector<uint8_t> Data;
  int Dims[3] = { 0, 0, 0 };
};

//------------------------------------------------------------------------------
static bool ScalarRangeTouchesOpacity(
  float cellMin,
  float cellMax,
  const double opacityTable[256],
  double scalarMin,
  double scalarRangeRecip255)
{
  if (cellMin > cellMax)
  {
    return false;
  }

  int idxMin = std::max(0,
    std::min(255, static_cast<int>((cellMin - scalarMin) * scalarRangeRecip255)));

  int idxMax = std::max(0,
    std::min(255, static_cast<int>((cellMax - scalarMin) * scalarRangeRecip255)));

  for (int i = idxMin; i <= idxMax; ++i)
  {
    if (opacityTable[i] > 0.0)
    {
      return true;
    }
  }

  return false;
}

//------------------------------------------------------------------------------
static std::vector<uint8_t> DilateOccupancy3D(
  const std::vector<uint8_t>& raw,
  int dimX,
  int dimY,
  int dimZ)
{
  std::vector<uint8_t> out(raw.size(), 255);

  vtkSMPTools::For(0, static_cast<vtkIdType>(raw.size()),
    [&](vtkIdType begin, vtkIdType end) {
      for (vtkIdType cellIdx = begin; cellIdx < end; ++cellIdx)
      {
        const int gx = static_cast<int>(cellIdx % dimX);
        const int gy = static_cast<int>((cellIdx / dimX) % dimY);
        const int gz = static_cast<int>(cellIdx / (dimX * dimY));

        const int x0 = std::max(0, gx - 1);
        const int x1 = std::min(dimX - 1, gx + 1);
        const int y0 = std::max(0, gy - 1);
        const int y1 = std::min(dimY - 1, gy + 1);
        const int z0 = std::max(0, gz - 1);
        const int z1 = std::min(dimZ - 1, gz + 1);

        bool solid = false;

        for (int nz = z0; nz <= z1 && !solid; ++nz)
        {
          for (int ny = y0; ny <= y1 && !solid; ++ny)
          {
            for (int nx = x0; nx <= x1 && !solid; ++nx)
            {
              if (raw[(nz * dimY + ny) * dimX + nx] == 0)
              {
                solid = true;
              }
            }
          }
        }

        out[cellIdx] = solid ? 0 : 255;
      }
    });

  return out;
}

//------------------------------------------------------------------------------
// The min-max occupancy grid is a downsample of the volume by DS voxels per
// cell. The empty-space skip advances whole macrocell cells, so the optimal
// DS tracks the sample distance: dense sampling (small step) wants fine cells
// to keep the re-marched boundary shell thin, while coarse sampling (large
// step) wants coarser cells so a single skip advances several sample steps.
// Measured on the DICOM study (Metal, 400x400): DS 2 best at sd <= 1, DS 4
// best at sd >= 2. Every DS is output-identical (the emptiness test is exact
// for U8 data), so this is a pure pipeline tuning knob. The finer DS=2 grid
// is restricted to the GPU-computed lattice (UseGPUMinMax): there its precise
// per-sample skip pays off. The CPU-computed lattice keeps DS=4, matching the
// pre-adaptive baseline so the CPU path does not slow the march down.
static int ComputeMacrocellDownsample(double sampleDistance, bool useGPUMinMax)
{
  // TEMP-DIAG (VTK_METAL_TEST_MM_DS): force the macrocell downsample factor
  // to A/B the DS=2-vs-4 crossover at fine sample distance post-transpose
  // (HARNESS_VS_APP_GAP §33). Investigation-only.
  if (const char* e = getenv("VTK_METAL_TEST_MM_DS"))
    return std::atoi(e);
  return (useGPUMinMax && sampleDistance < 1.5) ? 2 : 4;
}

// Two-level occupancy summary, DEFAULT-ON whenever the GPU minmax lattice is
// active (HARNESS_VS_APP_GAP §37.5). Builds a coarse R8 texture marking
// whole-block all-empty regions of the DILATED macrocell lattice so the
// fragment walk leaps multiple cells per lattice fetch. Block-leap landings
// differ from the per-cell walk chain into the accepted +-1-step fp-landing
// class at EVERY DS tier — the earlier "byte-identical" claim did not survive
// pixel measurement (§37.6: ~0.2% of pixels, mean ~1 LSB, rare outliers) —
// in exchange for -13% frame time at SD4 and -28% at SD0.5 (axz). 
// VTK_METAL_TEST_MM_BLOCKS=0 opts back out.
static int VolumeMinMaxBlockSize(double sampleDistance)
{
  // §37.18/§37.23 (VTK_METAL_TEST_MM_BLOCKSIZE): block-summary edge in fine
  // cells, TIERED like the lattice downsample itself:
  //   fine tier (sd < 1.5, DS=2 lattice): 16 — at half-voxel steps each ray
  //     crosses ~2x the blocks, so halving the leap-event count wins on every
  //     view measured (-22.9% axz / -21.3% obl vs raw; BS8 -18.8/-15.0).
  //   coarse tier (sd >= 1.5, DS=4 lattice): 8 — larger oblique/az45 wins
  //     outweigh BS16's axis-z relief (orbit-integrated -6.4% vs -3.6%,
  //     §37.22 table); BS16 remains the documented axis-chord remedy here.
  // Explicit env overrides both tiers ({4,8,16,32}; powers of two only so
  // the shader's exact fp-reciprocal index math stays exact).
  if (const char* e = getenv("VTK_METAL_TEST_MM_BLOCKSIZE"))
  {
    const int v = std::atoi(e);
    if (v == 4 || v == 8 || v == 16 || v == 32) return v;
  }
  return sampleDistance < 1.5 ? 16 : 8;
}

// Unified gate for the two-level occupancy summary. The former fine-SD-only
// gate (sampleDistance < 1.5) was removed: its rationale — "~+1 ms on the SD4
// obliques" — was refuted when re-measured on a healthy machine (§37.1/§37.3,
// 2026-08-23): blocks win or tie everywhere at both jitter settings. The
// sampleDistance parameter is kept for call-site stability.
static bool VolumeMinMaxBlocksWanted(bool gpuMinMax, double sampleDistance)
{
  (void)sampleDistance;
  if (!gpuMinMax)
    return false;
  if (const char* v = getenv("VTK_METAL_TEST_MM_BLOCKS"))
    return std::atoi(v) != 0;
  return true;
}

// §35.5 headroom A/B (VTK_METAL_TEST_MM_SUPER): third occupancy level —
// whole 8³-block groups of the block summary that are all-empty. Requires the
// blocks level to be active; investigation-only.
static bool VolumeMinMaxSuperWanted(bool gpuMinMax, double sampleDistance)
{
  if (const char* v = getenv("VTK_METAL_TEST_MM_SUPER"))
    return std::atoi(v) != 0 && VolumeMinMaxBlocksWanted(gpuMinMax, sampleDistance);
  return false;
}

// §35.14 async segment pre-pass gate (VTK_METAL_TEST_MM_SEG): value-parsed so
// "0" stays OFF (bare-getenv diagnostics treat presence as ON). Requires the
// GPU minmax lattice; per-frame readiness is checked at the encode sites.
static bool VolumeSegWanted()
{
  if (const char* v = getenv("VTK_METAL_TEST_MM_SEG"))
    return std::atoi(v) != 0;
  return false;
}

// §38 TF-adaptive opacity-saturation exit (VTK_METAL_TEST_EXIT_THETA): value-
// parsed like MM_SEG — absent/0/invalid = OFF (legacy 8-bit latch threshold);
// a value in (0,1] terminates the march at that accumulated opacity via
// fc_exitTheta specialization. Static per-frame constant: no per-ray or
// per-view branching (§25.5). Diagnostic/investigation knob, default OFF.
static float VolumeExitTheta()
{
  if (const char* v = getenv("VTK_METAL_TEST_EXIT_THETA"))
  {
    const float f = static_cast<float>(std::atof(v));
    if (f > 0.0f && f <= 1.0f)
      return f;
  }
  return 0.0f;
}
// TEMP-DIAG §35.14: run the full pre-pass machinery but keep the march on
// the legacy-preamble pipeline (fc_segHop=false) — isolates the cost of the
// consume code path from the atlas/build/offscreen/blit machinery.
static bool VolumeSegConsumeSuppressed()
{
  return getenv("VTK_METAL_TEST_MM_SEG_NOCONSUME") != nullptr;
}

// §38.6 / §36.4 Design B — Compute Marcher & Ray-Binned Marching gates
static bool VolumeComputeMarchWanted()
{
  if (const char* v = getenv("VTK_METAL_TEST_COMPUTE_MARCH"))
    return std::atoi(v) != 0;
  return false;
}

static bool VolumeRayBinnedWanted()
{
  if (const char* v = getenv("VTK_METAL_TEST_RAY_BINNED"))
    return std::atoi(v) != 0;
  return false;
}

// §38.6 compute-march probes (TEMP-DIAG): dispatch floor + forced march
// length + threadgroup shape. All default-off / no-op.
// Must match the MSL layout exactly: struct ComputeMarchControl { uint x4 }.
struct ComputeMarchControl
{
  uint32_t floorMode;
  uint32_t stepCap;
  uint32_t synthMode;
  uint32_t batchOverride;
};
static bool VolumeComputeMarchFloor()
{
  return getenv("VTK_METAL_TEST_CM_FLOOR") != nullptr;
}

// CM_SYNTH: rebuild ray setup in-kernel (no atlas raster pass at all).
static bool VolumeComputeMarchSynth()
{
  if (const char* v = getenv("VTK_METAL_TEST_CM_SYNTH"))
    return std::atoi(v) != 0;
  return false;
}

static int VolumeComputeMarchStepCap()
{
  if (const char* v = getenv("VTK_METAL_TEST_CM_FSTEPS"))
    return std::atoi(v);
  return 0;
}

// Fragment compile-time batch (VTK_METAL_TEST_FRAG_BATCH): mirrors the
// compute fc_cmBatch trick — compile the ladder at a fixed width so dead
// rungs and their registers are removed at PSO creation. 0 = runtime path.
static int VolumeFragBatch()
{
  if (const char* v = getenv("VTK_METAL_TEST_FRAG_BATCH"))
    return std::max(0, std::min(48, std::atoi(v)));
  return 0;
}

// TEMP-DIAG: compute-only unroll-batch override (register-pressure probe;
// fragment ladder keeps its own MaxBatchWidth).
static int VolumeComputeMarchBatch()
{
  if (const char* v = getenv("VTK_METAL_TEST_CM_BATCH"))
    return std::atoi(v);
  return 0;
}

static void VolumeComputeMarchTG(int& tgw, int& tgh)
{
  tgw = 8; tgh = 8;
  if (const char* v = getenv("VTK_METAL_TEST_CM_TG"))
    sscanf(v, "%dx%d", &tgw, &tgh);
}

// Floor decomposition: skip the atlas raster pass and/or the march encoder.
static bool VolumeComputeMarchNoAtlas()
{
  return getenv("VTK_METAL_TEST_CM_NOATLAS") != nullptr;
}
static bool VolumeComputeMarchNoMarch()
{
  return getenv("VTK_METAL_TEST_CM_NOMARCH") != nullptr;
}
// TEMP-DIAG: skip exposing the result to Phase 3b entirely (isolates blit
// share of the frame floor). Leaves the volume uncomposited — timing only.
static bool VolumeComputeMarchNoBlit()
{
  return getenv("VTK_METAL_TEST_CM_NOBLIT") != nullptr;
}

//------------------------------------------------------------------------------
// Root-caused platform hazard (see HARNESS_VS_APP_GAP.md §38.8): Metal command
// queues are handed out round-robin from two global firmware slots; one slot
// adds a fixed ~7 ms completion latency (≈3.4 ms pipelined throughput cost)
// to COMPUTE command buffers for the queue's entire lifetime. Render/blit
// submissions are unaffected. Slots stick per queue, so probing candidates at
// startup and keeping a fast one makes the compute march immune.
static id<MTLCommandQueue> ProbeAndSelectFastQueue(id<MTLDevice> device,
  int candidates)
{
  // Trivial kernel source compiled once here; independent of shader library.
  NSString* src = @R"(#include <metal_stdlib>
using namespace metal;
kernel void vtk_queue_probe() {
}
)";
  NSError* err = nil;
  id<MTLLibrary> srcLib = [device newLibraryWithSource:src options:nil error:&err];
  if (!srcLib) return nil;
  id<MTLFunction> fn = [srcLib newFunctionWithName:@"vtk_queue_probe"];
  if (!fn) return nil;
  id<MTLComputePipelineState> ps =
    [device newComputePipelineStateWithFunction:fn error:&err];
  if (!ps) return nil;

  id<MTLCommandQueue> bestQ = nil;
  double bestMs = 1e9;
  for (int i = 0; i < candidates; ++i)
  {
    id<MTLCommandQueue> qq = [device newCommandQueue];
    if (!qq) continue;
    id<MTLCommandBuffer> cb = [qq commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc dispatchThreadgroups:MTLSizeMake(1,1,1)
        threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [enc endEncoding];
    auto t0 = std::chrono::steady_clock::now();
    [cb commit];
    [cb waitUntilCompleted];
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "[cmqueue] probe queue %d: %.3f ms\n", i, ms);
    if (ms < bestMs) { bestMs = ms; bestQ = qq; }
  }
  fprintf(stderr, "[cmqueue] selected queue with %.3f ms probe latency\n",
    bestMs);
  return bestQ;
}

// §33.2 item 2 (VTK_METAL_TEST_MM_EPS): emptiness threshold for the GPU
// minmax build — cells whose max achievable opacity is <= eps are marked
// empty so the walk skips them. 0 (default) keeps the exact zero-opacity
// semantics. Approximate by design; image deltas must be quantified per eps.
static float VolumeMinMaxEps()
{
  if (const char* v = getenv("VTK_METAL_TEST_MM_EPS"))
    return static_cast<float>(std::atof(v));
  return 0.0f;
}

// §38.18.1: private-queue gating — qoff (window CB) is now the default.
// The fast private queue adds 3-slot Private SegPool/RayBin commit/wait
// serialization and a poisoned-slot hazard (§38.9.2). It is only used when
// explicitly opted in via VTK_METAL_TEST_CM_FASTQUEUE=1 (preferred) or the
// legacy VTK_METAL_TEST_CM_QUEUEPROBE=1. Absence or 0 → window CB (qoff).
static bool VolumeComputeMarchUseFastQueue()
{
  if (const char* v = getenv("VTK_METAL_TEST_CM_FASTQUEUE"))
    return std::atoi(v) != 0;
  if (const char* v = getenv("VTK_METAL_TEST_CM_QUEUEPROBE"))
    return std::atoi(v) != 0;
  return false; // qoff default
}

// §38.18.1: purge request via VTK_METAL_TEST_PURGE=1 (also accepts
// VTK_METAL_TEST_METAL_PURGE for foragability). Returns true once per
// GPURender when the flag is set so PurgeCaches can be triggered without
// a full ReleaseGraphicsResources cycle.
static bool VolumePurgeRequested()
{
  if (const char* v = getenv("VTK_METAL_TEST_PURGE"))
    return std::atoi(v) != 0;
  if (const char* v = getenv("VTK_METAL_TEST_METAL_PURGE"))
    return std::atoi(v) != 0;
  return false;
}


//------------------------------------------------------------------------------
static id<MTLTexture> CreateR8MinMaxTexture(
  id<MTLDevice> device,
  int dimX,
  int dimY,
  int dimZ,
  MTLStorageMode storage,
  MTLTextureUsage usage,
  NSUInteger mipmapLevels = 1)
{
  return NewTexture3D(
    device,
    MTLPixelFormatR8Unorm,
    static_cast<NSUInteger>(dimX),
    static_cast<NSUInteger>(dimY),
    static_cast<NSUInteger>(dimZ),
    usage,
    storage,
    true,
    mipmapLevels);
}

//------------------------------------------------------------------------------
static void AssignR8MinMaxTexture(
  void*& slot,
  id<MTLDevice> device,
  const std::vector<uint8_t>& data,
  int dimX,
  int dimY,
  int dimZ)
{
  id<MTLTexture> tex = CreateR8MinMaxTexture(
    device,
    dimX,
    dimY,
    dimZ,
    MTLStorageModeShared,
    MTLTextureUsageShaderRead);

  if (tex)
  {
    UploadR8Volume3D(tex, data.data(), dimX, dimY, dimZ);
  }

  AssignMetalObject(slot, tex);
}

//------------------------------------------------------------------------------
static void EncodeGPUMinMaxDilation(
  id<MTLComputeCommandEncoder> enc,
  id<MTLDevice> device,
  id<MTLTexture> volumeTex,
  id<MTLTexture> outputTex,
  const int volumeDims[3],
  const int minMaxDims[3],
  const MinMaxComputeUniforms& uniforms,
  id<MTLComputePipelineState> minMaxPipeline,
  id<MTLComputePipelineState> dilatePipeline)
{
  id<MTLTexture> scratchTex = CreateR8MinMaxTexture(
    device,
    minMaxDims[0],
    minMaxDims[1],
    minMaxDims[2],
    MTLStorageModePrivate,
    MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);

  if (!scratchTex)
  {
    return;
  }

  MTLSize gridSize = MTLSizeMake(
    static_cast<NSUInteger>(minMaxDims[0]),
    static_cast<NSUInteger>(minMaxDims[1]),
    static_cast<NSUInteger>(minMaxDims[2]));

  NSUInteger tgw = 8;
  MTLSize threadgroupSize = MTLSizeMake(
    std::min(tgw, static_cast<NSUInteger>(minMaxDims[0])),
    std::min(tgw, static_cast<NSUInteger>(minMaxDims[1])),
    std::min(tgw, static_cast<NSUInteger>(minMaxDims[2])));

  [enc setComputePipelineState:minMaxPipeline];
  [enc setTexture:volumeTex atIndex:0];
  [enc setTexture:scratchTex atIndex:1];
  [enc setBytes:&uniforms length:sizeof(uniforms) atIndex:0];
  [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

  [enc setComputePipelineState:dilatePipeline];
  [enc setTexture:scratchTex atIndex:0];
  [enc setTexture:outputTex atIndex:1];
  [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

  [scratchTex release];
}

//------------------------------------------------------------------------------
struct NormalComputeUniforms
{
  uint32_t dimX, dimY, dimZ;
  float gsX, gsY, gsZ;
  float scalarScale;
  float scalarBias;
  float gradNormFactor;
  float invBoundsX, invBoundsY, invBoundsZ;
  int32_t volTransposed;
};

static_assert(sizeof(NormalComputeUniforms) == 52,
  "NormalComputeUniforms must match Metal struct (52 bytes)");

static NormalComputeUniforms MakeNormalComputeUniforms(
  const int dims[3],
  const double scalarRange[2],
  float normalizationFactor,
  const double boundsSize[3],
  double avgSpacing)
{
  NormalComputeUniforms u{};

  double range = scalarRange[1] - scalarRange[0];
  if (range <= 0.0)
  {
    range = 1.0;
  }
  if (avgSpacing < 1e-10)
  {
    avgSpacing = 1.0;
  }

  u.dimX = static_cast<uint32_t>(dims[0]);
  u.dimY = static_cast<uint32_t>(dims[1]);
  u.dimZ = static_cast<uint32_t>(dims[2]);

  u.gsX = 1.0f / std::max(dims[0], 1);
  u.gsY = 1.0f / std::max(dims[1], 1);
  u.gsZ = 1.0f / std::max(dims[2], 1);

  u.scalarScale = 1.0f / std::max(
    static_cast<float>((scalarRange[1] - scalarRange[0]) / normalizationFactor),
    1e-6f);

  u.scalarBias =
    -(static_cast<float>(scalarRange[0] / normalizationFactor)) * u.scalarScale;

  // Match the march-path normalization: 0.5 * range / (normFactor * avgSpacing)
  u.gradNormFactor = static_cast<float>(
    range * 0.5 / (normalizationFactor * avgSpacing));

  double bs0 = std::max(boundsSize[0], 1e-10);
  double bs1 = std::max(boundsSize[1], 1e-10);
  double bs2 = std::max(boundsSize[2], 1e-10);
  u.invBoundsX = static_cast<float>(1.0 / bs0);
  u.invBoundsY = static_cast<float>(1.0 / bs1);
  u.invBoundsZ = static_cast<float>(1.0 / bs2);

  return u;
}

static void EncodeNormalCompute(
  id<MTLComputeCommandEncoder> enc,
  id<MTLTexture> volumeTex,
  id<MTLTexture> normalTex,
  const NormalComputeUniforms& uniforms,
  id<MTLComputePipelineState> pipeline)
{
  const int dims[3] = {
    static_cast<int>(volumeTex.width),
    static_cast<int>(volumeTex.height),
    static_cast<int>(volumeTex.depth)
  };

  [enc setComputePipelineState:pipeline];
  [enc setTexture:volumeTex atIndex:0];
  [enc setTexture:normalTex atIndex:1];
  [enc setBytes:&uniforms length:sizeof(uniforms) atIndex:0];

  MTLSize gridSize = MTLSizeMake(dims[0], dims[1], dims[2]);

  NSUInteger tgwMax = 16;
  NSUInteger tgwX = std::min(tgwMax, static_cast<NSUInteger>(dims[0]));
  NSUInteger tgwY = std::min(tgwMax, static_cast<NSUInteger>(dims[1]));
  NSUInteger tgwZ = std::min(
    static_cast<NSUInteger>(1024) / (tgwX * tgwY),
    static_cast<NSUInteger>(dims[2]));

  MTLSize threadgroupSize = MTLSizeMake(tgwX, tgwY, tgwZ);

  [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}

//------------------------------------------------------------------------------
// Refactoring 10: VolumeBounds — factor repeated model-bounds computation
struct VolumeBounds
{
  double Min[3];
  double Max[3];
  double Size[3];
};

static VolumeBounds ComputeVolumeBounds(vtkImageData* input, bool expandHalfCell = false)
{
  VolumeBounds bounds{};
  int ext[6];
  double origin[3], spacing[3];
  input->GetExtent(ext);
  input->GetOrigin(origin);
  input->GetSpacing(spacing);
  vtkMatrix3x3* direction = input->GetDirectionMatrix();

  // Cell-data proxies carry the cell scalars at point positions (the effective
  // input shifts the origin by half a cell so texel centers land on cell
  // centers), so their extent covers only the cell-center envelope. Expand the
  // index corners by half a cell on each side so the proxy box spans the full
  // cell extent, matching the OpenGL backend's vtkVolumeTexture::ComputeBounds
  // (iMax = extent[1] + IsCellData over the original cell-data extent). The
  // volume-to-texture matrix stays based on the unexpanded proxy extent, so the
  // boundary cells are reached through the clamp-to-edge sampling of the march.
  double expand = expandHalfCell ? 0.5 : 0.0;
  // Rotate the 8 extents corners through the image-data direction matrix and
  // take the axis-aligned bounds, mirroring the OpenGL backend's
  // vtkVolumeTexture::ComputeBounds (LoadedBoundsAA).
  double ijkCorners[8][3] = {
    { ext[0] - expand, ext[2] - expand, ext[4] - expand },
    { ext[1] + expand, ext[2] - expand, ext[4] - expand },
    { ext[0] - expand, ext[3] + expand, ext[4] - expand },
    { ext[1] + expand, ext[3] + expand, ext[4] - expand },
    { ext[0] - expand, ext[2] - expand, ext[5] + expand },
    { ext[1] + expand, ext[2] - expand, ext[5] + expand },
    { ext[0] - expand, ext[3] + expand, ext[5] + expand },
    { ext[1] + expand, ext[3] + expand, ext[5] + expand },
  };
  double xyz[3];
  bounds.Min[0] = bounds.Min[1] = bounds.Min[2] = VTK_DOUBLE_MAX;
  bounds.Max[0] = bounds.Max[1] = bounds.Max[2] = VTK_DOUBLE_MIN;
  for (int i = 0; i < 8; ++i)
  {
    vtkImageData::TransformContinuousIndexToPhysicalPoint(
      ijkCorners[i][0], ijkCorners[i][1], ijkCorners[i][2], origin, spacing,
      direction->GetData(), xyz);
    for (int k = 0; k < 3; ++k)
    {
      bounds.Min[k] = std::min(bounds.Min[k], xyz[k]);
      bounds.Max[k] = std::max(bounds.Max[k], xyz[k]);
    }
  }
  for (int i = 0; i < 3; ++i)
  {
    bounds.Size[i] = bounds.Max[i] - bounds.Min[i];
    if (bounds.Size[i] < 1e-10)
      bounds.Size[i] = 1.0;
  }
  return bounds;
}

inline float NormalizeToVolumeSpace(const VolumeBounds& bounds, int axis, double value)
{
  return static_cast<float>((value - bounds.Min[axis]) / bounds.Size[axis]);
}

//------------------------------------------------------------------------------
// Refactoring 11: transfer-function table helpers
static uint8_t ColorToByte(double x)
{
  return static_cast<uint8_t>(std::clamp(x, 0.0, 1.0) * 255.0);
}

// Fill a RGBA8 transfer function row without opacity pre-integration.
// Used for label-map transfer functions: binary label masks typically have
// opacity 0 or 1, so pre-integration is a no-op. Matches OpenGL's
// vtkOpenGLVolumeMaskTransferFunction2D::InternalUpdate which stores raw opacity.
static void FillTransferFunctionRGBA8(
  vtkColorTransferFunction* colorFunc,
  vtkPiecewiseFunction* opacityFunc,
  double scalarMin,
  double scalarMax,
  int width,
  uint8_t* row)
{
  std::vector<double> rgb(width * 3);
  std::vector<double> alpha(width);
  colorFunc->GetTable(scalarMin, scalarMax, width, rgb.data());
  opacityFunc->GetTable(scalarMin, scalarMax, width, alpha.data());
  for (int i = 0; i < width; ++i)
  {
    row[i * 4 + 0] = ColorToByte(rgb[i * 3 + 0]);
    row[i * 4 + 1] = ColorToByte(rgb[i * 3 + 1]);
    row[i * 4 + 2] = ColorToByte(rgb[i * 3 + 2]);
    row[i * 4 + 3] = ColorToByte(alpha[i]);
  }
}

// Apply the OpenGL backend's blend-mode-specific opacity correction
// (vtkOpenGLVolumeOpacityTable::InternalUpdate): COMPOSITE pre-integrates the
// opacity with 1 - (1-a)^factor, ADDITIVE scales it by factor, and all other
// blend modes (MIP/MinIP/Average) use the raw table values.
static double ApplyOpacityBlendCorrection(double a, double factor, int blendMode)
{
  if (a <= 0.0001 || factor <= 0.0)
  {
    return a;
  }
  if (blendMode == vtkVolumeMapper::COMPOSITE_BLEND)
  {
    return 1.0 - std::pow(1.0 - a, factor);
  }
  if (blendMode == vtkVolumeMapper::ADDITIVE_BLEND)
  {
    return a * factor;
  }
  return a;
}

// Fill a RGBA8 transfer function row with the blend-mode-dependent opacity
// correction, matching the OpenGL backend's approach
// (vtkOpenGLVolumeOpacityTable::InternalUpdate). COMPOSITE pre-integrates with
// 1 - (1-a)^factor, ADDITIVE scales by factor, all other blend modes are raw.
static void FillTransferFunctionRGBA8WithPreIntegration(
  vtkColorTransferFunction* colorFunc,
  vtkPiecewiseFunction* opacityFunc,
  double scalarMin,
  double scalarMax,
  int width,
  uint8_t* row,
  double factor,
  int blendMode)
{
  std::vector<double> rgb(width * 3);
  std::vector<double> alpha(width);
  colorFunc->GetTable(scalarMin, scalarMax, width, rgb.data());
  opacityFunc->GetTable(scalarMin, scalarMax, width, alpha.data());
  for (int i = 0; i < width; ++i)
  {
    double a = ApplyOpacityBlendCorrection(alpha[i], factor, blendMode);
    row[i * 4 + 0] = ColorToByte(rgb[i * 3 + 0]);
    row[i * 4 + 1] = ColorToByte(rgb[i * 3 + 1]);
    row[i * 4 + 2] = ColorToByte(rgb[i * 3 + 2]);
    row[i * 4 + 3] = ColorToByte(a);
  }
}

// Fill a RGBA16Float transfer function row with the blend-mode-dependent
// opacity correction, matching the OpenGL backend's approach
// (vtkOpenGLVolumeOpacityTable::InternalUpdate). Unlike the RGBA8 variant, the
// alpha channel retains full float precision so that low-opacity transfer
// functions (e.g. 0.005 max opacity) do not round to zero after the 8-bit
// quantization step.
static void FillTransferFunctionRGBA16FWithPreIntegration(
  vtkColorTransferFunction* colorFunc,
  vtkPiecewiseFunction* opacityFunc,
  double colorMin,
  double colorMax,
  double opacityMin,
  double opacityMax,
  int width,
  uint16_t* row,
  double factor,
  int blendMode)
{
  std::vector<double> rgb(width * 3);
  std::vector<double> alpha(width);
  colorFunc->GetTable(colorMin, colorMax, width, rgb.data());
  opacityFunc->GetTable(opacityMin, opacityMax, width, alpha.data());
  for (int i = 0; i < width; ++i)
  {
    double a = ApplyOpacityBlendCorrection(alpha[i], factor, blendMode);
    row[i * 4 + 0] = FloatToHalf(static_cast<float>(std::clamp(rgb[i * 3 + 0], 0.0, 1.0)));
    row[i * 4 + 1] = FloatToHalf(static_cast<float>(std::clamp(rgb[i * 3 + 1], 0.0, 1.0)));
    row[i * 4 + 2] = FloatToHalf(static_cast<float>(std::clamp(rgb[i * 3 + 2], 0.0, 1.0)));
    row[i * 4 + 3] = FloatToHalf(static_cast<float>(std::clamp(a, 0.0, 1.0)));
  }
}

// Two-range variant used by dependent multi-component volumes (OpenGL
// UpdateColorTransferFunction(component) / UpdateOpacityTransferFunction(
// component) parity): the RGB channels map one scalar range (component 0) while
// the alpha channel maps another (the last component). The shader samples the
// shared RGBA table at the two corresponding normalized coordinates.
static void FillTransferFunctionRGBA16FWithPreIntegration(
  vtkColorTransferFunction* colorFunc,
  vtkPiecewiseFunction* opacityFunc,
  double scalarMin,
  double scalarMax,
  int width,
  uint16_t* row,
  double factor,
  int blendMode)
{
  FillTransferFunctionRGBA16FWithPreIntegration(
    colorFunc, opacityFunc, scalarMin, scalarMax, scalarMin, scalarMax,
    width, row, factor, blendMode);
}

// Compute the transfer function table width using the same rule as the OpenGL
// backend (vtkOpenGLVolumeLookupTable::ComputeIdealTextureSize +
// GetMaximumSupportedTextureWidth): the ideal width is the min-sample estimate
// of the color and opacity functions over the scalar range, rounded up to a
// power of two and clamped to a 1024-texel floor. The estimate is
// (range / minimum node spacing), which is unbounded for pathological
// transfer functions with near-coincident nodes, so it is additionally capped
// at kMaxTransferFunctionWidth (Apple GPUs support up to 16384-wide 2D
// textures; no dense realistic transfer function approaches this). The
// combined RGBA table shares a single width, so the larger of the two
// estimates is used.
static const int kMaxTransferFunctionWidth = 16384;

static int ComputeTransferFunctionWidth(
  vtkColorTransferFunction* colorFunc,
  vtkPiecewiseFunction* opacityFunc,
  const double range[2])
{
  int idealWidth = 1024;
  if (colorFunc)
  {
    idealWidth = std::max(
      idealWidth, colorFunc->EstimateMinNumberOfSamples(range[0], range[1]));
  }
  if (opacityFunc)
  {
    idealWidth = std::max(
      idealWidth, opacityFunc->EstimateMinNumberOfSamples(range[0], range[1]));
  }
  return std::min(kMaxTransferFunctionWidth,
    std::max(1024, vtkMath::NearestPowerOfTwo(idealWidth)));
}

//------------------------------------------------------------------------------
// Refactoring 12: create a dummy 3D texture filled with a constant value
static id<MTLTexture> CreateDummy3DTexture(
  id<MTLDevice> device,
  MTLPixelFormat format,
  const void* initialValue,
  NSUInteger initialValueSize)
{
  id<MTLTexture> tex = NewTexture3D(
    device, format, 1, 1, 1,
    MTLTextureUsageShaderRead, MTLStorageModeShared);
  if (tex)
  {
    MTLRegion region = MTLRegionMake3D(0, 0, 0, 1, 1, 1);
    [tex replaceRegion:region
            mipmapLevel:0
                  slice:0
              withBytes:initialValue
            bytesPerRow:initialValueSize
          bytesPerImage:initialValueSize];
  }
  return tex;
}

//------------------------------------------------------------------------------
// Refactoring 13: bind a texture or its fallback
static inline void SetFragmentTextureOrFallback(
  id<MTLRenderCommandEncoder> encoder,
  NSUInteger index,
  void* texture,
  void* fallback)
{
  id<MTLTexture> tex = texture
    ? (__bridge id<MTLTexture>)texture
    : (__bridge id<MTLTexture>)fallback;
  [encoder setFragmentTexture:tex atIndex:index];
}

static inline void SetComputeTextureOrFallback(
  id<MTLComputeCommandEncoder> encoder,
  NSUInteger index,
  void* texture,
  void* fallback)
{
  id<MTLTexture> tex = texture
    ? (__bridge id<MTLTexture>)texture
    : (__bridge id<MTLTexture>)fallback;
  if (tex)
  {
    [encoder setTexture:tex atIndex:index];
  }
}

}

VTK_ABI_NAMESPACE_BEGIN

vtkStandardNewMacro(vtkMetalGPUVolumeRayCastMapper);

//------------------------------------------------------------------------------
vtkOverrideAttribute* vtkMetalGPUVolumeRayCastMapper::CreateOverrideAttributes()
{
  return vtkOverrideAttribute::CreateAttributeChain("RenderingBackend", "Metal", nullptr);
}

//------------------------------------------------------------------------------
int vtkMetalGPUVolumeRayCastMapper::IsRenderSupported(
  vtkRenderWindow* vtkNotUsed(window), vtkVolumeProperty* vtkNotUsed(property))
{
  // The Metal backend supports GPU ray casting for all inputs and blend modes
  // handled by GPURender (image data, rectilinear grids, cell data, multi-
  // component independent / additive modes, 2D transfer functions, ...).
  return 1;
}

//------------------------------------------------------------------------------
vtkMetalGPUVolumeRayCastMapper::vtkMetalGPUVolumeRayCastMapper()
{
  this->SampleDistance = 1.0f;
  this->FrameSemaphore = dispatch_semaphore_create(3);
}

//------------------------------------------------------------------------------
vtkMetalGPUVolumeRayCastMapper::~vtkMetalGPUVolumeRayCastMapper()
{
  this->WaitForInFlightFrames();
  this->ReleaseGraphicsResources(nullptr);
#if !__has_feature(objc_arc)
  if (this->FrameSemaphore)
  {
    dispatch_release((dispatch_semaphore_t)this->FrameSemaphore);
    this->FrameSemaphore = nullptr;
  }
#endif
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::WaitForInFlightFrames()
{
  if (!this->FrameSemaphore)
  {
    return;
  }
  dispatch_semaphore_t sem = (__bridge dispatch_semaphore_t)this->FrameSemaphore;
  for (int i = 0; i < 3; ++i)
  {
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
  }
  // Restore the semaphore to valid state so a future GPURender (e.g. after
  // ReleaseGraphicsResources for a window resize) does not deadlock.
  for (int i = 0; i < 3; ++i)
  {
    dispatch_semaphore_signal(sem);
  }
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "PreferHalfPrecision: " << this->PreferHalfPrecision << "\n";
  os << indent << "UseGPUMinMax: " << this->UseGPUMinMax << "\n";
  os << indent << "UseFullscreenCameraInside: " << this->UseFullscreenCameraInside << "\n";
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::EnsureShaderLibrary(void* mtlDeviceVoid)
{
  if (this->CachedShaderLibrary)
  {
    return true;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;
    NSError* error = nil;
    NSString* shaderSource = [NSString stringWithUTF8String:vtkMetalShaders];
    // A/B knob for the fastMathEnabled finding (metal_gap microbench: NO is
    // ~8% faster for the bare-fetch march). Unset = historical options:nil
    // behavior (fastMathEnabled=YES).
    const char* fmEnv = getenv("VTK_METAL_TEST_FAST_MATH");
    MTLCompileOptions* options = nil;
    if (fmEnv)
    {
      options = [[MTLCompileOptions alloc] init];
      options.fastMathEnabled = (strcmp(fmEnv, "0") == 0) ? NO : YES;
    }
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:options error:&error];

    if (!library)
    {
      vtkErrorMacro(<< "Failed to compile Metal shader library: "
                    << [[error localizedDescription] UTF8String]);
      return false;
    }

    AssignMetalObject(this->CachedShaderLibrary, library);
  }

  return true;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::SetPartitions(
  unsigned short x, unsigned short y, unsigned short z)
{
  if (x > 0 && y > 0 && z > 0)
  {
    this->Partitions[0] = x;
    this->Partitions[1] = y;
    this->Partitions[2] = z;
  }
  else
  {
    this->Partitions[0] = this->Partitions[1] = this->Partitions[2] = 1;
  }
  this->Modified();
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ComputeReductionFactor(double allocatedTime)
{
  if (!this->AutoAdjustSampleDistances)
  {
    this->ReductionFactor = 1.0 / this->ImageSampleDistance;
    return;
  }

  if (this->TimeToDraw)
  {
    double oldFactor = this->ReductionFactor;

    double timeToDraw;
    if (allocatedTime < 1.0)
    {
      timeToDraw = this->SmallTimeToDraw;
      if (timeToDraw == 0.0)
      {
        timeToDraw = this->BigTimeToDraw / 3.0;
      }
    }
    else
    {
      timeToDraw = this->BigTimeToDraw;
    }

    if (timeToDraw == 0.0)
    {
      timeToDraw = 10.0;
    }

    double fullTime = timeToDraw / this->ReductionFactor;
    double newFactor = allocatedTime / fullTime;

    this->ReductionFactor = (newFactor + oldFactor) / 2.0;

    // Discretize to avoid visual oscillation
    this->ReductionFactor = (this->ReductionFactor > 1.0) ? 1.0 : (this->ReductionFactor);

    if (this->ReductionFactor < 0.20)
    {
      this->ReductionFactor = 0.10;
    }
    else if (this->ReductionFactor < 0.50)
    {
      this->ReductionFactor = 0.20;
    }
    else if (this->ReductionFactor < 1.0)
    {
      this->ReductionFactor = 0.50;
    }

    // Clamp to user-specified bounds
    if (1.0 / this->ReductionFactor > this->MaximumImageSampleDistance)
    {
      this->ReductionFactor = 1.0 / this->MaximumImageSampleDistance;
    }
    if (1.0 / this->ReductionFactor < this->MinimumImageSampleDistance)
    {
      this->ReductionFactor = 1.0 / this->MinimumImageSampleDistance;
    }
  }
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::EnsureImageSampleResources(
  void* deviceVoid, int width, int height)
{
  // Force recreation if pixel format changed (e.g., BGRA8Unorm -> RGBA16Float)
  const int desiredFormat = MTLPixelFormatRGBA16Float;
  if (this->ImageSampleColorTexture && this->ImageSampleFBOWidth == width &&
    this->ImageSampleFBOHeight == height && this->ImageSamplePixelFormat == desiredFormat)
  {
    return true;
  }

  this->ReleaseImageSampleResources();

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)deviceVoid;

    // Create offscreen color texture (RGBA16Float for inter-block accumulation)
    id<MTLTexture> colorTex = NewTexture2D(
      device,
      MTLPixelFormatRGBA16Float,
      width, height,
      MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead,
      MTLStorageModePrivate);
    if (!colorTex)
    {
      vtkErrorMacro("Failed to create image-sample color texture");
      return false;
    }
    AssignMetalObject(this->ImageSampleColorTexture, colorTex);

    // Create blit pipeline (fullscreen quad that samples the offscreen texture)
    if (!this->EnsureShaderLibrary(deviceVoid))
    {
      this->ReleaseImageSampleResources();
      return false;
    }
    id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary;

    NSError* error = nil;
    id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_fullscreen_main"];
    id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragment_image_sample_blit"];
    if (!vertexFunc || !fragmentFunc)
    {
      vtkErrorMacro("Failed to find image-sample blit shader functions");
      [vertexFunc release];
      [fragmentFunc release];
      this->ReleaseImageSampleResources();
      return false;
    }

    MTLRenderPipelineDescriptor* pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunc;
    pipelineDesc.fragmentFunction = fragmentFunc;
    pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
    // Enable blending so the blit composites the accumulated volume over the background
    pipelineDesc.colorAttachments[0].blendingEnabled = YES;
    pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
    pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    pipelineDesc.rasterSampleCount = 1; // Always 1 for blit; MSAA is on the main pass

    id<MTLRenderPipelineState> pso =
      [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    [pipelineDesc release];
    if (!pso)
    {
      vtkErrorMacro(<< "Image-sample blit pipeline: " << [[error localizedDescription] UTF8String]);
      [vertexFunc release];
      [fragmentFunc release];
      this->ReleaseImageSampleResources();
      return false;
    }
    // The +1 from new() goes to the cache.  AssignRetainedMetalObject
    // adds a separate +1 for the member slot.
    {
      VolumePipelineKey k = { static_cast<uint32_t>(VolumePipelineType::ImageSampleBlit),
        MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1, 0 };
      this->PipelineCache[k] = (__bridge void*)pso;
    }
    AssignRetainedMetalObject(this->ImageSamplePipeline, pso);

    [vertexFunc release];
    [fragmentFunc release];

    this->ImageSampleFBOWidth = width;
    this->ImageSampleFBOHeight = height;
    this->ImageSamplePixelFormat = desiredFormat;
  }

  return true;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ReleaseImageSampleResources()
{
  ReleaseMetalObject(this->ImageSampleColorTexture);
  ReleaseMetalObject(this->ImageSampleDepthTexture);
  ReleaseMetalObject(this->ImageSamplePipeline);
  this->ImageSampleFBOWidth = 0;
  this->ImageSampleFBOHeight = 0;
  this->ImageSamplePixelFormat = 0;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::EnsureSlabResources(
  void* deviceVoid, int width, int height)
{
  if (this->SlabTextureA && this->SlabTextureB && this->SlabFBOWidth == width &&
    this->SlabFBOHeight == height)
  {
    return true;
  }

  this->ReleaseSlabResources();

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)deviceVoid;

    // Ping-pong color textures (BGRA8Unorm, matching the DirectScreen
    // pipeline's attachment format). Private storage: only the GPU reads or
    // writes them.
    id<MTLTexture> texA = NewTexture2D(device, MTLPixelFormatBGRA8Unorm,
      width, height,
      MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead,
      MTLStorageModePrivate);
    id<MTLTexture> texB = NewTexture2D(device, MTLPixelFormatBGRA8Unorm,
      width, height,
      MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead,
      MTLStorageModePrivate);
    if (!texA || !texB)
    {
      vtkErrorMacro("Failed to create slab ping-pong textures");
      return false;
    }
    AssignMetalObject(this->SlabTextureA, texA);
    AssignMetalObject(this->SlabTextureB, texB);

    // Per-pass depth attachment (the DirectScreen pipeline declares
    // Depth32Float). Cleared per pass; only the pass's own box z-test uses it.
    id<MTLTexture> depthTex = NewTexture2D(device, MTLPixelFormatDepth32Float,
      width, height,
      MTLTextureUsageRenderTarget, MTLStorageModePrivate);
    if (!depthTex)
    {
      vtkErrorMacro("Failed to create slab depth texture");
      this->ReleaseSlabResources();
      return false;
    }
    AssignMetalObject(this->SlabDepthTexture, depthTex);

    this->SlabFBOWidth = width;
    this->SlabFBOHeight = height;
  }

  return true;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ReleaseSlabResources()
{
  ReleaseMetalObject(this->SlabTextureA);
  ReleaseMetalObject(this->SlabTextureB);
  ReleaseMetalObject(this->SlabDepthTexture);
  this->SlabFBOWidth = 0;
  this->SlabFBOHeight = 0;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::EnsureRTTResources(
  void* deviceVoid, int width, int height, int depthScalarType)
{
  if (this->RTTColorTexture && this->RTTDepthTexture && this->RTTWidth == width &&
    this->RTTHeight == height && this->RTTDepthScalarType == depthScalarType)
  {
    return true;
  }

  this->ReleaseRTTResources();

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)deviceVoid;

    // RGBA8 matches the OpenGL RenderToImage color target
    // (Create2D(w, h, 4, VTK_UNSIGNED_CHAR)); GetColorImage outputs uchar anyway.
    id<MTLTexture> colorTex = NewTexture2D(device, MTLPixelFormatRGBA8Unorm, width, height,
      MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead, MTLStorageModeShared);
    if (!colorTex)
    {
      vtkErrorMacro("Failed to create RenderToImage color texture");
      return false;
    }
    AssignMetalObject(this->RTTColorTexture, colorTex);

    // Depth image scalar type: VTK_FLOAT (default), unsigned short or unsigned
    // char. Only R32Float is supported natively; the other types are converted
    // during the CPU readback in GetDepthImage.
    MTLPixelFormat depthFormat = MTLPixelFormatR32Float;
    id<MTLTexture> depthTex = NewTexture2D(
      device, depthFormat, width, height,
      MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead, MTLStorageModeShared);
    if (!depthTex)
    {
      vtkErrorMacro("Failed to create RenderToImage depth texture");
      this->ReleaseRTTResources();
      return false;
    }
    AssignMetalObject(this->RTTDepthTexture, depthTex);

    this->RTTWidth = width;
    this->RTTHeight = height;
    this->RTTDepthScalarType = depthScalarType;
  }

  return true;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ReleaseRTTResources()
{
  ReleaseMetalObject(this->RTTColorTexture);
  ReleaseMetalObject(this->RTTDepthTexture);
  this->RTTWidth = 0;
  this->RTTHeight = 0;
  this->RTTDepthScalarType = -1;
}

//------------------------------------------------------------------------------
// §35.14 segment pre-pass resources: three RGBA32Float ray-atlas planes, the
// per-pixel u32 offset map, the compacted gap pool and its claim counter, and
// the volume_segment_build compute pipeline. Cached by atlas size.
bool vtkMetalGPUVolumeRayCastMapper::EnsureSegResources(
  void* deviceVoid, void* mtlQueueVoid, int width, int height)
{
  (void)mtlQueueVoid;
  if (this->SegAtlasATexture && this->SegBuildComputePipeline &&
      this->SegMarchTexture &&
      this->SegAtlasWidth == width && this->SegAtlasHeight == height)
  {
    return true;
  }

  ReleaseMetalObject(this->SegAtlasATexture);
  ReleaseMetalObject(this->SegAtlasBTexture);
  ReleaseMetalObject(this->SegAtlasCTexture);
  ReleaseMetalObject(this->SegIndexBuffer);
  ReleaseMetalObject(this->SegPoolBuffer);
  ReleaseMetalObject(this->SegPoolCounterBuffer);
  ReleaseMetalObject(this->SegMarchTexture);

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)deviceVoid;

    MTLTextureDescriptor* td =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                         width:width height:height
                                                     mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> texA = [device newTextureWithDescriptor:td];
    id<MTLTexture> texB = [device newTextureWithDescriptor:td];
    id<MTLTexture> texC = [device newTextureWithDescriptor:td];
    if (!texA || !texB || !texC)
    {
      vtkErrorMacro("Failed to create segment ray-atlas textures");
      return false;
    }
    AssignMetalObject(this->SegAtlasATexture, texA);
    AssignMetalObject(this->SegAtlasBTexture, texB);
    AssignMetalObject(this->SegAtlasCTexture, texC);

    id<MTLBuffer> idxBuf = [device newBufferWithLength:(NSUInteger)width * height * 4
                                               options:MTLResourceStorageModePrivate];
    // Claim counter: shared so the CPU can zero it each frame.
    id<MTLBuffer> cntBuf = [device newBufferWithLength:4
                                               options:MTLResourceStorageModeShared];
    // Gap pool: cap at 192 MiB (48M words) or the device maximum. Overflowed
    // rays fall back to compositing every sample.
    const NSUInteger maxLen = [device maxBufferLength];
    const NSUInteger poolBytes = std::min<NSUInteger>((64u << 20), maxLen);
    id<MTLBuffer> poolBuf = [device newBufferWithLength:poolBytes
                                                options:MTLResourceStorageModePrivate];
    if (!idxBuf || !cntBuf || !poolBuf)
    {
      vtkErrorMacro("Failed to create segment buffers");
      return false;
    }
    AssignMetalObject(this->SegIndexBuffer, idxBuf);
    AssignMetalObject(this->SegPoolCounterBuffer, cntBuf);
    AssignMetalObject(this->SegPoolBuffer, poolBuf);
    this->SegPoolCapWords = poolBytes / 4;

    // Offscreen march target: RGBA16Float to MATCH the OffscreenLayer
    // pipeline's declared color format exactly.
    id<MTLTexture> marchTex = NewTexture2D(device, MTLPixelFormatRGBA16Float,
      width, height,
      MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
      MTLStorageModePrivate);
    if (!marchTex)
    {
      vtkErrorMacro("Failed to create segment march texture");
      return false;
    }
    AssignMetalObject(this->SegMarchTexture, marchTex);

    if (!this->SegBuildComputePipeline)
    {
      id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary;
      if (!library)
      {
        vtkErrorMacro("Segment build: shader library missing");
        return false;
      }
      NSError* err = nil;
      id<MTLFunction> fn = [library newFunctionWithName:@"volume_segment_build"];
      if (!fn)
      {
        vtkErrorMacro("Failed to find volume_segment_build");
        return false;
      }
      [fn release];
      // §38.17: the builder now calls synthesizeAtlasRay (synth-input mode),
      // pulling in fc_volRg8 / fc_volTransposed / fc_volTransposedY. Specialize
      // with the same values the compute march pipeline derives (feature-mask
      // policy, not raw env) so the builder's walk geometry matches the
      // marcher's bit for bit.
      MTLFunctionConstantValues* segFc =
        [[MTLFunctionConstantValues alloc] init];
      BOOL segVolRg8 = VolumeRg8PairActive() ? YES : NO;
      BOOL segTrX = (this->VolumeTextureAxisDepth == 1) ? YES : NO;
      BOOL segTrY = (this->VolumeTextureAxisDepth == 2) ? YES : NO;
      [segFc setConstantValue:&segVolRg8 type:MTLDataTypeBool
                     withName:@"fc_volRg8"];
      [segFc setConstantValue:&segTrX type:MTLDataTypeBool
                     withName:@"fc_volTransposed"];
      [segFc setConstantValue:&segTrY type:MTLDataTypeBool
                     withName:@"fc_volTransposedY"];
      NSError* segFcErr = nil;
      id<MTLFunction> segFnSpecialized =
        [library newFunctionWithName:@"volume_segment_build"
                      constantValues:segFc
                               error:&segFcErr];
      if (!segFnSpecialized)
      {
        vtkErrorMacro(<< "Segment build specialization failed: "
                      << [[segFcErr localizedDescription] UTF8String]);
        return false;
      }
      id<MTLComputePipelineState> cps =
        [device newComputePipelineStateWithFunction:segFnSpecialized error:&err];
      [segFnSpecialized release];
      if (!cps)
      {
        vtkErrorMacro(<< "Segment build pipeline failed: "
                      << [[err localizedDescription] UTF8String]);
        return false;
      }
      AssignMetalObject(this->SegBuildComputePipeline, cps);
    }

    this->SegAtlasWidth = width;
    this->SegAtlasHeight = height;
  }
  return true;
}

//------------------------------------------------------------------------------
// §38.6 / §36.4 Design B — Ensure resources for compute marcher and ray binning
bool vtkMetalGPUVolumeRayCastMapper::EnsureComputeMarchResources(
  void* deviceVoid, void* mtlQueueVoid, int width, int height)
{
  // CM_SYNTH never reads the atlas planes — skip their allocation entirely
  // (3x RGBA32F fullscreen: ~50 MB @2048² of untouched private memory).
  const bool synthLayout = VolumeComputeMarchSynth();
  if (!synthLayout &&
      !this->EnsureSegResources(deviceVoid, mtlQueueVoid, width, height))
  {
    return false;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)deviceVoid;
  if (!this->SegMarchTexture)
  {
    // Offscreen march target. Default RGBA16Float matches the OffscreenLayer
    // pipeline's declared color format; CM_RGBA8 switches to BGRA8Unorm —
    // the SAME precision class as the fragment DirectScreen path (which
    // renders into the 8-bit drawable), halving march-write + blit bytes.
    const bool rgba8 = getenv("VTK_METAL_TEST_CM_RGBA8") != nullptr;
    id<MTLTexture> marchTex = NewTexture2D(device,
      rgba8 ? MTLPixelFormatBGRA8Unorm : MTLPixelFormatRGBA16Float,
      width, height,
      MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead |
        MTLTextureUsageShaderWrite,
      MTLStorageModePrivate);
    if (!marchTex)
    {
      vtkErrorMacro("Failed to create compute march target");
      return false;
    }
    AssignMetalObject(this->SegMarchTexture, marchTex);
  }

  // 4 bins × width × height packed UVs
  const size_t neededBytes = static_cast<size_t>(width) * height * sizeof(uint32_t) * 4;
  if (!this->RayBinIndicesBuffer || this->RayBinIndicesCapBytes < neededBytes)
  {
    ReleaseMetalObject(this->RayBinIndicesBuffer);
    id<MTLBuffer> binBuf = [device newBufferWithLength:neededBytes
                                               options:MTLResourceStorageModePrivate];
    if (!binBuf)
    {
      vtkErrorMacro("Failed to allocate RayBinIndicesBuffer");
      return false;
    }
    AssignMetalObject(this->RayBinIndicesBuffer, binBuf);
    this->RayBinIndicesCapBytes = neededBytes;
  }

  if (!this->SegConsumeDbgBuffer)
  {
    id<MTLBuffer> dbgBuf = [device newBufferWithLength:16 * sizeof(uint32_t)
        options:MTLResourceStorageModeShared];
    AssignMetalObject(this->SegConsumeDbgBuffer, dbgBuf);
  }
  if (!this->RayBinCountersBuffer)
  {
    id<MTLBuffer> cntBuf = [device newBufferWithLength:16 * sizeof(uint32_t)
                                               options:MTLResourceStorageModeShared];
    if (!cntBuf)
    {
      vtkErrorMacro("Failed to allocate RayBinCountersBuffer");
      return false;
    }
    AssignMetalObject(this->RayBinCountersBuffer, cntBuf);
  }

  return true;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ReleaseGridTraversalResources()
{
  ReleaseMetalObject(this->OccupancyGridTexture);
  ReleaseMetalObject(this->GridTraversalUniformBuffer);
  this->CachedGridDims[0] = 0;
  this->CachedGridDims[1] = 0;
  this->CachedGridDims[2] = 0;
  this->GridTraversalResourcesValid = false;
  this->GridTraversalUploadTime.Modified();
}

//------------------------------------------------------------------------------
// WARNING: This is shared between the partitioned and non-partitioned paths.
// The partitioned path uses the global volume texture for single-pass grid
// traversal instead of per-block textures.
bool vtkMetalGPUVolumeRayCastMapper::CreateGlobalVolumeTexture(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkImageData* input, vtkDataArray* scalars)
{
  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mtlQueueVoid;

  int dims[3];
  input->GetDimensions(dims);
  if (dims[0] < 1 || dims[1] < 1 || dims[2] < 1)
  {
    vtkErrorMacro("Volume has zero dimensions");
    return false;
  }

  int dataType = scalars->GetDataType();
  int numComponents = scalars->GetNumberOfComponents();
  // Bound the conversion by the volume's voxel count: the staging buffer below
  // is sized from dims, but over-provisioned arrays (e.g. vtkImplicitArray in
  // TestSmartVolumeMapperImplicitArray) report more tuples than fit the texture.
  // The OpenGL backend reads only the block-size slice of the array
  // (vtkVolumeTexture::LoadTexture), so clamping restores parity and avoids an
  // out-of-bounds write into the dims-sized buffer.
  vtkIdType numTuples = std::min(scalars->GetNumberOfTuples(),
    static_cast<vtkIdType>(dims[0]) * dims[1] * dims[2]);

  VolumeFormat fmtInfo = ChooseVolumeFormat(
    dataType, numComponents, this->ScalarRange, this->PreferHalfPrecision);
  bool useHalf = this->PreferHalfPrecision &&
    HalfRangeIsSafe(this->ScalarRange[0], this->ScalarRange[1]);
  this->ScalarNormalizationFactor = fmtInfo.NormalizationFactor;

  bool gpuConversionUsed = false;
  int actualComponents = (numComponents == 3) ? 4 : numComponents;
  NSUInteger bytesPerRow = static_cast<NSUInteger>(dims[0]) * fmtInfo.BytesPerComponent *
    actualComponents;
  NSUInteger bytesPerImage = bytesPerRow * dims[1];
  NSUInteger totalBytes = bytesPerImage * dims[2];

  if (fmtInfo.NeedsConversion && this->UseGPUConversion)
  {
    const char* kernelName = nullptr;
         if (dataType == VTK_SHORT)        kernelName = useHalf ? "volume_convert_short_to_half"  : "volume_convert_short_to_float";
    else if (dataType == VTK_INT)          kernelName = useHalf ? "volume_convert_int_to_half"    : "volume_convert_int_to_float";
    else if (dataType == VTK_UNSIGNED_INT) kernelName = useHalf ? "volume_convert_uint_to_half"   : "volume_convert_uint_to_float";
    else if (dataType == VTK_FLOAT && useHalf) kernelName = "volume_convert_float_to_half";

    if (kernelName)
    {
      if (!this->EnsureConversionPipelines(mtlDeviceVoid))
      {
        return false;
      }

      id<MTLComputePipelineState> pipeline = nullptr;
           if (dataType == VTK_SHORT)        pipeline = (__bridge id<MTLComputePipelineState>)(useHalf ? this->ConvertShortToHalfPipeline : this->ConvertShortToFloatPipeline);
      else if (dataType == VTK_INT)          pipeline = (__bridge id<MTLComputePipelineState>)(useHalf ? this->ConvertIntToHalfPipeline : this->ConvertIntToFloatPipeline);
      else if (dataType == VTK_UNSIGNED_INT) pipeline = (__bridge id<MTLComputePipelineState>)(useHalf ? this->ConvertUIntToHalfPipeline : this->ConvertUIntToFloatPipeline);
      else if (dataType == VTK_FLOAT && useHalf) pipeline = (__bridge id<MTLComputePipelineState>)this->ConvertFloatToHalfPipeline;
      if (!pipeline)
      {
        vtkErrorMacro("GPU conversion pipeline not available for data type " << dataType);
        return false;
      }

      size_t srcBytesPerVoxel = static_cast<size_t>(vtkDataArray::GetDataTypeSize(dataType)) * numComponents;
      size_t totalSrcBytes = static_cast<size_t>(numTuples) * srcBytesPerVoxel;

      id<MTLBuffer> srcBuf = [device newBufferWithBytes:scalars->GetVoidPointer(0)
                                                  length:totalSrcBytes
                                                 options:MTLResourceStorageModeShared];
      if (!srcBuf)
      {
        vtkErrorMacro("Failed to create source buffer for GPU conversion");
        return false;
      }

      int outputComponents = actualComponents;
      ReleaseMetalObject(this->VolumeTexture);

      id<MTLTexture> tex = NewTexture3D(
        device,
        fmtInfo.Format,
        static_cast<NSUInteger>(dims[0]),
        static_cast<NSUInteger>(dims[1]),
        static_cast<NSUInteger>(dims[2]),
        MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead,
        MTLStorageModePrivate,
        VolumeGPUOptimizedContents());
      if (!tex)
      {
        vtkErrorMacro("Failed to create 3D volume texture for GPU conversion");
        [srcBuf release];
        return false;
      }
      AssignMetalObject(this->VolumeTexture, tex);

      VolumeConvertUniforms vu;
      vu.dimX = static_cast<uint32_t>(dims[0]);
      vu.dimY = static_cast<uint32_t>(dims[1]);
      vu.dimZ = static_cast<uint32_t>(dims[2]);
      vu.numComponents = static_cast<uint32_t>(numComponents);
      vu.outputComponents = static_cast<uint32_t>(outputComponents);
      vu._pad = 0;

      id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
      cmdBuf.label = @"VTK Volume GPU Convert";
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      enc.label = @"Volume Convert";
      [enc setComputePipelineState:pipeline];
      [enc setBuffer:srcBuf offset:0 atIndex:0];
      [enc setTexture:tex atIndex:0];
      [enc setBytes:&vu length:sizeof(vu) atIndex:1];

      MTLSize grid = MTLSizeMake(dims[0], dims[1], dims[2]);
      MTLSize tg = MTLSizeMake(8, 8, 8);
      [enc dispatchThreads:grid threadsPerThreadgroup:tg];
      [enc endEncoding];
      [cmdBuf commit];

      [srcBuf release];
      gpuConversionUsed = true;
    }
  }

  // RG8 pair-pack repack (see VolumeRg8PairActive): interleave adjacent
  // slices into RG texel pairs over a halved-depth texture.
  bool rg8Pair = false;
  int upDims[3] = { dims[0], dims[1], dims[2] };
  MTLPixelFormat upFormat = fmtInfo.Format;
  NSUInteger upBytesPerRow = bytesPerRow;
  NSUInteger upBytesPerImage = bytesPerImage;
  std::vector<uint8_t> rg8Storage;
  if (!gpuConversionUsed && VolumeRg8PairActive() && !fmtInfo.NeedsConversion &&
      dataType == VTK_UNSIGNED_CHAR && numComponents == 1)
  {
    rg8Pair = true;
    upDims[2] = (dims[2] + 1) / 2;
    upFormat = MTLPixelFormatRG8Unorm;
    upBytesPerRow = static_cast<NSUInteger>(dims[0]) * 2;
    upBytesPerImage = upBytesPerRow * static_cast<NSUInteger>(dims[1]);
    const size_t sliceTexels = static_cast<size_t>(dims[0]) * dims[1];
    rg8Storage.resize(sliceTexels * static_cast<size_t>(upDims[2]) * 2);
    const uint8_t* in = static_cast<const uint8_t*>(scalars->GetVoidPointer(0));
    for (int p = 0; p < upDims[2]; ++p)
    {
      uint8_t* dst = rg8Storage.data() + sliceTexels * 2 * static_cast<size_t>(p);
      const uint8_t* sa = in + sliceTexels * static_cast<size_t>(2 * p);
      const uint8_t* sb = (2 * p + 1 < dims[2]) ? sa + sliceTexels : sa;
      for (size_t i = 0; i < sliceTexels; ++i)
      {
        dst[2 * i + 0] = sa[i];
        dst[2 * i + 1] = sb[i];
      }
    }
  }

  // Transposed repack (see VolumeTransposedActive / VolumeTransposedAxisDepth):
  // upload the volume with the chosen orientation so the SHORTEST array
  // dimension occupies texture depth. Metal's private 3D tiling is axis-biased;
  // with a long extent as texture depth, trilinear depth-pair fetches under
  // per-pixel jitter phase scatter pay a large DRAM tax (2026-08-22 root
  // cause). X-depth maps fetch coords via .zyx, Y-depth via .xzy
  // (fc_volTransposed / fc_volTransposedY).
  bool volTransposed = false;
  int volAxisDepth = 0;
  bool volTransposedGPU = false;
  std::vector<uint8_t> transStorage;
  if (!gpuConversionUsed && !rg8Pair && VolumeTransposedActive() &&
      dataType == VTK_UNSIGNED_CHAR && numComponents == 1)
  {
    const int trAxis = VolumeTransposedAxisDepth(dims);
    if (trAxis != 0)
    {
      volTransposed = true;
      volAxisDepth = trAxis;
      const NSUInteger elemSize = static_cast<NSUInteger>(fmtInfo.BytesPerComponent) *
        actualComponents;
      if (trAxis == 2)
      {
        // Y-depth: texture holds (W,D,H); original Y moves into depth.
        upDims[1] = dims[2];
        upDims[2] = dims[1];
        upBytesPerRow = static_cast<NSUInteger>(dims[0]) * elemSize;
        upBytesPerImage = upBytesPerRow * static_cast<NSUInteger>(dims[2]);
      }
      else
      {
        // X-depth: texture holds (D,H,W); the slice axis becomes the width.
        upDims[0] = dims[2];
        upDims[2] = dims[0];
        upBytesPerRow = static_cast<NSUInteger>(dims[2]) * elemSize;
        upBytesPerImage = upBytesPerRow * static_cast<NSUInteger>(dims[1]);
      }
      volTransposedGPU = VolumeTransposeGPU();
      if (volTransposedGPU)
      {
        // §28: staging stays in ORIGINAL layout; the compute kernel writes the
        // swapped-dims texture directly from it (no transStorage, no blit).
        fprintf(stderr, "[TR] direct path: transposed (axis %c) dims %dx%dx%d -> %dx%dx%d (GPU kernel)\n",
          trAxis == 2 ? 'y' : 'x', dims[0], dims[1], dims[2],
          upDims[0], upDims[1], upDims[2]);
      }
      else
      {
      transStorage.resize(totalBytes);
      const int BS = 32;
      const uint8_t* in = static_cast<const uint8_t*>(scalars->GetVoidPointer(0));
      uint8_t* out = transStorage.data();
      std::chrono::steady_clock::time_point trT0;
      if (getenv("VTK_METAL_TEST_TR_BENCH")) trT0 = std::chrono::steady_clock::now();
      for (int xb = 0; xb < dims[0]; xb += BS)
        for (int zb = 0; zb < dims[2]; zb += BS)
          for (int yb = 0; yb < dims[1]; yb += BS)
          {
            const int xe = std::min(xb + BS, dims[0]);
            const int ze = std::min(zb + BS, dims[2]);
            const int ye = std::min(yb + BS, dims[1]);
            for (int x = xb; x < xe; ++x)
              for (int z = zb; z < ze; ++z)
                for (int y = yb; y < ye; ++y)
                {
                  const size_t srcOff =
                    (((static_cast<size_t>(z) * dims[1] + y) * dims[0]) + x) * elemSize;
                  size_t dstOff;
                  if (trAxis == 2)
                  {
                    // T(u=x, v=z, w=y): dst[(y*Z + z)*X + x]
                    dstOff =
                      ((static_cast<size_t>(y) * dims[2] + z) * dims[0] + x) * elemSize;
                  }
                  else
                  {
                    // T(u=z, v=y, w=x): dst[(x*H + y)*Z + z]
                    dstOff =
                      ((static_cast<size_t>(x) * dims[1] + y) * dims[2] + z) * elemSize;
                  }
                  std::memcpy(out + dstOff, in + srcOff, elemSize);
                }
          }
      if (getenv("VTK_METAL_TEST_TR_BENCH"))
        fprintf(stderr, "[TRBENCH] cpu blocked transpose: %.1f ms (%zu bytes)\n",
          std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - trT0).count(),
          transStorage.size());
      }
    }
  }
  this->VolumeTextureTransposed = volTransposed;
  this->VolumeTextureAxisDepth = volAxisDepth;

  if (!gpuConversionUsed)
  {
    if (VolumeRg8PairActive())
      fprintf(stderr, "[RG8] direct path: rg8Pair=%d dims %dx%dx%d -> upDims %dx%dx%d fmt=%d staging=%zu\n",
        rg8Pair, dims[0], dims[1], dims[2], upDims[0], upDims[1], upDims[2],
        (int)upFormat, rg8Pair ? rg8Storage.size() : (size_t)totalBytes);
    id<MTLBuffer> stagingBuf =
      [device newBufferWithLength:(rg8Pair ? rg8Storage.size() : totalBytes)
                          options:MTLResourceStorageModeShared];
    if (!stagingBuf)
    {
      vtkErrorMacro("Failed to create volume staging buffer");
      return false;
    }
    void* uploadPointer = [stagingBuf contents];

    if (fmtInfo.NeedsConversion)
    {
      ConvertVolumeData(scalars->GetVoidPointer(0), dataType, numComponents,
        numTuples, uploadPointer, useHalf, actualComponents, scalars);
    }
    else if (dataType == VTK_FLOAT && numComponents == 3)
    {
      Expand3To4<float>(
        static_cast<const float*>(scalars->GetVoidPointer(0)),
        static_cast<float*>(uploadPointer),
        numTuples,
        0.0f);
    }
    else if (dataType == VTK_UNSIGNED_CHAR && numComponents == 3)
    {
      Expand3To4<unsigned char>(
        static_cast<const unsigned char*>(scalars->GetVoidPointer(0)),
        static_cast<unsigned char*>(uploadPointer),
        numTuples,
        255);
    }
    else if (dataType == VTK_UNSIGNED_SHORT && this->ScalarNormalizationFactor == 255.0f)
    {
      const unsigned short* src =
        static_cast<const unsigned short*>(scalars->GetVoidPointer(0));
      unsigned char* dst = static_cast<unsigned char*>(uploadPointer);
      vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
        for (vtkIdType i = begin; i < end; ++i)
        {
          for (int c = 0; c < numComponents; ++c)
            dst[i * actualComponents + c] = static_cast<unsigned char>(std::min<unsigned short>(src[i * numComponents + c], 255));
          if (numComponents == 3)
            dst[i * 4 + 3] = 255;
        }
      });
    }
    else if (dataType == VTK_UNSIGNED_SHORT && numComponents == 3)
    {
      Expand3To4<unsigned short>(
        static_cast<const unsigned short*>(scalars->GetVoidPointer(0)),
        static_cast<unsigned short*>(uploadPointer),
        numTuples,
        65535);
    }
    else if (dataType == VTK_SHORT && numComponents == 3)
    {
      // RGBA16Snorm alpha channel: 32767 is the largest positive snorm value
      // and normalizes to 1.0 (fully opaque), matching the Expand3To4 fillers
      // used for the other 16-bit formats.
      Expand3To4<short>(
        static_cast<const short*>(scalars->GetVoidPointer(0)),
        static_cast<short*>(uploadPointer),
        numTuples,
        32767);
    }
    else
    {
      if (rg8Pair)
      {
        std::memcpy(uploadPointer, rg8Storage.data(), rg8Storage.size());
      }
      else if (volTransposed && !volTransposedGPU)
      {
        std::memcpy(uploadPointer, transStorage.data(), transStorage.size());
      }
      else
      {
        std::memcpy(uploadPointer, scalars->GetVoidPointer(0), totalBytes);
      }
    }

    id<MTLTexture> oldTex = (__bridge id<MTLTexture>)this->VolumeTexture;
    id<MTLTexture> tex = nil;

    if (oldTex &&
        oldTex.width == static_cast<NSUInteger>(upDims[0]) &&
        oldTex.height == static_cast<NSUInteger>(upDims[1]) &&
        oldTex.depth == static_cast<NSUInteger>(upDims[2]) &&
        oldTex.pixelFormat == upFormat)
    {
      tex = oldTex;
    }
    else
    {
      ReleaseMetalObject(this->VolumeTexture);

      tex = NewTexture3D(
        device,
        upFormat,
        static_cast<NSUInteger>(upDims[0]),
        static_cast<NSUInteger>(upDims[1]),
        static_cast<NSUInteger>(upDims[2]),
        volTransposedGPU ? MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite
                         : MTLTextureUsageShaderRead,
        MTLStorageModePrivate,
        VolumeGPUOptimizedContents());
      if (!tex)
      {
        vtkErrorMacro("Failed to create 3D volume texture");
        [stagingBuf release];
        return false;
      }
      AssignMetalObject(this->VolumeTexture, tex);
    }

    id<MTLCommandBuffer> uploadCmdBuf = [queue commandBuffer];
    if (volTransposedGPU)
    {
      // §28 GPU transpose: no blit. One compute pass writes the swapped-dims
      // volume texture directly from the ORIGINAL-layout staging buffer.
      if (!this->EnsureShaderLibrary((__bridge void*)device))
      {
        vtkErrorMacro("GPU transpose: no shader library");
        [stagingBuf release];
        return false;
      }
      if (!this->TransposeComputePipeline)
      {
        id<MTLLibrary> library =
          (__bridge id<MTLLibrary>)this->CachedShaderLibrary;
        id<MTLFunction> func =
          [library newFunctionWithName:@"volume_transpose_xz"];
        NSError* error = nil;
        id<MTLComputePipelineState> pso = func
          ? [device newComputePipelineStateWithFunction:func error:&error]
          : nil;
        if (func) [func release];
        if (!pso)
        {
          vtkErrorMacro(<< "Failed to create transpose compute pipeline: "
                        << (error ? [[error localizedDescription] UTF8String]
                                  : "unknown"));
          [stagingBuf release];
          return false;
        }
        AssignMetalObject(this->TransposeComputePipeline, pso);
      }
      id<MTLComputePipelineState> pso =
        (__bridge id<MTLComputePipelineState>)this->TransposeComputePipeline;
      NSUInteger mtt = [pso maxTotalThreadsPerThreadgroup];
      MTLSize tg = mtt >= 512 ? MTLSizeMake(8, 8, 8) : MTLSizeMake(4, 4, 4);
      MTLSize grid = MTLSizeMake(
        (static_cast<NSUInteger>(dims[0]) + tg.width - 1) / tg.width,
        (static_cast<NSUInteger>(dims[1]) + tg.height - 1) / tg.height,
        (static_cast<NSUInteger>(dims[2]) + tg.depth - 1) / tg.depth);
      std::chrono::steady_clock::time_point trT0;
      bool trBench = getenv("VTK_METAL_TEST_TR_BENCH") != nullptr;
      if (trBench) trT0 = std::chrono::steady_clock::now();
      id<MTLComputeCommandEncoder> enc = [uploadCmdBuf computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:stagingBuf offset:0 atIndex:0];
      // dst is [[texture(1)]] in volume_transpose_xz.
      [enc setTexture:tex atIndex:1];
      // MSL uint3 occupies 16 bytes in the constant address space.
      struct { uint32_t x, y, z, pad; } srcDims = {
        static_cast<uint32_t>(dims[0]), static_cast<uint32_t>(dims[1]),
        static_cast<uint32_t>(dims[2]), 0 };
      [enc setBytes:&srcDims length:sizeof(srcDims) atIndex:2];
      // Orientation code for volume_transpose_xz: 1=X-depth, 2=Y-depth.
      uint32_t trMode = static_cast<uint32_t>(volAxisDepth);
      [enc setBytes:&trMode length:sizeof(trMode) atIndex:3];
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [enc endEncoding];
      [uploadCmdBuf commit];
      if (trBench)
      {
        [uploadCmdBuf waitUntilCompleted];
        fprintf(stderr,
          "[TRBENCH] gpu transpose (staging copy excluded): %.1f ms\n",
          std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - trT0).count());
      }
      [stagingBuf release];
      return true;
    }
    id<MTLBlitCommandEncoder> blit = [uploadCmdBuf blitCommandEncoder];
      [blit copyFromBuffer:stagingBuf
              sourceOffset:0
       sourceBytesPerRow:(rg8Pair || volTransposed ? upBytesPerRow : bytesPerRow)
     sourceBytesPerImage:(rg8Pair || volTransposed ? upBytesPerImage : bytesPerImage)
               sourceSize:MTLSizeMake(upDims[0], upDims[1], upDims[2])
             toTexture:tex
      destinationSlice:0
      destinationLevel:0
     destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    [uploadCmdBuf commit];
    [stagingBuf release];
  }

  return true;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::EnsureGridTraversalResources(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkImageData* input,
  vtkVolume* vol)
{
  if (this->GridTraversalResourcesValid)
    return;

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mtlQueueVoid;

  int nx = std::max(1, static_cast<int>(this->Partitions[0]));
  int ny = std::max(1, static_cast<int>(this->Partitions[1]));
  int nz = std::max(1, static_cast<int>(this->Partitions[2]));

  // --- occupancy texture: reuse if dimensions match ---
  id<MTLTexture> occTex = (__bridge id<MTLTexture>)this->OccupancyGridTexture;
  bool needNewOccTex = !occTex ||
      occTex.width  != static_cast<NSUInteger>(nx) ||
      occTex.height != static_cast<NSUInteger>(ny) ||
      occTex.depth  != static_cast<NSUInteger>(nz);

  if (needNewOccTex)
  {
    ReleaseMetalObject(this->OccupancyGridTexture);
    MTLTextureDescriptor* occDesc = [[MTLTextureDescriptor alloc] init];
    occDesc.textureType = MTLTextureType3D;
    occDesc.pixelFormat = MTLPixelFormatR8Unorm;
    occDesc.width = static_cast<NSUInteger>(nx);
    occDesc.height = static_cast<NSUInteger>(ny);
    occDesc.depth = static_cast<NSUInteger>(nz);
    occDesc.usage = MTLTextureUsageShaderRead;
    occDesc.storageMode = MTLStorageModePrivate;
    occTex = [device newTextureWithDescriptor:occDesc];
    [occDesc release];
    if (!occTex)
    {
      vtkErrorMacro("Failed to create grid traversal occupancy texture");
      return;
    }
    AssignMetalObject(this->OccupancyGridTexture, occTex);
  }

  // Build occupancy data from partition grid and opacity function.
  // All bricks are active (255) when no opacity function or macrocell data.
  int gridCells = nx * ny * nz;
  std::vector<uint8_t> occupancy(gridCells, 255);

  vtkPiecewiseFunction* opacityFunc = nullptr;
  if (vol)
  {
    vtkVolumeProperty* property = vol->GetProperty();
    opacityFunc = property ? property->GetScalarOpacity() : nullptr;
  }

  if (opacityFunc &&
      !this->MacrocellScalarMin.empty() &&
      !this->MacrocellScalarMax.empty())
  {
    double opacityTable[256];
    opacityFunc->GetTable(
      this->ScalarRange[0],
      this->ScalarRange[1],
      256,
      opacityTable);

    const double scalarRange =
      this->ScalarRange[1] - this->ScalarRange[0];

    const double rangeRecip =
      (scalarRange > 0.0) ? (255.0 / scalarRange) : 1.0;

    const double rangeOffset = this->ScalarRange[0];

    int fullExt[6];
    input->GetExtent(fullExt);

    const int mcDims0 = this->MinMaxDims[0];
    const int mcDims1 = this->MinMaxDims[1];
    const int mcDims2 = this->MinMaxDims[2];

    const int DS = ComputeMacrocellDownsample(this->SampleDistance, this->UseGPUMinMax);

    const int fullX = fullExt[1] - fullExt[0] + 1;
    const int fullY = fullExt[3] - fullExt[2] + 1;
    const int fullZ = fullExt[5] - fullExt[4] + 1;

    // Convert to DilateOccupancy3D convention: 0 = active, 255 = empty.
    // We compute raw (undilated) occupancy, then dilate for conservative
    // empty-space skipping.
    std::vector<uint8_t> raw(gridCells, 255);

    auto brickRange = [](int i, int n, int full) -> std::pair<int, int>
    {
      if (n <= 0 || full <= 0) return { 0, -1 };
      vtkIdType a = (static_cast<vtkIdType>(i) * full) / n;
      vtkIdType b = (static_cast<vtkIdType>(i + 1) * full) / n - 1;
      return { static_cast<int>(a), static_cast<int>(b) };
    };

    for (int k = 0; k < nz; ++k)
    {
      for (int j = 0; j < ny; ++j)
      {
        for (int i = 0; i < nx; ++i)
        {
          auto [relX0, relX1] = brickRange(i, nx, fullX);
          auto [relY0, relY1] = brickRange(j, ny, fullY);
          auto [relZ0, relZ1] = brickRange(k, nz, fullZ);

          if (relX1 < relX0 || relY1 < relY0 || relZ1 < relZ0)
          {
            continue; // stays 255 (empty)
          }

          int mcX0 = std::max(0, relX0 / DS);
          int mcY0 = std::max(0, relY0 / DS);
          int mcZ0 = std::max(0, relZ0 / DS);

          int mcX1 = std::min(relX1 / DS, mcDims0 - 1);
          int mcY1 = std::min(relY1 / DS, mcDims1 - 1);
          int mcZ1 = std::min(relZ1 / DS, mcDims2 - 1);

          if (mcX1 < mcX0 || mcY1 < mcY0 || mcZ1 < mcZ0)
          {
            continue; // stays 255 (empty)
          }

          float brickMin = 1e30f;
          float brickMax = -1e30f;

          for (int mz = mcZ0; mz <= mcZ1; ++mz)
          {
            for (int my = mcY0; my <= mcY1; ++my)
            {
              for (int mx = mcX0; mx <= mcX1; ++mx)
              {
                vtkIdType ci =
                  (static_cast<vtkIdType>(mz) * mcDims1 + my) *
                  mcDims0 + mx;

                brickMin = std::min(
                  brickMin,
                  this->MacrocellScalarMin[ci]);

                brickMax = std::max(
                  brickMax,
                  this->MacrocellScalarMax[ci]);
              }
            }
          }

          bool active = ScalarRangeTouchesOpacity(
            brickMin,
            brickMax,
            opacityTable,
            rangeOffset,
            rangeRecip);

          size_t idx =
            (static_cast<size_t>(k) * ny +
             static_cast<size_t>(j)) * nx +
            static_cast<size_t>(i);

          // DilateOccupancy3D convention: 0 = active, 255 = empty
          raw[idx] = active ? 0 : 255;
        }
      }
    }

    // Dilate to avoid boundary artifacts from interpolation near brick edges.
    std::vector<uint8_t> dilated = DilateOccupancy3D(raw, nx, ny, nz);

    // Convert back: 0 in dilated -> 255 in occupancy (active)
    for (size_t i = 0; i < dilated.size(); ++i)
    {
      occupancy[i] = (dilated[i] == 0) ? 255 : 0;
    }
  }

  // Upload on the main queue so the blit is ordered before the render pass
  NSUInteger occBytesPerRow = static_cast<NSUInteger>(nx);
  NSUInteger occBytesPerImage = occBytesPerRow * ny;
  id<MTLBuffer> occStaging = [device newBufferWithBytes:occupancy.data()
                                                  length:occBytesPerImage * nz
                                                 options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> uploadCmdBuf = [queue commandBuffer];
  uploadCmdBuf.label = @"VTK Grid Traversal Occupancy Upload";
  id<MTLBlitCommandEncoder> blit = [uploadCmdBuf blitCommandEncoder];
  [blit copyFromBuffer:occStaging
          sourceOffset:0
   sourceBytesPerRow:occBytesPerRow
 sourceBytesPerImage:occBytesPerImage
          sourceSize:MTLSizeMake(nx, ny, nz)
           toTexture:occTex
    destinationSlice:0
    destinationLevel:0
   destinationOrigin:MTLOriginMake(0, 0, 0)];
  [blit endEncoding];
  [uploadCmdBuf commit];
  [occStaging release];

  // --- grid traversal uniform buffer: rebuild only if dimensions changed ---
  bool needNewUniforms = !this->GridTraversalUniformBuffer ||
    this->CachedGridDims[0] != nx ||
    this->CachedGridDims[1] != ny ||
    this->CachedGridDims[2] != nz;

  if (needNewUniforms)
  {
    GridTraversalUniforms gridUniforms;
    gridUniforms.GridDimsX = static_cast<int32_t>(nx);
    gridUniforms.GridDimsY = static_cast<int32_t>(ny);
    gridUniforms.GridDimsZ = static_cast<int32_t>(nz);
    gridUniforms._pad = 0;

    ReleaseMetalObject(this->GridTraversalUniformBuffer);
    id<MTLBuffer> gridBuf = [device newBufferWithBytes:&gridUniforms
                                                length:sizeof(GridTraversalUniforms)
                                               options:MTLResourceStorageModeShared];
    AssignMetalObject(this->GridTraversalUniformBuffer, gridBuf);

    this->CachedGridDims[0] = nx;
    this->CachedGridDims[1] = ny;
    this->CachedGridDims[2] = nz;
  }

  this->GridTraversalResourcesValid = true;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ReleaseGradientNormalTexture()
{
  ReleaseMetalObject(this->GradientNormalTexture);
  ReleaseMetalObject(this->NormalComputePipeline);
  this->NormalTextureDims[0] = 0;
  this->NormalTextureDims[1] = 0;
  this->NormalTextureDims[2] = 0;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::EnsureGradientNormalTexture(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkVolume* vol)
{
  if (!this->VolumeTexture)
  {
    return false;
  }

  if (!this->UsePrecomputedNormals)
  {
    this->ReleaseGradientNormalTexture();
    return false;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mtlQueueVoid;

  id<MTLTexture> volTex = (__bridge id<MTLTexture>)this->VolumeTexture;
  int dims[3] = { static_cast<int>(volTex.width), static_cast<int>(volTex.height), static_cast<int>(volTex.depth) };

  // Reuse if still valid — data, scalar range, and params haven't changed
  id<MTLTexture> oldTex = (__bridge id<MTLTexture>)this->GradientNormalTexture;
  bool stale = !oldTex ||
    static_cast<int>(oldTex.width) != dims[0] ||
    static_cast<int>(oldTex.height) != dims[1] ||
    static_cast<int>(oldTex.depth) != dims[2] ||
    this->VolumeUploadTime.GetMTime() > this->NormalTextureTime.GetMTime() ||
    this->GetMTime() > this->NormalTextureTime.GetMTime();

  if (!stale)
  {
    return true;
  }

  this->ReleaseGradientNormalTexture();

  if (!this->EnsureShaderLibrary(mtlDeviceVoid))
  {
    return false;
  }
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary;

  @autoreleasepool
  {
    // Create 3D normal texture (RGBA8Unorm: normal.xyz*0.5+0.5 in RGB, gradMag in A)
    id<MTLTexture> normalTex = NewTexture3D(
      device,
      MTLPixelFormatRGBA8Unorm,
      static_cast<NSUInteger>(dims[0]),
      static_cast<NSUInteger>(dims[1]),
      static_cast<NSUInteger>(dims[2]),
      MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
      MTLStorageModePrivate);
    if (!normalTex)
    {
      vtkErrorMacro("Failed to create gradient normal texture");
      return false;
    }
    AssignMetalObject(this->GradientNormalTexture, normalTex);
    this->NormalTextureDims[0] = dims[0];
    this->NormalTextureDims[1] = dims[1];
    this->NormalTextureDims[2] = dims[2];

    // Create compute pipeline state (cache it)
    if (!this->NormalComputePipeline)
    {
      id<MTLFunction> kernelFunc = [library newFunctionWithName:@"volume_compute_normals"];
      if (!kernelFunc)
      {
        vtkErrorMacro("Failed to find volume_compute_normals kernel");
        this->ReleaseGradientNormalTexture();
        return false;
      }

      NSError* error = nil;
      id<MTLComputePipelineState> cps =
        [device newComputePipelineStateWithFunction:kernelFunc error:&error];
      [kernelFunc release];
      if (!cps)
      {
        vtkErrorMacro(<< "Failed to create normal compute pipeline: "
                      << [[error localizedDescription] UTF8String]);
        this->ReleaseGradientNormalTexture();
        return false;
      }
      AssignMetalObject(this->NormalComputePipeline, cps);
    }

    // Build uniforms and dispatch compute using shared helpers
    if (!this->EnsureEffectiveInput())
    {
      return false;
    }
    vtkImageData* input = this->EffectiveInput;
    VolumeBounds vb = input ? ComputeVolumeBounds(input, this->CellFlag == 1) : VolumeBounds{};
    if (!input)
    {
      for (int k = 0; k < 3; ++k)
        vb.Size[k] = 1.0;
    }
    double cellSpacing[3] = {1.0, 1.0, 1.0};
    if (input)
    {
      input->GetSpacing(cellSpacing);
    }
    double avgSpacing =
      (fabs(cellSpacing[0]) + fabs(cellSpacing[1]) + fabs(cellSpacing[2])) / 3.0;
    NormalComputeUniforms u = MakeNormalComputeUniforms(
      dims, this->ScalarRange, this->ScalarNormalizationFactor, vb.Size, avgSpacing);
    // Orientation code for the data-space sampling kernels: 0=identity,
    // 1=X-depth (.zyx), 2=Y-depth (.xzy).
    u.volTransposed = this->VolumeTextureAxisDepth;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    cmdBuf.label = @"VTK Volume Normal Compute";

    id<MTLComputeCommandEncoder> compEnc = [cmdBuf computeCommandEncoder];
    EncodeNormalCompute(compEnc, volTex, normalTex, u,
      (__bridge id<MTLComputePipelineState>)this->NormalComputePipeline);
    [compEnc endEncoding];
    [cmdBuf commit];

    this->NormalTextureTime.Modified();
  }

  return this->GradientNormalTexture != nullptr;
}

//------------------------------------------------------------------------------
// §38.18.1: release all segment-pre-pass Private heaps and invalidate the
// per-camera cache. This is the heavyweight half of PurgeCaches and is also
// called directly by ReleaseGraphicsResources so destructors do not leak the
// 64 MB SegPool / RayBin Private heaps that otherwise survive until
// MTLDevice teardown (reboot-only clog, not DVFS).
void vtkMetalGPUVolumeRayCastMapper::ReleaseSegmentResources()
{
  ReleaseMetalObject(this->SegAtlasATexture);
  ReleaseMetalObject(this->SegAtlasBTexture);
  ReleaseMetalObject(this->SegAtlasCTexture);
  ReleaseMetalObject(this->SegIndexBuffer);
  ReleaseMetalObject(this->SegPoolBuffer);
  ReleaseMetalObject(this->SegPoolCounterBuffer);
  ReleaseMetalObject(this->SegMarchTexture);
  ReleaseMetalObject(this->SegBuildComputePipeline);
  ReleaseMetalObject(this->SegConsumeDbgBuffer);
  ReleaseMetalObject(this->SegDebugStageBuffer);
  // SegDummyBuffer is a tiny Shared fallback kept across purges to avoid
  // re-allocating per frame; do not release it here.
  this->SegAtlasWidth = 0;
  this->SegAtlasHeight = 0;
  this->SegPoolCapWords = 0;
  this->SegCacheValid = false;
  this->SegCacheWidth = 0;
  this->SegCacheHeight = 0;
  this->SegCachePoolCapWords = 0;
  this->SegCacheUniformBytes.clear();
  this->SegCacheUniformBytes.shrink_to_fit();
  this->SegCachePbdBytes.clear();
  this->SegCachePbdBytes.shrink_to_fit();
  this->SegCacheMinMaxTime.Modified();
  this->SegActiveThisFrame = false;
  this->SegDebugStageBytes = 0;
  this->SegLastClaimWords.store(0);
}

// §38.18.1: "reboot without reboot" — drains in-flight frames, releases
// Private-heap SegPool/RayBin (64 MB + 4×W×H) and the Compile-time PSO caches
// (ComputeMarchPipelineCache + SegBuild) that otherwise survive until
// ReleaseGraphicsResources/MTLDevice teardown. Callable from
// ReleaseGraphicsResources and on VTK_METAL_TEST_PURGE=1 or pool-cap thrash.
// Better alternatives evaluated (see header docs):
//  • Shared-storage fallback for RayBin/SegPool (avoids Private-heap pressure
//    entirely but costs GPU atomic bandwidth);
//  • MTLHeap + setPurgeableState / currentAllocatedSize vs
//    recommendedMaxWorkingSetSize auto-purge (lets the OS reclaim under pressure
//    without explicit PurgeCaches);
//  • LRU cap on PSO cache (e.g. 4 entries, ~256 MB observed) instead of
//    unbounded grow. The explicit purge is the minimal correctness fix; the
//    heap/purgeable and LRU options remain as follow-ups if the clog recurs.
void vtkMetalGPUVolumeRayCastMapper::PurgeCaches()
{
  // Drain GPU work that may reference the heaps/PSOs before releasing.
  this->WaitForInFlightFrames();

  // Compute PSO caches (256 MB observed with many feature-mask variants).
  for (auto& entry : this->ComputeMarchPipelineCache)
  {
    [(__bridge id)entry.second release];
  }
  this->ComputeMarchPipelineCache.clear();
  for (auto& entry : this->ComputeMarchBinnedPipelineCache)
  {
    [(__bridge id)entry.second release];
  }
  this->ComputeMarchBinnedPipelineCache.clear();
  ReleaseMetalObject(this->ComputeMarchPipeline);
  ReleaseMetalObject(this->RayBinClassifyPipeline);
  ReleaseMetalObject(this->ComputeMarchBinnedPipeline);
  ReleaseMetalObject(this->RayBinIndicesBuffer);
  ReleaseMetalObject(this->RayBinCountersBuffer);
  ReleaseMetalObject(this->ComputeMarchQueue);
  this->RayBinIndicesCapBytes = 0;

  // Cinematic heaps + PSO caches (shaded DVR single variant).
  this->ReleaseCinematicResources();

  // Segment heaps + per-camera cache.
  this->ReleaseSegmentResources();

  if (getenv("VTK_METAL_TEST_PURGE") || getenv("VTK_METAL_TEST_METAL_PURGE"))
  {
    fprintf(stderr, "[purge] vtkMetalGPUVolumeRayCastMapper::PurgeCaches completed\n");
  }
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ReleaseGraphicsResources(vtkWindow* vtkNotUsed(window))
{
  // Drain in-flight frames before releasing any resources they may reference.
  // Completion handlers fire on a libdispatch queue independent of the render
  // thread, so the 3 waits won't deadlock even if the render thread is the
  // caller. The destructor also calls WaitForInFlightFrames before calling us,
  // which is harmless — the second pass is 3 waits + 3 signals on an already-
  // drained semaphore, net zero.
  this->WaitForInFlightFrames();
  this->ReleaseImageSampleResources();
  this->ReleaseSlabResources();
  this->ReleaseRTTResources();

  ReleaseMetalObject(this->PipelineState);
  ReleaseMetalObject(this->VolumeTexture);
  ReleaseMetalObject(this->ColorOpacityTexture);
  ReleaseMetalObject(this->ComponentTransferFunctionTexture1);
  ReleaseMetalObject(this->ComponentTransferFunctionTexture2);
  ReleaseMetalObject(this->ComponentTransferFunctionTexture3);
  this->LastIndependentComponents = false;
  ReleaseMetalObject(this->GradientOpacityTexture);
  ReleaseMetalObject(this->Transfer2DTexture);
  ReleaseMetalObject(this->Transfer2DYAxisTexture);
  this->Transfer2DEnabled = false;
  this->Transfer2DUseGradient = false;
  ReleaseMetalObject(this->MinMaxTexture);
  ReleaseMetalObject(this->MinMaxScratchTexture);
  ReleaseMetalObject(this->MinMaxBlockTexture);
  ReleaseMetalObject(this->MinMaxSuperTexture);
  ReleaseMetalObject(this->MinMaxCountBuffer);
  this->ReleaseGradientNormalTexture();

  this->ReleaseMaskResources();

  ReleaseMetalObject(this->BlankingTexture);
  this->BlankingPoints = nullptr;
  this->BlankingCells = nullptr;

  // Phase 5: Release GPU min-max compute pipelines
  ReleaseMetalObject(this->MinMaxComputePipeline);
  ReleaseMetalObject(this->DilateComputePipeline);
  ReleaseMetalObject(this->BlockReduceComputePipeline);
  ReleaseMetalObject(this->MipBlockReduceComputePipeline);
  // SegDebugStageBuffer is owned by segment resources; released via
  // ReleaseSegmentResources below to keep Private-heap purge centralized.

  // Phase 7: Release GPU data-type conversion compute pipelines
  ReleaseMetalObject(this->ConvertShortToHalfPipeline);
  ReleaseMetalObject(this->ConvertShortToFloatPipeline);
  ReleaseMetalObject(this->ConvertIntToHalfPipeline);
  ReleaseMetalObject(this->ConvertIntToFloatPipeline);
  ReleaseMetalObject(this->ConvertUIntToHalfPipeline);
  ReleaseMetalObject(this->ConvertUIntToFloatPipeline);
  ReleaseMetalObject(this->ConvertFloatToHalfPipeline);
  ReleaseMetalObject(this->ConvertUShortToUCharPipeline);
  ReleaseMetalObject(this->DummyDepthTexture);
  ReleaseMetalObject(this->DummyVolumeTexture);
  ReleaseMetalObject(this->DummyMaskTexture);
  ReleaseMetalObject(this->DummyMinMaxTexture);
  ReleaseMetalObject(this->DepthStencilState);

  // Phase 1A: Release cached shader library
  ReleaseMetalObject(this->CachedShaderLibrary);

  // Phase 1C: Reset pipeline pre-warm guard so it re-warms after device loss / resize
  this->PipelinesPreWarmed = false;

  // Phase 1B: Clear pipeline cache
  for (auto& entry : this->PipelineCache)
  {
    [(__bridge id)entry.second release];
  }
  this->PipelineCache.clear();

  // §38.6 / §36.4 Design B: Clear compute march pipelines and buffers
  // (also covered by PurgeCaches — keep inline here to avoid an extra
  // WaitForInFlightFrames when called from ReleaseGraphicsResources itself).
  ReleaseMetalObject(this->ComputeMarchPipeline);
  ReleaseMetalObject(this->RayBinClassifyPipeline);
  ReleaseMetalObject(this->ComputeMarchBinnedPipeline);
  ReleaseMetalObject(this->RayBinIndicesBuffer);
  ReleaseMetalObject(this->RayBinCountersBuffer);
  ReleaseMetalObject(this->ComputeMarchQueue);
  this->RayBinIndicesCapBytes = 0;
  for (auto& entry : this->ComputeMarchPipelineCache)
  {
    [(__bridge id)entry.second release];
  }
  this->ComputeMarchPipelineCache.clear();
  for (auto& entry : this->ComputeMarchBinnedPipelineCache)
  {
    [(__bridge id)entry.second release];
  }
  this->ComputeMarchBinnedPipelineCache.clear();

  // Cinematic variant resources (shaded DVR single).
  this->ReleaseCinematicResources();

  // §38.18.1: release segment Private heaps (SegPool 64 MB etc.) that were
  // previously leaked across ReleaseGraphicsResources — the reboot-only clog
  // root cause.
  this->ReleaseSegmentResources();

  for (int i = 0; i < 3; ++i)
  {
    ReleaseMetalObject(this->UniformBuffers[i]);
  }

  this->ReleaseGridTraversalResources();

  ReleaseMetalObject(this->VertexBuffer);
  ReleaseMetalObject(this->IndexBuffer);
  ReleaseMetalObject(this->RectCoordsBuffer);
  ReleaseMetalObject(this->DummyRectCoordsBuffer);
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::GetReductionRatio(double ratio[3])
{
  // Image-space downsampling reduces the rendering resolution
  double imageRatio = 1.0 / this->ImageSampleDistance;
  ratio[0] = imageRatio;
  ratio[1] = imageRatio;
  ratio[2] = 1.0;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::PreRender(vtkRenderer* vtkNotUsed(ren),
  vtkVolume* vtkNotUsed(vol), double vtkNotUsed(datasetBounds)[6],
  double vtkNotUsed(scalarRange)[2], int vtkNotUsed(noOfComponents),
  unsigned int vtkNotUsed(numberOfLevels))
{
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::RenderBlock(vtkRenderer* vtkNotUsed(ren),
  vtkVolume* vtkNotUsed(vol), unsigned int vtkNotUsed(level))
{
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::PostRender(vtkRenderer* vtkNotUsed(ren),
  int vtkNotUsed(numberOfScalarComponents))
{
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::GetColorImage(vtkImageData* output)
{
  if (!output || !this->RTTColorTexture || this->RTTWidth <= 0 || this->RTTHeight <= 0)
  {
    vtkErrorMacro("GetColorImage early return: tex=" << (this->RTTColorTexture ? "set" : "null")
      << " w=" << this->RTTWidth << " h=" << this->RTTHeight);
    return;
  }

  // The RTT textures are written by the most recent render pass; drain the
  // in-flight frame semaphore so the GPU work is guaranteed complete before the
  // synchronous CPU readback below.
  this->WaitForInFlightFrames();

  @autoreleasepool
  {
    id<MTLTexture> tex = (__bridge id<MTLTexture>)this->RTTColorTexture;
    int w = this->RTTWidth;
    int h = this->RTTHeight;

    output->SetDimensions(w, h, 1);
    output->SetExtent(0, w - 1, 0, h - 1, 0, 0);
    output->SetOrigin(0.0, 0.0, 0.0);
    output->SetSpacing(1.0, 1.0, 1.0);
    output->AllocateScalars(VTK_UNSIGNED_CHAR, 4);

    unsigned char* ptr = static_cast<unsigned char*>(output->GetScalarPointer());
    const NSUInteger bytesPerRow = static_cast<NSUInteger>(w) * 4; // RGBA8Unorm
    void* tmp = malloc(bytesPerRow * static_cast<NSUInteger>(h));
    if (!tmp)
    {
      return;
    }

    [tex getBytes:tmp bytesPerRow:bytesPerRow
        bytesPerImage:bytesPerRow * static_cast<NSUInteger>(h)
           fromRegion:MTLRegionMake2D(0, 0, w, h)
          mipmapLevel:0 slice:0];

    unsigned char* src = static_cast<unsigned char*>(tmp);
    // Metal getBytes returns the top texture row first; vtkImageData expects
    // the first row at the bottom of the image (matching OpenGL readback), so
    // flip rows vertically. The RGBA8Unorm bytes are already 0-255.
    for (int y = 0; y < h; ++y)
    {
      const unsigned char* s = src + 4 * (y * w);
      const int dstRow = (h - 1 - y);
      unsigned char* o = ptr + 4 * (dstRow * w);
      std::memcpy(o, s, 4 * static_cast<NSUInteger>(w));
    }
  }
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::GetDepthImage(vtkImageData* output)
{
  if (!output || !this->RTTDepthTexture || this->RTTWidth <= 0 || this->RTTHeight <= 0)
  {
    return;
  }

  this->WaitForInFlightFrames();

  @autoreleasepool
  {
    id<MTLTexture> tex = (__bridge id<MTLTexture>)this->RTTDepthTexture;
    int w = this->RTTWidth;
    int h = this->RTTHeight;

    output->SetDimensions(w, h, 1);
    output->SetExtent(0, w - 1, 0, h - 1, 0, 0);
    output->SetOrigin(0.0, 0.0, 0.0);
    output->SetSpacing(1.0, 1.0, 1.0);

    int scalarType = this->RTTDepthScalarType;
    if (scalarType != VTK_FLOAT && scalarType != VTK_UNSIGNED_CHAR && scalarType != VTK_UNSIGNED_SHORT)
    {
      scalarType = VTK_FLOAT;
    }
    output->AllocateScalars(scalarType, 1);
    void* ptr = output->GetScalarPointer();
    const NSUInteger bytesPerRow = static_cast<NSUInteger>(w) * 4; // R32Float
    float* tmp = static_cast<float*>(malloc(bytesPerRow * static_cast<NSUInteger>(h)));
    if (!tmp)
    {
      return;
    }

    [tex getBytes:tmp bytesPerRow:bytesPerRow
        bytesPerImage:bytesPerRow * static_cast<NSUInteger>(h)
           fromRegion:MTLRegionMake2D(0, 0, w, h)
          mipmapLevel:0 slice:0];

    // Metal getBytes returns the top texture row first; vtkImageData expects
    // the first row at the bottom of the image, so flip rows vertically.
    for (int y = 0; y < h; ++y)
    {
      const float* s = tmp + y * w;
      const int dstRow = (h - 1 - y);
      if (scalarType == VTK_UNSIGNED_CHAR)
      {
        unsigned char* out = static_cast<unsigned char*>(ptr) + dstRow * w;
        for (int x = 0; x < w; ++x)
        {
          float v = (s[x] < 0.0f) ? 0.0f : ((s[x] > 1.0f) ? 1.0f : s[x]);
          out[x] = static_cast<unsigned char>(v * 255.0f + 0.5f);
        }
      }
      else if (scalarType == VTK_UNSIGNED_SHORT)
      {
        unsigned short* out = static_cast<unsigned short*>(ptr) + dstRow * w;
        for (int x = 0; x < w; ++x)
        {
          float v = (s[x] < 0.0f) ? 0.0f : ((s[x] > 1.0f) ? 1.0f : s[x]);
          out[x] = static_cast<unsigned short>(v * 65535.0f + 0.5f);
        }
      }
      else
      {
        float* out = static_cast<float*>(ptr) + dstRow * w;
        for (int x = 0; x < w; ++x)
        {
          float v = (s[x] < 0.0f) ? 0.0f : ((s[x] > 1.0f) ? 1.0f : s[x]);
          out[x] = v;
        }
      }
    }

    free(tmp);
  }
}

//------------------------------------------------------------------------------
// Effective-input abstraction.
//
// The Metal pipeline only understands vtkImageData with point scalars. This
// method builds an equivalent vtkImageData from:
//   - vtkRectilinearGrid point scalars: uniform-spacing proxy that reproduces
//     the grid bounds and keeps the scalar values at the same logical indices.
//   - Cell scalars (image data or rectilinear grid): a proxy with dimensions
//     equal to the cell count and origin shifted by half a cell so that the
//     shader's point-data texel-center convention samples the cell values at
//     the correct world positions (matching vtkVolumeTexture::AdjustExtentForCell
//     plus the identity cell-to-point matrix used for cell data in OpenGL).
//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::EnsureEffectiveInput()
{
  vtkDataSet* dataSet = this->TransformedInputs[0];
  if (!dataSet)
  {
    this->EffectiveInput = nullptr;
    return false;
  }

  vtkImageData* img = vtkImageData::SafeDownCast(dataSet);
  vtkRectilinearGrid* rGrid = vtkRectilinearGrid::SafeDownCast(dataSet);

  // Point data on an image data input can be used directly.
  if (img && !this->CellFlag)
  {
    // Never replace an existing effective input built from this dataset (e.g. a
    // cell-to-point proxy) with the raw image: that would drop the only
    // reference to the proxy and free it while callers may still hold a raw
    // pointer to it (use-after-free). Keep the existing input when it is valid
    // for the same source data.
    if (this->EffectiveInput && this->EffectiveInputSource == dataSet &&
      dataSet->GetMTime() <= this->EffectiveInputTime.GetMTime())
    {
      return true;
    }
    if (this->EffectiveInput != img)
    {
      this->EffectiveInput = img;
      this->EffectiveInputTime.Modified();
    }
    this->RectilinearInput = false;
    return true;
  }

  // Reuse the proxy when the source data has not changed.
  if (this->EffectiveInput && this->EffectiveInputSource == dataSet &&
    dataSet->GetMTime() <= this->EffectiveInputTime.GetMTime())
  {
    return true;
  }

  vtkDataArray* scalars = nullptr;
  if (!this->CellFlag)
  {
    scalars = dataSet->GetPointData()->GetScalars();
  }
  else
  {
    scalars = dataSet->GetCellData()->GetScalars();
  }
  if (!scalars)
  {
    return false;
  }

  vtkNew<vtkImageData> proxy;

  if (img && this->CellFlag)
  {
    int pdims[3];
    img->GetDimensions(pdims);
    int cdims[3] = { std::max(1, pdims[0] - 1), std::max(1, pdims[1] - 1),
      std::max(1, pdims[2] - 1) };
    double spacing[3], origin[3];
    img->GetSpacing(spacing);
    img->GetOrigin(origin);
    proxy->SetDimensions(cdims);
    // Shift the origin to the center of cell (0,0,0), rotating the half-cell
    // offset through the direction matrix so the cell-data proxy lands on the
    // same physical positions as the source cell centers.
    vtkMatrix3x3* dirMat = img->GetDirectionMatrix();
    double shiftedOrigin[3] = { 0.0, 0.0, 0.0 };
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        shiftedOrigin[i] += 0.5 * dirMat->GetElement(i, j) * spacing[j];
      }
      shiftedOrigin[i] += origin[i];
    }
    proxy->SetOrigin(shiftedOrigin);
    proxy->SetSpacing(spacing);
    proxy->SetDirectionMatrix(img->GetDirectionMatrix());
    proxy->GetPointData()->SetScalars(scalars);
    proxy->Modified();
    this->RectilinearInput = false;
  }
  else if (rGrid)
  {
    double bounds[6];
    rGrid->GetBounds(bounds);
    int dims[3];
    rGrid->GetDimensions(dims);

    if (!this->CellFlag)
    {
      // Point data: uniform-spacing proxy with identical bounds.
      double spacing[3];
      for (int i = 0; i < 3; ++i)
      {
        spacing[i] = (dims[i] > 1) ? (bounds[2 * i + 1] - bounds[2 * i]) / (dims[i] - 1) : 1.0;
      }
      proxy->SetDimensions(dims);
      proxy->SetOrigin(bounds[0], bounds[2], bounds[4]);
      proxy->SetSpacing(spacing);
      proxy->GetPointData()->SetScalars(scalars);
      proxy->Modified();
    }
    else
    {
      // Cell data: proxy sized to the cell count, shifted by half a cell.
      int cdims[3] = { std::max(1, dims[0] - 1), std::max(1, dims[1] - 1),
        std::max(1, dims[2] - 1) };
      double origin[3], spacing[3];
      for (int i = 0; i < 3; ++i)
      {
        double cellSize =
          (cdims[i] > 1) ? (bounds[2 * i + 1] - bounds[2 * i]) / cdims[i] : 1.0;
        origin[i] = bounds[2 * i] + 0.5 * cellSize;
        spacing[i] = cellSize;
      }
      proxy->SetDimensions(cdims);
      proxy->SetOrigin(origin);
      proxy->SetSpacing(spacing);
      proxy->GetPointData()->SetScalars(scalars);
      proxy->Modified();
    }

    // Rectilinear coordinate curves for the index-space remap (OpenGL
    // in_coordTexs / in_coordsScale / in_coordsBias parity): each axis is
    // normalized via the same GetScaleAndBias used by vtkVolumeTexture (so the
    // shader walk happens in [0,1] normalized coordinate space) and packed into
    // a float3-per-index array padded to the longest axis, matching the 1D
    // float3 coord texture GL uploads.
    this->RectilinearInput = true;
    vtkDataArray* axisCoords[3] = { rGrid->GetXCoordinates(), rGrid->GetYCoordinates(),
      rGrid->GetZCoordinates() };
    int sizes[3];
    for (int a = 0; a < 3; ++a)
    {
      sizes[a] = (axisCoords[a] && axisCoords[a]->GetNumberOfTuples() > 0)
        ? static_cast<int>(axisCoords[a]->GetNumberOfTuples())
        : 1;
    }
    int maxSize = std::max(sizes[0], std::max(sizes[1], sizes[2]));
    this->RectCoordsData.assign(static_cast<size_t>(maxSize) * 3, 0.0f);
    for (int a = 0; a < 3; ++a)
    {
      const double* r = axisCoords[a] ? axisCoords[a]->GetFiniteRange(0) : nullptr;
      float fRange[2] = { 0.0f, 1.0f };
      if (r)
      {
        fRange[0] = static_cast<float>(r[0]);
        fRange[1] = static_cast<float>(r[1]);
      }
      if (fRange[1] == fRange[0])
      {
        fRange[1] = fRange[0] + 1e-6f;
      }
      float scale = 1.0f / (fRange[1] - fRange[0]);
      float bias = -fRange[0] * scale;
      this->RectCoordsScale[a] = scale;
      this->RectCoordsBias[a] = bias;
      this->RectCoordsSizes[a] = static_cast<float>(sizes[a]);
      for (int i = 0; i < sizes[a]; ++i)
      {
        this->RectCoordsData[static_cast<size_t>(i) * 3 + a] =
          static_cast<float>(axisCoords[a]->GetTuple1(i)) * scale + bias;
      }
    }
  }
  else
  {
    return false;
  }

  this->EffectiveInput = proxy;
  this->EffectiveInputSource = dataSet;
  this->EffectiveInputTime.Modified();
  return this->EffectiveInput != nullptr;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ForceResourceReupload()
{
  // Null the derived GPU resources; every consumer re-creates them on demand
  // (UpdateVolumeTexture reloads when VolumeTexture is null, the min-max
  // paths recompute when MinMaxTexture is null, EnsureGradientNormalTexture
  // runs per frame, and the grid traversal rebuilds from the valid flag).
  ReleaseMetalObject(this->VolumeTexture);
  ReleaseMetalObject(this->MinMaxTexture);
  this->ReleaseGradientNormalTexture();
  this->GridTraversalResourcesValid = false;
}

bool vtkMetalGPUVolumeRayCastMapper::UpdateVolumeTexture(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkVolume* vol)
{
  if (!this->EnsureEffectiveInput())
  {
    return false;
  }
  vtkImageData* input = this->EffectiveInput;
  if (!input)
  {
    return false;
  }

  // Resolve the scalar array honoring the mapper's scalar mode (default active
  // scalars, or a named point/cell array for VTK_SCALAR_MODE_USE_*_FIELD_DATA).
  // This mirrors vtkGPUVolumeRayCastMapper::Update, which the OpenGL backend
  // relies on; reading GetPointData()->GetScalars() directly fails when the
  // test sets ScalarModeToUsePointFieldData + SelectScalarArray.
  // Use a local cellFlag (GetScalars sets it by reference) so the member still
  // reflects the original input's scalar association for later EnsureEffectiveInput
  // calls — mutating it here could cause the cell-data proxy to be replaced and
  // freed while GPURender still holds a raw pointer to it.
  int cellFlag = this->CellFlag;
  vtkDataArray* scalars = this->GetScalars(
    input, this->ScalarMode, this->ArrayAccessMode, this->ArrayId, this->ArrayName, cellFlag);
  if (!scalars)
  {
    return false;
  }

  bool doReload = (this->VolumeTexture == nullptr);
  doReload |= (input->GetMTime() > this->VolumeUploadTime.GetMTime());

  // Check if partitioning is active
  bool usePartitions = (this->Partitions[0] > 1 || this->Partitions[1] > 1 || this->Partitions[2] > 1);

  if (usePartitions)
  {
    vtkVolumeProperty* property = vol ? vol->GetProperty() : nullptr;
    vtkPiecewiseFunction* opacityFunc =
      property ? property->GetScalarOpacity() : nullptr;

    bool needsVolumeRebuild =
      this->VolumeTexture == nullptr ||
      input->GetMTime() > this->VolumeUploadTime.GetMTime();

    bool needsGridRebuild =
      needsVolumeRebuild ||
      this->GetMTime() > this->GridTraversalUploadTime.GetMTime() ||
      (opacityFunc &&
       opacityFunc->GetMTime() > this->GridTraversalUploadTime.GetMTime()) ||
      this->CachedGridDims[0] != static_cast<int>(this->Partitions[0]) ||
      this->CachedGridDims[1] != static_cast<int>(this->Partitions[1]) ||
      this->CachedGridDims[2] != static_cast<int>(this->Partitions[2]);

    if (needsVolumeRebuild)
    {
      VolumeBounds vb = ComputeVolumeBounds(input, this->CellFlag == 1);
      this->ModelBounds[0] = vb.Min[0];
      this->ModelBounds[1] = vb.Max[0];
      this->ModelBounds[2] = vb.Min[1];
      this->ModelBounds[3] = vb.Max[1];
      this->ModelBounds[4] = vb.Min[2];
      this->ModelBounds[5] = vb.Max[2];

      this->VolumeNumComponents = scalars->GetNumberOfComponents();

      ReleaseMetalObject(this->VolumeTexture);
      if (!this->CreateGlobalVolumeTexture(mtlDeviceVoid, mtlQueueVoid, input, scalars))
      {
        return false;
      }

      this->VolumeUploadTime.Modified();
      this->UpdateRectilinearCoordsBuffer(mtlDeviceVoid);
    }

    if (needsGridRebuild &&
        opacityFunc &&
        (this->MacrocellScalarMin.empty() || this->MacrocellScalarMax.empty()))
    {
      this->UpdateMinMaxTexture(mtlDeviceVoid, vol, input, scalars, true);
    }

    if (needsGridRebuild)
    {
      this->GridTraversalResourcesValid = false;
    }

    if (!this->GridTraversalResourcesValid && this->VolumeTexture)
    {
      this->EnsureGridTraversalResources(mtlDeviceVoid, mtlQueueVoid, input, vol);
      this->GridTraversalUploadTime.Modified();
    }

    return true;
  }

  if (doReload)
  {
    @autoreleasepool
    {
      id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;
      id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mtlQueueVoid;

      int dims[3];
      input->GetDimensions(dims);

      if (dims[0] < 1 || dims[1] < 1 || dims[2] < 1)
      {
        vtkErrorMacro("Volume has zero dimensions");
        return false;
      }

      int dataType = scalars->GetDataType();
      int numComponents = scalars->GetNumberOfComponents();
      // Bound the conversion by the volume's voxel count: the staging buffer
      // below is sized from dims, but over-provisioned arrays (e.g.
      // vtkImplicitArray in TestSmartVolumeMapperImplicitArray) report more
      // tuples than fit the texture. The OpenGL backend reads only the
      // block-size slice of the array (vtkVolumeTexture::LoadTexture), so
      // clamping restores parity and avoids an out-of-bounds write.
      vtkIdType numTuples = std::min(scalars->GetNumberOfTuples(),
        static_cast<vtkIdType>(dims[0]) * dims[1] * dims[2]);

      if (numComponents < 1 || numComponents > 4)
      {
        vtkErrorMacro("Unsupported number of scalar components: " << numComponents);
        return false;
      }

      this->VolumeNumComponents = numComponents;

      // Store model-space bounds using image extent (handles non-zero extents and negative spacing)
      {
        VolumeBounds vb = ComputeVolumeBounds(input, this->CellFlag == 1);
        this->ModelBounds[0] = vb.Min[0];
        this->ModelBounds[1] = vb.Max[0];
        this->ModelBounds[2] = vb.Min[1];
        this->ModelBounds[3] = vb.Max[1];
        this->ModelBounds[4] = vb.Min[2];
        this->ModelBounds[5] = vb.Max[2];
      }

      // Select optimal texture format for this data type
      VolumeFormat fmtInfo = ChooseVolumeFormat(
        dataType, numComponents, this->ScalarRange, this->PreferHalfPrecision);
      bool useHalf = this->PreferHalfPrecision &&
        HalfRangeIsSafe(this->ScalarRange[0], this->ScalarRange[1]);
      this->ScalarNormalizationFactor = fmtInfo.NormalizationFactor;

      bool gpuConversionUsed = false;

      int actualComponents = (numComponents == 3) ? 4 : numComponents;
      NSUInteger bytesPerRow = static_cast<NSUInteger>(dims[0]) * fmtInfo.BytesPerComponent *
        actualComponents;
      NSUInteger bytesPerImage = bytesPerRow * dims[1];
      NSUInteger totalBytes = bytesPerImage * dims[2];

      // Phase 7: GPU data type conversion (replaces CPU vtkSMPTools loop)
      // For short/int/uint/double data types, dispatch a Metal compute kernel
      // that reads from a shared buffer and writes directly to the 3D texture.
      if (fmtInfo.NeedsConversion && this->UseGPUConversion)
      {
        // Determine kernel name based on (dataType, useHalf) pair
        const char* kernelName = nullptr;
             if (dataType == VTK_SHORT)        kernelName = useHalf ? "volume_convert_short_to_half"  : "volume_convert_short_to_float";
        else if (dataType == VTK_INT)          kernelName = useHalf ? "volume_convert_int_to_half"    : "volume_convert_int_to_float";
        else if (dataType == VTK_UNSIGNED_INT) kernelName = useHalf ? "volume_convert_uint_to_half"   : "volume_convert_uint_to_float";
        else if (dataType == VTK_FLOAT && useHalf) kernelName = "volume_convert_float_to_half";
        // Note: VTK_DOUBLE is not supported in Metal device address space; falls through to CPU.

        if (!kernelName)
        {
          // Metal cannot read double in device address space.
          // Fall through to CPU conversion.
        }
        else
        {
          if (!this->EnsureConversionPipelines(mtlDeviceVoid))
          {
            return false;
          }

          id<MTLComputePipelineState> pipeline = nullptr;
               if (dataType == VTK_SHORT)        pipeline = (__bridge id<MTLComputePipelineState>)(useHalf ? this->ConvertShortToHalfPipeline : this->ConvertShortToFloatPipeline);
          else if (dataType == VTK_INT)          pipeline = (__bridge id<MTLComputePipelineState>)(useHalf ? this->ConvertIntToHalfPipeline : this->ConvertIntToFloatPipeline);
          else if (dataType == VTK_UNSIGNED_INT) pipeline = (__bridge id<MTLComputePipelineState>)(useHalf ? this->ConvertUIntToHalfPipeline : this->ConvertUIntToFloatPipeline);
          else if (dataType == VTK_FLOAT && useHalf) pipeline = (__bridge id<MTLComputePipelineState>)this->ConvertFloatToHalfPipeline;
          if (!pipeline)
          {
            vtkErrorMacro("GPU conversion pipeline not available for data type " << dataType);
            return false;
          }

          int outputComponents = (numComponents == 3) ? 4 : numComponents;
          size_t srcBytesPerVoxel = static_cast<size_t>(vtkDataArray::GetDataTypeSize(dataType)) * numComponents;
          size_t totalSrcBytes = static_cast<size_t>(numTuples) * srcBytesPerVoxel;

          id<MTLBuffer> srcBuf = [device newBufferWithBytes:scalars->GetVoidPointer(0)
                                                     length:totalSrcBytes
                                                    options:MTLResourceStorageModeShared];
          if (!srcBuf)
          {
            vtkErrorMacro("Failed to create source buffer for GPU conversion");
            return false;
          }

          // Ensure texture has ShaderWrite usage for compute kernel output
          ReleaseMetalObject(this->VolumeTexture);

          id<MTLTexture> tex = NewTexture3D(
            device,
            fmtInfo.Format,
            static_cast<NSUInteger>(dims[0]),
            static_cast<NSUInteger>(dims[1]),
            static_cast<NSUInteger>(dims[2]),
            MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead,
            MTLStorageModePrivate,
            VolumeGPUOptimizedContents());
          if (!tex)
          {
            vtkErrorMacro("Failed to create 3D volume texture for GPU conversion");
            [srcBuf release];
            return false;
          }
          AssignMetalObject(this->VolumeTexture, tex);

          // Dispatch compute kernel
          VolumeConvertUniforms vu;
          vu.dimX = static_cast<uint32_t>(dims[0]);
          vu.dimY = static_cast<uint32_t>(dims[1]);
          vu.dimZ = static_cast<uint32_t>(dims[2]);
          vu.numComponents = static_cast<uint32_t>(numComponents);
          vu.outputComponents = static_cast<uint32_t>(outputComponents);
          vu._pad = 0;

          id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
          cmdBuf.label = @"VTK Volume GPU Convert";
          id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
          enc.label = @"Volume Convert";
          [enc setComputePipelineState:pipeline];
          [enc setBuffer:srcBuf offset:0 atIndex:0];
          [enc setTexture:tex atIndex:0];
          [enc setBytes:&vu length:sizeof(vu) atIndex:1];

          MTLSize grid = MTLSizeMake(dims[0], dims[1], dims[2]);
          MTLSize tg = MTLSizeMake(8, 8, 8);
          [enc dispatchThreads:grid threadsPerThreadgroup:tg];
          [enc endEncoding];
          [cmdBuf commit];

          [srcBuf release];
          gpuConversionUsed = true;
        }
      }
       // RG8 pair-pack repack (see VolumeRg8PairActive): interleave adjacent
       // slices into RG texel pairs over a halved-depth texture.
       bool rg8Pair = false;
       int upDims[3] = { dims[0], dims[1], dims[2] };
       MTLPixelFormat upFormat = fmtInfo.Format;
       NSUInteger upBytesPerRow = bytesPerRow;
       NSUInteger upBytesPerImage = bytesPerImage;
       std::vector<uint8_t> rg8Storage;
       if (!gpuConversionUsed && VolumeRg8PairActive() && !fmtInfo.NeedsConversion &&
           dataType == VTK_UNSIGNED_CHAR && numComponents == 1)
       {
         rg8Pair = true;
         upDims[2] = (dims[2] + 1) / 2;
         upFormat = MTLPixelFormatRG8Unorm;
         upBytesPerRow = static_cast<NSUInteger>(dims[0]) * 2;
         upBytesPerImage = upBytesPerRow * static_cast<NSUInteger>(dims[1]);
         const size_t sliceTexels = static_cast<size_t>(dims[0]) * dims[1];
         rg8Storage.resize(sliceTexels * static_cast<size_t>(upDims[2]) * 2);
         const uint8_t* in = static_cast<const uint8_t*>(scalars->GetVoidPointer(0));
         for (int p = 0; p < upDims[2]; ++p)
         {
           uint8_t* dst = rg8Storage.data() + sliceTexels * 2 * static_cast<size_t>(p);
           const uint8_t* sa = in + sliceTexels * static_cast<size_t>(2 * p);
           const uint8_t* sb = (2 * p + 1 < dims[2]) ? sa + sliceTexels : sa;
           for (size_t i = 0; i < sliceTexels; ++i)
           {
             dst[2 * i + 0] = sa[i];
             dst[2 * i + 1] = sb[i];
           }
         }
       }

       // Transposed repack (see VolumeTransposedActiveDepth policy): upload
       // the volume with the SHORTEST array dimension in texture depth.
       // Metal's private 3D tiling is axis-biased; with a long extent as
       // texture depth, trilinear depth-pair fetches under per-pixel jitter
       // phase scatter pay a large DRAM tax (2026-08-22 root cause). X-depth
       // maps fetch coords via .zyx, Y-depth via .xzy.
        bool volTransposed = false;
        int volAxisDepth = 0;
        bool volTransposedGPU = false;
        std::vector<uint8_t> transStorage;
        if (!gpuConversionUsed && !rg8Pair && VolumeTransposedActive() &&
            dataType == VTK_UNSIGNED_CHAR && numComponents == 1)
        {
          const int trAxis = VolumeTransposedAxisDepth(dims);
          if (trAxis != 0)
          {
          volTransposed = true;
          volAxisDepth = trAxis;
          const NSUInteger elemSize = static_cast<NSUInteger>(fmtInfo.BytesPerComponent) *
            actualComponents;
          if (trAxis == 2)
          {
            // Y-depth: texture holds (W,D,H); original Y moves into depth.
            upDims[1] = dims[2];
            upDims[2] = dims[1];
            upBytesPerRow = static_cast<NSUInteger>(dims[0]) * elemSize;
            upBytesPerImage = upBytesPerRow * static_cast<NSUInteger>(dims[2]);
          }
          else
          {
            // X-depth: texture holds (D,H,W); the slice axis becomes the width.
            upDims[0] = dims[2];
            upDims[2] = dims[0];
            upBytesPerRow = static_cast<NSUInteger>(dims[2]) * elemSize;
            upBytesPerImage = upBytesPerRow * static_cast<NSUInteger>(dims[1]);
          }
          volTransposedGPU = VolumeTransposeGPU();
          if (volTransposedGPU)
          {
            // §28: staging stays in ORIGINAL layout; the compute kernel writes
            // the swapped-dims texture directly from it.
            fprintf(stderr, "[TR] UpdateVolumeTexture: transposed (axis %c) dims %dx%dx%d -> %dx%dx%d (GPU kernel)\n",
              trAxis == 2 ? 'y' : 'x', dims[0], dims[1], dims[2],
              upDims[0], upDims[1], upDims[2]);
          }
          else
          {
          transStorage.resize(totalBytes);
          // X-depth: T(u=z, v=y, w=x): dst[(x*H + y)*Z + z] = V[(z*H + y)*W + x]
          // Y-depth: T(u=x, v=z, w=y): dst[(y*Z + z)*W + x] = V[(z*H + y)*W + x]
          // (element indices; element size elemSize bytes)
          const int BS = 32;
          const uint8_t* in = static_cast<const uint8_t*>(scalars->GetVoidPointer(0));
          uint8_t* out = transStorage.data();
          std::chrono::steady_clock::time_point trT0;
          if (getenv("VTK_METAL_TEST_TR_BENCH")) trT0 = std::chrono::steady_clock::now();
          for (int xb = 0; xb < dims[0]; xb += BS)
            for (int zb = 0; zb < dims[2]; zb += BS)
              for (int yb = 0; yb < dims[1]; yb += BS)
              {
                const int xe = std::min(xb + BS, dims[0]);
                const int ze = std::min(zb + BS, dims[2]);
                const int ye = std::min(yb + BS, dims[1]);
                for (int x = xb; x < xe; ++x)
                  for (int z = zb; z < ze; ++z)
                    for (int y = yb; y < ye; ++y)
                    {
                      const size_t srcOff =
                        (((static_cast<size_t>(z) * dims[1] + y) * dims[0]) + x) * elemSize;
                      size_t dstOff;
                      if (trAxis == 2)
                        dstOff =
                          ((static_cast<size_t>(y) * dims[2] + z) * dims[0] + x) * elemSize;
                      else
                        dstOff =
                          ((static_cast<size_t>(x) * dims[1] + y) * dims[2] + z) * elemSize;
                      std::memcpy(out + dstOff, in + srcOff, elemSize);
                    }
              }
          if (getenv("VTK_METAL_TEST_TR_BENCH"))
            fprintf(stderr, "[TRBENCH] cpu blocked transpose: %.1f ms (%zu bytes)\n",
              std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - trT0).count(),
              transStorage.size());
          fprintf(stderr, "[TR] UpdateVolumeTexture: transposed (axis %c) dims %dx%dx%d -> %dx%dx%d\n",
            trAxis == 2 ? 'y' : 'x', dims[0], dims[1], dims[2],
            upDims[0], upDims[1], upDims[2]);
         // TEMP-DIAG (VTK_METAL_TEST_TR_DUMP): verify repacked bytes. Under
         // the mapping T(i,j,k) = V(k, j, i), T's width-i slice at i=D/2 must
         // equal V's axial slice D/2 exactly.
         if (getenv("VTK_METAL_TEST_TR_DUMP"))
         {
           const int W = dims[0], H = dims[1], D = dims[2];
           const uint8_t* vin = static_cast<const uint8_t*>(scalars->GetVoidPointer(0));
           const int dslice = D / 2;
           FILE* f1 = fopen("/tmp/tr_orig_slice.ppm", "wb");
           FILE* f2 = fopen("/tmp/tr_tran_slice.ppm", "wb");
           if (f1 && f2)
           {
             fprintf(f1, "P6\n%d %d\n255\n", W, H);
             fprintf(f2, "P6\n%d %d\n255\n", W, H);
             for (int y = 0; y < H; ++y)
               for (int x = 0; x < W; ++x)
               {
                 uint8_t a = vin[((static_cast<size_t>(dslice) * H) + y) * W + x];
                 uint8_t b = transStorage[static_cast<size_t>(x) * upBytesPerImage +
                                          static_cast<size_t>(y) * upBytesPerRow + dslice];
                 uint8_t pa[3] = { a, a, a };
                 uint8_t pb[3] = { b, b, b };
                 fwrite(pa, 1, 3, f1);
                 fwrite(pb, 1, 3, f2);
               }
              fprintf(stderr, "[TR] dumped /tmp/tr_orig_slice.ppm /tmp/tr_tran_slice.ppm\n");
            }
             if (f1) fclose(f1);
             if (f2) fclose(f2);
            // Full-volume FNV-1a of the repacked bytes (reference for the
            // GPU-kernel checksum printed after upload).
            uint64_t h = 1469598103934665603ULL;
            for (size_t i = 0; i < transStorage.size(); ++i)
              h = (h ^ transStorage[i]) * 1099511628211ULL;
            fprintf(stderr, "[TR] transStorage fnv1a %llu\n",
              (unsigned long long)h);
           }
           }
          } // end if (trAxis != 0)
         }
         this->VolumeTextureTransposed = volTransposed;
        this->VolumeTextureAxisDepth = volAxisDepth;

        if (!gpuConversionUsed)
       {
         if (VolumeRg8PairActive())
           fprintf(stderr, "[RG8] UpdateVolumeTexture: rg8Pair=%d dims %dx%dx%d -> upDims %dx%dx%d staging=%zu\n",
             rg8Pair, dims[0], dims[1], dims[2], upDims[0], upDims[1], upDims[2],
             rg8Pair ? rg8Storage.size() : (size_t)totalBytes);
         id<MTLBuffer> stagingBuf = [device newBufferWithLength:(rg8Pair ? rg8Storage.size() : totalBytes)
                                                       options:MTLResourceStorageModeShared];
        if (!stagingBuf)
        {
          vtkErrorMacro("Failed to create volume staging buffer");
          return false;
        }
        void* uploadPointer = [stagingBuf contents];

        if (fmtInfo.NeedsConversion)
        {
          ConvertVolumeData(scalars->GetVoidPointer(0), dataType, numComponents,
            numTuples, uploadPointer, useHalf, actualComponents, scalars);
        }
        else if (dataType == VTK_FLOAT && numComponents == 3)
        {
          Expand3To4<float>(
            static_cast<const float*>(scalars->GetVoidPointer(0)),
            static_cast<float*>(uploadPointer),
            numTuples,
            0.0f);
        }
        else if (dataType == VTK_UNSIGNED_CHAR && numComponents == 3)
        {
          Expand3To4<unsigned char>(
            static_cast<const unsigned char*>(scalars->GetVoidPointer(0)),
            static_cast<unsigned char*>(uploadPointer),
            numTuples,
            255);
        }
        else if (dataType == VTK_UNSIGNED_SHORT && this->ScalarNormalizationFactor == 255.0f)
        {
          const unsigned short* src =
            static_cast<const unsigned short*>(scalars->GetVoidPointer(0));
          unsigned char* dst = static_cast<unsigned char*>(uploadPointer);
          vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
            for (vtkIdType i = begin; i < end; ++i)
            {
              for (int c = 0; c < numComponents; ++c)
                dst[i * actualComponents + c] = static_cast<unsigned char>(std::min<unsigned short>(src[i * numComponents + c], 255));
              if (numComponents == 3)
                dst[i * 4 + 3] = 255;
            }
          });
        }
        else if (dataType == VTK_UNSIGNED_SHORT && numComponents == 3)
        {
          Expand3To4<unsigned short>(
            static_cast<const unsigned short*>(scalars->GetVoidPointer(0)),
            static_cast<unsigned short*>(uploadPointer),
            numTuples,
            65535);
        }
        else if (dataType == VTK_SHORT && numComponents == 3)
        {
          Expand3To4<short>(
            static_cast<const short*>(scalars->GetVoidPointer(0)),
            static_cast<short*>(uploadPointer),
            numTuples,
            32767);
        }
        else
        {
          if (rg8Pair)
          {
            std::memcpy(uploadPointer, rg8Storage.data(), rg8Storage.size());
          }
          else if (volTransposed && !volTransposedGPU)
          {
            std::memcpy(uploadPointer, transStorage.data(), transStorage.size());
          }
          else
          {
            std::memcpy(uploadPointer, scalars->GetVoidPointer(0), totalBytes);
          }
        }

      id<MTLTexture> oldTex = (__bridge id<MTLTexture>)this->VolumeTexture;
      id<MTLTexture> tex = nil;

      if (oldTex &&
          oldTex.width == static_cast<NSUInteger>(upDims[0]) &&
          oldTex.height == static_cast<NSUInteger>(upDims[1]) &&
          oldTex.depth == static_cast<NSUInteger>(upDims[2]) &&
          oldTex.pixelFormat == upFormat)
      {
        tex = oldTex;
      }
      else
      {
        ReleaseMetalObject(this->VolumeTexture);

        tex = NewTexture3D(
          device,
          upFormat,
          static_cast<NSUInteger>(upDims[0]),
          static_cast<NSUInteger>(upDims[1]),
          static_cast<NSUInteger>(upDims[2]),
          volTransposedGPU ? MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite
                           : MTLTextureUsageShaderRead,
          getenv("VTK_METAL_TEST_TR_GPU") ? MTLStorageModeShared
                                          : MTLStorageModePrivate,
          VolumeGPUOptimizedContents());
        if (!tex)
        {
          vtkErrorMacro("Failed to create 3D volume texture");
          return false;
        }
        AssignMetalObject(this->VolumeTexture, tex);
      }

      id<MTLCommandBuffer> uploadCmdBuf = [queue commandBuffer];
      if (volTransposedGPU)
      {
        // §28 GPU transpose: no blit. One compute pass writes the swapped-dims
        // volume texture directly from the ORIGINAL-layout staging buffer.
        if (!this->EnsureShaderLibrary((__bridge void*)device))
        {
          vtkErrorMacro("GPU transpose: no shader library");
          [stagingBuf release];
          return false;
        }
        if (!this->TransposeComputePipeline)
        {
          id<MTLLibrary> library =
            (__bridge id<MTLLibrary>)this->CachedShaderLibrary;
          id<MTLFunction> func =
            [library newFunctionWithName:@"volume_transpose_xz"];
          NSError* error = nil;
          id<MTLComputePipelineState> pso = func
            ? [device newComputePipelineStateWithFunction:func error:&error]
            : nil;
          if (func) [func release];
          if (!pso)
          {
            vtkErrorMacro(<< "Failed to create transpose compute pipeline: "
                          << (error ? [[error localizedDescription] UTF8String]
                                    : "unknown"));
            [stagingBuf release];
            return false;
          }
          AssignMetalObject(this->TransposeComputePipeline, pso);
        }
        id<MTLComputePipelineState> pso =
          (__bridge id<MTLComputePipelineState>)this->TransposeComputePipeline;
        NSUInteger mtt = [pso maxTotalThreadsPerThreadgroup];
        MTLSize tg = mtt >= 512 ? MTLSizeMake(8, 8, 8) : MTLSizeMake(4, 4, 4);
        MTLSize grid = MTLSizeMake(
          (static_cast<NSUInteger>(dims[0]) + tg.width - 1) / tg.width,
          (static_cast<NSUInteger>(dims[1]) + tg.height - 1) / tg.height,
          (static_cast<NSUInteger>(dims[2]) + tg.depth - 1) / tg.depth);
        std::chrono::steady_clock::time_point trT0;
        bool trBench = getenv("VTK_METAL_TEST_TR_BENCH") != nullptr;
        if (trBench || getenv("VTK_METAL_TEST_TR_DUMP"))
        {
          // Verify the staging bytes the kernel will read.
          const uint8_t* sb = static_cast<const uint8_t*>([stagingBuf contents]);
          uint64_t h = 1469598103934665603ULL;
          for (size_t i = 0; i < totalBytes; ++i)
            h = (h ^ sb[i]) * 1099511628211ULL;
          fprintf(stderr, "[TR] staging fnv1a %llu\n", (unsigned long long)h);
        }
        if (trBench) trT0 = std::chrono::steady_clock::now();
        id<MTLComputeCommandEncoder> enc = [uploadCmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:stagingBuf offset:0 atIndex:0];
        // dst is [[texture(1)]] in volume_transpose_xz.
        [enc setTexture:tex atIndex:1];
        // MSL uint3 occupies 16 bytes in the constant address space.
        struct { uint32_t x, y, z, pad; } srcDims = {
          static_cast<uint32_t>(dims[0]), static_cast<uint32_t>(dims[1]),
          static_cast<uint32_t>(dims[2]), 0 };
        [enc setBytes:&srcDims length:sizeof(srcDims) atIndex:2];
        // Orientation code for volume_transpose_xz: 1=X-depth, 2=Y-depth.
        uint32_t trMode = static_cast<uint32_t>(volAxisDepth);
        [enc setBytes:&trMode length:sizeof(trMode) atIndex:3];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [uploadCmdBuf commit];
        if (trBench || getenv("VTK_METAL_TEST_TR_DUMP"))
        {
          [uploadCmdBuf waitUntilCompleted];
          fprintf(stderr, "[TR] cmd status %ld error %ld\n",
            (long)[uploadCmdBuf status], (long)[uploadCmdBuf error].code);
        }
        if (trBench)
        {
          fprintf(stderr,
            "[TRBENCH] gpu transpose (staging copy excluded): %.1f ms\n",
            std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - trT0).count());
        }
      }
      else
      {
      id<MTLBlitCommandEncoder> blit = [uploadCmdBuf blitCommandEncoder];
      [blit copyFromBuffer:stagingBuf
              sourceOffset:0
       sourceBytesPerRow:(rg8Pair || volTransposed ? upBytesPerRow : bytesPerRow)
     sourceBytesPerImage:(rg8Pair || volTransposed ? upBytesPerImage : bytesPerImage)
              sourceSize:MTLSizeMake(upDims[0], upDims[1], upDims[2])
               toTexture:tex
        destinationSlice:0
        destinationLevel:0
       destinationOrigin:MTLOriginMake(0, 0, 0)];
      [blit endEncoding];
      [uploadCmdBuf commit];
      } // end else (CPU blit path)
      if (getenv("VTK_METAL_TEST_TR_DUMP") && volTransposed)
      {
        // TEMP-DIAG: read the GPU texture back (Shared storage — run with
        // VTK_METAL_TEST_TR_GPU=1) and checksum the FULL volume against the
        // CPU repack's [TR] transStorage fnv1a.
        [uploadCmdBuf waitUntilCompleted];
        id<MTLTexture> texR = (__bridge id<MTLTexture>)this->VolumeTexture;
        if (texR.storageMode == MTLStorageModeShared)
        {
          std::vector<uint8_t> gpu(totalBytes);
          [texR getBytes:gpu.data() bytesPerRow:upBytesPerRow
           bytesPerImage:upBytesPerImage
              fromRegion:MTLRegionMake3D(0, 0, 0, upDims[0], upDims[1], upDims[2])
             mipmapLevel:0 slice:0];
          uint64_t h = 1469598103934665603ULL;
          for (size_t i = 0; i < gpu.size(); ++i)
            h = (h ^ gpu[i]) * 1099511628211ULL;
          fprintf(stderr, "[TR] gpu texture fnv1a %llu\n", (unsigned long long)h);
          const int W = dims[0], H = dims[1];
          const int dslice = upDims[0] / 2;
          FILE* f3 = fopen("/tmp/tr_gpu_slice.ppm", "wb");
          if (f3)
          {
            fprintf(f3, "P6\n%d %d\n255\n", W, H);
            for (int y = 0; y < H; ++y)
              for (int x = 0; x < W; ++x)
              {
                size_t off = static_cast<size_t>(x) * upBytesPerImage +
                             static_cast<size_t>(y) * upBytesPerRow + dslice;
                uint8_t v = gpu[off];
                uint8_t pv[3] = { v, v, v };
                fwrite(pv, 1, 3, f3);
              }
            fclose(f3);
            fprintf(stderr, "[TR] dumped /tmp/tr_gpu_slice.ppm\n");
          }
          // Occupancy profile along the destination x' extent (the transposed
          // slice axis): nonzero fraction over a sampled (y,z) grid per plane.
          fprintf(stderr, "[TR] occupancy along x':");
          for (int xp = 0; xp < static_cast<int>(upDims[0]); xp += 128)
          {
            int nz = 0, tot = 0;
            for (int z = 0; z < static_cast<int>(upDims[2]); z += 8)
              for (int y = 0; y < static_cast<int>(upDims[1]); y += 8)
              {
                ++tot;
                if (gpu[static_cast<size_t>(xp) * upBytesPerImage +
                        static_cast<size_t>(y) * upBytesPerRow + z] != 0)
                  ++nz;
              }
            fprintf(stderr, " %d:%.2f", xp,
              tot ? static_cast<double>(nz) / tot : 0.0);
          }
          fprintf(stderr, "\n");
        }
        else
        {
          fprintf(stderr,
            "[TR] TR_DUMP readback needs shared storage (set VTK_METAL_TEST_TR_GPU=1)\n");
        }
      }
      // Release our reference to the staging buffer. Metal keeps the buffer
      // alive internally until the command buffer completes.
      [stagingBuf release];

      } // end if (!gpuConversionUsed)

      this->VolumeUploadTime.Modified();
      this->UpdateRectilinearCoordsBuffer(mtlDeviceVoid);
    }
  }

  // Upload cell/point ghost-array blanking flags (vtkUniformGrid). Kept in
  // sync with the effective input here so the blanking texture stays valid for
  // the whole render frame.
  if (!this->UpdateBlankingTexture(mtlDeviceVoid, mtlQueueVoid, input))
  {
    return false;
  }

  // Phase 1C: Pre-warm common pipeline permutations to avoid first-use hitches.
  // PSO compilation takes 50-200ms on Apple Silicon; warming them up here on the
  // render thread (synchronous) ensures subsequent toggles are frame-rate smooth.
  if (!this->PipelinesPreWarmed)
  {
    static const uint32_t kPreWarmMasks[] = {
      0,
      VolumeFeature_Shading,
      VolumeFeature_Shading | VolumeFeature_GradientOpacity,
      VolumeFeature_Shading | VolumeFeature_MinMax,
      VolumeFeature_Shading | VolumeFeature_GradientOpacity | VolumeFeature_MinMax,
      VolumeFeature_Shading | VolumeFeature_Mask,
      VolumeFeature_Shading | VolumeFeature_Mask | VolumeFeature_MinMax,
      VolumeFeature_Shading | VolumeFeature_NormalTexture,
      VolumeFeature_Shading | VolumeFeature_NormalTexture | VolumeFeature_MinMax,
      VolumeFeature_Shading | VolumeFeature_ComputeNormalFromOpacity,
    };

    struct PreWarmSpec {
      uint32_t type;
      uint32_t colorFormat;
      uint32_t depthFormat;
      uint32_t samples;
    };

    int sampleCount = this->CurrentSampleCount > 0 ? this->CurrentSampleCount : 1;
    uint32_t sc = static_cast<uint32_t>(sampleCount);
    PreWarmSpec specs[] = {
      { static_cast<uint32_t>(VolumePipelineType::DirectScreen),
        MTLPixelFormatBGRA8Unorm, MTLPixelFormatDepth32Float, sc },
      { static_cast<uint32_t>(VolumePipelineType::OffscreenLayer),
        MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1 },
      { static_cast<uint32_t>(VolumePipelineType::FullscreenDirect),
        MTLPixelFormatBGRA8Unorm, MTLPixelFormatDepth32Float, sc },
      { static_cast<uint32_t>(VolumePipelineType::FullscreenOffscreen),
        MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1 },
      { static_cast<uint32_t>(VolumePipelineType::GridTraversalDirect),
        MTLPixelFormatBGRA8Unorm, MTLPixelFormatDepth32Float, sc },
      { static_cast<uint32_t>(VolumePipelineType::GridTraversalOffscreen),
        MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1 },
    };

    for (auto& spec : specs)
    {
      for (uint32_t mask : kPreWarmMasks)
      {
        this->GetOrCreateVolumePipeline(mtlDeviceVoid,
          spec.type, spec.colorFormat, spec.depthFormat, spec.samples, mask);
      }
    }
    this->PipelinesPreWarmed = true;
  }

  return true;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::UpdateRectilinearCoordsBuffer(void* mtlDeviceVoid)
{
  // Rebuilds the fragment buffer(5) coord-curve data (float3 per index, each
  // axis GetScaleAndBias-normalized, padded to the longest axis). Only built for
  // rectilinear inputs; released whenever the input is not rectilinear.
  ReleaseMetalObject(this->RectCoordsBuffer);
  if (!this->RectilinearInput || this->RectCoordsData.empty())
  {
    return;
  }
  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;
    NSUInteger byteCount =
      static_cast<NSUInteger>(this->RectCoordsData.size()) * sizeof(float);
    id<MTLBuffer> buf = [device newBufferWithBytes:this->RectCoordsData.data()
                                            length:byteCount
                                           options:MTLResourceStorageModeShared];
    AssignMetalObject(this->RectCoordsBuffer, buf);
  }
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::UpdateBlankingTexture(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkImageData* input)
{
  (void)mtlQueueVoid;

  // The ghost arrays live on the original dataset, not the effective input:
  // for cell data EnsureEffectiveInput builds a cell-sized proxy that carries
  // only the scalars, so reading the ghost arrays here from `input` would find
  // none and silently disable blanking.
  vtkImageData* source = vtkImageData::SafeDownCast(this->TransformedInputs[0]);
  vtkSmartPointer<vtkUnsignedCharArray> cellBlank =
    source ? source->GetCellGhostArray() : nullptr;
  vtkSmartPointer<vtkUnsignedCharArray> pointBlank =
    source ? source->GetPointGhostArray() : nullptr;
  bool blankCells = (cellBlank != nullptr);
  bool blankPoints = (pointBlank != nullptr);

  // No blanking present: release the texture (if any) so the uniforms disable
  // the blanking path this frame.
  if (!blankCells && !blankPoints)
  {
    this->BlankingPoints = nullptr;
    this->BlankingCells = nullptr;
    ReleaseMetalObject(this->BlankingTexture);
    return true;
  }

  bool doReload = (this->BlankingTexture == nullptr);
  doReload |= (source && source->GetMTime() > this->BlankingUploadTime.GetMTime());
  doReload |= (this->BlankingPoints != pointBlank);
  doReload |= (this->BlankingCells != cellBlank);
  doReload |= (cellBlank && cellBlank->GetMTime() > this->BlankingUploadTime.GetMTime());
  doReload |= (pointBlank && pointBlank->GetMTime() > this->BlankingUploadTime.GetMTime());

  this->BlankingPoints = pointBlank;
  this->BlankingCells = cellBlank;

  if (!doReload)
  {
    return true;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

    int dims[3];
    input->GetDimensions(dims);
    if (dims[0] < 1 || dims[1] < 1 || dims[2] < 1)
    {
      return false;
    }

    // The blanking texture has the same dimensions as the volume texture so
    // texel (i,j,k) aligns with the volume texel sampled at the same
    // coordinate, matching the OpenGL backend's Create3DFromRaw(blockSize).
    // Two components: .x = point-blank flag, .y = cell-blank flag.
    // For cell data the volume texture is cell-sized (built from the cell-data
    // proxy), so dims == the original cell dimensions; the original point
    // dimensions are one larger along each axis.
    const vtkIdType nx = dims[0];
    const vtkIdType ny = dims[1];
    const vtkIdType nz = dims[2];
    const vtkIdType numTexels = nx * ny * nz;

    int pdims[3] = { static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz) };
    if (source)
    {
      source->GetDimensions(pdims);
    }
    const vtkIdType p0 = pdims[0];
    const vtkIdType p1 = pdims[1];
    const vtkIdType p2 = pdims[2];
    const vtkIdType c0 = std::max<vtkIdType>(1, p0 - 1);
    const vtkIdType c1 = std::max<vtkIdType>(1, p1 - 1);
    const vtkIdType c2 = std::max<vtkIdType>(1, p2 - 1);
    // True when the volume texture (dims) holds one texel per cell, i.e. the
    // cell-data path. For point data dims equals the original point dims.
    const bool cellData = (c0 == nx && c1 == ny && c2 == nz);

    std::vector<unsigned char> blankingData(static_cast<size_t>(numTexels) * 2, 0);

    if (blankPoints)
    {
      const unsigned char* src = pointBlank->GetPointer(0);
      for (vtkIdType k = 0; k < nz; ++k)
      {
        for (vtkIdType j = 0; j < ny; ++j)
        {
          for (vtkIdType i = 0; i < nx; ++i)
          {
            vtkIdType texel = (k * ny + j) * nx + i;
            // Point data: texel (i,j,k) is point (i,j,k) (ptId == texel).
            // Cell data: the texel is the cell's lower-corner point; the shader
            // samples the +-half-step neighbors so the other corners are
            // caught by their own texels.
            vtkIdType ptId = (k * p1 + j) * p0 + i;
            blankingData[texel * 2 + 0] = src[ptId];
          }
        }
      }
    }

    if (blankCells)
    {
      const unsigned char* src = cellBlank->GetPointer(0);
      for (vtkIdType k = 0; k < nz; ++k)
      {
        const vtkIdType kc = (k < nz - 1) ? k : nz - 2;
        for (vtkIdType j = 0; j < ny; ++j)
        {
          const vtkIdType jc = (j < ny - 1) ? j : ny - 2;
          for (vtkIdType i = 0; i < nx; ++i)
          {
            const vtkIdType ic = (i < nx - 1) ? i : nx - 2;
            vtkIdType texel = (k * ny + j) * nx + i;
            vtkIdType cellId;
            if (cellData)
            {
              // Cell data: texel (i,j,k) is cell (i,j,k) directly.
              cellId = (k * c1 + j) * c0 + i;
            }
            else
            {
              // Point data: store cell (i,j,k)'s flag at its lower-corner point
              // texel, clamping the last point slice onto the last cell.
              cellId = (kc * c1 + jc) * c0 + ic;
            }
            blankingData[texel * 2 + 1] = src[cellId];
          }
        }
      }
    }

    ReleaseMetalObject(this->BlankingTexture);

    id<MTLTexture> tex = NewTexture3D(
      device,
      MTLPixelFormatRG8Unorm,
      static_cast<NSUInteger>(nx),
      static_cast<NSUInteger>(ny),
      static_cast<NSUInteger>(nz),
      MTLTextureUsageShaderRead,
      MTLStorageModeShared);
    if (!tex)
    {
      vtkErrorMacro("Failed to create blanking texture");
      return false;
    }
    AssignMetalObject(this->BlankingTexture, tex);

    MTLRegion region = MTLRegionMake3D(0, 0, 0, nx, ny, nz);
    NSUInteger bytesPerRow = static_cast<NSUInteger>(nx) * 2;
    NSUInteger bytesPerImage = bytesPerRow * ny;
    [tex replaceRegion:region
          mipmapLevel:0
                slice:0
            withBytes:blankingData.data()
          bytesPerRow:bytesPerRow
        bytesPerImage:bytesPerImage];

    this->BlankingUploadTime.Modified();
  }

  return this->BlankingTexture != nullptr;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::UpdateTransferFunctionTexture(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkVolume* vol,
  double actualSampleDistance)
{
  vtkVolumeProperty* property = vol->GetProperty();
  if (!property)
  {
    return false;
  }

  vtkColorTransferFunction* colorFunc = property->GetRGBTransferFunction();
  vtkPiecewiseFunction* opacityFunc = property->GetScalarOpacity();
  if (!colorFunc || !opacityFunc)
  {
    return false;
  }

  // Dependent RGBA (4-component): the opacity LUT is built over the LAST
  // component's scalar range (OpenGL UpdateTransferFunctions RGBA branch passes
  // numComp - 1 to UpdateOpacityTransferFunction). The shader samples it with
  // the raw normalized fetch (scalar.w), so the table range alone determines
  // the sampled opacity. Color is unused in this mode (raw scalar.xyz).
  const bool dependentRGBA =
    (this->VolumeNumComponents == 4) && property && !property->GetIndependentComponents();
  // Dependent 2-component (LA): color maps the FIRST component's range and
  // opacity the LAST component's range (OpenGL UpdateColorTransferFunction(0)
  // / UpdateOpacityTransferFunction(numComp - 1) parity). The shader samples
  // the shared RGBA table at the first-component-normalized coordinate for RGB
  // and at the last-component-normalized coordinate for A.
  const bool dependentLA =
    (this->VolumeNumComponents == 2) && property && !property->GetIndependentComponents();
  const double* colorRange =
    dependentLA ? this->ComponentScalarRange[0] : this->ScalarRange;
  const double* opacityRange =
    dependentLA ? this->ComponentScalarRange[1]
                : (dependentRGBA ? this->ComponentScalarRange[3] : this->ScalarRange);
  // Table width estimate: keep the pre-existing range choice (the last
  // component's range for dependent RGBA, component 0's range otherwise) so
  // the width is unchanged for all existing paths.
  const double* widthRange =
    dependentRGBA ? this->ComponentScalarRange[3] : this->ScalarRange;

  bool sampleDistChanged =
    (actualSampleDistance != this->LastTransferFunctionSampleDist);
  // The opacity correction applied at table-build time depends on the blend
  // mode (OpenGL vtkOpenGLVolumeOpacityTable::NeedsUpdate tracks LastBlendMode):
  // COMPOSITE pre-integrates, ADDITIVE scales, MIP/MinIP/Average stay raw. A
  // blend-mode change must therefore rebuild the table even when the transfer
  // functions themselves are unchanged.
  bool blendModeChanged =
    (this->GetBlendMode() != this->LastTransferFunctionBlendMode);

  // Primary change detection uses the pre-existing table range (widthRange): it
  // equals the color range for every path except dependent RGBA, where the RGB
  // channels of the shared table are never sampled (raw scalar RGB is used), so
  // only the last-component range (widthRange == opacityRange) needs tracking
  // there. The opacity range only needs separate tracking when dependent LA
  // splits the table across two ranges; for all other paths it equals
  // widthRange, so no redundant comparisons are performed.
  bool scalarRangeChanged =
    (widthRange[0] != this->LastTransferFunctionScalarRange[0]) ||
    (widthRange[1] != this->LastTransferFunctionScalarRange[1]) ||
    (dependentLA &&
      ((opacityRange[0] != this->LastTransferFunctionOpacityScalarRange[0]) ||
       (opacityRange[1] != this->LastTransferFunctionOpacityScalarRange[1])));

  bool doReload = (this->ColorOpacityTexture == nullptr);
  doReload |= (colorFunc->GetMTime() > this->TransferFunctionUploadTime.GetMTime());
  doReload |= (opacityFunc->GetMTime() > this->TransferFunctionUploadTime.GetMTime());
  doReload |= scalarRangeChanged;
  doReload |= sampleDistChanged;
  doReload |= blendModeChanged;

  if (doReload)
  {
    // Compute pre-integration factor (sampleDistance / unitDistance), same as
    // vtkOpenGLVolumeOpacityTable::InternalUpdate. OpenGL parity: for
    // dependent modes the opacity function is sampled over the LAST component
    // (vtkVolumeInputHelper::UpdateOpacityTransferFunction passes numComp - 1
    // for RGBA/LA), so the unit distance follows that component too. For
    // 1-component volumes the index is 0 and this matches the classic path.
    const int opacityComp =
      (dependentLA || dependentRGBA) ? (this->VolumeNumComponents - 1) : 0;
    double unitDist = property->GetScalarOpacityUnitDistance(opacityComp);
    if (unitDist <= 0.0) unitDist = 1.0;
    double preIntegrationFactor = actualSampleDistance / unitDist;

    const int tfWidth = ComputeTransferFunctionWidth(
      colorFunc, opacityFunc, widthRange);

    @autoreleasepool
    {
      id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

      std::vector<uint16_t> tfData(static_cast<size_t>(tfWidth) * 4);
      FillTransferFunctionRGBA16FWithPreIntegration(
        colorFunc, opacityFunc,
        colorRange[0], colorRange[1],
        opacityRange[0], opacityRange[1],
        tfWidth, tfData.data(),
        preIntegrationFactor, this->GetBlendMode());

      // Swap rather than rewrite: in-flight frames on the GPU may still be
      // sampling the old texture.  Metal command buffers retain a strong
      // reference to every resource encoded into them until execution
      // completes, so the old texture stays alive for those in-flight frames.
      // We always allocate a fresh texture here — it is not yet referenced
      // by any command buffer — then populate it via replaceRegion before
      // the next GPURender can bind it (GPURender runs single-threaded on
      // the render pass, so there is no race between AssignMetalObject and
      // replaceRegion).  If allocation fails, the slot remains nullptr and
      // doReload will re-trigger next frame.
      ReleaseMetalObject(this->ColorOpacityTexture);

      id<MTLTexture> tex = NewTexture2D(
        device,
        MTLPixelFormatRGBA16Float,
        static_cast<NSUInteger>(tfWidth), 1,
        MTLTextureUsageShaderRead,
        MTLStorageModeShared);
      if (!tex)
      {
        vtkErrorMacro("Failed to create transfer function texture");
        return false;
      }
      AssignMetalObject(this->ColorOpacityTexture, tex);

      MTLRegion region = MTLRegionMake2D(0, 0, tfWidth, 1);
      [tex replaceRegion:region
            mipmapLevel:0
              withBytes:tfData.data()
            bytesPerRow:static_cast<NSUInteger>(tfWidth) * 8];

      this->LastTransferFunctionScalarRange[0] = widthRange[0];
      this->LastTransferFunctionScalarRange[1] = widthRange[1];
      this->LastTransferFunctionOpacityScalarRange[0] = opacityRange[0];
      this->LastTransferFunctionOpacityScalarRange[1] = opacityRange[1];
      this->LastTransferFunctionSampleDist = actualSampleDistance;
      this->LastTransferFunctionBlendMode = this->GetBlendMode();
      this->TransferFunctionUploadTime.Modified();
    }
  }

  // Independent multi-component transfer functions: components 1..3 get their
  // own color/opacity tables built over their own scalar ranges (OpenGL
  // OpacityTables[i]/RGBTables[i] parity). Component 0 reuses ColorOpacityTexture.
  const bool independentComps =
    property->GetIndependentComponents() && this->VolumeNumComponents > 1;

  if (independentComps)
  {
    bool compChanged = (this->VolumeNumComponents != this->LastVolumeNumComponents) ||
      !this->LastIndependentComponents;

    bool compRangeChanged = false;
    for (int c = 0; c < 4; ++c)
    {
      if (this->ComponentScalarRange[c][0] != this->LastComponentScalarRange[c][0] ||
        this->ComponentScalarRange[c][1] != this->LastComponentScalarRange[c][1])
      {
        compRangeChanged = true;
      }
    }

    bool doCompReload = (this->ComponentTransferFunctionTexture1 == nullptr) ||
      compChanged || compRangeChanged || blendModeChanged;
    for (int c = 1; c < std::min(4, this->VolumeNumComponents); ++c)
    {
      vtkColorTransferFunction* cf = property->GetRGBTransferFunction(c);
      vtkPiecewiseFunction* of = property->GetScalarOpacity(c);
      if (cf && of)
      {
        doCompReload |= (cf->GetMTime() > this->ComponentTransferFunctionUpdateTime.GetMTime());
        doCompReload |= (of->GetMTime() > this->ComponentTransferFunctionUpdateTime.GetMTime());
      }
    }

    if (doCompReload)
    {
      @autoreleasepool
      {
        id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

        void* compSlots[3] = { &this->ComponentTransferFunctionTexture1,
                               &this->ComponentTransferFunctionTexture2,
                               &this->ComponentTransferFunctionTexture3 };

        for (int c = 1; c < std::min(4, this->VolumeNumComponents); ++c)
        {
          vtkColorTransferFunction* cf = property->GetRGBTransferFunction(c);
          vtkPiecewiseFunction* of = property->GetScalarOpacity(c);
          if (!cf || !of)
          {
            continue;
          }

          const int tfWidth =
            ComputeTransferFunctionWidth(cf, of, this->ComponentScalarRange[c]);

          double unitDist = property->GetScalarOpacityUnitDistance(c);
          if (unitDist <= 0.0)
          {
            unitDist = 1.0;
          }
          double preIntegrationFactor = actualSampleDistance / unitDist;

          std::vector<uint16_t> tfData(static_cast<size_t>(tfWidth) * 4);
          FillTransferFunctionRGBA16FWithPreIntegration(
            cf, of,
            this->ComponentScalarRange[c][0], this->ComponentScalarRange[c][1],
            tfWidth, tfData.data(),
            preIntegrationFactor, this->GetBlendMode());

          // Swap rather than rewrite — see the single-path block above.
          ReleaseMetalObject(*static_cast<void**>(compSlots[c - 1]));

          id<MTLTexture> tex = NewTexture2D(
            device,
            MTLPixelFormatRGBA16Float,
            static_cast<NSUInteger>(tfWidth), 1,
            MTLTextureUsageShaderRead,
            MTLStorageModeShared);
          if (!tex)
          {
            vtkErrorMacro("Failed to create component transfer function texture");
            return false;
          }
          AssignMetalObject(*static_cast<void**>(compSlots[c - 1]), tex);

          MTLRegion region = MTLRegionMake2D(0, 0, tfWidth, 1);
          [tex replaceRegion:region
                mipmapLevel:0
                  withBytes:tfData.data()
                bytesPerRow:static_cast<NSUInteger>(tfWidth) * 8];
        }

        for (int c = 0; c < 4; ++c)
        {
          this->LastComponentScalarRange[c][0] = this->ComponentScalarRange[c][0];
          this->LastComponentScalarRange[c][1] = this->ComponentScalarRange[c][1];
        }
        this->LastVolumeNumComponents = this->VolumeNumComponents;
        this->LastIndependentComponents = true;
        this->ComponentTransferFunctionUpdateTime.Modified();
      }
    }
  }
  else if (this->LastIndependentComponents)
  {
    // Transitioned out of independent mode: drop the per-component textures.
    this->LastIndependentComponents = false;
    this->LastVolumeNumComponents = 1;
    ReleaseMetalObject(this->ComponentTransferFunctionTexture1);
    ReleaseMetalObject(this->ComponentTransferFunctionTexture2);
    ReleaseMetalObject(this->ComponentTransferFunctionTexture3);
  }

  return this->ColorOpacityTexture != nullptr;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::UpdateGradientOpacityTexture(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkVolume* vol)
{
  vtkVolumeProperty* property = vol->GetProperty();
  if (!property || !property->HasGradientOpacity())
  {
    return false;
  }

  vtkPiecewiseFunction* gradOpacityFunc = property->GetGradientOpacity();
  if (!gradOpacityFunc)
  {
    return false;
  }

  // Gradient opacity LUT range (OpenGL vtkVolumeInputHelper parity):
  //   single-component / independent mode  -> component 0's scalar range
  //   dependent multi-component (LA/RGBA)  -> the LAST component's scalar range
  //     (UpdateGradientOpacityTransferFunction passes numComp - 1 for RGBA/LA).
  // The shader normalizes the gradient magnitude against the range of the
  // component the gradient is computed on (component 0 for LA), so the LUT
  // coordinate saturates to 1.0 at data boundaries regardless of the table
  // range — hence the table range alone determines the sampled opacity.
  const bool dependentMulti =
    (this->VolumeNumComponents > 1) && (property->GetIndependentComponents() == 0);
  const int gradComp = dependentMulti ? (this->VolumeNumComponents - 1) : 0;
  const double* gradRange =
    dependentMulti ? this->ComponentScalarRange[gradComp] : this->ScalarRange;
  double scalarRange = gradRange[1] - gradRange[0];
  if (scalarRange <= 0.0)
  {
    scalarRange = 1.0;
  }
  double gradMax = scalarRange * 0.25;

  bool doReload = (this->GradientOpacityTexture == nullptr);
  doReload |= (gradOpacityFunc->GetMTime() > this->GradientOpacityUploadTime.GetMTime());
  doReload |= (gradRange[0] != this->LastGradientOpacityScalarRange[0] ||
               gradRange[1] != this->LastGradientOpacityScalarRange[1]);

  if (doReload)
  {
    @autoreleasepool
    {
      id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

      // Build 256-entry gradient opacity lookup table.
      // Range: [0, 0.25 * scalarRange] — matches the normalization in the shader
      // where gradient magnitude is normalized to [0, 0.25 * dataRange].

      unsigned char gradData[256 * 4]; // RGBA8Unorm (R channel used)
      double table[256];
      gradOpacityFunc->GetTable(0.0, gradMax, 256, table);

      for (int i = 0; i < 256; ++i)
      {
        unsigned char val =
          static_cast<unsigned char>(std::max(0.0, std::min(1.0, table[i])) * 255.0);
        gradData[i * 4 + 0] = val;
        gradData[i * 4 + 1] = val;
        gradData[i * 4 + 2] = val;
        gradData[i * 4 + 3] = 255;
      }

      // Swap (not in-place update) — see UpdateTransferFunctionTexture for
      // the full rationale: in-flight GPU frames may still be sampling the
      // old texture, so always allocate fresh to avoid torn reads.
      ReleaseMetalObject(this->GradientOpacityTexture);

      id<MTLTexture> tex = NewTexture2D(
        device,
        MTLPixelFormatRGBA8Unorm,
        256, 1,
        MTLTextureUsageShaderRead,
        MTLStorageModeShared);
      if (!tex)
      {
        vtkErrorMacro("Failed to create gradient opacity texture");
        return false;
      }
      AssignMetalObject(this->GradientOpacityTexture, tex);

      MTLRegion region = MTLRegionMake2D(0, 0, 256, 1);
      [tex replaceRegion:region
            mipmapLevel:0
              withBytes:gradData
            bytesPerRow:256 * 4];

      this->LastGradientOpacityScalarRange[0] = gradRange[0];
      this->LastGradientOpacityScalarRange[1] = gradRange[1];
      this->GradientOpacityUploadTime.Modified();
    }
  }

  return this->GradientOpacityTexture != nullptr;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::UpdateTransfer2DTexture(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkVolume* vol)
{
  vtkVolumeProperty* property = vol ? vol->GetProperty() : nullptr;
  if (!property || property->GetTransferFunctionMode() != vtkVolumeProperty::TF_2D)
  {
    ReleaseMetalObject(this->Transfer2DTexture);
    this->Transfer2DEnabled = false;
    return true;
  }

  vtkImageData* transfer2D = property->GetTransferFunction2D();
  if (!transfer2D)
  {
    ReleaseMetalObject(this->Transfer2DTexture);
    this->Transfer2DEnabled = false;
    return true;
  }

  bool doReload = (this->Transfer2DTexture == nullptr);
  doReload |= (transfer2D->GetMTime() > this->Transfer2DUploadTime.GetMTime());
  doReload |= (property->GetMTime() > this->Transfer2DUploadTime.GetMTime());

  if (doReload)
  {
    int dims[3];
    transfer2D->GetDimensions(dims);
    if (dims[0] < 1 || dims[1] < 1)
    {
      vtkErrorMacro("2D transfer function image has zero dimensions");
      return false;
    }

    vtkDataArray* scalars = transfer2D->GetPointData()->GetScalars();
    if (!scalars || scalars->GetNumberOfComponents() < 4)
    {
      vtkErrorMacro("2D transfer function image must be RGBA");
      return false;
    }

    int dataType = scalars->GetDataType();
    vtkIdType numTuples = scalars->GetNumberOfTuples();
    const int w = dims[0];
    const int h = dims[1];

    @autoreleasepool
    {
      id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

      std::vector<uint16_t> tf2DData(static_cast<size_t>(w) * h * 4);
      // Convert to half-float RGBA. ConvertVolumeData writes component c
      // (c < numComp) from the source; RGBA sources map 1:1 to the 4 outputs.
      ConvertVolumeData(scalars->GetVoidPointer(0), dataType, 4,
        numTuples, tf2DData.data(), true, 4, scalars);

      // Swap rather than in-place update (see UpdateTransferFunctionTexture).
      ReleaseMetalObject(this->Transfer2DTexture);

      id<MTLTexture> tex = NewTexture2D(
        device,
        MTLPixelFormatRGBA16Float,
        static_cast<NSUInteger>(w), static_cast<NSUInteger>(h),
        MTLTextureUsageShaderRead,
        MTLStorageModeShared);
      if (!tex)
      {
        vtkErrorMacro("Failed to create 2D transfer function texture");
        return false;
      }
      AssignMetalObject(this->Transfer2DTexture, tex);

      MTLRegion region = MTLRegionMake2D(0, 0, w, h);
      [tex replaceRegion:region
            mipmapLevel:0
              withBytes:tf2DData.data()
            bytesPerRow:static_cast<NSUInteger>(w) * 8];

      this->Transfer2DUploadTime.Modified();
    }
  }

  this->Transfer2DEnabled = (this->Transfer2DTexture != nullptr);
  return this->Transfer2DEnabled;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::UpdateTransfer2DYAxisTexture(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkVolume* vol, vtkImageData* input)
{
  if (!this->Transfer2DEnabled || !input)
  {
    return true;
  }

  // Legacy TF_2D mode (e.g. TestGPURayCastTransfer2D): no usable Y-axis array,
  // so the shader uses the gradient magnitude as the second axis (matching the
  // OpenGL backend's Transfer2DUseGradient=true path).
  const auto useGradientFallback = [this]() {
    this->Transfer2DUseGradient = true;
    if (this->Transfer2DYAxisTexture)
    {
      ReleaseMetalObject(this->Transfer2DYAxisTexture);
      this->Transfer2DYAxisUploadTime.Modified();
      this->Transfer2DYAxisArrayName.clear();
    }
  };

  const char* yName = this->GetTransfer2DYAxisArray();
  if (!yName)
  {
    useGradientFallback();
    return true;
  }

  vtkDataArray* arr = nullptr;
  if (input->GetPointData())
  {
    arr = input->GetPointData()->GetArray(yName);
  }
  if (!arr && input->GetCellData())
  {
    arr = input->GetCellData()->GetArray(yName);
  }
  if (!arr)
  {
    useGradientFallback();
    return true;
  }

  this->Transfer2DUseGradient = false;

  int dims[3];
  input->GetDimensions(dims);
  if (dims[0] < 1 || dims[1] < 1 || dims[2] < 1)
  {
    return false;
  }

  bool doReload = (this->Transfer2DYAxisTexture == nullptr);
  doReload |= (arr->GetMTime() > this->Transfer2DYAxisUploadTime.GetMTime());
  doReload |= (input->GetMTime() > this->Transfer2DYAxisUploadTime.GetMTime());
  doReload |= (this->Transfer2DYAxisArrayName != std::string(yName));

  if (doReload)
  {
    double range[2];
    arr->GetRange(range, 0);
    if (range[1] == range[0])
    {
      range[1] = range[0] + 1.0;
    }
    this->Transfer2DYAxisRange[0] = range[0];
    this->Transfer2DYAxisRange[1] = range[1];

    int dataType = arr->GetDataType();
    vtkIdType numTuples = arr->GetNumberOfTuples();
    bool useHalf = this->PreferHalfPrecision && HalfRangeIsSafe(range[0], range[1]);

    @autoreleasepool
    {
      id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

      size_t nBytes = static_cast<size_t>(numTuples) * (useHalf ? 2 : 4);
      std::vector<uint8_t> yData(nBytes);
      ConvertVolumeData(arr->GetVoidPointer(0), dataType, 1,
        numTuples, yData.data(), useHalf, 1, arr);

      // Swap rather than in-place update (see UpdateTransferFunctionTexture).
      ReleaseMetalObject(this->Transfer2DYAxisTexture);

      id<MTLTexture> tex = NewTexture3D(
        device,
        useHalf ? MTLPixelFormatR16Float : MTLPixelFormatR32Float,
        static_cast<NSUInteger>(dims[0]),
        static_cast<NSUInteger>(dims[1]),
        static_cast<NSUInteger>(dims[2]),
        MTLTextureUsageShaderRead,
        MTLStorageModeShared);
      if (!tex)
      {
        vtkErrorMacro("Failed to create 2D TF Y-axis scalar texture");
        return false;
      }
      AssignMetalObject(this->Transfer2DYAxisTexture, tex);

      NSUInteger bytesPerRow = static_cast<NSUInteger>(dims[0]) * (useHalf ? 2 : 4);
      NSUInteger bytesPerImage = bytesPerRow * static_cast<NSUInteger>(dims[1]);
      MTLRegion region = MTLRegionMake3D(0, 0, 0, dims[0], dims[1], dims[2]);
      [tex replaceRegion:region
            mipmapLevel:0
                  slice:0
              withBytes:yData.data()
            bytesPerRow:bytesPerRow
          bytesPerImage:bytesPerImage];

      this->Transfer2DYAxisArrayName = std::string(yName);
      this->Transfer2DYAxisUploadTime.Modified();
    }
  }

  return this->Transfer2DYAxisTexture != nullptr;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::UpdateMaskTexture(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkVolume* vol)
{
  vtkImageData* maskInput = this->MaskInput;
  if (!maskInput)
  {
    return false;
  }

  vtkVolumeProperty* property = vol->GetProperty();
  if (!property)
  {
    return false;
  }

  // Get the scalar array from the mask input
  int isCellData = 0;
  vtkDataArray* arr = this->GetScalars(
    maskInput, this->ScalarMode, this->ArrayAccessMode,
    this->ArrayId, this->ArrayName, isCellData);
  if (!arr)
  {
    return false;
  }

  bool doReload = (this->MaskTexture == nullptr);
  doReload |= (maskInput->GetMTime() > this->MaskUpdateTime.GetMTime());
  doReload |= (arr->GetMTime() > this->MaskUpdateTime.GetMTime());

  if (doReload)
  {
    @autoreleasepool
    {
      id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

      // Get mask dimensions
      int dims[3];
      maskInput->GetDimensions(dims);

      // Get the data pointer
      int numComponents = arr->GetNumberOfComponents();

      // Choose mask texture format based on data type to minimize memory
      vtkIdType numTuples = arr->GetNumberOfTuples();
      int dataType = arr->GetDataType();
      MTLPixelFormat chosenFormat;
      NSUInteger bytesPerPixel = 0;
      const void* uploadSrc = nullptr;
      std::vector<uint8_t> byteData;
      std::vector<uint16_t> shortData;
      std::vector<float> floatData;

      if (dataType == VTK_UNSIGNED_CHAR)
      {
        chosenFormat = MTLPixelFormatR8Unorm;
        bytesPerPixel = 1;
        byteData.resize(numTuples);
        const unsigned char* src = static_cast<const unsigned char*>(arr->GetVoidPointer(0));
        if (numComponents == 1)
        {
          std::memcpy(byteData.data(), src, numTuples);
        }
        else
        {
          vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
            for (vtkIdType i = begin; i < end; ++i)
              byteData[i] = src[i * numComponents];
          });
        }
        uploadSrc = byteData.data();
      }
      else if (dataType == VTK_UNSIGNED_SHORT)
      {
        chosenFormat = MTLPixelFormatR16Unorm;
        bytesPerPixel = 2;
        shortData.resize(numTuples);
        const unsigned short* src = static_cast<const unsigned short*>(arr->GetVoidPointer(0));
        if (numComponents == 1)
        {
          std::memcpy(shortData.data(), src, numTuples * sizeof(uint16_t));
        }
        else
        {
          vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
            for (vtkIdType i = begin; i < end; ++i)
              shortData[i] = src[i * numComponents];
          });
        }
        uploadSrc = shortData.data();
      }
      else
      {
        chosenFormat = MTLPixelFormatR32Float;
        bytesPerPixel = 4;
        floatData.resize(numTuples);

        if (dataType == VTK_FLOAT)
        {
          const float* src = static_cast<const float*>(arr->GetVoidPointer(0));
          if (numComponents == 1)
          {
            std::memcpy(floatData.data(), src, numTuples * sizeof(float));
          }
          else
          {
            vtkSMPTools::For(0, numTuples, [&](vtkIdType begin, vtkIdType end) {
              for (vtkIdType i = begin; i < end; ++i)
                floatData[i] = src[i * numComponents];
            });
          }
        }
        else
        {
          for (vtkIdType i = 0; i < numTuples; ++i)
          {
            floatData[i] = static_cast<float>(arr->GetComponent(i, 0));
          }
        }
        uploadSrc = floatData.data();
      }

      // Swap (not in-place update) — in-flight GPU frames may still be
      // sampling the old texture.  Releasing the old texture before
      // allocating the new one marginally reduces peak memory, which
      // matters for potentially large (e.g. 512^3 R8 = 256 MB) masks.
      // During the brief window between NewTexture3D and the old texture's
      // last in-flight frame completion, both old and new copies coexist
      // transiently.  If this ever causes memory pressure, switch to a
      // fenced ring of 3 copies indexed by UniformFrameIndex % 3.
      ReleaseMetalObject(this->MaskTexture);

      id<MTLTexture> tex = NewTexture3D(
        device,
        chosenFormat,
        static_cast<NSUInteger>(dims[0]),
        static_cast<NSUInteger>(dims[1]),
        static_cast<NSUInteger>(dims[2]),
        MTLTextureUsageShaderRead,
        MTLStorageModeShared);
      if (!tex)
      {
        vtkErrorMacro("Failed to create mask texture");
        return false;
      }
      AssignMetalObject(this->MaskTexture, tex);

      // Upload mask data to texture
      MTLRegion region = MTLRegionMake3D(0, 0, 0, dims[0], dims[1], dims[2]);
      NSUInteger maskBytesPerRow = static_cast<NSUInteger>(dims[0]) * bytesPerPixel;
      NSUInteger maskBytesPerImage = maskBytesPerRow * dims[1];
      [tex replaceRegion:region
            mipmapLevel:0
                  slice:0
              withBytes:uploadSrc
            bytesPerRow:maskBytesPerRow
          bytesPerImage:maskBytesPerImage];

      this->MaskUpdateTime.Modified();
    }
  }

  return this->MaskTexture != nullptr;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::UpdateLabelMapTransferTexture(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkVolume* vol)
{
  vtkVolumeProperty* property = vol->GetProperty();
  if (!property)
  {
    return false;
  }

  // Get label map labels
  std::set<int> labels = property->GetLabelMapLabels();
  if (labels.empty())
  {
    return false;
  }

  // Get the maximum label value
  int maxLabel = *(labels.rbegin());
  int numLabels = maxLabel + 1; // +1 because label 0 is included

  // Check if we need to reload
  vtkMTimeType latestMTime = 0;
  for (int label : labels)
  {
    vtkColorTransferFunction* colorFunc = property->GetLabelColor(label);
    vtkPiecewiseFunction* opacityFunc = property->GetLabelScalarOpacity(label);
    if (colorFunc && colorFunc->GetMTime() > latestMTime)
    {
      latestMTime = colorFunc->GetMTime();
    }
    if (opacityFunc && opacityFunc->GetMTime() > latestMTime)
    {
      latestMTime = opacityFunc->GetMTime();
    }
  }

  bool scalarRangeChanged =
    (this->ScalarRange[0] != this->LastLabelMapScalarRange[0]) ||
    (this->ScalarRange[1] != this->LastLabelMapScalarRange[1]);

  bool labelSetChanged =
    (maxLabel != this->LastLabelMapMaxLabel) ||
    (labels.size() != this->LastLabelMapLabelCount);

  bool doReload = (this->LabelMapTransferTexture == nullptr);
  doReload |= (latestMTime > this->MaskUpdateTime.GetMTime());
  doReload |= scalarRangeChanged;
  doReload |= labelSetChanged;

  if (doReload)
  {
    @autoreleasepool
    {
      id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

      // Create 2D label map transfer function texture
      // Width = 1024 (scalar range samples), Height = numLabels (one row per label)
      const int tfWidth = 1024;
      const int tfHeight = numLabels;

      // Get the scalar range from the volume texture
      double scalarRange[2] = { this->ScalarRange[0], this->ScalarRange[1] };

      // Build the 2D transfer function texture data (RGBA8Unorm)
      std::vector<uint8_t> tfData(tfWidth * tfHeight * 4);

      // Row 0: label 0 (default) - zeros (will be filled by default TF)
      std::fill(tfData.begin(), tfData.begin() + tfWidth * 4, static_cast<uint8_t>(0));

      // Rows 1..maxLabel: per-label transfer functions
      for (int label = 1; label < numLabels; ++label)
      {
        uint8_t* rowPtr = tfData.data() + label * tfWidth * 4;

        // Get color transfer function for this label
        vtkColorTransferFunction* colorFunc = property->GetLabelColor(label);
        if (!colorFunc)
        {
          colorFunc = property->GetRGBTransferFunction(); // fallback to default
        }

        // Get opacity function for this label
        vtkPiecewiseFunction* opacityFunc = property->GetLabelScalarOpacity(label);
        if (!opacityFunc)
        {
          opacityFunc = property->GetScalarOpacity(); // fallback to default
        }

        if (colorFunc && opacityFunc)
        {
          FillTransferFunctionRGBA8(
            colorFunc, opacityFunc,
            scalarRange[0], scalarRange[1],
            tfWidth, rowPtr);
        }
        else if (colorFunc)
        {
          std::vector<double> colorTable(tfWidth * 3);
          colorFunc->GetTable(scalarRange[0], scalarRange[1], tfWidth, colorTable.data());
          for (int i = 0; i < tfWidth; ++i)
          {
            rowPtr[i * 4 + 0] = ColorToByte(colorTable[i * 3 + 0]);
            rowPtr[i * 4 + 1] = ColorToByte(colorTable[i * 3 + 1]);
            rowPtr[i * 4 + 2] = ColorToByte(colorTable[i * 3 + 2]);
            rowPtr[i * 4 + 3] = 0;
          }
        }
        else if (opacityFunc)
        {
          std::vector<double> opacityTable(tfWidth);
          opacityFunc->GetTable(scalarRange[0], scalarRange[1], tfWidth, opacityTable.data());
          for (int i = 0; i < tfWidth; ++i)
          {
            rowPtr[i * 4 + 0] = 0;
            rowPtr[i * 4 + 1] = 0;
            rowPtr[i * 4 + 2] = 0;
            rowPtr[i * 4 + 3] = ColorToByte(opacityTable[i]);
          }
        }
      }

      // Swap (not in-place update) — see UpdateTransferFunctionTexture for
      // the full rationale.  This 2D texture (1024 × numLabels × 4 B) is
      // small enough that per-reload allocation is trivially cheap.
      ReleaseMetalObject(this->LabelMapTransferTexture);

      id<MTLTexture> tex = NewTexture2D(
        device,
        MTLPixelFormatRGBA8Unorm,
        static_cast<NSUInteger>(tfWidth),
        static_cast<NSUInteger>(tfHeight),
        MTLTextureUsageShaderRead,
        MTLStorageModeShared);
      if (!tex)
      {
        vtkErrorMacro("Failed to create label map transfer texture");
        return false;
      }
      AssignMetalObject(this->LabelMapTransferTexture, tex);

      // Upload data to texture
      MTLRegion region = MTLRegionMake2D(0, 0, tfWidth, tfHeight);
      [tex replaceRegion:region
            mipmapLevel:0
              withBytes:tfData.data()
            bytesPerRow:tfWidth * 4];

      this->LastLabelMapScalarRange[0] = this->ScalarRange[0];
      this->LastLabelMapScalarRange[1] = this->ScalarRange[1];
      this->LastLabelMapMaxLabel = maxLabel;
      this->LastLabelMapLabelCount = labels.size();
      this->MaskUpdateTime.Modified();
    }
  }

  return this->LabelMapTransferTexture != nullptr;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::SetMaskUniforms(void* uniforms, vtkVolume* vol)
{
  VolumeMapperUniforms* u = static_cast<VolumeMapperUniforms*>(uniforms);

  vtkImageData* maskInput = this->MaskInput;
  vtkVolumeProperty* property = vol->GetProperty();
  const bool hasMask =
    maskInput && property &&
    (this->MaskType == vtkGPUVolumeRayCastMapper::LabelMapMaskType ||
      this->MaskType == vtkGPUVolumeRayCastMapper::BinaryMaskType);

  if (hasMask)
  {
    u->UseMask = 1.0f;
    u->MaskBlendFactor = this->MaskBlendFactor;
    u->MaskType = (this->MaskType == vtkGPUVolumeRayCastMapper::BinaryMaskType) ? 1.0f : 0.0f;

    if (this->MaskType == vtkGPUVolumeRayCastMapper::BinaryMaskType)
    {
      // Binary mask: samples whose mask value (raw * scale + bias) is <= 0 are
      // skipped by the shader (mirrors OpenGL's BinaryMaskImplementation).
      // The mask texture is unorm-normalized at sample time, so scale=1/bias=0
      // recovers the [0,1] value directly (0 → skipped, 1 → kept).
      u->MaskScale = 1.0f;
      u->MaskBias = 0.0f;
      u->LabelMapNumLabels = 0.0f;
    }
    else
    {
      u->MaskScale = 1.0f;  // Default scale for unsigned char mask
      u->MaskBias = 0.0f;   // Default bias for unsigned char mask

      // Compute mask scale/bias based on the mask data type
      int cellFlag = 0;
      vtkDataArray* arr = this->GetScalars(
        maskInput, this->ScalarMode, this->ArrayAccessMode,
        this->ArrayId, this->ArrayName, cellFlag);
      if (arr)
      {
        int dataType = arr->GetDataType();
        if (dataType == VTK_UNSIGNED_CHAR)
        {
          u->MaskScale = 1.0f / 255.0f;
          u->MaskBias = 0.0f;
        }
        else if (dataType == VTK_CHAR)
        {
          u->MaskScale = 2.0f / 255.0f;
          u->MaskBias = -1.0f;
        }
        else if (dataType == VTK_UNSIGNED_SHORT)
        {
          u->MaskScale = 1.0f / 65535.0f;
          u->MaskBias = 0.0f;
        }
        else if (dataType == VTK_SHORT)
        {
          u->MaskScale = 2.0f / 65535.0f;
          u->MaskBias = -1.0f;
        }
        else
        {
          // For float or other types, compute from range
          double range[2];
          arr->GetRange(range);
          double dataRange = range[1] - range[0];
          if (dataRange > 0.0)
          {
            u->MaskScale = static_cast<float>(1.0 / dataRange);
            u->MaskBias = static_cast<float>(-range[0] / dataRange);
          }
        }
      }

      // Get the number of labels for quantization
      // numLabels = maxLabel + 1, so the shader can use label indices directly.
      std::set<int> labels = property->GetLabelMapLabels();
      int maxLabel = labels.empty() ? 0 : *(labels.rbegin());
      int numLabels = labels.empty() ? 0 : maxLabel + 1;
      u->LabelMapNumLabels = static_cast<float>(numLabels);
      // Determine scale to convert sampled mask value back to label index.
      // Unorm formats normalize to [0,1] at sample time, so we scale back.
      id<MTLTexture> maskTex = (__bridge id<MTLTexture>)this->MaskTexture;
      if (maskTex)
      {
        switch (maskTex.pixelFormat)
        {
          case MTLPixelFormatR8Unorm:
            u->MaskScale = 255.0f;
            break;
          case MTLPixelFormatR16Unorm:
            u->MaskScale = 65535.0f;
            break;
          default:
            u->MaskScale = 1.0f;
            break;
        }
      }
      else
      {
        u->MaskScale = 1.0f;
      }
      u->MaskBias = 0.0f;
    }
  }
  else
  {
    u->UseMask = 0.0f;
    u->MaskType = 0.0f;
    u->MaskBlendFactor = 0.0f;
    u->MaskScale = 1.0f;
    u->MaskBias = 0.0f;
    u->LabelMapNumLabels = 0.0f;
  }
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::ReleaseMaskResources()
{
  ReleaseMetalObject(this->MaskTexture);
  ReleaseMetalObject(this->LabelMapTransferTexture);
  ReleaseMetalObject(this->LabelMapGradientOpacityTexture);
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::EnsureMinMaxComputePipelines(void* mtlDeviceVoid)
{
  if (this->MinMaxComputePipeline && this->DilateComputePipeline)
  {
    return true;
  }

  if (!this->EnsureShaderLibrary(mtlDeviceVoid))
  {
    return false;
  }

  id<MTLDevice> dev = (__bridge id<MTLDevice>)mtlDeviceVoid;
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary;

  @autoreleasepool
  {
    if (!this->MinMaxComputePipeline)
    {
      id<MTLFunction> func = [library newFunctionWithName:@"volume_compute_minmax"];
      if (!func)
      {
        vtkErrorMacro("Failed to find volume_compute_minmax kernel");
        return false;
      }
      NSError* error = nil;
      id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:func error:&error];
      [func release];
      if (!pso)
      {
        vtkErrorMacro(<< "Failed to create minmax compute pipeline: "
                      << [[error localizedDescription] UTF8String]);
        return false;
      }
      AssignMetalObject(this->MinMaxComputePipeline, pso);
    }

    if (!this->DilateComputePipeline)
    {
      id<MTLFunction> func = [library newFunctionWithName:@"volume_dilate_minmax"];
      if (!func)
      {
        vtkErrorMacro("Failed to find volume_dilate_minmax kernel");
        return false;
      }
      NSError* error = nil;
      id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:func error:&error];
      [func release];
      if (!pso)
      {
        vtkErrorMacro(<< "Failed to create dilate compute pipeline: "
                      << [[error localizedDescription] UTF8String]);
        return false;
      }
      AssignMetalObject(this->DilateComputePipeline, pso);
    }

    if (!this->BlockReduceComputePipeline)
    {
      id<MTLFunction> func = [library newFunctionWithName:@"volume_reduce_minmax_blocks"];
      if (!func)
      {
        vtkErrorMacro("Failed to find volume_reduce_minmax_blocks kernel");
        return false;
      }
      NSError* error = nil;
      id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:func error:&error];
      [func release];
      if (!pso)
      {
        vtkErrorMacro(<< "Failed to create minmax block-reduce pipeline: "
                      << [[error localizedDescription] UTF8String]);
        return false;
      }
      AssignMetalObject(this->BlockReduceComputePipeline, pso);
    }

    // §38.16 (VTK_METAL_TEST_MM_MIP): reduce variant writing into mip level
    // log2(blockSize) of the fine lattice.
    if (!this->MipBlockReduceComputePipeline)
    {
      id<MTLFunction> func = [library newFunctionWithName:@"volume_reduce_minmax_mipblocks"];
      if (!func)
      {
        vtkErrorMacro("Failed to find volume_reduce_minmax_mipblocks kernel");
        return false;
      }
      NSError* error = nil;
      id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:func error:&error];
      [func release];
      if (!pso)
      {
        vtkErrorMacro(<< "Failed to create mip block-reduce pipeline: "
                      << [[error localizedDescription] UTF8String]);
        return false;
      }
      AssignMetalObject(this->MipBlockReduceComputePipeline, pso);
    }

    if (!this->SuperReduceComputePipeline)
    {
      id<MTLFunction> func =
        [library newFunctionWithName:@"volume_reduce_minmax_superblocks"];
      if (!func)
      {
        vtkErrorMacro("Failed to find volume_reduce_minmax_superblocks kernel");
        return false;
      }
      NSError* error = nil;
      id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:func error:&error];
      [func release];
      if (!pso)
      {
        vtkErrorMacro(<< "Failed to create minmax super-reduce pipeline: "
                      << [[error localizedDescription] UTF8String]);
        return false;
      }
      AssignMetalObject(this->SuperReduceComputePipeline, pso);
    }
  }

  return true;
}

//------------------------------------------------------------------------------
// Phase 7: Ensure compute pipelines for GPU data type conversion exist.
// Creates all 8 conversion pipelines (4 input types × half/float output).
// Returns true on success, false on failure (caller falls back to CPU).
bool vtkMetalGPUVolumeRayCastMapper::EnsureConversionPipelines(void* mtlDeviceVoid)
{
  if (!this->EnsureShaderLibrary(mtlDeviceVoid))
  {
    return false;
  }

  id<MTLDevice> dev = (__bridge id<MTLDevice>)mtlDeviceVoid;
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary;

  @autoreleasepool
  {
#define VTK_CREATE_CONVERT_PIPELINE(kernelName, member) \
    if (!member) { \
      id<MTLFunction> func = [library newFunctionWithName:@kernelName]; \
      if (!func) { \
        vtkErrorMacro("Failed to find " << kernelName << " kernel"); \
        return false; \
      } \
      NSError* error = nil; \
      id<MTLComputePipelineState> pso = \
        [dev newComputePipelineStateWithFunction:func error:&error]; \
      [func release]; \
      if (!pso) { \
        vtkErrorMacro(<< "Failed to create " << kernelName << " pipeline: " \
                      << [[error localizedDescription] UTF8String]); \
        return false; \
      } \
      AssignMetalObject(member, pso); \
    }

    VTK_CREATE_CONVERT_PIPELINE("volume_convert_short_to_half", this->ConvertShortToHalfPipeline);
    VTK_CREATE_CONVERT_PIPELINE("volume_convert_short_to_float", this->ConvertShortToFloatPipeline);
    VTK_CREATE_CONVERT_PIPELINE("volume_convert_int_to_half", this->ConvertIntToHalfPipeline);
    VTK_CREATE_CONVERT_PIPELINE("volume_convert_int_to_float", this->ConvertIntToFloatPipeline);
    VTK_CREATE_CONVERT_PIPELINE("volume_convert_uint_to_half", this->ConvertUIntToHalfPipeline);
    VTK_CREATE_CONVERT_PIPELINE("volume_convert_uint_to_float", this->ConvertUIntToFloatPipeline);
    VTK_CREATE_CONVERT_PIPELINE("volume_convert_float_to_half", this->ConvertFloatToHalfPipeline);
    VTK_CREATE_CONVERT_PIPELINE("volume_convert_ushort_to_uchar", this->ConvertUShortToUCharPipeline);
    // Note: VTK_DOUBLE kernels are not provided (Metal does not support 'double' in device address space).

#undef VTK_CREATE_CONVERT_PIPELINE
  }

  return true;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::ComputeMinMaxGPU(
  void* mtlDeviceVoid, void* mtlQueueVoid, vtkVolume* vol,
  vtkImageData* input, vtkDataArray* scalars)
{
  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mtlQueueVoid;

  id<MTLTexture> volTex = (__bridge id<MTLTexture>)this->VolumeTexture;
  if (!volTex)
  {
    return false;
  }
  // The occupancy lattice AND its consumer (the fragment-shader walk via
  // BuildGlobalPerBlockData's MinMaxInfo) live in DATA space: the march's
  // evalPoint and ctpScale/ctpOffset stay original-volume coords under
  // VOLTRANSPOSE (only volume fetches swizzle inside sampleVolumeScalar).
  // volume_compute_minmax likewise expects u.volDim* in data space — it
  // computes pos from them and applies the data->texture swizzle itself.
  // So derive DATA dims here, NOT the (axis-swapped) texture extents:
  // X-depth texture holds (D,H,W), Y-depth holds (W,D,H).
  int dims[3] = { static_cast<int>(volTex.width),
                  static_cast<int>(volTex.height),
                  static_cast<int>(volTex.depth) };
  if (this->VolumeTextureAxisDepth == 1)
  {
    std::swap(dims[0], dims[2]);
  }
  else if (this->VolumeTextureAxisDepth == 2)
  {
    std::swap(dims[1], dims[2]);
  }
  // (RG8 pair-packing also keeps axisDepth==0 but halves texture depth; its
  // lattice conventions were never defined for minmax — keep its pre-existing
  // texture-dims behavior untouched.)
  if (input && !VolumeRg8PairActive())
  {
    int inDims[3];
    input->GetDimensions(inDims);
    if (inDims[0] != dims[0] || inDims[1] != dims[1] || inDims[2] != dims[2])
    {
      vtkErrorMacro("ComputeMinMaxGPU: data-dims mismatch input="
                    << inDims[0] << "x" << inDims[1] << "x" << inDims[2]
                    << " recovered=" << dims[0] << "x" << dims[1] << "x"
                    << dims[2] << " axisDepth=" << this->VolumeTextureAxisDepth);
      std::copy(inDims, inDims + 3, dims);
    }
  }

  const int DS = ComputeMacrocellDownsample(this->SampleDistance, this->UseGPUMinMax);
  int mmDims[3] = {
    std::max(1, (dims[0] + DS - 1) / DS),
    std::max(1, (dims[1] + DS - 1) / DS),
    std::max(1, (dims[2] + DS - 1) / DS)
  };
  if (getenv("VTK_METAL_TEST_TR_DUMP") || getenv("VTK_METAL_TEST_TR_BENCH"))
  {
    fprintf(stderr,
      "[TRMM] gpu minmax lattice: axis=%d dataDims %dx%dx%d mmDims %dx%dx%d (DS=%d)\n",
      this->VolumeTextureAxisDepth, dims[0], dims[1], dims[2],
      mmDims[0], mmDims[1], mmDims[2], DS);
  }

  // Two-level summary wish (VTK_METAL_TEST_MM_BLOCKS): part of the cache key
  // below so enabling it mid-process (test-app reload) rebuilds the lattice.
  const bool wantBlocks =
    VolumeMinMaxBlocksWanted(this->UseGPUMinMax, this->SampleDistance);
  // §35.5 (VTK_METAL_TEST_MM_SUPER): third level; requires the blocks level.
  const bool wantSuper =
    VolumeMinMaxSuperWanted(this->UseGPUMinMax, this->SampleDistance);
  const int blockSize = VolumeMinMaxBlockSize(this->SampleDistance);
  const int blkDims[3] = {
    std::max(1, (mmDims[0] + blockSize - 1) / blockSize),
    std::max(1, (mmDims[1] + blockSize - 1) / blockSize),
    std::max(1, (mmDims[2] + blockSize - 1) / blockSize)
  };
  // Supers stay fixed at 64-cell tiles regardless of block size.
  const int blocksPerSuper = std::max(1, 64 / blockSize);
  const int sbDims[3] = {
    std::max(1, (blkDims[0] + blocksPerSuper - 1) / blocksPerSuper),
    std::max(1, (blkDims[1] + blocksPerSuper - 1) / blocksPerSuper),
    std::max(1, (blkDims[2] + blocksPerSuper - 1) / blocksPerSuper)
  };

  // Timestamp-based caching: skip recompute when nothing changed. The grid
  // dims are part of the key so switching sample-distance tiers (which moves
  // DS) rebuilds the occupancy grid for the new cell size.
  if (getenv("VTK_METAL_TEST_TR_DUMP") || getenv("VTK_METAL_TEST_TR_BENCH"))
  {
    vtkVolumeProperty* propDbg = vol ? vol->GetProperty() : nullptr;
    vtkPiecewiseFunction* opFuncDbg = propDbg ? propDbg->GetScalarOpacity() : nullptr;
    fprintf(stderr,
      "[TRMMCACHE] tex=%d input=%d opFunc=%d inMT=%llu upMT=%llu opMT=%llu "
      "cachedDims %dx%dx%d want %dx%dx%d\n",
      this->MinMaxTexture ? 1 : 0, input ? 1 : 0, opFuncDbg ? 1 : 0,
      static_cast<unsigned long long>(input ? input->GetMTime() : 0),
      static_cast<unsigned long long>(this->MinMaxUploadTime.GetMTime()),
      static_cast<unsigned long long>(opFuncDbg ? opFuncDbg->GetMTime() : 0),
      this->MinMaxDims[0], this->MinMaxDims[1], this->MinMaxDims[2],
      mmDims[0], mmDims[1], mmDims[2]);
  }
  if (input && this->MinMaxTexture)
  {
    vtkVolumeProperty* property = vol ? vol->GetProperty() : nullptr;
    vtkPiecewiseFunction* opFunc = property ? property->GetScalarOpacity() : nullptr;
    if (opFunc &&
        input->GetMTime() <= this->MinMaxUploadTime.GetMTime() &&
        opFunc->GetMTime() <= this->MinMaxUploadTime.GetMTime() &&
        this->MinMaxDims[0] == mmDims[0] &&
        this->MinMaxDims[1] == mmDims[1] &&
        this->MinMaxDims[2] == mmDims[2] &&
        (!wantBlocks ||
         (this->MinMaxBlockTexture != nullptr &&
          this->MinMaxBlockSize == blockSize &&
          this->MinMaxBlockDims[0] == blkDims[0] &&
          this->MinMaxBlockDims[1] == blkDims[1] &&
          this->MinMaxBlockDims[2] == blkDims[2])) &&
        (!wantSuper ||
         (this->MinMaxSuperTexture != nullptr &&
          this->MinMaxSuperDims[0] == sbDims[0] &&
          this->MinMaxSuperDims[1] == sbDims[1] &&
          this->MinMaxSuperDims[2] == sbDims[2])))
    {
      return true;
    }
  }

  this->MinMaxDims[0] = mmDims[0];
  this->MinMaxDims[1] = mmDims[1];
  this->MinMaxDims[2] = mmDims[2];

  if (!this->EnsureMinMaxComputePipelines(mtlDeviceVoid))
  {
    return false;
  }

  @autoreleasepool
  {
    // Release old scratch texture (helper creates its own)
    ReleaseMetalObject(this->MinMaxScratchTexture);

    // --- Build opacity prefix table from transfer function ---
    vtkVolumeProperty* property = vol ? vol->GetProperty() : nullptr;
    vtkPiecewiseFunction* opFunc = property ? property->GetScalarOpacity() : nullptr;
    if (!opFunc)
    {
      return false;
    }

    double opacityTable[256];
    opFunc->GetTable(this->ScalarRange[0], this->ScalarRange[1], 256, opacityTable);

    uint32_t opacityPrefix[257];
    opacityPrefix[0] = 0;
    for (int i = 0; i < 256; ++i)
    {
      opacityPrefix[i + 1] = opacityPrefix[i] + (opacityTable[i] > 0.0 ? 1u : 0u);
    }

    double scalarRange = this->ScalarRange[1] - this->ScalarRange[0];
    if (scalarRange <= 0.0) scalarRange = 1.0;

    // --- Build uniforms ---
    // IMPORTANT: texture sampler returns normalized values for R8Unorm/R16Unorm
    // formats but raw values for R32Float.  We must adjust scalarMin/scalarScale
    // to work in the sampler's output space (raw/normalized) so the TF index
    // computation matches the CPU path.
    float normFactor = this->ScalarNormalizationFactor;

    MinMaxComputeUniforms u;
    u.mmDimX = static_cast<uint32_t>(mmDims[0]);
    u.mmDimY = static_cast<uint32_t>(mmDims[1]);
    u.mmDimZ = static_cast<uint32_t>(mmDims[2]);
    u.volDimX = static_cast<uint32_t>(dims[0]);
    u.volDimY = static_cast<uint32_t>(dims[1]);
    u.volDimZ = static_cast<uint32_t>(dims[2]);
    u.ds = static_cast<float>(DS);
    u.scalarMin = static_cast<float>(this->ScalarRange[0] / normFactor);
    u.scalarScale = static_cast<float>(255.0 * normFactor / scalarRange);
    u._pad = 0.0f;
    // Orientation code for the data-space sampling kernels: 0=identity,
    // 1=X-depth (.zyx), 2=Y-depth (.xzy).
    u.volTransposed = this->VolumeTextureAxisDepth;
    memcpy(u.opacityPrefix, opacityPrefix, sizeof(opacityPrefix));
    // §33.2 item 2 (VTK_METAL_TEST_MM_EPS): ship the TF opacity table so the
    // kernel can threshold max achievable opacity (mmEps=0 keeps exact
    // prefix-sum semantics). Same table the prefix was built from.
    for (int i = 0; i < 256; ++i)
    {
      u.opacityLut[i] = static_cast<float>(std::max(0.0, opacityTable[i]));
    }
    u.mmEps = VolumeMinMaxEps();
    u._pad2[0] = u._pad2[1] = u._pad2[2] = 0u;

    // --- Reuse or create persistent MinMax texture ---
    // §38.16 (VTK_METAL_TEST_MM_MIP): when the mip-fusion fix is active the
    // fine lattice carries the block summary in mip level log2(blockSize),
    // so it allocates with that many mip levels (level 0 layout/sampling is
    // unchanged).
    const bool wantMip =
      wantBlocks && getenv("VTK_METAL_TEST_MM_MIP") != nullptr &&
      // The mirror blit runs after mmEnc, so a same-encoder super reduce
      // would read a stale standalone texture — keep MIP and supers mutually
      // exclusive (supers stay an opt-in diagnostic).
      !VolumeMinMaxSuperWanted(this->UseGPUMinMax, this->SampleDistance);
    NSUInteger mmMipLevels = 1;
    int mmBsLod = 0;
    if (wantMip)
    {
      while ((1 << mmBsLod) < blockSize && mmBsLod < 5)
        ++mmBsLod;
      if ((1 << mmBsLod) < blockSize)
        ++mmBsLod; // non-power-of-two block size: use the next level up
      mmMipLevels = static_cast<NSUInteger>(mmBsLod) + 1;
    }
    id<MTLTexture> permTex = (__bridge id<MTLTexture>)this->MinMaxTexture;
    if (!permTex ||
        permTex.width != static_cast<NSUInteger>(mmDims[0]) ||
        permTex.height != static_cast<NSUInteger>(mmDims[1]) ||
        permTex.depth != static_cast<NSUInteger>(mmDims[2]) ||
        permTex.mipmapLevelCount != mmMipLevels ||
        permTex.storageMode != MTLStorageModePrivate ||
        permTex.pixelFormat != MTLPixelFormatR8Unorm)
    {
      permTex = CreateR8MinMaxTexture(
        device,
        mmDims[0],
        mmDims[1],
        mmDims[2],
        MTLStorageModePrivate,
        MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
        mmMipLevels);
      if (!permTex)
      {
        vtkErrorMacro("Failed to create persistent min-max texture");
        return false;
      }
      AssignMetalObject(this->MinMaxTexture, permTex);
    }

    // --- Command buffer ---
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    cmdBuf.label = @"VTK Volume MinMax Compute";
    id<MTLComputeCommandEncoder> mmEnc = [cmdBuf computeCommandEncoder];
    mmEnc.label = @"Volume Compute MinMax";

    EncodeGPUMinMaxDilation(
      mmEnc,
      device,
      volTex,
      permTex,
      dims,
      mmDims,
      u,
      (__bridge id<MTLComputePipelineState>)this->MinMaxComputePipeline,
      (__bridge id<MTLComputePipelineState>)this->DilateComputePipeline);

    // Two-level summary (VTK_METAL_TEST_MM_BLOCKS): reduce the DILATED fine
    // lattice into whole-block all-empty marks. Runs on the same encoder right
    // after dilation so ordering is implicit.
    if (wantBlocks)
    {
      id<MTLTexture> blockTex = (__bridge id<MTLTexture>)this->MinMaxBlockTexture;
      // §38.16 (VTK_METAL_TEST_MM_PADTEX): allocate the summary with dims
      // rounded up to >=64^3. Content occupies the same [0,blkDims) region
      // (reduce kernel writes only real texels; march reads are index-clamped
      // to the real dims), so decisions are unchanged — this only moves the
      // resource's size class/placement. Diagnostic for the tiny-allocation
      // placement hypothesis behind the small-viewport blocks-on tax.
      // Restricted to no-supers configs: the superblock reduce derives its
      // bounds from get_width() and would misread pad texels as real blocks.
      int padDims[3] = { blkDims[0], blkDims[1], blkDims[2] };
      if (getenv("VTK_METAL_TEST_MM_PADTEX") != nullptr && !wantSuper)
      {
        for (int k = 0; k < 3; ++k)
          if (padDims[k] < 64)
            padDims[k] = 64;
      }
      if (!blockTex ||
          blockTex.width != static_cast<NSUInteger>(padDims[0]) ||
          blockTex.height != static_cast<NSUInteger>(padDims[1]) ||
          blockTex.depth != static_cast<NSUInteger>(padDims[2]) ||
          blockTex.pixelFormat != MTLPixelFormatR8Unorm)
      {
        blockTex = CreateR8MinMaxTexture(
          device,
          padDims[0],
          padDims[1],
          padDims[2],
          MTLStorageModePrivate,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
        if (!blockTex)
        {
          vtkErrorMacro("Failed to create minmax block-summary texture");
          return false;
        }
        AssignMetalObject(this->MinMaxBlockTexture, blockTex);
      }
      this->MinMaxBlockDims[0] = blkDims[0];
      this->MinMaxBlockDims[1] = blkDims[1];
      this->MinMaxBlockDims[2] = blkDims[2];
      this->MinMaxBlockSize = blockSize;

      // Empty-block counter (shared uint): zeroed on the CPU before the
      // dispatch reads it; the command buffer's completion handler converts
      // it into MinMaxEmptyBlockFraction without stalling the frame.
      if (!this->MinMaxCountBuffer)
      {
        id<MTLBuffer> cntBuf = [device newBufferWithLength:sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
        if (!cntBuf)
        {
          vtkErrorMacro("Failed to create minmax counter buffer");
          return false;
        }
        AssignMetalObject(this->MinMaxCountBuffer, cntBuf);
      }
      const NSUInteger totalBlocks =
        static_cast<NSUInteger>(blkDims[0]) * blkDims[1] * blkDims[2];
      id<MTLBuffer> cntBuf = (__bridge id<MTLBuffer>)this->MinMaxCountBuffer;
      memset(cntBuf.contents, 0, sizeof(uint32_t));

      [mmEnc setComputePipelineState:
        (__bridge id<MTLComputePipelineState>)(
          wantMip ? this->MipBlockReduceComputePipeline
                  : this->BlockReduceComputePipeline)];
      [mmEnc setTexture:permTex atIndex:0];
      // texture(1) is the reduce DESTINATION: the fine lattice's mip level
      // under MM_MIP, the standalone block texture otherwise.
      [mmEnc setTexture:(wantMip ? permTex : blockTex) atIndex:1];
      [mmEnc setBytes:&blockSize length:sizeof(blockSize) atIndex:0];
      [mmEnc setBuffer:(__bridge id<MTLBuffer>)this->MinMaxCountBuffer
                 offset:0 atIndex:1];
      if (wantMip)
      {
        // §38.16: destination mip level for the fused summary.
        uint dstLod = static_cast<uint>(mmBsLod);
        [mmEnc setBytes:&dstLod length:sizeof(dstLod) atIndex:2];
      }
      MTLSize blkGrid = MTLSizeMake(
        static_cast<NSUInteger>(blkDims[0]),
        static_cast<NSUInteger>(blkDims[1]),
        static_cast<NSUInteger>(blkDims[2]));
      NSUInteger blkTg = 8;
      MTLSize blkThreadgroup = MTLSizeMake(
        std::min(blkTg, static_cast<NSUInteger>(blkDims[0])),
        std::min(blkTg, static_cast<NSUInteger>(blkDims[1])),
        std::min(blkTg, static_cast<NSUInteger>(blkDims[2])));
      [mmEnc dispatchThreads:blkGrid threadsPerThreadgroup:blkThreadgroup];
      this->MinMaxEmptyBlockTotal = totalBlocks;
      __weak vtkMetalGPUVolumeRayCastMapper* weakThis = this;
      id<MTLBuffer> handlerBuf = cntBuf;
      [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer>) {
        vtkMetalGPUVolumeRayCastMapper* m = weakThis;
        if (!m) return;
        const uint32_t empties =
          *static_cast<const uint32_t*>(handlerBuf.contents);
        const float frac = (totalBlocks > 0 && empties <= totalBlocks)
          ? static_cast<float>(empties) / static_cast<float>(totalBlocks)
          : 1.0f;
        m->MinMaxEmptyBlockFraction.store(frac, std::memory_order_relaxed);
        if (getenv("VTK_METAL_TEST_TR_DUMP") || getenv("VTK_METAL_TEST_TR_BENCH"))
        {
          fprintf(stderr, "[TRMM] empty-block fraction %.4f (%u/%zu)\n",
            frac, empties, static_cast<size_t>(totalBlocks));
        }
      }];
      if (getenv("VTK_METAL_TEST_TR_DUMP") || getenv("VTK_METAL_TEST_TR_BENCH"))
      {
        fprintf(stderr, "[TRMM] block summary %dx%dx%d (block=%d cells)\n",
          blkDims[0], blkDims[1], blkDims[2], blockSize);
      }

      // §35.5 (VTK_METAL_TEST_MM_SUPER): reduce the BLOCK summary into
      // all-empty 8³-block groups. Same encoder, so ordering after the block
      // reduce is implicit.
      if (wantSuper)
      {
        id<MTLTexture> superTex = (__bridge id<MTLTexture>)this->MinMaxSuperTexture;
        if (!superTex ||
            superTex.width != static_cast<NSUInteger>(sbDims[0]) ||
            superTex.height != static_cast<NSUInteger>(sbDims[1]) ||
            superTex.depth != static_cast<NSUInteger>(sbDims[2]) ||
            superTex.pixelFormat != MTLPixelFormatR8Unorm)
        {
          superTex = CreateR8MinMaxTexture(
            device,
            sbDims[0],
            sbDims[1],
            sbDims[2],
            MTLStorageModePrivate,
            MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
          if (!superTex)
          {
            vtkErrorMacro("Failed to create minmax super-summary texture");
            return false;
          }
          AssignMetalObject(this->MinMaxSuperTexture, superTex);
        }
        this->MinMaxSuperDims[0] = sbDims[0];
        this->MinMaxSuperDims[1] = sbDims[1];
        this->MinMaxSuperDims[2] = sbDims[2];

        [mmEnc setComputePipelineState:
          (__bridge id<MTLComputePipelineState>)this->SuperReduceComputePipeline];
        [mmEnc setTexture:blockTex atIndex:0];
        [mmEnc setTexture:superTex atIndex:1];
        [mmEnc setBytes:&blocksPerSuper length:sizeof(blocksPerSuper) atIndex:0];
        MTLSize sbGrid = MTLSizeMake(
          static_cast<NSUInteger>(sbDims[0]),
          static_cast<NSUInteger>(sbDims[1]),
          static_cast<NSUInteger>(sbDims[2]));
        NSUInteger sbTg = 8;
        MTLSize sbThreadgroup = MTLSizeMake(
          std::min(sbTg, static_cast<NSUInteger>(sbDims[0])),
          std::min(sbTg, static_cast<NSUInteger>(sbDims[1])),
          std::min(sbTg, static_cast<NSUInteger>(sbDims[2])));
        [mmEnc dispatchThreads:sbGrid threadsPerThreadgroup:sbThreadgroup];
        if (getenv("VTK_METAL_TEST_TR_DUMP") || getenv("VTK_METAL_TEST_TR_BENCH"))
        {
          fprintf(stderr, "[TRMM] super summary %dx%dx%d (8 blocks/group)\n",
            sbDims[0], sbDims[1], sbDims[2]);
        }
      }
      else
      {
        ReleaseMetalObject(this->MinMaxSuperTexture);
      }
    }
    else if (this->MinMaxBlockTexture || this->MinMaxSuperTexture)
    {
      // Feature turned off (or GPU path disabled): drop the summaries so the
      // cache key above stays honest.
      ReleaseMetalObject(this->MinMaxBlockTexture);
      this->MinMaxBlockSize = 0;
      ReleaseMetalObject(this->MinMaxSuperTexture);
    }

    [mmEnc endEncoding];
    // §38.16 (VTK_METAL_TEST_MM_MIP): mirror the fused summary back into the
    // standalone block texture so its other consumers (segment build, super
    // reduce, non-MIP pipelines) keep reading current values.
    if (wantMip)
    {
      id<MTLBlitCommandEncoder> blitEnc = [cmdBuf blitCommandEncoder];
      blitEnc.label = @"Volume MipSummary Mirror";
      [blitEnc copyFromTexture:permTex
                   sourceSlice:0
                   sourceLevel:static_cast<NSUInteger>(mmBsLod)
                  sourceOrigin:MTLOriginMake(0, 0, 0)
                    sourceSize:MTLSizeMake(
                                 static_cast<NSUInteger>(blkDims[0]),
                                 static_cast<NSUInteger>(blkDims[1]),
                                 static_cast<NSUInteger>(blkDims[2]))
                     toTexture:(__bridge id<MTLTexture>)this->MinMaxBlockTexture
              destinationSlice:0
              destinationLevel:0
             destinationOrigin:MTLOriginMake(0, 0, 0)];
      [blitEnc endEncoding];
    }
    [cmdBuf commit];

    // TEMP-DIAG (VTK_METAL_TEST_TR_DUMP): verify block-summary consistency —
    // every block marked empty must cover only empty fine cells. Copies both
    // lattices to shared staging and checks on the CPU.
    if (wantBlocks &&
        (getenv("VTK_METAL_TEST_TR_DUMP") || getenv("VTK_METAL_TEST_TR_BENCH")))
    {
      [cmdBuf waitUntilCompleted];
      id<MTLTexture> fineSrc = (__bridge id<MTLTexture>)this->MinMaxTexture;
      id<MTLTexture> blkSrc = (__bridge id<MTLTexture>)this->MinMaxBlockTexture;
      id<MTLTexture> fineStg = CreateR8MinMaxTexture(device,
        static_cast<int>(fineSrc.width), static_cast<int>(fineSrc.height),
        static_cast<int>(fineSrc.depth), MTLStorageModeShared,
        MTLTextureUsageShaderRead);
      id<MTLTexture> blkStg = CreateR8MinMaxTexture(device,
        static_cast<int>(blkSrc.width), static_cast<int>(blkSrc.height),
        static_cast<int>(blkSrc.depth), MTLStorageModeShared,
        MTLTextureUsageShaderRead);
      id<MTLCommandBuffer> cpBuf = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [cpBuf blitCommandEncoder];
      [blit copyFromTexture:fineSrc toTexture:fineStg];
      [blit copyFromTexture:blkSrc toTexture:blkStg];
      [blit endEncoding];
      [cpBuf commit];
      [cpBuf waitUntilCompleted];

      const int fd[3] = { static_cast<int>(fineSrc.width),
                          static_cast<int>(fineSrc.height),
                          static_cast<int>(fineSrc.depth) };
      const int bd[3] = { static_cast<int>(blkSrc.width),
                          static_cast<int>(blkSrc.height),
                          static_cast<int>(blkSrc.depth) };
      std::vector<uint8_t> fbytes(
        static_cast<size_t>(fd[0]) * fd[1] * fd[2]);
      std::vector<uint8_t> bbytes(
        static_cast<size_t>(bd[0]) * bd[1] * bd[2]);
      [fineStg getBytes:fbytes.data()
          bytesPerRow:static_cast<NSUInteger>(fd[0])
          bytesPerImage:static_cast<NSUInteger>(fd[0]) * fd[1]
          fromRegion:MTLRegionMake3D(0, 0, 0,
            static_cast<NSUInteger>(fd[0]), static_cast<NSUInteger>(fd[1]),
            static_cast<NSUInteger>(fd[2]))
          mipmapLevel:0 slice:0];
      [blkStg getBytes:bbytes.data()
          bytesPerRow:static_cast<NSUInteger>(bd[0])
          bytesPerImage:static_cast<NSUInteger>(bd[0]) * bd[1]
          fromRegion:MTLRegionMake3D(0, 0, 0,
            static_cast<NSUInteger>(bd[0]), static_cast<NSUInteger>(bd[1]),
            static_cast<NSUInteger>(bd[2]))
          mipmapLevel:0 slice:0];
      long violations = 0;
      int bsZ = VolumeMinMaxBlockSize(this->SampleDistance);
      for (int bz = 0; bz < bd[2]; ++bz)
        for (int by = 0; by < bd[1]; ++by)
          for (int bx = 0; bx < bd[0]; ++bx)
          {
            if (bbytes[(static_cast<size_t>(bz) * bd[1] + by) * bd[0] + bx] < 128)
              continue; // solid block
            const int x0 = bx * blockSize, y0 = by * blockSize, z0 = bz * blockSize;
            const int x1 = std::min(x0 + blockSize, fd[0]);
            const int y1 = std::min(y0 + blockSize, fd[1]);
            const int z1 = std::min(z0 + blockSize, fd[2]);
            for (int z = z0; z < z1 && violations < 32; ++z)
              for (int y = y0; y < y1 && violations < 32; ++y)
                for (int x = x0; x < x1 && violations < 32; ++x)
                {
                  if (fbytes[(static_cast<size_t>(z) * fd[1] + y) * fd[0] + x] < 128)
                  {
                    ++violations;
                    fprintf(stderr,
                      "[TRMM] VIOLATION block(%d,%d,%d) empty covers fine "
                      "SOLID cell (%d,%d,%d)\n", bx, by, bz, x, y, z);
                  }
                }
          }
      fprintf(stderr,
        "[TRMM] block-consistency: %ld violations (fine %dx%dx%d blocks %dx%dx%d bs=%d)\n",
        violations, fd[0], fd[1], fd[2], bd[0], bd[1], bd[2], blockSize);
      [fineStg release];
      [blkStg release];
    }

    this->MinMaxUploadTime.Modified();
  }

  return this->MinMaxTexture != nullptr;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::UpdateMinMaxTexture(
  void* mtlDeviceVoid, vtkVolume* vol, vtkImageData* input, vtkDataArray* scalars,
  bool skipGlobalTexture)
{
  if (!input || !scalars || !vol)
  {
    return false;
  }

  vtkVolumeProperty* property = vol->GetProperty();
  vtkPiecewiseFunction* opFunc = property ? property->GetScalarOpacity() : nullptr;

  // Without an opacity function we cannot determine empty space — disable acceleration
  if (!opFunc)
  {
    return false;
  }

  int dims[3];
  input->GetDimensions(dims);

  // Downsample factor for the min-max occupancy grid (DS voxels per cell).
  // The finer DS=2 grid is only used when the lattice is computed on the GPU
  // (UseGPUMinMax): its precise per-sample skip pays off there. The CPU path
  // keeps DS=4 so the slower CPU-computed occupancy behaves like the
  // pre-adaptive baseline instead of degrading the march.
  const int DS = ComputeMacrocellDownsample(this->SampleDistance, this->UseGPUMinMax);
  int mmDims[3] = {
    std::max(1, (dims[0] + DS - 1) / DS),
    std::max(1, (dims[1] + DS - 1) / DS),
    std::max(1, (dims[2] + DS - 1) / DS)
  };

  // When skipGlobalTexture is true (partitioned mode), MinMaxTexture is
  // intentionally never created. Use MacrocellScalarMin as the validity
  // sentinel instead, so we don't re-run the full voxel scan every frame.
  bool doReload;
  if (skipGlobalTexture)
  {
    doReload = this->MacrocellScalarMin.empty();
  }
  else
  {
    doReload = (this->MinMaxTexture == nullptr);
  }
  doReload |= (input->GetMTime() > this->MinMaxUploadTime.GetMTime());
  doReload |= (opFunc->GetMTime() > this->MinMaxUploadTime.GetMTime());
  // Sample-distance tier changes move DS and therefore the grid dims: rebuild.
  doReload |= (this->MinMaxDims[0] != mmDims[0] ||
               this->MinMaxDims[1] != mmDims[1] ||
               this->MinMaxDims[2] != mmDims[2]);

  if (!doReload)
  {
    if (skipGlobalTexture)
    {
      return !this->MacrocellScalarMin.empty();
    }
    return this->MinMaxTexture != nullptr;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

    this->MinMaxDims[0] = mmDims[0];
    this->MinMaxDims[1] = mmDims[1];
    this->MinMaxDims[2] = mmDims[2];

    int dataType = scalars->GetDataType();
    int extents[6];
    input->GetExtent(extents);
    vtkIdType inc[3];
    input->GetIncrements(inc);

    const void* dataPtr = scalars->GetVoidPointer(0);

    // Precompute the 256-entry opacity lookup table ONCE.
    // IsBlockEmpty rebuilds this per macrocell (~1.4M times for a CT volume);
    // doing it here eliminates that redundancy entirely.
    double opacityTable[256];
    opFunc->GetTable(this->ScalarRange[0], this->ScalarRange[1], 256, opacityTable);
    const double scalarRange = this->ScalarRange[1] - this->ScalarRange[0];
    const double rangeRecip = (scalarRange > 0.0) ? (255.0 / scalarRange) : 1.0;
    const double rangeOffset = this->ScalarRange[0];

    // Snapshot loop-invariant values for the parallel lambda
    const int mmDims0 = mmDims[0];
    const int mmDims1 = mmDims[1];
    const int mmDims2 = mmDims[2];
    const int extXSize = extents[1] - extents[0] + 1;
    const int extYSize = extents[3] - extents[2] + 1;
    const int extZSize = extents[5] - extents[4] + 1;
    const vtkIdType inc0 = inc[0];
    const vtkIdType inc1 = inc[1];
    const vtkIdType inc2 = inc[2];

    vtkIdType numCells = static_cast<vtkIdType>(mmDims0) * mmDims1 * mmDims2;

    // Per-macrocell scalar min/max — consumed later by UpdateBlockTextures
    // to compute per-block ranges without re-walking every voxel.
    this->MacrocellScalarMin.resize(numCells, 1e30f);
    this->MacrocellScalarMax.resize(numCells, -1e30f);
    float* mcMin = this->MacrocellScalarMin.data();
    float* mcMax = this->MacrocellScalarMax.data();

    // RAW buffer for occupancy (only needed for global texture)
    std::vector<uint8_t> rawMinMax;
    if (!skipGlobalTexture)
    {
      rawMinMax.resize(numCells, 255);
    }

    // Parallel scan: each macrocell is independent — perfect for vtkSMPTools
    vtkSMPTools::For(0, numCells, [&](vtkIdType begin, vtkIdType end) {
      for (vtkIdType cellIdx = begin; cellIdx < end; ++cellIdx)
      {
        const int gx = static_cast<int>(cellIdx % mmDims0);
        const int gy = static_cast<int>((cellIdx / mmDims0) % mmDims1);
        const int gz = static_cast<int>(cellIdx / (mmDims0 * mmDims1));

        const int zStart = gz * DS;
        const int zEnd = std::min(zStart + DS, extZSize);
        const int yStart = gy * DS;
        const int yEnd = std::min(yStart + DS, extYSize);
        const int xStart = gx * DS;
        const int xEnd = std::min(xStart + DS, extXSize);

        float cellMin = 1e30f;
        float cellMax = -1e30f;

        for (int z = zStart; z < zEnd; ++z)
        {
          for (int y = yStart; y < yEnd; ++y)
          {
            for (int x = xStart; x < xEnd; ++x)
            {
              float v = 0.0f;
              switch (dataType)
              {
                case VTK_FLOAT:
                  v = static_cast<float>(
                    static_cast<const float*>(dataPtr)[z * inc2 + y * inc1 + x * inc0]);
                  break;
                case VTK_UNSIGNED_CHAR:
                  v = static_cast<float>(
                    static_cast<const unsigned char*>(dataPtr)[z * inc2 + y * inc1 + x * inc0]);
                  break;
                case VTK_UNSIGNED_SHORT:
                  v = static_cast<float>(
                    static_cast<const unsigned short*>(dataPtr)[z * inc2 + y * inc1 + x * inc0]);
                  break;
                case VTK_SHORT:
                  v = static_cast<float>(
                    static_cast<const short*>(dataPtr)[z * inc2 + y * inc1 + x * inc0]);
                  break;
                case VTK_INT:
                  v = static_cast<float>(
                    static_cast<const int*>(dataPtr)[z * inc2 + y * inc1 + x * inc0]);
                  break;
                case VTK_UNSIGNED_INT:
                  v = static_cast<float>(
                    static_cast<const unsigned int*>(dataPtr)[z * inc2 + y * inc1 + x * inc0]);
                  break;
                default:
                {
                  vtkIdType tupleIdx = z * (inc2 / inc0) + y * (inc1 / inc0) + x;
                  v = static_cast<float>(scalars->GetComponent(tupleIdx, 0));
                  break;
                }
              }
              if (v < cellMin) cellMin = v;
              if (v > cellMax) cellMax = v;
            }
          }
        }

        // Store macrocell scalar range for later block-range reduction
        mcMin[cellIdx] = cellMin;
        mcMax[cellIdx] = cellMax;

        if (!skipGlobalTexture)
        {
          rawMinMax[cellIdx] = ScalarRangeTouchesOpacity(cellMin, cellMax, opacityTable, rangeOffset, rangeRecip) ? 0 : 255;
        }
      }
    });

    if (!skipGlobalTexture)
    {
      std::vector<uint8_t> minMaxData = DilateOccupancy3D(rawMinMax, mmDims0, mmDims1, mmDims2);

      AssignR8MinMaxTexture(this->MinMaxTexture, device, minMaxData, mmDims0, mmDims1, mmDims2);
    }

    this->MinMaxUploadTime.Modified();
  }

  if (skipGlobalTexture)
  {
    return !this->MacrocellScalarMin.empty();
  }
  return this->MinMaxTexture != nullptr;
}


//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::IsCameraInside(
  vtkRenderer* ren, vtkVolume* vol)
{
  vtkNew<vtkMatrix4x4> dataToWorld;
  vol->GetModelToWorldMatrix(dataToWorld);

  vtkCamera* cam = ren->GetActiveCamera();

  double planes[24];
  cam->GetFrustumPlanes(ren->GetTiledAspectRatio(), planes);

  // Transform 8 bounding-box corners from data-space to world-space
  double geometry[24] = {
    this->ModelBounds[0], this->ModelBounds[2], this->ModelBounds[4],
    this->ModelBounds[1], this->ModelBounds[2], this->ModelBounds[4],
    this->ModelBounds[1], this->ModelBounds[3], this->ModelBounds[4],
    this->ModelBounds[0], this->ModelBounds[3], this->ModelBounds[4],
    this->ModelBounds[0], this->ModelBounds[2], this->ModelBounds[5],
    this->ModelBounds[1], this->ModelBounds[2], this->ModelBounds[5],
    this->ModelBounds[1], this->ModelBounds[3], this->ModelBounds[5],
    this->ModelBounds[0], this->ModelBounds[3], this->ModelBounds[5],
  };

  double in[4];
  in[3] = 1.0;
  double out[4];
  double worldGeometry[24];

  for (int i = 0; i < 8; ++i)
  {
    in[0] = geometry[i * 3];
    in[1] = geometry[i * 3 + 1];
    in[2] = geometry[i * 3 + 2];
    dataToWorld->MultiplyPoint(in, out);
    worldGeometry[i * 3] = out[0] / out[3];
    worldGeometry[i * 3 + 1] = out[1] / out[3];
    worldGeometry[i * 3 + 2] = out[2] / out[3];
  }

  // Test if near frustum plane (index 4*4=16) intersects the bounding box
  // Returns true if some corners are on positive side and some on negative side
  bool hasPositive = false;
  bool hasNegative = false;
  bool hasZero = false;

  for (int i = 0; i < 8; ++i)
  {
    double val = planes[4 * 4] * worldGeometry[i * 3] +
      planes[4 * 4 + 1] * worldGeometry[i * 3 + 1] +
      planes[4 * 4 + 2] * worldGeometry[i * 3 + 2] +
      planes[4 * 4 + 3];

    if (val < 0)
    {
      hasNegative = true;
    }
    else if (val > 0)
    {
      hasPositive = true;
    }
    else
    {
      hasZero = true;
    }
  }

  return hasZero || (hasNegative && hasPositive);
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::SetClippingPlaneUniforms(
  void* uniformsVoid, vtkRenderer* ren, vtkVolume* vol,
  vtkMatrix4x4* modelMatrix, vtkMatrix4x4* invModelMatrix)
{
  VolumeMapperUniforms* uniforms = static_cast<VolumeMapperUniforms*>(uniformsVoid);

  if (!this->GetClippingPlanes())
  {
    uniforms->UseClipping = 0.0f;
    uniforms->NumClippingPlanes = 0.0f;
    return;
  }

  uniforms->UseClipping = 1.0f;

  VolumeBounds bounds;
  bounds.Min[0] = this->ModelBounds[0];
  bounds.Max[0] = this->ModelBounds[1];
  bounds.Min[1] = this->ModelBounds[2];
  bounds.Max[1] = this->ModelBounds[3];
  bounds.Min[2] = this->ModelBounds[4];
  bounds.Max[2] = this->ModelBounds[5];
  for (int k = 0; k < 3; ++k)
  {
    bounds.Size[k] = bounds.Max[k] - bounds.Min[k];
    if (bounds.Size[k] < 1e-10)
      bounds.Size[k] = 1.0;
  }

  // Planes are stored as alternating origin/normal float[4] each 16 bytes.
  // ClippingPlane0Origin is at the base of the 8-plane block; each plane
  // occupies 8 floats (origin[4] + normal[4]).
  float* planeData = &uniforms->ClippingPlane0Origin[0];

  int numPlanes = 0;

  this->ClippingPlanes->InitTraversal();
  vtkPlane* plane;
  while ((plane = this->ClippingPlanes->GetNextItem()) && numPlanes < 8)
  {
    double planeOrigin[3], planeNormal[3];
    plane->GetOrigin(planeOrigin);
    plane->GetNormal(planeNormal);

    // Origin: transform as a point from world to model/data space.
    double originLocal[4] = { planeOrigin[0], planeOrigin[1], planeOrigin[2], 1.0 };
    invModelMatrix->MultiplyPoint(originLocal, originLocal);

    if (fabs(originLocal[3]) > 1e-12)
    {
      originLocal[0] /= originLocal[3];
      originLocal[1] /= originLocal[3];
      originLocal[2] /= originLocal[3];
    }

    double originVol[3] = {
      (originLocal[0] - bounds.Min[0]) / bounds.Size[0],
      (originLocal[1] - bounds.Min[1]) / bounds.Size[1],
      (originLocal[2] - bounds.Min[2]) / bounds.Size[2]
    };

    // Normal: transform as a normal/covector.
    // We want transpose(modelMatrix) * normalWorld.
    // VTK matrices are accessed as GetElement(row, col).
    // For column c of transpose(M):
    //   (M^T n)[c] = sum_r M[r][c] * n[r]
    double normalModel[3];
    normalModel[0] =
      modelMatrix->GetElement(0, 0) * planeNormal[0] +
      modelMatrix->GetElement(1, 0) * planeNormal[1] +
      modelMatrix->GetElement(2, 0) * planeNormal[2];

    normalModel[1] =
      modelMatrix->GetElement(0, 1) * planeNormal[0] +
      modelMatrix->GetElement(1, 1) * planeNormal[1] +
      modelMatrix->GetElement(2, 1) * planeNormal[2];

    normalModel[2] =
      modelMatrix->GetElement(0, 2) * planeNormal[0] +
      modelMatrix->GetElement(1, 2) * planeNormal[1] +
      modelMatrix->GetElement(2, 2) * planeNormal[2];

    // Scale into normalized volume space.
    double normalVol[3] = {
      normalModel[0] * bounds.Size[0],
      normalModel[1] * bounds.Size[1],
      normalModel[2] * bounds.Size[2]
    };

    double normalLen = sqrt(
      normalVol[0] * normalVol[0] +
      normalVol[1] * normalVol[1] +
      normalVol[2] * normalVol[2]);

    if (normalLen > 1e-10)
    {
      normalVol[0] /= normalLen;
      normalVol[1] /= normalLen;
      normalVol[2] /= normalLen;
    }

    // Store as float4 (origin.xyz, 1.0) and (normal.xyz, 0.0)
    float* origin = planeData + numPlanes * 8;
    float* normal = origin + 4;
    origin[0] = static_cast<float>(originVol[0]);
    origin[1] = static_cast<float>(originVol[1]);
    origin[2] = static_cast<float>(originVol[2]);
    origin[3] = 1.0f;
    normal[0] = static_cast<float>(normalVol[0]);
    normal[1] = static_cast<float>(normalVol[1]);
    normal[2] = static_cast<float>(normalVol[2]);
    normal[3] = 0.0f;

    numPlanes++;
  }

  if (this->ClippingPlanes->GetNumberOfItems() >= 8)
  {
    vtkWarningMacro("More than 8 clipping planes provided; extras ignored.");
  }

  uniforms->NumClippingPlanes = static_cast<float>(numPlanes);
}
//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::BuildVolumeLightUniforms(
  vtkRenderer* ren, vtkVolume* vol, vtkMatrix4x4* invModelMatrix,
  const double vtkNotUsed(modelBounds)[6], const double vtkNotUsed(boundsSize)[3],
  VolumeLightUniforms& out)
{
  memset(&out, 0, sizeof(out));

  vtkVolumeProperty* property = vol->GetProperty();
  if (!property || !property->GetShade()) {
    out.lightCount = 0;
    out.defaultLighting = 1;
    return;
  }

  out.twoSidedLighting = ren->GetTwoSidedLighting() ? 1 : 0;

  vtkLightCollection* lc = ren->GetLights();
  vtkLight* light = nullptr;
  vtkCollectionSimpleIterator sit;

  int totalLights = 0;
  int positionalLights = 0;
  bool isDefault = true;

  // First pass: count lights and determine if default (single headlight)
  for (lc->InitTraversal(sit); (light = lc->GetNextLight(sit));) {
    if (light->GetSwitch() <= 0.0) continue;
    totalLights++;
    if (light->GetPositional()) positionalLights++;
    if (totalLights > 1 || light->GetIntensity() != 1.0 ||
        light->GetLightType() != VTK_LIGHT_TYPE_HEADLIGHT) {
      isDefault = false;
    }
  }

  out.lightCount = std::min(totalLights, 8);
  out.numPositionalLights = positionalLights;
  out.defaultLighting = isDefault ? 1 : 0;

  if (out.lightCount == 0) return;

  // Second pass: fill light data
  // OpenGL convention: positional lights first, then directional
  int posIdx = 0;
  int dirIdx = positionalLights;

  for (lc->InitTraversal(sit); (light = lc->GetNextLight(sit));) {
    if (light->GetSwitch() <= 0.0) continue;

    int idx = light->GetPositional() ? posIdx++ : dirIdx++;
    if (idx >= 8) break;

    VolumeLightData& L = out.lights[idx];
    double intensity = light->GetIntensity();

    // Colors (pre-multiplied by intensity, matching OpenGL)
    double* aColor = light->GetAmbientColor();
    double* dColor = light->GetDiffuseColor();
    double* sColor = light->GetSpecularColor();
    for (int c = 0; c < 3; ++c) {
      L.ambientColor[c]  = static_cast<float>(aColor[c] * intensity);
      L.diffuseColor[c]  = static_cast<float>(dColor[c] * intensity);
      L.specularColor[c] = static_cast<float>(sColor[c] * intensity);
    }
    L.ambientColor[3] = L.diffuseColor[3] = L.specularColor[3] = 1.0f;

    // Direction: transform to model (data) space.
    // The gradient normal used by the shader is expressed per data unit
    // (computeGradientFast), and the shader's view/frag positions are converted
    // into data space (boundsSize), so the light direction must live in the same
    // space. OpenGL computes in_lightDirection in eye space; the model-to-eye
    // transform is rigid, so a data-space direction is equivalent.
    double* lfp = light->GetTransformedFocalPoint();
    double* lp  = light->GetTransformedPosition();
    double lightDir[3];
    vtkMath::Subtract(lfp, lp, lightDir);
    vtkMath::Normalize(lightDir);

    // Transform direction to model space
    double dirLocal[4] = { lightDir[0], lightDir[1], lightDir[2], 0.0 };
    invModelMatrix->MultiplyPoint(dirLocal, dirLocal);
    double dLen = sqrt(dirLocal[0]*dirLocal[0] + dirLocal[1]*dirLocal[1] + dirLocal[2]*dirLocal[2]);
    if (dLen > 1e-10) {
      dirLocal[0] /= dLen; dirLocal[1] /= dLen; dirLocal[2] /= dLen;
    }
    L.direction[0] = static_cast<float>(dirLocal[0]);
    L.direction[1] = static_cast<float>(dirLocal[1]);
    L.direction[2] = static_cast<float>(dirLocal[2]);
    L.direction[3] = static_cast<float>(light->GetConeAngle());

    if (light->GetPositional()) {
      L.position[3] = 1.0f;  // type = positional

      // Transform position to model (data) space
      double posWorld[4] = { lp[0], lp[1], lp[2], 1.0 };
      double posLocal[4];
      invModelMatrix->MultiplyPoint(posWorld, posLocal);
      if (fabs(posLocal[3]) > 1e-12) {
        posLocal[0] /= posLocal[3];
        posLocal[1] /= posLocal[3];
        posLocal[2] /= posLocal[3];
      }
      L.position[0] = static_cast<float>(posLocal[0]);
      L.position[1] = static_cast<float>(posLocal[1]);
      L.position[2] = static_cast<float>(posLocal[2]);

      // Attenuation (raw values; the shader distance is in data units, which
      // match OpenGL's eye-space distance up to a rigid transform).
      double* attn = light->GetAttenuationValues();
      L.attenuation[0] = static_cast<float>(attn[0]);  // constant
      L.attenuation[1] = static_cast<float>(attn[1]);  // linear
      L.attenuation[2] = static_cast<float>(attn[2]);  // quadratic
      L.attenuation[3] = static_cast<float>(light->GetExponent());  // spot exponent
    } else {
      L.position[3] = 0.0f;  // type = directional
      L.attenuation[0] = 1.0f;  // no attenuation for directional
    }
  }
}

//------------------------------------------------------------------------------
uint32_t vtkMetalGPUVolumeRayCastMapper::VolumeLightingFeatureBits(
  const VolumeLightUniforms& lights)
{
  uint32_t bits = 0;
  if (lights.defaultLighting != 0)
    bits |= VolumeFeature_DefaultLighting;
  bits |= (static_cast<uint32_t>(lights.lightCount) & 0xFu) << VolumeFeature_LightCountShift;
  return bits;
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::SetupBuffers(
  void* mtlDeviceVoid, vtkRenderer* ren, vtkVolume* vol, vtkImageData* input)
{
  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

    if (!this->UniformBuffers[0])
    {
      for (int i = 0; i < 3; ++i)
      {
        id<MTLBuffer> buf = [device newBufferWithLength:VolumeUniformBufferSize
                                                options:MTLResourceStorageModeShared];
        if (!buf)
        {
          vtkErrorMacro("Failed to create uniform buffer");
          return false;
        }
        AssignMetalObject(this->UniformBuffers[i], buf);
      }
      // Zeroed fallback for the fragment buffer(5) rectilinear-coord slot; the
      // shader only reads it when UseRectilinear is set, so a zeroed dummy is
      // safe for all other inputs (matches the SetFragmentTextureOrFallback
      // pattern used for textures).
      id<MTLBuffer> dummyRect =
        [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
      AssignMetalObject(this->DummyRectCoordsBuffer, dummyRect);
    }

    // Use model-space bounds for vertex positions (using extent for correctness)
    if (input)
    {
      VolumeBounds vb = ComputeVolumeBounds(input, this->CellFlag == 1);
      this->ModelBounds[0] = vb.Min[0];
      this->ModelBounds[1] = vb.Max[0];
      this->ModelBounds[2] = vb.Min[1];
      this->ModelBounds[3] = vb.Max[1];
      this->ModelBounds[4] = vb.Min[2];
      this->ModelBounds[5] = vb.Max[2];
    }

    // Check if camera is inside the volume
    bool cameraInside = this->IsCameraInside(ren, vol);

    // Phase 6: Fullscreen camera-inside path skips vertex buffer creation entirely.
    // The fullscreen triangle is generated by vertex_fullscreen_main with no buffers.
    if (cameraInside && this->UseFullscreenCameraInside)
    {
      this->CameraWasInsideInLastUpdate = true;
      return true;
    }

    // Check if geometry needs rebuild
    bool needsVertexRebuild = !this->VertexBuffer;
    needsVertexRebuild |= (this->VolumeUploadTime.GetMTime() > this->VertexBufferUploadTime.GetMTime());
    needsVertexRebuild |= (this->GetMTime() > this->VertexBufferUploadTime.GetMTime());
    needsVertexRebuild |= (cameraInside != this->CameraWasInsideInLastUpdate);

    if (cameraInside)
    {
      vtkCamera* cam = ren->GetActiveCamera();
      if (cam)
      {
        needsVertexRebuild |= (cam->GetMTime() > this->VertexBufferUploadTime.GetMTime());
      }
      needsVertexRebuild |= (ren->GetMTime() > this->VertexBufferUploadTime.GetMTime());
    }

    if (needsVertexRebuild)
    {
      // Camera outside: densified proxy box (OpenGL parity)
      // Camera inside: clip against near plane, densify, triangulate
      if (!cameraInside)
      {
        // Camera outside: densified proxy box (OpenGL parity). GL renders the
        // 8-corner box through vtkDensifyPolyData(2) (centroid fan, 108
        // triangles) even when the camera is outside (see
        // vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::RenderVolumeGeometry).
        // The coarse 12-triangle box rounds the rasterizer's interpolated
        // anchor up to ~28 ulps off GL's. Replicate GL's exact densified
        // geometry (same corner order, triangle set/winding, double centroids
        // cast to float32) so the interpolated in.texcoord matches GL's
        // ip_textureCoords. Vertices are uploaded in dataset (model) space —
        // GL feeds BBoxPolyData positions straight to in_vertexPos — and the
        // vertex shader forwards them via UseDataSpaceBoxVertices.
        vtkNew<vtkPolyData> boxSource;
        {
          vtkNew<vtkCellArray> cells;
          vtkNew<vtkPoints> points;
          points->SetDataTypeToDouble();
          // GL's DataGeometry corner order {000,100,010,110,001,101,011,111}
          // in model space (identical to the camera-inside boxSource corners).
          double corners[24] = {
            this->ModelBounds[0], this->ModelBounds[2], this->ModelBounds[4],
            this->ModelBounds[1], this->ModelBounds[2], this->ModelBounds[4],
            this->ModelBounds[0], this->ModelBounds[3], this->ModelBounds[4],
            this->ModelBounds[1], this->ModelBounds[3], this->ModelBounds[4],
            this->ModelBounds[0], this->ModelBounds[2], this->ModelBounds[5],
            this->ModelBounds[1], this->ModelBounds[2], this->ModelBounds[5],
            this->ModelBounds[0], this->ModelBounds[3], this->ModelBounds[5],
            this->ModelBounds[1], this->ModelBounds[3], this->ModelBounds[5],
          };
          for (int i = 0; i < 8; ++i)
          {
            points->InsertNextPoint(corners + i * 3);
          }
          // 6 faces 12 triangles (GL's tris with GL's 0-2-1 winding swap)
          int tris[36] = {
            0, 1, 2,
            1, 3, 2,
            1, 5, 3,
            5, 7, 3,
            5, 4, 7,
            4, 6, 7,
            4, 0, 6,
            0, 2, 6,
            2, 3, 6,
            3, 7, 6,
            0, 4, 1,
            1, 4, 5
          };
          for (int i = 0; i < 12; ++i)
          {
            cells->InsertNextCell(3);
            cells->InsertCellPoint(tris[i * 3]);
            cells->InsertCellPoint(tris[i * 3 + 2]);
            cells->InsertCellPoint(tris[i * 3 + 1]);
          }
          boxSource->SetPoints(points);
          boxSource->SetPolys(cells);
        }

        vtkNew<vtkDensifyPolyData> densifyPolyData;
        densifyPolyData->SetInputData(boxSource);
        densifyPolyData->SetNumberOfSubdivisions(2);
        densifyPolyData->Update();

        vtkPolyData* finalPolyData = densifyPolyData->GetOutput();
        vtkPoints* points = finalPolyData->GetPoints();
        vtkCellArray* polys = finalPolyData->GetPolys();

        // OpenGL parity: GL uploads the densified double points as float32
        // (in_vertexPos). Mirror the exact float32 values here.
        std::vector<float> vertices;
        vertices.reserve(points->GetNumberOfPoints() * 3);
        for (vtkIdType i = 0; i < points->GetNumberOfPoints(); ++i)
        {
          double pt[3];
          points->GetPoint(i, pt);
          vertices.push_back(static_cast<float>(pt[0]));
          vertices.push_back(static_cast<float>(pt[1]));
          vertices.push_back(static_cast<float>(pt[2]));
        }

        // All densified cells are triangles; emit their 3 vertex ids in order.
        std::vector<unsigned int> indices;
        vtkIdType npts;
        const vtkIdType* pts;
        polys->InitTraversal();
        while (polys->GetNextCell(npts, pts))
        {
          if (npts < 3) continue;
          indices.push_back(static_cast<unsigned int>(pts[0]));
          indices.push_back(static_cast<unsigned int>(pts[1]));
          indices.push_back(static_cast<unsigned int>(pts[2]));
        }

        if (indices.empty())
        {
          vtkErrorMacro("Densified proxy box produced no triangles");
          return false;
        }

        this->IndexCount = static_cast<int>(indices.size());

        // Release old buffers
        if (this->VertexBuffer)
        {
          ReleaseMetalObject(this->VertexBuffer);
        }
        if (this->IndexBuffer)
        {
          ReleaseMetalObject(this->IndexBuffer);
        }

        {
          id<MTLBuffer> vbuf = [device newBufferWithBytes:vertices.data()
                                                  length:vertices.size() * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
          if (!vbuf)
          {
            vtkErrorMacro("Failed to create vertex buffer");
            return false;
          }
          AssignMetalObject(this->VertexBuffer, vbuf);
        }

        if (getenv("VTK_METAL_TEST_DUMP_UNIFORMS") && !getenv("VTK_METAL_TEST_DUMP_VERT_DONE"))
        {
          setenv("VTK_METAL_TEST_DUMP_VERT_DONE", "1", 1);
          FILE* f = fopen("/tmp/app_verts.bin", "wb");
          fwrite(vertices.data(), 1, vertices.size() * sizeof(float), f);
          fclose(f);
          f = fopen("/tmp/app_idxs.bin", "wb");
          fwrite(indices.data(), 1, indices.size() * sizeof(unsigned int), f);
          fclose(f);
          fprintf(stderr, "dumped %zu verts %zu idxs\n", vertices.size(), indices.size());
        }

        {
          id<MTLBuffer> ibuf = [device newBufferWithBytes:indices.data()
                                                  length:indices.size() * sizeof(unsigned int)
                                                 options:MTLResourceStorageModeShared];
          if (!ibuf)
          {
            vtkErrorMacro("Failed to create index buffer");
            return false;
          }
          AssignMetalObject(this->IndexBuffer, ibuf);
        }

        this->CameraWasInsideInLastUpdate = false;
      }
      else
      {
        // Camera inside: clip bounding box against near plane
        vtkNew<vtkPolyData> boxSource;

        {
          vtkNew<vtkCellArray> cells;
          vtkNew<vtkPoints> points;
          points->SetDataTypeToDouble();

          // Corner order matches GL's ijkCorners {000,100,010,110,001,101,011,111}
          // so that the shared tris[36] triangulates the faces with identical diagonals.
          double geometry[24] = {
            this->ModelBounds[0], this->ModelBounds[2], this->ModelBounds[4],
            this->ModelBounds[1], this->ModelBounds[2], this->ModelBounds[4],
            this->ModelBounds[0], this->ModelBounds[3], this->ModelBounds[4],
            this->ModelBounds[1], this->ModelBounds[3], this->ModelBounds[4],
            this->ModelBounds[0], this->ModelBounds[2], this->ModelBounds[5],
            this->ModelBounds[1], this->ModelBounds[2], this->ModelBounds[5],
            this->ModelBounds[0], this->ModelBounds[3], this->ModelBounds[5],
            this->ModelBounds[1], this->ModelBounds[3], this->ModelBounds[5],
          };

          for (int i = 0; i < 8; ++i)
          {
            points->InsertNextPoint(geometry + i * 3);
          }

          // 6 faces 12 triangles (clockwise winding for vtkClipConvexPolyData)
          int tris[36] = {
            0, 1, 2,
            1, 3, 2,
            1, 5, 3,
            5, 7, 3,
            5, 4, 7,
            4, 6, 7,
            4, 0, 6,
            0, 2, 6,
            2, 3, 6,
            3, 7, 6,
            0, 4, 1,
            1, 4, 5
          };

          for (int i = 0; i < 12; ++i)
          {
            cells->InsertNextCell(3);
            // Clockwise convention for vtkClipConvexPolyData
            cells->InsertCellPoint(tris[i * 3]);
            cells->InsertCellPoint(tris[i * 3 + 2]);
            cells->InsertCellPoint(tris[i * 3 + 1]);
          }

          boxSource->SetPoints(points);
          boxSource->SetPolys(cells);
        }

        // Clip bounding box against near plane
        vtkNew<vtkMatrix4x4> dataToWorld;
        vol->GetModelToWorldMatrix(dataToWorld);

        vtkCamera* cam = ren->GetActiveCamera();

        double fplanes[24];
        cam->GetFrustumPlanes(ren->GetTiledAspectRatio(), fplanes);

        // Extract near frustum plane (index 4*4=16)
        double pOrigin[4];
        pOrigin[3] = 1.0;
        double pNormal[3];

        for (int i = 0; i < 3; ++i)
        {
          pNormal[i] = fplanes[16 + i];
          pOrigin[i] = -fplanes[16 + 3] * fplanes[16 + i];
        }

        // Transform origin point to volume coordinates using inverse matrix
        vtkNew<vtkMatrix4x4> worldToData;
        vtkMatrix4x4::Invert(dataToWorld, worldToData);
        worldToData->MultiplyPoint(pOrigin, pOrigin);

        // Transform the near-plane normal to volume coordinates using the
        // TRANSPOSE of the model matrix (not the inverse transpose), exactly
        // like vtkOpenGLGPUVolumeRayCastMapper::RenderVolumeGeometry. For
        // x_world = M x_obj, the plane n_world . x_world = d becomes
        // (M^T n_world) . x_obj = d, so n_obj = M^T n_world. The inverse
        // transpose is the correct transform for the reverse (object-to-world)
        // direction only and diverges under a non-uniform model scale.
        double* dmat = dataToWorld->GetData();
        dataToWorld->Transpose();
        double pNormalV[3];
        pNormalV[0] = pNormal[0] * dmat[0] + pNormal[1] * dmat[1] + pNormal[2] * dmat[2];
        pNormalV[1] = pNormal[0] * dmat[4] + pNormal[1] * dmat[5] + pNormal[2] * dmat[6];
        pNormalV[2] = pNormal[0] * dmat[8] + pNormal[1] * dmat[9] + pNormal[2] * dmat[10];
        vtkMath::Normalize(pNormalV);
        dataToWorld->Transpose();

        // Apply offset to prevent hardware near-plane clipping
        double offset = (cam->GetClippingRange()[1] - cam->GetClippingRange()[0]) * 0.001;
        double minOffset = static_cast<double>(std::numeric_limits<float>::epsilon()) * 1000.0;
        offset = offset < minOffset ? minOffset : offset;

        for (int i = 0; i < 3; ++i)
        {
          pOrigin[i] += (pNormalV[i] * offset);
        }

        vtkNew<vtkPlane> nearPlane;
        nearPlane->SetOrigin(pOrigin);
        nearPlane->SetNormal(pNormalV);

        vtkNew<vtkPlaneCollection> planes;
        planes->RemoveAllItems();
        planes->AddItem(nearPlane);

        vtkNew<vtkClipConvexPolyData> clip;
        clip->SetInputData(boxSource);
        clip->SetPlanes(planes);

        // Clip, then densify — no vtkTriangleFilter. The OpenGL backend draws
        // the first 3 vertices of every densified cell directly (see
        // vtkOpenGLGPUVolumeRayCastMapper::vtkInternal::RenderVolumeGeometry),
        // and we must replicate that exactly: the extra triangulation of the
        // clipped cap/side polygons produces interior-spanning triangles (and
        // misses the near-plane cap at some pixels), which breaks the
        // fragment anchors vs GL.
        vtkNew<vtkDensifyPolyData> densifyPolyData;
        densifyPolyData->SetInputConnection(clip->GetOutputPort());
        densifyPolyData->SetNumberOfSubdivisions(2);
        densifyPolyData->Update();

        vtkPolyData* finalPolyData = densifyPolyData->GetOutput();
        vtkPoints* points = finalPolyData->GetPoints();
        vtkCellArray* polys = finalPolyData->GetPolys();

        // OpenGL parity: upload the clipped/densified vertices in dataset
        // (model) space directly — GL feeds this polydata's positions to the
        // vertex shader as in_vertexPos without normalization. The vertex
        // shader detects the camera-inside path via useCameraInsideNearClip and
        // forwards in.position unchanged, so the interpolated fragment anchor is
        // a dataset-space position just like GL's ip_vertexPos.
        std::vector<float> vertices;
        vertices.reserve(points->GetNumberOfPoints() * 3);

        for (vtkIdType i = 0; i < points->GetNumberOfPoints(); ++i)
        {
          double pt[3];
          points->GetPoint(i, pt);
          vertices.push_back(static_cast<float>(pt[0]));
          vertices.push_back(static_cast<float>(pt[1]));
          vertices.push_back(static_cast<float>(pt[2]));
        }

        // Build the index list exactly like the OpenGL backend: the first 3
        // vertices of every cell (densified triangles plus the clipped cap/side
        // polygons that pass through densify un-triangulated).
        std::vector<unsigned int> indices;
        vtkIdType npts;
        const vtkIdType* pts;

        polys->InitTraversal();
        while (polys->GetNextCell(npts, pts))
        {
          if (npts < 3) continue;
          indices.push_back(static_cast<unsigned int>(pts[0]));
          indices.push_back(static_cast<unsigned int>(pts[1]));
          indices.push_back(static_cast<unsigned int>(pts[2]));
        }

        if (indices.empty())
        {
          vtkErrorMacro("Camera-inside clipped proxy produced no triangles");
          return false;
        }

        this->IndexCount = static_cast<int>(indices.size());

        // Release old buffers
        if (this->VertexBuffer)
        {
          ReleaseMetalObject(this->VertexBuffer);
        }
        if (this->IndexBuffer)
        {
          ReleaseMetalObject(this->IndexBuffer);
        }

        // Create new vertex buffer
        {
          id<MTLBuffer> vbuf = [device newBufferWithBytes:vertices.data()
                                                  length:vertices.size() * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
          if (!vbuf)
          {
            vtkErrorMacro("Failed to create vertex buffer");
            return false;
          }
          AssignMetalObject(this->VertexBuffer, vbuf);
        }

        // Create new index buffer
        {
          id<MTLBuffer> ibuf = [device newBufferWithBytes:indices.data()
                                                  length:indices.size() * sizeof(unsigned int)
                                                 options:MTLResourceStorageModeShared];
          if (!ibuf)
          {
            vtkErrorMacro("Failed to create index buffer");
            return false;
          }
          AssignMetalObject(this->IndexBuffer, ibuf);
        }

        this->CameraWasInsideInLastUpdate = true;
      }

      this->VertexBufferUploadTime.Modified();
    }
  }

  return this->VertexBuffer && this->IndexBuffer && this->UniformBuffers[0];
}

//------------------------------------------------------------------------------
bool vtkMetalGPUVolumeRayCastMapper::SetupPipeline(void* mtlDeviceVoid, vtkRenderer* ren)
{
  // Get sample count and invalidate PSO if it changed (e.g., MSAA toggled)
  auto* metalRenderWindow = vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());
  int sampleCount = metalRenderWindow ? metalRenderWindow->GetEffectiveSampleCount() : 1;

  if (this->PipelineState && sampleCount != this->CurrentSampleCount)
  {
    ReleaseMetalObject(this->PipelineState);
  }

  if (this->PipelineState)
  {
    return true;
  }

  if (!this->EnsureShaderLibrary(mtlDeviceVoid))
  {
    return false;
  }

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

    // Create dummy depth texture (1x1 R32Float with value 1.0) for when no real depth texture is bound
    if (!this->DummyDepthTexture)
    {
      id<MTLTexture> dummyTex = NewTexture2D(
        device,
        MTLPixelFormatR32Float,
        1, 1,
        MTLTextureUsageShaderRead,
        MTLStorageModeShared);
      if (dummyTex)
      {
        float one = 1.0f;
        MTLRegion region = MTLRegionMake2D(0, 0, 1, 1);
        [dummyTex replaceRegion:region mipmapLevel:0 withBytes:&one bytesPerRow:sizeof(float)];
        AssignMetalObject(this->DummyDepthTexture, dummyTex);
      }
    }

    // Note: constexpr samplers in MetalShaders.metal replace all runtime
    // sampler state objects. No sampler creation needed here.

    // Create dummy 3D textures for fallback bindings (prevent nil texture binds).
    if (!this->DummyVolumeTexture)
    {
      float zeroFloat = 0.0f;
      AssignMetalObject(this->DummyVolumeTexture,
        CreateDummy3DTexture(device, MTLPixelFormatR32Float, &zeroFloat, sizeof(float)));
    }

    if (!this->DummyMaskTexture)
    {
      float zeroFloat = 0.0f;
      AssignMetalObject(this->DummyMaskTexture,
        CreateDummy3DTexture(device, MTLPixelFormatR32Float, &zeroFloat, sizeof(float)));
    }

    if (!this->DummyMinMaxTexture)
    {
      uint8_t zeroByte = 0;
      AssignMetalObject(this->DummyMinMaxTexture,
        CreateDummy3DTexture(device, MTLPixelFormatR8Unorm, &zeroByte, sizeof(uint8_t)));
    }

    // Create and cache a depth stencil state.
    if (!this->DepthStencilState)
    {
      MTLDepthStencilDescriptor* dsDesc = [[MTLDepthStencilDescriptor alloc] init];
      dsDesc.depthCompareFunction = MTLCompareFunctionLessEqual;
      dsDesc.depthWriteEnabled = NO;
      id<MTLDepthStencilState> ds = [device newDepthStencilStateWithDescriptor:dsDesc];
      [dsDesc release];
      AssignMetalObject(this->DepthStencilState, ds);
    }

    // Create volume rendering pipelines via the caching helper.
    // Base pipelines (featureMask=0) are created here for backward compat;
    // specialized pipelines with non-zero feature masks are created on demand
    // in GPURender() via GetOrCreateVolumePipeline.
    // GetOrCreateVolumePipeline returns a non-owning pointer (cache owns +1);
    // use AssignRetainedMetalObject to give each member its own +1.
    void* pso = this->GetOrCreateVolumePipeline(mtlDeviceVoid,
      static_cast<uint32_t>(VolumePipelineType::DirectScreen),
      MTLPixelFormatBGRA8Unorm, MTLPixelFormatDepth32Float,
      static_cast<uint32_t>(sampleCount), 0);
    if (!pso)
    {
      return false;
    }
    AssignRetainedMetalObject(this->PipelineState, (__bridge id)pso);
    this->CurrentSampleCount = sampleCount;

    // Pre-create fullscreen camera-inside pipelines (cached in PipelineCache).
    // FullscreenDirect: BGRA + depth + blending, matching DirectScreen.
    void* fsDirectPso = this->GetOrCreateVolumePipeline(mtlDeviceVoid,
      static_cast<uint32_t>(VolumePipelineType::FullscreenDirect),
      MTLPixelFormatBGRA8Unorm, MTLPixelFormatDepth32Float,
      static_cast<uint32_t>(sampleCount), 0);
    if (!fsDirectPso)
    {
      return false;
    }
    // FullscreenOffscreen: RGBA16Float, no depth, no blending.
    void* fsOffscreenPso = this->GetOrCreateVolumePipeline(mtlDeviceVoid,
      static_cast<uint32_t>(VolumePipelineType::FullscreenOffscreen),
      MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1, 0);
    if (!fsOffscreenPso)
    {
      return false;
    }
  }

  return true;
}

//------------------------------------------------------------------------------
void* vtkMetalGPUVolumeRayCastMapper::GetOrCreateVolumePipeline(
  void* mtlDeviceVoid, uint32_t type, uint32_t colorFormat,
  uint32_t depthFormat, uint32_t sampleCount, uint32_t featureMask)
{
  if (getenv("VTK_METAL_TEST_MARCH_DEBUG"))
    fprintf(stderr, "[FRAG-ENTER] type=%u feat=0x%x extraPending=%d\n", type, featureMask, VolumeFragBatch());
  VolumePipelineKey key = { type, colorFormat, depthFormat, sampleCount, featureMask,
    // featureMaskExtra: low bits carry the volume orientation code
    // (VolumeTextureAxisDepth 0/1/2), bit 2 the block-summary walk gate —
    // fc_mmBlocks pipelines must not be shared with non-block ones — and
    // bit 3 the super-summary gate (fc_mmSuper).
    static_cast<uint32_t>(this->VolumeTextureAxisDepth) |
      ((VolumeMinMaxBlocksWanted(this->UseGPUMinMax, this->SampleDistance) &&
        this->MinMaxBlockTexture != nullptr) ? 4u : 0u) |
      ((VolumeMinMaxSuperWanted(this->UseGPUMinMax, this->SampleDistance) &&
        this->MinMaxSuperTexture != nullptr) ? 8u : 0u) |
      // §35.14 segment consume gate — bit 16, RTT-family pipelines only (the
      // fullscreen/selection binders do not bind the segment buffer slots).
      (((type == static_cast<uint32_t>(VolumePipelineType::RenderToImage) ||
         type == static_cast<uint32_t>(VolumePipelineType::RayAtlas) ||
         type == static_cast<uint32_t>(VolumePipelineType::OffscreenLayer)) &&
        VolumeSegWanted() && !VolumeSegConsumeSuppressed() &&
        this->SegBuildComputePipeline != nullptr &&
        this->SegAtlasATexture != nullptr) ? 16u : 0u) |
      // SD-aware batch cap — bit 32 (fc_fineSD): fine SD <1.5 → shade cap 2 vs 4
      ((this->SampleDistance < 1.5f) ? 32u : 0u) |
      // Grad variants — bit 128 (fc_grad4) and 512 (fc_gradNearest)
      ((std::getenv("VTK_METAL_TEST_GRAD4") != nullptr) ? 128u : 0u) |
      ((std::getenv("VTK_METAL_TEST_GRAD_NEAREST") != nullptr) ? 512u : 0u) |
      // §38 TF-adaptive exit threshold — bit 64 (fc_exitTheta): pipelines
      // with a uniform-supplied saturation exit must not share with the
      // legacy-latch ones.
      ((VolumeExitTheta() > 0.0f) ? 64u : 0u) |
      // §17 SD4 fixed overhead specializations: depth/cameraInside/dense bypass + volume nearest coarse + quadGrad
      ((std::getenv("VTK_METAL_TEST_DEPTH") != nullptr) ? (1u<<16) : 0u) |
      ((std::getenv("VTK_METAL_TEST_CAMERA_INSIDE") != nullptr) ? (1u<<17) : 0u) |
      ((std::getenv("VTK_METAL_TEST_DENSE") != nullptr) ? (1u<<18) : 0u) |
      ((std::getenv("VTK_METAL_TEST_VOLUME_NEAREST") != nullptr) ? (1u<<19) : 0u) |
      ((std::getenv("VTK_METAL_TEST_QUAD_GRAD") != nullptr) ? (1u<<20) : 0u) |
      // Fragment compile-time batch specialization — encode width in
      // featureMaskExtra bits [10:15] so each compile-time width gets its
      // own PSO (occupancy probe, mirrors fc_cmBatch trick for compute).
      (static_cast<uint32_t>(VolumeFragBatch()) << 10) |
      // §38.15/38.16 block-summary tap bisects — bit 256 (fc_mmNoTap),
      // bit 1024 (fc_mmRead), bit 2048 (fc_mmAltTap).
      0u };
  auto it = this->PipelineCache.find(key);
  if (it != this->PipelineCache.end())
  {
    return it->second;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary;
  if (!library)
  {
    return nullptr;
  }

  NSError* error = nil;

  // Determine the fragment function name and whether to use vertex_volume_main
  // or vertex_fullscreen_main based on pipeline type.
  NSString* fragName = @"fragment_volume_main";
  bool useVolumeVertex = true;

  switch (static_cast<VolumePipelineType>(type))
  {
    case VolumePipelineType::DirectScreen:
    case VolumePipelineType::OffscreenLayer:
      fragName = @"fragment_volume_main";
      useVolumeVertex = true;
      break;
    case VolumePipelineType::FullscreenDirect:
    case VolumePipelineType::FullscreenOffscreen:
      fragName = @"fragment_volume_fullscreen_main";
      useVolumeVertex = false;
      break;
    case VolumePipelineType::ImageSampleBlit:
      fragName = @"fragment_image_sample_blit";
      useVolumeVertex = false;
      break;
    case VolumePipelineType::GridTraversalDirect:
    case VolumePipelineType::GridTraversalOffscreen:
      fragName = @"fragment_volume_grid_traversal_main";
      useVolumeVertex = false;
      break;
    case VolumePipelineType::RenderToImage:
      fragName = @"fragment_volume_rtt_main";
      useVolumeVertex = true;
      break;
    case VolumePipelineType::SelectionDirect:
      fragName = @"fragment_volume_selection_main";
      useVolumeVertex = true;
      break;
    case VolumePipelineType::SelectionFullscreen:
      fragName = @"fragment_volume_fullscreen_selection_main";
      useVolumeVertex = false;
      break;
    case VolumePipelineType::RayAtlas:
      // §35.14 ray-atlas pre-pass: same proxy geometry + vertex function as
      // the main volume pass; the fragment writes the march setup instead of
      // marching.
      fragName = @"fragment_volume_ray_atlas";
      useVolumeVertex = true;
      break;
  }

  // Create function constants for shader specialization.
  // Only volume fragment shaders (volume_main/volume_accum_main) use
  // function constants; composite and blit passes do not.
  id<MTLFunction> fragFunc = nil;
  VolumePipelineType pt = static_cast<VolumePipelineType>(type);
  BOOL hasFeatureConstants = (pt == VolumePipelineType::DirectScreen ||
    pt == VolumePipelineType::OffscreenLayer ||
    pt == VolumePipelineType::FullscreenDirect ||
    pt == VolumePipelineType::FullscreenOffscreen ||
    pt == VolumePipelineType::GridTraversalDirect ||
    pt == VolumePipelineType::GridTraversalOffscreen ||
    pt == VolumePipelineType::RenderToImage ||
    pt == VolumePipelineType::SelectionDirect ||
    pt == VolumePipelineType::SelectionFullscreen ||
    pt == VolumePipelineType::RayAtlas);

  if (hasFeatureConstants)
  {
    MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];

    BOOL shading = (featureMask & VolumeFeature_Shading) ? YES : NO;
    BOOL gradOp  = (featureMask & VolumeFeature_GradientOpacity) ? YES : NO;
    BOOL mask    = (featureMask & VolumeFeature_Mask) ? YES : NO;
    BOOL minmax  = (featureMask & VolumeFeature_MinMax) ? YES : NO;
    BOOL normalTex = (featureMask & VolumeFeature_NormalTexture) ? YES : NO;
    BOOL linearInterp = (featureMask & VolumeFeature_LinearInterpolation) ? YES : NO;
    BOOL computeNormalFromOpacity =
      (featureMask & VolumeFeature_ComputeNormalFromOpacity) ? YES : NO;
    BOOL independentComp =
      (featureMask & VolumeFeature_IndependentComponents) ? YES : NO;

    [constants setConstantValue:&shading type:MTLDataTypeBool withName:@"fc_shading"];
    [constants setConstantValue:&gradOp  type:MTLDataTypeBool withName:@"fc_gradientOpacity"];
    [constants setConstantValue:&mask    type:MTLDataTypeBool withName:@"fc_mask"];
    [constants setConstantValue:&minmax  type:MTLDataTypeBool withName:@"fc_minmax"];
    [constants setConstantValue:&normalTex type:MTLDataTypeBool withName:@"fc_normalTexture"];
    [constants setConstantValue:&linearInterp type:MTLDataTypeBool withName:@"fc_linearInterpolation"];
    [constants setConstantValue:&computeNormalFromOpacity type:MTLDataTypeBool
                        withName:@"fc_computeNormalFromOpacity"];
    [constants setConstantValue:&independentComp type:MTLDataTypeBool
                        withName:@"fc_independentComponents"];
    BOOL transfer2D = (featureMask & VolumeFeature_Transfer2D) ? YES : NO;
    BOOL rectilinear = (featureMask & VolumeFeature_Rectilinear) ? YES : NO;
    BOOL defaultLighting = (featureMask & VolumeFeature_DefaultLighting) ? YES : NO;
    int lightCount = (featureMask >> VolumeFeature_LightCountShift) & 0xFu;
    BOOL dependentRGBA = (featureMask & VolumeFeature_DependentRGBA) ? YES : NO;
    BOOL dependentLA = (featureMask & VolumeFeature_DependentLA) ? YES : NO;
    BOOL renderToTexture = (featureMask & VolumeFeature_RenderToImage) ? YES : NO;

    [constants setConstantValue:&transfer2D type:MTLDataTypeBool withName:@"fc_transfer2D"];
    [constants setConstantValue:&rectilinear type:MTLDataTypeBool withName:@"fc_rectilinear"];
    [constants setConstantValue:&defaultLighting type:MTLDataTypeBool
                        withName:@"fc_defaultLighting"];
    [constants setConstantValue:&lightCount type:MTLDataTypeInt withName:@"fc_lightCount"];
    [constants setConstantValue:&dependentRGBA type:MTLDataTypeBool withName:@"fc_dependentRGBA"];
    [constants setConstantValue:&dependentLA type:MTLDataTypeBool withName:@"fc_dependentLA"];
    [constants setConstantValue:&renderToTexture type:MTLDataTypeBool
                        withName:@"fc_renderToTexture"];
    BOOL cropping = (featureMask & VolumeFeature_Cropping) ? YES : NO;
    BOOL blanking = (featureMask & VolumeFeature_Blanking) ? YES : NO;
    [constants setConstantValue:&cropping type:MTLDataTypeBool withName:@"fc_cropping"];
    [constants setConstantValue:&blanking type:MTLDataTypeBool withName:@"fc_blanking"];

    // March-experiment selector (fc_marchVariant): decoded from the feature
    // mask bits 24-27 so each variant gets its own specialized pipeline.
    const int marchVariant =
      (featureMask & VolumeFeature_MarchVariantMask) >> VolumeFeature_MarchVariantShift;
    [constants setConstantValue:&marchVariant type:MTLDataTypeInt
                       withName:@"fc_marchVariant"];

    // Composite slab tiling (fc_slabMode): decoded from the VolumeFeature_Slab
    // bit. When clear (numSlabs=1, the default) the shader compiles the slab
    // index partition out entirely, so non-slab pipelines are bit-identical.
    int slabMode = (featureMask & VolumeFeature_Slab) ? 1 : 0;
    [constants setConstantValue:&slabMode type:MTLDataTypeInt withName:@"fc_slabMode"];

    // V31 back-edge exit experiment (fc_doExit): decoded from the dedicated
    // feature bit so the do-while reshaped march gets its own pipeline.
    BOOL marchDoExit = (featureMask & VolumeFeature_MarchDoExit) ? YES : NO;
    [constants setConstantValue:&marchDoExit type:MTLDataTypeBool
                       withName:@"fc_doExit"];

    // RG8 pair-packed slices experiment (fc_volRg8): pair-tap sampler.
    BOOL volRg8 = (featureMask & VolumeFeature_VolRg8) ? YES : NO;
    [constants setConstantValue:&volRg8 type:MTLDataTypeBool
                       withName:@"fc_volRg8"];

    // Transposed volume representation experiment (fc_volTransposed): the
    // texture stores x<->z transposed; scalar fetches map coords via .zyx.
    BOOL volTransposed = (featureMask & VolumeFeature_VolTransposed) ? YES : NO;
    if (getenv("VTK_METAL_TEST_MARCH_DEBUG"))
      fprintf(stderr, "[march] mask=0x%08x axis=%d\n", featureMask,
        this->VolumeTextureAxisDepth);
    [constants setConstantValue:&volTransposed type:MTLDataTypeBool
                       withName:@"fc_volTransposed"];
    // §29 orientation companion: X-depth (.zyx) vs Y-depth (.xzy) fetch maps.
    BOOL volTransposedY = (this->VolumeTextureAxisDepth == 2) ? YES : NO;
    [constants setConstantValue:&volTransposedY type:MTLDataTypeBool
                       withName:@"fc_volTransposedY"];
    // Two-level occupancy summary (VTK_METAL_TEST_MM_BLOCKS): compile-time
    // gate for the block-summary walk. Requires the summary texture to exist;
    // the shader additionally requires real (>1³) dims so dummy-bound
    // pipelines stay inert.
    BOOL mmBlocks = (VolumeMinMaxBlocksWanted(this->UseGPUMinMax,
                         this->SampleDistance) &&
                     this->MinMaxBlockTexture != nullptr) ? YES : NO;
    [constants setConstantValue:&mmBlocks type:MTLDataTypeBool
                       withName:@"fc_mmBlocks"];
    BOOL mmSuper = (VolumeMinMaxSuperWanted(this->UseGPUMinMax,
                        this->SampleDistance) &&
                    this->MinMaxSuperTexture != nullptr) ? YES : NO;
    [constants setConstantValue:&mmSuper type:MTLDataTypeBool
                       withName:@"fc_mmSuper"];
    // §35.14 async segment pre-pass consume (fc_segHop, featureMaskExtra bit
    // 16): compile-time switch between the legacy preamble walk and the
    // streaming gap consume fed by volume_segment_build.
    BOOL segHop = ((VolumeSegWanted() && !VolumeSegConsumeSuppressed() &&
                    this->SegBuildComputePipeline != nullptr &&
                    this->SegAtlasATexture != nullptr)) ? YES : NO;
    [constants setConstantValue:&segHop type:MTLDataTypeBool
                       withName:@"fc_segHop"];
    // §38 TF-adaptive exit threshold (fc_exitTheta, key bit 64): compile-time
    // switch between the legacy 8-bit latch exit and the uniform-supplied
    // accumulated-opacity threshold.
    BOOL exitTheta = (VolumeExitTheta() > 0.0f) ? YES : NO;
    [constants setConstantValue:&exitTheta type:MTLDataTypeBool
                       withName:@"fc_exitTheta"];
    // Fragment compile-time batch specialization (VTK_METAL_TEST_FRAG_BATCH):
    int fragBatchFc = VolumeFragBatch();
    [constants setConstantValue:&fragBatchFc type:MTLDataTypeInt
                       withName:@"fc_fragBatch"];
    BOOL grad4 = (std::getenv("VTK_METAL_TEST_GRAD4") != nullptr) ? YES : NO;
    [constants setConstantValue:&grad4 type:MTLDataTypeBool withName:@"fc_grad4"];
    BOOL gradFloat = (std::getenv("VTK_METAL_TEST_GRAD_FLOAT") != nullptr) ? YES : NO;
    [constants setConstantValue:&gradFloat type:MTLDataTypeBool withName:@"fc_gradFloat"];
    // SD-aware batch cap / grad: fine SD <0.75 world units → shade cap 2 vs 4 and 4-fetch grad
    // Threshold 0.75 separates fine (0.5) from coarse (4) and excludes default VolumeRayCast (1.0) to keep thr 0.18
    BOOL fineSD = (this->SampleDistance < 0.75f) ? YES : NO;
    [constants setConstantValue:&fineSD type:MTLDataTypeBool withName:@"fc_fineSD"];
    BOOL gradNearest = (std::getenv("VTK_METAL_TEST_GRAD_NEAREST") != nullptr) ? YES : NO;
    [constants setConstantValue:&gradNearest type:MTLDataTypeBool withName:@"fc_gradNearest"];
    // §17 SD4 fixed overhead specializations: depth/cameraInside dead-strip, dense coarse bypass, volume nearest coarse
    BOOL useDepthTexture = (std::getenv("VTK_METAL_TEST_DEPTH") != nullptr) ? YES : NO;
    [constants setConstantValue:&useDepthTexture type:MTLDataTypeBool withName:@"fc_useDepthTexture"];
    BOOL useCameraInside = (std::getenv("VTK_METAL_TEST_CAMERA_INSIDE") != nullptr) ? YES : NO;
    [constants setConstantValue:&useCameraInside type:MTLDataTypeBool withName:@"fc_useCameraInside"];
    BOOL dense = (std::getenv("VTK_METAL_TEST_DENSE") != nullptr) ? YES : NO;
    [constants setConstantValue:&dense type:MTLDataTypeBool withName:@"fc_dense"];
    BOOL volumeNearestCoarse = (std::getenv("VTK_METAL_TEST_VOLUME_NEAREST") != nullptr) ? YES : NO;
    [constants setConstantValue:&volumeNearestCoarse type:MTLDataTypeBool withName:@"fc_volumeNearestCoarse"];
    BOOL quadGrad = (std::getenv("VTK_METAL_TEST_QUAD_GRAD") != nullptr) ? YES : NO;
    [constants setConstantValue:&quadGrad type:MTLDataTypeBool withName:@"fc_quadGrad"];

    // §38.15/38.16 block-summary tap bisects (fc_mmNoTap / fc_mmRead).


    // Blend mode function constant: 0=composite, 1=MIP, 2=MinIP, 3=AverageIP,
    // 4=additive (vtkVolumeMapper::BlendMode). Encoded in the feature mask so
    // each blend mode gets its own specialized pipeline.
    int blendMode = 0;
    if (featureMask & VolumeFeature_BlendMaximumIntensity)
      blendMode = static_cast<int>(vtkVolumeMapper::MAXIMUM_INTENSITY_BLEND);
    else if (featureMask & VolumeFeature_BlendMinimumIntensity)
      blendMode = static_cast<int>(vtkVolumeMapper::MINIMUM_INTENSITY_BLEND);
    else if (featureMask & VolumeFeature_BlendAverageIntensity)
      blendMode = static_cast<int>(vtkVolumeMapper::AVERAGE_INTENSITY_BLEND);
    else if (featureMask & VolumeFeature_BlendAdditive)
      blendMode = static_cast<int>(vtkVolumeMapper::ADDITIVE_BLEND);
    [constants setConstantValue:&blendMode type:MTLDataTypeInt withName:@"fc_blendMode"];

    if (getenv("VTK_METAL_TEST_MARCH_DEBUG"))
      fprintf(stderr,
        "[march] variant=%d blendMode=%d shading=%d gradOp=%d minmax=%d "
        "mask=%d blanking=%d crop=%d rect=%d tf2d=%d indep=%d depRGBA=%d depLA=%d "
        "rt=%d slab=%d (unrolled-activates=%d)\n",
        marchVariant, blendMode, shading, gradOp, minmax, mask, blanking, cropping,
        rectilinear, transfer2D, independentComp, dependentRGBA, dependentLA,
        renderToTexture, slabMode,
        (marchVariant >= 6 && blendMode == 0 && !cropping && !mask && !blanking &&
         !rectilinear && !transfer2D && !independentComp && !dependentRGBA && !dependentLA));
    fprintf(stderr, "[march] fc_doExit=%d fc_volRg8=%d fc_volTransposed=%d\n",
      marchDoExit ? 1 : 0, volRg8 ? 1 : 0, volTransposed ? 1 : 0);

    fragFunc = [library newFunctionWithName:fragName
                             constantValues:constants
                                      error:&error];
    [constants release];
  }
  else
  {
    // Non-volume pipelines (composite, blit) have no function constants.
    fragFunc = [library newFunctionWithName:fragName];
  }

  if (!fragFunc)
  {
    vtkErrorMacro(<< "Failed to find fragment function " << [fragName UTF8String]);
    return nullptr;
  }

  id<MTLFunction> vertexFunc = nil;
  if (useVolumeVertex)
  {
    vertexFunc = [library newFunctionWithName:@"vertex_volume_main"];
  }
  else
  {
    vertexFunc = [library newFunctionWithName:@"vertex_fullscreen_main"];
  }

  if (!vertexFunc)
  {
    vtkErrorMacro("Failed to find vertex function for pipeline");
    [fragFunc release];
    return nullptr;
  }

  MTLRenderPipelineDescriptor* pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
  pipelineDesc.vertexFunction = vertexFunc;
  pipelineDesc.fragmentFunction = fragFunc;

  if (useVolumeVertex)
  {
    MTLVertexDescriptor* vertexDesc = [[MTLVertexDescriptor alloc] init];
    vertexDesc.attributes[0].format = MTLVertexFormatFloat3;
    vertexDesc.attributes[0].offset = 0;
    vertexDesc.attributes[0].bufferIndex = 0;
    vertexDesc.layouts[0].stride = sizeof(float) * 3;
    vertexDesc.layouts[0].stepRate = 1;
    vertexDesc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    pipelineDesc.vertexDescriptor = vertexDesc;
    [vertexDesc release];
  }

  pipelineDesc.colorAttachments[0].pixelFormat = (MTLPixelFormat)colorFormat;

  // RenderToImage exports color to attachment 0 and the depth image to
  // attachment 1 (R32Float, the default DepthImageScalarType).
  if (pt == VolumePipelineType::RenderToImage)
  {
    pipelineDesc.colorAttachments[1].pixelFormat = MTLPixelFormatR32Float;
  }

  // §35.14 ray-atlas pipeline: three RGBA32Float setup planes.
  if (pt == VolumePipelineType::RayAtlas)
  {
    pipelineDesc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Float;
    pipelineDesc.colorAttachments[2].pixelFormat = MTLPixelFormatRGBA32Float;
    pipelineDesc.colorAttachments[1].blendingEnabled = NO;
    pipelineDesc.colorAttachments[2].blendingEnabled = NO;
  }

  // The selection pipelines additionally write the picking IDs
  // ({voxelId, propId, compositeIndex}) to an RGBA32Uint attachment 1 — the
  // same format used by the surface mappers' picking pipelines and read back
  // by vtkMetalHardwareSelector.
  if (pt == VolumePipelineType::SelectionDirect ||
    pt == VolumePipelineType::SelectionFullscreen)
  {
    pipelineDesc.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Uint;
    pipelineDesc.colorAttachments[1].blendingEnabled = NO;
  }

  // DirectScreen and FullscreenDirect use blending; offscreen pipelines do not.
  if (pt == VolumePipelineType::DirectScreen || pt == VolumePipelineType::FullscreenDirect ||
      pt == VolumePipelineType::GridTraversalDirect ||
      pt == VolumePipelineType::SelectionDirect ||
      pt == VolumePipelineType::SelectionFullscreen)
  {
    pipelineDesc.colorAttachments[0].blendingEnabled = YES;
    pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
    pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
  }
  else if (pt == VolumePipelineType::RenderToImage)
  {
    // The OpenGL RenderToImage pass is UNBLENDED: the raycast shader writes its
    // raw (premultiplied) color over the cleared white RTT background, so no
    // ONE/ONE_MINUS_SRC_ALPHA compositing happens on attachment 0 (see
    // vtkOpenGLGPUVolumeRayCastMapper.cxx vtkglClearColor(1.0,1.0,1.0,0.0)).
    // Blending here would inject (1-alpha)*255 into every RTT pixel and produce
    // the contour-concentrated 47,878-px GL-vs-Metal residual on
    // TestGPURayCastRenderToTexture (VolumeRayCastBackendComparisonFindingsUpdate84.md).
    pipelineDesc.colorAttachments[0].blendingEnabled = NO;
    pipelineDesc.colorAttachments[1].blendingEnabled = NO;
  }
  else
  {
    pipelineDesc.colorAttachments[0].blendingEnabled = NO;
  }

  if (depthFormat != MTLPixelFormatInvalid)
  {
    pipelineDesc.depthAttachmentPixelFormat = (MTLPixelFormat)depthFormat;
  }

  pipelineDesc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
  pipelineDesc.rasterSampleCount = sampleCount;

  id<MTLRenderPipelineState> pso =
    [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
  [pipelineDesc release];
  [vertexFunc release];
  [fragFunc release];

  if (!pso)
  {
    vtkErrorMacro(<< "Pipeline creation failed for type " << type << ": "
                  << [[error localizedDescription] UTF8String]);
    return nullptr;
  }
  // TEMP-DIAG fragment batch occupancy stats (register pressure probe).
  if (getenv("VTK_METAL_TEST_MARCH_DEBUG"))
  {
    int fragBatchFcDbg = VolumeFragBatch();
    fprintf(stderr, "[fragpso] type=%u mask=0x%x extra=0x%x fragBatch=%d\n",
      type, featureMask, key.featureMaskExtra, fragBatchFcDbg);
  }

  // Cache and return.
  // The +1 from new() is owned by the cache.
  // The caller receives a non-owning handle; use AssignRetainedMetalObject
  // when storing into a member slot.
  this->PipelineCache[key] = (__bridge void*)pso;
  return (__bridge void*)pso;
}

//------------------------------------------------------------------------------
// §38.6 / §36.4 Design B — Get or create specialized compute marcher pipeline
void* vtkMetalGPUVolumeRayCastMapper::GetOrCreateComputeMarchPipeline(
  void* mtlDeviceVoid, uint32_t featureMask, bool binned)
{
  VolumePipelineKey key;
  key.type = binned ? 100u : 101u;
  key.colorFormat = static_cast<uint32_t>(MTLPixelFormatRGBA16Float);
  key.depthFormat = static_cast<uint32_t>(MTLPixelFormatInvalid);
  key.sampleCount = 1;
  key.featureMask = featureMask;
  key.featureMaskExtra = this->VolumeTextureAxisDepth;
  // fc_cmBatch specialization (register-pressure diet): bake the env value
  // into the cache key so each width compiles its own PSO. The stride-split
  // flag rides in bit 16 (TEMP-DIAG key encoding).
  int cmBatchFcKey = 0;
  bool cmSplitKey = false;
  if (const char* v = getenv("VTK_METAL_TEST_CM_BATCH"))
    cmBatchFcKey = std::max(0, std::min(48, std::atoi(v)));
  cmSplitKey = getenv("VTK_METAL_TEST_CM_SPLIT") != nullptr;

  key.sampleCount = static_cast<uint32_t>(cmBatchFcKey) + 1 |
                    (cmSplitKey ? (1u << 16) : 0u) |

                    // §38.17 segment consume for the compute marcher.
                    ((getenv("VTK_METAL_TEST_MM_SEG") != nullptr &&
                      getenv("VTK_METAL_TEST_MM_SEG_NOCONSUME") == nullptr)
                      ? (1u << 26) : 0u);

  auto& cache = binned ? this->ComputeMarchBinnedPipelineCache : this->ComputeMarchPipelineCache;
  auto it = cache.find(key);
  if (it != cache.end())
  {
    return it->second;
  }

  id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)mtlDeviceVoid;
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary;
  if (!library)
  {
    vtkErrorMacro("Compute march pipeline: shader library missing");
    return nullptr;
  }

  MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];

  BOOL shading = (featureMask & VolumeFeature_Shading) ? YES : NO;
  BOOL gradOp  = (featureMask & VolumeFeature_GradientOpacity) ? YES : NO;
  BOOL mask    = (featureMask & VolumeFeature_Mask) ? YES : NO;
  BOOL minmax  = (featureMask & VolumeFeature_MinMax) ? YES : NO;
  BOOL normalTex = (featureMask & VolumeFeature_NormalTexture) ? YES : NO;
  BOOL linearInterp = (featureMask & VolumeFeature_LinearInterpolation) ? YES : NO;
  BOOL computeNormalFromOpacity =
    (featureMask & VolumeFeature_ComputeNormalFromOpacity) ? YES : NO;
  BOOL independentComp =
    (featureMask & VolumeFeature_IndependentComponents) ? YES : NO;

  [constants setConstantValue:&shading type:MTLDataTypeBool withName:@"fc_shading"];
  [constants setConstantValue:&gradOp  type:MTLDataTypeBool withName:@"fc_gradientOpacity"];
  [constants setConstantValue:&mask    type:MTLDataTypeBool withName:@"fc_mask"];
  [constants setConstantValue:&minmax  type:MTLDataTypeBool withName:@"fc_minmax"];
  [constants setConstantValue:&normalTex type:MTLDataTypeBool withName:@"fc_normalTexture"];
  [constants setConstantValue:&linearInterp type:MTLDataTypeBool withName:@"fc_linearInterpolation"];
  [constants setConstantValue:&computeNormalFromOpacity type:MTLDataTypeBool
                      withName:@"fc_computeNormalFromOpacity"];
  [constants setConstantValue:&independentComp type:MTLDataTypeBool
                      withName:@"fc_independentComponents"];
  BOOL transfer2D = (featureMask & VolumeFeature_Transfer2D) ? YES : NO;
  BOOL rectilinear = (featureMask & VolumeFeature_Rectilinear) ? YES : NO;
  BOOL defaultLighting = (featureMask & VolumeFeature_DefaultLighting) ? YES : NO;
  int lightCount = (featureMask >> VolumeFeature_LightCountShift) & 0xFu;
  BOOL dependentRGBA = (featureMask & VolumeFeature_DependentRGBA) ? YES : NO;
  BOOL dependentLA = (featureMask & VolumeFeature_DependentLA) ? YES : NO;
  BOOL renderToTexture = (featureMask & VolumeFeature_RenderToImage) ? YES : NO;

  [constants setConstantValue:&transfer2D type:MTLDataTypeBool withName:@"fc_transfer2D"];
  [constants setConstantValue:&rectilinear type:MTLDataTypeBool withName:@"fc_rectilinear"];
  [constants setConstantValue:&defaultLighting type:MTLDataTypeBool
                      withName:@"fc_defaultLighting"];
  [constants setConstantValue:&lightCount type:MTLDataTypeInt withName:@"fc_lightCount"];
  [constants setConstantValue:&dependentRGBA type:MTLDataTypeBool withName:@"fc_dependentRGBA"];
  [constants setConstantValue:&dependentLA type:MTLDataTypeBool withName:@"fc_dependentLA"];
  [constants setConstantValue:&renderToTexture type:MTLDataTypeBool
                      withName:@"fc_renderToTexture"];
  BOOL cropping = (featureMask & VolumeFeature_Cropping) ? YES : NO;
  BOOL blanking = (featureMask & VolumeFeature_Blanking) ? YES : NO;
  [constants setConstantValue:&cropping type:MTLDataTypeBool withName:@"fc_cropping"];
  [constants setConstantValue:&blanking type:MTLDataTypeBool withName:@"fc_blanking"];

  const int marchVariant =
    (featureMask & VolumeFeature_MarchVariantMask) >> VolumeFeature_MarchVariantShift;
  [constants setConstantValue:&marchVariant type:MTLDataTypeInt
                     withName:@"fc_marchVariant"];

  // §38.10: compile-time compute-march ladder width (0 = runtime-driven).
  int cmBatchFc = 0;
  if (const char* v = getenv("VTK_METAL_TEST_CM_BATCH"))
    cmBatchFc = std::max(0, std::min(48, std::atoi(v)));
  [constants setConstantValue:&cmBatchFc type:MTLDataTypeInt
                     withName:@"fc_cmBatch"];

  // §38.12: stride-parity split of the main 32-rung body.
  BOOL cmSplitFc = getenv("VTK_METAL_TEST_CM_SPLIT") != nullptr ? YES : NO;
  [constants setConstantValue:&cmSplitFc type:MTLDataTypeBool
                     withName:@"fc_cmSplit"];

  // §38.15/38.16 block-summary tap bisects (fc_mmNoTap / fc_mmRead).

  // §38.17 segment consume for the compute marcher.
  BOOL cmSegHopFc =
    (getenv("VTK_METAL_TEST_MM_SEG") != nullptr &&
     getenv("VTK_METAL_TEST_MM_SEG_NOCONSUME") == nullptr) ? YES : NO;
  [constants setConstantValue:&cmSegHopFc type:MTLDataTypeBool
                     withName:@"fc_cmSegHop"];

  int slabMode = (featureMask & VolumeFeature_Slab) ? 1 : 0;
  [constants setConstantValue:&slabMode type:MTLDataTypeInt withName:@"fc_slabMode"];

  BOOL marchDoExit = (featureMask & VolumeFeature_MarchDoExit) ? YES : NO;
  [constants setConstantValue:&marchDoExit type:MTLDataTypeBool
                     withName:@"fc_doExit"];

  BOOL volRg8 = (featureMask & VolumeFeature_VolRg8) ? YES : NO;
  [constants setConstantValue:&volRg8 type:MTLDataTypeBool
                     withName:@"fc_volRg8"];

  BOOL volTransposed = (featureMask & VolumeFeature_VolTransposed) ? YES : NO;
  [constants setConstantValue:&volTransposed type:MTLDataTypeBool
                     withName:@"fc_volTransposed"];
  BOOL volTransposedY = (this->VolumeTextureAxisDepth == 2) ? YES : NO;
  [constants setConstantValue:&volTransposedY type:MTLDataTypeBool
                     withName:@"fc_volTransposedY"];

  BOOL mmBlocks = (VolumeMinMaxBlocksWanted(this->UseGPUMinMax,
                       this->SampleDistance) &&
                   this->MinMaxBlockTexture != nullptr) ? YES : NO;
  [constants setConstantValue:&mmBlocks type:MTLDataTypeBool
                     withName:@"fc_mmBlocks"];
  BOOL mmSuper = (VolumeMinMaxSuperWanted(this->UseGPUMinMax,
                      this->SampleDistance) &&
                  this->MinMaxSuperTexture != nullptr) ? YES : NO;
  [constants setConstantValue:&mmSuper type:MTLDataTypeBool
                     withName:@"fc_mmSuper"];

  BOOL segHop = NO;
  [constants setConstantValue:&segHop type:MTLDataTypeBool
                     withName:@"fc_segHop"];

  BOOL exitTheta = (VolumeExitTheta() > 0.0f) ? YES : NO;
  [constants setConstantValue:&exitTheta type:MTLDataTypeBool
                     withName:@"fc_exitTheta"];

  int blendMode = 0;
  if (featureMask & VolumeFeature_BlendMaximumIntensity)
    blendMode = static_cast<int>(vtkVolumeMapper::MAXIMUM_INTENSITY_BLEND);
  else if (featureMask & VolumeFeature_BlendMinimumIntensity)
    blendMode = static_cast<int>(vtkVolumeMapper::MINIMUM_INTENSITY_BLEND);
  else if (featureMask & VolumeFeature_BlendAverageIntensity)
    blendMode = static_cast<int>(vtkVolumeMapper::AVERAGE_INTENSITY_BLEND);
  else if (featureMask & VolumeFeature_BlendAdditive)
    blendMode = static_cast<int>(vtkVolumeMapper::ADDITIVE_BLEND);
  [constants setConstantValue:&blendMode type:MTLDataTypeInt withName:@"fc_blendMode"];

  NSError* err = nil;
  NSString* funcName = binned ? @"volume_compute_march_binned" : @"volume_compute_march";
  id<MTLFunction> fn = [library newFunctionWithName:funcName constantValues:constants error:&err];
  [constants release];
  if (!fn)
  {
    vtkErrorMacro("Failed to specialize " << [funcName UTF8String] << ": "
                  << [[err localizedDescription] UTF8String]);
    return nullptr;
  }

  id<MTLComputePipelineState> cps = [mtlDevice newComputePipelineStateWithFunction:fn error:&err];
  [fn release];
  if (!cps)
  {
    vtkErrorMacro("Failed to create compute pipeline for " << [funcName UTF8String] << ": "
                  << [[err localizedDescription] UTF8String]);
    return nullptr;
  }

  // TEMP-DIAG §38.10: occupancy stats (register pressure shows up as a low
  // maxTotalThreadsPerThreadgroup vs the ~1024 a trivial kernel gets).
  fprintf(stderr, "[cmpso] %s binned=%d: maxThreadsPerTG=%lu execWidth=%lu\n",
    [funcName UTF8String], (int)binned,
    (unsigned long)cps.maxTotalThreadsPerThreadgroup,
    (unsigned long)cps.threadExecutionWidth);

  void* res = (__bridge void*)cps;
  cache[key] = res;
  return res;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::BindComputeMarchTextures(
  void* encoderVoid, void* atlasAVoid, void* atlasBVoid, void* atlasCVoid, void* outColorVoid)
{
  id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)encoderVoid;
  id<MTLTexture> atlasA = (__bridge id<MTLTexture>)atlasAVoid;
  id<MTLTexture> atlasB = (__bridge id<MTLTexture>)atlasBVoid;
  id<MTLTexture> atlasC = (__bridge id<MTLTexture>)atlasCVoid;
  id<MTLTexture> outColor = (__bridge id<MTLTexture>)outColorVoid;

  [enc setTexture:atlasA atIndex:0];
  [enc setTexture:atlasB atIndex:1];
  [enc setTexture:atlasC atIndex:3];
  [enc setTexture:outColor atIndex:4];
  SetComputeTextureOrFallback(enc, 5, this->VolumeTexture, this->DummyVolumeTexture);
  [enc setTexture:(__bridge id<MTLTexture>)this->ColorOpacityTexture atIndex:6];
  SetComputeTextureOrFallback(enc, 7, this->ComponentTransferFunctionTexture1, this->ColorOpacityTexture);
  SetComputeTextureOrFallback(enc, 8, this->ComponentTransferFunctionTexture2, this->ColorOpacityTexture);
  SetComputeTextureOrFallback(enc, 9, this->ComponentTransferFunctionTexture3, this->ColorOpacityTexture);
  SetComputeTextureOrFallback(enc, 10, this->Transfer2DTexture, this->ColorOpacityTexture);
  SetComputeTextureOrFallback(enc, 11, this->Transfer2DYAxisTexture, this->DummyVolumeTexture);
  SetComputeTextureOrFallback(enc, 12, this->GradientOpacityTexture, this->ColorOpacityTexture);
  SetComputeTextureOrFallback(enc, 13, this->MaskTexture, this->DummyMaskTexture);
  SetComputeTextureOrFallback(enc, 14, this->LabelMapTransferTexture, this->ColorOpacityTexture);
  SetComputeTextureOrFallback(enc, 15, this->MinMaxTexture, this->DummyMinMaxTexture);
  SetComputeTextureOrFallback(enc, 16, this->MinMaxBlockTexture, this->DummyMinMaxTexture);
  // §38.16 alt-slot bisect (VTK_METAL_TEST_MM_ALTSLOT): re-point the slot-17
  // binding at the block summary so the walk's tap can read the same bytes
  // through a different binding index (supers must be off — enforced by the
  // knob check in GetOrCreateComputeMarchPipeline callers via fc gating).
  SetComputeTextureOrFallback(enc, 17,
    getenv("VTK_METAL_TEST_MM_ALTSLOT") ? this->MinMaxBlockTexture
                                        : this->MinMaxSuperTexture,
    this->DummyMinMaxTexture);
  SetComputeTextureOrFallback(enc, 18, this->GradientNormalTexture, this->DummyVolumeTexture);
  SetComputeTextureOrFallback(enc, 19, this->BlankingTexture, this->DummyVolumeTexture);
}

//------------------------------------------------------------------------------
// Cinematic — shaded DVR helpers (single 8x8, wax AO/SSS, no Woodcock)
bool vtkMetalGPUVolumeRayCastMapper::EnsureCinematicResources(void* deviceVoid, int width, int height)
{
  if (this->CinematicAccumTextureA && this->CinematicAccumTextureB &&
      this->CinematicAccumWidth == width && this->CinematicAccumHeight == height)
  {
    return true;
  }
  ReleaseMetalObject(this->CinematicAccumTextureA);
  ReleaseMetalObject(this->CinematicAccumTextureB);
  ReleaseMetalObject(this->CinematicDenoiseTexture);
  @autoreleasepool {
    id<MTLDevice> device = (__bridge id<MTLDevice>)deviceVoid;
    // Ping-pong accumulation — RGBA16Float, coherent 8x8 tiles, shared ColorOpacity fetch
    id<MTLTexture> texA = NewTexture2D(device, MTLPixelFormatRGBA16Float, width, height,
      MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget, MTLStorageModePrivate);
    id<MTLTexture> texB = NewTexture2D(device, MTLPixelFormatRGBA16Float, width, height,
      MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget, MTLStorageModePrivate);
    id<MTLTexture> denoiseTex = NewTexture2D(device, MTLPixelFormatRGBA16Float, width, height,
      MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, MTLStorageModePrivate);
    if (!texA || !texB) return false;
    AssignMetalObject(this->CinematicAccumTextureA, texA);
    AssignMetalObject(this->CinematicAccumTextureB, texB);
    if (denoiseTex) AssignMetalObject(this->CinematicDenoiseTexture, denoiseTex);
    this->CinematicAccumWidth = width;
    this->CinematicAccumHeight = height;
    this->CinematicAccumCount = 0;
    this->CinematicAccumValid = false;
  }
  return true;
}

void vtkMetalGPUVolumeRayCastMapper::ReleaseCinematicResources()
{
  ReleaseMetalObject(this->CinematicAccumTextureA);
  ReleaseMetalObject(this->CinematicAccumTextureB);
  ReleaseMetalObject(this->CinematicDenoiseTexture);
  ReleaseMetalObject(this->CinematicDenoisePipeline);
  ReleaseMetalObject(this->CinematicComputePipeline);
  for (auto &e : this->CinematicComputePipelineCache) [(__bridge id)e.second release];
  this->CinematicComputePipelineCache.clear();
  this->CinematicAccumWidth = 0;
  this->CinematicAccumHeight = 0;
  this->CinematicAccumCount = 0;
  this->CinematicAccumValid = false;
}

void* vtkMetalGPUVolumeRayCastMapper::GetOrCreateCinematicComputePipeline(void* mtlDeviceVoid, uint32_t featureMask)
{
  VolumePipelineKey key;
  key.type = 103u;
  key.colorFormat = static_cast<uint32_t>(MTLPixelFormatRGBA16Float);
  key.depthFormat = static_cast<uint32_t>(MTLPixelFormatInvalid);
  key.sampleCount = 1;
  key.featureMask = featureMask;
  key.featureMaskExtra = this->VolumeTextureAxisDepth |
    ((this->CinematicDenoise > 0.0f) ? (1u << 23) : 0u);
  auto &cache = this->CinematicComputePipelineCache;
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)mtlDeviceVoid;
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary;
  if (!library) return nullptr;
  MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
  BOOL shading = (featureMask & VolumeFeature_Shading) ? YES : NO;
  BOOL gradOp = (featureMask & VolumeFeature_GradientOpacity) ? YES : NO;
  BOOL mask = (featureMask & VolumeFeature_Mask) ? YES : NO;
  BOOL minmax = (featureMask & VolumeFeature_MinMax) ? YES : NO;
  BOOL normalTex = (featureMask & VolumeFeature_NormalTexture) ? YES : NO;
   BOOL linearInterp = YES;
   BOOL computeNormalFromOpacity = (featureMask & VolumeFeature_ComputeNormalFromOpacity) ? YES : NO;
   BOOL independentComp = (featureMask & VolumeFeature_IndependentComponents) ? YES : NO;
   [constants setConstantValue:&shading type:MTLDataTypeBool withName:@"fc_shading"];
   [constants setConstantValue:&gradOp type:MTLDataTypeBool withName:@"fc_gradientOpacity"];
   [constants setConstantValue:&mask type:MTLDataTypeBool withName:@"fc_mask"];
   [constants setConstantValue:&minmax type:MTLDataTypeBool withName:@"fc_minmax"];
   [constants setConstantValue:&normalTex type:MTLDataTypeBool withName:@"fc_normalTexture"];
   [constants setConstantValue:&linearInterp type:MTLDataTypeBool withName:@"fc_linearInterpolation"];
   [constants setConstantValue:&computeNormalFromOpacity type:MTLDataTypeBool withName:@"fc_computeNormalFromOpacity"];
   [constants setConstantValue:&independentComp type:MTLDataTypeBool withName:@"fc_independentComponents"];
   BOOL transfer2D = (featureMask & VolumeFeature_Transfer2D) ? YES : NO;
   BOOL rectilinear = (featureMask & VolumeFeature_Rectilinear) ? YES : NO;
   BOOL defaultLighting = (featureMask & VolumeFeature_DefaultLighting) ? YES : NO;
   int lightCount = (featureMask >> VolumeFeature_LightCountShift) & 0xFu;
   BOOL dependentRGBA = (featureMask & VolumeFeature_DependentRGBA) ? YES : NO;
   BOOL dependentLA = (featureMask & VolumeFeature_DependentLA) ? YES : NO;
   BOOL renderToTexture = (featureMask & VolumeFeature_RenderToImage) ? YES : NO;
   [constants setConstantValue:&transfer2D type:MTLDataTypeBool withName:@"fc_transfer2D"];
   [constants setConstantValue:&rectilinear type:MTLDataTypeBool withName:@"fc_rectilinear"];
   [constants setConstantValue:&defaultLighting type:MTLDataTypeBool withName:@"fc_defaultLighting"];
   [constants setConstantValue:&lightCount type:MTLDataTypeInt withName:@"fc_lightCount"];
   [constants setConstantValue:&dependentRGBA type:MTLDataTypeBool withName:@"fc_dependentRGBA"];
   [constants setConstantValue:&dependentLA type:MTLDataTypeBool withName:@"fc_dependentLA"];
   [constants setConstantValue:&renderToTexture type:MTLDataTypeBool withName:@"fc_renderToTexture"];
   BOOL cropping = (featureMask & VolumeFeature_Cropping) ? YES : NO;
   BOOL blanking = (featureMask & VolumeFeature_Blanking) ? YES : NO;
   [constants setConstantValue:&cropping type:MTLDataTypeBool withName:@"fc_cropping"];
   [constants setConstantValue:&blanking type:MTLDataTypeBool withName:@"fc_blanking"];
   const int marchVariant = (featureMask & VolumeFeature_MarchVariantMask) >> VolumeFeature_MarchVariantShift;
   [constants setConstantValue:&marchVariant type:MTLDataTypeInt withName:@"fc_marchVariant"];
   int slabMode = (featureMask & VolumeFeature_Slab) ? 1 : 0;
   [constants setConstantValue:&slabMode type:MTLDataTypeInt withName:@"fc_slabMode"];
   BOOL marchDoExit = (featureMask & VolumeFeature_MarchDoExit) ? YES : NO;
   [constants setConstantValue:&marchDoExit type:MTLDataTypeBool withName:@"fc_doExit"];
   BOOL volRg8 = (featureMask & VolumeFeature_VolRg8) ? YES : NO;
   [constants setConstantValue:&volRg8 type:MTLDataTypeBool withName:@"fc_volRg8"];
   BOOL volTransposed = (featureMask & VolumeFeature_VolTransposed) ? YES : NO;
   [constants setConstantValue:&volTransposed type:MTLDataTypeBool withName:@"fc_volTransposed"];
   BOOL volTransposedY = (this->VolumeTextureAxisDepth == 2) ? YES : NO;
   [constants setConstantValue:&volTransposedY type:MTLDataTypeBool withName:@"fc_volTransposedY"];
   BOOL mmBlocks = (VolumeMinMaxBlocksWanted(this->UseGPUMinMax, this->SampleDistance) && this->MinMaxBlockTexture != nullptr) ? YES : NO;
   [constants setConstantValue:&mmBlocks type:MTLDataTypeBool withName:@"fc_mmBlocks"];
   BOOL mmSuper = (VolumeMinMaxSuperWanted(this->UseGPUMinMax, this->SampleDistance) && this->MinMaxSuperTexture != nullptr) ? YES : NO;
   [constants setConstantValue:&mmSuper type:MTLDataTypeBool withName:@"fc_mmSuper"];
   BOOL segHop = NO;
   [constants setConstantValue:&segHop type:MTLDataTypeBool withName:@"fc_segHop"];
   BOOL exitTheta = (VolumeExitTheta() > 0.0f) ? YES : NO;
   [constants setConstantValue:&exitTheta type:MTLDataTypeBool withName:@"fc_exitTheta"];
    // Cinematic — no fc_cinematic/fc_denoise (reads u.cinematicEnabled)
  // Unused FV constants keep default NO
  BOOL useDepthTexture = NO, useCameraInside = NO, dense = NO, volNearestCoarse = NO, quadGrad = NO, grad4 = NO, gradNearest = NO, fineSD = NO, gradFloat = NO;
  int fragBatchFc = 0, cmBatchFc = 0; BOOL cmSplitFc = NO, cmSegHopFc = NO;
  [constants setConstantValue:&grad4 type:MTLDataTypeBool withName:@"fc_grad4"];
  [constants setConstantValue:&gradFloat type:MTLDataTypeBool withName:@"fc_gradFloat"];
  [constants setConstantValue:&fineSD type:MTLDataTypeBool withName:@"fc_fineSD"];
  [constants setConstantValue:&gradNearest type:MTLDataTypeBool withName:@"fc_gradNearest"];
  [constants setConstantValue:&useDepthTexture type:MTLDataTypeBool withName:@"fc_useDepthTexture"];
  [constants setConstantValue:&useCameraInside type:MTLDataTypeBool withName:@"fc_useCameraInside"];
  [constants setConstantValue:&dense type:MTLDataTypeBool withName:@"fc_dense"];
  [constants setConstantValue:&volNearestCoarse type:MTLDataTypeBool withName:@"fc_volumeNearestCoarse"];
  [constants setConstantValue:&quadGrad type:MTLDataTypeBool withName:@"fc_quadGrad"];
  [constants setConstantValue:&fragBatchFc type:MTLDataTypeInt withName:@"fc_fragBatch"];
  [constants setConstantValue:&cmBatchFc type:MTLDataTypeInt withName:@"fc_cmBatch"];
  [constants setConstantValue:&cmSplitFc type:MTLDataTypeBool withName:@"fc_cmSplit"];
  [constants setConstantValue:&cmSegHopFc type:MTLDataTypeBool withName:@"fc_cmSegHop"];
  int blendMode = 0;
  if (featureMask & VolumeFeature_BlendMaximumIntensity) blendMode = static_cast<int>(vtkVolumeMapper::MAXIMUM_INTENSITY_BLEND);
  else if (featureMask & VolumeFeature_BlendMinimumIntensity) blendMode = static_cast<int>(vtkVolumeMapper::MINIMUM_INTENSITY_BLEND);
  else if (featureMask & VolumeFeature_BlendAverageIntensity) blendMode = static_cast<int>(vtkVolumeMapper::AVERAGE_INTENSITY_BLEND);
  else if (featureMask & VolumeFeature_BlendAdditive) blendMode = static_cast<int>(vtkVolumeMapper::ADDITIVE_BLEND);
  [constants setConstantValue:&blendMode type:MTLDataTypeInt withName:@"fc_blendMode"];
  NSError* err = nil;
  NSString* funcName = @"volume_compute_march_cinematic";
  id<MTLFunction> fn = [library newFunctionWithName:funcName constantValues:constants error:&err];
  [constants release];
  if (!fn) { vtkErrorMacro("Failed to specialize " << [funcName UTF8String] << ": " << [[err localizedDescription] UTF8String]); return nullptr; }
  id<MTLComputePipelineState> cps = [mtlDevice newComputePipelineStateWithFunction:fn error:&err];
  [fn release];
  if (!cps) { vtkErrorMacro("Failed to create cinematic pipeline " << [funcName UTF8String] << ": " << [[err localizedDescription] UTF8String]); return nullptr; }
#if __has_feature(objc_arc)
  void* res = (__bridge_retained void*)cps;
#else
  void* res = (__bridge void*)cps;
#endif
  cache[key] = res;
  return res;
}

bool vtkMetalGPUVolumeRayCastMapper::DispatchCinematicCompute(void* deviceVoid, void* queueVoid, void* cmdBufVoid, vtkRenderer* ren, vtkVolume* vol, void* uniformBufVoid, const void* pbdVoid, const void* lightUniformsVoid, int width, int height)
{
  @autoreleasepool {
    id<MTLDevice> device = (__bridge id<MTLDevice>)deviceVoid;
    id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)cmdBufVoid;
    if (!this->EnsureCinematicResources(deviceVoid, width, height)) return false;
    // Build feature mask for cinematic (reuse same mask as fragment but cinematic bit implied)
    VolumeMapperUniforms* uniforms = static_cast<VolumeMapperUniforms*>(uniformBufVoid);
    uint32_t featureMask = 0;
    if (uniforms->UseGradientShading > 0.5f) featureMask |= VolumeFeature_Shading;
    if (uniforms->UseGradientOpacity > 0.5f) featureMask |= VolumeFeature_GradientOpacity;
    if (uniforms->UseMask > 0.5f) featureMask |= VolumeFeature_Mask;
    if (uniforms->UseMinMaxAccel > 0.5f) featureMask |= VolumeFeature_MinMax;
    if (uniforms->UseNormalTexture > 0.5f) featureMask |= VolumeFeature_NormalTexture;
    if (uniforms->UseLinearVolumeInterpolation > 0.5f) featureMask |= VolumeFeature_LinearInterpolation;
    const VolumeLightUniforms* __lightsTmp = static_cast<const VolumeLightUniforms*>(lightUniformsVoid);
    if (__lightsTmp) {
      if (__lightsTmp->defaultLighting) featureMask |= VolumeFeature_DefaultLighting;
      featureMask |= (static_cast<uint32_t>(__lightsTmp->lightCount) & 0xFu) << VolumeFeature_LightCountShift;
      if (__lightsTmp->lightCount==0) featureMask |= VolumeFeature_DefaultLighting; // headlight fallback
    }
    if (uniforms->UseCropping > 0.5f) featureMask |= VolumeFeature_Cropping;
    if (uniforms->UseBlanking > 0.5f) featureMask |= VolumeFeature_Blanking;
    featureMask |= BlendModeToFeatureFlag(this->GetBlendMode());
    // Cinematic is direct 8x8 dispatch (synthetic rays, no atlas/binned).
    // Binned Woodcock path deleted — at 1 spp it is speckle; shaded DVR is the product.
    bool useB = (this->CinematicAccumCount % 2) == 1;
    id<MTLTexture> accumCurr = (__bridge id<MTLTexture>)(useB ? this->CinematicAccumTextureB : this->CinematicAccumTextureA);
    id<MTLTexture> accumPrev = (__bridge id<MTLTexture>)(useB ? this->CinematicAccumTextureA : this->CinematicAccumTextureB);
    void* pso = this->GetOrCreateCinematicComputePipeline(deviceVoid, featureMask);
    if (!pso) return false;
    PerBlockData pbd = *static_cast<const PerBlockData*>(pbdVoid);
    id<MTLBuffer> uniformBuf = (__bridge id<MTLBuffer>)uniformBufVoid;
    id<MTLBuffer> rectBuf = this->RectCoordsBuffer ? (__bridge id<MTLBuffer>)this->RectCoordsBuffer : (__bridge id<MTLBuffer>)this->DummyRectCoordsBuffer;
    const VolumeLightUniforms* lights = static_cast<const VolumeLightUniforms*>(lightUniformsVoid);
    {
      id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
      enc.label = @"VTK Cinematic";
      [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso];
      [enc setTexture:accumCurr atIndex:4];
      SetComputeTextureOrFallback(enc, 5, this->VolumeTexture, this->DummyVolumeTexture);
      [enc setTexture:(__bridge id<MTLTexture>)this->ColorOpacityTexture atIndex:6];
      SetComputeTextureOrFallback(enc, 7, this->ComponentTransferFunctionTexture1, this->ColorOpacityTexture);
      SetComputeTextureOrFallback(enc, 8, this->ComponentTransferFunctionTexture2, this->ColorOpacityTexture);
      SetComputeTextureOrFallback(enc, 9, this->ComponentTransferFunctionTexture3, this->ColorOpacityTexture);
      SetComputeTextureOrFallback(enc, 10, this->Transfer2DTexture, this->ColorOpacityTexture);
      SetComputeTextureOrFallback(enc, 11, this->Transfer2DYAxisTexture, this->DummyVolumeTexture);
      SetComputeTextureOrFallback(enc, 12, this->GradientOpacityTexture, this->ColorOpacityTexture);
      SetComputeTextureOrFallback(enc, 13, this->MaskTexture, this->DummyMaskTexture);
      SetComputeTextureOrFallback(enc, 14, this->LabelMapTransferTexture, this->ColorOpacityTexture);
      SetComputeTextureOrFallback(enc, 15, this->MinMaxTexture, this->DummyMinMaxTexture);
      SetComputeTextureOrFallback(enc, 16, this->MinMaxBlockTexture, this->DummyMinMaxTexture);
      SetComputeTextureOrFallback(enc, 17, this->MinMaxSuperTexture, this->DummyMinMaxTexture);
      SetComputeTextureOrFallback(enc, 18, this->GradientNormalTexture, this->DummyVolumeTexture);
      SetComputeTextureOrFallback(enc, 19, this->BlankingTexture, this->DummyVolumeTexture);
      [enc setTexture:(__bridge id<MTLTexture>)(this->DepthTextureOcclusion?this->DepthTextureOcclusion:this->DummyDepthTexture) atIndex:20];
      [enc setTexture:accumPrev atIndex:21];
      [enc setBuffer:uniformBuf offset:0 atIndex:1];
      [enc setBytes:&pbd length:sizeof(pbd) atIndex:2];
      [enc setBuffer:rectBuf offset:0 atIndex:3];
      [enc setBytes:lights length:sizeof(VolumeLightUniforms) atIndex:4];
      MTLSize tg=MTLSizeMake(8,8,1);
      MTLSize groups=MTLSizeMake((width+7)/8,(height+7)/8,1);
      [enc dispatchThreadgroups:groups threadsPerThreadgroup:tg];
      [enc endEncoding];
    }
    // Optional denoise — cached PSO (was recreated every frame; hitch)
    if (this->CinematicDenoise > 0.01f && this->CinematicDenoiseTexture) {
      id<MTLTexture> denoiseTex = (__bridge id<MTLTexture>)this->CinematicDenoiseTexture;
      if (!this->CinematicDenoisePipeline) {
        id<MTLLibrary> lib = (__bridge id<MTLLibrary>)this->CachedShaderLibrary;
        id<MTLFunction> denoiseFn = [lib newFunctionWithName:@"volume_cinematic_denoise"];
        if (denoiseFn) {
          NSError* e=nil;
          id<MTLComputePipelineState> dps=[device newComputePipelineStateWithFunction:denoiseFn error:&e];
          [denoiseFn release];
          if (dps) AssignMetalObject(this->CinematicDenoisePipeline, dps);
          if (e) vtkErrorMacro("Denoise PSO: " << [[e localizedDescription] UTF8String]);
        }
      }
      id<MTLComputePipelineState> dps = (__bridge id<MTLComputePipelineState>)this->CinematicDenoisePipeline;
      if (dps) {
        id<MTLComputeCommandEncoder> denEnc=[commandBuffer computeCommandEncoder];
        denEnc.label=@"VTK Cinematic Denoise";
        [denEnc setComputePipelineState:dps];
        [denEnc setTexture:accumCurr atIndex:0];
        [denEnc setTexture:denoiseTex atIndex:1];
        float w = this->CinematicDenoise;
        // 8-tap AO is stable at 1 spp; bilateral at 0.35 smears gyri into clay — 0 for single-frame stills
        if (this->CinematicAccumCount < 4) w = 0.0f;
        else w = std::min(w, 0.22f);
        [denEnc setBytes:&w length:sizeof(w) atIndex:0];
        MTLSize tg=MTLSizeMake(8,8,1);
        MTLSize groups=MTLSizeMake((width+7)/8,(height+7)/8,1);
        [denEnc dispatchThreadgroups:groups threadsPerThreadgroup:tg];
        [denEnc endEncoding];
        accumCurr = denoiseTex;
      }
    }
    // Expose to Phase 3b blit
    AssignRetainedMetalObject(this->ImageSampleColorTexture, accumCurr);
    this->ImageSampleFBOWidth = width;
    this->ImageSampleFBOHeight = height;
    this->CinematicAccumValid = true;
    return true;
  }
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::BindEncoderResources(
  void* encoderVoid, void* uniformBufVoid, void* pipelineStateVoid, bool hasDepth)
{
  id<MTLRenderCommandEncoder> encoder =
    (__bridge id<MTLRenderCommandEncoder>)encoderVoid;
  id<MTLBuffer> uniformBuf = (__bridge id<MTLBuffer>)uniformBufVoid;

  // Set pipeline and render state (use provided pipeline, or fall back to default)
  id<MTLRenderPipelineState> pipeline;
  if (pipelineStateVoid)
  {
    pipeline = (__bridge id<MTLRenderPipelineState>)pipelineStateVoid;
  }
  else
  {
    pipeline = (__bridge id<MTLRenderPipelineState>)this->PipelineState;
  }
  [encoder setRenderPipelineState:pipeline];
  // Back-face culling parity with GL (vtkVolumeStateRAII: GL_CULL_FACE +
  // glCullFace(GL_BACK), default front face GL_CCW). Now that the boxSource
  // corner order matches GL's ijkCorners, the clip/densify mesh is
  // byte-identical to GL's, so the front-facing winding must be
  // counter-clockwise like GL's — the previous MTLWindingClockwise rendered
  // the byte-identical mesh fully culled (dark image).
  [encoder setFrontFacingWinding:MTLWindingCounterClockwise];
  [encoder setCullMode:MTLCullModeBack];

  // Only bind depth state if the pipeline uses depth testing.
  if (this->DepthStencilState && hasDepth)
  {
    id<MTLDepthStencilState> ds =
      (__bridge id<MTLDepthStencilState>)this->DepthStencilState;
    [encoder setDepthStencilState:ds];
  }

  // Bind buffers
  id<MTLBuffer> vertexBuf = (__bridge id<MTLBuffer>)this->VertexBuffer;
  [encoder setVertexBuffer:vertexBuf offset:0 atIndex:0];
  [encoder setVertexBuffer:uniformBuf offset:0 atIndex:1];
  [encoder setFragmentBuffer:uniformBuf offset:0 atIndex:1];

  // Rectilinear coord curves (float3 per index). Zeroed dummy for non-rectilinear.
  id<MTLBuffer> rectCoordsBuf =
    this->RectCoordsBuffer ? (__bridge id<MTLBuffer>)this->RectCoordsBuffer
                           : (__bridge id<MTLBuffer>)this->DummyRectCoordsBuffer;
  [encoder setFragmentBuffer:rectCoordsBuf offset:0 atIndex:5];

  // §35.14 segment pre-pass buffers ([[buffer(6)]] index map pre-offset is
  // applied in-shader; [[buffer(7)]] gap pool). Slots are always bound so the
  // declared fragment parameters stay valid even when the feature is off.
  if (!this->SegDummyBuffer)
  {
    AssignRetainedMetalObject(this->SegDummyBuffer,
      [[encoder device] newBufferWithLength:16
                                    options:MTLResourceStorageModeShared]);
  }
  id<MTLBuffer> segIdxBuf = (this->SegActiveThisFrame && this->SegIndexBuffer)
    ? (__bridge id<MTLBuffer>)this->SegIndexBuffer
    : (__bridge id<MTLBuffer>)this->SegDummyBuffer;
  id<MTLBuffer> segPoolBuf = (this->SegActiveThisFrame && this->SegPoolBuffer)
    ? (__bridge id<MTLBuffer>)this->SegPoolBuffer
    : (__bridge id<MTLBuffer>)this->SegDummyBuffer;
  [encoder setFragmentBuffer:segIdxBuf offset:0 atIndex:6];
  [encoder setFragmentBuffer:segPoolBuf offset:0 atIndex:7];

  SetFragmentTextureOrFallback(encoder, 0, this->VolumeTexture, this->DummyVolumeTexture);
  [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->ColorOpacityTexture atIndex:1];
  SetFragmentTextureOrFallback(encoder, 2, this->DepthTextureOcclusion, this->DummyDepthTexture);
  SetFragmentTextureOrFallback(encoder, 3, this->GradientOpacityTexture, this->ColorOpacityTexture);
  SetFragmentTextureOrFallback(encoder, 4, this->MaskTexture, this->DummyMaskTexture);
  SetFragmentTextureOrFallback(encoder, 5, this->LabelMapTransferTexture, this->ColorOpacityTexture);
  SetFragmentTextureOrFallback(encoder, 6, this->MinMaxTexture, this->DummyMinMaxTexture);
  SetFragmentTextureOrFallback(encoder, 16, this->MinMaxBlockTexture, this->DummyMinMaxTexture);
  SetFragmentTextureOrFallback(encoder, 17, this->MinMaxSuperTexture, this->DummyMinMaxTexture);
  SetFragmentTextureOrFallback(encoder, 7, this->GradientNormalTexture, this->DummyVolumeTexture);
  SetFragmentTextureOrFallback(encoder, 9, this->Transfer2DTexture, this->ColorOpacityTexture);
  SetFragmentTextureOrFallback(encoder, 10, this->Transfer2DYAxisTexture, this->DummyVolumeTexture);
  SetFragmentTextureOrFallback(encoder, 11, this->BlankingTexture, this->DummyVolumeTexture);
  SetFragmentTextureOrFallback(encoder, 12, this->ComponentTransferFunctionTexture1, this->ColorOpacityTexture);
  SetFragmentTextureOrFallback(encoder, 13, this->ComponentTransferFunctionTexture2, this->ColorOpacityTexture);
  SetFragmentTextureOrFallback(encoder, 14, this->ComponentTransferFunctionTexture3, this->ColorOpacityTexture);
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::BindFullscreenTextures(
  void* encoderVoid, void* uniformBufVoid,
  void* volTexVoid, void* minMaxTexVoid, void* normalTexVoid,
  bool useDepth, const void* pbd, uint32_t cullMode)
{
  id<MTLRenderCommandEncoder> encoder =
    (__bridge id<MTLRenderCommandEncoder>)encoderVoid;
  id<MTLBuffer> uniformBuf = (__bridge id<MTLBuffer>)uniformBufVoid;

  [encoder setCullMode:(MTLCullMode)cullMode];
  if (this->DepthStencilState && useDepth)
  {
    [encoder setDepthStencilState:(__bridge id<MTLDepthStencilState>)this->DepthStencilState];
  }

  [encoder setVertexBytes:pbd length:sizeof(PerBlockData) atIndex:2];
  [encoder setFragmentBytes:pbd length:sizeof(PerBlockData) atIndex:2];
  [encoder setFragmentBuffer:uniformBuf offset:0 atIndex:1];

  // Rectilinear coord curves (float3 per index). Zeroed dummy for non-rectilinear.
  id<MTLBuffer> rectCoordsBuf =
    this->RectCoordsBuffer ? (__bridge id<MTLBuffer>)this->RectCoordsBuffer
                           : (__bridge id<MTLBuffer>)this->DummyRectCoordsBuffer;
  [encoder setFragmentBuffer:rectCoordsBuf offset:0 atIndex:5];

  SetFragmentTextureOrFallback(encoder, 0, volTexVoid, this->DummyVolumeTexture);
  [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->ColorOpacityTexture atIndex:1];
  SetFragmentTextureOrFallback(encoder, 2, this->DepthTextureOcclusion, this->DummyDepthTexture);
  SetFragmentTextureOrFallback(encoder, 3, this->GradientOpacityTexture, this->ColorOpacityTexture);
  SetFragmentTextureOrFallback(encoder, 4, this->MaskTexture, this->DummyMaskTexture);
  SetFragmentTextureOrFallback(encoder, 5, this->LabelMapTransferTexture, this->ColorOpacityTexture);
  SetFragmentTextureOrFallback(encoder, 6, minMaxTexVoid, this->DummyMinMaxTexture);
  // Block summary only exists for the global (non-partitioned) lattice; the
  // per-partition path binds the dummy (walk falls back to per-cell fetches).
  SetFragmentTextureOrFallback(encoder, 16, this->MinMaxBlockTexture, this->DummyMinMaxTexture);
  SetFragmentTextureOrFallback(encoder, 17, this->MinMaxSuperTexture, this->DummyMinMaxTexture);
  SetFragmentTextureOrFallback(encoder, 7, normalTexVoid, this->DummyVolumeTexture);
  SetFragmentTextureOrFallback(encoder, 9, this->Transfer2DTexture, this->ColorOpacityTexture);
  SetFragmentTextureOrFallback(encoder, 10, this->Transfer2DYAxisTexture, this->DummyVolumeTexture);
  SetFragmentTextureOrFallback(encoder, 11, this->BlankingTexture, this->DummyVolumeTexture);
  SetFragmentTextureOrFallback(encoder, 12, this->ComponentTransferFunctionTexture1, this->ColorOpacityTexture);
  SetFragmentTextureOrFallback(encoder, 13, this->ComponentTransferFunctionTexture2, this->ColorOpacityTexture);
  SetFragmentTextureOrFallback(encoder, 14, this->ComponentTransferFunctionTexture3, this->ColorOpacityTexture);
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::BindGridTraversalTextures(
  void* encoderVoid, void* uniformBufVoid,
  void* volTexVoid, void* minMaxTexVoid, void* normalTexVoid,
  bool useDepth, const void* pbd, uint32_t cullMode)
{
  id<MTLRenderCommandEncoder> encoder =
    (__bridge id<MTLRenderCommandEncoder>)encoderVoid;

  this->BindFullscreenTextures(encoderVoid, uniformBufVoid,
    volTexVoid, minMaxTexVoid, normalTexVoid, useDepth, pbd, cullMode);

  SetFragmentTextureOrFallback(encoder, 8, this->OccupancyGridTexture, this->DummyVolumeTexture);

  id<MTLBuffer> gridBuf = (__bridge id<MTLBuffer>)this->GridTraversalUniformBuffer;
  if (!gridBuf)
  {
    vtkErrorMacro("GridTraversalUniformBuffer is nil");
    return;
  }
  [encoder setFragmentBuffer:gridBuf offset:0 atIndex:3];
}

//------------------------------------------------------------------------------

void vtkMetalGPUVolumeRayCastMapper::BuildPerBlockData(PerBlockData& pbd,
  const VolumeMapperUniforms* uniforms)
{
  pbd.VolumeBoundsMin[0] = uniforms->VolumeBoundsMin[0];
  pbd.VolumeBoundsMin[1] = uniforms->VolumeBoundsMin[1];
  pbd.VolumeBoundsMin[2] = uniforms->VolumeBoundsMin[2];
  pbd.VolumeBoundsMin[3] = 1.0f;

  pbd.VolumeBoundsMax[0] = uniforms->VolumeBoundsMax[0];
  pbd.VolumeBoundsMax[1] = uniforms->VolumeBoundsMax[1];
  pbd.VolumeBoundsMax[2] = uniforms->VolumeBoundsMax[2];
  pbd.VolumeBoundsMax[3] = 1.0f;

  pbd.TextureBoundsMin[0] = uniforms->VolumeBoundsMin[0];
  pbd.TextureBoundsMin[1] = uniforms->VolumeBoundsMin[1];
  pbd.TextureBoundsMin[2] = uniforms->VolumeBoundsMin[2];
  pbd.TextureBoundsMin[3] = 1.0f;

  pbd.TextureBoundsMax[0] = uniforms->VolumeBoundsMax[0];
  pbd.TextureBoundsMax[1] = uniforms->VolumeBoundsMax[1];
  pbd.TextureBoundsMax[2] = uniforms->VolumeBoundsMax[2];
  pbd.TextureBoundsMax[3] = 1.0f;

  pbd.GradientStep[0] = uniforms->GradientStep[0];
  pbd.GradientStep[1] = uniforms->GradientStep[1];
  pbd.GradientStep[2] = uniforms->GradientStep[2];
  pbd.GradientStep[3] = 0.0f;

  pbd.MinMaxInfo[0] = uniforms->UseMinMaxAccel;
  pbd.MinMaxInfo[1] = uniforms->MinMaxDimX;
  pbd.MinMaxInfo[2] = uniforms->MinMaxDimY;
  pbd.MinMaxInfo[3] = uniforms->MinMaxDimZ;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::BuildGlobalPerBlockData(
  PerBlockData& pbd, vtkImageData* input)
{
  int dims[3];
  input->GetDimensions(dims);

  for (int k = 0; k < 3; ++k)
  {
    pbd.VolumeBoundsMin[k]  = 0.0f;
    pbd.VolumeBoundsMax[k]  = 1.0f;
    pbd.TextureBoundsMin[k] = 0.0f;
    pbd.TextureBoundsMax[k] = 1.0f;
    pbd.GradientStep[k]     = (dims[k] > 1) ? 1.0f / (dims[k] - 1) : 1.0f;
  }
  pbd.VolumeBoundsMin[3]  = 1.0f;
  pbd.VolumeBoundsMax[3]  = 1.0f;
  pbd.TextureBoundsMin[3] = 1.0f;
  pbd.TextureBoundsMax[3] = 1.0f;
  pbd.GradientStep[3]     = 0.0f;

  pbd.MinMaxInfo[0] = this->MinMaxTexture ? 1.0f : 0.0f;
  pbd.MinMaxInfo[1] = this->MinMaxTexture ? static_cast<float>(this->MinMaxDims[0]) : 0.0f;
  pbd.MinMaxInfo[2] = this->MinMaxTexture ? static_cast<float>(this->MinMaxDims[1]) : 0.0f;
  pbd.MinMaxInfo[3] = this->MinMaxTexture ? static_cast<float>(this->MinMaxDims[2]) : 0.0f;
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::DrawBlocks(
  void* encoderVoid, void* uniformBufVoid, vtkRenderer* vtkNotUsed(ren),
  vtkVolume* vtkNotUsed(vol),
  void* uniformsVoid, vtkMatrix4x4* vtkNotUsed(invModelMatrix),
  int slabIndex, int slabCount)
{
  id<MTLRenderCommandEncoder> encoder =
    (__bridge id<MTLRenderCommandEncoder>)encoderVoid;
  id<MTLBuffer> uniformBuf = (__bridge id<MTLBuffer>)uniformBufVoid;
  VolumeMapperUniforms* uniforms = static_cast<VolumeMapperUniforms*>(uniformsVoid);
  id<MTLBuffer> indexBuf = (__bridge id<MTLBuffer>)this->IndexBuffer;

  PerBlockData pbd = {};
  BuildPerBlockData(pbd, uniforms);
  pbd.SlabInfo[0] = static_cast<float>(slabIndex);
  pbd.SlabInfo[1] = static_cast<float>(slabCount);
  const float spatial = VolumeSlabSpatial();
  if (slabCount > 1 && spatial)
  {
    pbd.SlabInfo[2] = static_cast<float>(VolumeSlabAxis(uniforms->CameraVolumePos));
    pbd.SlabInfo[3] = static_cast<float>(spatial);
  }

  [encoder setVertexBytes:&pbd length:sizeof(PerBlockData) atIndex:2];
  [encoder setFragmentBytes:&pbd length:sizeof(PerBlockData) atIndex:2];

  [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                      indexCount:this->IndexCount
                       indexType:MTLIndexTypeUInt32
                     indexBuffer:indexBuf
               indexBufferOffset:0];
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::DrawBlocksFullscreen(
  void* encoderVoid, void* uniformBufVoid, vtkRenderer* ren,
  vtkVolume* vtkNotUsed(vol),
  void* uniformsVoid, vtkMatrix4x4* vtkNotUsed(invModelMatrix),
  bool useDirectPipeline, uint32_t lightingFeatureBits)
{
  id<MTLRenderCommandEncoder> encoder =
    (__bridge id<MTLRenderCommandEncoder>)encoderVoid;
  id<MTLBuffer> uniformBuf = (__bridge id<MTLBuffer>)uniformBufVoid;
  VolumeMapperUniforms* uniforms = static_cast<VolumeMapperUniforms*>(uniformsVoid);

  uint32_t pipelineType = useDirectPipeline
    ? (uniforms->SelectionMode > 0.5f
        ? static_cast<uint32_t>(VolumePipelineType::SelectionFullscreen)
        : static_cast<uint32_t>(VolumePipelineType::FullscreenDirect))
    : static_cast<uint32_t>(VolumePipelineType::FullscreenOffscreen);

  int featureMask = 0;
  if (uniforms->UseGradientShading > 0.5f)
    featureMask |= VolumeFeature_Shading;
  if (uniforms->UseGradientOpacity > 0.5f)
    featureMask |= VolumeFeature_GradientOpacity;
  if (uniforms->UseMask > 0.5f)
    featureMask |= VolumeFeature_Mask;
  if (uniforms->UseMinMaxAccel > 0.5f)
    featureMask |= VolumeFeature_MinMax;
  if (uniforms->UseNormalTexture > 0.5f)
    featureMask |= VolumeFeature_NormalTexture;
  if (uniforms->UseComputeNormalFromOpacity > 0.5f)
    featureMask |= VolumeFeature_ComputeNormalFromOpacity;
  if (uniforms->UseLinearVolumeInterpolation > 0.5f)
    featureMask |= VolumeFeature_LinearInterpolation;
  featureMask |= BlendModeToFeatureFlag(this->GetBlendMode());
  if (VolumeFeatureIndependentPath(*uniforms, featureMask))
    featureMask |= VolumeFeature_IndependentComponents;
  if (uniforms->UseTransfer2D > 0.5f)
    featureMask |= VolumeFeature_Transfer2D;
  if (uniforms->UseRectilinear > 0.5f)
    featureMask |= VolumeFeature_Rectilinear;
  if (uniforms->UseDependentRGBA > 0.5f)
    featureMask |= VolumeFeature_DependentRGBA;
  if (uniforms->UseDependentLA > 0.5f)
    featureMask |= VolumeFeature_DependentLA;
  featureMask |= lightingFeatureBits;
  if (uniforms->UseCropping > 0.5f)
    featureMask |= VolumeFeature_Cropping;
  if (uniforms->UseBlanking > 0.5f)
    featureMask |= VolumeFeature_Blanking;

  // March-experiment selector (VTK_METAL_TEST_MARCH_VARIANT): encoded into the
  // feature mask so each experiment gets its own specialized pipeline.
  if (const int marchVariant = VolumeMarchVariant(); marchVariant != 0)
  {
    featureMask |=
      (static_cast<uint32_t>(marchVariant) & 0xFu) << VolumeFeature_MarchVariantShift;
  }

  // V31 back-edge exit experiment (VTK_METAL_TEST_DOEXIT): dedicated feature
  // bit so the reshaped march gets its own specialized pipeline.
  if (VolumeMarchDoExit())
  {
    featureMask |= VolumeFeature_MarchDoExit;
  }

  // RG8 pair-packed slice representation experiment (VTK_METAL_TEST_RG8):
  // dedicated feature bit so the pair-tap sampler gets its own pipeline.
  if (VolumeRg8PairActive())
  {
    featureMask |= VolumeFeature_VolRg8;
  }

  // Transposed volume representation (VTK_METAL_TEST_VOLTRANSPOSE):
  // dedicated feature bit so the swizzled-fetch march gets its own pipeline.
  // Gated on the POLICY result recorded by this frame's volume upload (not
  // the raw env): when dims make depth already shortest the upload stays
  // identity and the swizzled pipeline must NOT be selected. The orientation
  // itself rides the key's featureMaskExtra (fc_volTransposedY).
  if (this->VolumeTextureAxisDepth != 0)
  {
    featureMask |= VolumeFeature_VolTransposed;
  }

  // Composite slab tiling (VTK_METAL_TEST_NUM_SLABS): only on the blended
  // direct path (useDirectPipeline), and only for composite blending — the
  // (ONE, ONE_MINUS_SRC_ALPHA) accumulation of per-slab partial composites is
  // exact only because front-to-back premultiplied `over` is associative. The
  // count is resolved per frame (adaptive to the view alignment).
  const int numSlabs =
    (useDirectPipeline && this->GetBlendMode() == vtkVolumeMapper::COMPOSITE_BLEND &&
     ResolveNumSlabs(uniforms->CameraVolumePos) > 1)
      ? ResolveNumSlabs(uniforms->CameraVolumePos)
      : 1;
  if (numSlabs > 1)
  {
    featureMask |= VolumeFeature_Slab;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)
    (static_cast<vtkMetalRenderWindow*>(ren->GetRenderWindow()))->GetMetalDevice();

  uint32_t colorFormat = useDirectPipeline ? MTLPixelFormatBGRA8Unorm : MTLPixelFormatRGBA16Float;
  uint32_t depthFormat = useDirectPipeline ? MTLPixelFormatDepth32Float : MTLPixelFormatInvalid;
  auto* metalRenderWindow = vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());
  uint32_t sampleCount = useDirectPipeline
    ? static_cast<uint32_t>(metalRenderWindow ? metalRenderWindow->GetEffectiveSampleCount() : 1)
    : 1;

  void* pso = this->GetOrCreateVolumePipeline(
    (__bridge void*)device, pipelineType, colorFormat, depthFormat, sampleCount, featureMask);
  if (!pso) return;
  [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)pso];

  PerBlockData pbd = {};
  BuildPerBlockData(pbd, uniforms);

  this->BindFullscreenTextures(encoder, uniformBuf,
    this->VolumeTexture,
    this->MinMaxTexture,
    this->GradientNormalTexture,
    useDirectPipeline, &pbd, MTLCullModeBack);

  // Back-to-front slab passes: each composites only its ray-length-fraction
  // index range from zero; the (ONE, ONE_MINUS_SRC_ALPHA) blend on attachment 0
  // accumulates them. Premultiplied `over` is associative only back-to-front.
  // Spatial mode (VTK_METAL_TEST_SLAB_SPATIAL, SLAB_BENCHMARKS.md §5.2):
  // SlabInfo[2] = dominant view axis, SlabInfo[3] = 1; the shader then splits
  // by uniform planes perpendicular to that axis, keeping each pass' fetch set
  // a thin flat band of the volume instead of a ray-space wedge.
  if (numSlabs > 1 && VolumeSlabSpatial())
  {
    pbd.SlabInfo[2] = static_cast<float>(VolumeSlabAxis(uniforms->CameraVolumePos));
    pbd.SlabInfo[3] = VolumeSlabSpatial();
  }
  for (int s = numSlabs - 1; s >= 0; --s)
  {
    pbd.SlabInfo[0] = static_cast<float>(s);
    pbd.SlabInfo[1] = static_cast<float>(numSlabs);
    [encoder setFragmentBytes:&pbd length:sizeof(PerBlockData) atIndex:2];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  }
}

//------------------------------------------------------------------------------
void vtkMetalGPUVolumeRayCastMapper::GPURender(vtkRenderer* ren, vtkVolume* vol)
{
  @autoreleasepool
  {
  auto* metalRenderer = vtkMetalRenderer::SafeDownCast(ren);
  auto* metalRenderWindow = vtkMetalRenderWindow::SafeDownCast(ren->GetRenderWindow());
  if (!metalRenderer || !metalRenderWindow)
  {
    return;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)metalRenderWindow->GetMetalDevice();
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)metalRenderWindow->GetMetalQueue();
  id<MTLCommandBuffer> commandBuffer =
    (__bridge id<MTLCommandBuffer>)metalRenderWindow->GetCurrentCommandBuffer();

  if (!device || !commandBuffer)
  {
    return;
  }

  void* mtlDevice = metalRenderWindow->GetMetalDevice();
  void* mtlQueue = metalRenderWindow->GetMetalQueue();

  // §38.18.1: "reboot without reboot" — if the env flag is set, purge the
  // Private-heap SegPool/RayBin (64 MB + 4×W×H) and the PSO caches that are
  // otherwise retained until ReleaseGraphicsResources/MTLDevice teardown.
  // Must run BEFORE the per-frame semaphore wait at 10136 so the 3-slot
  // drain inside PurgeCaches cannot deadlock against the held slot.
  if (VolumePurgeRequested())
  {
    this->PurgeCaches();
  }

  if (getenv("VTK_METAL_TEST_DUMP_UNIFORMS"))
  {
    double* bp = this->ModelBounds;
    vtkCamera* cam = ren->GetActiveCamera();
    double* cpos = cam ? cam->GetPosition() : nullptr;
    fprintf(stderr, "GPURender enter: bounds=(%.2f,%.2f,%.2f)-(%.2f,%.2f,%.2f) cam=%s\n",
      bp[0], bp[1], bp[2], bp[3], bp[4], bp[5],
      cpos ? "set" : "null");
  }

  if (!this->EnsureEffectiveInput())
  {
    return;
  }
  vtkImageData* input = this->EffectiveInput;
  if (!input)
  {
    return;
  }

  // Cache scalar range once (used by both TF texture and uniforms). Resolve the
  // active scalar array via the mapper's scalar mode (see UpdateVolumeTexture).
  // Note: pass a local cellFlag so GetScalars (which takes it by reference and
  // sets it based on where the scalars were found) cannot clobber the member,
  // which still describes the original input's scalar association.
  int cellFlag = this->CellFlag;
  vtkDataArray* scalars = this->GetScalars(
    input, this->ScalarMode, this->ArrayAccessMode, this->ArrayId, this->ArrayName, cellFlag);
  if (scalars)
  {
    scalars->GetRange(this->ScalarRange, 0);
    // Per-component scalar ranges for the independent multi-component path
    // (OpenGL ScalarRange[n] parity).
    const int numComp = scalars->GetNumberOfComponents();
    for (int c = 0; c < 4; ++c)
    {
      if (c < numComp)
      {
        scalars->GetRange(this->ComponentScalarRange[c], c);
      }
      else
      {
        this->ComponentScalarRange[c][0] = 0.0;
        this->ComponentScalarRange[c][1] = 1.0;
      }
    }
  }
  else
  {
    this->ScalarRange[0] = 0.0;
    this->ScalarRange[1] = 1.0;
    for (int c = 0; c < 4; ++c)
    {
      this->ComponentScalarRange[c][0] = 0.0;
      this->ComponentScalarRange[c][1] = 1.0;
    }
  }

  // Phase 5: GPU-accelerated min-max generation.
  // For single-block volumes with UseGPUMinMax, we must upload the volume
  // texture first, then dispatch compute kernels. For partitioned volumes
  // (or when GPU min-max is disabled), the CPU path runs before volume
  // upload so that UpdateBlockTextures can reuse the per-macrocell data.
  bool usePartitions = (this->Partitions[0] > 1 || this->Partitions[1] > 1 || this->Partitions[2] > 1);

  if (!this->UseMinMaxAcceleration)
  {
    // Master switch off: no occupancy lattice is built at all, so the shader
    // marches every sample (useMinMax == false -> raw, unaccelerated ray cast).
    // This is the apples-to-apples comparison for backends without min-max
    // acceleration. UpdateBlockTextures falls back to walking the voxels for
    // per-block ranges when MacrocellScalarMin is empty.
    this->MacrocellScalarMin.clear();
    this->MacrocellScalarMax.clear();
    ReleaseMetalObject(this->MinMaxTexture);
    if (!this->UpdateVolumeTexture(mtlDevice, mtlQueue, vol))
    {
      return;
    }
    // UpdateVolumeTexture may rebuild the macrocell lattice internally for
    // partitioned volumes; drop it again so the march stays unaccelerated.
    ReleaseMetalObject(this->MinMaxTexture);
  }
  else if (this->UseGPUMinMax && !usePartitions)
  {
    // GPU min-max path: upload volume texture first, then dispatch compute.
    if (!this->UpdateVolumeTexture(mtlDevice, mtlQueue, vol))
    {
      return;
    }

    if (!this->ComputeMinMaxGPU(mtlDevice, mtlQueue, vol, input, scalars))
    {
      // GPU path failed — release any private texture so the CPU
      // fallback's replaceRegion (which needs StorageModeShared) works.
      ReleaseMetalObject(this->MinMaxTexture);
      this->UpdateMinMaxTexture(mtlDevice, vol, input, scalars, false);
    }
  }
  else if (this->UseGPUMinMax && usePartitions)
  {
    // GPU min-max path for partitioned volumes: skip the expensive CPU macrocell
    // voxel scan. UpdateBlockTextures generates per-block minmax textures on GPU
    // directly from each block's uploaded texture, reusing volume_compute_minmax
    // and volume_dilate_minmax compute kernels with block-specific uniforms.
    //
    // Still run a lightweight CPU macrocell scan to populate MacrocellScalarMin/Max
    // so that UpdateBlockTextures (called inside UpdateVolumeTexture) can compute
    // per-block scalar ranges via macrocell reduction instead of an expensive
    // per-voxel walk that defeats the purpose of GPU acceleration.
    if (this->MacrocellScalarMin.empty() || this->MacrocellScalarMax.empty())
    {
      this->UpdateMinMaxTexture(mtlDevice, vol, input, scalars, true);
    }
    if (!this->UpdateVolumeTexture(mtlDevice, mtlQueue, vol))
    {
      return;
    }
  }
  else
  {
    // CPU min-max path: compute from raw scalar data before volume upload.
    // When UseGPUMinMax is false the lattice uses DS=4 (see
    // ComputeMacrocellDownsample), matching the historical cell size so the
    // CPU-computed occupancy skip behaves like the pre-adaptive baseline.
    this->UpdateMinMaxTexture(mtlDevice, vol, input, scalars, usePartitions);

    if (!this->UpdateVolumeTexture(mtlDevice, mtlQueue, vol))
    {
      return;
    }
  }

  this->EnsureGradientNormalTexture(mtlDevice, mtlQueue, vol);

  // Compute actual sample distance early so it can be used for opacity pre-integration
  // (matches the OpenGL mapper's vtkInternal::ComputeSampleDistance).
  this->ComputeReductionFactor(vol->GetAllocatedRenderTime());

  double actualSampleDistance;
  if (this->AutoAdjustSampleDistances)
  {
    vtkNew<vtkMatrix4x4> modelToWorld;
    vol->GetModelToWorldMatrix(modelToWorld);
    double cellSpacing[3];
    input->GetSpacing(cellSpacing);
    double minWorldSpacing = VTK_DOUBLE_MAX;
    for (int i = 0; i < 3; ++i)
    {
      double tmp = modelToWorld->GetElement(0, i);
      double tmp2 = tmp * tmp;
      tmp = modelToWorld->GetElement(1, i);
      tmp2 += tmp * tmp;
      tmp = modelToWorld->GetElement(2, i);
      tmp2 += tmp * tmp;
      double worldSpacing = fabs(cellSpacing[i] * sqrt(tmp2));
      minWorldSpacing = std::min(worldSpacing, minWorldSpacing);
    }
    actualSampleDistance = static_cast<float>(minWorldSpacing);
    if (this->ReductionFactor < 1.0 && this->ReductionFactor != 0.0)
    {
      actualSampleDistance /= static_cast<float>(this->ReductionFactor);
    }
  }
  else if (this->LockSampleDistanceToInputSpacing)
  {
    double cellSpacing[3];
    input->GetSpacing(cellSpacing);
    int extents[6];
    input->GetExtent(extents);
    double spacingDist = (cellSpacing[0] + cellSpacing[1] + cellSpacing[2]) / 6.0;
    double avgNumVoxels = pow(
      static_cast<double>((extents[1] - extents[0]) * (extents[3] - extents[2]) *
        (extents[5] - extents[4])),
      0.333);
    if (avgNumVoxels < 100)
    {
      spacingDist *= 0.01 + (1 - 0.01) * avgNumVoxels / 100;
    }
    float d = static_cast<float>(spacingDist);
    float sample = static_cast<float>(this->SampleDistance);
    actualSampleDistance =
      (sample / d < 0.999f || sample / d > 1.001f) ? d : this->SampleDistance;
  }
  else
  {
    actualSampleDistance = this->GetSampleDistance();
  }

  if (!this->UpdateTransferFunctionTexture(mtlDevice, mtlQueue, vol, actualSampleDistance))
  {
    return;
  }
  this->UpdateGradientOpacityTexture(mtlDevice, mtlQueue, vol);

  // 2D transfer function mode (TF_2D): upload the 2D lookup image and the
  // Y-axis scalar array ("Temp" in TestGPURayCastTransfer2DYScalars).
  if (!this->UpdateTransfer2DTexture(mtlDevice, mtlQueue, vol))
  {
    return;
  }
  if (this->Transfer2DEnabled)
  {
    this->UpdateTransfer2DYAxisTexture(mtlDevice, mtlQueue, vol, input);
  }

  // Update mask / label map textures
  if (this->MaskInput &&
    (this->MaskType == vtkGPUVolumeRayCastMapper::LabelMapMaskType ||
      this->MaskType == vtkGPUVolumeRayCastMapper::BinaryMaskType))
  {
    this->UpdateMaskTexture(mtlDevice, mtlQueue, vol);
    if (this->MaskType == vtkGPUVolumeRayCastMapper::LabelMapMaskType)
    {
      this->UpdateLabelMapTransferTexture(mtlDevice, mtlQueue, vol);
    }
  }

  if (!this->SetupBuffers(mtlDevice, ren, vol, input))
  {
    return;
  }
  if (!this->SetupPipeline(mtlDevice, ren))
  {
    return;
  }

  // Update uniforms
  VolumeMapperUniforms uniforms = {};

  // Debug iter PPM: repurpose _padCropFlags[0] as a flag. When set, the shader
  // outputs iteration count / 256 in the red channel instead of the normal
  // color, so a PNG/PPM readback gives the iter distribution.
  uniforms._padCropFlags[0] = getenv("METAL_ITER") ? 1.0f : 0.0f;

  // JSCALE probe (_padCropFlags[1], VTK_METAL_TEST_JSCALE): scales the
  // per-pixel jitter phase spread toward the coherent j0 lattice
  // (1 + s*(noise-1)); always written so the shader sees a valid [0,1]
  // scale (default 1.0 = native j1; 0 = coherent j0-equivalent phase).
  uniforms._padCropFlags[1] = 1.0f;
  if (const char* js = getenv("VTK_METAL_TEST_JSCALE"))
  {
    const float v = static_cast<float>(std::atof(js));
    if (v >= 0.0f && v <= 1.0f) uniforms._padCropFlags[1] = v;
  }

  // NOPREFETCH probe (_padCropFlags[2], VTK_METAL_TEST_NOPREFETCH): drop the
  // prefetch-ahead pipeline so the march issues one volume fetch per
  // iteration like OpenGL's composed loop.
  if (getenv("VTK_METAL_TEST_NOPREFETCH")) uniforms._padCropFlags[2] = 1.0f;

  // TEMP probe z (_padCropFlags[3], VTK_METAL_TEST_PROBE_Z).
  // VTK_METAL_TEST_PROBE_RAW=1 negates the value -> raw texture-plane probe.
  if (const char* pzv = getenv("VTK_METAL_TEST_PROBE_Z"))
  {
    uniforms._padCropFlags[3] = static_cast<float>(std::atof(pzv));
    if (getenv("VTK_METAL_TEST_PROBE_RAW"))
      uniforms._padCropFlags[3] = -uniforms._padCropFlags[3];
  }

  // TEMP-DIAG minmax walk probe (_padCropFlags[4], MM_PROBE): with METAL_ITER,
  // the debug exit returns the baseline march's visits/crossings/skipped-steps
  // instead of marchIter (R*64/G*16/B*64 decode). Investigation-only.
  if (getenv("MM_PROBE")) uniforms._padCropFlags[4] = 1.0f;

  vtkNew<vtkMatrix4x4> modelMatrix;
  vol->GetModelToWorldMatrix(modelMatrix);

  vtkNew<vtkMatrix4x4> invModelMatrix;
  vtkMatrix4x4::Invert(modelMatrix, invModelMatrix);

  for (int r = 0; r < 4; ++r)
  {
    for (int c = 0; c < 4; ++c)
    {
      uniforms.VolumeToWorldMatrix[c * 4 + r] = modelMatrix->GetElement(r, c);
      uniforms.WorldToVolumeMatrix[c * 4 + r] = invModelMatrix->GetElement(r, c);
    }
  }

  // TextureToVolume / VolumeToTexture (OpenGL TextureToDataset parity): maps
  // [0,1] texture coordinates to model-space (rotated dataset) coordinates,
  // including the image-data direction matrix. The box geometry spans the
  // rotated AABB, so the vertex shader uses VolumeToTexture to compute the
  // texture coordinate of each corner and the fragment shader maps every
  // sample position back to texture space before looking up the volume.
  {
    int ext[6];
    double origin[3], spacing[3];
    input->GetExtent(ext);
    input->GetOrigin(origin);
    input->GetSpacing(spacing);
    vtkMatrix3x3* dir = input->GetDirectionMatrix();

    double width[3];
    for (int c = 0; c < 3; ++c)
    {
      width[c] = std::fabs(spacing[c]) * static_cast<double>(ext[2 * c + 1] - ext[2 * c]);
    }

    double blockOrigin[3];
    vtkImageData::TransformContinuousIndexToPhysicalPoint(
      ext[0], ext[2], ext[4], origin, spacing, dir->GetData(), blockOrigin);

    vtkNew<vtkMatrix4x4> texToVol;
    texToVol->Identity();
    for (int c = 0; c < 3; ++c)
    {
      for (int i = 0; i < 3; ++i)
      {
        texToVol->SetElement(c, i, dir->GetElement(c, i) * width[i]);
      }
      texToVol->SetElement(c, 3, blockOrigin[c]);
    }

    vtkNew<vtkMatrix4x4> volToTex;
    vtkMatrix4x4::Invert(texToVol, volToTex);

    for (int r = 0; r < 4; ++r)
    {
      for (int c = 0; c < 4; ++c)
      {
        uniforms.TextureToVolumeMatrix[c * 4 + r] = texToVol->GetElement(r, c);
        uniforms.VolumeToTextureMatrix[c * 4 + r] = volToTex->GetElement(r, c);
      }
    }
  }

  VolumeBounds vb;
  vb.Min[0] = this->ModelBounds[0];
  vb.Max[0] = this->ModelBounds[1];
  vb.Min[1] = this->ModelBounds[2];
  vb.Max[1] = this->ModelBounds[3];
  vb.Min[2] = this->ModelBounds[4];
  vb.Max[2] = this->ModelBounds[5];
  for (int k = 0; k < 3; ++k)
  {
    vb.Size[k] = vb.Max[k] - vb.Min[k];
    if (vb.Size[k] < 1e-10)
      vb.Size[k] = 1.0;
  }

  uniforms.VolumeBoundsMin[0] = static_cast<float>(vb.Min[0]);
  uniforms.VolumeBoundsMin[1] = static_cast<float>(vb.Min[1]);
  uniforms.VolumeBoundsMin[2] = static_cast<float>(vb.Min[2]);
  uniforms.VolumeBoundsMin[3] = 1.0f;

  uniforms.VolumeBoundsMax[0] = static_cast<float>(vb.Max[0]);
  uniforms.VolumeBoundsMax[1] = static_cast<float>(vb.Max[1]);
  uniforms.VolumeBoundsMax[2] = static_cast<float>(vb.Max[2]);
  uniforms.VolumeBoundsMax[3] = 1.0f;

  double* camPosWorld = ren->GetActiveCamera()->GetPosition();
  double camPosVolume[4] = { camPosWorld[0], camPosWorld[1], camPosWorld[2], 1.0 };
  invModelMatrix->MultiplyPoint(camPosVolume, camPosVolume);

  uniforms.CameraVolumePos[0] = NormalizeToVolumeSpace(vb, 0, camPosVolume[0]);
  uniforms.CameraVolumePos[1] = NormalizeToVolumeSpace(vb, 1, camPosVolume[1]);
  uniforms.CameraVolumePos[2] = NormalizeToVolumeSpace(vb, 2, camPosVolume[2]);
  uniforms.CameraVolumePos[3] = 1.0f;

  // Camera-inside near-plane clip (OpenGL near-plane proxy-clip parity): when
  // the near frustum plane crosses the bounding box, OpenGL clips the proxy box
  // against the near plane (pushed into the volume by a precision offset) and
  // starts the march there, so the eye->near-plane slab is never sampled. The
  // fullscreen/proxy shaders reconstruct the ray from the eye and only intersect
  // the box, so setupVolumeRay clamps the entry to this plane. The plane is
  // expressed in [0,1] normalized volume space (origin via
  // NormalizeToVolumeSpace, normal scaled by the per-axis bounds size) so the
  // per-ray intersection distance is directly comparable to the box t-range.
  uniforms.UseCameraInsideNearClip = this->IsCameraInside(ren, vol) ? 1.0f : 0.0f;
  uniforms.UseDataSpaceBoxVertices =
    (uniforms.UseCameraInsideNearClip > 0.5f) ? 0.0f : 1.0f;
  uniforms.CameraInsideNearPlaneOrigin[3] = 1.0f;
  uniforms.CameraInsideNearPlaneNormal[3] = 0.0f;
  if (uniforms.UseCameraInsideNearClip > 0.5f)
  {
    vtkCamera* cam = ren->GetActiveCamera();
    double fplanes[24];
    cam->GetFrustumPlanes(ren->GetTiledAspectRatio(), fplanes);

    // Near frustum plane (index 4*4=16) in world space.
    double pNormal[3];
    double pOrigin[4] = { 0.0, 0.0, 0.0, 1.0 };
    for (int i = 0; i < 3; ++i)
    {
      pNormal[i] = fplanes[16 + i];
      pOrigin[i] = -fplanes[16 + 3] * fplanes[16 + i];
    }

    // Transform origin to model (data) space via the inverse model matrix, and
    // the normal via the TRANSPOSE of the model matrix (same convention as the
    // clipped-proxy geometry build in SetupBuffers and OpenGL's
    // RenderVolumeGeometry: for x_world = M x_obj, n_obj = M^T n_world; the
    // inverse transpose is the reverse-direction transform and diverges under
    // a non-uniform model scale).
    invModelMatrix->MultiplyPoint(pOrigin, pOrigin);
    double* dmat = modelMatrix->GetData();
    modelMatrix->Transpose();
    double pNormalV[3];
    pNormalV[0] = pNormal[0] * dmat[0] + pNormal[1] * dmat[1] + pNormal[2] * dmat[2];
    pNormalV[1] = pNormal[0] * dmat[4] + pNormal[1] * dmat[5] + pNormal[2] * dmat[6];
    pNormalV[2] = pNormal[0] * dmat[8] + pNormal[1] * dmat[9] + pNormal[2] * dmat[10];
    vtkMath::Normalize(pNormalV);
    modelMatrix->Transpose();

    // Precision offset identical to OpenGL's (and SetupBuffers'): a fraction of
    // the near-far distance, floored for very small volumes to avoid hardware
    // near-plane clipping.
    double offset = (cam->GetClippingRange()[1] - cam->GetClippingRange()[0]) * 0.001;
    double minOffset = static_cast<double>(std::numeric_limits<float>::epsilon()) * 1000.0;
    offset = offset < minOffset ? minOffset : offset;
    for (int i = 0; i < 3; ++i)
    {
      pOrigin[i] += (pNormalV[i] * offset);
    }

    uniforms.CameraInsideNearPlaneOrigin[0] = NormalizeToVolumeSpace(vb, 0, pOrigin[0]);
    uniforms.CameraInsideNearPlaneOrigin[1] = NormalizeToVolumeSpace(vb, 1, pOrigin[1]);
    uniforms.CameraInsideNearPlaneOrigin[2] = NormalizeToVolumeSpace(vb, 2, pOrigin[2]);

    // Normal in normalized volume space: componentwise scaled by the bounds size
    // (the [0,1] frame stretches each axis by 1/Size), then renormalized.
    double pNormalN[3];
    for (int i = 0; i < 3; ++i)
    {
      pNormalN[i] = pNormalV[i] * vb.Size[i];
    }
    vtkMath::Normalize(pNormalN);
    uniforms.CameraInsideNearPlaneNormal[0] = static_cast<float>(pNormalN[0]);
    uniforms.CameraInsideNearPlaneNormal[1] = static_cast<float>(pNormalN[1]);
    uniforms.CameraInsideNearPlaneNormal[2] = static_cast<float>(pNormalN[2]);
  }

  // Parallel-projection support (OpenGL in_projectionDirection parity): the
  // fragment shader needs a constant ray direction expressed in [0,1] volume
  // space. Transform the camera's world direction of projection into model
  // (volume) space, then map it to the normalized volume frame (same mapping
  // as cameraVolumePos) and normalize. With the perspective path unchanged
  // (rays converge to CameraVolumePos) parallel cameras now cast parallel rays.
  const bool parallelProjection =
    ren->GetActiveCamera()->GetParallelProjection();
  uniforms.UseParallelProjection = parallelProjection ? 1.0f : 0.0f;
  uniforms.ProjectionDirection[0] = 0.0f;
  uniforms.ProjectionDirection[1] = 0.0f;
  uniforms.ProjectionDirection[2] = 1.0f;
  uniforms.ProjectionDirection[3] = 0.0f;
  if (parallelProjection)
  {
    double dir[3];
    ren->GetActiveCamera()->GetDirectionOfProjection(dir);
    double dirV[4] = { dir[0], dir[1], dir[2], 0.0 };
    invModelMatrix->MultiplyPoint(dirV, dirV);
    if (vb.Size[0] > 1e-10)
      dirV[0] /= vb.Size[0];
    if (vb.Size[1] > 1e-10)
      dirV[1] /= vb.Size[1];
    if (vb.Size[2] > 1e-10)
      dirV[2] /= vb.Size[2];
    double len = vtkMath::Norm(dirV);
    if (len > 1e-12)
    {
      uniforms.ProjectionDirection[0] = static_cast<float>(dirV[0] / len);
      uniforms.ProjectionDirection[1] = static_cast<float>(dirV[1] / len);
      uniforms.ProjectionDirection[2] = static_cast<float>(dirV[2] / len);
    }
  }

  double maxBoundsSize = std::max({ vb.Size[0], vb.Size[1], vb.Size[2] });

  uniforms.SampleDistanceHalf =
    FloatToHalf(static_cast<float>(actualSampleDistance / maxBoundsSize));

  // Opacity pre-integration is baked into the transfer function texture
  // on the CPU at TF-build time (matches OpenGL backend).
  // Set to 1.0 (no-op) in case any shader variant still reads this field.
  // Invariant: the march step's physical length equals actualSampleDistance,
  // enforced in-shader by physicalSampleStep, matching this pre-integration factor.
  uniforms.OpacityPreIntegrationFactorHalf = FloatToHalf(1.0f);

  {
    float normFactor = this->ScalarNormalizationFactor;
    // OpenGL in_volume_scale parity: GL's GetScaleAndBias divides by max+1 for
    // normalized integer formats (R8Unorm/R16Unorm -> glScale = 1/256, 1/65536),
    // so ScalarMin/ScalarMax must be uploaded in that space for the shader's
    // 1.0f/max(scalarMax - scalarMin, 1e-4) to reproduce GL's in_volume_scale
    // bit-for-bit (USHORT 0..4370: max = 4370/65536 = 0.0666809 -> scale =
    // 14.9967966 == GL's). Float data keeps /normFactor (== /1; GL glScale = 1).
    float glDenom = normFactor;
    if (normFactor == 255.0f || normFactor == 65535.0f)
    {
      glDenom = normFactor + 1.0f;
    }
    uniforms.ScalarMinHalf = FloatToHalf(static_cast<float>(this->ScalarRange[0] / glDenom));
    uniforms.ScalarMaxHalf = FloatToHalf(static_cast<float>(
      (this->ScalarRange[1] > this->ScalarRange[0]
         ? this->ScalarRange[1]
         : this->ScalarRange[0] + 1.0) /
      glDenom));
  }

  // Independent multi-component support (OpenGL in_scalarsRange parity):
  // per-component scalar ranges (divided by the volume normalization factor,
  // in the same [0,1] space as the normalized sample), component weights, and
  // the component/independent flags consumed by the shader's
  // useIndependentPath branch.
  {
    vtkVolumeProperty* prop = vol->GetProperty();
    const bool independent =
      prop && prop->GetIndependentComponents() && this->VolumeNumComponents > 1;
    uniforms.UseIndependentComponents = independent ? 1.0f : 0.0f;
    // Dependent 2-component mode (LA): color is the color LUT at the first
    // component's normalized value and opacity is the opacity LUT at the LAST
    // component's normalized value (OpenGL computeColor/computeOpacity LA
    // parity; the table splits RGB over component 0's range and A over the last
    // component's range).
    uniforms.UseDependentLA =
      prop && !prop->GetIndependentComponents() && (this->VolumeNumComponents == 2) ? 1.0f : 0.0f;
    // 4-component dependent mode treats the volume as raw RGBA: the shader uses
    // scalar.xyz directly for color and the opacity LUT for scalar.w (OpenGL
    // computeColor/computeOpacity RGBA parity).
    uniforms.UseDependentRGBA =
      prop && !prop->GetIndependentComponents() && (this->VolumeNumComponents == 4) ? 1.0f : 0.0f;
    uniforms.NumComponents =
      static_cast<uint32_t>(std::max(1, std::min(4, this->VolumeNumComponents)));

    float normFactor = this->ScalarNormalizationFactor;
    for (int c = 0; c < 4; ++c)
    {
      uniforms.ComponentWeight[c] =
        prop ? static_cast<float>(prop->GetComponentWeight(c)) : 1.0f;
      double cMin = this->ComponentScalarRange[c][0];
      double cMax = (this->ComponentScalarRange[c][1] > this->ComponentScalarRange[c][0])
        ? this->ComponentScalarRange[c][1]
        : this->ComponentScalarRange[c][0] + 1.0;
      uniforms.ScalarMinCompHalf[c] = FloatToHalf(static_cast<float>(cMin / normFactor));
      uniforms.ScalarMaxCompHalf[c] = FloatToHalf(static_cast<float>(cMax / normFactor));
    }
  }

  uniforms.UseJittering = this->GetUseJittering() ? 1.0f : 0.0f;
  uniforms.UseIGNJitter = this->GetUseIGNJitter() ? 1.0f : 0.0f;
  uniforms.JitterBlockSize = static_cast<float>(this->GetJitterBlockSize());
  // GL-parity jitter (VTK_METAL_TEST_JITTER_PARITY=1): sample the blue-noise
  // tile at gl_FragCoord.xy/64 like GL (block = viewport/64 px) instead of
  // per-pixel. Reproduces GL's jitter field exactly and keeps SIMT lanes in
  // lockstep (~+4-5% harness at 2048/SD4 vs +60% per-pixel). nSize 0 selects
  // the shader's parity branch (MetalShaders.metal sampleJitterNoise).
  if (const char* parity = std::getenv("VTK_METAL_TEST_JITTER_PARITY"); parity && std::atoi(parity) != 0)
  {
    const char* bsEnv = std::getenv("VTK_METAL_TEST_JITTER_BLOCK_SIZE");
    uniforms.JitterBlockSize = bsEnv ? static_cast<float>(std::atof(bsEnv)) : 0.0f;
  }

  // Non-divergent march: uniform per-frame iteration bound.
  // - variant 4: VTK_METAL_TEST_MARCH_STEPS wins; otherwise a frame-max bound
  //   (longest ray-box chord for the current camera / sample distance) so every
  //   fragment reaches at least its own legacy per-fragment maxSteps. 0 =
  //   legacy per-fragment loop bound (variant 4 disabled).
  // - variant 5: hybrid uniform-main + divergent-tail. VTK_METAL_TEST_MARCH_STEPS
  //   wins as the uniform main-loop count; otherwise the frame-average chord is
  //   used so the uniform main phase covers the bulk of every ray with SIMT
  //   lanes locked, and only the ~15-20 % longer rays spill into the tail.
  // - any other variant: MARCH_STEPS still caps the per-fragment bound (the
  //   fixed-steps probe; MaxStepsFrame is 0 in production so the shader's
  //   baseline loop bound is untouched).
  uniforms.MaxStepsFrame = 0.0f;
  {
    const int fixedSteps = VolumeMarchSteps();
    if (fixedSteps > 0)
    {
      uniforms.MaxStepsFrame = static_cast<float>(fixedSteps);
      if (getenv("VTK_METAL_TEST_MARCH_DEBUG"))
        fprintf(stderr, "[march] MARCH_STEPS fixedSteps=%d -> maxStepsFrame=%.1f\n",
          fixedSteps, uniforms.MaxStepsFrame);
    }
    else if (VolumeMarchVariant() == 4 || VolumeMarchVariant() == 5)
    {
      const float camV[3] = {
        uniforms.CameraVolumePos[0], uniforms.CameraVolumePos[1],
        uniforms.CameraVolumePos[2]};
      if (getenv("VTK_METAL_TEST_MARCH_DEBUG"))
        fprintf(stderr, "[march] camV=(%.3f,%.3f,%.3f) size=(%.1f,%.1f,%.1f)\n",
          camV[0], camV[1], camV[2], vb.Size[0], vb.Size[1], vb.Size[2]);
      double meanChordMM = 0.0;
      const double maxChordMM = ComputeMaxChordMM(vb.Size[0], vb.Size[1], vb.Size[2],
        camV, uniforms.UseParallelProjection > 0.5f, uniforms.ProjectionDirection,
        actualSampleDistance, &meanChordMM);
      const double boundChordMM =
        (VolumeMarchVariant() == 5) ? meanChordMM : maxChordMM;
      uniforms.MaxStepsFrame = static_cast<float>(std::max(
        1.0, std::ceil(boundChordMM / std::max(actualSampleDistance, 1e-9))));
      if (getenv("VTK_METAL_TEST_MARCH_DEBUG"))
        fprintf(stderr, "[march] variant=%d maxStepsFrame=%.1f maxChordMM=%.2f meanChordMM=%.2f sampleDist=%.3f\n",
          VolumeMarchVariant(), uniforms.MaxStepsFrame, maxChordMM, meanChordMM,
          actualSampleDistance);
    }
  }

  // Adaptive-width march cap for fc_marchVariant 9: SINGLE-TIER 32 for all
  // sample distances (HARNESS_VS_APP_GAP §37.11, 2026-08-23). The old SD-tier
  // map {<2:48, <3:16, else:8} predates the block-summary leaps; with blocks
  // default-on the solid runs between skips are long enough that wide batches
  // no longer waste slots, and 32 measured fastest in EVERY cell probed —
  // mm and raw arms alike, SD0.5/2.5/4, 400²..4096² (raw@cap-parity also
  // refuted the fine-tier 48 rationale). The ladder tops out at 48 unrolled
  // fetches; caps >= 48 dispatch identically. VTK_METAL_TEST_MARCH_CAP
  // overrides for A/B.
  uniforms.MaxBatchWidth = 32.0f;
  if (VolumeMarchVariant() == 9)
  {
    if (const char* cap = getenv("VTK_METAL_TEST_MARCH_CAP"))
    {
      uniforms.MaxBatchWidth = static_cast<float>(std::max(1, std::atoi(cap)));
    }
  }
  // §37.15 block-or-nothing preamble mode: skip the per-cell lattice walk;
  // trust only super/block certification. Opt-in via env for A/B; output is
  // byte-identical to the default walk (dropped skips cover provably-zero
  // samples at unchanged positions).
  uniforms.MmBlocksOnly = 0.0f;
  if (const char* bo = getenv("VTK_METAL_TEST_MM_BLOCKSONLY"))
  {
    uniforms.MmBlocksOnly = std::atof(bo) != 0.0 ? 1.0f : 0.0f;
  }
  // §37.17 leap-granularity: 2 (default) = super+block leaps.
  uniforms.MmLeapLevel = 2.0f;
  if (const char* ll = getenv("VTK_METAL_TEST_MM_LEAPLEVEL"))
  {
    uniforms.MmLeapLevel = static_cast<float>(std::atoi(ll));
  }
  // §37.18: keep the shader's index math in sync with the built texture.
  uniforms.MmBlockSizeCells = static_cast<float>(VolumeMinMaxBlockSize(this->SampleDistance));
  // §38 TF-adaptive exit: legacy 8-bit latch value unless EXIT_THETA is set
  // (the value only drives comparisons when the fc_exitTheta pipeline is
  // selected, but keep the field always valid).
  uniforms.ExitAlpha = VolumeExitTheta() > 0.0f
    ? VolumeExitTheta()
    : static_cast<float>(1.0 - 1.0 / 255.0);
  // §37.19 warp-coherent skip probe: opt-in A/B.
  uniforms.MmWarpMin = 0.0f;
  if (const char* wm = getenv("VTK_METAL_TEST_MM_WARPMIN"))
  {
    // Value doubles as the minimum warp-wide leap worth acting on
    // (0/absent = feature off; e.g. MM_WARPMIN=4 skips unless >=4).
    uniforms.MmWarpMin = static_cast<float>(std::atoi(wm));
  }
  // Final color window/level (matches OpenGL's in_scale/in_bias, applied in the
  // shader after the ray cast as rgb * scale + bias * alpha).
  if (this->FinalColorWindow != 0.0)
  {
    uniforms.FinalColorScale = static_cast<float>(1.0 / this->FinalColorWindow);
    uniforms.FinalColorBias =
      static_cast<float>(0.5 - this->FinalColorLevel / this->FinalColorWindow);
  }
  else
  {
    uniforms.FinalColorScale = 1.0f;
    uniforms.FinalColorBias = 0.0f;
  }

  // Uniform-grid blanking (ghost arrays): enabled when a blanking texture was
  // uploaded for this frame. Mode mirrors the OpenGL backend
  // (1 = cell blanking, 2 = point blanking, 3 = both).
  uniforms.UseBlanking = this->BlankingTexture ? 1.0f : 0.0f;
  uniforms.BlankingMode = 0.0f;
  if (this->BlankingTexture)
  {
    bool hasCells = (this->BlankingCells != nullptr);
    bool hasPoints = (this->BlankingPoints != nullptr);
    uniforms.BlankingMode =
      (hasCells && hasPoints) ? 3.0f : (hasPoints ? 2.0f : (hasCells ? 1.0f : 0.0f));
  }

  // Gradient-based shading uniforms
  {
    vtkVolumeProperty* property = vol->GetProperty();
    bool shadeOn = property && property->GetShade();
    bool hasGradOp = property && property->HasGradientOpacity();

    uniforms.UseGradientShading = shadeOn ? 1.0f : 0.0f;
    // Gradient opacity applies whenever the property declares it, independent of
    // shading (OpenGL vtkVolumeShaderComposer HasGradientOpacity parity).
    uniforms.UseGradientOpacity = hasGradOp ? 1.0f : 0.0f;
    // Match the OpenGL backend: the property's interpolation type applies to the
    // volume data, transfer-function and gradient-opacity textures
    // (vtkVolumeInputHelper). Defaults to nearest.
    uniforms.UseLinearVolumeInterpolation =
      (property && property->GetInterpolationType() == VTK_LINEAR_INTERPOLATION) ? 1.0f : 0.0f;

    // Gradient step: 1/(dims-1) per axis for central differences, matching the
    // OpenGL backend's CellStep (vtkVolumeTexture.cxx: 1/extent-span), which is
    // applied in adjusted (cellToPoint) texture space. Using 1/dims here would
    // shift the gradient stencil by ~3%, visibly changing specular/diffuse on
    // high-frequency (aliased) data.
    int dims[3];
    input->GetDimensions(dims);
    for (int k = 0; k < 3; ++k)
    {
      uniforms.GradientStep[k] = (dims[k] > 1) ? 1.0f / (dims[k] - 1) : 1.0f;
    }

    // Gradient opacity normalization range. OpenGL's computeGradient uses the
    // range of the component the gradient is computed on: component 0 for the
    // single-input LA/1-comp paths, but the LAST component for dependent RGBA
    // (ComputeLightingDeclaration passes lightingComponent = 3). The gradient
    // itself is still computed on component 0 in the Metal shader, but the
    // normalization range must follow GL so gradient.w saturates at the same
    // boundary magnitude.
    const bool dependentRGBA =
      (this->VolumeNumComponents == 4) && property && !property->GetIndependentComponents();
    const double* gradNormRange =
      dependentRGBA ? this->ComponentScalarRange[3] : this->ScalarRange;
    double scalarRange = gradNormRange[1] - gradNormRange[0];
    if (scalarRange <= 0.0)
      scalarRange = 1.0;
    double cellSpacing[3];
    input->GetSpacing(cellSpacing);
    double avgSpacing =
      (fabs(cellSpacing[0]) + fabs(cellSpacing[1]) + fabs(cellSpacing[2])) / 3.0;
    if (avgSpacing < 1e-10)
      avgSpacing = 1.0;
    uniforms.GradientOpacityMin = 0.0f;
    uniforms.GradientOpacityMax = static_cast<float>(
      (scalarRange * 0.5) / (this->ScalarNormalizationFactor * avgSpacing));

    // Material properties from volume property
    if (property)
    {
      double amb = property->GetAmbient();
      double dif = property->GetDiffuse();
      double spec = property->GetSpecular();
      double power = property->GetSpecularPower();
      uniforms.AmbientColor[0] = uniforms.AmbientColor[1] = uniforms.AmbientColor[2] =
        static_cast<float>(amb);
      uniforms.DiffuseColor[0] = uniforms.DiffuseColor[1] = uniforms.DiffuseColor[2] =
        static_cast<float>(dif);
      uniforms.SpecularColor[0] = uniforms.SpecularColor[1] = uniforms.SpecularColor[2] =
        static_cast<float>(spec);
      uniforms.Shininess = static_cast<float>(power);

      // Per-component materials for the independent multi-component path
      // (OpenGL in_ambient[i]/in_diffuse[i]/in_specular[i]/in_shininess[i]).
      for (int c = 0; c < 4; ++c)
      {
        double ambC = property->GetAmbient(c);
        double difC = property->GetDiffuse(c);
        double speC = property->GetSpecular(c);
        double powC = property->GetSpecularPower(c);
        uniforms.AmbientColorComp[c][0] = uniforms.AmbientColorComp[c][1] =
          uniforms.AmbientColorComp[c][2] = static_cast<float>(ambC);
        uniforms.DiffuseColorComp[c][0] = uniforms.DiffuseColorComp[c][1] =
          uniforms.DiffuseColorComp[c][2] = static_cast<float>(difC);
        uniforms.SpecularColorComp[c][0] = uniforms.SpecularColorComp[c][1] =
          uniforms.SpecularColorComp[c][2] = static_cast<float>(speC);
        uniforms.ShininessComp[c] = static_cast<float>(powC);
      }
    }

    // Light direction: headlight (camera-to-volume direction in volume [0,1] space)
    // The gradient normal points inward (toward increasing scalar), matching the
    // OpenGL convention where normals are negated in the lighting calculation.
    double camDirWorld[3];
    ren->GetActiveCamera()->GetDirectionOfProjection(camDirWorld);
    // Transform to volume-local [0,1] space using inverse model matrix
    double camDirLocal[4] = { camDirWorld[0], camDirWorld[1], camDirWorld[2], 0.0 };
    invModelMatrix->MultiplyPoint(camDirLocal, camDirLocal);
    // Account for anisotropic bounds scaling in texture space:
    // direction in [0,1] space must be divided by bounds size to preserve anisotropy
    camDirLocal[0] /= vb.Size[0];
    camDirLocal[1] /= vb.Size[1];
    camDirLocal[2] /= vb.Size[2];
    // Normalize in volume [0,1] space
    double dirLen = sqrt(camDirLocal[0] * camDirLocal[0] + camDirLocal[1] * camDirLocal[1] +
      camDirLocal[2] * camDirLocal[2]);
    if (dirLen > 1e-10)
    {
      camDirLocal[0] /= dirLen;
      camDirLocal[1] /= dirLen;
      camDirLocal[2] /= dirLen;
    }
    uniforms.LightDirection[0] = static_cast<float>(camDirLocal[0]);
    uniforms.LightDirection[1] = static_cast<float>(camDirLocal[1]);
    uniforms.LightDirection[2] = static_cast<float>(camDirLocal[2]);
  }

  // Cropping regions
  if (this->GetCropping())
  {
    uniforms.UseCropping = 1.0f;

    double croppingRegionPlanes[6];
    this->GetCroppingRegionPlanes(croppingRegionPlanes);

    // Clamp to loaded bounds (same as OpenGL mapper)
    for (int i = 0; i < 3; ++i)
    {
      int minIdx = i * 2;
      int maxIdx = i * 2 + 1;
      croppingRegionPlanes[minIdx] =
        std::max(croppingRegionPlanes[minIdx], this->ModelBounds[minIdx]);
      croppingRegionPlanes[minIdx] =
        std::min(croppingRegionPlanes[minIdx], this->ModelBounds[maxIdx]);
      croppingRegionPlanes[maxIdx] =
        std::max(croppingRegionPlanes[maxIdx], this->ModelBounds[minIdx]);
      croppingRegionPlanes[maxIdx] =
        std::min(croppingRegionPlanes[maxIdx], this->ModelBounds[maxIdx]);
    }

    // Convert from model/data coordinates to volume-local [0,1] space.
    uniforms.CroppingPlanes[0] = NormalizeToVolumeSpace(vb, 0, croppingRegionPlanes[0]);
    uniforms.CroppingPlanes[1] = NormalizeToVolumeSpace(vb, 0, croppingRegionPlanes[1]);
    uniforms.CroppingPlanes[2] = NormalizeToVolumeSpace(vb, 1, croppingRegionPlanes[2]);
    uniforms.CroppingPlanes[3] = NormalizeToVolumeSpace(vb, 1, croppingRegionPlanes[3]);
    uniforms.CroppingPlanes2[0] = NormalizeToVolumeSpace(vb, 2, croppingRegionPlanes[4]);
    uniforms.CroppingPlanes2[1] = NormalizeToVolumeSpace(vb, 2, croppingRegionPlanes[5]);
    uniforms.CroppingPlanes2[2] = 0.0f;
    uniforms.CroppingPlanes2[3] = 0.0f;

    uniforms.CroppingBitmask = static_cast<uint32_t>(this->GetCroppingRegionFlags());
  }
  else
  {
    uniforms.UseCropping = 0.0f;
  }

  // Clipping planes
  this->SetClippingPlaneUniforms(&uniforms, ren, vol, modelMatrix, invModelMatrix);

  // Mask / label map
  this->SetMaskUniforms(&uniforms, vol);

  // AverageIP scalar range. The average range is pre-divided by the volume
  // normalization factor so the shader compares it against scalarMin +
  // (scalarMax - scalarMin) * scalarNorm in the same normalized-by-normFactor
  // units as the ScalarMin/Max uniforms.
  {
    double avgRange[2];
    this->GetAverageIPScalarRange(avgRange);
    if (avgRange[1] < avgRange[0])
    {
      double tmp = avgRange[1];
      avgRange[1] = avgRange[0];
      avgRange[0] = tmp;
    }
    uniforms.AverageIPRangeMin =
      static_cast<float>(avgRange[0] / this->ScalarNormalizationFactor);
    uniforms.AverageIPRangeMax =
      static_cast<float>(avgRange[1] / this->ScalarNormalizationFactor);
  }

  // Volume light uniforms for multi-light shading
  VolumeLightUniforms lightUniforms = {};
  {
    double bs[3] = {
      vb.Size[0], vb.Size[1], vb.Size[2]
    };
    this->BuildVolumeLightUniforms(ren, vol, invModelMatrix, this->ModelBounds, bs, lightUniforms);
  }
  uint32_t lightingFeatureBits = VolumeLightingFeatureBits(lightUniforms);

  // Capture the scene depth texture for early ray termination.
  // The depth buffer is written by opaque geometry in the earlier render pass.
  // When MSAA is active, the depth texture is multisampled and cannot be sampled
  // directly by a shader — disable depth occlusion in that case.
  int sampleCount = metalRenderWindow ? metalRenderWindow->GetEffectiveSampleCount() : 1;
  this->DepthTextureOcclusion = (sampleCount > 1) ? nullptr : metalRenderWindow->GetDepthTexture();

  // Depth texture flag — set to 1 when we have a real scene depth texture.
  // RenderToImage casts against an empty scene, so occlusion is disabled there.
  uniforms.UseDepthTexture = (this->DepthTextureOcclusion && !this->RenderToImage) ? 1.0f : 0.0f;

  // Min-max acceleration texture
  uniforms.UseMinMaxAccel = this->MinMaxTexture ? 1.0f : 0.0f;
  uniforms.MinMaxDimX = static_cast<float>(this->MinMaxDims[0]);
  uniforms.MinMaxDimY = static_cast<float>(this->MinMaxDims[1]);
  uniforms.MinMaxDimZ = static_cast<float>(this->MinMaxDims[2]);

  // RenderToImage mode — depth image export. Scene-depth occlusion is
  // intentionally disabled so the volume is cast against an empty scene.
  uniforms.UseRenderToImage = this->RenderToImage ? 1.0f : 0.0f;
  uniforms.ClampDepthToBackface = this->ClampDepthToBackface ? 1.0f : 0.0f;

  // 2D transfer function mode — sample the primary scalar against the second
  // axis in the 2D lookup image. The second axis is either the Y-axis scalar
  // array (yNorm = yRaw * scale + bias) or, when no array is set, the gradient
  // magnitude computed in the shader (OpenGL parity).
  bool tf2dActive = (this->Transfer2DEnabled && this->Transfer2DTexture);
  uniforms.UseTransfer2D = tf2dActive ? 1.0f : 0.0f;
  uniforms.Transfer2DUseGradient = this->Transfer2DUseGradient ? 1.0f : 0.0f;
  if (tf2dActive)
  {
    double r0 = this->Transfer2DYAxisRange[0];
    double r1 = this->Transfer2DYAxisRange[1];
    double span = r1 - r0;
    if (span <= 0.0)
      span = 1.0;
    uniforms.Transfer2DYAxisScale = static_cast<float>(1.0 / span);
    uniforms.Transfer2DYAxisBias = static_cast<float>(-r0 / span);
  }

  // Precomputed gradient normal texture (Phase 4)
  bool hasNormalTexture = (this->GradientNormalTexture != nullptr);
  uniforms.UseNormalTexture = hasNormalTexture ? 1.0f : 0.0f;

  // Shade with the opacity-field gradient instead of the scalar gradient
  // (OpenGL vtkVolumeMapper::GetComputeNormalFromOpacity parity).
  uniforms.UseComputeNormalFromOpacity = this->ComputeNormalFromOpacity ? 1.0f : 0.0f;

  // Build feature mask for shader function constant specialization.
  int featureMask = 0;
  if (uniforms.UseGradientShading > 0.5f)
    featureMask |= VolumeFeature_Shading;
  if (uniforms.UseGradientOpacity > 0.5f)
    featureMask |= VolumeFeature_GradientOpacity;
  if (uniforms.UseMask > 0.5f)
    featureMask |= VolumeFeature_Mask;
  if (uniforms.UseMinMaxAccel > 0.5f)
    featureMask |= VolumeFeature_MinMax;
  if (hasNormalTexture)
    featureMask |= VolumeFeature_NormalTexture;
  if (uniforms.UseComputeNormalFromOpacity > 0.5f)
    featureMask |= VolumeFeature_ComputeNormalFromOpacity;
  if (uniforms.UseLinearVolumeInterpolation > 0.5f)
    featureMask |= VolumeFeature_LinearInterpolation;
  featureMask |= BlendModeToFeatureFlag(this->GetBlendMode());
  if (VolumeFeatureIndependentPath(uniforms, featureMask))
    featureMask |= VolumeFeature_IndependentComponents;
  if (uniforms.UseTransfer2D > 0.5f)
    featureMask |= VolumeFeature_Transfer2D;
  if (this->RectilinearInput)
    featureMask |= VolumeFeature_Rectilinear;
  if (uniforms.UseDependentRGBA > 0.5f)
    featureMask |= VolumeFeature_DependentRGBA;
  if (uniforms.UseDependentLA > 0.5f)
    featureMask |= VolumeFeature_DependentLA;
  featureMask |= lightingFeatureBits;
  if (this->RenderToImage)
    featureMask |= VolumeFeature_RenderToImage;
  if (uniforms.UseCropping > 0.5f)
    featureMask |= VolumeFeature_Cropping;
  if (uniforms.UseBlanking > 0.5f)
    featureMask |= VolumeFeature_Blanking;

  // March-experiment selector (VTK_METAL_TEST_MARCH_VARIANT): encoded into the
  // feature mask so each experiment gets its own specialized pipeline.
  if (const int marchVariant = VolumeMarchVariant(); marchVariant != 0)
  {
    featureMask |=
      (static_cast<uint32_t>(marchVariant) & 0xFu) << VolumeFeature_MarchVariantShift;
  }

  // V31 back-edge exit experiment (VTK_METAL_TEST_DOEXIT): dedicated feature
  // bit so the reshaped march gets its own specialized pipeline.
  if (VolumeMarchDoExit())
  {
    featureMask |= VolumeFeature_MarchDoExit;
  }

  // RG8 pair-packed slice representation experiment (VTK_METAL_TEST_RG8):
  // dedicated feature bit so the pair-tap sampler gets its own pipeline.
  if (VolumeRg8PairActive())
  {
    featureMask |= VolumeFeature_VolRg8;
  }

  // Transposed volume representation (VTK_METAL_TEST_VOLTRANSPOSE):
  // dedicated feature bit so the swizzled-fetch march gets its own pipeline.
  // Gated on the POLICY result recorded by this frame's volume upload (not
  // the raw env): when dims make depth already shortest the upload stays
  // identity and the swizzled pipeline must NOT be selected. The orientation
  // itself rides the key's featureMaskExtra (fc_volTransposedY).
  if (this->VolumeTextureAxisDepth != 0)
  {
    featureMask |= VolumeFeature_VolTransposed;
  }

  // Hardware-selection support (vtkHardwareSelector): during a selection render
  // with CELLS field association the volume is ray-cast with the selection
  // pipeline that writes {voxelId, propId, compositeIndex} into the RGBA32Uint
  // picking attachment wherever the ray accumulates opacity — the Metal
  // equivalent of the OpenGL backend's CheckPickingState / PickingActorPassExit.
  // Without this the volume would be invisible to hardware picking and the
  // picker would report whatever polygonal prop lies behind it.
  bool selectionRender = false;
  uint32_t selectionPropId = 0;
  if (vtkHardwareSelector* selector = ren->GetSelector())
  {
    if (selector->GetFieldAssociation() == vtkDataObject::FIELD_ASSOCIATION_CELLS)
    {
      vtkMetalHardwareSelector* metalSelector =
        vtkMetalHardwareSelector::SafeDownCast(selector);
      int propId = metalSelector ? metalSelector->GetPropID(vol) : -1;
      if (propId >= 0)
      {
        selectionRender = true;
        selectionPropId = static_cast<uint32_t>(propId);
        int ext[6];
        input->GetExtent(ext);
        // The selector treats index 0 as "empty space" (the id attachment
        // encodes ids as value + 1), so the maximum cell id is the number of
        // voxels. Mirror the OpenGL backend's EndPicking bookkeeping.
        selector->UpdateMaximumCellId(static_cast<vtkIdType>(ext[1] - ext[0] + 1) *
          (ext[3] - ext[2] + 1) * (ext[5] - ext[4] + 1));
      }
    }
  }
  uniforms.SelectionMode = selectionRender ? 1.0f : 0.0f;
  uniforms.SelectionPropId = selectionPropId;
  uniforms.SelectionCompositeIndex = 0;
  int selDims[3];
  input->GetDimensions(selDims);
  uniforms.SelectionVolumeDimX = static_cast<uint32_t>(selDims[0]);
  uniforms.SelectionVolumeDimY = static_cast<uint32_t>(selDims[1]);
  uniforms.SelectionVolumeDimZ = static_cast<uint32_t>(selDims[2]);

  // Image-space downsampling requires offscreen rendering at reduced resolution.
  // Partitioned volumes no longer force offscreen rendering because grid traversal
  // composites correctly in a single pass.
  // Cinematic writes ImageSampleColorTexture; Phase 3b only composites when useImageSampling is true,
  // so ISD must be honored for cinematic as well (was && !CinematicRendering — made ISD 0.5 a no-op).
  const float imageSampleDist = this->ImageSampleDistance;
  const bool useImageSampling = (imageSampleDist != 1.0f) || this->CinematicRendering;

  // Viewport size for depth texture UV computation in the shader. This must be
  // the tile-cropped renderer rect (GetTiledSizeAndOrigin, matching
  // vtkOpenGLGPUVolumeRayCastMapper.cxx) — the renderer's encoder renders with
  // that rect as the viewport, and the shader divides in.position.xy by it.
  // ren->GetSize() would return the virtual tiled size (vtkWindowToImageFilter),
  // which is larger than the physical drawable and scales the UVs wrong.
  int tileOrigin[2], tileSize[2];
  ren->GetTiledSizeAndOrigin(&tileSize[0], &tileSize[1], &tileOrigin[0], &tileOrigin[1]);
  int renderWidth = tileSize[0];
  int renderHeight = tileSize[1];
  if (useImageSampling)
  {
    renderWidth = std::max(1, static_cast<int>(tileSize[0] / imageSampleDist));
    renderHeight = std::max(1, static_cast<int>(tileSize[1] / imageSampleDist));
  }
  uniforms.ViewportSize[0] = static_cast<float>(renderWidth);
  uniforms.ViewportSize[1] = static_cast<float>(renderHeight);

  // Compute view-projection matrix via generic vtkCamera API.
  // Try the Metal camera first (has a precomputed cached layout), fall back to
  // computing it from vtkCamera::GetViewTransformMatrix /
  // GetProjectionTransformMatrix so the mapper stays functional even if the
  // camera override is not in place.
  vtkMetalCamera* metalCamera = vtkMetalCamera::SafeDownCast(ren->GetActiveCamera());
  if (metalCamera)
  {
    const float* sceneData = static_cast<const float*>(metalCamera->GetCachedSceneTransforms());
    const float* V = sceneData;         // ViewMatrix at offset 0
    const float* P = sceneData + 16;    // ProjectionMatrix at offset 64 (16 floats)
    for (int c = 0; c < 4; ++c)
    {
      for (int r = 0; r < 4; ++r)
      {
        uniforms.ViewProjectionMatrix[c * 4 + r] = P[0 * 4 + r] * V[c * 4 + 0] +
          P[1 * 4 + r] * V[c * 4 + 1] + P[2 * 4 + r] * V[c * 4 + 2] +
          P[3 * 4 + r] * V[c * 4 + 3];
      }
    }
  }
  else
  {
    // Generic fallback: compute VP from vtkCamera matrices.
    // Metal clip-space uses Z in [0,1], so nearz=0, farz=1.
    vtkCamera* cam = ren->GetActiveCamera();
    double aspect = (tileSize[1] > 0) ? static_cast<double>(tileSize[0]) / tileSize[1] : 1.0;
    vtkMatrix4x4* V4 = cam->GetViewTransformMatrix();
    vtkMatrix4x4* P4 = cam->GetProjectionTransformMatrix(aspect, 0.0, 1.0);
    // Compute P*V column-major (Metal convention: column vectors)
    for (int c = 0; c < 4; ++c)
    {
      for (int r = 0; r < 4; ++r)
      {
        float sum = 0.0f;
        for (int k = 0; k < 4; ++k)
          sum += static_cast<float>(P4->GetElement(r, k)) *
                 static_cast<float>(V4->GetElement(k, c));
        uniforms.ViewProjectionMatrix[c * 4 + r] = sum;
      }
    }
  }

  // Compute inverse view-projection matrix for depth buffer occlusion.
  // Used in the fragment shader to unproject depth values to world space.
  // NDCToVolumeMatrix folds that inverse, WorldToVolumeMatrix and the
  // volume-bounds normalize into one transform so the shader does a single
  // matrix-vector multiply per fragment instead of two matrix chains plus a
  // bounds re-normalize.
  {
    simd_float4x4 vpMat;
    memcpy(&vpMat, uniforms.ViewProjectionMatrix, sizeof(vpMat));
    float det = simd::determinant(vpMat);
    if (fabs(det) > 1e-10f)
    {
      simd_float4x4 invVP = simd::inverse(vpMat);
      memcpy(uniforms.InverseViewProjection, &invVP, sizeof(invVP));

      simd_float4x4 w2v;
      memcpy(&w2v, uniforms.WorldToVolumeMatrix, sizeof(w2v));

      // Volume-bounds normalize: v' = (v - boundsMin) / boundsSize, using the
      // same float32 bounds and 1e-6 size clamp the shader applied before.
      simd_float4x4 norm;
      float bsz[3];
      for (int k = 0; k < 3; ++k)
      {
        bsz[k] = std::max(
          uniforms.VolumeBoundsMax[k] - uniforms.VolumeBoundsMin[k], 1e-6f);
      }
      norm.columns[0] = simd_make_float4(1.0f / bsz[0], 0.0f, 0.0f, 0.0f);
      norm.columns[1] = simd_make_float4(0.0f, 1.0f / bsz[1], 0.0f, 0.0f);
      norm.columns[2] = simd_make_float4(0.0f, 0.0f, 1.0f / bsz[2], 0.0f);
      norm.columns[3] = simd_make_float4(
        -uniforms.VolumeBoundsMin[0] / bsz[0],
        -uniforms.VolumeBoundsMin[1] / bsz[1],
        -uniforms.VolumeBoundsMin[2] / bsz[2],
        1.0f);

      simd_float4x4 ndcToVolume = simd_mul(norm, simd_mul(w2v, invVP));
      memcpy(uniforms.NDCToVolumeMatrix, &ndcToVolume, sizeof(ndcToVolume));
    }
    else
    {
      memset(uniforms.InverseViewProjection, 0, sizeof(uniforms.InverseViewProjection));
      memset(uniforms.NDCToVolumeMatrix, 0, sizeof(uniforms.NDCToVolumeMatrix));
    }
  }

  // Rectilinear-grid index-space remap uniforms (OpenGL in_coordsScale /
  // in_coordsBias parity). The fragment shader walks RectCoordsBuffer (buffer 5)
  // only when UseRectilinear is set; for image/cell-data inputs the flag stays 0
  // and sampling uses the uniform-spacing proxy directly.
  uniforms.UseRectilinear = this->RectilinearInput ? 1.0f : 0.0f;
  uniforms.RectCoordsSizes[0] = this->RectCoordsSizes[0];
  uniforms.RectCoordsSizes[1] = this->RectCoordsSizes[1];
  uniforms.RectCoordsSizes[2] = this->RectCoordsSizes[2];
  uniforms.RectCoordsSizes[3] = 0.0f;
  uniforms.RectCoordsScale[0] = this->RectCoordsScale[0];
  uniforms.RectCoordsScale[1] = this->RectCoordsScale[1];
  uniforms.RectCoordsScale[2] = this->RectCoordsScale[2];
  uniforms.RectCoordsScale[3] = 0.0f;
  uniforms.RectCoordsBias[0] = this->RectCoordsBias[0];
  uniforms.RectCoordsBias[1] = this->RectCoordsBias[1];
  uniforms.RectCoordsBias[2] = this->RectCoordsBias[2];
  uniforms.RectCoordsBias[3] = 0.0f;

  // Cinematic uniforms — shaded DVR 1 spp (Samples/Bounces/Denoise reserved, blend is AO sigma)
  {
    vtkVolumeProperty* cprop = vol->GetProperty();
    uniforms.CinematicEnabled = this->CinematicRendering ? 1.0f : 0.0f;
    uniforms.CinematicSamples = static_cast<uint32_t>(std::clamp(this->CinematicSamples, 1, 1024));
    uniforms.CinematicMaxBounces = static_cast<uint32_t>(std::clamp(this->CinematicMaxBounces, 1, 8));
    uniforms.CinematicScatteringAnisotropy = cprop ? cprop->GetScatteringAnisotropy() : 0.0f;
    uniforms.CinematicReach = this->GlobalIlluminationReach;
    uniforms.CinematicBlend = this->VolumetricScatteringBlending;
    uniforms.CinematicDenoise = this->CinematicDenoise;
    if (cprop) {
      float sc[3]; cprop->GetSubsurfaceColor(sc);
      uniforms.SubsurfaceColor[0]=sc[0]; uniforms.SubsurfaceColor[1]=sc[1]; uniforms.SubsurfaceColor[2]=sc[2];
      uniforms.SubsurfaceStrength = cprop->GetSubsurfaceStrength();
    } else {
      uniforms.SubsurfaceColor[0]=1.0f; uniforms.SubsurfaceColor[1]=1.0f; uniforms.SubsurfaceColor[2]=1.0f;
      uniforms.SubsurfaceStrength=0.0f;
    }
    // CinematicMajorantSigma reserved for future Woodcock path (1 spp DVR does not use it)
    // Previous 256-bin histogram every frame was unread in shader (shader uses blend as sigma).
    uniforms.CinematicMajorantSigma = 1e-4f;
    bool cineReset = false;
    double camMTime = ren->GetActiveCamera()->GetMTime();
    double tfMTime = cprop ? cprop->GetMTime() : 0;
    if (camMTime != this->CinematicLastCameraMTime) cineReset=true;
    if (tfMTime != this->CinematicLastTransferMTime) cineReset=true;
    if (this->VolumeUploadTime.GetMTime() != this->CinematicLastVolumeTime.GetMTime()) cineReset=true;
    if (!this->CinematicRendering) cineReset=true;
    if (cineReset) {
      this->CinematicAccumCount = 0;
      this->CinematicAccumValid = false;
      this->CinematicLastCameraMTime = camMTime;
      this->CinematicLastTransferMTime = tfMTime;
      this->CinematicLastVolumeTime = this->VolumeUploadTime;
    }
    if (this->CinematicRendering) {
      this->CinematicAccumCount = std::min<uint32_t>(this->CinematicAccumCount + 1, static_cast<uint32_t>(this->CinematicSamples));
      uniforms.CinematicAccumCount = this->CinematicAccumCount;
      uniforms.CinematicFrameSeed = this->CinematicFrameSeed++;
    } else {
      uniforms.CinematicAccumCount = 0;
      uniforms.CinematicFrameSeed = 0;
    }
    uniforms._padCinematic = 0.0f;
    uniforms._padCinematicEnd[0]=0.0f; uniforms._padCinematicEnd[1]=0.0f; uniforms._padCinematicEnd[2]=0.0f;
    // Do not override user CinematicBlend — shader uses it as AO sigma (max(blend,1.0)).
  }

  // Wait for the uniform buffer slot for this frame to be free
  dispatch_semaphore_wait((dispatch_semaphore_t)this->FrameSemaphore, DISPATCH_TIME_FOREVER);

  // RAII guard: signals the semaphore on scope exit (early return, exception, etc.).
  // Dismiss after the completion handler is installed below.
  struct SemaphoreSignalGuard {
    dispatch_semaphore_t sem;
    bool active = true;
    ~SemaphoreSignalGuard() { if (active && sem) dispatch_semaphore_signal(sem); }
    void dismiss() { active = false; }
  } semGuard{ (__bridge dispatch_semaphore_t)this->FrameSemaphore };

  int bufIdx = this->UniformFrameIndex % 3;
  this->UniformFrameIndex++;

  // Update uniform buffer (now includes viewProjection + inverseViewProjection)
  id<MTLBuffer> uniformBuf = (__bridge id<MTLBuffer>)this->UniformBuffers[bufIdx];
  memcpy([uniformBuf contents], &uniforms, sizeof(uniforms));

  // Investigation: dump every render (not just the first) so the last frame's
  // uniforms — with the final camera — land in /tmp/app_uniforms.bin. The first
  // render of a fresh window runs before the camera is fully established.
  if (getenv("VTK_METAL_TEST_DUMP_UNIFORMS"))
  {
    FILE* f = fopen("/tmp/app_uniforms.bin", "wb");
    fwrite(&uniforms, 1, sizeof(uniforms), f);
    fclose(f);
    PerBlockData pbdTmp = {};
    BuildPerBlockData(pbdTmp, &uniforms);
    f = fopen("/tmp/app_pbd.bin", "wb");
    fwrite(&pbdTmp, 1, sizeof(pbdTmp), f);
    fclose(f);
    f = fopen("/tmp/app_light.bin", "wb");
    fwrite(&lightUniforms, 1, sizeof(lightUniforms), f);
    fclose(f);
  }

  if (getenv("VTK_METAL_TEST_DUMP_UNIFORMS"))
  {
    fprintf(stderr,
      "UNIF cam=%.8f,%.8f,%.8f bmin=%.4f,%.4f,%.4f bmax=%.4f,%.4f,%.4f vp=%.0fx%.0f sd=%.6f\n",
      uniforms.CameraVolumePos[0], uniforms.CameraVolumePos[1], uniforms.CameraVolumePos[2],
      uniforms.VolumeBoundsMin[0], uniforms.VolumeBoundsMin[1], uniforms.VolumeBoundsMin[2],
      uniforms.VolumeBoundsMax[0], uniforms.VolumeBoundsMax[1], uniforms.VolumeBoundsMax[2],
      uniforms.ViewportSize[0], uniforms.ViewportSize[1],
      (float)uniforms.SampleDistanceHalf / 65536.0f);
    fprintf(stderr, "  invVP: ");
    for (int k = 0; k < 16; ++k)
    {
      fprintf(stderr, "%.7g,", uniforms.InverseViewProjection[k]);
    }
    fprintf(stderr, "\n");
  }

  // Phase 6: Determine if camera is inside the volume for the fullscreen ray-cast path.
  bool cameraInside = this->UseFullscreenCameraInside && this->IsCameraInside(ren, vol);

  if (this->RenderToImage)
  {
    // RenderToImage: render the volume into window-sized offscreen color+depth
    // textures. The color and depth are exported later via GetColorImage() /
    // GetDepthImage(). The cleared background is opaque white (1,1,1,0), matching
    // the OpenGL RenderToImage framebuffer setup.
    int rttWidth = std::max(1, renderWidth);
    int rttHeight = std::max(1, renderHeight);

    if (!this->EnsureRTTResources(mtlDevice, rttWidth, rttHeight, this->DepthImageScalarType))
    {
      return;
    }

    // Must end the renderer's active encoder before creating a new one
    id<MTLRenderCommandEncoder> currentEncoder =
      (__bridge id<MTLRenderCommandEncoder>)metalRenderWindow->GetCurrentRenderCommandEncoder();
    if (currentEncoder)
    {
      [currentEncoder endEncoding];
      metalRenderWindow->SetCurrentRenderCommandEncoder(nullptr);
    }

    // ---- §35.14 segment pre-pass (VTK_METAL_TEST_MM_SEG=1) ----
    // Stage 1 rasterizes a per-pixel ray-atlas through the SAME proxy geometry
    // and vertex function as the pass below; stage 2 walks the occupancy
    // lattice once per ray in a compute kernel and records skip gaps; the RTT
    // march consumes them with integer tests (fc_segHop pipelines). Encoded on
    // the same command buffer ahead of the RTT encoder; inter-encoder order is
    // preserved.
    this->SegActiveThisFrame = false;
    const bool segWanted = VolumeSegWanted() &&
        this->UseMinMaxAcceleration && this->UseGPUMinMax && !usePartitions &&
        this->MinMaxTexture != nullptr;
    if (getenv("VTK_METAL_TEST_MM_SEG_DEBUG"))
      fprintf(stderr, "[seg] wanted=%d accel=%d gpuMM=%d partitions=%d mmTex=%p\n",
        (int)VolumeSegWanted(), (int)this->UseMinMaxAcceleration,
        (int)this->UseGPUMinMax, (int)usePartitions, (void*)this->MinMaxTexture);
    if (segWanted &&
        this->EnsureSegResources(mtlDevice, mtlQueue, rttWidth, rttHeight))
    {
      id<MTLBuffer> segCntBuf =
        (__bridge id<MTLBuffer>)this->SegPoolCounterBuffer;
      *(uint32_t*)segCntBuf.contents = 0;
      [segCntBuf didModifyRange:NSMakeRange(0, 4)];

      id<MTLTexture> atlasA = (__bridge id<MTLTexture>)this->SegAtlasATexture;
      id<MTLTexture> atlasB = (__bridge id<MTLTexture>)this->SegAtlasBTexture;
      id<MTLTexture> atlasC = (__bridge id<MTLTexture>)this->SegAtlasCTexture;

      // -- stage 1: ray-atlas raster --
      MTLRenderPassDescriptor* arpd =
        [MTLRenderPassDescriptor renderPassDescriptor];
      arpd.colorAttachments[0].texture = atlasA;
      arpd.colorAttachments[0].loadAction = MTLLoadActionDontCare;
      arpd.colorAttachments[0].storeAction = MTLStoreActionStore;
      arpd.colorAttachments[1].texture = atlasB;
      arpd.colorAttachments[1].loadAction = MTLLoadActionDontCare;
      arpd.colorAttachments[1].storeAction = MTLStoreActionStore;
      arpd.colorAttachments[2].texture = atlasC;
      arpd.colorAttachments[2].loadAction = MTLLoadActionDontCare;
      arpd.colorAttachments[2].storeAction = MTLStoreActionStore;

      id<MTLRenderCommandEncoder> atlasEnc =
        [commandBuffer renderCommandEncoderWithDescriptor:arpd];
      atlasEnc.label = @"VTK Volume Segment RayAtlas";
      MTLViewport amvp;
      amvp.originX = 0; amvp.originY = 0;
      amvp.width = rttWidth; amvp.height = rttHeight;
      amvp.znear = 0.0; amvp.zfar = 1.0;
      [atlasEnc setViewport:amvp];

      void* atlasPso = this->GetOrCreateVolumePipeline(mtlDevice,
        static_cast<uint32_t>(VolumePipelineType::RayAtlas),
        MTLPixelFormatRGBA32Float, MTLPixelFormatInvalid, 1, featureMask);
      if (!atlasPso)
      {
        [atlasEnc endEncoding];
      }
      else
      {
        this->BindEncoderResources(atlasEnc, uniformBuf, atlasPso, false);
        this->DrawBlocks(atlasEnc, uniformBuf, ren, vol, &uniforms,
          invModelMatrix);
        [atlasEnc endEncoding];

        // -- stage 2: segment builder --
        PerBlockData segPbd;
        this->BuildPerBlockData(segPbd, &uniforms);
        // x=width y=height z=poolCapWords w=maxGaps (< shader's gaps[64])
        uint32_t segMeta[4] = { static_cast<uint32_t>(rttWidth),
                                static_cast<uint32_t>(rttHeight),
                                static_cast<uint32_t>(this->SegPoolCapWords),
                                16u };  // maxGaps == builder register capacity
        id<MTLComputeCommandEncoder> segEnc =
          [commandBuffer computeCommandEncoder];
        [segEnc setComputePipelineState:(__bridge id<MTLComputePipelineState>)
          this->SegBuildComputePipeline];
        [segEnc setTexture:atlasA atIndex:0];
        [segEnc setTexture:atlasB atIndex:1];
        [segEnc setTexture:(__bridge id<MTLTexture>)this->MinMaxTexture
                   atIndex:2];
        [segEnc setTexture:atlasC atIndex:3];
        // §38.17: the builder gained synth-input bindings; atlas-input sites
        // pass flags bit0 = 0.
        [segEnc setTexture:(__bridge id<MTLTexture>)this->VolumeTexture atIndex:4];
        [segEnc setTexture:(__bridge id<MTLTexture>)(this->DepthTextureOcclusion
            ? this->DepthTextureOcclusion : this->DummyDepthTexture) atIndex:5];
        [segEnc setTexture:(__bridge id<MTLTexture>)(this->MinMaxBlockTexture
            ? this->MinMaxBlockTexture : this->DummyMinMaxTexture) atIndex:6];
        [segEnc setBuffer:(__bridge id<MTLBuffer>)this->SegIndexBuffer
                   offset:0 atIndex:0];
        [segEnc setBuffer:segCntBuf offset:0 atIndex:1];
        [segEnc setBuffer:(__bridge id<MTLBuffer>)this->SegPoolBuffer
                   offset:0 atIndex:2];
        [segEnc setBytes:&segMeta length:sizeof(segMeta) atIndex:3];
        [segEnc setBytes:&segPbd length:sizeof(segPbd) atIndex:4];
        [segEnc setBuffer:(__bridge id<MTLBuffer>)this->DummyRectCoordsBuffer
                   offset:0 atIndex:7];
        {
          uint32_t buildFlagsRtt = 0u;
          [segEnc setBytes:&buildFlagsRtt length:sizeof(buildFlagsRtt) atIndex:6];
        }
        [segEnc setBuffer:(__bridge id<MTLBuffer>)uniformBuf offset:0 atIndex:5];
        MTLSize segTg = MTLSizeMake(8, 8, 1);
        MTLSize segGroups = MTLSizeMake((rttWidth + 7) / 8, (rttHeight + 7) / 8, 1);
        [segEnc dispatchThreadgroups:segGroups threadsPerThreadgroup:segTg];
        [segEnc endEncoding];

        this->SegActiveThisFrame = true;
        if (getenv("VTK_METAL_TEST_MM_SEG_DEBUG"))
          fprintf(stderr, "[seg] pre-pass encoded %dx%d\n", rttWidth, rttHeight);
      }
    }

    id<MTLTexture> rttColor = (__bridge id<MTLTexture>)this->RTTColorTexture;
    id<MTLTexture> rttDepth = (__bridge id<MTLTexture>)this->RTTDepthTexture;

    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = rttColor;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].clearColor = MTLClearColorMake(1.0, 1.0, 1.0, 0.0);
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.colorAttachments[1].texture = rttDepth;
    rpd.colorAttachments[1].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[1].clearColor = MTLClearColorMake(1.0, 0.0, 0.0, 0.0);
    rpd.colorAttachments[1].storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> rttEnc =
      [commandBuffer renderCommandEncoderWithDescriptor:rpd];
    rttEnc.label = @"VTK Volume RenderToImage";

    MTLViewport mvp;
    mvp.originX = 0; mvp.originY = 0;
    mvp.width = rttWidth; mvp.height = rttHeight;
    mvp.znear = 0.0; mvp.zfar = 1.0;
    [rttEnc setViewport:mvp];

    void* rttPso = this->GetOrCreateVolumePipeline(mtlDevice,
      static_cast<uint32_t>(VolumePipelineType::RenderToImage),
      MTLPixelFormatRGBA8Unorm, MTLPixelFormatInvalid, 1, featureMask);
    if (!rttPso) { [rttEnc endEncoding]; return; }

    [rttEnc setFragmentBytes:&lightUniforms length:sizeof(lightUniforms) atIndex:4];

    this->BindEncoderResources(rttEnc, uniformBuf, rttPso, false);

    // Draw the proxy geometry (clipped convex hull of the volume box) with the
    // RenderToImage fragment shader, mirroring the standard on-screen path.
    this->DrawBlocks(rttEnc, uniformBuf, ren, vol, &uniforms, invModelMatrix);
    [rttEnc endEncoding];
  }
  else if (useImageSampling && !this->CinematicRendering)
  {
    // Image-space downsampling: render to offscreen texture at reduced resolution,
    // then blit to screen. This cuts fragment count by up to 4x at 0.5x scale.
    int fboWidth = std::max(1, static_cast<int>(tileSize[0] / imageSampleDist));
    int fboHeight = std::max(1, static_cast<int>(tileSize[1] / imageSampleDist));

    if (!this->EnsureImageSampleResources(mtlDevice, fboWidth, fboHeight))
    {
      return;
    }

    // Must end the renderer's active encoder before creating a new one
    id<MTLRenderCommandEncoder> currentEncoder =
      (__bridge id<MTLRenderCommandEncoder>)metalRenderWindow->GetCurrentRenderCommandEncoder();
    if (currentEncoder)
    {
      [currentEncoder endEncoding];
      metalRenderWindow->SetCurrentRenderCommandEncoder(nullptr);
    }

    // Render volume to offscreen texture
    id<MTLTexture> offscreenColor =
      (__bridge id<MTLTexture>)this->ImageSampleColorTexture;

    // Prefer grid traversal for partitioned volumes. If resources are
    // unavailable, fall back to rendering the global volume as a single
    // block (slower but avoids a hard failure).
    bool useGridTraversal = (usePartitions && this->VolumeTexture &&
      this->GridTraversalResourcesValid);

    if (useGridTraversal)
    {
      // Single-pass grid traversal: march all bricks along each pixel ray using 3D DDA
      // through the global volume texture. This is the preferred path for partitioned
      // volumes — no per-brick layers, no sorting passes, exact front-to-back per pixel.
      MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
      rpd.colorAttachments[0].texture = offscreenColor;
      rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
      rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
      rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

      id<MTLRenderCommandEncoder> gridEnc =
        [commandBuffer renderCommandEncoderWithDescriptor:rpd];
      gridEnc.label = @"VTK Grid Traversal";

      MTLViewport mvp;
      mvp.originX = 0; mvp.originY = 0;
      mvp.width = fboWidth; mvp.height = fboHeight;
      mvp.znear = 0.0; mvp.zfar = 1.0;
      [gridEnc setViewport:mvp];

      void* gridPso = this->GetOrCreateVolumePipeline(mtlDevice,
        static_cast<uint32_t>(VolumePipelineType::GridTraversalOffscreen),
        MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1, featureMask);
      if (!gridPso) { [gridEnc endEncoding]; return; }
      [gridEnc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)gridPso];

      PerBlockData pbd;
      this->BuildGlobalPerBlockData(pbd, input);

      id<MTLTexture> volTex = (__bridge id<MTLTexture>)this->VolumeTexture;
      id<MTLTexture> mmTex = (__bridge id<MTLTexture>)this->MinMaxTexture;
      id<MTLTexture> normTex = (__bridge id<MTLTexture>)this->GradientNormalTexture;

      this->BindGridTraversalTextures(gridEnc, uniformBuf,
        (__bridge void*)volTex,
        (__bridge void*)mmTex,
        (__bridge void*)normTex,
        false, &pbd, MTLCullModeNone);

      [gridEnc setFragmentBytes:&lightUniforms length:sizeof(lightUniforms) atIndex:4];
      [gridEnc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [gridEnc endEncoding];
    }
    else
    {
      MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
      rpd.colorAttachments[0].texture = offscreenColor;
      rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
      rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
      rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

      id<MTLRenderCommandEncoder> offscreenEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:rpd];
      offscreenEncoder.label = @"VTK Volume ImageSample Offscreen";

      MTLViewport metalViewport;
      metalViewport.originX = 0;
      metalViewport.originY = 0;
      metalViewport.width = fboWidth;
      metalViewport.height = fboHeight;
      metalViewport.znear = 0.0;
      metalViewport.zfar = 1.0;
      [offscreenEncoder setViewport:metalViewport];

      [offscreenEncoder setFragmentBytes:&lightUniforms length:sizeof(lightUniforms) atIndex:4];

      if (cameraInside)
      {
        this->DrawBlocksFullscreen(offscreenEncoder, uniformBuf, ren, vol,
          &uniforms, invModelMatrix, false, lightingFeatureBits);
      }
      else
      {
        void* activePipeline = this->GetOrCreateVolumePipeline(mtlDevice,
          static_cast<uint32_t>(VolumePipelineType::OffscreenLayer),
          MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1, featureMask);
        this->BindEncoderResources(offscreenEncoder, uniformBuf, activePipeline, false);

        this->DrawBlocks(offscreenEncoder, uniformBuf, ren, vol, &uniforms, invModelMatrix);
      }

      [offscreenEncoder endEncoding];
    }
    // Note: The renderer's blit phase (Phase 3b) will blit the offscreen
    // texture to the screen after all volumes are rendered.
  }
  else
  {
    // Standard path: render directly to the current encoder
    id<MTLRenderCommandEncoder> encoder =
      (__bridge id<MTLRenderCommandEncoder>)metalRenderWindow->GetCurrentRenderCommandEncoder();

    if (!encoder)
    {
      return;
    }

    // Prefer grid traversal for partitioned volumes. Falls back to single-block
    // fullscreen or proxy-geometry if grid traversal resources are unavailable.
    bool useGridTraversalDirect = (usePartitions && this->VolumeTexture &&
      this->GridTraversalResourcesValid);

    [encoder setFragmentBytes:&lightUniforms length:sizeof(lightUniforms) atIndex:4];

    if (useGridTraversalDirect)
    {
      void* gridPsoDirect = this->GetOrCreateVolumePipeline(mtlDevice,
        static_cast<uint32_t>(VolumePipelineType::GridTraversalDirect),
        MTLPixelFormatBGRA8Unorm, MTLPixelFormatDepth32Float,
        static_cast<uint32_t>(sampleCount), featureMask);
      if (!gridPsoDirect) { return; }
      [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)gridPsoDirect];

      PerBlockData pbd;
      this->BuildGlobalPerBlockData(pbd, input);

      id<MTLTexture> volTex = (__bridge id<MTLTexture>)this->VolumeTexture;
      id<MTLTexture> mmTex = (__bridge id<MTLTexture>)this->MinMaxTexture;
      id<MTLTexture> normTex = (__bridge id<MTLTexture>)this->GradientNormalTexture;

      this->BindGridTraversalTextures(encoder, uniformBuf,
        (__bridge void*)volTex,
        (__bridge void*)mmTex,
        (__bridge void*)normTex,
        true, &pbd, MTLCullModeNone);

      [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    else if (cameraInside)
    {
      // Fullscreen ray-cast path — no proxy geometry, no vertex/index buffers.
      this->DrawBlocksFullscreen(encoder, uniformBuf, ren, vol,
        &uniforms, invModelMatrix, true, lightingFeatureBits);
    }
    else
    {
      // Composite slab tiling (VTK_METAL_TEST_NUM_SLABS): split the DirectScreen
      // (or selection) pass into front-to-back slab passes. Multi-pass rendering
      // needs each pass to know the far-side composite alpha (the framebuffer
      // [[color(0)]] fetch is unavailable here), so the passes render into two
      // alternating private textures, each sampling the other as its near side.
      // The final texture is exposed through the ImageSample members and blended
      // over the drawable by the renderer's Phase 3b blit. The slab bit only
      // reaches this direct pipeline's mask; offscreen/RTT/grid pipelines keep
      // their unmodified mask and stay single-pass.
      const int resolvedSlabs = ResolveNumSlabs(uniforms.CameraVolumePos);
      const int numSlabs =
        (!selectionRender && this->GetBlendMode() == vtkVolumeMapper::COMPOSITE_BLEND &&
         resolvedSlabs > 1)
          ? resolvedSlabs
          : 1;
      uint32_t directMask = featureMask;
      if (numSlabs > 1)
      {
        directMask |= VolumeFeature_Slab;
      }
      // The ping-pong slab textures are single-sample (EnsureSlabResources),
      // so the slab path must use a 1-sample pipeline regardless of the
      // window's MSAA setting; the drawable's sample count only applies to
      // the single-pass path and to Phase 3b's resolve blit.
      const uint32_t psoSampleCount =
        (numSlabs > 1) ? 1 : static_cast<uint32_t>(sampleCount);
      void* directPso = this->GetOrCreateVolumePipeline(mtlDevice,
        static_cast<uint32_t>(selectionRender
          ? VolumePipelineType::SelectionDirect
          : VolumePipelineType::DirectScreen),
        MTLPixelFormatBGRA8Unorm, MTLPixelFormatDepth32Float,
        psoSampleCount, directMask);
      const int slabOnly = []() -> int {
        if (const char* v = std::getenv("VTK_METAL_TEST_SLAB_ONLY"))
          return std::atoi(v);
        return -1;
      }();

      // Cinematic — shaded DVR 1 spp (single 8x8, no binned, cine_accum is 1 spp fade)
        if (this->CinematicRendering && !selectionRender && !usePartitions && !cameraInside) {
          // End current encoder for compute dispatch (same precedent as slab/compute paths)
          id<MTLRenderCommandEncoder> curEnc = (__bridge id<MTLRenderCommandEncoder>)metalRenderWindow->GetCurrentRenderCommandEncoder();
          if (curEnc) { [curEnc endEncoding]; metalRenderWindow->SetCurrentRenderCommandEncoder(nullptr); }
          PerBlockData cinePbd; this->BuildPerBlockData(cinePbd, &uniforms);
          if (!this->DispatchCinematicCompute((__bridge void*)device, (__bridge void*)queue, (__bridge void*)commandBuffer,
                ren, vol, (__bridge void*)uniformBuf, &cinePbd, &lightUniforms, renderWidth, renderHeight)) {
            // Fallback to single-pass fragment if cinematic dispatch fails
            this->BindEncoderResources(encoder, uniformBuf, directPso, true);
            for (int s = numSlabs - 1; s >= 0; --s) { this->DrawBlocks(encoder, uniformBuf, ren, vol, &uniforms, invModelMatrix, s, numSlabs); }
          }
          return;
        } else if (numSlabs > 1)
      {
        // Ping-pong offscreen path. End the renderer's active encoder (the
        // image-sample path precedent; the renderer's Phase 3b re-opens a pass
        // on the drawable for the final blit).
        id<MTLRenderCommandEncoder> currentEncoder =
          (__bridge id<MTLRenderCommandEncoder>)metalRenderWindow->GetCurrentRenderCommandEncoder();
        if (currentEncoder)
        {
          [currentEncoder endEncoding];
          metalRenderWindow->SetCurrentRenderCommandEncoder(nullptr);
        }

        if (!this->EnsureSlabResources(mtlDevice, renderWidth, renderHeight))
        {
          return;
        }
        id<MTLTexture> texA = (__bridge id<MTLTexture>)this->SlabTextureA;
        id<MTLTexture> texB = (__bridge id<MTLTexture>)this->SlabTextureB;
        id<MTLTexture> slabDepth = (__bridge id<MTLTexture>)this->SlabDepthTexture;

        id<MTLTexture> finalTex = nil;
        // Draw the slab passes in RAY order (front-to-back): each pass's
        // feedback texture (the other ping-pong target) was written by the
        // pass covering the ray's next-nearer range, so the inherited composite
        // is the ray's near side — the exact state a single-pass march would
        // have accumulated before this band, which makes the saturation latch
        // and the per-sample weights match the single-pass march (see the
        // slabFar init in the shader). Ray-fraction passes always run
        // front-to-back (their index ranges are in ray space); spatial passes
        // run in the dominant-axis direction of travel, which is +axis when
        // the camera sits on the -axis side of the volume.
        bool raysAscending = true;
        if (VolumeSlabSpatial())
        {
          const int slabAxis = VolumeSlabAxis(uniforms.CameraVolumePos);
          raysAscending =
            (uniforms.CameraVolumePos[slabAxis] <
             0.5f * (uniforms.VolumeBoundsMin[slabAxis] + uniforms.VolumeBoundsMax[slabAxis]));
        }
        const int slabStart = raysAscending ? 0 : numSlabs - 1;
        const int slabStep = raysAscending ? 1 : -1;
        for (int s = slabStart; raysAscending ? s < numSlabs : s >= 0; s += slabStep)
        {
          if (slabOnly >= 0 && s != slabOnly) { continue; }
          // Alternate targets so each pass's feedback texture was written by
          // the pass covering the ray's next-nearer range (the near-side
          // composite).
          id<MTLTexture> target = (s % 2) ? texA : texB;
          id<MTLTexture> feedback = (s % 2) ? texB : texA;
          finalTex = target;

          MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
          rpd.colorAttachments[0].texture = target;
          rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
          rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
          rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
          rpd.depthAttachment.texture = slabDepth;
          rpd.depthAttachment.loadAction = MTLLoadActionClear;
          rpd.depthAttachment.clearDepth = 1.0;
          rpd.depthAttachment.storeAction = MTLStoreActionDontCare;

          id<MTLRenderCommandEncoder> slabEnc =
            [commandBuffer renderCommandEncoderWithDescriptor:rpd];
          slabEnc.label = @"VTK Volume Slab PingPong";

          MTLViewport vp;
          vp.originX = 0; vp.originY = 0;
          vp.width = renderWidth; vp.height = renderHeight;
          vp.znear = 0.0; vp.zfar = 1.0;
          [slabEnc setViewport:vp];

          this->BindEncoderResources(slabEnc, uniformBuf, directPso, true);
          [slabEnc setFragmentTexture:feedback atIndex:15];

          this->DrawBlocks(slabEnc, uniformBuf, ren, vol, &uniforms, invModelMatrix, s, numSlabs);
          [slabEnc endEncoding];
          if (slabOnly >= 0) { break; }
        }

        // Expose the final composite to the renderer's Phase 3b blit pass,
        // which blends it over the drawable with (ONE, ONE_MINUS_SRC_ALPHA).
        AssignRetainedMetalObject(this->ImageSampleColorTexture, finalTex);
        this->ImageSampleFBOWidth = renderWidth;
        this->ImageSampleFBOHeight = renderHeight;
      }
      else if (!selectionRender &&
               this->GetBlendMode() == vtkVolumeMapper::COMPOSITE_BLEND &&
               VolumeComputeMarchWanted() && !usePartitions)
      {
        // §38.6 / §36.4 Design B — Compute Marcher / Ray-Binned Marching
        id<MTLRenderCommandEncoder> curEnc =
          (__bridge id<MTLRenderCommandEncoder>)
            metalRenderWindow->GetCurrentRenderCommandEncoder();
        if (curEnc)
        {
          [curEnc endEncoding];
          metalRenderWindow->SetCurrentRenderCommandEncoder(nullptr);
        }

        if (!this->EnsureComputeMarchResources(mtlDevice, mtlQueue,
              renderWidth, renderHeight))
        {
          return;
        }

        id<MTLTexture> atlasA = (__bridge id<MTLTexture>)this->SegAtlasATexture;
        id<MTLTexture> atlasB = (__bridge id<MTLTexture>)this->SegAtlasBTexture;
        id<MTLTexture> atlasC = (__bridge id<MTLTexture>)this->SegAtlasCTexture;
        // §38.17: target captured AFTER the segment-build block below —
        // EnsureSegResources may recreate SegMarchTexture (the pre-capture
        // handle could be a stale texture without ShaderWrite usage, making
        // the compute march's writes undefined on frame 1).
        id<MTLTexture> target = nullptr;

        // -- stage 1: ray-atlas raster through volume proxy geometry --
        // CLEAR, not DontCare: the compute marcher runs one thread per screen
        // pixel and trusts the atlas blindly. Texels outside the proxy
        // footprint must read back as zeros (steps<=0 => immediate exit) or a
        // garbage A.w / NaN evalStep combination marches forever and wedges
        // the command buffer (waitUntilCompleted never returns).
        if (!(VolumeComputeMarchNoAtlas() || VolumeComputeMarchSynth()))
        {
        MTLRenderPassDescriptor* arpd =
          [MTLRenderPassDescriptor renderPassDescriptor];
        arpd.colorAttachments[0].texture = atlasA;
        arpd.colorAttachments[0].loadAction = MTLLoadActionClear;
        arpd.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
        arpd.colorAttachments[0].storeAction = MTLStoreActionStore;
        arpd.colorAttachments[1].texture = atlasB;
        arpd.colorAttachments[1].loadAction = MTLLoadActionClear;
        arpd.colorAttachments[1].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
        arpd.colorAttachments[1].storeAction = MTLStoreActionStore;
        arpd.colorAttachments[2].texture = atlasC;
        arpd.colorAttachments[2].loadAction = MTLLoadActionClear;
        arpd.colorAttachments[2].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
        arpd.colorAttachments[2].storeAction = MTLStoreActionStore;

        id<MTLRenderCommandEncoder> atlasEnc =
          [commandBuffer renderCommandEncoderWithDescriptor:arpd];
        atlasEnc.label = @"VTK Volume Compute RayAtlas";
        MTLViewport amvp;
        amvp.originX = 0; amvp.originY = 0;
        amvp.width = renderWidth; amvp.height = renderHeight;
        amvp.znear = 0.0; amvp.zfar = 1.0;
        [atlasEnc setViewport:amvp];

        void* atlasPso = this->GetOrCreateVolumePipeline(mtlDevice,
          static_cast<uint32_t>(VolumePipelineType::RayAtlas),
          MTLPixelFormatRGBA32Float, MTLPixelFormatInvalid, 1, featureMask);
        if (!atlasPso)
        {
          [atlasEnc endEncoding];
          return;
        }
        this->BindEncoderResources(atlasEnc, uniformBuf, atlasPso, false);
        this->DrawBlocks(atlasEnc, uniformBuf, ren, vol, &uniforms,
          invModelMatrix);
        [atlasEnc endEncoding];
        } // !VolumeComputeMarchNoAtlas

        PerBlockData compPbd;
        this->BuildPerBlockData(compPbd, &uniforms);

        id<MTLBuffer> rectCoordsBuf =
          this->RectCoordsBuffer ? (__bridge id<MTLBuffer>)this->RectCoordsBuffer
                                 : (__bridge id<MTLBuffer>)this->DummyRectCoordsBuffer;

        // §38.17 MM_SEG_DEBUG: stage offset of the target-readback region
        // (past the 64 B/ray builder mirror).
        const NSUInteger segImgOff =
          static_cast<NSUInteger>(renderWidth) * renderHeight * 64;
        // §38.17: segment build for the compute marcher. Runs the SAME
        // builder kernel as the fragment RTT path, but with synth input when
        // CM_SYNTH is active (no atlas raster exists to read). Ordering:
        // summaries built earlier on this buffer; the march encoders below
        // consume the pool only after this encoder ends.
        const bool cmSeg =
          VolumeSegWanted() && !VolumeSegConsumeSuppressed() &&
          !usePartitions && this->MinMaxTexture != nullptr &&
          (VolumeComputeMarchSynth() ||
           (!VolumeComputeMarchNoAtlas() && !VolumeComputeMarchNoMarch()));
        bool segLive = false;
        // MM_SEG_NOBUILD: bind real buffers but skip the builder dispatch
        // (isolation: consume reads stale records; image must still show).
        const bool segNoBuild =
          getenv("VTK_METAL_TEST_MM_SEG_NOBUILD") != nullptr;
        if (cmSeg &&
            this->EnsureSegResources(mtlDevice, mtlQueue,
                                     renderWidth, renderHeight))
        {
          // §38.17 per-camera cache: static views reuse the previous pool.
          // Bench minMaxTime jitters every other frame (7211->9475) even
          // with static volume — ignore it for now; width/height/cap is
          // sufficient for the static bench to amortize.
          if (getenv("VTK_METAL_TEST_MM_SEG_DEBUG"))
            fprintf(stderr, "[segcm] check %p valid=%d cache %dx%d cap %zu minMax %lu vs %dx%d cap %zu minMax %lu\n",
                    (void*)this, (int)this->SegCacheValid, this->SegCacheWidth, this->SegCacheHeight,
                    this->SegCachePoolCapWords,
                    (unsigned long)(this->SegCacheValid ? this->SegCacheMinMaxTime.GetMTime() : 0),
                    renderWidth, renderHeight, this->SegPoolCapWords,
                    (unsigned long)this->MinMaxUploadTime.GetMTime());
          bool segCacheHit = false;
          if (this->SegCacheValid && this->SegCacheWidth == renderWidth &&
              this->SegCacheHeight == renderHeight &&
              this->SegCachePoolCapWords == this->SegPoolCapWords)
          {
            segCacheHit = true;
          }
          if (segCacheHit)
          {
            this->SegActiveThisFrame = true;
            segLive = true;
            if (getenv("VTK_METAL_TEST_MM_SEG_DEBUG"))
              fprintf(stderr, "[segcm] cache HIT (%dx%d)\n",
                      renderWidth, renderHeight);
          }
          else
          {
            if (getenv("VTK_METAL_TEST_MM_SEG_DEBUG") && this->SegCacheValid)
            {
              if (this->SegCacheWidth != renderWidth ||
                  this->SegCacheHeight != renderHeight)
                fprintf(stderr, "[segcm] cache MISS: dims %dx%d vs %dx%d\n",
                        this->SegCacheWidth, this->SegCacheHeight,
                        renderWidth, renderHeight);
              else if (this->SegCachePoolCapWords != this->SegPoolCapWords)
                fprintf(stderr, "[segcm] cache MISS: cap %zu vs %zu\n",
                        this->SegCachePoolCapWords, this->SegPoolCapWords);
              else if (this->SegCacheMinMaxTime.GetMTime() !=
                       this->MinMaxUploadTime.GetMTime())
                fprintf(stderr, "[segcm] cache MISS: minMaxTime %lu vs %lu\n",
                        (unsigned long)this->SegCacheMinMaxTime.GetMTime(),
                        (unsigned long)this->MinMaxUploadTime.GetMTime());
              else if (this->SegCacheUniformBytes.size() != sizeof(uniforms))
                fprintf(stderr, "[segcm] cache MISS: uniform size %zu vs %zu\n",
                        this->SegCacheUniformBytes.size(), sizeof(uniforms));
              else if (this->SegCachePbdBytes.size() != sizeof(compPbd))
                fprintf(stderr, "[segcm] cache MISS: pbd size %zu vs %zu\n",
                        this->SegCachePbdBytes.size(), sizeof(compPbd));
              else if (memcmp(this->SegCacheUniformBytes.data(), &uniforms,
                              sizeof(uniforms)) != 0)
              {
                size_t first = 0;
                const uint8_t* a = this->SegCacheUniformBytes.data();
                const uint8_t* b = reinterpret_cast<const uint8_t*>(&uniforms);
                for (size_t k = 0; k < sizeof(uniforms); ++k)
                  if (a[k] != b[k]) { first = k; break; }
                fprintf(stderr,
                        "[segcm] cache MISS: uniform diff at byte %zu "
                        "(%02x vs %02x)\n",
                        first, a[first], b[first]);
              }
              else if (memcmp(this->SegCachePbdBytes.data(), &compPbd,
                              sizeof(compPbd)) != 0)
              {
                size_t first = 0;
                const uint8_t* a = this->SegCachePbdBytes.data();
                const uint8_t* b = reinterpret_cast<const uint8_t*>(&compPbd);
                for (size_t k = 0; k < sizeof(compPbd); ++k)
                  if (a[k] != b[k]) { first = k; break; }
                fprintf(stderr,
                        "[segcm] cache MISS: pbd diff at byte %zu "
                        "(%02x vs %02x)\n",
                        first, a[first], b[first]);
              }
            }
          id<MTLBuffer> segCntBuf =
            (__bridge id<MTLBuffer>)this->SegPoolCounterBuffer;
          *(uint32_t*)segCntBuf.contents = 0;  // Shared: no didModifyRange.

          uint32_t segMeta[4] = { static_cast<uint32_t>(renderWidth),
                                  static_cast<uint32_t>(renderHeight),
                                  static_cast<NSUInteger>(this->SegPoolCapWords)
                                    ? static_cast<uint32_t>(this->SegPoolCapWords)
                                    : 0u,
                                  16u };
          id<MTLComputeCommandEncoder> segEnc =
            segNoBuild ? nil : [commandBuffer computeCommandEncoder];
          segEnc.label = @"VTK Volume Segment Build (Compute)";
          [segEnc setComputePipelineState:(__bridge id<MTLComputePipelineState>)
            this->SegBuildComputePipeline];
          [segEnc setTexture:(__bridge id<MTLTexture>)this->SegAtlasATexture atIndex:0];
          [segEnc setTexture:(__bridge id<MTLTexture>)this->SegAtlasBTexture atIndex:1];
          [segEnc setTexture:(__bridge id<MTLTexture>)this->MinMaxTexture atIndex:2];
          [segEnc setTexture:(__bridge id<MTLTexture>)this->SegAtlasCTexture atIndex:3];
          [segEnc setTexture:(__bridge id<MTLTexture>)this->VolumeTexture atIndex:4];
          [segEnc setTexture:(__bridge id<MTLTexture>)(this->DepthTextureOcclusion
              ? this->DepthTextureOcclusion : this->DummyDepthTexture)
                       atIndex:5];
          [segEnc setTexture:(__bridge id<MTLTexture>)(this->MinMaxBlockTexture
              ? this->MinMaxBlockTexture : this->DummyMinMaxTexture) atIndex:6];
          [segEnc setBuffer:(__bridge id<MTLBuffer>)this->SegIndexBuffer offset:0 atIndex:0];
          [segEnc setBuffer:segCntBuf offset:0 atIndex:1];
          [segEnc setBuffer:(__bridge id<MTLBuffer>)this->SegPoolBuffer offset:0 atIndex:2];
          [segEnc setBytes:&segMeta length:sizeof(segMeta) atIndex:3];
          [segEnc setBytes:&compPbd length:sizeof(compPbd) atIndex:4];
          [segEnc setBuffer:(__bridge id<MTLBuffer>)uniformBuf offset:0 atIndex:5];
          uint32_t buildFlags = VolumeComputeMarchSynth() ? 1u : 0u;
          // Stage-2 builder block leaps (VTK_METAL_TEST_MM_SEG_LEAPS):
          // parity-divergent vs the march preamble (162k px @512², cause
          // unresolved — §38.17); default off keeps stage-1 fine-only gaps.
          if (getenv("VTK_METAL_TEST_MM_SEG_LEAPS") != nullptr &&
              this->MinMaxBlockTexture != nullptr &&
              this->MinMaxBlockSize > 0)
            buildFlags |= 2u;
          if (getenv("VTK_METAL_TEST_MM_SEG_DEBUG"))
          {
            buildFlags |= 2u;
            if (!this->SegDebugStageBuffer ||
                this->SegDebugStageBytes <
                  static_cast<size_t>(renderWidth) * renderHeight * 72)
            {
              // 64 B/ray mirror + one RGBA16Float frame for target readback.
              const size_t dbgBytes =
                static_cast<size_t>(renderWidth) * renderHeight * 72;
              id<MTLBuffer> st = [device
                newBufferWithLength:dbgBytes
                options:MTLResourceStorageModeShared];
              AssignMetalObject(this->SegDebugStageBuffer, st);
              this->SegDebugStageBytes = dbgBytes;
            }
            [segEnc setBuffer:(__bridge id<MTLBuffer>)this->SegDebugStageBuffer
                       offset:0 atIndex:7];
          }
          else
          {
            [segEnc setBuffer:(__bridge id<MTLBuffer>)this->DummyRectCoordsBuffer
                       offset:0 atIndex:7];
          }
          [segEnc setBytes:&buildFlags length:sizeof(buildFlags) atIndex:6];
          MTLSize segTg = MTLSizeMake(8, 8, 1);
          MTLSize segGroups = MTLSizeMake((renderWidth + 7) / 8,
                                          (renderHeight + 7) / 8, 1);
           if (!segNoBuild)
          {
            [segEnc dispatchThreadgroups:segGroups
                 threadsPerThreadgroup:segTg];
          }
          [segEnc endEncoding];
           if (!segNoBuild)
          {
            this->SegCacheValid = true;
            this->SegCacheWidth = renderWidth;
            this->SegCacheHeight = renderHeight;
            this->SegCacheUniformBytes.assign(
              reinterpret_cast<const uint8_t*>(&uniforms),
              reinterpret_cast<const uint8_t*>(&uniforms) + sizeof(uniforms));
            this->SegCachePbdBytes.assign(
              reinterpret_cast<const uint8_t*>(&compPbd),
              reinterpret_cast<const uint8_t*>(&compPbd) + sizeof(compPbd));
            this->SegCachePoolCapWords = this->SegPoolCapWords;
            this->SegCacheMinMaxTime = this->MinMaxUploadTime;
          }
          this->SegActiveThisFrame = true;
          segLive = true;
          if (getenv("VTK_METAL_TEST_MM_SEG_DEBUG"))
          {
            // Builder mirror is shared storage: printed at CB completion.
            __weak vtkMetalGPUVolumeRayCastMapper* weakThis = this;
            id<MTLBuffer> dbgCnt = segCntBuf;
            id<MTLBuffer> dbgStage =
              (__bridge id<MTLBuffer>)this->SegDebugStageBuffer;
            [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
              vtkMetalGPUVolumeRayCastMapper* m = weakThis;
              if (!m) return;
              const uint32_t claims =
                *static_cast<const uint32_t*>(dbgCnt.contents);
              const uint32_t* mir =
                static_cast<const uint32_t*>(dbgStage.contents);
              const uint32_t fpc =
                (m->SegAtlasHeight / 2) * m->SegAtlasWidth
                  + m->SegAtlasWidth / 2;
              const uint32_t* mc = mir + fpc * 16u;
              // Consume-state outlet is its own 16-word shared buffer.
              const uint32_t* cons = static_cast<const uint32_t*>(
                ((__bridge id<MTLBuffer>)m->SegConsumeDbgBuffer).contents);
              // March-target readback lives past the mirror region.
              const uint16_t* img = reinterpret_cast<const uint16_t*>(
                dbgStage.contents) +
                (size_t(m->SegAtlasWidth) * m->SegAtlasHeight * 64) / 2;
              double acc = 0.0; uint16_t mx = 0;
              const size_t npix = size_t(m->SegAtlasWidth) * m->SegAtlasHeight;
              for (size_t k = 0; k < npix * 4; ++k)
              {
                acc += img[k];
                if (img[k] > mx) mx = img[k];
              }
              fprintf(stderr,
                "[segcm] claims=%u cap=%u | center: off=%u cnt=%u "
                "steps=%u guard=%u | g0=%#x g1=%#x g2=%#x g3=%#x g4=%#x "
                "g5=%#x g6=%#x g7=%#x | epZ=%#x esZ=%#x ss=%#x | "
                "consume: i=%u idx=%u cnt=%u opac=%#x | target mean=%.*f "
                "max=%u | useBlocks/bsI=%#x\n",
                claims, m->SegPoolCapWords,
                mc[0], mc[1], mc[2], mc[3],
                mc[4], mc[5], mc[6], mc[7], mc[8], mc[9], mc[10], mc[11],
                mc[12], mc[13], mc[14],
                cons[0], cons[1], cons[2], cons[3],
                3, acc / (double(npix) * 4.0), mx, mc[15]);
            }];
          }
          } // else: cache miss — built this frame
        }

        // §38.17: capture the (possibly recreated) march target now.
        target = (__bridge id<MTLTexture>)this->SegMarchTexture;

        // RAY_BINNED requires the atlas raster (classify reads atlasA);
        // synth mode has no atlas, so binning falls back to the 2D marcher.
        const bool binned = VolumeRayBinnedWanted() && !VolumeComputeMarchSynth();
        if (binned && !VolumeComputeMarchNoMarch())
        {
          // -- stage 2a: ray bin classification --
          id<MTLBuffer> binCntBuf = (__bridge id<MTLBuffer>)this->RayBinCountersBuffer;
          std::memset(binCntBuf.contents, 0, 16 * sizeof(uint32_t));
          [binCntBuf didModifyRange:NSMakeRange(0, 16 * sizeof(uint32_t))];

          const uint32_t numBins = 4u;
          const uint32_t binCap = static_cast<uint32_t>(renderWidth * renderHeight);
          uint32_t binMeta[4] = { static_cast<uint32_t>(renderWidth),
                                  static_cast<uint32_t>(renderHeight),
                                  binCap,
                                  numBins };

          if (!this->RayBinClassifyPipeline)
          {
            id<MTLLibrary> library = (__bridge id<MTLLibrary>)this->CachedShaderLibrary;
            id<MTLFunction> fn = [library newFunctionWithName:@"volume_ray_bin_classify"];
            if (fn)
            {
              NSError* err = nil;
              id<MTLComputePipelineState> cps = [mtlDevice newComputePipelineStateWithFunction:fn error:&err];
              [fn release];
              if (cps)
              {
                AssignMetalObject(this->RayBinClassifyPipeline, cps);
              }
            }
          }

          if (this->RayBinClassifyPipeline)
          {
            id<MTLComputeCommandEncoder> classEnc = [commandBuffer computeCommandEncoder];
            classEnc.label = @"VTK Volume RayBin Classify";
            [classEnc setComputePipelineState:(__bridge id<MTLComputePipelineState>)this->RayBinClassifyPipeline];
            [classEnc setTexture:atlasA atIndex:0];
            [classEnc setBuffer:binCntBuf offset:0 atIndex:0];
            [classEnc setBuffer:(__bridge id<MTLBuffer>)this->RayBinIndicesBuffer offset:0 atIndex:1];
            [classEnc setBytes:&binMeta length:sizeof(binMeta) atIndex:2];
            MTLSize classTg = MTLSizeMake(16, 16, 1);
            MTLSize classGroups = MTLSizeMake((renderWidth + 15) / 16, (renderHeight + 15) / 16, 1);
            [classEnc dispatchThreadgroups:classGroups threadsPerThreadgroup:classTg];
            [classEnc endEncoding];
          }

          // -- stage 2b: binned compute marcher --
          void* marchBinnedPso = this->GetOrCreateComputeMarchPipeline(mtlDevice, featureMask, true);
          if (marchBinnedPso)
          {
            // Only classified rays are written back, so define the rest of
            // the target first (its contents persist across frames).
            MTLRenderPassDescriptor* crpd =
              [MTLRenderPassDescriptor renderPassDescriptor];
            crpd.colorAttachments[0].texture = target;
            crpd.colorAttachments[0].loadAction = MTLLoadActionClear;
            crpd.colorAttachments[0].clearColor =
              MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
            crpd.colorAttachments[0].storeAction = MTLStoreActionStore;
            id<MTLRenderCommandEncoder> clearEnc =
              [commandBuffer renderCommandEncoderWithDescriptor:crpd];
            [clearEnc endEncoding];

            id<MTLComputeCommandEncoder> marchEnc = [commandBuffer computeCommandEncoder];
            marchEnc.label = @"VTK Volume ComputeMarch Binned";
            [marchEnc setComputePipelineState:(__bridge id<MTLComputePipelineState>)marchBinnedPso];

            this->BindComputeMarchTextures((__bridge void*)marchEnc, (__bridge void*)atlasA,
              (__bridge void*)atlasB, (__bridge void*)atlasC, (__bridge void*)target);

            [marchEnc setBuffer:(__bridge id<MTLBuffer>)this->RayBinIndicesBuffer offset:0 atIndex:0];
            // Live per-bin counts (buffer 5): threads beyond this frame's
            // count must exit without marching — unwritten index slots hold
            // stale UVs from previous frames (private storage persists).
            [marchEnc setBuffer:binCntBuf offset:0 atIndex:5];
            [marchEnc setBuffer:(__bridge id<MTLBuffer>)uniformBuf offset:0 atIndex:1];
            [marchEnc setBytes:&compPbd length:sizeof(compPbd) atIndex:2];
            [marchEnc setBuffer:rectCoordsBuf offset:0 atIndex:3];
            [marchEnc setBytes:&lightUniforms length:sizeof(lightUniforms) atIndex:4];
            // §38.17: segment consume inputs (slots 8/9) for the binned path.
            [marchEnc setBuffer:(__bridge id<MTLBuffer>)(segLive
                ? this->SegIndexBuffer : this->DummyRectCoordsBuffer) offset:0 atIndex:8];
            [marchEnc setBuffer:(__bridge id<MTLBuffer>)(segLive
                ? this->SegPoolBuffer : this->DummyRectCoordsBuffer) offset:0 atIndex:9];

            ComputeMarchControl cmc{ VolumeComputeMarchFloor() ? 1u : 0u,
                                     static_cast<uint32_t>(VolumeComputeMarchStepCap()),
                                     0u,
                                     static_cast<uint32_t>(VolumeComputeMarchBatch()) };
            [marchEnc setBytes:&cmc length:sizeof(cmc) atIndex:7];
            for (uint32_t bIdx = 0; bIdx < numBins; ++bIdx)
            {
              uint32_t binnedMeta[2] = { binCap, bIdx * binCap };
              [marchEnc setBytes:&binnedMeta length:sizeof(binnedMeta) atIndex:6];

              // TEMP-DIAG: TG width for the flat binned dispatch. Default 64
              // (8x8-equivalent); 32x1 reproduced the §38.8 occupancy-cliff
              // class on first healthy measurement (44 vs 18 ms obl).
              int binTgW = 64;
              if (const char* v = getenv("VTK_METAL_TEST_CM_BIN_TG"))
              {
                int n = atoi(v);
                if (n >= 32 && n <= 1024 &&
                    (n & (n - 1)) == 0)
                {
                  binTgW = n;
                }
              }
              MTLSize tg = MTLSizeMake(binTgW, 1, 1);
              MTLSize groups = MTLSizeMake((binCap + binTgW - 1) / binTgW, 1, 1);
              [marchEnc dispatchThreadgroups:groups threadsPerThreadgroup:tg];
            }
            [marchEnc endEncoding];
          }
        }
        else if (!VolumeComputeMarchNoMarch())
        {
          // -- stage 2: 2D compute marcher --
          void* marchPso = this->GetOrCreateComputeMarchPipeline(mtlDevice, featureMask, false);
          if (marchPso)
          {
            id<MTLCommandQueue> marchQ = (__bridge id<MTLCommandQueue>)this->ComputeMarchQueue;
            if (!marchQ && VolumeComputeMarchSynth() &&
                VolumeComputeMarchUseFastQueue())
            {
              // SYNTH needs nothing from the window queue before marching, so
              // the march can run on a probe-selected fast-slot queue; the CPU
              // wait below keeps Phase-3b ordering trivially correct. qoff
              // (window CB) is now the default (§38.18.1) — fast queue is
              // opt-in via VTK_METAL_TEST_CM_FASTQUEUE=1 (or legacy
              // VTK_METAL_TEST_CM_QUEUEPROBE=1) to avoid the 3-slot
              // Private-heap serialization and poisoned-slot hazard.
              marchQ = ProbeAndSelectFastQueue((__bridge id<MTLDevice>)mtlDevice, 3);
              this->ComputeMarchQueue = (__bridge void*)marchQ;
              [(__bridge id)marchQ retain];
            }
            id<MTLCommandBuffer> cbUse = marchQ ? [marchQ commandBuffer] : commandBuffer;
            // §38.17: the segment build ran on `commandBuffer`; consuming
            // from another queue's buffer would race it.
            if (segLive) cbUse = commandBuffer;
            id<MTLComputeCommandEncoder> marchEnc = [cbUse computeCommandEncoder];
            marchEnc.label = @"VTK Volume ComputeMarch 2D";
            if (getenv("VTK_METAL_TEST_MM_SEG_DEBUG"))
            {
              id<MTLTexture> tgtTex = (__bridge id<MTLTexture>)target;
              id<MTLTexture> segTex =
                (__bridge id<MTLTexture>)this->SegMarchTexture;
              fprintf(stderr,
                "[segcm] 2D march target=%p usage=%#lx SegMarch=%p "
                "usage=%#lx dims=%ux%u\n",
                (void*)tgtTex,
                tgtTex ? (unsigned long)[tgtTex usage] : 0ul,
                (void*)segTex,
                segTex ? (unsigned long)[segTex usage] : 0ul,
                this->SegAtlasWidth, this->SegAtlasHeight);
            }
            [marchEnc setComputePipelineState:(__bridge id<MTLComputePipelineState>)marchPso];

            this->BindComputeMarchTextures((__bridge void*)marchEnc, (__bridge void*)atlasA,
              (__bridge void*)atlasB, (__bridge void*)atlasC, (__bridge void*)target);

            [marchEnc setBuffer:(__bridge id<MTLBuffer>)uniformBuf offset:0 atIndex:1];
            [marchEnc setBytes:&compPbd length:sizeof(compPbd) atIndex:2];
            [marchEnc setBuffer:rectCoordsBuf offset:0 atIndex:3];
            [marchEnc setBytes:&lightUniforms length:sizeof(lightUniforms) atIndex:4];
            // §38.17: segment consume inputs (slots 8/9) + consume-debug out.
            [marchEnc setBuffer:(__bridge id<MTLBuffer>)(segLive
                ? this->SegIndexBuffer : this->DummyRectCoordsBuffer) offset:0 atIndex:8];
            [marchEnc setBuffer:(__bridge id<MTLBuffer>)(segLive
                ? this->SegPoolBuffer : this->DummyRectCoordsBuffer) offset:0 atIndex:9];
            [marchEnc setBuffer:(__bridge id<MTLBuffer>)this->SegConsumeDbgBuffer
                      offset:0 atIndex:10];

            ComputeMarchControl cmc{ VolumeComputeMarchFloor() ? 1u : 0u,
                                     static_cast<uint32_t>(VolumeComputeMarchStepCap()),
                                     VolumeComputeMarchSynth() ? 1u : 0u,
                                     static_cast<uint32_t>(VolumeComputeMarchBatch()) };
            [marchEnc setBytes:&cmc length:sizeof(cmc) atIndex:7];
            // Scene-depth early-out source for CM_SYNTH (setupVolumeRay).
            [marchEnc setTexture:(__bridge id<MTLTexture>)(this->DepthTextureOcclusion
                ? this->DepthTextureOcclusion : this->DummyDepthTexture)
                        atIndex:20];

            int tgw = 8, tgh = 8;
            VolumeComputeMarchTG(tgw, tgh);
            MTLSize tg = MTLSizeMake(tgw, tgh, 1);
            MTLSize groups = MTLSizeMake((renderWidth + tgw - 1) / tgw,
                                         (renderHeight + tgh - 1) / tgh, 1);
            [marchEnc dispatchThreadgroups:groups threadsPerThreadgroup:tg];
            [marchEnc endEncoding];
            if (cbUse != commandBuffer)
            {
              // Private-queue march: commit + CPU-wait so the result is
              // ordered before Phase 3b samples it from the window queue.
              [cbUse commit];
              [cbUse waitUntilCompleted];
            }
          }
        }

        // §38.17 MM_SEG_DEBUG: read back the march target AFTER the march.
        if (getenv("VTK_METAL_TEST_MM_SEG_DEBUG"))
        {
          id<MTLBlitCommandEncoder> dbgBlit =
            [commandBuffer blitCommandEncoder];
          [dbgBlit copyFromTexture:(__bridge id<MTLTexture>)this->SegMarchTexture
                       sourceSlice:0 sourceLevel:0
                      sourceOrigin:MTLOriginMake(0,0,0)
                        sourceSize:MTLSizeMake(renderWidth, renderHeight, 1)
                          toBuffer:(__bridge id<MTLBuffer>)
                            this->SegDebugStageBuffer
                      destinationOffset:segImgOff
                 destinationBytesPerRow:renderWidth * 8
               destinationBytesPerImage:renderWidth * renderHeight * 8];
          [dbgBlit endEncoding];
        }

        // Phase 3b blits this over the drawable.
        if (!VolumeComputeMarchNoBlit())
        {
          AssignRetainedMetalObject(this->ImageSampleColorTexture, target);
          this->ImageSampleFBOWidth = renderWidth;
          this->ImageSampleFBOHeight = renderHeight;
        }
      }
      else if (!selectionRender &&
               this->GetBlendMode() == vtkVolumeMapper::COMPOSITE_BLEND &&
               VolumeSegWanted() && this->UseMinMaxAcceleration &&
               this->UseGPUMinMax && !usePartitions &&
               this->MinMaxTexture != nullptr)
      {
        if (getenv("VTK_METAL_TEST_MM_SEG_DEBUG"))
          fprintf(stderr, "[segdirect] branch entered\n");
        // §35.14 segment pre-pass on the standard direct path (VTK_METAL_TEST_MM_SEG=1).
        // The drawable encoder is already open, so follow the slab ping-pong
        // precedent: end it, encode the ray-atlas raster + segment-builder
        // compute, march the volume into an offscreen RGBA16Float target with a
        // segment-consuming pipeline, then expose the result through the
        // ImageSample members so the renderer's Phase 3b blit composites it.
        id<MTLRenderCommandEncoder> curEnc =
          (__bridge id<MTLRenderCommandEncoder>)
            metalRenderWindow->GetCurrentRenderCommandEncoder();
        if (curEnc)
        {
          [curEnc endEncoding];
          metalRenderWindow->SetCurrentRenderCommandEncoder(nullptr);
        }

        if (!this->EnsureSegResources(mtlDevice, mtlQueue,
              renderWidth, renderHeight))
        {
          return;
        }

        id<MTLBuffer> segCntBuf =
          (__bridge id<MTLBuffer>)this->SegPoolCounterBuffer;
        // TEMP-DIAG §35.14: last frame's final claim count (buffer completed
        // behind the frame semaphore before we get here).
        this->SegLastClaimWords.store(
          *static_cast<uint32_t*>(segCntBuf.contents));
        if (getenv("VTK_METAL_TEST_MM_SEG_DEBUG") &&
            this->SegLastClaimWords.load() > 0)
          fprintf(stderr, "[seg] prev claims: %u words (%.1f MB)\n",
            this->SegLastClaimWords.load(),
            4.0 * this->SegLastClaimWords.load() / (1024.0 * 1024.0));
        *(uint32_t*)segCntBuf.contents = 0;
        [segCntBuf didModifyRange:NSMakeRange(0, 4)];

        id<MTLTexture> atlasA = (__bridge id<MTLTexture>)this->SegAtlasATexture;
        id<MTLTexture> atlasB = (__bridge id<MTLTexture>)this->SegAtlasBTexture;
        id<MTLTexture> atlasC = (__bridge id<MTLTexture>)this->SegAtlasCTexture;

        // -- stage 1: ray-atlas raster (same proxy geometry + vertex fn) --
        static const bool segSkipAtlas =
          getenv("VTK_METAL_TEST_MM_SEG_SKIP_ATLAS") != nullptr;
        static const bool segAtlasNoDraw =
          getenv("VTK_METAL_TEST_MM_SEG_ATLAS_NODRAW") != nullptr;
        if (!segSkipAtlas)
        {
        MTLRenderPassDescriptor* arpd =
          [MTLRenderPassDescriptor renderPassDescriptor];
        arpd.colorAttachments[0].texture = atlasA;
        arpd.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        arpd.colorAttachments[0].storeAction = MTLStoreActionStore;
        arpd.colorAttachments[1].texture = atlasB;
        arpd.colorAttachments[1].loadAction = MTLLoadActionDontCare;
        arpd.colorAttachments[1].storeAction = MTLStoreActionStore;
        arpd.colorAttachments[2].texture = atlasC;
        arpd.colorAttachments[2].loadAction = MTLLoadActionDontCare;
        arpd.colorAttachments[2].storeAction = MTLStoreActionStore;

        if (segAtlasNoDraw)
        {
          // TEMP-DIAG §35.14: pass + barriers only, no geometry, no PSO.
          id<MTLRenderCommandEncoder> emptyEnc =
            [commandBuffer renderCommandEncoderWithDescriptor:arpd];
          [emptyEnc endEncoding];
        }
        else
        {
        id<MTLRenderCommandEncoder> atlasEnc =
          [commandBuffer renderCommandEncoderWithDescriptor:arpd];
        atlasEnc.label = @"VTK Volume Segment RayAtlas";
        MTLViewport amvp;
        amvp.originX = 0; amvp.originY = 0;
        amvp.width = renderWidth; amvp.height = renderHeight;
        amvp.znear = 0.0; amvp.zfar = 1.0;
        [atlasEnc setViewport:amvp];

        void* atlasPso = this->GetOrCreateVolumePipeline(mtlDevice,
          static_cast<uint32_t>(VolumePipelineType::RayAtlas),
          MTLPixelFormatRGBA32Float, MTLPixelFormatInvalid, 1, featureMask);
        if (!atlasPso)
        {
          [atlasEnc endEncoding];
          return;
        }
        this->BindEncoderResources(atlasEnc, uniformBuf, atlasPso, false);
        this->DrawBlocks(atlasEnc, uniformBuf, ren, vol, &uniforms,
          invModelMatrix);
        [atlasEnc endEncoding];
        } // segAtlasNoDraw
        } // segSkipAtlas

        // -- stage 2: segment builder --
        PerBlockData segPbd;
        this->BuildPerBlockData(segPbd, &uniforms);
        // x=width y=height z=poolCapWords w=maxGaps (< shader's gaps[64])
        uint32_t segMeta[4] = { static_cast<uint32_t>(renderWidth),
                                static_cast<uint32_t>(renderHeight),
                                static_cast<uint32_t>(this->SegPoolCapWords),
                                16u };  // maxGaps == builder register capacity
        id<MTLComputeCommandEncoder> segEnc =
          [commandBuffer computeCommandEncoder];
        [segEnc setComputePipelineState:(__bridge id<MTLComputePipelineState>)
          this->SegBuildComputePipeline];
        [segEnc setTexture:atlasA atIndex:0];
        [segEnc setTexture:atlasB atIndex:1];
        [segEnc setTexture:(__bridge id<MTLTexture>)this->MinMaxTexture
                   atIndex:2];
        [segEnc setTexture:atlasC atIndex:3];
        // §38.17: the builder gained synth-input bindings; atlas-input sites
        // pass flags bit0 = 0.
        [segEnc setTexture:(__bridge id<MTLTexture>)this->VolumeTexture atIndex:4];
        [segEnc setTexture:(__bridge id<MTLTexture>)(this->DepthTextureOcclusion
            ? this->DepthTextureOcclusion : this->DummyDepthTexture) atIndex:5];
        [segEnc setTexture:(__bridge id<MTLTexture>)(this->MinMaxBlockTexture
            ? this->MinMaxBlockTexture : this->DummyMinMaxTexture) atIndex:6];
        [segEnc setBuffer:(__bridge id<MTLBuffer>)this->SegIndexBuffer
                   offset:0 atIndex:0];
        [segEnc setBuffer:segCntBuf offset:0 atIndex:1];
        [segEnc setBuffer:(__bridge id<MTLBuffer>)this->SegPoolBuffer
                   offset:0 atIndex:2];
        [segEnc setBytes:&segMeta length:sizeof(segMeta) atIndex:3];
        [segEnc setBytes:&segPbd length:sizeof(segPbd) atIndex:4];
        [segEnc setBuffer:(__bridge id<MTLBuffer>)this->DummyRectCoordsBuffer
                   offset:0 atIndex:7];
        {
          uint32_t buildFlagsRtt = 0u;
          [segEnc setBytes:&buildFlagsRtt length:sizeof(buildFlagsRtt) atIndex:6];
        }
        [segEnc setBuffer:(__bridge id<MTLBuffer>)uniformBuf offset:0 atIndex:5];
        MTLSize segTg = MTLSizeMake(8, 8, 1);
        MTLSize segGroups = MTLSizeMake((renderWidth + 7) / 8,
          (renderHeight + 7) / 8, 1);
        [segEnc dispatchThreadgroups:segGroups threadsPerThreadgroup:segTg];
        [segEnc endEncoding];

        this->SegActiveThisFrame = true;

        // -- stage 3: offscreen march with the consuming pipeline --
        static const bool segSkipMarch =
          getenv("VTK_METAL_TEST_MM_SEG_SKIP_MARCH") != nullptr;
        if (!segSkipMarch)
        {
        id<MTLTexture> target = (__bridge id<MTLTexture>)this->SegMarchTexture;

        MTLRenderPassDescriptor* mrpd =
          [MTLRenderPassDescriptor renderPassDescriptor];
        mrpd.colorAttachments[0].texture = target;
        mrpd.colorAttachments[0].loadAction = MTLLoadActionClear;
        mrpd.colorAttachments[0].clearColor =
          MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
        mrpd.colorAttachments[0].storeAction = MTLStoreActionStore;

        id<MTLRenderCommandEncoder> marchEnc =
          [commandBuffer renderCommandEncoderWithDescriptor:mrpd];
        marchEnc.label = @"VTK Volume SegmentMarch";
        MTLViewport mmvp;
        mmvp.originX = 0; mmvp.originY = 0;
        mmvp.width = renderWidth; mmvp.height = renderHeight;
        mmvp.znear = 0.0; mmvp.zfar = 1.0;
        [marchEnc setViewport:mmvp];

        void* offPso = this->GetOrCreateVolumePipeline(mtlDevice,
          static_cast<uint32_t>(VolumePipelineType::OffscreenLayer),
          MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1, featureMask);
        if (!offPso)
        {
          [marchEnc endEncoding];
          return;
        }
        [marchEnc setFragmentBytes:&lightUniforms
                            length:sizeof(lightUniforms) atIndex:4];
        this->BindEncoderResources(marchEnc, uniformBuf, offPso, false);
        this->DrawBlocks(marchEnc, uniformBuf, ren, vol, &uniforms,
          invModelMatrix);
        [marchEnc endEncoding];

        // Phase 3b blits this over the drawable.
        AssignRetainedMetalObject(this->ImageSampleColorTexture, target);
        this->ImageSampleFBOWidth = renderWidth;
        this->ImageSampleFBOHeight = renderHeight;
        } // segSkipMarch
      }
      else
      {
        // Single-pass: draw straight into the renderer's encoder.
        this->BindEncoderResources(encoder, uniformBuf, directPso, true);
        for (int s = numSlabs - 1; s >= 0; --s)
        {
          if (slabOnly >= 0 && s != slabOnly) { continue; }
          this->DrawBlocks(encoder, uniformBuf, ren, vol, &uniforms, invModelMatrix, s, numSlabs);
          if (slabOnly >= 0) { break; }
        }
      }
    }
  }

  // Signal the semaphore when the GPU finishes this frame's command buffer.
  // Retain the semaphore under MRC to prevent use-after-free if the mapper
  // is destroyed before the handler fires.
  dispatch_semaphore_t sem = (__bridge dispatch_semaphore_t)this->FrameSemaphore;
#if !__has_feature(objc_arc)
  dispatch_retain(sem);
#endif
  [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
    if (getenv("VTK_METAL_TEST_PRINT_GPU_MS"))
    {
      const double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
      vtkGenericWarningMacro("volume cmdbuf GPU ms=" << ms << " status=" << (int)cb.status);
    }
    dispatch_semaphore_signal(sem);
#if !__has_feature(objc_arc)
    dispatch_release(sem);
#endif
  }];

  // Completion handler is installed — the guard is no longer needed.
  semGuard.dismiss();

  } // @autoreleasepool
}

// ============================================================================
// Partitioned volumes are rendered using single-pass grid traversal.
// The occupancy grid marks empty brick regions. Per-brick textures, sorting,
// layer compositing, and framebuffer-fetch accumulation are no longer used.

VTK_ABI_NAMESPACE_END