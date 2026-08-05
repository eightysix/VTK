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
#include "vtkSMPTools.h"
#include "vtkMath.h"
#include "vtkClipConvexPolyData.h"
#include "vtkDensifyPolyData.h"
#include "vtkTriangleFilter.h"
#include "vtkPlaneCollection.h"
#include "vtkPlane.h"
#include "vtkLightCollection.h"
#include "vtkLight.h"
#include "vtkPolyData.h"
#include "vtkPoints.h"
#include "vtkCellArray.h"
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
};

static_assert(sizeof(VolumeMapperUniforms) == 996,
  "VolumeMapperUniforms must be 996 bytes to match Metal shader struct");

static_assert(offsetof(VolumeMapperUniforms, UseCropping) == 640, "");
static_assert(offsetof(VolumeMapperUniforms, UseClipping) == 644, "");
static_assert(offsetof(VolumeMapperUniforms, NumClippingPlanes) == 648, "");
static_assert(offsetof(VolumeMapperUniforms, ClippingPlane0Origin) == 672, "");
static_assert(offsetof(VolumeMapperUniforms, UseMask) == 928, "");
static_assert(offsetof(VolumeMapperUniforms, UseDepthTexture) == 948, "");
static_assert(offsetof(VolumeMapperUniforms, UseNormalTexture) == 952, "");
static_assert(offsetof(VolumeMapperUniforms, UseMinMaxAccel) == 960, "");

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

// Per-block data for volume rendering — must match Metal PerBlockData struct
struct PerBlockData {
  float VolumeBoundsMin[4]; // 0..15
  float VolumeBoundsMax[4]; // 16..31
  float TextureBoundsMin[4]; // 32..47
  float TextureBoundsMax[4]; // 48..63
  float GradientStep[4];    // 64..79  (xyz + pad)
  float MinMaxInfo[4];      // 80..95  (useMinMax, dimX, dimY, dimZ)
};

static_assert(sizeof(PerBlockData) == 96,
  "PerBlockData must be 96 bytes to match Metal shader struct");

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
};

static_assert(sizeof(MinMaxComputeUniforms) == 4*6 + 4 + 4 + 4 + 4 + 257*4,
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

// Decode IEEE 754 half-precision bits back to a float (diagnostics / CPU
// readback). Mirrors FloatToHalf on the encode side.
inline float HalfToFloatBits(uint16_t h)
{
  uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  uint32_t exponent = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x3FF;
  uint32_t f;
  if (exponent == 0)
  {
    if (mantissa == 0)
    {
      f = sign;
    }
    else
    {
      // Denormal: normalize into a normal float
      int e = -1;
      uint32_t m = mantissa;
      do
      {
        ++e;
        m <<= 1;
      } while ((m & 0x400) == 0);
      m &= 0x3FF;
      f = sign | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
    }
  }
  else if (exponent == 31)
  {
    f = sign | 0x7F800000 | (mantissa << 13);
  }
  else
  {
    f = sign | ((uint32_t)(exponent - 15 + 127) << 23) | (mantissa << 13);
  }
  float result;
  std::memcpy(&result, &f, sizeof(result));
  return result;
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
  MTLStorageMode storage)
{
  MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
  desc.textureType = MTLTextureType3D;
  desc.pixelFormat = format;
  desc.width = width;
  desc.height = height;
  desc.depth = depth;
  desc.mipmapLevelCount = 1;
  desc.usage = usage;
  desc.storageMode = storage;
  id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
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
static id<MTLTexture> CreateR8MinMaxTexture(
  id<MTLDevice> device,
  int dimX,
  int dimY,
  int dimZ,
  MTLStorageMode storage,
  MTLTextureUsage usage)
{
  return NewTexture3D(
    device,
    MTLPixelFormatR8Unorm,
    static_cast<NSUInteger>(dimX),
    static_cast<NSUInteger>(dimY),
    static_cast<NSUInteger>(dimZ),
    usage,
    storage);
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
};

static_assert(sizeof(NormalComputeUniforms) == 48,
  "NormalComputeUniforms must match Metal struct (48 bytes)");

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

static VolumeBounds ComputeVolumeBounds(vtkImageData* input)
{
  VolumeBounds bounds{};
  int ext[6];
  double origin[3], spacing[3];
  input->GetExtent(ext);
  input->GetOrigin(origin);
  input->GetSpacing(spacing);
  double x0 = origin[0] + spacing[0] * ext[0];
  double x1 = origin[0] + spacing[0] * ext[1];
  double y0 = origin[1] + spacing[1] * ext[2];
  double y1 = origin[1] + spacing[1] * ext[3];
  double z0 = origin[2] + spacing[2] * ext[4];
  double z1 = origin[2] + spacing[2] * ext[5];
  bounds.Min[0] = std::min(x0, x1);
  bounds.Max[0] = std::max(x0, x1);
  bounds.Min[1] = std::min(y0, y1);
  bounds.Max[1] = std::max(y0, y1);
  bounds.Min[2] = std::min(z0, z1);
  bounds.Max[2] = std::max(z0, z1);
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

// Fill a RGBA8 transfer function row with CPU-side opacity pre-integration,
// matching the OpenGL backend's approach (vtkOpenGLVolumeOpacityTable::InternalUpdate).
// COMPOSITE blend only; additive blend would need a *= factor instead of pow.
// preIntegrationFactor = sampleDistance / unitDistance.
static void FillTransferFunctionRGBA8WithPreIntegration(
  vtkColorTransferFunction* colorFunc,
  vtkPiecewiseFunction* opacityFunc,
  double scalarMin,
  double scalarMax,
  int width,
  uint8_t* row,
  double preIntegrationFactor)
{
  std::vector<double> rgb(width * 3);
  std::vector<double> alpha(width);
  colorFunc->GetTable(scalarMin, scalarMax, width, rgb.data());
  opacityFunc->GetTable(scalarMin, scalarMax, width, alpha.data());
  for (int i = 0; i < width; ++i)
  {
    double a = alpha[i];
    if (a > 0.0001 && preIntegrationFactor > 0.0)
    {
      a = 1.0 - std::pow(1.0 - a, preIntegrationFactor);
    }
    row[i * 4 + 0] = ColorToByte(rgb[i * 3 + 0]);
    row[i * 4 + 1] = ColorToByte(rgb[i * 3 + 1]);
    row[i * 4 + 2] = ColorToByte(rgb[i * 3 + 2]);
    row[i * 4 + 3] = ColorToByte(a);
  }
}

// Fill a RGBA16Float transfer function row with CPU-side opacity pre-integration,
// matching the OpenGL backend's approach (vtkOpenGLVolumeOpacityTable::InternalUpdate).
// Unlike the RGBA8 variant, the alpha channel retains full float precision so
// that low-opacity transfer functions (e.g. 0.005 max opacity) do not round to
// zero after the 8-bit quantization step. COMPOSITE blend only; additive blend
// would need a *= factor instead of pow.
static void FillTransferFunctionRGBA16FWithPreIntegration(
  vtkColorTransferFunction* colorFunc,
  vtkPiecewiseFunction* opacityFunc,
  double scalarMin,
  double scalarMax,
  int width,
  uint16_t* row,
  double preIntegrationFactor)
{
  std::vector<double> rgb(width * 3);
  std::vector<double> alpha(width);
  colorFunc->GetTable(scalarMin, scalarMax, width, rgb.data());
  opacityFunc->GetTable(scalarMin, scalarMax, width, alpha.data());
  for (int i = 0; i < width; ++i)
  {
    double a = alpha[i];
    if (a > 0.0001 && preIntegrationFactor > 0.0)
    {
      a = 1.0 - std::pow(1.0 - a, preIntegrationFactor);
    }
    row[i * 4 + 0] = FloatToHalf(static_cast<float>(std::clamp(rgb[i * 3 + 0], 0.0, 1.0)));
    row[i * 4 + 1] = FloatToHalf(static_cast<float>(std::clamp(rgb[i * 3 + 1], 0.0, 1.0)));
    row[i * 4 + 2] = FloatToHalf(static_cast<float>(std::clamp(rgb[i * 3 + 2], 0.0, 1.0)));
    row[i * 4 + 3] = FloatToHalf(static_cast<float>(std::clamp(a, 0.0, 1.0)));
  }
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
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];

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

    id<MTLTexture> colorTex = NewTexture2D(device, MTLPixelFormatRGBA16Float, width, height,
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
  vtkIdType numTuples = scalars->GetNumberOfTuples();

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
        MTLStorageModePrivate);
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

  if (!gpuConversionUsed)
  {
    id<MTLBuffer> stagingBuf = [device newBufferWithLength:totalBytes
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
    else
    {
      std::memcpy(uploadPointer, scalars->GetVoidPointer(0), totalBytes);
    }

    id<MTLTexture> oldTex = (__bridge id<MTLTexture>)this->VolumeTexture;
    id<MTLTexture> tex = nil;

    if (oldTex &&
        oldTex.width == static_cast<NSUInteger>(dims[0]) &&
        oldTex.height == static_cast<NSUInteger>(dims[1]) &&
        oldTex.depth == static_cast<NSUInteger>(dims[2]) &&
        oldTex.pixelFormat == fmtInfo.Format)
    {
      tex = oldTex;
    }
    else
    {
      ReleaseMetalObject(this->VolumeTexture);

      tex = NewTexture3D(
        device,
        fmtInfo.Format,
        static_cast<NSUInteger>(dims[0]),
        static_cast<NSUInteger>(dims[1]),
        static_cast<NSUInteger>(dims[2]),
        MTLTextureUsageShaderRead,
        MTLStorageModePrivate);
      if (!tex)
      {
        vtkErrorMacro("Failed to create 3D volume texture");
        [stagingBuf release];
        return false;
      }
      AssignMetalObject(this->VolumeTexture, tex);
    }

    id<MTLCommandBuffer> uploadCmdBuf = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [uploadCmdBuf blitCommandEncoder];
    [blit copyFromBuffer:stagingBuf
            sourceOffset:0
     sourceBytesPerRow:bytesPerRow
   sourceBytesPerImage:bytesPerImage
            sourceSize:MTLSizeMake(dims[0], dims[1], dims[2])
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

    const int DS = 4;

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
    VolumeBounds vb = input ? ComputeVolumeBounds(input) : VolumeBounds{};
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
  this->ReleaseRTTResources();

  ReleaseMetalObject(this->PipelineState);
  ReleaseMetalObject(this->VolumeTexture);
  ReleaseMetalObject(this->ColorOpacityTexture);
  ReleaseMetalObject(this->GradientOpacityTexture);
  ReleaseMetalObject(this->Transfer2DTexture);
  ReleaseMetalObject(this->Transfer2DYAxisTexture);
  this->Transfer2DEnabled = false;
  ReleaseMetalObject(this->MinMaxTexture);
  ReleaseMetalObject(this->MinMaxScratchTexture);
  this->ReleaseGradientNormalTexture();

  this->ReleaseMaskResources();

  // Phase 5: Release GPU min-max compute pipelines
  ReleaseMetalObject(this->MinMaxComputePipeline);
  ReleaseMetalObject(this->DilateComputePipeline);

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

  for (int i = 0; i < 3; ++i)
  {
    ReleaseMetalObject(this->UniformBuffers[i]);
  }

  this->ReleaseGridTraversalResources();

  ReleaseMetalObject(this->VertexBuffer);
  ReleaseMetalObject(this->IndexBuffer);
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
    vtkErrorMacro("RTT_DEBUG GetColorImage early return: tex=" << (this->RTTColorTexture ? "set" : "null")
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
    const NSUInteger bytesPerRow = static_cast<NSUInteger>(w) * 8; // RGBA16Float
    void* tmp = malloc(bytesPerRow * static_cast<NSUInteger>(h));
    if (!tmp)
    {
      return;
    }

    [tex getBytes:tmp bytesPerRow:bytesPerRow
        bytesPerImage:bytesPerRow * static_cast<NSUInteger>(h)
           fromRegion:MTLRegionMake2D(0, 0, w, h)
          mipmapLevel:0 slice:0];

    __fp16* src = static_cast<__fp16*>(tmp);
    // Metal getBytes returns the top texture row first; vtkImageData expects
    // the first row at the bottom of the image (matching OpenGL readback), so
    // flip rows vertically.
    for (int y = 0; y < h; ++y)
    {
      for (int x = 0; x < w; ++x)
      {
        const __fp16* s = src + 4 * (y * w + x);
        const int dstRow = (h - 1 - y);
        unsigned char* o = ptr + 4 * (dstRow * w + x);
        for (int c = 0; c < 4; ++c)
        {
          float v = (float)s[c];
          v = (v < 0.0f) ? 0.0f : ((v > 1.0f) ? 1.0f : v);
          o[c] = static_cast<unsigned char>(v * 255.0f + 0.5f);
        }
      }
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
    if (this->EffectiveInput != img)
    {
      this->EffectiveInput = img;
      this->EffectiveInputTime.Modified();
    }
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
    double shiftedOrigin[3];
    for (int i = 0; i < 3; ++i)
    {
      shiftedOrigin[i] = origin[i] + 0.5 * spacing[i];
    }
    proxy->SetOrigin(shiftedOrigin);
    proxy->SetSpacing(spacing);
    proxy->GetPointData()->SetScalars(scalars);
    proxy->Modified();
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
  vtkDataArray* scalars = this->GetScalars(
    input, this->ScalarMode, this->ArrayAccessMode, this->ArrayId, this->ArrayName, this->CellFlag);
  if (!scalars)
  {
    return false;
  }

  // TEMP DEBUG
  {
    int ddims[3];
    input->GetDimensions(ddims);
    vtkIdType cIdx = ((ddims[2] / 2) * ddims[1] + ddims[1] / 2) * ddims[0] + ddims[0] / 2;
    double rng[2];
    scalars->GetRange(rng);
    std::cout << "[UVT] name=" << (scalars->GetName() ? scalars->GetName() : "-")
              << " nc=" << scalars->GetNumberOfComponents() << " range=[" << rng[0] << "," << rng[1] << "]"
              << " centerVal=" << scalars->GetComponent(cIdx, 0)
              << " activeScalars=" << (input->GetPointData()->GetScalars() ? input->GetPointData()->GetScalars()->GetName() : "(null)")
              << std::endl;
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
      VolumeBounds vb = ComputeVolumeBounds(input);
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
      vtkIdType numTuples = scalars->GetNumberOfTuples();

      if (numComponents < 1 || numComponents > 4)
      {
        vtkErrorMacro("Unsupported number of scalar components: " << numComponents);
        return false;
      }

      this->VolumeNumComponents = numComponents;

      // Store model-space bounds using image extent (handles non-zero extents and negative spacing)
      {
        VolumeBounds vb = ComputeVolumeBounds(input);
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
            MTLStorageModePrivate);
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
      if (!gpuConversionUsed)
      {
        id<MTLBuffer> stagingBuf = [device newBufferWithLength:totalBytes
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
        else
        {
          std::memcpy(uploadPointer, scalars->GetVoidPointer(0), totalBytes);
        }

      id<MTLTexture> oldTex = (__bridge id<MTLTexture>)this->VolumeTexture;
      id<MTLTexture> tex = nil;

      if (oldTex &&
          oldTex.width == static_cast<NSUInteger>(dims[0]) &&
          oldTex.height == static_cast<NSUInteger>(dims[1]) &&
          oldTex.depth == static_cast<NSUInteger>(dims[2]) &&
          oldTex.pixelFormat == fmtInfo.Format)
      {
        tex = oldTex;
      }
      else
      {
        ReleaseMetalObject(this->VolumeTexture);

        tex = NewTexture3D(
          device,
          fmtInfo.Format,
          static_cast<NSUInteger>(dims[0]),
          static_cast<NSUInteger>(dims[1]),
          static_cast<NSUInteger>(dims[2]),
          MTLTextureUsageShaderRead,
          MTLStorageModePrivate);
        if (!tex)
        {
          vtkErrorMacro("Failed to create 3D volume texture");
          return false;
        }
        AssignMetalObject(this->VolumeTexture, tex);
      }

      id<MTLCommandBuffer> uploadCmdBuf = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [uploadCmdBuf blitCommandEncoder];
      [blit copyFromBuffer:stagingBuf
              sourceOffset:0
       sourceBytesPerRow:bytesPerRow
     sourceBytesPerImage:bytesPerImage
              sourceSize:MTLSizeMake(dims[0], dims[1], dims[2])
               toTexture:tex
        destinationSlice:0
        destinationLevel:0
       destinationOrigin:MTLOriginMake(0, 0, 0)];
      [blit endEncoding];
      [uploadCmdBuf commit];
      // Release our reference to the staging buffer. Metal keeps the buffer
      // alive internally until the command buffer completes.
      [stagingBuf release];

      } // end if (!gpuConversionUsed)

      this->VolumeUploadTime.Modified();
    }
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

  bool sampleDistChanged =
    (actualSampleDistance != this->LastTransferFunctionSampleDist);

  bool scalarRangeChanged =
    (this->ScalarRange[0] != this->LastTransferFunctionScalarRange[0]) ||
    (this->ScalarRange[1] != this->LastTransferFunctionScalarRange[1]);

  bool doReload = (this->ColorOpacityTexture == nullptr);
  doReload |= (colorFunc->GetMTime() > this->TransferFunctionUploadTime.GetMTime());
  doReload |= (opacityFunc->GetMTime() > this->TransferFunctionUploadTime.GetMTime());
  doReload |= scalarRangeChanged;
  doReload |= sampleDistChanged;

  if (doReload)
  {
    // Compute pre-integration factor (sampleDistance / unitDistance), same as
    // vtkOpenGLVolumeOpacityTable::InternalUpdate.
    double unitDist = property->GetScalarOpacityUnitDistance(0);
    if (unitDist <= 0.0) unitDist = 1.0;
    double preIntegrationFactor = actualSampleDistance / unitDist;

    const int tfWidth = 256;

    @autoreleasepool
    {
      id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

      std::vector<uint16_t> tfData(static_cast<size_t>(tfWidth) * 4);
      FillTransferFunctionRGBA16FWithPreIntegration(
        colorFunc, opacityFunc,
        this->ScalarRange[0], this->ScalarRange[1],
        tfWidth, tfData.data(),
        preIntegrationFactor);

      // TEMP DEBUG: dump TF table stats
      {
        float maxA = 0.0f, minA = 1.0f, maxR = 0.0f;
        for (int i = 0; i < tfWidth; ++i)
        {
          float a = HalfToFloatBits(tfData[i * 4 + 3]);
          float r = HalfToFloatBits(tfData[i * 4 + 0]);
          maxA = std::max(maxA, a); minA = std::min(minA, a);
          maxR = std::max(maxR, r);
        }
        std::cout << "[TFDBG] range=[" << this->ScalarRange[0] << "," << this->ScalarRange[1]
                  << "] alphaMin=" << minA << " alphaMax=" << maxA
                  << " colorMaxR=" << maxR
                  << " preIntFactor=" << preIntegrationFactor
                  << " blendMode=" << (this->GetBlendMode())
                  << " indep=" << (property->GetIndependentComponents() ? 1 : 0)
                  << " tfMode=" << static_cast<int>(property->GetTransferFunctionMode())
                  << std::endl;
      }

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

      this->LastTransferFunctionScalarRange[0] = this->ScalarRange[0];
      this->LastTransferFunctionScalarRange[1] = this->ScalarRange[1];
      this->LastTransferFunctionSampleDist = actualSampleDistance;
      this->TransferFunctionUploadTime.Modified();
    }
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

  bool doReload = (this->GradientOpacityTexture == nullptr);
  doReload |= (gradOpacityFunc->GetMTime() > this->GradientOpacityUploadTime.GetMTime());

  if (doReload)
  {
    @autoreleasepool
    {
      id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

      // Build 256-entry gradient opacity lookup table.
      // Range: [0, 0.25 * scalarRange] — matches the normalization in the shader
      // where gradient magnitude is normalized to [0, 0.25 * dataRange].
      double scalarRange = this->ScalarRange[1] - this->ScalarRange[0];
      if (scalarRange <= 0.0)
      {
        scalarRange = 1.0;
      }
      double gradMax = scalarRange * 0.25;

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

  const char* yName = this->GetTransfer2DYAxisArray();
  if (!yName)
  {
    vtkErrorMacro("TF_2D mode requires a Y-axis scalar array (SetTransfer2DYAxisArray)");
    this->Transfer2DEnabled = false;
    return false;
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
    vtkErrorMacro("TF_2D mode: Y-axis array '" << yName << "' not found on input");
    this->Transfer2DEnabled = false;
    return false;
  }

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

  if (maskInput && property &&
      this->MaskType == vtkGPUVolumeRayCastMapper::LabelMapMaskType)
  {
    u->UseMask = 1.0f;
    u->MaskBlendFactor = this->MaskBlendFactor;
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
  else
  {
    u->UseMask = 0.0f;
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

  if (!this->VolumeTexture)
  {
    return false;
  }

  // Timestamp-based caching: skip recompute when nothing changed.
  if (input && this->MinMaxTexture)
  {
    vtkVolumeProperty* property = vol ? vol->GetProperty() : nullptr;
    vtkPiecewiseFunction* opFunc = property ? property->GetScalarOpacity() : nullptr;
    if (opFunc &&
        input->GetMTime() <= this->MinMaxUploadTime.GetMTime() &&
        opFunc->GetMTime() <= this->MinMaxUploadTime.GetMTime())
    {
      return true;
    }
  }

  id<MTLTexture> volTex = (__bridge id<MTLTexture>)this->VolumeTexture;
  int dims[3] = { static_cast<int>(volTex.width),
                  static_cast<int>(volTex.height),
                  static_cast<int>(volTex.depth) };

  const int DS = 4;
  int mmDims[3] = {
    std::max(1, (dims[0] + DS - 1) / DS),
    std::max(1, (dims[1] + DS - 1) / DS),
    std::max(1, (dims[2] + DS - 1) / DS)
  };
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
    memcpy(u.opacityPrefix, opacityPrefix, sizeof(opacityPrefix));

    // --- Reuse or create persistent MinMax texture ---
    id<MTLTexture> permTex = (__bridge id<MTLTexture>)this->MinMaxTexture;
    if (!permTex ||
        permTex.width != static_cast<NSUInteger>(mmDims[0]) ||
        permTex.height != static_cast<NSUInteger>(mmDims[1]) ||
        permTex.depth != static_cast<NSUInteger>(mmDims[2]) ||
        permTex.storageMode != MTLStorageModePrivate ||
        permTex.pixelFormat != MTLPixelFormatR8Unorm)
    {
      permTex = CreateR8MinMaxTexture(
        device,
        mmDims[0],
        mmDims[1],
        mmDims[2],
        MTLStorageModePrivate,
        MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
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

    [mmEnc endEncoding];
    [cmdBuf commit];

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

  if (!doReload)
  {
    if (skipGlobalTexture)
    {
      return this->MacrocellScalarMin.empty() == false;
    }
  if (skipGlobalTexture)
  {
    return !this->MacrocellScalarMin.empty();
  }
  return this->MinMaxTexture != nullptr;
}

  @autoreleasepool
  {
    id<MTLDevice> device = (__bridge id<MTLDevice>)mtlDeviceVoid;

    int dims[3];
    input->GetDimensions(dims);

    // Downsample factor: 4x in each dimension
    const int DS = 4;
    int mmDims[3] = {
      std::max(1, (dims[0] + DS - 1) / DS),
      std::max(1, (dims[1] + DS - 1) / DS),
      std::max(1, (dims[2] + DS - 1) / DS)
    };
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
  const double modelBounds[6], const double boundsSize[3],
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

    // Direction: transform to volume-local [0,1] space
    double* lfp = light->GetTransformedFocalPoint();
    double* lp  = light->GetTransformedPosition();
    double lightDir[3];
    vtkMath::Subtract(lfp, lp, lightDir);
    vtkMath::Normalize(lightDir);

    // Transform direction to model space, then to normalized volume space
    double dirLocal[4] = { lightDir[0], lightDir[1], lightDir[2], 0.0 };
    invModelMatrix->MultiplyPoint(dirLocal, dirLocal);
    // Scale by bounds size for normalized space (direction, not position)
    dirLocal[0] /= boundsSize[0];
    dirLocal[1] /= boundsSize[1];
    dirLocal[2] /= boundsSize[2];
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

      // Transform position to normalized volume [0,1] space
      double posWorld[4] = { lp[0], lp[1], lp[2], 1.0 };
      double posLocal[4];
      invModelMatrix->MultiplyPoint(posWorld, posLocal);
      if (fabs(posLocal[3]) > 1e-12) {
        posLocal[0] /= posLocal[3];
        posLocal[1] /= posLocal[3];
        posLocal[2] /= posLocal[3];
      }
      for (int a = 0; a < 3; ++a) {
        L.position[a] = static_cast<float>((posLocal[a] - modelBounds[a * 2]) / boundsSize[a]);
      }

      // Attenuation
      double* attn = light->GetAttenuationValues();
      double charSize = std::max({boundsSize[0], boundsSize[1], boundsSize[2]});
      L.attenuation[0] = static_cast<float>(attn[0]);  // constant
      L.attenuation[1] = static_cast<float>(attn[1] * charSize);  // linear (scaled)
      L.attenuation[2] = static_cast<float>(attn[2] * charSize * charSize);  // quadratic (scaled)
      L.attenuation[3] = static_cast<float>(light->GetExponent());  // spot exponent
    } else {
      L.position[3] = 0.0f;  // type = directional
      L.attenuation[0] = 1.0f;  // no attenuation for directional
    }
  }
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
        id<MTLBuffer> buf = [device newBufferWithLength:sizeof(VolumeMapperUniforms)
                                                options:MTLResourceStorageModeShared];
        if (!buf)
        {
          vtkErrorMacro("Failed to create uniform buffer");
          return false;
        }
        AssignMetalObject(this->UniformBuffers[i], buf);
      }
    }

    // Use model-space bounds for vertex positions (using extent for correctness)
    if (input)
    {
      VolumeBounds vb = ComputeVolumeBounds(input);
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
      // Camera outside: simple 8-vertex box (original fast path)
      // Camera inside: clip against near plane, densify, triangulate
      if (!cameraInside)
      {
        float vertices[24];
        unsigned int indices[36];

        // Unit cube [0,1] — the vertex shader scales to each block's model-space bounds.
        float unitVerts[] = {
          0,0,0, 1,0,0, 1,1,0, 0,1,0,
          0,0,1, 1,0,1, 1,1,1, 0,1,1
        };
        memcpy(vertices, unitVerts, sizeof(unitVerts));

        indices[0] = 0;  indices[1] = 2;  indices[2] = 1;  indices[3] = 0;  indices[4] = 3;  indices[5] = 2;
        indices[6] = 4;  indices[7] = 5;  indices[8] = 6;  indices[9] = 4;  indices[10] = 6; indices[11] = 7;
        indices[12] = 0; indices[13] = 7; indices[14] = 3; indices[15] = 0; indices[16] = 4; indices[17] = 7;
        indices[18] = 1; indices[19] = 2; indices[20] = 6; indices[21] = 1; indices[22] = 6; indices[23] = 5;
        indices[24] = 3; indices[25] = 6; indices[26] = 2; indices[27] = 3; indices[28] = 7; indices[29] = 6;
        indices[30] = 0; indices[31] = 1; indices[32] = 5; indices[33] = 0; indices[34] = 5; indices[35] = 4;

        this->IndexCount = sizeof(indices) / sizeof(unsigned int);

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
          id<MTLBuffer> vbuf = [device newBufferWithBytes:vertices
                                                  length:sizeof(vertices)
                                                 options:MTLResourceStorageModeShared];
          if (!vbuf)
          {
            vtkErrorMacro("Failed to create vertex buffer");
            return false;
          }
          AssignMetalObject(this->VertexBuffer, vbuf);
        }

        {
          id<MTLBuffer> ibuf = [device newBufferWithBytes:indices
                                                  length:sizeof(indices)
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

        // Transform normal to volume coordinates using inverse transpose
        // For transforming normals from world space to object space, we need
        // the inverse transpose of the model matrix
        vtkNew<vtkMatrix4x4> worldToData;
        vtkMatrix4x4::Invert(dataToWorld, worldToData);

        // Transform origin point to volume coordinates using inverse matrix
        worldToData->MultiplyPoint(pOrigin, pOrigin);

        // Transform normal using transpose of inverse (i.e., inverse transpose)
        double* invMat = worldToData->GetData();
        double pNormalV[3];
        pNormalV[0] = pNormal[0] * invMat[0] + pNormal[1] * invMat[1] + pNormal[2] * invMat[2];
        pNormalV[1] = pNormal[0] * invMat[4] + pNormal[1] * invMat[5] + pNormal[2] * invMat[6];
        pNormalV[2] = pNormal[0] * invMat[8] + pNormal[1] * invMat[9] + pNormal[2] * invMat[10];
        vtkMath::Normalize(pNormalV);

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

        // Clip, densify, then triangulate to guarantee triangle output
        vtkNew<vtkDensifyPolyData> densifyPolyData;
        densifyPolyData->SetInputConnection(clip->GetOutputPort());
        densifyPolyData->SetNumberOfSubdivisions(2);

        vtkNew<vtkTriangleFilter> triFilter;
        triFilter->SetInputConnection(densifyPolyData->GetOutputPort());
        triFilter->Update();

        vtkPolyData* finalPolyData = triFilter->GetOutput();
        vtkPoints* points = finalPolyData->GetPoints();
        vtkCellArray* polys = finalPolyData->GetPolys();

        // Normalize clipped points from model-space to [0,1] for the vertex shader
        double bmin[3] = { this->ModelBounds[0], this->ModelBounds[2], this->ModelBounds[4] };
        double bsize[3] = {
          this->ModelBounds[1] - this->ModelBounds[0],
          this->ModelBounds[3] - this->ModelBounds[2],
          this->ModelBounds[5] - this->ModelBounds[4]
        };
        for (int k = 0; k < 3; ++k)
        {
          if (std::fabs(bsize[k]) < 1e-10) bsize[k] = 1.0;
        }

        std::vector<float> vertices;
        vertices.reserve(points->GetNumberOfPoints() * 3);

        for (vtkIdType i = 0; i < points->GetNumberOfPoints(); ++i)
        {
          double pt[3];
          points->GetPoint(i, pt);
          vertices.push_back(static_cast<float>((pt[0] - bmin[0]) / bsize[0]));
          vertices.push_back(static_cast<float>((pt[1] - bmin[1]) / bsize[1]));
          vertices.push_back(static_cast<float>((pt[2] - bmin[2]) / bsize[2]));
        }

        std::vector<unsigned int> indices;
        vtkIdType npts;
        const vtkIdType* pts;

        polys->InitTraversal();
        while (polys->GetNextCell(npts, pts))
        {
          if (npts != 3) continue;
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
  VolumePipelineKey key = { type, colorFormat, depthFormat, sampleCount, featureMask };
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
    pt == VolumePipelineType::RenderToImage);

  if (hasFeatureConstants)
  {
    MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];

    BOOL shading = (featureMask & VolumeFeature_Shading) ? YES : NO;
    BOOL gradOp  = (featureMask & VolumeFeature_GradientOpacity) ? YES : NO;
    BOOL mask    = (featureMask & VolumeFeature_Mask) ? YES : NO;
    BOOL minmax  = (featureMask & VolumeFeature_MinMax) ? YES : NO;
    BOOL normalTex = (featureMask & VolumeFeature_NormalTexture) ? YES : NO;
    BOOL linearInterp = (featureMask & VolumeFeature_LinearInterpolation) ? YES : NO;

    [constants setConstantValue:&shading type:MTLDataTypeBool withName:@"fc_shading"];
    [constants setConstantValue:&gradOp  type:MTLDataTypeBool withName:@"fc_gradientOpacity"];
    [constants setConstantValue:&mask    type:MTLDataTypeBool withName:@"fc_mask"];
    [constants setConstantValue:&minmax  type:MTLDataTypeBool withName:@"fc_minmax"];
    [constants setConstantValue:&normalTex type:MTLDataTypeBool withName:@"fc_normalTexture"];
    [constants setConstantValue:&linearInterp type:MTLDataTypeBool withName:@"fc_linearInterpolation"];

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

  // DirectScreen and FullscreenDirect use blending; offscreen pipelines do not.
  if (pt == VolumePipelineType::DirectScreen || pt == VolumePipelineType::FullscreenDirect ||
      pt == VolumePipelineType::GridTraversalDirect)
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
    // Composite the volume over the cleared (white) RTT background, matching the
    // OpenGL RenderToImage framebuffer setup.
    pipelineDesc.colorAttachments[0].blendingEnabled = YES;
    pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
    pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
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

  // Cache and return.
  // The +1 from new() is owned by the cache.
  // The caller receives a non-owning handle; use AssignRetainedMetalObject
  // when storing into a member slot.
  this->PipelineCache[key] = (__bridge void*)pso;
  return (__bridge void*)pso;
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

  SetFragmentTextureOrFallback(encoder, 0, this->VolumeTexture, this->DummyVolumeTexture);
  [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->ColorOpacityTexture atIndex:1];
  SetFragmentTextureOrFallback(encoder, 2, this->DepthTextureOcclusion, this->DummyDepthTexture);
  SetFragmentTextureOrFallback(encoder, 3, this->GradientOpacityTexture, this->ColorOpacityTexture);
  SetFragmentTextureOrFallback(encoder, 4, this->MaskTexture, this->DummyMaskTexture);
  SetFragmentTextureOrFallback(encoder, 5, this->LabelMapTransferTexture, this->ColorOpacityTexture);
  SetFragmentTextureOrFallback(encoder, 6, this->MinMaxTexture, this->DummyMinMaxTexture);
  SetFragmentTextureOrFallback(encoder, 7, this->GradientNormalTexture, this->DummyVolumeTexture);
  SetFragmentTextureOrFallback(encoder, 9, this->Transfer2DTexture, this->ColorOpacityTexture);
  SetFragmentTextureOrFallback(encoder, 10, this->Transfer2DYAxisTexture, this->DummyVolumeTexture);
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

  SetFragmentTextureOrFallback(encoder, 0, volTexVoid, this->DummyVolumeTexture);
  [encoder setFragmentTexture:(__bridge id<MTLTexture>)this->ColorOpacityTexture atIndex:1];
  SetFragmentTextureOrFallback(encoder, 2, this->DepthTextureOcclusion, this->DummyDepthTexture);
  SetFragmentTextureOrFallback(encoder, 3, this->GradientOpacityTexture, this->ColorOpacityTexture);
  SetFragmentTextureOrFallback(encoder, 4, this->MaskTexture, this->DummyMaskTexture);
  SetFragmentTextureOrFallback(encoder, 5, this->LabelMapTransferTexture, this->ColorOpacityTexture);
  SetFragmentTextureOrFallback(encoder, 6, minMaxTexVoid, this->DummyMinMaxTexture);
  SetFragmentTextureOrFallback(encoder, 7, normalTexVoid, this->DummyVolumeTexture);
  SetFragmentTextureOrFallback(encoder, 9, this->Transfer2DTexture, this->ColorOpacityTexture);
  SetFragmentTextureOrFallback(encoder, 10, this->Transfer2DYAxisTexture, this->DummyVolumeTexture);
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
  void* uniformsVoid, vtkMatrix4x4* vtkNotUsed(invModelMatrix))
{
  id<MTLRenderCommandEncoder> encoder =
    (__bridge id<MTLRenderCommandEncoder>)encoderVoid;
  id<MTLBuffer> uniformBuf = (__bridge id<MTLBuffer>)uniformBufVoid;
  VolumeMapperUniforms* uniforms = static_cast<VolumeMapperUniforms*>(uniformsVoid);
  id<MTLBuffer> indexBuf = (__bridge id<MTLBuffer>)this->IndexBuffer;

  PerBlockData pbd = {};
  BuildPerBlockData(pbd, uniforms);

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
  bool useDirectPipeline)
{
  id<MTLRenderCommandEncoder> encoder =
    (__bridge id<MTLRenderCommandEncoder>)encoderVoid;
  id<MTLBuffer> uniformBuf = (__bridge id<MTLBuffer>)uniformBufVoid;
  VolumeMapperUniforms* uniforms = static_cast<VolumeMapperUniforms*>(uniformsVoid);

  uint32_t pipelineType = useDirectPipeline
    ? static_cast<uint32_t>(VolumePipelineType::FullscreenDirect)
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
  if (uniforms->UseLinearVolumeInterpolation > 0.5f)
    featureMask |= VolumeFeature_LinearInterpolation;

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

  [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
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
  vtkDataArray* scalars = this->GetScalars(
    input, this->ScalarMode, this->ArrayAccessMode, this->ArrayId, this->ArrayName, this->CellFlag);
  if (scalars)
  {
    scalars->GetRange(this->ScalarRange, 0);
  }
  else
  {
    this->ScalarRange[0] = 0.0;
    this->ScalarRange[1] = 1.0;
  }

  // TEMP DEBUG
  {
    int ddims[3];
    input->GetDimensions(ddims);
    int nc = scalars ? scalars->GetNumberOfComponents() : -1;
    const char* an = scalars ? scalars->GetName() : "(null)";
    std::cout << "[MDBG2] GPURender dims=" << ddims[0] << "x" << ddims[1] << "x" << ddims[2]
              << " nc=" << nc << " name=" << (an ? an : "-")
              << " scalarRange=[" << this->ScalarRange[0] << "," << this->ScalarRange[1] << "]"
              << " tfMode=" << (vol->GetProperty() ? static_cast<int>(vol->GetProperty()->GetTransferFunctionMode()) : -1)
              << " has2D=" << (vol->GetProperty() && vol->GetProperty()->GetTransferFunction2D() ? 1 : 0)
              << std::endl;
  }

  // Phase 5: GPU-accelerated min-max generation.
  // For single-block volumes with UseGPUMinMax, we must upload the volume
  // texture first, then dispatch compute kernels. For partitioned volumes
  // (or when GPU min-max is disabled), the CPU path runs before volume
  // upload so that UpdateBlockTextures can reuse the per-macrocell data.
  bool usePartitions = (this->Partitions[0] > 1 || this->Partitions[1] > 1 || this->Partitions[2] > 1);

  if (this->UseGPUMinMax && !usePartitions)
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
    actualSampleDistance = minWorldSpacing;
    if (this->ReductionFactor < 1.0 && this->ReductionFactor != 0.0)
    {
      actualSampleDistance /= this->ReductionFactor;
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
  if (this->MaskInput && this->MaskType == vtkGPUVolumeRayCastMapper::LabelMapMaskType)
  {
    this->UpdateMaskTexture(mtlDevice, mtlQueue, vol);
    this->UpdateLabelMapTransferTexture(mtlDevice, mtlQueue, vol);
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
    uniforms.ScalarMinHalf = FloatToHalf(static_cast<float>(this->ScalarRange[0] / normFactor));
    uniforms.ScalarMaxHalf = FloatToHalf(static_cast<float>(
      (this->ScalarRange[1] > this->ScalarRange[0]
         ? this->ScalarRange[1]
         : this->ScalarRange[0] + 1.0) /
      normFactor));
  }

  uniforms.UseJittering = this->GetUseJittering() ? 1.0f : 0.0f;

  // Gradient-based shading uniforms
  {
    vtkVolumeProperty* property = vol->GetProperty();
    bool shadeOn = property && property->GetShade();
    bool hasGradOp = property && property->HasGradientOpacity();

    uniforms.UseGradientShading = shadeOn ? 1.0f : 0.0f;
    uniforms.UseGradientOpacity = (shadeOn && hasGradOp) ? 1.0f : 0.0f;
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

    // Gradient opacity normalization range
    double scalarRange = this->ScalarRange[1] - this->ScalarRange[0];
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

  // Volume light uniforms for multi-light shading
  VolumeLightUniforms lightUniforms = {};
  {
    double bs[3] = {
      vb.Size[0], vb.Size[1], vb.Size[2]
    };
    this->BuildVolumeLightUniforms(ren, vol, invModelMatrix, this->ModelBounds, bs, lightUniforms);
  }

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

  // 2D transfer function mode — sample the primary scalar against the Y-axis
  // scalar array in the 2D lookup image. yNorm = yRaw * scale + bias.
  bool tf2dActive = (this->Transfer2DEnabled && this->Transfer2DTexture &&
    this->Transfer2DYAxisTexture);
  uniforms.UseTransfer2D = tf2dActive ? 1.0f : 0.0f;
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
  if (uniforms.UseLinearVolumeInterpolation > 0.5f)
    featureMask |= VolumeFeature_LinearInterpolation;

  // Image-space downsampling requires offscreen rendering at reduced resolution.
  // Partitioned volumes no longer force offscreen rendering because grid traversal
  // composites correctly in a single pass.
  const float imageSampleDist = this->ImageSampleDistance;
  const bool useImageSampling = (imageSampleDist != 1.0f);

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
  {
    simd_float4x4 vpMat;
    memcpy(&vpMat, uniforms.ViewProjectionMatrix, sizeof(vpMat));
    float det = simd::determinant(vpMat);
    if (fabs(det) > 1e-10f)
    {
      simd_float4x4 invVP = simd::inverse(vpMat);
      memcpy(uniforms.InverseViewProjection, &invVP, sizeof(invVP));
    }
    else
    {
      memset(uniforms.InverseViewProjection, 0, sizeof(uniforms.InverseViewProjection));
    }
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
      MTLPixelFormatRGBA16Float, MTLPixelFormatInvalid, 1, featureMask);
    if (!rttPso) { [rttEnc endEncoding]; return; }

    [rttEnc setFragmentBytes:&lightUniforms length:sizeof(lightUniforms) atIndex:4];

    this->BindEncoderResources(rttEnc, uniformBuf, rttPso, false);

    // Draw the proxy geometry (clipped convex hull of the volume box) with the
    // RenderToImage fragment shader, mirroring the standard on-screen path.
    this->DrawBlocks(rttEnc, uniformBuf, ren, vol, &uniforms, invModelMatrix);
    [rttEnc endEncoding];
  }
  else if (useImageSampling)
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
          &uniforms, invModelMatrix, false);
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
        &uniforms, invModelMatrix, true);
    }
    else
    {
      void* directPso = this->GetOrCreateVolumePipeline(mtlDevice,
        static_cast<uint32_t>(VolumePipelineType::DirectScreen),
        MTLPixelFormatBGRA8Unorm, MTLPixelFormatDepth32Float,
        static_cast<uint32_t>(sampleCount), featureMask);
      this->BindEncoderResources(encoder, uniformBuf, directPso, true);

      // Draw volume — handle partitioned (multi-block) and single-block cases
      this->DrawBlocks(encoder, uniformBuf, ren, vol, &uniforms, invModelMatrix);
    }
  }

  // Signal the semaphore when the GPU finishes this frame's command buffer.
  // Retain the semaphore under MRC to prevent use-after-free if the mapper
  // is destroyed before the handler fires.
  dispatch_semaphore_t sem = (__bridge dispatch_semaphore_t)this->FrameSemaphore;
#if !__has_feature(objc_arc)
  dispatch_retain(sem);
#endif
  [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
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
