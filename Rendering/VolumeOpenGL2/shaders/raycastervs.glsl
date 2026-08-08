//VTK::System::Dec

// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

//////////////////////////////////////////////////////////////////////////////
///
/// Uniforms, attributes, and globals
///
//////////////////////////////////////////////////////////////////////////////
//VTK::CustomUniforms::Dec

//VTK::Base::Dec

//VTK::Termination::Dec

//VTK::Cropping::Dec

//VTK::Shading::Dec

//////////////////////////////////////////////////////////////////////////////
///
/// Inputs
///
//////////////////////////////////////////////////////////////////////////////
in vec3 in_vertexPos;

//////////////////////////////////////////////////////////////////////////////
///
/// Outputs
///
//////////////////////////////////////////////////////////////////////////////
/// 3D texture coordinates for texture lookup in the fragment shader
out vec3 ip_textureCoords;
out vec3 ip_vertexPos;
/// Debug: exact clip-space position (P*V*M*v) for GL/Metal byte-comparison
out vec4 ip_debugClip;
/// Debug: flat (exact per-vertex) clip-space position for the tiny-triangle dump
flat out vec4 ip_debugClipFlat;
/// Debug: vertex index (flat), for aligning the per-vertex clip dump with Metal's vertex logs
flat out int ip_vid;
/// Debug: per-vertex clip dump via tiny triangles (mode 1 = active)
uniform int in_debugVertexMode;
uniform int in_debugVertexCount;

void main()
{
  /// Get clipspace position
  //VTK::ComputeClipPos::Impl

  /// Compute texture coordinates
  //VTK::ComputeTextureCoords::Impl

  /// Copy incoming vertex position for the fragment shader
  ip_vertexPos = in_vertexPos;
}
