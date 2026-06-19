// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause

struct SceneTransform {
  viewport: vec4<f32>,
  view: mat4x4<f32>,
  projection: mat4x4<f32>,
  normal: mat3x3<f32>,
  inverted_projection: mat4x4<f32>,
  flags: u32
}

struct VolumeMapperUniforms {
  worldToVolume: mat4x4<f32>,
  volumeToWorld: mat4x4<f32>,
  volumeBoundsMin: vec4<f32>,
  volumeBoundsMax: vec4<f32>,
  cameraVolumePos: vec4<f32>,
  sampleDistance: f32,
  scalarMin: f32,      // raw scalar data minimum (e.g. ~37 for RTAnalytic)
  scalarMax: f32,      // raw scalar data maximum (e.g. ~276 for RTAnalytic)
  padding: f32,
}


@group(0) @binding(0) var<uniform> sceneTransform: SceneTransform;

@group(1) @binding(0) var<uniform> volumeUniforms: VolumeMapperUniforms;
// R32Float is 'UnfilterableFloat' in WebGPU — use texture_3d<f32> with textureLoad(),
// not textureSample(), to avoid requiring the float32-filterable feature.
@group(1) @binding(1) var volumeTexture: texture_3d<f32>;
// Non-filtering sampler used for the transfer-function lookup (RGBA8Unorm, filterable).
@group(1) @binding(2) var transferFunctionTexture: texture_2d<f32>;
@group(1) @binding(3) var transferFunctionSampler: sampler;

struct VertexInput {
  @location(0) position: vec3<f32>,
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) localPos: vec3<f32>,
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  // position is in local space (volume coordinates)
  output.position = sceneTransform.projection * sceneTransform.view * volumeUniforms.volumeToWorld * vec4<f32>(input.position, 1.0);
  
  // localPos maps the position to a [0, 1] range inside the volume bounds
  let boundsSize = volumeUniforms.volumeBoundsMax.xyz - volumeUniforms.volumeBoundsMin.xyz;
  output.localPos = (input.position - volumeUniforms.volumeBoundsMin.xyz) / boundsSize;
  
  return output;
}

struct FragmentInput {
  @location(0) localPos: vec3<f32>,
}

struct FragmentOutput {
  // Color attachment 0: main rendered color
  @location(0) color: vec4<f32>,
  // Color attachment 1: hardware selector IDs (RGBA32Uint).
  // Volume rendering does not participate in picking; write zero to suppress
  // the "attachment state mismatch" WebGPU validation error.
  @location(1) selectorId: vec4<u32>,
}

@fragment
fn fragmentMain(input: FragmentInput) -> FragmentOutput {
  var output: FragmentOutput;
  output.selectorId = vec4<u32>(0u, 0u, 0u, 0u);

  // Dimensions of the volume texture (needed for textureLoad integer coords)
  let dims = vec3<i32>(textureDimensions(volumeTexture, 0));

  // Ray casting starting point (in volume/local texture coordinates [0,1]^3)
  let startPoint = input.localPos;
  
  // Ray direction in volume space (from camera to fragment)
  let cameraPos = volumeUniforms.cameraVolumePos.xyz;
  let rayDir = normalize(startPoint - cameraPos);
  
  // Ray marching
  var currentPoint = startPoint;
  var accumulatedColor = vec3<f32>(0.0);
  var accumulatedOpacity = 0.0;
  
  let stepSize = volumeUniforms.sampleDistance;
  let maxSteps = 1000;
  
  for (var i = 0; i < maxSteps; i = i + 1) {
    // Check if we are outside the volume texture bounds [0, 1]^3
    if (any(currentPoint < vec3<f32>(0.0)) || any(currentPoint > vec3<f32>(1.0))) {
      break;
    }
    
    // Convert normalized [0,1] coordinate to integer texel coordinates and
    // sample the scalar value without filtering (avoids float32-filterable requirement).
    let texCoord = vec3<i32>(clamp(
      vec3<f32>(currentPoint) * vec3<f32>(dims),
      vec3<f32>(0.0),
      vec3<f32>(dims - vec3<i32>(1))
    ));
    let rawScalar = textureLoad(volumeTexture, texCoord, 0).r;

    // Normalise raw voxel scalar into [0,1] using the actual data range so
    // that the transfer-function texture is indexed correctly regardless of
    // whether the volume stores unsigned bytes (0–255) or floats (e.g. 37–276).
    let scalarNorm = clamp(
      (rawScalar - volumeUniforms.scalarMin) /
      (volumeUniforms.scalarMax - volumeUniforms.scalarMin),
      0.0, 1.0);
    
    // Map through the pre-baked RGBA transfer function texture.
    let colorOpacity = textureSampleLevel(transferFunctionTexture, transferFunctionSampler, vec2<f32>(scalarNorm, 0.5), 0.0);
    
    // Front-to-back compositing
    let sampleOpacity = colorOpacity.a;
    let sampleColor = colorOpacity.rgb;
    
    accumulatedColor = accumulatedColor + (1.0 - accumulatedOpacity) * sampleColor * sampleOpacity;
    accumulatedOpacity = accumulatedOpacity + (1.0 - accumulatedOpacity) * sampleOpacity;
    
    if (accumulatedOpacity >= 0.95) {
      accumulatedOpacity = 1.0;
      break;
    }
    
    // Step along the ray
    currentPoint = currentPoint + rayDir * stepSize;
  }
  
  output.color = vec4<f32>(accumulatedColor, accumulatedOpacity);
  return output;
}
