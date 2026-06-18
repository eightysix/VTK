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
  padding1: f32,
  padding2: f32,
  padding3: f32,
}

@group(0) @binding(0) var<uniform> sceneTransform: SceneTransform;

@group(1) @binding(0) var<uniform> volumeUniforms: VolumeMapperUniforms;
@group(1) @binding(1) var volumeTexture: texture_3d<f32>;
@group(1) @binding(2) var volumeSampler: sampler;
@group(1) @binding(3) var transferFunctionTexture: texture_2d<f32>;
@group(1) @binding(4) var transferFunctionSampler: sampler;

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
  @location(0) color: vec4<f32>,
}

@fragment
fn fragmentMain(input: FragmentInput) -> FragmentOutput {
  var output: FragmentOutput;
  
  // Ray casting starting point (in volume/local texture coordinates)
  let startPoint = input.localPos;
  
  // Ray direction in volume space (from camera to fragment)
  let cameraPos = volumeUniforms.cameraVolumePos.xyz;
  let rayDir = normalize(startPoint - cameraPos);
  
  // Ray marching setup
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
    
    // Sample scalar value from 3D texture
    let scalarVal = textureSampleLevel(volumeTexture, volumeSampler, currentPoint, 0.0).r;
    
    // Map scalar value through transfer function 1D/2D texture (using y = 0.5)
    let colorOpacity = textureSampleLevel(transferFunctionTexture, transferFunctionSampler, vec2<f32>(scalarVal, 0.5), 0.0);
    
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
